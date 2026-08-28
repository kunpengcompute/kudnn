/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
 *
 * KuDNN is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *        http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */

#include <memory>
#include <vector>
#include <algorithm>

#include <arm_sve.h>

#include "kudnn.hpp"

#include "cg_codegen.hpp"

#include "kudnn_jit.hpp"
#include "kernels/reorder/kudnn_reorder_jit.hpp"
#include "kernels/reorder/kudnn_reorder_helpers.hpp"
#include "kernels/reorder/kudnn_reorder_support_functions.hpp"
#include "kudnn_wrapper_helpers.hpp"
#include "threading_impl.hpp"
#include "kernels/reorder/sve/bf16any32/kudnn_fast_jit_impl_bfp16any32_bfp16any32.hpp"

namespace KuDNN {
namespace Detail {

template <typename SrcType, typename DstType>
FastReorderKernelCodeGen<SrcType, DstType>::FastReorderKernelCodeGen(
    SizeType inHeightSize, SizeType inWidthSize, bool hasAlpha)
{
    Preamble();
    if (hasAlpha) {
        Fmov(tmp, alpha);
    }

    Mov(zero, 0);
    Mov(blockSizeReg, blockSize);
    Mov(heightSize, inHeightSize);
    Mov(widthSize, inWidthSize);

    Mov(widthCmp, widthSize);
    Sub(widthCmp, widthCmp, blockSizeReg);

    Mov(heightCmp, heightSize);
    Sub(heightCmp, heightCmp, blockSizeReg);

    Mov(ih, 0);
    if (blockSize < inHeightSize) {
        ARMCG::Label ihLoopHead;
        L(ihLoopHead);
        {
            GenerateInHeightLoopBody(blockSize, inWidthSize, hasAlpha);
            Add(ih, ih, blockSizeReg);
            Cmp(ih, heightCmp);
        }
        B(ARMCG::Cond::LO, ihLoopHead);
    }
    GenerateInHeightLoopBody(inHeightSize % blockSize == 0 ?
                             blockSize : (inHeightSize % blockSize),
                             inWidthSize, hasAlpha);
    Postamble();
    Ret();
};

template <typename SrcType, typename DstType>
void FastReorderKernelCodeGen<SrcType, DstType>::GenerateInHeightLoopBody(
    SizeType currBlockHeightVC, SizeType inWidthSizeVC, bool hasAlpha)
{
    Mov(iw, 0);
    if (blockSize < inWidthSizeVC) {
        ARMCG::Label iwLoopHead;
        L(iwLoopHead);
        {
            GenerateInWidthLoopBody(currBlockHeightVC, blockSize, hasAlpha);
            Add(iw, iw, blockSizeReg);
            Cmp(iw, widthCmp);
        }
        B(ARMCG::Cond::LO, iwLoopHead);
    }
    GenerateInWidthLoopBody(currBlockHeightVC, (inWidthSizeVC % blockSize == 0
                            ? blockSize : (inWidthSizeVC % blockSize)), hasAlpha);
}

template <typename SrcType, typename DstType>
void FastReorderKernelCodeGen<SrcType, DstType>::GenerateInWidthLoopBody(
    SizeType currBlockHeight, SizeType currBlockWidth, bool hasAlpha)
{
    SizeType svlen = Vector::Length<SrcType>();
    SizeType inHeightResid = ((currBlockHeight % svlen) == 0) ? svlen : (currBlockHeight % svlen);

    ARMCG::Label maxIhIsMin;
    Mov(maxIh, ih);
    Add(maxIh, maxIh, blockSizeReg);
    Cmp(maxIh, heightSize);
    B(ARMCG::Cond::LE, maxIhIsMin);
    Mov(maxIh, heightSize);
    L(maxIhIsMin);
    Sub(maxIh, maxIh, svlen);

    Mov(localIh, ih);
    if (currBlockHeight > svlen) {
        ARMCG::Label localIhLoopHead;
        L(localIhLoopHead);
        {
            GenerateLocalIhLoopBody(svlen, currBlockWidth, hasAlpha);
            Add(localIh, localIh, svlen);
            Cmp(localIh, maxIh);
        }
        B(ARMCG::Cond::LO, localIhLoopHead);
    }
    GenerateLocalIhLoopBody(inHeightResid, currBlockWidth, hasAlpha);
}

template <typename SrcType, typename DstType>
void FastReorderKernelCodeGen<SrcType, DstType>::GenerateLocalIhLoopBody(
    SizeType inHeight, SizeType currBlockWidth, bool hasAlpha)
{
    SizeType svlen = Vector::Length<SrcType>();
    SizeType inWidthResid = ((currBlockWidth) % svlen == 0) ? svlen : (currBlockWidth) % svlen;

    ARMCG::Label maxIwIsMin;
    Mov(maxIw, iw);
    Add(maxIw, maxIw, blockSizeReg);
    Cmp(maxIw, widthSize);
    B(ARMCG::Cond::LE, maxIwIsMin);
    Mov(maxIw, widthSize);
    L(maxIwIsMin);
    Sub(maxIw, maxIw, svlen);

    Mov(localIw, iw);
    if (currBlockWidth > svlen) {
        ARMCG::Label localIwLoopHead;
        L(localIwLoopHead);
        {
            GenerateLocalIwLoopBody(inHeight, svlen, hasAlpha);
            Add(localIw, localIw, svlen);
            Cmp(localIw, maxIw);
        }
        B(ARMCG::Cond::LO, localIwLoopHead);
    }
    GenerateLocalIwLoopBody(inHeight, inWidthResid, hasAlpha);
}

template <typename SrcType, typename DstType>
void FastReorderKernelCodeGen<SrcType, DstType>::GenerateLocalIwLoopBody(
    SizeType inHeight, SizeType inWidth, bool hasAlpha)
{
    SizeType svlen = Vector::Length<SrcType>();
    SizeType logSvlen = Math::Log2(svlen);
    SizeType regOffset = svlen / NUM_2;
    bool needToConvert = false;

    Mul(localSrcStride, localIh, strideIn);
    Mul(localDstStride, localIw, strideOut);

    Add(localSrcStride, localSrcStride, localIw, ARMCG::ShMod::LSL, Math::Log2(sizeof(SrcType)));
    Add(localDstStride, localDstStride, localIh, ARMCG::ShMod::LSL, Math::Log2(sizeof(SrcType)));

    Mov(loadOffset, B_ptr);
    Mov(storeOffset, B_buffer);
    Add(loadOffset, loadOffset, localSrcStride);
    Add(storeOffset, storeOffset, localDstStride);

    Mov(height, inHeight);
    Mov(width, inWidth);
    Whilelt(KUDNN_JIT_Generator::GetPReg<SrcType>(0), zero, width);
    LoaderFastReorderCore(inHeight);
    ZipperFastReorderCore(svlen, logSvlen, regOffset);
    Whilelt(KUDNN_JIT_Generator::GetPReg<SrcType>(1), zero, height);

    if (hasAlpha) {
        MulFastReorder(svlen, logSvlen, needToConvert);
    }

    StorerFastReorderCore(svlen, logSvlen, inWidth, needToConvert, hasAlpha);
}

template <typename SrcType, typename DstType>
void FastReorderKernelCodeGen<SrcType, DstType>::LoaderFastReorderCore(
    SizeType inHeight)
{
    for (SizeType i = 0; i < inHeight; ++i) {
        KUDNN_JIT_Generator::Ld1(KUDNN_JIT_Generator::GetZReg<SrcType>(i),
            KUDNN_JIT_Generator::GetPReg<SrcType>(0), Ptr(loadOffset));
        Add(loadOffset, loadOffset, strideIn);
    }
}

template <typename SrcType, typename DstType>
void FastReorderKernelCodeGen<SrcType, DstType>::ZipperFastReorderCore(SizeType svlen, SizeType logSvlen,
                                                                       SizeType regOffset)
{
    for (SizeType i = 0; i < logSvlen; ++i) {
        for (SizeType j = 0; j < svlen; j+=NUM_2) {
            Zip1(KUDNN_JIT_Generator::GetZReg<SrcType>(GetRegIndexInverse(j, i, svlen)),
                 KUDNN_JIT_Generator::GetZReg<SrcType>(GetRegIndex(j/NUM_2, i, svlen)),
                 KUDNN_JIT_Generator::GetZReg<SrcType>(GetRegIndex(j/NUM_2, i, svlen) + regOffset));
            Zip2(KUDNN_JIT_Generator::GetZReg<SrcType>(GetRegIndexInverse(j + NUM_1, i, svlen)),
                 KUDNN_JIT_Generator::GetZReg<SrcType>(GetRegIndex(j/NUM_2, i, svlen)),
                 KUDNN_JIT_Generator::GetZReg<SrcType>(GetRegIndex(j/NUM_2, i, svlen) + regOffset));
        }
    }
}

template <typename SrcType, typename DstType>
void FastReorderKernelCodeGen<SrcType, DstType>::MulFastReorder(SizeType svlen, SizeType logSvlen, bool &needToConvert)
{
    Fmov(alphaTmpOffset, tmp);
    if constexpr (std::is_same_v<SrcType, float>) {
        if constexpr (std::is_same_v<DstType, std::int32_t>) {
            Mov(intMaxTmp, REORDER_S32_TO_F32_SAT_CONSTANT_HEX);
            Mov(ARMCG::ZRegS(GetRegIndexInverse(NUM_1, logSvlen, svlen)), intMaxTmp);
        }
        Mov(ARMCG::ZRegS(GetRegIndexInverse(0, logSvlen, svlen)), alphaTmpOffset);
        for (SizeType i = 0; i < svlen; ++i) {
            Fmul(KUDNN_JIT_Generator::GetZReg<SrcType>(GetRegIndex(i, logSvlen, svlen)),
                KUDNN_JIT_Generator::GetPReg<SrcType>(1),
                KUDNN_JIT_Generator::GetZReg<SrcType>(GetRegIndexInverse(0, logSvlen, svlen)));
            if constexpr (std::is_same_v<DstType, std::int32_t>) {
                Fmin(ARMCG::ZRegS(GetRegIndex(i, logSvlen, svlen)), ARMCG::PRegS(1),
                    ARMCG::ZRegS(GetRegIndexInverse(NUM_1, logSvlen, svlen)));
            }
        }
    } else if constexpr (std::is_same_v<SrcType, std::int32_t>) {
        if constexpr (std::is_same_v<DstType, std::int32_t>) {
            Mov(intMaxTmp, REORDER_S32_TO_F32_SAT_CONSTANT_HEX);
            Mov(ARMCG::ZRegS(GetRegIndexInverse(NUM_1, logSvlen, svlen)), intMaxTmp);
        }
        Mov(ARMCG::ZRegS(GetRegIndexInverse(NUM_0, logSvlen, svlen)), alphaTmpOffset);
        for (uint32_t i = 0; i < svlen; ++i) {
            Scvtf(ARMCG::ZRegS(GetRegIndex(i, logSvlen, svlen)), ARMCG::PRegS(1),
                ARMCG::ZRegS(GetRegIndex(i, logSvlen, svlen)));
            Fmul(ARMCG::ZRegS(GetRegIndex(i, logSvlen, svlen)), ARMCG::PRegS(1),
                ARMCG::ZRegS(GetRegIndexInverse(NUM_0, logSvlen, svlen)));
            if constexpr (std::is_same_v<DstType, std::int32_t>) {
                Fmin(ARMCG::ZRegS(GetRegIndex(i, logSvlen, svlen)), ARMCG::PRegS(1),
                    ARMCG::ZRegS(GetRegIndexInverse(NUM_1, logSvlen, svlen)));
            }
        }
        if constexpr (std::is_same_v<DstType, std::int32_t>) {needToConvert = true;}
    } else if constexpr (std::is_same_v<SrcType, __fp16>) {
        Ptrue(ARMCG::PRegS(NUM_2));
        Mov(ARMCG::ZRegS(GetRegIndexInverse(NUM_0, logSvlen, svlen)), alphaTmpOffset);
        Dup(ARMCG::ZRegH(GetRegIndexInverse(NUM_1, logSvlen, svlen)), NUM_0);
        for (SizeType i = 0; i < svlen; ++i) {
            Zip1(ARMCG::ZRegH(GetRegIndexInverse(NUM_2, logSvlen, svlen)),
                ARMCG::ZRegH(GetRegIndex(i, logSvlen, svlen)),
                ARMCG::ZRegH(GetRegIndexInverse(NUM_1, logSvlen, svlen)));
            Zip2(ARMCG::ZRegH(GetRegIndexInverse(NUM_3, logSvlen, svlen)),
                ARMCG::ZRegH(GetRegIndex(i, logSvlen, svlen)),
                ARMCG::ZRegH(GetRegIndexInverse(NUM_1, logSvlen, svlen)));
            Fcvt(ARMCG::ZRegS(GetRegIndexInverse(NUM_2, logSvlen, svlen)), ARMCG::PRegS(NUM_2),
                ARMCG::ZRegH(GetRegIndexInverse(NUM_2, logSvlen, svlen)));
            Fcvt(ARMCG::ZRegS(GetRegIndexInverse(NUM_3, logSvlen, svlen)), ARMCG::PRegS(NUM_2),
                ARMCG::ZRegH(GetRegIndexInverse(NUM_3, logSvlen, svlen)));
            Fmul(ARMCG::ZRegS(GetRegIndexInverse(NUM_2, logSvlen, svlen)),
                ARMCG::ZRegS(GetRegIndexInverse(NUM_2, logSvlen, svlen)),
                ARMCG::ZRegS(GetRegIndexInverse(NUM_0, logSvlen, svlen)));
            Fmul(ARMCG::ZRegS(GetRegIndexInverse(NUM_3, logSvlen, svlen)),
                ARMCG::ZRegS(GetRegIndexInverse(NUM_3, logSvlen, svlen)),
                ARMCG::ZRegS(GetRegIndexInverse(NUM_0, logSvlen, svlen)));
            Fcvt(ARMCG::ZRegH(GetRegIndexInverse(NUM_2, logSvlen, svlen)), ARMCG::PRegS(NUM_2),
                ARMCG::ZRegS(GetRegIndexInverse(NUM_2, logSvlen, svlen)));
            Fcvt(ARMCG::ZRegH(GetRegIndexInverse(NUM_3, logSvlen, svlen)), ARMCG::PRegS(NUM_2),
                ARMCG::ZRegS(GetRegIndexInverse(NUM_3, logSvlen, svlen)));
            Uzp1(ARMCG::ZRegH(GetRegIndex(i, logSvlen, svlen)),
                ARMCG::ZRegH(GetRegIndexInverse(NUM_2, logSvlen, svlen)),
                ARMCG::ZRegH(GetRegIndexInverse(NUM_3, logSvlen, svlen)));
        }
    } else if constexpr (std::is_same_v<SrcType, __bf16>) {
        Whilelt(KUDNN_JIT_Generator::GetPReg<float32_t>(NUM_2), zero, height);
        Mov(ARMCG::ZRegS(GetRegIndexInverse(NUM_0, logSvlen, svlen)), alphaTmpOffset);
        Dup(ARMCG::ZRegH(GetRegIndexInverse(NUM_1, logSvlen, svlen)), NUM_0);
        for (SizeType i = 0; i < svlen; ++i) {
            Zip1(ARMCG::ZRegH(GetRegIndexInverse(NUM_2, logSvlen, svlen)),
                ARMCG::ZRegH(GetRegIndexInverse(NUM_1, logSvlen, svlen)),
                ARMCG::ZRegH(GetRegIndex(i, logSvlen, svlen)));
            Zip2(ARMCG::ZRegH(GetRegIndexInverse(NUM_3, logSvlen, svlen)),
                ARMCG::ZRegH(GetRegIndexInverse(NUM_1, logSvlen, svlen)),
                ARMCG::ZRegH(GetRegIndex(i, logSvlen, svlen)));
            Fmul(ARMCG::ZRegS(GetRegIndexInverse(NUM_2, logSvlen, svlen)),
                ARMCG::ZRegS(GetRegIndexInverse(NUM_2, logSvlen, svlen)),
                ARMCG::ZRegS(GetRegIndexInverse(NUM_0, logSvlen, svlen)));
            Fmul(ARMCG::ZRegS(GetRegIndexInverse(NUM_3, logSvlen, svlen)),
                ARMCG::ZRegS(GetRegIndexInverse(NUM_3, logSvlen, svlen)),
                ARMCG::ZRegS(GetRegIndexInverse(NUM_0, logSvlen, svlen)));
            Bfcvtnt(ARMCG::ZRegH(GetRegIndexInverse(NUM_2, logSvlen, svlen)), ARMCG::PRegS(NUM_2),
                ARMCG::ZRegS(GetRegIndexInverse(NUM_2, logSvlen, svlen)));
            Bfcvtnt(ARMCG::ZRegH(GetRegIndexInverse(NUM_3, logSvlen, svlen)), ARMCG::PRegS(NUM_2),
                ARMCG::ZRegS(GetRegIndexInverse(NUM_3, logSvlen, svlen)));
            Uzp2(ARMCG::ZRegH(GetRegIndex(i, logSvlen, svlen)),
                ARMCG::ZRegH(GetRegIndexInverse(NUM_2, logSvlen, svlen)),
                ARMCG::ZRegH(GetRegIndexInverse(NUM_3, logSvlen, svlen)));
        }
    }
}

template <typename SrcType, typename DstType>
void FastReorderKernelCodeGen<SrcType, DstType>::StorerFastReorderCore(SizeType svlen, SizeType logSvlen,
    SizeType inWidth, bool needToConvert, bool hasAlpha)
{
    for (SizeType i = 0; i < inWidth; ++i) {
        if ((!std::is_same_v<SrcType, DstType> && std::is_same_v<SrcType, float>) || needToConvert) {
            Frinti(ARMCG::ZRegS(GetRegIndex(i, logSvlen, svlen)), ARMCG::PRegS(1),
                   ARMCG::ZRegS(GetRegIndex(i, logSvlen, svlen)));
            Fcvtzs(ARMCG::ZRegS(GetRegIndex(i, logSvlen, svlen)), ARMCG::PRegS(1),
                   ARMCG::ZRegS(GetRegIndex(i, logSvlen, svlen)));
        } else if ((!std::is_same_v<SrcType, DstType> && std::is_same_v<SrcType, std::int32_t>) && !hasAlpha) {
            Scvtf(ARMCG::ZRegS(GetRegIndex(i, logSvlen, svlen)), ARMCG::PRegS(1),
                  ARMCG::ZRegS(GetRegIndex(i, logSvlen, svlen)));
        }
        KUDNN_JIT_Generator::St1(KUDNN_JIT_Generator::GetZReg<SrcType>(GetRegIndex(i, logSvlen, svlen)),
            KUDNN_JIT_Generator::GetPReg<SrcType>(1), Ptr(storeOffset));
        Add(storeOffset, storeOffset, strideOut);
    }
}

template <typename SrcType, typename DstType>
FastReorderImplGenerator<FastReorderKernelCodeGen<SrcType, DstType>>::FastImpl::FastImpl(
    const ReorderInfo &task, SizeType nThreads) : ReorderImplBase(task, nThreads)
{
    VERBOSE_PRINT;

    Shape srcStridesVC = Service::GetShapeAccordingToLayout(task.srcInfo.GetStrides(), task.srcInfo.GetLayout());
    Shape dstStridesVC = Service::GetShapeAccordingToLayout(task.dstInfo.GetStrides(), task.srcInfo.GetLayout());
    reorderedDims = Service::BroadcastTo5D(Service::GetShapeAccordingToLayout(task.dstInfo.GetDims(),
        task.srcInfo.GetLayout()));

    reorderedStridesSrc = Service::BroadcastTo5D(srcStridesVC, false);
    reorderedStridesDst = Service::BroadcastTo5D(dstStridesVC, false);

    for (SizeType iVC = reorderedDims.GetNumDims() - 1; iVC > 0; --iVC) {
        ReorderHelpers::ReduceDimsAndStrides(reorderedDims, reorderedStridesSrc, reorderedStridesDst);
    }
    for (SizeType iVC = reorderedDims.GetNumDims() - 1; iVC > 0; --iVC) {
        ReorderHelpers::CompressDimsAndStrides(reorderedDims, reorderedStridesSrc, reorderedStridesDst);
    }

    SizeType optNumThreads = std::min(nThreads, Threading::ShouldCalculateOptThreads() ? GetOptNumThreads() : nThreads);
    std::vector<std::pair<Shape, Shape>> reorderWork = Threading::SplitDims(reorderedDims, optNumThreads);

    SizeType mFullBlockThread = reorderWork[0].first[IDX_3];
    SizeType mResidBlockThread = reorderedDims[IDX_3] % mFullBlockThread;

    SizeType nFullBlockThread = reorderWork[0].first[IDX_4];
    SizeType nResidBlockThread = reorderedDims[IDX_4] % nFullBlockThread;

    CodeGenWrapper<FastReorderKernelCodeGen<SrcType, DstType>> cgw0;
    cgw0.Initialize(static_cast<SizeType>(mResidBlockThread),
        static_cast<SizeType>(nResidBlockThread), true);
    CodeGenWrapper<FastReorderKernelCodeGen<SrcType, DstType>> cgw1;
    cgw1.Initialize(static_cast<SizeType>(mResidBlockThread),
        static_cast<SizeType>(nFullBlockThread), true);
    CodeGenWrapper<FastReorderKernelCodeGen<SrcType, DstType>> cgw2;
    cgw2.Initialize(static_cast<SizeType>(mFullBlockThread),
        static_cast<SizeType>(nResidBlockThread), true);
    CodeGenWrapper<FastReorderKernelCodeGen<SrcType, DstType>> cgw3;
    cgw3.Initialize(static_cast<SizeType>(mFullBlockThread),
        static_cast<SizeType>(nFullBlockThread), true);

    CodeGenWrapper<FastReorderKernelCodeGen<SrcType, DstType>> cgw4;
    cgw4.Initialize(static_cast<SizeType>(mResidBlockThread),
        static_cast<SizeType>(nResidBlockThread), false);
    CodeGenWrapper<FastReorderKernelCodeGen<SrcType, DstType>> cgw5;
    cgw5.Initialize(static_cast<SizeType>(mResidBlockThread),
        static_cast<SizeType>(nFullBlockThread), false);
    CodeGenWrapper<FastReorderKernelCodeGen<SrcType, DstType>> cgw6;
    cgw6.Initialize(static_cast<SizeType>(mFullBlockThread),
        static_cast<SizeType>(nResidBlockThread), false);
    CodeGenWrapper<FastReorderKernelCodeGen<SrcType, DstType>> cgw7;
    cgw7.Initialize(static_cast<SizeType>(mFullBlockThread),
        static_cast<SizeType>(nFullBlockThread), false);
   
    reorderCodegenKernels = {std::move(cgw0), std::move(cgw1), std::move(cgw2), std::move(cgw3),
                             std::move(cgw4), std::move(cgw5), std::move(cgw6), std::move(cgw7)};

    perThreadParams.reserve(reorderWork.size());
    reorderWorkSize = reorderWork.size();

    for (SizeType iVC = 0; iVC < reorderWork.size(); ++iVC) {
        Shape start = Service::BroadcastTo5D(reorderWork[iVC].second, false);
        Shape end = start + Service::BroadcastTo5D(reorderWork[iVC].first, true);
        SizeType m_idx = ((end[IDX_3] - start[IDX_3]) < mFullBlockThread) ? 0 : 2;
        SizeType n_idx = ((end[IDX_4] - start[IDX_4]) < nFullBlockThread) ? 0 : 1;
        perThreadParams.push_back({start, end, m_idx + n_idx});
    }
}

template <typename SrcType, typename DstType>
void FastReorderImplGenerator<FastReorderKernelCodeGen<SrcType, DstType>>::FastImpl::RunSequential(
    IntType threadID, const void* src, void* dst, float alpha) const noexcept(false)
{
    VERBOSE_PRINT;
    const SrcType* typedSrc = static_cast<const SrcType*>(src);
    DstType* typedDst = static_cast<DstType*>(dst);
    SizeType idxAlpha = (alpha == 1.0f) ? 1 : 0;

    IntType srcKernelStrideVC = perThreadParams[threadID].threadStart[IDX_3] * reorderedStridesSrc[IDX_3] +
                              perThreadParams[threadID].threadStart[IDX_4] * reorderedStridesSrc[IDX_4];
    IntType dstKernelStrideVC = perThreadParams[threadID].threadStart[IDX_3] * reorderedStridesDst[IDX_3] +
                              perThreadParams[threadID].threadStart[IDX_4] * reorderedStridesDst[IDX_4];

    for (SizeType i0 = perThreadParams[threadID].threadStart[IDX_0];
        i0 < perThreadParams[threadID].threadEnd[IDX_0]; ++i0) {
        for (SizeType i1 = perThreadParams[threadID].threadStart[IDX_1];
            i1 < perThreadParams[threadID].threadEnd[IDX_1]; ++i1) {
            for (SizeType i2 = perThreadParams[threadID].threadStart[IDX_2];
                i2 < perThreadParams[threadID].threadEnd[IDX_2]; ++i2) {
                IntType srcBatchStrideVC = i0 * reorderedStridesSrc[NUM_0] +
                                         i1 * reorderedStridesSrc[NUM_1] +
                                         i2 * reorderedStridesSrc[NUM_2];
                IntType dstBatchStrideVC = i0 * reorderedStridesDst[NUM_0] +
                                         i1 * reorderedStridesDst[NUM_1] +
                                         i2 * reorderedStridesDst[NUM_2];
                auto kernel = reorderCodegenKernels[idxAlpha * NUM_4 + perThreadParams[threadID].kernelIdx].template
                GetCode<void(*)(const SrcType*, DstType*, SizeType, SizeType, float)>();
                kernel(
                    typedSrc + srcBatchStrideVC + srcKernelStrideVC,
                    typedDst + dstBatchStrideVC + dstKernelStrideVC,
                    reorderedStridesSrc[NUM_3] * sizeof(SrcType),
                    reorderedStridesDst[NUM_4] * sizeof(DstType),
                    alpha);
            }
        }
    }
}

template <typename SrcType, typename DstType>
SizeType FastReorderImplGenerator<FastReorderKernelCodeGen<SrcType, DstType>>::FastImpl::GetOptNumThreads(
    ) const noexcept
{
    SizeType numElements = reorderedDims.GetTotalDimsSize();
#ifdef KUDNNL_920F
    SizeType l2UpperPart = 32758; // 32758 is measured for F32 920F
    if (L1_CACHE_SIZE_BYTES / sizeof(SrcType) > numElements) {
        return NUM_1;
    } else if (numElements < L2_CACHE_SIZE_BYTES / sizeof(SrcType)) {
        if (numElements > l2UpperPart) {
            return NUM_4;
        }
        return NUM_1;
    } else {
        return GetSpecifiedThreadCount<SrcType>(L3_CACHE_SIZE_BYTES / sizeof(SrcType));
    }
#else
    SizeType l1LowerPart = 9000;   // 9000 is measured for F32 920B
    SizeType l2LowerPart = 160000; // 160000 is measured for F32 920B
    if constexpr (sizeof(SrcType) == sizeof(__bf16)) {
        l1LowerPart = 16384;  // 16384 is measured for BF16/F16 920B
        l2LowerPart = 393216; // 393216 is measured for BF16/F16 920B
    }
    if (L1_CACHE_SIZE_BYTES / sizeof(SrcType) > numElements) {
        if (numElements < l1LowerPart) {
            return NUM_1;
        }
        return NUM_4;
    } else if (numElements < L2_CACHE_SIZE_BYTES / sizeof(SrcType)) {
        if (numElements <= l2LowerPart) {
            return NUM_8;
        }
        return NUM_16;
    } else {
        return NUM_40;
    }
#endif
}

template
struct FastReorderKernelCodeGen<float, float>;
template
struct FastReorderKernelCodeGen<std::int32_t, std::int32_t>;
template
struct FastReorderKernelCodeGen<float, std::int32_t>;
template
struct FastReorderKernelCodeGen<std::int32_t, float>;
template
struct FastReorderKernelCodeGen<__fp16, __fp16>;
template
struct FastReorderKernelCodeGen<__bf16, __bf16>;

template
struct FastReorderImplGenerator<FastReorderKernelCodeGen<float, float>>;
template
struct FastReorderImplGenerator<FastReorderKernelCodeGen<std::int32_t, std::int32_t>>;
template
struct FastReorderImplGenerator<FastReorderKernelCodeGen<float, std::int32_t>>;
template
struct FastReorderImplGenerator<FastReorderKernelCodeGen<std::int32_t, float>>;
template
struct FastReorderImplGenerator<FastReorderKernelCodeGen<__fp16, __fp16>>;
template
struct FastReorderImplGenerator<FastReorderKernelCodeGen<__bf16, __bf16>>;

} // Detail
} // KuDNN
