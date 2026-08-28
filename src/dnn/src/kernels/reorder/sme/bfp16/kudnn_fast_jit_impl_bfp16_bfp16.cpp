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
#include <array>
#include <algorithm>

#include "kudnn.hpp"

#include "cg_codegen.hpp"

#include "kudnn_jit.hpp"
#include "kernels/reorder/kudnn_reorder_support_functions.hpp"
#include "kernels/reorder/kudnn_reorder_jit.hpp"
#include "kernels/reorder/kudnn_reorder_helpers.hpp"
#include "kudnn_wrapper_helpers.hpp"
#include "threading_impl.hpp"
#include "kernels/reorder/sme/bfp16/kudnn_fast_jit_impl_bfp16_bfp16.hpp"

namespace KuDNN {
namespace Detail {

template <typename SrcType, typename DstType>
FastReorderKernelCodeGenMC<SrcType, DstType>::FastReorderKernelCodeGenMC(
    SizeType inHeightSize, SizeType inWidthSize, bool hasAlpha)
{
    Preamble();

    if (hasAlpha) {
        Fmov(tmp, alpha);
    }

    Smstart(ARMCG::StreamMode::SMZA);

    Mov(zero, 0);

    Mov(blockSizeReg, blockSize);
    Mov(heightSize, inHeightSize);
    Mov(widthSize, inWidthSize);

    Mov(widthCmp, widthSize);
    Sub(widthCmp, widthCmp, blockSizeReg);

    Mov(heightCmp, heightSize);
    Sub(heightCmp, heightCmp, blockSizeReg);

    Mov(ih, 0);

    // constants to perform continious stores from ZA tiles
    Mov(tmpLDStC0, NUM_0);
    Mov(tmpLDStC1, NUM_8);
    Mov(tmpLDStC2, NUM_16);
    Mov(tmpLDStC3, NUM_24);

    // all 2 tiles are zeroed
    Zero({ARMCG::ZARegH(NUM_0), ARMCG::ZARegH(NUM_1)});

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
    Smstop(ARMCG::StreamMode::SMZA);
    Postamble();
    Ret();
};

template <typename SrcType, typename DstType>
void FastReorderKernelCodeGenMC<SrcType, DstType>::GenerateInHeightLoopBody(
    SizeType currBlockHeightMC, SizeType inWidthSizeMC, bool hasAlpha)
{
    Mov(iw, 0);
    if (blockSize < inWidthSizeMC) {
        ARMCG::Label iwLoopHead;
        L(iwLoopHead);
        {
            GenerateInWidthLoopBody(currBlockHeightMC, blockSize, hasAlpha);
            Add(iw, iw, blockSizeReg);
            Cmp(iw, widthCmp);
        }
        B(ARMCG::Cond::LO, iwLoopHead);
    }
    GenerateInWidthLoopBody(currBlockHeightMC, (inWidthSizeMC % blockSize == 0
                            ? blockSize : (inWidthSizeMC % blockSize)), hasAlpha);
}

template <typename SrcType, typename DstType>
void FastReorderKernelCodeGenMC<SrcType, DstType>::GenerateInWidthLoopBody(
    SizeType currBlockHeight, SizeType currBlockWidth, bool hasAlpha)
{
    SizeType vecLenMC = Vector::Length<SrcType>();
    SizeType inHeightResid = ((currBlockHeight % vecLenMC) == 0) ? vecLenMC : (currBlockHeight % vecLenMC);

    ARMCG::Label maxIhIsMin;
    Mov(maxIh, ih);
    Add(maxIh, maxIh, blockSizeReg);
    Cmp(maxIh, heightSize);
    B(ARMCG::Cond::LE, maxIhIsMin);
    Mov(maxIh, heightSize);
    L(maxIhIsMin);
    Sub(maxIh, maxIh, vecLenMC);

    Mov(localIh, ih);
    if (currBlockHeight > vecLenMC) {
        ARMCG::Label localIhLoopHead;
        L(localIhLoopHead);
        {
            GenerateLocalIhLoopBody(vecLenMC, currBlockWidth, hasAlpha);
            Add(localIh, localIh, vecLenMC);
            Cmp(localIh, maxIh);
        }
        B(ARMCG::Cond::LO, localIhLoopHead);
    }
    GenerateLocalIhLoopBody(inHeightResid, currBlockWidth, hasAlpha);
}

template <typename SrcType, typename DstType>
void FastReorderKernelCodeGenMC<SrcType, DstType>::GenerateLocalIhLoopBody(
    SizeType inHeight, SizeType currBlockWidth, bool hasAlpha)
{
    SizeType vecLenMC = Vector::Length<SrcType>();
    SizeType inWidthResid = ((currBlockWidth) % vecLenMC == 0) ? vecLenMC : (currBlockWidth) % vecLenMC;

    ARMCG::Label maxIwIsMin;
    Mov(maxIw, iw);
    Add(maxIw, maxIw, blockSizeReg);
    Cmp(maxIw, widthSize);
    B(ARMCG::Cond::LE, maxIwIsMin);
    Mov(maxIw, widthSize);
    L(maxIwIsMin);
    Sub(maxIw, maxIw, vecLenMC);

    Mov(localIw, iw);
    if (currBlockWidth > vecLenMC) {
        ARMCG::Label localIwLoopHead;
        L(localIwLoopHead);
        {
            GenerateLocalIwLoopBody(inHeight, vecLenMC, hasAlpha);
            Add(localIw, localIw, vecLenMC);
            Cmp(localIw, maxIw);
        }
        B(ARMCG::Cond::LO, localIwLoopHead);
    }
    GenerateLocalIwLoopBody(inHeight, inWidthResid, hasAlpha);
}

template <typename SrcType, typename DstType>
void FastReorderKernelCodeGenMC<SrcType, DstType>::GenerateLocalIwLoopBody(
    SizeType inHeight, SizeType inWidth, bool hasAlpha)
{
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
    Whilelt(KUDNN_JIT_Generator::GetPReg<SrcType>(1), zero, height);

    LoaderFastReorderCore(inHeight);
    if (hasAlpha) {
        StoreMulFastReorder(inWidth);
    } else {
        StorerFastReorderCore(inWidth);
    }
}

template <typename SrcType, typename DstType>
void FastReorderKernelCodeGenMC<SrcType, DstType>::LoaderFastReorderCore(SizeType inHeight)
{
    for (SizeType i = 0; i < inHeight; ++i) {
        ARMCG::CodeGenerator::Ld1h(ARMCG::ZARegHH(0), ARMCG::WReg(NUM_12 + i / NUM_8), i % NUM_8,
            KUDNN_JIT_Generator::GetPReg<SrcType>(0),
            ARMCG::AdrScSc(loadOffset, zero, ARMCG::ShMod::LSL, static_cast<uint32_t>(1)));
        Add(loadOffset, loadOffset, strideIn);
    }
}

template <typename SrcType, typename DstType>
void FastReorderKernelCodeGenMC<SrcType, DstType>::StorerFastReorderCore(SizeType inWidth)
{
    for (SizeType i = 0; i < inWidth; ++i) {
        ARMCG::CodeGenerator::St1h(ARMCG::ZARegVH(0), ARMCG::WReg(NUM_12 + i / NUM_8), i % NUM_8,
            KUDNN_JIT_Generator::GetPReg<SrcType>(1),
            ARMCG::AdrScSc(storeOffset, zero, ARMCG::ShMod::LSL, static_cast<uint32_t>(1)));
        Add(storeOffset, storeOffset, strideOut);
    }
}

template <typename SrcType, typename DstType>
void FastReorderKernelCodeGenMC<SrcType, DstType>::StoreMulFastReorder(SizeType inWidth)
{
    Ptrue(ARMCG::PRegS(NUM_2));
    Ptrue(ARMCG::PRegS(NUM_3));
    Dup(ARMCG::ZRegS(NUM_0), tmp);
    Dup(ARMCG::ZRegH(NUM_1), 0);
    for (SizeType i = 0; i < inWidth; ++i) {
        ARMCG::CodeGenerator::Mova(ARMCG::ZRegH(NUM_4), KUDNN_JIT_Generator::GetPReg<SrcType>(1),
            ARMCG::ZARegVH(0), ARMCG::WReg(NUM_12 + i / NUM_8), i % NUM_8);
        if constexpr (std::is_same_v<SrcType, __fp16>) {
            Zip1(ARMCG::ZRegH(NUM_2), ARMCG::ZRegH(NUM_4), ARMCG::ZRegH(NUM_1));
            Zip2(ARMCG::ZRegH(NUM_3), ARMCG::ZRegH(NUM_4), ARMCG::ZRegH(NUM_1));
            Fcvt(ARMCG::ZRegS(NUM_2), ARMCG::PRegS(NUM_2), ARMCG::ZRegH(NUM_2));
            Fcvt(ARMCG::ZRegS(NUM_3), ARMCG::PRegS(NUM_2), ARMCG::ZRegH(NUM_3));
            Fmul(ARMCG::ZRegS(NUM_2), ARMCG::ZRegS(NUM_2), ARMCG::ZRegS(NUM_0));
            Fmul(ARMCG::ZRegS(NUM_3), ARMCG::ZRegS(NUM_3), ARMCG::ZRegS(NUM_0));
            Fcvt(ARMCG::ZRegH(NUM_2), ARMCG::PRegS(NUM_2), ARMCG::ZRegS(NUM_2));
            Fcvt(ARMCG::ZRegH(NUM_3), ARMCG::PRegS(NUM_2), ARMCG::ZRegS(NUM_3));
            Uzp1(ARMCG::ZRegH(NUM_4), ARMCG::ZRegH(NUM_2), ARMCG::ZRegH(NUM_3));
        } else if constexpr (std::is_same_v<SrcType, __bf16>) {
            Zip1(ARMCG::ZRegH(NUM_2), ARMCG::ZRegH(NUM_1), ARMCG::ZRegH(NUM_4));
            Zip2(ARMCG::ZRegH(NUM_3), ARMCG::ZRegH(NUM_1), ARMCG::ZRegH(NUM_4));
            Fmul(ARMCG::ZRegS(NUM_2), ARMCG::ZRegS(NUM_2), ARMCG::ZRegS(NUM_0));
            Fmul(ARMCG::ZRegS(NUM_3), ARMCG::ZRegS(NUM_3), ARMCG::ZRegS(NUM_0));
            Bfcvtnt(ARMCG::ZRegH(NUM_2), ARMCG::PRegS(NUM_3), ARMCG::ZRegS(NUM_2));
            Bfcvtnt(ARMCG::ZRegH(NUM_3), ARMCG::PRegS(NUM_3), ARMCG::ZRegS(NUM_3));
            Uzp2(ARMCG::ZRegH(NUM_4), ARMCG::ZRegH(NUM_2), ARMCG::ZRegH(NUM_3));
        }

        KUDNN_JIT_Generator::St1(KUDNN_JIT_Generator::GetZReg<SrcType>(NUM_4),
            KUDNN_JIT_Generator::GetPReg<SrcType>(NUM_1), Ptr(storeOffset));
        Add(storeOffset, storeOffset, strideOut);
    }
}

template <typename SrcType, typename DstType>
FastReorderImplGenerator<FastReorderKernelCodeGenMC<SrcType, DstType>>::FastImpl::FastImpl(
    const ReorderInfo &task, SizeType nThreads) : ReorderImplBase(task, nThreads)
{
    VERBOSE_PRINT;

    Shape srcStridesMC = Service::GetShapeAccordingToLayout(task.srcInfo.GetStrides(), task.srcInfo.GetLayout());
    Shape dstStridesMC = Service::GetShapeAccordingToLayout(task.dstInfo.GetStrides(), task.srcInfo.GetLayout());
    reorderedDims = Service::BroadcastTo5D(Service::GetShapeAccordingToLayout(task.dstInfo.GetDims(),
        task.srcInfo.GetLayout()));

    reorderedStridesSrc = Service::BroadcastTo5D(srcStridesMC, false);
    reorderedStridesDst = Service::BroadcastTo5D(dstStridesMC, false);

    for (SizeType iMC = reorderedDims.GetNumDims() - 1; iMC > 0; --iMC) {
        ReorderHelpers::ReduceDimsAndStrides(reorderedDims, reorderedStridesSrc, reorderedStridesDst);
    }
    for (SizeType iMC = reorderedDims.GetNumDims() - 1; iMC > 0; --iMC) {
        ReorderHelpers::CompressDimsAndStrides(reorderedDims, reorderedStridesSrc, reorderedStridesDst);
    }

    SizeType optNumThreads = std::min(nThreads, Threading::ShouldCalculateOptThreads() ? GetOptNumThreads() : nThreads);
    std::vector<std::pair<Shape, Shape>> reorderWork = Threading::SplitDims(reorderedDims, optNumThreads);

    SizeType mFullBlockThread = reorderWork[0].first[IDX_3];
    SizeType mResidBlockThread = reorderedDims[IDX_3] % mFullBlockThread;

    SizeType nFullBlockThread = reorderWork[0].first[IDX_4];
    SizeType nResidBlockThread = reorderedDims[IDX_4] % nFullBlockThread;

    CodeGenWrapper<FastReorderKernelCodeGenMC<SrcType, DstType>> cgw0;
    cgw0.Initialize(static_cast<SizeType>(mResidBlockThread),
        static_cast<SizeType>(nResidBlockThread), true);
    CodeGenWrapper<FastReorderKernelCodeGenMC<SrcType, DstType>> cgw1;
    cgw1.Initialize(static_cast<SizeType>(mResidBlockThread),
        static_cast<SizeType>(nFullBlockThread), true);
    CodeGenWrapper<FastReorderKernelCodeGenMC<SrcType, DstType>> cgw2;
    cgw2.Initialize(static_cast<SizeType>(mFullBlockThread),
        static_cast<SizeType>(nResidBlockThread), true);
    CodeGenWrapper<FastReorderKernelCodeGenMC<SrcType, DstType>> cgw3;
    cgw3.Initialize(static_cast<SizeType>(mFullBlockThread),
        static_cast<SizeType>(nFullBlockThread), true);

    CodeGenWrapper<FastReorderKernelCodeGenMC<SrcType, DstType>> cgw4;
    cgw4.Initialize(static_cast<SizeType>(mResidBlockThread),
        static_cast<SizeType>(nResidBlockThread), false);
    CodeGenWrapper<FastReorderKernelCodeGenMC<SrcType, DstType>> cgw5;
    cgw5.Initialize(static_cast<SizeType>(mResidBlockThread),
        static_cast<SizeType>(nFullBlockThread), false);
    CodeGenWrapper<FastReorderKernelCodeGenMC<SrcType, DstType>> cgw6;
    cgw6.Initialize(static_cast<SizeType>(mFullBlockThread),
        static_cast<SizeType>(nResidBlockThread), false);
    CodeGenWrapper<FastReorderKernelCodeGenMC<SrcType, DstType>> cgw7;
    cgw7.Initialize(static_cast<SizeType>(mFullBlockThread),
        static_cast<SizeType>(nFullBlockThread), false);
   
    reorderCodegenKernels = {std::move(cgw0), std::move(cgw1), std::move(cgw2), std::move(cgw3),
                             std::move(cgw4), std::move(cgw5), std::move(cgw6), std::move(cgw7)};

    perThreadParams.reserve(reorderWork.size());
    reorderWorkSize = reorderWork.size();

    for (SizeType iMC = 0; iMC < reorderWork.size(); ++iMC) {
        Shape start = Service::BroadcastTo5D(reorderWork[iMC].second, false);
        Shape end = start + Service::BroadcastTo5D(reorderWork[iMC].first, true);
        SizeType m_idx = ((end[IDX_3] - start[IDX_3]) < mFullBlockThread) ? 0 : 2;
        SizeType n_idx = ((end[IDX_4] - start[IDX_4]) < nFullBlockThread) ? 0 : 1;
        perThreadParams.push_back({start, end, m_idx + n_idx});
    }
}

template <typename SrcType, typename DstType>
void FastReorderImplGenerator<FastReorderKernelCodeGenMC<SrcType, DstType>>::FastImpl::RunSequential(
    IntType threadID, const void* src, void* dst, float alpha) const noexcept(false)
{
    VERBOSE_PRINT;
    const SrcType* typedSrc = static_cast<const SrcType*>(src);
    DstType* typedDst = static_cast<DstType*>(dst);
    SizeType idxAlpha = (alpha == 1.0f) ? 1 : 0;

    IntType srcKernelStrideMC = perThreadParams[threadID].threadStart[IDX_3] * reorderedStridesSrc[IDX_3] +
                              perThreadParams[threadID].threadStart[IDX_4] * reorderedStridesSrc[IDX_4];
    IntType dstKernelStrideMC = perThreadParams[threadID].threadStart[IDX_3] * reorderedStridesDst[IDX_3] +
                              perThreadParams[threadID].threadStart[IDX_4] * reorderedStridesDst[IDX_4];

    for (SizeType i0 = perThreadParams[threadID].threadStart[IDX_0];
        i0 < perThreadParams[threadID].threadEnd[IDX_0]; ++i0) {
        for (SizeType i1 = perThreadParams[threadID].threadStart[IDX_1];
            i1 < perThreadParams[threadID].threadEnd[IDX_1]; ++i1) {
            for (SizeType i2 = perThreadParams[threadID].threadStart[IDX_2];
                i2 < perThreadParams[threadID].threadEnd[IDX_2]; ++i2) {
                IntType srcBatchStrideMC = i0 * reorderedStridesSrc[NUM_0] +
                                         i1 * reorderedStridesSrc[NUM_1] +
                                         i2 * reorderedStridesSrc[NUM_2];
                IntType dstBatchStrideMC = i0 * reorderedStridesDst[NUM_0] +
                                         i1 * reorderedStridesDst[NUM_1] +
                                         i2 * reorderedStridesDst[NUM_2];
                auto kernel = reorderCodegenKernels[idxAlpha * NUM_4 + perThreadParams[threadID].kernelIdx].template
                GetCode<void(*)(const SrcType*, DstType*, SizeType, SizeType, float)>();
                kernel(
                    typedSrc + srcBatchStrideMC + srcKernelStrideMC,
                    typedDst + dstBatchStrideMC + dstKernelStrideMC,
                    reorderedStridesSrc[NUM_3] * sizeof(SrcType),
                    reorderedStridesDst[NUM_4] * sizeof(DstType),
                    alpha);
            }
        }
    }
}

template <typename SrcType, typename DstType>
SizeType FastReorderImplGenerator<FastReorderKernelCodeGenMC<SrcType, DstType>>::FastImpl::GetOptNumThreads(
    ) const noexcept
{
    SizeType numElements = reorderedDims.GetTotalDimsSize();

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
}

template
struct FastReorderKernelCodeGenMC<__fp16, __fp16>;
template
struct FastReorderKernelCodeGenMC<__bf16, __bf16>;

template
struct FastReorderImplGenerator<FastReorderKernelCodeGenMC<__fp16, __fp16>>;
template
struct FastReorderImplGenerator<FastReorderKernelCodeGenMC<__bf16, __bf16>>;

} // Detail
} // KuDNN
