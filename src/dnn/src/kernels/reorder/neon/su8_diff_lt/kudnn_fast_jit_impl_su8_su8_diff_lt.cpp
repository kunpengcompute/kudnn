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

#include <arm_sve.h>

#include "kudnn.hpp"

#include "cg_codegen.hpp"

#include "kudnn_jit.hpp"
#include "kernels/reorder/kudnn_reorder_support_functions.hpp"
#include "kernels/reorder/kudnn_reorder_jit.hpp"
#include "kernels/reorder/kudnn_reorder_helpers.hpp"
#include "kudnn_wrapper_helpers.hpp"
#include "threading_impl.hpp"
#include "kernels/reorder/neon/su8_diff_lt/kudnn_fast_jit_impl_su8_su8_diff_lt.hpp"

namespace KuDNN {
namespace Detail {

constexpr int REG_INDEX_NUM_2 = 2;
constexpr int REG_INDEX_NUM_3 = 3;
constexpr int REG_INDEX_NUM_4 = 4;
constexpr int REG_INDEX_NUM_5 = 5;
constexpr int REG_INDEX_NUM_6 = 6;

constexpr int REG_LEN_NUM_8 = 8;
constexpr int REG_LEN_NUM_16 = 16;
constexpr int REG_LEN_NUM_24 = 24;


template <typename SrcType, typename DstType>
FastReorderSU8SU8DiffLtKernelCodeGen<SrcType, DstType>::FastReorderSU8SU8DiffLtKernelCodeGen(
    SizeType inHeightSizeNeon, SizeType inWidthSizeNeon, bool hasAlpha)
{
    Preamble();
    if (hasAlpha) {
        Fmov(alphaW, alpha);
    }

    Mov(zero, 0);
    Mov(blockSizeReg, blockSize);
    Mov(heightSize, inHeightSizeNeon);
    Mov(widthSize, inWidthSizeNeon);

    Mov(widthCmp, widthSize);
    Sub(widthCmp, widthCmp, blockSizeReg);

    Mov(heightCmp, heightSize);
    Sub(heightCmp, heightCmp, blockSizeReg);

    Mov(ih, 0);
    if (blockSize < inHeightSizeNeon) {
        ARMCG::Label ihLoopHead;
        L(ihLoopHead);
        {
            GenerateInHeightLoopBody(blockSize, inWidthSizeNeon, hasAlpha);
            Add(ih, ih, blockSizeReg);
            Cmp(ih, heightCmp);
        }
        B(ARMCG::Cond::LO, ihLoopHead);
    }
    GenerateInHeightLoopBody((inHeightSizeNeon % blockSize == 0) ?
                             blockSize : (inHeightSizeNeon % blockSize),
                             inWidthSizeNeon, hasAlpha);
    Postamble();
    Ret();
};

template <typename SrcType, typename DstType>
void FastReorderSU8SU8DiffLtKernelCodeGen<SrcType, DstType>::GenerateInHeightLoopBody(
    SizeType currBlockHeightNeon, SizeType inWidthSizeNeon, bool hasAlpha)
{
    Mov(iw, 0);
    if (blockSize < inWidthSizeNeon) {
        ARMCG::Label iwLoopHead;
        L(iwLoopHead);
        {
            GenerateInWidthLoopBody(currBlockHeightNeon, blockSize, hasAlpha);
            Add(iw, iw, blockSizeReg);
            Cmp(iw, widthCmp);
        }
        B(ARMCG::Cond::LO, iwLoopHead);
    }
    GenerateInWidthLoopBody(currBlockHeightNeon, (inWidthSizeNeon % blockSize == 0)
                            ? blockSize : (inWidthSizeNeon % blockSize), hasAlpha);
}

template <typename SrcType, typename DstType>
void FastReorderSU8SU8DiffLtKernelCodeGen<SrcType, DstType>::GenerateInWidthLoopBody(
    SizeType currBlockHeight, SizeType currBlockWidth, bool hasAlpha)
{
    SizeType vecLen = NUM_16 / sizeof(SrcType);
    SizeType inHeightResid = ((currBlockHeight % vecLen) == 0) ? vecLen : (currBlockHeight % vecLen);

    ARMCG::Label maxIhIsMin;
    Mov(maxIh, ih);
    Add(maxIh, maxIh, blockSizeReg);
    Cmp(maxIh, heightSize);
    B(ARMCG::Cond::LE, maxIhIsMin);
    Mov(maxIh, heightSize);
    L(maxIhIsMin);
    Sub(maxIh, maxIh, vecLen);

    Mov(localIh, ih);
    if (currBlockHeight > vecLen) {
        ARMCG::Label localIhLoopHead;
        L(localIhLoopHead);
        {
            GenerateLocalIhLoopBody(vecLen, currBlockWidth, hasAlpha);
            Add(localIh, localIh, vecLen);
            Cmp(localIh, maxIh);
        }
        B(ARMCG::Cond::LO, localIhLoopHead);
    }
    GenerateLocalIhLoopBody(inHeightResid, currBlockWidth, hasAlpha);
}

template <typename SrcType, typename DstType>
void FastReorderSU8SU8DiffLtKernelCodeGen<SrcType, DstType>::GenerateLocalIhLoopBody(
    SizeType inHeight, SizeType currBlockWidth, bool hasAlpha)
{
    SizeType vecLen = NUM_16 / sizeof(SrcType);
    SizeType inWidthResid = ((currBlockWidth % vecLen) == 0) ? vecLen : (currBlockWidth % vecLen);

    ARMCG::Label maxIwIsMin;
    Mov(maxIw, iw);
    Add(maxIw, maxIw, blockSizeReg);
    Cmp(maxIw, widthSize);
    B(ARMCG::Cond::LE, maxIwIsMin);
    Mov(maxIw, widthSize);
    L(maxIwIsMin);
    Sub(maxIw, maxIw, vecLen);

    Mov(localIw, iw);
    if (currBlockWidth > vecLen) {
        ARMCG::Label localIwLoopHead;
        L(localIwLoopHead);
        {
            GenerateLocalIwLoopBody(inHeight, vecLen, hasAlpha);
            Add(localIw, localIw, vecLen);
            Cmp(localIw, maxIw);
        }
        B(ARMCG::Cond::LO, localIwLoopHead);
    }
    GenerateLocalIwLoopBody(inHeight, inWidthResid, hasAlpha);
}

template <typename SrcType, typename DstType>
void FastReorderSU8SU8DiffLtKernelCodeGen<SrcType, DstType>::GenerateLocalIwLoopBody(
    SizeType inHeight, SizeType inWidth, bool hasAlpha)
{
    SizeType vecLen = NUM_16 / sizeof(SrcType);
    SizeType logVecLen = Math::Log2(vecLen);
    SizeType regOffset = vecLen / NUM_2;

    Mul(localSrcStride, localIh, strideIn);
    Mul(localDstStride, localIw, strideOut);

    Add(localSrcStride, localSrcStride, localIw, ARMCG::ShMod::LSL, Math::Log2(sizeof(SrcType)));
    Add(localDstStride, localDstStride, localIh, ARMCG::ShMod::LSL, Math::Log2(sizeof(SrcType)));

    Mov(loadOffset, bPtr);
    Mov(storeOffset, bBuffer);
    Add(loadOffset, loadOffset, localSrcStride);
    Add(storeOffset, storeOffset, localDstStride);

    Mov(height, inHeight);
    Mov(width, inWidth);
    LoaderFastReorderCore(vecLen, inHeight, inWidth);
    ZipperFastReorderCore(vecLen, logVecLen, regOffset);

    if (hasAlpha) {
        MulFastReorder(vecLen, logVecLen);
    }

    StorerFastReorderCore(vecLen, logVecLen, inHeight, inWidth, hasAlpha);
}

template <typename SrcType, typename DstType>
void FastReorderSU8SU8DiffLtKernelCodeGen<SrcType, DstType>::LoaderFastReorderCore(
    SizeType vecLen, SizeType inHeight, SizeType inWidth)
{
    for (SizeType i = 0; i < inHeight; ++i) {
        if (inWidth == vecLen) {
            CodeGenerator::Ld1(ARMCG::VReg16BList(ARMCG::VReg16B(i)), Ptr(loadOffset));
        } else {
            KUDNN_JIT_Generator::Mov(tmp, loadOffset);
            for (SizeType j = 0; j < inWidth; ++j) {
                CodeGenerator::Ld1(ARMCG::VReg16B(i)[j], ARMCG::PostPtr(tmp, std::int32_t{NUM_1}));
            }
        }
        Add(loadOffset, loadOffset, strideIn);
    }
}

template <typename SrcType, typename DstType>
void FastReorderSU8SU8DiffLtKernelCodeGen<SrcType, DstType>::ZipperFastReorderCore(SizeType vecLen, SizeType logVecLen,
                                                                                   SizeType regOffset)
{
    for (SizeType i = 0; i < logVecLen; ++i) {
        for (SizeType j = 0; j < vecLen; j += NUM_2) {
            Zip1(ARMCG::VReg16B(GetRegIndexInverse(j, i, vecLen)),
                 ARMCG::VReg16B(GetRegIndex(j / NUM_2, i, vecLen)),
                 ARMCG::VReg16B(GetRegIndex(j / NUM_2, i, vecLen) + regOffset));
            Zip2(ARMCG::VReg16B(GetRegIndexInverse(j + NUM_1, i, vecLen)),
                 ARMCG::VReg16B(GetRegIndex(j / NUM_2, i, vecLen)),
                 ARMCG::VReg16B(GetRegIndex(j / NUM_2, i, vecLen) + regOffset));
        }
    }
}

template <typename SrcType, typename DstType>
void FastReorderSU8SU8DiffLtKernelCodeGen<SrcType, DstType>::Shr(const ARMCG::VReg4S &vd,
    const ARMCG::VReg4S &vn, const uint32_t sh)
{
    if constexpr (std::is_same_v<std::int8_t, SrcType>) {
        CodeGenerator::Sshr(vd, vn, sh);
    } else {
        CodeGenerator::Ushr(vd, vn, sh);
    }
}

template <typename SrcType, typename DstType>
void FastReorderSU8SU8DiffLtKernelCodeGen<SrcType, DstType>::MulFastReorder(SizeType vecLen, SizeType logVecLen)
{
    Dup(ARMCG::VReg4S(GetRegIndexInverse(REG_INDEX_NUM_6, logVecLen, vecLen)), alphaW);
    for (SizeType i = 0; i < vecLen; ++i) {
        // shiftleft to "forget" info about higher bits
        CodeGenerator::Shl(ARMCG::VReg4S(GetRegIndexInverse(REG_INDEX_NUM_3, logVecLen, vecLen)),
            ARMCG::VReg4S(GetRegIndex(i, logVecLen, vecLen)), 0); // 0 bits is a size of 0 std::(u)int8_t elements
        CodeGenerator::Shl(ARMCG::VReg4S(GetRegIndexInverse(REG_INDEX_NUM_2, logVecLen, vecLen)),
            ARMCG::VReg4S(GetRegIndex(i, logVecLen, vecLen)), REG_LEN_NUM_8);
        CodeGenerator::Shl(ARMCG::VReg4S(GetRegIndexInverse(1, logVecLen, vecLen)),
            ARMCG::VReg4S(GetRegIndex(i, logVecLen, vecLen)), REG_LEN_NUM_16);
        CodeGenerator::Shl(ARMCG::VReg4S(GetRegIndexInverse(0, logVecLen, vecLen)),
            ARMCG::VReg4S(GetRegIndex(i, logVecLen, vecLen)), REG_LEN_NUM_24);

        // shiftright to "forget" info about lower bits
        Shr(ARMCG::VReg4S(GetRegIndexInverse(REG_INDEX_NUM_3, logVecLen, vecLen)),
            ARMCG::VReg4S(GetRegIndexInverse(REG_INDEX_NUM_3, logVecLen, vecLen)), REG_LEN_NUM_24);
        Shr(ARMCG::VReg4S(GetRegIndexInverse(REG_INDEX_NUM_2, logVecLen, vecLen)),
            ARMCG::VReg4S(GetRegIndexInverse(REG_INDEX_NUM_2, logVecLen, vecLen)), REG_LEN_NUM_24);
        Shr(ARMCG::VReg4S(GetRegIndexInverse(1, logVecLen, vecLen)),
            ARMCG::VReg4S(GetRegIndexInverse(1, logVecLen, vecLen)), REG_LEN_NUM_24);
        Shr(ARMCG::VReg4S(GetRegIndexInverse(0, logVecLen, vecLen)),
            ARMCG::VReg4S(GetRegIndexInverse(0, logVecLen, vecLen)), REG_LEN_NUM_24);

        // convert to float
        Scvtf(ARMCG::VReg4S(GetRegIndexInverse(REG_INDEX_NUM_3, logVecLen, vecLen)),
            ARMCG::VReg4S(GetRegIndexInverse(REG_INDEX_NUM_3, logVecLen, vecLen)));
        Scvtf(ARMCG::VReg4S(GetRegIndexInverse(REG_INDEX_NUM_2, logVecLen, vecLen)),
            ARMCG::VReg4S(GetRegIndexInverse(REG_INDEX_NUM_2, logVecLen, vecLen)));
        Scvtf(ARMCG::VReg4S(GetRegIndexInverse(1, logVecLen, vecLen)),
            ARMCG::VReg4S(GetRegIndexInverse(1, logVecLen, vecLen)));
        Scvtf(ARMCG::VReg4S(GetRegIndexInverse(0, logVecLen, vecLen)),
            ARMCG::VReg4S(GetRegIndexInverse(0, logVecLen, vecLen)));

        // multiply by alpha
        Fmul(ARMCG::VReg4S(GetRegIndexInverse(REG_INDEX_NUM_3, logVecLen, vecLen)),
            ARMCG::VReg4S(GetRegIndexInverse(REG_INDEX_NUM_3, logVecLen, vecLen)),
            ARMCG::VReg4S(GetRegIndexInverse(REG_INDEX_NUM_6, logVecLen, vecLen)));
        Fmul(ARMCG::VReg4S(GetRegIndexInverse(REG_INDEX_NUM_2, logVecLen, vecLen)),
            ARMCG::VReg4S(GetRegIndexInverse(REG_INDEX_NUM_2, logVecLen, vecLen)),
            ARMCG::VReg4S(GetRegIndexInverse(REG_INDEX_NUM_6, logVecLen, vecLen)));
        Fmul(ARMCG::VReg4S(GetRegIndexInverse(1, logVecLen, vecLen)),
            ARMCG::VReg4S(GetRegIndexInverse(1, logVecLen, vecLen)),
            ARMCG::VReg4S(GetRegIndexInverse(REG_INDEX_NUM_6, logVecLen, vecLen)));
        Fmul(ARMCG::VReg4S(GetRegIndexInverse(0, logVecLen, vecLen)),
            ARMCG::VReg4S(GetRegIndexInverse(0, logVecLen, vecLen)),
            ARMCG::VReg4S(GetRegIndexInverse(REG_INDEX_NUM_6, logVecLen, vecLen)));

        // round to nearest integer
        Frinti(ARMCG::VReg4S(GetRegIndexInverse(REG_INDEX_NUM_3, logVecLen, vecLen)),
            ARMCG::VReg4S(GetRegIndexInverse(REG_INDEX_NUM_3, logVecLen, vecLen)));
        Frinti(ARMCG::VReg4S(GetRegIndexInverse(REG_INDEX_NUM_2, logVecLen, vecLen)),
            ARMCG::VReg4S(GetRegIndexInverse(REG_INDEX_NUM_2, logVecLen, vecLen)));
        Frinti(ARMCG::VReg4S(GetRegIndexInverse(1, logVecLen, vecLen)),
            ARMCG::VReg4S(GetRegIndexInverse(1, logVecLen, vecLen)));
        Frinti(ARMCG::VReg4S(GetRegIndexInverse(0, logVecLen, vecLen)),
            ARMCG::VReg4S(GetRegIndexInverse(0, logVecLen, vecLen)));

        // convert to int32
        Fcvtzs(ARMCG::VReg4S(GetRegIndexInverse(REG_INDEX_NUM_3, logVecLen, vecLen)),
            ARMCG::VReg4S(GetRegIndexInverse(REG_INDEX_NUM_3, logVecLen, vecLen)));
        Fcvtzs(ARMCG::VReg4S(GetRegIndexInverse(REG_INDEX_NUM_2, logVecLen, vecLen)),
            ARMCG::VReg4S(GetRegIndexInverse(REG_INDEX_NUM_2, logVecLen, vecLen)));
        Fcvtzs(ARMCG::VReg4S(GetRegIndexInverse(1, logVecLen, vecLen)),
            ARMCG::VReg4S(GetRegIndexInverse(1, logVecLen, vecLen)));
        Fcvtzs(ARMCG::VReg4S(GetRegIndexInverse(0, logVecLen, vecLen)),
            ARMCG::VReg4S(GetRegIndexInverse(0, logVecLen, vecLen)));

        // Movi operates with unsigned int8 numbers and can't be used here (mvni can be used instead)
        Mov(tmpW, std::numeric_limits<DstType>::min());
        Dup(ARMCG::VReg4S(GetRegIndexInverse(REG_INDEX_NUM_4, logVecLen, vecLen)), tmpW);
        Smax(ARMCG::VReg4S(GetRegIndexInverse(REG_INDEX_NUM_3, logVecLen, vecLen)),
            ARMCG::VReg4S(GetRegIndexInverse(REG_INDEX_NUM_3, logVecLen, vecLen)),
            ARMCG::VReg4S(GetRegIndexInverse(REG_INDEX_NUM_4, logVecLen, vecLen)));
        Smax(ARMCG::VReg4S(GetRegIndexInverse(REG_INDEX_NUM_2, logVecLen, vecLen)),
            ARMCG::VReg4S(GetRegIndexInverse(REG_INDEX_NUM_2, logVecLen, vecLen)),
            ARMCG::VReg4S(GetRegIndexInverse(REG_INDEX_NUM_4, logVecLen, vecLen)));
        Smax(ARMCG::VReg4S(GetRegIndexInverse(1, logVecLen, vecLen)),
            ARMCG::VReg4S(GetRegIndexInverse(1, logVecLen, vecLen)),
            ARMCG::VReg4S(GetRegIndexInverse(REG_INDEX_NUM_4, logVecLen, vecLen)));
        Smax(ARMCG::VReg4S(GetRegIndexInverse(0, logVecLen, vecLen)),
            ARMCG::VReg4S(GetRegIndexInverse(0, logVecLen, vecLen)),
            ARMCG::VReg4S(GetRegIndexInverse(REG_INDEX_NUM_4, logVecLen, vecLen)));

        // Movi operates with unsigned int8 numbers and can't be used here (mvni can be used instead)
        Mov(tmpW, std::numeric_limits<DstType>::max());
        Dup(ARMCG::VReg4S(GetRegIndexInverse(REG_INDEX_NUM_5, logVecLen, vecLen)), tmpW);
        Smin(ARMCG::VReg4S(GetRegIndexInverse(REG_INDEX_NUM_3, logVecLen, vecLen)),
            ARMCG::VReg4S(GetRegIndexInverse(REG_INDEX_NUM_3, logVecLen, vecLen)),
            ARMCG::VReg4S(GetRegIndexInverse(REG_INDEX_NUM_5, logVecLen, vecLen)));
        Smin(ARMCG::VReg4S(GetRegIndexInverse(REG_INDEX_NUM_2, logVecLen, vecLen)),
            ARMCG::VReg4S(GetRegIndexInverse(REG_INDEX_NUM_2, logVecLen, vecLen)),
            ARMCG::VReg4S(GetRegIndexInverse(REG_INDEX_NUM_5, logVecLen, vecLen)));
        Smin(ARMCG::VReg4S(GetRegIndexInverse(1, logVecLen, vecLen)),
            ARMCG::VReg4S(GetRegIndexInverse(1, logVecLen, vecLen)),
            ARMCG::VReg4S(GetRegIndexInverse(REG_INDEX_NUM_5, logVecLen, vecLen)));
        Smin(ARMCG::VReg4S(GetRegIndexInverse(0, logVecLen, vecLen)),
            ARMCG::VReg4S(GetRegIndexInverse(0, logVecLen, vecLen)),
            ARMCG::VReg4S(GetRegIndexInverse(REG_INDEX_NUM_5, logVecLen, vecLen)));

        // leftshift to glue the sign
        CodeGenerator::Shl(ARMCG::VReg4S(GetRegIndexInverse(REG_INDEX_NUM_3, logVecLen, vecLen)),
            ARMCG::VReg4S(GetRegIndexInverse(REG_INDEX_NUM_3, logVecLen, vecLen)), REG_LEN_NUM_24);
        CodeGenerator::Shl(ARMCG::VReg4S(GetRegIndexInverse(REG_INDEX_NUM_2, logVecLen, vecLen)),
            ARMCG::VReg4S(GetRegIndexInverse(REG_INDEX_NUM_2, logVecLen, vecLen)), REG_LEN_NUM_24);
        CodeGenerator::Shl(ARMCG::VReg4S(GetRegIndexInverse(1, logVecLen, vecLen)),
            ARMCG::VReg4S(GetRegIndexInverse(1, logVecLen, vecLen)), REG_LEN_NUM_24);
        CodeGenerator::Shl(ARMCG::VReg4S(GetRegIndexInverse(0, logVecLen, vecLen)),
            ARMCG::VReg4S(GetRegIndexInverse(0, logVecLen, vecLen)), REG_LEN_NUM_24);

        // rightshift to initial position
        CodeGenerator::Ushr(ARMCG::VReg4S(GetRegIndexInverse(REG_INDEX_NUM_2, logVecLen, vecLen)),
            ARMCG::VReg4S(GetRegIndexInverse(REG_INDEX_NUM_2, logVecLen, vecLen)), REG_LEN_NUM_8);
        CodeGenerator::Ushr(ARMCG::VReg4S(GetRegIndexInverse(1, logVecLen, vecLen)),
            ARMCG::VReg4S(GetRegIndexInverse(1, logVecLen, vecLen)), REG_LEN_NUM_16);
        CodeGenerator::Ushr(ARMCG::VReg4S(GetRegIndexInverse(0, logVecLen, vecLen)),
            ARMCG::VReg4S(GetRegIndexInverse(0, logVecLen, vecLen)), REG_LEN_NUM_24);

        // unite into one vector
        Orr(ARMCG::VReg16B(GetRegIndexInverse(0, logVecLen, vecLen)),
            ARMCG::VReg16B(GetRegIndexInverse(0, logVecLen, vecLen)),
            ARMCG::VReg16B(GetRegIndexInverse(1, logVecLen, vecLen)));
        Orr(ARMCG::VReg16B(GetRegIndexInverse(REG_INDEX_NUM_2, logVecLen, vecLen)),
            ARMCG::VReg16B(GetRegIndexInverse(REG_INDEX_NUM_2, logVecLen, vecLen)),
            ARMCG::VReg16B(GetRegIndexInverse(REG_INDEX_NUM_3, logVecLen, vecLen)));
        Orr(ARMCG::VReg16B(GetRegIndex(i, logVecLen, vecLen)),
            ARMCG::VReg16B(GetRegIndexInverse(0, logVecLen, vecLen)),
            ARMCG::VReg16B(GetRegIndexInverse(REG_INDEX_NUM_2, logVecLen, vecLen)));
    }
}

template <typename SrcType, typename DstType>
void FastReorderSU8SU8DiffLtKernelCodeGen<SrcType, DstType>::StorerFastReorderCore(SizeType vecLen, SizeType logVecLen,
                                                                                   SizeType inHeight, SizeType inWidth,
                                                                                   bool hasAlpha)
{
    if (std::is_same_v<SrcType, std::int8_t> && std::is_same_v<DstType, std::uint8_t> && !hasAlpha) {
        CodeGenerator::Movi(ARMCG::VReg16B(GetRegIndexInverse(0, logVecLen, vecLen)),
            std::numeric_limits<std::uint8_t>::min());
    } else if (std::is_same_v<SrcType, std::uint8_t> && std::is_same_v<DstType, std::int8_t>  && !hasAlpha) {
        CodeGenerator::Movi(ARMCG::VReg16B(GetRegIndexInverse(0, logVecLen, vecLen)),
            std::numeric_limits<std::int8_t>::max());
    }
    for (SizeType i = 0; i < inWidth; ++i) {
        if (std::is_same_v<SrcType, std::int8_t> && std::is_same_v<DstType, std::uint8_t>  && !hasAlpha) {
            CodeGenerator::Smax(ARMCG::VReg16B(GetRegIndex(i, logVecLen, vecLen)),
                ARMCG::VReg16B(GetRegIndex(i, logVecLen, vecLen)),
                ARMCG::VReg16B(GetRegIndexInverse(0, logVecLen, vecLen)));
        } else if (std::is_same_v<SrcType, std::uint8_t> && std::is_same_v<DstType, std::int8_t> && !hasAlpha) {
            CodeGenerator::Umin(ARMCG::VReg16B(GetRegIndex(i, logVecLen, vecLen)),
                ARMCG::VReg16B(GetRegIndex(i, logVecLen, vecLen)),
                ARMCG::VReg16B(GetRegIndexInverse(0, logVecLen, vecLen)));
        }
        if (inHeight == vecLen) {
            CodeGenerator::St1(ARMCG::VReg16BList(ARMCG::VReg16B(GetRegIndex(i, logVecLen, vecLen))),
                Ptr(storeOffset));
        } else {
            KUDNN_JIT_Generator::Mov(tmp, storeOffset);
            for (SizeType j = 0; j < inHeight; ++j) {
                CodeGenerator::St1(ARMCG::VReg16B(GetRegIndex(i, logVecLen, vecLen))[j],
                    ARMCG::PostPtr(tmp, std::int32_t{NUM_1}));
            }
        }
        Add(storeOffset, storeOffset, strideOut);
    }
}

template <typename SrcType, typename DstType>
FastReorderImplGenerator<FastReorderSU8SU8DiffLtKernelCodeGen<SrcType, DstType>>::FastImpl::FastImpl(
    const ReorderInfo &task, SizeType nThreads) : ReorderImplBase(task, nThreads)
{
    VERBOSE_PRINT;

    Shape srcStridesNeon = Service::GetShapeAccordingToLayout(task.srcInfo.GetStrides(), task.srcInfo.GetLayout());
    Shape dstStridesNeon = Service::GetShapeAccordingToLayout(task.dstInfo.GetStrides(), task.srcInfo.GetLayout());
    reorderedDims = Service::BroadcastTo5D(Service::GetShapeAccordingToLayout(task.dstInfo.GetDims(),
        task.srcInfo.GetLayout()));

    reorderedStridesSrc = Service::BroadcastTo5D(srcStridesNeon, false);
    reorderedStridesDst = Service::BroadcastTo5D(dstStridesNeon, false);

    for (SizeType iNeon = reorderedDims.GetNumDims() - 1; iNeon > 0; --iNeon) {
        ReorderHelpers::ReduceDimsAndStrides(reorderedDims, reorderedStridesSrc, reorderedStridesDst);
    }
    for (SizeType iNeon = reorderedDims.GetNumDims() - 1; iNeon > 0; --iNeon) {
        ReorderHelpers::CompressDimsAndStrides(reorderedDims, reorderedStridesSrc, reorderedStridesDst);
    }

    SizeType optNumThreads = std::min(nThreads, Threading::ShouldCalculateOptThreads() ? GetOptNumThreads() : nThreads);
    std::vector<std::pair<Shape, Shape>> reorderWork = Threading::SplitDims(reorderedDims, optNumThreads);

    SizeType mFullBlockThread = reorderWork[0].first[IDX_3];
    SizeType mResidBlockThread = reorderedDims[IDX_3] % mFullBlockThread;

    SizeType nFullBlockThread = reorderWork[0].first[IDX_4];
    SizeType nResidBlockThread = reorderedDims[IDX_4] % nFullBlockThread;

    CodeGenWrapper<FastReorderSU8SU8DiffLtKernelCodeGen<SrcType, DstType>> cgw0;
    cgw0.Initialize(static_cast<SizeType>(mResidBlockThread),
        static_cast<SizeType>(nResidBlockThread), true);
    CodeGenWrapper<FastReorderSU8SU8DiffLtKernelCodeGen<SrcType, DstType>> cgw1;
    cgw1.Initialize(static_cast<SizeType>(mResidBlockThread),
        static_cast<SizeType>(nFullBlockThread), true);
    CodeGenWrapper<FastReorderSU8SU8DiffLtKernelCodeGen<SrcType, DstType>> cgw2;
    cgw2.Initialize(static_cast<SizeType>(mFullBlockThread),
        static_cast<SizeType>(nResidBlockThread), true);
    CodeGenWrapper<FastReorderSU8SU8DiffLtKernelCodeGen<SrcType, DstType>> cgw3;
    cgw3.Initialize(static_cast<SizeType>(mFullBlockThread),
        static_cast<SizeType>(nFullBlockThread), true);

    CodeGenWrapper<FastReorderSU8SU8DiffLtKernelCodeGen<SrcType, DstType>> cgw4;
    cgw4.Initialize(static_cast<SizeType>(mResidBlockThread),
        static_cast<SizeType>(nResidBlockThread), false);
    CodeGenWrapper<FastReorderSU8SU8DiffLtKernelCodeGen<SrcType, DstType>> cgw5;
    cgw5.Initialize(static_cast<SizeType>(mResidBlockThread),
        static_cast<SizeType>(nFullBlockThread), false);
    CodeGenWrapper<FastReorderSU8SU8DiffLtKernelCodeGen<SrcType, DstType>> cgw6;
    cgw6.Initialize(static_cast<SizeType>(mFullBlockThread),
        static_cast<SizeType>(nResidBlockThread), false);
    CodeGenWrapper<FastReorderSU8SU8DiffLtKernelCodeGen<SrcType, DstType>> cgw7;
    cgw7.Initialize(static_cast<SizeType>(mFullBlockThread),
        static_cast<SizeType>(nFullBlockThread), false);
   
    reorderCodegenKernels = {std::move(cgw0), std::move(cgw1), std::move(cgw2), std::move(cgw3),
                             std::move(cgw4), std::move(cgw5), std::move(cgw6), std::move(cgw7)};

    perThreadParams.reserve(reorderWork.size());
    reorderWorkSize = reorderWork.size();

    for (SizeType iNeon = 0; iNeon < reorderWork.size(); ++iNeon) {
        Shape start = Service::BroadcastTo5D(reorderWork[iNeon].second, false);
        Shape end = start + Service::BroadcastTo5D(reorderWork[iNeon].first, true);
        SizeType m_idx = ((end[IDX_3] - start[IDX_3]) < mFullBlockThread) ? 0 : NUM_2;
        SizeType n_idx = ((end[IDX_4] - start[IDX_4]) < nFullBlockThread) ? 0 : NUM_1;
        perThreadParams.push_back({start, end, m_idx + n_idx});
    }
}

template <typename SrcType, typename DstType>
void FastReorderImplGenerator<FastReorderSU8SU8DiffLtKernelCodeGen<SrcType, DstType>>::FastImpl::RunSequential(
    IntType threadID, const void* src, void* dst, float alpha) const noexcept(false)
{
    VERBOSE_PRINT;
    const SrcType* typedSrc = static_cast<const SrcType*>(src);
    DstType* typedDst = static_cast<DstType*>(dst);
    SizeType idxAlpha = (alpha == 1.0f) ? 1 : 0;

    IntType srcKernelStrideNeon = perThreadParams[threadID].threadStart[IDX_3] * reorderedStridesSrc[IDX_3] +
                              perThreadParams[threadID].threadStart[IDX_4] * reorderedStridesSrc[IDX_4];
    IntType dstKernelStrideNeon = perThreadParams[threadID].threadStart[IDX_3] * reorderedStridesDst[IDX_3] +
                              perThreadParams[threadID].threadStart[IDX_4] * reorderedStridesDst[IDX_4];

    for (SizeType i0 = perThreadParams[threadID].threadStart[IDX_0];
        i0 < perThreadParams[threadID].threadEnd[IDX_0]; ++i0) {
        for (SizeType i1 = perThreadParams[threadID].threadStart[IDX_1];
            i1 < perThreadParams[threadID].threadEnd[IDX_1]; ++i1) {
            for (SizeType i2 = perThreadParams[threadID].threadStart[IDX_2];
                i2 < perThreadParams[threadID].threadEnd[IDX_2]; ++i2) {
                IntType srcBatchStrideNeon = i0 * reorderedStridesSrc[NUM_0] +
                                         i1 * reorderedStridesSrc[NUM_1] +
                                         i2 * reorderedStridesSrc[NUM_2];
                IntType dstBatchStrideNeon = i0 * reorderedStridesDst[NUM_0] +
                                         i1 * reorderedStridesDst[NUM_1] +
                                         i2 * reorderedStridesDst[NUM_2];
                auto kernel = reorderCodegenKernels[idxAlpha * NUM_4 + perThreadParams[threadID].kernelIdx].template
                    GetCode<void(*)(const SrcType*, DstType*, SizeType, SizeType, float)>();
                kernel(
                    typedSrc + srcBatchStrideNeon + srcKernelStrideNeon,
                    typedDst + dstBatchStrideNeon + dstKernelStrideNeon,
                    reorderedStridesSrc[NUM_3] * sizeof(SrcType),
                    reorderedStridesDst[NUM_4] * sizeof(DstType),
                    alpha);
            }
        }
    }
}

template <typename SrcType, typename DstType>
SizeType FastReorderImplGenerator<FastReorderSU8SU8DiffLtKernelCodeGen<SrcType, DstType>>::FastImpl::GetOptNumThreads(
    ) const noexcept
{
    IntType numElements = reorderedDims.GetTotalDimsSize();
    if (reorderedStridesSrc == reorderedStridesDst) {
        return std::max(IntType{1}, numElements / 20000); // 20000 is a constant which will be changed
        //                                                    after perf measurements
    } else {
        return std::max(IntType{1}, numElements / 15000); // 15000 is a constant which will be changed
        //                                                    after perf measurements
    }
}

template
struct FastReorderSU8SU8DiffLtKernelCodeGen<std::int8_t, std::int8_t>;
template
struct FastReorderSU8SU8DiffLtKernelCodeGen<std::int8_t, std::uint8_t>;
template
struct FastReorderSU8SU8DiffLtKernelCodeGen<std::uint8_t, std::int8_t>;
template
struct FastReorderSU8SU8DiffLtKernelCodeGen<std::uint8_t, std::uint8_t>;

template
struct FastReorderImplGenerator<FastReorderSU8SU8DiffLtKernelCodeGen<std::int8_t, std::int8_t>>;
template
struct FastReorderImplGenerator<FastReorderSU8SU8DiffLtKernelCodeGen<std::int8_t, std::uint8_t>>;
template
struct FastReorderImplGenerator<FastReorderSU8SU8DiffLtKernelCodeGen<std::uint8_t, std::int8_t>>;
template
struct FastReorderImplGenerator<FastReorderSU8SU8DiffLtKernelCodeGen<std::uint8_t, std::uint8_t>>;

} // Detail
} // KuDNN
