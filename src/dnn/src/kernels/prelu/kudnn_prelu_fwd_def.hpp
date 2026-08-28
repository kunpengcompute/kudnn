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

#ifndef KUDNN_PRELU_FWD_DEF_HPP
#define KUDNN_PRELU_FWD_DEF_HPP

#include "kudnn.hpp"

#include "vector_wrappers.hpp"
#include "kudnn_prelu_impl.hpp"
#include "kernels/prelu/kudnn_prelu_kernels.hpp"

namespace KuDNN {
namespace Detail {
namespace PReLU {

// FWD

template <typename CompType, typename SrcType, typename WeightsType, typename OutType, bool IsDup,
    ARCH Arch = ARCH::VC>
struct PreluKernelFWD final {
    static SizeType LoopInc() noexcept
    {
        if constexpr (Arch == ARCH::VC) {
            return Vector::UnrollLength<SrcType, CompType, OutType, WeightsType, float>();
        } else {
            return NUM_2 * Vector::NeonLanes<CompType>();
        }
    }

    static void Do(SizeType loopIdx, SizeType loopRange, const SrcType *src, const WeightsType *wei, OutType *dst,
        SizeType idxSrc, SizeType idxWeights, SizeType idxDst,
        typename Vector::Vec<CompType, Arch>::Type &cmpConst0,
        typename Vector::Vec<CompType, Arch>::Type &cmpConst1,
        typename Vector::Vec<CompType, Arch>::Type &cmpConst2,
        typename Vector::Vec<CompType, Arch>::Type &cmpConst3) noexcept
    {
        typename Vector::Vec<CompType, Arch>::Type src0;
        typename Vector::Vec<CompType, Arch>::Type src1;
        typename Vector::Vec<CompType, Arch>::Type src2;
        typename Vector::Vec<CompType, Arch>::Type src3;
        // load Src
        {
            svbool_t ldSrcPG0;
            svbool_t ldSrcPG1;
            svbool_t ldSrcPG2;
            svbool_t ldSrcPG3;
            Vector::InitPGN<SrcType, Vector::GetUsedRegsN<SrcType, WeightsType, CompType, OutType>()>::Do(
                ldSrcPG0, ldSrcPG1, ldSrcPG2, ldSrcPG3, loopIdx, loopRange);
            Vector::LoadN<SrcType, CompType, Vector::GetUsedRegsN<SrcType, WeightsType, CompType, OutType>(),
                Arch>::Do(src + idxSrc + loopIdx, ldSrcPG0, ldSrcPG1, ldSrcPG2, ldSrcPG3, src0, src1, src2, src3);
        }
        typename Vector::Vec<CompType, Arch>::Type wei0;
        typename Vector::Vec<CompType, Arch>::Type wei1;
        typename Vector::Vec<CompType, Arch>::Type wei2;
        typename Vector::Vec<CompType, Arch>::Type wei3;
        // load Weights
        {
            svbool_t ldWeiPG0;
            svbool_t ldWeiPG1;
            svbool_t ldWeiPG2;
            svbool_t ldWeiPG3;
            Vector::InitPGN<WeightsType, Vector::GetUsedRegsN<WeightsType, SrcType, CompType, OutType>()>::Do(
                ldWeiPG0, ldWeiPG1, ldWeiPG2, ldWeiPG3, loopIdx, loopRange);
            if constexpr (IsDup) {
                Vector::DupWrapperN<WeightsType, CompType,
                    Vector::GetUsedRegsN<WeightsType, SrcType, CompType, OutType>(), Arch>::Do(
                    wei, idxWeights, loopIdx, ldWeiPG0, ldWeiPG1, ldWeiPG2, ldWeiPG3, wei0, wei1, wei2, wei3);
            } else {
                Vector::LoadWrapperN<WeightsType, CompType,
                    Vector::GetUsedRegsN<WeightsType, SrcType, CompType, OutType>(), Arch>::Do(
                    wei, idxWeights, loopIdx, ldWeiPG0, ldWeiPG1, ldWeiPG2, ldWeiPG3, wei0, wei1, wei2, wei3);
            }
        }
        // compute
        svbool_t opPG0;
        svbool_t opPG1;
        svbool_t opPG2;
        svbool_t opPG3;
        Vector::InitPGN<CompType, Vector::GetUsedRegsN<CompType, SrcType, WeightsType, OutType>()>::Do(
            opPG0, opPG1, opPG2, opPG3, loopIdx, loopRange);
        typename Vector::MaskVec<CompType, Arch>::Type mask0;
        typename Vector::MaskVec<CompType, Arch>::Type mask1;
        typename Vector::MaskVec<CompType, Arch>::Type mask2;
        typename Vector::MaskVec<CompType, Arch>::Type mask3;
        Vector::LtN<CompType, Vector::GetUsedRegsN<CompType, SrcType, WeightsType, OutType>(), Arch>::Do(
            opPG0, opPG1, opPG2, opPG3, src0, src1, src2, src3, cmpConst0, cmpConst1, cmpConst2, cmpConst3,
            mask0, mask1, mask2, mask3);
        typename Vector::Vec<CompType, Arch>::Type negPart0;
        typename Vector::Vec<CompType, Arch>::Type negPart1;
        typename Vector::Vec<CompType, Arch>::Type negPart2;
        typename Vector::Vec<CompType, Arch>::Type negPart3;
        Vector::MulXN<CompType, Vector::GetUsedRegsN<CompType, SrcType, WeightsType, OutType>(), Arch>::Do(
            opPG0, opPG1, opPG2, opPG3, src0, src1, src2, src3, wei0, wei1, wei2, wei3,
            negPart0, negPart1, negPart2, negPart3);
        typename Vector::Vec<CompType, Arch>::Type res0;
        typename Vector::Vec<CompType, Arch>::Type res1;
        typename Vector::Vec<CompType, Arch>::Type res2;
        typename Vector::Vec<CompType, Arch>::Type res3;
        Vector::SelN<CompType, Vector::GetUsedRegsN<CompType, SrcType, WeightsType, OutType>(), Arch>::Do(
            mask0, mask1, mask2, mask3, negPart0, negPart1, negPart2, negPart3, src0, src1, src2, src3,
            res0, res1, res2, res3);
        // store results
        svbool_t stPG0;
        svbool_t stPG1;
        svbool_t stPG2;
        svbool_t stPG3;
        Vector::InitPGN<OutType, Vector::GetUsedRegsN<OutType, CompType, SrcType, WeightsType>()>::Do(
            stPG0, stPG1, stPG2, stPG3, loopIdx, loopRange);
        Vector::StoreN<OutType, CompType, Vector::GetUsedRegsN<OutType, CompType, SrcType, WeightsType>(), Arch>::Do(
            stPG0, stPG1, stPG2, stPG3, res0, res1, res2, res3, dst + idxDst + loopIdx);
    }
};

template <typename CompType, typename SrcType, typename WeightsType, typename OutType>
void KernelNonSharedAxesLoad(const Shape &batchSplitDims, const SrcType *src, const WeightsType *wei,
    OutType *dst) noexcept
{
    SizeType size = batchSplitDims.GetTotalDimsSize();
#if USE_NEON
    SizeType neonLoop = size - (size % (NUM_2 * Vector::NeonLanes<CompType>()));
    SizeType tailLoop = size % (NUM_2 * Vector::NeonLanes<CompType>());
    typename Vector::Vec<CompType, ARCH::NEON>::Type cmpConst0;
    typename Vector::Vec<CompType, ARCH::NEON>::Type cmpConst1;
    typename Vector::Vec<CompType, ARCH::NEON>::Type cmpConst2;
    typename Vector::Vec<CompType, ARCH::NEON>::Type cmpConst3;
    typename Vector::Vec<CompType, ARCH::VC>::Type cmpConst4;
    typename Vector::Vec<CompType, ARCH::VC>::Type cmpConst5;
    typename Vector::Vec<CompType, ARCH::VC>::Type cmpConst6;
    typename Vector::Vec<CompType, ARCH::VC>::Type cmpConst7;
    Vector::DupFloatN<SrcType, CompType, Vector::GetUsedRegsN<SrcType, WeightsType, OutType, CompType>(),
        ARCH::NEON>::Do(0.0f, cmpConst0, cmpConst1, cmpConst2, cmpConst3);
    Vector::DupFloatN<SrcType, CompType, Vector::GetUsedRegsN<SrcType, WeightsType, OutType, CompType>(),
        ARCH::VC>::Do(0.0f, cmpConst4, cmpConst5, cmpConst6, cmpConst7);
    Service::UnrollFor::Do<1>(neonLoop, PreluKernelFWD<CompType, SrcType, WeightsType, OutType, false, ARCH::NEON>{},
        src, wei, dst, 0, 0, 0, cmpConst0, cmpConst1, cmpConst2, cmpConst3);
    Service::UnrollFor::Do<1>(tailLoop, PreluKernelFWD<CompType, SrcType, WeightsType, OutType, false, ARCH::VC>{},
        src + neonLoop, wei + neonLoop, dst + neonLoop, 0, 0, 0, cmpConst4, cmpConst5, cmpConst6, cmpConst7);
#else
    typename Vector::Vec<CompType, ARCH::VC>::Type cmpConst0;
    typename Vector::Vec<CompType, ARCH::VC>::Type cmpConst1;
    typename Vector::Vec<CompType, ARCH::VC>::Type cmpConst2;
    typename Vector::Vec<CompType, ARCH::VC>::Type cmpConst3;
    Vector::DupFloatN<SrcType, CompType, Vector::GetUsedRegsN<SrcType, WeightsType, OutType, CompType>(),
        ARCH::VC>::Do(0.0f, cmpConst0, cmpConst1, cmpConst2, cmpConst3);
    Service::UnrollFor::Do<1>(size, PreluKernelFWD<CompType, SrcType, WeightsType, OutType, false, ARCH::VC>{},
        src, wei, dst, 0, 0, 0, cmpConst0, cmpConst1, cmpConst2, cmpConst3);
#endif
}

template <typename CompType, typename SrcType, typename WeightsType, typename OutType>
void KernelNonSharedAxesDup(const Shape &batchSplitDims, const SrcType *src, const WeightsType *wei,
    OutType *dst) noexcept
{
    SizeType size = batchSplitDims.GetTotalDimsSize();
#if USE_NEON
    SizeType neonLoop = size - (size % (NUM_2 * Vector::NeonLanes<CompType>()));
    SizeType tailLoop = size % (NUM_2 * Vector::NeonLanes<CompType>());
    typename Vector::Vec<CompType, ARCH::NEON>::Type cmpConst0;
    typename Vector::Vec<CompType, ARCH::NEON>::Type cmpConst1;
    typename Vector::Vec<CompType, ARCH::NEON>::Type cmpConst2;
    typename Vector::Vec<CompType, ARCH::NEON>::Type cmpConst3;
    typename Vector::Vec<CompType, ARCH::VC>::Type cmpConst4;
    typename Vector::Vec<CompType, ARCH::VC>::Type cmpConst5;
    typename Vector::Vec<CompType, ARCH::VC>::Type cmpConst6;
    typename Vector::Vec<CompType, ARCH::VC>::Type cmpConst7;
    Vector::DupFloatN<SrcType, CompType, Vector::GetUsedRegsN<SrcType, WeightsType, OutType, CompType>(),
        ARCH::NEON>::Do(0.0f, cmpConst0, cmpConst1, cmpConst2, cmpConst3);
    Vector::DupFloatN<SrcType, CompType, Vector::GetUsedRegsN<SrcType, WeightsType, OutType, CompType>(),
        ARCH::VC>::Do(0.0f, cmpConst4, cmpConst5, cmpConst6, cmpConst7);
    Service::UnrollFor::Do<1>(neonLoop, PreluKernelFWD<CompType, SrcType, WeightsType, OutType, true, ARCH::NEON>{},
        src, wei, dst, 0, 0, 0, cmpConst0, cmpConst1, cmpConst2, cmpConst3);
    Service::UnrollFor::Do<1>(tailLoop, PreluKernelFWD<CompType, SrcType, WeightsType, OutType, true, ARCH::VC>{},
        src + neonLoop, wei, dst + neonLoop, 0, 0, 0, cmpConst4, cmpConst5, cmpConst6, cmpConst7);
#else
    typename Vector::Vec<CompType, ARCH::VC>::Type cmpConst0;
    typename Vector::Vec<CompType, ARCH::VC>::Type cmpConst1;
    typename Vector::Vec<CompType, ARCH::VC>::Type cmpConst2;
    typename Vector::Vec<CompType, ARCH::VC>::Type cmpConst3;
    Vector::DupFloatN<SrcType, CompType, Vector::GetUsedRegsN<SrcType, WeightsType, OutType, CompType>(),
        ARCH::VC>::Do(0.0f, cmpConst0, cmpConst1, cmpConst2, cmpConst3);
    Service::UnrollFor::Do<1>(size, PreluKernelFWD<CompType, SrcType, WeightsType, OutType, true, ARCH::VC>{},
        src, wei, dst, 0, 0, 0, cmpConst0, cmpConst1, cmpConst2, cmpConst3);
#endif
}

template <typename CompType, typename SrcType, typename WeightsType, typename OutType>
void KernelSharedAxesLoad(const Shape &batchSplitDims, const SrcType *src, const WeightsType *wei, OutType *dst,
    const Shape &reorderedStridesSrc, const Shape &reorderedStridesWei, const Shape &reorderedStridesDst) noexcept
{
#if USE_NEON
    typename Vector::Vec<CompType, ARCH::NEON>::Type cmpConst0;
    typename Vector::Vec<CompType, ARCH::NEON>::Type cmpConst1;
    typename Vector::Vec<CompType, ARCH::NEON>::Type cmpConst2;
    typename Vector::Vec<CompType, ARCH::NEON>::Type cmpConst3;
    typename Vector::Vec<CompType, ARCH::VC>::Type cmpConst4;
    typename Vector::Vec<CompType, ARCH::VC>::Type cmpConst5;
    typename Vector::Vec<CompType, ARCH::VC>::Type cmpConst6;
    typename Vector::Vec<CompType, ARCH::VC>::Type cmpConst7;
    Vector::DupFloatN<SrcType, CompType, Vector::GetUsedRegsN<SrcType, WeightsType, CompType, OutType>(),
        ARCH::NEON>::Do(0.0f, cmpConst0, cmpConst1, cmpConst2, cmpConst3);
    Vector::DupFloatN<SrcType, CompType, Vector::GetUsedRegsN<SrcType, WeightsType, CompType, OutType>(),
        ARCH::VC>::Do(0.0f, cmpConst4, cmpConst5, cmpConst6, cmpConst7);
#else
    typename Vector::Vec<CompType, ARCH::VC>::Type cmpConst0;
    typename Vector::Vec<CompType, ARCH::VC>::Type cmpConst1;
    typename Vector::Vec<CompType, ARCH::VC>::Type cmpConst2;
    typename Vector::Vec<CompType, ARCH::VC>::Type cmpConst3;
    Vector::DupFloatN<SrcType, CompType, Vector::GetUsedRegsN<SrcType, WeightsType, CompType, OutType>(),
        ARCH::VC>::Do(0.0f, cmpConst0, cmpConst1, cmpConst2, cmpConst3);
#endif

    for (SizeType i0 = 0; i0 < batchSplitDims[IDX_0]; ++i0) {
        SizeType idxSrc0 = i0 * reorderedStridesSrc[IDX_0];
        SizeType idxWeights0 = i0 * reorderedStridesWei[IDX_0];
        SizeType idxDst0 = i0 * reorderedStridesDst[IDX_0];
        for (SizeType i1 = 0; i1 < batchSplitDims[IDX_1]; ++i1) {
            SizeType idxSrc1 = i1 * reorderedStridesSrc[IDX_1];
            SizeType idxWeights1 = i1 * reorderedStridesWei[IDX_1];
            SizeType idxDst1 = i1 * reorderedStridesDst[IDX_1];
            for (SizeType i2 = 0; i2 < batchSplitDims[IDX_2]; ++i2) {
                SizeType idxSrc2 = i2 * reorderedStridesSrc[IDX_2];
                SizeType idxWeights2 = i2 * reorderedStridesWei[IDX_2];
                SizeType idxDst2 = i2 * reorderedStridesDst[IDX_2];
                for (SizeType i3 = 0; i3 < batchSplitDims[IDX_3]; ++i3) {
                    SizeType idxSrc3 = i3 * reorderedStridesSrc[IDX_3];
                    SizeType idxWeights3 = i3 * reorderedStridesWei[IDX_3];
                    SizeType idxDst3 = i3 * reorderedStridesDst[IDX_3];
                    SizeType idxSrc = idxSrc0 + idxSrc1 + idxSrc2 + idxSrc3;
                    SizeType idxWeights = idxWeights0 + idxWeights1 + idxWeights2 + idxWeights3;
                    SizeType idxDst = idxDst0 + idxDst1 + idxDst2 + idxDst3;
#if USE_NEON
                    SizeType neonLoop = batchSplitDims[IDX_4] - (batchSplitDims[IDX_4] % (
                        NUM_2 * Vector::NeonLanes<CompType>()));
                    SizeType tailLoop = batchSplitDims[IDX_4] % (
                        NUM_2 * Vector::NeonLanes<CompType>());
                    Service::UnrollFor::Do<1>(neonLoop,
                        PreluKernelFWD<CompType, SrcType, WeightsType, OutType, false, ARCH::NEON>{},
                        src, wei, dst,
                        idxSrc, idxWeights, idxDst,
                        cmpConst0, cmpConst1, cmpConst2, cmpConst3);
                    Service::UnrollFor::Do<1>(tailLoop,
                        PreluKernelFWD<CompType, SrcType, WeightsType, OutType, false, ARCH::VC>{},
                        src + neonLoop, wei + neonLoop, dst + neonLoop,
                        idxSrc, idxWeights, idxDst,
                        cmpConst4, cmpConst5, cmpConst6, cmpConst7);
#else
                    Service::UnrollFor::Do<1>(batchSplitDims[IDX_4],
                        PreluKernelFWD<CompType, SrcType, WeightsType, OutType, false, ARCH::VC>{},
                        src, wei, dst,
                        idxSrc, idxWeights, idxDst,
                        cmpConst0, cmpConst1, cmpConst2, cmpConst3);
#endif
                }
            }
        }
    }
}

template <typename CompType, typename SrcType, typename WeightsType, typename OutType>
void KernelSharedAxesDup(const Shape &batchSplitDims, const SrcType *src, const WeightsType *wei, OutType *dst,
    const Shape &reorderedStridesSrc, const Shape &reorderedStridesWei, const Shape &reorderedStridesDst) noexcept
{
#if USE_NEON
    typename Vector::Vec<CompType, ARCH::NEON>::Type cmpConst0;
    typename Vector::Vec<CompType, ARCH::NEON>::Type cmpConst1;
    typename Vector::Vec<CompType, ARCH::NEON>::Type cmpConst2;
    typename Vector::Vec<CompType, ARCH::NEON>::Type cmpConst3;
    typename Vector::Vec<CompType, ARCH::VC>::Type cmpConst4;
    typename Vector::Vec<CompType, ARCH::VC>::Type cmpConst5;
    typename Vector::Vec<CompType, ARCH::VC>::Type cmpConst6;
    typename Vector::Vec<CompType, ARCH::VC>::Type cmpConst7;
    Vector::DupFloatN<SrcType, CompType, Vector::GetUsedRegsN<SrcType, WeightsType, CompType, OutType>(),
        ARCH::NEON>::Do(0.0f, cmpConst0, cmpConst1, cmpConst2, cmpConst3);
    Vector::DupFloatN<SrcType, CompType, Vector::GetUsedRegsN<SrcType, WeightsType, CompType, OutType>(),
        ARCH::VC>::Do(0.0f, cmpConst4, cmpConst5, cmpConst6, cmpConst7);
#else
    typename Vector::Vec<CompType, ARCH::VC>::Type cmpConst0;
    typename Vector::Vec<CompType, ARCH::VC>::Type cmpConst1;
    typename Vector::Vec<CompType, ARCH::VC>::Type cmpConst2;
    typename Vector::Vec<CompType, ARCH::VC>::Type cmpConst3;
    Vector::DupFloatN<SrcType, CompType, Vector::GetUsedRegsN<SrcType, WeightsType, CompType, OutType>(),
        ARCH::VC>::Do(0.0f, cmpConst0, cmpConst1, cmpConst2, cmpConst3);
#endif

    for (SizeType i0 = 0; i0 < batchSplitDims[IDX_0]; ++i0) {
        SizeType idxSrc0 = i0 * reorderedStridesSrc[IDX_0];
        SizeType idxWeights0 = i0 * reorderedStridesWei[IDX_0];
        SizeType idxDst0 = i0 * reorderedStridesDst[IDX_0];
        for (SizeType i1 = 0; i1 < batchSplitDims[IDX_1]; ++i1) {
            SizeType idxSrc1 = i1 * reorderedStridesSrc[IDX_1];
            SizeType idxWeights1 = i1 * reorderedStridesWei[IDX_1];
            SizeType idxDst1 = i1 * reorderedStridesDst[IDX_1];
            for (SizeType i2 = 0; i2 < batchSplitDims[IDX_2]; ++i2) {
                SizeType idxSrc2 = i2 * reorderedStridesSrc[IDX_2];
                SizeType idxWeights2 = i2 * reorderedStridesWei[IDX_2];
                SizeType idxDst2 = i2 * reorderedStridesDst[IDX_2];
                for (SizeType i3 = 0; i3 < batchSplitDims[IDX_3]; ++i3) {
                    SizeType idxSrc3 = i3 * reorderedStridesSrc[IDX_3];
                    SizeType idxWeights3 = i3 * reorderedStridesWei[IDX_3];
                    SizeType idxDst3 = i3 * reorderedStridesDst[IDX_3];
                    SizeType idxSrc = idxSrc0 + idxSrc1 + idxSrc2 + idxSrc3;
                    SizeType idxWeights = idxWeights0 + idxWeights1 + idxWeights2 + idxWeights3;
                    SizeType idxDst = idxDst0 + idxDst1 + idxDst2 + idxDst3;
#if USE_NEON
                    SizeType neonLoop = batchSplitDims[IDX_4] - (batchSplitDims[IDX_4] % (
                        NUM_2 * Vector::NeonLanes<CompType>()));
                    SizeType tailLoop = batchSplitDims[IDX_4] % (
                        NUM_2 * Vector::NeonLanes<CompType>());
                    Service::UnrollFor::Do<1>(neonLoop,
                        PreluKernelFWD<CompType, SrcType, WeightsType, OutType, true, ARCH::NEON>{},
                        src, wei, dst,
                        idxSrc, idxWeights, idxDst,
                        cmpConst0, cmpConst1, cmpConst2, cmpConst3);
                    Service::UnrollFor::Do<1>(tailLoop,
                        PreluKernelFWD<CompType, SrcType, WeightsType, OutType, true, ARCH::VC>{},
                        src + neonLoop, wei, dst + neonLoop,
                        idxSrc, idxWeights, idxDst,
                        cmpConst4, cmpConst5, cmpConst6, cmpConst7);
#else
                    Service::UnrollFor::Do<1>(batchSplitDims[IDX_4],
                        PreluKernelFWD<CompType, SrcType, WeightsType, OutType, true, ARCH::VC>{},
                        src, wei, dst,
                        idxSrc, idxWeights, idxDst,
                        cmpConst0, cmpConst1, cmpConst2, cmpConst3);
#endif
                }
            }
        }
    }
}

template <typename CompType, typename SrcType, typename WeightsType, typename OutType>
void KernelGeneric(const Shape &batchSplitDims, const SrcType *src, const WeightsType *wei, OutType *dst,
                   const Shape &reorderedStridesSrc, const Shape &reorderedStridesWei,
                   const Shape &reorderedStridesDst) noexcept
{
    for (SizeType i0 = 0; i0 < batchSplitDims[IDX_0]; ++i0) {
        SizeType idxSrc0 = i0 * reorderedStridesSrc[IDX_0];
        SizeType idxWei0 = i0 * reorderedStridesWei[IDX_0];
        SizeType idxDst0 = i0 * reorderedStridesDst[IDX_0];
        for (SizeType i1 = 0; i1 < batchSplitDims[IDX_1]; ++i1) {
            SizeType idxSrc1 = i1 * reorderedStridesSrc[IDX_1];
            SizeType idxWei1 = i1 * reorderedStridesWei[IDX_1];
            SizeType idxDst1 = i1 * reorderedStridesDst[IDX_1];
            for (SizeType i2 = 0; i2 < batchSplitDims[IDX_2]; ++i2) {
                SizeType idxSrc2 = i2 * reorderedStridesSrc[IDX_2];
                SizeType idxWei2 = i2 * reorderedStridesWei[IDX_2];
                SizeType idxDst2 = i2 * reorderedStridesDst[IDX_2];
                for (SizeType i3 = 0; i3 < batchSplitDims[IDX_3]; ++i3) {
                    SizeType idxSrc3 = i3 * reorderedStridesSrc[IDX_3];
                    SizeType idxWei3 = i3 * reorderedStridesWei[IDX_3];
                    SizeType idxDst3 = i3 * reorderedStridesDst[IDX_3];
                    for (SizeType i4 = 0; i4 < batchSplitDims[IDX_4]; ++i4) {
                        SizeType idxSrc = idxSrc0 + idxSrc1 + idxSrc2 + idxSrc3 + i4 * reorderedStridesSrc[IDX_4];
                        SizeType idxWei = idxWei0 + idxWei1 + idxWei2 + idxWei3 + i4 * reorderedStridesWei[IDX_4];
                        SizeType idxDst = idxDst0 + idxDst1 + idxDst2 + idxDst3 + i4 * reorderedStridesDst[IDX_4];
                        if (Vector::Convert<SrcType, CompType>::Do(src[idxSrc]) > 0.0f) {
                            dst[idxDst] = Vector::Convert<SrcType, OutType>::Do(src[idxSrc]);
                        } else {
                            dst[idxDst] = Vector::Convert<CompType, OutType>::Do(
                                (Vector::Convert<SrcType, CompType>::Do(src[idxSrc]) *
                                Vector::Convert<WeightsType, CompType>::Do(wei[idxWei])));
                        }
                    }
                }
            }
        }
    }
}

} // PReLU FWD
} // Detail
} // KuDNN

#endif //  KUDNN_PRELU_FWD_DEF_HPP
