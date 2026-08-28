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


#ifndef KUDNN_PRELU_BWD_DEF_HPP
#define KUDNN_PRELU_BWD_DEF_HPP

#include "kudnn.hpp"

#include "vector_wrappers.hpp"
#include "kudnn_prelu_impl.hpp"
#include "kernels/prelu/kudnn_prelu_kernels.hpp"

namespace KuDNN {
namespace Detail {
namespace PReLU {

// BWD

template <typename CompType, typename SrcType, typename WeightsType, typename OutType>
struct PreluChannelSharedKernelBWD final {
    static SizeType LoopInc() noexcept
    {
        return Vector::UnrollLength<SrcType, WeightsType, CompType, OutType>();
    }

    static void Do(SizeType loopIdx, SizeType loopRange, const SrcType *src, SrcType *diffSrc, const OutType *diffDst,
        typename Vector::Vec<CompType>::Type &cmpConst0, typename Vector::Vec<CompType>::Type &cmpConst1,
        typename Vector::Vec<CompType>::Type &cmpConst2, typename Vector::Vec<CompType>::Type &cmpConst3,
        typename Vector::Vec<CompType>::Type &wei0, typename Vector::Vec<CompType>::Type &wei1,
        typename Vector::Vec<CompType>::Type &wei2, typename Vector::Vec<CompType>::Type &wei3,
        typename Vector::Vec<CompType>::Type &accConst0, typename Vector::Vec<CompType>::Type &accConst1,
        typename Vector::Vec<CompType>::Type &accConst2, typename Vector::Vec<CompType>::Type &accConst3) noexcept
    {
        typename Vector::Vec<CompType>::Type dd0;
        typename Vector::Vec<CompType>::Type dd1;
        typename Vector::Vec<CompType>::Type dd2;
        typename Vector::Vec<CompType>::Type dd3;
        // load DiffDst
        {
            svbool_t ldPG0;
            svbool_t ldPG1;
            svbool_t ldPG2;
            svbool_t ldPG3;
            Vector::InitPGN<OutType, Vector::GetUsedRegsN<OutType, SrcType, WeightsType, CompType>()>::Do(
                ldPG0, ldPG1, ldPG2, ldPG3, loopIdx, loopRange);
            Vector::LoadN<OutType, CompType, Vector::GetUsedRegsN<OutType, SrcType, WeightsType, CompType>()>::Do(
                diffDst + loopIdx, ldPG0, ldPG1, ldPG2, ldPG3, dd0, dd1, dd2, dd3);
        }
        typename Vector::Vec<CompType>::Type src0;
        typename Vector::Vec<CompType>::Type src1;
        typename Vector::Vec<CompType>::Type src2;
        typename Vector::Vec<CompType>::Type src3;
        // load Src
        {
            svbool_t ldPG0;
            svbool_t ldPG1;
            svbool_t ldPG2;
            svbool_t ldPG3;
            Vector::InitPGN<SrcType, Vector::GetUsedRegsN<SrcType, WeightsType, CompType, OutType>()>::Do(
                ldPG0, ldPG1, ldPG2, ldPG3, loopIdx, loopRange);
            Vector::LoadN<SrcType, CompType, Vector::GetUsedRegsN<SrcType, WeightsType, CompType, OutType>()>::Do(
                src + loopIdx, ldPG0, ldPG1, ldPG2, ldPG3, src0, src1, src2, src3);
        }
        // compute DiffSrc
        svbool_t opPG0;
        svbool_t opPG1;
        svbool_t opPG2;
        svbool_t opPG3;
        Vector::InitPGN<CompType, Vector::GetUsedRegsN<CompType, OutType, SrcType, WeightsType>()>::Do(
            opPG0, opPG1, opPG2, opPG3, loopIdx, loopRange);

        svbool_t mask0;
        svbool_t mask1;
        svbool_t mask2;
        svbool_t mask3;
        Vector::LeN<CompType, Vector::GetUsedRegsN<CompType, OutType, SrcType, WeightsType>()>::Do(
            opPG0, opPG1, opPG2, opPG3, src0, src1, src2, src3, cmpConst0, cmpConst1, cmpConst2, cmpConst3,
            mask0, mask1, mask2, mask3);
        typename Vector::Vec<CompType>::Type negPart0;
        typename Vector::Vec<CompType>::Type negPart1;
        typename Vector::Vec<CompType>::Type negPart2;
        typename Vector::Vec<CompType>::Type negPart3;
        Vector::MulXN<CompType, Vector::GetUsedRegsN<CompType, OutType, SrcType, WeightsType>()>::Do(
            opPG0, opPG1, opPG2, opPG3, dd0, dd1, dd2, dd3, wei0, wei1, wei2, wei3,
            negPart0, negPart1, negPart2, negPart3);
        typename Vector::Vec<CompType>::Type res0;
        typename Vector::Vec<CompType>::Type res1;
        typename Vector::Vec<CompType>::Type res2;
        typename Vector::Vec<CompType>::Type res3;
        Vector::SelN<CompType, Vector::GetUsedRegsN<CompType, OutType, SrcType, WeightsType>()>::Do(
            mask0, mask1, mask2, mask3, negPart0, negPart1, negPart2, negPart3, dd0, dd1, dd2, dd3,
            res0, res1, res2, res3);
        // store DiffSrc
        svbool_t stPG0;
        svbool_t stPG1;
        svbool_t stPG2;
        svbool_t stPG3;
        Vector::InitPGN<SrcType, Vector::GetUsedRegsN<SrcType, OutType, CompType, WeightsType>()>::Do(
            stPG0, stPG1, stPG2, stPG3, loopIdx, loopRange);
        Vector::StoreN<SrcType, CompType, Vector::GetUsedRegsN<SrcType, OutType, CompType, WeightsType>()>::Do(
            stPG0, stPG1, stPG2, stPG3, res0, res1, res2, res3, diffSrc + loopIdx);
        // accumulate
        typename Vector::Vec<CompType>::Type min0;
        typename Vector::Vec<CompType>::Type min1;
        typename Vector::Vec<CompType>::Type min2;
        typename Vector::Vec<CompType>::Type min3;
        Vector::MinXN<CompType, Vector::GetUsedRegsN<CompType, OutType, SrcType, WeightsType>()>::Do(
            opPG0, opPG1, opPG2, opPG3, src0, src1, src2, src3, cmpConst0, cmpConst1, cmpConst2, cmpConst3,
            min0, min1, min2, min3);
        Vector::MulXN<CompType, Vector::GetUsedRegsN<CompType, OutType, SrcType, WeightsType>()>::Do(
            opPG0, opPG1, opPG2, opPG3, dd0, dd1, dd2, dd3, min0, min1, min2, min3, res0, res1, res2, res3);

        Vector::AddMN<CompType, Vector::GetUsedRegsN<CompType, OutType, SrcType, WeightsType>()>::Do(
            opPG0, opPG1, opPG2, opPG3, accConst0, accConst1, accConst2, accConst3, res0, res1, res2, res3,
            accConst0, accConst1, accConst2, accConst3);
    }
};

template <typename CompType, typename SrcType, typename WeightsType, typename OutType>
void KernelChannelShared(const Shape &batchSplitDims, const SrcType *src, SrcType *diffSrc, const WeightsType *wei,
                         WeightsType *diffWeights, const OutType *diffDst, const Shape &reorderedDims) noexcept
{
    /* for this case we don't support multithreading,
        so ArrayProduct of batchSplitDims = 1 */
    Service::Unused(batchSplitDims);

    SizeType size = reorderedDims.GetTotalDimsSize();

    typename Vector::Vec<CompType>::Type wei0;
    typename Vector::Vec<CompType>::Type wei1;
    typename Vector::Vec<CompType>::Type wei2;
    typename Vector::Vec<CompType>::Type wei3;
    Vector::DupN<WeightsType, CompType, Vector::GetUsedRegsN<WeightsType, SrcType, CompType, OutType>()>::Do(
        wei[0], wei0, wei1, wei2, wei3);
    typename Vector::Vec<CompType>::Type cmpConst0;
    typename Vector::Vec<CompType>::Type cmpConst1;
    typename Vector::Vec<CompType>::Type cmpConst2;
    typename Vector::Vec<CompType>::Type cmpConst3;
    Vector::DupFloatN<SrcType, CompType, Vector::GetUsedRegsN<SrcType, WeightsType, CompType, OutType>()>::Do(
        0.0f, cmpConst0, cmpConst1, cmpConst2, cmpConst3);
    typename Vector::Vec<CompType>::Type accConst0;
    typename Vector::Vec<CompType>::Type accConst1;
    typename Vector::Vec<CompType>::Type accConst2;
    typename Vector::Vec<CompType>::Type accConst3;
    Vector::DupFloatN<SrcType, CompType, Vector::GetUsedRegsN<SrcType, WeightsType, CompType, OutType>()>::Do(
        0.0f, accConst0, accConst1, accConst2, accConst3);
    Service::UnrollFor::Do<1>(size, PreluChannelSharedKernelBWD<CompType, SrcType, WeightsType, OutType>{},
        src, diffSrc, diffDst, cmpConst0, cmpConst1, cmpConst2, cmpConst3,
        wei0, wei1, wei2, wei3, accConst0, accConst1, accConst2,  accConst3);
    svbool_t globalOpPG0;
    svbool_t globalOpPG1;
    svbool_t globalOpPG2;
    svbool_t globalOpPG3;
    Vector::InitPGN<CompType, Vector::GetUsedRegsN<CompType, SrcType, WeightsType, OutType>()>::Do(
        globalOpPG0, globalOpPG1, globalOpPG2, globalOpPG3, 0, size);
    /*
        If the WeightsType is int32, we should use int32 for calculations
        since f32 is not precise when values exceed 2^24.
    */
    if constexpr (std::is_same<WeightsType, int32_t>::value) {
        typename Vector::Vec<WeightsType>::Type intAccConst0;
        typename Vector::Vec<WeightsType>::Type intAccConst1;
        typename Vector::Vec<WeightsType>::Type intAccConst2;
        typename Vector::Vec<WeightsType>::Type intAccConst3;
        Vector::ConvertN<CompType, WeightsType,
            Vector::GetUsedRegsN<CompType, SrcType, WeightsType, OutType>()>::Do(
            globalOpPG0, globalOpPG1, globalOpPG2, globalOpPG3, accConst0, accConst1, accConst2, accConst3,
            intAccConst0, intAccConst1, intAccConst2, intAccConst3);
        diffWeights[IDX_0] = Vector::AddvN<WeightsType,
            Vector::GetUsedRegsN<CompType, SrcType, WeightsType, OutType>()>::Do(
            globalOpPG0, globalOpPG1, globalOpPG2, globalOpPG3, intAccConst0, intAccConst1, intAccConst2, intAccConst3);
    } else {
        diffWeights[IDX_0] = Vector::Convert<CompType, WeightsType>::Do(
            Vector::AddvN<CompType, Vector::GetUsedRegsN<CompType, SrcType, WeightsType, OutType>()>::Do(
                globalOpPG0, globalOpPG1, globalOpPG2, globalOpPG3, accConst0, accConst1, accConst2, accConst3));
    }
}


template <typename CompType, typename SrcType, typename WeightsType, typename OutType,
    typename LoadWei, typename UpdDiffWei>
struct PreluSharedAxesKernelBWD final {
    static SizeType LoopInc() noexcept
    {
        return Vector::UnrollLength<SrcType, WeightsType, CompType, OutType, float>();
    }

    static void DoLoad(SizeType loopIdx, SizeType loopRange, const SrcType *src, const WeightsType *wei,
        const OutType *diffDst, SizeType idxWeights, SizeType idxSrcLoop, SizeType idxDstLoop,
        typename Vector::Vec<CompType>::Type &src0, typename Vector::Vec<CompType>::Type &src1,
        typename Vector::Vec<CompType>::Type &src2, typename Vector::Vec<CompType>::Type &src3,
        typename Vector::Vec<CompType>::Type &dd0, typename Vector::Vec<CompType>::Type &dd1,
        typename Vector::Vec<CompType>::Type &dd2, typename Vector::Vec<CompType>::Type &dd3,
        typename Vector::Vec<CompType>::Type &wei0, typename Vector::Vec<CompType>::Type &wei1,
        typename Vector::Vec<CompType>::Type &wei2, typename Vector::Vec<CompType>::Type &wei3) noexcept
    {
        // load Src
        {
            svbool_t ldPG0;
            svbool_t ldPG1;
            svbool_t ldPG2;
            svbool_t ldPG3;
            Vector::InitPGN<SrcType, Vector::GetUsedRegsN<SrcType, WeightsType, CompType, OutType, float>()>::Do(
                ldPG0, ldPG1, ldPG2, ldPG3, loopIdx, loopRange);
            Vector::LoadN<SrcType, CompType,
                Vector::GetUsedRegsN<SrcType, WeightsType, CompType, OutType, float>()>::Do(
                src + idxSrcLoop, ldPG0, ldPG1, ldPG2, ldPG3, src0, src1, src2, src3);
        }
        // load DiffDst
        {
            svbool_t ldPG0;
            svbool_t ldPG1;
            svbool_t ldPG2;
            svbool_t ldPG3;
            Vector::InitPGN<OutType, Vector::GetUsedRegsN<OutType, SrcType, WeightsType, CompType, float>()>::Do(
                ldPG0, ldPG1, ldPG2, ldPG3, loopIdx, loopRange);
            Vector::LoadN<OutType, CompType,
                Vector::GetUsedRegsN<OutType, SrcType, WeightsType, CompType, float>()>::Do(
                diffDst + idxDstLoop, ldPG0, ldPG1, ldPG2, ldPG3, dd0, dd1, dd2, dd3);
        }
        // load Wei
        {
            svbool_t ldPG0;
            svbool_t ldPG1;
            svbool_t ldPG2;
            svbool_t ldPG3;
            Vector::InitPGN<WeightsType, Vector::GetUsedRegsN<WeightsType, SrcType, CompType, OutType, float>()>::Do(
                ldPG0, ldPG1, ldPG2, ldPG3, loopIdx, loopRange);
            LoadWei::Do(wei, idxWeights, loopIdx, ldPG0, ldPG1, ldPG2, ldPG3, wei0, wei1, wei2, wei3);
        }
    }

    static void Do(SizeType loopIdx, SizeType loopRange, const SrcType *src, SrcType *diffSrc,
        const WeightsType *wei, const OutType *diffDst,
        typename Vector::Vec<CompType>::Type &cmpConst0, typename Vector::Vec<CompType>::Type &cmpConst1,
        typename Vector::Vec<CompType>::Type &cmpConst2, typename Vector::Vec<CompType>::Type &cmpConst3,
        std::vector<float, Service::AlignedAllocator<float>> &diffWeiTemp,
        SizeType idxSrc, SizeType idxWeights, SizeType idxDst) noexcept
    {
        SizeType idxSrcLoop = idxSrc + loopIdx;
        SizeType idxDstLoop = idxDst + loopIdx;

        typename Vector::Vec<CompType>::Type src0;
        typename Vector::Vec<CompType>::Type src1;
        typename Vector::Vec<CompType>::Type src2;
        typename Vector::Vec<CompType>::Type src3;
        typename Vector::Vec<CompType>::Type dd0;
        typename Vector::Vec<CompType>::Type dd1;
        typename Vector::Vec<CompType>::Type dd2;
        typename Vector::Vec<CompType>::Type dd3;
        typename Vector::Vec<CompType>::Type wei0;
        typename Vector::Vec<CompType>::Type wei1;
        typename Vector::Vec<CompType>::Type wei2;
        typename Vector::Vec<CompType>::Type wei3;
        DoLoad(loopIdx, loopRange, src, wei, diffDst, idxWeights, idxSrcLoop, idxDstLoop, src0, src1, src2, src3,
            dd0, dd1, dd2, dd3, wei0, wei1, wei2, wei3);
        // compute DiffSrc
        svbool_t opPG0;
        svbool_t opPG1;
        svbool_t opPG2;
        svbool_t opPG3;
        Vector::InitPGN<CompType, Vector::GetUsedRegsN<CompType, WeightsType, SrcType, OutType, float>()>::Do(
            opPG0, opPG1, opPG2, opPG3, loopIdx, loopRange);
        svbool_t mask0;
        svbool_t mask1;
        svbool_t mask2;
        svbool_t mask3;
        Vector::LeN<CompType, Vector::GetUsedRegsN<CompType, WeightsType, SrcType, OutType, float>()>::Do(
            opPG0, opPG1, opPG2, opPG3, src0, src1, src2, src3, cmpConst0, cmpConst1, cmpConst2, cmpConst3,
            mask0, mask1, mask2, mask3);
        typename Vector::Vec<CompType>::Type negPart0;
        typename Vector::Vec<CompType>::Type negPart1;
        typename Vector::Vec<CompType>::Type negPart2;
        typename Vector::Vec<CompType>::Type negPart3;
        Vector::MulXN<CompType, Vector::GetUsedRegsN<CompType, WeightsType, SrcType, OutType, float>()>::Do(
            opPG0, opPG1, opPG2, opPG3, dd0, dd1, dd2, dd3, wei0, wei1, wei2, wei3,
            negPart0, negPart1, negPart2, negPart3);
        typename Vector::Vec<CompType>::Type res0;
        typename Vector::Vec<CompType>::Type res1;
        typename Vector::Vec<CompType>::Type res2;
        typename Vector::Vec<CompType>::Type res3;
        Vector::SelN<CompType, Vector::GetUsedRegsN<CompType, WeightsType, SrcType, OutType, float>()>::Do(
            mask0, mask1, mask2, mask3, negPart0, negPart1, negPart2, negPart3, dd0, dd1, dd2, dd3,
            res0, res1, res2, res3);
        // store DiffSrc
        svbool_t stPG0;
        svbool_t stPG1;
        svbool_t stPG2;
        svbool_t stPG3;
        Vector::InitPGN<SrcType, Vector::GetUsedRegsN<SrcType, OutType, CompType, WeightsType, float>()>::Do(
            stPG0, stPG1, stPG2, stPG3, loopIdx, loopRange);
        Vector::StoreN<SrcType, CompType, Vector::GetUsedRegsN<SrcType, OutType, CompType, WeightsType, float>()>::Do(
            stPG0, stPG1, stPG2, stPG3, res0, res1, res2, res3, diffSrc + idxSrcLoop);
        // accumulate
        typename Vector::Vec<CompType>::Type min0;
        typename Vector::Vec<CompType>::Type min1;
        typename Vector::Vec<CompType>::Type min2;
        typename Vector::Vec<CompType>::Type min3;
        Vector::MinXN<CompType, Vector::GetUsedRegsN<CompType, WeightsType, SrcType, OutType, float>()>::Do(
            opPG0, opPG1, opPG2, opPG3, src0, src1, src2, src3, cmpConst0, cmpConst1, cmpConst2, cmpConst3,
            min0, min1, min2, min3);
        Vector::MulXN<CompType, Vector::GetUsedRegsN<CompType, WeightsType, SrcType, OutType, float>()>::Do(
            opPG0, opPG1, opPG2, opPG3, dd0, dd1, dd2, dd3, min0, min1, min2, min3, res0, res1, res2, res3);

        UpdDiffWei::Do(opPG0, opPG1, opPG2, opPG3, res0, res1, res2, res3, diffWeiTemp.data(),
            idxWeights, loopIdx, loopIdx, loopRange);
    }
};

template <typename CompType, typename SrcType, typename WeightsType, typename OutType,
    typename LoadWei, typename UpdDiffWei>
void KernelSharedAxes(const Shape &batchSplitDims, const SrcType *src, SrcType *diffSrc, const WeightsType *wei,
                      WeightsType *diffWeights, const OutType *diffDst, const Shape &reorderedStridesSrc,
                      const Shape &reorderedStridesWei, const Shape &reorderedWeiDims,
                      const Shape &reorderedStridesDst, const Shape &reorderedDims,
                      LoadWei &&lwei, UpdDiffWei &&udwei) noexcept
{
    Service::Unused(lwei);
    Service::Unused(udwei);
    const WeightsType iniVal = Vector::Convert<float, WeightsType>::Do(0.0f);

    std::vector<float, Service::AlignedAllocator<float>> diffWeiTemp(
        batchSplitDims.GetTotalDimsSize(), 0.0f);
    std::fill(diffWeights, diffWeights +
        batchSplitDims.GetTotalDimsSize(), iniVal);

    std::array<SizeType, MAX_DIMS> dims;
    for (SizeType i = 0; i < dims.size(); ++i) {
        dims[i] = (reorderedWeiDims[i] != 1) ? batchSplitDims[i] : reorderedDims[i];
    }

    typename Vector::Vec<CompType>::Type cmpConst0;
    typename Vector::Vec<CompType>::Type cmpConst1;
    typename Vector::Vec<CompType>::Type cmpConst2;
    typename Vector::Vec<CompType>::Type cmpConst3;
    Vector::DupFloatN<SrcType, CompType, Vector::GetUsedRegsN<SrcType, WeightsType, CompType, OutType>()>::Do(
        0.0f, cmpConst0, cmpConst1, cmpConst2, cmpConst3);
    for (SizeType i0 = 0; i0 < dims[IDX_0]; ++i0) {
        SizeType idxSrc0 = i0 * reorderedStridesSrc[IDX_0];
        SizeType idxWeights0 = i0 * reorderedStridesWei[IDX_0];
        SizeType idxDst0 = i0 * reorderedStridesDst[IDX_0];
        for (SizeType i1 = 0; i1 < dims[IDX_1]; ++i1) {
            SizeType idxSrc1 = i1 * reorderedStridesSrc[IDX_1];
            SizeType idxWeights1 = i1 * reorderedStridesWei[IDX_1];
            SizeType idxDst1 = i1 * reorderedStridesDst[IDX_1];
            for (SizeType i2 = 0; i2 < dims[IDX_2]; ++i2) {
                SizeType idxSrc2 = i2 * reorderedStridesSrc[IDX_2];
                SizeType idxWeights2 = i2 * reorderedStridesWei[IDX_2];
                SizeType idxDst2 = i2 * reorderedStridesDst[IDX_2];
                for (SizeType i3 = 0; i3 < dims[IDX_3]; ++i3) {
                    SizeType idxSrc3 = i3 * reorderedStridesSrc[IDX_3];
                    SizeType idxWeights3 = i3 * reorderedStridesWei[IDX_3];
                    SizeType idxDst3 = i3 * reorderedStridesDst[IDX_3];
                    SizeType idxSrc = idxSrc0 + idxSrc1 + idxSrc2 + idxSrc3;
                    SizeType idxWeights = idxWeights0 + idxWeights1 + idxWeights2 + idxWeights3;
                    SizeType idxDst = idxDst0 + idxDst1 + idxDst2 + idxDst3;
                    Service::UnrollFor::Do<1>(dims[IDX_4],
                        PreluSharedAxesKernelBWD<CompType, SrcType, WeightsType, OutType, LoadWei, UpdDiffWei>{},
                        src, diffSrc, wei, diffDst, cmpConst0, cmpConst1, cmpConst2, cmpConst3,
                        diffWeiTemp, idxSrc, idxWeights, idxDst);
                }
            }
        }
    }
    for (SizeType i = 0; i < dims.size(); ++i) {
        dims[i] = (reorderedWeiDims[i] != 1) ? batchSplitDims[i] : 1;
    }
    for (SizeType i = 0; i < Service::ArrayProduct({dims.data(), dims.size()}, dims.size(), 0); ++i) {
        diffWeights[i] = Vector::Convert<float, WeightsType>::Do(diffWeiTemp[i]);
    }
}


template <typename CompType, typename SrcType, typename WeightsType, typename OutType>
struct PreluWholeTensorKernelBWD final {
    static SizeType LoopInc() noexcept
    {
        return Vector::UnrollLength<SrcType, WeightsType, CompType, OutType>();
    }

    static void DoLoad(SizeType loopIdx, SizeType loopRange, const SrcType *src, const WeightsType *wei,
        const OutType *diffDst,
        typename Vector::Vec<CompType>::Type &src0, typename Vector::Vec<CompType>::Type &src1,
        typename Vector::Vec<CompType>::Type &src2, typename Vector::Vec<CompType>::Type &src3,
        typename Vector::Vec<CompType>::Type &dd0, typename Vector::Vec<CompType>::Type &dd1,
        typename Vector::Vec<CompType>::Type &dd2, typename Vector::Vec<CompType>::Type &dd3,
        typename Vector::Vec<CompType>::Type &wei0, typename Vector::Vec<CompType>::Type &wei1,
        typename Vector::Vec<CompType>::Type &wei2, typename Vector::Vec<CompType>::Type &wei3) noexcept
    {
        svbool_t ldPG0;
        svbool_t ldPG1;
        svbool_t ldPG2;
        svbool_t ldPG3;
        // load Src
        Vector::InitPGN<SrcType, Vector::GetUsedRegsN<SrcType, WeightsType, CompType, OutType>()>::Do(
            ldPG0, ldPG1, ldPG2, ldPG3, loopIdx, loopRange);
        Vector::LoadN<SrcType, CompType, Vector::GetUsedRegsN<SrcType, WeightsType, CompType, OutType>()>::Do(
            src + loopIdx, ldPG0, ldPG1, ldPG2, ldPG3, src0, src1, src2, src3);

        // load DiffDst
        Vector::InitPGN<OutType, Vector::GetUsedRegsN<OutType, SrcType, WeightsType, CompType>()>::Do(
            ldPG0, ldPG1, ldPG2, ldPG3, loopIdx, loopRange);
        Vector::LoadN<OutType, CompType, Vector::GetUsedRegsN<OutType, SrcType, WeightsType, CompType>()>::Do(
            diffDst + loopIdx, ldPG0, ldPG1, ldPG2, ldPG3, dd0, dd1, dd2, dd3);

        // load Wei
        Vector::InitPGN<WeightsType, Vector::GetUsedRegsN<WeightsType, SrcType, CompType, OutType>()>::Do(
            ldPG0, ldPG1, ldPG2, ldPG3, loopIdx, loopRange);
        Vector::LoadN<WeightsType, CompType, Vector::GetUsedRegsN<WeightsType, SrcType, CompType, OutType>()>::Do(
            wei + loopIdx, ldPG0, ldPG1, ldPG2, ldPG3, wei0, wei1, wei2, wei3);
    }

    static void Do(SizeType loopIdx, SizeType loopRange, const SrcType *src,
        SrcType *diffSrc, const WeightsType *wei, WeightsType *diffWeights, const OutType *diffDst,
        typename Vector::Vec<CompType>::Type &cmpConst0, typename Vector::Vec<CompType>::Type &cmpConst1,
        typename Vector::Vec<CompType>::Type &cmpConst2, typename Vector::Vec<CompType>::Type &cmpConst3) noexcept
    {
        typename Vector::Vec<CompType>::Type src0;
        typename Vector::Vec<CompType>::Type src1;
        typename Vector::Vec<CompType>::Type src2;
        typename Vector::Vec<CompType>::Type src3;
        typename Vector::Vec<CompType>::Type wei0;
        typename Vector::Vec<CompType>::Type wei1;
        typename Vector::Vec<CompType>::Type wei2;
        typename Vector::Vec<CompType>::Type wei3;
        typename Vector::Vec<CompType>::Type dd0;
        typename Vector::Vec<CompType>::Type dd1;
        typename Vector::Vec<CompType>::Type dd2;
        typename Vector::Vec<CompType>::Type dd3;
        DoLoad(loopIdx, loopRange, src, wei, diffDst, src0, src1, src2, src3, dd0, dd1, dd2, dd3,
            wei0, wei1, wei2, wei3);
        // compute DiffSrc
        svbool_t opPG0;
        svbool_t opPG1;
        svbool_t opPG2;
        svbool_t opPG3;
        svbool_t mask0;
        svbool_t mask1;
        svbool_t mask2;
        svbool_t mask3;
        Vector::InitPGN<CompType, Vector::GetUsedRegsN<CompType, SrcType, WeightsType, OutType>()>::Do(
            opPG0, opPG1, opPG2, opPG3, loopIdx, loopRange);
        Vector::LeN<CompType, Vector::GetUsedRegsN<CompType, SrcType, WeightsType, OutType>()>::Do(
            opPG0, opPG1, opPG2, opPG3, src0, src1, src2, src3, cmpConst0, cmpConst1, cmpConst2, cmpConst3,
            mask0, mask1, mask2, mask3);
        typename Vector::Vec<CompType>::Type negPart0;
        typename Vector::Vec<CompType>::Type negPart1;
        typename Vector::Vec<CompType>::Type negPart2;
        typename Vector::Vec<CompType>::Type negPart3;
        Vector::MulXN<CompType, Vector::GetUsedRegsN<CompType, SrcType, WeightsType, OutType>()>::Do(
            opPG0, opPG1, opPG2, opPG3, dd0, dd1, dd2, dd3, wei0, wei1, wei2, wei3,
            negPart0, negPart1, negPart2, negPart3);
        typename Vector::Vec<CompType>::Type res0;
        typename Vector::Vec<CompType>::Type res1;
        typename Vector::Vec<CompType>::Type res2;
        typename Vector::Vec<CompType>::Type res3;
        svbool_t stPG0;
        svbool_t stPG1;
        svbool_t stPG2;
        svbool_t stPG3;
        Vector::SelN<CompType, Vector::GetUsedRegsN<CompType, SrcType, WeightsType, OutType>()>::Do(
            mask0, mask1, mask2, mask3, negPart0, negPart1, negPart2, negPart3, dd0, dd1, dd2, dd3,
            res0, res1, res2, res3);

        // store DiffSrc
        Vector::InitPGN<SrcType, Vector::GetUsedRegsN<SrcType, OutType, CompType, WeightsType>()>::Do(
            stPG0, stPG1, stPG2, stPG3, loopIdx, loopRange);
        Vector::StoreN<SrcType, CompType, Vector::GetUsedRegsN<SrcType, OutType, CompType, WeightsType>()>::Do(
            stPG0, stPG1, stPG2, stPG3, res0, res1, res2, res3, diffSrc + loopIdx);

        typename Vector::Vec<CompType>::Type min0;
        typename Vector::Vec<CompType>::Type min1;
        typename Vector::Vec<CompType>::Type min2;
        typename Vector::Vec<CompType>::Type min3;
        Vector::MinXN<CompType, Vector::GetUsedRegsN<CompType, SrcType, WeightsType, OutType>()>::Do(
            opPG0, opPG1, opPG2, opPG3, src0, src1, src2, src3, cmpConst0, cmpConst1, cmpConst2, cmpConst3,
            min0, min1, min2, min3);
        Vector::MulXN<CompType, Vector::GetUsedRegsN<CompType, SrcType, WeightsType, OutType>()>::Do(
            opPG0, opPG1, opPG2, opPG3, dd0, dd1, dd2, dd3, min0, min1, min2, min3, res0, res1, res2, res3);
        // store diffWeights
        Vector::InitPGN<WeightsType, Vector::GetUsedRegsN<WeightsType, OutType, CompType, SrcType>()>::Do(
            stPG0, stPG1, stPG2, stPG3, loopIdx, loopRange);
        Vector::StoreN<WeightsType, CompType, Vector::GetUsedRegsN<WeightsType, OutType, CompType, SrcType>()>::Do(
            stPG0, stPG1, stPG2, stPG3, res0, res1, res2, res3, diffWeights + loopIdx);
    }
};

template <typename CompType, typename SrcType, typename WeightsType, typename OutType>
void KernelWholeTensor(const Shape &batchSplitDims, const SrcType *src, SrcType *diffSrc, const WeightsType *wei,
                       WeightsType *diffWeights, const OutType *diffDst) noexcept
{
    SizeType size = batchSplitDims.GetTotalDimsSize();

    typename Vector::Vec<CompType>::Type cmpConst0;
    typename Vector::Vec<CompType>::Type cmpConst1;
    typename Vector::Vec<CompType>::Type cmpConst2;
    typename Vector::Vec<CompType>::Type cmpConst3;
    Vector::DupFloatN<SrcType, CompType, Vector::GetUsedRegsN<SrcType, WeightsType, CompType, OutType>()>::Do(
        0.0f, cmpConst0, cmpConst1, cmpConst2, cmpConst3);
    Service::UnrollFor::Do<1>(size, PreluWholeTensorKernelBWD<CompType, SrcType, WeightsType, OutType>{}, src,
        diffSrc, wei, diffWeights, diffDst, cmpConst0, cmpConst1, cmpConst2, cmpConst3);
}

} // PReLU BWD
} // Detail
} // KuDNN

#endif //  KUDNN_PRELU_BWD_DEF_HPP
