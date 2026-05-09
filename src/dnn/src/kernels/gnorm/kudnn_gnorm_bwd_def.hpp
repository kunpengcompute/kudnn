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

#ifndef KUDNN_GNORM_BWD_DEF_HPP
#define KUDNN_GNORM_BWD_DEF_HPP

#include "kudnn.hpp"

#include "vector_wrappers.hpp"
#include "kernels/gnorm/kudnn_gnorm_kernels.hpp"

namespace KuDNN {
namespace Detail {
namespace GNorm {

// BWD
template <typename CompType, typename InType, typename OutType, typename ScaleShiftType>
struct GnormAbxBWD final {
    static SizeType LoopInc() noexcept
    {
        return Vector::UnrollLength<InType, CompType, OutType, ScaleShiftType, float>();
    }

    static void Do(
        SizeType loopIdx, SizeType loopRange, const InType *src, const OutType *diffDst,
        typename Vector::Vec<CompType>::Type &mean0, typename Vector::Vec<CompType>::Type &mean1,
        typename Vector::Vec<CompType>::Type &mean2, typename Vector::Vec<CompType>::Type &mean3,
        typename Vector::Vec<CompType>::Type &invSqrtVar0, typename Vector::Vec<CompType>::Type &invSqrtVar1,
        typename Vector::Vec<CompType>::Type &invSqrtVar2, typename Vector::Vec<CompType>::Type &invSqrtVar3,
        typename Vector::Vec<CompType>::Type &diffBetaVec0, typename Vector::Vec<CompType>::Type &diffBetaVec1,
        typename Vector::Vec<CompType>::Type &diffBetaVec2, typename Vector::Vec<CompType>::Type &diffBetaVec3,
        typename Vector::Vec<CompType>::Type &diffGammaVec0, typename Vector::Vec<CompType>::Type &diffGammaVec1,
        typename Vector::Vec<CompType>::Type &diffGammaVec2, typename Vector::Vec<CompType>::Type &diffGammaVec3)
    {
        typename Vector::Vec<CompType>::Type diffDst0;
        typename Vector::Vec<CompType>::Type diffDst1;
        typename Vector::Vec<CompType>::Type diffDst2;
        typename Vector::Vec<CompType>::Type diffDst3;
        {
            svbool_t lddstPG0;
            svbool_t lddstPG1;
            svbool_t lddstPG2;
            svbool_t lddstPG3;
            Vector::InitPGN<OutType, Vector::GetUsedRegsN<OutType, InType, CompType, ScaleShiftType, float>()>::Do(
                lddstPG0, lddstPG1, lddstPG2, lddstPG3, loopIdx, loopRange);
            Vector::LoadN<OutType, CompType, Vector::GetUsedRegsN<OutType, InType, CompType, ScaleShiftType, float>()>::
                Do(diffDst + loopIdx, lddstPG0, lddstPG1, lddstPG2, lddstPG3, diffDst0, diffDst1, diffDst2, diffDst3);
        }
        svbool_t opPG0;
        svbool_t opPG1;
        svbool_t opPG2;
        svbool_t opPG3;
        Vector::InitPGN<CompType, Vector::GetUsedRegsN<CompType, InType, OutType, ScaleShiftType, float>()>::Do(
            opPG0, opPG1, opPG2, opPG3, loopIdx, loopRange);
        Vector::AddMN<CompType, Vector::GetUsedRegsN<CompType, InType, OutType, ScaleShiftType, float>()>::Do(
            opPG0, opPG1, opPG2, opPG3, diffBetaVec0, diffBetaVec1, diffBetaVec2, diffBetaVec3, diffDst0, diffDst1,
            diffDst2, diffDst3, diffBetaVec0, diffBetaVec1, diffBetaVec2, diffBetaVec3); // diffBetaVec0 += dd
        typename Vector::Vec<CompType>::Type src0;
        typename Vector::Vec<CompType>::Type src1;
        typename Vector::Vec<CompType>::Type src2;
        typename Vector::Vec<CompType>::Type src3;
        {
            svbool_t ldsrcPG0;
            svbool_t ldsrcPG1;
            svbool_t ldsrcPG2;
            svbool_t ldsrcPG3;
            Vector::InitPGN<InType, Vector::GetUsedRegsN<InType, OutType, CompType, ScaleShiftType, float>()>::Do(
                ldsrcPG0, ldsrcPG1, ldsrcPG2, ldsrcPG3, loopIdx, loopRange);
            Vector::LoadN<InType, CompType, Vector::GetUsedRegsN<InType, CompType, OutType, ScaleShiftType, float>()>::
                Do(src + loopIdx, ldsrcPG0, ldsrcPG1, ldsrcPG2, ldsrcPG3, src0, src1, src2, src3); // load s
        }
        typename Vector::Vec<CompType>::Type tmp0;
        typename Vector::Vec<CompType>::Type tmp1;
        typename Vector::Vec<CompType>::Type tmp2;
        typename Vector::Vec<CompType>::Type tmp3;
        Vector::SubXN<CompType, Vector::GetUsedRegsN<CompType, InType, OutType, ScaleShiftType, float>()>::Do(
            opPG0, opPG1, opPG2, opPG3, src0, src1, src2, src3, mean0, mean1, mean2, mean3, tmp0, tmp1, tmp2,
            tmp3); // tmp0 = s - mean
        Vector::MulMN<CompType, Vector::GetUsedRegsN<CompType, InType, OutType, ScaleShiftType, float>()>::Do(
            opPG0, opPG1, opPG2, opPG3, tmp0, tmp1, tmp2, tmp3, diffDst0, diffDst1, diffDst2, diffDst3, tmp0, tmp1,
            tmp2, tmp3); // tmp0 *= dd
        Vector::MulMN<CompType, Vector::GetUsedRegsN<CompType, InType, OutType, ScaleShiftType, float>()>::Do(
            opPG0, opPG1, opPG2, opPG3, tmp0, tmp1, tmp2, tmp3, invSqrtVar0, invSqrtVar1, invSqrtVar2, invSqrtVar3,
            tmp0, tmp1, tmp2, tmp3); // tmp0 *= sqrt_variance
        // diffGammaVec0 += tmp0 (dd * (s - mean))
        Vector::AddMN<CompType, Vector::GetUsedRegsN<CompType, InType, OutType, ScaleShiftType, float>()>::Do(
            opPG0, opPG1, opPG2, opPG3, diffGammaVec0, diffGammaVec1, diffGammaVec2, diffGammaVec3, tmp0, tmp1, tmp2,
            tmp3, diffGammaVec0, diffGammaVec1, diffGammaVec2, diffGammaVec3);
    }

    static void DoCalcStat(
        SizeType loopIdx, SizeType loopRange, const InType *src, SizeType outerSize, svbool_t opPG0, svbool_t opPG1,
        svbool_t opPG2, svbool_t opPG3, typename Vector::Vec<CompType>::Type &mean0,
        typename Vector::Vec<CompType>::Type &mean1, typename Vector::Vec<CompType>::Type &mean2,
        typename Vector::Vec<CompType>::Type &mean3, typename Vector::Vec<CompType>::Type &diffBetaVec0,
        typename Vector::Vec<CompType>::Type &diffBetaVec1, typename Vector::Vec<CompType>::Type &diffBetaVec2,
        typename Vector::Vec<CompType>::Type &diffBetaVec3, typename Vector::Vec<CompType>::Type &diffGammaVec0,
        typename Vector::Vec<CompType>::Type &diffGammaVec1, typename Vector::Vec<CompType>::Type &diffGammaVec2,
        typename Vector::Vec<CompType>::Type &diffGammaVec3, typename Vector::Vec<CompType>::Type &invSqrtVariance0,
        typename Vector::Vec<CompType>::Type &invSqrtVariance1, typename Vector::Vec<CompType>::Type &invSqrtVariance2,
        typename Vector::Vec<CompType>::Type &invSqrtVariance3, typename Vector::Vec<CompType>::Type &gammaVec0,
        typename Vector::Vec<CompType>::Type &gammaVec1, typename Vector::Vec<CompType>::Type &gammaVec2,
        typename Vector::Vec<CompType>::Type &gammaVec3, typename Vector::Vec<CompType>::Type &dSrcVec0,
        typename Vector::Vec<CompType>::Type &dSrcVec1, typename Vector::Vec<CompType>::Type &dSrcVec2,
        typename Vector::Vec<CompType>::Type &dSrcVec3)
    {
        typename Vector::Vec<CompType>::Type src0;
        typename Vector::Vec<CompType>::Type src1;
        typename Vector::Vec<CompType>::Type src2;
        typename Vector::Vec<CompType>::Type src3;
        {
            svbool_t ldsrcPG0;
            svbool_t ldsrcPG1;
            svbool_t ldsrcPG2;
            svbool_t ldsrcPG3;
            Vector::InitPGN<InType, Vector::GetUsedRegsN<InType, OutType, CompType, ScaleShiftType, float>()>::Do(
                ldsrcPG0, ldsrcPG1, ldsrcPG2, ldsrcPG3, loopIdx, loopRange);
            Vector::LoadN<InType, CompType, Vector::GetUsedRegsN<InType, CompType, OutType, ScaleShiftType, float>()>::
                Do(src + loopIdx, ldsrcPG0, ldsrcPG1, ldsrcPG2, ldsrcPG3, src0, src1, src2, src3); // load s
        }
        Vector::SubXN<CompType, Vector::GetUsedRegsN<CompType, InType, OutType, ScaleShiftType, float>()>::Do(
            opPG0, opPG1, opPG2, opPG3, src0, src1, src2, src3, mean0, mean1, mean2, mean3, src0, src1, src2,
            src3); // s-= mean
        Vector::MulMN<CompType, Vector::GetUsedRegsN<CompType, InType, OutType, ScaleShiftType, float>()>::Do(
            opPG0, opPG1, opPG2, opPG3, src0, src1, src2, src3, diffGammaVec0, diffGammaVec1, diffGammaVec2,
            diffGammaVec3, src0, src1, src2, src3); // s *= diff_gamma
        Vector::MulMN<CompType, Vector::GetUsedRegsN<CompType, InType, OutType, ScaleShiftType, float>()>::Do(
            opPG0, opPG1, opPG2, opPG3, src0, src1, src2, src3, invSqrtVariance0, invSqrtVariance1, invSqrtVariance2,
            invSqrtVariance3, src0, src1, src2, src3); // s *= sqrt_variance
        Vector::AddMN<CompType, Vector::GetUsedRegsN<CompType, InType, OutType, ScaleShiftType, float>()>::Do(
            opPG0, opPG1, opPG2, opPG3, diffBetaVec0, diffBetaVec1, diffBetaVec2, diffBetaVec3, src0, src1, src2, src3,
            src0, src1, src2, src3); // s += diff_beta
        Vector::DivnXN<CompType, Vector::GetUsedRegsN<CompType, InType, OutType, ScaleShiftType, float>()>::Do(
            opPG0, opPG1, opPG2, opPG3, src0, src1, src2, src3, outerSize, outerSize, outerSize, outerSize, src0, src1,
            src2, src3); // s /= outerSize
        Vector::MulMN<CompType, Vector::GetUsedRegsN<CompType, InType, OutType, ScaleShiftType, float>()>::Do(
            opPG0, opPG1, opPG2, opPG3, src0, src1, src2, src3, gammaVec0, gammaVec1, gammaVec2, gammaVec3, src0, src1,
            src2, src3); // s *= sclae
        Vector::SubXN<CompType, Vector::GetUsedRegsN<CompType, InType, OutType, ScaleShiftType, float>()>::Do(
            opPG0, opPG1, opPG2, opPG3, dSrcVec0, dSrcVec1, dSrcVec2, dSrcVec3, src0, src1, src2, src3, dSrcVec0,
            dSrcVec1, dSrcVec2, dSrcVec3); // dSrcVec0 -= s
    }

    static void Do(
        SizeType loopIdx, SizeType loopRange, const InType *src, const OutType *diffDst, const ScaleShiftType *scale,
        bool useScale, bool calculateStats, typename Vector::Vec<CompType>::Type &mean0,
        typename Vector::Vec<CompType>::Type &mean1, typename Vector::Vec<CompType>::Type &mean2,
        typename Vector::Vec<CompType>::Type &mean3, typename Vector::Vec<CompType>::Type &diffBetaVec0,
        typename Vector::Vec<CompType>::Type &diffBetaVec1, typename Vector::Vec<CompType>::Type &diffBetaVec2,
        typename Vector::Vec<CompType>::Type &diffBetaVec3, typename Vector::Vec<CompType>::Type &diffGammaVec0,
        typename Vector::Vec<CompType>::Type &diffGammaVec1, typename Vector::Vec<CompType>::Type &diffGammaVec2,
        typename Vector::Vec<CompType>::Type &diffGammaVec3, typename Vector::Vec<CompType>::Type &invSqrtVariance0,
        typename Vector::Vec<CompType>::Type &invSqrtVariance1, typename Vector::Vec<CompType>::Type &invSqrtVariance2,
        typename Vector::Vec<CompType>::Type &invSqrtVariance3, InType *diffSrc, SizeType outerSize, SizeType scaleIdx)
    {
        typename Vector::Vec<CompType>::Type gammaVec0;
        typename Vector::Vec<CompType>::Type gammaVec1;
        typename Vector::Vec<CompType>::Type gammaVec2;
        typename Vector::Vec<CompType>::Type gammaVec3;
        if (useScale) {
            Vector::DupN<ScaleShiftType, CompType,
                         Vector::GetUsedRegsN<InType, CompType, OutType, ScaleShiftType, float>()>::Do(scale[scaleIdx],
                                                                                                       gammaVec0,
                                                                                                       gammaVec1,
                                                                                                       gammaVec2,
                                                                                                       gammaVec3);
        } else {
            Vector::DupFloatN<InType, CompType,
                              Vector::GetUsedRegsN<InType, CompType, OutType, ScaleShiftType, float>()>::Do(1.0f,
                                                                                                            gammaVec0,
                                                                                                            gammaVec1,
                                                                                                            gammaVec2,
                                                                                                            gammaVec3);
        }
        typename Vector::Vec<CompType>::Type diffDst0;
        typename Vector::Vec<CompType>::Type diffDst1;
        typename Vector::Vec<CompType>::Type diffDst2;
        typename Vector::Vec<CompType>::Type diffDst3;
        svbool_t lddstPG0;
        svbool_t lddstPG1;
        svbool_t lddstPG2;
        svbool_t lddstPG3;
        Vector::InitPGN<OutType, Vector::GetUsedRegsN<OutType, InType, CompType, ScaleShiftType, float>()>::Do(
            lddstPG0, lddstPG1, lddstPG2, lddstPG3, loopIdx, loopRange);
        Vector::LoadN<OutType, CompType, Vector::GetUsedRegsN<OutType, InType, CompType, ScaleShiftType, float>()>::Do(
            diffDst + loopIdx, lddstPG0, lddstPG1, lddstPG2, lddstPG3, diffDst0, diffDst1, diffDst2, diffDst3);

        typename Vector::Vec<CompType>::Type dSrcVec0;
        typename Vector::Vec<CompType>::Type dSrcVec1;
        typename Vector::Vec<CompType>::Type dSrcVec2;
        typename Vector::Vec<CompType>::Type dSrcVec3;
        svbool_t opPG0;
        svbool_t opPG1;
        svbool_t opPG2;
        svbool_t opPG3;
        Vector::InitPGN<CompType, Vector::GetUsedRegsN<CompType, InType, OutType, ScaleShiftType, float>()>::Do(
            opPG0, opPG1, opPG2, opPG3, loopIdx, loopRange);
        // dSrcVec0 = dd * gammaVec0
        Vector::MulMN<CompType, Vector::GetUsedRegsN<CompType, InType, OutType, ScaleShiftType, float>()>::Do(
            opPG0, opPG1, opPG2, opPG3, diffDst0, diffDst1, diffDst2, diffDst3, gammaVec0, gammaVec1, gammaVec2,
            gammaVec3, dSrcVec0, dSrcVec1, dSrcVec2, dSrcVec3);
        if (calculateStats) {
            DoCalcStat(loopIdx, loopRange, src, outerSize, opPG0, opPG1, opPG2, opPG3, mean0, mean1, mean2, mean3,
                       diffBetaVec0, diffBetaVec1, diffBetaVec2, diffBetaVec3, diffGammaVec0, diffGammaVec1,
                       diffGammaVec2, diffGammaVec3, invSqrtVariance0, invSqrtVariance1, invSqrtVariance2,
                       invSqrtVariance3, gammaVec0, gammaVec1, gammaVec2, gammaVec3, dSrcVec0, dSrcVec1, dSrcVec2,
                       dSrcVec3);
        }
        Vector::MulMN<CompType, Vector::GetUsedRegsN<CompType, InType, OutType, ScaleShiftType, float>()>::Do(
            opPG0, opPG1, opPG2, opPG3, dSrcVec0, dSrcVec1, dSrcVec2, dSrcVec3, invSqrtVariance0, invSqrtVariance1,
            invSqrtVariance2, invSqrtVariance3, dSrcVec0, dSrcVec1, dSrcVec2, dSrcVec3); // dSrcVec0 *= invSqrtVariance0
        svbool_t stPG0;
        svbool_t stPG1;
        svbool_t stPG2;
        svbool_t stPG3;
        Vector::InitPGN<InType, Vector::GetUsedRegsN<InType, OutType, CompType, ScaleShiftType, float>()>::Do(
            stPG0, stPG1, stPG2, stPG3, loopIdx, loopRange);
        Vector::StoreN<InType, CompType, Vector::GetUsedRegsN<InType, OutType, CompType, ScaleShiftType, float>()>::Do(
            stPG0, stPG1, stPG2, stPG3, dSrcVec0, dSrcVec1, dSrcVec2, dSrcVec3, diffSrc + loopIdx); // store dSrcVec0
    }
};

template <typename CompType, typename InType, typename OutType, typename ScaleShiftType>
void EvaluteAbx(SizeType batchNum, SizeType outerSize, const InType *src, const float *mean, const float *variance,
                const OutType *diffDst, const ScaleShiftType *scale, InType *diffSrc, float *diffScale,
                float *diffShift, float eps, NormalizationFlags flags, SizeType channelSize,
                SizeType groupSize) noexcept
{
    bool calculateStats = !static_cast<bool>(flags & NormalizationFlags::USE_GLOBAL_STATS);
    bool useScale = static_cast<bool>(flags & NormalizationFlags::USE_SCALE);
    bool useShift = static_cast<bool>(flags & NormalizationFlags::USE_SHIFT);
    svbool_t pgTrue = svptrue_b32();
    SizeType groupNum = channelSize / groupSize;
    for (SizeType g = 0; g < groupNum; g++) {
        SizeType dhw = outerSize / groupSize;
        for (SizeType c = 0; c < groupSize; c++) {
            CompType diffBeta = 0.0f;
            typename Vector::Vec<CompType>::Type diffBetaVec0;
            typename Vector::Vec<CompType>::Type diffBetaVec1;
            typename Vector::Vec<CompType>::Type diffBetaVec2;
            typename Vector::Vec<CompType>::Type diffBetaVec3;
            Vector::DupFloatN<
                InType, CompType,
                Vector::GetUsedRegsN<InType, CompType, OutType, ScaleShiftType, float>()>::Do(0.0f, diffBetaVec0,
                                                                                              diffBetaVec1,
                                                                                              diffBetaVec2,
                                                                                              diffBetaVec3);
            CompType diffGamma = 0.0f;
            typename Vector::Vec<CompType>::Type diffGammaVec0;
            typename Vector::Vec<CompType>::Type diffGammaVec1;
            typename Vector::Vec<CompType>::Type diffGammaVec2;
            typename Vector::Vec<CompType>::Type diffGammaVec3;
            Vector::DupFloatN<
                InType, CompType,
                Vector::GetUsedRegsN<InType, CompType, OutType, ScaleShiftType, float>()>::Do(0.0f, diffGammaVec0,
                                                                                              diffGammaVec1,
                                                                                              diffGammaVec2,
                                                                                              diffGammaVec3);

            if (calculateStats || useScale || useShift) {
                for (SizeType b = 0; b < batchNum; b++) {
                    CompType invSqrtVariance = 1.0f / std::sqrt(variance[b * groupNum + g] + eps);
                    typename Vector::Vec<CompType>::Type invSqrtVariance0;
                    typename Vector::Vec<CompType>::Type invSqrtVariance1;
                    typename Vector::Vec<CompType>::Type invSqrtVariance2;
                    typename Vector::Vec<CompType>::Type invSqrtVariance3;
                    typename Vector::Vec<CompType>::Type mean0;
                    typename Vector::Vec<CompType>::Type mean1;
                    typename Vector::Vec<CompType>::Type mean2;
                    typename Vector::Vec<CompType>::Type mean3;
                    Vector::DupFloatN<InType, CompType,
                                      Vector::GetUsedRegsN<InType, CompType, OutType, ScaleShiftType, float>()>::
                        Do(mean[b * groupNum + g], mean0, mean1, mean2, mean3);
                    Vector::DupFloatN<InType, CompType,
                                      Vector::GetUsedRegsN<InType, CompType, OutType, ScaleShiftType, float>()>::
                        Do(invSqrtVariance, invSqrtVariance0, invSqrtVariance1, invSqrtVariance2, invSqrtVariance3);
                    Service::UnrollFor::Do<1>(dhw, GnormAbxBWD<CompType, InType, OutType, ScaleShiftType>{},
                                              src + (b * channelSize + g * groupSize + c) * dhw,
                                              diffDst + (b * channelSize + g * groupSize + c) * dhw, mean0, mean1,
                                              mean2, mean3, invSqrtVariance0, invSqrtVariance1, invSqrtVariance2,
                                              invSqrtVariance3, diffBetaVec0, diffBetaVec1, diffBetaVec2, diffBetaVec3,
                                              diffGammaVec0, diffGammaVec1, diffGammaVec2, diffGammaVec3);
                    diffBeta = Vector::
                        AddvN<CompType, Vector::GetUsedRegsN<CompType, InType, OutType, ScaleShiftType, float>()>::Do(
                            pgTrue, pgTrue, pgTrue, pgTrue, diffBetaVec0, diffBetaVec1, diffBetaVec2, diffBetaVec3);

                    diffGamma = Vector::
                        AddvN<CompType, Vector::GetUsedRegsN<CompType, InType, OutType, ScaleShiftType, float>()>::Do(
                            pgTrue, pgTrue, pgTrue, pgTrue, diffGammaVec0, diffGammaVec1, diffGammaVec2, diffGammaVec3);

                    if (useScale) {
                        diffScale[g * groupSize + c] = diffGamma;
                    }
                    if (useShift) {
                        diffShift[g * groupSize + c] = diffBeta;
                    }
                }
            }

            Vector::DupFloatN<
                InType, CompType,
                Vector::GetUsedRegsN<InType, CompType, OutType, ScaleShiftType, float>()>::Do(diffBeta, diffBetaVec0,
                                                                                              diffBetaVec1,
                                                                                              diffBetaVec2,
                                                                                              diffBetaVec3);
            Vector::DupFloatN<
                InType, CompType,
                Vector::GetUsedRegsN<InType, CompType, OutType, ScaleShiftType, float>()>::Do(diffGamma, diffGammaVec0,
                                                                                              diffGammaVec1,
                                                                                              diffGammaVec2,
                                                                                              diffGammaVec3);

            for (SizeType b = 0; b < batchNum; b++) {
                CompType invSqrtVariance = 1.0f / std::sqrt(variance[b * groupNum + g] + eps);
                typename Vector::Vec<CompType>::Type invSqrtVariance0;
                typename Vector::Vec<CompType>::Type invSqrtVariance1;
                typename Vector::Vec<CompType>::Type invSqrtVariance2;
                typename Vector::Vec<CompType>::Type invSqrtVariance3;
                typename Vector::Vec<CompType>::Type mean0;
                typename Vector::Vec<CompType>::Type mean1;
                typename Vector::Vec<CompType>::Type mean2;
                typename Vector::Vec<CompType>::Type mean3;
                Vector::DupFloatN<InType, CompType,
                                  Vector::GetUsedRegsN<InType, CompType, OutType, ScaleShiftType, float>()>::
                    Do(mean[b * groupNum + g], mean0, mean1, mean2, mean3);
                Vector::DupFloatN<InType, CompType,
                                  Vector::GetUsedRegsN<InType, CompType, OutType, ScaleShiftType, float>()>::
                    Do(invSqrtVariance, invSqrtVariance0, invSqrtVariance1, invSqrtVariance2, invSqrtVariance3);
                Service::UnrollFor::Do<1>(dhw, GnormAbxBWD<CompType, InType, OutType, ScaleShiftType>{},
                                          src + (b * channelSize + g * groupSize + c) * dhw,
                                          diffDst + (b * channelSize + g * groupSize + c) * dhw, scale, useScale,
                                          calculateStats, mean0, mean1, mean2, mean3, diffBetaVec0, diffBetaVec1,
                                          diffBetaVec2, diffBetaVec3, diffGammaVec0, diffGammaVec1, diffGammaVec2,
                                          diffGammaVec3, invSqrtVariance0, invSqrtVariance1, invSqrtVariance2,
                                          invSqrtVariance3, diffSrc + (b * channelSize + g * groupSize + c) * dhw,
                                          outerSize, g * groupSize + c);
            }
        }
    }
}

template <typename CompType, typename InType, typename OutType, typename ScaleShiftType>
struct GnormAxbBWD final {
    static SizeType LoopInc() noexcept
    {
        return Vector::UnrollLength<InType, CompType, OutType, ScaleShiftType, float>();
    }

    static void Do(
        SizeType loopIdx, SizeType loopRange, const InType *src, const OutType *diffDst,
        typename Vector::Vec<CompType>::Type &mean0, typename Vector::Vec<CompType>::Type &mean1,
        typename Vector::Vec<CompType>::Type &mean2, typename Vector::Vec<CompType>::Type &mean3,
        typename Vector::Vec<CompType>::Type &invSqrtVar0, typename Vector::Vec<CompType>::Type &invSqrtVar1,
        typename Vector::Vec<CompType>::Type &invSqrtVar2, typename Vector::Vec<CompType>::Type &invSqrtVar3,
        typename Vector::Vec<CompType>::Type &diffBetaVec0, typename Vector::Vec<CompType>::Type &diffBetaVec1,
        typename Vector::Vec<CompType>::Type &diffBetaVec2, typename Vector::Vec<CompType>::Type &diffBetaVec3,
        typename Vector::Vec<CompType>::Type &diffGammaVec0, typename Vector::Vec<CompType>::Type &diffGammaVec1,
        typename Vector::Vec<CompType>::Type &diffGammaVec2, typename Vector::Vec<CompType>::Type &diffGammaVec3)
    {
        typename Vector::Vec<CompType>::Type diffDst0;
        typename Vector::Vec<CompType>::Type diffDst1;
        typename Vector::Vec<CompType>::Type diffDst2;
        typename Vector::Vec<CompType>::Type diffDst3;
        {
            svbool_t lddstPG0;
            svbool_t lddstPG1;
            svbool_t lddstPG2;
            svbool_t lddstPG3;
            Vector::InitPGN<OutType, Vector::GetUsedRegsN<OutType, InType, CompType, ScaleShiftType, float>()>::Do(
                lddstPG0, lddstPG1, lddstPG2, lddstPG3, loopIdx, loopRange);
            Vector::LoadN<OutType, CompType, Vector::GetUsedRegsN<OutType, InType, CompType, ScaleShiftType, float>()>::
                Do(diffDst + loopIdx, lddstPG0, lddstPG1, lddstPG2, lddstPG3, diffDst0, diffDst1, diffDst2, diffDst3);
        }
        svbool_t opPG0;
        svbool_t opPG1;
        svbool_t opPG2;
        svbool_t opPG3;
        Vector::InitPGN<CompType, Vector::GetUsedRegsN<CompType, InType, OutType, ScaleShiftType, float>()>::Do(
            opPG0, opPG1, opPG2, opPG3, loopIdx, loopRange);
        Vector::AddMN<CompType, Vector::GetUsedRegsN<CompType, InType, OutType, ScaleShiftType, float>()>::Do(
            opPG0, opPG1, opPG2, opPG3, diffBetaVec0, diffBetaVec1, diffBetaVec2, diffBetaVec3, diffDst0, diffDst1,
            diffDst2, diffDst3, diffBetaVec0, diffBetaVec1, diffBetaVec2, diffBetaVec3); // diffBetaVec0 += dd
        typename Vector::Vec<CompType>::Type src0;
        typename Vector::Vec<CompType>::Type src1;
        typename Vector::Vec<CompType>::Type src2;
        typename Vector::Vec<CompType>::Type src3;
        {
            svbool_t ldsrcPG0;
            svbool_t ldsrcPG1;
            svbool_t ldsrcPG2;
            svbool_t ldsrcPG3;
            Vector::InitPGN<InType, Vector::GetUsedRegsN<InType, OutType, CompType, ScaleShiftType, float>()>::Do(
                ldsrcPG0, ldsrcPG1, ldsrcPG2, ldsrcPG3, loopIdx, loopRange);
            Vector::LoadN<InType, CompType, Vector::GetUsedRegsN<InType, CompType, OutType, ScaleShiftType, float>()>::
                Do(src + loopIdx, ldsrcPG0, ldsrcPG1, ldsrcPG2, ldsrcPG3, src0, src1, src2, src3); // load s
        }
        typename Vector::Vec<CompType>::Type tmp0;
        typename Vector::Vec<CompType>::Type tmp1;
        typename Vector::Vec<CompType>::Type tmp2;
        typename Vector::Vec<CompType>::Type tmp3;
        Vector::SubXN<CompType, Vector::GetUsedRegsN<CompType, InType, OutType, ScaleShiftType, float>()>::Do(
            opPG0, opPG1, opPG2, opPG3, src0, src1, src2, src3, mean0, mean1, mean2, mean3, tmp0, tmp1, tmp2,
            tmp3); // tmp0 = s - mean
        Vector::MulMN<CompType, Vector::GetUsedRegsN<CompType, InType, OutType, ScaleShiftType, float>()>::Do(
            opPG0, opPG1, opPG2, opPG3, tmp0, tmp1, tmp2, tmp3, diffDst0, diffDst1, diffDst2, diffDst3, tmp0, tmp1,
            tmp2, tmp3); // tmp0 *= dd
        Vector::MulMN<CompType, Vector::GetUsedRegsN<CompType, InType, OutType, ScaleShiftType, float>()>::Do(
            opPG0, opPG1, opPG2, opPG3, tmp0, tmp1, tmp2, tmp3, invSqrtVar0, invSqrtVar1, invSqrtVar2, invSqrtVar3,
            tmp0, tmp1, tmp2, tmp3); // tmp0 *= sqrt_variance
        // diffGammaVec0 += tmp0 (dd * (s - mean))
        Vector::AddMN<CompType, Vector::GetUsedRegsN<CompType, InType, OutType, ScaleShiftType, float>()>::Do(
            opPG0, opPG1, opPG2, opPG3, diffGammaVec0, diffGammaVec1, diffGammaVec2, diffGammaVec3, tmp0, tmp1, tmp2,
            tmp3, diffGammaVec0, diffGammaVec1, diffGammaVec2, diffGammaVec3);
    }

    static void Do(SizeType loopIdx, SizeType loopRange, const OutType *diffDst,
                   typename Vector::Vec<CompType>::Type &invSqrtVariance0,
                   typename Vector::Vec<CompType>::Type &invSqrtVariance1,
                   typename Vector::Vec<CompType>::Type &invSqrtVariance2,
                   typename Vector::Vec<CompType>::Type &invSqrtVariance3, InType *diffSrc)
    {
        typename Vector::Vec<CompType>::Type gammaVec0;
        typename Vector::Vec<CompType>::Type gammaVec1;
        typename Vector::Vec<CompType>::Type gammaVec2;
        typename Vector::Vec<CompType>::Type gammaVec3;

        Vector::DupFloatN<InType, CompType,
                          Vector::GetUsedRegsN<InType, CompType, OutType, ScaleShiftType, float>()>::Do(1.0f, gammaVec0,
                                                                                                        gammaVec1,
                                                                                                        gammaVec2,
                                                                                                        gammaVec3);

        typename Vector::Vec<CompType>::Type diffDst0;
        typename Vector::Vec<CompType>::Type diffDst1;
        typename Vector::Vec<CompType>::Type diffDst2;
        typename Vector::Vec<CompType>::Type diffDst3;
        {
            svbool_t lddstPG0;
            svbool_t lddstPG1;
            svbool_t lddstPG2;
            svbool_t lddstPG3;
            Vector::InitPGN<OutType, Vector::GetUsedRegsN<OutType, InType, CompType, ScaleShiftType, float>()>::Do(
                lddstPG0, lddstPG1, lddstPG2, lddstPG3, loopIdx, loopRange);
            Vector::LoadN<OutType, CompType, Vector::GetUsedRegsN<OutType, InType, CompType, ScaleShiftType, float>()>::
                Do(diffDst + loopIdx, lddstPG0, lddstPG1, lddstPG2, lddstPG3, diffDst0, diffDst1, diffDst2, diffDst3);
        }
        typename Vector::Vec<CompType>::Type dSrcVec0;
        typename Vector::Vec<CompType>::Type dSrcVec1;
        typename Vector::Vec<CompType>::Type dSrcVec2;
        typename Vector::Vec<CompType>::Type dSrcVec3;
        svbool_t opPG0;
        svbool_t opPG1;
        svbool_t opPG2;
        svbool_t opPG3;
        Vector::InitPGN<CompType, Vector::GetUsedRegsN<CompType, InType, OutType, ScaleShiftType, float>()>::Do(
            opPG0, opPG1, opPG2, opPG3, loopIdx, loopRange);
        // dSrcVec0 = dd * gammaVec0
        Vector::MulMN<CompType, Vector::GetUsedRegsN<CompType, InType, OutType, ScaleShiftType, float>()>::Do(
            opPG0, opPG1, opPG2, opPG3, diffDst0, diffDst1, diffDst2, diffDst3, gammaVec0, gammaVec1, gammaVec2,
            gammaVec3, dSrcVec0, dSrcVec1, dSrcVec2, dSrcVec3);
        Vector::MulMN<CompType, Vector::GetUsedRegsN<CompType, InType, OutType, ScaleShiftType, float>()>::Do(
            opPG0, opPG1, opPG2, opPG3, dSrcVec0, dSrcVec1, dSrcVec2, dSrcVec3, invSqrtVariance0, invSqrtVariance1,
            invSqrtVariance2, invSqrtVariance3, dSrcVec0, dSrcVec1, dSrcVec2, dSrcVec3); // dSrcVec0 *= invSqrtVariance0
        svbool_t stPG0;
        svbool_t stPG1;
        svbool_t stPG2;
        svbool_t stPG3;
        Vector::InitPGN<InType, Vector::GetUsedRegsN<InType, OutType, CompType, ScaleShiftType, float>()>::Do(
            stPG0, stPG1, stPG2, stPG3, loopIdx, loopRange);
        Vector::StoreN<InType, CompType, Vector::GetUsedRegsN<InType, OutType, CompType, ScaleShiftType, float>()>::Do(
            stPG0, stPG1, stPG2, stPG3, dSrcVec0, dSrcVec1, dSrcVec2, dSrcVec3, diffSrc + loopIdx); // store dSrcVec0
    }
};

static inline void EvaluteAxbSclar(SizeType batchNum, SizeType outerSize, const float *src, const float *mean,
                                   const float *variance, const float *diffDst, const float *scale, float *diffSrc,
                                   float *diffScale, float *diffShift, float eps, NormalizationFlags flags,
                                   SizeType channelSize, SizeType groupSize) noexcept
{
    bool calculateStats = !static_cast<bool>(flags & NormalizationFlags::USE_GLOBAL_STATS);
    bool useScale = static_cast<bool>(flags & NormalizationFlags::USE_SCALE);
    bool useShift = static_cast<bool>(flags & NormalizationFlags::USE_SHIFT);
    SizeType groupNum = channelSize / groupSize;
    SizeType dhw = outerSize / groupSize;
    for (SizeType c = 0; c < channelSize; c++) {
        SizeType groupIdx = c / groupSize;
        float gamma = useScale ? scale[c] : 1.0f;
        float diffGamma = 0;
        float diffBeta = 0;
        for (SizeType b = 0; b < batchNum; b++) {
            float vMean = mean[b * groupNum + groupIdx];
            float invSqrtVariance = 1.0f / std::sqrt(variance[b * groupNum + groupIdx] + eps);
            for (SizeType i = 0; i < dhw; i++) {
                diffGamma += (src[b * channelSize * dhw + i * channelSize + c] - vMean) *
                             diffDst[b * channelSize * dhw + i * channelSize + c] * invSqrtVariance;
                diffBeta += diffDst[b * channelSize * dhw + i * channelSize + c];
            }
        }
        if (useScale) {
            diffScale[c] = diffGamma;
        }
        if (useShift) {
            diffShift[c] = diffBeta;
        }
        for (SizeType b = 0; b < batchNum; b++) {
            float vMean = mean[b * groupNum + groupIdx];
            float invSqrtVariance = 1.0f / std::sqrt(variance[b * groupNum + groupIdx] + eps);
            for (SizeType i = 0; i < dhw; i++) {
                diffSrc[b * channelSize * dhw + i * channelSize + c] =
                    diffDst[b * channelSize * dhw + i * channelSize + c];
                if (calculateStats) {
                    diffSrc[b * channelSize * dhw + i * channelSize + c] -=
                        diffBeta / outerSize + (src[b * channelSize * dhw + i * channelSize + c] - vMean) * diffGamma *
                                                   invSqrtVariance / outerSize;
                }
                diffSrc[b * channelSize * dhw + i * channelSize + c] *= gamma * invSqrtVariance;
            }
        }
    }
}

template <typename CompType, typename InType, typename OutType, typename ScaleShiftType>
void EvaluteAxb(SizeType batchNum, SizeType outerSize, const InType *src, const float *mean, const float *variance,
                const OutType *diffDst, const ScaleShiftType *scale, InType *diffSrc, float *diffScale,
                float *diffShift, float eps, NormalizationFlags flags, SizeType channelSize,
                SizeType groupSize) noexcept
{
    bool calculateStats = !static_cast<bool>(flags & NormalizationFlags::USE_GLOBAL_STATS);
    bool useScale = static_cast<bool>(flags & NormalizationFlags::USE_SCALE);
    bool useShift = static_cast<bool>(flags & NormalizationFlags::USE_SHIFT);
    SizeType groupNum = channelSize / groupSize;
    SizeType dhw = outerSize / groupSize;
    if (calculateStats || useScale || useShift) {
        if constexpr (std::is_same_v<InType, float>) {
            EvaluteAxbSclar(batchNum, outerSize, src, mean, variance, diffDst, scale, diffSrc, diffScale, diffShift,
                            eps, flags, channelSize, groupSize);
        }
    } else {
        for (SizeType g = 0; g < groupNum; g++) {
            CompType diffBeta = 0.0f;
            typename Vector::Vec<CompType>::Type diffBetaVec0;
            typename Vector::Vec<CompType>::Type diffBetaVec1;
            typename Vector::Vec<CompType>::Type diffBetaVec2;
            typename Vector::Vec<CompType>::Type diffBetaVec3;
            Vector::DupFloatN<
                InType, CompType,
                Vector::GetUsedRegsN<InType, CompType, OutType, ScaleShiftType, float>()>::Do(0.0f, diffBetaVec0,
                                                                                              diffBetaVec1,
                                                                                              diffBetaVec2,
                                                                                              diffBetaVec3);
            CompType diffGamma = 0.0f;
            typename Vector::Vec<CompType>::Type diffGammaVec0;
            typename Vector::Vec<CompType>::Type diffGammaVec1;
            typename Vector::Vec<CompType>::Type diffGammaVec2;
            typename Vector::Vec<CompType>::Type diffGammaVec3;
            Vector::DupFloatN<
                InType, CompType,
                Vector::GetUsedRegsN<InType, CompType, OutType, ScaleShiftType, float>()>::Do(0.0f, diffGammaVec0,
                                                                                              diffGammaVec1,
                                                                                              diffGammaVec2,
                                                                                              diffGammaVec3);

            Vector::DupFloatN<
                InType, CompType,
                Vector::GetUsedRegsN<InType, CompType, OutType, ScaleShiftType, float>()>::Do(diffBeta, diffBetaVec0,
                                                                                              diffBetaVec1,
                                                                                              diffBetaVec2,
                                                                                              diffBetaVec3);
            Vector::DupFloatN<
                InType, CompType,
                Vector::GetUsedRegsN<InType, CompType, OutType, ScaleShiftType, float>()>::Do(diffGamma, diffGammaVec0,
                                                                                              diffGammaVec1,
                                                                                              diffGammaVec2,
                                                                                              diffGammaVec3);

            for (SizeType b = 0; b < batchNum; b++) {
                CompType invSqrtVariance = 1.0f / std::sqrt(variance[b * groupNum + g] + eps);
                typename Vector::Vec<CompType>::Type invSqrtVariance0;
                typename Vector::Vec<CompType>::Type invSqrtVariance1;
                typename Vector::Vec<CompType>::Type invSqrtVariance2;
                typename Vector::Vec<CompType>::Type invSqrtVariance3;
                typename Vector::Vec<CompType>::Type mean0;
                typename Vector::Vec<CompType>::Type mean1;
                typename Vector::Vec<CompType>::Type mean2;
                typename Vector::Vec<CompType>::Type mean3;
                Vector::DupFloatN<InType, CompType,
                                  Vector::GetUsedRegsN<InType, CompType, OutType, ScaleShiftType, float>()>::
                    Do(mean[b * groupNum + g], mean0, mean1, mean2, mean3);
                Vector::DupFloatN<InType, CompType,
                                  Vector::GetUsedRegsN<InType, CompType, OutType, ScaleShiftType, float>()>::
                    Do(invSqrtVariance, invSqrtVariance0, invSqrtVariance1, invSqrtVariance2, invSqrtVariance3);
                for (SizeType i = 0; i < dhw; i++) {
                    Service::UnrollFor::Do<1>(groupSize, GnormAxbBWD<CompType, InType, OutType, ScaleShiftType>{},
                                              diffDst + b * channelSize * dhw + g * groupSize + i * channelSize,
                                              invSqrtVariance0, invSqrtVariance1, invSqrtVariance2, invSqrtVariance3,
                                              diffSrc + b * channelSize * dhw + g * groupSize + i * channelSize);
                }
            }
        }
    }
}

} // namespace GNorm
} // namespace Detail
} // namespace KuDNN

#endif //  KUDNN_GNORM_BWD_DEF_HPP
