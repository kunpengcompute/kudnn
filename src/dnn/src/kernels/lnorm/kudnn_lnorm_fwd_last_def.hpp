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

#ifndef KUDNN_LNORM_FWD_LAST_DEF_HPP
#define KUDNN_LNORM_FWD_LAST_DEF_HPP

#include "kudnn.hpp"

#include "vector_wrappers.hpp"
#include "kernels/lnorm/kudnn_lnorm_kernels.hpp"

namespace KuDNN {
namespace Detail {
namespace LNorm {

template <typename CompType, typename InType, typename OutType, typename ScaleShiftType>
struct LayerNormLastFWD final {
    static SizeType LoopInc() noexcept
    {
        return Vector::UnrollLength<InType, CompType, OutType, ScaleShiftType, float>();
    }
    static void Do(SizeType loopIdx, SizeType loopRange, const InType *src, typename Vector::Vec<float>::Type &vMean0,
                   typename Vector::Vec<float>::Type &vMean1, typename Vector::Vec<float>::Type &vMean2,
                   typename Vector::Vec<float>::Type &vMean3) noexcept
    {
        svbool_t ldPG0;
        svbool_t ldPG1;
        svbool_t ldPG2;
        svbool_t ldPG3;
        Vector::InitPGN<InType, Vector::GetUsedRegsN<InType, OutType, CompType, ScaleShiftType, float>()>::Do(
            ldPG0, ldPG1, ldPG2, ldPG3, loopIdx, loopRange);
        typename Vector::Vec<float>::Type data0;
        typename Vector::Vec<float>::Type data1;
        typename Vector::Vec<float>::Type data2;
        typename Vector::Vec<float>::Type data3;
        Vector::LoadN<InType, float, Vector::GetUsedRegsN<InType, OutType, CompType, ScaleShiftType, float>()>::Do(
            src + loopIdx, ldPG0, ldPG1, ldPG2, ldPG3, data0, data1, data2, data3);
        svbool_t opPG0;
        svbool_t opPG1;
        svbool_t opPG2;
        svbool_t opPG3;
        Vector::InitPGN<float, Vector::GetUsedRegsN<CompType, InType, OutType, ScaleShiftType, float>()>::Do(
            opPG0, opPG1, opPG2, opPG3, loopIdx, loopRange);
        Vector::AddMN<float, Vector::GetUsedRegsN<CompType, InType, OutType, ScaleShiftType, float>()>::Do(
            opPG0, opPG1, opPG2, opPG3, vMean0, vMean1, vMean2, vMean3, data0, data1, data2, data3, vMean0, vMean1,
            vMean2, vMean3);
    }
    static void Do(SizeType loopIdx, SizeType loopRange, const InType *src, typename Vector::Vec<float>::Type &vMean0,
                   typename Vector::Vec<float>::Type &vMean1, typename Vector::Vec<float>::Type &vMean2,
                   typename Vector::Vec<float>::Type &vMean3, typename Vector::Vec<float>::Type &vVariance0,
                   typename Vector::Vec<float>::Type &vVariance1, typename Vector::Vec<float>::Type &vVariance2,
                   typename Vector::Vec<float>::Type &vVariance3) noexcept
    {
        svbool_t ldPG0;
        svbool_t ldPG1;
        svbool_t ldPG2;
        svbool_t ldPG3;
        Vector::InitPGN<InType, Vector::GetUsedRegsN<InType, CompType, OutType, ScaleShiftType, float>()>::Do(
            ldPG0, ldPG1, ldPG2, ldPG3, loopIdx, loopRange);
        typename Vector::Vec<float>::Type data0;
        typename Vector::Vec<float>::Type data1;
        typename Vector::Vec<float>::Type data2;
        typename Vector::Vec<float>::Type data3;
        Vector::LoadN<InType, float, Vector::GetUsedRegsN<InType, CompType, OutType, ScaleShiftType, float>()>::Do(
            src + loopIdx, ldPG0, ldPG1, ldPG2, ldPG3, data0, data1, data2, data3);
        svbool_t opPG0;
        svbool_t opPG1;
        svbool_t opPG2;
        svbool_t opPG3;
        Vector::InitPGN<float, Vector::GetUsedRegsN<float, InType, CompType, OutType, ScaleShiftType>()>::Do(
            opPG0, opPG1, opPG2, opPG3, loopIdx, loopRange);
        typename Vector::Vec<float>::Type m0;
        typename Vector::Vec<float>::Type m1;
        typename Vector::Vec<float>::Type m2;
        typename Vector::Vec<float>::Type m3;
        Vector::SubXN<float, Vector::GetUsedRegsN<float, InType, CompType, OutType, ScaleShiftType>()>::Do(
            opPG0, opPG1, opPG2, opPG3, data0, data1, data2, data3, vMean0, vMean1, vMean2, vMean3, m0, m1, m2, m3);
        Vector::MlaMN<float, Vector::GetUsedRegsN<float, InType, CompType, OutType, ScaleShiftType>()>::Do(
            opPG0, opPG1, opPG2, opPG3, vVariance0, vVariance1, vVariance2, vVariance3, m0, m1, m2, m3, m0, m1, m2, m3,
            vVariance0, vVariance1, vVariance2, vVariance3);
    }

    static void DoScale(svbool_t &opPG0, svbool_t &opPG1, svbool_t &opPG2, svbool_t &opPG3, SizeType loopIdx,
                        SizeType loopRange, const ScaleShiftType *scale,
                        typename Vector::Vec<CompType>::Type &invSqrtVarianceVec0,
                        typename Vector::Vec<CompType>::Type &invSqrtVarianceVec1,
                        typename Vector::Vec<CompType>::Type &invSqrtVarianceVec2,
                        typename Vector::Vec<CompType>::Type &invSqrtVarianceVec3,
                        typename Vector::Vec<CompType>::Type &constants0,
                        typename Vector::Vec<CompType>::Type &constants1,
                        typename Vector::Vec<CompType>::Type &constants2,
                        typename Vector::Vec<CompType>::Type &constants3) noexcept
    {
        typename Vector::Vec<CompType>::Type scaleVal0;
        typename Vector::Vec<CompType>::Type scaleVal1;
        typename Vector::Vec<CompType>::Type scaleVal2;
        typename Vector::Vec<CompType>::Type scaleVal3;
        svbool_t ldscalepg0;
        svbool_t ldscalepg1;
        svbool_t ldscalepg2;
        svbool_t ldscalepg3;
        Vector::InitPGN<ScaleShiftType, Vector::GetUsedRegsN<ScaleShiftType, CompType, float, InType, OutType>()>::Do(
            ldscalepg0, ldscalepg1, ldscalepg2, ldscalepg3, loopIdx, loopRange);
        Vector::LoadN<ScaleShiftType, CompType,
                      Vector::GetUsedRegsN<ScaleShiftType, CompType, float, InType, OutType>()>::Do(scale + loopIdx,
                                                                                                    ldscalepg0,
                                                                                                    ldscalepg1,
                                                                                                    ldscalepg2,
                                                                                                    ldscalepg3,
                                                                                                    scaleVal0,
                                                                                                    scaleVal1,
                                                                                                    scaleVal2,
                                                                                                    scaleVal3);
        Vector::MulXN<CompType, Vector::GetUsedRegsN<CompType, ScaleShiftType, float, InType, OutType>()>::Do(
            opPG0, opPG1, opPG2, opPG3, scaleVal0, scaleVal1, scaleVal2, scaleVal3, invSqrtVarianceVec0,
            invSqrtVarianceVec1, invSqrtVarianceVec2, invSqrtVarianceVec3, constants0, constants1, constants2,
            constants3);
    }

    static void Do(SizeType loopIdx, SizeType loopRange, const InType *src, const ScaleShiftType *scale,
                   const ScaleShiftType *shift, bool useScale, bool useShift,
                   typename Vector::Vec<CompType>::Type &invSqrtVarianceVec0,
                   typename Vector::Vec<CompType>::Type &invSqrtVarianceVec1,
                   typename Vector::Vec<CompType>::Type &invSqrtVarianceVec2,
                   typename Vector::Vec<CompType>::Type &invSqrtVarianceVec3,
                   typename Vector::Vec<CompType>::Type &constants0, typename Vector::Vec<CompType>::Type &constants1,
                   typename Vector::Vec<CompType>::Type &constants2, typename Vector::Vec<CompType>::Type &constants3,
                   typename Vector::Vec<CompType>::Type &vMean0, typename Vector::Vec<CompType>::Type &vMean1,
                   typename Vector::Vec<CompType>::Type &vMean2, typename Vector::Vec<CompType>::Type &vMean3,
                   OutType *dst) noexcept
    {
        svbool_t opPG0;
        svbool_t opPG1;
        svbool_t opPG2;
        svbool_t opPG3;
        Vector::InitPGN<CompType, Vector::GetUsedRegsN<CompType, float, InType, OutType, ScaleShiftType>()>::Do(
            opPG0, opPG1, opPG2, opPG3, loopIdx, loopRange);
        if (useScale) {
            DoScale(opPG0, opPG1, opPG2, opPG3, loopIdx, loopRange, scale, invSqrtVarianceVec0, invSqrtVarianceVec1,
                    invSqrtVarianceVec2, invSqrtVarianceVec3, constants0, constants1, constants2, constants3);
        }
        typename Vector::Vec<CompType>::Type shiftVal0;
        typename Vector::Vec<CompType>::Type shiftVal1;
        typename Vector::Vec<CompType>::Type shiftVal2;
        typename Vector::Vec<CompType>::Type shiftVal3;
        if (useShift) {
            svbool_t ldshiftpg0;
            svbool_t ldshiftpg1;
            svbool_t ldshiftpg2;
            svbool_t ldshiftpg3;
            Vector::InitPGN<ScaleShiftType,
                            Vector::GetUsedRegsN<ScaleShiftType, CompType, float, InType, OutType>()>::Do(ldshiftpg0,
                                                                                                          ldshiftpg1,
                                                                                                          ldshiftpg2,
                                                                                                          ldshiftpg3,
                                                                                                          loopIdx,
                                                                                                          loopRange);
            Vector::LoadN<ScaleShiftType, CompType,
                          Vector::GetUsedRegsN<ScaleShiftType, CompType, float, InType, OutType>()>::Do(shift + loopIdx,
                                                                                                        ldshiftpg0,
                                                                                                        ldshiftpg1,
                                                                                                        ldshiftpg2,
                                                                                                        ldshiftpg3,
                                                                                                        shiftVal0,
                                                                                                        shiftVal1,
                                                                                                        shiftVal2,
                                                                                                        shiftVal3);
        } else {
            Vector::DupFloatN<ScaleShiftType, CompType,
                              Vector::GetUsedRegsN<ScaleShiftType, CompType, float, InType, OutType>()>::Do(0.0f,
                                                                                                            shiftVal0,
                                                                                                            shiftVal1,
                                                                                                            shiftVal2,
                                                                                                            shiftVal3);
        }
        svbool_t ldPG0;
        svbool_t ldPG1;
        svbool_t ldPG2;
        svbool_t ldPG3;
        Vector::InitPGN<InType, Vector::GetUsedRegsN<InType, ScaleShiftType, CompType, float, OutType>()>::Do(
            ldPG0, ldPG1, ldPG2, ldPG3, loopIdx, loopRange);
        typename Vector::Vec<CompType>::Type data0;
        typename Vector::Vec<CompType>::Type data1;
        typename Vector::Vec<CompType>::Type data2;
        typename Vector::Vec<CompType>::Type data3;
        Vector::LoadN<InType, CompType, Vector::GetUsedRegsN<InType, ScaleShiftType, CompType, float, OutType>()>::Do(
            src + loopIdx, ldPG0, ldPG1, ldPG2, ldPG3, data0, data1, data2, data3);
        Vector::SubXN<CompType, Vector::GetUsedRegsN<CompType, InType, ScaleShiftType, float, OutType>()>::Do(
            opPG0, opPG1, opPG2, opPG3, data0, data1, data2, data3, vMean0, vMean1, vMean2, vMean3, data0, data1, data2,
            data3);
        Vector::MlaXN<CompType, Vector::GetUsedRegsN<CompType, InType, ScaleShiftType, float, OutType>()>::Do(
            opPG0, opPG1, opPG2, opPG3, shiftVal0, shiftVal1, shiftVal2, shiftVal3, constants0, constants1, constants2,
            constants3, data0, data1, data2, data3, data0, data1, data2, data3);
        svbool_t stPG0;
        svbool_t stPG1;
        svbool_t stPG2;
        svbool_t stPG3;
        Vector::InitPGN<OutType, Vector::GetUsedRegsN<OutType, CompType, InType, ScaleShiftType, float>()>::Do(
            stPG0, stPG1, stPG2, stPG3, loopIdx, loopRange);
        Vector::StoreN<OutType, CompType, Vector::GetUsedRegsN<OutType, CompType, InType, ScaleShiftType, float>()>::Do(
            stPG0, stPG1, stPG2, stPG3, data0, data1, data2, data3, dst + loopIdx);
    }
};

template <typename CompType, typename InType, typename OutType, typename ScaleShiftType>
void EvaluateLastAxis(SizeType execAcrossAxis, const InType *src, OutType *dst, const ScaleShiftType *scale,
                      const ScaleShiftType *shift, float *mean, float *variance, bool saveStats, float eps,
                      NormalizationFlags flags, SizeType axisSize) noexcept
{
    bool calculateStats = !static_cast<bool>(flags & NormalizationFlags::USE_GLOBAL_STATS);
    bool useScale = static_cast<bool>(flags & NormalizationFlags::USE_SCALE);
    bool useShift = static_cast<bool>(flags & NormalizationFlags::USE_SHIFT);
    for (SizeType n = 0; n < execAcrossAxis; ++n) {
        float vMean = calculateStats ? 0.0f : mean[n];
        typename Vector::Vec<float>::Type vMean0;
        typename Vector::Vec<float>::Type vMean1;
        typename Vector::Vec<float>::Type vMean2;
        typename Vector::Vec<float>::Type vMean3;
        Vector::DupFloatN<InType, float, Vector::GetUsedRegsN<InType, OutType, CompType, ScaleShiftType, float>()>::Do(
            vMean, vMean0, vMean1, vMean2, vMean3);
        float vVariance = calculateStats ? 0.0f : variance[n];
        svbool_t pgTrue = svptrue_b32();
        typename Vector::Vec<float>::Type vVariance0;
        typename Vector::Vec<float>::Type vVariance1;
        typename Vector::Vec<float>::Type vVariance2;
        typename Vector::Vec<float>::Type vVariance3;
        Vector::DupFloatN<InType, float, Vector::GetUsedRegsN<InType, OutType, CompType, ScaleShiftType, float>()>::Do(
            vVariance, vVariance0, vVariance1, vVariance2, vVariance3);
        if (calculateStats) {
            Service::UnrollFor::Do<1>(axisSize, LayerNormLastFWD<CompType, InType, OutType, ScaleShiftType>{},
                                      src + n * axisSize, vMean0, vMean1, vMean2, vMean3);
            vMean = Vector::AddvN<float, Vector::GetUsedRegsN<float, InType, OutType, CompType, ScaleShiftType>()>::Do(
                pgTrue, pgTrue, pgTrue, pgTrue, vMean0, vMean1, vMean2, vMean3);
            vMean /= axisSize;
            Vector::DupFloatN<InType, float,
                              Vector::GetUsedRegsN<InType, OutType, CompType, ScaleShiftType, float>()>::Do(vMean,
                                                                                                            vMean0,
                                                                                                            vMean1,
                                                                                                            vMean2,
                                                                                                            vMean3);
            Service::UnrollFor::Do<1>(axisSize, LayerNormLastFWD<CompType, InType, OutType, ScaleShiftType>{},
                                      src + n * axisSize, vMean0, vMean1, vMean2, vMean3, vVariance0, vVariance1,
                                      vVariance2, vVariance3);
            vVariance =
                Vector::AddvN<float, Vector::GetUsedRegsN<float, InType, OutType, CompType, ScaleShiftType>()>::Do(
                    pgTrue, pgTrue, pgTrue, pgTrue, vVariance0, vVariance1, vVariance2, vVariance3);
            vVariance /= axisSize;
            if (saveStats) {
                mean[n] = vMean;
                variance[n] = vVariance;
            }
        }
        float invSqrtVariance = 1.0f / std::sqrt(vVariance + eps);
        typename Vector::Vec<CompType>::Type invSqrtVarianceVec0;
        typename Vector::Vec<CompType>::Type invSqrtVarianceVec1;
        typename Vector::Vec<CompType>::Type invSqrtVarianceVec2;
        typename Vector::Vec<CompType>::Type invSqrtVarianceVec3;
        Vector::DupFloatN<InType, CompType, Vector::GetUsedRegsN<InType, OutType, CompType, ScaleShiftType, float>()>::
            Do(invSqrtVariance, invSqrtVarianceVec0, invSqrtVarianceVec1, invSqrtVarianceVec2, invSqrtVarianceVec3);
        typename Vector::Vec<CompType>::Type constants0;
        typename Vector::Vec<CompType>::Type constants1;
        typename Vector::Vec<CompType>::Type constants2;
        typename Vector::Vec<CompType>::Type constants3;
        Vector::DupFloatN<InType, CompType,
                          Vector::GetUsedRegsN<InType, OutType, CompType, ScaleShiftType, float>()>::Do(invSqrtVariance,
                                                                                                        constants0,
                                                                                                        constants1,
                                                                                                        constants2,
                                                                                                        constants3);
        typename Vector::Vec<CompType>::Type vMeanVect0;
        typename Vector::Vec<CompType>::Type vMeanVect1;
        typename Vector::Vec<CompType>::Type vMeanVect2;
        typename Vector::Vec<CompType>::Type vMeanVect3;
        Vector::DupFloatN<InType, CompType,
                          Vector::GetUsedRegsN<InType, OutType, CompType, ScaleShiftType, float>()>::Do(vMean,
                                                                                                        vMeanVect0,
                                                                                                        vMeanVect1,
                                                                                                        vMeanVect2,
                                                                                                        vMeanVect3);
        Service::UnrollFor::Do<1>(axisSize, LayerNormLastFWD<CompType, InType, OutType, ScaleShiftType>{},
                                  src + n * axisSize, scale, shift, useScale, useShift, invSqrtVarianceVec0,
                                  invSqrtVarianceVec1, invSqrtVarianceVec2, invSqrtVarianceVec3, constants0, constants1,
                                  constants2, constants3, vMeanVect0, vMeanVect1, vMeanVect2, vMeanVect3,
                                  dst + n * axisSize);
    }
}

} // namespace LNorm
} // namespace Detail
} // namespace KuDNN

#endif //  KUDNN_LNORM_FWD_LAST_DEF_HPP
