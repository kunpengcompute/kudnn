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

#ifndef KUDNN_GNORM_FWD_DEF_HPP
#define KUDNN_GNORM_FWD_DEF_HPP

#include "kudnn.hpp"

#include "vector_wrappers.hpp"
#include "kernels/gnorm/kudnn_gnorm_kernels.hpp"

namespace KuDNN {
namespace Detail {
namespace GNorm {

template <typename CompType, typename InType, typename OutType, typename ScaleShiftType>
struct GroupNormCrossFWD final {
    static SizeType LoopInc() noexcept
    {
        return Vector::UnrollLength<InType, CompType, OutType, ScaleShiftType, float>();
    }

    // for mean calculation. Mean is computed in float precision
    static void Do(SizeType loopIdx, SizeType loopRange, const InType *src, typename Vector::Vec<float>::Type &vMean0,
                   typename Vector::Vec<float>::Type &vMean1, typename Vector::Vec<float>::Type &vMean2,
                   typename Vector::Vec<float>::Type &vMean3)
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

    // for variance calculation. Variance is computed in float precision
    static void Do(SizeType loopIdx, SizeType loopRange, const InType *src, typename Vector::Vec<float>::Type &vMean0,
                   typename Vector::Vec<float>::Type &vMean1, typename Vector::Vec<float>::Type &vMean2,
                   typename Vector::Vec<float>::Type &vMean3, typename Vector::Vec<float>::Type &vVariance0,
                   typename Vector::Vec<float>::Type &vVariance1, typename Vector::Vec<float>::Type &vVariance2,
                   typename Vector::Vec<float>::Type &vVariance3)
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

    static void DoScale(svbool_t &opPG0, svbool_t &opPG1, svbool_t &opPG2, svbool_t &opPG3, const ScaleShiftType *scale,
                        SizeType scaleshiftIdx, typename Vector::Vec<CompType>::Type &invSqrtVarianceVec0,
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
        Vector::DupN<ScaleShiftType, CompType,
                     Vector::GetUsedRegsN<ScaleShiftType, float, InType, CompType, OutType>()>::Do(scale[scaleshiftIdx],
                                                                                                   scaleVal0, scaleVal1,
                                                                                                   scaleVal2,
                                                                                                   scaleVal3);
        Vector::MulXN<CompType, Vector::GetUsedRegsN<CompType, ScaleShiftType, float, InType, OutType>()>::Do(
            opPG0, opPG1, opPG2, opPG3, scaleVal0, scaleVal1, scaleVal2, scaleVal3, invSqrtVarianceVec0,
            invSqrtVarianceVec1, invSqrtVarianceVec2, invSqrtVarianceVec3, constants0, constants1, constants2,
            constants3);
    }

    static void Do(SizeType loopIdx, SizeType loopRange, const InType *src, const ScaleShiftType *scale,
                   const ScaleShiftType *shift, bool useScale, bool useShift, SizeType innerSize, SizeType group,
                   SizeType channelPerGroup, SizeType channel,
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
        Service::Unused(innerSize);
        svbool_t opPG0;
        svbool_t opPG1;
        svbool_t opPG2;
        svbool_t opPG3;
        Vector::InitPGN<CompType, Vector::GetUsedRegsN<CompType, float, InType, OutType, ScaleShiftType>()>::Do(
            opPG0, opPG1, opPG2, opPG3, loopIdx, loopRange);
        SizeType scaleshiftIdx = channel + group * channelPerGroup;
        if (useScale) {
            DoScale(opPG0, opPG1, opPG2, opPG3, scale, scaleshiftIdx, invSqrtVarianceVec0, invSqrtVarianceVec1,
                    invSqrtVarianceVec2, invSqrtVarianceVec3, constants0, constants1, constants2, constants3);
        }

        typename Vector::Vec<CompType>::Type shiftVal0;
        typename Vector::Vec<CompType>::Type shiftVal1;
        typename Vector::Vec<CompType>::Type shiftVal2;
        typename Vector::Vec<CompType>::Type shiftVal3;
        if (useShift) {
            Vector::DupN<
                ScaleShiftType, CompType,
                Vector::GetUsedRegsN<ScaleShiftType, float, InType, CompType, OutType>()>::Do(shift[scaleshiftIdx],
                                                                                              shiftVal0, shiftVal1,
                                                                                              shiftVal2, shiftVal3);
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
void EvaluateCrossAxis(SizeType outerSize, const InType *src, OutType *dst, SizeType groupSize,
                       const ScaleShiftType *scale, const ScaleShiftType *shift, float *mean, float *variance,
                       bool saveStats, float eps, NormalizationFlags flags, SizeType axisSize,
                       SizeType innerSize) noexcept
{
    bool calculateStats = !static_cast<bool>(flags & NormalizationFlags::USE_GLOBAL_STATS);
    bool useScale = static_cast<bool>(flags & NormalizationFlags::USE_SCALE);
    bool useShift = static_cast<bool>(flags & NormalizationFlags::USE_SHIFT);
    SizeType channelPerGroup = axisSize / groupSize;
    SizeType dataPerGroup = innerSize * channelPerGroup;
    typename Vector::Vec<float>::Type dataPerGroupVec0;
    typename Vector::Vec<float>::Type dataPerGroupVec1;
    typename Vector::Vec<float>::Type dataPerGroupVec2;
    typename Vector::Vec<float>::Type dataPerGroupVec3;
    Vector::DupFloatN<InType, float, Vector::GetUsedRegsN<InType, CompType, OutType, ScaleShiftType, float>()>::Do(
        dataPerGroup, dataPerGroupVec0, dataPerGroupVec1, dataPerGroupVec2, dataPerGroupVec3);
    for (SizeType ou = 0; ou < outerSize; ++ou) {
        for (SizeType group = 0; group < groupSize; ++group) {
            SizeType offset = ou * axisSize * innerSize + group * channelPerGroup * innerSize;
            SizeType offsetStats = ou * groupSize + group;
            typename Vector::Vec<float>::Type vMean0;
            typename Vector::Vec<float>::Type vMean1;
            typename Vector::Vec<float>::Type vMean2;
            typename Vector::Vec<float>::Type vMean3;
            typename Vector::Vec<float>::Type vVariance0;
            typename Vector::Vec<float>::Type vVariance1;
            typename Vector::Vec<float>::Type vVariance2;
            typename Vector::Vec<float>::Type vVariance3;

            float vMean = calculateStats ? 0.0f : mean[offsetStats];
            float vVariance = calculateStats ? 0.0f : variance[offsetStats];
            Vector::DupFloatN<InType, float,
                              Vector::GetUsedRegsN<InType, OutType, CompType, ScaleShiftType, float>()>::Do(vMean,
                                                                                                            vMean0,
                                                                                                            vMean1,
                                                                                                            vMean2,
                                                                                                            vMean3);
            Vector::DupFloatN<InType, float,
                              Vector::GetUsedRegsN<InType, OutType, CompType, ScaleShiftType, float>()>::Do(vVariance,
                                                                                                            vVariance0,
                                                                                                            vVariance1,
                                                                                                            vVariance2,
                                                                                                            vVariance3);

            if (calculateStats) {
                svbool_t pgTrue = svptrue_b32();
                Service::UnrollFor::Do<1>(innerSize * channelPerGroup,
                                          GroupNormCrossFWD<CompType, InType, OutType, ScaleShiftType>{}, src + offset,
                                          vMean0, vMean1, vMean2, vMean3);
                vMean =
                    Vector::AddvN<float, Vector::GetUsedRegsN<float, InType, OutType, CompType, ScaleShiftType>()>::Do(
                        pgTrue, pgTrue, pgTrue, pgTrue, vMean0, vMean1, vMean2, vMean3);
                vMean /= dataPerGroup;
                Vector::DupFloatN<InType, float,
                                  Vector::GetUsedRegsN<InType, OutType, CompType, ScaleShiftType, float>()>::Do(vMean,
                                                                                                                vMean0,
                                                                                                                vMean1,
                                                                                                                vMean2,
                                                                                                                vMean3);
                Service::UnrollFor::Do<1>(
                    innerSize * channelPerGroup, GroupNormCrossFWD<CompType, InType, OutType, ScaleShiftType>{},
                    src + offset, vMean0, vMean1, vMean2, vMean3, vVariance0, vVariance1, vVariance2, vVariance3);
                vVariance =
                    Vector::AddvN<float, Vector::GetUsedRegsN<float, InType, OutType, CompType, ScaleShiftType>()>::Do(
                        pgTrue, pgTrue, pgTrue, pgTrue, vVariance0, vVariance1, vVariance2, vVariance3);
                vVariance /= dataPerGroup;
                if (saveStats) {
                    mean[offsetStats] = vMean;
                    variance[offsetStats] = vVariance;
                }
            }

            float invSqrtVariance = 1.0f / std::sqrt(vVariance + eps);
            typename Vector::Vec<CompType>::Type invSqrtVarianceVec0;
            typename Vector::Vec<CompType>::Type invSqrtVarianceVec1;
            typename Vector::Vec<CompType>::Type invSqrtVarianceVec2;
            typename Vector::Vec<CompType>::Type invSqrtVarianceVec3;
            Vector::DupFloatN<InType, CompType,
                              Vector::GetUsedRegsN<InType, OutType, CompType, ScaleShiftType, float>()>::
                Do(invSqrtVariance, invSqrtVarianceVec0, invSqrtVarianceVec1, invSqrtVarianceVec2, invSqrtVarianceVec3);
            typename Vector::Vec<CompType>::Type constants0;
            typename Vector::Vec<CompType>::Type constants1;
            typename Vector::Vec<CompType>::Type constants2;
            typename Vector::Vec<CompType>::Type constants3;
            Vector::DupFloatN<
                InType, CompType,
                Vector::GetUsedRegsN<InType, OutType, CompType, ScaleShiftType, float>()>::Do(invSqrtVariance,
                                                                                              constants0, constants1,
                                                                                              constants2, constants3);
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

            for (SizeType c = 0; c < channelPerGroup; ++c) {
                Service::UnrollFor::Do<1>(innerSize, GroupNormCrossFWD<CompType, InType, OutType, ScaleShiftType>{},
                                          src + offset + c * innerSize, scale, shift, useScale, useShift, innerSize,
                                          group, channelPerGroup, c, invSqrtVarianceVec0, invSqrtVarianceVec1,
                                          invSqrtVarianceVec2, invSqrtVarianceVec3, constants0, constants1, constants2,
                                          constants3, vMeanVect0, vMeanVect1, vMeanVect2, vMeanVect3,
                                          dst + offset + c * innerSize);
            }
        }
    }
}

template <typename CompType, typename InType, typename OutType, typename ScaleShiftType>
struct GroupNormLastFWD final {
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
                        SizeType loopRange, const ScaleShiftType *scale, SizeType g,
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
                      Vector::GetUsedRegsN<ScaleShiftType, CompType, float, InType, OutType>()>::Do(scale + loopIdx +
                                                                                                        g * loopRange,
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

    static void DoShift(SizeType loopIdx, SizeType loopRange, const ScaleShiftType *shift, SizeType g,
                        typename Vector::Vec<CompType>::Type &shiftVal0,
                        typename Vector::Vec<CompType>::Type &shiftVal1,
                        typename Vector::Vec<CompType>::Type &shiftVal2,
                        typename Vector::Vec<CompType>::Type &shiftVal3) noexcept
    {
        svbool_t ldshiftpg0;
        svbool_t ldshiftpg1;
        svbool_t ldshiftpg2;
        svbool_t ldshiftpg3;
        Vector::InitPGN<ScaleShiftType, Vector::GetUsedRegsN<ScaleShiftType, CompType, float, InType, OutType>()>::Do(
            ldshiftpg0, ldshiftpg1, ldshiftpg2, ldshiftpg3, loopIdx, loopRange);
        Vector::LoadN<ScaleShiftType, CompType,
                      Vector::GetUsedRegsN<ScaleShiftType, CompType, float, InType, OutType>()>::Do(shift + loopIdx +
                                                                                                        g * loopRange,
                                                                                                    ldshiftpg0,
                                                                                                    ldshiftpg1,
                                                                                                    ldshiftpg2,
                                                                                                    ldshiftpg3,
                                                                                                    shiftVal0,
                                                                                                    shiftVal1,
                                                                                                    shiftVal2,
                                                                                                    shiftVal3);
    }

    static void Do(SizeType loopIdx, SizeType loopRange, const InType *src, const ScaleShiftType *scale,
                   const ScaleShiftType *shift, bool useScale, bool useShift, SizeType g,
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
            DoScale(opPG0, opPG1, opPG2, opPG3, loopIdx, loopRange, scale, g, invSqrtVarianceVec0, invSqrtVarianceVec1,
                    invSqrtVarianceVec2, invSqrtVarianceVec3, constants0, constants1, constants2, constants3);
        }
        typename Vector::Vec<CompType>::Type shiftVal0;
        typename Vector::Vec<CompType>::Type shiftVal1;
        typename Vector::Vec<CompType>::Type shiftVal2;
        typename Vector::Vec<CompType>::Type shiftVal3;
        if (useShift) {
            DoShift(loopIdx, loopRange, shift, g, shiftVal0, shiftVal1, shiftVal2, shiftVal3);
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
void EvaluateLastAxis(SizeType outerSize, SizeType batchSize, const InType *src, OutType *dst, SizeType groupSize,
                      const ScaleShiftType *scale, const ScaleShiftType *shift, float *mean, float *variance,
                      bool saveStats, float eps, NormalizationFlags flags, SizeType axisSize) noexcept
{
    bool calculateStats = !static_cast<bool>(flags & NormalizationFlags::USE_GLOBAL_STATS);
    bool useScale = static_cast<bool>(flags & NormalizationFlags::USE_SCALE);
    bool useShift = static_cast<bool>(flags & NormalizationFlags::USE_SHIFT);
    SizeType dhw = outerSize / batchSize;
    SizeType channelPerGroup = axisSize / groupSize;
    SizeType dataPerGroup = dhw * channelPerGroup;
    for (SizeType a = 0; a < batchSize; ++a) {
        for (SizeType g = 0; g < groupSize; ++g) {
            SizeType offsetStats = a * groupSize + g;
            float vMean = calculateStats ? 0.0f : mean[offsetStats];
            typename Vector::Vec<float>::Type vMean0;
            typename Vector::Vec<float>::Type vMean1;
            typename Vector::Vec<float>::Type vMean2;
            typename Vector::Vec<float>::Type vMean3;
            Vector::DupFloatN<InType, float,
                              Vector::GetUsedRegsN<InType, OutType, CompType, ScaleShiftType, float>()>::Do(vMean,
                                                                                                            vMean0,
                                                                                                            vMean1,
                                                                                                            vMean2,
                                                                                                            vMean3);
            float vVariance = calculateStats ? 0.0f : variance[offsetStats];
            svbool_t pgTrue = svptrue_b32();
            typename Vector::Vec<float>::Type vVariance0;
            typename Vector::Vec<float>::Type vVariance1;
            typename Vector::Vec<float>::Type vVariance2;
            typename Vector::Vec<float>::Type vVariance3;
            Vector::DupFloatN<InType, float,
                              Vector::GetUsedRegsN<InType, OutType, CompType, ScaleShiftType, float>()>::Do(vVariance,
                                                                                                            vVariance0,
                                                                                                            vVariance1,
                                                                                                            vVariance2,
                                                                                                            vVariance3);

            if (calculateStats) {
                for (SizeType n = 0; n < dhw; ++n) {
                    SizeType offset = a * dhw * axisSize + n * axisSize + g * channelPerGroup;
                    Service::UnrollFor::Do<1>(channelPerGroup,
                                              GroupNormLastFWD<CompType, InType, OutType, ScaleShiftType>{},
                                              src + offset, vMean0, vMean1, vMean2, vMean3);
                    float tmpMean =
                        Vector::AddvN<float, Vector::GetUsedRegsN<float, InType, OutType, CompType, ScaleShiftType>()>::
                            Do(pgTrue, pgTrue, pgTrue, pgTrue, vMean0, vMean1, vMean2, vMean3);
                    vMean += tmpMean;
                    Vector::DupFloatN<
                        InType, float,
                        Vector::GetUsedRegsN<InType, OutType, CompType, ScaleShiftType, float>()>::Do(0.0f, vMean0,
                                                                                                      vMean1, vMean2,
                                                                                                      vMean3);
                }
                vMean /= dataPerGroup;

                Vector::DupFloatN<InType, float,
                                  Vector::GetUsedRegsN<InType, OutType, CompType, ScaleShiftType, float>()>::Do(vMean,
                                                                                                                vMean0,
                                                                                                                vMean1,
                                                                                                                vMean2,
                                                                                                                vMean3);

                for (SizeType n = 0; n < dhw; ++n) {
                    SizeType offset = a * dhw * axisSize + n * axisSize + g * channelPerGroup;
                    Service::UnrollFor::Do<1>(
                        channelPerGroup, GroupNormLastFWD<CompType, InType, OutType, ScaleShiftType>{}, src + offset,
                        vMean0, vMean1, vMean2, vMean3, vVariance0, vVariance1, vVariance2, vVariance3);
                    float tmpVariance =
                        Vector::AddvN<float, Vector::GetUsedRegsN<float, InType, OutType, CompType, ScaleShiftType>()>::
                            Do(pgTrue, pgTrue, pgTrue, pgTrue, vVariance0, vVariance1, vVariance2, vVariance3);
                    vVariance += tmpVariance;
                    Vector::DupFloatN<
                        InType, float,
                        Vector::GetUsedRegsN<InType, OutType, CompType, ScaleShiftType, float>()>::Do(0.0f, vVariance0,
                                                                                                      vVariance1,
                                                                                                      vVariance2,
                                                                                                      vVariance3);
                }
                vVariance /= dataPerGroup;
            }

            if (saveStats) {
                mean[offsetStats] = vMean;
                variance[offsetStats] = vVariance;
            }
            float invSqrtVariance = 1.0f / std::sqrt(vVariance + eps);
            typename Vector::Vec<CompType>::Type invSqrtVarianceVec0;
            typename Vector::Vec<CompType>::Type invSqrtVarianceVec1;
            typename Vector::Vec<CompType>::Type invSqrtVarianceVec2;
            typename Vector::Vec<CompType>::Type invSqrtVarianceVec3;
            Vector::DupFloatN<InType, CompType,
                              Vector::GetUsedRegsN<InType, OutType, CompType, ScaleShiftType, float>()>::
                Do(invSqrtVariance, invSqrtVarianceVec0, invSqrtVarianceVec1, invSqrtVarianceVec2, invSqrtVarianceVec3);
            typename Vector::Vec<CompType>::Type constants0;
            typename Vector::Vec<CompType>::Type constants1;
            typename Vector::Vec<CompType>::Type constants2;
            typename Vector::Vec<CompType>::Type constants3;
            Vector::DupFloatN<
                InType, CompType,
                Vector::GetUsedRegsN<InType, OutType, CompType, ScaleShiftType, float>()>::Do(invSqrtVariance,
                                                                                              constants0, constants1,
                                                                                              constants2, constants3);
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
            for (SizeType n = 0; n < dhw; ++n) {
                SizeType offset = a * dhw * axisSize + n * axisSize + g * channelPerGroup;
                Service::UnrollFor::Do<1>(channelPerGroup,
                                          GroupNormLastFWD<CompType, InType, OutType, ScaleShiftType>{}, src + offset,
                                          scale, shift, useScale, useShift, g, invSqrtVarianceVec0, invSqrtVarianceVec1,
                                          invSqrtVarianceVec2, invSqrtVarianceVec3, constants0, constants1, constants2,
                                          constants3, vMeanVect0, vMeanVect1, vMeanVect2, vMeanVect3, dst + offset);
            }
        }
    }
}

} // namespace GNorm
} // namespace Detail
} // namespace KuDNN

#endif //  KUDNN_GNORM_FWD_DEF_HPP
