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

#ifndef KUDNN_LNORM_BWD_LAST_DEF_HPP
#define KUDNN_LNORM_BWD_LAST_DEF_HPP

#include "kudnn.hpp"

#include "vector_wrappers.hpp"
#include "kernels/lnorm/kudnn_lnorm_kernels.hpp"
#include "kudnn_lnorm_bwd_last_layer_def.hpp"

namespace KuDNN {
namespace Detail {
namespace LNorm {

template <typename CompType, typename InType, typename OutType, typename ScaleShiftType>
void EvaluateRewriteScaleShift(ScaleShiftType *diffScale,
    ScaleShiftType *diffShift, float *bufScale, float *bufShift, NormalizationFlags flags,
    SizeType axisSize, SizeType numThreads) noexcept
{
    bool useScale = static_cast<bool>(flags & NormalizationFlags::USE_SCALE);
    bool useShift = static_cast<bool>(flags & NormalizationFlags::USE_SHIFT);
    if (useScale) {
        for (SizeType i = 0; i < axisSize; ++i) {
            float tmp = 0.0f;
            for (SizeType thread = 0; thread < numThreads; ++thread) {
                tmp += bufScale[i + axisSize * thread];
            }
            diffScale[i] = Vector::Convert<float, ScaleShiftType>::Do(tmp);
        }
    }

    if (useShift) {
        for (SizeType i = 0; i < axisSize; ++i) {
            float tmp = 0.0f;
            for (SizeType thread = 0; thread < numThreads; ++thread) {
                tmp += bufShift[i + axisSize * thread];
            }
            diffShift[i] = Vector::Convert<float, ScaleShiftType>::Do(tmp);
        }
    }
}

template <typename CompType, typename InType, typename OutType, typename ScaleShiftType>
void EvaluateScaleShiftLastAxis(SizeType outerSize, const InType *src, const float *mean, const float *variance,
    const OutType *diffDst, ScaleShiftType *diffScale,
    ScaleShiftType *diffShift, float *bufScale, float *bufShift, float eps, NormalizationFlags flags,
    SizeType axisSize, SizeType axisSizeConst) noexcept
{
    bool useScale = static_cast<bool>(flags & NormalizationFlags::USE_SCALE);
    bool useShift = static_cast<bool>(flags & NormalizationFlags::USE_SHIFT);
    bool skipMean = static_cast<bool>(flags & NormalizationFlags::RMS_NORM);
    Service::Unused(axisSize, diffScale, diffShift);
    {
        if (useScale || useShift) {
            for (SizeType ou = 0; ou < outerSize; ++ou) {
                float var = variance[ou];
                float m = skipMean ? 0.0f : mean[ou];
                float denom = std::sqrt(var + eps);
                float invSqrtVar = 1.0f / denom;
                typename Vector::Vec<CompType>::Type mean0;
                typename Vector::Vec<CompType>::Type mean1;
                typename Vector::Vec<CompType>::Type mean2;
                typename Vector::Vec<CompType>::Type mean3;
                Vector::DupFloatN<InType, CompType,
                    Vector::GetUsedRegsN<InType, CompType, OutType, ScaleShiftType, float>()>::Do(
                    m, mean0, mean1, mean2, mean3);
                typename Vector::Vec<CompType>::Type invSqrtVar0;
                typename Vector::Vec<CompType>::Type invSqrtVar1;
                typename Vector::Vec<CompType>::Type invSqrtVar2;
                typename Vector::Vec<CompType>::Type invSqrtVar3;
                Vector::DupFloatN<InType, CompType,
                    Vector::GetUsedRegsN<InType, CompType, OutType, ScaleShiftType, float>()>::Do(
                    invSqrtVar, invSqrtVar0, invSqrtVar1, invSqrtVar2, invSqrtVar3);
                Service::UnrollFor::Do<1>(axisSizeConst, LayerNormLastBWD<CompType, InType, OutType, ScaleShiftType>{},
                    src + ou * axisSizeConst, diffDst + ou * axisSizeConst, useScale, useShift, bufScale, bufShift,
                    mean0, mean1, mean2, mean3, invSqrtVar0, invSqrtVar1, invSqrtVar2, invSqrtVar3);
            }
        }
    }
}

template <typename CompType, typename InType, typename OutType, typename ScaleShiftType>
static void EvaluateLastAxisCalcStat(const InType *src, const OutType *diffDst, const ScaleShiftType *scale, SizeType n,
    SizeType axisSize, CompType &ddGammaX, CompType &ddGamma, svbool_t &pgTrue,
    CompType &invSqrtVariance, bool useScale,
    typename Vector::Vec<CompType>::Type &mean0, typename Vector::Vec<CompType>::Type &mean1,
    typename Vector::Vec<CompType>::Type &mean2, typename Vector::Vec<CompType>::Type &mean3,
    typename Vector::Vec<CompType>::Type &ddGammaVec0, typename Vector::Vec<CompType>::Type &ddGammaVec1,
    typename Vector::Vec<CompType>::Type &ddGammaVec2, typename Vector::Vec<CompType>::Type &ddGammaVec3,
    typename Vector::Vec<CompType>::Type &ddGammaXVec0, typename Vector::Vec<CompType>::Type &ddGammaXVec1,
    typename Vector::Vec<CompType>::Type &ddGammaXVec2, typename Vector::Vec<CompType>::Type &ddGammaXVec3) noexcept
{
    Service::UnrollFor::Do<1>(axisSize, LayerNormLastBWD<CompType, InType, OutType, ScaleShiftType>{},
        src + n * axisSize, diffDst + n * axisSize, scale, useScale,
        mean0, mean1, mean2, mean3, ddGammaVec0, ddGammaVec1, ddGammaVec2, ddGammaVec3,
        ddGammaXVec0, ddGammaXVec1, ddGammaXVec2, ddGammaXVec3);

    ddGamma = Vector::AddvN<CompType, Vector::GetUsedRegsN<CompType, InType, OutType, ScaleShiftType, float>()>::Do(
        pgTrue, pgTrue, pgTrue, pgTrue, ddGammaVec0, ddGammaVec1, ddGammaVec2, ddGammaVec3);
    ddGammaX = Vector::AddvN<CompType, Vector::GetUsedRegsN<CompType, InType, OutType, ScaleShiftType, float>()>::Do(
        pgTrue, pgTrue, pgTrue, pgTrue, ddGammaXVec0, ddGammaXVec1, ddGammaXVec2, ddGammaXVec3);
    ddGammaX *= invSqrtVariance;
}

template <typename CompType, typename InType, typename OutType, typename ScaleShiftType>
void EvaluateLastAxis(SizeType outerSize, const InType *src, const float *mean, const float *variance,
    const OutType *diffDst, const ScaleShiftType *scale, InType *diffSrc, float eps,
    NormalizationFlags flags, SizeType axisSize) noexcept
{
    bool calculateStats = !static_cast<bool>(flags & NormalizationFlags::USE_GLOBAL_STATS);
    bool useScale = static_cast<bool>(flags & NormalizationFlags::USE_SCALE);
    bool skipMean = static_cast<bool>(flags & NormalizationFlags::RMS_NORM);
    svbool_t pgTrue = svptrue_b32();
    for (SizeType n = 0; n < outerSize; ++n) {
        CompType invSqrtVariance = 1.0f / std::sqrt(variance[n] + eps);
        CompType ddGamma = 0.0f;
        typename Vector::Vec<CompType>::Type ddGammaVec0;
        typename Vector::Vec<CompType>::Type ddGammaVec1;
        typename Vector::Vec<CompType>::Type ddGammaVec2;
        typename Vector::Vec<CompType>::Type ddGammaVec3;
        Vector::DupFloatN<InType, CompType,
            Vector::GetUsedRegsN<InType, CompType, OutType, ScaleShiftType, float>()>::Do(
            0.0f, ddGammaVec0, ddGammaVec1, ddGammaVec2, ddGammaVec3);
        CompType ddGammaX = 0.0f;
        typename Vector::Vec<CompType>::Type ddGammaXVec0;
        typename Vector::Vec<CompType>::Type ddGammaXVec1;
        typename Vector::Vec<CompType>::Type ddGammaXVec2;
        typename Vector::Vec<CompType>::Type ddGammaXVec3;
        Vector::DupFloatN<InType, CompType,
            Vector::GetUsedRegsN<InType, CompType, OutType, ScaleShiftType, float>()>::Do(
            0.0f, ddGammaXVec0, ddGammaXVec1, ddGammaXVec2, ddGammaXVec3);
        float m = skipMean ? 0.0f : mean[n];
        typename Vector::Vec<CompType>::Type mean0;
        typename Vector::Vec<CompType>::Type mean1;
        typename Vector::Vec<CompType>::Type mean2;
        typename Vector::Vec<CompType>::Type mean3;
        Vector::DupFloatN<InType, CompType,
            Vector::GetUsedRegsN<InType, CompType, OutType, ScaleShiftType, float>()>::Do(
            m, mean0, mean1, mean2, mean3);

        if (calculateStats) {
            EvaluateLastAxisCalcStat(src, diffDst, scale, n, axisSize, ddGammaX, ddGamma, pgTrue, invSqrtVariance,
                useScale, mean0, mean1, mean2, mean3, ddGammaVec0, ddGammaVec1, ddGammaVec2, ddGammaVec3,
                ddGammaXVec0, ddGammaXVec1, ddGammaXVec2, ddGammaXVec3);
        }

        Vector::DupFloatN<InType, CompType,
            Vector::GetUsedRegsN<InType, CompType, OutType, ScaleShiftType, float>()>::Do(
            ddGamma, ddGammaVec0, ddGammaVec1, ddGammaVec2, ddGammaVec3);
        Vector::DupFloatN<InType, CompType,
            Vector::GetUsedRegsN<InType, CompType, OutType, ScaleShiftType, float>()>::Do(
            ddGammaX, ddGammaXVec0, ddGammaXVec1, ddGammaXVec2, ddGammaXVec3);

        typename Vector::Vec<CompType>::Type invSqrtVariance0;
        typename Vector::Vec<CompType>::Type invSqrtVariance1;
        typename Vector::Vec<CompType>::Type invSqrtVariance2;
        typename Vector::Vec<CompType>::Type invSqrtVariance3;
        Vector::DupFloatN<InType, CompType,
            Vector::GetUsedRegsN<InType, CompType, OutType, ScaleShiftType, float>()>::Do(
            invSqrtVariance, invSqrtVariance0, invSqrtVariance1, invSqrtVariance2, invSqrtVariance3);
        Service::UnrollFor::Do<1>(axisSize, LayerNormLastBWD<CompType, InType, OutType, ScaleShiftType>{},
            src + n * axisSize, diffDst + n * axisSize, scale, useScale, calculateStats,
            mean0, mean1, mean2, mean3, ddGammaVec0, ddGammaVec1, ddGammaVec2, ddGammaVec3,
            ddGammaXVec0, ddGammaXVec1, ddGammaXVec2, ddGammaXVec3,
            invSqrtVariance0, invSqrtVariance1, invSqrtVariance2, invSqrtVariance3, diffSrc + n * axisSize);
    }
}

} // LNorm
} // Detail
} // KuDNN

#endif //  KUDNN_LNORM_BWD_LAST_DEF_HPP