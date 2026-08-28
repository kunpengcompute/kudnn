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

#ifndef KUDNN_LNORM_FWD_LAST_DEF_VC_HPP
#define KUDNN_LNORM_FWD_LAST_DEF_VC_HPP

#include "kudnn.hpp"
#include "vector_wrappers.hpp"
#include "kernels/lnorm/kudnn_lnorm_kernels.hpp"
#include "kernels/lnorm/kudnn_lnorm_fwd_struct.hpp"

namespace KuDNN {
namespace Detail {
namespace LNorm {

template <typename CompType, typename InType, typename OutType, typename ScaleShiftType>
void EvaluateLastAxisVC(SizeType execAcrossAxis, const InType *src, OutType *dst, const ScaleShiftType *scale,
    const ScaleShiftType *shift, float *mean, float *variance,
    bool saveStats, float eps, NormalizationFlags flags, SizeType axisSize) noexcept
{
    bool calculateStats = !static_cast<bool>(flags & NormalizationFlags::USE_GLOBAL_STATS);
    bool useScale = static_cast<bool>(flags & NormalizationFlags::USE_SCALE);
    bool useShift = static_cast<bool>(flags & NormalizationFlags::USE_SHIFT);
    bool skipMean = static_cast<bool>(flags & NormalizationFlags::RMS_NORM);

    for (SizeType n = 0; n < execAcrossAxis; ++n) {
        float vMean = (calculateStats || skipMean) ? 0.0f : mean[n];
        float vVariance = calculateStats ? 0.0f : variance[n];
        svbool_t pgTrue = svptrue_b32();

        typename Vector::Vec<float>::Type vMean0;
        typename Vector::Vec<float>::Type vMean1;
        typename Vector::Vec<float>::Type vMean2;
        typename Vector::Vec<float>::Type vMean3;
        Vector::DupFloatN<InType, float, Vector::GetUsedRegsN<InType, OutType, CompType, ScaleShiftType, float>(),
            ARCH::VC>::Do(vMean, vMean0, vMean1, vMean2, vMean3);
        
        typename Vector::Vec<float>::Type vVariance0;
        typename Vector::Vec<float>::Type vVariance1;
        typename Vector::Vec<float>::Type vVariance2;
        typename Vector::Vec<float>::Type vVariance3;
        Vector::DupFloatN<InType, float, Vector::GetUsedRegsN<InType, OutType, CompType, ScaleShiftType, float>(),
            ARCH::VC>::Do(vVariance, vVariance0, vVariance1, vVariance2, vVariance3);

        if (calculateStats) {
            if (!skipMean) {
                Service::UnrollFor::Do<1>(axisSize, LayerNormLastFWD<CompType, InType, OutType, ScaleShiftType,
                ARCH::VC>{}, src + n * axisSize, vMean0, vMean1, vMean2, vMean3);
                float tmpMean = Vector::AddvN<float,
                    Vector::GetUsedRegsN<float, InType, OutType, CompType, ScaleShiftType>(), ARCH::VC>::Do(
                    pgTrue, pgTrue, pgTrue, pgTrue, vMean0, vMean1, vMean2, vMean3);
                vMean += tmpMean;
                Vector::DupFloatN<InType, float,
                    Vector::GetUsedRegsN<InType, OutType, CompType, ScaleShiftType, float>(), ARCH::VC>::Do(
                    0.0f, vMean0, vMean1, vMean2, vMean3);
            }
            vMean /= axisSize;
            Vector::DupFloatN<InType, float,
                Vector::GetUsedRegsN<InType, OutType, CompType, ScaleShiftType, float>(), ARCH::VC>::Do(
                vMean, vMean0, vMean1, vMean2, vMean3);

            Service::UnrollFor::Do<1>(axisSize, LayerNormLastFWD<CompType, InType, OutType, ScaleShiftType,
                ARCH::VC>{}, src + n * axisSize, vMean0, vMean1, vMean2, vMean3,
                vVariance0, vVariance1, vVariance2, vVariance3);
            float tmpVariance = Vector::AddvN<float,
                Vector::GetUsedRegsN<float, InType, OutType, CompType, ScaleShiftType>(), ARCH::VC>::Do(
                pgTrue, pgTrue, pgTrue, pgTrue, vVariance0, vVariance1, vVariance2, vVariance3);
            vVariance += tmpVariance;
            Vector::DupFloatN<InType, float,
                Vector::GetUsedRegsN<InType, OutType, CompType, ScaleShiftType, float>(), ARCH::VC>::Do(
                0.0f, vVariance0, vVariance1, vVariance2, vVariance3);
            vVariance /= axisSize;

            if (saveStats) {
                if (!skipMean) {
                    mean[n] = vMean;
                }
                variance[n] = vVariance;
            }
        }
        float invSqrtVariance = 1.0f / std::sqrt(vVariance + eps);

        typename Vector::Vec<CompType, ARCH::VC>::Type invSqrtVarianceVec0;
        typename Vector::Vec<CompType, ARCH::VC>::Type invSqrtVarianceVec1;
        typename Vector::Vec<CompType, ARCH::VC>::Type invSqrtVarianceVec2;
        typename Vector::Vec<CompType, ARCH::VC>::Type invSqrtVarianceVec3;
        Vector::DupFloatN<InType, CompType,
            Vector::GetUsedRegsN<InType, OutType, CompType, ScaleShiftType, float>(), ARCH::VC>::Do(
            invSqrtVariance, invSqrtVarianceVec0, invSqrtVarianceVec1, invSqrtVarianceVec2, invSqrtVarianceVec3);
        typename Vector::Vec<CompType, ARCH::VC>::Type constants0;
        typename Vector::Vec<CompType, ARCH::VC>::Type constants1;
        typename Vector::Vec<CompType, ARCH::VC>::Type constants2;
        typename Vector::Vec<CompType, ARCH::VC>::Type constants3;
        Vector::DupFloatN<InType, CompType,
            Vector::GetUsedRegsN<InType, OutType, CompType, ScaleShiftType, float>(), ARCH::VC>::Do(
            invSqrtVariance, constants0, constants1, constants2, constants3);
        typename Vector::Vec<CompType, ARCH::VC>::Type vMeanVect0;
        typename Vector::Vec<CompType, ARCH::VC>::Type vMeanVect1;
        typename Vector::Vec<CompType, ARCH::VC>::Type vMeanVect2;
        typename Vector::Vec<CompType, ARCH::VC>::Type vMeanVect3;
        Vector::DupFloatN<InType, CompType,
            Vector::GetUsedRegsN<InType, OutType, CompType, ScaleShiftType, float>(), ARCH::VC>::Do(
            vMean, vMeanVect0, vMeanVect1, vMeanVect2, vMeanVect3);

        Service::UnrollFor::Do<1>(axisSize, LayerNormLastFWD<CompType, InType, OutType, ScaleShiftType, ARCH::VC>{},
            src + n * axisSize, scale, shift, useScale, useShift,
            invSqrtVarianceVec0, invSqrtVarianceVec1, invSqrtVarianceVec2, invSqrtVarianceVec3,
            constants0, constants1, constants2, constants3,
            vMeanVect0, vMeanVect1, vMeanVect2, vMeanVect3, dst + n * axisSize);
    }
}

} // LNorm
} // Detail
} // KuDNN

#endif //  KUDNN_LNORM_FWD_LAST_DEF_HPP
