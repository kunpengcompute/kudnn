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

#ifndef KUDNN_LNORM_FWD_LAST_DEF_NEON_HPP
#define KUDNN_LNORM_FWD_LAST_DEF_NEON_HPP

#include "kudnn.hpp"
#include "vector_wrappers.hpp"
#include "kernels/lnorm/kudnn_lnorm_kernels.hpp"
#include "kernels/lnorm/kudnn_lnorm_fwd_struct.hpp"

namespace KuDNN {
namespace Detail {
namespace LNorm {

template <typename CompType, typename InType, typename OutType, typename ScaleShiftType>
void EvaluateLastAxisNEON(SizeType execAcrossAxis, const InType *src, OutType *dst, const ScaleShiftType *scale,
    const ScaleShiftType *shift, float *mean, float *variance,
    bool saveStats, float eps, NormalizationFlags flags, SizeType axisSize) noexcept
{
    bool calculateStats = !static_cast<bool>(flags & NormalizationFlags::USE_GLOBAL_STATS);
    bool useScale = static_cast<bool>(flags & NormalizationFlags::USE_SCALE);
    bool useShift = static_cast<bool>(flags & NormalizationFlags::USE_SHIFT);
    bool skipMean = static_cast<bool>(flags & NormalizationFlags::RMS_NORM);

    SizeType neonLoop = axisSize - (axisSize % LayerNormLastFWD<CompType, InType, OutType, ScaleShiftType,
        ARCH::NEON>::LoopInc());
    SizeType tailLoop = axisSize % LayerNormLastFWD<CompType, InType, OutType, ScaleShiftType, ARCH::NEON>::LoopInc();

    for (SizeType n = 0; n < execAcrossAxis; ++n) {
        float vMean = (calculateStats || skipMean) ? 0.0f : mean[n];
        float vVariance = calculateStats ? 0.0f : variance[n];
        svbool_t pgTrue = svptrue_b32();

        typename Vector::Vec<float, ARCH::NEON>::Type vMean0;
        typename Vector::Vec<float, ARCH::NEON>::Type vMean1;
        typename Vector::Vec<float, ARCH::NEON>::Type vMean2;
        typename Vector::Vec<float, ARCH::NEON>::Type vMean3;
        typename Vector::Vec<float, ARCH::VC>::Type vMean4;
        typename Vector::Vec<float, ARCH::VC>::Type vMean5;
        typename Vector::Vec<float, ARCH::VC>::Type vMean6;
        typename Vector::Vec<float, ARCH::VC>::Type vMean7;
        Vector::DupFloatN<InType, float, Vector::GetUsedRegsN<InType, OutType, CompType, ScaleShiftType, float>(),
            ARCH::NEON>::Do(vMean, vMean0, vMean1, vMean2, vMean3);
        Vector::DupFloatN<InType, float, Vector::GetUsedRegsN<InType, OutType, CompType, ScaleShiftType, float>(),
            ARCH::VC>::Do(vMean, vMean4, vMean5, vMean6, vMean7);

        typename Vector::Vec<float, ARCH::NEON>::Type vVariance0;
        typename Vector::Vec<float, ARCH::NEON>::Type vVariance1;
        typename Vector::Vec<float, ARCH::NEON>::Type vVariance2;
        typename Vector::Vec<float, ARCH::NEON>::Type vVariance3;
        typename Vector::Vec<float, ARCH::VC>::Type vVariance4;
        typename Vector::Vec<float, ARCH::VC>::Type vVariance5;
        typename Vector::Vec<float, ARCH::VC>::Type vVariance6;
        typename Vector::Vec<float, ARCH::VC>::Type vVariance7;
        Vector::DupFloatN<InType, float, Vector::GetUsedRegsN<InType, OutType, CompType, ScaleShiftType, float>(),
            ARCH::NEON>::Do(vVariance, vVariance0, vVariance1, vVariance2, vVariance3);
        Vector::DupFloatN<InType, float, Vector::GetUsedRegsN<InType, OutType, CompType, ScaleShiftType, float>(),
            ARCH::VC>::Do(vVariance, vVariance4, vVariance5, vVariance6, vVariance7);

        if (calculateStats) {
            if (!skipMean) {
                Service::UnrollFor::Do<1>(neonLoop,
                    LayerNormLastFWD<CompType, InType, OutType, ScaleShiftType, ARCH::NEON>{},
                    src + n * axisSize, vMean0, vMean1, vMean2, vMean3);
                float tmpMean0 = Vector::AddvN<float,
                    Vector::GetUsedRegsN<float, InType, OutType, CompType, ScaleShiftType>(), ARCH::NEON>::Do(
                    pgTrue, pgTrue, pgTrue, pgTrue, vMean0, vMean1, vMean2, vMean3);
                vMean += tmpMean0;
                Vector::DupFloatN<InType, float,
                    Vector::GetUsedRegsN<InType, OutType, CompType, ScaleShiftType, float>(), ARCH::NEON>::Do(
                    0.0f, vMean0, vMean1, vMean2, vMean3);
                
                Service::UnrollFor::Do<1>(tailLoop,
                    LayerNormLastFWD<CompType, InType, OutType, ScaleShiftType, ARCH::VC>{},
                    src + n * axisSize + neonLoop, vMean4, vMean5, vMean6, vMean7);
                float tmpMean1 = Vector::AddvN<float,
                    Vector::GetUsedRegsN<float, InType, OutType, CompType, ScaleShiftType>(), ARCH::VC>::Do(
                    pgTrue, pgTrue, pgTrue, pgTrue, vMean4, vMean5, vMean6, vMean7);
                vMean += tmpMean1;
                Vector::DupFloatN<InType, float,
                    Vector::GetUsedRegsN<InType, OutType, CompType, ScaleShiftType, float>(), ARCH::VC>::Do(
                    0.0f, vMean4, vMean5, vMean6, vMean7);
            }
            vMean /= axisSize;

            Vector::DupFloatN<InType, float,
                Vector::GetUsedRegsN<InType, OutType, CompType, ScaleShiftType, float>(), ARCH::NEON>::Do(
                vMean, vMean0, vMean1, vMean2, vMean3);
            Vector::DupFloatN<InType, float,
                Vector::GetUsedRegsN<InType, OutType, CompType, ScaleShiftType, float>(), ARCH::VC>::Do(
                vMean, vMean4, vMean5, vMean6, vMean7);

            Service::UnrollFor::Do<1>(neonLoop, LayerNormLastFWD<CompType, InType, OutType, ScaleShiftType,
                ARCH::NEON>{}, src + n * axisSize, vMean0, vMean1, vMean2, vMean3,
                vVariance0, vVariance1, vVariance2, vVariance3);
            float tmpVariance0 = Vector::AddvN<float,
                Vector::GetUsedRegsN<float, InType, OutType, CompType, ScaleShiftType>(), ARCH::NEON>::Do(
                pgTrue, pgTrue, pgTrue, pgTrue, vVariance0, vVariance1, vVariance2, vVariance3);
            vVariance += tmpVariance0;
            Vector::DupFloatN<InType, float,
                Vector::GetUsedRegsN<InType, OutType, CompType, ScaleShiftType, float>(), ARCH::NEON>::Do(
                0.0f, vVariance0, vVariance1, vVariance2, vVariance3);
            
            Service::UnrollFor::Do<1>(tailLoop, LayerNormLastFWD<CompType, InType, OutType, ScaleShiftType,
                ARCH::VC>{}, src + n * axisSize + neonLoop, vMean4, vMean5, vMean6, vMean7,
                vVariance4, vVariance5, vVariance6, vVariance7);
            float tmpVariance1 = Vector::AddvN<float,
                Vector::GetUsedRegsN<float, InType, OutType, CompType, ScaleShiftType>(), ARCH::VC>::Do(
                pgTrue, pgTrue, pgTrue, pgTrue, vVariance4, vVariance5, vVariance6, vVariance7);
            vVariance += tmpVariance1;
            Vector::DupFloatN<InType, float,
                Vector::GetUsedRegsN<InType, OutType, CompType, ScaleShiftType, float>(), ARCH::VC>::Do(
                0.0f, vVariance4, vVariance5, vVariance6, vVariance7);

            vVariance /= axisSize;

            if (saveStats) {
                if (!skipMean) {
                    mean[n] = vMean;
                }
                variance[n] = vVariance;
            }
        }
        float invSqrtVariance = 1.0f / std::sqrt(vVariance + eps);

        typename Vector::Vec<CompType, ARCH::NEON>::Type invSqrtVarianceVec0;
        typename Vector::Vec<CompType, ARCH::NEON>::Type invSqrtVarianceVec1;
        typename Vector::Vec<CompType, ARCH::NEON>::Type invSqrtVarianceVec2;
        typename Vector::Vec<CompType, ARCH::NEON>::Type invSqrtVarianceVec3;
        Vector::DupFloatN<InType, CompType,
            Vector::GetUsedRegsN<InType, OutType, CompType, ScaleShiftType, float>(), ARCH::NEON>::Do(
            invSqrtVariance, invSqrtVarianceVec0, invSqrtVarianceVec1, invSqrtVarianceVec2, invSqrtVarianceVec3);
        typename Vector::Vec<float, ARCH::NEON>::Type constants0;
        typename Vector::Vec<float, ARCH::NEON>::Type constants1;
        typename Vector::Vec<float, ARCH::NEON>::Type constants2;
        typename Vector::Vec<float, ARCH::NEON>::Type constants3;
        Vector::DupFloatN<InType, CompType,
            Vector::GetUsedRegsN<InType, OutType, CompType, ScaleShiftType, float>(), ARCH::NEON>::Do(
            invSqrtVariance, constants0, constants1, constants2, constants3);
        typename Vector::Vec<float, ARCH::NEON>::Type vMeanVect0;
        typename Vector::Vec<float, ARCH::NEON>::Type vMeanVect1;
        typename Vector::Vec<float, ARCH::NEON>::Type vMeanVect2;
        typename Vector::Vec<float, ARCH::NEON>::Type vMeanVect3;
        Vector::DupFloatN<InType, CompType,
            Vector::GetUsedRegsN<InType, OutType, CompType, ScaleShiftType, float>(), ARCH::NEON>::Do(
            vMean, vMeanVect0, vMeanVect1, vMeanVect2, vMeanVect3);
        
        typename Vector::Vec<CompType, ARCH::VC>::Type invSqrtVarianceVec4;
        typename Vector::Vec<CompType, ARCH::VC>::Type invSqrtVarianceVec5;
        typename Vector::Vec<CompType, ARCH::VC>::Type invSqrtVarianceVec6;
        typename Vector::Vec<CompType, ARCH::VC>::Type invSqrtVarianceVec7;
        Vector::DupFloatN<InType, CompType,
            Vector::GetUsedRegsN<InType, OutType, CompType, ScaleShiftType, float>(), ARCH::VC>::Do(
            invSqrtVariance, invSqrtVarianceVec4, invSqrtVarianceVec5, invSqrtVarianceVec6, invSqrtVarianceVec7);
        typename Vector::Vec<float, ARCH::VC>::Type constants4;
        typename Vector::Vec<float, ARCH::VC>::Type constants5;
        typename Vector::Vec<float, ARCH::VC>::Type constants6;
        typename Vector::Vec<float, ARCH::VC>::Type constants7;
        Vector::DupFloatN<InType, CompType,
            Vector::GetUsedRegsN<InType, OutType, CompType, ScaleShiftType, float>(), ARCH::VC>::Do(
            invSqrtVariance, constants4, constants5, constants6, constants7);
        typename Vector::Vec<float, ARCH::VC>::Type vMeanVect4;
        typename Vector::Vec<float, ARCH::VC>::Type vMeanVect5;
        typename Vector::Vec<float, ARCH::VC>::Type vMeanVect6;
        typename Vector::Vec<float, ARCH::VC>::Type vMeanVect7;
        Vector::DupFloatN<InType, CompType,
            Vector::GetUsedRegsN<InType, OutType, CompType, ScaleShiftType, float>(), ARCH::VC>::Do(
            vMean, vMeanVect4, vMeanVect5, vMeanVect6, vMeanVect7);

        Service::UnrollFor::Do<1>(neonLoop, LayerNormLastFWD<CompType, InType, OutType, ScaleShiftType, ARCH::NEON>{},
            src + n * axisSize, scale, shift, useScale, useShift,
            invSqrtVarianceVec0, invSqrtVarianceVec1, invSqrtVarianceVec2, invSqrtVarianceVec3,
            constants0, constants1, constants2, constants3,
            vMeanVect0, vMeanVect1, vMeanVect2, vMeanVect3, dst + n * axisSize);
        Service::UnrollFor::Do<1>(tailLoop, LayerNormLastFWD<CompType, InType, OutType, ScaleShiftType, ARCH::VC>{},
            src + n * axisSize + neonLoop, scale + neonLoop, shift + neonLoop, useScale, useShift,
            invSqrtVarianceVec4, invSqrtVarianceVec5, invSqrtVarianceVec6, invSqrtVarianceVec7,
            constants4, constants5, constants6, constants7,
            vMeanVect4, vMeanVect5, vMeanVect6, vMeanVect7, dst + n * axisSize + neonLoop);
    }
}

} // LNorm
} // Detail
} // KuDNN

#endif //  KUDNN_LNORM_FWD_LAST_DEF_HPP
