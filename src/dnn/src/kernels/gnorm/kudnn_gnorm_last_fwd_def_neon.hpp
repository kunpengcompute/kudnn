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

#ifndef KUDNN_GNORM_LAST_FWD_DEF_NEON_HPP
#define KUDNN_GNORM_LAST_FWD_DEF_NEON_HPP

#include "kudnn.hpp"

#include "vector_wrappers.hpp"
#include "kernels/gnorm/kudnn_gnorm_kernels.hpp"
#include "kernels/gnorm/kudnn_gnorm_calculate_struct.hpp"

namespace KuDNN {
namespace Detail {
namespace GNorm {

template <typename CompType, typename InType, typename OutType, typename ScaleShiftType>
void EvaluateLastAxisNeon(SizeType outerSize, SizeType batchSize, const InType *src, OutType *dst, SizeType groupSize,
    const ScaleShiftType *scale, const ScaleShiftType *shift, float *mean, float *variance,
    bool saveStats, float eps, NormalizationFlags flags, SizeType axisSize) noexcept
{
    bool calculateStats = !static_cast<bool>(flags & NormalizationFlags::USE_GLOBAL_STATS);
    bool useScale = static_cast<bool>(flags & NormalizationFlags::USE_SCALE);
    bool useShift = static_cast<bool>(flags & NormalizationFlags::USE_SHIFT);
    SizeType dhw = outerSize / batchSize;
    SizeType channelPerGroup = axisSize / groupSize;
    SizeType dataPerGroup = dhw * channelPerGroup;

    SizeType neonLoop = channelPerGroup - (channelPerGroup %
                        GroupNormLastFWD<CompType, InType, OutType, ScaleShiftType, ARCH::NEON>::LoopInc());
    SizeType tailLoop = channelPerGroup %
                        GroupNormLastFWD<CompType, InType, OutType, ScaleShiftType, ARCH::NEON>::LoopInc(); // Use VC

    for (SizeType a = 0; a < batchSize; ++a) {
        for (SizeType g = 0; g < groupSize; ++g) {
            SizeType offsetStats = a * groupSize + g;
            float vMean1 = calculateStats ? 0.0f : mean[offsetStats];
            float vVariance1 = calculateStats ? 0.0f : variance[offsetStats];
            svbool_t pgTrue = svptrue_b32();

            typename Vector::Vec<float, ARCH::NEON>::Type vMean10;
            typename Vector::Vec<float, ARCH::NEON>::Type vMean11;
            typename Vector::Vec<float, ARCH::NEON>::Type vMean12;
            typename Vector::Vec<float, ARCH::NEON>::Type vMean13;
            typename Vector::Vec<float, ARCH::VC>::Type vMean14;
            typename Vector::Vec<float, ARCH::VC>::Type vMean15;
            typename Vector::Vec<float, ARCH::VC>::Type vMean16;
            typename Vector::Vec<float, ARCH::VC>::Type vMean17;
            typename Vector::Vec<float, ARCH::NEON>::Type vVariance10;
            typename Vector::Vec<float, ARCH::NEON>::Type vVariance11;
            typename Vector::Vec<float, ARCH::NEON>::Type vVariance12;
            typename Vector::Vec<float, ARCH::NEON>::Type vVariance13;
            typename Vector::Vec<float, ARCH::VC>::Type vVariance14;
            typename Vector::Vec<float, ARCH::VC>::Type vVariance15;
            typename Vector::Vec<float, ARCH::VC>::Type vVariance16;
            typename Vector::Vec<float, ARCH::VC>::Type vVariance17;
            Vector::DupFloatN<InType, float,
                Vector::GetUsedRegsN<InType, OutType, CompType, ScaleShiftType, float>(), ARCH::NEON>::Do(
                vMean1, vMean10, vMean11, vMean12, vMean13);
            Vector::DupFloatN<InType, float,
                Vector::GetUsedRegsN<InType, OutType, CompType, ScaleShiftType, float>(), ARCH::VC>::Do(
                vMean1, vMean14, vMean15, vMean16, vMean17);
            Vector::DupFloatN<InType, float,
                Vector::GetUsedRegsN<InType, OutType, CompType, ScaleShiftType, float>(), ARCH::NEON>::Do(
                vVariance1, vVariance10, vVariance11, vVariance12, vVariance13);
            Vector::DupFloatN<InType, float,
                Vector::GetUsedRegsN<InType, OutType, CompType, ScaleShiftType, float>(), ARCH::VC>::Do(
                vVariance1, vVariance14, vVariance15, vVariance16, vVariance17);

            if (calculateStats) {
                for (SizeType n = 0; n < dhw; ++n) {
                    SizeType offset = a * dhw * axisSize + n * axisSize + g * channelPerGroup;

                    Service::UnrollFor::Do<1>(neonLoop,
                        GroupNormLastFWD<CompType, InType, OutType, ScaleShiftType, ARCH::NEON>{},
                        src + offset, vMean10, vMean11, vMean12, vMean13);
                    float tmpMean = Vector::AddvN<float,
                        Vector::GetUsedRegsN<float, InType, OutType, CompType, ScaleShiftType>(), ARCH::NEON>::Do(
                        pgTrue, pgTrue, pgTrue, pgTrue, vMean10, vMean11, vMean12, vMean13);
                    vMean1 += tmpMean;
                    Vector::DupFloatN<InType, float,
                        Vector::GetUsedRegsN<InType, OutType, CompType, ScaleShiftType, float>(), ARCH::NEON>::Do(
                        0.0f, vMean10, vMean11, vMean12, vMean13);

                    Service::UnrollFor::Do<1>(tailLoop,
                        GroupNormLastFWD<CompType, InType, OutType, ScaleShiftType, ARCH::VC>{},
                        src + offset + neonLoop, vMean14, vMean15, vMean16, vMean17);
                    float tmpMean1 = Vector::AddvN<float,
                        Vector::GetUsedRegsN<float, InType, OutType, CompType, ScaleShiftType>(), ARCH::VC>::Do(
                        pgTrue, pgTrue, pgTrue, pgTrue, vMean14, vMean15, vMean16, vMean17);
                    vMean1 += tmpMean1;
                    Vector::DupFloatN<InType, float,
                        Vector::GetUsedRegsN<InType, OutType, CompType, ScaleShiftType, float>(), ARCH::VC>::Do(
                        0.0f, vMean14, vMean15, vMean16, vMean17);
                }
                vMean1 /= dataPerGroup;

                Vector::DupFloatN<InType, float,
                    Vector::GetUsedRegsN<InType, OutType, CompType, ScaleShiftType, float>(), ARCH::NEON>::Do(
                    vMean1, vMean10, vMean11, vMean12, vMean13);
                Vector::DupFloatN<InType, float,
                    Vector::GetUsedRegsN<InType, OutType, CompType, ScaleShiftType, float>(), ARCH::VC>::Do(
                    vMean1, vMean14, vMean15, vMean16, vMean17);

                for (SizeType n = 0; n < dhw; ++n) {
                    SizeType offset = a * dhw * axisSize + n * axisSize + g * channelPerGroup;

                    Service::UnrollFor::Do<1>(neonLoop,
                        GroupNormLastFWD<CompType, InType, OutType, ScaleShiftType, ARCH::NEON>{},
                        src + offset, vMean10, vMean11, vMean12, vMean13,
                        vVariance10, vVariance11, vVariance12, vVariance13);
                    float tmpVariance = Vector::AddvN<float,
                        Vector::GetUsedRegsN<float, InType, OutType, CompType, ScaleShiftType>(), ARCH::NEON>::Do(
                        pgTrue, pgTrue, pgTrue, pgTrue, vVariance10, vVariance11, vVariance12, vVariance13);
                    vVariance1 += tmpVariance;
                    Vector::DupFloatN<InType, float,
                        Vector::GetUsedRegsN<InType, OutType, CompType, ScaleShiftType, float>(), ARCH::NEON>::Do(
                        0.0f, vVariance10, vVariance11, vVariance12, vVariance13);

                    Service::UnrollFor::Do<1>(tailLoop,
                        GroupNormLastFWD<CompType, InType, OutType, ScaleShiftType, ARCH::VC>{},
                        src + offset + neonLoop, vMean14, vMean15, vMean16, vMean17,
                        vVariance14, vVariance15, vVariance16, vVariance17);
                    float tmpVariance1 = Vector::AddvN<float,
                        Vector::GetUsedRegsN<float, InType, OutType, CompType, ScaleShiftType>(), ARCH::VC>::Do(
                        pgTrue, pgTrue, pgTrue, pgTrue, vVariance14, vVariance15, vVariance16, vVariance17);
                    vVariance1 += tmpVariance1;
                    Vector::DupFloatN<InType, float,
                        Vector::GetUsedRegsN<InType, OutType, CompType, ScaleShiftType, float>(), ARCH::VC>::Do(
                        0.0f, vVariance14, vVariance15, vVariance16, vVariance17);
                }
                vVariance1 /= dataPerGroup;
            }

            if (saveStats) {
                mean[offsetStats] = vMean1;
                variance[offsetStats] = vVariance1;
            }
            float invSqrtVariance1 = 1.0f / std::sqrt(vVariance1 + eps);

            typename Vector::Vec<CompType, ARCH::NEON>::Type invSqrtVarianceVec10;
            typename Vector::Vec<CompType, ARCH::NEON>::Type invSqrtVarianceVec11;
            typename Vector::Vec<CompType, ARCH::NEON>::Type invSqrtVarianceVec12;
            typename Vector::Vec<CompType, ARCH::NEON>::Type invSqrtVarianceVec13;
            Vector::DupFloatN<InType, CompType,
                Vector::GetUsedRegsN<InType, OutType, CompType, ScaleShiftType, float>(), ARCH::NEON>::Do(
                invSqrtVariance1,
                invSqrtVarianceVec10, invSqrtVarianceVec11, invSqrtVarianceVec12, invSqrtVarianceVec13);
            typename Vector::Vec<float, ARCH::NEON>::Type constants10;
            typename Vector::Vec<float, ARCH::NEON>::Type constants11;
            typename Vector::Vec<float, ARCH::NEON>::Type constants12;
            typename Vector::Vec<float, ARCH::NEON>::Type constants13;
            Vector::DupFloatN<InType, CompType,
                Vector::GetUsedRegsN<InType, OutType, CompType, ScaleShiftType, float>(), ARCH::NEON>::Do(
                invSqrtVariance1, constants10, constants11, constants12, constants13);
            typename Vector::Vec<float, ARCH::NEON>::Type vMeanVect10;
            typename Vector::Vec<float, ARCH::NEON>::Type vMeanVect11;
            typename Vector::Vec<float, ARCH::NEON>::Type vMeanVect12;
            typename Vector::Vec<float, ARCH::NEON>::Type vMeanVect13;
            Vector::DupFloatN<InType, CompType,
                Vector::GetUsedRegsN<InType, OutType, CompType, ScaleShiftType, float>(), ARCH::NEON>::Do(
                vMean1, vMeanVect10, vMeanVect11, vMeanVect12, vMeanVect13);

            typename Vector::Vec<CompType, ARCH::VC>::Type invSqrtVarianceVec14;
            typename Vector::Vec<CompType, ARCH::VC>::Type invSqrtVarianceVec15;
            typename Vector::Vec<CompType, ARCH::VC>::Type invSqrtVarianceVec16;
            typename Vector::Vec<CompType, ARCH::VC>::Type invSqrtVarianceVec17;
            Vector::DupFloatN<InType, CompType,
                Vector::GetUsedRegsN<InType, OutType, CompType, ScaleShiftType, float>(), ARCH::VC>::Do(
                invSqrtVariance1,
                invSqrtVarianceVec14, invSqrtVarianceVec15, invSqrtVarianceVec16, invSqrtVarianceVec17);
            typename Vector::Vec<float, ARCH::VC>::Type constants14;
            typename Vector::Vec<float, ARCH::VC>::Type constants15;
            typename Vector::Vec<float, ARCH::VC>::Type constants16;
            typename Vector::Vec<float, ARCH::VC>::Type constants17;
            Vector::DupFloatN<InType, CompType,
                Vector::GetUsedRegsN<InType, OutType, CompType, ScaleShiftType, float>(), ARCH::VC>::Do(
                invSqrtVariance1, constants14, constants15, constants16, constants17);
            typename Vector::Vec<float, ARCH::VC>::Type vMeanVect14;
            typename Vector::Vec<float, ARCH::VC>::Type vMeanVect15;
            typename Vector::Vec<float, ARCH::VC>::Type vMeanVect16;
            typename Vector::Vec<float, ARCH::VC>::Type vMeanVect17;
            Vector::DupFloatN<InType, CompType,
                Vector::GetUsedRegsN<InType, OutType, CompType, ScaleShiftType, float>(), ARCH::VC>::Do(
                vMean1, vMeanVect14, vMeanVect15, vMeanVect16, vMeanVect17);

            for (SizeType n = 0; n < dhw; ++n) {
                SizeType offset = a * dhw * axisSize + n * axisSize + g * channelPerGroup;

                Service::UnrollFor::Do<1>(neonLoop,
                    GroupNormLastFWD<CompType, InType, OutType, ScaleShiftType, ARCH::NEON>{},
                    src + offset, scale + g * channelPerGroup, shift + g * channelPerGroup, useScale, useShift, g,
                    invSqrtVarianceVec10, invSqrtVarianceVec11, invSqrtVarianceVec12, invSqrtVarianceVec13,
                    constants10, constants11, constants12, constants13,
                    vMeanVect10, vMeanVect11, vMeanVect12, vMeanVect13, dst + offset);
                Service::UnrollFor::Do<1>(tailLoop,
                    GroupNormLastFWD<CompType, InType, OutType, ScaleShiftType, ARCH::VC>{},
                    src + offset + neonLoop, scale + g * channelPerGroup + neonLoop,
                    shift + g * channelPerGroup + neonLoop, useScale, useShift, g,
                    invSqrtVarianceVec14, invSqrtVarianceVec15, invSqrtVarianceVec16, invSqrtVarianceVec17,
                    constants14, constants15, constants16, constants17,
                    vMeanVect14, vMeanVect15, vMeanVect16, vMeanVect17, dst + offset + neonLoop);
            }
        }
    }
}

} // GNorm
} // Detail
} // KuDNN

#endif //  KUDNN_GNORM_FWD_DEF_HPP