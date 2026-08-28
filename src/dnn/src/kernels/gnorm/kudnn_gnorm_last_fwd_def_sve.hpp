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

#ifndef KUDNN_GNORM_LAST_FWD_DEF_VC_HPP
#define KUDNN_GNORM_LAST_FWD_DEF_VC_HPP

#include "kudnn.hpp"
#include "vector_wrappers.hpp"
#include "kernels/gnorm/kudnn_gnorm_kernels.hpp"
#include "kernels/gnorm/kudnn_gnorm_calculate_struct.hpp"

namespace KuDNN {
namespace Detail {
namespace GNorm {

template <typename CompType, typename InType, typename OutType, typename ScaleShiftType>
void EvaluateLastAxisVC(SizeType outerSize, SizeType batchSize, const InType *src, OutType *dst, SizeType groupSize,
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
            float vVariance = calculateStats ? 0.0f : variance[offsetStats];
            svbool_t pgTrue = svptrue_b32();

            typename Vector::Vec<float, ARCH::VC>::Type vMean0;
            typename Vector::Vec<float, ARCH::VC>::Type vMean1;
            typename Vector::Vec<float, ARCH::VC>::Type vMean2;
            typename Vector::Vec<float, ARCH::VC>::Type vMean3;
            Vector::DupFloatN<InType, float,
                Vector::GetUsedRegsN<InType, OutType, CompType, ScaleShiftType, float>(), ARCH::VC>::Do(
                vMean, vMean0, vMean1, vMean2, vMean3);
            typename Vector::Vec<float, ARCH::VC>::Type vVariance0;
            typename Vector::Vec<float, ARCH::VC>::Type vVariance1;
            typename Vector::Vec<float, ARCH::VC>::Type vVariance2;
            typename Vector::Vec<float, ARCH::VC>::Type vVariance3;
            Vector::DupFloatN<InType, float,
                Vector::GetUsedRegsN<InType, OutType, CompType, ScaleShiftType, float>(), ARCH::VC>::Do(
                vVariance, vVariance0, vVariance1, vVariance2, vVariance3);

            if (calculateStats) {
                for (SizeType n = 0; n < dhw; ++n) {
                    SizeType offset = a * dhw * axisSize + n * axisSize + g * channelPerGroup;

                    Service::UnrollFor::Do<1>(channelPerGroup,
                        GroupNormLastFWD<CompType, InType, OutType, ScaleShiftType, ARCH::VC>{},
                        src + offset, vMean0, vMean1, vMean2, vMean3);
                    float tmpMean = Vector::AddvN<float,
                        Vector::GetUsedRegsN<float, InType, OutType, CompType, ScaleShiftType>(), ARCH::VC>::Do(
                        pgTrue, pgTrue, pgTrue, pgTrue, vMean0, vMean1, vMean2, vMean3);
                    vMean += tmpMean;
                    Vector::DupFloatN<InType, float,
                        Vector::GetUsedRegsN<InType, OutType, CompType, ScaleShiftType, float>(), ARCH::VC>::Do(
                        0.0f, vMean0, vMean1, vMean2, vMean3);
                }
                vMean /= dataPerGroup;

                Vector::DupFloatN<InType, float,
                    Vector::GetUsedRegsN<InType, OutType, CompType, ScaleShiftType, float>(), ARCH::VC>::Do(
                    vMean, vMean0, vMean1, vMean2, vMean3);

                for (SizeType n = 0; n < dhw; ++n) {
                    SizeType offset = a * dhw * axisSize + n * axisSize + g * channelPerGroup;

                    Service::UnrollFor::Do<1>(channelPerGroup,
                        GroupNormLastFWD<CompType, InType, OutType, ScaleShiftType, ARCH::VC>{},
                        src + offset, vMean0, vMean1, vMean2, vMean3, vVariance0, vVariance1, vVariance2, vVariance3);
                    float tmpVariance = Vector::AddvN<float,
                        Vector::GetUsedRegsN<float, InType, OutType, CompType, ScaleShiftType>(), ARCH::VC>::Do(
                        pgTrue, pgTrue, pgTrue, pgTrue, vVariance0, vVariance1, vVariance2, vVariance3);
                    vVariance += tmpVariance;
                    Vector::DupFloatN<InType, float,
                        Vector::GetUsedRegsN<InType, OutType, CompType, ScaleShiftType, float>(), ARCH::VC>::Do(
                        0.0f, vVariance0, vVariance1, vVariance2, vVariance3);
                }
                vVariance /= dataPerGroup;
            }

            if (saveStats) {
                mean[offsetStats] = vMean;
                variance[offsetStats] = vVariance;
            }
            float invSqrtVariance1 = 1.0f / std::sqrt(vVariance + eps);

            typename Vector::Vec<CompType, ARCH::VC>::Type invSqrtVarianceVec10;
            typename Vector::Vec<CompType, ARCH::VC>::Type invSqrtVarianceVec11;
            typename Vector::Vec<CompType, ARCH::VC>::Type invSqrtVarianceVec12;
            typename Vector::Vec<CompType, ARCH::VC>::Type invSqrtVarianceVec13;
            Vector::DupFloatN<InType, CompType,
                Vector::GetUsedRegsN<InType, OutType, CompType, ScaleShiftType, float>(), ARCH::VC>::Do(
                invSqrtVariance1,
                invSqrtVarianceVec10, invSqrtVarianceVec11, invSqrtVarianceVec12, invSqrtVarianceVec13);
            typename Vector::Vec<float, ARCH::VC>::Type constants10;
            typename Vector::Vec<float, ARCH::VC>::Type constants11;
            typename Vector::Vec<float, ARCH::VC>::Type constants12;
            typename Vector::Vec<float, ARCH::VC>::Type constants13;
            Vector::DupFloatN<InType, CompType,
                Vector::GetUsedRegsN<InType, OutType, CompType, ScaleShiftType, float>(), ARCH::VC>::Do(
                invSqrtVariance1, constants10, constants11, constants12, constants13);
            typename Vector::Vec<float, ARCH::VC>::Type vMeanVect10;
            typename Vector::Vec<float, ARCH::VC>::Type vMeanVect11;
            typename Vector::Vec<float, ARCH::VC>::Type vMeanVect12;
            typename Vector::Vec<float, ARCH::VC>::Type vMeanVect13;
            Vector::DupFloatN<InType, CompType,
                Vector::GetUsedRegsN<InType, OutType, CompType, ScaleShiftType, float>(), ARCH::VC>::Do(
                vMean, vMeanVect10, vMeanVect11, vMeanVect12, vMeanVect13);

            for (SizeType n = 0; n < dhw; ++n) {
                SizeType offset = a * dhw * axisSize + n * axisSize + g * channelPerGroup;
                Service::UnrollFor::Do<1>(channelPerGroup,
                    GroupNormLastFWD<CompType, InType, OutType, ScaleShiftType, ARCH::VC>{},
                    src + offset, scale + g * channelPerGroup, shift + g * channelPerGroup, useScale, useShift, g,
                    invSqrtVarianceVec10, invSqrtVarianceVec11, invSqrtVarianceVec12, invSqrtVarianceVec13,
                    constants10, constants11, constants12, constants13,
                    vMeanVect10, vMeanVect11, vMeanVect12, vMeanVect13, dst + offset);
            }
        }
    }
}

} // GNorm
} // Detail
} // KuDNN

#endif //  KUDNN_GNORM_FWD_DEF_HPP