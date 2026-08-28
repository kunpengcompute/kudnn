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

#ifndef KUDNN_GNORM_CROSS_FWD_DEF_VC_HPP
#define KUDNN_GNORM_CROSS_FWD_DEF_VC_HPP

#include "kudnn.hpp"

#include "vector_wrappers.hpp"
#include "kernels/gnorm/kudnn_gnorm_kernels.hpp"
#include "kernels/gnorm/kudnn_gnorm_calculate_struct.hpp"

namespace KuDNN {
namespace Detail {
namespace GNorm {

template <typename CompType, typename InType, typename OutType, typename ScaleShiftType>
void EvaluateCrossAxisVC(SizeType outerSize, const InType *src, OutType *dst, SizeType groupSize,
    const ScaleShiftType *scale, const ScaleShiftType *shift, float *mean, float *variance, SizeType start,
    bool saveStats, float eps, NormalizationFlags flags, SizeType axisSize, SizeType innerSize) noexcept
{
    bool calculateStats = !static_cast<bool>(flags & NormalizationFlags::USE_GLOBAL_STATS);
    bool useScale = static_cast<bool>(flags & NormalizationFlags::USE_SCALE);
    bool useShift = static_cast<bool>(flags & NormalizationFlags::USE_SHIFT);
    SizeType channelPerGroup = axisSize / groupSize;
    SizeType dataPerGroup = innerSize * channelPerGroup;

    typename Vector::Vec<float, ARCH::VC>::Type dataPerGroupVec0;
    typename Vector::Vec<float, ARCH::VC>::Type dataPerGroupVec1;
    typename Vector::Vec<float, ARCH::VC>::Type dataPerGroupVec2;
    typename Vector::Vec<float, ARCH::VC>::Type dataPerGroupVec3;
    Vector::DupFloatN<InType, float,
        Vector::GetUsedRegsN<InType, CompType, OutType, ScaleShiftType, float>(), ARCH::VC>::Do(
        dataPerGroup, dataPerGroupVec0, dataPerGroupVec1, dataPerGroupVec2, dataPerGroupVec3);
    
    for (SizeType group = 0; group < outerSize; ++group) {
        SizeType startGroup = (start + group) % groupSize;
        SizeType offset = group * channelPerGroup * innerSize;
        SizeType offsetStats = group;
        float vMean = calculateStats ? 0.0f : mean[offsetStats];
        float vVariance = calculateStats ? 0.0f : variance[offsetStats];

        typename Vector::Vec<float, ARCH::VC>::Type vMean0;
        typename Vector::Vec<float, ARCH::VC>::Type vMean1;
        typename Vector::Vec<float, ARCH::VC>::Type vMean2;
        typename Vector::Vec<float, ARCH::VC>::Type vMean3;
        typename Vector::Vec<float, ARCH::VC>::Type vVariance0;
        typename Vector::Vec<float, ARCH::VC>::Type vVariance1;
        typename Vector::Vec<float, ARCH::VC>::Type vVariance2;
        typename Vector::Vec<float, ARCH::VC>::Type vVariance3;
        Vector::DupFloatN<InType, float,
            Vector::GetUsedRegsN<InType, OutType, CompType, ScaleShiftType, float>(), ARCH::VC>::Do(
            vMean, vMean0, vMean1, vMean2, vMean3);
        Vector::DupFloatN<InType, float,
            Vector::GetUsedRegsN<InType, OutType, CompType, ScaleShiftType, float>(), ARCH::VC>::Do(
            vVariance, vVariance0, vVariance1, vVariance2, vVariance3);

        if (calculateStats) {
            svbool_t pgTrue = svptrue_b32();
            Service::UnrollFor::Do<1>(innerSize * channelPerGroup,
                GroupNormCrossFWD<CompType, InType, OutType, ScaleShiftType>{},
                src + offset, vMean0, vMean1, vMean2, vMean3);
            vMean = Vector::AddvN<float,
                Vector::GetUsedRegsN<float, InType, OutType, CompType, ScaleShiftType>(), ARCH::VC>::Do(
                pgTrue, pgTrue, pgTrue, pgTrue, vMean0, vMean1, vMean2, vMean3);
            vMean /= dataPerGroup;

            Vector::DupFloatN<InType, float,
                Vector::GetUsedRegsN<InType, OutType, CompType, ScaleShiftType, float>(), ARCH::VC>::Do(
                vMean, vMean0, vMean1, vMean2, vMean3);
            Service::UnrollFor::Do<1>(innerSize * channelPerGroup,
                GroupNormCrossFWD<CompType, InType, OutType, ScaleShiftType, ARCH::VC>{},
                src + offset, vMean0, vMean1, vMean2, vMean3, vVariance0, vVariance1, vVariance2, vVariance3);
            vVariance = Vector::AddvN<float,
                Vector::GetUsedRegsN<float, InType, OutType, CompType, ScaleShiftType>(), ARCH::VC>::Do(
                pgTrue, pgTrue, pgTrue, pgTrue, vVariance0, vVariance1, vVariance2, vVariance3);
            vVariance /= dataPerGroup;
            if (saveStats) {
                mean[offsetStats] = vMean;
                variance[offsetStats] = vVariance;
            }
        }

        float invSqrtVariance0 = 1.0f / std::sqrt(vVariance + eps);
        typename Vector::Vec<CompType, ARCH::VC>::Type invSqrtVarianceVec00;
        typename Vector::Vec<CompType, ARCH::VC>::Type invSqrtVarianceVec01;
        typename Vector::Vec<CompType, ARCH::VC>::Type invSqrtVarianceVec02;
        typename Vector::Vec<CompType, ARCH::VC>::Type invSqrtVarianceVec03;
        Vector::DupFloatN<InType, CompType,
            Vector::GetUsedRegsN<InType, OutType, CompType, ScaleShiftType, float>(), ARCH::VC>::Do(
            invSqrtVariance0, invSqrtVarianceVec00, invSqrtVarianceVec01, invSqrtVarianceVec02, invSqrtVarianceVec03);
        typename Vector::Vec<CompType, ARCH::VC>::Type constants00;
        typename Vector::Vec<CompType, ARCH::VC>::Type constants01;
        typename Vector::Vec<CompType, ARCH::VC>::Type constants02;
        typename Vector::Vec<CompType, ARCH::VC>::Type constants03;
        Vector::DupFloatN<InType, CompType,
            Vector::GetUsedRegsN<InType, OutType, CompType, ScaleShiftType, float>(), ARCH::VC>::Do(
            invSqrtVariance0, constants00, constants01, constants02, constants03);
        typename Vector::Vec<CompType, ARCH::VC>::Type vMeanVect00;
        typename Vector::Vec<CompType, ARCH::VC>::Type vMeanVect01;
        typename Vector::Vec<CompType, ARCH::VC>::Type vMeanVect02;
        typename Vector::Vec<CompType, ARCH::VC>::Type vMeanVect03;
        Vector::DupFloatN<InType, CompType,
            Vector::GetUsedRegsN<InType, OutType, CompType, ScaleShiftType, float>(), ARCH::VC>::Do(
            vMean, vMeanVect00, vMeanVect01, vMeanVect02, vMeanVect03);

        for (SizeType c = 0; c < channelPerGroup; ++c) {
            Service::UnrollFor::Do<1>(innerSize,
                GroupNormCrossFWD<CompType, InType, OutType, ScaleShiftType, ARCH::VC>{},
                src + offset + c * innerSize, scale, shift, startGroup, useScale, useShift,
                innerSize, group, channelPerGroup, c,
                invSqrtVarianceVec00, invSqrtVarianceVec01, invSqrtVarianceVec02, invSqrtVarianceVec03,
                constants00, constants01, constants02, constants03,
                vMeanVect00, vMeanVect01, vMeanVect02, vMeanVect03, dst + offset + c * innerSize);
        }
    }
}

} // GNorm
} // Detail
} // KuDNN

#endif //  KUDNN_GNORM_FWD_DEF_HPP