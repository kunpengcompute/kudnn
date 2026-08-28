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

#ifndef KUDNN_GNORM_CROSS_FWD_DEF_NEON_HPP
#define KUDNN_GNORM_CROSS_FWD_DEF_NEON_HPP

#include "kudnn.hpp"

#include "vector_wrappers.hpp"
#include "kernels/gnorm/kudnn_gnorm_kernels.hpp"
#include "kernels/gnorm/kudnn_gnorm_calculate_struct.hpp"

namespace KuDNN {
namespace Detail {
namespace GNorm {

template <typename CompType, typename InType, typename OutType, typename ScaleShiftType>
void EvaluateCrossAxisNeon(SizeType outerSize, const InType *src, OutType *dst, SizeType groupSize,
    const ScaleShiftType *scale, const ScaleShiftType *shift, float *mean, float *variance, SizeType start,
    bool saveStats, float eps, NormalizationFlags flags, SizeType axisSize, SizeType innerSize) noexcept
{
    bool calculateStats = !static_cast<bool>(flags & NormalizationFlags::USE_GLOBAL_STATS);
    bool useScale = static_cast<bool>(flags & NormalizationFlags::USE_SCALE);
    bool useShift = static_cast<bool>(flags & NormalizationFlags::USE_SHIFT);
    SizeType channelPerGroup = axisSize / groupSize;
    SizeType dataPerGroup = innerSize * channelPerGroup;

    typename Vector::Vec<float, ARCH::NEON>::Type dataPerGroupVec0;
    typename Vector::Vec<float, ARCH::NEON>::Type dataPerGroupVec1;
    typename Vector::Vec<float, ARCH::NEON>::Type dataPerGroupVec2;
    typename Vector::Vec<float, ARCH::NEON>::Type dataPerGroupVec3;
    Vector::DupFloatN<InType, float,
        Vector::GetUsedRegsN<InType, CompType, OutType, ScaleShiftType, float>(), ARCH::NEON>::Do(
        dataPerGroup, dataPerGroupVec0, dataPerGroupVec1, dataPerGroupVec2, dataPerGroupVec3);
    
    SizeType neonLoop = innerSize * channelPerGroup - (innerSize * channelPerGroup %
                        GroupNormCrossFWD<CompType, InType, OutType, ScaleShiftType, ARCH::NEON>::LoopInc());
    SizeType tailLoop = innerSize * channelPerGroup %
                        GroupNormCrossFWD<CompType, InType, OutType, ScaleShiftType, ARCH::NEON>::LoopInc(); // Use VC
    SizeType neonLoop1 = innerSize - (innerSize %
                        GroupNormCrossFWD<CompType, InType, OutType, ScaleShiftType, ARCH::NEON>::LoopInc());
    SizeType tailLoop1 = innerSize %
                        GroupNormCrossFWD<CompType, InType, OutType, ScaleShiftType, ARCH::NEON>::LoopInc(); // Use VC

    for (SizeType group = 0; group < outerSize; ++group) {
        SizeType startGroup = (start + group) % groupSize;
        SizeType offset = group * channelPerGroup * innerSize;
        SizeType offsetStats = group;
        float vMean0 = calculateStats ? 0.0f : mean[offsetStats];
        float vVariance0 = calculateStats ? 0.0f : variance[offsetStats];

        typename Vector::Vec<float, ARCH::NEON>::Type vMean00;
        typename Vector::Vec<float, ARCH::NEON>::Type vMean01;
        typename Vector::Vec<float, ARCH::NEON>::Type vMean02;
        typename Vector::Vec<float, ARCH::NEON>::Type vMean03;
        typename Vector::Vec<float, ARCH::VC>::Type vMean04;
        typename Vector::Vec<float, ARCH::VC>::Type vMean05;
        typename Vector::Vec<float, ARCH::VC>::Type vMean06;
        typename Vector::Vec<float, ARCH::VC>::Type vMean07;
        typename Vector::Vec<float, ARCH::NEON>::Type vVariance00;
        typename Vector::Vec<float, ARCH::NEON>::Type vVariance01;
        typename Vector::Vec<float, ARCH::NEON>::Type vVariance02;
        typename Vector::Vec<float, ARCH::NEON>::Type vVariance03;
        typename Vector::Vec<float, ARCH::VC>::Type vVariance04;
        typename Vector::Vec<float, ARCH::VC>::Type vVariance05;
        typename Vector::Vec<float, ARCH::VC>::Type vVariance06;
        typename Vector::Vec<float, ARCH::VC>::Type vVariance07;
        
        Vector::DupFloatN<InType, float,
            Vector::GetUsedRegsN<CompType, CompType, CompType, ScaleShiftType, float>(), ARCH::NEON>::Do(
            vMean0, vMean00, vMean01, vMean02, vMean03);
        Vector::DupFloatN<InType, float,
            Vector::GetUsedRegsN<CompType, CompType, CompType, ScaleShiftType, float>(), ARCH::NEON>::Do(
            vVariance0, vVariance00, vVariance01, vVariance02, vVariance03);
        Vector::DupFloatN<InType, float,
            Vector::GetUsedRegsN<InType, OutType, CompType, ScaleShiftType, float>(), ARCH::VC>::Do(
            vMean0, vMean04, vMean05, vMean06, vMean07);
        Vector::DupFloatN<InType, float,
            Vector::GetUsedRegsN<InType, OutType, CompType, ScaleShiftType, float>(), ARCH::VC>::Do(
            vVariance0, vVariance04, vVariance05, vVariance06, vVariance07);

        if (calculateStats) {
            svbool_t pgTrue = svptrue_b32();

            Service::UnrollFor::Do<1>(neonLoop,
                GroupNormCrossFWD<CompType, InType, OutType, ScaleShiftType, ARCH::NEON>{},
                src + offset, vMean00, vMean01, vMean02, vMean03);

            Service::UnrollFor::Do<1>(tailLoop,
                GroupNormCrossFWD<CompType, InType, OutType, ScaleShiftType, ARCH::VC>{},
                src + offset + neonLoop, vMean04, vMean05, vMean06, vMean07);
            vMean0 = Vector::AddvN<float,
                Vector::GetUsedRegsN<float, InType, OutType, CompType, ScaleShiftType>(), ARCH::NEON>::Do(
                pgTrue, pgTrue, pgTrue, pgTrue, vMean00, vMean01, vMean02, vMean03);
            float vMean0_ = Vector::AddvN<float,
                Vector::GetUsedRegsN<float, InType, OutType, CompType, ScaleShiftType>(), ARCH::VC>::Do(
                pgTrue, pgTrue, pgTrue, pgTrue, vMean04, vMean05, vMean06, vMean07);
            vMean0 += vMean0_;
            vMean0 /= dataPerGroup;
            
            Vector::DupFloatN<InType, float,
                Vector::GetUsedRegsN<InType, OutType, CompType, ScaleShiftType, float>(), ARCH::NEON>::Do(
                vMean0, vMean00, vMean01, vMean02, vMean03);
            Service::UnrollFor::Do<1>(neonLoop,
                GroupNormCrossFWD<CompType, InType, OutType, ScaleShiftType, ARCH::NEON>{},
                src + offset, vMean00, vMean01, vMean02, vMean03, vVariance00, vVariance01, vVariance02, vVariance03);
            Vector::DupFloatN<InType, float,
                Vector::GetUsedRegsN<InType, OutType, CompType, ScaleShiftType, float>(), ARCH::VC>::Do(
                vMean0, vMean04, vMean05, vMean06, vMean07);
            Service::UnrollFor::Do<1>(tailLoop,
                GroupNormCrossFWD<CompType, InType, OutType, ScaleShiftType, ARCH::VC>{},
                src + offset + neonLoop,
                vMean04, vMean05, vMean06, vMean07, vVariance04, vVariance05, vVariance06, vVariance07);
            vVariance0 = Vector::AddvN<float,
                Vector::GetUsedRegsN<float, InType, OutType, CompType, ScaleShiftType>(), ARCH::NEON>::Do(
                pgTrue, pgTrue, pgTrue, pgTrue, vVariance00, vVariance01, vVariance02, vVariance03);
            float vVariance0_ = Vector::AddvN<float,
                Vector::GetUsedRegsN<float, InType, OutType, CompType, ScaleShiftType>(), ARCH::VC>::Do(
                pgTrue, pgTrue, pgTrue, pgTrue, vVariance04, vVariance05, vVariance06, vVariance07);
            vVariance0 += vVariance0_;
            vVariance0 /= dataPerGroup;
            if (saveStats) {
                mean[offsetStats] = vMean0;
                variance[offsetStats] = vVariance0;
            }
        }

        float invSqrtVariance0 = 1.0f / std::sqrt(vVariance0 + eps);

        typename Vector::Vec<CompType, ARCH::NEON>::Type invSqrtVarianceVec00;
        typename Vector::Vec<CompType, ARCH::NEON>::Type invSqrtVarianceVec01;
        typename Vector::Vec<CompType, ARCH::NEON>::Type invSqrtVarianceVec02;
        typename Vector::Vec<CompType, ARCH::NEON>::Type invSqrtVarianceVec03;
        Vector::DupFloatN<InType, CompType,
            Vector::GetUsedRegsN<InType, OutType, CompType, ScaleShiftType, float>(), ARCH::NEON>::Do(
            invSqrtVariance0, invSqrtVarianceVec00, invSqrtVarianceVec01, invSqrtVarianceVec02, invSqrtVarianceVec03);
        typename Vector::Vec<CompType, ARCH::NEON>::Type constants00;
        typename Vector::Vec<CompType, ARCH::NEON>::Type constants01;
        typename Vector::Vec<CompType, ARCH::NEON>::Type constants02;
        typename Vector::Vec<CompType, ARCH::NEON>::Type constants03;
        Vector::DupFloatN<InType, CompType,
            Vector::GetUsedRegsN<InType, OutType, CompType, ScaleShiftType, float>(), ARCH::NEON>::Do(
            invSqrtVariance0, constants00, constants01, constants02, constants03);
        typename Vector::Vec<CompType, ARCH::NEON>::Type vMeanVect00;
        typename Vector::Vec<CompType, ARCH::NEON>::Type vMeanVect01;
        typename Vector::Vec<CompType, ARCH::NEON>::Type vMeanVect02;
        typename Vector::Vec<CompType, ARCH::NEON>::Type vMeanVect03;
        Vector::DupFloatN<InType, CompType,
            Vector::GetUsedRegsN<InType, OutType, CompType, ScaleShiftType, float>(), ARCH::NEON>::Do(
            vMean0, vMeanVect00, vMeanVect01, vMeanVect02, vMeanVect03);

        // VC tail
        typename Vector::Vec<CompType, ARCH::VC>::Type invSqrtVarianceVec04;
        typename Vector::Vec<CompType, ARCH::VC>::Type invSqrtVarianceVec05;
        typename Vector::Vec<CompType, ARCH::VC>::Type invSqrtVarianceVec06;
        typename Vector::Vec<CompType, ARCH::VC>::Type invSqrtVarianceVec07;
        Vector::DupFloatN<InType, CompType,
            Vector::GetUsedRegsN<InType, OutType, CompType, ScaleShiftType, float>(), ARCH::VC>::Do(
            invSqrtVariance0, invSqrtVarianceVec04, invSqrtVarianceVec05, invSqrtVarianceVec06, invSqrtVarianceVec07);
        typename Vector::Vec<CompType, ARCH::VC>::Type constants04;
        typename Vector::Vec<CompType, ARCH::VC>::Type constants05;
        typename Vector::Vec<CompType, ARCH::VC>::Type constants06;
        typename Vector::Vec<CompType, ARCH::VC>::Type constants07;
        Vector::DupFloatN<InType, CompType,
            Vector::GetUsedRegsN<InType, OutType, CompType, ScaleShiftType, float>(), ARCH::VC>::Do(
            invSqrtVariance0, constants04, constants05, constants06, constants07);
        typename Vector::Vec<CompType, ARCH::VC>::Type vMeanVect04;
        typename Vector::Vec<CompType, ARCH::VC>::Type vMeanVect05;
        typename Vector::Vec<CompType, ARCH::VC>::Type vMeanVect06;
        typename Vector::Vec<CompType, ARCH::VC>::Type vMeanVect07;
        Vector::DupFloatN<InType, CompType,
            Vector::GetUsedRegsN<InType, OutType, CompType, ScaleShiftType, float>(), ARCH::VC>::Do(
            vMean0, vMeanVect04, vMeanVect05, vMeanVect06, vMeanVect07);

        for (SizeType c = 0; c < channelPerGroup; ++c) {
            Service::UnrollFor::Do<1>(neonLoop1,
                GroupNormCrossFWD<CompType, InType, OutType, ScaleShiftType, ARCH::NEON>{},
                src + offset + c * innerSize, scale, shift, startGroup, useScale, useShift,
                innerSize, group, channelPerGroup, c,
                invSqrtVarianceVec00, invSqrtVarianceVec01, invSqrtVarianceVec02, invSqrtVarianceVec03,
                constants00, constants01, constants02, constants03,
                vMeanVect00, vMeanVect01, vMeanVect02, vMeanVect03, dst + offset + c * innerSize);

            Service::UnrollFor::Do<1>(tailLoop1,
                GroupNormCrossFWD<CompType, InType, OutType, ScaleShiftType, ARCH::VC>{},
                src + offset + c * innerSize + neonLoop1, scale, shift, startGroup, useScale, useShift,
                innerSize, group, channelPerGroup, c,
                invSqrtVarianceVec04, invSqrtVarianceVec05, invSqrtVarianceVec06, invSqrtVarianceVec07,
                constants04, constants05, constants06, constants07,
                vMeanVect04, vMeanVect05, vMeanVect06, vMeanVect07, dst + offset + c * innerSize + neonLoop1);
        }
    }
}

} // GNorm
} // Detail
} // KuDNN

#endif //  KUDNN_GNORM_FWD_DEF_HPP