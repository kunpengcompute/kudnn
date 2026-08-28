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

#ifndef KUDNN_LNORM_BWD_CROSS_DEF_HPP
#define KUDNN_LNORM_BWD_CROSS_DEF_HPP

#include "kudnn.hpp"

#include "vector_wrappers.hpp"
#include "kernels/lnorm/kudnn_lnorm_kernels.hpp"
#include "kudnn_lnorm_bwd_cross_layer_def.hpp"

namespace KuDNN {
namespace Detail {
namespace LNorm {

template <typename CompType, typename InType, typename OutType, typename ScaleShiftType>
void EvaluateCrossAxis(SizeType outerSize, const InType *src, const float *mean, const float *variance,
    const OutType *diffDst, const ScaleShiftType *scale, InType *diffSrc, float eps,
    NormalizationFlags flags, SizeType axisSize, SizeType innerSize) noexcept
{
    bool calculateStats = !static_cast<bool>(flags & NormalizationFlags::USE_GLOBAL_STATS);
    bool skipMean = static_cast<bool>(flags & NormalizationFlags::RMS_NORM);
    SizeType loopInc = Vector::UnrollLength<InType, CompType, OutType>();
    for (SizeType ou = 0; ou < outerSize; ++ou) {
        for (SizeType in = 0; in < innerSize; in += loopInc) {
            SizeType statsOffset = ou * innerSize + in;
            svbool_t ldsrcPG0;
            svbool_t ldsrcPG1;
            svbool_t ldsrcPG2;
            svbool_t ldsrcPG3;
            Vector::InitPGN<InType, Vector::GetUsedRegsN<InType, OutType, CompType, ScaleShiftType, float>()>::Do(
                ldsrcPG0, ldsrcPG1, ldsrcPG2, ldsrcPG3, in, innerSize);
            svbool_t lddstPG0;
            svbool_t lddstPG1;
            svbool_t lddstPG2;
            svbool_t lddstPG3;
            Vector::InitPGN<OutType, Vector::GetUsedRegsN<OutType, InType, CompType, ScaleShiftType, float>()>::Do(
                lddstPG0, lddstPG1, lddstPG2, lddstPG3, in, innerSize);
            svbool_t opPG0;
            svbool_t opPG1;
            svbool_t opPG2;
            svbool_t opPG3;
            Vector::InitPGN<CompType, Vector::GetUsedRegsN<CompType, InType, OutType, ScaleShiftType, float>()>::Do(
                opPG0, opPG1, opPG2, opPG3, in, innerSize);
            svbool_t stPG0;
            svbool_t stPG1;
            svbool_t stPG2;
            svbool_t stPG3;
            Vector::InitPGN<InType, Vector::GetUsedRegsN<InType, OutType, CompType, ScaleShiftType, float>()>::Do(
                stPG0, stPG1, stPG2, stPG3, in, innerSize);
            typename Vector::Vec<CompType>::Type mean0;
            typename Vector::Vec<CompType>::Type mean1;
            typename Vector::Vec<CompType>::Type mean2;
            typename Vector::Vec<CompType>::Type mean3;
            typename Vector::Vec<CompType>::Type variance0;
            typename Vector::Vec<CompType>::Type variance1;
            typename Vector::Vec<CompType>::Type variance2;
            typename Vector::Vec<CompType>::Type variance3;
            svbool_t ldStatspg0;
            svbool_t ldStatspg1;
            svbool_t ldStatspg2;
            svbool_t ldStatspg3;
            Vector::InitPGN<float, Vector::GetUsedRegsN<float, InType, CompType, OutType, ScaleShiftType>()>::Do(
                ldStatspg0, ldStatspg1, ldStatspg2, ldStatspg3, in, innerSize);
            if (!skipMean) {
                Vector::LoadN<float, CompType,
                    Vector::GetUsedRegsN<float, InType, CompType, OutType, ScaleShiftType>()>::Do(
                    static_cast<const float *>(mean) + statsOffset, ldStatspg0, ldStatspg1, ldStatspg2, ldStatspg3,
                    mean0, mean1, mean2, mean3);
            } else {
                Vector::DupFloatN<InType, CompType,
                    Vector::GetUsedRegsN<InType, CompType, OutType, ScaleShiftType, float>()>::Do(
                    0.0f, mean0, mean1, mean2, mean3);
            }
            Vector::LoadN<float, CompType,
                Vector::GetUsedRegsN<float, InType, CompType, OutType, ScaleShiftType>()>::Do(
                static_cast<const float *>(variance) + statsOffset,
                ldStatspg0, ldStatspg1, ldStatspg2, ldStatspg3, variance0, variance1, variance2, variance3);
            typename Vector::Vec<CompType>::Type eps0;
            typename Vector::Vec<CompType>::Type eps1;
            typename Vector::Vec<CompType>::Type eps2;
            typename Vector::Vec<CompType>::Type eps3;
            Vector::DupFloatN<InType, CompType,
                Vector::GetUsedRegsN<InType, CompType, OutType, ScaleShiftType, float>()>::Do(
                eps, eps0, eps1, eps2, eps3);
            typename Vector::Vec<CompType>::Type epsVar0;
            typename Vector::Vec<CompType>::Type epsVar1;
            typename Vector::Vec<CompType>::Type epsVar2;
            typename Vector::Vec<CompType>::Type epsVar3;
            Vector::AddXN<CompType, Vector::GetUsedRegsN<CompType, InType, OutType, ScaleShiftType, float>()>::Do(
                opPG0, opPG1, opPG2, opPG3, eps0, eps1, eps2, eps3,
                variance0, variance1, variance2, variance3, epsVar0, epsVar1, epsVar2, epsVar3);
            typename Vector::Vec<CompType>::Type invSqrtVar0;
            typename Vector::Vec<CompType>::Type invSqrtVar1;
            typename Vector::Vec<CompType>::Type invSqrtVar2;
            typename Vector::Vec<CompType>::Type invSqrtVar3;
            Vector::SqrtXN<CompType, Vector::GetUsedRegsN<CompType, InType, OutType, ScaleShiftType, float>()>::Do(
                opPG0, opPG1, opPG2, opPG3, epsVar0,  epsVar1, epsVar2,
                epsVar3, invSqrtVar0, invSqrtVar1, invSqrtVar2, invSqrtVar3);
            typename Vector::Vec<CompType>::Type ones0;
            typename Vector::Vec<CompType>::Type ones1;
            typename Vector::Vec<CompType>::Type ones2;
            typename Vector::Vec<CompType>::Type ones3;
            Vector::DupFloatN<InType, CompType,
                Vector::GetUsedRegsN<InType, CompType, OutType, ScaleShiftType, float>()>::Do(
                1.0f, ones0, ones1, ones2, ones3);
            Vector::DivXN<CompType, Vector::GetUsedRegsN<CompType, InType, OutType, ScaleShiftType, float>()>::Do(
                opPG0, opPG1, opPG2, opPG3, ones0, ones1, ones2, ones3,
                invSqrtVar0, invSqrtVar1, invSqrtVar2, invSqrtVar3,
                invSqrtVar0, invSqrtVar1, invSqrtVar2, invSqrtVar3);

            // ddGamma and ddgammaX init
            typename Vector::Vec<CompType>::Type ddGamma0;
            typename Vector::Vec<CompType>::Type ddGamma1;
            typename Vector::Vec<CompType>::Type ddGamma2;
            typename Vector::Vec<CompType>::Type ddGamma3;
            Vector::DupFloatN<InType, CompType,
                Vector::GetUsedRegsN<InType, CompType, OutType, ScaleShiftType, float>()>::Do(
                0.0f, ddGamma0, ddGamma1, ddGamma2, ddGamma3);
            typename Vector::Vec<CompType>::Type ddGammaX0;
            typename Vector::Vec<CompType>::Type ddGammaX1;
            typename Vector::Vec<CompType>::Type ddGammaX2;
            typename Vector::Vec<CompType>::Type ddGammaX3;
            Vector::DupFloatN<InType, CompType,
                Vector::GetUsedRegsN<InType, CompType, OutType, ScaleShiftType, float>()>::Do(
                0.0f, ddGammaX0, ddGammaX1, ddGammaX2, ddGammaX3);

            if (calculateStats) {
                Service::UnrollFor::Do<1>(axisSize,
                    LayerNormCrossBWDAxisSize<CompType, InType, OutType, ScaleShiftType>{},
                    src + ou * axisSize * innerSize, diffDst + ou * axisSize * innerSize,
                    scale, innerSize, in, ldsrcPG0, ldsrcPG1, ldsrcPG2, ldsrcPG3,
                    lddstPG0, lddstPG1, lddstPG2, lddstPG3, opPG0, opPG1, opPG2, opPG3,
                    mean0, mean1, mean2, mean3, ddGamma0, ddGamma1, ddGamma2, ddGamma3,
                    ddGammaX0, ddGammaX1, ddGammaX2, ddGammaX3);
                Vector::MulXN<CompType, Vector::GetUsedRegsN<CompType, InType, OutType, ScaleShiftType, float>()>::Do(
                    opPG0, opPG1, opPG2, opPG3, ddGammaX0, ddGammaX1,
                    ddGammaX2, ddGammaX3, invSqrtVar0, invSqrtVar1, invSqrtVar2, invSqrtVar3,
                    ddGammaX0, ddGammaX1, ddGammaX2, ddGammaX3);
            }
            Service::UnrollFor::Do<1>(axisSize, LayerNormCrossBWDAxisSize<CompType, InType, OutType, ScaleShiftType>{},
                src + ou * axisSize * innerSize, diffDst + ou * axisSize * innerSize,
                scale, calculateStats, innerSize, in, ldsrcPG0, ldsrcPG1, ldsrcPG2, ldsrcPG3,
                lddstPG0, lddstPG1, lddstPG2, lddstPG3,
                opPG0, opPG1, opPG2, opPG3, stPG0, stPG1, stPG2, stPG3, mean0, mean1, mean2, mean3,
                ddGamma0, ddGamma1, ddGamma2, ddGamma3, ddGammaX0, ddGammaX1, ddGammaX2, ddGammaX3,
                invSqrtVar0, invSqrtVar1, invSqrtVar2, invSqrtVar3, diffSrc + ou * axisSize * innerSize);
        }
    }
}

template <typename CompType, typename InType, typename OutType, typename ScaleShiftType>
void EvaluateScaleShiftCrossAxis(SizeType outerSize, const InType *src, const float *mean, const float *variance,
    const OutType *diffDst, ScaleShiftType *diffScale,
    ScaleShiftType *diffShift, float *bufScale, float *bufShift, float eps, NormalizationFlags flags,
    SizeType axisSizeConst, SizeType innerSize) noexcept
{
    bool useScale = static_cast<bool>(flags & NormalizationFlags::USE_SCALE);
    bool useShift = static_cast<bool>(flags & NormalizationFlags::USE_SHIFT);
    bool skipMean = static_cast<bool>(flags & NormalizationFlags::RMS_NORM);
    if (useScale || useShift) {
        for (SizeType ax = 0; ax < axisSizeConst; ++ax) {
            typename Vector::Vec<CompType>::Type diffGammaConst0;
            typename Vector::Vec<CompType>::Type diffGammaConst1;
            typename Vector::Vec<CompType>::Type diffGammaConst2;
            typename Vector::Vec<CompType>::Type diffGammaConst3;
            Vector::DupFloatN<InType, CompType,
                Vector::GetUsedRegsN<InType, CompType, OutType, ScaleShiftType, float>()>::Do(
                0.0f, diffGammaConst0, diffGammaConst1, diffGammaConst2, diffGammaConst3);
            typename Vector::Vec<CompType>::Type diffBetaConst0;
            typename Vector::Vec<CompType>::Type diffBetaConst1;
            typename Vector::Vec<CompType>::Type diffBetaConst2;
            typename Vector::Vec<CompType>::Type diffBetaConst3;
            Vector::DupFloatN<InType, CompType,
                Vector::GetUsedRegsN<InType, CompType, OutType, ScaleShiftType, float>()>::Do(
                0.0f, diffBetaConst0, diffBetaConst1, diffBetaConst2, diffBetaConst3);
            for (SizeType ou = 0; ou < outerSize; ++ou) {
                Service::UnrollFor::Do<1>(innerSize,
                    LayerNormCrossBWDInnerSize<CompType, InType, OutType, ScaleShiftType>{},
                    src + ou * axisSizeConst * innerSize + ax * innerSize, diffDst +
                    ou * axisSizeConst * innerSize + ax * innerSize,
                    mean + ou * innerSize, variance + ou * innerSize, eps,
                    diffGammaConst0, diffGammaConst1, diffGammaConst2, diffGammaConst3,
                    diffBetaConst0, diffBetaConst1, diffBetaConst2, diffBetaConst3, skipMean);
            }
            svbool_t globalOpPG0;
            svbool_t globalOpPG1;
            svbool_t globalOpPG2;
            svbool_t globalOpPG3;
            Vector::InitPGN<CompType, Vector::GetUsedRegsN<CompType, InType, OutType, ScaleShiftType, float>()>::Do(
                globalOpPG0, globalOpPG1, globalOpPG2, globalOpPG3, 0, innerSize);
            CompType diffGamma = Vector::AddvN<CompType,
                Vector::GetUsedRegsN<CompType, InType, OutType, ScaleShiftType, float>()>::Do(globalOpPG0, globalOpPG1,
                globalOpPG2, globalOpPG3, diffGammaConst0, diffGammaConst1, diffGammaConst2, diffGammaConst3);
            CompType diffBeta = Vector::AddvN<CompType,
                Vector::GetUsedRegsN<CompType, InType, OutType, ScaleShiftType, float>()>::Do(globalOpPG0, globalOpPG1,
                globalOpPG2, globalOpPG3, diffBetaConst0, diffBetaConst1, diffBetaConst2, diffBetaConst3);
            if (diffScale && useScale) {
                bufScale[ax] += diffGamma;
            }
            if (diffShift && useShift) {
                bufShift[ax] += diffBeta;
            }
        }
    }
}

} // LNorm
} // Detail
} // KuDNN

#endif //  KUDNN_LNORM_BWD_CROSS_DEF_HPP