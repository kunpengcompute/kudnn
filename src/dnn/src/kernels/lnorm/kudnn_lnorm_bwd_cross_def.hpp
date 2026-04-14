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

namespace KuDNN {
namespace Detail {
namespace LNorm {

template <typename CompType, typename InType, typename OutType, typename ScaleShiftType>
struct LayerNormCrossBWDInnerSize final {
    static SizeType LoopInc() noexcept
    {
        return Vector::UnrollLength<InType, CompType, OutType, ScaleShiftType, float>();
    }

    static void Do(SizeType loopIdx, SizeType loopRange, const InType *src, const OutType *diffDst,
        const float *mean, const float *variance, float eps,
        typename Vector::Vec<CompType>::Type &vDiffGamma0, typename Vector::Vec<CompType>::Type &vDiffGamma1,
        typename Vector::Vec<CompType>::Type &vDiffGamma2, typename Vector::Vec<CompType>::Type &vDiffGamma3,
        typename Vector::Vec<CompType>::Type &vDiffBeta0, typename Vector::Vec<CompType>::Type &vDiffBeta1,
        typename Vector::Vec<CompType>::Type &vDiffBeta2, typename Vector::Vec<CompType>::Type &vDiffBeta3)

    {
        svbool_t ldsrcPG0;
        svbool_t ldsrcPG1;
        svbool_t ldsrcPG2;
        svbool_t ldsrcPG3;
        Vector::InitPGN<InType, Vector::GetUsedRegsN<InType, OutType, CompType, ScaleShiftType, float>()>::Do(
            ldsrcPG0, ldsrcPG1, ldsrcPG2, ldsrcPG3, loopIdx, loopRange);
        svbool_t lddstPG0;
        svbool_t lddstPG1;
        svbool_t lddstPG2;
        svbool_t lddstPG3;
        Vector::InitPGN<OutType, Vector::GetUsedRegsN<OutType, InType, CompType, ScaleShiftType, float>()>::Do(
            lddstPG0, lddstPG1, lddstPG2, lddstPG3, loopIdx, loopRange);
        svbool_t opPG0;
        svbool_t opPG1;
        svbool_t opPG2;
        svbool_t opPG3;
        Vector::InitPGN<CompType, Vector::GetUsedRegsN<CompType, InType, OutType, ScaleShiftType, float>()>::Do(
            opPG0, opPG1, opPG2, opPG3, loopIdx, loopRange);
        // stats load
        SizeType statsOffset = loopIdx;
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
        Vector::InitPGN<float,
            Vector::GetUsedRegsN<float, InType, CompType, OutType, ScaleShiftType>()>::Do(
            ldStatspg0, ldStatspg1, ldStatspg2, ldStatspg3, loopIdx, loopRange);
        Vector::LoadN<float, CompType,
            Vector::GetUsedRegsN<float, InType, CompType, OutType, ScaleShiftType>()>::Do(
            static_cast<const float *>(mean) + statsOffset, ldStatspg0, ldStatspg1,
            ldStatspg2, ldStatspg3, mean0, mean1, mean2, mean3);
        Vector::LoadN<float, CompType,
            Vector::GetUsedRegsN<float, InType, CompType, OutType, ScaleShiftType>()>::Do(
            static_cast<const float *>(variance) + statsOffset, ldStatspg0, ldStatspg1,
            ldStatspg2, ldStatspg3, variance0, variance1, variance2, variance3);
        // src and diff dst load
        SizeType offset = loopIdx;
        typename Vector::Vec<CompType>::Type src0;
        typename Vector::Vec<CompType>::Type src1;
        typename Vector::Vec<CompType>::Type src2;
        typename Vector::Vec<CompType>::Type src3;
        Vector::LoadN<InType, CompType, Vector::GetUsedRegsN<InType, CompType, OutType, ScaleShiftType, float>()>::Do(
            src + offset, ldsrcPG0, ldsrcPG1, ldsrcPG2, ldsrcPG3, src0, src1, src2, src3);
        typename Vector::Vec<CompType>::Type diffDst0;
        typename Vector::Vec<CompType>::Type diffDst1;
        typename Vector::Vec<CompType>::Type diffDst2;
        typename Vector::Vec<CompType>::Type diffDst3;
        Vector::LoadN<OutType, CompType, Vector::GetUsedRegsN<OutType, InType, CompType, ScaleShiftType, float>()>::Do(
            diffDst + offset, lddstPG0, lddstPG1, lddstPG2, lddstPG3, diffDst0, diffDst1, diffDst2, diffDst3);
        // eps load
        typename Vector::Vec<CompType>::Type eps0;
        typename Vector::Vec<CompType>::Type eps1;
        typename Vector::Vec<CompType>::Type eps2;
        typename Vector::Vec<CompType>::Type eps3;
        Vector::DupFloatN<InType, CompType,
            Vector::GetUsedRegsN<InType, CompType, OutType, ScaleShiftType, float>()>::Do(
            eps, eps0, eps1, eps2, eps3);
        // inv sqrt variance computation
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
            opPG0, opPG1, opPG2, opPG3, epsVar0, epsVar1,
            epsVar2, epsVar3, invSqrtVar0, invSqrtVar1, invSqrtVar2, invSqrtVar3);
        typename Vector::Vec<CompType>::Type ones0;
        typename Vector::Vec<CompType>::Type ones1;
        typename Vector::Vec<CompType>::Type ones2;
        typename Vector::Vec<CompType>::Type ones3;
        Vector::DupFloatN<InType, CompType,
            Vector::GetUsedRegsN<InType, CompType, OutType, ScaleShiftType, float>()>::Do(
            1.0f, ones0, ones1, ones2, ones3);
        Vector::DivXN<CompType, Vector::GetUsedRegsN<CompType, InType, OutType, ScaleShiftType, float>()>::Do(
            opPG0, opPG1, opPG2, opPG3, ones0, ones1, ones2, ones3, invSqrtVar0, invSqrtVar1, invSqrtVar2, invSqrtVar3,
            invSqrtVar0, invSqrtVar1, invSqrtVar2, invSqrtVar3);
        // diff gamma evaluation
        typename Vector::Vec<CompType>::Type tmp0;
        typename Vector::Vec<CompType>::Type tmp1;
        typename Vector::Vec<CompType>::Type tmp2;
        typename Vector::Vec<CompType>::Type tmp3;
        Vector::SubXN<CompType, Vector::GetUsedRegsN<CompType, InType, OutType, ScaleShiftType, float>()>::Do(
            opPG0, opPG1, opPG2, opPG3, src0, src1, src2, src3,
            mean0, mean1, mean2, mean3, tmp0, tmp1, tmp2, tmp3);
        Vector::MulXN<CompType, Vector::GetUsedRegsN<CompType, InType, OutType, ScaleShiftType, float>()>::Do(
            opPG0, opPG1, opPG2, opPG3, tmp0, tmp1, tmp2, tmp3,
            diffDst0, diffDst1, diffDst2, diffDst3, tmp0, tmp1, tmp2, tmp3);
        Vector::MulXN<CompType, Vector::GetUsedRegsN<CompType, InType, OutType, ScaleShiftType, float>()>::Do(
            opPG0, opPG1, opPG2, opPG3, tmp0, tmp1, tmp2, tmp3,
            invSqrtVar0, invSqrtVar1, invSqrtVar2, invSqrtVar3, tmp0, tmp1, tmp2, tmp3);
        Vector::AddMN<CompType, Vector::GetUsedRegsN<CompType, InType, OutType, ScaleShiftType, float>()>::Do(
            opPG0, opPG1, opPG2, opPG3, vDiffGamma0, vDiffGamma1, vDiffGamma2, vDiffGamma3, tmp0, tmp1, tmp2, tmp3,
            vDiffGamma0, vDiffGamma1, vDiffGamma2, vDiffGamma3);
        Vector::AddMN<CompType, Vector::GetUsedRegsN<CompType, InType, OutType, ScaleShiftType, float>()>::Do(
            opPG0, opPG1, opPG2, opPG3, vDiffBeta0, vDiffBeta1, vDiffBeta2, vDiffBeta3,
            diffDst0, diffDst1, diffDst2, diffDst3, vDiffBeta0, vDiffBeta1, vDiffBeta2, vDiffBeta3);
    }
};

template <typename CompType, typename InType, typename OutType, typename ScaleShiftType>
struct LayerNormCrossBWDAxisSize final {
    static SizeType LoopInc() noexcept
    {
        return 1;
    }
    
    static void Do(SizeType loopIdx, SizeType loopRange, const InType *src,
        const OutType *diffDst, const ScaleShiftType *scale,
        SizeType innerSize, SizeType in,
        svbool_t ldsrcPG0, svbool_t ldsrcPG1, svbool_t ldsrcPG2, svbool_t ldsrcPG3,
        svbool_t lddstPG0, svbool_t lddstPG1, svbool_t lddstPG2, svbool_t lddstPG3,
        svbool_t opPG0, svbool_t opPG1, svbool_t opPG2, svbool_t opPG3,
        typename Vector::Vec<CompType>::Type &mean0, typename Vector::Vec<CompType>::Type &mean1,
        typename Vector::Vec<CompType>::Type &mean2, typename Vector::Vec<CompType>::Type &mean3,
        typename Vector::Vec<CompType>::Type &ddGamma0, typename Vector::Vec<CompType>::Type &ddGamma1,
        typename Vector::Vec<CompType>::Type &ddGamma2, typename Vector::Vec<CompType>::Type &ddGamma3,
        typename Vector::Vec<CompType>::Type &ddGammaX0, typename Vector::Vec<CompType>::Type &ddGammaX1,
        typename Vector::Vec<CompType>::Type &ddGammaX2, typename Vector::Vec<CompType>::Type &ddGammaX3)
    {
        Service::Unused(loopRange);
        // scale init
        const CompType gamma = scale ? Vector::Convert<ScaleShiftType, CompType>::Do(scale[loopIdx]) : 1.0f;
        typename Vector::Vec<CompType>::Type gamma0;
        typename Vector::Vec<CompType>::Type gamma1;
        typename Vector::Vec<CompType>::Type gamma2;
        typename Vector::Vec<CompType>::Type gamma3;
        Vector::DupFloatN<InType, CompType,
            Vector::GetUsedRegsN<InType, CompType, OutType, ScaleShiftType, float>()>::Do(
            gamma, gamma0, gamma1, gamma2, gamma3);
        SizeType offset = loopIdx * innerSize + in;
        // diff dst load
        typename Vector::Vec<CompType>::Type diffDst0;
        typename Vector::Vec<CompType>::Type diffDst1;
        typename Vector::Vec<CompType>::Type diffDst2;
        typename Vector::Vec<CompType>::Type diffDst3;
        Vector::LoadN<OutType, CompType, Vector::GetUsedRegsN<OutType, InType, CompType, ScaleShiftType, float>()>::Do(
            diffDst + offset, lddstPG0, lddstPG1, lddstPG2, lddstPG3, diffDst0, diffDst1, diffDst2, diffDst3);
        // src load
        typename Vector::Vec<CompType>::Type src0;
        typename Vector::Vec<CompType>::Type src1;
        typename Vector::Vec<CompType>::Type src2;
        typename Vector::Vec<CompType>::Type src3;
        Vector::LoadN<InType, CompType, Vector::GetUsedRegsN<InType, CompType, OutType, ScaleShiftType, float>()>::Do(
            src + offset, ldsrcPG0, ldsrcPG1, ldsrcPG2, ldsrcPG3, src0, src1, src2, src3);
        // ddGamma update
        typename Vector::Vec<CompType>::Type tmp0;
        typename Vector::Vec<CompType>::Type tmp1;
        typename Vector::Vec<CompType>::Type tmp2;
        typename Vector::Vec<CompType>::Type tmp3;
        Vector::MulXN<CompType, Vector::GetUsedRegsN<CompType, InType, OutType, ScaleShiftType, float>()>::Do(
            opPG0, opPG1, opPG2, opPG3, diffDst0, diffDst1,
            diffDst2, diffDst3, gamma0, gamma1, gamma2, gamma3, tmp0, tmp1, tmp2, tmp3);
        Vector::AddXN<CompType, Vector::GetUsedRegsN<CompType, InType, OutType, ScaleShiftType, float>()>::Do(
            opPG0, opPG1, opPG2, opPG3, ddGamma0, ddGamma1,
            ddGamma2, ddGamma3, tmp0, tmp1, tmp2, tmp3, ddGamma0, ddGamma1, ddGamma2, ddGamma3);
        // ddGammaX update
        typename Vector::Vec<CompType>::Type multiplier0;
        typename Vector::Vec<CompType>::Type multiplier1;
        typename Vector::Vec<CompType>::Type multiplier2;
        typename Vector::Vec<CompType>::Type multiplier3;
        Vector::SubXN<CompType, Vector::GetUsedRegsN<CompType, InType, OutType, ScaleShiftType, float>()>::Do(
            opPG0, opPG1, opPG2, opPG3, src0, src1, src2, src3,
            mean0, mean1, mean2, mean3, multiplier0, multiplier1, multiplier2, multiplier3);
        Vector::MulXN<CompType, Vector::GetUsedRegsN<CompType, InType, OutType, ScaleShiftType, float>()>::Do(
            opPG0, opPG1, opPG2, opPG3, tmp0, tmp1, tmp2, tmp3,
            multiplier0, multiplier1, multiplier2, multiplier3, tmp0, tmp1, tmp2, tmp3);
        Vector::AddXN<CompType, Vector::GetUsedRegsN<CompType, InType, OutType, ScaleShiftType, float>()>::Do(
            opPG0, opPG1, opPG2, opPG3, ddGammaX0, ddGammaX1,
            ddGammaX2, ddGammaX3, tmp0, tmp1, tmp2, tmp3, ddGammaX0, ddGammaX1, ddGammaX2, ddGammaX3);
    }

    static void DoCalcStat(SizeType loopRange, const InType *src, SizeType offset, svbool_t opPG0,
        svbool_t opPG1, svbool_t opPG2, svbool_t opPG3, svbool_t ldsrcPG0, svbool_t ldsrcPG1, svbool_t ldsrcPG2,
        svbool_t ldsrcPG3, typename Vector::Vec<CompType>::Type &mean0, typename Vector::Vec<CompType>::Type &mean1,
        typename Vector::Vec<CompType>::Type &mean2, typename Vector::Vec<CompType>::Type &mean3,
        typename Vector::Vec<CompType>::Type &ddGamma0, typename Vector::Vec<CompType>::Type &ddGamma1,
        typename Vector::Vec<CompType>::Type &ddGamma2, typename Vector::Vec<CompType>::Type &ddGamma3,
        typename Vector::Vec<CompType>::Type &ddGammaX0, typename Vector::Vec<CompType>::Type &ddGammaX1,
        typename Vector::Vec<CompType>::Type &ddGammaX2, typename Vector::Vec<CompType>::Type &ddGammaX3,
        typename Vector::Vec<CompType>::Type &invSqrtVar0, typename Vector::Vec<CompType>::Type &invSqrtVar1,
        typename Vector::Vec<CompType>::Type &invSqrtVar2, typename Vector::Vec<CompType>::Type &invSqrtVar3,
        typename Vector::Vec<CompType>::Type &dSrcVec0, typename Vector::Vec<CompType>::Type &dSrcVec1,
        typename Vector::Vec<CompType>::Type &dSrcVec2, typename Vector::Vec<CompType>::Type &dSrcVec3)
    {
        typename Vector::Vec<CompType>::Type axisSize0;
        typename Vector::Vec<CompType>::Type axisSize1;
        typename Vector::Vec<CompType>::Type axisSize2;
        typename Vector::Vec<CompType>::Type axisSize3;
        Vector::DupFloatN<InType, CompType, Vector::GetUsedRegsN<InType, CompType, OutType,
            ScaleShiftType, float>()>::Do(loopRange, axisSize0, axisSize1, axisSize2, axisSize3);
        typename Vector::Vec<CompType>::Type normDdGamma0;
        typename Vector::Vec<CompType>::Type normDdGamma1;
        typename Vector::Vec<CompType>::Type normDdGamma2;
        typename Vector::Vec<CompType>::Type normDdGamma3;
        Vector::DivXN<CompType, Vector::GetUsedRegsN<CompType, InType, OutType, ScaleShiftType, float>()>::Do(
            opPG0, opPG1, opPG2, opPG3, ddGamma0, ddGamma1, ddGamma2, ddGamma3,
            axisSize0, axisSize1, axisSize2, axisSize3, normDdGamma0, normDdGamma1, normDdGamma2, normDdGamma3);
        Vector::SubXN<CompType, Vector::GetUsedRegsN<CompType, InType, OutType, ScaleShiftType, float>()>::Do(
            opPG0, opPG1, opPG2, opPG3, dSrcVec0, dSrcVec1, dSrcVec2, dSrcVec3, normDdGamma0, normDdGamma1,
            normDdGamma2, normDdGamma3, dSrcVec0, dSrcVec1, dSrcVec2, dSrcVec3);
        // src load
        typename Vector::Vec<CompType>::Type src0;
        typename Vector::Vec<CompType>::Type src1;
        typename Vector::Vec<CompType>::Type src2;
        typename Vector::Vec<CompType>::Type src3;
        Vector::LoadN<InType, CompType, Vector::GetUsedRegsN<InType, CompType, OutType, ScaleShiftType, float>()>::Do(
            src + offset, ldsrcPG0, ldsrcPG1, ldsrcPG2, ldsrcPG3, src0, src1, src2, src3);

        typename Vector::Vec<CompType>::Type tmp0;
        typename Vector::Vec<CompType>::Type tmp1;
        typename Vector::Vec<CompType>::Type tmp2;
        typename Vector::Vec<CompType>::Type tmp3;
        Vector::SubXN<CompType, Vector::GetUsedRegsN<CompType, InType, OutType, ScaleShiftType, float>()>::Do(
            opPG0, opPG1, opPG2, opPG3, src0, src1, src2, src3,
            mean0, mean1, mean2, mean3, tmp0, tmp1, tmp2, tmp3);
        Vector::MulMN<CompType, Vector::GetUsedRegsN<CompType, InType, OutType, ScaleShiftType, float>()>::Do(
            opPG0, opPG1, opPG2, opPG3, tmp0, tmp1, tmp2, tmp3,
            ddGammaX0, ddGammaX1, ddGammaX2, ddGammaX3, tmp0, tmp1, tmp2, tmp3);
        Vector::MulMN<CompType, Vector::GetUsedRegsN<CompType, InType, OutType, ScaleShiftType, float>()>::Do(
            opPG0, opPG1, opPG2, opPG3, tmp0, tmp1, tmp2, tmp3,
            invSqrtVar0, invSqrtVar1, invSqrtVar2, invSqrtVar3, tmp0, tmp1, tmp2, tmp3);
        typename Vector::Vec<CompType>::Type invAxis0;
        typename Vector::Vec<CompType>::Type invAxis1;
        typename Vector::Vec<CompType>::Type invAxis2;
        typename Vector::Vec<CompType>::Type invAxis3;
        Vector::DupFloatN<InType, CompType,
            Vector::GetUsedRegsN<InType, CompType, OutType, ScaleShiftType, float>()>::Do(
            1.0f / loopRange, invAxis0, invAxis1, invAxis2, invAxis3);
        Vector::MulMN<CompType, Vector::GetUsedRegsN<CompType, InType, OutType, ScaleShiftType, float>()>::Do(
            opPG0, opPG1, opPG2, opPG3, tmp0, tmp1, tmp2, tmp3,
            invAxis0, invAxis1, invAxis2, invAxis3, tmp0, tmp1, tmp2, tmp3);
        Vector::SubXN<CompType, Vector::GetUsedRegsN<CompType, InType, OutType, ScaleShiftType, float>()>::Do(
            opPG0, opPG1, opPG2, opPG3, dSrcVec0, dSrcVec1,
            dSrcVec2, dSrcVec3,  tmp0, tmp1, tmp2, tmp3, dSrcVec0, dSrcVec1, dSrcVec2, dSrcVec3);
    }

    static void Do(SizeType loopIdx, SizeType loopRange, const InType *src,
        const OutType *diffDst, const ScaleShiftType *scale, bool calculateStats,
        SizeType innerSize, SizeType in,
        svbool_t ldsrcPG0, svbool_t ldsrcPG1, svbool_t ldsrcPG2, svbool_t ldsrcPG3,
        svbool_t lddstPG0, svbool_t lddstPG1, svbool_t lddstPG2, svbool_t lddstPG3,
        svbool_t opPG0, svbool_t opPG1, svbool_t opPG2, svbool_t opPG3,
        svbool_t stPG0, svbool_t stPG1, svbool_t stPG2, svbool_t stPG3,
        typename Vector::Vec<CompType>::Type &mean0, typename Vector::Vec<CompType>::Type &mean1,
        typename Vector::Vec<CompType>::Type &mean2, typename Vector::Vec<CompType>::Type &mean3,
        typename Vector::Vec<CompType>::Type &ddGamma0, typename Vector::Vec<CompType>::Type &ddGamma1,
        typename Vector::Vec<CompType>::Type &ddGamma2, typename Vector::Vec<CompType>::Type &ddGamma3,
        typename Vector::Vec<CompType>::Type &ddGammaX0, typename Vector::Vec<CompType>::Type &ddGammaX1,
        typename Vector::Vec<CompType>::Type &ddGammaX2, typename Vector::Vec<CompType>::Type &ddGammaX3,
        typename Vector::Vec<CompType>::Type &invSqrtVar0, typename Vector::Vec<CompType>::Type &invSqrtVar1,
        typename Vector::Vec<CompType>::Type &invSqrtVar2, typename Vector::Vec<CompType>::Type &invSqrtVar3,
        InType *diffSrc)
    {
        SizeType offset = loopIdx * innerSize + in;
        const CompType gamma = scale ? Vector::Convert<ScaleShiftType, CompType>::Do(scale[loopIdx]) : 1.0f;
        typename Vector::Vec<CompType>::Type gamma0;
        typename Vector::Vec<CompType>::Type gamma1;
        typename Vector::Vec<CompType>::Type gamma2;
        typename Vector::Vec<CompType>::Type gamma3;
        Vector::DupFloatN<InType, CompType,
            Vector::GetUsedRegsN<InType, CompType, OutType, ScaleShiftType, float>()>::Do(
            gamma, gamma0, gamma1, gamma2, gamma3);
        typename Vector::Vec<CompType>::Type diffDst0;
        typename Vector::Vec<CompType>::Type diffDst1;
        typename Vector::Vec<CompType>::Type diffDst2;
        typename Vector::Vec<CompType>::Type diffDst3;
        Vector::LoadN<OutType, CompType, Vector::GetUsedRegsN<OutType, InType, CompType, ScaleShiftType, float>()>::Do(
            diffDst + offset, lddstPG0, lddstPG1, lddstPG2, lddstPG3, diffDst0, diffDst1, diffDst2, diffDst3);
        typename Vector::Vec<CompType>::Type dSrcVec0;
        typename Vector::Vec<CompType>::Type dSrcVec1;
        typename Vector::Vec<CompType>::Type dSrcVec2;
        typename Vector::Vec<CompType>::Type dSrcVec3;
        Vector::MulMN<CompType, Vector::GetUsedRegsN<CompType, InType, OutType, ScaleShiftType, float>()>::Do(
            opPG0, opPG1, opPG2, opPG3, diffDst0, diffDst1,
            diffDst2, diffDst3, gamma0, gamma1, gamma2, gamma3, dSrcVec0, dSrcVec1, dSrcVec2, dSrcVec3);
        if (calculateStats) {
            DoCalcStat(loopRange, src, offset, opPG0, opPG1, opPG2, opPG3, ldsrcPG0, ldsrcPG1, ldsrcPG2, ldsrcPG3,
                mean0, mean1, mean2, mean3, ddGamma0, ddGamma1, ddGamma2, ddGamma3, ddGammaX0, ddGammaX1, ddGammaX2,
                ddGammaX3, invSqrtVar0, invSqrtVar1, invSqrtVar2, invSqrtVar3, dSrcVec0, dSrcVec1, dSrcVec2, dSrcVec3);
        }
        Vector::MulMN<CompType, Vector::GetUsedRegsN<CompType, InType, OutType, ScaleShiftType, float>()>::Do(
            opPG0, opPG1, opPG2, opPG3, dSrcVec0, dSrcVec1,
            dSrcVec2, dSrcVec3, invSqrtVar0, invSqrtVar1, invSqrtVar2, invSqrtVar3,
            dSrcVec0, dSrcVec1, dSrcVec2, dSrcVec3);
        Vector::StoreN<InType, CompType,
            Vector::GetUsedRegsN<InType, OutType, CompType, ScaleShiftType, float>()>::Do(
            stPG0, stPG1, stPG2, stPG3, dSrcVec0, dSrcVec1, dSrcVec2, dSrcVec3, diffSrc + offset);
    }
};

template <typename CompType, typename InType, typename OutType, typename ScaleShiftType>
void EvaluateCrossAxis(SizeType outerSize, const InType *src, const float *mean, const float *variance,
    const OutType *diffDst, const ScaleShiftType *scale, InType *diffSrc, float eps,
    NormalizationFlags flags, SizeType axisSize, SizeType innerSize) noexcept
{
    bool calculateStats = !static_cast<bool>(flags & NormalizationFlags::USE_GLOBAL_STATS);
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
            Vector::InitPGN<InType,
                Vector::GetUsedRegsN<InType, OutType, CompType, ScaleShiftType, float>()>::Do(
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
            Vector::InitPGN<float,
                Vector::GetUsedRegsN<float, InType, CompType, OutType, ScaleShiftType>()>::Do(
                ldStatspg0, ldStatspg1, ldStatspg2, ldStatspg3, in, innerSize);
            Vector::LoadN<float, CompType,
                Vector::GetUsedRegsN<float, InType, CompType, OutType, ScaleShiftType>()>::Do(
                static_cast<const float *>(mean) + statsOffset, ldStatspg0, ldStatspg1, ldStatspg2, ldStatspg3,
                mean0, mean1, mean2, mean3);
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
                invSqrtVar0, invSqrtVar1, invSqrtVar2, invSqrtVar3,
                diffSrc + ou * axisSize * innerSize);
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
                    diffBetaConst0, diffBetaConst1, diffBetaConst2, diffBetaConst3);
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