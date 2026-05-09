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

namespace KuDNN {
namespace Detail {
namespace LNorm {

template <typename CompType, typename InType, typename OutType, typename ScaleShiftType>
struct LayerNormLastBWD final {
    static SizeType LoopInc() noexcept
    {
        return Vector::UnrollLength<InType, CompType, OutType, ScaleShiftType, float>();
    }

    static void DoScale(SizeType loopIdx, SizeType loopRange, float *bufScale, svbool_t &scaleShiftPG0,
                        svbool_t &scaleShiftPG1, svbool_t &scaleShiftPG2, svbool_t &scaleShiftPG3, svbool_t &opPG0,
                        svbool_t &opPG1, svbool_t &opPG2, svbool_t &opPG3,
                        typename Vector::Vec<CompType>::Type &multiplierC0,
                        typename Vector::Vec<CompType>::Type &multiplierC1,
                        typename Vector::Vec<CompType>::Type &multiplierC2,
                        typename Vector::Vec<CompType>::Type &multiplierC3)
    {
        typename Vector::Vec<CompType>::Type diffGamma0;
        typename Vector::Vec<CompType>::Type diffGamma1;
        typename Vector::Vec<CompType>::Type diffGamma2;
        typename Vector::Vec<CompType>::Type diffGamma3;
        Vector::LoadN<float, CompType, Vector::GetUsedRegsN<float, InType, CompType, OutType, ScaleShiftType>()>::Do(
            bufScale + loopIdx, scaleShiftPG0, scaleShiftPG1, scaleShiftPG2, scaleShiftPG3, diffGamma0, diffGamma1,
            diffGamma2, diffGamma3);
        Vector::AddXN<CompType, Vector::GetUsedRegsN<CompType, InType, OutType, ScaleShiftType, float>()>::Do(
            opPG0, opPG1, opPG2, opPG3, diffGamma0, diffGamma1, diffGamma2, diffGamma3, multiplierC0, multiplierC1,
            multiplierC2, multiplierC3, diffGamma0, diffGamma1, diffGamma2, diffGamma3);
        Vector::InitPGN<float, Vector::GetUsedRegsN<float, InType, CompType, OutType, ScaleShiftType>()>::Do(
            scaleShiftPG0, scaleShiftPG1, scaleShiftPG2, scaleShiftPG3, loopIdx, loopRange);
        Vector::StoreN<float, CompType, Vector::GetUsedRegsN<float, InType, CompType, OutType, ScaleShiftType>()>::Do(
            scaleShiftPG0, scaleShiftPG1, scaleShiftPG2, scaleShiftPG3, diffGamma0, diffGamma1, diffGamma2, diffGamma3,
            bufScale + loopIdx);
    }

    static void DoShift(SizeType loopIdx, SizeType loopRange, float *bufShift, svbool_t &scaleShiftPG0,
                        svbool_t &scaleShiftPG1, svbool_t &scaleShiftPG2, svbool_t &scaleShiftPG3, svbool_t &opPG0,
                        svbool_t &opPG1, svbool_t &opPG2, svbool_t &opPG3,
                        typename Vector::Vec<CompType>::Type &diffDst0, typename Vector::Vec<CompType>::Type &diffDst1,
                        typename Vector::Vec<CompType>::Type &diffDst2, typename Vector::Vec<CompType>::Type &diffDst3)
    {
        typename Vector::Vec<CompType>::Type diffBeta0;
        typename Vector::Vec<CompType>::Type diffBeta1;
        typename Vector::Vec<CompType>::Type diffBeta2;
        typename Vector::Vec<CompType>::Type diffBeta3;
        Vector::LoadN<float, CompType, Vector::GetUsedRegsN<float, InType, CompType, OutType, ScaleShiftType>()>::Do(
            bufShift + loopIdx, scaleShiftPG0, scaleShiftPG1, scaleShiftPG2, scaleShiftPG3, diffBeta0, diffBeta1,
            diffBeta2, diffBeta3);
        Vector::AddXN<CompType, Vector::GetUsedRegsN<CompType, InType, OutType, ScaleShiftType, float>()>::Do(
            opPG0, opPG1, opPG2, opPG3, diffBeta0, diffBeta1, diffBeta2, diffBeta3, diffDst0, diffDst1, diffDst2,
            diffDst3, diffBeta0, diffBeta1, diffBeta2, diffBeta3);
        Vector::InitPGN<float, Vector::GetUsedRegsN<float, InType, CompType, OutType, ScaleShiftType>()>::Do(
            scaleShiftPG0, scaleShiftPG1, scaleShiftPG2, scaleShiftPG3, loopIdx, loopRange);
        Vector::StoreN<float, CompType, Vector::GetUsedRegsN<float, InType, CompType, OutType, ScaleShiftType>()>::Do(
            scaleShiftPG0, scaleShiftPG1, scaleShiftPG2, scaleShiftPG3, diffBeta0, diffBeta1, diffBeta2, diffBeta3,
            bufShift + loopIdx);
    }

    static void Do(SizeType loopIdx, SizeType loopRange, const InType *src, const OutType *diffDst, bool useScale,
                   bool useShift, float *bufScale, float *bufShift, typename Vector::Vec<CompType>::Type &mean0,
                   typename Vector::Vec<CompType>::Type &mean1, typename Vector::Vec<CompType>::Type &mean2,
                   typename Vector::Vec<CompType>::Type &mean3, typename Vector::Vec<CompType>::Type &invSqrtVar0,
                   typename Vector::Vec<CompType>::Type &invSqrtVar1, typename Vector::Vec<CompType>::Type &invSqrtVar2,
                   typename Vector::Vec<CompType>::Type &invSqrtVar3)
    {
        typename Vector::Vec<CompType>::Type src0;
        typename Vector::Vec<CompType>::Type src1;
        typename Vector::Vec<CompType>::Type src2;
        typename Vector::Vec<CompType>::Type src3;
        svbool_t ldsrcPG0;
        svbool_t ldsrcPG1;
        svbool_t ldsrcPG2;
        svbool_t ldsrcPG3;
        Vector::InitPGN<InType, Vector::GetUsedRegsN<InType, OutType, CompType, ScaleShiftType, float>()>::Do(
            ldsrcPG0, ldsrcPG1, ldsrcPG2, ldsrcPG3, loopIdx, loopRange);
        Vector::LoadN<InType, CompType, Vector::GetUsedRegsN<InType, CompType, OutType, ScaleShiftType, float>()>::Do(
            src + loopIdx, ldsrcPG0, ldsrcPG1, ldsrcPG2, ldsrcPG3, src0, src1, src2, src3);

        typename Vector::Vec<CompType>::Type diffDst0;
        typename Vector::Vec<CompType>::Type diffDst1;
        typename Vector::Vec<CompType>::Type diffDst2;
        typename Vector::Vec<CompType>::Type diffDst3;
        svbool_t lddstPG0;
        svbool_t lddstPG1;
        svbool_t lddstPG2;
        svbool_t lddstPG3;
        Vector::InitPGN<OutType, Vector::GetUsedRegsN<OutType, InType, CompType, ScaleShiftType, float>()>::Do(
            lddstPG0, lddstPG1, lddstPG2, lddstPG3, loopIdx, loopRange);
        Vector::LoadN<OutType, CompType, Vector::GetUsedRegsN<OutType, InType, CompType, ScaleShiftType, float>()>::Do(
            diffDst + loopIdx, lddstPG0, lddstPG1, lddstPG2, lddstPG3, diffDst0, diffDst1, diffDst2, diffDst3);

        svbool_t opPG0;
        svbool_t opPG1;
        svbool_t opPG2;
        svbool_t opPG3;
        Vector::InitPGN<CompType, Vector::GetUsedRegsN<CompType, InType, OutType, ScaleShiftType, float>()>::Do(
            opPG0, opPG1, opPG2, opPG3, loopIdx, loopRange);
        typename Vector::Vec<CompType>::Type multiplierA0;
        typename Vector::Vec<CompType>::Type multiplierA1;
        typename Vector::Vec<CompType>::Type multiplierA2;
        typename Vector::Vec<CompType>::Type multiplierA3;
        Vector::SubMN<CompType, Vector::GetUsedRegsN<CompType, InType, OutType, ScaleShiftType, float>()>::Do(
            opPG0, opPG1, opPG2, opPG3, src0, src1, src2, src3, mean0, mean1, mean2, mean3, multiplierA0, multiplierA1,
            multiplierA2, multiplierA3);
        typename Vector::Vec<CompType>::Type multiplierB0;
        typename Vector::Vec<CompType>::Type multiplierB1;
        typename Vector::Vec<CompType>::Type multiplierB2;
        typename Vector::Vec<CompType>::Type multiplierB3;
        Vector::MulMN<CompType, Vector::GetUsedRegsN<CompType, InType, OutType, ScaleShiftType, float>()>::Do(
            opPG0, opPG1, opPG2, opPG3, diffDst0, diffDst1, diffDst2, diffDst3, multiplierA0, multiplierA1,
            multiplierA2, multiplierA3, multiplierB0, multiplierB1, multiplierB2, multiplierB3);
        typename Vector::Vec<CompType>::Type multiplierC0;
        typename Vector::Vec<CompType>::Type multiplierC1;
        typename Vector::Vec<CompType>::Type multiplierC2;
        typename Vector::Vec<CompType>::Type multiplierC3;
        Vector::MulMN<CompType, Vector::GetUsedRegsN<CompType, InType, OutType, ScaleShiftType, float>()>::Do(
            opPG0, opPG1, opPG2, opPG3, multiplierB0, multiplierB1, multiplierB2, multiplierB3, invSqrtVar0,
            invSqrtVar1, invSqrtVar2, invSqrtVar3, multiplierC0, multiplierC1, multiplierC2, multiplierC3);
        svbool_t scaleShiftPG0;
        svbool_t scaleShiftPG1;
        svbool_t scaleShiftPG2;
        svbool_t scaleShiftPG3;
        Vector::InitPGN<float, Vector::GetUsedRegsN<float, InType, CompType, OutType, ScaleShiftType>()>::Do(
            scaleShiftPG0, scaleShiftPG1, scaleShiftPG2, scaleShiftPG3, loopIdx, loopRange);
        if (useScale) {
            DoScale(loopIdx, loopRange, bufScale, scaleShiftPG0, scaleShiftPG1, scaleShiftPG2, scaleShiftPG3, opPG0,
                    opPG1, opPG2, opPG3, multiplierC0, multiplierC1, multiplierC2, multiplierC3);
        }
        if (useShift) {
            DoShift(loopIdx, loopRange, bufShift, scaleShiftPG0, scaleShiftPG1, scaleShiftPG2, scaleShiftPG3, opPG0,
                    opPG1, opPG2, opPG3, diffDst0, diffDst1, diffDst2, diffDst3);
        }
    }

    static void Do(SizeType loopIdx, SizeType loopRange, const InType *src, const OutType *diffDst,
                   const ScaleShiftType *scale, bool useScale, typename Vector::Vec<CompType>::Type &mean0,
                   typename Vector::Vec<CompType>::Type &mean1, typename Vector::Vec<CompType>::Type &mean2,
                   typename Vector::Vec<CompType>::Type &mean3, typename Vector::Vec<CompType>::Type &ddGammaVec0,
                   typename Vector::Vec<CompType>::Type &ddGammaVec1, typename Vector::Vec<CompType>::Type &ddGammaVec2,
                   typename Vector::Vec<CompType>::Type &ddGammaVec3,
                   typename Vector::Vec<CompType>::Type &ddGammaXVec0,
                   typename Vector::Vec<CompType>::Type &ddGammaXVec1,
                   typename Vector::Vec<CompType>::Type &ddGammaXVec2,
                   typename Vector::Vec<CompType>::Type &ddGammaXVec3)
    {
        typename Vector::Vec<CompType>::Type gammaVec0;
        typename Vector::Vec<CompType>::Type gammaVec1;
        typename Vector::Vec<CompType>::Type gammaVec2;
        typename Vector::Vec<CompType>::Type gammaVec3;
        if (useScale) {
            svbool_t scalePG0;
            svbool_t scalePG1;
            svbool_t scalePG2;
            svbool_t scalePG3;
            Vector::InitPGN<ScaleShiftType,
                            Vector::GetUsedRegsN<ScaleShiftType, InType, CompType, OutType, float>()>::Do(scalePG0,
                                                                                                          scalePG1,
                                                                                                          scalePG2,
                                                                                                          scalePG3,
                                                                                                          loopIdx,
                                                                                                          loopRange);
            Vector::LoadN<ScaleShiftType, CompType,
                          Vector::GetUsedRegsN<ScaleShiftType, InType, CompType, OutType, ScaleShiftType, float>()>::
                Do(static_cast<const ScaleShiftType *>(scale) + loopIdx, scalePG0, scalePG1, scalePG2, scalePG3,
                   gammaVec0, gammaVec1, gammaVec2, gammaVec3);
        } else {
            Vector::DupFloatN<InType, CompType,
                              Vector::GetUsedRegsN<InType, CompType, OutType, ScaleShiftType, float>()>::Do(1.0f,
                                                                                                            gammaVec0,
                                                                                                            gammaVec1,
                                                                                                            gammaVec2,
                                                                                                            gammaVec3);
        }
        typename Vector::Vec<CompType>::Type diffDst0;
        typename Vector::Vec<CompType>::Type diffDst1;
        typename Vector::Vec<CompType>::Type diffDst2;
        typename Vector::Vec<CompType>::Type diffDst3;
        {
            svbool_t lddstPG0;
            svbool_t lddstPG1;
            svbool_t lddstPG2;
            svbool_t lddstPG3;
            Vector::InitPGN<OutType, Vector::GetUsedRegsN<OutType, InType, CompType, ScaleShiftType, float>()>::Do(
                lddstPG0, lddstPG1, lddstPG2, lddstPG3, loopIdx, loopRange);
            Vector::LoadN<OutType, CompType, Vector::GetUsedRegsN<OutType, InType, CompType, ScaleShiftType, float>()>::
                Do(diffDst + loopIdx, lddstPG0, lddstPG1, lddstPG2, lddstPG3, diffDst0, diffDst1, diffDst2, diffDst3);
        }
        typename Vector::Vec<CompType>::Type diffDstMulGamma0;
        typename Vector::Vec<CompType>::Type diffDstMulGamma1;
        typename Vector::Vec<CompType>::Type diffDstMulGamma2;
        typename Vector::Vec<CompType>::Type diffDstMulGamma3;
        svbool_t opPG0;
        svbool_t opPG1;
        svbool_t opPG2;
        svbool_t opPG3;
        Vector::InitPGN<CompType, Vector::GetUsedRegsN<CompType, InType, OutType, ScaleShiftType, float>()>::Do(
            opPG0, opPG1, opPG2, opPG3, loopIdx, loopRange);
        Vector::MulMN<CompType, Vector::GetUsedRegsN<CompType, InType, OutType, ScaleShiftType, float>()>::Do(
            opPG0, opPG1, opPG2, opPG3, diffDst0, diffDst1, diffDst2, diffDst3, gammaVec0, gammaVec1, gammaVec2,
            gammaVec3, diffDstMulGamma0, diffDstMulGamma1, diffDstMulGamma2, diffDstMulGamma3);
        Vector::AddMN<CompType, Vector::GetUsedRegsN<CompType, InType, OutType, ScaleShiftType, float>()>::Do(
            opPG0, opPG1, opPG2, opPG3, ddGammaVec0, ddGammaVec1, ddGammaVec2, ddGammaVec3, diffDstMulGamma0,
            diffDstMulGamma1, diffDstMulGamma2, diffDstMulGamma3, ddGammaVec0, ddGammaVec1, ddGammaVec2, ddGammaVec3);
        typename Vector::Vec<CompType>::Type src0;
        typename Vector::Vec<CompType>::Type src1;
        typename Vector::Vec<CompType>::Type src2;
        typename Vector::Vec<CompType>::Type src3;
        {
            svbool_t ldsrcPG0, ldsrcPG1, ldsrcPG2, ldsrcPG3;
            Vector::InitPGN<InType, Vector::GetUsedRegsN<InType, OutType, CompType, ScaleShiftType, float>()>::Do(
                ldsrcPG0, ldsrcPG1, ldsrcPG2, ldsrcPG3, loopIdx, loopRange);
            Vector::LoadN<InType, CompType, Vector::GetUsedRegsN<InType, CompType, OutType, ScaleShiftType, float>()>::
                Do(src + loopIdx, ldsrcPG0, ldsrcPG1, ldsrcPG2, ldsrcPG3, src0, src1, src2, src3);
        }
        typename Vector::Vec<CompType>::Type tmp0;
        typename Vector::Vec<CompType>::Type tmp1;
        typename Vector::Vec<CompType>::Type tmp2;
        typename Vector::Vec<CompType>::Type tmp3;
        Vector::SubXN<CompType, Vector::GetUsedRegsN<CompType, InType, OutType, ScaleShiftType, float>()>::Do(
            opPG0, opPG1, opPG2, opPG3, src0, src1, src2, src3, mean0, mean1, mean2, mean3, tmp0, tmp1, tmp2, tmp3);
        // We cannot use the Mla instruction here due to the absence of intermediate rounding.
        // Using Mla would result in a more precise answer than required by the DNN.
        // To maintain the desired level of precision and align with the DNN's requirements,
        // we instead use separate Multiply and Add operations.
        // This approach allows us to control the precision and rounding behavior more accurately,
        // ensuring our calculations match the expected output of the DNN.
        Vector::MulMN<CompType, Vector::GetUsedRegsN<CompType, InType, OutType, ScaleShiftType, float>()>::Do(
            opPG0, opPG1, opPG2, opPG3, tmp0, tmp1, tmp2, tmp3, diffDstMulGamma0, diffDstMulGamma1, diffDstMulGamma2,
            diffDstMulGamma3, tmp0, tmp1, tmp2, tmp3);
        Vector::AddMN<CompType, Vector::GetUsedRegsN<CompType, InType, OutType, ScaleShiftType, float>()>::Do(
            opPG0, opPG1, opPG2, opPG3, ddGammaXVec0, ddGammaXVec1, ddGammaXVec2, ddGammaXVec3, tmp0, tmp1, tmp2, tmp3,
            ddGammaXVec0, ddGammaXVec1, ddGammaXVec2, ddGammaXVec3);
    }

    static void DoScale(SizeType loopIdx, SizeType loopRange, const ScaleShiftType *scale,
                        typename Vector::Vec<CompType>::Type &gammaVec0,
                        typename Vector::Vec<CompType>::Type &gammaVec1,
                        typename Vector::Vec<CompType>::Type &gammaVec2,
                        typename Vector::Vec<CompType>::Type &gammaVec3)
    {
        svbool_t scalePG0;
        svbool_t scalePG1;
        svbool_t scalePG2;
        svbool_t scalePG3;
        Vector::InitPGN<ScaleShiftType, Vector::GetUsedRegsN<ScaleShiftType, InType, CompType, OutType, float>()>::Do(
            scalePG0, scalePG1, scalePG2, scalePG3, loopIdx, loopRange);
        Vector::LoadN<ScaleShiftType, CompType,
                      Vector::GetUsedRegsN<ScaleShiftType, InType, CompType, OutType, float>()>::
            Do(static_cast<const ScaleShiftType *>(scale) + loopIdx, scalePG0, scalePG1, scalePG2, scalePG3, gammaVec0,
               gammaVec1, gammaVec2, gammaVec3);
    }

    static void DoCalcStat(
        const InType *src, SizeType loopIdx, SizeType loopRange, svbool_t &opPG0, svbool_t &opPG1, svbool_t &opPG2,
        svbool_t &opPG3, typename Vector::Vec<CompType>::Type &mean0, typename Vector::Vec<CompType>::Type &mean1,
        typename Vector::Vec<CompType>::Type &mean2, typename Vector::Vec<CompType>::Type &mean3,
        typename Vector::Vec<CompType>::Type &ddGammaVec0, typename Vector::Vec<CompType>::Type &ddGammaVec1,
        typename Vector::Vec<CompType>::Type &ddGammaVec2, typename Vector::Vec<CompType>::Type &ddGammaVec3,
        typename Vector::Vec<CompType>::Type &ddGammaXVec0, typename Vector::Vec<CompType>::Type &ddGammaXVec1,
        typename Vector::Vec<CompType>::Type &ddGammaXVec2, typename Vector::Vec<CompType>::Type &ddGammaXVec3,
        typename Vector::Vec<CompType>::Type &invSqrtVariance0, typename Vector::Vec<CompType>::Type &invSqrtVariance1,
        typename Vector::Vec<CompType>::Type &invSqrtVariance2, typename Vector::Vec<CompType>::Type &invSqrtVariance3,
        typename Vector::Vec<CompType>::Type &dSrcVec0, typename Vector::Vec<CompType>::Type &dSrcVec1,
        typename Vector::Vec<CompType>::Type &dSrcVec2, typename Vector::Vec<CompType>::Type &dSrcVec3)
    {
        typename Vector::Vec<CompType>::Type src0;
        typename Vector::Vec<CompType>::Type src1;
        typename Vector::Vec<CompType>::Type src2;
        typename Vector::Vec<CompType>::Type src3;
        {
            svbool_t ldsrcPG0;
            svbool_t ldsrcPG1;
            svbool_t ldsrcPG2;
            svbool_t ldsrcPG3;
            Vector::InitPGN<InType, Vector::GetUsedRegsN<InType, OutType, CompType, ScaleShiftType, float>()>::Do(
                ldsrcPG0, ldsrcPG1, ldsrcPG2, ldsrcPG3, loopIdx, loopRange);
            Vector::LoadN<InType, CompType, Vector::GetUsedRegsN<InType, CompType, OutType, ScaleShiftType, float>()>::
                Do(src + loopIdx, ldsrcPG0, ldsrcPG1, ldsrcPG2, ldsrcPG3, src0, src1, src2, src3);
        }
        Vector::SubXN<CompType, Vector::GetUsedRegsN<CompType, InType, OutType, ScaleShiftType, float>()>::Do(
            opPG0, opPG1, opPG2, opPG3, src0, src1, src2, src3, mean0, mean1, mean2, mean3, src0, src1, src2, src3);
        Vector::MulMN<CompType, Vector::GetUsedRegsN<CompType, InType, OutType, ScaleShiftType, float>()>::Do(
            opPG0, opPG1, opPG2, opPG3, src0, src1, src2, src3, ddGammaXVec0, ddGammaXVec1, ddGammaXVec2, ddGammaXVec3,
            src0, src1, src2, src3);
        Vector::MulMN<CompType, Vector::GetUsedRegsN<CompType, InType, OutType, ScaleShiftType, float>()>::Do(
            opPG0, opPG1, opPG2, opPG3, src0, src1, src2, src3, invSqrtVariance0, invSqrtVariance1, invSqrtVariance2,
            invSqrtVariance3, src0, src1, src2, src3);
        Vector::AddMN<CompType, Vector::GetUsedRegsN<CompType, InType, OutType, ScaleShiftType, float>()>::Do(
            opPG0, opPG1, opPG2, opPG3, ddGammaVec0, ddGammaVec1, ddGammaVec2, ddGammaVec3, src0, src1, src2, src3,
            src0, src1, src2, src3);
        Vector::DivnXN<CompType, Vector::GetUsedRegsN<CompType, InType, OutType, ScaleShiftType, float>()>::Do(
            opPG0, opPG1, opPG2, opPG3, src0, src1, src2, src3, loopRange, loopRange, loopRange, loopRange, src0, src1,
            src2, src3);
        Vector::SubXN<CompType, Vector::GetUsedRegsN<CompType, InType, OutType, ScaleShiftType, float>()>::Do(
            opPG0, opPG1, opPG2, opPG3, dSrcVec0, dSrcVec1, dSrcVec2, dSrcVec3, src0, src1, src2, src3, dSrcVec0,
            dSrcVec1, dSrcVec2, dSrcVec3);
    }

    static void Do(
        SizeType loopIdx, SizeType loopRange, const InType *src, const OutType *diffDst, const ScaleShiftType *scale,
        bool useScale, bool calculateStats, typename Vector::Vec<CompType>::Type &mean0,
        typename Vector::Vec<CompType>::Type &mean1, typename Vector::Vec<CompType>::Type &mean2,
        typename Vector::Vec<CompType>::Type &mean3, typename Vector::Vec<CompType>::Type &ddGammaVec0,
        typename Vector::Vec<CompType>::Type &ddGammaVec1, typename Vector::Vec<CompType>::Type &ddGammaVec2,
        typename Vector::Vec<CompType>::Type &ddGammaVec3, typename Vector::Vec<CompType>::Type &ddGammaXVec0,
        typename Vector::Vec<CompType>::Type &ddGammaXVec1, typename Vector::Vec<CompType>::Type &ddGammaXVec2,
        typename Vector::Vec<CompType>::Type &ddGammaXVec3, typename Vector::Vec<CompType>::Type &invSqrtVariance0,
        typename Vector::Vec<CompType>::Type &invSqrtVariance1, typename Vector::Vec<CompType>::Type &invSqrtVariance2,
        typename Vector::Vec<CompType>::Type &invSqrtVariance3, InType *diffSrc)
    {
        typename Vector::Vec<CompType>::Type gammaVec0;
        typename Vector::Vec<CompType>::Type gammaVec1;
        typename Vector::Vec<CompType>::Type gammaVec2;
        typename Vector::Vec<CompType>::Type gammaVec3;
        if (useScale) {
            DoScale(loopIdx, loopRange, scale, gammaVec0, gammaVec1, gammaVec2, gammaVec3);
        } else {
            Vector::DupFloatN<InType, CompType,
                              Vector::GetUsedRegsN<InType, CompType, OutType, ScaleShiftType, float>()>::Do(1.0f,
                                                                                                            gammaVec0,
                                                                                                            gammaVec1,
                                                                                                            gammaVec2,
                                                                                                            gammaVec3);
        }
        typename Vector::Vec<CompType>::Type diffDst0;
        typename Vector::Vec<CompType>::Type diffDst1;
        typename Vector::Vec<CompType>::Type diffDst2;
        typename Vector::Vec<CompType>::Type diffDst3;
        svbool_t lddstPG0;
        svbool_t lddstPG1;
        svbool_t lddstPG2;
        svbool_t lddstPG3;
        Vector::InitPGN<OutType, Vector::GetUsedRegsN<OutType, InType, CompType, ScaleShiftType, float>()>::Do(
            lddstPG0, lddstPG1, lddstPG2, lddstPG3, loopIdx, loopRange);
        Vector::LoadN<OutType, CompType, Vector::GetUsedRegsN<OutType, InType, CompType, ScaleShiftType, float>()>::Do(
            diffDst + loopIdx, lddstPG0, lddstPG1, lddstPG2, lddstPG3, diffDst0, diffDst1, diffDst2, diffDst3);

        typename Vector::Vec<CompType>::Type dSrcVec0;
        typename Vector::Vec<CompType>::Type dSrcVec1;
        typename Vector::Vec<CompType>::Type dSrcVec2;
        typename Vector::Vec<CompType>::Type dSrcVec3;
        svbool_t opPG0;
        svbool_t opPG1;
        svbool_t opPG2;
        svbool_t opPG3;
        Vector::InitPGN<CompType, Vector::GetUsedRegsN<CompType, InType, OutType, ScaleShiftType, float>()>::Do(
            opPG0, opPG1, opPG2, opPG3, loopIdx, loopRange);
        Vector::MulMN<CompType, Vector::GetUsedRegsN<CompType, InType, OutType, ScaleShiftType, float>()>::Do(
            opPG0, opPG1, opPG2, opPG3, diffDst0, diffDst1, diffDst2, diffDst3, gammaVec0, gammaVec1, gammaVec2,
            gammaVec3, dSrcVec0, dSrcVec1, dSrcVec2, dSrcVec3);
        if (calculateStats) {
            DoCalcStat(src, loopIdx, loopRange, opPG0, opPG1, opPG2, opPG3, mean0, mean1, mean2, mean3, ddGammaVec0,
                       ddGammaVec1, ddGammaVec2, ddGammaVec3, ddGammaXVec0, ddGammaXVec1, ddGammaXVec2, ddGammaXVec3,
                       invSqrtVariance0, invSqrtVariance1, invSqrtVariance2, invSqrtVariance3, dSrcVec0, dSrcVec1,
                       dSrcVec2, dSrcVec3);
        }
        Vector::MulMN<CompType, Vector::GetUsedRegsN<CompType, InType, OutType, ScaleShiftType, float>()>::Do(
            opPG0, opPG1, opPG2, opPG3, dSrcVec0, dSrcVec1, dSrcVec2, dSrcVec3, invSqrtVariance0, invSqrtVariance1,
            invSqrtVariance2, invSqrtVariance3, dSrcVec0, dSrcVec1, dSrcVec2, dSrcVec3);
        svbool_t stPG0;
        svbool_t stPG1;
        svbool_t stPG2;
        svbool_t stPG3;
        Vector::InitPGN<InType, Vector::GetUsedRegsN<InType, OutType, CompType, ScaleShiftType, float>()>::Do(
            stPG0, stPG1, stPG2, stPG3, loopIdx, loopRange);
        Vector::StoreN<InType, CompType, Vector::GetUsedRegsN<InType, OutType, CompType, ScaleShiftType, float>()>::Do(
            stPG0, stPG1, stPG2, stPG3, dSrcVec0, dSrcVec1, dSrcVec2, dSrcVec3, diffSrc + loopIdx);
    }
};

template <typename CompType, typename InType, typename OutType, typename ScaleShiftType>
void EvaluateRewriteScaleShift(ScaleShiftType *diffScale, ScaleShiftType *diffShift, float *bufScale, float *bufShift,
                               NormalizationFlags flags, SizeType axisSize, SizeType numThreads) noexcept
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
                                const OutType *diffDst, ScaleShiftType *diffScale, ScaleShiftType *diffShift,
                                float *bufScale, float *bufShift, float eps, NormalizationFlags flags,
                                SizeType axisSize, SizeType axisSizeConst) noexcept
{
    bool useScale = static_cast<bool>(flags & NormalizationFlags::USE_SCALE);
    bool useShift = static_cast<bool>(flags & NormalizationFlags::USE_SHIFT);
    Service::Unused(axisSize, diffScale, diffShift);
    {
        if (useScale || useShift) {
            for (SizeType ou = 0; ou < outerSize; ++ou) {
                float var = variance[ou];
                float m = mean[ou];
                float denom = std::sqrt(var + eps);
                float invSqrtVar = 1.0f / denom;
                typename Vector::Vec<CompType>::Type mean0;
                typename Vector::Vec<CompType>::Type mean1;
                typename Vector::Vec<CompType>::Type mean2;
                typename Vector::Vec<CompType>::Type mean3;
                Vector::DupFloatN<InType, CompType,
                                  Vector::GetUsedRegsN<InType, CompType, OutType, ScaleShiftType, float>()>::Do(m,
                                                                                                                mean0,
                                                                                                                mean1,
                                                                                                                mean2,
                                                                                                                mean3);
                typename Vector::Vec<CompType>::Type invSqrtVar0;
                typename Vector::Vec<CompType>::Type invSqrtVar1;
                typename Vector::Vec<CompType>::Type invSqrtVar2;
                typename Vector::Vec<CompType>::Type invSqrtVar3;
                Vector::DupFloatN<InType, CompType,
                                  Vector::GetUsedRegsN<InType, CompType, OutType, ScaleShiftType, float>()>::
                    Do(invSqrtVar, invSqrtVar0, invSqrtVar1, invSqrtVar2, invSqrtVar3);
                Service::UnrollFor::Do<1>(axisSizeConst, LayerNormLastBWD<CompType, InType, OutType, ScaleShiftType>{},
                                          src + ou * axisSizeConst, diffDst + ou * axisSizeConst, useScale, useShift,
                                          bufScale, bufShift, mean0, mean1, mean2, mean3, invSqrtVar0, invSqrtVar1,
                                          invSqrtVar2, invSqrtVar3);
            }
        }
    }
}

template <typename CompType, typename InType, typename OutType, typename ScaleShiftType>
static void EvaluateLastAxisCalcStat(
    const InType *src, const OutType *diffDst, const ScaleShiftType *scale, SizeType n, SizeType axisSize,
    CompType &ddGammaX, CompType &ddGamma, svbool_t &pgTrue, CompType &invSqrtVariance, bool useScale,
    typename Vector::Vec<CompType>::Type &mean0, typename Vector::Vec<CompType>::Type &mean1,
    typename Vector::Vec<CompType>::Type &mean2, typename Vector::Vec<CompType>::Type &mean3,
    typename Vector::Vec<CompType>::Type &ddGammaVec0, typename Vector::Vec<CompType>::Type &ddGammaVec1,
    typename Vector::Vec<CompType>::Type &ddGammaVec2, typename Vector::Vec<CompType>::Type &ddGammaVec3,
    typename Vector::Vec<CompType>::Type &ddGammaXVec0, typename Vector::Vec<CompType>::Type &ddGammaXVec1,
    typename Vector::Vec<CompType>::Type &ddGammaXVec2, typename Vector::Vec<CompType>::Type &ddGammaXVec3) noexcept
{
    Service::UnrollFor::Do<1>(axisSize, LayerNormLastBWD<CompType, InType, OutType, ScaleShiftType>{},
                              src + n * axisSize, diffDst + n * axisSize, scale, useScale, mean0, mean1, mean2, mean3,
                              ddGammaVec0, ddGammaVec1, ddGammaVec2, ddGammaVec3, ddGammaXVec0, ddGammaXVec1,
                              ddGammaXVec2, ddGammaXVec3);
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
    svbool_t pgTrue = svptrue_b32();

    for (SizeType n = 0; n < outerSize; ++n) {
        CompType invSqrtVariance = 1.0f / std::sqrt(variance[n] + eps);
        CompType ddGamma = 0.0f;
        typename Vector::Vec<CompType>::Type ddGammaVec0;
        typename Vector::Vec<CompType>::Type ddGammaVec1;
        typename Vector::Vec<CompType>::Type ddGammaVec2;
        typename Vector::Vec<CompType>::Type ddGammaVec3;
        Vector::DupFloatN<InType, CompType,
                          Vector::GetUsedRegsN<InType, CompType, OutType, ScaleShiftType, float>()>::Do(0.0f,
                                                                                                        ddGammaVec0,
                                                                                                        ddGammaVec1,
                                                                                                        ddGammaVec2,
                                                                                                        ddGammaVec3);
        CompType ddGammaX = 0.0f;
        typename Vector::Vec<CompType>::Type ddGammaXVec0;
        typename Vector::Vec<CompType>::Type ddGammaXVec1;
        typename Vector::Vec<CompType>::Type ddGammaXVec2;
        typename Vector::Vec<CompType>::Type ddGammaXVec3;
        Vector::DupFloatN<InType, CompType,
                          Vector::GetUsedRegsN<InType, CompType, OutType, ScaleShiftType, float>()>::Do(0.0f,
                                                                                                        ddGammaXVec0,
                                                                                                        ddGammaXVec1,
                                                                                                        ddGammaXVec2,
                                                                                                        ddGammaXVec3);
        typename Vector::Vec<CompType>::Type mean0;
        typename Vector::Vec<CompType>::Type mean1;
        typename Vector::Vec<CompType>::Type mean2;
        typename Vector::Vec<CompType>::Type mean3;
        Vector::DupFloatN<InType, CompType,
                          Vector::GetUsedRegsN<InType, CompType, OutType, ScaleShiftType, float>()>::Do(mean[n], mean0,
                                                                                                        mean1, mean2,
                                                                                                        mean3);
        if (calculateStats) {
            EvaluateLastAxisCalcStat(src, diffDst, scale, n, axisSize, ddGammaX, ddGamma, pgTrue, invSqrtVariance,
                                     useScale, mean0, mean1, mean2, mean3, ddGammaVec0, ddGammaVec1, ddGammaVec2,
                                     ddGammaVec3, ddGammaXVec0, ddGammaXVec1, ddGammaXVec2, ddGammaXVec3);
        }

        Vector::DupFloatN<InType, CompType,
                          Vector::GetUsedRegsN<InType, CompType, OutType, ScaleShiftType, float>()>::Do(ddGamma,
                                                                                                        ddGammaVec0,
                                                                                                        ddGammaVec1,
                                                                                                        ddGammaVec2,
                                                                                                        ddGammaVec3);
        Vector::DupFloatN<InType, CompType,
                          Vector::GetUsedRegsN<InType, CompType, OutType, ScaleShiftType, float>()>::Do(ddGammaX,
                                                                                                        ddGammaXVec0,
                                                                                                        ddGammaXVec1,
                                                                                                        ddGammaXVec2,
                                                                                                        ddGammaXVec3);
        typename Vector::Vec<CompType>::Type invSqrtVariance0;
        typename Vector::Vec<CompType>::Type invSqrtVariance1;
        typename Vector::Vec<CompType>::Type invSqrtVariance2;
        typename Vector::Vec<CompType>::Type invSqrtVariance3;
        Vector::DupFloatN<InType, CompType, Vector::GetUsedRegsN<InType, CompType, OutType, ScaleShiftType, float>()>::
            Do(invSqrtVariance, invSqrtVariance0, invSqrtVariance1, invSqrtVariance2, invSqrtVariance3);
        Service::UnrollFor::Do<1>(axisSize, LayerNormLastBWD<CompType, InType, OutType, ScaleShiftType>{},
                                  src + n * axisSize, diffDst + n * axisSize, scale, useScale, calculateStats, mean0,
                                  mean1, mean2, mean3, ddGammaVec0, ddGammaVec1, ddGammaVec2, ddGammaVec3, ddGammaXVec0,
                                  ddGammaXVec1, ddGammaXVec2, ddGammaXVec3, invSqrtVariance0, invSqrtVariance1,
                                  invSqrtVariance2, invSqrtVariance3, diffSrc + n * axisSize);
    }
}

} // namespace LNorm
} // namespace Detail
} // namespace KuDNN

#endif //  KUDNN_LNORM_BWD_LAST_DEF_HPP
