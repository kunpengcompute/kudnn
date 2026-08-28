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

#include "kernels/rnn/rnn_kernel_operators.hpp"
#include "kernels/vml/vml_sve_la_tanh.h"
#include "kernels/vml/vml_sve_la_sigmoid.h"

namespace KuDNN {
namespace Detail {
namespace Rnn {

svfloat32_t RnnTanhFWDF32X(svfloat32_t x, svfloat32_t va, svbool_t pg)
{
    Service::Unused(va);
    return SvTanhF32X(x, pg);
}

svfloat32_t RnnLogisticFWDF32X(svfloat32_t x, svfloat32_t va, svbool_t pg)
{
    Service::Unused(va);
    svfloat32_t argI = svneg_f32_x(pg, x);
    svfloat32_t res = SvSigmoidF32X(argI, pg);
    return res;
}

svfloat32_t RnnReLUFWDF32X(svfloat32_t x, svfloat32_t va, svbool_t pg)
{
    svbool_t cmpLtZero = svcmplt_n_f32(pg, x, FLOAT_ZERO);
    svfloat32_t res = svmul_f32_m(cmpLtZero, x, va);
    return res;
}

svfloat32_t RnnLinearF32X(svfloat32_t x, svfloat32_t va, svbool_t pg)
{
    svfloat32_t res = svmul_f32_x(pg, x, va);
    return res;
}

void VanillaRnnVCFWDF32(const int64_t len, const float *scratch, const float *bias, float *cellDstLay,
    float* cellDstIter, float *cellWsGates, svfloat32_t (*actFunc)(svfloat32_t, svfloat32_t, svbool_t),
    const float alpha)
{
    /*
     * rnn operation
     * a_t = W * h + U * x + b_h
     * h_t = func(a_t) default func is relu
     */
    int64_t i;
    svfloat32_t va = svdup_n_f32(alpha);
    int loopRnn = svcntw();
    for (i = 0; i < len; i += loopRnn) {
        svbool_t pg = svwhilelt_b32_s64(i, len);
        svfloat32_t scratchArgI = svld1_f32(pg, scratch + i);
        svfloat32_t biasI = svld1_f32(pg, bias + i);
        scratchArgI = svadd_f32_x(pg, scratchArgI, biasI);

        svfloat32_t h = actFunc(scratchArgI, va, pg);
        svst1_f32(pg, cellDstLay + i, h);
        if (cellDstIter) {
            svst1_f32(pg, cellDstIter + i, h);
        }
        if (cellWsGates) {
            svst1_f32(pg, cellWsGates + i, h);
        }
    }
}

void VanillaRnnVCFWDBF16(const int64_t len, const float *scratch, const __bf16 *bias, __bf16 *cellDstLay,
    __bf16* cellDstIter, __bf16 *cellWsGates, svfloat32_t (*actFunc)(svfloat32_t, svfloat32_t, svbool_t),
    const float alpha)
{
    int64_t i;
    svfloat32_t va = svdup_n_f32(alpha);
    int loopRnn = svcnth();
    int numF32 = svcntw();
    for (i = 0; i < len; i += loopRnn) {
        svbool_t pg = svwhilelt_b16_s64(i, len);
        svbfloat16_t biasI = svld1_bf16(pg, bias + i);

        svfloat32_t biasIBot = SvCvtF32BF16BottomX(biasI);
        svfloat32_t biasITop = SvCvtF32BF16TopX(biasI);

        svbool_t pgBot = svwhilelt_b32_s64(i, len);
        svbool_t pgTop = svwhilelt_b32_s64(i + numF32, len);
        svfloat32_t argIBot = svld1_f32(pgBot, scratch + i);
        svfloat32_t argITop = svld1_f32(pgTop, scratch + i + numF32);
        svfloat32_t dG0Bot = actFunc(svadd_f32_x(pgBot, argIBot, biasIBot), va, pgBot);
        svfloat32_t dG0Top = actFunc(svadd_f32_x(pgTop, argITop, biasITop), va, pgTop);
        svbfloat16_t h = SvCvtBF16F32X(pgBot, dG0Bot, pgTop, dG0Top);

        svst1_bf16(pg, cellDstLay + i, h);
        if (cellDstIter) {
            svst1_bf16(pg, cellDstIter + i, h);
        }
        if (cellWsGates) {
            svst1_bf16(pg, cellWsGates + i, h);
        }
    }
}

void VanillaRnnVCFWDBF16F32(const int64_t len, const float *scratch, const float *bias, __bf16 *cellDstLay,
    __bf16* cellDstIter, __bf16 *cellWsGates, svfloat32_t (*actFunc)(svfloat32_t, svfloat32_t, svbool_t),
    const float alpha)
{
    int64_t i;
    svfloat32_t va = svdup_n_f32(alpha);
    int loopRnn = svcnth();
    int numF32 = svcntw();
    for (i = 0; i < len; i += loopRnn) {
        svbool_t pg = svwhilelt_b16_s64(i, len);
        svbool_t pgBot = svwhilelt_b32_s64(i, len);
        svbool_t pgTop = svwhilelt_b32_s64(i + numF32, len);
        svfloat32_t argIBot = svld1_f32(pgBot, scratch + i);
        svfloat32_t argITop = svld1_f32(pgTop, scratch + i + numF32);
        svfloat32_t biasIBot = svld1_f32(pgBot, bias + i);
        svfloat32_t biasITop = svld1_f32(pgTop, bias + i + numF32);
        svfloat32_t dG0Bot = actFunc(svadd_f32_x(pgBot, argIBot, biasIBot), va, pgBot);
        svfloat32_t dG0Top = actFunc(svadd_f32_x(pgTop, argITop, biasITop), va, pgTop);
        svbfloat16_t h = SvCvtBF16F32X(pgBot, dG0Bot, pgTop, dG0Top);

        svst1_bf16(pg, cellDstLay + i, h);
        if (cellWsGates) {
            svst1_bf16(pg, cellWsGates + i, h);
        }
        if (cellDstIter) {
            svst1_bf16(pg, cellDstIter + i, h);
        }
    }
}

svfloat32_t RnnDTanh(svfloat32_t x, svfloat32_t va, svbool_t pg)
{
    Service::Unused(va);
    return svmsb_f32_x(pg, x, x, VML_VC_F32_ONE);
}

svfloat32_t RnnDLogistic(svfloat32_t x, svfloat32_t va, svbool_t pg)
{
    Service::Unused(va);
    return svmls_f32_x(pg, x, x, x);
}

svfloat32_t RnnDAlphaReLU(svfloat32_t x, svfloat32_t va, svbool_t pg)
{
    svbool_t gtZero = svcmpgt_f32(pg, x, svdup_f32(0.0f));
    svfloat32_t resultOne = VML_VC_F32_ONE;
    return svsel_f32(gtZero, resultOne, va);
}

inline void RnnVCBWDCore(const float *cellDiffDstLay, const float *cellDiffDstIter,
    svfloat32_t (*actFunc)(svfloat32_t, svfloat32_t, svbool_t), svfloat32_t alpha, svfloat32_t *dG0,
    svfloat32_t gates, svbool_t pg)
{
    svfloat32_t ddl = svld1_f32(pg, cellDiffDstLay);
    svfloat32_t ddi = svld1_f32(pg, cellDiffDstIter);
    svfloat32_t dHt = svadd_f32_x(pg, ddl, ddi);

    *dG0 = svmul_f32_x(pg, dHt, actFunc(gates, alpha, pg));
}

void RnnVCBWDF32(const int64_t len, const float *cellDiffDstLay, const float *cellDiffDstIter,
    float *scratch, const float *cellWsGates, svfloat32_t (*actFunc)(svfloat32_t, svfloat32_t, svbool_t),
    const float alpha)
{
    int64_t i;
    svfloat32_t va = svdup_n_f32(alpha);
    int loopRnn = svcntw();
    for (i = 0; i < len; i += loopRnn) {
        svbool_t pg = svwhilelt_b32_s64(i, len);
        svfloat32_t argI = svld1_f32(pg, cellWsGates + i);

        svfloat32_t dG0;
        RnnVCBWDCore(cellDiffDstLay + i, cellDiffDstIter + i, actFunc, va, &dG0, argI, pg);
        svst1_f32(pg, scratch + i, dG0);
    }
}

void RnnVCBWDBF16(const int64_t len, const float *cellDiffDstLay, const float *cellDiffDstIter,
    __bf16 *scratch, const __bf16 *cellWsGates, svfloat32_t (*actFunc)(svfloat32_t, svfloat32_t, svbool_t),
    const float alpha)
{
    int64_t i;
    svfloat32_t va = svdup_n_f32(alpha);
    int loopRnn = svcnth();
    int numF32 = svcntw();
    for (i = 0; i < len; i += loopRnn) {
        svbool_t pg = svwhilelt_b16_s64(i, len);

        svbfloat16_t argI = svld1_bf16(pg, cellWsGates + i);

        svfloat32_t argIBot = SvCvtF32BF16BottomX(argI);
        svfloat32_t argITop = SvCvtF32BF16TopX(argI);

        svfloat32_t dG0Bot;
        svbool_t pgBot = svwhilelt_b32_s64(i, len);
        RnnVCBWDCore(cellDiffDstLay + i, cellDiffDstIter + i, actFunc, va, &dG0Bot, argIBot, pgBot);

        svfloat32_t dG0Top;
        svbool_t pgTop = svwhilelt_b32_s64(i + numF32, len);
        RnnVCBWDCore(cellDiffDstLay + i + numF32, cellDiffDstIter + i + numF32, actFunc, va, &dG0Top, argITop, pgTop);

        svbfloat16_t dScratchRes = SvCvtBF16F32X(pgBot, dG0Bot, pgTop, dG0Top);
        svst1_bf16(pg, scratch + i, dScratchRes);
    }
}

} // Rnn
} // Detail
} // KuDNN
