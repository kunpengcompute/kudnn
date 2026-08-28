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

#ifndef VML_VC_LA_SOFTRELU_H
#define VML_VC_LA_SOFTRELU_H

#include "kernels/vml/vml_sve_la_exp.h"
#include "kernels/vml/vml_sve_la_sigmoid.h"

#define SOFT_RELU_LOG1P_C0  svdup_f32(-0x1.68c6fap-26f)
#define SOFT_RELU_LOG1P_C1  svdup_f32(0x1.ffff1cp-1f)
#define SOFT_RELU_LOG1P_C2  svdup_f32(-0x1.002eb2p-1f)
#define SOFT_RELU_LOG1P_C3  svdup_f32(0x1.4e0c1p-2f)
#define SOFT_RELU_LOG1P_C4  svdup_f32(-0x1.47af7ep-2f)
#define SOFT_RELU_LOG1P_C5  svdup_f32(-0x1.6c3a08p-3f)
#define SOFT_RELU_LOG1P_C6  svdup_f32(-0x1.51e3ccp0f)
#define SOFT_RELU_LOG1P_C7  svdup_f32(-0x1.c3d024p0f)
#define SOFT_RELU_LOG1P_C8  svdup_f32(-0x1.89a61cp0f)
#define SOFT_RELU_126       svdup_f32(126.0f)
#define SOFT_RELU_MANTISSA_SIGN_MASK    svdup_u32(0x807fffff)

__attribute__((always_inline)) inline svfloat32_t SvSoftReluF32X(svfloat32_t va, svfloat32_t x, svbool_t pg)
{
    // ln(1 + exp(ax)) = n * ln(2) + ln(2^-n + exp(r))
    svfloat32_t ax = svmul_f32_x(pg, va, x);
    svbool_t pgRetX = svcmpge_f32(pg, ax, VC_EXPF_UPBOUND);

    ax = svmax_f32_x(pg, ax, VC_EXPF_LOWBOUND);
    ax = svmin_f32_x(pg, ax, VC_EXPF_UPBOUND);

    svfloat32_t fx = svmad_f32_x(pg, ax, VC_EXPF_INVLN2HI, VC_FLOAT_HALF);
    svfloat32_t n = svrintm_f32_x(pg, fx); /* round off to nearest integer */
    svfloat32_t r = svmls_f32_x(pg, ax, n, VC_EXPF_LN2HI);

    svfloat32_t expr = svmla_f32_x(pg, VC_EXPF_C4, r, VC_EXPF_C5);
    expr = svmla_f32_x(pg, VC_EXPF_C3, r, expr);
    expr = svmla_f32_x(pg, VC_EXPF_C2, r, expr);
    expr = svmla_f32_x(pg, VC_EXPF_C1, r, expr);
    expr = svmla_f32_x(pg, VC_EXPF_C0, r, expr);

    // ln(2^-n + exp(r)) = ln((2^-(n-1) + 2 * exp(r))/2.f)
    svfloat32_t negN1 = svsub_f32_x(pg, VC_FLOAT_ONE, n);
    svfloat32_t y = svscale_f32_x(pg, VC_FLOAT_ONE, svcvt_s32_f32_x(pg, negN1));
    y = svmad_f32_x(pg, expr, VC_FLOAT_TWO, y);
    y = svmul_f32_x(pg, y, VC_FLOAT_HALF);

    // ln(y) = m * ln(2) + ln(1 + z)    0.5 <= (1 + z) < 1
    svuint32_t unitY1 = svreinterpret_u32_f32(y);
    svuint32_t unitY2 = svlsr_n_u32_x(pg, unitY1, FLOAT_MANTISSA_BIT_LEN);
    svfloat32_t m = svcvt_f32_u32_x(pg, unitY2);
    m = svsub_f32_x(pg, m, SOFT_RELU_126);

    unitY1 = svand_u32_x(pg, unitY1, SOFT_RELU_MANTISSA_SIGN_MASK);
    unitY1 = svorr_u32_x(pg, VC_FLOAT_HALF_BIT, unitY1);
    svfloat32_t z = svreinterpret_f32_u32(unitY1);
    z = svsub_f32_x(pg, z, VC_FLOAT_ONE);

    // Polynomial Fitting of ln(1 + z)
    svfloat32_t res = svmad_f32_x(pg, SOFT_RELU_LOG1P_C8, z, SOFT_RELU_LOG1P_C7);
    res = svmad_f32_x(pg, res, z, SOFT_RELU_LOG1P_C6);
    res = svmad_f32_x(pg, res, z, SOFT_RELU_LOG1P_C5);
    res = svmad_f32_x(pg, res, z, SOFT_RELU_LOG1P_C4);
    res = svmad_f32_x(pg, res, z, SOFT_RELU_LOG1P_C3);
    res = svmad_f32_x(pg, res, z, SOFT_RELU_LOG1P_C2);
    res = svmad_f32_x(pg, res, z, SOFT_RELU_LOG1P_C1);
    res = svmad_f32_x(pg, res, z, SOFT_RELU_LOG1P_C0);

    // res = m * ln(2) + n * ln(2) + ln(1 + z)
    n = svmul_f32_x(pg, n, VC_FLOAT_LN2);
    m = svmul_f32_x(pg, m, VC_FLOAT_LN2);
#ifdef __clang__
    __asm__ __volatile__("" : "+w"(n));
    __asm__ __volatile__("" : "+w"(m));
#endif
    res = svadd_f32_x(pg, res, n);
    res = svadd_f32_x(pg, res, m);

    res = svdiv_f32_x(pg, res, va);

    res = svsel_f32(pgRetX, x, res);

    return res;
}

__attribute__((always_inline)) inline svfloat32_t SvSoftReluBwdF32X(svfloat32_t va, svfloat32_t srcX,
                                                                    svfloat32_t ddX, svbool_t pg)
{
    svfloat32_t ax = svmul_f32_x(pg, va, srcX);
    svfloat32_t res = SvSigmoidF32X(ax, pg);
    res = svmul_f32_x(pg, res, ddX);

    return res;
}

#endif
