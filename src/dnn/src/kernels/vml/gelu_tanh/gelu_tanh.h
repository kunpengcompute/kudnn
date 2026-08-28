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

#ifndef VML_VC_LA_GELU_TANH_H
#define VML_VC_LA_GELU_TANH_H

#include "kernels/vml/vml_sve_la_tanh.h"

#define VC_GELU_TANHF_FITTING             svdup_n_f32(0x1.6e4e26p-5f) // 0x3d372713 0.044715
#define VC_GELU_TANHF_FITTING_THREE_TIMES svdup_n_f32(0x1.12ba9ep-3f) // 0x3e095d4f 0.134145
#define VC_GELU_TANHF_SQRT_TWO_OVER_PI    svdup_n_f32(0x1.988454p-1f) // 0x3f4c422a

inline svfloat32_t SvGeluTanhF32X(svfloat32_t x, svbool_t pg)
{
    // compute G(x) = sqrt_root_two_over_pi * x * (1 + fitting_const * x * x)
    svfloat32_t x2 = svmul_f32_x(pg, x, x);
    svfloat32_t y = svmla_f32_x(pg, VC_FLOAT_ONE, x2, VC_GELU_TANHF_FITTING);
    y = svmul_f32_x(pg, y, x);
    y = svmul_f32_x(pg, y, VC_GELU_TANHF_SQRT_TWO_OVER_PI);

    // compute tanh(G(x))
    svfloat32_t tanhY = SvTanhF32XNew2(y, pg);

    // compute 0.5 * x * (1 + tanh(G(x)))
    svfloat32_t res = svadd_f32_x(pg, VC_FLOAT_ONE, tanhY);
    res = svmul_f32_x(pg, res, VC_FLOAT_HALF);
    res = svmul_f32_x(pg, res, x);

    return res;
}

inline svfloat32_t SvGeluTanhBwdF32X(svfloat32_t x, svbool_t pg)
{
    // compute G1(x) = sqrt_root_two_over_pi * x * (1 + fitting_const * x^2)
    // compute G2(x) = sqrt_root_two_over_pi * x * (1 + 3 * fitting_const * x^2)
    svfloat32_t x2 = svmul_f32_x(pg, x, x);
    svfloat32_t g1 = svmla_f32_x(pg, VC_FLOAT_ONE, x2, VC_GELU_TANHF_FITTING);
    svfloat32_t g2 = svmla_f32_x(pg, VC_FLOAT_ONE, x2, VC_GELU_TANHF_FITTING_THREE_TIMES);

    svfloat32_t nx = svmul_f32_x(pg, x, VC_GELU_TANHF_SQRT_TWO_OVER_PI);
    g1 = svmul_f32_x(pg, g1, nx);
    g2 = svmul_f32_x(pg, g2, nx);

    // compute T = tanh(G1(x))
    svfloat32_t t = SvTanhF32XNew(g1, pg);

    // compute 0.5 * (1 + T) * (1 + G2 * (1 - T))
    // 1) R = G2 * (1 - T) = G2 - G2 * T
    // 2) Q = 1 + T
    // 3) res = Q * (1 + R) = Q + Q * R
    svfloat32_t q = svadd_f32_x(pg, VC_FLOAT_ONE, t);
    svfloat32_t res = svmla_f32_x(pg, q, q, svmls_f32_x(pg, g2, g2, t));
    return svmul_f32_x(pg, res, VC_FLOAT_HALF);
}

// same with SvGeluTanhBwdF32X, tanh is replaced for better perf
inline svfloat32_t SvGeluTanhBwdF32X2(svfloat32_t x, svbool_t pg)
{
    svfloat32_t x2 = svmul_f32_x(pg, x, x);
    svfloat32_t g1 = svmla_f32_x(pg, VC_FLOAT_ONE, x2, VC_GELU_TANHF_FITTING);
    svfloat32_t g2 = svmla_f32_x(pg, VC_FLOAT_ONE, x2, VC_GELU_TANHF_FITTING_THREE_TIMES);

    svfloat32_t nx = svmul_f32_x(pg, x, VC_GELU_TANHF_SQRT_TWO_OVER_PI);
    g1 = svmul_f32_x(pg, g1, nx);
    g2 = svmul_f32_x(pg, g2, nx);

    svfloat32_t t = SvTanhF32XNew2(g1, pg);

    svfloat32_t q = svadd_f32_x(pg, VC_FLOAT_ONE, t);
    svfloat32_t res = svmla_f32_x(pg, q, q, svmls_f32_x(pg, g2, g2, t));
    return svmul_f32_x(pg, res, VC_FLOAT_HALF);
}

#endif
