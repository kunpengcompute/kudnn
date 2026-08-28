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

#include "kpl_mpfr.h"
#include <stdio.h>
#include <math.h>

void mpfr_set_double(mpfr_t x, double y)
{
    uint64_t uy = Asuint64(y);
    set_mpfr_d(x, uy >> 32, uy & 0xffffffff);
}

int mpfr_powxa(mpfr_t rop, const mpfr_t a, const mpfr_t y, const mpfr_t x, mpfr_rnd_t rnd)
{
    float alpha = mpfr_get_flt(a, rnd);
    float beta = mpfr_get_flt(y, rnd);
    float s = mpfr_get_flt(x, rnd);

    float res = alpha * powf((float)s, beta);
    mpfr_set_flt(rop, res, rnd);
    return 0;
}

int mpfr_powxa_bwd(mpfr_t rop, const mpfr_t a, const mpfr_t b, const mpfr_t x, const mpfr_t d, mpfr_rnd_t rnd)
{
    float alpha = mpfr_get_flt(a, rnd);
    float beta = mpfr_get_flt(b, rnd);
    float s = mpfr_get_flt(x, rnd);
    float dd = mpfr_get_flt(d, rnd);
    if (beta == 0) {
        mpfr_set_flt(rop, 0.0, rnd);
        return 0;
    }

    float v = (alpha * beta) * powf((float)s, (beta - 1.0));
    float res = dd * v;
    mpfr_set_flt(rop, res, rnd);
    return 0;
}

int mpfr_linear(mpfr_t rop, const mpfr_t a, const mpfr_t b, const mpfr_t x, mpfr_rnd_t rnd)
{
    float fx = mpfr_get_flt(x, rnd);
    float fy = mpfr_get_flt(b, rnd);
    float fz = mpfr_get_flt(a, rnd);
    float res = fma(fz, fx, fy);
    mpfr_set_flt(rop, res, rnd);
    return 0;
}

int mpfr_relu(mpfr_t rop, const mpfr_t a, const mpfr_t x, mpfr_rnd_t rnd)
{
    double dx = mpfr_get_d(x, rnd);
    if (dx > 0) {
        mpfr_set_d(rop, dx, rnd);
    } else {
        float fx = mpfr_get_flt(x, rnd);
        float fa = mpfr_get_flt(a, rnd);
        float res = fx * fa;
        mpfr_set_flt(rop, res, rnd);
    }
    return 0;
}

int mpfr_relu_bwd(mpfr_t rop, const mpfr_t a, const mpfr_t x, const mpfr_t y, mpfr_rnd_t rnd)
{
    float alpha = mpfr_get_flt(a, rnd);
    float s = mpfr_get_flt(x, rnd);
    float dd = mpfr_get_flt(y, rnd);
    float res = s > 0 ? 1 : alpha;
    res = dd * res;
    mpfr_set_flt(rop, res, rnd);
    return 0;
}

int mpfr_sigmoid(mpfr_t rop, const mpfr_t x, mpfr_rnd_t rnd)
{
    mpfr_t exp_op;
    mpfr_init2(exp_op, MPFR_PREC);

    mpfr_neg(exp_op, x, rnd);
    mpfr_exp(exp_op, exp_op, rnd);
    mpfr_add_ui(exp_op, exp_op, 1, rnd);
    mpfr_ui_div(rop, 1, exp_op, rnd);

    mpfr_clear(exp_op);
    return 0;
}

float logistic_fwd(float s)
{
    float exp_overflow_bound = 88.72283172607421875;
    float in = (float)-s;
    return in < exp_overflow_bound ? (1.0f / (1.0f + expf(in))) : 0.0f;
}

int mpfr_sigmoid_bwd(mpfr_t rop, const mpfr_t s, const mpfr_t dd, mpfr_rnd_t rnd)
{
    float srcX = mpfr_get_flt(s, rnd);
    float ddX = mpfr_get_flt(dd, rnd);
    float v = logistic_fwd(srcX);
    float res = ddX * v * (1 - v);

    mpfr_set_flt(rop, res, rnd);
    return 0;
}

int mpfr_abs_bwd(mpfr_t rop, const mpfr_t x, const mpfr_t y, mpfr_rnd_t rnd)
{
    double dx = mpfr_get_d(x, rnd);
    double dd = mpfr_get_d(y, rnd);
    if (dx == 0) {
        mpfr_set_d(rop, 0, rnd);
    } else if (dx > 0) {
        mpfr_set_flt(rop, dd, rnd);
    } else {
        mpfr_set_flt(rop, -dd, rnd);
    }
    return 0;
}

int mpfr_exp_bwd(mpfr_t rop, const mpfr_t s, const mpfr_t dd, mpfr_rnd_t rnd)
{
    float srcX = mpfr_get_flt(s, rnd);
    float ddX = mpfr_get_flt(dd, rnd);
    float res = ddX * expf(srcX);

    mpfr_set_flt(rop, res, rnd);

    return 0;
}

int mpfr_elu(mpfr_t rop, const mpfr_t a, const mpfr_t x, mpfr_rnd_t rnd)
{
    float s = mpfr_get_flt(x, rnd);
    float alpha = mpfr_get_flt(a, rnd);

    float res = s > 0 ? s : alpha * expm1f(s);

    mpfr_set_flt(rop, res, rnd);
    return 0;
}

int mpfr_elu_bwd(mpfr_t rop, const mpfr_t a, const mpfr_t x, const mpfr_t y, mpfr_rnd_t rnd)
{
    float s = mpfr_get_flt(x, rnd);
    float dd = mpfr_get_flt(y, rnd);
    float alpha = mpfr_get_flt(a, rnd);

    float res = dd * (s > 0 ? 1 : alpha * expf(s));

    mpfr_set_flt(rop, res, rnd);
    return 0;
}

int mpfr_hardsigmoid(mpfr_t rop, const mpfr_t a, const mpfr_t b, const mpfr_t x, mpfr_rnd_t rnd)
{
    float fa = mpfr_get_flt(a, rnd);
    float fb = mpfr_get_flt(b, rnd);
    float fx = mpfr_get_flt(x, rnd);
    float res = fma(fa, fx, fb);

    if (res > 1.0f) {
        mpfr_set_flt(rop, 1.0, rnd);
    } else if (res < 0.0f) {
        mpfr_set_flt(rop, 0.0, rnd);
    } else {
        mpfr_set_flt(rop, res, rnd);
    }

    return 0;
}

int mpfr_hardsigmoid_bwd(mpfr_t rop, const mpfr_t a, const mpfr_t b, const mpfr_t x, const mpfr_t y, mpfr_rnd_t rnd)
{
    float fa = mpfr_get_flt(a, rnd);
    float fb = mpfr_get_flt(b, rnd);
    float fx = mpfr_get_flt(x, rnd);
    float dd = mpfr_get_flt(y, rnd);
    float res = fma(fa, fx, fb);

    if (res > 0.0f && res < 1.0f) {
        mpfr_set_flt(rop, dd * fa, rnd);
    } else {
        mpfr_set_flt(rop, 0.0, rnd);
    }

    return 0;
}

int mpfr_hardswish(mpfr_t rop, const mpfr_t a, const mpfr_t b, const mpfr_t x, mpfr_rnd_t rnd)
{
    float fa = mpfr_get_flt(a, rnd);
    float fb = mpfr_get_flt(b, rnd);
    float fx = mpfr_get_flt(x, rnd);
    float res = fma(fa, fx, fb);

    if (res > 1.0f) {
        mpfr_set_flt(rop, 1.0 * fx, rnd);
    } else if (res > 0.0f) {
        mpfr_set_flt(rop, res * fx, rnd);
    } else {
        mpfr_set_flt(rop, 0.0 * fx, rnd);
    }

    return 0;
}

int mpfr_hardswish_bwd(mpfr_t rop, const mpfr_t a, const mpfr_t b, const mpfr_t x, const mpfr_t y, mpfr_rnd_t rnd)
{
    double alpha = mpfr_get_d(a, rnd);
    double beta = mpfr_get_d(b, rnd);
    double s = mpfr_get_d(x, rnd);
    double dd = mpfr_get_d(y, rnd);
    float v = alpha * s + beta;
    float w = 2.f * alpha * s + beta;

    float res = v <= 0.f ? 0.f : v >= 1.f ? dd : dd * w;

    mpfr_set_flt(rop, res, rnd);

    return 0;
}

int mpfr_sqr_bwd(mpfr_t rop, const mpfr_t x, const mpfr_t y, mpfr_rnd_t rnd)
{
    float s = mpfr_get_flt(x, rnd);
    float dd = mpfr_get_flt(y, rnd);
    float res = dd * 2 * s;
    mpfr_set_flt(rop, res, rnd);
    return 0;
}

int mpfr_sqrt_bwd(mpfr_t rop, const mpfr_t ds, const mpfr_t dd, mpfr_rnd_t rnd)
{
    mpfr_t half;
    mpfr_init2(half, MPFR_PREC);

    mpfr_set_flt(half, 0.5f, rnd);
    mpfr_sqrt(rop, ds, rnd);
    mpfr_div(rop, half, rop, rnd);
    mpfr_mul(rop, rop, dd, rnd);

    return 0;
}

int mpfr_log_bwd(mpfr_t rop, const mpfr_t ds, const mpfr_t dd, mpfr_rnd_t rnd)
{
    mpfr_div(rop, dd, ds, rnd);
    return 0;
}

int mpfr_tanh_bwd(mpfr_t rop, const mpfr_t x, const mpfr_t y, mpfr_rnd_t rnd)
{
    mpfr_t tanh_op;
    mpfr_t const_one;
    mpfr_t op1;
    mpfr_t op2;
    mpfr_init2(tanh_op, MPFR_PREC);
    mpfr_init2(const_one, MPFR_PREC);
    mpfr_init2(op1, MPFR_PREC);
    mpfr_init2(op2, MPFR_PREC);

    mpfr_set_flt(const_one, 1.0f, rnd);
    mpfr_tanh(tanh_op, x, rnd);
    mpfr_sub(op1, const_one, tanh_op, rnd);
    mpfr_add(op2, const_one, tanh_op, rnd);
    mpfr_mul(rop, y, op1, rnd);
    mpfr_mul(rop, rop, op2, rnd);

    mpfr_clear(tanh_op);
    mpfr_clear(const_one);
    mpfr_clear(op1);
    mpfr_clear(op2);
    return 0;
}

int mpfr_clip(mpfr_t rop, const mpfr_t a, const mpfr_t b, const mpfr_t x, mpfr_rnd_t rnd)
{
    float alpha = mpfr_get_flt(a, rnd);
    float beta = mpfr_get_flt(b, rnd);
    float s = mpfr_get_flt(x, rnd);

    float res = s > alpha ? s : alpha;
    res = res > beta ? beta : res;

    mpfr_set_flt(rop, res, rnd);
    return 0;
}

int mpfr_clip_bwd(mpfr_t rop, const mpfr_t a, const mpfr_t b, const mpfr_t x, const mpfr_t y, mpfr_rnd_t rnd)
{
    float alpha = mpfr_get_flt(a, rnd);
    float beta = mpfr_get_flt(b, rnd);
    float s = mpfr_get_flt(x, rnd);
    float dd = mpfr_get_flt(y, rnd);

    float res = dd * (alpha < s && s <= beta ? 1 : 0);

    mpfr_set_flt(rop, res, rnd);
    return 0;
}

int mpfr_clip_v2(mpfr_t rop, const mpfr_t a, const mpfr_t b, const mpfr_t x, mpfr_rnd_t rnd)
{
    float alpha = mpfr_get_flt(a, rnd);
    float beta = mpfr_get_flt(b, rnd);
    float s = mpfr_get_flt(x, rnd);

    float res = s > alpha ? s : alpha;
    res = res < beta ? res : beta;

    mpfr_set_flt(rop, res, rnd);
    return 0;
}

int mpfr_clip_v2_bwd(mpfr_t rop, const mpfr_t a, const mpfr_t b, const mpfr_t x, const mpfr_t y, mpfr_rnd_t rnd)
{
    float alpha = mpfr_get_flt(a, rnd);
    float beta = mpfr_get_flt(b, rnd);
    float s = mpfr_get_flt(x, rnd);
    float dd = mpfr_get_flt(y, rnd);

    float res = dd * (alpha < s && s < beta ? 1 : 0);

    mpfr_set_flt(rop, res, rnd);
    return 0;
}

int mpfr_swish(mpfr_t rop, const mpfr_t a, const mpfr_t x, mpfr_rnd_t rnd)
{
    float alpha = mpfr_get_flt(a, rnd);
    float s = mpfr_get_flt(x, rnd);

    float res = s * logistic_fwd(alpha * s);
    mpfr_set_flt(rop, res, rnd);
    return 0;
}

int mpfr_swish_bwd(mpfr_t rop, const mpfr_t a, const mpfr_t x, const mpfr_t y, mpfr_rnd_t rnd)
{
    float alpha = mpfr_get_flt(a, rnd);
    float s = mpfr_get_flt(x, rnd);
    float dd = mpfr_get_flt(y, rnd);

    float v = logistic_fwd(alpha * s);
    float res = dd * (v + s * alpha * v * (1 - v));
    mpfr_set_flt(rop, res, rnd);
    return 0;
}

float soft_relu_fwd(float s, float alpha) {
    float exp_overflow_bound = 88.72283172607421875;
    float in = (float)s * (float)alpha;
    float v = (in < exp_overflow_bound ? log1pf(expf(in)) : in);
    return (v / alpha);
}

float soft_relu_bwd(float dd, float s, float alpha) {
    float in = (float)s * (float)alpha;
    return (dd * logistic_fwd(in));
}

int mpfr_mish(mpfr_t rop, const mpfr_t x, mpfr_rnd_t rnd)
{
    float s = mpfr_get_flt(x, rnd);

    float res = s * tanhf(soft_relu_fwd(s, 1.0f));
    mpfr_set_flt(rop, res, rnd);
    return 0;
}

int mpfr_mish_bwd(mpfr_t rop, const mpfr_t x, const mpfr_t y, mpfr_rnd_t rnd)
{
    float s = mpfr_get_flt(x, rnd);
    float dd = mpfr_get_flt(y, rnd);

    const float tanh = tanhf(soft_relu_fwd(s, 1.0f));
    const float srelu_bwd = soft_relu_bwd(1.0f, s, 1.0f);
    const float derivative = tanh + s * srelu_bwd * (1 - powf(tanh, 2.0f));
    float res = dd * derivative;

    mpfr_set_flt(rop, res, rnd);
    return 0;
}

int mpfr_gelu_tanh(mpfr_t rop, const mpfr_t x, mpfr_rnd_t rnd)
{
    float s = mpfr_get_flt(x, rnd);
    float sqrt_2_over_pi = 0.79788458347320556640625f;
    float fitting_const = 0.044715f;
    float v = tanhf(sqrt_2_over_pi * s * (1 + fitting_const * s * s));
    float res = 0.5 * s * (1.0 + v);
    mpfr_set_flt(rop, res, rnd);
    return 0;
}

int mpfr_gelu_tanh_bwd(mpfr_t rop, const mpfr_t x, const mpfr_t y, mpfr_rnd_t rnd)
{
    float s = mpfr_get_flt(x, rnd);
    float dd = mpfr_get_flt(y, rnd);
    float sqrt_2_over_pi = 0.79788458347320556640625f;
    float fitting_const = 0.044715f;
    float g = s * sqrt_2_over_pi * (1 + fitting_const * s * s);
    float dg = sqrt_2_over_pi * (1 + 3 * fitting_const * s * s);
    float v = tanhf(g);
    float res = dd * 0.5 * (1.0 + v) * (1.0 + s * (1 - v) * dg);
    mpfr_set_flt(rop, res, rnd);
    return 0;
}

int mpfr_soft_relu(mpfr_t rop, const mpfr_t a, const mpfr_t x, mpfr_rnd_t rnd)
{
    float exp_overflow_bound = 88.72283172607421875;
    float s = mpfr_get_flt(x, rnd);
    float alpha = mpfr_get_flt(a, rnd);
    float in = s * alpha;
    float v = (in < exp_overflow_bound ? (log1pf(expf(in))) : in);

    mpfr_set_flt(rop, v / alpha, rnd);
    return 0;
}

int mpfr_soft_relu_bwd(mpfr_t rop, const mpfr_t a, const mpfr_t x, const mpfr_t y, mpfr_rnd_t rnd)
{
    double dd = mpfr_get_d(y, rnd);
    float s = mpfr_get_flt(x, rnd);
    float alpha = mpfr_get_flt(a, rnd);
    float in = s * alpha;
    float temp = 1.f / (1.f + expf(-in));

    mpfr_set_flt(rop, dd * temp, rnd);
    return 0;
}
int mpfr_clipv2dst_bwd(mpfr_t rop, const mpfr_t a, const mpfr_t b, const mpfr_t x, const mpfr_t y, mpfr_rnd_t rnd) {
    float alpha = mpfr_get_flt(a, rnd);
    float beta = mpfr_get_flt(b, rnd);
    float s = mpfr_get_flt(x, rnd);
    float dd = mpfr_get_flt(y, rnd);

    float res = (s > alpha && s < beta) ? dd : 0.f;

    mpfr_set_flt(rop, res, rnd);

    return 0;
}

int mpfr_sqrtdst_bwd(mpfr_t rop, const mpfr_t x, const mpfr_t y, mpfr_rnd_t rnd) {
    mpfr_t half;
    mpfr_init2(half, MPFR_PREC);
    mpfr_set_flt(half, 2.f, rnd);
    mpfr_mul(rop, x, half, rnd);
    mpfr_div(rop, y, rop, rnd);
    return 0;
}

int mpfr_eludst_bwd(mpfr_t rop, const mpfr_t a, const mpfr_t x, const mpfr_t y, mpfr_rnd_t rnd) {
    float s = mpfr_get_flt(x, rnd);
    if (s > 0.f) {
        float dd = mpfr_get_flt(y, rnd);
        mpfr_set_flt(rop, dd, rnd);
    } else {
        mpfr_add(rop, x, a, rnd);
        mpfr_mul(rop, y, rop, rnd);
    }
    return 0;
}

int mpfr_reludst_bwd(mpfr_t rop, const mpfr_t a, const mpfr_t x, const mpfr_t y, mpfr_rnd_t rnd) {
    float s = mpfr_get_flt(x, rnd);
    if (s > 0.f) {
        float dd = mpfr_get_flt(y, rnd);
        mpfr_set_flt(rop, dd, rnd);
    } else {
        mpfr_mul(rop, y, a, rnd);
    }
    return 0;
}

int mpfr_expdst_bwd(mpfr_t rop, const mpfr_t ds, const mpfr_t dd, mpfr_rnd_t rnd)
{
    float a = mpfr_get_flt(ds, rnd);
    float b = mpfr_get_flt(dd, rnd);
    mpfr_set_flt(rop, a * b, rnd);
    return 0;
}

int mpfr_tanhdst_bwd(mpfr_t rop, const mpfr_t ds, const mpfr_t dd, mpfr_rnd_t rnd)
{
    mpfr_t tmp1;
    mpfr_t tmp2;
    mpfr_t fone;
    mpfr_init2(fone, MPFR_PREC);
    mpfr_set_flt(fone, 1.0, rnd);
    mpfr_init2(tmp1, MPFR_PREC);
    mpfr_init2(tmp2, MPFR_PREC);

    mpfr_sub(tmp1, fone, ds, rnd);
    mpfr_add(tmp2, fone, ds, rnd);
    mpfr_mul(tmp1, tmp1, tmp2, rnd);
    mpfr_mul(rop, tmp1, dd, rnd);

    mpfr_clear(tmp1);
    mpfr_clear(tmp2);
    mpfr_clear(fone);
    return 0;
}

int mpfr_logisticdst_bwd(mpfr_t rop, const mpfr_t ds, const mpfr_t dd, mpfr_rnd_t rnd)
{
    mpfr_t fone;
    mpfr_init2(fone, MPFR_PREC);
    mpfr_set_flt(fone, 1.0, rnd);

    mpfr_t tmp;
    mpfr_init2(tmp, MPFR_PREC);
    mpfr_sub(tmp, fone, ds, rnd);
    mpfr_mul(rop, ds, tmp, rnd);
    mpfr_mul(rop, rop, dd, rnd);

    mpfr_clear(fone);
    mpfr_clear(tmp);
    return 0;
}

int mpfr_gelu_erf(mpfr_t rop, const mpfr_t src, mpfr_rnd_t rnd)
{
    double srcX = mpfr_get_d(src, rnd);
    const float sqrt_2_over_2 = 0.707106769084930419921875f;
    float v = srcX * sqrt_2_over_2;
    mpfr_set_flt(rop, 0.5f * srcX * (1.f + erff(v)), rnd);

    return 0;
}

int mpfr_gelu_erf_bwd(mpfr_t rop, const mpfr_t src, const mpfr_t dd, mpfr_rnd_t rnd)
{
    double srcX = mpfr_get_d(src, rnd);
    double ddX = mpfr_get_d(dd, rnd);
    const float two_over_sqrt_pi = 1.12837922573089599609375f;
    const float sqrt_2_over_2 = 0.707106769084930419921875f;
    float v = srcX * sqrt_2_over_2;
    mpfr_set_flt(rop, ddX * 0.5f * (1.f + erff(v) + v * two_over_sqrt_pi * expf(-v * v)), rnd);

    return 0;
}
