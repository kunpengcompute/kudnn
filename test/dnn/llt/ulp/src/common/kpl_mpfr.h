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

#ifndef KPL_MPFR_H
#define KPL_MPFR_H

#include <mpfr.h>
#include <stdint.h>
#include "types.h"
#include "com_complex.h"

int mpfr_mish(mpfr_t rop, const mpfr_t x, mpfr_rnd_t rnd);
int mpfr_mish_bwd(mpfr_t rop, const mpfr_t x, const mpfr_t y, mpfr_rnd_t rnd);
int mpfr_powxa(mpfr_t rop, const mpfr_t a, const mpfr_t y, const mpfr_t x, mpfr_rnd_t rnd);
int mpfr_powxa_bwd(mpfr_t rop, const mpfr_t a, const mpfr_t y, const mpfr_t x, const mpfr_t d, mpfr_rnd_t rnd);

int mpfr_linear(mpfr_t rop, const mpfr_t a, const mpfr_t b, const mpfr_t x, mpfr_rnd_t rnd);
int mpfr_relu(mpfr_t rop, const mpfr_t a, const mpfr_t x, mpfr_rnd_t rnd);
int mpfr_sigmoid(mpfr_t rop, const mpfr_t x, mpfr_rnd_t rnd);
int mpfr_gelu_tanh(mpfr_t rop, const mpfr_t x, mpfr_rnd_t rnd);
int mpfr_abs_bwd(mpfr_t rop, const mpfr_t x, const mpfr_t y, mpfr_rnd_t rnd);
int mpfr_exp_bwd(mpfr_t rop, const mpfr_t x, const mpfr_t y, mpfr_rnd_t rnd);
int mpfr_sqrt_bwd(mpfr_t rop, const mpfr_t x, const mpfr_t y, mpfr_rnd_t rnd);
int mpfr_log_bwd(mpfr_t rop, const mpfr_t x, const mpfr_t y, mpfr_rnd_t rnd);
int mpfr_tanh_bwd(mpfr_t rop, const mpfr_t x, const mpfr_t y, mpfr_rnd_t rnd);
int mpfr_elu(mpfr_t rop, const mpfr_t a, const mpfr_t x, mpfr_rnd_t rnd);
int mpfr_elu_bwd(mpfr_t rop, const mpfr_t a, const mpfr_t x, const mpfr_t y, mpfr_rnd_t rnd);
int mpfr_gelu_tanh_bwd(mpfr_t rop, const mpfr_t x, const mpfr_t y, mpfr_rnd_t rnd);
int mpfr_soft_relu(mpfr_t rop, const mpfr_t a, const mpfr_t x, mpfr_rnd_t rnd);
int mpfr_soft_relu_bwd(mpfr_t rop, const mpfr_t a, const mpfr_t x, const mpfr_t y, mpfr_rnd_t rnd);

int mpfr_swish(mpfr_t rop, const mpfr_t a, const mpfr_t x, mpfr_rnd_t rnd);
int mpfr_swish_bwd(mpfr_t rop, const mpfr_t a, const mpfr_t x, const mpfr_t y, mpfr_rnd_t rnd);

int mpfr_hardsigmoid(mpfr_t rop, const mpfr_t a, const mpfr_t b, const mpfr_t x, mpfr_rnd_t rnd);
int mpfr_hardsigmoid_bwd(mpfr_t rop, const mpfr_t a, const mpfr_t b, const mpfr_t x, const mpfr_t y, mpfr_rnd_t rnd);
int mpfr_hardswish(mpfr_t rop, const mpfr_t a, const mpfr_t b, const mpfr_t x, mpfr_rnd_t rnd);
int mpfr_hardswish_bwd(mpfr_t rop, const mpfr_t a, const mpfr_t b, const mpfr_t x, const mpfr_t y, mpfr_rnd_t rnd);

int mpfr_clip(mpfr_t rop, const mpfr_t a, const mpfr_t b, const mpfr_t x, mpfr_rnd_t rnd);
int mpfr_clip_bwd(mpfr_t rop, const mpfr_t a, const mpfr_t b, const mpfr_t x, const mpfr_t y, mpfr_rnd_t rnd);
int mpfr_clip_v2(mpfr_t rop, const mpfr_t a, const mpfr_t b, const mpfr_t x, mpfr_rnd_t rnd);
int mpfr_clip_v2_bwd(mpfr_t rop, const mpfr_t a, const mpfr_t b, const mpfr_t x, const mpfr_t y, mpfr_rnd_t rnd);

int mpfr_clipv2dst_bwd(mpfr_t rop, const mpfr_t a, const mpfr_t b, const mpfr_t x, const mpfr_t y, mpfr_rnd_t rnd);
int mpfr_eludst_bwd(mpfr_t rop, const mpfr_t a, const mpfr_t x, const mpfr_t y, mpfr_rnd_t rnd);
int mpfr_reludst_bwd(mpfr_t rop, const mpfr_t a, const mpfr_t x, const mpfr_t y, mpfr_rnd_t rnd);
int mpfr_sqrtdst_bwd(mpfr_t rop, const mpfr_t x, const mpfr_t y, mpfr_rnd_t rnd);

int mpfr_expdst_bwd(mpfr_t rop, const mpfr_t x, const mpfr_t y, mpfr_rnd_t rnd);
int mpfr_tanhdst_bwd(mpfr_t rop, const mpfr_t x, const mpfr_t y, mpfr_rnd_t rnd);
int mpfr_logisticdst_bwd(mpfr_t rop, const mpfr_t x, const mpfr_t y, mpfr_rnd_t rnd);

int mpfr_gelu_erf(mpfr_t rop, const mpfr_t src, mpfr_rnd_t rnd);
int mpfr_gelu_erf_bwd(mpfr_t rop, const mpfr_t ds, const mpfr_t dd, mpfr_rnd_t rnd);

int mpfr_relu_bwd(mpfr_t rop, const mpfr_t a, const mpfr_t x, const mpfr_t y, mpfr_rnd_t rnd);
int mpfr_sigmoid_bwd(mpfr_t rop, const mpfr_t s, const mpfr_t dd, mpfr_rnd_t rnd);
int mpfr_sqr_bwd(mpfr_t rop, const mpfr_t x, const mpfr_t y, mpfr_rnd_t rnd);
#endif
