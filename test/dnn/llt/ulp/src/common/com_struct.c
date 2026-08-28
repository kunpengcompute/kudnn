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

#include "vml_mode.h"
#include "com_struct.h"

const struct FunctionInfo g_functions[] = {
    {"", "__vml_absh", "", (mathFunc_t)mpfr_abs, (mathFunc_t)__vml_absh, NULL, vmlh, ABS, FLAG_FWD, 0, {0, 0}},
    {"", "__vml_absb", "", (mathFunc_t)mpfr_abs, (mathFunc_t)__vml_absb, NULL, vmlb, ABS, FLAG_FWD, 0, {0, 0}},
    {"", "__vml_absf", "", (mathFunc_t)mpfr_abs, (mathFunc_t)__vml_absf, NULL, vmlf, ABS, FLAG_FWD, 0, {0, 0}},
    {"", "__vml_absh_bwd", "", (mathFunc_t)mpfr_abs_bwd, (mathFunc_t)__vml_absh_bwd, NULL, vmlh2, ABS, FLAG_BWD, 0, {0, 0}},
    {"", "__vml_absb_bwd", "", (mathFunc_t)mpfr_abs_bwd, (mathFunc_t)__vml_absb_bwd, NULL, vmlb2, ABS, FLAG_BWD, 0, {0, 0}},
    {"", "__vml_absf_bwd", "", (mathFunc_t)mpfr_abs_bwd, (mathFunc_t)__vml_absf_bwd, NULL, vmlf2, ABS, FLAG_BWD, 0, {0, 0}},

    {"", "__vml_hardsigmoidh", "", (mathFunc_t)mpfr_hardsigmoid, (mathFunc_t)__vml_hardsigmoidh, NULL, vmlxyh, HARDSIGMOID, FLAG_FWD, 0, {0, 0}},
    {"", "__vml_hardsigmoidb", "", (mathFunc_t)mpfr_hardsigmoid, (mathFunc_t)__vml_hardsigmoidb, NULL, vmlxyb, HARDSIGMOID, FLAG_FWD, 0, {0, 0}},
    {"", "__vml_hardsigmoidf", "", (mathFunc_t)mpfr_hardsigmoid, (mathFunc_t)__vml_hardsigmoidf, NULL, vmlxyf, HARDSIGMOID, FLAG_FWD, 0, {0, 0}},
    {"", "__vml_hardsigmoidh_bwd", "", (mathFunc_t)mpfr_hardsigmoid_bwd, (mathFunc_t)__vml_hardsigmoidh_bwd, NULL, vmlxyh2, HARDSIGMOID, FLAG_BWD, 0, {0, 0}},
    {"", "__vml_hardsigmoidb_bwd", "", (mathFunc_t)mpfr_hardsigmoid_bwd, (mathFunc_t)__vml_hardsigmoidb_bwd, NULL, vmlxyb2, HARDSIGMOID, FLAG_BWD, 0, {0, 0}},
    {"", "__vml_hardsigmoidf_bwd", "", (mathFunc_t)mpfr_hardsigmoid_bwd, (mathFunc_t)__vml_hardsigmoidf_bwd, NULL, vmlxyf2, HARDSIGMOID, FLAG_BWD, 0, {0, 0}},

    {"", "__vml_hardswishh", "", (mathFunc_t)mpfr_hardswish, (mathFunc_t)__vml_hardswishh, NULL, vmlxyh, HARDSWISH, FLAG_FWD, 0, {0, 0}},
    {"", "__vml_hardswishb", "", (mathFunc_t)mpfr_hardswish, (mathFunc_t)__vml_hardswishb, NULL, vmlxyb, HARDSWISH, FLAG_FWD, 0, {0, 0}},
    {"", "__vml_hardswishf", "", (mathFunc_t)mpfr_hardswish, (mathFunc_t)__vml_hardswishf, NULL, vmlxyf, HARDSWISH, FLAG_FWD, 0, {0, 0}},
    {"", "__vml_hardswishh_bwd", "", (mathFunc_t)mpfr_hardswish_bwd, (mathFunc_t)__vml_hardswishh_bwd, NULL, vmlxyh2, HARDSWISH, FLAG_BWD, 0, {0, 0}},
    {"", "__vml_hardswishb_bwd", "", (mathFunc_t)mpfr_hardswish_bwd, (mathFunc_t)__vml_hardswishb_bwd, NULL, vmlxyb2, HARDSWISH, FLAG_BWD, 0, {0, 0}},
    {"", "__vml_hardswishf_bwd", "", (mathFunc_t)mpfr_hardswish_bwd, (mathFunc_t)__vml_hardswishf_bwd, NULL, vmlxyf2, HARDSWISH, FLAG_BWD, 0, {0, 0}},

    {"", "__vml_exph", "", (mathFunc_t)mpfr_exp, (mathFunc_t)__vml_exph, NULL, vmlh, EXP, FLAG_FWD, 0, {0, 0}},
    {"", "__vml_expb", "", (mathFunc_t)mpfr_exp, (mathFunc_t)__vml_expb, NULL, vmlb, EXP, FLAG_FWD, 0, {0, 0}},
    {"", "__vml_expf", "", (mathFunc_t)mpfr_exp, (mathFunc_t)__vml_expf, NULL, vmlf, EXP, FLAG_FWD, 0, {0, 0}},
    {"", "__vml_exph_bwd", "", (mathFunc_t)mpfr_exp_bwd, (mathFunc_t)__vml_exph_bwd, NULL, vmlh2, EXP, FLAG_BWD, 0, {0, 0}},
    {"", "__vml_expb_bwd", "", (mathFunc_t)mpfr_exp_bwd, (mathFunc_t)__vml_expb_bwd, NULL, vmlb2, EXP, FLAG_BWD, 0, {0, 0}},
    {"", "__vml_expf_bwd", "", (mathFunc_t)mpfr_exp_bwd, (mathFunc_t)__vml_expf_bwd, NULL, vmlf2, EXP, FLAG_BWD, 0, {0, 0}},

    {"", "__vml_eluh", "", (mathFunc_t)mpfr_elu, (mathFunc_t)__vml_eluh, NULL, vmlxh, ELU, FLAG_FWD, 0, {0, 0}},
    {"", "__vml_elub", "", (mathFunc_t)mpfr_elu, (mathFunc_t)__vml_elub, NULL, vmlxb, ELU, FLAG_FWD, 0, {0, 0}},
    {"", "__vml_eluf", "", (mathFunc_t)mpfr_elu, (mathFunc_t)__vml_eluf, NULL, vmlxf, ELU, FLAG_FWD, 0, {0, 0}},
    {"", "__vml_eluh_bwd", "", (mathFunc_t)mpfr_elu_bwd, (mathFunc_t)__vml_eluh_bwd, NULL, vmlxh2, ELU, FLAG_BWD, 0, {0, 0}},
    {"", "__vml_elub_bwd", "", (mathFunc_t)mpfr_elu_bwd, (mathFunc_t)__vml_elub_bwd, NULL, vmlxb2, ELU, FLAG_BWD, 0, {0, 0}},
    {"", "__vml_eluf_bwd", "", (mathFunc_t)mpfr_elu_bwd, (mathFunc_t)__vml_eluf_bwd, NULL, vmlxf2, ELU, FLAG_BWD, 0, {0, 0}},

    {"", "__vml_soft_reluh", "", (mathFunc_t)mpfr_soft_relu, (mathFunc_t)__vml_soft_reluh, NULL, vmlxh, SRELU, FLAG_FWD, 0, {0, 0}},
    {"", "__vml_soft_relub", "", (mathFunc_t)mpfr_soft_relu, (mathFunc_t)__vml_soft_relub, NULL, vmlxb, SRELU, FLAG_FWD, 0, {0, 0}},
    {"", "__vml_soft_reluf", "", (mathFunc_t)mpfr_soft_relu, (mathFunc_t)__vml_soft_reluf, NULL, vmlxf, SRELU, FLAG_FWD, 0, {0, 0}},
    {"", "__vml_soft_reluh_bwd", "", (mathFunc_t)mpfr_soft_relu_bwd, (mathFunc_t)__vml_soft_reluh_bwd, NULL, vmlxh2, SRELU, FLAG_BWD, 0, {0, 0}},
    {"", "__vml_soft_relub_bwd", "", (mathFunc_t)mpfr_soft_relu_bwd, (mathFunc_t)__vml_soft_relub_bwd, NULL, vmlxb2, SRELU, FLAG_BWD, 0, {0, 0}},
    {"", "__vml_soft_reluf_bwd", "", (mathFunc_t)mpfr_soft_relu_bwd, (mathFunc_t)__vml_soft_reluf_bwd, NULL, vmlxf2, SRELU, FLAG_BWD, 0, {0, 0}},

    {"", "__vml_logh", "", (mathFunc_t)mpfr_log, (mathFunc_t)__vml_logh, NULL, vmlh, LOG, FLAG_FWD, 0, {0, 0}},
    {"", "__vml_logb", "", (mathFunc_t)mpfr_log, (mathFunc_t)__vml_logb, NULL, vmlb, LOG, FLAG_FWD, 0, {0, 0}},
    {"", "__vml_logf", "", (mathFunc_t)mpfr_log, (mathFunc_t)__vml_logf, NULL, vmlf, LOG, FLAG_FWD, 0, {0, 0}},
    {"", "__vml_logh_bwd", "", (mathFunc_t)mpfr_log_bwd, (mathFunc_t)__vml_logh_bwd, NULL, vmlh2, LOG, FLAG_BWD, 0, {0, 0}},
    {"", "__vml_logb_bwd", "", (mathFunc_t)mpfr_log_bwd, (mathFunc_t)__vml_logb_bwd, NULL, vmlb2, LOG, FLAG_BWD, 0, {0, 0}},
    {"", "__vml_logf_bwd", "", (mathFunc_t)mpfr_log_bwd, (mathFunc_t)__vml_logf_bwd, NULL, vmlf2, LOG, FLAG_BWD, 0, {0, 0}},

    {"", "__vml_nearbyinth", "", (mathFunc_t)mpfr_rint, (mathFunc_t)__vml_nearbyinth, NULL, vmlh, ROUND, FLAG_FWD, 0, {0, 0}},
    {"", "__vml_nearbyintf", "", (mathFunc_t)mpfr_rint, (mathFunc_t)__vml_nearbyintf, NULL, vmlf, ROUND, FLAG_FWD, 0, {0, 0}},
    {"", "__vml_nearbyintb", "", (mathFunc_t)mpfr_rint, (mathFunc_t)__vml_nearbyintb, NULL, vmlb, ROUND, FLAG_FWD, 0, {0, 0}},

    {"", "__vml_sqrth", "", (mathFunc_t)mpfr_sqrt, (mathFunc_t)__vml_sqrth, NULL, vmlh, SQRT, FLAG_FWD, 0, {0, 0}},
    {"", "__vml_sqrtb", "", (mathFunc_t)mpfr_sqrt, (mathFunc_t)__vml_sqrtb, NULL, vmlb, SQRT, FLAG_FWD, 0, {0, 0}},
    {"", "__vml_sqrtf", "", (mathFunc_t)mpfr_sqrt, (mathFunc_t)__vml_sqrtf, NULL, vmlf, SQRT, FLAG_FWD, 0, {0, 0}},
    {"", "__vml_sqrth_bwd", "", (mathFunc_t)mpfr_sqrt_bwd, (mathFunc_t)__vml_sqrth_bwd, NULL, vmlh2, SQRT, FLAG_BWD, 0, {0, 0}},
    {"", "__vml_sqrtb_bwd", "", (mathFunc_t)mpfr_sqrt_bwd, (mathFunc_t)__vml_sqrtb_bwd, NULL, vmlb2, SQRT, FLAG_BWD, 0, {0, 0}},
    {"", "__vml_sqrtf_bwd", "", (mathFunc_t)mpfr_sqrt_bwd, (mathFunc_t)__vml_sqrtf_bwd, NULL, vmlf2, SQRT, FLAG_BWD, 0, {0, 0}},

    {"", "__vml_sqrh", "", (mathFunc_t)mpfr_sqr, (mathFunc_t)__vml_sqrh, NULL, vmlh, SQUARE, FLAG_FWD, 0, {0, 0}},
    {"", "__vml_sqrb", "", (mathFunc_t)mpfr_sqr, (mathFunc_t)__vml_sqrb, NULL, vmlb, SQUARE, FLAG_FWD, 0, {0, 0}},
    {"", "__vml_sqrf", "", (mathFunc_t)mpfr_sqr, (mathFunc_t)__vml_sqrf, NULL, vmlf, SQUARE, FLAG_FWD, 0, {0, 0}},
    {"", "__vml_sqrh_bwd", "", (mathFunc_t)mpfr_sqr_bwd, (mathFunc_t)__vml_sqrh_bwd, NULL, vmlh2, SQUARE, FLAG_BWD, 0, {0, 0}},
    {"", "__vml_sqrb_bwd", "", (mathFunc_t)mpfr_sqr_bwd, (mathFunc_t)__vml_sqrb_bwd, NULL, vmlb2, SQUARE, FLAG_BWD, 0, {0, 0}},
    {"", "__vml_sqrf_bwd", "", (mathFunc_t)mpfr_sqr_bwd, (mathFunc_t)__vml_sqrf_bwd, NULL, vmlf2, SQUARE, FLAG_BWD, 0, {0, 0}},

    {"", "__vml_tanhh", "", (mathFunc_t)mpfr_tanh, (mathFunc_t)__vml_tanhh, NULL, vmlh, TANH, FLAG_FWD, 0, {0, 0}},
    {"", "__vml_tanhb", "", (mathFunc_t)mpfr_tanh, (mathFunc_t)__vml_tanhb, NULL, vmlb, TANH, FLAG_FWD, 0, {0, 0}},
    {"", "__vml_tanhf", "", (mathFunc_t)mpfr_tanh, (mathFunc_t)__vml_tanhf, NULL, vmlf, TANH, FLAG_FWD, 0, {0, 0}},
    {"", "__vml_tanhh_bwd", "", (mathFunc_t)mpfr_tanh_bwd, (mathFunc_t)__vml_tanhh_bwd, NULL, vmlh2, TANH, FLAG_BWD, 0, {0, 0}},
    {"", "__vml_tanhb_bwd", "", (mathFunc_t)mpfr_tanh_bwd, (mathFunc_t)__vml_tanhb_bwd, NULL, vmlb2, TANH, FLAG_BWD, 0, {0, 0}},
    {"", "__vml_tanhf_bwd", "", (mathFunc_t)mpfr_tanh_bwd, (mathFunc_t)__vml_tanhf_bwd, NULL, vmlf2, TANH, FLAG_BWD, 0, {0, 0}},

    {"", "__vml_mishf", "", (mathFunc_t)mpfr_mish, (mathFunc_t)__vml_mishf, NULL, vmlf, MISH, FLAG_FWD, 0, {0, 0}},
    {"", "__vml_mishb", "", (mathFunc_t)mpfr_mish, (mathFunc_t)__vml_mishb, NULL, vmlb, MISH, FLAG_FWD, 0, {0, 0}},
    {"", "__vml_mishh", "", (mathFunc_t)mpfr_mish, (mathFunc_t)__vml_mishh, NULL, vmlh, MISH, FLAG_FWD, 0, {0, 0}},
    {"", "__vml_mishf_bwd", "", (mathFunc_t)mpfr_mish_bwd, (mathFunc_t)__vml_mishf_bwd, NULL, vmlf2, MISH, FLAG_BWD, 0, {0, 0}},
    {"", "__vml_mishb_bwd", "", (mathFunc_t)mpfr_mish_bwd, (mathFunc_t)__vml_mishb_bwd, NULL, vmlb2, MISH, FLAG_BWD, 0, {0, 0}},
    {"", "__vml_mishh_bwd", "", (mathFunc_t)mpfr_mish_bwd, (mathFunc_t)__vml_mishh_bwd, NULL, vmlh2, MISH, FLAG_BWD, 0, {0, 0}},

    {"", "__vml_gelu_tanhf", "", (mathFunc_t)mpfr_gelu_tanh, (mathFunc_t)__vml_gelu_tanhf, NULL, vmlf, GELU_TANH, FLAG_FWD, 0, {0, 0}},
    {"", "__vml_gelu_tanhb", "", (mathFunc_t)mpfr_gelu_tanh, (mathFunc_t)__vml_gelu_tanhb, NULL, vmlb, GELU_TANH, FLAG_FWD, 0, {0, 0}},
    {"", "__vml_gelu_tanhh", "", (mathFunc_t)mpfr_gelu_tanh, (mathFunc_t)__vml_gelu_tanhh, NULL, vmlh, GELU_TANH, FLAG_FWD, 0, {0, 0}},
    {"", "__vml_gelu_tanhf_bwd", "", (mathFunc_t)mpfr_gelu_tanh_bwd, (mathFunc_t)__vml_gelu_tanhf_bwd, NULL, vmlf2, GELU_TANH, FLAG_BWD, 0, {0, 0}},
    {"", "__vml_gelu_tanhb_bwd", "", (mathFunc_t)mpfr_gelu_tanh_bwd, (mathFunc_t)__vml_gelu_tanhb_bwd, NULL, vmlb2, GELU_TANH, FLAG_BWD, 0, {0, 0}},
    {"", "__vml_gelu_tanhh_bwd", "", (mathFunc_t)mpfr_gelu_tanh_bwd, (mathFunc_t)__vml_gelu_tanhh_bwd, NULL, vmlh2, GELU_TANH, FLAG_BWD, 0, {0, 0}},

    {"", "__vml_powxah", "", (mathFunc_t)mpfr_powxa, (mathFunc_t)__vml_powxah, NULL, vmlxyh, POW, FLAG_FWD, 0, {0, 0, 0, 0, 0, 0}},
    {"", "__vml_powxab", "", (mathFunc_t)mpfr_powxa, (mathFunc_t)__vml_powxab, NULL, vmlxyb, POW, FLAG_FWD, 0, {0, 0, 0, 0, 0, 0}},
    {"", "__vml_powxaf", "", (mathFunc_t)mpfr_powxa, (mathFunc_t)__vml_powxaf, NULL, vmlxyf, POW, FLAG_FWD, 0, {0, 0, 0, 0, 0, 0}},
    {"", "__vml_powxah_bwd", "", (mathFunc_t)mpfr_powxa_bwd, (mathFunc_t)__vml_powxah_bwd, NULL, vmlxyh2, POW, FLAG_BWD, 0, {0, 0, 0, 0, 0, 0}},
    {"", "__vml_powxab_bwd", "", (mathFunc_t)mpfr_powxa_bwd, (mathFunc_t)__vml_powxab_bwd, NULL, vmlxyb2, POW, FLAG_BWD, 0, {0, 0, 0, 0, 0, 0}},
    {"", "__vml_powxaf_bwd", "", (mathFunc_t)mpfr_powxa_bwd, (mathFunc_t)__vml_powxaf_bwd, NULL, vmlxyf2, POW, FLAG_BWD, 0, {0, 0, 0, 0, 0, 0}},

    {"", "__vml_sigmoidh", "", (mathFunc_t)mpfr_sigmoid, (mathFunc_t)__vml_sigmoidh, NULL, vmlh, LOGISTIC, FLAG_FWD, 0, {0, 0}},
    {"", "__vml_sigmoidb", "", (mathFunc_t)mpfr_sigmoid, (mathFunc_t)__vml_sigmoidb, NULL, vmlb, LOGISTIC, FLAG_FWD, 0, {0, 0}},
    {"", "__vml_sigmoidf", "", (mathFunc_t)mpfr_sigmoid, (mathFunc_t)__vml_sigmoidf, NULL, vmlf, LOGISTIC, FLAG_FWD, 0, {0, 0}},
    {"", "__vml_sigmoidh_bwd", "", (mathFunc_t)mpfr_sigmoid_bwd, (mathFunc_t)__vml_sigmoidh_bwd, NULL, vmlh2, LOGISTIC, FLAG_BWD, 0, {0, 0}},
    {"", "__vml_sigmoidb_bwd", "", (mathFunc_t)mpfr_sigmoid_bwd, (mathFunc_t)__vml_sigmoidb_bwd, NULL, vmlb2, LOGISTIC, FLAG_BWD, 0, {0, 0}},
    {"", "__vml_sigmoidf_bwd", "", (mathFunc_t)mpfr_sigmoid_bwd, (mathFunc_t)__vml_sigmoidf_bwd, NULL, vmlf2, LOGISTIC, FLAG_BWD, 0, {0, 0}},

    {"", "__vml_swishf", "", (mathFunc_t)mpfr_swish, (mathFunc_t)__vml_swishf, NULL, vmlxf, SWISH, FLAG_FWD, 0, {0, 0}},
    {"", "__vml_swishb", "", (mathFunc_t)mpfr_swish, (mathFunc_t)__vml_swishb, NULL, vmlxb, SWISH, FLAG_FWD, 0, {0, 0}},
    {"", "__vml_swishh", "", (mathFunc_t)mpfr_swish, (mathFunc_t)__vml_swishh, NULL, vmlxh, SWISH, FLAG_FWD, 0, {0, 0}},
    {"", "__vml_swishf_bwd", "", (mathFunc_t)mpfr_swish_bwd, (mathFunc_t)__vml_swishf_bwd, NULL, vmlxf2, SWISH, FLAG_BWD, 0, {0, 0}},
    {"", "__vml_swishb_bwd", "", (mathFunc_t)mpfr_swish_bwd, (mathFunc_t)__vml_swishb_bwd, NULL, vmlxb2, SWISH, FLAG_BWD, 0, {0, 0}},
    {"", "__vml_swishh_bwd", "", (mathFunc_t)mpfr_swish_bwd, (mathFunc_t)__vml_swishh_bwd, NULL, vmlxh2, SWISH, FLAG_BWD, 0, {0, 0}},

    {"", "__vml_reluh", "", (mathFunc_t)mpfr_relu, (mathFunc_t)__vml_reluh, NULL, vmlxh, RELU, FLAG_FWD, 0, {0, 0, 0, 0}},
    {"", "__vml_relub", "", (mathFunc_t)mpfr_relu, (mathFunc_t)__vml_relub, NULL, vmlxb, RELU, FLAG_FWD, 0, {0, 0, 0, 0}},
    {"", "__vml_reluf", "", (mathFunc_t)mpfr_relu, (mathFunc_t)__vml_reluf, NULL, vmlxf, RELU, FLAG_FWD, 0, {0, 0, 0, 0}},
    {"", "__vml_reluh_bwd", "", (mathFunc_t)mpfr_relu_bwd, (mathFunc_t)__vml_reluh_bwd, NULL, vmlxh2, RELU, FLAG_BWD, 0, {0, 0, 0, 0}},
    {"", "__vml_relub_bwd", "", (mathFunc_t)mpfr_relu_bwd, (mathFunc_t)__vml_relub_bwd, NULL, vmlxb2, RELU, FLAG_BWD, 0, {0, 0, 0, 0}},
    {"", "__vml_reluf_bwd", "", (mathFunc_t)mpfr_relu_bwd, (mathFunc_t)__vml_reluf_bwd, NULL, vmlxf2, RELU, FLAG_BWD, 0, {0, 0, 0, 0}},

    {"", "__vml_linearh", "", (mathFunc_t)mpfr_linear, (mathFunc_t)__vml_linearh, NULL, vmlxyh, LINEAR, FLAG_FWD, 0, {0, 0, 0, 0}},
    {"", "__vml_linearb", "", (mathFunc_t)mpfr_linear, (mathFunc_t)__vml_linearb, NULL, vmlxyb, LINEAR, FLAG_FWD, 0, {0, 0, 0, 0}},
    {"", "__vml_linearf", "", (mathFunc_t)mpfr_linear, (mathFunc_t)__vml_linearf, NULL, vmlxyf, LINEAR, FLAG_FWD, 0, {0, 0, 0, 0}},
    {"", "__vml_linearh_bwd", "", (mathFunc_t)mpfr_linear, (mathFunc_t)__vml_linearh, NULL, vmlxyh, LINEAR, FLAG_BWD, 0, {0, 0, 0, 0}},
    {"", "__vml_linearb_bwd", "", (mathFunc_t)mpfr_linear, (mathFunc_t)__vml_linearb, NULL, vmlxyb, LINEAR, FLAG_BWD, 0, {0, 0, 0, 0}},
    {"", "__vml_linearf_bwd", "", (mathFunc_t)mpfr_linear, (mathFunc_t)__vml_linearf, NULL, vmlxyf, LINEAR, FLAG_BWD, 0, {0, 0, 0, 0}},

    {"", "__vml_cliph", "", (mathFunc_t)mpfr_clip, (mathFunc_t)__vml_cliph, NULL, vmlxyh, CLIP, FLAG_FWD, 0, {0, 0}},
    {"", "__vml_clipb", "", (mathFunc_t)mpfr_clip, (mathFunc_t)__vml_clipb, NULL, vmlxyb, CLIP, FLAG_FWD, 0, {0, 0}},
    {"", "__vml_clipf", "", (mathFunc_t)mpfr_clip, (mathFunc_t)__vml_clipf, NULL, vmlxyf, CLIP, FLAG_FWD, 0, {0, 0}},
    {"", "__vml_cliph_bwd", "", (mathFunc_t)mpfr_clip_bwd, (mathFunc_t)__vml_cliph_bwd, NULL, vmlxyh2, CLIP, FLAG_BWD, 0, {0, 0}},
    {"", "__vml_clipb_bwd", "", (mathFunc_t)mpfr_clip_bwd, (mathFunc_t)__vml_clipb_bwd, NULL, vmlxyb2, CLIP, FLAG_BWD, 0, {0, 0}},
    {"", "__vml_clipf_bwd", "", (mathFunc_t)mpfr_clip_bwd, (mathFunc_t)__vml_clipf_bwd, NULL, vmlxyf2, CLIP, FLAG_BWD, 0, {0, 0}},

    {"", "__vml_clip_v2h", "", (mathFunc_t)mpfr_clip_v2, (mathFunc_t)__vml_clip_v2h, NULL, vmlxyh, CLIP_V2, FLAG_FWD, 0, {0, 0}},
    {"", "__vml_clip_v2b", "", (mathFunc_t)mpfr_clip_v2, (mathFunc_t)__vml_clip_v2b, NULL, vmlxyb, CLIP_V2, FLAG_FWD, 0, {0, 0}},
    {"", "__vml_clip_v2f", "", (mathFunc_t)mpfr_clip_v2, (mathFunc_t)__vml_clip_v2f, NULL, vmlxyf, CLIP_V2, FLAG_FWD, 0, {0, 0}},
    {"", "__vml_clip_v2h_bwd", "", (mathFunc_t)mpfr_clip_v2_bwd, (mathFunc_t)__vml_clip_v2h_bwd, NULL, vmlxyh2, CLIP_V2, FLAG_BWD, 0, {0, 0}},
    {"", "__vml_clip_v2b_bwd", "", (mathFunc_t)mpfr_clip_v2_bwd, (mathFunc_t)__vml_clip_v2b_bwd, NULL, vmlxyb2, CLIP_V2, FLAG_BWD, 0, {0, 0}},
    {"", "__vml_clip_v2f_bwd", "", (mathFunc_t)mpfr_clip_v2_bwd, (mathFunc_t)__vml_clip_v2f_bwd, NULL, vmlxyf2, CLIP_V2, FLAG_BWD, 0, {0, 0}},

    {"", "__vml_clipv2dstf_bwd", "", (mathFunc_t)mpfr_clipv2dst_bwd, (mathFunc_t)__vml_clipv2dstf_bwd, NULL, vmlxyf2, CLIP_V2_DST, FLAG_BWD, 0, {0, 0}},
    {"", "__vml_clipv2dsth_bwd", "", (mathFunc_t)mpfr_clipv2dst_bwd, (mathFunc_t)__vml_clipv2dsth_bwd, NULL, vmlxyh2, CLIP_V2_DST, FLAG_BWD, 0, {0, 0}},
    {"", "__vml_clipv2dstb_bwd", "", (mathFunc_t)mpfr_clipv2dst_bwd, (mathFunc_t)__vml_clipv2dstb_bwd, NULL, vmlxyb2, CLIP_V2_DST, FLAG_BWD, 0, {0, 0}},

    {"", "__vml_sqrtdstf_bwd", "", (mathFunc_t)mpfr_sqrtdst_bwd, (mathFunc_t)__vml_sqrtdstf_bwd, NULL, vmlf2, SQRT_DST, FLAG_BWD, 0, {0, 0}},
    {"", "__vml_sqrtdsth_bwd", "", (mathFunc_t)mpfr_sqrtdst_bwd, (mathFunc_t)__vml_sqrtdsth_bwd, NULL, vmlh2, SQRT_DST, FLAG_BWD, 0, {0, 0}},
    {"", "__vml_sqrtdstb_bwd", "", (mathFunc_t)mpfr_sqrtdst_bwd, (mathFunc_t)__vml_sqrtdstb_bwd, NULL, vmlb2, SQRT_DST, FLAG_BWD, 0, {0, 0}},

    {"", "__vml_eludstf_bwd", "", (mathFunc_t)mpfr_eludst_bwd, (mathFunc_t)__vml_eludstf_bwd, NULL, vmlxf2, ELU_DST, FLAG_BWD, 0, {0, 0}},
    {"", "__vml_eludsth_bwd", "", (mathFunc_t)mpfr_eludst_bwd, (mathFunc_t)__vml_eludsth_bwd, NULL, vmlxh2, ELU_DST, FLAG_BWD, 0, {0, 0}},
    {"", "__vml_eludstb_bwd", "", (mathFunc_t)mpfr_eludst_bwd, (mathFunc_t)__vml_eludstb_bwd, NULL, vmlxb2, ELU_DST, FLAG_BWD, 0, {0, 0}},

    {"", "__vml_reludstf_bwd", "", (mathFunc_t)mpfr_reludst_bwd, (mathFunc_t)__vml_reludstf_bwd, NULL, vmlxf2, RELU_DST, FLAG_BWD, 0, {0, 0}},
    {"", "__vml_reludsth_bwd", "", (mathFunc_t)mpfr_reludst_bwd, (mathFunc_t)__vml_reludsth_bwd, NULL, vmlxh2, RELU_DST, FLAG_BWD, 0, {0, 0}},
    {"", "__vml_reludstb_bwd", "", (mathFunc_t)mpfr_reludst_bwd, (mathFunc_t)__vml_reludstb_bwd, NULL, vmlxb2, RELU_DST, FLAG_BWD, 0, {0, 0}},

    {"", "__vml_logisticdstf_bwd", "", (mathFunc_t)mpfr_logisticdst_bwd, (mathFunc_t)__vml_logisticdstf_bwd, NULL, vmlf2, LOGISTIC_DST, FLAG_BWD, 0, {0, 0}},
    {"", "__vml_logisticdsth_bwd", "", (mathFunc_t)mpfr_logisticdst_bwd, (mathFunc_t)__vml_logisticdsth_bwd, NULL, vmlh2, LOGISTIC_DST, FLAG_BWD, 0, {0, 0}},
    {"", "__vml_logisticdstb_bwd", "", (mathFunc_t)mpfr_logisticdst_bwd, (mathFunc_t)__vml_logisticdstb_bwd, NULL, vmlb2, LOGISTIC_DST, FLAG_BWD, 0, {0, 0}},

    {"", "__vml_expdstf_bwd", "", (mathFunc_t)mpfr_expdst_bwd, (mathFunc_t)__vml_expdstf_bwd, NULL, vmlf2, EXP_DST, FLAG_BWD, 0, {0, 0}},
    {"", "__vml_expdsth_bwd", "", (mathFunc_t)mpfr_expdst_bwd, (mathFunc_t)__vml_expdsth_bwd, NULL, vmlh2, EXP_DST, FLAG_BWD, 0, {0, 0}},
    {"", "__vml_expdstb_bwd", "", (mathFunc_t)mpfr_expdst_bwd, (mathFunc_t)__vml_expdstb_bwd, NULL, vmlb2, EXP_DST, FLAG_BWD, 0, {0, 0}},

    {"", "__vml_tanhdstf_bwd", "", (mathFunc_t)mpfr_tanhdst_bwd, (mathFunc_t)__vml_tanhdstf_bwd, NULL, vmlf2, TANH_DST, FLAG_BWD, 0, {0, 0}},
    {"", "__vml_tanhdsth_bwd", "", (mathFunc_t)mpfr_tanhdst_bwd, (mathFunc_t)__vml_tanhdsth_bwd, NULL, vmlh2, TANH_DST, FLAG_BWD, 0, {0, 0}},
    {"", "__vml_tanhdstb_bwd", "", (mathFunc_t)mpfr_tanhdst_bwd, (mathFunc_t)__vml_tanhdstb_bwd, NULL, vmlb2, TANH_DST, FLAG_BWD, 0, {0, 0}},

    {"", "__vml_gelu_erfh", "", (mathFunc_t)mpfr_gelu_erf, (mathFunc_t)__vml_gelu_erfh, NULL, vmlh, GELU_ERF, FLAG_FWD, 0, {0, 0}},
    {"", "__vml_gelu_erfb", "", (mathFunc_t)mpfr_gelu_erf, (mathFunc_t)__vml_gelu_erfb, NULL, vmlb, GELU_ERF, FLAG_FWD, 0, {0, 0}},
    {"", "__vml_gelu_erff", "", (mathFunc_t)mpfr_gelu_erf, (mathFunc_t)__vml_gelu_erff, NULL, vmlf, GELU_ERF, FLAG_FWD, 0, {0, 0}},
    {"", "__vml_gelu_erfh_bwd", "", (mathFunc_t)mpfr_gelu_erf_bwd, (mathFunc_t)__vml_gelu_erfh_bwd, NULL, vmlh2, GELU_ERF, FLAG_BWD, 0, {0, 0}},
    {"", "__vml_gelu_erfb_bwd", "", (mathFunc_t)mpfr_gelu_erf_bwd, (mathFunc_t)__vml_gelu_erfb_bwd, NULL, vmlb2, GELU_ERF, FLAG_BWD, 0, {0, 0}},
    {"", "__vml_gelu_erff_bwd", "", (mathFunc_t)mpfr_gelu_erf_bwd, (mathFunc_t)__vml_gelu_erff_bwd, NULL, vmlf2, GELU_ERF, FLAG_BWD, 0, {0, 0}},

    {0}
};
const int nfunctions = (sizeof(g_functions) / sizeof(g_functions[0]));

/**
 * @generate random seeds.
 **/
uint64_t GetRandomU64()
{
    uint64_t seed;
    int fd = open("/dev/urandom", O_RDONLY);
    read(fd, &seed, sizeof(uint64_t));
    close(fd);
    return seed;
}

void gencase_h_new(__fp16 *arr, struct Interval *interval, TEST_MODE testMode, int64_t length)
{
    for (int64_t i = 0; i < length; i++) {
        uint32_t res;
        cases_normal_float16(&res, interval->low, interval->high);
        *(arr + i) = Asfloat16(res);
    }
}

void gencase_b_new(__bf16 *arr, struct Interval *interval, TEST_MODE testMode, int64_t length)
{
    for (int64_t i = 0; i < length; i++) {
        uint32_t res;
        cases_normal_float16(&res, interval->low, interval->high);
        *(arr + i) = Asbfloat16(res);
    }
}

void gencase_f_new(float *arr, struct Interval *interval, TEST_MODE testMode, int64_t length)
{
    for (int64_t i = 0; i < length; i++) {
        uint32_t res;
        cases_normal_float(&res, interval->low, interval->high);
        *(arr + i) = Asfloat(res);
    }
}

int get_exp_h(mpfr_t x)
{
    /* TODO: pow of 2 cases.  */
    if (mpfr_regular_p(x)) {
        mpfr_exp_t e = mpfr_get_exp(x) - prec_h;
        if (e < emin_h) {
            e = emin_h - 1;
        }
        if (e > emax_h - prec_h) {
            e = emax_h - prec_h;
        }
        return e;
    }
    if (mpfr_zero_p(x)) {
        return emin_h - 1;
    }
    if (mpfr_inf_p(x)) {
        return emax_h - prec_h;
    }
    /* NaN.  */
    return 0;
}

int get_exp_f(mpfr_t x)
{
    /* TODO: pow of 2 cases.  */
    if (mpfr_regular_p(x)) {
        mpfr_exp_t e = mpfr_get_exp(x) - prec_f;
        if (e < emin_f) {
            e = emin_f - 1;
        }
        if (e > emax_f - prec_f) {
            e = emax_f - prec_f;
        }
        return e;
    }
    if (mpfr_zero_p(x)) {
        return emin_f - 1;
    }
    if (mpfr_inf_p(x)) {
        return emax_f - prec_f;
    }
    /* NaN.  */
    return 0;
}

void CalUlpF16(struct MpfrRes* mpfrRes, int t, mpfr_t my)
{
    int t2;

    MPFR_DECL_INIT(mr, prec_h);
    MPFR_DECL_INIT(me, prec_mpfr_h);
    mpfr_clear_flags();

    mpfrRes->ulpexp = get_exp_h(my);
    /* Double rounding.  */
    t2 = mpfr_set(mr, my, FE_TONEAREST);
    if (t2) {
        t = t2;
    }

    mpfr_set_emin(emin_h);
    mpfr_set_emax(emax_h);
    t = mpfr_check_range(mr, t, FE_TONEAREST);
    t = mpfr_subnormalize(mr, t, FE_TONEAREST);
    mpfr_set_emax(MPFR_EMAX_DEFAULT);
    mpfr_set_emin(MPFR_EMIN_DEFAULT);
    mpfrRes->res.h = mpfr_get_d(mr, FE_TONEAREST);
    if (!isfinite(mpfrRes->res.h)) {
        mpfrRes->resTail = 0;
        mpfr_set_si_2exp(mr, signbit(mpfrRes->res.h) ? -1 : 1, 1024, MPFR_RNDN);
    }
    mpfr_sub(me, my, mr, MPFR_RNDN);
    mpfr_mul_2si(me, me, -mpfrRes->ulpexp, MPFR_RNDN);
    mpfrRes->resTail = mpfr_get_d(me, MPFR_RNDN);
}

void CalUlpFloat(struct MpfrRes* mpfrRes, int t, mpfr_t my)
{
    int t2;

    MPFR_DECL_INIT(mr, prec_f);
    MPFR_DECL_INIT(me, prec_mpfr_f);
    mpfr_clear_flags();

    mpfrRes->ulpexp = get_exp_f(my);
    /* Double rounding.  */
    t2 = mpfr_set(mr, my, FE_TONEAREST);
    if (t2) {
        t = t2;
    }
    mpfr_set_emin(emin_f);
    mpfr_set_emax(emax_f);
    t = mpfr_check_range(mr, t, FE_TONEAREST);
    t = mpfr_subnormalize(mr, t, FE_TONEAREST);
    mpfr_set_emax(MPFR_EMAX_DEFAULT);
    mpfr_set_emin(MPFR_EMIN_DEFAULT);
    mpfrRes->res.f = mpfr_get_d(mr, FE_TONEAREST);
    if (!isfinite(mpfrRes->res.f)) {
        mpfrRes->resTail = 0;
        mpfr_set_si_2exp(mr, signbit(mpfrRes->res.f) ? -1 : 1, 1024, MPFR_RNDN);
    }
    mpfr_sub(me, my, mr, MPFR_RNDN);
    mpfr_mul_2si(me, me, -mpfrRes->ulpexp, MPFR_RNDN);
    mpfrRes->resTail = mpfr_get_d(me, MPFR_RNDN);
}

static inline int call_mpfr_f3(mpfr_t y, const struct FunctionInfo *funcInfo, float *cases, mpfr_rnd_t r)
{
    MPFR_DECL_INIT(x, prec_f);
    mpfr_set_flt(x, cases[0], MPFR_RNDN);
    MPFR_DECL_INIT(x2, prec_f);
    mpfr_set_flt(x2, cases[1], MPFR_RNDN);
    MPFR_DECL_INIT(x3, prec_f);
    mpfr_set_flt(x3, cases[2], MPFR_RNDN);
    return ((mpfr_vmlf3)(funcInfo->mpfrApi))(y, x, x2, x3, r);
}

static inline int call_mpfr_f4(mpfr_t y, const struct FunctionInfo *funcInfo, float *cases, mpfr_rnd_t r)
{
    MPFR_DECL_INIT(x, prec_f);
    mpfr_set_flt(x, cases[0], MPFR_RNDN);
    MPFR_DECL_INIT(x2, prec_f);
    mpfr_set_flt(x2, cases[1], MPFR_RNDN);
    MPFR_DECL_INIT(x3, prec_f);
    mpfr_set_flt(x3, cases[2], MPFR_RNDN);
    MPFR_DECL_INIT(x4, prec_f);
    mpfr_set_flt(x4, cases[3], MPFR_RNDN);
    return ((mpfr_vmlf4)(funcInfo->mpfrApi))(y, x, x2, x3, x4, r);
}

static inline int call_mpfr_hf2(mpfr_t y, const struct FunctionInfo *funcInfo, float *cases, mpfr_rnd_t r)
{
    MPFR_DECL_INIT(x, prec_f);
    mpfr_set_flt(x, (float)((__fp16)cases[0]), MPFR_RNDN);
    MPFR_DECL_INIT(x2, prec_f);
    mpfr_set_flt(x2, cases[1], MPFR_RNDN);
    MPFR_DECL_INIT(x3, prec_f);
    mpfr_set_flt(x3, cases[2], MPFR_RNDN);
    return ((mpfr_vmlf3)(funcInfo->mpfrApi))(y, x, x2, x3, r);
}

static inline int call_mpfr_f2(mpfr_t y, const struct FunctionInfo *funcInfo, float *cases, mpfr_rnd_t r)
{
    MPFR_DECL_INIT(x, prec_f);
    mpfr_set_flt(x, cases[0], MPFR_RNDN);
    MPFR_DECL_INIT(x2, prec_f);
    mpfr_set_flt(x2, cases[1], MPFR_RNDN);
    return ((mpfr_vmlf2)(funcInfo->mpfrApi))(y, x, x2, r);
}

static inline int call_mpfr_fh(mpfr_t y, const struct FunctionInfo *funcInfo, float *cases, mpfr_rnd_t r)
{
    MPFR_DECL_INIT(x, prec_f);
    mpfr_set_flt(x, cases[0], MPFR_RNDN);
    MPFR_DECL_INIT(x2, prec_h);
    mpfr_set_flt(x2, (__fp16)cases[1], MPFR_RNDN);
    return ((mpfr_vmlf2)(funcInfo->mpfrApi))(y, x, x2, r);
}

static inline int call_mpfr_f1(mpfr_t y, const struct FunctionInfo *funcInfo, float *cases, mpfr_rnd_t r)
{
    MPFR_DECL_INIT(x, prec_f);
    mpfr_set_flt(x, cases[0], MPFR_RNDN);
    return ((mpfr_vmlf1)(funcInfo->mpfrApi))(y, x, r);
}

static inline int call_mpfr_h1(mpfr_t y, const struct FunctionInfo *funcInfo, __fp16 *cases, mpfr_rnd_t r)
{
    MPFR_DECL_INIT(x, prec_h);
    mpfr_set_flt(x, cases[0], MPFR_RNDN);
    return ((mpfr_vmlh1)(funcInfo->mpfrApi))(y, x, r);
}

static inline int call_mpfr_h2(mpfr_t y, const struct FunctionInfo *funcInfo, __fp16 *cases, mpfr_rnd_t r)
{
    MPFR_DECL_INIT(x, prec_h);
    mpfr_set_flt(x, cases[0], MPFR_RNDN);
    MPFR_DECL_INIT(x2, prec_h);
    mpfr_set_flt(x2, cases[1], MPFR_RNDN);
    return ((mpfr_vmlf2)(funcInfo->mpfrApi))(y, x, x2, r);
}

void mpfr_result_hf2_common(const struct FunctionInfo *funcInfo, float *cases, struct MpfrRes* mpfrRes,
    int func(mpfr_t, const struct FunctionInfo *, float *, mpfr_rnd_t))
{
    int t;
    MPFR_DECL_INIT(my, prec_mpfr_h);
    t = func(my, funcInfo, cases, FE_TONEAREST);
    CalUlpF16(mpfrRes, t, my);
}

void mpfr_result_f_common(const struct FunctionInfo *funcInfo, float *cases, struct MpfrRes *mpfrRes,
    int func(mpfr_t, const struct FunctionInfo *, float *, mpfr_rnd_t))
{
    int t;
    MPFR_DECL_INIT(my, prec_mpfr_f);
    t = func(my, funcInfo, cases, FE_TONEAREST);
    CalUlpFloat(mpfrRes, t, my);
}

void mpfr_result_h_common(const struct FunctionInfo *funcInfo, __fp16 *cases, struct MpfrRes* mpfrRes,
    int func(mpfr_t, const struct FunctionInfo *, __fp16 *, mpfr_rnd_t))
{
    int t;
    MPFR_DECL_INIT(my, prec_mpfr_h);
    t = func(my, funcInfo, cases, FE_TONEAREST);
    CalUlpF16(mpfrRes, t, my);
}

int mpfr_result_f3(const struct FunctionInfo *funcInfo, float *cases, struct MpfrRes* mpfrRes)
{
    mpfr_result_f_common(funcInfo, cases, mpfrRes, call_mpfr_f3);
    return 0;
}

int mpfr_result_f4(const struct FunctionInfo *funcInfo, float *cases, struct MpfrRes* mpfrRes)
{
    mpfr_result_f_common(funcInfo, cases, mpfrRes, call_mpfr_f4);
    return 0;
}

int mpfr_result_hf2(const struct FunctionInfo *funcInfo, float *cases, struct MpfrRes* mpfrRes)
{
    mpfr_result_hf2_common(funcInfo, cases, mpfrRes, call_mpfr_hf2);
    return 0;
}

int mpfr_result_f2(const struct FunctionInfo *funcInfo, float *cases, struct MpfrRes* mpfrRes)
{
    mpfr_result_f_common(funcInfo, cases, mpfrRes, call_mpfr_f2);
    return 0;
}

int mpfr_result_fh(const struct FunctionInfo *funcInfo, float *cases, struct MpfrRes* mpfrRes)
{
    mpfr_result_hf2_common(funcInfo, cases, mpfrRes, call_mpfr_fh);
    return 0;
}

int mpfr_result_f(const struct FunctionInfo *funcInfo, float *cases, struct MpfrRes* mpfrRes)
{
    mpfr_result_f_common(funcInfo, cases, mpfrRes, call_mpfr_f1);
    return 0;
}

int mpfr_result_h(const struct FunctionInfo *funcInfo, __fp16 *cases, struct MpfrRes* mpfrRes)
{
    mpfr_result_h_common(funcInfo, cases, mpfrRes, call_mpfr_h1);
    return 0;
}

int mpfr_result_h2(const struct FunctionInfo *funcInfo, __fp16 *cases, struct MpfrRes* mpfrRes)
{
    mpfr_result_h_common(funcInfo, cases, mpfrRes, call_mpfr_h2);
    return 0;
}

/* ulperr_xxx function calculate ulp error */
double ulperr_h(__fp16 got, struct MpfrRes *mpfrRes)
{
    __fp16 want = mpfrRes->res.h;
    __fp16 d;
    double e;

    if (Asuint16(got) == Asuint16(want)) {
        return 0.0;
    }
    if (signbit(got) != signbit(want)) {
        /* -nan and nan all treated as NaN.  */
        return isnan(got) && isnan(want) ? 0 : INFINITY;
    }
    if (!isfinite(want) || !isfinite(got)) {
        if (isnan(got) != isnan(want)) {
            return INFINITY;
        }
        if (isnan(want)) {
            return 0;
        }
        if (isinf(got)) {
            got = copysign(0x1p15, got);
            want *= 0.5f;
        }
        if (isinf(want)) {
            want = copysign(0x1p15, want);
            got *= 0.5f;
        }
    }

    /* TODO: incorrect when got vs want cross a powof2 boundary
     * error = got > want
     *   ? got - want - tail ulp - 0.5 ulp
     *   : got - want - tail ulp + 0.5 ulp;
     */
    d = got - want;
    e = d > 0 ? -mpfrRes->resTail - 0.5 : -mpfrRes->resTail + 0.5;

    return scalbn(d, -mpfrRes->ulpexp) + e;
}

/* ulperr_xxx function calculate ulp error */
double ulperr_f(float got, struct MpfrRes *mpfrRes)
{
    float want = mpfrRes->res.f;
    float d;
    double e;

    if (Asuint(got) == Asuint(want)) {
        return 0.0;
    }
    if (signbit(got) != signbit(want)) {
        /* -nan and nan all treated as NaN.  */
        return isnan(got) && isnan(want) ? 0 : INFINITY;
    }
    if (!isfinite(want) || !isfinite(got)) {
        if (isnan(got) != isnan(want)) {
            return INFINITY;
        }
        if (isnan(want)) {
            return 0;
        }
        if (isinf(got)) {
            got = copysignf(0x1p127f, got);
            want *= 0.5f;
        }
        if (isinf(want)) {
            want = copysignf(0x1p127f, want);
            got *= 0.5f;
        }
    }

    /* TODO: incorrect when got vs want cross a powof2 boundary
     * error = got > want
     *   ? got - want - tail ulp - 0.5 ulp
     *   : got - want - tail ulp + 0.5 ulp;
     */
    d = got - want;
    e = d > 0 ? -mpfrRes->resTail - 0.5 : -mpfrRes->resTail + 0.5;

    return scalbnf(d, -mpfrRes->ulpexp) + e;
}

double dnn_ulperr_d(double res, double mpfrRes, double base, double error)
{
    double maxerror;
    if (fabs(mpfrRes) > base) {
        maxerror = fabs((res - mpfrRes) / mpfrRes);
    } else {
        maxerror = fabs(res - mpfrRes);
    }
    maxerror = fmax(error, fabs(maxerror));
    return maxerror;
}

/* getnum_xxx function parse input interval */
uint64_t getnum_h(const char *str)
{
    uint64_t sign = 0;

    if (str[0] == '+') {
        str++;
    } else if (str[0] == '-') {
        sign = 1ULL << 15;
        str++;
    }
    /* 0xXXXX is treated as bit representation, '-' flips the sign bit.  */
    if (str[0] == '0' && tolower(str[1]) == 'x' && strchr(str, 'p') == 0) {
        return sign ^ strtoull(str, 0, 0);
    }

    return sign | Asuint16(strtof(str, 0)) ;
}

uint64_t getnum_b(const char *str)
{
    uint64_t sign = 0;

    if (str[0] == '+') {
        str++;
    } else if (str[0] == '-') {
        sign = 1ULL << 15;
        str++;
    }
    /* 0xXXXX is treated as bit representation, '-' flips the sign bit.  */
    if (str[0] == '0' && tolower(str[1]) == 'x' && strchr(str, 'p') == 0) {
        return (sign ^ strtoull(str, 0, 0)) & 0xffff;
    }

    float value = strtof(str, 0);
    return sign | (Asuint(value) >> 16);
}

/* getnum_xxx function parse input interval */
uint64_t getnum_f(const char *str)
{
    uint64_t sign = 0;

    if (str[0] == '+') {
        str++;
    } else if (str[0] == '-') {
        sign = 1ULL << 31;
        str++;
    }
    /* 0xXXXX is treated as bit representation, '-' flips the sign bit.  */
    if (str[0] == '0' && tolower(str[1]) == 'x' && strchr(str, 'p') == 0) {
        return sign ^ strtoull(str, 0, 0);
    }

    return sign | Asuint(strtof(str, 0)) ;
}

void MemoryInit(void *dst, int value, int length)
{
    uint8_t *pDst = (uint8_t *)dst;
    for (int i = 0; i < length; i++) {
        pDst[i] = value;
    }
}
