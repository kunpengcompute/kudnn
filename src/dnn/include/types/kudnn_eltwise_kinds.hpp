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

#ifndef KUDNN_ELTWISE_KINDS_HPP
#define KUDNN_ELTWISE_KINDS_HPP

namespace KuDNN {

// Enumeration of common data types which are supported in KuDNN.
enum class ActivationFunction {
    UNIMPLEMENTED = 0,
    ABS,
    EXP,
    LINEAR,
    LOG,
    RELU,
    ROUND,
    SQRT,
    SQUARE,
    TANH,
    SIGMOID,
    POW,
    CLIP,
    CLIP_V2,
    HARDSIGMOID,
    HARDSWISH,
    EXP_DST,
    LOGISTIC_DST,
    TANH_DST,
    ELU,
    GELU_TANH,
    CLIP_V2_DST,
    ELU_DST,
    RELU_DST,
    SQRT_DST,
    SWISH,
    SRELU,
    GELU_ERF,
    MISH
};

} // KuDNN

#endif // KUDNN_ELTWISE_KINDS_HPP
