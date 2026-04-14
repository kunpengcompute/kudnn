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

#include <memory>

#include "kudnn.hpp"

#include "internal_funcs.hpp"
#include "kernels/kernel_funcs.h"
#include "threading_impl.hpp"

#include <iostream>
#include <unordered_map>

namespace KuDNN {
#ifdef KUDNNL_920Pro
static constexpr SizeType SIGMOID_FWD_ELE_NUM_NUM_1 = 20000;
static constexpr SizeType SIGMOID_FWD_ELE_NUM_STEP1 = 10000;
static constexpr SizeType SIGMOID_FWD_ELE_NUM_NUM_2 = 30000;
static constexpr SizeType SIGMOID_FWD_ELE_NUM_STEP2 = 8000;
#else
static constexpr SizeType SIGMOID_FWD_ELE_NUM_STEP1 = 2500;
static constexpr SizeType SIGMOID_FWD_ELE_NUM_1 = 1200000;
static constexpr SizeType SIGMOID_FWD_ELE_NUM_2 = 10000000;
static constexpr SizeType SIGMOID_FWD_ELE_NUM_STEP2 = 83333;
#endif

static constexpr SizeType RELU_FWD_ELE_NUM_STEP1 = 25000;
static constexpr SizeType RELU_FWD_ELE_NUM_1 = 8000000;
static constexpr SizeType RELU_FWD_ELE_NUM_2 = 16000000;
static constexpr SizeType RELU_FWD_ELE_NUM_3 = 20000000;

static constexpr SizeType LINEAR_FWD_ELE_NUM_STEP1 = 25000;
static constexpr SizeType LINEAR_FWD_ELE_NUM_1 = 8000000;
static constexpr SizeType LINEAR_FWD_ELE_NUM_2 = 16000000;
static constexpr SizeType LINEAR_FWD_ELE_NUM_STEP2 = 133333;

static constexpr SizeType ABS_FWD_ELE_NUM_STEP1 = 25000;
static constexpr SizeType ABS_FWD_ELE_NUM_1 = 8000000;
static constexpr SizeType ABS_FWD_ELE_NUM_2 = 16000000;
static constexpr SizeType ABS_FWD_ELE_NUM_3 = 20000000;

static constexpr SizeType CLIP_FWD_ELE_NUM_STEP1 = 25000;
static constexpr SizeType CLIP_FWD_ELE_NUM_1 = 8000000;
static constexpr SizeType CLIP_FWD_ELE_NUM_2 = 16000000;
static constexpr SizeType CLIP_FWD_ELE_NUM_STEP2 = 133333;

static constexpr SizeType CLIP_V2_FWD_ELE_NUM_STEP1 = 25000;
static constexpr SizeType CLIP_V2_FWD_ELE_NUM_1 = 8000000;
static constexpr SizeType CLIP_V2_FWD_ELE_NUM_2 = 16000000;
static constexpr SizeType CLIP_V2_FWD_ELE_NUM_STEP2 = 133333;

#ifdef KUDNNL_920Pro
static constexpr SizeType ELU_FWD_ELE_NUM = 30000;
static constexpr SizeType ELU_FWD_ELE_NUM_STEP = 12000;
#else
static constexpr SizeType ELU_FWD_ELE_NUM_STEP1 = 2500;
static constexpr SizeType ELU_FWD_ELE_NUM_1 = 60000;
static constexpr SizeType ELU_FWD_ELE_NUM_2 = 1000000;
static constexpr SizeType ELU_FWD_ELE_NUM_3 = 8000000;
static constexpr SizeType ELU_FWD_ELE_NUM_STEP2 = 66666;
#endif

static constexpr SizeType EXP_FWD_ELE_NUM_STEP1 = 5000;
static constexpr SizeType EXP_FWD_ELE_NUM_1 = 1600000;
static constexpr SizeType EXP_FWD_ELE_NUM_STEP2 = 20000;

static constexpr SizeType GELU_ERF_FWD_ELE_NUM_STEP1 = 2500;
static constexpr SizeType GELU_ERF_FWD_ELE_NUM_1 = 50000;
static constexpr SizeType GELU_ERF_FWD_ELE_NUM_2 = 800000;
static constexpr SizeType GELU_ERF_FWD_ELE_NUM_3 = 5000000;
static constexpr SizeType GELU_ERF_FWD_ELE_NUM_STEP2 = 41666;

static constexpr SizeType GELU_TANH_FWD_ELE_NUM_STEP1 = 1000;
static constexpr SizeType GELU_TANH_FWD_ELE_NUM_1 = 300000;
static constexpr SizeType GELU_TANH_FWD_ELE_NUM_2 = 3000000;
static constexpr SizeType GELU_TANH_FWD_ELE_NUM_STEP2 = 25000;

static constexpr SizeType HARDSIGMOID_FWD_ELE_NUM_STEP1 = 25000;
static constexpr SizeType HARDSIGMOID_FWD_ELE_NUM_1 = 8000000;
static constexpr SizeType HARDSIGMOID_FWD_ELE_NUM_2 = 16000000;
static constexpr SizeType HARDSIGMOID_FWD_ELE_NUM_STEP2 = 133333;

static constexpr SizeType HARDSWISH_FWD_ELE_NUM_STEP1 = 25000;
static constexpr SizeType HARDSWISH_FWD_ELE_NUM_1 = 8000000;
static constexpr SizeType HARDSWISH_FWD_ELE_NUM_2 = 20000000;
static constexpr SizeType HARDSWISH_FWD_ELE_NUM_STEP2 = 166666;

static constexpr SizeType LOG_FWD_ELE_NUM_STEP1 = 2500;
static constexpr SizeType LOG_FWD_ELE_NUM_1 = 60000;
static constexpr SizeType LOG_FWD_ELE_NUM_2 = 800000;
static constexpr SizeType LOG_FWD_ELE_NUM_3 = 8000000;
static constexpr SizeType LOG_FWD_ELE_NUM_STEP2 = 66666;

static constexpr SizeType MISH_FWD_ELE_NUM_STEP1 = 2500;
static constexpr SizeType MISH_FWD_ELE_NUM_1 = 50000;
static constexpr SizeType MISH_FWD_ELE_NUM_2 = 1000000;
static constexpr SizeType MISH_FWD_ELE_NUM_3 = 5000000;
static constexpr SizeType MISH_FWD_ELE_NUM_STEP2 = 41666;

static constexpr SizeType POW_FWD_ELE_NUM_STEP1 = 25000;
static constexpr SizeType POW_FWD_ELE_NUM_1 = 8000000;
static constexpr SizeType POW_FWD_ELE_NUM_2 = 16000000;
static constexpr SizeType POW_FWD_ELE_NUM_3 = 20000000;

static constexpr SizeType ROUND_FWD_ELE_NUM_STEP1 = 25000;
static constexpr SizeType ROUND_FWD_ELE_NUM_1 = 8000000;
static constexpr SizeType ROUND_FWD_ELE_NUM_2 = 20000000;
static constexpr SizeType ROUND_FWD_ELE_NUM_STEP2 = 166666;

static constexpr SizeType SRELU_FWD_ELE_NUM_STEP1 = 1000;
static constexpr SizeType SRELU_FWD_ELE_NUM_1 = 500000;
static constexpr SizeType SRELU_FWD_ELE_NUM_2 = 3000000;
static constexpr SizeType SRELU_FWD_ELE_NUM_STEP2 = 25000;

static constexpr SizeType SQRT_FWD_ELE_NUM_STEP1 = 5000;
static constexpr SizeType SQRT_FWD_ELE_NUM_1 = 120000;
static constexpr SizeType SQRT_FWD_ELE_NUM_2 = 1600000;
static constexpr SizeType SQRT_FWD_ELE_NUM_STEP2 = 20000;

static constexpr SizeType SQUARE_FWD_ELE_NUM_STEP1 = 25000;
static constexpr SizeType SQUARE_FWD_ELE_NUM_1 = 8000000;
static constexpr SizeType SQUARE_FWD_ELE_NUM_2 = 16000000;
static constexpr SizeType SQUARE_FWD_ELE_NUM_STEP2 = 133333;

static constexpr SizeType SWISH_FWD_ELE_NUM_STEP1 = 2500;
static constexpr SizeType SWISH_FWD_ELE_NUM_1 = 60000;
static constexpr SizeType SWISH_FWD_ELE_NUM_2 = 1000000;
static constexpr SizeType SWISH_FWD_ELE_NUM_3 = 10000000;
static constexpr SizeType SWISH_FWD_ELE_NUM_STEP2 = 83333;

static constexpr SizeType TANH_FWD_ELE_NUM_STEP1 = 2500;
static constexpr SizeType TANH_FWD_ELE_NUM_1 = 1000000;
static constexpr SizeType TANH_FWD_ELE_NUM_2 = 8000000;
static constexpr SizeType TANH_FWD_ELE_NUM_STEP2 = 66666;

static constexpr SizeType SIGMOID_BWD_ELE_NUM_STEP1 = 2500;
static constexpr SizeType SIGMOID_BWD_ELE_NUM_1 = 1600000;
static constexpr SizeType SIGMOID_BWD_ELE_NUM_2 = 10000000;
static constexpr SizeType SIGMOID_BWD_ELE_NUM_STEP2 = 83333;

static constexpr SizeType RELU_BWD_ELE_NUM_STEP1 = 25000;
static constexpr SizeType RELU_BWD_ELE_NUM_1 = 5000000;
static constexpr SizeType RELU_BWD_ELE_NUM_2 = 10000000;
static constexpr SizeType RELU_BWD_ELE_NUM_3 = 16000000;

static constexpr SizeType LINEAR_BWD_ELE_NUM_STEP1 = 25000;
static constexpr SizeType LINEAR_BWD_ELE_NUM_1 = 8000000;
static constexpr SizeType LINEAR_BWD_ELE_NUM_2 = 16000000;
static constexpr SizeType LINEAR_BWD_ELE_NUM_STEP2 = 133333;

static constexpr SizeType ABS_BWD_ELE_NUM_STEP1 = 25000;
static constexpr SizeType ABS_BWD_ELE_NUM_1 = 5000000;
static constexpr SizeType ABS_BWD_ELE_NUM_2 = 12000000;
static constexpr SizeType ABS_BWD_ELE_NUM_3 = 16000000;

static constexpr SizeType CLIP_BWD_ELE_NUM_STEP1 = 25000;
static constexpr SizeType CLIP_BWD_ELE_NUM_1 = 5000000;
static constexpr SizeType CLIP_BWD_ELE_NUM_2 = 12000000;
static constexpr SizeType CLIP_BWD_ELE_NUM_3 = 16000000;

static constexpr SizeType CLIP_V2_BWD_ELE_NUM_STEP1 = 25000;
static constexpr SizeType CLIP_V2_BWD_ELE_NUM_1 = 5000000;
static constexpr SizeType CLIP_V2_BWD_ELE_NUM_2 = 12000000;
static constexpr SizeType CLIP_V2_BWD_ELE_NUM_3 = 16000000;

#ifdef KUDNNL_920Pro
static constexpr SizeType ELU_BWD_ELE_NUM = 10000;
static constexpr SizeType ELU_BWD_ELE_NUM_STEP = 4000;
#else
static constexpr SizeType ELU_BWD_ELE_NUM_STEP1 = 2500;
static constexpr SizeType ELU_BWD_ELE_NUM_1 = 1200000;
static constexpr SizeType ELU_BWD_ELE_NUM_2 = 16000000;
static constexpr SizeType ELU_BWD_ELE_NUM_3 = 20000000;
#endif

static constexpr SizeType EXP_BWD_ELE_NUM_STEP1 = 2500;
static constexpr SizeType EXP_BWD_ELE_NUM_1 = 1600000;
static constexpr SizeType EXP_BWD_ELE_NUM_2 = 16000000;
static constexpr SizeType EXP_BWD_ELE_NUM_STEP2 = 133333;

static constexpr SizeType GELU_ERF_BWD_ELE_NUM_STEP1 = 1000;
static constexpr SizeType GELU_ERF_BWD_ELE_NUM_1 = 800000;
static constexpr SizeType GELU_ERF_BWD_ELE_NUM_2 = 3000000;
static constexpr SizeType GELU_ERF_BWD_ELE_NUM_3 = 4000000;

static constexpr SizeType GELU_TANH_BWD_ELE_NUM_STEP1 = 1000;
static constexpr SizeType GELU_TANH_BWD_ELE_NUM_1 = 300000;
static constexpr SizeType GELU_TANH_BWD_ELE_NUM_2 = 1200000;
static constexpr SizeType GELU_TANH_BWD_ELE_NUM_3 = 3000000;

#ifdef KUDNNL_920Pro
static constexpr SizeType HARDSIGMOID_BWD_ELE_NUM_1 = 30000;
static constexpr SizeType HARDSIGMOID_BWD_ELE_NUM_STEP1 = 16000;
static constexpr SizeType HARDSIGMOID_BWD_ELE_NUM_2 = 80000;
static constexpr SizeType HARDSIGMOID_BWD_ELE_NUM_STEP2 = 14400;
#else
static constexpr SizeType HARDSIGMOID_BWD_ELE_NUM_STEP1 = 10000;
static constexpr SizeType HARDSIGMOID_BWD_ELE_NUM_1 = 500000;
static constexpr SizeType HARDSIGMOID_BWD_ELE_NUM_2 = 4000000;
static constexpr SizeType HARDSIGMOID_BWD_ELE_NUM_3 = 12000000;
static constexpr SizeType HARDSIGMOID_BWD_ELE_NUM_4 = 16000000;
#endif

#ifdef KUDNNL_920Pro
static constexpr SizeType HARDSWISH_BWD_ELE_NUM = 30000;
static constexpr SizeType HARDSWISH_BWD_ELE_NUM_STEP = 10000;
#else
static constexpr SizeType HARDSWISH_BWD_ELE_NUM_STEP1 = 5000;
static constexpr SizeType HARDSWISH_BWD_ELE_NUM_1 = 4000000;
static constexpr SizeType HARDSWISH_BWD_ELE_NUM_2 = 12000000;
static constexpr SizeType HARDSWISH_BWD_ELE_NUM_3 = 16000000;
#endif

static constexpr SizeType LOG_BWD_ELE_NUM_STEP1 = 5000;
static constexpr SizeType LOG_BWD_ELE_NUM_1 = 3000000;
static constexpr SizeType LOG_BWD_ELE_NUM_2 = 12000000;
static constexpr SizeType LOG_BWD_ELE_NUM_3 = 20000000;

static constexpr SizeType MISH_BWD_ELE_NUM_STEP1 = 1000;
static constexpr SizeType MISH_BWD_ELE_NUM_1 = 500000;
static constexpr SizeType MISH_BWD_ELE_NUM_2 = 3000000;
static constexpr SizeType MISH_BWD_ELE_NUM_3 = 5000000;

static constexpr SizeType POW_BWD_ELE_NUM_STEP1 = 25000;
static constexpr SizeType POW_BWD_ELE_NUM_1 = 5000000;
static constexpr SizeType POW_BWD_ELE_NUM_2 = 12000000;
static constexpr SizeType POW_BWD_ELE_NUM_3 = 16000000;

#ifdef KUDNNL_920Pro
static constexpr SizeType SRELU_BWD_ELE_NUM = 5000;
static constexpr SizeType SRELU_BWD_ELE_NUM_STEP = 1600;
#else
static constexpr SizeType SRELU_BWD_ELE_NUM_STEP1 = 2500;
static constexpr SizeType SRELU_BWD_ELE_NUM_1 = 1200000;
static constexpr SizeType SRELU_BWD_ELE_NUM_2 = 16000000;
static constexpr SizeType SRELU_BWD_ELE_NUM_STEP2 = 133333;
#endif

static constexpr SizeType SQRT_BWD_ELE_NUM_STEP1 = 2500;
static constexpr SizeType SQRT_BWD_ELE_NUM_1 = 1600000;
static constexpr SizeType SQRT_BWD_ELE_NUM_2 = 16000000;
static constexpr SizeType SQRT_BWD_ELE_NUM_STEP2 = 133333;

static constexpr SizeType SQUARE_BWD_ELE_NUM_STEP1 = 25000;
static constexpr SizeType SQUARE_BWD_ELE_NUM_1 = 5000000;
static constexpr SizeType SQUARE_BWD_ELE_NUM_2 = 12000000;
static constexpr SizeType SQUARE_BWD_ELE_NUM_3 = 16000000;

#ifdef KUDNNL_920Pro
static constexpr SizeType SWISH_BWD_ELE_NUM = 8000;
static constexpr SizeType SWISH_BWD_ELE_NUM_STEP = 2000;
#else
static constexpr SizeType SWISH_BWD_ELE_NUM_STEP1 = 2500;
static constexpr SizeType SWISH_BWD_ELE_NUM_1 = 1000000;
static constexpr SizeType SWISH_BWD_ELE_NUM_2 = 3000000;
static constexpr SizeType SWISH_BWD_ELE_NUM_3 = 10000000;
#endif

#ifdef KUDNNL_920Pro
static constexpr SizeType TANH_BWD_ELE_NUM = 5000;
static constexpr SizeType TANH_BWD_ELE_NUM_STEP = 2000;
#else
static constexpr SizeType TANH_BWD_ELE_NUM_STEP1 = 1000;
static constexpr SizeType TANH_BWD_ELE_NUM_1 = 800000;
static constexpr SizeType TANH_BWD_ELE_NUM_2 = 3000000;
static constexpr SizeType TANH_BWD_ELE_NUM_3 = 8000000;
#endif

static constexpr SizeType CLIP_V2_DST_BWD_ELE_NUM_STEP1 = 25000;
static constexpr SizeType CLIP_V2_DST_BWD_ELE_NUM_1 = 5000000;
static constexpr SizeType CLIP_V2_DST_BWD_ELE_NUM_2 = 12000000;
static constexpr SizeType CLIP_V2_DST_BWD_ELE_NUM_3 = 16000000;

static constexpr SizeType ELU_DST_BWD_ELE_NUM_STEP1 = 25000;
static constexpr SizeType ELU_DST_BWD_ELE_NUM_1 = 5000000;
static constexpr SizeType ELU_DST_BWD_ELE_NUM_2 = 12000000;
static constexpr SizeType ELU_DST_BWD_ELE_NUM_3 = 16000000;

static constexpr SizeType EXP_DST_BWD_ELE_NUM_STEP1 = 25000;
static constexpr SizeType EXP_DST_BWD_ELE_NUM_1 = 800000;
static constexpr SizeType EXP_DST_BWD_ELE_NUM_2 = 5000000;
static constexpr SizeType EXP_DST_BWD_ELE_NUM_3 = 12000000;
static constexpr SizeType EXP_DST_BWD_ELE_NUM_4 = 16000000;

static constexpr SizeType LOGISTIC_DST_BWD_ELE_NUM_STEP1 = 25000;
static constexpr SizeType LOGISTIC_DST_BWD_ELE_NUM_1 = 5000000;
static constexpr SizeType LOGISTIC_DST_BWD_ELE_NUM_2 = 10000000;
static constexpr SizeType LOGISTIC_DST_BWD_ELE_NUM_STEP2 = 83333;

static constexpr SizeType RELU_DST_BWD_ELE_NUM_STEP1 = 25000;
static constexpr SizeType RELU_DST_BWD_ELE_NUM_1 = 5000000;
static constexpr SizeType RELU_DST_BWD_ELE_NUM_2 = 10000000;
static constexpr SizeType RELU_DST_BWD_ELE_NUM_3 = 16000000;

static constexpr SizeType SQRT_DST_BWD_ELE_NUM_STEP1 = 8000;
static constexpr SizeType SQRT_DST_BWD_ELE_NUM_1 = 300000;
static constexpr SizeType SQRT_DST_BWD_ELE_NUM_2 = 3000000;
static constexpr SizeType SQRT_DST_BWD_ELE_NUM_3 = 12000000;
static constexpr SizeType SQRT_DST_BWD_ELE_NUM_STEP2 = 100000;

static constexpr SizeType TANH_DST_BWD_ELE_NUM_STEP1 = 25000;
static constexpr SizeType TANH_DST_BWD_ELE_NUM_1 = 160000;
static constexpr SizeType TANH_DST_BWD_ELE_NUM_2 = 5000000;
static constexpr SizeType TANH_DST_BWD_ELE_NUM_3 = 12000000;
static constexpr SizeType TANH_DST_BWD_ELE_NUM_4 = 16000000;

#ifdef KUDNNL_920Pro
static constexpr SizeType ELE_THREADS_1NUMA = 38;
static constexpr SizeType ELE_THREADS_2NUMA = 76;
static constexpr SizeType ELE_THREADS_3NUMA = 114;
static constexpr SizeType ELE_THREADS_4NUMA = 152;
#else
static constexpr SizeType ELE_THREADS_1NUMA = 40;
static constexpr SizeType ELE_THREADS_2NUMA = 80;
static constexpr SizeType ELE_THREADS_3NUMA = 120;
static constexpr SizeType ELE_THREADS_4NUMA = 160;
#endif

static constexpr SizeType OTHER_ELE_NUM_DEFAULT = 15000;

namespace Detail {

    const std::unordered_map<ActivationFunction, std::tuple<void *, void *, void *>> eltFuncFwdMap = {
        {ActivationFunction::ABS, std::make_tuple(reinterpret_cast<void *>(AbsLaSveFWDF32),
            reinterpret_cast<void *>(AbsLaSveFWDF16), reinterpret_cast<void *>(AbsLaSveFWDBF16))},
        {ActivationFunction::EXP, std::make_tuple(reinterpret_cast<void *>(ExpLaSveFWDF32),
            reinterpret_cast<void *>(ExpLaSveFWDF16), reinterpret_cast<void *>(ExpLaSveFWDBF16))},
        {ActivationFunction::GELU_ERF, std::make_tuple(reinterpret_cast<void *>(GeluErfLaSveFWDF32),
            reinterpret_cast<void *>(GeluErfLaSveFWDF16), reinterpret_cast<void *>(GeluErfLaSveFWDBF16))},
        {ActivationFunction::GELU_TANH, std::make_tuple(reinterpret_cast<void *>(GeluTanhLaSveFWDF32),
            reinterpret_cast<void *>(GeluTanhLaSveFWDF16), reinterpret_cast<void *>(GeluTanhLaSveFWDBF16))},
        {ActivationFunction::LOG, std::make_tuple(reinterpret_cast<void *>(LnLaSveFWDF32),
            reinterpret_cast<void *>(LnLaSveFWDF16), reinterpret_cast<void *>(LnLaSveFWDBF16))},
        {ActivationFunction::SIGMOID, std::make_tuple(reinterpret_cast<void *>(SigmoidSveFWDF32),
            reinterpret_cast<void *>(SigmoidSveFWDF16), reinterpret_cast<void *>(SigmoidSveFWDBF16))},
        {ActivationFunction::MISH, std::make_tuple(reinterpret_cast<void *>(MishLaSveFWDF32),
            reinterpret_cast<void *>(MishLaSveFWDF16), reinterpret_cast<void *>(MishLaSveFWDBF16))},
        {ActivationFunction::ROUND, std::make_tuple(reinterpret_cast<void *>(NearbyintSveFWDF32),
            reinterpret_cast<void *>(NearbyintSveFWDF16), reinterpret_cast<void *>(NearbyintSveFWDBF16))},
        {ActivationFunction::SQRT, std::make_tuple(reinterpret_cast<void *>(SqrtLaSveFWDF32),
            reinterpret_cast<void *>(SqrtLaSveFWDF16), reinterpret_cast<void *>(SqrtLaSveFWDBF16))},
        {ActivationFunction::SQUARE, std::make_tuple(reinterpret_cast<void *>(SqrLaSveFWDF32),
            reinterpret_cast<void *>(SqrLaSveFWDF16), reinterpret_cast<void *>(SqrLaSveFWDBF16))},
        {ActivationFunction::TANH, std::make_tuple(reinterpret_cast<void *>(TanhLaSveFWDF32),
            reinterpret_cast<void *>(TanhLaSveFWDF16), reinterpret_cast<void *>(TanhLaSveFWDBF16))},
        {ActivationFunction::ELU, std::make_tuple(reinterpret_cast<void *>(EluLaSveFWDF32),
            reinterpret_cast<void *>(EluLaSveFWDF16), reinterpret_cast<void *>(EluLaSveFWDBF16))},
        {ActivationFunction::ELU_DST, std::make_tuple(reinterpret_cast<void *>(EluDstSveFWDF32),
            reinterpret_cast<void *>(EluDstSveFWDF16), reinterpret_cast<void *>(EluDstSveFWDBF16))},
        {ActivationFunction::RELU, std::make_tuple(reinterpret_cast<void *>(ReluSveFWDF32),
            reinterpret_cast<void *>(ReluSveFWDF16), reinterpret_cast<void *>(ReluSveFWDBF16))},
        {ActivationFunction::SRELU, std::make_tuple(reinterpret_cast<void *>(SoftReluLaSveFWDF32),
            reinterpret_cast<void *>(SoftReluLaSveFWDF16), reinterpret_cast<void *>(SoftReluLaSveFWDBF16))},
        {ActivationFunction::SWISH, std::make_tuple(reinterpret_cast<void *>(SwishLaSveFWDF32),
            reinterpret_cast<void *>(SwishLaSveFWDF16), reinterpret_cast<void *>(SwishLaSveFWDBF16))},
        {ActivationFunction::CLIP, std::make_tuple(reinterpret_cast<void *>(ClipSveFWDF32),
            reinterpret_cast<void *>(ClipSveFWDF16), reinterpret_cast<void *>(ClipSveFWDBF16))},
        {ActivationFunction::CLIP_V2, std::make_tuple(reinterpret_cast<void *>(ClipV2SveFWDF32),
            reinterpret_cast<void *>(ClipV2SveFWDF16), reinterpret_cast<void *>(ClipV2SveFWDBF16))},
        {ActivationFunction::HARDSIGMOID, std::make_tuple(reinterpret_cast<void *>(HardSigmoidLaSveFWDF32),
            reinterpret_cast<void *>(HardSigmoidLaSveFWDF16), reinterpret_cast<void *>(HardSigmoidLaSveFWDBF16))},
        {ActivationFunction::HARDSWISH, std::make_tuple(reinterpret_cast<void *>(HardSwishSveFWDF32),
            reinterpret_cast<void *>(HardSwishSveFWDF16), reinterpret_cast<void *>(HardSwishSveFWDBF16))},
        {ActivationFunction::LINEAR, std::make_tuple(reinterpret_cast<void *>(LinearSveF32),
            reinterpret_cast<void *>(LinearSveF16), reinterpret_cast<void *>(LinearSveBF16))},
        {ActivationFunction::POW, std::make_tuple(reinterpret_cast<void *>(PowxaLaSveFWDF32),
            reinterpret_cast<void *>(PowxaLaSveFWDF16), reinterpret_cast<void *>(PowxaLaSveFWDBF16))},
    };

    const std::unordered_map<ActivationFunction, std::tuple<void *, void *, void *>> eltFuncBwdMap = {
        {ActivationFunction::ABS, std::make_tuple(reinterpret_cast<void *>(AbsLaSveBWDF32),
            reinterpret_cast<void *>(AbsLaSveBWDF16), reinterpret_cast<void *>(AbsLaSveBWDBF16))},
        {ActivationFunction::EXP, std::make_tuple(reinterpret_cast<void *>(ExpLaSveBWDF32),
            reinterpret_cast<void *>(ExpLaSveBWDF16), reinterpret_cast<void *>(ExpLaSveBWDBF16))},
        {ActivationFunction::EXP_DST, std::make_tuple(reinterpret_cast<void *>(ExpLaDstSveBWDF32),
            reinterpret_cast<void *>(ExpLaDstSveBWDF16), reinterpret_cast<void *>(ExpLaDstSveBWDBF16))},
        {ActivationFunction::GELU_ERF, std::make_tuple(reinterpret_cast<void *>(GeluErfLaSveBWDF32),
            reinterpret_cast<void *>(GeluErfLaSveBWDF16), reinterpret_cast<void *>(GeluErfLaSveBWDBF16))},
        {ActivationFunction::GELU_TANH, std::make_tuple(reinterpret_cast<void *>(GeluTanhLaSveBWDF32),
            reinterpret_cast<void *>(GeluTanhLaSveBWDF16), reinterpret_cast<void *>(GeluTanhLaSveBWDBF16))},
        {ActivationFunction::LOG, std::make_tuple(reinterpret_cast<void *>(LnLaSveBWDF32),
            reinterpret_cast<void *>(LnLaSveBWDF16), reinterpret_cast<void *>(LnLaSveBWDBF16))},
        {ActivationFunction::LOGISTIC_DST, std::make_tuple(reinterpret_cast<void *>(LogisticDstSveBWDF32),
            reinterpret_cast<void *>(LogisticDstSveBWDF16), reinterpret_cast<void *>(LogisticDstSveBWDBF16))},
        {ActivationFunction::SIGMOID, std::make_tuple(reinterpret_cast<void *>(SigmoidSveBWDF32),
            reinterpret_cast<void *>(SigmoidSveBWDF16), reinterpret_cast<void *>(SigmoidSveBWDBF16))},
        {ActivationFunction::MISH, std::make_tuple(reinterpret_cast<void *>(MishLaSveBWDF32),
            reinterpret_cast<void *>(MishLaSveBWDF16), reinterpret_cast<void *>(MishLaSveBWDBF16))},
        {ActivationFunction::SQRT, std::make_tuple(reinterpret_cast<void *>(SqrtLaSveBWDF32),
            reinterpret_cast<void *>(SqrtLaSveBWDF16), reinterpret_cast<void *>(SqrtLaSveBWDBF16))},
        {ActivationFunction::SQRT_DST, std::make_tuple(reinterpret_cast<void *>(SqrtDstSveBWDF32),
            reinterpret_cast<void *>(SqrtDstSveBWDF16), reinterpret_cast<void *>(SqrtDstSveBWDBF16))},
        {ActivationFunction::SQUARE, std::make_tuple(reinterpret_cast<void *>(SqrLaSveBWDF32),
            reinterpret_cast<void *>(SqrLaSveBWDF16), reinterpret_cast<void *>(SqrLaSveBWDBF16))},
        {ActivationFunction::TANH, std::make_tuple(reinterpret_cast<void *>(TanhLaSveBWDF32),
            reinterpret_cast<void *>(TanhLaSveBWDF16), reinterpret_cast<void *>(TanhLaSveBWDBF16))},
        {ActivationFunction::TANH_DST, std::make_tuple(reinterpret_cast<void *>(TanhDstSveBWDF32),
            reinterpret_cast<void *>(TanhDstSveBWDF16), reinterpret_cast<void *>(TanhDstSveBWDBF16))},
        {ActivationFunction::ELU, std::make_tuple(reinterpret_cast<void *>(EluLaSveBWDF32),
            reinterpret_cast<void *>(EluLaSveBWDF16), reinterpret_cast<void *>(EluLaSveBWDBF16))},
        {ActivationFunction::ELU_DST, std::make_tuple(reinterpret_cast<void *>(EluDstSveBWDF32),
            reinterpret_cast<void *>(EluDstSveBWDF16), reinterpret_cast<void *>(EluDstSveBWDBF16))},
        {ActivationFunction::RELU, std::make_tuple(reinterpret_cast<void *>(ReluSveBWDF32),
            reinterpret_cast<void *>(ReluSveBWDF16), reinterpret_cast<void *>(ReluSveBWDBF16))},
        {ActivationFunction::RELU_DST, std::make_tuple(reinterpret_cast<void *>(ReluDstSveBWDF32),
            reinterpret_cast<void *>(ReluDstSveBWDF16), reinterpret_cast<void *>(ReluDstSveBWDBF16))},
        {ActivationFunction::SRELU, std::make_tuple(reinterpret_cast<void *>(SoftReluLaSveBWDF32),
            reinterpret_cast<void *>(SoftReluLaSveBWDF16), reinterpret_cast<void *>(SoftReluLaSveBWDBF16))},
        {ActivationFunction::SWISH, std::make_tuple(reinterpret_cast<void *>(SwishLaSveBWDF32),
            reinterpret_cast<void *>(SwishLaSveBWDF16), reinterpret_cast<void *>(SwishLaSveBWDBF16))},
        {ActivationFunction::CLIP, std::make_tuple(reinterpret_cast<void *>(ClipSveBWDF32),
            reinterpret_cast<void *>(ClipSveBWDF16), reinterpret_cast<void *>(ClipSveBWDBF16))},
        {ActivationFunction::CLIP_V2, std::make_tuple(reinterpret_cast<void *>(ClipV2SveBWDF32),
            reinterpret_cast<void *>(ClipV2SveBWDF16), reinterpret_cast<void *>(ClipV2SveBWDBF16))},
        {ActivationFunction::CLIP_V2_DST, std::make_tuple(reinterpret_cast<void *>(ClipV2DstSveBWDF32),
            reinterpret_cast<void *>(ClipV2DstSveBWDF16), reinterpret_cast<void *>(ClipV2DstSveBWDBF16))},
        {ActivationFunction::HARDSIGMOID, std::make_tuple(reinterpret_cast<void *>(HardSigmoidLaSveBWDF32),
            reinterpret_cast<void *>(HardSigmoidLaSveBWDF16), reinterpret_cast<void *>(HardSigmoidLaSveBWDBF16))},
        {ActivationFunction::HARDSWISH, std::make_tuple(reinterpret_cast<void *>(HardSwishSveBWDF32),
            reinterpret_cast<void *>(HardSwishSveBWDF16), reinterpret_cast<void *>(HardSwishSveBWDBF16))},
        {ActivationFunction::POW, std::make_tuple(reinterpret_cast<void *>(PowxaLaSveBWDF32),
            reinterpret_cast<void *>(PowxaLaSveBWDF16), reinterpret_cast<void *>(PowxaLaSveBWDBF16))},
    };

class ActivationLayerImplFWD final : public Parallelizable {
public:
    ActivationLayerImplFWD(const TensorInfo &srcInfo, const TensorInfo &dstInfo,
        ActivationFunction kind, float alpha, float beta) noexcept(false)
        : activationKind(kind), srcType(srcInfo.GetType()), dstType(dstInfo.GetType()),
        alphaInfo(alpha), betaInfo(beta)
    {
        Service::ThrowOnStatus(ActivationLayerFWD::ValidateInput(
            srcInfo, dstInfo, kind, alphaInfo, betaInfo), "Activation layer");
        reorderedDims = Service::BroadcastTo5D(Service::GetShapeAccordingToLayout(srcInfo.GetDims(),
            srcInfo.GetLayout()));
        reorderedStridesSrc = Service::BroadcastTo5D(Service::GetShapeAccordingToLayout(srcInfo.GetStrides(),
            srcInfo.GetLayout()));
        reorderedStridesDst  = Service::BroadcastTo5D(Service::GetShapeAccordingToLayout(dstInfo.GetStrides(),
            dstInfo.GetLayout()));
        Parallelizable::Initialize(GetOptNumThreads(), reorderedDims);
    }

    void RunSequential(const Shape &batchSplitDims, const void *src, void *dst) const noexcept(false)
    {
        auto item = eltFuncFwdMap.find(activationKind);
        if (item == eltFuncFwdMap.end()) {
            return;
        }

        const void *calSrc = src;
        void *calDst = dst;
        auto calSrcType = srcType;

        if (Service::OneOf(srcType, Element::TypeT::S32, Element::TypeT::S8, Element::TypeT::U8) ||
            srcType != dstType) {
            if ((calSrc == nullptr) || (calDst == nullptr)) {
                return;
            }

            void *tmp = Service::AlignedAlloc(batchSplitDims.GetTotalDimsSize() * sizeof(float));
            ReorderLayer rSrc({
                batchSplitDims, calSrcType, Layout::ABCDE}, {
                batchSplitDims, Element::TypeT::F32, Layout::ABCDE});
            rSrc.Run(calSrc, tmp);
            calSrc = tmp;
            calDst = Service::AlignedAlloc(batchSplitDims.GetTotalDimsSize() * sizeof(float));
            calSrcType = Element::TypeT::F32;
        }

        if (Service::OneOf(activationKind, ActivationFunction::ABS, ActivationFunction::EXP,
            ActivationFunction::GELU_ERF, ActivationFunction::GELU_TANH, ActivationFunction::LOG,
            ActivationFunction::SIGMOID, ActivationFunction::MISH, ActivationFunction::ROUND,
            ActivationFunction::SQRT, ActivationFunction::SQUARE, ActivationFunction::TANH)) {
            if (calSrcType == Element::TypeT::F32) {
                reinterpret_cast<EltwiseFwdFuncF32X>(std::get<0x0>(item->second))(
                    static_cast<int>(batchSplitDims.GetTotalDimsSize()),
                    static_cast<const float *>(calSrc), static_cast<float *>(calDst));
            } else if (calSrcType == Element::TypeT::F16) {
                reinterpret_cast<EltwiseFwdFuncF16X>(std::get<0x1>(item->second))(
                    static_cast<int>(batchSplitDims.GetTotalDimsSize()),
                    static_cast<const __fp16 *>(calSrc), static_cast<__fp16 *>(calDst));
            } else if (calSrcType == Element::TypeT::BF16) {
                reinterpret_cast<EltwiseFwdFuncBF16X>(std::get<0x2>(item->second))(
                    static_cast<int>(batchSplitDims.GetTotalDimsSize()),
                    static_cast<const __bf16 *>(calSrc), static_cast<__bf16 *>(calDst));
            }
        } else if (Service::OneOf(activationKind, ActivationFunction::ELU, ActivationFunction::ELU_DST,
            ActivationFunction::RELU, ActivationFunction::SRELU, ActivationFunction::SWISH)) {
            if (calSrcType == Element::TypeT::F32) {
                reinterpret_cast<EltwiseFwdFuncF32AX>(std::get<0x0>(item->second))(
                    static_cast<int>(batchSplitDims.GetTotalDimsSize()), alphaInfo,
                    static_cast<const float *>(calSrc), static_cast<float *>(calDst));
            } else if (calSrcType == Element::TypeT::F16) {
                reinterpret_cast<EltwiseFwdFuncF16AX>(std::get<0x1>(item->second))(
                    static_cast<int>(batchSplitDims.GetTotalDimsSize()), alphaInfo,
                    static_cast<const __fp16 *>(calSrc), static_cast<__fp16 *>(calDst));
            } else if (calSrcType == Element::TypeT::BF16) {
                reinterpret_cast<EltwiseFwdFuncBF16AX>(std::get<0x2>(item->second))(
                    static_cast<int>(batchSplitDims.GetTotalDimsSize()), alphaInfo,
                    static_cast<const __bf16 *>(calSrc), static_cast<__bf16 *>(calDst));
            }
        } else if (Service::OneOf(activationKind, ActivationFunction::CLIP, ActivationFunction::CLIP_V2,
            ActivationFunction::HARDSIGMOID, ActivationFunction::HARDSWISH, ActivationFunction::LINEAR,
            ActivationFunction::POW)) {
            if (calSrcType == Element::TypeT::F32) {
                reinterpret_cast<EltwiseFwdFuncF32ABX>(std::get<0x0>(item->second))(
                    static_cast<int>(batchSplitDims.GetTotalDimsSize()), alphaInfo, betaInfo,
                    static_cast<const float *>(calSrc), static_cast<float *>(calDst));
            } else if (calSrcType == Element::TypeT::F16) {
                reinterpret_cast<EltwiseFwdFuncF16ABX>(std::get<0x1>(item->second))(
                    static_cast<int>(batchSplitDims.GetTotalDimsSize()), alphaInfo, betaInfo,
                    static_cast<const __fp16 *>(calSrc), static_cast<__fp16 *>(calDst));
            } else if (calSrcType == Element::TypeT::BF16) {
                reinterpret_cast<EltwiseFwdFuncBF16ABX>(std::get<0x2>(item->second))(
                    static_cast<int>(batchSplitDims.GetTotalDimsSize()), alphaInfo, betaInfo,
                    static_cast<const __bf16 *>(calSrc), static_cast<__bf16 *>(calDst));
            }
        }

        if (Service::OneOf(srcType, Element::TypeT::S32, Element::TypeT::S8, Element::TypeT::U8) ||
            srcType != dstType) {
            Service::Deallocate(const_cast<void *>(calSrc));

            ReorderLayer rDst({
                batchSplitDims, Element::TypeT::F32, Layout::ABCDE}, {
                batchSplitDims, dstType, Layout::ABCDE});
            rDst.Run(calDst, dst);
            Service::Deallocate(calDst);
            return;
        }
    }

    void RunSequential(const Shape &batchSplitDims, const Shape &batchStrides, const void *src,
        void *dst) const noexcept(false)
    {
        SizeType strideSrc = 0;
        SizeType strideDst = 0;
        for (SizeType i = 0; i < batchStrides.GetNumDims(); ++i) {
            strideSrc += batchStrides[i] * reorderedStridesSrc[i] * srcType.GetSize();
            strideDst += batchStrides[i] * reorderedStridesDst[i] * dstType.GetSize();
        }
        RunSequential(batchSplitDims, static_cast<const std::byte *>(src) + strideSrc,
            static_cast<std::byte *>(dst) + strideDst);
    }

    void Run(const void *src, void *dst) noexcept(false)
    {
        if ((src == nullptr) || (dst == nullptr)) {
            throw Service::LogicError {"Activation Layer: one of input matrices is nullptr"};
        }
        Threading::BatchThreader(*this, src, dst);
    }

private:
    ActivationFunction activationKind;
    Element::Type srcType;
    Element::Type dstType;
    float alphaInfo;
    float betaInfo;
    Shape reorderedDims;
    Shape reorderedStridesSrc;
    Shape reorderedStridesDst;

    SizeType GetOptNumThreads() noexcept
    {
        SizeType totalSize = reorderedDims.GetTotalDimsSize();
        switch (activationKind) {
            case ActivationFunction::SIGMOID: {
#ifdef KUDNNL_920Pro
                if (totalSize < SIGMOID_FWD_ELE_NUM_NUM_1) {
                    return 1;
                }
                if (totalSize < SIGMOID_FWD_ELE_NUM_NUM_2) {
                    return (totalSize + SIGMOID_FWD_ELE_NUM_STEP1 - 1) / SIGMOID_FWD_ELE_NUM_STEP1;
                }
                return std::min((totalSize + SIGMOID_FWD_ELE_NUM_STEP2 - 1) / SIGMOID_FWD_ELE_NUM_STEP2,
                    ELE_THREADS_1NUMA);
#else
                if (totalSize < SIGMOID_FWD_ELE_NUM_1) {
                    return std::min((totalSize + SIGMOID_FWD_ELE_NUM_STEP1 - 1) / SIGMOID_FWD_ELE_NUM_STEP1,
                        ELE_THREADS_1NUMA);
                } else if (totalSize < SIGMOID_FWD_ELE_NUM_2) {
                    return ELE_THREADS_2NUMA;
                } else {
                    return std::min(((totalSize + SIGMOID_FWD_ELE_NUM_STEP2 - 1) / SIGMOID_FWD_ELE_NUM_STEP2),
                        static_cast<KuDNN::SizeType>(DEFAULT_MAX_NUM_THREADS));
                }
#endif
                break;
            }
            case ActivationFunction::RELU: {
                if (totalSize < RELU_FWD_ELE_NUM_1) {
                    return std::min(((totalSize + RELU_FWD_ELE_NUM_STEP1 - 1) / RELU_FWD_ELE_NUM_STEP1),
                        ELE_THREADS_1NUMA);
                } else if (totalSize < RELU_FWD_ELE_NUM_2) {
                    return ELE_THREADS_2NUMA;
                } else if (totalSize < RELU_FWD_ELE_NUM_3) {
                    return ELE_THREADS_3NUMA;
                } else {
                    return DEFAULT_MAX_NUM_THREADS;
                }
                break;
            }
            case ActivationFunction::LINEAR: {
                if (totalSize < LINEAR_FWD_ELE_NUM_1) {
                    return std::min(((totalSize + LINEAR_FWD_ELE_NUM_STEP1 - 1) / LINEAR_FWD_ELE_NUM_STEP1),
                        ELE_THREADS_1NUMA);
                } else if (totalSize < LINEAR_FWD_ELE_NUM_2) {
                    return ELE_THREADS_2NUMA;
                } else {
                    return std::min(((totalSize + LINEAR_FWD_ELE_NUM_STEP2 - 1) / LINEAR_FWD_ELE_NUM_STEP2),
                        static_cast<KuDNN::SizeType>(DEFAULT_MAX_NUM_THREADS));
                }
                break;
            }
            case ActivationFunction::ABS: {
                if (totalSize < ABS_FWD_ELE_NUM_1) {
                    return std::min(((totalSize + ABS_FWD_ELE_NUM_STEP1 - 1) / ABS_FWD_ELE_NUM_STEP1),
                        ELE_THREADS_1NUMA);
                } else if (totalSize < ABS_FWD_ELE_NUM_2) {
                    return ELE_THREADS_2NUMA;
                } else if (totalSize < ABS_FWD_ELE_NUM_3) {
                    return ELE_THREADS_3NUMA;
                } else {
                    return DEFAULT_MAX_NUM_THREADS;
                }
                break;
            }
            case ActivationFunction::CLIP: {
                if (totalSize < CLIP_FWD_ELE_NUM_1) {
                    return std::min(((totalSize + CLIP_FWD_ELE_NUM_STEP1 - 1) / CLIP_FWD_ELE_NUM_STEP1),
                        ELE_THREADS_1NUMA);
                } else if (totalSize < CLIP_FWD_ELE_NUM_2) {
                    return ELE_THREADS_2NUMA;
                } else {
                    return std::min(((totalSize + CLIP_FWD_ELE_NUM_STEP2 - 1) / CLIP_FWD_ELE_NUM_STEP2),
                        static_cast<KuDNN::SizeType>(DEFAULT_MAX_NUM_THREADS));
                }
                break;
            }
            case ActivationFunction::CLIP_V2: {
                if (totalSize < CLIP_V2_FWD_ELE_NUM_1) {
                    return std::min(((totalSize + CLIP_V2_FWD_ELE_NUM_STEP1 - 1) / CLIP_V2_FWD_ELE_NUM_STEP1),
                        ELE_THREADS_1NUMA);
                } else if (totalSize < CLIP_V2_FWD_ELE_NUM_2) {
                    return ELE_THREADS_2NUMA;
                } else {
                    return std::min(((totalSize + CLIP_V2_FWD_ELE_NUM_STEP2 - 1) / CLIP_V2_FWD_ELE_NUM_STEP2),
                        static_cast<KuDNN::SizeType>(DEFAULT_MAX_NUM_THREADS));
                }
                break;
            }
            case ActivationFunction::ELU: {
#ifdef KUDNNL_920Pro
                if (totalSize < ELU_FWD_ELE_NUM) {
                    return 1;
                }
                return std::min((totalSize + ELU_FWD_ELE_NUM_STEP - 1) / ELU_FWD_ELE_NUM_STEP,
                    ELE_THREADS_1NUMA);
#else
                if (totalSize < ELU_FWD_ELE_NUM_1) {
                    return (totalSize + ELU_FWD_ELE_NUM_STEP1 - 1) / ELU_FWD_ELE_NUM_STEP1;
                } else if (totalSize < ELU_FWD_ELE_NUM_2) {
                    return ELE_THREADS_1NUMA;
                } else if (totalSize < ELU_FWD_ELE_NUM_3) {
                    return ELE_THREADS_2NUMA;
                } else {
                    return std::min(((totalSize + ELU_FWD_ELE_NUM_STEP2 - 1) / ELU_FWD_ELE_NUM_STEP2),
                        static_cast<KuDNN::SizeType>(DEFAULT_MAX_NUM_THREADS));
                }
#endif
                break;
            }
            case ActivationFunction::EXP: {
                if (totalSize < EXP_FWD_ELE_NUM_1) {
                    return std::min(((totalSize + EXP_FWD_ELE_NUM_STEP1 - 1) / EXP_FWD_ELE_NUM_STEP1),
                        ELE_THREADS_1NUMA);
                } else {
                    return std::min(((totalSize + EXP_FWD_ELE_NUM_STEP2 - 1) / EXP_FWD_ELE_NUM_STEP2),
                        static_cast<KuDNN::SizeType>(DEFAULT_MAX_NUM_THREADS));
                }
                break;
            }
            case ActivationFunction::GELU_ERF: {
                if (totalSize < GELU_ERF_FWD_ELE_NUM_1) {
                    return (totalSize + GELU_ERF_FWD_ELE_NUM_STEP1 - 1) / GELU_ERF_FWD_ELE_NUM_STEP1;
                } else if (totalSize < GELU_ERF_FWD_ELE_NUM_2) {
                    return ELE_THREADS_1NUMA;
                } else if (totalSize < GELU_ERF_FWD_ELE_NUM_3) {
                    return ELE_THREADS_2NUMA;
                } else {
                    return std::min(((totalSize + GELU_ERF_FWD_ELE_NUM_STEP2 - 1) / GELU_ERF_FWD_ELE_NUM_STEP2),
                        static_cast<KuDNN::SizeType>(DEFAULT_MAX_NUM_THREADS));
                }
                break;
            }
            case ActivationFunction::GELU_TANH: {
                if (totalSize < GELU_TANH_FWD_ELE_NUM_1) {
                    return std::min(((totalSize + GELU_TANH_FWD_ELE_NUM_STEP1 - 1) / GELU_TANH_FWD_ELE_NUM_STEP1),
                        ELE_THREADS_1NUMA);
                } else if (totalSize < GELU_TANH_FWD_ELE_NUM_2) {
                    return ELE_THREADS_2NUMA;
                } else {
                    return std::min(((totalSize + GELU_TANH_FWD_ELE_NUM_STEP2 - 1) / GELU_TANH_FWD_ELE_NUM_STEP2),
                        static_cast<KuDNN::SizeType>(DEFAULT_MAX_NUM_THREADS));
                }
                break;
            }
            case ActivationFunction::HARDSIGMOID: {
                if (totalSize < HARDSIGMOID_FWD_ELE_NUM_1) {
                    return std::min(((totalSize + HARDSIGMOID_FWD_ELE_NUM_STEP1 - 1) / HARDSIGMOID_FWD_ELE_NUM_STEP1),
                        ELE_THREADS_1NUMA);
                } else if (totalSize < HARDSIGMOID_FWD_ELE_NUM_2) {
                    return ELE_THREADS_2NUMA;
                } else {
                    return std::min(((totalSize + HARDSIGMOID_FWD_ELE_NUM_STEP2 - 1) / HARDSIGMOID_FWD_ELE_NUM_STEP2),
                        static_cast<KuDNN::SizeType>(DEFAULT_MAX_NUM_THREADS));
                }
                break;
            }
            case ActivationFunction::HARDSWISH: {
                if (totalSize < HARDSWISH_FWD_ELE_NUM_1) {
                    return std::min(((totalSize + HARDSWISH_FWD_ELE_NUM_STEP1 - 1) / HARDSWISH_FWD_ELE_NUM_STEP1),
                        ELE_THREADS_1NUMA);
                } else if (totalSize < HARDSWISH_FWD_ELE_NUM_2) {
                    return ELE_THREADS_2NUMA;
                } else {
                    return std::min(((totalSize + HARDSWISH_FWD_ELE_NUM_STEP2 - 1) / HARDSWISH_FWD_ELE_NUM_STEP2),
                        static_cast<KuDNN::SizeType>(DEFAULT_MAX_NUM_THREADS));
                }
                break;
            }
            case ActivationFunction::LOG: {
                if (totalSize < LOG_FWD_ELE_NUM_1) {
                    return (totalSize + LOG_FWD_ELE_NUM_STEP1 - 1) / LOG_FWD_ELE_NUM_STEP1;
                } else if (totalSize < LOG_FWD_ELE_NUM_2) {
                    return ELE_THREADS_1NUMA;
                } else if (totalSize < LOG_FWD_ELE_NUM_3) {
                    return ELE_THREADS_2NUMA;
                } else {
                    return std::min(((totalSize + LOG_FWD_ELE_NUM_STEP2 - 1) / LOG_FWD_ELE_NUM_STEP2),
                        static_cast<KuDNN::SizeType>(DEFAULT_MAX_NUM_THREADS));
                }
                break;
            }
            case ActivationFunction::MISH: {
                if (totalSize < MISH_FWD_ELE_NUM_1) {
                    return (totalSize + MISH_FWD_ELE_NUM_STEP1 - 1) / MISH_FWD_ELE_NUM_STEP1;
                } else if (totalSize < MISH_FWD_ELE_NUM_2) {
                    return ELE_THREADS_1NUMA;
                } else if (totalSize < MISH_FWD_ELE_NUM_3) {
                    return ELE_THREADS_2NUMA;
                } else {
                    return std::min(((totalSize + MISH_FWD_ELE_NUM_STEP2 - 1) / MISH_FWD_ELE_NUM_STEP2),
                        static_cast<KuDNN::SizeType>(DEFAULT_MAX_NUM_THREADS));
                }
                break;
            }
            case ActivationFunction::POW: {
                if (totalSize < POW_FWD_ELE_NUM_1) {
                    return std::min(((totalSize + POW_FWD_ELE_NUM_STEP1 - 1) / POW_FWD_ELE_NUM_STEP1),
                        ELE_THREADS_1NUMA);
                } else if (totalSize < POW_FWD_ELE_NUM_2) {
                    return ELE_THREADS_2NUMA;
                } else if (totalSize < POW_FWD_ELE_NUM_3) {
                    return ELE_THREADS_3NUMA;
                } else {
                    return DEFAULT_MAX_NUM_THREADS;
                }
                break;
            }
            case ActivationFunction::ROUND: {
                if (totalSize < ROUND_FWD_ELE_NUM_1) {
                    return std::min(((totalSize + ROUND_FWD_ELE_NUM_STEP1 - 1) / ROUND_FWD_ELE_NUM_STEP1),
                        ELE_THREADS_1NUMA);
                } else if (totalSize < ROUND_FWD_ELE_NUM_2) {
                    return ELE_THREADS_2NUMA;
                } else {
                    return std::min(((totalSize + ROUND_FWD_ELE_NUM_STEP2 - 1) / ROUND_FWD_ELE_NUM_STEP2),
                        static_cast<KuDNN::SizeType>(DEFAULT_MAX_NUM_THREADS));
                }
                break;
            }
            case ActivationFunction::SRELU: {
                if (totalSize < SRELU_FWD_ELE_NUM_1) {
                    return std::min(((totalSize + SRELU_FWD_ELE_NUM_STEP1 - 1) / SRELU_FWD_ELE_NUM_STEP1),
                        ELE_THREADS_1NUMA);
                } else if (totalSize < SRELU_FWD_ELE_NUM_2) {
                    return ELE_THREADS_2NUMA;
                } else {
                    return std::min(((totalSize + SRELU_FWD_ELE_NUM_STEP2 - 1) / SRELU_FWD_ELE_NUM_STEP2),
                        static_cast<KuDNN::SizeType>(DEFAULT_MAX_NUM_THREADS));
                }
                break;
            }
            case ActivationFunction::SQRT: {
                if (totalSize < SQRT_FWD_ELE_NUM_1) {
                    return (totalSize + SQRT_FWD_ELE_NUM_STEP1 - 1) / SQRT_FWD_ELE_NUM_STEP1;
                } else if (totalSize < SQRT_FWD_ELE_NUM_2) {
                    return ELE_THREADS_1NUMA;
                } else {
                    return (totalSize + SQRT_FWD_ELE_NUM_STEP2 - 1) / SQRT_FWD_ELE_NUM_STEP2;
                }
                break;
            }
            case ActivationFunction::SQUARE: {
                if (totalSize < SQUARE_FWD_ELE_NUM_1) {
                    return std::min(((totalSize + SQUARE_FWD_ELE_NUM_STEP1 - 1) / SQUARE_FWD_ELE_NUM_STEP1),
                        ELE_THREADS_1NUMA);
                } else if (totalSize < SQUARE_FWD_ELE_NUM_2) {
                    return ELE_THREADS_2NUMA;
                } else {
                    return std::min(((totalSize + SQUARE_FWD_ELE_NUM_STEP2 - 1) / SQUARE_FWD_ELE_NUM_STEP2),
                        static_cast<KuDNN::SizeType>(DEFAULT_MAX_NUM_THREADS));
                }
                break;
            }
            case ActivationFunction::SWISH: {
                if (totalSize < SWISH_FWD_ELE_NUM_1) {
                    return (totalSize + SWISH_FWD_ELE_NUM_STEP1 - 1) / SWISH_FWD_ELE_NUM_STEP1;
                } else if (totalSize < SWISH_FWD_ELE_NUM_2) {
                    return ELE_THREADS_1NUMA;
                } else if (totalSize < SWISH_FWD_ELE_NUM_3) {
                    return ELE_THREADS_2NUMA;
                } else {
                    return std::min(((totalSize + SWISH_FWD_ELE_NUM_STEP2 - 1) / SWISH_FWD_ELE_NUM_STEP2),
                        static_cast<KuDNN::SizeType>(DEFAULT_MAX_NUM_THREADS));
                }
                break;
            }
            case ActivationFunction::TANH: {
                if (totalSize < TANH_FWD_ELE_NUM_1) {
                    return std::min(((totalSize + TANH_FWD_ELE_NUM_STEP1 - 1) / TANH_FWD_ELE_NUM_STEP1),
                        ELE_THREADS_1NUMA);
                } else if (totalSize < TANH_FWD_ELE_NUM_2) {
                    return ELE_THREADS_2NUMA;
                } else {
                    return std::min(((totalSize + TANH_FWD_ELE_NUM_STEP2 - 1) / TANH_FWD_ELE_NUM_STEP2),
                        static_cast<KuDNN::SizeType>(DEFAULT_MAX_NUM_THREADS));
                }
                break;
            }
            default: {
                return (totalSize + OTHER_ELE_NUM_DEFAULT - 1) / OTHER_ELE_NUM_DEFAULT;
            }
        }
    }
};

class ActivationLayerImplBWD final : public Parallelizable {
public:
    ActivationLayerImplBWD(const TensorInfo &dsInfo, const TensorInfo &ddInfo, const TensorInfo &srcInfo,
        ActivationFunction kind, float alpha, float beta) noexcept(false)
        : activationKind(kind), dsType(dsInfo.GetType()), ddType(ddInfo.GetType()), srcType(srcInfo.GetType()),
        alphaInfo(alpha), betaInfo(beta)
    {
        if (kind == ActivationFunction::LINEAR) {
            betaInfo = 0.0f;
        }
        Service::ThrowOnStatus(ActivationLayerBWD::ValidateInput(
            dsInfo, ddInfo, srcInfo, kind, alphaInfo, betaInfo), "Activation layer");
        reorderedDims = Service::BroadcastTo5D(Service::GetShapeAccordingToLayout(srcInfo.GetDims(),
            srcInfo.GetLayout()));
        reorderedStridesDs  = Service::BroadcastTo5D(Service::GetShapeAccordingToLayout(dsInfo.GetStrides(),
            dsInfo.GetLayout()));
        reorderedStridesDd  = Service::BroadcastTo5D(Service::GetShapeAccordingToLayout(ddInfo.GetStrides(),
            ddInfo.GetLayout()));
        reorderedStridesSrc = Service::BroadcastTo5D(Service::GetShapeAccordingToLayout(srcInfo.GetStrides(),
            srcInfo.GetLayout()));
        Parallelizable::Initialize(GetOptNumThreads(), reorderedDims);
    }

    void RunSequential(const Shape &batchSplitDims, void *ds,
        const void *dd, const void *src) const noexcept(false)
    {
        if (activationKind == ActivationFunction::LINEAR) {
            if (srcType == Element::TypeT::F32) {
                LinearSveF32(static_cast<int>(batchSplitDims.GetTotalDimsSize()), alphaInfo, betaInfo,
                    static_cast<const float *>(dd), static_cast<float *>(ds));
            } else if (srcType == Element::TypeT::F16) {
                LinearSveF16(static_cast<int>(batchSplitDims.GetTotalDimsSize()), alphaInfo, betaInfo,
                    static_cast<const __fp16 *>(dd), static_cast<__fp16 *>(ds));
            } else if (srcType == Element::TypeT::BF16) {
                LinearSveBF16(static_cast<int>(batchSplitDims.GetTotalDimsSize()), alphaInfo, betaInfo,
                    static_cast<const __bf16 *>(dd), static_cast<__bf16 *>(ds));
            }
        }

        auto item = eltFuncBwdMap.find(activationKind);
        if (item == eltFuncBwdMap.end()) {
            return;
        }

        if (Service::OneOf(activationKind, ActivationFunction::ABS, ActivationFunction::EXP,
            ActivationFunction::EXP_DST, ActivationFunction::GELU_ERF, ActivationFunction::GELU_TANH,
            ActivationFunction::LOG, ActivationFunction::LOGISTIC_DST, ActivationFunction::SIGMOID,
            ActivationFunction::MISH, ActivationFunction::SQRT, ActivationFunction::SQRT_DST,
            ActivationFunction::SQUARE, ActivationFunction::TANH, ActivationFunction::TANH_DST)) {
            if (srcType == Element::TypeT::F32) {
                reinterpret_cast<EltwiseBwdFuncF32X>(std::get<0x0>(item->second))(
                    static_cast<int>(batchSplitDims.GetTotalDimsSize()),
                    static_cast<const float *>(src), static_cast<const float *>(dd), static_cast<float *>(ds));
            } else if (srcType == Element::TypeT::F16) {
                reinterpret_cast<EltwiseBwdFuncF16X>(std::get<0x1>(item->second))(
                    static_cast<int>(batchSplitDims.GetTotalDimsSize()),
                    static_cast<const __fp16 *>(src), static_cast<const __fp16 *>(dd), static_cast<__fp16 *>(ds));
            } else if (srcType == Element::TypeT::BF16) {
                reinterpret_cast<EltwiseBwdFuncBF16X>(std::get<0x2>(item->second))(
                    static_cast<int>(batchSplitDims.GetTotalDimsSize()),
                    static_cast<const __bf16 *>(src), static_cast<const __bf16 *>(dd), static_cast<__bf16 *>(ds));
            }
        } else if (Service::OneOf(activationKind, ActivationFunction::ELU, ActivationFunction::ELU_DST,
            ActivationFunction::RELU, ActivationFunction::RELU_DST, ActivationFunction::SRELU,
            ActivationFunction::SWISH)) {
            if (srcType == Element::TypeT::F32) {
                reinterpret_cast<EltwiseBwdFuncF32AX>(std::get<0x0>(item->second))(
                    static_cast<int>(batchSplitDims.GetTotalDimsSize()), alphaInfo,
                    static_cast<const float *>(src), static_cast<const float *>(dd), static_cast<float *>(ds));
            } else if (srcType == Element::TypeT::F16) {
                reinterpret_cast<EltwiseBwdFuncF16AX>(std::get<0x1>(item->second))(
                    static_cast<int>(batchSplitDims.GetTotalDimsSize()), alphaInfo,
                    static_cast<const __fp16 *>(src), static_cast<const __fp16 *>(dd), static_cast<__fp16 *>(ds));
            } else if (srcType == Element::TypeT::BF16) {
                reinterpret_cast<EltwiseBwdFuncBF16AX>(std::get<0x2>(item->second))(
                    static_cast<int>(batchSplitDims.GetTotalDimsSize()), alphaInfo,
                    static_cast<const __bf16 *>(src), static_cast<const __bf16 *>(dd), static_cast<__bf16 *>(ds));
            }
        } else if (Service::OneOf(activationKind, ActivationFunction::CLIP, ActivationFunction::CLIP_V2,
            ActivationFunction::CLIP_V2_DST, ActivationFunction::HARDSIGMOID, ActivationFunction::HARDSWISH,
            ActivationFunction::POW)) {
            if (srcType == Element::TypeT::F32) {
                reinterpret_cast<EltwiseBwdFuncF32ABX>(std::get<0x0>(item->second))(
                    static_cast<int>(batchSplitDims.GetTotalDimsSize()), alphaInfo, betaInfo,
                    static_cast<const float *>(src), static_cast<const float *>(dd), static_cast<float *>(ds));
            } else if (srcType == Element::TypeT::F16) {
                reinterpret_cast<EltwiseBwdFuncF16ABX>(std::get<0x1>(item->second))(
                    static_cast<int>(batchSplitDims.GetTotalDimsSize()), alphaInfo, betaInfo,
                    static_cast<const __fp16 *>(src), static_cast<const __fp16 *>(dd), static_cast<__fp16 *>(ds));
            } else if (srcType == Element::TypeT::BF16) {
                reinterpret_cast<EltwiseBwdFuncBF16ABX>(std::get<0x2>(item->second))(
                    static_cast<int>(batchSplitDims.GetTotalDimsSize()), alphaInfo, betaInfo,
                    static_cast<const __bf16 *>(src), static_cast<const __bf16 *>(dd), static_cast<__bf16 *>(ds));
            }
        }
    }

    void RunSequential(const Shape &batchSplitDims, const Shape &batchStrides, void *ds,
        const void *dd, const void *src) const noexcept(false)
    {
        SizeType strideDs = 0;
        SizeType strideDd = 0;
        SizeType strideSrc = 0;
        for (SizeType i = 0; i < batchStrides.GetNumDims(); ++i) {
            strideDs += batchStrides[i] * reorderedStridesDs[i] * dsType.GetSize();
            strideDd += batchStrides[i] * reorderedStridesDd[i] * ddType.GetSize();
            strideSrc += batchStrides[i] * reorderedStridesSrc[i] * srcType.GetSize();
        }
        RunSequential(batchSplitDims, static_cast<std::byte *>(ds) + strideDs,
            static_cast<const std::byte *>(dd) + strideDd, static_cast<const std::byte *>(src) + strideSrc);
    }

    void Run(void *ds, const void *dd, const void *src) noexcept(false)
    {
        if ((ds == nullptr) || (dd == nullptr) || (src == nullptr)) {
            throw Service::LogicError {"Activation Layer: one of input matrices is nullptr"};
        }
        Threading::BatchThreader(*this, ds, dd, src);
    }

private:
    ActivationFunction activationKind;
    Element::Type dsType;
    Element::Type ddType;
    Element::Type srcType;
    float alphaInfo;
    float betaInfo;
    Shape reorderedDims;
    Shape reorderedStridesDs;
    Shape reorderedStridesDd;
    Shape reorderedStridesSrc;

    SizeType GetOptNumThreads() const noexcept
    {
        SizeType totalSize = reorderedDims.GetTotalDimsSize();
        switch (activationKind) {
            case ActivationFunction::SIGMOID: {
                if (totalSize < SIGMOID_BWD_ELE_NUM_1) {
                    return std::min(((totalSize + SIGMOID_BWD_ELE_NUM_STEP1 - 1) / SIGMOID_BWD_ELE_NUM_STEP1),
                        ELE_THREADS_1NUMA);
                } else if (totalSize < SIGMOID_BWD_ELE_NUM_2) {
                    return ELE_THREADS_2NUMA;
                } else {
                    return std::min(((totalSize + SIGMOID_BWD_ELE_NUM_STEP2 - 1) / SIGMOID_BWD_ELE_NUM_STEP2),
                        static_cast<KuDNN::SizeType>(DEFAULT_MAX_NUM_THREADS));
                }
                break;
            }
            case ActivationFunction::RELU: {
                if (totalSize < RELU_BWD_ELE_NUM_1) {
                    return std::min(((totalSize + RELU_BWD_ELE_NUM_STEP1 - 1) / RELU_BWD_ELE_NUM_STEP1),
                        ELE_THREADS_1NUMA);
                } else if (totalSize < RELU_BWD_ELE_NUM_2) {
                    return ELE_THREADS_2NUMA;
                } else if (totalSize < RELU_BWD_ELE_NUM_3) {
                    return ELE_THREADS_3NUMA;
                } else {
                    return DEFAULT_MAX_NUM_THREADS;
                }
                break;
            }
            case ActivationFunction::LINEAR: {
                if (totalSize < LINEAR_BWD_ELE_NUM_1) {
                    return std::min(((totalSize + LINEAR_BWD_ELE_NUM_STEP1 - 1) / LINEAR_BWD_ELE_NUM_STEP1),
                        ELE_THREADS_1NUMA);
                } else if (totalSize < LINEAR_BWD_ELE_NUM_2) {
                    return ELE_THREADS_2NUMA;
                } else {
                    return std::min(((totalSize + LINEAR_BWD_ELE_NUM_STEP2 - 1) / LINEAR_BWD_ELE_NUM_STEP2),
                        static_cast<KuDNN::SizeType>(DEFAULT_MAX_NUM_THREADS));
                }
                break;
            }
            case ActivationFunction::ABS: {
                if (totalSize < ABS_BWD_ELE_NUM_1) {
                    return std::min(((totalSize + ABS_BWD_ELE_NUM_STEP1 - 1) / ABS_BWD_ELE_NUM_STEP1),
                        ELE_THREADS_1NUMA);
                } else if (totalSize < ABS_BWD_ELE_NUM_2) {
                    return ELE_THREADS_2NUMA;
                } else if (totalSize < ABS_BWD_ELE_NUM_3) {
                    return ELE_THREADS_3NUMA;
                } else {
                    return DEFAULT_MAX_NUM_THREADS;
                }
                break;
            }
            case ActivationFunction::CLIP: {
                if (totalSize < CLIP_BWD_ELE_NUM_1) {
                    return std::min(((totalSize + CLIP_BWD_ELE_NUM_STEP1 - 1) / CLIP_BWD_ELE_NUM_STEP1),
                        ELE_THREADS_1NUMA);
                } else if (totalSize < CLIP_BWD_ELE_NUM_2) {
                    return ELE_THREADS_2NUMA;
                } else if (totalSize < CLIP_BWD_ELE_NUM_3) {
                    return ELE_THREADS_3NUMA;
                } else {
                    return DEFAULT_MAX_NUM_THREADS;
                }
                break;
            }
            case ActivationFunction::CLIP_V2: {
                if (totalSize < CLIP_V2_BWD_ELE_NUM_1) {
                    return std::min(((totalSize + CLIP_V2_BWD_ELE_NUM_STEP1 - 1) / CLIP_V2_BWD_ELE_NUM_STEP1),
                        ELE_THREADS_1NUMA);
                } else if (totalSize < CLIP_V2_BWD_ELE_NUM_2) {
                    return ELE_THREADS_2NUMA;
                } else if (totalSize < CLIP_V2_BWD_ELE_NUM_3) {
                    return ELE_THREADS_3NUMA;
                } else {
                    return DEFAULT_MAX_NUM_THREADS;
                }
                break;
            }
            case ActivationFunction::ELU: {
#ifdef KUDNNL_920Pro
                if (totalSize < ELU_BWD_ELE_NUM) {
                    return 1;
                }
                return std::min((totalSize + ELU_BWD_ELE_NUM_STEP - 1) / ELU_BWD_ELE_NUM_STEP,
                    ELE_THREADS_1NUMA);
#else
                if (totalSize < ELU_BWD_ELE_NUM_1) {
                    return std::min(((totalSize + ELU_BWD_ELE_NUM_STEP1 - 1) / ELU_BWD_ELE_NUM_STEP1),
                        ELE_THREADS_1NUMA);
                } else if (totalSize < ELU_BWD_ELE_NUM_2) {
                    return ELE_THREADS_2NUMA;
                } else if (totalSize < ELU_BWD_ELE_NUM_3) {
                    return ELE_THREADS_3NUMA;
                } else {
                    return DEFAULT_MAX_NUM_THREADS;
                }
#endif
                break;
            }
            case ActivationFunction::EXP: {
                if (totalSize < EXP_BWD_ELE_NUM_1) {
                    return std::min(((totalSize + EXP_BWD_ELE_NUM_STEP1 - 1) / EXP_BWD_ELE_NUM_STEP1),
                        ELE_THREADS_1NUMA);
                } else if (totalSize < EXP_BWD_ELE_NUM_2) {
                    return ELE_THREADS_2NUMA;
                } else {
                    return (totalSize + EXP_BWD_ELE_NUM_STEP2 - 1) / EXP_BWD_ELE_NUM_STEP2;
                }
                break;
            }
            case ActivationFunction::GELU_ERF: {
                if (totalSize < GELU_ERF_BWD_ELE_NUM_1) {
                    return std::min(((totalSize + GELU_ERF_BWD_ELE_NUM_STEP1 - 1) / GELU_ERF_BWD_ELE_NUM_STEP1),
                        ELE_THREADS_1NUMA);
                } else if (totalSize < GELU_ERF_BWD_ELE_NUM_2) {
                    return ELE_THREADS_2NUMA;
                } else if (totalSize < GELU_ERF_BWD_ELE_NUM_3) {
                    return ELE_THREADS_3NUMA;
                } else {
                    return DEFAULT_MAX_NUM_THREADS;
                }
                break;
            }
            case ActivationFunction::GELU_TANH: {
                if (totalSize < GELU_TANH_BWD_ELE_NUM_1) {
                    return std::min(((totalSize + GELU_TANH_BWD_ELE_NUM_STEP1 - 1) / GELU_TANH_BWD_ELE_NUM_STEP1),
                        ELE_THREADS_1NUMA);
                } else if (totalSize < GELU_TANH_BWD_ELE_NUM_2) {
                    return ELE_THREADS_2NUMA;
                } else if (totalSize < GELU_TANH_BWD_ELE_NUM_3) {
                    return ELE_THREADS_3NUMA;
                } else {
                    return DEFAULT_MAX_NUM_THREADS;
                }
                break;
            }
            case ActivationFunction::HARDSIGMOID: {
#ifdef KUDNNL_920Pro
                if (totalSize < HARDSIGMOID_BWD_ELE_NUM_1) {
                    return 1;
                }
                if (totalSize < HARDSIGMOID_BWD_ELE_NUM_2) {
                    return (totalSize + HARDSIGMOID_BWD_ELE_NUM_STEP1 - 1) / HARDSIGMOID_BWD_ELE_NUM_STEP1;
                }
                return std::min((totalSize + HARDSIGMOID_BWD_ELE_NUM_STEP2 - 1) / HARDSIGMOID_BWD_ELE_NUM_STEP2,
                    ELE_THREADS_1NUMA);
#else
                if (totalSize < HARDSIGMOID_BWD_ELE_NUM_1) {
                    return (totalSize + HARDSIGMOID_BWD_ELE_NUM_STEP1 - 1) / HARDSIGMOID_BWD_ELE_NUM_STEP1;
                } else if (totalSize < HARDSIGMOID_BWD_ELE_NUM_2) {
                    return ELE_THREADS_1NUMA;
                } else if (totalSize < HARDSIGMOID_BWD_ELE_NUM_3) {
                    return ELE_THREADS_2NUMA;
                } else if (totalSize < HARDSIGMOID_BWD_ELE_NUM_4) {
                    return ELE_THREADS_3NUMA;
                } else {
                    return DEFAULT_MAX_NUM_THREADS;
                }
#endif
                break;
            }
            case ActivationFunction::HARDSWISH: {
#ifdef KUDNNL_920Pro
                if (totalSize < HARDSWISH_BWD_ELE_NUM) {
                    return 1;
                }
                return std::min((totalSize + HARDSWISH_BWD_ELE_NUM_STEP - 1) / HARDSWISH_BWD_ELE_NUM_STEP,
                    ELE_THREADS_1NUMA);
#else
                if (totalSize < HARDSWISH_BWD_ELE_NUM_1) {
                    return std::min(((totalSize + HARDSWISH_BWD_ELE_NUM_STEP1 - 1) / HARDSWISH_BWD_ELE_NUM_STEP1),
                        ELE_THREADS_1NUMA);
                } else if (totalSize < HARDSWISH_BWD_ELE_NUM_2) {
                    return ELE_THREADS_2NUMA;
                } else if (totalSize < HARDSWISH_BWD_ELE_NUM_3) {
                    return ELE_THREADS_3NUMA;
                } else {
                    return ELE_THREADS_4NUMA;
                }
#endif
                break;
            }
            case ActivationFunction::LOG: {
                if (totalSize < LOG_BWD_ELE_NUM_1) {
                    return std::min(((totalSize + LOG_BWD_ELE_NUM_STEP1 - 1) / LOG_BWD_ELE_NUM_STEP1),
                        ELE_THREADS_1NUMA);
                } else if (totalSize < LOG_BWD_ELE_NUM_2) {
                    return ELE_THREADS_2NUMA;
                } else if (totalSize < LOG_BWD_ELE_NUM_3) {
                    return ELE_THREADS_3NUMA;
                } else {
                    return DEFAULT_MAX_NUM_THREADS;
                }
                break;
            }
            case ActivationFunction::MISH: {
                if (totalSize < MISH_BWD_ELE_NUM_1) {
                    return std::min(((totalSize + MISH_BWD_ELE_NUM_STEP1 - 1) / MISH_BWD_ELE_NUM_STEP1),
                        ELE_THREADS_1NUMA);
                } else if (totalSize < MISH_BWD_ELE_NUM_2) {
                    return ELE_THREADS_2NUMA;
                } else if (totalSize < MISH_BWD_ELE_NUM_3) {
                    return ELE_THREADS_3NUMA;
                } else {
                    return DEFAULT_MAX_NUM_THREADS;
                }
                break;
            }
            case ActivationFunction::POW: {
                if (totalSize < POW_BWD_ELE_NUM_1) {
                    return std::min(((totalSize + POW_BWD_ELE_NUM_STEP1 - 1) / POW_BWD_ELE_NUM_STEP1),
                        ELE_THREADS_1NUMA);
                } else if (totalSize < POW_BWD_ELE_NUM_2) {
                    return ELE_THREADS_2NUMA;
                } else if (totalSize < POW_BWD_ELE_NUM_3) {
                    return ELE_THREADS_3NUMA;
                } else {
                    return DEFAULT_MAX_NUM_THREADS;
                }
                break;
            }
            case ActivationFunction::SRELU: {
#ifdef KUDNNL_920Pro
                if (totalSize < SRELU_BWD_ELE_NUM) {
                    return 1;
                }
                return std::min((totalSize + SRELU_BWD_ELE_NUM_STEP - 1) / SRELU_BWD_ELE_NUM_STEP,
                    ELE_THREADS_1NUMA);
#else
                if (totalSize < SRELU_BWD_ELE_NUM_1) {
                    return std::min(((totalSize + SRELU_BWD_ELE_NUM_STEP1 - 1) / SRELU_BWD_ELE_NUM_STEP1),
                        ELE_THREADS_1NUMA);
                } else if (totalSize < SRELU_BWD_ELE_NUM_2) {
                    return ELE_THREADS_2NUMA;
                } else {
                    return std::min(((totalSize + SRELU_BWD_ELE_NUM_STEP2 - 1) / SRELU_BWD_ELE_NUM_STEP2),
                        static_cast<KuDNN::SizeType>(DEFAULT_MAX_NUM_THREADS));
                }
#endif
                break;
            }
            case ActivationFunction::SQRT: {
                if (totalSize < SQRT_BWD_ELE_NUM_1) {
                    return std::min(((totalSize + SQRT_BWD_ELE_NUM_STEP1 - 1) / SQRT_BWD_ELE_NUM_STEP1),
                        ELE_THREADS_1NUMA);
                } else if (totalSize < SQRT_BWD_ELE_NUM_2) {
                    return ELE_THREADS_2NUMA;
                } else {
                    return (totalSize + SQRT_BWD_ELE_NUM_STEP2 - 1) / SQRT_BWD_ELE_NUM_STEP2;
                }
                break;
            }
            case ActivationFunction::SQUARE: {
                if (totalSize < SQUARE_BWD_ELE_NUM_1) {
                    return std::min(((totalSize + SQUARE_BWD_ELE_NUM_STEP1 - 1) / SQUARE_BWD_ELE_NUM_STEP1),
                        ELE_THREADS_1NUMA);
                } else if (totalSize < SQUARE_BWD_ELE_NUM_2) {
                    return ELE_THREADS_2NUMA;
                } else if (totalSize < SQUARE_BWD_ELE_NUM_3) {
                    return ELE_THREADS_3NUMA;
                } else {
                    return DEFAULT_MAX_NUM_THREADS;
                }
                break;
            }
            case ActivationFunction::SWISH: {
#ifdef KUDNNL_920Pro
                if (totalSize < SWISH_BWD_ELE_NUM) {
                    return 1;
                }
                return std::min((totalSize + SWISH_BWD_ELE_NUM_STEP - 1) / SWISH_BWD_ELE_NUM_STEP,
                    ELE_THREADS_1NUMA);
#else
                if (totalSize < SWISH_BWD_ELE_NUM_1) {
                    return std::min(((totalSize + SWISH_BWD_ELE_NUM_STEP1 - 1) / SWISH_BWD_ELE_NUM_STEP1),
                        ELE_THREADS_1NUMA);
                } else if (totalSize < SWISH_BWD_ELE_NUM_2) {
                    return ELE_THREADS_2NUMA;
                } else if (totalSize < SWISH_BWD_ELE_NUM_3) {
                    return ELE_THREADS_3NUMA;
                } else {
                    return DEFAULT_MAX_NUM_THREADS;
                }
#endif
                break;
            }
            case ActivationFunction::TANH: {
#ifdef KUDNNL_920Pro
                if (totalSize < TANH_BWD_ELE_NUM) {
                    return 1;
                }
                return std::min((totalSize + TANH_BWD_ELE_NUM_STEP - 1) / TANH_BWD_ELE_NUM_STEP,
                    ELE_THREADS_1NUMA);
#else
                if (totalSize < TANH_BWD_ELE_NUM_1) {
                    return std::min(((totalSize + TANH_BWD_ELE_NUM_STEP1 - 1) / TANH_BWD_ELE_NUM_STEP1),
                        ELE_THREADS_1NUMA);
                } else if (totalSize < TANH_BWD_ELE_NUM_2) {
                    return ELE_THREADS_2NUMA;
                } else if (totalSize < TANH_BWD_ELE_NUM_3) {
                    return ELE_THREADS_3NUMA;
                } else {
                    return DEFAULT_MAX_NUM_THREADS;
                }
#endif
                break;
            }
            case ActivationFunction::CLIP_V2_DST: {
                if (totalSize < CLIP_V2_DST_BWD_ELE_NUM_1) {
                    return std::min(((totalSize + CLIP_V2_DST_BWD_ELE_NUM_STEP1 - 1) / CLIP_V2_DST_BWD_ELE_NUM_STEP1),
                        ELE_THREADS_1NUMA);
                } else if (totalSize < CLIP_V2_DST_BWD_ELE_NUM_2) {
                    return ELE_THREADS_2NUMA;
                } else if (totalSize < CLIP_V2_DST_BWD_ELE_NUM_3) {
                    return ELE_THREADS_3NUMA;
                } else {
                    return ELE_THREADS_4NUMA;
                }
                break;
            }
            case ActivationFunction::ELU_DST: {
                if (totalSize < ELU_DST_BWD_ELE_NUM_1) {
                    return std::min(((totalSize + ELU_DST_BWD_ELE_NUM_STEP1 - 1) / ELU_DST_BWD_ELE_NUM_STEP1),
                        ELE_THREADS_1NUMA);
                } else if (totalSize < ELU_DST_BWD_ELE_NUM_2) {
                    return ELE_THREADS_2NUMA;
                } else if (totalSize < ELU_DST_BWD_ELE_NUM_3) {
                    return ELE_THREADS_3NUMA;
                } else {
                    return ELE_THREADS_4NUMA;
                }
                break;
            }
            case ActivationFunction::EXP_DST: {
                if (totalSize < EXP_DST_BWD_ELE_NUM_1) {
                    return (totalSize + EXP_DST_BWD_ELE_NUM_STEP1 - 1) / EXP_DST_BWD_ELE_NUM_STEP1;
                } else if (totalSize < EXP_DST_BWD_ELE_NUM_2) {
                    return ELE_THREADS_1NUMA;
                } else if (totalSize < EXP_DST_BWD_ELE_NUM_3) {
                    return ELE_THREADS_2NUMA;
                } else if (totalSize < EXP_DST_BWD_ELE_NUM_4) {
                    return ELE_THREADS_3NUMA;
                } else {
                    return ELE_THREADS_4NUMA;
                }
                break;
            }
            case ActivationFunction::LOGISTIC_DST: {
                if (totalSize < LOGISTIC_DST_BWD_ELE_NUM_1) {
                    return std::min(((totalSize + LOGISTIC_DST_BWD_ELE_NUM_STEP1 - 1) / LOGISTIC_DST_BWD_ELE_NUM_STEP1),
                        ELE_THREADS_1NUMA);
                } else if (totalSize < LOGISTIC_DST_BWD_ELE_NUM_2) {
                    return ELE_THREADS_2NUMA;
                } else {
                    return (totalSize + LOGISTIC_DST_BWD_ELE_NUM_STEP2 - 1) / LOGISTIC_DST_BWD_ELE_NUM_STEP2;
                }
                break;
            }
            case ActivationFunction::RELU_DST: {
                if (totalSize < RELU_DST_BWD_ELE_NUM_1) {
                    return std::min(((totalSize + RELU_DST_BWD_ELE_NUM_STEP1 - 1) / RELU_DST_BWD_ELE_NUM_STEP1),
                        ELE_THREADS_1NUMA);
                } else if (totalSize < RELU_DST_BWD_ELE_NUM_2) {
                    return ELE_THREADS_2NUMA;
                } else if (totalSize < RELU_DST_BWD_ELE_NUM_3) {
                    return ELE_THREADS_3NUMA;
                } else {
                    return ELE_THREADS_4NUMA;
                }
                break;
            }
            case ActivationFunction::SQRT_DST: {
                if (totalSize < SQRT_DST_BWD_ELE_NUM_1) {
                    return (totalSize + SQRT_DST_BWD_ELE_NUM_STEP1 - 1) / SQRT_DST_BWD_ELE_NUM_STEP1;
                } else if (totalSize < SQRT_DST_BWD_ELE_NUM_2) {
                    return ELE_THREADS_1NUMA;
                } else if (totalSize < SQRT_DST_BWD_ELE_NUM_3) {
                    return ELE_THREADS_2NUMA;
                } else {
                    return (totalSize + SQRT_DST_BWD_ELE_NUM_STEP2 - 1) / SQRT_DST_BWD_ELE_NUM_STEP2;
                }
                break;
            }
            case ActivationFunction::TANH_DST: {
                if (totalSize < TANH_DST_BWD_ELE_NUM_1) {
                    return (totalSize + TANH_DST_BWD_ELE_NUM_STEP1 - 1) / TANH_DST_BWD_ELE_NUM_STEP1;
                } else if (totalSize < TANH_DST_BWD_ELE_NUM_2) {
                    return ELE_THREADS_1NUMA;
                } else if (totalSize < TANH_DST_BWD_ELE_NUM_3) {
                    return ELE_THREADS_2NUMA;
                } else if (totalSize < TANH_DST_BWD_ELE_NUM_4) {
                    return ELE_THREADS_3NUMA;
                } else {
                    return ELE_THREADS_4NUMA;
                }
                break;
            }
            default: {
                return (totalSize + OTHER_ELE_NUM_DEFAULT - 1) / OTHER_ELE_NUM_DEFAULT;
            }
        }
    }
};

} // Detail

ActivationLayerFWD::~ActivationLayerFWD() noexcept {}

ActivationLayerFWD::ActivationLayerFWD(const TensorInfo &srcInfo, const TensorInfo &dstInfo,
    ActivationFunction kind, float alpha, float beta) noexcept(false)
    : pImpl(new Detail::ActivationLayerImplFWD(srcInfo, dstInfo, kind, alpha, beta)) {}

ActivationLayerFWD::ActivationLayerFWD(const ActivationLayerFWD &other) noexcept(false)
    : pImpl(new Detail::ActivationLayerImplFWD{*(other.pImpl.get())}) {}

ActivationLayerFWD::ActivationLayerFWD(ActivationLayerFWD &&other) noexcept
    : pImpl(std::move(other.pImpl)) {}

ActivationLayerFWD& ActivationLayerFWD::operator=(const ActivationLayerFWD &other) noexcept(false)
{
    if (this != &other) {
        this->pImpl.reset(new Detail::ActivationLayerImplFWD{*(other.pImpl.get())});
    }
    return *this;
}

ActivationLayerFWD& ActivationLayerFWD::operator=(ActivationLayerFWD &&other) noexcept
{
    if (this != &other) {
        this->pImpl = std::move(other.pImpl);
    }
    return *this;
}

void ActivationLayerFWD::Run(const void *src, void *dst) const noexcept(false)
{
    pImpl->Run(src, dst);
}

Status ActivationLayerFWD::ValidateInput(const TensorInfo &srcInfo, const TensorInfo &dstInfo,
    ActivationFunction kind, float alpha, float beta) noexcept(false)
{
    Service::Unused(alpha, beta);

    if (srcInfo.GetLayout() != dstInfo.GetLayout()) {
        return Status::BAD_ARGUMENTS;
    }
    if (!(Service::OneOf(srcInfo.GetType(), Element::TypeT::F32, Element::TypeT::F16, Element::TypeT::BF16) ||
    (Service::OneOf(srcInfo.GetType(), Element::TypeT::S32, Element::TypeT::S8, Element::TypeT::U8) &&
    Service::OneOf(kind, ActivationFunction::RELU, ActivationFunction::LINEAR)))) {
        return Status::UNSUPPORTED;
    }
    if (!Service::OneOf(kind, ActivationFunction::ABS, ActivationFunction::CLIP, ActivationFunction::CLIP_V2,
        ActivationFunction::ELU, ActivationFunction::ELU_DST, ActivationFunction::EXP, ActivationFunction::GELU_ERF,
        ActivationFunction::GELU_TANH, ActivationFunction::HARDSIGMOID, ActivationFunction::HARDSWISH,
        ActivationFunction::LINEAR, ActivationFunction::LOG, ActivationFunction::SIGMOID,
        ActivationFunction::MISH, ActivationFunction::POW, ActivationFunction::RELU, ActivationFunction::ROUND,
        ActivationFunction::SRELU, ActivationFunction::SQRT, ActivationFunction::SQUARE, ActivationFunction::SWISH,
        ActivationFunction::TANH)) {
        return Status::UNSUPPORTED;
    }
    if ((Service::OneOf(kind, ActivationFunction::CLIP, ActivationFunction::CLIP_V2) && (alpha > beta)) ||
        (Service::OneOf(kind, ActivationFunction::SRELU) && (alpha == 0))) {
        return Status::UNSUPPORTED;
    }
    if (kind == ActivationFunction::UNIMPLEMENTED) {
        return Status::UNSUPPORTED;
    }
    if (!srcInfo.IsDense()) {
        return Status::UNSUPPORTED;
    }
    if (srcInfo.GetTotalTensorSize() == 0) {
        return Status::UNSUPPORTED;
    }
    return Status::SUCCESS;
}


ActivationLayerBWD::~ActivationLayerBWD() noexcept {}

ActivationLayerBWD::ActivationLayerBWD(const TensorInfo &dsInfo, const TensorInfo &ddInfo,
    const TensorInfo &srcInfo, ActivationFunction kind, float alpha, float beta) noexcept(false)
    : pImpl(new Detail::ActivationLayerImplBWD(dsInfo, ddInfo, srcInfo, kind, alpha, beta)) {}

ActivationLayerBWD::ActivationLayerBWD(const ActivationLayerBWD &other) noexcept(false)
    : pImpl(new Detail::ActivationLayerImplBWD{*(other.pImpl.get())}) {}

ActivationLayerBWD::ActivationLayerBWD(ActivationLayerBWD &&other) noexcept
    : pImpl(std::move(other.pImpl)) {}

ActivationLayerBWD& ActivationLayerBWD::operator=(const ActivationLayerBWD &other) noexcept(false)
{
    if (this != &other) {
        this->pImpl.reset(new Detail::ActivationLayerImplBWD{*(other.pImpl.get())});
    }
    return *this;
}

ActivationLayerBWD& ActivationLayerBWD::operator=(ActivationLayerBWD &&other) noexcept
{
    if (this != &other) {
        this->pImpl = std::move(other.pImpl);
    }
    return *this;
}

void ActivationLayerBWD::Run(void *ds, const void *dd, const void *src) const noexcept(false)
{
    pImpl->Run(ds, dd, src);
}

Status ActivationLayerBWD::ValidateInput(const TensorInfo &dsInfo, const TensorInfo &ddInfo,
    const TensorInfo &srcInfo, ActivationFunction kind, float alpha, float beta) noexcept
{
    Service::Unused(alpha, beta);
    if (dsInfo != ddInfo || dsInfo != srcInfo) {
        return Status::BAD_ARGUMENTS;
    }
    if (!Service::OneOf(dsInfo.GetType(), Element::TypeT::F32, Element::TypeT::F16, Element::TypeT::BF16)) {
        return Status::UNSUPPORTED;
    }
    if (!Service::OneOf(kind, ActivationFunction::ABS, ActivationFunction::CLIP, ActivationFunction::CLIP_V2,
        ActivationFunction::CLIP_V2_DST, ActivationFunction::ELU, ActivationFunction::ELU_DST,
        ActivationFunction::EXP, ActivationFunction::EXP_DST, ActivationFunction::GELU_ERF,
        ActivationFunction::GELU_TANH, ActivationFunction::HARDSIGMOID, ActivationFunction::HARDSWISH,
        ActivationFunction::LINEAR, ActivationFunction::LOG, ActivationFunction::SIGMOID,
        ActivationFunction::LOGISTIC_DST, ActivationFunction::MISH, ActivationFunction::POW,
        ActivationFunction::RELU, ActivationFunction::RELU_DST, ActivationFunction::SRELU,
        ActivationFunction::SQRT, ActivationFunction::SQRT_DST, ActivationFunction::SQUARE,
        ActivationFunction::SWISH, ActivationFunction::TANH, ActivationFunction::TANH_DST)) {
        return Status::UNSUPPORTED;
    }
    // the list of functions with backward support will be extended
    if ((Service::OneOf(kind, ActivationFunction::CLIP, ActivationFunction::CLIP_V2, ActivationFunction::CLIP_V2_DST)
        && (alpha > beta)) || (Service::OneOf(kind, ActivationFunction::ELU_DST, ActivationFunction::RELU_DST)
        && (alpha < 0))) {
        return Status::UNSUPPORTED;
    }
    if (!dsInfo.IsDense()) {
        return Status::UNSUPPORTED;
    }
    if (srcInfo.GetTotalTensorSize() == 0) {
        return Status::UNSUPPORTED;
    }
    return Status::SUCCESS;
}

} // KuDNN
