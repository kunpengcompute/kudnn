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

#pragma once

#define KPEX_DISPATCH_FLOAT_CASE(HINT, ...)                        \
    if (tensorType == at::kFloat) {                                \
        using HINT [[maybe_unused]] =  float;                      \
        __VA_ARGS__();                                             \
        dispatched = true;                                         \
    }

#define KPEX_DISPATCH_HALF_CASE(HINT, ...)                         \
    if (tensorType == at::kHalf) {                                 \
        using HINT [[maybe_unused]] =  float16_t;                  \
        __VA_ARGS__();                                             \
        dispatched = true;                                         \
    }

#define KPEX_DISPATCH_BF16_CASE(HINT, ...)                         \
    if (tensorType == at::kBFloat16) {                             \
        using HINT [[maybe_unused]] =  bfloat16_t;                 \
        __VA_ARGS__();                                             \
        dispatched = true;                                         \
    }

#define KPEX_DISPATCH_CONDITIONS(TYPE, ...)                        \
    [&] {                                                          \
        const auto &tensorType = TYPE;                             \
        bool dispatched = false;                                   \
        __VA_ARGS__                                                \
        if (!dispatched) {                                         \
            std::cout << "Unsupported Data Type" << std::endl;     \
        }                                                          \
    } ()

#define KPEX_DISPATCH_FLOATING_AND_HALF_AND_BF16(TYPE, HINT, ...)  \
    KPEX_DISPATCH_CONDITIONS(                                      \
        TYPE,                                                      \
        KPEX_DISPATCH_FLOAT_CASE(HINT, __VA_ARGS__)                \
        KPEX_DISPATCH_HALF_CASE(HINT, __VA_ARGS__)                 \
        KPEX_DISPATCH_BF16_CASE(HINT, __VA_ARGS__))

#define KPEX_DISPATCH_FLOATING_AND_HALF(TYPE, HINT, ...)           \
    KPEX_DISPATCH_CONDITIONS(                                      \
        TYPE,                                                      \
        KPEX_DISPATCH_FLOAT_CASE(HINT, __VA_ARGS__)                \
        KPEX_DISPATCH_HALF_CASE(HINT, __VA_ARGS__))

#define KPEX_DISPATCH_FLOATING(TYPE, HINT, ...)                    \
    KPEX_DISPATCH_CONDITIONS(                                      \
        TYPE,                                                      \
        KPEX_DISPATCH_FLOAT_CASE(HINT, __VA_ARGS__))

#define KPEX_DISPATCH_HALF(TYPE, HINT, ...)                        \
    KPEX_DISPATCH_CONDITIONS(                                      \
        TYPE,                                                      \
        KPEX_DISPATCH_HALF_CASE(HINT, __VA_ARGS__))
