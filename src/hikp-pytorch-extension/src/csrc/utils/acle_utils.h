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
#include <utility>

#ifdef KPEX_920F
typedef svfloat32_t simdfloat32_t
        __attribute__((aligned(32))) __attribute__((arm_sve_vector_bits(512)));
#else
typedef svfloat32_t simdfloat32_t
        __attribute__((aligned(32))) __attribute__((arm_sve_vector_bits(256)));
#endif

template <typename T, T... indexes, typename F>
constexpr void LoopUnFoldHelper(std::integer_sequence<T, indexes...>, F &&f)
{
    (f(std::integral_constant<T, indexes>{}), ...);
}

template <typename T, T count, typename F,
          typename = std::enable_if_t<std::is_invocable_v<F, T>>>
constexpr void LoopUnFoldFunc(F &&f)
{
    LoopUnFoldHelper(std::make_integer_sequence<T, count>{}, std::forward<F>(f));
}