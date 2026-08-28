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
#include <arm_fp16.h>
#include <arm_bf16.h>
#include <type_traits>
#include <torch/extension.h>

template <typename T>
inline T* ConditionalDataPtr(T* ptr, T* ptr2)
{
    return ptr;
}

template <typename T,
          typename std::enable_if_t<std::is_same_v<T, float16_t>, int> = 0>
inline T* ConditionalDataPtr(float* ptr, T* ptr2)
{
    return ptr2;
}

template <typename T,
          typename std::enable_if_t<std::is_same_v<T, bfloat16_t>, int> = 0>
inline T* ConditionalDataPtr(float* ptr, T* ptr2)
{
    return ptr2;
}

template <typename T>
inline T* TensorDataPtr(const at::Tensor& t)
{
    // for fp16 and bf16, libtorch_cpu.so has no symbol data_ptr<float16_t>
    using scalarType = std::conditional_t<std::is_same_v<T, float16_t>,
                       c10::Half,
                       std::conditional_t<std::is_same_v<T, bfloat16_t>, c10::BFloat16, T>>;
    return reinterpret_cast<T*>(t.data_ptr<scalarType>());
}


template <typename T>
inline const T* TensorConstDataPtr(const at::Tensor& t)
{
    // for fp16 and bf16, libtorch_cpu.so has no symbol data_ptr<float16_t>
    using scalarType = std::conditional_t<std::is_same_v<T, float16_t>,
                       c10::Half,
                       std::conditional_t<std::is_same_v<T, bfloat16_t>, c10::BFloat16, T>>;
    return reinterpret_cast<const T*>(t.const_data_ptr<scalarType>());
}