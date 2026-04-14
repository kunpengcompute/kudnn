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

#ifndef KUDNN_VECTOR_MEM_DUPFLOAT_HPP
#define KUDNN_VECTOR_MEM_DUPFLOAT_HPP

#include <cstdint>
#include <limits>
#include <arm_neon.h>
#include <arm_sve.h>

#include "kudnn.hpp"

#include "internal_funcs.hpp"
#include "vector_length.hpp"

namespace KuDNN {
namespace Detail {
namespace Vector {

template <>
struct DupFloatN<float, float, 1> final {
    static void Do(const float &src,
        typename Vec<float>::Type &in0, typename Vec<float>::Type &in1,
        typename Vec<float>::Type &in2, typename Vec<float>::Type &in3) noexcept
    {
        in0 = svdup_n_f32(src);
        Service::Unused(in1, in2, in3);
    }
};

template <>
struct DupFloatN<float, float, 2> final { // 2 means that duplication two time for one f32 src
    static void Do(const float &src,
        typename Vec<float>::Type &in0, typename Vec<float>::Type &in1, typename Vec<float>::Type &in2,
        typename Vec<float>::Type &in3) noexcept
    {
        in0 = svdup_n_f32(src);
        in1 = svdup_n_f32(src);
        Service::Unused(in2, in3);
    }
};

template <>
struct DupFloatN<float, float, 4> final { // 4 means that duplication four time for one f32 src
    static void Do(const float &src,
        typename Vec<float>::Type &in0, typename Vec<float>::Type &in1, typename Vec<float>::Type &in2,
        typename Vec<float>::Type &in3) noexcept
    {
        in0 = svdup_n_f32(src);
        in1 = svdup_n_f32(src);
        in2 = svdup_n_f32(src);
        in3 = svdup_n_f32(src);
    }
};

template <>
struct DupFloatN<__bf16, float, 1> final {
    static void Do(const float &src,
        typename Vec<float>::Type &in0, typename Vec<float>::Type &in1, typename Vec<float>::Type &in2,
        typename Vec<float>::Type &in3) noexcept
    {
        in0 = svdup_n_f32(src);
        in1 = svdup_n_f32(src);
        Service::Unused(in2, in3);
    }
};

template <>
struct DupFloatN<__bf16, float, 2> final { // 2 means that duplication two time for one f32 src
    static void Do(const float &src,
        typename Vec<float>::Type &in0, typename Vec<float>::Type &in1, typename Vec<float>::Type &in2,
        typename Vec<float>::Type &in3) noexcept
    {
        in0 = svdup_n_f32(src);
        in1 = svdup_n_f32(src);
        in2 = svdup_n_f32(src);
        in3 = svdup_n_f32(src);
    }
};

template <>
struct DupFloatN<__fp16, float, 1> final {
    static void Do(const float &src,
        typename Vec<float>::Type &in0, typename Vec<float>::Type &in1, typename Vec<float>::Type &in2,
        typename Vec<float>::Type &in3) noexcept
    {
        in0 = svdup_n_f32(src);
        in1 = svdup_n_f32(src);
        Service::Unused(in2, in3);
    }
};

template <>
struct DupFloatN<__fp16, float, 2> final { // 2 means that duplication two time for one f32 src
    static void Do(const float &src,
        typename Vec<float>::Type &in0, typename Vec<float>::Type &in1, typename Vec<float>::Type &in2,
        typename Vec<float>::Type &in3) noexcept
    {
        in0 = svdup_n_f32(src);
        in1 = svdup_n_f32(src);
        in2 = svdup_n_f32(src);
        in3 = svdup_n_f32(src);
    }
};

template <>
struct DupFloatN<std::int8_t, float, 1> final {
    static void Do(const float &src,
        typename Vec<float>::Type &in0, typename Vec<float>::Type &in1, typename Vec<float>::Type &in2,
        typename Vec<float>::Type &in3) noexcept
    {
        in0 = svdup_n_f32(src);
        in1 = svdup_n_f32(src);
        in2 = svdup_n_f32(src);
        in3 = svdup_n_f32(src);
    }
};

template <>
struct DupFloatN<std::uint8_t, float, 1> final {
    static void Do(const float &src,
        typename Vec<float>::Type &in0, typename Vec<float>::Type &in1, typename Vec<float>::Type &in2,
        typename Vec<float>::Type &in3) noexcept
    {
        in0 = svdup_n_f32(src);
        in1 = svdup_n_f32(src);
        in2 = svdup_n_f32(src);
        in3 = svdup_n_f32(src);
    }
};

template <>
struct DupFloatN<std::int32_t, float, 1> final {
    static void Do(const float &src,
        typename Vec<float>::Type &in0, typename Vec<float>::Type &in1,
        typename Vec<float>::Type &in2, typename Vec<float>::Type &in3) noexcept
    {
        in0 = svdup_n_f32(src);
        Service::Unused(in1, in2, in3);
    }
};

template <>
struct DupFloatN<std::int32_t, float, 2> final { // 2 means that duplication two time for one f32 src
    static void Do(const float &src,
        typename Vec<float>::Type &in0, typename Vec<float>::Type &in1, typename Vec<float>::Type &in2,
        typename Vec<float>::Type &in3) noexcept
    {
        in0 = svdup_n_f32(src);
        in1 = svdup_n_f32(src);
        Service::Unused(in2, in3);
    }
};

template <>
struct DupFloatN<std::int32_t, float, 4> final { // 4 means that duplication four time for one f32 src
    static void Do(const float &src,
        typename Vec<float>::Type &in0, typename Vec<float>::Type &in1, typename Vec<float>::Type &in2,
        typename Vec<float>::Type &in3) noexcept
    {
        in0 = svdup_n_f32(src);
        in1 = svdup_n_f32(src);
        in2 = svdup_n_f32(src);
        in3 = svdup_n_f32(src);
    }
};

// CompType is __fp16

template <>
struct DupFloatN<float, __fp16, 2> final { // 2 means that duplication two time for one f32 src
    static void Do(const float &src,
        typename Vec<__fp16>::Type &in0, typename Vec<__fp16>::Type &in1, typename Vec<__fp16>::Type &in2,
        typename Vec<__fp16>::Type &in3) noexcept
    {
        in0 = svdup_n_f16(src);
        Service::Unused(in1, in2, in3);
    }
};

template <>
struct DupFloatN<float, __fp16, 4> final { // 4 means that duplication four time for one f32 src
    static void Do(const float &src,
        typename Vec<__fp16>::Type &in0, typename Vec<__fp16>::Type &in1, typename Vec<__fp16>::Type &in2,
        typename Vec<__fp16>::Type &in3) noexcept
    {
        in0 = svdup_n_f16(src);
        in1 = svdup_n_f16(src);
        Service::Unused(in1, in2, in3);
    }
};

template <>
struct DupFloatN<__bf16, __fp16, 1> final {
    static void Do(const float &src,
        typename Vec<__fp16>::Type &in0, typename Vec<__fp16>::Type &in1, typename Vec<__fp16>::Type &in2,
        typename Vec<__fp16>::Type &in3) noexcept
    {
        in0 = svdup_n_f16(src);
        Service::Unused(in1, in2, in3);
    }
};

template <>
struct DupFloatN<__bf16, __fp16, 2> final { // 2 means that duplication two time for one f32 src
    static void Do(const float &src,
        typename Vec<__fp16>::Type &in0, typename Vec<__fp16>::Type &in1, typename Vec<__fp16>::Type &in2,
        typename Vec<__fp16>::Type &in3) noexcept
    {
        in0 = svdup_n_f16(src);
        in1 = svdup_n_f16(src);
        Service::Unused(in2, in3);
    }
};

template <>
struct DupFloatN<__fp16, __fp16, 1> final {
    static void Do(const float &src,
        typename Vec<__fp16>::Type &in0, typename Vec<__fp16>::Type &in1, typename Vec<__fp16>::Type &in2,
        typename Vec<__fp16>::Type &in3) noexcept
    {
        in0 = svdup_n_f16(src);
        Service::Unused(in1, in2, in3);
    }
};

template <>
struct DupFloatN<__fp16, __fp16, 2> final { // 2 means that duplication two time for one f32 src
    static void Do(const float &src,
        typename Vec<__fp16>::Type &in0, typename Vec<__fp16>::Type &in1, typename Vec<__fp16>::Type &in2,
        typename Vec<__fp16>::Type &in3) noexcept
    {
        in0 = svdup_n_f16(src);
        in1 = svdup_n_f16(src);
        Service::Unused(in2, in3);
    }
};

template <>
struct DupFloatN<std::int8_t, __fp16, 1> final {
    static void Do(const float &src,
        typename Vec<__fp16>::Type &in0, typename Vec<__fp16>::Type &in1, typename Vec<__fp16>::Type &in2,
        typename Vec<__fp16>::Type &in3) noexcept
    {
        in0 = svdup_n_f16(src);
        in1 = svdup_n_f16(src);
        Service::Unused(in2, in3);
    }
};

template <>
struct DupFloatN<std::uint8_t, __fp16, 1> final {
    static void Do(const float &src,
        typename Vec<__fp16>::Type &in0, typename Vec<__fp16>::Type &in1, typename Vec<__fp16>::Type &in2,
        typename Vec<__fp16>::Type &in3) noexcept
    {
        in0 = svdup_n_f16(src);
        in1 = svdup_n_f16(src);
        Service::Unused(in2, in3);
    }
};

template <>
struct DupFloatN<std::int32_t, __fp16, 2> final { // 2 means that duplication two time for one f32 src
    static void Do(const float &src,
        typename Vec<__fp16>::Type &in0, typename Vec<__fp16>::Type &in1, typename Vec<__fp16>::Type &in2,
        typename Vec<__fp16>::Type &in3) noexcept
    {
        in0 = svdup_n_f16(src);
        Service::Unused(in1, in2, in3);
    }
};

template <>
struct DupFloatN<std::int32_t, __fp16, 4> final { // 4 means that duplication four time for one f32 src
    static void Do(const float &src,
        typename Vec<__fp16>::Type &in0, typename Vec<__fp16>::Type &in1, typename Vec<__fp16>::Type &in2,
        typename Vec<__fp16>::Type &in3) noexcept
    {
        in0 = svdup_n_f16(src);
        in1 = svdup_n_f16(src);
        Service::Unused(in1, in2, in3);
    }
};

} // Vector
} // Detail
} // KuDNN

#endif // KUDNN_VECTOR_MEM_DUPFLOAT_HPP