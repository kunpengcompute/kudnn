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

#ifndef KUDNN_VECTOR_MEM_LOAD_COMPTYPE_NEON_HPP
#define KUDNN_VECTOR_MEM_LOAD_COMPTYPE_NEON_HPP

#ifdef KUDNNL_920G

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

// CompType is float

template <>
struct LoadN<float, float, 1, ARCH::NEON> final {
    static void Do(const float *src,
        const svbool_t& pg0, const svbool_t &pg1, const svbool_t &pg2, const svbool_t &pg3,
        typename Vec<float, ARCH::NEON>::Type &in0, typename Vec<float, ARCH::NEON>::Type &in1,
        typename Vec<float, ARCH::NEON>::Type &in2, typename Vec<float, ARCH::NEON>::Type &in3) noexcept
    {
        in0 = vld1q_f32_x2(src);
        Service::Unused(pg0, pg1, in1, pg2, in2, pg3, in3);
    }
};

template <>
struct LoadN<__fp16, float, 1, ARCH::NEON> final {
    static void Do(const __fp16 *src,
        const svbool_t& pg0, const svbool_t &pg1, const svbool_t &pg2, const svbool_t &pg3,
        typename Vec<float, ARCH::NEON>::Type &in0, typename Vec<float, ARCH::NEON>::Type &in1,
        typename Vec<float, ARCH::NEON>::Type &in2, typename Vec<float, ARCH::NEON>::Type &in3) noexcept
    {
        auto vecHalf = vld1q_f16_x2(src);
        in0.val[0] = vcvt_f32_f16(vget_low_f16(vecHalf.val[0]));
        in0.val[1] = vcvt_f32_f16(vget_high_f16(vecHalf.val[0]));
        in1.val[0] = vcvt_f32_f16(vget_low_f16(vecHalf.val[1]));
        in1.val[1] = vcvt_f32_f16(vget_high_f16(vecHalf.val[1]));
        Service::Unused(pg0, pg1, pg2, in2, pg3, in3);
    }
};

template <>
struct LoadN<__fp16, float, 2, ARCH::NEON> final {
    static void Do(const __fp16 *src,
        const svbool_t& pg0, const svbool_t &pg1, const svbool_t &pg2, const svbool_t &pg3,
        typename Vec<float, ARCH::NEON>::Type &in0, typename Vec<float, ARCH::NEON>::Type &in1,
        typename Vec<float, ARCH::NEON>::Type &in2, typename Vec<float, ARCH::NEON>::Type &in3) noexcept
    {
        auto vecHalf = vld1q_f16_x4(src);
        in0.val[0] = vcvt_f32_f16(vget_low_f16(vecHalf.val[0]));
        in0.val[1] = vcvt_f32_f16(vget_high_f16(vecHalf.val[0]));
        in1.val[0] = vcvt_f32_f16(vget_low_f16(vecHalf.val[1]));
        in1.val[1] = vcvt_f32_f16(vget_high_f16(vecHalf.val[1]));
        in2.val[0] = vcvt_f32_f16(vget_low_f16(vecHalf.val[2]));
        in2.val[1] = vcvt_f32_f16(vget_high_f16(vecHalf.val[2]));
        in3.val[0] = vcvt_f32_f16(vget_low_f16(vecHalf.val[3]));
        in3.val[1] = vcvt_f32_f16(vget_high_f16(vecHalf.val[3]));
        Service::Unused(pg0, pg1, pg2, pg3);
    }
};

template <>
struct LoadN<__bf16, float, 1, ARCH::NEON> final {
    static void Do(const __bf16 *src,
        const svbool_t& pg0, const svbool_t &pg1, const svbool_t &pg2, const svbool_t &pg3,
        typename Vec<float, ARCH::NEON>::Type &in0, typename Vec<float, ARCH::NEON>::Type &in1,
        typename Vec<float, ARCH::NEON>::Type &in2, typename Vec<float, ARCH::NEON>::Type &in3) noexcept
    {
        auto vecHalf = vld1q_bf16_x2(src);
        in0.val[0] = vcvt_f32_bf16(vget_low_bf16(vecHalf.val[0]));
        in0.val[1] = vcvt_f32_bf16(vget_high_bf16(vecHalf.val[0]));
        in1.val[0] = vcvt_f32_bf16(vget_low_bf16(vecHalf.val[1]));
        in1.val[1] = vcvt_f32_bf16(vget_high_bf16(vecHalf.val[1]));
        Service::Unused(pg0, pg1, pg2, in2, pg3, in3);
    }
};

template <>
struct LoadN<__bf16, float, 2, ARCH::NEON> final {
    static void Do(const __bf16 *src,
        const svbool_t& pg0, const svbool_t &pg1, const svbool_t &pg2, const svbool_t &pg3,
        typename Vec<float, ARCH::NEON>::Type &in0, typename Vec<float, ARCH::NEON>::Type &in1,
        typename Vec<float, ARCH::NEON>::Type &in2, typename Vec<float, ARCH::NEON>::Type &in3) noexcept
    {
        auto vecHalf = vld1q_bf16_x4(src);
        in0.val[0] = vcvt_f32_bf16(vget_low_bf16(vecHalf.val[0]));
        in0.val[1] = vcvt_f32_bf16(vget_high_bf16(vecHalf.val[0]));
        in1.val[0] = vcvt_f32_bf16(vget_low_bf16(vecHalf.val[1]));
        in1.val[1] = vcvt_f32_bf16(vget_high_bf16(vecHalf.val[1]));
        in2.val[0] = vcvt_f32_bf16(vget_low_bf16(vecHalf.val[2]));
        in2.val[1] = vcvt_f32_bf16(vget_high_bf16(vecHalf.val[2]));
        in3.val[0] = vcvt_f32_bf16(vget_low_bf16(vecHalf.val[3]));
        in3.val[1] = vcvt_f32_bf16(vget_high_bf16(vecHalf.val[3]));
        Service::Unused(pg0, pg1, pg2, pg3);
    }
};

// CompType is f16
template <>
struct LoadN<__fp16, __fp16, 1, ARCH::NEON> final {
    static void Do(const __fp16 *src,
        const svbool_t& pg0, const svbool_t &pg1, const svbool_t &pg2, const svbool_t &pg3,
        typename Vec<__fp16, ARCH::NEON>::Type &in0, typename Vec<__fp16, ARCH::NEON>::Type &in1,
        typename Vec<__fp16, ARCH::NEON>::Type &in2, typename Vec<__fp16, ARCH::NEON>::Type &in3) noexcept
    {
        in0 = vld1q_f16_x2(src);
        Service::Unused(pg0, in0, pg1, in1, pg2, in2, pg3, in3);
    }
};

template <>
struct LoadN<__bf16, __bf16, 1, ARCH::NEON> final {
    static void Do(const __bf16 *src, const svbool_t& pg0, const svbool_t &pg1,
        const svbool_t &pg2, const svbool_t &pg3, typename Vec<__bf16, ARCH::NEON>::Type &in0,
        typename Vec<__bf16, ARCH::NEON>::Type &in1, typename Vec<__bf16, ARCH::NEON>::Type &in2,
        typename Vec<__bf16, ARCH::NEON>::Type &in3) noexcept
    {
        in0 = vld1q_bf16_x2(src);
        Service::Unused(pg0, pg1, in1, pg2, in2, pg3, in3);
    }
};

template <>
struct LoadN<std::int8_t, std::int8_t, 1, ARCH::NEON> final {
    static void Do(const std::int8_t *src, const svbool_t& pg0, const svbool_t &pg1,
        const svbool_t &pg2, const svbool_t &pg3, typename Vec<std::int8_t, ARCH::NEON>::Type &in0,
        typename Vec<std::int8_t, ARCH::NEON>::Type &in1, typename Vec<std::int8_t, ARCH::NEON>::Type &in2,
        typename Vec<std::int8_t, ARCH::NEON>::Type &in3) noexcept
    {
        in0 = vld1q_s8_x2(src);
        Service::Unused(pg0, pg1, in1, pg2, in2, pg3, in3);
    }
};

template <>
struct LoadN<std::uint8_t, std::uint8_t, 1, ARCH::NEON> final {
    static void Do(const std::uint8_t *src, const svbool_t& pg0, const svbool_t &pg1,
        const svbool_t &pg2, const svbool_t &pg3, typename Vec<std::uint8_t, ARCH::NEON>::Type &in0,
        typename Vec<std::uint8_t, ARCH::NEON>::Type &in1, typename Vec<std::uint8_t, ARCH::NEON>::Type &in2,
        typename Vec<std::uint8_t, ARCH::NEON>::Type &in3) noexcept
    {
        in0 = vld1q_u8_x2(src);
        Service::Unused(pg0, pg1, in1, pg2, in2, pg3, in3);
    }
};

template <>
struct LoadN<std::int32_t, std::int32_t, 1, ARCH::NEON> final {
    static void Do(const std::int32_t *src,
        const svbool_t& pg0, const svbool_t &pg1, const svbool_t &pg2, const svbool_t &pg3,
        typename Vec<std::int32_t, ARCH::NEON>::Type &in0, typename Vec<std::int32_t, ARCH::NEON>::Type &in1,
        typename Vec<std::int32_t, ARCH::NEON>::Type &in2, typename Vec<std::int32_t, ARCH::NEON>::Type &in3) noexcept
    {
        in0 = vld1q_s32_x2(src);
        Service::Unused(pg0, pg1, in1, pg2, in2, pg3, in3);
    }
};

} // Vector
} // Detail
} // KuDNN

#endif // KUDNN_VECTOR_MEM_LOAD_COMPTYPE_NEON_HPP
#endif // KUDNNL_920G