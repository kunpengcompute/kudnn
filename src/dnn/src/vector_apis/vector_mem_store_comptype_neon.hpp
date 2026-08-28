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

#ifndef KUDNN_VECTOR_MEM_STORE_COMPTYPE_NEON_HPP
#define KUDNN_VECTOR_MEM_STORE_COMPTYPE_NEON_HPP

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
struct StoreN<float, float, 1, ARCH::NEON> final {
    static void Do(const svbool_t& pg0, const svbool_t &pg1, const svbool_t &pg2, const svbool_t &pg3,
        const typename Vec<float, ARCH::NEON>::Type &out0,
        const typename Vec<float, ARCH::NEON>::Type &out1,
        const typename Vec<float, ARCH::NEON>::Type &out2,
        const typename Vec<float, ARCH::NEON>::Type &out3, float *dst) noexcept
    {
        vst1q_f32_x2(dst, out0);
        Service::Unused(pg0, pg1, out1, pg2, out2, pg3, out3);
    }
};

template <>
struct StoreN<float, float, 4, ARCH::NEON> final {
    static void Do(const svbool_t& pg0, const svbool_t &pg1, const svbool_t &pg2, const svbool_t &pg3,
        const typename Vec<float, ARCH::NEON>::Type &out0,
        const typename Vec<float, ARCH::NEON>::Type &out1,
        const typename Vec<float, ARCH::NEON>::Type &out2,
        const typename Vec<float, ARCH::NEON>::Type &out3, float *dst) noexcept
    {
        vst1q_f32_x2(dst, out0);
        vst1q_f32_x2(dst + 8, out1);
        vst1q_f32_x2(dst + 16, out2);
        vst1q_f32_x2(dst + 24, out3);
        Service::Unused(pg0, pg1, out1, pg2, out2, pg3, out3);
    }
};

template <>
struct StoreN<__bf16, float, 1, ARCH::NEON> final {
    static void Do(const svbool_t& pg0, const svbool_t &pg1, const svbool_t &pg2, const svbool_t &pg3,
        const typename Vec<float, ARCH::NEON>::Type &out0,
        const typename Vec<float, ARCH::NEON>::Type &out1,
        const typename Vec<float, ARCH::NEON>::Type &out2,
        const typename Vec<float, ARCH::NEON>::Type &out3, __bf16 *dst) noexcept
    {
        vst1q_bf16(dst, vcombine_bf16(vcvt_bf16_f32(out0.val[0]), vcvt_bf16_f32(out0.val[1])));
        vst1q_bf16(dst + Vector::NeonLanes<__bf16>(), vcombine_bf16(vcvt_bf16_f32(out1.val[0]),
            vcvt_bf16_f32(out1.val[1])));

        Service::Unused(pg0, pg1, out1, pg2, out2, pg3, out3);
    }
};

template <>
struct StoreN<__fp16, float, 1, ARCH::NEON> final {
    static void Do(const svbool_t& pg0, const svbool_t &pg1, const svbool_t &pg2, const svbool_t &pg3,
        const typename Vec<float, ARCH::NEON>::Type &out0,
        const typename Vec<float, ARCH::NEON>::Type &out1,
        const typename Vec<float, ARCH::NEON>::Type &out2,
        const typename Vec<float, ARCH::NEON>::Type &out3, __fp16 *dst) noexcept
    {
        vst1q_f16(dst, vcombine_f16(vcvt_f16_f32(out0.val[0]), vcvt_f16_f32(out0.val[1])));
        vst1q_f16(dst + Vector::NeonLanes<__fp16>(), vcombine_f16(vcvt_f16_f32(out1.val[0]),
            vcvt_f16_f32(out1.val[1])));

        Service::Unused(pg0, pg1, out1, pg2, out2, pg3, out3);
    }
};

template <>
struct StoreN<__fp16, __fp16, 1, ARCH::NEON> final {
    static void Do(const svbool_t& pg0, const svbool_t &pg1, const svbool_t &pg2, const svbool_t &pg3,
        const typename Vec<__fp16, ARCH::NEON>::Type &out0,
        const typename Vec<__fp16, ARCH::NEON>::Type &out1,
        const typename Vec<__fp16, ARCH::NEON>::Type &out2,
        const typename Vec<__fp16, ARCH::NEON>::Type &out3, __fp16 *dst) noexcept
    {
        vst1q_f16_x2(dst, out0);
        Service::Unused(pg0, pg1, out1, pg2, out2, pg3, out3);
    }
};

template <>
struct StoreN<__bf16, __bf16, 1, ARCH::NEON> final {
    static void Do(const svbool_t& pg0, const svbool_t &pg1, const svbool_t &pg2, const svbool_t &pg3,
        const typename Vec<__bf16, ARCH::NEON>::Type &out0,
        const typename Vec<__bf16, ARCH::NEON>::Type &out1,
        const typename Vec<__bf16, ARCH::NEON>::Type &out2,
        const typename Vec<__bf16, ARCH::NEON>::Type &out3, __bf16 *dst) noexcept
    {
        vst1q_bf16_x2(dst, out0);
        Service::Unused(pg0, pg1, out1, pg2, out2, pg3, out3);
    }
};

template <>
struct StoreN<std::int8_t, std::int8_t, 1, ARCH::NEON> final {
    static void Do(const svbool_t& pg0, const svbool_t &pg1, const svbool_t &pg2, const svbool_t &pg3,
        const typename Vec<std::int8_t, ARCH::NEON>::Type &out0,
        const typename Vec<std::int8_t, ARCH::NEON>::Type &out1,
        const typename Vec<std::int8_t, ARCH::NEON>::Type &out2,
        const typename Vec<std::int8_t, ARCH::NEON>::Type &out3, std::int8_t *dst) noexcept
    {
        vst1q_s8_x2(dst, out0);
        Service::Unused(pg0, pg1, out1, pg2, out2, pg3, out3);
    }
};

template <>
struct StoreN<std::uint8_t, std::uint8_t, 1, ARCH::NEON> final {
    static void Do(const svbool_t& pg0, const svbool_t &pg1, const svbool_t &pg2, const svbool_t &pg3,
        const typename Vec<std::uint8_t, ARCH::NEON>::Type &out0,
        const typename Vec<std::uint8_t, ARCH::NEON>::Type &out1,
        const typename Vec<std::uint8_t, ARCH::NEON>::Type &out2,
        const typename Vec<std::uint8_t, ARCH::NEON>::Type &out3, std::uint8_t *dst) noexcept
    {
        vst1q_u8_x2(dst, out0);
        Service::Unused(pg0, pg1, out1, pg2, out2, pg3, out3);
    }
};

template <>
struct StoreN<std::int32_t, std::int32_t, 1, ARCH::NEON> final {
    static void Do(const svbool_t& pg0, const svbool_t &pg1, const svbool_t &pg2, const svbool_t &pg3,
        const typename Vec<std::int32_t, ARCH::NEON>::Type &out0,
        const typename Vec<std::int32_t, ARCH::NEON>::Type &out1,
        const typename Vec<std::int32_t, ARCH::NEON>::Type &out2,
        const typename Vec<std::int32_t, ARCH::NEON>::Type &out3, std::int32_t *dst) noexcept
    {
        vst1q_s32_x2(dst, out0);
        Service::Unused(pg0, pg1, out1, pg2, out2, pg3, out3);
    }
};

template <>
struct StoreN<std::int32_t, std::int32_t, 2, ARCH::NEON> final {
    static void Do(const svbool_t& pg0, const svbool_t &pg1, const svbool_t &pg2, const svbool_t &pg3,
        const typename Vec<std::int32_t, ARCH::NEON>::Type &out0,
        const typename Vec<std::int32_t, ARCH::NEON>::Type &out1,
        const typename Vec<std::int32_t, ARCH::NEON>::Type &out2,
        const typename Vec<std::int32_t, ARCH::NEON>::Type &out3, std::int32_t *dst) noexcept
    {
        vst1q_s32_x2(dst, out0);
        vst1q_s32_x2(dst + 8, out1);
        Service::Unused(pg0, pg1, out1, pg2, out2, pg3, out3);
    }
};

} // Vector
} // Detail
} // KuDNN

#endif // KUDNN_VECTOR_MEM_STORE_COMPTYPE_NEON_HPP
#endif // KUDNNL_920G