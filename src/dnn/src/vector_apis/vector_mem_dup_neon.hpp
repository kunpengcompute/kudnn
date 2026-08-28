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

#ifndef KUDNN_VECTOR_MEM_DUP_NEON_HPP
#define KUDNN_VECTOR_MEM_DUP_NEON_HPP

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
struct DupN<float, float, 1, ARCH::NEON> final {
    static void Do(const float &src,
        typename Vec<float, ARCH::NEON>::Type &in0, typename Vec<float, ARCH::NEON>::Type &in1,
        typename Vec<float, ARCH::NEON>::Type &in2, typename Vec<float, ARCH::NEON>::Type &in3) noexcept
    {
        float32x4_t srcVec = vdupq_n_f32(src);
        in0.val[0] = srcVec;
        in0.val[1] = srcVec;
        Service::Unused(in1, in2, in3);
    }
};

template <>
struct DupN<std::int32_t, std::int32_t, 1, ARCH::NEON> final {
    static void Do(const std::int32_t &src,
        typename Vec<std::int32_t, ARCH::NEON>::Type &in0,
        typename Vec<std::int32_t, ARCH::NEON>::Type &in1,
        typename Vec<std::int32_t, ARCH::NEON>::Type &in2,
        typename Vec<std::int32_t, ARCH::NEON>::Type &in3) noexcept
    {
        int32x4_t srcVec = vdupq_n_s32(src);
        in0.val[0] = srcVec;
        in0.val[1] = srcVec;
        Service::Unused(in1, in2, in3);
    }
};

template <>
struct DupN<std::int32_t, std::int32_t, 2, ARCH::NEON> final {
    static void Do(const std::int32_t &src,
        typename Vec<std::int32_t, ARCH::NEON>::Type &in0,
        typename Vec<std::int32_t, ARCH::NEON>::Type &in1,
        typename Vec<std::int32_t, ARCH::NEON>::Type &in2,
        typename Vec<std::int32_t, ARCH::NEON>::Type &in3) noexcept
    {
        int32x4_t srcVec = vdupq_n_s32(src);
        in0.val[0] = srcVec;
        in0.val[1] = srcVec;
        in1.val[0] = srcVec;
        in1.val[1] = srcVec;
        Service::Unused(in2, in3);
    }
};

// CompType is __fp16
template <>
struct DupN<__fp16, __fp16, 1, ARCH::NEON> final {
    static void Do(const __fp16 &src,
        typename Vec<__fp16, ARCH::NEON>::Type &in0, typename Vec<__fp16, ARCH::NEON>::Type &in1,
        typename Vec<__fp16, ARCH::NEON>::Type &in2, typename Vec<__fp16, ARCH::NEON>::Type &in3) noexcept
    {
        float16x8_t srcVec = vdupq_n_f16(src);
        in0.val[0] = srcVec;
        in0.val[1] = srcVec;
        Service::Unused(in1, in2, in3);
    }
};


} // Vector
} // Detail
} // KuDNN

#endif // KUDNNL_920G

#endif // KUDNN_VECTOR_MEM_DUP_HPP_NEON