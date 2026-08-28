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

#ifndef KUDNN_VECTOR_MEM_DUPFLOAT_NEON_HPP
#define KUDNN_VECTOR_MEM_DUPFLOAT_NEON_HPP

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

template <>
struct DupFloatN<float, float, 1, ARCH::NEON> final {
    static void Do(const float &src,
        typename Vec<float, ARCH::NEON>::Type &in0, typename Vec<float, ARCH::NEON>::Type &in1,
        typename Vec<float, ARCH::NEON>::Type &in2, typename Vec<float, ARCH::NEON>::Type &in3) noexcept
    {
        in0.val[0] = vdupq_n_f32(src);
        in0.val[1] = vdupq_n_f32(src);
        Service::Unused(in1, in2, in3);
    }
};

template <>
struct DupFloatN<__fp16, float, 1, ARCH::NEON> final {
    static void Do(const float &src,
        typename Vec<float, ARCH::NEON>::Type &in0, typename Vec<float, ARCH::NEON>::Type &in1,
        typename Vec<float, ARCH::NEON>::Type &in2, typename Vec<float, ARCH::NEON>::Type &in3) noexcept
    {
        auto val = vdupq_n_f32(src);
        in0.val[0] = val;
        in0.val[1] = val;
        in1.val[0] = val;
        in1.val[1] = val;
        Service::Unused(in2, in3);
    }
};

template <>
struct DupFloatN<__fp16, float, 2, ARCH::NEON> final {
    static void Do(const float &src,
        typename Vec<float, ARCH::NEON>::Type &in0, typename Vec<float, ARCH::NEON>::Type &in1,
        typename Vec<float, ARCH::NEON>::Type &in2, typename Vec<float, ARCH::NEON>::Type &in3) noexcept
    {
        auto val = vdupq_n_f32(src);
        in0.val[0] = val;
        in0.val[1] = val;
        in1.val[0] = val;
        in1.val[1] = val;
        in2.val[0] = val;
        in2.val[1] = val;
        in3.val[0] = val;
        in3.val[1] = val;
    }
};


template <>
struct DupFloatN<__bf16, float, 1, ARCH::NEON> final {
    static void Do(const float &src,
        typename Vec<float, ARCH::NEON>::Type &in0, typename Vec<float, ARCH::NEON>::Type &in1,
        typename Vec<float, ARCH::NEON>::Type &in2, typename Vec<float, ARCH::NEON>::Type &in3) noexcept
    {
        auto val = vdupq_n_f32(src);
        in0.val[0] = val;
        in0.val[1] = val;
        in1.val[0] = val;
        in1.val[1] = val;
        Service::Unused(in2, in3);
    }
};

template <>
struct DupFloatN<__bf16, float, 2, ARCH::NEON> final {
    static void Do(const float &src,
        typename Vec<float, ARCH::NEON>::Type &in0, typename Vec<float, ARCH::NEON>::Type &in1,
        typename Vec<float, ARCH::NEON>::Type &in2, typename Vec<float, ARCH::NEON>::Type &in3) noexcept
    {
        auto val = vdupq_n_f32(src);
        in0.val[0] = val;
        in0.val[1] = val;
        in1.val[0] = val;
        in1.val[1] = val;
        in2.val[0] = val;
        in2.val[1] = val;
        in3.val[0] = val;
        in3.val[1] = val;
    }
};

template <>
struct DupFloatN<__fp16, __fp16, 1, ARCH::NEON> final {
    static void Do(const float &src,
        typename Vec<__fp16, ARCH::NEON>::Type &in0,
        typename Vec<__fp16, ARCH::NEON>::Type &in1,
        typename Vec<__fp16, ARCH::NEON>::Type &in2,
        typename Vec<__fp16, ARCH::NEON>::Type &in3) noexcept
    {
        in0.val[0] = vdupq_n_f16(src);
        in0.val[1] = vdupq_n_f16(src);
        Service::Unused(in1, in2, in3);
    }
};

} // Vector
} // Detail
} // KuDNN

#endif // KUDNNL_920G
#endif // KUDNN_VECTOR_MEM_DUPFLOAT_HPP_NEON