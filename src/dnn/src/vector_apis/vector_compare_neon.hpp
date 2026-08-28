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

#ifndef KUDNN_VECTOR_COMPARE_NEON
#define KUDNN_VECTOR_COMPARE_NEON

#ifdef KUDNNL_920G

#include <arm_sve.h>

#include "internal_funcs.hpp"

namespace KuDNN {
namespace Detail {
namespace Vector {


template <typename ComputeType, SizeType N>
struct GeN<ComputeType, N, ARCH::NEON> final {
    static inline void Compare(const typename Vec<float, ARCH::NEON>::Type &lhs,
        const typename Vec<float, ARCH::NEON>::Type &rhs,
        typename MaskVec<float, ARCH::NEON>::Type &out) noexcept
    {
        out.val[0] = vcgeq_f32(lhs.val[0], rhs.val[0]);
        out.val[1] = vcgeq_f32(lhs.val[1], rhs.val[1]);
    }
    static inline void Compare(const typename Vec<float16_t, ARCH::NEON>::Type &lhs,
        const typename Vec<float16_t, ARCH::NEON>::Type &rhs,
        typename MaskVec<float16_t, ARCH::NEON>::Type &out) noexcept
    {
        out.val[0] = vcgeq_f16(lhs.val[0], rhs.val[0]);
        out.val[1] = vcgeq_f16(lhs.val[1], rhs.val[1]);
    }
    static void
    Do(const svbool_t &pg0, const svbool_t &pg1, const svbool_t &pg2,
    const svbool_t &pg3,
    const typename Vec<ComputeType, ARCH::NEON>::Type &lhs0,
    const typename Vec<ComputeType, ARCH::NEON>::Type &lhs1,
    const typename Vec<ComputeType, ARCH::NEON>::Type &lhs2,
    const typename Vec<ComputeType, ARCH::NEON>::Type &lhs3,
    const typename Vec<ComputeType, ARCH::NEON>::Type &rhs0,
    const typename Vec<ComputeType, ARCH::NEON>::Type &rhs1,
    const typename Vec<ComputeType, ARCH::NEON>::Type &rhs2,
    const typename Vec<ComputeType, ARCH::NEON>::Type &rhs3,
    typename MaskVec<ComputeType, ARCH::NEON>::Type &out0,
    typename MaskVec<ComputeType, ARCH::NEON>::Type &out1,
    typename MaskVec<ComputeType, ARCH::NEON>::Type &out2,
    typename MaskVec<ComputeType, ARCH::NEON>::Type &out3) noexcept
    {
        static_assert(N <= 4, "Pseudo unroll by 4 is maximum");
        Compare(lhs0, rhs0, out0);
        if constexpr (N > 1) {
            Compare(lhs1, rhs1, out1);
            if constexpr (N > 2) {
                Compare(lhs2, rhs2, out2);
                if constexpr (N > 3) {
                    Compare(lhs3, rhs3, out3);
                }
            }
        }
        Service::Unused(pg0, pg1, pg2, pg3);
    }
};

template <typename ComputeType, SizeType N>
struct GtN<ComputeType, N, ARCH::NEON> final {
    static inline void Compare(const typename Vec<float, ARCH::NEON>::Type &lhs,
        const typename Vec<float, ARCH::NEON>::Type &rhs,
        typename MaskVec<float, ARCH::NEON>::Type &out) noexcept
    {
        out.val[0] = vcgtq_f32(lhs.val[0], rhs.val[0]);
        out.val[1] = vcgtq_f32(lhs.val[1], rhs.val[1]);
    }
    static inline void Compare(const typename Vec<float16_t, ARCH::NEON>::Type &lhs,
        const typename Vec<float16_t, ARCH::NEON>::Type &rhs,
        typename MaskVec<float16_t, ARCH::NEON>::Type &out) noexcept
    {
        out.val[0] = vcgtq_f16(lhs.val[0], rhs.val[0]);
        out.val[1] = vcgtq_f16(lhs.val[1], rhs.val[1]);
    }
    static void
    Do(const svbool_t &pg0, const svbool_t &pg1, const svbool_t &pg2,
    const svbool_t &pg3,
    const typename Vec<ComputeType, ARCH::NEON>::Type &lhs0,
    const typename Vec<ComputeType, ARCH::NEON>::Type &lhs1,
    const typename Vec<ComputeType, ARCH::NEON>::Type &lhs2,
    const typename Vec<ComputeType, ARCH::NEON>::Type &lhs3,
    const typename Vec<ComputeType, ARCH::NEON>::Type &rhs0,
    const typename Vec<ComputeType, ARCH::NEON>::Type &rhs1,
    const typename Vec<ComputeType, ARCH::NEON>::Type &rhs2,
    const typename Vec<ComputeType, ARCH::NEON>::Type &rhs3,
    typename MaskVec<ComputeType, ARCH::NEON>::Type &out0,
    typename MaskVec<ComputeType, ARCH::NEON>::Type &out1,
    typename MaskVec<ComputeType, ARCH::NEON>::Type &out2,
    typename MaskVec<ComputeType, ARCH::NEON>::Type &out3) noexcept
    {
        static_assert(N <= 4, "Pseudo unroll by 4 is maximum");
        Compare(lhs0, rhs0, out0);
        if constexpr (N > 1) {
            Compare(lhs1, rhs1, out1);
            if constexpr (N > 2) {
                Compare(lhs2, rhs2, out2);
                if constexpr (N > 3) {
                    Compare(lhs3, rhs3, out3);
                }
            }
        }
        Service::Unused(pg0, pg1, pg2, pg3);
    }
};

template <typename ComputeType, SizeType N>
struct LeN<ComputeType, N, ARCH::NEON> final {
    static inline void Compare(const typename Vec<float, ARCH::NEON>::Type &lhs,
        const typename Vec<float, ARCH::NEON>::Type &rhs,
        typename MaskVec<float, ARCH::NEON>::Type &out) noexcept
    {
        out.val[0] = vcleq_f32(lhs.val[0], rhs.val[0]);
        out.val[1] = vcleq_f32(lhs.val[1], rhs.val[1]);
    }
    static inline void Compare(const typename Vec<float16_t, ARCH::NEON>::Type &lhs,
        const typename Vec<float16_t, ARCH::NEON>::Type &rhs,
        typename MaskVec<float16_t, ARCH::NEON>::Type &out) noexcept
    {
        out.val[0] = vcleq_f16(lhs.val[0], rhs.val[0]);
        out.val[1] = vcleq_f16(lhs.val[1], rhs.val[1]);
    }
    static void
    Do(const svbool_t &pg0, const svbool_t &pg1, const svbool_t &pg2,
    const svbool_t &pg3,
    const typename Vec<ComputeType, ARCH::NEON>::Type &lhs0,
    const typename Vec<ComputeType, ARCH::NEON>::Type &lhs1,
    const typename Vec<ComputeType, ARCH::NEON>::Type &lhs2,
    const typename Vec<ComputeType, ARCH::NEON>::Type &lhs3,
    const typename Vec<ComputeType, ARCH::NEON>::Type &rhs0,
    const typename Vec<ComputeType, ARCH::NEON>::Type &rhs1,
    const typename Vec<ComputeType, ARCH::NEON>::Type &rhs2,
    const typename Vec<ComputeType, ARCH::NEON>::Type &rhs3,
    typename MaskVec<ComputeType, ARCH::NEON>::Type &out0,
    typename MaskVec<ComputeType, ARCH::NEON>::Type &out1,
    typename MaskVec<ComputeType, ARCH::NEON>::Type &out2,
    typename MaskVec<ComputeType, ARCH::NEON>::Type &out3) noexcept
    {
        static_assert(N <= 4, "Pseudo unroll by 4 is maximum");
        Compare(lhs0, rhs0, out0);
        if constexpr (N > 1) {
            Compare(lhs1, rhs1, out1);
            if constexpr (N > 2) {
                Compare(lhs2, rhs2, out2);
                if constexpr (N > 3) {
                    Compare(lhs3, rhs3, out3);
                }
            }
        }
        Service::Unused(pg0, pg1, pg2, pg3);
    }
};

template <typename ComputeType, SizeType N>
struct LtN<ComputeType, N, ARCH::NEON> final {
    static inline void Compare(const typename Vec<float, ARCH::NEON>::Type &lhs,
        const typename Vec<float, ARCH::NEON>::Type &rhs,
        typename MaskVec<float, ARCH::NEON>::Type &out) noexcept
    {
        out.val[0] = vcltq_f32(lhs.val[0], rhs.val[0]);
        out.val[1] = vcltq_f32(lhs.val[1], rhs.val[1]);
    }
    static inline void Compare(const typename Vec<float16_t, ARCH::NEON>::Type &lhs,
        const typename Vec<float16_t, ARCH::NEON>::Type &rhs,
        typename MaskVec<float16_t, ARCH::NEON>::Type &out) noexcept
    {
        out.val[0] = vcltq_f16(lhs.val[0], rhs.val[0]);
        out.val[1] = vcltq_f16(lhs.val[1], rhs.val[1]);
    }
    static void
    Do(const svbool_t &pg0, const svbool_t &pg1, const svbool_t &pg2,
    const svbool_t &pg3,
    const typename Vec<ComputeType, ARCH::NEON>::Type &lhs0,
    const typename Vec<ComputeType, ARCH::NEON>::Type &lhs1,
    const typename Vec<ComputeType, ARCH::NEON>::Type &lhs2,
    const typename Vec<ComputeType, ARCH::NEON>::Type &lhs3,
    const typename Vec<ComputeType, ARCH::NEON>::Type &rhs0,
    const typename Vec<ComputeType, ARCH::NEON>::Type &rhs1,
    const typename Vec<ComputeType, ARCH::NEON>::Type &rhs2,
    const typename Vec<ComputeType, ARCH::NEON>::Type &rhs3,
    typename MaskVec<ComputeType, ARCH::NEON>::Type &out0,
    typename MaskVec<ComputeType, ARCH::NEON>::Type &out1,
    typename MaskVec<ComputeType, ARCH::NEON>::Type &out2,
    typename MaskVec<ComputeType, ARCH::NEON>::Type &out3) noexcept
    {
        static_assert(N <= 4, "Pseudo unroll by 4 is maximum");
        Compare(lhs0, rhs0, out0);
        if constexpr (N > 1) {
            Compare(lhs1, rhs1, out1);
            if constexpr (N > 2) {
                Compare(lhs2, rhs2, out2);
                if constexpr (N > 3) {
                    Compare(lhs3, rhs3, out3);
                }
            }
        }
        Service::Unused(pg0, pg1, pg2, pg3);
    }
};

template <typename ComputeType, SizeType N>
struct EqN<ComputeType, N, ARCH::NEON> final {
    static inline void Compare(const typename Vec<float, ARCH::NEON>::Type &lhs,
        const typename Vec<float, ARCH::NEON>::Type &rhs,
        typename MaskVec<float, ARCH::NEON>::Type &out) noexcept
    {
        out.val[0] = vceqq_f32(lhs.val[0], rhs.val[0]);
        out.val[1] = vceqq_f32(lhs.val[1], rhs.val[1]);
    }
    static inline void Compare(const typename Vec<float16_t, ARCH::NEON>::Type &lhs,
        const typename Vec<float16_t, ARCH::NEON>::Type &rhs,
        typename MaskVec<float16_t, ARCH::NEON>::Type &out) noexcept
    {
        out.val[0] = vceqq_f16(lhs.val[0], rhs.val[0]);
        out.val[1] = vceqq_f16(lhs.val[1], rhs.val[1]);
    }
    static void
    Do(const svbool_t &pg0, const svbool_t &pg1, const svbool_t &pg2,
    const svbool_t &pg3,
    const typename Vec<ComputeType, ARCH::NEON>::Type &lhs0,
    const typename Vec<ComputeType, ARCH::NEON>::Type &lhs1,
    const typename Vec<ComputeType, ARCH::NEON>::Type &lhs2,
    const typename Vec<ComputeType, ARCH::NEON>::Type &lhs3,
    const typename Vec<ComputeType, ARCH::NEON>::Type &rhs0,
    const typename Vec<ComputeType, ARCH::NEON>::Type &rhs1,
    const typename Vec<ComputeType, ARCH::NEON>::Type &rhs2,
    const typename Vec<ComputeType, ARCH::NEON>::Type &rhs3,
    typename MaskVec<ComputeType, ARCH::NEON>::Type &out0,
    typename MaskVec<ComputeType, ARCH::NEON>::Type &out1,
    typename MaskVec<ComputeType, ARCH::NEON>::Type &out2,
    typename MaskVec<ComputeType, ARCH::NEON>::Type &out3) noexcept
    {
        static_assert(N <= 4, "Pseudo unroll by 4 is maximum");
        Compare(lhs0, rhs0, out0);
        if constexpr (N > 1) {
            Compare(lhs1, rhs1, out1);
            if constexpr (N > 2) {
                Compare(lhs2, rhs2, out2);
                if constexpr (N > 3) {
                    Compare(lhs3, rhs3, out3);
                }
            }
        }
        Service::Unused(pg0, pg1, pg2, pg3);
    }
};

template <typename ComputeType, SizeType N>
struct NeN<ComputeType, N, ARCH::NEON> final {
    static inline void Compare(const typename Vec<float, ARCH::NEON>::Type &lhs,
        const typename Vec<float, ARCH::NEON>::Type &rhs,
        typename MaskVec<float, ARCH::NEON>::Type &out) noexcept
    {
        out.val[0] = veorq_u32(vceqq_f32(lhs.val[0], rhs.val[0]), vdupq_n_u32(0xFFFFFFFF));
        out.val[1] = veorq_u32(vceqq_f32(lhs.val[1], rhs.val[1]), vdupq_n_u32(0xFFFFFFFF));
    }
    static inline void Compare(const typename Vec<float16_t, ARCH::NEON>::Type &lhs,
        const typename Vec<float16_t, ARCH::NEON>::Type &rhs,
        typename MaskVec<float16_t, ARCH::NEON>::Type &out) noexcept
    {
        out.val[0] = veorq_u16(vceqq_f16(lhs.val[0], rhs.val[0]), vdupq_n_u16(0xFFFF));
        out.val[1] = veorq_u16(vceqq_f16(lhs.val[1], rhs.val[1]), vdupq_n_u16(0xFFFF));
    }
    static void
    Do(const svbool_t &pg0, const svbool_t &pg1, const svbool_t &pg2,
    const svbool_t &pg3,
    const typename Vec<ComputeType, ARCH::NEON>::Type &lhs0,
    const typename Vec<ComputeType, ARCH::NEON>::Type &lhs1,
    const typename Vec<ComputeType, ARCH::NEON>::Type &lhs2,
    const typename Vec<ComputeType, ARCH::NEON>::Type &lhs3,
    const typename Vec<ComputeType, ARCH::NEON>::Type &rhs0,
    const typename Vec<ComputeType, ARCH::NEON>::Type &rhs1,
    const typename Vec<ComputeType, ARCH::NEON>::Type &rhs2,
    const typename Vec<ComputeType, ARCH::NEON>::Type &rhs3,
    typename MaskVec<ComputeType, ARCH::NEON>::Type &out0,
    typename MaskVec<ComputeType, ARCH::NEON>::Type &out1,
    typename MaskVec<ComputeType, ARCH::NEON>::Type &out2,
    typename MaskVec<ComputeType, ARCH::NEON>::Type &out3) noexcept
    {
        static_assert(N <= 4, "Pseudo unroll by 4 is maximum");
        Compare(lhs0, rhs0, out0);
        if constexpr (N > 1) {
            Compare(lhs1, rhs1, out1);
            if constexpr (N > 2) {
                Compare(lhs2, rhs2, out2);
                if constexpr (N > 3) {
                    Compare(lhs3, rhs3, out3);
                }
            }
        }
        Service::Unused(pg0, pg1, pg2, pg3);
    }
};

template <typename ComputeType, SizeType N>
struct MaxXN<ComputeType, N, ARCH::NEON> final {
    static inline void Compare(const typename Vec<float, ARCH::NEON>::Type &lhs,
        const typename Vec<float, ARCH::NEON>::Type &rhs,
        typename Vec<float, ARCH::NEON>::Type &out) noexcept
    {
        out.val[0] = vmaxq_f32(lhs.val[0], rhs.val[0]);
        out.val[1] = vmaxq_f32(lhs.val[1], rhs.val[1]);
    }
    static inline void Compare(const typename Vec<float16_t, ARCH::NEON>::Type &lhs,
        const typename Vec<float16_t, ARCH::NEON>::Type &rhs,
        typename Vec<float16_t, ARCH::NEON>::Type &out) noexcept
    {
        out.val[0] = vmaxq_f16(lhs.val[0], rhs.val[0]);
        out.val[1] = vmaxq_f16(lhs.val[1], rhs.val[1]);
    }
    static void
    Do(const svbool_t &pg0, const svbool_t &pg1, const svbool_t &pg2,
    const svbool_t &pg3,
    const typename Vec<ComputeType, ARCH::NEON>::Type &lhs0,
    const typename Vec<ComputeType, ARCH::NEON>::Type &lhs1,
    const typename Vec<ComputeType, ARCH::NEON>::Type &lhs2,
    const typename Vec<ComputeType, ARCH::NEON>::Type &lhs3,
    const typename Vec<ComputeType, ARCH::NEON>::Type &rhs0,
    const typename Vec<ComputeType, ARCH::NEON>::Type &rhs1,
    const typename Vec<ComputeType, ARCH::NEON>::Type &rhs2,
    const typename Vec<ComputeType, ARCH::NEON>::Type &rhs3,
    typename Vec<ComputeType, ARCH::NEON>::Type &out0,
    typename Vec<ComputeType, ARCH::NEON>::Type &out1,
    typename Vec<ComputeType, ARCH::NEON>::Type &out2,
    typename Vec<ComputeType, ARCH::NEON>::Type &out3) noexcept
    {
        static_assert(N <= 4, "Pseudo unroll by 4 is maximum");
        Compare(lhs0, rhs0, out0);
        if constexpr (N > 1) {
            Compare(lhs1, rhs1, out1);
            if constexpr (N > 2) {
                Compare(lhs2, rhs2, out2);
                if constexpr (N > 3) {
                    Compare(lhs3, rhs3, out3);
                }
            }
        }
        Service::Unused(pg0, pg1, pg2, pg3);
    }
};

template <typename ComputeType, SizeType N>
struct MaxMN<ComputeType, N, ARCH::NEON> final {
    static void Do(const svbool_t& pg0, const svbool_t &pg1,
        const svbool_t &pg2, const svbool_t &pg3,
        const typename Vec<ComputeType, ARCH::NEON>::Type &lhs0,
        const typename Vec<ComputeType, ARCH::NEON>::Type &lhs1,
        const typename Vec<ComputeType, ARCH::NEON>::Type &lhs2,
        const typename Vec<ComputeType, ARCH::NEON>::Type &lhs3,
        const typename Vec<ComputeType, ARCH::NEON>::Type &rhs0,
        const typename Vec<ComputeType, ARCH::NEON>::Type &rhs1,
        const typename Vec<ComputeType, ARCH::NEON>::Type &rhs2,
        const typename Vec<ComputeType, ARCH::NEON>::Type &rhs3,
        typename Vec<ComputeType, ARCH::NEON>::Type &out0,
        typename Vec<ComputeType, ARCH::NEON>::Type &out1,
        typename Vec<ComputeType, ARCH::NEON>::Type &out2,
        typename Vec<ComputeType, ARCH::NEON>::Type &out3) noexcept
    {
        Service::Unused(pg0, pg1, pg2, pg3);
        static_assert(N <= 4, "Pseudo unroll by 4 is maximum");
        out0.val[0] = vmaxq_f32(lhs0.val[0], rhs0.val[0]);
        out0.val[1] = vmaxq_f32(lhs0.val[1], rhs0.val[1]);
        if constexpr (N > 1) {
            out1.val[0] = vmaxq_f32(lhs1.val[0], rhs1.val[0]);
            out1.val[1] = vmaxq_f32(lhs1.val[1], rhs1.val[1]);
            if constexpr (N > 2) {
                out2.val[0] = vmaxq_f32(lhs2.val[0], rhs2.val[0]);
                out2.val[1] = vmaxq_f32(lhs2.val[1], rhs2.val[1]);
                if constexpr (N > 3) {
                    out3.val[0] = vmaxq_f32(lhs3.val[0], rhs3.val[0]);
                    out3.val[1] = vmaxq_f32(lhs3.val[1], rhs3.val[1]);
                }
            }
        }
    }
};

template <typename ComputeType, SizeType N>
struct MinXN<ComputeType, N, ARCH::NEON> final {
    static inline void Compare(const typename Vec<float, ARCH::NEON>::Type &lhs,
        const typename Vec<float, ARCH::NEON>::Type &rhs,
        typename Vec<float, ARCH::NEON>::Type &out) noexcept
    {
        out.val[0] = vminq_f32(lhs.val[0], rhs.val[0]);
        out.val[1] = vminq_f32(lhs.val[1], rhs.val[1]);
    }
    static inline void Compare(const typename Vec<float16_t, ARCH::NEON>::Type &lhs,
        const typename Vec<float16_t, ARCH::NEON>::Type &rhs,
        typename Vec<float16_t, ARCH::NEON>::Type &out) noexcept
    {
        out.val[0] = vminq_f16(lhs.val[0], rhs.val[0]);
        out.val[1] = vminq_f16(lhs.val[1], rhs.val[1]);
    }
    static void
    Do(const svbool_t &pg0, const svbool_t &pg1, const svbool_t &pg2,
    const svbool_t &pg3,
    const typename Vec<ComputeType, ARCH::NEON>::Type &lhs0,
    const typename Vec<ComputeType, ARCH::NEON>::Type &lhs1,
    const typename Vec<ComputeType, ARCH::NEON>::Type &lhs2,
    const typename Vec<ComputeType, ARCH::NEON>::Type &lhs3,
    const typename Vec<ComputeType, ARCH::NEON>::Type &rhs0,
    const typename Vec<ComputeType, ARCH::NEON>::Type &rhs1,
    const typename Vec<ComputeType, ARCH::NEON>::Type &rhs2,
    const typename Vec<ComputeType, ARCH::NEON>::Type &rhs3,
    typename Vec<ComputeType, ARCH::NEON>::Type &out0,
    typename Vec<ComputeType, ARCH::NEON>::Type &out1,
    typename Vec<ComputeType, ARCH::NEON>::Type &out2,
    typename Vec<ComputeType, ARCH::NEON>::Type &out3) noexcept
    {
        static_assert(N <= 4, "Pseudo unroll by 4 is maximum");
        Compare(lhs0, rhs0, out0);
        if constexpr (N > 1) {
            Compare(lhs1, rhs1, out1);
            if constexpr (N > 2) {
                Compare(lhs2, rhs2, out2);
                if constexpr (N > 3) {
                    Compare(lhs3, rhs3, out3);
                }
            }
        }
        Service::Unused(pg0, pg1, pg2, pg3);
    }
};

template <typename ComputeType, SizeType N>
struct MinMN<ComputeType, N, ARCH::NEON> final {
    static void Do(const svbool_t& pg0, const svbool_t &pg1,
        const svbool_t &pg2, const svbool_t &pg3,
        const typename Vec<ComputeType, ARCH::NEON>::Type &lhs0,
        const typename Vec<ComputeType, ARCH::NEON>::Type &lhs1,
        const typename Vec<ComputeType, ARCH::NEON>::Type &lhs2,
        const typename Vec<ComputeType, ARCH::NEON>::Type &lhs3,
        const typename Vec<ComputeType, ARCH::NEON>::Type &rhs0,
        const typename Vec<ComputeType, ARCH::NEON>::Type &rhs1,
        const typename Vec<ComputeType, ARCH::NEON>::Type &rhs2,
        const typename Vec<ComputeType, ARCH::NEON>::Type &rhs3,
        typename Vec<ComputeType, ARCH::NEON>::Type &out0,
        typename Vec<ComputeType, ARCH::NEON>::Type &out1,
        typename Vec<ComputeType, ARCH::NEON>::Type &out2,
        typename Vec<ComputeType, ARCH::NEON>::Type &out3) noexcept
    {
        Service::Unused(pg0, pg1, pg2, pg3);
        static_assert(N <= 4, "Pseudo unroll by 4 is maximum");
        out0.val[0] = vminq_f32(lhs0.val[0], rhs0.val[0]);
        out0.val[1] = vminq_f32(lhs0.val[1], rhs0.val[1]);
        if constexpr (N > 1) {
            out1.val[0] = vminq_f32(lhs1.val[0], rhs1.val[0]);
            out1.val[1] = vminq_f32(lhs1.val[1], rhs1.val[1]);
            if constexpr (N > 2) {
                out2.val[0] = vminq_f32(lhs2.val[0], rhs2.val[0]);
                out2.val[1] = vminq_f32(lhs2.val[1], rhs2.val[1]);
                if constexpr (N > 3) {
                    out3.val[0] = vminq_f32(lhs3.val[0], rhs3.val[0]);
                    out3.val[1] = vminq_f32(lhs3.val[1], rhs3.val[1]);
                }
            }
        }
    }
};

} // namespace Vector
} // namespace Detail
} // namespace KuDNN

#endif // KUDNNL_920G
#endif // KUDNN_VECTOR_COMPARE