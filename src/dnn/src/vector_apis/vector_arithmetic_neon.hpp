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
#ifndef KUDNN_VECTOR_ARITHMETIC_NEON_HPP
#define KUDNN_VECTOR_ARITHMETIC_NEON_HPP

#ifdef KUDNNL_920G

#include <arm_sve.h>
#include <arm_neon.h>

#include "internal_funcs.hpp"
#include "vector_arithmetic.hpp"

namespace KuDNN {
namespace Detail {
namespace Vector {

template <typename ComputeType, SizeType N>
struct AddXN<ComputeType, N, ARCH::NEON> final {
    using InputType = typename Vector::Vec<ComputeType, ARCH::NEON>::Type;
    using OutType = typename Vector::Vec<ComputeType, ARCH::NEON>::Type;
    static inline void Operation(const typename Vec<float, ARCH::NEON>::Type &lhs,
        const typename Vec<float, ARCH::NEON>::Type &rhs,
        typename Vec<float, ARCH::NEON>::Type &out) noexcept
    {
        out.val[0] = vaddq_f32(lhs.val[0], rhs.val[0]);
        out.val[1] = vaddq_f32(lhs.val[1], rhs.val[1]);
    }

    static inline void Operation(const typename Vec<float16_t, ARCH::NEON>::Type &lhs,
        const typename Vec<float16_t, ARCH::NEON>::Type &rhs,
        typename Vec<float16_t, ARCH::NEON>::Type &out) noexcept
    {
        out.val[0] = vaddq_f16(lhs.val[0], rhs.val[0]);
        out.val[1] = vaddq_f16(lhs.val[1], rhs.val[1]);
    }

    static void Do(const svbool_t& pg0, const svbool_t &pg1, const svbool_t &pg2, const svbool_t &pg3,
        const InputType &lhs0, const InputType &lhs1, const InputType &lhs2, const InputType &lhs3,
        const InputType &rhs0, const InputType &rhs1, const InputType &rhs2, const InputType &rhs3,
        OutType &out0, OutType &out1, OutType &out2, OutType &out3) noexcept
    {
        static_assert(N <= 4, "AddXN: Pseudo unroll by 4 is maximum");
        Operation(lhs0, rhs0, out0);
        if constexpr (N > 1) {
            Operation(lhs1, rhs1, out1);
            if constexpr (N > 2) {
                Operation(lhs2, rhs2, out2);
                if constexpr (N > 3) {
                    Operation(lhs3, rhs3, out3);
                }
            }
        }
        Service::Unused(pg0, pg1, pg2, pg3);
    }
};

template <typename ComputeType, SizeType N>
struct DivXN<ComputeType, N, ARCH::NEON> final {
    using InputType = typename Vector::Vec<ComputeType, ARCH::NEON>::Type;
    using OutType = typename Vector::Vec<ComputeType, ARCH::NEON>::Type;
    static inline void Operation(const typename Vec<float, ARCH::NEON>::Type &lhs,
        const typename Vec<float, ARCH::NEON>::Type &rhs,
        typename Vec<float, ARCH::NEON>::Type &out) noexcept
    {
        out.val[0] = vdivq_f32(lhs.val[0], rhs.val[0]);
        out.val[1] = vdivq_f32(lhs.val[1], rhs.val[1]);
    }

    static inline void Operation(const typename Vec<float16_t, ARCH::NEON>::Type &lhs,
        const typename Vec<float16_t, ARCH::NEON>::Type &rhs,
        typename Vec<float16_t, ARCH::NEON>::Type &out) noexcept
    {
        out.val[0] = vdivq_f16(lhs.val[0], rhs.val[0]);
        out.val[1] = vdivq_f16(lhs.val[1], rhs.val[1]);
    }
    static void Do(const svbool_t& pg0, const svbool_t &pg1, const svbool_t &pg2, const svbool_t &pg3,
        const InputType &lhs0, const InputType &lhs1, const InputType &lhs2, const InputType &lhs3,
        const InputType &rhs0, const InputType &rhs1, const InputType &rhs2, const InputType &rhs3,
        OutType &out0, OutType &out1, OutType &out2, OutType &out3) noexcept
    {
        static_assert(N <= 4, "DivXN: Pseudo unroll by 4 is maximum");
        Operation(lhs0, rhs0, out0);
        if constexpr (N > 1) {
            Operation(lhs1, rhs1, out1);
            if constexpr (N > 2) {
                Operation(lhs2, rhs2, out2);
                if constexpr (N > 3) {
                    Operation(lhs3, rhs3, out3);
                }
            }
        }
        Service::Unused(pg0, pg1, pg2, pg3);
    }
};

template <typename ComputeType, SizeType N>
struct DivMN<ComputeType, N, ARCH::NEON> final {
    static inline void Operation(const typename Vec<float, ARCH::NEON>::Type &lhs,
        const typename Vec<float, ARCH::NEON>::Type &rhs,
        typename Vec<float, ARCH::NEON>::Type &out) noexcept
    {
        out.val[0] = vdivq_f32(lhs.val[0], rhs.val[0]);
        out.val[1] = vdivq_f32(lhs.val[1], rhs.val[1]);
    }
    static inline void Operation(const typename Vec<float16_t, ARCH::NEON>::Type &lhs,
        const typename Vec<float16_t, ARCH::NEON>::Type &rhs,
        typename Vec<float16_t, ARCH::NEON>::Type &out) noexcept
    {
        out.val[0] = vdivq_f16(lhs.val[0], rhs.val[0]);
        out.val[1] = vdivq_f16(lhs.val[1], rhs.val[1]);
    }
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
        typename Vec<ComputeType, ARCH::NEON>::Type &out0, typename Vec<ComputeType, ARCH::NEON>::Type &out1,
        typename Vec<ComputeType, ARCH::NEON>::Type &out2, typename Vec<ComputeType, ARCH::NEON>::Type &out3) noexcept
    {
        static_assert(N <= 4, "Pseudo unroll by 4 is maximum");
        Operation(lhs0, rhs0, out0);
        if constexpr (N > 1) {
            Operation(lhs1, rhs1, out1);
            if constexpr (N > 2) {
                Operation(lhs2, rhs2, out2);
                if constexpr (N > 3) {
                    Operation(lhs3, rhs3, out3);
                }
            }
        }
        Service::Unused(pg0, pg1, pg2, pg3);
    }
};

template <typename ComputeType, SizeType N>
struct MulMaskXN<ComputeType, N, ARCH::NEON> final {
    using InputType = typename Vector::Vec<ComputeType, ARCH::NEON>::Type;
    using OutType = typename Vector::Vec<ComputeType, ARCH::NEON>::Type;
    static inline void Operation(const typename Vec<float, ARCH::NEON>::Type &lhs,
        const typename Vec<float, ARCH::NEON>::Type &rhs,
        const typename MaskVec<ComputeType, ARCH::NEON>::Type &pg,
        typename Vec<float, ARCH::NEON>::Type &out) noexcept
    {
        float32x4_t pNotZero0 = vmulq_f32(lhs.val[0], rhs.val[0]);
        float32x4_t pNotZero1 = vmulq_f32(lhs.val[1], rhs.val[1]);
        out.val[0] = vbslq_f32(pg.val[0], pNotZero0, out.val[0]);
        out.val[1] = vbslq_f32(pg.val[1], pNotZero1, out.val[1]);
    }
    static inline void Operation(const typename Vec<float16_t, ARCH::NEON>::Type &lhs,
        const typename Vec<float16_t, ARCH::NEON>::Type &rhs,
        const typename MaskVec<ComputeType, ARCH::NEON>::Type &pg,
        typename Vec<float16_t, ARCH::NEON>::Type &out) noexcept
    {
        float16x8_t pNotZero0 = vmulq_f16(lhs.val[0], rhs.val[0]);
        float16x8_t pNotZero1 = vmulq_f16(lhs.val[1], rhs.val[1]);
        out.val[0] = vbslq_f16(pg.val[0], pNotZero0, out.val[0]);
        out.val[1] = vbslq_f16(pg.val[1], pNotZero1, out.val[1]);
    }
    static void Do(const typename MaskVec<ComputeType, ARCH::NEON>::Type &pg0,
        const typename MaskVec<ComputeType, ARCH::NEON>::Type &pg1,
        const typename MaskVec<ComputeType, ARCH::NEON>::Type &pg2,
        const typename MaskVec<ComputeType, ARCH::NEON>::Type &pg3,
        const InputType &lhs0, const InputType &lhs1, const InputType &lhs2, const InputType &lhs3,
        const InputType &rhs0, const InputType &rhs1, const InputType &rhs2, const InputType &rhs3,
        OutType &out0, OutType &out1, OutType &out2, OutType &out3) noexcept
    {
        static_assert(N <= 4, "MulMaskXN: Pseudo unroll by 4 is maximum");
        Operation(lhs0, rhs0, pg0, out0);
        if constexpr (N > 1) {
            Operation(lhs1, rhs1, pg1, out1);
            if constexpr (N > 2) {
                Operation(lhs2, rhs2, pg2, out2);
                if constexpr (N > 3) {
                    Operation(lhs3, rhs3, pg3, out3);
                }
            }
        }
        Service::Unused(pg0, pg1, pg2, pg3);
    }
};

template <typename ComputeType, SizeType N>
struct MulXN<ComputeType, N, ARCH::NEON> final {
    using InputType = typename Vector::Vec<ComputeType, ARCH::NEON>::Type;
    using OutType = typename Vector::Vec<ComputeType, ARCH::NEON>::Type;
    static inline void Operation(const typename Vec<float, ARCH::NEON>::Type &lhs,
        const typename Vec<float, ARCH::NEON>::Type &rhs,
        typename Vec<float, ARCH::NEON>::Type &out) noexcept
    {
        out.val[0] = vmulq_f32(lhs.val[0], rhs.val[0]);
        out.val[1] = vmulq_f32(lhs.val[1], rhs.val[1]);
    }
    static inline void Operation(const typename Vec<float16_t, ARCH::NEON>::Type &lhs,
        const typename Vec<float16_t, ARCH::NEON>::Type &rhs,
        typename Vec<float16_t, ARCH::NEON>::Type &out) noexcept
    {
        out.val[0] = vmulq_f16(lhs.val[0], rhs.val[0]);
        out.val[1] = vmulq_f16(lhs.val[1], rhs.val[1]);
    }
    static void Do(const svbool_t &pg0, const svbool_t &pg1, const svbool_t &pg2, const svbool_t &pg3,
        const InputType &lhs0, const InputType &lhs1, const InputType &lhs2, const InputType &lhs3,
        const InputType &rhs0, const InputType &rhs1, const InputType &rhs2, const InputType &rhs3,
        OutType &out0, OutType &out1, OutType &out2, OutType &out3) noexcept
    {
        static_assert(N <= 4, "MulXN: Pseudo unroll by 4 is maximum");
        Operation(lhs0, rhs0, out0);
        if constexpr (N > 1) {
            Operation(lhs1, rhs1, out1);
            if constexpr (N > 2) {
                Operation(lhs2, rhs2, out2);
                if constexpr (N > 3) {
                    Operation(lhs3, rhs3, out3);
                }
            }
        }
        Service::Unused(pg0, pg1, pg2, pg3);
    }
};

template <typename ComputeType, SizeType N>
struct SubXN<ComputeType, N, ARCH::NEON> final {
    using InputType = typename Vector::Vec<ComputeType, ARCH::NEON>::Type;
    using OutType = typename Vector::Vec<ComputeType, ARCH::NEON>::Type;
    static inline void Operation(const typename Vec<float, ARCH::NEON>::Type &lhs,
        const typename Vec<float, ARCH::NEON>::Type &rhs,
        typename Vec<float, ARCH::NEON>::Type &out) noexcept
    {
        out.val[0] = vsubq_f32(lhs.val[0], rhs.val[0]);
        out.val[1] = vsubq_f32(lhs.val[1], rhs.val[1]);
    }
    static inline void Operation(const typename Vec<float16_t, ARCH::NEON>::Type &lhs,
        const typename Vec<float16_t, ARCH::NEON>::Type &rhs,
        typename Vec<float16_t, ARCH::NEON>::Type &out) noexcept
    {
        out.val[0] = vsubq_f16(lhs.val[0], rhs.val[0]);
        out.val[1] = vsubq_f16(lhs.val[1], rhs.val[1]);
    }
    static void Do(const svbool_t& pg0, const svbool_t &pg1, const svbool_t &pg2, const svbool_t &pg3,
        const InputType &lhs0, const InputType &lhs1, const InputType &lhs2, const InputType &lhs3,
        const InputType &rhs0, const InputType &rhs1, const InputType &rhs2, const InputType &rhs3,
        OutType &out0, OutType &out1, OutType &out2, OutType &out3) noexcept
    {
        static_assert(N <= 4, "SubXN: Pseudo unroll by 4 is maximum");
        Operation(lhs0, rhs0, out0);
        if constexpr (N > 1) {
            Operation(lhs1, rhs1, out1);
            if constexpr (N > 2) {
                Operation(lhs2, rhs2, out2);
                if constexpr (N > 3) {
                    Operation(lhs3, rhs3, out3);
                }
            }
        }
        Service::Unused(pg0, pg1, pg2, pg3);
    }
};

template <typename ComputeType, SizeType N>
struct MlaMN<ComputeType, N, ARCH::NEON> final {
    using InputType = typename Vector::Vec<ComputeType, ARCH::NEON>::Type;
    using OutType = typename Vector::Vec<ComputeType, ARCH::NEON>::Type;
    static void Do(const svbool_t& pg0, const svbool_t &pg1, const svbool_t &pg2, const svbool_t &pg3,
        const InputType &x0, const InputType &x1, const InputType &x2, const InputType &x3,
        const InputType &y0, const InputType &y1, const InputType &y2, const InputType &y3,
        const InputType &z0, const InputType &z1, const InputType &z2, const InputType &z3,
        OutType &out0, OutType &out1, OutType &out2, OutType &out3) noexcept
    {
        static_assert(N <= 4, "MlaMN: Pseudo unroll by 4 is maximum");
        Service::Unused(pg0, pg1, pg2, pg3);
        if constexpr (std::is_same_v<ComputeType, float>) {
            out0.val[0] = vfmaq_f32(x0.val[0], y0.val[0], z0.val[0]);
            out0.val[1] = vfmaq_f32(x0.val[1], y0.val[1], z0.val[1]);
            if constexpr (N > 1) {
                out1.val[0] = vfmaq_f32(x1.val[0], y1.val[0], z1.val[0]);
                out1.val[1] = vfmaq_f32(x1.val[1], y1.val[1], z1.val[1]);
                if constexpr (N > 2) {
                    out2.val[0] = vfmaq_f32(x2.val[0], y2.val[0], z2.val[0]);
                    out2.val[1] = vfmaq_f32(x2.val[1], y2.val[1], z2.val[1]);
                    if constexpr (N > 3) {
                        out3.val[0] = vfmaq_f32(x3.val[0], y3.val[0], z3.val[0]);
                        out3.val[1] = vfmaq_f32(x3.val[1], y3.val[1], z3.val[1]);
                    }
                }
            }
        }
    }
};

template <typename ComputeType, SizeType N>
struct MlaXN<ComputeType, N, ARCH::NEON> final {
    using InputType = typename Vector::Vec<ComputeType, ARCH::NEON>::Type;
    using OutType = typename Vector::Vec<ComputeType, ARCH::NEON>::Type;
    static void Do(const svbool_t& pg0, const svbool_t &pg1, const svbool_t &pg2, const svbool_t &pg3,
        const InputType &x0, const InputType &x1, const InputType &x2, const InputType &x3,
        const InputType &y0, const InputType &y1, const InputType &y2, const InputType &y3,
        const InputType &z0, const InputType &z1, const InputType &z2, const InputType &z3,
        OutType &out0, OutType &out1, OutType &out2, OutType &out3) noexcept
    {
        static_assert(N <= 4, "MlaXN: Pseudo unroll by 4 is maximum");
        Service::Unused(pg0, pg1, pg2, pg3);
        if constexpr (std::is_same_v<ComputeType, float>) {
            out0.val[0] = vfmaq_f32(x0.val[0], y0.val[0], z0.val[0]);
            out0.val[1] = vfmaq_f32(x0.val[1], y0.val[1], z0.val[1]);
            if constexpr (N > 1) {
                out1.val[0] = vfmaq_f32(x1.val[0], y1.val[0], z1.val[0]);
                out1.val[1] = vfmaq_f32(x1.val[1], y1.val[1], z1.val[1]);
                if constexpr (N > 2) {
                    out2.val[0] = vfmaq_f32(x2.val[0], y2.val[0], z2.val[0]);
                    out2.val[1] = vfmaq_f32(x2.val[1], y2.val[1], z2.val[1]);
                    if constexpr (N > 3) {
                        out3.val[0] = vfmaq_f32(x3.val[0], y3.val[0], z3.val[0]);
                        out3.val[1] = vfmaq_f32(x3.val[1], y3.val[1], z3.val[1]);
                    }
                }
            }
        }
    }
};

template <typename ComputeType, SizeType N>
struct MulMN<ComputeType, N, ARCH::NEON> final {
    using InputType = typename Vector::Vec<ComputeType, ARCH::NEON>::Type;
    using OutType = typename Vector::Vec<ComputeType, ARCH::NEON>::Type;
    static void Do(const svbool_t& pg0, const svbool_t &pg1, const svbool_t &pg2, const svbool_t &pg3,
        const InputType &lhs0, const InputType &lhs1, const InputType &lhs2, const InputType &lhs3,
        const InputType &rhs0, const InputType &rhs1, const InputType &rhs2, const InputType &rhs3,
        OutType &out0, OutType &out1, OutType &out2, OutType &out3) noexcept
    {
        static_assert(N <= 4, "MulMN: Pseudo unroll by 4 is maximum");
        Service::Unused(pg0, pg1, pg2, pg3);
        if constexpr (std::is_same_v<ComputeType, float>) {
            out0.val[0] = vmulq_f32(lhs0.val[0], rhs0.val[0]);
            out0.val[1] = vmulq_f32(lhs0.val[1], rhs0.val[1]);
            if constexpr (N > 1) {
                out1.val[0] = vmulq_f32(lhs1.val[0], rhs1.val[0]);
                out1.val[1] = vmulq_f32(lhs1.val[1], rhs1.val[1]);
                if constexpr (N > 2) {
                    out2.val[0] = vmulq_f32(lhs2.val[0], rhs2.val[0]);
                    out2.val[1] = vmulq_f32(lhs2.val[1], rhs2.val[1]);
                    if constexpr (N > 3) {
                        out3.val[0] = vmulq_f32(lhs3.val[0], rhs3.val[0]);
                        out3.val[1] = vmulq_f32(lhs3.val[1], rhs3.val[1]);
                    }
                }
            }
        }
    }
};

template <typename ComputeType, SizeType N>
struct MulnXN<ComputeType, N, ARCH::NEON> final {
    using InputType = typename Vector::Vec<ComputeType, ARCH::NEON>::Type;
    using OutType = typename Vector::Vec<ComputeType, ARCH::NEON>::Type;
    static inline void Operation(const typename Vec<float, ARCH::NEON>::Type &lhs,
        const ComputeType &rhs, typename Vec<float, ARCH::NEON>::Type &out) noexcept
    {
        out.val[0] = vmulq_n_f32(lhs.val[0], rhs);
        out.val[1] = vmulq_n_f32(lhs.val[1], rhs);
    }
    static void Do(const svbool_t& pg0, const svbool_t &pg1,
        const svbool_t &pg2, const svbool_t &pg3,
        const typename Vec<ComputeType, ARCH::NEON>::Type &lhs0,
        const typename Vec<ComputeType, ARCH::NEON>::Type &lhs1,
        const typename Vec<ComputeType, ARCH::NEON>::Type &lhs2,
        const typename Vec<ComputeType, ARCH::NEON>::Type &lhs3,
        const ComputeType &rhs0, const ComputeType &rhs1,
        const ComputeType &rhs2, const ComputeType &rhs3,
        typename Vec<ComputeType, ARCH::NEON>::Type &out0, typename Vec<ComputeType, ARCH::NEON>::Type &out1,
        typename Vec<ComputeType, ARCH::NEON>::Type &out2, typename Vec<ComputeType, ARCH::NEON>::Type &out3) noexcept
    {
        static_assert(N <= 4, "Pseudo unroll by 4 is maximum");
        Operation(lhs0, rhs0, out0);
        if constexpr (N > 1) {
            Operation(lhs1, rhs1, out1);
            if constexpr (N > 2) {
                Operation(lhs2, rhs2, out2);
                if constexpr (N > 3) {
                    Operation(lhs3, rhs3, out3);
                }
            }
        }
        Service::Unused(pg0, pg1, pg2, pg3);
    }
};

template <typename ComputeType, SizeType N>
struct AddMN<ComputeType, N, ARCH::NEON> final {
    using InputType = typename Vector::Vec<ComputeType, ARCH::NEON>::Type;
    using OutType = typename Vector::Vec<ComputeType, ARCH::NEON>::Type;
    static void Do(const svbool_t& pg0, const svbool_t &pg1, const svbool_t &pg2, const svbool_t &pg3,
        const InputType &lhs0, const InputType &lhs1, const InputType &lhs2, const InputType &lhs3,
        const InputType &rhs0, const InputType &rhs1, const InputType &rhs2, const InputType &rhs3,
        OutType &out0, OutType &out1, OutType &out2, OutType &out3) noexcept
    {
        static_assert(N <= 4, "AddMN: Pseudo unroll by 4 is maximum");
        Service::Unused(pg0, pg1, pg2, pg3);
        if constexpr (std::is_same_v<ComputeType, float>) {
            out0.val[0] = vaddq_f32(lhs0.val[0], rhs0.val[0]);
            out0.val[1] = vaddq_f32(lhs0.val[1], rhs0.val[1]);
            if constexpr (N > 1) {
                out1.val[0] = vaddq_f32(lhs1.val[0], rhs1.val[0]);
                out1.val[1] = vaddq_f32(lhs1.val[1], rhs1.val[1]);
                if constexpr (N > 2) {
                    out2.val[0] = vaddq_f32(lhs2.val[0], rhs2.val[0]);
                    out2.val[1] = vaddq_f32(lhs2.val[1], rhs2.val[1]);
                    if constexpr (N > 3) {
                        out3.val[0] = vaddq_f32(lhs3.val[0], rhs3.val[0]);
                        out3.val[1] = vaddq_f32(lhs3.val[1], rhs3.val[1]);
                    }
                }
            }
        }
    }
};

} // Vector
} // Detail
} // KuDNN

#endif // KUDNNL_920G
#endif // KUDNN_VECTOR_ARIMETHIC_NEON