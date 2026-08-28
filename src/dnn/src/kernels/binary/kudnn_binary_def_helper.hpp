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

#ifndef KUDNN_BINARY_DEF_HELPER_HPP
#define KUDNN_BINARY_DEF_HELPER_HPP

#include <arm_sve.h>

#include "kudnn.hpp"

#include "vector_wrappers.hpp"
#include "kernels/binary/kudnn_binary_kernels.hpp"

namespace KuDNN {
namespace Detail {
namespace Binary {

namespace Helpers {

template <typename OpWrapper, typename ComputeType, SizeType N, ARCH Arch = ARCH::VC>
struct ApeZN final {
    static typename Vector::Vec<ComputeType, Arch>::Type DupHelper(ComputeType in);

    static void Do(const svbool_t& pg0, const svbool_t &pg1, const svbool_t &pg2,
        const svbool_t &pg3,
        const typename Vector::Vec<ComputeType, Arch>::Type &lhs0,
        const typename Vector::Vec<ComputeType, Arch>::Type &lhs1,
        const typename Vector::Vec<ComputeType, Arch>::Type &lhs2,
        const typename Vector::Vec<ComputeType, Arch>::Type &lhs3,
        const typename Vector::Vec<ComputeType, Arch>::Type &rhs0,
        const typename Vector::Vec<ComputeType, Arch>::Type &rhs1,
        const typename Vector::Vec<ComputeType, Arch>::Type &rhs2,
        const typename Vector::Vec<ComputeType, Arch>::Type &rhs3,
        typename Vector::Vec<ComputeType, Arch>::Type &out0,
        typename Vector::Vec<ComputeType, Arch>::Type &out1,
        typename Vector::Vec<ComputeType, Arch>::Type &out2,
        typename Vector::Vec<ComputeType, Arch>::Type &out3) noexcept;
};

// Add Predicate Elements (APE)
template <typename OpWrapper, typename ComputeType, SizeType N>
struct ApeZN<OpWrapper, ComputeType, N, ARCH::VC> final {
    static typename Vector::Vec<ComputeType, ARCH::VC>::Type DupHelper(ComputeType in)
    {
        if constexpr (std::is_same_v<ComputeType, float>) {
            return svdup_f32(in);
        } else { // f16
            return svdup_f16(in);
        }
    }

    static void Do(const svbool_t& pg0, const svbool_t &pg1, const svbool_t &pg2,
        const svbool_t &pg3,
        const typename Vector::Vec<ComputeType, ARCH::VC>::Type &lhs0,
        const typename Vector::Vec<ComputeType, ARCH::VC>::Type &lhs1,
        const typename Vector::Vec<ComputeType, ARCH::VC>::Type &lhs2,
        const typename Vector::Vec<ComputeType, ARCH::VC>::Type &lhs3,
        const typename Vector::Vec<ComputeType, ARCH::VC>::Type &rhs0,
        const typename Vector::Vec<ComputeType, ARCH::VC>::Type &rhs1,
        const typename Vector::Vec<ComputeType, ARCH::VC>::Type &rhs2,
        const typename Vector::Vec<ComputeType, ARCH::VC>::Type &rhs3,
        typename Vector::Vec<ComputeType, ARCH::VC>::Type &out0,
        typename Vector::Vec<ComputeType, ARCH::VC>::Type &out1,
        typename Vector::Vec<ComputeType, ARCH::VC>::Type &out2,
        typename Vector::Vec<ComputeType, ARCH::VC>::Type &out3) noexcept
    {
        static_assert(N <= 4, "Pseudo unroll by 4 is maximum");
        svbool_t pgOut0;
        svbool_t pgOut1;
        svbool_t pgOut2;
        svbool_t pgOut3;
        OpWrapper::Do(pg0, pg1, pg2, pg3, lhs0, lhs1, lhs2, lhs3, rhs0, rhs1, rhs2, rhs3,
            pgOut0, pgOut1, pgOut2, pgOut3);
        out0 = svadd_z(pgOut0, DupHelper(1.0), DupHelper(0.0));
        if constexpr (N > 1) {
            out1 = svadd_z(pgOut1, DupHelper(1.0), DupHelper(0.0));
            if constexpr (N > 2) {
                out2 = svadd_z(pgOut2, DupHelper(1.0), DupHelper(0.0));
                if constexpr (N > 3) {
                    out3 = svadd_z(pgOut3, DupHelper(1.0), DupHelper(0.0));
                }
            }
        }
    }
};

#ifdef KUDNNL_920G
// Add Predicate Elements (APE)
template <typename OpWrapper, typename ComputeType, SizeType N>
struct ApeZN<OpWrapper, ComputeType, N, ARCH::NEON> final {
    using InputType = typename Vector::Vec<ComputeType, ARCH::NEON>::Type;
    using OutType = typename Vector::Vec<ComputeType, ARCH::NEON>::Type;
    static typename Vector::Vec<ComputeType, ARCH::NEON>::Type DupHelper(ComputeType in)
    {
        if constexpr (std::is_same_v<ComputeType, float>) {
            return vld1q_f32_x2(in);
        } else { // f16
            return vld1q_f16_x2(in);
        }
    }

    static inline void Operation(const typename Vector::MaskVec<float, ARCH::NEON>::Type &pg,
    typename Vector::Vec<float, ARCH::NEON>::Type &out) noexcept
    {
        const float32x4_t vecOne = vmovq_n_f32(1.0f);
        const float32x4_t vecZero = vmovq_n_f32(0.0f);
        out.val[0] = vbslq_f32(pg.val[0], vecOne, vecZero);
        out.val[1] = vbslq_f32(pg.val[1], vecOne, vecZero);
    }

    static inline void Operation(const typename Vector::MaskVec<float16_t, ARCH::NEON>::Type &pg,
    typename Vector::Vec<float16_t, ARCH::NEON>::Type &out) noexcept
    {
        const float16x8_t vecOne = vmovq_n_f16(1.0f);
        const float16x8_t vecZero = vmovq_n_f16(0.0f);
        out.val[0] = vbslq_f16(pg.val[0], vecOne, vecZero);
        out.val[1] = vbslq_f16(pg.val[1], vecOne, vecZero);
    }
    static void Do(const svbool_t& pg0, const svbool_t &pg1, const svbool_t &pg2,
        const svbool_t &pg3,
        const InputType &lhs0, const InputType &lhs1, const InputType &lhs2, const InputType &lhs3,
        const InputType &rhs0, const InputType &rhs1, const InputType &rhs2, const InputType &rhs3,
        OutType &out0, OutType &out1, OutType &out2, OutType &out3) noexcept
    {
        static_assert(N <= 4, "Pseudo unroll by 4 is maximum");
        typename Vector::MaskVec<ComputeType, ARCH::NEON>::Type pgOut0;
        typename Vector::MaskVec<ComputeType, ARCH::NEON>::Type pgOut1;
        typename Vector::MaskVec<ComputeType, ARCH::NEON>::Type pgOut2;
        typename Vector::MaskVec<ComputeType, ARCH::NEON>::Type pgOut3;
        OpWrapper::Do(pg0, pg1, pg2, pg3, lhs0, lhs1, lhs2, lhs3, rhs0, rhs1, rhs2, rhs3,
            pgOut0, pgOut1, pgOut2, pgOut3);
        Operation(pgOut0, out0);
        if constexpr (N > 1) {
            Operation(pgOut1, out1);
            if constexpr (N > 2) {
                Operation(pgOut2, out2);
                if constexpr (N > 3) {
                    Operation(pgOut3, out3);
                }
            }
        }
    }
};
#endif

template <typename ComputeType, SizeType N, ARCH Arch = ARCH::VC>
struct GeNWrapper final {
    using InputType = typename Vector::Vec<ComputeType, Arch>::Type;
    using OutType = typename Vector::MaskVec<ComputeType, Arch>::Type;
    static void Do(const svbool_t& pg0, const svbool_t &pg1, const svbool_t &pg2, const svbool_t &pg3,
        const InputType &lhs0, const InputType &lhs1, const InputType &lhs2, const InputType &lhs3,
        const InputType &rhs0, const InputType &rhs1, const InputType &rhs2, const InputType &rhs3,
        OutType &out0, OutType &out1, OutType &out2, OutType &out3) noexcept
    {
        Vector::GeN<ComputeType, N, Arch>::Do(pg0, pg1, pg2, pg3, lhs0, lhs1, lhs2, lhs3,
            rhs0, rhs1, rhs2, rhs3, out0, out1, out2, out3);
    }
};

template <typename ComputeType, SizeType N, ARCH Arch = ARCH::VC>
struct GtNWrapper final {
    using InputType = typename Vector::Vec<ComputeType, Arch>::Type;
    using OutType = typename Vector::MaskVec<ComputeType, Arch>::Type;
    static void Do(const svbool_t& pg0, const svbool_t &pg1, const svbool_t &pg2, const svbool_t &pg3,
        const InputType &lhs0, const InputType &lhs1, const InputType &lhs2, const InputType &lhs3,
        const InputType &rhs0, const InputType &rhs1, const InputType &rhs2, const InputType &rhs3,
        OutType &out0, OutType &out1, OutType &out2, OutType &out3) noexcept
    {
        Vector::GtN<ComputeType, N, Arch>::Do(pg0, pg1, pg2, pg3, lhs0, lhs1, lhs2, lhs3,
            rhs0, rhs1, rhs2, rhs3, out0, out1, out2, out3);
    }
};

template <typename ComputeType, SizeType N, ARCH Arch = ARCH::VC>
struct LeNWrapper final {
    using InputType = typename Vector::Vec<ComputeType, Arch>::Type;
    using OutType = typename Vector::MaskVec<ComputeType, Arch>::Type;
    static void Do(const svbool_t& pg0, const svbool_t &pg1, const svbool_t &pg2, const svbool_t &pg3,
       const InputType &lhs0, const InputType &lhs1, const InputType &lhs2, const InputType &lhs3,
       const InputType &rhs0, const InputType &rhs1, const InputType &rhs2, const InputType &rhs3,
       OutType &out0, OutType &out1, OutType &out2, OutType &out3) noexcept
    {
        Vector::LeN<ComputeType, N, Arch>::Do(pg0, pg1, pg2, pg3, lhs0, lhs1, lhs2, lhs3,
            rhs0, rhs1, rhs2, rhs3, out0, out1, out2, out3);
    }
};

template <typename ComputeType, SizeType N, ARCH Arch = ARCH::VC>
struct LtNWrapper final {
    using InputType = typename Vector::Vec<ComputeType, Arch>::Type;
    using OutType = typename Vector::MaskVec<ComputeType, Arch>::Type;
    static void Do(const svbool_t& pg0, const svbool_t &pg1, const svbool_t &pg2, const svbool_t &pg3,
    const InputType &lhs0, const InputType &lhs1, const InputType &lhs2, const InputType &lhs3,
    const InputType &rhs0, const InputType &rhs1, const InputType &rhs2, const InputType &rhs3,
    OutType &out0, OutType &out1, OutType &out2, OutType &out3) noexcept
    {
        Vector::LtN<ComputeType, N, Arch>::Do(pg0, pg1, pg2, pg3, lhs0, lhs1, lhs2, lhs3,
            rhs0, rhs1, rhs2, rhs3, out0, out1, out2, out3);
    }
};

template <typename ComputeType, SizeType N, ARCH Arch = ARCH::VC>
struct EqNWrapper final {
    using InputType = typename Vector::Vec<ComputeType, Arch>::Type;
    using OutType = typename Vector::MaskVec<ComputeType, Arch>::Type;
    static void Do(const svbool_t& pg0, const svbool_t &pg1, const svbool_t &pg2, const svbool_t &pg3,
    const InputType &lhs0, const InputType &lhs1, const InputType &lhs2, const InputType &lhs3,
    const InputType &rhs0, const InputType &rhs1, const InputType &rhs2, const InputType &rhs3,
    OutType &out0, OutType &out1, OutType &out2, OutType &out3) noexcept
    {
        Vector::EqN<ComputeType, N, Arch>::Do(pg0, pg1, pg2, pg3, lhs0, lhs1, lhs2, lhs3,
            rhs0, rhs1, rhs2, rhs3, out0, out1, out2, out3);
    }
};

template <typename ComputeType, SizeType N, ARCH Arch = ARCH::VC>
struct NeNWrapper final {
    using InputType = typename Vector::Vec<ComputeType, Arch>::Type;
    using OutType = typename Vector::MaskVec<ComputeType, Arch>::Type;
    static void Do(const svbool_t& pg0, const svbool_t &pg1, const svbool_t &pg2, const svbool_t &pg3,
    const InputType &lhs0, const InputType &lhs1, const InputType &lhs2, const InputType &lhs3,
    const InputType &rhs0, const InputType &rhs1, const InputType &rhs2, const InputType &rhs3,
    OutType &out0, OutType &out1, OutType &out2, OutType &out3) noexcept
    {
        Vector::NeN<ComputeType, N, Arch>::Do(pg0, pg1, pg2, pg3, lhs0, lhs1, lhs2, lhs3,
            rhs0, rhs1, rhs2, rhs3, out0, out1, out2, out3);
    }
};

} // Helpers

} // Binary
} // Detail
} // KuDNN

#endif //  KUDNN_BINARY_DEF_HELPER_HPP
