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

#ifndef KUDNN_VECTOR_ARITHMETIC_VC_MUL_DIV
#define KUDNN_VECTOR_ARITHMETIC_VC_MUL_DIV

#include <arm_sve.h>

#include "internal_funcs.hpp"
#include "arm_neon.h"
#include "vector_arithmetic.hpp"

namespace KuDNN {
namespace Detail {
namespace Vector {

template <typename ComputeType, SizeType N>
struct DivXN<ComputeType, N, ARCH::VC> final {
    static void Do(const svbool_t& pg0, const svbool_t &pg1,
        const svbool_t &pg2, const svbool_t &pg3,
        const typename Vec<ComputeType>::Type &lhs0, const typename Vec<ComputeType>::Type &lhs1,
        const typename Vec<ComputeType>::Type &lhs2, const typename Vec<ComputeType>::Type &lhs3,
        const typename Vec<ComputeType>::Type &rhs0, const typename Vec<ComputeType>::Type &rhs1,
        const typename Vec<ComputeType>::Type &rhs2, const typename Vec<ComputeType>::Type &rhs3,
        typename Vec<ComputeType>::Type &out0, typename Vec<ComputeType>::Type &out1,
        typename Vec<ComputeType>::Type &out2, typename Vec<ComputeType>::Type &out3) noexcept
    {
        static_assert(N <= 4, "Pseudo unroll by 4 is maximum");
        out0 = svdiv_x(pg0, lhs0, rhs0);
        if constexpr (N > 1) {
            out1 = svdiv_x(pg1, lhs1, rhs1);
            if constexpr (N > 2) {
                out2 = svdiv_x(pg2, lhs2, rhs2);
                if constexpr (N > 3) {
                    out3 = svdiv_x(pg3, lhs3, rhs3);
                }
            }
        }
    }
};

template <typename ComputeType, SizeType N>
struct DivMN<ComputeType, N, ARCH::VC> final {
    static void Do(const svbool_t& pg0, const svbool_t &pg1,
        const svbool_t &pg2, const svbool_t &pg3,
        const typename Vec<ComputeType>::Type &lhs0, const typename Vec<ComputeType>::Type &lhs1,
        const typename Vec<ComputeType>::Type &lhs2, const typename Vec<ComputeType>::Type &lhs3,
        const typename Vec<ComputeType>::Type &rhs0, const typename Vec<ComputeType>::Type &rhs1,
        const typename Vec<ComputeType>::Type &rhs2, const typename Vec<ComputeType>::Type &rhs3,
        typename Vec<ComputeType>::Type &out0, typename Vec<ComputeType>::Type &out1,
        typename Vec<ComputeType>::Type &out2, typename Vec<ComputeType>::Type &out3) noexcept
    {
        static_assert(N <= 4, "Pseudo unroll by 4 is maximum");
        out0 = svdiv_m(pg0, lhs0, rhs0);
        if constexpr (N > 1) {
            out1 = svdiv_m(pg1, lhs1, rhs1);
            if constexpr (N > 2) {
                out2 = svdiv_m(pg2, lhs2, rhs2);
                if constexpr (N > 3) {
                    out3 = svdiv_m(pg3, lhs3, rhs3);
                }
            }
        }
    }
};

template <typename ComputeType, SizeType N>
struct DivnXN final {
    static void Do(const svbool_t& pg0, const svbool_t &pg1,
        const svbool_t &pg2, const svbool_t &pg3,
        const typename Vec<ComputeType>::Type &lhs0, const typename Vec<ComputeType>::Type &lhs1,
        const typename Vec<ComputeType>::Type &lhs2, const typename Vec<ComputeType>::Type &lhs3,
        const ComputeType &rhs0, const ComputeType &rhs1,
        const ComputeType &rhs2, const ComputeType &rhs3,
        typename Vec<ComputeType>::Type &out0, typename Vec<ComputeType>::Type &out1,
        typename Vec<ComputeType>::Type &out2, typename Vec<ComputeType>::Type &out3) noexcept
    {
        static_assert(N <= 4, "Pseudo unroll by 4 is maximum");
        out0 = svdiv_x(pg0, lhs0, rhs0);
        if constexpr (N > 1) {
            out1 = svdiv_x(pg1, lhs1, rhs1);
            if constexpr (N > 2) {
                out2 = svdiv_x(pg2, lhs2, rhs2);
                if constexpr (N > 3) {
                    out3 = svdiv_x(pg3, lhs3, rhs3);
                }
            }
        }
    }
};

template <typename ComputeType, SizeType N>
struct MulMaskXN<ComputeType, N, ARCH::VC> final {
    static void Do(const svbool_t& pg0, const svbool_t &pg1,
        const svbool_t &pg2, const svbool_t &pg3,
        const typename Vec<ComputeType>::Type &lhs0, const typename Vec<ComputeType>::Type &lhs1,
        const typename Vec<ComputeType>::Type &lhs2, const typename Vec<ComputeType>::Type &lhs3,
        const typename Vec<ComputeType>::Type &rhs0, const typename Vec<ComputeType>::Type &rhs1,
        const typename Vec<ComputeType>::Type &rhs2, const typename Vec<ComputeType>::Type &rhs3,
        typename Vec<ComputeType>::Type &out0, typename Vec<ComputeType>::Type &out1,
        typename Vec<ComputeType>::Type &out2, typename Vec<ComputeType>::Type &out3) noexcept
    {
        static_assert(N <= 4, "Pseudo unroll by 4 is maximum");
        out0 = svmul_x(pg0, lhs0, rhs0);
        if constexpr (N > 1) {
            out1 = svmul_x(pg1, lhs1, rhs1);
            if constexpr (N > 2) {
                out2 = svmul_x(pg2, lhs2, rhs2);
                if constexpr (N > 3) {
                    out3 = svmul_x(pg3, lhs3, rhs3);
                }
            }
        }
    }
};

template <typename ComputeType, SizeType N>
struct MulXN<ComputeType, N, ARCH::VC> final {
    static void Do(const svbool_t& pg0, const svbool_t &pg1,
        const svbool_t &pg2, const svbool_t &pg3,
        const typename Vec<ComputeType>::Type &lhs0, const typename Vec<ComputeType>::Type &lhs1,
        const typename Vec<ComputeType>::Type &lhs2, const typename Vec<ComputeType>::Type &lhs3,
        const typename Vec<ComputeType>::Type &rhs0, const typename Vec<ComputeType>::Type &rhs1,
        const typename Vec<ComputeType>::Type &rhs2, const typename Vec<ComputeType>::Type &rhs3,
        typename Vec<ComputeType>::Type &out0, typename Vec<ComputeType>::Type &out1,
        typename Vec<ComputeType>::Type &out2, typename Vec<ComputeType>::Type &out3) noexcept
    {
        static_assert(N <= 4, "Pseudo unroll by 4 is maximum");
        out0 = svmul_x(pg0, lhs0, rhs0);
        if constexpr (N > 1) {
            out1 = svmul_x(pg1, lhs1, rhs1);
            if constexpr (N > 2) {
                out2 = svmul_x(pg2, lhs2, rhs2);
                if constexpr (N > 3) {
                    out3 = svmul_x(pg3, lhs3, rhs3);
                }
            }
        }
    }
};

template <typename ComputeType, SizeType N>
struct MulnXN<ComputeType, N, ARCH::VC> final {
    static void Do(const svbool_t& pg0, const svbool_t &pg1,
        const svbool_t &pg2, const svbool_t &pg3,
        const typename Vec<ComputeType>::Type &lhs0, const typename Vec<ComputeType>::Type &lhs1,
        const typename Vec<ComputeType>::Type &lhs2, const typename Vec<ComputeType>::Type &lhs3,
        const ComputeType &rhs0, const ComputeType &rhs1,
        const ComputeType &rhs2, const ComputeType &rhs3,
        typename Vec<ComputeType>::Type &out0, typename Vec<ComputeType>::Type &out1,
        typename Vec<ComputeType>::Type &out2, typename Vec<ComputeType>::Type &out3) noexcept
    {
        static_assert(N <= 4, "Pseudo unroll by 4 is maximum");
        out0 = svmul_x(pg0, lhs0, rhs0);
        if constexpr (N > 1) {
            out1 = svmul_x(pg1, lhs1, rhs1);
            if constexpr (N > 2) {
                out2 = svmul_x(pg2, lhs2, rhs2);
                if constexpr (N > 3) {
                    out3 = svmul_x(pg3, lhs3, rhs3);
                }
            }
        }
    }
};

template <typename ComputeType, SizeType N>
struct MulZN final {
    static void Do(const svbool_t& pg0, const svbool_t &pg1,
        const svbool_t &pg2, const svbool_t &pg3,
        const typename Vec<ComputeType>::Type &lhs0, const typename Vec<ComputeType>::Type &lhs1,
        const typename Vec<ComputeType>::Type &lhs2, const typename Vec<ComputeType>::Type &lhs3,
        const typename Vec<ComputeType>::Type &rhs0, const typename Vec<ComputeType>::Type &rhs1,
        const typename Vec<ComputeType>::Type &rhs2, const typename Vec<ComputeType>::Type &rhs3,
        typename Vec<ComputeType>::Type &out0, typename Vec<ComputeType>::Type &out1,
        typename Vec<ComputeType>::Type &out2, typename Vec<ComputeType>::Type &out3) noexcept
    {
        static_assert(N <= 4, "Pseudo unroll by 4 is maximum");
        out0 = svmul_z(pg0, lhs0, rhs0);
        if constexpr (N > 1) {
            out1 = svmul_z(pg1, lhs1, rhs1);
            if constexpr (N > 2) {
                out2 = svmul_z(pg2, lhs2, rhs2);
                if constexpr (N > 3) {
                    out3 = svmul_z(pg3, lhs3, rhs3);
                }
            }
        }
    }
};

template <typename ComputeType, SizeType N>
struct MulMN<ComputeType, N, ARCH::VC> final {
    static void Do(const svbool_t& pg0, const svbool_t &pg1,
        const svbool_t &pg2, const svbool_t &pg3,
        const typename Vec<ComputeType>::Type &lhs0, const typename Vec<ComputeType>::Type &lhs1,
        const typename Vec<ComputeType>::Type &lhs2, const typename Vec<ComputeType>::Type &lhs3,
        const typename Vec<ComputeType>::Type &rhs0, const typename Vec<ComputeType>::Type &rhs1,
        const typename Vec<ComputeType>::Type &rhs2, const typename Vec<ComputeType>::Type &rhs3,
        typename Vec<ComputeType>::Type &out0, typename Vec<ComputeType>::Type &out1,
        typename Vec<ComputeType>::Type &out2, typename Vec<ComputeType>::Type &out3) noexcept
    {
        static_assert(N <= 4, "Pseudo unroll by 4 is maximum");
        out0 = svmul_m(pg0, lhs0, rhs0);
        if constexpr (N > 1) {
            out1 = svmul_m(pg1, lhs1, rhs1);
            if constexpr (N > 2) {
                out2 = svmul_m(pg2, lhs2, rhs2);
                if constexpr (N > 3) {
                    out3 = svmul_m(pg3, lhs3, rhs3);
                }
            }
        }
    }
};

} // Vector
} // Detail
} // KuDNN

#endif // KUDNN_VECTOR_ARITHMETIC_VC_MUL_DIV