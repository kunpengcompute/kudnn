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

#ifndef KUDNN_VECTOR_ARITHMETIC_VC
#define KUDNN_VECTOR_ARITHMETIC_VC

#include <arm_sve.h>

#include "internal_funcs.hpp"
#include "arm_neon.h"
#include "vector_arithmetic.hpp"

namespace KuDNN {
namespace Detail {
namespace Vector {

// currently supports only for 1, 2 elements
// the same logic as zip1 but operates joint vector
// to perfrom single Zip1 it is required to combine zip1 and zip2
//          +---------+---------+          +---------+---------+
//  src00   | src00.l | src00.h |   src01  | src10.l | src10.h |
//          +---------+---------+          +---------+---------+
//              |                                           |
//              v                                           v
//          +---------+---------+          +---------+---------+
//  out00   | src00.l | src10.l |   out01  | src00.h | src10.h |
//          +---------+---------+          +---------+---------+
// input:   src00, src10
// output:  out00 = zip1(src00, src10) out01 = zip(src00, src10)
template <typename ComputeType, SizeType N>
struct Zip1N final {
    static void Do(const typename Vec<ComputeType>::Type &lhs0, const typename Vec<ComputeType>::Type &lhs1,
        const typename Vec<ComputeType>::Type &lhs2, const typename Vec<ComputeType>::Type &lhs3,
        const typename Vec<ComputeType>::Type &rhs0, const typename Vec<ComputeType>::Type &rhs1,
        const typename Vec<ComputeType>::Type &rhs2, const typename Vec<ComputeType>::Type &rhs3,
        typename Vec<ComputeType>::Type &out0, typename Vec<ComputeType>::Type &out1,
        typename Vec<ComputeType>::Type &out2, typename Vec<ComputeType>::Type &out3) noexcept
    {
        Service::Unused(lhs1, lhs2, lhs3, rhs1, rhs2, rhs3, out2, out3);
        static_assert(N <= 2, "Currently supports only 2");
        out0 = svzip1(lhs0, rhs0);
        if constexpr (N > 1) {
            out1 = svzip2(lhs0, rhs0);
        }
    }
};

// currently supports only for 1, 2 elements
// the same logic as zip2 but operates joint vector
// to perfrom single Zip2 it is required to combine zip1 and zip2
//          +---------+---------+          +---------+---------+
//  src10   | src01.l | src01.h |   src11  | src11.l | src11.h |
//          +---------+---------+          +---------+---------+
//              |                                           |
//              v                                           v
//          +---------+---------+          +---------+---------+
//  out10   | src01.l | src11.l |   out11  | src01.h | src11.h |
//          +---------+---------+          +---------+---------+
// input:   src01, src11
// output:  out10 = zip1(src01, src11) out11 = zip(src01, src11)
template <typename ComputeType, SizeType N>
struct Zip2N final {
    static void Do(const typename Vec<ComputeType>::Type &lhs0, const typename Vec<ComputeType>::Type &lhs1,
        const typename Vec<ComputeType>::Type &lhs2, const typename Vec<ComputeType>::Type &lhs3,
        const typename Vec<ComputeType>::Type &rhs0, const typename Vec<ComputeType>::Type &rhs1,
        const typename Vec<ComputeType>::Type &rhs2, const typename Vec<ComputeType>::Type &rhs3,
        typename Vec<ComputeType>::Type &out0, typename Vec<ComputeType>::Type &out1,
        typename Vec<ComputeType>::Type &out2, typename Vec<ComputeType>::Type &out3) noexcept
    {
        Service::Unused(lhs0, lhs2, lhs3, rhs0, rhs2, rhs3, out2, out3);
        static_assert(N <= 2, "Currently supports only 2");
        out0 = svzip1(lhs1, rhs1);
        if constexpr (N > 1) {
            out1 = svzip2(lhs1, rhs1);
        }
    }
};

template <typename ComputeType, SizeType N>
struct AddXN<ComputeType, N, ARCH::VC> final {
    static void Do(const svbool_t& pg0, const svbool_t &pg1, const svbool_t &pg2, const svbool_t &pg3,
        const typename Vec<ComputeType>::Type &lhs0, const typename Vec<ComputeType>::Type &lhs1,
        const typename Vec<ComputeType>::Type &lhs2, const typename Vec<ComputeType>::Type &lhs3,
        const typename Vec<ComputeType>::Type &rhs0, const typename Vec<ComputeType>::Type &rhs1,
        const typename Vec<ComputeType>::Type &rhs2, const typename Vec<ComputeType>::Type &rhs3,
        typename Vec<ComputeType>::Type &out0, typename Vec<ComputeType>::Type &out1,
        typename Vec<ComputeType>::Type &out2, typename Vec<ComputeType>::Type &out3) noexcept
    {
        static_assert(N <= 4, "Pseudo unroll by 4 is maximum");
        out0 = svadd_x(pg0, lhs0, rhs0);
        if constexpr (N > 1) {
            out1 = svadd_x(pg1, lhs1, rhs1);
            if constexpr (N > 2) {
                out2 = svadd_x(pg2, lhs2, rhs2);
                if constexpr (N > 3) {
                    out3 = svadd_x(pg3, lhs3, rhs3);
                }
            }
        }
    }
};

template <typename ComputeType, SizeType N>
struct AddnXN final {
    static void Do(const svbool_t& pg0, const svbool_t &pg1, const svbool_t &pg2, const svbool_t &pg3,
        const typename Vec<ComputeType>::Type &lhs0, const typename Vec<ComputeType>::Type &lhs1,
        const typename Vec<ComputeType>::Type &lhs2, const typename Vec<ComputeType>::Type &lhs3,
        ComputeType &rhs0, ComputeType &rhs1,
        ComputeType &rhs2, ComputeType &rhs3,
        typename Vec<ComputeType>::Type &out0, typename Vec<ComputeType>::Type &out1,
        typename Vec<ComputeType>::Type &out2, typename Vec<ComputeType>::Type &out3) noexcept
    {
        static_assert(N <= 4, "Pseudo unroll by 4 is maximum");
        out0 = svadd_x(pg0, lhs0, rhs0);
        if constexpr (N > 1) {
            out1 = svadd_x(pg1, lhs1, rhs1);
            if constexpr (N > 2) {
                out2 = svadd_x(pg2, lhs2, rhs2);
                if constexpr (N > 3) {
                    out3 = svadd_x(pg3, lhs3, rhs3);
                }
            }
        }
    }
};

template <typename ComputeType, SizeType N>
struct AddMN<ComputeType, N, ARCH::VC> final {
    static void Do(const svbool_t& pg0, const svbool_t &pg1, const svbool_t &pg2, const svbool_t &pg3,
        const typename Vec<ComputeType>::Type &lhs0, const typename Vec<ComputeType>::Type &lhs1,
        const typename Vec<ComputeType>::Type &lhs2, const typename Vec<ComputeType>::Type &lhs3,
        const typename Vec<ComputeType>::Type &rhs0, const typename Vec<ComputeType>::Type &rhs1,
        const typename Vec<ComputeType>::Type &rhs2, const typename Vec<ComputeType>::Type &rhs3,
        typename Vec<ComputeType>::Type &out0, typename Vec<ComputeType>::Type &out1,
        typename Vec<ComputeType>::Type &out2, typename Vec<ComputeType>::Type &out3) noexcept
    {
        static_assert(N <= 4, "Pseudo unroll by 4 is maximum");
        out0 = svadd_m(pg0, lhs0, rhs0);
        if constexpr (N > 1) {
            out1 = svadd_m(pg1, lhs1, rhs1);
            if constexpr (N > 2) {
                out2 = svadd_m(pg2, lhs2, rhs2);
                if constexpr (N > 3) {
                    out3 = svadd_m(pg3, lhs3, rhs3);
                }
            }
        }
    }
};

template <typename ComputeType, SizeType N>
struct NmlsZN final {
    static void Do(const svbool_t& pg0, const svbool_t &pg1,
        const svbool_t &pg2, const svbool_t &pg3,
        const typename Vec<ComputeType>::Type &x0, const typename Vec<ComputeType>::Type &x1,
        const typename Vec<ComputeType>::Type &x2, const typename Vec<ComputeType>::Type &x3,
        const typename Vec<ComputeType>::Type &y0, const typename Vec<ComputeType>::Type &y1,
        const typename Vec<ComputeType>::Type &y2, const typename Vec<ComputeType>::Type &y3,
        const typename Vec<ComputeType>::Type &z0, const typename Vec<ComputeType>::Type &z1,
        const typename Vec<ComputeType>::Type &z2, const typename Vec<ComputeType>::Type &z3,
        typename Vec<ComputeType>::Type &out0, typename Vec<ComputeType>::Type &out1,
        typename Vec<ComputeType>::Type &out2, typename Vec<ComputeType>::Type &out3) noexcept
    {
        static_assert(N <= 4, "Pseudo unroll by 4 is maximum");
        out0 = svnmls_z(pg0, x0, y0, z0);
        if constexpr (N > 1) {
            out1 = svnmls_z(pg1, x1, y1, z1);
            if constexpr (N > 2) {
                out2 = svnmls_z(pg2, x2, y2, z2);
                if constexpr (N > 3) {
                    out3 = svnmls_z(pg3, x3, y3, z3);
                }
            }
        }
    }
};

template <typename ComputeType, SizeType N>
struct MlaMN<ComputeType, N, ARCH::VC> final {
    static void Do(const svbool_t& pg0, const svbool_t &pg1,
        const svbool_t &pg2, const svbool_t &pg3,
        const typename Vec<ComputeType>::Type &x0, const typename Vec<ComputeType>::Type &x1,
        const typename Vec<ComputeType>::Type &x2, const typename Vec<ComputeType>::Type &x3,
        const typename Vec<ComputeType>::Type &y0, const typename Vec<ComputeType>::Type &y1,
        const typename Vec<ComputeType>::Type &y2, const typename Vec<ComputeType>::Type &y3,
        const typename Vec<ComputeType>::Type &z0, const typename Vec<ComputeType>::Type &z1,
        const typename Vec<ComputeType>::Type &z2, const typename Vec<ComputeType>::Type &z3,
        typename Vec<ComputeType>::Type &out0, typename Vec<ComputeType>::Type &out1,
        typename Vec<ComputeType>::Type &out2, typename Vec<ComputeType>::Type &out3) noexcept
    {
        static_assert(N <= 4, "Pseudo unroll by 4 is maximum");
        out0 = svmla_m(pg0, x0, y0, z0);
        if constexpr (N > 1) {
            out1 = svmla_m(pg1, x1, y1, z1);
            if constexpr (N > 2) {
                out2 = svmla_m(pg2, x2, y2, z2);
                if constexpr (N > 3) {
                    out3 = svmla_m(pg3, x3, y3, z3);
                }
            }
        }
    }
};

template <typename ComputeType, SizeType N>
struct MlaXN<ComputeType, N, ARCH::VC> final {
    static void Do(const svbool_t& pg0, const svbool_t &pg1,
        const svbool_t &pg2, const svbool_t &pg3,
        const typename Vec<ComputeType>::Type &x0, const typename Vec<ComputeType>::Type &x1,
        const typename Vec<ComputeType>::Type &x2, const typename Vec<ComputeType>::Type &x3,
        const typename Vec<ComputeType>::Type &y0, const typename Vec<ComputeType>::Type &y1,
        const typename Vec<ComputeType>::Type &y2, const typename Vec<ComputeType>::Type &y3,
        const typename Vec<ComputeType>::Type &z0, const typename Vec<ComputeType>::Type &z1,
        const typename Vec<ComputeType>::Type &z2, const typename Vec<ComputeType>::Type &z3,
        typename Vec<ComputeType>::Type &out0, typename Vec<ComputeType>::Type &out1,
        typename Vec<ComputeType>::Type &out2, typename Vec<ComputeType>::Type &out3) noexcept
    {
        static_assert(N <= 4, "Pseudo unroll by 4 is maximum");
        out0 = svmla_x(pg0, x0, y0, z0);
        if constexpr (N > 1) {
            out1 = svmla_x(pg1, x1, y1, z1);
            if constexpr (N > 2) {
                out2 = svmla_x(pg2, x2, y2, z2);
                if constexpr (N > 3) {
                    out3 = svmla_x(pg3, x3, y3, z3);
                }
            }
        }
    }
};

template <typename ComputeType, SizeType N>
struct MlaZN final {
    static void Do(const svbool_t& pg0, const svbool_t &pg1,
        const svbool_t &pg2, const svbool_t &pg3,
        const typename Vec<ComputeType>::Type &x0, const typename Vec<ComputeType>::Type &x1,
        const typename Vec<ComputeType>::Type &x2, const typename Vec<ComputeType>::Type &x3,
        const typename Vec<ComputeType>::Type &y0, const typename Vec<ComputeType>::Type &y1,
        const typename Vec<ComputeType>::Type &y2, const typename Vec<ComputeType>::Type &y3,
        const typename Vec<ComputeType>::Type &z0, const typename Vec<ComputeType>::Type &z1,
        const typename Vec<ComputeType>::Type &z2, const typename Vec<ComputeType>::Type &z3,
        typename Vec<ComputeType>::Type &out0, typename Vec<ComputeType>::Type &out1,
        typename Vec<ComputeType>::Type &out2, typename Vec<ComputeType>::Type &out3) noexcept
    {
        static_assert(N <= 4, "Pseudo unroll by 4 is maximum");
        out0 = svmla_z(pg0, x0, y0, z0);
        if constexpr (N > 1) {
            out1 = svmla_z(pg1, x1, y1, z1);
            if constexpr (N > 2) {
                out2 = svmla_z(pg2, x2, y2, z2);
                if constexpr (N > 3) {
                    out3 = svmla_z(pg3, x3, y3, z3);
                }
            }
        }
    }
};

template <typename ComputeType, SizeType N>
struct SubXN<ComputeType, N, ARCH::VC> final {
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
        out0 = svsub_x(pg0, lhs0, rhs0);
        if constexpr (N > 1) {
            out1 = svsub_x(pg1, lhs1, rhs1);
            if constexpr (N > 2) {
                out2 = svsub_x(pg2, lhs2, rhs2);
                if constexpr (N > 3) {
                    out3 = svsub_x(pg3, lhs3, rhs3);
                }
            }
        }
    }
};

template <typename ComputeType, SizeType N>
struct SubZN final {
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
        out0 = svsub_z(pg0, lhs0, rhs0);
        if constexpr (N > 1) {
            out1 = svsub_z(pg1, lhs1, rhs1);
            if constexpr (N > 2) {
                out2 = svsub_z(pg2, lhs2, rhs2);
                if constexpr (N > 3) {
                    out3 = svsub_z(pg3, lhs3, rhs3);
                }
            }
        }
    }
};

template <typename ComputeType, SizeType N>
struct SubMN final {
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
        out0 = svsub_m(pg0, lhs0, rhs0);
        if constexpr (N > 1) {
            out1 = svsub_m(pg1, lhs1, rhs1);
            if constexpr (N > 2) {
                out2 = svsub_m(pg2, lhs2, rhs2);
                if constexpr (N > 3) {
                    out3 = svsub_m(pg3, lhs3, rhs3);
                }
            }
        }
    }
};

template <typename ComputeType, SizeType N>
struct SubnXN final {
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
        out0 = svsub_x(pg0, lhs0, rhs0);
        if constexpr (N > 1) {
            out1 = svsub_x(pg1, lhs1, rhs1);
            if constexpr (N > 2) {
                out2 = svsub_x(pg2, lhs2, rhs2);
                if constexpr (N > 3) {
                    out3 = svsub_x(pg3, lhs3, rhs3);
                }
            }
        }
    }
};

template <typename ComputeType, SizeType N>
struct MadZN final {
    static void Do(const svbool_t& pg0, const svbool_t &pg1,
        const svbool_t &pg2, const svbool_t &pg3,
        typename Vec<ComputeType>::Type &x0, typename Vec<ComputeType>::Type &x1,
        typename Vec<ComputeType>::Type &x2, typename Vec<ComputeType>::Type &x3,
        typename Vec<ComputeType>::Type &y0, typename Vec<ComputeType>::Type &y1,
        typename Vec<ComputeType>::Type &y2, typename Vec<ComputeType>::Type &y3,
        typename Vec<ComputeType>::Type &z0, typename Vec<ComputeType>::Type &z1,
        typename Vec<ComputeType>::Type &z2, typename Vec<ComputeType>::Type &z3,
        typename Vec<ComputeType>::Type &out0, typename Vec<ComputeType>::Type &out1,
        typename Vec<ComputeType>::Type &out2, typename Vec<ComputeType>::Type &out3) noexcept
    {
        static_assert(N <= 4, "Pseudo unroll by 4 is maximum");
        out0 = svmad_z(pg0, x0, y0, z0);
        if constexpr (N > 1) {
            out1 = svmad_z(pg1, x1, y1, z1);
            if constexpr (N > 2) {
                out2 = svmad_z(pg2, x2, y2, z2);
                if constexpr (N > 3) {
                    out3 = svmad_z(pg3, x3, y3, z3);
                }
            }
        }
    }
};

template <typename ComputeType, SizeType N>
struct SqrtXN final {
    static void Do(const svbool_t& pg0, const svbool_t &pg1,
        const svbool_t &pg2, const svbool_t &pg3,
        const typename Vec<ComputeType>::Type &lhs0, const typename Vec<ComputeType>::Type &lhs1,
        const typename Vec<ComputeType>::Type &lhs2, const typename Vec<ComputeType>::Type &lhs3,
        typename Vec<ComputeType>::Type &out0, typename Vec<ComputeType>::Type &out1,
        typename Vec<ComputeType>::Type &out2, typename Vec<ComputeType>::Type &out3) noexcept
    {
        static_assert(N <= 4, "Pseudo unroll by 4 is maximum");
        out0 = svsqrt_x(pg0, lhs0);
        if constexpr (N > 1) {
            out1 = svsqrt_x(pg1, lhs1);
            if constexpr (N > 2) {
                out2 = svsqrt_x(pg2, lhs2);
                if constexpr (N > 3) {
                    out3 = svsqrt_x(pg3, lhs3);
                }
            }
        }
    }
};

} // Vector
} // Detail
} // KuDNN

#endif // KUDNN_VECTOR_ARIMETHIC