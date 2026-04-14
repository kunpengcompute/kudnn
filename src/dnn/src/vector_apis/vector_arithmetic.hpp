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

#ifndef KUDNN_VECTOR_ARITHMETIC
#define KUDNN_VECTOR_ARITHMETIC

#include <arm_sve.h>

#include "internal_funcs.hpp"

namespace KuDNN {
namespace Detail {
namespace Vector {

template <typename ComputeType, SizeType N>
struct AddXN final {
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
struct AddMN final {
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
struct MlaMN final {
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
struct MlaXN final {
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
struct DivXN final {
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
struct DivMN final {
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
struct MulXN final {
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
struct MulnXN final {
    static void Do(const svbool_t& pg0, const svbool_t &pg1,
        const svbool_t &pg2, const svbool_t &pg3,
        const typename Vec<ComputeType>::Type &lhs0, const typename Vec<ComputeType>::Type &lhs1,
        const typename Vec<ComputeType>::Type &lhs2, const typename Vec<ComputeType>::Type &lhs3,
        ComputeType &rhs0, ComputeType &rhs1,
        ComputeType &rhs2, ComputeType &rhs3,
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
struct MulMN final {
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

template <typename ComputeType, SizeType N>
struct SubXN final {
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