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

#ifndef KUDNN_VECTOR_COMPARE
#define KUDNN_VECTOR_COMPARE

#include <arm_sve.h>

#include "internal_funcs.hpp"

namespace KuDNN {
namespace Detail {
namespace Vector {

// Logical operation

template <typename ComputeType, SizeType N>
struct EqN final {
    static void Do(const svbool_t &pg0, const svbool_t &pg1, const svbool_t &pg2, const svbool_t &pg3,
                   const typename Vec<ComputeType>::Type &lhs0, const typename Vec<ComputeType>::Type &lhs1,
                   const typename Vec<ComputeType>::Type &lhs2, const typename Vec<ComputeType>::Type &lhs3,
                   const typename Vec<ComputeType>::Type &rhs0, const typename Vec<ComputeType>::Type &rhs1,
                   const typename Vec<ComputeType>::Type &rhs2, const typename Vec<ComputeType>::Type &rhs3,
                   svbool_t &out0, svbool_t &out1, svbool_t &out2, svbool_t &out3) noexcept
    {
        static_assert(N <= 4, "Pseudo unroll by 4 is maximum");
        out0 = svcmpeq(pg0, lhs0, rhs0);
        if constexpr (N > 1) {
            out1 = svcmpeq(pg1, lhs1, rhs1);
            if constexpr (N > 2) {
                out2 = svcmpeq(pg2, lhs2, rhs2);
                if constexpr (N > 3) {
                    out3 = svcmpeq(pg3, lhs3, rhs3);
                }
            }
        }
    }
};

template <typename ComputeType, SizeType N>
struct GeN final {
    static void Do(const svbool_t &pg0, const svbool_t &pg1, const svbool_t &pg2, const svbool_t &pg3,
                   const typename Vec<ComputeType>::Type &lhs0, const typename Vec<ComputeType>::Type &lhs1,
                   const typename Vec<ComputeType>::Type &lhs2, const typename Vec<ComputeType>::Type &lhs3,
                   const typename Vec<ComputeType>::Type &rhs0, const typename Vec<ComputeType>::Type &rhs1,
                   const typename Vec<ComputeType>::Type &rhs2, const typename Vec<ComputeType>::Type &rhs3,
                   svbool_t &out0, svbool_t &out1, svbool_t &out2, svbool_t &out3) noexcept
    {
        static_assert(N <= 4, "Pseudo unroll by 4 is maximum");
        out0 = svcmpge(pg0, lhs0, rhs0);
        if constexpr (N > 1) {
            out1 = svcmpge(pg1, lhs1, rhs1);
            if constexpr (N > 2) {
                out2 = svcmpge(pg2, lhs2, rhs2);
                if constexpr (N > 3) {
                    out3 = svcmpge(pg3, lhs3, rhs3);
                }
            }
        }
    }
};

template <typename ComputeType, SizeType N>
struct GtN final {
    static void Do(const svbool_t &pg0, const svbool_t &pg1, const svbool_t &pg2, const svbool_t &pg3,
                   const typename Vec<ComputeType>::Type &lhs0, const typename Vec<ComputeType>::Type &lhs1,
                   const typename Vec<ComputeType>::Type &lhs2, const typename Vec<ComputeType>::Type &lhs3,
                   const typename Vec<ComputeType>::Type &rhs0, const typename Vec<ComputeType>::Type &rhs1,
                   const typename Vec<ComputeType>::Type &rhs2, const typename Vec<ComputeType>::Type &rhs3,
                   svbool_t &out0, svbool_t &out1, svbool_t &out2, svbool_t &out3) noexcept
    {
        static_assert(N <= 4, "Pseudo unroll by 4 is maximum");
        out0 = svcmpgt(pg0, lhs0, rhs0);
        if constexpr (N > 1) {
            out1 = svcmpgt(pg1, lhs1, rhs1);
            if constexpr (N > 2) {
                out2 = svcmpgt(pg2, lhs2, rhs2);
                if constexpr (N > 3) {
                    out3 = svcmpgt(pg3, lhs3, rhs3);
                }
            }
        }
    }
};

template <typename ComputeType, SizeType N>
struct LeN final {
    static void Do(const svbool_t &pg0, const svbool_t &pg1, const svbool_t &pg2, const svbool_t &pg3,
                   const typename Vec<ComputeType>::Type &lhs0, const typename Vec<ComputeType>::Type &lhs1,
                   const typename Vec<ComputeType>::Type &lhs2, const typename Vec<ComputeType>::Type &lhs3,
                   const typename Vec<ComputeType>::Type &rhs0, const typename Vec<ComputeType>::Type &rhs1,
                   const typename Vec<ComputeType>::Type &rhs2, const typename Vec<ComputeType>::Type &rhs3,
                   svbool_t &out0, svbool_t &out1, svbool_t &out2, svbool_t &out3) noexcept
    {
        static_assert(N <= 4, "Pseudo unroll by 4 is maximum");
        out0 = svcmple(pg0, lhs0, rhs0);
        if constexpr (N > 1) {
            out1 = svcmple(pg1, lhs1, rhs1);
            if constexpr (N > 2) {
                out2 = svcmple(pg2, lhs2, rhs2);
                if constexpr (N > 3) {
                    out3 = svcmple(pg3, lhs3, rhs3);
                }
            }
        }
    }
};

template <typename ComputeType, SizeType N>
struct LtN final {
    static void Do(const svbool_t &pg0, const svbool_t &pg1, const svbool_t &pg2, const svbool_t &pg3,
                   const typename Vec<ComputeType>::Type &lhs0, const typename Vec<ComputeType>::Type &lhs1,
                   const typename Vec<ComputeType>::Type &lhs2, const typename Vec<ComputeType>::Type &lhs3,
                   const typename Vec<ComputeType>::Type &rhs0, const typename Vec<ComputeType>::Type &rhs1,
                   const typename Vec<ComputeType>::Type &rhs2, const typename Vec<ComputeType>::Type &rhs3,
                   svbool_t &out0, svbool_t &out1, svbool_t &out2, svbool_t &out3) noexcept
    {
        static_assert(N <= 4, "Pseudo unroll by 4 is maximum");
        out0 = svcmplt(pg0, lhs0, rhs0);
        if constexpr (N > 1) {
            out1 = svcmplt(pg1, lhs1, rhs1);
            if constexpr (N > 2) {
                out2 = svcmplt(pg2, lhs2, rhs2);
                if constexpr (N > 3) {
                    out3 = svcmplt(pg3, lhs3, rhs3);
                }
            }
        }
    }
};

template <typename ComputeType, SizeType N>
struct NeN final {
    static void Do(const svbool_t &pg0, const svbool_t &pg1, const svbool_t &pg2, const svbool_t &pg3,
                   const typename Vec<ComputeType>::Type &lhs0, const typename Vec<ComputeType>::Type &lhs1,
                   const typename Vec<ComputeType>::Type &lhs2, const typename Vec<ComputeType>::Type &lhs3,
                   const typename Vec<ComputeType>::Type &rhs0, const typename Vec<ComputeType>::Type &rhs1,
                   const typename Vec<ComputeType>::Type &rhs2, const typename Vec<ComputeType>::Type &rhs3,
                   svbool_t &out0, svbool_t &out1, svbool_t &out2, svbool_t &out3) noexcept
    {
        static_assert(N <= 4, "Pseudo unroll by 4 is maximum");
        out0 = svcmpne(pg0, lhs0, rhs0);
        if constexpr (N > 1) {
            out1 = svcmpne(pg1, lhs1, rhs1);
            if constexpr (N > 2) {
                out2 = svcmpne(pg2, lhs2, rhs2);
                if constexpr (N > 3) {
                    out3 = svcmpne(pg3, lhs3, rhs3);
                }
            }
        }
    }
};

template <typename ComputeType, SizeType N>
struct MaxXN final {
    static void Do(const svbool_t &pg0, const svbool_t &pg1, const svbool_t &pg2, const svbool_t &pg3,
                   const typename Vec<ComputeType>::Type &lhs0, const typename Vec<ComputeType>::Type &lhs1,
                   const typename Vec<ComputeType>::Type &lhs2, const typename Vec<ComputeType>::Type &lhs3,
                   const typename Vec<ComputeType>::Type &rhs0, const typename Vec<ComputeType>::Type &rhs1,
                   const typename Vec<ComputeType>::Type &rhs2, const typename Vec<ComputeType>::Type &rhs3,
                   typename Vec<ComputeType>::Type &out0, typename Vec<ComputeType>::Type &out1,
                   typename Vec<ComputeType>::Type &out2, typename Vec<ComputeType>::Type &out3) noexcept
    {
        static_assert(N <= 4, "Pseudo unroll by 4 is maximum");
        out0 = svmax_x(pg0, lhs0, rhs0);
        if constexpr (N > 1) {
            out1 = svmax_x(pg1, lhs1, rhs1);
            if constexpr (N > 2) {
                out2 = svmax_x(pg2, lhs2, rhs2);
                if constexpr (N > 3) {
                    out3 = svmax_x(pg3, lhs3, rhs3);
                }
            }
        }
    }
};

template <typename ComputeType, SizeType N>
struct MaxMN final {
    static void Do(const svbool_t &pg0, const svbool_t &pg1, const svbool_t &pg2, const svbool_t &pg3,
                   const typename Vec<ComputeType>::Type &lhs0, const typename Vec<ComputeType>::Type &lhs1,
                   const typename Vec<ComputeType>::Type &lhs2, const typename Vec<ComputeType>::Type &lhs3,
                   const typename Vec<ComputeType>::Type &rhs0, const typename Vec<ComputeType>::Type &rhs1,
                   const typename Vec<ComputeType>::Type &rhs2, const typename Vec<ComputeType>::Type &rhs3,
                   typename Vec<ComputeType>::Type &out0, typename Vec<ComputeType>::Type &out1,
                   typename Vec<ComputeType>::Type &out2, typename Vec<ComputeType>::Type &out3) noexcept
    {
        static_assert(N <= 4, "Pseudo unroll by 4 is maximum");
        out0 = svmax_m(pg0, lhs0, rhs0);
        if constexpr (N > 1) {
            out1 = svmax_m(pg1, lhs1, rhs1);
            if constexpr (N > 2) {
                out2 = svmax_m(pg2, lhs2, rhs2);
                if constexpr (N > 3) {
                    out3 = svmax_m(pg3, lhs3, rhs3);
                }
            }
        }
    }
};

template <typename ComputeType, SizeType N>
struct MinXN final {
    static void Do(const svbool_t &pg0, const svbool_t &pg1, const svbool_t &pg2, const svbool_t &pg3,
                   const typename Vec<ComputeType>::Type &lhs0, const typename Vec<ComputeType>::Type &lhs1,
                   const typename Vec<ComputeType>::Type &lhs2, const typename Vec<ComputeType>::Type &lhs3,
                   const typename Vec<ComputeType>::Type &rhs0, const typename Vec<ComputeType>::Type &rhs1,
                   const typename Vec<ComputeType>::Type &rhs2, const typename Vec<ComputeType>::Type &rhs3,
                   typename Vec<ComputeType>::Type &out0, typename Vec<ComputeType>::Type &out1,
                   typename Vec<ComputeType>::Type &out2, typename Vec<ComputeType>::Type &out3) noexcept
    {
        static_assert(N <= 4, "Pseudo unroll by 4 is maximum");
        out0 = svmin_x(pg0, lhs0, rhs0);
        if constexpr (N > 1) {
            out1 = svmin_x(pg1, lhs1, rhs1);
            if constexpr (N > 2) {
                out2 = svmin_x(pg2, lhs2, rhs2);
                if constexpr (N > 3) {
                    out3 = svmin_x(pg3, lhs3, rhs3);
                }
            }
        }
    }
};

template <typename ComputeType, SizeType N>
struct MinMN final {
    static void Do(const svbool_t &pg0, const svbool_t &pg1, const svbool_t &pg2, const svbool_t &pg3,
                   const typename Vec<ComputeType>::Type &lhs0, const typename Vec<ComputeType>::Type &lhs1,
                   const typename Vec<ComputeType>::Type &lhs2, const typename Vec<ComputeType>::Type &lhs3,
                   const typename Vec<ComputeType>::Type &rhs0, const typename Vec<ComputeType>::Type &rhs1,
                   const typename Vec<ComputeType>::Type &rhs2, const typename Vec<ComputeType>::Type &rhs3,
                   typename Vec<ComputeType>::Type &out0, typename Vec<ComputeType>::Type &out1,
                   typename Vec<ComputeType>::Type &out2, typename Vec<ComputeType>::Type &out3) noexcept
    {
        static_assert(N <= 4, "Pseudo unroll by 4 is maximum");
        out0 = svmin_m(pg0, lhs0, rhs0);
        if constexpr (N > 1) {
            out1 = svmin_m(pg1, lhs1, rhs1);
            if constexpr (N > 2) {
                out2 = svmin_m(pg2, lhs2, rhs2);
                if constexpr (N > 3) {
                    out3 = svmin_m(pg3, lhs3, rhs3);
                }
            }
        }
    }
};

} // namespace Vector
} // namespace Detail
} // namespace KuDNN

#endif // KUDNN_VECTOR_COMPARE
