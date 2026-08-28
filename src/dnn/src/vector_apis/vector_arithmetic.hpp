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
#include <arm_neon.h>

#include "internal_funcs.hpp"

namespace KuDNN {
namespace Detail {
namespace Vector {

template <typename ComputeType, SizeType N, ARCH Arch = ARCH::VC>
struct AddXN final {
    static void Do(const svbool_t& pg0, const svbool_t &pg1, const svbool_t &pg2, const svbool_t &pg3,
        const typename Vec<ComputeType, Arch>::Type &lhs0, const typename Vec<ComputeType, Arch>::Type &lhs1,
        const typename Vec<ComputeType, Arch>::Type &lhs2, const typename Vec<ComputeType, Arch>::Type &lhs3,
        const typename Vec<ComputeType, Arch>::Type &rhs0, const typename Vec<ComputeType, Arch>::Type &rhs1,
        const typename Vec<ComputeType, Arch>::Type &rhs2, const typename Vec<ComputeType, Arch>::Type &rhs3,
        typename Vec<ComputeType, Arch>::Type &out0, typename Vec<ComputeType, Arch>::Type &out1,
        typename Vec<ComputeType, Arch>::Type &out2, typename Vec<ComputeType, Arch>::Type &out3) noexcept;
};

template <typename ComputeType, SizeType N, ARCH Arch = ARCH::VC>
struct DivXN final {
    static void Do(const svbool_t& pg0, const svbool_t &pg1,
        const svbool_t &pg2, const svbool_t &pg3,
        const typename Vec<ComputeType, Arch>::Type &lhs0, const typename Vec<ComputeType, Arch>::Type &lhs1,
        const typename Vec<ComputeType, Arch>::Type &lhs2, const typename Vec<ComputeType, Arch>::Type &lhs3,
        const typename Vec<ComputeType, Arch>::Type &rhs0, const typename Vec<ComputeType, Arch>::Type &rhs1,
        const typename Vec<ComputeType, Arch>::Type &rhs2, const typename Vec<ComputeType, Arch>::Type &rhs3,
        typename Vec<ComputeType, Arch>::Type &out0, typename Vec<ComputeType, Arch>::Type &out1,
        typename Vec<ComputeType, Arch>::Type &out2, typename Vec<ComputeType, Arch>::Type &out3) noexcept;
};

template <typename ComputeType, SizeType N, ARCH Arch = ARCH::VC>
struct DivMN final {
    static void Do(const svbool_t& pg0, const svbool_t &pg1,
        const svbool_t &pg2, const svbool_t &pg3,
        const typename Vec<ComputeType>::Type &lhs0, const typename Vec<ComputeType>::Type &lhs1,
        const typename Vec<ComputeType>::Type &lhs2, const typename Vec<ComputeType>::Type &lhs3,
        const typename Vec<ComputeType>::Type &rhs0, const typename Vec<ComputeType>::Type &rhs1,
        const typename Vec<ComputeType>::Type &rhs2, const typename Vec<ComputeType>::Type &rhs3,
        typename Vec<ComputeType>::Type &out0, typename Vec<ComputeType>::Type &out1,
        typename Vec<ComputeType>::Type &out2, typename Vec<ComputeType>::Type &out3) noexcept;
};

template <typename ComputeType, SizeType N, ARCH Arch = ARCH::VC>
struct MulMaskXN final {
    static void Do(const typename MaskVec<ComputeType, Arch>::Type &pg0,
        const typename MaskVec<ComputeType, Arch>::Type &pg1,
        const typename MaskVec<ComputeType, Arch>::Type &pg2, const typename MaskVec<ComputeType, Arch>::Type &pg3,
        const typename Vec<ComputeType, Arch>::Type &lhs0, const typename Vec<ComputeType, Arch>::Type &lhs1,
        const typename Vec<ComputeType, Arch>::Type &lhs2, const typename Vec<ComputeType, Arch>::Type &lhs3,
        const typename Vec<ComputeType, Arch>::Type &rhs0, const typename Vec<ComputeType, Arch>::Type &rhs1,
        const typename Vec<ComputeType, Arch>::Type &rhs2, const typename Vec<ComputeType, Arch>::Type &rhs3,
        typename Vec<ComputeType, Arch>::Type &out0, typename Vec<ComputeType, Arch>::Type &out1,
        typename Vec<ComputeType, Arch>::Type &out2, typename Vec<ComputeType, Arch>::Type &out3) noexcept;
};

template <typename ComputeType, SizeType N, ARCH Arch = ARCH::VC>
struct SubXN final {
    static void Do(const svbool_t& pg0, const svbool_t &pg1,
        const svbool_t &pg2, const svbool_t &pg3,
        const typename Vec<ComputeType, Arch>::Type &lhs0, const typename Vec<ComputeType, Arch>::Type &lhs1,
        const typename Vec<ComputeType, Arch>::Type &lhs2, const typename Vec<ComputeType, Arch>::Type &lhs3,
        const typename Vec<ComputeType, Arch>::Type &rhs0, const typename Vec<ComputeType, Arch>::Type &rhs1,
        const typename Vec<ComputeType, Arch>::Type &rhs2, const typename Vec<ComputeType, Arch>::Type &rhs3,
        typename Vec<ComputeType, Arch>::Type &out0, typename Vec<ComputeType, Arch>::Type &out1,
        typename Vec<ComputeType, Arch>::Type &out2, typename Vec<ComputeType, Arch>::Type &out3) noexcept;
};

template <typename ComputeType, SizeType N, ARCH Arch = ARCH::VC>
struct MulXN final {
    static void Do(const svbool_t& pg0, const svbool_t &pg1,
        const svbool_t &pg2, const svbool_t &pg3,
        const typename Vec<ComputeType, Arch>::Type &lhs0, const typename Vec<ComputeType, Arch>::Type &lhs1,
        const typename Vec<ComputeType, Arch>::Type &lhs2, const typename Vec<ComputeType, Arch>::Type &lhs3,
        const typename Vec<ComputeType, Arch>::Type &rhs0, const typename Vec<ComputeType, Arch>::Type &rhs1,
        const typename Vec<ComputeType, Arch>::Type &rhs2, const typename Vec<ComputeType, Arch>::Type &rhs3,
        typename Vec<ComputeType, Arch>::Type &out0, typename Vec<ComputeType, Arch>::Type &out1,
        typename Vec<ComputeType, Arch>::Type &out2, typename Vec<ComputeType, Arch>::Type &out3) noexcept;
};

template <typename ComputeType, SizeType N, ARCH Arch = ARCH::VC>
struct MulnXN final {
    static void Do(const svbool_t& pg0, const svbool_t &pg1,
        const svbool_t &pg2, const svbool_t &pg3,
        const typename Vec<ComputeType, Arch>::Type &lhs0, const typename Vec<ComputeType, Arch>::Type &lhs1,
        const typename Vec<ComputeType, Arch>::Type &lhs2, const typename Vec<ComputeType, Arch>::Type &lhs3,
        const ComputeType &rhs0, const ComputeType &rhs1,
        const ComputeType &rhs2, const ComputeType &rhs3,
        typename Vec<ComputeType, Arch>::Type &out0, typename Vec<ComputeType, Arch>::Type &out1,
        typename Vec<ComputeType, Arch>::Type &out2, typename Vec<ComputeType, Arch>::Type &out3) noexcept;
};

template <typename ComputeType, SizeType N, ARCH Arch = ARCH::VC>
struct MlaMN final {
    static void Do(const svbool_t& pg0, const svbool_t &pg1,
        const svbool_t &pg2, const svbool_t &pg3,
        const typename Vec<ComputeType, Arch>::Type &x0, const typename Vec<ComputeType, Arch>::Type &x1,
        const typename Vec<ComputeType, Arch>::Type &x2, const typename Vec<ComputeType, Arch>::Type &x3,
        const typename Vec<ComputeType, Arch>::Type &y0, const typename Vec<ComputeType, Arch>::Type &y1,
        const typename Vec<ComputeType, Arch>::Type &y2, const typename Vec<ComputeType, Arch>::Type &y3,
        const typename Vec<ComputeType, Arch>::Type &z0, const typename Vec<ComputeType, Arch>::Type &z1,
        const typename Vec<ComputeType, Arch>::Type &z2, const typename Vec<ComputeType, Arch>::Type &z3,
        typename Vec<ComputeType, Arch>::Type &out0, typename Vec<ComputeType, Arch>::Type &out1,
        typename Vec<ComputeType, Arch>::Type &out2, typename Vec<ComputeType, Arch>::Type &out3) noexcept;
};

template <typename ComputeType, SizeType N, ARCH Arch = ARCH::VC>
struct MlaXN final {
    static void Do(const svbool_t& pg0, const svbool_t &pg1,
        const svbool_t &pg2, const svbool_t &pg3,
        const typename Vec<ComputeType, Arch>::Type &x0, const typename Vec<ComputeType, Arch>::Type &x1,
        const typename Vec<ComputeType, Arch>::Type &x2, const typename Vec<ComputeType, Arch>::Type &x3,
        const typename Vec<ComputeType, Arch>::Type &y0, const typename Vec<ComputeType, Arch>::Type &y1,
        const typename Vec<ComputeType, Arch>::Type &y2, const typename Vec<ComputeType, Arch>::Type &y3,
        const typename Vec<ComputeType, Arch>::Type &z0, const typename Vec<ComputeType, Arch>::Type &z1,
        const typename Vec<ComputeType, Arch>::Type &z2, const typename Vec<ComputeType, Arch>::Type &z3,
        typename Vec<ComputeType, Arch>::Type &out0, typename Vec<ComputeType, Arch>::Type &out1,
        typename Vec<ComputeType, Arch>::Type &out2, typename Vec<ComputeType, Arch>::Type &out3) noexcept;
};

template <typename ComputeType, SizeType N, ARCH Arch = ARCH::VC>
struct MulMN final {
    static void Do(const svbool_t& pg0, const svbool_t &pg1,
        const svbool_t &pg2, const svbool_t &pg3,
        const typename Vec<ComputeType, Arch>::Type &lhs0, const typename Vec<ComputeType, Arch>::Type &lhs1,
        const typename Vec<ComputeType, Arch>::Type &lhs2, const typename Vec<ComputeType, Arch>::Type &lhs3,
        const typename Vec<ComputeType, Arch>::Type &rhs0, const typename Vec<ComputeType, Arch>::Type &rhs1,
        const typename Vec<ComputeType, Arch>::Type &rhs2, const typename Vec<ComputeType, Arch>::Type &rhs3,
        typename Vec<ComputeType, Arch>::Type &out0, typename Vec<ComputeType, Arch>::Type &out1,
        typename Vec<ComputeType, Arch>::Type &out2, typename Vec<ComputeType, Arch>::Type &out3) noexcept;
};

template <typename ComputeType, SizeType N, ARCH Arch = ARCH::VC>
struct AddMN final {
    static void Do(const svbool_t& pg0, const svbool_t &pg1, const svbool_t &pg2, const svbool_t &pg3,
        const typename Vec<ComputeType, Arch>::Type &lhs0, const typename Vec<ComputeType, Arch>::Type &lhs1,
        const typename Vec<ComputeType, Arch>::Type &lhs2, const typename Vec<ComputeType, Arch>::Type &lhs3,
        const typename Vec<ComputeType, Arch>::Type &rhs0, const typename Vec<ComputeType, Arch>::Type &rhs1,
        const typename Vec<ComputeType, Arch>::Type &rhs2, const typename Vec<ComputeType, Arch>::Type &rhs3,
        typename Vec<ComputeType, Arch>::Type &out0, typename Vec<ComputeType, Arch>::Type &out1,
        typename Vec<ComputeType, Arch>::Type &out2, typename Vec<ComputeType, Arch>::Type &out3) noexcept;
};

} // Vector
} // Detail
} // KuDNN

#endif // KUDNN_VECTOR_ARIMETHIC