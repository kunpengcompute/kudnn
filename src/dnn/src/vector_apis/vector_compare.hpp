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
#include <arm_neon.h>

#include "internal_funcs.hpp"

namespace KuDNN {
namespace Detail {
namespace Vector {

// Logical operation
template <typename ComputeType, SizeType N, ARCH Arch = ARCH::VC>
struct EqN final {
    static void Do(const svbool_t& pg0, const svbool_t &pg1,
        const svbool_t &pg2, const svbool_t &pg3,
        const typename Vec<ComputeType>::Type &lhs0, const typename Vec<ComputeType>::Type &lhs1,
        const typename Vec<ComputeType>::Type &lhs2, const typename Vec<ComputeType>::Type &lhs3,
        const typename Vec<ComputeType>::Type &rhs0, const typename Vec<ComputeType>::Type &rhs1,
        const typename Vec<ComputeType>::Type &rhs2, const typename Vec<ComputeType>::Type &rhs3,
        typename MaskVec<ComputeType, Arch>::Type &out0,
        typename MaskVec<ComputeType, Arch>::Type &out1,
        typename MaskVec<ComputeType, Arch>::Type &out2,
        typename MaskVec<ComputeType, Arch>::Type &out3) noexcept;
};

template <typename ComputeType, SizeType N, ARCH Arch = ARCH::VC>
struct GeN final {
    static void Do(const svbool_t& pg0, const svbool_t &pg1,
        const svbool_t &pg2, const svbool_t &pg3,
        const typename Vec<ComputeType>::Type &lhs0, const typename Vec<ComputeType>::Type &lhs1,
        const typename Vec<ComputeType>::Type &lhs2, const typename Vec<ComputeType>::Type &lhs3,
        const typename Vec<ComputeType>::Type &rhs0, const typename Vec<ComputeType>::Type &rhs1,
        const typename Vec<ComputeType>::Type &rhs2, const typename Vec<ComputeType>::Type &rhs3,
        typename MaskVec<ComputeType, Arch>::Type &out0,
        typename MaskVec<ComputeType, Arch>::Type &out1,
        typename MaskVec<ComputeType, Arch>::Type &out2,
        typename MaskVec<ComputeType, Arch>::Type &out3) noexcept;
};

template <typename ComputeType, SizeType N, ARCH Arch = ARCH::VC>
struct GtN final {
    static void Do(const svbool_t& pg0, const svbool_t &pg1,
        const svbool_t &pg2, const svbool_t &pg3,
        const typename Vec<ComputeType>::Type &lhs0, const typename Vec<ComputeType>::Type &lhs1,
        const typename Vec<ComputeType>::Type &lhs2, const typename Vec<ComputeType>::Type &lhs3,
        const typename Vec<ComputeType>::Type &rhs0, const typename Vec<ComputeType>::Type &rhs1,
        const typename Vec<ComputeType>::Type &rhs2, const typename Vec<ComputeType>::Type &rhs3,
        typename MaskVec<ComputeType, Arch>::Type &out0,
        typename MaskVec<ComputeType, Arch>::Type &out1,
        typename MaskVec<ComputeType, Arch>::Type &out2,
        typename MaskVec<ComputeType, Arch>::Type &out3) noexcept;
};

template <typename ComputeType, SizeType N, ARCH Arch = ARCH::VC>
struct LeN final {
    static void Do(const svbool_t& pg0, const svbool_t &pg1,
        const svbool_t &pg2, const svbool_t &pg3,
        const typename Vec<ComputeType>::Type &lhs0, const typename Vec<ComputeType>::Type &lhs1,
        const typename Vec<ComputeType>::Type &lhs2, const typename Vec<ComputeType>::Type &lhs3,
        const typename Vec<ComputeType>::Type &rhs0, const typename Vec<ComputeType>::Type &rhs1,
        const typename Vec<ComputeType>::Type &rhs2, const typename Vec<ComputeType>::Type &rhs3,
        typename MaskVec<ComputeType, Arch>::Type &out0,
        typename MaskVec<ComputeType, Arch>::Type &out1,
        typename MaskVec<ComputeType, Arch>::Type &out2,
        typename MaskVec<ComputeType, Arch>::Type &out3) noexcept;
};

template <typename ComputeType, SizeType N, ARCH Arch = ARCH::VC>
struct LtN final {
    static void Do(const svbool_t& pg0, const svbool_t &pg1,
        const svbool_t &pg2, const svbool_t &pg3,
        const typename Vec<ComputeType>::Type &lhs0, const typename Vec<ComputeType>::Type &lhs1,
        const typename Vec<ComputeType>::Type &lhs2, const typename Vec<ComputeType>::Type &lhs3,
        const typename Vec<ComputeType>::Type &rhs0, const typename Vec<ComputeType>::Type &rhs1,
        const typename Vec<ComputeType>::Type &rhs2, const typename Vec<ComputeType>::Type &rhs3,
        typename MaskVec<ComputeType, Arch>::Type &out0,
        typename MaskVec<ComputeType, Arch>::Type &out1,
        typename MaskVec<ComputeType, Arch>::Type &out2,
        typename MaskVec<ComputeType, Arch>::Type &out3) noexcept;
};

template <typename ComputeType, SizeType N, ARCH Arch = ARCH::VC>
struct NeN final {
    static void Do(const svbool_t& pg0, const svbool_t &pg1,
        const svbool_t &pg2, const svbool_t &pg3,
        const typename Vec<ComputeType>::Type &lhs0, const typename Vec<ComputeType>::Type &lhs1,
        const typename Vec<ComputeType>::Type &lhs2, const typename Vec<ComputeType>::Type &lhs3,
        const typename Vec<ComputeType>::Type &rhs0, const typename Vec<ComputeType>::Type &rhs1,
        const typename Vec<ComputeType>::Type &rhs2, const typename Vec<ComputeType>::Type &rhs3,
        svbool_t &out0, svbool_t &out1,
        svbool_t &out2, svbool_t &out3) noexcept;
};

template <typename ComputeType, SizeType N, ARCH Arch = ARCH::VC>
struct MaxXN final {
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
struct MaxMN final {
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
struct MinXN final {
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
struct MinMN final {
    static void Do(const svbool_t& pg0, const svbool_t &pg1,
        const svbool_t &pg2, const svbool_t &pg3,
        const typename Vec<ComputeType>::Type &lhs0, const typename Vec<ComputeType>::Type &lhs1,
        const typename Vec<ComputeType>::Type &lhs2, const typename Vec<ComputeType>::Type &lhs3,
        const typename Vec<ComputeType>::Type &rhs0, const typename Vec<ComputeType>::Type &rhs1,
        const typename Vec<ComputeType>::Type &rhs2, const typename Vec<ComputeType>::Type &rhs3,
        typename Vec<ComputeType>::Type &out0, typename Vec<ComputeType>::Type &out1,
        typename Vec<ComputeType>::Type &out2, typename Vec<ComputeType>::Type &out3) noexcept;
};

} // Vector
} // Detail
} // KuDNN

#endif // KUDNN_VECTOR_COMPARE