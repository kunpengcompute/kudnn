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

#ifndef KUDNN_BINARY_KERNELS_HPP
#define KUDNN_BINARY_KERNELS_HPP

#include "kudnn.hpp"

namespace KuDNN {
namespace Detail {
namespace Binary {

template <typename CompType, typename Src0Type, typename Src1Type, typename OutType>
void EvaluateAdd(const Src0Type *lhs, const Src1Type *rhs, OutType *dst,
    const float scale0, const float scale1, const Shape &dims, const Shape &stridesSrc0,
    const Shape &stridesSrc1, const Shape &stridesDst, bool isSeqKernel) noexcept;

template <typename CompType, typename Src0Type, typename Src1Type, typename OutType>
void EvaluateDiv(const Src0Type *lhs, const Src1Type *rhs, OutType *dst,
    const float scale0, const float scale1, const Shape &dims, const Shape &stridesSrc0,
    const Shape &stridesSrc1, const Shape &stridesDst, bool isSeqKernel) noexcept;

template <typename CompType, typename Src0Type, typename Src1Type, typename OutType>
void EvaluateMax(const Src0Type *lhs, const Src1Type *rhs, OutType *dst,
    const float scale0, const float scale1, const Shape &dims, const Shape &stridesSrc0,
    const Shape &stridesSrc1, const Shape &stridesDst, bool isSeqKernel) noexcept;

template <typename CompType, typename Src0Type, typename Src1Type, typename OutType>
void EvaluateMin(const Src0Type *lhs, const Src1Type *rhs, OutType *dst,
    const float scale0, const float scale1, const Shape &dims, const Shape &stridesSrc0,
    const Shape &stridesSrc1, const Shape &stridesDst, bool isSeqKernel) noexcept;

template <typename CompType, typename Src0Type, typename Src1Type, typename OutType>
void EvaluateMul(const Src0Type *lhs, const Src1Type *rhs, OutType *dst,
    const float scale0, const float scale1, const Shape &dims, const Shape &stridesSrc0,
    const Shape &stridesSrc1, const Shape &stridesDst, bool isSeqKernel) noexcept;

template <typename CompType, typename Src0Type, typename Src1Type, typename OutType>
void EvaluateSub(const Src0Type *lhs, const Src1Type *rhs, OutType *dst,
    const float scale0, const float scale1, const Shape &dims, const Shape &stridesSrc0,
    const Shape &stridesSrc1, const Shape &stridesDst, bool isSeqKernel) noexcept;

template <typename CompType, typename Src0Type, typename Src1Type, typename OutType>
void EvaluateGe(const Src0Type *lhs, const Src1Type *rhs, OutType *dst,
    const float scale0, const float scale1, const Shape &dims, const Shape &stridesSrc0,
    const Shape &stridesSrc1, const Shape &stridesDst, bool isSeqKernel) noexcept;

template <typename CompType, typename Src0Type, typename Src1Type, typename OutType>
void EvaluateGt(const Src0Type *lhs, const Src1Type *rhs, OutType *dst,
    const float scale0, const float scale1, const Shape &dims, const Shape &stridesSrc0,
    const Shape &stridesSrc1, const Shape &stridesDst, bool isSeqKernel) noexcept;

template <typename CompType, typename Src0Type, typename Src1Type, typename OutType>
void EvaluateLe(const Src0Type *lhs, const Src1Type *rhs, OutType *dst,
    const float scale0, const float scale1, const Shape &dims, const Shape &stridesSrc0,
    const Shape &stridesSrc1, const Shape &stridesDst, bool isSeqKernel) noexcept;

template <typename CompType, typename Src0Type, typename Src1Type, typename OutType>
void EvaluateLt(const Src0Type *lhs, const Src1Type *rhs, OutType *dst,
    const float scale0, const float scale1, const Shape &dims, const Shape &stridesSrc0,
    const Shape &stridesSrc1, const Shape &stridesDst, bool isSeqKernel) noexcept;

template <typename CompType, typename Src0Type, typename Src1Type, typename OutType>
void EvaluateEq(const Src0Type *lhs, const Src1Type *rhs, OutType *dst,
    const float scale0, const float scale1, const Shape &dims, const Shape &stridesSrc0,
    const Shape &stridesSrc1, const Shape &stridesDst, bool isSeqKernel) noexcept;

template <typename CompType, typename Src0Type, typename Src1Type, typename OutType>
void EvaluateNe(const Src0Type *lhs, const Src1Type *rhs, OutType *dst,
    const float scale0, const float scale1, const Shape &dims, const Shape &stridesSrc0,
    const Shape &stridesSrc1, const Shape &stridesDst, bool isSeqKernel) noexcept;

} // Binary
} // Detail
} // KuDNN

#endif //  KUDNN_BINARY_KERNELS_HPP
