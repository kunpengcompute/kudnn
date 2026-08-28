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

#ifndef KUDNN_REDUCTION_KERNELS_HPP
#define KUDNN_REDUCTION_KERNELS_HPP

#include "kudnn.hpp"

namespace KuDNN {
namespace Detail {
namespace Reduction {

template <typename CompType, typename InType, typename OutType>
void EvaluateMin(const Shape &start, const Shape &end, const InType *src, OutType *dst, float* dstBuf,
    CompType acc, ReductionFunction kind, SizeType reduceSize, SizeType idleSize,
    Shape &reorderedDimsSrc, Shape &reorderedDimsDst,
    Shape &reorderedStridesSrc, Shape &reorderedStridesDst, float p, float eps) noexcept;

template <typename CompType, typename InType, typename OutType>
void EvaluateMax(const Shape &start, const Shape &end, const InType *src, OutType *dst, float* dstBuf,
    CompType acc, ReductionFunction kind, SizeType reduceSize, SizeType idleSize,
    Shape &reorderedDimsSrc, Shape &reorderedDimsDst,
    Shape &reorderedStridesSrc, Shape &reorderedStridesDst, float p, float eps) noexcept;

template <typename CompType, typename InType, typename OutType>
void EvaluateMean(const Shape &start, const Shape &end, const InType *src, OutType *dst, float* dstBuf,
    CompType acc, ReductionFunction kind, SizeType reduceSize, SizeType idleSize,
    Shape &reorderedDimsSrc, Shape &reorderedDimsDst,
    Shape &reorderedStridesSrc, Shape &reorderedStridesDst, float p, float eps) noexcept;

template <typename CompType, typename InType, typename OutType>
void EvaluateSum(const Shape &start, const Shape &end, const InType *src, OutType *dst, float* dstBuf,
    CompType acc, ReductionFunction kind, SizeType reduceSize, SizeType idleSize,
    Shape &reorderedDimsSrc, Shape &reorderedDimsDst,
    Shape &reorderedStridesSrc, Shape &reorderedStridesDst, float p, float eps) noexcept;

template <typename CompType, typename InType, typename OutType>
void EvaluateMul(const Shape &start, const Shape &end, const InType *src, OutType *dst, float* dstBuf,
    CompType acc, ReductionFunction kind, SizeType reduceSize, SizeType idleSize,
    Shape &reorderedDimsSrc, Shape &reorderedDimsDst,
    Shape &reorderedStridesSrc, Shape &reorderedStridesDst, float p, float eps) noexcept;

template <typename CompType, typename InType, typename OutType>
void EvaluateNormLp(const Shape &start, const Shape &end, const InType *src, OutType *dst, float* dstBuf,
    CompType acc, ReductionFunction kind, SizeType reduceSize, SizeType idleSize,
    Shape &reorderedDimsSrc, Shape &reorderedDimsDst,
    Shape &reorderedStridesSrc, Shape &reorderedStridesDst, float p, float eps) noexcept;

} // Reduction
} // Detail
} // KuDNN

#endif //  KUDNN_REDUCTION_KERNELS_HPP
