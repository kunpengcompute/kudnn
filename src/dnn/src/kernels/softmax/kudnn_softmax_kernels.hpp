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

#ifndef KUDNN_SOFTMAX_KERNELS_HPP
#define KUDNN_SOFTMAX_KERNELS_HPP

#include "kudnn.hpp"

namespace KuDNN {
namespace Detail {
namespace Softmax {

// for FWD

template <typename CompType, typename InType, typename OutType>
void EvaluateLastAxisFWD(const InType *src, OutType *dst, SizeType axisSize, SizeType outerSize,
                         SoftmaxAlgorithmKind algorithm) noexcept;

template <typename CompType, typename InType, typename OutType>
void EvaluateCrossAxisFWD(const InType *src, OutType *dst, SizeType innerSize, SizeType axisSize, SizeType outerSize,
                          SoftmaxAlgorithmKind algorithm) noexcept;

// for BWD

template <typename CompType, typename InType, typename OutType>
void EvaluateLastAxisBWD(const OutType *dst, const OutType *dstDiff, InType *srcDiff, SizeType axisSize,
                         SizeType outerSize, SizeType axisNum, SoftmaxAlgorithmKind algorithm) noexcept;

template <typename CompType, typename InType, typename OutType>
void EvaluateCrossAxisBWD(const OutType *dst, const OutType *dstDiff, InType *srcDiff, SizeType innerSize,
                          SizeType axisSize, SizeType outerSize, SoftmaxAlgorithmKind algorithm) noexcept;

} // namespace Softmax
} // namespace Detail
} // namespace KuDNN

#endif //  KUDNN_SOFTMAX_KERNELS_HPP
