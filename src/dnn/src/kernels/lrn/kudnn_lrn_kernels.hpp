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

#ifndef KUDNN_LRN_KERNELS_HPP
#define KUDNN_LRN_KERNELS_HPP

#include "kudnn.hpp"

namespace KuDNN {
namespace Detail {
namespace LRN {

// FWD

template <typename CompType, typename InType, typename OutType>
void LocalResponseNormGenericAbxFWD(const InType *src, OutType *dst, float alpha, float beta, float k,
    SizeType localSize, const Shape &start, const Shape &end, const Shape &stridesSrc, const Shape &stridesDst,
    const Shape reorderedDimsSrc, const SizeType numDims,
    LRNormAlgorithmKind algorithm) noexcept;

template <typename CompType, typename InType, typename OutType>
void LocalResponseNormGenericAxbFWD(const InType *src, OutType *dst, float alpha, float beta, float k,
    SizeType localSize, const Shape &start, const Shape &end, const Shape &stridesSrc, const Shape &stridesDst,
    const Shape reorderedDimsSrc, const SizeType numDims,
    LRNormAlgorithmKind algorithm) noexcept;

// Bwd
template <typename CompType, typename InType, typename OutType>
void LocalResponseNormGenericAbxBWD(const InType *src, const OutType *diffDst, InType *diffSrc, float alpha, float beta,
    float k, SizeType localSize, const Shape &start, const Shape &end,
    const Shape &stridesSrc, const Shape &stridesDiffDst, const Shape &stridesDiffSrc, const Shape dimsSrc,
    const SizeType numDims, LRNormAlgorithmKind algorithm) noexcept;

template <typename CompType, typename InType, typename OutType>
void LocalResponseNormGenericAxbBWD(const InType *src, const OutType *diffDst, InType *diffSrc, float alpha, float beta,
    float k, SizeType localSize, const Shape &start, const Shape &end,
    const Shape &stridesSrc, const Shape &stridesDiffDst, const Shape &stridesDiffSrc, const Shape dimsSrc,
    const SizeType numDims, LRNormAlgorithmKind algorithm) noexcept;

} // LRN
} // Detail
} // KuDNN

#endif //  KUDNN_LRN_KERNELS_HPP
