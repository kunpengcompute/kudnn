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

#include "kernels/lrn/kudnn_lrn_def.hpp"

namespace KuDNN {
namespace Detail {
namespace LRN {

template
void LocalResponseNormGenericAbxFWD<COMP_TYPE, IN_TYPE, OUT_TYPE>(
    const IN_TYPE *src, OUT_TYPE *dst, float alpha, float beta, float k, SizeType localSize,
    const Shape &start, const Shape &end,
    const Shape &stridesSrc, const Shape &stridesDst, const Shape reorderedDimsSrc, const SizeType numDims,
    LRNormAlgorithmKind algorithm) noexcept;

template
void LocalResponseNormGenericAxbFWD<COMP_TYPE, IN_TYPE, OUT_TYPE>(
    const IN_TYPE *src, OUT_TYPE *dst, float alpha, float beta, float k, SizeType localSize,
    const Shape &start, const Shape &end,
    const Shape &stridesSrc, const Shape &stridesDst, const Shape reorderedDimsSrc, const SizeType numDims,
    LRNormAlgorithmKind algorithm) noexcept;


template
void LocalResponseNormGenericAbxBWD<COMP_TYPE, IN_TYPE, OUT_TYPE>(
    const IN_TYPE *src, const OUT_TYPE *diffDst, IN_TYPE *diffSrc, float alpha, float beta,
    float k, SizeType localSize, const Shape &start, const Shape &end,
    const Shape &stridesSrc, const Shape &stridesDiffDst, const Shape &stridesDiffSrc,
    const Shape reorderedDimsSrc, const SizeType numDims, LRNormAlgorithmKind algorithm) noexcept;

template
void LocalResponseNormGenericAxbBWD<COMP_TYPE, IN_TYPE, OUT_TYPE>(
    const IN_TYPE *src, const OUT_TYPE *diffDst, IN_TYPE *diffSrc, float alpha, float beta,
    float k, SizeType localSize, const Shape &start, const Shape &end,
    const Shape &stridesSrc, const Shape &stridesDiffDst, const Shape &stridesDiffSrc,
    const Shape reorderedDimsSrc, const SizeType numDims, LRNormAlgorithmKind algorithm) noexcept;


} // LRN
} // Detail
} // KuDNN
