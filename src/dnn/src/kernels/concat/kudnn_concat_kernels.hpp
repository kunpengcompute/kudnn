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

#ifndef KUDNN_CONCAT_KERNELS_HPP
#define KUDNN_CONCAT_KERNELS_HPP

#include <vector>

#include "kudnn.hpp"

namespace KuDNN {
namespace Detail {
namespace Concat {

template <typename CompType, typename InType, typename OutType>
void EvaluateConcat(const std::vector<Shape> &concatSrcDims, const Shape &concatDstDim,
    const void **typedSrc, OutType *typedDst, const SizeType dim, const SizeType nSrc) noexcept;

} // Concat
} // Detail
} // KuDNN

#endif // KUDNN_CONCAT_KERNELS_HPP