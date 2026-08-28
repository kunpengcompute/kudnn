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

#include "kernels/reduction/kudnn_reduction_def.hpp"

namespace KuDNN {
namespace Detail {
namespace Reduction {

template
void CONCAT_MACRO(Evaluate, FUNC_KIND)<COMP_TYPE, IN_TYPE, OUT_TYPE>(
    const Shape &start, const Shape &end, const IN_TYPE *src, OUT_TYPE *dst, float* dstBuf,
    COMP_TYPE acc, ReductionFunction kind, SizeType reduceSize, SizeType idleSize,
    Shape &reorderedDimsSrc, Shape &reorderedDimsDst,
    Shape &reorderedStridesSrc, Shape &reorderedStridesDst, float p, float eps) noexcept;

} // Reduction
} // Detail
} // KuDNN
