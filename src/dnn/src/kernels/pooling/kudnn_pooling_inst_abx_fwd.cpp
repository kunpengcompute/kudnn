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

#include "kernels/pooling/kudnn_pooling_abx_fwd_def.hpp"

namespace KuDNN {
namespace Detail {
namespace Pooling {

template
void EvaluateAbxFWD<POOL_TYPE, COMP_TYPE, SRC_TYPE, DST_TYPE, WS_TYPE, USE_WS>(
    const Shape &start, const Shape &end, const SRC_TYPE *typedSrc, DST_TYPE *typedDst,
    WS_TYPE *typedWs, const Shape &initialDims, const Shape &outputDims, const Shape &stridesSrc,
    const Shape &stridesDst, const Shape &kernel, const Shape &strides,
    const Shape &dilation, const Shape &paddingL, SizeType ndims, COMP_TYPE init,
    const std::vector<ReorderLayer> &reorders) noexcept;

} // Pooling
} // Detail
} // KuDNN
