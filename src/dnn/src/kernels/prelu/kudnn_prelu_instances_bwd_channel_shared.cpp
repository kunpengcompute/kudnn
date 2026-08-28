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

#include <cstdint>

#include "kernels/prelu/kudnn_prelu_bwd_def.hpp"

namespace KuDNN {
namespace Detail {
namespace PReLU {

template
void KernelChannelShared<COMP_TYPE, SRC_TYPE, WEIGHTS_TYPE, OUT_TYPE>(
    const Shape &batchSplitDims, const SRC_TYPE *src, SRC_TYPE *diffSrc, const WEIGHTS_TYPE *wei,
    WEIGHTS_TYPE *diffWeights, const OUT_TYPE *diffDst, const Shape &reorderedDims) noexcept;

} // PReLU
} // Detail
} // KuDNN
