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

#include "kernels/softmax/kudnn_softmax_fwd_def.h"

namespace KuDNN {
namespace Detail {
namespace Softmax {

template
void EvaluateLastAxisFWD<COMP_TYPE, IN_TYPE, OUT_TYPE>(
    const IN_TYPE *src, OUT_TYPE *dst, SizeType axisSize,
    SizeType outerSize, AlgorithmKind algorithm) noexcept;

} // Softmax
} // Detail
} // KuDNN
