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

#include "kernels/binary/kudnn_binary_def.hpp"

namespace KuDNN {
namespace Detail {
namespace Binary {

template
void CONCAT_MACRO(Evaluate, FUNC_KIND)<COMP_TYPE, SRC0_TYPE, SRC1_TYPE, OUT_TYPE>(
    const SRC0_TYPE *lhs, const SRC1_TYPE *rhs, OUT_TYPE *dst,
    const float scale0, const float scale1, const Shape &dims, const Shape &stridesSrc0,
    const Shape &stridesSrc1, const Shape &stridesDst, bool isSeqKernel) noexcept;

} // Binary
} // Detail
} // KuDNN
