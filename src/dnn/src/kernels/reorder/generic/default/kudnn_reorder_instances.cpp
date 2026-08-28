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

#include "kernels/reorder/generic/default/kudnn_reorder_def.hpp"

namespace KuDNN {
namespace Detail {
namespace Reorder {

template
void ReorderDefault<IN_TYPE, OUT_TYPE>(const Shape &start, const Shape &end, const void *typedSrcDefault,
    void *typedDstDefault, const Shape &reorderedStridesSrc, const Shape &reorderedStridesDst, float alpha) noexcept;

} // Reorder
} // Detail
} // KuDNN
