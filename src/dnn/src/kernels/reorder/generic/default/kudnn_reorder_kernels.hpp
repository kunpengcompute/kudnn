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

#ifndef KUDNN_REORDER_KERNELS_HPP
#define KUDNN_REORDER_KERNELS_HPP

#include "kudnn.hpp"

namespace KuDNN {
namespace Detail {
namespace Reorder {

template <typename InType, typename OutType>
void ReorderDefault(const Shape &start, const Shape &end, const void *typedSrcDefault, void *typedDstDefault,
    const Shape &reorderedStridesSrc, const Shape &reorderedStridesDst, float alpha) noexcept;

} // Reorder
} // Detail
} // KuDNN

#endif //  KUDNN_REORDER_KERNELS_HPP
