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

#ifndef KUDNN_FAST_REORDER_KERNELS_HPP
#define KUDNN_FAST_REORDER_KERNELS_HPP

#include "kudnn.hpp"

namespace KuDNN {
namespace Detail {
namespace Reorder {

template <typename InType, typename OutType>
void ReorderSameLayouts(const Shape &start, const Shape &end, const InType *typedSrc, OutType *typedDst,
    const Shape &reorderedStridesSrc, const Shape &reorderedStridesDst, float alpha) noexcept;

} // Reorder
} // Detail
} // KuDNN

#endif //  KUDNN_FAST_REORDER_KERNELS_HPP
