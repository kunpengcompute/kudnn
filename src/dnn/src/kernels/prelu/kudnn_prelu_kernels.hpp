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

#ifndef KUDNN_PRELU_KERNELS_HPP
#define KUDNN_PRELU_KERNELS_HPP

#include "kudnn.hpp"

namespace KuDNN {
namespace Detail {
namespace PReLU {

// FWD

template <typename CompType, typename SrcType, typename WeightsType, typename OutType>
void KernelNonSharedAxesLoad(const Shape &batchSplitDims, const SrcType *src, const WeightsType *wei,
    OutType *dst) noexcept;

template <typename CompType, typename SrcType, typename WeightsType, typename OutType>
void KernelNonSharedAxesDup(const Shape &batchSplitDims, const SrcType *src, const WeightsType *wei,
    OutType *dst) noexcept;

template <typename CompType, typename SrcType, typename WeightsType, typename OutType>
void KernelSharedAxesLoad(const Shape &batchSplitDims, const SrcType *src, const WeightsType *wei, OutType *dst,
    const Shape &reorderedStridesSrc, const Shape &reorderedStridesWei, const Shape &reorderedStridesDst) noexcept;

template <typename CompType, typename SrcType, typename WeightsType, typename OutType>
void KernelSharedAxesDup(const Shape &batchSplitDims, const SrcType *src, const WeightsType *wei, OutType *dst,
    const Shape &reorderedStridesSrc, const Shape &reorderedStridesWei, const Shape &reorderedStridesDst) noexcept;

template <typename CompType, typename SrcType, typename WeightsType, typename OutType>
void KernelGeneric(const Shape &batchSplitDims, const SrcType *src, const WeightsType *wei, OutType *dst,
    const Shape &reorderedStridesSrc, const Shape &reorderedStridesWei, const Shape &reorderedStridesDst) noexcept;

// BWD

template <typename CompType, typename SrcType, typename WeightsType, typename OutType>
void KernelChannelShared(const Shape &batchSplitDims, const SrcType *src, SrcType *diffSrc, const WeightsType *wei,
                         WeightsType *diffWeights, const OutType *diffDst, const Shape &reorderedDims) noexcept;

template <typename CompType, typename SrcType, typename WeightsType, typename OutType,
    typename LoadWei, typename UpdDiffWei>
void KernelSharedAxes(const Shape &batchSplitDims, const SrcType *src, SrcType *diffSrc, const WeightsType *wei,
                      WeightsType *diffWeights, const OutType *diffDst, const Shape &reorderedStridesSrc,
                      const Shape &reorderedStridesWei, const Shape &reorderedWeiDims,
                      const Shape &reorderedStridesDst, const Shape &reorderedDims,
                      LoadWei &&lwei, UpdDiffWei &&udwei) noexcept;

template <typename CompType, typename SrcType, typename WeightsType, typename OutType>
void KernelWholeTensor(const Shape &batchSplitDims, const SrcType *src, SrcType *diffSrc, const WeightsType *wei,
                       WeightsType *diffWeights, const OutType *diffDst) noexcept;
} // PReLU
} // Detail
} // KuDNN

#endif //  KUDNN_PRELU_KERNELS_HPP
