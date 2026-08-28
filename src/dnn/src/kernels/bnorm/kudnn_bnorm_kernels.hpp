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

#ifndef KUDNN_BNORM_KERNELS_HPP
#define KUDNN_BNORM_KERNELS_HPP

#include "kudnn.hpp"

namespace KuDNN {
namespace Detail {
namespace BNorm {

// FWD
template<typename CompType>
void EvaluateMulFactor(CompType *mulFactor, const CompType *variance, const CompType *scale,
    NormalizationFlags flags, SizeType channelSize, const CompType eps);

template <typename CompType, typename InType, typename OutType>
void EvaluateMeanCNCrossAxis(const Shape &loopDims, const InType *src, CompType *meanCN, SizeType outerSize,
    SizeType channelSize, SizeType innerSize);

template <typename CompType, typename InType, typename OutType>
void EvaluateVarianceCNCrossAxis(const Shape &loopDims, const InType *src, const CompType *mean,
    CompType *varianceCN, SizeType outerSize, SizeType channelSize, SizeType innerSize);

template <typename CompType, typename InType, typename OutType>
void EvaluateCrossAxis(const Shape &loopDims, const InType *src, OutType *dst,
                       const CompType *shift, const CompType *mean, const CompType *mulFactor,
                       NormalizationFlags flags, SizeType channelSize,
                       SizeType innerSize, std::uint8_t *ws, Propagation pKind) noexcept;

template <typename CompType, typename InType, typename OutType>
void EvaluateBNormAXB(const InType *src, OutType *dst, const CompType *shift, const CompType *mean,
    const CompType *mulFactor, NormalizationFlags flags, SizeType channelSize, SizeType loopRange,
    std::uint8_t *ws, Propagation pKind);

template <typename CompType, typename InType, typename OutType>
void EvaluateMeanCNFwdAXB(const InType *src, CompType *meanCN, SizeType channelSize, SizeType loopRange);

template <typename CompType, typename InType, typename OutType>
void EvaluateVarianceCNFwdAXB(const InType *src, const CompType *mean, CompType *varianceCN,
    SizeType channelSize, SizeType loopRange);

template <typename CompType>
void MergeMeanVar(const CompType *meanVarCN, CompType *meanVar, SizeType channelSize, SizeType sumSize,
    SizeType loopRange, SizeType loopCh);

template <typename InType>
SizeType GetSvlen();

// BWD
template <typename CompType, typename InType, typename OutType>
void EvaluateStatsCrossAxisBwd(SizeType outerSize, const InType *src, const OutType *diffDst,
    const CompType *mean, const CompType *var, CompType *diffScale, CompType *diffShift, CompType eps,
    SizeType axisSize, SizeType channelSize, SizeType innerSize, const std::uint8_t *ws) noexcept;

template <typename CompType, typename InType, typename OutType>
void EvaluateCrossAxisBwd(SizeType axisSize, const InType *src, const OutType *diffDst,
    const CompType *mean, const CompType *var, const CompType *scale, InType *diffSrc,
    CompType *diffScale, CompType *diffShift, const CompType eps, NormalizationFlags flags,
    SizeType outerSize, SizeType channelSize, SizeType innerSize, const std::uint8_t *ws) noexcept;

} // BNorm
} // Detail
} // KuDNN

#endif //  KUDNN_BNORM_KERNELS_HPP
