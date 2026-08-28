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

#include "kernels/bnorm/kudnn_bnorm_fwd_def_axb.hpp"
#include "kernels/bnorm/kudnn_bnorm_fwd_def_cross_axis.hpp"

namespace KuDNN {
namespace Detail {
namespace BNorm {

template
void EvaluateMulFactor<float>(float *mulFactor, const float *variance, const float *scale,
    NormalizationFlags flags, SizeType channelSize, const float eps);

template
void EvaluateMeanCNCrossAxis<COMP_TYPE, IN_TYPE, OUT_TYPE>(const Shape &loopDims, const IN_TYPE *src,
    COMP_TYPE *meanCN, SizeType outerSize, SizeType channelSize, SizeType innerSize);

template
void EvaluateVarianceCNCrossAxis<COMP_TYPE, IN_TYPE, OUT_TYPE>(const Shape &loopDims, const IN_TYPE *src,
    const COMP_TYPE *mean, COMP_TYPE *varianceCN, SizeType outerSize, SizeType channelSize, SizeType innerSize);

template
void EvaluateCrossAxis<COMP_TYPE, IN_TYPE, OUT_TYPE>(const Shape &loopDims, const IN_TYPE *src, OUT_TYPE *dst,
    const COMP_TYPE *shift, const COMP_TYPE *mean, const COMP_TYPE *mulFactor,
    NormalizationFlags flags, SizeType channelSize,
    SizeType innerSize, std::uint8_t *ws, Propagation pKind) noexcept;

template
void EvaluateBNormAXB<COMP_TYPE, IN_TYPE, OUT_TYPE>(const IN_TYPE *src, OUT_TYPE *dst, const COMP_TYPE *shift,
    const COMP_TYPE *mean, const COMP_TYPE *mulFactor, NormalizationFlags flags,
    SizeType channelSize, SizeType loopRange, std::uint8_t *ws, Propagation pKind);

template
void EvaluateMeanCNFwdAXB<COMP_TYPE, IN_TYPE, OUT_TYPE>(const IN_TYPE *src, COMP_TYPE *meanCN,
    SizeType channelSize, SizeType loopRange);

template
void EvaluateVarianceCNFwdAXB<COMP_TYPE, IN_TYPE, OUT_TYPE>(const IN_TYPE *src, const COMP_TYPE *mean,
    COMP_TYPE *varianceCN, SizeType channelSize, SizeType loopRange);

template
void MergeMeanVar<float>(const float *meanVarCN, float *meanVar, SizeType channelSize, SizeType sumSize,
    SizeType loopRange, SizeType loopCh);

template
SizeType GetSvlen<float>();

} // namespace BNorm
} // namespace Detail
} // namespace KuDNN
