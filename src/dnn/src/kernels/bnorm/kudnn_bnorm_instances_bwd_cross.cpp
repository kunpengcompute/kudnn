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

#include "kernels/bnorm/kudnn_bnorm_bwd_def.hpp"

namespace KuDNN {
namespace Detail {
namespace BNorm {

template
void EvaluateStatsCrossAxisBwd<COMP_TYPE, IN_TYPE, OUT_TYPE>(SizeType outerSize, const IN_TYPE *src,
    const OUT_TYPE *diffDst, const COMP_TYPE *mean, const COMP_TYPE *var, COMP_TYPE *diffScale,
    COMP_TYPE *diffShift, COMP_TYPE eps, SizeType axisSize, SizeType channelSize, SizeType innerSize,
    const std::uint8_t *ws) noexcept;

template
void EvaluateCrossAxisBwd<COMP_TYPE, IN_TYPE, OUT_TYPE>(SizeType axisSize, const IN_TYPE *src, const OUT_TYPE *diffDst,
    const COMP_TYPE *mean, const COMP_TYPE *var, const COMP_TYPE *scale, IN_TYPE *diffSrc,
    COMP_TYPE *diffScale, COMP_TYPE *diffShift, const COMP_TYPE eps, NormalizationFlags flags,
    SizeType outerSize, SizeType channelSize, SizeType innerSize, const std::uint8_t *ws) noexcept;

} // namespace BNorm
} // namespace Detail
} // namespace KuDNN
