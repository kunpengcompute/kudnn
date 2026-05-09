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

#include "kernels/lnorm/kudnn_lnorm_bwd_last_def.hpp"

namespace KuDNN {
namespace Detail {
namespace LNorm {

template void EvaluateScaleShiftLastAxis<COMP_TYPE, IN_TYPE, OUT_TYPE, SCALE_SHIFT_TYPE>(
    SizeType outerSize, const IN_TYPE *src, const float *mean, const float *variance, const OUT_TYPE *diffDst,
    SCALE_SHIFT_TYPE *diffScale, SCALE_SHIFT_TYPE *diffShift, float *bufScale, float *bufShift, float eps,
    NormalizationFlags flags, SizeType axisSize, SizeType axisSizeConst) noexcept;

template void EvaluateLastAxis<COMP_TYPE, IN_TYPE, OUT_TYPE, SCALE_SHIFT_TYPE>(
    SizeType outerSize, const IN_TYPE *src, const float *mean, const float *variance, const OUT_TYPE *diffDst,
    const SCALE_SHIFT_TYPE *scale, IN_TYPE *diffSrc, float eps, NormalizationFlags flags, SizeType axisSize) noexcept;

template void EvaluateRewriteScaleShift<COMP_TYPE, IN_TYPE, OUT_TYPE, SCALE_SHIFT_TYPE>(
    SCALE_SHIFT_TYPE *diffScale, SCALE_SHIFT_TYPE *diffShift, float *bufScale, float *bufShift,
    NormalizationFlags flags, SizeType axisSize, SizeType numThreads) noexcept;

} // namespace LNorm
} // namespace Detail
} // namespace KuDNN
