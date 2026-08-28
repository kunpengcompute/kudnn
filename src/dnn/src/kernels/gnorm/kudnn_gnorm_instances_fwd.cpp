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

#include "kernels/gnorm/kudnn_gnorm_cross_fwd_def.hpp"
#include "kernels/gnorm/kudnn_gnorm_last_fwd_def.hpp"

namespace KuDNN {
namespace Detail {
namespace GNorm {

template
void EvaluateCrossAxis<COMP_TYPE, IN_TYPE, OUT_TYPE, SCALE_SHIFT_TYPE>(
    SizeType outerSize, const IN_TYPE *src, OUT_TYPE *dst, SizeType groupSize, const SCALE_SHIFT_TYPE *scale,
    const SCALE_SHIFT_TYPE *shift, float *mean, float *variance, SizeType start, bool saveStats, float eps,
    NormalizationFlags flags, SizeType axisSize, SizeType innerSize) noexcept;

template
void EvaluateLastAxis<COMP_TYPE, IN_TYPE, OUT_TYPE, SCALE_SHIFT_TYPE>(
    SizeType outerSize, SizeType batchSize, const IN_TYPE *src, OUT_TYPE *dst, SizeType groupSize,
    const SCALE_SHIFT_TYPE *scale, const SCALE_SHIFT_TYPE *shift, float *mean, float *variance,
    bool saveStats, float eps, NormalizationFlags flags, SizeType axisSize) noexcept;

} // GNorm
} // Detail
} // KuDNN
