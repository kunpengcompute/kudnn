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

#ifndef KUDNN_GNORM_LAST_FWD_DEF_HPP
#define KUDNN_GNORM_LAST_FWD_DEF_HPP

#include "kudnn.hpp"
#include "vector_wrappers.hpp"
#include "kernels/gnorm/kudnn_gnorm_kernels.hpp"
#include "kernels/gnorm/kudnn_gnorm_last_fwd_def_sve.hpp"
#if USE_NEON
#include "kernels/gnorm/kudnn_gnorm_last_fwd_def_neon.hpp"
#endif

namespace KuDNN {
namespace Detail {
namespace GNorm {

template <typename CompType, typename InType, typename OutType, typename ScaleShiftType>
void EvaluateLastAxis(SizeType outerSize, SizeType batchSize, const InType *src, OutType *dst, SizeType groupSize,
    const ScaleShiftType *scale, const ScaleShiftType *shift, float *mean, float *variance,
    bool saveStats, float eps, NormalizationFlags flags, SizeType axisSize) noexcept
{
#if USE_NEON
    EvaluateLastAxisNeon<CompType, InType, OutType, ScaleShiftType>(outerSize, batchSize, src, dst, groupSize,
        scale, shift, mean, variance, saveStats, eps, flags, axisSize);
#else
    EvaluateLastAxisVC<CompType, InType, OutType, ScaleShiftType>(outerSize, batchSize, src, dst, groupSize,
        scale, shift, mean, variance, saveStats, eps, flags, axisSize);
#endif
}

} // GNorm
} // Detail
} // KuDNN

#endif //  KUDNN_GNORM_FWD_DEF_HPP