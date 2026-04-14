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

#ifndef KUDNN_GNORM_KERNELS_HPP
#define KUDNN_GNORM_KERNELS_HPP

#include "kudnn.hpp"

namespace KuDNN {
namespace Detail {
namespace GNorm {

// FWD
template <typename CompType, typename InType, typename OutType, typename ScaleShiftType>
void EvaluateCrossAxis(SizeType outerSize, const InType *src, OutType *dst, SizeType groupSize,
                       const ScaleShiftType *scale, const ScaleShiftType *shift, float *mean, float *variance,
                       bool saveStats, float eps, NormalizationFlags flags, SizeType axisSize,
                       SizeType innerSize) noexcept;

template <typename CompType, typename InType, typename OutType, typename ScaleShiftType>
void EvaluateLastAxis(SizeType outerSize, SizeType batchSize, const InType *src, OutType *dst, SizeType groupSize,
    const ScaleShiftType *scale, const ScaleShiftType *shift, float *mean, float *variance,
    bool saveStats, float eps, NormalizationFlags flags, SizeType axisSize) noexcept;

// BWD
template <typename CompType, typename InType, typename OutType, typename ScaleShiftType>
void EvaluteAbx(SizeType batchNum, SizeType outerSize, const InType *src, const float *mean, const float *variance,
    const OutType *diffDst, const ScaleShiftType *scale, InType *diffSrc, float *diffScale, float *diffShift,
    float eps, NormalizationFlags flags, SizeType channelSize, SizeType groupSize) noexcept;

template <typename CompType, typename InType, typename OutType, typename ScaleShiftType>
void EvaluteAxb(SizeType batchNum, SizeType outerSize, const InType *src, const float *mean, const float *variance,
    const OutType *diffDst, const ScaleShiftType *scale, InType *diffSrc, float *diffScale, float *diffShift,
    float eps, NormalizationFlags flags, SizeType channelSize, SizeType groupSize) noexcept;

} // GNorm
} // Detail
} // KuDNN

#endif //  KUDNN_GNORM_KERNELS_HPP
