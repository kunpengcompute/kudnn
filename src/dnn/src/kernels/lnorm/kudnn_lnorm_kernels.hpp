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
#ifndef KUDNN_LNORM_KERNELS_HPP
#define KUDNN_LNORM_KERNELS_HPP

#include "kudnn.hpp"

namespace KuDNN {
namespace Detail {
namespace LNorm {

// FWD

template <typename CompType, typename InType, typename OutType, typename ScaleShiftType>
void EvaluateLastAxis(SizeType execAcrossAxis, const InType *src, OutType *dst, const ScaleShiftType *scale,
                      const ScaleShiftType *shift, float *mean, float *variance, bool saveStats, float eps,
                      NormalizationFlags flags, SizeType axisSize) noexcept;

template <typename CompType, typename InType, typename OutType, typename ScaleShiftType>
void EvaluateCrossAxis(SizeType outerSize, const InType *src, OutType *dst, const ScaleShiftType *scale,
                       const ScaleShiftType *shift, float *mean, float *variance, bool saveStats, float eps,
                       NormalizationFlags flags, SizeType axisSize, SizeType innerSize) noexcept;

// BWD

template <typename CompType, typename InType, typename OutType, typename ScaleShiftType>
void EvaluateLastAxis(SizeType outerSize, const InType *src, const float *mean, const float *variance,
                      const OutType *diffDst, const ScaleShiftType *scale, InType *diffSrc, float eps,
                      NormalizationFlags flags, SizeType axisSize) noexcept;

template <typename CompType, typename InType, typename OutType, typename ScaleShiftType>
void EvaluateScaleShiftLastAxis(SizeType outerSize, const InType *src, const float *mean, const float *variance,
                                const OutType *diffDst, ScaleShiftType *diffScale, ScaleShiftType *diffShift,
                                float *bufScale, float *bufShift, float eps, NormalizationFlags flags,
                                SizeType axisSize, SizeType axisSizeConst) noexcept;

template <typename CompType, typename InType, typename OutType, typename ScaleShiftType>
void EvaluateCrossAxis(SizeType outerSize, const InType *src, const float *mean, const float *variance,
                       const OutType *diffDst, const ScaleShiftType *scale, InType *diffSrc, float eps,
                       NormalizationFlags flags, SizeType axisSize, SizeType innerSize) noexcept;

template <typename CompType, typename InType, typename OutType, typename ScaleShiftType>
void EvaluateScaleShiftCrossAxis(SizeType outerSize, const InType *src, const float *mean, const float *variance,
                                 const OutType *diffDst, ScaleShiftType *diffScale, ScaleShiftType *diffShift,
                                 float *bufScale, float *bufShift, float eps, NormalizationFlags flags,
                                 SizeType axisSizeConst, SizeType innerSize) noexcept;

template <typename CompType, typename InType, typename OutType, typename ScaleShiftType>
void EvaluateRewriteScaleShift(ScaleShiftType *diffScale, ScaleShiftType *diffShift, float *bufScale, float *bufShift,
                               NormalizationFlags flags, SizeType axisSize, SizeType numThreads) noexcept;

} // namespace LNorm
} // namespace Detail
} // namespace KuDNN

#endif //  KUDNN_LNORM_KERNELS_HPP
