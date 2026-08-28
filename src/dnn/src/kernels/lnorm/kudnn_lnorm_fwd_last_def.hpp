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

#ifndef KUDNN_LNORM_FWD_LAST_DEF_HPP
#define KUDNN_LNORM_FWD_LAST_DEF_HPP

#include "kudnn.hpp"
#include "vector_wrappers.hpp"
#include "kernels/lnorm/kudnn_lnorm_kernels.hpp"
#include "kernels/lnorm/kudnn_lnorm_fwd_last_def_sve.hpp"
#if USE_NEON
#include "kernels/lnorm/kudnn_lnorm_fwd_last_def_neon.hpp"
#endif

namespace KuDNN {
namespace Detail {
namespace LNorm {

template <typename CompType, typename InType, typename OutType, typename ScaleShiftType>
void EvaluateLastAxis(SizeType execAcrossAxis, const InType *src, OutType *dst, const ScaleShiftType *scale,
    const ScaleShiftType *shift, float *mean, float *variance,
    bool saveStats, float eps, NormalizationFlags flags, SizeType axisSize) noexcept
{
#if USE_NEON
    EvaluateLastAxisNEON<CompType, InType, OutType, ScaleShiftType>(execAcrossAxis, src, dst, scale, shift,
        mean, variance, saveStats, eps, flags, axisSize);
#else
    EvaluateLastAxisVC<CompType, InType, OutType, ScaleShiftType>(execAcrossAxis, src, dst, scale, shift,
        mean, variance, saveStats, eps, flags, axisSize);
#endif
}

} // LNorm
} // Detail
} // KuDNN

#endif //  KUDNN_LNORM_FWD_LAST_DEF_HPP
