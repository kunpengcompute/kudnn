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

#ifndef KUDNN_BNORM_FWD_DEF_HPP_CROSS_AXIS
#define KUDNN_BNORM_FWD_DEF_HPP_CROSS_AXIS

#include <unordered_map>
#include <functional>
#include <cmath>

#include "vector_wrappers.hpp"
#include "kernels/bnorm/kudnn_bnorm_kernels.hpp"
#include "kudnn_bnorm_cmd_loop_unrol.hpp"
#include "kudnn_bnorm_cmd_loop_unrol_neon.hpp"

namespace KuDNN {
namespace Detail {
namespace BNorm {

template<typename CompType>
void EvaluateMulFactor(CompType *mulFactor, const CompType *variance, const CompType *scale,
    NormalizationFlags flags, SizeType channelSize, const CompType eps)
{
    bool useScale = static_cast<bool>(flags & NormalizationFlags::USE_SCALE);
    SizeType loopIdx = 0;
    SizeType svStep = svcntw();
    for (; loopIdx < channelSize; loopIdx += svStep) {
        svbool_t pg = svwhilelt_b32_u64(loopIdx, channelSize);
        svfloat32_t data = svld1_f32(pg, variance + loopIdx);
        svfloat32_t tmp0 = svadd_n_f32_z(pg, data, eps);
        svfloat32_t tmp1 = svsqrt_f32_z(pg, tmp0);
        svfloat32_t svScale;
        if (useScale) {
            svScale = svld1_f32(pg, scale + loopIdx);
            data = svdiv_f32_z(pg, svScale, tmp1);
        } else {
            data = svdivr_n_f32_z(pg, tmp1, 1.0f);
        }
        svst1_f32(pg, mulFactor + loopIdx, data);
    }
}

template <typename CompType, typename InType, typename OutType>
void EvaluateMeanCNCrossAxis(const Shape &loopDims, const InType *src, CompType *meanCN, SizeType outerSize,
    SizeType channelSize, SizeType innerSize)
{
    if constexpr (IsExecute<InType>()) {
        const InType* curSrc = src;
        SizeType offsetSrcN = 0;
        CompType vMean = 0;
        svbool_t pgTrue = svptrue_b32();
        for (SizeType n = 0; n < loopDims[IDX_0]; n++) {
            offsetSrcN = n * innerSize * channelSize;
            for (SizeType c = 0; c < loopDims[IDX_1]; c++) {
#if USE_NEON
                SizeType neonLoop = innerSize - (innerSize % (NUM_2 * Vector::NeonLanes<CompType>()));
                SizeType tailLoop = innerSize % (NUM_2 * Vector::NeonLanes<CompType>());
                float32x4x2_t svMean0 = {vdupq_n_f32(0.0f), vdupq_n_f32(0.0f)};
                svfloat32_t svMean1 = svdup_f32(0.0f);
                BatchMeanFWDNEON(neonLoop, curSrc + offsetSrcN, svMean0);
                BatchMeanFWD(tailLoop, curSrc + offsetSrcN + neonLoop, svMean1);
                vMean = vaddvq_f32(svMean0.val[0]) + vaddvq_f32(svMean0.val[1]) + svaddv(pgTrue, svMean1);

#else
                svfloat32_t svMean = svdup_f32(0.0f);
                BatchMeanFWD(innerSize, curSrc + offsetSrcN, svMean);
                vMean = svaddv(pgTrue, svMean);
#endif
                meanCN[n + c * outerSize] = vMean;
                offsetSrcN += innerSize;
            }
        }
    }
}

template <typename CompType, typename InType, typename OutType>
void EvaluateVarianceCNCrossAxis(const Shape &loopDims, const InType *src, const CompType *mean,
    CompType *varianceCN, SizeType outerSize, SizeType channelSize, SizeType innerSize)
{
    if constexpr (IsExecute<InType>()) {
        const InType* curSrc = src;
        SizeType offsetSrcN = 0;
        CompType vVar = 0;
        svbool_t pgTrue = svptrue_b32();
        for (SizeType n = 0; n < loopDims[IDX_0]; n++) {
            offsetSrcN = n * innerSize * channelSize;
            for (SizeType c = 0; c < loopDims[IDX_1]; c++) {
#if USE_NEON
                SizeType neonLoop = innerSize - (innerSize % (NUM_2 * Vector::NeonLanes<CompType>()));
                SizeType tailLoop = innerSize % (NUM_2 * Vector::NeonLanes<CompType>());
                float32x4x2_t svMean0 = {vdupq_n_f32(mean[c]), vdupq_n_f32(mean[c])};
                svfloat32_t svMean1 = svdup_f32(mean[c]);
                float32x4x2_t svVar0 = {vdupq_n_f32(0.0f), vdupq_n_f32(0.0f)};
                svfloat32_t svVar1 = svdup_f32(0.0f);
                BatchVarianceFWDNEON(neonLoop, curSrc + offsetSrcN, svMean0, svVar0);
                BatchVarianceFWD(tailLoop, curSrc + offsetSrcN + neonLoop, svMean1, svVar1);
                vVar = vaddvq_f32(svVar0.val[0]) + vaddvq_f32(svVar0.val[1]) + svaddv(pgTrue, svVar1);
#else
                svfloat32_t svMean = svdup_f32(mean[c]);
                svfloat32_t svVar = svdup_f32(0.0f);
                BatchVarianceFWD(innerSize, curSrc + offsetSrcN, svMean, svVar);
                vVar = svaddv(pgTrue, svVar);
#endif
                varianceCN[n + c * outerSize] = vVar;
                offsetSrcN += innerSize;
            }
        }
    }
}

template <typename CompType, typename InType, typename OutType>
void  EvaluateCrossAxis(const Shape &loopDims, const InType *src, OutType *dst,
    const CompType *shift, const CompType *mean, const CompType *mulFactor, NormalizationFlags flags,
    SizeType channelSize, SizeType innerSize, std::uint8_t *ws, Propagation pKind) noexcept
{
    if constexpr (IsExecute<InType>()) {
        bool useShift = static_cast<bool>(flags & NormalizationFlags::USE_SHIFT);
        bool fuseRelu = static_cast<bool>(flags & NormalizationFlags::FUSE_NORM_RELU);
        SizeType offsetN = 0;

        for (SizeType ou = 0; ou < loopDims[IDX_0]; ++ou) {
            offsetN = ou * channelSize * innerSize;
            for (SizeType ch = 0; ch < loopDims[IDX_1]; ++ch) {
#if USE_NEON
                if (fuseRelu) {
                    svfloat32_t svMean = svdup_f32(mean[ch]);
                    svfloat32_t svMF = svdup_f32(mulFactor[ch]);
                    svfloat32_t svSh = useShift ? svdup_f32(shift[ch]) : svdup_f32(0.0f);
                    EvaluateBatchNormCrossFWD(innerSize, src + offsetN, dst + offsetN,
                        svMean, svMF, svSh, ws + offsetN, pKind, fuseRelu);
                } else {
                    SizeType neonLoop = innerSize - (innerSize % (NUM_2 * Vector::NeonLanes<CompType>()));
                    SizeType tailLoop = innerSize % (NUM_2 * Vector::NeonLanes<CompType>());
                    float32x4x2_t svMean0 = {vdupq_n_f32(mean[ch]), vdupq_n_f32(mean[ch])};
                    svfloat32_t svMean1 = svdup_f32(mean[ch]);
                    float32x4x2_t svMF0 = {vdupq_n_f32(mulFactor[ch]), vdupq_n_f32(mulFactor[ch])};
                    svfloat32_t svMF1 = svdup_f32(mulFactor[ch]);
                    float32x4_t svs = useShift ? vdupq_n_f32(shift[ch]) : vdupq_n_f32(0.0f);
                    float32x4x2_t svSh0 = {svs, svs};
                    svfloat32_t svSh1 = useShift ? svdup_f32(shift[ch]) : svdup_f32(0.0f);
                    EvaluateBatchNormCrossFWDNEON(neonLoop, src + offsetN, dst + offsetN,
                        svMean0, svMF0, svSh0, ws + offsetN, pKind, fuseRelu);
                    EvaluateBatchNormCrossFWD(tailLoop, src + offsetN + neonLoop, dst + offsetN + neonLoop,
                        svMean1, svMF1, svSh1, ws + offsetN + neonLoop, pKind, fuseRelu);
                }
#else
                svfloat32_t svMean = svdup_f32(mean[ch]);
                svfloat32_t svMF = svdup_f32(mulFactor[ch]);
                svfloat32_t svSh = useShift ? svdup_f32(shift[ch]) : svdup_f32(0.0f);
                EvaluateBatchNormCrossFWD(innerSize, src + offsetN, dst + offsetN,
                    svMean, svMF, svSh, ws + offsetN, pKind, fuseRelu);
#endif
                offsetN += innerSize;
            }
        }
    }
}

} // namespace BNorm
} // namespace Detail
} // namespace KuDNN

#endif // KUDNN_BNORM_FWD_DEF_HPP
