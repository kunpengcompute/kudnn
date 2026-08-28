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

#ifndef KUDNN_BNORM_BWD_DEF_HPP
#define KUDNN_BNORM_BWD_DEF_HPP

#include <unordered_map>
#include <functional>
#include <cmath>

#include "vector_wrappers.hpp"
#include "kudnn_bnorm_cmd_loop_unrol.hpp"
#include "kernels/bnorm/kudnn_bnorm_kernels.hpp"

namespace KuDNN {
namespace Detail {
namespace BNorm {

template <typename CompType, typename InType, typename OutType>
void EvaluateStatsCrossAxisBwd(SizeType outerSize, const InType *src, const OutType *diffDst,
    const CompType *mean, const CompType *var, CompType *diffScale, CompType *diffShift, CompType eps,
    SizeType axisSize, SizeType channelSize, SizeType innerSize, const std::uint8_t *ws) noexcept
{
    svbool_t pgTrue = svptrue_b32();
    SizeType step = innerSize * channelSize;
    for (SizeType ch = 0; ch < axisSize; ++ch) {
        SizeType offset = ch * innerSize;
        CompType sqrtVar = std::sqrt(var[ch] + eps);

        svfloat32_t vDiffGamma = svdup_f32(0.0f);
        svfloat32_t vDiffBeta = svdup_f32(0.0f);

        for (SizeType ou = 0; ou < outerSize; ++ou) {
            for (SizeType in = 0; in < innerSize; in += svcntw()) {
                svbool_t pg = svwhilelt_b32_u64(in, innerSize);
                svfloat32_t vSrc = LoadBNormBwd(pg, src + offset + in);
                svfloat32_t vDiffDst = LoadBNormBwd(pg, diffDst + offset + in);

                if (ws) {
                    svfloat32_t vWs = LoadBNormBwd(pg, ws + offset + in);
                    svbool_t pgEq = svcmpeq_n_f32(pg, vWs, 0.0f);
                    vDiffDst = svmul_n_f32_m(pgEq, vDiffDst, 0.0f);
                }

                vSrc = svsub_n_f32_m(pg, vSrc, mean[ch]);
                vDiffGamma = svmla_f32_m(pg, vDiffGamma, vSrc, vDiffDst);
                vDiffBeta = svadd_f32_m(pg, vDiffBeta, vDiffDst);
            }
            offset += step;
        }

        diffScale[ch] = svaddv_f32(pgTrue, vDiffGamma) / sqrtVar;
        diffShift[ch] = svaddv_f32(pgTrue, vDiffBeta);
    }
}

template <typename CompType, typename InType, typename OutType>
void EvaluateCrossAxisBwd(SizeType axisSize, const InType *src, const OutType *diffDst,
    const CompType *mean, const CompType *var, const CompType *scale, InType *diffSrc,
    CompType *diffScale, CompType *diffShift, const CompType eps, NormalizationFlags flags,
    SizeType outerSize, SizeType channelSize, SizeType innerSize, const std::uint8_t *ws) noexcept
{
    bool useScale = static_cast<bool>(flags & NormalizationFlags::USE_SCALE);
    bool useGlobStats = static_cast<bool>(flags & NormalizationFlags::USE_GLOBAL_STATS);
    for (SizeType ou = 0; ou < axisSize; ++ou) {
        SizeType offset = ou * channelSize * innerSize;
        for (SizeType ch = 0; ch < channelSize; ++ch) {
            float srcScaler = ((useScale) ? scale[ch] : 1.0f) / std::sqrt(var[ch] + eps);
            float diffGammaScaler = 1.0f;
            float diffBetaScaler = 0.0f;
            if (!useGlobStats) {
                // negative sign for fma instr usage later
                diffGammaScaler = (-1.0f) * diffScale[ch] / (innerSize * outerSize * std::sqrt(var[ch] + eps));
                diffBetaScaler = diffShift[ch] / (innerSize * outerSize);
            }

            for (SizeType in = 0; in < innerSize; in += svcntw()) {
                svbool_t pg = svwhilelt_b32_u64(in, innerSize);
                svfloat32_t vSrc = LoadBNormBwd(pg, src + offset + in);

                // in case of (fuse_norm_relu && !ws[d_off]), need to change diffDst load to DupFloatN(0.0f)
                svfloat32_t vDiffDst = LoadBNormBwd(pg, diffDst + offset + in);
                if (ws) {
                    svfloat32_t vWs = LoadBNormBwd(pg, ws + offset + in);
                    svbool_t pgEq = svcmpeq_n_f32(pg, vWs, 0.0f);
                    vDiffDst = svmul_n_f32_m(pgEq, vDiffDst, 0.0f);
                }

                if (!useGlobStats) {
                    vSrc = svsub_n_f32_m(pg, vSrc, mean[ch]);
                    vDiffDst = svmla_n_f32_m(pg, vDiffDst, vSrc, diffGammaScaler);
                    vDiffDst = svsub_n_f32_m(pg, vDiffDst, diffBetaScaler);
                }
                vDiffDst = svmul_n_f32_m(pg, vDiffDst, srcScaler);
                StoreBNorm(diffSrc + offset + in, pg, vDiffDst);
            }
            offset += innerSize;
        }
    }
}

} // namespace BNorm
} // namespace Detail
} // namespace KuDNN

#endif // KUDNN_BNORM_BWD_DEF_HPP