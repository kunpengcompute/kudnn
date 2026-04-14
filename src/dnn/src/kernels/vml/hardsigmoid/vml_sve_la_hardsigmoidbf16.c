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

#include "kernels/vml/vml_sve_base.h"

void HardSigmoidLaSveFWDBF16(const int64_t len, const float alpha, const float beta, const __bf16 *src, __bf16 *dst)
{
    int64_t i;
    int loopHardSigmoidh = svcnth();
    svfloat32_t alphaVec = svdup_n_f32(alpha);
    svfloat32_t betaVec = svdup_n_f32(beta);
    for (i = 0; i < len; i += loopHardSigmoidh) {
        svbool_t pgB16 = svwhilelt_b16_s64(i, len);
        svbool_t pgB32 = svwhilelt_b32_s64(i, len);

        svbfloat16_t x = svld1_bf16(pgB16, src + i);

        svfloat32_t xBot = SvCvtF32BF16BottomX(x);
        svfloat32_t xTop = SvCvtF32BF16TopX(x);

        svfloat32_t resBot = svmad_f32_x(pgB32, xBot, alphaVec, betaVec);
        svfloat32_t resTop = svmad_f32_x(pgB32, xTop, alphaVec, betaVec);
        resBot = svminnm_f32_x(pgB32, resBot, VectorComputation_FLOAT_ONE);
        resTop = svminnm_f32_x(pgB32, resTop, VectorComputation_FLOAT_ONE);
        resBot = svmaxnm_f32_x(pgB32, resBot, VectorComputation_FLOAT_ZERO);
        resTop = svmaxnm_f32_x(pgB32, resTop, VectorComputation_FLOAT_ZERO);

        svbfloat16_t res = SvCvtBF16F32FullX(resBot, resTop);
        svst1_bf16(pgB16, dst + i, res);
    }
}

void HardSigmoidLaSveBWDBF16(const int64_t len, const float alpha, const float beta,
    const __bf16 *src, const __bf16 *dd, __bf16 *ds)
{
    int64_t i;
    int loopHardSigmoidh = svcnth();
    svfloat32_t alphaVec = svdup_n_f32(alpha);
    svfloat32_t betaVec = svdup_n_f32(beta);
    for (i = 0; i < len; i += loopHardSigmoidh) {
        svbool_t pgB16 = svwhilelt_b16_s64(i, len);
        svbool_t pgB32 = svwhilelt_b32_s64(i, len);

        svbfloat16_t srcX = svld1_bf16(pgB16, src + i);
        svbfloat16_t ddX = svld1_bf16(pgB16, dd + i);

        svfloat32_t srcBot = SvCvtF32BF16BottomX(srcX);
        svfloat32_t srcTop = SvCvtF32BF16TopX(srcX);

        svfloat32_t ddBot = SvCvtF32BF16BottomX(ddX);
        svfloat32_t ddTop = SvCvtF32BF16TopX(ddX);

        svfloat32_t srcBotRes = svmad_f32_x(pgB32, srcBot, alphaVec, betaVec);
        svfloat32_t srcTopRes = svmad_f32_x(pgB32, srcTop, alphaVec, betaVec);

        svbool_t cmpBot = svcmplt_f32(pgB32, srcBotRes, VectorComputation_FLOAT_ONE);
        svbool_t cmpTop = svcmplt_f32(pgB32, srcTopRes, VectorComputation_FLOAT_ONE);
        cmpBot = svcmpgt_f32(cmpBot, srcBotRes, VectorComputation_FLOAT_ZERO);
        cmpTop = svcmpgt_f32(cmpTop, srcTopRes, VectorComputation_FLOAT_ZERO);

        svfloat32_t ddBotRes = svmul_f32_z(cmpBot, ddBot, alphaVec);
        svfloat32_t ddTopRes = svmul_f32_z(cmpTop, ddTop, alphaVec);

        ddBotRes = SvF32CvtBf16InfSpecial(pgB32, ddBotRes);
        ddTopRes = SvF32CvtBf16InfSpecial(pgB32, ddTopRes);

        svbfloat16_t res = SvCvtBF16F32FullX(ddBotRes, ddTopRes);
        svst1_bf16(pgB16, ds + i, res);
    }
}
