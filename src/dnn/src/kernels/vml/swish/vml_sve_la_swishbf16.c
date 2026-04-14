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

#include "kernels/vml/vml_sve_la_sigmoid.h"
#include "kernels/vml/common_macro.h"

void SwishLaSveFWDBF16(const int64_t len, const float alpha, const __bf16 *src, __bf16 *dst)
{
    int64_t i;
    int loopSwishh = svcnth();
    svfloat32_t va = svdup_n_f32(alpha);
    for (i = 0; i < len; i += loopSwishh) {
        svbool_t pgB16 = svwhilelt_b16_s64(i, len);
        svbool_t pgB32 = svwhilelt_b32_s64(i, len);

        svbfloat16_t x = svld1_bf16(pgB16, src + i);
        svfloat32_t xBot = SvCvtF32BF16BottomX(x);
        svfloat32_t xTop = SvCvtF32BF16TopX(x);

        svfloat32_t axBot = svmul_f32_x(pgB32, xBot, va);
        svfloat32_t axTop = svmul_f32_x(pgB32, xTop, va);

        svfloat32_t resBot = SvSigmoidF32X(svneg_f32_x(pgB32, axBot), pgB32);
        resBot = svmul_f32_x(pgB32, xBot, resBot);
        svfloat32_t resTop = SvSigmoidF32X(svneg_f32_x(pgB32, axTop), pgB32);
        resTop = svmul_f32_x(pgB32, xTop, resTop);

        svbfloat16_t res = SvCvtBF16F32FullX(resBot, resTop);
        svst1_bf16(pgB16, dst + i, res);
    }
}

#undef VML_VectorComputation_UNFOLD_N
#define VML_VectorComputation_UNFOLD_N 3

static void SwishMainKernelBwdBF16(svfloat32_t va, const __bf16 *src, const __bf16 *dd, __bf16 *ds, int step)
{
    svbool_t pgB16 = svptrue_b16();
    svbool_t pgB32 = svptrue_b32();
    VML_VectorComputation_DECLARE(svbfloat16_t, srcX);
    VML_VectorComputation_DECLARE(svbfloat16_t, ddX);

    VML_VectorComputation_LD_STEP(srcX, svld1_bf16, pgB16, src, step);
    VML_VectorComputation_LD_STEP(ddX, svld1_bf16, pgB16, dd, step);

    VML_VectorComputation_DECLARE(svfloat32_t, srcBot);
    VML_VectorComputation_FUNC_NOPG_1PARA(srcBot, SvCvtF32BF16BottomX, srcX);
    VML_VectorComputation_DECLARE(svfloat32_t, srcTop);
    VML_VectorComputation_FUNC_NOPG_1PARA(srcTop, SvCvtF32BF16TopX, srcX);
    VML_VectorComputation_DECLARE(svfloat32_t, ddBot);
    VML_VectorComputation_FUNC_NOPG_1PARA(ddBot, SvCvtF32BF16BottomX, ddX);
    VML_VectorComputation_DECLARE(svfloat32_t, ddTop);
    VML_VectorComputation_FUNC_NOPG_1PARA(ddTop, SvCvtF32BF16TopX, ddX);

    VML_VectorComputation_FUNC_PG_N2_2PARA(srcBot, svmul_f32_x, pgB32, srcBot, va);
    VML_VectorComputation_FUNC_PG_N2_2PARA(srcTop, svmul_f32_x, pgB32, srcTop, va);
    VML_VectorComputation_DECLARE(svfloat32_t, xSrcBotSigmoid);
    VML_VectorComputation_DECLARE(svfloat32_t, xSrcTopSigmoid);
    VML_VectorComputation_FUNC_PG_1PARA(xSrcBotSigmoid, svneg_f32_x, pgB32, srcBot);
    VML_VectorComputation_FUNC_PG_1PARA(xSrcTopSigmoid, svneg_f32_x, pgB32, srcTop);
    VML_VectorComputation_FUNC_NOPG_N2_2PARA(xSrcBotSigmoid, SvSigmoidF32X, xSrcBotSigmoid, pgB32);
    VML_VectorComputation_FUNC_NOPG_N2_2PARA(xSrcTopSigmoid, SvSigmoidF32X, xSrcTopSigmoid, pgB32);

    VML_VectorComputation_FUNC_PG_3PARA(srcBot, svmls_f32_x, pgB32, srcBot, srcBot, xSrcBotSigmoid);
    VML_VectorComputation_FUNC_PG_3PARA(srcTop, svmls_f32_x, pgB32, srcTop, srcTop, xSrcTopSigmoid);
    VML_VectorComputation_FUNC_PG_3PARA(srcBot, svmla_f32_x, pgB32, xSrcBotSigmoid, xSrcBotSigmoid, srcBot);
    VML_VectorComputation_FUNC_PG_3PARA(srcTop, svmla_f32_x, pgB32, xSrcTopSigmoid, xSrcTopSigmoid, srcTop);

    VML_VectorComputation_FUNC_PG_2PARA(srcBot, svmul_f32_x, pgB32, ddBot, srcBot);
    VML_VectorComputation_FUNC_PG_2PARA(srcTop, svmul_f32_x, pgB32, ddTop, srcTop);
    VML_VectorComputation_FUNC_NOPG_N1_2PARA(srcBot, SvF32CvtBf16InfSpecial, pgB32, srcBot);
    VML_VectorComputation_FUNC_NOPG_N1_2PARA(srcTop, SvF32CvtBf16InfSpecial, pgB32, srcTop);

    VML_VectorComputation_FUNC_NOPG_2PARA(srcX, SvCvtBF16F32FullX, srcBot, srcTop);
    VML_VectorComputation_ST_STEP(svst1_bf16, pgB16, ds, step, srcX);
}

void SwishLaSveBWDBF16(const int64_t len, const float alpha, const __bf16 *src, const __bf16 *dd, __bf16 *ds)
{
    int64_t i = 0;
    int loopSwishh = svcnth();
    svfloat32_t va = svdup_n_f32(alpha);
    for (; i <= len - loopSwishh * VML_VectorComputation_UNFOLD_N; i += loopSwishh * VML_VectorComputation_UNFOLD_N) {
        SwishMainKernelBwdBF16(va, src + i, dd + i, ds + i, loopSwishh);
    }
    for (; i < len; i += loopSwishh) {
        svbool_t pgB16 = svwhilelt_b16_s64(i, len);
        svbool_t pgB32 = svwhilelt_b32_s64(i, len);

        svbfloat16_t srcX = svld1_bf16(pgB16, src + i);
        svbfloat16_t ddX = svld1_bf16(pgB16, dd + i);

        svfloat32_t srcBot = SvCvtF32BF16BottomX(srcX);
        svfloat32_t srcTop = SvCvtF32BF16TopX(srcX);

        svfloat32_t ddBot = SvCvtF32BF16BottomX(ddX);
        svfloat32_t ddTop = SvCvtF32BF16TopX(ddX);

        svfloat32_t xSrcBot = svmul_f32_x(pgB32, srcBot, va);
        svfloat32_t xSrcTop = svmul_f32_x(pgB32, srcTop, va);
        svfloat32_t xSrcBotSigmoid = SvSigmoidF32X(svneg_f32_x(pgB32, xSrcBot), pgB32);
        svfloat32_t xSrcTopSigmoid = SvSigmoidF32X(svneg_f32_x(pgB32, xSrcTop), pgB32);

        svfloat32_t srcBotRes = svmls_f32_x(pgB32, xSrcBot, xSrcBot, xSrcBotSigmoid);
        srcBotRes = svmla_f32_x(pgB32, xSrcBotSigmoid, xSrcBotSigmoid, srcBotRes);
        svfloat32_t srcTopRes = svmls_f32_x(pgB32, xSrcTop, xSrcTop, xSrcTopSigmoid);
        srcTopRes = svmla_f32_x(pgB32, xSrcTopSigmoid, xSrcTopSigmoid, srcTopRes);

        svfloat32_t ddBotRes = svmul_f32_x(pgB32, ddBot, srcBotRes);
        svfloat32_t ddTopRes = svmul_f32_x(pgB32, ddTop, srcTopRes);

        ddBotRes = SvF32CvtBf16InfSpecial(pgB32, ddBotRes);
        ddTopRes = SvF32CvtBf16InfSpecial(pgB32, ddTopRes);

        svbfloat16_t res = SvCvtBF16F32FullX(ddBotRes, ddTopRes);
        svst1_bf16(pgB16, ds + i, res);
    }
}
