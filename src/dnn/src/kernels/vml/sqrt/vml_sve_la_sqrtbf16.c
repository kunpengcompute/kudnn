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
#include "kernels/vml/common_macro.h"

void SqrtLaSveFWDBF16(const int64_t len, const __bf16 *src, __bf16 *dst)
{
    int64_t i;
    int loopSqrth = svcnth();
    for (i = 0; i < len; i += loopSqrth) {
        svbool_t pgB16 = svwhilelt_b16_s64(i, len);
        svbool_t pgB32 = svwhilelt_b32_s64(i, len);

        svbfloat16_t x = svld1_bf16(pgB16, src + i);

        // convert to fp32
        svfloat32_t resBot = SvCvtF32BF16BottomX(x);
        svfloat32_t resTop = SvCvtF32BF16TopX(x);

        resBot = svsqrt_f32_x(pgB32, resBot);
        resTop = svsqrt_f32_x(pgB32, resTop);

        // convert back to bf16
        svbfloat16_t res = SvCvtBF16F32FullX(resBot, resTop);
        svst1_bf16(pgB16, dst + i, res);
    }
}

#undef VML_VectorComputation_UNFOLD_N
#define VML_VectorComputation_UNFOLD_N 8

static void SqrtMainKernelBwdBF16(const __bf16 *src, const __bf16 *dd, __bf16 *ds, int step)
{
    svbool_t pgB16 = svptrue_b16();
    svbool_t pgB32 = svptrue_b32();
    VML_VectorComputation_DECLARE(svbfloat16_t, srcX);
    VML_VectorComputation_DECLARE(svbfloat16_t, ddX);
    VML_VectorComputation_LD_STEP(srcX, svld1_bf16, pgB16, src, step);
    VML_VectorComputation_LD_STEP(ddX, svld1_bf16, pgB16, dd, step);

    // convert to fp32
    VML_VectorComputation_DECLARE(svfloat32_t, srcBot);
    VML_VectorComputation_FUNC_NOPG_1PARA(srcBot, SvCvtF32BF16BottomX, srcX);
    VML_VectorComputation_DECLARE(svfloat32_t, srcTop);
    VML_VectorComputation_FUNC_NOPG_1PARA(srcTop, SvCvtF32BF16TopX, srcX);
    VML_VectorComputation_DECLARE(svfloat32_t, ddBot);
    VML_VectorComputation_FUNC_NOPG_1PARA(ddBot, SvCvtF32BF16BottomX, ddX);
    VML_VectorComputation_DECLARE(svfloat32_t, ddTop);
    VML_VectorComputation_FUNC_NOPG_1PARA(ddTop, SvCvtF32BF16TopX, ddX);

    // sqrt compute
    VML_VectorComputation_FUNC_PG_1PARA(srcBot, svsqrt_f32_x, pgB32, srcBot);
    VML_VectorComputation_FUNC_PG_1PARA(srcTop, svsqrt_f32_x, pgB32, srcTop);
    VML_VectorComputation_FUNC_PG_N2_2PARA(ddBot, svmul_f32_x, pgB32, ddBot, VectorComputation_FLOAT_HALF);
    VML_VectorComputation_FUNC_PG_N2_2PARA(ddTop, svmul_f32_x, pgB32, ddTop, VectorComputation_FLOAT_HALF);
    VML_VectorComputation_FUNC_PG_2PARA(srcBot, svdiv_f32_x, pgB32, ddBot, srcBot);
    VML_VectorComputation_FUNC_PG_2PARA(srcTop, svdiv_f32_x, pgB32, ddTop, srcTop);

    // convert back to bf16 and store
    VML_VectorComputation_FUNC_NOPG_N1_2PARA(srcBot, SvF32CvtBf16InfSpecial, pgB32, srcBot);
    VML_VectorComputation_FUNC_NOPG_N1_2PARA(srcTop, SvF32CvtBf16InfSpecial, pgB32, srcTop);
    VML_VectorComputation_FUNC_NOPG_2PARA(srcX, SvCvtBF16F32FullX, srcBot, srcTop);
    VML_VectorComputation_ST_STEP(svst1_bf16, pgB16, ds, step, srcX);
}

void SqrtLaSveBWDBF16(const int64_t len, const __bf16 *src, const __bf16 *dd, __bf16 *ds)
{
    int64_t i = 0;
    int loopSqrh = svcnth();
    for (; i <= len - loopSqrh * VML_VectorComputation_UNFOLD_N; i += loopSqrh * VML_VectorComputation_UNFOLD_N) {
        SqrtMainKernelBwdBF16(src + i, dd + i, ds + i, loopSqrh);
    }
    for (; i < len; i += loopSqrh) {
        svbool_t pgB16 = svwhilelt_b16_s64(i, len);
        svbool_t pgB32 = svwhilelt_b32_s64(i, len);

        svbfloat16_t srcX = svld1_bf16(pgB16, src + i);
        svbfloat16_t ddX = svld1_bf16(pgB16, dd + i);

        // convert to fp32
        svfloat32_t xBot = SvCvtF32BF16BottomX(srcX);
        svfloat32_t xTop = SvCvtF32BF16TopX(srcX);

        svfloat32_t resBot = SvCvtF32BF16BottomX(ddX);
        svfloat32_t resTop = SvCvtF32BF16TopX(ddX);

        xBot = svsqrt_f32_x(pgB32, xBot);
        xTop = svsqrt_f32_x(pgB32, xTop);
        xBot = svdiv_f32_x(pgB32, VectorComputation_FLOAT_HALF, xBot);
        xTop = svdiv_f32_x(pgB32, VectorComputation_FLOAT_HALF, xTop);

        resBot = svmul_f32_x(pgB32, resBot, xBot);
        resTop = svmul_f32_x(pgB32, resTop, xTop);

        resBot = SvF32CvtBf16InfSpecial(pgB32, resBot);
        resTop = SvF32CvtBf16InfSpecial(pgB32, resTop);

        // convert back to bf16
        svbfloat16_t res = SvCvtBF16F32FullX(resBot, resTop);
        svst1_bf16(pgB16, ds + i, res);
    }
}
