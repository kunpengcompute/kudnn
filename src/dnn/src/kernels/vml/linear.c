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

#include "vml_sve_base.h"
#include "common_macro.h"

static void LinearMainKernelF32(svfloat32_t a, const float *src, svfloat32_t b, float *dst, int step)
{
    svbool_t pg = svptrue_b32();
    VML_VectorComputation_DECLARE(svfloat32_t, x);
    VML_VectorComputation_LD_STEP(x, svld1_f32, pg, src, step);

    VML_VectorComputation_DECLARE(svfloat32_t, res);
    VML_VectorComputation_FUNC_PG_N13_3PARA(res, svmad_f32_x, pg, a, x, b);

    VML_VectorComputation_ST_STEP(svst1_f32, pg, dst, step, res);
}

void LinearSveF32(const int64_t len, const float alpha, const float beta, const float *src, float *dst)
{
    int64_t i;
    int64_t loopLinearf = svcntw();
    svfloat32_t va = svdup_n_f32(alpha);
    svfloat32_t vb = svdup_n_f32(beta);

    for (i = 0; i <= len - loopLinearf * VML_VectorComputation_UNFOLD_N; i += loopLinearf * VML_VectorComputation_UNFOLD_N) {
        LinearMainKernelF32(va, src + i, vb, dst + i, loopLinearf);
    }
    for (; i < len; i += loopLinearf) {
        svbool_t pg = svwhilelt_b32_s64(i, len);
        svfloat32_t x = svld1_f32(pg, src + i);
        svfloat32_t y = svmad_f32_x(pg, va, x, vb);
        svst1_f32(pg, dst + i, y);
    }
}

static void LinearMainKernelF16(svfloat32_t a, const __fp16 *src, svfloat32_t b, __fp16 *dst, int step)
{
    svbool_t pgB16 = svptrue_b16();
    svbool_t pgB32 = svptrue_b32();

    VML_VectorComputation_DECLARE(svfloat32_t, xEven);
    VML_VectorComputation_DECLARE(svfloat32_t, xOdd);

    LOAD_CONVERT_FP16_TO_FP32_1(src, pgB16, xEven, xOdd, step);

    // mul + add
    VML_VectorComputation_DECLARE(svfloat32_t, resEven);
    VML_VectorComputation_DECLARE(svfloat32_t, resOdd);
    VML_VectorComputation_FUNC_PG_N13_3PARA(resEven, svmad_f32_x, pgB32, a, xEven, b);
    VML_VectorComputation_FUNC_PG_N13_3PARA(resOdd, svmad_f32_x, pgB32, a, xOdd, b);

    CONVERT_STORE_FP32_TO_FP16(dst, pgB16, resEven, resOdd, step);
}

void LinearSveF16(const int64_t len, const float alpha, const float beta, const __fp16 *src, __fp16 *dst)
{
    int64_t i;
    int64_t loopLinearf = svcnth();
    svfloat32_t va = svdup_n_f32(alpha);
    svfloat32_t vb = svdup_n_f32(beta);

    for (i = 0; i <= len - loopLinearf * VML_VectorComputation_UNFOLD_N; i += loopLinearf * VML_VectorComputation_UNFOLD_N) {
        LinearMainKernelF16(va, src + i, vb, dst + i, loopLinearf);
    }
    for (; i < len; i += loopLinearf) {
        svbool_t pgB16 = svwhilelt_b16_s64(i, len);
        svbool_t pgB32 = svwhilelt_b32_s64(i, len);

        svfloat16_t x = svld1_f16(pgB16, src + i);

        svfloat32_t xfEven = svcvt_f32_f16_x(pgB16, x);
        svfloat32_t xfOdd = CvtOddF32F16(x, pgB16, pgB32);

        svfloat32_t resEven = svmad_f32_x(pgB32, va, xfEven, vb);
        svfloat32_t resOdd = svmad_f32_x(pgB32, va, xfOdd, vb);

        svfloat16_t res = svcvt_f16_f32_x(pgB32, resEven);

        x = CvtOddF16F32(resOdd, pgB32);

        res = svadd_f16_x(pgB16, res, x);
        svst1_f16(pgB16, dst + i, res);
    }
}

static void LinearMainKernelBF16(svfloat32_t a, const __bf16 *src, svfloat32_t b, __bf16 *dst, int step)
{
    svbool_t pgB16 = svptrue_b16();
    svbool_t pgB32 = svptrue_b32();

    VML_VectorComputation_DECLARE(svfloat32_t, xBot);
    VML_VectorComputation_DECLARE(svfloat32_t, xTop);

    LOAD_CONVERT_BF16_TO_FP32_1(src, pgB16, xBot, xTop, step);

    // linear compute
    VML_VectorComputation_DECLARE(svfloat32_t, resBot);
    VML_VectorComputation_DECLARE(svfloat32_t, resTop);
    VML_VectorComputation_FUNC_PG_N13_3PARA(resBot, svmad_f32_x, pgB32, a, xBot, b);
    VML_VectorComputation_FUNC_PG_N13_3PARA(resTop, svmad_f32_x, pgB32, a, xTop, b);

    VML_VectorComputation_FUNC_PG_1PARA(resBot, SvF32CvtBf16InfSpecial, pgB32, resBot);
    VML_VectorComputation_FUNC_PG_1PARA(resTop, SvF32CvtBf16InfSpecial, pgB32, resTop);

    CONVERT_STORE_FP32_TO_BF16(dst, pgB16, resBot, resTop, step);
}

void LinearSveBF16(const int64_t len, const float alpha, const float beta, const __bf16 *src, __bf16 *dst)
{
    int64_t i;
    int64_t loopLinearf = svcnth();
    svfloat32_t va = svdup_n_f32(alpha);
    svfloat32_t vb = svdup_n_f32(beta);

    for (i = 0; i <= len - loopLinearf * VML_VectorComputation_UNFOLD_N; i += loopLinearf * VML_VectorComputation_UNFOLD_N) {
        LinearMainKernelBF16(va, src + i, vb, dst + i, loopLinearf);
    }

    for (; i < len; i += loopLinearf) {
        svbool_t pgB16 = svwhilelt_b16_s64(i, len);
        svbool_t pgB32 = svwhilelt_b32_s64(i, len);

        svbfloat16_t x = svld1_bf16(pgB16, src + i);

        // convert to fp32
        svfloat32_t xBot = SvCvtF32BF16BottomX(x);
        svfloat32_t xTop = SvCvtF32BF16TopX(x);

        // mul + add
        svfloat32_t resBot = svmad_f32_x(pgB32, va, xBot, vb);
        svfloat32_t resTop = svmad_f32_x(pgB32, va, xTop, vb);

        resBot = SvF32CvtBf16InfSpecial(pgB32, resBot);
        resTop = SvF32CvtBf16InfSpecial(pgB32, resTop);

        // convert back to bf16
        svbfloat16_t res = SvCvtBF16F32FullX(resBot, resTop);
        svst1_bf16(pgB16, dst + i, res);
    }
}
