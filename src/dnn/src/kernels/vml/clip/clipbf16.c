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
#define VML_VectorComputation_UNFOLD_N 3
#include "kernels/vml/common_macro.h"

static void ClipMainKernelFwdBF16(float alpha, float beta, const __bf16 *src, __bf16 *dst, int step)
{
    svbool_t pgB16 = svptrue_b16();
    svbool_t pgB32 = svptrue_b32();

    VML_VectorComputation_DECLARE(svfloat32_t, xBot);
    VML_VectorComputation_DECLARE(svfloat32_t, xTop);

    LOAD_CONVERT_BF16_TO_FP32_1(src, pgB16, xBot, xTop, step);

    // clip compute
    VML_VectorComputation_DECLARE(svfloat32_t, resBot);
    VML_VectorComputation_DECLARE(svfloat32_t, resTop);
    VML_VectorComputation_DECLARE(svbool_t, pgRetAlpha);
    VML_VectorComputation_DECLARE(svbool_t, pgRetBeta);

    VML_VectorComputation_FUNC_PG_N2_2PARA(pgRetBeta, svcmpgt_n_f32, pgB32, xBot, beta);
    VML_VectorComputation_FUNC_NOPG_N3_3PARA(resBot, svdup_n_f32_m, xBot, pgRetBeta, beta);
    VML_VectorComputation_FUNC_PG_N2_2PARA(pgRetAlpha, svcmple_n_f32, pgB32, xBot, alpha);
    VML_VectorComputation_FUNC_NOPG_N3_3PARA(resBot, svdup_n_f32_m, resBot, pgRetAlpha, alpha);
    VML_VectorComputation_FUNC_PG_N2_2PARA(pgRetBeta, svcmpgt_n_f32, pgB32, xTop, beta);
    VML_VectorComputation_FUNC_NOPG_N3_3PARA(resTop, svdup_n_f32_m, xTop, pgRetBeta, beta);
    VML_VectorComputation_FUNC_PG_N2_2PARA(pgRetAlpha, svcmple_n_f32, pgB32, xTop, alpha);
    VML_VectorComputation_FUNC_NOPG_N3_3PARA(resTop, svdup_n_f32_m, resTop, pgRetAlpha, alpha);

    VML_VectorComputation_FUNC_PG_1PARA(resBot, SvF32CvtBf16InfSpecial, pgB32, resBot);
    VML_VectorComputation_FUNC_PG_1PARA(resTop, SvF32CvtBf16InfSpecial, pgB32, resTop);

    CONVERT_STORE_FP32_TO_BF16(dst, pgB16, resBot, resTop, step);
}

void ClipSveFWDBF16(const int64_t len, const float alpha, const float beta, const __bf16 *src, __bf16 *dst)
{
    int64_t i;
    int64_t loopCliph = svcnth();

    for (i = 0; i <= len - loopCliph * VML_VectorComputation_UNFOLD_N;
         i += loopCliph * VML_VectorComputation_UNFOLD_N) {
        ClipMainKernelFwdBF16(alpha, beta, src + i, dst + i, loopCliph);
    }
    for (; i < len; i += loopCliph) {
        svbool_t pgB16 = svwhilelt_b16_s64(i, len);
        svbool_t pgB32 = svwhilelt_b32_s64(i, len);

        svbfloat16_t x = svld1_bf16(pgB16, src + i);

        svfloat32_t resBot = SvCvtF32BF16BottomX(x);
        svfloat32_t resTop = SvCvtF32BF16TopX(x);

        resBot = svdup_f32_m(resBot, svcmpgt_n_f32(pgB32, resBot, beta), beta);
        resBot = svdup_f32_m(resBot, svcmple_n_f32(pgB32, resBot, alpha), alpha);
        resTop = svdup_f32_m(resTop, svcmpgt_n_f32(pgB32, resTop, beta), beta);
        resTop = svdup_f32_m(resTop, svcmple_n_f32(pgB32, resTop, alpha), alpha);

        resBot = SvF32CvtBf16InfSpecial(pgB32, resBot);
        resTop = SvF32CvtBf16InfSpecial(pgB32, resTop);

        // convert back to bf16
        svbfloat16_t res = SvCvtBF16F32FullX(resBot, resTop);
        svst1_bf16(pgB16, dst + i, res);
    }
}

#undef VML_VectorComputation_UNFOLD_N
#define VML_VectorComputation_UNFOLD_N 4

static void ClipMainKernelBwdBF16(float alpha, float beta, const __bf16 *src, const __bf16 *dd, __bf16 *ds, int step)
{
    svbool_t pgB16 = svptrue_b16();
    svbool_t pgB32 = svptrue_b32();

    VML_VectorComputation_DECLARE(svfloat32_t, xBot);
    VML_VectorComputation_DECLARE(svfloat32_t, xTop);
    VML_VectorComputation_DECLARE(svfloat32_t, ddBot);
    VML_VectorComputation_DECLARE(svfloat32_t, ddTop);

    LOAD_CONVERT_BF16_TO_FP32_2(src, dd, pgB16, xBot, xTop, ddBot, ddTop, step);

    // relu compute
    VML_VectorComputation_DECLARE(svfloat32_t, resBot);
    VML_VectorComputation_DECLARE(svfloat32_t, resTop);
    VML_VectorComputation_DECLARE(svbool_t, pgRetZeroUp);
    VML_VectorComputation_DECLARE(svbool_t, pgRetZeroDown);

    VML_VectorComputation_FUNC_PG_N2_2PARA(pgRetZeroUp, svcmpgt_n_f32, pgB32, xBot, beta);
    VML_VectorComputation_FUNC_NOPG_N3_3PARA(resBot, svdup_f32_m, ddBot, pgRetZeroUp, FLOAT_ZERO);
    VML_VectorComputation_FUNC_PG_N2_2PARA(pgRetZeroDown, svcmple_n_f32, pgB32, xBot, alpha);
    VML_VectorComputation_FUNC_NOPG_N3_3PARA(resBot, svdup_f32_m, resBot, pgRetZeroDown, FLOAT_ZERO);
    VML_VectorComputation_FUNC_PG_N2_2PARA(pgRetZeroUp, svcmpgt_n_f32, pgB32, xTop, beta);
    VML_VectorComputation_FUNC_NOPG_N3_3PARA(resTop, svdup_f32_m, ddTop, pgRetZeroUp, FLOAT_ZERO);
    VML_VectorComputation_FUNC_PG_N2_2PARA(pgRetZeroDown, svcmple_n_f32, pgB32, xTop, alpha);
    VML_VectorComputation_FUNC_NOPG_N3_3PARA(resTop, svdup_f32_m, resTop, pgRetZeroDown, FLOAT_ZERO);

    CONVERT_STORE_FP32_TO_BF16(ds, pgB16, resBot, resTop, step);
}

void ClipSveBWDBF16(const int64_t len, const float alpha, const float beta, const __bf16 *src, const __bf16 *dd,
                    __bf16 *ds)
{
    int64_t i = 0;
    int64_t loopCliph = svcnth();

    for (; i <= len - loopCliph * VML_VectorComputation_UNFOLD_N; i += loopCliph * VML_VectorComputation_UNFOLD_N) {
        ClipMainKernelBwdBF16(alpha, beta, src + i, dd + i, ds + i, loopCliph);
    }
    for (; i < len; i += loopCliph) {
        svbool_t pgB16 = svwhilelt_b16_s64(i, len);
        svbool_t pgB32 = svwhilelt_b32_s64(i, len);

        svbfloat16_t srcX = svld1_bf16(pgB16, src + i);
        svbfloat16_t ddX = svld1_bf16(pgB16, dd + i);

        // convert to fp32
        svfloat32_t xBot = SvCvtF32BF16BottomX(srcX);
        svfloat32_t xTop = SvCvtF32BF16TopX(srcX);

        svfloat32_t resBot = SvCvtF32BF16BottomX(ddX);
        svfloat32_t resTop = SvCvtF32BF16TopX(ddX);

        resBot = svdup_f32_m(resBot, svcmpgt_n_f32(pgB32, xBot, beta), FLOAT_ZERO);
        resBot = svdup_f32_m(resBot, svcmple_n_f32(pgB32, xBot, alpha), FLOAT_ZERO);
        resTop = svdup_f32_m(resTop, svcmpgt_n_f32(pgB32, xTop, beta), FLOAT_ZERO);
        resTop = svdup_f32_m(resTop, svcmple_n_f32(pgB32, xTop, alpha), FLOAT_ZERO);

        // convert back to bf16
        svbfloat16_t res = SvCvtBF16F32FullX(resBot, resTop);
        svst1_bf16(pgB16, ds + i, res);
    }
}
