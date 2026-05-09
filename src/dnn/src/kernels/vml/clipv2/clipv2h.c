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

static void ClipV2MainKernelFwdF16(const float alpha, const float beta, const __fp16 *src, __fp16 *dst, int step)
{
    svbool_t pg = svptrue_b16();
    VML_VectorComputation_DECLARE(svfloat16_t, res);
    VML_VectorComputation_LD_STEP(res, svld1_f16, pg, src, step);

    VML_VectorComputation_DECLARE(svbool_t, pgRetAlpha);
    VML_VectorComputation_FUNC_PG_N2_2PARA(pgRetAlpha, svcmple_n_f16, pg, res, alpha);
    VML_VectorComputation_DECLARE(svbool_t, pgRetBeta);
    VML_VectorComputation_FUNC_PG_N2_2PARA(pgRetBeta, svcmpge_n_f16, pg, res, beta);

    VML_VectorComputation_FUNC_NOPG_N3_3PARA(res, svdup_f16_m, res, pgRetAlpha, alpha);
    VML_VectorComputation_FUNC_NOPG_N3_3PARA(res, svdup_f16_m, res, pgRetBeta, beta);

    VML_VectorComputation_ST_STEP(svst1_f16, pg, dst, step, res);
}

void ClipV2SveFWDF16(const int64_t len, const float alpha, const float beta, const __fp16 *src, __fp16 *dst)
{
    int64_t i;
    int64_t loopClipV2h = svcnth();

    for (i = 0; i <= len - loopClipV2h * VML_VectorComputation_UNFOLD_N;
         i += loopClipV2h * VML_VectorComputation_UNFOLD_N) {
        ClipV2MainKernelFwdF16(alpha, beta, src + i, dst + i, loopClipV2h);
    }
    for (; i < len; i += loopClipV2h) {
        svbool_t pg = svwhilelt_b16_s64(i, len);
        svfloat16_t res = svld1_f16(pg, src + i);
        res = svdup_f16_m(res, svcmpge_n_f16(pg, res, beta), beta);
        res = svdup_f16_m(res, svcmple_n_f16(pg, res, alpha), alpha);
        svst1_f16(pg, dst + i, res);
    }
}

#undef VML_VectorComputation_UNFOLD_N
#define VML_VectorComputation_UNFOLD_N 6

static void ClipV2MainKernelBwdF16(svfloat32_t va, svfloat32_t vb, const __fp16 *src, const __fp16 *dd, __fp16 *ds,
                                   int step)
{
    svbool_t pgB32 = svptrue_b32();
    svbool_t pgB16 = svptrue_b16();
    VML_VectorComputation_DECLARE(svfloat32_t, xEven);
    VML_VectorComputation_DECLARE(svfloat32_t, xOdd);
    VML_VectorComputation_DECLARE(svfloat32_t, ddEven);
    VML_VectorComputation_DECLARE(svfloat32_t, ddOdd);
    // clip_v2 compute
    VML_VectorComputation_DECLARE(svfloat32_t, resEven);
    VML_VectorComputation_DECLARE(svfloat32_t, resOdd);
    VML_VectorComputation_DECLARE(svbool_t, pgRetZero);
    LOAD_CONVERT_FP16_TO_FP32_2(src, dd, pgB16, xEven, xOdd, ddEven, ddOdd, step);

    VML_VectorComputation_FUNC_PG_N2_2PARA(pgRetZero, svcmple_f32, pgB32, xEven, va);
    VML_VectorComputation_FUNC_NOPG_N3_3PARA(resEven, svdup_f32_m, ddEven, pgRetZero, FLOAT_ZERO);
    VML_VectorComputation_FUNC_PG_N2_2PARA(pgRetZero, svcmpge_f32, pgB32, xEven, vb);
    VML_VectorComputation_FUNC_NOPG_N3_3PARA(resEven, svdup_f32_m, resEven, pgRetZero, FLOAT_ZERO);

    VML_VectorComputation_FUNC_PG_N2_2PARA(pgRetZero, svcmple_f32, pgB32, xOdd, va);
    VML_VectorComputation_FUNC_NOPG_N3_3PARA(resOdd, svdup_f32_m, ddOdd, pgRetZero, FLOAT_ZERO);
    VML_VectorComputation_FUNC_PG_N2_2PARA(pgRetZero, svcmpge_f32, pgB32, xOdd, vb);
    VML_VectorComputation_FUNC_NOPG_N3_3PARA(resOdd, svdup_f32_m, resOdd, pgRetZero, FLOAT_ZERO);

    CONVERT_STORE_FP32_TO_FP16(ds, pgB16, resEven, resOdd, step);
}

void ClipV2SveBWDF16(const int64_t len, const float alpha, const float beta, const __fp16 *src, const __fp16 *dd,
                     __fp16 *ds)
{
    int64_t i = 0;
    int64_t loopClipV2h = svcnth();
    svfloat32_t va = svdup_n_f32(alpha);
    svfloat32_t vb = svdup_n_f32(beta);

    for (; i <= len - loopClipV2h * VML_VectorComputation_UNFOLD_N; i += loopClipV2h * VML_VectorComputation_UNFOLD_N) {
        ClipV2MainKernelBwdF16(va, vb, src + i, dd + i, ds + i, loopClipV2h);
    }
    for (; i < len; i += loopClipV2h) {
        svbool_t pgB16 = svwhilelt_b16_s64(i, len);
        svbool_t pgB32 = svwhilelt_b32_s64(i, len);

        svfloat16_t x = svld1_f16(pgB16, src + i);
        svfloat32_t xEven = svcvt_f32_f16_x(pgB16, x);
        svfloat32_t xOdd = CvtOddF32F16(x, pgB16, pgB32);

        svfloat16_t ddX = svld1_f16(pgB16, dd + i);
        svfloat32_t ddEven = svcvt_f32_f16_x(pgB16, ddX);
        svfloat32_t ddOdd = CvtOddF32F16(ddX, pgB16, pgB32);

        // clip_v2 compute
        svfloat32_t resEven = svdup_f32_m(ddEven, svcmpge_f32(pgB32, xEven, vb), FLOAT_ZERO);
        resEven = svdup_f32_m(resEven, svcmple_f32(pgB32, xEven, va), FLOAT_ZERO);

        svfloat32_t resOdd = svdup_f32_m(ddOdd, svcmpge_f32(pgB32, xOdd, vb), FLOAT_ZERO);
        resOdd = svdup_f32_m(resOdd, svcmple_f32(pgB32, xOdd, va), FLOAT_ZERO);

        // convert back to fp16
        svfloat16_t res = svcvt_f16_f32_x(pgB32, resEven); /* resEven(f16) */
        x = CvtOddF16F32(resOdd, pgB32);                   /* resOdd(f16) */
        res = svsel_f16(pgB32, res, x);

        svst1_f16(pgB16, ds + i, res);
    }
}
