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

#include "kernels/vml/vml_sve_la_tanh.h"
#define VML_VectorComputation_UNFOLD_N 8
#include "kernels/vml/common_macro.h"

static void KplVmlVsTanhLaSveMainN(const float *src, float *dst, int step)
{
    svbool_t pg = svptrue_b32();
    VML_VectorComputation_DECLARE(svfloat32_t, x);
    VML_VectorComputation_LD_STEP(x, svld1_f32, pg, src, step);

    VML_VectorComputation_DECLARE(svfloat32_t, f2x);
    VML_VectorComputation_FUNC_PG_N2_2PARA(f2x, svmul_f32_x, pg, x, VectorComputation_FLOAT_TWO);
    VML_VectorComputation_DECLARE(svfloat32_t, e2m1);
    VML_VectorComputation_FUNC_PG_1PARA(e2m1, VsExpm1InlineSve, pg, f2x);
    VML_VectorComputation_DECLARE(svfloat32_t, e2a1);
    VML_VectorComputation_FUNC_PG_N2_2PARA(e2a1, svadd_f32_x, pg, e2m1, VectorComputation_FLOAT_TWO);
    VML_VectorComputation_DECLARE(svfloat32_t, res);
    VML_VectorComputation_FUNC_PG_2PARA(res, svdiv_f32_x, pg, e2m1, e2a1);

    VML_VectorComputation_DECLARE(svbool_t, cmpSp);
    VML_VectorComputation_FUNC_PG_N2_2PARA(cmpSp, svcmpge_f32, pg, x, VectorComputation_VSTANH_SP_TH);
    VML_VectorComputation_FUNC_NOPG_N3_3PARA(res, svdup_n_f32_m, res, cmpSp, FLOAT_ONE);

    VML_VectorComputation_DECLARE(svbool_t, cmpSpNeg);
    VML_VectorComputation_FUNC_PG_N2_2PARA(cmpSpNeg, svcmple_f32, pg, x, VectorComputation_VSTANH_SP_TH_NEG);
    VML_VectorComputation_FUNC_NOPG_N3_3PARA(res, svdup_n_f32_m, res, cmpSpNeg, FLOAT_NEG_ONE);

    VML_VectorComputation_DECLARE(svbool_t, cmpEqZero);
    VML_VectorComputation_FUNC_PG_N2_2PARA(cmpEqZero, svcmpeq_n_f32, pg, x, FLOAT_ZERO);
    VML_VectorComputation_FUNC_NOPG_3PARA(res, svsel_f32, cmpEqZero, x, res);

    VML_VectorComputation_ST_STEP(svst1_f32, pg, dst, step, res);
}

void TanhLaSveFWDF32(const int64_t len, const float *src, float *dst)
{
    int64_t i;
    int loopTanhf = svcntw();
    for (i = 0; i <= len - loopTanhf * VML_VectorComputation_UNFOLD_N; i += loopTanhf * VML_VectorComputation_UNFOLD_N) {
        KplVmlVsTanhLaSveMainN(src + i, dst + i, loopTanhf);
    }
    for (; i < len; i += loopTanhf) {
        svbool_t pg = svwhilelt_b32_s64(i, len);
        svfloat32_t x = svld1_f32(pg, src + i);
        svfloat32_t y = SvTanhF32X(x, pg);
        svst1_f32(pg, dst + i, y);
    }
}

void TanhLaSveBWDF32(const int64_t len, const float *src, const float *dd, float *ds)
{
    int64_t i;
    int loopTanhf = svcntw();
    for (i = 0; i < len; i += loopTanhf) {
        svbool_t pg = svwhilelt_b32_s64(i, len);
        svfloat32_t srcX = svld1_f32(pg, src + i);
        svfloat32_t ddX = svld1_f32(pg, dd + i);

        svfloat32_t tanhX = SvTanhF32X(srcX, pg);
        svfloat32_t res = svmls_f32_x(pg, VectorComputation_FLOAT_ONE, tanhX, tanhX);
        res = svmul_f32_x(pg, ddX, res);
        svst1_f32(pg, ds + i, res);
    }
}
