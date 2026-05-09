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

#include "kernels/vml/vml_sve_la_exp.h"
#define VML_VectorComputation_UNFOLD_N 6
#include "kernels/vml/common_macro.h"

static void KplVmlVsExpLaSveMainN(const float *src, float *dst, int step)
{
    VML_VectorComputation_DECLARE(svfloat32_t, x);
    svbool_t pg = svptrue_b32();

    VML_VectorComputation_LD_STEP(x, svld1_f32, pg, src, step);
    VML_VectorComputation_DECLARE(svbool_t, pgFixZero);
    VML_VectorComputation_FUNC_PG_N2_2PARA(pgFixZero, svcmple, pg, x, VectorComputation_EXPF_LOWBOUND);
    VML_VectorComputation_DECLARE(svbool_t, pgFixInf);
    VML_VectorComputation_FUNC_PG_N2_2PARA(pgFixInf, svcmpge, pg, x, VectorComputation_EXPF_UPBOUND);
    VML_VectorComputation_DECLARE(svbool_t, pgSpR);
    VML_VectorComputation_FUNC_PG_N2_2PARA(pgSpR, svcmpge, pg, x, VectorComputation_EXPF_TH_R);
    VML_VectorComputation_FUNC_NOPG_N3_3PARA(x, svsub_f32_m, pgSpR, x, VectorComputation_FLOAT_TWO);

    VML_VectorComputation_DECLARE(svfloat32_t, ki);
    VML_VectorComputation_FUNC_PG_N23_3PARA(ki, svmad_f32_x, pg, x, VectorComputation_EXPF_INVLN2HI,
                                            VectorComputation_EXP2F_SHIFT);
    VML_VectorComputation_DECLARE(svfloat32_t, kd);
    VML_VectorComputation_FUNC_PG_N2_2PARA(kd, svsub_f32_x, pg, ki, VectorComputation_EXP2F_SHIFT);
    VML_VectorComputation_DECLARE(svuint32_t, k);
    VML_VectorComputation_FUNC_NOPG_1PARA(k, svreinterpret_u32_f32, ki);
    VML_VectorComputation_DECLARE(svfloat32_t, expk);
    VML_VectorComputation_FUNC_NOPG_1PARA(expk, svexpa_f32, k);
    VML_VectorComputation_DECLARE(svfloat32_t, r);
    VML_VectorComputation_FUNC_PG_N3_3PARA(r, svnmls_f32_x, pg, kd, x, VectorComputation_EXPF_INVLN2HI);
    VML_VectorComputation_FUNC_PG_N3_3PARA(r, svmla_f32_x, pg, r, x, VectorComputation_EXPF_INVLN2LO);

    // polynomial of exp2
    VML_VectorComputation_DECLARE(svfloat32_t, poly);
    VML_VectorComputation_FUNC_PG_N13_3PARA(poly, svmla_f32_x, pg, VectorComputation_EXP2F_C1, r,
                                            VectorComputation_EXP2F_C0);
    VML_VectorComputation_FUNC_PG_N1_3PARA(poly, svmla_f32_x, pg, VectorComputation_EXP2F_C2, poly, r);
    VML_VectorComputation_DECLARE(svfloat32_t, res);
    VML_VectorComputation_FUNC_PG_2PARA(res, svmul_f32_x, pg, expk, poly);
    VML_VectorComputation_FUNC_NOPG_N3_3PARA(res, svmul_n_f32_m, pgSpR, res, 0x1.d8e64cp2f);
    VML_VectorComputation_FUNC_NOPG_N3_3PARA(res, svdup_f32_m, res, pgFixZero, FLOAT_ZERO);
    VML_VectorComputation_FUNC_NOPG_N3_3PARA(res, svdup_f32_m, res, pgFixInf, INFINITY);

    VML_VectorComputation_ST_STEP(svst1_f32, pg, dst, step, res);
}

void ExpLaSveFWDF32(const int64_t len, const float *src, float *dst)
{
    int64_t i;
    int loopExpf = svcntw();
    for (i = 0; i <= len - loopExpf * VML_VectorComputation_UNFOLD_N; i += loopExpf * VML_VectorComputation_UNFOLD_N) {
        KplVmlVsExpLaSveMainN(src + i, dst + i, loopExpf);
    }
    for (; i < len; i += loopExpf) {
        svbool_t pg = svwhilelt_b32_s64(i, len);
        svfloat32_t x = svld1_f32(pg, src + i);
        svfloat32_t y = SvExpF32X(x, pg);
        svst1_f32(pg, dst + i, y);
    }
}

static void ExpMainKernelBwdF32(const float *src, const float *dd, float *ds, int step)
{
    VML_VectorComputation_DECLARE(svfloat32_t, srcX);
    VML_VectorComputation_DECLARE(svfloat32_t, ddX);
    svbool_t pg = svptrue_b32();

    VML_VectorComputation_LD_STEP(srcX, svld1_f32, pg, src, step);
    VML_VectorComputation_LD_STEP(ddX, svld1_f32, pg, dd, step);
    VML_VectorComputation_DECLARE(svbool_t, pgRetZero);
    VML_VectorComputation_FUNC_PG_N2_2PARA(pgRetZero, svcmple, pg, srcX, VectorComputation_EXPF_LOWBOUND);
    VML_VectorComputation_DECLARE(svbool_t, pgRetInf);
    VML_VectorComputation_FUNC_PG_N2_2PARA(pgRetInf, svcmpge, pg, srcX, VectorComputation_EXPF_UPBOUND);
    VML_VectorComputation_DECLARE(svbool_t, pgSpR);
    VML_VectorComputation_FUNC_PG_N2_2PARA(pgSpR, svcmpge, pg, srcX, VectorComputation_EXPF_TH_R);
    VML_VectorComputation_FUNC_NOPG_N3_3PARA(srcX, svsub_f32_m, pgSpR, srcX, VectorComputation_FLOAT_TWO);

    VML_VectorComputation_DECLARE(svfloat32_t, ki);
    VML_VectorComputation_FUNC_PG_N23_3PARA(ki, svmad_f32_x, pg, srcX, VectorComputation_EXPF_INVLN2HI,
                                            VectorComputation_EXP2F_SHIFT);
    VML_VectorComputation_DECLARE(svfloat32_t, kd);
    VML_VectorComputation_FUNC_PG_N2_2PARA(kd, svsub_f32_x, pg, ki, VectorComputation_EXP2F_SHIFT);
    VML_VectorComputation_DECLARE(svuint32_t, k);
    VML_VectorComputation_FUNC_NOPG_1PARA(k, svreinterpret_u32_f32, ki);
    VML_VectorComputation_DECLARE(svfloat32_t, expk);
    VML_VectorComputation_FUNC_NOPG_1PARA(expk, svexpa_f32, k);
    VML_VectorComputation_DECLARE(svfloat32_t, r);
    VML_VectorComputation_FUNC_PG_N3_3PARA(r, svnmls_f32_x, pg, kd, srcX, VectorComputation_EXPF_INVLN2HI);
    VML_VectorComputation_FUNC_PG_N3_3PARA(r, svmla_f32_x, pg, r, srcX, VectorComputation_EXPF_INVLN2LO);

    // polynomial of exp2
    VML_VectorComputation_DECLARE(svfloat32_t, poly);
    VML_VectorComputation_FUNC_PG_N13_3PARA(poly, svmla_f32_x, pg, VectorComputation_EXP2F_C1, r,
                                            VectorComputation_EXP2F_C0);
    VML_VectorComputation_FUNC_PG_N1_3PARA(poly, svmla_f32_x, pg, VectorComputation_EXP2F_C2, poly, r);
    VML_VectorComputation_DECLARE(svfloat32_t, partialX);
    VML_VectorComputation_FUNC_PG_2PARA(partialX, svmul_f32_x, pg, expk, poly);
    VML_VectorComputation_FUNC_NOPG_N3_3PARA(partialX, svmul_n_f32_m, pgSpR, partialX, 0x1.d8e64cp2f); /* exp(2) */

    VML_VectorComputation_FUNC_NOPG_N3_3PARA(partialX, svdup_f32_m, partialX, pgRetZero, FLOAT_ZERO);
    VML_VectorComputation_FUNC_NOPG_N3_3PARA(partialX, svdup_f32_m, partialX, pgRetInf, INFINITY);
    VML_VectorComputation_DECLARE(svfloat32_t, res);
    VML_VectorComputation_FUNC_PG_2PARA(res, svmul_f32_x, pg, ddX, partialX);

    VML_VectorComputation_ST_STEP(svst1_f32, pg, ds, step, res);
}

void ExpLaSveBWDF32(const int64_t len, const float *src, const float *dd, float *ds)
{
    int64_t i;
    int loopExpf = svcntw();
    for (i = 0; i <= len - loopExpf * VML_VectorComputation_UNFOLD_N; i += loopExpf * VML_VectorComputation_UNFOLD_N) {
        ExpMainKernelBwdF32(src + i, dd + i, ds + i, loopExpf);
    }
    for (; i < len; i += loopExpf) {
        svbool_t pg = svwhilelt_b32_s64(i, len);
        svfloat32_t srcX = svld1_f32(pg, src + i);
        svfloat32_t ddX = svld1_f32(pg, dd + i);
        svfloat32_t partialX = SvExpF32X(srcX, pg);

        svfloat32_t res = svmul_f32_x(pg, ddX, partialX);
        svst1_f32(pg, ds + i, res);
    }
}
