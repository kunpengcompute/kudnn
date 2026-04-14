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

static void EluMainKernelFwdF32(svfloat32_t va, const float *src, float *dst, int step)
{
    VML_VectorComputation_DECLARE(svfloat32_t, x);
    svbool_t pg = svptrue_b32();

    VML_VectorComputation_LD_STEP(x, svld1_f32, pg, src, step);
    VML_VectorComputation_DECLARE(svbool_t, pgRetZero);
    VML_VectorComputation_FUNC_PG_N2_2PARA(pgRetZero, svcmple, pg, x, VectorComputation_EXPF_LOWBOUND);
    VML_VectorComputation_DECLARE(svbool_t, pgRetX);
    VML_VectorComputation_FUNC_PG_N2_2PARA(pgRetX, svcmpgt, pg, x, VectorComputation_FLOAT_ZERO);

    // exp
    VML_VectorComputation_DECLARE(svfloat32_t, ki);
    VML_VectorComputation_FUNC_PG_N23_3PARA(ki, svmad_f32_x, pg, x, VectorComputation_EXPF_INVLN2HI, VectorComputation_EXP2F_SHIFT);
    VML_VectorComputation_DECLARE(svfloat32_t, kd);
    VML_VectorComputation_FUNC_PG_N2_2PARA(kd, svsub_f32_x, pg, ki, VectorComputation_EXP2F_SHIFT);
    VML_VectorComputation_DECLARE(svuint32_t, k);
    VML_VectorComputation_FUNC_NOPG_1PARA(k, svreinterpret_u32_f32, ki);
    VML_VectorComputation_DECLARE(svfloat32_t, expk);
    VML_VectorComputation_FUNC_NOPG_1PARA(expk, svexpa_f32, k);
    VML_VectorComputation_DECLARE(svfloat32_t, r);
    VML_VectorComputation_FUNC_PG_N3_3PARA(r, svnmls_f32_x, pg, kd, x, VectorComputation_EXPF_INVLN2HI);
    VML_VectorComputation_FUNC_PG_N3_3PARA(r, svmla_f32_x, pg, r, x, VectorComputation_EXPF_INVLN2LO);
    VML_VectorComputation_DECLARE(svfloat32_t, poly);
    VML_VectorComputation_FUNC_PG_N13_3PARA(poly, svmla_f32_x, pg, VectorComputation_EXP2F_C1, r, VectorComputation_EXP2F_C0);
    VML_VectorComputation_FUNC_PG_N1_3PARA(poly, svmla_f32_x, pg, VectorComputation_EXP2F_C2, poly, r);
    VML_VectorComputation_DECLARE(svfloat32_t, res);
    VML_VectorComputation_FUNC_PG_2PARA(res, svmul_f32_x, pg, expk, poly);
    VML_VectorComputation_FUNC_NOPG_N2_3PARA(res, svsel_f32, pgRetZero, VectorComputation_FLOAT_ZERO, res);

    VML_VectorComputation_FUNC_PG_N13_3PARA(res, svnmls_f32_x, pg, va, res, va);
    VML_VectorComputation_FUNC_NOPG_3PARA(res, svsel_f32, pgRetX, x, res);

    VML_VectorComputation_ST_STEP(svst1_f32, pg, dst, step, res);
}

void EluLaSveFWDF32(const int64_t len, const float alpha, const float *src, float *dst)
{
    int64_t i;
    int loopEluf = svcntw();
    svfloat32_t va = svdup_n_f32(alpha);
    for (i = 0; i <= len - loopEluf * VML_VectorComputation_UNFOLD_N; i += loopEluf * VML_VectorComputation_UNFOLD_N) {
        EluMainKernelFwdF32(va, src + i, dst + i, loopEluf);
    }
    for (; i < len; i += loopEluf) {
        svbool_t pg = svwhilelt_b32_s64(i, len);
        svfloat32_t x = svld1_f32(pg, src + i);
        svbool_t pgRetX = svcmpgt(pg, x, VectorComputation_FLOAT_ZERO);
        svfloat32_t res = SvExpForNegF32X(x, pg);
        res = svnmls_f32_x(pg, va, res, va);
        res = svsel_f32(pgRetX, x, res);
        svst1_f32(pg, dst + i, res);
    }
}

static void EluMainKernelBwdF32(svfloat32_t va, const float *src, const float *dd, float *ds, int step)
{
    VML_VectorComputation_DECLARE(svfloat32_t, x);
    VML_VectorComputation_DECLARE(svfloat32_t, ddX);
    svbool_t pg = svptrue_b32();

    VML_VectorComputation_LD_STEP(x, svld1_f32, pg, src, step);
    VML_VectorComputation_LD_STEP(ddX, svld1_f32, pg, dd, step);
    VML_VectorComputation_DECLARE(svbool_t, pgRetZero);
    VML_VectorComputation_FUNC_PG_N2_2PARA(pgRetZero, svcmple, pg, x, VectorComputation_EXPF_LOWBOUND);
    VML_VectorComputation_DECLARE(svbool_t, pgRetDdx);
    VML_VectorComputation_FUNC_PG_N2_2PARA(pgRetDdx, svcmpgt, pg, x, VectorComputation_FLOAT_ZERO);

    // exp
    VML_VectorComputation_DECLARE(svfloat32_t, ki);
    VML_VectorComputation_FUNC_PG_N23_3PARA(ki, svmad_f32_x, pg, x, VectorComputation_EXPF_INVLN2HI, VectorComputation_EXP2F_SHIFT);
    VML_VectorComputation_DECLARE(svfloat32_t, kd);
    VML_VectorComputation_FUNC_PG_N2_2PARA(kd, svsub_f32_x, pg, ki, VectorComputation_EXP2F_SHIFT);
    VML_VectorComputation_DECLARE(svuint32_t, k);
    VML_VectorComputation_FUNC_NOPG_1PARA(k, svreinterpret_u32_f32, ki);
    VML_VectorComputation_DECLARE(svfloat32_t, expk);
    VML_VectorComputation_FUNC_NOPG_1PARA(expk, svexpa_f32, k);
    VML_VectorComputation_DECLARE(svfloat32_t, r);
    VML_VectorComputation_FUNC_PG_N3_3PARA(r, svnmls_f32_x, pg, kd, x, VectorComputation_EXPF_INVLN2HI);
    VML_VectorComputation_FUNC_PG_N3_3PARA(r, svmla_f32_x, pg, r, x, VectorComputation_EXPF_INVLN2LO);
    VML_VectorComputation_DECLARE(svfloat32_t, poly);
    VML_VectorComputation_FUNC_PG_N13_3PARA(poly, svmla_f32_x, pg, VectorComputation_EXP2F_C1, r, VectorComputation_EXP2F_C0);
    VML_VectorComputation_FUNC_PG_N1_3PARA(poly, svmla_f32_x, pg, VectorComputation_EXP2F_C2, poly, r);
    VML_VectorComputation_DECLARE(svfloat32_t, partialX);
    VML_VectorComputation_FUNC_PG_2PARA(partialX, svmul_f32_x, pg, expk, poly);
    VML_VectorComputation_FUNC_NOPG_N2_3PARA(partialX, svsel_f32, pgRetZero, VectorComputation_FLOAT_ZERO, partialX);

    VML_VectorComputation_DECLARE(svfloat32_t, res);
    VML_VectorComputation_FUNC_PG_N1_2PARA(res, svmul_f32_x, pg, va, partialX);
    VML_VectorComputation_FUNC_PG_2PARA(res, svmul_f32_x, pg, ddX, res);

    VML_VectorComputation_FUNC_NOPG_3PARA(res, svsel_f32, pgRetDdx, ddX, res);

    VML_VectorComputation_ST_STEP(svst1_f32, pg, ds, step, res);
}

void EluLaSveBWDF32(const int64_t len, const float alpha, const float *src, const float *dd, float *ds)
{
    int64_t i;
    int loopEluf = svcntw();
    svfloat32_t va = svdup_n_f32(alpha);
    for (i = 0; i <= len - loopEluf * VML_VectorComputation_UNFOLD_N; i += loopEluf * VML_VectorComputation_UNFOLD_N) {
        EluMainKernelBwdF32(va, src + i, dd + i, ds + i, loopEluf);
    }
    for (; i < len; i += loopEluf) {
        svbool_t pg = svwhilelt_b32_s64(i, len);
        svfloat32_t srcX = svld1_f32(pg, src + i);
        svbool_t pgRetDD = svcmpgt_n_f32(pg, srcX, FLOAT_ZERO);
        svfloat32_t ddX = svld1_f32(pg, dd + i);
        svfloat32_t res = svmul_f32_x(pg, ddX, va);

        svfloat32_t partialX = SvExpForNegF32X(srcX, pg);
        res = svmul_f32_x(pg, res, partialX);

        res = svsel_f32(pgRetDD, ddX, res);

        svst1_f32(pg, ds + i, res);
    }
}
