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
#define VML_VC_UNFOLD_N 6
#include "kernels/vml/common_macro.h"

static void EluMainKernelFwdF32(svfloat32_t va, const float *src, float *dst, int step)
{
    VML_VC_DECLARE(svfloat32_t, x);
    svbool_t pg = svptrue_b32();

    VML_VC_LD_STEP(x, svld1_f32, pg, src, step);
    VML_VC_DECLARE(svbool_t, pgRetX);
    VML_VC_FUNC_PG_N2_2PARA(pgRetX, svcmpgt, pg, x, VC_FLOAT_ZERO);

    // exp
    VML_VC_DECLARE(svfloat32_t, res);
    VML_VC_EXP(pg, x, res);

    VML_VC_FUNC_PG_N2_2PARA(res, svsub_f32_x, pg, res, VC_FLOAT_ONE);
    VML_VC_FUNC_PG_N2_2PARA(res, svmul_f32_x, pg, res, va);
    VML_VC_FUNC_NOPG_3PARA(res, svsel_f32, pgRetX, x, res);

    VML_VC_ST_STEP(svst1_f32, pg, dst, step, res);
}

void EluLaVCFWDF32(const int64_t len, const float alpha, const float *src, float *dst)
{
    int64_t i;
    int loopEluf = svcntw();
    svfloat32_t va = svdup_n_f32(alpha);
    for (i = 0; i <= len - loopEluf * VML_VC_UNFOLD_N; i += loopEluf * VML_VC_UNFOLD_N) {
        EluMainKernelFwdF32(va, src + i, dst + i, loopEluf);
    }
    for (; i < len; i += loopEluf) {
        svbool_t pg = svwhilelt_b32_s64(i, len);
        svfloat32_t x = svld1_f32(pg, src + i);
        svbool_t pgRetX = svcmpgt(pg, x, VC_FLOAT_ZERO);
        svfloat32_t res = SvExpF32X(x, pg);
        res = svsub_f32_x(pg, res, VC_FLOAT_ONE);
        res = svmul_f32_x(pg, res, va);
        res = svsel_f32(pgRetX, x, res);
        svst1_f32(pg, dst + i, res);
    }
}

static void EluMainKernelBwdF32(svfloat32_t va, const float *src, const float *dd, float *ds, int step)
{
    VML_VC_DECLARE(svfloat32_t, x);
    VML_VC_DECLARE(svfloat32_t, ddX);
    svbool_t pg = svptrue_b32();

    VML_VC_LD_STEP(x, svld1_f32, pg, src, step);
    VML_VC_LD_STEP(ddX, svld1_f32, pg, dd, step);
    VML_VC_DECLARE(svbool_t, pgRetDdx);
    VML_VC_FUNC_PG_N2_2PARA(pgRetDdx, svcmpgt, pg, x, VC_FLOAT_ZERO);

    // exp
    VML_VC_DECLARE(svfloat32_t, res);
    VML_VC_EXP(pg, x, res);

    VML_VC_FUNC_PG_N1_2PARA(res, svmul_f32_x, pg, va, res);
    VML_VC_FUNC_PG_2PARA(res, svmul_f32_x, pg, ddX, res);

    VML_VC_FUNC_NOPG_3PARA(res, svsel_f32, pgRetDdx, ddX, res);

    VML_VC_ST_STEP(svst1_f32, pg, ds, step, res);
}

void EluLaVCBWDF32(const int64_t len, const float alpha, const float *src, const float *dd, float *ds)
{
    int64_t i;
    int loopEluf = svcntw();
    svfloat32_t va = svdup_n_f32(alpha);
    for (i = 0; i <= len - loopEluf * VML_VC_UNFOLD_N; i += loopEluf * VML_VC_UNFOLD_N) {
        EluMainKernelBwdF32(va, src + i, dd + i, ds + i, loopEluf);
    }
    for (; i < len; i += loopEluf) {
        svbool_t pg = svwhilelt_b32_s64(i, len);
        svfloat32_t srcX = svld1_f32(pg, src + i);
        svbool_t pgRetDD = svcmpgt_n_f32(pg, srcX, FLOAT_ZERO);
        svfloat32_t ddX = svld1_f32(pg, dd + i);
        svfloat32_t res = svmul_f32_x(pg, ddX, va);

        svfloat32_t partialX = SvExpF32X(srcX, pg);
        res = svmul_f32_x(pg, res, partialX);

        res = svsel_f32(pgRetDD, ddX, res);

        svst1_f32(pg, ds + i, res);
    }
}
