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

static void HardSwishMainKernelFwdF32(svfloat32_t va, svfloat32_t vb, const float *src, float *dst, int step)
{
    svbool_t pg = svptrue_b32();
    VML_VC_DECLARE(svfloat32_t, x);
    VML_VC_LD_STEP(x, svld1_f32, pg, src, step);

    VML_VC_DECLARE(svfloat32_t, res);
    VML_VC_FUNC_PG_N13_3PARA(res, svmad_f32_x, pg, va, x, vb);
    VML_VC_FUNC_PG_N1_2PARA(res, svmin_f32_x, pg, VC_FLOAT_ONE, res);
    VML_VC_FUNC_PG_N1_2PARA(res, svmax_f32_x, pg, VC_FLOAT_ZERO, res);
    VML_VC_FUNC_PG_2PARA(res, svmul_f32_x, pg, res, x);

    VML_VC_ST_STEP(svst1_f32, pg, dst, step, res);
}

void HardSwishVCFWDF32(const int64_t len, const float alpha, const float beta, const float *src, float *dst)
{
    int64_t i;
    int64_t loopHardSwishf = svcntw();
    svfloat32_t va = svdup_n_f32(alpha);
    svfloat32_t vb = svdup_n_f32(beta);

    for (i = 0; i <= len - loopHardSwishf * VML_VC_UNFOLD_N; i += loopHardSwishf * VML_VC_UNFOLD_N) {
        HardSwishMainKernelFwdF32(va, vb, src + i, dst + i, loopHardSwishf);
    }
    for (; i < len; i += loopHardSwishf) {
        svbool_t pg = svwhilelt_b32_s64(i, len);
        svfloat32_t x = svld1_f32(pg, src + i);

        svfloat32_t res = svmad_f32_x(pg, va, x, vb); /* alpha * s + beta */
        res = svmin_f32_x(pg, VC_FLOAT_ONE, res); /* min(1, alpha * s + beta) */
        res = svmax_f32_x(pg, VC_FLOAT_ZERO, res); /* max(0, min(1, alpha * s + beta)) */
        res = svmul_f32_x(pg, res, x); /* s * max(0, min(1, alpha * s + beta)) */
        svst1_f32(pg, dst + i, res);
    }
}

static void HardSwishMainKernelBwdF32(svfloat32_t va, svfloat32_t vb, svfloat32_t halfVb,
                                      const float *src, const float *dd, float *ds, int step)
{
    svbool_t pg = svptrue_b32();
    VML_VC_DECLARE(svfloat32_t, x);
    VML_VC_LD_STEP(x, svld1_f32, pg, src, step);
    VML_VC_DECLARE(svfloat32_t, ddX);
    VML_VC_LD_STEP(ddX, svld1_f32, pg, dd, step);

    VML_VC_DECLARE(svbool_t, pgRetDD);
    VML_VC_DECLARE(svbool_t, pgMain);
    VML_VC_DECLARE(svfloat32_t, res);

    VML_VC_FUNC_PG_N13_3PARA(res, svmad_f32_x, pg, va, x, vb);
    VML_VC_FUNC_PG_N2_2PARA(pgRetDD, svcmpge, pg, res, VC_FLOAT_ONE);
    VML_VC_FUNC_PG_N2_2PARA(pgMain, svcmpgt, pg, res, VC_FLOAT_ZERO);
    VML_VC_FUNC_NOPG_N3_3PARA(pgMain, svcmplt, pgMain, res, VC_FLOAT_ONE);
    VML_VC_FUNC_PG_N23_3PARA(res, svmad_f32_x, pg, x, va, halfVb);
    VML_VC_FUNC_PG_N2_2PARA(res, svmul_f32_x, pg, res, VC_FLOAT_TWO);
    VML_VC_FUNC_NOPG_3PARA(res, svmul_f32_z, pgMain, res, ddX);
    VML_VC_FUNC_NOPG_3PARA(res, svsel_f32, pgRetDD, ddX, res);

    VML_VC_ST_STEP(svst1_f32, pg, ds, step, res);
}

void HardSwishVCBWDF32(const int64_t len, const float alpha, const float beta,
                        const float *src, const float *dd, float *ds)
{
    int64_t i;
    int64_t loopHardSwishf = svcntw();
    svfloat32_t va = svdup_n_f32(alpha);
    svfloat32_t vb = svdup_n_f32(beta);
    svfloat32_t halfVb = svdup_n_f32(beta * FLOAT_HALF);

    for (i = 0; i <= len - loopHardSwishf * VML_VC_UNFOLD_N; i += loopHardSwishf * VML_VC_UNFOLD_N) {
        HardSwishMainKernelBwdF32(va, vb, halfVb, src + i, dd + i, ds + i, loopHardSwishf);
    }
    for (; i < len; i += loopHardSwishf) {
        svbool_t pg = svwhilelt_b32_s64(i, len);
        svfloat32_t x = svld1_f32(pg, src + i);
        svfloat32_t ddX = svld1_f32(pg, dd + i);

        svfloat32_t res = svmad_f32_x(pg, va, x, vb); /* alpha * s + beta */
        svbool_t pgRetDD = svcmpge(pg, res, VC_FLOAT_ONE); /* alpha * s + beta >= 1 */
        svbool_t pgMain = svcmpgt(pg, res, VC_FLOAT_ZERO); /* 0 < alpha * s + beta < 1 */
        pgMain = svcmplt(pgMain, res, VC_FLOAT_ONE); /* 0 < alpha * s + beta < 1 */
        res = svmad_f32_x(pg, x, va, halfVb); /* alpha * s + beta / 2 */
        res = svmul_f32_x(pg, res, VC_FLOAT_TWO); /* 2 * alpha * s + beta */
        res = svmul_f32_z(pgMain, res, ddX); /* dd * (2 * alpha * s + beta) for pgMain, zero for others */
        res = svsel_f32(pgRetDD, ddX, res);

        svst1_f32(pg, ds + i, res);
    }
}