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

static void KplVmlVsExpLaVCMainN(const float *src, float *dst, int step)
{
    VML_VC_DECLARE(svfloat32_t, x);
    svbool_t pg = svptrue_b32();
    VML_VC_LD_STEP(x, svld1_f32, pg, src, step);
    VML_VC_DECLARE(svfloat32_t, res);
    VML_VC_EXP(pg, x, res);

    VML_VC_ST_STEP(svst1_f32, pg, dst, step, res);
}

void ExpLaVCFWDF32(const int64_t len, const float *src, float *dst)
{
    int64_t i;
    int loopExpf = svcntw();
    for (i = 0; i <= len - loopExpf * VML_VC_UNFOLD_N; i += loopExpf * VML_VC_UNFOLD_N) {
        KplVmlVsExpLaVCMainN(src + i, dst + i, loopExpf);
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
    VML_VC_DECLARE(svfloat32_t, srcX);
    VML_VC_DECLARE(svfloat32_t, ddX);
    svbool_t pg = svptrue_b32();

    VML_VC_LD_STEP(srcX, svld1_f32, pg, src, step);
    VML_VC_LD_STEP(ddX, svld1_f32, pg, dd, step);
    VML_VC_DECLARE(svfloat32_t, res);
    VML_VC_EXP(pg, srcX, res);
    VML_VC_FUNC_PG_2PARA(res, svmul_f32_x, pg, ddX, res);

    VML_VC_ST_STEP(svst1_f32, pg, ds, step, res);
}

void ExpLaVCBWDF32(const int64_t len, const float *src, const float *dd, float *ds)
{
    int64_t i;
    int loopExpf = svcntw();
    for (i = 0; i <= len - loopExpf * VML_VC_UNFOLD_N; i += loopExpf * VML_VC_UNFOLD_N) {
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
