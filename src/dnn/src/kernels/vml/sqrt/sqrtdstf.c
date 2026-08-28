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

static void SqrtMainKernelBwdF32(svfloat32_t a, const float *src, const float *dd, float *ds, int step)
{
    svbool_t pg = svptrue_b32();
    VML_VC_DECLARE(svfloat32_t, src_x);
    VML_VC_DECLARE(svfloat32_t, dd_x);
    VML_VC_DECLARE(svfloat32_t, res);

    VML_VC_LD_STEP(src_x, svld1_f32, pg, src, step);

    VML_VC_LD_STEP(dd_x, svld1_f32, pg, dd, step);

    VML_VC_FUNC_PG_N2_2PARA(res, svmul_f32_m, pg, src_x, a);
    VML_VC_FUNC_PG_2PARA(res, svdiv_f32_x, pg, dd_x, res);
    VML_VC_ST_STEP(svst1_f32, pg, ds, step, res);
}

void SqrtDstVCBWDF32(const int64_t len, const float *src, const float *dd, float *ds)
{
    int64_t i = 0;
    int64_t loopSqrtDstf = svcntw();
    svfloat32_t va = svdup_n_f32(FLOAT_TWO);

    for (; i <= len - loopSqrtDstf * VML_VC_UNFOLD_N; i += loopSqrtDstf * VML_VC_UNFOLD_N) {
        SqrtMainKernelBwdF32(va, src + i, dd + i, ds + i, loopSqrtDstf);
    }
    for (; i < len; i += loopSqrtDstf) {
        svbool_t pg = svwhilelt_b32_s64(i, len);
        svfloat32_t srcX = svld1_f32(pg, src + i);
        svfloat32_t ddX = svld1_f32(pg, dd + i);

        svfloat32_t res = svmul_f32_m(pg, srcX, va);
        res = svdiv_f32_x(pg, ddX, res);

        svst1_f32(pg, ds + i, res);
    }
}