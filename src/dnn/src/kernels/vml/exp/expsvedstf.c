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

static void ExpMainKernelF32(const float *src, const float *dd, int step, float *ds)
{
    svbool_t pg = svptrue_b32();
    VML_VC_DECLARE(svfloat32_t, srcX);
    VML_VC_DECLARE(svfloat32_t, ddX);
    VML_VC_LD_STEP(srcX, svld1, pg, src, step);
    VML_VC_LD_STEP(ddX, svld1, pg, dd, step);
    VML_VC_FUNC_PG_2PARA(srcX, svmul_f32_x, pg, srcX, ddX);
    VML_VC_ST_STEP(svst1, pg, ds, step, srcX);
}

void ExpLaDstVCBWDF32(const int64_t len, const float *src, const float *dd, float *ds)
{
    int size = svcntw();
    int64_t i = 0;
    for (; i <= len - size * VML_VC_UNFOLD_N; i += size * VML_VC_UNFOLD_N) {
        ExpMainKernelF32(src + i, dd + i, size, ds + i);
    }

    for (; i < len; i += size) {
        svbool_t pg = svwhilelt_b32(i, len);
        svfloat32_t vec1 = svld1(pg, src + i);
        svfloat32_t vec2 = svld1(pg, dd + i);
        svfloat32_t vecResult = svmul_f32_x(pg, vec1, vec2);
        svst1(pg, ds + i, vecResult);
    }
}