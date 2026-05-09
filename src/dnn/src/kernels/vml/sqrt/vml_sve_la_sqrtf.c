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

#include "kernels/vml/common_macro.h"
#include "kernels/vml/vml_sve_base.h"

static void SqrtLaSveKernelFWDF32(const float *src, float *dst, int step)
{
    svbool_t pg = svptrue_b32();
    VML_VectorComputation_DECLARE(svfloat32_t, x);
    VML_VectorComputation_LD_STEP(x, svld1_f32, pg, src, step);
    VML_VectorComputation_DECLARE(svfloat32_t, res);
    VML_VectorComputation_FUNC_PG_1PARA(res, svsqrt_f32_x, pg, x);
    VML_VectorComputation_ST_STEP(svst1_f32, pg, dst, step, res);
}

void SqrtLaSveFWDF32(const int64_t len, const float *src, float *dst)
{
    int64_t i = 0;
    int64_t loopClipSqrtf = svcntw();
    for (; i <= len - loopClipSqrtf * VML_VectorComputation_UNFOLD_N;
         i += loopClipSqrtf * VML_VectorComputation_UNFOLD_N) {
        SqrtLaSveKernelFWDF32(src + i, dst + i, loopClipSqrtf);
    }
    for (; i < len; i += loopClipSqrtf) {
        svbool_t pg = svwhilelt_b32_s64(i, len);
        svfloat32_t x = svld1_f32(pg, src + i);
        svfloat32_t res = svsqrt_f32_x(pg, x);
        svst1_f32(pg, dst + i, res);
    }
}

void SqrtLaSveBWDF32(const int64_t len, const float *src, const float *dd, float *ds)
{
    int64_t i;
    int loopSqrtf = svcntw();
    for (i = 0; i < len; i += loopSqrtf) {
        svbool_t pg = svwhilelt_b32_s64(i, len);
        svfloat32_t srcX = svld1_f32(pg, src + i);
        svfloat32_t res = svld1_f32(pg, dd + i);
        srcX = svsqrt_f32_x(pg, srcX);
        res = svmul_f32_x(pg, res, VectorComputation_FLOAT_HALF);
        res = svdiv_f32_x(pg, res, srcX);

        svst1_f32(pg, ds + i, res);
    }
}
