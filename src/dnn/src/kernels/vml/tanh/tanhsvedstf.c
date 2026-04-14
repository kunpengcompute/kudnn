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

static void TanhKernelF32(const float *src, const float *dd, int step, float *ds)
{
    svbool_t pg = svptrue_b32();
    VML_VectorComputation_DECLARE(svfloat32_t, srcX);
    VML_VectorComputation_DECLARE(svfloat32_t, tmp);
    VML_VectorComputation_DECLARE(svfloat32_t, ddX);
    VML_VectorComputation_LD_STEP(srcX, svld1, pg, src, step);
    VML_VectorComputation_LD_STEP(ddX, svld1, pg, dd, step);

    VML_VectorComputation_FUNC_PG_N1_2PARA(tmp, svadd_f32_x, pg, VectorComputation_FLOAT_ONE, srcX);
    VML_VectorComputation_FUNC_PG_N1_2PARA(srcX, svsub_f32_x, pg, VectorComputation_FLOAT_ONE, srcX);
    VML_VectorComputation_FUNC_PG_2PARA(srcX, svmul_f32_x, pg, tmp, srcX);
    VML_VectorComputation_FUNC_PG_2PARA(srcX, svmul_f32_x, pg, ddX, srcX);
    VML_VectorComputation_ST_STEP(svst1, pg, ds, step, srcX);
}

void TanhDstSveBWDF32(const int64_t len, const float *src, const float *dd, float *ds)
{
    int64_t i = 0;
    int32_t size = svcntw();
    for (; i <= len - size * VML_VectorComputation_UNFOLD_N; i += size * VML_VectorComputation_UNFOLD_N) {
        TanhKernelF32(src + i, dd + i, size, ds + i);
    }

    for (; i < len; i += size) {
        svbool_t pg = svwhilelt_b32(i, len);
        svfloat32_t vec1 = svld1(pg, src + i);
        svfloat32_t vec2 = svld1(pg, dd + i);
        svfloat32_t vecResult = svsub_f32_x(pg, VectorComputation_FLOAT_ONE, vec1);
        svfloat32_t tmpResult = svadd_f32_x(pg, VectorComputation_FLOAT_ONE, vec1);
        vecResult = svmul_f32_x(pg, tmpResult, vecResult);
        vecResult = svmul_f32_x(pg, vecResult, vec2);
        svst1(pg, ds + i, vecResult);
    }
}