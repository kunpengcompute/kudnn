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

static void ExpMainKernelF16(const __fp16 *src, const __fp16 *dd, int step, __fp16 *ds)
{
    svbool_t pg = svptrue_b16();
    VML_VectorComputation_DECLARE(svfloat16_t, srcX);
    VML_VectorComputation_DECLARE(svfloat16_t, ddX);
    VML_VectorComputation_LD_STEP(srcX, svld1_f16, pg, src, step);
    VML_VectorComputation_LD_STEP(ddX, svld1_f16, pg, dd, step);
    VML_VectorComputation_FUNC_PG_2PARA(srcX, svmul_f16_x, pg, srcX, ddX);
    VML_VectorComputation_ST_STEP(svst1_f16, pg, ds, step, srcX);
}

void ExpLaDstSveBWDF16(const int64_t len, const __fp16 *src, const __fp16 *dd, __fp16 *ds)
{
    int size = svcnth();
    int64_t i = 0;
    for (; i <= len - size * VML_VectorComputation_UNFOLD_N; i += size * VML_VectorComputation_UNFOLD_N) {
        ExpMainKernelF16(src + i, dd + i, size, ds + i);
    }

    for (; i < len; i += size) {
        svbool_t pg = svwhilelt_b16(i, len);
        svfloat16_t vec1 = svld1_f16(pg, src + i);
        svfloat16_t vec2 = svld1_f16(pg, dd + i);
        svfloat16_t vecResult = svmul_f16_x(pg, vec1, vec2);
        svst1_f16(pg, ds + i, vecResult);
    }
}