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

static void ReluMainKernelBwdF32(svfloat32_t a, const float *src, const float *dd, float *ds, int step)
{
    svbool_t pg = svptrue_b32();
    VML_VectorComputation_DECLARE(svfloat32_t, src_x);
    VML_VectorComputation_DECLARE(svfloat32_t, dd_x);
    VML_VectorComputation_DECLARE(svfloat32_t, res);
    VML_VectorComputation_DECLARE(svbool_t, mask_lez);

    VML_VectorComputation_LD_STEP(src_x, svld1_f32, pg, src, step);
    VML_VectorComputation_FUNC_PG_N2_2PARA(mask_lez, svcmple_n_f32, pg, src_x, 0.0f);

    VML_VectorComputation_LD_STEP(dd_x, svld1_f32, pg, dd, step);

    VML_VectorComputation_FUNC_UPG_N2_2PARA(res, svmul_f32_m, mask_lez, dd_x, a);

    VML_VectorComputation_ST_STEP(svst1_f32, pg, ds, step, res);
}

void ReluDstSveBWDF32(const int64_t len, const float alpha, const float *src, const float *dd, float *ds)
{
    int64_t i = 0;
    int64_t loopReluDstf = svcntw();
    svfloat32_t va = svdup_n_f32(alpha);

    for (; i <= len - loopReluDstf * VML_VectorComputation_UNFOLD_N;
         i += loopReluDstf * VML_VectorComputation_UNFOLD_N) {
        ReluMainKernelBwdF32(va, src + i, dd + i, ds + i, loopReluDstf);
    }
    for (; i < len; i += loopReluDstf) {
        svbool_t pg = svwhilelt_b32_s64(i, len);
        svfloat32_t srcX = svld1_f32(pg, src + i);
        svfloat32_t ddX = svld1_f32(pg, dd + i);

        svbool_t maskLez = svcmple_n_f32(pg, srcX, 0.0f);
        svfloat32_t res = svmul_f32_m(maskLez, ddX, va);

        svst1_f32(pg, ds + i, res);
    }
}
