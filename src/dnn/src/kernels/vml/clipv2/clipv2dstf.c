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

static void ClipV2DstMainKernelBwdF32(float alpha, float beta, const float *src, const float *dd, float *ds, int step)
{
    svbool_t pg = svptrue_b32();
    VML_VectorComputation_DECLARE(svfloat32_t, srcX);
    VML_VectorComputation_LD_STEP(srcX, svld1_f32, pg, src, step);
    VML_VectorComputation_DECLARE(svfloat32_t, res);
    VML_VectorComputation_LD_STEP(res, svld1_f32, pg, dd, step);

    VML_VectorComputation_DECLARE(svbool_t, pgRetZero);
    VML_VectorComputation_FUNC_PG_N2_2PARA(pgRetZero, svcmple_n_f32, pg, srcX, alpha);
    VML_VectorComputation_FUNC_NOPG_N3_3PARA(res, svdup_f32_m, res, pgRetZero, FLOAT_ZERO);
    VML_VectorComputation_FUNC_PG_N2_2PARA(pgRetZero, svcmpge_n_f32, pg, srcX, beta);
    VML_VectorComputation_FUNC_NOPG_N3_3PARA(res, svdup_f32_m, res, pgRetZero, FLOAT_ZERO);

    VML_VectorComputation_ST_STEP(svst1_f32, pg, ds, step, res);
}

void ClipV2DstSveBWDF32(const int64_t len, const float alpha, const float beta, const float *src, const float *dd,
    float *ds)
{
    int64_t i = 0;
    int64_t loopClipV2Dstf = svcntw();

    for (; i <= len - loopClipV2Dstf * VML_VectorComputation_UNFOLD_N; i += loopClipV2Dstf * VML_VectorComputation_UNFOLD_N) {
        ClipV2DstMainKernelBwdF32(alpha, beta, src + i, dd + i, ds + i, loopClipV2Dstf);
    }
    for (; i < len; i += loopClipV2Dstf) {
        svbool_t pg = svwhilelt_b32_s64(i, len);
        svfloat32_t srcX = svld1_f32(pg, src + i);
        svfloat32_t res = svld1_f32(pg, dd + i);
        res = svdup_f32_m(res, svcmpge_n_f32(pg, srcX, beta), FLOAT_ZERO);
        res = svdup_f32_m(res, svcmple_n_f32(pg, srcX, alpha), FLOAT_ZERO);
        svst1_f32(pg, ds + i, res);
    }
    return;
}