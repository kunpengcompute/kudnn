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
#define VML_VectorComputation_UNFOLD_N 3
#include "kernels/vml/common_macro.h"

static void AbsMainKernelF32(const float *src, float *dst, int step)
{
    svuint32_t nanBit = svdup_n_u32(FLOAT_NAN_BIT);
    svbool_t pgB32 = svptrue_b32();
    VML_VectorComputation_DECLARE(svuint32_t, ux);
    VML_VectorComputation_DECLARE(svuint32_t, absux);
    VML_VectorComputation_DECLARE(svbool_t, cpg);
    const uint32_t *s = (uint32_t *)src;
    uint32_t *d = (uint32_t *)dst;
    VML_VectorComputation_LD_STEP(ux, svld1_u32, pgB32, s, step);
    VML_VectorComputation_FUNC_PG_N2_2PARA(absux, svand_n_u32_x, pgB32, ux, FLOAT_ABS_MASK);
    VML_VectorComputation_FUNC_PG_N2_2PARA(cpg, svcmpgt_n_u32, pgB32, absux, FLOAT_INF_BIT);
    VML_VectorComputation_FUNC_NOPG_N2_3PARA(absux, svsel_u32, cpg, nanBit, absux);
    VML_VectorComputation_ST_STEP(svst1_u32, pgB32, d, step, absux);
}

void AbsLaSveFWDF32(const int64_t len, const float *src, float *dst)
{
    int64_t i = 0;
    int loopCnt = svcntw();
    for (; i <= len - loopCnt * VML_VectorComputation_UNFOLD_N; i += loopCnt * VML_VectorComputation_UNFOLD_N) {
        AbsMainKernelF32(src + i, dst + i, loopCnt);
    }
    for (; i < len; i += loopCnt) {
        svbool_t pg = svwhilelt_b32_s64(i, len);
        svuint32_t ux = svld1_u32(pg, (uint32_t *)(src + i));
        svuint32_t absux = svand_n_u32_x(pg, ux, FLOAT_ABS_MASK);
        svbool_t cpg = svcmpgt_n_u32(pg, absux, FLOAT_INF_BIT);
        absux = svsel_u32(cpg, svdup_n_u32(FLOAT_NAN_BIT), absux);
        svst1_u32(pg, (uint32_t *)(dst + i), absux);
    }
}

void AbsLaSveBWDF32(const int64_t len, const float *src, const float *dd, float *ds)
{
    int64_t i;
    int loopCnt = svcntw();
    for (i = 0; i < len; i += loopCnt) {
        svbool_t pg = svwhilelt_b32_s64(i, len);
        svfloat32_t srcX = svld1_f32(pg, src + i);
        svfloat32_t ddX = svld1_f32(pg, dd + i);

        svbool_t cmpLtZero = svcmplt_n_f32(pg, srcX, 0.0f);
        svbool_t cmpEqZero = svcmpeq_n_f32(pg, srcX, 0.0f);

        svfloat32_t res = svneg_f32_m(ddX, cmpLtZero, ddX);
        res = svsel_f32(cmpEqZero, srcX, res);
        svst1_f32(pg, ds + i, res);
    }
}
