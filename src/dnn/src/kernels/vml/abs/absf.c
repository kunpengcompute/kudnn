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

#include "arm_neon.h"
#include "kernels/vml/vml_sve_base.h"
#define VML_VC_UNFOLD_N 4
#include "kernels/vml/common_macro.h"

void AbsLaFWDF32(const int64_t len, const float *src, float *dst)
{
    int64_t i = 0;
#ifndef KUDNNL_920G
    int loopCntAbsf = svcntw();
    for (; i <= len - loopCntAbsf * VML_VC_UNFOLD_N; i += loopCntAbsf * VML_VC_UNFOLD_N) {
        svbool_t pgB32 = svptrue_b32();
        VML_VC_DECLARE(svfloat32_t, x);
        VML_VC_DECLARE(svfloat32_t, res);
        VML_VC_LD_STEP(x, svld1_f32, pgB32, src + i, loopCntAbsf);
        VML_VC_FUNC_PG_1PARA(res, svabs_f32_x, pgB32, x);
        VML_VC_ST_STEP(svst1_f32, pgB32, dst + i, loopCntAbsf, res);
    }
    for (; i < len; i += loopCntAbsf) {
        svbool_t pg = svwhilelt_b32_s64(i, len);
        svfloat32_t x = svld1_f32(pg, src + i);
        svfloat32_t res = svabs_f32_x(pg, x);
        svst1_f32(pg, dst + i, res);
    }
#else
    #define LOOP_N 4
    for (; i <= len - 0x4 * LOOP_N; i += 0x4 * LOOP_N) {
        float32x4_t x0 = vld1q_f32(src + i + 0x00);
        float32x4_t x1 = vld1q_f32(src + i + 0x04);
        float32x4_t x2 = vld1q_f32(src + i + 0x08);
        float32x4_t x3 = vld1q_f32(src + i + 0x0C);

        float32x4_t res0 = vabsq_f32(x0);
        float32x4_t res1 = vabsq_f32(x1);
        float32x4_t res2 = vabsq_f32(x2);
        float32x4_t res3 = vabsq_f32(x3);

        vst1q_f32(dst + i + 0x00, res0);
        vst1q_f32(dst + i + 0x04, res1);
        vst1q_f32(dst + i + 0x08, res2);
        vst1q_f32(dst + i + 0x0C, res3);
    }
    for (; i < len; i++) {
        __asm__ volatile("fabs %s0, %s1 \n\t" : "=w"(dst[i]) : "w"(src[i]));
    }
#endif
}

void AbsLaVCBWDF32(const int64_t len, const float *src, const float *dd, float *ds)
{
    int64_t i = 0;
    int loopCnt = svcntw();
    svbool_t pg32 = svptrue_b32();
    for (; i <= len - loopCnt * VML_VC_UNFOLD_N; i += loopCnt * VML_VC_UNFOLD_N) {
        VML_VC_DECLARE(svuint32_t, uX);
        VML_VC_DECLARE(svuint32_t, udd);
        VML_VC_DECLARE(svuint32_t, xSign);
        VML_VC_DECLARE(svfloat32_t, x);
        VML_VC_DECLARE(svbool_t, cmpNeqZero);
        VML_VC_DECLARE(svuint32_t, uRes);

        const uint32_t *xPtr = (const uint32_t *)(src + i);
        const uint32_t *ddPtr = (const uint32_t *)(dd + i);
        uint32_t *dstPtr = (uint32_t *)(ds + i);
        VML_VC_LD_STEP(uX, svld1_u32, pg32, xPtr, loopCnt);
        VML_VC_LD_STEP(udd, svld1_u32, pg32, ddPtr, loopCnt);

        VML_VC_FUNC_PG_N2_2PARA(xSign, svand_n_u32_x, pg32, uX, FLOAT_SIGN_BIT);
        VML_VC_FUNC_NOPG_1PARA(x, svreinterpret_f32_u32, uX);
        VML_VC_FUNC_PG_N2_2PARA(cmpNeqZero, svcmpne_n_f32, pg32, x, FLOAT_ZERO);
        VML_VC_FUNC_NOPG_3PARA(uRes, sveor_u32_z, cmpNeqZero, xSign, udd);
        VML_VC_ST_STEP(svst1_u32, pg32, dstPtr, loopCnt, uRes);
    }
    for (; i < len; i += loopCnt) {
        svbool_t pg = svwhilelt_b32_s64(i, len);
        svuint32_t uX = svld1_u32(pg, (const uint32_t *)(src + i));
        svuint32_t udd = svld1_u32(pg, (const uint32_t *)(dd + i));

        svuint32_t xSign = svand_n_u32_x(pg, uX, FLOAT_SIGN_BIT);

        svbool_t cmpNeqZero = svcmpne_n_f32(pg, svreinterpret_f32_u32(uX), FLOAT_ZERO);
        svuint32_t uRes = sveor_u32_z(cmpNeqZero, xSign, udd);

        svst1_u32(pg, (uint32_t *)(ds + i), uRes);
    }
}
