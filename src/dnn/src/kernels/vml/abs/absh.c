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

void AbsLaFWDF16(const int64_t len, const __fp16 *src, __fp16 *dst)
{
    int64_t i = 0;
#ifndef KUDNNL_920G
    int loopAbsh = svcnth();
    for (; i <= len - loopAbsh * VML_VC_UNFOLD_N; i += loopAbsh * VML_VC_UNFOLD_N) {
        svbool_t pgB16 = svptrue_b16();
        VML_VC_DECLARE(svfloat16_t, x);
        VML_VC_DECLARE(svfloat16_t, res);
        VML_VC_LD_STEP(x, svld1_f16, pgB16, src + i, loopAbsh);
        VML_VC_FUNC_PG_1PARA(res, svabs_f16_x, pgB16, x);
        VML_VC_ST_STEP(svst1_f16, pgB16, dst + i, loopAbsh, res);
    }
    for (; i < len; i += loopAbsh) {
        svbool_t pg = svwhilelt_b16_s64(i, len);
        svfloat16_t x = svld1_f16(pg, src + i);
        svfloat16_t res = svabs_f16_x(pg, x);
        svst1_f16(pg, dst + i, res);
    }
#else
    #define LOOP_N 4
    for (; i <= len - 0x8 * LOOP_N; i += 0x8 * LOOP_N) {
        float16x8_t x0 = vld1q_f16(src + i + 0x00);
        float16x8_t x1 = vld1q_f16(src + i + 0x08);
        float16x8_t x2 = vld1q_f16(src + i + 0x10);
        float16x8_t x3 = vld1q_f16(src + i + 0x18);

        float16x8_t res0 = vabsq_f16(x0);
        float16x8_t res1 = vabsq_f16(x1);
        float16x8_t res2 = vabsq_f16(x2);
        float16x8_t res3 = vabsq_f16(x3);

        vst1q_f16(dst + i + 0x00, res0);
        vst1q_f16(dst + i + 0x08, res1);
        vst1q_f16(dst + i + 0x10, res2);
        vst1q_f16(dst + i + 0x18, res3);
    }
    for (; i < len; i++) {
        __fp16 x = src[i];
        __fp16 y;
        __asm__ volatile("fabs %h0, %h1 \n\t" : "=w"(y) : "w"(x));
        dst[i] = y;
    }
#endif
}

void AbsLaBWDF16(const int64_t len, const __fp16 *src, const __fp16 *dd, __fp16 *ds)
{
    int64_t i = 0;
    int loopAbsh = svcnth();
    svbool_t pg16 = svptrue_b16();
    for (; i <= len - loopAbsh * VML_VC_UNFOLD_N; i += loopAbsh * VML_VC_UNFOLD_N) {
        VML_VC_DECLARE(svuint16_t, uX);
        VML_VC_DECLARE(svuint16_t, udd);
        VML_VC_DECLARE(svuint16_t, xSign);
        VML_VC_DECLARE(svfloat16_t, x);
        VML_VC_DECLARE(svbool_t, cmpNeqZero);
        VML_VC_DECLARE(svuint16_t, uRes);

        const uint16_t *xPtr = (const uint16_t *)(src + i);
        const uint16_t *ddPtr = (const uint16_t *)(dd + i);
        uint16_t *dstPtr = (uint16_t *)(ds + i);
        VML_VC_LD_STEP(uX, svld1_u16, pg16, xPtr, loopAbsh);
        VML_VC_LD_STEP(udd, svld1_u16, pg16, ddPtr, loopAbsh);

        VML_VC_FUNC_PG_N2_2PARA(xSign, svand_n_u16_x, pg16, uX, F16_SIGN_MASK);
        VML_VC_FUNC_NOPG_1PARA(x, svreinterpret_f16_u16, uX);
        VML_VC_FUNC_PG_N2_2PARA(cmpNeqZero, svcmpne_n_f16, pg16, x, FLOAT_ZERO);
        VML_VC_FUNC_NOPG_3PARA(uRes, sveor_u16_z, cmpNeqZero, xSign, udd);
        VML_VC_ST_STEP(svst1_u16, pg16, dstPtr, loopAbsh, uRes);
    }
    for (; i < len; i += loopAbsh) {
        svbool_t pg = svwhilelt_b16_s64(i, len);
        svuint16_t uX = svld1_u16(pg, (const uint16_t *)(src + i));
        svuint16_t udd = svld1_u16(pg, (const uint16_t *)(dd + i));

        svuint16_t xSign = svand_n_u16_x(pg, uX, F16_SIGN_MASK);

        svbool_t cmpNeqZero = svcmpne_n_f16(pg, svreinterpret_f16_u16(uX), FLOAT_ZERO);
        svuint16_t uRes = sveor_u16_z(cmpNeqZero, xSign, udd);

        svst1_u16(pg, (uint16_t *)(ds + i), uRes);
    }
}
