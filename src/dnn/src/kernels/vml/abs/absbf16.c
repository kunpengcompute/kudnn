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

void AbsLaFWDBF16(const int64_t len, const __bf16 *src, __bf16 *dst)
{
    int64_t i = 0;
#ifndef KUDNNL_920G
    int loopAbsh = svcnth();
    for (; i <= len - loopAbsh * VML_VC_UNFOLD_N; i += loopAbsh * VML_VC_UNFOLD_N) {
        svbool_t pgB16 = svptrue_b16();
        VML_VC_DECLARE(svuint16_t, x);
        VML_VC_DECLARE(svuint16_t, res);

        const uint16_t *xPtr = (const uint16_t *)(src + i);
        uint16_t *dstPtr = (uint16_t *)(dst + i);
        VML_VC_LD_STEP(x, svld1_u16, pgB16, xPtr, loopAbsh);
        VML_VC_FUNC_PG_N2_2PARA(res, svand_u16_x, pgB16, x, VML_VC_U16_ABS);
        VML_VC_ST_STEP(svst1_u16, pgB16, dstPtr, loopAbsh, res);
    }
    for (; i < len; i += loopAbsh) {
        svbool_t pg = svwhilelt_b16_s64(i, len);
        svuint16_t ux = svld1_u16(pg, (const uint16_t *)(src + i));
        svuint16_t res = svand_u16_x(pg, ux, VML_VC_U16_ABS);
        svst1_u16(pg, (uint16_t *)(dst + i), res);
    }
#else
    #define LOOP_N 4
    for (; i <= len - 0x8 * LOOP_N; i += 0x8 * LOOP_N) {
        uint16x8_t uX0 = vld1q_u16((const uint16_t *)(src + i + 0x00));
        uint16x8_t uX1 = vld1q_u16((const uint16_t *)(src + i + 0x08));
        uint16x8_t uX2 = vld1q_u16((const uint16_t *)(src + i + 0x10));
        uint16x8_t uX3 = vld1q_u16((const uint16_t *)(src + i + 0x18));

        uint16x8_t uDst0 = vandq_u16(uX0, vdupq_n_u16(BF16_ABS_MASK));
        uint16x8_t uDst1 = vandq_u16(uX1, vdupq_n_u16(BF16_ABS_MASK));
        uint16x8_t uDst2 = vandq_u16(uX2, vdupq_n_u16(BF16_ABS_MASK));
        uint16x8_t uDst3 = vandq_u16(uX3, vdupq_n_u16(BF16_ABS_MASK));

        vst1q_u16((uint16_t *)(dst + i + 0x00), uDst0);
        vst1q_u16((uint16_t *)(dst + i + 0x08), uDst1);
        vst1q_u16((uint16_t *)(dst + i + 0x10), uDst2);
        vst1q_u16((uint16_t *)(dst + i + 0x18), uDst3);
    }
    for (; i < len; i++) {
        uint16_t uX = *(const uint16_t *)(src + i);
        *(uint16_t *)(dst + i) = uX & BF16_ABS_MASK;
    }
#endif
}

void AbsLaVCBWDBF16(const int64_t len, const __bf16 *src, const __bf16 *dd, __bf16 *ds)
{
    int64_t i = 0;
    int loopAbsh = svcnth();
    svbool_t pg16 = svptrue_b16();
    for (; i <= len - loopAbsh * VML_VC_UNFOLD_N; i += loopAbsh * VML_VC_UNFOLD_N) {
        VML_VC_DECLARE(svuint16_t, uX);
        VML_VC_DECLARE(svuint16_t, udd);
        VML_VC_DECLARE(svuint16_t, xSign);
        VML_VC_DECLARE(svbool_t, cmpNeqZero);
        VML_VC_DECLARE(svuint16_t, uRes);

        const uint16_t *xPtr = (const uint16_t *)(src + i);
        const uint16_t *ddPtr = (const uint16_t *)(dd + i);
        uint16_t *dstPtr = (uint16_t *)(ds + i);
        VML_VC_LD_STEP(uX, svld1_u16, pg16, xPtr, loopAbsh);
        VML_VC_LD_STEP(udd, svld1_u16, pg16, ddPtr, loopAbsh);

        VML_VC_FUNC_PG_N2_2PARA(xSign, svand_n_u16_x, pg16, uX, BF16_SIGN_MASK);
        VML_VC_FUNC_PG_N2_2PARA(cmpNeqZero, svcmpne_n_u16, pg16, uX, BF16_ZERO_BIT);
        VML_VC_FUNC_NOPG_3PARA(uRes, sveor_u16_z, cmpNeqZero, xSign, udd);
        VML_VC_ST_STEP(svst1_u16, pg16, dstPtr, loopAbsh, uRes);
    }
    for (; i < len; i += loopAbsh) {
        svbool_t pg = svwhilelt_b16_s64(i, len);
        svuint16_t uX = svld1_u16(pg, (const uint16_t *)(src + i));
        svuint16_t udd = svld1_u16(pg, (const uint16_t *)(dd + i));

        svuint16_t xSign = svand_n_u16_x(pg, uX, BF16_SIGN_MASK);

        svbool_t cmpNeqZero = svcmpne_n_u16(pg, uX, BF16_ZERO_BIT);
        svuint16_t uRes = sveor_u16_z(cmpNeqZero, xSign, udd);

        svst1_u16(pg, (uint16_t *)(ds + i), uRes);
    }
}
