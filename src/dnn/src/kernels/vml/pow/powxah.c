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

#include "vml_sve_powx_calculations.h"

static inline void VhPowxaF16FwdNYP0(const int64_t len, const float alpha, __fp16 *dst)
{
    int64_t i;
    int loopPowxah = svcnth();
    for (i = 0; i < len; i += loopPowxah) {
        svbool_t pg = svwhilelt_b16_s64(i, len);
        svst1_f16(pg, dst + i, svdup_n_f16((__fp16)alpha));
    }
}

static void VhPowxaF16FwdNYPN(const int64_t len, const float alpha, const float beta, const __fp16 *src, __fp16 *dst)
{
    if (UNLIKELY(beta == 0x1.2p3f)) {
        RUN_POWXA_F16_FWD_N_PARA_MODE(len, alpha, src, dst, VCPowxaF32X9);
    }
    if (UNLIKELY(beta == 0x1p3f)) {
        RUN_POWXA_F16_FWD_N_PARA_MODE(len, alpha, src, dst, VCPowxaF32X8);
    }
    if (UNLIKELY(beta == 0x1.cp2f)) {
        RUN_POWXA_F16_FWD_N_PARA_MODE(len, alpha, src, dst, VCPowxaF32X7);
    }
    if (UNLIKELY(beta == 0x1.8p2f)) {
        RUN_POWXA_F16_FWD_N_PARA_MODE(len, alpha, src, dst, VCPowxaF32X6);
    }
    if (UNLIKELY(beta == 0x1.4p2f)) {
        RUN_POWXA_F16_FWD_N_PARA_MODE(len, alpha, src, dst, VCPowxaF32X5);
    }
    if (UNLIKELY(beta == 0x1p2f)) {
        RUN_POWXA_F16_FWD_N_PARA_MODE(len, alpha, src, dst, VCPowxaF32X4);
    }
    if (UNLIKELY(beta == 0x1.8p1f)) {
        RUN_POWXA_F16_FWD_N_PARA_MODE(len, alpha, src, dst, VCPowxaF32X3);
    }
    if (UNLIKELY(beta == 0x1p1f)) {
        RUN_POWXA_F16_FWD_N_PARA_MODE(len, alpha, src, dst, VCPowxaF32X2);
    }
    if (UNLIKELY(beta == 0x1p0f)) {
        RUN_POWXA_F16_FWD_N_PARA_MODE(len, alpha, src, dst, svmul_n_f32_x);
    }
    if (UNLIKELY(beta == 0x0p0f)) {
        VhPowxaF16FwdNYP0(len, alpha, dst);
    }
}

void PowxaLaVCFWDF16(const int64_t len, const float alpha, const float beta, const __fp16 *src, __fp16 *dst)
{
    if (UNLIKELY(beta == 0x1p-1f)) {
        RUN_POWXA_F16_FWD_N_PARA_MODE(len, alpha, src, dst, VsPowxSqrtVC);
        return;
    }
    if (UNLIKELY(Frinta(beta) == beta && (beta >= MAX_POWXF_SP_TH_LO) && (beta < MAX_POWXF_SP_TH_HI))) {
        VhPowxaF16FwdNYPN(len, alpha, beta, src, dst);
        return;
    }
    int64_t i;
    int loopPowxah = svcnth();
    for (i = 0; i < len; i += loopPowxah) {
        svbool_t pgB16 = svwhilelt_b16_s64(i, len);
        svbool_t pgB32 = svwhilelt_b32_s64(i, len);

        svfloat16_t x = svld1_f16(pgB16, src + i);
        svfloat32_t xEven = svcvt_f32_f16_x(pgB16, x);
        svfloat32_t xOdd = CvtOddF32F16(x, pgB16, pgB32);

        svfloat32_t resEven = SvPowxaF32X(xEven, beta, alpha, pgB32);
        svfloat32_t resOdd = SvPowxaF32X(xOdd, beta, alpha, pgB32);

        svfloat16_t yEven = svcvt_f16_f32_x(pgB32, resEven);
        svfloat16_t yOdd = CvtOddF16F32(resOdd, pgB32);
        svfloat16_t y = svadd_f16_x(pgB16, yEven, yOdd);
        svst1_f16(pgB16, dst + i, y);
    }
}

static inline void VhPowxaLaBwdZero(const int64_t len, __fp16 *ds)
{
    int64_t i;
    int loopPowxah = svcnth();
    for (i = 0; i < len; i += loopPowxah) {
        svbool_t pg = svwhilelt_b16_s64(i, len);
        svst1_f16(pg, ds + i, VML_VC_F16_ZERO);
    }
}

static void VhPowxaF16BwdNYP1(const int64_t len, const float alpha, const __fp16 *dd, __fp16 *ds)
{
    int64_t i;
    int loopPowxah = svcnth();
    for (i = 0; i < len; i += loopPowxah) {
        svbool_t pgB16 = svwhilelt_b16_s64(i, len);
        svbool_t pgB32 = svwhilelt_b32_s64(i, len);

        svfloat16_t x = svld1_f16(pgB16, dd + i);
        svfloat32_t xEven = svcvt_f32_f16_x(pgB16, x);
        svfloat32_t xOdd = CvtOddF32F16(x, pgB16, pgB32);

        svfloat32_t resEven = svmul_n_f32_x(pgB32, xEven, alpha);
        svfloat32_t resOdd = svmul_n_f32_x(pgB32, xOdd, alpha);

        svfloat16_t yEven = svcvt_f16_f32_x(pgB32, resEven);
        svfloat16_t yOdd = CvtOddF16F32(resOdd, pgB32);
        svfloat16_t y = svadd_f16_x(pgB16, yEven, yOdd);
        svst1_f16(pgB16, ds + i, y);
    }
}

static void VhPowxaF16BwdNYPN(const int64_t len, const float alpha, const float beta,
    const __fp16 *src, const __fp16 *dd, __fp16 *ds)
{
    if (UNLIKELY(beta == 0x1.2p3f)) {
        RUN_POWXA_F16_BWD_N_PARA_MODE(len, alpha, src, dd, ds, VCPowxaF32X8);
    }
    if (UNLIKELY(beta == 0x1p3f)) {
        RUN_POWXA_F16_BWD_N_PARA_MODE(len, alpha, src, dd, ds, VCPowxaF32X7);
    }
    if (UNLIKELY(beta == 0x1.cp2f)) {
        RUN_POWXA_F16_BWD_N_PARA_MODE(len, alpha, src, dd, ds, VCPowxaF32X6);
    }
    if (UNLIKELY(beta == 0x1.8p2f)) {
        RUN_POWXA_F16_BWD_N_PARA_MODE(len, alpha, src, dd, ds, VCPowxaF32X5);
    }
    if (UNLIKELY(beta == 0x1.4p2f)) {
        RUN_POWXA_F16_BWD_N_PARA_MODE(len, alpha, src, dd, ds, VCPowxaF32X4);
    }
    if (UNLIKELY(beta == 0x1p2f)) {
        RUN_POWXA_F16_BWD_N_PARA_MODE(len, alpha, src, dd, ds, VCPowxaF32X3);
    }
    if (UNLIKELY(beta == 0x1.8p1f)) {
        RUN_POWXA_F16_BWD_N_PARA_MODE(len, alpha, src, dd, ds, VCPowxaF32X2);
    }
    if (UNLIKELY(beta == 0x1p1f)) {
        RUN_POWXA_F16_BWD_N_PARA_MODE(len, alpha, src, dd, ds, svmul_n_f32_x);
    }
    if (UNLIKELY(beta == 0x1p0f)) {
        VhPowxaF16BwdNYP1(len, alpha, dd, ds);
    }
}

void PowxaLaVCBWDF16(const int64_t len, const float alpha, const float beta,
    const __fp16 *src, const __fp16 *dd, __fp16 *ds)
{
    if (UNLIKELY(beta == 0.0f)) {
        VhPowxaLaBwdZero(len, ds);
        return;
    }
    if (UNLIKELY(beta == 0x1.8p0f)) {
        RUN_POWXA_F16_BWD_N_PARA_MODE(len, (alpha * beta), src, dd, ds, VsPowxSqrtVC);
        return;
    }
    // pow(x, y) y = 1 2 3 4 5 6 7 8 9
    if (UNLIKELY(Frinta(beta) == beta && (beta > MAX_POWXF_SP_TH_LO) && (beta < MAX_POWXF_SP_TH_HI))) {
        VhPowxaF16BwdNYPN(len, (alpha * beta), beta, src, dd, ds);
        return;
    }
    int64_t i;
    int loopPowxah = svcnth();
    for (i = 0; i < len; i += loopPowxah) {
        svbool_t pgB16 = svwhilelt_b16_s64(i, len);
        svbool_t pgB32 = svwhilelt_b32_s64(i, len);
        svfloat16_t srcX = svld1_f16(pgB16, src + i);
        svfloat16_t ddX = svld1_f16(pgB16, dd + i);

        svfloat32_t srcXfEven = svcvt_f32_f16_x(pgB16, srcX);
        svfloat32_t srcXfOdd = CvtOddF32F16(srcX, pgB16, pgB32);

        svfloat32_t srcXEvenRes = SvPowxaF32X(srcXfEven, (beta - 0x1p0f), (alpha * beta), pgB32);
        svfloat32_t srcXOddRes = SvPowxaF32X(srcXfOdd, (beta - 0x1p0f), (alpha * beta), pgB32);

        svfloat32_t ddXfEven = svcvt_f32_f16_x(pgB16, ddX);
        svfloat32_t ddXfOdd = CvtOddF32F16(ddX, pgB16, pgB32);

        svfloat32_t ddXEvenRes = svmul_f32_x(pgB32, ddXfEven, srcXEvenRes);
        svfloat32_t ddXOddRes = svmul_f32_x(pgB32, ddXfOdd, srcXOddRes);

        svfloat16_t resEven = svcvt_f16_f32_x(pgB32, ddXEvenRes);
        svfloat16_t resOdd = CvtOddF16F32(ddXOddRes, pgB32);
        svfloat16_t res = svadd_f16_x(pgB16, resEven, resOdd);

        svst1_f16(pgB16, ds + i, res);
    }
}
