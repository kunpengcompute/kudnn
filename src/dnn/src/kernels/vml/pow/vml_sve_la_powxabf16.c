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

static void SvePowxaBF16FwdNYP0(const int64_t len, const float alpha, __bf16 *dst)
{
    int64_t i;
    int loopPowxah = svcnth();
    svfloat32_t va = svdup_n_f32(alpha);
    svfloat32_t newVa = SvF32CvtBf16InfSpecial(svptrue_b32(), va);
    for (i = 0; i < len; i += loopPowxah) {
        svbool_t pg = svwhilelt_b16_s64(i, len);
        svbfloat16_t res = SvCvtBF16F32FullX(newVa, newVa);
        svst1_bf16(pg, dst + i, res);
    }
}

static void SvePowxaBF16FwdNYPN(const int64_t len, const float alpha, const float beta, const __bf16 *src, __bf16 *dst)
{
    if (unlikely(beta == 0x1.2p3f)) {
        RUN_POWXA_BF16_FWD_N_PARA_MODE(len, alpha, src, dst, SvePowxaF32X9);
    }
    if (unlikely(beta == 0x1p3f)) {
        RUN_POWXA_BF16_FWD_N_PARA_MODE(len, alpha, src, dst, SvePowxaF32X8);
    }
    if (unlikely(beta == 0x1.cp2f)) {
        RUN_POWXA_BF16_FWD_N_PARA_MODE(len, alpha, src, dst, SvePowxaF32X7);
    }
    if (unlikely(beta == 0x1.8p2f)) {
        RUN_POWXA_BF16_FWD_N_PARA_MODE(len, alpha, src, dst, SvePowxaF32X6);
    }
    if (unlikely(beta == 0x1.4p2f)) {
        RUN_POWXA_BF16_FWD_N_PARA_MODE(len, alpha, src, dst, SvePowxaF32X5);
    }
    if (unlikely(beta == 0x1p2f)) {
        RUN_POWXA_BF16_FWD_N_PARA_MODE(len, alpha, src, dst, SvePowxaF32X4);
    }
    if (unlikely(beta == 0x1.8p1f)) {
        RUN_POWXA_BF16_FWD_N_PARA_MODE(len, alpha, src, dst, SvePowxaF32X3);
    }
    if (unlikely(beta == 0x1p1f)) {
        RUN_POWXA_BF16_FWD_N_PARA_MODE(len, alpha, src, dst, SvePowxaF32X2);
    }
    if (unlikely(beta == 0x1p0f)) {
        RUN_POWXA_BF16_FWD_N_PARA_MODE(len, alpha, src, dst, svmul_n_f32_x);
    }
    if (unlikely(beta == 0x0p0f)) {
        SvePowxaBF16FwdNYP0(len, alpha, dst);
    }
}

void PowxaLaSveFWDBF16(const int64_t len, const float alpha, const float beta, const __bf16 *src, __bf16 *dst)
{
    if (unlikely(beta == 0x1p-1f)) {
        RUN_POWXA_BF16_FWD_N_PARA_MODE(len, alpha, src, dst, VsPowxSqrtSve);
        return;
    }
    if (unlikely(Frinta(beta) == beta && (beta >= MAX_POWXF_SP_TH_LO) && (beta < MAX_POWXF_SP_TH_HI))) {
        SvePowxaBF16FwdNYPN(len, alpha, beta, src, dst);
        return;
    }
    int64_t i;
    int loopPowxah = svcnth();
    for (i = 0; i < len; i += loopPowxah) {
        svbool_t pgB16 = svwhilelt_b16_s64(i, len);
        svbool_t pgB32 = svwhilelt_b32_s64(i, len);

        svbfloat16_t x = svld1_bf16(pgB16, src + i);
        svfloat32_t xBot = SvCvtF32BF16BottomX(x);
        svfloat32_t xTop = SvCvtF32BF16TopX(x);

        svfloat32_t resBot = SvPowxaF32X(xBot, beta, alpha, pgB32);
        svfloat32_t resTop = SvPowxaF32X(xTop, beta, alpha, pgB32);

        resBot = SvF32CvtBf16InfSpecial(pgB32, resBot);
        resTop = SvF32CvtBf16InfSpecial(pgB32, resTop);

        svbfloat16_t res = SvCvtBF16F32FullX(resBot, resTop);
        svst1_bf16(pgB16, dst + i, res);
    }
}

static inline void SvePowxaLaBF16BwdZero(const int64_t len, __bf16 *ds)
{
    int64_t i;
    int loopPowxah = svcnth();
    for (i = 0; i < len; i += loopPowxah) {
        svbool_t pg = svwhilelt_b16_s64(i, len);
        svbfloat16_t res = svreinterpret_bf16_u16(VML_VectorComputation_U16_ZERO);
        svst1_bf16(pg, ds + i, res);
    }
}

static void SvePowxaBF16BwdNYP1(const int64_t len, const float alpha, const __bf16 *dd, __bf16 *ds)
{
    int64_t i;
    int loopPowxah = svcnth();
    for (i = 0; i < len; i += loopPowxah) {
        svbool_t pgB16 = svwhilelt_b16_s64(i, len);
        svbool_t pgB32 = svwhilelt_b32_s64(i, len);
        svbfloat16_t ddX = svld1_bf16(pgB16, dd + i);
        svfloat32_t ddBot = SvCvtF32BF16BottomX(ddX);
        svfloat32_t ddTop = SvCvtF32BF16TopX(ddX);

        svfloat32_t resBot = svmul_n_f32_x(pgB32, ddBot, alpha);
        svfloat32_t resTop = svmul_n_f32_x(pgB32, ddTop, alpha);

        resBot = SvF32CvtBf16InfSpecial(pgB32, resBot);
        resTop = SvF32CvtBf16InfSpecial(pgB32, resTop);

        svbfloat16_t res = SvCvtBF16F32FullX(resBot, resTop);
        svst1_bf16(pgB16, ds + i, res);
    }
}

static void SvePowxaBF16BwdNYPN(const int64_t len, const float alpha, const float beta,
    const __bf16 *src, const __bf16 *dd, __bf16 *ds)
{
    if (unlikely(beta == 0x1.2p3f)) {
        RUN_POWXA_BF16_BWD_N_PARA_MODE(len, alpha, src, dd, ds, SvePowxaF32X8);
    }
    if (unlikely(beta == 0x1p3f)) {
        RUN_POWXA_BF16_BWD_N_PARA_MODE(len, alpha, src, dd, ds, SvePowxaF32X7);
    }
    if (unlikely(beta == 0x1.cp2f)) {
        RUN_POWXA_BF16_BWD_N_PARA_MODE(len, alpha, src, dd, ds, SvePowxaF32X6);
    }
    if (unlikely(beta == 0x1.8p2f)) {
        RUN_POWXA_BF16_BWD_N_PARA_MODE(len, alpha, src, dd, ds, SvePowxaF32X5);
    }
    if (unlikely(beta == 0x1.4p2f)) {
        RUN_POWXA_BF16_BWD_N_PARA_MODE(len, alpha, src, dd, ds, SvePowxaF32X4);
    }
    if (unlikely(beta == 0x1p2f)) {
        RUN_POWXA_BF16_BWD_N_PARA_MODE(len, alpha, src, dd, ds, SvePowxaF32X3);
    }
    if (unlikely(beta == 0x1.8p1f)) {
        RUN_POWXA_BF16_BWD_N_PARA_MODE(len, alpha, src, dd, ds, SvePowxaF32X2);
    }
    if (unlikely(beta == 0x1p1f)) {
        RUN_POWXA_BF16_BWD_N_PARA_MODE(len, alpha, src, dd, ds, svmul_n_f32_x);
    }
    if (unlikely(beta == 0x1p0f)) {
        SvePowxaBF16BwdNYP1(len, alpha, dd, ds);
    }
}

void PowxaLaSveBWDBF16(const int64_t len, const float alpha, const float beta,
    const __bf16 *src, const __bf16 *dd, __bf16 *ds)
{
    if (unlikely(beta == 0.0f)) {
        SvePowxaLaBF16BwdZero(len, ds);
        return;
    }
    if (unlikely(beta == 0x1.8p0f)) {
        RUN_POWXA_BF16_BWD_N_PARA_MODE(len, (alpha * beta), src, dd, ds, VsPowxSqrtSve);
        return;
    }
    // pow(x, y) y = 1 2 3 4 5 6 7 8 9
    if (unlikely(Frinta(beta) == beta && (beta > MAX_POWXF_SP_TH_LO) && (beta < MAX_POWXF_SP_TH_HI))) {
        SvePowxaBF16BwdNYPN(len, (alpha * beta), beta, src, dd, ds);
        return;
    }
    int64_t i;
    int loopPowxah = svcnth();
    for (i = 0; i < len; i += loopPowxah) {
        svbool_t pgB16 = svwhilelt_b16_s64(i, len);
        svbool_t pgB32 = svwhilelt_b32_s64(i, len);
        svbfloat16_t srcX = svld1_bf16(pgB16, src + i);
        svbfloat16_t ddX = svld1_bf16(pgB16, dd + i);

        svfloat32_t srcBot = SvCvtF32BF16BottomX(srcX);
        svfloat32_t srcTop = SvCvtF32BF16TopX(srcX);
        svfloat32_t resBot = SvPowxaF32X(srcBot, (beta - 0x1p0f), (alpha * beta), pgB32);
        svfloat32_t resTop = SvPowxaF32X(srcTop, (beta - 0x1p0f), (alpha * beta), pgB32);

        svfloat32_t ddBot = SvCvtF32BF16BottomX(ddX);
        svfloat32_t ddTop = SvCvtF32BF16TopX(ddX);
        resBot = svmul_f32_x(pgB32, resBot, ddBot);
        resTop = svmul_f32_x(pgB32, resTop, ddTop);

        resBot = SvF32CvtBf16InfSpecial(pgB32, resBot);
        resTop = SvF32CvtBf16InfSpecial(pgB32, resTop);

        svbfloat16_t res = SvCvtBF16F32FullX(resBot, resTop);
        svst1_bf16(pgB16, ds + i, res);
    }
}
