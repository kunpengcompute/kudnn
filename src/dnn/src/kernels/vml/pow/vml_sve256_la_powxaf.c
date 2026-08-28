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

void PowxaLaVCFWDF32(const int64_t len, const float alpha, const float y, const float *src, float *dst)
{
    if (UNLIKELY(y == 0x1p-1f)) {
        VsPowxaLaNYPH(len, src, alpha, dst);
        return;
    }
    if (UNLIKELY(Frinta(y) == y && (y >= MAX_POWXF_SP_TH_LO) && (y < MAX_POWXF_SP_TH_HI))) {
        VsPowxaVCNYPN(len, src, y, alpha, dst);
        return;
    }
    int64_t i;
    int loopPowxaf = svcntw();
    for (i = 0; i < len; i += loopPowxaf) {
        svbool_t pg = svwhilelt_b32_s64(i, len);
        svfloat32_t x = svld1_f32(pg, src + i);
        svfloat32_t result = SvPowxaF32X(x, y, alpha, pg);
        svst1_f32(pg, dst + i, result);
    }
}

static inline void VsPowxaLaBwdZero(const int64_t len, float *ds)
{
    int64_t i;
    int loopPowxaf = svcntw();
    for (i = 0; i < len; i += loopPowxaf) {
        svbool_t pg = svwhilelt_b32_s64(i, len);
        svst1_f32(pg, ds + i, VC_FLOAT_ZERO);
    }
}

static inline void VsPowxaVCBwdNYP1(const int64_t len, const float alpha, const float *dd, float *ds)
{
    int64_t i;
    int loopPowxaf = svcntw();
    for (i = 0; i < len; i += loopPowxaf) {
        svbool_t pg = svwhilelt_b32_s64(i, len);
        svfloat32_t dX = svld1_f32(pg, dd + i);
        svfloat32_t res = svmul_n_f32_x(pg, dX, alpha);
        svst1_f32(pg, ds + i, res);
    }
}

static void VsPowxaVCBwdNYPN(const int64_t len, const float alpha, const float beta,
    const float *src, const float *dd, float *ds)
{
    if (UNLIKELY(beta == 0x1.2p3f)) {
        RUN_POWXA_F32_BWD_N_PARA_MODE(len, alpha, src, dd, ds, VCPowxaF32X8);
    }
    if (UNLIKELY(beta == 0x1p3f)) {
        RUN_POWXA_F32_BWD_N_PARA_MODE(len, alpha, src, dd, ds, VCPowxaF32X7);
    }
    if (UNLIKELY(beta == 0x1.cp2f)) {
        RUN_POWXA_F32_BWD_N_PARA_MODE(len, alpha, src, dd, ds, VCPowxaF32X6);
    }
    if (UNLIKELY(beta == 0x1.8p2f)) {
        RUN_POWXA_F32_BWD_N_PARA_MODE(len, alpha, src, dd, ds, VCPowxaF32X5);
    }
    if (UNLIKELY(beta == 0x1.4p2f)) {
        RUN_POWXA_F32_BWD_N_PARA_MODE(len, alpha, src, dd, ds, VCPowxaF32X4);
    }
    if (UNLIKELY(beta == 0x1p2f)) {
        RUN_POWXA_F32_BWD_N_PARA_MODE(len, alpha, src, dd, ds, VCPowxaF32X3);
    }
    if (UNLIKELY(beta == 0x1.8p1f)) {
        RUN_POWXA_F32_BWD_N_PARA_MODE(len, alpha, src, dd, ds, VCPowxaF32X2);
    }
    if (UNLIKELY(beta == 0x1p1f)) {
        RUN_POWXA_F32_BWD_N_PARA_MODE(len, alpha, src, dd, ds, svmul_n_f32_x);
    }
    if (UNLIKELY(beta == 0x1p0f)) {
        VsPowxaVCBwdNYP1(len, alpha, dd, ds);
    }
}

void PowxaLaVCBWDF32(const int64_t len, const float alpha, const float beta,
    const float *src, const float *dd, float *ds)
{
    if (UNLIKELY(beta == 0.0f)) {
        VsPowxaLaBwdZero(len, ds);
        return;
    }
    if (UNLIKELY(beta == 0x1.8p0f)) {
        RUN_POWXA_F32_BWD_N_PARA_MODE(len, (alpha * beta), src, dd, ds, VsPowxSqrtVC);
        return;
    }
    // pow(x, y) y = 1 2 3 4 5 6 7 8 9
    if (UNLIKELY(Frinta(beta) == beta && (beta > MAX_POWXF_SP_TH_LO) && (beta < MAX_POWXF_SP_TH_HI))) {
        VsPowxaVCBwdNYPN(len, (alpha * beta), beta, src, dd, ds);
        return;
    }
    int64_t i;
    int loopPowxaf = svcntw();
    for (i = 0; i < len; i += loopPowxaf) {
        svbool_t pg = svwhilelt_b32_s64(i, len);
        svfloat32_t sX = svld1_f32(pg, src + i);
        svfloat32_t dX = svld1_f32(pg, dd + i);
        svfloat32_t res = SvPowxaF32X(sX, (beta - 0x1p0f), (alpha * beta), pg);
        res = svmul_f32_x(pg, res, dX);
        svst1_f32(pg, ds + i, res);
    }
}
