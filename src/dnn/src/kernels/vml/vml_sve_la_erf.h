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

#ifndef VML_VC_LA_ERF_H
#define VML_VC_LA_ERF_H

#include "vml_sve_la_exp.h"

#define GELU_ERF_C0     svdup_f32(0x1.04f20cp-2f)
#define GELU_ERF_C1     svdup_f32(-0x1.23531cp-2f)
#define GELU_ERF_C2     svdup_f32(0x1.6be1c6p0f)
#define GELU_ERF_C3     svdup_f32(-0x1.7401c6p0f)
#define GELU_ERF_C4     svdup_f32(0x1.0fb844p0f)
#define GELU_ERF_APPROX_CONST   svdup_f32(0x1.4f740ap-2f)

// erf = sign * (1 - r * t * exp(-x*x))
__attribute__((always_inline)) inline svfloat32_t SvErfF32X(svfloat32_t x, svbool_t pg)
{
    // t = 1 / (p*|x| + 1)
    svfloat32_t absX = svabs_f32_x(pg, x);
    svfloat32_t temp = svmad_f32_x(pg, GELU_ERF_APPROX_CONST, absX, VC_FLOAT_ONE);
    svfloat32_t t = svdiv_f32_x(pg, VC_FLOAT_ONE, temp);

    // -exp(-x*x)
    svfloat32_t negX2 = svmls_f32_x(pg, VC_FLOAT_ZERO, x, x);
    svfloat32_t expNegX2 = SvExpF32X(negX2, pg);
    svfloat32_t negExpNegX2 = svsub_f32_x(pg, VC_FLOAT_ZERO, expNegX2);

    // r = poly(t)
    svfloat32_t r = svmad_f32_x(pg, GELU_ERF_C4, t, GELU_ERF_C3);
    r = svmad_f32_x(pg, r, t, GELU_ERF_C2);
    r = svmad_f32_x(pg, r, t, GELU_ERF_C1);
    r = svmad_f32_x(pg, r, t, GELU_ERF_C0);

    svbool_t pgNeg = svcmpge_u32(pg, svreinterpret_u32_f32(x), VC_FLOAT_SIGN_BIT);
    svfloat32_t res = svmul_f32_x(pg, t, negExpNegX2);
    res = svmad_f32_x(pg, res, r, VC_FLOAT_ONE);
    res = svmul_f32_m(pgNeg, res, VC_FLOAT_NEG_ONE);

    return res;
}

#define VC_ERF_TBL_SIZE 0x1000U
extern const float g_erfTbl[VC_ERF_TBL_SIZE];
extern const float g_erfExpTbl[VC_ERF_TBL_SIZE];
#define VC_VSERF_MAX       svdup_n_f32(0x1.f5a002p1f)
#define VC_VSERF_SHIFT     svdup_n_f32(0x1p+14f)
#define VC_VSERF_IDXMASK   svdup_n_u32(0xfffU)
#define VC_VSERF_TALOR_C1  svdup_n_f32(-0x1.555556p-2f)
#define VML_VSERF_ERFTABLE  g_erfTbl
#define VML_VSERF_EXPTABLE  g_erfExpTbl

__attribute__((always_inline)) inline svfloat32_t SvErfF32XNew(svfloat32_t x, svbool_t pg)
{
    svfloat32_t absX = SvAndF32X(pg, x, VC_FLOAT_ABS_MASK);
    svuint32_t sign = svand_u32_x(pg, svreinterpret_u32_f32(x), VC_FLOAT_SIGN_BIT);
    absX = svmin_f32_x(pg, absX, VC_VSERF_MAX);
    svfloat32_t rX = svadd_f32_x(pg, absX, VC_VSERF_SHIFT);
    svuint32_t idx = svand_u32_x(pg, svreinterpret_u32_f32(rX), VC_VSERF_IDXMASK);
    svfloat32_t erfX0 = svld1_gather_u32index_f32(pg, VML_VSERF_ERFTABLE, idx);
    svfloat32_t expNegx2 = svld1_gather_u32index_f32(pg, VML_VSERF_EXPTABLE, idx);
    svfloat32_t x0 = svsub_f32_x(pg, rX, VC_VSERF_SHIFT);
    svfloat32_t r = svsub_f32_x(pg, absX, x0);
    svfloat32_t taylor1 = svmul_f32_x(pg, r, expNegx2);
    svfloat32_t poly1 = svmla_f32_x(pg, svneg_f32_x(pg, x0), VC_VSERF_TALOR_C1, r);
    svfloat32_t poly2 = svmla_f32_x(pg, VC_FLOAT_ONE, poly1, r);
    svfloat32_t y = svmla_f32_x(pg, erfX0, poly2, taylor1);
    y = SvOrrF32X(pg, y, sign);
    return y;
}

#endif
