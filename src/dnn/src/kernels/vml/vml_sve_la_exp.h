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

#ifndef VML_VectorComputation_LA_EXP_H
#define VML_VectorComputation_LA_EXP_H

#include "vml_sve_base.h"

struct SveExpfData {
    float invLn2Hi;
    float invLn2Lo;
    float splThR;
    float upBound;
    float lowBound;
};

extern const struct SveExpfData G_VectorComputation_VSEXP_DATA;
#define VectorComputation_EXPF_INVLN2HI svdup_f32(G_VectorComputation_VSEXP_DATA.invLn2Hi)
#define VectorComputation_EXPF_INVLN2LO svdup_f32(G_VectorComputation_VSEXP_DATA.invLn2Lo)
#define VectorComputation_EXPF_TH_R svdup_f32(G_VectorComputation_VSEXP_DATA.splThR)
#define VectorComputation_EXPF_UPBOUND svdup_f32(G_VectorComputation_VSEXP_DATA.upBound)
#define VectorComputation_EXPF_LOWBOUND svdup_f32(G_VectorComputation_VSEXP_DATA.lowBound)

#define VectorComputation_EXPF_LN2HI  svdup_f32(0x1.62e43p-1f)
#define VectorComputation_EXPF_C0     svdup_f32(0x1p0f)
#define VectorComputation_EXPF_C1     svdup_f32(0x1.fffff6p-1f)
#define VectorComputation_EXPF_C2     svdup_f32(0x1.fffdc6p-2f)
#define VectorComputation_EXPF_C3     svdup_f32(0x1.555a8p-3f)
#define VectorComputation_EXPF_C4     svdup_f32(0x1.573a1ap-5f)
#define VectorComputation_EXPF_C5     svdup_f32(0x1.0f9f9cp-7f)

#define VectorComputation_EXP2F_POLY_ORDER 3U

struct SveExp2fData {
    float exp2fShift;
    float splTh;
    float poly[VectorComputation_EXP2F_POLY_ORDER];
};

extern const struct SveExp2fData G_VectorComputation_VSEXP2_DATA;
#define VectorComputation_EXP2F_C0 svdup_f32(G_VectorComputation_VSEXP2_DATA.poly[0])
#define VectorComputation_EXP2F_C1 svdup_f32(G_VectorComputation_VSEXP2_DATA.poly[1])
#define VectorComputation_EXP2F_C2 svdup_f32(G_VectorComputation_VSEXP2_DATA.poly[2])
#define VectorComputation_EXP2F_SHIFT svdup_f32(G_VectorComputation_VSEXP2_DATA.exp2fShift)
#define VectorComputation_EXP2F_TH svdup_f32(G_VectorComputation_VSEXP2_DATA.splTh)

/* exp(x) = 2^n * 2^r
 * x = ln2*(n+r), with r in [-0x1p-6, 0x1p-6].
 */
__attribute__((always_inline)) inline svfloat32_t SvExpF32X(svfloat32_t x, svbool_t pg)
{
    svbool_t pgRetZero = svcmple(pg, x, VectorComputation_EXPF_LOWBOUND);
    svbool_t pgRetInf = svcmpge(pg, x, VectorComputation_EXPF_UPBOUND);
    svbool_t pgSpR = svcmpge(pg, x, VectorComputation_EXPF_TH_R);
    svfloat32_t x1 = svsub_f32_m(pgSpR, x, VectorComputation_FLOAT_TWO);

    svfloat32_t ki = svmad_f32_x(pg, x1, VectorComputation_EXPF_INVLN2HI, VectorComputation_EXP2F_SHIFT);
    svfloat32_t kd = svsub_f32_x(pg, ki, VectorComputation_EXP2F_SHIFT);
    svuint32_t k = svreinterpret_u32_f32(ki);
    svfloat32_t expk = svexpa_f32(k);
    svfloat32_t r = svnmls_f32_x(pg, kd, x1, VectorComputation_EXPF_INVLN2HI);
    r = svmla_f32_x(pg, r, x1, VectorComputation_EXPF_INVLN2LO);

    // polynomial of exp2
    svfloat32_t poly = svmla_f32_x(pg, VectorComputation_EXP2F_C1, r, VectorComputation_EXP2F_C0);
    poly = svmla_f32_x(pg, VectorComputation_EXP2F_C2, poly, r);
    svfloat32_t res = svmul_f32_x(pg, expk, poly);
    res = svmul_n_f32_m(pgSpR, res, 0x1.d8e64cp2f); /* e^2 = 0x1.d8e64cp2f */
    res = svdup_n_f32_m(res, pgRetZero, FLOAT_ZERO);
    res = svdup_n_f32_m(res, pgRetInf, INFINITY);
    return res;
}

// same with SvExpF32X but x < 0
__attribute__((always_inline)) inline svfloat32_t SvExpForNegF32X(svfloat32_t x, svbool_t pg)
{
    svbool_t pgRetZero = svcmple(pg, x, VectorComputation_EXPF_LOWBOUND);

    svfloat32_t ki = svmad_f32_x(pg, x, VectorComputation_EXPF_INVLN2HI, VectorComputation_EXP2F_SHIFT);
    svfloat32_t kd = svsub_f32_x(pg, ki, VectorComputation_EXP2F_SHIFT);
    svuint32_t k = svreinterpret_u32_f32(ki);
    svfloat32_t expk = svexpa_f32(k);
    svfloat32_t r = svnmls_f32_x(pg, kd, x, VectorComputation_EXPF_INVLN2HI);
    r = svmla_f32_x(pg, r, x, VectorComputation_EXPF_INVLN2LO);

    // polynomial of exp2
    svfloat32_t poly = svmla_f32_x(pg, VectorComputation_EXP2F_C1, r, VectorComputation_EXP2F_C0);
    poly = svmla_f32_x(pg, VectorComputation_EXP2F_C2, poly, r);
    svfloat32_t res = svmul_f32_x(pg, expk, poly);
    res = svsel_f32(pgRetZero, VectorComputation_FLOAT_ZERO, res);
    return res;
}

struct VectorComputation_EXPM1F_DATA {
    /* Grouped together so they can be loaded as one quadword so can use _lane forms of svmla/svmls */
    float c2, c4, ln2Hi, ln2Lo;
    float c0, c1, c3, invLn2, maxThr, shift;
};
extern __attribute__((aligned(0x20))) const struct VectorComputation_EXPM1F_DATA G_VectorComputation_EXPM1F_DATA;

inline svfloat32_t VsExpm1PolySve(svbool_t pg, svfloat32_t r,
    svfloat32_t laneVal, const struct VectorComputation_EXPM1F_DATA *d)
{
    svfloat32_t r2 = svmul_f32_x(pg, r, r);

    svfloat32_t fP0403 = svmla_lane_f32(svdup_n_f32(d->c1), r, laneVal, 0x0);
    svfloat32_t fP0605 = svmla_lane_f32(svdup_n_f32(d->c3), r, laneVal, 0x1);
    svfloat32_t fP0603 = svmla_f32_x(pg, fP0403, r2, fP0605);
    svfloat32_t fP0602 = svmla_f32_x(pg, svdup_n_f32(d->c0), r, fP0603);
    svfloat32_t p = svmla_f32_x(pg, r, r2, fP0602);

    return p;
}

inline svfloat32_t VsExpm1InlineSve(svbool_t pg, svfloat32_t x)
{
    const struct VectorComputation_EXPM1F_DATA *d = PTR_BARRIER(&G_VectorComputation_EXPM1F_DATA);
    svfloat32_t laneVal = svld1rq_f32(svptrue_b32(), &d->c2);
    svfloat32_t z = svmla_n_f32_x(pg, svdup_n_f32(d->shift), x, d->invLn2);
    svfloat32_t exp2n = svreinterpret_f32_u32(svlsl_n_u32_x(pg, svreinterpret_u32_f32(z),
        FLOAT_MANTISSA_BIT_LEN));
    svfloat32_t n = svsub_n_f32_x(pg, z, d->shift);
    svfloat32_t r = svmls_lane_f32(svmls_lane_f32(x, n, laneVal, 0x2), n, laneVal, 0x3);
    svfloat32_t p = VsExpm1PolySve(pg, r, laneVal, d);

    svfloat32_t t = svsub_f32_x(pg, exp2n, VectorComputation_FLOAT_ONE);
    return svmla_f32_x(pg, t, exp2n, p);
}

#endif
