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

#ifndef VML_VC_LA_EXP_H
#define VML_VC_LA_EXP_H

#include "vml_sve_base.h"
#include "vml_sve_la_exp.h"

struct VCExpfData {
    float invLn2Hi;
    float invLn2Lo;
    float splThR;
    float upBound;
    float lowBound;
};

extern const struct VCExpfData G_VC_VSEXP_DATA;
#define VC_EXPF_INVLN2HI svdup_f32(G_VC_VSEXP_DATA.invLn2Hi)
#define VC_EXPF_INVLN2LO svdup_f32(G_VC_VSEXP_DATA.invLn2Lo)
#define VC_EXPF_TH_R svdup_f32(G_VC_VSEXP_DATA.splThR)
#define VC_EXPF_UPBOUND svdup_f32(G_VC_VSEXP_DATA.upBound)
#define VC_EXPF_LOWBOUND svdup_f32(G_VC_VSEXP_DATA.lowBound)

#define VC_EXPF_LN2HI  svdup_f32(0x1.62e43p-1f)
#define VC_EXPF_C0     svdup_f32(0x1p0f)
#define VC_EXPF_C1     svdup_f32(0x1.fffff6p-1f)
#define VC_EXPF_C2     svdup_f32(0x1.fffdc6p-2f)
#define VC_EXPF_C3     svdup_f32(0x1.555a8p-3f)
#define VC_EXPF_C4     svdup_f32(0x1.573a1ap-5f)
#define VC_EXPF_C5     svdup_f32(0x1.0f9f9cp-7f)

#define VC_EXP2F_POLY_ORDER 3U

struct VCExp2fData {
    float exp2fShift;
    float splTh;
    float poly[VC_EXP2F_POLY_ORDER];
};

extern const struct VCExp2fData G_VC_VSEXP2_DATA;
#define VC_EXP2F_C0 svdup_f32(G_VC_VSEXP2_DATA.poly[0])
#define VC_EXP2F_C1 svdup_f32(G_VC_VSEXP2_DATA.poly[1])
#define VC_EXP2F_C2 svdup_f32(G_VC_VSEXP2_DATA.poly[2])
#define VC_EXP2F_SHIFT svdup_f32(G_VC_VSEXP2_DATA.exp2fShift)
#define VC_EXP2F_TH svdup_f32(G_VC_VSEXP2_DATA.splTh)

/* exp (x)
 * = exp(n * ln(2) + r)
 * = 2^(n-1) * exp(r) * 2
 */
__attribute__((always_inline)) inline svfloat32_t SvExpF32X(svfloat32_t x, svbool_t pg)
{
    svbool_t pgRetZero = svcmple(pg, x, VC_EXPF_LOWBOUND);
    svfloat32_t aux0 = svmin_f32_x(pg, x, VC_EXPF_UPBOUND);
    aux0 = svmax_f32_x(pg, aux0, VC_EXPF_LOWBOUND);

    svfloat32_t x1 = svmul_f32_x(pg, aux0, VC_EXPF_INVLN2HI);
    x1 = svadd_f32_x(pg, x1, VC_FLOAT_HALF);
    x1 = svrintm_f32_x(pg, x1);

    aux0 = svmsb_f32_x(pg, x1, VC_EXPF_LN2HI, aux0);
    x1 = svsub_f32_x(pg, x1, VC_FLOAT_ONE);

    // compute 2^(n-1)
    svint32_t aux1 = svcvt_s32_x(pg, x1);
    aux1 = svadd_n_s32_x(pg, aux1, FLOAT_EXP_BIAS);
    aux1 = svlsl_n_s32_x(pg, aux1, FLOAT_MANTISSA_BIT_LEN);

    // exp(r)
    svfloat32_t poly = svmad_f32_x(pg, VC_EXPF_C5, aux0, VC_EXPF_C4);
    poly = svmad_f32_x(pg, poly, aux0, VC_EXPF_C3);
    poly = svmad_f32_x(pg, poly, aux0, VC_EXPF_C2);
    poly = svmad_f32_x(pg, poly, aux0, VC_EXPF_C1);
    poly = svmad_f32_x(pg, poly, aux0, VC_EXPF_C0);

    svfloat32_t res = svreinterpret_f32_s32(aux1);
    res = svmul_f32_x(pg, poly, res);
    res = svmul_f32_x(pg, res, VC_FLOAT_TWO);
    res = svsel_f32(pgRetZero, VC_FLOAT_ZERO, res);

    return res;
}

#define VML_VC_EXP(pg, x, res) \
    do { \
        VML_VC_DECLARE(svbool_t, pgRetZero); \
        VML_VC_FUNC_PG_N2_2PARA(pgRetZero, svcmple, pg, x, VC_EXPF_LOWBOUND); \
        VML_VC_DECLARE(svfloat32_t, aux0); \
        VML_VC_FUNC_PG_N2_2PARA(aux0, svmin_f32_x, pg, x, VC_EXPF_UPBOUND); \
        VML_VC_FUNC_PG_N2_2PARA(aux0, svmax_f32_x, pg, aux0, VC_EXPF_LOWBOUND); \
        VML_VC_DECLARE(svfloat32_t, x1); \
        VML_VC_FUNC_PG_N2_2PARA(x1, svmul_f32_x, pg, aux0, VC_EXPF_INVLN2HI); \
        VML_VC_FUNC_PG_N2_2PARA(x1, svadd_f32_x, pg, x1, VC_FLOAT_HALF); \
        VML_VC_FUNC_PG_1PARA(x1, svrintm_f32_x, pg, x1); \
        VML_VC_FUNC_PG_N2_3PARA(aux0, svmsb_f32_x, pg, x1, VC_EXPF_LN2HI, aux0); \
        VML_VC_FUNC_PG_N2_2PARA(x1, svsub_f32_x, pg, x1, VC_FLOAT_ONE); \
        VML_VC_DECLARE(svint32_t, aux1); \
        VML_VC_FUNC_PG_1PARA(aux1, svcvt_s32_x, pg, x1); \
        VML_VC_FUNC_PG_N2_2PARA(aux1, svadd_n_s32_x, pg, aux1, FLOAT_EXP_BIAS); \
        VML_VC_FUNC_PG_N2_2PARA(aux1, svlsl_n_s32_x, pg, aux1, FLOAT_MANTISSA_BIT_LEN); \
        VML_VC_DECLARE(svfloat32_t, poly); \
        VML_VC_FUNC_PG_N13_3PARA(poly, svmad_f32_x, pg, VC_EXPF_C5, aux0, VC_EXPF_C4); \
        VML_VC_FUNC_PG_N3_3PARA(poly, svmad_f32_x, pg, poly, aux0, VC_EXPF_C3); \
        VML_VC_FUNC_PG_N3_3PARA(poly, svmad_f32_x, pg, poly, aux0, VC_EXPF_C2); \
        VML_VC_FUNC_PG_N3_3PARA(poly, svmad_f32_x, pg, poly, aux0, VC_EXPF_C1); \
        VML_VC_FUNC_PG_N3_3PARA(poly, svmad_f32_x, pg, poly, aux0, VC_EXPF_C0); \
        VML_VC_FUNC_NOPG_1PARA(res, svreinterpret_f32_s32, aux1); \
        VML_VC_FUNC_PG_2PARA(res, svmul_f32_x, pg, poly, res); \
        VML_VC_FUNC_PG_N2_2PARA(res, svmul_f32_x, pg, res, VC_FLOAT_TWO); \
        VML_VC_FUNC_NOPG_N2_3PARA(res, svsel_f32, pgRetZero, VC_FLOAT_ZERO, res); \
    } while (0)

struct VC_EXPM1F_DATA {
    /* Grouped together so they can be loaded as one quadword so can use _lane forms of svmla/svmls */
    float c2, c4, ln2Hi, ln2Lo;
    float c0, c1, c3, invLn2, maxThr, shift;
};
extern __attribute__((aligned(0x20))) const struct VC_EXPM1F_DATA G_VC_EXPM1F_DATA;

inline svfloat32_t VsExpm1PolyVC(svbool_t pg, svfloat32_t r,
    svfloat32_t laneVal, const struct VC_EXPM1F_DATA *d)
{
    svfloat32_t r2 = svmul_f32_x(pg, r, r);

    svfloat32_t fP0403 = svmla_lane_f32(svdup_n_f32(d->c1), r, laneVal, 0x0);
    svfloat32_t fP0605 = svmla_lane_f32(svdup_n_f32(d->c3), r, laneVal, 0x1);
    svfloat32_t fP0603 = svmla_f32_x(pg, fP0403, r2, fP0605);
    svfloat32_t fP0602 = svmla_f32_x(pg, svdup_n_f32(d->c0), r, fP0603);
    svfloat32_t p = svmla_f32_x(pg, r, r2, fP0602);

    return p;
}

inline svfloat32_t VsExpm1InlineVC(svbool_t pg, svfloat32_t x)
{
    const struct VC_EXPM1F_DATA *d = PTR_BARRIER(&G_VC_EXPM1F_DATA);
    svfloat32_t laneVal = svld1rq_f32(svptrue_b32(), &d->c2);
    svfloat32_t z = svmla_n_f32_x(pg, svdup_n_f32(d->shift), x, d->invLn2);
    svfloat32_t exp2n = svreinterpret_f32_u32(svlsl_n_u32_x(pg, svreinterpret_u32_f32(z),
        FLOAT_MANTISSA_BIT_LEN));
    svfloat32_t n = svsub_n_f32_x(pg, z, d->shift);
    svfloat32_t r = svmls_lane_f32(svmls_lane_f32(x, n, laneVal, 0x2), n, laneVal, 0x3);
    svfloat32_t p = VsExpm1PolyVC(pg, r, laneVal, d);

    svfloat32_t t = svsub_f32_x(pg, exp2n, VC_FLOAT_ONE);
    return svmla_f32_x(pg, t, exp2n, p);
}

#endif
