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
#ifndef VML_VC_LA_TANH_H
#define VML_VC_LA_TANH_H

#include "vml_sve_la_exp.h"

#define VC_VSTANH_SP_TH     svdup_n_f32(0x1.154246p3f)
#define VC_VSTANH_SP_TH_NEG svdup_n_f32(-0x1.154246p3f)

__attribute__((always_inline)) inline svfloat32_t SvTanhF32X(svfloat32_t x, svbool_t pg)
{
    svbool_t cmpEqZero = svcmpeq_n_f32(pg, x, FLOAT_ZERO);
    svbool_t cmpSp = svcmpge_f32(pg, x, VC_VSTANH_SP_TH);
    svbool_t cmpSpNeg = svcmple_f32(pg, x, VC_VSTANH_SP_TH_NEG);

    /* e^2x - 1 */
    svfloat32_t e2m1 = VsExpm1InlineVC(pg, svmul_f32_x(pg, x, VC_FLOAT_TWO));

    /* e^2x + 1 */
    svfloat32_t e2a1 = svadd_f32_x(pg, e2m1, VC_FLOAT_TWO);

    /* tanh(x) = (e^2x - 1) / (e^2x + 1) */
    svfloat32_t res = svdiv_f32_x(pg, e2m1, e2a1);

    res = svdup_n_f32_m(res, cmpSp, FLOAT_ONE);
    res = svdup_n_f32_m(res, cmpSpNeg, FLOAT_NEG_ONE);
    res = svsel_f32(cmpEqZero, x, res);

    return res;
}

#define VC_TANHF_TBL_SIZE 32
struct VmlVCTanhfData {
    float min;
    float max;
    float idxMin;
    float idxMax;
    uint32_t idxShift;
    uint32_t idxBias;
    uint32_t idxMask;
    float polyDegree0[VC_TANHF_TBL_SIZE];
    float polyDegree1[VC_TANHF_TBL_SIZE];
    float polyDegree2[VC_TANHF_TBL_SIZE];
    float polyDegree3[VC_TANHF_TBL_SIZE];
    float polyDegree4[VC_TANHF_TBL_SIZE];
    float polyDegree5[VC_TANHF_TBL_SIZE];
    float polyDegree6[VC_TANHF_TBL_SIZE];
};

extern const struct VmlVCTanhfData G_VML_VSTANH_DATA;

#define VC_TANHF_POLY_COEFFS_DEGREE0 G_VML_VSTANH_DATA.polyDegree0
#define VC_TANHF_POLY_COEFFS_DEGREE1 G_VML_VSTANH_DATA.polyDegree1
#define VC_TANHF_POLY_COEFFS_DEGREE2 G_VML_VSTANH_DATA.polyDegree2
#define VC_TANHF_POLY_COEFFS_DEGREE3 G_VML_VSTANH_DATA.polyDegree3
#define VC_TANHF_POLY_COEFFS_DEGREE4 G_VML_VSTANH_DATA.polyDegree4
#define VC_TANHF_POLY_COEFFS_DEGREE5 G_VML_VSTANH_DATA.polyDegree5
#define VC_TANHF_POLY_COEFFS_DEGREE6 G_VML_VSTANH_DATA.polyDegree6
#define VC_TANHF_IDX_SHIFT           svdup_n_u32(G_VML_VSTANH_DATA.idxShift)
#define VC_TANHF_IDX_BIAS            svdup_n_u32(G_VML_VSTANH_DATA.idxBias)
#define VC_TANHF_IDX_MASK            svdup_n_u32(G_VML_VSTANH_DATA.idxMask)
#define VC_TANHF_IDX_MIN             svdup_n_f32(G_VML_VSTANH_DATA.idxMin)
#define VC_TANHF_IDX_MAX             svdup_n_f32(G_VML_VSTANH_DATA.idxMax)
#define VC_TANHF_MIN                 svdup_n_f32(G_VML_VSTANH_DATA.min)
#define VC_TANHF_MAX                 svdup_n_f32(G_VML_VSTANH_DATA.max)

inline svfloat32_t VCTanhF32Poly(svbool_t pg, svuint32_t idx, svfloat32_t r)
{
    svfloat32_t c6 = svld1_gather_u32index_f32(pg, VC_TANHF_POLY_COEFFS_DEGREE6, idx);
    svfloat32_t c5 = svld1_gather_u32index_f32(pg, VC_TANHF_POLY_COEFFS_DEGREE5, idx);
    svfloat32_t c4 = svld1_gather_u32index_f32(pg, VC_TANHF_POLY_COEFFS_DEGREE4, idx);
    svfloat32_t c3 = svld1_gather_u32index_f32(pg, VC_TANHF_POLY_COEFFS_DEGREE3, idx);
    svfloat32_t c2 = svld1_gather_u32index_f32(pg, VC_TANHF_POLY_COEFFS_DEGREE2, idx);
    svfloat32_t c1 = svld1_gather_u32index_f32(pg, VC_TANHF_POLY_COEFFS_DEGREE1, idx);
    svfloat32_t c0 = svld1_gather_u32index_f32(pg, VC_TANHF_POLY_COEFFS_DEGREE0, idx);

    svfloat32_t p = svmla_f32_x(pg, c4, svmla_f32_x(pg, c5, c6, r), r);
    p = svmla_f32_x(pg, c2, svmla_f32_x(pg, c3, p, r), r);
    return svmla_f32_x(pg, c0, svmla_f32_x(pg, c1, p, r), r);
}

// xAbs<=min tanh(x)=x, xAbs>=max tanh(x)=1
__attribute__((always_inline)) inline svfloat32_t SvTanhF32XNew(svfloat32_t x, svbool_t pg)
{
    svfloat32_t xAbs = svabs_f32_x(pg, x);
    svuint32_t sign = svand_u32_x(pg, svreinterpret_u32_f32(x), VC_FLOAT_SIGN_BIT);

    svfloat32_t xAbsNew = svmin_f32_x(pg, svmax_f32_x(pg, xAbs, VC_TANHF_IDX_MIN), VC_TANHF_IDX_MAX);
    svuint32_t uxAbsNew = svreinterpret_u32_f32(xAbsNew);
    svuint32_t idx = svsub_u32_x(pg, svlsr_u32_x(pg, uxAbsNew, VC_TANHF_IDX_SHIFT), VC_TANHF_IDX_BIAS);

    svfloat32_t r = svsub_f32_x(pg, xAbs, SvAndF32X(pg, xAbs, VC_TANHF_IDX_MASK));
    svfloat32_t p = VCTanhF32Poly(pg, idx, r);

    svfloat32_t y = svsel_f32(svcmple_f32(pg, xAbs, VC_TANHF_MIN), x, p);
    y = svsel_f32(svcmpge_f32(pg, xAbs, VC_TANHF_MAX), VC_FLOAT_ONE, y);
    return SvOrrF32X(pg, y, sign);
}

#define VC_TANH_RANGE                  svdup_n_f32(0x1.99999ap-5f)
#define VC_TANH_NEG_ONE_THIRD          svdup_n_f32(-0x1.555556p-2f)

__attribute__((always_inline)) inline svfloat32_t SvTanhF32XNew2(svfloat32_t x, svbool_t pg)
{
    svfloat32_t xAbs = svabs_f32_x(pg, x);
    svbool_t pg1 = svcmplt(pg, xAbs, VC_TANH_RANGE);

    // xAbs not in range tanh(x)=1 - 2/(1+exp(2*x))
    // 1+exp(2*x)
    svfloat32_t t0 = svadd_f32_x(pg, x, x); // 2 * x
    t0 = SvExpF32X(t0, pg);
    t0 = svadd_f32_x(pg, VC_FLOAT_ONE, t0);
    // 1/(1+exp(2*x))
    svfloat32_t t1 = svrecpe_f32(t0);
    svfloat32_t t2 = svrecps_f32(t0, t1);
    t2 = svmul_f32_x(pg, t2, t1);
    t0 = svrecps_f32(t0, t2);
    t0 = svmul_f32_x(pg, t0, t2);
    // 1 - 2/(1+exp(2*x))
    svfloat32_t res1 = svadd_f32_x(pg, t0, t0);
    res1 = svsub_f32_x(pg, VC_FLOAT_ONE, res1);

    // xAbs in range tanh(x)=x*(1-(x^2)/3)
    svfloat32_t x2 = svmul_f32_x(pg, x, x); // x * x
    svfloat32_t t3 = svmad_f32_x(pg, VC_TANH_NEG_ONE_THIRD, x2, VC_FLOAT_ONE);
    svfloat32_t res2 = svmul_f32_x(pg, x, t3);

    return svsel_f32(pg1, res2, res1);
}

#endif
