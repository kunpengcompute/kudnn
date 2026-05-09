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

#ifndef VML_VectorComputation_LA_POW_H
#define VML_VectorComputation_LA_POW_H

#include "kernels/vml/vml_sve_base.h"
#include "vml_pow.h"

#define VectorComputation_VSPOW_SHIFT svdup_n_f32(+0x1.803f8000p+0017f)
#define VectorComputation_VSPOW_LN2 svdup_n_f32(+0x1.62e43000p-0001f)
#define VectorComputation_VSPOW_LN2_LO svdup_n_f32(-0x1.05c61000p-0029f)
#define VectorComputation_VSPOW_EXP_C0 svdup_f32(1.0f)
#define VectorComputation_VSPOW_EXP_C1 svdup_f32(0x1.62e584p-1f)
#define VectorComputation_VSPOW_EXP_C2 svdup_f32(0x1.ebfc28p-3f)
#define VectorComputation_VSPOW_LOG_C0 svdup_f32(0x1.715476p0f)
#define VectorComputation_VSPOW_LOG_C0_LO svdup_f32(0x1.4ae198p-26f)
#define VectorComputation_VSPOW_LOG_C1 svdup_f32(-0x1.715476p-1f)
#define VectorComputation_VSPOW_LOG_C2 svdup_f32(0x1.ec709ep-2f)
#define VectorComputation_VSPOW_LOG_C3 svdup_f32(-0x1.7167dap-2f)
#define VectorComputation_VSPOW_LOG_C4 svdup_f32(0x1.278e0ap-2f)
#define VSPOW_EXP_MAX 0x42fc0000U
#define VSPOW_SP_TH_MASK (0x7e800000U - POWF_X_TH_LO)
#define VSPOW_LOG_IDX_MASK 0x1fU
#define VSPOW_LOG_TAB_SIZE 32

inline svfloat32_t VsPowLnGetPartSve(svbool_t pg, svfloat32_t x, svfloat32_t *n, svuint32_t *idx)
{
    svuint32_t uRcp = svreinterpret_u32_f32(svrecpe_f32(x));
    uRcp = svand_n_u32_x(pg, svadd_n_u32_x(pg, uRcp, 0x00020000U), 0xfffc0000U); /* round to nearest at mant 6 bit and
                                                                                    mask the high 5 bit mant */
    svuint32_t uExp = svlsr_n_u32_x(pg, uRcp, FLOAT_MANTISSA_BIT_LEN);
    svint32_t sn = svsubr_n_s32_x(pg, svreinterpret_s32_u32(uExp), FLOAT_EXP_BIAS);
    svuint32_t uMant = svlsr_n_u32_x(pg, uRcp, 0x12U); /* right shift 18 bits to get rid of mant 6 - 23 bit */
    *n = svcvt_f32_s32_x(pg, sn);
    svfloat32_t r = svmla_f32_x(pg, VectorComputation_FLOAT_NEG_ONE, x, svreinterpret_f32_u32(uRcp));
    *idx = svand_n_u32_x(pg, uMant, VSPOW_LOG_IDX_MASK);
    return r;
}

inline svfloat32_t VsPowLnTaylorSve(svbool_t pg, svfloat32_t r, svfloat32_t lnLo)
{
    svfloat32_t poly = svmla_f32_x(pg, VectorComputation_VSPOW_LOG_C3, r, VectorComputation_VSPOW_LOG_C4);
    poly = svmla_f32_x(pg, VectorComputation_VSPOW_LOG_C2, poly, r);
    poly = svmla_f32_x(pg, VectorComputation_VSPOW_LOG_C1, poly, r);
    poly = svmla_f32_x(pg, VectorComputation_VSPOW_LOG_C0_LO, poly, r);
    poly = svmla_f32_x(pg, lnLo, poly, r);

    return poly;
}

inline svfloat32_t VsPowLnGetResSve(svbool_t pg, svfloat32_t x, svfloat32_t *lo)
{
    svfloat32_t n;
    svuint32_t idx;
    svfloat32_t r = VsPowLnGetPartSve(pg, x, &n, &idx);
    svfloat32_t p3 = svadd_f32_x(pg, n, svld1_gather_u32index_f32(pg, G_VML_VSPOW_LOG2_TAB_VectorComputation, idx));
    svfloat32_t poly =
        VsPowLnTaylorSve(pg, r, svld1_gather_u32index_f32(pg, G_VML_VSPOW_LOG2_POLY_LOW_TAB_VectorComputation, idx));

    svfloat32_t p4 = svmla_f32_x(pg, p3, r, VectorComputation_VSPOW_LOG_C0);
    svfloat32_t p4Lo = svmla_f32_x(pg, svsub_f32_x(pg, p3, p4), r, VectorComputation_VSPOW_LOG_C0);

    svfloat32_t log2x = svadd_f32_x(pg, p4, poly);
    *lo = svadd_f32_x(pg, p4Lo, svsub_f32_x(pg, poly, svsub_f32_x(pg, log2x, p4)));
    return log2x;
}
/*
 * x^y = 2^(ylog2(x))
 * step 1: calc log2(x)
 * x = 2^xExp * (1 + xMan) * (1 + r)
 * log2(x) = xExp + log2(1 + xMan) + log2(1 + r)
 * = (xExp + p1) + p2
 * = p3 + p2 = (p3 + r*C0) + poly = p4 + poly
 * get p1 by tlb search, get p2 by polynominal expansion
 * return ylog2xHi and ylog2xtail
 */
inline svfloat32_t VsPowYLnXSve(svbool_t pg, svfloat32_t x, svfloat32_t y, svfloat32_t *tail)
{
    svfloat32_t log2xLo;
    svfloat32_t log2x = VsPowLnGetResSve(pg, x, &log2xLo);
    svfloat32_t yLog2x = svmul_f32_x(pg, log2x, y);
    svfloat32_t yLog2xLo = svmls_f32_x(pg, yLog2x, log2x, y);
    *tail = svmls_f32_x(pg, yLog2xLo, log2xLo, y);
    return yLog2x;
}

/*
 * calc 2^(x + tail), here tail is remainder of x bit 5~23 from decimal point
 * x + tail = k1 + k2 + tail,
 * k1 is integer, k2 is 4bit after decimal point,
 * 2^(x + tail) = 2^k1 * (2^k2 * 2^tail)
 * = exp2P1 * (exp2P2 * exp2P3) = exp2P1 * exp2P4
 */
inline svfloat32_t VsPowExpSve(svbool_t pg, svfloat32_t xHi, svfloat32_t xLo)
{
    svfloat32_t exp2SrcHi = svsub_f32_x(pg, xHi, xLo);
    svfloat32_t exp2SrcLo = svsub_f32_x(pg, svsub_f32_x(pg, xHi, exp2SrcHi), xLo);

    svfloat32_t z = svadd_f32_x(pg, exp2SrcHi, VectorComputation_VSPOW_SHIFT);
    svfloat32_t kd = svsub_f32_x(pg, z, VectorComputation_VSPOW_SHIFT);

    svfloat32_t rExp = svsub_f32_x(pg, exp2SrcHi, kd);
    rExp = svadd_f32_x(pg, rExp, exp2SrcLo);

    svfloat32_t expK = svexpa_f32(svreinterpret_u32_f32(z));
    svfloat32_t polyExp = svmla_f32_x(pg, VectorComputation_VSPOW_EXP_C1, rExp, VectorComputation_VSPOW_EXP_C2);
    polyExp = svmla_f32_x(pg, VectorComputation_VSPOW_EXP_C0, polyExp, rExp);

    return svmul_f32_x(pg, expK, polyExp);
}

inline svfloat32_t VsPowLaSveMain(svbool_t pg, svfloat32_t x, svfloat32_t y)
{
    svuint32_t ux = svreinterpret_u32_f32(x);
    svbool_t cmp = svcmpge_n_u32(pg, svsub_n_u32_x(pg, ux, POWF_X_TH_LO), VSPOW_SP_TH_MASK);

    svfloat32_t yLog2xLo;
    svfloat32_t yLog2x = VsPowYLnXSve(pg, x, y, &yLog2xLo);

    svbool_t cmpExp = svcmpge_n_u32(pg, svreinterpret_u32_f32(svabs_f32_x(pg, yLog2x)), VSPOW_EXP_MAX);

    svfloat32_t result = VsPowExpSve(pg, yLog2x, yLog2xLo);

    svbool_t cmpSp = svorr_z(pg, cmp, cmpExp);
    if (unlikely(svptest_any(pg, cmpSp))) {
        result = VsFunc2f1Special(cmpSp, x, y, result, powf);
    }
    return result;
}

/* vspowx special case: y is positive integer or 0.5 */
inline svfloat32_t VsPowxaSPF32(svbool_t pg, const svfloat32_t x, const float y)
{
    svfloat32_t res = svdup_n_f32(1.0f);
    const int iy = (int)y;
    for (int i = 0; i < iy; i++) {
        res = svmul_f32_x(pg, res, x);
    }
    return res;
}

#endif
