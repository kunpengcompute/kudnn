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

#ifndef VML_VC_LA_LN_H
#define VML_VC_LA_LN_H

#include "kernels/vml/vml_sve_base.h"

#define VSLN_OFF (0x3f2aaaabU)

extern const float G_VML_VC_VSLOG_POLY3578[VC_FLOAT_LD_POLY_SIZE];
extern const float G_VML_VC_VSLN_POLY[0x3];

#define VC_VSLN_POLY_C2 svdup_n_f32(G_VML_VC_VSLN_POLY[0x0])
#define VC_VSLN_POLY_C3 svdup_n_f32(G_VML_VC_VSLOG_POLY3578[0x0])
#define VC_VSLN_POLY_C4 svdup_n_f32(G_VML_VC_VSLN_POLY[0x1])
#define VC_VSLN_POLY_C5 svdup_n_f32(G_VML_VC_VSLOG_POLY3578[0x1])
#define VC_VSLN_POLY_C6 svdup_n_f32(G_VML_VC_VSLN_POLY[0x2])
#define VC_VSLN_POLY_C7 svdup_n_f32(G_VML_VC_VSLOG_POLY3578[0x2])
#define VC_VSLN_POLY_C8 svdup_n_f32(G_VML_VC_VSLOG_POLY3578[0x3])

inline svfloat32_t VsLnLaVCPolyNew(svbool_t pg, svfloat32_t r, svfloat32_t n, float ln2)
{
    svfloat32_t poly = svld1_f32(svptrue_b32(), G_VML_VC_VSLOG_POLY3578);
    svfloat32_t r2 = svmul_f32_x(pg, r, r);
    svfloat32_t taylor67 = svmla_lane_f32(VC_VSLN_POLY_C6, r, poly, 0x2ULL);
    svfloat32_t taylor45 = svmla_lane_f32(VC_VSLN_POLY_C4, r, poly, 0x1ULL);
    svfloat32_t taylor23 = svmla_lane_f32(VC_VSLN_POLY_C2, r, poly, 0x0ULL);
    svfloat32_t taylor = svmla_lane_f32(taylor67, r2, poly, 0x3ULL);
    taylor = svmad_f32_x(pg, taylor, r2, taylor45);
    taylor = svmad_f32_x(pg, taylor, r2, taylor23);
    svfloat32_t resHi = svmla_n_f32_x(pg, r, n, ln2);
    return svmad_f32_x(pg, taylor, r2, resHi);
}

__attribute__((always_inline)) inline svfloat32_t SvLnF32X(svfloat32_t x, svbool_t pg)
{
    svuint32_t ux = svreinterpret_u32_f32(x);
    svbool_t cmpEqInf = svcmpeq_u32(pg, ux, VC_FLOAT_INF_BIT);
    svbool_t cmpEqZero = svcmpeq_n_f32(pg, x, FLOAT_ZERO);
    svbool_t cmpLeZero = svcmple_n_f32(pg, x, FLOAT_ZERO);

    // x = m * 2^n, 0x1.8p-1 <= m < 0x1.8p0
    svuint32_t uxSubLnfOffset = svsub_n_u32_x(pg, ux, VSLN_OFF);
    svint32_t sn = svasr_n_s32_x(pg, svreinterpret_s32_u32(uxSubLnfOffset), FLOAT_MANTISSA_BIT_LEN);
    svfloat32_t n = svcvt_f32_x(pg, sn);
    svuint32_t uxSubLnfOffsetMant = svand_n_u32_x(pg, uxSubLnfOffset, FLOAT_MANTISSA_MASK);
    // r = m - 1
    svuint32_t uxAbsScale = svadd_n_u32_x(pg, uxSubLnfOffsetMant, VSLN_OFF);
    svfloat32_t r = svsub_n_f32_x(pg, svreinterpret_f32_u32(uxAbsScale), FLOAT_ONE);
    svfloat32_t res = VsLnLaVCPolyNew(pg, r, n, LN2F);

    res = svsel_f32(cmpLeZero, svreinterpret_f32_u32(svdup_n_u32(0x7ff80000)), res);
    res = svsel_f32(cmpEqZero, VC_FLOAT_NEG_INF, res);
    res = svsel_f32(cmpEqInf, VC_FLOAT_INF, res);

    return res;
}

#endif
