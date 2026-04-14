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

#include "vml_sve_la_ln.h"
#define VML_VectorComputation_UNFOLD_N 4
#include "kernels/vml/common_macro.h"

static void KplVmlVsLnLaSveMainN(const float *src, float *dst, int step)
{
    svbool_t pg = svptrue_b32();
    VML_VectorComputation_DECLARE(svfloat32_t, x);
    VML_VectorComputation_DECLARE(svuint32_t, ux);
    VML_VectorComputation_DECLARE(svuint32_t, uxSubLnfOffset);
    VML_VectorComputation_DECLARE(svuint32_t, uxSubLnfOffsetMant);
    VML_VectorComputation_DECLARE(svuint32_t, uxAbsScale);
    VML_VectorComputation_DECLARE(svfloat32_t, m);
    VML_VectorComputation_DECLARE(svfloat32_t, r);
    VML_VectorComputation_DECLARE(svfloat32_t, r2);
    VML_VectorComputation_DECLARE(svfloat32_t, taylor);
    VML_VectorComputation_DECLARE(svint32_t, n);
    VML_VectorComputation_DECLARE(svfloat32_t, resHi);
    VML_VectorComputation_DECLARE(svfloat32_t, res);

    VML_VectorComputation_LD_STEP(x, svld1_f32, pg, src, step);
    VML_VectorComputation_FUNC_NOPG_1PARA(ux, svreinterpret_u32_f32, x);

    VML_VectorComputation_DECLARE(svbool_t, cmpEqInf);
    VML_VectorComputation_FUNC_PG_N2_2PARA(cmpEqInf, svcmpeq_u32, pg, ux, VectorComputation_FLOAT_INF_BIT);
    VML_VectorComputation_DECLARE(svbool_t, cmpEqZero);
    VML_VectorComputation_FUNC_PG_N2_2PARA(cmpEqZero, svcmpeq_n_f32, pg, x, FLOAT_ZERO);
    VML_VectorComputation_DECLARE(svbool_t, cmpLeZero);
    VML_VectorComputation_FUNC_PG_N2_2PARA(cmpLeZero, svcmple_n_f32, pg, x, FLOAT_ZERO);

    VML_VectorComputation_FUNC_PG_N2_2PARA(uxSubLnfOffset, svsub_n_u32_x, pg, ux, VSLN_OFF);
    VML_VectorComputation_FUNC_PG_N2_2PARA(uxSubLnfOffsetMant, svand_n_u32_x, pg, uxSubLnfOffset, FLOAT_MANTISSA_MASK);
    VML_VectorComputation_FUNC_NOPG_1PARA(n, svreinterpret_s32_u32, uxSubLnfOffset);
    VML_VectorComputation_FUNC_PG_N2_2PARA(uxAbsScale, svadd_n_u32_x, pg, uxSubLnfOffsetMant, VSLN_OFF);
    VML_VectorComputation_FUNC_NOPG_1PARA(m, svreinterpret_f32_u32, uxAbsScale);
    VML_VectorComputation_FUNC_PG_N2_2PARA(r, svsub_n_f32_x, pg, m, FLOAT_ONE);

    VML_VectorComputation_FUNC_PG_N2_2PARA(n, svasr_n_s32_x, pg, n, FLOAT_MANTISSA_BIT_LEN);
    VML_VectorComputation_FUNC_PG_1PARA(resHi, svcvt_f32_x, pg, n);

    VML_VectorComputation_FUNC_PG_N13_3PARA(taylor, svmad_f32_x, pg, VectorComputation_VSLN_POLY_C8, r, VectorComputation_VSLN_POLY_C7);
    VML_VectorComputation_FUNC_PG_N3_3PARA(taylor, svmad_f32_x, pg, taylor, r, VectorComputation_VSLN_POLY_C6);
    VML_VectorComputation_FUNC_PG_N3_3PARA(taylor, svmad_f32_x, pg, taylor, r, VectorComputation_VSLN_POLY_C5);
    VML_VectorComputation_FUNC_PG_N3_3PARA(taylor, svmad_f32_x, pg, taylor, r, VectorComputation_VSLN_POLY_C4);
    VML_VectorComputation_FUNC_PG_N3_3PARA(taylor, svmad_f32_x, pg, taylor, r, VectorComputation_VSLN_POLY_C3);
    VML_VectorComputation_FUNC_PG_N3_3PARA(taylor, svmad_f32_x, pg, taylor, r, VectorComputation_VSLN_POLY_C2);
    VML_VectorComputation_FUNC_PG_2PARA(r2, svmul_f32_x, pg, r, r);
    VML_VectorComputation_FUNC_PG_N2_3PARA(resHi, svmad_f32_x, pg, resHi, VectorComputation_FLOAT_LN2, r);
    VML_VectorComputation_FUNC_PG_3PARA(res, svmad_f32_x, pg, r2, taylor, resHi);

    VML_VectorComputation_FUNC_NOPG_N2_3PARA(res, svsel_f32, cmpLeZero, svreinterpret_f32_u32(svdup_n_u32(0x7fc00000)), res);
    VML_VectorComputation_FUNC_NOPG_N2_3PARA(res, svsel_f32, cmpEqZero, VectorComputation_FLOAT_NEG_INF, res);
    VML_VectorComputation_FUNC_NOPG_N2_3PARA(res, svsel_f32, cmpEqInf, VectorComputation_FLOAT_INF, res);

    VML_VectorComputation_ST_STEP(svst1_f32, pg, dst, step, res);
}

void LnLaSveFWDF32(const int64_t len, const float *src, float *dst)
{
    int loopLnf = svcntw();
    int64_t i;
    for (i = 0; i <= len - loopLnf * VML_VectorComputation_UNFOLD_N; i += loopLnf * VML_VectorComputation_UNFOLD_N) {
        KplVmlVsLnLaSveMainN(src + i, dst + i, loopLnf);
    }
    for (; i < len; i += loopLnf) {
        svbool_t pg = svwhilelt_b32_s64(i, len);
        svfloat32_t x = svld1_f32(pg, src + i);
        svfloat32_t y = SvLnF32X(x, pg);
        svst1_f32(pg, dst + i, y);
    }
}

void LnLaSveBWDF32(const int64_t len, const float *src, const float *dd, float *ds)
{
    int loopLnf = svcntw();
    int64_t i;
    for (i = 0; i < len; i += loopLnf) {
        svbool_t pg = svwhilelt_b32_s64(i, len);
        svfloat32_t x = svld1_f32(pg, src + i);
        svfloat32_t ddX = svld1_f32(pg, dd + i);

        svfloat32_t res = svdiv_f32_x(pg, ddX, x);
        svst1_f32(pg, ds + i, res);
    }
}
