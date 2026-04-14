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
 
#include "kernels/vml/vml_sve_la_exp.h"

#define VectorComputation_EXPH_TABLE_BITS 5
#define VectorComputation_EXPH_N (1 << VectorComputation_EXPH_TABLE_BITS)

#define VectorComputation_EXPH_LOWBOUND svdup_n_f16(-0x1.708p3f) // ~-11.5
#define VectorComputation_EXPH_UPBOUND svdup_n_f16(0x1.630p3f) // ~11.09

void ExpLaSveFWDF16(const int64_t len, const __fp16 *src, __fp16 *dst)
{
    int64_t i;
    int loopExph = svcnth();
    for (i = 0; i < len; i += loopExph) {
        svbool_t pgB16 = svwhilelt_b16_s64(i, len);
        svbool_t pgB32 = svwhilelt_b32_s64(i, len);

        // convert to fp32
        svfloat16_t x = svld1_f16(pgB16, src + i);
        svbool_t pgRetZero = svcmple_f16(pgB16, x, VectorComputation_EXPH_LOWBOUND);
        svbool_t pgRetInf = svcmpge_f16(pgB16, x, VectorComputation_EXPH_UPBOUND);
        svfloat32_t xEven = svcvt_f32_f16_x(pgB16, x);
        svfloat32_t xOdd = CvtOddF32F16(x, pgB16, pgB32);

        svfloat32_t ki = svmad_f32_x(pgB32, xEven, VectorComputation_EXPF_INVLN2HI, VectorComputation_EXP2F_SHIFT);
        svfloat32_t kd = svsub_f32_x(pgB32, ki, VectorComputation_EXP2F_SHIFT);
        svuint32_t k = svreinterpret_u32_f32(ki);
        svfloat32_t expk = svexpa_f32(k);
        svfloat32_t r = svnmls_f32_x(pgB32, kd, xEven, VectorComputation_EXPF_INVLN2HI);
        r = svmla_f32_x(pgB32, r, xEven, VectorComputation_EXPF_INVLN2LO);
        svfloat32_t poly = svmla_f32_x(pgB32, VectorComputation_EXP2F_C1, r, VectorComputation_EXP2F_C0);
        poly = svmla_f32_x(pgB32, VectorComputation_EXP2F_C2, poly, r);
        svfloat32_t resEven = svmul_f32_x(pgB32, expk, poly);

        ki = svmad_f32_x(pgB32, xOdd, VectorComputation_EXPF_INVLN2HI, VectorComputation_EXP2F_SHIFT);
        kd = svsub_f32_x(pgB32, ki, VectorComputation_EXP2F_SHIFT);
        k = svreinterpret_u32_f32(ki);
        expk = svexpa_f32(k);
        r = svnmls_f32_x(pgB32, kd, xOdd, VectorComputation_EXPF_INVLN2HI);
        r = svmla_f32_x(pgB32, r, xOdd, VectorComputation_EXPF_INVLN2LO);
        poly = svmla_f32_x(pgB32, VectorComputation_EXP2F_C1, r, VectorComputation_EXP2F_C0);
        poly = svmla_f32_x(pgB32, VectorComputation_EXP2F_C2, poly, r);
        svfloat32_t resOdd = svmul_f32_x(pgB32, expk, poly);

        // convert back to fp16
        svfloat16_t res = svcvt_f16_f32_x(pgB32, resEven);
        x = CvtOddF16F32(resOdd, pgB32); /* resOdd(f16) */
        res = svsel_f16(pgB32, res, x);

        res = svdup_n_f16_m(res, pgRetZero, FLOAT_ZERO);
        res = svdup_n_f16_m(res, pgRetInf, INFINITY);

        svst1_f16(pgB16, dst + i, res);
    }
}

void ExpLaSveBWDF16(const int64_t len, const __fp16 *src, const __fp16 *dd, __fp16 *ds)
{
    int64_t i;
    int loopExph = svcnth();
    for (i = 0; i < len; i += loopExph) {
        svbool_t pgB16 = svwhilelt_b16_s64(i, len);
        svbool_t pgB32 = svwhilelt_b32_s64(i, len);

        svfloat16_t srcX = svld1_f16(pgB16, src + i);
        svfloat32_t xEven = svcvt_f32_f16_x(pgB16, srcX);
        svfloat32_t xOdd = CvtOddF32F16(srcX, pgB16, pgB32);

        svfloat16_t ddX = svld1_f16(pgB16, dd + i);
        svfloat32_t ddEven = svcvt_f32_f16_x(pgB16, ddX);
        svfloat32_t ddOdd = CvtOddF32F16(ddX, pgB16, pgB32);
        svbool_t pgRetInf = svacge_n_f16(pgB16, ddX, INFINITY);
        pgRetInf = svcmpgt_n_f16(pgRetInf, srcX, -0x1.ap6f); /* exp(-0x1.ap6f) = 0 */

        svfloat32_t resEven = SvExpF32X(xEven, pgB32);
        svfloat32_t resOdd = SvExpF32X(xOdd, pgB32);
        resEven = svmul_f32_m(pgB32, resEven, ddEven);
        resOdd = svmul_f32_m(pgB32, resOdd, ddOdd);

        svfloat16_t res = svcvt_f16_f32_x(pgB32, resEven); /* resEven(f16) */
        srcX = CvtOddF16F32(resOdd, pgB32); /* resOdd(f16) */
        res = svsel_f16(pgB32, res, srcX);
        res = svsel_f16(pgRetInf, ddX, res);

        svst1_f16(pgB16, ds + i, res);
    }
}
