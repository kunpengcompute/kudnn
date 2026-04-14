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

#include "kernels/vml/vml_sve_la_tanh.h"

#define FP16_MANTISSA_BIT_LEN 10
#define VectorComputation_VHTANH_INVLN2 svdup_n_f16(0x1.714p+0f)
#define VectorComputation_VHTANH_LN2HI svdup_n_f16(0x1.62cp-1f)
#define VectorComputation_VHTANH_LN2LO svdup_n_f16(0x1.218p-12f)
#define VectorComputation_VHTANH_SHIFT svdup_n_f16(0x1.83cp10f)
#define VectorComputation_VHTANH_TINY_TH svdup_n_f16(0x1.34p-2f)
#define VectorComputation_VHTANH_ONE_TH svdup_n_f16(0x1.0p2)
#define VectorComputation_VHTANH_NEG_ONE_TH svdup_n_f16(-0x1.fa8p1)

/* sollya: p=fpminimax((exp(x)-1)/x, [|0,1,2|], [|10...|], [-log(2)/2,log(2)/2], relative)
 * sollya: supnorm(1+x*p,exp(x),[-log(2)/2,log(2)/2],relative,0x1p-10);
 * error: [0x1.58dd792p-13;0x1.5930fec355cp-13]
 */
#define VectorComputation_VHTANH_C0 svdup_n_f16(0x1p0f)
#define VectorComputation_VHTANH_C1 svdup_n_f16(0x1.02p-1f)
#define VectorComputation_VHTANH_C2 svdup_n_f16(0x1.568p-3f)

/* sollya: p=fpminimax(tanh(x), [|1,3|], [|10...|], [0, 0x1.34p-2], relative)
 * sollya: supnorm(p,tanh(x),[0,0x1.34p-2],relative,0x1p-10)
 * error: [0x1.8d66b19p-13;0x1.8dc6f06f00ep-13]
 */
#define VectorComputation_VHTANH_TINY_C0 svdup_n_f16(0x1p0f)
#define VectorComputation_VHTANH_TINY_C1 svdup_n_f16(-0x1.4b8p-2f)

__attribute__((always_inline)) static inline svfloat16_t VhExpm1PolySveF16(svbool_t pg, svfloat16_t r)
{
    svfloat16_t poly = svmla_f16_x(pg, VectorComputation_VHTANH_C1, r, VectorComputation_VHTANH_C2);
    poly = svmla_f16_x(pg, VectorComputation_VHTANH_C0, r, poly);
    poly = svmul_f16_x(pg, poly, r);
    return poly;
}

__attribute__((always_inline)) static inline svfloat16_t VhExpm1InlineSveF16(svbool_t pg, svfloat16_t x)
{
    svfloat16_t z = svmla_f16_x(pg, VectorComputation_VHTANH_SHIFT, x, VectorComputation_VHTANH_INVLN2);
    svfloat16_t exp2n = svreinterpret_f16_u16(svlsl_n_u16_x(pg, svreinterpret_u16_f16(z), FP16_MANTISSA_BIT_LEN));
    svfloat16_t n = svsub_f16_x(pg, z, VectorComputation_VHTANH_SHIFT);
    svfloat16_t r = svmls_f16_x(pg, svmls_f16_x(pg, x, n, VectorComputation_VHTANH_LN2HI), n, VectorComputation_VHTANH_LN2LO);
    svfloat16_t p = VhExpm1PolySveF16(pg, r);

    svfloat16_t t = svsub_f16_x(pg, exp2n, VML_VectorComputation_F16_ONE);
    return svmla_f16_x(pg, t, exp2n, p);
}

static svfloat16_t SvTanhF16X(svfloat16_t x, svbool_t pg)
{
    svfloat16_t xAbs = svabs_f16_x(pg, x);
    svbool_t cmpTiny = svacge_f16(pg, xAbs, VectorComputation_VHTANH_TINY_TH);
    svbool_t pgRetOne = svcmpge_f16(pg, x, VectorComputation_VHTANH_ONE_TH);
    svbool_t pgRetNegOne = svcmple_f16(pg, x, VectorComputation_VHTANH_NEG_ONE_TH);
    svuint16_t xSign = svand_u16_x(pg, svreinterpret_u16_f16(x), svdup_n_u16(0x8000U));

    /* e^2x - 1 */
    svfloat16_t e2m1 = VhExpm1InlineSveF16(pg, svmul_f16_x(pg, xAbs, VML_VectorComputation_F16_TWO));
    /* e^2x + 1 */
    svfloat16_t e2a1 = svadd_f16_x(pg, e2m1, VML_VectorComputation_F16_TWO);
    /* tanh(x) = (e^2x - 1) / (e^2x + 1) */
    svfloat16_t res = svdiv_f16_x(pg, e2m1, e2a1);

    svfloat16_t x2 = svmul_f16_x(pg, xAbs, xAbs);
    svfloat16_t poly = svmla_f16_x(pg, VectorComputation_VHTANH_TINY_C0, x2, VectorComputation_VHTANH_TINY_C1);
    svfloat16_t resTiny = svmul_f16_x(pg, xAbs, poly);

    res = svsel_f16(cmpTiny, res, resTiny);
    res = SvOrrF16X(pg, res, xSign);
    res = svdup_n_f16_m(res, pgRetOne, FLOAT_ONE);
    res = svdup_n_f16_m(res, pgRetNegOne, FLOAT_NEG_ONE);

    return res;
}

void TanhLaSveFWDF16(const int64_t len, const __fp16 *src, __fp16 *dst)
{
    int64_t i;
    int loopTanhh = svcnth();
    for (i = 0; i < len; i += loopTanhh) {
        svbool_t pg = svwhilelt_b16_s64(i, len);
        svfloat16_t x = svld1_f16(pg, src + i);
        svfloat16_t res = SvTanhF16X(x, pg);
        svst1_f16(pg, dst + i, res);
    }
}

void TanhLaSveBWDF16(const int64_t len, const __fp16 *src, const __fp16 *dd, __fp16 *ds)
{
    int64_t i;
    int loopTanhh = svcnth();
    for (i = 0; i < len; i += loopTanhh) {
        svbool_t pgB16 = svwhilelt_b16_s64(i, len);
        svbool_t pgB32 = svwhilelt_b32_s64(i, len);

        svfloat16_t srcX = svld1_f16(pgB16, src + i);
        svfloat16_t ddX = svld1_f16(pgB16, dd + i);

        svfloat32_t srcXfEven = svcvt_f32_f16_x(pgB16, srcX);
        svfloat32_t srcXfOdd = CvtOddF32F16(srcX, pgB16, pgB32);

        svfloat32_t ddXfEven = svcvt_f32_f16_x(pgB16, ddX);
        svfloat32_t ddXfOdd = CvtOddF32F16(ddX, pgB16, pgB32);

        svfloat32_t srcXEvenTanhX = SvTanhF32X(srcXfEven, pgB32);
        svfloat32_t srcXOddTanhX = SvTanhF32X(srcXfOdd, pgB32);

        svfloat32_t ddXEvenRes = svmls_f32_x(pgB32, VectorComputation_FLOAT_ONE, srcXEvenTanhX, srcXEvenTanhX);
        svfloat32_t ddXOddRes = svmls_f32_x(pgB32, VectorComputation_FLOAT_ONE, srcXOddTanhX, srcXOddTanhX);
        ddXEvenRes = svmul_f32_x(pgB32, ddXfEven, ddXEvenRes);
        ddXOddRes = svmul_f32_x(pgB32, ddXfOdd, ddXOddRes);

        svfloat16_t resEven = svcvt_f16_f32_x(pgB32, ddXEvenRes);
        svfloat16_t resOdd = CvtOddF16F32(ddXOddRes, pgB32);
        svfloat16_t res = svadd_f16_x(pgB16, resEven, resOdd);

        svst1_f16(pgB16, ds + i, res);
    }
}
