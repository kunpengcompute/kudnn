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

void TanhLaVCFWDF16(const int64_t len, const __fp16 *src, __fp16 *dst)
{
    int64_t i;
    int loopTanhh = svcnth();
    for (i = 0; i < len; i += loopTanhh) {
        svbool_t pgB16 = svwhilelt_b16_s64(i, len);
        svbool_t pgB32 = svwhilelt_b32_s64(i, len);

        svfloat16_t x = svld1_f16(pgB16, src + i);

        svfloat32_t xEven = svcvt_f32_f16_x(pgB16, x);
        svfloat32_t xOdd = CvtOddF32F16(x, pgB16, pgB32);

        svfloat32_t xEvenRes = SvTanhF32X(xEven, pgB32);
        svfloat32_t xOddRes = SvTanhF32X(xOdd, pgB32);

        svfloat16_t resEven = svcvt_f16_f32_x(pgB32, xEvenRes);
        svfloat16_t resOdd = CvtOddF16F32(xOddRes, pgB32);
        svfloat16_t res = svadd_f16_x(pgB16, resEven, resOdd);

        svst1_f16(pgB16, dst + i, res);
    }
}

void TanhLaVCBWDF16(const int64_t len, const __fp16 *src, const __fp16 *dd, __fp16 *ds)
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

        svfloat32_t ddXEvenRes = svmls_f32_x(pgB32, VC_FLOAT_ONE, srcXEvenTanhX, srcXEvenTanhX);
        svfloat32_t ddXOddRes = svmls_f32_x(pgB32, VC_FLOAT_ONE, srcXOddTanhX, srcXOddTanhX);
        ddXEvenRes = svmul_f32_x(pgB32, ddXfEven, ddXEvenRes);
        ddXOddRes = svmul_f32_x(pgB32, ddXfOdd, ddXOddRes);

        svfloat16_t resEven = svcvt_f16_f32_x(pgB32, ddXEvenRes);
        svfloat16_t resOdd = CvtOddF16F32(ddXOddRes, pgB32);
        svfloat16_t res = svadd_f16_x(pgB16, resEven, resOdd);

        svst1_f16(pgB16, ds + i, res);
    }
}
