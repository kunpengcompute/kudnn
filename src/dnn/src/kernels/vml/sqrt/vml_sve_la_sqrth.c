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

#include "kernels/vml/vml_sve_base.h"

void SqrtLaSveFWDF16(const int64_t len, const __fp16 *src, __fp16 *dst)
{
    int64_t i;
    int loopSqrth = svcnth();
    for (i = 0; i < len; i += loopSqrth) {
        svbool_t pg = svwhilelt_b16_s64(i, len);
        svfloat16_t x = svld1_f16(pg, src + i);
        svfloat16_t res = svsqrt_f16_x(pg, x);
        svst1_f16(pg, dst + i, res);
    }
}

void SqrtLaSveBWDF16(const int64_t len, const __fp16 *src, const __fp16 *dd, __fp16 *ds)
{
    int64_t i;
    int loopSqrth = svcnth();
    for (i = 0; i < len; i += loopSqrth) {
        svbool_t pgB16 = svwhilelt_b16_s64(i, len);
        svbool_t pgB32 = svwhilelt_b32_s64(i, len);

        svfloat16_t srcX = svld1_f16(pgB16, src + i);
        svfloat32_t xEven = svcvt_f32_f16_x(pgB16, srcX);
        svfloat32_t xOdd = CvtOddF32F16(srcX, pgB16, pgB32);

        svfloat16_t ddX = svld1_f16(pgB16, dd + i);
        svfloat32_t ddEven = svcvt_f32_f16_x(pgB16, ddX);
        svfloat32_t ddOdd = CvtOddF32F16(ddX, pgB16, pgB32);

        /* res = 0.5 * dd / sqrt(src) */
        xEven = svsqrt_f32_x(pgB32, xEven);
        xOdd = svsqrt_f32_x(pgB32, xOdd);
        svfloat32_t resEven = svmul_f32_m(pgB32, ddEven, VectorComputation_FLOAT_HALF);
        svfloat32_t resOdd = svmul_f32_m(pgB32, ddOdd, VectorComputation_FLOAT_HALF);
        resEven = svdiv_f32_x(pgB32, resEven, xEven);
        resOdd = svdiv_f32_x(pgB32, resOdd, xOdd);

        svfloat16_t res = svcvt_f16_f32_x(pgB32, resEven); /* resEven(f16) */
        srcX = CvtOddF16F32(resOdd, pgB32); /* resOdd(f16) */
        res = svsel_f16(pgB32, res, srcX);

        svst1_f16(pgB16, ds + i, res);
    }
}
