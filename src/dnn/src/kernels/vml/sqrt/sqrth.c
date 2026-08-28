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
#ifndef KUDNNL_920G
#include "kernels/vml/common_macro.h"
#else
#include "arm_neon.h"
#define LOOP_N 4
#endif

void SqrtLaFWDF16(const int64_t len, const __fp16 *src, __fp16 *dst)
{
    int64_t i = 0;
#ifndef KUDNNL_920G
    int loopSqrth = svcnth();
    for (; i < len; i += loopSqrth) {
        svbool_t pg = svwhilelt_b16_s64(i, len);
        svfloat16_t x = svld1_f16(pg, src + i);
        svfloat16_t res = svsqrt_f16_x(pg, x);
        svst1_f16(pg, dst + i, res);
    }
#else
    for (; i <= len - 0x8 * LOOP_N; i += 0x8 * LOOP_N) {
        float16x8_t x0 = vld1q_f16(src + i + 0x00);
        float16x8_t x1 = vld1q_f16(src + i + 0x08);
        float16x8_t x2 = vld1q_f16(src + i + 0x10);
        float16x8_t x3 = vld1q_f16(src + i + 0x18);

        float16x8_t res0 = vsqrtq_f16(x0);
        float16x8_t res1 = vsqrtq_f16(x1);
        float16x8_t res2 = vsqrtq_f16(x2);
        float16x8_t res3 = vsqrtq_f16(x3);

        vst1q_f16(dst + i + 0x00, res0);
        vst1q_f16(dst + i + 0x08, res1);
        vst1q_f16(dst + i + 0x10, res2);
        vst1q_f16(dst + i + 0x18, res3);
    }
    for (; i < len; i++) {
        __fp16 x = src[i];
        __fp16 y;
        __asm__ volatile("fsqrt %h0, %h1 \n\t" : "=w"(y) : "w"(x));
        dst[i] = y;
    }
#endif
}

void SqrtLaVCBWDF16(const int64_t len, const __fp16 *src, const __fp16 *dd, __fp16 *ds)
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
        svfloat32_t resEven = svmul_f32_m(pgB32, ddEven, VC_FLOAT_HALF);
        svfloat32_t resOdd = svmul_f32_m(pgB32, ddOdd, VC_FLOAT_HALF);
        resEven = svdiv_f32_x(pgB32, resEven, xEven);
        resOdd = svdiv_f32_x(pgB32, resOdd, xOdd);

        svfloat16_t res = svcvt_f16_f32_x(pgB32, resEven); /* resEven(f16) */
        srcX = CvtOddF16F32(resOdd, pgB32); /* resOdd(f16) */
        res = svsel_f16(pgB32, res, srcX);

        svst1_f16(pgB16, ds + i, res);
    }
}
