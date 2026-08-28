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

#include "arm_neon.h"
#include "kernels/vml/vml_sve_base.h"

#define LOOP_N 4

void SqrLaFWDF16(const int64_t len, const __fp16 *src, __fp16 *dst)
{
    int64_t i = 0;
#ifndef KUDNNL_920G
    int loopSqrh = svcnth();
    for (; i < len; i += loopSqrh) {
        svbool_t pg = svwhilelt_b16_s64(i, len);
        svfloat16_t x = svld1_f16(pg, src + i);
        svfloat16_t res = svmul_f16_x(pg, x, x);
        svst1_f16(pg, dst + i, res);
    }
#else
    for (; i <= len - 0x8 * LOOP_N; i += 0x8 * LOOP_N) {
        float16x8_t x0 = vld1q_f16(src + i + 0x00);
        float16x8_t x1 = vld1q_f16(src + i + 0x08);
        float16x8_t x2 = vld1q_f16(src + i + 0x10);
        float16x8_t x3 = vld1q_f16(src + i + 0x18);

        float16x8_t res0 = vmulq_f16(x0, x0);
        float16x8_t res1 = vmulq_f16(x1, x1);
        float16x8_t res2 = vmulq_f16(x2, x2);
        float16x8_t res3 = vmulq_f16(x3, x3);

        vst1q_f16(dst + i + 0x00, res0);
        vst1q_f16(dst + i + 0x08, res1);
        vst1q_f16(dst + i + 0x10, res2);
        vst1q_f16(dst + i + 0x18, res3);
    }
    for (; i < len; i++) {
        __fp16 x = src[i];
        __fp16 y;
        __asm__ volatile("fmul %h0, %h1, %h1 \n\t" : "=w"(y) : "w"(x));
        dst[i] = y;
    }
#endif
}

void SqrLaVCBWDF16(const int64_t len, const __fp16 *src, const __fp16 *dd, __fp16 *ds)
{
    int64_t i = 0;
#ifndef KUDNNL_920G
    int loopSqrh = svcnth();
    for (; i < len; i += loopSqrh) {
        svbool_t pg = svwhilelt_b16_s64(i, len);
        svfloat16_t srcX = svld1_f16(pg, src + i);
        svfloat16_t res = svld1_f16(pg, dd + i);
        svfloat16_t x2 = svadd_f16_x(pg, srcX, srcX);
        res = svmul_f16_x(pg, res, x2);

        svst1_f16(pg, ds + i, res);
    }
#else
    for (; i <= len - 0x8 * LOOP_N; i += 0x8 * LOOP_N) {
        float16x8_t x0 = vld1q_f16(src + i + 0x00);
        float16x8_t x1 = vld1q_f16(src + i + 0x08);
        float16x8_t x2 = vld1q_f16(src + i + 0x10);
        float16x8_t x3 = vld1q_f16(src + i + 0x18);

        float16x8_t d0 = vld1q_f16(dd + i + 0x00);
        float16x8_t d1 = vld1q_f16(dd + i + 0x08);
        float16x8_t d2 = vld1q_f16(dd + i + 0x10);
        float16x8_t d3 = vld1q_f16(dd + i + 0x18);

        float16x8_t xT0 = vaddq_f16(x0, x0);
        float16x8_t xT1 = vaddq_f16(x1, x1);
        float16x8_t xT2 = vaddq_f16(x2, x2);
        float16x8_t xT3 = vaddq_f16(x3, x3);

        float16x8_t res0 = vmulq_f16(xT0, d0);
        float16x8_t res1 = vmulq_f16(xT1, d1);
        float16x8_t res2 = vmulq_f16(xT2, d2);
        float16x8_t res3 = vmulq_f16(xT3, d3);

        vst1q_f16(ds + i + 0x00, res0);
        vst1q_f16(ds + i + 0x08, res1);
        vst1q_f16(ds + i + 0x10, res2);
        vst1q_f16(ds + i + 0x18, res3);
    }
    for (; i < len; i++) {
        __fp16 x = src[i];
        __fp16 d = dd[i];
        __fp16 c2 = FLOAT_TWO;
        __fp16 y;
        __asm__ volatile("fmul %h0, %h1, %h2 \n\t" : "=w"(y) : "w"(x), "w"(c2));
        __asm__ volatile("fmul %h0, %h1, %h2 \n\t" : "=w"(y) : "w"(y), "w"(d));
        ds[i] = y;
    }
#endif
}
