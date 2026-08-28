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

void SqrLaFWDF32(const int64_t len, const float *src, float *dst)
{
    int64_t i = 0;
#ifndef KUDNNL_920G
    int loopCnt = svcntw();
    for (; i < len; i += loopCnt) {
        svbool_t pg = svwhilelt_b32_s64(i, len);
        svfloat32_t a = svld1_f32(pg, &src[i]);
        svfloat32_t c = svmul_f32_x(pg, a, a);
        svst1_f32(pg, &dst[i], c);
    }
#else
    for (; i <= len - 0x4 * LOOP_N; i += 0x4 * LOOP_N) {
        float32x4_t x0 = vld1q_f32(src + i + 0x00);
        float32x4_t x1 = vld1q_f32(src + i + 0x04);
        float32x4_t x2 = vld1q_f32(src + i + 0x08);
        float32x4_t x3 = vld1q_f32(src + i + 0x0C);

        float32x4_t res0 = vmulq_f32(x0, x0);
        float32x4_t res1 = vmulq_f32(x1, x1);
        float32x4_t res2 = vmulq_f32(x2, x2);
        float32x4_t res3 = vmulq_f32(x3, x3);

        vst1q_f32(dst + i + 0x00, res0);
        vst1q_f32(dst + i + 0x04, res1);
        vst1q_f32(dst + i + 0x08, res2);
        vst1q_f32(dst + i + 0x0C, res3);
    }
    for (; i < len; i++) {
        __asm__ volatile("fmul %s0, %s1, %s1 \n\t" : "=w"(dst[i]) : "w"(src[i]));
    }
#endif
}

void SqrLaBWDF32(const int64_t len, const float *src, const float *dd, float *ds)
{
    int64_t i = 0;
#ifndef KUDNNL_920G
    int loopCnt = svcntw();
    for (; i < len; i += loopCnt) {
        svbool_t pg = svwhilelt_b32_s64(i, len);
        svfloat32_t srcX = svld1_f32(pg, src + i);
        svfloat32_t res = svld1_f32(pg, dd + i);
        svfloat32_t x2 = svadd_f32_x(pg, srcX, srcX);
        res = svmul_f32_x(pg, res, x2);

        svst1_f32(pg, ds + i, res);
    }
#else
    for (; i <= len - 0x4 * LOOP_N; i += 0x4 * LOOP_N) {
        float32x4_t x0 = vld1q_f32(src + i + 0x00);
        float32x4_t x1 = vld1q_f32(src + i + 0x04);
        float32x4_t x2 = vld1q_f32(src + i + 0x08);
        float32x4_t x3 = vld1q_f32(src + i + 0x0C);

        float32x4_t d0 = vld1q_f32(dd + i + 0x00);
        float32x4_t d1 = vld1q_f32(dd + i + 0x04);
        float32x4_t d2 = vld1q_f32(dd + i + 0x08);
        float32x4_t d3 = vld1q_f32(dd + i + 0x0C);

        float32x4_t xT0 = vaddq_f32(x0, x0);
        float32x4_t xT1 = vaddq_f32(x1, x1);
        float32x4_t xT2 = vaddq_f32(x2, x2);
        float32x4_t xT3 = vaddq_f32(x3, x3);

        float32x4_t res0 = vmulq_f32(xT0, d0);
        float32x4_t res1 = vmulq_f32(xT1, d1);
        float32x4_t res2 = vmulq_f32(xT2, d2);
        float32x4_t res3 = vmulq_f32(xT3, d3);

        vst1q_f32(ds + i + 0x00, res0);
        vst1q_f32(ds + i + 0x04, res1);
        vst1q_f32(ds + i + 0x08, res2);
        vst1q_f32(ds + i + 0x0C, res3);
    }
    for (; i < len; i++) {
        ds[i] = dd[i] * (2.0f * src[i]);
    }
#endif
}
