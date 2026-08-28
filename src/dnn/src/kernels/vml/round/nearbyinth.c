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

void NearbyintFWDF16(const int64_t len, const __fp16 *src, __fp16 *dst)
{
    int64_t i = 0;
#ifndef KUDNNL_920G
    int loopNearbyInth = svcnth();
    for (; i < len; i += loopNearbyInth) {
        svbool_t pg = svwhilelt_b16_s64(i, len);
        svfloat16_t x = svld1_f16(pg, src + i);
        svfloat16_t res = svrinti_f16_x(pg, x);
        svst1_f16(pg, dst + i, res);
    }
#else
    #define LOOP_N 4
    for (; i <= len - 0x8 * LOOP_N; i += 0x8 * LOOP_N) {
        float16x8_t x0 = vld1q_f16(src + i + 0x00);
        float16x8_t x1 = vld1q_f16(src + i + 0x08);
        float16x8_t x2 = vld1q_f16(src + i + 0x10);
        float16x8_t x3 = vld1q_f16(src + i + 0x18);

        float16x8_t res0 = vrndiq_f16(x0);
        float16x8_t res1 = vrndiq_f16(x1);
        float16x8_t res2 = vrndiq_f16(x2);
        float16x8_t res3 = vrndiq_f16(x3);

        vst1q_f16(dst + i + 0x00, res0);
        vst1q_f16(dst + i + 0x08, res1);
        vst1q_f16(dst + i + 0x10, res2);
        vst1q_f16(dst + i + 0x18, res3);
    }
    for (; i < len; i++) {
        __fp16 x = src[i];
        __fp16 y;
        __asm__ volatile("frinti %h0, %h1 \n\t" : "=w"(y) : "w"(x));
        dst[i] = y;
    }
#endif
}
