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

void NearbyintFWDF32(const int64_t len, const float *src, float *dst)
{
    int64_t i = 0;
#ifndef KUDNNL_920G
    int loopCnt = svcntw();
    for (; i < len; i += loopCnt) {
        svbool_t pg = svwhilelt_b32_s64(i, len);
        svfloat32_t x = svld1_f32(pg, src + i);
        svfloat32_t y = svrinti_f32_x(pg, x);
        svst1_f32(pg, dst + i, y);
    }
#else
    #define LOOP_N 4
    for (; i <= len - 0x4 * LOOP_N; i += 0x4 * LOOP_N) {
        float32x4_t x0 = vld1q_f32(src + i + 0x00);
        float32x4_t x1 = vld1q_f32(src + i + 0x04);
        float32x4_t x2 = vld1q_f32(src + i + 0x08);
        float32x4_t x3 = vld1q_f32(src + i + 0x0C);

        float32x4_t res0 = vrndiq_f32(x0);
        float32x4_t res1 = vrndiq_f32(x1);
        float32x4_t res2 = vrndiq_f32(x2);
        float32x4_t res3 = vrndiq_f32(x3);

        vst1q_f32(dst + i + 0x00, res0);
        vst1q_f32(dst + i + 0x04, res1);
        vst1q_f32(dst + i + 0x08, res2);
        vst1q_f32(dst + i + 0x0C, res3);
    }
    for (; i < len; i++) {
        __asm__ volatile("frinti %s0, %s1 \n\t" : "=w"(dst[i]) : "w"(src[i]));
    }
#endif
}
