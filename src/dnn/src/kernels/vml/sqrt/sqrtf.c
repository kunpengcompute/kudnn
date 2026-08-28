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

void SqrtLaFWDF32(const int64_t len, const float *src, float *dst)
{
    int64_t i = 0;
#ifndef KUDNNL_920G
    int64_t loopClipSqrtf = svcntw();
    for (; i <= len - loopClipSqrtf * VML_VC_UNFOLD_N; i += loopClipSqrtf * VML_VC_UNFOLD_N) {
        svbool_t pg = svptrue_b32();
        VML_VC_DECLARE(svfloat32_t, x);
        VML_VC_LD_STEP(x, svld1_f32, pg, src + i, loopClipSqrtf);
        VML_VC_DECLARE(svfloat32_t, res);
        VML_VC_FUNC_PG_1PARA(res, svsqrt_f32_x, pg, x);
        VML_VC_ST_STEP(svst1_f32, pg, dst + i, loopClipSqrtf, res);
    }
    for (; i < len; i += loopClipSqrtf) {
        svbool_t pg = svwhilelt_b32_s64(i, len);
        svfloat32_t x = svld1_f32(pg, src + i);
        svfloat32_t res = svsqrt_f32_x(pg, x);
        svst1_f32(pg, dst + i, res);
    }
#else
    for (; i <= len - 0x4 * LOOP_N; i += 0x4 * LOOP_N) {
        float32x4_t x0 = vld1q_f32(src + i + 0x00);
        float32x4_t x1 = vld1q_f32(src + i + 0x04);
        float32x4_t x2 = vld1q_f32(src + i + 0x08);
        float32x4_t x3 = vld1q_f32(src + i + 0x0C);

        float32x4_t res0 = vsqrtq_f32(x0);
        float32x4_t res1 = vsqrtq_f32(x1);
        float32x4_t res2 = vsqrtq_f32(x2);
        float32x4_t res3 = vsqrtq_f32(x3);

        vst1q_f32(dst + i + 0x00, res0);
        vst1q_f32(dst + i + 0x04, res1);
        vst1q_f32(dst + i + 0x08, res2);
        vst1q_f32(dst + i + 0x0C, res3);
    }
    for (; i < len; i++) {
        __asm__ volatile("fsqrt %s0, %s1 \n\t" : "=w"(dst[i]) : "w"(src[i]));
    }
#endif
}

void SqrtLaVCBWDF32(const int64_t len, const float *src, const float *dd, float *ds)
{
    int64_t i;
    int loopSqrtf = svcntw();
    for (i = 0; i < len; i += loopSqrtf) {
        svbool_t pg = svwhilelt_b32_s64(i, len);
        svfloat32_t srcX = svld1_f32(pg, src + i);
        svfloat32_t res = svld1_f32(pg, dd + i);
        srcX = svsqrt_f32_x(pg, srcX);
        res = svmul_f32_x(pg, res, VC_FLOAT_HALF);
        res = svdiv_f32_x(pg, res, srcX);

        svst1_f32(pg, ds + i, res);
    }
}
