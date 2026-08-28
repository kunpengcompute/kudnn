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
#include "kernels/vml/common_macro.h"
#define LOOP_N 4

void ClipFWDF32(const int64_t len, const float alpha, const float beta, const float *src, float *dst)
{
    int64_t i = 0;
#ifndef KUDNNL_920G
    int64_t loopClipf = svcntw();
    for (; i < len; i += loopClipf) {
        svbool_t pg = svwhilelt_b32_s64(i, len);

        svfloat32_t x = svld1_f32(pg, src + i);
        svfloat32_t maxV = svmaxnm_n_f32_x(pg, x, alpha);
        svfloat32_t res = svminnm_n_f32_x(pg, maxV, beta);

        svst1_f32(pg, dst + i, res);
    }
#else
    for (; i <= len - 0x4 * LOOP_N; i += 0x4 * LOOP_N) {
        float32x4_t x0 = vld1q_f32(src + i + 0x00);
        float32x4_t x1 = vld1q_f32(src + i + 0x04);
        float32x4_t x2 = vld1q_f32(src + i + 0x08);
        float32x4_t x3 = vld1q_f32(src + i + 0x0C);

        float32x4_t maxV0 = vmaxnmq_f32(x0, vdupq_n_f32(alpha));
        float32x4_t maxV1 = vmaxnmq_f32(x1, vdupq_n_f32(alpha));
        float32x4_t maxV2 = vmaxnmq_f32(x2, vdupq_n_f32(alpha));
        float32x4_t maxV3 = vmaxnmq_f32(x3, vdupq_n_f32(alpha));

        float32x4_t res0 = vminnmq_f32(maxV0, vdupq_n_f32(beta));
        float32x4_t res1 = vminnmq_f32(maxV1, vdupq_n_f32(beta));
        float32x4_t res2 = vminnmq_f32(maxV2, vdupq_n_f32(beta));
        float32x4_t res3 = vminnmq_f32(maxV3, vdupq_n_f32(beta));

        vst1q_f32(dst + i + 0x00, res0);
        vst1q_f32(dst + i + 0x04, res1);
        vst1q_f32(dst + i + 0x08, res2);
        vst1q_f32(dst + i + 0x0C, res3);
    }
    for (; i < len; i++) {
        float res;
        __asm__ volatile("fmaxnm %s0, %s1, %s2 \n\t" : "=w"(res) : "w"(src[i]), "w"(alpha));
        __asm__ volatile("fminnm %s0, %s1, %s2 \n\t" : "=w"(res) : "w"(res), "w"(beta));
        dst[i] = res;
    }
#endif
}

static void ClipMainKernelBwdF32(float alpha, float beta, const float *src, const float *dd, float *ds, int step)
{
    svbool_t pg = svptrue_b32();
    VML_VC_DECLARE(svfloat32_t, srcX);
    VML_VC_LD_STEP(srcX, svld1_f32, pg, src, step);
    VML_VC_DECLARE(svfloat32_t, res);
    VML_VC_LD_STEP(res, svld1_f32, pg, dd, step);

    VML_VC_DECLARE(svbool_t, pgRetZero);
    VML_VC_FUNC_PG_N2_2PARA(pgRetZero, svcmple_n_f32, pg, srcX, alpha);
    VML_VC_FUNC_NOPG_N3_3PARA(res, svdup_f32_m, res, pgRetZero, FLOAT_ZERO);
    VML_VC_FUNC_PG_N2_2PARA(pgRetZero, svcmpgt_n_f32, pg, srcX, beta);
    VML_VC_FUNC_NOPG_N3_3PARA(res, svdup_f32_m, res, pgRetZero, FLOAT_ZERO);

    VML_VC_ST_STEP(svst1_f32, pg, ds, step, res);
}

void ClipVCBWDF32(const int64_t len, const float alpha, const float beta, const float *src, const float *dd, float *ds)
{
    int64_t i;
    int64_t loopClipf = svcntw();

    for (i = 0; i <= len - loopClipf * VML_VC_UNFOLD_N; i += loopClipf * VML_VC_UNFOLD_N) {
        ClipMainKernelBwdF32(alpha, beta, src + i, dd + i, ds + i, loopClipf);
    }
    for (; i < len; i += loopClipf) {
        svbool_t pg = svwhilelt_b32_s64(i, len);
        svfloat32_t srcX = svld1_f32(pg, src + i);
        svfloat32_t res = svld1_f32(pg, dd + i);
        res = svdup_f32_m(res, svcmpgt_n_f32(pg, srcX, beta), FLOAT_ZERO);
        res = svdup_f32_m(res, svcmple_n_f32(pg, srcX, alpha), FLOAT_ZERO);
        svst1_f32(pg, ds + i, res);
    }
}