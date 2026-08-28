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

void ClipFWDF16(const int64_t len, const float alpha, const float beta, const __fp16 *src, __fp16 *dst)
{
    int64_t i = 0;
    __fp16 ha = (__fp16)alpha;
    __fp16 hb = (__fp16)beta;
#ifndef KUDNNL_920G
    int64_t loopCliph = svcnth();
    for (; i < len; i += loopCliph) {
        svbool_t pg = svwhilelt_b16_s64(i, len);

        svfloat16_t x = svld1_f16(pg, src + i);
        svfloat16_t maxV = svmaxnm_n_f16_x(pg, x, ha);
        svfloat16_t res = svminnm_n_f16_x(pg, maxV, hb);

        svst1_f16(pg, dst + i, res);
    }
#else
    for (; i <= len - 0x8 * LOOP_N; i += 0x8 * LOOP_N) {
        float16x8_t x0 = vld1q_f16(src + i + 0x00);
        float16x8_t x1 = vld1q_f16(src + i + 0x08);
        float16x8_t x2 = vld1q_f16(src + i + 0x10);
        float16x8_t x3 = vld1q_f16(src + i + 0x18);

        float16x8_t maxV0 = vmaxnmq_f16(x0, vdupq_n_f16(ha));
        float16x8_t maxV1 = vmaxnmq_f16(x1, vdupq_n_f16(ha));
        float16x8_t maxV2 = vmaxnmq_f16(x2, vdupq_n_f16(ha));
        float16x8_t maxV3 = vmaxnmq_f16(x3, vdupq_n_f16(ha));

        float16x8_t res0 = vminnmq_f16(maxV0, vdupq_n_f16(hb));
        float16x8_t res1 = vminnmq_f16(maxV1, vdupq_n_f16(hb));
        float16x8_t res2 = vminnmq_f16(maxV2, vdupq_n_f16(hb));
        float16x8_t res3 = vminnmq_f16(maxV3, vdupq_n_f16(hb));

        vst1q_f16(dst + i + 0x00, res0);
        vst1q_f16(dst + i + 0x08, res1);
        vst1q_f16(dst + i + 0x10, res2);
        vst1q_f16(dst + i + 0x18, res3);
    }
    for (; i < len; i++) {
        __fp16 x = src[i];
        __fp16 res, maxV;
        __asm__ volatile("fmaxnm %h0, %h1, %h2 \n\t" : "=w"(maxV) : "w"(x), "w"(ha));
        __asm__ volatile("fminnm %h0, %h1, %h2 \n\t" : "=w"(res) : "w"(maxV), "w"(hb));
        dst[i] = res;
    }

#endif
}

#undef VML_VC_UNFOLD_N
#define VML_VC_UNFOLD_N 6

static void ClipMainKernelBwdF16(svfloat32_t va, svfloat32_t vb,
                                 const __fp16 *src, const __fp16 *dd, __fp16 *ds, int step)
{
    svbool_t pgB32 = svptrue_b32();
    svbool_t pgB16 = svptrue_b16();

    VML_VC_DECLARE(svfloat32_t, xEven);
    VML_VC_DECLARE(svfloat32_t, xOdd);
    VML_VC_DECLARE(svfloat32_t, ddEven);
    VML_VC_DECLARE(svfloat32_t, ddOdd);
    // clip compute
    VML_VC_DECLARE(svfloat32_t, resEven);
    VML_VC_DECLARE(svfloat32_t, resOdd);
    VML_VC_DECLARE(svbool_t, pgRetZero);

    LOAD_CONVERT_FP16_TO_FP32_2(src, dd, pgB16, xEven, xOdd, ddEven, ddOdd, step);

    VML_VC_FUNC_PG_N2_2PARA(pgRetZero, svcmple_f32, pgB32, xEven, va);
    VML_VC_FUNC_NOPG_N3_3PARA(resEven, svdup_f32_m, ddEven, pgRetZero, FLOAT_ZERO);
    VML_VC_FUNC_PG_N2_2PARA(pgRetZero, svcmpgt_f32, pgB32, xEven, vb);
    VML_VC_FUNC_NOPG_N3_3PARA(resEven, svdup_f32_m, resEven, pgRetZero, FLOAT_ZERO);

    VML_VC_FUNC_PG_N2_2PARA(pgRetZero, svcmple_f32, pgB32, xOdd, va);
    VML_VC_FUNC_NOPG_N3_3PARA(resOdd, svdup_f32_m, ddOdd, pgRetZero, FLOAT_ZERO);
    VML_VC_FUNC_PG_N2_2PARA(pgRetZero, svcmpgt_f32, pgB32, xOdd, vb);
    VML_VC_FUNC_NOPG_N3_3PARA(resOdd, svdup_f32_m, resOdd, pgRetZero, FLOAT_ZERO);

    CONVERT_STORE_FP32_TO_FP16(ds, pgB16, resEven, resOdd, step);
}

static void ClipVCBWDF16ConvertF32(const int64_t len, const float alpha, const float beta,
    const __fp16 *src, const __fp16 *dd, __fp16 *ds)
{
    int64_t i = 0;
    int64_t loopCliph = svcnth();
    svfloat32_t va = svdup_n_f32(alpha);
    svfloat32_t vb = svdup_n_f32(beta);

    for (; i <= len - loopCliph * VML_VC_UNFOLD_N; i += loopCliph * VML_VC_UNFOLD_N) {
        ClipMainKernelBwdF16(va, vb, src + i, dd + i, ds + i, loopCliph);
    }
    for (; i < len; i += loopCliph) {
        svbool_t pgB16 = svwhilelt_b16_s64(i, len);
        svbool_t pgB32 = svwhilelt_b32_s64(i, len);

        svfloat16_t x = svld1_f16(pgB16, src + i);
        svfloat32_t xEven = svcvt_f32_f16_x(pgB16, x);
        svfloat32_t xOdd = CvtOddF32F16(x, pgB16, pgB32);

        svfloat16_t ddX = svld1_f16(pgB16, dd + i);
        svfloat32_t ddEven = svcvt_f32_f16_x(pgB16, ddX);
        svfloat32_t ddOdd = CvtOddF32F16(ddX, pgB16, pgB32);

        // clip compute
        svfloat32_t resEven = svdup_f32_m(ddEven, svcmpgt_f32(pgB32, xEven, vb), FLOAT_ZERO);
        resEven = svdup_f32_m(resEven, svcmple_f32(pgB32, xEven, va), FLOAT_ZERO);

        svfloat32_t resOdd = svdup_f32_m(ddOdd, svcmpgt_f32(pgB32, xOdd, vb), FLOAT_ZERO);
        resOdd = svdup_f32_m(resOdd, svcmple_f32(pgB32, xOdd, va), FLOAT_ZERO);

        // convert back to fp16
        svfloat16_t res = svcvt_f16_f32_x(pgB32, resEven); /* resEven(f16) */
        x = CvtOddF16F32(resOdd, pgB32); /* resOdd(f16) */
        res = svsel_f16(pgB32, res, x);

        svst1_f16(pgB16, ds + i, res);
    }
}

static void ClipVCBWDF16Pure(const int64_t len, const __fp16 aHi, const __fp16 aLo,
    const __fp16 bHi, const __fp16 bLo, const __fp16 *src, const __fp16 *dd, __fp16 *ds)
{
    int64_t i = 0;
    int64_t loopCliph = svcnth();

    for (; i < len; i += loopCliph) {
        svbool_t pg = svwhilelt_b16_s64(i, len);
        svfloat16_t srcX = svld1_f16(pg, src + i);

        svfloat16_t left  = svsub_n_f16_x(pg, srcX, aHi);
        svfloat16_t right = svsub_n_f16_x(pg, srcX, bHi);

        svfloat16_t res = svld1_f16(pg, dd + i);
        res = svdup_f16_m(res, svcmple_n_f16(pg, left, aLo), FLOAT_ZERO);
        res = svdup_f16_m(res, svcmpgt_n_f16(pg, right, bLo), FLOAT_ZERO);

        svst1_f16(pg, ds + i, res);
    }
}

void ClipVCBWDF16(const int64_t len, const float alpha, const float beta,
    const __fp16 *src, const __fp16 *dd, __fp16 *ds)
{
    __fp16 aHi = (__fp16)alpha;
    __fp16 aLo = (__fp16)(alpha - (float)aHi);
    __fp16 bHi = (__fp16)beta;
    __fp16 bLo = (__fp16)(beta - (float)bHi);
    float fa = (float)aHi + (float)aLo;
    float fb = (float)bHi + (float)bLo;
    if (fa == alpha && fb == beta) {
        ClipVCBWDF16Pure(len, aHi, aLo, bHi, bLo, src, dd, ds);
    } else {
        ClipVCBWDF16ConvertF32(len, alpha, beta, src, dd, ds);
    }
}