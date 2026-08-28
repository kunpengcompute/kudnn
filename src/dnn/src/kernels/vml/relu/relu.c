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
#include "kernels/vml/common_macro.h"

static void ReluMainKernelFwdF32(svfloat32_t a, const float *src, float *dst, int step)
{
    svbool_t pg = svptrue_b32();
    VML_VC_DECLARE(svfloat32_t, x);
    VML_VC_DECLARE(svfloat32_t, res);
    VML_VC_DECLARE(svbool_t, mask_lez);

    VML_VC_LD_STEP(x, svld1_f32, pg, src, step);
    VML_VC_FUNC_PG_N2_2PARA(mask_lez, svcmple_n_f32, pg, x, 0.0f);

    VML_VC_FUNC_UPG_N2_2PARA(res, svmul_f32_m, mask_lez, x, a);

    VML_VC_ST_STEP(svst1_f32, pg, dst, step, res);
}

void ReluVCFWDF32(const int64_t len, const float alpha, const float *src, float *dst)
{
    int64_t i;
    int64_t loopCnt = svcntw();
    svfloat32_t va = svdup_n_f32(alpha);

    for (i = 0; i <= len - loopCnt * VML_VC_UNFOLD_N; i += loopCnt * VML_VC_UNFOLD_N) {
        ReluMainKernelFwdF32(va, src + i, dst + i, loopCnt);
    }
    for (; i < len; i += loopCnt) {
        svbool_t pg = svwhilelt_b32_s64(i, len);
        svfloat32_t x = svld1_f32(pg, src + i);

        svbool_t maskLez = svcmple_n_f32(pg, x, 0.0f);
        svfloat32_t res = svmul_f32_m(maskLez, x, va);

        svst1_f32(pg, dst + i, res);
    }
}

static void ReluMainKernelBwdF32(svfloat32_t a, const float *src, const float *dd, float *ds, int step)
{
    svbool_t pg = svptrue_b32();
    VML_VC_DECLARE(svfloat32_t, src_x);
    VML_VC_DECLARE(svfloat32_t, dd_x);
    VML_VC_DECLARE(svfloat32_t, res);
    VML_VC_DECLARE(svbool_t, mask_lez);

    VML_VC_LD_STEP(src_x, svld1_f32, pg, src, step);
    VML_VC_FUNC_PG_N2_2PARA(mask_lez, svcmple_n_f32, pg, src_x, 0.0f);

    VML_VC_LD_STEP(dd_x, svld1_f32, pg, dd, step);

    VML_VC_FUNC_UPG_N2_2PARA(res, svmul_f32_m, mask_lez, dd_x, a);

    VML_VC_ST_STEP(svst1_f32, pg, ds, step, res);
}

void ReluVCBWDF32(const int64_t len, const float alpha, const float *src, const float *dd, float *ds)
{
    int64_t i;
    int64_t loopCnt = svcntw();
    svfloat32_t va = svdup_n_f32(alpha);

    for (i = 0; i <= len - loopCnt * VML_VC_UNFOLD_N; i += loopCnt * VML_VC_UNFOLD_N) {
        ReluMainKernelBwdF32(va, src + i, dd + i, ds + i, loopCnt);
    }
    for (; i < len; i += loopCnt) {
        svbool_t pg = svwhilelt_b32_s64(i, len);
        svfloat32_t srcX = svld1_f32(pg, src + i);
        svfloat32_t ddX = svld1_f32(pg, dd + i);

        svbool_t maskLez = svcmple_n_f32(pg, srcX, 0.0f);
        svfloat32_t res = svmul_f32_m(maskLez, ddX, va);

        svst1_f32(pg, ds + i, res);
    }
}

static void ReluMainKernelFwdF16(svfloat32_t a, const __fp16 *src, __fp16 *dst, int step)
{
    svbool_t pgB16 = svptrue_b16();
    svbool_t pgB32 = svptrue_b32();

    VML_VC_DECLARE(svfloat32_t, xEven);
    VML_VC_DECLARE(svfloat32_t, xOdd);

    LOAD_CONVERT_FP16_TO_FP32_1(src, pgB16, xEven, xOdd, step);

    // relu compute
    VML_VC_DECLARE(svfloat32_t, resEven);
    VML_VC_DECLARE(svfloat32_t, resOdd);
    VML_VC_DECLARE(svbool_t, maskLezEven);
    VML_VC_DECLARE(svbool_t, maskLezOdd);
    VML_VC_FUNC_PG_N2_2PARA(maskLezEven, svcmple_n_f32, pgB32, xEven, FLOAT_ZERO);
    VML_VC_FUNC_UPG_N2_2PARA(resEven, svmul_f32_m, maskLezEven, xEven, a);

    VML_VC_FUNC_PG_N2_2PARA(maskLezOdd, svcmple_n_f32, pgB32, xOdd, FLOAT_ZERO);
    VML_VC_FUNC_UPG_N2_2PARA(resOdd, svmul_f32_m, maskLezOdd, xOdd, a);

    CONVERT_STORE_FP32_TO_FP16(dst, pgB16, resEven, resOdd, step);
}

void ReluVCFWDF16(const int64_t len, const float alpha, const __fp16 *src, __fp16 *dst)
{
    int64_t i;
    int64_t loopCnt = svcnth();
    svfloat32_t va = svdup_n_f32(alpha);

    for (i = 0; i <= len - loopCnt * VML_VC_UNFOLD_N; i += loopCnt * VML_VC_UNFOLD_N) {
        ReluMainKernelFwdF16(va, src + i, dst + i, loopCnt);
    }
    for (; i < len; i += loopCnt) {
        svbool_t pgB16 = svwhilelt_b16_s64(i, len);
        svbool_t pgB32 = svwhilelt_b32_s64(i, len);

        svfloat16_t x = svld1_f16(pgB16, src + i);

        svfloat32_t xEven = svcvt_f32_f16_x(pgB16, x);
        svfloat32_t xOdd = CvtOddF32F16(x, pgB16, pgB32);

        svbool_t maskLezEven = svcmple_n_f32(pgB32, xEven, 0.0f);
        svfloat32_t resEven = svmul_f32_m(maskLezEven, xEven, va);

        svbool_t maskLezOdd = svcmple_n_f32(pgB32, xOdd, 0.0f);
        svfloat32_t resOdd = svmul_f32_m(maskLezOdd, xOdd, va);

        svfloat16_t res = svcvt_f16_f32_x(pgB32, resEven);
        x = CvtOddF16F32(resOdd, pgB32);
        res = svadd_f16_x(pgB16, res, x);

        svst1_f16(pgB16, dst + i, res);
    }
}

static void ReluMainKernelBwdF16(svfloat32_t a, const __fp16 *src, const __fp16 *dd, __fp16 *ds, int step)
{
    svbool_t pgB32 = svptrue_b32();
    svbool_t pgB16 = svptrue_b16();

    VML_VC_DECLARE(svfloat32_t, xEven);
    VML_VC_DECLARE(svfloat32_t, xOdd);
    VML_VC_DECLARE(svfloat32_t, ddEven);
    VML_VC_DECLARE(svfloat32_t, ddOdd);

    LOAD_CONVERT_FP16_TO_FP32_2(src, dd, pgB16, xEven, xOdd, ddEven, ddOdd, step);

    // relu bwd compute
    VML_VC_DECLARE(svbool_t, maskLezEven);
    VML_VC_DECLARE(svfloat32_t, resEven);
    VML_VC_FUNC_PG_N2_2PARA(maskLezEven, svcmple_n_f32, pgB32, xEven, FLOAT_ZERO);
    VML_VC_FUNC_UPG_N2_2PARA(resEven, svmul_f32_m, maskLezEven, ddEven, a);

    VML_VC_DECLARE(svbool_t, maskLezOdd);
    VML_VC_DECLARE(svfloat32_t, resOdd);
    VML_VC_FUNC_PG_N2_2PARA(maskLezOdd, svcmple_n_f32, pgB32, xOdd, FLOAT_ZERO);
    VML_VC_FUNC_UPG_N2_2PARA(resOdd, svmul_f32_m, maskLezOdd, ddOdd, a);

    CONVERT_STORE_FP32_TO_FP16(ds, pgB16, resEven, resOdd, step);
}

void ReluVCBWDF16(const int64_t len, const float alpha, const __fp16 *src, const __fp16 *dd, __fp16 *ds)
{
    int64_t i;
    int64_t loopCnt = svcnth();
    svfloat32_t va = svdup_n_f32(alpha);

    for (i = 0; i <= len - loopCnt * VML_VC_UNFOLD_N; i += loopCnt * VML_VC_UNFOLD_N) {
        ReluMainKernelBwdF16(va, src + i, dd + i, ds + i, loopCnt);
    }
    for (; i < len; i += loopCnt) {
        svbool_t pgB16 = svwhilelt_b16_s64(i, len);
        svbool_t pgB32 = svwhilelt_b32_s64(i, len);

        svfloat16_t srcX = svld1_f16(pgB16, src + i);
        svfloat16_t ddX = svld1_f16(pgB16, dd + i);

        svfloat32_t srcEven = svcvt_f32_f16_x(pgB16, srcX);
        svfloat32_t srcOdd = CvtOddF32F16(srcX, pgB16, pgB32);

        svfloat32_t ddEven = svcvt_f32_f16_x(pgB16, ddX);
        svfloat32_t ddOdd = CvtOddF32F16(ddX, pgB16, pgB32);

        svbool_t maskLezEven = svcmple_n_f32(pgB32, srcEven, 0.0f);
        svbool_t maskLezOdd = svcmple_n_f32(pgB32, srcOdd, 0.0f);

        svfloat32_t resEven = svmul_f32_m(maskLezEven, ddEven, va);
        svfloat32_t resOdd = svmul_f32_m(maskLezOdd, ddOdd, va);

        svfloat16_t res = svcvt_f16_f32_x(pgB32, resEven);
        srcX = CvtOddF16F32(resOdd, pgB32);
        res = svadd_f16_x(pgB16, res, srcX);

        svst1_f16(pgB16, ds + i, res);
    }
}

static void ReluMainKernelFwdBF16(svfloat32_t a, const __bf16 *src, __bf16 *dst, int step)
{
    svbool_t pgB16 = svptrue_b16();
    svbool_t pgB32 = svptrue_b32();

    VML_VC_DECLARE(svfloat32_t, xBot);
    VML_VC_DECLARE(svfloat32_t, xTop);

    LOAD_CONVERT_BF16_TO_FP32_1(src, pgB16, xBot, xTop, step);

    // relu compute
    VML_VC_DECLARE(svfloat32_t, resBot);
    VML_VC_DECLARE(svfloat32_t, resTop);
    VML_VC_DECLARE(svbool_t, maskLezBot);
    VML_VC_DECLARE(svbool_t, maskLezTop);
    VML_VC_FUNC_PG_N2_2PARA(maskLezBot, svcmple_n_f32, pgB32, xBot, 0.0f);
    VML_VC_FUNC_UPG_N2_2PARA(resBot, svmul_f32_m, maskLezBot, xBot, a);

    VML_VC_FUNC_PG_N2_2PARA(maskLezTop, svcmple_n_f32, pgB32, xTop, 0.0f);
    VML_VC_FUNC_UPG_N2_2PARA(resTop, svmul_f32_m, maskLezTop, xTop, a);

    VML_VC_FUNC_PG_1PARA(resBot, SvF32CvtBf16InfSpecial, pgB32, resBot);
    VML_VC_FUNC_PG_1PARA(resTop, SvF32CvtBf16InfSpecial, pgB32, resTop);

    CONVERT_STORE_FP32_TO_BF16(dst, pgB32, pgB16, resBot, resTop, step);
}

void ReluVCFWDBF16(const int64_t len, const float alpha, const __bf16 *src, __bf16 *dst)
{
    int64_t i;
    int64_t loopCnt = svcnth();
    int numF32 = svcntw();
    svfloat32_t va = svdup_n_f32(alpha);

    for (i = 0; i <= len - loopCnt * VML_VC_UNFOLD_N; i += loopCnt * VML_VC_UNFOLD_N) {
        ReluMainKernelFwdBF16(va, src + i, dst + i, loopCnt);
    }

    for (; i < len; i += loopCnt) {
        svbool_t pgB16 = svwhilelt_b16_s64(i, len);
        svbool_t pgB32 = svwhilelt_b32_s64(i, len);
        svbool_t pgF32Bot = pgB32;
        svbool_t pgF32Top = svwhilelt_b32_s64(i + numF32, len);

        svbfloat16_t x = svld1_bf16(pgB16, src + i);

        // convert to fp32
        svfloat32_t xBot = SvCvtF32BF16BottomX(x);
        svfloat32_t xTop = SvCvtF32BF16TopX(x);

        // relu compute
        svbool_t maskLezBot = svcmple_n_f32(pgB32, xBot, 0.0f);
        svfloat32_t resBot = svmul_f32_m(maskLezBot, xBot, va);

        svbool_t maskLezTop = svcmple_n_f32(pgB32, xTop, 0.0f);
        svfloat32_t resTop = svmul_f32_m(maskLezTop, xTop, va);

        resBot = SvF32CvtBf16InfSpecial(pgB32, resBot);
        resTop = SvF32CvtBf16InfSpecial(pgB32, resTop);

        // convert back to bf16
        svbfloat16_t res = SvCvtBF16F32X(pgF32Bot, resBot, pgF32Top, resTop);
        svst1_bf16(pgB16, dst + i, res);
    }
}

static void ReluMainKernelBwdBF16(svfloat32_t a, const __bf16 *src, const __bf16 *dd, __bf16 *ds, int step)
{
    svbool_t pgB16 = svptrue_b16();
    svbool_t pgB32 = svptrue_b32();

    VML_VC_DECLARE(svfloat32_t, xBot);
    VML_VC_DECLARE(svfloat32_t, xTop);
    VML_VC_DECLARE(svfloat32_t, ddBot);
    VML_VC_DECLARE(svfloat32_t, ddTop);

    LOAD_CONVERT_BF16_TO_FP32_2(src, dd, pgB16, xBot, xTop, ddBot, ddTop, step);

    // relu bwd compute

    VML_VC_DECLARE(svbool_t, maskLezBot);
    VML_VC_DECLARE(svfloat32_t, resBot);
    VML_VC_FUNC_PG_N2_2PARA(maskLezBot, svcmple_n_f32, pgB32, xBot, FLOAT_ZERO);
    VML_VC_FUNC_UPG_N2_2PARA(resBot, svmul_f32_m, maskLezBot, ddBot, a);

    VML_VC_DECLARE(svbool_t, maskLezTop);
    VML_VC_DECLARE(svfloat32_t, resTop);
    VML_VC_FUNC_PG_N2_2PARA(maskLezTop, svcmple_n_f32, pgB32, xTop, FLOAT_ZERO);
    VML_VC_FUNC_UPG_N2_2PARA(resTop, svmul_f32_m, maskLezTop, ddTop, a);

    CONVERT_STORE_FP32_TO_BF16(ds, pgB32, pgB16, resBot, resTop, step);
}

void ReluVCBWDBF16(const int64_t len, const float alpha, const __bf16 *src, const __bf16 *dd, __bf16 *ds)
{
    int64_t i;
    int64_t loopCnt = svcnth();
    svfloat32_t va = svdup_n_f32(alpha);

    for (i = 0; i <= len - loopCnt * VML_VC_UNFOLD_N; i += loopCnt * VML_VC_UNFOLD_N) {
        ReluMainKernelBwdBF16(va, src + i, dd + i, ds + i, loopCnt);
    }

    for (; i < len; i += loopCnt) {
        svbool_t pgB16 = svwhilelt_b16_s64(i, len);
        svbool_t pgB32 = svwhilelt_b32_s64(i, len);

        svbfloat16_t srcX = svld1_bf16(pgB16, src + i);
        svbfloat16_t ddX = svld1_bf16(pgB16, dd + i);

        // convert src to fp32
        svfloat32_t srcBot = SvCvtF32BF16BottomX(srcX);
        svfloat32_t srcTop = SvCvtF32BF16TopX(srcX);

        // convert dd to fp32
        svfloat32_t ddBot = SvCvtF32BF16BottomX(ddX);
        svfloat32_t ddTop = SvCvtF32BF16TopX(ddX);

        // relu bwd compute
        svbool_t maskLezBot = svcmple_n_f32(pgB32, srcBot, 0.0f);
        svbool_t maskLezOdd = svcmple_n_f32(pgB32, srcTop, 0.0f);

        svfloat32_t resBot = svmul_f32_m(maskLezBot, ddBot, va);
        svfloat32_t resTop = svmul_f32_m(maskLezOdd, ddTop, va);

        resBot = SvF32CvtBf16InfSpecial(pgB32, resBot);
        resTop = SvF32CvtBf16InfSpecial(pgB32, resTop);

        // convert back to bf16
        svbfloat16_t res = SvCvtBF16F32FullX(resBot, resTop);
        svst1_bf16(pgB16, ds + i, res);
    }
}

void ReluVCFWDS32(const int64_t len, const float alpha, const int32_t *src, int32_t *dst)
{
    int64_t i;
    int64_t loopCnt = svcntw();
    svfloat32_t va = svdup_n_f32(alpha);

    for (i = 0; i < len; i += loopCnt) {
        svbool_t pg = svwhilelt_b32_s64(i, len);
        svint32_t x = svld1_s32(pg, src + i);

        svfloat32_t xF32 = svcvt_f32_s32_x(pg, x);

        svbool_t maskLez = svcmple_f32(pg, xF32, VC_FLOAT_ZERO);
        svfloat32_t res = svmul_f32_m(maskLez, xF32, va);

        res = svadd_f32_x(pg, res, VC_FLOAT_HALF);
        res = svrintm_f32_x(pg, res);
        svint32_t resS32 = svcvt_s32_f32_x(pg, res);

        svst1_s32(pg, dst + i, resS32);
    }
}

void ReluVCFWDS8(const int64_t len, const float alpha, const int8_t *src, int8_t *dst)
{
    for (int64_t i = 0; i < len; i++) {
        float val = (float)src[i];
        val = (val <= 0.f ? alpha : 1.f) * val;
        dst[i] = Float2Int8(val);
    }
}

void ReluVCFWDU8(const int64_t len, const float alpha, const uint8_t *src, uint8_t *dst)
{
    for (int64_t i = 0; i < len; i++) {
        float val = (float)src[i];
        val = (val <= 0.f ? alpha : 1.f) * val;
        dst[i] = Float2Uint8(val);
    }
}