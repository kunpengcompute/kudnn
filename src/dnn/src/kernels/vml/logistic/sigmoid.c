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

#include "kernels/vml/vml_sve_la_sigmoid.h"
#include "kernels/vml/common_macro.h"
#ifdef __clang__
__attribute__((optnone))
#endif
static void SigmoidMainKernelF32(const float *src, float *dst, int step, const float *diffDst)
{
    VML_VC_DECLARE(svfloat32_t, x);
    svbool_t pg = svptrue_b32();

    VML_VC_LD_STEP(x, svld1_f32, pg, src, step);
    VML_VC_FUNC_PG_1PARA(x, svneg_f32_x, pg, x);

    VML_VC_DECLARE(svfloat32_t, fwd_res);
    VML_VC_EXP(pg, x, fwd_res);
    VML_VC_FUNC_PG_N1_2PARA(fwd_res, svadd_f32_x, pg, VML_VC_F32_ONE, fwd_res);
    VML_VC_FUNC_PG_N1_2PARA(fwd_res, svdiv_f32_x, pg, VML_VC_F32_ONE, fwd_res);

    if (!diffDst) { // forward computation has ended
        VML_VC_ST_STEP(svst1_f32, pg, dst, step, fwd_res);
    } else {
        VML_VC_DECLARE(svfloat32_t, dd);
        VML_VC_LD_STEP(dd, svld1_f32, pg, diffDst, step);
        VML_VC_DECLARE(svfloat32_t, res);
        VML_VC_FUNC_PG_2PARA(res, svmul_f32_x, pg, dd, fwd_res);

        VML_VC_DECLARE(svfloat32_t, one_m_fwd);
        VML_VC_FUNC_PG_N1_2PARA(one_m_fwd, svsub_f32_x, pg, VC_FLOAT_ONE, fwd_res);

        VML_VC_FUNC_PG_2PARA(res, svmul_f32_x, pg, res, one_m_fwd);

        VML_VC_ST_STEP(svst1_f32, pg, dst, step, res);
    }
}

void SigmoidVCFWDF32(const int64_t len, const float *src, float *dst)
{
    int64_t i;
    int64_t loopSigmoidf = svcntw();
    for (i = 0; i <= len - loopSigmoidf * VML_VC_UNFOLD_N; i += loopSigmoidf * VML_VC_UNFOLD_N) {
        SigmoidMainKernelF32(src + i, dst + i, loopSigmoidf, NULL);
    }

    for (; i < len; i += loopSigmoidf) {
        svbool_t pg = svwhilelt_b32_s64(i, len);
        svfloat32_t x = svld1_f32(pg, src + i);
        x = svmul_f32_x(pg, x, svdup_n_f32(-1.0f));

        svfloat32_t y = SvSigmoidF32X(x, pg);
        svst1_f32(pg, dst + i, y);
    }
}

void SigmoidVCBWDF32(const int64_t len, const float *src, const float *diffDst, float *diffSrc)
{
    int64_t i;
    int64_t loopSigmoidf = svcntw();
    for (i = 0; i <= len - loopSigmoidf * VML_VC_UNFOLD_N; i += loopSigmoidf * VML_VC_UNFOLD_N) {
        SigmoidMainKernelF32(src + i, diffSrc + i, loopSigmoidf, diffDst + i);
    }

    for (; i < len; i += loopSigmoidf) {
        svbool_t pg = svwhilelt_b32_s64(i, len);
        svfloat32_t x = svld1_f32(pg, src + i);
        x = svmul_f32_x(pg, x, svdup_n_f32(-1.0f));

        svfloat32_t yFwd = SvSigmoidF32X(x, pg);

        svfloat32_t dd = svld1_f32(pg, diffDst + i);
        svfloat32_t res = svmul_f32_x(pg, dd, yFwd);
        svfloat32_t oneMY = svsub_f32_x(pg, svdup_n_f32(1.0f), yFwd);
        res = svmul_f32_x(pg, res, oneMY);

        svst1_f32(pg, diffSrc + i, res);
    }
}
#ifdef __clang__
__attribute__((optnone))
#endif
static void SigmoidMainKernelF16(const __fp16 *src, __fp16 *dst, int step, const __fp16 *diffDst)
{
    VML_VC_DECLARE(svfloat16_t, x);
    VML_VC_DECLARE(svfloat16_t, fwd_res);
    VML_VC_DECLARE(svfloat32_t, x_even);
    VML_VC_DECLARE(svfloat32_t, x_odd);
    svbool_t pgB16 = svptrue_b16();
    svbool_t pgB32 = svptrue_b32();

    VML_VC_LD_STEP(x, svld1_f16, pgB16, src, step);

    // even values convert
    VML_VC_FUNC_PG_1PARA(x_even, svcvt_f32_f16_x, pgB16, x);

    // right shift for odd values convert
    VML_VC_DECLARE(svuint32_t, shifted_src);
    VML_VC_FUNC_NOPG_1PARA(shifted_src, svreinterpret_u32_f16, x);
    VML_VC_FUNC_PG_N2_2PARA(shifted_src, svlsr_n_u32_x, pgB32, shifted_src, F16_SHIFT);
    VML_VC_FUNC_NOPG_1PARA(x, svreinterpret_f16_u32, shifted_src);
    VML_VC_FUNC_PG_1PARA(x_odd, svcvt_f32_f16_x, pgB16, x);

    // ========================

    VML_VC_FUNC_PG_1PARA(x_even, svneg_f32_x, pgB32, x_even);
    VML_VC_DECLARE(svfloat32_t, fwd_res_even);
    VML_VC_EXP(pgB32, x_even, fwd_res_even);
    VML_VC_FUNC_PG_N1_2PARA(fwd_res_even, svadd_f32_x, pgB32, VML_VC_F32_ONE, fwd_res_even);
    VML_VC_FUNC_PG_N1_2PARA(fwd_res_even, svdiv_f32_x, pgB32, VML_VC_F32_ONE, fwd_res_even);

    // ========================

    VML_VC_FUNC_PG_1PARA(x_odd, svneg_f32_x, pgB32, x_odd);
    VML_VC_DECLARE(svfloat32_t, fwd_res_odd);
    VML_VC_EXP(pgB32, x_odd, fwd_res_odd);
    VML_VC_FUNC_PG_N1_2PARA(fwd_res_odd, svadd_f32_x, pgB32, VML_VC_F32_ONE, fwd_res_odd);
    VML_VC_FUNC_PG_N1_2PARA(fwd_res_odd, svdiv_f32_x, pgB32, VML_VC_F32_ONE, fwd_res_odd);

    // ========================

    if (!diffDst) { // forward computation has ended
        // even values convert back
        VML_VC_FUNC_PG_1PARA(fwd_res, svcvt_f16_f32_x, pgB32, fwd_res_even);

        // odd values convert back (reuse of shifted_src registers)
        VML_VC_FUNC_PG_1PARA(x, svcvt_f16_f32_x, pgB32, fwd_res_odd);
        VML_VC_FUNC_NOPG_1PARA(shifted_src, svreinterpret_u32_f16, x);
        VML_VC_FUNC_PG_N2_2PARA(shifted_src, svlsl_n_u32_x, pgB32, shifted_src, F16_SHIFT);
        VML_VC_FUNC_NOPG_1PARA(x, svreinterpret_f16_u32, shifted_src);

        VML_VC_FUNC_PG_2PARA(fwd_res, svadd_f16_x, pgB16, fwd_res, x);

        VML_VC_ST_STEP(svst1_f16, pgB16, dst, step, fwd_res);
    } else {
        VML_VC_DECLARE(svfloat16_t, dd);
        VML_VC_DECLARE(svfloat32_t, dd_even);
        VML_VC_DECLARE(svfloat32_t, dd_odd);
        VML_VC_LD_STEP(dd, svld1_f16, pgB16, diffDst, step);

        // even values convert
        VML_VC_FUNC_PG_1PARA(dd_even, svcvt_f32_f16_x, pgB16, dd);

        // right shift for odd values convert
        VML_VC_FUNC_NOPG_1PARA(shifted_src, svreinterpret_u32_f16, dd);
        VML_VC_FUNC_PG_N2_2PARA(shifted_src, svlsr_n_u32_x, pgB32, shifted_src, F16_SHIFT);
        VML_VC_FUNC_NOPG_1PARA(dd, svreinterpret_f16_u32, shifted_src);
        VML_VC_FUNC_PG_1PARA(dd_odd, svcvt_f32_f16_x, pgB16, dd);

        // ======
        VML_VC_DECLARE(svfloat32_t, res_even);
        VML_VC_FUNC_PG_2PARA(res_even, svmul_f32_x, pgB32, dd_even, fwd_res_even);

        VML_VC_DECLARE(svfloat32_t, one_m_fwd);
        VML_VC_FUNC_PG_N1_2PARA(one_m_fwd, svsub_f32_x, pgB32, svdup_n_f32(1.0f), fwd_res_even);

        VML_VC_FUNC_PG_2PARA(res_even, svmul_f32_x, pgB32, res_even, one_m_fwd);
        // ======
        VML_VC_DECLARE(svfloat32_t, res_odd);
        VML_VC_FUNC_PG_2PARA(res_odd, svmul_f32_x, pgB32, dd_odd, fwd_res_odd);

        VML_VC_FUNC_PG_N1_2PARA(one_m_fwd, svsub_f32_x, pgB32, svdup_n_f32(1.0f), fwd_res_odd);

        VML_VC_FUNC_PG_2PARA(res_odd, svmul_f32_x, pgB32, res_odd, one_m_fwd);
        // ======

        // even values convert back
        VML_VC_DECLARE(svfloat16_t, res);
        VML_VC_FUNC_PG_1PARA(res, svcvt_f16_f32_x, pgB32, res_even);

        // odd values convert back (reuse of shifted_src registers)
        VML_VC_FUNC_PG_1PARA(x, svcvt_f16_f32_x, pgB32, res_odd);
        VML_VC_FUNC_NOPG_1PARA(shifted_src, svreinterpret_u32_f16, x);
        VML_VC_FUNC_PG_N2_2PARA(shifted_src, svlsl_n_u32_x, pgB32, shifted_src, F16_SHIFT);
        VML_VC_FUNC_NOPG_1PARA(x, svreinterpret_f16_u32, shifted_src);

        VML_VC_FUNC_PG_2PARA(res, svadd_f16_x, pgB16, res, x);

        VML_VC_ST_STEP(svst1_f16, pgB16, dst, step, res);
    }
}

void SigmoidVCFWDF16(const int64_t len, const __fp16 *src, __fp16 *dst)
{
    int64_t i;
    int64_t loopSigmoidf = svcnth();
    for (i = 0; i <= len - loopSigmoidf * VML_VC_UNFOLD_N; i += loopSigmoidf * VML_VC_UNFOLD_N) {
        SigmoidMainKernelF16(src + i, dst + i, loopSigmoidf, NULL);
    }

    for (; i < len; i += loopSigmoidf) {
        svbool_t pgB16 = svwhilelt_b16_s64(i, len);
        svbool_t pgB32 = svwhilelt_b32_s64(i, len);
        svfloat16_t x = svld1_f16(pgB16, src + i);

        svfloat32_t xEven = svcvt_f32_f16_x(pgB16, x);
        svfloat32_t xOdd = CvtOddF32F16(x, pgB16, pgB32);

        xEven = svmul_f32_x(pgB32, xEven, svdup_n_f32(-1.0f));
        xOdd = svmul_f32_x(pgB32, xOdd, svdup_n_f32(-1.0f));

        svfloat32_t resEven = SvSigmoidF32X(xEven, pgB32);
        svfloat32_t resOdd = SvSigmoidF32X(xOdd, pgB32);

        svfloat16_t res = svcvt_f16_f32_x(pgB32, resEven);
        x = CvtOddF16F32(resOdd, pgB32);
        res = svadd_f16_x(pgB16, res, x);
        svst1_f16(pgB16, dst + i, res);
    }
}

void SigmoidVCBWDF16(const int64_t len, const __fp16 *src, const __fp16 *diffDst, __fp16 *diffSrc)
{
    int64_t i;
    int64_t loopSigmoidf = svcnth();
    for (i = 0; i <= len - loopSigmoidf * VML_VC_UNFOLD_N; i += loopSigmoidf * VML_VC_UNFOLD_N) {
        SigmoidMainKernelF16(src + i, diffSrc + i, loopSigmoidf, diffDst + i);
    }

    for (; i < len; i += loopSigmoidf) {
        svbool_t pgB16 = svwhilelt_b16_s64(i, len);
        svbool_t pgB32 = svwhilelt_b32_s64(i, len);
        svfloat16_t x = svld1_f16(pgB16, src + i);

        svfloat32_t xEven = svcvt_f32_f16_x(pgB16, x);
        svfloat32_t xOdd = CvtOddF32F16(x, pgB16, pgB32);

        xEven = svmul_f32_x(pgB32, xEven, svdup_n_f32(-1.0f));
        xOdd = svmul_f32_x(pgB32, xOdd, svdup_n_f32(-1.0f));

        svfloat32_t resFwdEven = SvSigmoidF32X(xEven, pgB32);
        svfloat32_t resFwdOdd = SvSigmoidF32X(xOdd, pgB32);

        svfloat16_t dd = svld1_f16(pgB16, diffDst + i);

        svfloat32_t ddEven = svcvt_f32_f16_x(pgB16, dd);
        svfloat32_t ddOdd = CvtOddF32F16(dd, pgB16, pgB32);

        svfloat32_t resEven = svmul_f32_x(pgB32, ddEven, resFwdEven);
        svfloat32_t oneMYEven = svsub_f32_x(pgB32, svdup_n_f32(1.0f), resFwdEven);
        resEven = svmul_f32_x(pgB32, resEven, oneMYEven);

        svfloat32_t resOdd = svmul_f32_x(pgB32, ddOdd, resFwdOdd);
        svfloat32_t oneMYOdd = svsub_f32_x(pgB32, svdup_n_f32(1.0f), resFwdOdd);
        resOdd = svmul_f32_x(pgB32, resOdd, oneMYOdd);

        svfloat16_t res = svcvt_f16_f32_x(pgB32, resEven);
        x = CvtOddF16F32(resOdd, pgB32);
        res = svadd_f16_x(pgB16, res, x);
        svst1_f16(pgB16, diffSrc + i, res);
    }
}
#ifdef __clang__
__attribute__((optnone))
#endif
static void SigmoidMainKernelBF16(const __bf16 *src, __bf16 *dst, int step, const __bf16 *diffDst)
{
    VML_VC_DECLARE(svbfloat16_t, x);
    VML_VC_DECLARE(svfloat32_t, xBot);
    VML_VC_DECLARE(svfloat32_t, xTop);
    svbool_t pgB16 = svptrue_b16();
    svbool_t pgB32 = svptrue_b32();

    VML_VC_LD_STEP(x, svld1_bf16, pgB16, src, step);

    // convert to fp32
    VML_VC_DECLARE(svuint32_t, uScale32);
    VML_VC_DECLARE(svuint16_t, uScale16);
    VML_VC_FUNC_NOPG_1PARA(uScale32, svreinterpret_u32_bf16, x);
    VML_VC_FUNC_NOPG_1PARA(uScale16, svreinterpret_u16_u32, uScale32);

    VML_VC_DECLARE(svuint16_t, vecBot);
    VML_VC_DECLARE(svuint16_t, vecTop);
    VML_VC_FUNC_NOPG_N1_2PARA(vecBot, svzip1_u16, VML_VC_U16_ZERO, uScale16);
    VML_VC_FUNC_NOPG_N1_2PARA(vecTop, svzip2_u16, VML_VC_U16_ZERO, uScale16);

    VML_VC_FUNC_NOPG_1PARA(xBot, svreinterpret_f32_u16, vecBot);
    VML_VC_FUNC_NOPG_1PARA(xTop, svreinterpret_f32_u16, vecTop);

    // ========================
    VML_VC_FUNC_PG_1PARA(xBot, svneg_f32_x, pgB32, xBot);

    VML_VC_DECLARE(svfloat32_t, fwdResBot);
    VML_VC_EXP(pgB32, xBot, fwdResBot);
    VML_VC_FUNC_PG_N1_2PARA(fwdResBot, svadd_f32_x, pgB32, VML_VC_F32_ONE, fwdResBot);
    VML_VC_FUNC_PG_N1_2PARA(fwdResBot, svdiv_f32_x, pgB32, VML_VC_F32_ONE, fwdResBot);

    // ========================
    VML_VC_FUNC_PG_1PARA(xTop, svneg_f32_x, pgB32, xTop);

    VML_VC_DECLARE(svfloat32_t, fwdResTop);
    VML_VC_EXP(pgB32, xTop, fwdResTop);
    VML_VC_FUNC_PG_N1_2PARA(fwdResTop, svadd_f32_x, pgB32, VML_VC_F32_ONE, fwdResTop);
    VML_VC_FUNC_PG_N1_2PARA(fwdResTop, svdiv_f32_x, pgB32, VML_VC_F32_ONE, fwdResTop);

    // ========================

    if (!diffDst) { // forward computation has ended
        // convert back to bf16 and store
        CONVERT_STORE_FP32_TO_BF16(dst, pgB32, pgB16, fwdResBot, fwdResTop, step);
    } else {
        VML_VC_DECLARE(svbfloat16_t, dd);
        VML_VC_DECLARE(svfloat32_t, ddBot);
        VML_VC_DECLARE(svfloat32_t, ddTop);
        VML_VC_LD_STEP(dd, svld1_bf16, pgB16, diffDst, step);

        // convert to fp32
        VML_VC_FUNC_NOPG_1PARA(uScale32, svreinterpret_u32_bf16, dd);
        VML_VC_FUNC_NOPG_1PARA(uScale16, svreinterpret_u16_u32, uScale32);

        VML_VC_FUNC_NOPG_N1_2PARA(vecBot, svzip1_u16, VML_VC_U16_ZERO, uScale16);
        VML_VC_FUNC_NOPG_N1_2PARA(vecTop, svzip2_u16, VML_VC_U16_ZERO, uScale16);

        VML_VC_FUNC_NOPG_1PARA(ddBot, svreinterpret_f32_u16, vecBot);
        VML_VC_FUNC_NOPG_1PARA(ddTop, svreinterpret_f32_u16, vecTop);

        // ======
        VML_VC_DECLARE(svfloat32_t, resBot);
        VML_VC_FUNC_PG_2PARA(resBot, svmul_f32_x, pgB32, ddBot, fwdResBot);

        VML_VC_DECLARE(svfloat32_t, one_m_fwd);
        VML_VC_FUNC_PG_N1_2PARA(one_m_fwd, svsub_f32_x, pgB32, svdup_n_f32(1.0f), fwdResBot);

        VML_VC_FUNC_PG_2PARA(resBot, svmul_f32_x, pgB32, resBot, one_m_fwd);
        // ======
        VML_VC_DECLARE(svfloat32_t, resTop);
        VML_VC_FUNC_PG_2PARA(resTop, svmul_f32_x, pgB32, ddTop, fwdResTop);

        VML_VC_FUNC_PG_N1_2PARA(one_m_fwd, svsub_f32_x, pgB32, svdup_n_f32(1.0f), fwdResTop);

        VML_VC_FUNC_PG_2PARA(resTop, svmul_f32_x, pgB32, resTop, one_m_fwd);

        // convert back to bf16 and store
        CONVERT_STORE_FP32_TO_BF16(dst, pgB32, pgB16, resBot, resTop, step);
    }
}

void SigmoidVCFWDBF16(const int64_t len, const __bf16 *src, __bf16 *dst)
{
    int64_t i;
    int64_t loopSigmoidf = svcnth();
    int numF32 = svcntw();
    for (i = 0; i <= len - loopSigmoidf * VML_VC_UNFOLD_N; i += loopSigmoidf * VML_VC_UNFOLD_N) {
        SigmoidMainKernelBF16(src + i, dst + i, loopSigmoidf, NULL);
    }

    for (; i < len; i += loopSigmoidf) {
        svbool_t pgB16 = svwhilelt_b16_s64(i, len);
        svbool_t pgB32 = svwhilelt_b32_s64(i, len);
        svbool_t pgF32Bot = pgB32;
        svbool_t pgF32Top = svwhilelt_b32_s64(i + numF32, len);
        svbfloat16_t x = svld1_bf16(pgB16, src + i);

        // convert to fp32
        svuint32_t uScale32 = svreinterpret_u32_bf16(x);
        svuint16_t uScale16 = svreinterpret_u16_u32(uScale32);

        svuint16_t vecBot = svzip1_u16(VML_VC_U16_ZERO, uScale16);
        svuint16_t vecTop = svzip2_u16(VML_VC_U16_ZERO, uScale16);

        svfloat32_t xBot = svreinterpret_f32_u16(vecBot);
        svfloat32_t xTop = svreinterpret_f32_u16(vecTop);

        xBot = svmul_f32_x(pgB32, xBot, svdup_n_f32(-1.0f));
        xTop = svmul_f32_x(pgB32, xTop, svdup_n_f32(-1.0f));

        svfloat32_t resBot = SvSigmoidF32X(xBot, pgB32);
        svfloat32_t resTop = SvSigmoidF32X(xTop, pgB32);

        // convert back to bf16
        svbfloat16_t res = SvCvtBF16F32X(pgF32Bot, resBot, pgF32Top, resTop);
        svst1_bf16(pgB16, dst + i, res);
    }
}

void SigmoidVCBWDBF16(const int64_t len, const __bf16 *src, const __bf16 *diffDst, __bf16 *diffSrc)
{
    int64_t i;
    int64_t loopSigmoidf = svcnth();
    for (i = 0; i <= len - loopSigmoidf * VML_VC_UNFOLD_N; i += loopSigmoidf * VML_VC_UNFOLD_N) {
        SigmoidMainKernelBF16(src + i, diffSrc + i, loopSigmoidf, diffDst + i);
    }

    for (; i < len; i += loopSigmoidf) {
        svbool_t pgB16 = svwhilelt_b16_s64(i, len);
        svbool_t pgB32 = svwhilelt_b32_s64(i, len);
        svbfloat16_t x = svld1_bf16(pgB16, src + i);

        // convert to fp32
        svuint32_t uScale32 = svreinterpret_u32_bf16(x);
        svuint16_t uScale16 = svreinterpret_u16_u32(uScale32);
        svuint16_t vecBot = svzip1_u16(VML_VC_U16_ZERO, uScale16);
        svuint16_t vecTop = svzip2_u16(VML_VC_U16_ZERO, uScale16);
        svfloat32_t xBot = svreinterpret_f32_u16(vecBot);
        svfloat32_t xTop = svreinterpret_f32_u16(vecTop);

        xBot = svmul_f32_x(pgB32, xBot, svdup_n_f32(-1.0f));
        xTop = svmul_f32_x(pgB32, xTop, svdup_n_f32(-1.0f));

        svfloat32_t resFwdBot = SvSigmoidF32X(xBot, pgB32);
        svfloat32_t resFwdTop = SvSigmoidF32X(xTop, pgB32);

        svbfloat16_t dd = svld1_bf16(pgB16, diffDst + i);

        // convert to fp32
        uScale32 = svreinterpret_u32_bf16(dd);
        uScale16 = svreinterpret_u16_u32(uScale32);
        vecBot = svzip1_u16(VML_VC_U16_ZERO, uScale16);
        vecTop = svzip2_u16(VML_VC_U16_ZERO, uScale16);
        svfloat32_t ddBot = svreinterpret_f32_u16(vecBot);
        svfloat32_t ddTop = svreinterpret_f32_u16(vecTop);

        // ========
        svfloat32_t resBot = svmul_f32_x(pgB32, ddBot, resFwdBot);
        svfloat32_t oneMYBot = svsub_f32_x(pgB32, svdup_n_f32(1.0f), resFwdBot);
        resBot = svmul_f32_x(pgB32, resBot, oneMYBot);
        // ========
        svfloat32_t resTop = svmul_f32_x(pgB32, ddTop, resFwdTop);
        svfloat32_t oneMYTop = svsub_f32_x(pgB32, svdup_n_f32(1.0f), resFwdTop);
        resTop = svmul_f32_x(pgB32, resTop, oneMYTop);

        // convert back to bf16
        svuint16_t uSvTmp1 = svreinterpret_u16_f32(resBot);
        svuint16_t uSvTmp2 = svreinterpret_u16_f32(resTop);
        svuint16_t res = svuzp2_u16(uSvTmp1, uSvTmp2);

        svst1_bf16(pgB16, diffSrc + i, svreinterpret_bf16_u16(res));
    }
}
