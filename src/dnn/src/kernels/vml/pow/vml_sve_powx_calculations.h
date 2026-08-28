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

#ifndef VML_VC_POWX_CALCULATIONS_H
#define VML_VC_POWX_CALCULATIONS_H

#include "vml_sve_la_pow.h"
#define VML_VC_UNFOLD_N 4
#include "kernels/vml/common_macro.h"

#define RUN_POWXA_F32_BWD_N_PARA_MODE(length, alaphIn, srcIn, ddIn, dstOut, func) do { \
    int64_t i; \
    int loopPowxaf = svcntw(); \
    for (i = 0; i < (length); i += loopPowxaf) { \
        svbool_t pg = svwhilelt_b32_s64(i, length); \
        svfloat32_t sX = svld1_f32(pg, (srcIn) + (i)); \
        svfloat32_t dX = svld1_f32(pg, (ddIn) + (i)); \
        svst1_f32(pg, (dstOut) + (i), svmul_f32_x(pg, (func)(pg, sX, alaphIn), dX)); \
    } \
} while (0)

#define RUN_POWXA_F16_FWD_MAIN_CAL_MODE(pgB16, pgB32, i, alaphIn, srcIn, dstOut, func) do { \
    svfloat16_t x = svld1_f16(pgB16, (srcIn) + (i)); \
    svfloat32_t xEven = svcvt_f32_f16_x(pgB16, x); \
    svfloat32_t xOdd = CvtOddF32F16(x, pgB16, pgB32); \
    svfloat32_t resEven = (func)(pgB32, xEven, alaphIn); \
    svfloat32_t resOdd = (func)(pgB32, xOdd, alaphIn); \
    svfloat16_t y = svadd_f16_x(pgB16, svcvt_f16_f32_x(pgB32, resEven), CvtOddF16F32(resOdd, pgB32)); \
    svst1_f16(pgB16, (dstOut) + (i), y); \
} while (0)

#define RUN_POWXA_F16_FWD_N_PARA_MODE(length, alaphIn, srcIn, dstOut, func) do { \
    int64_t i; \
    int loopPowxah = svcnth(); \
    for (i = 0; i < (length); i += loopPowxah) { \
        svbool_t pgB16 = svwhilelt_b16_s64(i, length); \
        svbool_t pgB32 = svwhilelt_b32_s64(i, length); \
        RUN_POWXA_F16_FWD_MAIN_CAL_MODE(pgB16, pgB32, i, alaphIn, srcIn, dstOut, func); \
    } \
} while (0)

#define RUN_POWXA_F16_BWD_MAIN_CAL_MODE(pgB16, pgB32, i, alaphIn, srcIn, ddIn, dstOut, func) do { \
    svfloat16_t srcX = svld1_f16(pgB16, (srcIn) + (i)); \
    svfloat16_t ddX = svld1_f16(pgB16, (ddIn) + (i)); \
    svfloat32_t srcXfEven = svcvt_f32_f16_x(pgB16, srcX); \
    svfloat32_t srcXfOdd = CvtOddF32F16(srcX, pgB16, pgB32); \
    svfloat32_t srcXEvenRes = (func)(pgB32, srcXfEven, alaphIn); \
    svfloat32_t srcXOddRes = (func)(pgB32, srcXfOdd, alaphIn); \
    svfloat16_t res = VCGetF16BwdMulDdX(pgB16, ddX, pgB32, srcXEvenRes, srcXOddRes); \
    svst1_f16(pgB16, (dstOut) + (i), res); \
} while (0)

#define RUN_POWXA_F16_BWD_N_PARA_MODE(length, alaphIn, srcIn, ddIn, dstOut, func) do { \
    int64_t i; \
    int loopPowxah = svcnth(); \
    for (i = 0; i < (length); i += loopPowxah) { \
        svbool_t pgB16 = svwhilelt_b16_s64(i, length); \
        svbool_t pgB32 = svwhilelt_b32_s64(i, length); \
        RUN_POWXA_F16_BWD_MAIN_CAL_MODE(pgB16, pgB32, i, alaphIn, srcIn, ddIn, dstOut, func); \
    } \
} while (0)

#define RUN_POWXA_BF16_FWD_MAIN_CAL_MODE(pgB16, pgB32, i, alaphIn, srcIn, dstOut, func) do { \
    svbfloat16_t x = svld1_bf16(pgB16, (srcIn) + (i)); \
    svfloat32_t xBot = SvCvtF32BF16BottomX(x); \
    svfloat32_t xTop = SvCvtF32BF16TopX(x); \
    svfloat32_t resBot = SvF32CvtBf16InfSpecial(pgB32, (func)(pgB32, xBot, alaphIn)); \
    svfloat32_t resTop = SvF32CvtBf16InfSpecial(pgB32, (func)(pgB32, xTop, alaphIn)); \
    svbfloat16_t res = SvCvtBF16F32FullX(resBot, resTop); \
    svst1_bf16(pgB16, (dstOut) + (i), res); \
} while (0)

#define RUN_POWXA_BF16_FWD_N_PARA_MODE(length, alaphIn, srcIn, dstOut, func) do { \
    int64_t i; \
    int loopPowxah = svcnth(); \
    for (i = 0; i < (length); i += loopPowxah) { \
        svbool_t pgB16 = svwhilelt_b16_s64(i, length); \
        svbool_t pgB32 = svwhilelt_b32_s64(i, length); \
        RUN_POWXA_BF16_FWD_MAIN_CAL_MODE(pgB16, pgB32, i, alaphIn, srcIn, dstOut, func); \
    } \
} while (0)

#define RUN_POWXA_BF16_BWD_MAIN_CAL_MODE(pgB16, pgB32, i, alaphIn, srcIn, ddIn, dstOut, func) do { \
    svbfloat16_t srcX = svld1_bf16(pgB16, (srcIn) + (i)); \
    svbfloat16_t ddX = svld1_bf16(pgB16, (ddIn) + (i)); \
    svfloat32_t resBot = (func)(pgB32, SvCvtF32BF16BottomX(srcX), alaphIn); \
    svfloat32_t resTop = (func)(pgB32, SvCvtF32BF16TopX(srcX), alaphIn); \
    resBot = SvF32CvtBf16InfSpecial(pgB32, svmul_f32_x(pgB32, resBot, SvCvtF32BF16BottomX(ddX))); \
    resTop = SvF32CvtBf16InfSpecial(pgB32, svmul_f32_x(pgB32, resTop, SvCvtF32BF16TopX(ddX))); \
    svbfloat16_t res = SvCvtBF16F32FullX(resBot, resTop); \
    svst1_bf16(pgB16, (dstOut) + (i), res); \
} while (0)

#define RUN_POWXA_BF16_BWD_N_PARA_MODE(length, alaphIn, srcIn, ddIn, dstOut, func) do { \
    int64_t i; \
    int loopPowxah = svcnth(); \
    for (i = 0; i < (length); i += loopPowxah) { \
        svbool_t pgB16 = svwhilelt_b16_s64(i, length); \
        svbool_t pgB32 = svwhilelt_b32_s64(i, length); \
        RUN_POWXA_BF16_BWD_MAIN_CAL_MODE(pgB16, pgB32, i, alaphIn, srcIn, ddIn, dstOut, func); \
    } \
} while (0)

inline svfloat32_t VCPowxaF32X2(svbool_t pg, svfloat32_t x, const float alpha)
{
    svfloat32_t x2 = svmul_f32_x(pg, x, x);
    svfloat32_t res = svmul_n_f32_x(pg, x2, alpha);
    return res;
}

inline svfloat32_t VCPowxaF32X3(svbool_t pg, svfloat32_t x, const float alpha)
{
    svfloat32_t x2 = svmul_f32_x(pg, x, x);
    svfloat32_t x3 = svmul_f32_x(pg, x2, x);
    svfloat32_t res = svmul_n_f32_x(pg, x3, alpha);
    return res;
}

inline svfloat32_t VCPowxaF32X4(svbool_t pg, svfloat32_t x, const float alpha)
{
    svfloat32_t x2 = svmul_f32_x(pg, x, x);
    svfloat32_t x4 = svmul_f32_x(pg, x2, x2);
    svfloat32_t res = svmul_n_f32_x(pg, x4, alpha);
    return res;
}

inline svfloat32_t VCPowxaF32X5(svbool_t pg, svfloat32_t x, const float alpha)
{
    svfloat32_t x2 = svmul_f32_x(pg, x, x);
    svfloat32_t x4 = svmul_f32_x(pg, x2, x2);
    svfloat32_t x5 = svmul_f32_x(pg, x4, x);
    svfloat32_t res = svmul_n_f32_x(pg, x5, alpha);
    return res;
}

inline svfloat32_t VCPowxaF32X6(svbool_t pg, svfloat32_t x, const float alpha)
{
    svfloat32_t x2 = svmul_f32_x(pg, x, x);
    svfloat32_t x4 = svmul_f32_x(pg, x2, x2);
    svfloat32_t x6 = svmul_f32_x(pg, x4, x2);
    svfloat32_t res = svmul_n_f32_x(pg, x6, alpha);
    return res;
}

inline svfloat32_t VCPowxaF32X7(svbool_t pg, svfloat32_t x, const float alpha)
{
    svfloat32_t x2 = svmul_f32_x(pg, x, x);
    svfloat32_t x4 = svmul_f32_x(pg, x2, x2);
    svfloat32_t x3 = svmul_f32_x(pg, x2, x);
    svfloat32_t x7 = svmul_f32_x(pg, x4, x3);
    svfloat32_t res = svmul_n_f32_x(pg, x7, alpha);
    return res;
}

inline svfloat32_t VCPowxaF32X8(svbool_t pg, svfloat32_t x, const float alpha)
{
    svfloat32_t x2 = svmul_f32_x(pg, x, x);
    svfloat32_t x4 = svmul_f32_x(pg, x2, x2);
    svfloat32_t x8 = svmul_f32_x(pg, x4, x4);
    svfloat32_t res = svmul_n_f32_x(pg, x8, alpha);
    return res;
}

inline svfloat32_t VCPowxaF32X9(svbool_t pg, svfloat32_t x, const float alpha)
{
    svfloat32_t x2 = svmul_f32_x(pg, x, x);
    svfloat32_t x4 = svmul_f32_x(pg, x2, x2);
    svfloat32_t x8 = svmul_f32_x(pg, x4, x4);
    svfloat32_t x9 = svmul_f32_x(pg, x8, x);
    svfloat32_t res = svmul_n_f32_x(pg, x9, alpha);
    return res;
}

__attribute__((always_inline)) inline void VsPowxaVCNYP0(const int64_t len, const float alpha, float *dst)
{
    int64_t i;
    int loopPowxaf = svcntw();
    for (i = 0; i <= len - loopPowxaf * VML_VC_UNFOLD_N; i += loopPowxaf * VML_VC_UNFOLD_N) {
        svbool_t pg = svptrue_b32();
        VML_VC_DECLARE(svfloat32_t, res);
        VML_VC_FUNC_NOPG_N_1PARA(res, svdup_n_f32, alpha);
        VML_VC_ST_STEP(svst1_f32, pg, dst + i, loopPowxaf, res);
    }
    for (; i < len; i += loopPowxaf) {
        svbool_t pg = svwhilelt_b32_s64(i, len);
        svst1_f32(pg, dst + i, svdup_n_f32(alpha));
    }
}

__attribute__((always_inline)) inline void VsPowxaVCNYP1(const int64_t len, const float *src,
    const float alpha, float *dst)
{
    int64_t i;
    int loopPowxaf = svcntw();
    for (i = 0; i <= len - loopPowxaf * VML_VC_UNFOLD_N; i += loopPowxaf * VML_VC_UNFOLD_N) {
        svbool_t pg = svptrue_b32();
        VML_VC_DECLARE(svfloat32_t, x);
        VML_VC_LD_STEP(x, svld1_f32, pg, src + i, loopPowxaf);
        VML_VC_DECLARE(svfloat32_t, res);
        VML_VC_FUNC_PG_N2_2PARA(res, svmul_n_f32_x, pg, x, alpha);
        VML_VC_ST_STEP(svst1_f32, pg, dst + i, loopPowxaf, res);
    }
    for (; i < len; i += loopPowxaf) {
        svbool_t pg = svwhilelt_b32_s64(i, len);
        svfloat32_t x = svld1_f32(pg, src + i);
        svfloat32_t res = svmul_n_f32_x(pg, x, alpha);
        svst1_f32(pg, dst + i, res);
    }
}

__attribute__((always_inline)) inline void VsPowxaVCNYP2(const int64_t len, const float *src,
    const float alpha, float *dst)
{
    int64_t i;
    int loopPowxaf = svcntw();
    for (i = 0; i <= len - loopPowxaf * VML_VC_UNFOLD_N; i += loopPowxaf * VML_VC_UNFOLD_N) {
        svbool_t pg = svptrue_b32();
        VML_VC_DECLARE(svfloat32_t, x);
        VML_VC_DECLARE(svfloat32_t, res);
        VML_VC_LD_STEP(x, svld1_f32, pg, src + i, loopPowxaf);
        VML_VC_FUNC_PG_2PARA(res, svmul_f32_x, pg, x, x);
        VML_VC_FUNC_PG_N2_2PARA(res, svmul_n_f32_x, pg, res, alpha);
        VML_VC_ST_STEP(svst1_f32, pg, dst + i, loopPowxaf, res);
    }
    for (; i < len; i += loopPowxaf) {
        svbool_t pg = svwhilelt_b32_s64(i, len);
        svfloat32_t x = svld1_f32(pg, src + i);
        svfloat32_t res = VCPowxaF32X2(pg, x, alpha);
        svst1_f32(pg, dst + i, res);
    }
}

__attribute__((always_inline)) inline void VsPowxaVCNYP3(const int64_t len, const float *src,
    const float alpha, float *dst)
{
    int64_t i;
    int loopPowxaf = svcntw();
    for (i = 0; i <= len - loopPowxaf * VML_VC_UNFOLD_N; i += loopPowxaf * VML_VC_UNFOLD_N) {
        svbool_t pg = svptrue_b32();
        VML_VC_DECLARE(svfloat32_t, x);
        VML_VC_DECLARE(svfloat32_t, x2);
        VML_VC_DECLARE(svfloat32_t, res);
        VML_VC_LD_STEP(x, svld1_f32, pg, src + i, loopPowxaf);
        VML_VC_FUNC_PG_2PARA(x2, svmul_f32_x, pg, x, x);
        VML_VC_FUNC_PG_2PARA(res, svmul_f32_x, pg, x2, x);
        VML_VC_FUNC_PG_N2_2PARA(res, svmul_n_f32_x, pg, res, alpha);
        VML_VC_ST_STEP(svst1_f32, pg, dst + i, loopPowxaf, res);
    }
    for (; i < len; i += loopPowxaf) {
        svbool_t pg = svwhilelt_b32_s64(i, len);
        svfloat32_t x = svld1_f32(pg, src + i);
        svfloat32_t res = VCPowxaF32X3(pg, x, alpha);
        svst1_f32(pg, dst + i, res);
    }
}

__attribute__((always_inline)) inline void VsPowxaVCNYP4(const int64_t len, const float *src,
    const float alpha, float *dst)
{
    int64_t i;
    int loopPowxaf = svcntw();
    for (i = 0; i <= len - loopPowxaf * VML_VC_UNFOLD_N; i += loopPowxaf * VML_VC_UNFOLD_N) {
        svbool_t pg = svptrue_b32();
        VML_VC_DECLARE(svfloat32_t, x);
        VML_VC_DECLARE(svfloat32_t, x2);
        VML_VC_DECLARE(svfloat32_t, res);
        VML_VC_LD_STEP(x, svld1_f32, pg, src + i, loopPowxaf);
        VML_VC_FUNC_PG_2PARA(x2, svmul_f32_x, pg, x, x);
        VML_VC_FUNC_PG_2PARA(res, svmul_f32_x, pg, x2, x2);
        VML_VC_FUNC_PG_N2_2PARA(res, svmul_n_f32_x, pg, res, alpha);
        VML_VC_ST_STEP(svst1_f32, pg, dst + i, loopPowxaf, res);
    }
    for (; i < len; i += loopPowxaf) {
        svbool_t pg = svwhilelt_b32_s64(i, len);
        svfloat32_t x = svld1_f32(pg, src + i);
        svfloat32_t res = VCPowxaF32X4(pg, x, alpha);
        svst1_f32(pg, dst + i, res);
    }
}

__attribute__((always_inline)) inline void VsPowxaVCNYP5(const int64_t len, const float *src,
    const float alpha, float *dst)
{
    int64_t i;
    int loopPowxaf = svcntw();
    for (i = 0; i <= len - loopPowxaf * VML_VC_UNFOLD_N; i += loopPowxaf * VML_VC_UNFOLD_N) {
        svbool_t pg = svptrue_b32();
        VML_VC_DECLARE(svfloat32_t, x);
        VML_VC_DECLARE(svfloat32_t, x2);
        VML_VC_DECLARE(svfloat32_t, res);
        VML_VC_LD_STEP(x, svld1_f32, pg, src + i, loopPowxaf);
        VML_VC_FUNC_PG_2PARA(x2, svmul_f32_x, pg, x, x);
        VML_VC_FUNC_PG_2PARA(res, svmul_f32_x, pg, x2, x2);
        VML_VC_FUNC_PG_2PARA(res, svmul_f32_x, pg, res, x);
        VML_VC_FUNC_PG_N2_2PARA(res, svmul_n_f32_x, pg, res, alpha);
        VML_VC_ST_STEP(svst1_f32, pg, dst + i, loopPowxaf, res);
    }
    for (; i < len; i += loopPowxaf) {
        svbool_t pg = svwhilelt_b32_s64(i, len);
        svfloat32_t x = svld1_f32(pg, src + i);
        svfloat32_t res = VCPowxaF32X5(pg, x, alpha);
        svst1_f32(pg, dst + i, res);
    }
}

__attribute__((always_inline)) inline void VsPowxaVCNYP6(const int64_t len, const float *src,
    const float alpha, float *dst)
{
    int64_t i;
    int loopPowxaf = svcntw();
    for (i = 0; i <= len - loopPowxaf * VML_VC_UNFOLD_N; i += loopPowxaf * VML_VC_UNFOLD_N) {
        svbool_t pg = svptrue_b32();
        VML_VC_DECLARE(svfloat32_t, x);
        VML_VC_DECLARE(svfloat32_t, x2);
        VML_VC_DECLARE(svfloat32_t, res);
        VML_VC_LD_STEP(x, svld1_f32, pg, src + i, loopPowxaf);
        VML_VC_FUNC_PG_2PARA(x2, svmul_f32_x, pg, x, x);
        VML_VC_FUNC_PG_2PARA(res, svmul_f32_x, pg, x2, x2);
        VML_VC_FUNC_PG_2PARA(res, svmul_f32_x, pg, res, x2);
        VML_VC_FUNC_PG_N2_2PARA(res, svmul_n_f32_x, pg, res, alpha);
        VML_VC_ST_STEP(svst1_f32, pg, dst + i, loopPowxaf, res);
    }
    for (; i < len; i += loopPowxaf) {
        svbool_t pg = svwhilelt_b32_s64(i, len);
        svfloat32_t x = svld1_f32(pg, src + i);
        svfloat32_t res = VCPowxaF32X6(pg, x, alpha);
        svst1_f32(pg, dst + i, res);
    }
}

__attribute__((always_inline)) inline void VsPowxaVCNYP7(const int64_t len, const float *src,
    const float alpha, float *dst)
{
    int64_t i;
    int loopPowxaf = svcntw();
    for (i = 0; i <= len - loopPowxaf * VML_VC_UNFOLD_N; i += loopPowxaf * VML_VC_UNFOLD_N) {
        svbool_t pg = svptrue_b32();
        VML_VC_DECLARE(svfloat32_t, x);
        VML_VC_DECLARE(svfloat32_t, x2);
        VML_VC_DECLARE(svfloat32_t, x3);
        VML_VC_DECLARE(svfloat32_t, x4);
        VML_VC_DECLARE(svfloat32_t, res);
        VML_VC_LD_STEP(x, svld1_f32, pg, src + i, loopPowxaf);
        VML_VC_FUNC_PG_2PARA(x2, svmul_f32_x, pg, x, x);
        VML_VC_FUNC_PG_2PARA(x4, svmul_f32_x, pg, x2, x2);
        VML_VC_FUNC_PG_2PARA(x3, svmul_f32_x, pg, x2, x);
        VML_VC_FUNC_PG_2PARA(res, svmul_f32_x, pg, x4, x3);
        VML_VC_FUNC_PG_N2_2PARA(res, svmul_n_f32_x, pg, res, alpha);
        VML_VC_ST_STEP(svst1_f32, pg, dst + i, loopPowxaf, res);
    }
    for (; i < len; i += loopPowxaf) {
        svbool_t pg = svwhilelt_b32_s64(i, len);
        svfloat32_t x = svld1_f32(pg, src + i);
        svfloat32_t res = VCPowxaF32X7(pg, x, alpha);
        svst1_f32(pg, dst + i, res);
    }
}

__attribute__((always_inline)) inline void VsPowxaVCNYP8(const int64_t len, const float *src,
    const float alpha, float *dst)
{
    int64_t i;
    int loopPowxaf = svcntw();
    for (i = 0; i <= len - loopPowxaf * VML_VC_UNFOLD_N; i += loopPowxaf * VML_VC_UNFOLD_N) {
        svbool_t pg = svptrue_b32();
        VML_VC_DECLARE(svfloat32_t, x);
        VML_VC_DECLARE(svfloat32_t, x2);
        VML_VC_DECLARE(svfloat32_t, x4);
        VML_VC_DECLARE(svfloat32_t, res);
        VML_VC_LD_STEP(x, svld1_f32, pg, src + i, loopPowxaf);
        VML_VC_FUNC_PG_2PARA(x2, svmul_f32_x, pg, x, x);
        VML_VC_FUNC_PG_2PARA(x4, svmul_f32_x, pg, x2, x2);
        VML_VC_FUNC_PG_2PARA(res, svmul_f32_x, pg, x4, x4);
        VML_VC_FUNC_PG_N2_2PARA(res, svmul_n_f32_x, pg, res, alpha);
        VML_VC_ST_STEP(svst1_f32, pg, dst + i, loopPowxaf, res);
    }
    for (; i < len; i += loopPowxaf) {
        svbool_t pg = svwhilelt_b32_s64(i, len);
        svfloat32_t x = svld1_f32(pg, src + i);
        svfloat32_t res = VCPowxaF32X8(pg, x, alpha);
        svst1_f32(pg, dst + i, res);
    }
}

__attribute__((always_inline)) inline void VsPowxaVCNYP9(const int64_t len, const float *src,
    const float alpha, float *dst)
{
    int64_t i;
    int loopPowxaf = svcntw();
    for (i = 0; i <= len - loopPowxaf * VML_VC_UNFOLD_N; i += loopPowxaf * VML_VC_UNFOLD_N) {
        svbool_t pg = svptrue_b32();
        VML_VC_DECLARE(svfloat32_t, x);
        VML_VC_DECLARE(svfloat32_t, x2);
        VML_VC_DECLARE(svfloat32_t, x4);
        VML_VC_DECLARE(svfloat32_t, res);
        VML_VC_LD_STEP(x, svld1_f32, pg, src + i, loopPowxaf);
        VML_VC_FUNC_PG_2PARA(x2, svmul_f32_x, pg, x, x);
        VML_VC_FUNC_PG_2PARA(x4, svmul_f32_x, pg, x2, x2);
        VML_VC_FUNC_PG_2PARA(res, svmul_f32_x, pg, x4, x4);
        VML_VC_FUNC_PG_2PARA(res, svmul_f32_x, pg, res, x);
        VML_VC_FUNC_PG_N2_2PARA(res, svmul_n_f32_x, pg, res, alpha);
        VML_VC_ST_STEP(svst1_f32, pg, dst + i, loopPowxaf, res);
    }
    for (; i < len; i += loopPowxaf) {
        svbool_t pg = svwhilelt_b32_s64(i, len);
        svfloat32_t x = svld1_f32(pg, src + i);
        svfloat32_t res = VCPowxaF32X9(pg, x, alpha);
        svst1_f32(pg, dst + i, res);
    }
}

__attribute__((always_inline)) inline svbool_t VsPowxSpecialCmpVC(svbool_t pg, svfloat32_t x)
{
    svuint32_t uX = svreinterpret_u32_f32(x);
    svuint32_t xExp = svand_u32_x(pg, uX, VC_FLOAT_NEG_INF_BIT);
    return svcmpge_u32(pg, xExp, VC_FLOAT_INF_BIT);
}

__attribute__((always_inline)) inline svfloat32_t VsPowxSqrtVC(svbool_t pg, svfloat32_t x, const float alpha)
{
    svfloat32_t result = svsqrt_f32_x(pg, x);
    svbool_t cmp = VsPowxSpecialCmpVC(pg, x);
    if (UNLIKELY(svptest_any(pg, cmp))) {
        result = VsFunc2f1Special(cmp, x, svdup_n_f32(0x1p-1f), result, powf);
    }
    result = svmul_n_f32_x(pg, result, alpha);
    return result;
}

__attribute__((always_inline)) inline void VsPowxaLaNYPH(const int64_t len, const float *src,
    const float alpha, float *dst)
{
    int64_t i;
    int loopPowxaf = svcntw();
    for (i = 0; i < len; i += loopPowxaf) {
        svbool_t pg = svwhilelt_b32_s64(i, len);
        svfloat32_t x = svld1_f32(pg, src + i);
        svfloat32_t res = VsPowxSqrtVC(pg, x, alpha);
        svst1_f32(pg, dst + i, res);
    }
}

__attribute__((always_inline)) inline void VsPowxaVCNYPN(const int64_t len, const float *src, const float y,
    const float alpha, float *dst)
{
    if (UNLIKELY(y == 0x1.2p3f)) {
        VsPowxaVCNYP9(len, src, alpha, dst);
    }
    if (UNLIKELY(y == 0x1p3f)) {
        VsPowxaVCNYP8(len, src, alpha, dst);
    }
    if (UNLIKELY(y == 0x1.cp2f)) {
        VsPowxaVCNYP7(len, src, alpha, dst);
    }
    if (UNLIKELY(y == 0x1.8p2f)) {
        VsPowxaVCNYP6(len, src, alpha, dst);
    }
    if (UNLIKELY(y == 0x1.4p2f)) {
        VsPowxaVCNYP5(len, src, alpha, dst);
    }
    if (UNLIKELY(y == 0x1p2f)) {
        VsPowxaVCNYP4(len, src, alpha, dst);
    }
    if (UNLIKELY(y == 0x1.8p1f)) {
        VsPowxaVCNYP3(len, src, alpha, dst);
    }
    if (UNLIKELY(y == 0x1p1f)) {
        VsPowxaVCNYP2(len, src, alpha, dst);
    }
    if (UNLIKELY(y == 0x1p0f)) {
        VsPowxaVCNYP1(len, src, alpha, dst);
    }
    if (UNLIKELY(y == 0x0p0f)) {
        VsPowxaVCNYP0(len, alpha, dst);
    }
}

__attribute__((always_inline)) inline svfloat32_t SvPowxaF32X(svfloat32_t x, const float sy,
    const float salpha, svbool_t pg)
{
    if ((Frintaf(sy) == sy) && (sy >= MAX_POWXF_SP_TH_LO) && (sy < MAX_POWXF_SP_TH_HI)) {
        return svmul_n_f32_x(pg, VsPowxaSPF32(pg, x, sy), salpha);
    }
    const svfloat32_t y = svdup_n_f32(sy);
    svfloat32_t result = svmul_n_f32_x(pg, VsPowLaVCMain(pg, x, y), salpha);
    return result;
}

#endif
