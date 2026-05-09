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

#ifndef VML_VectorComputation_BASE_H
#define VML_VectorComputation_BASE_H

#include <arm_sve.h>
#include <math.h>
#include "common.h"

#define VectorComputation_FLOAT_LD_POLY_SIZE 16
#define VectorComputation_FLOAT_MANTISSA_BIT_LEN svdup_n_u32(FLOAT_MANTISSA_BIT_LEN)
#define VectorComputation_FLOAT_SIGN_BIT svdup_n_u32(FLOAT_SIGN_BIT)
#define VectorComputation_FLOAT_ABS_MASK svdup_n_u32(FLOAT_ABS_MASK)
#define VectorComputation_FLOAT_MANTISSA_MASK svdup_n_u32(FLOAT_MANTISSA_MASK)
#define VectorComputation_FLOAT_SUBNORMAL_BIT svdup_n_u32(FLOAT_SUBNORMAL_BIT)
#define VectorComputation_FLOAT_INF_BIT svdup_n_u32(FLOAT_INF_BIT)
#define VectorComputation_FLOAT_NEG_INF_BIT svdup_n_u32(FLOAT_NEG_INF_BIT)
#define VectorComputation_FLOAT_SHIFT svdup_n_f32(FLOAT_SHIFT)
#define VectorComputation_FLOAT_EXP_MASK svdup_n_u32(FLOAT_EXP_MASK)
#define VectorComputation_FLOAT_SIGN_EXP_MASK svdup_n_u32(FLOAT_SIGN_EXP_MASK)
#define VectorComputation_FLOAT_ONE svdup_n_f32(FLOAT_ONE)
#define VectorComputation_FLOAT_NEG_ONE svdup_n_f32(FLOAT_NEG_ONE)
#define VectorComputation_FLOAT_ZERO svdup_n_f32(FLOAT_ZERO)
#define VectorComputation_FLOAT_HALF svdup_n_f32(FLOAT_HALF)
#define VectorComputation_FLOAT_LN2 svdup_n_f32(LN2F)
#define VectorComputation_FLOAT_TWO svdup_n_f32(FLOAT_TWO)
#define VectorComputation_FLOAT_INF svdup_n_f32(INFINITY)
#define VectorComputation_FLOAT_NEG_INF svdup_n_f32(-INFINITY)
#define VectorComputation_FLOAT_HALF_BIT svdup_n_u32(FLOAT_HALF_BIT)

#define VML_VectorComputation_F32_ZERO svdup_n_f32(0.0f)
#define VML_VectorComputation_F32_ONE svdup_n_f32(0x1p0f)
#define VML_VectorComputation_F32_INVSQRT2 svdup_n_f32(0x1.6a09e6p-1f) // 1/sqrt(2)

#define VML_VectorComputation_U16_ZERO svdup_n_u16(0U)
#define VML_VectorComputation_U16_ABS svdup_n_u16(0x7fffU)
#define VML_VectorComputation_U16_SIGN svdup_n_u16(0x8000U)
#define VML_VectorComputation_U16_NEG_ZERO VML_VectorComputation_U16_SIGN

#define VML_VectorComputation_F16_ZERO svdup_n_f16(0.0f16)
#define VML_VectorComputation_F16_HALF svdup_n_f16(0.5f)
#define VML_VectorComputation_F16_ONE svdup_n_f16(1.0f)
#define VML_VectorComputation_F16_TWO svdup_n_f16(2.0f)

#define VML_VectorComputation_F16_INF svdup_n_u16(F16_INF_BIT)
#define LN_MIN_F32 svdup_f32(-87.0f)
#define LN_MAX_F32 svdup_f32(87.0f)
#define INVLN2_F32 svdup_n_f32(0x1.715476p+0f)
#define NEGINVLN2_F32 svdup_n_f32(-0x1.715476p+0f)
#define SHIFTER_F32 svdup_n_f32(0x1.803f8p17f)
#define POLY_C0_F32 svdup_n_f32(0x1.62e584p-1f)
#define POLY_C1_F32 svdup_n_f32(0x1.ebfc28p-3f)
#define FLOAT_ZERO_VEC_F32 svdup_n_f32(0.0f)
#define FLOAT_ONE_VEC_F32 svdup_n_f32(1.0f)
#define INVLN2_F32_LOW svdup_n_f32(0x1.4ae0cp-26f)
#define LN_MIN_F16 svdup_f16(-87.0f)
#define INVLN2_F16 svdup_n_f16(0x1.715476p+0f)
#define SHIFTER_F16 svdup_n_f16(+0x1.78000000p+0005)
#define POLY_C0_F16 svdup_n_f16(0x1.62e584p-1f)
#define POLY_C1_F16 svdup_n_f16(0x1.ebfc28p-3f)

inline svfloat32_t SvAndF32X(svbool_t pg, svfloat32_t f, svuint32_t u)
{
    svuint32_t fu = svreinterpret_u32_f32(f);
    fu = svand_u32_x(pg, fu, u);
    svfloat32_t fs = svreinterpret_f32_u32(fu);
    return fs;
}

inline svfloat32_t SvOrrF32X(svbool_t pg, svfloat32_t f, svuint32_t u)
{
    svuint32_t fu = svreinterpret_u32_f32(f);
    fu = svorr_u32_x(pg, fu, u);
    svfloat32_t fs = svreinterpret_f32_u32(fu);
    return fs;
}

inline svfloat16_t SvOrrF16X(svbool_t pg, svfloat16_t f, svuint16_t u)
{
    svuint16_t fu = svreinterpret_u16_f16(f);
    fu = svorr_u16_x(pg, fu, u);
    svfloat16_t fs = svreinterpret_f16_u16(fu);
    return fs;
}

// when x is normal and uint(absx) >= 0x7f7f8000, bf16(x) = 0x7f80
static inline svfloat32_t SvF32CvtBf16InfSpecial(svbool_t pg, svfloat32_t x)
{
    svuint32_t ux = svreinterpret_u32_f32(x);
    svuint32_t uAbsX = svand_u32_x(pg, ux, VectorComputation_FLOAT_ABS_MASK);
    svbool_t pgGeMin = svcmpge_n_u32(pg, uAbsX, 0x7f7f8000U);
    svbool_t pgLtInf = svcmplt_n_u32(pg, uAbsX, 0x7f800000U);
    svbool_t pgSp = svand_b_z(pg, pgGeMin, pgLtInf);
    if (unlikely(svptest_any(pg, pgSp))) {
        svuint32_t signInf = svorr_n_u32_x(pg, svand_u32_x(pg, ux, VectorComputation_FLOAT_SIGN_BIT), 0x7f800000U);
        return svsel_f32(pgSp, svreinterpret_f32_u32(signInf), x);
    }
    return x;
}

inline svbfloat16_t SvCvtBF16F32FullX(svfloat32_t bot, svfloat32_t top)
{
    svuint16_t uBot = svreinterpret_u16_f32(bot);
    svuint16_t uTop = svreinterpret_u16_f32(top);
    svuint16_t uRes = svuzp2_u16(uBot, uTop);
    svbfloat16_t res = svreinterpret_bf16_u16(uRes);
    return res;
}

inline svfloat32_t SvCvtF32BF16BottomX(svbfloat16_t x)
{
    svuint16_t uScale16 = svreinterpret_u16_bf16(x);
    svuint16_t vecBot = svzip1_u16(VML_VectorComputation_U16_ZERO, uScale16);
    svfloat32_t xBot = svreinterpret_f32_u16(vecBot);
    return xBot;
}

inline svfloat32_t SvCvtF32BF16TopX(svbfloat16_t x)
{
    svuint16_t uScale16 = svreinterpret_u16_bf16(x);
    svuint16_t vecTop = svzip2_u16(VML_VectorComputation_U16_ZERO, uScale16);
    svfloat32_t xTop = svreinterpret_f32_u16(vecTop);
    return xTop;
}

inline svfloat16_t VhFunc1f1Special(svbool_t pgSp, svfloat16_t x, svfloat16_t y, float16_t (*func)(float16_t))
{
    svfloat16_t t = y;
    svbool_t pn = svpnext_b16(pgSp, svpfalse());
    while (svptest_any(pgSp, pn)) {
        float16_t xr = svlastb_f16(pn, x);
        float16_t ret = func(xr);
        t = svdup_f16_m(t, pn, ret);
        pn = svpnext_b16(pgSp, pn);
    }

    return t;
}

inline svfloat32_t VsFunc2f1Special(svbool_t pgSp, svfloat32_t x1, svfloat32_t x2, svfloat32_t y,
                                    float32_t (*func)(float32_t, float32_t))
{
    svfloat32_t t = y;
    svbool_t pn = svpnext_b32(pgSp, svpfalse());
    while (svptest_any(pgSp, pn)) {
        float32_t x1r = svlastb_f32(pn, x1);
        float32_t x2r = svlastb_f32(pn, x2);
        float32_t ret = func(x1r, x2r);
        t = svdup_f32_m(t, pn, ret);
        pn = svpnext_b32(pgSp, pn);
    }

    return t;
}

static inline svfloat16_t SveGetF16BwdMulDdX(svbool_t pgB16, svfloat16_t ddX, svbool_t pgB32, svfloat32_t xEvenRes,
                                             svfloat32_t xOddRes)
{
    svfloat32_t dEven = svcvt_f32_f16_x(pgB16, ddX);
    svfloat32_t dOdd = CvtOddF32F16(ddX, pgB16, pgB32);
    svfloat32_t dEvenRes = svmul_f32_x(pgB32, xEvenRes, dEven);
    svfloat32_t dOddRes = svmul_f32_x(pgB32, xOddRes, dOdd);
    svfloat16_t resEven = svcvt_f16_f32_x(pgB32, dEvenRes);
    svfloat16_t resOdd = CvtOddF16F32(dOddRes, pgB32);
    svfloat16_t res = svadd_f16_x(pgB16, resEven, resOdd);
    return res;
}

inline uint32_t SveCvtU16ToU32(uint16_t u)
{
    return (uint32_t)u << F16_SHIFT;
}

inline uint16_t SveCvtU32ToU16(uint32_t u)
{
    uint32_t res = u + F16_ABS_MASK + ((u >> F16_SHIFT) & 1U);
    res = res >> F16_SHIFT;

    return (uint16_t)res;
}

inline uint16_t AsUint16(__bf16 b)
{
    union {
        __bf16 b;
        uint16_t i;
    } u = {b};
    return u.i;
}

inline __bf16 AsBF16(uint16_t i)
{
    union {
        uint16_t i;
        __bf16 b;
    } u = {i};
    return u.b;
}

inline uint32_t AsUint(float f)
{
    union {
        float f;
        uint32_t i;
    } u = {f};
    return u.i;
}

inline float AsFloat(uint32_t i)
{
    union {
        uint32_t i;
        float f;
    } u = {i};
    return u.f;
}

#define LOAD_CONVERT_BF16_TO_FP32_1(src, pgB16, xBot, xTop, step)                                               \
    do {                                                                                                        \
        VML_VectorComputation_DECLARE(svbfloat16_t, x);                                                         \
        VML_VectorComputation_DECLARE(svuint32_t, uScale32);                                                    \
        VML_VectorComputation_DECLARE(svuint16_t, uScale16);                                                    \
        VML_VectorComputation_DECLARE(svuint16_t, vecBot);                                                      \
        VML_VectorComputation_DECLARE(svuint16_t, vecTop);                                                      \
        VML_VectorComputation_LD_STEP(x, svld1_bf16, pgB16, src, step);                                         \
        VML_VectorComputation_FUNC_NOPG_1PARA(uScale32, svreinterpret_u32_bf16, x);                             \
        VML_VectorComputation_FUNC_NOPG_1PARA(uScale16, svreinterpret_u16_u32, uScale32);                       \
        VML_VectorComputation_FUNC_NOPG_N1_2PARA(vecBot, svzip1_u16, VML_VectorComputation_U16_ZERO, uScale16); \
        VML_VectorComputation_FUNC_NOPG_N1_2PARA(vecTop, svzip2_u16, VML_VectorComputation_U16_ZERO, uScale16); \
        VML_VectorComputation_FUNC_NOPG_1PARA(xBot, svreinterpret_f32_u16, vecBot);                             \
        VML_VectorComputation_FUNC_NOPG_1PARA(xTop, svreinterpret_f32_u16, vecTop);                             \
    } while (0)

#define LOAD_CONVERT_BF16_TO_FP32_2(src, dd, pgB16, xBot, xTop, ddBot, ddTop, step)                             \
    do {                                                                                                        \
        VML_VectorComputation_DECLARE(svbfloat16_t, x);                                                         \
        VML_VectorComputation_DECLARE(svbfloat16_t, ddX);                                                       \
        VML_VectorComputation_DECLARE(svuint32_t, uScale32);                                                    \
        VML_VectorComputation_DECLARE(svuint16_t, uScale16);                                                    \
        VML_VectorComputation_DECLARE(svuint16_t, vecBot);                                                      \
        VML_VectorComputation_DECLARE(svuint16_t, vecTop);                                                      \
        VML_VectorComputation_LD_STEP(x, svld1_bf16, pgB16, src, step);                                         \
        VML_VectorComputation_LD_STEP(ddX, svld1_bf16, pgB16, dd, step);                                        \
        VML_VectorComputation_FUNC_NOPG_1PARA(uScale32, svreinterpret_u32_bf16, x);                             \
        VML_VectorComputation_FUNC_NOPG_1PARA(uScale16, svreinterpret_u16_u32, uScale32);                       \
        VML_VectorComputation_FUNC_NOPG_N1_2PARA(vecBot, svzip1_u16, VML_VectorComputation_U16_ZERO, uScale16); \
        VML_VectorComputation_FUNC_NOPG_N1_2PARA(vecTop, svzip2_u16, VML_VectorComputation_U16_ZERO, uScale16); \
        VML_VectorComputation_FUNC_NOPG_1PARA(xBot, svreinterpret_f32_u16, vecBot);                             \
        VML_VectorComputation_FUNC_NOPG_1PARA(xTop, svreinterpret_f32_u16, vecTop);                             \
        VML_VectorComputation_FUNC_NOPG_1PARA(uScale32, svreinterpret_u32_bf16, ddX);                           \
        VML_VectorComputation_FUNC_NOPG_1PARA(uScale16, svreinterpret_u16_u32, uScale32);                       \
        VML_VectorComputation_FUNC_NOPG_N1_2PARA(vecBot, svzip1_u16, VML_VectorComputation_U16_ZERO, uScale16); \
        VML_VectorComputation_FUNC_NOPG_N1_2PARA(vecTop, svzip2_u16, VML_VectorComputation_U16_ZERO, uScale16); \
        VML_VectorComputation_FUNC_NOPG_1PARA(ddBot, svreinterpret_f32_u16, vecBot);                            \
        VML_VectorComputation_FUNC_NOPG_1PARA(ddTop, svreinterpret_f32_u16, vecTop);                            \
    } while (0)

#define CONVERT_STORE_FP32_TO_BF16(dst, pgB16, resBot, resTop, step)                  \
    do {                                                                              \
        VML_VectorComputation_DECLARE(svbfloat16_t, res);                             \
        VML_VectorComputation_DECLARE(svuint16_t, vecBot);                            \
        VML_VectorComputation_DECLARE(svuint16_t, vecTop);                            \
        VML_VectorComputation_DECLARE(svuint16_t, uScale16);                          \
        VML_VectorComputation_FUNC_NOPG_1PARA(vecBot, svreinterpret_u16_f32, resBot); \
        VML_VectorComputation_FUNC_NOPG_1PARA(vecTop, svreinterpret_u16_f32, resTop); \
        VML_VectorComputation_FUNC_NOPG_2PARA(uScale16, svuzp2_u16, vecBot, vecTop);  \
        VML_VectorComputation_FUNC_NOPG_1PARA(res, svreinterpret_bf16_u16, uScale16); \
        VML_VectorComputation_ST_STEP(svst1_bf16, pgB16, dst, step, res);             \
    } while (0)

#define LOAD_CONVERT_FP16_TO_FP32_1(src, pgB16, xEven, xOdd, step)                                 \
    do {                                                                                           \
        VML_VectorComputation_DECLARE(svfloat16_t, x);                                             \
        VML_VectorComputation_DECLARE(svuint32_t, tempU32);                                        \
        VML_VectorComputation_DECLARE(svfloat16_t, tempF16);                                       \
        VML_VectorComputation_LD_STEP(x, svld1_f16, pgB16, src, step);                             \
        VML_VectorComputation_FUNC_PG_1PARA(xEven, svcvt_f32_f16_x, pgB16, x);                     \
        VML_VectorComputation_FUNC_NOPG_1PARA(tempU32, svreinterpret_u32_f16, x);                  \
        VML_VectorComputation_FUNC_PG_N2_2PARA(tempU32, svlsr_n_u32_x, pgB32, tempU32, F16_SHIFT); \
        VML_VectorComputation_FUNC_NOPG_1PARA(tempF16, svreinterpret_f16_u32, tempU32);            \
        VML_VectorComputation_FUNC_PG_1PARA(xOdd, svcvt_f32_f16_x, pgB16, tempF16);                \
    } while (0)

#define LOAD_CONVERT_FP16_TO_FP32_2(src, dd, pgB16, xEven, xOdd, ddEven, ddOdd, step)              \
    do {                                                                                           \
        VML_VectorComputation_DECLARE(svfloat16_t, x);                                             \
        VML_VectorComputation_DECLARE(svfloat16_t, ddX);                                           \
        VML_VectorComputation_DECLARE(svuint32_t, tempU32);                                        \
        VML_VectorComputation_DECLARE(svfloat16_t, tempF16);                                       \
        VML_VectorComputation_LD_STEP(x, svld1_f16, pgB16, src, step);                             \
        VML_VectorComputation_LD_STEP(ddX, svld1_f16, pgB16, dd, step);                            \
        VML_VectorComputation_FUNC_PG_1PARA(xEven, svcvt_f32_f16_z, pgB16, x);                     \
        VML_VectorComputation_FUNC_NOPG_1PARA(tempU32, svreinterpret_u32_f16, x);                  \
        VML_VectorComputation_FUNC_PG_N2_2PARA(tempU32, svlsr_n_u32_x, pgB32, tempU32, F16_SHIFT); \
        VML_VectorComputation_FUNC_NOPG_1PARA(tempF16, svreinterpret_f16_u32, tempU32);            \
        VML_VectorComputation_FUNC_PG_1PARA(xOdd, svcvt_f32_f16_x, pgB16, tempF16);                \
        VML_VectorComputation_FUNC_PG_1PARA(ddEven, svcvt_f32_f16_z, pgB16, ddX);                  \
        VML_VectorComputation_FUNC_NOPG_N23_3PARA(ddOdd, CvtOddF32F16, ddX, pgB16, pgB32);         \
        VML_VectorComputation_FUNC_NOPG_1PARA(tempU32, svreinterpret_u32_f16, ddX);                \
        VML_VectorComputation_FUNC_PG_N2_2PARA(tempU32, svlsr_n_u32_x, pgB32, tempU32, F16_SHIFT); \
        VML_VectorComputation_FUNC_NOPG_1PARA(tempF16, svreinterpret_f16_u32, tempU32);            \
        VML_VectorComputation_FUNC_PG_1PARA(ddOdd, svcvt_f32_f16_x, pgB16, tempF16);               \
    } while (0)

#define CONVERT_STORE_FP32_TO_FP16(dst, pgB16, resEven, resOdd, step)                                \
    do {                                                                                             \
        VML_VectorComputation_DECLARE(svfloat16_t, res);                                             \
        VML_VectorComputation_DECLARE(svuint32_t, uScale32);                                         \
        VML_VectorComputation_DECLARE(svfloat16_t, tempF16);                                         \
        VML_VectorComputation_FUNC_PG_1PARA(res, svcvt_f16_f32_x, pgB32, resEven);                   \
        VML_VectorComputation_FUNC_PG_1PARA(tempF16, svcvt_f16_f32_x, pgB32, resOdd);                \
        VML_VectorComputation_FUNC_NOPG_1PARA(uScale32, svreinterpret_u32_f16, tempF16);             \
        VML_VectorComputation_FUNC_PG_N2_2PARA(uScale32, svlsl_n_u32_x, pgB32, uScale32, F16_SHIFT); \
        VML_VectorComputation_FUNC_NOPG_1PARA(tempF16, svreinterpret_f16_u32, uScale32);             \
        VML_VectorComputation_FUNC_PG_2PARA(res, svsel_f16, pgB32, res, tempF16);                    \
        VML_VectorComputation_ST_STEP(svst1_f16, pgB16, dst, step, res);                             \
    } while (0)

#endif
