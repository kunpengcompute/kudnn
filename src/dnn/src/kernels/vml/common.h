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

#ifndef KPL_COMMON_H
#define KPL_COMMON_H

#include "kernels/kernel_funcs.h"

/* Optionally used extensions.  */
#ifdef __GNUC__
#define likely(x) __builtin_expect(!!(x), 1)
#define unlikely(x) __builtin_expect(x, 0)
#else
#define likely(x) (x)
#define unlikely(x) (x)
#endif

#define FLOAT_MANTISSA_BIT_LEN      23
#define FLOAT_ABS_MASK              0x7fffffffU
#define FLOAT_SIGN_BIT              0x80000000U
#define FLOAT_INF_BIT               0x7f800000U
#define FLOAT_NEG_INF_BIT           0xff800000U
#define FLOAT_SUBNORMAL_BIT         0x00800000U
#define FLOAT_NAN_BIT               0x7fc00000U
#define FLOAT_EXP_BIAS              127
#define FLOAT_EXP_MASK              0x7f800000U
#define FLOAT_SIGN_EXP_MASK         0xff800000U
#define FLOAT_MANTISSA_MASK         0x7fffffU
#define FLOAT_SHIFT                 (0x1.8p23f)
#define FLOAT_ZERO                  (0.0f)
#define FLOAT_HALF                  (0.5f)
#define FLOAT_TWO                   (2.0f)
#define FLOAT_ONE                   (1.0f)
#define FLOAT_NEG_ONE               (-1.0f)
#define FLOAT_HALF_BIT              0x3f000000

#define F16_SHIFT                   (16)
#define F16_ABS_MASK                0x7fffU
#define F16_INF_BIT                 0x7c00U
#define F16_NAN_BIT                 0x7e00U
#define F16_NEG_INF_BIT             0xfc00U
#define F16_SUBNORMAL_BIT           0x400U

#define LN2F                        (0x1.62e43p-1f)

#define PTR_BARRIER(ptr)                                                      \
    ({                                                                        \
        __typeof (ptr) ptrTmp = (ptr);                                        \
        __asm("" : "+r"(ptrTmp));                                             \
        ptrTmp;                                                               \
    })

inline double Frinta(double x)
{
    double y;
#ifdef __aarch64__
    __asm__ volatile("frinta %d0, %d1 \n\t" : "=w"(y) : "w"(x) :);
#else
    y = x + 0x1.8p52;
    y -= 0x1.8p52;
#endif
    return y;
}

inline float Frintaf(float x)
{
    float y;
#ifdef __aarch64__
    __asm__ volatile("frinta %s0, %s1 \n\t" : "=w"(y) : "w"(x) :);
#else
    y = x + 0x1.8p23;
    y -= 0x1.8p23;
#endif
    return y;
}

#endif
