/*
 * types.h
 *
 * Copyright (c) 2005-2018, Arm Limited.
 * SPDX-License-Identifier: MIT
 */

#ifndef mathtest_types_h
#define mathtest_types_h

#include <limits.h>
#include <stdint.h>
#include <arm_sve.h>

#define MPFR_PREC 96 /* good enough for float or double + a few extra bits */

static inline uint16_t Asuint16(__fp16 f)
{
    union {
        __fp16 f;
        uint16_t i;
    } u = {f};
    return u.i;
}

static inline uint16_t Asuintbf16(__bf16 f)
{
    union {
        __bf16 f;
        uint16_t i;
    } u = {f};
    return u.i;
}

static inline uint32_t Asuint(float f)
{
    union {
        float f;
        uint32_t i;
    } u = {f};
    return u.i;
}

static inline uint64_t Asuint64(double f)
{
    union {
        double f;
        uint64_t i;
    } u = {f};
    return u.i;
}

static inline __fp16 Asfloat16(uint16_t i)
{
    union {
        uint16_t i;
        __fp16 f;
    } u = {i};
    return u.f;
}

static inline __bf16 Asbfloat16(uint16_t i)
{
    union {
        uint16_t i;
        __bf16 f;
    } u = {i};
    return u.f;
}

static inline float Asfloat(uint32_t i)
{
    union {
        uint32_t i;
        float f;
    } u = {i};
    return u.f;
}

static inline double Asdouble(uint64_t i)
{
    union {
        uint64_t i;
        double f;
    } u = {i};
    return u.f;
}

#if UINT_MAX == 4294967295
typedef unsigned int uint32;
typedef int int32;

#define I32 ""
#elif ULONG_MAX == 4294967295
typedef unsigned long uint32;
typedef long int32;
#define I32 "l"
#else
#error Could not find an unsigned 32-bit integer type
#endif

#endif
