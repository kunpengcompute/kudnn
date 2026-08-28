/*
 * dotest.c - actually generate mathlib test cases
 *
 * Copyright (c) 1999-2018, Arm Limited.
 * SPDX-License-Identifier: MIT
 */

#include "random.h"
#include "kpl_mpfr.h"

/*
 * Convert back and forth between IEEE bit patterns and the
 * mpfr_t/mpc_t types.
 */

void set_mpfr_d(mpfr_t x, uint32 h, uint32 l)
{
    uint64_t hl = ((uint64_t)h << 32) | l;
    uint32 exp = (hl >> 52) & 0x7ff;
    int64_t mantissa = hl & (((uint64_t)1 << 52) - 1);
    int sign = (hl >> 63) ? -1 : +1;
    if (exp == 0x7ff) {
        if (mantissa == 0)
            mpfr_set_inf(x, sign);
        else
            mpfr_set_nan(x);
    } else if (exp == 0 && mantissa == 0) {
        mpfr_set_ui(x, 0, GMP_RNDN);
        mpfr_setsign(x, x, sign < 0, GMP_RNDN);
    } else {
        if (exp != 0)
            mantissa |= ((uint64_t)1 << 52);
        else
            exp++;
        mpfr_set_sj_2exp(x, mantissa * sign, (int)exp - 0x3ff - 52, GMP_RNDN);
    }
}

void GenHigh32Bits(uint32 *out, uint32 lowbound, uint32 highbound)
{
    if((lowbound & 0x80000000) == (highbound & 0x80000000)) {
        if (lowbound & 0x80000000) {
            out[0] = highbound + random_upto(lowbound - highbound);
        } else {
            out[0] = lowbound + random_upto(highbound - lowbound);
        }
    } else {
        uint32_t nega_range = lowbound - 0x80000000 + 1;
        uint32_t posi_range = highbound;
        uint32_t rand_value = random_upto(nega_range + posi_range);
        if (rand_value < nega_range) {
            out[0] = lowbound - rand_value;
        } else {
            out[0] = rand_value - nega_range;
        }
    }
}

void cases_normal_float(uint32 *out, uint64_t low, uint64_t high)
{
    if (low == high) {
        out[0] = low;
        return;
    }
    uint32 lowbound = low;
    uint32 highbound = high;
    GenHigh32Bits(out, lowbound, highbound);
}

void cases_normal_float16(uint32_t *out, uint64_t low, uint64_t high)
{
    if (low == high) {
        out[0] = low;
        return;
    }
    uint16_t lowbound = low;
    uint16_t highbound = high;
    if ((lowbound & 0x8000) == (highbound & 0x8000)) {
        if (lowbound & 0x8000) {
            out[0] = highbound + random_upto(lowbound - highbound);
        } else {
            out[0] = lowbound + random_upto(highbound - lowbound);
        }
    } else {
        uint16_t nega_range = lowbound - 0x8000 + 1;
        uint16_t posi_range = highbound;
        uint16_t rand_value = random_upto(nega_range + posi_range);
        if (rand_value < nega_range) {
            out[0] = lowbound - rand_value;
        } else {
            out[0] = rand_value - nega_range;
        }
    }
}
