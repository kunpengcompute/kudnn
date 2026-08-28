/*
 * intern.h
 *
 * Copyright (c) 1999-2018, Arm Limited.
 * SPDX-License-Identifier: MIT
 */

#ifndef complex_common_h
#define complex_common_h

#include <mpfr.h>
#include "types.h"

void set_mpfr_d(mpfr_t x, uint32 h, uint32 l);

void cases_normal_float(uint32 *out, uint64_t lowbound, uint64_t highbound);
void cases_normal_float16(uint32_t *out, uint64_t lowbound, uint64_t highbound);

#endif
