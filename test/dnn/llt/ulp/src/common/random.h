/*
 * random.h - header for random.c
 *
 * Copyright (c) 2009-2018, Arm Limited.
 * SPDX-License-Identifier: MIT
 */

#ifndef RANDOM_H
#define RANDOM_H

#include "types.h"
#include "random.h"
#include <stdlib.h>
#include <stdio.h>

void seed_random(uint32 seed);
uint32 random32(void);
uint32 random_upto(uint32 limit);
uint32 random_upto_biased(uint32 limit, int bias);

#endif
