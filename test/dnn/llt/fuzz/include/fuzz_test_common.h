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

#ifndef FUZZ_TEST_COMMON_H
#define FUZZ_TEST_COMMON_H

#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <cmath>
#include <ctime>

#include <map>

#include "kudnn.hpp"
#include "secodeFuzz.h"

typedef struct {
    const char *cmd;
    void (*func)(int);
    int testCount;
} Cmdinfo;

#ifndef KUDNN_UNUSED
#define KUDNN_UNUSED(x) ((void)(x))
#endif // KUDNN_UNUSED

#define DEFAULT_TEST_NUM     100

#define INDEX0                0
#define INDEX1                1
#define INDEX2                2
#define INDEX3                3
#define INDEX4                4
#define INDEX5                5
#define INDEX6                6
#define INDEX7                7
#define INDEX8                8
#define INDEX9                9
#define INDEX10               10
#define INDEX11               11
#define INDEX12               12
#define INDEX13               13
#define INDEX14               14
#define INDEX15               15
#define INDEX16               16
#define INDEX17               17
#define INDEX18               18
#define INDEX19               19
#define INDEX20               20
#define INDEX21               21
#define INDEX22               22
#define INDEX23               23
#define INDEX24               24
#define INDEX25               25
#define INDEX26               26
#define INDEX27               27
#endif
