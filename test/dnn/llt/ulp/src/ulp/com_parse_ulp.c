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

#include "com_parse_ulp.h"

void ParseUlpVMLF(int argc, char **argv, struct UlpInfo *ulpInfo)
{
    /* parse cases interval */
    int i = 0, j = 0;
    while (j < argc && i < MAX_ARGS_NUM) {
        ulpInfo->interval[i].high = ulpInfo->interval[i].low = getnum_f(argv[j++]);
        if (j < argc && strcmp(argv[j], "x")) {
            ulpInfo->interval[i].high = getnum_f(argv[j++]);
        }
        i++;
        if (j < argc && !strcmp(argv[j], "x")) {
            j++;
        } else {
            break;
        }
    }

    /* parse test times */
    if (j < argc) {
        ulpInfo->times = strtoull(argv[j], 0, 0);
    }
}

void ParseUlpVMLH(int argc, char **argv, struct UlpInfo *ulpInfo)
{
    /* parse cases interval */
    int i = 0, j = 0;
    while (j < argc && i < MAX_ARGS_NUM) {
        ulpInfo->interval[i].high = ulpInfo->interval[i].low = getnum_h(argv[j++]);
        if (j < argc && strcmp(argv[j], "x")) {
            ulpInfo->interval[i].high = getnum_h(argv[j++]);
        }
        i++;
        if (j < argc && !strcmp(argv[j], "x")) {
            j++;
        } else {
            break;
        }
    }

    /* parse test times */
    if (j < argc) {
        ulpInfo->times = strtoull(argv[j], 0, 0);
    }
}

void ParseUlpVMLB(int argc, char **argv, struct UlpInfo *ulpInfo)
{
    /* parse cases interval */
    int i = 0, j = 0;
    while (j < argc && i < MAX_ARGS_NUM) {
        ulpInfo->interval[i].high = ulpInfo->interval[i].low = getnum_b(argv[j++]);
        if (j < argc && strcmp(argv[j], "x")) {
            ulpInfo->interval[i].high = getnum_b(argv[j++]);
        }
        i++;
        if (j < argc && !strcmp(argv[j], "x")) {
            j++;
        } else {
            break;
        }
    }

    /* parse test times */
    if (j < argc) {
        ulpInfo->times = strtoull(argv[j], 0, 0);
    }
}
