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

#include "com_testcase_ulp.h"
#include "com_parse_ulp.h"

const struct UlpCaseInfo g_ulpcases[] = {
    {vmlb, (mathFunc_t)ParseUlpVMLB, (mathFunc_t)UlpCaseFuncVMLB},
    {vmlh, (mathFunc_t)ParseUlpVMLH, (mathFunc_t)UlpCaseFuncVMLH},
    {vmlf, (mathFunc_t)ParseUlpVMLF, (mathFunc_t)UlpCaseFuncVMLF},
    {vmlxh, (mathFunc_t)ParseUlpVMLF, (mathFunc_t)UlpCaseFuncVMLXH},
    {vmlxb, (mathFunc_t)ParseUlpVMLF, (mathFunc_t)UlpCaseFuncVMLXB},
    {vmlxf, (mathFunc_t)ParseUlpVMLF, (mathFunc_t)UlpCaseFuncVMLXF},
    {vmlxyh, (mathFunc_t)ParseUlpVMLF, (mathFunc_t)UlpCaseFuncVMLXYH},
    {vmlxyb, (mathFunc_t)ParseUlpVMLF, (mathFunc_t)UlpCaseFuncVMLXYB},
    {vmlxyf, (mathFunc_t)ParseUlpVMLF, (mathFunc_t)UlpCaseFuncVMLXYF},

    {vmlh2, (mathFunc_t)ParseUlpVMLH, (mathFunc_t)UlpCaseFuncVMLH2},
    {vmlb2, (mathFunc_t)ParseUlpVMLB, (mathFunc_t)UlpCaseFuncVMLB2},
    {vmlf2, (mathFunc_t)ParseUlpVMLF, (mathFunc_t)UlpCaseFuncVMLF2},
    {vmlxh2, (mathFunc_t)ParseUlpVMLF, (mathFunc_t)UlpCaseFuncVMLXH2},
    {vmlxb2, (mathFunc_t)ParseUlpVMLF, (mathFunc_t)UlpCaseFuncVMLXB2},
    {vmlxf2, (mathFunc_t)ParseUlpVMLF, (mathFunc_t)UlpCaseFuncVMLXF2},
    {vmlxyh2, (mathFunc_t)ParseUlpVMLF, (mathFunc_t)UlpCaseFuncVMLXYH2},
    {vmlxyb2, (mathFunc_t)ParseUlpVMLF, (mathFunc_t)UlpCaseFuncVMLXYB2},
    {vmlxyf2, (mathFunc_t)ParseUlpVMLF, (mathFunc_t)UlpCaseFuncVMLXYF2},
};
const int nulpcases = (sizeof(g_ulpcases) / sizeof(g_ulpcases[0]));
