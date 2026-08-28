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

#ifndef COM_TESTCASE_ULP_H
#define COM_TESTCASE_ULP_H

#include "com_struct.h"

struct UlpCaseInfo {
    FUNC_TYPE funcType;
    mathFunc_t parse;
    mathFunc_t func;
};
extern const int nulpcases;
extern const struct UlpCaseInfo g_ulpcases[];

struct CaseCnt {
    unsigned long long success;
    unsigned long long skipped;
    unsigned long long cntfail;
    unsigned long long total;
};

void calCaseCnt(struct CaseCnt *casecnt, double maxerror, struct UlpInfo *ulpInfo);
void printFloat(double error, struct UlpInfo *ulpInfo, int argsNum, struct CaseCnt *casecnt);
void printFloat16(double error, struct UlpInfo *ulpInfo, int argsNum, struct CaseCnt *casecnt);
void printBfloat16(double error, struct UlpInfo *ulpInfo, int argsNum, struct CaseCnt *casecnt);

// FWD
void UlpCaseFuncVMLH(struct UlpInfo *ulpInfo);
void UlpCaseFuncVMLB(struct UlpInfo *ulpInfo);
void UlpCaseFuncVMLF(struct UlpInfo *ulpInfo);
void UlpCaseFuncVMLXH(struct UlpInfo *ulpInfo);
void UlpCaseFuncVMLXB(struct UlpInfo *ulpInfo);
void UlpCaseFuncVMLXF(struct UlpInfo *ulpInfo);
void UlpCaseFuncVMLXYH(struct UlpInfo *ulpInfo);
void UlpCaseFuncVMLXYB(struct UlpInfo *ulpInfo);
void UlpCaseFuncVMLXYF(struct UlpInfo *ulpInfo);

// BWD
void UlpCaseFuncVMLH2(struct UlpInfo *ulpInfo);
void UlpCaseFuncVMLB2(struct UlpInfo *ulpInfo);
void UlpCaseFuncVMLF2(struct UlpInfo *ulpInfo);
void UlpCaseFuncVMLXH2(struct UlpInfo *ulpInfo);
void UlpCaseFuncVMLXB2(struct UlpInfo *ulpInfo);
void UlpCaseFuncVMLXF2(struct UlpInfo *ulpInfo);
void UlpCaseFuncVMLXYH2(struct UlpInfo *ulpInfo);
void UlpCaseFuncVMLXYB2(struct UlpInfo *ulpInfo);
void UlpCaseFuncVMLXYF2(struct UlpInfo *ulpInfo);

#endif
