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

#include "fuzz_test_common.h"

static void DtFuzzPostOps(int counts)
{
    char funcName[] = "DtFuzzPostOps";
    DT_FUZZ_START(0, counts, funcName, 0)
    {
        int eleIndex = 0;
        int kindIdx = *(float *)DT_SetGetNumberRange(&g_Element[eleIndex++], 0, 0, counts);

        char format[] = "PostOps Sum: index %zu";
        printf(format, kindIdx);
        // since post-ops are aimed only to store parameters which will be checked
        // applying post-ops to some of routines means that there's no sense to test
        // Append* function, since they don't perform any validity checks
        // the only thing to check is Kind() function since
        // the arguments could be out of range
        try {
            KuDNN::PostOps postOps1;
            postOps1.AppendSum(3.5f);
            auto kind = postOps1.Kind(kindIdx);
            (void)kind;
        } catch (const KuDNN::Service::LogicError& e) {
            printf("Invalid Kind index");
        }
        printf("\n");
    }
    DT_FUZZ_END()
}

static void DtFuzzPostOpsConstr(int counts)
{
    // Since PostOps has only default constructor, we can't pass any parameters
    // So we can ignore counts not to retest the same
    (void)counts;
    char funcName[] = "DtFuzzPostopsConstr";
    DT_FUZZ_START(0, 1, funcName, 0)
    {
        char format[] = "PostOps op";
        printf("%s", format);
        try {
            KuDNN::PostOps postOps0;
            KuDNN::PostOps postOps1(postOps0);
            KuDNN::PostOps postOps2 = postOps0;
            KuDNN::PostOps postOps3;
            KuDNN::PostOps postOps4;
            KuDNN::PostOps postOps5 = std::move(postOps3);
            KuDNN::PostOps postOps6(std::move(postOps4));
        } catch (const KuDNN::Service::Unsupported& e) {
            printf("Unsupported");
        } catch (const KuDNN::Service::LogicError& e) {
            printf("LogicError");
        }
        printf("\n");
    }
    DT_FUZZ_END()
}

static constexpr int g_postOpsFuzzTestCount = 25000;

static Cmdinfo g_postOpsCmdTable[] = {
    {"PostOpsConstructor", DtFuzzPostOpsConstr, 1},
    {"PostOps", DtFuzzPostOps, g_postOpsFuzzTestCount},
};

int main(int argc, char *argv[])
{
    DT_Enable_Leak_Check(0, 0);
    if (argc > 1) {
        g_postOpsCmdTable[0] = {"PostOpsConstructor", DtFuzzPostOpsConstr, 1};
        g_postOpsCmdTable[1] = {"PostOps", DtFuzzPostOps, atoi(argv[1])};
    }

    int numOfTestItem = sizeof(g_postOpsCmdTable) / sizeof(g_postOpsCmdTable[0]);
    int testCount = 0;

    for (int testCaseIdx = 0; testCaseIdx < numOfTestItem; testCaseIdx++) {
        printf("g_postOpsCmdTable[%d] = %s test %d time\n", testCaseIdx, g_postOpsCmdTable[testCaseIdx].cmd, testCount);

        testCount = g_postOpsCmdTable[testCaseIdx].testCount;
        g_postOpsCmdTable[testCaseIdx].func(testCount);
    }
    return 0;
}
