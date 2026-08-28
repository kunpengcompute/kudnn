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

#include <iostream>
#include "fuzz_common.h"
#include "fuzz_test_interface.h"

static std::map<std::string, FuzzTestInfo> g_operatorTestMap = {
    {"gating_attention", {DtGatingAttentionFuzz, 10}},
    {"triangle_multiplication", {DtTriangleMultiplicationFuzz, 10}},
    {"outer_product_mean", {DtOuterProductMeanFuzz, 10}},
    {"multi_head_attention", {DtMultiHeadAttentionFuzz, 10}},
    {"linear", {DtLinearFuzz, 10}},
    {"sigmoid", {DtSigmoidFuzz, 10}},
    {"flash_attention", {DtFlashAttentionFuzz, 10}},
    {"paged_attention", {DtPagedAttentionFuzz, 10}},
    {"attention", {DtAttentionFuzz, 10}},
};

int DtFuzzTest(std::string testOperator)
{
    DT_Enable_Leak_Check(0, 0);

    auto it = g_operatorTestMap.find(testOperator);
    if (it == g_operatorTestMap.end()) {
        std::cout << "no operator found in map" << std::endl;
        return 1;
    }

    auto info = g_operatorTestMap[testOperator];
    info.func(info.testCount);
    return 0;
}