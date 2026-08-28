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
#include <ATen/ops/empty.h>
#include <ATen/record_function.h>
#include "sigmoid.h"
#include "fuzz_common.h"
#include "fuzz_test_interface.h"
 
static constexpr int64_t g_dimRangeInitVal = 4;
static constexpr int64_t g_dimRangeLowerBound = 1;
static constexpr int64_t g_dimRangeUpperBound = 1000;
 
void DtSigmoidFuzz(int testCount)
{
    char funcName[] = "DtSigmoidFuzz";
    DT_FUZZ_START(0, testCount, funcName, 0)
    {
        int eleIndex = 0;
        int64_t inputNum = *(int *)DT_SetGetNumberRangeS64(&g_Element[eleIndex++], g_dimRangeInitVal,
            g_dimRangeLowerBound, g_dimRangeUpperBound);
        
        at::Tensor input = torch::randn({inputNum});
 
        common::HikpSigmoidFwd(input);
    }
    DT_FUZZ_END()
}