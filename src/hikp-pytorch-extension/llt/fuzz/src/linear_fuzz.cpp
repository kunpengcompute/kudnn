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
#include "linear.h"
#include "fuzz_common.h"
#include "fuzz_test_interface.h"

static constexpr int64_t g_dimRangeInitVal = 4;
static constexpr int64_t g_dimRangeLowerBound = 1;
static constexpr int64_t g_dimRangeUpperBound = 30;

void DtLinearFuzz(int testCount)
{
    char funcName[] = "DtLinearFuzz";
    DT_FUZZ_START(0, testCount, funcName, 0)
    {
        int eleIndex = 0;
        int64_t dim = *(int *)DT_SetGetNumberRangeS64(&g_Element[eleIndex++], g_dimRangeInitVal,
            g_dimRangeLowerBound, g_dimRangeUpperBound);
        int64_t inputFeat = *(int *)DT_SetGetNumberRangeS64(&g_Element[eleIndex++], g_dimRangeInitVal,
            g_dimRangeLowerBound, g_dimRangeUpperBound);
        int64_t outputFeat = *(int *)DT_SetGetNumberRangeS64(&g_Element[eleIndex++], g_dimRangeInitVal,
            g_dimRangeLowerBound, g_dimRangeUpperBound);
        
        at::Tensor input = torch::randn({dim, inputFeat});
        at::Tensor weight = torch::randn({outputFeat, inputFeat});
        at::Tensor bias = torch::randn({outputFeat});
        std::optional<at::Tensor> biasOpt = bias;

        common::KpexLinear(input, weight, biasOpt);
    }
    DT_FUZZ_END()
}