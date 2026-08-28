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
#include "triangle_multiplication.h"
#include "fuzz_common.h"
#include "fuzz_test_interface.h"

static constexpr int64_t g_dimRangeInitVal = 4;
static constexpr int64_t g_dimRangeLowerBound = 1;
static constexpr int64_t g_dimRangeUpperBound = 30;

void DtTriangleMultiplicationFuzz(int testCount)
{
    char funcName[] = "DtTriangleMultiplicationFuzz";
    DT_FUZZ_START(0, testCount, funcName, 0)
    {
        int eleIndex = 0;
        int64_t nSequence = *(int *)DT_SetGetNumberRangeS64(&g_Element[eleIndex++], g_dimRangeInitVal,
            g_dimRangeLowerBound, g_dimRangeUpperBound);
        int64_t nChannel = *(int *)DT_SetGetNumberRangeS64(&g_Element[eleIndex++], g_dimRangeInitVal,
            g_dimRangeLowerBound, g_dimRangeUpperBound);
        int64_t numIntermediateChannel = *(int *)DT_SetGetNumberRangeS64(&g_Element[eleIndex++], g_dimRangeInitVal,
            g_dimRangeLowerBound, g_dimRangeUpperBound);
        int64_t blockSize = *(int *)DT_SetGetNumberRangeS64(&g_Element[eleIndex++], g_dimRangeInitVal,
            g_dimRangeLowerBound, g_dimRangeUpperBound);

        at::Tensor act = torch::randn({nSequence, nSequence, nChannel});
        at::Tensor mask = torch::randn({nSequence, nSequence, 1});
        at::Tensor layerNormWeight = torch::randn({nChannel});
        at::Tensor layerNormBias = torch::randn({nChannel});
        at::Tensor leftProjWeight = torch::randn({numIntermediateChannel, nChannel});
        at::Tensor leftProjBias = torch::randn({numIntermediateChannel});
        at::Tensor rightProjWeight = torch::randn({numIntermediateChannel, nChannel});
        at::Tensor rightProjBias = torch::randn({numIntermediateChannel});
        at::Tensor leftGateWeight = torch::randn({numIntermediateChannel, nChannel});
        at::Tensor leftGateBias = torch::randn({numIntermediateChannel});
        at::Tensor rightGateWeight = torch::randn({numIntermediateChannel, nChannel});
        at::Tensor rightGateBias = torch::randn({numIntermediateChannel});
        at::Tensor outProjWeight = torch::randn({nChannel, numIntermediateChannel});
        at::Tensor outProjBias = torch::randn({nChannel});
        at::Tensor centerLayerNormWeight = torch::randn({numIntermediateChannel});
        at::Tensor centerLayerNormBias = torch::randn({numIntermediateChannel});
        at::Tensor gatingLinearWeight = torch::randn({nChannel, nChannel});
        at::Tensor gatingLinearBias = torch::randn({nChannel});

        alphafold::TriangleMultiplicationForward(act, mask, 1, centerLayerNormWeight, layerNormBias,
            leftProjWeight,leftProjBias, leftGateWeight, leftGateBias, rightProjWeight, rightProjBias,
            rightGateWeight, rightGateBias, centerLayerNormWeight, centerLayerNormBias, outProjWeight,
            outProjBias, gatingLinearWeight, gatingLinearBias, blockSize);
        
        act = torch::randn({nSequence, nSequence, nChannel});
        mask = torch::randn({nSequence, nSequence, 1});
        
        alphafold::TriangleMultiplicationForward(act, mask, 0, centerLayerNormWeight, layerNormBias,
            leftProjWeight,leftProjBias, leftGateWeight, leftGateBias, rightProjWeight, rightProjBias,
            rightGateWeight, rightGateBias, centerLayerNormWeight, centerLayerNormBias, outProjWeight,
            outProjBias, gatingLinearWeight, gatingLinearBias, blockSize);
    }
    DT_FUZZ_END()
}