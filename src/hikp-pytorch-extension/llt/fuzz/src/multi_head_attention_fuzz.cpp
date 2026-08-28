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
#include "MultiHeadAttention.h"
#include "fuzz_common.h"
#include "fuzz_test_interface.h"
 
static constexpr int64_t g_dimRangeInitVal = 4;
static constexpr int64_t g_dimRangeLowerBound = 1;
static constexpr int64_t g_dimRangeUpperBound = 30;
 
static constexpr int64_t g_headRangeInitVal = 4;
static constexpr int64_t g_headRangeLowerBound = 1;
static constexpr int64_t g_headRangeUpperBound = 8;
 
void DtMultiHeadAttentionFuzz(int testCount)
{
    char funcName[] = "DtMultiHeadAttentionFuzz";
    DT_FUZZ_START(0, testCount, funcName, 0)
    {
        int eleIndex = 0;
        int64_t nBatch = *(int *)DT_SetGetNumberRangeS64(&g_Element[eleIndex++], g_dimRangeInitVal,
            g_dimRangeLowerBound, g_dimRangeUpperBound);
        int64_t nSequence = *(int *)DT_SetGetNumberRangeS64(&g_Element[eleIndex++], g_dimRangeInitVal,
            g_dimRangeLowerBound, g_dimRangeUpperBound);
        int64_t nChannel = *(int *)DT_SetGetNumberRangeS64(&g_Element[eleIndex++], g_dimRangeInitVal,
            g_dimRangeLowerBound, g_dimRangeUpperBound);
        int64_t nHead = *(int *)DT_SetGetNumberRangeS64(&g_Element[eleIndex++], g_headRangeInitVal,
            g_headRangeLowerBound, g_headRangeUpperBound);
        int64_t keyDim = *(int *)DT_SetGetNumberRangeS64(&g_Element[eleIndex++], g_dimRangeInitVal,
            g_dimRangeLowerBound, g_dimRangeUpperBound);
        int64_t valueDim = *(int *)DT_SetGetNumberRangeS64(&g_Element[eleIndex++], g_dimRangeInitVal,
            g_dimRangeLowerBound, g_dimRangeUpperBound);
        int64_t outputDim = *(int *)DT_SetGetNumberRangeS64(&g_Element[eleIndex++], g_dimRangeInitVal,
            g_dimRangeLowerBound, g_dimRangeUpperBound);
        int64_t blockSize = *(int *)DT_SetGetNumberRangeS64(&g_Element[eleIndex++], g_dimRangeInitVal,
            g_dimRangeLowerBound, g_dimRangeUpperBound);
        
        at::Tensor qData = torch::randn({nBatch, nSequence, nChannel});
        at::Tensor mData = torch::randn({nBatch, nSequence, nChannel});
        at::Tensor bias = torch::randn({nBatch, 1, 1, nSequence});
        at::Tensor nonbatchedBias = torch::randn({nHead, nSequence, nSequence});
        at::Tensor queryW = torch::randn({nChannel, nHead, keyDim});
        at::Tensor keyW = torch::randn({nChannel, nHead, keyDim});
        at::Tensor valueW = torch::randn({nChannel, nHead, valueDim});
        at::Tensor outputW = torch::randn({nHead, valueDim, outputDim});
        at::Tensor outputBias = torch::randn({outputDim});
 
        common::MultiHeadAttentionFwd(qData, mData, queryW, keyW, valueW, bias,
            nonbatchedBias, outputW, outputBias, blockSize);
    }
    DT_FUZZ_END()
}