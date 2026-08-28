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
#include "paged_attention.h"
#include "fuzz_common.h"
#include "fuzz_test_interface.h"

static constexpr int64_t g_batchRangeInitVal = 2;
static constexpr int64_t g_batchRangeLowerBound = 1;
static constexpr int64_t g_batchRangeUpperBound = 10;

static constexpr int64_t g_headRangeInitVal = 4;
static constexpr int64_t g_headRangeLowerBound = 1;
static constexpr int64_t g_headRangeUpperBound = 8;

void DtPagedAttentionFuzz(int testCount)
{
    char funcName[] = "DtPagedAttentionFuzz";
    DT_FUZZ_START(0, testCount, funcName, 0)
    {
        int eleIndex = 0;
        constexpr int64_t blockSize = 16;
        constexpr int64_t maxNumBlocksPerSeq = 65;
        constexpr int64_t numBlocks = 32;
        constexpr int64_t numElem = 8;
        constexpr int64_t headSize = 128;

        int64_t batch = *(int *)DT_SetGetNumberRangeS64(&g_Element[eleIndex++], g_batchRangeInitVal,
            g_batchRangeLowerBound, g_batchRangeUpperBound);
        int64_t nHeads = *(int *)DT_SetGetNumberRangeS64(&g_Element[eleIndex++], g_headRangeInitVal,
            g_headRangeLowerBound, g_headRangeUpperBound);

        at::Tensor query = torch::randn({batch, nHeads, headSize}, at::kHalf);
        at::Tensor kCaches = torch::randn({numBlocks, nHeads, headSize / numElem, blockSize, numElem}, at::kHalf);
        at::Tensor vCaches = torch::randn({numBlocks, nHeads, headSize, blockSize}, at::kHalf);
        at::Tensor seqLensLst = torch::randint(512, 1024, {batch}, at::kInt);
        at::Tensor blockTable = torch::randint(0, numBlocks, {batch, maxNumBlocksPerSeq}, at::kInt);

        common::ScaledDotProductPagedAttention(query, kCaches, vCaches, blockTable, seqLensLst);
    }
    DT_FUZZ_END()
}