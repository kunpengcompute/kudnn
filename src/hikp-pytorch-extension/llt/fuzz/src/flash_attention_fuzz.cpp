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
#include "flash_attention.h"
#include "fuzz_common.h"
#include "fuzz_test_interface.h"

static constexpr int64_t g_batchRangeInitVal = 2;
static constexpr int64_t g_batchRangeLowerBound = 1;
static constexpr int64_t g_batchRangeUpperBound = 10;

static constexpr int64_t g_dimRangeInitVal = 4;
static constexpr int64_t g_dimRangeLowerBound = 1;
static constexpr int64_t g_dimRangeUpperBound = 30;

static constexpr int64_t g_headRangeInitVal = 4;
static constexpr int64_t g_headRangeLowerBound = 1;
static constexpr int64_t g_headRangeUpperBound = 8;

void DtFlashAttentionTestTemplate(int64_t min, int64_t max, int &eleIndex)
{
    int64_t batch = *(int *)DT_SetGetNumberRangeS64(&g_Element[eleIndex++], g_batchRangeInitVal,
        g_batchRangeLowerBound, g_batchRangeUpperBound);
    int64_t nHeads = *(int *)DT_SetGetNumberRangeS64(&g_Element[eleIndex++], g_headRangeInitVal,
        g_headRangeLowerBound, g_headRangeUpperBound);
    int64_t qSeqLen = *(int *)DT_SetGetNumberRangeS64(&g_Element[eleIndex++], min, min, max);
    int64_t kvSeqLen = *(int *)DT_SetGetNumberRangeS64(&g_Element[eleIndex++], min, min, max);
    int64_t headSize = *(int *)DT_SetGetNumberRangeS64(&g_Element[eleIndex++], g_dimRangeInitVal,
        g_dimRangeLowerBound, g_dimRangeUpperBound);
    std::vector<at::ScalarType> types = {at::kFloat, at::kHalf};
    int n = types.size();
    auto scale = std::make_optional<double>(0.0);

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            at::Tensor q = torch::randn({batch, nHeads, qSeqLen, headSize}, types[i]);
            at::Tensor k = torch::randn({batch, nHeads, kvSeqLen, headSize}, types[i]);
            at::Tensor v = torch::randn({batch, nHeads, kvSeqLen, headSize}, types[i]);
            auto attnMask = std::make_optional<at::Tensor>(torch::randn({batch, nHeads, qSeqLen, kvSeqLen}, types[j]));
            common::ScaledDotProductFlashAttention(q, k, v, 0.0, false, attnMask, scale);
            common::ScaledDotProductFlashAttention(q, k, v, 0.0, true, attnMask, scale);
        }
    }
}

void DtFlashAttentionFuzz(int testCount)
{
    char funcName[] = "DtFlashAttentionFuzz";
    DT_FUZZ_START(0, testCount, funcName, 0)
    {
        int eleIndex = 0;
        DtFlashAttentionTestTemplate(16, 192, eleIndex);
        DtFlashAttentionTestTemplate(256, 512, eleIndex);
        DtFlashAttentionTestTemplate(768, 1024, eleIndex);
    }
    DT_FUZZ_END()
}