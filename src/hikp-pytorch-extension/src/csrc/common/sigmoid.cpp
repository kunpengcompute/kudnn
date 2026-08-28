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
#include "vla.h"
#include "sigmoid.h"
#include "fused_ops.h"

namespace common {

at::Tensor HikpSigmoidFwd(at::Tensor &input)
{
    int64_t numel = input.numel();

    if (numel == 0) {
        return input;
    }

    int64_t numelThreshold = 1e6;
    int64_t numOfThreadsMax = omp_get_max_threads();
    int64_t numOfThreadsOpt = std::max(numel / numelThreshold, 1L);
    int64_t numOfThreads = std::min(numOfThreadsMax, numOfThreadsOpt);
    int64_t everyElem = numel / numOfThreads + 1;

    auto output = input.new_empty(input.sizes(), input.options());
    auto inputPtr = GetVLAPtr<float>(input);
    auto outputPtr = GetVLAPtr<float>(output);

#pragma omp parallel for num_threads(numOfThreads)
    for (int64_t i = 0 ; i < numel; i += everyElem) {
        int64_t num = std::min(everyElem, numel - i);
        SigMoid(&inputPtr[i], num, &outputPtr[i]);
    }

    return output;
}

}