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
#include "linear.h"
#include "utils/fused_ops.h"
#include "utils/dispatch.h"
#include "utils/blas_wrapper.h"
#include "utils/scalar_type.h"

namespace common {

void LinearImpl(at::Tensor &input, at::Tensor &weight,
    at::Tensor &out, std::optional<at::Tensor> &bias, int64_t rowSize, int64_t colSize, int64_t innerSize)
{
    constexpr int64_t optNumLoops = 100;
    bool applyBias = bias.has_value() && bias.value().numel();
    int64_t innerDim = input.size(-1);
    int64_t inputSize = input.numel() / innerDim;
    int64_t outputSize = weight.size(0);

    const float *inputPtr = TensorConstDataPtr<float>(input);
    const float *weightPtr = TensorConstDataPtr<float>(weight);
    float *outPtr = TensorDataPtr<float>(out);
    float *biasPtr = applyBias ? TensorDataPtr<float>(bias.value()) : nullptr;
    int64_t nThreads = omp_get_max_threads();
    int64_t loops = ((inputSize + rowSize - 1) / rowSize) *
                    ((outputSize + colSize - 1) / colSize) *
                    ((innerDim + innerSize - 1) / innerSize);
    int64_t optNumThreads = std::max(loops / optNumLoops, static_cast<int64_t>(1));
    int64_t ompNumThreads = std::min(nThreads, optNumThreads);

#pragma omp parallel for collapse(2) proc_bind(close) num_threads(ompNumThreads)
    for (int64_t i = 0; i < inputSize; i += rowSize) {
        for (int64_t j = 0; j < outputSize; j += colSize) {
            for (int64_t k = 0; k < innerDim; k += innerSize) {
                int64_t row = std::min(rowSize, inputSize - i);
                int64_t col = std::min(colSize, outputSize - j);
                int64_t inner = std::min(innerSize, innerDim - k);
#ifdef KPEX_920F
                UniGemm(
                    false, true,
                    row,
                    col,
                    inner,
                    static_cast<float>(1.0),
                    inputPtr + i * innerDim + k,
                    innerDim,
                    weightPtr + j * innerDim + k,
                    innerDim,
                    k == 0 ? static_cast<float>(0) : static_cast<float>(1),
                    outPtr + i * outputSize + j,
                    outputSize, 1);
#else
                BlasGemm(
                    CblasNoTrans, CblasTrans,
                    row,
                    col,
                    inner,
                    static_cast<float>(1.0),
                    inputPtr + i * innerDim + k,
                    innerDim,
                    weightPtr + j * innerDim + k,
                    innerDim,
                    k == 0 ? static_cast<float>(0) : static_cast<float>(1),
                    outPtr + i * outputSize + j,
                    outputSize);
#endif
                if (applyBias && k + innerSize >= innerDim) {
                    MatAddVec(row, col, outputSize, biasPtr + j, outPtr + i * outputSize + j);
                }
            }
        }
    }
}

at::Tensor KpexLinear(at::Tensor &input, at::Tensor &weight, std::optional<at::Tensor> &bias)
{
    constexpr int64_t innerSplitThreshold = 512;
    constexpr int64_t innerSplitSize = 512;
    constexpr int64_t colSplitSize = 128;
    constexpr int64_t rowSplitSize = 128;
    input = input.contiguous();
    int64_t innerDim = input.size(-1);
    auto shape = input.sizes().vec();
    shape.back() = weight.size(0);
    at::Tensor out = at::empty(shape, input.options());
    KPEX_DISPATCH_FLOATING(input.scalar_type(), T, [&] {
        if (innerDim >= innerSplitThreshold) {
            LinearImpl(input, weight, out, bias, rowSplitSize, colSplitSize, innerSplitSize);
        } else {
            LinearImpl(input, weight, out, bias, rowSplitSize, colSplitSize, innerDim);
        }
    });

    return out;
}

}