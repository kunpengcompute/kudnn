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

#pragma once
#include <ATen/core/Tensor.h>
#include <torch/extension.h>

namespace alphafold {
at::Tensor TriangleMultiplicationForward(at::Tensor &act, at::Tensor &mask,
    int equation, at::Tensor &layerNormWeight, at::Tensor &layerNormBias,
    at::Tensor &leftProjWeight, at::Tensor &leftProjBias, at::Tensor &leftGateWeight, at::Tensor &leftGateBias,
    at::Tensor &rightProjWeight, at::Tensor &rightProjBias, at::Tensor &rightGateWeight, at::Tensor &rightGateBias,
    at::Tensor &centerLayerNormWeight, at::Tensor &centerLayerNormBias, at::Tensor &outputProjWeight,
    at::Tensor &outputProjBias, at::Tensor &gatingLinearWeight, at::Tensor &gatingLinearBias, int64_t blockSize);
}