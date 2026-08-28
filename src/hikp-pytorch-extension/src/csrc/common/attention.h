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

#include <ATen/core/Tensor.h>
#include <torch/extension.h>

namespace common {

std::tuple<at::Tensor, at::Tensor> ScaledDotProductAttention(
    const at::Tensor &query, const at::Tensor &key, const at::Tensor &value,
    double dropout = 0.0, bool isCausal = false,
    const std::optional<at::Tensor> &attnMask = std::nullopt, std::optional<double> scale = std::nullopt);

}
