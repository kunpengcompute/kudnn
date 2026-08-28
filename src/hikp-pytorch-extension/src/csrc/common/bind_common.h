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
#include <torch/extension.h>
#include "linear.h"
#include "MultiHeadAttention.h"
#include "sigmoid.h"
#include "flash_attention.h"
#include "paged_attention.h"
#include "attention.h"

void BindCommonOperator(pybind11::module &m)
{
    auto common = m.def_submodule("common");

    common.def("hikp_linear_fwd", &common::KpexLinear, py::arg("input"),
        py::arg("weight"), py::arg("bias") = std::nullopt);

    common.def("hikp_sigmoid_fwd", &common::HikpSigmoidFwd, py::arg("input"));

    common.def("multi_head_attention_fwd", &common::MultiHeadAttentionFwd, py::arg("qdata"), py::arg("mdata"),
        py::arg("queryWeight"), py::arg("keyWeight"), py::arg("valueWeight"),
        py::arg("logitsBias"), py::arg("nonbatchedBias"),
        py::arg("outputWeight"), py::arg("outputBias"), py::arg("blockSize"));

    common.def("scaled_dot_product_flash_attention_cpu", &common::ScaledDotProductFlashAttention,
        py::arg("query"), py::arg("key"), py::arg("value"),
        py::arg("dropout"), py::arg("isCausal"),
        py::arg("attnMask") = std::nullopt, py::arg("scale") = std::nullopt);

    common.def("scaled_dot_product_paged_attention_cpu", &common::ScaledDotProductPagedAttention,
        py::arg("query"), py::arg("kCache"), py::arg("vCache"), py::arg("blockTables"), py::arg("seqLens"));
    common.def("scaled_dot_product_attention_cpu", &common::ScaledDotProductAttention,
        py::arg("query"), py::arg("key"), py::arg("value"),
        py::arg("dropout"), py::arg("isCausal"),
        py::arg("attnMask") = std::nullopt, py::arg("scale") = std::nullopt);
}
