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
#include "gating_attention.h"
#include "outer_product_mean.h"
#include "triangle_multiplication.h"

void BindAlphafoldOperator(pybind11::module &m)
{
    auto alphafold = m.def_submodule("alphafold");

    alphafold.def("gating_attention_fwd", &alphafold::GatingAttentionFwd, py::arg("qdata"), py::arg("mdata"),
        py::arg("queryWeight"), py::arg("keyWeight"), py::arg("valueWeight"),
        py::arg("logitsBias"), py::arg("nonbatchedBias"),
        py::arg("gatingWeight"), py::arg("gatingBias"),
        py::arg("outputWeight"), py::arg("outputBias"), py::arg("blockSize"));

    alphafold.def("outer_product_mean_fwd", &alphafold::OuterProductMean, py::arg("act"), py::arg("mask"),
        py::arg("sequenceLen"), py::arg("layerNormWeight"), py::arg("layerNormBias"),
        py::arg("leftProjWeight"), py::arg("leftProjBias"),
        py::arg("rightProjWeight"), py::arg("rightProjBias"),
        py::arg("outputProjWeight"), py::arg("outputProjBias"),
        py::arg("blockSize"));

    alphafold.def("triangle_multiplication_fwd", &alphafold::TriangleMultiplicationForward,
        py::arg("act"), py::arg("mask"), py::arg("equation"), py::arg("layerNormWeight"),
        py::arg("layerNormBias"), py::arg("leftProjWeight"), py::arg("leftProjBias"), py::arg("leftGateWeight"),
        py::arg("leftGateBias"), py::arg("rightProjWeight"), py::arg("rightProjBias"), py::arg("rightGateWeight"),
        py::arg("rightGateBias"), py::arg("centerLayerNormWeight"), py::arg("centerLayerNormBias"),
        py::arg("outputProjWeight"), py::arg("outputProjBias"),
        py::arg("gatingLinearWeight"), py::arg("gatingLinearBias"), py::arg("block_size"));
}