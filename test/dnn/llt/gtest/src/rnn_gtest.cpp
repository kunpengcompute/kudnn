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

#include "gtest/gtest.h"
#include "kudnn.hpp"
#include "kudnn_tensor_data.hpp"

static void SetRnnFwd(KuDNN::RnnCfg &rnn, KuDNN::Element::Type &type)
{
    rnn.nDir = ((rnn.execDir == KuDNN::Detail::ExecutionDirectionT::L2R) ||
        (rnn.execDir == KuDNN::Detail::ExecutionDirectionT::R2L)) ? 1 : 2;
    rnn.weightsLayerLd = rnn.nGates * rnn.dhc;
    rnn.weightsIterLd = rnn.nGates * rnn.dhc;

    rnn.scratchGatesLd = std::max(rnn.dlc, rnn.nGates * rnn.dhc);
    rnn.srcLayerLd = rnn.slc;
    rnn.srcIterLd = rnn.sic;
    rnn.srcIterCLd = 1;
    rnn.dstLayerLd = rnn.dlc;
    rnn.dstIterLd = rnn.dic;
    rnn.dstIterCLd = 1;
    rnn.projHtLd = rnn.dhc;
    rnn.nBias = rnn.nGates + rnn.isLbr;

    rnn.wsStatesLayerLd = std::max(rnn.dlc, std::max(rnn.dhc, rnn.sic));
    rnn.wsStatesIterLd = rnn.wsStatesLayerLd;
    rnn.wsStatesLayerSize = (size_t)(rnn.nLayer + 1) * rnn.nDir * (rnn.nIter + 1) * rnn.mb *
        rnn.wsStatesLayerLd * type.GetSize();
    rnn.scratchGatesNld = rnn.mb;
    rnn.weightsProjectionLd = rnn.dic;
    rnn.wsGatesLd = rnn.dhc * rnn.nGates;
    rnn.wsGatesNld = rnn.mb;
    rnn.alpha = 1;
}

// forward
static void ForwardCopyConstructor1()
{
    const std::vector<KuDNN::TensorInfo> srcInfos = {{{1, 1, 1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCDE}};
    const std::vector<KuDNN::TensorInfo> weightInfos = {{{1, 1, 1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCDE}};
    const std::vector<KuDNN::TensorInfo> dstInfos = {{{1, 1, 1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCDE}};
    KuDNN::TensorInfo biasInfo = {{1, 1, 1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCDE};
    KuDNN::RnnCfg rnn = {KuDNN::Detail::ExecutionDirectionT::L2R};
    rnn.nLayer = 1;
    rnn.nIter = 1;
    rnn.mb = 1;
    rnn.dhc = 1;
    rnn.sic = 1;
    rnn.slc = 1;
    rnn.dic = 1;
    rnn.dlc = 1;
    rnn.nGates = 1;
    KuDNN::Element::Type srcType(KuDNN::Element::TypeT::F32);
    SetRnnFwd(rnn, srcType);

    KuDNN::RnnFWD rnnLayer1(srcInfos, weightInfos, dstInfos, biasInfo,
                           rnn, KuDNN::RnnAlgorithm::RNN, KuDNN::ActivateFunctionRNN::RELU);
    KuDNN::RnnFWD rnnLayer2(rnnLayer1);
    for (size_t i = 0; i < g_Tensor_data.size(); i += 4) {
        const std::vector<KuDNN::TensorInfo> srcInfosLoop = {g_Tensor_data[i]};
        const std::vector<KuDNN::TensorInfo> weightInfosLoop = {g_Tensor_data[i + 1]};
        const std::vector<KuDNN::TensorInfo> dstInfosLoop = {g_Tensor_data[i + 2]};
        KuDNN::TensorInfo biasInfoLoop = g_Tensor_data[i + 3];

        KuDNN::RnnFWD rnnLayer1(srcInfosLoop, weightInfosLoop, dstInfosLoop, biasInfoLoop,
                               rnn, KuDNN::RnnAlgorithm::RNN, KuDNN::ActivateFunctionRNN::RELU);
        KuDNN::RnnFWD rnnLayer2(rnnLayer1);
    }
}

static void ForwardCopyConstructor2()
{
    const std::vector<KuDNN::TensorInfo> srcInfos = {{{1, 1, 1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCDE}};
    const std::vector<KuDNN::TensorInfo> weightInfos = {{{1, 1, 1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCDE}};
    const std::vector<KuDNN::TensorInfo> dstInfos = {{{1, 1, 1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCDE}};
    KuDNN::TensorInfo biasInfo = {{1, 1, 1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCDE};
    KuDNN::RnnCfg rnn = {KuDNN::Detail::ExecutionDirectionT::L2R};
    rnn.nLayer = 1;
    rnn.nIter = 1;
    rnn.mb = 1;
    rnn.dhc = 1;
    rnn.sic = 1;
    rnn.slc = 1;
    rnn.dic = 1;
    rnn.dlc = 1;
    rnn.nGates = 1;
    KuDNN::Element::Type srcType(KuDNN::Element::TypeT::F32);
    SetRnnFwd(rnn, srcType);

    KuDNN::RnnFWD rnnLayer1(srcInfos, weightInfos, dstInfos, biasInfo,
                           rnn, KuDNN::RnnAlgorithm::RNN, KuDNN::ActivateFunctionRNN::RELU);
    KuDNN::RnnFWD rnnLayer2(srcInfos, weightInfos, dstInfos, biasInfo,
                           rnn, KuDNN::RnnAlgorithm::RNN, KuDNN::ActivateFunctionRNN::RELU);
    rnnLayer2 = rnnLayer1;
    for (size_t i = 0; i < g_Tensor_data.size(); i += 4) {
        const std::vector<KuDNN::TensorInfo> srcInfosLoop = {g_Tensor_data[i]};
        const std::vector<KuDNN::TensorInfo> weightInfosLoop = {g_Tensor_data[i + 1]};
        const std::vector<KuDNN::TensorInfo> dstInfosLoop = {g_Tensor_data[i + 2]};
        KuDNN::TensorInfo biasInfoLoop = g_Tensor_data[i + 3];

        KuDNN::RnnFWD rnnLayer1(srcInfosLoop, weightInfosLoop, dstInfosLoop, biasInfoLoop,
                               rnn, KuDNN::RnnAlgorithm::RNN, KuDNN::ActivateFunctionRNN::RELU);
        KuDNN::RnnFWD rnnLayer2(srcInfosLoop, weightInfosLoop, dstInfosLoop, biasInfoLoop,
                               rnn, KuDNN::RnnAlgorithm::RNN, KuDNN::ActivateFunctionRNN::RELU);
        rnnLayer2 = rnnLayer1;
    }
}

static void ForwardMoveConstructor1()
{
    const std::vector<KuDNN::TensorInfo> srcInfos = {{{1, 1, 1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCDE}};
    const std::vector<KuDNN::TensorInfo> weightInfos = {{{1, 1, 1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCDE}};
    const std::vector<KuDNN::TensorInfo> dstInfos = {{{1, 1, 1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCDE}};
    KuDNN::TensorInfo biasInfo = {{1, 1, 1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCDE};
    KuDNN::RnnCfg rnn = {KuDNN::Detail::ExecutionDirectionT::L2R};
    rnn.nLayer = 1;
    rnn.nIter = 1;
    rnn.mb = 1;
    rnn.dhc = 1;
    rnn.sic = 1;
    rnn.slc = 1;
    rnn.dic = 1;
    rnn.dlc = 1;
    rnn.nGates = 1;
    KuDNN::Element::Type srcType(KuDNN::Element::TypeT::F32);
    SetRnnFwd(rnn, srcType);

    KuDNN::RnnFWD rnnLayer1(srcInfos, weightInfos, dstInfos, biasInfo,
                           rnn, KuDNN::RnnAlgorithm::RNN, KuDNN::ActivateFunctionRNN::RELU);
    KuDNN::RnnFWD rnnLayer2(std::move(rnnLayer1));
    for (size_t i = 0; i < g_Tensor_data.size(); i += 4) {
        const std::vector<KuDNN::TensorInfo> srcInfosLoop = {g_Tensor_data[i]};
        const std::vector<KuDNN::TensorInfo> weightInfosLoop = {g_Tensor_data[i + 1]};
        const std::vector<KuDNN::TensorInfo> dstInfosLoop = {g_Tensor_data[i + 2]};
        KuDNN::TensorInfo biasInfoLoop = g_Tensor_data[i + 3];

        KuDNN::RnnFWD rnnLayer1(srcInfosLoop, weightInfosLoop, dstInfosLoop, biasInfoLoop,
                               rnn, KuDNN::RnnAlgorithm::RNN, KuDNN::ActivateFunctionRNN::RELU);
        KuDNN::RnnFWD rnnLayer2(std::move(rnnLayer1));
    }
}

static void ForwardMoveConstructor2()
{
    const std::vector<KuDNN::TensorInfo> srcInfos = {{{1, 1, 1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCDE}};
    const std::vector<KuDNN::TensorInfo> weightInfos = {{{1, 1, 1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCDE}};
    const std::vector<KuDNN::TensorInfo> dstInfos = {{{1, 1, 1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCDE}};
    KuDNN::TensorInfo biasInfo = {{1, 1, 1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCDE};
    KuDNN::RnnCfg rnn = {KuDNN::Detail::ExecutionDirectionT::L2R};
    rnn.nLayer = 1;
    rnn.nIter = 1;
    rnn.mb = 1;
    rnn.dhc = 1;
    rnn.sic = 1;
    rnn.slc = 1;
    rnn.dic = 1;
    rnn.dlc = 1;
    rnn.nGates = 1;
    KuDNN::Element::Type srcType(KuDNN::Element::TypeT::F32);
    SetRnnFwd(rnn, srcType);

    KuDNN::RnnFWD rnnLayer1(srcInfos, weightInfos, dstInfos, biasInfo,
                           rnn, KuDNN::RnnAlgorithm::RNN, KuDNN::ActivateFunctionRNN::RELU);
    KuDNN::RnnFWD rnnLayer2(srcInfos, weightInfos, dstInfos, biasInfo,
                           rnn, KuDNN::RnnAlgorithm::RNN, KuDNN::ActivateFunctionRNN::RELU);
    rnnLayer2 = std::move(rnnLayer1);
    for (size_t i = 0; i < g_Tensor_data.size(); i += 4) {
        const std::vector<KuDNN::TensorInfo> srcInfosLoop = {g_Tensor_data[i]};
        const std::vector<KuDNN::TensorInfo> weightInfosLoop = {g_Tensor_data[i + 1]};
        const std::vector<KuDNN::TensorInfo> dstInfosLoop = {g_Tensor_data[i + 2]};
        KuDNN::TensorInfo biasInfoLoop = g_Tensor_data[i + 3];

        KuDNN::RnnFWD rnnLayer1(srcInfosLoop, weightInfosLoop, dstInfosLoop, biasInfoLoop,
                               rnn, KuDNN::RnnAlgorithm::RNN, KuDNN::ActivateFunctionRNN::RELU);
        KuDNN::RnnFWD rnnLayer2(srcInfosLoop, weightInfosLoop, dstInfosLoop, biasInfoLoop,
                               rnn, KuDNN::RnnAlgorithm::RNN, KuDNN::ActivateFunctionRNN::RELU);
        rnnLayer2 = std::move(rnnLayer1);
    }
}

static void BackwardCopyConstructor1()
{
    const std::vector<KuDNN::TensorInfo> srcInfos = {{{1, 1, 1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCDE}};
    const std::vector<KuDNN::TensorInfo> weightInfos = {{{1, 1, 1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCDE}};
    const std::vector<KuDNN::TensorInfo> dstInfos = {{{1, 1, 1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCDE}};
    KuDNN::TensorInfo biasInfo = {{1, 1, 1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCDE};
    const std::vector<KuDNN::TensorInfo> diffSrcInfos = {{{1, 1, 1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCDE}};
    const std::vector<KuDNN::TensorInfo> diffWeightInfos = {{{1, 1, 1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCDE}};
    const std::vector<KuDNN::TensorInfo> diffDstInfos = {{{1, 1, 1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCDE}};
    KuDNN::TensorInfo diffBiasInfo = {{1, 1, 1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCDE};
    KuDNN::RnnCfg cfg = {KuDNN::Detail::ExecutionDirectionT::L2R};
    cfg.sic = 1;
    cfg.mb = 1;
    cfg.nGates = 1;
    cfg.dhc = 1;
    cfg.weightsIterLd = 1;
    cfg.scratchGatesLd = 1;
    cfg.wsDiffStatesIterLd = 1;
    cfg.nLayer = 1;
    cfg.slc = 1;
    cfg.diffWeightsLayerLd = 1;
    KuDNN::RnnBWD rnnLayer1(srcInfos, weightInfos, dstInfos, biasInfo,
                           diffSrcInfos, diffWeightInfos, diffDstInfos, diffBiasInfo,
                           cfg, KuDNN::RnnAlgorithm::RNN, KuDNN::ActivateFunctionRNN::RELU);
    KuDNN::RnnBWD rnnLayer2(rnnLayer1);
    for (size_t i = 0; i < g_Tensor_data.size(); i += 7) {
        const std::vector<KuDNN::TensorInfo> srcInfosLoop = {g_Tensor_data[i]};
        const std::vector<KuDNN::TensorInfo> weightInfosLoop = {g_Tensor_data[i + 1]};
        const std::vector<KuDNN::TensorInfo> dstInfosLoop = {g_Tensor_data[i + 2]};
        KuDNN::TensorInfo biasInfoLoop = g_Tensor_data[i + 3];
        const std::vector<KuDNN::TensorInfo> diffSrcInfosLoop = {g_Tensor_data[i + 4]};
        const std::vector<KuDNN::TensorInfo> diffWeightInfosLoop = {g_Tensor_data[i + 5]};
        const std::vector<KuDNN::TensorInfo> diffDstInfosLoop = {g_Tensor_data[i + 6]};
        KuDNN::TensorInfo diffBiasInfoLoop = {g_Tensor_data[i + 7]};

        KuDNN::RnnBWD rnnLayer1(srcInfosLoop, weightInfosLoop, dstInfosLoop, biasInfoLoop,
                               diffSrcInfosLoop, diffWeightInfosLoop, diffDstInfosLoop, diffBiasInfoLoop,
                               cfg, KuDNN::RnnAlgorithm::RNN, KuDNN::ActivateFunctionRNN::RELU);
        KuDNN::RnnBWD rnnLayer2(rnnLayer1);
    }
}

static void BackwardCopyConstructor2()
{
    const std::vector<KuDNN::TensorInfo> srcInfos = {{{1, 1, 1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCDE}};
    const std::vector<KuDNN::TensorInfo> weightInfos = {{{1, 1, 1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCDE}};
    const std::vector<KuDNN::TensorInfo> dstInfos = {{{1, 1, 1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCDE}};
    KuDNN::TensorInfo biasInfo = {{1, 1, 1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCDE};
    const std::vector<KuDNN::TensorInfo> diffSrcInfos = {{{1, 1, 1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCDE}};
    const std::vector<KuDNN::TensorInfo> diffWeightInfos = {{{1, 1, 1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCDE}};
    const std::vector<KuDNN::TensorInfo> diffDstInfos = {{{1, 1, 1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCDE}};
    KuDNN::TensorInfo diffBiasInfo = {{1, 1, 1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCDE};
    KuDNN::RnnCfg cfg = {KuDNN::Detail::ExecutionDirectionT::L2R};
    cfg.sic = 1;
    cfg.mb = 1;
    cfg.nGates = 1;
    cfg.dhc = 1;
    cfg.weightsIterLd = 1;
    cfg.scratchGatesLd = 1;
    cfg.wsDiffStatesIterLd = 1;
    cfg.nLayer = 1;
    cfg.slc = 1;
    cfg.diffWeightsLayerLd = 1;
    KuDNN::RnnBWD rnnLayer1(srcInfos, weightInfos, dstInfos, biasInfo,
                           diffSrcInfos, diffWeightInfos, diffDstInfos, diffBiasInfo,
                           cfg, KuDNN::RnnAlgorithm::RNN, KuDNN::ActivateFunctionRNN::RELU);
    KuDNN::RnnBWD rnnLayer2(srcInfos, weightInfos, dstInfos, biasInfo,
                           diffSrcInfos, diffWeightInfos, diffDstInfos, diffBiasInfo,
                           cfg, KuDNN::RnnAlgorithm::RNN, KuDNN::ActivateFunctionRNN::RELU);
    rnnLayer2 = rnnLayer1;
    for (size_t i = 0; i < g_Tensor_data.size(); i += 7) {
        const std::vector<KuDNN::TensorInfo> srcInfosLoop = {g_Tensor_data[i]};
        const std::vector<KuDNN::TensorInfo> weightInfosLoop = {g_Tensor_data[i + 1]};
        const std::vector<KuDNN::TensorInfo> dstInfosLoop = {g_Tensor_data[i + 2]};
        KuDNN::TensorInfo biasInfoLoop = g_Tensor_data[i + 3];
        const std::vector<KuDNN::TensorInfo> diffSrcInfosLoop = {g_Tensor_data[i + 4]};
        const std::vector<KuDNN::TensorInfo> diffWeightInfosLoop = {g_Tensor_data[i + 5]};
        const std::vector<KuDNN::TensorInfo> diffDstInfosLoop = {g_Tensor_data[i + 6]};
        KuDNN::TensorInfo diffBiasInfoLoop = {g_Tensor_data[i + 7]};

        KuDNN::RnnBWD rnnLayer1(srcInfosLoop, weightInfosLoop, dstInfosLoop, biasInfoLoop,
                               diffSrcInfosLoop, diffWeightInfosLoop, diffDstInfosLoop, diffBiasInfoLoop,
                               cfg, KuDNN::RnnAlgorithm::RNN, KuDNN::ActivateFunctionRNN::RELU);
        KuDNN::RnnBWD rnnLayer2(srcInfosLoop, weightInfosLoop, dstInfosLoop, biasInfoLoop,
                               diffSrcInfosLoop, diffWeightInfosLoop, diffDstInfosLoop, diffBiasInfoLoop,
                               cfg, KuDNN::RnnAlgorithm::RNN, KuDNN::ActivateFunctionRNN::RELU);
        rnnLayer2 = rnnLayer1;
    }
}

static void BackwardMoveConstructor1()
{
    const std::vector<KuDNN::TensorInfo> srcInfos = {{{1, 1, 1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCDE}};
    const std::vector<KuDNN::TensorInfo> weightInfos = {{{1, 1, 1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCDE}};
    const std::vector<KuDNN::TensorInfo> dstInfos = {{{1, 1, 1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCDE}};
    KuDNN::TensorInfo biasInfo = {{1, 1, 1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCDE};
    const std::vector<KuDNN::TensorInfo> diffSrcInfos = {{{1, 1, 1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCDE}};
    const std::vector<KuDNN::TensorInfo> diffWeightInfos = {{{1, 1, 1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCDE}};
    const std::vector<KuDNN::TensorInfo> diffDstInfos = {{{1, 1, 1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCDE}};
    KuDNN::TensorInfo diffBiasInfo = {{1, 1, 1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCDE};
    KuDNN::RnnCfg cfg = {KuDNN::Detail::ExecutionDirectionT::L2R};
    cfg.sic = 1;
    cfg.mb = 1;
    cfg.nGates = 1;
    cfg.dhc = 1;
    cfg.weightsIterLd = 1;
    cfg.scratchGatesLd = 1;
    cfg.wsDiffStatesIterLd = 1;
    cfg.nLayer = 1;
    cfg.slc = 1;
    cfg.diffWeightsLayerLd = 1;
    KuDNN::RnnBWD rnnLayer1(srcInfos, weightInfos, dstInfos, biasInfo,
                           diffSrcInfos, diffWeightInfos, diffDstInfos, diffBiasInfo,
                           cfg, KuDNN::RnnAlgorithm::RNN, KuDNN::ActivateFunctionRNN::RELU);
    KuDNN::RnnBWD rnnLayer2(std::move(rnnLayer1));
    for (size_t i = 0; i < g_Tensor_data.size(); i += 7) {
        const std::vector<KuDNN::TensorInfo> srcInfosLoop = {g_Tensor_data[i]};
        const std::vector<KuDNN::TensorInfo> weightInfosLoop = {g_Tensor_data[i + 1]};
        const std::vector<KuDNN::TensorInfo> dstInfosLoop = {g_Tensor_data[i + 2]};
        KuDNN::TensorInfo biasInfoLoop = g_Tensor_data[i + 3];
        const std::vector<KuDNN::TensorInfo> diffSrcInfosLoop = {g_Tensor_data[i + 4]};
        const std::vector<KuDNN::TensorInfo> diffWeightInfosLoop = {g_Tensor_data[i + 5]};
        const std::vector<KuDNN::TensorInfo> diffDstInfosLoop = {g_Tensor_data[i + 6]};
        KuDNN::TensorInfo diffBiasInfoLoop = {g_Tensor_data[i + 7]};

        KuDNN::RnnBWD rnnLayer1(srcInfosLoop, weightInfosLoop, dstInfosLoop, biasInfoLoop,
                               diffSrcInfosLoop, diffWeightInfosLoop, diffDstInfosLoop, diffBiasInfoLoop,
                               cfg, KuDNN::RnnAlgorithm::RNN, KuDNN::ActivateFunctionRNN::RELU);
        KuDNN::RnnBWD rnnLayer2(std::move(rnnLayer1));
    }
}

static void BackwardMoveConstructor2()
{
    const std::vector<KuDNN::TensorInfo> srcInfos = {{{1, 1, 1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCDE}};
    const std::vector<KuDNN::TensorInfo> weightInfos = {{{1, 1, 1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCDE}};
    const std::vector<KuDNN::TensorInfo> dstInfos = {{{1, 1, 1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCDE}};
    KuDNN::TensorInfo biasInfo = {{1, 1, 1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCDE};
    const std::vector<KuDNN::TensorInfo> diffSrcInfos = {{{1, 1, 1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCDE}};
    const std::vector<KuDNN::TensorInfo> diffWeightInfos = {{{1, 1, 1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCDE}};
    const std::vector<KuDNN::TensorInfo> diffDstInfos = {{{1, 1, 1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCDE}};
    KuDNN::TensorInfo diffBiasInfo = {{1, 1, 1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCDE};
        KuDNN::RnnCfg cfg = {KuDNN::Detail::ExecutionDirectionT::L2R};
    cfg.sic = 1;
    cfg.mb = 1;
    cfg.nGates = 1;
    cfg.dhc = 1;
    cfg.weightsIterLd = 1;
    cfg.scratchGatesLd = 1;
    cfg.wsDiffStatesIterLd = 1;
    cfg.nLayer = 1;
    cfg.slc = 1;
    cfg.diffWeightsLayerLd = 1;
    KuDNN::RnnBWD rnnLayer1(srcInfos, weightInfos, dstInfos, biasInfo,
                           diffSrcInfos, diffWeightInfos, diffDstInfos, diffBiasInfo,
                           cfg, KuDNN::RnnAlgorithm::RNN, KuDNN::ActivateFunctionRNN::RELU);
    KuDNN::RnnBWD rnnLayer2(srcInfos, weightInfos, dstInfos, biasInfo,
                           diffSrcInfos, diffWeightInfos, diffDstInfos, diffBiasInfo,
                           cfg, KuDNN::RnnAlgorithm::RNN, KuDNN::ActivateFunctionRNN::RELU);
    rnnLayer2 = std::move(rnnLayer1);
    for (size_t i = 0; i < g_Tensor_data.size(); i += 7) {
        const std::vector<KuDNN::TensorInfo> srcInfosLoop = {g_Tensor_data[i]};
        const std::vector<KuDNN::TensorInfo> weightInfosLoop = {g_Tensor_data[i + 1]};
        const std::vector<KuDNN::TensorInfo> dstInfosLoop = {g_Tensor_data[i + 2]};
        KuDNN::TensorInfo biasInfoLoop = g_Tensor_data[i + 3];
        const std::vector<KuDNN::TensorInfo> diffSrcInfosLoop = {g_Tensor_data[i + 4]};
        const std::vector<KuDNN::TensorInfo> diffWeightInfosLoop = {g_Tensor_data[i + 5]};
        const std::vector<KuDNN::TensorInfo> diffDstInfosLoop = {g_Tensor_data[i + 6]};
        KuDNN::TensorInfo diffBiasInfoLoop = {g_Tensor_data[i + 7]};

        KuDNN::RnnBWD rnnLayer1(srcInfosLoop, weightInfosLoop, dstInfosLoop, biasInfoLoop,
                               diffSrcInfosLoop, diffWeightInfosLoop, diffDstInfosLoop, diffBiasInfoLoop,
                               cfg, KuDNN::RnnAlgorithm::RNN, KuDNN::ActivateFunctionRNN::RELU);
        KuDNN::RnnBWD rnnLayer2(srcInfosLoop, weightInfosLoop, dstInfosLoop, biasInfoLoop,
                               diffSrcInfosLoop, diffWeightInfosLoop, diffDstInfosLoop, diffBiasInfoLoop,
                               cfg, KuDNN::RnnAlgorithm::RNN, KuDNN::ActivateFunctionRNN::RELU);
    rnnLayer2 = std::move(rnnLayer1);
    }
}

// test cases
TEST(RnnGTests, FWDCopyConstructor1)
{
    EXPECT_NO_THROW(ForwardCopyConstructor1());
}

TEST(RnnGTests, FWDCopyConstructor2)
{
    EXPECT_NO_THROW(ForwardCopyConstructor2());
}

TEST(RnnGTests, FWDMoveConstructor1)
{
    EXPECT_NO_THROW(ForwardMoveConstructor1());
}

TEST(RnnGTests, FWDMoveConstructor2)
{
    EXPECT_NO_THROW(ForwardMoveConstructor2());
}

TEST(RnnGTests, BWDCopyConstructor1)
{
    EXPECT_NO_THROW(BackwardCopyConstructor1());
}

TEST(RnnGTests, BWDCopyConstructor2)
{
    EXPECT_NO_THROW(BackwardCopyConstructor2());
}

TEST(RnnGTests, BWDMoveConstructor1)
{
    EXPECT_NO_THROW(BackwardMoveConstructor1());
}

TEST(RnnGTests, BWDMoveConstructor2)
{
    EXPECT_NO_THROW(BackwardMoveConstructor2());
}
