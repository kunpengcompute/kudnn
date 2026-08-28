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

// forward
static void ForwardCopyConstructor1()
{
    const KuDNN::TensorInfo srcTensor = {{1, 1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCD};
    const KuDNN::TensorInfo weightsTensor = {{1, 1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCD};
    const KuDNN::TensorInfo dstTensor = {{1, 1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCD};
    const KuDNN::TensorInfo biasTensor = {{1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::A};
    KuDNN::Shape strides(1, 1);
    KuDNN::Shape dilates(0, 0);
    KuDNN::Shape paddingL(0, 0);
    KuDNN::Shape paddingR(0, 0);
    KuDNN::DeconvolutionAlgorithm alg(KuDNN::DeconvolutionAlgorithm::DIRECT);
    KuDNN::DeconvolutionLayerFWD deconvFwdLayer1(srcTensor, weightsTensor, dstTensor, biasTensor, strides, dilates,
        paddingL, paddingR, alg);
    KuDNN::DeconvolutionLayerFWD deconvFwdLayer2(deconvFwdLayer1);
    for (size_t i = 0; i < g_Tensor_data.size() / 4; i += 4) {
        KuDNN::DeconvolutionLayerFWD deconvFwdLayer1(g_Tensor_data[i], g_Tensor_data[i + 1], g_Tensor_data[i + 2],
            g_Tensor_data[i + 3], strides, dilates, paddingL, paddingR, alg);
        KuDNN::DeconvolutionLayerFWD deconvFwdLayer2(deconvFwdLayer1);
    }
}

static void ForwardCopyConstructor2()
{
    const KuDNN::TensorInfo srcTensor = {{1, 1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCD};
    const KuDNN::TensorInfo weightsTensor = {{1, 1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCD};
    const KuDNN::TensorInfo dstTensor = {{1, 1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCD};
    const KuDNN::TensorInfo biasTensor = {{1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::A};
    KuDNN::Shape strides(1, 1);
    KuDNN::Shape paddingL(0, 0);
    KuDNN::Shape paddingR(0, 0);
    KuDNN::DeconvolutionAlgorithm alg(KuDNN::DeconvolutionAlgorithm::DIRECT);
    KuDNN::DeconvolutionLayerFWD deconvFwdLayer1(srcTensor, weightsTensor, dstTensor, biasTensor, strides,
        paddingL, paddingR, alg);
    KuDNN::DeconvolutionLayerFWD deconvFwdLayer2(srcTensor, weightsTensor, dstTensor, biasTensor, strides,
        paddingL, paddingR, alg);
    deconvFwdLayer2 = deconvFwdLayer1;
    for (size_t i = 0; i < g_Tensor_data.size() / 4; i += 4) {
        KuDNN::DeconvolutionLayerFWD deconvFwdLayer1(g_Tensor_data[i], g_Tensor_data[i + 1], g_Tensor_data[i + 2],
            g_Tensor_data[i + 3], strides, paddingL, paddingR, alg);
        KuDNN::DeconvolutionLayerFWD deconvFwdLayer2(g_Tensor_data[i], g_Tensor_data[i + 1], g_Tensor_data[i + 2],
            g_Tensor_data[i + 3], strides, paddingL, paddingR, alg);
        deconvFwdLayer2 = deconvFwdLayer1;
    }
}

static void ForwardMoveConstructor1()
{
    const KuDNN::TensorInfo srcTensor = {{1, 1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCD};
    const KuDNN::TensorInfo weightsTensor = {{1, 1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCD};
    const KuDNN::TensorInfo dstTensor = {{1, 1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCD};
    const KuDNN::TensorInfo biasTensor = {{1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::A};
    KuDNN::Shape strides(1, 1);
    KuDNN::Shape dilates(0, 0);
    KuDNN::Shape paddingL(0, 0);
    KuDNN::Shape paddingR(0, 0);
    KuDNN::DeconvolutionAlgorithm alg(KuDNN::DeconvolutionAlgorithm::DIRECT);
    KuDNN::DeconvolutionLayerFWD deconvFwdLayer1(srcTensor, weightsTensor, dstTensor, biasTensor, strides, dilates,
        paddingL, paddingR, alg);
    KuDNN::DeconvolutionLayerFWD deconvFwdLayer2(std::move(deconvFwdLayer1));
    for (size_t i = 0; i < g_Tensor_data.size() / 4; i += 4) {
        KuDNN::DeconvolutionLayerFWD deconvFwdLayer1(g_Tensor_data[i], g_Tensor_data[i + 1], g_Tensor_data[i + 2], g_Tensor_data[i + 3],
            strides, dilates, paddingL, paddingR, alg);
        KuDNN::DeconvolutionLayerFWD deconvFwdLayer2(std::move(deconvFwdLayer1));
    }
}

static void ForwardMoveConstructor2()
{
    const KuDNN::TensorInfo srcTensor = {{1, 1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCD};
    const KuDNN::TensorInfo weightsTensor = {{1, 1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCD};
    const KuDNN::TensorInfo dstTensor = {{1, 1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCD};
    const KuDNN::TensorInfo biasTensor = {{1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::A};
    KuDNN::Shape strides(1, 1);
    KuDNN::Shape dilates(0, 0);
    KuDNN::Shape paddingL(0, 0);
    KuDNN::Shape paddingR(0, 0);
    KuDNN::DeconvolutionAlgorithm alg(KuDNN::DeconvolutionAlgorithm::DIRECT);
    KuDNN::DeconvolutionLayerFWD deconvFwdLayer1(srcTensor, weightsTensor, dstTensor, biasTensor, strides, dilates,
        paddingL, paddingR, alg);
    KuDNN::DeconvolutionLayerFWD deconvFwdLayer2(srcTensor, weightsTensor, dstTensor, biasTensor, strides, dilates,
        paddingL, paddingR, alg);
    deconvFwdLayer2 = std::move(deconvFwdLayer1);
    for (size_t i = 0; i < g_Tensor_data.size() / 4; i += 4) {
        KuDNN::DeconvolutionLayerFWD deconvFwdLayer1(g_Tensor_data[i], g_Tensor_data[i + 1], g_Tensor_data[i + 2], g_Tensor_data[i + 3],
            strides, dilates, paddingL, paddingR, alg);
        KuDNN::DeconvolutionLayerFWD deconvFwdLayer2(g_Tensor_data[i], g_Tensor_data[i + 1], g_Tensor_data[i + 2], g_Tensor_data[i + 3],
            strides, dilates, paddingL, paddingR, alg);
        deconvFwdLayer2 = std::move(deconvFwdLayer1);
    }
}

// backward data
static void BackwardDataCopyConstructor1()
{
    const KuDNN::TensorInfo diffSrcTensor = {{1, 1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCD};
    const KuDNN::TensorInfo weightsTensor = {{1, 1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCD};
    const KuDNN::TensorInfo diffDstTensor = {{1, 1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCD};
    KuDNN::Shape strides(1, 1);
    KuDNN::Shape dilates(0, 0);
    KuDNN::Shape paddingL(0, 0);
    KuDNN::Shape paddingR(0, 0);
    KuDNN::DeconvolutionAlgorithm alg(KuDNN::DeconvolutionAlgorithm::DIRECT);
    KuDNN::DeconvolutionLayerBWDData deconvBwdLayer1(diffDstTensor, weightsTensor, diffSrcTensor, strides, dilates,
        paddingL, paddingR, alg);
    KuDNN::DeconvolutionLayerBWDData deconvBwdLayer2(deconvBwdLayer1);
    for (size_t i = 0; i < g_Tensor_data.size() / 3; i += 3) {
        KuDNN::DeconvolutionLayerBWDData deconvBwdLayer1(g_Tensor_data[i], g_Tensor_data[i + 1], g_Tensor_data[i + 2],
            strides, dilates, paddingL, paddingR, alg);
        KuDNN::DeconvolutionLayerBWDData deconvBwdLayer2(deconvBwdLayer1);
    }
}

static void BackwardDataCopyConstructor2()
{
    const KuDNN::TensorInfo diffSrcTensor = {{1, 1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCD};
    const KuDNN::TensorInfo weightsTensor = {{1, 1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCD};
    const KuDNN::TensorInfo diffDstTensor = {{1, 1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCD};
    KuDNN::Shape strides(1, 1);
    KuDNN::Shape paddingL(0, 0);
    KuDNN::Shape paddingR(0, 0);
    KuDNN::DeconvolutionAlgorithm alg(KuDNN::DeconvolutionAlgorithm::DIRECT);
    KuDNN::DeconvolutionLayerBWDData deconvBwdLayer1(diffDstTensor, weightsTensor, diffSrcTensor, strides,
        paddingL, paddingR, alg);
    KuDNN::DeconvolutionLayerBWDData deconvBwdLayer2(deconvBwdLayer1);
    deconvBwdLayer2 = deconvBwdLayer1;
    // deconvBwdLayer2 = deconvBwdLayer1;
    for (size_t i = 0; i < g_Tensor_data.size() / 3; i += 3) {
        KuDNN::DeconvolutionLayerBWDData deconvBwdLayer1(g_Tensor_data[i], g_Tensor_data[i + 1], g_Tensor_data[i + 2],
            strides, paddingL, paddingR, alg);
        KuDNN::DeconvolutionLayerBWDData deconvBwdLayer2(deconvBwdLayer1);
        deconvBwdLayer2 = deconvBwdLayer1;
    }
}

static void BackwardDataMoveConstructor1()
{
    const KuDNN::TensorInfo diffSrcTensor = {{1, 1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCD};
    const KuDNN::TensorInfo weightsTensor = {{1, 1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCD};
    const KuDNN::TensorInfo diffDstTensor = {{1, 1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCD};
    KuDNN::Shape strides(1, 1);
    KuDNN::Shape dilates(0, 0);
    KuDNN::Shape paddingL(0, 0);
    KuDNN::Shape paddingR(0, 0);
    KuDNN::DeconvolutionAlgorithm alg(KuDNN::DeconvolutionAlgorithm::DIRECT);
    KuDNN::DeconvolutionLayerBWDData deconvBwdLayer1(diffDstTensor, weightsTensor, diffSrcTensor, strides, dilates,
        paddingL, paddingR, alg);
    KuDNN::DeconvolutionLayerBWDData deconvBwdLayer2(std::move(deconvBwdLayer1));
    for (size_t i = 0; i < g_Tensor_data.size() / 3; i += 3) {
        KuDNN::DeconvolutionLayerBWDData deconvBwdLayer1(g_Tensor_data[i], g_Tensor_data[i + 1], g_Tensor_data[i + 2],
            strides, dilates, paddingL, paddingR, alg);
        KuDNN::DeconvolutionLayerBWDData deconvBwdLayer2(std::move(deconvBwdLayer1));
    }
}

static void BackwardDataMoveConstructor2()
{
    const KuDNN::TensorInfo diffSrcTensor = {{1, 1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCD};
    const KuDNN::TensorInfo weightsTensor = {{1, 1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCD};
    const KuDNN::TensorInfo diffDstTensor = {{1, 1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCD};
    KuDNN::Shape strides(1, 1);
    KuDNN::Shape dilates(0, 0);
    KuDNN::Shape paddingL(0, 0);
    KuDNN::Shape paddingR(0, 0);
    KuDNN::DeconvolutionAlgorithm alg(KuDNN::DeconvolutionAlgorithm::DIRECT);
    KuDNN::DeconvolutionLayerBWDData deconvBwdLayer1(diffDstTensor, weightsTensor, diffSrcTensor, strides, dilates,
        paddingL, paddingR, alg);
    KuDNN::DeconvolutionLayerBWDData deconvBwdLayer2(std::move(deconvBwdLayer1));
    deconvBwdLayer2 = std::move(deconvBwdLayer1);
    for (size_t i = 0; i < g_Tensor_data.size() / 3; i += 3) {
        KuDNN::DeconvolutionLayerBWDData deconvBwdLayer1(g_Tensor_data[i], g_Tensor_data[i + 1], g_Tensor_data[i + 2],
            strides, dilates, paddingL, paddingR, alg);
        KuDNN::DeconvolutionLayerBWDData deconvBwdLayer2(std::move(deconvBwdLayer1));
        deconvBwdLayer2 = std::move(deconvBwdLayer1);
    }
}

// backward weights
static void BackwardWeightsCopyConstructor1()
{
    const KuDNN::TensorInfo srcTensor = {{1, 1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCD};
    const KuDNN::TensorInfo diffWeightsTensor = {{1, 1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCD};
    const KuDNN::TensorInfo diffDstTensor = {{1, 1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCD};
    const KuDNN::TensorInfo diffBiasTensor = {{1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::A};
    KuDNN::Shape strides(1, 1);
    KuDNN::Shape dilates(0, 0);
    KuDNN::Shape paddingL(0, 0);
    KuDNN::Shape paddingR(0, 0);
    KuDNN::DeconvolutionAlgorithm alg(KuDNN::DeconvolutionAlgorithm::DIRECT);
    KuDNN::DeconvolutionLayerBWDWeights deconvBwdWeightsLayer1(diffDstTensor, srcTensor, diffWeightsTensor,
        diffBiasTensor, strides, dilates, paddingL, paddingR, alg);
    KuDNN::DeconvolutionLayerBWDWeights deconvBwdWeightsLayer2(deconvBwdWeightsLayer1);
    for (size_t i = 0; i < g_Tensor_data.size() / 4; i += 4) {
        KuDNN::DeconvolutionLayerBWDWeights deconvBwdWeightsLayer1(g_Tensor_data[i], g_Tensor_data[i + 1], g_Tensor_data[i + 2],
             g_Tensor_data[i + 3], strides, dilates, paddingL, paddingR, alg);
        KuDNN::DeconvolutionLayerBWDWeights deconvBwdWeightsLayer2(deconvBwdWeightsLayer1);
    }
}

static void BackwardWeightsCopyConstructor2()
{
    const KuDNN::TensorInfo srcTensor = {{1, 1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCD};
    const KuDNN::TensorInfo diffWeightsTensor = {{1, 1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCD};
    const KuDNN::TensorInfo diffDstTensor = {{1, 1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCD};
    const KuDNN::TensorInfo diffBiasTensor = {{1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::A};
    KuDNN::Shape strides(1, 1);
    KuDNN::Shape paddingL(0, 0);
    KuDNN::Shape paddingR(0, 0);
    KuDNN::DeconvolutionAlgorithm alg(KuDNN::DeconvolutionAlgorithm::DIRECT);
    KuDNN::DeconvolutionLayerBWDWeights deconvBwdWeightsLayer1(diffDstTensor, srcTensor, diffWeightsTensor,
        diffBiasTensor, strides, paddingL, paddingR, alg);
    KuDNN::DeconvolutionLayerBWDWeights deconvBwdWeightsLayer2(deconvBwdWeightsLayer1);
    deconvBwdWeightsLayer2 = deconvBwdWeightsLayer1;
    for (size_t i = 0; i < g_Tensor_data.size() / 4; i += 4) {
        KuDNN::DeconvolutionLayerBWDWeights deconvBwdWeightsLayer1(g_Tensor_data[i], g_Tensor_data[i + 1], g_Tensor_data[i + 2],
             g_Tensor_data[i + 3], strides, paddingL, paddingR, alg);
        KuDNN::DeconvolutionLayerBWDWeights deconvBwdWeightsLayer2(deconvBwdWeightsLayer1);
        deconvBwdWeightsLayer2 = deconvBwdWeightsLayer1;
    }
}

static void BackwardWeightsMoveConstructor1()
{
    const KuDNN::TensorInfo srcTensor = {{1, 1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCD};
    const KuDNN::TensorInfo diffWeightsTensor = {{1, 1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCD};
    const KuDNN::TensorInfo diffDstTensor = {{1, 1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCD};
    const KuDNN::TensorInfo diffBiasTensor = {{1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::A};
    KuDNN::Shape strides(1, 1);
    KuDNN::Shape dilates(0, 0);
    KuDNN::Shape paddingL(0, 0);
    KuDNN::Shape paddingR(0, 0);
    KuDNN::DeconvolutionAlgorithm alg(KuDNN::DeconvolutionAlgorithm::DIRECT);
    KuDNN::DeconvolutionLayerBWDWeights deconvBwdWeightsLayer1(diffDstTensor, srcTensor, diffWeightsTensor,
        diffBiasTensor, strides, dilates, paddingL, paddingR, alg);
    KuDNN::DeconvolutionLayerBWDWeights deconvBwdWeightsLayer2(std::move(deconvBwdWeightsLayer1));
    for (size_t i = 0; i < g_Tensor_data.size() / 4; i += 4) {
        KuDNN::DeconvolutionLayerBWDWeights deconvBwdWeightsLayer1(g_Tensor_data[i], g_Tensor_data[i + 1], g_Tensor_data[i + 2],
             g_Tensor_data[i + 3], strides, dilates, paddingL, paddingR, alg);
        KuDNN::DeconvolutionLayerBWDWeights deconvBwdWeightsLayer2(std::move(deconvBwdWeightsLayer1));
    }
}

static void BackwardWeightsMoveConstructor2()
{
    const KuDNN::TensorInfo srcTensor = {{1, 1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCD};
    const KuDNN::TensorInfo diffWeightsTensor = {{1, 1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCD};
    const KuDNN::TensorInfo diffDstTensor = {{1, 1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCD};
    const KuDNN::TensorInfo diffBiasTensor = {{1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::A};
    KuDNN::Shape strides(1, 1);
    KuDNN::Shape dilates(0, 0);
    KuDNN::Shape paddingL(0, 0);
    KuDNN::Shape paddingR(0, 0);
    KuDNN::DeconvolutionAlgorithm alg(KuDNN::DeconvolutionAlgorithm::DIRECT);
    KuDNN::DeconvolutionLayerBWDWeights deconvBwdWeightsLayer1(diffDstTensor, srcTensor, diffWeightsTensor,
        diffBiasTensor, strides, dilates, paddingL, paddingR, alg);
    KuDNN::DeconvolutionLayerBWDWeights deconvBwdWeightsLayer2(diffDstTensor, srcTensor, diffWeightsTensor,
        diffBiasTensor, strides, dilates, paddingL, paddingR, alg);
    deconvBwdWeightsLayer2 = std::move(deconvBwdWeightsLayer1);
    for (size_t i = 0; i < g_Tensor_data.size() / 4; i += 4) {
        KuDNN::DeconvolutionLayerBWDWeights deconvBwdWeightsLayer1(g_Tensor_data[i], g_Tensor_data[i + 1], g_Tensor_data[i + 2],
             g_Tensor_data[i + 3], strides, dilates, paddingL, paddingR, alg);
        KuDNN::DeconvolutionLayerBWDWeights deconvBwdWeightsLayer2(g_Tensor_data[i], g_Tensor_data[i + 1], g_Tensor_data[i + 2],
             g_Tensor_data[i + 3], strides, dilates, paddingL, paddingR, alg);
        deconvBwdWeightsLayer2 = std::move(deconvBwdWeightsLayer1);
    }
}

// test cases
TEST(DeconvGTests, FWDCopyConstructor1)
{
    EXPECT_NO_THROW(ForwardCopyConstructor1());
}

TEST(DeconvGTests, FWDCopyConstructor2)
{
    EXPECT_NO_THROW(ForwardCopyConstructor2());
}

TEST(DeconvGTests, FWDMoveConstructor1)
{
    EXPECT_NO_THROW(ForwardMoveConstructor1());
}

TEST(DeconvGTests, FWDMoveConstructor2)
{
    EXPECT_NO_THROW(ForwardMoveConstructor2());
}

TEST(DeconvGTests, BackwardDataCopyConstructor1)
{
    EXPECT_NO_THROW(BackwardDataCopyConstructor1());
}

TEST(DeconvGTests, BackwardDataCopyConstructor2)
{
    EXPECT_NO_THROW(BackwardDataCopyConstructor2());
}

TEST(DeconvGTests, BackwardDataMoveConstructor1)
{
    EXPECT_NO_THROW(BackwardDataMoveConstructor1());
}

TEST(DeconvGTests, BackwardDataMoveConstructor2)
{
    EXPECT_NO_THROW(BackwardDataMoveConstructor2());
}

TEST(DeconvGTests, BackwardWeightsCopyConstructor1)
{
    EXPECT_NO_THROW(BackwardWeightsCopyConstructor1());
}

TEST(DeconvGTests, BackwardWeightsCopyConstructor2)
{
    EXPECT_NO_THROW(BackwardWeightsCopyConstructor2());
}

TEST(DeconvGTests, BackwardWeightsMoveConstructor1)
{
    EXPECT_NO_THROW(BackwardWeightsMoveConstructor1());
}

TEST(DeconvGTests, BackwardWeightsMoveConstructor2)
{
    EXPECT_NO_THROW(BackwardWeightsMoveConstructor2());
}