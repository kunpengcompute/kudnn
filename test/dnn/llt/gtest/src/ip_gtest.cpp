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
    const KuDNN::TensorInfo dstTensor = {{1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::AB};
    const KuDNN::TensorInfo biasTensor = {{1},  KuDNN::Element::TypeT::F32, KuDNN::Layout::A};
    KuDNN::InnerProductLayerFWD ipLayerFwdLayer1(srcTensor, weightsTensor, dstTensor, biasTensor);
    KuDNN::InnerProductLayerFWD ipLayerFwdLayer2(ipLayerFwdLayer1);
    for (size_t i = 0; i < g_Tensor_data.size() / 4; i += 4) {
        KuDNN::InnerProductLayerFWD ipLayerFwdLayer1(g_Tensor_data[i], g_Tensor_data[i + 1], g_Tensor_data[i + 2],
            g_Tensor_data[i + 3]);
        KuDNN::InnerProductLayerFWD ipLayerFwdLayer2(ipLayerFwdLayer1);
    }
}

static void ForwardCopyConstructor2()
{
    const KuDNN::TensorInfo srcTensor = {{1, 1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCD};
    const KuDNN::TensorInfo weightsTensor = {{1, 1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCD};
    const KuDNN::TensorInfo dstTensor = {{1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::AB};
    const KuDNN::TensorInfo biasTensor = {{1},  KuDNN::Element::TypeT::F32, KuDNN::Layout::A};
    KuDNN::InnerProductLayerFWD ipLayerFwdLayer1(srcTensor, weightsTensor, dstTensor, biasTensor);
    KuDNN::InnerProductLayerFWD ipLayerFwdLayer2(ipLayerFwdLayer1);
    ipLayerFwdLayer2 = ipLayerFwdLayer1;
    for (size_t i = 0; i < g_Tensor_data.size() / 4; i += 4) {
        KuDNN::InnerProductLayerFWD ipLayerFwdLayer1(g_Tensor_data[i], g_Tensor_data[i + 1], g_Tensor_data[i + 2],
            g_Tensor_data[i + 3]);
        KuDNN::InnerProductLayerFWD ipLayerFwdLayer2(ipLayerFwdLayer1);
        ipLayerFwdLayer2 = ipLayerFwdLayer1;
    }
}

static void ForwardMoveConstructor1()
{
    const KuDNN::TensorInfo srcTensor = {{1, 1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCD};
    const KuDNN::TensorInfo weightsTensor = {{1, 1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCD};
    const KuDNN::TensorInfo dstTensor = {{1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::AB};
    const KuDNN::TensorInfo biasTensor = {{1},  KuDNN::Element::TypeT::F32, KuDNN::Layout::A};
    KuDNN::InnerProductLayerFWD ipLayerFwdLayer1(srcTensor, weightsTensor, dstTensor, biasTensor);
    KuDNN::InnerProductLayerFWD ipLayerFwdLayer2(std::move(ipLayerFwdLayer1));
    for (size_t i = 0; i < g_Tensor_data.size() / 4; i += 4) {
        KuDNN::InnerProductLayerFWD ipLayerFwdLayer1(g_Tensor_data[i], g_Tensor_data[i + 1], g_Tensor_data[i + 2],
            g_Tensor_data[i + 3]);
        KuDNN::InnerProductLayerFWD ipLayerFwdLayer2(std::move(ipLayerFwdLayer1));
    }
}

static void ForwardMoveConstructor2()
{
    const KuDNN::TensorInfo srcTensor = {{1, 1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCD};
    const KuDNN::TensorInfo weightsTensor = {{1, 1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCD};
    const KuDNN::TensorInfo dstTensor = {{1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::AB};
    const KuDNN::TensorInfo biasTensor = {{1},  KuDNN::Element::TypeT::F32, KuDNN::Layout::A};
    KuDNN::InnerProductLayerFWD ipLayerFwdLayer1(srcTensor, weightsTensor, dstTensor, biasTensor);
    KuDNN::InnerProductLayerFWD ipLayerFwdLayer2(srcTensor, weightsTensor, dstTensor, biasTensor);
    ipLayerFwdLayer2 = std::move(ipLayerFwdLayer1);
    for (size_t i = 0; i < g_Tensor_data.size() / 4; i += 4) {
        KuDNN::InnerProductLayerFWD ipLayerFwdLayer1(g_Tensor_data[i], g_Tensor_data[i + 1], g_Tensor_data[i + 2],
            g_Tensor_data[i + 3]);
        KuDNN::InnerProductLayerFWD ipLayerFwdLayer2(g_Tensor_data[i], g_Tensor_data[i + 1], g_Tensor_data[i + 2],
            g_Tensor_data[i + 3]);
        ipLayerFwdLayer2 = std::move(ipLayerFwdLayer1);
    }
}

// backward data
static void BackwardDataCopyConstructor1()
{
    const KuDNN::TensorInfo diffSrcTensor = {{1, 1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCD};
    const KuDNN::TensorInfo weightsTensor = {{1, 1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCD};
    const KuDNN::TensorInfo diffDstTensor = {{1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::AB};
    KuDNN::InnerProductLayerBWDData ipLayerBwdDataLayer1(diffDstTensor, weightsTensor, diffSrcTensor);
    KuDNN::InnerProductLayerBWDData ipLayerBwdDataLayer2(ipLayerBwdDataLayer1);
    for (size_t i = 0; i < g_Tensor_data.size() / 3; i += 3) {
        KuDNN::InnerProductLayerBWDData ipLayerBwdDataLayer1(g_Tensor_data[i], g_Tensor_data[i + 1], g_Tensor_data[i + 2]);
        KuDNN::InnerProductLayerBWDData ipLayerBwdDataLayer2(ipLayerBwdDataLayer1);
    }
}

static void BackwardDataCopyConstructor2()
{
    const KuDNN::TensorInfo diffSrcTensor = {{1, 1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCD};
    const KuDNN::TensorInfo weightsTensor = {{1, 1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCD};
    const KuDNN::TensorInfo diffDstTensor = {{1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::AB};
    KuDNN::InnerProductLayerBWDData ipLayerBwdDataLayer1(diffDstTensor, weightsTensor, diffSrcTensor);
    KuDNN::InnerProductLayerBWDData ipLayerBwdDataLayer2(ipLayerBwdDataLayer1);
    ipLayerBwdDataLayer2 = ipLayerBwdDataLayer1;
    for (size_t i = 0; i < g_Tensor_data.size() / 3; i += 3) {
        KuDNN::InnerProductLayerBWDData ipLayerBwdDataLayer1(g_Tensor_data[i], g_Tensor_data[i + 1], g_Tensor_data[i + 2]);
        KuDNN::InnerProductLayerBWDData ipLayerBwdDataLayer2(ipLayerBwdDataLayer1);
        ipLayerBwdDataLayer2 = ipLayerBwdDataLayer1;
    }
}

static void BackwardDataMoveConstructor1()
{
    const KuDNN::TensorInfo diffSrcTensor = {{1, 1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCD};
    const KuDNN::TensorInfo weightsTensor = {{1, 1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCD};
    const KuDNN::TensorInfo diffDstTensor = {{1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::AB};
    KuDNN::InnerProductLayerBWDData ipLayerBwdDataLayer1(diffDstTensor, weightsTensor, diffSrcTensor);
    KuDNN::InnerProductLayerBWDData ipLayerBwdDataLayer2(std::move(ipLayerBwdDataLayer1));
    for (size_t i = 0; i < g_Tensor_data.size() / 3; i += 3) {
        KuDNN::InnerProductLayerBWDData ipLayerBwdDataLayer1(g_Tensor_data[i], g_Tensor_data[i + 1], g_Tensor_data[i + 2]);
        KuDNN::InnerProductLayerBWDData ipLayerBwdDataLayer2(std::move(ipLayerBwdDataLayer1));
    }
}

static void BackwardDataMoveConstructor2()
{
    const KuDNN::TensorInfo diffSrcTensor = {{1, 1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCD};
    const KuDNN::TensorInfo weightsTensor = {{1, 1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCD};
    const KuDNN::TensorInfo diffDstTensor = {{1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::AB};
    KuDNN::InnerProductLayerBWDData ipLayerBwdDataLayer1(diffDstTensor, weightsTensor, diffSrcTensor);
    KuDNN::InnerProductLayerBWDData ipLayerBwdDataLayer2(diffDstTensor, weightsTensor, diffSrcTensor);
    ipLayerBwdDataLayer2 = std::move(ipLayerBwdDataLayer1);
    for (size_t i = 0; i < g_Tensor_data.size() / 3; i += 3) {
        KuDNN::InnerProductLayerBWDData ipLayerBwdDataLayer1(g_Tensor_data[i], g_Tensor_data[i + 1], g_Tensor_data[i + 2]);
        KuDNN::InnerProductLayerBWDData ipLayerBwdDataLayer2(g_Tensor_data[i], g_Tensor_data[i + 1], g_Tensor_data[i + 2]);
        ipLayerBwdDataLayer2 = std::move(ipLayerBwdDataLayer1);
    }
}

// backward weights
static void BackwardWeightsCopyConstructor1()
{
    const KuDNN::TensorInfo srcTensor = {{1, 1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCD};
    const KuDNN::TensorInfo diffWeightsTensor = {{1, 1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCD};
    const KuDNN::TensorInfo diffDstTensor = {{1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::AB};
    const KuDNN::TensorInfo diffBiasTensor = {{1},  KuDNN::Element::TypeT::F32, KuDNN::Layout::A};
    KuDNN::InnerProductLayerBWDWeights ipLayerBwdWeightsLayer1(diffDstTensor, srcTensor, diffWeightsTensor,
        diffBiasTensor);
    KuDNN::InnerProductLayerBWDWeights ipLayerBwdWeightsLayer2(ipLayerBwdWeightsLayer1);
    for (size_t i = 0; i < g_Tensor_data.size() / 4; i += 4) {
        KuDNN::InnerProductLayerBWDWeights ipLayerBwdWeightsLayer1(g_Tensor_data[i], g_Tensor_data[i + 1], g_Tensor_data[i + 2],
            g_Tensor_data[i + 3]);
        KuDNN::InnerProductLayerBWDWeights ipLayerBwdWeightsLayer2(ipLayerBwdWeightsLayer1);
    }
}

static void BackwardWeightsCopyConstructor2()
{
    const KuDNN::TensorInfo srcTensor = {{1, 1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCD};
    const KuDNN::TensorInfo diffWeightsTensor = {{1, 1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCD};
    const KuDNN::TensorInfo diffDstTensor = {{1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::AB};
    const KuDNN::TensorInfo diffBiasTensor = {{1},  KuDNN::Element::TypeT::F32, KuDNN::Layout::A};
    KuDNN::InnerProductLayerBWDWeights ipLayerBwdWeightsLayer1(diffDstTensor, srcTensor, diffWeightsTensor,
        diffBiasTensor);
    KuDNN::InnerProductLayerBWDWeights ipLayerBwdWeightsLayer2(ipLayerBwdWeightsLayer1);
    ipLayerBwdWeightsLayer2 = ipLayerBwdWeightsLayer1;
    for (size_t i = 0; i < g_Tensor_data.size() / 4; i += 4) {
        KuDNN::InnerProductLayerBWDWeights ipLayerBwdWeightsLayer1(g_Tensor_data[i], g_Tensor_data[i + 1], g_Tensor_data[i + 2],
            g_Tensor_data[i + 3]);
        KuDNN::InnerProductLayerBWDWeights ipLayerBwdWeightsLayer2(ipLayerBwdWeightsLayer1);
        ipLayerBwdWeightsLayer2 = ipLayerBwdWeightsLayer1;
    }
}

static void BackwardWeightsMoveConstructor1()
{
    const KuDNN::TensorInfo srcTensor = {{1, 1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCD};
    const KuDNN::TensorInfo diffWeightsTensor = {{1, 1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCD};
    const KuDNN::TensorInfo diffDstTensor = {{1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::AB};
    const KuDNN::TensorInfo diffBiasTensor = {{1},  KuDNN::Element::TypeT::F32, KuDNN::Layout::A};
    KuDNN::InnerProductLayerBWDWeights ipLayerBwdWeightsLayer1(diffDstTensor, srcTensor, diffWeightsTensor,
        diffBiasTensor);
    KuDNN::InnerProductLayerBWDWeights ipLayerBwdWeightsLayer2(std::move(ipLayerBwdWeightsLayer1));
    for (size_t i = 0; i < g_Tensor_data.size() / 4; i += 4) {
        KuDNN::InnerProductLayerBWDWeights ipLayerBwdWeightsLayer1(g_Tensor_data[i], g_Tensor_data[i + 1], g_Tensor_data[i + 2],
            g_Tensor_data[i + 3]);
        KuDNN::InnerProductLayerBWDWeights ipLayerBwdWeightsLayer2(std::move(ipLayerBwdWeightsLayer1));
    }
}

static void BackwardWeightsMoveConstructor2()
{
    const KuDNN::TensorInfo srcTensor = {{1, 1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCD};
    const KuDNN::TensorInfo diffWeightsTensor = {{1, 1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCD};
    const KuDNN::TensorInfo diffDstTensor = {{1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::AB};
    const KuDNN::TensorInfo diffBiasTensor = {{1},  KuDNN::Element::TypeT::F32, KuDNN::Layout::A};
    KuDNN::InnerProductLayerBWDWeights ipLayerBwdWeightsLayer1(diffDstTensor, srcTensor, diffWeightsTensor,
        diffBiasTensor);
    KuDNN::InnerProductLayerBWDWeights ipLayerBwdWeightsLayer2(diffDstTensor, srcTensor, diffWeightsTensor,
        diffBiasTensor);
    ipLayerBwdWeightsLayer2 = std::move(ipLayerBwdWeightsLayer1);
    for (size_t i = 0; i < g_Tensor_data.size() / 4; i += 4) {
        KuDNN::InnerProductLayerBWDWeights ipLayerBwdWeightsLayer1(g_Tensor_data[i], g_Tensor_data[i + 1], g_Tensor_data[i + 2],
            g_Tensor_data[i + 3]);
        KuDNN::InnerProductLayerBWDWeights ipLayerBwdWeightsLayer2(g_Tensor_data[i], g_Tensor_data[i + 1], g_Tensor_data[i + 2],
            g_Tensor_data[i + 3]);
        ipLayerBwdWeightsLayer2 = std::move(ipLayerBwdWeightsLayer1);
    }
}

// test cases
TEST(InnerProductGTests, FWDCopyConstructor1)
{
    EXPECT_NO_THROW(ForwardCopyConstructor1());
}

TEST(InnerProductGTests, FWDCopyConstructor2)
{
    EXPECT_NO_THROW(ForwardCopyConstructor2());
}

TEST(InnerProductGTests, FWDMoveConstructor1)
{
    EXPECT_NO_THROW(ForwardMoveConstructor1());
}

TEST(InnerProductGTests, FWDMoveConstructor2)
{
    EXPECT_NO_THROW(ForwardMoveConstructor2());
}

TEST(InnerProductGTests, BackwardDataCopyConstructor1)
{
    EXPECT_NO_THROW(BackwardDataCopyConstructor1());
}

TEST(InnerProductGTests, BackwardDataCopyConstructor2)
{
    EXPECT_NO_THROW(BackwardDataCopyConstructor2());
}

TEST(InnerProductGTests, BackwardDataMoveConstructor1)
{
    EXPECT_NO_THROW(BackwardDataMoveConstructor1());
}

TEST(InnerProductGTests, BackwardDataMoveConstructor2)
{
    EXPECT_NO_THROW(BackwardDataMoveConstructor2());
}

TEST(InnerProductGTests, BackwardWeightsCopyConstructor1)
{
    EXPECT_NO_THROW(BackwardWeightsCopyConstructor1());
}

TEST(InnerProductGTests, BackwardWeightsCopyConstructor2)
{
    EXPECT_NO_THROW(BackwardWeightsCopyConstructor2());
}

TEST(InnerProductGTests, BackwardWeightsMoveConstructor1)
{
    EXPECT_NO_THROW(BackwardWeightsMoveConstructor1());
}

TEST(InnerProductGTests, BackwardWeightsMoveConstructor2)
{
    EXPECT_NO_THROW(BackwardWeightsMoveConstructor2());
}