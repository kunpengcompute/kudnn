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
    const KuDNN::TensorInfo srcInfo = {{1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABC};
    const KuDNN::TensorInfo dstInfo = {{1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABC};
    KuDNN::ResamplingAlg algKind(KuDNN::ResamplingAlg::NEAREST_NEIGHBOR);
    KuDNN::ResamplingLayerFWD resamplingLayer1(srcInfo, dstInfo, algKind);
    KuDNN::ResamplingLayerFWD resamplingLayer2(resamplingLayer1);
    for (size_t i = 0; i < g_Tensor_data.size() / 2; i += 2) {
        KuDNN::ResamplingLayerFWD resamplingLayer1(g_Tensor_data[i], g_Tensor_data[i + 1], algKind);
        KuDNN::ResamplingLayerFWD resamplingLayer2(resamplingLayer1);
    }
}

static void ForwardCopyConstructor2()
{
    const KuDNN::TensorInfo srcInfo = {{1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABC};
    const KuDNN::TensorInfo dstInfo = {{1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABC};
    KuDNN::ResamplingAlg algKind(KuDNN::ResamplingAlg::NEAREST_NEIGHBOR);
    KuDNN::ResamplingLayerFWD resamplingLayer1(srcInfo, dstInfo, algKind);
    KuDNN::ResamplingLayerFWD resamplingLayer2(srcInfo, dstInfo, algKind);
    resamplingLayer2 = resamplingLayer1;
    for (size_t i = 0; i < g_Tensor_data.size() / 2; i += 2) {
        KuDNN::ResamplingLayerFWD resamplingLayer1(g_Tensor_data[i], g_Tensor_data[i + 1], algKind);
        KuDNN::ResamplingLayerFWD resamplingLayer2(g_Tensor_data[i], g_Tensor_data[i + 1], algKind);
        resamplingLayer2 = resamplingLayer1;
    }
}

static void ForwardMoveConstructor1()
{
    const KuDNN::TensorInfo srcInfo = {{1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABC};
    const KuDNN::TensorInfo dstInfo = {{1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABC};
    KuDNN::ResamplingAlg algKind(KuDNN::ResamplingAlg::NEAREST_NEIGHBOR);
    KuDNN::ResamplingLayerFWD resamplingLayer1(srcInfo, dstInfo, algKind);
    KuDNN::ResamplingLayerFWD resamplingLayer2(std::move(resamplingLayer1));
    for (size_t i = 0; i < g_Tensor_data.size() / 2; i += 2) {
        KuDNN::ResamplingLayerFWD resamplingLayer1(g_Tensor_data[i], g_Tensor_data[i + 1], algKind);
        KuDNN::ResamplingLayerFWD resamplingLayer2(std::move(resamplingLayer1));
    }
}

static void ForwardMoveConstructor2()
{
    const KuDNN::TensorInfo srcInfo = {{1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABC};
    const KuDNN::TensorInfo dstInfo = {{1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABC};
    KuDNN::ResamplingAlg algKind(KuDNN::ResamplingAlg::NEAREST_NEIGHBOR);
    KuDNN::ResamplingLayerFWD resamplingLayer1(srcInfo, dstInfo, algKind);
    KuDNN::ResamplingLayerFWD resamplingLayer2(srcInfo, dstInfo, algKind);
    resamplingLayer2 = std::move(resamplingLayer1);
    for (size_t i = 0; i < g_Tensor_data.size() / 2; i += 2) {
        KuDNN::ResamplingLayerFWD resamplingLayer1(g_Tensor_data[i], g_Tensor_data[i + 1], algKind);
        KuDNN::ResamplingLayerFWD resamplingLayer2(g_Tensor_data[i], g_Tensor_data[i + 1], algKind);
        resamplingLayer2 = std::move(resamplingLayer1);
    }
}

// backward
static void BackwardCopyConstructor1()
{
    const KuDNN::TensorInfo diffSrcInfo = {{1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABC};
    const KuDNN::TensorInfo diffDstInfo = {{1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABC};
    KuDNN::ResamplingAlg algKind(KuDNN::ResamplingAlg::NEAREST_NEIGHBOR);
    KuDNN::ResamplingLayerBWD resamplingLayer1(diffDstInfo, diffSrcInfo, algKind);
    KuDNN::ResamplingLayerBWD resamplingLayer2(resamplingLayer1);
    for (size_t i = 0; i < g_Tensor_data.size() / 2; i += 2) {
        KuDNN::ResamplingLayerBWD resamplingLayer1(g_Tensor_data[i], g_Tensor_data[i + 1], algKind);
        KuDNN::ResamplingLayerBWD resamplingLayer2(resamplingLayer1);
    }
}

static void BackwardCopyConstructor2()
{
    const KuDNN::TensorInfo diffSrcInfo = {{1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABC};
    const KuDNN::TensorInfo diffDstInfo = {{1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABC};
    KuDNN::ResamplingAlg algKind(KuDNN::ResamplingAlg::NEAREST_NEIGHBOR);
    KuDNN::ResamplingLayerBWD resamplingLayer1(diffDstInfo, diffSrcInfo, algKind);
    KuDNN::ResamplingLayerBWD resamplingLayer2(diffDstInfo, diffSrcInfo, algKind);
    resamplingLayer2 = resamplingLayer1;
    for (size_t i = 0; i < g_Tensor_data.size() / 2; i += 2) {
        KuDNN::ResamplingLayerBWD resamplingLayer1(g_Tensor_data[i], g_Tensor_data[i + 1], algKind);
        KuDNN::ResamplingLayerBWD resamplingLayer2(g_Tensor_data[i], g_Tensor_data[i + 1], algKind);
        resamplingLayer2 = resamplingLayer1;
    }
}

static void BackwardMoveConstructor1()
{
    const KuDNN::TensorInfo diffSrcInfo = {{1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABC};
    const KuDNN::TensorInfo diffDstInfo = {{1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABC};
    KuDNN::ResamplingAlg algKind(KuDNN::ResamplingAlg::NEAREST_NEIGHBOR);
    KuDNN::ResamplingLayerBWD resamplingLayer1(diffDstInfo, diffSrcInfo, algKind);
    KuDNN::ResamplingLayerBWD resamplingLayer2(std::move(resamplingLayer1));
    for (size_t i = 0; i < g_Tensor_data.size() / 2; i += 2) {
        KuDNN::ResamplingLayerBWD resamplingLayer1(g_Tensor_data[i], g_Tensor_data[i + 1], algKind);
        KuDNN::ResamplingLayerBWD resamplingLayer2(std::move(resamplingLayer1));
    }
}

static void BackwardMoveConstructor2()
{
    const KuDNN::TensorInfo diffSrcInfo = {{1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABC};
    const KuDNN::TensorInfo diffDstInfo = {{1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABC};
    KuDNN::ResamplingAlg algKind(KuDNN::ResamplingAlg::NEAREST_NEIGHBOR);
    KuDNN::ResamplingLayerBWD resamplingLayer1(diffDstInfo, diffSrcInfo, algKind);
    KuDNN::ResamplingLayerBWD resamplingLayer2(diffDstInfo, diffSrcInfo, algKind);
    resamplingLayer2 = std::move(resamplingLayer1);
    for (size_t i = 0; i < g_Tensor_data.size() / 2; i += 2) {
        KuDNN::ResamplingLayerBWD resamplingLayer1(g_Tensor_data[i], g_Tensor_data[i + 1], algKind);
        KuDNN::ResamplingLayerBWD resamplingLayer2(g_Tensor_data[i], g_Tensor_data[i + 1], algKind);
        resamplingLayer2 = std::move(resamplingLayer1);
    }
}

// test cases
TEST(ResamplingGTests, FWDCopyConstructor1)
{
    EXPECT_NO_THROW(ForwardCopyConstructor1());
}

TEST(ResamplingGTests, FWDCopyConstructor2)
{
    EXPECT_NO_THROW(ForwardCopyConstructor2());
}

TEST(ResamplingGTests, FWDMoveConstructor1)
{
    EXPECT_NO_THROW(ForwardMoveConstructor1());
}

TEST(ResamplingGTests, FWDMoveConstructor2)
{
    EXPECT_NO_THROW(ForwardMoveConstructor2());
}

TEST(ResamplingGTests, BWDCopyConstructor1)
{
    EXPECT_NO_THROW(BackwardCopyConstructor1());
}

TEST(ResamplingGTests, BWDCopyConstructor2)
{
    EXPECT_NO_THROW(BackwardCopyConstructor2());
}

TEST(ResamplingGTests, BWDMoveConstructor1)
{
    EXPECT_NO_THROW(BackwardMoveConstructor1());
}

TEST(ResamplingGTests, BWDMoveConstructor2)
{
    EXPECT_NO_THROW(BackwardMoveConstructor2());
}