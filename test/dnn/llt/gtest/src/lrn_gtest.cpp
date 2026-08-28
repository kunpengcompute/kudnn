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
    float alpha = 1.0;
    float beta = 1.0;
    float k = 1.0;
    size_t localSize = 10;
    KuDNN::LRNormAlgorithmKind algKind(KuDNN::LRNormAlgorithmKind::WITHIN);
    KuDNN::LocalResponseNormalizationLayerFWD lrnLayer1(srcInfo, dstInfo, alpha, beta, k, localSize, algKind);
    KuDNN::LocalResponseNormalizationLayerFWD lrnLayer2(lrnLayer1);
    for (size_t i = 0; i < g_Tensor_data.size() / 2; i += 2) {
        KuDNN::LocalResponseNormalizationLayerFWD lrnLayer1(g_Tensor_data[i], g_Tensor_data[i + 1], alpha, beta, k, localSize, algKind);
        KuDNN::LocalResponseNormalizationLayerFWD lrnLayer2(lrnLayer1);
    }
}

static void ForwardCopyConstructor2()
{
    const KuDNN::TensorInfo srcInfo = {{1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABC};
    const KuDNN::TensorInfo dstInfo = {{1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABC};
    float alpha = 1.0;
    float beta = 1.0;
    float k = 1.0;
    size_t localSize = 10;
    KuDNN::LRNormAlgorithmKind algKind(KuDNN::LRNormAlgorithmKind::WITHIN);
    KuDNN::LocalResponseNormalizationLayerFWD lrnLayer1(srcInfo, dstInfo, alpha, beta, k, localSize, algKind);
    KuDNN::LocalResponseNormalizationLayerFWD lrnLayer2(srcInfo, dstInfo, alpha, beta, k, localSize, algKind);
    lrnLayer2 = lrnLayer1;
    for (size_t i = 0; i < g_Tensor_data.size() / 2; i += 2) {
        KuDNN::LocalResponseNormalizationLayerFWD lrnLayer1(g_Tensor_data[i], g_Tensor_data[i + 1], alpha, beta, k, localSize, algKind);
        KuDNN::LocalResponseNormalizationLayerFWD lrnLayer2(g_Tensor_data[i], g_Tensor_data[i + 1], alpha, beta, k, localSize, algKind);
        lrnLayer2 = lrnLayer1;
    }
}

static void ForwardMoveConstructor1()
{
    const KuDNN::TensorInfo srcInfo = {{1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABC};
    const KuDNN::TensorInfo dstInfo = {{1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABC};
    float alpha = 1.0;
    float beta = 1.0;
    float k = 1.0;
    size_t localSize = 10;
    KuDNN::LRNormAlgorithmKind algKind(KuDNN::LRNormAlgorithmKind::WITHIN);
    KuDNN::LocalResponseNormalizationLayerFWD lrnLayer1(srcInfo, dstInfo, alpha, beta, k, localSize, algKind);
    KuDNN::LocalResponseNormalizationLayerFWD lrnLayer2(std::move(lrnLayer1));
    for (size_t i = 0; i < g_Tensor_data.size() / 2; i += 2) {
        KuDNN::LocalResponseNormalizationLayerFWD lrnLayer1(g_Tensor_data[i], g_Tensor_data[i + 1], alpha, beta, k, localSize, algKind);
        KuDNN::LocalResponseNormalizationLayerFWD lrnLayer2(std::move(lrnLayer1));
    }
}

static void ForwardMoveConstructor2()
{
    const KuDNN::TensorInfo srcInfo = {{1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABC};
    const KuDNN::TensorInfo dstInfo = {{1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABC};
    float alpha = 1.0;
    float beta = 1.0;
    float k = 1.0;
    size_t localSize = 10;
    KuDNN::LRNormAlgorithmKind algKind(KuDNN::LRNormAlgorithmKind::WITHIN);
    KuDNN::LocalResponseNormalizationLayerFWD lrnLayer1(srcInfo, dstInfo, alpha, beta, k, localSize, algKind);
    KuDNN::LocalResponseNormalizationLayerFWD lrnLayer2(srcInfo, dstInfo, alpha, beta, k, localSize, algKind);
    lrnLayer2 = std::move(lrnLayer1);
    for (size_t i = 0; i < g_Tensor_data.size() / 2; i += 2) {
        KuDNN::LocalResponseNormalizationLayerFWD lrnLayer1(g_Tensor_data[i], g_Tensor_data[i + 1], alpha, beta, k, localSize, algKind);
        KuDNN::LocalResponseNormalizationLayerFWD lrnLayer2(g_Tensor_data[i], g_Tensor_data[i + 1], alpha, beta, k, localSize, algKind);
        lrnLayer2 = std::move(lrnLayer1);
    }
}

// backward
static void BackwardCopyConstructor1()
{
    const KuDNN::TensorInfo srcInfo = {{1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABC};
    const KuDNN::TensorInfo diffSrcInfo = {{1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABC};
    const KuDNN::TensorInfo diffDstInfo = {{1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABC};
    float alpha = 1.0;
    float beta = 1.0;
    float k = 1.0;
    size_t localSize = 10;
    KuDNN::LRNormAlgorithmKind algKind(KuDNN::LRNormAlgorithmKind::WITHIN);
    KuDNN::LocalResponseNormalizationLayerBWD lrnLayer1(srcInfo, diffDstInfo, diffSrcInfo, alpha, beta, k, localSize, algKind);
    KuDNN::LocalResponseNormalizationLayerBWD lrnLayer2(lrnLayer1);
    for (size_t i = 0; i < g_Tensor_data.size() / 3; i += 3) {
        KuDNN::LocalResponseNormalizationLayerBWD lrnLayer1(g_Tensor_data[i], g_Tensor_data[i + 1], g_Tensor_data[i + 2], 
            alpha, beta, k, localSize, algKind);
        KuDNN::LocalResponseNormalizationLayerBWD lrnLayer2(lrnLayer1);
    }
}

static void BackwardCopyConstructor2()
{
    const KuDNN::TensorInfo srcInfo = {{1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABC};
    const KuDNN::TensorInfo diffSrcInfo = {{1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABC};
    const KuDNN::TensorInfo diffDstInfo = {{1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABC};
    float alpha = 1.0;
    float beta = 1.0;
    float k = 1.0;
    size_t localSize = 10;
    KuDNN::LRNormAlgorithmKind algKind(KuDNN::LRNormAlgorithmKind::WITHIN);
    KuDNN::LocalResponseNormalizationLayerBWD lrnLayer1(srcInfo, diffDstInfo, diffSrcInfo, alpha, beta, k, localSize, algKind);
    KuDNN::LocalResponseNormalizationLayerBWD lrnLayer2(srcInfo, diffDstInfo, diffSrcInfo, alpha, beta, k, localSize, algKind);
    lrnLayer2 = lrnLayer1;
    for (size_t i = 0; i < g_Tensor_data.size() / 3; i += 3) {
        KuDNN::LocalResponseNormalizationLayerBWD lrnLayer1(g_Tensor_data[i], g_Tensor_data[i + 1], g_Tensor_data[i + 2], 
            alpha, beta, k, localSize, algKind);
        KuDNN::LocalResponseNormalizationLayerBWD lrnLayer2(g_Tensor_data[i], g_Tensor_data[i + 1], g_Tensor_data[i + 2], 
            alpha, beta, k, localSize, algKind);
        lrnLayer2 = lrnLayer1;
    }
}

static void BackwardMoveConstructor1()
{
    const KuDNN::TensorInfo srcInfo = {{1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABC};
    const KuDNN::TensorInfo diffSrcInfo = {{1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABC};
    const KuDNN::TensorInfo diffDstInfo = {{1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABC};
    float alpha = 1.0;
    float beta = 1.0;
    float k = 1.0;
    size_t localSize = 10;
    KuDNN::LRNormAlgorithmKind algKind(KuDNN::LRNormAlgorithmKind::WITHIN);
    KuDNN::LocalResponseNormalizationLayerBWD lrnLayer1(srcInfo, diffDstInfo, diffSrcInfo, alpha, beta, k, localSize, algKind);
    KuDNN::LocalResponseNormalizationLayerBWD lrnLayer2(std::move(lrnLayer1));
    for (size_t i = 0; i < g_Tensor_data.size() / 3; i += 3) {
        KuDNN::LocalResponseNormalizationLayerBWD lrnLayer1(g_Tensor_data[i], g_Tensor_data[i + 1], g_Tensor_data[i + 2], 
            alpha, beta, k, localSize, algKind);
        KuDNN::LocalResponseNormalizationLayerBWD lrnLayer2(std::move(lrnLayer1));
    }
}

static void BackwardMoveConstructor2()
{
    const KuDNN::TensorInfo srcInfo = {{1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABC};
    const KuDNN::TensorInfo diffSrcInfo = {{1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABC};
    const KuDNN::TensorInfo diffDstInfo = {{1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABC};
    float alpha = 1.0;
    float beta = 1.0;
    float k = 1.0;
    size_t localSize = 10;
    KuDNN::LRNormAlgorithmKind algKind(KuDNN::LRNormAlgorithmKind::WITHIN);
    KuDNN::LocalResponseNormalizationLayerBWD lrnLayer1(srcInfo, diffDstInfo, diffSrcInfo, alpha, beta, k, localSize, algKind);
    KuDNN::LocalResponseNormalizationLayerBWD lrnLayer2(srcInfo, diffDstInfo, diffSrcInfo, alpha, beta, k, localSize, algKind);
    lrnLayer2 = std::move(lrnLayer1);
    for (size_t i = 0; i < g_Tensor_data.size() / 3; i += 3) {
        KuDNN::LocalResponseNormalizationLayerBWD lrnLayer1(g_Tensor_data[i], g_Tensor_data[i + 1], g_Tensor_data[i + 2], 
            alpha, beta, k, localSize, algKind);
        KuDNN::LocalResponseNormalizationLayerBWD lrnLayer2(g_Tensor_data[i], g_Tensor_data[i + 1], g_Tensor_data[i + 2], 
            alpha, beta, k, localSize, algKind);
        lrnLayer2 = std::move(lrnLayer1);
    }
}

// test cases
TEST(LrnGTests, FWDCopyConstructor1)
{
    EXPECT_NO_THROW(ForwardCopyConstructor1());
}

TEST(LrnGTests, FWDCopyConstructor2)
{
    EXPECT_NO_THROW(ForwardCopyConstructor2());
}

TEST(LrnGTests, FWDMoveConstructor1)
{
    EXPECT_NO_THROW(ForwardMoveConstructor1());
}

TEST(LrnGTests, FWDMoveConstructor2)
{
    EXPECT_NO_THROW(ForwardMoveConstructor2());
}

TEST(LrnGTests, BWDCopyConstructor1)
{
    EXPECT_NO_THROW(BackwardCopyConstructor1());
}

TEST(LrnGTests, BWDCopyConstructor2)
{
    EXPECT_NO_THROW(BackwardCopyConstructor2());
}

TEST(LrnGTests, BWDMoveConstructor1)
{
    EXPECT_NO_THROW(BackwardMoveConstructor1());
}

TEST(LrnGTests, BWDMoveConstructor2)
{
    EXPECT_NO_THROW(BackwardMoveConstructor2());
}