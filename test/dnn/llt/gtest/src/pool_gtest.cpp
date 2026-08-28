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
    const KuDNN::TensorInfo srcInfo = {{1, 1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCD};
    const KuDNN::TensorInfo dstInfo = {{1, 1, 2, 2}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCD};
    KuDNN::Shape strides(1, 1);
    KuDNN::Shape kernel(1, 1);
    KuDNN::Shape dilates(1, 1);
    KuDNN::Shape paddingL(1, 1);
    KuDNN::Shape paddingR(0, 0);
    KuDNN::Propagation propKind(KuDNN::Propagation::FORWARD_INFERENCE);
    KuDNN::PoolingFunction algKind(KuDNN::PoolingFunction::MAX);
    KuDNN::PoolingLayerFWD poolLayer1(propKind, algKind, srcInfo, dstInfo, strides, kernel, dilates, paddingL, paddingR);
    KuDNN::PoolingLayerFWD poolLayer2(poolLayer1);
    for (size_t i = 0; i < g_Tensor_data.size() / 2; i += 2) {
        KuDNN::PoolingLayerFWD poolLayer1(propKind, algKind, g_Tensor_data[i], g_Tensor_data[i + 1], strides, kernel, dilates, paddingL, paddingR);
        KuDNN::PoolingLayerFWD poolLayer2(poolLayer1);
    }
}

static void ForwardCopyConstructor2()
{
    const KuDNN::TensorInfo srcInfo = {{1, 1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCD};
    const KuDNN::TensorInfo dstInfo = {{1, 1, 2, 2}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCD};
    KuDNN::Shape strides(1, 1);
    KuDNN::Shape kernel(1, 1);
    KuDNN::Shape dilates(1, 1);
    KuDNN::Shape paddingL(1, 1);
    KuDNN::Shape paddingR(0, 0);
    KuDNN::Propagation propKind(KuDNN::Propagation::FORWARD_INFERENCE);
    KuDNN::PoolingFunction algKind(KuDNN::PoolingFunction::MAX);
    KuDNN::PoolingLayerFWD poolLayer1(propKind, algKind, srcInfo, dstInfo, strides, kernel, dilates, paddingL, paddingR);
    KuDNN::PoolingLayerFWD poolLayer2(propKind, algKind, srcInfo, dstInfo, strides, kernel, dilates, paddingL, paddingR);
    poolLayer2 = poolLayer1;
    for (size_t i = 0; i < g_Tensor_data.size() / 2; i += 2) {
        KuDNN::PoolingLayerFWD poolLayer1(propKind, algKind, g_Tensor_data[i], g_Tensor_data[i + 1], strides, kernel, dilates, paddingL, paddingR);
        KuDNN::PoolingLayerFWD poolLayer2(propKind, algKind, g_Tensor_data[i], g_Tensor_data[i + 1], strides, kernel, dilates, paddingL, paddingR);
        poolLayer2 = poolLayer1;
    }
}

static void ForwardMoveConstructor1()
{
    const KuDNN::TensorInfo srcInfo = {{1, 1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCD};
    const KuDNN::TensorInfo dstInfo = {{1, 1, 2, 2}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCD};
    KuDNN::Shape strides(1, 1);
    KuDNN::Shape kernel(1, 1);
    KuDNN::Shape dilates(1, 1);
    KuDNN::Shape paddingL(1, 1);
    KuDNN::Shape paddingR(0, 0);
    KuDNN::Propagation propKind(KuDNN::Propagation::FORWARD_INFERENCE);
    KuDNN::PoolingFunction algKind(KuDNN::PoolingFunction::MAX);
    KuDNN::PoolingLayerFWD poolLayer1(propKind, algKind, srcInfo, dstInfo, strides, kernel, dilates, paddingL, paddingR);
    KuDNN::PoolingLayerFWD poolLayer2(std::move(poolLayer1));
    for (size_t i = 0; i < g_Tensor_data.size() / 2; i += 2) {
        KuDNN::PoolingLayerFWD poolLayer1(propKind, algKind, g_Tensor_data[i], g_Tensor_data[i + 1], strides, kernel, dilates, paddingL, paddingR);
        KuDNN::PoolingLayerFWD poolLayer2(std::move(poolLayer1));
    }
}

static void ForwardMoveConstructor2()
{
    const KuDNN::TensorInfo srcInfo = {{1, 1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCD};
    const KuDNN::TensorInfo dstInfo = {{1, 1, 2, 2}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCD};
    KuDNN::Shape strides(1, 1);
    KuDNN::Shape kernel(1, 1);
    KuDNN::Shape dilates(1, 1);
    KuDNN::Shape paddingL(1, 1);
    KuDNN::Shape paddingR(0, 0);
    KuDNN::Propagation propKind(KuDNN::Propagation::FORWARD_INFERENCE);
    KuDNN::PoolingFunction algKind(KuDNN::PoolingFunction::MAX);
    KuDNN::PoolingLayerFWD poolLayer1(propKind, algKind, srcInfo, dstInfo, strides, kernel, dilates, paddingL, paddingR);
    KuDNN::PoolingLayerFWD poolLayer2(propKind, algKind, srcInfo, dstInfo, strides, kernel, dilates, paddingL, paddingR);
    poolLayer2 = std::move(poolLayer1);
    for (size_t i = 0; i < g_Tensor_data.size() / 2; i += 2) {
        KuDNN::PoolingLayerFWD poolLayer1(propKind, algKind, g_Tensor_data[i], g_Tensor_data[i + 1], strides, kernel, dilates, paddingL, paddingR);
        KuDNN::PoolingLayerFWD poolLayer2(propKind, algKind, g_Tensor_data[i], g_Tensor_data[i + 1], strides, kernel, dilates, paddingL, paddingR);
        poolLayer2 = std::move(poolLayer1);
    }
}

// backward
static void BackwardCopyConstructor1()
{
    const KuDNN::TensorInfo srcInfo = {{1, 1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCD};
    const KuDNN::TensorInfo dstInfo = {{1, 1, 2, 2}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCD};
    KuDNN::Shape strides(1, 1);
    KuDNN::Shape kernel(1, 1);
    KuDNN::Shape dilates(1, 1);
    KuDNN::Shape paddingL(1, 1);
    KuDNN::Shape paddingR(0, 0);
    KuDNN::PoolingFunction algKind(KuDNN::PoolingFunction::MAX);
    KuDNN::PoolingLayerBWD poolLayer1(algKind, srcInfo, dstInfo, strides, kernel, dilates, paddingL, paddingR);
    KuDNN::PoolingLayerBWD poolLayer2(poolLayer1);
    for (size_t i = 0; i < g_Tensor_data.size() / 2; i += 2) {
        KuDNN::PoolingLayerBWD poolLayer1(algKind, g_Tensor_data[i], g_Tensor_data[i + 1], strides, kernel, dilates, paddingL, paddingR);
        KuDNN::PoolingLayerBWD poolLayer2(poolLayer1);
    }
}

static void BackwardCopyConstructor2()
{
    const KuDNN::TensorInfo srcInfo = {{1, 1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCD};
    const KuDNN::TensorInfo dstInfo = {{1, 1, 2, 2}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCD};
    KuDNN::Shape strides(1, 1);
    KuDNN::Shape kernel(1, 1);
    KuDNN::Shape dilates(1, 1);
    KuDNN::Shape paddingL(1, 1);
    KuDNN::Shape paddingR(0, 0);
    KuDNN::PoolingFunction algKind(KuDNN::PoolingFunction::MAX);
    KuDNN::PoolingLayerBWD poolLayer1(algKind, srcInfo, dstInfo, strides, kernel, dilates, paddingL, paddingR);
    KuDNN::PoolingLayerBWD poolLayer2(algKind, srcInfo, dstInfo, strides, kernel, dilates, paddingL, paddingR);
    poolLayer2 = poolLayer1;
    for (size_t i = 0; i < g_Tensor_data.size() / 2; i += 2) {
        KuDNN::PoolingLayerBWD poolLayer1(algKind, g_Tensor_data[i], g_Tensor_data[i + 1], strides, kernel, dilates, paddingL, paddingR);
        KuDNN::PoolingLayerBWD poolLayer2(algKind, g_Tensor_data[i], g_Tensor_data[i + 1], strides, kernel, dilates, paddingL, paddingR);
        poolLayer2 = poolLayer1;
    }
}

static void BackwardMoveConstructor1()
{
    const KuDNN::TensorInfo srcInfo = {{1, 1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCD};
    const KuDNN::TensorInfo dstInfo = {{1, 1, 2, 2}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCD};
    KuDNN::Shape strides(1, 1);
    KuDNN::Shape kernel(1, 1);
    KuDNN::Shape dilates(1, 1);
    KuDNN::Shape paddingL(1, 1);
    KuDNN::Shape paddingR(0, 0);
    KuDNN::PoolingFunction algKind(KuDNN::PoolingFunction::MAX);
    KuDNN::PoolingLayerBWD poolLayer1(algKind, srcInfo, dstInfo, strides, kernel, dilates, paddingL, paddingR);
    KuDNN::PoolingLayerBWD poolLayer2(std::move(poolLayer1));
    for (size_t i = 0; i < g_Tensor_data.size() / 2; i += 2) {
        KuDNN::PoolingLayerBWD poolLayer1(algKind, g_Tensor_data[i], g_Tensor_data[i + 1], strides, kernel, dilates, paddingL, paddingR);
        KuDNN::PoolingLayerBWD poolLayer2(std::move(poolLayer1));
    }
}

static void BackwardMoveConstructor2()
{
    const KuDNN::TensorInfo srcInfo = {{1, 1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCD};
    const KuDNN::TensorInfo dstInfo = {{1, 1, 2, 2}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCD};
    KuDNN::Shape strides(1, 1);
    KuDNN::Shape kernel(1, 1);
    KuDNN::Shape dilates(1, 1);
    KuDNN::Shape paddingL(1, 1);
    KuDNN::Shape paddingR(0, 0);
    KuDNN::PoolingFunction algKind(KuDNN::PoolingFunction::MAX);
    KuDNN::PoolingLayerBWD poolLayer1(algKind, srcInfo, dstInfo, strides, kernel, dilates, paddingL, paddingR);
    KuDNN::PoolingLayerBWD poolLayer2(algKind, srcInfo, dstInfo, strides, kernel, dilates, paddingL, paddingR);
    poolLayer2 = std::move(poolLayer1);
    for (size_t i = 0; i < g_Tensor_data.size() / 2; i += 2) {
        KuDNN::PoolingLayerBWD poolLayer1(algKind, g_Tensor_data[i], g_Tensor_data[i + 1], strides, kernel, dilates, paddingL, paddingR);
        KuDNN::PoolingLayerBWD poolLayer2(algKind, g_Tensor_data[i], g_Tensor_data[i + 1], strides, kernel, dilates, paddingL, paddingR);
        poolLayer2 = std::move(poolLayer1);
    }
}

// test cases
TEST(PoolGTests, FWDCopyConstructor1)
{
    EXPECT_NO_THROW(ForwardCopyConstructor1());
}

TEST(PoolGTests, FWDCopyConstructor2)
{
    EXPECT_NO_THROW(ForwardCopyConstructor2());
}

TEST(PoolGTests, FWDMoveConstructor1)
{
    EXPECT_NO_THROW(ForwardMoveConstructor1());
}

TEST(PoolGTests, FWDMoveConstructor2)
{
    EXPECT_NO_THROW(ForwardMoveConstructor2());
}

TEST(PoolGTests, BWDCopyConstructor1)
{
    EXPECT_NO_THROW(BackwardCopyConstructor1());
}

TEST(PoolGTests, BWDCopyConstructor2)
{
    EXPECT_NO_THROW(BackwardCopyConstructor2());
}

TEST(PoolGTests, BWDMoveConstructor1)
{
    EXPECT_NO_THROW(BackwardMoveConstructor1());
}

TEST(PoolGTests, BWDMoveConstructor2)
{
    EXPECT_NO_THROW(BackwardMoveConstructor2());
}