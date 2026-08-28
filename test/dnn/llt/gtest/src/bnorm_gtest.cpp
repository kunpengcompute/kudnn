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
    const KuDNN::TensorInfo statInfo = {{1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::A};
    const KuDNN::TensorInfo scaleShiftInfo = {{1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::A};
    const KuDNN::TensorInfo dstInfo = {{1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABC};
    KuDNN::BatchNormalizationLayerFWD bnormLayer1(KuDNN::Propagation::FORWARD_INFERENCE, srcInfo, statInfo, scaleShiftInfo, dstInfo,
        KuDNN::NormalizationFlags::NONE);
    KuDNN::BatchNormalizationLayerFWD bnormLayer2(bnormLayer1);
    for (size_t i = 0; i < g_Tensor_data.size() / 4; i += 4) {
        KuDNN::BatchNormalizationLayerFWD bnormLayer1(KuDNN::Propagation::FORWARD_INFERENCE, g_Tensor_data[i], g_Tensor_data[i + 1],
            g_Tensor_data[i + 2], g_Tensor_data[i + 3], KuDNN::NormalizationFlags::NONE);
        KuDNN::BatchNormalizationLayerFWD bnormLayer2(bnormLayer1);
    }
}

static void ForwardCopyConstructor2()
{
    const KuDNN::TensorInfo srcInfo = {{1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABC};
    const KuDNN::TensorInfo statInfo = {{1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::A};
    const KuDNN::TensorInfo scaleShiftInfo = {{1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::A};
    const KuDNN::TensorInfo dstInfo = {{1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABC};
    KuDNN::BatchNormalizationLayerFWD bnormLayer1(KuDNN::Propagation::FORWARD_INFERENCE, srcInfo, statInfo, scaleShiftInfo, dstInfo,
        KuDNN::NormalizationFlags::NONE);
    KuDNN::BatchNormalizationLayerFWD bnormLayer2 = bnormLayer1;
    for (size_t i = 0; i < g_Tensor_data.size() / 4; i += 4) {
        KuDNN::BatchNormalizationLayerFWD bnormLayer1(KuDNN::Propagation::FORWARD_INFERENCE, g_Tensor_data[i], g_Tensor_data[i + 1], 
            g_Tensor_data[i + 2], g_Tensor_data[i + 3], KuDNN::NormalizationFlags::NONE);
        KuDNN::BatchNormalizationLayerFWD bnormLayer2 = bnormLayer1;
    }
}

static void ForwardCopyConstructor3()
{
    const KuDNN::TensorInfo srcInfo = {{1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABC};
    const KuDNN::TensorInfo statInfo = {{1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::A};
    const KuDNN::TensorInfo scaleShiftInfo = {{1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::A};
    const KuDNN::TensorInfo dstInfo = {{1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABC};
    KuDNN::BatchNormalizationLayerFWD bnormLayer1(KuDNN::Propagation::FORWARD_INFERENCE, srcInfo, statInfo, scaleShiftInfo, dstInfo,
        KuDNN::NormalizationFlags::NONE);
    KuDNN::BatchNormalizationLayerFWD bnormLayer2(bnormLayer1);
    bnormLayer2 = bnormLayer1;
    for (size_t i = 0; i < g_Tensor_data.size() / 4; i += 4) {
        KuDNN::BatchNormalizationLayerFWD bnormLayer1(KuDNN::Propagation::FORWARD_INFERENCE, g_Tensor_data[i], g_Tensor_data[i + 1], 
            g_Tensor_data[i + 2], g_Tensor_data[i + 3], KuDNN::NormalizationFlags::NONE);
        KuDNN::BatchNormalizationLayerFWD bnormLayer2 = bnormLayer1;
    }
}

static void ForwardMoveConstructor1()
{
    const KuDNN::TensorInfo srcInfo = {{1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABC};
    const KuDNN::TensorInfo statInfo = {{1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::A};
    const KuDNN::TensorInfo scaleShiftInfo = {{1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::A};
    const KuDNN::TensorInfo dstInfo = {{1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABC};
    KuDNN::BatchNormalizationLayerFWD bnormLayer1(KuDNN::Propagation::FORWARD_INFERENCE, srcInfo, statInfo, scaleShiftInfo, dstInfo,
        KuDNN::NormalizationFlags::NONE);
    KuDNN::BatchNormalizationLayerFWD bnormLayer2(std::move(bnormLayer1));
    for (size_t i = 0; i < g_Tensor_data.size() / 4; i += 4) {
        KuDNN::BatchNormalizationLayerFWD bnormLayer1(KuDNN::Propagation::FORWARD_INFERENCE, g_Tensor_data[i], g_Tensor_data[i + 1],
            g_Tensor_data[i + 2], g_Tensor_data[i + 3], KuDNN::NormalizationFlags::NONE);
        KuDNN::BatchNormalizationLayerFWD bnormLayer2(std::move(bnormLayer1));
    }
}

static void ForwardMoveConstructor2()
{
    const KuDNN::TensorInfo srcInfo = {{1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABC};
    const KuDNN::TensorInfo statInfo = {{1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::A};
    const KuDNN::TensorInfo scaleShiftInfo = {{1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::A};
    const KuDNN::TensorInfo dstInfo = {{1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABC};
    KuDNN::BatchNormalizationLayerFWD bnormLayer1(KuDNN::Propagation::FORWARD_INFERENCE, srcInfo, statInfo, scaleShiftInfo, dstInfo,
        KuDNN::NormalizationFlags::NONE);
    KuDNN::BatchNormalizationLayerFWD bnormLayer2(KuDNN::Propagation::FORWARD_INFERENCE, srcInfo, statInfo, scaleShiftInfo, dstInfo,
        KuDNN::NormalizationFlags::NONE);
    bnormLayer2 = std::move(bnormLayer1);
    for (size_t i = 0; i < g_Tensor_data.size() / 4; i += 4) {
        KuDNN::BatchNormalizationLayerFWD bnormLayer1(KuDNN::Propagation::FORWARD_INFERENCE, g_Tensor_data[i], g_Tensor_data[i + 1],
            g_Tensor_data[i + 2], g_Tensor_data[i + 3], KuDNN::NormalizationFlags::NONE);
        KuDNN::BatchNormalizationLayerFWD bnormLayer2(KuDNN::Propagation::FORWARD_INFERENCE, g_Tensor_data[i], g_Tensor_data[i + 1],
            g_Tensor_data[i + 2], g_Tensor_data[i + 3], KuDNN::NormalizationFlags::NONE);
        bnormLayer2 = std::move(bnormLayer1);
    }
}

// backward
static void BackwardCopyConstructor1()
{
    const KuDNN::TensorInfo srcInfo = {{1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABC};
    const KuDNN::TensorInfo diffSrcInfo = {{1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABC};
    const KuDNN::TensorInfo statInfo = {{1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::A};
    const KuDNN::TensorInfo scaleShiftInfo = {{1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::A};
    const KuDNN::TensorInfo diffScaleShiftInfo = {{1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::A};
    const KuDNN::TensorInfo diffDstInfo = {{1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABC};
    KuDNN::BatchNormalizationLayerBWD bnormLayer1(KuDNN::Propagation::BACKWARD_DATA, srcInfo, diffDstInfo,
        statInfo, scaleShiftInfo, diffSrcInfo, diffScaleShiftInfo, KuDNN::NormalizationFlags::NONE);
    KuDNN::BatchNormalizationLayerBWD bnormLayer2(bnormLayer1);
    for (size_t i = 0; i < g_Tensor_data.size() / 6; i += 6) {
        KuDNN::BatchNormalizationLayerBWD bnormLayer1(KuDNN::Propagation::BACKWARD_DATA,
            g_Tensor_data[i], g_Tensor_data[i + 1], g_Tensor_data[i + 2], g_Tensor_data[i + 3], 
            g_Tensor_data[i + 4], g_Tensor_data[i + 5], KuDNN::NormalizationFlags::NONE);
        KuDNN::BatchNormalizationLayerBWD bnormLayer2(bnormLayer1);
    }
}

static void BackwardCopyConstructor2()
{
    const KuDNN::TensorInfo srcInfo = {{1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABC};
    const KuDNN::TensorInfo diffSrcInfo = {{1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABC};
    const KuDNN::TensorInfo statInfo = {{1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::A};
    const KuDNN::TensorInfo scaleShiftInfo = {{1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::A};
    const KuDNN::TensorInfo diffScaleShiftInfo = {{1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::A};
    const KuDNN::TensorInfo diffDstInfo = {{1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABC};
    KuDNN::BatchNormalizationLayerBWD bnormLayer1(KuDNN::Propagation::BACKWARD_DATA, srcInfo, diffDstInfo,
        statInfo, scaleShiftInfo, diffSrcInfo, diffScaleShiftInfo, KuDNN::NormalizationFlags::NONE);
    KuDNN::BatchNormalizationLayerBWD bnormLayer2(KuDNN::Propagation::BACKWARD_DATA, srcInfo, diffDstInfo,
        statInfo, scaleShiftInfo, diffSrcInfo, diffScaleShiftInfo, KuDNN::NormalizationFlags::NONE);
    bnormLayer2 = bnormLayer1;
    for (size_t i = 0; i < g_Tensor_data.size() / 6; i += 6) {
        KuDNN::BatchNormalizationLayerBWD bnormLayer1(KuDNN::Propagation::BACKWARD_DATA,
            g_Tensor_data[i], g_Tensor_data[i + 1], g_Tensor_data[i + 2], g_Tensor_data[i + 3], 
            g_Tensor_data[i + 4], g_Tensor_data[i + 5], KuDNN::NormalizationFlags::NONE);
        KuDNN::BatchNormalizationLayerBWD bnormLayer2(KuDNN::Propagation::BACKWARD_DATA,
            g_Tensor_data[i], g_Tensor_data[i + 1], g_Tensor_data[i + 2], g_Tensor_data[i + 3], 
            g_Tensor_data[i + 4], g_Tensor_data[i + 5], KuDNN::NormalizationFlags::NONE);
        bnormLayer2 = bnormLayer1;
    }
}

static void BackwardMoveConstructor1()
{
    const KuDNN::TensorInfo srcInfo = {{1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABC};
    const KuDNN::TensorInfo diffSrcInfo = {{1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABC};
    const KuDNN::TensorInfo statInfo = {{1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::A};
    const KuDNN::TensorInfo scaleShiftInfo = {{1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::A};
    const KuDNN::TensorInfo diffScaleShiftInfo = {{1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::A};
    const KuDNN::TensorInfo diffDstInfo = {{1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABC};
    KuDNN::BatchNormalizationLayerBWD bnormLayer1(KuDNN::Propagation::BACKWARD_DATA, srcInfo, diffDstInfo,
        statInfo, scaleShiftInfo, diffSrcInfo, diffScaleShiftInfo, KuDNN::NormalizationFlags::NONE);
    KuDNN::BatchNormalizationLayerBWD bnormLayer2(std::move(bnormLayer1));
    for (size_t i = 0; i < g_Tensor_data.size() / 6; i += 6) {
        KuDNN::BatchNormalizationLayerBWD bnormLayer1(KuDNN::Propagation::BACKWARD_DATA,
            g_Tensor_data[i], g_Tensor_data[i + 1], g_Tensor_data[i + 2], g_Tensor_data[i + 3], 
            g_Tensor_data[i + 4], g_Tensor_data[i + 5], KuDNN::NormalizationFlags::NONE);
        KuDNN::BatchNormalizationLayerBWD bnormLayer2(std::move(bnormLayer1));
    }
}

static void BackwardMoveConstructor2()
{
    const KuDNN::TensorInfo srcInfo = {{1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABC};
    const KuDNN::TensorInfo diffSrcInfo = {{1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABC};
    const KuDNN::TensorInfo statInfo = {{1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::A};
    const KuDNN::TensorInfo scaleShiftInfo = {{1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::A};
    const KuDNN::TensorInfo diffScaleShiftInfo = {{1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::A};
    const KuDNN::TensorInfo diffDstInfo = {{1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABC};
    KuDNN::BatchNormalizationLayerBWD bnormLayer1(KuDNN::Propagation::BACKWARD_DATA, srcInfo, diffDstInfo,
        statInfo, scaleShiftInfo, diffSrcInfo, diffScaleShiftInfo, KuDNN::NormalizationFlags::NONE);
    KuDNN::BatchNormalizationLayerBWD bnormLayer2(KuDNN::Propagation::BACKWARD_DATA, srcInfo, diffDstInfo,
        statInfo, scaleShiftInfo, diffSrcInfo, diffScaleShiftInfo, KuDNN::NormalizationFlags::NONE);
    bnormLayer2 = std::move(bnormLayer1);
    for (size_t i = 0; i < g_Tensor_data.size() / 6; i += 6) {
        KuDNN::BatchNormalizationLayerBWD bnormLayer1(KuDNN::Propagation::BACKWARD_DATA,
            g_Tensor_data[i], g_Tensor_data[i + 1], g_Tensor_data[i + 2], g_Tensor_data[i + 3], 
            g_Tensor_data[i + 4], g_Tensor_data[i + 5], KuDNN::NormalizationFlags::NONE);
        KuDNN::BatchNormalizationLayerBWD bnormLayer2(KuDNN::Propagation::BACKWARD_DATA,
            g_Tensor_data[i], g_Tensor_data[i + 1], g_Tensor_data[i + 2], g_Tensor_data[i + 3], 
            g_Tensor_data[i + 4], g_Tensor_data[i + 5], KuDNN::NormalizationFlags::NONE);
        bnormLayer2 = std::move(bnormLayer1);
    }
}

// test cases
TEST(BnormGTests, FWDCopyConstructor1)
{
    EXPECT_NO_THROW(ForwardCopyConstructor1());
}

TEST(BnormGTests, FWDCopyConstructor2)
{
    EXPECT_NO_THROW(ForwardCopyConstructor2());
}

TEST(BnormGTests, FWDCopyConstructor3)
{
    EXPECT_NO_THROW(ForwardCopyConstructor3());
}

TEST(BnormGTests, FWDMoveConstructor1)
{
    EXPECT_NO_THROW(ForwardMoveConstructor1());
}

TEST(BnormGTests, FWDMoveConstructor2)
{
    EXPECT_NO_THROW(ForwardMoveConstructor2());
}

TEST(BnormGTests, BWDCopyConstructor1)
{
    EXPECT_NO_THROW(BackwardCopyConstructor1());
}

TEST(BnormGTests, BWDCopyConstructor2)
{
    EXPECT_NO_THROW(BackwardCopyConstructor2());
}

TEST(BnormGTests, BWDMoveConstructor1)
{
    EXPECT_NO_THROW(BackwardMoveConstructor1());
}

TEST(BnormGTests, BWDMoveConstructor2)
{
    EXPECT_NO_THROW(BackwardMoveConstructor2());
}