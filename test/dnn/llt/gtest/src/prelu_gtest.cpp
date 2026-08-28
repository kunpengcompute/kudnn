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
    const KuDNN::TensorInfo srcTensor = {{1, 1, 1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCDE};
    const KuDNN::TensorInfo weiTensor = {{1, 1, 1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCDE};
    const KuDNN::TensorInfo dstTensor = {{1, 1, 1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCDE};
    KuDNN::PReLULayerFWD preluLayer1(srcTensor, weiTensor, dstTensor);
    KuDNN::PReLULayerFWD preluLayer2(preluLayer1);
    for (size_t i = 0; i < g_Tensor_data.size() / 3; i += 3) {
        KuDNN::PReLULayerFWD preluLayer1(g_Tensor_data[i], g_Tensor_data[i + 1], g_Tensor_data[i + 2]);
        KuDNN::PReLULayerFWD preluLayer2(preluLayer1);
    }
}

static void ForwardCopyConstructor2()
{
    const KuDNN::TensorInfo srcTensor = {{1, 1, 1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCDE};
    const KuDNN::TensorInfo weiTensor = {{1, 1, 1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCDE};
    const KuDNN::TensorInfo dstTensor = {{1, 1, 1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCDE};
    KuDNN::PReLULayerFWD preluLayer1(srcTensor, weiTensor, dstTensor);
    KuDNN::PReLULayerFWD preluLayer2(srcTensor, weiTensor, dstTensor);
    preluLayer2 = preluLayer1;
    for (size_t i = 0; i < g_Tensor_data.size() / 3; i += 3) {
        KuDNN::PReLULayerFWD preluLayer1(g_Tensor_data[i], g_Tensor_data[i + 1], g_Tensor_data[i + 2]);
        KuDNN::PReLULayerFWD preluLayer2(g_Tensor_data[i], g_Tensor_data[i + 1], g_Tensor_data[i + 2]);
        preluLayer2 = preluLayer1;
    }
}

static void ForwardMoveConstructor1()
{
    const KuDNN::TensorInfo srcTensor = {{1, 1, 1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCDE};
    const KuDNN::TensorInfo weiTensor = {{1, 1, 1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCDE};
    const KuDNN::TensorInfo dstTensor = {{1, 1, 1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCDE};
    KuDNN::PReLULayerFWD preluLayer1(srcTensor, weiTensor, dstTensor);
    KuDNN::PReLULayerFWD preluLayer2(std::move(preluLayer1));
    for (size_t i = 0; i < g_Tensor_data.size() / 3; i += 3) {
        KuDNN::PReLULayerFWD preluLayer1(g_Tensor_data[i], g_Tensor_data[i + 1], g_Tensor_data[i + 2]);
        KuDNN::PReLULayerFWD preluLayer2(std::move(preluLayer1));
    }
}

static void ForwardMoveConstructor2()
{
    const KuDNN::TensorInfo srcTensor = {{1, 1, 1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCDE};
    const KuDNN::TensorInfo weiTensor = {{1, 1, 1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCDE};
    const KuDNN::TensorInfo dstTensor = {{1, 1, 1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCDE};
    KuDNN::PReLULayerFWD preluLayer1(srcTensor, weiTensor, dstTensor);
    KuDNN::PReLULayerFWD preluLayer2(srcTensor, weiTensor, dstTensor);
    preluLayer2 = std::move(preluLayer1);
    for (size_t i = 0; i < g_Tensor_data.size() / 3; i += 3) {
        KuDNN::PReLULayerFWD preluLayer1(g_Tensor_data[i], g_Tensor_data[i + 1], g_Tensor_data[i + 2]);
        KuDNN::PReLULayerFWD preluLayer2(g_Tensor_data[i], g_Tensor_data[i + 1], g_Tensor_data[i + 2]);
        preluLayer2 = std::move(preluLayer1);
    }
}

// backward
static void BackwardCopyConstructor1()
{
    const KuDNN::TensorInfo srcTensor = {{1, 1, 1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCDE};
    const KuDNN::TensorInfo diffSrcTensor = {{1, 1, 1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCDE};
    const KuDNN::TensorInfo weiTensor = {{1, 1, 1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCDE};
    const KuDNN::TensorInfo diffWeiTensor = {{1, 1, 1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCDE};
    const KuDNN::TensorInfo dstTensor = {{1, 1, 1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCDE};
    KuDNN::PReLULayerBWD preluLayer1(srcTensor, diffSrcTensor, weiTensor, diffWeiTensor, dstTensor);
    KuDNN::PReLULayerBWD preluLayer2(preluLayer1);
    for (size_t i = 0; i < g_Tensor_data.size() / 5; i += 5) {
        KuDNN::PReLULayerBWD preluLayer1(g_Tensor_data[i], g_Tensor_data[i + 1], g_Tensor_data[i + 2], g_Tensor_data[i + 3],
            g_Tensor_data[i + 4]);
        KuDNN::PReLULayerBWD preluLayer2(preluLayer1);
    }
}

static void BackwardCopyConstructor2()
{
    const KuDNN::TensorInfo srcTensor = {{1, 1, 1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCDE};
    const KuDNN::TensorInfo diffSrcTensor = {{1, 1, 1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCDE};
    const KuDNN::TensorInfo weiTensor = {{1, 1, 1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCDE};
    const KuDNN::TensorInfo diffWeiTensor = {{1, 1, 1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCDE};
    const KuDNN::TensorInfo dstTensor = {{1, 1, 1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCDE};
    KuDNN::PReLULayerBWD preluLayer1(srcTensor, diffSrcTensor, weiTensor, diffWeiTensor, dstTensor);
    KuDNN::PReLULayerBWD preluLayer2(preluLayer1);
    preluLayer2 = preluLayer1;
    for (size_t i = 0; i < g_Tensor_data.size() / 5; i += 5) {
        KuDNN::PReLULayerBWD preluLayer1(g_Tensor_data[i], g_Tensor_data[i + 1], g_Tensor_data[i + 2], g_Tensor_data[i + 3],
            g_Tensor_data[i + 4]);
        KuDNN::PReLULayerBWD preluLayer2(preluLayer1);
        preluLayer2 = preluLayer1;
    }
}

static void BackwardMoveConstructor1()
{
    const KuDNN::TensorInfo srcTensor = {{1, 1, 1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCDE};
    const KuDNN::TensorInfo diffSrcTensor = {{1, 1, 1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCDE};
    const KuDNN::TensorInfo weiTensor = {{1, 1, 1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCDE};
    const KuDNN::TensorInfo diffWeiTensor = {{1, 1, 1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCDE};
    const KuDNN::TensorInfo dstTensor = {{1, 1, 1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCDE};
    KuDNN::PReLULayerBWD preluLayer1(srcTensor, diffSrcTensor, weiTensor, diffWeiTensor, dstTensor);
    KuDNN::PReLULayerBWD preluLayer2(std::move(preluLayer1));
    for (size_t i = 0; i < g_Tensor_data.size() / 5; i += 5) {
        KuDNN::PReLULayerBWD preluLayer1(g_Tensor_data[i], g_Tensor_data[i + 1], g_Tensor_data[i + 2], g_Tensor_data[i + 3],
            g_Tensor_data[i + 4]);
        KuDNN::PReLULayerBWD preluLayer2(std::move(preluLayer1));
    }
}

static void BackwardMoveConstructor2()
{
    const KuDNN::TensorInfo srcTensor = {{1, 1, 1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCDE};
    const KuDNN::TensorInfo diffSrcTensor = {{1, 1, 1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCDE};
    const KuDNN::TensorInfo weiTensor = {{1, 1, 1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCDE};
    const KuDNN::TensorInfo diffWeiTensor = {{1, 1, 1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCDE};
    const KuDNN::TensorInfo dstTensor = {{1, 1, 1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCDE};
    KuDNN::PReLULayerBWD preluLayer1(srcTensor, diffSrcTensor, weiTensor, diffWeiTensor, dstTensor);
    KuDNN::PReLULayerBWD preluLayer2(preluLayer1);
    preluLayer2 = std::move(preluLayer1);
    for (size_t i = 0; i < g_Tensor_data.size() / 5; i += 5) {
        KuDNN::PReLULayerBWD preluLayer1(g_Tensor_data[i], g_Tensor_data[i + 1], g_Tensor_data[i + 2], g_Tensor_data[i + 3],
            g_Tensor_data[i + 4]);
        KuDNN::PReLULayerBWD preluLayer2(preluLayer1);
        preluLayer2 = std::move(preluLayer1);
    }
}


// test cases
TEST(PReLUGTests, FWDCopyConstructor1)
{
    EXPECT_NO_THROW(ForwardCopyConstructor1());
}

TEST(PReLUGTests, FWDCopyConstructor2)
{
    EXPECT_NO_THROW(ForwardCopyConstructor2());
}

TEST(PReLUGTests, FWDMoveConstructor1)
{
    EXPECT_NO_THROW(ForwardMoveConstructor1());
}

TEST(PReLUGTests, FWDMoveConstructor2)
{
    EXPECT_NO_THROW(ForwardMoveConstructor2());
}

TEST(PReLUGTests, BWDCopyConstructor1)
{
    EXPECT_NO_THROW(BackwardCopyConstructor1());
}

TEST(PReLUGTests, BWDCopyConstructor2)
{
    EXPECT_NO_THROW(BackwardCopyConstructor2());
}

TEST(PReLUGTests, BWDMoveConstructor1)
{
    EXPECT_NO_THROW(BackwardMoveConstructor1());
}

TEST(PReLUGTests, BWDMoveConstructor2)
{
    EXPECT_NO_THROW(BackwardMoveConstructor2());
}