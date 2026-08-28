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

static void CopyConstructor1()
{
    const KuDNN::TensorInfo srcInfo = {{2, 1, 1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCDE};
    const KuDNN::TensorInfo dstInfo = {{2, 1, 1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCDE};
    KuDNN::SizeType axis = 0;
    KuDNN::SizeType group = 1;
    KuDNN::ShuffleLayer shuffleLayer1(srcInfo, dstInfo, axis, group);
    KuDNN::ShuffleLayer shuffleLayer2(shuffleLayer1);
    for (size_t i = 0; i < g_Tensor_data.size() / 2; i += 2) {
        KuDNN::ShuffleLayer shuffleLayer1(g_Tensor_data[i], g_Tensor_data[i + 1], axis, group);
        KuDNN::ShuffleLayer shuffleLayer2(shuffleLayer1);
    }
}

static void CopyConstructor2()
{
    const KuDNN::TensorInfo srcInfo = {{2, 1, 1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCDE};
    const KuDNN::TensorInfo dstInfo = {{2, 1, 1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCDE};
    KuDNN::SizeType axis = 0;
    KuDNN::SizeType group = 1;
    KuDNN::ShuffleLayer shuffleLayer1(srcInfo, dstInfo, axis, group);
    KuDNN::ShuffleLayer shuffleLayer2(shuffleLayer1);
    shuffleLayer2 = shuffleLayer1;
    for (size_t i = 0; i < g_Tensor_data.size() / 2; i += 2) {
        KuDNN::ShuffleLayer shuffleLayer1(g_Tensor_data[i], g_Tensor_data[i + 1], axis, group);
        KuDNN::ShuffleLayer shuffleLayer2(shuffleLayer1);
        shuffleLayer2 = shuffleLayer1;
    }
}

static void MoveConstructor1()
{
    const KuDNN::TensorInfo srcInfo = {{2, 1, 1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCDE};
    const KuDNN::TensorInfo dstInfo = {{2, 1, 1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCDE};
    KuDNN::SizeType axis = 0;
    KuDNN::SizeType group = 1;
    KuDNN::ShuffleLayer shuffleLayer1(srcInfo, dstInfo, axis, group);
    KuDNN::ShuffleLayer shuffleLayer2(std::move(shuffleLayer1));
    for (size_t i = 0; i < g_Tensor_data.size() / 2; i += 2) {
        KuDNN::ShuffleLayer shuffleLayer1(g_Tensor_data[i], g_Tensor_data[i + 1], axis, group);
        KuDNN::ShuffleLayer shuffleLayer2(std::move(shuffleLayer1));
    }
}

static void MoveConstructor2()
{
    const KuDNN::TensorInfo srcInfo = {{2, 1, 1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCDE};
    const KuDNN::TensorInfo dstInfo = {{2, 1, 1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCDE};
    KuDNN::SizeType axis = 0;
    KuDNN::SizeType group = 1;
    KuDNN::ShuffleLayer shuffleLayer1(srcInfo, dstInfo, axis, group);
    KuDNN::ShuffleLayer shuffleLayer2(srcInfo, dstInfo, axis, group);
    shuffleLayer2 = std::move(shuffleLayer1);
    for (size_t i = 0; i < g_Tensor_data.size() / 2; i += 2) {
        KuDNN::ShuffleLayer shuffleLayer1(g_Tensor_data[i], g_Tensor_data[i + 1], axis, group);
        KuDNN::ShuffleLayer shuffleLayer2(g_Tensor_data[i], g_Tensor_data[i + 1], axis, group);
        shuffleLayer2 = std::move(shuffleLayer1);
    }
}

// test cases
TEST(ShuffleGTests, CopyConstructor1)
{
    EXPECT_NO_THROW(CopyConstructor1());
}

TEST(ShuffleGTests, CopyConstructor2)
{
    EXPECT_NO_THROW(CopyConstructor2());
}

TEST(ShuffleGTests, MoveConstructor1)
{
    EXPECT_NO_THROW(MoveConstructor1());
}

TEST(ShuffleGTests, MoveConstructor2)
{
    EXPECT_NO_THROW(MoveConstructor2());
}