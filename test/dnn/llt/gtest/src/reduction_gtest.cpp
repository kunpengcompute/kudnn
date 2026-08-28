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
    const KuDNN::TensorInfo srcTensor = {{2, 2, 2, 2, 2}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCDE};
    const KuDNN::TensorInfo dstTensor = {{1, 1, 1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCDE};
    KuDNN::ReductionLayer reductionLayer1(srcTensor, dstTensor, KuDNN::ReductionFunction::MAX);
    KuDNN::ReductionLayer reductionLayer2(reductionLayer1);
    for (size_t i = 0; i < g_Tensor_data.size() / 2; i += 2) {
        KuDNN::ReductionLayer reductionLayer1(g_Tensor_data[i], g_Tensor_data[i + 1], KuDNN::ReductionFunction::MAX);
        KuDNN::ReductionLayer reductionLayer2(reductionLayer1);
    }
}

static void CopyConstructor2()
{
    const KuDNN::TensorInfo srcTensor = {{2, 2, 2, 2, 2}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCDE};
    const KuDNN::TensorInfo dstTensor = {{1, 1, 1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCDE};
    KuDNN::ReductionLayer reductionLayer1(srcTensor, dstTensor, KuDNN::ReductionFunction::MAX);
    KuDNN::ReductionLayer reductionLayer2(reductionLayer1);
    reductionLayer2 = reductionLayer1;
    for (size_t i = 0; i < g_Tensor_data.size() / 2; i += 2) {
        KuDNN::ReductionLayer reductionLayer1(g_Tensor_data[i], g_Tensor_data[i + 1], KuDNN::ReductionFunction::MAX);
        KuDNN::ReductionLayer reductionLayer2(reductionLayer1);
        reductionLayer2 = reductionLayer1;
    }
}

static void MoveConstructor1()
{
    const KuDNN::TensorInfo srcTensor = {{2, 2, 2, 2, 2}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCDE};
    const KuDNN::TensorInfo dstTensor = {{1, 1, 1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCDE};
    KuDNN::ReductionLayer reductionLayer1(srcTensor, dstTensor, KuDNN::ReductionFunction::MAX);
    KuDNN::ReductionLayer reductionLayer2(std::move(reductionLayer1));
    for (size_t i = 0; i < g_Tensor_data.size() / 2; i += 2) {
        KuDNN::ReductionLayer reductionLayer1(g_Tensor_data[i], g_Tensor_data[i + 1], KuDNN::ReductionFunction::MAX);
        KuDNN::ReductionLayer reductionLayer2(std::move(reductionLayer1));
    }
}

static void MoveConstructor2()
{
    const KuDNN::TensorInfo srcTensor = {{2, 2, 2, 2, 2}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCDE};
    const KuDNN::TensorInfo dstTensor = {{1, 1, 1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCDE};
    KuDNN::ReductionLayer reductionLayer1(srcTensor, dstTensor, KuDNN::ReductionFunction::MAX);
    KuDNN::ReductionLayer reductionLayer2(reductionLayer1);
    reductionLayer2 = std::move(reductionLayer1);
    for (size_t i = 0; i < g_Tensor_data.size() / 2; i += 2) {
        KuDNN::ReductionLayer reductionLayer1(g_Tensor_data[i], g_Tensor_data[i + 1], KuDNN::ReductionFunction::MAX);
        KuDNN::ReductionLayer reductionLayer2(reductionLayer1);
        reductionLayer2 = std::move(reductionLayer1);
    }
}

// test cases
TEST(ReductionGTests, CopyConstructor1)
{
    EXPECT_NO_THROW(CopyConstructor1());
}

TEST(ReductionGTests, CopyConstructor2)
{
    EXPECT_NO_THROW(CopyConstructor2());
}

TEST(ReductionGTests, MoveConstructor1)
{
    EXPECT_NO_THROW(MoveConstructor1());
}

TEST(ReductionGTests, MoveConstructor2)
{
    EXPECT_NO_THROW(MoveConstructor2());
}
