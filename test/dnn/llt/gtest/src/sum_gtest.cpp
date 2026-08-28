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
    const KuDNN::TensorInfo src0Tensor = {{1, 1, 1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCDE};
    const KuDNN::TensorInfo src1Tensor = {{1, 1, 1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCDE};
    const KuDNN::TensorInfo dstTensor = {{1, 1, 1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCDE};
    const std::vector<KuDNN::TensorInfo> srcInfoVec = {src0Tensor, src1Tensor};
    const std::vector<float> scales = {0.15, -0.15};
    KuDNN::SumLayer sumLayer1(srcInfoVec, scales.data(), dstTensor);
    KuDNN::SumLayer sumLayer2(sumLayer1);
    for (size_t i = 0; i < g_Tensor_data.size() / 3; i += 3) {
        const std::vector<KuDNN::TensorInfo> srcInfoVec = {g_Tensor_data[i], g_Tensor_data[i + 1]};
        KuDNN::SumLayer sumLayer1(srcInfoVec, scales.data(), g_Tensor_data[i + 2]);
        KuDNN::SumLayer sumLayer2(sumLayer1);
    }
}

static void CopyConstructor2()
{
    const KuDNN::TensorInfo src0Tensor = {{1, 1, 1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCDE};
    const KuDNN::TensorInfo src1Tensor = {{1, 1, 1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCDE};
    const KuDNN::TensorInfo dstTensor = {{1, 1, 1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCDE};
    const std::vector<KuDNN::TensorInfo> srcInfoVec = {src0Tensor, src1Tensor};
    const std::vector<float> scales = {0.15, -0.15};
    KuDNN::SumLayer sumLayer1(srcInfoVec, scales.data(), dstTensor);
    KuDNN::SumLayer sumLayer2(sumLayer1);
    sumLayer2 = sumLayer1;
    for (size_t i = 0; i < g_Tensor_data.size() / 3; i += 3) {
        const std::vector<KuDNN::TensorInfo> srcInfoVec = {g_Tensor_data[i], g_Tensor_data[i + 1]};
        KuDNN::SumLayer sumLayer1(srcInfoVec, scales.data(), g_Tensor_data[i + 2]);
        KuDNN::SumLayer sumLayer2(sumLayer1);
        sumLayer2 = sumLayer1;
    }
}

static void MoveConstructor1()
{
    const KuDNN::TensorInfo src0Tensor = {{1, 1, 1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCDE};
    const KuDNN::TensorInfo src1Tensor = {{1, 1, 1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCDE};
    const KuDNN::TensorInfo dstTensor = {{1, 1, 1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCDE};
    const std::vector<KuDNN::TensorInfo> srcInfoVec = {src0Tensor, src1Tensor};
    const std::vector<float> scales = {0.15, -0.15};
    KuDNN::SumLayer sumLayer1(srcInfoVec, scales.data(), dstTensor);
    KuDNN::SumLayer sumLayer2(std::move(sumLayer1));
    for (size_t i = 0; i < g_Tensor_data.size() / 3; i += 3) {
        const std::vector<KuDNN::TensorInfo> srcInfoVec = {g_Tensor_data[i], g_Tensor_data[i + 1]};
        KuDNN::SumLayer sumLayer1(srcInfoVec, scales.data(), g_Tensor_data[i + 2]);
        KuDNN::SumLayer sumLayer2(std::move(sumLayer1));
    }
}

static void MoveConstructor2()
{
    const KuDNN::TensorInfo src0Tensor = {{1, 1, 1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCDE};
    const KuDNN::TensorInfo src1Tensor = {{1, 1, 1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCDE};
    const KuDNN::TensorInfo dstTensor = {{1, 1, 1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCDE};
    const std::vector<KuDNN::TensorInfo> srcInfoVec = {src0Tensor, src1Tensor};
    const std::vector<float> scales = {0.15, -0.15};
    KuDNN::SumLayer sumLayer1(srcInfoVec, scales.data(), dstTensor);
    KuDNN::SumLayer sumLayer2(srcInfoVec, scales.data(), dstTensor);
    sumLayer2 = (std::move(sumLayer1));
    for (size_t i = 0; i < g_Tensor_data.size() / 3; i += 3) {
        const std::vector<KuDNN::TensorInfo> srcInfoVec = {g_Tensor_data[i], g_Tensor_data[i + 1]};
        KuDNN::SumLayer sumLayer1(srcInfoVec, scales.data(), g_Tensor_data[i + 2]);
        KuDNN::SumLayer sumLayer2(srcInfoVec, scales.data(), g_Tensor_data[i + 2]);
        sumLayer2 = (std::move(sumLayer1));
    }
}

// test cases
TEST(SumGTests, CopyConstructor1)
{
    EXPECT_NO_THROW(CopyConstructor1());
}

TEST(SumGTests, CopyConstructor2)
{
    EXPECT_NO_THROW(CopyConstructor2());
}

TEST(SumGTests, MoveConstructor1)
{
    EXPECT_NO_THROW(MoveConstructor1());
}

TEST(SumGTests, MoveConstructor2)
{
    EXPECT_NO_THROW(MoveConstructor2());
}