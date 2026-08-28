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
    KuDNN::ConcatLayer concatLayer1(srcInfoVec, 0, dstTensor);
    KuDNN::ConcatLayer concatLayer2(concatLayer1);
    for (size_t i = 0; i < g_Tensor_data.size() / 3; i += 3) {
        const std::vector<KuDNN::TensorInfo> srcInfoVec = {g_Tensor_data[i], g_Tensor_data[i + 1]};
        KuDNN::ConcatLayer concatLayer1(srcInfoVec, 0, g_Tensor_data[i + 2]);
        KuDNN::ConcatLayer concatLayer2(concatLayer1);
    }
}

static void CopyConstructor2()
{
    const KuDNN::TensorInfo src0Tensor = {{1, 1, 1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCDE};
    const KuDNN::TensorInfo src1Tensor = {{1, 1, 1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCDE};
    const KuDNN::TensorInfo dstTensor = {{1, 1, 1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCDE};
    const std::vector<KuDNN::TensorInfo> srcInfoVec = {src0Tensor, src1Tensor};
    KuDNN::ConcatLayer concatLayer1(srcInfoVec, 0, dstTensor);
    KuDNN::ConcatLayer concatLayer2(concatLayer1);
    concatLayer2 = concatLayer1;
    for (size_t i = 0; i < g_Tensor_data.size() / 3; i += 3) {
        const std::vector<KuDNN::TensorInfo> srcInfoVec = {g_Tensor_data[i], g_Tensor_data[i + 1]};
        KuDNN::ConcatLayer concatLayer1(srcInfoVec, 0, g_Tensor_data[i + 2]);
        KuDNN::ConcatLayer concatLayer2(concatLayer1);
        concatLayer2 = concatLayer1;
    }
}

static void MoveConstructor1()
{
    const KuDNN::TensorInfo src0Tensor = {{1, 1, 1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCDE};
    const KuDNN::TensorInfo src1Tensor = {{1, 1, 1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCDE};
    const KuDNN::TensorInfo dstTensor = {{1, 1, 1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCDE};
    const std::vector<KuDNN::TensorInfo> srcInfoVec = {src0Tensor, src1Tensor};
    KuDNN::ConcatLayer concatLayer1(srcInfoVec, 0, dstTensor);
    KuDNN::ConcatLayer concatLayer2(std::move(concatLayer1));
    for (size_t i = 0; i < g_Tensor_data.size() / 3; i += 3) {
        const std::vector<KuDNN::TensorInfo> srcInfoVec = {g_Tensor_data[i], g_Tensor_data[i + 1]};
        KuDNN::ConcatLayer concatLayer1(srcInfoVec, 0, g_Tensor_data[i + 2]);
        KuDNN::ConcatLayer concatLayer2(std::move(concatLayer1));
    }
}

static void MoveConstructor2()
{
    const KuDNN::TensorInfo src0Tensor = {{1, 1, 1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCDE};
    const KuDNN::TensorInfo src1Tensor = {{1, 1, 1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCDE};
    const KuDNN::TensorInfo dstTensor = {{1, 1, 1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCDE};
    const std::vector<KuDNN::TensorInfo> srcInfoVec = {src0Tensor, src1Tensor};
    KuDNN::ConcatLayer concatLayer1(srcInfoVec, 0, dstTensor);
    KuDNN::ConcatLayer concatLayer2(srcInfoVec, 0, dstTensor);
    concatLayer2 = std::move(concatLayer1);
    for (size_t i = 0; i < g_Tensor_data.size() / 3; i += 3) {
        const std::vector<KuDNN::TensorInfo> srcInfoVec = {g_Tensor_data[i], g_Tensor_data[i + 1]};
        KuDNN::ConcatLayer concatLayer1(srcInfoVec, 0, g_Tensor_data[i + 2]);
        KuDNN::ConcatLayer concatLayer2(srcInfoVec, 0, g_Tensor_data[i + 2]);
        concatLayer2 = std::move(concatLayer1);
    }
}

// test cases
TEST(ConcatGTests, CopyConstructor1)
{
    EXPECT_NO_THROW(CopyConstructor1());
}

TEST(ConcatGTests, CopyConstructor2)
{
    EXPECT_NO_THROW(CopyConstructor2());
}

TEST(ConcatGTests, MoveConstructor1)
{
    EXPECT_NO_THROW(MoveConstructor1());
}

TEST(ConcatGTests, MoveConstructor2)
{
    EXPECT_NO_THROW(MoveConstructor2());
}