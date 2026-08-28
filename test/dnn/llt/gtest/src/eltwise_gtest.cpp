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

#include <cstdlib>
#include <exception>
#include <iostream>
#include <vector>

#include "gtest/gtest.h"
#include "kudnn.hpp"
#include "kudnn_tensor_data.hpp"

using SizeType = KuDNN::SizeType;

// forward
static void ForwardCopyConstructor1()
{
    const KuDNN::TensorInfo srcTensor = {{1, 1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCD};
    const KuDNN::TensorInfo dstTensor = {{1, 1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCD};
    KuDNN::ActivationFunction algKind = KuDNN::ActivationFunction::ABS;
    KuDNN::ActivationLayerFWD activationLayer1(srcTensor, dstTensor, algKind);
    KuDNN::ActivationLayerFWD activationLayer2(activationLayer1);
    for (size_t i = 0; i < g_Tensor_data.size() / 2; i += 2) {
        KuDNN::ActivationLayerFWD activationLayer1(g_Tensor_data[i], g_Tensor_data[i + 1], algKind);
        KuDNN::ActivationLayerFWD activationLayer2(activationLayer1);
    }
}

static void ForwardCopyConstructor2()
{
    const KuDNN::TensorInfo srcTensor = {{1, 1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCD};
    const KuDNN::TensorInfo dstTensor = {{1, 1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCD};
    KuDNN::ActivationFunction algKind = KuDNN::ActivationFunction::ABS;
    const KuDNN::ActivationLayerFWD activationLayer1 = KuDNN::ActivationLayerFWD(srcTensor, dstTensor, algKind);
    KuDNN::ActivationLayerFWD activationLayer2 = KuDNN::ActivationLayerFWD(srcTensor, dstTensor, algKind);
    activationLayer2 = activationLayer1;
    for (size_t i = 0; i < g_Tensor_data.size() / 2; i += 2) {
        const KuDNN::ActivationLayerFWD activationLayer1(g_Tensor_data[i], g_Tensor_data[i + 1], algKind);
        KuDNN::ActivationLayerFWD activationLayer2(g_Tensor_data[i], g_Tensor_data[i + 1], algKind);
        activationLayer2 = activationLayer1;
    }
}

static void ForwardMoveConstructor1()
{
    const KuDNN::TensorInfo srcTensor = {{1, 1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCD};
    const KuDNN::TensorInfo dstTensor = {{1, 1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCD};
    KuDNN::ActivationFunction algKind = KuDNN::ActivationFunction::ABS;
    KuDNN::ActivationLayerFWD activationLayer1(srcTensor, dstTensor, algKind);
    KuDNN::ActivationLayerFWD activationLayer2(std::move(activationLayer1));
    for (size_t i = 0; i < g_Tensor_data.size() / 2; i += 2) {
        KuDNN::ActivationLayerFWD activationLayer1(g_Tensor_data[i], g_Tensor_data[i + 1], algKind);
        KuDNN::ActivationLayerFWD activationLayer2(std::move(activationLayer1));
    }
}

static void ForwardMoveConstructor2()
{
    const KuDNN::TensorInfo srcTensor = {{1, 1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCD};
    const KuDNN::TensorInfo dstTensor = {{1, 1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCD};
    KuDNN::ActivationFunction algKind = KuDNN::ActivationFunction::ABS;
    KuDNN::ActivationLayerFWD activationLayer1(srcTensor, dstTensor, algKind);
    KuDNN::ActivationLayerFWD activationLayer2(srcTensor, dstTensor, algKind);
    activationLayer2 = std::move(activationLayer1);
    for (size_t i = 0; i < g_Tensor_data.size() / 2; i += 2) {
        KuDNN::ActivationLayerFWD activationLayer1(g_Tensor_data[i], g_Tensor_data[i + 1], algKind);
        KuDNN::ActivationLayerFWD activationLayer2(g_Tensor_data[i], g_Tensor_data[i + 1], algKind);
        activationLayer2 = std::move(activationLayer1);
    }
}

// backward
static void BackwardCopyConstructor1()
{
    const KuDNN::TensorInfo srcTensor = {{1, 1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCD};
    const KuDNN::TensorInfo diffDstTensor = {{1, 1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCD};
    const KuDNN::TensorInfo diffSrcTensor = {{1, 1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCD};
    KuDNN::ActivationFunction algKind = KuDNN::ActivationFunction::RELU;
    KuDNN::ActivationLayerBWD activationLayer1(diffSrcTensor, diffDstTensor, srcTensor, algKind);
    KuDNN::ActivationLayerBWD activationLayer2(activationLayer1);
    for (size_t i = 0; i < g_Tensor_data.size() / 3; i += 3) {
        KuDNN::ActivationLayerBWD activationLayer1(g_Tensor_data[i], g_Tensor_data[i + 1], g_Tensor_data[i + 2], algKind);
        KuDNN::ActivationLayerBWD activationLayer2(activationLayer1);
    }
}

static void BackwardCopyConstructor2()
{
    const KuDNN::TensorInfo srcTensor = {{1, 1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCD};
    const KuDNN::TensorInfo diffDstTensor = {{1, 1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCD};
    const KuDNN::TensorInfo diffSrcTensor = {{1, 1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCD};
    KuDNN::ActivationFunction algKind = KuDNN::ActivationFunction::RELU;
    KuDNN::ActivationLayerBWD activationLayer1(diffSrcTensor, diffDstTensor, srcTensor, algKind);
    KuDNN::ActivationLayerBWD activationLayer2(diffSrcTensor, diffDstTensor, srcTensor, algKind);
    activationLayer2 = activationLayer1;
    for (size_t i = 0; i < g_Tensor_data.size() / 3; i += 3) {
        KuDNN::ActivationLayerBWD activationLayer1(g_Tensor_data[i], g_Tensor_data[i + 1], g_Tensor_data[i + 2], algKind);
        KuDNN::ActivationLayerBWD activationLayer2(g_Tensor_data[i], g_Tensor_data[i + 1], g_Tensor_data[i + 2], algKind);
        activationLayer2 = activationLayer1;
    }
}

static void BackwardMoveConstructor1()
{
    const KuDNN::TensorInfo srcTensor = {{1, 1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCD};
    const KuDNN::TensorInfo diffDstTensor = {{1, 1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCD};
    const KuDNN::TensorInfo diffSrcTensor = {{1, 1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCD};
    KuDNN::ActivationFunction algKind = KuDNN::ActivationFunction::RELU;
    KuDNN::ActivationLayerBWD activationLayer1(diffSrcTensor, diffDstTensor, srcTensor, algKind);
    KuDNN::ActivationLayerBWD activationLayer2(std::move(activationLayer1));
    for (size_t i = 0; i < g_Tensor_data.size() / 3; i += 3) {
        KuDNN::ActivationLayerBWD activationLayer1(g_Tensor_data[i], g_Tensor_data[i + 1], g_Tensor_data[i + 2], algKind);
        KuDNN::ActivationLayerBWD activationLayer2(std::move(activationLayer1));
    }
}

static void BackwardMoveConstructor2()
{
    const KuDNN::TensorInfo srcTensor = {{1, 1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCD};
    const KuDNN::TensorInfo diffDstTensor = {{1, 1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCD};
    const KuDNN::TensorInfo diffSrcTensor = {{1, 1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCD};
    KuDNN::ActivationFunction algKind = KuDNN::ActivationFunction::RELU;
    KuDNN::ActivationLayerBWD activationLayer1(diffSrcTensor, diffDstTensor, srcTensor, algKind);
    KuDNN::ActivationLayerBWD activationLayer2(diffSrcTensor, diffDstTensor, srcTensor, algKind);
    activationLayer2 = std::move(activationLayer1);
    for (size_t i = 0; i < g_Tensor_data.size() / 3; i += 3) {
        KuDNN::ActivationLayerBWD activationLayer1(g_Tensor_data[i], g_Tensor_data[i + 1], g_Tensor_data[i + 2], algKind);
        KuDNN::ActivationLayerBWD activationLayer2(g_Tensor_data[i], g_Tensor_data[i + 1], g_Tensor_data[i + 2], algKind);
        activationLayer2 = std::move(activationLayer1);
    }
}

// test cases
TEST(EltwiseGTests, FWDCopyConstructor1)
{
    EXPECT_NO_THROW(ForwardCopyConstructor1());
}

TEST(EltwiseGTests, FWDCopyConstructor)
{
    EXPECT_NO_THROW(ForwardCopyConstructor2());
}

TEST(EltwiseGTests, FWDMoveConstructor1)
{
    EXPECT_NO_THROW(ForwardMoveConstructor1());
}

TEST(EltwiseGTests, FWDMoveConstructor2)
{
    EXPECT_NO_THROW(ForwardMoveConstructor2());
}

TEST(EltwiseGTests, BWDCopyConstructor1)
{
    EXPECT_NO_THROW(BackwardCopyConstructor1());
}

TEST(EltwiseGTests, BWDCopyConstructor2)
{
    EXPECT_NO_THROW(BackwardCopyConstructor2());
}

TEST(EltwiseGTests, BWDMoveConstructor1)
{
    EXPECT_NO_THROW(BackwardMoveConstructor1());
}

TEST(EltwiseGTests, BWDMoveConstructor2)
{
    EXPECT_NO_THROW(BackwardMoveConstructor2());
}

// forward
template <typename T>
static bool ForwardFunc1(const KuDNN::TensorInfo &srcTensor, const KuDNN::TensorInfo &dstTensor, KuDNN::ActivationFunction algKind)
{
    SizeType srcSize = srcTensor.GetTotalTensorSize();
    SizeType dstSize = dstTensor.GetTotalTensorSize();

    T *src = (T *)malloc(srcSize * sizeof(T));
    T *dst = (T *)malloc(dstSize * sizeof(T));

    KuDNN::ActivationLayerFWD activationLayer(srcTensor, dstTensor, algKind);

    activationLayer.Run(src, dst);
    return true;
}

// backward
template <typename T>
static bool BackwardFunc1(const KuDNN::TensorInfo &diffSrcTensor, const KuDNN::TensorInfo &diffDstTensor, const KuDNN::TensorInfo &srcTensor, KuDNN::ActivationFunction algKind)
{
    SizeType diffSrcSize = diffSrcTensor.GetTotalTensorSize();
    SizeType diffDstSize = diffDstTensor.GetTotalTensorSize();
    SizeType srcSize = srcTensor.GetTotalTensorSize();

    T *src = (T *)malloc(srcSize * sizeof(T));
    T *diffDst = (T *)malloc(diffDstSize * sizeof(T));
    T *diffSrc = (T *)malloc(diffSrcSize * sizeof(T));

    KuDNN::ActivationLayerBWD activationLayer(diffSrcTensor, diffDstTensor, srcTensor, algKind);
    activationLayer.Run(diffSrc, diffDst, src);
    return true;
}

// 获取所有枚举值的函数
std::vector<KuDNN::ActivationFunction> getAllActivationFunctions() {
    return {
        KuDNN::ActivationFunction::UNIMPLEMENTED,
        KuDNN::ActivationFunction::ABS,
        KuDNN::ActivationFunction::EXP,
        KuDNN::ActivationFunction::LINEAR,
        KuDNN::ActivationFunction::LOG,
        KuDNN::ActivationFunction::RELU,
        KuDNN::ActivationFunction::ROUND,
        KuDNN::ActivationFunction::SQRT,
        KuDNN::ActivationFunction::SQUARE,
        KuDNN::ActivationFunction::TANH,
        KuDNN::ActivationFunction::SIGMOID,
        KuDNN::ActivationFunction::POW,
        KuDNN::ActivationFunction::CLIP,
        KuDNN::ActivationFunction::CLIP_V2,
        KuDNN::ActivationFunction::HARDSIGMOID,
        KuDNN::ActivationFunction::HARDSWISH,
        KuDNN::ActivationFunction::EXP_DST,
        KuDNN::ActivationFunction::LOGISTIC_DST,
        KuDNN::ActivationFunction::TANH_DST,
        KuDNN::ActivationFunction::ELU,
        KuDNN::ActivationFunction::GELU_TANH,
        KuDNN::ActivationFunction::CLIP_V2_DST,
        KuDNN::ActivationFunction::ELU_DST,
        KuDNN::ActivationFunction::RELU_DST,
        KuDNN::ActivationFunction::SQRT_DST,
        KuDNN::ActivationFunction::SWISH,
        KuDNN::ActivationFunction::SRELU,
        KuDNN::ActivationFunction::GELU_ERF,
        KuDNN::ActivationFunction::MISH
    };
}

TEST(EltwiseGTests, ForwardFunc4DF32_1)
{
    const KuDNN::TensorInfo srcTensor = {{1, 1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCD};
    const KuDNN::TensorInfo dstTensor = {{1, 1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCD};
    KuDNN::ActivationFunction algKind = KuDNN::ActivationFunction::ABS;
    EXPECT_NO_THROW(ForwardFunc1<float>(srcTensor, dstTensor, algKind));
}

TEST(EltwiseGTests, ForwardFunc4DF32_allKind)
{

    const KuDNN::TensorInfo srcTensor = {{8, 11, 17, 23}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCD};
    const KuDNN::TensorInfo dstTensor = {{8, 11, 17, 23}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCD};

    for(auto algKind : getAllActivationFunctions())
    {
        try {
            ForwardFunc1<float>(srcTensor, dstTensor, algKind);
        } catch (const std::exception& e) {
            std::cout << "algKind: " << static_cast<int>(algKind) << " error: " << e.what() << std::endl;
            continue;
        }
    }
}

TEST(EltwiseGTests, BackwardFunc4DF32_allKind)
{
    const KuDNN::TensorInfo srcTensor = {{8, 11, 17, 23}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCD};
    const KuDNN::TensorInfo diffDstTensor = {{8, 11, 17, 23}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCD};
    const KuDNN::TensorInfo diffSrcTensor = {{8, 11, 17, 23}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCD};

    for(auto algKind : getAllActivationFunctions())
    {
        try {
            BackwardFunc1<float>(diffSrcTensor, diffDstTensor, srcTensor, algKind);
        } catch (const std::exception& e) {
            std::cout << "algKind: " << static_cast<int>(algKind) << " error: " << e.what() << std::endl;
            continue;
        }
    }
}

TEST(EltwiseGTests, ForwardFunc5DF32_allKind)
{

    const KuDNN::TensorInfo srcTensor = {{7, 8, 11, 17, 23}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ACDEB};
    const KuDNN::TensorInfo dstTensor = {{7, 8, 11, 17, 23}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ACDEB};

    for(auto algKind : getAllActivationFunctions())
    {
        try {
            ForwardFunc1<float>(srcTensor, dstTensor, algKind);
        } catch (const std::exception& e) {
            std::cout << "algKind: " << static_cast<int>(algKind) << " error: " << e.what() << std::endl;
            continue;
        }
    }
}

TEST(EltwiseGTests, BackwardFunc5DF32_allKind)
{
    const KuDNN::TensorInfo srcTensor = {{8, 11, 17, 23, 34}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCED};
    const KuDNN::TensorInfo diffDstTensor = {{8, 11, 17, 23, 34}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCED};
    const KuDNN::TensorInfo diffSrcTensor = {{8, 11, 17, 23, 34}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCED};

    for(auto algKind : getAllActivationFunctions())
    {
        try {
            BackwardFunc1<float>(diffSrcTensor, diffDstTensor, srcTensor, algKind);
        } catch (const std::exception& e) {
            std::cout << "algKind: " << static_cast<int>(algKind) << " error: " << e.what() << std::endl;
            continue;
        }
    }
}