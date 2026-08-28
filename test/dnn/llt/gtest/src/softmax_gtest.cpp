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

#include <cstring>
#include <cmath>
#include <ctime>
#include <random>
#include <iostream>

#include "gtest/gtest.h"
#include "kudnn.hpp"
#include "kudnn_tensor_data.hpp"

// forward
static void ForwardCopyConstructor1()
{
    const KuDNN::TensorInfo srcTensor = {{1, 1, 1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCDE};
    const KuDNN::TensorInfo dstTensor = {{1, 1, 1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCDE};
    KuDNN::SizeType axis = 1;
    KuDNN::SoftmaxLayerFWD softmaxLayer1(srcTensor, dstTensor, axis, KuDNN::AlgorithmKind::SOFTMAX);
    KuDNN::SoftmaxLayerFWD softmaxLayer2(softmaxLayer1);
    for (size_t i = 0; i < g_Tensor_data.size() / 2; i += 2) {
        KuDNN::SoftmaxLayerFWD softmaxLayer1(g_Tensor_data[i], g_Tensor_data[i + 1], axis, KuDNN::AlgorithmKind::SOFTMAX);
        KuDNN::SoftmaxLayerFWD softmaxLayer2(softmaxLayer1);
    }
}

static void ForwardCopyConstructor2()
{
    const KuDNN::TensorInfo srcTensor = {{1, 1, 1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCDE};
    const KuDNN::TensorInfo dstTensor = {{1, 1, 1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCDE};
    KuDNN::SizeType axis = 1;
    KuDNN::SoftmaxLayerFWD softmaxLayer1(srcTensor, dstTensor, axis, KuDNN::AlgorithmKind::SOFTMAX);
    KuDNN::SoftmaxLayerFWD softmaxLayer2(softmaxLayer1);
    softmaxLayer2 = softmaxLayer1;
    for (size_t i = 0; i < g_Tensor_data.size() / 2; i += 2) {
        KuDNN::SoftmaxLayerFWD softmaxLayer1(g_Tensor_data[i], g_Tensor_data[i + 1], axis, KuDNN::AlgorithmKind::SOFTMAX);
        KuDNN::SoftmaxLayerFWD softmaxLayer2(softmaxLayer1);
        softmaxLayer2 = softmaxLayer1;
    }
}

static void ForwardMoveConstructor1()
{
    const KuDNN::TensorInfo srcTensor = {{1, 1, 1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCDE};
    const KuDNN::TensorInfo dstTensor = {{1, 1, 1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCDE};
    KuDNN::SizeType axis = 1;
    KuDNN::SoftmaxLayerFWD softmaxLayer1(srcTensor, dstTensor, axis, KuDNN::AlgorithmKind::SOFTMAX);
    KuDNN::SoftmaxLayerFWD softmaxLayer2(std::move(softmaxLayer1));
    for (size_t i = 0; i < g_Tensor_data.size() / 2; i += 2) {
        KuDNN::SoftmaxLayerFWD softmaxLayer1(g_Tensor_data[i], g_Tensor_data[i + 1], axis, KuDNN::AlgorithmKind::SOFTMAX);
        KuDNN::SoftmaxLayerFWD softmaxLayer2(std::move(softmaxLayer1));
    }
}

static void ForwardMoveConstructor2()
{
    const KuDNN::TensorInfo srcTensor = {{1, 1, 1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCDE};
    const KuDNN::TensorInfo dstTensor = {{1, 1, 1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCDE};
    KuDNN::SizeType axis = 1;
    KuDNN::SoftmaxLayerFWD softmaxLayer1(srcTensor, dstTensor, axis, KuDNN::AlgorithmKind::SOFTMAX);
    KuDNN::SoftmaxLayerFWD softmaxLayer2(srcTensor, dstTensor, axis, KuDNN::AlgorithmKind::SOFTMAX);
    softmaxLayer2 = std::move(softmaxLayer1);
    for (size_t i = 0; i < g_Tensor_data.size() / 2; i += 2) {
        KuDNN::SoftmaxLayerFWD softmaxLayer1(g_Tensor_data[i], g_Tensor_data[i + 1], axis, KuDNN::AlgorithmKind::SOFTMAX);
        KuDNN::SoftmaxLayerFWD softmaxLayer2(g_Tensor_data[i], g_Tensor_data[i + 1], axis, KuDNN::AlgorithmKind::SOFTMAX);
        softmaxLayer2 = std::move(softmaxLayer1);
    }
}

// backward
static void BackwardCopyConstructor1()
{
    const KuDNN::TensorInfo dstTensor = {{1, 1, 1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCDE};
    const KuDNN::TensorInfo diffDstTensor = {{1, 1, 1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCDE};
    const KuDNN::TensorInfo diffSrcTensor = {{1, 1, 1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCDE};
    KuDNN::SizeType axis = 1;
    KuDNN::SoftmaxLayerBWD softmaxLayer1(dstTensor, diffDstTensor, diffSrcTensor, axis, KuDNN::AlgorithmKind::SOFTMAX);
    KuDNN::SoftmaxLayerBWD softmaxLayer2(softmaxLayer1);
    for (size_t i = 0; i < g_Tensor_data.size() / 3; i += 3) {
        KuDNN::SoftmaxLayerBWD softmaxLayer1(g_Tensor_data[i], g_Tensor_data[i + 1], g_Tensor_data[i + 2], axis, KuDNN::AlgorithmKind::SOFTMAX);
        KuDNN::SoftmaxLayerBWD softmaxLayer2(softmaxLayer1);
    }
}

static void BackwardCopyConstructor2()
{
    const KuDNN::TensorInfo dstTensor = {{1, 1, 1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCDE};
    const KuDNN::TensorInfo diffDstTensor = {{1, 1, 1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCDE};
    const KuDNN::TensorInfo diffSrcTensor = {{1, 1, 1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCDE};
    KuDNN::SizeType axis = 1;
    KuDNN::SoftmaxLayerBWD softmaxLayer1(dstTensor, diffDstTensor, diffSrcTensor, axis, KuDNN::AlgorithmKind::SOFTMAX);
    KuDNN::SoftmaxLayerBWD softmaxLayer2(softmaxLayer1);
    softmaxLayer2 = softmaxLayer1;
    for (size_t i = 0; i < g_Tensor_data.size() / 3; i += 3) {
        KuDNN::SoftmaxLayerBWD softmaxLayer1(g_Tensor_data[i], g_Tensor_data[i + 1], g_Tensor_data[i + 2], axis, KuDNN::AlgorithmKind::SOFTMAX);
        KuDNN::SoftmaxLayerBWD softmaxLayer2(softmaxLayer1);
        softmaxLayer2 = softmaxLayer1;
    }
}

static void BackwardMoveConstructor1()
{
    const KuDNN::TensorInfo dstTensor = {{1, 1, 1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCDE};
    const KuDNN::TensorInfo diffDstTensor = {{1, 1, 1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCDE};
    const KuDNN::TensorInfo diffSrcTensor = {{1, 1, 1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCDE};
    KuDNN::SizeType axis = 1;
    KuDNN::SoftmaxLayerBWD softmaxLayer1(dstTensor, diffDstTensor, diffSrcTensor, axis, KuDNN::AlgorithmKind::SOFTMAX);
    KuDNN::SoftmaxLayerBWD softmaxLayer2(std::move(softmaxLayer1));
    for (size_t i = 0; i < g_Tensor_data.size() / 3; i += 3) {
        KuDNN::SoftmaxLayerBWD softmaxLayer1(g_Tensor_data[i], g_Tensor_data[i + 1], g_Tensor_data[i + 2], axis, KuDNN::AlgorithmKind::SOFTMAX);
        KuDNN::SoftmaxLayerBWD softmaxLayer2(std::move(softmaxLayer1));
        std::cout << "BackwardMoveConstructor1 : " << i << std::endl;
    }
}

static void BackwardMoveConstructor2()
{
    const KuDNN::TensorInfo dstTensor = {{1, 1, 1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCDE};
    const KuDNN::TensorInfo diffDstTensor = {{1, 1, 1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCDE};
    const KuDNN::TensorInfo diffSrcTensor = {{1, 1, 1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCDE};
    KuDNN::SizeType axis = 1;
    KuDNN::SoftmaxLayerBWD softmaxLayer1(dstTensor, diffDstTensor, diffSrcTensor, axis, KuDNN::AlgorithmKind::SOFTMAX);
    KuDNN::SoftmaxLayerBWD softmaxLayer2(dstTensor, diffDstTensor, diffSrcTensor, axis, KuDNN::AlgorithmKind::SOFTMAX);
    softmaxLayer2 = std::move(softmaxLayer1);
    for (size_t i = 0; i < g_Tensor_data.size() / 3; i += 3) {
        KuDNN::SoftmaxLayerBWD softmaxLayer1(g_Tensor_data[i], g_Tensor_data[i + 1], g_Tensor_data[i + 2], axis, KuDNN::AlgorithmKind::SOFTMAX);
        KuDNN::SoftmaxLayerBWD softmaxLayer2(g_Tensor_data[i], g_Tensor_data[i + 1], g_Tensor_data[i + 2], axis, KuDNN::AlgorithmKind::SOFTMAX);
        softmaxLayer2 = std::move(softmaxLayer1);
    }
}


// test cases
TEST(SoftmaxGTests, FWDCopyConstructor1)
{
    EXPECT_NO_THROW(ForwardCopyConstructor1());
}

TEST(SoftmaxGTests, FWDCopyConstructor2)
{
    EXPECT_NO_THROW(ForwardCopyConstructor2());
}

TEST(SoftmaxGTests, FWDMoveConstructor1)
{
    EXPECT_NO_THROW(ForwardMoveConstructor1());
}

TEST(SoftmaxGTests, FWDMoveConstructor2)
{
    EXPECT_NO_THROW(ForwardMoveConstructor2());
}

TEST(SoftmaxGTests, BWDCopyConstructor1)
{
    EXPECT_NO_THROW(BackwardCopyConstructor1());
}

TEST(SoftmaxGTests, BWDCopyConstructor2)
{
    EXPECT_NO_THROW(BackwardCopyConstructor2());
}

TEST(SoftmaxGTests, BWDMoveConstructor1)
{
    EXPECT_NO_THROW(BackwardMoveConstructor1());
}

TEST(SoftmaxGTests, BWDMoveConstructor2)
{
    EXPECT_NO_THROW(BackwardMoveConstructor2());
}

using Shape = KuDNN::Shape;
using SizeType = KuDNN::SizeType;

static const float maxError = 1e-3;

static void softmaxHelp(float *src, int start, int end)
{
    if (start > end || !src) return;

    float max_val = src[start];
    for (int i = start + 1; i <= end; ++i) {
        if (src[i] > max_val) {
            max_val = src[i];
        }
    }
    float sum_exp = 0.0f;
    for (int i = start; i <= end; ++i) {
        sum_exp += std::exp(src[i] - max_val);
    }
    for (int i = start; i <= end; ++i) {
        src[i] = std::exp(src[i] - max_val) / sum_exp;
    }
}

static void softmaxSimple(Shape shape, float *src)
{
    SizeType numDims = shape.GetNumDims();
    SizeType outSize = 1;
    for (SizeType i = 0; i < numDims - 1; ++i) {
        outSize *= shape[i];
    }
    SizeType innerSize = shape[numDims - 1];

    for (SizeType i = 0; i < outSize; ++i) {
        softmaxHelp(src, static_cast<int>(i * innerSize), static_cast<int>((i + 1) * innerSize - 1));
    }
}

static void dataInit(float **src, float **dst, float **dstRef, SizeType totalSize)
{
    *src = (float *)malloc(totalSize * sizeof(float));
    *dst = (float *)malloc(totalSize * sizeof(float));
    *dstRef = (float *)malloc(totalSize * sizeof(float));

    if (*src == nullptr || *dst == nullptr || *dstRef == nullptr) {
        std::cerr << "Memory allocation failed" << std::endl;
        return;
    }
    std::uniform_real_distribution<float> u(-1, 1);
    std::default_random_engine e(time(NULL));

    for (SizeType i = 0; i < totalSize; ++i) {
        (*src)[i] = u(e);
    }
    memcpy(*dstRef, *src, totalSize * sizeof(float));
}

static void dataFree(float **src, float **dst, float **dstRef)
{
    free(*src);
    free(*dst);
    free(*dstRef);

    *src = nullptr;
    *dst = nullptr;
    *dstRef = nullptr;
}

static bool FWD_1D_FP32(const Shape &shape)
{
    const KuDNN::TensorInfo srcTensor = {shape, KuDNN::Element::TypeT::F32, KuDNN::Layout::A};
    const KuDNN::TensorInfo dstTensor = {shape, KuDNN::Element::TypeT::F32, KuDNN::Layout::A};
    SizeType axis = 0;
    KuDNN::SoftmaxLayerFWD softmaxLayerFwd(srcTensor, dstTensor, axis, KuDNN::AlgorithmKind::SOFTMAX);

    float *src = nullptr, *dst = nullptr, *dstRef = nullptr;
    SizeType totalSize = srcTensor.GetTotalTensorSize();
    dataInit(&src, &dst, &dstRef, totalSize);
    softmaxSimple(shape, dstRef);
    softmaxLayerFwd.Run(src, dst);
    for (SizeType i = 0; i < totalSize; ++i) {
        float error = std::abs(*(dst + i) - *(dstRef + i));
        if (error > maxError) {
            dataFree(&src, &dst, &dstRef);
            return false;
        }
    }
    dataFree(&src, &dst, &dstRef);
    return true;
}

static bool FWD_2D_FP32(const Shape &shape)
{
    const KuDNN::TensorInfo srcTensor = {shape, KuDNN::Element::TypeT::F32, KuDNN::Layout::AB};
    const KuDNN::TensorInfo dstTensor = {shape, KuDNN::Element::TypeT::F32, KuDNN::Layout::AB};
    SizeType axis = 1;
    KuDNN::SoftmaxLayerFWD softmaxLayerFwd(srcTensor, dstTensor, axis, KuDNN::AlgorithmKind::SOFTMAX);

    float *src = nullptr, *dst = nullptr, *dstRef = nullptr;
    SizeType totalSize = srcTensor.GetTotalTensorSize();
    dataInit(&src, &dst, &dstRef, totalSize);
    softmaxSimple(shape, dstRef);
    softmaxLayerFwd.Run(src, dst);
    for (SizeType i = 0; i < totalSize; ++i) {
        float error = std::abs(*(dst + i) - *(dstRef + i));
        if (error > maxError) {
            dataFree(&src, &dst, &dstRef);
            return false;
        }
    }
    dataFree(&src, &dst, &dstRef);
    return true;
}

static bool FWD_3D_FP32(const Shape &shape)
{
    const KuDNN::TensorInfo srcTensor = {shape, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABC};
    const KuDNN::TensorInfo dstTensor = {shape, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABC};
    SizeType axis = 2;
    KuDNN::SoftmaxLayerFWD softmaxLayerFwd(srcTensor, dstTensor, axis, KuDNN::AlgorithmKind::SOFTMAX);

    float *src = nullptr, *dst = nullptr, *dstRef = nullptr;
    SizeType totalSize = srcTensor.GetTotalTensorSize();
    dataInit(&src, &dst, &dstRef, totalSize);
    softmaxSimple(shape, dstRef);
    softmaxLayerFwd.Run(src, dst);
    for (SizeType i = 0; i < totalSize; ++i) {
        float error = std::abs(*(dst + i) - *(dstRef + i));
        if (error > maxError) {
            dataFree(&src, &dst, &dstRef);
            return false;
        }
    }
    dataFree(&src, &dst, &dstRef);
    return true;
}

static bool FWD_4D_FP32(const Shape &shape)
{
    const KuDNN::TensorInfo srcTensor = {shape, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCD};
    const KuDNN::TensorInfo dstTensor = {shape, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCD};
    SizeType axis = 3;
    KuDNN::SoftmaxLayerFWD softmaxLayerFwd(srcTensor, dstTensor, axis, KuDNN::AlgorithmKind::SOFTMAX);

    float *src = nullptr, *dst = nullptr, *dstRef = nullptr;
    SizeType totalSize = srcTensor.GetTotalTensorSize();
    dataInit(&src, &dst, &dstRef, totalSize);
    softmaxSimple(shape, dstRef);
    softmaxLayerFwd.Run(src, dst);
    for (SizeType i = 0; i < totalSize; ++i) {
        float error = std::abs(*(dst + i) - *(dstRef + i));
        if (error > maxError) {
            dataFree(&src, &dst, &dstRef);
            return false;
        }
    }
    dataFree(&src, &dst, &dstRef);
    return true;
}

static bool FWD_5D_FP32(const Shape &shape)
{
    const KuDNN::TensorInfo srcTensor = {shape, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCDE};
    const KuDNN::TensorInfo dstTensor = {shape, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCDE};
    SizeType axis = 4;
    KuDNN::SoftmaxLayerFWD softmaxLayerFwd(srcTensor, dstTensor, axis, KuDNN::AlgorithmKind::SOFTMAX);

    float *src = nullptr, *dst = nullptr, *dstRef = nullptr;
    SizeType totalSize = srcTensor.GetTotalTensorSize();
    dataInit(&src, &dst, &dstRef, totalSize);
    softmaxSimple(shape, dstRef);
    softmaxLayerFwd.Run(src, dst);
    for (SizeType i = 0; i < totalSize; ++i) {
        float error = std::abs(*(dst + i) - *(dstRef + i));
        if (error > maxError) {
            dataFree(&src, &dst, &dstRef);
            return false;
        }
    }
    dataFree(&src, &dst, &dstRef);
    return true;
}

static bool FWD_5D_FP32_FP16(const Shape &shape)
{
    const KuDNN::TensorInfo srcTensor = {shape, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCDE};
    const KuDNN::TensorInfo dstTensor = {shape, KuDNN::Element::TypeT::F16, KuDNN::Layout::ABCED};
    SizeType axis = 2;
    KuDNN::SoftmaxLayerFWD softmaxLayerFwd(srcTensor, dstTensor, axis, KuDNN::AlgorithmKind::SOFTMAX);

    float *src = nullptr, *dst = nullptr, *dstRef = nullptr;
    SizeType totalSize = srcTensor.GetTotalTensorSize();
    dataInit(&src, &dst, &dstRef, totalSize);
    softmaxLayerFwd.Run(src, dst);
    dataFree(&src, &dst, &dstRef);
    return true;
}

static void BWD_dataInit(float **dst, float **dstDiff, float **srcDiff, float **dstRef, SizeType totalSize)
{
    *dstDiff = (float *)malloc(totalSize * sizeof(float));
    *dst = (float *)malloc(totalSize * sizeof(float));
    *srcDiff = (float *)malloc(totalSize * sizeof(float));
    *dstRef = (float *)malloc(totalSize * sizeof(float));

    if (*dst == nullptr || *dstDiff == nullptr || *srcDiff == nullptr || *dstRef == nullptr) {
        std::cerr << "Memory allocation failed" << std::endl;
        return;
    }
    std::uniform_real_distribution<float> u(-1, 1);
    std::default_random_engine e(time(NULL));

    for (SizeType i = 0; i < totalSize; ++i) {
        (*dst)[i] = u(e);
    }
    memcpy(*dstRef, *dst, totalSize * sizeof(float));
}

static bool BWD_5D_FP32(const Shape &shape)
{
    const KuDNN::TensorInfo srcDiffTensor = {shape, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCDE};
    const KuDNN::TensorInfo dstTensor = {shape, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCDE};
    const KuDNN::TensorInfo dstDiffTensor = {shape, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCDE};
    SizeType axis = 4;
    KuDNN::SoftmaxLayerBWD SoftmaxLayerBWD(dstTensor, dstDiffTensor, srcDiffTensor, axis, KuDNN::AlgorithmKind::SOFTMAX);

    float *srcDiff = nullptr, *dst = nullptr, *dstDiff = nullptr, *dstRef = nullptr;
    SizeType totalSize = dstTensor.GetTotalTensorSize();
    BWD_dataInit(&dst, &dstDiff, &srcDiff, &dstRef, totalSize);
    SoftmaxLayerBWD.Run(dst, dstDiff, srcDiff);
    dataFree(&dst, &dstDiff, &srcDiff);
    free(dstRef);
    return true;
}

static bool BWD_5D_FP32_FP16(const Shape &shape)
{
    const KuDNN::TensorInfo srcDiffTensor = {shape, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCDE};
    const KuDNN::TensorInfo dstTensor = {shape, KuDNN::Element::TypeT::F16, KuDNN::Layout::ABCDE};
    const KuDNN::TensorInfo dstDiffTensor = {shape, KuDNN::Element::TypeT::F16, KuDNN::Layout::ABCDE};
    SizeType axis = 1;
    KuDNN::SoftmaxLayerBWD SoftmaxLayerBWD(dstTensor, dstDiffTensor, srcDiffTensor, axis, KuDNN::AlgorithmKind::SOFTMAX);

    float *srcDiff = nullptr, *dst = nullptr, *dstDiff = nullptr, *dstRef = nullptr;
    SizeType totalSize = dstTensor.GetTotalTensorSize();
    BWD_dataInit(&dst, &dstDiff, &srcDiff, &dstRef, totalSize);
    SoftmaxLayerBWD.Run(dst, dstDiff, srcDiff);
    dataFree(&dst, &dstDiff, &srcDiff);
    free(dstRef);
    return true;
}

static void InputParamExceptTest1()
{
    Shape srcShape(-2, 2, 2, 2, 2);
    const KuDNN::TensorInfo srcTensor = {srcShape, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCDE};
    const KuDNN::TensorInfo dstTensor = {{2, 2, 2, 2, 2}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCDE};
    SizeType axis = 4;
    KuDNN::SoftmaxLayerFWD softmaxLayerFwd(srcTensor, dstTensor, axis, KuDNN::AlgorithmKind::SOFTMAX);
}

static void InputParamExceptTest2()
{
    Shape srcShape(2, 2, 2, 2, 2);
    const KuDNN::TensorInfo srcTensor = {srcShape, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCDE};
    const KuDNN::TensorInfo dstTensor = {{1, 2, 2, 2, 2}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCDE};
    SizeType axis = 4;
    KuDNN::SoftmaxLayerFWD softmaxLayerFwd(srcTensor, dstTensor, axis, KuDNN::AlgorithmKind::SOFTMAX);
}

static void InputParamExceptTest3()
{
    Shape srcShape(2, 2, 2, 2, 2);
    const KuDNN::TensorInfo srcTensor = {srcShape, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCD};
    const KuDNN::TensorInfo dstTensor = {{2, 2, 2, 2, 2}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCDE};
    SizeType axis = 4;
    KuDNN::SoftmaxLayerFWD softmaxLayerFwd(srcTensor, dstTensor, axis, KuDNN::AlgorithmKind::SOFTMAX);
}

static void InputParamExceptTest4()
{
    Shape srcShape(2, 2, 2, 2, 2);
    const KuDNN::TensorInfo srcTensor = {srcShape, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCDE};
    const KuDNN::TensorInfo dstTensor = {{2, 2, 2, 2, 2}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCDE};
    SizeType axis = 4;
    float *src = nullptr, *dst = nullptr;
    KuDNN::SoftmaxLayerFWD softmaxLayerFwd(srcTensor, dstTensor, axis, KuDNN::AlgorithmKind::SOFTMAX);
    softmaxLayerFwd.Run(src, dst);
}

static void InputParamExceptTest5()
{
    Shape srcShape(2, -2, 2, 2, 2);
    const KuDNN::TensorInfo srcTensor = {srcShape, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCDE};
    const KuDNN::TensorInfo dstTensor = {{2, 2, 2, 2, 2}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCDE};
    SizeType axis = 4;
    KuDNN::SoftmaxLayerFWD softmaxLayerFwd(srcTensor, dstTensor, axis, KuDNN::AlgorithmKind::SOFTMAX);
}

static void InputParamExceptTest6()
{
    Shape srcShape(2, 2, -2, 2, 2);
    const KuDNN::TensorInfo srcTensor = {srcShape, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCDE};
    const KuDNN::TensorInfo dstTensor = {{2, 2, 2, 2, 2}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCDE};
    SizeType axis = 4;
    KuDNN::SoftmaxLayerFWD softmaxLayerFwd(srcTensor, dstTensor, axis, KuDNN::AlgorithmKind::SOFTMAX);
}

static void InputParamExceptTest7()
{
    Shape srcShape(2, 2, 2, -2, 2);
    const KuDNN::TensorInfo srcTensor = {srcShape, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCDE};
    const KuDNN::TensorInfo dstTensor = {{2, 2, 2, 2, 2}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCDE};
    SizeType axis = 4;
    KuDNN::SoftmaxLayerFWD softmaxLayerFwd(srcTensor, dstTensor, axis, KuDNN::AlgorithmKind::SOFTMAX);
}

static void InputParamExceptTest8()
{
    Shape dstShape(2, 2, 2, 2, -2);
    const KuDNN::TensorInfo srcTensor = {{2, 2, 2, 2, 2}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCDE};
    const KuDNN::TensorInfo dstTensor = {dstShape, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCDE};
    SizeType axis = 4;
    KuDNN::SoftmaxLayerFWD softmaxLayerFwd(srcTensor, dstTensor, axis, KuDNN::AlgorithmKind::SOFTMAX);
}

static void InputParamExceptTest9()
{
    Shape dstShape(2, 2, 2);
    const KuDNN::TensorInfo srcTensor = {{2, 2, 2}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABC};
    const KuDNN::TensorInfo dstTensor = {dstShape, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABC};
    SizeType axis = -1;
    KuDNN::SoftmaxLayerFWD softmaxLayerFwd(srcTensor, dstTensor, axis, KuDNN::AlgorithmKind::SOFTMAX);
}

static void InputParamExceptTest10()
{
    Shape dstShape(2, 2, 2);
    const KuDNN::TensorInfo srcTensor = {{2, 2, 2}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABC};
    const KuDNN::TensorInfo dstTensor = {dstShape, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABC};
    SizeType axis = 0;
    KuDNN::SoftmaxLayerFWD softmaxLayerFwd(srcTensor, dstTensor, axis, KuDNN::AlgorithmKind::SOFTMAX);
}

static void InputParamExceptTest11()
{
    Shape dstShape(2, 2, 2);
    const KuDNN::TensorInfo srcTensor = {{2, 2, 2}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABC};
    const KuDNN::TensorInfo dstTensor = {dstShape, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABC};
    SizeType axis = 3;
    KuDNN::SoftmaxLayerFWD softmaxLayerFwd(srcTensor, dstTensor, axis, KuDNN::AlgorithmKind::SOFTMAX);
}

static void InputParamExceptTest12()
{
    Shape dstShape(2, 2, 2);
    const KuDNN::TensorInfo srcTensor = {{2, 2, 2}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABC};
    const KuDNN::TensorInfo dstTensor = {dstShape, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABC};
    SizeType axis = 4;
    KuDNN::SoftmaxLayerFWD softmaxLayerFwd(srcTensor, dstTensor, axis, KuDNN::AlgorithmKind::SOFTMAX);
}

// test
TEST(SoftmaxGTests, Param_ShapeExcept1_Test)
{
    EXPECT_THROW(InputParamExceptTest1(), KuDNN::Service::LogicError);
}

TEST(SoftmaxGTests, Param_ShapeExcept2_Test)
{
    EXPECT_THROW(InputParamExceptTest2(), KuDNN::Service::LogicError);
}

TEST(SoftmaxGTests, Param_Layout_Except_Test)
{
    EXPECT_THROW(InputParamExceptTest3(), KuDNN::Service::LogicError);
}

TEST(SoftmaxGTests, Param_NullPtr_Except_Test)
{
    EXPECT_THROW(InputParamExceptTest4(), KuDNN::Service::LogicError);
}

TEST(SoftmaxGTests, Param_ShapeExceptD2_Test)
{
    EXPECT_THROW(InputParamExceptTest5(), KuDNN::Service::LogicError);
}

TEST(SoftmaxGTests, Param_ShapeExceptD3_Test)
{
    EXPECT_THROW(InputParamExceptTest6(), KuDNN::Service::LogicError);
}

TEST(SoftmaxGTests, Param_ShapeExceptM_Test)
{
    EXPECT_THROW(InputParamExceptTest7(), KuDNN::Service::LogicError);
}

TEST(SoftmaxGTests, Param_ShapeExceptN_Test)
{
    EXPECT_THROW(InputParamExceptTest8(), KuDNN::Service::LogicError);
}

TEST(SoftmaxGTests, Param_Axis_Except_NegativeOne_Test)
{
    EXPECT_THROW(InputParamExceptTest9(), KuDNN::Service::LogicError);
}

TEST(SoftmaxGTests, Param_Axis_Except_Zero_Test)
{
    EXPECT_NO_THROW(InputParamExceptTest10());
}

TEST(SoftmaxGTests, Param_Axis_Except_Three_Test)
{
    EXPECT_THROW(InputParamExceptTest11(), KuDNN::Service::LogicError);
}

TEST(SoftmaxGTests, Param_Axis_Except_Four_Test)
{
    EXPECT_THROW(InputParamExceptTest12(), KuDNN::Service::LogicError);
}

TEST(SoftmaxGTests, FWD_1D_FP32)
{
    Shape shape(100);
    EXPECT_TRUE(FWD_1D_FP32(shape));
}

TEST(SoftmaxGTests, FWD_2D_FP32)
{
    Shape shape(40, 50);
    EXPECT_TRUE(FWD_2D_FP32(shape));
}

TEST(SoftmaxGTests, FWD_3D_FP32)
{
    Shape shape(20, 40, 50);
    EXPECT_TRUE(FWD_3D_FP32(shape));
}

TEST(SoftmaxGTests, FWD_4D_FP32)
{
    Shape shape(10, 5, 8, 5);
    EXPECT_TRUE(FWD_4D_FP32(shape));
}

TEST(SoftmaxGTests, FWD_5D_FP32)
{
    Shape shape(23, 17, 8, 37, 13);
    EXPECT_TRUE(FWD_5D_FP32(shape));
    EXPECT_TRUE(FWD_5D_FP32_FP16(shape));
}

TEST(SoftmaxGTests, BWD_5D_FP32)
{
    Shape shape(11, 5, 4, 5, 5);
    EXPECT_TRUE(BWD_5D_FP32(shape));

    Shape shape_2(23, 17, 8, 37, 13);
    EXPECT_TRUE(BWD_5D_FP32_FP16(shape_2));
}