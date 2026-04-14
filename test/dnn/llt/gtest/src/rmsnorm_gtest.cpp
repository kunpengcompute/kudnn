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
#include <iostream>
#include <cmath>
#include <ctime>
#include <random>
#include <arm_neon.h>

#include "gtest/gtest.h"
#include "kudnn.hpp"
#include "kudnn_tensor_data.hpp"

using SizeType = KuDNN::SizeType;
using Shape = KuDNN::Shape;
using TensorInfo = const KuDNN::TensorInfo;
extern KuDNN::Element::TypeT TypeF16;
const int MAX_THS = 100;
const float FACTOR_THS = 0.02;
float errBoundRMSn = 1e-3;

template <typename T>
static void rmsnormRef(SizeType outerSize, SizeType innerSize, const T *src, T *dst, const T *scale, bool scaleApply,
    float *variance, const float eps, bool global_stats = false)
{
    int threadsUsed = (int)ceil((float)outerSize * FACTOR_THS);
    threadsUsed = threadsUsed > MAX_THS ? MAX_THS : threadsUsed;
    // variance compute
    if (!global_stats) {
#pragma omp parallel for num_threads(threadsUsed) schedule(static)
        for (SizeType i = 0; i < outerSize; ++i) {
            float sum = 0.0f;
            for (SizeType j = 0; j < innerSize; ++j) {
                float srcVal = (float)*(src + i * innerSize + j);
                sum += srcVal * srcVal;
            }
            *(variance + i) = sum / innerSize;
        }
    }
    // norm
#pragma omp parallel for collapse(2) num_threads(threadsUsed) schedule(static)
    for (SizeType i = 0; i < outerSize; ++i) {
        float varVal = *(variance + i);
        float invStd = 1.0f / std::sqrt(varVal + eps);
        for (SizeType j = 0; j < innerSize; ++j) {
            float srcVal = (float)*(src + i * innerSize + j);
            float normalized = srcVal * invStd;
            if (scaleApply) {
                normalized *= scale[j];
            }
            *(dst + i * innerSize + j) = static_cast<T>(normalized);
        }
    }
}
// forward
template <typename T>
static bool ForwardFunc1(const TensorInfo &srcInfo, const TensorInfo &statInfo, const TensorInfo &scaleInfo,
    const TensorInfo &dstInfo, KuDNN::NormalizationFlags flags)
{
    KuDNN::RMSNormalizationLayerFWD rmsLayer1(srcInfo, statInfo, scaleInfo, dstInfo, flags);

    SizeType srcSize = srcInfo.GetTotalTensorSize();
    SizeType dstSize = dstInfo.GetTotalTensorSize();
    SizeType statSize = statInfo.GetTotalTensorSize();
    SizeType innerSize = scaleInfo.GetTotalTensorSize();

    T *src = (T *)malloc(srcSize * sizeof(T));
    T *dst = (T *)malloc(dstSize * sizeof(T));
    T *dstRef = (T *)malloc(dstSize * sizeof(T));
    float *variance = (float *)malloc(statSize * sizeof(float));
    T *scale = (T *)malloc(innerSize * sizeof(T));
    float eps = 1e-5;
    if (src == nullptr || dst == nullptr || dstRef == nullptr || variance == nullptr || scale == nullptr) {
        std::cerr << "Memory allocation failed" << std::endl;
        return false;
    }

    bool global_stats = static_cast<bool>(flags & KuDNN::NormalizationFlags::USE_GLOBAL_STATS);

    // generate random test data
    std::uniform_real_distribution<float> u(-1, 1);
    std::default_random_engine e(time(NULL));
    int threadsUsed = (int)ceil((float)srcSize * FACTOR_THS);
    threadsUsed = threadsUsed > MAX_THS ? MAX_THS : threadsUsed;
#pragma omp parallel for num_threads(threadsUsed) schedule(static)
    for (SizeType i = 0; i < srcSize; ++i) {
        *(src + i) = (T)u(e);
    }
    for (SizeType j = 0; j < innerSize; ++j) {
        *(scale + j) = (T)u(e);
    }
    if (global_stats) {
        for (SizeType i = 0; i < statSize; i++) {
            *(variance + i) = 0.1f;
        }
    }

    float error = 0.0;
    rmsLayer1.Run(src, dst, scale, variance, true, eps);
    bool scaleApply = static_cast<bool>(flags & KuDNN::NormalizationFlags::USE_SCALE);
    rmsnormRef<T>(statSize, innerSize, src, dstRef, scale, scaleApply, variance, eps, global_stats);
#pragma omp parallel for reduction(+ : error) num_threads(threadsUsed)
    for (SizeType i = 0; i < dstSize; ++i) {
        error += *(dst + i) - *(dstRef + i);
    }
    free(src);
    free(dst);
    free(dstRef);
    free(variance);
    free(scale);
    error = std::abs(error) / dstSize;
    std::cout << "run error: " << error << std::endl;
    return error < errBoundRMSn;
}

template <typename T>
static bool FWDSaveStats(const TensorInfo &srcInfo, const TensorInfo &statInfo, const TensorInfo &scaleInfo,
    const TensorInfo &dstInfo, KuDNN::NormalizationFlags flags)
{
    flags = KuDNN::NormalizationFlags::NONE;
    KuDNN::RMSNormalizationLayerFWD rmsLayer1(srcInfo, statInfo, scaleInfo, dstInfo, flags);
    SizeType srcSize = srcInfo.GetTotalTensorSize();
    SizeType dstSize = dstInfo.GetTotalTensorSize();
    SizeType statSize = statInfo.GetTotalTensorSize();
    SizeType innerSize = scaleInfo.GetTotalTensorSize();
    T *src = (T *)malloc(srcSize * sizeof(T));
    T *dst = (T *)malloc(dstSize * sizeof(T));
    float *variance = (float *)malloc(statSize * sizeof(float));
    float *varianceRef = (float *)malloc(statSize * sizeof(float));
    T *scale = nullptr;
    float eps = 1e-5;
    if (src == nullptr || dst == nullptr || variance == nullptr || varianceRef == nullptr) {
        std::cerr << "Memory allocation failed" << std::endl;
        return false;
    }
    // generate random test data
    std::uniform_real_distribution<float> u(-1, 1);
    std::default_random_engine e(time(NULL));
    int threadsUsed = (int)ceil((float)srcSize * FACTOR_THS);
    threadsUsed = threadsUsed > MAX_THS ? MAX_THS : threadsUsed;
#pragma omp parallel for num_threads(threadsUsed) schedule(static)
    for (SizeType i = 0; i < srcSize; ++i) {
        *(src + i) = (T)u(e);
    }

    float error = 0.0f;
    rmsLayer1.Run(src, dst, scale, variance, true, eps);
    bool scaleApply = false;
    rmsnormRef<T>(statSize, innerSize, src, dst, scale, scaleApply, varianceRef, eps);
    free(src);
    free(dst);
    free(variance);
    free(varianceRef);
    error = std::abs(error) / statSize;
    std::cout << "run error: " << error << std::endl;
    return error < errBoundRMSn;
}

static void FWDSrcDstType()
{
    const KuDNN::TensorInfo srcInfo = {{1, 1, 1}, KuDNN::Element::TypeT::S32, KuDNN::Layout::ABC};
    const KuDNN::TensorInfo statInfo = {{1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::AB};
    const KuDNN::TensorInfo scaleInfo = {{1}, KuDNN::Element::TypeT::F16, KuDNN::Layout::A};
    const KuDNN::TensorInfo dstInfo = {{1, 1, 1}, KuDNN::Element::TypeT::S32, KuDNN::Layout::ABC};
    KuDNN::RMSNormalizationLayerFWD rmsLayer1(srcInfo, statInfo, scaleInfo, dstInfo, KuDNN::NormalizationFlags::NONE);
}

static void FWDScaleType()
{
    const KuDNN::TensorInfo srcInfo = {{1, 1, 1}, KuDNN::Element::TypeT::F16, KuDNN::Layout::ABC};
    const KuDNN::TensorInfo statInfo = {{1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::AB};
    const KuDNN::TensorInfo scaleInfo = {{1}, KuDNN::Element::TypeT::S32, KuDNN::Layout::A};
    const KuDNN::TensorInfo dstInfo = {{1, 1, 1}, KuDNN::Element::TypeT::F16, KuDNN::Layout::ABC};
    KuDNN::RMSNormalizationLayerFWD rmsLayer1(
        srcInfo, statInfo, scaleInfo, dstInfo, KuDNN::NormalizationFlags::USE_SCALE);
}

static void FWDShiftType()
{
    const KuDNN::TensorInfo srcInfo = {{1, 1, 1}, KuDNN::Element::TypeT::F16, KuDNN::Layout::ABC};
    const KuDNN::TensorInfo statInfo = {{1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::AB};
    const KuDNN::TensorInfo scaleInfo = {{1}, KuDNN::Element::TypeT::S32, KuDNN::Layout::A};
    const KuDNN::TensorInfo dstInfo = {{1, 1, 1}, KuDNN::Element::TypeT::F16, KuDNN::Layout::ABC};
    KuDNN::RMSNormalizationLayerFWD rmsLayer1(
        srcInfo, statInfo, scaleInfo, dstInfo, KuDNN::NormalizationFlags::USE_SHIFT);
}

static void FWDGlobalStatsType()
{
    const KuDNN::TensorInfo srcInfo = {{1, 1, 1}, KuDNN::Element::TypeT::F16, KuDNN::Layout::ABC};
    const KuDNN::TensorInfo statInfo = {{1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::AB};
    const KuDNN::TensorInfo scaleInfo = {{1}, KuDNN::Element::TypeT::S32, KuDNN::Layout::A};
    const KuDNN::TensorInfo dstInfo = {{1, 1, 1}, KuDNN::Element::TypeT::F16, KuDNN::Layout::ABC};
    KuDNN::RMSNormalizationLayerFWD rmsLayer1(
        srcInfo, statInfo, scaleInfo, dstInfo, KuDNN::NormalizationFlags::USE_GLOBAL_STATS);
}

static void ForwardFlagType()
{
    const KuDNN::TensorInfo srcInfo = {{1, 1, 1}, KuDNN::Element::TypeT::F16, KuDNN::Layout::ABC};
    const KuDNN::TensorInfo statInfo = {{1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::AB};
    const KuDNN::TensorInfo scaleInfo = {{1}, KuDNN::Element::TypeT::S32, KuDNN::Layout::A};
    const KuDNN::TensorInfo dstInfo = {{1, 1, 1}, KuDNN::Element::TypeT::F16, KuDNN::Layout::ABC};
    KuDNN::RMSNormalizationLayerFWD rmsLayer1(
        srcInfo, statInfo, scaleInfo, dstInfo, KuDNN::NormalizationFlags::FUSE_NORM_RELU);
}

static void ForwardCopyConstructor1()
{
    const KuDNN::TensorInfo srcInfo = {{1, 1, 1}, KuDNN::Element::TypeT::F16, KuDNN::Layout::ABC};
    const KuDNN::TensorInfo statInfo = {{1, 1}, KuDNN::Element::TypeT::F16, KuDNN::Layout::AB};
    const KuDNN::TensorInfo scaleInfo = {{1}, KuDNN::Element::TypeT::F16, KuDNN::Layout::A};
    const KuDNN::TensorInfo dstInfo = {{1, 1, 1}, KuDNN::Element::TypeT::F16, KuDNN::Layout::ABC};
    KuDNN::RMSNormalizationLayerFWD rmsLayer1(srcInfo, statInfo, scaleInfo, dstInfo, KuDNN::NormalizationFlags::NONE);
    KuDNN::RMSNormalizationLayerFWD rmsLayer2(rmsLayer1);
    for (size_t i = 0; i < g_Tensor_data.size() / 4; i += 4) {
        KuDNN::RMSNormalizationLayerFWD rmsLayer1(g_Tensor_data[i],
            g_Tensor_data[i + 1],
            g_Tensor_data[i + 2],
            g_Tensor_data[i + 3],
            KuDNN::NormalizationFlags::NONE);
        KuDNN::RMSNormalizationLayerFWD rmsLayer2(rmsLayer1);
    }
}

static void ForwardCopyConstructor2()
{
    const KuDNN::TensorInfo srcInfo = {{1, 1, 1}, KuDNN::Element::TypeT::F16, KuDNN::Layout::ABC};
    const KuDNN::TensorInfo statInfo = {{1, 1}, KuDNN::Element::TypeT::F16, KuDNN::Layout::AB};
    const KuDNN::TensorInfo scaleInfo = {{1}, KuDNN::Element::TypeT::F16, KuDNN::Layout::A};
    const KuDNN::TensorInfo dstInfo = {{1, 1, 1}, KuDNN::Element::TypeT::F16, KuDNN::Layout::ABC};
    KuDNN::RMSNormalizationLayerFWD rmsLayer1(srcInfo, statInfo, scaleInfo, dstInfo, KuDNN::NormalizationFlags::NONE);
    KuDNN::RMSNormalizationLayerFWD rmsLayer2(srcInfo, statInfo, scaleInfo, dstInfo, KuDNN::NormalizationFlags::NONE);
    rmsLayer2 = rmsLayer1;
    for (size_t i = 0; i < g_Tensor_data.size() / 4; i += 4) {
        KuDNN::RMSNormalizationLayerFWD rmsLayer1(g_Tensor_data[i],
            g_Tensor_data[i + 1],
            g_Tensor_data[i + 2],
            g_Tensor_data[i + 3],
            KuDNN::NormalizationFlags::NONE);
        KuDNN::RMSNormalizationLayerFWD rmsLayer2(g_Tensor_data[i],
            g_Tensor_data[i + 1],
            g_Tensor_data[i + 2],
            g_Tensor_data[i + 3],
            KuDNN::NormalizationFlags::NONE);
        rmsLayer2 = rmsLayer1;
    }
}

static void ForwardMoveConstructor1()
{
    const KuDNN::TensorInfo srcInfo = {{1, 1, 1}, KuDNN::Element::TypeT::F16, KuDNN::Layout::ABC};
    const KuDNN::TensorInfo statInfo = {{1, 1}, KuDNN::Element::TypeT::F16, KuDNN::Layout::AB};
    const KuDNN::TensorInfo scaleInfo = {{1}, KuDNN::Element::TypeT::F16, KuDNN::Layout::A};
    const KuDNN::TensorInfo dstInfo = {{1, 1, 1}, KuDNN::Element::TypeT::F16, KuDNN::Layout::ABC};
    KuDNN::RMSNormalizationLayerFWD rmsLayer1(srcInfo, statInfo, scaleInfo, dstInfo, KuDNN::NormalizationFlags::NONE);
    KuDNN::RMSNormalizationLayerFWD rmsLayer2(std::move(rmsLayer1));
    for (size_t i = 0; i < g_Tensor_data.size() / 4; i += 4) {
        KuDNN::RMSNormalizationLayerFWD rmsLayer1(g_Tensor_data[i],
            g_Tensor_data[i + 1],
            g_Tensor_data[i + 2],
            g_Tensor_data[i + 3],
            KuDNN::NormalizationFlags::NONE);
        KuDNN::RMSNormalizationLayerFWD rmsLayer2(std::move(rmsLayer1));
    }
}

static void ForwardMoveConstructor2()
{
    const KuDNN::TensorInfo srcInfo = {{1, 1, 1}, KuDNN::Element::TypeT::F16, KuDNN::Layout::ABC};
    const KuDNN::TensorInfo statInfo = {{1, 1}, KuDNN::Element::TypeT::F16, KuDNN::Layout::AB};
    const KuDNN::TensorInfo scaleInfo = {{1}, KuDNN::Element::TypeT::F16, KuDNN::Layout::A};
    const KuDNN::TensorInfo dstInfo = {{1, 1, 1}, KuDNN::Element::TypeT::F16, KuDNN::Layout::ABC};
    KuDNN::RMSNormalizationLayerFWD rmsLayer1(srcInfo, statInfo, scaleInfo, dstInfo, KuDNN::NormalizationFlags::NONE);
    KuDNN::RMSNormalizationLayerFWD rmsLayer2(srcInfo, statInfo, scaleInfo, dstInfo, KuDNN::NormalizationFlags::NONE);
    rmsLayer2 = std::move(rmsLayer1);
    for (size_t i = 0; i < g_Tensor_data.size() / 4; i += 4) {
        KuDNN::RMSNormalizationLayerFWD rmsLayer1(g_Tensor_data[i],
            g_Tensor_data[i + 1],
            g_Tensor_data[i + 2],
            g_Tensor_data[i + 3],
            KuDNN::NormalizationFlags::NONE);
        KuDNN::RMSNormalizationLayerFWD rmsLayer2(g_Tensor_data[i],
            g_Tensor_data[i + 1],
            g_Tensor_data[i + 2],
            g_Tensor_data[i + 3],
            KuDNN::NormalizationFlags::NONE);
        rmsLayer2 = std::move(rmsLayer1);
    }
}

static void InputParamExcept1()
{
    Shape shape(-5, 5, 5, 5, 15);
    TensorInfo srcInfo = {shape, TypeF16, KuDNN::Layout::ABCDE};
    TensorInfo statInfo = {{shape[0], shape[1], shape[2], shape[3]}, TypeF16, KuDNN::Layout::ABCD};
    TensorInfo scaleInfo = {{shape[4]}, TypeF16, KuDNN::Layout::A};
    TensorInfo dstInfo = {shape, TypeF16, KuDNN::Layout::ABCDE};
    KuDNN::NormalizationFlags flags = KuDNN::NormalizationFlags::NONE;
    GTEST_EXPECT_TRUE(ForwardFunc1<__fp16>(srcInfo, statInfo, scaleInfo, dstInfo, flags));
}

static void InputParamExcept2()
{
    Shape shape(5, -5, 5, 5, 15);
    TensorInfo srcInfo = {shape, TypeF16, KuDNN::Layout::ABCDE};
    TensorInfo statInfo = {{shape[0], shape[1], shape[2], shape[3]}, TypeF16, KuDNN::Layout::ABCD};
    TensorInfo scaleInfo = {{shape[4]}, TypeF16, KuDNN::Layout::A};
    TensorInfo dstInfo = {shape, TypeF16, KuDNN::Layout::ABCDE};
    KuDNN::NormalizationFlags flags = KuDNN::NormalizationFlags::NONE;
    GTEST_EXPECT_TRUE(ForwardFunc1<__fp16>(srcInfo, statInfo, scaleInfo, dstInfo, flags));
}

static void InputParamExcept3()
{
    Shape shape(5, 5, -5, 5, 15);
    TensorInfo srcInfo = {shape, TypeF16, KuDNN::Layout::ABCDE};
    TensorInfo statInfo = {{shape[0], shape[1], shape[2], shape[3]}, TypeF16, KuDNN::Layout::ABCD};
    TensorInfo scaleInfo = {{shape[4]}, TypeF16, KuDNN::Layout::A};
    TensorInfo dstInfo = {shape, TypeF16, KuDNN::Layout::ABCDE};
    KuDNN::NormalizationFlags flags = KuDNN::NormalizationFlags::NONE;
    GTEST_EXPECT_TRUE(ForwardFunc1<__fp16>(srcInfo, statInfo, scaleInfo, dstInfo, flags));
}

static void InputParamExcept4()
{
    Shape shape(5, 5, 5, -5, 15);
    TensorInfo srcInfo = {shape, TypeF16, KuDNN::Layout::ABCDE};
    TensorInfo statInfo = {{shape[0], shape[1], shape[2], shape[3]}, TypeF16, KuDNN::Layout::ABCD};
    TensorInfo scaleInfo = {{shape[4]}, TypeF16, KuDNN::Layout::A};
    TensorInfo dstInfo = {shape, TypeF16, KuDNN::Layout::ABCDE};
    KuDNN::NormalizationFlags flags = KuDNN::NormalizationFlags::NONE;
    GTEST_EXPECT_TRUE(ForwardFunc1<__fp16>(srcInfo, statInfo, scaleInfo, dstInfo, flags));
}

static void InputParamExcept5()
{
    Shape shape(5, 5, 5, 5, -15);
    TensorInfo srcInfo = {shape, TypeF16, KuDNN::Layout::ABCDE};
    TensorInfo statInfo = {{shape[0], shape[1], shape[2], shape[3]}, TypeF16, KuDNN::Layout::ABCD};
    TensorInfo scaleInfo = {{shape[4]}, TypeF16, KuDNN::Layout::A};
    TensorInfo dstInfo = {shape, TypeF16, KuDNN::Layout::ABCDE};
    KuDNN::NormalizationFlags flags = KuDNN::NormalizationFlags::NONE;
    GTEST_EXPECT_TRUE(ForwardFunc1<__fp16>(srcInfo, statInfo, scaleInfo, dstInfo, flags));
}
// test cases
TEST(RMSnormGTests, ForwardFunc2DF16_1)
{
    Shape shape(100, 100);
    TensorInfo srcInfo = {shape, TypeF16, KuDNN::Layout::AB};
    TensorInfo statInfo = {{shape[0]}, TypeF16, KuDNN::Layout::A};
    TensorInfo scaleInfo = {{shape[1]}, TypeF16, KuDNN::Layout::A};
    TensorInfo dstInfo = {shape, TypeF16, KuDNN::Layout::AB};
    KuDNN::NormalizationFlags flags = KuDNN::NormalizationFlags::NONE;
    GTEST_EXPECT_TRUE(ForwardFunc1<__fp16>(srcInfo, statInfo, scaleInfo, dstInfo, flags));
}

TEST(RMSnormGTests, ForwardFunc2DF16_2)
{
    Shape shape(100, 100);
    TensorInfo srcInfo = {shape, TypeF16, KuDNN::Layout::AB};
    TensorInfo statInfo = {{shape[0]}, TypeF16, KuDNN::Layout::A};
    TensorInfo scaleInfo = {{shape[1]}, TypeF16, KuDNN::Layout::A};
    TensorInfo dstInfo = {shape, TypeF16, KuDNN::Layout::AB};
    KuDNN::NormalizationFlags flags = KuDNN::NormalizationFlags::USE_SCALE;
    GTEST_EXPECT_TRUE(ForwardFunc1<__fp16>(srcInfo, statInfo, scaleInfo, dstInfo, flags));
}

TEST(RMSnormGTests, ForwardFunc2DF16_3)
{
    Shape shape(100, 100);
    TensorInfo srcInfo = {shape, TypeF16, KuDNN::Layout::AB};
    TensorInfo statInfo = {{shape[0]}, TypeF16, KuDNN::Layout::A};
    TensorInfo scaleInfo = {{shape[1]}, TypeF16, KuDNN::Layout::A};
    TensorInfo dstInfo = {shape, TypeF16, KuDNN::Layout::AB};
    KuDNN::NormalizationFlags flags = KuDNN::NormalizationFlags::USE_SHIFT;
    GTEST_EXPECT_TRUE(ForwardFunc1<__fp16>(srcInfo, statInfo, scaleInfo, dstInfo, flags));
}

TEST(RMSnormGTests, ForwardFunc2DF16_4)
{
    Shape shape(100, 100);
    TensorInfo srcInfo = {shape, TypeF16, KuDNN::Layout::AB};
    TensorInfo statInfo = {{shape[0]}, TypeF16, KuDNN::Layout::A};
    TensorInfo scaleInfo = {{shape[1]}, TypeF16, KuDNN::Layout::A};
    TensorInfo dstInfo = {shape, TypeF16, KuDNN::Layout::AB};
    KuDNN::NormalizationFlags flags = KuDNN::NormalizationFlags::USE_GLOBAL_STATS;
    GTEST_EXPECT_TRUE(ForwardFunc1<__fp16>(srcInfo, statInfo, scaleInfo, dstInfo, flags));
}

TEST(RMSnormGTests, ForwardFunc3DF16_1)
{
    Shape shape(16, 3600, 72);
    TensorInfo srcInfo = {shape, TypeF16, KuDNN::Layout::ABC};
    TensorInfo statInfo = {{shape[0], shape[1]}, TypeF16, KuDNN::Layout::AB};
    TensorInfo scaleInfo = {{shape[2]}, TypeF16, KuDNN::Layout::A};
    TensorInfo dstInfo = {shape, TypeF16, KuDNN::Layout::ABC};
    KuDNN::NormalizationFlags flags = KuDNN::NormalizationFlags::NONE;
    GTEST_EXPECT_TRUE(ForwardFunc1<__fp16>(srcInfo, statInfo, scaleInfo, dstInfo, flags));
}

TEST(RMSnormGTests, ForwardFunc3DF16_2)
{
    Shape shape(16, 3600, 72);
    TensorInfo srcInfo = {shape, TypeF16, KuDNN::Layout::ABC};
    TensorInfo statInfo = {{shape[0], shape[1]}, TypeF16, KuDNN::Layout::AB};
    TensorInfo scaleInfo = {{shape[2]}, TypeF16, KuDNN::Layout::A};
    TensorInfo dstInfo = {shape, TypeF16, KuDNN::Layout::ABC};
    KuDNN::NormalizationFlags flags = KuDNN::NormalizationFlags::USE_SCALE;
    GTEST_EXPECT_TRUE(ForwardFunc1<__fp16>(srcInfo, statInfo, scaleInfo, dstInfo, flags));
}

TEST(RMSnormGTests, ForwardFunc3DF16_3)
{
    Shape shape(16, 3600, 72);
    TensorInfo srcInfo = {shape, TypeF16, KuDNN::Layout::ABC};
    TensorInfo statInfo = {{shape[0], shape[1]}, TypeF16, KuDNN::Layout::AB};
    TensorInfo scaleInfo = {{shape[2]}, TypeF16, KuDNN::Layout::A};
    TensorInfo dstInfo = {shape, TypeF16, KuDNN::Layout::ABC};
    KuDNN::NormalizationFlags flags = KuDNN::NormalizationFlags::USE_SHIFT;
    GTEST_EXPECT_TRUE(ForwardFunc1<__fp16>(srcInfo, statInfo, scaleInfo, dstInfo, flags));
}

TEST(RMSnormGTests, ForwardFunc3DF16_4)
{
    Shape shape(16, 3600, 72);
    TensorInfo srcInfo = {shape, TypeF16, KuDNN::Layout::ABC};
    TensorInfo statInfo = {{shape[0], shape[1]}, TypeF16, KuDNN::Layout::AB};
    TensorInfo scaleInfo = {{shape[2]}, TypeF16, KuDNN::Layout::A};
    TensorInfo dstInfo = {shape, TypeF16, KuDNN::Layout::ABC};
    KuDNN::NormalizationFlags flags = KuDNN::NormalizationFlags::USE_GLOBAL_STATS;
    GTEST_EXPECT_TRUE(ForwardFunc1<__fp16>(srcInfo, statInfo, scaleInfo, dstInfo, flags));
}

TEST(RMSnormGTests, ForwardFunc4DF16_1)
{
    Shape shape(7200, 16, 9, 72);
    TensorInfo srcInfo = {shape, TypeF16, KuDNN::Layout::ABCD};
    TensorInfo statInfo = {{shape[0], shape[1], shape[2]}, TypeF16, KuDNN::Layout::ABC};
    TensorInfo scaleInfo = {{shape[3]}, TypeF16, KuDNN::Layout::A};
    TensorInfo dstInfo = {shape, TypeF16, KuDNN::Layout::ABCD};
    KuDNN::NormalizationFlags flags = KuDNN::NormalizationFlags::NONE;
    GTEST_EXPECT_TRUE(ForwardFunc1<__fp16>(srcInfo, statInfo, scaleInfo, dstInfo, flags));
}

TEST(RMSnormGTests, ForwardFunc4DF16_2)
{
    Shape shape(7200, 16, 9, 72);
    TensorInfo srcInfo = {shape, TypeF16, KuDNN::Layout::ABCD};
    TensorInfo statInfo = {{shape[0], shape[1], shape[2]}, TypeF16, KuDNN::Layout::ABC};
    TensorInfo scaleInfo = {{shape[3]}, TypeF16, KuDNN::Layout::A};
    TensorInfo dstInfo = {shape, TypeF16, KuDNN::Layout::ABCD};
    KuDNN::NormalizationFlags flags = KuDNN::NormalizationFlags::USE_SCALE;
    GTEST_EXPECT_TRUE(ForwardFunc1<__fp16>(srcInfo, statInfo, scaleInfo, dstInfo, flags));
}

TEST(RMSnormGTests, ForwardFunc4DF16_3)
{
    Shape shape(7200, 16, 9, 72);
    TensorInfo srcInfo = {shape, TypeF16, KuDNN::Layout::ABCD};
    TensorInfo statInfo = {{shape[0], shape[1], shape[2]}, TypeF16, KuDNN::Layout::ABC};
    TensorInfo scaleInfo = {{shape[3]}, TypeF16, KuDNN::Layout::A};
    TensorInfo dstInfo = {shape, TypeF16, KuDNN::Layout::ABCD};
    KuDNN::NormalizationFlags flags = KuDNN::NormalizationFlags::USE_SHIFT;
    GTEST_EXPECT_TRUE(ForwardFunc1<__fp16>(srcInfo, statInfo, scaleInfo, dstInfo, flags));
}

TEST(RMSnormGTests, ForwardFunc4DF16_4)
{
    Shape shape(7200, 16, 9, 72);
    TensorInfo srcInfo = {shape, TypeF16, KuDNN::Layout::ABCD};
    TensorInfo statInfo = {{shape[0], shape[1], shape[2]}, TypeF16, KuDNN::Layout::ABC};
    TensorInfo scaleInfo = {{shape[3]}, TypeF16, KuDNN::Layout::A};
    TensorInfo dstInfo = {shape, TypeF16, KuDNN::Layout::ABCD};
    KuDNN::NormalizationFlags flags = KuDNN::NormalizationFlags::USE_GLOBAL_STATS;
    GTEST_EXPECT_TRUE(ForwardFunc1<__fp16>(srcInfo, statInfo, scaleInfo, dstInfo, flags));
}

TEST(RMSnormGTests, ForwardFunc5DF16_1)
{
    Shape shape(5, 5, 5, 5, 15);
    TensorInfo srcInfo = {shape, TypeF16, KuDNN::Layout::ABCDE};
    TensorInfo statInfo = {{shape[0], shape[1], shape[2], shape[3]}, TypeF16, KuDNN::Layout::ABCD};
    TensorInfo scaleInfo = {{shape[4]}, TypeF16, KuDNN::Layout::A};
    TensorInfo dstInfo = {shape, TypeF16, KuDNN::Layout::ABCDE};
    KuDNN::NormalizationFlags flags = KuDNN::NormalizationFlags::NONE;
    GTEST_EXPECT_TRUE(ForwardFunc1<__fp16>(srcInfo, statInfo, scaleInfo, dstInfo, flags));
}

TEST(RMSnormGTests, ForwardFunc5DF16_2)
{
    Shape shape(5, 5, 5, 5, 15);
    TensorInfo srcInfo = {shape, TypeF16, KuDNN::Layout::ABCDE};
    TensorInfo statInfo = {{shape[0], shape[1], shape[2], shape[3]}, TypeF16, KuDNN::Layout::ABCD};
    TensorInfo scaleInfo = {{shape[4]}, TypeF16, KuDNN::Layout::A};
    TensorInfo dstInfo = {shape, TypeF16, KuDNN::Layout::ABCDE};
    KuDNN::NormalizationFlags flags = KuDNN::NormalizationFlags::USE_SCALE;
    GTEST_EXPECT_TRUE(ForwardFunc1<__fp16>(srcInfo, statInfo, scaleInfo, dstInfo, flags));
}

TEST(RMSnormGTests, ForwardFunc5DF16_3)
{
    Shape shape(5, 5, 5, 5, 15);
    TensorInfo srcInfo = {shape, TypeF16, KuDNN::Layout::ABCDE};
    TensorInfo statInfo = {{shape[0], shape[1], shape[2], shape[3]}, TypeF16, KuDNN::Layout::ABCD};
    TensorInfo scaleInfo = {{shape[4]}, TypeF16, KuDNN::Layout::A};
    TensorInfo dstInfo = {shape, TypeF16, KuDNN::Layout::ABCDE};
    KuDNN::NormalizationFlags flags = KuDNN::NormalizationFlags::USE_SHIFT;
    GTEST_EXPECT_TRUE(ForwardFunc1<__fp16>(srcInfo, statInfo, scaleInfo, dstInfo, flags));
}

TEST(RMSnormGTests, ForwardFunc5DF16_4)
{
    Shape shape(5, 5, 5, 5, 15);
    TensorInfo srcInfo = {shape, TypeF16, KuDNN::Layout::ABCDE};
    TensorInfo statInfo = {{shape[0], shape[1], shape[2], shape[3]}, TypeF16, KuDNN::Layout::ABCD};
    TensorInfo scaleInfo = {{shape[4]}, TypeF16, KuDNN::Layout::A};
    TensorInfo dstInfo = {shape, TypeF16, KuDNN::Layout::ABCDE};
    KuDNN::NormalizationFlags flags = KuDNN::NormalizationFlags::USE_GLOBAL_STATS;
    GTEST_EXPECT_TRUE(ForwardFunc1<__fp16>(srcInfo, statInfo, scaleInfo, dstInfo, flags));
}

TEST(RMSnormGTests, ForwardFunc5DF16_5)
{
    Shape shape(5, 5, 5, 5, 15);
    TensorInfo srcInfo = {shape, TypeF16, KuDNN::Layout::ACDEB};
    TensorInfo statInfo = {{shape[0], shape[1], shape[2], shape[3]}, TypeF16, KuDNN::Layout::ABCD};
    TensorInfo scaleInfo = {{shape[4]}, TypeF16, KuDNN::Layout::A};
    TensorInfo dstInfo = {shape, TypeF16, KuDNN::Layout::ACDEB};
    KuDNN::NormalizationFlags flags = KuDNN::NormalizationFlags::USE_GLOBAL_STATS;
    GTEST_EXPECT_TRUE(ForwardFunc1<__fp16>(srcInfo, statInfo, scaleInfo, dstInfo, flags));
}

TEST(RMSnormGTests, ForwardFunc5DF16_6)
{
    Shape shape(5, 5, 5, 5, 15);
    TensorInfo srcInfo = {shape, TypeF16, KuDNN::Layout::ACDEB};
    TensorInfo statInfo = {{shape[0], shape[1], shape[2], shape[3]}, TypeF16, KuDNN::Layout::ABCD};
    TensorInfo scaleInfo = {{shape[4]}, TypeF16, KuDNN::Layout::A};
    TensorInfo dstInfo = {shape, TypeF16, KuDNN::Layout::ACDEB};
    KuDNN::NormalizationFlags flags = KuDNN::NormalizationFlags::USE_SHIFT;
    EXPECT_NO_THROW(ForwardFunc1<__fp16>(srcInfo, statInfo, scaleInfo, dstInfo, flags));
}
TEST(RMSnormGTests, FWDSaveStats)
{
    Shape shape(16, 360, 72);
    TensorInfo srcInfo = {shape, TypeF16, KuDNN::Layout::ABC};
    TensorInfo statInfo = {{shape[0], shape[1]}, TypeF16, KuDNN::Layout::AB};
    TensorInfo scaleInfo = {{shape[2]}, TypeF16, KuDNN::Layout::A};
    TensorInfo dstInfo = {shape, TypeF16, KuDNN::Layout::ABC};
    KuDNN::NormalizationFlags flags = KuDNN::NormalizationFlags::NONE;
    GTEST_EXPECT_TRUE(FWDSaveStats<__fp16>(srcInfo, statInfo, scaleInfo, dstInfo, flags));
}

TEST(RMSnormGTests, FWDSrcDstType)
{
    EXPECT_ANY_THROW(FWDSrcDstType());
}

TEST(RMSnormGTests, FWDScaleType)
{
    EXPECT_ANY_THROW(FWDScaleType());
}

TEST(RMSnormGTests, FWDShiftType)
{
    EXPECT_NO_THROW(FWDShiftType());
}

TEST(RMSnormGTests, FWDGlobalStatsType)
{
    EXPECT_NO_THROW(FWDGlobalStatsType());
}

TEST(RMSnormGTests, ForwardFlagType)
{
    EXPECT_THROW(ForwardFlagType(), KuDNN::Service::Unsupported);
}

TEST(RMSnormGTests, FWDCopyConstructor1)
{
    EXPECT_NO_THROW(ForwardCopyConstructor1());
}

TEST(RMSnormGTests, FWDCopyConstructor2)
{
    EXPECT_NO_THROW(ForwardCopyConstructor2());
}

TEST(RMSnormGTests, FWDMoveConstructor1)
{
    EXPECT_NO_THROW(ForwardMoveConstructor1());
}

TEST(RMSnormGTests, FWDMoveConstructor2)
{
    EXPECT_NO_THROW(ForwardMoveConstructor2());
}

TEST(RMSnormGTests, InputParamExceptD1)
{
    EXPECT_THROW(InputParamExcept1(), KuDNN::Service::LogicError);
}

TEST(RMSnormGTests, InputParamExceptD2)
{
    EXPECT_THROW(InputParamExcept2(), KuDNN::Service::LogicError);
}

TEST(RMSnormGTests, InputParamExceptD3)
{
    EXPECT_THROW(InputParamExcept3(), KuDNN::Service::LogicError);
}

TEST(RMSnormGTests, InputParamExceptD4)
{
    EXPECT_THROW(InputParamExcept4(), KuDNN::Service::LogicError);
}

TEST(RMSnormGTests, InputParamExceptD5)
{
    EXPECT_THROW(InputParamExcept5(), KuDNN::Service::LogicError);
}