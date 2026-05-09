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
#include <cmath>
#include <cmath>
#include <random>

using Shape = KuDNN::Shape;
using SizeType = KuDNN::SizeType;
float ErrorOneData = 1e-2;

template <typename T>
static void reluSimple(T* dst, SizeType size){
    for (SizeType i = 0; i < size; ++i) {
        dst[i] = std::max(static_cast<T>(0), dst[i]);
    }
}

template <typename T>
static void siluSimple(T* dst, SizeType size) {
    if (!dst || size == 0) return;
    for (SizeType i = 0; i < size; ++i) {
        T sigmoid_x = static_cast<T>(1) / (static_cast<T>(1) + std::exp(static_cast<float>(-dst[i])));
        dst[i] = dst[i] * sigmoid_x;
    }
}

template <typename T>
static void geluErfSimple(T* dst, SizeType size) {
    if (!dst || size == 0) return;

    const T sqrt_2 = static_cast<T>(std::sqrt(2.0));
    for (size_t i = 0; i < size; ++i) {
        T phi_x = static_cast<T>(0.5) * (static_cast<T>(1) + std::erf(static_cast<float>(dst[i] / sqrt_2)));
        dst[i] = dst[i] * phi_x;
    }
}

template <typename T>
static void geluTanhSimple(T* dst, SizeType size) {
    if (!dst || size == 0) return;

    const T sqrt_2_over_pi = static_cast<T>(std::sqrt(static_cast<T>(2) / M_PI));
    const T coeff = static_cast<T>(0.044715);

    for (size_t i = 0; i < size; ++i) {
        T inner = sqrt_2_over_pi * (dst[i] + coeff * std::pow(static_cast<float>(dst[i]), 3));
        T tanh_inner = std::tanh(static_cast<float>(inner));
        dst[i] = static_cast<T>(0.5) * dst[i] * (static_cast<T>(1) + tanh_inner);
    }
}


template <typename T>
static void linearActivationSimple(SizeType batch, SizeType m, SizeType n, SizeType k, const T* src, T* weight, T* dst, T* bias, KuDNN::ActivationFunction kind){
    for (SizeType bh = 0; bh < batch; ++bh) {
        for (SizeType i = 0; i < m; ++i) {
            for (SizeType j = 0; j < n; ++j) {
                T sum = 0;
                for (SizeType l = 0; l < k; ++l) {
                    sum += src[bh * m * k + i * k + l] * weight[bh * n * k + n * l + j];
                }
                dst[bh * m * n + i * n + j] = sum + bias[bh * m * n + i * n + j];
            }
        }
    }
    if(kind == KuDNN::ActivationFunction::RELU){
        reluSimple<T>(dst, batch * m * n);
    } else if(kind == KuDNN::ActivationFunction::SWISH){
        siluSimple<T>(dst, batch * m * n);
    } else if(kind == KuDNN::ActivationFunction::GELU_ERF){
        geluErfSimple<T>(dst, batch * m * n);
    } else if(kind == KuDNN::ActivationFunction::GELU_TANH){
        geluTanhSimple<T>(dst, batch * m * n);
    }
}

template <typename T>
static void floatDataInit(T **src, T **weight, T **dst, T **dstRef, T **bias, SizeType srcTotalSize, SizeType weightTotalSize, SizeType dstTotalSize)
{
    *src = (T *)malloc(srcTotalSize * sizeof(T));
    *weight = (T *)malloc(weightTotalSize * sizeof(T));
    *dst = (T *)malloc(dstTotalSize * sizeof(T));
    *dstRef = (T *)malloc(dstTotalSize * sizeof(T));
    *bias = (T *)malloc(dstTotalSize * sizeof(T));

    if (*src == nullptr || *weight == nullptr || *dst == nullptr || *dstRef == nullptr || *bias == nullptr) {
        std::cerr << "Memory allocation failed" << std::endl;
        return;
    }
    std::uniform_real_distribution<float> u(-1, 1);
    static std::default_random_engine e(time(NULL));

    for (SizeType i = 0; i < srcTotalSize; ++i) {
        (*src)[i] = (T)u(e);
    }
    for (SizeType i = 0; i < weightTotalSize; ++i) {
        (*weight)[i] = (T)u(e);
    }
    for (SizeType i = 0; i < dstTotalSize; ++i) {
        (*bias)[i] = (T)u(e);
    }
    for (SizeType i = 0; i < dstTotalSize; ++i) {
        (*dst)[i] = (T)0;
    }
}

static std::pair<float, float> compareEle(float ref, float diff, KuDNN::Element::TypeT type, int K)
{
    float e = 0.0f;
    float eps = 0.0f;
    std::pair<float, float> pairEps = {0.0f, 0.0f};
    switch (type) {
        case KuDNN::Element::TypeT::F16:
            eps = 1e-3 * K;
            e = (std::fabs(ref) > eps) ? diff / ref : static_cast<float>(diff);
            pairEps = {e, eps};
            break;
        case KuDNN::Element::TypeT::BF16:
            eps = 1e-2 * K;
            e = (std::fabs(ref) > eps) ? diff / ref : static_cast<float>(diff);
            pairEps = {e, eps};
            break;
        case KuDNN::Element::TypeT::F32:
            eps = 1e-4;
            e = (std::fabs(ref) > 1e-4) ? diff / ref : static_cast<float>(diff);
            pairEps = {e, eps};
            break;
        default:
            eps = 1;
            e = diff;
            pairEps = {e, eps};
            break;
    }
    return pairEps;
}

template <typename T>
static float findMaxValue(T* value1, SizeType valueSize1, T* value2, SizeType valueSize2) {
    T maxVal = value1[0];

    for (SizeType i = 0; i < valueSize1; ++i) {
        maxVal = std::max(maxVal, value1[i]);
    }

    for (SizeType i = 0; i < valueSize2; ++i) {
        maxVal = std::max(maxVal, value2[i]);
    }
    return static_cast<float>(maxVal);
}

template <typename T>
static bool Test_Template_Function(const KuDNN::TensorInfo srcTensor, const KuDNN::TensorInfo weightTensor,
                                   const KuDNN::TensorInfo dstTensor, const KuDNN::TensorInfo biasTensor,
                                   float alpha, float beta, KuDNN::ActivationFunction algKind, int numThreads){

    KuDNN::LinearActivationLayerFWD linearActivationLayerFwd(srcTensor, weightTensor, dstTensor, biasTensor, alpha, beta, algKind, numThreads);

    T *src = nullptr, *weight = nullptr, *dst = nullptr, *dstRef = nullptr, *bias = nullptr;
    floatDataInit<T>(&src, &weight, &dst, &dstRef, &bias, srcTensor.GetTotalTensorSize(), weightTensor.GetTotalTensorSize(), dstTensor.GetTotalTensorSize());

    Shape srcShape = srcTensor.GetDims();
    Shape weightShape = weightTensor.GetDims();
    SizeType srcDimNums = srcShape.GetNumDims();
    SizeType weightDimNums = weightShape.GetNumDims();
    SizeType m = srcShape[srcDimNums - 2];
    SizeType n = weightShape[weightDimNums - 1];
    SizeType k = srcShape[srcDimNums - 1];
    SizeType batch = dstTensor.GetTotalTensorSize() / (m * n);

    linearActivationSimple(batch,m, n, k, src, weight, dstRef, bias, algKind);

    linearActivationLayerFwd.Run(src, weight, dst, bias, numThreads);

    float maxValue = findMaxValue<T>(src, batch * m * k, weight, batch * n * k);

    bool flag = true;
    for (SizeType i = 0; i < batch * m * n; ++i) {
        float diff = std::abs(static_cast<float>(*(dst + i) - *(dstRef + i)));
        auto pairEps = compareEle(maxValue, diff, KuDNN::Element::MatchType<T>(), k);

        if(std::abs(pairEps.first) > pairEps.second){
            flag = false;
            break;
        }
    }
    if (src != nullptr){
        free(src);
    }
    if (weight != nullptr){
        free(weight);
    }
    if (dst != nullptr){
        free(dst);
    }
    if (dstRef != nullptr){
        free(dstRef);
    }
    if (bias != nullptr){
        free(bias);
    }
    return flag;
}

static void NoPackCopyConstructor1()
{
    const KuDNN::TensorInfo srcTensor = {{1, 1, 1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCDE};
    const KuDNN::TensorInfo weightTensor = {{1, 1, 1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCDE};
    const KuDNN::TensorInfo dstTensor = {{1, 1, 1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCDE};
    const KuDNN::TensorInfo biasTensor = {{1, 1, 1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCDE};
    float alpha = 1.0f;
    float beta = 0.0f;
    KuDNN::ActivationFunction algKind = KuDNN::ActivationFunction::RELU;
    int numthreads = 0;
    KuDNN::LinearActivationLayerFWD linearActivationLayerFwd1(srcTensor, weightTensor, dstTensor, biasTensor, alpha, beta, algKind, numthreads);
    KuDNN::LinearActivationLayerFWD linearActivationLayerFwd2(linearActivationLayerFwd1);
    for (size_t i = 0; i < g_Tensor_data.size() / 4; i += 4) {
        KuDNN::LinearActivationLayerFWD linearActivationLayerFwd1(g_Tensor_data[i], g_Tensor_data[i + 1], g_Tensor_data[i + 2],
                                                                                          g_Tensor_data[i + 3], alpha, beta, algKind, numthreads);
        KuDNN::LinearActivationLayerFWD linearActivationLayerFwd2(linearActivationLayerFwd1);
    }
}

static void NoPackMoveConstructor1()
{
    const KuDNN::TensorInfo srcTensor = {{1, 1, 1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCDE};
    const KuDNN::TensorInfo weightTensor = {{1, 1, 1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCDE};
    const KuDNN::TensorInfo dstTensor = {{1, 1, 1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCDE};
    const KuDNN::TensorInfo biasTensor = {{1, 1, 1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCDE};
    float alpha = 1.0f;
    float beta = 0.0f;
    KuDNN::ActivationFunction algKind = KuDNN::ActivationFunction::RELU;
    int numthreads = 0;
    KuDNN::LinearActivationLayerFWD linearActivationLayerFwd1(srcTensor, weightTensor, dstTensor, biasTensor, alpha, beta, algKind, numthreads);
    KuDNN::LinearActivationLayerFWD linearActivationLayerFwd2(std::move(linearActivationLayerFwd1));
    for (size_t i = 0; i < g_Tensor_data.size() / 4; i += 4) {
        KuDNN::LinearActivationLayerFWD linearActivationLayerFwd1(
                g_Tensor_data[i], g_Tensor_data[i + 1], g_Tensor_data[i + 2], g_Tensor_data[i + 3], alpha, beta, algKind, numthreads);
        KuDNN::LinearActivationLayerFWD linearActivationLayerFwd2(std::move(linearActivationLayerFwd1));
    }
}

static void InputParamExceptTest1(){
    Shape srcShape(-1, 1, 1, 1, 1);
    const KuDNN::TensorInfo srcTensor = {srcShape, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCDE};
    const KuDNN::TensorInfo weightTensor = {{1, 1, 1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCDE};
    const KuDNN::TensorInfo dstTensor = {{1, 1, 1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCDE};
    const KuDNN::TensorInfo biasTensor = {{1, 1, 1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCDE};
    float alpha = 1.0f;
    float beta = 0.0f;
    KuDNN::ActivationFunction algKind = KuDNN::ActivationFunction::RELU;
    int numthreads = 0;
    KuDNN::LinearActivationLayerFWD linearActivationLayerFwd1(srcTensor, weightTensor, dstTensor, biasTensor, alpha, beta, algKind, numthreads);
}

static void InputParamExceptTest2(){
    const KuDNN::TensorInfo srcTensor = {{1, 1, 1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCDE};
    const KuDNN::TensorInfo weightTensor = {{1, 1, 1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCDE};
    const KuDNN::TensorInfo dstTensor = {{1, 1, 1, 1, 1}, KuDNN::Element::TypeT::S8, KuDNN::Layout::ABCDE};
    const KuDNN::TensorInfo biasTensor = {{1, 1, 1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCDE};
    float alpha = 1.0f;
    float beta = 0.0f;
    KuDNN::ActivationFunction algKind = KuDNN::ActivationFunction::SWISH;
    int numthreads = 0;
    KuDNN::LinearActivationLayerFWD linearActivationLayerFwd1(srcTensor, weightTensor, dstTensor, biasTensor, alpha, beta, algKind, numthreads);
}

static void InputParamExceptTest3(){
    const KuDNN::TensorInfo srcTensor = {{1, 1, 1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCD};
    const KuDNN::TensorInfo weightTensor = {{1, 1, 1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCDE};
    const KuDNN::TensorInfo dstTensor = {{1, 1, 1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCDE};
    const KuDNN::TensorInfo biasTensor = {{1, 1, 1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCDE};
    float alpha = 1.0f;
    float beta = 0.0f;
    KuDNN::ActivationFunction algKind = KuDNN::ActivationFunction::RELU;
    int numthreads = 0;
    KuDNN::LinearActivationLayerFWD linearActivationLayerFwd1(srcTensor, weightTensor, dstTensor, biasTensor, alpha, beta, algKind, numthreads);
}

static void InputParamExceptTest4(){
    const KuDNN::TensorInfo srcTensor = {{1, 1, 1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCDE};
    const KuDNN::TensorInfo weightTensor = {{1, 1, 2, 2, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCDE};
    const KuDNN::TensorInfo dstTensor = {{1, 1, 1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCDE};
    const KuDNN::TensorInfo biasTensor = {{1, 1, 1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCDE};
    float alpha = 1.0f;
    float beta = 0.0f;
    KuDNN::ActivationFunction algKind = KuDNN::ActivationFunction::RELU;
    int numthreads = 0;
    KuDNN::LinearActivationLayerFWD linearActivationLayerFwd1(srcTensor, weightTensor, dstTensor, biasTensor, alpha, beta, algKind, numthreads);
}

static void InputNullptrExceptTest5(){
    const KuDNN::TensorInfo srcTensor = {{1, 1, 1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCDE};
    const KuDNN::TensorInfo weightTensor = {{1, 1, 1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCDE};
    const KuDNN::TensorInfo dstTensor = {{1, 1, 1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCDE};
    const KuDNN::TensorInfo biasTensor = {{1, 1, 1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCDE};
    float alpha = 1.0f;
    float beta = 0.0f;
    KuDNN::ActivationFunction algKind = KuDNN::ActivationFunction::RELU;
    int numthreads = 0;
    float *src = nullptr, *weight = nullptr, *dst = nullptr, *bias = nullptr;
    KuDNN::LinearActivationLayerFWD linearActivationLayerFwd1(srcTensor, weightTensor, dstTensor, biasTensor, alpha, beta, algKind, numthreads);
    linearActivationLayerFwd1.Run(src, weight, dst, bias, numthreads);
}

static void InputParamExceptTest6(){
    Shape srcShape(1, -1, 1, 1, 1);
    const KuDNN::TensorInfo srcTensor = {srcShape, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCDE};
    const KuDNN::TensorInfo weightTensor = {{1, 1, 1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCDE};
    const KuDNN::TensorInfo dstTensor = {{1, 1, 1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCDE};
    const KuDNN::TensorInfo biasTensor = {{1, 1, 1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCDE};
    float alpha = 1.0f;
    float beta = 0.0f;
    KuDNN::ActivationFunction algKind = KuDNN::ActivationFunction::RELU;
    int numthreads = 0;
    KuDNN::LinearActivationLayerFWD linearActivationLayerFwd1(srcTensor, weightTensor, dstTensor, biasTensor, alpha, beta, algKind, numthreads);
}

static void InputParamExceptTest7(){
    Shape srcShape(1, 1, -1, 1, 1);
    const KuDNN::TensorInfo srcTensor = {srcShape, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCDE};
    const KuDNN::TensorInfo weightTensor = {{1, 1, 1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCDE};
    const KuDNN::TensorInfo dstTensor = {{1, 1, 1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCDE};
    const KuDNN::TensorInfo biasTensor = {{1, 1, 1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCDE};
    float alpha = 1.0f;
    float beta = 0.0f;
    KuDNN::ActivationFunction algKind = KuDNN::ActivationFunction::RELU;
    int numthreads = 0;
    KuDNN::LinearActivationLayerFWD linearActivationLayerFwd1(srcTensor, weightTensor, dstTensor, biasTensor, alpha, beta, algKind, numthreads);
}

static void InputParamExceptTest8(){
    Shape srcShape(1, 1, 1, -1, 1);
    const KuDNN::TensorInfo srcTensor = {srcShape, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCDE};
    const KuDNN::TensorInfo weightTensor = {{1, 1, 1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCDE};
    const KuDNN::TensorInfo dstTensor = {{1, 1, 1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCDE};
    const KuDNN::TensorInfo biasTensor = {{1, 1, 1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCDE};
    float alpha = 1.0f;
    float beta = 0.0f;
    KuDNN::ActivationFunction algKind = KuDNN::ActivationFunction::RELU;
    int numthreads = 0;
    KuDNN::LinearActivationLayerFWD linearActivationLayerFwd1(srcTensor, weightTensor, dstTensor, biasTensor, alpha, beta, algKind, numthreads);
}

static void InputParamExceptTest9(){
    Shape weiShape(1, 1, 1, 1, -1);
    const KuDNN::TensorInfo srcTensor = {{1, 1, 1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCDE};
    const KuDNN::TensorInfo weightTensor = {weiShape, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCDE};
    const KuDNN::TensorInfo dstTensor = {{1, 1, 1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCDE};
    const KuDNN::TensorInfo biasTensor = {{1, 1, 1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCDE};
    float alpha = 1.0f;
    float beta = 0.0f;
    KuDNN::ActivationFunction algKind = KuDNN::ActivationFunction::RELU;
    int numthreads = 0;
    KuDNN::LinearActivationLayerFWD linearActivationLayerFwd1(srcTensor, weightTensor, dstTensor, biasTensor, alpha, beta, algKind, numthreads);
}

static void InputParamExceptTest10(){
    Shape srcShape(1, 1, 1, 1, 1);
    const KuDNN::TensorInfo srcTensor = {srcShape, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCDE};
    const KuDNN::TensorInfo weightTensor = {{1, 1, 1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCDE};
    const KuDNN::TensorInfo dstTensor = {{1, 1, 1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCDE};
    const KuDNN::TensorInfo biasTensor = {{1, 1, 1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCDE};
    float alpha = 1.0f;
    float beta = 0.0f;
    KuDNN::ActivationFunction algKind = KuDNN::ActivationFunction::UNIMPLEMENTED;
    int numthreads = 0;
    KuDNN::LinearActivationLayerFWD linearActivationLayerFwd1(srcTensor, weightTensor, dstTensor, biasTensor, alpha, beta, algKind, numthreads);
}

static void InputParamExceptTest11(){
    Shape srcShape(1, 1, 1, 1, 1);
    const KuDNN::TensorInfo srcTensor = {srcShape, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCDE};
    const KuDNN::TensorInfo weightTensor = {{1, 1, 1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCDE};
    const KuDNN::TensorInfo dstTensor = {{1, 1, 1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCDE};
    const KuDNN::TensorInfo biasTensor = {{1, 1, 1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCDE};
    float alpha = 1.0f;
    float beta = 0.0f;
    KuDNN::ActivationFunction algKind = KuDNN::ActivationFunction::UNIMPLEMENTED;
    int numthreads = -1;
    KuDNN::LinearActivationLayerFWD linearActivationLayerFwd1(srcTensor, weightTensor, dstTensor, biasTensor, alpha, beta, algKind, numthreads);
}
// test

TEST(LinearActivationTests, NoPackCopyConstructor1)
{
EXPECT_NO_THROW(NoPackCopyConstructor1());
}

TEST(LinearActivationTests, NoPackMoveConstructor1)
{
EXPECT_NO_THROW(NoPackMoveConstructor1());
}

TEST(LinearActivationTests, Param_ShapeExcept1_Test)
{
EXPECT_THROW(InputParamExceptTest1(), KuDNN::Service::LogicError);
}

TEST(LinearActivationTests, Param_ShapeExcept2_Test)
{
EXPECT_THROW(InputParamExceptTest4(), KuDNN::Service::LogicError);
}

TEST(LinearActivationTests, Param_DataType_Except_Test)
{
EXPECT_THROW(InputParamExceptTest2(), KuDNN::Service::Unsupported);
}

TEST(LinearActivationTests, Param_Layout_Except_Test)
{
EXPECT_THROW(InputParamExceptTest3(), KuDNN::Service::LogicError);
}

TEST(LinearActivationTests, Input_Nullptr_Except_Test)
{
EXPECT_THROW(InputNullptrExceptTest5(), KuDNN::Service::LogicError);
}

TEST(LinearActivationTests, Param_ShapeExceptD2_Test)
{
    EXPECT_THROW(InputParamExceptTest6(), KuDNN::Service::LogicError);
}

TEST(LinearActivationTests, Param_ShapeExceptD3_Test)
{
    EXPECT_THROW(InputParamExceptTest7(), KuDNN::Service::LogicError);
}

TEST(LinearActivationTests, Param_ShapeExceptM_Test)
{
    EXPECT_THROW(InputParamExceptTest8(), KuDNN::Service::LogicError);
}

TEST(LinearActivationTests, Param_ShapeExceptN_Test)
{
    EXPECT_THROW(InputParamExceptTest9(), KuDNN::Service::LogicError);
}

TEST(LinearActivationTests, Param_ActivationFunction_Except_Test)
{
    EXPECT_THROW(InputParamExceptTest10(), KuDNN::Service::Unsupported);
}

TEST(LinearActivationTests, Param_NumThreads_Except_Test)
{
    EXPECT_THROW(InputParamExceptTest11(), KuDNN::Service::LogicError);
}
// test RELU FP32
TEST(LinearActivationTests, FWD_LINEAR_RELU_2D_FP32)
{
const KuDNN::TensorInfo srcTensor = {{110, 20}, KuDNN::Element::TypeT::F32, KuDNN::Layout::AB};
const KuDNN::TensorInfo weightTensor = {{20, 200}, KuDNN::Element::TypeT::F32, KuDNN::Layout::AB};
const KuDNN::TensorInfo dstTensor = {{110, 200}, KuDNN::Element::TypeT::F32, KuDNN::Layout::AB};
const KuDNN::TensorInfo biasTensor = {{110, 200}, KuDNN::Element::TypeT::F32, KuDNN::Layout::AB};
float alpha = 0.0f;
float beta = 0.0f;
KuDNN::ActivationFunction algKind = KuDNN::ActivationFunction::RELU;
int numThreads = 0;
EXPECT_TRUE(Test_Template_Function<float>(srcTensor, weightTensor, dstTensor, biasTensor, alpha, beta, algKind, numThreads));
}

TEST(LinearActivationTests, FWD_LINEAR_RELU_3D_FP32)
{
const KuDNN::TensorInfo srcTensor = {{2, 110, 20}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABC};
const KuDNN::TensorInfo weightTensor = {{2, 20, 200}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABC};
const KuDNN::TensorInfo dstTensor = {{2 , 110, 200}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABC};
const KuDNN::TensorInfo biasTensor = {{2, 110, 200}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABC};
float alpha = 0.0f;
float beta = 0.0f;
KuDNN::ActivationFunction algKind = KuDNN::ActivationFunction::RELU;
int numThreads = 0;
EXPECT_TRUE(Test_Template_Function<float>(srcTensor, weightTensor, dstTensor, biasTensor, alpha, beta, algKind, numThreads));
}

TEST(LinearActivationTests, FWD_LINEAR_RELU_4D_FP32)
{
const KuDNN::TensorInfo srcTensor = {{3, 2, 110, 20}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCD};
const KuDNN::TensorInfo weightTensor = {{3, 2, 20, 200}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCD};
const KuDNN::TensorInfo dstTensor = {{3, 2 , 110, 200}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCD};
const KuDNN::TensorInfo biasTensor = {{3, 2, 110, 200}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCD};
float alpha = 0.0f;
float beta = 0.0f;
KuDNN::ActivationFunction algKind = KuDNN::ActivationFunction::RELU;
int numThreads = 0;
EXPECT_TRUE(Test_Template_Function<float>(srcTensor, weightTensor, dstTensor, biasTensor, alpha, beta, algKind, numThreads));
}

TEST(LinearActivationTests, FWD_LINEAR_RELU_5D_FP32)
{
const KuDNN::TensorInfo srcTensor = {{4, 3, 2, 110, 20}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCDE};
const KuDNN::TensorInfo weightTensor = {{4, 3, 2, 20, 200}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCDE};
const KuDNN::TensorInfo dstTensor = {{4, 3, 2 , 110, 200}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCDE};
const KuDNN::TensorInfo biasTensor = {{4, 3, 2, 110, 200}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCDE};
float alpha = 0.0f;
float beta = 0.0f;
KuDNN::ActivationFunction algKind = KuDNN::ActivationFunction::RELU;
int numThreads = 0;
EXPECT_TRUE(Test_Template_Function<float>(srcTensor, weightTensor, dstTensor, biasTensor, alpha, beta, algKind, numThreads));
}

// test RELU FP16
TEST(LinearActivationTests, FWD_LINEAR_RELU_2D_FP16)
{
const KuDNN::TensorInfo srcTensor = {{110, 20}, KuDNN::Element::TypeT::F16, KuDNN::Layout::AB};
const KuDNN::TensorInfo weightTensor = {{20, 200}, KuDNN::Element::TypeT::F16, KuDNN::Layout::AB};
const KuDNN::TensorInfo dstTensor = {{110, 200}, KuDNN::Element::TypeT::F16, KuDNN::Layout::AB};
const KuDNN::TensorInfo biasTensor = {{110, 200}, KuDNN::Element::TypeT::F16, KuDNN::Layout::AB};
float alpha = 0.0f;
float beta = 0.0f;
KuDNN::ActivationFunction algKind = KuDNN::ActivationFunction::RELU;
int numThreads = 0;
EXPECT_TRUE(Test_Template_Function<__fp16>(srcTensor, weightTensor, dstTensor, biasTensor, alpha, beta, algKind, numThreads));
}

TEST(LinearActivationTests, FWD_LINEAR_RELU_3D_FP16)
{
const KuDNN::TensorInfo srcTensor = {{2, 110, 20}, KuDNN::Element::TypeT::F16, KuDNN::Layout::ABC};
const KuDNN::TensorInfo weightTensor = {{2, 20, 200}, KuDNN::Element::TypeT::F16, KuDNN::Layout::ABC};
const KuDNN::TensorInfo dstTensor = {{2 , 110, 200}, KuDNN::Element::TypeT::F16, KuDNN::Layout::ABC};
const KuDNN::TensorInfo biasTensor = {{2, 110, 200}, KuDNN::Element::TypeT::F16, KuDNN::Layout::ABC};
float alpha = 0.0f;
float beta = 0.0f;
KuDNN::ActivationFunction algKind = KuDNN::ActivationFunction::RELU;
int numThreads = 0;
EXPECT_TRUE(Test_Template_Function<__fp16>(srcTensor, weightTensor, dstTensor, biasTensor, alpha, beta, algKind, numThreads));
}

TEST(LinearActivationTests, FWD_LINEAR_RELU_4D_FP16)
{
const KuDNN::TensorInfo srcTensor = {{3, 2, 110, 20}, KuDNN::Element::TypeT::F16, KuDNN::Layout::ABCD};
const KuDNN::TensorInfo weightTensor = {{3, 2, 20, 200}, KuDNN::Element::TypeT::F16, KuDNN::Layout::ABCD};
const KuDNN::TensorInfo dstTensor = {{3, 2 , 110, 200}, KuDNN::Element::TypeT::F16, KuDNN::Layout::ABCD};
const KuDNN::TensorInfo biasTensor = {{3, 2, 110, 200}, KuDNN::Element::TypeT::F16, KuDNN::Layout::ABCD};
float alpha = 0.0f;
float beta = 0.0f;
KuDNN::ActivationFunction algKind = KuDNN::ActivationFunction::RELU;
int numThreads = 0;
EXPECT_TRUE(Test_Template_Function<__fp16>(srcTensor, weightTensor, dstTensor, biasTensor, alpha, beta, algKind, numThreads));
}

TEST(LinearActivationTests, FWD_LINEAR_RELU_5D_FP16)
{
const KuDNN::TensorInfo srcTensor = {{4, 3, 2, 110, 20}, KuDNN::Element::TypeT::F16, KuDNN::Layout::ABCDE};
const KuDNN::TensorInfo weightTensor = {{4, 3, 2, 20, 200}, KuDNN::Element::TypeT::F16, KuDNN::Layout::ABCDE};
const KuDNN::TensorInfo dstTensor = {{4, 3, 2 , 110, 200}, KuDNN::Element::TypeT::F16, KuDNN::Layout::ABCDE};
const KuDNN::TensorInfo biasTensor = {{4, 3, 2, 110, 200}, KuDNN::Element::TypeT::F16, KuDNN::Layout::ABCDE};
float alpha = 0.0f;
float beta = 0.0f;
KuDNN::ActivationFunction algKind = KuDNN::ActivationFunction::RELU;
int numThreads = 0;
EXPECT_TRUE(Test_Template_Function<__fp16>(srcTensor, weightTensor, dstTensor, biasTensor, alpha, beta, algKind, numThreads));
}

// test RELU BF16
TEST(LinearActivationTests, FWD_LINEAR_RELU_2D_BF16)
{
const KuDNN::TensorInfo srcTensor = {{110, 20}, KuDNN::Element::TypeT::BF16, KuDNN::Layout::AB};
const KuDNN::TensorInfo weightTensor = {{20, 200}, KuDNN::Element::TypeT::BF16, KuDNN::Layout::AB};
const KuDNN::TensorInfo dstTensor = {{110, 200}, KuDNN::Element::TypeT::BF16, KuDNN::Layout::AB};
const KuDNN::TensorInfo biasTensor = {{110, 200}, KuDNN::Element::TypeT::BF16, KuDNN::Layout::AB};
float alpha = 0.0f;
float beta = 0.0f;
KuDNN::ActivationFunction algKind = KuDNN::ActivationFunction::RELU;
int numThreads = 0;
EXPECT_TRUE(Test_Template_Function<__bf16>(srcTensor, weightTensor, dstTensor, biasTensor, alpha, beta, algKind, numThreads));
}

TEST(LinearActivationTests, FWD_LINEAR_RELU_3D_BF16)
{
const KuDNN::TensorInfo srcTensor = {{2, 110, 20}, KuDNN::Element::TypeT::BF16, KuDNN::Layout::ABC};
const KuDNN::TensorInfo weightTensor = {{2, 20, 200}, KuDNN::Element::TypeT::BF16, KuDNN::Layout::ABC};
const KuDNN::TensorInfo dstTensor = {{2 , 110, 200}, KuDNN::Element::TypeT::BF16, KuDNN::Layout::ABC};
const KuDNN::TensorInfo biasTensor = {{2, 110, 200}, KuDNN::Element::TypeT::BF16, KuDNN::Layout::ABC};
float alpha = 0.0f;
float beta = 0.0f;
KuDNN::ActivationFunction algKind = KuDNN::ActivationFunction::RELU;
int numThreads = 0;
EXPECT_TRUE(Test_Template_Function<__bf16>(srcTensor, weightTensor, dstTensor, biasTensor, alpha, beta, algKind, numThreads));
}

TEST(LinearActivationTests, FWD_LINEAR_RELU_4D_BF16)
{
const KuDNN::TensorInfo srcTensor = {{3, 2, 110, 20}, KuDNN::Element::TypeT::BF16, KuDNN::Layout::ABCD};
const KuDNN::TensorInfo weightTensor = {{3, 2, 20, 200}, KuDNN::Element::TypeT::BF16, KuDNN::Layout::ABCD};
const KuDNN::TensorInfo dstTensor = {{3, 2 , 110, 200}, KuDNN::Element::TypeT::BF16, KuDNN::Layout::ABCD};
const KuDNN::TensorInfo biasTensor = {{3, 2, 110, 200}, KuDNN::Element::TypeT::BF16, KuDNN::Layout::ABCD};
float alpha = 0.0f;
float beta = 0.0f;
KuDNN::ActivationFunction algKind = KuDNN::ActivationFunction::RELU;
int numThreads = 0;
EXPECT_TRUE(Test_Template_Function<__bf16>(srcTensor, weightTensor, dstTensor, biasTensor, alpha, beta, algKind, numThreads));
}

TEST(LinearActivationTests, FWD_LINEAR_RELU_5D_BF16)
{
const KuDNN::TensorInfo srcTensor = {{4, 3, 2, 110, 20}, KuDNN::Element::TypeT::BF16, KuDNN::Layout::ABCDE};
const KuDNN::TensorInfo weightTensor = {{4, 3, 2, 20, 200}, KuDNN::Element::TypeT::BF16, KuDNN::Layout::ABCDE};
const KuDNN::TensorInfo dstTensor = {{4, 3, 2 , 110, 200}, KuDNN::Element::TypeT::BF16, KuDNN::Layout::ABCDE};
const KuDNN::TensorInfo biasTensor = {{4, 3, 2, 110, 200}, KuDNN::Element::TypeT::BF16, KuDNN::Layout::ABCDE};
float alpha = 0.0f;
float beta = 0.0f;
KuDNN::ActivationFunction algKind = KuDNN::ActivationFunction::RELU;
int numThreads = 0;
EXPECT_TRUE(Test_Template_Function<__bf16>(srcTensor, weightTensor, dstTensor, biasTensor, alpha, beta, algKind, numThreads));
}

// test SILU FP32
TEST(LinearActivationTests, FWD_LINEAR_SILU_2D_FP32)
{
const KuDNN::TensorInfo srcTensor = {{110, 20}, KuDNN::Element::TypeT::F32, KuDNN::Layout::AB};
const KuDNN::TensorInfo weightTensor = {{20, 200}, KuDNN::Element::TypeT::F32, KuDNN::Layout::AB};
const KuDNN::TensorInfo dstTensor = {{110, 200}, KuDNN::Element::TypeT::F32, KuDNN::Layout::AB};
const KuDNN::TensorInfo biasTensor = {{110, 200}, KuDNN::Element::TypeT::F32, KuDNN::Layout::AB};
float alpha = 1.0f;
float beta = 0.0f;
KuDNN::ActivationFunction algKind = KuDNN::ActivationFunction::SWISH;
int numThreads = 0;
EXPECT_TRUE(Test_Template_Function<float>(srcTensor, weightTensor, dstTensor, biasTensor, alpha, beta, algKind, numThreads));
}

TEST(LinearActivationTests, FWD_LINEAR_SILU_3D_FP32)
{
const KuDNN::TensorInfo srcTensor = {{2, 110, 20}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABC};
const KuDNN::TensorInfo weightTensor = {{2, 20, 200}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABC};
const KuDNN::TensorInfo dstTensor = {{2 , 110, 200}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABC};
const KuDNN::TensorInfo biasTensor = {{2, 110, 200}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABC};
float alpha = 1.0f;
float beta = 0.0f;
KuDNN::ActivationFunction algKind = KuDNN::ActivationFunction::SWISH;
int numThreads = 0;
EXPECT_TRUE(Test_Template_Function<float>(srcTensor, weightTensor, dstTensor, biasTensor, alpha, beta, algKind, numThreads));
}

TEST(LinearActivationTests, FWD_LINEAR_SILU_4D_FP32)
{
const KuDNN::TensorInfo srcTensor = {{3, 2, 110, 20}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCD};
const KuDNN::TensorInfo weightTensor = {{3, 2, 20, 200}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCD};
const KuDNN::TensorInfo dstTensor = {{3, 2 , 110, 200}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCD};
const KuDNN::TensorInfo biasTensor = {{3, 2, 110, 200}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCD};
float alpha = 1.0f;
float beta = 0.0f;
KuDNN::ActivationFunction algKind = KuDNN::ActivationFunction::SWISH;
int numThreads = 0;
EXPECT_TRUE(Test_Template_Function<float>(srcTensor, weightTensor, dstTensor, biasTensor, alpha, beta, algKind, numThreads));
}

TEST(LinearActivationTests, FWD_LINEAR_SILU_5D_FP32)
{
const KuDNN::TensorInfo srcTensor = {{4, 3, 2, 110, 20}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCDE};
const KuDNN::TensorInfo weightTensor = {{4, 3, 2, 20, 200}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCDE};
const KuDNN::TensorInfo dstTensor = {{4, 3, 2 , 110, 200}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCDE};
const KuDNN::TensorInfo biasTensor = {{4, 3, 2, 110, 200}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCDE};
float alpha = 1.0f;
float beta = 0.0f;
KuDNN::ActivationFunction algKind = KuDNN::ActivationFunction::SWISH;
int numThreads = 0;
EXPECT_TRUE(Test_Template_Function<float>(srcTensor, weightTensor, dstTensor, biasTensor, alpha, beta, algKind, numThreads));
}

// test SILU FP16
TEST(LinearActivationTests, FWD_LINEAR_SILU_2D_FP16)
{
const KuDNN::TensorInfo srcTensor = {{110, 20}, KuDNN::Element::TypeT::F16, KuDNN::Layout::AB};
const KuDNN::TensorInfo weightTensor = {{20, 200}, KuDNN::Element::TypeT::F16, KuDNN::Layout::AB};
const KuDNN::TensorInfo dstTensor = {{110, 200}, KuDNN::Element::TypeT::F16, KuDNN::Layout::AB};
const KuDNN::TensorInfo biasTensor = {{110, 200}, KuDNN::Element::TypeT::F16, KuDNN::Layout::AB};
float alpha = 1.0f;
float beta = 0.0f;
KuDNN::ActivationFunction algKind = KuDNN::ActivationFunction::SWISH;
int numThreads = 0;
EXPECT_TRUE(Test_Template_Function<__fp16>(srcTensor, weightTensor, dstTensor, biasTensor, alpha, beta, algKind, numThreads));
}

TEST(LinearActivationTests, FWD_LINEAR_SILU_3D_FP16)
{
const KuDNN::TensorInfo srcTensor = {{2, 110, 20}, KuDNN::Element::TypeT::F16, KuDNN::Layout::ABC};
const KuDNN::TensorInfo weightTensor = {{2, 20, 200}, KuDNN::Element::TypeT::F16, KuDNN::Layout::ABC};
const KuDNN::TensorInfo dstTensor = {{2 , 110, 200}, KuDNN::Element::TypeT::F16, KuDNN::Layout::ABC};
const KuDNN::TensorInfo biasTensor = {{2, 110, 200}, KuDNN::Element::TypeT::F16, KuDNN::Layout::ABC};
float alpha = 1.0f;
float beta = 0.0f;
KuDNN::ActivationFunction algKind = KuDNN::ActivationFunction::SWISH;
int numThreads = 0;
EXPECT_TRUE(Test_Template_Function<__fp16>(srcTensor, weightTensor, dstTensor, biasTensor, alpha, beta, algKind, numThreads));
}

TEST(LinearActivationTests, FWD_LINEAR_SILU_4D_FP16)
{
const KuDNN::TensorInfo srcTensor = {{3, 2, 110, 20}, KuDNN::Element::TypeT::F16, KuDNN::Layout::ABCD};
const KuDNN::TensorInfo weightTensor = {{3, 2, 20, 200}, KuDNN::Element::TypeT::F16, KuDNN::Layout::ABCD};
const KuDNN::TensorInfo dstTensor = {{3, 2 , 110, 200}, KuDNN::Element::TypeT::F16, KuDNN::Layout::ABCD};
const KuDNN::TensorInfo biasTensor = {{3, 2, 110, 200}, KuDNN::Element::TypeT::F16, KuDNN::Layout::ABCD};
float alpha = 1.0f;
float beta = 0.0f;
KuDNN::ActivationFunction algKind = KuDNN::ActivationFunction::SWISH;
int numThreads = 0;
EXPECT_TRUE(Test_Template_Function<__fp16>(srcTensor, weightTensor, dstTensor, biasTensor, alpha, beta, algKind, numThreads));
}

TEST(LinearActivationTests, FWD_LINEAR_SILU_5D_FP16)
{
const KuDNN::TensorInfo srcTensor = {{4, 3, 2, 110, 20}, KuDNN::Element::TypeT::F16, KuDNN::Layout::ABCDE};
const KuDNN::TensorInfo weightTensor = {{4, 3, 2, 20, 200}, KuDNN::Element::TypeT::F16, KuDNN::Layout::ABCDE};
const KuDNN::TensorInfo dstTensor = {{4, 3, 2 , 110, 200}, KuDNN::Element::TypeT::F16, KuDNN::Layout::ABCDE};
const KuDNN::TensorInfo biasTensor = {{4, 3, 2, 110, 200}, KuDNN::Element::TypeT::F16, KuDNN::Layout::ABCDE};
float alpha = 1.0f;
float beta = 0.0f;
KuDNN::ActivationFunction algKind = KuDNN::ActivationFunction::SWISH;
int numThreads = 0;
EXPECT_TRUE(Test_Template_Function<__fp16>(srcTensor, weightTensor, dstTensor, biasTensor, alpha, beta, algKind, numThreads));
}

// test SILU BF16
TEST(LinearActivationTests, FWD_LINEAR_SILU_2D_BF16)
{
const KuDNN::TensorInfo srcTensor = {{110, 20}, KuDNN::Element::TypeT::BF16, KuDNN::Layout::AB};
const KuDNN::TensorInfo weightTensor = {{20, 200}, KuDNN::Element::TypeT::BF16, KuDNN::Layout::AB};
const KuDNN::TensorInfo dstTensor = {{110, 200}, KuDNN::Element::TypeT::BF16, KuDNN::Layout::AB};
const KuDNN::TensorInfo biasTensor = {{110, 200}, KuDNN::Element::TypeT::BF16, KuDNN::Layout::AB};
float alpha = 1.0f;
float beta = 0.0f;
KuDNN::ActivationFunction algKind = KuDNN::ActivationFunction::SWISH;
int numThreads = 0;
EXPECT_TRUE(Test_Template_Function<__bf16>(srcTensor, weightTensor, dstTensor, biasTensor, alpha, beta, algKind, numThreads));
}

TEST(LinearActivationTests, FWD_LINEAR_SILU_3D_BF16)
{
const KuDNN::TensorInfo srcTensor = {{2, 110, 20}, KuDNN::Element::TypeT::BF16, KuDNN::Layout::ABC};
const KuDNN::TensorInfo weightTensor = {{2, 20, 200}, KuDNN::Element::TypeT::BF16, KuDNN::Layout::ABC};
const KuDNN::TensorInfo dstTensor = {{2 , 110, 200}, KuDNN::Element::TypeT::BF16, KuDNN::Layout::ABC};
const KuDNN::TensorInfo biasTensor = {{2, 110, 200}, KuDNN::Element::TypeT::BF16, KuDNN::Layout::ABC};
float alpha = 1.0f;
float beta = 0.0f;
KuDNN::ActivationFunction algKind = KuDNN::ActivationFunction::SWISH;
int numThreads = 0;
EXPECT_TRUE(Test_Template_Function<__bf16>(srcTensor, weightTensor, dstTensor, biasTensor, alpha, beta, algKind, numThreads));
}

TEST(LinearActivationTests, FWD_LINEAR_SILU_4D_BF16)
{
const KuDNN::TensorInfo srcTensor = {{3, 2, 110, 20}, KuDNN::Element::TypeT::BF16, KuDNN::Layout::ABCD};
const KuDNN::TensorInfo weightTensor = {{3, 2, 20, 200}, KuDNN::Element::TypeT::BF16, KuDNN::Layout::ABCD};
const KuDNN::TensorInfo dstTensor = {{3, 2 , 110, 200}, KuDNN::Element::TypeT::BF16, KuDNN::Layout::ABCD};
const KuDNN::TensorInfo biasTensor = {{3, 2, 110, 200}, KuDNN::Element::TypeT::BF16, KuDNN::Layout::ABCD};
float alpha = 1.0f;
float beta = 0.0f;
KuDNN::ActivationFunction algKind = KuDNN::ActivationFunction::SWISH;
int numThreads = 0;
EXPECT_TRUE(Test_Template_Function<__bf16>(srcTensor, weightTensor, dstTensor, biasTensor, alpha, beta, algKind, numThreads));
}

TEST(LinearActivationTests, FWD_LINEAR_SILU_5D_BF16)
{
const KuDNN::TensorInfo srcTensor = {{4, 3, 2, 110, 20}, KuDNN::Element::TypeT::BF16, KuDNN::Layout::ABCDE};
const KuDNN::TensorInfo weightTensor = {{4, 3, 2, 20, 200}, KuDNN::Element::TypeT::BF16, KuDNN::Layout::ABCDE};
const KuDNN::TensorInfo dstTensor = {{4, 3, 2 , 110, 200}, KuDNN::Element::TypeT::BF16, KuDNN::Layout::ABCDE};
const KuDNN::TensorInfo biasTensor = {{4, 3, 2, 110, 200}, KuDNN::Element::TypeT::BF16, KuDNN::Layout::ABCDE};
float alpha = 1.0f;
float beta = 0.0f;
KuDNN::ActivationFunction algKind = KuDNN::ActivationFunction::SWISH;
int numThreads = 0;
EXPECT_TRUE(Test_Template_Function<__bf16>(srcTensor, weightTensor, dstTensor, biasTensor, alpha, beta, algKind, numThreads));
}

// test GELU_ERF FP32
TEST(LinearActivationTests, FWD_LINEAR_GELU_ERF_2D_FP32)
{
const KuDNN::TensorInfo srcTensor = {{110, 20}, KuDNN::Element::TypeT::F32, KuDNN::Layout::AB};
const KuDNN::TensorInfo weightTensor = {{20, 200}, KuDNN::Element::TypeT::F32, KuDNN::Layout::AB};
const KuDNN::TensorInfo dstTensor = {{110, 200}, KuDNN::Element::TypeT::F32, KuDNN::Layout::AB};
const KuDNN::TensorInfo biasTensor = {{110, 200}, KuDNN::Element::TypeT::F32, KuDNN::Layout::AB};
float alpha = 1.0f;
float beta = 0.0f;
KuDNN::ActivationFunction algKind = KuDNN::ActivationFunction::GELU_ERF;
int numThreads = 0;
EXPECT_TRUE(Test_Template_Function<float>(srcTensor, weightTensor, dstTensor, biasTensor, alpha, beta, algKind, numThreads));
}

TEST(LinearActivationTests, FWD_LINEAR_GELU_ERF_3D_FP32)
{
const KuDNN::TensorInfo srcTensor = {{2, 110, 20}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABC};
const KuDNN::TensorInfo weightTensor = {{2, 20, 200}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABC};
const KuDNN::TensorInfo dstTensor = {{2 , 110, 200}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABC};
const KuDNN::TensorInfo biasTensor = {{2, 110, 200}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABC};
float alpha = 1.0f;
float beta = 0.0f;
KuDNN::ActivationFunction algKind = KuDNN::ActivationFunction::GELU_ERF;
int numThreads = 0;
EXPECT_TRUE(Test_Template_Function<float>(srcTensor, weightTensor, dstTensor, biasTensor, alpha, beta, algKind, numThreads));
}

TEST(LinearActivationTests, FWD_LINEAR_GELU_ERF_4D_FP32)
{
const KuDNN::TensorInfo srcTensor = {{3, 2, 110, 20}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCD};
const KuDNN::TensorInfo weightTensor = {{3, 2, 20, 200}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCD};
const KuDNN::TensorInfo dstTensor = {{3, 2 , 110, 200}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCD};
const KuDNN::TensorInfo biasTensor = {{3, 2, 110, 200}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCD};
float alpha = 1.0f;
float beta = 0.0f;
KuDNN::ActivationFunction algKind = KuDNN::ActivationFunction::GELU_ERF;
int numThreads = 0;
EXPECT_TRUE(Test_Template_Function<float>(srcTensor, weightTensor, dstTensor, biasTensor, alpha, beta, algKind, numThreads));
}

TEST(LinearActivationTests, FWD_LINEAR_GELU_ERF_5D_FP32)
{
const KuDNN::TensorInfo srcTensor = {{4, 3, 2, 110, 20}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCDE};
const KuDNN::TensorInfo weightTensor = {{4, 3, 2, 20, 200}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCDE};
const KuDNN::TensorInfo dstTensor = {{4, 3, 2 , 110, 200}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCDE};
const KuDNN::TensorInfo biasTensor = {{4, 3, 2, 110, 200}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCDE};
float alpha = 1.0f;
float beta = 0.0f;
KuDNN::ActivationFunction algKind = KuDNN::ActivationFunction::GELU_ERF;
int numThreads = 0;
EXPECT_TRUE(Test_Template_Function<float>(srcTensor, weightTensor, dstTensor, biasTensor, alpha, beta, algKind, numThreads));
}

// test GELU_ERF FP16
TEST(LinearActivationTests, FWD_LINEAR_GELU_ERF_2D_FP16)
{
const KuDNN::TensorInfo srcTensor = {{110, 20}, KuDNN::Element::TypeT::F16, KuDNN::Layout::AB};
const KuDNN::TensorInfo weightTensor = {{20, 200}, KuDNN::Element::TypeT::F16, KuDNN::Layout::AB};
const KuDNN::TensorInfo dstTensor = {{110, 200}, KuDNN::Element::TypeT::F16, KuDNN::Layout::AB};
const KuDNN::TensorInfo biasTensor = {{110, 200}, KuDNN::Element::TypeT::F16, KuDNN::Layout::AB};
float alpha = 1.0f;
float beta = 0.0f;
KuDNN::ActivationFunction algKind = KuDNN::ActivationFunction::GELU_ERF;
int numThreads = 0;
EXPECT_TRUE(Test_Template_Function<__fp16>(srcTensor, weightTensor, dstTensor, biasTensor, alpha, beta, algKind, numThreads));
}

TEST(LinearActivationTests, FWD_LINEAR_GELU_ERF_3D_FP16)
{
const KuDNN::TensorInfo srcTensor = {{2, 110, 20}, KuDNN::Element::TypeT::F16, KuDNN::Layout::ABC};
const KuDNN::TensorInfo weightTensor = {{2, 20, 200}, KuDNN::Element::TypeT::F16, KuDNN::Layout::ABC};
const KuDNN::TensorInfo dstTensor = {{2 , 110, 200}, KuDNN::Element::TypeT::F16, KuDNN::Layout::ABC};
const KuDNN::TensorInfo biasTensor = {{2, 110, 200}, KuDNN::Element::TypeT::F16, KuDNN::Layout::ABC};
float alpha = 1.0f;
float beta = 0.0f;
KuDNN::ActivationFunction algKind = KuDNN::ActivationFunction::GELU_ERF;
int numThreads = 0;
EXPECT_TRUE(Test_Template_Function<__fp16>(srcTensor, weightTensor, dstTensor, biasTensor, alpha, beta, algKind, numThreads));
}

TEST(LinearActivationTests, FWD_LINEAR_GELU_ERF_4D_FP16)
{
const KuDNN::TensorInfo srcTensor = {{3, 2, 110, 20}, KuDNN::Element::TypeT::F16, KuDNN::Layout::ABCD};
const KuDNN::TensorInfo weightTensor = {{3, 2, 20, 200}, KuDNN::Element::TypeT::F16, KuDNN::Layout::ABCD};
const KuDNN::TensorInfo dstTensor = {{3, 2 , 110, 200}, KuDNN::Element::TypeT::F16, KuDNN::Layout::ABCD};
const KuDNN::TensorInfo biasTensor = {{3, 2, 110, 200}, KuDNN::Element::TypeT::F16, KuDNN::Layout::ABCD};
float alpha = 1.0f;
float beta = 0.0f;
KuDNN::ActivationFunction algKind = KuDNN::ActivationFunction::GELU_ERF;
int numThreads = 0;
EXPECT_TRUE(Test_Template_Function<__fp16>(srcTensor, weightTensor, dstTensor, biasTensor, alpha, beta, algKind, numThreads));
}

TEST(LinearActivationTests, FWD_LINEAR_GELU_ERF_5D_FP16)
{
const KuDNN::TensorInfo srcTensor = {{4, 3, 2, 110, 20}, KuDNN::Element::TypeT::F16, KuDNN::Layout::ABCDE};
const KuDNN::TensorInfo weightTensor = {{4, 3, 2, 20, 200}, KuDNN::Element::TypeT::F16, KuDNN::Layout::ABCDE};
const KuDNN::TensorInfo dstTensor = {{4, 3, 2 , 110, 200}, KuDNN::Element::TypeT::F16, KuDNN::Layout::ABCDE};
const KuDNN::TensorInfo biasTensor = {{4, 3, 2, 110, 200}, KuDNN::Element::TypeT::F16, KuDNN::Layout::ABCDE};
float alpha = 1.0f;
float beta = 0.0f;
KuDNN::ActivationFunction algKind = KuDNN::ActivationFunction::GELU_ERF;
int numThreads = 0;
EXPECT_TRUE(Test_Template_Function<__fp16>(srcTensor, weightTensor, dstTensor, biasTensor, alpha, beta, algKind, numThreads));
}

// test GELU_ERF BF16
TEST(LinearActivationTests, FWD_LINEAR_GELU_ERF_2D_BF16)
{
const KuDNN::TensorInfo srcTensor = {{110, 20}, KuDNN::Element::TypeT::BF16, KuDNN::Layout::AB};
const KuDNN::TensorInfo weightTensor = {{20, 200}, KuDNN::Element::TypeT::BF16, KuDNN::Layout::AB};
const KuDNN::TensorInfo dstTensor = {{110, 200}, KuDNN::Element::TypeT::BF16, KuDNN::Layout::AB};
const KuDNN::TensorInfo biasTensor = {{110, 200}, KuDNN::Element::TypeT::BF16, KuDNN::Layout::AB};
float alpha = 1.0f;
float beta = 0.0f;
KuDNN::ActivationFunction algKind = KuDNN::ActivationFunction::GELU_ERF;
int numThreads = 0;
EXPECT_TRUE(Test_Template_Function<__bf16>(srcTensor, weightTensor, dstTensor, biasTensor, alpha, beta, algKind, numThreads));
}

TEST(LinearActivationTests, FWD_LINEAR_GELU_ERF_3D_BF16)
{
const KuDNN::TensorInfo srcTensor = {{2, 110, 20}, KuDNN::Element::TypeT::BF16, KuDNN::Layout::ABC};
const KuDNN::TensorInfo weightTensor = {{2, 20, 200}, KuDNN::Element::TypeT::BF16, KuDNN::Layout::ABC};
const KuDNN::TensorInfo dstTensor = {{2 , 110, 200}, KuDNN::Element::TypeT::BF16, KuDNN::Layout::ABC};
const KuDNN::TensorInfo biasTensor = {{2, 110, 200}, KuDNN::Element::TypeT::BF16, KuDNN::Layout::ABC};
float alpha = 1.0f;
float beta = 0.0f;
KuDNN::ActivationFunction algKind = KuDNN::ActivationFunction::GELU_ERF;
int numThreads = 0;
EXPECT_TRUE(Test_Template_Function<__bf16>(srcTensor, weightTensor, dstTensor, biasTensor, alpha, beta, algKind, numThreads));
}

TEST(LinearActivationTests, FWD_LINEAR_GELU_ERF_4D_BF16)
{
const KuDNN::TensorInfo srcTensor = {{3, 2, 110, 20}, KuDNN::Element::TypeT::BF16, KuDNN::Layout::ABCD};
const KuDNN::TensorInfo weightTensor = {{3, 2, 20, 200}, KuDNN::Element::TypeT::BF16, KuDNN::Layout::ABCD};
const KuDNN::TensorInfo dstTensor = {{3, 2 , 110, 200}, KuDNN::Element::TypeT::BF16, KuDNN::Layout::ABCD};
const KuDNN::TensorInfo biasTensor = {{3, 2, 110, 200}, KuDNN::Element::TypeT::BF16, KuDNN::Layout::ABCD};
float alpha = 1.0f;
float beta = 0.0f;
KuDNN::ActivationFunction algKind = KuDNN::ActivationFunction::GELU_ERF;
int numThreads = 0;
EXPECT_TRUE(Test_Template_Function<__bf16>(srcTensor, weightTensor, dstTensor, biasTensor, alpha, beta, algKind, numThreads));
}

TEST(LinearActivationTests, FWD_LINEAR_GELU_ERF_5D_BF16)
{
const KuDNN::TensorInfo srcTensor = {{4, 3, 2, 110, 20}, KuDNN::Element::TypeT::BF16, KuDNN::Layout::ABCDE};
const KuDNN::TensorInfo weightTensor = {{4, 3, 2, 20, 200}, KuDNN::Element::TypeT::BF16, KuDNN::Layout::ABCDE};
const KuDNN::TensorInfo dstTensor = {{4, 3, 2 , 110, 200}, KuDNN::Element::TypeT::BF16, KuDNN::Layout::ABCDE};
const KuDNN::TensorInfo biasTensor = {{4, 3, 2, 110, 200}, KuDNN::Element::TypeT::BF16, KuDNN::Layout::ABCDE};
float alpha = 1.0f;
float beta = 0.0f;
KuDNN::ActivationFunction algKind = KuDNN::ActivationFunction::GELU_ERF;
int numThreads = 0;
EXPECT_TRUE(Test_Template_Function<__bf16>(srcTensor, weightTensor, dstTensor, biasTensor, alpha, beta, algKind, numThreads));
}

// test GELU_TANH FP32
TEST(LinearActivationTests, FWD_LINEAR_GELU_TANH_2D_FP32)
{
const KuDNN::TensorInfo srcTensor = {{110, 20}, KuDNN::Element::TypeT::F32, KuDNN::Layout::AB};
const KuDNN::TensorInfo weightTensor = {{20, 200}, KuDNN::Element::TypeT::F32, KuDNN::Layout::AB};
const KuDNN::TensorInfo dstTensor = {{110, 200}, KuDNN::Element::TypeT::F32, KuDNN::Layout::AB};
const KuDNN::TensorInfo biasTensor = {{110, 200}, KuDNN::Element::TypeT::F32, KuDNN::Layout::AB};
float alpha = 1.0f;
float beta = 0.0f;
KuDNN::ActivationFunction algKind = KuDNN::ActivationFunction::GELU_TANH;
int numThreads = 0;
EXPECT_TRUE(Test_Template_Function<float>(srcTensor, weightTensor, dstTensor, biasTensor, alpha, beta, algKind, numThreads));
}

TEST(LinearActivationTests, FWD_LINEAR_GELU_TANH_3D_FP32)
{
const KuDNN::TensorInfo srcTensor = {{2, 110, 20}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABC};
const KuDNN::TensorInfo weightTensor = {{2, 20, 200}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABC};
const KuDNN::TensorInfo dstTensor = {{2 , 110, 200}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABC};
const KuDNN::TensorInfo biasTensor = {{2, 110, 200}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABC};
float alpha = 1.0f;
float beta = 0.0f;
KuDNN::ActivationFunction algKind = KuDNN::ActivationFunction::GELU_TANH;
int numThreads = 0;
EXPECT_TRUE(Test_Template_Function<float>(srcTensor, weightTensor, dstTensor, biasTensor, alpha, beta, algKind, numThreads));
}

TEST(LinearActivationTests, FWD_LINEAR_GELU_TANH_4D_FP32)
{
const KuDNN::TensorInfo srcTensor = {{3, 2, 110, 20}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCD};
const KuDNN::TensorInfo weightTensor = {{3, 2, 20, 200}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCD};
const KuDNN::TensorInfo dstTensor = {{3, 2 , 110, 200}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCD};
const KuDNN::TensorInfo biasTensor = {{3, 2, 110, 200}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCD};
float alpha = 1.0f;
float beta = 0.0f;
KuDNN::ActivationFunction algKind = KuDNN::ActivationFunction::GELU_TANH;
int numThreads = 0;
EXPECT_TRUE(Test_Template_Function<float>(srcTensor, weightTensor, dstTensor, biasTensor, alpha, beta, algKind, numThreads));
}

TEST(LinearActivationTests, FWD_LINEAR_GELU_TANH_5D_FP32)
{
const KuDNN::TensorInfo srcTensor = {{4, 3, 2, 110, 20}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCDE};
const KuDNN::TensorInfo weightTensor = {{4, 3, 2, 20, 200}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCDE};
const KuDNN::TensorInfo dstTensor = {{4, 3, 2 , 110, 200}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCDE};
const KuDNN::TensorInfo biasTensor = {{4, 3, 2, 110, 200}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCDE};
float alpha = 1.0f;
float beta = 0.0f;
KuDNN::ActivationFunction algKind = KuDNN::ActivationFunction::GELU_TANH;
int numThreads = 0;
EXPECT_TRUE(Test_Template_Function<float>(srcTensor, weightTensor, dstTensor, biasTensor, alpha, beta, algKind, numThreads));
}

// test GELU_ERF FP16
TEST(LinearActivationTests, FWD_LINEAR_GELU_TANH_2D_FP16)
{
const KuDNN::TensorInfo srcTensor = {{110, 20}, KuDNN::Element::TypeT::F16, KuDNN::Layout::AB};
const KuDNN::TensorInfo weightTensor = {{20, 200}, KuDNN::Element::TypeT::F16, KuDNN::Layout::AB};
const KuDNN::TensorInfo dstTensor = {{110, 200}, KuDNN::Element::TypeT::F16, KuDNN::Layout::AB};
const KuDNN::TensorInfo biasTensor = {{110, 200}, KuDNN::Element::TypeT::F16, KuDNN::Layout::AB};
float alpha = 1.0f;
float beta = 0.0f;
KuDNN::ActivationFunction algKind = KuDNN::ActivationFunction::GELU_TANH;
int numThreads = 0;
EXPECT_TRUE(Test_Template_Function<__fp16>(srcTensor, weightTensor, dstTensor, biasTensor, alpha, beta, algKind, numThreads));
}

TEST(LinearActivationTests, FWD_LINEAR_GELU_TANH_3D_FP16)
{
const KuDNN::TensorInfo srcTensor = {{2, 110, 20}, KuDNN::Element::TypeT::F16, KuDNN::Layout::ABC};
const KuDNN::TensorInfo weightTensor = {{2, 20, 200}, KuDNN::Element::TypeT::F16, KuDNN::Layout::ABC};
const KuDNN::TensorInfo dstTensor = {{2 , 110, 200}, KuDNN::Element::TypeT::F16, KuDNN::Layout::ABC};
const KuDNN::TensorInfo biasTensor = {{2, 110, 200}, KuDNN::Element::TypeT::F16, KuDNN::Layout::ABC};
float alpha = 1.0f;
float beta = 0.0f;
KuDNN::ActivationFunction algKind = KuDNN::ActivationFunction::GELU_TANH;
int numThreads = 0;
EXPECT_TRUE(Test_Template_Function<__fp16>(srcTensor, weightTensor, dstTensor, biasTensor, alpha, beta, algKind, numThreads));
}

TEST(LinearActivationTests, FWD_LINEAR_GELU_TANH_4D_FP16)
{
const KuDNN::TensorInfo srcTensor = {{3, 2, 110, 20}, KuDNN::Element::TypeT::F16, KuDNN::Layout::ABCD};
const KuDNN::TensorInfo weightTensor = {{3, 2, 20, 200}, KuDNN::Element::TypeT::F16, KuDNN::Layout::ABCD};
const KuDNN::TensorInfo dstTensor = {{3, 2 , 110, 200}, KuDNN::Element::TypeT::F16, KuDNN::Layout::ABCD};
const KuDNN::TensorInfo biasTensor = {{3, 2, 110, 200}, KuDNN::Element::TypeT::F16, KuDNN::Layout::ABCD};
float alpha = 1.0f;
float beta = 0.0f;
KuDNN::ActivationFunction algKind = KuDNN::ActivationFunction::GELU_TANH;
int numThreads = 0;
EXPECT_TRUE(Test_Template_Function<__fp16>(srcTensor, weightTensor, dstTensor, biasTensor, alpha, beta, algKind, numThreads));
}

TEST(LinearActivationTests, FWD_LINEAR_GELU_TANH_5D_FP16)
{
const KuDNN::TensorInfo srcTensor = {{4, 3, 2, 110, 20}, KuDNN::Element::TypeT::F16, KuDNN::Layout::ABCDE};
const KuDNN::TensorInfo weightTensor = {{4, 3, 2, 20, 200}, KuDNN::Element::TypeT::F16, KuDNN::Layout::ABCDE};
const KuDNN::TensorInfo dstTensor = {{4, 3, 2 , 110, 200}, KuDNN::Element::TypeT::F16, KuDNN::Layout::ABCDE};
const KuDNN::TensorInfo biasTensor = {{4, 3, 2, 110, 200}, KuDNN::Element::TypeT::F16, KuDNN::Layout::ABCDE};
float alpha = 1.0f;
float beta = 0.0f;
KuDNN::ActivationFunction algKind = KuDNN::ActivationFunction::GELU_TANH;
int numThreads = 0;
EXPECT_TRUE(Test_Template_Function<__fp16>(srcTensor, weightTensor, dstTensor, biasTensor, alpha, beta, algKind, numThreads));
}

// test GELU_ERF BF16
TEST(LinearActivationTests, FWD_LINEAR_GELU_TANH_2D_BF16)
{
const KuDNN::TensorInfo srcTensor = {{110, 20}, KuDNN::Element::TypeT::BF16, KuDNN::Layout::AB};
const KuDNN::TensorInfo weightTensor = {{20, 200}, KuDNN::Element::TypeT::BF16, KuDNN::Layout::AB};
const KuDNN::TensorInfo dstTensor = {{110, 200}, KuDNN::Element::TypeT::BF16, KuDNN::Layout::AB};
const KuDNN::TensorInfo biasTensor = {{110, 200}, KuDNN::Element::TypeT::BF16, KuDNN::Layout::AB};
float alpha = 1.0f;
float beta = 0.0f;
KuDNN::ActivationFunction algKind = KuDNN::ActivationFunction::GELU_TANH;
int numThreads = 0;
EXPECT_TRUE(Test_Template_Function<__bf16>(srcTensor, weightTensor, dstTensor, biasTensor, alpha, beta, algKind, numThreads));
}

TEST(LinearActivationTests, FWD_LINEAR_GELU_TANH_3D_BF16)
{
const KuDNN::TensorInfo srcTensor = {{2, 110, 20}, KuDNN::Element::TypeT::BF16, KuDNN::Layout::ABC};
const KuDNN::TensorInfo weightTensor = {{2, 20, 200}, KuDNN::Element::TypeT::BF16, KuDNN::Layout::ABC};
const KuDNN::TensorInfo dstTensor = {{2 , 110, 200}, KuDNN::Element::TypeT::BF16, KuDNN::Layout::ABC};
const KuDNN::TensorInfo biasTensor = {{2, 110, 200}, KuDNN::Element::TypeT::BF16, KuDNN::Layout::ABC};
float alpha = 1.0f;
float beta = 0.0f;
KuDNN::ActivationFunction algKind = KuDNN::ActivationFunction::GELU_TANH;
int numThreads = 0;
EXPECT_TRUE(Test_Template_Function<__bf16>(srcTensor, weightTensor, dstTensor, biasTensor, alpha, beta, algKind, numThreads));
}

TEST(LinearActivationTests, FWD_LINEAR_GELU_TANH_4D_BF16)
{
const KuDNN::TensorInfo srcTensor = {{3, 2, 110, 20}, KuDNN::Element::TypeT::BF16, KuDNN::Layout::ABCD};
const KuDNN::TensorInfo weightTensor = {{3, 2, 20, 200}, KuDNN::Element::TypeT::BF16, KuDNN::Layout::ABCD};
const KuDNN::TensorInfo dstTensor = {{3, 2 , 110, 200}, KuDNN::Element::TypeT::BF16, KuDNN::Layout::ABCD};
const KuDNN::TensorInfo biasTensor = {{3, 2, 110, 200}, KuDNN::Element::TypeT::BF16, KuDNN::Layout::ABCD};
float alpha = 1.0f;
float beta = 0.0f;
KuDNN::ActivationFunction algKind = KuDNN::ActivationFunction::GELU_TANH;
int numThreads = 0;
EXPECT_TRUE(Test_Template_Function<__bf16>(srcTensor, weightTensor, dstTensor, biasTensor, alpha, beta, algKind, numThreads));
}

TEST(LinearActivationTests, FWD_LINEAR_GELU_TANH_5D_BF16)
{
const KuDNN::TensorInfo srcTensor = {{4, 3, 2, 110, 20}, KuDNN::Element::TypeT::BF16, KuDNN::Layout::ABCDE};
const KuDNN::TensorInfo weightTensor = {{4, 3, 2, 20, 200}, KuDNN::Element::TypeT::BF16, KuDNN::Layout::ABCDE};
const KuDNN::TensorInfo dstTensor = {{4, 3, 2 , 110, 200}, KuDNN::Element::TypeT::BF16, KuDNN::Layout::ABCDE};
const KuDNN::TensorInfo biasTensor = {{4, 3, 2, 110, 200}, KuDNN::Element::TypeT::BF16, KuDNN::Layout::ABCDE};
float alpha = 1.0f;
float beta = 0.0f;
KuDNN::ActivationFunction algKind = KuDNN::ActivationFunction::GELU_TANH;
int numThreads = 0;
EXPECT_TRUE(Test_Template_Function<__bf16>(srcTensor, weightTensor, dstTensor, biasTensor, alpha, beta, algKind, numThreads));
}
