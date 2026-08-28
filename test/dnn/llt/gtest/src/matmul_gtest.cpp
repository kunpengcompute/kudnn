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

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <functional>
#include <iostream>
#include <limits.h>
#include <map>
#include <memory>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "gtest/gtest.h"
#include "kudnn.hpp"
#include "kudnn_tensor_data.hpp"

using SizeType = KuDNN::SizeType;
using Shape = KuDNN::Shape;

static void NoPackCopyConstructor1()
{
    const KuDNN::TensorInfo matATensor = {{1, 1, 1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCDE};
    const KuDNN::TensorInfo matBTensor = {{1, 1, 1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCDE};
    const KuDNN::TensorInfo matCTensor = {{1, 1, 1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCDE};
    const KuDNN::TensorInfo biasTensor = {{1, 1, 1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCDE};
    KuDNN::Gemm gemmLayer1(matATensor, matBTensor, matCTensor, biasTensor);
    KuDNN::Gemm gemmLayer2(gemmLayer1);
    for (size_t i = 0; i < g_Tensor_data.size() / 4; i += 4) {
        KuDNN::Gemm gemmLayer1(g_Tensor_data[i], g_Tensor_data[i + 1], g_Tensor_data[i + 2], g_Tensor_data[i + 3]);
        KuDNN::Gemm gemmLayer2(gemmLayer1);
    }
}

static void NoPackCopyConstructor2()
{
    const KuDNN::TensorInfo matATensor = {{1, 1, 1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCDE};
    const KuDNN::TensorInfo matBTensor = {{1, 1, 1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCDE};
    const KuDNN::TensorInfo matCTensor = {{1, 1, 1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCDE};
    const KuDNN::TensorInfo biasTensor = {{1, 1, 1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCDE};
    KuDNN::Gemm gemmLayer1(matATensor, matBTensor, matCTensor, biasTensor);
    KuDNN::Gemm gemmLayer2(gemmLayer1);
    gemmLayer2 = gemmLayer1;
    for (size_t i = 0; i < g_Tensor_data.size() / 4; i += 4) {
        KuDNN::Gemm gemmLayer1(g_Tensor_data[i], g_Tensor_data[i + 1], g_Tensor_data[i + 2], g_Tensor_data[i + 3]);
        KuDNN::Gemm gemmLayer2(gemmLayer1);
        gemmLayer2 = gemmLayer1;
    }
}

static void NoPackMoveConstructor1()
{
    const KuDNN::TensorInfo matATensor = {{1, 1, 1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCDE};
    const KuDNN::TensorInfo matBTensor = {{1, 1, 1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCDE};
    const KuDNN::TensorInfo matCTensor = {{1, 1, 1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCDE};
    const KuDNN::TensorInfo biasTensor = {{1, 1, 1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCDE};
    KuDNN::Gemm gemmLayer1(matATensor, matBTensor, matCTensor, biasTensor);
    KuDNN::Gemm gemmLayer2(std::move(gemmLayer1));
    for (size_t i = 0; i < g_Tensor_data.size() / 4; i += 4) {
        KuDNN::Gemm gemmLayer1(g_Tensor_data[i], g_Tensor_data[i + 1], g_Tensor_data[i + 2], g_Tensor_data[i + 3]);
        KuDNN::Gemm gemmLayer2(std::move(gemmLayer1));
    }
}

static void NoPackMoveConstructor2()
{
    const KuDNN::TensorInfo matATensor = {{1, 1, 1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCDE};
    const KuDNN::TensorInfo matBTensor = {{1, 1, 1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCDE};
    const KuDNN::TensorInfo matCTensor = {{1, 1, 1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCDE};
    const KuDNN::TensorInfo biasTensor = {{1, 1, 1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCDE};
    KuDNN::Gemm gemmLayer1(matATensor, matBTensor, matCTensor, biasTensor);
    KuDNN::Gemm gemmLayer2(gemmLayer1);
    gemmLayer2 = std::move(gemmLayer1);
    for (size_t i = 0; i < g_Tensor_data.size() / 4; i += 4) {
        KuDNN::Gemm gemmLayer1(g_Tensor_data[i], g_Tensor_data[i + 1], g_Tensor_data[i + 2], g_Tensor_data[i + 3]);
        KuDNN::Gemm gemmLayer2(gemmLayer1);
        gemmLayer2 = std::move(gemmLayer1);
    }
}

namespace utils {

template <typename T1, typename T2, typename T3>
static void MatmulRef(const KuDNN::TensorInfo& srcTensor, const KuDNN::TensorInfo& weiTensor, T1* A, T2* B, T3* C)
{
    Shape srcShape = srcTensor.GetDims();
    Shape weiShape = weiTensor.GetDims();
    KuDNN::Layout srcLayout = srcTensor.GetLayout();
    KuDNN::Layout weiLayout = weiTensor.GetLayout();

    SizeType numDims = srcShape.GetNumDims();
    SizeType M = srcShape[numDims - 2], N = weiShape[numDims - 1], K = srcShape[numDims - 1];
    SizeType batch = srcShape.GetTotalDimsSize() / M / K;

#pragma omp parallel for collapse(3)
    for (SizeType b = 0; b < batch; b++) {
        for (SizeType i = 0; i < M; i++) {
            for (SizeType j = 0; j < N; j++) {
                C[b * M * N + i * N + j] = 0.0f;
                for (SizeType k = 0; k < K; k++) {
                    if (srcLayout == KuDNN::Layout::ABCDE && weiLayout == KuDNN::Layout::ABCDE) {
                        C[b * M * N + i * N + j] +=
                            static_cast<T3>(A[b * M * K + i * K + k]) * static_cast<T3>(B[b * N * K + k * N + j]);
                    } else if (srcLayout == KuDNN::Layout::ABCDE && weiLayout == KuDNN::Layout::ABCED) {
                        C[b * M * N + i * N + j] +=
                            static_cast<T3>(A[b * M * K + i * K + k]) * static_cast<T3>(B[b * N * K + j * K + k]);
                    } else if (srcLayout == KuDNN::Layout::ABCED && weiLayout == KuDNN::Layout::ABCDE) {
                        C[b * M * N + i * N + j] +=
                            static_cast<T3>(A[b * M * K + k * M + i]) * static_cast<T3>(B[b * N * K + k * N + j]);
                    } else {
                        C[b * M * N + i * N + j] +=
                            static_cast<T3>(A[b * M * K + k * M + i]) * static_cast<T3>(B[b * N * K + j * K + k]);
                    }
                }
            }
        }
    }
}

template <typename T1, typename T2>
void matrixAdd(T1* A, T2* B, size_t size)
{
    for (size_t i = 0; i < size; i++) {
        *(A + i) += static_cast<T1>(*(B + i));
    }
}

template <typename T>
std::function<T()> generateRandomNumber()
{
    std::default_random_engine e(time(NULL));
    if (KuDNN::Element::MatchType<T>() == KuDNN::Element::TypeT::S8) {
        std::uniform_int_distribution<int> u(-125, 125);
        return [e, u]() mutable {
            return u(e);
        };
    } else if (KuDNN::Element::MatchType<T>() == KuDNN::Element::TypeT::S32) {
        std::uniform_int_distribution<int> u(-65536, 65536);
        return [e, u]() mutable {
            return u(e);
        };
    } else {
        std::uniform_real_distribution<float> u(-1, 1);
        return [e, u]() mutable {
            return u(e);
        };
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
        case KuDNN::Element::TypeT::S8:
            e = diff;
            eps = K / 350 + 1;
            pairEps = {e, eps};
            break;
        case KuDNN::Element::TypeT::U8:
            e = diff;
            eps = K / 700 + 1;
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

double getPerformance(Shape src, Shape wei, double duration)
{
    double computeUtilization =
        (2.0f * src[0] * src[1] * src[2] * src[3] * wei[3] * wei[4]) / duration / 1e6;  // GFLOPS
    return computeUtilization;
}

// DataWriter class for writing data to a file
class DataWriter {
public:
    explicit DataWriter(const std::string& filePath) : filePath(filePath)
    {
    }

    void WriteData(const std::string& data)
    {
        std::ofstream file(filePath, std::ios::out | std::ios::app);
        if (file.is_open()) {
            file << data;
            file.close();
        } else {
            std::cerr << "Failed to open file: " << filePath << std::endl;
        }
    }

    std::string MakeLine(Shape src, Shape wei, double perf, double duration)
    {
        std::string line = "D1: " + std::to_string(src[0]) + ", D2: " + std::to_string(src[1]) +
                           ", D3: " + std::to_string(src[2]) + ", M: " + std::to_string(src[3]) +
                           ", N: " + std::to_string(wei[4]) + ", K: " + std::to_string(wei[3]);
        line.append(", perf: " + std::to_string(perf) + "GFLOPS, Time:" + std::to_string(duration) + "ms");
        return line;
    }

    void SetFilePath(const std::string& fp)
    {
        this->filePath = fp;
    }

private:
    std::string filePath;  // File path
};

};  // namespace utils

static utils::DataWriter g_writer("./matmul_correctness_test_result.txt");

template <typename srcType, typename weiType, typename dstType, typename biasType>
void Matmul(Shape& srcShape, Shape& weiShape, Shape& dstShape, Shape& biasShape, int transSrc = 0, int transWei = 0)
{
    KuDNN::Layout srcLayout = KuDNN::Layout::ABCDE;
    KuDNN::Layout weiLayout = KuDNN::Layout::ABCDE;
    if (transSrc != 0) {
        srcLayout = KuDNN::Layout::ABCED;
    }
    if (transWei != 0) {
        weiLayout = KuDNN::Layout::ABCED;
    }
    const KuDNN::TensorInfo srcTensor = {srcShape, KuDNN::Element::MatchType<srcType>(), srcLayout};
    const KuDNN::TensorInfo weiTensor = {weiShape, KuDNN::Element::MatchType<weiType>(), weiLayout};
    const KuDNN::TensorInfo dstTensor = {dstShape, KuDNN::Element::MatchType<dstType>(), KuDNN::Layout::ABCDE};
    const KuDNN::TensorInfo biasTensor = {biasShape, KuDNN::Element::MatchType<biasType>(), KuDNN::Layout::ABCDE};

    SizeType srcSize = srcShape[0] * srcShape[1] * srcShape[2] * srcShape[3] * srcShape[4];
    SizeType dstSize = dstShape[0] * dstShape[1] * dstShape[2] * dstShape[3] * dstShape[4];
    SizeType weiSize = weiShape[0] * weiShape[1] * weiShape[2] * weiShape[3] * weiShape[4];
    SizeType biaSize = biasShape[0] * biasShape[1] * biasShape[2] * biasShape[3] * biasShape[4];

    void* src = malloc(srcSize * sizeof(srcType));
    void* wei = malloc(weiSize * sizeof(weiType));
    void* dst = malloc(dstSize * sizeof(dstType));
    void* bias = malloc(biaSize * sizeof(biasType));
    void* ref = malloc(dstSize * sizeof(dstType));
    if (src == nullptr || wei == nullptr || dst == nullptr || bias == nullptr || ref == nullptr) {
        throw std::runtime_error("malloc failed");
    }

    srcType* src_Type = static_cast<srcType*>(src);
    weiType* wei_Type = static_cast<weiType*>(wei);
    dstType* dst_Type = static_cast<dstType*>(dst);
    biasType* bias_Type = static_cast<biasType*>(bias);
    dstType* ref_Type = static_cast<dstType*>(ref);

    // initial randomnum generator
    auto srcRandomGenerator = utils::generateRandomNumber<srcType>();
    auto weiRandomGenerator = utils::generateRandomNumber<weiType>();
    auto biasRandomGenerator = utils::generateRandomNumber<biasType>();

// initial matA、matB、matC、matBias
#pragma omp parallel for
    for (SizeType i = 0; i < srcSize; ++i) {
        *(src_Type + i) = (srcType)srcRandomGenerator();
    }
#pragma omp parallel for
    for (SizeType i = 0; i < weiSize; ++i) {
        *(wei_Type + i) = (weiType)weiRandomGenerator();
    }
#pragma omp parallel for
    for (SizeType i = 0; i < dstSize; ++i) {
        if constexpr (std::is_same_v<dstType, __bf16>) {
            *(dst_Type + i) = static_cast<__bf16>(0.0f);
        } else {
            *(dst_Type + i) = 0;
        }
    }
#pragma omp parallel for
    for (SizeType i = 0; i < biaSize; ++i) {
        *(bias_Type + i) = (biasType)biasRandomGenerator();
    }
#pragma omp parallel for
    for (SizeType i = 0; i < dstSize; ++i) {
        if constexpr (std::is_same_v<dstType, __bf16>) {
            *(ref_Type + i) = static_cast<__bf16>(0.0f);
        } else {
            *(ref_Type + i) = 0;
        }
    }
    decltype(std::chrono::high_resolution_clock::now()) start;
    decltype(std::chrono::high_resolution_clock::now()) end;

    try {
        KuDNN::Gemm gemmLayer(srcTensor, weiTensor, dstTensor, biasTensor);
        start = std::chrono::high_resolution_clock::now();
        gemmLayer.Run(src, wei, dst, bias, 1, 0);
        end = std::chrono::high_resolution_clock::now();
    } catch (...) {
        free(src);
        free(wei);
        free(dst);
        free(bias);
        free(ref);
        throw;
    }

    float dstMax = 0;
    for (SizeType i = 0; i < dstSize; ++i) {
        dstType tmp = *(dst_Type + i);
        if constexpr (std::is_same_v<dstType, __bf16>) {
            dstMax = std::max(dstMax, std::fabs(static_cast<float>(tmp)));
        } else {
            dstMax = std::max(dstMax, std::fabs((float)tmp));
        }
    }

    // difftime
    double duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    // recode dims、perf、tCP、time
    double perf = utils::getPerformance(srcShape, weiShape, duration);
    g_writer.WriteData(g_writer.MakeLine(srcShape, weiShape, perf, duration) + "\n");
    // compute preference
    utils::MatmulRef(srcTensor, weiTensor, src_Type, wei_Type, ref_Type);
    utils::matrixAdd(ref_Type, bias_Type, dstSize);

    for (SizeType i = 0; i < dstSize; ++i) {
        float ref, dst;
        if constexpr (std::is_same_v<dstType, __bf16>) {
            ref = static_cast<float>(*(ref_Type + i));
            dst = static_cast<float>(*(dst_Type + i));
        } else {
            ref = *(ref_Type + i);
            dst = *(dst_Type + i);
        }
        float diff = fabs(ref - dst);
        auto pairEps = utils::compareEle(dstMax, diff, KuDNN::Element::MatchType<srcType>(), srcShape[4]);
        ASSERT_NEAR(std::fabs(pairEps.first), 0.0, pairEps.second);
    }

    if (src != nullptr) {
        free(src);
    }
    if (wei != nullptr) {
        free(wei);
    }
    if (dst != nullptr) {
        free(dst);
    }
    if (bias != nullptr) {
        free(bias);
    }
    if (ref != nullptr) {
        free(ref);
    }
}

class FunctionBase {
public:
    virtual ~FunctionBase() = default;
    virtual void call(Shape& srcShape, Shape& weiShape, Shape& dstShape, Shape& biasShape, int transSrc = 0,
                      int transWei = 0) const = 0;
};

template <typename srcType, typename weiType, typename dstType, typename biasType>
class MatmulCaller : public FunctionBase {
public:
    void call(Shape& srcShape, Shape& weiShape, Shape& dstShape, Shape& biasShape, int transSrc = 0,
              int transWei = 0) const override
    {
        Matmul<srcType, weiType, dstType, biasType>(srcShape, weiShape, dstShape, biasShape, transSrc, transWei);
    }
};

static std::vector<std::shared_ptr<FunctionBase>> g_matmulCallers = {
    std::make_shared<MatmulCaller<__fp16, __fp16, __fp16, __fp16>>(),
    std::make_shared<MatmulCaller<__fp16, __fp16, float, float>>(),
    std::make_shared<MatmulCaller<std::int8_t, std::int8_t, std::int32_t, std::int32_t>>(),
    std::make_shared<MatmulCaller<__fp16, std::int8_t, __fp16, float>>(),
    std::make_shared<MatmulCaller<__bf16, __bf16, __bf16, float>>(),
    std::make_shared<MatmulCaller<__bf16, __bf16, float, float>>(),
    std::make_shared<MatmulCaller<float, float, float, float>>(),
    std::make_shared<MatmulCaller<std::uint8_t, std::int8_t, std::int32_t, std::int32_t>>(),
    std::make_shared<MatmulCaller<std::int8_t, std::uint8_t, std::int32_t, std::int32_t>>(),
    std::make_shared<MatmulCaller<std::uint8_t, std::uint8_t, std::int32_t, std::int32_t>>(),
    std::make_shared<MatmulCaller<__fp16, std::uint8_t, __fp16, float>>(),
};

static std::map<int, std::string> g_nameMap = {
    {0, "FP16_FP16_FP16_FP16"},     {1, "FP16_FP16_FP32_FP32"},    {2, "INT8_INT8_INT32_INT32"},
    {3, "FP16_INT8_FP16_FP32"},     {4, "BF16_BF16_BF16_FP32"},    {5, "BF16_BF16_FP32_FP32"},
    {6, "FP32_FP32_FP32_FP32"},     {7, "UINT8_INT8_INT32_INT32"}, {8, "INT8_UINT8_INT32_INT32"},
    {9, "UINT8_UINT8_INT32_INT32"}, {10, "FP16_UINT8_FP16_FP32"},
};

// zero dims in shapes -> totalSize == 0 -> Gemm ValidateInput returns UNSUPPORTED
static void InputParamExcept_FP16()
{
    int D1 = 0, D2 = 1, D3 = 1, M = 5, N = 5, K = 5;
    Shape srcShape(D1, D2, D3, M, K), weiShape(D1, D2, D3, K, N), dstShape(D1, D2, D3, M, N),
        biasShape(D1, D2, D3, M, N);
    Matmul<__fp16, __fp16, __fp16, __fp16>(srcShape, weiShape, dstShape, biasShape);
}

static void InputParamExcept_FP16FP32()
{
    int D1 = 1, D2 = 0, D3 = 1, M = 5, N = 5, K = 5;
    Shape srcShape(D1, D2, D3, M, K), weiShape(D1, D2, D3, K, N), dstShape(D1, D2, D3, M, N),
        biasShape(D1, D2, D3, M, N);
    Matmul<__fp16, __fp16, float, float>(srcShape, weiShape, dstShape, biasShape);
}

static void InputParamExcept_W8A8()
{
    int D1 = 1, D2 = 1, D3 = 0, M = 5, N = 5, K = 5;
    Shape srcShape(D1, D2, D3, M, K), weiShape(D1, D2, D3, K, N), dstShape(D1, D2, D3, M, N),
        biasShape(D1, D2, D3, M, N);
    Matmul<std::int8_t, std::int8_t, std::int32_t, std::int32_t>(srcShape, weiShape, dstShape, biasShape);
}

static void InputParamExcept_W8A16()
{
    int D1 = 1, D2 = 1, D3 = 1, M = 0, N = 5, K = 5;
    Shape srcShape(D1, D2, D3, M, K), weiShape(D1, D2, D3, K, N), dstShape(D1, D2, D3, M, N),
        biasShape(D1, D2, D3, M, N);
    Matmul<__fp16, std::int8_t, __fp16, float>(srcShape, weiShape, dstShape, biasShape);
}

static void InputParamExcept_BF16()
{
    int D1 = 1, D2 = 1, D3 = 1, M = 5, N = 5, K = 5;
    Shape srcShape(D1, D2, D3, M, K), weiShape(D1, D2, D3, K, N), dstShape(D1, D2, D3, M, 0),
        biasShape(D1, D2, D3, M, 0);
    Matmul<__bf16, __bf16, __bf16, float>(srcShape, weiShape, dstShape, biasShape);
}

static void InputParamExcept_BF16FP32()
{
    int D1 = 1, D2 = 0, D3 = 1, M = 0, N = 5, K = 5;
    Shape srcShape(D1, D2, D3, M, K), weiShape(D1, D2, D3, K, N), dstShape(D1, D2, D3, M, N),
        biasShape(D1, D2, D3, M, N);
    Matmul<__bf16, __bf16, float, float>(srcShape, weiShape, dstShape, biasShape);
}

// test cases
TEST(MatmulGTests, NoPackCopyConstructor1)
{
    EXPECT_NO_THROW(NoPackCopyConstructor1());
}

TEST(MatmulGTests, NoPackCopyConstructor2)
{
    EXPECT_NO_THROW(NoPackCopyConstructor2());
}

TEST(MatmulGTests, NoPackMoveConstructor1)
{
    EXPECT_NO_THROW(NoPackMoveConstructor1());
}

TEST(MatmulGTests, NoPackMoveConstructor2)
{
    EXPECT_NO_THROW(NoPackMoveConstructor2());
}

TEST(MatmulGTests, InputParamExcept_FP16)
{
    EXPECT_THROW(InputParamExcept_FP16(), KuDNN::Service::Unsupported);
}

TEST(MatmulGTests, InputParamExcept_FP16FP32)
{
    EXPECT_THROW(InputParamExcept_FP16FP32(), KuDNN::Service::Unsupported);
}

TEST(MatmulGTests, InputParamExcept_W8A8)
{
    EXPECT_THROW(InputParamExcept_W8A8(), KuDNN::Service::Unsupported);
}

TEST(MatmulGTests, InputParamExcept_W8A16)
{
    EXPECT_THROW(InputParamExcept_W8A16(), KuDNN::Service::Unsupported);
}

TEST(MatmulGTests, InputParamExcept_BF16)
{
    EXPECT_THROW(InputParamExcept_BF16(), KuDNN::Service::Unsupported);
}

TEST(MatmulGTests, InputParamExcept_BF16FP32)
{
    EXPECT_THROW(InputParamExcept_BF16FP32(), KuDNN::Service::Unsupported);
}

// M N K
static std::vector<std::vector<int>> common_MNK = {{5, 5, 5}, {20, 20, 10}, {200, 100, 200}};
static std::vector<std::vector<int>> common_Dims = {{1, 1, 1},   {1, 1, 10},  {1, 1, 20}, {1, 10, 10},
                                                    {1, 10, 20}, {5, 10, 10}, {5, 10, 20}};
static std::vector<std::vector<int>> freParas = {{1, 1, 1, 4096, 3072, 5120}, {1, 1, 1, 4096, 32768, 512}};

static std::vector<int> GenerateNumParas(int n)
{
    std::vector<int> paras;
    for (int i = 0; i < n; ++i) {
        paras.push_back(i);
    }
    return paras;
}

static std::vector<int> numParas = GenerateNumParas(common_MNK.size() * common_Dims.size() + freParas.size());
static std::vector<int> numThreads = {0};
static std::vector<int> g_matmulCallersIndex = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
static std::vector<std::vector<int>> testParas;

using TestParas = std::tuple<int, int, int>;
class MTest : public ::testing::TestWithParam<TestParas> {
protected:
    static void SetUpTestSuite()
    {
        for (SizeType i = 0; i < common_MNK.size(); ++i) {
            for (SizeType j = 0; j < common_Dims.size(); ++j) {
                std::vector<int> combined;
                combined.insert(combined.end(), common_Dims[j].begin(), common_Dims[j].end());
                combined.insert(combined.end(), common_MNK[i].begin(), common_MNK[i].end());
                testParas.emplace_back(combined);
            }
        }
        testParas.insert(testParas.end(), freParas.begin(), freParas.end());
    }
};

static std::string ParamName(const ::testing::TestParamInfo<TestParas>& info)
{
    std::stringstream ss;
    int index = std::get<2>(info.param);
    int numThreads = std::get<1>(info.param);
    ss << g_nameMap[std::get<0>(info.param) % g_nameMap.size()];
    ss << "_index";
    ss << std::to_string(index);
    ss << "_numthreads";
    ss << std::to_string(numThreads);
    return ss.str();
}

INSTANTIATE_TEST_SUITE_P(MTests, MTest,
                         ::testing::Combine(::testing::ValuesIn(g_matmulCallersIndex), ::testing::ValuesIn(numThreads),
                                            ::testing::ValuesIn(numParas)),
                         ParamName);

TEST_P(MTest, Matmul)
{
    size_t index = std::get<2>(GetParam());
    int matmulCallerIndex = std::get<0>(GetParam());

    if (index < testParas.size()) {
        int D1 = testParas[index][0], D2 = testParas[index][1], D3 = testParas[index][2], M = testParas[index][3],
            N = testParas[index][4], K = testParas[index][5];
        Shape srcShape(D1, D2, D3, M, K), weiShape(D1, D2, D3, K, N), dstShape(D1, D2, D3, M, N),
            biasShape(D1, D2, D3, M, N);
        g_matmulCallers[matmulCallerIndex]->call(srcShape, weiShape, dstShape, biasShape);
    }
}

TEST_P(MTest, MatmulTransSrc)
{
    size_t index = std::get<2>(GetParam());
    int matmulCallerIndex = std::get<0>(GetParam());

    if (index < testParas.size()) {
        int D1 = testParas[index][0], D2 = testParas[index][1], D3 = testParas[index][2], M = testParas[index][3],
            N = testParas[index][4], K = testParas[index][5];
        Shape srcShape(D1, D2, D3, M, K), weiShape(D1, D2, D3, K, N), dstShape(D1, D2, D3, M, N),
            biasShape(D1, D2, D3, M, N);
        g_matmulCallers[matmulCallerIndex]->call(srcShape, weiShape, dstShape, biasShape, 1, 0);
    }
}

TEST_P(MTest, MatmulTransWei)
{
    size_t index = std::get<2>(GetParam());
    int matmulCallerIndex = std::get<0>(GetParam());

    if (index < testParas.size()) {
        int D1 = testParas[index][0], D2 = testParas[index][1], D3 = testParas[index][2], M = testParas[index][3],
            N = testParas[index][4], K = testParas[index][5];
        Shape srcShape(D1, D2, D3, M, K), weiShape(D1, D2, D3, K, N), dstShape(D1, D2, D3, M, N),
            biasShape(D1, D2, D3, M, N);
        g_matmulCallers[matmulCallerIndex]->call(srcShape, weiShape, dstShape, biasShape, 0, 1);
    }
}

TEST_P(MTest, MatmulTransSrcWei)
{
    size_t index = std::get<2>(GetParam());
    int matmulCallerIndex = std::get<0>(GetParam());

    if (index < testParas.size()) {
        int D1 = testParas[index][0], D2 = testParas[index][1], D3 = testParas[index][2], M = testParas[index][3],
            N = testParas[index][4], K = testParas[index][5];
        Shape srcShape(D1, D2, D3, M, K), weiShape(D1, D2, D3, K, N), dstShape(D1, D2, D3, M, N),
            biasShape(D1, D2, D3, M, N);
        g_matmulCallers[matmulCallerIndex]->call(srcShape, weiShape, dstShape, biasShape, 1, 1);
    }
}

/*==============================MultiPerfTest==================================*/
// M N K
static std::vector<std::vector<int>> common_MNKMulti = {};
static std::vector<std::vector<int>> common_DimsMulti = {};
static std::vector<std::vector<int>> freParasMulti = {
    {1, 1, 1, 4096, 3072, 5120}, {1, 1, 1, 4096, 1536, 5120}, {1, 1, 1, 4096, 5120, 1536}, {1, 1, 1, 4096, 5120, 3072},
    {1, 1, 1, 4096, 4096, 1152}, {1, 1, 1, 4096, 4608, 1152}, {1, 1, 1, 4096, 1152, 3256}, {1, 1, 1, 4096, 1152, 4608}};

static std::vector<int> numParasMulti =
    GenerateNumParas(common_MNKMulti.size() * common_DimsMulti.size() + freParasMulti.size());
static std::vector<std::vector<int>> testParasMulti;

class MTestMP : public ::testing::TestWithParam<TestParas> {
protected:
    static void SetUpTestSuite()
    {
        for (SizeType i = 0; i < common_MNKMulti.size(); ++i) {
            for (SizeType j = 0; j < common_DimsMulti.size(); ++j) {
                std::vector<int> combined;
                combined.insert(combined.end(), common_DimsMulti[j].begin(), common_DimsMulti[j].end());
                combined.insert(combined.end(), common_MNKMulti[i].begin(), common_MNKMulti[i].end());
                testParasMulti.emplace_back(combined);
            }
        }
        testParasMulti.insert(testParasMulti.end(), freParasMulti.begin(), freParasMulti.end());
    }
};

INSTANTIATE_TEST_SUITE_P(MTestMPs, MTestMP,
                         ::testing::Combine(::testing::ValuesIn(g_matmulCallersIndex), ::testing::ValuesIn(numThreads),
                                            ::testing::ValuesIn(numParasMulti)),
                         ParamName);

TEST_P(MTestMP, MatmulMultiPerf)
{
    size_t index = std::get<2>(GetParam());
    int matmulCallerIndex = std::get<0>(GetParam());
    std::string testname = g_nameMap[matmulCallerIndex];

    if (index < testParasMulti.size()) {
        int D1 = testParasMulti[index][0], D2 = testParasMulti[index][1], D3 = testParasMulti[index][2],
            M = testParasMulti[index][3], N = testParasMulti[index][4], K = testParasMulti[index][5];
        g_writer.SetFilePath("./matmul_test_result_multi_" + testname + ".txt");
        g_writer.WriteData(testname + "_" + std::to_string(index) + ", ");
        Shape srcShape(D1, D2, D3, M, K), weiShape(D1, D2, D3, K, N), dstShape(D1, D2, D3, M, N),
            biasShape(D1, D2, D3, M, N);
        g_matmulCallers[matmulCallerIndex]->call(srcShape, weiShape, dstShape, biasShape);
    }
}

/*=======================SinglePerfTest===========================*/
// M N K
static std::vector<std::vector<int>> common_MNKSin = {{512, 512, 192},   {512, 1024, 192}, {1024, 512, 192},
                                                      {1024, 1024, 192}, {512, 128, 512},  {512, 128, 1024},
                                                      {1024, 128, 512},  {1024, 128, 1024}};
static std::vector<std::vector<int>> common_DimsSin = {{1, 1, 128}};
static std::vector<std::vector<int>> freParasSin = {};

static std::vector<int> numParasSin =
    GenerateNumParas(common_MNKSin.size() * common_DimsSin.size() + freParasSin.size());
static std::vector<std::vector<int>> testParasSin;

class MTestSP : public ::testing::TestWithParam<TestParas> {
protected:
    static void SetUpTestSuite()
    {
        for (SizeType i = 0; i < common_MNKSin.size(); ++i) {
            for (SizeType j = 0; j < common_DimsSin.size(); ++j) {
                std::vector<int> combined;
                combined.insert(combined.end(), common_DimsSin[j].begin(), common_DimsSin[j].end());
                combined.insert(combined.end(), common_MNKSin[i].begin(), common_MNKSin[i].end());
                testParasSin.emplace_back(combined);
            }
        }
        testParasSin.insert(testParasSin.end(), freParasSin.begin(), freParasSin.end());
    }
};

INSTANTIATE_TEST_SUITE_P(MTestSPs, MTestSP,
                         ::testing::Combine(::testing::ValuesIn(g_matmulCallersIndex), ::testing::ValuesIn(numThreads),
                                            ::testing::ValuesIn(numParasSin)),
                         ParamName);

TEST_P(MTestSP, MatmulSinglePerf)
{
    size_t index = std::get<2>(GetParam());
    int matmulCallerIndex = std::get<0>(GetParam());
    std::string testname = g_nameMap[matmulCallerIndex];

    if (index < testParasSin.size()) {
        int D1 = testParasSin[index][0], D2 = testParasSin[index][1], D3 = testParasSin[index][2],
            M = testParasSin[index][3], N = testParasSin[index][4], K = testParasSin[index][5];
        g_writer.SetFilePath("./matmul_test_result_single_" + testname + ".txt");
        g_writer.WriteData(testname + "_" + std::to_string(index) + ", ");
        Shape srcShape(D1, D2, D3, M, K), weiShape(D1, D2, D3, K, N), dstShape(D1, D2, D3, M, N),
            biasShape(D1, D2, D3, M, N);
        g_matmulCallers[matmulCallerIndex]->call(srcShape, weiShape, dstShape, biasShape);
    }
}

/*****************************************************************************************************/
/********************************************原linearActivation用例***********************************/
/*****************************************************************************************************/

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
                                   const KuDNN::TensorInfo dstTensor, const KuDNN::TensorInfo biasTensor, float alpha,
                                   float beta, KuDNN::ActivationFunction algKind)
{
    KuDNN::PostOps post_ops;
    KuDNN::Attributes attributesInfo;
    post_ops.AppendEltwise(algKind, alpha, beta);
    attributesInfo.SetPostOps(post_ops);
    KuDNN::PostOpsDataPtrs poPtrs = {nullptr};
    KuDNN::Gemm gemmLayer1(srcTensor, weightTensor, dstTensor, biasTensor, attributesInfo);

    T *src = nullptr, *weight = nullptr, *dst = nullptr, *dstRef = nullptr, *bias = nullptr;
    floatDataInit<T>(&src, &weight, &dst, &dstRef, &bias, srcTensor.GetTotalTensorSize(),
                     weightTensor.GetTotalTensorSize(), dstTensor.GetTotalTensorSize());

    Shape srcShape = srcTensor.GetDims();
    Shape weightShape = weightTensor.GetDims();
    SizeType srcDimNums = srcShape.GetNumDims();
    SizeType weightDimNums = weightShape.GetNumDims();
    SizeType m = srcShape[srcDimNums - 2];
    SizeType n = weightShape[weightDimNums - 1];
    SizeType k = srcShape[srcDimNums - 1];
    SizeType batch = dstTensor.GetTotalTensorSize() / (m * n);

    linearActivationSimple(batch, m, n, k, src, weight, dstRef, bias, algKind);

    gemmLayer1.Run(src, weight, dst, bias, poPtrs, alpha, beta);

    float maxValue = findMaxValue<T>(src, batch * m * k, weight, batch * n * k);

    bool flag = true;
    for (SizeType i = 0; i < batch * m * n; ++i) {
        float diff = std::abs(static_cast<float>(*(dst + i) - *(dstRef + i)));
        auto pairEps = utils::compareEle(maxValue, diff, KuDNN::Element::MatchType<T>(), k);

        if (std::abs(pairEps.first) > pairEps.second) {
            flag = false;
            break;
        }
    }
    if (src != nullptr) {
        free(src);
    }
    if (weight != nullptr) {
        free(weight);
    }
    if (dst != nullptr) {
        free(dst);
    }
    if (dstRef != nullptr) {
        free(dstRef);
    }
    if (bias != nullptr) {
        free(bias);
    }
    return flag;
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
    
    EXPECT_TRUE(Test_Template_Function<float>(srcTensor, weightTensor, dstTensor, biasTensor, alpha, beta, algKind));
}

TEST(LinearActivationTests, FWD_LINEAR_RELU_3D_FP32)
{
    const KuDNN::TensorInfo srcTensor = {{2, 110, 20}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABC};
    const KuDNN::TensorInfo weightTensor = {{2, 20, 200}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABC};
    const KuDNN::TensorInfo dstTensor = {{2, 110, 200}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABC};
    const KuDNN::TensorInfo biasTensor = {{2, 110, 200}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABC};

    KuDNN::ActivationFunction algKind = KuDNN::ActivationFunction::RELU;
    float alpha = 1.0f;
    float beta = 1.0f;

    EXPECT_TRUE(Test_Template_Function<float>(srcTensor, weightTensor, dstTensor, biasTensor, alpha, beta, algKind));
}
TEST(LinearActivationTests, FWD_LINEAR_RELU_4D_FP32)
{
    const KuDNN::TensorInfo srcTensor = {{3, 2, 110, 20}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCD};
    const KuDNN::TensorInfo weightTensor = {{3, 2, 20, 200}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCD};
    const KuDNN::TensorInfo dstTensor = {{3, 2, 110, 200}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCD};
    const KuDNN::TensorInfo biasTensor = {{3, 2, 110, 200}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCD};
    float alpha = 0.0f;
    float beta = 0.0f;
    KuDNN::ActivationFunction algKind = KuDNN::ActivationFunction::RELU;
    
    EXPECT_TRUE(Test_Template_Function<float>(srcTensor, weightTensor, dstTensor, biasTensor, alpha, beta, algKind));
}

TEST(LinearActivationTests, FWD_LINEAR_RELU_5D_FP32)
{
    const KuDNN::TensorInfo srcTensor = {{4, 3, 2, 110, 20}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCDE};
    const KuDNN::TensorInfo weightTensor = {{4, 3, 2, 20, 200}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCDE};
    const KuDNN::TensorInfo dstTensor = {{4, 3, 2, 110, 200}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCDE};
    const KuDNN::TensorInfo biasTensor = {{4, 3, 2, 110, 200}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCDE};
    float alpha = 0.0f;
    float beta = 0.0f;
    KuDNN::ActivationFunction algKind = KuDNN::ActivationFunction::RELU;
    
    EXPECT_TRUE(Test_Template_Function<float>(srcTensor, weightTensor, dstTensor, biasTensor, alpha, beta, algKind));
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
    
    EXPECT_TRUE(Test_Template_Function<__fp16>(srcTensor, weightTensor, dstTensor, biasTensor, alpha, beta, algKind));
}

TEST(LinearActivationTests, FWD_LINEAR_RELU_3D_FP16)
{
    const KuDNN::TensorInfo srcTensor = {{2, 110, 20}, KuDNN::Element::TypeT::F16, KuDNN::Layout::ABC};
    const KuDNN::TensorInfo weightTensor = {{2, 20, 200}, KuDNN::Element::TypeT::F16, KuDNN::Layout::ABC};
    const KuDNN::TensorInfo dstTensor = {{2, 110, 200}, KuDNN::Element::TypeT::F16, KuDNN::Layout::ABC};
    const KuDNN::TensorInfo biasTensor = {{2, 110, 200}, KuDNN::Element::TypeT::F16, KuDNN::Layout::ABC};
    float alpha = 0.0f;
    float beta = 0.0f;
    KuDNN::ActivationFunction algKind = KuDNN::ActivationFunction::RELU;
    
    EXPECT_TRUE(Test_Template_Function<__fp16>(srcTensor, weightTensor, dstTensor, biasTensor, alpha, beta, algKind));
}

TEST(LinearActivationTests, FWD_LINEAR_RELU_4D_FP16)
{
    const KuDNN::TensorInfo srcTensor = {{3, 2, 110, 20}, KuDNN::Element::TypeT::F16, KuDNN::Layout::ABCD};
    const KuDNN::TensorInfo weightTensor = {{3, 2, 20, 200}, KuDNN::Element::TypeT::F16, KuDNN::Layout::ABCD};
    const KuDNN::TensorInfo dstTensor = {{3, 2, 110, 200}, KuDNN::Element::TypeT::F16, KuDNN::Layout::ABCD};
    const KuDNN::TensorInfo biasTensor = {{3, 2, 110, 200}, KuDNN::Element::TypeT::F16, KuDNN::Layout::ABCD};
    float alpha = 0.0f;
    float beta = 0.0f;
    KuDNN::ActivationFunction algKind = KuDNN::ActivationFunction::RELU;
    
    EXPECT_TRUE(Test_Template_Function<__fp16>(srcTensor, weightTensor, dstTensor, biasTensor, alpha, beta, algKind));
}

TEST(LinearActivationTests, FWD_LINEAR_RELU_5D_FP16)
{
    const KuDNN::TensorInfo srcTensor = {{4, 3, 2, 110, 20}, KuDNN::Element::TypeT::F16, KuDNN::Layout::ABCDE};
    const KuDNN::TensorInfo weightTensor = {{4, 3, 2, 20, 200}, KuDNN::Element::TypeT::F16, KuDNN::Layout::ABCDE};
    const KuDNN::TensorInfo dstTensor = {{4, 3, 2, 110, 200}, KuDNN::Element::TypeT::F16, KuDNN::Layout::ABCDE};
    const KuDNN::TensorInfo biasTensor = {{4, 3, 2, 110, 200}, KuDNN::Element::TypeT::F16, KuDNN::Layout::ABCDE};
    float alpha = 0.0f;
    float beta = 0.0f;
    KuDNN::ActivationFunction algKind = KuDNN::ActivationFunction::RELU;
    
    EXPECT_TRUE(Test_Template_Function<__fp16>(srcTensor, weightTensor, dstTensor, biasTensor, alpha, beta, algKind));
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
    
    EXPECT_TRUE(Test_Template_Function<__bf16>(srcTensor, weightTensor, dstTensor, biasTensor, alpha, beta, algKind));
}

TEST(LinearActivationTests, FWD_LINEAR_RELU_3D_BF16)
{
    const KuDNN::TensorInfo srcTensor = {{2, 110, 20}, KuDNN::Element::TypeT::BF16, KuDNN::Layout::ABC};
    const KuDNN::TensorInfo weightTensor = {{2, 20, 200}, KuDNN::Element::TypeT::BF16, KuDNN::Layout::ABC};
    const KuDNN::TensorInfo dstTensor = {{2, 110, 200}, KuDNN::Element::TypeT::BF16, KuDNN::Layout::ABC};
    const KuDNN::TensorInfo biasTensor = {{2, 110, 200}, KuDNN::Element::TypeT::BF16, KuDNN::Layout::ABC};
    float alpha = 0.0f;
    float beta = 0.0f;
    KuDNN::ActivationFunction algKind = KuDNN::ActivationFunction::RELU;
    
    EXPECT_TRUE(Test_Template_Function<__bf16>(srcTensor, weightTensor, dstTensor, biasTensor, alpha, beta, algKind));
}

TEST(LinearActivationTests, FWD_LINEAR_RELU_4D_BF16)
{
    const KuDNN::TensorInfo srcTensor = {{3, 2, 110, 20}, KuDNN::Element::TypeT::BF16, KuDNN::Layout::ABCD};
    const KuDNN::TensorInfo weightTensor = {{3, 2, 20, 200}, KuDNN::Element::TypeT::BF16, KuDNN::Layout::ABCD};
    const KuDNN::TensorInfo dstTensor = {{3, 2, 110, 200}, KuDNN::Element::TypeT::BF16, KuDNN::Layout::ABCD};
    const KuDNN::TensorInfo biasTensor = {{3, 2, 110, 200}, KuDNN::Element::TypeT::BF16, KuDNN::Layout::ABCD};
    float alpha = 0.0f;
    float beta = 0.0f;
    KuDNN::ActivationFunction algKind = KuDNN::ActivationFunction::RELU;
    
    EXPECT_TRUE(Test_Template_Function<__bf16>(srcTensor, weightTensor, dstTensor, biasTensor, alpha, beta, algKind));
}

TEST(LinearActivationTests, FWD_LINEAR_RELU_5D_BF16)
{
    const KuDNN::TensorInfo srcTensor = {{4, 3, 2, 110, 20}, KuDNN::Element::TypeT::BF16, KuDNN::Layout::ABCDE};
    const KuDNN::TensorInfo weightTensor = {{4, 3, 2, 20, 200}, KuDNN::Element::TypeT::BF16, KuDNN::Layout::ABCDE};
    const KuDNN::TensorInfo dstTensor = {{4, 3, 2, 110, 200}, KuDNN::Element::TypeT::BF16, KuDNN::Layout::ABCDE};
    const KuDNN::TensorInfo biasTensor = {{4, 3, 2, 110, 200}, KuDNN::Element::TypeT::BF16, KuDNN::Layout::ABCDE};
    float alpha = 0.0f;
    float beta = 0.0f;
    KuDNN::ActivationFunction algKind = KuDNN::ActivationFunction::RELU;
    
    EXPECT_TRUE(Test_Template_Function<__bf16>(srcTensor, weightTensor, dstTensor, biasTensor, alpha, beta, algKind));
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
    
    EXPECT_TRUE(Test_Template_Function<float>(srcTensor, weightTensor, dstTensor, biasTensor, alpha, beta, algKind));
}

TEST(LinearActivationTests, FWD_LINEAR_SILU_3D_FP32)
{
    const KuDNN::TensorInfo srcTensor = {{2, 110, 20}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABC};
    const KuDNN::TensorInfo weightTensor = {{2, 20, 200}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABC};
    const KuDNN::TensorInfo dstTensor = {{2, 110, 200}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABC};
    const KuDNN::TensorInfo biasTensor = {{2, 110, 200}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABC};
    float alpha = 1.0f;
    float beta = 0.0f;
    KuDNN::ActivationFunction algKind = KuDNN::ActivationFunction::SWISH;
    
    EXPECT_TRUE(Test_Template_Function<float>(srcTensor, weightTensor, dstTensor, biasTensor, alpha, beta, algKind));
}

TEST(LinearActivationTests, FWD_LINEAR_SILU_4D_FP32)
{
    const KuDNN::TensorInfo srcTensor = {{3, 2, 110, 20}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCD};
    const KuDNN::TensorInfo weightTensor = {{3, 2, 20, 200}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCD};
    const KuDNN::TensorInfo dstTensor = {{3, 2, 110, 200}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCD};
    const KuDNN::TensorInfo biasTensor = {{3, 2, 110, 200}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCD};
    float alpha = 1.0f;
    float beta = 0.0f;
    KuDNN::ActivationFunction algKind = KuDNN::ActivationFunction::SWISH;
    
    EXPECT_TRUE(Test_Template_Function<float>(srcTensor, weightTensor, dstTensor, biasTensor, alpha, beta, algKind));
}

TEST(LinearActivationTests, FWD_LINEAR_SILU_5D_FP32)
{
    const KuDNN::TensorInfo srcTensor = {{4, 3, 2, 110, 20}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCDE};
    const KuDNN::TensorInfo weightTensor = {{4, 3, 2, 20, 200}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCDE};
    const KuDNN::TensorInfo dstTensor = {{4, 3, 2, 110, 200}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCDE};
    const KuDNN::TensorInfo biasTensor = {{4, 3, 2, 110, 200}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCDE};
    float alpha = 1.0f;
    float beta = 0.0f;
    KuDNN::ActivationFunction algKind = KuDNN::ActivationFunction::SWISH;
    
    EXPECT_TRUE(Test_Template_Function<float>(srcTensor, weightTensor, dstTensor, biasTensor, alpha, beta, algKind));
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
    
    EXPECT_TRUE(Test_Template_Function<__fp16>(srcTensor, weightTensor, dstTensor, biasTensor, alpha, beta, algKind));
}

TEST(LinearActivationTests, FWD_LINEAR_SILU_3D_FP16)
{
    const KuDNN::TensorInfo srcTensor = {{2, 110, 20}, KuDNN::Element::TypeT::F16, KuDNN::Layout::ABC};
    const KuDNN::TensorInfo weightTensor = {{2, 20, 200}, KuDNN::Element::TypeT::F16, KuDNN::Layout::ABC};
    const KuDNN::TensorInfo dstTensor = {{2, 110, 200}, KuDNN::Element::TypeT::F16, KuDNN::Layout::ABC};
    const KuDNN::TensorInfo biasTensor = {{2, 110, 200}, KuDNN::Element::TypeT::F16, KuDNN::Layout::ABC};
    float alpha = 1.0f;
    float beta = 0.0f;
    KuDNN::ActivationFunction algKind = KuDNN::ActivationFunction::SWISH;
    
    EXPECT_TRUE(Test_Template_Function<__fp16>(srcTensor, weightTensor, dstTensor, biasTensor, alpha, beta, algKind));
}

TEST(LinearActivationTests, FWD_LINEAR_SILU_4D_FP16)
{
    const KuDNN::TensorInfo srcTensor = {{3, 2, 110, 20}, KuDNN::Element::TypeT::F16, KuDNN::Layout::ABCD};
    const KuDNN::TensorInfo weightTensor = {{3, 2, 20, 200}, KuDNN::Element::TypeT::F16, KuDNN::Layout::ABCD};
    const KuDNN::TensorInfo dstTensor = {{3, 2, 110, 200}, KuDNN::Element::TypeT::F16, KuDNN::Layout::ABCD};
    const KuDNN::TensorInfo biasTensor = {{3, 2, 110, 200}, KuDNN::Element::TypeT::F16, KuDNN::Layout::ABCD};
    float alpha = 1.0f;
    float beta = 0.0f;
    KuDNN::ActivationFunction algKind = KuDNN::ActivationFunction::SWISH;
    
    EXPECT_TRUE(Test_Template_Function<__fp16>(srcTensor, weightTensor, dstTensor, biasTensor, alpha, beta, algKind));
}

TEST(LinearActivationTests, FWD_LINEAR_SILU_5D_FP16)
{
    const KuDNN::TensorInfo srcTensor = {{4, 3, 2, 110, 20}, KuDNN::Element::TypeT::F16, KuDNN::Layout::ABCDE};
    const KuDNN::TensorInfo weightTensor = {{4, 3, 2, 20, 200}, KuDNN::Element::TypeT::F16, KuDNN::Layout::ABCDE};
    const KuDNN::TensorInfo dstTensor = {{4, 3, 2, 110, 200}, KuDNN::Element::TypeT::F16, KuDNN::Layout::ABCDE};
    const KuDNN::TensorInfo biasTensor = {{4, 3, 2, 110, 200}, KuDNN::Element::TypeT::F16, KuDNN::Layout::ABCDE};
    float alpha = 1.0f;
    float beta = 0.0f;
    KuDNN::ActivationFunction algKind = KuDNN::ActivationFunction::SWISH;
    
    EXPECT_TRUE(Test_Template_Function<__fp16>(srcTensor, weightTensor, dstTensor, biasTensor, alpha, beta, algKind));
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
    
    EXPECT_TRUE(Test_Template_Function<__bf16>(srcTensor, weightTensor, dstTensor, biasTensor, alpha, beta, algKind));
}

TEST(LinearActivationTests, FWD_LINEAR_SILU_3D_BF16)
{
    const KuDNN::TensorInfo srcTensor = {{2, 110, 20}, KuDNN::Element::TypeT::BF16, KuDNN::Layout::ABC};
    const KuDNN::TensorInfo weightTensor = {{2, 20, 200}, KuDNN::Element::TypeT::BF16, KuDNN::Layout::ABC};
    const KuDNN::TensorInfo dstTensor = {{2, 110, 200}, KuDNN::Element::TypeT::BF16, KuDNN::Layout::ABC};
    const KuDNN::TensorInfo biasTensor = {{2, 110, 200}, KuDNN::Element::TypeT::BF16, KuDNN::Layout::ABC};
    float alpha = 1.0f;
    float beta = 0.0f;
    KuDNN::ActivationFunction algKind = KuDNN::ActivationFunction::SWISH;
    
    EXPECT_TRUE(Test_Template_Function<__bf16>(srcTensor, weightTensor, dstTensor, biasTensor, alpha, beta, algKind));
}

TEST(LinearActivationTests, FWD_LINEAR_SILU_4D_BF16)
{
    const KuDNN::TensorInfo srcTensor = {{3, 2, 110, 20}, KuDNN::Element::TypeT::BF16, KuDNN::Layout::ABCD};
    const KuDNN::TensorInfo weightTensor = {{3, 2, 20, 200}, KuDNN::Element::TypeT::BF16, KuDNN::Layout::ABCD};
    const KuDNN::TensorInfo dstTensor = {{3, 2, 110, 200}, KuDNN::Element::TypeT::BF16, KuDNN::Layout::ABCD};
    const KuDNN::TensorInfo biasTensor = {{3, 2, 110, 200}, KuDNN::Element::TypeT::BF16, KuDNN::Layout::ABCD};
    float alpha = 1.0f;
    float beta = 0.0f;
    KuDNN::ActivationFunction algKind = KuDNN::ActivationFunction::SWISH;
    
    EXPECT_TRUE(Test_Template_Function<__bf16>(srcTensor, weightTensor, dstTensor, biasTensor, alpha, beta, algKind));
}

TEST(LinearActivationTests, FWD_LINEAR_SILU_5D_BF16)
{
    const KuDNN::TensorInfo srcTensor = {{4, 3, 2, 110, 20}, KuDNN::Element::TypeT::BF16, KuDNN::Layout::ABCDE};
    const KuDNN::TensorInfo weightTensor = {{4, 3, 2, 20, 200}, KuDNN::Element::TypeT::BF16, KuDNN::Layout::ABCDE};
    const KuDNN::TensorInfo dstTensor = {{4, 3, 2, 110, 200}, KuDNN::Element::TypeT::BF16, KuDNN::Layout::ABCDE};
    const KuDNN::TensorInfo biasTensor = {{4, 3, 2, 110, 200}, KuDNN::Element::TypeT::BF16, KuDNN::Layout::ABCDE};
    float alpha = 1.0f;
    float beta = 0.0f;
    KuDNN::ActivationFunction algKind = KuDNN::ActivationFunction::SWISH;
    
    EXPECT_TRUE(Test_Template_Function<__bf16>(srcTensor, weightTensor, dstTensor, biasTensor, alpha, beta, algKind));
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
    
    EXPECT_TRUE(Test_Template_Function<float>(srcTensor, weightTensor, dstTensor, biasTensor, alpha, beta, algKind));
}

TEST(LinearActivationTests, FWD_LINEAR_GELU_ERF_3D_FP32)
{
    const KuDNN::TensorInfo srcTensor = {{2, 110, 20}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABC};
    const KuDNN::TensorInfo weightTensor = {{2, 20, 200}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABC};
    const KuDNN::TensorInfo dstTensor = {{2, 110, 200}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABC};
    const KuDNN::TensorInfo biasTensor = {{2, 110, 200}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABC};
    float alpha = 1.0f;
    float beta = 0.0f;
    KuDNN::ActivationFunction algKind = KuDNN::ActivationFunction::GELU_ERF;
    
    EXPECT_TRUE(Test_Template_Function<float>(srcTensor, weightTensor, dstTensor, biasTensor, alpha, beta, algKind));
}

TEST(LinearActivationTests, FWD_LINEAR_GELU_ERF_4D_FP32)
{
    const KuDNN::TensorInfo srcTensor = {{3, 2, 110, 20}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCD};
    const KuDNN::TensorInfo weightTensor = {{3, 2, 20, 200}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCD};
    const KuDNN::TensorInfo dstTensor = {{3, 2, 110, 200}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCD};
    const KuDNN::TensorInfo biasTensor = {{3, 2, 110, 200}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCD};
    float alpha = 1.0f;
    float beta = 0.0f;
    KuDNN::ActivationFunction algKind = KuDNN::ActivationFunction::GELU_ERF;
    
    EXPECT_TRUE(Test_Template_Function<float>(srcTensor, weightTensor, dstTensor, biasTensor, alpha, beta, algKind));
}

TEST(LinearActivationTests, FWD_LINEAR_GELU_ERF_5D_FP32)
{
    const KuDNN::TensorInfo srcTensor = {{4, 3, 2, 110, 20}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCDE};
    const KuDNN::TensorInfo weightTensor = {{4, 3, 2, 20, 200}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCDE};
    const KuDNN::TensorInfo dstTensor = {{4, 3, 2, 110, 200}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCDE};
    const KuDNN::TensorInfo biasTensor = {{4, 3, 2, 110, 200}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCDE};
    float alpha = 1.0f;
    float beta = 0.0f;
    KuDNN::ActivationFunction algKind = KuDNN::ActivationFunction::GELU_ERF;
    
    EXPECT_TRUE(Test_Template_Function<float>(srcTensor, weightTensor, dstTensor, biasTensor, alpha, beta, algKind));
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
    
    EXPECT_TRUE(Test_Template_Function<__fp16>(srcTensor, weightTensor, dstTensor, biasTensor, alpha, beta, algKind));
}

TEST(LinearActivationTests, FWD_LINEAR_GELU_ERF_3D_FP16)
{
    const KuDNN::TensorInfo srcTensor = {{2, 110, 20}, KuDNN::Element::TypeT::F16, KuDNN::Layout::ABC};
    const KuDNN::TensorInfo weightTensor = {{2, 20, 200}, KuDNN::Element::TypeT::F16, KuDNN::Layout::ABC};
    const KuDNN::TensorInfo dstTensor = {{2, 110, 200}, KuDNN::Element::TypeT::F16, KuDNN::Layout::ABC};
    const KuDNN::TensorInfo biasTensor = {{2, 110, 200}, KuDNN::Element::TypeT::F16, KuDNN::Layout::ABC};
    float alpha = 1.0f;
    float beta = 0.0f;
    KuDNN::ActivationFunction algKind = KuDNN::ActivationFunction::GELU_ERF;
    
    EXPECT_TRUE(Test_Template_Function<__fp16>(srcTensor, weightTensor, dstTensor, biasTensor, alpha, beta, algKind));
}

TEST(LinearActivationTests, FWD_LINEAR_GELU_ERF_4D_FP16)
{
    const KuDNN::TensorInfo srcTensor = {{3, 2, 110, 20}, KuDNN::Element::TypeT::F16, KuDNN::Layout::ABCD};
    const KuDNN::TensorInfo weightTensor = {{3, 2, 20, 200}, KuDNN::Element::TypeT::F16, KuDNN::Layout::ABCD};
    const KuDNN::TensorInfo dstTensor = {{3, 2, 110, 200}, KuDNN::Element::TypeT::F16, KuDNN::Layout::ABCD};
    const KuDNN::TensorInfo biasTensor = {{3, 2, 110, 200}, KuDNN::Element::TypeT::F16, KuDNN::Layout::ABCD};
    float alpha = 1.0f;
    float beta = 0.0f;
    KuDNN::ActivationFunction algKind = KuDNN::ActivationFunction::GELU_ERF;
    
    EXPECT_TRUE(Test_Template_Function<__fp16>(srcTensor, weightTensor, dstTensor, biasTensor, alpha, beta, algKind));
}

TEST(LinearActivationTests, FWD_LINEAR_GELU_ERF_5D_FP16)
{
    const KuDNN::TensorInfo srcTensor = {{4, 3, 2, 110, 20}, KuDNN::Element::TypeT::F16, KuDNN::Layout::ABCDE};
    const KuDNN::TensorInfo weightTensor = {{4, 3, 2, 20, 200}, KuDNN::Element::TypeT::F16, KuDNN::Layout::ABCDE};
    const KuDNN::TensorInfo dstTensor = {{4, 3, 2, 110, 200}, KuDNN::Element::TypeT::F16, KuDNN::Layout::ABCDE};
    const KuDNN::TensorInfo biasTensor = {{4, 3, 2, 110, 200}, KuDNN::Element::TypeT::F16, KuDNN::Layout::ABCDE};
    float alpha = 1.0f;
    float beta = 0.0f;
    KuDNN::ActivationFunction algKind = KuDNN::ActivationFunction::GELU_ERF;
    
    EXPECT_TRUE(Test_Template_Function<__fp16>(srcTensor, weightTensor, dstTensor, biasTensor, alpha, beta, algKind));
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
    
    EXPECT_TRUE(Test_Template_Function<__bf16>(srcTensor, weightTensor, dstTensor, biasTensor, alpha, beta, algKind));
}

TEST(LinearActivationTests, FWD_LINEAR_GELU_ERF_3D_BF16)
{
    const KuDNN::TensorInfo srcTensor = {{2, 110, 20}, KuDNN::Element::TypeT::BF16, KuDNN::Layout::ABC};
    const KuDNN::TensorInfo weightTensor = {{2, 20, 200}, KuDNN::Element::TypeT::BF16, KuDNN::Layout::ABC};
    const KuDNN::TensorInfo dstTensor = {{2, 110, 200}, KuDNN::Element::TypeT::BF16, KuDNN::Layout::ABC};
    const KuDNN::TensorInfo biasTensor = {{2, 110, 200}, KuDNN::Element::TypeT::BF16, KuDNN::Layout::ABC};
    float alpha = 1.0f;
    float beta = 0.0f;
    KuDNN::ActivationFunction algKind = KuDNN::ActivationFunction::GELU_ERF;
    
    EXPECT_TRUE(Test_Template_Function<__bf16>(srcTensor, weightTensor, dstTensor, biasTensor, alpha, beta, algKind));
}

TEST(LinearActivationTests, FWD_LINEAR_GELU_ERF_4D_BF16)
{
    const KuDNN::TensorInfo srcTensor = {{3, 2, 110, 20}, KuDNN::Element::TypeT::BF16, KuDNN::Layout::ABCD};
    const KuDNN::TensorInfo weightTensor = {{3, 2, 20, 200}, KuDNN::Element::TypeT::BF16, KuDNN::Layout::ABCD};
    const KuDNN::TensorInfo dstTensor = {{3, 2, 110, 200}, KuDNN::Element::TypeT::BF16, KuDNN::Layout::ABCD};
    const KuDNN::TensorInfo biasTensor = {{3, 2, 110, 200}, KuDNN::Element::TypeT::BF16, KuDNN::Layout::ABCD};
    float alpha = 1.0f;
    float beta = 0.0f;
    KuDNN::ActivationFunction algKind = KuDNN::ActivationFunction::GELU_ERF;
    
    EXPECT_TRUE(Test_Template_Function<__bf16>(srcTensor, weightTensor, dstTensor, biasTensor, alpha, beta, algKind));
}

TEST(LinearActivationTests, FWD_LINEAR_GELU_ERF_5D_BF16)
{
    const KuDNN::TensorInfo srcTensor = {{4, 3, 2, 110, 20}, KuDNN::Element::TypeT::BF16, KuDNN::Layout::ABCDE};
    const KuDNN::TensorInfo weightTensor = {{4, 3, 2, 20, 200}, KuDNN::Element::TypeT::BF16, KuDNN::Layout::ABCDE};
    const KuDNN::TensorInfo dstTensor = {{4, 3, 2, 110, 200}, KuDNN::Element::TypeT::BF16, KuDNN::Layout::ABCDE};
    const KuDNN::TensorInfo biasTensor = {{4, 3, 2, 110, 200}, KuDNN::Element::TypeT::BF16, KuDNN::Layout::ABCDE};
    float alpha = 1.0f;
    float beta = 0.0f;
    KuDNN::ActivationFunction algKind = KuDNN::ActivationFunction::GELU_ERF;
    
    EXPECT_TRUE(Test_Template_Function<__bf16>(srcTensor, weightTensor, dstTensor, biasTensor, alpha, beta, algKind));
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
    
    EXPECT_TRUE(Test_Template_Function<float>(srcTensor, weightTensor, dstTensor, biasTensor, alpha, beta, algKind));
}

TEST(LinearActivationTests, FWD_LINEAR_GELU_TANH_3D_FP32)
{
    const KuDNN::TensorInfo srcTensor = {{2, 110, 20}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABC};
    const KuDNN::TensorInfo weightTensor = {{2, 20, 200}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABC};
    const KuDNN::TensorInfo dstTensor = {{2, 110, 200}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABC};
    const KuDNN::TensorInfo biasTensor = {{2, 110, 200}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABC};
    float alpha = 1.0f;
    float beta = 0.0f;
    KuDNN::ActivationFunction algKind = KuDNN::ActivationFunction::GELU_TANH;
    
    EXPECT_TRUE(Test_Template_Function<float>(srcTensor, weightTensor, dstTensor, biasTensor, alpha, beta, algKind));
}

TEST(LinearActivationTests, FWD_LINEAR_GELU_TANH_4D_FP32)
{
    const KuDNN::TensorInfo srcTensor = {{3, 2, 110, 20}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCD};
    const KuDNN::TensorInfo weightTensor = {{3, 2, 20, 200}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCD};
    const KuDNN::TensorInfo dstTensor = {{3, 2, 110, 200}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCD};
    const KuDNN::TensorInfo biasTensor = {{3, 2, 110, 200}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCD};
    float alpha = 1.0f;
    float beta = 0.0f;
    KuDNN::ActivationFunction algKind = KuDNN::ActivationFunction::GELU_TANH;
    
    EXPECT_TRUE(Test_Template_Function<float>(srcTensor, weightTensor, dstTensor, biasTensor, alpha, beta, algKind));
}

TEST(LinearActivationTests, FWD_LINEAR_GELU_TANH_5D_FP32)
{
    const KuDNN::TensorInfo srcTensor = {{4, 3, 2, 110, 20}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCDE};
    const KuDNN::TensorInfo weightTensor = {{4, 3, 2, 20, 200}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCDE};
    const KuDNN::TensorInfo dstTensor = {{4, 3, 2, 110, 200}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCDE};
    const KuDNN::TensorInfo biasTensor = {{4, 3, 2, 110, 200}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCDE};
    float alpha = 1.0f;
    float beta = 0.0f;
    KuDNN::ActivationFunction algKind = KuDNN::ActivationFunction::GELU_TANH;
    
    EXPECT_TRUE(Test_Template_Function<float>(srcTensor, weightTensor, dstTensor, biasTensor, alpha, beta, algKind));
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
    
    EXPECT_TRUE(Test_Template_Function<__fp16>(srcTensor, weightTensor, dstTensor, biasTensor, alpha, beta, algKind));
}

TEST(LinearActivationTests, FWD_LINEAR_GELU_TANH_3D_FP16)
{
    const KuDNN::TensorInfo srcTensor = {{2, 110, 20}, KuDNN::Element::TypeT::F16, KuDNN::Layout::ABC};
    const KuDNN::TensorInfo weightTensor = {{2, 20, 200}, KuDNN::Element::TypeT::F16, KuDNN::Layout::ABC};
    const KuDNN::TensorInfo dstTensor = {{2, 110, 200}, KuDNN::Element::TypeT::F16, KuDNN::Layout::ABC};
    const KuDNN::TensorInfo biasTensor = {{2, 110, 200}, KuDNN::Element::TypeT::F16, KuDNN::Layout::ABC};
    float alpha = 1.0f;
    float beta = 0.0f;
    KuDNN::ActivationFunction algKind = KuDNN::ActivationFunction::GELU_TANH;
    
    EXPECT_TRUE(Test_Template_Function<__fp16>(srcTensor, weightTensor, dstTensor, biasTensor, alpha, beta, algKind));
}

TEST(LinearActivationTests, FWD_LINEAR_GELU_TANH_4D_FP16)
{
    const KuDNN::TensorInfo srcTensor = {{3, 2, 110, 20}, KuDNN::Element::TypeT::F16, KuDNN::Layout::ABCD};
    const KuDNN::TensorInfo weightTensor = {{3, 2, 20, 200}, KuDNN::Element::TypeT::F16, KuDNN::Layout::ABCD};
    const KuDNN::TensorInfo dstTensor = {{3, 2, 110, 200}, KuDNN::Element::TypeT::F16, KuDNN::Layout::ABCD};
    const KuDNN::TensorInfo biasTensor = {{3, 2, 110, 200}, KuDNN::Element::TypeT::F16, KuDNN::Layout::ABCD};
    float alpha = 1.0f;
    float beta = 0.0f;
    KuDNN::ActivationFunction algKind = KuDNN::ActivationFunction::GELU_TANH;
    
    EXPECT_TRUE(Test_Template_Function<__fp16>(srcTensor, weightTensor, dstTensor, biasTensor, alpha, beta, algKind));
}

TEST(LinearActivationTests, FWD_LINEAR_GELU_TANH_5D_FP16)
{
    const KuDNN::TensorInfo srcTensor = {{4, 3, 2, 110, 20}, KuDNN::Element::TypeT::F16, KuDNN::Layout::ABCDE};
    const KuDNN::TensorInfo weightTensor = {{4, 3, 2, 20, 200}, KuDNN::Element::TypeT::F16, KuDNN::Layout::ABCDE};
    const KuDNN::TensorInfo dstTensor = {{4, 3, 2, 110, 200}, KuDNN::Element::TypeT::F16, KuDNN::Layout::ABCDE};
    const KuDNN::TensorInfo biasTensor = {{4, 3, 2, 110, 200}, KuDNN::Element::TypeT::F16, KuDNN::Layout::ABCDE};
    float alpha = 1.0f;
    float beta = 0.0f;
    KuDNN::ActivationFunction algKind = KuDNN::ActivationFunction::GELU_TANH;
    
    EXPECT_TRUE(Test_Template_Function<__fp16>(srcTensor, weightTensor, dstTensor, biasTensor, alpha, beta, algKind));
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
    
    EXPECT_TRUE(Test_Template_Function<__bf16>(srcTensor, weightTensor, dstTensor, biasTensor, alpha, beta, algKind));
}

TEST(LinearActivationTests, FWD_LINEAR_GELU_TANH_3D_BF16)
{
    const KuDNN::TensorInfo srcTensor = {{2, 110, 20}, KuDNN::Element::TypeT::BF16, KuDNN::Layout::ABC};
    const KuDNN::TensorInfo weightTensor = {{2, 20, 200}, KuDNN::Element::TypeT::BF16, KuDNN::Layout::ABC};
    const KuDNN::TensorInfo dstTensor = {{2, 110, 200}, KuDNN::Element::TypeT::BF16, KuDNN::Layout::ABC};
    const KuDNN::TensorInfo biasTensor = {{2, 110, 200}, KuDNN::Element::TypeT::BF16, KuDNN::Layout::ABC};
    float alpha = 1.0f;
    float beta = 0.0f;
    KuDNN::ActivationFunction algKind = KuDNN::ActivationFunction::GELU_TANH;
    
    EXPECT_TRUE(Test_Template_Function<__bf16>(srcTensor, weightTensor, dstTensor, biasTensor, alpha, beta, algKind));
}

TEST(LinearActivationTests, FWD_LINEAR_GELU_TANH_4D_BF16)
{
    const KuDNN::TensorInfo srcTensor = {{3, 2, 110, 20}, KuDNN::Element::TypeT::BF16, KuDNN::Layout::ABCD};
    const KuDNN::TensorInfo weightTensor = {{3, 2, 20, 200}, KuDNN::Element::TypeT::BF16, KuDNN::Layout::ABCD};
    const KuDNN::TensorInfo dstTensor = {{3, 2, 110, 200}, KuDNN::Element::TypeT::BF16, KuDNN::Layout::ABCD};
    const KuDNN::TensorInfo biasTensor = {{3, 2, 110, 200}, KuDNN::Element::TypeT::BF16, KuDNN::Layout::ABCD};
    float alpha = 1.0f;
    float beta = 0.0f;
    KuDNN::ActivationFunction algKind = KuDNN::ActivationFunction::GELU_TANH;
    
    EXPECT_TRUE(Test_Template_Function<__bf16>(srcTensor, weightTensor, dstTensor, biasTensor, alpha, beta, algKind));
}

TEST(LinearActivationTests, FWD_LINEAR_GELU_TANH_5D_BF16)
{
    const KuDNN::TensorInfo srcTensor = {{4, 3, 2, 110, 20}, KuDNN::Element::TypeT::BF16, KuDNN::Layout::ABCDE};
    const KuDNN::TensorInfo weightTensor = {{4, 3, 2, 20, 200}, KuDNN::Element::TypeT::BF16, KuDNN::Layout::ABCDE};
    const KuDNN::TensorInfo dstTensor = {{4, 3, 2, 110, 200}, KuDNN::Element::TypeT::BF16, KuDNN::Layout::ABCDE};
    const KuDNN::TensorInfo biasTensor = {{4, 3, 2, 110, 200}, KuDNN::Element::TypeT::BF16, KuDNN::Layout::ABCDE};
    float alpha = 1.0f;
    float beta = 0.0f;
    KuDNN::ActivationFunction algKind = KuDNN::ActivationFunction::GELU_TANH;
    
    EXPECT_TRUE(Test_Template_Function<__bf16>(srcTensor, weightTensor, dstTensor, biasTensor, alpha, beta, algKind));
}
