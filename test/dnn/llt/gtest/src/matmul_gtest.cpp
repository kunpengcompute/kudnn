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
#include <cstdlib>
#include <ctime>
#include <cmath>
#include <vector>
#include <random>
#include <algorithm>
#include <stdexcept>
#include <iostream>
#include <chrono>
#include <memory>
#include <string>
#include <limits.h>
#include <omp.h>
#include <arm_neon.h>

using SizeType = KuDNN::SizeType;
using Shape = KuDNN::Shape;

static void NoPackCopyConstructor1()
{
    const KuDNN::TensorInfo matATensor = {{1, 1, 1, 1, 1}, KuDNN::Element::MatchType<float>(), KuDNN::Layout::ABCDE};
    const KuDNN::TensorInfo matBTensor = {{1, 1, 1, 1, 1}, KuDNN::Element::MatchType<float>(), KuDNN::Layout::ABCDE};
    const KuDNN::TensorInfo matCTensor = {{1, 1, 1, 1, 1}, KuDNN::Element::MatchType<float>(), KuDNN::Layout::ABCDE};
    const KuDNN::TensorInfo biasTensor = {{1, 1, 1, 1, 1}, KuDNN::Element::MatchType<float>(), KuDNN::Layout::ABCDE};
    KuDNN::Gemm gemmLayer1(matATensor, matBTensor, matCTensor, biasTensor);
    KuDNN::Gemm gemmLayer2(gemmLayer1);
    for (size_t i = 0; i < g_Tensor_data.size() / 4; i += 4) {
        KuDNN::Gemm gemmLayer1(
            g_Tensor_data[i], g_Tensor_data[i + 1], g_Tensor_data[i + 2], g_Tensor_data[i + 3]);
        KuDNN::Gemm gemmLayer2(gemmLayer1);
    }
}

static void NoPackCopyConstructor2()
{
    const KuDNN::TensorInfo matATensor = {{1, 1, 1, 1, 1}, KuDNN::Element::MatchType<float>(), KuDNN::Layout::ABCDE};
    const KuDNN::TensorInfo matBTensor = {{1, 1, 1, 1, 1}, KuDNN::Element::MatchType<float>(), KuDNN::Layout::ABCDE};
    const KuDNN::TensorInfo matCTensor = {{1, 1, 1, 1, 1}, KuDNN::Element::MatchType<float>(), KuDNN::Layout::ABCDE};
    const KuDNN::TensorInfo biasTensor = {{1, 1, 1, 1, 1}, KuDNN::Element::MatchType<float>(), KuDNN::Layout::ABCDE};
    KuDNN::Gemm gemmLayer1(matATensor, matBTensor, matCTensor, biasTensor);
    KuDNN::Gemm gemmLayer2(gemmLayer1);
    gemmLayer2 = gemmLayer1;
    for (size_t i = 0; i < g_Tensor_data.size() / 4; i += 4) {
        KuDNN::Gemm gemmLayer1(
            g_Tensor_data[i], g_Tensor_data[i + 1], g_Tensor_data[i + 2], g_Tensor_data[i + 3]);
        KuDNN::Gemm gemmLayer2(gemmLayer1);
        gemmLayer2 = gemmLayer1;
    }
}

static void NoPackMoveConstructor1()
{
    const KuDNN::TensorInfo matATensor = {{1, 1, 1, 1, 1}, KuDNN::Element::MatchType<float>(), KuDNN::Layout::ABCDE};
    const KuDNN::TensorInfo matBTensor = {{1, 1, 1, 1, 1}, KuDNN::Element::MatchType<float>(), KuDNN::Layout::ABCDE};
    const KuDNN::TensorInfo matCTensor = {{1, 1, 1, 1, 1}, KuDNN::Element::MatchType<float>(), KuDNN::Layout::ABCDE};
    const KuDNN::TensorInfo biasTensor = {{1, 1, 1, 1, 1}, KuDNN::Element::MatchType<float>(), KuDNN::Layout::ABCDE};
    KuDNN::Gemm gemmLayer1(matATensor, matBTensor, matCTensor, biasTensor);
    KuDNN::Gemm gemmLayer2(std::move(gemmLayer1));
    for (size_t i = 0; i < g_Tensor_data.size() / 4; i += 4) {
        KuDNN::Gemm gemmLayer1(
            g_Tensor_data[i], g_Tensor_data[i + 1], g_Tensor_data[i + 2], g_Tensor_data[i + 3]);
        KuDNN::Gemm gemmLayer2(std::move(gemmLayer1));
    }
}

static void NoPackMoveConstructor2()
{
    const KuDNN::TensorInfo matATensor = {{1, 1, 1, 1, 1}, KuDNN::Element::MatchType<float>(), KuDNN::Layout::ABCDE};
    const KuDNN::TensorInfo matBTensor = {{1, 1, 1, 1, 1}, KuDNN::Element::MatchType<float>(), KuDNN::Layout::ABCDE};
    const KuDNN::TensorInfo matCTensor = {{1, 1, 1, 1, 1}, KuDNN::Element::MatchType<float>(), KuDNN::Layout::ABCDE};
    const KuDNN::TensorInfo biasTensor = {{1, 1, 1, 1, 1}, KuDNN::Element::MatchType<float>(), KuDNN::Layout::ABCDE};
    KuDNN::Gemm gemmLayer1(matATensor, matBTensor, matCTensor, biasTensor);
    KuDNN::Gemm gemmLayer2(gemmLayer1);
    gemmLayer2 = std::move(gemmLayer1);
    for (size_t i = 0; i < g_Tensor_data.size() / 4; i += 4) {
        KuDNN::Gemm gemmLayer1(
            g_Tensor_data[i], g_Tensor_data[i + 1], g_Tensor_data[i + 2], g_Tensor_data[i + 3]);
        KuDNN::Gemm gemmLayer2(gemmLayer1);
        gemmLayer2 = std::move(gemmLayer1);
    }
}

namespace utils {

template <typename T1, typename T2, typename T3>
static void MatmulRef(const KuDNN::TensorInfo &srcTensor, const KuDNN::TensorInfo &weiTensor, T1 *A, T2 *B, T3 *C)
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
                    } else if (srcLayout == KuDNN::Layout::ABCED && weiLayout == KuDNN::Layout::ABCDE){
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
void matrixAdd(T1 *A, T2 *B, size_t size)
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
        return [e, u]() mutable { return u(e); };
    } else if (KuDNN::Element::MatchType<T>() == KuDNN::Element::TypeT::S32) {
        std::uniform_int_distribution<int> u(-65536, 65536);
        return [e, u]() mutable { return u(e); };
    } else {
        std::uniform_real_distribution<float> u(-1, 1);
        return [e, u]() mutable { return u(e); };
    }
}

std::pair<float, float> compareEle(float ref, float diff, KuDNN::Element::TypeT type, int K)
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
    explicit DataWriter(const std::string &filePath) : filePath(filePath)
    {}

    void WriteData(const std::string &data)
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

    void SetFilePath(const std::string &fp)
    {
        this->filePath = fp;
    }

private:
    std::string filePath;  // File path
};

};  // namespace utils

static utils::DataWriter g_writer("./matmul_correctness_test_result.txt");

template <typename srcType, typename weiType, typename dstType, typename biasType>
void Matmul(Shape &srcShape, Shape &weiShape, Shape &dstShape, Shape &biasShape, int numThreads = 0, int transSrc = 0, int transWei = 0)
{
    KuDNN::Layout srcLayout = KuDNN::Layout::ABCDE;
    KuDNN::Layout weiLayout = KuDNN::Layout::ABCDE;
    if(transSrc != 0){
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

    void *src = malloc(srcSize * sizeof(srcType));
    void *wei = malloc(weiSize * sizeof(weiType));
    void *dst = malloc(dstSize * sizeof(dstType));
    void *bias = malloc(biaSize * sizeof(biasType));
    void *ref = malloc(dstSize * sizeof(dstType));
    if (src == nullptr || wei == nullptr || dst == nullptr || bias == nullptr || ref == nullptr) {
        throw std::runtime_error("malloc failed");
    }

    srcType *src_Type = static_cast<srcType *>(src);
    weiType *wei_Type = static_cast<weiType *>(wei);
    dstType *dst_Type = static_cast<dstType *>(dst);
    biasType *bias_Type = static_cast<biasType *>(bias);
    dstType *ref_Type = static_cast<dstType *>(ref);

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
        if constexpr (std::is_same_v<dstType, bfloat16_t>) {
            *(dst_Type + i) = vcvth_bf16_f32(0);
        }else {
            *(dst_Type + i) = 0;
        }
    }
    #pragma omp parallel for
    for (SizeType i = 0; i < biaSize; ++i) {
        *(bias_Type + i) = (biasType)biasRandomGenerator();
    }
    #pragma omp parallel for
    for (SizeType i = 0; i < dstSize; ++i) {
        if constexpr (std::is_same_v<dstType, bfloat16_t>) {
            *(ref_Type + i) = vcvth_bf16_f32(0);
        }else {
            *(ref_Type + i) = 0;
        }
    }
    decltype(std::chrono::high_resolution_clock::now()) start;
    decltype(std::chrono::high_resolution_clock::now()) end;

    try {
        KuDNN::Gemm gemmLayer(srcTensor, weiTensor, dstTensor, biasTensor, numThreads);
        start = std::chrono::high_resolution_clock::now();
        gemmLayer.Run(src, wei, dst, bias, 1, 0, numThreads);
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
        if constexpr (std::is_same_v<dstType, bfloat16_t>) {
            dstMax = std::max(dstMax, std::fabs(vcvtah_f32_bf16(tmp)));
        }
        else {
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
        if constexpr (std::is_same_v<dstType, bfloat16_t>) {
            ref = vcvtah_f32_bf16(*(ref_Type + i));
            dst = vcvtah_f32_bf16(*(dst_Type + i));
        }
        else {
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
    virtual void call(Shape &srcShape, Shape &weiShape, Shape &dstShape, Shape &biasShape, int numThreads = 0, int transSrc = 0, int transWei = 0) const = 0;
};

template <typename srcType, typename weiType, typename dstType, typename biasType>
class MatmulCaller : public FunctionBase {
public:
    void call(Shape &srcShape, Shape &weiShape, Shape &dstShape, Shape &biasShape, int numThreads = 0, int transSrc = 0, int transWei = 0) const override {
        Matmul<srcType, weiType, dstType, biasType>(srcShape, weiShape, dstShape, biasShape, numThreads, transSrc, transWei);
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
    {0, "FP16_FP16_FP16_FP16"},
    {1, "FP16_FP16_FP32_FP32"},
    {2, "INT8_INT8_INT32_INT32"},
    {3, "FP16_INT8_FP16_FP32"},
    {4, "BF16_BF16_BF16_FP32"},
    {5, "BF16_BF16_FP32_FP32"},
    {6, "FP32_FP32_FP32_FP32"},
    {7, "UINT8_INT8_INT32_INT32"},
    {8, "INT8_UINT8_INT32_INT32"},
    {9, "UINT8_UINT8_INT32_INT32"},
    {10, "FP16_UINT8_FP16_FP32"},
};

// numThreads input test
void throwException(int numThreads)
{
    int M = 5, N = 5, K = 5, D1 = 1, D2 = 1, D3 = 1;
    Shape srcShape(D1, D2, D3, M, K), weiShape(D1, D2, D3, K, N), dstShape(D1, D2, D3, M, N),
        biasShape(D1, D2, D3, M, N);
    Matmul<__fp16, __fp16, __fp16, __fp16>(srcShape, weiShape, dstShape, biasShape, numThreads);
}

static void InputParamExcept_FP16()
{
    int D1 = -1, D2 = 1, D3 = 1, M = 5, N = 5, K = 5;
    Shape srcShape(D1, D2, D3, M, K), weiShape(D1, D2, D3, K, N), dstShape(D1, D2, D3, M, N),
        biasShape(D1, D2, D3, M, N);
    Matmul<__fp16, __fp16, __fp16, __fp16>(srcShape, weiShape, dstShape, biasShape, 0);
}

static void InputParamExcept_FP16FP32()
{
    int D1 = 1, D2 = -1, D3 = 1, M = 5, N = 5, K = 5;
    Shape srcShape(D1, D2, D3, M, K), weiShape(D1, D2, D3, K, N), dstShape(D1, D2, D3, M, N),
        biasShape(D1, D2, D3, M, N);
    Matmul<__fp16, __fp16, float, float>(srcShape, weiShape, dstShape, biasShape, 0);
}

static void InputParamExcept_W8A8()
{
    int D1 = 1, D2 = 1, D3 = -1, M = 5, N = 5, K = 5;
    Shape srcShape(D1, D2, D3, M, K), weiShape(D1, D2, D3, K, N), dstShape(D1, D2, D3, M, N),
        biasShape(D1, D2, D3, M, N);
    Matmul<std::int8_t, std::int8_t, std::int32_t, std::int32_t>(srcShape, weiShape, dstShape, biasShape, 0);
}

static void InputParamExcept_W8A16()
{
    int D1 = 1, D2 = 1, D3 = 1, M = -5, N = 5, K = 5;
    Shape srcShape(D1, D2, D3, M, K), weiShape(D1, D2, D3, K, N), dstShape(D1, D2, D3, M, N),
        biasShape(D1, D2, D3, M, N);
    Matmul<__fp16, std::int8_t, __fp16, float>(srcShape, weiShape, dstShape, biasShape, 0);
}

static void InputParamExcept_BF16()
{
    int D1 = 1, D2 = 1, D3 = 1, M = 5, N = -5, K = 5;
    Shape srcShape(D1, D2, D3, M, K), weiShape(D1, D2, D3, K, N), dstShape(D1, D2, D3, M, N),
        biasShape(D1, D2, D3, M, N);
    Matmul<__bf16, __bf16, __bf16, float>(srcShape, weiShape, dstShape, biasShape, 0);
}

static void InputParamExcept_BF16FP32()
{
    int D1 = 1, D2 = -1, D3 = 1, M = -5, N = 5, K = 5;
    Shape srcShape(D1, D2, D3, M, K), weiShape(D1, D2, D3, K, N), dstShape(D1, D2, D3, M, N),
        biasShape(D1, D2, D3, M, N);
    Matmul<__bf16, __bf16, float, float>(srcShape, weiShape, dstShape, biasShape, 0);
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

TEST(MatmulGTests, Matmul2D_FP16_NumThreads_BadInput1_ExceedsUpperBound)
{
    EXPECT_THROW(throwException(10000000), KuDNN::Service::LogicError);
}

TEST(MatmulGTests, Matmul2D_FP16_NumThreads_BadInput2_ExceedsLowerBound)
{
    EXPECT_THROW(throwException(-20), KuDNN::Service::LogicError);
}

TEST(MatmulGTests, InputParamExcept_FP16)
{
    EXPECT_THROW(InputParamExcept_FP16(), KuDNN::Service::LogicError);
}

TEST(MatmulGTests, InputParamExcept_FP16FP32)
{
    EXPECT_THROW(InputParamExcept_FP16FP32(), KuDNN::Service::LogicError);
}

TEST(MatmulGTests, InputParamExcept_W8A8)
{
    EXPECT_THROW(InputParamExcept_W8A8(), KuDNN::Service::LogicError);
}

TEST(MatmulGTests, InputParamExcept_W8A16)
{
    EXPECT_THROW(InputParamExcept_W8A16(), KuDNN::Service::LogicError);
}

TEST(MatmulGTests, InputParamExcept_BF16)
{
    EXPECT_THROW(InputParamExcept_BF16(), KuDNN::Service::LogicError);
}

TEST(MatmulGTests, InputParamExcept_BF16FP32)
{
    EXPECT_THROW(InputParamExcept_BF16FP32(), KuDNN::Service::LogicError);
}

// M N K numThreads
static std::vector<std::vector<int>> common_MNK = {{5, 5, 5}, {20, 20, 10}, {200, 100, 200}};
static std::vector<std::vector<int>> common_Dims = {
    {1, 1, 1}, {1, 1, 10}, {1, 1, 20}, {1, 10, 10}, {1, 10, 20}, {5, 10, 10}, {5, 10, 20}};
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

std::string ParamName(const ::testing::TestParamInfo<TestParas> &info) {
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
    ::testing::Combine(::testing::ValuesIn(g_matmulCallersIndex), ::testing::ValuesIn(numThreads), ::testing::ValuesIn(numParas)),
    ParamName
    );

TEST_P(MTest, Matmul)
{
    int numThreads = std::get<1>(GetParam());
    size_t index = std::get<2>(GetParam());
    int matmulCallerIndex = std::get<0>(GetParam());

    if (index >= 0 && index < testParas.size()) {
        int D1 = testParas[index][0], D2 = testParas[index][1], D3 = testParas[index][2], M = testParas[index][3],
            N = testParas[index][4], K = testParas[index][5];
        Shape srcShape(D1, D2, D3, M, K), weiShape(D1, D2, D3, K, N), dstShape(D1, D2, D3, M, N),
            biasShape(D1, D2, D3, M, N);
        g_matmulCallers[matmulCallerIndex]->call(srcShape, weiShape, dstShape, biasShape, numThreads);
    }
}

TEST_P(MTest, NoJitMatmul)
{
    int numThreads = std::get<1>(GetParam());
    size_t index = std::get<2>(GetParam());
    int matmulCallerIndex = std::get<0>(GetParam());

    if (index >= 0 && index < testParas.size()/2) {
        int D1 = testParas[index][0], D2 = testParas[index][1], D3 = testParas[index][2], M = testParas[index][3],
            N = testParas[index][4], K = testParas[index][5];
        Shape srcShape(D1, D2, D3, M, K), weiShape(D1, D2, D3, K, N), dstShape(D1, D2, D3, M, N),
            biasShape(D1, D2, D3, M, N);
        setenv("KUDNN_NOT_USE_JIT", "1", 1);
        g_matmulCallers[matmulCallerIndex]->call(srcShape, weiShape, dstShape, biasShape, numThreads);
        setenv("KUDNN_NOT_USE_JIT", "0", 1);
    }
}

TEST_P(MTest, ThreadOmpMatmul)
{
    int numThreads = std::get<1>(GetParam());
    size_t index = std::get<2>(GetParam());
    int matmulCallerIndex = std::get<0>(GetParam());

    setenv("KUDNN_THREAD_TYPE", "OMP", 1);
    if (index >= 0 && index < testParas.size()/2) {
        int D1 = testParas[index][0], D2 = testParas[index][1], D3 = testParas[index][2], M = testParas[index][3],
            N = testParas[index][4], K = testParas[index][5];
        Shape srcShape(D1, D2, D3, M, K), weiShape(D1, D2, D3, K, N), dstShape(D1, D2, D3, M, N),
            biasShape(D1, D2, D3, M, N);
        g_matmulCallers[matmulCallerIndex]->call(srcShape, weiShape, dstShape, biasShape, numThreads);
    }
    setenv("KUDNN_THREAD_TYPE", "KUPL", 1);
}


TEST_P(MTest, MatmulTransSrc)
{
    int numThreads = std::get<1>(GetParam());
    size_t index = std::get<2>(GetParam());
    int matmulCallerIndex = std::get<0>(GetParam());

    if (index >= 0 && index < testParas.size()) {
        int D1 = testParas[index][0], D2 = testParas[index][1], D3 = testParas[index][2], M = testParas[index][3],
            N = testParas[index][4], K = testParas[index][5];
        Shape srcShape(D1, D2, D3, M, K), weiShape(D1, D2, D3, K, N), dstShape(D1, D2, D3, M, N),
            biasShape(D1, D2, D3, M, N);
        g_matmulCallers[matmulCallerIndex]->call(srcShape, weiShape, dstShape, biasShape, numThreads, 1, 0);
    }
}

TEST_P(MTest, MatmulTransWei)
{
    int numThreads = std::get<1>(GetParam());
    size_t index = std::get<2>(GetParam());
    int matmulCallerIndex = std::get<0>(GetParam());

    if (index >= 0 && index < testParas.size()) {
        int D1 = testParas[index][0], D2 = testParas[index][1], D3 = testParas[index][2], M = testParas[index][3],
            N = testParas[index][4], K = testParas[index][5];
        Shape srcShape(D1, D2, D3, M, K), weiShape(D1, D2, D3, K, N), dstShape(D1, D2, D3, M, N),
            biasShape(D1, D2, D3, M, N);
        g_matmulCallers[matmulCallerIndex]->call(srcShape, weiShape, dstShape, biasShape, numThreads, 0, 1);
    }
}

TEST_P(MTest, MatmulTransSrcWei)
{
    int numThreads = std::get<1>(GetParam());
    size_t index = std::get<2>(GetParam());
    int matmulCallerIndex = std::get<0>(GetParam());

    if (index >= 0 && index < testParas.size()) {
        int D1 = testParas[index][0], D2 = testParas[index][1], D3 = testParas[index][2], M = testParas[index][3],
            N = testParas[index][4], K = testParas[index][5];
        Shape srcShape(D1, D2, D3, M, K), weiShape(D1, D2, D3, K, N), dstShape(D1, D2, D3, M, N),
            biasShape(D1, D2, D3, M, N);
        g_matmulCallers[matmulCallerIndex]->call(srcShape, weiShape, dstShape, biasShape, numThreads, 1, 1);
    }
}

/*==============================MultiPerfTest==================================*/
// M N K numThreads
static std::vector<std::vector<int>> common_MNKMulti = {};
static std::vector<std::vector<int>> common_DimsMulti = {};
static std::vector<std::vector<int>> freParasMulti = {
    {1, 1, 1, 4096, 3072, 5120},
    {1, 1, 1, 4096, 1536, 5120},
    {1, 1, 1, 4096, 5120, 1536},
    {1, 1, 1, 4096, 5120, 3072},
    {1, 1, 1, 4096, 4096, 1152},
    {1, 1, 1, 4096, 4608, 1152},
    {1, 1, 1, 4096, 1152, 3256},
    {1, 1, 1, 4096, 1152, 4608}};

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
    ::testing::Combine(::testing::ValuesIn(g_matmulCallersIndex), ::testing::ValuesIn(numThreads), ::testing::ValuesIn(numParasMulti)),
    ParamName
    );

TEST_P(MTestMP, MatmulMultiPerf)
{
    int numThreads = std::get<1>(GetParam());
    size_t index = std::get<2>(GetParam());
    int matmulCallerIndex = std::get<0>(GetParam());
    std::string testname = g_nameMap[matmulCallerIndex];

    if (index >= 0 && index < testParasMulti.size()) {
        int D1 = testParasMulti[index][0], D2 = testParasMulti[index][1], D3 = testParasMulti[index][2],
        M = testParasMulti[index][3], N = testParasMulti[index][4], K = testParasMulti[index][5];
        g_writer.SetFilePath("./matmul_test_result_multi_" + testname + ".txt");
        g_writer.WriteData(testname + "_" + std::to_string(index) + ", ");
        Shape srcShape(D1, D2, D3, M, K), weiShape(D1, D2, D3, K, N), dstShape(D1, D2, D3, M, N),
            biasShape(D1, D2, D3, M, N);
        g_matmulCallers[matmulCallerIndex]->call(srcShape, weiShape, dstShape, biasShape, numThreads);
    }
}

/*=======================SinglePerfTest===========================*/
// M N K numThreads
static std::vector<std::vector<int>> common_MNKSin = {
    {512, 512, 192},
    {512, 1024, 192},
    {1024, 512, 192},
    {1024, 1024, 192},
    {512, 128, 512},
    {512, 128, 1024},
    {1024, 128, 512},
    {1024, 128, 1024}};
static std::vector<std::vector<int>> common_DimsSin = {{1, 1, 128}};
static std::vector<std::vector<int>> freParasSin = {};

static std::vector<int> numParasSin = GenerateNumParas(
    common_MNKSin.size() * common_DimsSin.size() + freParasSin.size()
);
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
    ::testing::Combine(::testing::ValuesIn(g_matmulCallersIndex), ::testing::ValuesIn(numThreads), ::testing::ValuesIn(numParasSin)),
    ParamName
    );

TEST_P(MTestSP, MatmulSinglePerf)
{
    int numThreads = std::get<1>(GetParam());
    size_t index = std::get<2>(GetParam());
    int matmulCallerIndex = std::get<0>(GetParam());
    std::string testname = g_nameMap[matmulCallerIndex];

    if (index >= 0 && index < testParasSin.size()) {
        int D1 = testParasSin[index][0], D2 = testParasSin[index][1], D3 = testParasSin[index][2],
        M = testParasSin[index][3], N = testParasSin[index][4], K = testParasSin[index][5];
        g_writer.SetFilePath("./matmul_test_result_single_" + testname + ".txt");
        g_writer.WriteData(testname + "_" + std::to_string(index) + ", ");
        Shape srcShape(D1, D2, D3, M, K), weiShape(D1, D2, D3, K, N), dstShape(D1, D2, D3, M, N),
            biasShape(D1, D2, D3, M, N);
        g_matmulCallers[matmulCallerIndex]->call(srcShape, weiShape, dstShape, biasShape, numThreads);
    }
}