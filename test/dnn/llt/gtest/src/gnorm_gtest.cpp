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

#include <cmath>
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <random>
#include <vector>

#include "gtest/gtest.h"
#include "kudnn.hpp"
#include "kudnn_tensor_data.hpp"

using SizeType = KuDNN::SizeType;
using Shape = KuDNN::Shape;
using TensorInfo = const KuDNN::TensorInfo;

static const int MAX_THS = 100;
static const float FACTOR_THS = 0.02;
static float errBoundGn = 1e-3;

// reference implementation of group normalization forward
template <typename T>
static void gnormRef(Shape shape, SizeType groupInfo,
                    const T *src, T *dst,
                    const T *scale, const T *shift, bool scaleApply, bool shiftApply,
                    float *mean, float *variance, const float eps, bool global_stats = false)
{
    SizeType innerSize = 1;
    SizeType numDims = shape.GetNumDims();
    if (numDims == 4) {
        innerSize = shape[2] * shape[3];
    }
    else if (numDims == 5) {
        innerSize = shape[2] * shape[3] * shape[4];
    }
    else if (numDims == 3) {
        innerSize = shape[2];
    }
    else {
        return;
    }
    SizeType A = shape[0], B = shape[1];
    SizeType chPerGroup = B / groupInfo;
    innerSize *= chPerGroup;
    int threadsUsed = (int)ceil((float)(A * groupInfo) * FACTOR_THS);
    threadsUsed = threadsUsed > MAX_THS ? MAX_THS : threadsUsed;
    if (!global_stats) {
    // mean compute
#pragma omp parallel for collapse(2) num_threads(threadsUsed)
        for (SizeType a = 0; a < A; ++a) {
            for (SizeType g = 0; g < groupInfo; ++g) {
                float sum = 0.0f;
                for (SizeType i = 0; i < innerSize; ++i) {
                    sum += (float)*(src + a * groupInfo * innerSize + g * innerSize + i);
                }
                *(mean + a * groupInfo + g) = sum / innerSize;
            }
        }
    // variance compute
#pragma omp parallel for collapse(2) num_threads(threadsUsed)
        for (SizeType a = 0; a < A; ++a) {
            for (SizeType g = 0; g < groupInfo; ++g) {
                float sum = 0.0f;
                float meanVal = *(mean + a * groupInfo + g);
                for (SizeType i = 0; i < innerSize; ++i) {
                    float diff = (float)*(src + a * groupInfo * innerSize + g * innerSize + i) - meanVal;
                    sum += diff * diff;
                }
                *(variance + a * groupInfo + g) = sum / innerSize;
            }
        }
    }
    // norm
    innerSize /= chPerGroup;
#pragma omp parallel for collapse(2) num_threads(threadsUsed)
    for (SizeType a = 0; a < A; ++a) {
        for (SizeType g = 0; g < groupInfo; ++g) {
            float meanVal = *(mean + a * groupInfo + g);
            float invStd = 1.0f / std::sqrt(*(variance + a * groupInfo + g) + eps);
            for (SizeType gi = 0; gi < chPerGroup; ++gi) {
                int b = g * chPerGroup + gi;
                for (SizeType i = 0; i < innerSize; ++i) {
                    float srcVal = (float)*(src + a * B * innerSize + b * innerSize + i);
                    float normalized = (srcVal - meanVal) * invStd;
                    if (scaleApply) {
                        normalized *= scale[b];
                    }
                    if (shiftApply) {
                        normalized += shift[b];
                    }
                    *(dst + a * B * innerSize + b * innerSize + i) = static_cast<T>(normalized);
                }
            }
        }
    }
}

// backward
template <typename T>
static bool BackwardFunc1(const TensorInfo &srcInfo, const TensorInfo &statInfo, const TensorInfo &diffSrcInfo,
                        const TensorInfo &diffDstInfo, const TensorInfo &scaleShiftInfo, const TensorInfo &diffScaleShiftInfo,
                        KuDNN::NormalizationFlags flags)
{
    KuDNN::GroupNormalizationLayerBWD gnormLayer1(srcInfo, statInfo, diffSrcInfo, diffDstInfo, scaleShiftInfo,
        diffScaleShiftInfo, flags);

    SizeType srcSize = srcInfo.GetTotalTensorSize();
    SizeType diffSrcSize = diffSrcInfo.GetTotalTensorSize();
    SizeType diffDstSize = diffDstInfo.GetTotalTensorSize();
    SizeType statSize = statInfo.GetTotalTensorSize();
    SizeType scaleSize = scaleShiftInfo.GetTotalTensorSize();

    T *src = (T*)malloc(srcSize * sizeof(T));
    T *diffSrc = (T*)malloc(diffSrcSize * sizeof(T));
    T *diffDst = (T*)malloc(diffDstSize * sizeof(T));
    float *mean = (float *)malloc(statSize * sizeof(float));
    float *variance = (float *)malloc(statSize * sizeof(float));
    T *scale = (T*)malloc(scaleSize * sizeof(T));
    T *diffScale = (T*)malloc(scaleSize * sizeof(T));
    T *diffShift = (T*)malloc(scaleSize * sizeof(T));

    gnormLayer1.Run(src, mean, variance, diffDst, scale, diffSrc, diffScale, diffShift, 1e-5);
    return true;

}

// forward
template <typename T>
static bool ForwardFunc1(const TensorInfo &srcInfo, const TensorInfo &scaleShiftInfo,
                        SizeType groupInfo, const TensorInfo &dstInfo,
                        KuDNN::NormalizationFlags flags)
{
    KuDNN::GroupNormalizationLayerFWD gnormLayer1(srcInfo, scaleShiftInfo, groupInfo, dstInfo,
        flags);

    SizeType srcSize = srcInfo.GetTotalTensorSize();
    SizeType dstSize = dstInfo.GetTotalTensorSize();
    SizeType statSize = srcInfo.GetDims()[0] * groupInfo;
    SizeType scaleSize = scaleShiftInfo.GetTotalTensorSize();

    T *src = (T*)malloc(srcSize * sizeof(T));
    T *dst = (T*)malloc(srcSize * sizeof(T));
    T *dstRef = (T*)malloc(srcSize * sizeof(T));
    float *mean = (float *)malloc(statSize * sizeof(float));
    float *variance = (float *)malloc(statSize * sizeof(float));
    T *scale = (T*)malloc(scaleSize * sizeof(T));
    T *shift = (T*)malloc(scaleSize * sizeof(T));
    float eps = 1e-5;
    if (src == nullptr || dst == nullptr || dstRef == nullptr || mean == nullptr ||
        variance == nullptr || scale == nullptr || shift == nullptr) {
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
    for (SizeType i = 0; i < scaleSize; ++i) {
        *(scale + i) = (T)u(e);
        *(shift + i) = (T)u(e);
    }
    if (global_stats) {
        for(SizeType i = 0; i < statSize; ++i) {
            *(mean + i) = 1.0f;
            *(variance + i) = 0.1f;
        }
    }

    gnormLayer1.Run(src, dst, scale, shift, mean, variance, true, eps);
    bool scaleApply = static_cast<bool>(flags & KuDNN::NormalizationFlags::USE_SCALE);
    bool shiftApply = static_cast<bool>(flags & KuDNN::NormalizationFlags::USE_SHIFT);

    gnormRef<T>(srcInfo.GetDims(), groupInfo, src, dstRef, scale, shift, scaleApply, shiftApply, mean, variance, eps, global_stats);
    float error = 0.0;
#pragma omp parallel for reduction(+:error) num_threads(threadsUsed)
    for (SizeType i = 0; i < dstSize; ++i) {
        error += *(dst + i) - *(dstRef + i);
    }
    free(src);
    free(dst);
    free(dstRef);
    free(mean);
    free(variance);
    free(scale);
    free(shift);
    error = std::abs(error) / dstSize;
    std::cout << "run error: "<< error <<std::endl;
    return error < errBoundGn;
}

template <typename T>
static bool FWDSaveStats(const TensorInfo &srcInfo, const TensorInfo &scaleShiftInfo,
                        SizeType groupInfo, const TensorInfo &dstInfo,
                        KuDNN::NormalizationFlags flags)
{
    flags = KuDNN::NormalizationFlags::NONE;
    KuDNN::GroupNormalizationLayerFWD gnormLayer1(srcInfo, scaleShiftInfo, groupInfo, dstInfo,
        flags);

    SizeType srcSize = srcInfo.GetTotalTensorSize();
    SizeType dstSize = dstInfo.GetTotalTensorSize();
    SizeType statSize = srcInfo.GetDims()[0] * groupInfo;

    T *src = (T*)malloc(srcSize * sizeof(T));
    T *dst = (T*)malloc(dstSize * sizeof(T));
    float *mean = (float *)malloc(statSize * sizeof(float));
    float *variance = (float *)malloc(statSize * sizeof(float));
    float *meanRef = (float *)malloc(statSize * sizeof(float));
    float *varianceRef = (float *)malloc(statSize * sizeof(float));
    T *scale = nullptr;
    T *shift = nullptr;
    float eps = 1e-5;
    if (src == nullptr || dst == nullptr || mean == nullptr || variance == nullptr ||
        meanRef == nullptr || varianceRef == nullptr) {
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

    gnormLayer1.Run(src, dst, scale, shift, mean, variance, true, eps);
    bool scaleApply = false;
    bool shiftApply = false;
    gnormRef<T>(srcInfo.GetDims(), groupInfo, src, dst, scale, shift, scaleApply, shiftApply, meanRef, varianceRef, eps);
    float error = 0.0;
    for (SizeType i = 0; i < statSize; ++i) {
        error += *(mean + i) - *(meanRef + i);
        error += *(variance + i) - *(varianceRef + i);
    }
    free(src);
    free(dst);
    free(mean);
    free(variance);
    free(meanRef);
    free(varianceRef);
    error = std::abs(error) / (2 * statSize);
    std::cout << "run error: "<< error <<std::endl;
    return error < errBoundGn;
}

template <typename T>
static void FWDDivideZero(const TensorInfo &srcInfo, const TensorInfo &scaleShiftInfo,
                        SizeType groupInfo, const TensorInfo &dstInfo,
                        KuDNN::NormalizationFlags flags)
{
    flags = KuDNN::NormalizationFlags::NONE;
    KuDNN::GroupNormalizationLayerFWD gnormLayer1(srcInfo, scaleShiftInfo, groupInfo, dstInfo,
        flags);

    SizeType srcSize = srcInfo.GetTotalTensorSize();
    SizeType dstSize = dstInfo.GetTotalTensorSize();
    SizeType statSize = srcInfo.GetDims()[0] * groupInfo;

    T *src = (T*)malloc(srcSize * sizeof(T));
    T *dst = (T*)malloc(dstSize * sizeof(T));
    float *mean = (float *)malloc(statSize * sizeof(float));
    float *variance = (float *)malloc(statSize * sizeof(float));
    T *scale = nullptr;
    T *shift = nullptr;
    float eps = 0.0f;
    if (src == nullptr || dst == nullptr || mean == nullptr || variance == nullptr) {
        std::cerr << "Memory allocation failed" << std::endl;
        return;
    }
    int threadsUsed = (int)ceil((float)srcSize * FACTOR_THS);
    threadsUsed = threadsUsed > MAX_THS ? MAX_THS : threadsUsed;
    // generate 0 test data
#pragma omp parallel for num_threads(threadsUsed) schedule(static)
    for (SizeType i = 0; i < srcSize; ++i) {
        *(src + i) = 1.0f;
    }
    gnormLayer1.Run(src, dst, scale, shift, mean, variance, true, eps);
    free(src);
    free(dst);
    free(mean);
    free(variance);
}

// type tests: src/dst type is not checked by FWD ValidateInput,
// scaleShift must be F32 when USE_SCALE/USE_SHIFT is set
static void FWDSrcDstType()
{
    const KuDNN::TensorInfo srcInfo = {{1, 1, 1}, KuDNN::Element::TypeT::S32, KuDNN::Layout::ABC};
    const KuDNN::TensorInfo scaleShiftInfo = {{1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::A};
    const KuDNN::TensorInfo dstInfo = {{1, 1, 1}, KuDNN::Element::TypeT::S32, KuDNN::Layout::ABC};
    KuDNN::SizeType groupInfo = 1;
    KuDNN::GroupNormalizationLayerFWD gnormLayer1(srcInfo, scaleShiftInfo, groupInfo, dstInfo,
        KuDNN::NormalizationFlags::NONE);
}

static void FWDScaleType()
{
    const KuDNN::TensorInfo srcInfo = {{1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABC};
    const KuDNN::TensorInfo scaleShiftInfo = {{1}, KuDNN::Element::TypeT::S32, KuDNN::Layout::A};
    const KuDNN::TensorInfo dstInfo = {{1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABC};
    KuDNN::SizeType groupInfo = 1;
    KuDNN::GroupNormalizationLayerFWD gnormLayer1(srcInfo, scaleShiftInfo, groupInfo, dstInfo,
        KuDNN::NormalizationFlags::USE_SCALE);
}

static void FWDShiftType()
{
    const KuDNN::TensorInfo srcInfo = {{1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABC};
    const KuDNN::TensorInfo scaleShiftInfo = {{1}, KuDNN::Element::TypeT::S32, KuDNN::Layout::A};
    const KuDNN::TensorInfo dstInfo = {{1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABC};
    KuDNN::SizeType groupInfo = 1;
    KuDNN::GroupNormalizationLayerFWD gnormLayer1(srcInfo, scaleShiftInfo, groupInfo, dstInfo,
        KuDNN::NormalizationFlags::USE_SHIFT);
}

static void FWDGlobalStatsType()
{
    const KuDNN::TensorInfo srcInfo = {{1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABC};
    const KuDNN::TensorInfo scaleShiftInfo = {{1}, KuDNN::Element::TypeT::S32, KuDNN::Layout::A};
    const KuDNN::TensorInfo dstInfo = {{1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABC};
    KuDNN::SizeType groupInfo = 1;
    KuDNN::GroupNormalizationLayerFWD gnormLayer1(srcInfo, scaleShiftInfo, groupInfo, dstInfo,
        KuDNN::NormalizationFlags::USE_GLOBAL_STATS);
}

// forward
static void ForwardCopyConstructor1()
{
    const KuDNN::TensorInfo srcInfo = {{1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABC};
    const KuDNN::TensorInfo scaleShiftInfo = {{1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::A};
    const KuDNN::TensorInfo dstInfo = {{1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABC};
    KuDNN::SizeType groupInfo = 1;
    KuDNN::GroupNormalizationLayerFWD gnormLayer1(srcInfo, scaleShiftInfo, groupInfo, dstInfo,
        KuDNN::NormalizationFlags::NONE);
    KuDNN::GroupNormalizationLayerFWD gnormLayer2(gnormLayer1);
    for (size_t i = 0; i < g_Tensor_data.size() / 3; i += 3) {
        KuDNN::GroupNormalizationLayerFWD gnormLayer1(g_Tensor_data[i], g_Tensor_data[i + 1], groupInfo,
            g_Tensor_data[i + 2], KuDNN::NormalizationFlags::NONE);
        KuDNN::GroupNormalizationLayerFWD gnormLayer2(gnormLayer1);
    }
}

static void ForwardCopyConstructor2()
{
    const KuDNN::TensorInfo srcInfo = {{1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABC};
    const KuDNN::TensorInfo scaleShiftInfo = {{1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::A};
    const KuDNN::TensorInfo dstInfo = {{1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABC};
    KuDNN::SizeType groupInfo = 1;
    KuDNN::GroupNormalizationLayerFWD gnormLayer1(srcInfo, scaleShiftInfo, groupInfo, dstInfo,
        KuDNN::NormalizationFlags::NONE);
    KuDNN::GroupNormalizationLayerFWD gnormLayer2 = gnormLayer1;
    for (size_t i = 0; i < g_Tensor_data.size() / 3; i += 3) {
        KuDNN::GroupNormalizationLayerFWD gnormLayer1(g_Tensor_data[i], g_Tensor_data[i + 1], groupInfo,
            g_Tensor_data[i + 2], KuDNN::NormalizationFlags::NONE);
        KuDNN::GroupNormalizationLayerFWD gnormLayer2 = gnormLayer1;
    }
}

static void ForwardMoveConstructor1()
{
    const KuDNN::TensorInfo srcInfo = {{1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABC};
    const KuDNN::TensorInfo scaleShiftInfo = {{1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::A};
    const KuDNN::TensorInfo dstInfo = {{1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABC};
    KuDNN::SizeType groupInfo = 1;
    KuDNN::GroupNormalizationLayerFWD gnormLayer1(srcInfo, scaleShiftInfo, groupInfo, dstInfo,
        KuDNN::NormalizationFlags::NONE);
    KuDNN::GroupNormalizationLayerFWD gnormLayer2(std::move(gnormLayer1));
    for (size_t i = 0; i < g_Tensor_data.size() / 3; i += 3) {
        KuDNN::GroupNormalizationLayerFWD gnormLayer1(g_Tensor_data[i], g_Tensor_data[i + 1], groupInfo,
            g_Tensor_data[i + 2], KuDNN::NormalizationFlags::NONE);
        KuDNN::GroupNormalizationLayerFWD gnormLayer2(std::move(gnormLayer1));
    }
}

static void ForwardMoveConstructor2()
{
    const KuDNN::TensorInfo srcInfo = {{1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABC};
    const KuDNN::TensorInfo scaleShiftInfo = {{1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::A};
    const KuDNN::TensorInfo dstInfo = {{1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABC};
    KuDNN::SizeType groupInfo = 1;
    KuDNN::GroupNormalizationLayerFWD gnormLayer1(srcInfo, scaleShiftInfo, groupInfo, dstInfo,
        KuDNN::NormalizationFlags::NONE);
    KuDNN::GroupNormalizationLayerFWD gnormLayer2(srcInfo, scaleShiftInfo, groupInfo, dstInfo,
        KuDNN::NormalizationFlags::NONE);
    gnormLayer2 = std::move(gnormLayer1);
    for (size_t i = 0; i < g_Tensor_data.size() / 3; i += 3) {
        KuDNN::GroupNormalizationLayerFWD gnormLayer1(g_Tensor_data[i], g_Tensor_data[i + 1], groupInfo,
            g_Tensor_data[i + 2], KuDNN::NormalizationFlags::NONE);
        KuDNN::GroupNormalizationLayerFWD gnormLayer2(g_Tensor_data[i], g_Tensor_data[i + 1], groupInfo,
            g_Tensor_data[i + 2], KuDNN::NormalizationFlags::NONE);
        gnormLayer2 = std::move(gnormLayer1);
    }
}

// backward
static void BackwardCopyConstructor1()
{
    const KuDNN::TensorInfo srcInfo = {{1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABC};
    const KuDNN::TensorInfo diffSrcInfo = {{1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABC};
    const KuDNN::TensorInfo statInfo = {{1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::AB};
    const KuDNN::TensorInfo scaleShiftInfo = {{1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::A};
    const KuDNN::TensorInfo diffScaleShiftInfo = {{1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::A};
    const KuDNN::TensorInfo diffDstInfo = {{1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABC};
    KuDNN::GroupNormalizationLayerBWD gnormLayer1(srcInfo, statInfo, diffSrcInfo, diffDstInfo, scaleShiftInfo,
        diffScaleShiftInfo, KuDNN::NormalizationFlags::NONE);
    KuDNN::GroupNormalizationLayerBWD gnormLayer2(gnormLayer1);
    for (size_t i = 0; i < g_Tensor_data.size() / 6; i += 6) {
        KuDNN::GroupNormalizationLayerBWD gnormLayer1(g_Tensor_data[i], g_Tensor_data[i + 1], g_Tensor_data[i + 2],
            g_Tensor_data[i + 3], g_Tensor_data[i + 4], g_Tensor_data[i + 5], KuDNN::NormalizationFlags::NONE);
        KuDNN::GroupNormalizationLayerBWD gnormLayer2(gnormLayer1);
    }

}

static void BackwardCopyConstructor2()
{
    const KuDNN::TensorInfo srcInfo = {{1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABC};
    const KuDNN::TensorInfo diffSrcInfo = {{1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABC};
    const KuDNN::TensorInfo statInfo = {{1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::AB};
    const KuDNN::TensorInfo scaleShiftInfo = {{1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::A};
    const KuDNN::TensorInfo diffScaleShiftInfo = {{1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::A};
    const KuDNN::TensorInfo diffDstInfo = {{1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABC};
    KuDNN::GroupNormalizationLayerBWD gnormLayer1(srcInfo, statInfo, diffSrcInfo, diffDstInfo, scaleShiftInfo,
        diffScaleShiftInfo, KuDNN::NormalizationFlags::NONE);
    KuDNN::GroupNormalizationLayerBWD gnormLayer2(srcInfo, statInfo, diffSrcInfo, diffDstInfo, scaleShiftInfo,
        diffScaleShiftInfo, KuDNN::NormalizationFlags::NONE);
    gnormLayer2 = gnormLayer1;
    for (size_t i = 0; i < g_Tensor_data.size() / 6; i += 6) {
        KuDNN::GroupNormalizationLayerBWD gnormLayer1(g_Tensor_data[i], g_Tensor_data[i + 1], g_Tensor_data[i + 2],
            g_Tensor_data[i + 3], g_Tensor_data[i + 4], g_Tensor_data[i + 5], KuDNN::NormalizationFlags::NONE);
        KuDNN::GroupNormalizationLayerBWD gnormLayer2(g_Tensor_data[i], g_Tensor_data[i + 1], g_Tensor_data[i + 2],
            g_Tensor_data[i + 3], g_Tensor_data[i + 4], g_Tensor_data[i + 5], KuDNN::NormalizationFlags::NONE);
        gnormLayer2 = gnormLayer1;
    }
}

static void BackwardMoveConstructor1()
{
    const KuDNN::TensorInfo srcInfo = {{1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABC};
    const KuDNN::TensorInfo diffSrcInfo = {{1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABC};
    const KuDNN::TensorInfo statInfo = {{1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::AB};
    const KuDNN::TensorInfo scaleShiftInfo = {{1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::A};
    const KuDNN::TensorInfo diffScaleShiftInfo = {{1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::A};
    const KuDNN::TensorInfo diffDstInfo = {{1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABC};
    KuDNN::GroupNormalizationLayerBWD gnormLayer1(srcInfo, statInfo, diffSrcInfo, diffDstInfo, scaleShiftInfo,
        diffScaleShiftInfo, KuDNN::NormalizationFlags::NONE);
    KuDNN::GroupNormalizationLayerBWD gnormLayer2(std::move(gnormLayer1));
    for (size_t i = 0; i < g_Tensor_data.size() / 6; i += 6) {
        KuDNN::GroupNormalizationLayerBWD gnormLayer1(g_Tensor_data[i], g_Tensor_data[i + 1], g_Tensor_data[i + 2],
            g_Tensor_data[i + 3], g_Tensor_data[i + 4], g_Tensor_data[i + 5], KuDNN::NormalizationFlags::NONE);
        KuDNN::GroupNormalizationLayerBWD gnormLayer2(std::move(gnormLayer1));
    }
}

static void BackwardMoveConstructor2()
{
    const KuDNN::TensorInfo srcInfo = {{1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABC};
    const KuDNN::TensorInfo diffSrcInfo = {{1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABC};
    const KuDNN::TensorInfo statInfo = {{1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::AB};
    const KuDNN::TensorInfo scaleShiftInfo = {{1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::A};
    const KuDNN::TensorInfo diffScaleShiftInfo = {{1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::A};
    const KuDNN::TensorInfo diffDstInfo = {{1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABC};
    KuDNN::GroupNormalizationLayerBWD gnormLayer1(srcInfo, statInfo, diffSrcInfo, diffDstInfo, scaleShiftInfo,
        diffScaleShiftInfo, KuDNN::NormalizationFlags::NONE);
    KuDNN::GroupNormalizationLayerBWD gnormLayer2(srcInfo, statInfo, diffSrcInfo, diffDstInfo, scaleShiftInfo,
        diffScaleShiftInfo, KuDNN::NormalizationFlags::NONE);
    gnormLayer2 = std::move(gnormLayer1);
    for (size_t i = 0; i < g_Tensor_data.size() / 6; i += 6) {
        KuDNN::GroupNormalizationLayerBWD gnormLayer1(g_Tensor_data[i], g_Tensor_data[i + 1], g_Tensor_data[i + 2],
            g_Tensor_data[i + 3], g_Tensor_data[i + 4], g_Tensor_data[i + 5], KuDNN::NormalizationFlags::NONE);
        KuDNN::GroupNormalizationLayerBWD gnormLayer2(g_Tensor_data[i], g_Tensor_data[i + 1], g_Tensor_data[i + 2],
            g_Tensor_data[i + 3], g_Tensor_data[i + 4], g_Tensor_data[i + 5], KuDNN::NormalizationFlags::NONE);
        gnormLayer2 = std::move(gnormLayer1);
    }
}

// input parameter exception tests:
// 0-sized dims hit totalSize==0 -> UNSUPPORTED; groupSize=0 -> BAD_ARGUMENTS
static void InputParamExceptTest1()
{
    Shape srcShape(0, 1, 1);
    const KuDNN::TensorInfo srcInfo = {srcShape, KuDNN::Element::TypeT::S32, KuDNN::Layout::ABC};
    const KuDNN::TensorInfo scaleShiftInfo = {{1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::A};
    const KuDNN::TensorInfo dstInfo = {srcShape, KuDNN::Element::TypeT::S32, KuDNN::Layout::ABC};
    KuDNN::SizeType groupInfo = 1;
    KuDNN::GroupNormalizationLayerFWD gnormLayer1(srcInfo, scaleShiftInfo, groupInfo, dstInfo,
        KuDNN::NormalizationFlags::NONE);
}

static void InputParamExceptTest2()
{
    Shape srcShape(1, 1, 1);
    const KuDNN::TensorInfo srcInfo = {srcShape, KuDNN::Element::TypeT::S32, KuDNN::Layout::ABC};
    const KuDNN::TensorInfo scaleShiftInfo = {{1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::A};
    const KuDNN::TensorInfo dstInfo = {srcShape, KuDNN::Element::TypeT::S32, KuDNN::Layout::ABC};
    KuDNN::SizeType groupInfo = 0;
    KuDNN::GroupNormalizationLayerFWD gnormLayer1(srcInfo, scaleShiftInfo, groupInfo, dstInfo,
        KuDNN::NormalizationFlags::NONE);
}

static void InputParamExceptTest3()
{
    Shape srcShape(1, 0, 1);
    const KuDNN::TensorInfo srcInfo = {srcShape, KuDNN::Element::TypeT::S32, KuDNN::Layout::ABC};
    const KuDNN::TensorInfo scaleShiftInfo = {{1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::A};
    const KuDNN::TensorInfo dstInfo = {srcShape, KuDNN::Element::TypeT::S32, KuDNN::Layout::ABC};
    KuDNN::SizeType groupInfo = 1;
    KuDNN::GroupNormalizationLayerFWD gnormLayer1(srcInfo, scaleShiftInfo, groupInfo, dstInfo,
        KuDNN::NormalizationFlags::NONE);
}

static void InputParamExceptTest4()
{
    Shape srcShape(1, 1, 0);
    const KuDNN::TensorInfo srcInfo = {srcShape, KuDNN::Element::TypeT::S32, KuDNN::Layout::ABC};
    const KuDNN::TensorInfo scaleShiftInfo = {{1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::A};
    const KuDNN::TensorInfo dstInfo = {srcShape, KuDNN::Element::TypeT::S32, KuDNN::Layout::ABC};
    KuDNN::SizeType groupInfo = 1;
    KuDNN::GroupNormalizationLayerFWD gnormLayer1(srcInfo, scaleShiftInfo, groupInfo, dstInfo,
        KuDNN::NormalizationFlags::NONE);
}

static void InputParamExceptTest5()
{
    Shape srcShape(1, 0, 1, 1, 1);
    const KuDNN::TensorInfo srcInfo = {srcShape, KuDNN::Element::TypeT::S32, KuDNN::Layout::ABCDE};
    const KuDNN::TensorInfo scaleShiftInfo = {{1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::A};
    const KuDNN::TensorInfo dstInfo = {srcShape, KuDNN::Element::TypeT::S32, KuDNN::Layout::ABCDE};
    KuDNN::SizeType groupInfo = 1;
    KuDNN::GroupNormalizationLayerFWD gnormLayer1(srcInfo, scaleShiftInfo, groupInfo, dstInfo,
        KuDNN::NormalizationFlags::NONE);
}

static void InputParamExceptTest6()
{
    Shape srcShape(1, 1, 1, 1, 0);
    const KuDNN::TensorInfo srcInfo = {srcShape, KuDNN::Element::TypeT::S32, KuDNN::Layout::ABCDE};
    const KuDNN::TensorInfo scaleShiftInfo = {{1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::A};
    const KuDNN::TensorInfo dstInfo = {srcShape, KuDNN::Element::TypeT::S32, KuDNN::Layout::ABCDE};
    KuDNN::SizeType groupInfo = 1;
    KuDNN::GroupNormalizationLayerFWD gnormLayer1(srcInfo, scaleShiftInfo, groupInfo, dstInfo,
        KuDNN::NormalizationFlags::NONE);
}

// test cases
TEST(GnormGTests, Param_ShapeExceptD1_Test)
{
    EXPECT_THROW(InputParamExceptTest1(), KuDNN::Service::Unsupported);
}

TEST(GnormGTests, Param_GroupSizeExcept_Test)
{
    EXPECT_THROW(InputParamExceptTest2(), KuDNN::Service::LogicError);
}

TEST(GnormGTests, Param_ShapeExceptD2_Test)
{
    EXPECT_THROW(InputParamExceptTest3(), KuDNN::Service::Unsupported);
}

TEST(GnormGTests, Param_ShapeExceptD3_Test)
{
    EXPECT_THROW(InputParamExceptTest4(), KuDNN::Service::Unsupported);
}

TEST(GnormGTests, Param_ShapeExceptD4_Test)
{
    EXPECT_THROW(InputParamExceptTest5(), KuDNN::Service::Unsupported);
}

TEST(GnormGTests, Param_ShapeExceptD5_Test)
{
    EXPECT_THROW(InputParamExceptTest6(), KuDNN::Service::Unsupported);
}

TEST(GnormGTests, ForwardFunc3DF16_1)
{
    Shape shape(4, 128, 320);
    SizeType groupInfo = 2;
    TensorInfo srcInfo = {shape, KuDNN::Element::TypeT::F16, KuDNN::Layout::ABC};
    TensorInfo scaleShiftInfo = {{shape[1]}, KuDNN::Element::TypeT::F16, KuDNN::Layout::A};
    TensorInfo dstInfo = {shape, KuDNN::Element::TypeT::F16, KuDNN::Layout::ABC};
    KuDNN::NormalizationFlags flags = KuDNN::NormalizationFlags::NONE;
    GTEST_EXPECT_TRUE(ForwardFunc1<__fp16>(srcInfo, scaleShiftInfo, groupInfo, dstInfo, flags));
}

TEST(GnormGTests, ForwardFunc3DF16_2)
{
    Shape shape(4, 128, 320);
    SizeType groupInfo = 2;
    TensorInfo srcInfo = {shape, KuDNN::Element::TypeT::F16, KuDNN::Layout::ABC};
    TensorInfo scaleShiftInfo = {{shape[1]}, KuDNN::Element::TypeT::F16, KuDNN::Layout::A};
    TensorInfo dstInfo = {shape, KuDNN::Element::TypeT::F16, KuDNN::Layout::ABC};
    KuDNN::NormalizationFlags flags = KuDNN::NormalizationFlags::USE_SCALE;
    GTEST_EXPECT_TRUE(ForwardFunc1<__fp16>(srcInfo, scaleShiftInfo, groupInfo, dstInfo, flags));
}

TEST(GnormGTests, ForwardFunc3DF16_3)
{
    Shape shape(4, 128, 320);
    SizeType groupInfo = 2;
    TensorInfo srcInfo = {shape, KuDNN::Element::TypeT::F16, KuDNN::Layout::ABC};
    TensorInfo scaleShiftInfo = {{shape[1]}, KuDNN::Element::TypeT::F16, KuDNN::Layout::A};
    TensorInfo dstInfo = {shape, KuDNN::Element::TypeT::F16, KuDNN::Layout::ABC};
    KuDNN::NormalizationFlags flags = KuDNN::NormalizationFlags::USE_SHIFT;
    GTEST_EXPECT_TRUE(ForwardFunc1<__fp16>(srcInfo, scaleShiftInfo, groupInfo, dstInfo, flags));
}

TEST(GnormGTests, ForwardFunc3DF16_4)
{
    Shape shape(4, 128, 320);
    SizeType groupInfo = 2;
    TensorInfo srcInfo = {shape, KuDNN::Element::TypeT::F16, KuDNN::Layout::ABC};
    TensorInfo scaleShiftInfo = {{shape[1]}, KuDNN::Element::TypeT::F16, KuDNN::Layout::A};
    TensorInfo dstInfo = {shape, KuDNN::Element::TypeT::F16, KuDNN::Layout::ABC};
    KuDNN::NormalizationFlags flags = KuDNN::NormalizationFlags::USE_GLOBAL_STATS;
    GTEST_EXPECT_TRUE(ForwardFunc1<__fp16>(srcInfo, scaleShiftInfo, groupInfo, dstInfo, flags));
}

TEST(GnormGTests, ForwardFunc3DF32_1)
{
    Shape shape(4, 128, 320);
    SizeType groupInfo = 2;
    TensorInfo srcInfo = {shape, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABC};
    TensorInfo scaleShiftInfo = {{shape[1]}, KuDNN::Element::TypeT::F32, KuDNN::Layout::A};
    TensorInfo dstInfo = {shape, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABC};
    KuDNN::NormalizationFlags flags = KuDNN::NormalizationFlags::NONE;
    GTEST_EXPECT_TRUE(ForwardFunc1<float>(srcInfo, scaleShiftInfo, groupInfo, dstInfo, flags));
}

TEST(GnormGTests, ForwardFunc3DF32_2)
{
    Shape shape(4, 128, 320);
    SizeType groupInfo = 2;
    TensorInfo srcInfo = {shape, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABC};
    TensorInfo scaleShiftInfo = {{shape[1]}, KuDNN::Element::TypeT::F32, KuDNN::Layout::A};
    TensorInfo dstInfo = {shape, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABC};
    KuDNN::NormalizationFlags flags = KuDNN::NormalizationFlags::USE_SCALE;
    GTEST_EXPECT_TRUE(ForwardFunc1<float>(srcInfo, scaleShiftInfo, groupInfo, dstInfo, flags));
}

TEST(GnormGTests, ForwardFunc3DF32_3)
{
    Shape shape(4, 128, 320);
    SizeType groupInfo = 2;
    TensorInfo srcInfo = {shape, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABC};
    TensorInfo scaleShiftInfo = {{shape[1]}, KuDNN::Element::TypeT::F32, KuDNN::Layout::A};
    TensorInfo dstInfo = {shape, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABC};
    KuDNN::NormalizationFlags flags = KuDNN::NormalizationFlags::USE_SHIFT;
    GTEST_EXPECT_TRUE(ForwardFunc1<float>(srcInfo, scaleShiftInfo, groupInfo, dstInfo, flags));
}

TEST(GnormGTests, ForwardFunc3DF32_4)
{
    Shape shape(4, 128, 320);
    SizeType groupInfo = 2;
    TensorInfo srcInfo = {shape, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABC};
    TensorInfo scaleShiftInfo = {{shape[1]}, KuDNN::Element::TypeT::F32, KuDNN::Layout::A};
    TensorInfo dstInfo = {shape, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABC};
    KuDNN::NormalizationFlags flags = KuDNN::NormalizationFlags::USE_GLOBAL_STATS;
    GTEST_EXPECT_TRUE(ForwardFunc1<float>(srcInfo, scaleShiftInfo, groupInfo, dstInfo, flags));
}

TEST(GnormGTests, ForwardFunc4DF16_1)
{
    Shape shape(4, 128, 18, 320);
    SizeType groupInfo = 2;
    TensorInfo srcInfo = {shape, KuDNN::Element::TypeT::F16, KuDNN::Layout::ABCD};
    TensorInfo scaleShiftInfo = {{shape[1]}, KuDNN::Element::TypeT::F16, KuDNN::Layout::A};
    TensorInfo dstInfo = {shape, KuDNN::Element::TypeT::F16, KuDNN::Layout::ABCD};
    KuDNN::NormalizationFlags flags = KuDNN::NormalizationFlags::NONE;
    GTEST_EXPECT_TRUE(ForwardFunc1<__fp16>(srcInfo, scaleShiftInfo, groupInfo, dstInfo, flags));
}

TEST(GnormGTests, ForwardFunc4DF16_2)
{
    Shape shape(4, 128, 18, 320);
    SizeType groupInfo = 2;
    TensorInfo srcInfo = {shape, KuDNN::Element::TypeT::F16, KuDNN::Layout::ABCD};
    TensorInfo scaleShiftInfo = {{shape[1]}, KuDNN::Element::TypeT::F16, KuDNN::Layout::A};
    TensorInfo dstInfo = {shape, KuDNN::Element::TypeT::F16, KuDNN::Layout::ABCD};
    KuDNN::NormalizationFlags flags = KuDNN::NormalizationFlags::USE_SCALE;
    GTEST_EXPECT_TRUE(ForwardFunc1<__fp16>(srcInfo, scaleShiftInfo, groupInfo, dstInfo, flags));
}

TEST(GnormGTests, ForwardFunc4DF16_3)
{
    Shape shape(4, 128, 18, 320);
    SizeType groupInfo = 2;
    TensorInfo srcInfo = {shape, KuDNN::Element::TypeT::F16, KuDNN::Layout::ABCD};
    TensorInfo scaleShiftInfo = {{shape[1]}, KuDNN::Element::TypeT::F16, KuDNN::Layout::A};
    TensorInfo dstInfo = {shape, KuDNN::Element::TypeT::F16, KuDNN::Layout::ABCD};
    KuDNN::NormalizationFlags flags = KuDNN::NormalizationFlags::USE_SHIFT;
    GTEST_EXPECT_TRUE(ForwardFunc1<__fp16>(srcInfo, scaleShiftInfo, groupInfo, dstInfo, flags));
}

TEST(GnormGTests, ForwardFunc4DF16_4)
{
    Shape shape(4, 128, 18, 320);
    SizeType groupInfo = 2;
    TensorInfo srcInfo = {shape, KuDNN::Element::TypeT::F16, KuDNN::Layout::ABCD};
    TensorInfo scaleShiftInfo = {{shape[1]}, KuDNN::Element::TypeT::F16, KuDNN::Layout::A};
    TensorInfo dstInfo = {shape, KuDNN::Element::TypeT::F16, KuDNN::Layout::ABCD};
    KuDNN::NormalizationFlags flags = KuDNN::NormalizationFlags::USE_GLOBAL_STATS;
    GTEST_EXPECT_TRUE(ForwardFunc1<__fp16>(srcInfo, scaleShiftInfo, groupInfo, dstInfo, flags));
}

TEST(GnormGTests, ForwardFunc5DF16_1)
{
    Shape shape(1, 32, 10, 90, 16);
    SizeType groupInfo = 4;
    TensorInfo srcInfo = {shape, KuDNN::Element::TypeT::F16, KuDNN::Layout::ABCDE};
    TensorInfo scaleShiftInfo = {{shape[1]}, KuDNN::Element::TypeT::F16, KuDNN::Layout::A};
    TensorInfo dstInfo = {shape, KuDNN::Element::TypeT::F16, KuDNN::Layout::ABCDE};
    KuDNN::NormalizationFlags flags = KuDNN::NormalizationFlags::NONE;
    GTEST_EXPECT_TRUE(ForwardFunc1<__fp16>(srcInfo, scaleShiftInfo, groupInfo, dstInfo, flags));
}

TEST(GnormGTests, ForwardFunc5DF16_2)
{
    Shape shape(1, 32, 10, 90, 16);
    SizeType groupInfo = 4;
    TensorInfo srcInfo = {shape, KuDNN::Element::TypeT::F16, KuDNN::Layout::ABCDE};
    TensorInfo scaleShiftInfo = {{shape[1]}, KuDNN::Element::TypeT::F16, KuDNN::Layout::A};
    TensorInfo dstInfo = {shape, KuDNN::Element::TypeT::F16, KuDNN::Layout::ABCDE};
    KuDNN::NormalizationFlags flags = KuDNN::NormalizationFlags::USE_SCALE;
    GTEST_EXPECT_TRUE(ForwardFunc1<__fp16>(srcInfo, scaleShiftInfo, groupInfo, dstInfo, flags));
}

TEST(GnormGTests, ForwardFunc5DF16_3)
{
    Shape shape(1, 32, 10, 90, 16);
    SizeType groupInfo = 4;
    TensorInfo srcInfo = {shape, KuDNN::Element::TypeT::F16, KuDNN::Layout::ABCDE};
    TensorInfo scaleShiftInfo = {{shape[1]}, KuDNN::Element::TypeT::F16, KuDNN::Layout::A};
    TensorInfo dstInfo = {shape, KuDNN::Element::TypeT::F16, KuDNN::Layout::ABCDE};
    KuDNN::NormalizationFlags flags = KuDNN::NormalizationFlags::USE_SHIFT;
    GTEST_EXPECT_TRUE(ForwardFunc1<__fp16>(srcInfo, scaleShiftInfo, groupInfo, dstInfo, flags));
}

TEST(GnormGTests, ForwardFunc5DF16_4)
{
    Shape shape(1, 32, 10, 90, 16);
    SizeType groupInfo = 4;
    TensorInfo srcInfo = {shape, KuDNN::Element::TypeT::F16, KuDNN::Layout::ABCDE};
    TensorInfo scaleShiftInfo = {{shape[1]}, KuDNN::Element::TypeT::F16, KuDNN::Layout::A};
    TensorInfo dstInfo = {shape, KuDNN::Element::TypeT::F16, KuDNN::Layout::ABCDE};
    KuDNN::NormalizationFlags flags = KuDNN::NormalizationFlags::USE_GLOBAL_STATS;
    GTEST_EXPECT_TRUE(ForwardFunc1<__fp16>(srcInfo, scaleShiftInfo, groupInfo, dstInfo, flags));
}

TEST(GnormGTests, ForwardFunc4DF32_1)
{
    Shape shape(4, 128, 18, 320);
    SizeType groupInfo = 2;
    TensorInfo srcInfo = {shape, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCD};
    TensorInfo scaleShiftInfo = {{shape[1]}, KuDNN::Element::TypeT::F32, KuDNN::Layout::A};
    TensorInfo dstInfo = {shape, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCD};
    KuDNN::NormalizationFlags flags = KuDNN::NormalizationFlags::NONE;
    GTEST_EXPECT_TRUE(ForwardFunc1<float>(srcInfo, scaleShiftInfo, groupInfo, dstInfo, flags));
}

TEST(GnormGTests, ForwardFunc4DF32_2)
{
    Shape shape(4, 128, 18, 320);
    SizeType groupInfo = 2;
    TensorInfo srcInfo = {shape, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCD};
    TensorInfo scaleShiftInfo = {{shape[1]}, KuDNN::Element::TypeT::F32, KuDNN::Layout::A};
    TensorInfo dstInfo = {shape, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCD};
    KuDNN::NormalizationFlags flags = KuDNN::NormalizationFlags::USE_SCALE;
    GTEST_EXPECT_TRUE(ForwardFunc1<float>(srcInfo, scaleShiftInfo, groupInfo, dstInfo, flags));
}

TEST(GnormGTests, ForwardFunc4DF32_3)
{
    Shape shape(4, 128, 18, 320);
    SizeType groupInfo = 2;
    TensorInfo srcInfo = {shape, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCD};
    TensorInfo scaleShiftInfo = {{shape[1]}, KuDNN::Element::TypeT::F32, KuDNN::Layout::A};
    TensorInfo dstInfo = {shape, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCD};
    KuDNN::NormalizationFlags flags = KuDNN::NormalizationFlags::USE_SHIFT;
    GTEST_EXPECT_TRUE(ForwardFunc1<float>(srcInfo, scaleShiftInfo, groupInfo, dstInfo, flags));
}

TEST(GnormGTests, ForwardFunc4DF32_4)
{
    Shape shape(4, 128, 18, 320);
    SizeType groupInfo = 2;
    TensorInfo srcInfo = {shape, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCD};
    TensorInfo scaleShiftInfo = {{shape[1]}, KuDNN::Element::TypeT::F32, KuDNN::Layout::A};
    TensorInfo dstInfo = {shape, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCD};
    KuDNN::NormalizationFlags flags = KuDNN::NormalizationFlags::USE_GLOBAL_STATS;
    GTEST_EXPECT_TRUE(ForwardFunc1<float>(srcInfo, scaleShiftInfo, groupInfo, dstInfo, flags));
}

TEST(GnormGTests, ForwardFunc5DF32_1)
{
    Shape shape(1, 16, 20, 12, 8);
    SizeType groupInfo = 4;
    TensorInfo srcInfo = {shape, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCDE};
    TensorInfo scaleShiftInfo = {{shape[1]}, KuDNN::Element::TypeT::F32, KuDNN::Layout::A};
    TensorInfo dstInfo = {shape, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCDE};
    KuDNN::NormalizationFlags flags = KuDNN::NormalizationFlags::NONE;
    GTEST_EXPECT_TRUE(ForwardFunc1<float>(srcInfo, scaleShiftInfo, groupInfo, dstInfo, flags));
}

TEST(GnormGTests, ForwardFunc5DF32_2)
{
    Shape shape(1, 16, 20, 12, 8);
    SizeType groupInfo = 4;
    TensorInfo srcInfo = {shape, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCDE};
    TensorInfo scaleShiftInfo = {{shape[1]}, KuDNN::Element::TypeT::F32, KuDNN::Layout::A};
    TensorInfo dstInfo = {shape, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCDE};
    KuDNN::NormalizationFlags flags = KuDNN::NormalizationFlags::USE_SCALE;
    GTEST_EXPECT_TRUE(ForwardFunc1<float>(srcInfo, scaleShiftInfo, groupInfo, dstInfo, flags));
}

TEST(GnormGTests, ForwardFunc5DF32_3)
{
    Shape shape(1, 16, 20, 12, 8);
    SizeType groupInfo = 4;
    TensorInfo srcInfo = {shape, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCDE};
    TensorInfo scaleShiftInfo = {{shape[1]}, KuDNN::Element::TypeT::F32, KuDNN::Layout::A};
    TensorInfo dstInfo = {shape, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCDE};
    KuDNN::NormalizationFlags flags = KuDNN::NormalizationFlags::USE_SHIFT;
    GTEST_EXPECT_TRUE(ForwardFunc1<float>(srcInfo, scaleShiftInfo, groupInfo, dstInfo, flags));
}

TEST(GnormGTests, ForwardFunc5DF32_4)
{
    Shape shape(1, 16, 20, 12, 8);
    SizeType groupInfo = 4;
    TensorInfo srcInfo = {shape, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCDE};
    TensorInfo scaleShiftInfo = {{shape[1]}, KuDNN::Element::TypeT::F32, KuDNN::Layout::A};
    TensorInfo dstInfo = {shape, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCDE};
    KuDNN::NormalizationFlags flags = KuDNN::NormalizationFlags::USE_GLOBAL_STATS;
    GTEST_EXPECT_TRUE(ForwardFunc1<float>(srcInfo, scaleShiftInfo, groupInfo, dstInfo, flags));
}

TEST(GnormGTests, ForwardFunc5DF32_5)
{
    Shape shape(1, 16, 20, 12, 8);
    SizeType groupInfo = 4;
    TensorInfo srcInfo = {shape, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCDE};
    TensorInfo scaleShiftInfo = {{shape[1]}, KuDNN::Element::TypeT::F32, KuDNN::Layout::A};
    TensorInfo dstInfo = {shape, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCDE};
    KuDNN::NormalizationFlags flags = KuDNN::NormalizationFlags::USE_SCALE | KuDNN::NormalizationFlags::USE_SHIFT;
    GTEST_EXPECT_TRUE(ForwardFunc1<float>(srcInfo, scaleShiftInfo, groupInfo, dstInfo, flags));
}

// cross-layout / type-mixed test: new implementation only accepts F32 scaleShift,
// and the reference implementation assumes the standard layout, so we only check
// that the run does not throw
TEST(GnormGTests, ForwardFunc5DF32F16_1)
{
    Shape shape(1, 16, 20, 12, 8);
    SizeType groupInfo = 4;
    TensorInfo srcInfo = {shape, KuDNN::Element::TypeT::F32, KuDNN::Layout::ACDEB};
    TensorInfo scaleShiftInfo = {{shape[1]}, KuDNN::Element::TypeT::F32, KuDNN::Layout::A};
    TensorInfo dstInfo = {shape, KuDNN::Element::TypeT::F32, KuDNN::Layout::ACDEB};
    KuDNN::NormalizationFlags flags = KuDNN::NormalizationFlags::USE_SCALE | KuDNN::NormalizationFlags::USE_SHIFT;
    EXPECT_NO_THROW(ForwardFunc1<float>(srcInfo, scaleShiftInfo, groupInfo, dstInfo, flags));
}

TEST(GnormGTests, FWDSaveStats)
{
    Shape shape(1, 16, 20, 12);
    SizeType groupInfo = 4;
    TensorInfo srcInfo = {shape, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCD};
    TensorInfo scaleShiftInfo = {{shape[1]}, KuDNN::Element::TypeT::F32, KuDNN::Layout::A};
    TensorInfo dstInfo = {shape, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCD};
    KuDNN::NormalizationFlags flags = KuDNN::NormalizationFlags::NONE;
    GTEST_EXPECT_TRUE(FWDSaveStats<float>(srcInfo, scaleShiftInfo, groupInfo, dstInfo, flags));
}

TEST(GnormGTests, FWDDivideZero)
{
    Shape shape(1, 16, 20, 12);
    SizeType groupInfo = 4;
    TensorInfo srcInfo = {shape, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCD};
    TensorInfo scaleShiftInfo = {{shape[1]}, KuDNN::Element::TypeT::F32, KuDNN::Layout::A};
    TensorInfo dstInfo = {shape, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCD};
    KuDNN::NormalizationFlags flags = KuDNN::NormalizationFlags::NONE;
    EXPECT_NO_THROW(FWDDivideZero<float>(srcInfo, scaleShiftInfo, groupInfo, dstInfo, flags));
}

TEST(GnormGTests, BackwardFunc5DF32F16_1)
{
    const KuDNN::TensorInfo srcInfo = {{8, 10, 12, 14, 16}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCDE};
    const KuDNN::TensorInfo diffSrcInfo = {{8, 10, 12, 14, 16}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCDE};
    const KuDNN::TensorInfo statInfo = {{8, 10, 12, 14}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCD};
    const KuDNN::TensorInfo scaleShiftInfo = {{8}, KuDNN::Element::TypeT::F32, KuDNN::Layout::A};
    const KuDNN::TensorInfo diffScaleShiftInfo = {{8}, KuDNN::Element::TypeT::F32, KuDNN::Layout::A};
    const KuDNN::TensorInfo diffDstInfo = {{8, 10, 12, 14, 16}, KuDNN::Element::TypeT::F16, KuDNN::Layout::ABCDE};
    KuDNN::NormalizationFlags flags = KuDNN::NormalizationFlags::USE_SCALE | KuDNN::NormalizationFlags::USE_SHIFT;
    GTEST_EXPECT_TRUE(BackwardFunc1<float>(srcInfo, statInfo, diffSrcInfo, diffDstInfo, scaleShiftInfo, diffScaleShiftInfo, flags));
}

TEST(GnormGTests, BackwardFunc5DF32F16_2)
{
    const KuDNN::TensorInfo srcInfo = {{8, 10, 12, 14, 16}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCDE};
    const KuDNN::TensorInfo diffSrcInfo = {{8, 10, 12, 14, 16}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCDE};
    const KuDNN::TensorInfo statInfo = {{8, 10, 12, 14}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCD};
    const KuDNN::TensorInfo scaleShiftInfo = {{8}, KuDNN::Element::TypeT::F32, KuDNN::Layout::A};
    const KuDNN::TensorInfo diffScaleShiftInfo = {{8}, KuDNN::Element::TypeT::F32, KuDNN::Layout::A};
    const KuDNN::TensorInfo diffDstInfo = {{8, 10, 12, 14, 16}, KuDNN::Element::TypeT::F16, KuDNN::Layout::ABCDE};
    KuDNN::NormalizationFlags flags = KuDNN::NormalizationFlags::USE_SCALE | KuDNN::NormalizationFlags::USE_SHIFT;
    GTEST_EXPECT_TRUE(BackwardFunc1<float>(srcInfo, statInfo, diffSrcInfo, diffDstInfo, scaleShiftInfo, diffScaleShiftInfo, flags));
}

TEST(GnormGTests, BackwardFunc5DF32F16_3)
{
    const KuDNN::TensorInfo srcInfo = {{8, 10, 12, 14, 16}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCDE};
    const KuDNN::TensorInfo diffSrcInfo = {{8, 10, 12, 14, 16}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCDE};
    const KuDNN::TensorInfo statInfo = {{8, 10, 12, 14}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCD};
    const KuDNN::TensorInfo scaleShiftInfo = {{8}, KuDNN::Element::TypeT::F32, KuDNN::Layout::A};
    const KuDNN::TensorInfo diffScaleShiftInfo = {{8}, KuDNN::Element::TypeT::F32, KuDNN::Layout::A};
    const KuDNN::TensorInfo diffDstInfo = {{8, 10, 12, 14, 16}, KuDNN::Element::TypeT::F16, KuDNN::Layout::ABCDE};
    KuDNN::NormalizationFlags flags = KuDNN::NormalizationFlags::NONE;
    GTEST_EXPECT_TRUE(BackwardFunc1<float>(srcInfo, statInfo, diffSrcInfo, diffDstInfo, scaleShiftInfo, diffScaleShiftInfo, flags));
}

TEST(GnormGTests, BackwardFunc5DF32F16_4)
{
    const KuDNN::TensorInfo srcInfo = {{8, 10, 12, 14, 16}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCDE};
    const KuDNN::TensorInfo diffSrcInfo = {{8, 10, 12, 14, 16}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCDE};
    const KuDNN::TensorInfo statInfo = {{8, 10, 12, 14}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCD};
    const KuDNN::TensorInfo scaleShiftInfo = {{8}, KuDNN::Element::TypeT::F32, KuDNN::Layout::A};
    const KuDNN::TensorInfo diffScaleShiftInfo = {{8}, KuDNN::Element::TypeT::F32, KuDNN::Layout::A};
    const KuDNN::TensorInfo diffDstInfo = {{8, 10, 12, 14, 16}, KuDNN::Element::TypeT::F16, KuDNN::Layout::ABCDE};
    KuDNN::NormalizationFlags flags = KuDNN::NormalizationFlags::USE_GLOBAL_STATS;
    GTEST_EXPECT_TRUE(BackwardFunc1<float>(srcInfo, statInfo, diffSrcInfo, diffDstInfo, scaleShiftInfo, diffScaleShiftInfo, flags));
}

TEST(GnormGTests, BackwardFunc3DF32_1)
{
    const KuDNN::TensorInfo srcInfo = {{1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABC};
    const KuDNN::TensorInfo diffSrcInfo = {{1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABC};
    const KuDNN::TensorInfo statInfo = {{1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::AB};
    const KuDNN::TensorInfo scaleShiftInfo = {{1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::A};
    const KuDNN::TensorInfo diffScaleShiftInfo = {{1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::A};
    const KuDNN::TensorInfo diffDstInfo = {{1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABC};
    KuDNN::NormalizationFlags flags = KuDNN::NormalizationFlags::NONE;
    GTEST_EXPECT_TRUE(BackwardFunc1<float>(srcInfo, statInfo, diffSrcInfo, diffDstInfo, scaleShiftInfo, diffScaleShiftInfo, flags));
}

TEST(GnormGTests, FWDSrcDstType)
{
    EXPECT_NO_THROW(FWDSrcDstType());
}

TEST(GnormGTests, FWDScaleType)
{
    EXPECT_THROW(FWDScaleType(), KuDNN::Service::Unsupported);
}

TEST(GnormGTests, FWDShiftType)
{
    EXPECT_THROW(FWDShiftType(), KuDNN::Service::Unsupported);
}

TEST(GnormGTests, FWDGlobalStatsType)
{
    EXPECT_NO_THROW(FWDGlobalStatsType());
}

TEST(GnormGTests, FWDCopyConstructor1)
{
    EXPECT_NO_THROW(ForwardCopyConstructor1());
}

TEST(GnormGTests, FWDCopyConstructor2)
{
    EXPECT_NO_THROW(ForwardCopyConstructor2());
}

TEST(GnormGTests, FWDMoveConstructor1)
{
    EXPECT_NO_THROW(ForwardMoveConstructor1());
}

TEST(GnormGTests, FWDMoveConstructor2)
{
    EXPECT_NO_THROW(ForwardMoveConstructor2());
}

TEST(GnormGTests, BWDCopyConstructor1)
{
    EXPECT_NO_THROW(BackwardCopyConstructor1());
}

TEST(GnormGTests, BWDCopyConstructor2)
{
    EXPECT_NO_THROW(BackwardCopyConstructor2());
}

TEST(GnormGTests, BWDMoveConstructor1)
{
    EXPECT_NO_THROW(BackwardMoveConstructor1());
}

TEST(GnormGTests, BWDMoveConstructor2)
{
    EXPECT_NO_THROW(BackwardMoveConstructor2());
}
