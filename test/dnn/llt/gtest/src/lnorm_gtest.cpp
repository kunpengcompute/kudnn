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
KuDNN::Element::TypeT TypeF16 = KuDNN::Element::TypeT::F16;
KuDNN::Element::TypeT TypeF32 = KuDNN::Element::TypeT::F32;

const int MAX_THS = 100;
const float FACTOR_THS = 0.02;
float errBoundLn = 1e-3;

template <typename T>
static void lnormRef(SizeType outerSize, SizeType innerSize,
                    const T *src, T *dst, const T *scale, const T *shift,
                    bool scaleApply, bool shiftApply,
                    float *mean, float *variance, const float eps, bool global_stats = false)
{
    int threadsUsed = (int)ceil((float)outerSize * FACTOR_THS);
    threadsUsed = threadsUsed > MAX_THS ? MAX_THS : threadsUsed;
    if (!global_stats) {
    // mean compute
#pragma omp parallel for num_threads(threadsUsed) schedule(static)
        for (SizeType i = 0; i < outerSize; ++i) {
            float sum = 0.0f;
            for (SizeType j = 0; j < innerSize; ++j) {
                sum += (float)*(src + i * innerSize + j);
            }
            *(mean + i) = sum / innerSize;
        }
    // variance compute
#pragma omp parallel for num_threads(threadsUsed) schedule(static)
        for (SizeType i = 0; i < outerSize; ++i) {
            float sum = 0.0f;
            float meanVal = *(mean + i);
            for (SizeType j = 0; j < innerSize; ++j) {
                float diff = (float)*(src + i * innerSize + j) - meanVal;
                sum += diff * diff;
            }
            *(variance + i) = sum / innerSize;
        }
    }
    // norm
#pragma omp parallel for collapse(2) num_threads(threadsUsed)
    for (SizeType i = 0; i < outerSize; ++i) {
        float meanVal = *(mean + i);
        float varVal = *(variance + i);
        float invStd = 1.0f / std::sqrt(varVal + eps);
        for (SizeType j = 0; j < innerSize; ++j) {
            float srcVal = (float)*(src + i * innerSize + j);
            float normalized = (srcVal - meanVal) * invStd;
            if (scaleApply) {
                normalized *= scale[j];
            }
            if (shiftApply) {
                normalized += shift[j];
            }
            *(dst + i * innerSize + j) = static_cast<T>(normalized);
        }
    }
}

// backward
template <typename T>
static bool BackwardFunc1(const KuDNN::TensorInfo srcInfo, const KuDNN::TensorInfo statInfo, const KuDNN::TensorInfo diffSrcInfo,
                        const KuDNN::TensorInfo diffDstInfo, const KuDNN::TensorInfo scaleShiftInfo, const KuDNN::TensorInfo diffScaleShiftInfo,
                        KuDNN::NormalizationFlags flags)
{
    KuDNN::NormalizationLayerBWD lnormLayer1(srcInfo, statInfo, diffSrcInfo, diffDstInfo, scaleShiftInfo,
        diffScaleShiftInfo, flags);
    SizeType srcSize = srcInfo.GetTotalTensorSize();
    SizeType diffSrcSize = diffSrcInfo.GetTotalTensorSize();
    SizeType statSize = statInfo.GetTotalTensorSize();
    SizeType innerSize = scaleShiftInfo.GetTotalTensorSize();
    SizeType diffDstSize = diffDstInfo.GetTotalTensorSize();
    SizeType diffScaleSize = diffScaleShiftInfo.GetTotalTensorSize();
    T *src = (T *)malloc(srcSize * sizeof(T));
    T *diffSrc = (T *)malloc(diffSrcSize * sizeof(T));
    float *mean = (float *)malloc(statSize * sizeof(float));
    float *variance = (float *)malloc(statSize * sizeof(float));
    T *scale = (T *)malloc(innerSize * sizeof(T));
    T *diffShift = (T *)malloc(innerSize * sizeof(T));
    T *diffScale = (T *)malloc(diffScaleSize * sizeof(T));
    T *diffDst = (T *)malloc(diffDstSize * sizeof(T));

    lnormLayer1.Run(src, mean, variance, diffDst, scale, diffSrc, diffScale,  diffShift, 1e-5);
    free(src);
    free(diffSrc);
    free(mean);
    free(variance);
    free(scale);
    free(diffShift);
    free(diffScale);
    free(diffDst);
    return true;
}
// forward
template <typename T>
static bool ForwardFunc1(const TensorInfo &srcInfo, const TensorInfo &statInfo,
                        const TensorInfo &scaleShiftInfo, const TensorInfo &dstInfo,
                        KuDNN::NormalizationFlags flags)
{
    KuDNN::NormalizationLayerFWD lnormLayer1(srcInfo, statInfo, scaleShiftInfo, dstInfo,
        flags);
    SizeType srcSize = srcInfo.GetTotalTensorSize();
    SizeType dstSize = dstInfo.GetTotalTensorSize();
    SizeType statSize = statInfo.GetTotalTensorSize();
    SizeType innerSize = scaleShiftInfo.GetTotalTensorSize();
    T *src = (T *)malloc(srcSize * sizeof(T));
    T *dst = (T *)malloc(dstSize * sizeof(T));
    T *dstRef = (T *)malloc(dstSize * sizeof(T));
    float *mean = (float *)malloc(statSize * sizeof(float));
    float *variance = (float *)malloc(statSize * sizeof(float));
    T *scale = (T *)malloc(innerSize * sizeof(T));
    T *shift = (T *)malloc(innerSize * sizeof(T));
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
    for (SizeType j = 0; j < innerSize; ++j) {
        *(scale + j) = (T)u(e);
        *(shift + j) = (T)u(e);
    }
    if (global_stats) {
        for (SizeType i = 0; i < statSize; i++) {
            *(mean + i) = 1.0f;
            *(variance + i) = 0.1f;
        }
    }

    float error = 0.0f;
    lnormLayer1.Run(src, dst, scale, shift, mean, variance, true, eps);
    bool scaleApply = static_cast<bool>(flags & KuDNN::NormalizationFlags::USE_SCALE);
    bool shiftApply = static_cast<bool>(flags & KuDNN::NormalizationFlags::USE_SHIFT);
    lnormRef<T>(statSize, innerSize, src, dstRef, scale, shift, scaleApply, shiftApply, mean, variance, eps, global_stats);
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
    return error < errBoundLn;
}

template <typename T>
static bool FWDSaveStats(const TensorInfo &srcInfo, const TensorInfo &statInfo,
                        const TensorInfo &scaleShiftInfo, const TensorInfo &dstInfo,
                        KuDNN::NormalizationFlags flags)
{
    flags = KuDNN::NormalizationFlags::NONE;
    KuDNN::NormalizationLayerFWD lnormLayer1(srcInfo, statInfo, scaleShiftInfo, dstInfo,
        flags);
    SizeType srcSize = srcInfo.GetTotalTensorSize();
    SizeType dstSize = dstInfo.GetTotalTensorSize();
    SizeType statSize = statInfo.GetTotalTensorSize();
    SizeType innerSize = scaleShiftInfo.GetTotalTensorSize();
    T *src = (T *)malloc(srcSize * sizeof(T));
    T *dst = (T *)malloc(dstSize * sizeof(T));
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

    float error = 0.0f;
    lnormLayer1.Run(src, dst, scale, shift, mean, variance, true, eps);
    bool scaleApply = false;
    bool shiftApply = false;
    lnormRef<T>(statSize, innerSize, src, dst, scale, shift, scaleApply, shiftApply, meanRef, varianceRef, eps);
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
    return error < errBoundLn;
}

static void FWDSrcDstType()
{
    const KuDNN::TensorInfo srcInfo = {{1, 1, 1}, KuDNN::Element::TypeT::S32, KuDNN::Layout::ABC};
    const KuDNN::TensorInfo statInfo = {{1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::AB};
    const KuDNN::TensorInfo scaleShiftInfo = {{1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::A};
    const KuDNN::TensorInfo dstInfo = {{1, 1, 1}, KuDNN::Element::TypeT::S32, KuDNN::Layout::ABC};
    KuDNN::NormalizationLayerFWD lnormLayer1(srcInfo, statInfo, scaleShiftInfo, dstInfo,
        KuDNN::NormalizationFlags::NONE);
}

static void FWDScaleShiftType()
{
    const KuDNN::TensorInfo srcInfo = {{1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABC};
    const KuDNN::TensorInfo statInfo = {{1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::AB};
    const KuDNN::TensorInfo scaleShiftInfo = {{1}, KuDNN::Element::TypeT::S32, KuDNN::Layout::A};
    const KuDNN::TensorInfo dstInfo = {{1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABC};
    KuDNN::NormalizationLayerFWD lnormLayer1(srcInfo, statInfo, scaleShiftInfo, dstInfo,
        KuDNN::NormalizationFlags::USE_SCALE);
}

static void FWDGlobalStatsType()
{
    const KuDNN::TensorInfo srcInfo = {{1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABC};
    const KuDNN::TensorInfo statInfo = {{1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::AB};
    const KuDNN::TensorInfo scaleShiftInfo = {{1}, KuDNN::Element::TypeT::S32, KuDNN::Layout::A};
    const KuDNN::TensorInfo dstInfo = {{1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABC};
    KuDNN::NormalizationLayerFWD lnormLayer1(srcInfo, statInfo, scaleShiftInfo, dstInfo,
        KuDNN::NormalizationFlags::USE_GLOBAL_STATS);
}

static void FWDFlagType()
{
    const KuDNN::TensorInfo srcInfo = {{1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABC};
    const KuDNN::TensorInfo statInfo = {{1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::AB};
    const KuDNN::TensorInfo scaleShiftInfo = {{1}, KuDNN::Element::TypeT::S32, KuDNN::Layout::A};
    const KuDNN::TensorInfo dstInfo = {{1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABC};
    KuDNN::NormalizationLayerFWD lnormLayer1(srcInfo, statInfo, scaleShiftInfo, dstInfo,
        KuDNN::NormalizationFlags::FUSE_NORM_RELU);
}

static void ForwardCopyConstructor1()
{
    const KuDNN::TensorInfo srcInfo = {{2, 3, 4}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABC};
    const KuDNN::TensorInfo statInfo = {{2, 3}, KuDNN::Element::TypeT::F32, KuDNN::Layout::AB};
    const KuDNN::TensorInfo scaleShiftInfo = {{4}, KuDNN::Element::TypeT::F32, KuDNN::Layout::A};
    const KuDNN::TensorInfo dstInfo = {{2, 3, 4}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABC};
    KuDNN::NormalizationLayerFWD lnormLayer1(srcInfo, statInfo, scaleShiftInfo, dstInfo,
        KuDNN::NormalizationFlags::NONE);
    KuDNN::NormalizationLayerFWD lnormLayer2(lnormLayer1);
    for (size_t i = 0; i < g_Tensor_data.size() / 4; i += 4) {
        KuDNN::NormalizationLayerFWD lnormLayer1(g_Tensor_data[i], g_Tensor_data[i + 1], g_Tensor_data[i + 2],
            g_Tensor_data[i + 3], KuDNN::NormalizationFlags::NONE);
        KuDNN::NormalizationLayerFWD lnormLayer2(lnormLayer1);
    }
}

static void ForwardCopyConstructor2()
{
    const KuDNN::TensorInfo srcInfo = {{1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABC};
    const KuDNN::TensorInfo statInfo = {{1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::AB};
    const KuDNN::TensorInfo scaleShiftInfo = {{1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::A};
    const KuDNN::TensorInfo dstInfo = {{1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABC};
    KuDNN::NormalizationLayerFWD lnormLayer1(srcInfo, statInfo, scaleShiftInfo, dstInfo,
        KuDNN::NormalizationFlags::NONE);
    KuDNN::NormalizationLayerFWD lnormLayer2 = lnormLayer1;
    for (size_t i = 0; i < g_Tensor_data.size() / 4; i += 4) {
        KuDNN::NormalizationLayerFWD lnormLayer1(g_Tensor_data[i], g_Tensor_data[i + 1], g_Tensor_data[i + 2],
            g_Tensor_data[i + 3], KuDNN::NormalizationFlags::NONE);
        KuDNN::NormalizationLayerFWD lnormLayer2 = lnormLayer1;
    }
}

static void ForwardMoveConstructor1()
{
    const KuDNN::TensorInfo srcInfo = {{1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABC};
    const KuDNN::TensorInfo statInfo = {{1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::AB};
    const KuDNN::TensorInfo scaleShiftInfo = {{1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::A};
    const KuDNN::TensorInfo dstInfo = {{1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABC};
    KuDNN::NormalizationLayerFWD lnormLayer1(srcInfo, statInfo, scaleShiftInfo, dstInfo,
        KuDNN::NormalizationFlags::NONE);
    KuDNN::NormalizationLayerFWD lnormLayer2(std::move(lnormLayer1));
    for (size_t i = 0; i < g_Tensor_data.size() / 4; i += 4) {
        KuDNN::NormalizationLayerFWD lnormLayer1(g_Tensor_data[i], g_Tensor_data[i + 1], g_Tensor_data[i + 2],
            g_Tensor_data[i + 3], KuDNN::NormalizationFlags::NONE);
        KuDNN::NormalizationLayerFWD lnormLayer2(std::move(lnormLayer1));
    }
}

static void ForwardMoveConstructor2()
{
    const KuDNN::TensorInfo srcInfo = {{1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABC};
    const KuDNN::TensorInfo statInfo = {{1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::AB};
    const KuDNN::TensorInfo scaleShiftInfo = {{1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::A};
    const KuDNN::TensorInfo dstInfo = {{1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABC};
    KuDNN::NormalizationLayerFWD lnormLayer1(srcInfo, statInfo, scaleShiftInfo, dstInfo,
        KuDNN::NormalizationFlags::NONE);
    KuDNN::NormalizationLayerFWD lnormLayer2(srcInfo, statInfo, scaleShiftInfo, dstInfo,
        KuDNN::NormalizationFlags::NONE);
    lnormLayer2 = std::move(lnormLayer1);
    for (size_t i = 0; i < g_Tensor_data.size() / 4; i += 4) {
        KuDNN::NormalizationLayerFWD lnormLayer1(g_Tensor_data[i], g_Tensor_data[i + 1], g_Tensor_data[i + 2],
            g_Tensor_data[i + 3], KuDNN::NormalizationFlags::NONE);
        KuDNN::NormalizationLayerFWD lnormLayer2(g_Tensor_data[i], g_Tensor_data[i + 1], g_Tensor_data[i + 2],
            g_Tensor_data[i + 3], KuDNN::NormalizationFlags::NONE);
        lnormLayer2 = std::move(lnormLayer1);
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
    KuDNN::NormalizationLayerBWD lnormLayer1(srcInfo, statInfo, diffSrcInfo, diffDstInfo, scaleShiftInfo,
        diffScaleShiftInfo, KuDNN::NormalizationFlags::NONE);
    KuDNN::NormalizationLayerBWD lnormLayer2(lnormLayer1);
    for (size_t i = 0; i < g_Tensor_data.size() / 6; i += 6) {
        KuDNN::NormalizationLayerBWD lnormLayer1(g_Tensor_data[i], g_Tensor_data[i + 1], g_Tensor_data[i + 2],
            g_Tensor_data[i + 3], g_Tensor_data[i + 4], g_Tensor_data[i + 5], KuDNN::NormalizationFlags::NONE);
        KuDNN::NormalizationLayerBWD lnormLayer2(lnormLayer1);
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
    KuDNN::NormalizationLayerBWD lnormLayer1(srcInfo, statInfo, diffSrcInfo, diffDstInfo, scaleShiftInfo,
        diffScaleShiftInfo, KuDNN::NormalizationFlags::NONE);
    KuDNN::NormalizationLayerBWD lnormLayer2(srcInfo, statInfo, diffSrcInfo, diffDstInfo, scaleShiftInfo,
        diffScaleShiftInfo, KuDNN::NormalizationFlags::NONE);
    lnormLayer2 = lnormLayer1;
    for (size_t i = 0; i < g_Tensor_data.size() / 6; i += 6) {
        KuDNN::NormalizationLayerBWD lnormLayer1(g_Tensor_data[i], g_Tensor_data[i + 1], g_Tensor_data[i + 2],
            g_Tensor_data[i + 3], g_Tensor_data[i + 4], g_Tensor_data[i + 5], KuDNN::NormalizationFlags::NONE);
        KuDNN::NormalizationLayerBWD lnormLayer2(g_Tensor_data[i], g_Tensor_data[i + 1], g_Tensor_data[i + 2],
            g_Tensor_data[i + 3], g_Tensor_data[i + 4], g_Tensor_data[i + 5], KuDNN::NormalizationFlags::NONE);
        lnormLayer2 = lnormLayer1;
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
    KuDNN::NormalizationLayerBWD lnormLayer1(srcInfo, statInfo, diffSrcInfo, diffDstInfo, scaleShiftInfo,
        diffScaleShiftInfo, KuDNN::NormalizationFlags::NONE);
    KuDNN::NormalizationLayerBWD lnormLayer2(std::move(lnormLayer1));
    for (size_t i = 0; i < g_Tensor_data.size() / 6; i += 6) {
        KuDNN::NormalizationLayerBWD lnormLayer1(g_Tensor_data[i], g_Tensor_data[i + 1], g_Tensor_data[i + 2],
            g_Tensor_data[i + 3], g_Tensor_data[i + 4], g_Tensor_data[i + 5], KuDNN::NormalizationFlags::NONE);
        KuDNN::NormalizationLayerBWD lnormLayer2(std::move(lnormLayer1));
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
    KuDNN::NormalizationLayerBWD lnormLayer1(srcInfo, statInfo, diffSrcInfo, diffDstInfo, scaleShiftInfo,
        diffScaleShiftInfo, KuDNN::NormalizationFlags::NONE);
    KuDNN::NormalizationLayerBWD lnormLayer2(srcInfo, statInfo, diffSrcInfo, diffDstInfo, scaleShiftInfo,
        diffScaleShiftInfo, KuDNN::NormalizationFlags::NONE);
    lnormLayer2 = std::move(lnormLayer1);
    for (size_t i = 0; i < g_Tensor_data.size() / 6; i += 6) {
        KuDNN::NormalizationLayerBWD lnormLayer1(g_Tensor_data[i], g_Tensor_data[i + 1], g_Tensor_data[i + 2],
            g_Tensor_data[i + 3], g_Tensor_data[i + 4], g_Tensor_data[i + 5], KuDNN::NormalizationFlags::NONE);
        KuDNN::NormalizationLayerBWD lnormLayer2(g_Tensor_data[i], g_Tensor_data[i + 1], g_Tensor_data[i + 2],
            g_Tensor_data[i + 3], g_Tensor_data[i + 4], g_Tensor_data[i + 5], KuDNN::NormalizationFlags::NONE);
        lnormLayer2 = std::move(lnormLayer1);
    }
}

static void InputParamExceptTest1()
{
    Shape shape(-1, 100);
    TensorInfo srcInfo = {shape, TypeF16, KuDNN::Layout::AB};
    TensorInfo statInfo = {{shape[0]}, TypeF16, KuDNN::Layout::A};
    TensorInfo scaleShiftInfo = {{shape[1]}, TypeF16, KuDNN::Layout::A};
    TensorInfo dstInfo = {shape, TypeF16, KuDNN::Layout::AB};
    KuDNN::NormalizationFlags flags = KuDNN::NormalizationFlags::USE_SCALE | KuDNN::NormalizationFlags::USE_SHIFT;
    KuDNN::NormalizationLayerFWD lnormLayer1(srcInfo, statInfo, scaleShiftInfo, dstInfo, flags);
}

static void InputParamExceptTest2()
{
    Shape shape(1, -1, 1, 1, 1);
    TensorInfo srcInfo = {shape, TypeF16, KuDNN::Layout::AB};
    TensorInfo statInfo = {{shape[0]}, TypeF16, KuDNN::Layout::A};
    TensorInfo scaleShiftInfo = {{shape[1]}, TypeF16, KuDNN::Layout::A};
    TensorInfo dstInfo = {shape, TypeF16, KuDNN::Layout::AB};
    KuDNN::NormalizationFlags flags = KuDNN::NormalizationFlags::USE_SCALE | KuDNN::NormalizationFlags::USE_SHIFT;
    KuDNN::NormalizationLayerFWD lnormLayer1(srcInfo, statInfo, scaleShiftInfo, dstInfo, flags);
}

static void InputParamExceptTest3()
{
    Shape shape(1, 100, -1, 1, 1);
    TensorInfo srcInfo = {shape, TypeF16, KuDNN::Layout::AB};
    TensorInfo statInfo = {{shape[0]}, TypeF16, KuDNN::Layout::A};
    TensorInfo scaleShiftInfo = {{shape[1]}, TypeF16, KuDNN::Layout::A};
    TensorInfo dstInfo = {shape, TypeF16, KuDNN::Layout::AB};
    KuDNN::NormalizationFlags flags = KuDNN::NormalizationFlags::USE_SCALE | KuDNN::NormalizationFlags::USE_SHIFT;
    KuDNN::NormalizationLayerFWD lnormLayer1(srcInfo, statInfo, scaleShiftInfo, dstInfo, flags);
}

static void InputParamExceptTest4()
{
    Shape shape(1, 100, 1, -1, 1);
    TensorInfo srcInfo = {shape, TypeF16, KuDNN::Layout::AB};
    TensorInfo statInfo = {{shape[0]}, TypeF16, KuDNN::Layout::A};
    TensorInfo scaleShiftInfo = {{shape[1]}, TypeF16, KuDNN::Layout::A};
    TensorInfo dstInfo = {shape, TypeF16, KuDNN::Layout::AB};
    KuDNN::NormalizationFlags flags = KuDNN::NormalizationFlags::USE_SCALE | KuDNN::NormalizationFlags::USE_SHIFT;
    KuDNN::NormalizationLayerFWD lnormLayer1(srcInfo, statInfo, scaleShiftInfo, dstInfo, flags);
}

static void InputParamExceptTest5()
{
    Shape shape(1, 100, 1, 1, -1);
    TensorInfo srcInfo = {shape, TypeF16, KuDNN::Layout::AB};
    TensorInfo statInfo = {{shape[0]}, TypeF16, KuDNN::Layout::A};
    TensorInfo scaleShiftInfo = {{shape[1]}, TypeF16, KuDNN::Layout::A};
    TensorInfo dstInfo = {shape, TypeF16, KuDNN::Layout::AB};
    KuDNN::NormalizationFlags flags = KuDNN::NormalizationFlags::USE_SCALE | KuDNN::NormalizationFlags::USE_SHIFT;
    KuDNN::NormalizationLayerFWD lnormLayer1(srcInfo, statInfo, scaleShiftInfo, dstInfo, flags);
}

// test cases
TEST(LnormGTests, Param_ShapeExceptD1_Test)
{
    EXPECT_THROW(InputParamExceptTest1(), KuDNN::Service::LogicError);
}

TEST(LnormGTests, Param_ShapeExceptD2_Test)
{
    EXPECT_THROW(InputParamExceptTest2(), KuDNN::Service::LogicError);
}

TEST(LnormGTests, Param_ShapeExceptD3_Test)
{
    EXPECT_THROW(InputParamExceptTest3(), KuDNN::Service::LogicError);
}

TEST(LnormGTests, Param_ShapeExceptD4_Test)
{
    EXPECT_THROW(InputParamExceptTest4(), KuDNN::Service::LogicError);
}

TEST(LnormGTests, Param_ShapeExceptD5_Test)
{
    EXPECT_THROW(InputParamExceptTest5(), KuDNN::Service::LogicError);
}

TEST(LnormGTests, ForwardFunc2DF16_1)
{
    Shape shape(100, 100);
    TensorInfo srcInfo = {shape, TypeF16, KuDNN::Layout::AB};
    TensorInfo statInfo = {{shape[0]}, TypeF16, KuDNN::Layout::A};
    TensorInfo scaleShiftInfo = {{shape[1]}, TypeF16, KuDNN::Layout::A};
    TensorInfo dstInfo = {shape, TypeF16, KuDNN::Layout::AB};
    KuDNN::NormalizationFlags flags = KuDNN::NormalizationFlags::NONE;
    GTEST_EXPECT_TRUE(ForwardFunc1<__fp16>(srcInfo, statInfo, scaleShiftInfo, dstInfo, flags));
}

TEST(LnormGTests, ForwardFunc2DF16_2)
{
    Shape shape(100, 100);
    TensorInfo srcInfo = {shape, TypeF16, KuDNN::Layout::AB};
    TensorInfo statInfo = {{shape[0]}, TypeF16, KuDNN::Layout::A};
    TensorInfo scaleShiftInfo = {{shape[1]}, TypeF16, KuDNN::Layout::A};
    TensorInfo dstInfo = {shape, TypeF16, KuDNN::Layout::AB};
    KuDNN::NormalizationFlags flags = KuDNN::NormalizationFlags::USE_SCALE;
    GTEST_EXPECT_TRUE(ForwardFunc1<__fp16>(srcInfo, statInfo, scaleShiftInfo, dstInfo, flags));
}

TEST(LnormGTests, ForwardFunc2DF16_3)
{
    Shape shape(100, 100);
    TensorInfo srcInfo = {shape, TypeF16, KuDNN::Layout::AB};
    TensorInfo statInfo = {{shape[0]}, TypeF16, KuDNN::Layout::A};
    TensorInfo scaleShiftInfo = {{shape[1]}, TypeF16, KuDNN::Layout::A};
    TensorInfo dstInfo = {shape, TypeF16, KuDNN::Layout::AB};
    KuDNN::NormalizationFlags flags = KuDNN::NormalizationFlags::USE_SHIFT;
    GTEST_EXPECT_TRUE(ForwardFunc1<__fp16>(srcInfo, statInfo, scaleShiftInfo, dstInfo, flags));
}

TEST(LnormGTests, ForwardFunc2DF16_4)
{
    Shape shape(100, 100);
    TensorInfo srcInfo = {shape, TypeF16, KuDNN::Layout::AB};
    TensorInfo statInfo = {{shape[0]}, TypeF16, KuDNN::Layout::A};
    TensorInfo scaleShiftInfo = {{shape[1]}, TypeF16, KuDNN::Layout::A};
    TensorInfo dstInfo = {shape, TypeF16, KuDNN::Layout::AB};
    KuDNN::NormalizationFlags flags = KuDNN::NormalizationFlags::USE_GLOBAL_STATS;
    GTEST_EXPECT_TRUE(ForwardFunc1<__fp16>(srcInfo, statInfo, scaleShiftInfo, dstInfo, flags));
}

TEST(LnormGTests, ForwardFunc2DF16_5)
{
    Shape shape(100, 100);
    TensorInfo srcInfo = {shape, TypeF16, KuDNN::Layout::AB};
    TensorInfo statInfo = {{shape[0]}, TypeF16, KuDNN::Layout::A};
    TensorInfo scaleShiftInfo = {{shape[1]}, TypeF16, KuDNN::Layout::A};
    TensorInfo dstInfo = {shape, TypeF16, KuDNN::Layout::AB};
    KuDNN::NormalizationFlags flags = KuDNN::NormalizationFlags::USE_SCALE | KuDNN::NormalizationFlags::USE_SHIFT;
    GTEST_EXPECT_TRUE(ForwardFunc1<__fp16>(srcInfo, statInfo, scaleShiftInfo, dstInfo, flags));
}

TEST(LnormGTests, ForwardFunc3DF16_1)
{
    Shape shape(2, 3240, 1152);
    TensorInfo srcInfo = {shape, TypeF16, KuDNN::Layout::ABC};
    TensorInfo statInfo = {{shape[0], shape[1]}, TypeF16, KuDNN::Layout::AB};
    TensorInfo scaleShiftInfo = {{shape[2]}, TypeF16, KuDNN::Layout::A};
    TensorInfo dstInfo = {shape, TypeF16, KuDNN::Layout::ABC};
    KuDNN::NormalizationFlags flags = KuDNN::NormalizationFlags::NONE;
    GTEST_EXPECT_TRUE(ForwardFunc1<__fp16>(srcInfo, statInfo, scaleShiftInfo, dstInfo, flags));
}

TEST(LnormGTests, ForwardFunc3DF16_2)
{
    Shape shape(2, 3240, 1152);
    TensorInfo srcInfo = {shape, TypeF16, KuDNN::Layout::ABC};
    TensorInfo statInfo = {{shape[0], shape[1]}, TypeF16, KuDNN::Layout::AB};
    TensorInfo scaleShiftInfo = {{shape[2]}, TypeF16, KuDNN::Layout::A};
    TensorInfo dstInfo = {shape, TypeF16, KuDNN::Layout::ABC};
    KuDNN::NormalizationFlags flags = KuDNN::NormalizationFlags::USE_SCALE;
    GTEST_EXPECT_TRUE(ForwardFunc1<__fp16>(srcInfo, statInfo, scaleShiftInfo, dstInfo, flags));
}

TEST(LnormGTests, ForwardFunc3DF16_3)
{
    Shape shape(2, 3240, 1152);
    TensorInfo srcInfo = {shape, TypeF16, KuDNN::Layout::ABC};
    TensorInfo statInfo = {{shape[0], shape[1]}, TypeF16, KuDNN::Layout::AB};
    TensorInfo scaleShiftInfo = {{shape[2]}, TypeF16, KuDNN::Layout::A};
    TensorInfo dstInfo = {shape, TypeF16, KuDNN::Layout::ABC};
    KuDNN::NormalizationFlags flags = KuDNN::NormalizationFlags::USE_SHIFT;
    GTEST_EXPECT_TRUE(ForwardFunc1<__fp16>(srcInfo, statInfo, scaleShiftInfo, dstInfo, flags));
}

TEST(LnormGTests, ForwardFunc3DF16_4)
{
    Shape shape(2, 3240, 1152);
    TensorInfo srcInfo = {shape, TypeF16, KuDNN::Layout::ABC};
    TensorInfo statInfo = {{shape[0], shape[1]}, TypeF16, KuDNN::Layout::AB};
    TensorInfo scaleShiftInfo = {{shape[2]}, TypeF16, KuDNN::Layout::A};
    TensorInfo dstInfo = {shape, TypeF16, KuDNN::Layout::ABC};
    KuDNN::NormalizationFlags flags = KuDNN::NormalizationFlags::USE_GLOBAL_STATS;
    GTEST_EXPECT_TRUE(ForwardFunc1<__fp16>(srcInfo, statInfo, scaleShiftInfo, dstInfo, flags));
}

TEST(LnormGTests, ForwardFunc4DF16_1)
{
    Shape shape(4, 14, 40, 44);
    TensorInfo srcInfo = {shape, TypeF16, KuDNN::Layout::ABCD};
    TensorInfo statInfo = {{shape[0], shape[1], shape[2]}, TypeF16, KuDNN::Layout::ABC};
    TensorInfo scaleShiftInfo = {{shape[3]}, TypeF16, KuDNN::Layout::A};
    TensorInfo dstInfo = {shape, TypeF16, KuDNN::Layout::ABCD};
    KuDNN::NormalizationFlags flags = KuDNN::NormalizationFlags::NONE;
    GTEST_EXPECT_TRUE(ForwardFunc1<__fp16>(srcInfo, statInfo, scaleShiftInfo, dstInfo, flags));
}

TEST(LnormGTests, ForwardFunc4DF16_2)
{
    Shape shape(4, 14, 40, 44);
    TensorInfo srcInfo = {shape, TypeF16, KuDNN::Layout::ABCD};
    TensorInfo statInfo = {{shape[0], shape[1], shape[2]}, TypeF16, KuDNN::Layout::ABC};
    TensorInfo scaleShiftInfo = {{shape[3]}, TypeF16, KuDNN::Layout::A};
    TensorInfo dstInfo = {shape, TypeF16, KuDNN::Layout::ABCD};
    KuDNN::NormalizationFlags flags = KuDNN::NormalizationFlags::USE_SCALE;
    GTEST_EXPECT_TRUE(ForwardFunc1<__fp16>(srcInfo, statInfo, scaleShiftInfo, dstInfo, flags));
}

TEST(LnormGTests, ForwardFunc4DF16_3)
{
    Shape shape(4, 14, 40, 44);
    TensorInfo srcInfo = {shape, TypeF16, KuDNN::Layout::ABCD};
    TensorInfo statInfo = {{shape[0], shape[1], shape[2]}, TypeF16, KuDNN::Layout::ABC};
    TensorInfo scaleShiftInfo = {{shape[3]}, TypeF16, KuDNN::Layout::A};
    TensorInfo dstInfo = {shape, TypeF16, KuDNN::Layout::ABCD};
    KuDNN::NormalizationFlags flags = KuDNN::NormalizationFlags::USE_SHIFT;
    GTEST_EXPECT_TRUE(ForwardFunc1<__fp16>(srcInfo, statInfo, scaleShiftInfo, dstInfo, flags));
}

TEST(LnormGTests, ForwardFunc4DF16_4)
{
    Shape shape(4, 14, 40, 44);
    TensorInfo srcInfo = {shape, TypeF16, KuDNN::Layout::ABCD};
    TensorInfo statInfo = {{shape[0], shape[1], shape[2]}, TypeF16, KuDNN::Layout::ABC};
    TensorInfo scaleShiftInfo = {{shape[3]}, TypeF16, KuDNN::Layout::A};
    TensorInfo dstInfo = {shape, TypeF16, KuDNN::Layout::ABCD};
    KuDNN::NormalizationFlags flags = KuDNN::NormalizationFlags::USE_GLOBAL_STATS;
    GTEST_EXPECT_TRUE(ForwardFunc1<__fp16>(srcInfo, statInfo, scaleShiftInfo, dstInfo, flags));
}

TEST(LnormGTests, ForwardFunc5DF16_1)
{
    Shape shape(5, 5, 5, 5, 5);
    TensorInfo srcInfo = {shape, TypeF16, KuDNN::Layout::ABCDE};
    TensorInfo statInfo = {{shape[0], shape[1], shape[2], shape[3]}, TypeF16, KuDNN::Layout::ABCD};
    TensorInfo scaleShiftInfo = {{shape[4]}, TypeF16, KuDNN::Layout::A};
    TensorInfo dstInfo = {shape, TypeF16, KuDNN::Layout::ABCDE};
    KuDNN::NormalizationFlags flags = KuDNN::NormalizationFlags::NONE;
    GTEST_EXPECT_TRUE(ForwardFunc1<__fp16>(srcInfo, statInfo, scaleShiftInfo, dstInfo, flags));
}

TEST(LnormGTests, ForwardFunc5DF16_2)
{
    Shape shape(5, 5, 5, 5, 5);
    TensorInfo srcInfo = {shape, TypeF16, KuDNN::Layout::ABCDE};
    TensorInfo statInfo = {{shape[0], shape[1], shape[2], shape[3]}, TypeF16, KuDNN::Layout::ABCD};
    TensorInfo scaleShiftInfo = {{shape[4]}, TypeF16, KuDNN::Layout::A};
    TensorInfo dstInfo = {shape, TypeF16, KuDNN::Layout::ABCDE};
    KuDNN::NormalizationFlags flags = KuDNN::NormalizationFlags::USE_SCALE;
    GTEST_EXPECT_TRUE(ForwardFunc1<__fp16>(srcInfo, statInfo, scaleShiftInfo, dstInfo, flags));
}

TEST(LnormGTests, ForwardFunc5DF16_3)
{
    Shape shape(5, 5, 5, 5, 5);
    TensorInfo srcInfo = {shape, TypeF16, KuDNN::Layout::ABCDE};
    TensorInfo statInfo = {{shape[0], shape[1], shape[2], shape[3]}, TypeF16, KuDNN::Layout::ABCD};
    TensorInfo scaleShiftInfo = {{shape[4]}, TypeF16, KuDNN::Layout::A};
    TensorInfo dstInfo = {shape, TypeF16, KuDNN::Layout::ABCDE};
    KuDNN::NormalizationFlags flags = KuDNN::NormalizationFlags::USE_SHIFT;
    GTEST_EXPECT_TRUE(ForwardFunc1<__fp16>(srcInfo, statInfo, scaleShiftInfo, dstInfo, flags));
}

TEST(LnormGTests, ForwardFunc5DF16_4)
{
    Shape shape(5, 5, 5, 5, 5);
    TensorInfo srcInfo = {shape, TypeF16, KuDNN::Layout::ABCDE};
    TensorInfo statInfo = {{shape[0], shape[1], shape[2], shape[3]}, TypeF16, KuDNN::Layout::ABCD};
    TensorInfo scaleShiftInfo = {{shape[4]}, TypeF16, KuDNN::Layout::A};
    TensorInfo dstInfo = {shape, TypeF16, KuDNN::Layout::ABCDE};
    KuDNN::NormalizationFlags flags = KuDNN::NormalizationFlags::USE_GLOBAL_STATS;
    GTEST_EXPECT_TRUE(ForwardFunc1<__fp16>(srcInfo, statInfo, scaleShiftInfo, dstInfo, flags));
}

TEST(LnormGTests, ForwardFunc2DF32_1)
{
    Shape shape(100, 100);
    TensorInfo srcInfo = {shape, TypeF32, KuDNN::Layout::AB};
    TensorInfo statInfo = {{shape[0]}, TypeF32, KuDNN::Layout::A};
    TensorInfo scaleShiftInfo = {{shape[1]}, TypeF32, KuDNN::Layout::A};
    TensorInfo dstInfo = {shape, TypeF32, KuDNN::Layout::AB};
    KuDNN::NormalizationFlags flags = KuDNN::NormalizationFlags::NONE;
    GTEST_EXPECT_TRUE(ForwardFunc1<float>(srcInfo, statInfo, scaleShiftInfo, dstInfo, flags));
}

TEST(LnormGTests, ForwardFunc2DF32_2)
{
    Shape shape(100, 100);
    TensorInfo srcInfo = {shape, TypeF32, KuDNN::Layout::AB};
    TensorInfo statInfo = {{shape[0]}, TypeF32, KuDNN::Layout::A};
    TensorInfo scaleShiftInfo = {{shape[1]}, TypeF32, KuDNN::Layout::A};
    TensorInfo dstInfo = {shape, TypeF32, KuDNN::Layout::AB};
    KuDNN::NormalizationFlags flags = KuDNN::NormalizationFlags::USE_SCALE;
    GTEST_EXPECT_TRUE(ForwardFunc1<float>(srcInfo, statInfo, scaleShiftInfo, dstInfo, flags));
}

TEST(LnormGTests, ForwardFunc2DF32_3)
{
    Shape shape(100, 100);
    TensorInfo srcInfo = {shape, TypeF32, KuDNN::Layout::AB};
    TensorInfo statInfo = {{shape[0]}, TypeF32, KuDNN::Layout::A};
    TensorInfo scaleShiftInfo = {{shape[1]}, TypeF32, KuDNN::Layout::A};
    TensorInfo dstInfo = {shape, TypeF32, KuDNN::Layout::AB};
    KuDNN::NormalizationFlags flags = KuDNN::NormalizationFlags::USE_SHIFT;
    GTEST_EXPECT_TRUE(ForwardFunc1<float>(srcInfo, statInfo, scaleShiftInfo, dstInfo, flags));
}

TEST(LnormGTests, ForwardFunc2DF32_4)
{
    Shape shape(100, 100);
    TensorInfo srcInfo = {shape, TypeF32, KuDNN::Layout::AB};
    TensorInfo statInfo = {{shape[0]}, TypeF32, KuDNN::Layout::A};
    TensorInfo scaleShiftInfo = {{shape[1]}, TypeF32, KuDNN::Layout::A};
    TensorInfo dstInfo = {shape, TypeF32, KuDNN::Layout::AB};
    KuDNN::NormalizationFlags flags = KuDNN::NormalizationFlags::USE_GLOBAL_STATS;
    GTEST_EXPECT_TRUE(ForwardFunc1<float>(srcInfo, statInfo, scaleShiftInfo, dstInfo, flags));
}

TEST(LnormGTests, ForwardFunc2DF32_5)
{
    Shape shape(100, 100);
    TensorInfo srcInfo = {shape, TypeF32, KuDNN::Layout::AB};
    TensorInfo statInfo = {{shape[0]}, TypeF32, KuDNN::Layout::A};
    TensorInfo scaleShiftInfo = {{shape[1]}, TypeF32, KuDNN::Layout::A};
    TensorInfo dstInfo = {shape, TypeF32, KuDNN::Layout::AB};
    KuDNN::NormalizationFlags flags = KuDNN::NormalizationFlags::USE_SCALE | KuDNN::NormalizationFlags::USE_SHIFT;
    GTEST_EXPECT_TRUE(ForwardFunc1<float>(srcInfo, statInfo, scaleShiftInfo, dstInfo, flags));
}

TEST(LnormGTests, ForwardFunc3DF32_1)
{
    Shape shape(2, 3240, 1152);
    TensorInfo srcInfo = {shape, TypeF32, KuDNN::Layout::ABC};
    TensorInfo statInfo = {{shape[0], shape[1]}, TypeF32, KuDNN::Layout::AB};
    TensorInfo scaleShiftInfo = {{shape[2]}, TypeF32, KuDNN::Layout::A};
    TensorInfo dstInfo = {shape, TypeF32, KuDNN::Layout::ABC};
    KuDNN::NormalizationFlags flags = KuDNN::NormalizationFlags::NONE;
    GTEST_EXPECT_TRUE(ForwardFunc1<float>(srcInfo, statInfo, scaleShiftInfo, dstInfo, flags));
}

TEST(LnormGTests, ForwardFunc3DF32_2)
{
    Shape shape(2, 3240, 1152);
    TensorInfo srcInfo = {shape, TypeF32, KuDNN::Layout::ABC};
    TensorInfo statInfo = {{shape[0], shape[1]}, TypeF32, KuDNN::Layout::AB};
    TensorInfo scaleShiftInfo = {{shape[2]}, TypeF32, KuDNN::Layout::A};
    TensorInfo dstInfo = {shape, TypeF32, KuDNN::Layout::ABC};
    KuDNN::NormalizationFlags flags = KuDNN::NormalizationFlags::USE_SCALE;
    GTEST_EXPECT_TRUE(ForwardFunc1<float>(srcInfo, statInfo, scaleShiftInfo, dstInfo, flags));
}

TEST(LnormGTests, ForwardFunc3DF32_3)
{
    Shape shape(2, 3240, 1152);
    TensorInfo srcInfo = {shape, TypeF32, KuDNN::Layout::ABC};
    TensorInfo statInfo = {{shape[0], shape[1]}, TypeF32, KuDNN::Layout::AB};
    TensorInfo scaleShiftInfo = {{shape[2]}, TypeF32, KuDNN::Layout::A};
    TensorInfo dstInfo = {shape, TypeF32, KuDNN::Layout::ABC};
    KuDNN::NormalizationFlags flags = KuDNN::NormalizationFlags::USE_SHIFT;
    GTEST_EXPECT_TRUE(ForwardFunc1<float>(srcInfo, statInfo, scaleShiftInfo, dstInfo, flags));
}

TEST(LnormGTests, ForwardFunc3DF32_4)
{
    Shape shape(2, 3240, 1152);
    TensorInfo srcInfo = {shape, TypeF32, KuDNN::Layout::ABC};
    TensorInfo statInfo = {{shape[0], shape[1]}, TypeF32, KuDNN::Layout::AB};
    TensorInfo scaleShiftInfo = {{shape[2]}, TypeF32, KuDNN::Layout::A};
    TensorInfo dstInfo = {shape, TypeF32, KuDNN::Layout::ABC};
    KuDNN::NormalizationFlags flags = KuDNN::NormalizationFlags::USE_GLOBAL_STATS;
    GTEST_EXPECT_TRUE(ForwardFunc1<float>(srcInfo, statInfo, scaleShiftInfo, dstInfo, flags));
}


TEST(LnormGTests, ForwardFunc4DF32_1)
{
    Shape shape(14, 4, 44, 40);
    TensorInfo srcInfo = {shape, TypeF32, KuDNN::Layout::ABCD};
    TensorInfo statInfo = {{shape[0], shape[1], shape[2]}, TypeF32, KuDNN::Layout::ABC};
    TensorInfo scaleShiftInfo = {{shape[3]}, TypeF32, KuDNN::Layout::A};
    TensorInfo dstInfo = {shape, TypeF32, KuDNN::Layout::ABCD};
    KuDNN::NormalizationFlags flags = KuDNN::NormalizationFlags::NONE;
    GTEST_EXPECT_TRUE(ForwardFunc1<float>(srcInfo, statInfo, scaleShiftInfo, dstInfo, flags));
}

TEST(LnormGTests, ForwardFunc4DF32_2)
{
    Shape shape(14, 4, 44, 40);
    TensorInfo srcInfo = {shape, TypeF32, KuDNN::Layout::ABCD};
    TensorInfo statInfo = {{shape[0], shape[1], shape[2]}, TypeF32, KuDNN::Layout::ABC};
    TensorInfo scaleShiftInfo = {{shape[3]}, TypeF32, KuDNN::Layout::A};
    TensorInfo dstInfo = {shape, TypeF32, KuDNN::Layout::ABCD};
    KuDNN::NormalizationFlags flags = KuDNN::NormalizationFlags::USE_SCALE;
    GTEST_EXPECT_TRUE(ForwardFunc1<float>(srcInfo, statInfo, scaleShiftInfo, dstInfo, flags));
}

TEST(LnormGTests, ForwardFunc4DF32_3)
{
    Shape shape(14, 4, 44, 40);
    TensorInfo srcInfo = {shape, TypeF32, KuDNN::Layout::ABCD};
    TensorInfo statInfo = {{shape[0], shape[1], shape[2]}, TypeF32, KuDNN::Layout::ABC};
    TensorInfo scaleShiftInfo = {{shape[3]}, TypeF32, KuDNN::Layout::A};
    TensorInfo dstInfo = {shape, TypeF32, KuDNN::Layout::ABCD};
    KuDNN::NormalizationFlags flags = KuDNN::NormalizationFlags::USE_SHIFT;
    GTEST_EXPECT_TRUE(ForwardFunc1<float>(srcInfo, statInfo, scaleShiftInfo, dstInfo, flags));
}

TEST(LnormGTests, ForwardFunc4DF32_4)
{
    Shape shape(14, 4, 44, 40);
    TensorInfo srcInfo = {shape, TypeF32, KuDNN::Layout::ABCD};
    TensorInfo statInfo = {{shape[0], shape[1], shape[2]}, TypeF32, KuDNN::Layout::ABC};
    TensorInfo scaleShiftInfo = {{shape[3]}, TypeF32, KuDNN::Layout::A};
    TensorInfo dstInfo = {shape, TypeF32, KuDNN::Layout::ABCD};
    KuDNN::NormalizationFlags flags = KuDNN::NormalizationFlags::USE_GLOBAL_STATS;
    GTEST_EXPECT_TRUE(ForwardFunc1<float>(srcInfo, statInfo, scaleShiftInfo, dstInfo, flags));
}

TEST(LnormGTests, ForwardFunc5DF32_1)
{
    Shape shape(5, 5, 5, 5, 5);
    TensorInfo srcInfo = {shape, TypeF32, KuDNN::Layout::ABCDE};
    TensorInfo statInfo = {{shape[0], shape[1], shape[2], shape[3]}, TypeF32, KuDNN::Layout::ABCD};
    TensorInfo scaleShiftInfo = {{shape[4]}, TypeF32, KuDNN::Layout::A};
    TensorInfo dstInfo = {shape, TypeF32, KuDNN::Layout::ABCDE};
    KuDNN::NormalizationFlags flags = KuDNN::NormalizationFlags::NONE;
    GTEST_EXPECT_TRUE(ForwardFunc1<float>(srcInfo, statInfo, scaleShiftInfo, dstInfo, flags));
}

TEST(LnormGTests, ForwardFunc5DF32_2)
{
    Shape shape(5, 5, 5, 5, 5);
    TensorInfo srcInfo = {shape, TypeF32, KuDNN::Layout::ABCDE};
    TensorInfo statInfo = {{shape[0], shape[1], shape[2], shape[3]}, TypeF32, KuDNN::Layout::ABCD};
    TensorInfo scaleShiftInfo = {{shape[4]}, TypeF32, KuDNN::Layout::A};
    TensorInfo dstInfo = {shape, TypeF32, KuDNN::Layout::ABCDE};
    KuDNN::NormalizationFlags flags = KuDNN::NormalizationFlags::USE_SCALE;
    GTEST_EXPECT_TRUE(ForwardFunc1<float>(srcInfo, statInfo, scaleShiftInfo, dstInfo, flags));
}

TEST(LnormGTests, ForwardFunc5DF32_3)
{
    Shape shape(5, 5, 5, 5, 5);
    TensorInfo srcInfo = {shape, TypeF32, KuDNN::Layout::ABCDE};
    TensorInfo statInfo = {{shape[0], shape[1], shape[2], shape[3]}, TypeF32, KuDNN::Layout::ABCD};
    TensorInfo scaleShiftInfo = {{shape[4]}, TypeF32, KuDNN::Layout::A};
    TensorInfo dstInfo = {shape, TypeF32, KuDNN::Layout::ABCDE};
    KuDNN::NormalizationFlags flags = KuDNN::NormalizationFlags::USE_SHIFT;
    GTEST_EXPECT_TRUE(ForwardFunc1<float>(srcInfo, statInfo, scaleShiftInfo, dstInfo, flags));
}

TEST(LnormGTests, ForwardFunc5DF32_4)
{
    Shape shape(5, 5, 5, 5, 5);
    TensorInfo srcInfo = {shape, TypeF32, KuDNN::Layout::ABCDE};
    TensorInfo statInfo = {{shape[0], shape[1], shape[2], shape[3]}, TypeF32, KuDNN::Layout::ABCD};
    TensorInfo scaleShiftInfo = {{shape[4]}, TypeF32, KuDNN::Layout::A};
    TensorInfo dstInfo = {shape, TypeF32, KuDNN::Layout::ABCDE};
    KuDNN::NormalizationFlags flags = KuDNN::NormalizationFlags::USE_GLOBAL_STATS;
    GTEST_EXPECT_TRUE(ForwardFunc1<float>(srcInfo, statInfo, scaleShiftInfo, dstInfo, flags));
}

TEST(LnormGTests, ForwardFunc5DF32_5)
{
    Shape shape(5, 5, 5, 5, 5);
    TensorInfo srcInfo = {shape, TypeF32, KuDNN::Layout::ACDEB};
    TensorInfo statInfo = {{shape[0], shape[1], shape[2], shape[3]}, TypeF32, KuDNN::Layout::ABCD};
    TensorInfo scaleShiftInfo = {{shape[4]}, TypeF32, KuDNN::Layout::A};
    TensorInfo dstInfo = {shape, TypeF32, KuDNN::Layout::ACDEB};
    KuDNN::NormalizationFlags flags = KuDNN::NormalizationFlags::USE_GLOBAL_STATS;
    GTEST_EXPECT_TRUE(ForwardFunc1<float>(srcInfo, statInfo, scaleShiftInfo, dstInfo, flags));
}

TEST(LnormGTests, ForwardFunc5DF32_6)
{
    Shape shape(5, 5, 5, 5, 5);
    TensorInfo srcInfo = {shape, TypeF32, KuDNN::Layout::ACDEB};
    TensorInfo statInfo = {{shape[0], shape[1], shape[2], shape[3]}, TypeF32, KuDNN::Layout::ABCD};
    TensorInfo scaleShiftInfo = {{shape[4]}, TypeF32, KuDNN::Layout::A};
    TensorInfo dstInfo = {shape, TypeF32, KuDNN::Layout::ACDEB};
    KuDNN::NormalizationFlags flags = KuDNN::NormalizationFlags::USE_SHIFT;
    GTEST_EXPECT_TRUE(ForwardFunc1<float>(srcInfo, statInfo, scaleShiftInfo, dstInfo, flags));
}

TEST(LnormGTests, ForwardFunc5DF32F16_4)
{
    Shape shape(5, 5, 5, 5, 5);
    TensorInfo srcInfo = {shape, TypeF32, KuDNN::Layout::ABCDE};
    TensorInfo statInfo = {{shape[0], shape[1], shape[2], shape[3]}, TypeF32, KuDNN::Layout::ABDC};
    TensorInfo scaleShiftInfo = {{shape[4]}, TypeF16, KuDNN::Layout::A};
    TensorInfo dstInfo = {shape, TypeF32, KuDNN::Layout::ACBDE};
    KuDNN::NormalizationFlags flags = KuDNN::NormalizationFlags::NONE;
    GTEST_EXPECT_TRUE(ForwardFunc1<float>(srcInfo, statInfo, scaleShiftInfo, dstInfo, flags));
}


TEST(LnormGTests, FWDSaveStats)
{
    Shape shape(5, 5, 5, 5, 5);
    TensorInfo srcInfo = {shape, TypeF32, KuDNN::Layout::ABCDE};
    TensorInfo statInfo = {{shape[0], shape[1], shape[2], shape[3]}, TypeF32, KuDNN::Layout::ABCD};
    TensorInfo scaleShiftInfo = {{shape[4]}, TypeF32, KuDNN::Layout::A};
    TensorInfo dstInfo = {shape, TypeF32, KuDNN::Layout::ABCDE};
    KuDNN::NormalizationFlags flags = KuDNN::NormalizationFlags::NONE;
    GTEST_EXPECT_TRUE(FWDSaveStats<float>(srcInfo, statInfo, scaleShiftInfo, dstInfo, flags));
}


TEST(LnormGTests, BackwardFunc5DF32_1)
{
    const KuDNN::TensorInfo srcInfo = {{10, 10, 10, 10, 10}, KuDNN::Element::TypeT::F16, KuDNN::Layout::ABCED};
    const KuDNN::TensorInfo diffSrcInfo = {{10, 10, 10, 10, 10}, KuDNN::Element::TypeT::F16, KuDNN::Layout::ABCED};
    const KuDNN::TensorInfo statInfo = {{10, 10, 10, 10}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCD};
    const KuDNN::TensorInfo scaleShiftInfo = {{10}, KuDNN::Element::TypeT::BF16, KuDNN::Layout::A};
    const KuDNN::TensorInfo diffScaleShiftInfo = {{10}, KuDNN::Element::TypeT::BF16    , KuDNN::Layout::A};
    const KuDNN::TensorInfo diffDstInfo = {{10, 10, 10, 10, 10}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCDE};
    KuDNN::NormalizationFlags flags = KuDNN::NormalizationFlags::NONE;
    GTEST_EXPECT_TRUE(BackwardFunc1<float>(srcInfo, statInfo, diffSrcInfo, diffDstInfo, scaleShiftInfo,
        diffScaleShiftInfo, flags));
    flags = KuDNN::NormalizationFlags::USE_SCALE | KuDNN::NormalizationFlags::USE_SHIFT;
    GTEST_EXPECT_TRUE(BackwardFunc1<float>(srcInfo, statInfo, diffSrcInfo, diffDstInfo, scaleShiftInfo,
        diffScaleShiftInfo, flags));
}

TEST(LnormGTests, BackwardFunc5DF32_2)
{
    const KuDNN::TensorInfo srcInfo = {{10, 10, 10, 10, 10}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCDE};
    const KuDNN::TensorInfo diffSrcInfo = {{10, 10, 10, 10, 10}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCDE};
    const KuDNN::TensorInfo statInfo = {{10, 10, 10, 10}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCD};
    const KuDNN::TensorInfo scaleShiftInfo = {{10}, KuDNN::Element::TypeT::F32, KuDNN::Layout::A};
    const KuDNN::TensorInfo diffScaleShiftInfo = {{10}, KuDNN::Element::TypeT::F32    , KuDNN::Layout::A};
    const KuDNN::TensorInfo diffDstInfo = {{10, 10, 10, 10, 10}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCDE};
    KuDNN::NormalizationFlags flags = KuDNN::NormalizationFlags::NONE;
    GTEST_EXPECT_TRUE(BackwardFunc1<float>(srcInfo, statInfo, diffSrcInfo, diffDstInfo, scaleShiftInfo,
        diffScaleShiftInfo, flags));
    flags = KuDNN::NormalizationFlags::USE_SCALE | KuDNN::NormalizationFlags::USE_SHIFT;
    GTEST_EXPECT_TRUE(BackwardFunc1<float>(srcInfo, statInfo, diffSrcInfo, diffDstInfo, scaleShiftInfo,
        diffScaleShiftInfo, flags));
}

TEST(LnormGTests, FWDSrcDstType)
{
    EXPECT_ANY_THROW(FWDSrcDstType());
}

TEST(LnormGTests, FWDScaleShiftType)
{
    EXPECT_ANY_THROW(FWDScaleShiftType());
}

TEST(LnormGTests, FWDGlobalStatsType)
{
    EXPECT_NO_THROW(FWDGlobalStatsType());
}

TEST(LnormGTests, FWDFlagType)
{
    EXPECT_THROW(FWDFlagType(), KuDNN::Service::Unsupported);
}

TEST(LnormGTests, FWDCopyConstructor1)
{
    EXPECT_NO_THROW(ForwardCopyConstructor1());
}

TEST(LnormGTests, FWDCopyConstructor2)
{
    EXPECT_NO_THROW(ForwardCopyConstructor2());
}

TEST(LnormGTests, FWDMoveConstructor1)
{
    EXPECT_NO_THROW(ForwardMoveConstructor1());
}

TEST(LnormGTests, FWDMoveConstructor2)
{
    EXPECT_NO_THROW(ForwardMoveConstructor2());
}

TEST(LnormGTests, BWDCopyConstructor1)
{
    EXPECT_NO_THROW(BackwardCopyConstructor1());
}

TEST(LnormGTests, BWDCopyConstructor2)
{
    EXPECT_NO_THROW(BackwardCopyConstructor2());
}

TEST(LnormGTests, BWDMoveConstructor1)
{
    EXPECT_NO_THROW(BackwardMoveConstructor1());
}

TEST(LnormGTests, BWDMoveConstructor2)
{
    EXPECT_NO_THROW(BackwardMoveConstructor2());
}
