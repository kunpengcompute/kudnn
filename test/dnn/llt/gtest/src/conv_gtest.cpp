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
#include <map>
#include <string>

#include <arm_neon.h>

#include "gtest/gtest.h"
#include "kudnn.hpp"
#include "kudnn_tensor_data.hpp"

using SizeType = KuDNN::SizeType;
using Shape = KuDNN::Shape;
const int MAX_THS = 100;
const float FACTOR_THS = 0.02;
float errBoundConv = 1e-3;

template <typename T>
struct typeMap;

template <>
struct typeMap<__fp16> {
    static constexpr KuDNN::Element::TypeT val = KuDNN::Element::TypeT::F16;
};

template <>
struct typeMap<float> {
    static constexpr KuDNN::Element::TypeT val = KuDNN::Element::TypeT::F32;
};

template <typename T>
static void conv2dRef(Shape &srcShape, Shape &weiShape, Shape &dstShape, Shape &strides, Shape &dilates,
    Shape &paddingL, Shape &paddingR, const T *src, const T *wei, T *dst, const T *bia)
{
    SizeType N = srcShape[0], IC = srcShape[1], IH = srcShape[2], IW = srcShape[3];
    SizeType KH = weiShape[2], KW = weiShape[3];
    SizeType OC = dstShape[1], OH = dstShape[2], OW = dstShape[3];
    SizeType SH = strides[0], SW = strides[1];
    SizeType DH = dilates[0], DW = dilates[1];
    SizeType PH_L = paddingL[0], PW_L = paddingL[1];
    SizeType PH_R = paddingR[0], PW_R = paddingR[1];
    OH = (IH + PH_L + PH_R - 1 - (KH - 1) * (DH + 1)) / SH + 1;
    OW = (IW + PW_L + PW_R - 1 - (KW - 1) * (DW + 1)) / SW + 1;
    int threadsUsed = (int)ceil((float)(N * OC * OH * OW) * FACTOR_THS);
    threadsUsed = threadsUsed > MAX_THS ? MAX_THS : threadsUsed;
#pragma omp parallel for collapse(4) num_threads(threadsUsed)
    for (SizeType n = 0; n < N; ++n) {
        for (SizeType oc = 0; oc < OC; ++oc) {
            T biaVal = *(bia + oc);
            for (SizeType oh = 0; oh < OH; ++oh) {
                for (SizeType ow = 0; ow < OW; ++ow) {
                    T sum = (T)0.0f;
                    for (SizeType ic = 0; ic < IC; ++ic) {
                        for (SizeType kh = 0; kh < KH; ++kh) {
                            for (SizeType kw = 0; kw < KW; ++kw) {
                                SizeType srcX = oh * SH + kh * (DH + 1) - PH_L;
                                SizeType srcY = ow * SW + kw * (DW + 1) - PW_L;
                                T srcVal = (T)0.0f;
                                if (srcX >= 0 && srcX < IH && srcY >= 0 && srcY < IW) {
                                    srcVal = *(src + n * IC * IH * IW + ic * IH * IW + srcX * IW + srcY);
                                }
                                T weiVal = *(wei + oc * IC * KH * KW + ic * KH * KW + kh * KW + kw);
                                sum += srcVal * weiVal;
                            }
                        }
                    }
                    *(dst + n * OC * OH * OW + oc * OH * OW + oh * OW + ow) = sum + biaVal;
                }
            }
        }
    }
}

template <typename T>
static void conv3dRef(Shape &srcShape, Shape &weiShape, Shape &dstShape, Shape &strides, Shape &dilates,
    Shape &paddingL, Shape &paddingR, const T *src, const T *wei, T *dst, const T *bia)
{
    SizeType N = srcShape[0], IC = srcShape[1], ID = srcShape[2], IH = srcShape[3], IW = srcShape[4];
    SizeType KD = weiShape[2], KH = weiShape[3], KW = weiShape[4];
    SizeType OC = dstShape[1], OD = dstShape[2], OH = dstShape[2], OW = dstShape[3];
    SizeType SD = strides[0], SH = strides[1], SW = strides[2];
    SizeType DD = dilates[0], DH = dilates[1], DW = dilates[2];
    SizeType PD_L = paddingL[0], PH_L = paddingL[1], PW_L = paddingL[2];
    SizeType PD_R = paddingR[0], PH_R = paddingR[1], PW_R = paddingR[2];
    OD = (ID + PD_L + PD_R - 1 - (KD - 1) * (DD + 1)) / SD + 1;
    OH = (IH + PH_L + PH_R - 1 - (KH - 1) * (DH + 1)) / SH + 1;
    OW = (IW + PW_L + PW_R - 1 - (KW - 1) * (DW + 1)) / SW + 1;
    int threadsUsed = (int)ceil((float)(N * OC * OD * OH * OW) * FACTOR_THS);
    threadsUsed = threadsUsed > MAX_THS ? MAX_THS : threadsUsed;
#pragma omp parallel for collapse(5) num_threads(threadsUsed)
    for (SizeType n = 0; n < N; ++n) {
        for (SizeType oc = 0; oc < OC; ++oc) {
            T biaVal = *(bia + oc);
            for (SizeType od = 0; od < OD; ++od) {
                for (SizeType oh = 0; oh < OH; ++oh) {
                    for (SizeType ow = 0; ow < OW; ++ow) {
                        T sum = (T)0.0f;
                        for (SizeType ic = 0; ic < IC; ++ic) {
                            for (SizeType kd = 0; kd < KD; ++kd) {
                                for (SizeType kh = 0; kh < KH; ++kh) {
                                    for (SizeType kw = 0; kw < KW; ++kw) {
                                        SizeType srcZ = od * SD + kd * (DD + 1) - PD_L;
                                        SizeType srcX = oh * SH + kh * (DH + 1) - PH_L;
                                        SizeType srcY = ow * SW + kw * (DW + 1) - PW_L;
                                        T srcVal = (T)0.0f;
                                        if (srcZ >= 0 && srcZ < ID && srcX >= 0 && srcX < IH && srcY >= 0 &&
                                            srcY < IW) {
                                            srcVal = *(src + n * IC * ID * IH * IW + ic * ID * IH * IW +
                                                       srcZ * IH * IW + srcX * IW + srcY);
                                        }
                                        T weiVal = *(wei + oc * IC * KD * KH * KW + ic * KD * KH * KW + kd * KH * KW +
                                                     kh * KW + kw);
                                        sum += srcVal * weiVal;
                                    }
                                }
                            }
                        }
                        *(dst + n * OC * OD * OH * OW + oc * OD * OH * OW + od * OH * OW + oh * OW + ow) = sum + biaVal;
                    }
                }
            }
        }
    }
}
template <typename T>
static bool Conv2dBWDWeightFunc1(Shape &diffDstShape, Shape &srcShape, Shape &diffWeightsShape, Shape &diffBiasShape,
            Shape &strides, Shape &dilates, Shape &paddingL, Shape &paddingR, KuDNN::ConvolutionAlgorithm &alg)
{
    SizeType N = srcShape[0], IC = srcShape[1], IH = srcShape[2], IW = srcShape[3];
    SizeType OC = diffDstShape[1], OH = diffDstShape[2], OW = diffDstShape[3];
    SizeType KH = diffWeightsShape[2], KW = diffWeightsShape[3];


    const KuDNN::TensorInfo diffDstTensor = {diffDstShape, typeMap<T>::val, KuDNN::Layout::ABCD};
    const KuDNN::TensorInfo srcTensor = {srcShape, typeMap<T>::val, KuDNN::Layout::ABCD};
    const KuDNN::TensorInfo diffWeightsTensor = {diffWeightsShape, typeMap<T>::val, KuDNN::Layout::ABCD};
    const KuDNN::TensorInfo diffBiasTensor = {diffBiasShape, typeMap<T>::val, KuDNN::Layout::A};

    KuDNN::ConvolutionLayerBWDWeights convBwdLayer1(diffDstTensor, srcTensor, diffWeightsTensor, diffBiasTensor, strides, dilates, paddingL, paddingR, alg);

    SizeType srcSize = N * IC * IH * IW;
    SizeType dstSize = N * OC * OH * OW;
    SizeType weiSize = OC * IC * KH * KW;
    SizeType biaSize = OC;

    T *diffWeights = (T *)malloc(weiSize * sizeof(T));
    T *diffBias = (T *)malloc(biaSize * sizeof(T));
    T *diffDst = (T *)malloc(dstSize * sizeof(T));
    T *src = (T *)malloc(srcSize * sizeof(T));
    
    convBwdLayer1.Run(diffDst, src, diffWeights, diffBias);
    
    return true;
}

template <typename T>
static bool Conv2dBWDDataFunc1(Shape &diffDstShape, Shape &weightsShape, Shape &diffSrcShape, Shape &strides, Shape &dilates,
    Shape &paddingL, Shape &paddingR, KuDNN::ConvolutionAlgorithm &alg)
{
    SizeType N = diffSrcShape[0], IC = diffSrcShape[1], IH = diffSrcShape[2], IW = diffSrcShape[3];
    SizeType OC = diffDstShape[1], OH = diffDstShape[2], OW = diffDstShape[3];
    SizeType KH = weightsShape[2], KW = weightsShape[3];

    const KuDNN::TensorInfo diffDstTensor = {diffDstShape, typeMap<T>::val, KuDNN::Layout::ABCD};
    const KuDNN::TensorInfo weightsTensor = {weightsShape, typeMap<T>::val, KuDNN::Layout::ABCD};
    const KuDNN::TensorInfo diffSrcTensor = {diffSrcShape, typeMap<T>::val, KuDNN::Layout::ABCD};
    
    KuDNN::ConvolutionLayerBWDData convBwdLayer1(diffDstTensor, weightsTensor, diffSrcTensor, strides, dilates, paddingL, paddingR, alg);

    SizeType srcSize = N * IC * IH * IW;
    SizeType dstSize = N * OC * OH * OW;
    SizeType weiSize = OC * IC * KH * KW;

    T *diffSrc = (T *)malloc(srcSize * sizeof(T));
    T *diffDst = (T *)malloc(dstSize * sizeof(T));
    T *weights = (T *)malloc(weiSize * sizeof(T));

    convBwdLayer1.Run(diffDst, weights, diffSrc);

    return true;

}

template <typename T>
static bool Conv2dFWDFunc1(Shape &srcShape, Shape &weiShape, Shape &dstShape, Shape &strides, Shape &dilates,
    Shape &paddingL, Shape &paddingR, KuDNN::ConvolutionAlgorithm &alg)
{
    SizeType N = srcShape[0], IC = srcShape[1], IH = srcShape[2], IW = srcShape[3];
    SizeType OC = dstShape[1], OH = dstShape[2], OW = dstShape[3];
    SizeType KH = weiShape[2], KW = weiShape[3];
    const KuDNN::TensorInfo srcTensor = {srcShape, typeMap<T>::val, KuDNN::Layout::ABCD};
    const KuDNN::TensorInfo weightsTensor = {weiShape, typeMap<T>::val, KuDNN::Layout::ABCD};
    const KuDNN::TensorInfo dstTensor = {dstShape, typeMap<T>::val, KuDNN::Layout::ABCD};
    const KuDNN::TensorInfo biasTensor = {{OC}, typeMap<T>::val, KuDNN::Layout::A};

    KuDNN::ConvolutionLayerFWD convFwdLayer1(
        srcTensor, weightsTensor, dstTensor, biasTensor, strides, dilates, paddingL, paddingR, alg);

    SizeType srcSize = N * IC * IH * IW;
    SizeType dstSize = N * OC * OH * OW;
    SizeType weiSize = OC * IC * KH * KW;
    SizeType biaSize = OC;
    T *src = (T *)malloc(srcSize * sizeof(T));
    T *dst = (T *)malloc(dstSize * sizeof(T));
    T *dstRef = (T *)malloc(dstSize * sizeof(T));
    T *wei = (T *)malloc(weiSize * sizeof(T));
    T *bia = (T *)malloc(biaSize * sizeof(T));
    if (src == nullptr || dst == nullptr || dstRef == nullptr || wei == nullptr || bia == nullptr) {
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
    for (SizeType i = 0; i < weiSize; ++i) {
        *(wei + i) = (T)u(e);
    }
    for (SizeType i = 0; i < biaSize; ++i) {
        *(bia + i) = (T)u(e);
    }
    convFwdLayer1.Run(src, wei, dst, bia);
    conv2dRef<T>(srcShape, weiShape, dstShape, strides, dilates, paddingL, paddingR, src, wei, dstRef, bia);
    float error = 0.0;
#pragma omp parallel for reduction(+ : error) num_threads(threadsUsed)
    for (SizeType i = 0; i < dstSize; ++i) {
        error += *(dst + i) - *(dstRef + i);
    }
    free(src);
    free(dst);
    free(dstRef);
    free(wei);
    free(bia);
    error = std::abs(error) / dstSize;
    std::cout << "run error: " << error << std::endl;
    return error < errBoundConv;
}


template <typename T>
static bool Conv3dBWDWeightFunc1(Shape &diffDstShape, Shape &srcShape, Shape &diffWeightsShape, Shape &diffBiasShape,
            Shape &strides, Shape &dilates, Shape &paddingL, Shape &paddingR, KuDNN::ConvolutionAlgorithm &alg)
{
    SizeType N = srcShape[0], IC = srcShape[1], ID = srcShape[2], IH = srcShape[3], IW = srcShape[4];
    SizeType OC = diffDstShape[1], OD = diffDstShape[2], OH = diffDstShape[3], OW = diffDstShape[4];
    SizeType KD = diffWeightsShape[2], KH = diffWeightsShape[3], KW = diffWeightsShape[4];

    const KuDNN::TensorInfo diffDstTensor = {diffDstShape, typeMap<T>::val, KuDNN::Layout::ABCDE};
    const KuDNN::TensorInfo srcTensor = {srcShape, typeMap<T>::val, KuDNN::Layout::ABCDE};
    const KuDNN::TensorInfo diffWeightsTensor = {diffWeightsShape, typeMap<T>::val, KuDNN::Layout::ABCDE};
    const KuDNN::TensorInfo diffBiasTensor = {diffBiasShape, typeMap<T>::val, KuDNN::Layout::A};

    KuDNN::ConvolutionLayerBWDWeights convBwdLayer1(diffDstTensor, srcTensor, diffWeightsTensor, diffBiasTensor, strides, dilates, paddingL, paddingR, alg);

    SizeType srcSize = N * IC * IH * IW * ID;
    SizeType dstSize = N * OC * OH * OW * OD;
    SizeType weiSize = OC * IC * KH * KW * KD;
    SizeType biaSize = OC;

    T *diffWeights = (T *)malloc(weiSize * sizeof(T));
    T *diffBias = (T *)malloc(biaSize * sizeof(T));
    T *diffDst = (T *)malloc(dstSize * sizeof(T));
    T *src = (T *)malloc(srcSize * sizeof(T));

    convBwdLayer1.Run(diffDst, src, diffWeights, diffBias);

    return true;
}


template <typename T>
static bool Conv3dBWDDataFunc1(Shape &diffDstShape, Shape &weightsShape, Shape &diffSrcShape, Shape &strides, Shape &dilates,
    Shape &paddingL, Shape &paddingR, KuDNN::ConvolutionAlgorithm &alg)
{
    SizeType N = diffSrcShape[0], IC = diffSrcShape[1], ID = diffSrcShape[2], IH = diffSrcShape[3], IW = diffSrcShape[4];
    SizeType OC = diffDstShape[1], OD = diffDstShape[2], OH = diffDstShape[3], OW = diffDstShape[4];
    SizeType KD = weightsShape[2], KH = weightsShape[3], KW = weightsShape[4];

    const KuDNN::TensorInfo diffDstTensor = {diffDstShape, typeMap<T>::val, KuDNN::Layout::ABCDE};
    const KuDNN::TensorInfo weightsTensor = {weightsShape, typeMap<T>::val, KuDNN::Layout::ABCDE};
    const KuDNN::TensorInfo diffSrcTensor = {diffSrcShape, typeMap<T>::val, KuDNN::Layout::ABCDE};
    
    KuDNN::ConvolutionLayerBWDData convBwdLayer1(diffDstTensor, weightsTensor, diffSrcTensor, strides, dilates, paddingL, paddingR, alg);

    SizeType srcSize = N * IC * IH * IW * ID;
    SizeType dstSize = N * OC * OH * OW * OD;
    SizeType weiSize = OC * IC * KH * KW * KD;

    T *diffSrc = (T *)malloc(srcSize * sizeof(T));
    T *diffDst = (T *)malloc(dstSize * sizeof(T));
    T *weights = (T *)malloc(weiSize * sizeof(T));

    convBwdLayer1.Run(diffDst, weights, diffSrc);

    return true;

}


template <typename T>
static bool Conv3dFWDFunc1(Shape &srcShape, Shape &weiShape, Shape &dstShape, Shape &strides, Shape &dilates,
    Shape &paddingL, Shape &paddingR, KuDNN::ConvolutionAlgorithm &alg)
{
    SizeType N = srcShape[0], IC = srcShape[1], ID = srcShape[2], IH = srcShape[3], IW = srcShape[4];
    SizeType OC = dstShape[1], OD = dstShape[2], OH = dstShape[3], OW = dstShape[4];
    SizeType KD = weiShape[2], KH = weiShape[3], KW = weiShape[4];
    const KuDNN::TensorInfo srcTensor = {srcShape, typeMap<T>::val, KuDNN::Layout::ABCDE};
    const KuDNN::TensorInfo weightsTensor = {weiShape, typeMap<T>::val, KuDNN::Layout::ABCDE};
    const KuDNN::TensorInfo dstTensor = {dstShape, typeMap<T>::val, KuDNN::Layout::ABCDE};
    const KuDNN::TensorInfo biasTensor = {{OC}, typeMap<T>::val, KuDNN::Layout::A};

    KuDNN::ConvolutionLayerFWD convFwdLayer1(
        srcTensor, weightsTensor, dstTensor, biasTensor, strides, dilates, paddingL, paddingR, alg);

    SizeType srcSize = N * IC * ID * IH * IW;
    SizeType dstSize = N * OC * OD * OH * OW;
    SizeType weiSize = OC * IC * KD * KH * KW;
    SizeType biaSize = OC;
    T *src = (T *)malloc(srcSize * sizeof(T));
    T *dst = (T *)malloc(dstSize * sizeof(T));
    T *dstRef = (T *)malloc(dstSize * sizeof(T));
    T *wei = (T *)malloc(weiSize * sizeof(T));
    T *bia = (T *)malloc(biaSize * sizeof(T));
    if (src == nullptr || dst == nullptr || dstRef == nullptr || wei == nullptr || bia == nullptr) {
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
    for (SizeType i = 0; i < weiSize; ++i) {
        *(wei + i) = (T)u(e);
    }
    for (SizeType i = 0; i < biaSize; ++i) {
        *(bia + i) = (T)u(e);
    }
    conv3dRef<T>(srcShape, weiShape, dstShape, strides, dilates, paddingL, paddingR, src, wei, dstRef, bia);
    convFwdLayer1.Run(src, wei, dst, bia);
    float error = 0.0;
#pragma omp parallel for reduction(+ : error) num_threads(threadsUsed)
    for (SizeType i = 0; i < dstSize; ++i) {
        error += *(dst + i) - *(dstRef + i);
    }
    free(src);
    free(dst);
    free(dstRef);
    free(wei);
    free(bia);
    error = std::abs(error) / dstSize;
    std::cout << "run error: " << error << std::endl;
    return error < errBoundConv;
}

static void FWDAlgSupported1()
{
    const KuDNN::TensorInfo srcTensor = {{1, 1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCD};
    const KuDNN::TensorInfo weightsTensor = {{1, 1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCD};
    const KuDNN::TensorInfo dstTensor = {{1, 1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCD};
    const KuDNN::TensorInfo biasTensor = {{1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::A};
    KuDNN::Shape strides(1, 1);
    KuDNN::Shape dilates(0, 0);
    KuDNN::Shape paddingL(0, 0);
    KuDNN::Shape paddingR(0, 0);
    KuDNN::ConvolutionAlgorithm alg(KuDNN::ConvolutionAlgorithm::UNIMPLEMENTED);
    KuDNN::ConvolutionLayerFWD convFwdLayer(
        srcTensor, weightsTensor, dstTensor, biasTensor, strides, dilates, paddingL, paddingR, alg);
}

static void FWDAlgSupported2()
{
    const KuDNN::TensorInfo srcTensor = {{1, 1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCD};
    const KuDNN::TensorInfo weightsTensor = {{1, 1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCD};
    const KuDNN::TensorInfo dstTensor = {{1, 1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCD};
    const KuDNN::TensorInfo biasTensor = {{1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::A};
    KuDNN::Shape strides(1, 1);
    KuDNN::Shape dilates(0, 0);
    KuDNN::Shape paddingL(0, 0);
    KuDNN::Shape paddingR(0, 0);
    KuDNN::ConvolutionAlgorithm alg(KuDNN::ConvolutionAlgorithm::WINOGRAD);
    KuDNN::ConvolutionLayerFWD convFwdLayer(
        srcTensor, weightsTensor, dstTensor, biasTensor, strides, dilates, paddingL, paddingR, alg);
}

static void ForwardCopyConstructor1()
{
    const KuDNN::TensorInfo srcTensor = {{1, 1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCD};
    const KuDNN::TensorInfo weightsTensor = {{1, 1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCD};
    const KuDNN::TensorInfo dstTensor = {{1, 1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCD};
    const KuDNN::TensorInfo biasTensor = {{1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::A};
    KuDNN::Shape strides(1, 1);
    KuDNN::Shape dilates(0, 0);
    KuDNN::Shape paddingL(0, 0);
    KuDNN::Shape paddingR(0, 0);
    KuDNN::ConvolutionAlgorithm alg(KuDNN::ConvolutionAlgorithm::AUTO);
    KuDNN::ConvolutionLayerFWD convFwdLayer1(
        srcTensor, weightsTensor, dstTensor, biasTensor, strides, dilates, paddingL, paddingR, alg);
    KuDNN::ConvolutionLayerFWD convFwdLayer2(convFwdLayer1);
    for (size_t i = 0; i < g_Tensor_data.size() / 4; i += 4) {
        KuDNN::ConvolutionLayerFWD convFwdLayer1(g_Tensor_data[i],
            g_Tensor_data[i + 1],
            g_Tensor_data[i + 2],
            g_Tensor_data[i + 3],
            strides,
            dilates,
            paddingL,
            paddingR,
            alg);
        KuDNN::ConvolutionLayerFWD convFwdLayer2(convFwdLayer1);
    }
}

static void ForwardCopyConstructor2()
{
    const KuDNN::TensorInfo srcTensor = {{1, 1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCD};
    const KuDNN::TensorInfo weightsTensor = {{1, 1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCD};
    const KuDNN::TensorInfo dstTensor = {{1, 1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCD};
    const KuDNN::TensorInfo biasTensor = {{1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::A};
    KuDNN::Shape strides(1, 1);
    KuDNN::Shape dilates(0, 0);
    KuDNN::Shape paddingL(0, 0);
    KuDNN::Shape paddingR(0, 0);
    KuDNN::ConvolutionAlgorithm alg(KuDNN::ConvolutionAlgorithm::AUTO);
    KuDNN::ConvolutionLayerFWD convFwdLayer1(
        srcTensor, weightsTensor, dstTensor, biasTensor, strides, dilates, paddingL, paddingR, alg);
    KuDNN::ConvolutionLayerFWD convFwdLayer2(
        srcTensor, weightsTensor, dstTensor, biasTensor, strides, dilates, paddingL, paddingR, alg);
    convFwdLayer2 = convFwdLayer1;
    for (size_t i = 0; i < g_Tensor_data.size() / 4; i += 4) {
        KuDNN::ConvolutionLayerFWD convFwdLayer1(g_Tensor_data[i],
            g_Tensor_data[i + 1],
            g_Tensor_data[i + 2],
            g_Tensor_data[i + 3],
            strides,
            dilates,
            paddingL,
            paddingR,
            alg);
        KuDNN::ConvolutionLayerFWD convFwdLayer2(g_Tensor_data[i],
            g_Tensor_data[i + 1],
            g_Tensor_data[i + 2],
            g_Tensor_data[i + 3],
            strides,
            dilates,
            paddingL,
            paddingR,
            alg);
        convFwdLayer2 = convFwdLayer1;
    }
}

static void ForwardMoveConstructor1()
{
    const KuDNN::TensorInfo srcTensor = {{1, 1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCD};
    const KuDNN::TensorInfo weightsTensor = {{1, 1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCD};
    const KuDNN::TensorInfo dstTensor = {{1, 1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCD};
    const KuDNN::TensorInfo biasTensor = {{1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::A};
    KuDNN::Shape strides(1, 1);
    KuDNN::Shape dilates(0, 0);
    KuDNN::Shape paddingL(0, 0);
    KuDNN::Shape paddingR(0, 0);
    KuDNN::ConvolutionAlgorithm alg(KuDNN::ConvolutionAlgorithm::AUTO);
    KuDNN::ConvolutionLayerFWD convFwdLayer1(
        srcTensor, weightsTensor, dstTensor, biasTensor, strides, dilates, paddingL, paddingR, alg);
    KuDNN::ConvolutionLayerFWD convFwdLayer2(std::move(convFwdLayer1));
    for (size_t i = 0; i < g_Tensor_data.size() / 4; i += 4) {
        KuDNN::ConvolutionLayerFWD convFwdLayer1(g_Tensor_data[i],
            g_Tensor_data[i + 1],
            g_Tensor_data[i + 2],
            g_Tensor_data[i + 3],
            strides,
            dilates,
            paddingL,
            paddingR,
            alg);
        KuDNN::ConvolutionLayerFWD convFwdLayer2(std::move(convFwdLayer1));
    }
}

static void ForwardMoveConstructor2()
{
    const KuDNN::TensorInfo srcTensor = {{1, 1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCD};
    const KuDNN::TensorInfo weightsTensor = {{1, 1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCD};
    const KuDNN::TensorInfo dstTensor = {{1, 1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCD};
    const KuDNN::TensorInfo biasTensor = {{1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::A};
    KuDNN::Shape strides(1, 1);
    KuDNN::Shape dilates(0, 0);
    KuDNN::Shape paddingL(0, 0);
    KuDNN::Shape paddingR(0, 0);
    KuDNN::ConvolutionAlgorithm alg(KuDNN::ConvolutionAlgorithm::AUTO);
    KuDNN::ConvolutionLayerFWD convFwdLayer1(
        srcTensor, weightsTensor, dstTensor, biasTensor, strides, dilates, paddingL, paddingR, alg);
    KuDNN::ConvolutionLayerFWD convFwdLayer2(
        srcTensor, weightsTensor, dstTensor, biasTensor, strides, dilates, paddingL, paddingR, alg);
    convFwdLayer2 = convFwdLayer1;
    for (size_t i = 0; i < g_Tensor_data.size() / 4; i += 4) {
        KuDNN::ConvolutionLayerFWD convFwdLayer1(g_Tensor_data[i],
            g_Tensor_data[i + 1],
            g_Tensor_data[i + 2],
            g_Tensor_data[i + 3],
            strides,
            dilates,
            paddingL,
            paddingR,
            alg);
        KuDNN::ConvolutionLayerFWD convFwdLayer2(g_Tensor_data[i],
            g_Tensor_data[i + 1],
            g_Tensor_data[i + 2],
            g_Tensor_data[i + 3],
            strides,
            dilates,
            paddingL,
            paddingR,
            alg);
        convFwdLayer2 = convFwdLayer1;
    }
}


// backward data
static void BackwardDataCopyConstructor1()
{
    const KuDNN::TensorInfo diffSrcTensor = {{1, 1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCD};
    const KuDNN::TensorInfo weightsTensor = {{1, 1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCD};
    const KuDNN::TensorInfo diffDstTensor = {{1, 1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCD};
    KuDNN::Shape strides(1, 1);
    KuDNN::Shape dilates(0, 0);
    KuDNN::Shape paddingL(0, 0);
    KuDNN::Shape paddingR(0, 0);
    KuDNN::ConvolutionAlgorithm alg(KuDNN::ConvolutionAlgorithm::AUTO);
    KuDNN::ConvolutionLayerBWDData convBwdLayer1(
        diffDstTensor, weightsTensor, diffSrcTensor, strides, dilates, paddingL, paddingR, alg);
    KuDNN::ConvolutionLayerBWDData convBwdLayer2(convBwdLayer1);
    for (size_t i = 0; i < g_Tensor_data.size() / 3; i += 3) {
        KuDNN::ConvolutionLayerBWDData convBwdLayer1(
            g_Tensor_data[i], g_Tensor_data[i + 1], g_Tensor_data[i + 2], strides, dilates, paddingL, paddingR, alg);
        KuDNN::ConvolutionLayerBWDData convBwdLayer2(convBwdLayer1);
    }
}

static void BackwardDataCopyConstructor2()
{
    const KuDNN::TensorInfo diffSrcTensor = {{1, 1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCD};
    const KuDNN::TensorInfo weightsTensor = {{1, 1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCD};
    const KuDNN::TensorInfo diffDstTensor = {{1, 1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCD};
    KuDNN::Shape strides(1, 1);
    KuDNN::Shape dilates(0, 0);
    KuDNN::Shape paddingL(0, 0);
    KuDNN::Shape paddingR(0, 0);
    KuDNN::ConvolutionAlgorithm alg(KuDNN::ConvolutionAlgorithm::AUTO);
    KuDNN::ConvolutionLayerBWDData convBwdLayer1(
        diffDstTensor, weightsTensor, diffSrcTensor, strides, dilates, paddingL, paddingR, alg);
    KuDNN::ConvolutionLayerBWDData convBwdLayer2(
        diffDstTensor, weightsTensor, diffSrcTensor, strides, dilates, paddingL, paddingR, alg);
    convBwdLayer2 = convBwdLayer1;
    for (size_t i = 0; i < g_Tensor_data.size() / 3; i += 3) {
        KuDNN::ConvolutionLayerBWDData convBwdLayer1(
            g_Tensor_data[i], g_Tensor_data[i + 1], g_Tensor_data[i + 2], strides, dilates, paddingL, paddingR, alg);
        KuDNN::ConvolutionLayerBWDData convBwdLayer2(
            g_Tensor_data[i], g_Tensor_data[i + 1], g_Tensor_data[i + 2], strides, dilates, paddingL, paddingR, alg);
        convBwdLayer2 = convBwdLayer1;
    }
}

static void BackwardDataMoveConstructor1()
{
    const KuDNN::TensorInfo diffSrcTensor = {{1, 1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCD};
    const KuDNN::TensorInfo weightsTensor = {{1, 1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCD};
    const KuDNN::TensorInfo diffDstTensor = {{1, 1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCD};
    KuDNN::Shape strides(1, 1);
    KuDNN::Shape dilates(0, 0);
    KuDNN::Shape paddingL(0, 0);
    KuDNN::Shape paddingR(0, 0);
    KuDNN::ConvolutionAlgorithm alg(KuDNN::ConvolutionAlgorithm::AUTO);
    KuDNN::ConvolutionLayerBWDData convBwdLayer1(
        diffDstTensor, weightsTensor, diffSrcTensor, strides, dilates, paddingL, paddingR, alg);
    KuDNN::ConvolutionLayerBWDData convBwdLayer2(std::move(convBwdLayer1));
    for (size_t i = 0; i < g_Tensor_data.size() / 3; i += 3) {
        KuDNN::ConvolutionLayerBWDData convBwdLayer1(
            g_Tensor_data[i], g_Tensor_data[i + 1], g_Tensor_data[i + 2], strides, dilates, paddingL, paddingR, alg);
        KuDNN::ConvolutionLayerBWDData convBwdLayer2(std::move(convBwdLayer1));
    }
}

static void BackwardDataMoveConstructor2()
{
    const KuDNN::TensorInfo diffSrcTensor = {{1, 1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCD};
    const KuDNN::TensorInfo weightsTensor = {{1, 1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCD};
    const KuDNN::TensorInfo diffDstTensor = {{1, 1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCD};
    KuDNN::Shape strides(1, 1);
    KuDNN::Shape dilates(0, 0);
    KuDNN::Shape paddingL(0, 0);
    KuDNN::Shape paddingR(0, 0);
    KuDNN::ConvolutionAlgorithm alg(KuDNN::ConvolutionAlgorithm::AUTO);
    KuDNN::ConvolutionLayerBWDData convBwdLayer1(
        diffDstTensor, weightsTensor, diffSrcTensor, strides, dilates, paddingL, paddingR, alg);
    KuDNN::ConvolutionLayerBWDData convBwdLayer2(
        diffDstTensor, weightsTensor, diffSrcTensor, strides, dilates, paddingL, paddingR, alg);
    convBwdLayer2 = std::move(convBwdLayer1);
    for (size_t i = 0; i < g_Tensor_data.size() / 3; i += 3) {
        KuDNN::ConvolutionLayerBWDData convBwdLayer1(
            g_Tensor_data[i], g_Tensor_data[i + 1], g_Tensor_data[i + 2], strides, dilates, paddingL, paddingR, alg);
        KuDNN::ConvolutionLayerBWDData convBwdLayer2(
            g_Tensor_data[i], g_Tensor_data[i + 1], g_Tensor_data[i + 2], strides, dilates, paddingL, paddingR, alg);
        convBwdLayer2 = std::move(convBwdLayer1);
    }
}

// backward weights
static void BackwardWeightsCopyConstructor1()
{
    const KuDNN::TensorInfo srcTensor = {{1, 1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCD};
    const KuDNN::TensorInfo diffWeightsTensor = {{1, 1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCD};
    const KuDNN::TensorInfo diffDstTensor = {{1, 1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCD};
    const KuDNN::TensorInfo diffBiasTensor = {{1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::A};
    KuDNN::Shape strides(1, 1);
    KuDNN::Shape dilates(0, 0);
    KuDNN::Shape paddingL(0, 0);
    KuDNN::Shape paddingR(0, 0);
    KuDNN::ConvolutionAlgorithm alg(KuDNN::ConvolutionAlgorithm::AUTO);
    KuDNN::ConvolutionLayerBWDWeights convBwdWeightsLayer1(
        diffDstTensor, srcTensor, diffWeightsTensor, diffBiasTensor, strides, dilates, paddingL, paddingR, alg);
    KuDNN::ConvolutionLayerBWDWeights convBwdWeightsLayer2(convBwdWeightsLayer1);
    for (size_t i = 0; i < g_Tensor_data.size() / 4; i += 4) {
        KuDNN::ConvolutionLayerBWDWeights convBwdWeightsLayer1(g_Tensor_data[i],
            g_Tensor_data[i + 1],
            g_Tensor_data[i + 2],
            g_Tensor_data[i + 3],
            strides,
            dilates,
            paddingL,
            paddingR,
            alg);
        KuDNN::ConvolutionLayerBWDWeights convBwdWeightsLayer2(convBwdWeightsLayer1);
    }
}

static void BackwardWeightsCopyConstructor2()
{
    const KuDNN::TensorInfo srcTensor = {{1, 1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCD};
    const KuDNN::TensorInfo diffWeightsTensor = {{1, 1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCD};
    const KuDNN::TensorInfo diffDstTensor = {{1, 1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCD};
    const KuDNN::TensorInfo diffBiasTensor = {{1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::A};
    KuDNN::Shape strides(1, 1);
    KuDNN::Shape dilates(0, 0);
    KuDNN::Shape paddingL(0, 0);
    KuDNN::Shape paddingR(0, 0);
    KuDNN::ConvolutionAlgorithm alg(KuDNN::ConvolutionAlgorithm::AUTO);
    KuDNN::ConvolutionLayerBWDWeights convBwdWeightsLayer1(
        diffDstTensor, srcTensor, diffWeightsTensor, diffBiasTensor, strides, dilates, paddingL, paddingR, alg);
    KuDNN::ConvolutionLayerBWDWeights convBwdWeightsLayer2(
        diffDstTensor, srcTensor, diffWeightsTensor, diffBiasTensor, strides, dilates, paddingL, paddingR, alg);
    convBwdWeightsLayer2 = convBwdWeightsLayer1;
    for (size_t i = 0; i < g_Tensor_data.size() / 4; i += 4) {
        KuDNN::ConvolutionLayerBWDWeights convBwdWeightsLayer1(g_Tensor_data[i],
            g_Tensor_data[i + 1],
            g_Tensor_data[i + 2],
            g_Tensor_data[i + 3],
            strides,
            dilates,
            paddingL,
            paddingR,
            alg);
        KuDNN::ConvolutionLayerBWDWeights convBwdWeightsLayer2(g_Tensor_data[i],
            g_Tensor_data[i + 1],
            g_Tensor_data[i + 2],
            g_Tensor_data[i + 3],
            strides,
            dilates,
            paddingL,
            paddingR,
            alg);
        convBwdWeightsLayer2 = convBwdWeightsLayer1;
    }
}

static void BackwardWeightsMoveConstructor1()
{
    const KuDNN::TensorInfo srcTensor = {{1, 1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCD};
    const KuDNN::TensorInfo diffWeightsTensor = {{1, 1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCD};
    const KuDNN::TensorInfo diffDstTensor = {{1, 1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCD};
    const KuDNN::TensorInfo diffBiasTensor = {{1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::A};
    KuDNN::Shape strides(1, 1);
    KuDNN::Shape dilates(0, 0);
    KuDNN::Shape paddingL(0, 0);
    KuDNN::Shape paddingR(0, 0);
    KuDNN::ConvolutionAlgorithm alg(KuDNN::ConvolutionAlgorithm::AUTO);
    KuDNN::ConvolutionLayerBWDWeights convBwdWeightsLayer1(
        diffDstTensor, srcTensor, diffWeightsTensor, diffBiasTensor, strides, dilates, paddingL, paddingR, alg);
    KuDNN::ConvolutionLayerBWDWeights convBwdWeightsLayer2(std::move(convBwdWeightsLayer1));
    for (size_t i = 0; i < g_Tensor_data.size() / 4; i += 4) {
        KuDNN::ConvolutionLayerBWDWeights convBwdWeightsLayer1(g_Tensor_data[i],
            g_Tensor_data[i + 1],
            g_Tensor_data[i + 2],
            g_Tensor_data[i + 3],
            strides,
            dilates,
            paddingL,
            paddingR,
            alg);
        KuDNN::ConvolutionLayerBWDWeights convBwdWeightsLayer2(std::move(convBwdWeightsLayer1));
    }
}

static void BackwardWeightsMoveConstructor2()
{
    const KuDNN::TensorInfo srcTensor = {{1, 1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCD};
    const KuDNN::TensorInfo diffWeightsTensor = {{1, 1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCD};
    const KuDNN::TensorInfo diffDstTensor = {{1, 1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCD};
    const KuDNN::TensorInfo diffBiasTensor = {{1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::A};
    KuDNN::Shape strides(1, 1);
    KuDNN::Shape dilates(0, 0);
    KuDNN::Shape paddingL(0, 0);
    KuDNN::Shape paddingR(0, 0);
    KuDNN::ConvolutionAlgorithm alg(KuDNN::ConvolutionAlgorithm::AUTO);
    KuDNN::ConvolutionLayerBWDWeights convBwdWeightsLayer1(
        diffDstTensor, srcTensor, diffWeightsTensor, diffBiasTensor, strides, dilates, paddingL, paddingR, alg);
    KuDNN::ConvolutionLayerBWDWeights convBwdWeightsLayer2(
        diffDstTensor, srcTensor, diffWeightsTensor, diffBiasTensor, strides, dilates, paddingL, paddingR, alg);
    convBwdWeightsLayer2 = std::move(convBwdWeightsLayer1);
    for (size_t i = 0; i < g_Tensor_data.size() / 4; i += 4) {
        KuDNN::ConvolutionLayerBWDWeights convBwdWeightsLayer1(g_Tensor_data[i],
            g_Tensor_data[i + 1],
            g_Tensor_data[i + 2],
            g_Tensor_data[i + 3],
            strides,
            dilates,
            paddingL,
            paddingR,
            alg);
        KuDNN::ConvolutionLayerBWDWeights convBwdWeightsLayer2(g_Tensor_data[i],
            g_Tensor_data[i + 1],
            g_Tensor_data[i + 2],
            g_Tensor_data[i + 3],
            strides,
            dilates,
            paddingL,
            paddingR,
            alg);
        convBwdWeightsLayer2 = std::move(convBwdWeightsLayer1);
    }
}

static void ForwardJITCopyConstructor1()
{
    const KuDNN::TensorInfo srcTensor = {{1, 1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCD};
    const KuDNN::TensorInfo weightsTensor = {{1, 1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCD};
    const KuDNN::TensorInfo dstTensor = {{1, 1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCD};
    const KuDNN::TensorInfo biasTensor = {{1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::A};
    KuDNN::Shape strides(1, 1);
    KuDNN::Shape dilates(0, 0);
    KuDNN::Shape paddingL(0, 0);
    KuDNN::Shape paddingR(0, 0);

    KuDNN::ConvolutionJITFWD convFwdLayer1(0x1f, srcTensor, weightsTensor, dstTensor, biasTensor, strides, paddingL, dilates,
                                        KuDNN::Propagation::FORWARD_TRAINING, 1 ,false);
    KuDNN::ConvolutionJITFWD convFwdLayer2(convFwdLayer1);
}

static void ForwardJITCopyConstructor2()
{
    const KuDNN::TensorInfo srcTensor = {{16, 16, 16, 16}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCD};
    const KuDNN::TensorInfo weightsTensor = {{16, 16, 16, 16, 16}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCDE};
    const KuDNN::TensorInfo dstTensor = {{16, 16, 16, 16}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCD};
    const KuDNN::TensorInfo biasTensor = {{16}, KuDNN::Element::TypeT::F32, KuDNN::Layout::A};
    KuDNN::Shape strides(0, 0);
    KuDNN::Shape dilates(0, 0);
    KuDNN::Shape paddingL(0, 0);
    KuDNN::Shape paddingR(0, 0);

    KuDNN::ConvolutionJITFWD convFwdLayer1(0x1f, srcTensor, weightsTensor, dstTensor, biasTensor, strides, paddingL, dilates,
                                        KuDNN::Propagation::FORWARD_TRAINING, 1 ,false);
    KuDNN::ConvolutionJITFWD convFwdLayer2 = convFwdLayer1;
}

static void ForwardJITMoveConstructor1()
{
    const KuDNN::TensorInfo srcTensor = {{1, 1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCD};
    const KuDNN::TensorInfo weightsTensor = {{1, 1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCD};
    const KuDNN::TensorInfo dstTensor = {{1, 1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCD};
    const KuDNN::TensorInfo biasTensor = {{1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::A};
    KuDNN::Shape strides(1, 1);
    KuDNN::Shape dilates(0, 0);
    KuDNN::Shape paddingL(0, 0);
    KuDNN::Shape paddingR(0, 0);

    KuDNN::ConvolutionJITFWD convFwdLayer1(0x1f, srcTensor, weightsTensor, dstTensor, biasTensor, strides, paddingL, dilates, 
                                        KuDNN::Propagation::FORWARD_INFERENCE, 1 ,false);
    KuDNN::ConvolutionJITFWD convFwdLayer2(std::move(convFwdLayer1));
}

static void ForwardJITMoveConstructor2()
{
    const KuDNN::TensorInfo srcTensor = {{1, 1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCD};
    const KuDNN::TensorInfo weightsTensor = {{1, 1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCD};
    const KuDNN::TensorInfo dstTensor = {{1, 1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCD};
    const KuDNN::TensorInfo biasTensor = {{1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::A};
    KuDNN::Shape strides(1, 1);
    KuDNN::Shape dilates(0, 0);
    KuDNN::Shape paddingL(0, 0);
    KuDNN::Shape paddingR(0, 0);

    KuDNN::ConvolutionJITFWD convFwdLayer1(0x1f, srcTensor, weightsTensor, dstTensor, biasTensor, strides, paddingL, dilates,
                                        KuDNN::Propagation::FORWARD_INFERENCE, 2, true);
    KuDNN::ConvolutionJITFWD convFwdLayer2(0x1f, srcTensor, weightsTensor, dstTensor, biasTensor, strides, paddingL, dilates,
                                        KuDNN::Propagation::FORWARD_INFERENCE, 2, false);
    convFwdLayer2 = std::move(convFwdLayer1);
}

static void InputParamExcept_Conv2dF16()
{
    SizeType N = -2, IC = 24, IH = 360, IW = 640;
    SizeType OC = 12, OH = 0, OW = 0;
    SizeType KH = 1, KW = 1;
    Shape strides(1, 1), dilates(0, 0), paddingL(0, 0), paddingR(0, 0);
    OH = (IH + paddingL[0] + paddingR[0] - 1 - (KH - 1) * (dilates[0] + 1)) / strides[0] + 1;
    OW = (IW + paddingL[1] + paddingR[1] - 1 - (KW - 1) * (dilates[1] + 1)) / strides[1] + 1;
    Shape srcShape(N, IC, IH, IW), weiShape(OC, IC, KH, KW), dstShape(N, OC, OH, OW);
    KuDNN::ConvolutionAlgorithm alg(KuDNN::ConvolutionAlgorithm::DIRECT);
    Conv2dFWDFunc1<__fp16>(srcShape, weiShape, dstShape, strides, dilates, paddingL, paddingR, alg);
}

static void InputParamExcept_Conv2dF32()
{
    SizeType N = 4, IC = -4, IH = 128, IW = 100;
    SizeType OC = 5, OH = 0, OW = 0;
    SizeType KH = 3, KW = 3;
    Shape strides(1, 1), dilates(1, 1), paddingL(1, 1), paddingR(1, 1);
    OH = (IH + paddingL[0] + paddingR[0] - 1 - (KH - 1) * (dilates[0] + 1)) / strides[0] + 1;
    OW = (IW + paddingL[1] + paddingR[1] - 1 - (KW - 1) * (dilates[1] + 1)) / strides[1] + 1;
    Shape srcShape(N, IC, IH, IW), weiShape(OC, IC, KH, KW), dstShape(N, OC, OH, OW);
    KuDNN::ConvolutionAlgorithm alg(KuDNN::ConvolutionAlgorithm::AUTO);
    Conv2dFWDFunc1<float>(srcShape, weiShape, dstShape, strides, dilates, paddingL, paddingR, alg);
}

static void InputParamExcept_Conv3dF16()
{
    SizeType N = 1, IC = 5, ID = -20, IH = 90, IW = 160;
    SizeType OC = 10, OD = 0, OH = 0, OW = 0;
    SizeType KD = 1, KH = 1, KW = 1;
    Shape strides(1, 1, 1), dilates(0, 0, 0), paddingL(0, 0, 0), paddingR(0, 0, 0);
    OD = (ID + paddingL[0] + paddingR[0] - 1 - (KD - 1) * (dilates[0] + 1)) / strides[0] + 1;
    OH = (IH + paddingL[1] + paddingR[1] - 1 - (KH - 1) * (dilates[1] + 1)) / strides[1] + 1;
    OW = (IW + paddingL[2] + paddingR[2] - 1 - (KW - 1) * (dilates[2] + 1)) / strides[2] + 1;
    Shape srcShape(N, IC, ID, IH, IW), weiShape(OC, IC, KD, KH, KW), dstShape(N, OC, OD, OH, OW);
    KuDNN::ConvolutionAlgorithm alg(KuDNN::ConvolutionAlgorithm::AUTO);
    Conv3dFWDFunc1<__fp16>(srcShape, weiShape, dstShape, strides, dilates, paddingL, paddingR, alg);
}

static void InputParamExcept_Conv3dF32()
{
    SizeType N = 1, IC = 4, ID = 7, IH = 92, IW = -162;
    SizeType OC = 16, OD = 0, OH = 0, OW = 0;
    SizeType KD = 3, KH = 3, KW = 3;
    Shape strides(1, 1, 1), dilates(1, 1, 1), paddingL(1, 1, 1), paddingR(1, 1, 1);
    OD = (ID + paddingL[0] + paddingR[0] - 1 - (KD - 1) * (dilates[0] + 1)) / strides[0] + 1;
    OH = (IH + paddingL[1] + paddingR[1] - 1 - (KH - 1) * (dilates[1] + 1)) / strides[1] + 1;
    OW = (IW + paddingL[2] + paddingR[2] - 1 - (KW - 1) * (dilates[2] + 1)) / strides[2] + 1;
    Shape srcShape(N, IC, ID, IH, IW), weiShape(OC, IC, KD, KH, KW), dstShape(N, OC, OD, OH, OW);
    KuDNN::ConvolutionAlgorithm alg(KuDNN::ConvolutionAlgorithm::AUTO);
    Conv3dFWDFunc1<float>(srcShape, weiShape, dstShape, strides, dilates, paddingL, paddingR, alg);
}

// test cases
TEST(ConvGTests, Conv2dFWDFunc1F16_1)
{
    SizeType N = 2, IC = 24, IH = 360, IW = 640;
    SizeType OC = 12, OH = 0, OW = 0;
    SizeType KH = 1, KW = 1;
    Shape strides(1, 1), dilates(0, 0), paddingL(0, 0), paddingR(0, 0);
    OH = (IH + paddingL[0] + paddingR[0] - 1 - (KH - 1) * (dilates[0] + 1)) / strides[0] + 1;
    OW = (IW + paddingL[1] + paddingR[1] - 1 - (KW - 1) * (dilates[1] + 1)) / strides[1] + 1;
    Shape srcShape(N, IC, IH, IW), weiShape(OC, IC, KH, KW), dstShape(N, OC, OH, OW);
    KuDNN::ConvolutionAlgorithm alg(KuDNN::ConvolutionAlgorithm::DIRECT);
    bool res = Conv2dFWDFunc1<__fp16>(srcShape, weiShape, dstShape, strides, dilates, paddingL, paddingR, alg);
    GTEST_EXPECT_TRUE(res);
}

TEST(ConvGTests, Conv2dFWDFunc1F16_2)
{
    SizeType N = 2, IC = 3, IH = 720, IW = 1280;
    SizeType OC = 3, OH = 0, OW = 0;
    SizeType KH = 1, KW = 1;
    Shape strides(1, 1), dilates(0, 0), paddingL(1, 1), paddingR(1, 1);
    OH = (IH + paddingL[0] + paddingR[0] - 1 - (KH - 1) * (dilates[0] + 1)) / strides[0] + 1;
    OW = (IW + paddingL[1] + paddingR[1] - 1 - (KW - 1) * (dilates[1] + 1)) / strides[1] + 1;
    Shape srcShape(N, IC, IH, IW), weiShape(OC, IC, KH, KW), dstShape(N, OC, OH, OW);
    KuDNN::ConvolutionAlgorithm alg(KuDNN::ConvolutionAlgorithm::DIRECT);
    bool res = Conv2dFWDFunc1<__fp16>(srcShape, weiShape, dstShape, strides, dilates, paddingL, paddingR, alg);
    GTEST_EXPECT_TRUE(res);
}

TEST(ConvGTests, Conv2dFWDFunc1F16_3)
{
    SizeType N = 2, IC = 16, IH = 360, IW = 640;
    SizeType OC = 8, OH = 0, OW = 0;
    SizeType KH = 3, KW = 3;
    Shape strides(1, 1), dilates(1, 1), paddingL(2, 2), paddingR(2, 2);
    OH = (IH + paddingL[0] + paddingR[0] - 1 - (KH - 1) * (dilates[0] + 1)) / strides[0] + 1;
    OW = (IW + paddingL[1] + paddingR[1] - 1 - (KW - 1) * (dilates[1] + 1)) / strides[1] + 1;
    Shape srcShape(N, IC, IH, IW), weiShape(OC, IC, KH, KW), dstShape(N, OC, OH, OW);
    KuDNN::ConvolutionAlgorithm alg(KuDNN::ConvolutionAlgorithm::AUTO);
    bool res = Conv2dFWDFunc1<__fp16>(srcShape, weiShape, dstShape, strides, dilates, paddingL, paddingR, alg);
    GTEST_EXPECT_TRUE(res);
}

TEST(ConvGTests, Conv2dFWDFunc1F16_4)
{
    SizeType N = 2, IC = 9, IH = 128, IW = 128;
    SizeType OC = 9, OH = 0, OW = 0;
    SizeType KH = 3, KW = 3;
    Shape strides(2, 2), dilates(1, 1), paddingL(1, 1), paddingR(2, 2);
    OH = (IH + paddingL[0] + paddingR[0] - 1 - (KH - 1) * (dilates[0] + 1)) / strides[0] + 1;
    OW = (IW + paddingL[1] + paddingR[1] - 1 - (KW - 1) * (dilates[1] + 1)) / strides[1] + 1;
    Shape srcShape(N, IC, IH, IW), weiShape(OC, IC, KH, KW), dstShape(N, OC, OH, OW);
    KuDNN::ConvolutionAlgorithm alg(KuDNN::ConvolutionAlgorithm::AUTO);
    bool res = Conv2dFWDFunc1<__fp16>(srcShape, weiShape, dstShape, strides, dilates, paddingL, paddingR, alg);
    GTEST_EXPECT_TRUE(res);
}

TEST(ConvGTests, Conv2dFWDFunc1F32)
{
    SizeType N = 4, IC = 4, IH = 128, IW = 100;
    SizeType OC = 5, OH = 0, OW = 0;
    SizeType KH = 3, KW = 3;
    Shape strides(1, 1), dilates(1, 1), paddingL(1, 1), paddingR(1, 1);
    OH = (IH + paddingL[0] + paddingR[0] - 1 - (KH - 1) * (dilates[0] + 1)) / strides[0] + 1;
    OW = (IW + paddingL[1] + paddingR[1] - 1 - (KW - 1) * (dilates[1] + 1)) / strides[1] + 1;
    Shape srcShape(N, IC, IH, IW), weiShape(OC, IC, KH, KW), dstShape(N, OC, OH, OW);
    KuDNN::ConvolutionAlgorithm alg(KuDNN::ConvolutionAlgorithm::AUTO);
    bool res = Conv2dFWDFunc1<float>(srcShape, weiShape, dstShape, strides, dilates, paddingL, paddingR, alg);
    GTEST_EXPECT_TRUE(res);
}

TEST(ConvGTests, Conv2dBWDDataFunc1F32)
{
    SizeType N = 4, IC = 4, IH = 128, IW = 100;
    SizeType OC = 5, OH = 0, OW = 0;
    SizeType KH = 3, KW = 3;
    Shape strides(1, 1), dilates(1, 1), paddingL(1, 1), paddingR(1, 1);
    OH = (IH + paddingL[0] + paddingR[0] - 1 - (KH - 1) * (dilates[0] + 1)) / strides[0] + 1;
    OW = (IW + paddingL[1] + paddingR[1] - 1 - (KW - 1) * (dilates[1] + 1)) / strides[1] + 1;
    Shape srcShape(N, IC, IH, IW), weiShape(OC, IC, KH, KW), dstShape(N, OC, OH, OW);
    KuDNN::ConvolutionAlgorithm alg(KuDNN::ConvolutionAlgorithm::AUTO);
    bool res = Conv2dBWDDataFunc1<float>(dstShape, weiShape, srcShape, strides, dilates, paddingL, paddingR, alg);
    GTEST_EXPECT_TRUE(res);
}

TEST(ConvGTests, Conv2dBWDWeightsFunc1F32)
{
    SizeType N = 4, IC = 4, IH = 128, IW = 100;
    SizeType OC = 5, OH = 0, OW = 0;
    SizeType KH = 3, KW = 3;
    Shape strides(1, 1), dilates(1, 1), paddingL(1, 1), paddingR(1, 1);
    OH = (IH + paddingL[0] + paddingR[0] - 1 - (KH - 1) * (dilates[0] + 1)) / strides[0] + 1;
    OW = (IW + paddingL[1] + paddingR[1] - 1 - (KW - 1) * (dilates[1] + 1)) / strides[1] + 1;
    Shape srcShape(N, IC, IH, IW), weiShape(OC, IC, KH, KW), dstShape(N, OC, OH, OW);
    Shape diffBiasShape(OC);
    KuDNN::ConvolutionAlgorithm alg(KuDNN::ConvolutionAlgorithm::AUTO);
    bool res = Conv2dBWDWeightFunc1<float>(dstShape, srcShape, weiShape, diffBiasShape, strides, dilates, paddingL, paddingR, alg);
    GTEST_EXPECT_TRUE(res);
}
TEST(ConvGTests, Conv3dFWDFunc1F16_1)
{
    SizeType N = 1, IC = 5, ID = 20, IH = 90, IW = 160;
    SizeType OC = 10, OD = 0, OH = 0, OW = 0;
    SizeType KD = 1, KH = 1, KW = 1;
    Shape strides(1, 1, 1), dilates(0, 0, 0), paddingL(0, 0, 0), paddingR(0, 0, 0);
    OD = (ID + paddingL[0] + paddingR[0] - 1 - (KD - 1) * (dilates[0] + 1)) / strides[0] + 1;
    OH = (IH + paddingL[1] + paddingR[1] - 1 - (KH - 1) * (dilates[1] + 1)) / strides[1] + 1;
    OW = (IW + paddingL[2] + paddingR[2] - 1 - (KW - 1) * (dilates[2] + 1)) / strides[2] + 1;
    Shape srcShape(N, IC, ID, IH, IW), weiShape(OC, IC, KD, KH, KW), dstShape(N, OC, OD, OH, OW);
    KuDNN::ConvolutionAlgorithm alg(KuDNN::ConvolutionAlgorithm::AUTO);
    bool res = Conv3dFWDFunc1<__fp16>(srcShape, weiShape, dstShape, strides, dilates, paddingL, paddingR, alg);
    GTEST_EXPECT_TRUE(res);
}

TEST(ConvGTests, Conv3dFWDFunc1F16_2)
{
    SizeType N = 1, IC = 10, ID = 20, IH = 90, IW = 160;
    SizeType OC = 5, OD = 0, OH = 0, OW = 0;
    SizeType KD = 1, KH = 1, KW = 1;
    Shape strides(1, 1, 1), dilates(0, 0, 0), paddingL(1, 1, 1), paddingR(1, 1, 1);
    OD = (ID + paddingL[0] + paddingR[0] - 1 - (KD - 1) * (dilates[0] + 1)) / strides[0] + 1;
    OH = (IH + paddingL[1] + paddingR[1] - 1 - (KH - 1) * (dilates[1] + 1)) / strides[1] + 1;
    OW = (IW + paddingL[2] + paddingR[2] - 1 - (KW - 1) * (dilates[2] + 1)) / strides[2] + 1;
    Shape srcShape(N, IC, ID, IH, IW), weiShape(OC, IC, KD, KH, KW), dstShape(N, OC, OD, OH, OW);
    KuDNN::ConvolutionAlgorithm alg(KuDNN::ConvolutionAlgorithm::AUTO);
    bool res = Conv3dFWDFunc1<__fp16>(srcShape, weiShape, dstShape, strides, dilates, paddingL, paddingR, alg);
    GTEST_EXPECT_TRUE(res);
}

TEST(ConvGTests, Conv3dFWDFunc1F16_3)
{
    SizeType N = 1, IC = 16, ID = 22, IH = 92, IW = 162;
    SizeType OC = 16, OD = 0, OH = 0, OW = 0;
    SizeType KD = 3, KH = 3, KW = 3;
    Shape strides(1, 1, 1), dilates(0, 0, 0), paddingL(0, 0, 0), paddingR(0, 0, 0);
    OD = (ID + paddingL[0] + paddingR[0] - 1 - (KD - 1) * (dilates[0] + 1)) / strides[0] + 1;
    OH = (IH + paddingL[1] + paddingR[1] - 1 - (KH - 1) * (dilates[1] + 1)) / strides[1] + 1;
    OW = (IW + paddingL[2] + paddingR[2] - 1 - (KW - 1) * (dilates[2] + 1)) / strides[2] + 1;
    Shape srcShape(N, IC, ID, IH, IW), weiShape(OC, IC, KD, KH, KW), dstShape(N, OC, OD, OH, OW);
    KuDNN::ConvolutionAlgorithm alg(KuDNN::ConvolutionAlgorithm::AUTO);
    bool res = Conv3dFWDFunc1<__fp16>(srcShape, weiShape, dstShape, strides, dilates, paddingL, paddingR, alg);
    GTEST_EXPECT_TRUE(res);
}

TEST(ConvGTests, Conv3dFWDFunc1F16_4)
{
    SizeType N = 1, IC = 32, ID = 7, IH = 92, IW = 162;
    SizeType OC = 32, OD = 0, OH = 0, OW = 0;
    SizeType KD = 3, KH = 3, KW = 3;
    Shape strides(1, 1, 1), dilates(0, 0, 0), paddingL(1, 1, 1), paddingR(1, 1, 1);
    OD = (ID + paddingL[0] + paddingR[0] - 1 - (KD - 1) * (dilates[0] + 1)) / strides[0] + 1;
    OH = (IH + paddingL[1] + paddingR[1] - 1 - (KH - 1) * (dilates[1] + 1)) / strides[1] + 1;
    OW = (IW + paddingL[2] + paddingR[2] - 1 - (KW - 1) * (dilates[2] + 1)) / strides[2] + 1;
    Shape srcShape(N, IC, ID, IH, IW), weiShape(OC, IC, KD, KH, KW), dstShape(N, OC, OD, OH, OW);
    KuDNN::ConvolutionAlgorithm alg(KuDNN::ConvolutionAlgorithm::AUTO);
    bool res = Conv3dFWDFunc1<__fp16>(srcShape, weiShape, dstShape, strides, dilates, paddingL, paddingR, alg);
    GTEST_EXPECT_TRUE(res);
}

TEST(ConvGTests, Conv3dFWDFunc1F16_5)
{
    SizeType N = 1, IC = 32, ID = 7, IH = 92, IW = 162;
    SizeType OC = 32, OD = 0, OH = 0, OW = 0;
    SizeType KD = 3, KH = 3, KW = 3;
    Shape strides(1, 1, 1), dilates(1, 1, 1), paddingL(2, 2, 2), paddingR(1, 1, 1);
    OD = (ID + paddingL[0] + paddingR[0] - 1 - (KD - 1) * (dilates[0] + 1)) / strides[0] + 1;
    OH = (IH + paddingL[1] + paddingR[1] - 1 - (KH - 1) * (dilates[1] + 1)) / strides[1] + 1;
    OW = (IW + paddingL[2] + paddingR[2] - 1 - (KW - 1) * (dilates[2] + 1)) / strides[2] + 1;
    Shape srcShape(N, IC, ID, IH, IW), weiShape(OC, IC, KD, KH, KW), dstShape(N, OC, OD, OH, OW);
    KuDNN::ConvolutionAlgorithm alg(KuDNN::ConvolutionAlgorithm::AUTO);
    bool res = Conv3dFWDFunc1<__fp16>(srcShape, weiShape, dstShape, strides, dilates, paddingL, paddingR, alg);
    GTEST_EXPECT_TRUE(res);
}

TEST(ConvGTests, Conv3dFWDFunc1F32)
{
    SizeType N = 1, IC = 4, ID = 7, IH = 92, IW = 162;
    SizeType OC = 16, OD = 0, OH = 0, OW = 0;
    SizeType KD = 3, KH = 3, KW = 3;
    Shape strides(1, 1, 1), dilates(1, 1, 1), paddingL(1, 1, 1), paddingR(1, 1, 1);
    OD = (ID + paddingL[0] + paddingR[0] - 1 - (KD - 1) * (dilates[0] + 1)) / strides[0] + 1;
    OH = (IH + paddingL[1] + paddingR[1] - 1 - (KH - 1) * (dilates[1] + 1)) / strides[1] + 1;
    OW = (IW + paddingL[2] + paddingR[2] - 1 - (KW - 1) * (dilates[2] + 1)) / strides[2] + 1;
    Shape srcShape(N, IC, ID, IH, IW), weiShape(OC, IC, KD, KH, KW), dstShape(N, OC, OD, OH, OW);
    KuDNN::ConvolutionAlgorithm alg(KuDNN::ConvolutionAlgorithm::AUTO);
    bool res = Conv3dFWDFunc1<float>(srcShape, weiShape, dstShape, strides, dilates, paddingL, paddingR, alg);
    GTEST_EXPECT_TRUE(res);
}

TEST(ConvGTests, Conv3dBWDDataFunc1F32)
{
    SizeType N = 1, IC = 4, ID = 7, IH = 92, IW = 162;
    SizeType OC = 16, OD = 0, OH = 0, OW = 0;
    SizeType KD = 3, KH = 3, KW = 3;
    Shape strides(1, 1, 1), dilates(1, 1, 1), paddingL(1, 1, 1), paddingR(1, 1, 1);
    OD = (ID + paddingL[0] + paddingR[0] - 1 - (KD - 1) * (dilates[0] + 1)) / strides[0] + 1;
    OH = (IH + paddingL[1] + paddingR[1] - 1 - (KH - 1) * (dilates[1] + 1)) / strides[1] + 1;
    OW = (IW + paddingL[2] + paddingR[2] - 1 - (KW - 1) * (dilates[2] + 1)) / strides[2] + 1;
    Shape srcShape(N, IC, ID, IH, IW), weiShape(OC, IC, KD, KH, KW), dstShape(N, OC, OD, OH, OW);
    KuDNN::ConvolutionAlgorithm alg(KuDNN::ConvolutionAlgorithm::AUTO);
    bool res = Conv3dBWDDataFunc1<float>(dstShape, weiShape, srcShape, strides, dilates, paddingL, paddingR, alg);
    GTEST_EXPECT_TRUE(res);
}

TEST(ConvGTests, Conv3dBWDWeightsFunc1F32)
{
    SizeType N = 1, IC = 4, ID = 7, IH = 92, IW = 162;
    SizeType OC = 16, OD = 0, OH = 0, OW = 0;
    SizeType KD = 3, KH = 3, KW = 3;
    Shape strides(1, 1, 1), dilates(1, 1, 1), paddingL(1, 1, 1), paddingR(1, 1, 1);
    OD = (ID + paddingL[0] + paddingR[0] - 1 - (KD - 1) * (dilates[0] + 1)) / strides[0] + 1;
    OH = (IH + paddingL[1] + paddingR[1] - 1 - (KH - 1) * (dilates[1] + 1)) / strides[1] + 1;
    OW = (IW + paddingL[2] + paddingR[2] - 1 - (KW - 1) * (dilates[2] + 1)) / strides[2] + 1;
    Shape srcShape(N, IC, ID, IH, IW), weiShape(OC, IC, KD, KH, KW), dstShape(N, OC, OD, OH, OW);
    Shape diffBiasShape(OC);
    KuDNN::ConvolutionAlgorithm alg(KuDNN::ConvolutionAlgorithm::AUTO);
    bool res = Conv3dBWDWeightFunc1<float>(dstShape, srcShape, weiShape, diffBiasShape, strides, dilates, paddingL, paddingR, alg);
    GTEST_EXPECT_TRUE(res);
}

TEST(ConvGTests, InputParamExcept_Conv2dF16)
{
    EXPECT_THROW(InputParamExcept_Conv2dF16(), KuDNN::Service::LogicError);
}

TEST(ConvGTests, InputParamExcept_Conv2dF32)
{
    EXPECT_THROW(InputParamExcept_Conv2dF32(), KuDNN::Service::LogicError);
}

TEST(ConvGTests, InputParamExcept_Conv3dF16)
{
    EXPECT_THROW(InputParamExcept_Conv3dF16(), KuDNN::Service::LogicError);
}

TEST(ConvGTests, InputParamExcept_Conv3dF32)
{
    EXPECT_THROW(InputParamExcept_Conv3dF32(), KuDNN::Service::LogicError);
}

TEST(ConvGTests, FWDAlgSupported1)
{
    EXPECT_ANY_THROW(FWDAlgSupported1());
}

TEST(ConvGTests, FWDAlgSupported2)
{
    EXPECT_ANY_THROW(FWDAlgSupported2());
}

TEST(ConvGTests, FWDCopyConstructor1)
{
    EXPECT_NO_THROW(ForwardCopyConstructor1());
}

TEST(ConvGTests, FWDCopyConstructor2)
{
    EXPECT_NO_THROW(ForwardCopyConstructor2());
}

TEST(ConvGTests, FWDMoveConstructor1)
{
    EXPECT_NO_THROW(ForwardMoveConstructor1());
}

TEST(ConvGTests, FWDMoveConstructor2)
{
    EXPECT_NO_THROW(ForwardMoveConstructor2());
}

TEST(ConvGTests, BackwardDataCopyConstructor1)
{
    EXPECT_NO_THROW(BackwardDataCopyConstructor1());
}

TEST(ConvGTests, BackwardDataCopyConstructor2)
{
    EXPECT_NO_THROW(BackwardDataCopyConstructor2());
}

TEST(ConvGTests, BackwardDataMoveConstructor1)
{
    EXPECT_NO_THROW(BackwardDataMoveConstructor1());
}

TEST(ConvGTests, BackwardDataMoveConstructor2)
{
    EXPECT_NO_THROW(BackwardDataMoveConstructor2());
}

TEST(ConvGTests, BackwardWeightsCopyConstructor1)
{
    EXPECT_NO_THROW(BackwardWeightsCopyConstructor1());
}

TEST(ConvGTests, BackwardWeightsCopyConstructor2)
{
    EXPECT_NO_THROW(BackwardWeightsCopyConstructor2());
}

TEST(ConvGTests, BackwardWeightsMoveConstructor1)
{
    EXPECT_NO_THROW(BackwardWeightsMoveConstructor1());
}

TEST(ConvGTests, BackwardWeightsMoveConstructor2)
{
    EXPECT_NO_THROW(BackwardWeightsMoveConstructor2());
}

TEST(ConvGTests, ForwardJITCopyConstructor1)
{
    EXPECT_NO_THROW(ForwardJITCopyConstructor1());
}

TEST(ConvGTests, ForwardJITCopyConstructor2)
{
    ForwardJITCopyConstructor2();
}

TEST(ConvGTests, ForwardJITMoveConstructor1)
{
    EXPECT_NO_THROW(ForwardJITMoveConstructor1());
}

TEST(ConvGTests, ForwardJITMoveConstructor2)
{
    EXPECT_NO_THROW(ForwardJITMoveConstructor2());
}

