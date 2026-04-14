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
#ifdef __clang__
#include <arm_neon.h> // Needed to build with clang
#endif

using Shape = KuDNN::Shape;
using SizeType = KuDNN::SizeType;
float errorOneData = 9e-3;

enum class DataType {
    FP32,
    FP16,
    BF16
};

static void CvtBf16ToFp32(const __bf16 *src, float *dst, SizeType size){
    for (SizeType i=0; i < size; i++){
        dst[i] = vcvtah_f32_bf16(src[i]);
    }
}

static void CvtFp32ToBf16(const float *src, __bf16 *dst, SizeType size){
    for (SizeType i=0; i < size; i++){
        dst[i] = vcvth_bf16_f32(src[i]);
    }
}

static void CvtFp16ToFp32(const __fp16 *src, float *dst, SizeType size){
    for (SizeType i=0; i < size; i++){
        dst[i] = src[i];
    }
}

static void CvtFp32ToFp16(const float *src, __fp16 *dst, SizeType size){
    for (SizeType i=0; i < size; i++){
        dst[i] = src[i];
    }
}

static void linearSimpleFloat(SizeType batch, SizeType m, SizeType n, SizeType k, const float* src, float* weight, float* dst, float* bias, int transSrc=0, int transDst=0){
    for (SizeType bh = 0; bh < batch; ++bh) {
        for (SizeType i = 0; i < m; ++i) {
            for (SizeType j = 0; j < n; ++j) {
                float sum = 0;
                for (SizeType l = 0; l < k; ++l) {
                    if (transSrc == 0 && transDst == 0) {
                        sum += src[bh * m * k + i * k + l] * weight[bh * n * k + n * l + j];
                    } else if (transSrc == 1 && transDst == 0) {
                        sum += src[bh * m * k + i * k + l] * weight[bh * n * k + j * k + l];
                    } else if (transSrc == 0 && transDst == 1) {
                        sum += src[bh * m * k + l * m + i] * weight[bh * n * k + n * l + j];
                    } else {
                        sum += src[bh * m * k + l * m + i] * weight[bh * n * k + j * k + l];
                    }
                }
                if (bias){
                    dst[bh * m * n + i * n + j] = sum + bias[bh * m * n + i * n + j];
                } else{
                    dst[bh * m * n + i * n + j] = sum;
                }
            }
        }
    }
}

static void linearSimpleFp16(SizeType batch, SizeType m, SizeType n, SizeType k, const __fp16* src, __fp16* weight, __fp16* dst, __fp16* bias){
    float *src_f = (float *)malloc(batch * m * k * sizeof(float ));
    float *weight_f = (float *)malloc(batch * k * n * sizeof(float ));
    float *dst_f = (float *)malloc(batch * m * n * sizeof(float ));
    float *bias_f = (float *)malloc(batch * m * n * sizeof(float ));
    CvtFp16ToFp32(src, src_f, batch * m * k);
    CvtFp16ToFp32(weight, weight_f, batch * k * n);

    for (SizeType bh = 0; bh < batch; ++bh) {
        for (SizeType i = 0; i < m; ++i) {
            for (SizeType j = 0; j < n; ++j) {
                float sum = 0;
                for (SizeType l = 0; l < k; ++l) {
                    sum += src_f[bh * m * k + i * k + l] * weight_f[bh * n * k + n * l + j];
                }
                if (bias){
                    CvtFp16ToFp32(bias, bias_f, batch * m * n);
                    dst_f[bh * m * n + i * n + j] = sum + bias_f[bh * m * n + i * n + j];
                } else{
                    dst_f[bh * m * n + i * n + j] = sum;
                }
            }
        }
    }
    CvtFp32ToFp16(dst_f, dst, batch * m * n);
    if (src_f != nullptr){
        free(src_f);
    }
    if (weight_f != nullptr){
        free(weight_f);
    }
    if (dst_f != nullptr){
        free(dst_f);
    }
    if (bias_f != nullptr){
        free(bias_f);
    }
}


static void linearSimpleBf16(SizeType batch, SizeType m, SizeType n, SizeType k, const __bf16* src, __bf16* weight, __bf16* dst, __bf16* bias){
    float *src_f = (float *)malloc(batch * m * k * sizeof(float ));
    float *weight_f = (float *)malloc(batch * k * n * sizeof(float ));
    float *dst_f = (float *)malloc(batch * m * n * sizeof(float ));
    float *bias_f = (float *)malloc(batch * m * n * sizeof(float ));
    CvtBf16ToFp32(src, src_f, batch * m * k);
    CvtBf16ToFp32(weight, weight_f, batch * k * n);

    for (SizeType bh = 0; bh < batch; ++bh) {
        for (SizeType i = 0; i < m; ++i) {
            for (SizeType j = 0; j < n; ++j) {
                float sum = 0;
                for (SizeType l = 0; l < k; ++l) {
                    sum += src_f[bh * m * k + i * k + l] * weight_f[bh * n * k + n * l + j];
                }
                if (bias){
                    CvtBf16ToFp32(bias, bias_f, batch * m * n);
                    dst_f[bh * m * n + i * n + j] = sum + bias_f[bh * m * n + i * n + j];
                } else{
                    dst_f[bh * m * n + i * n + j] = sum;
                }
            }
        }
    }
    CvtFp32ToBf16(dst_f, dst, batch * m * n);
    if (src_f != nullptr){
        free(src_f);
    }
    if (weight_f != nullptr){
        free(weight_f);
    }
    if (dst_f != nullptr){
        free(dst_f);
    }
    if (bias_f != nullptr){
        free(bias_f);
    }
}


static void addFloat(const float *dst, float *res, float* linearRes, float gamma, SizeType size){
    for (SizeType i=0; i < size; i++){
        linearRes[i] = dst[i] + res[i] * gamma;
    }
}

static void addF16(const __fp16 *dst, __fp16 *res, __fp16 *linearRes, float gamma, SizeType size){

    float *dst_f = (float *)malloc(size * sizeof(float));
    float *res_f = (float *)malloc(size * sizeof(float));
    float *linearRes_f = (float *)malloc(size * sizeof(float));
    CvtFp16ToFp32(dst, dst_f, size);
    CvtFp16ToFp32(res, res_f, size);

    for (SizeType i=0; i < size; i++){
        linearRes_f[i] = dst_f[i] + res_f[i] * gamma;
    }

    CvtFp32ToFp16(linearRes_f, linearRes, size);
    if (dst_f != nullptr){
        free(dst_f);
    }
    if (res_f != nullptr){
        free(res_f);
    }
    if (linearRes_f != nullptr){
        free(linearRes_f);
    }
}

static void addBF16(const __bf16 *dst, __bf16 *res, __bf16 *linearRes, float gamma, SizeType size){
    float *dst_f = (float *)malloc(size * sizeof(float));
    float *res_f = (float *)malloc(size * sizeof(float));
    float *linearRes_f = (float *)malloc(size * sizeof(float));
    CvtBf16ToFp32(dst, dst_f, size);
    CvtBf16ToFp32(res, res_f, size);

    for (SizeType i=0; i < size; i++){
        linearRes_f[i] = dst_f[i] + res_f[i] * gamma;
    }
    CvtFp32ToBf16(linearRes_f, linearRes, size);
    if (dst_f != nullptr){
        free(dst_f);
    }
    if (res_f != nullptr){
        free(res_f);
    }
    if (linearRes_f != nullptr){
        free(linearRes_f);
    }
}


template <typename T>
static void DataInit(T **src, T **weight, T **dst, T **dstRef, T **bias, T **res, T **linearRes,T **linearResRef,
                     SizeType srcTotalSize, SizeType weightTotalSize, SizeType dstTotalSize,
                     SizeType resTotalSize, SizeType linearResTotalSize)
{
    *src = (T *)malloc(srcTotalSize * sizeof(T));
    *weight = (T *)malloc(weightTotalSize * sizeof(T));
    *dst = (T *)malloc(dstTotalSize * sizeof(T));
    *dstRef = (T *)malloc(dstTotalSize * sizeof(T));
    *bias = (T *)malloc(dstTotalSize * sizeof(T));
    *res = (T *)malloc(resTotalSize * sizeof(T));
    *linearRes = (T *)malloc(linearResTotalSize * sizeof(T));
    *linearResRef = (T *)malloc(linearResTotalSize * sizeof(T));

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

    for (SizeType i = 0; i < resTotalSize; ++i) {
        (*res)[i] = (T)u(e);
    }
}

static std::pair<float, float> compareEle(float ref, float diff, KuDNN::Element::TypeT type, SizeType K, SizeType L)
{
    float e = 0.0f;
    float eps = 0.0f;
    std::pair<float, float> pairEps = {0.0f, 0.0f};
    switch (type) {
        case KuDNN::Element::TypeT::F16:
            eps = 1e-3 * K * L;
            e = (std::fabs(ref) > eps) ? diff / ref : static_cast<float>(diff);
            pairEps = {e, eps};
            break;
        case KuDNN::Element::TypeT::BF16:
            eps = 1e-2 * K * L;
            e = (std::fabs(ref) > eps) ? diff / ref : static_cast<float>(diff);
            pairEps = {e, eps};
            break;
        case KuDNN::Element::TypeT::F32:
            eps = 1e-3;
            e = (std::fabs(ref) > eps) ? diff / ref : static_cast<float>(diff);
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

static bool Test_Template_Function_FP32(const KuDNN::TensorInfo srcTensor, const KuDNN::TensorInfo weightTensor,
                                        const KuDNN::TensorInfo dstTensor, const KuDNN::TensorInfo biasTensor,
                                        const KuDNN::TensorInfo resTensor, const KuDNN::TensorInfo linearResTensor,
                                        float alpha, float beta, float gamma, KuDNN::ResOpsFunction algKind, 
                                        int numThreads, int transSrc = 0, int transDst = 0){

    KuDNN::LinearResFWD LinearResFWD(srcTensor, weightTensor, dstTensor, biasTensor, resTensor, linearResTensor, alpha, beta, gamma, algKind, numThreads);

    float *src = nullptr, *weight = nullptr, *dst = nullptr, *dstRef = nullptr, *bias = nullptr, *res = nullptr, *linearRes = nullptr, *linearResRef = nullptr;
    DataInit<float>(&src, &weight, &dst, &dstRef, &bias, &res, &linearRes, &linearResRef, srcTensor.GetTotalTensorSize(), weightTensor.GetTotalTensorSize(),
                    dstTensor.GetTotalTensorSize(), resTensor.GetTotalTensorSize(), linearResTensor.GetTotalTensorSize());

    Shape srcShape = srcTensor.GetDims();
    Shape weightShape = weightTensor.GetDims();
    Shape resShape = resTensor.GetDims();
    SizeType srcDimNums = srcShape.GetNumDims();
    SizeType weightDimNums = weightShape.GetNumDims();
    SizeType resDimNums = resShape.GetNumDims();
    SizeType dst_row = srcShape[srcDimNums - 2];             // m
    SizeType dst_col = weightShape[weightDimNums - 1];       // n
    SizeType src_col = srcShape[srcDimNums - 1];             // k
    SizeType res_col = resShape[resDimNums - 1];             // l
    SizeType LinearBatch = dstTensor.GetTotalTensorSize() / (dst_row * dst_col);
    SizeType PostOpsBatch = linearResTensor.GetTotalTensorSize() / (dst_row * res_col);

    SizeType L = 1; // compareEle: K *L

    linearSimpleFloat(LinearBatch, dst_row, dst_col, src_col, src, weight, dstRef, bias, transSrc, transDst);
    if(algKind == KuDNN::ResOpsFunction::RES_MUL){
        L = dst_col;
        linearSimpleFloat(PostOpsBatch, dst_row, res_col, dst_col, dstRef, res, linearResRef, nullptr, transSrc, transDst);
    } else if(algKind == KuDNN::ResOpsFunction::RES_IDENTIAL){
        addFloat(dstRef, res, linearResRef, 1.0f, linearResTensor.GetTotalTensorSize());
    } else{
        addFloat(dstRef, res, linearResRef, gamma, linearResTensor.GetTotalTensorSize());
    }

    LinearResFWD.Run(src, weight, dst, bias, res, linearRes, gamma);

    float maxValue = findMaxValue<float>(src, srcTensor.GetTotalTensorSize(), weight, weightTensor.GetTotalTensorSize());

    bool flag = true;
    for (SizeType i = 0; i < linearResTensor.GetTotalTensorSize(); ++i) {
        float diff = std::abs(static_cast<float>(*(linearRes + i) - *(linearResRef + i)));
        auto pairEps = compareEle(maxValue, diff, KuDNN::Element::MatchType<float>(), src_col, L);
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
    if (res != nullptr){
        free(res);
    }
    if (linearRes != nullptr){
        free(linearRes);
    }
    if (linearResRef != nullptr){
        free(linearResRef);
    }
    return flag;
}

static bool Test_Template_Function_FP16(const KuDNN::TensorInfo srcTensor, const KuDNN::TensorInfo weightTensor,
                                        const KuDNN::TensorInfo dstTensor, const KuDNN::TensorInfo biasTensor,
                                        const KuDNN::TensorInfo resTensor, const KuDNN::TensorInfo linearResTensor,
                                        float alpha, float beta, float gamma, KuDNN::ResOpsFunction algKind, int numThreads){

    KuDNN::LinearResFWD LinearResFWD(srcTensor, weightTensor, dstTensor, biasTensor, resTensor, linearResTensor, alpha, beta, gamma, algKind, numThreads);

    __fp16 *src = nullptr, *weight = nullptr, *dst = nullptr, *dstRef = nullptr, *bias = nullptr, *res = nullptr, *linearRes = nullptr, *linearResRef = nullptr;
    DataInit<__fp16>(&src, &weight, &dst, &dstRef, &bias, &res, &linearRes, &linearResRef, srcTensor.GetTotalTensorSize(), weightTensor.GetTotalTensorSize(),
                     dstTensor.GetTotalTensorSize(), resTensor.GetTotalTensorSize(), linearResTensor.GetTotalTensorSize());



    Shape srcShape = srcTensor.GetDims();
    Shape weightShape = weightTensor.GetDims();
    Shape resShape = resTensor.GetDims();
    SizeType srcDimNums = srcShape.GetNumDims();
    SizeType weightDimNums = weightShape.GetNumDims();
    SizeType resDimNums = resShape.GetNumDims();
    SizeType dst_row = srcShape[srcDimNums - 2];             // m
    SizeType dst_col = weightShape[weightDimNums - 1];       // n
    SizeType src_col = srcShape[srcDimNums - 1];             // k
    SizeType res_col = resShape[resDimNums - 1];             // l
    SizeType LinearBatch = dstTensor.GetTotalTensorSize() / (dst_row * dst_col);
    SizeType PostOpsBatch = linearResTensor.GetTotalTensorSize() / (dst_row * res_col);

    SizeType L = 1; // compareEle: K *L

    linearSimpleFp16(LinearBatch, dst_row, dst_col, src_col, src, weight, dstRef, bias);
    if(algKind == KuDNN::ResOpsFunction::RES_MUL){
        L = dst_col;
        linearSimpleFp16(PostOpsBatch, dst_row, res_col, dst_col, dstRef, res, linearResRef, nullptr);
    } else if(algKind == KuDNN::ResOpsFunction::RES_IDENTIAL){
        addF16(dstRef, res, linearResRef, 1.0f, linearResTensor.GetTotalTensorSize());
    } else{
        addF16(dstRef, res, linearResRef, gamma, linearResTensor.GetTotalTensorSize());
    }

    LinearResFWD.Run(src, weight, dst, bias, res, linearRes, gamma);

    float maxValue = findMaxValue<__fp16>(src, srcTensor.GetTotalTensorSize(), weight, weightTensor.GetTotalTensorSize());

    bool flag = true;
    for (SizeType i = 0; i < linearResTensor.GetTotalTensorSize(); ++i) {
        float diff = std::abs(static_cast<float>(*(linearRes + i) - *(linearResRef + i)));
        auto pairEps = compareEle(maxValue, diff, KuDNN::Element::MatchType<__fp16>(), src_col, L);
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
    if (res != nullptr){
        free(res);
    }
    if (linearRes != nullptr){
        free(linearRes);
    }
    if (linearResRef != nullptr){
        free(linearResRef);
    }
    return flag;
}

static bool Test_Template_Function_BF16(const KuDNN::TensorInfo srcTensor, const KuDNN::TensorInfo weightTensor,
                                        const KuDNN::TensorInfo dstTensor, const KuDNN::TensorInfo biasTensor,
                                        const KuDNN::TensorInfo resTensor, const KuDNN::TensorInfo linearResTensor,
                                        float alpha, float beta, float gamma, KuDNN::ResOpsFunction algKind, int numThreads){

    KuDNN::LinearResFWD LinearResFWD(srcTensor, weightTensor, dstTensor, biasTensor, resTensor, linearResTensor, alpha, beta, gamma, algKind, numThreads);

    __bf16 *src = nullptr, *weight = nullptr, *dst = nullptr, *dstRef = nullptr, *bias = nullptr, *res = nullptr, *linearRes = nullptr, *linearResRef = nullptr;
    DataInit<__bf16>(&src, &weight, &dst, &dstRef, &bias, &res, &linearRes, &linearResRef, srcTensor.GetTotalTensorSize(), weightTensor.GetTotalTensorSize(),
                     dstTensor.GetTotalTensorSize(), resTensor.GetTotalTensorSize(), linearResTensor.GetTotalTensorSize());

    Shape srcShape = srcTensor.GetDims();
    Shape weightShape = weightTensor.GetDims();
    Shape resShape = resTensor.GetDims();
    SizeType srcDimNums = srcShape.GetNumDims();
    SizeType weightDimNums = weightShape.GetNumDims();
    SizeType resDimNums = resShape.GetNumDims();
    SizeType dst_row = srcShape[srcDimNums - 2];             // m
    SizeType dst_col = weightShape[weightDimNums - 1];       // n
    SizeType src_col = srcShape[srcDimNums - 1];             // k
    SizeType res_col = resShape[resDimNums - 1];             // l
    SizeType LinearBatch = dstTensor.GetTotalTensorSize() / (dst_row * dst_col);
    SizeType PostOpsBatch = linearResTensor.GetTotalTensorSize() / (dst_row * res_col);
    SizeType linearResSize = linearResTensor.GetTotalTensorSize();

    SizeType L = 1; // compareEle: K *L

    linearSimpleBf16(LinearBatch, dst_row, dst_col, src_col, src, weight, dstRef, bias);
    if(algKind == KuDNN::ResOpsFunction::RES_MUL){
        L = dst_col;
        linearSimpleBf16(PostOpsBatch, dst_row, res_col, dst_col, dstRef, res, linearResRef, nullptr);
    } else if(algKind == KuDNN::ResOpsFunction::RES_IDENTIAL){
        addBF16(dstRef, res, linearResRef, 1.0f, linearResSize);
    } else{
        addBF16(dstRef, res, linearResRef, gamma, linearResSize);
    }

    LinearResFWD.Run(src, weight, dst, bias, res, linearRes, gamma);


    float *linearRes_f = (float *)malloc(linearResSize * sizeof(float));
    CvtBf16ToFp32(linearRes, linearRes_f, linearResSize);

    float *linearResRef_f = (float *)malloc(linearResSize * sizeof(float));
    CvtBf16ToFp32(linearResRef, linearResRef_f, linearResSize);

    float maxValue = findMaxValue<__bf16>(src, srcTensor.GetTotalTensorSize(), weight, weightTensor.GetTotalTensorSize());

    bool flag = true;
    for (SizeType i = 0; i < linearResSize; ++i) {
        float diff = std::abs(static_cast<float>(*(linearRes_f + i) - *(linearResRef_f + i)));
        auto pairEps = compareEle(maxValue, diff, KuDNN::Element::MatchType<__bf16>(), src_col, L);
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
    if (res != nullptr){
        free(res);
    }
    if (linearRes != nullptr){
        free(linearRes);
    }
    if (linearResRef != nullptr){
        free(linearResRef);
    }
    if (linearRes_f != nullptr){
        free(linearRes_f);
    }
    if (linearResRef_f != nullptr){
        free(linearResRef_f);
    }
    return flag;
}


static void InputParamExceptTest1(){
    Shape srcShape(-1, 1, 1, 1, 1);
    const KuDNN::TensorInfo srcTensor = {srcShape, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCDE};
    const KuDNN::TensorInfo weightTensor = {{1, 1, 1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCDE};
    const KuDNN::TensorInfo dstTensor = {{1, 1, 1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCDE};
    const KuDNN::TensorInfo biasTensor = {{1, 1, 1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCDE};
    const KuDNN::TensorInfo resTensor = {{1, 1, 1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCDE};
    const KuDNN::TensorInfo linearResTensor = {{1, 1, 1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCDE};
    float alpha = 1.0f;
    float beta = 0.0f;
    float gamma = 2.0f;
    KuDNN::ResOpsFunction algKind = KuDNN::ResOpsFunction::RES_MUL;
    int numThreads = 0;
    KuDNN::LinearResFWD LinearResFWD(srcTensor, weightTensor, dstTensor,
                                                             biasTensor, resTensor, linearResTensor, alpha, beta, gamma, algKind, numThreads);
}
static void InputParamExceptTest2(){
    const KuDNN::TensorInfo srcTensor = {{1, 1, 1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCDE};
    const KuDNN::TensorInfo weightTensor = {{1, 1, 1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCDE};
    const KuDNN::TensorInfo dstTensor = {{1, 1, 1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCDE};
    const KuDNN::TensorInfo biasTensor = {{1, 1, 1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCDE};
    const KuDNN::TensorInfo resTensor = {{1, 1, 1, 1, 1}, KuDNN::Element::TypeT::S32, KuDNN::Layout::ABCDE};
    const KuDNN::TensorInfo linearResTensor = {{1, 1, 1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCDE};
    float alpha = 1.0f;
    float beta = 0.0f;
    float gamma = 2.0f;
    KuDNN::ResOpsFunction algKind = KuDNN::ResOpsFunction::RES_MUL;
    int numThreads = 0;
    KuDNN::LinearResFWD LinearResFWD(srcTensor, weightTensor, dstTensor,
                                                             biasTensor, resTensor, linearResTensor, alpha, beta, gamma, algKind, numThreads);
}

static void InputParamExceptTest3(){
    const KuDNN::TensorInfo srcTensor = {{1, 1, 1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCD};
    const KuDNN::TensorInfo weightTensor = {{1, 1, 1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCDE};
    const KuDNN::TensorInfo dstTensor = {{1, 1, 1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCDE};
    const KuDNN::TensorInfo biasTensor = {{1, 1, 1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCDE};
    const KuDNN::TensorInfo resTensor = {{1, 1, 1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCDE};
    const KuDNN::TensorInfo linearResTensor = {{1, 1, 1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCDE};
    float alpha = 1.0f;
    float beta = 0.0f;
    float gamma = 2.0f;
    KuDNN::ResOpsFunction algKind = KuDNN::ResOpsFunction::RES_MUL;
    int numThreads = 0;
    KuDNN::LinearResFWD LinearResFWD(srcTensor, weightTensor, dstTensor,
                                                             biasTensor, resTensor, linearResTensor, alpha, beta, gamma, algKind, numThreads);
}

static void InputParamExceptTest4(){
    const KuDNN::TensorInfo srcTensor = {{1, 1, 1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCDE};
    const KuDNN::TensorInfo weightTensor = {{1, 1, 2, 2, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCDE};
    const KuDNN::TensorInfo dstTensor = {{1, 1, 1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCDE};
    const KuDNN::TensorInfo biasTensor = {{1, 1, 1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCDE};
    const KuDNN::TensorInfo resTensor = {{1, 1, 1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCDE};
    const KuDNN::TensorInfo linearResTensor = {{1, 1, 1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCDE};
    float alpha = 1.0f;
    float beta = 0.0f;
    float gamma = 2.0f;
    KuDNN::ResOpsFunction algKind = KuDNN::ResOpsFunction::RES_MUL;
    int numThreads = 0;
    KuDNN::LinearResFWD LinearResFWD(srcTensor, weightTensor, dstTensor,
                                                             biasTensor, resTensor, linearResTensor, alpha, beta, gamma, algKind, numThreads);
}

static void InputNullptrExceptTest5(){
    const KuDNN::TensorInfo srcTensor = {{1, 1, 1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCDE};
    const KuDNN::TensorInfo weightTensor = {{1, 1, 1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCDE};
    const KuDNN::TensorInfo dstTensor = {{1, 1, 1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCDE};
    const KuDNN::TensorInfo biasTensor = {{1, 1, 1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCDE};
    const KuDNN::TensorInfo resTensor = {{1, 1, 1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCDE};
    const KuDNN::TensorInfo linearResTensor = {{1, 1, 1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCDE};
    float alpha = 1.0f;
    float beta = 0.0f;
    float gamma = 2.0f;
    KuDNN::ResOpsFunction algKind = KuDNN::ResOpsFunction::RES_MUL;
    int numThreads = 0;
    float *src = nullptr, *weight = nullptr, *dst = nullptr,  *bias = nullptr, *res = nullptr, *linearRes = nullptr;
    KuDNN::LinearResFWD LinearResFWD(srcTensor, weightTensor, dstTensor,
                                                             biasTensor, resTensor, linearResTensor, alpha, beta, gamma, algKind, numThreads);
    LinearResFWD.Run(src, weight, dst, bias, res, linearRes, gamma);
}

static void InputParamExceptTest6(){
    Shape srcShape(1, -1, 1, 1, 1);
    const KuDNN::TensorInfo srcTensor = {srcShape, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCDE};
    const KuDNN::TensorInfo weightTensor = {{1, 1, 1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCDE};
    const KuDNN::TensorInfo dstTensor = {{1, 1, 1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCDE};
    const KuDNN::TensorInfo biasTensor = {{1, 1, 1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCDE};
    const KuDNN::TensorInfo resTensor = {{1, 1, 1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCDE};
    const KuDNN::TensorInfo linearResTensor = {{1, 1, 1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCDE};
    float alpha = 1.0f;
    float beta = 0.0f;
    float gamma = 2.0f;
    KuDNN::ResOpsFunction algKind = KuDNN::ResOpsFunction::RES_MUL;
    int numThreads = 0;
    KuDNN::LinearResFWD LinearResFWD(srcTensor, weightTensor, dstTensor,
                                                             biasTensor, resTensor, linearResTensor, alpha, beta, gamma, algKind, numThreads);
}

static void InputParamExceptTest7(){
    Shape srcShape(1, 1, -1, 1, 1);
    const KuDNN::TensorInfo srcTensor = {srcShape, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCDE};
    const KuDNN::TensorInfo weightTensor = {{1, 1, 1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCDE};
    const KuDNN::TensorInfo dstTensor = {{1, 1, 1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCDE};
    const KuDNN::TensorInfo biasTensor = {{1, 1, 1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCDE};
    const KuDNN::TensorInfo resTensor = {{1, 1, 1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCDE};
    const KuDNN::TensorInfo linearResTensor = {{1, 1, 1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCDE};
    float alpha = 1.0f;
    float beta = 0.0f;
    float gamma = 2.0f;
    KuDNN::ResOpsFunction algKind = KuDNN::ResOpsFunction::RES_MUL;
    int numThreads = 0;
    KuDNN::LinearResFWD LinearResFWD(srcTensor, weightTensor, dstTensor,
                                                             biasTensor, resTensor, linearResTensor, alpha, beta, gamma, algKind, numThreads);
}

static void InputParamExceptTest8(){
    Shape srcShape(1, 1, 1, -1, 1);
    const KuDNN::TensorInfo srcTensor = {srcShape, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCDE};
    const KuDNN::TensorInfo weightTensor = {{1, 1, 1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCDE};
    const KuDNN::TensorInfo dstTensor = {{1, 1, 1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCDE};
    const KuDNN::TensorInfo biasTensor = {{1, 1, 1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCDE};
    const KuDNN::TensorInfo resTensor = {{1, 1, 1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCDE};
    const KuDNN::TensorInfo linearResTensor = {{1, 1, 1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCDE};
    float alpha = 1.0f;
    float beta = 0.0f;
    float gamma = 2.0f;
    KuDNN::ResOpsFunction algKind = KuDNN::ResOpsFunction::RES_MUL;
    int numThreads = 0;
    KuDNN::LinearResFWD LinearResFWD(srcTensor, weightTensor, dstTensor,
                                                             biasTensor, resTensor, linearResTensor, alpha, beta, gamma, algKind, numThreads);
}

static void InputParamExceptTest9(){
    Shape weiShape(1, 1, 1, 1, -1);
    const KuDNN::TensorInfo srcTensor = {{1, 1, 1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCDE};
    const KuDNN::TensorInfo weightTensor = {weiShape, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCDE};
    const KuDNN::TensorInfo dstTensor = {{1, 1, 1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCDE};
    const KuDNN::TensorInfo biasTensor = {{1, 1, 1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCDE};
    const KuDNN::TensorInfo resTensor = {{1, 1, 1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCDE};
    const KuDNN::TensorInfo linearResTensor = {{1, 1, 1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCDE};
    float alpha = 1.0f;
    float beta = 0.0f;
    float gamma = 2.0f;
    KuDNN::ResOpsFunction algKind = KuDNN::ResOpsFunction::RES_MUL;
    int numThreads = 0;
    KuDNN::LinearResFWD LinearResFWD(srcTensor, weightTensor, dstTensor,
                                                             biasTensor, resTensor, linearResTensor, alpha, beta, gamma, algKind, numThreads);
}

static void InputParamExceptTest11(){
    Shape srcShape(1, 1, 1, 1, 1);
    const KuDNN::TensorInfo srcTensor = {srcShape, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCDE};
    const KuDNN::TensorInfo weightTensor = {{1, 1, 1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCDE};
    const KuDNN::TensorInfo dstTensor = {{1, 1, 1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCDE};
    const KuDNN::TensorInfo biasTensor = {{1, 1, 1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCDE};
    const KuDNN::TensorInfo resTensor = {{1, 1, 1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCDE};
    const KuDNN::TensorInfo linearResTensor = {{1, 1, 1, 1, 1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCDE};
    float alpha = 1.0f;
    float beta = 0.0f;
    float gamma = 2.0f;
    KuDNN::ResOpsFunction algKind = KuDNN::ResOpsFunction::RES_MUL;
    int numThreads = -1;
    KuDNN::LinearResFWD LinearResFWD(srcTensor, weightTensor, dstTensor,
                                                             biasTensor, resTensor, linearResTensor, alpha, beta, gamma, algKind, numThreads);
}


// test
TEST(LinearResTests, Param_ShapeExcept1_Test)
{
EXPECT_THROW(InputParamExceptTest1(), KuDNN::Service::LogicError);
}


TEST(LinearResTests, Param_ShapeExcept2_Test)
{
EXPECT_THROW(InputParamExceptTest4(), KuDNN::Service::LogicError);
}

TEST(LinearResTests, Param_DataType_Except_Test)
{
EXPECT_THROW(InputParamExceptTest2(), KuDNN::Service::Unsupported);
}

TEST(LinearResTests, Param_Layout_Except_Test)
{
EXPECT_THROW(InputParamExceptTest3(), KuDNN::Service::LogicError);
}

TEST(LinearResTests, Input_Nullptr_Except_Test)
{
EXPECT_THROW(InputNullptrExceptTest5(), KuDNN::Service::LogicError);
}

TEST(LinearResTests, Param_ShapeExceptD2_Test)
{
EXPECT_THROW(InputParamExceptTest6(), KuDNN::Service::LogicError);
}

TEST(LinearResTests, Param_ShapeExceptD3_Test)
{
EXPECT_THROW(InputParamExceptTest7(), KuDNN::Service::LogicError);
}

TEST(LinearResTests, Param_ShapeExceptM_Test)
{
EXPECT_THROW(InputParamExceptTest8(), KuDNN::Service::LogicError);
}

TEST(LinearResTests, Param_ShapeExceptN_Test)
{
EXPECT_THROW(InputParamExceptTest9(), KuDNN::Service::LogicError);
}

TEST(LinearResTests, Param_NumThreads_Except_Test)
{
EXPECT_THROW(InputParamExceptTest11(), KuDNN::Service::LogicError);
}

// test MUL FP32
TEST(LinearResTests, FWD_LINEAR_RES_MUL_2D_FP32)
{
const KuDNN::TensorInfo srcTensor = {{400, 200}, KuDNN::Element::TypeT::F32, KuDNN::Layout::AB};
const KuDNN::TensorInfo weightTensor = {{200, 500}, KuDNN::Element::TypeT::F32, KuDNN::Layout::AB};
const KuDNN::TensorInfo dstTensor = {{400, 500}, KuDNN::Element::TypeT::F32, KuDNN::Layout::AB};
const KuDNN::TensorInfo biasTensor = {{400, 500}, KuDNN::Element::TypeT::F32, KuDNN::Layout::AB};
const KuDNN::TensorInfo resTensor = {{500, 400}, KuDNN::Element::TypeT::F32, KuDNN::Layout::AB};
const KuDNN::TensorInfo linearResTensor = {{400, 400}, KuDNN::Element::TypeT::F32, KuDNN::Layout::AB};
float alpha = 1.0f;
float beta = 0.0f;
float gamma = 2.0f;
KuDNN::ResOpsFunction algKind = KuDNN::ResOpsFunction::RES_MUL;
int numThreads = 0;
EXPECT_TRUE(Test_Template_Function_FP32(srcTensor, weightTensor, dstTensor, biasTensor, resTensor,
                                        linearResTensor,alpha, beta, gamma,algKind, numThreads));
}

TEST(LinearResTests, FWD_LINEAR_RES_MUL_3D_FP32)
{
const KuDNN::TensorInfo srcTensor = {{2, 400, 200}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABC};
const KuDNN::TensorInfo weightTensor = {{2, 200, 500}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABC};
const KuDNN::TensorInfo dstTensor = {{2, 400, 500}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABC};
const KuDNN::TensorInfo biasTensor = {{2, 400, 500}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABC};
const KuDNN::TensorInfo resTensor = {{2, 500, 400}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABC};
const KuDNN::TensorInfo linearResTensor = {{2, 400, 400}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABC};
float alpha = 1.0f;
float beta = 0.0f;
float gamma = 2.0f;
KuDNN::ResOpsFunction algKind = KuDNN::ResOpsFunction::RES_MUL;
int numThreads = 0;
EXPECT_TRUE(Test_Template_Function_FP32(srcTensor, weightTensor, dstTensor, biasTensor, resTensor,
                                        linearResTensor,alpha, beta, gamma,algKind, numThreads));
}

TEST(LinearResTests, FWD_LINEAR_RES_MUL_5D_FP32)
{
const KuDNN::TensorInfo srcTensor = {{4, 3, 2, 40, 20}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCDE};
const KuDNN::TensorInfo weightTensor = {{4, 3, 2, 20, 50}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCDE};
const KuDNN::TensorInfo dstTensor = {{4, 3, 2, 40, 50}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCDE};
const KuDNN::TensorInfo biasTensor = {{4, 3, 2, 40, 50}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCDE};
const KuDNN::TensorInfo resTensor = {{4, 3, 2, 50, 40}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCDE};
const KuDNN::TensorInfo linearResTensor = {{4, 3, 2, 40, 40}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCDE};
float alpha = 1.0f;
float beta = 0.0f;
float gamma = 2.0f;
KuDNN::ResOpsFunction algKind = KuDNN::ResOpsFunction::RES_MUL;
int numThreads = 0;
EXPECT_TRUE(Test_Template_Function_FP32(srcTensor, weightTensor, dstTensor, biasTensor, resTensor,
                                        linearResTensor,alpha, beta, gamma,algKind, numThreads));
}

// test MUL FP16
TEST(LinearResTests, FWD_LINEAR_RES_MUL_2D_FP16)
{
const KuDNN::TensorInfo srcTensor = {{110, 200}, KuDNN::Element::TypeT::F16, KuDNN::Layout::AB};
const KuDNN::TensorInfo weightTensor = {{200, 300}, KuDNN::Element::TypeT::F16, KuDNN::Layout::AB};
const KuDNN::TensorInfo dstTensor = {{110, 300}, KuDNN::Element::TypeT::F16, KuDNN::Layout::AB};
const KuDNN::TensorInfo biasTensor = {{110, 300}, KuDNN::Element::TypeT::F16, KuDNN::Layout::AB};
const KuDNN::TensorInfo resTensor = {{300, 400}, KuDNN::Element::TypeT::F16, KuDNN::Layout::AB};
const KuDNN::TensorInfo linearResTensor = {{110, 400}, KuDNN::Element::TypeT::F16, KuDNN::Layout::AB};
float alpha = 1.0f;
float beta = 0.0f;
float gamma = 1.0f;
KuDNN::ResOpsFunction algKind = KuDNN::ResOpsFunction::RES_MUL;
int numThreads = 0;
EXPECT_TRUE(Test_Template_Function_FP16(srcTensor, weightTensor, dstTensor, biasTensor, resTensor,
                                        linearResTensor,alpha, beta, gamma,algKind, numThreads));
}

TEST(LinearResTests, FWD_LINEAR_RES_MUL_3D_FP16)
{
const KuDNN::TensorInfo srcTensor = {{2, 110, 200}, KuDNN::Element::TypeT::F16, KuDNN::Layout::ABC};
const KuDNN::TensorInfo weightTensor = {{2, 200, 300}, KuDNN::Element::TypeT::F16, KuDNN::Layout::ABC};
const KuDNN::TensorInfo dstTensor = {{2, 110, 300}, KuDNN::Element::TypeT::F16, KuDNN::Layout::ABC};
const KuDNN::TensorInfo biasTensor = {{2, 110, 300}, KuDNN::Element::TypeT::F16, KuDNN::Layout::ABC};
const KuDNN::TensorInfo resTensor = {{2, 300, 400}, KuDNN::Element::TypeT::F16, KuDNN::Layout::ABC};
const KuDNN::TensorInfo linearResTensor = {{2, 110, 400}, KuDNN::Element::TypeT::F16, KuDNN::Layout::ABC};
float alpha = 1.0f;
float beta = 0.0f;
float gamma = 1.0f;
KuDNN::ResOpsFunction algKind = KuDNN::ResOpsFunction::RES_MUL;
int numThreads = 0;
EXPECT_TRUE(Test_Template_Function_FP16(srcTensor, weightTensor, dstTensor, biasTensor, resTensor,
                                        linearResTensor,alpha, beta, gamma,algKind, numThreads));
}

TEST(LinearResTests, FWD_LINEAR_RES_MUL_5D_FP16)
{
const KuDNN::TensorInfo srcTensor = {{4, 3, 2, 11, 20}, KuDNN::Element::TypeT::F16, KuDNN::Layout::ABCDE};
const KuDNN::TensorInfo weightTensor = {{4, 3, 2, 20, 30}, KuDNN::Element::TypeT::F16, KuDNN::Layout::ABCDE};
const KuDNN::TensorInfo dstTensor = {{4, 3, 2, 11, 30}, KuDNN::Element::TypeT::F16, KuDNN::Layout::ABCDE};
const KuDNN::TensorInfo biasTensor = {{4, 3, 2, 11, 30}, KuDNN::Element::TypeT::F16, KuDNN::Layout::ABCDE};
const KuDNN::TensorInfo resTensor = {{4, 3, 2, 30, 40}, KuDNN::Element::TypeT::F16, KuDNN::Layout::ABCDE};
const KuDNN::TensorInfo linearResTensor = {{4, 3, 2, 11, 40}, KuDNN::Element::TypeT::F16, KuDNN::Layout::ABCDE};
float alpha = 1.0f;
float beta = 0.0f;
float gamma = 1.0f;
KuDNN::ResOpsFunction algKind = KuDNN::ResOpsFunction::RES_MUL;
int numThreads = 0;
EXPECT_TRUE(Test_Template_Function_FP16(srcTensor, weightTensor, dstTensor, biasTensor, resTensor,
                                        linearResTensor,alpha, beta, gamma,algKind, numThreads));
}

// test MUL BF16
TEST(LinearResTests, FWD_LINEAR_RES_MUL_2D_BF16)
{
const KuDNN::TensorInfo srcTensor = {{11, 20}, KuDNN::Element::TypeT::BF16, KuDNN::Layout::AB};
const KuDNN::TensorInfo weightTensor = {{20, 30}, KuDNN::Element::TypeT::BF16, KuDNN::Layout::AB};
const KuDNN::TensorInfo dstTensor = {{11, 30}, KuDNN::Element::TypeT::BF16, KuDNN::Layout::AB};
const KuDNN::TensorInfo biasTensor = {{11, 30}, KuDNN::Element::TypeT::BF16, KuDNN::Layout::AB};
const KuDNN::TensorInfo resTensor = {{30, 40}, KuDNN::Element::TypeT::BF16, KuDNN::Layout::AB};
const KuDNN::TensorInfo linearResTensor = {{11, 40}, KuDNN::Element::TypeT::BF16, KuDNN::Layout::AB};
float alpha = 1.0f;
float beta = 0.0f;
float gamma = 1.0f;
KuDNN::ResOpsFunction algKind = KuDNN::ResOpsFunction::RES_MUL;
int numThreads = 0;
EXPECT_TRUE(Test_Template_Function_BF16(srcTensor, weightTensor, dstTensor, biasTensor, resTensor,
                                        linearResTensor,alpha, beta, gamma,algKind, numThreads));
}

TEST(LinearResTests, FWD_LINEAR_RES_MUL_3D_BF16)
{
const KuDNN::TensorInfo srcTensor = {{2, 11, 20}, KuDNN::Element::TypeT::BF16, KuDNN::Layout::ABC};
const KuDNN::TensorInfo weightTensor = {{2, 20, 30}, KuDNN::Element::TypeT::BF16, KuDNN::Layout::ABC};
const KuDNN::TensorInfo dstTensor = {{2, 11, 30}, KuDNN::Element::TypeT::BF16, KuDNN::Layout::ABC};
const KuDNN::TensorInfo biasTensor = {{2, 11, 30}, KuDNN::Element::TypeT::BF16, KuDNN::Layout::ABC};
const KuDNN::TensorInfo resTensor = {{2, 30, 40}, KuDNN::Element::TypeT::BF16, KuDNN::Layout::ABC};
const KuDNN::TensorInfo linearResTensor = {{2, 11, 40}, KuDNN::Element::TypeT::BF16, KuDNN::Layout::ABC};
float alpha = 1.0f;
float beta = 0.0f;
float gamma = 1.0f;
KuDNN::ResOpsFunction algKind = KuDNN::ResOpsFunction::RES_MUL;
int numThreads = 0;
EXPECT_TRUE(Test_Template_Function_BF16(srcTensor, weightTensor, dstTensor, biasTensor, resTensor,
                                        linearResTensor,alpha, beta, gamma,algKind, numThreads));
}

TEST(LinearResTests, FWD_LINEAR_RES_MUL_5D_BF16)
{
const KuDNN::TensorInfo srcTensor = {{4, 3, 2, 11, 20}, KuDNN::Element::TypeT::BF16, KuDNN::Layout::ABCDE};
const KuDNN::TensorInfo weightTensor = {{4, 3, 2, 20, 30}, KuDNN::Element::TypeT::BF16, KuDNN::Layout::ABCDE};
const KuDNN::TensorInfo dstTensor = {{4, 3, 2, 11, 30}, KuDNN::Element::TypeT::BF16, KuDNN::Layout::ABCDE};
const KuDNN::TensorInfo biasTensor = {{4, 3, 2, 11, 30}, KuDNN::Element::TypeT::BF16, KuDNN::Layout::ABCDE};
const KuDNN::TensorInfo resTensor = {{4, 3, 2, 30, 40}, KuDNN::Element::TypeT::BF16, KuDNN::Layout::ABCDE};
const KuDNN::TensorInfo linearResTensor = {{4, 3, 2, 11, 40}, KuDNN::Element::TypeT::BF16, KuDNN::Layout::ABCDE};
float alpha = 1.0f;
float beta = 0.0f;
float gamma = 1.0f;
KuDNN::ResOpsFunction algKind = KuDNN::ResOpsFunction::RES_MUL;
int numThreads = 0;
EXPECT_TRUE(Test_Template_Function_BF16(srcTensor, weightTensor, dstTensor, biasTensor, resTensor,
                                        linearResTensor,alpha, beta, gamma,algKind, numThreads));
}

// test RES_EXT FP32
TEST(LinearResTests, FWD_LINEAR_RES_EXT_2D_FP32)
{
const KuDNN::TensorInfo srcTensor = {{40, 20}, KuDNN::Element::TypeT::F32, KuDNN::Layout::AB};
const KuDNN::TensorInfo weightTensor = {{20, 50}, KuDNN::Element::TypeT::F32, KuDNN::Layout::AB};
const KuDNN::TensorInfo dstTensor = {{40, 50}, KuDNN::Element::TypeT::F32, KuDNN::Layout::AB};
const KuDNN::TensorInfo biasTensor = {{40, 50}, KuDNN::Element::TypeT::F32, KuDNN::Layout::AB};
const KuDNN::TensorInfo resTensor = {{40, 50}, KuDNN::Element::TypeT::F32, KuDNN::Layout::AB};
const KuDNN::TensorInfo linearResTensor = {{40, 50}, KuDNN::Element::TypeT::F32, KuDNN::Layout::AB};
float alpha = 1.0f;
float beta = 0.0f;
float gamma = 2.0f;
KuDNN::ResOpsFunction algKind = KuDNN::ResOpsFunction::RES_EXT;
int numThreads = 0;
EXPECT_TRUE(Test_Template_Function_FP32(srcTensor, weightTensor, dstTensor, biasTensor, resTensor,
                                        linearResTensor,alpha, beta, gamma,algKind, numThreads));
}

TEST(LinearResTests, FWD_LINEAR_RES_EXT_3D_FP32)
{
const KuDNN::TensorInfo srcTensor = {{2, 40, 20}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABC};
const KuDNN::TensorInfo weightTensor = {{2, 20, 50}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABC};
const KuDNN::TensorInfo dstTensor = {{2, 40, 50}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABC};
const KuDNN::TensorInfo biasTensor = {{2, 40, 50}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABC};
const KuDNN::TensorInfo resTensor = {{2, 40, 50}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABC};
const KuDNN::TensorInfo linearResTensor = {{2, 40, 50}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABC};
float alpha = 1.0f;
float beta = 0.0f;
float gamma = 2.0f;
KuDNN::ResOpsFunction algKind = KuDNN::ResOpsFunction::RES_EXT;
int numThreads = 0;
EXPECT_TRUE(Test_Template_Function_FP32(srcTensor, weightTensor, dstTensor, biasTensor, resTensor,
                                        linearResTensor,alpha, beta, gamma,algKind, numThreads));
}

TEST(LinearResTests, FWD_LINEAR_RES_EXT_5D_FP32)
{
const KuDNN::TensorInfo srcTensor = {{4, 3, 2, 40, 20}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCDE};
const KuDNN::TensorInfo weightTensor = {{4, 3, 2, 20, 50}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCDE};
const KuDNN::TensorInfo dstTensor = {{4, 3, 2, 40, 50}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCDE};
const KuDNN::TensorInfo biasTensor = {{4, 3, 2, 40, 50}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCDE};
const KuDNN::TensorInfo resTensor = {{4, 3, 2, 40, 50}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCDE};
const KuDNN::TensorInfo linearResTensor = {{4, 3, 2, 40, 50}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCDE};
float alpha = 1.0f;
float beta = 0.0f;
float gamma = 2.0f;
KuDNN::ResOpsFunction algKind = KuDNN::ResOpsFunction::RES_EXT;
int numThreads = 0;
EXPECT_TRUE(Test_Template_Function_FP32(srcTensor, weightTensor, dstTensor, biasTensor, resTensor,
                                        linearResTensor,alpha, beta, gamma,algKind, numThreads));
}

// test RES_EXT FP16
TEST(LinearResTests, FWD_LINEAR_RES_EXT_2D_FP16)
{
const KuDNN::TensorInfo srcTensor = {{11, 20}, KuDNN::Element::TypeT::F16, KuDNN::Layout::AB};
const KuDNN::TensorInfo weightTensor = {{20, 30}, KuDNN::Element::TypeT::F16, KuDNN::Layout::AB};
const KuDNN::TensorInfo dstTensor = {{11, 30}, KuDNN::Element::TypeT::F16, KuDNN::Layout::AB};
const KuDNN::TensorInfo biasTensor = {{11, 30}, KuDNN::Element::TypeT::F16, KuDNN::Layout::AB};
const KuDNN::TensorInfo resTensor = {{11, 30}, KuDNN::Element::TypeT::F16, KuDNN::Layout::AB};
const KuDNN::TensorInfo linearResTensor = {{11, 30}, KuDNN::Element::TypeT::F16, KuDNN::Layout::AB};
float alpha = 1.0f;
float beta = 0.0f;
float gamma = 2.0f;
KuDNN::ResOpsFunction algKind = KuDNN::ResOpsFunction::RES_EXT;
int numThreads = 0;
EXPECT_TRUE(Test_Template_Function_FP16(srcTensor, weightTensor, dstTensor, biasTensor, resTensor,
                                        linearResTensor,alpha, beta, gamma,algKind, numThreads));
}

TEST(LinearResTests, FWD_LINEAR_RES_EXT_5D_FP16)
{
const KuDNN::TensorInfo srcTensor = {{4, 3, 2, 11, 20}, KuDNN::Element::TypeT::F16, KuDNN::Layout::ABCDE};
const KuDNN::TensorInfo weightTensor = {{4, 3, 2, 20, 30}, KuDNN::Element::TypeT::F16, KuDNN::Layout::ABCDE};
const KuDNN::TensorInfo dstTensor = {{4, 3, 2, 11, 30}, KuDNN::Element::TypeT::F16, KuDNN::Layout::ABCDE};
const KuDNN::TensorInfo biasTensor = {{4, 3, 2, 11, 30}, KuDNN::Element::TypeT::F16, KuDNN::Layout::ABCDE};
const KuDNN::TensorInfo resTensor = {{4, 3, 2, 11, 30}, KuDNN::Element::TypeT::F16, KuDNN::Layout::ABCDE};
const KuDNN::TensorInfo linearResTensor = {{4, 3, 2, 11, 30}, KuDNN::Element::TypeT::F16, KuDNN::Layout::ABCDE};
float alpha = 1.0f;
float beta = 0.0f;
float gamma = 2.0f;
KuDNN::ResOpsFunction algKind = KuDNN::ResOpsFunction::RES_EXT;
int numThreads = 0;
EXPECT_TRUE(Test_Template_Function_FP16(srcTensor, weightTensor, dstTensor, biasTensor, resTensor,
                                        linearResTensor,alpha, beta, gamma,algKind, numThreads));
}

// test RES_EXT BF16
TEST(LinearResTests, FWD_LINEAR_RES_EXT_2D_BF16)
{
const KuDNN::TensorInfo srcTensor = {{110, 200}, KuDNN::Element::TypeT::BF16, KuDNN::Layout::AB};
const KuDNN::TensorInfo weightTensor = {{200, 300}, KuDNN::Element::TypeT::BF16, KuDNN::Layout::AB};
const KuDNN::TensorInfo dstTensor = {{110, 300}, KuDNN::Element::TypeT::BF16, KuDNN::Layout::AB};
const KuDNN::TensorInfo biasTensor = {{110, 300}, KuDNN::Element::TypeT::BF16, KuDNN::Layout::AB};
const KuDNN::TensorInfo resTensor = {{110, 300}, KuDNN::Element::TypeT::BF16, KuDNN::Layout::AB};
const KuDNN::TensorInfo linearResTensor = {{110, 300}, KuDNN::Element::TypeT::BF16, KuDNN::Layout::AB};
float alpha = 1.0f;
float beta = 0.0f;
float gamma = 2.0f;
KuDNN::ResOpsFunction algKind = KuDNN::ResOpsFunction::RES_EXT;
int numThreads = 0;
EXPECT_TRUE(Test_Template_Function_BF16(srcTensor, weightTensor, dstTensor, biasTensor, resTensor,
                                        linearResTensor,alpha, beta, gamma,algKind, numThreads));
}

TEST(LinearResTests, FWD_LINEAR_RES_EXT_5D_BF16)
{
const KuDNN::TensorInfo srcTensor = {{4, 3, 2, 11, 20}, KuDNN::Element::TypeT::BF16, KuDNN::Layout::ABCDE};
const KuDNN::TensorInfo weightTensor = {{4, 3, 2, 20, 30}, KuDNN::Element::TypeT::BF16, KuDNN::Layout::ABCDE};
const KuDNN::TensorInfo dstTensor = {{4, 3, 2, 11, 30}, KuDNN::Element::TypeT::BF16, KuDNN::Layout::ABCDE};
const KuDNN::TensorInfo biasTensor = {{4, 3, 2, 11, 30}, KuDNN::Element::TypeT::BF16, KuDNN::Layout::ABCDE};
const KuDNN::TensorInfo resTensor = {{4, 3, 2, 11, 30}, KuDNN::Element::TypeT::BF16, KuDNN::Layout::ABCDE};
const KuDNN::TensorInfo linearResTensor = {{4, 3, 2, 11, 30}, KuDNN::Element::TypeT::BF16, KuDNN::Layout::ABCDE};
float alpha = 1.0f;
float beta = 0.0f;
float gamma = 2.0f;
KuDNN::ResOpsFunction algKind = KuDNN::ResOpsFunction::RES_EXT;
int numThreads = 0;
EXPECT_TRUE(Test_Template_Function_BF16(srcTensor, weightTensor, dstTensor, biasTensor, resTensor,
                                        linearResTensor,alpha, beta, gamma,algKind, numThreads));
}

// test RES_IDENTIAL FP32
TEST(LinearResTests, FWD_LINEAR_RES_IDENTIAL_2D_FP32)
{
const KuDNN::TensorInfo srcTensor = {{30, 20}, KuDNN::Element::TypeT::F32, KuDNN::Layout::AB};
const KuDNN::TensorInfo weightTensor = {{20, 20}, KuDNN::Element::TypeT::F32, KuDNN::Layout::AB};
const KuDNN::TensorInfo dstTensor = {{30, 20}, KuDNN::Element::TypeT::F32, KuDNN::Layout::AB};
const KuDNN::TensorInfo biasTensor = {{30, 20}, KuDNN::Element::TypeT::F32, KuDNN::Layout::AB};
const KuDNN::TensorInfo resTensor = {{30, 20}, KuDNN::Element::TypeT::F32, KuDNN::Layout::AB};
const KuDNN::TensorInfo linearResTensor = {{30, 20}, KuDNN::Element::TypeT::F32, KuDNN::Layout::AB};
float alpha = 1.0f;
float beta = 0.0f;
float gamma = 2.0f;
KuDNN::ResOpsFunction algKind = KuDNN::ResOpsFunction::RES_IDENTIAL;
int numThreads = 0;
EXPECT_TRUE(Test_Template_Function_FP32(srcTensor, weightTensor, dstTensor, biasTensor, resTensor,
                                        linearResTensor,alpha, beta, gamma,algKind, numThreads));
}

TEST(LinearResTests, FWD_LINEAR_RES_IDENTIAL_5D_FP32)
{
const KuDNN::TensorInfo srcTensor = {{4, 3, 2, 30, 20}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCDE};
const KuDNN::TensorInfo weightTensor = {{4, 3, 2, 20, 20}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCDE};
const KuDNN::TensorInfo dstTensor = {{4, 3, 2, 30, 20}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCDE};
const KuDNN::TensorInfo biasTensor = {{4, 3, 2, 30, 20}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCDE};
const KuDNN::TensorInfo resTensor = {{4, 3, 2, 30, 20}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCDE};
const KuDNN::TensorInfo linearResTensor = {{4, 3, 2, 30, 20}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCDE};
float alpha = 1.0f;
float beta = 0.0f;
float gamma = 2.0f;
KuDNN::ResOpsFunction algKind = KuDNN::ResOpsFunction::RES_IDENTIAL;
int numThreads = 0;
EXPECT_TRUE(Test_Template_Function_FP32(srcTensor, weightTensor, dstTensor, biasTensor, resTensor,
                                        linearResTensor,alpha, beta, gamma,algKind, numThreads));
}

// test RES_IDENTIAL FP16
TEST(LinearResTests, FWD_LINEAR_RES_IDENTIAL_2D_FP16)
{
const KuDNN::TensorInfo srcTensor = {{11, 20}, KuDNN::Element::TypeT::F16, KuDNN::Layout::AB};
const KuDNN::TensorInfo weightTensor = {{20, 30}, KuDNN::Element::TypeT::F16, KuDNN::Layout::AB};
const KuDNN::TensorInfo dstTensor = {{11, 30}, KuDNN::Element::TypeT::F16, KuDNN::Layout::AB};
const KuDNN::TensorInfo biasTensor = {{11, 30}, KuDNN::Element::TypeT::F16, KuDNN::Layout::AB};
const KuDNN::TensorInfo resTensor = {{11, 30}, KuDNN::Element::TypeT::F16, KuDNN::Layout::AB};
const KuDNN::TensorInfo linearResTensor = {{11, 30}, KuDNN::Element::TypeT::F16, KuDNN::Layout::AB};
float alpha = 1.0f;
float beta = 0.0f;
float gamma = 1.0f;
KuDNN::ResOpsFunction algKind = KuDNN::ResOpsFunction::RES_IDENTIAL;
int numThreads = 0;
EXPECT_TRUE(Test_Template_Function_FP16(srcTensor, weightTensor, dstTensor, biasTensor, resTensor,
                                        linearResTensor,alpha, beta, gamma,algKind, numThreads));
}

TEST(LinearResTests, FWD_LINEAR_RES_IDENTIAL_3D_FP16)
{
const KuDNN::TensorInfo srcTensor = {{2, 30, 20}, KuDNN::Element::TypeT::F16, KuDNN::Layout::ABC};
const KuDNN::TensorInfo weightTensor = {{2, 20, 20}, KuDNN::Element::TypeT::F16, KuDNN::Layout::ABC};
const KuDNN::TensorInfo dstTensor = {{2, 30, 20}, KuDNN::Element::TypeT::F16, KuDNN::Layout::ABC};
const KuDNN::TensorInfo biasTensor = {{2, 30, 20}, KuDNN::Element::TypeT::F16, KuDNN::Layout::ABC};
const KuDNN::TensorInfo resTensor = {{2, 30, 20}, KuDNN::Element::TypeT::F16, KuDNN::Layout::ABC};
const KuDNN::TensorInfo linearResTensor = {{2, 30, 20}, KuDNN::Element::TypeT::F16, KuDNN::Layout::ABC};
float alpha = 1.0f;
float beta = 0.0f;
float gamma = 1.0f;
KuDNN::ResOpsFunction algKind = KuDNN::ResOpsFunction::RES_IDENTIAL;
int numThreads = 0;
EXPECT_TRUE(Test_Template_Function_FP16(srcTensor, weightTensor, dstTensor, biasTensor, resTensor,
                                        linearResTensor,alpha, beta, gamma,algKind, numThreads));
}

TEST(LinearResTests, FWD_LINEAR_RES_IDENTIAL_4D_FP16)
{
const KuDNN::TensorInfo srcTensor = {{3, 2, 30, 20}, KuDNN::Element::TypeT::F16, KuDNN::Layout::ABCD};
const KuDNN::TensorInfo weightTensor = {{3, 2, 20, 20}, KuDNN::Element::TypeT::F16, KuDNN::Layout::ABCD};
const KuDNN::TensorInfo dstTensor = {{3, 2, 30, 20}, KuDNN::Element::TypeT::F16, KuDNN::Layout::ABCD};
const KuDNN::TensorInfo biasTensor = {{3, 2, 30, 20}, KuDNN::Element::TypeT::F16, KuDNN::Layout::ABCD};
const KuDNN::TensorInfo resTensor = {{3, 2, 30, 20}, KuDNN::Element::TypeT::F16, KuDNN::Layout::ABCD};
const KuDNN::TensorInfo linearResTensor = {{3, 2, 30, 20}, KuDNN::Element::TypeT::F16, KuDNN::Layout::ABCD};
float alpha = 1.0f;
float beta = 0.0f;
float gamma = 1.0f;
KuDNN::ResOpsFunction algKind = KuDNN::ResOpsFunction::RES_IDENTIAL;
int numThreads = 0;
EXPECT_TRUE(Test_Template_Function_FP16(srcTensor, weightTensor, dstTensor, biasTensor, resTensor,
                                        linearResTensor,alpha, beta, gamma,algKind, numThreads));
}

TEST(LinearResTests, FWD_LINEAR_RES_IDENTIAL_5D_FP16)
{
const KuDNN::TensorInfo srcTensor = {{4, 3, 2, 30, 20}, KuDNN::Element::TypeT::F16, KuDNN::Layout::ABCDE};
const KuDNN::TensorInfo weightTensor = {{4, 3, 2, 20, 20}, KuDNN::Element::TypeT::F16, KuDNN::Layout::ABCDE};
const KuDNN::TensorInfo dstTensor = {{4, 3, 2, 30, 20}, KuDNN::Element::TypeT::F16, KuDNN::Layout::ABCDE};
const KuDNN::TensorInfo biasTensor = {{4, 3, 2, 30, 20}, KuDNN::Element::TypeT::F16, KuDNN::Layout::ABCDE};
const KuDNN::TensorInfo resTensor = {{4, 3, 2, 30, 20}, KuDNN::Element::TypeT::F16, KuDNN::Layout::ABCDE};
const KuDNN::TensorInfo linearResTensor = {{4, 3, 2, 30, 20}, KuDNN::Element::TypeT::F16, KuDNN::Layout::ABCDE};
float alpha = 1.0f;
float beta = 0.0f;
float gamma = 1.0f;
KuDNN::ResOpsFunction algKind = KuDNN::ResOpsFunction::RES_IDENTIAL;
int numThreads = 0;
EXPECT_TRUE(Test_Template_Function_FP16(srcTensor, weightTensor, dstTensor, biasTensor, resTensor,
                                        linearResTensor,alpha, beta, gamma,algKind, numThreads));
}
// test RES_IDENTIAL BF16
TEST(LinearResTests, FWD_LINEAR_RES_IDENTIAL_2D_BF16)
{
const KuDNN::TensorInfo srcTensor = {{11, 20}, KuDNN::Element::TypeT::BF16, KuDNN::Layout::AB};
const KuDNN::TensorInfo weightTensor = {{20, 30}, KuDNN::Element::TypeT::BF16, KuDNN::Layout::AB};
const KuDNN::TensorInfo dstTensor = {{11, 30}, KuDNN::Element::TypeT::BF16, KuDNN::Layout::AB};
const KuDNN::TensorInfo biasTensor = {{11, 30}, KuDNN::Element::TypeT::BF16, KuDNN::Layout::AB};
const KuDNN::TensorInfo resTensor = {{11, 30}, KuDNN::Element::TypeT::BF16, KuDNN::Layout::AB};
const KuDNN::TensorInfo linearResTensor = {{11, 30}, KuDNN::Element::TypeT::BF16, KuDNN::Layout::AB};
float alpha = 1.0f;
float beta = 0.0f;
float gamma = 1.0f;
KuDNN::ResOpsFunction algKind = KuDNN::ResOpsFunction::RES_IDENTIAL;
int numThreads = 0;
EXPECT_TRUE(Test_Template_Function_BF16(srcTensor, weightTensor, dstTensor, biasTensor, resTensor,
                                        linearResTensor,alpha, beta, gamma,algKind, numThreads));
}

TEST(LinearResTests, FWD_LINEAR_RES_IDENTIAL_5D_BF16)
{
const KuDNN::TensorInfo srcTensor = {{4, 3, 2, 30, 20}, KuDNN::Element::TypeT::BF16, KuDNN::Layout::ABCDE};
const KuDNN::TensorInfo weightTensor = {{4, 3, 2, 20, 20}, KuDNN::Element::TypeT::BF16, KuDNN::Layout::ABCDE};
const KuDNN::TensorInfo dstTensor = {{4, 3, 2, 30, 20}, KuDNN::Element::TypeT::BF16, KuDNN::Layout::ABCDE};
const KuDNN::TensorInfo biasTensor = {{4, 3, 2, 30, 20}, KuDNN::Element::TypeT::BF16, KuDNN::Layout::ABCDE};
const KuDNN::TensorInfo resTensor = {{4, 3, 2, 30, 20}, KuDNN::Element::TypeT::BF16, KuDNN::Layout::ABCDE};
const KuDNN::TensorInfo linearResTensor = {{4, 3, 2, 30, 20}, KuDNN::Element::TypeT::BF16, KuDNN::Layout::ABCDE};
float alpha = 1.0f;
float beta = 0.0f;
float gamma = 1.0f;
KuDNN::ResOpsFunction algKind = KuDNN::ResOpsFunction::RES_IDENTIAL;
int numThreads = 0;
EXPECT_TRUE(Test_Template_Function_BF16(srcTensor, weightTensor, dstTensor, biasTensor, resTensor,
                                        linearResTensor,alpha, beta, gamma,algKind, numThreads));
}
