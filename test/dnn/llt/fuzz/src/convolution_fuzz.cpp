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

#include "fuzz_test_common.h"

static std::map<int, std::pair<std::string, KuDNN::Element::TypeT>> g_precisionMap = {
    {INDEX0, {"F32", KuDNN::Element::TypeT::F32}},
    {INDEX1, {"F16", KuDNN::Element::TypeT::F16}},
    {INDEX2, {"BF16", KuDNN::Element::TypeT::BF16}},
};

static std::map<int, std::pair<KuDNN::Element::TypeT, KuDNN::Element::TypeT>> g_intPrecisionMap = {
    {INDEX0, {KuDNN::Element::TypeT::S8, KuDNN::Element::TypeT::S8}},
    {INDEX1, {KuDNN::Element::TypeT::S8, KuDNN::Element::TypeT::U8}},
    {INDEX2, {KuDNN::Element::TypeT::S8, KuDNN::Element::TypeT::S32}},
    {INDEX2, {KuDNN::Element::TypeT::S8, KuDNN::Element::TypeT::F32}},
    {INDEX3, {KuDNN::Element::TypeT::U8, KuDNN::Element::TypeT::S8}},
    {INDEX4, {KuDNN::Element::TypeT::U8, KuDNN::Element::TypeT::U8}},
    {INDEX5, {KuDNN::Element::TypeT::U8, KuDNN::Element::TypeT::S32}},
    {INDEX5, {KuDNN::Element::TypeT::U8, KuDNN::Element::TypeT::F32}}
};

static std::map<int, std::pair<KuDNN::Element::TypeT, KuDNN::Element::TypeT>> g_bwdwMap = {
    {INDEX0, {KuDNN::Element::TypeT::F32, KuDNN::Element::TypeT::F32}},
    {INDEX1, {KuDNN::Element::TypeT::F16, KuDNN::Element::TypeT::F16}},
    {INDEX2, {KuDNN::Element::TypeT::BF16, KuDNN::Element::TypeT::BF16}},
    {INDEX3, {KuDNN::Element::TypeT::F16, KuDNN::Element::TypeT::F32}},
    {INDEX4, {KuDNN::Element::TypeT::BF16, KuDNN::Element::TypeT::F32}},
};

static std::map<int, std::pair<std::string, KuDNN::ConvolutionAlgorithm>> g_algMap = {
    {INDEX0, {"AUTO", KuDNN::ConvolutionAlgorithm::AUTO}},
    {INDEX1, {"DIRECT", KuDNN::ConvolutionAlgorithm::DIRECT}},
    {INDEX2, {"WINOGRAD", KuDNN::ConvolutionAlgorithm::WINOGRAD}},
};

static std::map<int, bool> g_biasMap = {
    {INDEX0, true},
    {INDEX1, false},
};

static std::map<int, KuDNN::Layout> g_layoutMap = {
    {INDEX0, KuDNN::Layout::ABCD},
    {INDEX1, KuDNN::Layout::ACDB},
};

static constexpr int g_dimRangeInitVal = 1;
static constexpr int g_dimRangeLowerBound = 1;
static constexpr int g_dimRangeUpperBound = 20;

static void DtFuzzConvolutionFwd(int counts)
{
    char funcName[] = "DtFuzzConvolutionFwd";
    DT_FUZZ_START(0, counts, funcName, 0)
    {
        int eleIndex = 0;
        int srcElemTypeIndex = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], 0, 0, g_precisionMap.size() - 1);
        int dstElemTypeIndex = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], 0, 0, g_precisionMap.size() - 1);
        int biasElemTypeIndex = dstElemTypeIndex;
        int algIndex = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], 0, 0, g_algMap.size() - 1);
        int hasBiasIndex = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], 0, 0, g_biasMap.size() - 1);
        int srcLayout = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], 0, 0, 1);
        int dstLayout = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], 0, 0, 1);
        size_t group = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], 1, 1, 5);
        size_t bc = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], g_dimRangeInitVal, g_dimRangeLowerBound,
            g_dimRangeUpperBound);
        size_t ic = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], g_dimRangeInitVal, g_dimRangeLowerBound,
            g_dimRangeUpperBound);
        size_t ih = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], g_dimRangeInitVal, g_dimRangeLowerBound,
            g_dimRangeUpperBound);
        size_t iw = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], g_dimRangeInitVal, g_dimRangeLowerBound,
            g_dimRangeUpperBound);

        size_t oc = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], g_dimRangeInitVal, g_dimRangeLowerBound,
            g_dimRangeUpperBound);
        size_t kh = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], g_dimRangeInitVal, g_dimRangeLowerBound,
            g_dimRangeUpperBound);
        size_t kw = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], g_dimRangeInitVal, g_dimRangeLowerBound,
            g_dimRangeUpperBound);

        size_t sh = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], g_dimRangeInitVal, g_dimRangeLowerBound,
            g_dimRangeUpperBound);
        size_t sw = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], g_dimRangeInitVal, g_dimRangeLowerBound,
            g_dimRangeUpperBound);
        size_t dh = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], 0, 0, 2);
        size_t dw = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], 0, 0, 2);
        size_t ph = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], g_dimRangeInitVal, g_dimRangeLowerBound,
            g_dimRangeUpperBound);
        size_t pw = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], g_dimRangeInitVal, g_dimRangeLowerBound,
            g_dimRangeUpperBound);

        size_t dkh = 1 + (kh -1) * (dh + 1);
        size_t dkw = 1 + (kw -1) * (dw + 1);
        int64_t tmpOh = std::ceil(((int64_t)ih - (int64_t)dkh + (int64_t)ph) / (int64_t)sh) + 1;
        int64_t tmpOw = std::ceil(((int64_t)iw - (int64_t)dkw + (int64_t)pw) / (int64_t)sw) + 1;
        size_t oh = static_cast<size_t>(std::max(tmpOh, (int64_t)1));
        size_t ow = static_cast<size_t>(std::max(tmpOw, (int64_t)1));

        bool hasBias =  g_biasMap[hasBiasIndex];
        KuDNN::Element::Type srcType(g_precisionMap[srcElemTypeIndex].second);
        KuDNN::Element::Type weightsType(g_precisionMap[srcElemTypeIndex].second);
        KuDNN::Element::Type dstType(g_precisionMap[dstElemTypeIndex].second);
        KuDNN::Element::Type biasType(g_precisionMap[biasElemTypeIndex].second);
        size_t srcElemSz = srcType.GetSize();
        size_t weightsElemSz = weightsType.GetSize();
        size_t dstElemSz = dstType.GetSize();
        size_t biasElemSz = biasType.GetSize();
        KuDNN::Shape strides(sh, sw);
        KuDNN::Shape dilates(dh, dw);
        KuDNN::Shape paddingL(ph, pw);
        KuDNN::Shape paddingR(0, 0);
        KuDNN::ConvolutionAlgorithm alg(g_algMap[algIndex].second);

        const KuDNN::TensorInfo srcTensor = {{bc, group * ic, ih, iw}, srcType, g_layoutMap[srcLayout]};
        const KuDNN::TensorInfo weightsTensor = {{group, group * oc, group * ic, kh, kw}, weightsType, KuDNN::Layout::ACDEB};
        const KuDNN::TensorInfo dstTensor = {{bc, group * oc, oh, ow}, dstType, g_layoutMap[dstLayout]};
        const KuDNN::TensorInfo biasTensor = {{oc}, biasType, KuDNN::Layout::A};
        const KuDNN::TensorInfo biasTensor2 = {{0}, biasType, KuDNN::Layout::A};

        char format[] = "Forward Convolution: %s, Precision(src:wei:dst:bias): %s:%s:%s:%s"
            ", srcLayout:%d, dstLayout:%d, hasBias:%d, group:%zu. bc=%zu, ic=%zu, ih=%zu, iw=%zu, oc=%zu, kh=%zu, kw=%zu, ph=%zu, pw=%zu, sh=%zu, sw=%zu, dh=%zu, dw=%zu"
            " oh=%zu, ow=%zu ";
        printf(format, g_algMap[algIndex].first.c_str(), g_precisionMap[srcElemTypeIndex].first.c_str(),
            g_precisionMap[srcElemTypeIndex].first.c_str(), g_precisionMap[dstElemTypeIndex].first.c_str(),
            g_precisionMap[biasElemTypeIndex].first.c_str(), srcLayout, dstLayout, hasBias, group, bc, ic, ih, iw, oc, kh, kw, ph, pw, sh, sw, dh, dw,
            oh, ow);
        void *src = nullptr;
        void *weight = nullptr;
        void *bias = nullptr;
        void *dst = nullptr;
        try {
            KuDNN::ConvolutionLayerFWD conv1(srcTensor, weightsTensor, dstTensor, biasTensor, strides, paddingL, paddingR, alg);
            KuDNN::ConvolutionLayerFWD conv2(srcTensor, weightsTensor, dstTensor, biasTensor, strides, dilates, paddingL, paddingR, alg);
            KuDNN::ConvolutionLayerFWD conv3(conv1);
            KuDNN::ConvolutionLayerFWD conv4(std::move(conv3));
            KuDNN::ConvolutionLayerFWD conv5 = conv1;
            KuDNN::ConvolutionLayerFWD conv6 = std::move(conv2);

            KuDNN::ConvolutionLayerFWD convFwdLayer(srcTensor, weightsTensor, dstTensor, biasTensor, strides, dilates,
                paddingL, paddingR, alg);
            KuDNN::ConvolutionLayerFWD convFwdLayerNobias(srcTensor, weightsTensor, dstTensor, biasTensor2, strides,
                dilates, paddingL, paddingR, alg);
            src = malloc(bc * ic * ih * iw * srcElemSz);
            if (src == nullptr) {
                std::cerr << "memory allocation failed" << std::endl;
                return;
            }

            weight = malloc(oc * ic * kh * kw * weightsElemSz);
            if (weight == nullptr) {
                free(src);
                std::cerr << "memory allocation failed" << std::endl;
                return;
            }

            dst = malloc(bc * oc * oh * ow * dstElemSz);
            if (dst == nullptr) {
                free(src);
                free(weight);
                std::cerr << "memory allocation failed" << std::endl;
                return;
            }

            if (hasBias) {
                bias = malloc(oc * biasElemSz);
                if (bias == nullptr) {
                    free(src);
                    free(weight);
                    free(dst);
                    std::cerr << "bias memory allocation failed" << std::endl;
                    return;
                }
            }
            if (hasBias) {
                convFwdLayer.Run(src, weight, dst, bias);
            } else {
                convFwdLayerNobias.Run(src, weight, dst, bias);
            }
            free(src);
            free(weight);
            if (bias != nullptr) {
                free(bias);
            }

            free(dst);
        } catch (const KuDNN::Service::Unsupported& e) {
            if (src != nullptr) {
                free(src);
            }
            if (weight != nullptr) {
                free(weight);
            }
            if (bias != nullptr) {
                free(bias);
            }
            if (dst != nullptr) {
                free(dst);
            }
            printf("Unsupported Configuration");
        } catch (const KuDNN::Service::LogicError& e) {
            if (src != nullptr) {
                free(src);
            }
            if (weight != nullptr) {
                free(weight);
            }
            if (bias != nullptr) {
                free(bias);
            }
            if (dst != nullptr) {
                free(dst);
            }
            printf("Bad shapes' arguments");
        }
        printf("\n");
    }
    DT_FUZZ_END()
}

static void DtFuzzConvolutionBwdData(int counts)
{
    char funcName[] = "DtFuzzConvolutionBwdData";
    DT_FUZZ_START(0, counts, funcName, 0)
    {
        int eleIndex = 0;
        int diffSrcElemTypeIndex = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], 0, 0,
            g_precisionMap.size() - 1);
        int diffDstElemTypeIndex = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], 0, 0,
            g_precisionMap.size() - 1);
        int algIndex = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], 0, 0, g_algMap.size() - 1);
        size_t bc = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], g_dimRangeInitVal, g_dimRangeLowerBound,
            g_dimRangeUpperBound);
        size_t ic = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], g_dimRangeInitVal, g_dimRangeLowerBound,
            g_dimRangeUpperBound);
        size_t ih = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], g_dimRangeInitVal, g_dimRangeLowerBound,
            g_dimRangeUpperBound);
        size_t iw = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], g_dimRangeInitVal, g_dimRangeLowerBound,
            g_dimRangeUpperBound);

        size_t oc = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], g_dimRangeInitVal, g_dimRangeLowerBound,
            g_dimRangeUpperBound);
        size_t kh = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], g_dimRangeInitVal, g_dimRangeLowerBound,
            g_dimRangeUpperBound);
        size_t kw = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], g_dimRangeInitVal, g_dimRangeLowerBound,
            g_dimRangeUpperBound);

        size_t sh = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], g_dimRangeInitVal, g_dimRangeLowerBound,
            g_dimRangeUpperBound);
        size_t sw = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], g_dimRangeInitVal, g_dimRangeLowerBound,
            g_dimRangeUpperBound);
        size_t dh = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], 0, 0, 2);
        size_t dw = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], 0, 0, 2);
        size_t ph = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], g_dimRangeInitVal, g_dimRangeLowerBound,
            g_dimRangeUpperBound);
        size_t pw = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], g_dimRangeInitVal, g_dimRangeLowerBound,
            g_dimRangeUpperBound);

        size_t dkh = 1 + (kh -1) * (dh + 1);
        size_t dkw = 1 + (kw -1) * (dw + 1);
        int64_t tmpOh = std::ceil(((int64_t)ih - (int64_t)dkh + (int64_t)ph) / (int64_t)sh) + 1;
        int64_t tmpOw = std::ceil(((int64_t)iw - (int64_t)dkw + (int64_t)pw) / (int64_t)sw) + 1;
        size_t oh = static_cast<size_t>(std::max(tmpOh, (int64_t)1));
        size_t ow = static_cast<size_t>(std::max(tmpOw, (int64_t)1));

        KuDNN::Element::Type diffSrcType(g_precisionMap[diffSrcElemTypeIndex].second);
        KuDNN::Element::Type weightsType(g_precisionMap[diffDstElemTypeIndex].second);
        KuDNN::Element::Type diffDstType(g_precisionMap[diffDstElemTypeIndex].second);
        size_t diffSrcElemSz = diffSrcType.GetSize();
        size_t weightsElemSz = weightsType.GetSize();
        size_t diffDstElemSz = diffDstType.GetSize();
        KuDNN::Shape strides(sh, sw);
        KuDNN::Shape dilates(dh, dw);
        KuDNN::Shape paddingL(ph, pw);
        KuDNN::Shape paddingR(0, 0);
        KuDNN::ConvolutionAlgorithm alg(g_algMap[algIndex].second);

        const KuDNN::TensorInfo diffSrcTensor = {{bc, ic, ih, iw}, diffSrcType, KuDNN::Layout::ABCD};
        const KuDNN::TensorInfo weightsTensor = {{oc, ic, kh, kw}, weightsType, KuDNN::Layout::ABCD};
        const KuDNN::TensorInfo diffDstTensor = {{bc, oc, oh, ow}, diffDstType, KuDNN::Layout::ABCD};

        char format[] = "Backward Data Convolution: %s, Precision(diffsrc:wei:diffDst): %s:%s:%s"
            ", bc=%zu, ic=%zu, ih=%zu, iw=%zu, oc=%zu, kh=%zu, kw=%zu, ph=%zu, pw=%zu, sh=%zu, sw=%zu, dh=%zu, dw=%zu"
            " oh=%zu, ow=%zu ";

        printf(format, g_algMap[algIndex].first.c_str(), g_precisionMap[diffSrcElemTypeIndex].first.c_str(),
            g_precisionMap[diffDstElemTypeIndex].first.c_str(), g_precisionMap[diffDstElemTypeIndex].first.c_str(),
            bc, ic, ih, iw, oc, kh, kw, ph, pw, sh, sw, dh, dw, oh, ow);

        void *diffSrc = nullptr;
        void *weight = nullptr;
        void *diffDst = nullptr;
        try {
            KuDNN::ConvolutionLayerBWDData convBwd1(diffDstTensor, weightsTensor, diffSrcTensor, strides, dilates,
                paddingL, paddingR, alg);
            KuDNN::ConvolutionLayerBWDData convBwd2(convBwd1);
            convBwd2 = convBwd1;
            KuDNN::ConvolutionLayerBWDData convBwd3(convBwd1);
            convBwd3 = std::move(convBwd1);
            KuDNN::ConvolutionLayerBWDData convBwd5(std::move(convBwd2));

            KuDNN::ConvolutionLayerBWDData convBwdLayer(diffDstTensor, weightsTensor, diffSrcTensor, strides, dilates,
                paddingL, paddingR, alg);
            diffSrc = malloc(bc * ic * ih * iw * diffSrcElemSz);
            if (diffSrc == nullptr) {
                std::cerr << "memory allocation failed" << std::endl;
                return;
            }

            weight = malloc(oc * ic * kh * kw * weightsElemSz);
            if (weight == nullptr) {
                free(diffSrc);
                std::cerr << "memory allocation failed" << std::endl;
                return;
            }

            diffDst = malloc(bc * oc * oh * ow * diffDstElemSz);
            if (diffDst == nullptr) {
                free(diffSrc);
                free(weight);
                std::cerr << "memory allocation failed" << std::endl;
                return;
            }

            convBwdLayer.Run(diffDst, weight, diffSrc);
            free(diffSrc);
            free(weight);
            free(diffDst);
        } catch (const KuDNN::Service::Unsupported& e) {
            if (diffSrc != nullptr) {
                free(diffSrc);
            }
            if (weight != nullptr) {
                free(weight);
            }
            if (diffDst != nullptr) {
                free(diffDst);
            }
            printf("Unsupported Configuration");
        } catch (const KuDNN::Service::LogicError& e) {
            if (diffSrc != nullptr) {
                free(diffSrc);
            }
            if (weight != nullptr) {
                free(weight);
            }
            if (diffDst != nullptr) {
                free(diffDst);
            }
            printf("Bad shapes' arguments");
        }
        printf("\n");
    }
    DT_FUZZ_END()
}

static void DtFuzzConvolutionBwdWeights(int counts)
{
    char funcName[] = "DtFuzzConvolutionBwdWeights";
    DT_FUZZ_START(0, counts, funcName, 0)
    {
        int eleIndex = 0;
        int elemTypeIndex = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], 0, 0,
            g_bwdwMap.size() - 1);
        int algIndex = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], 0, 0, g_algMap.size() - 1);
        int hasBiasIndex = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], 0, 0, g_biasMap.size() - 1);
        size_t bc = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], g_dimRangeInitVal, g_dimRangeLowerBound,
            g_dimRangeUpperBound);
        size_t ic = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], g_dimRangeInitVal, g_dimRangeLowerBound,
            g_dimRangeUpperBound);
        size_t ih = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], g_dimRangeInitVal, g_dimRangeLowerBound,
            g_dimRangeUpperBound);
        size_t iw = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], g_dimRangeInitVal, g_dimRangeLowerBound,
            g_dimRangeUpperBound);

        size_t oc = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], g_dimRangeInitVal, g_dimRangeLowerBound,
            g_dimRangeUpperBound);
        size_t kh = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], g_dimRangeInitVal, g_dimRangeLowerBound,
            g_dimRangeUpperBound);
        size_t kw = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], g_dimRangeInitVal, g_dimRangeLowerBound,
            g_dimRangeUpperBound);

        size_t sh = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], g_dimRangeInitVal, g_dimRangeLowerBound,
            g_dimRangeUpperBound);
        size_t sw = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], g_dimRangeInitVal, g_dimRangeLowerBound,
            g_dimRangeUpperBound);
        size_t dh = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], 0, 0, 2);
        size_t dw = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], 0, 0, 2);
        size_t ph = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], g_dimRangeInitVal, g_dimRangeLowerBound,
            g_dimRangeUpperBound);
        size_t pw = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], g_dimRangeInitVal, g_dimRangeLowerBound,
            g_dimRangeUpperBound);

        size_t dkh = 1 + (kh -1) * (dh + 1);
        size_t dkw = 1 + (kw -1) * (dw + 1);
        int64_t tmpOh = std::ceil(((int64_t)ih - (int64_t)dkh + (int64_t)ph) / (int64_t)sh) + 1;
        int64_t tmpOw = std::ceil(((int64_t)iw - (int64_t)dkw + (int64_t)pw) / (int64_t)sw) + 1;
        size_t oh = static_cast<size_t>(std::max(tmpOh, (int64_t)1));
        size_t ow = static_cast<size_t>(std::max(tmpOw, (int64_t)1));

        bool hasBias =  g_biasMap[hasBiasIndex];
        KuDNN::Element::Type srcType(g_bwdwMap[elemTypeIndex].first);
        KuDNN::Element::Type diffWeightsType(g_bwdwMap[elemTypeIndex].second);
        KuDNN::Element::Type diffDstType(g_bwdwMap[elemTypeIndex].first);
        KuDNN::Element::Type diffBiasType(g_bwdwMap[elemTypeIndex].second);
        size_t srcElemSz = srcType.GetSize();
        size_t diffWeightsElemSz = diffWeightsType.GetSize();
        size_t diffDstElemSz = diffDstType.GetSize();
        size_t diffBiasElemSz = diffBiasType.GetSize();
        KuDNN::Shape strides(sh, sw);
        KuDNN::Shape dilates(dh, dw);
        KuDNN::Shape paddingL(ph, pw);
        KuDNN::Shape paddingR(0, 0);
        KuDNN::ConvolutionAlgorithm alg(g_algMap[algIndex].second);

        const KuDNN::TensorInfo srcTensor = {{bc, ic, ih, iw}, srcType, KuDNN::Layout::ABCD};
        const KuDNN::TensorInfo diffWeightsTensor = {{oc, ic, kh, kw}, diffWeightsType, KuDNN::Layout::ABCD};
        const KuDNN::TensorInfo diffDstTensor = {{bc, oc, oh, ow}, diffDstType, KuDNN::Layout::ABCD};
        const KuDNN::TensorInfo diffBiasTensor = {{oc}, diffBiasType, KuDNN::Layout::A};

        std::vector<std::string> precision = {"F32:F32:F32:F32", "F16:F16:F16:F16", "BF16:BF16:BF16:BF16",
            "F16:F32:F16:F32", "BF16:F32:BF16:F32"};
        char format[] = "Backward Weights Convolution: %s, Precision(src:diffWei:diffDst:diffBias): %s"
            ", bc=%zu, ic=%zu, ih=%zu, iw=%zu, oc=%zu, kh=%zu, kw=%zu, ph=%zu, pw=%zu, sh=%zu, sw=%zu, dh=%zu, dw=%zu"
            " oh=%zu, ow=%zu ";

        printf(format, g_algMap[algIndex].first.c_str(), precision[elemTypeIndex].c_str(),
            bc, ic, ih, iw, oc, kh, kw, ph, pw, sh, sw, dh, dw,
            oh, ow);

        void *src = nullptr;
        void *diffWeights = nullptr;
        void *diffDst = nullptr;
        void *diffBias = nullptr;
        try {
            KuDNN::ConvolutionLayerBWDWeights conv1(diffDstTensor, srcTensor, diffWeightsTensor,
                diffBiasTensor, strides, dilates, paddingL, paddingR, alg);
            KuDNN::ConvolutionLayerBWDWeights conv2(diffDstTensor, srcTensor, diffWeightsTensor,
                diffBiasTensor, strides, paddingL, paddingR, alg);
            KuDNN::ConvolutionLayerBWDWeights conv3(conv1);
            conv3 = conv1;
            KuDNN::ConvolutionLayerBWDWeights conv4(std::move(conv3));
            KuDNN::ConvolutionLayerBWDWeights conv5(conv1);
            conv5 = std::move(conv2);

            KuDNN::ConvolutionLayerBWDWeights convBwdWeightsLayer(diffDstTensor, srcTensor, diffWeightsTensor,
                diffBiasTensor, strides, dilates, paddingL, paddingR, alg);
            src = malloc(bc * ic * ih * iw * srcElemSz);
            if (src == nullptr) {
                std::cerr << "memory allocation failed" << std::endl;
                return;
            }

            diffWeights = malloc(oc * ic * kh * kw * diffWeightsElemSz);
            if (diffWeights == nullptr) {
                free(src);
                std::cerr << "memory allocation failed" << std::endl;
                return;
            }

            diffDst = malloc(bc * oc * oh * ow * diffDstElemSz);
            if (diffDst == nullptr) {
                free(src);
                free(diffWeights);
                std::cerr << "memory allocation failed" << std::endl;
                return;
            }
            if (hasBias) {
                diffBias = malloc(oc * diffBiasElemSz);
                if (diffBias == nullptr) {
                    free(src);
                    free(diffWeights);
                    free(diffDst);
                    std::cerr << "memory allocation failed" << std::endl;
                    return;
                }
            }

            convBwdWeightsLayer.Run(diffDst, src, diffWeights, diffBias);
            free(src);
            free(diffWeights);
            free(diffDst);
            if (diffBias != nullptr) {
                free(diffBias);
            }
        } catch (const KuDNN::Service::Unsupported& e) {
            if (src != nullptr) {
                free(src);
            }
            if (diffWeights != nullptr) {
                free(diffWeights);
            }
            if (diffDst != nullptr) {
                free(diffDst);
            }
            if (diffBias != nullptr) {
                free(diffBias);
            }
            printf("Unsupported Configuration");
        } catch (const KuDNN::Service::LogicError& e) {
            if (src != nullptr) {
                free(src);
            }
            if (diffWeights != nullptr) {
                free(diffWeights);
            }
            if (diffDst != nullptr) {
                free(diffDst);
            }
            if (diffBias != nullptr) {
                free(diffBias);
            }
            printf("Bad shapes' arguments");
        }
        printf("\n");
    }
    DT_FUZZ_END()
}

static void DtFuzzBrgemmConvolutionFwd(int counts)
{
    char funcName[] = "DtFuzzBrgemmConvolutionFwd";
    DT_FUZZ_START(0, counts, funcName, 0)
    {
        size_t ic = 256;
        size_t oc = 58;

        int eleIndex = 0;
        size_t bc = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], g_dimRangeInitVal, g_dimRangeLowerBound,
            g_dimRangeUpperBound);
        size_t id = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], g_dimRangeInitVal, g_dimRangeLowerBound,
            g_dimRangeUpperBound);
        size_t ih = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], g_dimRangeInitVal, g_dimRangeLowerBound,
            g_dimRangeUpperBound);
        size_t iw = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], g_dimRangeInitVal, g_dimRangeLowerBound,
            g_dimRangeUpperBound);

        size_t kd = 1;
        size_t kh = 1;
        size_t kw = 1;

        size_t sd = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], g_dimRangeInitVal, g_dimRangeLowerBound,
            g_dimRangeUpperBound);
        size_t sh = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], g_dimRangeInitVal, g_dimRangeLowerBound,
            g_dimRangeUpperBound);
        size_t sw = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], g_dimRangeInitVal, g_dimRangeLowerBound,
            g_dimRangeUpperBound);

        size_t dd = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], 0, 0, 2);
        size_t dh = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], 0, 0, 2);
        size_t dw = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], 0, 0, 2);

        size_t pd = 0;
        size_t ph = 0;
        size_t pw = 0;
        
        size_t type = 0;

        size_t dkd = 1 + (kd -1) * (dd + 1);
        size_t dkh = 1 + (kh -1) * (dh + 1);
        size_t dkw = 1 + (kw -1) * (dw + 1);
        int64_t tmpOd = std::ceil(((int64_t)id - (int64_t)dkd + (int64_t)pd) / (int64_t)sd) + 1;
        int64_t tmpOh = std::ceil(((int64_t)ih - (int64_t)dkh + (int64_t)ph) / (int64_t)sh) + 1;
        int64_t tmpOw = std::ceil(((int64_t)iw - (int64_t)dkw + (int64_t)pw) / (int64_t)sw) + 1;
        size_t od = static_cast<size_t>(std::max(tmpOd, (int64_t)1));
        size_t oh = static_cast<size_t>(std::max(tmpOh, (int64_t)1));
        size_t ow = static_cast<size_t>(std::max(tmpOw, (int64_t)1));

        KuDNN::Element::Type srcType(g_precisionMap[type].second);
        KuDNN::Element::Type weightsType(g_precisionMap[type].second);
        KuDNN::Element::Type dstType(g_precisionMap[type].second);
        KuDNN::Element::Type biasType(g_precisionMap[type].second);
        size_t srcElemSz = srcType.GetSize();
        size_t weightsElemSz = weightsType.GetSize();
        size_t dstElemSz = dstType.GetSize();
        size_t biasElemSz = biasType.GetSize();

        KuDNN::Shape strides1d(sw);
        KuDNN::Shape dilates1d(dw);
        KuDNN::Shape paddingL1d(pw);
        KuDNN::Shape strides2d(sh, sw);
        KuDNN::Shape dilates2d(dh, dw);
        KuDNN::Shape paddingL2d(ph, pw);
        KuDNN::Shape strides3d(sd, sh, sw);
        KuDNN::Shape dilates3d(dd, dh, dw);
        KuDNN::Shape paddingL3d(pd, ph, pw);
        KuDNN::Propagation prop = KuDNN::Propagation::FORWARD_TRAINING;
#ifdef KUDNNL_920F
        int isa = 0x1f;
#else
        int isa = 0x07;
#endif
        int nthread = 1;
        bool isblock = true;
        bool bUseWino = false;

        const KuDNN::TensorInfo srcTensor1d = {{bc, ic, iw}, srcType, KuDNN::Layout::ACB};
        const KuDNN::TensorInfo weightsTensor1d = {{oc, ic, kw}, weightsType, KuDNN::Layout::ABC};
        const KuDNN::TensorInfo dstTensor1d = {{bc, oc, ow}, dstType, KuDNN::Layout::ACB};
        const KuDNN::TensorInfo biasTensor1d = {{0}, biasType, KuDNN::Layout::A};

        const KuDNN::TensorInfo srcTensor2d = {{bc, ic, ih, iw}, srcType, KuDNN::Layout::ACDB};
        const KuDNN::TensorInfo weightsTensor2d = {{oc, ic, kh, kw}, weightsType, KuDNN::Layout::ABCD};
        const KuDNN::TensorInfo dstTensor2d = {{bc, oc, oh, ow}, dstType, KuDNN::Layout::ACDB};
        const KuDNN::TensorInfo biasTensor2d = {{0}, biasType, KuDNN::Layout::A};

        const KuDNN::TensorInfo srcTensor3d = {{bc, ic, id, ih, iw}, srcType, KuDNN::Layout::ACDEB};
        const KuDNN::TensorInfo weightsTensor3d = {{oc, ic, kd, kh, kw}, weightsType, KuDNN::Layout::ABCDE};
        const KuDNN::TensorInfo dstTensor3d = {{bc, oc, od, oh, ow}, dstType, KuDNN::Layout::ACDEB};
        const KuDNN::TensorInfo biasTensor3d = {{0}, biasType, KuDNN::Layout::A};

        void *src = nullptr;
        void *weight = nullptr;
        void *bias = nullptr;
        void *dst = nullptr;
        try {
            char format[] = "Brgemm Convolution Forward, Precision(src:wei:dst:bias): %s:%s:%s:%s, "
            "bc=%zu, ic=%zu, id=%zu, ih=%zu, iw=%zu, oc=%zu, kd=%zu, kh=%zu, kw=%zu, "
            "pd=%zu, ph=%zu, pw=%zu, sd=%zu, sh=%zu, sw=%zu, dd=%zu, dh=%zu, dw=%zu, "
            "od=%zu, oh=%zu, ow=%zu ";
            printf(format, g_precisionMap[type].first.c_str(), g_precisionMap[type].first.c_str(),
                g_precisionMap[type].first.c_str(), g_precisionMap[type].first.c_str(),
                bc, ic, id, ih, iw, oc, kd, kh, kw, pd, ph, pw, sd, sh, sw, dd, dh, dw, od, oh, ow);
            KuDNN::ConvolutionLayerFWD convFwdLayer1d(isa, srcTensor1d, weightsTensor1d, dstTensor1d, biasTensor1d,
                strides1d, paddingL1d, dilates1d, prop, nthread, isblock, bUseWino);
            KuDNN::ConvolutionLayerFWD convFwdLayer2d(isa, srcTensor2d, weightsTensor2d, dstTensor2d, biasTensor2d,
                strides2d, paddingL2d, dilates2d, prop, nthread, isblock, bUseWino);
            KuDNN::ConvolutionLayerFWD convFwdLayer3d(isa, srcTensor3d, weightsTensor3d, dstTensor3d, biasTensor3d,
                strides3d, paddingL3d, dilates3d, prop, nthread, isblock, bUseWino);

            // 1d execute
           src = malloc(bc * ic * iw * srcElemSz * 2);
           weight = malloc(oc * ic * kw * weightsElemSz * 2);
           dst = malloc(bc * oc * ow * dstElemSz * 2);
           bias = malloc(oc * biasElemSz);
            if (src == nullptr || weight == nullptr || dst == nullptr || bias ==nullptr) {
                std::cerr << "memory allocation failed" << std::endl;
                if (src != nullptr) {
                    free(src);
                }
                if (weight != nullptr) {
                    free(weight);
                }
                if (dst != nullptr) {
                    free(dst);
                }
                if (bias != nullptr) {
                    free(bias);
                }
                return;
            }
            convFwdLayer1d.Run(src, weight, dst, bias);
            free(src);
            free(weight);
            free(bias);
            free(dst);

            // 2d execute
            src = malloc(bc * ic * ih * iw * srcElemSz * 2);
            weight = malloc(oc * ic * kh * kw * weightsElemSz * 2);
            dst = malloc(bc * oc * oh * ow * dstElemSz * 2);
            bias = malloc(oc * biasElemSz);
            if (src == nullptr || weight == nullptr || dst == nullptr || bias ==nullptr) {
                std::cerr << "memory allocation failed" << std::endl;
                if (src != nullptr) {
                    free(src);
                }
                if (weight != nullptr) {
                    free(weight);
                }
                if (dst != nullptr) {
                    free(dst);
                }
                if (bias != nullptr) {
                    free(bias);
                }
                return;
            }
            convFwdLayer2d.Run(src, weight, dst, bias);
            free(src);
            free(weight);
            free(bias);
            free(dst);

            // 3d execute
            src = malloc(bc * id * ic * ih * iw * srcElemSz * 2);
            weight = malloc(oc * ic * kd * kh * kw * weightsElemSz * 2);
            dst = malloc(bc * oc * od * oh * ow * dstElemSz * 2);
            bias = malloc(oc * biasElemSz);
            if (src == nullptr || weight == nullptr || dst == nullptr || bias ==nullptr) {
                std::cerr << "memory allocation failed" << std::endl;
                if (src != nullptr) {
                    free(src);
                }
                if (weight != nullptr) {
                    free(weight);
                }
                if (dst != nullptr) {
                    free(dst);
                }
                if (bias != nullptr) {
                    free(bias);
                }
                return;
            }
            convFwdLayer3d.Run(src, weight, dst, bias);
            free(src);
            free(weight);
            free(bias);
            free(dst);

        } catch (const KuDNN::Service::Unsupported& e) {
            if (src != nullptr) {
                free(src);
            }
            if (weight != nullptr) {
                free(weight);
            }
            if (bias != nullptr) {
                free(bias);
            }
            if (dst != nullptr) {
                free(dst);
            }
            printf("Unsupported Configuration");
        } catch (const KuDNN::Service::LogicError& e) {
            if (src != nullptr) {
                free(src);
            }
            if (weight != nullptr) {
                free(weight);
            }
            if (bias != nullptr) {
                free(bias);
            }
            if (dst != nullptr) {
                free(dst);
            }
            printf("Bad shapes' arguments");
        }
        printf("\n");
    }
    DT_FUZZ_END()
}

static void DtFuzzConvolutionJitFwd(int counts)
{
    char funcName[] = "DtFuzzConvolutionJitFwd";
    DT_FUZZ_START(0, counts, funcName, 0)
    {
        int eleIndex = 0;
        // make sure run in JIT
        size_t ic = 32;
        size_t oc = 32;

        // input args
        size_t bc = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], g_dimRangeInitVal, g_dimRangeLowerBound,
            g_dimRangeUpperBound);
        size_t id = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], g_dimRangeInitVal, g_dimRangeLowerBound,
            g_dimRangeUpperBound);
        size_t ih = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], g_dimRangeInitVal, g_dimRangeLowerBound,
            g_dimRangeUpperBound);
        size_t iw = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], g_dimRangeInitVal, g_dimRangeLowerBound,
            g_dimRangeUpperBound);

        size_t kd = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], g_dimRangeInitVal, g_dimRangeLowerBound,
            g_dimRangeUpperBound);
        size_t kh = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], g_dimRangeInitVal, g_dimRangeLowerBound,
            g_dimRangeUpperBound);
        size_t kw = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], g_dimRangeInitVal, g_dimRangeLowerBound,
            g_dimRangeUpperBound);

        size_t sd = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], g_dimRangeInitVal, g_dimRangeLowerBound,
            g_dimRangeUpperBound);
        size_t sh = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], g_dimRangeInitVal, g_dimRangeLowerBound,
            g_dimRangeUpperBound);
        size_t sw = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], g_dimRangeInitVal, g_dimRangeLowerBound,
            g_dimRangeUpperBound);

        size_t dd = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], 0, 0, 2);
        size_t dh = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], 0, 0, 2);
        size_t dw = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], 0, 0, 2);

        size_t pd = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], g_dimRangeInitVal, g_dimRangeLowerBound,
            g_dimRangeUpperBound);
        size_t ph = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], g_dimRangeInitVal, g_dimRangeLowerBound,
            g_dimRangeUpperBound);
        size_t pw = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], g_dimRangeInitVal, g_dimRangeLowerBound,
            g_dimRangeUpperBound);
        
        size_t type = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], 0, 0, g_precisionMap.size() - 1);

        size_t dkd = 1 + (kd -1) * (dd + 1);
        size_t dkh = 1 + (kh -1) * (dh + 1);
        size_t dkw = 1 + (kw -1) * (dw + 1);
        int64_t tmpOd = std::ceil(((int64_t)id - (int64_t)dkd + (int64_t)pd) / (int64_t)sd) + 1;
        int64_t tmpOh = std::ceil(((int64_t)ih - (int64_t)dkh + (int64_t)ph) / (int64_t)sh) + 1;
        int64_t tmpOw = std::ceil(((int64_t)iw - (int64_t)dkw + (int64_t)pw) / (int64_t)sw) + 1;
        size_t od = static_cast<size_t>(std::max(tmpOd, (int64_t)1));
        size_t oh = static_cast<size_t>(std::max(tmpOh, (int64_t)1));
        size_t ow = static_cast<size_t>(std::max(tmpOw, (int64_t)1));

        KuDNN::Element::Type srcType(g_precisionMap[type].second);
        KuDNN::Element::Type weightsType(g_precisionMap[type].second);
        KuDNN::Element::Type dstType(g_precisionMap[type].second);
        KuDNN::Element::Type biasType(g_precisionMap[type].second);
        size_t srcElemSz = srcType.GetSize();
        size_t weightsElemSz = weightsType.GetSize();
        size_t dstElemSz = dstType.GetSize();
        size_t biasElemSz = biasType.GetSize();

        KuDNN::Shape strides1d(sw);
        KuDNN::Shape dilates1d(dw);
        KuDNN::Shape paddingL1d(pw);
        KuDNN::Shape strides2d(sh, sw);
        KuDNN::Shape dilates2d(dh, dw);
        KuDNN::Shape paddingL2d(ph, pw);
        KuDNN::Shape strides3d(sd, sh, sw);
        KuDNN::Shape dilates3d(dd, dh, dw);
        KuDNN::Shape paddingL3d(pd, ph, pw);
        KuDNN::Propagation prop = KuDNN::Propagation::FORWARD_TRAINING;
#ifdef KUDNNL_920F
        int isa = 0x1f;
#else
        int isa = 0x07;
#endif
        int nthread = 1;
        bool isblock = true;
        bool bUseWino = false;

        const KuDNN::TensorInfo srcTensor1d = {{bc, ic, iw}, srcType, KuDNN::Layout::ACB};
        const KuDNN::TensorInfo weightsTensor1d = {{oc, ic, kw}, weightsType, KuDNN::Layout::ABC};
        const KuDNN::TensorInfo dstTensor1d = {{bc, oc, ow}, dstType, KuDNN::Layout::ACB};
        const KuDNN::TensorInfo biasTensor1d = {{oc}, biasType, KuDNN::Layout::A};

        const KuDNN::TensorInfo srcTensor2d = {{bc, ic, ih, iw}, srcType, KuDNN::Layout::ACDB};
        const KuDNN::TensorInfo weightsTensor2d = {{oc, ic, kh, kw}, weightsType, KuDNN::Layout::ABCD};
        const KuDNN::TensorInfo dstTensor2d = {{bc, oc, oh, ow}, dstType, KuDNN::Layout::ACDB};
        const KuDNN::TensorInfo biasTensor2d = {{oc}, biasType, KuDNN::Layout::A};

        const KuDNN::TensorInfo srcTensor3d = {{bc, ic, id, ih, iw}, srcType, KuDNN::Layout::ACDEB};
        const KuDNN::TensorInfo weightsTensor3d = {{oc, ic, kd, kh, kw}, weightsType, KuDNN::Layout::ABCDE};
        const KuDNN::TensorInfo dstTensor3d = {{bc, oc, od, oh, ow}, dstType, KuDNN::Layout::ACDEB};
        const KuDNN::TensorInfo biasTensor3d = {{oc}, biasType, KuDNN::Layout::A};

        void *src = nullptr;
        void *weight = nullptr;
        void *bias = nullptr;
        void *dst = nullptr;
        try {
            // constructors
            KuDNN::ConvolutionLayerFWD conv1(isa, srcTensor2d, weightsTensor2d, dstTensor2d, biasTensor2d,
                strides2d, paddingL2d, dilates2d, prop, nthread, isblock, bUseWino);
            KuDNN::ConvolutionLayerFWD conv2(isa, srcTensor2d, weightsTensor2d, dstTensor2d, biasTensor2d,
                strides2d, paddingL2d, dilates2d, prop, nthread, isblock, bUseWino);
            KuDNN::ConvolutionLayerFWD conv3(conv1);
            KuDNN::ConvolutionLayerFWD conv4(std::move(conv3));
            conv3 = conv1;
            conv4 = std::move(conv2);

            char format[] = "Forward ConvolutionJIT, Precision(src:wei:dst:bias): %s:%s:%s:%s, "
            "bc=%zu, ic=%zu, id=%zu, ih=%zu, iw=%zu, oc=%zu, kd=%zu, kh=%zu, kw=%zu, "
            "pd=%zu, ph=%zu, pw=%zu, sd=%zu, sh=%zu, sw=%zu, dd=%zu, dh=%zu, dw=%zu, "
            "od=%zu, oh=%zu, ow=%zu ";
            printf(format, g_precisionMap[type].first.c_str(), g_precisionMap[type].first.c_str(),
                g_precisionMap[type].first.c_str(), g_precisionMap[type].first.c_str(),
                bc, ic, id, ih, iw, oc, kd, kh, kw, pd, ph, pw, sd, sh, sw, dd, dh, dw, od, oh, ow);
            KuDNN::ConvolutionLayerFWD convFwdLayer1d(isa, srcTensor1d, weightsTensor1d, dstTensor1d, biasTensor1d,
                strides1d, paddingL1d, dilates1d, prop, nthread, isblock, bUseWino);
            KuDNN::ConvolutionLayerFWD convFwdLayer2d(isa, srcTensor2d, weightsTensor2d, dstTensor2d, biasTensor2d,
                strides2d, paddingL2d, dilates2d, prop, nthread, isblock, bUseWino);
            KuDNN::ConvolutionLayerFWD convFwdLayer3d(isa, srcTensor3d, weightsTensor3d, dstTensor3d, biasTensor3d,
                strides3d, paddingL3d, dilates3d, prop, nthread, isblock, bUseWino);

            // 1d execute
           src = malloc(bc * ic * iw * srcElemSz);
           weight = malloc(oc * ic * kw * weightsElemSz);
           dst = malloc(bc * oc * ow * dstElemSz);
           bias = malloc(oc * biasElemSz);
            if (src == nullptr || weight == nullptr || dst == nullptr || bias ==nullptr) {
                std::cerr << "memory allocation failed" << std::endl;
                if (src != nullptr) {
                    free(src);
                }
                if (weight != nullptr) {
                    free(weight);
                }
                if (dst != nullptr) {
                    free(dst);
                }
                if (bias != nullptr) {
                    free(bias);
                }
                return;
            }
            convFwdLayer1d.Run(src, weight, dst, bias);
            free(src);
            free(weight);
            free(bias);
            free(dst);

            // 2d execute
            src = malloc(bc * ic * ih * iw * srcElemSz);
            weight = malloc(oc * ic * kh * kw * weightsElemSz);
            dst = malloc(bc * oc * oh * ow * dstElemSz);
            bias = malloc(oc * biasElemSz);
            if (src == nullptr || weight == nullptr || dst == nullptr || bias ==nullptr) {
                std::cerr << "memory allocation failed" << std::endl;
                if (src != nullptr) {
                    free(src);
                }
                if (weight != nullptr) {
                    free(weight);
                }
                if (dst != nullptr) {
                    free(dst);
                }
                if (bias != nullptr) {
                    free(bias);
                }
                return;
            }
            convFwdLayer2d.Run(src, weight, dst, bias);
            free(src);
            free(weight);
            free(bias);
            free(dst);

            // 3d execute
            src = malloc(bc * id * ic * ih * iw * srcElemSz);
            weight = malloc(oc * ic * kd * kh * kw * weightsElemSz);
            dst = malloc(bc * oc * od * oh * ow * dstElemSz);
            bias = malloc(oc * biasElemSz);
            if (src == nullptr || weight == nullptr || dst == nullptr || bias ==nullptr) {
                std::cerr << "memory allocation failed" << std::endl;
                if (src != nullptr) {
                    free(src);
                }
                if (weight != nullptr) {
                    free(weight);
                }
                if (dst != nullptr) {
                    free(dst);
                }
                if (bias != nullptr) {
                    free(bias);
                }
                return;
            }
            convFwdLayer3d.Run(src, weight, dst, bias);
            free(src);
            free(weight);
            free(bias);
            free(dst);

        } catch (const KuDNN::Service::Unsupported& e) {
            if (src != nullptr) {
                free(src);
            }
            if (weight != nullptr) {
                free(weight);
            }
            if (bias != nullptr) {
                free(bias);
            }
            if (dst != nullptr) {
                free(dst);
            }
            printf("Unsupported Configuration");
        } catch (const KuDNN::Service::LogicError& e) {
            if (src != nullptr) {
                free(src);
            }
            if (weight != nullptr) {
                free(weight);
            }
            if (bias != nullptr) {
                free(bias);
            }
            if (dst != nullptr) {
                free(dst);
            }
            printf("Bad shapes' arguments");
        }
        printf("\n");
    }
    DT_FUZZ_END()
}

static void DtFuzzConvolutionMCJitFwd(int counts)
{
    char funcName[] = "DtFuzzConvolutionMCJitFwd";
    bool isRunOnece = false;
    DT_FUZZ_START(0, counts, funcName, 0)
    {
        int eleIndex = 0;
        // make sure run in JIT
        size_t ic = 16;
        size_t oc = 16;
        // input args
        size_t type = INDEX0;
        KuDNN::Element::Type srcType(g_precisionMap[type].second);
        KuDNN::Element::Type weightsType(g_precisionMap[type].second);
        KuDNN::Element::Type dstType(g_precisionMap[type].second);
        KuDNN::Element::Type biasType(g_precisionMap[type].second);
        size_t srcElemSz = srcType.GetSize();
        size_t weightsElemSz = weightsType.GetSize();
        size_t dstElemSz = dstType.GetSize();
        size_t biasElemSz = biasType.GetSize();
        KuDNN::Propagation prop = KuDNN::Propagation::FORWARD_TRAINING;

        // input args
        size_t bc = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], g_dimRangeInitVal, g_dimRangeLowerBound,
            g_dimRangeUpperBound);
        size_t ih = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], g_dimRangeInitVal, 1, 5);
        size_t iw = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], g_dimRangeInitVal, 1, 5);

        size_t kh = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], g_dimRangeInitVal, 1, 1);
        size_t kw = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], g_dimRangeInitVal, 1, 1);

        size_t sh = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], g_dimRangeInitVal, 1, 1);
        size_t sw = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], g_dimRangeInitVal, 1, 1);

        size_t dh = 0;
        size_t dw = 0;

        size_t ph = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], g_dimRangeInitVal, 1, 1);
        size_t pw = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], g_dimRangeInitVal, 1, 1);
        
        size_t dkh = 1 + (kh -1) * (dh + 1);
        size_t dkw = 1 + (kw -1) * (dw + 1);
        int64_t tmpOh = std::ceil(((int64_t)ih - (int64_t)dkh + (int64_t)ph) / (int64_t)sh) + 1;
        int64_t tmpOw = std::ceil(((int64_t)iw - (int64_t)dkw + (int64_t)pw) / (int64_t)sw) + 1;
        size_t oh = static_cast<size_t>(std::max(tmpOh, (int64_t)1));
        size_t ow = static_cast<size_t>(std::max(tmpOw, (int64_t)1));

        size_t ih0 = 2, iw0 = 2, kh0 = 1, kw0 = 1, sh0 = 1, sw0 = 1, dh0 = 0, dw0 = 0, ph0 = 1, pw0 = 1;
        size_t ih1 = 2, iw1 = 2, kh1 = 2, kw1 = 2, sh1 = 1, sw1 = 1, dh1 = 0, dw1 = 0, ph1 = 9, pw1 = 9;
        size_t ih2 = 4, iw2 = 4, kh2 = 1, kw2 = 1, sh2 = 2, sw2 = 2, dh2 = 0, dw2 = 0, ph2 = 1, pw2 = 1;
        size_t ih3 = 65, iw3 = 65, kh3 = 3, kw3 = 3, sh3 = 2, sw3 = 2, dh3 = 0, dw3 = 0, ph3 = 0, pw3 = 0;
        size_t ih4 = 24, iw4 = 24, kh4 = 1, kw4 = 1, sh4 = 1, sw4 = 1, dh4 = 0, dw4 = 0, ph4 = 0, pw4 = 0;
        size_t ih5 = 129, iw5 = 129, kh5 = 3, kw5 = 3, sh5 = 2, sw5 = 2, dh5 = 0, dw5 = 0, ph5 = 0, pw5 = 0;
        size_t dkh0 = 1 + (kh0 -1) * (dh0 + 1);
        size_t dkw0 = 1 + (kw0 -1) * (dw0 + 1);

        int64_t tmpOh0 = std::ceil(((int64_t)ih0 - (int64_t)dkh0 + (int64_t)ph0) / (int64_t)sh0) + 1;
        int64_t tmpOw0 = std::ceil(((int64_t)iw0 - (int64_t)dkw0 + (int64_t)pw0) / (int64_t)sw0) + 1;
        size_t oh0 = static_cast<size_t>(std::max(tmpOh0, (int64_t)1));
        size_t ow0 = static_cast<size_t>(std::max(tmpOw0, (int64_t)1));

        size_t dkh1 = 1 + (kh1 -1) * (dh1 + 1);
        size_t dkw1 = 1 + (kw1 -1) * (dw1 + 1);

        int64_t tmpOh1 = std::ceil(((int64_t)ih1 - (int64_t)dkh1 + (int64_t)ph1) / (int64_t)sh1) + 1;
        int64_t tmpOw1 = std::ceil(((int64_t)iw1 - (int64_t)dkw1 + (int64_t)pw1) / (int64_t)sw1) + 1;
        size_t oh1 = static_cast<size_t>(std::max(tmpOh1, (int64_t)1));
        size_t ow1 = static_cast<size_t>(std::max(tmpOw1, (int64_t)1));

        size_t dkh2 = 1 + (kh2 -1) * (dh2 + 1);
        size_t dkw2 = 1 + (kw2 -1) * (dw2 + 1);

        int64_t tmpOh2 = std::ceil(((int64_t)ih2 - (int64_t)dkh2 + (int64_t)ph2) / (int64_t)sh2) + 1;
        int64_t tmpOw2 = std::ceil(((int64_t)iw2 - (int64_t)dkw2 + (int64_t)pw2) / (int64_t)sw2) + 1;
        size_t oh2 = static_cast<size_t>(std::max(tmpOh2, (int64_t)1));
        size_t ow2 = static_cast<size_t>(std::max(tmpOw2, (int64_t)1));

        size_t dkh3 = 1 + (kh3 -1) * (dh3 + 1);
        size_t dkw3 = 1 + (kw3 -1) * (dw3 + 1);

        int64_t tmpOh3 = std::ceil(((int64_t)ih3 - (int64_t)dkh3 + (int64_t)ph3) / (int64_t)sh3) + 1;
        int64_t tmpOw3 = std::ceil(((int64_t)iw3 - (int64_t)dkw3 + (int64_t)pw3) / (int64_t)sw3) + 1;
        size_t oh3 = static_cast<size_t>(std::max(tmpOh3, (int64_t)1));
        size_t ow3 = static_cast<size_t>(std::max(tmpOw3, (int64_t)1));

        size_t dkh4 = 1 + (kh4 -1) * (dh4 + 1);
        size_t dkw4 = 1 + (kw4 -1) * (dw4 + 1);

        int64_t tmpOh4 = std::ceil(((int64_t)ih4 - (int64_t)dkh4 + (int64_t)ph4) / (int64_t)sh4) + 1;
        int64_t tmpOw4 = std::ceil(((int64_t)iw4 - (int64_t)dkw4 + (int64_t)pw4) / (int64_t)sw4) + 1;
        size_t oh4 = static_cast<size_t>(std::max(tmpOh4, (int64_t)1));
        size_t ow4 = static_cast<size_t>(std::max(tmpOw4, (int64_t)1));

        size_t dkh5 = 1 + (kh5 -1) * (dh5 + 1);
        size_t dkw5 = 1 + (kw5 -1) * (dw5 + 1);

        int64_t tmpOh5 = std::ceil(((int64_t)ih5 - (int64_t)dkh5 + (int64_t)ph5) / (int64_t)sh5) + 1;
        int64_t tmpOw5 = std::ceil(((int64_t)iw5 - (int64_t)dkw5 + (int64_t)pw5) / (int64_t)sw5) + 1;
        size_t oh5 = static_cast<size_t>(std::max(tmpOh5, (int64_t)1));
        size_t ow5 = static_cast<size_t>(std::max(tmpOw5, (int64_t)1));

        KuDNN::Shape strides2d(sh, sw);
        KuDNN::Shape dilates2d(dh, dw);
        KuDNN::Shape paddingL2d(ph, pw);

        KuDNN::Shape strides2d0(sh0, sw0);
        KuDNN::Shape dilates2d0(dh0, dw0);
        KuDNN::Shape paddingL2d0(ph0, pw0);

        KuDNN::Shape strides2d1(sh1, sw1);
        KuDNN::Shape dilates2d1(dh1, dw1);
        KuDNN::Shape paddingL2d1(ph1, pw1);

        KuDNN::Shape strides2d2(sh2, sw2);
        KuDNN::Shape dilates2d2(dh2, dw2);
        KuDNN::Shape paddingL2d2(ph2, pw2);

        KuDNN::Shape strides2d3(sh3, sw3);
        KuDNN::Shape dilates2d3(dh3, dw3);
        KuDNN::Shape paddingL2d3(ph3, pw3);

        KuDNN::Shape strides2d4(sh4, sw4);
        KuDNN::Shape dilates2d4(dh4, dw4);
        KuDNN::Shape paddingL2d4(ph4, pw4);

        KuDNN::Shape strides2d5(sh5, sw5);
        KuDNN::Shape dilates2d5(dh5, dw5);
        KuDNN::Shape paddingL2d5(ph5, pw5);
        
#ifdef KUDNNL_920F
        int isa = 0x1f;
#else
        int isa = 0x07;
#endif
        int nthread = 1;
        bool isblock = true;
        bool bUseWino = false;
        const KuDNN::TensorInfo srcTensor2d = {{bc, ic, ih, iw}, srcType, KuDNN::Layout::ABCD};
        const KuDNN::TensorInfo weightsTensor2d = {{oc, ic, kh0, kw}, weightsType, KuDNN::Layout::ABCD};
        const KuDNN::TensorInfo dstTensor2d = {{bc, oc, oh, ow}, dstType, KuDNN::Layout::ABCD};
        const KuDNN::TensorInfo biasTensor2d = {{oc}, biasType, KuDNN::Layout::A};

        const KuDNN::TensorInfo srcTensor2d0 = {{bc, ic, ih0, iw0}, srcType, KuDNN::Layout::ABCD};
        const KuDNN::TensorInfo weightsTensor2d0 = {{oc, ic, kh0, kw0}, weightsType, KuDNN::Layout::ABCD};
        const KuDNN::TensorInfo dstTensor2d0 = {{bc, oc, oh0, ow0}, dstType, KuDNN::Layout::ABCD};
        const KuDNN::TensorInfo biasTensor2d0 = {{oc}, biasType, KuDNN::Layout::A};

        const KuDNN::TensorInfo srcTensor2d1 = {{bc, ic, ih1, iw1}, srcType, KuDNN::Layout::ABCD};
        const KuDNN::TensorInfo weightsTensor2d1 = {{oc, ic, kh1, kw1}, weightsType, KuDNN::Layout::ABCD};
        const KuDNN::TensorInfo dstTensor2d1 = {{bc, oc, oh1, ow1}, dstType, KuDNN::Layout::ABCD};
        const KuDNN::TensorInfo biasTensor2d1 = {{oc}, biasType, KuDNN::Layout::A};

        const KuDNN::TensorInfo srcTensor2d2 = {{bc, ic, ih2, iw2}, srcType, KuDNN::Layout::ABCD};
        const KuDNN::TensorInfo weightsTensor2d2 = {{oc, ic, kh2, kw2}, weightsType, KuDNN::Layout::ABCD};
        const KuDNN::TensorInfo dstTensor2d2 = {{bc, oc, oh2, ow2}, dstType, KuDNN::Layout::ABCD};
        const KuDNN::TensorInfo biasTensor2d2 = {{oc}, biasType, KuDNN::Layout::A};

        const KuDNN::TensorInfo srcTensor2d3 = {{bc, ic, ih3, iw3}, srcType, KuDNN::Layout::ABCD};
        const KuDNN::TensorInfo weightsTensor2d3 = {{oc, ic, kh3, kw3}, weightsType, KuDNN::Layout::ABCD};
        const KuDNN::TensorInfo dstTensor2d3 = {{bc, oc, oh3, ow3}, dstType, KuDNN::Layout::ABCD};
        const KuDNN::TensorInfo biasTensor2d3 = {{oc}, biasType, KuDNN::Layout::A};

        const KuDNN::TensorInfo srcTensor2d4 = {{bc, ic, ih4, iw4}, srcType, KuDNN::Layout::ABCD};
        const KuDNN::TensorInfo weightsTensor2d4 = {{oc, ic, kh4, kw4}, weightsType, KuDNN::Layout::ABCD};
        const KuDNN::TensorInfo dstTensor2d4 = {{bc, oc, oh4, ow4}, dstType, KuDNN::Layout::ABCD};
        const KuDNN::TensorInfo biasTensor2d4 = {{oc}, biasType, KuDNN::Layout::A};

        const KuDNN::TensorInfo srcTensor2d5 = {{bc, ic, ih5, iw5}, srcType, KuDNN::Layout::ABCD};
        const KuDNN::TensorInfo weightsTensor2d5 = {{oc, ic, kh5, kw5}, weightsType, KuDNN::Layout::ABCD};
        const KuDNN::TensorInfo dstTensor2d5 = {{bc, oc, oh5, ow5}, dstType, KuDNN::Layout::ABCD};
        const KuDNN::TensorInfo biasTensor2d5 = {{oc}, biasType, KuDNN::Layout::A};

        void *src = nullptr;
        void *weight = nullptr;
        void *bias = nullptr;
        void *dst = nullptr;
        try {
            char format[] = "Forward DtFuzzConvolutionMCJitFwd, Precision(src:wei:dst:bias): %s:%s:%s:%s, "
            "bc=%zu, ic=%zu, ih=%zu, iw=%zu, oc=%zu, kh=%zu, kw=%zu, "
            "ph=%zu, pw=%zu, sh=%zu, sw=%zu, dh=%zu, dw=%zu, "
            "oh=%zu, ow=%zu ";
            printf(format, g_precisionMap[type].first.c_str(), g_precisionMap[type].first.c_str(),
                g_precisionMap[type].first.c_str(), g_precisionMap[type].first.c_str(),
                bc, ic, ih, iw, oc, kh, kw, ph, pw, sh, sw, dh, dw, oh, ow);
            KuDNN::ConvolutionLayerFWD convFwdLayer2d(isa, srcTensor2d, weightsTensor2d, dstTensor2d, biasTensor2d,
                strides2d, paddingL2d, dilates2d, prop, nthread, isblock, bUseWino);
            
            KuDNN::ConvolutionLayerFWD convFwdLayer2d0(isa, srcTensor2d0, weightsTensor2d0, dstTensor2d0, biasTensor2d0,
                strides2d0, paddingL2d0, dilates2d0, prop, nthread, isblock, bUseWino);
            
            KuDNN::ConvolutionLayerFWD convFwdLayer2d1(isa, srcTensor2d1, weightsTensor2d1, dstTensor2d1, biasTensor2d1,
                strides2d1, paddingL2d1, dilates2d1, prop, nthread, isblock, bUseWino);
            
            KuDNN::ConvolutionLayerFWD convFwdLayer2d2(isa, srcTensor2d2, weightsTensor2d2, dstTensor2d2, biasTensor2d2,
                strides2d2, paddingL2d2, dilates2d2, prop, nthread, isblock, bUseWino);
            
            KuDNN::ConvolutionLayerFWD convFwdLayer2d3(isa, srcTensor2d3, weightsTensor2d3, dstTensor2d3, biasTensor2d3,
                strides2d3, paddingL2d3, dilates2d3, prop, nthread, isblock, bUseWino);
            
            KuDNN::ConvolutionLayerFWD convFwdLayer2d4(isa, srcTensor2d4, weightsTensor2d4, dstTensor2d4, biasTensor2d4,
                strides2d4, paddingL2d4, dilates2d4, prop, nthread, isblock, bUseWino);

            KuDNN::ConvolutionLayerFWD convFwdLayer2d5(isa, srcTensor2d5, weightsTensor2d5, dstTensor2d5, biasTensor2d5,
                strides2d5, paddingL2d5, dilates2d5, prop, nthread, isblock, bUseWino);
            // 2d execute
            src = malloc(bc * ic * ih * iw * srcElemSz);
            weight = malloc(oc * ic * kh * kw * weightsElemSz);
            dst = malloc(bc * oc * oh * ow * dstElemSz);
            bias = malloc(oc * biasElemSz);
            if (src == nullptr || weight == nullptr || dst == nullptr || bias ==nullptr) {
                std::cerr << "memory allocation failed" << std::endl;
                if (src != nullptr) {
                    free(src);
                }
                if (weight != nullptr) {
                    free(weight);
                }
                if (dst != nullptr) {
                    free(dst);
                }
                if (bias != nullptr) {
                    free(bias);
                }
                return;
            }
            convFwdLayer2d.Run(src, weight, dst, bias);
            free(src);
            free(weight);
            free(bias);
            free(dst);
            if (!isRunOnece) {
                src = malloc(bc * ic * ih0 * iw0 * srcElemSz);
                weight = malloc(oc * ic * kh0 * kw0 * weightsElemSz);
                dst = malloc(bc * oc * oh0 * ow0 * dstElemSz);
                bias = malloc(oc * biasElemSz);
                if (src == nullptr || weight == nullptr || dst == nullptr || bias ==nullptr) {
                    std::cerr << "memory allocation failed" << std::endl;
                    if (src != nullptr) {
                        free(src);
                    }
                    if (weight != nullptr) {
                        free(weight);
                    }
                    if (dst != nullptr) {
                        free(dst);
                    }
                    if (bias != nullptr) {
                        free(bias);
                    }
                    return;
                }
                convFwdLayer2d0.Run(src, weight, dst, bias);
                free(src);
                free(weight);
                free(bias);
                free(dst);

                src = malloc(bc * ic * ih1 * iw1 * srcElemSz);
                weight = malloc(oc * ic * kh1 * kw1 * weightsElemSz);
                dst = malloc(bc * oc * oh1 * ow1 * dstElemSz);
                bias = malloc(oc * biasElemSz);
                if (src == nullptr || weight == nullptr || dst == nullptr || bias ==nullptr) {
                    std::cerr << "memory allocation failed" << std::endl;
                    if (src != nullptr) {
                        free(src);
                    }
                    if (weight != nullptr) {
                        free(weight);
                    }
                    if (dst != nullptr) {
                        free(dst);
                    }
                    if (bias != nullptr) {
                        free(bias);
                    }
                    return;
                }
                convFwdLayer2d1.Run(src, weight, dst, bias);
                free(src);
                free(weight);
                free(bias);
                free(dst);

                src = malloc(bc * ic * ih2 * iw2 * srcElemSz);
                weight = malloc(oc * ic * kh2 * kw2 * weightsElemSz);
                dst = malloc(bc * oc * oh2 * ow2 * dstElemSz);
                bias = malloc(oc * biasElemSz);
                if (src == nullptr || weight == nullptr || dst == nullptr || bias ==nullptr) {
                    std::cerr << "memory allocation failed" << std::endl;
                    if (src != nullptr) {
                        free(src);
                    }
                    if (weight != nullptr) {
                        free(weight);
                    }
                    if (dst != nullptr) {
                        free(dst);
                    }
                    if (bias != nullptr) {
                        free(bias);
                    }
                    return;
                }
                convFwdLayer2d2.Run(src, weight, dst, bias);
                free(src);
                free(weight);
                free(bias);
                free(dst);

                src = malloc(bc * ic * ih3 * iw3 * srcElemSz);
                weight = malloc(oc * ic * kh3 * kw3 * weightsElemSz);
                dst = malloc(bc * oc * oh3 * ow3 * dstElemSz);
                bias = malloc(oc * biasElemSz);
                if (src == nullptr || weight == nullptr || dst == nullptr || bias ==nullptr) {
                    std::cerr << "memory allocation failed" << std::endl;
                    if (src != nullptr) {
                        free(src);
                    }
                    if (weight != nullptr) {
                        free(weight);
                    }
                    if (dst != nullptr) {
                        free(dst);
                    }
                    if (bias != nullptr) {
                        free(bias);
                    }
                    return;
                }
                convFwdLayer2d3.Run(src, weight, dst, bias);
                free(src);
                free(weight);
                free(bias);
                free(dst);

                src = malloc(bc * ic * ih4 * iw4 * srcElemSz);
                weight = malloc(oc * ic * kh4 * kw4 * weightsElemSz);
                dst = malloc(bc * oc * oh4 * ow4 * dstElemSz);
                bias = malloc(oc * biasElemSz);
                if (src == nullptr || weight == nullptr || dst == nullptr || bias ==nullptr) {
                    std::cerr << "memory allocation failed" << std::endl;
                    if (src != nullptr) {
                        free(src);
                    }
                    if (weight != nullptr) {
                        free(weight);
                    }
                    if (dst != nullptr) {
                        free(dst);
                    }
                    if (bias != nullptr) {
                        free(bias);
                    }
                    return;
                }
                convFwdLayer2d4.Run(src, weight, dst, bias);
                free(src);
                free(weight);
                free(bias);
                free(dst);

                src = malloc(bc * ic * ih5 * iw5 * srcElemSz);
                weight = malloc(oc * ic * kh5 * kw5 * weightsElemSz);
                dst = malloc(bc * oc * oh5 * ow5 * dstElemSz);
                bias = malloc(oc * biasElemSz);
                if (src == nullptr || weight == nullptr || dst == nullptr || bias ==nullptr) {
                    std::cerr << "memory allocation failed" << std::endl;
                    if (src != nullptr) {
                        free(src);
                    }
                    if (weight != nullptr) {
                        free(weight);
                    }
                    if (dst != nullptr) {
                        free(dst);
                    }
                    if (bias != nullptr) {
                        free(bias);
                    }
                    return;
                }
                convFwdLayer2d5.Run(src, weight, dst, bias);
                free(src);
                free(weight);
                free(bias);
                free(dst);
            } else {
                isRunOnece = true;
            }
        } catch (const KuDNN::Service::Unsupported& e) {
            if (src != nullptr) {
                free(src);
            }
            if (weight != nullptr) {
                free(weight);
            }
            if (bias != nullptr) {
                free(bias);
            }
            if (dst != nullptr) {
                free(dst);
            }
            printf("Unsupported Configuration");
        } catch (const KuDNN::Service::LogicError& e) {
            if (src != nullptr) {
                free(src);
            }
            if (weight != nullptr) {
                free(weight);
            }
            if (bias != nullptr) {
                free(bias);
            }
            if (dst != nullptr) {
                free(dst);
            }
            printf("Bad shapes' arguments");
        }
        printf("\n");
    }
    DT_FUZZ_END()
}

static void DtFuzzConvolutionDwFwd(int counts)
{
    char funcName[] = "DtFuzzConvolutionDwFwd";
    bool isRunOnece = false;
    DT_FUZZ_START(0, counts, funcName, 0)
    {
        int eleIndex = 0;
        size_t ch = 32;

        // input args
        size_t bc = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], g_dimRangeInitVal, g_dimRangeLowerBound,
            g_dimRangeUpperBound);
        size_t ih = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], g_dimRangeInitVal, g_dimRangeLowerBound,
            g_dimRangeUpperBound);
        size_t iw = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], g_dimRangeInitVal, g_dimRangeLowerBound,
            g_dimRangeUpperBound);

        size_t kh = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], g_dimRangeInitVal, g_dimRangeLowerBound,
            g_dimRangeUpperBound);
        size_t kw = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], g_dimRangeInitVal, g_dimRangeLowerBound,
            g_dimRangeUpperBound);

        size_t sh = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], g_dimRangeInitVal, g_dimRangeLowerBound,
            g_dimRangeUpperBound);
        size_t sw = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], g_dimRangeInitVal, g_dimRangeLowerBound,
            g_dimRangeUpperBound);

        size_t dh = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], 0, 0, 2);
        size_t dw = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], 0, 0, 2);

        size_t ph = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], g_dimRangeInitVal, g_dimRangeLowerBound,
            g_dimRangeUpperBound);
        size_t pw = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], g_dimRangeInitVal, g_dimRangeLowerBound,
            g_dimRangeUpperBound);

        size_t type = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], 0, 0, g_precisionMap.size() - 1);
        size_t nthread = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], 1, 1, 4);

        size_t dkh = 1 + (kh -1) * (dh + 1);
        size_t dkw = 1 + (kw -1) * (dw + 1);
        int64_t tmpOh = std::ceil(((int64_t)ih - (int64_t)dkh + (int64_t)ph) / (int64_t)sh) + 1;
        int64_t tmpOw = std::ceil(((int64_t)iw - (int64_t)dkw + (int64_t)pw) / (int64_t)sw) + 1;
        size_t oh = static_cast<size_t>(std::max(tmpOh, (int64_t)1));
        size_t ow = static_cast<size_t>(std::max(tmpOw, (int64_t)1));

        KuDNN::Element::Type srcType(g_precisionMap[type].second);
        KuDNN::Element::Type weightsType(g_precisionMap[type].second);
        KuDNN::Element::Type dstType(g_precisionMap[type].second);
        KuDNN::Element::Type biasType(g_precisionMap[type].second);

        size_t srcElemSz = srcType.GetSize();
        size_t weightsElemSz = weightsType.GetSize();
        size_t dstElemSz = dstType.GetSize();
        size_t biasElemSz = biasType.GetSize();

        KuDNN::Shape strides2d(sh, sw);
        KuDNN::Shape dilates2d(dh, dw);
        KuDNN::Shape paddingL2d(ph, pw);
        KuDNN::Propagation prop = KuDNN::Propagation::FORWARD_TRAINING;
#ifdef KUDNNL_920F
        int isa = 0x1f;
#else
        int isa = 0x07;
#endif
        size_t ih0 = 3, iw0 = 3, kh0 = 2, kw0 = 2, sh0 = 1, sw0 = 1, dh0 = 0, dw0 = 0, ph0 = 0, pw0 = 0;
        const KuDNN::TensorInfo srcTensor2d = {{bc, ch, ih, iw}, srcType, KuDNN::Layout::ACDB};
        const KuDNN::TensorInfo weightsTensor2d = {{ch, ch, ch, kh, kw}, weightsType, KuDNN::Layout::ABCDE};
        const KuDNN::TensorInfo dstTensor2d = {{bc, ch, oh, ow}, dstType, KuDNN::Layout::ACDB};
        const KuDNN::TensorInfo biasTensor2d = {{ch}, biasType, KuDNN::Layout::A};

        size_t dkh0 = 1 + (kh0 -1) * (dh0 + 1);
        size_t dkw0 = 1 + (kw0 -1) * (dw0 + 1);
        int64_t tmpOh0 = std::ceil(((int64_t)ih0 - (int64_t)dkh0 + (int64_t)ph0) / (int64_t)sh0) + 1;
        int64_t tmpOw0 = std::ceil(((int64_t)iw0 - (int64_t)dkw0 + (int64_t)pw0) / (int64_t)sw0) + 1;
        size_t oh0 = static_cast<size_t>(std::max(tmpOh0, (int64_t)1));
        size_t ow0 = static_cast<size_t>(std::max(tmpOw0, (int64_t)1));

        KuDNN::Element::Type srcType0(g_precisionMap[INDEX0].second);
        KuDNN::Element::Type weightsType0(g_precisionMap[INDEX0].second);
        KuDNN::Element::Type dstType0(g_precisionMap[INDEX0].second);
        KuDNN::Element::Type biasType0(g_precisionMap[INDEX0].second);

        size_t srcElemSz0 = srcType0.GetSize();
        size_t weightsElemSz0 = weightsType0.GetSize();
        size_t dstElemSz0 = dstType0.GetSize();
        size_t biasElemSz0 = biasType0.GetSize();

        KuDNN::Shape strides2d0(sh0, sw0);
        KuDNN::Shape dilates2d0(dh0, dw0);
        KuDNN::Shape paddingL2d0(ph0, pw0);

        const KuDNN::TensorInfo srcTensor2d0 = {{bc, ch, ih0, iw0}, srcType, KuDNN::Layout::ABCD};
        const KuDNN::TensorInfo weightsTensor2d0 = {{ch, ch, ch, kh0, kw0}, weightsType, KuDNN::Layout::ABCDE};
        const KuDNN::TensorInfo dstTensor2d0 = {{bc, ch, oh0, ow0}, dstType, KuDNN::Layout::ABCD};
        const KuDNN::TensorInfo biasTensor2d0 = {{ch}, biasType, KuDNN::Layout::A};

        void *src = nullptr;
        void *weight = nullptr;
        void *bias = nullptr;
        void *dst = nullptr;
        try {
            char format[] = "Forward ConvolutionDW, Precision(src:wei:dst:bias): %s:%s:%s:%s, "
            "bc=%zu, group=%zu, ic=%zu, ih=%zu, iw=%zu, oc=%zu, kh=%zu, kw=%zu, "
            "ph=%zu, pw=%zu, sh=%zu, sw=%zu, dh=%zu, dw=%zu, oh=%zu, ow=%zu, nthread=%zu ";
            printf(format, g_precisionMap[type].first.c_str(), g_precisionMap[type].first.c_str(),
                g_precisionMap[type].first.c_str(), g_precisionMap[type].first.c_str(),
                bc, ch, ch, ih, iw, ch, kh, kw, ph, pw, sh, sw, dh, dw, oh, ow, nthread);
            KuDNN::ConvolutionLayerFWD convFwdLayer(isa, srcTensor2d, weightsTensor2d, dstTensor2d, biasTensor2d,
                strides2d, paddingL2d, dilates2d, prop, nthread, true, false);
            KuDNN::ConvolutionLayerFWD convFwdLayer0(isa, srcTensor2d0, weightsTensor2d0, dstTensor2d0, biasTensor2d0,
                strides2d0, paddingL2d0, dilates2d0, prop, nthread, true, false);

            src = malloc(bc * ch * ih * iw * srcElemSz);
            weight = malloc(ch * ch * kh * kw * weightsElemSz);
            dst = malloc(bc * ch * oh * ow * dstElemSz);
            bias = malloc(ch * biasElemSz);
            if (src == nullptr || weight == nullptr || dst == nullptr || bias ==nullptr) {
                std::cerr << "memory allocation failed" << std::endl;
                if (src != nullptr) {
                    free(src);
                }
                if (weight != nullptr) {
                    free(weight);
                }
                if (dst != nullptr) {
                    free(dst);
                }
                if (bias != nullptr) {
                    free(bias);
                }
                return;
            }

            convFwdLayer.Run(src, weight, dst, bias);
            free(src);
            free(weight);
            free(bias);
            free(dst);
            if (!isRunOnece) {
                src = malloc(bc * ch * ih0 * iw0 * srcElemSz0);
                weight = malloc(ch * ch * kh0 * kw0 * weightsElemSz0);
                dst = malloc(bc * ch * oh0 * ow0 * dstElemSz0);
                bias = malloc(ch * biasElemSz0);
                if (src == nullptr || weight == nullptr || dst == nullptr || bias ==nullptr) {
                    std::cerr << "memory allocation failed" << std::endl;
                    if (src != nullptr) {
                        free(src);
                    }
                    if (weight != nullptr) {
                        free(weight);
                    }
                    if (dst != nullptr) {
                        free(dst);
                    }
                    if (bias != nullptr) {
                        free(bias);
                    }
                    return;
                }
                convFwdLayer0.Run(src, weight, dst, bias);
                free(src);
                free(weight);
                free(bias);
                free(dst);
            } else {
                isRunOnece = true;
            }
        } catch (const KuDNN::Service::Unsupported& e) {
            if (src != nullptr) {
                free(src);
            }
            if (weight != nullptr) {
                free(weight);
            }
            if (bias != nullptr) {
                free(bias);
            }
            if (dst != nullptr) {
                free(dst);
            }
            printf("Unsupported Configuration");
        } catch (const KuDNN::Service::LogicError& e) {
            if (src != nullptr) {
                free(src);
            }
            if (weight != nullptr) {
                free(weight);
            }
            if (bias != nullptr) {
                free(bias);
            }
            if (dst != nullptr) {
                free(dst);
            }
            printf("Bad shapes' arguments");
        }
        printf("\n");
    }
    DT_FUZZ_END()
}

static void DtFuzzConvolutionGroupFwd(int counts)
{
    char funcName[] = "DtFuzzConvolutionGroupFwd";
    DT_FUZZ_START(0, counts, funcName, 0)
    {
        int eleIndex = 0;

        // input args
        size_t bc = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], g_dimRangeInitVal, g_dimRangeLowerBound,
            g_dimRangeUpperBound);
        size_t ih = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], g_dimRangeInitVal, g_dimRangeLowerBound,
            g_dimRangeUpperBound);
        size_t iw = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], g_dimRangeInitVal, g_dimRangeLowerBound,
            g_dimRangeUpperBound);

        size_t kh = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], g_dimRangeInitVal, g_dimRangeLowerBound,
            g_dimRangeUpperBound);
        size_t kw = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], g_dimRangeInitVal, g_dimRangeLowerBound,
            g_dimRangeUpperBound);

        size_t sh = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], g_dimRangeInitVal, g_dimRangeLowerBound,
            g_dimRangeUpperBound);
        size_t sw = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], g_dimRangeInitVal, g_dimRangeLowerBound,
            g_dimRangeUpperBound);

        size_t dh = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], 0, 0, 2);
        size_t dw = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], 0, 0, 2);

        size_t ph = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], g_dimRangeInitVal, g_dimRangeLowerBound,
            g_dimRangeUpperBound);
        size_t pw = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], g_dimRangeInitVal, g_dimRangeLowerBound,
            g_dimRangeUpperBound);

        size_t nthread = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], 1, 1, 4);
        size_t group = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], 2, 2, 4);
        size_t chIdx = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], 1, 1, 5);
        size_t ic = group * chIdx * 4;
        size_t oc = group * chIdx * 4;

        size_t dkh = 1 + (kh -1) * (dh + 1);
        size_t dkw = 1 + (kw -1) * (dw + 1);
        int64_t tmpOh = std::ceil(((int64_t)ih - (int64_t)dkh + (int64_t)ph) / (int64_t)sh) + 1;
        int64_t tmpOw = std::ceil(((int64_t)iw - (int64_t)dkw + (int64_t)pw) / (int64_t)sw) + 1;
        size_t oh = static_cast<size_t>(std::max(tmpOh, (int64_t)1));
        size_t ow = static_cast<size_t>(std::max(tmpOw, (int64_t)1));

        KuDNN::Element::Type srcType(KuDNN::Element::TypeT::F32);
        KuDNN::Element::Type weightsType(KuDNN::Element::TypeT::F32);
        KuDNN::Element::Type dstType(KuDNN::Element::TypeT::F32);
        KuDNN::Element::Type biasType(KuDNN::Element::TypeT::F32);

        size_t srcElemSz = srcType.GetSize();
        size_t weightsElemSz = weightsType.GetSize();
        size_t dstElemSz = dstType.GetSize();
        size_t biasElemSz = biasType.GetSize();

        KuDNN::Shape strides2d(sh, sw);
        KuDNN::Shape dilates2d(dh, dw);
        KuDNN::Shape paddingL2d(ph, pw);
        KuDNN::Propagation prop = KuDNN::Propagation::FORWARD_TRAINING;
#ifdef KUDNNL_920F
        int isa = 0x1f;
#else
        int isa = 0x07;
#endif

        const KuDNN::TensorInfo srcTensor2d = {{bc, ic, ih, iw}, srcType, KuDNN::Layout::ABCD};
        const KuDNN::TensorInfo weightsTensor2d = {{group, oc, ic, kh, kw}, weightsType, KuDNN::Layout::ABCDE};
        const KuDNN::TensorInfo dstTensor2d = {{bc, oc, oh, ow}, dstType, KuDNN::Layout::ABCD};
        const KuDNN::TensorInfo biasTensor2d = {{oc}, biasType, KuDNN::Layout::A};

        void *src = nullptr;
        void *weight = nullptr;
        void *bias = nullptr;
        void *dst = nullptr;
        try {

            char format[] = "Forward ConvolutionGroupFwd, Precision(src:wei:dst:bias): %s:%s:%s:%s, "
            "bc=%zu, group=%zu, ic=%zu, ih=%zu, iw=%zu, oc=%zu, kh=%zu, kw=%zu, "
            "ph=%zu, pw=%zu, sh=%zu, sw=%zu, dh=%zu, dw=%zu, oh=%zu, ow=%zu, nthread=%zu ";
            printf(format, g_precisionMap[0].first.c_str(), g_precisionMap[0].first.c_str(),
                g_precisionMap[0].first.c_str(), g_precisionMap[0].first.c_str(),
                bc, group, ic, ih, iw, oc, kh, kw, ph, pw, sh, sw, dh, dw, oh, ow, nthread);
            KuDNN::ConvolutionLayerFWD convFwdLayer(isa, srcTensor2d, weightsTensor2d, dstTensor2d, biasTensor2d,
                strides2d, paddingL2d, dilates2d, prop, nthread, true, false);

            src = malloc(bc * ic * ih * iw * srcElemSz);
            weight = malloc(oc * ic * kh * kw * weightsElemSz);
            dst = malloc(bc * oc * oh * ow * dstElemSz);
            bias = malloc(oc * biasElemSz);
            if (src == nullptr || weight == nullptr || dst == nullptr || bias == nullptr) {
                std::cerr << "memory allocation failed" << std::endl;
                if (src != nullptr) {
                    free(src);
                }
                if (weight != nullptr) {
                    free(weight);
                }
                if (dst != nullptr) {
                    free(dst);
                }
                if (bias != nullptr) {
                    free(bias);
                }
                return;
            }

            convFwdLayer.Run(src, weight, dst, bias);
            free(src);
            free(weight);
            free(bias);
            free(dst);

        } catch (const KuDNN::Service::Unsupported& e) {
            if (src != nullptr) {
                free(src);
            }
            if (weight != nullptr) {
                free(weight);
            }
            if (bias != nullptr) {
                free(bias);
            }
            if (dst != nullptr) {
                free(dst);
            }
            printf("Unsupported Configuration");
        } catch (const KuDNN::Service::LogicError& e) {
            if (src != nullptr) {
                free(src);
            }
            if (weight != nullptr) {
                free(weight);
            }
            if (bias != nullptr) {
                free(bias);
            }
            if (dst != nullptr) {
                free(dst);
            }
            printf("Bad shapes' arguments");
        }
        printf("\n");
    }
    DT_FUZZ_END()
}

#if defined(KUDNNL_920B) || defined(KUDNNL_920G)
static void DtFuzzConvolutionWinoFwd(int counts)
{
    char funcName[] = "DtFuzzConvolutionWinoFwd";
    DT_FUZZ_START(0, counts, funcName, 0)
    {
        int eleIndex = 0;
        // make sure run in JIT
        size_t ic = 32;
        size_t oc = 32;

        // input args
        size_t bc = 1;
        size_t ih = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], g_dimRangeInitVal, g_dimRangeLowerBound,
            g_dimRangeUpperBound);
        size_t iw = ih;

        size_t kh = 3;
        size_t kw = 3; 

        size_t sh = 1;
        size_t sw = 1;

        size_t dh = 0;
        size_t dw = 0;

        size_t ph = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], g_dimRangeInitVal, g_dimRangeLowerBound,
            g_dimRangeUpperBound);
        size_t pw = ph;
        
        size_t type = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], 0, 0, g_precisionMap.size() - 1);

        size_t dkh = 1 + (kh -1) * (dh + 1);
        int64_t tmpOh = std::ceil(((int64_t)ih - (int64_t)dkh + (int64_t)ph) / (int64_t)sh) + 1;
        size_t oh = static_cast<size_t>(std::max(tmpOh, (int64_t)1));
        size_t ow = oh;

        KuDNN::Element::Type srcType(KuDNN::Element::TypeT::F32);
        KuDNN::Element::Type weightsType(KuDNN::Element::TypeT::F32);
        KuDNN::Element::Type dstType(KuDNN::Element::TypeT::F32);
        KuDNN::Element::Type biasType(KuDNN::Element::TypeT::F32);
        size_t srcElemSz = srcType.GetSize();
        size_t weightsElemSz = weightsType.GetSize();
        size_t dstElemSz = dstType.GetSize();
        size_t biasElemSz = biasType.GetSize();

        KuDNN::Shape strides2d(sh, sw);
        KuDNN::Shape dilates2d(dh, dw);
        KuDNN::Shape paddingL2d(ph, pw);
        KuDNN::Propagation prop = KuDNN::Propagation::FORWARD_TRAINING;

        int isa = 0x07;

        int nthread = 1;
        bool isblock = true;
        bool bUseWino = true;

        const KuDNN::TensorInfo srcTensor2d = {{bc, ic, ih, iw}, srcType, KuDNN::Layout::ACDB};
        const KuDNN::TensorInfo weightsTensor2d = {{oc, ic, kh, kw}, weightsType, KuDNN::Layout::ABCD};
        const KuDNN::TensorInfo dstTensor2d = {{bc, oc, oh, ow}, dstType, KuDNN::Layout::ACDB};
        const KuDNN::TensorInfo biasTensor2d = {{oc}, biasType, KuDNN::Layout::A};

        void *src = nullptr;
        void *weight = nullptr;
        void *bias = nullptr;
        void *dst = nullptr;
        try {
            // constructors
            KuDNN::ConvolutionLayerFWD conv1(isa, srcTensor2d, weightsTensor2d, dstTensor2d, biasTensor2d,
                strides2d, paddingL2d, dilates2d, prop, nthread, isblock, bUseWino);
            KuDNN::ConvolutionLayerFWD conv2(isa, srcTensor2d, weightsTensor2d, dstTensor2d, biasTensor2d,
                strides2d, paddingL2d, dilates2d, prop, nthread, isblock, bUseWino);
            KuDNN::ConvolutionLayerFWD conv3(conv1);
            KuDNN::ConvolutionLayerFWD conv4(std::move(conv3));
            conv3 = conv1;
            conv4 = std::move(conv2);

            char format[] = "Forward Wino Convolution, Precision(src:wei:dst:bias): %s:%s:%s:%s, "
            "bc=%zu, ic=%zu, ih=%zu, iw=%zu, oc=%zu, kh=%zu, kw=%zu, "
            "ph=%zu, pw=%zu, sh=%zu, sw=%zu, dh=%zu, dw=%zu, "
            "oh=%zu, ow=%zu ";
            printf(format, g_precisionMap[type].first.c_str(), g_precisionMap[type].first.c_str(),
                g_precisionMap[type].first.c_str(), g_precisionMap[type].first.c_str(),
                bc, ic, ih, iw, oc, kh, kw, ph, pw, sh, sw, dh, dw, oh, ow);
            KuDNN::ConvolutionLayerFWD convFwdLayer2d(isa, srcTensor2d, weightsTensor2d, dstTensor2d, biasTensor2d,
                strides2d, paddingL2d, dilates2d, prop, nthread, isblock, bUseWino);

            // 2d execute
            src = malloc(bc * ic * ih * iw * srcElemSz);
            weight = malloc(oc * ic * kh * kw * weightsElemSz);
            dst = malloc(bc * oc * oh * ow * dstElemSz);
            bias = malloc(oc * biasElemSz);
            if (src == nullptr || weight == nullptr || dst == nullptr || bias == nullptr) {
                std::cerr << "memory allocation failed" << std::endl;
                if (src != nullptr) {
                    free(src);
                }
                if (weight != nullptr) {
                    free(weight);
                }
                if (dst != nullptr) {
                    free(dst);
                }
                if (bias != nullptr) {
                    free(bias);
                }
                return;
            }
            convFwdLayer2d.Run(src, weight, dst, bias);
            free(src);
            free(weight);
            free(bias);
            free(dst);

        } catch (const KuDNN::Service::Unsupported& e) {
            if (src != nullptr) {
                free(src);
            }
            if (weight != nullptr) {
                free(weight);
            }
            if (bias != nullptr) {
                free(bias);
            }
            if (dst != nullptr) {
                free(dst);
            }
            printf("Unsupported Configuration");
        } catch (const KuDNN::Service::LogicError& e) {
            if (src != nullptr) {
                free(src);
            }
            if (weight != nullptr) {
                free(weight);
            }
            if (bias != nullptr) {
                free(bias);
            }
            if (dst != nullptr) {
                free(dst);
            }
            printf("Bad shapes' arguments");
        }
        printf("\n");
    }
    DT_FUZZ_END()
}
#endif
static void DtFuzzConvolutionIntFwd(int counts)
{
    char funcName[] = "DtFuzzConvolutionIntFwd";
    DT_FUZZ_START(0, counts, funcName, 0)
    {
        int eleIndex = 0;
        size_t ic = 32;
        size_t oc = 32;
        int elemTypeIndex = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], 0, 0,
            g_intPrecisionMap.size() - 1);
        int algIndex = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], 0, 0, g_algMap.size() - 1);
        size_t bc = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], g_dimRangeInitVal, g_dimRangeLowerBound,
            g_dimRangeUpperBound);
        size_t id = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], g_dimRangeInitVal, g_dimRangeLowerBound,
            g_dimRangeUpperBound);
        size_t ih = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], g_dimRangeInitVal, g_dimRangeLowerBound,
            g_dimRangeUpperBound);
        size_t iw = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], g_dimRangeInitVal, g_dimRangeLowerBound,
            g_dimRangeUpperBound);
        size_t kd = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], g_dimRangeInitVal, g_dimRangeLowerBound,
            g_dimRangeUpperBound);
        size_t kh = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], g_dimRangeInitVal, g_dimRangeLowerBound,
            g_dimRangeUpperBound);
        size_t kw = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], g_dimRangeInitVal, g_dimRangeLowerBound,
            g_dimRangeUpperBound);

        size_t sd = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], g_dimRangeInitVal, g_dimRangeLowerBound,
            g_dimRangeUpperBound);
        size_t sh = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], g_dimRangeInitVal, g_dimRangeLowerBound,
            g_dimRangeUpperBound);
        size_t sw = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], g_dimRangeInitVal, g_dimRangeLowerBound,
            g_dimRangeUpperBound);

        size_t dd = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], 0, 0, 2);
        size_t dh = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], 0, 0, 2);
        size_t dw = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], 0, 0, 2);
        size_t pd = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], g_dimRangeInitVal, g_dimRangeLowerBound,
            g_dimRangeUpperBound);
        size_t ph = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], g_dimRangeInitVal, g_dimRangeLowerBound,
            g_dimRangeUpperBound);
        size_t pw = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], g_dimRangeInitVal, g_dimRangeLowerBound,
            g_dimRangeUpperBound);

        size_t dkd = 1 + (kd -1) * (dd + 1);
        size_t dkh = 1 + (kh -1) * (dh + 1);
        size_t dkw = 1 + (kw -1) * (dw + 1);
        int64_t tmpOd = std::ceil(((int64_t)id - (int64_t)dkd + (int64_t)pd) / (int64_t)sd) + 1;
        int64_t tmpOh = std::ceil(((int64_t)ih - (int64_t)dkh + (int64_t)ph) / (int64_t)sh) + 1;
        int64_t tmpOw = std::ceil(((int64_t)iw - (int64_t)dkw + (int64_t)pw) / (int64_t)sw) + 1;
        size_t od = static_cast<size_t>(std::max(tmpOd, (int64_t)1));
        size_t oh = static_cast<size_t>(std::max(tmpOh, (int64_t)1));
        size_t ow = static_cast<size_t>(std::max(tmpOw, (int64_t)1));

        KuDNN::Element::Type srcType(g_intPrecisionMap[elemTypeIndex].first);
        KuDNN::Element::Type weightsType = KuDNN::Element::TypeT::S8;
        KuDNN::Element::Type dstType(g_intPrecisionMap[elemTypeIndex].second);
        KuDNN::Element::Type biasType = KuDNN::Element::TypeT::F32;
        size_t srcElemSz = srcType.GetSize();
        size_t weightsElemSz = weightsType.GetSize();
        size_t dstElemSz = dstType.GetSize();
        size_t biasElemSz = biasType.GetSize();
        KuDNN::Shape strides1d(sw);
        KuDNN::Shape dilates1d(dw);
        KuDNN::Shape paddingL1d(pw);
        KuDNN::Shape strides2d(sh, sw);
        KuDNN::Shape dilates2d(dh, dw);
        KuDNN::Shape paddingL2d(ph, pw);
        KuDNN::Shape strides3d(sd, sh, sw);
        KuDNN::Shape dilates3d(dd, dh, dw);
        KuDNN::Shape paddingL3d(pd, ph, pw);
        KuDNN::Propagation prop = KuDNN::Propagation::FORWARD_TRAINING;
#ifdef KUDNNL_920F
        int isa = 0x1f;
#else
        int isa = 0x07;
#endif
        int nthread = 1;

        bool isblock = true;
        bool bUseWino = false;

        const KuDNN::TensorInfo srcTensor1d = {{bc, ic, iw}, srcType, KuDNN::Layout::ACB};
        const KuDNN::TensorInfo weightsTensor1d = {{oc, ic, kw}, weightsType, KuDNN::Layout::ABC};
        const KuDNN::TensorInfo dstTensor1d = {{bc, oc, ow}, dstType, KuDNN::Layout::ACB};
        const KuDNN::TensorInfo biasTensor1d = {{oc}, biasType, KuDNN::Layout::A};

        const KuDNN::TensorInfo srcTensor2d = {{bc, ic, ih, iw}, srcType, KuDNN::Layout::ACDB};
        const KuDNN::TensorInfo weightsTensor2d = {{oc, ic, kh, kw}, weightsType, KuDNN::Layout::ABCD};
        const KuDNN::TensorInfo dstTensor2d = {{bc, oc, oh, ow}, dstType, KuDNN::Layout::ACDB};
        const KuDNN::TensorInfo biasTensor2d = {{oc}, biasType, KuDNN::Layout::A};

        const KuDNN::TensorInfo srcTensor3d = {{bc, ic, id, ih, iw}, srcType, KuDNN::Layout::ACDEB};
        const KuDNN::TensorInfo weightsTensor3d = {{oc, ic, kd, kh, kw}, weightsType, KuDNN::Layout::ABCDE};
        const KuDNN::TensorInfo dstTensor3d = {{bc, oc, od, oh, ow}, dstType, KuDNN::Layout::ACDEB};
        const KuDNN::TensorInfo biasTensor3d = {{oc}, biasType, KuDNN::Layout::A};

        void *src = nullptr;
        void *weight = nullptr;
        void *bias = nullptr;
        void *dst = nullptr;
        try {
            // constructors
            KuDNN::ConvolutionLayerFWD conv1(isa, srcTensor2d, weightsTensor2d, dstTensor2d, biasTensor2d,
                strides2d, paddingL2d, dilates2d, prop, nthread, isblock, bUseWino);
            KuDNN::ConvolutionLayerFWD conv2(isa, srcTensor2d, weightsTensor2d, dstTensor2d, biasTensor2d,
                strides2d, paddingL2d, dilates2d, prop, nthread, isblock, bUseWino);
            KuDNN::ConvolutionLayerFWD conv3(conv1);
            KuDNN::ConvolutionLayerFWD conv4(std::move(conv3));
            conv3 = conv1;
            conv4 = std::move(conv2);

            std::vector<std::string> precision = {"S8:S8:S8", "S8:S8:U8", "S8:S8:S32", "S8:S8:F32",
                "U8:S8:S8", "U8:S8:U8", "U8:S8:S32", "U8:S8:F32"};
            char format[] = "Forward Convolution Int: %s, Precision(src:diffWei:diffDst): %s"
                ", bc=%zu, ic=%zu, ih=%zu, iw=%zu, oc=%zu, kh=%zu, kw=%zu, ph=%zu, pw=%zu, sh=%zu, sw=%zu, dh=%zu, dw=%zu"
                " oh=%zu, ow=%zu ";

            printf(format, g_algMap[algIndex].first.c_str(), precision[elemTypeIndex].c_str(),
                bc, ic, ih, iw, oc, kh, kw, ph, pw, sh, sw, dh, dw,
                oh, ow);
            KuDNN::ConvolutionLayerFWD convFwdLayer1d(isa, srcTensor1d, weightsTensor1d, dstTensor1d, biasTensor1d,
                strides1d, paddingL1d, dilates1d, prop, nthread, isblock, bUseWino);
            KuDNN::ConvolutionLayerFWD convFwdLayer2d(isa, srcTensor2d, weightsTensor2d, dstTensor2d, biasTensor2d,
                strides2d, paddingL2d, dilates2d, prop, nthread, isblock, bUseWino);
            KuDNN::ConvolutionLayerFWD convFwdLayer3d(isa, srcTensor3d, weightsTensor3d, dstTensor3d, biasTensor3d,
                strides3d, paddingL3d, dilates3d, prop, nthread, isblock, bUseWino);

            // 1d execute
           src = malloc(bc * ic * iw * srcElemSz);
           if (srcTensor1d.GetType() == KuDNN::Element::TypeT::U8) {
                weight = malloc(oc * ic * kw * weightsElemSz + 16);
           } else {
                weight = malloc(oc * ic * kw * weightsElemSz);
           }
           dst = malloc(bc * oc * ow * dstElemSz);
           bias = malloc(oc * biasElemSz);
            if (src == nullptr || weight == nullptr || dst == nullptr || bias ==nullptr) {
                std::cerr << "memory allocation failed" << std::endl;
                if (src != nullptr) {
                    free(src);
                }
                if (weight != nullptr) {
                    free(weight);
                }
                if (dst != nullptr) {
                    free(dst);
                }
                if (bias != nullptr) {
                    free(bias);
                }
                return;
            }
            convFwdLayer1d.Run(src, weight, dst, bias);
            free(src);
            free(weight);
            free(bias);
            free(dst);

            // 2d execute
            src = malloc(bc * ic * ih * iw * srcElemSz);
            if (srcTensor2d.GetType() == KuDNN::Element::TypeT::U8) {
                weight = malloc(oc * ic * kh * kw * weightsElemSz + 16);
            } else {
                weight = malloc(oc * ic * kh * kw * weightsElemSz);
            }
            
            dst = malloc(bc * oc * oh * ow * dstElemSz);
            bias = malloc(oc * biasElemSz);
            if (src == nullptr || weight == nullptr || dst == nullptr || bias ==nullptr) {
                std::cerr << "memory allocation failed" << std::endl;
                if (src != nullptr) {
                    free(src);
                }
                if (weight != nullptr) {
                    free(weight);
                }
                if (dst != nullptr) {
                    free(dst);
                }
                if (bias != nullptr) {
                    free(bias);
                }
                return;
            }
            convFwdLayer2d.Run(src, weight, dst, bias);
            free(src);
            free(weight);
            free(bias);
            free(dst);

            // 3d execute
            src = malloc(bc * id * ic * ih * iw * srcElemSz);
            if (srcTensor3d.GetType() == KuDNN::Element::TypeT::U8) {
                weight = malloc(oc * ic * kd * kh * kw * weightsElemSz + 16);
            } else {
                weight = malloc(oc * ic * kd * kh * kw * weightsElemSz);
            }
            
            dst = malloc(bc * oc * od * oh * ow * dstElemSz);
            bias = malloc(oc * biasElemSz);
            if (src == nullptr || weight == nullptr || dst == nullptr || bias ==nullptr) {
                std::cerr << "memory allocation failed" << std::endl;
                if (src != nullptr) {
                    free(src);
                }
                if (weight != nullptr) {
                    free(weight);
                }
                if (dst != nullptr) {
                    free(dst);
                }
                if (bias != nullptr) {
                    free(bias);
                }
                return;
            }
            convFwdLayer3d.Run(src, weight, dst, bias);
            free(src);
            free(weight);
            free(bias);
            free(dst);
        } catch (const KuDNN::Service::Unsupported& e) {
            if (src != nullptr) {
                free(src);
            }
            if (weight != nullptr) {
                free(weight);
            }
            if (dst != nullptr) {
                free(dst);
            }
            if (bias != nullptr) {
                free(bias);
            }
            printf("Unsupported Configuration");
        } catch (const KuDNN::Service::LogicError& e) {
            if (src != nullptr) {
                free(src);
            }
            if (weight != nullptr) {
                free(weight);
            }
            if (dst != nullptr) {
                free(dst);
            }
            if (bias != nullptr) {
                free(bias);
            }
            printf("Bad shapes' arguments");
        }
        printf("\n");
    }
    DT_FUZZ_END()
}


const int  g_convolutionFuzzTestCount = 8000;

Cmdinfo convolutionCmdTable[] = {
    {"BrgemmConvolutionFwd", DtFuzzBrgemmConvolutionFwd, g_convolutionFuzzTestCount},
    {"ConvolutionFwd", DtFuzzConvolutionFwd, g_convolutionFuzzTestCount},
    {"ConvolutionBwdData", DtFuzzConvolutionBwdData, g_convolutionFuzzTestCount},
    {"ConvolutionBwdWeights", DtFuzzConvolutionBwdWeights, g_convolutionFuzzTestCount},
    {"ConvolutionLayerFWD", DtFuzzConvolutionJitFwd, g_convolutionFuzzTestCount},
    {"ConvolutionDwFwd", DtFuzzConvolutionDwFwd, g_convolutionFuzzTestCount},
    {"ConvolutionGroupFwd", DtFuzzConvolutionGroupFwd, g_convolutionFuzzTestCount},
    {"ConvolutionMCJitFwd", DtFuzzConvolutionMCJitFwd, g_convolutionFuzzTestCount},
#if defined(KUDNNL_920B) || defined(KUDNNL_920G)
    {"ConvolutionWinoFwd", DtFuzzConvolutionWinoFwd, g_convolutionFuzzTestCount},
#endif
    {"ConvolutionIntFwd", DtFuzzConvolutionIntFwd, g_convolutionFuzzTestCount}
};

int main(int argc, char *argv[])
{
    DT_Enable_Leak_Check(0, 0);
    if (argc > 1) {
        convolutionCmdTable[0] = {"BrgemmConvolutionFwd", DtFuzzBrgemmConvolutionFwd, atoi(argv[1])};
        convolutionCmdTable[1] = {"ConvolutionFwd", DtFuzzConvolutionFwd, atoi(argv[1])};
        convolutionCmdTable[2] = {"ConvolutionBwdData", DtFuzzConvolutionBwdData, atoi(argv[1])};
        convolutionCmdTable[3] = {"ConvolutionBwdWeights", DtFuzzConvolutionBwdWeights, atoi(argv[1])};
        convolutionCmdTable[4] = {"ConvolutionLayerFWD", DtFuzzConvolutionJitFwd, atoi(argv[1])};
        convolutionCmdTable[5] = {"ConvolutionDwFwd", DtFuzzConvolutionDwFwd, atoi(argv[1])};
        convolutionCmdTable[6] = {"ConvolutionGroupFwd", DtFuzzConvolutionGroupFwd, atoi(argv[1])};
        convolutionCmdTable[7] = {"ConvolutionMCJitFwd", DtFuzzConvolutionMCJitFwd, atoi(argv[1])};
#if defined(KUDNNL_920B) || defined(KUDNNL_920G)
        convolutionCmdTable[8] = {"ConvolutionWinoFwd", DtFuzzConvolutionWinoFwd, atoi(argv[1])};
        convolutionCmdTable[9] = {"ConvolutionIntFwd", DtFuzzConvolutionIntFwd, atoi(argv[1])};
#else
        convolutionCmdTable[8] = {"ConvolutionIntFwd", DtFuzzConvolutionIntFwd, atoi(argv[1])};
#endif
        
    }

    int numOfTestItem = sizeof(convolutionCmdTable) / sizeof(convolutionCmdTable[0]);
    int testCount = 0;

    for (int testCaseIdx = 0; testCaseIdx < numOfTestItem; testCaseIdx++) {
        printf("convCmdTable[%d] = %s test %d time\n", testCaseIdx, convolutionCmdTable[testCaseIdx].cmd, testCount);
        testCount = convolutionCmdTable[testCaseIdx].testCount;
        convolutionCmdTable[testCaseIdx].func(testCount);
    }
    return 0;
}