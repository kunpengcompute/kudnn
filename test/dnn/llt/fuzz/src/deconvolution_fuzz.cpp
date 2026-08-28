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
    {INDEX3, {"S8", KuDNN::Element::TypeT::S8}},
    {INDEX4, {"U8", KuDNN::Element::TypeT::U8}},
    {INDEX5, {"S32", KuDNN::Element::TypeT::S32}},
};

static std::map<int, std::pair<std::string, KuDNN::DeconvolutionAlgorithm>> g_algMap = {
    {INDEX0, {"DIRECT", KuDNN::DeconvolutionAlgorithm::DIRECT}},
    {INDEX1, {"WINOGRAD", KuDNN::DeconvolutionAlgorithm::WINOGRAD}},
    {INDEX1, {"UNIMPLEMENTED", KuDNN::DeconvolutionAlgorithm::UNIMPLEMENTED}},
};

static std::map<int, bool> g_biasMap = {
    {INDEX0, true},
    {INDEX1, false},
};

static std::map<int, std::pair<std::string, KuDNN::Layout>> g_1dLayoutMap = {
    {INDEX0, {"ABC", KuDNN::Layout::ABC}},
    {INDEX1, {"ACB", KuDNN::Layout::ACB}},
};
static std::map<int, std::pair<std::string, KuDNN::Layout>> g_2dLayoutMap = {
    {INDEX0, {"ABCD", KuDNN::Layout::ABCD}},
    {INDEX1, {"ACBD", KuDNN::Layout::ACBD}},
};

static std::map<int, std::pair<std::string, KuDNN::Layout>> g_3dLayoutMap = {
    {INDEX0, {"ACDEB", KuDNN::Layout::ACDEB}},
    {INDEX1, {"ABCDE", KuDNN::Layout::ABCDE}},
};

void FreeObj(void **obj)
{
    if (*obj != nullptr) {
        free(*obj);
    }
    *obj = nullptr;
}

static constexpr int g_dimRangeInitVal = 2;
static constexpr int g_dimRangeLowerBound = 1;
static constexpr int g_dimRangeUpperBound = 20;

static void DtFuzzDeconvolutionFwd(int counts)
{
    int num = 0;
    char funcName[] = "DtFuzzDeconvolutionFwd";
    DT_FUZZ_START(0, counts, funcName, 0)
    {
        int eleIndex = 0;
        int srcElemTypeIndex = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], 0, 0, g_precisionMap.size() - 1);
        int dstElemTypeIndex = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], 0, 0, g_precisionMap.size() - 1);
        int biasElemTypeIndex = dstElemTypeIndex;
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
        size_t id = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], g_dimRangeInitVal, g_dimRangeLowerBound,
            g_dimRangeUpperBound);
        size_t kd = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], g_dimRangeInitVal, g_dimRangeLowerBound,
            g_dimRangeUpperBound);
        size_t od = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], g_dimRangeInitVal, g_dimRangeLowerBound,
            g_dimRangeUpperBound);
        size_t dh = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], 0, 0, 2);
        size_t dw = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], 0, 0, 2);
        size_t pt = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], 0, 0, g_dimRangeUpperBound);
        size_t pl = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], 0, 0, g_dimRangeUpperBound);
        size_t pb = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], 0, 0, g_dimRangeUpperBound);
        size_t pr = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], 0, 0, g_dimRangeUpperBound);

        bool hasBias =  g_biasMap[hasBiasIndex];
        KuDNN::Element::Type srcType(g_precisionMap[srcElemTypeIndex].second);
        KuDNN::Element::Type weightsType(g_precisionMap[srcElemTypeIndex].second);
        KuDNN::Element::Type dstType(g_precisionMap[dstElemTypeIndex].second);
        KuDNN::Element::Type biasType(g_precisionMap[biasElemTypeIndex].second);
        num++;
        if (num > 1000 && num < 1050) {
            srcType = KuDNN::Element::TypeT::U8;
            weightsType = KuDNN::Element::TypeT::S8;
            dstType = KuDNN::Element::TypeT::F32;
            biasType = KuDNN::Element::TypeT::F32;
        }
        size_t srcElemSz = srcType.GetSize();
        size_t weightsElemSz = weightsType.GetSize();
        size_t dstElemSz = dstType.GetSize();
        size_t biasElemSz = biasType.GetSize();
        KuDNN::Shape strides(sh, sw);
        KuDNN::Shape dilates(dh, dw);
        KuDNN::Shape paddingL(pt, pl);
        KuDNN::Shape paddingR(pb, pr);
        KuDNN::DeconvolutionAlgorithm alg(g_algMap[algIndex].second);
        int64_t tmpOh = ((int64_t)ih - 1) * (int64_t)sh + ((int64_t)kh - 1) * ((int64_t)dh + 1) -
            ((int64_t)pt + (int64_t)pb) + 1;
        int64_t tmpOw = ((int64_t)iw - 1) * (int64_t)sw + ((int64_t)kw - 1) * ((int64_t)dw + 1) -
            ((int64_t)pb + (int64_t)pr) + 1;
        size_t oh = static_cast<size_t>(std::max(tmpOh, (int64_t)1));
        size_t ow = static_cast<size_t>(std::max(tmpOw, (int64_t)1));

        void *src = nullptr;
        void *weight = nullptr;
        void *bias = nullptr;
        void *dst = nullptr;
        try {
            const KuDNN::TensorInfo srcTensor = {{bc, ic, ih, iw}, srcType, KuDNN::Layout::ABCD};
            const KuDNN::TensorInfo weightsTensor = {{oc, ic, kh, kw}, weightsType, KuDNN::Layout::ABCD};
            const KuDNN::TensorInfo dstTensor = {{bc, oc, oh, ow}, dstType, KuDNN::Layout::ABCD};
            const KuDNN::TensorInfo biasTensor = {{oc}, biasType, KuDNN::Layout::A};

            KuDNN::DeconvolutionLayerFWD deconv1(srcTensor, weightsTensor, dstTensor, biasTensor, strides,
            dilates, paddingL, paddingR, alg);
            KuDNN::DeconvolutionLayerFWD deconv2(srcTensor, weightsTensor, dstTensor, biasTensor, strides,
            paddingL, paddingR, alg);
            KuDNN::DeconvolutionLayerFWD deconv3(deconv1);
            deconv3 = deconv1;
            KuDNN::DeconvolutionLayerFWD deconv5(deconv1);
            deconv5 = std::move(deconv2);
            KuDNN::DeconvolutionLayerFWD deconv6(std::move(deconv3));

            int srcIndex1 = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], 0, 0, g_1dLayoutMap.size() - 1);
            int weiIndex1 = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], 0, 0, g_1dLayoutMap.size() - 1);
            int dstIndex1 = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], 0, 0, g_1dLayoutMap.size() - 1);
            KuDNN::Layout srcLay1 = g_1dLayoutMap[srcIndex1].second;
            KuDNN::Layout dstLay1 = g_1dLayoutMap[weiIndex1].second;
            KuDNN::Layout weiLay1 = g_1dLayoutMap[dstIndex1].second;

            int srcIndex2 = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], 0, 0, g_2dLayoutMap.size() - 1);
            int weiIndex2 = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], 0, 0, g_2dLayoutMap.size() - 1);
            int dstIndex2 = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], 0, 0, g_2dLayoutMap.size() - 1);
            KuDNN::Layout srcLay2 = g_2dLayoutMap[srcIndex2].second;
            KuDNN::Layout dstLay2 = g_2dLayoutMap[weiIndex2].second;
            KuDNN::Layout weiLay2 = g_2dLayoutMap[dstIndex2].second;

            int srcIndex3 = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], 0, 0, g_3dLayoutMap.size() - 1);
            int weiIndex3 = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], 0, 0, g_3dLayoutMap.size() - 1);
            int dstIndex3 = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], 0, 0, g_3dLayoutMap.size() - 1);
            KuDNN::Layout srcLay3 = g_3dLayoutMap[srcIndex3].second;
            KuDNN::Layout dstLay3 = g_3dLayoutMap[weiIndex3].second;
            KuDNN::Layout weiLay3 = g_3dLayoutMap[dstIndex3].second;

            const KuDNN::TensorInfo srcTensor1d = {{bc, ic, iw}, srcType, srcLay1};
            const KuDNN::TensorInfo weightsTensor1d = {{oc, ic, kw}, weightsType, weiLay1};
            const KuDNN::TensorInfo dstTensor1d = {{bc, oc, ow}, dstType, dstLay1};
            const KuDNN::TensorInfo biasTensor1d = {{oc}, biasType, KuDNN::Layout::A};

            KuDNN::TensorInfo srcTensor2d = {{bc, ic, ih, iw}, srcType, srcLay2};
            const KuDNN::TensorInfo weightsTensor2d = {{oc, ic, kh, kw}, weightsType, weiLay2};
            if(num % 2 == 1) {
                srcTensor2d = srcTensor1d;
            }
            const KuDNN::TensorInfo dstTensor2d = {{bc, oc, oh, ow}, dstType, dstLay2};
            const KuDNN::TensorInfo biasTensor2d = {{oc}, biasType, KuDNN::Layout::A};

            KuDNN::TensorInfo srcTensor3d = {{bc, ic, id, ih, iw}, srcType, srcLay3};
            const KuDNN::TensorInfo weightsTensor3d = {{oc, ic, kd, kh, kw}, weightsType, weiLay3};
            if(num % 2 == 1) {
                KuDNN::TensorInfo srcTensor2d = {{bc, ic, ih, iw}, srcType, srcLay2};
                srcTensor3d = srcTensor2d;
            }
            const KuDNN::TensorInfo dstTensor3d = {{bc, oc, od, oh, ow}, dstType, dstLay3};
            const KuDNN::TensorInfo biasTensor3d = {{oc}, biasType, KuDNN::Layout::A};

            char format[] = "Forward Deconv: %s, Precision(src:wei:dst:bias): %s:%s:%s:%s"
                ", bc=%zu, ic=%zu, ih=%zu, iw=%zu, oc=%zu, kh=%zu, kw=%zu, pt=%zu, pl=%zu, sh=%zu, sw=%zu, dh=%zu, dw=%zu"
                " oh=%zu, ow=%zu";
            printf(format, g_algMap[algIndex].first.c_str(), g_precisionMap[srcElemTypeIndex].first.c_str(),
                g_precisionMap[srcElemTypeIndex].first.c_str(), g_precisionMap[dstElemTypeIndex].first.c_str(),
                g_precisionMap[biasElemTypeIndex].first.c_str(), bc, ic, ih, iw, oc, kh, kw, pt, pl, sh, sw, dh, dw,
                oh, ow);
            //1d
            KuDNN::DeconvolutionLayerFWD deconvFwdLayer1d(srcTensor1d, weightsTensor1d, dstTensor1d, biasTensor1d,
                strides, dilates, paddingL, paddingR, alg);
            src = malloc(bc * ic * iw * srcElemSz);
            if (src == nullptr) {
                std::cerr << "memory allocation failed" << std::endl;
                return;
            }
            weight = malloc(oc * ic * kw * weightsElemSz);
            if (weight == nullptr) {
                FreeObj(&src);
                std::cerr << "memory allocation failed" << std::endl;
                return;
            }
            dst = malloc(bc * oc * ow * dstElemSz);
            if (dst == nullptr) {
                FreeObj(&src);
                FreeObj(&weight);
                std::cerr << "memory allocation failed" << std::endl;
                return;
            }
            if (hasBias) {
                bias = malloc(oc * biasElemSz);
                if (bias == nullptr) {
                    FreeObj(&src);
                    FreeObj(&weight);
                    FreeObj(&dst);
                    std::cerr << "bias memory allocation failed" << std::endl;
                    return;
                }
            }
            deconvFwdLayer1d.Run(src, weight, dst, bias);
            FreeObj(&src);
            FreeObj(&weight);
            if (bias != nullptr) {
                FreeObj(&bias);
            }
            FreeObj(&dst);
            //2d
            KuDNN::DeconvolutionLayerFWD deconvFwdLayer2d(srcTensor2d, weightsTensor2d, dstTensor2d, biasTensor2d,
                strides, dilates, paddingL, paddingR, alg);
            src = malloc(bc * ic * ih * iw * srcElemSz);
            if (src == nullptr) {
                std::cerr << "memory allocation failed" << std::endl;
                return;
            }
            weight = malloc(oc * ic * kh * kw * weightsElemSz);
            if (weight == nullptr) {
                FreeObj(&src);
                std::cerr << "memory allocation failed" << std::endl;
                return;
            }
            dst = malloc(bc * oc * oh * ow * dstElemSz);
            if (dst == nullptr) {
                FreeObj(&src);
                FreeObj(&weight);
                std::cerr << "memory allocation failed" << std::endl;
                return;
            }
            if (hasBias) {
                bias = malloc(oc * biasElemSz);
                if (bias == nullptr) {
                    FreeObj(&src);
                    FreeObj(&weight);
                    FreeObj(&dst);
                    std::cerr << "bias memory allocation failed" << std::endl;
                    return;
                }
            }
            deconvFwdLayer2d.Run(src, weight, dst, bias);
            FreeObj(&src);
            FreeObj(&weight);
            if (bias != nullptr) {
                FreeObj(&bias);
            }
            FreeObj(&dst);

            //3d
            KuDNN::DeconvolutionLayerFWD deconvFwdLayer3d(srcTensor3d, weightsTensor3d, dstTensor3d, biasTensor3d,
                strides, dilates, paddingL, paddingR, alg);
            src = malloc(bc * ic * ih * id * iw * srcElemSz);
            if (src == nullptr) {
                std::cerr << "memory allocation failed" << std::endl;
                return;
            }
            weight = malloc(oc * ic * kh * kd * kw * weightsElemSz);
            if (weight == nullptr) {
                FreeObj(&src);
                std::cerr << "memory allocation failed" << std::endl;
                return;
            }
            dst = malloc(bc * oc * oh * od * ow * dstElemSz);
            if (dst == nullptr) {
                FreeObj(&src);
                FreeObj(&weight);
                std::cerr << "memory allocation failed" << std::endl;
                return;
            }
            if (hasBias) {
                bias = malloc(oc * biasElemSz);
                if (bias == nullptr) {
                    FreeObj(&src);
                    FreeObj(&weight);
                    FreeObj(&dst);
                    std::cerr << "bias memory allocation failed" << std::endl;
                    return;
                }
            }
            deconvFwdLayer3d.Run(src, weight, dst, bias);
            FreeObj(&src);
            FreeObj(&weight);
            if (bias != nullptr) {
                FreeObj(&bias);
            }
            FreeObj(&dst);
        } catch (const KuDNN::Service::Unsupported& e) {
            if (src != nullptr) {
                FreeObj(&src);
            }
            if (weight != nullptr) {
                FreeObj(&weight);
            }
            if (bias != nullptr) {
                FreeObj(&bias);
            }
            if (dst != nullptr) {
                FreeObj(&dst);
            }
            printf("Unsupported Configuration");
        } catch (const KuDNN::Service::LogicError& e) {
            if (src != nullptr) {
                FreeObj(&src);
            }
            if (weight != nullptr) {
                FreeObj(&weight);
            }
            if (bias != nullptr) {
                FreeObj(&bias);
            }
            if (dst != nullptr) {
                FreeObj(&dst);
            }
            printf("Bad shapes' arguments");
        }
        printf("\n");
    }
    DT_FUZZ_END()
}

static void DtFuzzDeconvolutionBwdData(int counts)
{
    char funcName[] = "DtFuzzDeconvolutionBwdData";
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
        size_t id = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], g_dimRangeInitVal, g_dimRangeLowerBound,
            g_dimRangeUpperBound);
        size_t kd = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], g_dimRangeInitVal, g_dimRangeLowerBound,
            g_dimRangeUpperBound);
        size_t od = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], g_dimRangeInitVal, g_dimRangeLowerBound,
            g_dimRangeUpperBound);

        size_t dh = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], 0, 0, 2);
        size_t dw = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], 0, 0, 2);
        size_t pt = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], 0, 0, g_dimRangeUpperBound);
        size_t pl = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], 0, 0, g_dimRangeUpperBound);
        size_t pb = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], 0, 0, g_dimRangeUpperBound);
        size_t pr = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], 0, 0, g_dimRangeUpperBound);

        KuDNN::Element::Type diffSrcType(g_precisionMap[diffSrcElemTypeIndex].second);
        KuDNN::Element::Type weightsType(g_precisionMap[diffDstElemTypeIndex].second);
        KuDNN::Element::Type diffDstType(g_precisionMap[diffDstElemTypeIndex].second);
        size_t diffSrcElemSz = diffSrcType.GetSize();
        size_t weightsElemSz = weightsType.GetSize();
        size_t diffDstElemSz = diffDstType.GetSize();
        KuDNN::Shape strides(sh, sw);
        KuDNN::Shape dilates(dh, dw);
        KuDNN::Shape paddingL(pt, pl);
        KuDNN::Shape paddingR(pb, pr);
        KuDNN::DeconvolutionAlgorithm alg(g_algMap[algIndex].second);

        int64_t tmpOh = ((int64_t)ih - 1) * (int64_t)sh + ((int64_t)kh - 1) * ((int64_t)dh + 1) -
            ((int64_t)pt + (int64_t)pb) + 1;
        int64_t tmpOw = ((int64_t)iw - 1) * (int64_t)sw + ((int64_t)kw - 1) * ((int64_t)dw + 1) -
            ((int64_t)pb + (int64_t)pr) + 1;
        size_t oh = static_cast<size_t>(std::max(tmpOh, (int64_t)1));
        size_t ow = static_cast<size_t>(std::max(tmpOw, (int64_t)1));

        const KuDNN::TensorInfo diffSrcTensor = {{bc, ic, ih, iw}, diffSrcType, KuDNN::Layout::ABCD};
        const KuDNN::TensorInfo weightsTensor = {{oc, ic, kh, kw}, weightsType, KuDNN::Layout::ABCD};
        const KuDNN::TensorInfo diffDstTensor = {{bc, oc, oh, ow}, diffDstType, KuDNN::Layout::ABCD};

        char format[] = "Backward Data Deconv: %s, Precision(diffSrc:wei:diffDst): %s:%s:%s"
            ", bc=%zu, ic=%zu, ih=%zu, iw=%zu, oc=%zu, kh=%zu, kw=%zu, pt=%zu, pl=%zu, sh=%zu, sw=%zu, dh=%zu, dw=%zu"
            " oh=%zu, ow=%zu ";
        printf(format, g_algMap[algIndex].first.c_str(), g_precisionMap[diffSrcElemTypeIndex].first.c_str(),
            g_precisionMap[diffDstElemTypeIndex].first.c_str(), g_precisionMap[diffDstElemTypeIndex].first.c_str(),
            bc, ic, ih, iw, oc, kh, kw, pt, pl, sh, sw, dh, dw,oh, ow);
        void *diffSrc = nullptr;
        void *weight = nullptr;
        void *diffDst = nullptr;
        try {
            KuDNN::DeconvolutionLayerBWDData deconv1(diffDstTensor, weightsTensor, diffSrcTensor, strides, dilates,
                paddingL, paddingR, alg);
            KuDNN::DeconvolutionLayerBWDData deconv2(diffDstTensor, weightsTensor, diffSrcTensor, strides, paddingL,
                paddingR, alg);
            KuDNN::DeconvolutionLayerBWDData deconv3(deconv1);
            deconv3 = deconv1;
            KuDNN::DeconvolutionLayerBWDData deconv5(deconv1);
            deconv5 = std::move(deconv2);
            KuDNN::DeconvolutionLayerBWDData deconv6(std::move(deconv3));
            int srcIndex1 = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], 0, 0, g_1dLayoutMap.size() - 1);
            int weiIndex1 = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], 0, 0, g_1dLayoutMap.size() - 1);
            int dstIndex1 = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], 0, 0, g_1dLayoutMap.size() - 1);
            KuDNN::Layout srcLay1 = g_1dLayoutMap[srcIndex1].second;
            KuDNN::Layout dstLay1 = g_1dLayoutMap[weiIndex1].second;
            KuDNN::Layout weiLay1 = g_1dLayoutMap[dstIndex1].second;

            int srcIndex2 = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], 0, 0, g_2dLayoutMap.size() - 1);
            int weiIndex2 = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], 0, 0, g_2dLayoutMap.size() - 1);
            int dstIndex2 = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], 0, 0, g_2dLayoutMap.size() - 1);
            KuDNN::Layout srcLay2 = g_2dLayoutMap[srcIndex2].second;
            KuDNN::Layout dstLay2 = g_2dLayoutMap[weiIndex2].second;
            KuDNN::Layout weiLay2 = g_2dLayoutMap[dstIndex2].second;

            int srcIndex3 = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], 0, 0, g_3dLayoutMap.size() - 1);
            int weiIndex3 = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], 0, 0, g_3dLayoutMap.size() - 1);
            int dstIndex3 = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], 0, 0, g_3dLayoutMap.size() - 1);
            KuDNN::Layout srcLay3 = g_3dLayoutMap[srcIndex3].second;
            KuDNN::Layout dstLay3 = g_3dLayoutMap[weiIndex3].second;
            KuDNN::Layout weiLay3 = g_3dLayoutMap[dstIndex3].second;

            const KuDNN::TensorInfo diffSrcTensor1d = {{bc, ic, iw}, diffSrcType, srcLay1};
            const KuDNN::TensorInfo weightsTensor1d = {{oc, ic, kw}, weightsType, dstLay1};
            const KuDNN::TensorInfo diffDstTensor1d = {{bc, oc, ow}, diffDstType, weiLay1};

            const KuDNN::TensorInfo diffSrcTensor2d = {{bc, ic, ih, iw}, diffSrcType, srcLay2};
            const KuDNN::TensorInfo weightsTensor2d = {{oc, ic, kh, kw}, weightsType, weiLay2};
            const KuDNN::TensorInfo diffDstTensor2d = {{bc, oc, oh, ow}, diffDstType, dstLay2};

            const KuDNN::TensorInfo diffSrcTensor3d = {{bc, ic, ih, id, iw}, diffSrcType, srcLay3};
            const KuDNN::TensorInfo weightsTensor3d = {{oc, ic, kh, kd, kw}, weightsType, weiLay3};
            const KuDNN::TensorInfo diffDstTensor3d = {{bc, oc, oh, od, ow}, diffDstType, dstLay3};

            //1d
            KuDNN::DeconvolutionLayerBWDData deconvBwdDataLayer1d(diffDstTensor1d, weightsTensor1d, diffSrcTensor1d, strides,
                dilates, paddingL, paddingR, alg);
            diffSrc = malloc(bc * ic * iw * diffSrcElemSz);
            if (diffSrc == nullptr) {
                std::cerr << "memory allocation failed" << std::endl;
                return;
            }

            weight = malloc(oc * ic * kw * weightsElemSz);
            if (weight == nullptr) {
                FreeObj(&diffSrc);
                std::cerr << "memory allocation failed" << std::endl;
                return;
            }

            diffDst = malloc(bc * oc * ow * diffDstElemSz);
            if (diffDst == nullptr) {
                FreeObj(&diffSrc);
                FreeObj(&weight);
                std::cerr << "memory allocation failed" << std::endl;
                return;
            }
            deconvBwdDataLayer1d.Run(diffDst, weight, diffSrc);
            FreeObj(&diffDst);
            FreeObj(&weight);
            FreeObj(&diffSrc);

            //2d
            KuDNN::DeconvolutionLayerBWDData deconvBwdDataLayer2d(diffDstTensor2d, weightsTensor2d, diffSrcTensor2d, strides,
                dilates, paddingL, paddingR, alg);
            diffSrc = malloc(bc * ic * ih * iw * diffSrcElemSz);
            if (diffSrc == nullptr) {
                std::cerr << "memory allocation failed" << std::endl;
                return;
            }

            weight = malloc(oc * ic * kh * kw * weightsElemSz);
            if (weight == nullptr) {
                FreeObj(&diffSrc);
                std::cerr << "memory allocation failed" << std::endl;
                return;
            }

            diffDst = malloc(bc * oc * oh * ow * diffDstElemSz);
            if (diffDst == nullptr) {
                FreeObj(&diffSrc);
                FreeObj(&weight);
                std::cerr << "memory allocation failed" << std::endl;
                return;
            }
            deconvBwdDataLayer2d.Run(diffDst, weight, diffSrc);
            FreeObj(&diffDst);
            FreeObj(&weight);
            FreeObj(&diffSrc);

            //3d
            KuDNN::DeconvolutionLayerBWDData deconvBwdDataLayer3d(diffDstTensor3d, weightsTensor3d, diffSrcTensor3d, strides,
                dilates, paddingL, paddingR, alg);
            diffSrc = malloc(bc * ic * ih * iw * diffSrcElemSz);
            if (diffSrc == nullptr) {
                std::cerr << "memory allocation failed" << std::endl;
                return;
            }

            weight = malloc(oc * ic * kh * kw * weightsElemSz);
            if (weight == nullptr) {
                FreeObj(&diffSrc);
                std::cerr << "memory allocation failed" << std::endl;
                return;
            }

            diffDst = malloc(bc * oc * oh * ow * diffDstElemSz);
            if (diffDst == nullptr) {
                FreeObj(&diffSrc);
                FreeObj(&weight);
                std::cerr << "memory allocation failed" << std::endl;
                return;
            }
            deconvBwdDataLayer3d.Run(diffDst, weight, diffSrc);
            FreeObj(&diffDst);
            FreeObj(&weight);
            FreeObj(&diffSrc);
        } catch (const KuDNN::Service::Unsupported& e) {
            if (diffSrc != nullptr) {
                FreeObj(&diffSrc);
            }
            if (weight != nullptr) {
                FreeObj(&weight);
            }
            if (diffDst != nullptr) {
                FreeObj(&diffDst);
            }
            printf("Unsupported Configuration");
        } catch (const KuDNN::Service::LogicError& e) {
            if (diffSrc != nullptr) {
                FreeObj(&diffSrc);
            }
            if (weight != nullptr) {
                FreeObj(&weight);
            }
            if (diffDst != nullptr) {
                FreeObj(&diffDst);
            }
            printf("Bad shapes' arguments");
        }
        printf("\n");
    }
    DT_FUZZ_END()
}

static void DtFuzzDeconvolutionBwdWeights(int counts)
{
    char funcName[] = "DtFuzzDeconvolutionBwdWeights";
    DT_FUZZ_START(0, counts, funcName, 0)
    {
        int eleIndex = 0;
        int diffweightsElemTypeIndex = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], 0, 0,
            g_precisionMap.size() - 1);
        int diffDstElemTypeIndex = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], 0, 0,
            g_precisionMap.size() - 1);
        int diffBiasElemTypeIndex = diffDstElemTypeIndex;
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
        size_t id = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], g_dimRangeInitVal, g_dimRangeLowerBound,
            g_dimRangeUpperBound);
        size_t kd = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], g_dimRangeInitVal, g_dimRangeLowerBound,
            g_dimRangeUpperBound);
        size_t od = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], g_dimRangeInitVal, g_dimRangeLowerBound,
            g_dimRangeUpperBound);
        size_t dh = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], 0, 0, 2);
        size_t dw = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], 0, 0, 2);
        size_t pt = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], 0, 0, g_dimRangeUpperBound);
        size_t pl = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], 0, 0, g_dimRangeUpperBound);
        size_t pb = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], 0, 0, g_dimRangeUpperBound);
        size_t pr = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], 0, 0, g_dimRangeUpperBound);

        bool hasBias =  g_biasMap[hasBiasIndex];
        KuDNN::Element::Type srcType(g_precisionMap[diffDstElemTypeIndex].second);
        KuDNN::Element::Type diffWeightsType(g_precisionMap[diffweightsElemTypeIndex].second);
        KuDNN::Element::Type diffDstType(g_precisionMap[diffDstElemTypeIndex].second);
        KuDNN::Element::Type diffBiasType(g_precisionMap[diffBiasElemTypeIndex].second);
        size_t srcElemSz = srcType.GetSize();
        size_t diffWeightsElemSz = diffWeightsType.GetSize();
        size_t diffDstElemSz = diffDstType.GetSize();
        size_t diffBiasElemSz = diffBiasType.GetSize();
        KuDNN::Shape strides(sh, sw);
        KuDNN::Shape dilates(dh, dw);
        KuDNN::Shape paddingL(pt, pl);
        KuDNN::Shape paddingR(pb, pr);
        KuDNN::DeconvolutionAlgorithm alg(g_algMap[algIndex].second);

        int64_t tmpOh = ((int64_t)ih - 1) * (int64_t)sh + ((int64_t)kh - 1) * ((int64_t)dh + 1) -
            ((int64_t)pt + (int64_t)pb) + 1;
        int64_t tmpOw = ((int64_t)iw - 1) * (int64_t)sw + ((int64_t)kw - 1) * ((int64_t)dw + 1) -
            ((int64_t)pb + (int64_t)pr) + 1;
        size_t oh = static_cast<size_t>(std::max(tmpOh, (int64_t)1));
        size_t ow = static_cast<size_t>(std::max(tmpOw, (int64_t)1));

        const KuDNN::TensorInfo srcTensor = {{bc, ic, ih, iw}, srcType, KuDNN::Layout::ABCD};
        const KuDNN::TensorInfo diffWeightsTensor = {{oc, ic, kh, kw}, diffWeightsType, KuDNN::Layout::ABCD};
        const KuDNN::TensorInfo diffDstTensor = {{bc, oc, oh, ow}, diffDstType, KuDNN::Layout::ABCD};
        const KuDNN::TensorInfo diffBiasTensor = {{oc}, diffBiasType, KuDNN::Layout::A};

        char format[] = "Backward Weights Deconv: %s, Precision(src:diffWei:diffDst:diffBias): %s:%s:%s:%s"
            ", bc=%zu, ic=%zu, ih=%zu, iw=%zu, oc=%zu, kh=%zu, kw=%zu, pt=%zu, pl=%zu, sh=%zu, sw=%zu, dh=%zu, dw=%zu"
            " oh=%zu, ow=%zu ";
        printf(format, g_algMap[algIndex].first.c_str(), g_precisionMap[diffDstElemTypeIndex].first.c_str(),
            g_precisionMap[diffweightsElemTypeIndex].first.c_str(), g_precisionMap[diffDstElemTypeIndex].first.c_str(),
            g_precisionMap[diffBiasElemTypeIndex].first.c_str(), bc, ic, ih, iw, oc, kh, kw, pt, pl, sh, sw, dh, dw,
            oh, ow);
        void *src = nullptr;
        void *diffWeight = nullptr;
        void *diffBias = nullptr;
        void *diffDst = nullptr;
        try {
            KuDNN::DeconvolutionLayerBWDWeights deconv1(diffDstTensor, srcTensor, diffWeightsTensor, diffBiasTensor,
                strides, dilates, paddingL, paddingR, alg);
            KuDNN::DeconvolutionLayerBWDWeights deconv2(diffDstTensor, srcTensor, diffWeightsTensor, diffBiasTensor,
                strides, paddingL, paddingR, alg);
            KuDNN::DeconvolutionLayerBWDWeights deconv3(deconv1);
            deconv3 = deconv1;
            KuDNN::DeconvolutionLayerBWDWeights deconv5(deconv1);
            deconv5 = std::move(deconv2);
            KuDNN::DeconvolutionLayerBWDWeights deconv6(std::move(deconv3));

            int srcIndex1 = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], 0, 0, g_1dLayoutMap.size() - 1);
            int weiIndex1 = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], 0, 0, g_1dLayoutMap.size() - 1);
            int dstIndex1 = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], 0, 0, g_1dLayoutMap.size() - 1);
            KuDNN::Layout srcLay1 = g_1dLayoutMap[srcIndex1].second;
            KuDNN::Layout dstLay1 = g_1dLayoutMap[weiIndex1].second;
            KuDNN::Layout weiLay1 = g_1dLayoutMap[dstIndex1].second;

            int srcIndex2 = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], 0, 0, g_2dLayoutMap.size() - 1);
            int weiIndex2 = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], 0, 0, g_2dLayoutMap.size() - 1);
            int dstIndex2 = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], 0, 0, g_2dLayoutMap.size() - 1);
            KuDNN::Layout srcLay2 = g_2dLayoutMap[srcIndex2].second;
            KuDNN::Layout dstLay2 = g_2dLayoutMap[weiIndex2].second;
            KuDNN::Layout weiLay2 = g_2dLayoutMap[dstIndex2].second;

            int srcIndex3 = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], 0, 0, g_3dLayoutMap.size() - 1);
            int weiIndex3 = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], 0, 0, g_3dLayoutMap.size() - 1);
            int dstIndex3 = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], 0, 0, g_3dLayoutMap.size() - 1);
            KuDNN::Layout srcLay3 = g_3dLayoutMap[srcIndex3].second;
            KuDNN::Layout dstLay3 = g_3dLayoutMap[weiIndex3].second;
            KuDNN::Layout weiLay3 = g_3dLayoutMap[dstIndex3].second;

            const KuDNN::TensorInfo srcTensor1d = {{bc, ic, iw}, srcType, srcLay1};
            const KuDNN::TensorInfo diffWeightsTensor1d = {{oc, ic, kw}, diffWeightsType, weiLay1};
            const KuDNN::TensorInfo diffDstTensor1d = {{bc, oc, ow}, diffDstType, dstLay1};
            const KuDNN::TensorInfo diffBiasTensor1d = {{oc}, diffBiasType, KuDNN::Layout::A};

            const KuDNN::TensorInfo srcTensor2d = {{bc, ic, ih, iw}, srcType, srcLay2};
            const KuDNN::TensorInfo diffWeightsTensor2d = {{oc, ic, kh, kw}, diffWeightsType, weiLay2};
            const KuDNN::TensorInfo diffDstTensor2d = {{bc, oc, oh, ow}, diffDstType, dstLay2};
            const KuDNN::TensorInfo diffBiasTensor2d = {{oc}, diffBiasType, KuDNN::Layout::A};

            const KuDNN::TensorInfo srcTensor3d = {{bc, ic, ih, id, iw}, srcType, srcLay3};
            const KuDNN::TensorInfo diffWeightsTensor3d = {{oc, ic, kd, kh, kw}, diffWeightsType, weiLay3};
            const KuDNN::TensorInfo diffDstTensor3d = {{bc, oc, oh, od, ow}, diffDstType, dstLay3};
            const KuDNN::TensorInfo diffBiasTensor3d = {{oc}, diffBiasType, KuDNN::Layout::A};

            KuDNN::DeconvolutionLayerBWDWeights deconvFwdLayer1d(diffDstTensor1d, srcTensor1d, diffWeightsTensor1d,
                diffBiasTensor1d, strides, dilates, paddingL, paddingR, alg);
            src = malloc(bc * ic * iw * srcElemSz);
            if (src == nullptr) {
                std::cerr << "memory allocation failed" << std::endl;
                return;
            }

            diffWeight = malloc(oc * ic * kw * diffWeightsElemSz);
            if (diffWeight == nullptr) {
                FreeObj(&src);
                std::cerr << "memory allocation failed" << std::endl;
                return;
            }

            diffDst = malloc(bc * oc * ow * diffDstElemSz);
            if (diffDst == nullptr) {
                FreeObj(&src);
                FreeObj(&diffWeight);
                std::cerr << "memory allocation failed" << std::endl;
                return;
            }

            if (hasBias) {
                diffBias = malloc(oc * diffBiasElemSz);
                if (diffBias == nullptr) {
                    FreeObj(&src);
                    FreeObj(&diffWeight);
                    FreeObj(&diffDst);
                    std::cerr << "bias memory allocation failed" << std::endl;
                    return;
                }
            }

            deconvFwdLayer1d.Run(diffDst, src, diffWeight, diffBias);
            FreeObj(&src);
            FreeObj(&diffWeight);
            if (diffBias != nullptr) {
                FreeObj(&diffBias);
            }
            FreeObj(&diffDst);

            KuDNN::DeconvolutionLayerBWDWeights deconvFwdLayer2d(diffDstTensor2d, srcTensor2d, diffWeightsTensor2d,
                diffBiasTensor2d, strides, dilates, paddingL, paddingR, alg);
            src = malloc(bc * ic * ih * iw * srcElemSz);
            if (src == nullptr) {
                std::cerr << "memory allocation failed" << std::endl;
                return;
            }

            diffWeight = malloc(oc * ic * kh * kw * diffWeightsElemSz);
            if (diffWeight == nullptr) {
                FreeObj(&src);
                std::cerr << "memory allocation failed" << std::endl;
                return;
            }

            diffDst = malloc(bc * oc * oh * ow * diffDstElemSz);
            if (diffDst == nullptr) {
                FreeObj(&src);
                FreeObj(&diffWeight);
                std::cerr << "memory allocation failed" << std::endl;
                return;
            }

            if (hasBias) {
                diffBias = malloc(oc * diffBiasElemSz);
                if (diffBias == nullptr) {
                    FreeObj(&src);
                    FreeObj(&diffWeight);
                    FreeObj(&diffDst);
                    std::cerr << "bias memory allocation failed" << std::endl;
                    return;
                }
            }

            deconvFwdLayer2d.Run(diffDst, src, diffWeight, diffBias);
            FreeObj(&src);
            FreeObj(&diffWeight);
            if (diffBias != nullptr) {
                FreeObj(&diffBias);
            }
            FreeObj(&diffDst);
            KuDNN::DeconvolutionLayerBWDWeights deconvFwdLayer3d(diffDstTensor3d, srcTensor3d, diffWeightsTensor3d,
                diffBiasTensor3d, strides, dilates, paddingL, paddingR, alg);
            src = malloc(bc * ic * ih * id * iw * srcElemSz);
            if (src == nullptr) {
                std::cerr << "memory allocation failed" << std::endl;
                return;
            }

            diffWeight = malloc(oc * ic * kd * kh * kw * diffWeightsElemSz);
            if (diffWeight == nullptr) {
                FreeObj(&src);
                std::cerr << "memory allocation failed" << std::endl;
                return;
            }

            diffDst = malloc(bc * oc * od * oh * ow * diffDstElemSz);
            if (diffDst == nullptr) {
                FreeObj(&src);
                FreeObj(&diffWeight);
                std::cerr << "memory allocation failed" << std::endl;
                return;
            }

            if (hasBias) {
                diffBias = malloc(oc * diffBiasElemSz);
                if (diffBias == nullptr) {
                    FreeObj(&src);
                    FreeObj(&diffWeight);
                    FreeObj(&diffDst);
                    std::cerr << "bias memory allocation failed" << std::endl;
                    return;
                }
            }

            deconvFwdLayer3d.Run(diffDst, src, diffWeight, diffBias);
            FreeObj(&src);
            FreeObj(&diffWeight);
            if (diffBias != nullptr) {
                FreeObj(&diffBias);
            }
            FreeObj(&diffDst);
        } catch (const KuDNN::Service::Unsupported& e) {
            if (src != nullptr) {
                FreeObj(&src);
            }
            if (diffWeight != nullptr) {
                FreeObj(&diffWeight);
            }
            if (diffBias != nullptr) {
                FreeObj(&diffBias);
            }
            if (diffDst != nullptr) {
                FreeObj(&diffDst);
            }
            printf("Unsupported Configuration");
        } catch (const KuDNN::Service::LogicError& e) {
            if (src != nullptr) {
                FreeObj(&src);
            }
            if (diffWeight != nullptr) {
                FreeObj(&diffWeight);
            }
            if (diffBias != nullptr) {
                FreeObj(&diffBias);
            }
            if (diffDst != nullptr) {
                FreeObj(&diffDst);
            }
            printf("Bad shapes' arguments");
        }
        printf("\n");
    }
    DT_FUZZ_END()
}

const int  g_deconvFuzzTestCount = 38000;

Cmdinfo g_deconvCmdTable[] = {
    {"DeconvolutionFwd", DtFuzzDeconvolutionFwd, g_deconvFuzzTestCount},
    {"DeconvolutionBwdData", DtFuzzDeconvolutionBwdData, g_deconvFuzzTestCount},
    {"DeconvolutionBwdWeights", DtFuzzDeconvolutionBwdWeights, g_deconvFuzzTestCount},
};

int main(int argc, char *argv[])
{
    DT_Enable_Leak_Check(0,0);

    if (argc > 1) {
        g_deconvCmdTable[0] = {"DeconvolutionFwd", DtFuzzDeconvolutionFwd, atoi(argv[1])};
        g_deconvCmdTable[1] = {"DeconvolutionBwdData", DtFuzzDeconvolutionBwdData, atoi(argv[1])};
        g_deconvCmdTable[2] = {"DeconvolutionBwdWeights", DtFuzzDeconvolutionBwdWeights, atoi(argv[1])};
    }

    int testCount = 0;
    int numOfTestItem = sizeof(g_deconvCmdTable) / sizeof(g_deconvCmdTable[0]);

    if (argc > 2) {
        const char* inputCmd = argv[2];
        for (int testCaseIdx = 0; testCaseIdx < numOfTestItem; testCaseIdx++) {
            if (strcmp(g_deconvCmdTable[testCaseIdx].cmd, inputCmd) == 0) {
                printf("g_deconvCmdTable[%d] = %s test %d time\n", testCaseIdx, g_deconvCmdTable[testCaseIdx].cmd,
                    testCount);
                testCount = g_deconvCmdTable[testCaseIdx].testCount;
                g_deconvCmdTable[testCaseIdx].func(testCount);
                break;
            }
        }
    } else {
        for (int testCaseIdx = 0; testCaseIdx < numOfTestItem; testCaseIdx++) {
            testCount = g_deconvCmdTable[testCaseIdx].testCount;
            printf("g_deconvCmdTable[%d] = %s test %d time\n", testCaseIdx, g_deconvCmdTable[testCaseIdx].cmd, testCount);
            g_deconvCmdTable[testCaseIdx].func(testCount);
        }
    }

    return 0;
}