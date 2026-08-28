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
};

struct LayOutTypeFuzz {
    KuDNN::Layout src;
    KuDNN::Layout stat;
    KuDNN::Layout scale;
    KuDNN::Layout dst;
};
static std::map<int, std::pair<std::string, LayOutTypeFuzz>> g_LayOutType = {
    {INDEX0, {"ABC,AB,A,ABC", {KuDNN::Layout::ABC, KuDNN::Layout::AB, KuDNN::Layout::A, KuDNN::Layout::ABC}}},
    {INDEX1, {"BAC,BA,A,BAC", {KuDNN::Layout::BAC, KuDNN::Layout::BA, KuDNN::Layout::A, KuDNN::Layout::BAC}}},
    {INDEX2, {"CAB,AB,A,CAB", {KuDNN::Layout::CAB, KuDNN::Layout::AB, KuDNN::Layout::A, KuDNN::Layout::CAB}}},
    {INDEX3, {"CBA,BA,A,CBA", {KuDNN::Layout::CBA, KuDNN::Layout::BA, KuDNN::Layout::A, KuDNN::Layout::CBA}}},
    {INDEX4, {"bac,ab,a,bac", {KuDNN::Layout::BAC, KuDNN::Layout::AB, KuDNN::Layout::A, KuDNN::Layout::BAC}}},
};

struct LayOutTypeFuzzBwd {
    KuDNN::Layout src;
    KuDNN::Layout stat;
    KuDNN::Layout dsrc;
    KuDNN::Layout ddst;
};

static std::map<int, std::pair<std::string, LayOutTypeFuzzBwd>> g_LayOutTypeBwd = {
    {INDEX0, {"ABC,AB,A,ABC", {KuDNN::Layout::ACB, KuDNN::Layout::AB, KuDNN::Layout::ACB, KuDNN::Layout::ACB}}},
    {INDEX1, {"BAC,BA,A,ABC", {KuDNN::Layout::ACB, KuDNN::Layout::BA, KuDNN::Layout::ACB, KuDNN::Layout::ACB}}},
};

static std::map<int, std::pair<std::string, std::uint32_t>> g_flagsMap = {
    {INDEX0, {"None", static_cast<std::uint32_t>(KuDNN::NormalizationFlags::NONE)}},
    {INDEX1, {"G", static_cast<std::uint32_t>(KuDNN::NormalizationFlags::USE_GLOBAL_STATS)}},
    {INDEX2, {"C", static_cast<std::uint32_t>(KuDNN::NormalizationFlags::USE_SCALE)}},
    {INDEX3, {"H", static_cast<std::uint32_t>(KuDNN::NormalizationFlags::USE_SHIFT)}},
    {INDEX4, {"CH", static_cast<std::uint32_t>(KuDNN::NormalizationFlags::USE_SCALE) |
        static_cast<std::uint32_t>(KuDNN::NormalizationFlags::USE_SHIFT)}},
    {INDEX5, {"GCH", static_cast<std::uint32_t>(KuDNN::NormalizationFlags::USE_GLOBAL_STATS) |
        static_cast<std::uint32_t>(KuDNN::NormalizationFlags::USE_SCALE) |
        static_cast<std::uint32_t>(KuDNN::NormalizationFlags::USE_SHIFT)}},
};

static constexpr int g_dimRangeInitVal = 2;
static constexpr int g_dimRangeLowerBound = 2;
static constexpr int g_dimRangeUpperBound = 30;

static void DtNormalizationLayerFwd(int counts)
{
    char funcName[] = "DtNormalizationLayerFwd";
    DT_FUZZ_START(0, counts, funcName, 0)
    {
        int eleIndex = 0;
        int srcInfoElemTypeIndex = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], 0, 0,
            g_precisionMap.size() - 1);
        int statInfoElemTypeIndex = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], 0, 0,
            g_precisionMap.size() - 1);
        int scaleShiftInfoElemTypeIndex = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], 0, 0,
            g_precisionMap.size() - 1);
        int dstInfoElemTypeIndex = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], 0, 0,
            g_precisionMap.size() - 1);
        int flagsTypeIndex = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], 0, 0, g_flagsMap.size() - 1);
        size_t dim0 = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], g_dimRangeInitVal, g_dimRangeLowerBound,
            g_dimRangeUpperBound);
        size_t dim1 = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], g_dimRangeInitVal, g_dimRangeLowerBound,
            g_dimRangeUpperBound);
        size_t dim2 = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], g_dimRangeInitVal, g_dimRangeLowerBound,
            g_dimRangeUpperBound);

        KuDNN::Element::Type srcInfoType(g_precisionMap[srcInfoElemTypeIndex].second);
        KuDNN::Element::Type statInfoType(g_precisionMap[statInfoElemTypeIndex].second);
        KuDNN::Element::Type scaleShiftInfoType(g_precisionMap[scaleShiftInfoElemTypeIndex].second);
        KuDNN::Element::Type dstInfoType(g_precisionMap[dstInfoElemTypeIndex].second);

        int layoutElemTypeIndex = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], 0, 0, g_LayOutType.size() - 1);
        LayOutTypeFuzz layout(g_LayOutType[layoutElemTypeIndex].second);

        size_t srcInfoElemSz = srcInfoType.GetSize();
        size_t scaleShiftInfoElemSz = scaleShiftInfoType.GetSize();
        size_t dstInfoElemSz = dstInfoType.GetSize();

        KuDNN::NormalizationFlags lnrmFlag1(static_cast<KuDNN::NormalizationFlags>(g_flagsMap[flagsTypeIndex].second));
        lnrmFlag1 =  lnrmFlag1 & lnrmFlag1;
        lnrmFlag1 =  lnrmFlag1 | lnrmFlag1;
        lnrmFlag1 |=  lnrmFlag1;
        lnrmFlag1 &=  lnrmFlag1;

        KuDNN::NormalizationFlags lnrmFlag(static_cast<KuDNN::NormalizationFlags>(g_flagsMap[flagsTypeIndex].second));
        const KuDNN::TensorInfo srcInfoTensorInfo = {{dim0, dim1, dim2}, srcInfoType, layout.src};
        const KuDNN::TensorInfo statInfoTensorInfo = {{dim0, dim1}, statInfoType, layout.stat};
        const KuDNN::TensorInfo scaleShiftInfoTensorInfo = {{dim2}, scaleShiftInfoType, layout.scale};
        const KuDNN::TensorInfo dstInfoTensorInfo = {{dim0, dim1, dim2}, dstInfoType, layout.dst};

        char format[] = "DtLnormFwd: Precision(srcInfo:statInfo:scaleShiftInfo:dstInfo): %s:%s:%s:%s "
            " Shape: %zux%zux%zu, flags=%s layout:%s";
        printf(format, g_precisionMap[srcInfoElemTypeIndex].first.c_str(),
            g_precisionMap[statInfoElemTypeIndex].first.c_str(),
            g_precisionMap[scaleShiftInfoElemTypeIndex].first.c_str(),
            g_precisionMap[dstInfoElemTypeIndex].first.c_str(), dim0, dim1, dim2,
            g_flagsMap[flagsTypeIndex].first.c_str(),
            g_LayOutType[layoutElemTypeIndex].first.c_str());

        void *srcInfo = nullptr;
        void *scaleInfo = nullptr;
        void *shiftInfo = nullptr;
        void *dstInfo = nullptr;
        float *mean = nullptr;
        float *variance = nullptr;

        try {
            KuDNN::NormalizationLayerFWD norlayerfwd1(srcInfoTensorInfo, statInfoTensorInfo, scaleShiftInfoTensorInfo,
                dstInfoTensorInfo, lnrmFlag);
            KuDNN::NormalizationLayerFWD norlayerfwd2(norlayerfwd1);
            norlayerfwd2 = norlayerfwd1;

            KuDNN::NormalizationLayerFWD norlayerfwd3(std::move(norlayerfwd1));
            KuDNN::NormalizationLayerFWD norlayerfwd4(norlayerfwd2);
            norlayerfwd4 = std::move(norlayerfwd2);

            KuDNN::NormalizationLayerFWD NorLayerFwd(srcInfoTensorInfo, statInfoTensorInfo, scaleShiftInfoTensorInfo,
                dstInfoTensorInfo, lnrmFlag);
            srcInfo = malloc(std::max((size_t)1, dim0) * std::max((size_t)1, dim1) * std::max((size_t)1, dim2) *
                srcInfoElemSz);
            if (srcInfo == nullptr) {
                std::cerr<<"memory allocation failed"<<std::endl;
                return;
            }

            scaleInfo = malloc(std::max((size_t)1, dim2) * scaleShiftInfoElemSz);
            if (scaleInfo == nullptr) {
                free(srcInfo);
                std::cerr<<"memory allocation failed"<<std::endl;
                return;
            }

            shiftInfo = malloc(std::max((size_t)1, dim2) * scaleShiftInfoElemSz);
            if (shiftInfo == nullptr) {
                free(srcInfo);
                free(scaleInfo);
                std::cerr<<"memory allocation failed"<<std::endl;
                return;
            }

            dstInfo = malloc(std::max((size_t)1, dim0) * std::max((size_t)1, dim1) * std::max((size_t)1, dim2) *
                dstInfoElemSz);
            if (dstInfo == nullptr) {
                free(srcInfo);
                free(scaleInfo);
                free(shiftInfo);
                std::cerr<<"memory allocation failed"<<std::endl;
                return;
            }

            mean = static_cast<float *>(malloc(std::max((size_t)1, dim0) * std::max((size_t)1, dim1) * sizeof(float)));
            if (mean == nullptr) {
                free(srcInfo);
                free(scaleInfo);
                free(shiftInfo);
                free(dstInfo);
                return;
            }

            variance = static_cast<float *>(malloc(std::max((size_t)1, dim0) * std::max((size_t)1, dim1) * sizeof(float)));
            if (variance == nullptr) {
                free(srcInfo);
                free(scaleInfo);
                free(shiftInfo);
                free(dstInfo);
                free(mean);
                return;
            }

            NorLayerFwd.Run(srcInfo, dstInfo, scaleInfo, shiftInfo, mean, variance, true, 1.e-10f);
            free(srcInfo);
            free(scaleInfo);
            free(shiftInfo);
            free(dstInfo);
            free(mean);
            free(variance);
        } catch (const KuDNN::Service::Unsupported& e) {
            if (srcInfo != nullptr) {
                free(srcInfo);
            }
            if (scaleInfo != nullptr) {
                free(scaleInfo);
            }
            if (shiftInfo != nullptr) {
                free(shiftInfo);
            }
            if (dstInfo != nullptr) {
                free(dstInfo);
            }
            if (mean != nullptr) {
                free(mean);
            }
            if (variance != nullptr) {
                free(variance);
            }
            printf("Unsupported Configuration");
        } catch (const KuDNN::Service::LogicError& e) {
            if (srcInfo != nullptr) {
                free(srcInfo);
            }
            if (scaleInfo != nullptr) {
                free(scaleInfo);
            }
            if (shiftInfo != nullptr) {
                free(shiftInfo);
            }
            if (dstInfo != nullptr) {
                free(dstInfo);
            }
            if (mean != nullptr) {
                free(mean);
            }
            if (variance != nullptr) {
                free(variance);
            }
            printf("LogicError");
        }
        printf("\n");
    }
    DT_FUZZ_END()
}

static void DtNormalizationLayerFwdCrossLayout(int counts)
{
    char funcName[] = "DtNormalizationLayerFwdCrossLayout";
    DT_FUZZ_START(0, counts, funcName, 0)
    {
        int eleIndex = 0;
        int srcInfoElemTypeIndex = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], 0, 0,
            g_precisionMap.size() - 1);
        int statInfoElemTypeIndex = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], 0, 0,
            g_precisionMap.size() - 1);
        int scaleShiftInfoElemTypeIndex = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], 0, 0,
            g_precisionMap.size() - 1);
        int dstInfoElemTypeIndex = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], 0, 0,
            g_precisionMap.size() - 1);
        int flagsTypeIndex = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], 0, 0, g_flagsMap.size() - 1);
        size_t dim0 = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], g_dimRangeInitVal, g_dimRangeLowerBound,
            g_dimRangeUpperBound);
        size_t dim1 = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], g_dimRangeInitVal, g_dimRangeLowerBound,
            g_dimRangeUpperBound);
        size_t dim2 = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], g_dimRangeInitVal, g_dimRangeLowerBound,
            g_dimRangeUpperBound);

        KuDNN::Element::Type srcInfoType(g_precisionMap[srcInfoElemTypeIndex].second);
        KuDNN::Element::Type statInfoType(g_precisionMap[statInfoElemTypeIndex].second);
        KuDNN::Element::Type scaleShiftInfoType(g_precisionMap[scaleShiftInfoElemTypeIndex].second);
        KuDNN::Element::Type dstInfoType(g_precisionMap[dstInfoElemTypeIndex].second);
        size_t srcInfoElemSz = srcInfoType.GetSize();
        size_t scaleShiftInfoElemSz = scaleShiftInfoType.GetSize();
        size_t dstInfoElemSz = dstInfoType.GetSize();
        KuDNN::NormalizationFlags lnrmFlag(static_cast<KuDNN::NormalizationFlags>(g_flagsMap[flagsTypeIndex].second));

        const KuDNN::TensorInfo srcInfoTensorInfo = {{dim0, dim1, dim2}, srcInfoType, KuDNN::Layout::ACB};
        const KuDNN::TensorInfo statInfoTensorInfo = {{dim0, dim1}, statInfoType, KuDNN::Layout::AB};
        const KuDNN::TensorInfo scaleShiftInfoTensorInfo = {{dim2}, scaleShiftInfoType, KuDNN::Layout::A};
        const KuDNN::TensorInfo dstInfoTensorInfo = {{dim0, dim1, dim2}, dstInfoType, KuDNN::Layout::ACB};
        char format[] = "DtLnormFwdCrossLayout: Precision(srcInfo:statInfo:scaleShiftInfo:dstInfo): %s:%s:%s:%s"
            "Shape: %zux%zux%zu, flags = %s ";
        printf(format, g_precisionMap[srcInfoElemTypeIndex].first.c_str(),
            g_precisionMap[statInfoElemTypeIndex].first.c_str(),
            g_precisionMap[scaleShiftInfoElemTypeIndex].first.c_str(),
            g_precisionMap[dstInfoElemTypeIndex].first.c_str(), dim0, dim1, dim2,
            g_flagsMap[flagsTypeIndex].first.c_str());
        void *srcInfo = nullptr;
        void *scaleInfo = nullptr;
        void *shiftInfo = nullptr;
        void *dstInfo = nullptr;
        float *mean = nullptr;
        float *variance = nullptr;
        try {
            KuDNN::NormalizationLayerFWD NorLayer1(srcInfoTensorInfo, statInfoTensorInfo, scaleShiftInfoTensorInfo,
                dstInfoTensorInfo, lnrmFlag);
            KuDNN::NormalizationLayerFWD NorLayer2(NorLayer1);
            NorLayer2 = NorLayer1;
            KuDNN::NormalizationLayerFWD NorLayer3(NorLayer1);
            NorLayer3 = std::move(NorLayer1);
            KuDNN::NormalizationLayerFWD NorLayer5(std::move(NorLayer2));

            KuDNN::NormalizationLayerFWD NorLayerFwd(srcInfoTensorInfo, statInfoTensorInfo, scaleShiftInfoTensorInfo,
                dstInfoTensorInfo, lnrmFlag);
            srcInfo = malloc(std::max((size_t)1, dim0) * std::max((size_t)1, dim1) * std::max((size_t)1, dim2) *
                srcInfoElemSz);
            if (srcInfo == nullptr) {
                std::cerr<<"memory allocation failed"<<std::endl;
                return;
            }

            scaleInfo = malloc(std::max((size_t)1, dim2) * scaleShiftInfoElemSz);
            if (scaleInfo == nullptr) {
                free(srcInfo);
                std::cerr<<"memory allocation failed"<<std::endl;
                return;
            }

            shiftInfo = malloc(std::max((size_t)1, dim2) * scaleShiftInfoElemSz);
            if (shiftInfo == nullptr) {
                free(srcInfo);
                free(scaleInfo);
                std::cerr<<"memory allocation failed"<<std::endl;
                return;
            }

            dstInfo = malloc(std::max((size_t)1, dim0) * std::max((size_t)1, dim1) * std::max((size_t)1, dim2) *
                dstInfoElemSz);
            if (dstInfo == nullptr) {
                free(srcInfo);
                free(scaleInfo);
                free(shiftInfo);
                std::cerr<<"memory allocation failed"<<std::endl;
                return;
            }

            mean = static_cast<float *>(malloc(std::max((size_t)1, dim0) * std::max((size_t)1, dim1) * sizeof(float)));
            if (mean == nullptr) {
                free(srcInfo);
                free(scaleInfo);
                free(shiftInfo);
                free(dstInfo);
                return;
            }

            variance = static_cast<float *>(malloc(std::max((size_t)1, dim0) * std::max((size_t)1, dim1) * sizeof(float)));
            if (variance == nullptr) {
                free(srcInfo);
                free(scaleInfo);
                free(shiftInfo);
                free(dstInfo);
                free(mean);
                return;
            }

            NorLayerFwd.Run(srcInfo, dstInfo, scaleInfo, shiftInfo, mean, variance, true, 1.e-10f);
            free(srcInfo);
            free(scaleInfo);
            free(shiftInfo);
            free(dstInfo);
            free(mean);
            free(variance);
        } catch (const KuDNN::Service::Unsupported& e) {
            if (srcInfo != nullptr) {
                free(srcInfo);
            }
            if (scaleInfo != nullptr) {
                free(scaleInfo);
            }
            if (shiftInfo != nullptr) {
                free(shiftInfo);
            }
            if (dstInfo != nullptr) {
                free(dstInfo);
            }
            if (mean != nullptr) {
                free(mean);
            }
            if (variance != nullptr) {
                free(variance);
            }
            printf("Unsupported Configuration");
        } catch (const KuDNN::Service::LogicError& e) {
            if (srcInfo != nullptr) {
                free(srcInfo);
            }
            if (scaleInfo != nullptr) {
                free(scaleInfo);
            }
            if (shiftInfo != nullptr) {
                free(shiftInfo);
            }
            if (dstInfo != nullptr) {
                free(dstInfo);
            }
            if (mean != nullptr) {
                free(mean);
            }
            if (variance != nullptr) {
                free(variance);
            }
            printf("LogicError");
        }
        printf("\n");
    }
    DT_FUZZ_END()
}

static void DtNormalizationLayerBwd(int counts)
{
    char funcName[] = "DtNormalizationLayerBwd";
    DT_FUZZ_START(0, counts, funcName, 0)
    {
        int eleIndex = 0;
        int srcInfoElemTypeIndex = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], 0, 0,
            g_precisionMap.size() -1);
        int flagsTypeIndex = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], 0, 0, g_flagsMap.size() - 1);
        size_t dim0 = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], g_dimRangeInitVal, g_dimRangeLowerBound,
            g_dimRangeUpperBound);
        size_t dim1 = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], g_dimRangeInitVal, g_dimRangeLowerBound,
            g_dimRangeUpperBound);
        size_t dim2 = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], g_dimRangeInitVal, g_dimRangeLowerBound,
            g_dimRangeUpperBound);

        KuDNN::Element::Type srcInfoType(g_precisionMap[srcInfoElemTypeIndex].second);
        KuDNN::Element::Type diffSrcInfoType(srcInfoType);
        KuDNN::Element::Type diffDstInfoType(srcInfoType);
        KuDNN::Element::Type scInfoType(srcInfoType);
        KuDNN::Element::Type diffScInfoType(srcInfoType);
        size_t srcInfoElemSz = srcInfoType.GetSize();
        size_t diffSrcInfoElemSz = diffSrcInfoType.GetSize();
        size_t diffDstInfoElemSz = diffDstInfoType.GetSize();
        size_t scInfoElemSz = scInfoType.GetSize();
        size_t diffScInfoElemSz = diffScInfoType.GetSize();

        KuDNN::NormalizationFlags lnrmFlag(static_cast<KuDNN::NormalizationFlags>(g_flagsMap[flagsTypeIndex].second));

        const KuDNN::TensorInfo srcInfoTensorInfo = {{dim0, dim1, dim2}, srcInfoType, KuDNN::Layout::ABC};
        const KuDNN::TensorInfo statInfoTensorInfo = {{dim0, dim1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::AB};
        const KuDNN::TensorInfo diffSrcInfoTensorInfo = {{dim0, dim1, dim2}, diffSrcInfoType, KuDNN::Layout::ABC};
        const KuDNN::TensorInfo diffDstInfoTensorInfo = {{dim0, dim1, dim2}, diffDstInfoType, KuDNN::Layout::ABC};
        const KuDNN::TensorInfo scInfoTensorInfo = {{dim2}, scInfoType, KuDNN::Layout::A};
        const KuDNN::TensorInfo diffScInfoTensorInfo = {{dim2}, diffScInfoType, KuDNN::Layout::A};
        char format[] = "DtLnormBwd: Precision(srcInfo:statInfo:diffSrcInfo:diffDstInfo:scInfo:diffScInfo)"
            " %s:%s:%s:%s:%s:%s, Shape: %zux%zux%zu ";
        printf(format, g_precisionMap[srcInfoElemTypeIndex].first.c_str(),
            g_precisionMap[srcInfoElemTypeIndex].first.c_str(), "F32",
            g_precisionMap[srcInfoElemTypeIndex].first.c_str(), g_precisionMap[srcInfoElemTypeIndex].first.c_str(),
            g_precisionMap[srcInfoElemTypeIndex].first.c_str(), dim0, dim1, dim2);
        void *src = nullptr;
        float *mean = nullptr;
        float *variance = nullptr;
        void *diffDst = nullptr;
        void *scale = nullptr;
        void *diffSrc = nullptr;
        void *diffScale = nullptr;
        void *diffShift = nullptr;
        try {
            KuDNN::NormalizationLayerBWD norLayer1(srcInfoTensorInfo, statInfoTensorInfo, diffSrcInfoTensorInfo,
                diffDstInfoTensorInfo, scInfoTensorInfo, diffScInfoTensorInfo, lnrmFlag);
            KuDNN::NormalizationLayerBWD norLayer2(norLayer1);
            norLayer2 = norLayer1;
            KuDNN::NormalizationLayerBWD norLayer3(norLayer1);
            norLayer3 = std::move(norLayer1);
            KuDNN::NormalizationLayerBWD norLayer5(std::move(norLayer2));

            KuDNN::NormalizationLayerBWD NorLayerBwd(srcInfoTensorInfo, statInfoTensorInfo, diffSrcInfoTensorInfo,
                diffDstInfoTensorInfo, scInfoTensorInfo, diffScInfoTensorInfo, lnrmFlag);
            src = malloc(std::max((size_t)1, dim0) * std::max((size_t)1, dim1) * std::max((size_t)1, dim2) *
                srcInfoElemSz);
            if (src == nullptr) {
                std::cerr<<"memory allocation failed"<<std::endl;
                return;
            }

            mean = (float *)malloc(std::max((size_t)1, dim0) * std::max((size_t)1, dim1) * sizeof(float));
            if (mean == nullptr) {
                free(src);
                std::cerr<<"memory allocation failed"<<std::endl;
                return;
            }

            variance = (float *)malloc(std::max((size_t)1, dim0) * std::max((size_t)1, dim1) * sizeof(float));
            if (variance == nullptr) {
                free(src);
                free(mean);
                std::cerr<<"memory allocation failed"<<std::endl;
                return;
            }

            diffDst = malloc(std::max((size_t)1, dim0) * std::max((size_t)1, dim1) * std::max((size_t)1, dim2) *
                diffDstInfoElemSz);

            if (diffDst == nullptr) {
                free(src);
                free(mean);
                free(variance);
                std::cerr<<"memory allocation failed"<<std::endl;
                return;
            }

            scale = malloc(std::max((size_t)1, dim2) * scInfoElemSz);
            if (scale == nullptr) {
                free(src);
                free(mean);
                free(variance);
                free(diffDst);
                std::cerr<<"memory allocation failed"<<std::endl;
                return;
            }

            diffSrc = malloc(std::max((size_t)1, dim0) * std::max((size_t)1, dim1) * std::max((size_t)1, dim2) *
                diffSrcInfoElemSz);
            if (diffSrc == nullptr) {
                free(src);
                free(mean);
                free(variance);
                free(diffDst);
                free(scale);
                std::cerr<<"memory allocation failed"<<std::endl;
                return;
            }

            diffScale = malloc(std::max((size_t)1, dim2) * diffScInfoElemSz);
            if (diffScale == nullptr) {
                free(src);
                free(mean);
                free(variance);
                free(diffDst);
                free(scale);
                free(diffSrc);
                std::cerr<<"memory allocation failed"<<std::endl;
                return;
            }

            diffShift = malloc(std::max((size_t)1, dim2) * diffScInfoElemSz);
            if (diffShift == nullptr) {
                free(src);
                free(mean);
                free(variance);
                free(diffDst);
                free(scale);
                free(diffSrc);
                free(diffScale);
                std::cerr<<"memory allocation failed"<<std::endl;
                return;
            }

            NorLayerBwd.Run(src, mean, variance, diffDst, scale, diffSrc, diffScale, diffShift, 1.e-10f);
            free(src);
            free(mean);
            free(variance);
            free(diffDst);
            free(scale);
            free(diffSrc);
            free(diffScale);
            free(diffShift);
        } catch (const KuDNN::Service::Unsupported& e) {
            if (src != nullptr) {
                free(src);
            }
            if (mean != nullptr) {
                free(mean);
            }
            if (variance != nullptr) {
                free(variance);
            }
            if (diffDst != nullptr) {
                free(diffDst);
            }
            if (scale != nullptr) {
                free(scale);
            }
            if (diffSrc != nullptr) {
                free(diffSrc);
            }
            if (diffScale != nullptr) {
                free(diffScale);
            }
            if (diffShift != nullptr) {
                free(diffShift);
            }
            printf("Unsupported Configuration");
        } catch (const KuDNN::Service::LogicError& e) {
            if (src != nullptr) {
                free(src);
            }
            if (mean != nullptr) {
                free(mean);
            }
            if (variance != nullptr) {
                free(variance);
            }
            if (diffDst != nullptr) {
                free(diffDst);
            }
            if (scale != nullptr) {
                free(scale);
            }
            if (diffSrc != nullptr) {
                free(diffSrc);
            }
            if (diffScale != nullptr) {
                free(diffScale);
            }
            if (diffShift != nullptr) {
                free(diffShift);
            }
            printf("LogicError");
        }
        printf("\n");
    }
    DT_FUZZ_END()
}

static void DtNormalizationLayerBwdCrossLayout(int counts)
{
    char funcName[] = "DtNormalizationLayerBwdCrossLayout";
    DT_FUZZ_START(0, counts, funcName, 0)
    {
        int eleIndex = 0;
        int srcInfoElemTypeIndex = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], 0, 0,
            g_precisionMap.size() -1);
        int flagsTypeIndex = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], 0, 0, g_flagsMap.size() - 1);
        size_t dim0 = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], g_dimRangeInitVal, g_dimRangeLowerBound,
            g_dimRangeUpperBound);
        size_t dim1 = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], g_dimRangeInitVal, g_dimRangeLowerBound,
            g_dimRangeUpperBound);
        size_t dim2 = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], g_dimRangeInitVal, g_dimRangeLowerBound,
            g_dimRangeUpperBound);
        int dstInfoElemTypeIndex = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], g_dimRangeInitVal, g_dimRangeLowerBound,
            g_precisionMap.size() -1);
        KuDNN::Element::Type srcInfoType(g_precisionMap[srcInfoElemTypeIndex].second);
        KuDNN::Element::Type diffSrcInfoType(srcInfoType);
        KuDNN::Element::Type diffDstInfoType(g_precisionMap[dstInfoElemTypeIndex].second);
        KuDNN::Element::Type scInfoType(srcInfoType);
        KuDNN::Element::Type diffScInfoType(srcInfoType);
        size_t srcInfoElemSz = srcInfoType.GetSize();
        size_t diffSrcInfoElemSz = diffSrcInfoType.GetSize();
        size_t diffDstInfoElemSz = diffDstInfoType.GetSize();
        size_t scInfoElemSz = scInfoType.GetSize();
        size_t diffScInfoElemSz = diffScInfoType.GetSize();
        KuDNN::NormalizationFlags lnrmFlag(static_cast<KuDNN::NormalizationFlags>(g_flagsMap[flagsTypeIndex].second));
        int layoutElemTypeIndex = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], 0, 0,
            g_LayOutTypeBwd.size() - 1);
        LayOutTypeFuzzBwd layout(g_LayOutTypeBwd[layoutElemTypeIndex].second);
        const KuDNN::TensorInfo srcInfoTensorInfo = {{dim0, dim1, dim2}, srcInfoType, layout.src};
        const KuDNN::TensorInfo statInfoTensorInfo = {{dim0, dim1}, KuDNN::Element::TypeT::F32, layout.stat};
        const KuDNN::TensorInfo diffSrcInfoTensorInfo = {{dim0, dim1, dim2}, diffSrcInfoType, layout.dsrc};
        const KuDNN::TensorInfo diffDstInfoTensorInfo = {{dim0, dim1, dim2}, diffDstInfoType, layout.ddst};

        const KuDNN::TensorInfo scInfoTensorInfo = {{dim2}, scInfoType, KuDNN::Layout::A};
        const KuDNN::TensorInfo diffScInfoTensorInfo = {{dim2}, diffScInfoType, KuDNN::Layout::A};
        char format[] = "DtLnormBwdCrossLayout: Precision(srcInfo:statInfo:diffSrcInfo:diffDstInfo:scInfo:diffScInfo)"
            " %s:%s:%s:%s:%s:%s, Shape: %zux%zux%zu ";
        printf(format, g_precisionMap[srcInfoElemTypeIndex].first.c_str(),
            g_precisionMap[srcInfoElemTypeIndex].first.c_str(), "F32",
            g_precisionMap[srcInfoElemTypeIndex].first.c_str(), g_precisionMap[srcInfoElemTypeIndex].first.c_str(),
            g_precisionMap[srcInfoElemTypeIndex].first.c_str(), dim0, dim1, dim2);
        void *src = nullptr;
        float *mean = nullptr;
        float *variance = nullptr;
        void *diffDst = nullptr;
        void *scale = nullptr;
        void *diffSrc = nullptr;
        void *diffScale = nullptr;
        void *diffShift = nullptr;
        try {
            KuDNN::NormalizationLayerBWD norLayer1(srcInfoTensorInfo, statInfoTensorInfo, diffSrcInfoTensorInfo,
                diffDstInfoTensorInfo, scInfoTensorInfo, diffScInfoTensorInfo, lnrmFlag);
            KuDNN::NormalizationLayerBWD norLayer2(norLayer1);
            norLayer2 = norLayer1;
            KuDNN::NormalizationLayerBWD norLayer3(norLayer1);
            KuDNN::NormalizationLayerBWD norLayer4 = std::move(norLayer1);
            KuDNN::NormalizationLayerBWD norLayerBwd5(std::move(norLayer1));

            KuDNN::NormalizationLayerBWD NorLayerBwd(srcInfoTensorInfo, statInfoTensorInfo, diffSrcInfoTensorInfo,
                diffDstInfoTensorInfo, scInfoTensorInfo, diffScInfoTensorInfo, lnrmFlag);
            src = malloc(std::max((size_t)1, dim0) * std::max((size_t)1, dim1) * std::max((size_t)1, dim2) *
                srcInfoElemSz);
            if (src == nullptr) {
                std::cerr<<"memory allocation failed"<<std::endl;
                return;
            }

            mean = (float *)malloc(std::max((size_t)1, dim0) * std::max((size_t)1, dim1) * sizeof(float));
            if (mean == nullptr) {
                free(src);
                std::cerr<<"memory allocation failed"<<std::endl;
                return;
            }

            variance = (float *)malloc(std::max((size_t)1, dim0) * std::max((size_t)1, dim1) * sizeof(float));
            if (variance == nullptr) {
                free(src);
                free(mean);
                std::cerr<<"memory allocation failed"<<std::endl;
                return;
            }

            diffDst = malloc(std::max((size_t)1, dim0) * std::max((size_t)1, dim1) * std::max((size_t)1, dim2) *
                diffDstInfoElemSz);

            if (diffDst == nullptr) {
                free(src);
                free(mean);
                free(variance);
                std::cerr<<"memory allocation failed"<<std::endl;
                return;
            }

            scale = malloc(std::max((size_t)1, dim2) * scInfoElemSz);
            if (scale == nullptr) {
                free(src);
                free(mean);
                free(variance);
                free(diffDst);
                std::cerr<<"memory allocation failed"<<std::endl;
                return;
            }

            diffSrc = malloc(std::max((size_t)1, dim0) * std::max((size_t)1, dim1) * std::max((size_t)1, dim2) *
                diffSrcInfoElemSz);
            if (diffSrc == nullptr) {
                free(src);
                free(mean);
                free(variance);
                free(diffDst);
                free(scale);
                std::cerr<<"memory allocation failed"<<std::endl;
                return;
            }

            diffScale = malloc(std::max((size_t)1, dim2) * diffScInfoElemSz);
            if (diffScale == nullptr) {
                free(src);
                free(mean);
                free(variance);
                free(diffDst);
                free(scale);
                free(diffSrc);
                std::cerr<<"memory allocation failed"<<std::endl;
                return;
            }

            diffShift = malloc(std::max((size_t)1, dim2) * diffScInfoElemSz);
            if (diffShift == nullptr) {
                free(src);
                free(mean);
                free(variance);
                free(diffDst);
                free(scale);
                free(diffSrc);
                free(diffScale);
                std::cerr<<"memory allocation failed"<<std::endl;
                return;
            }

            NorLayerBwd.Run(src, mean, variance, diffDst, scale, diffSrc, diffScale, diffShift, 1.e-10f);
            free(src);
            free(mean);
            free(variance);
            free(diffDst);
            free(scale);
            free(diffSrc);
            free(diffScale);
            free(diffShift);
        } catch (const KuDNN::Service::Unsupported& e) {
            if (src != nullptr) {
                free(src);
            }
            if (mean != nullptr) {
                free(mean);
            }
            if (variance != nullptr) {
                free(variance);
            }
            if (diffDst != nullptr) {
                free(diffDst);
            }
            if (scale != nullptr) {
                free(scale);
            }
            if (diffSrc != nullptr) {
                free(diffSrc);
            }
            if (diffScale != nullptr) {
                free(diffScale);
            }
            if (diffShift != nullptr) {
                free(diffShift);
            }
            printf("Unsupported Configuration");
        } catch (const KuDNN::Service::LogicError& e) {
            if (src != nullptr) {
                free(src);
            }
            if (mean != nullptr) {
                free(mean);
            }
            if (variance != nullptr) {
                free(variance);
            }
            if (diffDst != nullptr) {
                free(diffDst);
            }
            if (scale != nullptr) {
                free(scale);
            }
            if (diffSrc != nullptr) {
                free(diffSrc);
            }
            if (diffScale != nullptr) {
                free(diffScale);
            }
            if (diffShift != nullptr) {
                free(diffShift);
            }
            printf("LogicError");
        }
        printf("\n");
    }
    DT_FUZZ_END()
}

static constexpr int g_lnFuzzTestCount = 8000;

static Cmdinfo g_lnCmdTable[] = {
    {"Nomalization_Layer_Fwd", DtNormalizationLayerFwd, g_lnFuzzTestCount},
    {"Nomalization_Layer_Fwd_Cross_Layout", DtNormalizationLayerFwdCrossLayout, g_lnFuzzTestCount},
    {"Nomalization_Layer_Bwd", DtNormalizationLayerBwd, g_lnFuzzTestCount},
    {"Nomalization_Layer_Bwd_Cross_Layout", DtNormalizationLayerBwdCrossLayout, g_lnFuzzTestCount}
};

int main(int argc, char *argv[])
{
    DT_Enable_Leak_Check(0, 0);
    if (argc > 1) {
        g_lnCmdTable[0] = {"Nomalization_Layer_Fwd", DtNormalizationLayerFwd, atoi(argv[1])};
        g_lnCmdTable[1] = {"Nomalization_Layer_Fwd_Cross_Layout", DtNormalizationLayerFwdCrossLayout, atoi(argv[1])};
        g_lnCmdTable[2] = {"Nomalization_Layer_Bwd", DtNormalizationLayerBwd, atoi(argv[1])};
        g_lnCmdTable[3] = {"Nomalization_Layer_Bwd_Cross_Layout", DtNormalizationLayerBwdCrossLayout, atoi(argv[1])};
    }

    int numOfTestItem = sizeof(g_lnCmdTable) / sizeof(g_lnCmdTable[0]);
    int testCount = 0;

    for (int testCaseIdx = 0; testCaseIdx < numOfTestItem; testCaseIdx++) {
        printf("g_lnCmdTable[%d] = %s test %d time\n", testCaseIdx, g_lnCmdTable[testCaseIdx].cmd, testCount);

        testCount = g_lnCmdTable[testCaseIdx].testCount;
        g_lnCmdTable[testCaseIdx].func(testCount);
    }
    return 0;
}