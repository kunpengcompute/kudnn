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
    {INDEX3, {"U8", KuDNN::Element::TypeT::U8}},
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

static constexpr int g_groupRangeInitVal = 1;
static constexpr int g_groupRangeLowerBound = 1;
static constexpr int g_groupRangeUpperBound = 5;

static void DtGroupNormalizationLayerFwd(int counts)
{
    char funcName[] = "DtGroupNormalizationLayerFwd";
    DT_FUZZ_START(0, counts, funcName, 0)
    {
        int eleIndex = 0;
        int srcInfoElemTypeIndex = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], 0, 0,
            g_precisionMap.size() - 1);
        int scaleShiftInfoElemTypeIndex = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex], 0, 0, 0);
        int dstInfoElemTypeIndex = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], 0, 0,
            g_precisionMap.size() - 1);
        int flagsTypeIndex = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], 0, 0, g_flagsMap.size() - 1);
        size_t dim0 = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], g_dimRangeInitVal, g_dimRangeLowerBound,
            g_dimRangeUpperBound);
        size_t dim1 = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], g_dimRangeInitVal, g_dimRangeLowerBound,
            g_dimRangeUpperBound);
        size_t dim2 = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], g_dimRangeInitVal, g_dimRangeLowerBound,
            g_dimRangeUpperBound);
        size_t dim3 = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], g_dimRangeInitVal, g_dimRangeLowerBound,
            g_dimRangeUpperBound);
        size_t dim4 = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], g_dimRangeInitVal, g_dimRangeLowerBound,
            g_dimRangeUpperBound);
        size_t groupSize = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], g_groupRangeInitVal,
            g_groupRangeLowerBound, g_groupRangeUpperBound);

        KuDNN::Element::Type srcInfoType(g_precisionMap[srcInfoElemTypeIndex].second);
        KuDNN::Element::Type scaleShiftInfoType(g_precisionMap[scaleShiftInfoElemTypeIndex].second);
        KuDNN::Element::Type dstInfoType(g_precisionMap[dstInfoElemTypeIndex].second);
        size_t srcInfoElemSz = srcInfoType.GetSize();
        size_t scaleShiftInfoElemSz = scaleShiftInfoType.GetSize();
        size_t dstInfoElemSz = dstInfoType.GetSize();

        KuDNN::NormalizationFlags gnrmFlag(static_cast<KuDNN::NormalizationFlags>(g_flagsMap[flagsTypeIndex].second));

        const KuDNN::TensorInfo srcInfoTensorInfo = {{dim0, dim1, dim2, dim3, dim4}, srcInfoType, KuDNN::Layout::ABCDE};
        const KuDNN::TensorInfo scaleShiftInfoTensorInfo = {{dim1}, scaleShiftInfoType, KuDNN::Layout::A};
        const KuDNN::TensorInfo dstInfoTensorInfo = {{dim0, dim1, dim2, dim3, dim4}, dstInfoType, KuDNN::Layout::ABCDE};
        char format[] = "DtGnormFwd: Precision(srcInfo:scaleShiftInfo:dstInfo): %s:%s:%s "
            " Shape: %zux%zux%zux%zux%zu , flags=%s ";
        printf(format, g_precisionMap[srcInfoElemTypeIndex].first.c_str(),
            g_precisionMap[scaleShiftInfoElemTypeIndex].first.c_str(),
            g_precisionMap[dstInfoElemTypeIndex].first.c_str(), dim0, dim1, dim2, dim3, dim4,
            g_flagsMap[flagsTypeIndex].first.c_str());
        void *srcInfo = nullptr;
        void *scaleInfo = nullptr;
        void *shiftInfo = nullptr;
        void *dstInfo = nullptr;
        float *mean = nullptr;
        float *variance = nullptr;
        try {
            KuDNN::GroupNormalizationLayerFWD gnorlayerfwd1(srcInfoTensorInfo, scaleShiftInfoTensorInfo, groupSize,
                dstInfoTensorInfo, gnrmFlag);
            KuDNN::GroupNormalizationLayerFWD gnorlayerfwd2(gnorlayerfwd1);
            gnorlayerfwd2 = gnorlayerfwd1;

            KuDNN::GroupNormalizationLayerFWD gnorlayerfwd3(std::move(gnorlayerfwd1));
            KuDNN::GroupNormalizationLayerFWD gnorlayerfwd4(gnorlayerfwd2);
            gnorlayerfwd4 = std::move(gnorlayerfwd2);

            KuDNN::GroupNormalizationLayerFWD GNorLayerFwd(srcInfoTensorInfo,  scaleShiftInfoTensorInfo, groupSize,
                dstInfoTensorInfo, gnrmFlag);
            srcInfo = malloc(std::max((size_t)1, dim0) * std::max((size_t)1, dim1) * std::max((size_t)1, dim2) *
                std::max((size_t)1, dim3) * std::max((size_t)1, dim4) * srcInfoElemSz);
            if (srcInfo == nullptr) {
                std::cerr<<"memory allocation failed"<<std::endl;
                return;
            }

            scaleInfo = malloc(std::max((size_t)1, dim1) * scaleShiftInfoElemSz);
            if (scaleInfo == nullptr) {
                free(srcInfo);
                std::cerr<<"memory allocation failed"<<std::endl;
                return;
            }

            shiftInfo = malloc(std::max((size_t)1, dim1) * scaleShiftInfoElemSz);
            if (shiftInfo == nullptr) {
                free(srcInfo);
                free(scaleInfo);
                std::cerr<<"memory allocation failed"<<std::endl;
                return;
            }

            dstInfo = malloc(std::max((size_t)1, dim0) * std::max((size_t)1, dim1) * std::max((size_t)1, dim2) *
                std::max((size_t)1, dim3) * std::max((size_t)1, dim4) *dstInfoElemSz);
            if (dstInfo == nullptr) {
                free(srcInfo);
                free(scaleInfo);
                free(shiftInfo);
                std::cerr<<"memory allocation failed"<<std::endl;
                return;
            }

            mean = static_cast<float *>(malloc(std::max((size_t)1, dim0) * std::max((size_t)1, groupSize) * sizeof(float)));
            if (mean == nullptr) {
                free(srcInfo);
                free(scaleInfo);
                free(shiftInfo);
                free(dstInfo);
                return;
            }

            variance = static_cast<float *>(malloc(std::max((size_t)1, dim0) * std::max((size_t)1, groupSize) * sizeof(float)));
            if (variance == nullptr) {
                free(srcInfo);
                free(scaleInfo);
                free(shiftInfo);
                free(dstInfo);
                free(mean);
                return;
            }

            GNorLayerFwd.Run(srcInfo, dstInfo, scaleInfo, shiftInfo, mean, variance, true, 1.e-10f);
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

static void DtGroupNormalizationLayerFwdCrossLayout(int counts)
{
    char funcName[] = "DtGroupNormalizationLayerFwdCrossLayout";
    DT_FUZZ_START(0, counts, funcName, 0)
    {
        int eleIndex = 0;
        int srcInfoElemTypeIndex = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], 0, 0,
            g_precisionMap.size() - 1);
        int scaleShiftInfoElemTypeIndex = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex], 0, 0, 0);
        int dstInfoElemTypeIndex = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], 0, 0,
            g_precisionMap.size() - 1);
        int flagsTypeIndex = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], 0, 0, g_flagsMap.size() - 1);
        size_t dim0 = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], g_dimRangeInitVal, g_dimRangeLowerBound,
            g_dimRangeUpperBound);
        size_t dim1 = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], g_dimRangeInitVal, g_dimRangeLowerBound,
            g_dimRangeUpperBound);
        size_t dim2 = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], g_dimRangeInitVal, g_dimRangeLowerBound,
            g_dimRangeUpperBound);
        size_t dim3 = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], g_dimRangeInitVal, g_dimRangeLowerBound,
            g_dimRangeUpperBound);
        size_t dim4 = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], g_dimRangeInitVal, g_dimRangeLowerBound,
            g_dimRangeUpperBound);
        size_t groupSize = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], g_groupRangeInitVal,
            g_groupRangeLowerBound, g_groupRangeUpperBound);

        KuDNN::Element::Type srcInfoType(g_precisionMap[srcInfoElemTypeIndex].second);
        KuDNN::Element::Type scaleShiftInfoType(g_precisionMap[scaleShiftInfoElemTypeIndex].second);
        KuDNN::Element::Type dstInfoType(g_precisionMap[dstInfoElemTypeIndex].second);
        size_t srcInfoElemSz = srcInfoType.GetSize();
        size_t scaleShiftInfoElemSz = scaleShiftInfoType.GetSize();
        size_t dstInfoElemSz = dstInfoType.GetSize();

        KuDNN::NormalizationFlags gnrmFlag(static_cast<KuDNN::NormalizationFlags>(g_flagsMap[flagsTypeIndex].second));

        const KuDNN::TensorInfo srcInfoTensorInfo = {{dim0, dim1, dim2, dim3, dim4}, srcInfoType, KuDNN::Layout::ACDEB};
        const KuDNN::TensorInfo scaleShiftInfoTensorInfo = {{dim4}, scaleShiftInfoType, KuDNN::Layout::A};
        const KuDNN::TensorInfo dstInfoTensorInfo = {{dim0, dim1, dim2, dim3, dim4}, dstInfoType, KuDNN::Layout::ACDEB};
        char format[] = "DtLnormFwdCrossLayout: Precision(srcInfo:scaleShiftInfo:dstInfo): %s:%s:%s"
            "Shape: %zux%zux%zux%zux%zu, flags = %s ";
        printf(format, g_precisionMap[srcInfoElemTypeIndex].first.c_str(),
            g_precisionMap[scaleShiftInfoElemTypeIndex].first.c_str(),
            g_precisionMap[dstInfoElemTypeIndex].first.c_str(), dim0, dim1, dim2, dim3, dim4,
            g_flagsMap[flagsTypeIndex].first.c_str());
        void *srcInfo = nullptr;
        void *scaleInfo = nullptr;
        void *shiftInfo = nullptr;
        void *dstInfo = nullptr;
        float *mean = nullptr;
        float *variance = nullptr;
        try {
            KuDNN::GroupNormalizationLayerFWD GNorLayer1(srcInfoTensorInfo, scaleShiftInfoTensorInfo, groupSize,
                dstInfoTensorInfo, gnrmFlag);
            KuDNN::GroupNormalizationLayerFWD GNorLayer2(GNorLayer1);
            GNorLayer2 = GNorLayer1;
            KuDNN::GroupNormalizationLayerFWD GNorLayer3(GNorLayer1);
            GNorLayer3 = std::move(GNorLayer1);
            KuDNN::GroupNormalizationLayerFWD GNorLayer5(std::move(GNorLayer2));

            KuDNN::GroupNormalizationLayerFWD GNorLayerFwd(srcInfoTensorInfo, scaleShiftInfoTensorInfo, groupSize,
                dstInfoTensorInfo, gnrmFlag);
            srcInfo = malloc(std::max((size_t)1, dim0) * std::max((size_t)1, dim1) * std::max((size_t)1, dim2) *
                std::max((size_t)1, dim3) * std::max((size_t)1, dim4) * srcInfoElemSz);
            if (srcInfo == nullptr) {
                std::cerr<<"memory allocation failed"<<std::endl;
                return;
            }

            scaleInfo = malloc(std::max((size_t)1, dim1) * scaleShiftInfoElemSz);
            if (scaleInfo == nullptr) {
                free(srcInfo);
                std::cerr<<"memory allocation failed"<<std::endl;
                return;
            }

            shiftInfo = malloc(std::max((size_t)1, dim1) * scaleShiftInfoElemSz);
            if (shiftInfo == nullptr) {
                free(srcInfo);
                free(scaleInfo);
                std::cerr<<"memory allocation failed"<<std::endl;
                return;
            }

            dstInfo = malloc(std::max((size_t)1, dim0) * std::max((size_t)1, dim1) * std::max((size_t)1, dim2) *
                std::max((size_t)1, dim3) * std::max((size_t)1, dim4) * dstInfoElemSz);
            if (dstInfo == nullptr) {
                free(srcInfo);
                free(scaleInfo);
                free(shiftInfo);
                std::cerr<<"memory allocation failed"<<std::endl;
                return;
            }

            mean = static_cast<float *>(malloc(std::max((size_t)1, dim0) * std::max((size_t)1, groupSize) * sizeof(float)));
            if (mean == nullptr) {
                free(srcInfo);
                free(scaleInfo);
                free(shiftInfo);
                free(dstInfo);
                return;
            }

            variance = static_cast<float *>(malloc(std::max((size_t)1, dim0) * std::max((size_t)1, groupSize) * sizeof(float)));
            if (variance == nullptr) {
                free(srcInfo);
                free(scaleInfo);
                free(shiftInfo);
                free(dstInfo);
                free(mean);
                return;
            }

            GNorLayerFwd.Run(srcInfo, dstInfo, scaleInfo, shiftInfo, mean, variance, true, 1.e-10f);
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

static void DtGroupNormalizationLayerBwd(int counts)
{
    char funcName[] = "DtGroupNormalizationLayerBwd";
    DT_FUZZ_START(0, counts, funcName, 0)
    {
        int eleIndex = 0;
        int srcInfoElemTypeIndex = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], 0, 0, 2);
        int diffSrcInfoElemTypeIndex = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], 0, 0, 2);
        int flagsTypeIndex = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], 0, 0, g_flagsMap.size() - 1);
        size_t dim0 = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], g_dimRangeInitVal, g_dimRangeLowerBound,
            g_dimRangeUpperBound);
        size_t dim1 = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], g_dimRangeInitVal, g_dimRangeLowerBound,
            g_dimRangeUpperBound);
        size_t dim2 = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], g_dimRangeInitVal, g_dimRangeLowerBound,
            g_dimRangeUpperBound);
        size_t dim3 = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], g_dimRangeInitVal, g_dimRangeLowerBound,
            g_dimRangeUpperBound);
        size_t dim4 = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], g_dimRangeInitVal, g_dimRangeLowerBound,
            g_dimRangeUpperBound);
        size_t groupSize = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], g_groupRangeInitVal,
            g_groupRangeLowerBound, g_groupRangeUpperBound);

        KuDNN::Element::Type srcInfoType(g_precisionMap[srcInfoElemTypeIndex].second);
        KuDNN::Element::Type diffSrcInfoType(g_precisionMap[diffSrcInfoElemTypeIndex].second);
        KuDNN::Element::Type diffDstInfoType(srcInfoType);
        KuDNN::Element::Type scInfoType(g_precisionMap[0].second);
        KuDNN::Element::Type diffScInfoType(g_precisionMap[0].second);
        size_t srcInfoElemSz = srcInfoType.GetSize();
        size_t diffSrcInfoElemSz = diffSrcInfoType.GetSize();
        size_t diffDstInfoElemSz = diffDstInfoType.GetSize();
        size_t scInfoElemSz = scInfoType.GetSize();
        size_t diffScInfoElemSz = diffScInfoType.GetSize();

        KuDNN::NormalizationFlags gnrmFlag(static_cast<KuDNN::NormalizationFlags>(g_flagsMap[flagsTypeIndex].second));

        const KuDNN::TensorInfo srcInfoTensorInfo = {{dim0, dim1, dim2, dim3, dim4}, srcInfoType, KuDNN::Layout::ABCDE};
        const KuDNN::TensorInfo statInfoTensorInfo = {{dim0, groupSize}, KuDNN::Element::TypeT::F32, KuDNN::Layout::AB};
        const KuDNN::TensorInfo diffSrcInfoTensorInfo = {{dim0, dim1, dim2, dim3, dim4}, diffSrcInfoType, KuDNN::Layout::ABCDE};
        const KuDNN::TensorInfo diffDstInfoTensorInfo = {{dim0, dim1, dim2, dim3, dim4}, diffDstInfoType, KuDNN::Layout::ABCDE};
        const KuDNN::TensorInfo scInfoTensorInfo = {{dim1}, scInfoType, KuDNN::Layout::A};
        const KuDNN::TensorInfo diffScInfoTensorInfo = {{dim1}, diffScInfoType, KuDNN::Layout::A};
        char format[] = "DtGnormBwd: Precision(srcInfo:statInfo:diffSrcInfo:diffDstInfo:scInfo:diffScInfo)"
            " %s:%s:%s:%s:%s:%s, Shape: %zux%zux%zux%zux%zu";
        printf(format, g_precisionMap[srcInfoElemTypeIndex].first.c_str(), "F32",
            g_precisionMap[diffSrcInfoElemTypeIndex].first.c_str(), g_precisionMap[srcInfoElemTypeIndex].first.c_str(),
            g_precisionMap[0].first.c_str(), g_precisionMap[0].first.c_str(), dim0, dim1, dim2, dim3, dim4);
        void *src = nullptr;
        float *mean = nullptr;
        float *variance = nullptr;
        void *diffDst = nullptr;
        void *scale = nullptr;
        void *diffSrc = nullptr;
        void *diffScale = nullptr;
        void *diffShift = nullptr;
        try {
            KuDNN::GroupNormalizationLayerBWD gnorLayer1(srcInfoTensorInfo, statInfoTensorInfo, diffSrcInfoTensorInfo,
                diffDstInfoTensorInfo, scInfoTensorInfo, diffScInfoTensorInfo, gnrmFlag);
            KuDNN::GroupNormalizationLayerBWD gnorLayer2(gnorLayer1);
            gnorLayer2 = gnorLayer1;
            KuDNN::GroupNormalizationLayerBWD gnorLayer3(gnorLayer1);
            gnorLayer3 = std::move(gnorLayer1);
            KuDNN::GroupNormalizationLayerBWD gnorLayer5(std::move(gnorLayer2));

            KuDNN::GroupNormalizationLayerBWD GNorLayerBwd(srcInfoTensorInfo, statInfoTensorInfo, diffSrcInfoTensorInfo,
                diffDstInfoTensorInfo, scInfoTensorInfo, diffScInfoTensorInfo, gnrmFlag);
            src = malloc(std::max((size_t)1, dim0) * std::max((size_t)1, dim1) * std::max((size_t)1, dim2) *
                std::max((size_t)1, dim3) * std::max((size_t)1, dim4) * srcInfoElemSz);
            if (src == nullptr) {
                std::cerr<<"memory allocation failed"<<std::endl;
                return;
            }

            mean = (float *)malloc(std::max((size_t)1, dim0) * std::max((size_t)1, groupSize) * sizeof(float));
            if (mean == nullptr) {
                free(src);
                std::cerr<<"memory allocation failed"<<std::endl;
                return;
            }

            variance = (float *)malloc(std::max((size_t)1, dim0) * std::max((size_t)1, groupSize) * sizeof(float));
            if (variance == nullptr) {
                free(src);
                free(mean);
                std::cerr<<"memory allocation failed"<<std::endl;
                return;
            }

            diffDst = malloc(std::max((size_t)1, dim0) * std::max((size_t)1, dim1) * std::max((size_t)1, dim2) *
                std::max((size_t)1, dim3) * std::max((size_t)1, dim4) * diffDstInfoElemSz);

            if (diffDst == nullptr) {
                free(src);
                free(mean);
                free(variance);
                std::cerr<<"memory allocation failed"<<std::endl;
                return;
            }

            scale = malloc(std::max((size_t)1, dim1) * scInfoElemSz);
            if (scale == nullptr) {
                free(src);
                free(mean);
                free(variance);
                free(diffDst);
                std::cerr<<"memory allocation failed"<<std::endl;
                return;
            }

            diffSrc = malloc(std::max((size_t)1, dim0) * std::max((size_t)1, dim1) * std::max((size_t)1, dim2) *
                std::max((size_t)1, dim3) * std::max((size_t)1, dim4) * diffSrcInfoElemSz);
            if (diffSrc == nullptr) {
                free(src);
                free(mean);
                free(variance);
                free(diffDst);
                free(scale);
                std::cerr<<"memory allocation failed"<<std::endl;
                return;
            }

            diffScale = malloc(std::max((size_t)1, dim1) * diffScInfoElemSz);
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

            diffShift = malloc(std::max((size_t)1, dim1) * diffScInfoElemSz);
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

            GNorLayerBwd.Run(src, mean, variance, diffDst, scale, diffSrc, diffScale, diffShift, 1.e-10f);
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

static void DtGroupNormalizationLayerBwdCrossLayout(int counts)
{
    char funcName[] = "DtGroupNormalizationLayerBwdCrossLayout";
    DT_FUZZ_START(0, counts, funcName, 0)
    {
        int eleIndex = 0;
        int srcInfoElemTypeIndex = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], 0, 0, 2);
        int diffSrcInfoElemTypeIndex = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], 0, 0, 2);
        int flagsTypeIndex = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], 0, 0, g_flagsMap.size() - 1);
        size_t dim0 = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], g_dimRangeInitVal, g_dimRangeLowerBound,
            g_dimRangeUpperBound);
        size_t dim1 = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], g_dimRangeInitVal, g_dimRangeLowerBound,
            g_dimRangeUpperBound);
        size_t dim2 = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], g_dimRangeInitVal, g_dimRangeLowerBound,
            g_dimRangeUpperBound);
        size_t dim3 = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], g_dimRangeInitVal, g_dimRangeLowerBound,
            g_dimRangeUpperBound);
        size_t dim4 = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], g_dimRangeInitVal, g_dimRangeLowerBound,
            g_dimRangeUpperBound);
        size_t groupSize = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], g_groupRangeInitVal,
            g_groupRangeLowerBound, g_groupRangeUpperBound);

        KuDNN::Element::Type srcInfoType(g_precisionMap[srcInfoElemTypeIndex].second);
        KuDNN::Element::Type diffSrcInfoType(g_precisionMap[diffSrcInfoElemTypeIndex].second);
        KuDNN::Element::Type diffDstInfoType(srcInfoType);
        KuDNN::Element::Type scInfoType(g_precisionMap[0].second);
        KuDNN::Element::Type diffScInfoType(g_precisionMap[0].second);
        size_t srcInfoElemSz = srcInfoType.GetSize();
        size_t diffSrcInfoElemSz = diffSrcInfoType.GetSize();
        size_t diffDstInfoElemSz = diffDstInfoType.GetSize();
        size_t scInfoElemSz = scInfoType.GetSize();
        size_t diffScInfoElemSz = diffScInfoType.GetSize();

        KuDNN::NormalizationFlags gnrmFlag(static_cast<KuDNN::NormalizationFlags>(g_flagsMap[flagsTypeIndex].second));

        const KuDNN::TensorInfo srcInfoTensorInfo = {{dim0, dim1, dim2, dim3, dim4}, srcInfoType, KuDNN::Layout::ACDEB};
        const KuDNN::TensorInfo statInfoTensorInfo = {{dim0, groupSize}, KuDNN::Element::TypeT::F32, KuDNN::Layout::AB};
        const KuDNN::TensorInfo diffSrcInfoTensorInfo = {{dim0, dim1, dim2, dim3, dim4}, diffSrcInfoType, KuDNN::Layout::ACDEB};
        const KuDNN::TensorInfo diffDstInfoTensorInfo = {{dim0, dim1, dim2, dim3, dim4}, diffDstInfoType, KuDNN::Layout::ACDEB};
        const KuDNN::TensorInfo scInfoTensorInfo = {{dim4}, scInfoType, KuDNN::Layout::A};
        const KuDNN::TensorInfo diffScInfoTensorInfo = {{dim4}, diffScInfoType, KuDNN::Layout::A};
        char format[] = "DtGnormBwdCrossLayout: Precision(srcInfo:statInfo:diffSrcInfo:diffDstInfo:scInfo:diffScInfo)"
            " %s:%s:%s:%s:%s:%s, Shape: %zux%zux%zux%zux%zu";
        printf(format, g_precisionMap[srcInfoElemTypeIndex].first.c_str(), "F32",
            g_precisionMap[diffSrcInfoElemTypeIndex].first.c_str(), g_precisionMap[srcInfoElemTypeIndex].first.c_str(),
            g_precisionMap[0].first.c_str(), g_precisionMap[0].first.c_str(), dim0, dim1, dim2, dim3, dim4);
        void *src = nullptr;
        float *mean = nullptr;
        float *variance = nullptr;
        void *diffDst = nullptr;
        void *scale = nullptr;
        void *diffSrc = nullptr;
        void *diffScale = nullptr;
        void *diffShift = nullptr;
        try {
            KuDNN::GroupNormalizationLayerBWD gnorLayer1(srcInfoTensorInfo, statInfoTensorInfo, diffSrcInfoTensorInfo,
                diffDstInfoTensorInfo, scInfoTensorInfo, diffScInfoTensorInfo, gnrmFlag);
            KuDNN::GroupNormalizationLayerBWD gnorLayer2(gnorLayer1);
            gnorLayer2 = gnorLayer1;
            KuDNN::GroupNormalizationLayerBWD gnorLayer3(gnorLayer1);
            KuDNN::GroupNormalizationLayerBWD gnorLayer4 = std::move(gnorLayer1);
            KuDNN::GroupNormalizationLayerBWD gnorLayerBwd5(std::move(gnorLayer1));

            KuDNN::GroupNormalizationLayerBWD GNorLayerBwd(srcInfoTensorInfo, statInfoTensorInfo, diffSrcInfoTensorInfo,
                diffDstInfoTensorInfo, scInfoTensorInfo, diffScInfoTensorInfo, gnrmFlag);
            src = malloc(std::max((size_t)1, dim0) * std::max((size_t)1, dim1) * std::max((size_t)1, dim2) *
                std::max((size_t)1, dim3) * std::max((size_t)1, dim4) * srcInfoElemSz);
            if (src == nullptr) {
                std::cerr<<"memory allocation failed"<<std::endl;
                return;
            }

            mean = (float *)malloc(std::max((size_t)1, dim0) * std::max((size_t)1, groupSize) * sizeof(float));
            if (mean == nullptr) {
                free(src);
                std::cerr<<"memory allocation failed"<<std::endl;
                return;
            }

            variance = (float *)malloc(std::max((size_t)1, dim0) * std::max((size_t)1, groupSize) * sizeof(float));
            if (variance == nullptr) {
                free(src);
                free(mean);
                std::cerr<<"memory allocation failed"<<std::endl;
                return;
            }

            diffDst = malloc(std::max((size_t)1, dim0) * std::max((size_t)1, dim1) * std::max((size_t)1, dim2) *
                std::max((size_t)1, dim3) * std::max((size_t)1, dim4) * diffDstInfoElemSz);

            if (diffDst == nullptr) {
                free(src);
                free(mean);
                free(variance);
                std::cerr<<"memory allocation failed"<<std::endl;
                return;
            }

            scale = malloc(std::max((size_t)1, dim1) * scInfoElemSz);
            if (scale == nullptr) {
                free(src);
                free(mean);
                free(variance);
                free(diffDst);
                std::cerr<<"memory allocation failed"<<std::endl;
                return;
            }

            diffSrc = malloc(std::max((size_t)1, dim0) * std::max((size_t)1, dim1) * std::max((size_t)1, dim2) *
                std::max((size_t)1, dim3) * std::max((size_t)1, dim4) * diffSrcInfoElemSz);
            if (diffSrc == nullptr) {
                free(src);
                free(mean);
                free(variance);
                free(diffDst);
                free(scale);
                std::cerr<<"memory allocation failed"<<std::endl;
                return;
            }

            diffScale = malloc(std::max((size_t)1, dim1) * diffScInfoElemSz);
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

            diffShift = malloc(std::max((size_t)1, dim1) * diffScInfoElemSz);
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

            GNorLayerBwd.Run(src, mean, variance, diffDst, scale, diffSrc, diffScale, diffShift, 1.e-10f);
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

static constexpr int g_gnFuzzTestCount = 1000;

static Cmdinfo g_gnCmdTable[] = {
    {"GroupNomalization_Layer_Fwd", DtGroupNormalizationLayerFwd, g_gnFuzzTestCount},
    {"GroupNomalization_Layer_Fwd_Cross_Layout", DtGroupNormalizationLayerFwdCrossLayout, g_gnFuzzTestCount},
    {"GroupNomalization_Layer_Bwd", DtGroupNormalizationLayerBwd, g_gnFuzzTestCount},
    {"GroupNomalization_Layer_Bwd_Cross_Layout", DtGroupNormalizationLayerBwdCrossLayout, g_gnFuzzTestCount}
};

int main(int argc, char *argv[])
{
    DT_Enable_Leak_Check(0, 0);
    if (argc > 1) {
        g_gnCmdTable[0] = {"GroupNomalization_Layer_Fwd", DtGroupNormalizationLayerFwd, atoi(argv[1])};
        g_gnCmdTable[1] = {"GroupNomalization_Layer_Fwd_Cross_Layout", DtGroupNormalizationLayerFwdCrossLayout, atoi(argv[1])};
        g_gnCmdTable[2] = {"GroupNomalization_Layer_Bwd", DtGroupNormalizationLayerBwd, atoi(argv[1])};
        g_gnCmdTable[3] = {"GroupNomalization_Layer_Bwd_Cross_Layout", DtGroupNormalizationLayerBwdCrossLayout, atoi(argv[1])};
    }

    int numOfTestItem = sizeof(g_gnCmdTable) / sizeof(g_gnCmdTable[0]);
    int testCount = 0;

    for (int testCaseIdx = 0; testCaseIdx < numOfTestItem; testCaseIdx++) {
        printf("g_gnCmdTable[%d] = %s test %d time\n", testCaseIdx, g_gnCmdTable[testCaseIdx].cmd, testCount);

        testCount = g_gnCmdTable[testCaseIdx].testCount;
        g_gnCmdTable[testCaseIdx].func(testCount);
    }
    return 0;
}