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
    {INDEX6, {"GCHR", static_cast<std::uint32_t>(KuDNN::NormalizationFlags::USE_GLOBAL_STATS) |
        static_cast<std::uint32_t>(KuDNN::NormalizationFlags::USE_SCALE) |
        static_cast<std::uint32_t>(KuDNN::NormalizationFlags::USE_SHIFT) |
        static_cast<std::uint32_t>(KuDNN::NormalizationFlags::FUSE_NORM_RELU)}},
};

static std::map<int, std::pair<std::string, std::uint32_t>> g_flagsBWDMap = {
    {INDEX4, {"CH", static_cast<std::uint32_t>(KuDNN::NormalizationFlags::USE_SCALE) |
        static_cast<std::uint32_t>(KuDNN::NormalizationFlags::USE_SHIFT)}},
    {INDEX5, {"GCH", static_cast<std::uint32_t>(KuDNN::NormalizationFlags::USE_GLOBAL_STATS) |
        static_cast<std::uint32_t>(KuDNN::NormalizationFlags::USE_SCALE) |
        static_cast<std::uint32_t>(KuDNN::NormalizationFlags::USE_SHIFT)}},
};

static std::map<int, std::pair<std::string, KuDNN::Layout>> g_layoutMap = {
    {INDEX0, {"ABX", KuDNN::Layout::ABC}},
    {INDEX1, {"AXB", KuDNN::Layout::ACB}},
};

static constexpr int g_dimRangeInitVal = 1;
static constexpr int g_dimRangeLowerBound = 1;
static constexpr int g_dimRangeUpperBound = 32;
static constexpr int g_chRangeInitVal = 1;
static constexpr int g_chRangeLowerBound = 1;
static constexpr int g_chRangeUpperBound = 320;
static constexpr int g_spRangeInitVal = 64;
static constexpr int g_spRangeLowerBound = 64;
static constexpr int g_spRangeUpperBound = 128;

static void DtBatchNormalizationLayerFwd(int counts)
{
    char funcName[] = "DtBatchNormalizationLayerFwd";
    DT_FUZZ_START(0, counts, funcName, 0)
    {
        int eleIndex = 0;
        int srcInfoElemTypeIndex = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], 0, 0,
            g_precisionMap.size() - 1);
        int flagsTypeIndex = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], 0, 0, g_flagsMap.size() - 1);
        int layoutIndex = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], 0, 0, g_layoutMap.size() - 1);

        size_t dim0 = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], g_dimRangeInitVal, g_dimRangeLowerBound,
            g_dimRangeUpperBound);
        size_t dim1 = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], g_chRangeInitVal, g_chRangeLowerBound,
            g_chRangeUpperBound);
        size_t dim2 = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], g_spRangeInitVal, g_spRangeLowerBound,
            g_spRangeUpperBound);

        int SaveStat = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], 0, 0, 1);
        bool isSaveStat = SaveStat > 0;

        KuDNN::Element::Type srcInfoType(g_precisionMap[srcInfoElemTypeIndex].second);
        KuDNN::Element::Type statInfoType(g_precisionMap[INDEX0].second);
        KuDNN::Element::Type scaleShiftInfoType(g_precisionMap[INDEX0].second);
        size_t srcInfoElemSz = srcInfoType.GetSize();
        size_t scaleShiftInfoElemSz = scaleShiftInfoType.GetSize();
        size_t dstInfoElemSz = srcInfoType.GetSize();
        KuDNN::Layout sdLayout = g_layoutMap[layoutIndex].second;

        KuDNN::NormalizationFlags bnrmFlag(static_cast<KuDNN::NormalizationFlags>(g_flagsMap[flagsTypeIndex].second));

        const KuDNN::TensorInfo srcInfoTensorInfo = {{dim0, dim1, dim2}, srcInfoType, sdLayout};
        const KuDNN::TensorInfo statInfoTensorInfo = {{dim1}, statInfoType, KuDNN::Layout::A};
        const KuDNN::TensorInfo scaleShiftInfoTensorInfo = {{dim1}, scaleShiftInfoType, KuDNN::Layout::A};
        const KuDNN::TensorInfo dstInfoTensorInfo = {{dim0, dim1, dim2}, srcInfoType, sdLayout};
        char format[] = "DtBatchNormalizationLayerFwd: Precision(srcInfo:statInfo:scaleShiftInfo:dstInfo): %s:%s:%s:%s "
            " tag=%s Shape: %zux%zux%zu, flags=%s ";
        printf(format, g_precisionMap[srcInfoElemTypeIndex].first.c_str(),
            g_precisionMap[INDEX0].first.c_str(),
            g_precisionMap[INDEX0].first.c_str(),
            g_precisionMap[srcInfoElemTypeIndex].first.c_str(), g_layoutMap[layoutIndex].first.c_str(),
            dim0, dim1, dim2, g_flagsMap[flagsTypeIndex].first.c_str());
        void *srcInfo = nullptr;
        float *scaleInfo = nullptr;
        float *shiftInfo = nullptr;
        void *dstInfo = nullptr;
        float *mean = nullptr;
        float *variance = nullptr;
        try {
            KuDNN::BatchNormalizationLayerFWD norlayerfwd1(KuDNN::Propagation::FORWARD_INFERENCE, srcInfoTensorInfo,
                statInfoTensorInfo, scaleShiftInfoTensorInfo, dstInfoTensorInfo, bnrmFlag);
            KuDNN::BatchNormalizationLayerFWD norlayerfwd2(norlayerfwd1);
            norlayerfwd2 = norlayerfwd1;

            KuDNN::BatchNormalizationLayerFWD norlayerfwd3(std::move(norlayerfwd1));
            KuDNN::BatchNormalizationLayerFWD norlayerfwd4(norlayerfwd2);
            norlayerfwd4 = std::move(norlayerfwd2);

            KuDNN::BatchNormalizationLayerFWD NorLayerFwd(KuDNN::Propagation::FORWARD_INFERENCE, srcInfoTensorInfo,
                statInfoTensorInfo, scaleShiftInfoTensorInfo, dstInfoTensorInfo, bnrmFlag);
            srcInfo = malloc(std::max((size_t)1, dim0) * std::max((size_t)1, dim1) * std::max((size_t)1, dim2) *
                srcInfoElemSz);
            if (srcInfo == nullptr) {
                std::cerr<<"memory allocation failed"<<std::endl;
                return;
            }

            scaleInfo = static_cast<float *>(malloc(std::max((size_t)1, dim1) * scaleShiftInfoElemSz));
            if (scaleInfo == nullptr) {
                free(srcInfo);
                std::cerr<<"memory allocation failed"<<std::endl;
                return;
            }

            shiftInfo = static_cast<float *>(malloc(std::max((size_t)1, dim1) * scaleShiftInfoElemSz));
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

            NorLayerFwd.Run(srcInfo, dstInfo, scaleInfo, shiftInfo, mean, variance, isSaveStat, 1.e-10f);
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

static void DtBatchNormalizationLayerFwdCrossLayout(int counts)
{
    char funcName[] = "DtBatchNormalizationCrossLayerFwd";
    DT_FUZZ_START(0, counts, funcName, 0)
    {
        int eleIndex = 0;
        int srcInfoElemTypeIndex = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], 0, 0,
            g_precisionMap.size() - 1);
        int flagsTypeIndex = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], 0, 0, g_flagsMap.size() - 1);
        int layoutIndex = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], 0, 0, g_layoutMap.size() - 1);

        size_t dim0 = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], g_dimRangeInitVal, g_dimRangeLowerBound,
            g_dimRangeUpperBound);
        size_t dim1 = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], g_chRangeInitVal, g_chRangeLowerBound,
            g_chRangeUpperBound);
        size_t dim2 = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], g_spRangeInitVal, g_spRangeLowerBound,
            g_spRangeUpperBound);

        int SaveStat = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], 0, 0, 1);
        bool isSaveStat = SaveStat > 0;

        KuDNN::Element::Type srcInfoType(g_precisionMap[srcInfoElemTypeIndex].second);
        KuDNN::Element::Type statInfoType(g_precisionMap[INDEX0].second);
        KuDNN::Element::Type scaleShiftInfoType(g_precisionMap[INDEX0].second);
        size_t srcInfoElemSz = srcInfoType.GetSize();
        size_t scaleShiftInfoElemSz = scaleShiftInfoType.GetSize();
        size_t dstInfoElemSz = srcInfoType.GetSize();
        KuDNN::Layout sdLayout = g_layoutMap[layoutIndex].second;

        KuDNN::NormalizationFlags bnrmFlag(static_cast<KuDNN::NormalizationFlags>(g_flagsMap[flagsTypeIndex].second));

        const KuDNN::TensorInfo srcInfoTensorInfo = {{dim0, dim1, dim2}, srcInfoType, sdLayout};
        const KuDNN::TensorInfo statInfoTensorInfo = {{dim1}, statInfoType, KuDNN::Layout::A};
        const KuDNN::TensorInfo scaleShiftInfoTensorInfo = {{dim1}, scaleShiftInfoType, KuDNN::Layout::A};
        const KuDNN::TensorInfo dstInfoTensorInfo = {{dim0, dim1, dim2}, srcInfoType, sdLayout};
        char format[] = "DtBatchNormalizationCrossLayerFwd: Precision(srcInfo:statInfo:scaleShiftInfo:dstInfo): %s:%s:%s:%s "
            " tag=%s Shape: %zux%zux%zu, flags=%s ";
        printf(format, g_precisionMap[srcInfoElemTypeIndex].first.c_str(),
            g_precisionMap[INDEX0].first.c_str(),
            g_precisionMap[INDEX0].first.c_str(),
            g_precisionMap[srcInfoElemTypeIndex].first.c_str(), g_layoutMap[layoutIndex].first.c_str(),
            dim0, dim1, dim2, g_flagsMap[flagsTypeIndex].first.c_str());
        void *srcInfo = nullptr;
        float *scaleInfo = nullptr;
        float *shiftInfo = nullptr;
        void *dstInfo = nullptr;
        float *mean = nullptr;
        float *variance = nullptr;
        int8_t *ws = nullptr;
        try {
            KuDNN::BatchNormalizationLayerFWD norlayerfwd1(KuDNN::Propagation::FORWARD_TRAINING, srcInfoTensorInfo,
                statInfoTensorInfo, scaleShiftInfoTensorInfo, dstInfoTensorInfo, bnrmFlag);
            KuDNN::BatchNormalizationLayerFWD norlayerfwd2(norlayerfwd1);
            norlayerfwd2 = norlayerfwd1;

            KuDNN::BatchNormalizationLayerFWD norlayerfwd3(std::move(norlayerfwd1));
            KuDNN::BatchNormalizationLayerFWD norlayerfwd4(norlayerfwd2);
            norlayerfwd4 = std::move(norlayerfwd2);

            KuDNN::BatchNormalizationLayerFWD NorLayerFwd(KuDNN::Propagation::FORWARD_TRAINING, srcInfoTensorInfo,
                statInfoTensorInfo, scaleShiftInfoTensorInfo, dstInfoTensorInfo, bnrmFlag);
            srcInfo = malloc(std::max((size_t)1, dim0) * std::max((size_t)1, dim1) * std::max((size_t)1, dim2) *
                srcInfoElemSz);
            if (srcInfo == nullptr) {
                std::cerr<<"memory allocation failed"<<std::endl;
                return;
            }

            scaleInfo = static_cast<float *>(malloc(std::max((size_t)1, dim1) * scaleShiftInfoElemSz));
            if (scaleInfo == nullptr) {
                free(srcInfo);
                std::cerr<<"memory allocation failed"<<std::endl;
                return;
            }

            shiftInfo = static_cast<float *>(malloc(std::max((size_t)1, dim1) * scaleShiftInfoElemSz));
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

            ws = static_cast<int8_t *>(malloc(std::max((size_t)1, dim0) * std::max((size_t)1, dim1) *
                std::max((size_t)1, dim2)));
            if (ws == nullptr) {
                free(srcInfo);
                free(scaleInfo);
                free(shiftInfo);
                free(dstInfo);
                free(mean);
                free(variance);
                return;
            }

            NorLayerFwd.Run(srcInfo, dstInfo, scaleInfo, shiftInfo, mean, variance, isSaveStat, 1.e-10f, ws);
            free(srcInfo);
            free(scaleInfo);
            free(shiftInfo);
            free(dstInfo);
            free(mean);
            free(variance);
            free(ws);
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
            if (ws != nullptr) {
                free(ws);
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
            if (ws != nullptr) {
                free(ws);
            }
            printf("LogicError");
        }
        printf("\n");
    }
    DT_FUZZ_END()
}

static void DtBatchNormalizationLayerBwd3D(int counts)
{
    char funcName[] = "DtBatchNormalizationLayerBwd3D";
    DT_FUZZ_START(0, counts, funcName, 0)
    {
        int eleIndex = 0;
        int srcInfoElemTypeIndex = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], 0, 0,
            g_precisionMap.size() - 2);
        int flagsTypeIndex = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], 0, 0, g_flagsMap.size() - 1);
        int layoutIndex = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], 0, 0, g_layoutMap.size() - 1);

        size_t dim0 = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], g_dimRangeInitVal, g_dimRangeLowerBound,
            g_dimRangeUpperBound);
        size_t dim1 = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], g_dimRangeInitVal, g_dimRangeLowerBound,
            g_dimRangeUpperBound);
        size_t dim2 = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], g_dimRangeInitVal, g_dimRangeLowerBound,
            g_dimRangeUpperBound);

        KuDNN::Element::Type srcInfoType(g_precisionMap[srcInfoElemTypeIndex].second);
        size_t srcInfoElemSz = srcInfoType.GetSize();
        KuDNN::Layout sdLayout = g_layoutMap[layoutIndex].second;

        KuDNN::NormalizationFlags bnrmFlag(static_cast<KuDNN::NormalizationFlags>(g_flagsMap[flagsTypeIndex].second));

        const KuDNN::TensorInfo srcInfoTensorInfo = {{dim0, dim1, dim2}, srcInfoType, sdLayout};
        const KuDNN::TensorInfo diffSrcInfoTensorInfo = {{dim0, dim1, dim2}, srcInfoType, sdLayout};
        const KuDNN::TensorInfo diffDstInfoTensorInfo = {{dim0, dim1, dim2}, srcInfoType, sdLayout};
        const KuDNN::TensorInfo statInfoTensorInfo = {{dim1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::A};
        const KuDNN::TensorInfo scInfoTensorInfo = {{dim1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::A};
        const KuDNN::TensorInfo diffScInfoTensorInfo = {{dim1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::A};
        // const KuDNN::TensorInfo diffScInfoTensorInfo = {{dim2}, diffScInfoType, KuDNN::Layout::A};
        char format[] = "DtBatchNormalizationLayerBwd: Precision(srcInfo:statInfo:diffSrcInfo:diffDstInfo:scInfo:diffScInfo)"
            " %s:%s:%s:%s:%s:%s, tag=%s Shape: %zux%zux%zu ";
        printf(format, g_precisionMap[srcInfoElemTypeIndex].first.c_str(),
            g_precisionMap[INDEX0].first.c_str(), g_precisionMap[srcInfoElemTypeIndex].first.c_str(),
            g_precisionMap[srcInfoElemTypeIndex].first.c_str(), g_precisionMap[INDEX0].first.c_str(),
            g_precisionMap[srcInfoElemTypeIndex].first.c_str(), g_layoutMap[layoutIndex].first.c_str(), dim0, dim1, dim2);
        void *src = nullptr;
        float *mean = nullptr;
        float *variance = nullptr;
        void *diffDst = nullptr;
        float *scale = nullptr;
        float *diffSrc = nullptr;
        float *diffScale = nullptr;
        float *diffShift = nullptr;
        int8_t *ws = nullptr;
        try {
            KuDNN::BatchNormalizationLayerBWD norLayer1(KuDNN::Propagation::BACKWARD_DATA,
                srcInfoTensorInfo, diffDstInfoTensorInfo, statInfoTensorInfo,
                scInfoTensorInfo, diffSrcInfoTensorInfo, diffScInfoTensorInfo, bnrmFlag);
            KuDNN::BatchNormalizationLayerBWD norLayer2(norLayer1);
            norLayer2 = norLayer1;
            KuDNN::BatchNormalizationLayerBWD norLayer3(norLayer1);
            norLayer3 = std::move(norLayer1);
            KuDNN::BatchNormalizationLayerBWD norLayer5(std::move(norLayer2));

            KuDNN::BatchNormalizationLayerBWD NorLayerBwd(KuDNN::Propagation::BACKWARD_DATA,
                srcInfoTensorInfo, srcInfoTensorInfo, statInfoTensorInfo,
                scInfoTensorInfo, srcInfoTensorInfo, scInfoTensorInfo, bnrmFlag);
            src = malloc(std::max((size_t)1, dim0) * std::max((size_t)1, dim1) * std::max((size_t)1, dim2) *
                srcInfoElemSz);
            if (src == nullptr) {
                std::cerr<<"memory allocation failed"<<std::endl;
                return;
            }

            mean = (float *)malloc(std::max((size_t)1, dim1) * sizeof(float));
            if (mean == nullptr) {
                free(src);
                std::cerr<<"memory allocation failed"<<std::endl;
                return;
            }

            variance = (float *)malloc(std::max((size_t)1, dim1) * sizeof(float));
            if (variance == nullptr) {
                free(src);
                free(mean);
                std::cerr<<"memory allocation failed"<<std::endl;
                return;
            }

            diffDst = malloc(std::max((size_t)1, dim0) * std::max((size_t)1, dim1) * std::max((size_t)1, dim2) *
                srcInfoElemSz);

            if (diffDst == nullptr) {
                free(src);
                free(mean);
                free(variance);
                std::cerr<<"memory allocation failed"<<std::endl;
                return;
            }

            scale = static_cast<float *>(malloc(std::max((size_t)1, dim1) * sizeof(float)));
            if (scale == nullptr) {
                free(src);
                free(mean);
                free(variance);
                free(diffDst);
                std::cerr<<"memory allocation failed"<<std::endl;
                return;
            }

            diffSrc = static_cast<float *>(malloc(std::max((size_t)1, dim0) * std::max((size_t)1, dim1) * std::max((size_t)1, dim2) *
                srcInfoElemSz));
            if (diffSrc == nullptr) {
                free(src);
                free(mean);
                free(variance);
                free(diffDst);
                free(scale);
                std::cerr<<"memory allocation failed"<<std::endl;
                return;
            }

            diffScale = static_cast<float *>(malloc(std::max((size_t)1, dim1) * sizeof(float)));
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

            diffShift = static_cast<float *>(malloc(std::max((size_t)1, dim1) * sizeof(float)));
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

            ws = static_cast<int8_t *>(malloc(std::max((size_t)1, dim0) * std::max((size_t)1, dim1) *
                std::max((size_t)1, dim2)));
            if (ws == nullptr) {
                free(src);
                free(mean);
                free(variance);
                free(diffDst);
                free(scale);
                free(diffSrc);
                free(diffScale);
                free(diffShift);
                std::cerr<<"memory allocation failed"<<std::endl;
                return;
            }

            NorLayerBwd.Run(src, diffDst, mean, variance, scale, diffSrc, diffScale, diffShift, 1.e-5f, ws);
            free(src);
            free(mean);
            free(variance);
            free(diffDst);
            free(scale);
            free(diffSrc);
            free(diffScale);
            free(diffShift);
            free(ws);
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
            if (ws != nullptr) {
                free(ws);
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
            if (ws != nullptr) {
                free(ws);
            }
            printf("LogicError");
        }
        printf("\n");
    }
    DT_FUZZ_END()
}

static void DtBatchNormalizationLayerBwd4D(int counts)
{
    char funcName[] = "DtBatchNormalizationLayerBwd4D";
    DT_FUZZ_START(0, counts, funcName, 0)
    {
        int eleIndex = 0;
        int srcInfoElemTypeIndex = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], 0, 0,
            g_precisionMap.size() - 2);
        int flagsTypeIndex = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], 0, 0, g_flagsBWDMap.size() - 1);

        size_t dim0 = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], g_dimRangeInitVal, g_dimRangeLowerBound,
            g_dimRangeUpperBound);
        size_t dim1 = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], g_dimRangeInitVal, g_dimRangeLowerBound,
            g_dimRangeUpperBound);
        size_t dim2 = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], g_dimRangeInitVal, g_dimRangeLowerBound,
            g_dimRangeUpperBound);
        size_t dim3 = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], g_dimRangeInitVal, g_dimRangeLowerBound,
            g_dimRangeUpperBound);

        KuDNN::Element::Type srcInfoType(g_precisionMap[srcInfoElemTypeIndex].second);
        size_t srcInfoElemSz = srcInfoType.GetSize();

        KuDNN::NormalizationFlags bnrmFlag(static_cast<KuDNN::NormalizationFlags>(g_flagsBWDMap[flagsTypeIndex].second));

        const KuDNN::TensorInfo srcInfoTensorInfo = {{dim0, dim1, dim2, dim3}, srcInfoType, KuDNN::Layout::ACDB};
        const KuDNN::TensorInfo diffSrcInfoTensorInfo = {{dim0, dim1, dim2, dim3}, srcInfoType, KuDNN::Layout::ACDB};
        const KuDNN::TensorInfo diffDstInfoTensorInfo = {{dim0, dim1, dim2, dim3}, srcInfoType, KuDNN::Layout::ACDB};
        const KuDNN::TensorInfo statInfoTensorInfo = {{dim1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::A};
        const KuDNN::TensorInfo scInfoTensorInfo = {{dim1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::A};
        const KuDNN::TensorInfo diffScInfoTensorInfo = {{dim1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::A};
        // const KuDNN::TensorInfo diffScInfoTensorInfo = {{dim2}, diffScInfoType, KuDNN::Layout::A};
        char format[] = "DtBatchNormalizationLayerBwd: Precision(srcInfo:statInfo:diffSrcInfo:diffDstInfo:scInfo:diffScInfo)"
            " %s:%s:%s:%s:%s:%s, tag=%s Shape: %zux%zux%zux%zu ";
        printf(format, g_precisionMap[srcInfoElemTypeIndex].first.c_str(),
            g_precisionMap[INDEX0].first.c_str(), g_precisionMap[srcInfoElemTypeIndex].first.c_str(),
            g_precisionMap[srcInfoElemTypeIndex].first.c_str(), g_precisionMap[INDEX0].first.c_str(),
            g_precisionMap[srcInfoElemTypeIndex].first.c_str(), "acdb", dim0, dim1, dim2, dim3);
        void *src = nullptr;
        float *mean = nullptr;
        float *variance = nullptr;
        void *diffDst = nullptr;
        float *scale = nullptr;
        float *diffSrc = nullptr;
        float *diffScale = nullptr;
        float *diffShift = nullptr;
        int8_t *ws = nullptr;
        try {
            KuDNN::BatchNormalizationLayerBWD norLayer1(KuDNN::Propagation::BACKWARD_DATA,
                srcInfoTensorInfo, diffDstInfoTensorInfo, statInfoTensorInfo,
                scInfoTensorInfo, diffSrcInfoTensorInfo, diffScInfoTensorInfo, bnrmFlag);
            KuDNN::BatchNormalizationLayerBWD norLayer2(norLayer1);
            norLayer2 = norLayer1;
            KuDNN::BatchNormalizationLayerBWD norLayer3(norLayer1);
            norLayer3 = std::move(norLayer1);
            KuDNN::BatchNormalizationLayerBWD norLayer5(std::move(norLayer2));

            KuDNN::BatchNormalizationLayerBWD NorLayerBwd(KuDNN::Propagation::BACKWARD_DATA,
                srcInfoTensorInfo, srcInfoTensorInfo, statInfoTensorInfo,
                scInfoTensorInfo, srcInfoTensorInfo, scInfoTensorInfo, bnrmFlag);
            src = malloc(std::max((size_t)1, dim0) * std::max((size_t)1, dim1) * std::max((size_t)1, dim2) *
                std::max((size_t)1, dim3) * srcInfoElemSz);
            if (src == nullptr) {
                std::cerr<<"memory allocation failed"<<std::endl;
                return;
            }

            mean = (float *)malloc(std::max((size_t)1, dim1) * sizeof(float));
            if (mean == nullptr) {
                free(src);
                std::cerr<<"memory allocation failed"<<std::endl;
                return;
            }

            variance = (float *)malloc(std::max((size_t)1, dim1) * sizeof(float));
            if (variance == nullptr) {
                free(src);
                free(mean);
                std::cerr<<"memory allocation failed"<<std::endl;
                return;
            }

            diffDst = malloc(std::max((size_t)1, dim0) * std::max((size_t)1, dim1) * std::max((size_t)1, dim2) *
                std::max((size_t)1, dim3) * srcInfoElemSz);

            if (diffDst == nullptr) {
                free(src);
                free(mean);
                free(variance);
                std::cerr<<"memory allocation failed"<<std::endl;
                return;
            }

            scale = static_cast<float *>(malloc(std::max((size_t)1, dim1) * sizeof(float)));
            if (scale == nullptr) {
                free(src);
                free(mean);
                free(variance);
                free(diffDst);
                std::cerr<<"memory allocation failed"<<std::endl;
                return;
            }

            diffSrc = static_cast<float *>(malloc(std::max((size_t)1, dim0) * std::max((size_t)1, dim1) * std::max((size_t)1, dim2) *
                std::max((size_t)1, dim3) * srcInfoElemSz));
            if (diffSrc == nullptr) {
                free(src);
                free(mean);
                free(variance);
                free(diffDst);
                free(scale);
                std::cerr<<"memory allocation failed"<<std::endl;
                return;
            }

            diffScale = static_cast<float *>(malloc(std::max((size_t)1, dim1) * sizeof(float)));
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

            diffShift = static_cast<float *>(malloc(std::max((size_t)1, dim1) * sizeof(float)));
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

            ws = static_cast<int8_t *>(malloc(std::max((size_t)1, dim0) * std::max((size_t)1, dim1) *
                std::max((size_t)1, dim2) * std::max((size_t)1, dim3)));
            if (ws == nullptr) {
                free(src);
                free(mean);
                free(variance);
                free(diffDst);
                free(scale);
                free(diffSrc);
                free(diffScale);
                free(diffShift);
                std::cerr<<"memory allocation failed"<<std::endl;
                return;
            }

            NorLayerBwd.Run(src, diffDst, mean, variance, scale, diffSrc, diffScale, diffShift, 1.e-5f, ws);
            free(src);
            free(mean);
            free(variance);
            free(diffDst);
            free(scale);
            free(diffSrc);
            free(diffScale);
            free(diffShift);
            free(ws);
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
            if (ws != nullptr) {
                free(ws);
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
            if (ws != nullptr) {
                free(ws);
            }
            printf("LogicError");
        }
        printf("\n");
    }
    DT_FUZZ_END()
}

static void DtBatchNormalizationLayerBwd5D(int counts)
{
    char funcName[] = "DtBatchNormalizationLayerBwd5D";
    DT_FUZZ_START(0, counts, funcName, 0)
    {
        int eleIndex = 0;
        int srcInfoElemTypeIndex = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], 0, 0,
            g_precisionMap.size() - 2);
        int flagsTypeIndex = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], 0, 0, g_flagsBWDMap.size() - 1);

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

        KuDNN::Element::Type srcInfoType(g_precisionMap[srcInfoElemTypeIndex].second);
        size_t srcInfoElemSz = srcInfoType.GetSize();

        KuDNN::NormalizationFlags bnrmFlag(static_cast<KuDNN::NormalizationFlags>(g_flagsBWDMap[flagsTypeIndex].second));

        const KuDNN::TensorInfo srcInfoTensorInfo = {{dim0, dim1, dim2, dim3, dim4}, srcInfoType, KuDNN::Layout::ACDEB};
        const KuDNN::TensorInfo diffSrcInfoTensorInfo = {{dim0, dim1, dim2, dim3, dim4}, srcInfoType, KuDNN::Layout::ACDEB};
        const KuDNN::TensorInfo diffDstInfoTensorInfo = {{dim0, dim1, dim2, dim3, dim4}, srcInfoType, KuDNN::Layout::ACDEB};
        const KuDNN::TensorInfo statInfoTensorInfo = {{dim1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::A};
        const KuDNN::TensorInfo scInfoTensorInfo = {{dim1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::A};
        const KuDNN::TensorInfo diffScInfoTensorInfo = {{dim1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::A};
        // const KuDNN::TensorInfo diffScInfoTensorInfo = {{dim2}, diffScInfoType, KuDNN::Layout::A};
        char format[] = "DtBatchNormalizationLayerBwd: Precision(srcInfo:statInfo:diffSrcInfo:diffDstInfo:scInfo:diffScInfo)"
            " %s:%s:%s:%s:%s:%s, tag=%s Shape: %zux%zux%zux%zu%zu ";
        printf(format, g_precisionMap[srcInfoElemTypeIndex].first.c_str(),
            g_precisionMap[INDEX0].first.c_str(), g_precisionMap[srcInfoElemTypeIndex].first.c_str(),
            g_precisionMap[srcInfoElemTypeIndex].first.c_str(), g_precisionMap[INDEX0].first.c_str(),
            g_precisionMap[srcInfoElemTypeIndex].first.c_str(), "acdeb", dim0, dim1, dim2, dim3, dim4);
        void *src = nullptr;
        float *mean = nullptr;
        float *variance = nullptr;
        void *diffDst = nullptr;
        float *scale = nullptr;
        float *diffSrc = nullptr;
        float *diffScale = nullptr;
        float *diffShift = nullptr;
        int8_t *ws = nullptr;
        try {
            KuDNN::BatchNormalizationLayerBWD norLayer1(KuDNN::Propagation::BACKWARD_DATA,
                srcInfoTensorInfo, diffDstInfoTensorInfo, statInfoTensorInfo,
                scInfoTensorInfo, diffSrcInfoTensorInfo, diffScInfoTensorInfo, bnrmFlag);
            KuDNN::BatchNormalizationLayerBWD norLayer2(norLayer1);
            norLayer2 = norLayer1;
            KuDNN::BatchNormalizationLayerBWD norLayer3(norLayer1);
            norLayer3 = std::move(norLayer1);
            KuDNN::BatchNormalizationLayerBWD norLayer5(std::move(norLayer2));

            KuDNN::BatchNormalizationLayerBWD NorLayerBwd(KuDNN::Propagation::BACKWARD_DATA,
                srcInfoTensorInfo, srcInfoTensorInfo, statInfoTensorInfo,
                scInfoTensorInfo, srcInfoTensorInfo, scInfoTensorInfo, bnrmFlag);
            src = malloc(std::max((size_t)1, dim0) * std::max((size_t)1, dim1) * std::max((size_t)1, dim2) *
                std::max((size_t)1, dim3) * std::max((size_t)1, dim4) * srcInfoElemSz);
            if (src == nullptr) {
                std::cerr<<"memory allocation failed"<<std::endl;
                return;
            }

            mean = (float *)malloc(std::max((size_t)1, dim1) * sizeof(float));
            if (mean == nullptr) {
                free(src);
                std::cerr<<"memory allocation failed"<<std::endl;
                return;
            }

            variance = (float *)malloc(std::max((size_t)1, dim1) * sizeof(float));
            if (variance == nullptr) {
                free(src);
                free(mean);
                std::cerr<<"memory allocation failed"<<std::endl;
                return;
            }

            diffDst = malloc(std::max((size_t)1, dim0) * std::max((size_t)1, dim1) * std::max((size_t)1, dim2) *
                std::max((size_t)1, dim3) * std::max((size_t)1, dim4) * srcInfoElemSz);

            if (diffDst == nullptr) {
                free(src);
                free(mean);
                free(variance);
                std::cerr<<"memory allocation failed"<<std::endl;
                return;
            }

            scale = static_cast<float *>(malloc(std::max((size_t)1, dim1) * sizeof(float)));
            if (scale == nullptr) {
                free(src);
                free(mean);
                free(variance);
                free(diffDst);
                std::cerr<<"memory allocation failed"<<std::endl;
                return;
            }

            diffSrc = static_cast<float *>(malloc(std::max((size_t)1, dim0) * std::max((size_t)1, dim1) * std::max((size_t)1, dim2) *
                std::max((size_t)1, dim3) * std::max((size_t)1, dim4) * srcInfoElemSz));
            if (diffSrc == nullptr) {
                free(src);
                free(mean);
                free(variance);
                free(diffDst);
                free(scale);
                std::cerr<<"memory allocation failed"<<std::endl;
                return;
            }

            diffScale = static_cast<float *>(malloc(std::max((size_t)1, dim1) * sizeof(float)));
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

            diffShift = static_cast<float *>(malloc(std::max((size_t)1, dim1) * sizeof(float)));
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

            ws = static_cast<int8_t *>(malloc(std::max((size_t)1, dim0) * std::max((size_t)1, dim1) *
                std::max((size_t)1, dim2) * std::max((size_t)1, dim3) * std::max((size_t)1, dim4)));
            if (ws == nullptr) {
                free(src);
                free(mean);
                free(variance);
                free(diffDst);
                free(scale);
                free(diffSrc);
                free(diffScale);
                free(diffShift);
                std::cerr<<"memory allocation failed"<<std::endl;
                return;
            }

            NorLayerBwd.Run(src, diffDst, mean, variance, scale, diffSrc, diffScale, diffShift, 1.e-5f, ws);
            free(src);
            free(mean);
            free(variance);
            free(diffDst);
            free(scale);
            free(diffSrc);
            free(diffScale);
            free(diffShift);
            free(ws);
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
            if (ws != nullptr) {
                free(ws);
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
            if (ws != nullptr) {
                free(ws);
            }
            printf("LogicError");
        }
        printf("\n");
    }
    DT_FUZZ_END()
}

static constexpr int g_bnFuzzTestCountFWD = 4000;
static constexpr int g_bnFuzzTestCountBWD = 1000;

static Cmdinfo g_bnCmdTable[] = {
    {"Nomalization_Layer_Fwd", DtBatchNormalizationLayerFwd, g_bnFuzzTestCountFWD},
    {"Nomalization_Layer_Fwd_Cross_Layout", DtBatchNormalizationLayerFwdCrossLayout, g_bnFuzzTestCountFWD},
    {"Nomalization_Layer_Bwd_3D", DtBatchNormalizationLayerBwd3D, g_bnFuzzTestCountBWD},
    {"Nomalization_Layer_Bwd_4D", DtBatchNormalizationLayerBwd4D, g_bnFuzzTestCountBWD},
    {"Nomalization_Layer_Bwd_5D", DtBatchNormalizationLayerBwd5D, g_bnFuzzTestCountBWD}
};

int main(int argc, char *argv[])
{
    DT_Enable_Leak_Check(0, 0);
    if (argc > 1) {
        g_bnCmdTable[0] = {"Nomalization_Layer_Fwd", DtBatchNormalizationLayerFwd, atoi(argv[1])};
        g_bnCmdTable[1] = {"Nomalization_Layer_Fwd_Cross_Layout", DtBatchNormalizationLayerFwdCrossLayout, atoi(argv[1])};
        g_bnCmdTable[2] = {"Nomalization_Layer_Bwd_3D", DtBatchNormalizationLayerBwd3D, atoi(argv[1])};
        g_bnCmdTable[3] = {"Nomalization_Layer_Bwd_4D", DtBatchNormalizationLayerBwd4D, atoi(argv[1])};
        g_bnCmdTable[4] = {"Nomalization_Layer_Bwd_5D", DtBatchNormalizationLayerBwd5D, atoi(argv[1])};
    }

    int numOfTestItem = sizeof(g_bnCmdTable) / sizeof(g_bnCmdTable[0]);
    int testCount = 0;
    if (argc > 2) {
        const char* inputCmd = argv[2];
        for (int testCaseIdx = 0; testCaseIdx < numOfTestItem; testCaseIdx++) {
            if (strcmp(g_bnCmdTable[testCaseIdx].cmd, inputCmd) == 0) {
                testCount = g_bnCmdTable[testCaseIdx].testCount;
                printf("g_bnCmdTable[%d] = %s test %d time\n", testCaseIdx, g_bnCmdTable[testCaseIdx].cmd,
                    testCount);
                g_bnCmdTable[testCaseIdx].func(testCount);
                break;
            }
        }
    } else {
        for (int testCaseIdx = 0; testCaseIdx < numOfTestItem; testCaseIdx++) {
            testCount = g_bnCmdTable[testCaseIdx].testCount;
            g_bnCmdTable[testCaseIdx].func(testCount);
            printf("g_bnCmdTable[%d] = %s test %d time\n", testCaseIdx, g_bnCmdTable[testCaseIdx].cmd, testCount);
        }
    }

    return 0;
}