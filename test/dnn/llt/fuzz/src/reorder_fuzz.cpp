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

static std::map<int, std::pair<std::string, KuDNN::Layout>> g_layoutMap = {
    {INDEX0, {"ABCDE", KuDNN::Layout::ABCDE}},
    {INDEX1, {"ABCED", KuDNN::Layout::ABCED}},
    {INDEX2, {"ABDEC", KuDNN::Layout::ABDEC}},
    {INDEX3, {"ACBDE", KuDNN::Layout::ACBDE}},
    {INDEX4, {"ACDEB", KuDNN::Layout::ACDEB}},
    {INDEX5, {"ADECB", KuDNN::Layout::ADECB}},
    {INDEX6, {"BACDE", KuDNN::Layout::BACDE}},
    {INDEX7, {"BCDEA", KuDNN::Layout::BCDEA}},
    {INDEX8, {"CDEAB", KuDNN::Layout::CDEAB}},
    {INDEX9, {"CDEBA", KuDNN::Layout::CDEBA}},
    {INDEX10, {"DECAB", KuDNN::Layout::DECAB}},
};

static std::map<int, std::pair<std::string, KuDNN::Element::TypeT>> g_precisionMap_fast = {
    {INDEX0, {"F32", KuDNN::Element::TypeT::F32}},
    {INDEX1, {"S8", KuDNN::Element::TypeT::S8}},
    {INDEX2, {"U8", KuDNN::Element::TypeT::U8}},
    {INDEX3, {"S32", KuDNN::Element::TypeT::S32}},
};

static std::map<int, std::pair<std::pair<std::string, std::string>,
        std::pair<KuDNN::Layout, KuDNN::Layout>>> g_layoutMap_fast = {
    {INDEX0, {{"AB", "BA"}, {KuDNN::Layout::AB, KuDNN::Layout::BA}}},
    {INDEX1, {{"BA", "AB"}, {KuDNN::Layout::BA, KuDNN::Layout::AB}}},
};


static constexpr int g_dimRangeInitVal = 4;
static constexpr int g_dimRangeLowerBound = 4;
static constexpr int g_dimRangeUpperBound = 8;
static constexpr int g_dimRangeInitVal_fast = 17;
static constexpr int g_dimRangeLowerBound_fast = 17;
static constexpr int g_dimRangeUpperBound_fast = 31;

static void DtFuzzReorder(int counts)
{
    char funcName[] = "DtFuzzReorder";
    DT_FUZZ_START(0, counts, funcName, 0)
    {
        int eleIndex = 0;
        int srcElemTypeIndex = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], 0, 0, g_precisionMap.size() - 1);
        int dstElemTypeIndex = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], 0, 0, g_precisionMap.size() - 1);
        int srcLayoutIndex = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], 0, 0, g_layoutMap.size() - 1);
        int dstLayoutIndex = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], 0, 0, g_layoutMap.size() - 1);
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
        float scale = *(float *)DT_SetGetNumberRangeFloatV3(eleIndex++, 1, 0, 2);

        KuDNN::Element::Type srcType(g_precisionMap[srcElemTypeIndex].second);
        KuDNN::Element::Type dstType(g_precisionMap[dstElemTypeIndex].second);
        size_t srcElemSz = srcType.GetSize();
        size_t dstElemSz = dstType.GetSize();

        KuDNN::Layout srcLayout(g_layoutMap[srcLayoutIndex].second);
        KuDNN::Layout dstLayout(g_layoutMap[dstLayoutIndex].second);

        const KuDNN::TensorInfo srcTensorInfo = {{dim0, dim1, dim2, dim3, dim4}, srcType, srcLayout};
        const KuDNN::TensorInfo dstTensorInfo = {{dim0, dim1, dim2, dim3, dim4}, dstType, dstLayout};
        char format[] = "Precision(src:dst): %s:%s, Layout(src:dst):%s:%s, Shape: %zux%zux%zux%zux%zu, ";
        printf(format, g_precisionMap[srcElemTypeIndex].first.c_str(), g_precisionMap[dstElemTypeIndex].first.c_str(),
            g_layoutMap[srcLayoutIndex].first.c_str(), g_layoutMap[dstLayoutIndex].first.c_str(),
            dim0, dim1, dim2, dim3, dim4);
        void *dst = nullptr;
        void *src = nullptr;
        try {
            KuDNN::ReorderLayer reorderLayer1(srcTensorInfo, dstTensorInfo);
            KuDNN::ReorderLayer reorderLayer2 = reorderLayer1;
            KuDNN::ReorderLayer reorderLayer3(reorderLayer1);
            reorderLayer3 = reorderLayer2;
            KuDNN::ReorderLayer reorderLayer4(srcTensorInfo, dstTensorInfo);
            reorderLayer4 = std::move(reorderLayer2);
            KuDNN::ReorderLayer reorderLayer5(std::move(reorderLayer1));

            KuDNN::ReorderLayer reorderLayer(srcTensorInfo, dstTensorInfo);

            src = malloc(std::max((size_t)1, dim0) * std::max((size_t)1, dim1) * std::max((size_t)1, dim2) *
                std::max((size_t)1, dim3) * std::max((size_t)1, dim4) * srcElemSz);
            if (src == nullptr) {
                std::cerr<<"memory allocation failed"<<std::endl;
                return;
            }

            dst = malloc(std::max((size_t)1, dim0) * std::max((size_t)1, dim1) * std::max((size_t)1, dim2) *
                std::max((size_t)1, dim3) * std::max((size_t)1, dim4) * dstElemSz);
            if (dst == nullptr) {
                free(src);
                std::cerr<<"memory allocation failed"<<std::endl;
                return;
            }
            reorderLayer.Run(src, dst, scale);
            free(src);
            free(dst);
        } catch (const KuDNN::Service::Unsupported& e) {
            if (dst != nullptr) {
                free(dst);
            }
            if (src != nullptr) {
                free(src);
            }
            printf("Exception: Unsupported");
        } catch (const KuDNN::Service::LogicError& e) {
            if (dst != nullptr) {
                free(dst);
            }
            if (src != nullptr) {
                free(src);
            }
            printf("Exception: Logic Error");
        }
        printf("\n");
    }
    DT_FUZZ_END()
}

static void DtFuzzReorder_fast(int counts)
{
    char funcName[] = "DtFuzzReorder_fast";
    DT_FUZZ_START(0, counts, funcName, 0)
    {
        int eleIndex = 0;
        int ElemTypeIndex = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], 0, 0,
            g_precisionMap_fast.size() - 1);
        int LayoutIndex = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], 0, 0,
            g_layoutMap_fast.size() - 1);
 
        size_t dim0 = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], g_dimRangeInitVal_fast,
            g_dimRangeLowerBound_fast, g_dimRangeUpperBound_fast);
        size_t dim1 = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], g_dimRangeInitVal_fast,
            g_dimRangeLowerBound_fast, g_dimRangeUpperBound_fast);

        KuDNN::Element::Type srcType(g_precisionMap_fast[ElemTypeIndex].second);
        KuDNN::Element::Type dstType(g_precisionMap_fast[ElemTypeIndex].second);
        size_t srcElemSz = srcType.GetSize();
        size_t dstElemSz = dstType.GetSize();

        KuDNN::Layout srcLayout(g_layoutMap_fast[LayoutIndex].second.first);
        KuDNN::Layout dstLayout(g_layoutMap_fast[LayoutIndex].second.second);

        const KuDNN::TensorInfo srcTensorInfo = {{dim0, dim1}, srcType, srcLayout};
        const KuDNN::TensorInfo dstTensorInfo = {{dim0, dim1}, dstType, dstLayout};
        char format[] = "Precision(src:dst): %s:%s, Layout(src:dst):%s:%s, Shape: %zux%zu, ";
        printf(format, g_precisionMap_fast[ElemTypeIndex].first.c_str(),
            g_precisionMap_fast[ElemTypeIndex].first.c_str(),
            g_layoutMap_fast[LayoutIndex].first.first.c_str(),
            g_layoutMap_fast[LayoutIndex].first.second.c_str(),
            dim0, dim1);
        void *dst = nullptr;
        void *src = nullptr;
        try {
            KuDNN::ReorderLayer reorderLayer1(srcTensorInfo, dstTensorInfo);
            KuDNN::ReorderLayer reorderLayer2 = reorderLayer1;
            KuDNN::ReorderLayer reorderLayer3(reorderLayer1);
            reorderLayer3 = reorderLayer2;
            KuDNN::ReorderLayer reorderLayer4(srcTensorInfo, dstTensorInfo);
            reorderLayer4 = std::move(reorderLayer2);
            KuDNN::ReorderLayer reorderLayer5(std::move(reorderLayer1));

            KuDNN::ReorderLayer reorderLayer(srcTensorInfo, dstTensorInfo);

            src = malloc(std::max((size_t)1, dim0) * std::max((size_t)1, dim1) * srcElemSz);
            if (src == nullptr) {
                std::cerr<<"memory allocation failed"<<std::endl;
                return;
            }

            dst = malloc(std::max((size_t)1, dim0) * std::max((size_t)1, dim1) * dstElemSz);
            if (dst == nullptr) {
                free(src);
                std::cerr<<"memory allocation failed"<<std::endl;
                return;
            }
            reorderLayer.Run(src, dst);
            free(src);
            free(dst);
        } catch (const KuDNN::Service::Unsupported& e) {
            if (dst != nullptr) {
                free(dst);
            }
            if (src != nullptr) {
                free(src);
            }
            printf("Exception: Unsupported");
        } catch (const KuDNN::Service::LogicError& e) {
            if (dst != nullptr) {
                free(dst);
            }
            if (src != nullptr) {
                free(src);
            }
            printf("Exception: Logic Error");
        }
        printf("\n");
    }
    DT_FUZZ_END()
}


static constexpr int g_reductionFuzzTestCount = 15000;
static constexpr int g_reductionFuzzTestCount_fast = 2000;

static Cmdinfo g_reorderCmdTable[] = {
    {"Reorder", DtFuzzReorder, g_reductionFuzzTestCount},
    {"ReorderFast", DtFuzzReorder_fast, g_reductionFuzzTestCount_fast}
};

int main(int argc, char *argv[])
{
    DT_Enable_Leak_Check(0, 0);
    if (argc > 1) {
        g_reorderCmdTable[0] = {"Reorder", DtFuzzReorder, atoi(argv[1])};
        g_reorderCmdTable[1] = {"ReorderFast", DtFuzzReorder_fast, atoi(argv[1])};
    }

    int numOfTestItem = sizeof(g_reorderCmdTable) / sizeof(g_reorderCmdTable[0]);
    int testCount = 0;

    for (int testCaseIdx = 0; testCaseIdx < numOfTestItem; testCaseIdx++) {
        printf("g_reorderCmdTable[%d] = %s test %d time\n", testCaseIdx, g_reorderCmdTable[testCaseIdx].cmd,
            testCount);
        testCount = g_reorderCmdTable[testCaseIdx].testCount;
        g_reorderCmdTable[testCaseIdx].func(testCount);
    }
    return 0;
}