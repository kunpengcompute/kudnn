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

static constexpr int g_dimRangeInitVal = 1;
static constexpr int g_dimRangeLowerBound = 1;
static constexpr int g_dimRangeUpperBound = 5;

static constexpr int g_groupRangeInitVal = 1;
static constexpr int g_groupRangeLowerBound = 1;
static constexpr int g_groupRangeUpperBound = 5;

static constexpr int g_axisRangeInitVal = 0;
static constexpr int g_axisRangeLowerBound = 0;
static constexpr int g_axisRangeUpperBound = 4;

static void DtFuzzShuffle(int counts)
{
    char funcName[] = "DtFuzzShuffle";
    DT_FUZZ_START(0, counts, funcName, 0)
    {
        int eleIndex = 0;
        int srcElemTypeIndex = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], 0, 0, g_precisionMap.size() - 1);
        int dstElemTypeIndex = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], 0, 0, g_precisionMap.size() - 1);
        int srcLayoutIndex = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], 0, 0, g_precisionMap.size() - 1);
        int dstLayoutIndex = srcLayoutIndex;
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
        size_t axis = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], g_axisRangeInitVal, g_axisRangeLowerBound,
            g_axisRangeUpperBound);

        KuDNN::Element::Type srcType(g_precisionMap[srcElemTypeIndex].second);
        KuDNN::Element::Type dstType = srcType;
        size_t srcElemSz = srcType.GetSize();
        size_t dstElemSz = dstType.GetSize();

        KuDNN::Layout srcLayout(g_layoutMap[srcLayoutIndex].second);
        KuDNN::Layout dstLayout(g_layoutMap[dstLayoutIndex].second);

        const KuDNN::TensorInfo srcTensorInfo = {{dim0, dim1, dim2, dim3, dim4}, srcType, srcLayout};
        const KuDNN::TensorInfo dstTensorInfo = {{dim0, dim1, dim2, dim3, dim4}, dstType, dstLayout};

        char format[] = "Precision(src:dst): %s:%s, Layout(src:dst):%s:%s, Shape: %zux%zux%zux%zux%zu, ";
        printf(format, g_layoutMap[srcLayoutIndex].first.c_str(), g_layoutMap[dstLayoutIndex].first.c_str(),
            g_precisionMap[srcElemTypeIndex].first.c_str(), g_precisionMap[dstElemTypeIndex].first.c_str(),
            dim0, dim1, dim2, dim3, dim4);
        void *dst = nullptr;
        void *src = nullptr;
        try {
            KuDNN::ShuffleLayer shuffleLayer1(srcTensorInfo, dstTensorInfo, axis, groupSize);
            KuDNN::ShuffleLayer shuffleLayer2 = shuffleLayer1;
            KuDNN::ShuffleLayer shuffleLayer3(shuffleLayer1);
            shuffleLayer3 = shuffleLayer2;
            KuDNN::ShuffleLayer shuffleLayer4(srcTensorInfo, dstTensorInfo, axis, groupSize);
            shuffleLayer4 = std::move(shuffleLayer2);
            KuDNN::ShuffleLayer shuffleLayer5(std::move(shuffleLayer1));

            KuDNN::ShuffleLayer shuffleLayer(srcTensorInfo, dstTensorInfo, axis, groupSize);

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
            shuffleLayer.Run(src, dst);
            free(src);
            free(dst);
        } catch (const KuDNN::Service::Unsupported& e) {
            if (dst != nullptr) {
                free(dst);
            }
            if (src != nullptr) {
                free(src);
            }
            printf("Unsupported");
        } catch (const KuDNN::Service::LogicError& e) {
            if (dst != nullptr) {
                free(dst);
            }
            if (src != nullptr) {
                free(src);
            }
            printf("Logic Error");
        }
        printf("\n");
    }
    DT_FUZZ_END()
}

static constexpr int g_reductionFuzzTestCount = 15000;

static Cmdinfo g_shuffleCmdTable[] = {
    {"Shuffle", DtFuzzShuffle, g_reductionFuzzTestCount},
};

int main(int argc, char *argv[])
{
    DT_Enable_Leak_Check(0, 0);
    if (argc > 1) {
        g_shuffleCmdTable[0] = {"Shuffle", DtFuzzShuffle, atoi(argv[1])};
    }

    int numOfTestItem = sizeof(g_shuffleCmdTable) / sizeof(g_shuffleCmdTable[0]);
    int testCount = 0;

    for (int testCaseIdx = 0; testCaseIdx < numOfTestItem; testCaseIdx++) {
        printf("g_shuffleCmdTable[%d] = %s test %d time\n", testCaseIdx, g_shuffleCmdTable[testCaseIdx].cmd,
            testCount);
        testCount = g_shuffleCmdTable[testCaseIdx].testCount;
        g_shuffleCmdTable[testCaseIdx].func(testCount);
    }
    return 0;
}