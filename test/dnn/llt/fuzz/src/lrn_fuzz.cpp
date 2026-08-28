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

static std::map<int, std::pair<std::string, std::uint32_t>> g_algorithmMap = {
    {INDEX0, {"WITHIN", static_cast<std::uint32_t>(KuDNN::LRNormAlgorithmKind::WITHIN)}},
    {INDEX1, {"ACROSS", static_cast<std::uint32_t>(KuDNN::LRNormAlgorithmKind::ACROSS)}},
};

static std::map<int, std::pair<std::string, KuDNN::Layout>> g_layoutMap = {
    {INDEX0, {"ABC", KuDNN::Layout::ABC}},
    {INDEX1, {"ACB", KuDNN::Layout::ACB}},
};

static constexpr int g_dimRangeInitVal = 2;
static constexpr int g_dimRangeLowerBound = 2;
static constexpr int g_dimRangeUpperBound = 30;

static void DtLocalResponseNormalizationLayerFwd(int counts)
{
    char funcName[] = "DtLocalResponseNormalizationLayerFwd";
    DT_FUZZ_START(0, counts, funcName, 0)
    {
        int eleIndex = 0;
        int srcInfoElemTypeIndex = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], 0, 0,
            g_precisionMap.size() - 1);
        int dstInfoElemTypeIndex = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], 0, 0,
            g_precisionMap.size() - 1);
        int algorithmTypeIndex = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], 0, 0,
            g_precisionMap.size() - 1);
        int layoutIndex = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], 0, 0, g_layoutMap.size() - 1);
        size_t dim0 = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], g_dimRangeInitVal, g_dimRangeLowerBound,
            g_dimRangeUpperBound);
        size_t dim1 = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], g_dimRangeInitVal, g_dimRangeLowerBound,
            g_dimRangeUpperBound);
        size_t dim2 = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], g_dimRangeInitVal, g_dimRangeLowerBound,
            g_dimRangeUpperBound);

        KuDNN::Element::Type srcInfoType(g_precisionMap[srcInfoElemTypeIndex].second);
        KuDNN::Element::Type dstInfoType(g_precisionMap[dstInfoElemTypeIndex].second);
        KuDNN::Layout layout(g_layoutMap[layoutIndex].second);
        size_t srcInfoElemSz = srcInfoType.GetSize();
        size_t dstInfoElemSz = dstInfoType.GetSize();

        KuDNN::LRNormAlgorithmKind lrnAlogrithm = (static_cast<KuDNN::LRNormAlgorithmKind>(g_algorithmMap[algorithmTypeIndex].second));

        const KuDNN::TensorInfo srcInfoTensorInfo = {{dim0, dim1, dim2}, srcInfoType, layout};
        const KuDNN::TensorInfo dstInfoTensorInfo = {{dim0, dim1, dim2}, dstInfoType, layout};
        float alphaInfo = 1.0;
        float betaInfo = 1.0;
        float kInfo = 1.0;
        size_t localSizeInfo = 10;
        
        void *srcInfo = nullptr;
        void *dstInfo = nullptr;
        try {
            KuDNN::LocalResponseNormalizationLayerFWD lrnfwd1(srcInfoTensorInfo, dstInfoTensorInfo, alphaInfo, betaInfo, kInfo,
                localSizeInfo, lrnAlogrithm);
            KuDNN::LocalResponseNormalizationLayerFWD lrnfwd2(lrnfwd1);
            lrnfwd2 = lrnfwd1;

            KuDNN::LocalResponseNormalizationLayerFWD lrnfwd3(std::move(lrnfwd1));
            KuDNN::LocalResponseNormalizationLayerFWD lrnfwd4(lrnfwd2);
            lrnfwd4 = std::move(lrnfwd2);

            KuDNN::LocalResponseNormalizationLayerFWD LRNorLayerFwd(srcInfoTensorInfo, dstInfoTensorInfo, alphaInfo, betaInfo, kInfo,
                localSizeInfo, lrnAlogrithm);
            
            srcInfo = malloc(std::max((size_t)1, dim0) * std::max((size_t)1, dim1) * std::max((size_t)1, dim2) *
                srcInfoElemSz);
            if (srcInfo == nullptr) {
                std::cerr<<"memory allocation failed"<<std::endl;
                return;
            }

            dstInfo = malloc(std::max((size_t)1, dim0) * std::max((size_t)1, dim1) * std::max((size_t)1, dim2) *
                dstInfoElemSz);
            if (dstInfo == nullptr) {
                free(srcInfo);
                std::cerr<<"memory allocation failed"<<std::endl;
                return;
            }
            LRNorLayerFwd.Run(srcInfo, dstInfo);

            free(srcInfo);
            free(dstInfo);
        } catch (const KuDNN::Service::Unsupported& e) {
            if (srcInfo != nullptr) {
                free(srcInfo);
            }
            if (dstInfo != nullptr) {
                free(dstInfo);
            }
            printf("Unsupported Configuration");
        } catch (const KuDNN::Service::LogicError& e) {
            if (srcInfo != nullptr) {
                free(srcInfo);
            }
            if (dstInfo != nullptr) {
                free(dstInfo);
            }
            printf("LogicError");
        }
        printf("\n");
    }
    DT_FUZZ_END()
}

static void DtLocalResponseNormalizationLayerBwd(int counts)
{
    char funcName[] = "DtLocalResponseNormalizationLayerBwd";
    DT_FUZZ_START(0, counts, funcName, 0)
    {
        int eleIndex = 0;
        int srcInfoElemTypeIndex = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], 0, 0,
            g_precisionMap.size() - 1);
        int diffDstInfoElemTypeIndex = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], 0, 0,
            g_precisionMap.size() - 1);
        int algorithmTypeIndex = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], 0, 0,
            g_precisionMap.size() - 1);
        int layoutIndex = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], 0, 0, g_layoutMap.size() - 1);
        size_t dim0 = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], g_dimRangeInitVal, g_dimRangeLowerBound,
            g_dimRangeUpperBound);
        size_t dim1 = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], g_dimRangeInitVal, g_dimRangeLowerBound,
            g_dimRangeUpperBound);
        size_t dim2 = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], g_dimRangeInitVal, g_dimRangeLowerBound,
            g_dimRangeUpperBound);

        KuDNN::Element::Type srcInfoType(g_precisionMap[srcInfoElemTypeIndex].second);
        KuDNN::Element::Type diffDstInfoType(g_precisionMap[diffDstInfoElemTypeIndex].second);
        KuDNN::Element::Type diffSrcInfoType(g_precisionMap[srcInfoElemTypeIndex].second);
        KuDNN::Layout layout(g_layoutMap[layoutIndex].second);
        size_t srcInfoElemSz = srcInfoType.GetSize();
        size_t diffDstInfoElemSz = diffDstInfoType.GetSize();
        size_t diffSrcInfoElemSz = diffSrcInfoType.GetSize();

        KuDNN::LRNormAlgorithmKind lrnAlogrithm = (static_cast<KuDNN::LRNormAlgorithmKind>(g_algorithmMap[algorithmTypeIndex].second));

        const KuDNN::TensorInfo srcInfoTensorInfo = {{dim0, dim1, dim2}, srcInfoType, layout};
        const KuDNN::TensorInfo diffDstInfoTensorInfo = {{dim0, dim1, dim2}, diffDstInfoType, layout};
        const KuDNN::TensorInfo diffSrcInfoTensorInfo = {{dim0, dim1, dim2}, diffSrcInfoType, layout};
        float alphaInfo = 1.0;
        float betaInfo = 1.0;
        float kInfo = 1.0;
        size_t localSizeInfo = 10;

        void *srcInfo = nullptr;
        void *diffDstInfo = nullptr;
        void *diffSrcInfo = nullptr;
        try {
            KuDNN::LocalResponseNormalizationLayerBWD lrnbwd1(srcInfoTensorInfo, diffDstInfoTensorInfo, diffSrcInfoTensorInfo, alphaInfo, betaInfo, kInfo,
                localSizeInfo, lrnAlogrithm);
            KuDNN::LocalResponseNormalizationLayerBWD lrnbwd2(lrnbwd1);
            lrnbwd2 = lrnbwd1;

            KuDNN::LocalResponseNormalizationLayerBWD lrnbwd3(std::move(lrnbwd1));
            KuDNN::LocalResponseNormalizationLayerBWD lrnbwd4(lrnbwd2);
            lrnbwd4 = std::move(lrnbwd2);

            KuDNN::LocalResponseNormalizationLayerBWD LRNorLayerBwd(srcInfoTensorInfo, diffDstInfoTensorInfo, diffSrcInfoTensorInfo, alphaInfo, betaInfo, kInfo,
                localSizeInfo, lrnAlogrithm);

            srcInfo = malloc(std::max((size_t)1, dim0) * std::max((size_t)1, dim1) * std::max((size_t)1, dim2) *
                srcInfoElemSz);
            if (srcInfo == nullptr) {
                std::cerr<<"memory allocation failed"<<std::endl;
                return;
            }

            diffDstInfo = malloc(std::max((size_t)1, dim0) * std::max((size_t)1, dim1) * std::max((size_t)1, dim2) *
                diffDstInfoElemSz);
            if (diffDstInfo == nullptr) {
                free(srcInfo);
                std::cerr<<"memory allocation failed"<<std::endl;
                return;
            }

            diffSrcInfo = malloc(std::max((size_t)1, dim0) * std::max((size_t)1, dim1) * std::max((size_t)1, dim2) *
                diffSrcInfoElemSz);
            if (diffDstInfo == nullptr) {
                free(srcInfo);
                free(diffDstInfo);
                std::cerr<<"memory allocation failed"<<std::endl;
                return;
            }
            
            LRNorLayerBwd.Run(srcInfo, diffDstInfo, diffSrcInfo);

            free(srcInfo);
            free(diffDstInfo);
            free(diffSrcInfo);
        } catch (const KuDNN::Service::Unsupported& e) {
            if (srcInfo != nullptr) {
                free(srcInfo);
            }
            if (diffDstInfo != nullptr) {
                free(diffDstInfo);
            }
            if (diffSrcInfo != nullptr) {
                free(diffSrcInfo);
            }
            printf("Unsupported Configuration");
        } catch (const KuDNN::Service::LogicError& e) {
            if (srcInfo != nullptr) {
                free(srcInfo);
            }
            if (diffDstInfo != nullptr) {
                free(diffDstInfo);
            }
            if (diffSrcInfo != nullptr) {
                free(diffSrcInfo);
            }
            printf("LogicError");
        }
        printf("\n");
    }
    DT_FUZZ_END()
}

static constexpr int g_lrnFuzzTestCount = 1000;

static Cmdinfo g_lrnCmdTable[] = {
    {"LRNomalization_Layer_Fwd", DtLocalResponseNormalizationLayerFwd, g_lrnFuzzTestCount},
    {"LRNomalization_Layer_Bwd", DtLocalResponseNormalizationLayerBwd, g_lrnFuzzTestCount}
};

int main(int argc, char *argv[])
{
    DT_Enable_Leak_Check(0, 0);
    if (argc > 1) {
        g_lrnCmdTable[0] = {"LRNomalization_Layer_Fwd", DtLocalResponseNormalizationLayerFwd, atoi(argv[1])};
        g_lrnCmdTable[1] = {"LRNomalization_Layer_Bwd", DtLocalResponseNormalizationLayerBwd, atoi(argv[1])};
    }

    int numOfTestItem = sizeof(g_lrnCmdTable) / sizeof(g_lrnCmdTable[0]);
    int testCount = 0;

    for (int testCaseIdx = 0; testCaseIdx < numOfTestItem; testCaseIdx++) {
        printf("g_lrnCmdTable[%d] = %s test %d time\n", testCaseIdx, g_lrnCmdTable[testCaseIdx].cmd, testCount);

        testCount = g_lrnCmdTable[testCaseIdx].testCount;
        g_lrnCmdTable[testCaseIdx].func(testCount);
    }
    return 0;
} 