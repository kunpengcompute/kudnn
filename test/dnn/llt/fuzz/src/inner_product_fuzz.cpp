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

static constexpr int g_dimRangeInitVal = 1;
static constexpr int g_dimRangeLowerBound = 1;
static constexpr int g_dimRangeUpperBound = 30;

static void DtInnerProductFwd(int counts)
{
    char funcName[] = "DtInnerProductFwd";
    DT_FUZZ_START(0, counts, funcName, 0)
    {
        int eleIndex = 0;
        int srcElemTypeIndex = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], 0, 0, g_precisionMap.size() - 1);
        int weightsElemTypeIndex =
            *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], 0, 0, g_precisionMap.size() - 1);
        int dstElemTypeIndex = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], 0, 0, g_precisionMap.size() - 1);
        int biasElemTypeIndex = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], 0, 0, g_precisionMap.size() - 1);
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

        KuDNN::Element::Type srcType(g_precisionMap[srcElemTypeIndex].second);
        KuDNN::Element::Type weightsType(g_precisionMap[weightsElemTypeIndex].second);
        KuDNN::Element::Type dstType(g_precisionMap[dstElemTypeIndex].second);
        KuDNN::Element::Type biasType(g_precisionMap[biasElemTypeIndex].second);
        size_t srcElemSz = srcType.GetSize();
        size_t weightsElemSz = weightsType.GetSize();
        size_t dstElemSz = dstType.GetSize();
        size_t biasElemSz = biasType.GetSize();

        // N-IC-ID-IH-IW
        const KuDNN::TensorInfo srcTensorInfo = {{dim0, dim1, dim2, dim3, dim4}, srcType, KuDNN::Layout::ABCDE};
        // OC-IC-ID-IH-IW
        const KuDNN::TensorInfo weightsTensorInfo = {{dim0, dim1, dim2, dim3, dim4}, weightsType, KuDNN::Layout::ABCDE};
        // N-OC
        const KuDNN::TensorInfo dstTensorInfo = {{dim0, dim0}, dstType, KuDNN::Layout::AB};
        // OC
        const KuDNN::TensorInfo biasTensorInfo = {{dim0}, biasType, KuDNN::Layout::A};
        char format[] = "DtInnerProductFwd: Precision(src:weights:dst:bias): %s:%s:%s:%s, Shape: %zux%zux%zux%zux%zu ";
        printf(format, g_precisionMap[srcElemTypeIndex].first.c_str(),
            g_precisionMap[weightsElemTypeIndex].first.c_str(), g_precisionMap[dstElemTypeIndex].first.c_str(),
            g_precisionMap[biasElemTypeIndex].first.c_str(), dim0, dim1, dim2, dim3, dim4);
        void *src = NULL;
        void *weights = NULL;
        void *dst = NULL;
        void *bias = NULL;
        try {
            KuDNN::InnerProductLayerFWD inner1(srcTensorInfo, weightsTensorInfo, dstTensorInfo, biasTensorInfo);
            KuDNN::InnerProductLayerFWD inner2(inner1);
            inner2 = inner1;
            KuDNN::InnerProductLayerFWD inner3(inner1);
            inner3 = std::move(inner1);
            KuDNN::InnerProductLayerFWD inner5(std::move(inner2));

            KuDNN::InnerProductLayerFWD ipLayerFwd(srcTensorInfo, weightsTensorInfo, dstTensorInfo, biasTensorInfo);
            src = malloc(std::max((size_t)1, dim0) * std::max((size_t)1, dim1) * std::max((size_t)1, dim2) *
                std::max((size_t)1, dim3) * std::max((size_t)1, dim4) * srcElemSz);
            if (src == NULL) {
                std::cerr<<"memory allocation failed"<<std::endl;
                return;
            }

            weights = malloc(std::max((size_t)1, dim0) * std::max((size_t)1, dim1) * std::max((size_t)1, dim2) *
                std::max((size_t)1, dim3) * std::max((size_t)1, dim4) * weightsElemSz);
            if (weights == NULL) {
                free(src);
                std::cerr<<"memory allocation failed"<<std::endl;
                return;
            }

            dst = malloc(std::max((size_t)1, dim0) * std::max((size_t)1, dim0) * dstElemSz);
            if (dst == NULL) {
                free(src);
                free(weights);
                std::cerr<<"memory allocation failed"<<std::endl;
                return;
            }

            bias = malloc(std::max((size_t)1, dim0) * biasElemSz);
            if (bias == NULL) {
                free(src);
                free(weights);
                free(dst);
                std::cerr<<"memory allocation failed"<<std::endl;
                return;
            }

            ipLayerFwd.Run(src, weights, dst, bias);
            free(src);
            free(weights);
            free(dst);
            free(bias);
        } catch (const KuDNN::Service::Unsupported& e) {
            if (src != NULL) {
                free(src);
            }
            if (weights != NULL) {
                free(weights);
            }
            if (dst != NULL) {
                free(dst);
            }
            if (bias != NULL) {
                free(bias);
            }
            printf("Unsupported Configuration");
        } catch (const KuDNN::Service::LogicError& e) {
            if (src != NULL) {
                free(src);
            }
            if (weights != NULL) {
                free(weights);
            }
            if (dst != NULL) {
                free(dst);
            }
            if (bias != NULL) {
                free(bias);
            }
            printf("Bad shapes' arguments");
        }
        printf("\n");
    }
    DT_FUZZ_END()
}

static void DtInnerProductBwd(int counts)
{
    char funcName[] = "DtInnerProductBwd";
    DT_FUZZ_START(0, counts, funcName, 0)
    {
        int eleIndex = 0;
        int diffDstElemTypeIndex =
            *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], 0, 0, g_precisionMap.size() - 1);
        int weightsElemTypeIndex =
            *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], 0, 0, g_precisionMap.size() - 1);
        int diffSrcElemTypeIndex =
            *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], 0, 0, g_precisionMap.size() - 1);
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

        KuDNN::Element::Type diffDstType(g_precisionMap[diffDstElemTypeIndex].second);
        KuDNN::Element::Type weightsType(g_precisionMap[weightsElemTypeIndex].second);
        KuDNN::Element::Type diffSrcType(g_precisionMap[diffSrcElemTypeIndex].second);
        size_t diffDstElemSz = diffDstType.GetSize();
        size_t weightsElemSz = weightsType.GetSize();
        size_t diffSrcElemSz = diffSrcType.GetSize();

        const KuDNN::TensorInfo diffSrcTensorInfo = {{dim0, dim1, dim2, dim3, dim4}, diffSrcType, KuDNN::Layout::ABCDE};
        const KuDNN::TensorInfo weightsTensorInfo = {{dim0, dim1, dim2, dim3, dim4}, weightsType, KuDNN::Layout::ABCDE};
        const KuDNN::TensorInfo diffDstTensorInfo = {{dim0, dim0}, diffDstType, KuDNN::Layout::AB};
        char format[] = "DtInnerProductBwd: Precision(diffDst:weights:diffSrc): %s:%s:%s, Shape: %zux%zux%zux%zux%zu ";
        printf(format, g_precisionMap[diffDstElemTypeIndex].first.c_str(),
            g_precisionMap[weightsElemTypeIndex].first.c_str(), g_precisionMap[diffSrcElemTypeIndex].first.c_str(),
            dim0, dim1, dim2, dim3, dim4);
        void *diffSrc = NULL;
        void *weights = NULL;
        void *diffDst = NULL;
        try {
            KuDNN::InnerProductLayerBWDData innerBWD1(diffDstTensorInfo, weightsTensorInfo, diffSrcTensorInfo);
            KuDNN::InnerProductLayerBWDData innerBWD2(innerBWD1);
            innerBWD2 = innerBWD1;
            KuDNN::InnerProductLayerBWDData innerBWD3(innerBWD2);
            innerBWD3 = std::move(innerBWD1);
            KuDNN::InnerProductLayerBWDData innerBWD5(std::move(innerBWD2));

            KuDNN::InnerProductLayerBWDData ipLayerBWD(diffDstTensorInfo, weightsTensorInfo, diffSrcTensorInfo);
            diffSrc = malloc(std::max((size_t)1, dim0) * std::max((size_t)1, dim1) * std::max((size_t)1, dim2) *
                std::max((size_t)1, dim3) * std::max((size_t)1, dim4) * diffSrcElemSz);
            if (diffSrc == NULL) {
                std::cerr<<"memory allocation failed"<<std::endl;
                return;
            }

            weights = malloc(std::max((size_t)1, dim0) * std::max((size_t)1, dim1) * std::max((size_t)1, dim2) *
                std::max((size_t)1, dim3) * std::max((size_t)1, dim4) * weightsElemSz);
            if (weights == NULL) {
                free(diffSrc);
                std::cerr<<"memory allocation failed"<<std::endl;
                return;
            }

            diffDst = malloc(std::max((size_t)1, dim0) * std::max((size_t)1, dim0) * diffDstElemSz);
            if (diffDst == NULL) {
                free(diffSrc);
                free(weights);
                std::cerr<<"memory allocation failed"<<std::endl;
                return;
            }

            ipLayerBWD.Run(diffDst, weights, diffSrc);
            free(diffDst);
            free(weights);
            free(diffSrc);
        } catch (const KuDNN::Service::Unsupported& e) {
            if (diffDst != NULL) {
                free(diffDst);
            }
            if (weights != NULL) {
                free(weights);
            }
            if (diffSrc != NULL) {
                free(diffSrc);
            }
            printf("Unsupported Configuration");
        } catch (const KuDNN::Service::LogicError& e) {
            if (diffDst != NULL) {
                free(diffDst);
            }
            if (weights != NULL) {
                free(weights);
            }
            if (diffSrc != NULL) {
                free(diffSrc);
            }
            printf("Bad shapes' arguments");
        }
        printf("\n");
    }
    DT_FUZZ_END()
}

static void DtInnerProductBWDWeights(int counts)
{
    char funcName[] = "DtInnerProductBWDWeights";
    DT_FUZZ_START(0, counts, funcName, 0)
    {
        int eleIndex = 0;
        int diffSrcElemTypeIndex =
            *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], 0, 0, g_precisionMap.size() - 1);
        int diffWeightsElemTypeIndex =
            *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], 0, 0, g_precisionMap.size() - 1);
        int diffDstElemTypeIndex =
            *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], 0, 0, g_precisionMap.size() - 1);
        int diffBiasElemTypeIndex =
            *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], 0, 0, g_precisionMap.size() - 1);
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

        KuDNN::Element::Type diffSrcType(g_precisionMap[diffSrcElemTypeIndex].second);
        KuDNN::Element::Type diffWeightsType(g_precisionMap[diffWeightsElemTypeIndex].second);
        KuDNN::Element::Type diffDstType(g_precisionMap[diffDstElemTypeIndex].second);
        KuDNN::Element::Type diffBiasType(g_precisionMap[diffBiasElemTypeIndex].second);
        size_t diffSrcElemSz = diffSrcType.GetSize();
        size_t diffWeightsElemSz = diffWeightsType.GetSize();
        size_t diffDstElemSz = diffDstType.GetSize();
        size_t diffBiasElemSz = diffBiasType.GetSize();

        const KuDNN::TensorInfo diffSrcTensorInfo = {{dim0, dim1, dim2, dim3, dim4}, diffSrcType, KuDNN::Layout::ABCDE};
        const KuDNN::TensorInfo diffWeightsTensorInfo = {{dim0, dim1, dim2, dim3, dim4},
            diffWeightsType, KuDNN::Layout::ABCDE};
        const KuDNN::TensorInfo diffDstTensorInfo = {{dim0, dim0}, diffDstType, KuDNN::Layout::AB};
        const KuDNN::TensorInfo diffBiasTensorInfo = {{dim0}, diffBiasType, KuDNN::Layout::A};
        char format[] = "DtIpBWDWeights: Precision(dSrc:dWeights:dDst:dBias): %s:%s:%s:%s, Shape: %zux%zux%zux%zux%zu ";
        printf(format, g_precisionMap[diffSrcElemTypeIndex].first.c_str(),
            g_precisionMap[diffWeightsElemTypeIndex].first.c_str(), g_precisionMap[diffDstElemTypeIndex].first.c_str(),
            g_precisionMap[diffBiasElemTypeIndex].first.c_str(), dim0, dim1, dim2, dim3, dim4);
        void *diffSrc = NULL;
        void *diffWeights = NULL;
        void *diffDst = NULL;
        void *diffBias = NULL;
        try {
            KuDNN::InnerProductLayerBWDWeights ipLayerBWD1(diffDstTensorInfo, diffSrcTensorInfo,
                diffWeightsTensorInfo, diffBiasTensorInfo);
            KuDNN::InnerProductLayerBWDWeights ipLayerBWD2(ipLayerBWD1);
            ipLayerBWD2 = ipLayerBWD1;
            KuDNN::InnerProductLayerBWDWeights ipLayerBWD3(ipLayerBWD1);
            ipLayerBWD3 = std::move(ipLayerBWD1);
            KuDNN::InnerProductLayerBWDWeights ipLayerBWD5(std::move(ipLayerBWD2));

            KuDNN::InnerProductLayerBWDWeights ipLayerBWDWeights(diffDstTensorInfo, diffSrcTensorInfo,
                diffWeightsTensorInfo, diffBiasTensorInfo);
            diffSrc = malloc(std::max((size_t)1, dim0) * std::max((size_t)1, dim1) * std::max((size_t)1, dim2) *
                std::max((size_t)1, dim3) * std::max((size_t)1, dim4) * diffSrcElemSz);
            if (diffSrc == NULL) {
                std::cerr<<"memory allocation failed"<<std::endl;
                return;
            }

            diffWeights = malloc(std::max((size_t)1, dim0) * std::max((size_t)1, dim1) *
                std::max((size_t)1, dim2) * std::max((size_t)1, dim3) * std::max((size_t)1, dim4) * diffWeightsElemSz);
            if (diffWeights == NULL) {
                free(diffSrc);
                std::cerr<<"memory allocation failed"<<std::endl;
                return;
            }

            diffDst = malloc(std::max((size_t)1, dim0) * std::max((size_t)1, dim0) * diffDstElemSz);
            if (diffDst == NULL) {
                free(diffSrc);
                free(diffWeights);
                std::cerr<<"memory allocation failed"<<std::endl;
                return;
            }

            diffBias = malloc(std::max((size_t)1, dim0) * diffBiasElemSz);
            if (diffBias == NULL) {
                free(diffSrc);
                free(diffWeights);
                free(diffDst);
                std::cerr<<"memory allocation failed"<<std::endl;
                return;
            }

            ipLayerBWDWeights.Run(diffDst, diffSrc, diffWeights, diffBias);
            free(diffDst);
            free(diffSrc);
            free(diffWeights);
            free(diffBias);
        } catch (const KuDNN::Service::Unsupported& e) {
            if (diffDst != NULL) {
                free(diffDst);
            }
            if (diffSrc != NULL) {
                free(diffSrc);
            }
            if (diffWeights != NULL) {
                free(diffWeights);
            }
            if (diffBias != NULL) {
                free(diffBias);
            }
            printf("Unsupported Configuration");
        } catch (const KuDNN::Service::LogicError& e) {
            if (diffDst != NULL) {
                free(diffDst);
            }
            if (diffSrc != NULL) {
                free(diffSrc);
            }
            if (diffWeights != NULL) {
                free(diffWeights);
            }
            if (diffBias != NULL) {
                free(diffBias);
            }
            printf("Bad shapes' arguments");
        }
        printf("\n");
    }
    DT_FUZZ_END()
}

static constexpr int g_ipFuzzTestCount = 12000;

static Cmdinfo g_ipCmdTable[] = {
    {"Inner_Product_Fwd", DtInnerProductFwd, g_ipFuzzTestCount},
    {"Inner_Product_Bwd", DtInnerProductBwd, g_ipFuzzTestCount},
    {"Inner_Product_Bwd_Weights", DtInnerProductBWDWeights, g_ipFuzzTestCount},
};

int main(int argc, char *argv[])
{
    DT_Enable_Leak_Check(0, 0);
    if (argc > 1) {
        g_ipCmdTable[0] = {"Inner_Product_Fwd", DtInnerProductFwd, atoi(argv[1])};
        g_ipCmdTable[1] = {"Inner_Product_Bwd", DtInnerProductBwd, atoi(argv[1])};
        g_ipCmdTable[2] = {"Inner_Product_Bwd_Weights", DtInnerProductBWDWeights, atoi(argv[1])};
    }

    int testCount = 0;
    int numOfTestItem = sizeof(g_ipCmdTable) / sizeof(g_ipCmdTable[0]);

    if (argc > 2) {
        const char* inputCmd = argv[2];
        for (int testCaseIdx = 0; testCaseIdx < numOfTestItem; testCaseIdx++) {
            if (strcmp(g_ipCmdTable[testCaseIdx].cmd, inputCmd) == 0) {
                testCount = g_ipCmdTable[testCaseIdx].testCount;
                printf("g_ipCmdTable[%d] = %s test %d time\n", testCaseIdx, g_ipCmdTable[testCaseIdx].cmd,
                    testCount);
                g_ipCmdTable[testCaseIdx].func(testCount);
                break;
            }
        }
    } else {
        for (int testCaseIdx = 0; testCaseIdx < numOfTestItem; testCaseIdx++) {
            testCount = g_ipCmdTable[testCaseIdx].testCount;
            printf("g_ipCmdTable[%d] = %s test %d time\n", testCaseIdx, g_ipCmdTable[testCaseIdx].cmd, testCount);
            g_ipCmdTable[testCaseIdx].func(testCount);
        }
    }

    return 0;
}