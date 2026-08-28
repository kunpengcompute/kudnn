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

#define WHOLE_TENSOR_INDEX           INDEX0
#define CHANNEL_SHARED_INDEX         INDEX1
#define SHARED_AXES_SCALAR_INDEX     INDEX2
#define SHARED_AXES_VECTOR_INDEX     INDEX3

static std::map<int, std::string> g_broadCastMap = {
    {WHOLE_TENSOR_INDEX, "WHOLE_TENSOR"},
    {CHANNEL_SHARED_INDEX, "CHANNEL_SHARED"},
    {SHARED_AXES_SCALAR_INDEX, "SHARED_AXES_SCALAR"},
    {SHARED_AXES_VECTOR_INDEX, "SHARED_AXES_VECTOR"},
};

#define NON_MIX_PRECISION_INDEX     INDEX0
#define MIX_PRECISION_INDEX         INDEX1

static std::map<int, std::string> g_mixPrecisionMap = {
    {NON_MIX_PRECISION_INDEX, "NON_MIX_PRECISION"},
    {MIX_PRECISION_INDEX, "MIX_PRECISION"},
};

#define SAME_LAYOUT_INDEX           INDEX0
#define DIFFERENT_LAYOUT_INDEX      INDEX1

static std::map<int, std::string> g_diffLayoutMap = {
    {SAME_LAYOUT_INDEX, "SAME_LAYOUT"},
    {DIFFERENT_LAYOUT_INDEX, "DIFFERENT_LAYOUT"},
};

#define SAME_DIM_NUM_INDEX          INDEX0
#define DIFFERENT_DIM_NUM_INDEX     INDEX1

static std::map<int, std::string> g_diffDimNumMap = {
    {SAME_DIM_NUM_INDEX, "SAME_DIM_NUM"},
    {DIFFERENT_DIM_NUM_INDEX, "DIFFERENT_DIM_NUM"},
};

#define SAME_DIM_SIZE_INDEX         INDEX0
#define DIFFERENT_DIM_SIZE_INDEX    INDEX1

static std::map<int, std::string> g_diffDimSizeMap = {
    {SAME_DIM_SIZE_INDEX, "SAME_DIM_SIZE"},
    {DIFFERENT_DIM_SIZE_INDEX, "DIFFERENT_DIM_SIZE"},
};

#define SAME_WEIGHT_SIZE_INDEX      INDEX0
#define DIFFERENT_WEIGHT_SIZE_INDEX INDEX1

static std::map<int, std::string> g_diffWSizeMap = {
    {SAME_WEIGHT_SIZE_INDEX, "SAME_WEIGHT_SIZE"},
    {DIFFERENT_WEIGHT_SIZE_INDEX, "DIFFERENT_WEIGHT_SIZE"},
};

static constexpr int g_dimRangeInitVal = 1;
static constexpr int g_dimRangeLowerBound = 0;
static constexpr int g_dimRangeUpperBound = 30;

static void DtFuzzPreluForward(int counts)
{
    char funcName[] = "DtFuzzPreLUForward";
    DT_FUZZ_START(0, counts, funcName, 0)
    {
        int eleIndex = 0;
        int srcElemTypeIndex = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], 0, 0, g_precisionMap.size() - 1);
        int weiElemTypeIndex = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], 0, 0, g_precisionMap.size() - 1);
        int dstElemTypeIndex = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], 0, 0, g_precisionMap.size() - 1);
        int broadcastIndex = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], 0, 0, g_broadCastMap.size() - 1);
        int mixPrecIndex = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], 0, 0, g_mixPrecisionMap.size() - 1);
        int diffLayoutIndex = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], 0, 0, g_diffLayoutMap.size() - 1);
        int diffDimNumIndex = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], 0, 0, g_diffDimNumMap.size() - 1);
        int diffDimSizeIndex = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], 0, 0, g_diffDimSizeMap.size() - 1);
        int diffWSizeIndex = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], 0, 0, g_diffWSizeMap.size() - 1);
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
        size_t weiDim0 = dim0;
        size_t weiDim1 = dim1;
        size_t weiDim2 = dim2;
        size_t weiDim3 = dim3;
        size_t weiDim4 = dim4;
        // mix precision or not
        switch (mixPrecIndex) {
            case NON_MIX_PRECISION_INDEX:
                weiElemTypeIndex = srcElemTypeIndex;
                dstElemTypeIndex = srcElemTypeIndex;
                break;
            default :
                break;
        }

        // broadcast type
        switch (broadcastIndex) {
            case WHOLE_TENSOR_INDEX:
                break;
            case CHANNEL_SHARED_INDEX:
                weiDim0 = 1;
                weiDim1 = 1;
                weiDim2 = 1;
                weiDim3 = 1;
                weiDim4 = 1;
                break;
            case SHARED_AXES_SCALAR_INDEX:
                weiDim1 = 1;
                weiDim2 = 1;
                break;
            default: // SHARED_AXES_VECTOR
                weiDim1 = 1;
                weiDim3 = 1;
                weiDim4 = 1;
                break;
        }

        KuDNN::Element::Type srcType(g_precisionMap[srcElemTypeIndex].second);
        KuDNN::Element::Type weiType(g_precisionMap[weiElemTypeIndex].second);
        KuDNN::Element::Type dstType(g_precisionMap[dstElemTypeIndex].second);
        size_t srcElemSz = srcType.GetSize();
        size_t weiElemSz = weiType.GetSize();
        size_t dstElemSz = dstType.GetSize();

        const KuDNN::TensorInfo srcTensorInfo = {{dim0, dim1, dim2, dim3, dim4 + (diffDimSizeIndex ? 1 : 0)},
            srcType, KuDNN::Layout::ABCDE};
        const KuDNN::TensorInfo weiTensorInfo = diffDimNumIndex ?
            KuDNN::TensorInfo{{weiDim0, weiDim1, weiDim2, weiDim3}, weiType, KuDNN::Layout::ABCD} :
            KuDNN::TensorInfo{{weiDim0, weiDim1, weiDim2, weiDim3, weiDim4}, weiType,
                diffWSizeIndex ? KuDNN::Layout::ACDEB : KuDNN::Layout::ABCDE};
        const KuDNN::TensorInfo dstTensorInfo = {{dim0, dim1, dim2, dim3, dim4}, dstType,
            diffLayoutIndex ? KuDNN::Layout::ACDEB : KuDNN::Layout::ABCDE};
        char format[] = "Forward PreLU, Precision(src:wei:dst): %s:%s:%s, Shape(src:weight): "
            "%zux%zux%zux%zux%zu:%zux%zux%zux%zux%zu ";
        printf(format, g_precisionMap[srcElemTypeIndex].first.c_str(), g_precisionMap[weiElemTypeIndex].first.c_str(),
            g_precisionMap[dstElemTypeIndex].first.c_str(), dim0, dim1, dim2, dim3, dim4, weiDim0, weiDim1, weiDim2,
            weiDim3, weiDim4);
        void *src = NULL;
        void *wei = NULL;
        void *dst = NULL;
        try {
            KuDNN::PReLULayerFWD prelu1(srcTensorInfo, weiTensorInfo, dstTensorInfo);
            KuDNN::PReLULayerFWD prelu2(prelu1);
            prelu2 = prelu1;
            KuDNN::PReLULayerFWD prelu4(prelu1);
            prelu4 = std::move(prelu1);
            KuDNN::PReLULayerFWD prelu5(std::move(prelu2));

            KuDNN::PReLULayerFWD preluLayer(srcTensorInfo, weiTensorInfo, dstTensorInfo);
            src = malloc(std::max((size_t)1, dim0) * std::max((size_t)1, dim1) * std::max((size_t)1, dim2) *
                std::max((size_t)1, dim3) * std::max((size_t)1, dim4) *  srcElemSz);
            if (src == NULL) {
                std::cerr<<"memory allocation failed"<<std::endl;
                return;
            }
            wei = malloc(std::max((size_t)1, weiDim0) * std::max((size_t)1, weiDim1) *
                std::max((size_t)1, weiDim2) * std::max((size_t)1, weiDim3) * std::max((size_t)1, weiDim4) * weiElemSz);
            if (wei == NULL) {
                free(src);
                std::cerr<<"memory allocation failed"<<std::endl;
                return;
            }
            dst = malloc(std::max((size_t)1, dim0) * std::max((size_t)1, dim1) * std::max((size_t)1, dim2) *
                std::max((size_t)1, dim3) * std::max((size_t)1, dim4) * dstElemSz);
            if (dst == NULL) {
                free(src);
                free(wei);
                std::cerr<<"memory allocation failed"<<std::endl;
                return;
            }
            preluLayer.Run(src, wei, dst);
            free(src);
            free(wei);
            free(dst);
        } catch (const KuDNN::Service::Unsupported& e) {
            if (src != NULL) {
                free(src);
            }
            if (wei != NULL) {
                free(wei);
            }
            if (dst != NULL) {
                free(dst);
            }
            printf("Unsupported Configuration");
        } catch (const KuDNN::Service::LogicError& e) {
            if (src != NULL) {
                free(src);
            }
            if (wei != NULL) {
                free(wei);
            }
            if (dst != NULL) {
                free(dst);
            }
            printf("Bad shapes' arguments");
        }

        printf("\n");
    }
    DT_FUZZ_END()
}

static void DtFuzzPreluBackward(int counts)
{
    char funcName[] = "DtFuzzPreLUBackward";
    DT_FUZZ_START(0, counts, funcName, 0)
    {
        int eleIndex = 0;
        int srcElemTypeIndex = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], 0, 0, g_precisionMap.size() - 1);
        int weiElemTypeIndex = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], 0, 0, g_precisionMap.size() - 1);
        int diffDstElemTypeIndex = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], 0, 0,
            g_precisionMap.size() - 1);
        int broadcastIndex = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], 0, 0, g_broadCastMap.size() - 1);
        int mixPrecIndex = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], 0, 0, g_mixPrecisionMap.size() - 1);
        int diffLayoutIndex = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], 0, 0, g_diffLayoutMap.size() - 1);
        int diffDimSizeIndex = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], 0, 0, g_diffDimSizeMap.size() - 1);
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
        size_t weiDim0 = dim0;
        size_t weiDim1 = dim1;
        size_t weiDim2 = dim2;
        size_t weiDim3 = dim3;
        size_t weiDim4 = dim4;
        // mix precision or not
        switch (mixPrecIndex) {
            case NON_MIX_PRECISION_INDEX:
                weiElemTypeIndex = srcElemTypeIndex;
                diffDstElemTypeIndex = srcElemTypeIndex;
                break;
            default :
                break;
        }
        // broadcast type
        switch (broadcastIndex) {
            case WHOLE_TENSOR_INDEX:
                break;
            case CHANNEL_SHARED_INDEX:
                weiDim0 = 1;
                weiDim1 = 1;
                weiDim2 = 1;
                weiDim3 = 1;
                weiDim4 = 1;
                break;
            case SHARED_AXES_SCALAR_INDEX:
                weiDim1 = 1;
                weiDim2 = 1;
                break;
            default: // SHARED_AXES_VECTOR
                weiDim1 = 1;
                weiDim3 = 1;
                weiDim4 = 1;
                break;
        }

        KuDNN::Element::Type srcType(g_precisionMap[srcElemTypeIndex].second);
        KuDNN::Element::Type diffSrcType = srcType;
        KuDNN::Element::Type weiType(g_precisionMap[weiElemTypeIndex].second);
        KuDNN::Element::Type diffWeiType = weiType;
        KuDNN::Element::Type diffDstType(g_precisionMap[diffDstElemTypeIndex].second);

        size_t srcElemSz = srcType.GetSize();
        size_t diffSrcElemSz = diffSrcType.GetSize();
        size_t weiElemSz = weiType.GetSize();
        size_t diffWeiElemSz = diffWeiType.GetSize();
        size_t diffDstElemSz = diffDstType.GetSize();

        const KuDNN::TensorInfo srcTensorInfo = {{dim0, dim1, dim2, dim3, dim4 + (diffDimSizeIndex ? 1 : 0)},
            srcType, KuDNN::Layout::ABCDE};
        const KuDNN::TensorInfo diffSrcTensorInfo = {{dim0, dim1, dim2, dim3, dim4}, diffSrcType, KuDNN::Layout::ABCDE};
        const KuDNN::TensorInfo weiTensorInfo = {{weiDim0, weiDim1, weiDim2, weiDim3, weiDim4}, weiType,
            diffLayoutIndex ? KuDNN::Layout::ACDEB : KuDNN::Layout::ABCDE};
        const KuDNN::TensorInfo diffWeiTensorInfo = {{weiDim0, weiDim1, weiDim2, weiDim3, weiDim4}, diffWeiType,
            KuDNN::Layout::ABCDE};
        const KuDNN::TensorInfo diffDstTensorInfo = {{dim0, dim1, dim2, dim3, dim4}, diffDstType, KuDNN::Layout::ABCDE};
        char format[] = "Backward PreLU, Precision(src:wei:dst): %s:%s:%s, Shape(src:weight): "
            "%zux%zux%zux%zux%zu:%zux%zux%zux%zux%zu ";
        printf(format, g_precisionMap[srcElemTypeIndex].first.c_str(), g_precisionMap[weiElemTypeIndex].first.c_str(),
            g_precisionMap[diffDstElemTypeIndex].first.c_str(), dim0, dim1, dim2, dim3, dim4, weiDim0, weiDim1, weiDim2,
            weiDim3, weiDim4);
        void *src = NULL;
        void *diffSrc = NULL;
        void *wei = NULL;
        void *diffWei = NULL;
        void *diffDst = NULL;
        try {
            KuDNN::PReLULayerBWD prelu1(srcTensorInfo, diffSrcTensorInfo, weiTensorInfo, diffWeiTensorInfo,
                diffDstTensorInfo);
            KuDNN::PReLULayerBWD prelu3(prelu1);
            prelu3 = prelu1;
            KuDNN::PReLULayerBWD prelu4(prelu1);
            prelu4 = std::move(prelu1);
            KuDNN::PReLULayerBWD prelu5(std::move(prelu3));

            KuDNN::PReLULayerBWD preluLayer(srcTensorInfo, diffSrcTensorInfo, weiTensorInfo, diffWeiTensorInfo,
                diffDstTensorInfo);
            src = malloc(std::max((size_t)1, dim0) * std::max((size_t)1, dim1) * std::max((size_t)1, dim2) *
                std::max((size_t)1, dim3) * std::max((size_t)1, dim4) * srcElemSz);
            if (src == NULL) {
                std::cerr<<"memory allocation failed"<<std::endl;
                return;
            }
            diffSrc = malloc(std::max((size_t)1, dim0) * std::max((size_t)1, dim1) * std::max((size_t)1, dim2) *
                std::max((size_t)1, dim3) * std::max((size_t)1, dim4) * diffSrcElemSz);
            if (diffSrc == NULL) {
                free(src);
                std::cerr<<"memory allocation failed"<<std::endl;
                return;
            }
            wei = malloc(std::max((size_t)1, weiDim0) * std::max((size_t)1, weiDim1) *
                std::max((size_t)1, weiDim2) * std::max((size_t)1, weiDim3) * std::max((size_t)1, weiDim4) * weiElemSz);
            if (wei == NULL) {
                free(src);
                free(diffSrc);
                std::cerr<<"memory allocation failed"<<std::endl;
                return;
            }
            diffWei = malloc(std::max((size_t)1, weiDim0) * std::max((size_t)1, weiDim1) *
                std::max((size_t)1, weiDim2) * std::max((size_t)1, weiDim3) * std::max((size_t)1, weiDim4) *
                diffWeiElemSz);
            if (diffWei == NULL) {
                free(src);
                free(diffSrc);
                free(wei);
                std::cerr<<"memory allocation failed"<<std::endl;
                return;
            }
            diffDst = malloc(std::max((size_t)1, dim0) * std::max((size_t)1, dim1) * std::max((size_t)1, dim2) *
                std::max((size_t)1, dim3) * std::max((size_t)1, dim4) * diffDstElemSz);
            if (diffDst == NULL) {
                free(src);
                free(diffSrc);
                free(wei);
                free(diffWei);
                std::cerr<<"memory allocation failed"<<std::endl;
                return;
            }
            preluLayer.Run(src, diffSrc, wei, diffWei, diffDst);
            free(src);
            free(diffSrc);
            free(wei);
            free(diffWei);
            free(diffDst);
        } catch (const KuDNN::Service::Unsupported& e) {
            if (src != NULL) {
                free(src);
            }
            if (diffSrc != NULL) {
                free(diffSrc);
            }
            if (wei != NULL) {
                free(wei);
            }
            if (diffWei != NULL) {
                free(diffWei);
            }
            if (diffDst != NULL) {
                free(diffDst);
            }
            printf("Unsupported Configuration");
        } catch (const KuDNN::Service::LogicError& e) {
            if (src != NULL) {
                free(src);
            }
            if (diffSrc != NULL) {
                free(diffSrc);
            }
            if (wei != NULL) {
                free(wei);
            }
            if (diffWei != NULL) {
                free(diffWei);
            }
            if (diffDst != NULL) {
                free(diffDst);
            }
            printf("Bad shapes' arguments");
        }

        printf("\n");
    }
    DT_FUZZ_END()
}

static constexpr int g_preluFuzzTestCount = 10000;

static Cmdinfo g_preluCmdTable[] = {
    {"Forward PreLU", DtFuzzPreluForward, g_preluFuzzTestCount},
    {"Backward PreLU", DtFuzzPreluBackward, g_preluFuzzTestCount},
};

int main(int argc, char *argv[])
{
    DT_Enable_Leak_Check(0, 0);
    if (argc > 1) {
        g_preluCmdTable[0] = {"ForwardPreLU", DtFuzzPreluForward, atoi(argv[1])};
        g_preluCmdTable[1] = {"BackwardPreLU", DtFuzzPreluBackward, atoi(argv[1])};
    }

    int numOfTestItem = sizeof(g_preluCmdTable) / sizeof(g_preluCmdTable[0]);
    int testCount = 0;
    if (argc > 2) {
        const char* inputCmd = argv[2];
        for (int testCaseIdx = 0; testCaseIdx < numOfTestItem; testCaseIdx++) {
            if (strcmp(g_preluCmdTable[testCaseIdx].cmd, inputCmd) == 0) {
                testCount = g_preluCmdTable[testCaseIdx].testCount;
                printf("g_preluCmdTable[%d] = %s test %d time\n", testCaseIdx, g_preluCmdTable[testCaseIdx].cmd,
                    testCount);
                g_preluCmdTable[testCaseIdx].func(testCount);
                break;
            }
        }
    } else {
        for (int testCaseIdx = 0; testCaseIdx < numOfTestItem; testCaseIdx++) {
            testCount = g_preluCmdTable[testCaseIdx].testCount;
            printf("g_preluCmdTable[%d] = %s test %d time\n", testCaseIdx, g_preluCmdTable[testCaseIdx].cmd, testCount);
            g_preluCmdTable[testCaseIdx].func(testCount);
        }
    }

    return 0;
}