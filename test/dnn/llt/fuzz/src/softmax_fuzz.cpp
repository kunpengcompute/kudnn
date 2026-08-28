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

static std::map<int, std::pair<std::string, std::uint32_t>> g_flagsMap = {
    {INDEX0, {"SOFTMAX", static_cast<std::uint32_t>(KuDNN::AlgorithmKind::SOFTMAX)}},
    {INDEX1, {"LOGSOFTMAX", static_cast<std::uint32_t>(KuDNN::AlgorithmKind::LOGSOFTMAX)}},
};

static constexpr int g_dimRangeInitVal = 1;
static constexpr int g_dimRangeLowerBound = 1;
static constexpr int g_dimRangeUpperBound = 30;

static void DtFuzzSoftmaxForward(int counts)
{
    char funcName[] = "DtFuzzSoftmaxForward";
    DT_FUZZ_START(0, counts, funcName, 0)
    {
        int eleIndex = 0;
        int srcElemTypeIndex = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], 0, 0, g_precisionMap.size() - 1);
        int dstElemTypeIndex = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], 0, 0, g_precisionMap.size() - 1);
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

        size_t axis = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], 0, 0, 4); // 5D Tensor, axis range [0, 4]
        KuDNN::AlgorithmKind algorithm(static_cast<KuDNN::AlgorithmKind>(g_flagsMap[flagsTypeIndex].second));
        KuDNN::Element::Type srcType(g_precisionMap[srcElemTypeIndex].second);
        KuDNN::Element::Type dstType(g_precisionMap[dstElemTypeIndex].second);
        size_t srcElemSz = srcType.GetSize();
        size_t dstElemSz = dstType.GetSize();

        const KuDNN::TensorInfo srcTensorInfo = {{dim0, dim1, dim2, dim3, dim4}, srcType, KuDNN::Layout::ABCDE};
        const KuDNN::TensorInfo dstTensorInfo = {{dim0, dim1, dim2, dim3, dim4}, dstType, KuDNN::Layout::ABCDE};
        char format[] = "Forward Softmax, Precision(src:dst): %s:%s, Shape: %zux%zux%zux%zux%zu, axis:%zu ";
        printf(format, g_precisionMap[srcElemTypeIndex].first.c_str(), g_precisionMap[dstElemTypeIndex].first.c_str(),
            dim0, dim1, dim2, dim3, dim4, axis);
        void *src = NULL;
        void *dst = NULL;
        try {
            KuDNN::SoftmaxLayerFWD softmax1(srcTensorInfo, dstTensorInfo, axis, algorithm);
            KuDNN::SoftmaxLayerFWD softmax2(softmax1);
            softmax2 = softmax1;

            KuDNN::SoftmaxLayerFWD softmax3(softmax1);
            softmax3 = std::move(softmax1);
            KuDNN::SoftmaxLayerFWD softmax5(std::move(softmax2));

            KuDNN::SoftmaxLayerFWD softmaxLayer(srcTensorInfo, dstTensorInfo, axis, algorithm);
            src = malloc(std::max((size_t)1, dim0) * std::max((size_t)1, dim1) * std::max((size_t)1, dim2) *
                std::max((size_t)1, dim3) * std::max((size_t)1, dim4) * srcElemSz);
            if (src == NULL) {
                std::cerr<<"memory allocation failed"<<std::endl;
                return;
            }
            dst = malloc(std::max((size_t)1, dim0) * std::max((size_t)1, dim1) * std::max((size_t)1, dim2) *
                std::max((size_t)1, dim3) * std::max((size_t)1, dim4) * dstElemSz);
            if (dst == NULL) {
                free(src);
                std::cerr<<"memory allocation failed"<<std::endl;
            }
            softmaxLayer.Run(src, dst);
            free(src);
            free(dst);
        } catch (const KuDNN::Service::Unsupported& e) {
            if (src != NULL) {
                free(src);
            }
            if (dst != NULL) {
                free(dst);
            }
            printf("Unsupported Configuration");
        } catch (const KuDNN::Service::LogicError& e) {
            if (src != NULL) {
                free(src);
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

static void DtFuzzSoftmaxBackward(int counts)
{
    char funcName[] = "DtFuzzSoftmaxBackward";
    DT_FUZZ_START(0, counts, funcName, 0)
    {
        int eleIndex = 0;
        int srcElemTypeIndex = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], 0, 0, g_precisionMap.size() - 1);
        int dstElemTypeIndex = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], 0, 0, g_precisionMap.size() - 1);
        int dstDiffElemTypeIndex = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], 0, 0,
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

        size_t axis = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], 0, 0, 4); // 5D Tensor, axis range [0, 4]
        KuDNN::AlgorithmKind algorithm(static_cast<KuDNN::AlgorithmKind>(g_flagsMap[flagsTypeIndex].second));
        KuDNN::Element::Type srcDiffType(g_precisionMap[srcElemTypeIndex].second);
        KuDNN::Element::Type dstType(g_precisionMap[dstElemTypeIndex].second);
        KuDNN::Element::Type dstDiffType(g_precisionMap[dstDiffElemTypeIndex].second);
        size_t srcDiffElemSz = srcDiffType.GetSize();
        size_t dstElemSz = dstType.GetSize();
        size_t dstDiffElemSz = dstDiffType.GetSize();

        const KuDNN::TensorInfo srcDiffTensorInfo = {{dim0, dim1, dim2, dim3, dim4}, srcDiffType, KuDNN::Layout::ABCDE};
        const KuDNN::TensorInfo dstTensorInfo = {{dim0, dim1, dim2, dim3, dim4}, dstType, KuDNN::Layout::ABCDE};
        const KuDNN::TensorInfo dstDiffTensorInfo = {{dim0, dim1, dim2, dim3, dim4}, dstDiffType, KuDNN::Layout::ABCDE};
        char format[] = "Backward Softmax, Precision(srcDiff:dst:dstDiff): %s:%s:%s, Shape: %zux%zux%zux%zux%zu, "
            "axis:%zu ";
        printf(format, g_precisionMap[srcElemTypeIndex].first.c_str(), g_precisionMap[dstElemTypeIndex].first.c_str(),
            g_precisionMap[dstDiffElemTypeIndex].first.c_str(), dim0, dim1, dim2, dim3, dim4, axis);
        void *srcDiff = NULL;
        void *dst = NULL;
        void *dstDiff = NULL;
        try {
            KuDNN::SoftmaxLayerBWD softmax1(dstTensorInfo, dstDiffTensorInfo, srcDiffTensorInfo, axis, algorithm);
            KuDNN::SoftmaxLayerBWD softmax2(softmax1);
            softmax2 = softmax1;
            KuDNN::SoftmaxLayerBWD softmax3(softmax1);
            KuDNN::SoftmaxLayerBWD softmax4 = std::move(softmax1);
            KuDNN::SoftmaxLayerBWD softmax5(std::move(softmax2));

            KuDNN::SoftmaxLayerBWD softmaxLayer(dstTensorInfo, dstDiffTensorInfo, srcDiffTensorInfo, axis, algorithm);
            srcDiff = malloc(std::max((size_t)1, dim0) * std::max((size_t)1, dim1) * std::max((size_t)1, dim2) *
                std::max((size_t)1, dim3) * std::max((size_t)1, dim4) * srcDiffElemSz);
            if (srcDiff == NULL) {
                std::cerr<<"memory allocation failed"<<std::endl;
                return;
            }
            dst = malloc(std::max((size_t)1, dim0) * std::max((size_t)1, dim1) * std::max((size_t)1, dim2) *
                std::max((size_t)1, dim3) * std::max((size_t)1, dim4) * dstElemSz);
            if (dst == NULL) {
                free(srcDiff);
                std::cerr<<"memory allocation failed"<<std::endl;
            }
            dstDiff = malloc(std::max((size_t)1, dim0) * std::max((size_t)1, dim1) * std::max((size_t)1, dim2) *
                std::max((size_t)1, dim3) * std::max((size_t)1, dim4) * dstDiffElemSz);
            if (dstDiff == NULL) {
                free(srcDiff);
                free(dstDiff);
                std::cerr<<"memory allocation failed"<<std::endl;
            }
            softmaxLayer.Run(dst, dstDiff, srcDiff);
            free(srcDiff);
            free(dst);
            free(dstDiff);
        } catch (const KuDNN::Service::Unsupported& e) {
            if (srcDiff != NULL) {
                free(srcDiff);
            }
            if (dst != NULL) {
                free(dst);
            }
            if (dstDiff != NULL) {
                free(dstDiff);
            }
            printf("Unsupported Configuration");
        } catch (const KuDNN::Service::LogicError& e) {
            if (srcDiff != NULL) {
                free(srcDiff);
            }
            if (dst != NULL) {
                free(dst);
            }
            if (dstDiff != NULL) {
                free(dstDiff);
            }
            printf("Bad shapes' arguments");
        }
        printf("\n");
    }
    DT_FUZZ_END()
}

static constexpr int g_softmaxFuzzTestCount = 25000;

static Cmdinfo g_softmaxCmdTable[] = {
    {"ForwardSoftmax", DtFuzzSoftmaxForward, g_softmaxFuzzTestCount},
    {"BackwardSoftmax", DtFuzzSoftmaxBackward, g_softmaxFuzzTestCount},
};

int main(int argc, char *argv[])
{
    DT_Enable_Leak_Check(0, 0);
    if (argc > 1) {
        g_softmaxCmdTable[0] = {"ForwardSoftmax", DtFuzzSoftmaxForward, atoi(argv[1])};
        g_softmaxCmdTable[1] = {"BackwardSoftmax", DtFuzzSoftmaxBackward, atoi(argv[1])};
    }

    int numOfTestItem = sizeof(g_softmaxCmdTable) / sizeof(g_softmaxCmdTable[0]);
    int testCount = 0;
    if (argc > 2) {
        const char* inputCmd = argv[2];
        for (int testCaseIdx = 0; testCaseIdx < numOfTestItem; testCaseIdx++) {
            if (strcmp(g_softmaxCmdTable[testCaseIdx].cmd, inputCmd) == 0) {
                testCount = g_softmaxCmdTable[testCaseIdx].testCount;
                printf("g_softmaxCmdTable[%d] = %s test %d time\n", testCaseIdx, g_softmaxCmdTable[testCaseIdx].cmd,
                    testCount);
                g_softmaxCmdTable[testCaseIdx].func(testCount);
                break;
            }
        }
    } else {
        for (int testCaseIdx = 0; testCaseIdx < numOfTestItem; testCaseIdx++) {
            printf("g_softmaxCmdTable[%d] = %s test %d time\n", testCaseIdx, g_softmaxCmdTable[testCaseIdx].cmd,
                testCount);
            testCount = g_softmaxCmdTable[testCaseIdx].testCount;
            g_softmaxCmdTable[testCaseIdx].func(testCount);
        }

    }

    return 0;
}