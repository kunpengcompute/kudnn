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

static std::map<int, std::pair<std::string, KuDNN::ReductionFunction>> g_reductionFuncMap = {
    {INDEX0, {"MAX", KuDNN::ReductionFunction::MAX}},
    {INDEX1, {"MIN", KuDNN::ReductionFunction::MIN}},
    {INDEX2, {"SUM", KuDNN::ReductionFunction::SUM}},
    {INDEX3, {"SUM", KuDNN::ReductionFunction::SUM}},
    {INDEX4, {"MUL", KuDNN::ReductionFunction::MUL}},
    {INDEX5, {"MEAN", KuDNN::ReductionFunction::MEAN}}
};

static std::map<int, std::pair<std::string, KuDNN::ReductionFunction>> g_normLpFuncMap = {
    {INDEX0, {"NORM_LP_MAX", KuDNN::ReductionFunction::NORM_LP_MAX}},
    {INDEX1, {"NORM_LP_SUM", KuDNN::ReductionFunction::NORM_LP_SUM}},
    {INDEX2, {"NORM_LP_POWER_P_MAX", KuDNN::ReductionFunction::NORM_LP_POWER_P_MAX}},
    {INDEX3, {"NORM_LP_POWER_P_SUM", KuDNN::ReductionFunction::NORM_LP_POWER_P_SUM}},
};

static std::map<int, std::pair<std::string, KuDNN::Element::TypeT>> g_precisionMap = {
    {INDEX0, {"F32", KuDNN::Element::TypeT::F32}},
    {INDEX1, {"F16", KuDNN::Element::TypeT::F16}},
    {INDEX2, {"BF16", KuDNN::Element::TypeT::BF16}},
    {INDEX3, {"S32", KuDNN::Element::TypeT::S32}},
    {INDEX4, {"S8", KuDNN::Element::TypeT::S8}},
    {INDEX5, {"U8", KuDNN::Element::TypeT::U8}}
};

static std::map<int, std::pair<std::string, KuDNN::Element::TypeT>> g_normLpPrecisionMap = {
    {INDEX0, {"F32", KuDNN::Element::TypeT::F32}},
    {INDEX1, {"F16", KuDNN::Element::TypeT::F16}},
    {INDEX2, {"BF16", KuDNN::Element::TypeT::BF16}},
};

static std::map<int, bool> g_lastDimIsOneMap = {
    {INDEX0, true},
    {INDEX1, false},
};

static constexpr int g_dimRangeInitVal = 1;
static constexpr int g_dimRangeLowerBound = 1;
static constexpr int g_dimRangeUpperBound = 30;

static void DtFuzzReduction(int counts)
{
    char funcName[] = "DtFuzzReduction";
    DT_FUZZ_START(0, counts, funcName, 0)
    {
        int eleIndex = 0;
        int src0ElemTypeIndex = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], 0, 0, g_precisionMap.size() - 1);
        int dstElemTypeIndex = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], 0, 0, g_precisionMap.size() - 1);
        int opIndex = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], 0, 0, g_reductionFuncMap.size() - 1);

        std::array<int, 5> oneDimsIndex;
        oneDimsIndex[0] = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], 0, 0, g_lastDimIsOneMap.size() - 1);
        oneDimsIndex[1] = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], 0, 0, g_lastDimIsOneMap.size() - 1);
        oneDimsIndex[2] = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], 0, 0, g_lastDimIsOneMap.size() - 1);
        oneDimsIndex[3] = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], 0, 0, g_lastDimIsOneMap.size() - 1);
        oneDimsIndex[4] = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], 0, 0, g_lastDimIsOneMap.size() - 1);

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

        KuDNN::Element::Type src0Type(g_precisionMap[src0ElemTypeIndex].second);
        KuDNN::Element::Type dstType(g_precisionMap[dstElemTypeIndex].second);
        size_t src0ElemSz = src0Type.GetSize();
        size_t dstElemSz = dstType.GetSize();

        std::array<size_t, 5> dstDims = {dim0, dim1, dim2, dim3, dim4};
        for (int i = 0; i < 5; i++) {
            if (g_lastDimIsOneMap[oneDimsIndex[i]]) {
                dstDims[i] = 1;
            }
        }

        const KuDNN::TensorInfo src0TensorInfo = {{dim0, dim1, dim2, dim3, dim4}, src0Type, KuDNN::Layout::ABCDE};
        const KuDNN::TensorInfo dstTensorInfo = {{dstDims[0], dstDims[1], dstDims[2], dstDims[3], dstDims[4]},
            dstType, KuDNN::Layout::ABCDE};
        KuDNN::ReductionFunction kind = g_reductionFuncMap[opIndex].second;
        char format[] = "Reduction op: %s, Precision(src0:dst): %s:%s, Shape: %zux%zux%zux%zux%zu:%zux%zux%zux%zux%zu ";
        printf(format, g_reductionFuncMap[opIndex].first.c_str(), g_precisionMap[src0ElemTypeIndex].first.c_str(),
            g_precisionMap[dstElemTypeIndex].first.c_str(), dim0, dim1, dim2, dim3, dim4,
            dstDims[0], dstDims[1], dstDims[2], dstDims[3], dstDims[4]);
        void *dst = nullptr;
        void *src0 = nullptr;
        try {
            KuDNN::ReductionLayer reduction1(src0TensorInfo, dstTensorInfo, kind);
            KuDNN::ReductionLayer reduction2 = reduction1;
            KuDNN::ReductionLayer reduction3(reduction1);
            reduction3 = reduction2;
            KuDNN::ReductionLayer reduction4(src0TensorInfo, dstTensorInfo, kind);
            reduction4 = std::move(reduction2);
            KuDNN::ReductionLayer reduction5(std::move(reduction1));

            KuDNN::ReductionLayer reductionLayer(src0TensorInfo, dstTensorInfo, kind);

            src0 = malloc(std::max((size_t)1, dim0) * std::max((size_t)1, dim1) * std::max((size_t)1, dim2) *
                std::max((size_t)1, dim3) * std::max((size_t)1, dim4) * src0ElemSz);
            if (src0 == NULL) {
                std::cerr<<"memory allocation failed"<<std::endl;
                return;
            }

            dst = malloc(std::max((size_t)1, dstDims[0]) * std::max((size_t)1, dstDims[1]) * std::max((size_t)1, dstDims[2]) *
                std::max((size_t)1, dstDims[3]) * std::max((size_t)1, dstDims[4]) * dstElemSz);
            if (dst == NULL) {
                free(src0);
                std::cerr<<"memory allocation failed"<<std::endl;
                return;
            }
            reductionLayer.Run(src0, dst);
            free(src0);
            free(dst);
        } catch (const KuDNN::Service::Unsupported& e) {
            if (dst != NULL) {
                free(dst);
            }
            if (src0 != NULL) {
                free(src0);
            }
            printf("Unsupported Configuration");
        } catch (const KuDNN::Service::LogicError& e) {
            if (dst != NULL) {
                free(dst);
            }
            if (src0 != NULL) {
                free(src0);
            }
            printf("Bad shapes' arguments");
        }
        printf("\n");
    }
    DT_FUZZ_END()
}

static void DtFuzzReductionNormLp(int counts)
{
    char funcName[] = "DtFuzzReductionNormLp";
    DT_FUZZ_START(0, counts, funcName, 0)
    {
        int eleIndex = 0;
        int src0ElemTypeIndex = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], 0, 0, g_normLpPrecisionMap.size() - 1);
        int dstElemTypeIndex = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], 0, 0, g_precisionMap.size() - 1);
        int opIndex = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], 0, 0, g_normLpFuncMap.size() - 1);

        std::array<int, 5> oneDimsIndex;
        oneDimsIndex[0] = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], 0, 0, g_lastDimIsOneMap.size() - 1);
        oneDimsIndex[1] = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], 0, 0, g_lastDimIsOneMap.size() - 1);
        oneDimsIndex[2] = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], 0, 0, g_lastDimIsOneMap.size() - 1);
        oneDimsIndex[3] = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], 0, 0, g_lastDimIsOneMap.size() - 1);
        oneDimsIndex[4] = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], 0, 0, g_lastDimIsOneMap.size() - 1);

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

        KuDNN::Element::Type src0Type(g_normLpPrecisionMap[src0ElemTypeIndex].second);
        KuDNN::Element::Type dstType(g_precisionMap[dstElemTypeIndex].second);
        size_t src0ElemSz = src0Type.GetSize();
        size_t dstElemSz = dstType.GetSize();

        std::array<size_t, 5> dstDims = {dim0, dim1, dim2, dim3, dim4};
        for (int i = 0; i < 5; i++) {
            if (g_lastDimIsOneMap[oneDimsIndex[i]]) {
                dstDims[i] = 1;
            }
        }

        const KuDNN::TensorInfo src0TensorInfo = {{dim0, dim1, dim2, dim3, dim4}, src0Type, KuDNN::Layout::ABCDE};
        const KuDNN::TensorInfo dstTensorInfo = {{dstDims[0], dstDims[1], dstDims[2], dstDims[3], dstDims[4]},
            dstType, KuDNN::Layout::ABCDE};
        KuDNN::ReductionFunction kind = g_normLpFuncMap[opIndex].second;
        char format[] = "Reduction op: %s, Precision(src0:dst): %s:%s, Shape: %zux%zux%zux%zux%zu:%zux%zux%zux%zux%zu ";
        printf(format, g_normLpFuncMap[opIndex].first.c_str(), g_normLpPrecisionMap[src0ElemTypeIndex].first.c_str(),
            g_precisionMap[dstElemTypeIndex].first.c_str(), dim0, dim1, dim2, dim3, dim4,
            dstDims[0], dstDims[1], dstDims[2], dstDims[3], dstDims[4]);
        void *dst = nullptr;
        void *src0 = nullptr;
        try {
            KuDNN::ReductionLayer reduction1(src0TensorInfo, dstTensorInfo, kind);
            KuDNN::ReductionLayer reduction2 = reduction1;
            KuDNN::ReductionLayer reduction3(reduction1);
            reduction3 = reduction2;
            KuDNN::ReductionLayer reduction4(src0TensorInfo, dstTensorInfo, kind);
            reduction4 = std::move(reduction2);
            KuDNN::ReductionLayer reduction5(std::move(reduction1));

            KuDNN::ReductionLayer reductionLayer(src0TensorInfo, dstTensorInfo, kind);

            src0 = malloc(std::max((size_t)1, dim0) * std::max((size_t)1, dim1) * std::max((size_t)1, dim2) *
                std::max((size_t)1, dim3) * std::max((size_t)1, dim4) * src0ElemSz);
            if (src0 == NULL) {
                std::cerr<<"memory allocation failed"<<std::endl;
                return;
            }

            dst = malloc(std::max((size_t)1, dstDims[0]) * std::max((size_t)1, dstDims[1]) * std::max((size_t)1, dstDims[2]) *
                std::max((size_t)1, dstDims[3]) * std::max((size_t)1, dstDims[4]) * dstElemSz);
            if (dst == NULL) {
                free(src0);
                std::cerr<<"memory allocation failed"<<std::endl;
                return;
            }
            reductionLayer.Run(src0, dst);
            free(src0);
            free(dst);
        } catch (const KuDNN::Service::Unsupported& e) {
            if (dst != NULL) {
                free(dst);
            }
            if (src0 != NULL) {
                free(src0);
            }
            printf("Unsupported Configuration");
        } catch (const KuDNN::Service::LogicError& e) {
            if (dst != NULL) {
                free(dst);
            }
            if (src0 != NULL) {
                free(src0);
            }
            printf("Bad shapes' arguments");
        }
        printf("\n");
    }
    DT_FUZZ_END()
}

static constexpr int g_reductionFuzzTestCount = 4000;

static Cmdinfo g_reductionCmdTable[] = {
    {"Reduction", DtFuzzReduction, g_reductionFuzzTestCount},
    {"ReductionNormLp", DtFuzzReductionNormLp, g_reductionFuzzTestCount},
};

int main(int argc, char *argv[])
{
    DT_Enable_Leak_Check(0, 0);
    if (argc > 1) {
        g_reductionCmdTable[0] = {"Reduction", DtFuzzReduction, atoi(argv[1])};
        g_reductionCmdTable[1] = {"ReductionNormLp", DtFuzzReductionNormLp, atoi(argv[1])};
    }

    int numOfTestItem = sizeof(g_reductionCmdTable) / sizeof(g_reductionCmdTable[0]);
    int testCount = 0;

    for (int testCaseIdx = 0; testCaseIdx < numOfTestItem; testCaseIdx++) {
        printf("g_reductionCmdTable[%d] = %s test %d time\n", testCaseIdx, g_reductionCmdTable[testCaseIdx].cmd,
            testCount);
        testCount = g_reductionCmdTable[testCaseIdx].testCount;
        g_reductionCmdTable[testCaseIdx].func(testCount);
    }
    return 0;
}