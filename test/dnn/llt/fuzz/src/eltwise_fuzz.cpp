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

static std::map<int, std::pair<std::string, KuDNN::ActivationFunction>> g_activationFuncMap = {
    {INDEX0, {"ABS", KuDNN::ActivationFunction::ABS}},
    {INDEX1, {"CLIP", KuDNN::ActivationFunction::CLIP}},
    {INDEX2, {"CLIP_V2", KuDNN::ActivationFunction::CLIP_V2}},
    {INDEX3, {"ELU", KuDNN::ActivationFunction::ELU}},
    {INDEX4, {"EXP", KuDNN::ActivationFunction::EXP}},
    {INDEX5, {"GELU_ERF", KuDNN::ActivationFunction::GELU_ERF}},
    {INDEX6, {"GELU_TANH", KuDNN::ActivationFunction::GELU_TANH}},
    {INDEX7, {"HARDSIGMOID", KuDNN::ActivationFunction::HARDSIGMOID}},
    {INDEX8, {"HARDSWISH", KuDNN::ActivationFunction::HARDSWISH}},
    {INDEX9, {"LOG", KuDNN::ActivationFunction::LOG}},
    {INDEX10, {"SIGMOID", KuDNN::ActivationFunction::SIGMOID}},
    {INDEX11, {"MISH", KuDNN::ActivationFunction::MISH}},
    {INDEX12, {"POW", KuDNN::ActivationFunction::POW}},
    {INDEX13, {"ROUND", KuDNN::ActivationFunction::ROUND}},
    {INDEX14, {"SRELU", KuDNN::ActivationFunction::SRELU}},
    {INDEX15, {"SQRT", KuDNN::ActivationFunction::SQRT}},
    {INDEX16, {"SQUARE", KuDNN::ActivationFunction::SQUARE}},
    {INDEX17, {"SWISH", KuDNN::ActivationFunction::SWISH}},
    {INDEX18, {"TANH", KuDNN::ActivationFunction::TANH}},
    // int type
    {INDEX19, {"LINEAR", KuDNN::ActivationFunction::LINEAR}},
    {INDEX20, {"RELU", KuDNN::ActivationFunction::RELU}},
    // bwd_dst
    {INDEX21, {"ELU_DST", KuDNN::ActivationFunction::ELU_DST}},
    {INDEX22, {"EXP_DST", KuDNN::ActivationFunction::EXP_DST}},
    {INDEX23, {"CLIP_V2_DST", KuDNN::ActivationFunction::CLIP_V2_DST}},
    {INDEX24, {"LOGISTIC_DST", KuDNN::ActivationFunction::LOGISTIC_DST}},
    {INDEX25, {"RELU_DST", KuDNN::ActivationFunction::RELU_DST}},
    {INDEX26, {"SQRT_DST", KuDNN::ActivationFunction::SQRT_DST}},
    {INDEX27, {"TANH_DST", KuDNN::ActivationFunction::TANH_DST}},
};

static std::map<int, std::pair<std::string, KuDNN::Element::TypeT>> g_precisionMap = {
    {INDEX0, {"F32", KuDNN::Element::TypeT::F32}},
    {INDEX1, {"F16", KuDNN::Element::TypeT::F16}},
    {INDEX2, {"BF16", KuDNN::Element::TypeT::BF16}},
    {INDEX3, {"S32", KuDNN::Element::TypeT::S32}},
    {INDEX4, {"U8", KuDNN::Element::TypeT::U8}},
    {INDEX5, {"S8", KuDNN::Element::TypeT::S8}},
};

static std::map<int, std::pair<std::string, float>> g_betaValueMap = {
    {INDEX0, {"0", 0}},
    {INDEX1, {"1", 1}},
    {INDEX2, {"2", 2}},
    {INDEX3, {"3", 3}},
    {INDEX4, {"4", 4}},
    {INDEX5, {"5", 5}},
    {INDEX6, {"6", 6}},
    {INDEX7, {"7", 7}},
    {INDEX8, {"8", 8}},
    {INDEX9, {"9", 9}},
    {INDEX10, {"10", 10}},
    {INDEX11, {"1.5", 0x1.8p0}},
    {INDEX12, {"0.5", 0x1p-1}},
};

static std::map<int, std::pair<std::string, bool>> g_diffDTypeMap = {
    {INDEX0, {"SAME_DTYPE_INDEX", false}},
    {INDEX1, {"DIFFERENT_DTYPE_INDEX", true}},
};

static constexpr int g_dimRangeInitVal = 1;
static constexpr int g_dimRangeLowerBound = 1;
static constexpr int g_dimRangeUpperBound = 30;

static void DtFuzzEltwiseForward(int counts)
{
    char funcName[] = "DtFuzzEltwiseForward";
    DT_FUZZ_START(0, counts, funcName, 0)
    {
        int eleIndex = 0;
        int opIndex = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], 0, 0, g_activationFuncMap.size() - 1);
        int srcElemTypeIndex = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], 0, 0, g_precisionMap.size() - 1);

        int betaValueIndex = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], 0, 0, g_betaValueMap.size() - 1);
        int diffDTypeIndex = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], 0, 0, g_diffDTypeMap.size() - 1);
        size_t dim0 = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], g_dimRangeInitVal, g_dimRangeLowerBound,
            g_dimRangeUpperBound);
        size_t dim1 = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], g_dimRangeInitVal, g_dimRangeLowerBound,
            g_dimRangeUpperBound);
        size_t dim2 = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], g_dimRangeInitVal, g_dimRangeLowerBound,
            g_dimRangeUpperBound);
        size_t dim3 = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], g_dimRangeInitVal, g_dimRangeLowerBound,
            g_dimRangeUpperBound);

        KuDNN::Element::Type srcType(g_precisionMap[srcElemTypeIndex].second);
        KuDNN::Element::Type dstType = diffDTypeIndex ? KuDNN::Element::Type(KuDNN::Element::TypeT::F32) : srcType;
        size_t srcElemSz = srcType.GetSize();
        size_t dstElemSz = dstType.GetSize();

        const KuDNN::TensorInfo srcTensorInfo = {{dim0, dim1, dim2, dim3}, srcType, KuDNN::Layout::ABCD};
        const KuDNN::TensorInfo dstTensorInfo = {{dim0, dim1, dim2, dim3}, dstType, KuDNN::Layout::ABCD};
        KuDNN::ActivationFunction kind = g_activationFuncMap[opIndex].second;
        char format[] = "Forward Eltwise op: %s, Precision(src:dst): %s:%s, Shape: %zux%zux%zux%zu ";
        printf(format, g_activationFuncMap[opIndex].first.c_str(), g_precisionMap[srcElemTypeIndex].first.c_str(),
            g_precisionMap[srcElemTypeIndex].first.c_str(), dim0, dim1, dim2, dim3);
        void *src = NULL;
        void *dst = NULL;
        try {
            KuDNN::ActivationLayerFWD activate1(srcTensorInfo, dstTensorInfo, kind, 2.f, g_betaValueMap[betaValueIndex].second);
            KuDNN::ActivationLayerFWD activate2(activate1);
            activate2 = activate1;
            KuDNN::ActivationLayerFWD activate3(activate1);
            activate3 = std::move(activate1);
            KuDNN::ActivationLayerFWD activationLayer(std::move(activate2));

            src = malloc(std::max((size_t)1, dim0) * std::max((size_t)1, dim1) * std::max((size_t)1, dim2) *
                std::max((size_t)1, dim3) * srcElemSz);
            if (src == NULL) {
                std::cerr<<"memory allocation failed"<<std::endl;
                return;
            }
            dst = malloc(std::max((size_t)1, dim0) * std::max((size_t)1, dim1) * std::max((size_t)1, dim2) *
                std::max((size_t)1, dim3) * dstElemSz);
            if (dst == NULL) {
                free(src);
                std::cerr<<"memory allocation failed"<<std::endl;
                return;
            }
            activationLayer.Run(src, dst);
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

static void DtFuzzEltwiseBackward(int counts)
{
    char funcName[] = "DtFuzzEltwiseBackward";
    DT_FUZZ_START(0, counts, funcName, 0)
    {
        int eleIndex = 0;
        int srcElemTypeIndex = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], 0, 0, g_precisionMap.size() - 1);
        int opIndex = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], 0, 0, g_activationFuncMap.size() - 1);
        int betaValueIndex = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], 0, 0, g_betaValueMap.size() - 1);
        size_t dim0 = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], g_dimRangeInitVal, g_dimRangeLowerBound,
            g_dimRangeUpperBound);
        size_t dim1 = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], g_dimRangeInitVal, g_dimRangeLowerBound,
            g_dimRangeUpperBound);
        size_t dim2 = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], g_dimRangeInitVal, g_dimRangeLowerBound,
            g_dimRangeUpperBound);
        size_t dim3 = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], g_dimRangeInitVal, g_dimRangeLowerBound,
            g_dimRangeUpperBound);

        KuDNN::Element::Type srcType(g_precisionMap[srcElemTypeIndex].second);
        KuDNN::Element::Type ddType = srcType;
        KuDNN::Element::Type dsType = srcType;
        size_t srcElemSz = srcType.GetSize();
        size_t ddElemSz = ddType.GetSize();
        size_t dsElemSz = dsType.GetSize();

        const KuDNN::TensorInfo srcTensorInfo = {{dim0, dim1, dim2, dim3}, srcType, KuDNN::Layout::ABCD};
        const KuDNN::TensorInfo ddTensorInfo = {{dim0, dim1, dim2, dim3}, ddType, KuDNN::Layout::ABCD};
        const KuDNN::TensorInfo dsTensorInfo = {{dim0, dim1, dim2, dim3}, dsType, KuDNN::Layout::ABCD};
        KuDNN::ActivationFunction kind = g_activationFuncMap[opIndex].second;
        char format[] = "Backward Eltwise op: %s, Precision(src:diff_src:diff_dst): %s:%s:%s, Shape: %zux%zux%zux%zu ";
        printf(format, g_activationFuncMap[opIndex].first.c_str(), g_precisionMap[srcElemTypeIndex].first.c_str(),
            g_precisionMap[srcElemTypeIndex].first.c_str(),  g_precisionMap[srcElemTypeIndex].first.c_str(),
            dim0, dim1, dim2, dim3);
        void *src = NULL;
        void *dd = NULL;
        void *ds = NULL;
        try {
            KuDNN::ActivationLayerBWD activation1(dsTensorInfo, ddTensorInfo, srcTensorInfo, kind);
            KuDNN::ActivationLayerBWD activation2(activation1);
            activation2 = activation1;
            KuDNN::ActivationLayerBWD activation3(activation1);
            activation3 = std::move(activation1);
            KuDNN::ActivationLayerBWD activation5(std::move(activation2));

            KuDNN::ActivationLayerBWD activationLayer(dsTensorInfo, ddTensorInfo, srcTensorInfo, kind,
                1, g_betaValueMap[betaValueIndex].second);
            src = malloc(std::max((size_t)1, dim0) * std::max((size_t)1, dim1) * std::max((size_t)1, dim2) *
                std::max((size_t)1, dim3) * srcElemSz);
            if (src == NULL) {
                std::cerr<<"memory allocation failed"<<std::endl;
                return;
            }
            dd = malloc(std::max((size_t)1, dim0) * std::max((size_t)1, dim1) * std::max((size_t)1, dim2) *
                std::max((size_t)1, dim3) * ddElemSz);
            if (dd == NULL) {
                free(src);
                std::cerr<<"memory allocation failed"<<std::endl;
                return;
            }
            ds = malloc(std::max((size_t)1, dim0) * std::max((size_t)1, dim1) * std::max((size_t)1, dim2) *
                std::max((size_t)1, dim3) * dsElemSz);
            if (ds == NULL) {
                free(src);
                free(dd);
                std::cerr<<"memory allocation failed"<<std::endl;
                return;
            }
            activationLayer.Run(ds, dd, src);
            free(src);
            free(dd);
            free(ds);
        } catch (const KuDNN::Service::Unsupported& e) {
            if (src != NULL) {
                free(src);
            }
            if (dd != NULL) {
                free(dd);
            }
            if (ds != NULL) {
                free(ds);
            }
            printf("Unsupported Configuration");
        } catch (const KuDNN::Service::LogicError& e) {
            if (src != NULL) {
                free(src);
            }
            if (dd != NULL) {
                free(dd);
            }
            if (ds != NULL) {
                free(ds);
            }
            printf("Bad shapes' arguments");
        }
        printf("\n");
    }
    DT_FUZZ_END()
}

static void DtFuzzEltwiseBackwardRLL(int counts)
{
    char funcName[] = "DtFuzzEltwiseBackward: Relu/Linear/Logistic";
    DT_FUZZ_START(0, counts, funcName, 0)
    {
        int eleIndex = 0;
        int srcElemTypeIndex = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], 0, 0, g_precisionMap.size() - 1);
        int opIndex = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], INDEX7, INDEX7,
            g_activationFuncMap.size() - 1);
        size_t dim0 = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], g_dimRangeInitVal, g_dimRangeLowerBound,
            g_dimRangeUpperBound);
        size_t dim1 = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], g_dimRangeInitVal, g_dimRangeLowerBound,
            g_dimRangeUpperBound);
        size_t dim2 = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], g_dimRangeInitVal, g_dimRangeLowerBound,
            g_dimRangeUpperBound);
        size_t dim3 = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], g_dimRangeInitVal, g_dimRangeLowerBound,
            g_dimRangeUpperBound);

        KuDNN::Element::Type srcType(g_precisionMap[srcElemTypeIndex].second);
        KuDNN::Element::Type ddType(g_precisionMap[srcElemTypeIndex].second);
        KuDNN::Element::Type dsType(g_precisionMap[srcElemTypeIndex].second);
        size_t srcElemSz = srcType.GetSize();
        size_t ddElemSz = ddType.GetSize();
        size_t dsElemSz = dsType.GetSize();

        const KuDNN::TensorInfo srcTensorInfo = {{dim0, dim1, dim2, dim3}, srcType, KuDNN::Layout::ABCD};
        const KuDNN::TensorInfo ddTensorInfo = {{dim0, dim1, dim2, dim3}, ddType, KuDNN::Layout::ABCD};
        const KuDNN::TensorInfo dsTensorInfo = {{dim0, dim1, dim2, dim3}, dsType, KuDNN::Layout::ABCD};
        KuDNN::ActivationFunction kind = g_activationFuncMap[opIndex].second;
        char format[] = "Backward Eltwise op: %s, Precision(src:diff_src:diff_dst): %s:%s:%s, Shape: %zux%zux%zux%zu ";
        printf(format, g_activationFuncMap[opIndex].first.c_str(), g_precisionMap[srcElemTypeIndex].first.c_str(),
            g_precisionMap[srcElemTypeIndex].first.c_str(), g_precisionMap[srcElemTypeIndex].first.c_str(),
            dim0, dim1, dim2, dim3);
        void *src = NULL;
        void *dd = NULL;
        void *ds = NULL;
        try {
            KuDNN::ActivationLayerBWD activate1(dsTensorInfo, ddTensorInfo, srcTensorInfo, kind);
            KuDNN::ActivationLayerBWD activate2(activate1);
            activate2 = activate1;
            KuDNN::ActivationLayerBWD activate4(activate2);
            activate4 = std::move(activate1);
            KuDNN::ActivationLayerBWD activate5(std::move(activate2));

            KuDNN::ActivationLayerBWD activationLayer(dsTensorInfo, ddTensorInfo, srcTensorInfo, kind);
            src = malloc(std::max((size_t)1, dim0) * std::max((size_t)1, dim1) * std::max((size_t)1, dim2) *
                std::max((size_t)1, dim3) * srcElemSz);
            if (src == NULL) {
                std::cerr<<"memory allocation failed"<<std::endl;
                return;
            }
            dd = malloc(std::max((size_t)1, dim0) * std::max((size_t)1, dim1) * std::max((size_t)1, dim2) *
                std::max((size_t)1, dim3) * ddElemSz);
            if (dd == NULL) {
                free(src);
                std::cerr<<"memory allocation failed"<<std::endl;
                return;
            }
            ds = malloc(std::max((size_t)1, dim0) * std::max((size_t)1, dim1) * std::max((size_t)1, dim2) *
                std::max((size_t)1, dim3) * dsElemSz);
            if (ds == NULL) {
                free(src);
                free(dd);
                std::cerr<<"memory allocation failed"<<std::endl;
                return;
            }
            activationLayer.Run(ds, dd, src);
            free(src);
            free(dd);
            free(ds);
        } catch (const KuDNN::Service::Unsupported& e) {
            if (src != NULL) {
                free(src);
            }
            if (dd != NULL) {
                free(dd);
            }
            if (ds != NULL) {
                free(ds);
            }
            printf("Unsupported Configuration");
        } catch (const KuDNN::Service::LogicError& e) {
            if (src != NULL) {
                free(src);
            }
            if (dd != NULL) {
                free(dd);
            }
            if (ds != NULL) {
                free(ds);
            }
            printf("Bad shapes' arguments");
        }
        printf("\n");
    }
    DT_FUZZ_END()
}


static constexpr int g_eltwiseFuzzTestCount = 100000;

static Cmdinfo g_eltwiseCmdTable[] = {
    {"Forward Eltwise", DtFuzzEltwiseForward, g_eltwiseFuzzTestCount},
    {"Backward Eltwise", DtFuzzEltwiseBackward, g_eltwiseFuzzTestCount},
    {"Backward Relu/Linear/Logistic", DtFuzzEltwiseBackwardRLL, 8000},
};

int main(int argc, char *argv[])
{
    DT_Enable_Leak_Check(0, 0);
    if (argc > 1) {
        g_eltwiseCmdTable[0] = {"Forward Eltwise", DtFuzzEltwiseForward, atoi(argv[1])};
        g_eltwiseCmdTable[1] = {"Backward Eltwise", DtFuzzEltwiseBackward, atoi(argv[1])};
        g_eltwiseCmdTable[2] = {"Backward Relu/Linear/Logistic", DtFuzzEltwiseBackwardRLL, atoi(argv[1])};
    }

    int numOfTestItem = sizeof(g_eltwiseCmdTable) / sizeof(g_eltwiseCmdTable[0]);
    int testCount = 0;

    for (int testCaseIdx = 0; testCaseIdx < numOfTestItem; testCaseIdx++) {
        printf("g_eltwiseCmdTable[%d] = %s test %d time\n", testCaseIdx, g_eltwiseCmdTable[testCaseIdx].cmd, testCount);

        testCount = g_eltwiseCmdTable[testCaseIdx].testCount;
        g_eltwiseCmdTable[testCaseIdx].func(testCount);
    }
    return 0;
}
