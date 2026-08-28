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

static std::map<int, std::pair<std::string, KuDNN::BinaryFunction>> g_binaryFuncMap = {
    {INDEX0, {"ADD", KuDNN::BinaryFunction::ADD}},
    {INDEX1, {"DIV", KuDNN::BinaryFunction::DIV}},
    {INDEX2, {"MAX", KuDNN::BinaryFunction::MAX}},
    {INDEX3, {"MIN", KuDNN::BinaryFunction::MIN}},
    {INDEX4, {"MUL", KuDNN::BinaryFunction::MUL}},
    {INDEX5, {"SUB", KuDNN::BinaryFunction::SUB}},
    {INDEX6, {"GE", KuDNN::BinaryFunction::GE}},
    {INDEX7, {"GT", KuDNN::BinaryFunction::GT}},
    {INDEX8, {"LE", KuDNN::BinaryFunction::LE}},
    {INDEX9, {"LT", KuDNN::BinaryFunction::LT}},
    {INDEX10, {"EQ", KuDNN::BinaryFunction::EQ}},
    {INDEX11, {"NE", KuDNN::BinaryFunction::NE}},
};

static std::map<int, std::pair<std::string, KuDNN::Element::TypeT>> g_precisionMap = {
    {INDEX0, {"F32", KuDNN::Element::TypeT::F32}},
    {INDEX1, {"F16", KuDNN::Element::TypeT::F16}},
    {INDEX2, {"BF16", KuDNN::Element::TypeT::BF16}},
};

static constexpr int g_dimRangeInitVal = 4;
static constexpr int g_dimRangeLowerBound = 1;
static constexpr int g_dimRangeUpperBound = 30;

static constexpr float g_scaleRangeInitVal = 1.0f;
static constexpr float g_scaleRangeLowerBound = -10.0f;
static constexpr float g_scaleRangeUpperBound = 10.0f;

static void DtFuzzBinaryOperation(const std::array<size_t, 5> dimSrc0, const std::array<size_t, 5> dimSrc1,
    const std::array<size_t, 5> dimDst, int src0ElemTypeIndex, int src1ElemTypeIndex, int dstElemTypeIndex,
    size_t dim0, size_t dim1, size_t dim2, size_t dim3, size_t dim4, int opIndex, float scale0, float scale1) {
    KuDNN::Element::Type src0Type(g_precisionMap[src0ElemTypeIndex].second);
    KuDNN::Element::Type src1Type(g_precisionMap[src1ElemTypeIndex].second);
    KuDNN::Element::Type dstType(g_precisionMap[dstElemTypeIndex].second);
    size_t src0ElemSz = src0Type.GetSize();
    size_t src1ElemSz = src1Type.GetSize();
    size_t dstElemSz = dstType.GetSize();

    const KuDNN::TensorInfo src0TensorInfo = {{dimSrc0[0], dimSrc0[1], dimSrc0[2], dimSrc0[3], dimSrc0[4]},
        src0Type, KuDNN::Layout::ABCDE};
    const KuDNN::TensorInfo src1TensorInfo = {{dimSrc1[0], dimSrc1[1], dimSrc1[2], dimSrc1[3], dimSrc1[4]},
        src1Type, KuDNN::Layout::ABCDE};
    const KuDNN::TensorInfo dstTensorInfo = {{dimDst[0], dimDst[1], dimDst[2], dimDst[3], dimDst[4]},
        dstType, KuDNN::Layout::ACDEB};
    KuDNN::BinaryFunction kind = g_binaryFuncMap[opIndex].second;
    char format[] = "Binary op: %s, Precision(src0:src1:dst): %s:%s:%s, Shape: %zux%zux%zux%zux%zu, Scale:%f:%f \n";
    printf(format, g_binaryFuncMap[opIndex].first.c_str(), g_precisionMap[src0ElemTypeIndex].first.c_str(),
        g_precisionMap[src1ElemTypeIndex].first.c_str(), g_precisionMap[dstElemTypeIndex].first.c_str(),
        dim0, dim1, dim2, dim3, dim4, scale0, scale1);
    void *src0 = NULL;
    void *src1 = NULL;
    void *dst = NULL;
    try {
        KuDNN::BinaryLayer binary1(src0TensorInfo, src1TensorInfo, dstTensorInfo, kind);
        KuDNN::BinaryLayer binary2(binary1);
        binary2 = binary1;
        KuDNN::BinaryLayer binary3(binary1);
        binary3 = std::move(binary2);
        KuDNN::BinaryLayer binary4(std::move(binary2));

        KuDNN::BinaryLayer binaryLayer(src0TensorInfo, src1TensorInfo, dstTensorInfo, kind);
        src0 = malloc(std::max((size_t)1, dim0) * std::max((size_t)1, dim1) * std::max((size_t)1, dim2) *
            std::max((size_t)1, dim3) * std::max((size_t)1, dim4) * src0ElemSz);
        if (src0 == NULL) {
            std::cerr<<"memory allocation failed"<<std::endl;
            return;
        }
        src1 = malloc(std::max((size_t)1, dim0) * std::max((size_t)1, dim1) * std::max((size_t)1, dim2) *
            std::max((size_t)1, dim3) * std::max((size_t)1, dim4) * src1ElemSz);
        if (src1 == NULL) {
            free(src0);
            std::cerr<<"memory allocation failed"<<std::endl;
            return;
        }
        dst = malloc(std::max((size_t)1, dim0) * std::max((size_t)1, dim1) * std::max((size_t)1, dim2) *
            std::max((size_t)1, dim3) * std::max((size_t)1, dim4) * dstElemSz);
        if (dst == NULL) {
            free(src0);
            free(src1);
            std::cerr<<"memory allocation failed"<<std::endl;
            return;
        }
        binaryLayer.Run(src0, src1, dst, scale0, scale1);
        free(src0);
        free(src1);
        free(dst);
    } catch (const KuDNN::Service::Unsupported& e) {
        if (src0 != NULL) {
            free(src0);
        }
        if (src1 != NULL) {
            free(src1);
        }
        if (dst != NULL) {
            free(dst);
        }
        printf("Unsupported Configuration");
    } catch (const KuDNN::Service::LogicError& e) {
        if (src0 != NULL) {
            free(src0);
        }
        if (src1 != NULL) {
            free(src1);
        }
        if (dst != NULL) {
            free(dst);
        }
        printf("Bad shapes' arguments");
    }
}

static void DtFuzzBinary(int counts)
{
    char funcName[] = "DtFuzzBinary";
    DT_FUZZ_START(0, counts, funcName, 0)
    {
        int eleIndex = 0;
        int src0ElemTypeIndex = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], 0, 0, g_precisionMap.size() - 1);
        int src1ElemTypeIndex = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], 0, 0, g_precisionMap.size() - 1);
        int dstElemTypeIndex = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], 0, 0, g_precisionMap.size() - 1);
        int opIndex = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], 0, 0, g_binaryFuncMap.size() - 1);
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
        float scale0 = *(float *)DT_SetGetNumberRangeFloatV3(eleIndex++, 1,
            0, 10);
        float scale1 = *(float *)DT_SetGetNumberRangeFloatV3(eleIndex++, g_scaleRangeInitVal,
            g_scaleRangeLowerBound, g_scaleRangeUpperBound);

        const std::array<size_t, 5> dimSrc0 = {dim0, dim1, dim2, dim3, dim4};
        const std::array<size_t, 5> dimSrc1 = {dim0, dim1, dim2, dim3, dim4};
        const std::array<size_t, 5> dimDst = {dim0, dim1, dim2, dim3, dim4};

        DtFuzzBinaryOperation(dimSrc0, dimSrc1, dimDst, src0ElemTypeIndex, src1ElemTypeIndex, dstElemTypeIndex,
            dim0, dim1, dim2, dim3, dim4, opIndex, scale0, scale1);
        DtFuzzBinaryOperation(dimSrc0, {1, 1, 1, 1, 1}, dimDst, src0ElemTypeIndex, src1ElemTypeIndex,
            dstElemTypeIndex, dim0, dim1, dim2, dim3, dim4, opIndex, scale0, scale1);
        DtFuzzBinaryOperation(dimSrc0, {1, dim1, 1, 1, 1}, dimDst, src0ElemTypeIndex, src1ElemTypeIndex,
            dstElemTypeIndex, dim0, dim1, dim2, dim3, dim4, opIndex, scale0, scale1);
        DtFuzzBinaryOperation(dimSrc0, {1, 1, 1, 1, dim4}, dimDst, src0ElemTypeIndex, src1ElemTypeIndex,
            dstElemTypeIndex, dim0, dim1, dim2, dim3, dim4, opIndex, scale0, scale1);

        printf("\n");
    }
    DT_FUZZ_END()
}

static void DtFuzzBinaryConstr(int counts)
{
    char funcName[] = "DtFuzzConstr";
    DT_FUZZ_START(0, counts, funcName, 0)
    {
        int eleIndex = 0;
        int src0ElemTypeIndex = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], 0, 0, g_precisionMap.size() - 1);
        int src1ElemTypeIndex = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], 0, 0, g_precisionMap.size() - 1);
        int dstElemTypeIndex = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], 0, 0, g_precisionMap.size() - 1);
        int opIndex = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], 0, 0, g_binaryFuncMap.size() - 1);
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
        KuDNN::Element::Type src1Type(g_precisionMap[src1ElemTypeIndex].second);
        KuDNN::Element::Type dstType(g_precisionMap[dstElemTypeIndex].second);

        const KuDNN::TensorInfo src0TensorInfo = {{dim0, dim1, dim2, dim3, dim4}, src0Type, KuDNN::Layout::ABCDE};
        const KuDNN::TensorInfo src1TensorInfo = {{dim0, dim1, dim2, dim3, dim4}, src1Type, KuDNN::Layout::ABCDE};
        const KuDNN::TensorInfo dstTensorInfo = {{dim0, dim1, dim2, dim3, dim4}, dstType, KuDNN::Layout::ABCDE};
        KuDNN::BinaryFunction kind = g_binaryFuncMap[opIndex].second;
        char format[] = "Binary op: %s, Precision(src0:src1:dst): %s:%s:%s, Shape: %zux%zux%zux%zux%zu ";
        printf(format, g_binaryFuncMap[opIndex].first.c_str(), g_precisionMap[src0ElemTypeIndex].first.c_str(),
            g_precisionMap[src1ElemTypeIndex].first.c_str(), g_precisionMap[dstElemTypeIndex].first.c_str(),
            dim0, dim1, dim2, dim3, dim4);
        try {
            KuDNN::BinaryLayer binaryLayer0(src0TensorInfo, src1TensorInfo, dstTensorInfo, kind);
            KuDNN::BinaryLayer binaryLayer1(binaryLayer0);
            KuDNN::BinaryLayer binaryLayer2 = binaryLayer0;
            KuDNN::BinaryLayer binaryLayer3(src0TensorInfo, src1TensorInfo, dstTensorInfo, kind);
            KuDNN::BinaryLayer binaryLayer4(src0TensorInfo, src1TensorInfo, dstTensorInfo, kind);
            KuDNN::BinaryLayer binaryLayer5 = std::move(binaryLayer3);
            KuDNN::BinaryLayer binaryLayer6(std::move(binaryLayer4));
        } catch (const KuDNN::Service::Unsupported& e) {
            printf("Unsupported");
        } catch (const KuDNN::Service::LogicError& e) {
            printf("LogicError");
        }
        printf("\n");
    }
    DT_FUZZ_END()
}

static constexpr int g_binaryFuzzTestCount = 25000;

static Cmdinfo g_binaryCmdTable[] = {
    {"BinaryConstructor", DtFuzzBinaryConstr, 200},
    {"Binary", DtFuzzBinary, g_binaryFuzzTestCount},
};

int main(int argc, char *argv[])
{
    DT_Enable_Leak_Check(0, 0);
    if (argc > 1) {
        g_binaryCmdTable[0] = {"BinaryConstructor", DtFuzzBinaryConstr, atoi(argv[1])};
        g_binaryCmdTable[1] = {"Binary", DtFuzzBinary, atoi(argv[1])};
    }

    int numOfTestItem = sizeof(g_binaryCmdTable) / sizeof(g_binaryCmdTable[0]);
    int testCount = 0;

    for (int testCaseIdx = 0; testCaseIdx < numOfTestItem; testCaseIdx++) {
        printf("g_binaryCmdTable[%d] = %s test %d time\n", testCaseIdx, g_binaryCmdTable[testCaseIdx].cmd, testCount);

        testCount = g_binaryCmdTable[testCaseIdx].testCount;
        g_binaryCmdTable[testCaseIdx].func(testCount);
    }
    return 0;
}