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
    {INDEX3, {"UNDEFINED", KuDNN::Element::TypeT::UNDEFINED}},
};

static std::map<int, std::pair<std::string, KuDNN::Layout>> g_layoutMap1D = {
    {INDEX0, {"A", KuDNN::Layout::A}},
    {INDEX1, {"UNDEFINED", KuDNN::Layout::UNDEFINED}},
};

static std::map<int, std::pair<std::string, KuDNN::Layout>> g_layoutMap2D = {
    {INDEX0, {"AB", KuDNN::Layout::AB}},
    {INDEX1, {"BA", KuDNN::Layout::BA}},
    {INDEX2, {"UNDEFINED", KuDNN::Layout::UNDEFINED}},
};

static std::map<int, std::pair<std::string, KuDNN::Layout>> g_layoutMap3D = {
    {INDEX0, {"ABC", KuDNN::Layout::ABC}},
    {INDEX1, {"ACB", KuDNN::Layout::ACB}},
    {INDEX2, {"BAC", KuDNN::Layout::BAC}},
    {INDEX3, {"BCA", KuDNN::Layout::BCA}},
    {INDEX4, {"CAB", KuDNN::Layout::CAB}},
    {INDEX5, {"CBA", KuDNN::Layout::CBA}},
    {INDEX6, {"UNDEFINED", KuDNN::Layout::UNDEFINED}},
};

static std::map<int, std::pair<std::string, KuDNN::Layout>> g_layoutMap4D = {
    {INDEX0, {"ABCD", KuDNN::Layout::ABCD}},
    {INDEX1, {"ABDC", KuDNN::Layout::ABDC}},
    {INDEX2, {"ACBD", KuDNN::Layout::ACBD}},
    {INDEX3, {"ACDB", KuDNN::Layout::ACDB}},
    {INDEX4, {"ADBC", KuDNN::Layout::ADBC}},
    {INDEX5, {"ADCB", KuDNN::Layout::ADCB}},
    {INDEX6, {"BACD", KuDNN::Layout::BACD}},
    {INDEX7, {"BCDA", KuDNN::Layout::BCDA}},
    {INDEX9, {"CDAB", KuDNN::Layout::CDAB}},
    {INDEX10, {"CDBA", KuDNN::Layout::CDBA}},
    {INDEX11, {"DCAB", KuDNN::Layout::DCAB}},
    {INDEX12, {"UNDEFINED", KuDNN::Layout::UNDEFINED}},
};

static std::map<int, std::pair<std::string, KuDNN::Layout>> g_layoutMap5D = {
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
    {INDEX11, {"UNDEFINED", KuDNN::Layout::UNDEFINED}},
};

static constexpr int g_dimRangeInitVal = 3;
static constexpr int g_dimRangeLowerBound = 1;
static constexpr int g_dimRangeUpperBound = 20;

static void DtFuzzTensor1D(int counts)
{
    char funcName[] = "DtFuzzTensor1D";
    DT_FUZZ_START(0, counts, funcName, 0)
    {
        int eleIndex = 0;
        int layoutIndex = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], 0, 0, g_layoutMap1D.size() - 1);
        int elemTypeIndex = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], 0, 0, g_precisionMap.size() - 1);
        size_t dim0 = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], g_dimRangeInitVal, g_dimRangeLowerBound,
            g_dimRangeUpperBound);
        try {
            KuDNN::Element::Type elemType(g_precisionMap[elemTypeIndex].second);
            KuDNN::Layout layout = g_layoutMap1D[layoutIndex].second;
            printf("1D Tensor: elemtype: %s, layout: %s ", g_precisionMap[elemTypeIndex].first.c_str(),
                g_layoutMap1D[layoutIndex].first.c_str());
            const KuDNN::TensorInfo srcTensor = KuDNN::TensorInfo({dim0}, elemType, layout);
            KuDNN::SizeType axisNum = srcTensor.GetAxisNumAccordingToLayout(0);
            KUDNN_UNUSED(axisNum);
        } catch (const KuDNN::Service::Unsupported& e) {
            printf("Unsupported Configuration");
        } catch (const KuDNN::Service::LogicError& e) {
            printf("Logic Error");
        }
        printf("\n");
    }
    DT_FUZZ_END()
}

static void DtFuzzTensor2D(int counts)
{
    char funcName[] = "DtFuzzTensor2D";
    DT_FUZZ_START(0, counts, funcName, 0)
    {
        int eleIndex = 0;
        int layoutIndex = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], 0, 0, g_layoutMap2D.size() - 1);
        int elemTypeIndex = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], 0, 0, g_precisionMap.size() - 1);
        size_t dim0 = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], g_dimRangeInitVal, g_dimRangeLowerBound,
            g_dimRangeUpperBound);
        try {
            KuDNN::Element::Type elemType(g_precisionMap[elemTypeIndex].second);
            KuDNN::Layout layout = g_layoutMap2D[layoutIndex].second;
            printf("2D Tensor: elemtype: %s, layout: %s ", g_precisionMap[elemTypeIndex].first.c_str(),
            g_layoutMap2D[layoutIndex].first.c_str());
            const KuDNN::TensorInfo srcTensor = KuDNN::TensorInfo({dim0, dim0}, elemType, layout);
            KuDNN::SizeType axisNum = srcTensor.GetAxisNumAccordingToLayout(1);
            KUDNN_UNUSED(axisNum);
        } catch (const KuDNN::Service::Unsupported& e) {
            printf("Unsupported Configuration");
        } catch (const KuDNN::Service::LogicError& e) {
            printf("Logic Error");
        }
        printf("\n");
    }
    DT_FUZZ_END()
}

static void DtFuzzTensor3D(int counts)
{
    char funcName[] = "DtFuzzTensor3D";
    DT_FUZZ_START(0, counts, funcName, 0)
    {
        int eleIndex = 0;
        int layoutIndex = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], 0, 0, g_layoutMap3D.size() - 1);
        int elemTypeIndex = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], 0, 0, g_precisionMap.size() - 1);
        size_t dim0 = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], g_dimRangeInitVal, g_dimRangeLowerBound,
            g_dimRangeUpperBound);
        try {
            KuDNN::Layout layout = g_layoutMap3D[layoutIndex].second;
            KuDNN::Element::Type elemType(g_precisionMap[elemTypeIndex].second);
            printf("3D Tensor: elemtype: %s, layout: %s", g_precisionMap[elemTypeIndex].first.c_str(),
                g_layoutMap3D[layoutIndex].first.c_str());
            const KuDNN::TensorInfo srcTensor = KuDNN::TensorInfo({dim0, dim0, dim0}, elemType, layout);
            KuDNN::SizeType axisNum = srcTensor.GetAxisNumAccordingToLayout(2);
            KUDNN_UNUSED(axisNum);
        } catch (const KuDNN::Service::Unsupported& e) {
            printf("Unsupported Configuration");
        } catch (const KuDNN::Service::LogicError& e) {
            printf("Logic Error");
        }
        printf("\n");
    }
    DT_FUZZ_END()
}

static void DtFuzzTensor4D(int counts)
{
    char funcName[] = "DtFuzzTensor4D";
    DT_FUZZ_START(0, counts, funcName, 0)
    {
        int eleIndex = 0;
        int layoutIndex = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], 0, 0, g_layoutMap4D.size() - 1);
        int elemTypeIndex = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], 0, 0, g_precisionMap.size() - 1);
        size_t dim0 = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], g_dimRangeInitVal, g_dimRangeLowerBound,
            g_dimRangeUpperBound);
        try {
            KuDNN::Element::Type elemType(g_precisionMap[elemTypeIndex].second);
            KuDNN::Layout layout = g_layoutMap4D[layoutIndex].second;
            printf("4D Tensor: elemtype: %s, layout: %s ", g_precisionMap[elemTypeIndex].first.c_str(),
            g_layoutMap4D[layoutIndex].first.c_str());
            const KuDNN::TensorInfo srcTensor = KuDNN::TensorInfo({dim0, dim0, dim0, dim0}, elemType, layout);
            KuDNN::SizeType axisNum = srcTensor.GetAxisNumAccordingToLayout(3);
            KUDNN_UNUSED(axisNum);
        } catch (const KuDNN::Service::Unsupported& e) {
            printf("Unsupported Configuration");
        } catch (const KuDNN::Service::LogicError& e) {
            printf("Logic Error");
        }
        printf("\n");
    }
    DT_FUZZ_END()
}

static void DtFuzzTensor5D(int counts)
{
    char funcName[] = "DtFuzzTensor5D";
    DT_FUZZ_START(0, counts, funcName, 0)
    {
        int eleIndex = 0;
        int layoutIndex = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], 0, 0, g_layoutMap5D.size() - 1);
        int elemTypeIndex = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], 0, 0, g_precisionMap.size() - 1);
        size_t dim0 = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], g_dimRangeInitVal, g_dimRangeLowerBound,
            g_dimRangeUpperBound);
        try {
            KuDNN::Element::Type elemType(g_precisionMap[elemTypeIndex].second);
            KuDNN::Layout layout = g_layoutMap5D[layoutIndex].second;
            printf("5D Tensor: elemtype: %s, layout: %s ", g_precisionMap[elemTypeIndex].first.c_str(),
                g_layoutMap5D[layoutIndex].first.c_str());
            const KuDNN::TensorInfo srcTensor = KuDNN::TensorInfo({dim0, dim0, dim0, dim0, dim0}, elemType, layout);
            KuDNN::TensorInfo dstTensor = srcTensor;
            KuDNN::SizeType axisNum = srcTensor.GetAxisNumAccordingToLayout(4);
            KUDNN_UNUSED(axisNum);
            KUDNN_UNUSED(dstTensor);
        } catch (const KuDNN::Service::Unsupported& e) {
            printf("Unsupported Configuration");
        } catch (const KuDNN::Service::LogicError& e) {
            printf("Logic Error");
        }
        printf("\n");
    }
    DT_FUZZ_END()
}

static constexpr int g_tensorFuzzTestCount = 1000;

static Cmdinfo g_tensorCmdTable[] = {
    {"DT_Tensor_1D", DtFuzzTensor1D, g_tensorFuzzTestCount},
    {"DT_Tensor_2D", DtFuzzTensor2D, g_tensorFuzzTestCount},
    {"DT_Tensor_3D", DtFuzzTensor3D, g_tensorFuzzTestCount},
    {"DT_Tensor_4D", DtFuzzTensor4D, g_tensorFuzzTestCount},
    {"DT_Tensor_5D", DtFuzzTensor5D, g_tensorFuzzTestCount},
};

int main(int argc, char *argv[])
{
    DT_Enable_Leak_Check(0, 0);
    if (argc > 1) {
        g_tensorCmdTable[0] = {"DT_Tensor_1D", DtFuzzTensor1D, atoi(argv[1])};
        g_tensorCmdTable[1] = {"DT_Tensor_2D", DtFuzzTensor2D, atoi(argv[1])};
        g_tensorCmdTable[2] = {"DT_Tensor_3D", DtFuzzTensor3D, atoi(argv[1])};
        g_tensorCmdTable[3] = {"DT_Tensor_4D", DtFuzzTensor4D, atoi(argv[1])};
        g_tensorCmdTable[4] = {"DT_Tensor_5D", DtFuzzTensor5D, atoi(argv[1])};
    }

    int numOfTestItem = sizeof(g_tensorCmdTable) / sizeof(g_tensorCmdTable[0]);
    int testCount = 0;

    for (int testCaseIdx = 0; testCaseIdx < numOfTestItem; testCaseIdx++) {
        printf("g_tensorCmdTable[%d] = %s test %d time\n", testCaseIdx, g_tensorCmdTable[testCaseIdx].cmd, testCount);

        testCount = g_tensorCmdTable[testCaseIdx].testCount;
        g_tensorCmdTable[testCaseIdx].func(testCount);
    }
    return 0;
}