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

static std::map<int, std::pair<std::string, KuDNN::Layout>> g_layoutMap1D = {
    {INDEX0, {"A", KuDNN::Layout::A}},
};

static std::map<int, std::pair<std::string, KuDNN::Layout>> g_layoutMap2D = {
    {INDEX0, {"AB", KuDNN::Layout::AB}},
    {INDEX1, {"BA", KuDNN::Layout::BA}},
};

static std::map<int, std::pair<std::string, KuDNN::Layout>> g_layoutMap3D = {
    {INDEX0, {"ABC", KuDNN::Layout::ABC}},
    {INDEX1, {"ACB", KuDNN::Layout::ACB}},
    {INDEX2, {"BAC", KuDNN::Layout::BAC}},
    {INDEX3, {"BCA", KuDNN::Layout::BCA}},
    {INDEX4, {"CAB", KuDNN::Layout::CAB}},
    {INDEX5, {"CBA", KuDNN::Layout::CBA}},
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
    {INDEX8, {"CDAB", KuDNN::Layout::CDAB}},
    {INDEX9, {"CDBA", KuDNN::Layout::CDBA}},
    {INDEX10, {"DCAB", KuDNN::Layout::DCAB}},
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
};

static constexpr int g_dimRangeInitVal = 5;
static constexpr int g_dimRangeLowerBound = 1;
static constexpr int g_dimRangeUpperBound = 20;

static constexpr int g_srcNum = 3;

static void DtConcatLayer(std::map<int, std::pair<std::string, KuDNN::Layout>> layoutMap)
{
    int eleIndex = 0;
    int srcTypeIndex = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], 0, 0,
        g_precisionMap.size() - 1);
    int dstTypeIndex = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], 0, 0,
        g_precisionMap.size() - 1);
    int layoutIndex = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], 0, 0,
        layoutMap.size() - 1);
    int dimNum = layoutMap[layoutIndex].first.length();
    int concatDim = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], 0, 0, dimNum - 1);

    std::vector<size_t> dstDims(dimNum);
    for (int i = 0; i < dimNum; i++) {
        if (i != concatDim) {
            dstDims[i] = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++],
                g_dimRangeInitVal, g_dimRangeLowerBound, g_dimRangeUpperBound);
        } else {
            dstDims[i] = 0;
        }
    }
    KuDNN::Element::Type srcType(g_precisionMap[srcTypeIndex].second);
    KuDNN::Element::Type dstType(g_precisionMap[dstTypeIndex].second);
    KuDNN::Layout layout(layoutMap[layoutIndex].second);
    std::vector<KuDNN::TensorInfo> srcInfoList;
    for (int j = 0; j < g_srcNum; j++) {
        std::vector<size_t> srcDims(dimNum);
        printf("srcShape[%d] ", j);
        for (int i = 0; i < dimNum; i++) {
            if (i != concatDim) {
                srcDims[i] = dstDims[i];
            } else {
                srcDims[i] = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++],
                    g_dimRangeInitVal, g_dimRangeLowerBound, g_dimRangeUpperBound);
                dstDims[i] += srcDims[i];
            }
            printf("%ld ", srcDims[i]);
        }
        printf("\n");
        KuDNN::Shape srcShape(srcDims.data(), static_cast<KuDNN::SizeType>(dimNum));
        KuDNN::TensorInfo srcInfo = {srcShape, srcType, layout};
        srcInfoList.push_back(srcInfo);
    }
    KuDNN::Shape dstShape(dstDims.data(), static_cast<KuDNN::SizeType>(dimNum));
    KuDNN::TensorInfo dstInfo = {dstShape, dstType, layout};

    char format[] = "DtSumLayer: Precision(srcInfo:dstInfo): %s:%s, Layout:%s, ConcatDim:%d\n";
    printf(format, g_precisionMap[srcTypeIndex].first.c_str(), g_precisionMap[dstTypeIndex].first.c_str(),
        layoutMap[layoutIndex].first.c_str(), concatDim);

    void **srcList = nullptr;
    void *dst = nullptr;
    try {
        KuDNN::ConcatLayer concat1(srcInfoList, concatDim, dstInfo);
        KuDNN::ConcatLayer concat2(concat1);
        concat2 = concat1;
        KuDNN::ConcatLayer concat4(std::move(concat1));
        KuDNN::ConcatLayer concat5(srcInfoList, concatDim, dstInfo);
        concat5 = std::move(concat2);
        KuDNN::ConcatLayer concatLayer(srcInfoList, concatDim, dstInfo);

        const size_t srcElemSize = srcType.GetSize();
        const size_t dstElemSize = dstType.GetSize();
        
        dst = (void *)malloc(dstInfo.GetTotalTensorSize() * dstElemSize);
        if (dst == nullptr) {
            std::cerr << "memory allocation failed" << std::endl;
            return;
        }
        srcList = (void **)malloc(g_srcNum * sizeof(void *));
        if (srcList == nullptr) {
            free(dst);
            std::cerr << "memory allocation failed" << std::endl;
            return;
        }
        int i;
        for (i = 0; i < g_srcNum; i++) {
            srcList[i] = (void *)malloc(srcInfoList[i].GetTotalTensorSize() * srcElemSize);
            if (srcList == nullptr) {
                break;
            }
        }
        if (i != g_srcNum) {
            for (int i = 0; i < g_srcNum; i++) {
                if (srcList[i] != nullptr) {
                    free(srcList[i]);
                } 
            }
            free(srcList);
            free(dst);
            std::cerr << "memory allocation failed" << std::endl;
            return;
        }

        concatLayer.Run((const void **)srcList, dst);

        for (int i = 0; i < g_srcNum; i++) {
            free(srcList[i]);
        }
        free(srcList);
        free(dst);
    } catch (const KuDNN::Service::Unsupported& e) {
        for (int i = 0; i < g_srcNum; i++) {
            free(srcList[i]);
        }
        free(srcList);
        free(dst);
        printf("Unsupported Configuration");
    } catch (const KuDNN::Service::LogicError& e) {
        for (int i = 0; i < g_srcNum; i++) {
            free(srcList[i]);
        }
        free(srcList);
        free(dst);
        printf("Logic Error");
    }
    printf("\n");
}

static void DtConcatLayer1D(int counts)
{
    char funcName[] = "DtConcatLayer1D";
    DT_FUZZ_START(0, counts, funcName, 0)
    {
        DtConcatLayer(g_layoutMap1D);
    }
    DT_FUZZ_END()
}

static void DtConcatLayer2D(int counts)
{
    char funcName[] = "DtConcatLayer2D";
    DT_FUZZ_START(0, counts, funcName, 0)
    {
        DtConcatLayer(g_layoutMap2D);
    }
    DT_FUZZ_END()
}

static void DtConcatLayer3D(int counts)
{
    char funcName[] = "DtConcatLayer3D";
    DT_FUZZ_START(0, counts, funcName, 0)
    {
        DtConcatLayer(g_layoutMap3D);
    }
    DT_FUZZ_END()
}

static void DtConcatLayer4D(int counts)
{
    char funcName[] = "DtConcatLayer4D";
    DT_FUZZ_START(0, counts, funcName, 0)
    {
        DtConcatLayer(g_layoutMap4D);
    }
    DT_FUZZ_END()
}

static void DtConcatLayer5D(int counts)
{
    char funcName[] = "DtConcatLayer5D";
    DT_FUZZ_START(0, counts, funcName, 0)
    {
        DtConcatLayer(g_layoutMap5D);
    }
    DT_FUZZ_END()
}

static constexpr int g_concatFuzzTestCount = 1000;

static Cmdinfo g_concatCmdTable[] = {
    {"DtConcatLayer1D", DtConcatLayer1D, g_concatFuzzTestCount},
    {"DtConcatLayer2D", DtConcatLayer2D, g_concatFuzzTestCount},
    {"DtConcatLayer3D", DtConcatLayer3D, g_concatFuzzTestCount},
    {"DtConcatLayer4D", DtConcatLayer4D, g_concatFuzzTestCount},
    {"DtConcatLayer5D", DtConcatLayer5D, g_concatFuzzTestCount},
};

int main(int argc, char *argv[])
{
    DT_Enable_Leak_Check(0, 0);
    if (argc > 1) {
        g_concatCmdTable[0] = {"DtConcatLayer1D", DtConcatLayer1D, atoi(argv[1])};
        g_concatCmdTable[1] = {"DtConcatLayer2D", DtConcatLayer2D, atoi(argv[1])};
        g_concatCmdTable[2] = {"DtConcatLayer3D", DtConcatLayer3D, atoi(argv[1])};
        g_concatCmdTable[3] = {"DtConcatLayer4D", DtConcatLayer4D, atoi(argv[1])};
        g_concatCmdTable[4] = {"DtConcatLayer5D", DtConcatLayer5D, atoi(argv[1])};
    }

    int numOfTestItem = sizeof(g_concatCmdTable) / sizeof(g_concatCmdTable[0]);
    for (int testCaseIdx = 0; testCaseIdx < numOfTestItem; testCaseIdx++) {
        printf("g_concatCmdTable[%d] = %s test %d time\n",
            testCaseIdx, g_concatCmdTable[testCaseIdx].cmd, g_concatCmdTable[testCaseIdx].testCount);
        g_concatCmdTable[testCaseIdx].func(g_concatCmdTable[testCaseIdx].testCount);
    }
    return 0;
}
