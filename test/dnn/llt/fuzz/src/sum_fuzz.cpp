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
using std::vector;

static std::map<int, std::pair<std::string, KuDNN::Element::TypeT>> g_precisionMap = {
    {INDEX0, {"F32", KuDNN::Element::TypeT::F32}},
    {INDEX1, {"F16", KuDNN::Element::TypeT::F16}},
    {INDEX2, {"BF16", KuDNN::Element::TypeT::BF16}},
};

static std::map<int, std::pair<std::string, KuDNN::Layout>> g_layoutMap = {
    {INDEX0, {"ABCDE", KuDNN::Layout::ABCDE}},
    {INDEX1, {"ACDEB", KuDNN::Layout::ACDEB}},
};

static constexpr int g_dimRangeInitVal = 5;
static constexpr int g_dimRangeLowerBound = 1;
static constexpr int g_dimRangeUpperBound = 20;

struct Dim {
    size_t dim0;
    size_t dim1;
    size_t dim2;
    size_t dim3;
    size_t dim4;
    size_t Total()
    {
        return dim0 * dim1 * dim2 * dim3 * dim4;
    }
};

// to cover all branches of SumLayer, we need to construct srcs of different length and number
// this class abstracts this functionality for easy of use
class ProblemGenerator {
public:
    ProblemGenerator() : srcPool_(9, nullptr), dst_(nullptr), srcNum_(0),
        srcTensorInfo_({1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::A),
        dstTensorInfo_({1}, KuDNN::Element::TypeT::F32, KuDNN::Layout::A)
    {
        scalePool_ = {-1.0f, -0.75f, -0.5f, -0.25f, 0.25f, 0.5f, 0.75f, 1.0f, 2.0f};
    }
    vector<void*> Construct(int srcNum, Dim pbDim, KuDNN::Element::Type srcType, KuDNN::Element::Type dstType, KuDNN::Layout layout)
    {
        srcNum_ = srcNum;
        srcTensorInfo_ = {{pbDim.dim0, pbDim.dim1, pbDim.dim2, pbDim.dim3, pbDim.dim4}, srcType, layout};
        dstTensorInfo_ = {{pbDim.dim0, pbDim.dim1, pbDim.dim2, pbDim.dim3, pbDim.dim4}, dstType, layout};
        for (int i = 0; i != srcNum; ++i) {
            srcPool_[i] = malloc(pbDim.Total() * srcType.GetSize());
            if (srcPool_[i] == nullptr) {
                SafeFreeSrcPool();
                throw KuDNN::Service::BadAlloc();
            }
        }
        dst_  = malloc(pbDim.Total() * dstType.GetSize());
        if (dst_ == nullptr) {
            SafeFreeSrcPool();
            throw KuDNN::Service::BadAlloc();
        }
        return vector<void*>(srcPool_.begin(), srcPool_.begin() + srcNum);
    }
    vector<KuDNN::TensorInfo> GetSrcTensorInfo() { return vector<KuDNN::TensorInfo>(srcNum_, srcTensorInfo_); }
    KuDNN::TensorInfo GetDstTensorInfo() { return dstTensorInfo_; }
    float* GetScales() { return scalePool_.data(); }
    void* GetDst() { return dst_; }
    void Destruct()
    {
        SafeFreeSrcPool();
        if (dst_ != nullptr) {
            free(dst_);
            dst_ = nullptr;
        }
    }

private:
    vector<float> scalePool_;
    vector<void*> srcPool_;
    void *dst_;
    int srcNum_;
    KuDNN::TensorInfo srcTensorInfo_;
    KuDNN::TensorInfo dstTensorInfo_;
    void SafeFreeSrcPool()
    {
        for (size_t i = 0; i != srcPool_.size(); ++i) {
            if (srcPool_[i] != nullptr) {
                free(srcPool_[i]);
                srcPool_[i] = nullptr;
            }
        }
    }
};

static void DtSumLayer(int counts)
{
    char funcName[] = "DtSumLayer";
    DT_FUZZ_START(0, counts, funcName, 0)
    {
        int eleIndex = 0;
        int srcInfoElemTypeIndex = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], 0, 0,
            g_precisionMap.size() - 1);
        int dstInfoElemTypeIndex = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], 0, 0,
            g_precisionMap.size() - 1);
        int layoutIndex = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], 0, 0,
            g_layoutMap.size() - 1);
        KuDNN::Element::Type srcInfoType(g_precisionMap[srcInfoElemTypeIndex].second);
        KuDNN::Element::Type dstInfoType(g_precisionMap[dstInfoElemTypeIndex].second);
        KuDNN::Layout layout(g_layoutMap[layoutIndex].second);

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
        Dim dim = {dim0, dim1, dim2, dim3, dim4};

        char format[] = "DtSumLayer: Precision(srcInfo:dstInfo): %s:%s, Layout:%s , Shape: %zux%zux%zux%zux%zu ";
        printf(format, g_precisionMap[srcInfoElemTypeIndex].first.c_str(),
            g_precisionMap[dstInfoElemTypeIndex].first.c_str(), g_layoutMap[layoutIndex].first.c_str(),
            dim0, dim1, dim2, dim3, dim4);

        // src number 1 to 9
        ProblemGenerator pg;
        try {
            for (int i = 1; i != 10; ++i) {
                auto srcs = pg.Construct(i, dim, srcInfoType, dstInfoType, layout);
                KuDNN::SumLayer sumLayer(pg.GetSrcTensorInfo(), pg.GetScales(), pg.GetDstTensorInfo());
                sumLayer.Run(const_cast<const void**>(srcs.data()), pg.GetDst());
                pg.Destruct();
            }
        } catch (const std::exception& e) {
            pg.Destruct();
            std::cerr << "Sum exception: " << e.what() << std::endl;
        }

        printf("\n");
    }
    DT_FUZZ_END()
}

static void DtSumLayerOnce(int counts)
{
    char funcName[] = "DtSumLayerOnce";
    DT_FUZZ_START(0, counts, funcName, 0)
    {
        int eleIndex = 0;
        int srcInfoElemTypeIndex = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], 0, 0,
            g_precisionMap.size() - 1);
        int dstInfoElemTypeIndex = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], 0, 0,
            g_precisionMap.size() - 1);
        int layoutIndex = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], 0, 0,
            g_layoutMap.size() - 1);
        KuDNN::Element::Type srcInfoType(g_precisionMap[srcInfoElemTypeIndex].second);
        KuDNN::Element::Type dstInfoType(g_precisionMap[dstInfoElemTypeIndex].second);
        KuDNN::Layout layout(g_layoutMap[layoutIndex].second);

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
        vector<float> scales = { -1.0f, 1.0f };

        char format[] = "DtSumLayerOnce: Precision(srcInfo:dstInfo): %s:%s, Layout:%s , Shape: %zux%zux%zux%zux%zu ";
        printf(format, g_precisionMap[srcInfoElemTypeIndex].first.c_str(),
            g_precisionMap[dstInfoElemTypeIndex].first.c_str(), g_layoutMap[layoutIndex].first.c_str(),
            dim0, dim1, dim2, dim3, dim4);

        // test assign and move
        const KuDNN::TensorInfo dummySrcTensorInfo = {{dim0, dim1, dim2, dim3, dim4}, srcInfoType, layout};
        const KuDNN::TensorInfo dummyDstTensorInfo = {{dim0, dim1, dim2, dim3, dim4}, dstInfoType, layout};
        const std::vector<KuDNN::TensorInfo> dummySrcInfo2 = {dummySrcTensorInfo, dummySrcTensorInfo};
        KuDNN::SumLayer sum1(dummySrcInfo2, scales.data(), dummyDstTensorInfo);
        KuDNN::SumLayer sum2(sum1);
        sum2 = sum1;
        KuDNN::SumLayer sum4(std::move(sum1));
        KuDNN::SumLayer sum5(dummySrcInfo2, scales.data(), dummyDstTensorInfo);
        sum5 = std::move(sum2);

        // test exception cases
        try {
            KuDNN::SumLayer sumInvalid(dummySrcInfo2, nullptr, dummyDstTensorInfo);
        } catch (const std::exception& e) {
        }
        try {
            KuDNN::SumLayer sumInvalid(dummySrcInfo2, scales.data(), dummyDstTensorInfo);
            const void *src[2] = {nullptr, nullptr};
            sumInvalid.Run(src, nullptr);
        } catch (const std::exception& e) {
        }
        try {
            KuDNN::SumLayer sumInvalid(dummySrcInfo2, scales.data(), dummyDstTensorInfo);
            const void *src[2] = {nullptr, nullptr};
            const void *dst[10];
            sumInvalid.Run(src, dst);
        } catch (const std::exception& e) {
        }
        printf("\n");
    }
    DT_FUZZ_END()
}

static constexpr int g_sumFuzzTestCount = 200;
static constexpr int g_sumFuzzTestCountOnce = 1;

static Cmdinfo g_sumCmdTable[] = {
    {"DtSumLayer", DtSumLayer, g_sumFuzzTestCount},
    {"DtSumLayerOnce", DtSumLayerOnce, g_sumFuzzTestCountOnce},
};

int main(int argc, char *argv[])
{
    DT_Enable_Leak_Check(0, 0);
    if (argc > 1) {
        g_sumCmdTable[0] = {"DtSumLayer", DtSumLayer, atoi(argv[1])};
        g_sumCmdTable[1] = {"DtSumLayerOnce", DtSumLayerOnce, atoi(argv[1])};
    }

    int numOfTestItem = sizeof(g_sumCmdTable) / sizeof(g_sumCmdTable[0]);
    int testCount = 0;
    for (int testCaseIdx = 0; testCaseIdx < numOfTestItem; testCaseIdx++) {
        printf("g_sumCmdTable[%d] = %s test %d time\n", testCaseIdx, g_sumCmdTable[testCaseIdx].cmd, testCount);
        testCount = g_sumCmdTable[testCaseIdx].testCount;
        g_sumCmdTable[testCaseIdx].func(testCount);
    }
    return 0;
}