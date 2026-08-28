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

static std::map<KuDNN::Layout, int> g_tag2DimMap = {
    {KuDNN::Layout::ABC, KuDNN::DIM_3},
    {KuDNN::Layout::ACB, KuDNN::DIM_3},
    {KuDNN::Layout::ABCD, KuDNN::DIM_4},
    {KuDNN::Layout::ACDB, KuDNN::DIM_4},
    {KuDNN::Layout::ABCDE, KuDNN::DIM_5},
    {KuDNN::Layout::ACDEB, KuDNN::DIM_5}
};

static std::vector<std::pair<std::string, KuDNN::Layout>> g_tagList = {
    {"ABC", KuDNN::Layout::ABC},
    {"ACB", KuDNN::Layout::ACB},
    {"ABCD", KuDNN::Layout::ABCD},
    {"ACDB", KuDNN::Layout::ACDB},
    {"ABCDE", KuDNN::Layout::ABCDE},
    {"ACDEB", KuDNN::Layout::ACDEB}
};

static std::vector<std::pair<std::string, KuDNN::ResamplingAlg>> g_algList = {
    {"NEAREST_NEIGHBOR", KuDNN::ResamplingAlg::NEAREST_NEIGHBOR},
    {"LINEAR", KuDNN::ResamplingAlg::LINEAR}
};

static std::vector<std::pair<std::string, KuDNN::Element::TypeT>> g_typeList = {
    {"F32", KuDNN::Element::TypeT::F32},
    {"F16", KuDNN::Element::TypeT::F16},
    {"BF16", KuDNN::Element::TypeT::BF16},
    {"S32", KuDNN::Element::TypeT::S32},
    {"S8", KuDNN::Element::TypeT::S8},
    {"U8", KuDNN::Element::TypeT::U8}
};

static std::string PrepareRunFuzzResampling(KuDNN::Shape &srcShape, KuDNN::Shape &dstShape, int dim,
    size_t mb, size_t c, size_t id, size_t ih, size_t iw, size_t od, size_t oh, size_t ow) {
    if (dim == 3) {
        srcShape = {mb, c, iw};
        dstShape = {mb, c, ow};
        return "mb" + std::to_string(mb) + "ic" + std::to_string(c) +
               "_iw" + std::to_string(iw) + "_ow" + std::to_string(ow);
    } else if (dim == 4) {
        srcShape = {mb, c, ih, iw};
        dstShape = {mb, c, oh, ow};
        return "mb" + std::to_string(mb) + "ic" + std::to_string(c) +
               "_ih" + std::to_string(ih) + "iw" + std::to_string(iw) +
               "_oh" + std::to_string(oh) + "ow" + std::to_string(ow);
    } else {
        srcShape = {mb, c, id, ih, iw};
        dstShape = {mb, c, od, oh, ow};
        return "mb" + std::to_string(mb) + "ic" + std::to_string(c) +
               "_id" + std::to_string(id) + "ih" + std::to_string(ih) + "iw" + std::to_string(iw) +
               "_od" + std::to_string(od) + "oh" + std::to_string(oh) + "ow" + std::to_string(ow);
    }
}

template <typename Layer>
static void RunFuzzResampling(const KuDNN::TensorInfo &srcTensorInfo, const KuDNN::TensorInfo &dstTensorInfo,
    KuDNN::ResamplingAlg alg)
{
    void *src = NULL;
    void *dst = NULL;
    try {
        // Copy and move.
        Layer resampling1(srcTensorInfo, dstTensorInfo, alg);
        Layer resampling2(resampling1);
        resampling2 = resampling1;
        Layer resampling3(resampling1);
        resampling3 = std::move(resampling1);
        Layer resampling4(std::move(resampling2));

        // Run resampling.
        Layer layer(srcTensorInfo, dstTensorInfo, alg);
        src = malloc(srcTensorInfo.GetTotalTensorSize() * srcTensorInfo.GetType().GetSize());
        if (src == NULL) {
            std::cerr << "memory allocation failed." << std::endl;
            return ;
        }
        dst = malloc(dstTensorInfo.GetTotalTensorSize() * dstTensorInfo.GetType().GetSize());
        if (dst == NULL) {
            std::cerr << "memory allocation failed." << std::endl;
            return ;
        }
        layer.Run(src, dst);
    } catch (const KuDNN::Service::Unsupported& e) {
        if (src != NULL) {
            free(src);
        }
        if (dst != NULL) {
            free(dst);
        }
        printf("Unsupported Configuration.");
    } catch (const KuDNN::Service::LogicError& e) {
        if (src != NULL) {
            free(src);
        }
        if (dst != NULL) {
            free(dst);
        }
        printf("Bad shapes' arguments.");
    }
    printf("\n");
    if (src != NULL) {
        free(src);
    }
    if (dst != NULL) {
        free(dst);
    }
}

static constexpr int g_dimRangeInitVal = 1;
static constexpr int g_dimRangeLowerBound = 1;
static constexpr int g_dimRangeUpperBound = 30;

static constexpr int g_resamplingFuzzRandomTestCount = 200;
static constexpr int g_resamplingFuzzCoverTestCount = 2;

static void DtFuzzResamplingRandom(int counts)
{
    char funcName[] = "DtFuzzResamplingRandom";

    DT_FUZZ_START(0, counts, funcName, 0)
    {
        int eleIndex = 0;
        size_t dim_iw = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], g_dimRangeInitVal, g_dimRangeLowerBound,
            g_dimRangeUpperBound);
        size_t dim_ih = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], g_dimRangeInitVal, g_dimRangeLowerBound,
            g_dimRangeUpperBound);
        size_t dim_id = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], g_dimRangeInitVal, g_dimRangeLowerBound,
            g_dimRangeUpperBound);
        size_t dim_ow = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], g_dimRangeInitVal, g_dimRangeLowerBound,
            g_dimRangeUpperBound);
        size_t dim_oh = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], g_dimRangeInitVal, g_dimRangeLowerBound,
            g_dimRangeUpperBound);
        size_t dim_od = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], g_dimRangeInitVal, g_dimRangeLowerBound,
            g_dimRangeUpperBound);
        size_t dim_c = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], g_dimRangeInitVal, g_dimRangeLowerBound,
            g_dimRangeUpperBound);
        size_t dim_mb = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], g_dimRangeInitVal, g_dimRangeLowerBound,
            g_dimRangeUpperBound);

        int tagIndex = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], 0, 0, g_tagList.size() - 1);
        int algTypeIndex = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], 0, 0, g_algList.size() - 1);
        int srcTypeIndex = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], 0, 0, g_typeList.size() - 1);
        int dstTypeIndex = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], 0, 0, g_typeList.size() - 1);

        const std::string &tagName = g_tagList[tagIndex].first;
        const std::string &algName = g_algList[algTypeIndex].first;
        const std::string &srcTypeName = g_typeList[srcTypeIndex].first;
        const std::string &dstTypeName = g_typeList[dstTypeIndex].first;

        KuDNN::Layout tag = g_tagList[tagIndex].second;
        KuDNN::ResamplingAlg alg = g_algList[algTypeIndex].second;
        KuDNN::Element::TypeT srcType = g_typeList[srcTypeIndex].second;
        KuDNN::Element::TypeT dstType = g_typeList[dstTypeIndex].second;
        int dim = g_tag2DimMap[tag];

        KuDNN::Shape srcShape;
        KuDNN::Shape dstShape;
        std::string probDesc = PrepareRunFuzzResampling(srcShape, dstShape, dim,
            dim_mb, dim_c, dim_id, dim_ih, dim_iw, dim_od, dim_oh, dim_ow);

        KuDNN::TensorInfo srcTensorInfo = {srcShape, srcType, tag};
        KuDNN::TensorInfo dstTensorInfo = {dstShape, dstType, tag};

        printf("Running resampling: --dir=FWD_D --alg=%s --tag=%s --sdt=%s --ddt=%s %s",
            algName.c_str(), tagName.c_str(), srcTypeName.c_str(), dstTypeName.c_str(), probDesc.c_str());
        RunFuzzResampling<KuDNN::ResamplingLayerFWD>(srcTensorInfo, dstTensorInfo, alg);

        printf("Running resampling: --dir=BWD_D --alg=%s --tag=%s --sdt=%s --ddt=%s %s",
            algName.c_str(), tagName.c_str(), srcTypeName.c_str(), dstTypeName.c_str(), probDesc.c_str());
        RunFuzzResampling<KuDNN::ResamplingLayerBWD>(srcTensorInfo, dstTensorInfo, alg);
    }
    DT_FUZZ_END()
}

static void DtFuzzResamplingCover(int counts)
{
    char funcName[] = "DtFuzzResamplingCover";

    DT_FUZZ_START(0, counts, funcName, 0)
    {
        int eleIndex = 0;
        size_t dim_iw = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], g_dimRangeInitVal, g_dimRangeLowerBound,
            g_dimRangeUpperBound);
        size_t dim_ih = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], g_dimRangeInitVal, g_dimRangeLowerBound,
            g_dimRangeUpperBound);
        size_t dim_id = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], g_dimRangeInitVal, g_dimRangeLowerBound,
            g_dimRangeUpperBound);
        size_t dim_ow = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], g_dimRangeInitVal, g_dimRangeLowerBound,
            g_dimRangeUpperBound);
        size_t dim_oh = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], g_dimRangeInitVal, g_dimRangeLowerBound,
            g_dimRangeUpperBound);
        size_t dim_od = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], g_dimRangeInitVal, g_dimRangeLowerBound,
            g_dimRangeUpperBound);
        size_t dim_c = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], g_dimRangeInitVal, g_dimRangeLowerBound,
            g_dimRangeUpperBound);
        size_t dim_mb = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], g_dimRangeInitVal, g_dimRangeLowerBound,
            g_dimRangeUpperBound);

        for (auto tagPair : g_tagList) {
            for (auto algPair : g_algList) {
                for (auto srcTypePair : g_typeList) {
                    for (auto dstTypePair : g_typeList) {
                        const std::string &tagName = tagPair.first;
                        const std::string &algName = algPair.first;
                        const std::string &srcTypeName = srcTypePair.first;
                        const std::string &dstTypeName = dstTypePair.first;

                        KuDNN::Layout tag = tagPair.second;
                        KuDNN::ResamplingAlg alg = algPair.second;
                        KuDNN::Element::TypeT srcType = srcTypePair.second;
                        KuDNN::Element::TypeT dstType = dstTypePair.second;
                        int dim = g_tag2DimMap[tag];

                        KuDNN::Shape srcShape;
                        KuDNN::Shape dstShape;
                        std::string probDesc = PrepareRunFuzzResampling(srcShape, dstShape, dim,
                            dim_mb, dim_c, dim_id, dim_ih, dim_iw, dim_od, dim_oh, dim_ow);

                        KuDNN::TensorInfo srcTensorInfo = {srcShape, srcType, tag};
                        KuDNN::TensorInfo dstTensorInfo = {dstShape, dstType, tag};

                        printf("Running resampling: --dir=FWD_D --alg=%s --tag=%s --sdt=%s --ddt=%s %s",
                            algName.c_str(), tagName.c_str(),
                            srcTypeName.c_str(), dstTypeName.c_str(), probDesc.c_str());
                        RunFuzzResampling<KuDNN::ResamplingLayerFWD>(srcTensorInfo, dstTensorInfo, alg);

                        printf("Running resampling: --dir=BWD_D --alg=%s --tag=%s --sdt=%s --ddt=%s %s",
                            algName.c_str(), tagName.c_str(),
                            srcTypeName.c_str(), dstTypeName.c_str(), probDesc.c_str());
                        RunFuzzResampling<KuDNN::ResamplingLayerBWD>(srcTensorInfo, dstTensorInfo, alg);
                    }
                }
            }
        }
    }
    DT_FUZZ_END()
}

Cmdinfo resamplingCmdTable[] = {
    {"ResamplingRandom", DtFuzzResamplingRandom, g_resamplingFuzzRandomTestCount},
    {"ResamplingCover", DtFuzzResamplingCover, g_resamplingFuzzCoverTestCount}
};

int main(int argc, char *argv[])
{
    DT_Enable_Leak_Check(0, 0);
    if (argc > 1) {
        resamplingCmdTable[0] = {"ResamplingRandom", DtFuzzResamplingRandom, atoi(argv[1])};
        resamplingCmdTable[1] = {"ResamplingCover", DtFuzzResamplingCover, atoi(argv[1])};
    }

    int numOfTestItem = sizeof(resamplingCmdTable) / sizeof(resamplingCmdTable[0]);
    int testCount = 0;
    for (int testCaseIdx = 0; testCaseIdx < numOfTestItem; testCaseIdx++) {
        printf("resamplingCmdTable[%d] = %s test %d time\n",
            testCaseIdx, resamplingCmdTable[testCaseIdx].cmd, testCount);
        testCount = resamplingCmdTable[testCaseIdx].testCount;
        resamplingCmdTable[testCaseIdx].func(testCount);
    }
    return 0;
}