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
    {INDEX3, {"S32", KuDNN::Element::TypeT::S32}},
    {INDEX4, {"S8", KuDNN::Element::TypeT::S8}},
    {INDEX5, {"U8", KuDNN::Element::TypeT::U8}},
};

static std::map<int, std::pair<std::string, KuDNN::Element::TypeT>> g_precisionMap_bwd = {
    {INDEX0, {"F32", KuDNN::Element::TypeT::F32}},
    {INDEX1, {"F16", KuDNN::Element::TypeT::F16}},
    {INDEX2, {"BF16", KuDNN::Element::TypeT::BF16}},
};

static std::map<int, std::pair<std::string, KuDNN::PoolingFunction>> g_algMap = {
    {INDEX0, {"UNDEFINED", KuDNN::PoolingFunction::UNDEFINED}},
    {INDEX1, {"MAX", KuDNN::PoolingFunction::MAX}},
    {INDEX2, {"AVG_EXCLUDE_PADDING", KuDNN::PoolingFunction::AVG_EXCLUDE_PADDING}},
    {INDEX3, {"AVG_INCLUDE_PADDING", KuDNN::PoolingFunction::AVG_INCLUDE_PADDING}},
};

static std::map<int, std::pair<std::string, KuDNN::Layout>> g_layoutMap = {
    {INDEX0, {"ABCDE", KuDNN::Layout::ABCDE}},
    {INDEX1, {"ACDEB", KuDNN::Layout::ACDEB}},
};

static std::map<int, std::pair<std::string, KuDNN::Propagation>> g_PropagationMap = {
    {INDEX0, {"FORWARD_TRAINING", KuDNN::Propagation::UNDEFINED}},
    {INDEX1, {"FORWARD_TRAINING", KuDNN::Propagation::FORWARD_TRAINING}},
    {INDEX2, {"FORWARD_INFERENCE", KuDNN::Propagation::FORWARD_INFERENCE}},
};

static constexpr int g_dimRangeInitVal = 1;
static constexpr int g_dimRangeLowerBound = 1;
static constexpr int g_dimRangeUpperBound = 16;

static void DtFuzzPoolingFwd(int counts)
{
    char funcName[] = "DtFuzzPoolingFwd";
    DT_FUZZ_START(0, counts, funcName, 0)
    {
        int eleIndex = 0;
        int srcElemTypeIndex = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], 0, 0, g_precisionMap.size() - 1);
        // int dstElemTypeIndex = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], 0, 0, g_precisionMap.size() - 1);
        int algIndex = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], 0, 0, g_algMap.size() - 1);
        int layoutIndex = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], 0, 0, g_layoutMap.size() - 1);
        int propagationIndex = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], 0, 0, g_PropagationMap.size() - 1);
        size_t bc = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], g_dimRangeInitVal, g_dimRangeLowerBound,
            g_dimRangeUpperBound);
        size_t ic = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], g_dimRangeInitVal, g_dimRangeLowerBound,
            g_dimRangeUpperBound);
        size_t id = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], g_dimRangeInitVal, g_dimRangeLowerBound,
            g_dimRangeUpperBound);
        size_t ih = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], g_dimRangeInitVal, g_dimRangeLowerBound,
            g_dimRangeUpperBound);
        size_t iw = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], g_dimRangeInitVal, g_dimRangeLowerBound,
            g_dimRangeUpperBound);

        size_t oc = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], g_dimRangeInitVal, g_dimRangeLowerBound,
            g_dimRangeUpperBound);
        size_t kd = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], g_dimRangeInitVal, g_dimRangeLowerBound,
            g_dimRangeUpperBound);
        size_t kh = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], g_dimRangeInitVal, g_dimRangeLowerBound,
            g_dimRangeUpperBound);
        size_t kw = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], g_dimRangeInitVal, g_dimRangeLowerBound,
            g_dimRangeUpperBound);

        size_t sd = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], g_dimRangeInitVal, g_dimRangeLowerBound,
            g_dimRangeUpperBound);
        size_t sh = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], g_dimRangeInitVal, g_dimRangeLowerBound,
            g_dimRangeUpperBound);
        size_t sw = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], g_dimRangeInitVal, g_dimRangeLowerBound,
            g_dimRangeUpperBound);

        size_t dd = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], 0, 0, 2);
        size_t dh = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], 0, 0, 2);
        size_t dw = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], 0, 0, 2);

        size_t pd = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], g_dimRangeInitVal, g_dimRangeLowerBound,
            g_dimRangeUpperBound);
        size_t ph = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], g_dimRangeInitVal, g_dimRangeLowerBound,
            g_dimRangeUpperBound);
        size_t pw = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], g_dimRangeInitVal, g_dimRangeLowerBound,
            g_dimRangeUpperBound);

        size_t dkd = 1 + (kd -1) * (dd + 1);
        size_t dkh = 1 + (kh -1) * (dh + 1);
        size_t dkw = 1 + (kw -1) * (dw + 1);

        int64_t tmpOd = std::ceil(((int64_t)id - (int64_t)dkd + (int64_t)pd) / (int64_t)sd) + 1;
        int64_t tmpOh = std::ceil(((int64_t)ih - (int64_t)dkh + (int64_t)ph) / (int64_t)sh) + 1;
        int64_t tmpOw = std::ceil(((int64_t)iw - (int64_t)dkw + (int64_t)pw) / (int64_t)sw) + 1;

        size_t od = static_cast<size_t>(std::max(tmpOd, (int64_t)1));
        size_t oh = static_cast<size_t>(std::max(tmpOh, (int64_t)1));
        size_t ow = static_cast<size_t>(std::max(tmpOw, (int64_t)1));

        KuDNN::Element::Type srcType(g_precisionMap[srcElemTypeIndex].second);
        KuDNN::Element::Type dstType(g_precisionMap[srcElemTypeIndex].second);
        KuDNN::PoolingFunction alg(g_algMap[algIndex].second);
        KuDNN::Layout layout(g_layoutMap[layoutIndex].second);

        KuDNN::Shape kernel(kd, kh, kw);
        KuDNN::Shape strides(sd, sh, sw);
        KuDNN::Shape dilates(dd, dh, dw);
        KuDNN::Shape paddingL(pd, ph, pw);
        KuDNN::Shape paddingR(0, 0, 0);
        const KuDNN::TensorInfo srcTensor = {{bc, ic, id, ih, iw}, srcType, layout};
        const KuDNN::TensorInfo dstTensor = {{bc, ic, od, oh, ow}, dstType, layout};

        size_t srcElemSz = srcType.GetSize();
        size_t dstElemSz = dstType.GetSize();

        KuDNN::Propagation propagation(g_PropagationMap[propagationIndex].second);

        char format[] = "Forward Pooling: %s, Precision(src:dst): %s:%s"
            ", bc=%zu, ic=%zu, id=%zu, ih=%zu, iw=%zu, oc=%zu, kd=%zu, kh=%zu, kw=%zu, pd=%zu, ph=%zu, pw=%zu, sd=%zu,"
            "sh=%zu, sw=%zu, dd=%zu ,dh=%zu, dw=%zu, od=%zu, oh=%zu, ow=%zu";
        printf(format, g_algMap[algIndex].first.c_str(), g_precisionMap[srcElemTypeIndex].first.c_str(),
            g_precisionMap[srcElemTypeIndex].first.c_str(), bc, ic, id, ih, iw, oc, kd, kh, kw, pd, ph, pw, sd, sh, sw,
            dd, dh, dw, od, oh, ow);

        void *src = nullptr;
        void *dst = nullptr;
        void *ws = nullptr;

        try {
            KuDNN::PoolingLayerFWD pool1(propagation, alg, srcTensor, dstTensor, strides, kernel, dilates, paddingL, paddingR);
            KuDNN::PoolingLayerFWD pool2(pool1);
            pool2 = pool1;
            KuDNN::PoolingLayerFWD pool3(pool1);
            pool3 = std::move(pool1);
            KuDNN::PoolingLayerFWD pool4 = (std::move(pool2));
            pool4.SetPadType(srcType);

            KuDNN::PoolingLayerFWD poolFwdLayer(propagation, alg, srcTensor, dstTensor, strides, kernel, dilates, paddingL, paddingR);
            src = malloc(bc * ic * id * ih * iw * srcElemSz);
            void *tmpsrc = src;
            if (src == nullptr) {
                std::cerr << "memory allocation failed" << std::endl;
                return;
            }

            dst = malloc(bc * ic * od * oh * ow * dstElemSz);
            if (dst == nullptr) {
                free(src);
                std::cerr << "memory allocation failed" << std::endl;
                return;
            }

            KuDNN::Element::Type wsType = KuDNN::Element::TypeT::S32;
            size_t wsElemSz = wsType.GetSize();
            ws = malloc(bc * ic * od * oh * ow * wsElemSz);
            if (ws == nullptr) {
                free(src);
                free(dst);
                std::cerr << "memory allocation failed" << std::endl;
                return;
            }

            poolFwdLayer.Run(src, dst, ws);
            free(tmpsrc);
            free(dst);
            free(ws);
        } catch (const KuDNN::Service::Unsupported& e) {
            if (src != nullptr) {
                free(src);
            }
            if (dst != nullptr) {
                free(dst);
            }
            if (ws != nullptr) {
                free(ws);
            }
            printf("Unsupported Configuration");
        } catch (const KuDNN::Service::LogicError& e) {
            if (src != nullptr) {
                free(src);
            }
            if (dst != nullptr) {
                free(dst);
            }
            if (ws != nullptr) {
                free(ws);
            }
            printf("Bad shapes' arguments");
        }
        printf("\n");
    }
    DT_FUZZ_END()
}

static void DtFuzzPoolingBwd(int counts)
{
    char funcName[] = "DtFuzzPoolingBwd";
    DT_FUZZ_START(0, counts, funcName, 0)
    {
        int eleIndex = 0;
        int srcElemTypeIndex = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], 0, 0, g_precisionMap_bwd.size() - 1);
        // int dstElemTypeIndex = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], 0, 0, g_precisionMap.size() - 1);
        int algIndex = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], 0, 0, g_algMap.size() - 1);
        int layoutIndex = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], 0, 0, g_layoutMap.size() - 1);
        size_t bc = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], g_dimRangeInitVal, g_dimRangeLowerBound,
            g_dimRangeUpperBound);
        size_t ic = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], g_dimRangeInitVal, g_dimRangeLowerBound,
            g_dimRangeUpperBound);
        size_t id = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], g_dimRangeInitVal, g_dimRangeLowerBound,
            g_dimRangeUpperBound);
        size_t ih = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], g_dimRangeInitVal, g_dimRangeLowerBound,
            g_dimRangeUpperBound);
        size_t iw = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], g_dimRangeInitVal, g_dimRangeLowerBound,
            g_dimRangeUpperBound);

        size_t oc = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], g_dimRangeInitVal, g_dimRangeLowerBound,
            g_dimRangeUpperBound);
        size_t kd = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], g_dimRangeInitVal, g_dimRangeLowerBound,
            g_dimRangeUpperBound);
        size_t kh = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], g_dimRangeInitVal, g_dimRangeLowerBound,
            g_dimRangeUpperBound);
        size_t kw = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], g_dimRangeInitVal, g_dimRangeLowerBound,
            g_dimRangeUpperBound);

        size_t sd = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], g_dimRangeInitVal, g_dimRangeLowerBound,
            g_dimRangeUpperBound);
        size_t sh = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], g_dimRangeInitVal, g_dimRangeLowerBound,
            g_dimRangeUpperBound);
        size_t sw = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], g_dimRangeInitVal, g_dimRangeLowerBound,
            g_dimRangeUpperBound);

        size_t dd = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], 0, 0, 2);
        size_t dh = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], 0, 0, 2);
        size_t dw = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], 0, 0, 2);

        size_t pd = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], g_dimRangeInitVal, g_dimRangeLowerBound,
            g_dimRangeUpperBound);
        size_t ph = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], g_dimRangeInitVal, g_dimRangeLowerBound,
            g_dimRangeUpperBound);
        size_t pw = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], g_dimRangeInitVal, g_dimRangeLowerBound,
            g_dimRangeUpperBound);

        size_t dkd = 1 + (kd -1) * (dd + 1);
        size_t dkh = 1 + (kh -1) * (dh + 1);
        size_t dkw = 1 + (kw -1) * (dw + 1);

        int64_t tmpOd = std::ceil(((int64_t)id - (int64_t)dkd + (int64_t)pd) / (int64_t)sd) + 1;
        int64_t tmpOh = std::ceil(((int64_t)ih - (int64_t)dkh + (int64_t)ph) / (int64_t)sh) + 1;
        int64_t tmpOw = std::ceil(((int64_t)iw - (int64_t)dkw + (int64_t)pw) / (int64_t)sw) + 1;

        size_t od = static_cast<size_t>(std::max(tmpOd, (int64_t)1));
        size_t oh = static_cast<size_t>(std::max(tmpOh, (int64_t)1));
        size_t ow = static_cast<size_t>(std::max(tmpOw, (int64_t)1));

        KuDNN::Element::Type srcType(g_precisionMap_bwd[srcElemTypeIndex].second);
        KuDNN::Element::Type dstType(g_precisionMap_bwd[srcElemTypeIndex].second);
        size_t srcElemSz = srcType.GetSize();
        size_t dstElemSz = dstType.GetSize();
        KuDNN::PoolingFunction alg(g_algMap[algIndex].second);
        KuDNN::Layout layout(g_layoutMap[layoutIndex].second);

        KuDNN::Shape kernel(kd, kh, kw);
        KuDNN::Shape strides(sd, sh, sw);
        KuDNN::Shape dilates(dd, dh, dw);
        KuDNN::Shape paddingL(pd, ph, pw);
        KuDNN::Shape paddingR(0, 0, 0);
        const KuDNN::TensorInfo srcTensor = {{bc, ic, id, ih, iw}, srcType, layout};
        const KuDNN::TensorInfo dstTensor = {{bc, ic, od, oh, ow}, dstType, layout};

        char format[] = "Backward Pooling: %s, Precision(src:dst): %s:%s"
            ", bc=%zu, ic=%zu, id=%zu, ih=%zu, iw=%zu, oc=%zu, kd=%zu, kh=%zu, kw=%zu, pd=%zu, ph=%zu, pw=%zu, sd=%zu,"
            "sh=%zu, sw=%zu, dd=%zu ,dh=%zu, dw=%zu, od=%zu, oh=%zu, ow=%zu";
        printf(format, g_algMap[algIndex].first.c_str(), g_precisionMap[srcElemTypeIndex].first.c_str(),
            g_precisionMap[srcElemTypeIndex].first.c_str(), bc, ic, id, ih, iw, oc, kd, kh, kw, pd, ph, pw, sd, sh, sw,
            dd, dh, dw, od, oh, ow);

        void *src = nullptr;
        void *dst = nullptr;
        void *ws = nullptr;

        try {
            KuDNN::PoolingLayerBWD pool1(alg, srcTensor, dstTensor, strides, kernel, dilates, paddingL, paddingR);
            KuDNN::PoolingLayerBWD pool2(pool1);
            pool2 = pool1;
            KuDNN::PoolingLayerBWD pool3(pool1);
            pool3 = std::move(pool1);
            KuDNN::PoolingLayerBWD pool4 = (std::move(pool2));

            KuDNN::PoolingLayerBWD poolBwdLayer(alg, srcTensor, dstTensor, strides, kernel, dilates, paddingL, paddingR);
            src = malloc(bc * ic * id * ih * iw * srcElemSz);
            void *tmpsrc = src;
            if (src == nullptr) {
                std::cerr << "memory allocation failed" << std::endl;
                return;
            }

            dst = malloc(bc * ic * od * oh * ow * dstElemSz);
            if (dst == nullptr) {
                free(src);
                std::cerr << "memory allocation failed" << std::endl;
                return;
            }
            KuDNN::Element::Type wsType = KuDNN::Element::TypeT::S32;
            size_t wsElemSz = wsType.GetSize();
            ws = malloc(bc * ic * od * oh * ow * wsElemSz);
            if (ws == nullptr) {
                free(src);
                free(dst);
                std::cerr << "memory allocation failed" << std::endl;
                return;
            }
            poolBwdLayer.Run(src, dst, ws);

            free(tmpsrc);
            free(dst);
            free(ws);
        } catch (const KuDNN::Service::Unsupported& e) {
            if (src != nullptr) {
                free(src);
            }
            if (dst != nullptr) {
                free(dst);
            }
            if (ws != nullptr) {
                free(ws);
            }
            printf("Unsupported Configuration");
        } catch (const KuDNN::Service::LogicError& e) {
            if (src != nullptr) {
                free(src);
            }
            if (dst != nullptr) {
                free(dst);
            }
            if (ws != nullptr) {
                free(ws);
            }
            printf("Bad shapes' arguments");
        }
        printf("\n");
    }
    DT_FUZZ_END()
}


const int  g_poololutionFuzzTestCount = 6000;

Cmdinfo poololutionCmdTable[] = {
    {"PoolingFwd", DtFuzzPoolingFwd, g_poololutionFuzzTestCount},
    {"PoolingBwd", DtFuzzPoolingBwd, g_poololutionFuzzTestCount},
};

int main(int argc, char *argv[])
{
    DT_Enable_Leak_Check(0, 0);
    if (argc > 1) {
        poololutionCmdTable[0] = {"PoolingFwd", DtFuzzPoolingFwd, atoi(argv[1])};
        poololutionCmdTable[1] = {"PoolingBwd", DtFuzzPoolingBwd, atoi(argv[1])};
    }

    int numOfTestItem = sizeof(poololutionCmdTable) / sizeof(poololutionCmdTable[0]);
    int testCount = 0;

    for (int testCaseIdx = 0; testCaseIdx < numOfTestItem; testCaseIdx++) {
        testCount = poololutionCmdTable[testCaseIdx].testCount;
        printf("poolCmdTable[%d] = %s test %d time\n", testCaseIdx, poololutionCmdTable[testCaseIdx].cmd, testCount);
        poololutionCmdTable[testCaseIdx].func(testCount);
    }
    return 0;
}