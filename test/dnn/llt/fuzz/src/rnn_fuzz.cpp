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

static std::map<int, std::pair<std::string, KuDNN::RnnAlgorithm>> g_rnnFuncMap = {
    {INDEX0, {"RNN", KuDNN::RnnAlgorithm::RNN}},
    {INDEX1, {"LSTM", KuDNN::RnnAlgorithm::LSTM}},
    {INDEX2, {"GRU", KuDNN::RnnAlgorithm::GRU}},
    {INDEX3, {"LBR_GRU", KuDNN::RnnAlgorithm::LBR_GRU}},
    {INDEX4, {"AUGRU", KuDNN::RnnAlgorithm::AUGRU}},
    {INDEX5, {"LBR_AUGRU", KuDNN::RnnAlgorithm::LBR_AUGRU}},
    {INDEX6, {"UNIMPLEMENTED", KuDNN::RnnAlgorithm::UNIMPLEMENTED}},
};

static std::map<int, std::pair<std::string, KuDNN::Element::TypeT>> g_precisionMap = {
    {INDEX0, {"F32", KuDNN::Element::TypeT::F32}},
    {INDEX1, {"BF16", KuDNN::Element::TypeT::BF16}},
    {INDEX2, {"F16", KuDNN::Element::TypeT::F16}},
};

static std::map<int, std::pair<std::string, KuDNN::ActivateFunctionRNN>> g_activationFuncMap = {
    {INDEX0, {"RELU", KuDNN::ActivateFunctionRNN::RELU}},
    {INDEX1, {"TANH", KuDNN::ActivateFunctionRNN::TANH}},
    {INDEX2, {"LOGISTIC", KuDNN::ActivateFunctionRNN::LOGISTIC}},
    {INDEX3, {"UNIMPLEMENTED", KuDNN::ActivateFunctionRNN::UNIMPLEMENTED}},
};

static std::map<int, std::pair<std::string, KuDNN::Detail::ExecutionDirectionT>> g_directionFuncMap = {
    {INDEX0, {"L2R", KuDNN::Detail::ExecutionDirectionT::L2R}},
    {INDEX1, {"R2L", KuDNN::Detail::ExecutionDirectionT::R2L}},
    {INDEX2, {"BI_CONCAT", KuDNN::Detail::ExecutionDirectionT::BI_CONCAT}},
    {INDEX3, {"BI_SUM", KuDNN::Detail::ExecutionDirectionT::BI_SUM}},
};

static std::map<int, bool> g_boolMap = {
    {INDEX0, false},
    {INDEX1, true},
};

bool CheckAnyNullPtr()
{
    return false;
}

template <typename... Args>
bool CheckAnyNullPtr(void* first, Args... args)
{
    if (first == nullptr) {
        return true;
    }
    return CheckAnyNullPtr(args...);
}

void *CreateRawPointer(int isAlloc, size_t sizeBytes)
{
    void *rawPtr = nullptr;
    if (isAlloc) {
        rawPtr = calloc(1, sizeBytes);
    }
    return rawPtr;
}

/**
 * @brief 释放void*数组中的有效内存
 * @param voidArray void*指针数组
 * @param elementCount 数组长度
 */
void ReleaseVoidArray(void **voidArray, int elementCount)
{
    for (int i = 0; i < elementCount; ++i) {
        if (voidArray[i] != nullptr) {
            free(voidArray[i]);
            voidArray[i] = nullptr;
        }
    }
}

static constexpr int g_dimRangeInitVal = 1;
static constexpr int g_dimRangeLowerBound = 1;
static constexpr int g_dimRangeUpperBound = 64;

static void SetRnnFwd(KuDNN::RnnCfg &rnn, KuDNN::Element::Type &type)
{
    rnn.nDir = ((rnn.execDir == KuDNN::Detail::ExecutionDirectionT::L2R) ||
        (rnn.execDir == KuDNN::Detail::ExecutionDirectionT::R2L)) ? 1 : 2;
    rnn.weightsLayerLd = rnn.nGates * rnn.dhc;
    rnn.weightsIterLd = rnn.nGates * rnn.dhc;

    rnn.scratchGatesLd = std::max(rnn.dlc, rnn.nGates * rnn.dhc);
    rnn.srcLayerLd = rnn.slc;
    rnn.srcIterLd = rnn.sic;
    rnn.srcIterCLd = 1;
    rnn.dstLayerLd = rnn.dlc;
    rnn.dstIterLd = rnn.dic;
    rnn.dstIterCLd = 1;
    rnn.projHtLd = rnn.dhc;
    rnn.nBias = rnn.nGates + rnn.isLbr;

    rnn.wsStatesLayerLd = std::max(rnn.dlc, std::max(rnn.dhc, rnn.sic));
    rnn.wsStatesIterLd = rnn.wsStatesLayerLd;
    rnn.wsStatesLayerSize = (size_t)(rnn.nLayer + 1) * rnn.nDir * (rnn.nIter + 1) * rnn.mb *
        rnn.wsStatesLayerLd * type.GetSize();
    rnn.scratchGatesNld = rnn.mb;
    rnn.weightsProjectionLd = rnn.dic;
    rnn.wsGatesLd = rnn.dhc * rnn.nGates;
    rnn.wsGatesNld = rnn.mb;
    rnn.alpha = 1;
}

static void DtFuzzRnnForward(int counts)
{
    char funcName[] = "DtFuzzRnnForward";
    DT_FUZZ_START(0, counts, funcName, 0)
    {
        int eleIndex = 0;
        int srcTypeIndex = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], 0, 0, g_precisionMap.size() - 1);
        int biasTypeIndex = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], 0, 0, g_precisionMap.size() - 1);
        int actFuncIndex = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], 0, 0, g_activationFuncMap.size() - 1);
        int dirIndex = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], 0, 0, g_directionFuncMap.size() - 1);
        int gatesLinearIndex = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], 0, 0, g_boolMap.size() - 1);

        KuDNN::Element::Type srcType(g_precisionMap[srcTypeIndex].second);
        KuDNN::Element::Type biasType(g_precisionMap[biasTypeIndex].second);
        KuDNN::ActivateFunctionRNN actfKind = g_activationFuncMap[actFuncIndex].second;
        KuDNN::RnnCfg rnn;
        rnn.nLayer = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], 1, 1, 11);
        rnn.nIter = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], 1, 1, 11);
        rnn.mb = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++],
            g_dimRangeInitVal, g_dimRangeLowerBound, g_dimRangeUpperBound);
        rnn.dhc = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++],
            g_dimRangeInitVal, g_dimRangeLowerBound, g_dimRangeUpperBound);
        rnn.sic = rnn.dhc;
        rnn.slc = rnn.sic;
        rnn.dic = rnn.dhc;
        rnn.dlc = rnn.dic;
        rnn.gatesFuncLinearMode = g_boolMap[gatesLinearIndex];
        if (rnn.nLayer == 1) {
            rnn.slc = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++],
                g_dimRangeInitVal, g_dimRangeLowerBound, g_dimRangeUpperBound);
        }
        if (rnn.nIter == 1) {
            rnn.sic = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++],
                g_dimRangeInitVal, g_dimRangeLowerBound, g_dimRangeUpperBound);
        }

        rnn.execDir = g_directionFuncMap[dirIndex].second;
        rnn.nGates = 1;
        rnn.isLbr = 0;

        SetRnnFwd(rnn, srcType);

        const std::vector<KuDNN::TensorInfo> srcInfos = {
            {KuDNN::Shape(rnn.nIter, rnn.mb, rnn.slc), srcType, KuDNN::Layout::ABC,
                KuDNN::Shape(rnn.mb * rnn.slc, g_dimRangeUpperBound, g_dimRangeUpperBound)},
        };
        const std::vector<KuDNN::TensorInfo> dstInfos = {
            {KuDNN::Shape(rnn.nIter, rnn.mb, rnn.dlc), srcType, KuDNN::Layout::ABC,
                KuDNN::Shape(rnn.mb * rnn.dlc, g_dimRangeUpperBound, g_dimRangeUpperBound)},
        };
        KuDNN::TensorInfo biasInfo = {KuDNN::Shape(rnn.nLayer, rnn.nDir, rnn.nBias, rnn.dhc),
            biasType, KuDNN::Layout::ABCD};
        const std::vector<KuDNN::TensorInfo> weightInfos = {
            {KuDNN::Shape(rnn.nLayer, rnn.nDir, rnn.slc, rnn.nGates, rnn.dhc), biasType, KuDNN::Layout::ABCDE},
        };

        int dstLayerRate = rnn.execDir == KuDNN::Detail::ExecutionDirectionT::BI_CONCAT ? 2 : 1;
        void *src[3] = {nullptr, nullptr, nullptr};
        void *weight[4] = {nullptr, nullptr, nullptr, nullptr};
        void *dst[3] = {nullptr, nullptr, nullptr};
        void *bias = nullptr;
        try {
            KuDNN::RnnFWD rnnFWD(srcInfos, weightInfos, dstInfos, biasInfo, rnn, KuDNN::RnnAlgorithm::RNN, actfKind);
            src[0] = calloc(1, rnn.nIter *  rnn.mb *  rnn.slc * srcType.GetSize());
            src[1] = calloc(1, rnn.nLayer * rnn.nDir *  rnn.mb *  rnn.sic * srcType.GetSize());
            weight[0] = calloc(1, rnn.nLayer * rnn.nDir * rnn.slc * rnn.nGates * rnn.dhc * srcType.GetSize());
            weight[1] = calloc(1, rnn.nLayer * rnn.nDir * rnn.sic * rnn.nGates * rnn.dhc * srcType.GetSize());
            dst[0] = calloc(1, rnn.nIter *  rnn.mb *  rnn.dlc * dstLayerRate * srcType.GetSize());
            dst[1] = calloc(1, rnn.nLayer * rnn.nDir *  rnn.mb *  rnn.dic * srcType.GetSize());
            bias = calloc(1, rnn.nLayer * rnn.nDir * rnn.nBias * rnn.dhc * sizeof(float));
            if (CheckAnyNullPtr(src[0], src[1], weight[0], weight[1], dst[0], dst[1], bias)) {
                goto FREE_SPACE;
            }

            rnnFWD.Run((const void **)src, (const void **)weight, dst, (const void *)bias, nullptr, nullptr);
        } catch (const KuDNN::Service::Unsupported& e) {
            printf("Unsupported data type! %s\n" , e.what());
        }
    FREE_SPACE:
        ReleaseVoidArray(src, 3);
        ReleaseVoidArray(weight, 4);
        ReleaseVoidArray(dst, 3);
        if (bias) {
            free(bias);
            bias = nullptr;
        }

    }
    DT_FUZZ_END()
}

static void DtFuzzLstmForward(int counts)
{
    char funcName[] = "DtFuzzLSTMForward";
    DT_FUZZ_START(0, counts, funcName, 0)
    {
        int eleIndex = 0;
        bool withPeepholeIndex = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], 0, 0, g_boolMap.size() - 1);
        bool withProjectionIndex = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], 0, 0, g_boolMap.size() - 1);
        bool peephole = g_boolMap[withPeepholeIndex];
        bool projection = g_boolMap[withProjectionIndex];

        int srcTypeIndex = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], 0, 0, g_precisionMap.size() - 1);
        int biasTypeIndex = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], 0, 0, g_precisionMap.size() - 1);
        int dirIndex = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], 0, 0, g_directionFuncMap.size() - 1);
        int gatesLinearIndex = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], 0, 0, g_boolMap.size() - 1);

        KuDNN::Element::Type srcType(g_precisionMap[srcTypeIndex].second);
        KuDNN::Element::Type biasType(g_precisionMap[biasTypeIndex].second);
        KuDNN::RnnCfg rnn;
        rnn.nLayer = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], 1, 1, 11);
        rnn.nIter = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], 1, 1, 11);
        rnn.mb = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++],
            g_dimRangeInitVal, g_dimRangeLowerBound, g_dimRangeUpperBound);
        rnn.dhc = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++],
            g_dimRangeInitVal, g_dimRangeLowerBound, g_dimRangeUpperBound);
        rnn.sic = rnn.dhc;
        rnn.slc = rnn.sic;
        rnn.dic = rnn.dhc;
        rnn.dlc = rnn.dic;

        rnn.gatesFuncLinearMode = g_boolMap[gatesLinearIndex];
        rnn.execDir = g_directionFuncMap[dirIndex].second;
        rnn.nGates = 4;

        SetRnnFwd(rnn, srcType);

        const std::vector<KuDNN::TensorInfo> srcInfos = {
            {KuDNN::Shape(rnn.nIter, rnn.mb, rnn.slc), srcType, KuDNN::Layout::ABC,
                KuDNN::Shape(rnn.mb * rnn.slc, g_dimRangeUpperBound, g_dimRangeUpperBound)},
        };
        const std::vector<KuDNN::TensorInfo> dstInfos = {
            {KuDNN::Shape(rnn.nIter, rnn.mb, rnn.dlc), srcType, KuDNN::Layout::ABC,
                KuDNN::Shape(rnn.mb * rnn.dlc, g_dimRangeUpperBound, g_dimRangeUpperBound)},
        };
        KuDNN::TensorInfo biasInfo = {KuDNN::Shape(rnn.nLayer, rnn.nDir, rnn.nBias, rnn.dhc),
            biasType, KuDNN::Layout::ABCD};
        const std::vector<KuDNN::TensorInfo> weightInfos = {
            {KuDNN::Shape(rnn.nLayer, rnn.nDir, rnn.slc, rnn.nGates, rnn.dhc), biasType, KuDNN::Layout::ABCDE},
        };

        int dstLayerRate = rnn.execDir == KuDNN::Detail::ExecutionDirectionT::BI_CONCAT ? 2 : 1;
        void *src[3] = {nullptr, nullptr, nullptr};
        void *weight[4] = {nullptr, nullptr, nullptr, nullptr};
        void *dst[3] = {nullptr, nullptr, nullptr};
        void *bias = nullptr;
        try {
            KuDNN::RnnFWD rnnFWD(srcInfos, weightInfos, dstInfos, biasInfo, rnn,
                                KuDNN::RnnAlgorithm::LSTM, KuDNN::ActivateFunctionRNN::UNIMPLEMENTED);
            src[0] = calloc(1, rnn.nIter *  rnn.mb *  rnn.slc * srcType.GetSize());
            src[1] = calloc(1, rnn.nLayer * rnn.nDir *  rnn.mb *  rnn.sic * srcType.GetSize());
            src[2] = calloc(1, rnn.nLayer * rnn.nDir *  rnn.mb *  rnn.dhc * sizeof(float));
            weight[0] = calloc(1, rnn.nLayer * rnn.nDir * rnn.slc * rnn.nGates * rnn.dhc * srcType.GetSize());
            weight[1] = calloc(1, rnn.nLayer * rnn.nDir * rnn.sic * rnn.nGates * rnn.dhc * srcType.GetSize());
            weight[2] = peephole ? calloc(1, rnn.nLayer * rnn.nDir * 0x3 * rnn.dhc * sizeof(float)) : nullptr;
            weight[3] = projection ? calloc(1, rnn.nLayer * rnn.nDir * rnn.nGates * rnn.weightsProjectionLd * rnn.dhc * sizeof(float)) : nullptr;
            dst[0] = calloc(1, rnn.nIter *  rnn.mb *  rnn.dlc * dstLayerRate * srcType.GetSize());
            dst[1] = calloc(1, rnn.nLayer * rnn.nDir *  rnn.mb *  rnn.dic * srcType.GetSize());
            dst[2] = calloc(1, rnn.nLayer * rnn.nDir *  rnn.mb *  rnn.dhc * sizeof(float));
            bias = calloc(1, rnn.nLayer * rnn.nDir * rnn.nBias * rnn.dhc * sizeof(float));
            if (CheckAnyNullPtr(src[0], src[1], weight[0], weight[1], dst[0], dst[1], bias)) {
                goto FREE_SPACE;
            }

            rnnFWD.Run((const void **)src, (const void **)weight, dst, (const void *)bias, nullptr, nullptr);
        } catch (const KuDNN::Service::Unsupported& e) {
            printf("Unsupported data type! %s\n" , e.what());
        }
    FREE_SPACE:
        ReleaseVoidArray(src, 3);
        ReleaseVoidArray(weight, 4);
        ReleaseVoidArray(dst, 3);
        if (bias) {
            free(bias);
            bias = nullptr;
        }

    }
    DT_FUZZ_END()
}

static void DtFuzzGruForward(int counts)
{
    char funcName[] = "DtFuzzGruForward";
    DT_FUZZ_START(0, counts, funcName, 0)
    {
        int eleIndex = 0;
        int srcTypeIndex = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], 0, 0, g_precisionMap.size() - 1);
        int biasTypeIndex = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], 0, 0, g_precisionMap.size() - 1);

        int dirIndex = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], 0, 0, g_directionFuncMap.size() - 1);
        int gatesLinearIndex = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], 0, 0, g_boolMap.size() - 1);

        KuDNN::Element::Type srcType(g_precisionMap[srcTypeIndex].second);
        KuDNN::Element::Type biasType(g_precisionMap[biasTypeIndex].second);
        KuDNN::ActivateFunctionRNN actfKind = g_activationFuncMap[INDEX3].second;
        KuDNN::RnnCfg rnn;
        rnn.nLayer = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], 1, 1, 11);
        rnn.nIter = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], 1, 1, 11);
        rnn.mb = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++],
            g_dimRangeInitVal, g_dimRangeLowerBound, g_dimRangeUpperBound);
        rnn.dhc = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++],
            g_dimRangeInitVal, g_dimRangeLowerBound, g_dimRangeUpperBound);
        rnn.sic = rnn.dhc;
        rnn.slc = rnn.sic;
        rnn.dic = rnn.dhc;
        rnn.dlc = rnn.dic;
        rnn.gatesFuncLinearMode = g_boolMap[gatesLinearIndex];
        if (rnn.nLayer == 1) {
            rnn.slc = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++],
                g_dimRangeInitVal, g_dimRangeLowerBound, g_dimRangeUpperBound);
        }

        rnn.execDir = g_directionFuncMap[dirIndex].second;
        rnn.nGates = 0x3;
        rnn.isLbr = 0;

        SetRnnFwd(rnn, srcType);

        const std::vector<KuDNN::TensorInfo> srcInfos = {
            {KuDNN::Shape(rnn.nIter, rnn.mb, rnn.slc), srcType, KuDNN::Layout::ABC,
                KuDNN::Shape(rnn.mb * rnn.slc, g_dimRangeUpperBound, g_dimRangeUpperBound)},
        };
        const std::vector<KuDNN::TensorInfo> dstInfos = {
            {KuDNN::Shape(rnn.nIter, rnn.mb, rnn.dlc), srcType, KuDNN::Layout::ABC,
                KuDNN::Shape(rnn.mb * rnn.dlc, g_dimRangeUpperBound, g_dimRangeUpperBound)},
        };
        KuDNN::TensorInfo biasInfo = {KuDNN::Shape(rnn.nLayer, rnn.nDir, rnn.nBias, rnn.dhc),
            biasType, KuDNN::Layout::ABCD};
        const std::vector<KuDNN::TensorInfo> weightInfos = {
            {KuDNN::Shape(rnn.nLayer, rnn.nDir, rnn.slc, rnn.nGates, rnn.dhc), biasType, KuDNN::Layout::ABCDE},
        };

        int dstLayerRate = rnn.execDir == KuDNN::Detail::ExecutionDirectionT::BI_CONCAT ? 2 : 1;
        void *src[3] = {nullptr, nullptr, nullptr};
        void *weight[4] = {nullptr, nullptr, nullptr, nullptr};
        void *dst[3] = {nullptr, nullptr, nullptr};
        void *bias = nullptr;
        try {
            KuDNN::RnnFWD rnnFWD(srcInfos, weightInfos, dstInfos, biasInfo, rnn, KuDNN::RnnAlgorithm::GRU, actfKind);
            src[0] = calloc(1, rnn.nIter *  rnn.mb *  rnn.slc * srcType.GetSize());
            src[1] = calloc(1, rnn.nLayer * rnn.nDir *  rnn.mb *  rnn.sic * srcType.GetSize());
            weight[0] = calloc(1, rnn.nLayer * rnn.nDir * rnn.slc * rnn.nGates * rnn.dhc * srcType.GetSize());
            weight[1] = calloc(1, rnn.nLayer * rnn.nDir * rnn.sic * rnn.nGates * rnn.dhc * srcType.GetSize());
            dst[0] = calloc(1, rnn.nIter *  rnn.mb *  rnn.dlc * dstLayerRate * srcType.GetSize());
            dst[1] = calloc(1, rnn.nLayer * rnn.nDir *  rnn.mb *  rnn.dic * srcType.GetSize());
            bias = calloc(1, rnn.nLayer * rnn.nDir * rnn.nBias * rnn.dhc * sizeof(float));
            if (CheckAnyNullPtr(src[0], src[1], weight[0], weight[1], dst[0], dst[1], bias)) {
                goto FREE_SPACE;
            }
            rnnFWD.Run((const void **)src, (const void **)weight, dst, (const void *)bias, nullptr, nullptr);
        } catch (const KuDNN::Service::Unsupported& e) {
            printf("Unsupported data type! %s\n" , e.what());
        }
    FREE_SPACE:
        ReleaseVoidArray(src, 3);
        ReleaseVoidArray(weight, 4);
        ReleaseVoidArray(dst, 3);
        if (bias) {
            free(bias);
            bias = nullptr;
        }

    }
    DT_FUZZ_END()
}

static void DtFuzzAuGruForward(int counts)
{
    char funcName[] = "DtFuzzAuGruForward";
    DT_FUZZ_START(0, counts, funcName, 0)
    {
        int eleIndex = 0;
        int srcTypeIndex = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], 0, 0, g_precisionMap.size() - 1);
        int biasTypeIndex = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], 0, 0, g_precisionMap.size() - 1);
        int gatesLinearIndex = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], 0, 0, g_boolMap.size() - 1);

        KuDNN::Element::Type srcType(g_precisionMap[srcTypeIndex].second);
        KuDNN::Element::Type biasType(g_precisionMap[biasTypeIndex].second);
        KuDNN::ActivateFunctionRNN actfKind = g_activationFuncMap[INDEX3].second;
        KuDNN::RnnCfg rnn;

        rnn.nLayer = 1;
        rnn.nIter = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], 1, 1, 11);
        rnn.mb = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++],
            g_dimRangeInitVal, g_dimRangeLowerBound, g_dimRangeUpperBound);
        rnn.dhc = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++],
            g_dimRangeInitVal, g_dimRangeLowerBound, g_dimRangeUpperBound);
        rnn.sic = rnn.dhc;
        rnn.slc = rnn.sic;
        rnn.dic = rnn.dhc;
        rnn.dlc = rnn.dic;
        rnn.gatesFuncLinearMode = g_boolMap[gatesLinearIndex];

        rnn.execDir = g_directionFuncMap[INDEX0].second;
        rnn.nGates = 0x3;
        rnn.isLbr = 0;

        SetRnnFwd(rnn, srcType);

        const std::vector<KuDNN::TensorInfo> srcInfos = {
            {KuDNN::Shape(rnn.nIter, rnn.mb, rnn.slc), srcType, KuDNN::Layout::ABC,
                KuDNN::Shape(rnn.mb * rnn.slc, g_dimRangeUpperBound, g_dimRangeUpperBound)},
        };
        const std::vector<KuDNN::TensorInfo> dstInfos = {
            {KuDNN::Shape(rnn.nIter, rnn.mb, rnn.dlc), srcType, KuDNN::Layout::ABC,
                KuDNN::Shape(rnn.mb * rnn.dlc, g_dimRangeUpperBound, g_dimRangeUpperBound)},
        };
        KuDNN::TensorInfo biasInfo = {KuDNN::Shape(rnn.nLayer, rnn.nDir, rnn.nBias, rnn.dhc),
            biasType, KuDNN::Layout::ABCD};
        const std::vector<KuDNN::TensorInfo> weightInfos = {
            {KuDNN::Shape(rnn.nLayer, rnn.nDir, rnn.slc, rnn.nGates, rnn.dhc), biasType, KuDNN::Layout::ABCDE},
        };

        int dstLayerRate = 1;
        void *src[3] = {nullptr, nullptr, nullptr};
        void *weight[4] = {nullptr, nullptr, nullptr, nullptr};
        void *dst[3] = {nullptr, nullptr, nullptr};
        void *bias = nullptr;
        void *augruAttentions = nullptr;
        try {
            KuDNN::RnnFWD rnnFWD(srcInfos, weightInfos, dstInfos, biasInfo, rnn, KuDNN::RnnAlgorithm::AUGRU, actfKind);
            src[0] = calloc(1, rnn.nIter *  rnn.mb *  rnn.slc * srcType.GetSize());
            src[1] = calloc(1, rnn.nLayer * rnn.nDir *  rnn.mb *  rnn.sic * srcType.GetSize());
            weight[0] = calloc(1, rnn.nLayer * rnn.nDir * rnn.slc * rnn.nGates * rnn.dhc * srcType.GetSize());
            weight[1] = calloc(1, rnn.nLayer * rnn.nDir * rnn.sic * rnn.nGates * rnn.dhc * srcType.GetSize());
            dst[0] = calloc(1, rnn.nIter *  rnn.mb *  rnn.dlc * dstLayerRate * srcType.GetSize());
            dst[1] = calloc(1, rnn.nLayer * rnn.nDir *  rnn.mb *  rnn.dic * srcType.GetSize());
            bias = calloc(1, rnn.nLayer * rnn.nDir * rnn.nBias * rnn.dhc * sizeof(float));
            augruAttentions = calloc(1, rnn.nIter * rnn.mb * rnn.dhc * srcType.GetSize());
            if (CheckAnyNullPtr(src[0], src[1], weight[0], weight[1], dst[0], dst[1], bias, augruAttentions)) {
                goto FREE_SPACE;
            }

            rnnFWD.Run((const void **)src, (const void **)weight, dst, (const void *)bias, nullptr, augruAttentions);
        } catch (const KuDNN::Service::Unsupported& e) {
            printf("Unsupported data type! %s\n" , e.what());
        }
    FREE_SPACE:
        ReleaseVoidArray(src, 3);
        ReleaseVoidArray(weight, 4);
        ReleaseVoidArray(dst, 3);
        if (bias) {
            free(bias);
            bias = nullptr;
        }
        if (augruAttentions) {
            free(augruAttentions);
            augruAttentions = nullptr;
        }

    }
    DT_FUZZ_END()
}

static void DtFuzzLbrGruForward(int counts)
{
    char funcName[] = "DtFuzzLbrGruForward";
    DT_FUZZ_START(0, counts, funcName, 0)
    {
        int eleIndex = 0;
        int srcTypeIndex = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], 0, 0, g_precisionMap.size() - 1);
        int biasTypeIndex = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], 0, 0, g_precisionMap.size() - 1);

        int dirIndex = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], 0, 0, g_directionFuncMap.size() - 1);
        int gatesLinearIndex = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], 0, 0, g_boolMap.size() - 1);

        KuDNN::Element::Type srcType(g_precisionMap[srcTypeIndex].second);
        KuDNN::Element::Type biasType(g_precisionMap[biasTypeIndex].second);
        KuDNN::ActivateFunctionRNN actfKind = g_activationFuncMap[INDEX3].second;
        KuDNN::RnnCfg rnn;
        rnn.nLayer = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], 1, 1, 11);
        rnn.nIter = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], 1, 1, 11);
        rnn.mb = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++],
            g_dimRangeInitVal, g_dimRangeLowerBound, g_dimRangeUpperBound);
        rnn.dhc = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++],
            g_dimRangeInitVal, g_dimRangeLowerBound, g_dimRangeUpperBound);
        rnn.sic = rnn.dhc;
        rnn.slc = rnn.sic;
        rnn.dic = rnn.dhc;
        rnn.dlc = rnn.dic;
        rnn.gatesFuncLinearMode = g_boolMap[gatesLinearIndex];
        if (rnn.nLayer == 1) {
            rnn.slc = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++],
                g_dimRangeInitVal, g_dimRangeLowerBound, g_dimRangeUpperBound);
        }

        rnn.execDir = g_directionFuncMap[dirIndex].second;
        rnn.nGates = 0x3;
        rnn.isLbr = 1;

        SetRnnFwd(rnn, srcType);

        const std::vector<KuDNN::TensorInfo> srcInfos = {
            {KuDNN::Shape(rnn.nIter, rnn.mb, rnn.slc), srcType, KuDNN::Layout::ABC,
                KuDNN::Shape(rnn.mb * rnn.slc, g_dimRangeUpperBound, g_dimRangeUpperBound)},
        };
        const std::vector<KuDNN::TensorInfo> dstInfos = {
            {KuDNN::Shape(rnn.nIter, rnn.mb, rnn.dlc), srcType, KuDNN::Layout::ABC,
                KuDNN::Shape(rnn.mb * rnn.dlc, g_dimRangeUpperBound, g_dimRangeUpperBound)},
        };
        KuDNN::TensorInfo biasInfo = {KuDNN::Shape(rnn.nLayer, rnn.nDir, rnn.nBias, rnn.dhc),
            biasType, KuDNN::Layout::ABCD};
        const std::vector<KuDNN::TensorInfo> weightInfos = {
            {KuDNN::Shape(rnn.nLayer, rnn.nDir, rnn.slc, rnn.nGates, rnn.dhc), biasType, KuDNN::Layout::ABCDE},
        };

        int dstLayerRate = rnn.execDir == KuDNN::Detail::ExecutionDirectionT::BI_CONCAT ? 2 : 1;
        void *src[3] = {nullptr, nullptr, nullptr};
        void *weight[4] = {nullptr, nullptr, nullptr, nullptr};
        void *dst[3] = {nullptr, nullptr, nullptr};
        void *bias = nullptr;
        try {
            KuDNN::RnnFWD rnnFWD(srcInfos, weightInfos, dstInfos, biasInfo, rnn, KuDNN::RnnAlgorithm::LBR_GRU, actfKind);
            src[0] = calloc(1, rnn.nIter *  rnn.mb *  rnn.slc * srcType.GetSize());
            src[1] = calloc(1, rnn.nLayer * rnn.nDir *  rnn.mb *  rnn.sic * srcType.GetSize());
            weight[0] = calloc(1, rnn.nLayer * rnn.nDir * rnn.slc * rnn.nGates * rnn.dhc * srcType.GetSize());
            weight[1] = calloc(1, rnn.nLayer * rnn.nDir * rnn.sic * rnn.nGates * rnn.dhc * srcType.GetSize());
            dst[0] = calloc(1, rnn.nIter *  rnn.mb *  rnn.dlc * dstLayerRate * srcType.GetSize());
            dst[1] = calloc(1, rnn.nLayer * rnn.nDir *  rnn.mb *  rnn.dic * srcType.GetSize());
            bias = calloc(1, rnn.nLayer * rnn.nDir * rnn.nBias * rnn.dhc * sizeof(float));
            if (CheckAnyNullPtr(src[0], src[1], weight[0], weight[1], dst[0], dst[1], bias)) {
                goto FREE_SPACE;
            }
            rnnFWD.Run((const void **)src, (const void **)weight, dst, (const void *)bias, nullptr, nullptr);
        } catch (const KuDNN::Service::Unsupported& e) {
            printf("Unsupported data type! %s\n" , e.what());
        }
    FREE_SPACE:
        ReleaseVoidArray(src, 3);
        ReleaseVoidArray(weight, 4);
        ReleaseVoidArray(dst, 3);
        if (bias) {
            free(bias);
            bias = nullptr;
        }

    }
    DT_FUZZ_END()
}

static void DtFuzzLbrAuGruForward(int counts)
{
    char funcName[] = "DtFuzzLbrAuGruForward";
    DT_FUZZ_START(0, counts, funcName, 0)
    {
        int eleIndex = 0;
        int srcTypeIndex = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], 0, 0, g_precisionMap.size() - 1);
        int biasTypeIndex = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], 0, 0, g_precisionMap.size() - 1);
        int gatesLinearIndex = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], 0, 0, g_boolMap.size() - 1);

        KuDNN::Element::Type srcType(g_precisionMap[srcTypeIndex].second);
        KuDNN::Element::Type biasType(g_precisionMap[biasTypeIndex].second);
        KuDNN::ActivateFunctionRNN actfKind = g_activationFuncMap[INDEX3].second;
        KuDNN::RnnCfg rnn;

        rnn.nLayer = 1;
        rnn.nIter = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], 1, 1, 11);
        rnn.mb = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++],
            g_dimRangeInitVal, g_dimRangeLowerBound, g_dimRangeUpperBound);
        rnn.dhc = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++],
            g_dimRangeInitVal, g_dimRangeLowerBound, g_dimRangeUpperBound);
        rnn.sic = rnn.dhc;
        rnn.slc = rnn.sic;
        rnn.dic = rnn.dhc;
        rnn.dlc = rnn.dic;

        rnn.execDir = g_directionFuncMap[INDEX0].second;
        rnn.nGates = 0x3;
        rnn.isLbr = 1;
        rnn.gatesFuncLinearMode = g_boolMap[gatesLinearIndex];

        SetRnnFwd(rnn, srcType);

        const std::vector<KuDNN::TensorInfo> srcInfos = {
            {KuDNN::Shape(rnn.nIter, rnn.mb, rnn.slc), srcType, KuDNN::Layout::ABC,
                KuDNN::Shape(rnn.mb * rnn.slc, g_dimRangeUpperBound, g_dimRangeUpperBound)},
        };
        const std::vector<KuDNN::TensorInfo> dstInfos = {
            {KuDNN::Shape(rnn.nIter, rnn.mb, rnn.dlc), srcType, KuDNN::Layout::ABC,
                KuDNN::Shape(rnn.mb * rnn.dlc, g_dimRangeUpperBound, g_dimRangeUpperBound)},
        };
        KuDNN::TensorInfo biasInfo = {KuDNN::Shape(rnn.nLayer, rnn.nDir, rnn.nBias, rnn.dhc),
            biasType, KuDNN::Layout::ABCD};
        const std::vector<KuDNN::TensorInfo> weightInfos = {
            {KuDNN::Shape(rnn.nLayer, rnn.nDir, rnn.slc, rnn.nGates, rnn.dhc), biasType, KuDNN::Layout::ABCDE},
        };

        int dstLayerRate = 1;
        void *src[3] = {nullptr, nullptr, nullptr};
        void *weight[4] = {nullptr, nullptr, nullptr, nullptr};
        void *dst[3] = {nullptr, nullptr, nullptr};
        void *bias = nullptr;
        void *augruAttentions = nullptr;
        try {
            KuDNN::RnnFWD rnnFWD(srcInfos, weightInfos, dstInfos, biasInfo, rnn, KuDNN::RnnAlgorithm::LBR_AUGRU, actfKind);
            src[0] = calloc(1, rnn.nIter *  rnn.mb *  rnn.slc * srcType.GetSize());
            src[1] = calloc(1, rnn.nLayer * rnn.nDir *  rnn.mb *  rnn.sic * srcType.GetSize());
            weight[0] = calloc(1, rnn.nLayer * rnn.nDir * rnn.slc * rnn.nGates * rnn.dhc * srcType.GetSize());
            weight[1] = calloc(1, rnn.nLayer * rnn.nDir * rnn.sic * rnn.nGates * rnn.dhc * srcType.GetSize());
            dst[0] = calloc(1, rnn.nIter *  rnn.mb *  rnn.dlc * dstLayerRate * srcType.GetSize());
            dst[1] = calloc(1, rnn.nLayer * rnn.nDir *  rnn.mb *  rnn.dic * srcType.GetSize());
            bias = calloc(1, rnn.nLayer * rnn.nDir * rnn.nBias * rnn.dhc * sizeof(float));
            augruAttentions = calloc(1, rnn.nIter * rnn.mb * rnn.dhc * srcType.GetSize());
            if (CheckAnyNullPtr(src[0], src[1], weight[0], weight[1], dst[0], dst[1], bias, augruAttentions)) {
                goto FREE_SPACE;
            }

            rnnFWD.Run((const void **)src, (const void **)weight, dst, (const void *)bias, nullptr, augruAttentions);
        } catch (const KuDNN::Service::Unsupported& e) {
            printf("Unsupported data type! %s\n" , e.what());
        }
    FREE_SPACE:
        ReleaseVoidArray(src, 3);
        ReleaseVoidArray(weight, 4);
        ReleaseVoidArray(dst, 3);
        if (bias) {
            free(bias);
            bias = nullptr;
        }
        if (augruAttentions) {
            free(augruAttentions);
            augruAttentions = nullptr;
        }

    }
    DT_FUZZ_END()
}

static void DtFuzzForwardSpecialTest(int counts)
{
    char funcName[] = "DtFuzzForwardSpecialTest";
    DT_FUZZ_START(0, counts, funcName, 0)
    {
        int eleIndex = 0;
        int srcTypeIndex = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], 0, 0, g_precisionMap.size() - 1);
        int biasTypeIndex = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], 0, 0, g_precisionMap.size() - 1);
        int dirIndex = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], 0, 0, g_directionFuncMap.size() - 1);
        int rnnFuncIndex = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], 0, 0, g_rnnFuncMap.size() - 1);
        int actFuncIndex = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], 0, 0, g_activationFuncMap.size() - 1);

        KuDNN::Element::Type srcType(g_precisionMap[srcTypeIndex].second);
        KuDNN::Element::Type biasType(g_precisionMap[biasTypeIndex].second);
        KuDNN::RnnAlgorithm funcT = g_rnnFuncMap[rnnFuncIndex].second;
        KuDNN::ActivateFunctionRNN actfKind = g_activationFuncMap[actFuncIndex].second;
        KuDNN::RnnCfg rnn;
        rnn.nLayer = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], 1, 1, 11);
        rnn.nIter = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], 1, 1, 11);
        rnn.mb = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++],
            g_dimRangeInitVal, g_dimRangeLowerBound, g_dimRangeUpperBound);
        rnn.dhc = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++],
            g_dimRangeInitVal, g_dimRangeLowerBound, g_dimRangeUpperBound);
        rnn.sic = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++],
            g_dimRangeInitVal, g_dimRangeLowerBound, g_dimRangeUpperBound);
        rnn.slc = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++],
            g_dimRangeInitVal, g_dimRangeLowerBound, g_dimRangeUpperBound);
        rnn.dic = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++],
            g_dimRangeInitVal, g_dimRangeLowerBound, g_dimRangeUpperBound);
        rnn.dlc = rnn.dic;

        rnn.execDir = g_directionFuncMap[dirIndex].second;
        rnn.nGates = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++],
            g_dimRangeInitVal, g_dimRangeLowerBound, g_dimRangeUpperBound);
        SetRnnFwd(rnn, srcType);

        const std::vector<KuDNN::TensorInfo> srcInfos = {
            {KuDNN::Shape(rnn.nIter, rnn.mb, rnn.slc), srcType, KuDNN::Layout::ABC,
                KuDNN::Shape(rnn.mb * rnn.slc, g_dimRangeUpperBound, g_dimRangeUpperBound)},
        };
        const std::vector<KuDNN::TensorInfo> dstInfos = {
            {KuDNN::Shape(rnn.nIter, rnn.mb, rnn.dlc), srcType, KuDNN::Layout::ABC,
                KuDNN::Shape(rnn.mb * rnn.dlc, g_dimRangeUpperBound, g_dimRangeUpperBound)},
        };
        KuDNN::TensorInfo biasInfo = {KuDNN::Shape(rnn.nLayer, rnn.nDir, rnn.nBias, rnn.dhc),
            srcType, KuDNN::Layout::ABCD};
        const std::vector<KuDNN::TensorInfo> weightInfos = {
            {KuDNN::Shape(rnn.nLayer, rnn.nDir, rnn.slc, rnn.nGates, rnn.dhc), biasType, KuDNN::Layout::ABCDE},
        };

        int dstLayerRate = rnn.execDir == KuDNN::Detail::ExecutionDirectionT::BI_CONCAT ? 2 : 1;
        void *src[3] = {nullptr, nullptr, nullptr};
        void *weight[4] = {nullptr, nullptr, nullptr, nullptr};
        void *dst[3] = {nullptr, nullptr, nullptr};
        void *bias = nullptr;
        void *augruAttentions = nullptr;

        try {
            KuDNN::RnnFWD rnn1(srcInfos, weightInfos, dstInfos, biasInfo, rnn, funcT, actfKind);
            KuDNN::RnnFWD rnn2(rnn1);
            rnn2 = rnn1;
            KuDNN::RnnFWD rnn3(rnn1);
            rnn3 = std::move(rnn1);
            KuDNN::RnnFWD rnn4(std::move(rnn2));
            KuDNN::RnnFWD rnnFWD(srcInfos, weightInfos, dstInfos, biasInfo, rnn, funcT, actfKind);
            src[0] = CreateRawPointer(*(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], 0, 0, 1),
                rnn.nIter *  rnn.mb *  rnn.slc * srcType.GetSize());
            src[1] = CreateRawPointer(*(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], 0, 0, 1),
                rnn.nLayer * rnn.nDir *  rnn.mb *  rnn.sic * srcType.GetSize());
            src[2] = CreateRawPointer(*(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], 0, 0, 1),
                rnn.nLayer * rnn.nDir *  rnn.mb *  rnn.dhc * srcType.GetSize());
            weight[0] = CreateRawPointer(*(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], 0, 0, 1),
                rnn.nLayer * rnn.nDir * rnn.slc * rnn.nGates * rnn.dhc * srcType.GetSize());
            weight[1] = CreateRawPointer(*(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], 0, 0, 1),
                rnn.nLayer * rnn.nDir * rnn.sic * rnn.nGates * rnn.dhc * srcType.GetSize());
            weight[2] = CreateRawPointer(*(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], 0, 0, 1),
                rnn.nLayer * rnn.nDir * 0x3 * rnn.dhc * sizeof(float));
            weight[3] = CreateRawPointer(*(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], 0, 0, 1),
                rnn.nLayer * rnn.nDir * rnn.nGates * rnn.weightsProjectionLd * rnn.dhc * sizeof(float));
            dst[0] = CreateRawPointer(*(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], 0, 0, 1),
                rnn.nIter *  rnn.mb *  rnn.dlc * dstLayerRate * srcType.GetSize());
            dst[1] = CreateRawPointer(*(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], 0, 0, 1),
                rnn.nLayer * rnn.nDir *  rnn.mb *  rnn.dic * srcType.GetSize());
            dst[2] = CreateRawPointer(*(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], 0, 0, 1),
                rnn.nLayer * rnn.nDir *  rnn.mb *  rnn.dhc * srcType.GetSize());
            bias = CreateRawPointer(*(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], 0, 0, 1),
                rnn.nLayer * rnn.nDir * rnn.nBias * g_dimRangeUpperBound * sizeof(float));
            augruAttentions = CreateRawPointer(*(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], 0, 0, 1),
                rnn.nIter * rnn.mb * rnn.dhc * srcType.GetSize());

            rnnFWD.Run((const void **)src, (const void **)weight, dst, (const void *)bias, nullptr, augruAttentions);
        } catch (const KuDNN::Service::Unsupported& e) {
            printf("Unsupported data type! %s\n" , e.what());
        } catch (const KuDNN::Service::LogicError& e) {
            printf("Input Error! %s\n" , e.what());
        }
        ReleaseVoidArray(src, 3);
        ReleaseVoidArray(weight, 4);
        ReleaseVoidArray(dst, 3);
        if (bias) {
            free(bias);
            bias = nullptr;
        }
        if (augruAttentions) {
            free(augruAttentions);
            augruAttentions = nullptr;
        }

    }
    DT_FUZZ_END()
}

static void SetRnnBwd(KuDNN::RnnCfg &rnn, KuDNN::Element::Type &srcType, KuDNN::Element::Type &biasType)
{
    rnn.nDir = ((rnn.execDir == KuDNN::Detail::ExecutionDirectionT::L2R) ||
        (rnn.execDir == KuDNN::Detail::ExecutionDirectionT::R2L)) ? 1 : 2;
    rnn.nBias = rnn.nGates + rnn.isLbr;
    rnn.weightsLayerLd = rnn.slc;
    rnn.weightsIterLd = rnn.sic;

    rnn.scratchGatesLd = std::max(rnn.dlc, rnn.nGates * rnn.dhc);
    rnn.srcLayerLd = rnn.slc;
    rnn.srcIterLd = rnn.sic;
    rnn.srcIterCLd = 1;
    rnn.dstLayerLd = rnn.dlc;
    rnn.dstIterLd = rnn.dic;
    rnn.dstIterCLd = 1;

    rnn.wsStatesLayerLd = std::max(rnn.dlc, std::max(rnn.slc, rnn.sic));
    rnn.wsStatesLayerNld = rnn.mb;
    rnn.wsStatesIterLd = rnn.wsStatesLayerLd;
    rnn.scratchGatesNld = rnn.mb;
    rnn.weightsProjectionLd = rnn.dhc;
    rnn.wsGatesLd = rnn.dhc * rnn.nGates;
    rnn.wsGatesNld = rnn.mb;
    rnn.alpha = 1;

    rnn.wsDiffStatesLayerLd = std::max(std::max(rnn.slc, rnn.dic), std::max(rnn.dhc, rnn.sic));
    rnn.wsDiffStatesIterLd = std::max(std::max(rnn.slc, rnn.dic), std::max(rnn.dhc, rnn.sic));
    rnn.wsDiffStatesIterCLd = rnn.dhc;
    rnn.diffWeightsLayerLd = rnn.nGates * rnn.dhc;
    rnn.diffWeightsIterLd = rnn.nGates * rnn.dhc;
    rnn.scratchDiffHtLd = rnn.wsDiffStatesLayerLd;
    rnn.wsHtLd = rnn.wsDiffStatesLayerLd;
    rnn.wsHtNld = rnn.mb;
    rnn.diffWeightsProjectionLd = rnn.dic;
    rnn.diffWeightsProjectionNld = rnn.dhc;

    rnn.wsDiffStatesLayerSize = (size_t)(rnn.nLayer + 1) * rnn.nDir * (rnn.nIter + 1) *
        rnn.mb * rnn.wsDiffStatesLayerLd * sizeof(float);
    rnn.wsDiffStatesIterSize = (size_t)(rnn.nLayer + 1) * rnn.nDir * (rnn.nIter + 1) *
        rnn.mb * rnn.wsDiffStatesIterLd * sizeof(float);
    rnn.wsDiffStatesIterCSize = (size_t)(rnn.nLayer + 1) * rnn.nDir * (rnn.nIter + 1) *
        rnn.mb * rnn.wsDiffStatesIterCLd * sizeof(float);
    rnn.scratchGatesSize = rnn.nIter * rnn.scratchGatesNld * rnn.scratchGatesLd * srcType.GetSize();

    rnn.wsGatesSize = (size_t)rnn.nLayer * rnn.nDir * rnn.nIter * rnn.wsGatesNld * rnn.wsGatesLd * srcType.GetSize();
    rnn.wsHtSize = (size_t)rnn.nLayer * rnn.nDir * rnn.nIter * rnn.wsHtNld * rnn.wsHtLd * srcType.GetSize();
    rnn.wsStatesLayerSize = (size_t)(rnn.nLayer + 1) * rnn.nDir * (rnn.nIter + 1) *
        rnn.mb * rnn.wsStatesLayerLd * srcType.GetSize();
    rnn.wsStatesIterCSize = (size_t)(rnn.nLayer + 1) * rnn.nDir * (rnn.nIter + 1) *
        rnn.mb * rnn.dhc * biasType.GetSize();
    rnn.wsGridCompSize = (size_t)rnn.isLbr * rnn.nLayer * rnn.nDir * rnn.nIter * rnn.mb * rnn.dhc * sizeof(float);

    rnn.wsGatesOffset = 0;
    rnn.wsHtOffset = rnn.wsGatesOffset + rnn.wsGatesSize;
    rnn.wsStatesLayerOffset = rnn.wsHtOffset + rnn.wsHtSize;
    rnn.wsStatesIterCOffset = rnn.wsStatesLayerOffset + rnn.wsStatesLayerSize;
    rnn.wsGridCompOffset = rnn.wsStatesIterCOffset + rnn.wsStatesIterCSize;
    rnn.wsBiasOffset = rnn.wsGridCompOffset + rnn.wsGridCompSize;
}

static void DtFuzzRnnBackward(int counts)
{
    char funcName[] = "DtFuzzRnnBackward";
    DT_FUZZ_START(0, counts, funcName, 0)
    {
        int eleIndex = 0;
        int srcTypeIndex = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], 0, 0, g_precisionMap.size() - 1);
        int biasTypeIndex = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], 0, 0, g_precisionMap.size() - 1);
        int actFuncIndex = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], 0, 0, g_activationFuncMap.size() - 1);
        int dirIndex = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], 0, 0, g_directionFuncMap.size() - 1);

        KuDNN::Element::Type srcType(g_precisionMap[srcTypeIndex].second);
        KuDNN::Element::Type biasType(g_precisionMap[biasTypeIndex].second);
        KuDNN::ActivateFunctionRNN actfKind = g_activationFuncMap[actFuncIndex].second;
        KuDNN::RnnCfg rnn;
        rnn.nLayer = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], 1, 1, 11);
        rnn.nIter = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], 1, 1, 11);
        rnn.mb = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++],
            g_dimRangeInitVal, g_dimRangeLowerBound, g_dimRangeUpperBound);
        rnn.dhc = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++],
            g_dimRangeInitVal, g_dimRangeLowerBound, g_dimRangeUpperBound);
        rnn.sic = rnn.dhc;
        rnn.slc = rnn.sic;
        rnn.dic = rnn.dhc;
        rnn.dlc = rnn.dic;
        if (rnn.nLayer == 1) {
            rnn.slc = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++],
                g_dimRangeInitVal, g_dimRangeLowerBound, g_dimRangeUpperBound);
        }
        if (rnn.nIter == 1) {
            rnn.sic = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++],
                g_dimRangeInitVal, g_dimRangeLowerBound, g_dimRangeUpperBound);
        }

        rnn.execDir = g_directionFuncMap[dirIndex].second;
        rnn.nGates = 1;
        rnn.isLbr = 0;
        SetRnnBwd(rnn, srcType, biasType);
        rnn.scratchCellSize = 0; // The value varies according to the algorithm.

        const std::vector<KuDNN::TensorInfo> srcInfos = {
            {KuDNN::Shape(rnn.nIter, rnn.mb, rnn.slc), srcType, KuDNN::Layout::ABC,
                KuDNN::Shape(rnn.mb * rnn.slc, g_dimRangeUpperBound, g_dimRangeUpperBound)},
        };
        const std::vector<KuDNN::TensorInfo> weightInfos = {
            {KuDNN::Shape(rnn.nLayer, rnn.nDir, rnn.slc, rnn.nGates, rnn.dhc), biasType, KuDNN::Layout::ABCDE},
        };
        const std::vector<KuDNN::TensorInfo> dstInfos = {
            {KuDNN::Shape(rnn.nIter, rnn.mb, rnn.dlc), srcType, KuDNN::Layout::ABC,
                KuDNN::Shape(rnn.mb * rnn.dlc, g_dimRangeUpperBound, g_dimRangeUpperBound)},
        };
        KuDNN::TensorInfo biasInfo = {KuDNN::Shape(rnn.nLayer, rnn.nDir, rnn.nBias, rnn.dhc),
            biasType, KuDNN::Layout::ABCD};

        const std::vector<KuDNN::TensorInfo> diffSrcInfos = {
            {KuDNN::Shape(rnn.nIter, rnn.mb, rnn.slc), KuDNN::Element::TypeT::F32, KuDNN::Layout::ABC,
                KuDNN::Shape(rnn.mb * rnn.slc, g_dimRangeUpperBound, g_dimRangeUpperBound)},
        };
        const std::vector<KuDNN::TensorInfo> diffWeightInfos = {
            {KuDNN::Shape(rnn.nLayer, rnn.nDir, rnn.slc, rnn.nGates, rnn.dhc),
                KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCDE},
        };
        const std::vector<KuDNN::TensorInfo> diffDstInfos = {
            {KuDNN::Shape(rnn.nIter, rnn.mb, rnn.dlc), KuDNN::Element::TypeT::F32, KuDNN::Layout::ABC,
                KuDNN::Shape(rnn.mb * rnn.dlc, g_dimRangeUpperBound, g_dimRangeUpperBound)},
        };
        KuDNN::TensorInfo diffBiasInfo = {KuDNN::Shape(rnn.nLayer, rnn.nDir, rnn.nBias, rnn.dhc),
            KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCD};

        int dstLayerRate = rnn.execDir == KuDNN::Detail::ExecutionDirectionT::BI_CONCAT ? 2 : 1;
        void *src[3] = {nullptr, nullptr, nullptr};
        void *weight[4] = {nullptr, nullptr, nullptr, nullptr};
        void *dst[3] = {nullptr, nullptr, nullptr};
        void *bias = nullptr;
        void *diffSrc[3] = {nullptr, nullptr, nullptr};
        void *diffWeight[4] = {nullptr, nullptr, nullptr, nullptr};
        void *diffDst[3] = {nullptr, nullptr, nullptr};
        void *diffBias = nullptr;
        void *wsBase = nullptr;
        try {
            KuDNN::RnnBWD rnnBWD(srcInfos, weightInfos, dstInfos, biasInfo, diffSrcInfos, diffWeightInfos,
                diffDstInfos, diffBiasInfo, rnn, KuDNN::RnnAlgorithm::RNN, actfKind);
            src[0] = calloc(1, rnn.nIter *  (rnn.mb *  rnn.slc + 1) * srcType.GetSize());
            src[1] = calloc(1, rnn.nLayer * rnn.nDir *  rnn.mb *  rnn.sic * srcType.GetSize());
            weight[0] = calloc(1, rnn.nLayer * rnn.nDir * rnn.slc * rnn.nGates * rnn.dhc * srcType.GetSize());
            weight[1] = calloc(1, rnn.nLayer * rnn.nDir * rnn.sic * rnn.nGates * rnn.dic * srcType.GetSize());
            dst[0] = calloc(1, rnn.nIter * (rnn.mb *  rnn.dlc + 1) * dstLayerRate * srcType.GetSize());
            dst[1] = calloc(1, rnn.nLayer * rnn.nDir *  rnn.mb *  rnn.dic * srcType.GetSize());
            bias = calloc(1, rnn.nLayer * rnn.nDir * rnn.nBias *sizeof(float));

            wsBase = calloc(1, rnn.wsBiasOffset);
            diffSrc[0] = calloc(1, rnn.nIter *  rnn.mb *  rnn.slc * sizeof(float));
            diffSrc[1] = calloc(1, rnn.nLayer * rnn.nDir *  rnn.mb *  rnn.sic * sizeof(float));
            diffWeight[0] = calloc(1, rnn.nLayer * rnn.nDir * rnn.slc * rnn.nGates * rnn.dhc * sizeof(float));
            diffWeight[1] = calloc(1, rnn.nLayer * rnn.nDir * rnn.sic * rnn.nGates * rnn.dhc * sizeof(float));
            diffDst[0] = calloc(1, rnn.nIter *  rnn.mb *  rnn.dlc * dstLayerRate * sizeof(float));
            diffDst[1] = calloc(1, rnn.nLayer * rnn.nDir *  rnn.mb *  rnn.dic * sizeof(float));
            diffBias = calloc(1, rnn.nLayer * rnn.nDir * rnn.nBias * rnn.dhc * sizeof(float));
            if (CheckAnyNullPtr(src[0], src[1], weight[0], weight[1], dst[0], dst[1], bias, wsBase,
                diffSrc[0], diffSrc[1], diffWeight[0], diffWeight[1], diffDst[0], diffDst[1], diffBias)) {
                goto FREE_SPACE;
            }

            rnnBWD.Run((const void **)src, (const void **)weight, (const void **)dst, (const void *)bias, wsBase,
                (const void **)diffDst, diffSrc, diffWeight, diffBias, nullptr, nullptr);
        } catch (const KuDNN::Service::Unsupported& e) {
            printf("Unsupported data type! %s\n" , e.what());
        }
    FREE_SPACE:
        ReleaseVoidArray(src, 3);
        ReleaseVoidArray(weight, 4);
        ReleaseVoidArray(dst, 3);
        if (bias) {
            free(bias);
            bias = nullptr;
        }
        if (wsBase) {
            free(wsBase);
            wsBase = nullptr;
        }
        ReleaseVoidArray(diffSrc, 3);
        ReleaseVoidArray(diffWeight, 4);
        ReleaseVoidArray(diffDst, 3);
        if (diffBias) {
            free(diffBias);
            diffBias = nullptr;
        }
    }
    DT_FUZZ_END()
}

static void DtFuzzLstmBackward(int counts)
{
    char funcName[] = "DtFuzzLstmBackward";
    DT_FUZZ_START(0, counts, funcName, 0)
    {
        int eleIndex = 0;
        int withPeepholeIndex = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], 0, 0, g_boolMap.size() - 1);
        int withProjectionIndex = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], 0, 0, g_boolMap.size() - 1);
        int srcTypeIndex = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], 0, 0, g_precisionMap.size() - 1);
        int biasTypeIndex = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], 0, 0, g_precisionMap.size() - 1);
        int dirIndex = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], 0, 0, g_directionFuncMap.size() - 1);
        if (biasTypeIndex == 1 || srcTypeIndex == 1) {
            srcTypeIndex = 1;
            withProjectionIndex = 0;
        }

        bool peephole = g_boolMap[withPeepholeIndex];
        bool projection = g_boolMap[withProjectionIndex];

        KuDNN::Element::Type srcType(g_precisionMap[srcTypeIndex].second);
        KuDNN::Element::Type biasType(g_precisionMap[biasTypeIndex].second);
        KuDNN::RnnCfg rnn;
        rnn.nLayer = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], 1, 1, 11);
        rnn.nIter = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], 1, 1, 11);
        rnn.mb = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++],
            g_dimRangeInitVal, g_dimRangeLowerBound, g_dimRangeUpperBound);
        rnn.dhc = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++],
            g_dimRangeInitVal, g_dimRangeLowerBound, g_dimRangeUpperBound);
        rnn.sic = rnn.dhc;
        rnn.slc = rnn.sic;
        rnn.dic = rnn.dhc;
        rnn.dlc = rnn.dic;

        rnn.execDir = g_directionFuncMap[dirIndex].second;
        rnn.nGates = 4;
        rnn.isLbr = 0;
        SetRnnBwd(rnn, srcType, biasType);
        rnn.scratchCellSize = 0; // The value varies according to the algorithm.

        const std::vector<KuDNN::TensorInfo> srcInfos = {
            {KuDNN::Shape(rnn.nIter, rnn.mb, rnn.slc), srcType, KuDNN::Layout::ABC,
                KuDNN::Shape(rnn.mb * rnn.slc, g_dimRangeUpperBound, g_dimRangeUpperBound)},
        };
        const std::vector<KuDNN::TensorInfo> weightInfos = {
            {KuDNN::Shape(rnn.nLayer, rnn.nDir, rnn.slc, rnn.nGates, rnn.dhc), biasType, KuDNN::Layout::ABCDE},
        };
        const std::vector<KuDNN::TensorInfo> dstInfos = {
            {KuDNN::Shape(rnn.nIter, rnn.mb, rnn.dlc), srcType, KuDNN::Layout::ABC,
                KuDNN::Shape(rnn.mb * rnn.dlc, g_dimRangeUpperBound, g_dimRangeUpperBound)},
        };
        KuDNN::TensorInfo biasInfo = {KuDNN::Shape(rnn.nLayer, rnn.nDir, rnn.nBias, rnn.dhc),
            biasType, KuDNN::Layout::ABCD};

        const std::vector<KuDNN::TensorInfo> diffSrcInfos = {
            {KuDNN::Shape(rnn.nIter, rnn.mb, rnn.slc), KuDNN::Element::TypeT::F32, KuDNN::Layout::ABC,
                KuDNN::Shape(rnn.mb * rnn.slc, g_dimRangeUpperBound, g_dimRangeUpperBound)},
        };
        const std::vector<KuDNN::TensorInfo> diffWeightInfos = {
            {KuDNN::Shape(rnn.nLayer, rnn.nDir, rnn.slc, rnn.nGates, rnn.dhc),
                KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCDE},
        };
        const std::vector<KuDNN::TensorInfo> diffDstInfos = {
            {KuDNN::Shape(rnn.nIter, rnn.mb, rnn.dlc), KuDNN::Element::TypeT::F32, KuDNN::Layout::ABC,
                KuDNN::Shape(rnn.mb * rnn.dlc, g_dimRangeUpperBound, g_dimRangeUpperBound)},
        };
        KuDNN::TensorInfo diffBiasInfo = {KuDNN::Shape(rnn.nLayer, rnn.nDir, rnn.nBias, rnn.dhc),
            KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCD};

        int dstLayerRate = rnn.execDir == KuDNN::Detail::ExecutionDirectionT::BI_CONCAT ? 2 : 1;
        void *src[3] = {nullptr, nullptr, nullptr};
        void *weight[4] = {nullptr, nullptr, nullptr, nullptr};
        void *dst[3] = {nullptr, nullptr, nullptr};
        void *bias = nullptr;
        void *diffSrc[3] = {nullptr, nullptr, nullptr};
        void *diffWeight[4] = {nullptr, nullptr, nullptr, nullptr};
        void *diffDst[3] = {nullptr, nullptr, nullptr};
        void *diffBias = nullptr;
        void *wsBase = nullptr;
        try {
            KuDNN::RnnBWD rnnBWD(srcInfos, weightInfos, dstInfos, biasInfo, diffSrcInfos, diffWeightInfos,
                diffDstInfos, diffBiasInfo, rnn, KuDNN::RnnAlgorithm::LSTM, KuDNN::ActivateFunctionRNN::UNIMPLEMENTED);
            src[0] = calloc(1, rnn.nIter *  (rnn.mb *  rnn.slc + 1) * srcType.GetSize());
            src[1] = calloc(1, rnn.nLayer * rnn.nDir *  rnn.mb *  rnn.sic * srcType.GetSize());
            src[2] = calloc(1, rnn.nLayer * rnn.nDir *  rnn.mb *  rnn.dhc * sizeof(float));
            weight[0] = calloc(1, rnn.nLayer * rnn.nDir * rnn.slc * rnn.nGates * rnn.dhc * srcType.GetSize());
            weight[1] = calloc(1, rnn.nLayer * rnn.nDir * rnn.sic * rnn.nGates * rnn.dic * srcType.GetSize());
            weight[2] = peephole ? calloc(1, rnn.nLayer * rnn.nDir * 0x3 * rnn.dhc * sizeof(float)) : nullptr;
            weight[3] = projection ? calloc(1, rnn.nLayer * rnn.nDir * rnn.weightsProjectionLd * rnn.dic * sizeof(float)) : nullptr;
            dst[0] = calloc(1, rnn.nIter *  rnn.nDir * (rnn.mb *  rnn.dlc + 1) * dstLayerRate * srcType.GetSize());
            dst[1] = calloc(1, rnn.nLayer * rnn.nDir *  rnn.mb *  rnn.dic * srcType.GetSize());
            dst[2] = calloc(1, rnn.nLayer * rnn.nDir *  rnn.mb *  rnn.dhc * sizeof(float));
            bias = calloc(1, rnn.nLayer * rnn.nDir * rnn.nBias * sizeof(float));

            wsBase = calloc(1, rnn.wsBiasOffset);
            diffSrc[0] = calloc(1, rnn.nIter *  rnn.mb *  rnn.slc * sizeof(float));
            diffSrc[1] = calloc(1, rnn.nLayer * rnn.nDir *  rnn.mb *  rnn.sic * sizeof(float));
            diffSrc[2] = calloc(1, rnn.nLayer * rnn.nDir *  rnn.mb *  rnn.dhc * sizeof(float));
            diffWeight[0] = calloc(1, rnn.nLayer * rnn.nDir * rnn.slc * rnn.nGates * rnn.dhc * sizeof(float));
            diffWeight[1] = calloc(1, rnn.nLayer * rnn.nDir * rnn.sic * rnn.nGates * rnn.dic * sizeof(float));
            diffWeight[2] = peephole ? calloc(1, rnn.nLayer * rnn.nDir * 0x3 * rnn.nGates * rnn.dhc * sizeof(float)) : nullptr;
            diffWeight[3] = projection ? calloc(1, rnn.nLayer * rnn.mb * rnn.nGates * rnn.nDir * rnn.diffWeightsProjectionLd * rnn.dhc * sizeof(float)) : nullptr;
            diffDst[0] = calloc(1, rnn.nIter *  rnn.mb *  rnn.nDir * rnn.dlc * dstLayerRate * sizeof(float));
            diffDst[1] = calloc(1, rnn.nLayer * rnn.nDir *  rnn.mb *  rnn.dic * sizeof(float));
            diffDst[2] = calloc(1, (rnn.nLayer * rnn.nDir + 1) *  rnn.mb *  rnn.dhc * sizeof(float));
            diffBias = calloc(1, rnn.nLayer * rnn.nDir * rnn.nBias * rnn.dhc * sizeof(float));
            if (CheckAnyNullPtr(src[0], src[1], weight[0], weight[1], dst[0], dst[1], bias, wsBase,
                diffSrc[0], diffSrc[1], diffWeight[0], diffWeight[1], diffDst[0], diffDst[1], diffDst[2], diffBias)) {
                goto FREE_SPACE;
            }

            rnnBWD.Run((const void **)src, (const void **)weight, (const void **)dst, (const void *)bias, wsBase,
                (const void **)diffDst, diffSrc, diffWeight, diffBias, nullptr, nullptr);
        } catch (const KuDNN::Service::Unsupported& e) {
            printf("Unsupported data type! %s\n" , e.what());
        }
    FREE_SPACE:
        ReleaseVoidArray(src, 3);
        ReleaseVoidArray(weight, 4);
        ReleaseVoidArray(dst, 3);
        if (bias) {
            free(bias);
            bias = nullptr;
        }
        if (wsBase) {
            free(wsBase);
            wsBase = nullptr;
        }
        ReleaseVoidArray(diffSrc, 3);
        ReleaseVoidArray(diffWeight, 4);
        ReleaseVoidArray(diffDst, 3);
        if (diffBias) {
            free(diffBias);
            diffBias = nullptr;
        }
    }
    DT_FUZZ_END()
}

static void DtFuzzGruBackward(int counts)
{
    char funcName[] = "DtFuzzGruBackward";
    DT_FUZZ_START(0, counts, funcName, 0)
    {
        int eleIndex = 0;
        int srcTypeIndex = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], 0, 0, g_precisionMap.size() - 1);
        int biasTypeIndex = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], 0, 0, g_precisionMap.size() - 1);
        int dirIndex = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], 0, 0, g_directionFuncMap.size() - 1);

        KuDNN::Element::Type srcType(g_precisionMap[srcTypeIndex].second);
        KuDNN::Element::Type biasType(g_precisionMap[biasTypeIndex].second);
        KuDNN::ActivateFunctionRNN actfKind = g_activationFuncMap[INDEX3].second;
        KuDNN::RnnCfg rnn;
        rnn.nLayer = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], 1, 1, 11);
        rnn.nIter = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], 1, 1, 11);
        rnn.mb = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++],
            g_dimRangeInitVal, g_dimRangeLowerBound, g_dimRangeUpperBound);
        rnn.dhc = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++],
            g_dimRangeInitVal, g_dimRangeLowerBound, g_dimRangeUpperBound);
        rnn.sic = rnn.dhc;
        rnn.slc = rnn.sic;
        rnn.dic = rnn.dhc;
        rnn.dlc = rnn.dic;
        if (rnn.nLayer == 1) {
            rnn.slc = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++],
                g_dimRangeInitVal, g_dimRangeLowerBound, g_dimRangeUpperBound);
        }

        rnn.dstLayerLd = rnn.sic;

        rnn.execDir = g_directionFuncMap[dirIndex].second;
        rnn.nGates = 0x3;
        rnn.isLbr = 0;

        SetRnnBwd(rnn, srcType, biasType);
        rnn.scratchCellSize = (size_t)rnn.wsStatesLayerNld * rnn.wsStatesLayerLd * sizeof(float); // The value varies according to the algorithm.

        const std::vector<KuDNN::TensorInfo> srcInfos = {
            {KuDNN::Shape(rnn.nIter, rnn.mb, rnn.slc), srcType, KuDNN::Layout::ABC,
                KuDNN::Shape(rnn.mb * rnn.slc, g_dimRangeUpperBound, g_dimRangeUpperBound)},
        };
        const std::vector<KuDNN::TensorInfo> weightInfos = {
            {KuDNN::Shape(rnn.nLayer, rnn.nDir, rnn.slc, rnn.nGates, rnn.dhc), biasType, KuDNN::Layout::ABCDE},
        };
        const std::vector<KuDNN::TensorInfo> dstInfos = {
            {KuDNN::Shape(rnn.nIter, rnn.mb, rnn.dlc), srcType, KuDNN::Layout::ABC,
                KuDNN::Shape(rnn.mb * rnn.dlc, g_dimRangeUpperBound, g_dimRangeUpperBound)},
        };
        KuDNN::TensorInfo biasInfo = {KuDNN::Shape(rnn.nLayer, rnn.nDir, rnn.nBias, rnn.dhc),
            biasType, KuDNN::Layout::ABCD};

        const std::vector<KuDNN::TensorInfo> diffSrcInfos = {
            {KuDNN::Shape(rnn.nIter, rnn.mb, rnn.slc), KuDNN::Element::TypeT::F32, KuDNN::Layout::ABC,
                KuDNN::Shape(rnn.mb * rnn.slc, g_dimRangeUpperBound, g_dimRangeUpperBound)},
        };
        const std::vector<KuDNN::TensorInfo> diffWeightInfos = {
            {KuDNN::Shape(rnn.nLayer, rnn.nDir, rnn.slc, rnn.nGates, rnn.dhc),
                KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCDE},
        };
        const std::vector<KuDNN::TensorInfo> diffDstInfos = {
            {KuDNN::Shape(rnn.nIter, rnn.mb, rnn.dlc), KuDNN::Element::TypeT::F32, KuDNN::Layout::ABC,
                KuDNN::Shape(rnn.mb * rnn.dlc, g_dimRangeUpperBound, g_dimRangeUpperBound)},
        };
        KuDNN::TensorInfo diffBiasInfo = {KuDNN::Shape(rnn.nLayer, rnn.nDir, rnn.nBias, rnn.dhc),
            KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCD};

        int dstLayerRate = rnn.execDir == KuDNN::Detail::ExecutionDirectionT::BI_CONCAT ? 2 : 1;
        void *src[3] = {nullptr, nullptr, nullptr};
        void *weight[4] = {nullptr, nullptr, nullptr, nullptr};
        void *dst[3] = {nullptr, nullptr, nullptr};
        void *bias = nullptr;
        void *diffSrc[3] = {nullptr, nullptr, nullptr};
        void *diffWeight[4] = {nullptr, nullptr, nullptr, nullptr};
        void *diffDst[3] = {nullptr, nullptr, nullptr};
        void *diffBias = nullptr;
        void *wsBase = nullptr;
        try {
            KuDNN::RnnBWD rnnBWD(srcInfos, weightInfos, dstInfos, biasInfo, diffSrcInfos, diffWeightInfos,
                diffDstInfos, diffBiasInfo, rnn, KuDNN::RnnAlgorithm::GRU, actfKind);
            src[0] = calloc(1, rnn.nIter *  (rnn.mb *  rnn.slc + 1) * srcType.GetSize());
            src[1] = calloc(1, rnn.nLayer * rnn.nDir *  rnn.mb *  rnn.sic * srcType.GetSize());
            weight[0] = calloc(1, rnn.nLayer * rnn.nDir * rnn.slc * rnn.nGates * rnn.dhc * srcType.GetSize());
            weight[1] = calloc(1, rnn.nLayer * rnn.nDir * rnn.sic * rnn.nGates * rnn.dic * srcType.GetSize());
            dst[0] = calloc(1, rnn.nIter *  (rnn.mb *  rnn.dlc + 1) * dstLayerRate * srcType.GetSize());
            dst[1] = calloc(1, rnn.nLayer * rnn.nDir *  rnn.mb *  rnn.dic * srcType.GetSize());
            bias = calloc(1, rnn.nLayer * rnn.nDir * rnn.nBias * sizeof(float));

            wsBase = calloc(1, rnn.wsBiasOffset);
            diffSrc[0] = calloc(1, rnn.nIter *  rnn.mb *  rnn.slc * sizeof(float));
            diffSrc[1] = calloc(1, rnn.nLayer * rnn.nDir *  rnn.mb *  rnn.sic * sizeof(float));
            diffWeight[0] = calloc(1, rnn.nLayer * rnn.nDir * rnn.slc * rnn.nGates * rnn.dhc * sizeof(float));
            diffWeight[1] = calloc(1, rnn.nLayer * rnn.nDir * rnn.sic * rnn.nGates * rnn.dhc * sizeof(float));
            diffDst[0] = calloc(1, rnn.nIter *  rnn.mb *  rnn.dlc * dstLayerRate * sizeof(float));
            diffDst[1] = calloc(1, rnn.nLayer * rnn.nDir *  rnn.mb *  rnn.dic * sizeof(float));
            diffBias = calloc(1, rnn.nLayer * rnn.nDir * rnn.nBias * rnn.dhc * sizeof(float));
            if (CheckAnyNullPtr(src[0], src[1], weight[0], weight[1], dst[0], dst[1], bias, wsBase,
                diffSrc[0], diffSrc[1], diffWeight[0], diffWeight[1], diffDst[0], diffDst[1], diffBias)) {
                goto FREE_SPACE;
            }
            rnnBWD.Run((const void **)src, (const void **)weight, (const void **)dst, (const void *)bias, wsBase,
                (const void **)diffDst, diffSrc, diffWeight, diffBias, nullptr, nullptr);
        } catch (const KuDNN::Service::Unsupported& e) {
            printf("Unsupported data type! %s\n" , e.what());
        }
    FREE_SPACE:
        ReleaseVoidArray(src, 2);
        ReleaseVoidArray(weight, 2);
        ReleaseVoidArray(dst, 2);
        if (bias) {
            free(bias);
            bias = nullptr;
        }
        if (wsBase) {
            free(wsBase);
            wsBase = nullptr;
        }
        ReleaseVoidArray(diffSrc, 2);
        ReleaseVoidArray(diffWeight, 2);
        ReleaseVoidArray(diffDst, 2);
        if (diffBias) {
            free(diffBias);
            diffBias = nullptr;
        }
    }
    DT_FUZZ_END()
}

static void DtFuzzAuGruBackward(int counts)
{
    char funcName[] = "DtFuzzAuGruBackward";
    DT_FUZZ_START(0, counts, funcName, 0)
    {
        int eleIndex = 0;
        int srcTypeIndex = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], 0, 0, g_precisionMap.size() - 1);
        int biasTypeIndex = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], 0, 0, g_precisionMap.size() - 1);

        KuDNN::Element::Type srcType(g_precisionMap[srcTypeIndex].second);
        KuDNN::Element::Type biasType(g_precisionMap[biasTypeIndex].second);
        KuDNN::ActivateFunctionRNN actfKind = g_activationFuncMap[INDEX3].second;
        KuDNN::RnnCfg rnn;
        rnn.nLayer = 1;
        rnn.nIter = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], 1, 1, 11);
        rnn.mb = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++],
            g_dimRangeInitVal, g_dimRangeLowerBound, g_dimRangeUpperBound);
        rnn.dhc = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++],
            g_dimRangeInitVal, g_dimRangeLowerBound, g_dimRangeUpperBound);
        rnn.sic = rnn.dhc;
        rnn.slc = rnn.sic;
        rnn.dic = rnn.dhc;
        rnn.dlc = rnn.dic;

        rnn.dstLayerLd = rnn.sic;

        rnn.execDir = g_directionFuncMap[INDEX0].second;
        rnn.nGates = 0x3;
        rnn.isLbr = 0;

        SetRnnBwd(rnn, srcType, biasType);
        rnn.scratchCellSize = (size_t)rnn.wsStatesLayerNld * rnn.wsStatesLayerLd * sizeof(float); // The value varies according to the algorithm.

        const std::vector<KuDNN::TensorInfo> srcInfos = {
            {KuDNN::Shape(rnn.nIter, rnn.mb, rnn.slc), srcType, KuDNN::Layout::ABC,
                KuDNN::Shape(rnn.mb * rnn.slc, g_dimRangeUpperBound, g_dimRangeUpperBound)},
        };
        const std::vector<KuDNN::TensorInfo> weightInfos = {
            {KuDNN::Shape(rnn.nLayer, rnn.nDir, rnn.slc, rnn.nGates, rnn.dhc), biasType, KuDNN::Layout::ABCDE},
        };
        const std::vector<KuDNN::TensorInfo> dstInfos = {
            {KuDNN::Shape(rnn.nIter, rnn.mb, rnn.dlc), srcType, KuDNN::Layout::ABC,
                KuDNN::Shape(rnn.mb * rnn.dlc, g_dimRangeUpperBound, g_dimRangeUpperBound)},
        };
        KuDNN::TensorInfo biasInfo = {KuDNN::Shape(rnn.nLayer, rnn.nDir, rnn.nBias, rnn.dhc),
            biasType, KuDNN::Layout::ABCD};

        const std::vector<KuDNN::TensorInfo> diffSrcInfos = {
            {KuDNN::Shape(rnn.nIter, rnn.mb, rnn.slc), KuDNN::Element::TypeT::F32, KuDNN::Layout::ABC,
                KuDNN::Shape(rnn.mb * rnn.slc, g_dimRangeUpperBound, g_dimRangeUpperBound)},
        };
        const std::vector<KuDNN::TensorInfo> diffWeightInfos = {
            {KuDNN::Shape(rnn.nLayer, rnn.nDir, rnn.slc, rnn.nGates, rnn.dhc),
                KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCDE},
        };
        const std::vector<KuDNN::TensorInfo> diffDstInfos = {
            {KuDNN::Shape(rnn.nIter, rnn.mb, rnn.dlc), KuDNN::Element::TypeT::F32, KuDNN::Layout::ABC,
                KuDNN::Shape(rnn.mb * rnn.dlc, g_dimRangeUpperBound, g_dimRangeUpperBound)},
        };
        KuDNN::TensorInfo diffBiasInfo = {KuDNN::Shape(rnn.nLayer, rnn.nDir, rnn.nBias, rnn.dhc),
            KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCD};

        int dstLayerRate = 1;
        void *src[3] = {nullptr, nullptr, nullptr};
        void *weight[4] = {nullptr, nullptr, nullptr, nullptr};
        void *dst[3] = {nullptr, nullptr, nullptr};
        void *bias = nullptr;
        void *diffSrc[3] = {nullptr, nullptr, nullptr};
        void *diffWeight[4] = {nullptr, nullptr, nullptr, nullptr};
        void *diffDst[3] = {nullptr, nullptr, nullptr};
        void *diffBias = nullptr;
        void *wsBase = nullptr;
        void *augruAttentions = nullptr;
        void *diffAugruAttention = nullptr;
        try {
            KuDNN::RnnBWD rnnBWD(srcInfos, weightInfos, dstInfos, biasInfo, diffSrcInfos, diffWeightInfos,
                diffDstInfos, diffBiasInfo, rnn, KuDNN::RnnAlgorithm::AUGRU, actfKind);
            src[0] = calloc(1, rnn.nIter *  (rnn.mb *  rnn.slc + 1) * srcType.GetSize());
            src[1] = calloc(1, rnn.nLayer * rnn.nDir *  rnn.mb *  rnn.sic * srcType.GetSize());
            weight[0] = calloc(1, rnn.nLayer * rnn.nDir * rnn.slc * rnn.nGates * rnn.dhc * srcType.GetSize());
            weight[1] = calloc(1, rnn.nLayer * rnn.nDir * rnn.sic * rnn.nGates * rnn.dic * srcType.GetSize());
            dst[0] = calloc(1, rnn.nIter *  (rnn.mb *  rnn.dlc + 1) * dstLayerRate * srcType.GetSize());
            dst[1] = calloc(1, rnn.nLayer * rnn.nDir *  rnn.mb *  rnn.dic * srcType.GetSize());
            bias = calloc(1, rnn.nLayer * rnn.nDir * rnn.nBias * sizeof(float));

            wsBase = calloc(1, rnn.wsBiasOffset);
            diffSrc[0] = calloc(1, rnn.nIter *  rnn.mb *  rnn.slc * sizeof(float));
            diffSrc[1] = calloc(1, rnn.nLayer * rnn.nDir *  rnn.mb *  rnn.sic * sizeof(float));
            diffWeight[0] = calloc(1, rnn.nLayer * rnn.nDir * rnn.slc * rnn.nGates * rnn.dhc * sizeof(float));
            diffWeight[1] = calloc(1, rnn.nLayer * rnn.nDir * rnn.sic * rnn.nGates * rnn.dhc * sizeof(float));
            diffDst[0] = calloc(1, rnn.nIter *  rnn.mb *  rnn.dlc * dstLayerRate * sizeof(float));
            diffDst[1] = calloc(1, rnn.nLayer * rnn.nDir *  rnn.mb *  rnn.dic * sizeof(float));
            diffBias = calloc(1, rnn.nLayer * rnn.nDir * rnn.nBias * rnn.dhc * sizeof(float));
            augruAttentions = calloc(1, rnn.nIter * rnn.mb * sizeof(float));
            diffAugruAttention = calloc(1, rnn.nIter * rnn.mb * sizeof(float));
            if (CheckAnyNullPtr(src[0], src[1], weight[0], weight[1], dst[0], dst[1], bias, wsBase,
                diffSrc[0], diffSrc[1], diffWeight[0], diffWeight[1], diffDst[0], diffDst[1], diffBias, augruAttentions,
                diffAugruAttention)) {
                goto FREE_SPACE;
            }
            rnnBWD.Run((const void **)src, (const void **)weight, (const void **)dst, (const void *)bias, wsBase,
                (const void **)diffDst, diffSrc, diffWeight, diffBias, augruAttentions, diffAugruAttention);
        } catch (const KuDNN::Service::Unsupported& e) {
            printf("Unsupported data type! %s\n" , e.what());
        }
    FREE_SPACE:
        ReleaseVoidArray(src, 3);
        ReleaseVoidArray(weight, 4);
        ReleaseVoidArray(dst, 3);
        if (bias) {
            free(bias);
            bias = nullptr;
        }
        if (wsBase) {
            free(wsBase);
            wsBase = nullptr;
        }
        ReleaseVoidArray(diffSrc, 3);
        ReleaseVoidArray(diffWeight, 4);
        ReleaseVoidArray(diffDst, 3);
        if (diffBias) {
            free(diffBias);
            diffBias = nullptr;
        }
        if (augruAttentions) {
            free(augruAttentions);
            augruAttentions = nullptr;
        }
        if (diffAugruAttention) {
            free(diffAugruAttention);
            diffAugruAttention = nullptr;
        }
    }
    DT_FUZZ_END()
}

static void DtFuzzLbrGruBackward(int counts)
{
    char funcName[] = "DtFuzzLbrGruBackward";
    DT_FUZZ_START(0, counts, funcName, 0)
    {
        int eleIndex = 0;
        int srcTypeIndex = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], 0, 0, g_precisionMap.size() - 1);
        int biasTypeIndex = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], 0, 0, g_precisionMap.size() - 1);
        int dirIndex = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], 0, 0, g_directionFuncMap.size() - 1);

        KuDNN::Element::Type srcType(g_precisionMap[srcTypeIndex].second);
        KuDNN::Element::Type biasType(g_precisionMap[biasTypeIndex].second);
        KuDNN::ActivateFunctionRNN actfKind = g_activationFuncMap[INDEX3].second;
        KuDNN::RnnCfg rnn;
        rnn.nLayer = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], 1, 1, 11);
        rnn.nIter = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], 1, 1, 11);
        rnn.mb = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++],
            g_dimRangeInitVal, g_dimRangeLowerBound, g_dimRangeUpperBound);
        rnn.dhc = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++],
            g_dimRangeInitVal, g_dimRangeLowerBound, g_dimRangeUpperBound);

        rnn.sic = rnn.dhc;
        rnn.slc = rnn.sic;
        rnn.dic = rnn.dhc;
        rnn.dlc = rnn.dic;
        if (rnn.nLayer == 1) {
            rnn.slc = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++],
                g_dimRangeInitVal, g_dimRangeLowerBound, g_dimRangeUpperBound);
        }
        rnn.dstLayerLd = rnn.sic;

        rnn.execDir = g_directionFuncMap[dirIndex].second;
        rnn.nGates = 0x3;
        rnn.isLbr = 1;
        SetRnnBwd(rnn, srcType, biasType);
        rnn.scratchCellSize = (size_t)rnn.scratchGatesNld * rnn.scratchGatesLd * sizeof(float); // The value varies according to the algorithm.

        const std::vector<KuDNN::TensorInfo> srcInfos = {
            {KuDNN::Shape(rnn.nIter, rnn.mb, rnn.slc), srcType, KuDNN::Layout::ABC,
                KuDNN::Shape(rnn.mb * rnn.slc, g_dimRangeUpperBound, g_dimRangeUpperBound)},
        };
        const std::vector<KuDNN::TensorInfo> weightInfos = {
            {KuDNN::Shape(rnn.nLayer, rnn.nDir, rnn.slc, rnn.nGates, rnn.dhc), biasType, KuDNN::Layout::ABCDE},
        };
        const std::vector<KuDNN::TensorInfo> dstInfos = {
            {KuDNN::Shape(rnn.nIter, rnn.mb, rnn.dlc), srcType, KuDNN::Layout::ABC,
                KuDNN::Shape(rnn.mb * rnn.dlc, g_dimRangeUpperBound, g_dimRangeUpperBound)},
        };
        KuDNN::TensorInfo biasInfo = {KuDNN::Shape(rnn.nLayer, rnn.nDir, rnn.nBias, rnn.dhc),
            biasType, KuDNN::Layout::ABCD};

        const std::vector<KuDNN::TensorInfo> diffSrcInfos = {
            {KuDNN::Shape(rnn.nIter, rnn.mb, rnn.slc), KuDNN::Element::TypeT::F32, KuDNN::Layout::ABC,
                KuDNN::Shape(rnn.mb * rnn.slc, g_dimRangeUpperBound, g_dimRangeUpperBound)},
        };
        const std::vector<KuDNN::TensorInfo> diffWeightInfos = {
            {KuDNN::Shape(rnn.nLayer, rnn.nDir, rnn.slc, rnn.nGates, rnn.dhc),
                KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCDE},
        };
        const std::vector<KuDNN::TensorInfo> diffDstInfos = {
            {KuDNN::Shape(rnn.nIter, rnn.mb, rnn.dlc), KuDNN::Element::TypeT::F32, KuDNN::Layout::ABC,
                KuDNN::Shape(rnn.mb * rnn.dlc, g_dimRangeUpperBound, g_dimRangeUpperBound)},
        };
        KuDNN::TensorInfo diffBiasInfo = {KuDNN::Shape(rnn.nLayer, rnn.nDir, rnn.nBias, rnn.dhc),
            KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCD};

        int dstLayerRate = rnn.execDir == KuDNN::Detail::ExecutionDirectionT::BI_CONCAT ? 2 : 1;
        void *src[3] = {nullptr, nullptr, nullptr};
        void *weight[4] = {nullptr, nullptr, nullptr, nullptr};
        void *dst[3] = {nullptr, nullptr, nullptr};
        void *bias = nullptr;
        void *diffSrc[3] = {nullptr, nullptr, nullptr};
        void *diffWeight[4] = {nullptr, nullptr, nullptr, nullptr};
        void *diffDst[3] = {nullptr, nullptr, nullptr};
        void *diffBias = nullptr;
        void *wsBase = nullptr;
        try {
            KuDNN::RnnBWD rnnBWD(srcInfos, weightInfos, dstInfos, biasInfo, diffSrcInfos, diffWeightInfos,
                diffDstInfos, diffBiasInfo, rnn, KuDNN::RnnAlgorithm::LBR_GRU, actfKind);
            src[0] = calloc(1, rnn.nIter *  (rnn.mb *  rnn.slc + 1) * srcType.GetSize());
            src[1] = calloc(1, rnn.nLayer * rnn.nDir *  rnn.mb *  rnn.sic * srcType.GetSize());
            weight[0] = calloc(1, rnn.nLayer * rnn.nDir * rnn.slc * rnn.nGates * rnn.dhc * srcType.GetSize());
            weight[1] = calloc(1, rnn.nLayer * rnn.nDir * rnn.sic * rnn.nGates * rnn.dic * srcType.GetSize());
            dst[0] = calloc(1, rnn.nIter *  (rnn.mb *  rnn.dlc + 1) * dstLayerRate * srcType.GetSize());
            dst[1] = calloc(1, rnn.nLayer * rnn.nDir *  rnn.mb *  rnn.dic * srcType.GetSize());
            bias = calloc(1, rnn.nLayer * rnn.nDir * rnn.nBias * sizeof(float));

            wsBase = calloc(1, rnn.wsBiasOffset * sizeof(float));
            diffSrc[0] = calloc(1, rnn.nIter *  rnn.mb *  rnn.slc * sizeof(float));
            diffSrc[1] = calloc(1, rnn.nLayer * rnn.nDir *  rnn.mb *  rnn.sic * sizeof(float));
            diffWeight[0] = calloc(1, rnn.nLayer * rnn.nDir * rnn.slc * rnn.nGates * rnn.dhc * sizeof(float));
            diffWeight[1] = calloc(1, rnn.nLayer * rnn.nDir * rnn.sic * rnn.nGates * rnn.dhc * sizeof(float));
            diffDst[0] = calloc(1, rnn.nIter *  rnn.mb *  rnn.dlc * dstLayerRate * sizeof(float));
            diffDst[1] = calloc(1, rnn.nLayer * rnn.nDir *  rnn.mb *  rnn.dic * sizeof(float));
            diffBias = calloc(1, rnn.nLayer * rnn.nDir * rnn.nBias * rnn.dhc * sizeof(float));
            if (CheckAnyNullPtr(src[0], src[1], weight[0], weight[1], dst[0], dst[1], bias, wsBase,
                diffSrc[0], diffSrc[1], diffWeight[0], diffWeight[1], diffDst[0], diffDst[1], diffBias)) {
                goto FREE_SPACE;
            }
            rnnBWD.Run((const void **)src, (const void **)weight, (const void **)dst, (const void *)bias, wsBase,
                (const void **)diffDst, diffSrc, diffWeight, diffBias, nullptr, nullptr);
        } catch (const KuDNN::Service::Unsupported& e) {
            printf("Unsupported data type! %s\n" , e.what());
        }
    FREE_SPACE:
        ReleaseVoidArray(src, 3);
        ReleaseVoidArray(weight, 4);
        ReleaseVoidArray(dst, 3);
        if (bias) {
            free(bias);
            bias = nullptr;
        }
        if (wsBase) {
            free(wsBase);
            wsBase = nullptr;
        }
        ReleaseVoidArray(diffSrc, 3);
        ReleaseVoidArray(diffWeight, 4);
        ReleaseVoidArray(diffDst, 3);
        if (diffBias) {
            free(diffBias);
            diffBias = nullptr;
        }
    }
    DT_FUZZ_END()
}

static void DtFuzzLbrAuGruBackward(int counts)
{
    char funcName[] = "DtFuzzLbrAuGruBackward";
    DT_FUZZ_START(0, counts, funcName, 0)
    {
        int eleIndex = 0;
        int srcTypeIndex = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], 0, 0, g_precisionMap.size() - 1);
        int biasTypeIndex = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], 0, 0, g_precisionMap.size() - 1);

        KuDNN::Element::Type srcType(g_precisionMap[srcTypeIndex].second);
        KuDNN::Element::Type biasType(g_precisionMap[biasTypeIndex].second);
        KuDNN::ActivateFunctionRNN actfKind = g_activationFuncMap[INDEX3].second;
        KuDNN::RnnCfg rnn;
        rnn.nLayer = 1;
        rnn.nIter = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], 1, 1, 11);
        rnn.mb = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++],
            g_dimRangeInitVal, g_dimRangeLowerBound, g_dimRangeUpperBound);
        rnn.dhc = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++],
            g_dimRangeInitVal, g_dimRangeLowerBound, g_dimRangeUpperBound);
        rnn.sic = rnn.dhc;
        rnn.slc = rnn.sic;
        rnn.dic = rnn.dhc;
        rnn.dlc = rnn.dic;

        rnn.dstLayerLd = rnn.sic;

        rnn.execDir = g_directionFuncMap[INDEX0].second;
        rnn.nGates = 0x3;
        rnn.isLbr = 1;

        SetRnnBwd(rnn, srcType, biasType);
        rnn.scratchCellSize = (size_t)rnn.wsStatesLayerNld * rnn.wsStatesLayerLd * sizeof(float); // The value varies according to the algorithm.

        const std::vector<KuDNN::TensorInfo> srcInfos = {
            {KuDNN::Shape(rnn.nIter, rnn.mb, rnn.slc), srcType, KuDNN::Layout::ABC,
                KuDNN::Shape(rnn.mb * rnn.slc, g_dimRangeUpperBound, g_dimRangeUpperBound)},
        };
        const std::vector<KuDNN::TensorInfo> weightInfos = {
            {KuDNN::Shape(rnn.nLayer, rnn.nDir, rnn.slc, rnn.nGates, rnn.dhc), biasType, KuDNN::Layout::ABCDE},
        };
        const std::vector<KuDNN::TensorInfo> dstInfos = {
            {KuDNN::Shape(rnn.nIter, rnn.mb, rnn.dlc), srcType, KuDNN::Layout::ABC,
                KuDNN::Shape(rnn.mb * rnn.dlc, g_dimRangeUpperBound, g_dimRangeUpperBound)},
        };
        KuDNN::TensorInfo biasInfo = {KuDNN::Shape(rnn.nLayer, rnn.nDir, rnn.nBias, rnn.dhc),
            biasType, KuDNN::Layout::ABCD};

        const std::vector<KuDNN::TensorInfo> diffSrcInfos = {
            {KuDNN::Shape(rnn.nIter, rnn.mb, rnn.slc), KuDNN::Element::TypeT::F32, KuDNN::Layout::ABC,
                KuDNN::Shape(rnn.mb * rnn.slc, g_dimRangeUpperBound, g_dimRangeUpperBound)},
        };
        const std::vector<KuDNN::TensorInfo> diffWeightInfos = {
            {KuDNN::Shape(rnn.nLayer, rnn.nDir, rnn.slc, rnn.nGates, rnn.dhc),
                KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCDE},
        };
        const std::vector<KuDNN::TensorInfo> diffDstInfos = {
            {KuDNN::Shape(rnn.nIter, rnn.mb, rnn.dlc), KuDNN::Element::TypeT::F32, KuDNN::Layout::ABC,
                KuDNN::Shape(rnn.mb * rnn.dlc, g_dimRangeUpperBound, g_dimRangeUpperBound)},
        };
        KuDNN::TensorInfo diffBiasInfo = {KuDNN::Shape(rnn.nLayer, rnn.nDir, rnn.nBias, rnn.dhc),
            KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCD};

        int dstLayerRate = 1;
        void *src[3] = {nullptr, nullptr, nullptr};
        void *weight[4] = {nullptr, nullptr, nullptr, nullptr};
        void *dst[3] = {nullptr, nullptr, nullptr};
        void *bias = nullptr;
        void *diffSrc[3] = {nullptr, nullptr, nullptr};
        void *diffWeight[4] = {nullptr, nullptr, nullptr, nullptr};
        void *diffDst[3] = {nullptr, nullptr, nullptr};
        void *diffBias = nullptr;
        void *wsBase = nullptr;
        void *augruAttentions = nullptr;
        void *diffAugruAttention = nullptr;
        try {
            KuDNN::RnnBWD rnnBWD(srcInfos, weightInfos, dstInfos, biasInfo, diffSrcInfos, diffWeightInfos,
                diffDstInfos, diffBiasInfo, rnn, KuDNN::RnnAlgorithm::LBR_AUGRU, actfKind);
            src[0] = calloc(1, rnn.nIter *  (rnn.mb *  rnn.slc + 1) * srcType.GetSize());
            src[1] = calloc(1, rnn.nLayer * rnn.nDir *  rnn.mb *  rnn.sic * srcType.GetSize());
            weight[0] = calloc(1, rnn.nLayer * rnn.nDir * rnn.slc * rnn.nGates * rnn.dhc * srcType.GetSize());
            weight[1] = calloc(1, rnn.nLayer * rnn.nDir * rnn.sic * rnn.nGates * rnn.dic * srcType.GetSize());
            dst[0] = calloc(1, rnn.nIter *  (rnn.mb *  rnn.dlc + 1) * dstLayerRate * srcType.GetSize());
            dst[1] = calloc(1, rnn.nLayer * rnn.nDir *  rnn.mb *  rnn.dic * srcType.GetSize());
            bias = calloc(1, rnn.nLayer * rnn.nDir * rnn.nBias * sizeof(float));

            wsBase = calloc(1, rnn.wsBiasOffset * sizeof(float));
            diffSrc[0] = calloc(1, rnn.nIter *  rnn.mb *  rnn.slc * sizeof(float));
            diffSrc[1] = calloc(1, rnn.nLayer * rnn.nDir *  rnn.mb *  rnn.sic * sizeof(float));
            diffWeight[0] = calloc(1, rnn.nLayer * rnn.nDir * rnn.slc * rnn.nGates * rnn.dhc * sizeof(float));
            diffWeight[1] = calloc(1, rnn.nLayer * rnn.nDir * rnn.sic * rnn.nGates * rnn.dhc * sizeof(float));
            diffDst[0] = calloc(1, rnn.nIter *  rnn.mb *  rnn.dlc * dstLayerRate * sizeof(float));
            diffDst[1] = calloc(1, rnn.nLayer * rnn.nDir *  rnn.mb *  rnn.dic * sizeof(float));
            diffBias = calloc(1, rnn.nLayer * rnn.nDir * rnn.nBias * rnn.dhc * sizeof(float));
            augruAttentions = calloc(1, rnn.nIter * rnn.mb * sizeof(float));
            diffAugruAttention = calloc(1, rnn.nIter * rnn.mb * sizeof(float));
            if (CheckAnyNullPtr(src[0], src[1], weight[0], weight[1], dst[0], dst[1], bias, wsBase,
                diffSrc[0], diffSrc[1], diffWeight[0], diffWeight[1], diffDst[0], diffDst[1], diffBias, augruAttentions,
                diffAugruAttention)) {
                goto FREE_SPACE;
            }
            rnnBWD.Run((const void **)src, (const void **)weight, (const void **)dst, (const void *)bias, wsBase,
                (const void **)diffDst, diffSrc, diffWeight, diffBias, augruAttentions, diffAugruAttention);
        } catch (const KuDNN::Service::Unsupported& e) {
            printf("Unsupported data type! %s\n" , e.what());
        }
    FREE_SPACE:
        ReleaseVoidArray(src, 2);
        ReleaseVoidArray(weight, 2);
        ReleaseVoidArray(dst, 2);
        if (bias) {
            free(bias);
            bias = nullptr;
        }
        if (wsBase) {
            free(wsBase);
            wsBase = nullptr;
        }
        ReleaseVoidArray(diffSrc, 2);
        ReleaseVoidArray(diffWeight, 2);
        ReleaseVoidArray(diffDst, 2);
        if (diffBias) {
            free(diffBias);
            diffBias = nullptr;
        }
        if (augruAttentions) {
            free(augruAttentions);
            augruAttentions = nullptr;
        }
        if (diffAugruAttention) {
            free(diffAugruAttention);
            diffAugruAttention = nullptr;
        }
    }
    DT_FUZZ_END()
}

static void DtFuzzBackwardSpecialTest(int counts)
{
    char funcName[] = "DtFuzzBackwardSpecialTest";
    DT_FUZZ_START(0, counts, funcName, 0)
    {
        int eleIndex = 0;
        int srcTypeIndex = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], 0, 0, g_precisionMap.size() - 1);
        int weightTypeIndex = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], 0, 0, g_precisionMap.size() - 1);
        int dstTypeIndex = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], 0, 0, g_precisionMap.size() - 1);
        int biasTypeIndex = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], 0, 0, g_precisionMap.size() - 1);
        int diffSrcTypeIndex = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], 0, 0, g_precisionMap.size() - 1);
        int diffWeightTypeIndex = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], 0, 0, g_precisionMap.size() - 1);
        int diffDstTypeIndex = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], 0, 0, g_precisionMap.size() - 1);
        int diffBiasTypeIndex = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], 0, 0, g_precisionMap.size() - 1);
        int actFuncIndex = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], 0, 0, g_activationFuncMap.size() - 1);
        int rnnFuncIndex = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], 0, 0, g_rnnFuncMap.size() - 1);
        int dirIndex = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], 0, 0, g_directionFuncMap.size() - 1);

        KuDNN::Element::Type srcType(g_precisionMap[srcTypeIndex].second);
        KuDNN::Element::Type weightType(g_precisionMap[weightTypeIndex].second);
        KuDNN::Element::Type dstType(g_precisionMap[dstTypeIndex].second);
        KuDNN::Element::Type biasType(g_precisionMap[biasTypeIndex].second);
        KuDNN::Element::Type diffSrcType(g_precisionMap[diffSrcTypeIndex].second);
        KuDNN::Element::Type diffWeightType(g_precisionMap[diffWeightTypeIndex].second);
        KuDNN::Element::Type diffDstType(g_precisionMap[diffDstTypeIndex].second);
        KuDNN::Element::Type diffBiasType(g_precisionMap[diffBiasTypeIndex].second);
        KuDNN::RnnAlgorithm funcT = g_rnnFuncMap[rnnFuncIndex].second;
        KuDNN::ActivateFunctionRNN actfKind = g_activationFuncMap[actFuncIndex].second;
        KuDNN::RnnCfg rnn;
        rnn.nLayer = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], 1, 1, 11);
        rnn.nIter = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], 1, 1, 11);
        rnn.mb = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++],
            g_dimRangeInitVal, g_dimRangeLowerBound, g_dimRangeUpperBound);
        rnn.dhc = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++],
            g_dimRangeInitVal, g_dimRangeLowerBound, g_dimRangeUpperBound);
        rnn.sic = rnn.dhc;
        rnn.slc = rnn.sic;
        rnn.dic = rnn.dhc;
        rnn.dlc = rnn.dic;
        if (rnn.nLayer == 1) {
            rnn.slc = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++],
                g_dimRangeInitVal, g_dimRangeLowerBound, g_dimRangeUpperBound);
        }
        if (rnn.nIter == 1) {
            rnn.sic = *(int *)DT_SetGetNumberRange(&g_Element[eleIndex++],
                g_dimRangeInitVal, g_dimRangeLowerBound, g_dimRangeUpperBound);
        }

        rnn.execDir = g_directionFuncMap[dirIndex].second;
        rnn.nGates = 1;
        rnn.isLbr = 0;
        SetRnnBwd(rnn, srcType, biasType);
        rnn.scratchCellSize = 0; // The value varies according to the algorithm.

        const std::vector<KuDNN::TensorInfo> srcInfos = {
            {KuDNN::Shape(rnn.nIter, rnn.mb, rnn.slc), srcType, KuDNN::Layout::ABC,
                KuDNN::Shape(rnn.mb * rnn.slc, g_dimRangeUpperBound, g_dimRangeUpperBound)},
        };
        const std::vector<KuDNN::TensorInfo> weightInfos = {
            {KuDNN::Shape(rnn.nLayer, rnn.nDir, rnn.slc, rnn.nGates, rnn.dhc), weightType, KuDNN::Layout::ABCDE},
        };
        const std::vector<KuDNN::TensorInfo> dstInfos = {
            {KuDNN::Shape(rnn.nIter, rnn.mb, rnn.dlc), dstType, KuDNN::Layout::ABC,
                KuDNN::Shape(rnn.mb * rnn.dlc, g_dimRangeUpperBound, g_dimRangeUpperBound)},
        };
        KuDNN::TensorInfo biasInfo = {KuDNN::Shape(rnn.nLayer, rnn.nDir, rnn.nBias, rnn.dhc),
            biasType, KuDNN::Layout::ABCD};

        const std::vector<KuDNN::TensorInfo> diffSrcInfos = {
            {KuDNN::Shape(rnn.nIter, rnn.mb, rnn.slc), diffSrcType, KuDNN::Layout::ABC,
                KuDNN::Shape(rnn.mb * rnn.slc, g_dimRangeUpperBound, g_dimRangeUpperBound)},
        };
        const std::vector<KuDNN::TensorInfo> diffWeightInfos = {
            {KuDNN::Shape(rnn.nLayer, rnn.nDir, rnn.slc, rnn.nGates, rnn.dhc),
                diffWeightType, KuDNN::Layout::ABCDE},
        };
        const std::vector<KuDNN::TensorInfo> diffDstInfos = {
            {KuDNN::Shape(rnn.nIter, rnn.mb, rnn.dlc), diffDstType, KuDNN::Layout::ABC,
                KuDNN::Shape(rnn.mb * rnn.dlc, g_dimRangeUpperBound, g_dimRangeUpperBound)},
        };
        KuDNN::TensorInfo diffBiasInfo = {KuDNN::Shape(rnn.nLayer, rnn.nDir, rnn.nBias, rnn.dhc),
            diffBiasType, KuDNN::Layout::ABCD};

        int dstLayerRate = rnn.execDir == KuDNN::Detail::ExecutionDirectionT::BI_CONCAT ? 2 : 1;
        void *src[3] = {nullptr, nullptr, nullptr};
        void *weight[4] = {nullptr, nullptr, nullptr, nullptr};
        void *dst[3] = {nullptr, nullptr, nullptr};
        void *bias = nullptr;
        void *diffSrc[3] = {nullptr, nullptr, nullptr};
        void *diffWeight[4] = {nullptr, nullptr, nullptr, nullptr};
        void *diffDst[3] = {nullptr, nullptr, nullptr};
        void *diffBias = nullptr;
        void *wsBase = nullptr;
        void *augruAttentions = nullptr;
        void *diffaugruAttentions = nullptr;
        try {
            KuDNN::RnnBWD rnn1(srcInfos, weightInfos, dstInfos, biasInfo, diffSrcInfos, diffWeightInfos,
                diffDstInfos, diffBiasInfo, rnn, funcT, actfKind);
            KuDNN::RnnBWD rnn2(rnn1);
            rnn2 = rnn1;
            KuDNN::RnnBWD rnn3(rnn1);
            rnn3 = std::move(rnn1);
            KuDNN::RnnBWD rnn4(std::move(rnn2));
            KuDNN::RnnBWD rnnBWD(srcInfos, weightInfos, dstInfos, biasInfo, diffSrcInfos, diffWeightInfos,
                diffDstInfos, diffBiasInfo, rnn, funcT, actfKind);
            src[0] = CreateRawPointer(*(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], 0, 0, 1),
                rnn.nIter *  (rnn.mb *  rnn.slc + 1) * srcType.GetSize());
            src[1] = CreateRawPointer(*(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], 0, 0, 1),
                rnn.nLayer * rnn.nDir *  rnn.mb *  rnn.sic * srcType.GetSize());
            weight[0] = CreateRawPointer(*(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], 0, 0, 1),
                rnn.nLayer * rnn.nDir * rnn.slc * rnn.nGates * rnn.dhc * srcType.GetSize());
            weight[1] =CreateRawPointer(*(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], 0, 0, 1),
                rnn.nLayer * rnn.nDir * rnn.sic * rnn.nGates * rnn.dhc * srcType.GetSize());
            weight[2] = CreateRawPointer(*(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], 0, 0, 1),
                rnn.nLayer * rnn.nDir * 0x3 * rnn.dhc * srcType.GetSize());
            weight[3] = CreateRawPointer(*(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], 0, 0, 1),
                rnn.nLayer * rnn.nDir * rnn.weightsProjectionLd * rnn.dhc * rnn.dhc * srcType.GetSize());
            dst[0] = CreateRawPointer(*(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], 0, 0, 1),
                rnn.nIter *  (rnn.mb *  rnn.dlc + 1) * dstLayerRate * srcType.GetSize());
            dst[1] = CreateRawPointer(*(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], 0, 0, 1),
                rnn.nLayer * rnn.nDir *  rnn.mb *  rnn.dic * srcType.GetSize());
            bias = CreateRawPointer(*(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], 0, 0, 1),
                rnn.nLayer * rnn.nDir * rnn.nBias * rnn.dhc * sizeof(float));
            augruAttentions = CreateRawPointer(*(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], 0, 0, 1),
                rnn.nIter * rnn.mb * rnn.dhc * srcType.GetSize());

            wsBase = CreateRawPointer(*(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], 0, 0, 1),
                rnn.wsBiasOffset);
            diffSrc[0] = CreateRawPointer(*(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], 0, 0, 1),
                rnn.nIter *  rnn.mb *  rnn.slc * sizeof(float));
            diffSrc[1] = CreateRawPointer(*(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], 0, 0, 1),
                rnn.nLayer * rnn.nDir *  rnn.mb *  rnn.sic * sizeof(float));
            diffWeight[0] = CreateRawPointer(*(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], 0, 0, 1),
                rnn.nLayer * rnn.nDir * rnn.slc * rnn.nGates * rnn.dhc * sizeof(float));
            diffWeight[1] = CreateRawPointer(*(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], 0, 0, 1),
                rnn.nLayer * rnn.nDir * rnn.sic * rnn.nGates * rnn.dhc * sizeof(float));
            diffWeight[2] = CreateRawPointer(*(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], 0, 0, 1),
                rnn.nLayer * rnn.nDir * 0x3 * rnn.dhc * sizeof(float));
            diffWeight[3] = CreateRawPointer(*(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], 0, 0, 1),
                rnn.nLayer * rnn.mb * rnn.nGates * rnn.nDir * rnn.diffWeightsProjectionLd * rnn.dhc * sizeof(float));
            diffDst[0] = CreateRawPointer(*(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], 0, 0, 1),
                rnn.nIter *  rnn.mb *  rnn.dlc * dstLayerRate * sizeof(float));
            diffDst[1] = CreateRawPointer(*(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], 0, 0, 1),
                rnn.nLayer * rnn.nDir *  rnn.mb *  rnn.dic * sizeof(float));
            diffDst[2] = CreateRawPointer(*(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], 0, 0, 1),
                (rnn.nLayer * rnn.nDir + 1) *  rnn.mb *  rnn.dhc * sizeof(float));
            diffBias = CreateRawPointer(*(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], 0, 0, 1),
                rnn.nLayer * rnn.nDir * rnn.nBias * rnn.dhc * sizeof(float));
            diffaugruAttentions = CreateRawPointer(*(int *)DT_SetGetNumberRange(&g_Element[eleIndex++], 0, 0, 1),
                rnn.nIter * rnn.mb * rnn.dhc * srcType.GetSize());
            rnnBWD.Run((const void **)src, (const void **)weight, (const void **)dst, (const void *)bias, wsBase,
                (const void **)diffDst, diffSrc, diffWeight, diffBias, augruAttentions, diffaugruAttentions);
        } catch (const KuDNN::Service::Unsupported& e) {
            printf("Unsupported data type! %s\n" , e.what());
        } catch (const KuDNN::Service::LogicError& e) {
            printf("Input Error! %s\n" , e.what());
        }
        ReleaseVoidArray(src, 3);
        ReleaseVoidArray(weight, 4);
        ReleaseVoidArray(dst, 3);
        if (bias) {
            free(bias);
            bias = nullptr;
        }
        if (wsBase) {
            free(wsBase);
            wsBase = nullptr;
        }
        ReleaseVoidArray(diffSrc, 3);
        ReleaseVoidArray(diffWeight, 4);
        ReleaseVoidArray(diffDst, 3);
        if (diffBias) {
            free(diffBias);
            diffBias = nullptr;
        }
        if (augruAttentions) {
            free(augruAttentions);
            augruAttentions = nullptr;
        }
        if (diffaugruAttentions) {
            free(diffaugruAttentions);
            diffaugruAttentions = nullptr;
        }
    }
    DT_FUZZ_END()
}

const int  g_rnnFuzzTestCount = 1000;

Cmdinfo rnnCmdTable[] = {
    {"RnnForwardNcsp", DtFuzzRnnForward, g_rnnFuzzTestCount},
    {"LstmForwardNcsp", DtFuzzLstmForward, g_rnnFuzzTestCount},
    {"GruForwardNcsp", DtFuzzGruForward, g_rnnFuzzTestCount},
    {"AuGruForwardNcsp", DtFuzzAuGruForward, g_rnnFuzzTestCount},
    {"LbrGruForwardNcsp", DtFuzzLbrGruForward, g_rnnFuzzTestCount},
    {"LbrAuGruForwardNcsp", DtFuzzLbrAuGruForward, g_rnnFuzzTestCount},
    {"ForwardSpecialTest", DtFuzzForwardSpecialTest, 1000},

    {"RnnBackwardNcsp", DtFuzzRnnBackward, g_rnnFuzzTestCount},
    {"LstmBackwardNcsp", DtFuzzLstmBackward, g_rnnFuzzTestCount},
    {"GruBackwardNcsp", DtFuzzGruBackward, g_rnnFuzzTestCount},
    {"AuGruBackwardNcsp", DtFuzzAuGruBackward, g_rnnFuzzTestCount},
    {"LbrGruBackwardNcsp", DtFuzzLbrGruBackward, g_rnnFuzzTestCount},
    {"LbrAuGruBackwardNcsp", DtFuzzLbrAuGruBackward, g_rnnFuzzTestCount},
    {"BackwardSpecialTest", DtFuzzBackwardSpecialTest, 1000},
};

int main(int argc, char *argv[])
{
    DT_Enable_Leak_Check(0, 0);
    int numOfTestItem = sizeof(rnnCmdTable) / sizeof(rnnCmdTable[0]);
    if (argc > 1) {
        int count = atoi(argv[1]);
        for (int testCaseIdx = 0; testCaseIdx < numOfTestItem; testCaseIdx++) {
            rnnCmdTable[testCaseIdx].testCount = count;
        }
    }

    int testCount = 0;
    for (int testCaseIdx = 0; testCaseIdx < numOfTestItem; testCaseIdx++) {
        printf("rnnCmdTable[%d] = %s test %d time\n", testCaseIdx, rnnCmdTable[testCaseIdx].cmd, testCount);
        testCount = rnnCmdTable[testCaseIdx].testCount;
        rnnCmdTable[testCaseIdx].func(testCount);
    }
    return 0;
}
