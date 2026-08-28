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

#ifndef KUDNN_BNORM_FWD_DEF_HPP_VC
#define KUDNN_BNORM_FWD_DEF_HPP_VC

#include <unordered_map>
#include <functional>
#include <cmath>
#include <arm_neon.h>

#include "vector_wrappers.hpp"
#include "kernels/bnorm/kudnn_bnorm_kernels.hpp"
#include "kudnn_bnorm_cmd_loop_unrol.hpp"
#include "kudnn_bnorm_cmd_loop_unrol_neon.hpp"

namespace KuDNN {
namespace Detail {
namespace BNorm {

template <SizeType CBLKLen>
void BNormFwdAXBMeanVariance(const float *meanVarCN, float *meanVar, SizeType loopRange, SizeType sumSize,
    SizeType channelSize, SizeType tailSize)
{
    svfloat32_t svMeanVar0;
    svfloat32_t svMeanVar1;
    svfloat32_t svMeanVar2;
    svfloat32_t svMeanVar3;
    SizeType offsetN = 0;
    svbool_t pg0;
    svbool_t pg1;
    svbool_t pg2;
    svbool_t pg3;
    InitPgLoopUnrol<CBLKLen>(tailSize, pg0, pg1, pg2, pg3);
    DupLoopUnrol<CBLKLen>(0.0f, svMeanVar0, svMeanVar1, svMeanVar2, svMeanVar3);
    for (SizeType loopIdx = 0; loopIdx < loopRange; loopIdx++) {
        svfloat32_t data0;
        svfloat32_t data1;
        svfloat32_t data2;
        svfloat32_t data3;
        LoadLoopUnrol<CBLKLen>(meanVarCN + offsetN, pg0, pg1, pg2, pg3, data0, data1, data2, data3);
        AddXLoopUnrol<CBLKLen>(pg0, pg1, pg2, pg3, svMeanVar0, svMeanVar1, svMeanVar2, svMeanVar3,
            data0, data1, data2, data3);
        offsetN += channelSize;
    }
    float sumSizeF = static_cast<float>(sumSize);
    DivnXLoopUnrol<CBLKLen>(sumSizeF, pg0, pg1, pg2, pg3, svMeanVar0, svMeanVar1, svMeanVar2, svMeanVar3);
    StoreLoopUnrol<CBLKLen, float>(meanVar, pg0, pg1, pg2, pg3, svMeanVar0, svMeanVar1, svMeanVar2, svMeanVar3);
}

using BNormFwdAXBMeanVarianceFunc = std::function<void(const float *, float *, SizeType, SizeType, SizeType, SizeType)>;

const std::unordered_map<SizeType, BNormFwdAXBMeanVarianceFunc> EVAL_BNORM_AXB_FWD_MEANVAR_FUNC_MAP = {
    {NUM_1, BNormFwdAXBMeanVariance<NUM_1>}, {NUM_2, BNormFwdAXBMeanVariance<NUM_2>},
    {NUM_3, BNormFwdAXBMeanVariance<NUM_3>}, {NUM_4, BNormFwdAXBMeanVariance<NUM_4>}
};

template <typename CompType>
void MergeMeanVar(const CompType *meanVarCN, CompType *meanVar, SizeType channelSize, SizeType sumSize,
    SizeType loopRange, SizeType loopCh)
{
    SizeType svStep = svcntw();
    CBLKParams cblkp;
    CalCBLKParams(cblkp, loopCh, svStep);
    SizeType offsetN = 0;
    for (SizeType cblkIdx = 0; cblkIdx < cblkp.cblkRange; cblkIdx += TAG_SIZE) {
        auto item = EVAL_BNORM_AXB_FWD_MEANVAR_FUNC_MAP.find(TAG_SIZE);
        if (item != EVAL_BNORM_AXB_FWD_MEANVAR_FUNC_MAP.end()) {
            auto func = item->second;
            func(meanVarCN + offsetN, meanVar + offsetN, loopRange, sumSize, channelSize, svStep);
        }
        offsetN += TAG_SIZE * svStep;
    }
    if (cblkp.tagTail != 0) {
        auto item = EVAL_BNORM_AXB_FWD_MEANVAR_FUNC_MAP.find(cblkp.tagTail);
        if (item != EVAL_BNORM_AXB_FWD_MEANVAR_FUNC_MAP.end()) {
            auto func = item->second;
            func(meanVarCN + offsetN, meanVar + offsetN, loopRange, sumSize, channelSize, cblkp.tailSize);
        }
    }
}

template <SizeType CBLKLen, typename InType>
void BNormFwdAXBMeanCN(const void *src, float *meanCN, SizeType loopRange, SizeType channelSize, SizeType tailSize)
{
    const InType *srcIn = static_cast<const InType *>(src);
    svfloat32_t svMean0;
    svfloat32_t svMean1;
    svfloat32_t svMean2;
    svfloat32_t svMean3;
    SizeType offsetN = 0;
    svbool_t pg0;
    svbool_t pg1;
    svbool_t pg2;
    svbool_t pg3;
    InitPgLoopUnrol<CBLKLen>(tailSize, pg0, pg1, pg2, pg3);
    DupLoopUnrol<CBLKLen>(0.0f, svMean0, svMean1, svMean2, svMean3);
    for (SizeType loopIdx = 0; loopIdx < loopRange; loopIdx++) {
        svfloat32_t data0;
        svfloat32_t data1;
        svfloat32_t data2;
        svfloat32_t data3;
        LoadLoopUnrol<CBLKLen>(srcIn + offsetN, pg0, pg1, pg2, pg3, data0, data1, data2, data3);
        AddXLoopUnrol<CBLKLen>(pg0, pg1, pg2, pg3, svMean0, svMean1, svMean2, svMean3, data0, data1, data2, data3);
        offsetN += channelSize;
    }
    StoreLoopUnrol<CBLKLen, float>(meanCN, pg0, pg1, pg2, pg3, svMean0, svMean1, svMean2, svMean3);
}

using BNormFwdAXBMeanCNFunc = std::function<void(const void *, float *, SizeType, SizeType, SizeType)>;

const std::unordered_map<SizeType, BNormFwdAXBMeanCNFunc> EVAL_BNORM_AXB_FWD_MEANCN_FUNC_MAP = {
    {NUM_11, BNormFwdAXBMeanCN<NUM_1, float>}, {NUM_21, BNormFwdAXBMeanCN<NUM_2, float>},
    {NUM_31, BNormFwdAXBMeanCN<NUM_3, float>}, {NUM_41, BNormFwdAXBMeanCN<NUM_4, float>},
    {NUM_12, BNormFwdAXBMeanCN<NUM_1, __fp16>}, {NUM_22, BNormFwdAXBMeanCN<NUM_2, __fp16>},
    {NUM_32, BNormFwdAXBMeanCN<NUM_3, __fp16>}, {NUM_42, BNormFwdAXBMeanCN<NUM_4, __fp16>},
    {NUM_13, BNormFwdAXBMeanCN<NUM_1, __bf16>}, {NUM_23, BNormFwdAXBMeanCN<NUM_2, __bf16>},
    {NUM_33, BNormFwdAXBMeanCN<NUM_3, __bf16>}, {NUM_43, BNormFwdAXBMeanCN<NUM_4, __bf16>},
    {NUM_14, BNormFwdAXBMeanCN<NUM_1, std::int8_t>}, {NUM_24, BNormFwdAXBMeanCN<NUM_2, std::int8_t>},
    {NUM_34, BNormFwdAXBMeanCN<NUM_3, std::int8_t>}, {NUM_44, BNormFwdAXBMeanCN<NUM_4, std::int8_t>}
};

template <typename CompType, typename InType, typename OutType>
void EvaluateMeanCNFwdAXB(const InType *src, CompType *meanCN, SizeType channelSize, SizeType loopRange)
{
    if constexpr (IsExecute<InType>()) {
        SizeType svStep = svcntw();
        CBLKParams cblkp;
        CalCBLKParams(cblkp, channelSize, svStep);
        SizeType offsetN = 0;
        SizeType typeIdx = GetDataTypeId<InType>();
        for (SizeType cblkIdx = 0; cblkIdx < cblkp.cblkRange; cblkIdx += TAG_SIZE) {
            auto item = EVAL_BNORM_AXB_FWD_MEANCN_FUNC_MAP.find(TAG_SIZE * NUM_10 + typeIdx);
            if (item != EVAL_BNORM_AXB_FWD_MEANCN_FUNC_MAP.end()) {
                auto func = item->second;
                func(src + offsetN, meanCN + offsetN, loopRange, channelSize, svStep);
            }
            offsetN += TAG_SIZE * svStep;
        }
        if (cblkp.tagTail != 0) {
            auto item = EVAL_BNORM_AXB_FWD_MEANCN_FUNC_MAP.find(cblkp.tagTail * NUM_10 + typeIdx);
            if (item != EVAL_BNORM_AXB_FWD_MEANCN_FUNC_MAP.end()) {
                auto func = item->second;
                func(src + offsetN, meanCN + offsetN, loopRange, channelSize, cblkp.tailSize);
            }
        }
    }
}

template <SizeType CBLKLen, typename InType>
void BNormFwdAXBVarianceCN(const void *src, const float *mean, float *varianceCN,
    SizeType loopRange, SizeType channelSize, SizeType tailSize)
{
    const InType *srcIn = static_cast<const InType *>(src);
    svfloat32_t svMean0;
    svfloat32_t svMean1;
    svfloat32_t svMean2;
    svfloat32_t svMean3;
    svfloat32_t svVar0;
    svfloat32_t svVar1;
    svfloat32_t svVar2;
    svfloat32_t svVar3;
    svbool_t pg0;
    svbool_t pg1;
    svbool_t pg2;
    svbool_t pg3;
    SizeType offsetN = 0;
    InitPgLoopUnrol<CBLKLen>(tailSize, pg0, pg1, pg2, pg3);
    DupLoopUnrol<CBLKLen>(0.0f, svVar0, svVar1, svVar2, svVar3);
    LoadLoopUnrol<CBLKLen>(mean, pg0, pg1, pg2, pg3, svMean0, svMean1, svMean2, svMean3);
    for (SizeType loopIdx = 0; loopIdx < loopRange; loopIdx++) {
        svfloat32_t data0;
        svfloat32_t data1;
        svfloat32_t data2;
        svfloat32_t data3;
        LoadLoopUnrol<CBLKLen>(srcIn + offsetN, pg0, pg1, pg2, pg3, data0, data1, data2, data3);
        SubLoopUnrol<CBLKLen>(pg0, pg1, pg2, pg3, data0, data1, data2, data3, svMean0, svMean1, svMean2, svMean3);
        MlaXLoopUnrol<CBLKLen>(pg0, pg1, pg2, pg3, data0, data1, data2, data3, data0, data1, data2, data3,
            svVar0, svVar1, svVar2, svVar3);
        offsetN += channelSize;
    }
    StoreLoopUnrol<CBLKLen>(varianceCN, pg0, pg1, pg2, pg3, svVar0, svVar1, svVar2, svVar3);
}

using BNormFwdAXBVarianceCNFunc = std::function<void(const void *, const float *, float *,
    SizeType, SizeType, SizeType)>;

const std::unordered_map<SizeType, BNormFwdAXBVarianceCNFunc> EVAL_BNORM_AXB_FWD_VARCN_FUNC_MAP = {
    {NUM_11, BNormFwdAXBVarianceCN<NUM_1, float>}, {NUM_21, BNormFwdAXBVarianceCN<NUM_2, float>},
    {NUM_31, BNormFwdAXBVarianceCN<NUM_3, float>}, {NUM_41, BNormFwdAXBVarianceCN<NUM_4, float>},
    {NUM_12, BNormFwdAXBVarianceCN<NUM_1, __fp16>}, {NUM_22, BNormFwdAXBVarianceCN<NUM_2, __fp16>},
    {NUM_32, BNormFwdAXBVarianceCN<NUM_3, __fp16>}, {NUM_42, BNormFwdAXBVarianceCN<NUM_4, __fp16>},
    {NUM_13, BNormFwdAXBVarianceCN<NUM_1, __bf16>}, {NUM_23, BNormFwdAXBVarianceCN<NUM_2, __bf16>},
    {NUM_33, BNormFwdAXBVarianceCN<NUM_3, __bf16>}, {NUM_43, BNormFwdAXBVarianceCN<NUM_4, __bf16>},
    {NUM_14, BNormFwdAXBVarianceCN<NUM_1, std::int8_t>}, {NUM_24, BNormFwdAXBVarianceCN<NUM_2, std::int8_t>},
    {NUM_34, BNormFwdAXBVarianceCN<NUM_3, std::int8_t>}, {NUM_44, BNormFwdAXBVarianceCN<NUM_4, std::int8_t>}
};

template <typename CompType, typename InType, typename OutType>
void EvaluateVarianceCNFwdAXB(const InType *src, const CompType *mean, CompType *varianceCN,
    SizeType channelSize, SizeType loopRange)
{
    if constexpr (IsExecute<InType>()) {
        SizeType svStep = svcntw();
        CBLKParams cblkp;
        CalCBLKParams(cblkp, channelSize, svStep);
        SizeType offsetN = 0;
        SizeType typeIdx = GetDataTypeId<InType>();
        for (SizeType cblkIdx = 0; cblkIdx < cblkp.cblkRange; cblkIdx += TAG_SIZE) {
            auto item = EVAL_BNORM_AXB_FWD_VARCN_FUNC_MAP.find(TAG_SIZE * NUM_10 + typeIdx);
            if (item != EVAL_BNORM_AXB_FWD_VARCN_FUNC_MAP.end()) {
                auto func = item->second;
                func(src + offsetN, mean + offsetN, varianceCN + offsetN, loopRange, channelSize, svStep);
            }
            offsetN += TAG_SIZE * svStep;
        }
        if (cblkp.tagTail != 0) {
            auto item = EVAL_BNORM_AXB_FWD_VARCN_FUNC_MAP.find(cblkp.tagTail * NUM_10 + typeIdx);
            if (item != EVAL_BNORM_AXB_FWD_VARCN_FUNC_MAP.end()) {
                auto func = item->second;
                func(src + offsetN, mean + offsetN, varianceCN + offsetN, loopRange, channelSize, cblkp.tailSize);
            }
        }
    }
}

template <SizeType CBLKLen, typename InType>
void BNormFwdAXB(const void *src, void *dst, const float* shift, const float* mean, const float* mulFactor,
    SizeType loopRange, SizeType channelSize, SizeType tailSize, bool useShift, bool fuseRelu,
    std::uint8_t *ws, Propagation pKind)
{
    Service::Unused(ws);
    Service::Unused(fuseRelu);
    Service::Unused(pKind);
    const InType *srcIn = static_cast<const InType *>(src);
    InType *dstIn = static_cast<InType *>(dst);
    svbool_t pg0;
    svbool_t pg1;
    svbool_t pg2;
    svbool_t pg3;
    InitPgLoopUnrol<CBLKLen>(tailSize, pg0, pg1, pg2, pg3);
    svfloat32_t svMean0;
    svfloat32_t svMean1;
    svfloat32_t svMean2;
    svfloat32_t svMean3;
    svfloat32_t svMF0;
    svfloat32_t svMF1;
    svfloat32_t svMF2;
    svfloat32_t svMF3;
    svfloat32_t svSh0;
    svfloat32_t svSh1;
    svfloat32_t svSh2;
    svfloat32_t svSh3;
    svfloat32_t data0;
    svfloat32_t data1;
    svfloat32_t data2;
    svfloat32_t data3;
    LoadLoopUnrol<CBLKLen, float>(mean, pg0, pg1, pg2, pg3, svMean0, svMean1, svMean2, svMean3);
    LoadLoopUnrol<CBLKLen, float>(mulFactor, pg0, pg1, pg2, pg3, svMF0, svMF1, svMF2, svMF3);
    if (useShift) {
        LoadLoopUnrol<CBLKLen, float>(shift, pg0, pg1, pg2, pg3, svSh0, svSh1, svSh2, svSh3);
    } else {
        DupLoopUnrol<CBLKLen>(0.0f, svSh0, svSh1, svSh2, svSh3);
    }
    SizeType offsetN = 0;
    for (SizeType loopIdx = 0; loopIdx < loopRange; loopIdx++) {
        LoadLoopUnrol<CBLKLen, InType>(srcIn + offsetN, pg0, pg1, pg2, pg3, data0, data1, data2, data3);
        SubLoopUnrol<CBLKLen>(pg0, pg1, pg2, pg3, data0, data1, data2, data3, svMean0, svMean1, svMean2, svMean3);
        MadXLoopUnrol<CBLKLen>(pg0, pg1, pg2, pg3, data0, data1, data2, data3,
            svMF0, svMF1, svMF2, svMF3, svSh0, svSh1, svSh2, svSh3);
        StoreLoopUnrol<CBLKLen, InType>(dstIn + offsetN, pg0, pg1, pg2, pg3, data0, data1, data2, data3);
        offsetN += channelSize;
    }
}

using BNormFwdAXBFunc = std::function<void(const void *, void *, const float*, const float*, const float*,
    SizeType, SizeType, SizeType, bool, bool, std::uint8_t *, Propagation)>;

const std::unordered_map<SizeType, BNormFwdAXBFunc> EVAL_BNORM_FWD_AXB_FUNC_MAP = {
    {NUM_11, BNormFwdAXB<NUM_1, float>}, {NUM_21, BNormFwdAXB<NUM_2, float>},
    {NUM_31, BNormFwdAXB<NUM_3, float>}, {NUM_41, BNormFwdAXB<NUM_4, float>},
    {NUM_12, BNormFwdAXB<NUM_1, __fp16>}, {NUM_22, BNormFwdAXB<NUM_2, __fp16>},
    {NUM_32, BNormFwdAXB<NUM_3, __fp16>}, {NUM_42, BNormFwdAXB<NUM_4, __fp16>},
    {NUM_13, BNormFwdAXB<NUM_1, __bf16>}, {NUM_23, BNormFwdAXB<NUM_2, __bf16>},
    {NUM_33, BNormFwdAXB<NUM_3, __bf16>}, {NUM_43, BNormFwdAXB<NUM_4, __bf16>},
    {NUM_14, BNormFwdAXB<NUM_1, std::int8_t>}, {NUM_24, BNormFwdAXB<NUM_2, std::int8_t>},
    {NUM_34, BNormFwdAXB<NUM_3, std::int8_t>}, {NUM_44, BNormFwdAXB<NUM_4, std::int8_t>}
};

template <typename CompType, typename InType, typename OutType>
void EvaluateBNormAXB(const InType *src, OutType *dst, const CompType *shift, const CompType *mean,
    const CompType *mulFactor, NormalizationFlags flags, SizeType channelSize, SizeType loopRange,
    std::uint8_t *ws, Propagation pKind)
{
    if constexpr (IsExecute<InType>()) {
        bool useShift = static_cast<bool>(flags & NormalizationFlags::USE_SHIFT);
        bool fuseRelu = static_cast<bool>(flags & NormalizationFlags::FUSE_NORM_RELU);
        SizeType svStep = svcntw();
        CBLKParams cblkp;
        CalCBLKParams(cblkp, channelSize, svStep);
        SizeType typeIdx = GetDataTypeId<InType>();
        SizeType offsetN = 0;
        for (SizeType cblkIdx = 0; cblkIdx < cblkp.cblkRange; cblkIdx += TAG_SIZE) {
            auto item = EVAL_BNORM_FWD_AXB_FUNC_MAP.find(TAG_SIZE * NUM_10 + typeIdx);
            if (item != EVAL_BNORM_FWD_AXB_FUNC_MAP.end()) {
                auto func = item->second;
                func(src + offsetN, dst + offsetN, shift + offsetN, mean + offsetN, mulFactor + offsetN,
                    loopRange, channelSize, svStep, useShift, fuseRelu, ws, pKind);
            }
            offsetN += TAG_SIZE * svStep;
        }
        if (cblkp.tagTail != 0) {
            auto item = EVAL_BNORM_FWD_AXB_FUNC_MAP.find(cblkp.tagTail * NUM_10 + typeIdx);
            if (item != EVAL_BNORM_FWD_AXB_FUNC_MAP.end()) {
                auto func = item->second;
                func(src + offsetN, dst + offsetN, shift + offsetN, mean + offsetN, mulFactor + offsetN,
                    loopRange, channelSize, cblkp.tailSize, useShift, fuseRelu, ws, pKind);
            }
        }
    }
}

} // namespace BNorm
} // namespace Detail
} // namespace KuDNN

#endif // KUDNN_BNORM_FWD_DEF_HPP
