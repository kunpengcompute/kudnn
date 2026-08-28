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
#ifndef KUDNN_BNORM_CMD_LOOP_UNROL_NEON_HPP
#define KUDNN_BNORM_CMD_LOOP_UNROL_NEON_HPP

#include "kudnn.hpp"

namespace KuDNN {
namespace Detail {
namespace BNorm {

template <typename InType>
inline void LoadBNormNEON(const InType *src, float32x4x2_t &out, const SizeType &loopIdx, const SizeType &loopRange)
{
    if constexpr (std::is_same_v<InType, float>) {
        out = vld1q_f32_x2(src);
    }
    Service::Unused(loopIdx, loopRange);
}

template <typename OutType>
inline void StoreBNormNEON(OutType *dst, const float32x4x2_t &svOut)
{
    if constexpr (std::is_same_v<OutType, float>) {
        vst1q_f32_x2(dst, svOut);
    }
}

template <SizeType CBLKLen, typename InType>
void LoadLoopUnrolNEON(const InType *src, float32x4x2_t &res0, float32x4x2_t &res1,
    float32x4x2_t &res2, float32x4x2_t &res3)
{
    const SizeType svStep = NUM_2 * Vector::NeonLanes<InType>();
    if constexpr (CBLKLen > NUM_0) {
        LoadBNormNEON(src, res0, NUM_0, NUM_0);
    }
    if constexpr (CBLKLen > NUM_1) {
        LoadBNormNEON(src + svStep, res1, NUM_0, NUM_0);
    }
    if constexpr (CBLKLen > NUM_2) {
        LoadBNormNEON(src + svStep * NUM_2, res2, NUM_0, NUM_0);
    }
    if constexpr (CBLKLen > NUM_3) {
        LoadBNormNEON(src + svStep * NUM_3, res3, NUM_0, NUM_0);
    }
}

template <SizeType CBLKLen, typename OutType>
void StoreLoopUnrolNEON(OutType *dst, float32x4x2_t &out0, float32x4x2_t &out1,
    float32x4x2_t &out2, float32x4x2_t &out3)
{
    const SizeType svStep = NUM_2 * Vector::NeonLanes<OutType>();
    if constexpr (CBLKLen > NUM_0) {
        StoreBNormNEON(dst, out0);
    }
    if constexpr (CBLKLen > NUM_1) {
        StoreBNormNEON(dst + svStep, out1);
    }
    if constexpr (CBLKLen > NUM_2) {
        StoreBNormNEON(dst + svStep * NUM_2, out2);
    }
    if constexpr (CBLKLen > NUM_3) {
        StoreBNormNEON(dst + svStep * NUM_3, out3);
    }
}

template <SizeType CBLKLen>
void AddXLoopUnrolNEON(
    float32x4x2_t &znd0, float32x4x2_t &znd1, float32x4x2_t &znd2, float32x4x2_t &znd3,
    float32x4x2_t &zm0, float32x4x2_t &zm1, float32x4x2_t &zm2, float32x4x2_t &zm3)
{
    if constexpr (CBLKLen > NUM_0) {
        znd0.val[0] = vaddq_f32(znd0.val[0], zm0.val[0]);
        znd0.val[1] = vaddq_f32(znd0.val[1], zm0.val[1]);
    }
    if constexpr (CBLKLen > NUM_1) {
        znd1.val[0] = vaddq_f32(znd1.val[0], zm1.val[0]);
        znd1.val[1] = vaddq_f32(znd1.val[1], zm1.val[1]);
    }
    if constexpr (CBLKLen > NUM_2) {
        znd2.val[0] = vaddq_f32(znd2.val[0], zm2.val[0]);
        znd2.val[1] = vaddq_f32(znd2.val[1], zm2.val[1]);
    }
    if constexpr (CBLKLen > NUM_3) {
        znd3.val[0] = vaddq_f32(znd3.val[0], zm3.val[0]);
        znd3.val[1] = vaddq_f32(znd3.val[1], zm3.val[1]);
    }
}

template <SizeType CBLKLen>
void SubLoopUnrolNEON(
    float32x4x2_t &znd0, float32x4x2_t &znd1, float32x4x2_t &znd2, float32x4x2_t &znd3,
    float32x4x2_t &zm0, float32x4x2_t &zm1, float32x4x2_t &zm2, float32x4x2_t &zm3)
{
    if constexpr (CBLKLen > NUM_0) {
        znd0.val[0] = vsubq_f32(znd0.val[0], zm0.val[0]);
        znd0.val[1] = vsubq_f32(znd0.val[1], zm0.val[1]);
    }
    if constexpr (CBLKLen > NUM_1) {
        znd1.val[0] = vsubq_f32(znd1.val[0], zm1.val[0]);
        znd1.val[1] = vsubq_f32(znd1.val[1], zm1.val[1]);
    }
    if constexpr (CBLKLen > NUM_2) {
        znd2.val[0] = vsubq_f32(znd2.val[0], zm2.val[0]);
        znd2.val[1] = vsubq_f32(znd2.val[1], zm2.val[1]);
    }
    if constexpr (CBLKLen > NUM_3) {
        znd3.val[0] = vsubq_f32(znd3.val[0], zm3.val[0]);
        znd3.val[1] = vsubq_f32(znd3.val[1], zm3.val[1]);
    }
}

template <SizeType CBLKLen>
void MadXLoopUnrolNEON(
    float32x4x2_t &znd0, float32x4x2_t &znd1, float32x4x2_t &znd2, float32x4x2_t &znd3,
    float32x4x2_t &zm0, float32x4x2_t &zm1, float32x4x2_t &zm2, float32x4x2_t &zm3,
    float32x4x2_t &za0, float32x4x2_t &za1, float32x4x2_t &za2, float32x4x2_t &za3)
{
    if constexpr (CBLKLen > NUM_0) {
        znd0.val[0] = vfmaq_f32(za0.val[0], znd0.val[0], zm0.val[0]);
        znd0.val[1] = vfmaq_f32(za0.val[1], znd0.val[1], zm0.val[1]);
    }
    if constexpr (CBLKLen > NUM_1) {
        znd1.val[0] = vfmaq_f32(za1.val[0], znd1.val[0], zm1.val[0]);
        znd1.val[1] = vfmaq_f32(za1.val[1], znd1.val[1], zm1.val[1]);
    }
    if constexpr (CBLKLen > NUM_2) {
        znd2.val[0] = vfmaq_f32(za2.val[0], znd2.val[0], zm2.val[0]);
        znd2.val[1] = vfmaq_f32(za2.val[1], znd2.val[1], zm2.val[1]);
    }
    if constexpr (CBLKLen > NUM_3) {
        znd3.val[0] = vfmaq_f32(za3.val[0], znd3.val[0], zm3.val[0]);
        znd3.val[1] = vfmaq_f32(za3.val[1], znd3.val[1], zm3.val[1]);
    }
}

template <SizeType CBLKLen>
void MlaXLoopUnrolNEON(
    float32x4x2_t &zn0, float32x4x2_t &zn1, float32x4x2_t &zn2, float32x4x2_t &zn3,
    float32x4x2_t &zm0, float32x4x2_t &zm1, float32x4x2_t &zm2, float32x4x2_t &zm3,
    float32x4x2_t &zad0, float32x4x2_t &zad1, float32x4x2_t &zad2, float32x4x2_t &zad3)
{
    if constexpr (CBLKLen > NUM_0) {
        zad0.val[0] = vmlaq_f32(zad0.val[0], zn0.val[0], zm0.val[0]);
        zad0.val[1] = vmlaq_f32(zad0.val[1], zn0.val[1], zm0.val[1]);
    }
    if constexpr (CBLKLen > NUM_1) {
        zad1.val[0] = vmlaq_f32(zad1.val[0], zn1.val[0], zm1.val[0]);
        zad1.val[1] = vmlaq_f32(zad1.val[1], zn1.val[1], zm1.val[1]);
    }
    if constexpr (CBLKLen > NUM_2) {
        zad2.val[0] = vmlaq_f32(zad2.val[0], zn2.val[0], zm2.val[0]);
        zad2.val[1] = vmlaq_f32(zad2.val[1], zn2.val[1], zm2.val[1]);
    }
    if constexpr (CBLKLen > NUM_3) {
        zad3.val[0] = vmlaq_f32(zad3.val[0], zn3.val[0], zm3.val[0]);
        zad3.val[1] = vmlaq_f32(zad3.val[1], zn3.val[1], zm3.val[1]);
    }
}
 
template <SizeType CBLKLen>
void DupLoopUnrolNEON(const float &val, float32x4x2_t &znd0, float32x4x2_t &znd1,
    float32x4x2_t &znd2, float32x4x2_t &znd3)
{
    if constexpr (CBLKLen > NUM_0) {
        float32x4_t srcVec = vdupq_n_f32(val);
        znd0.val[0] = srcVec;
        znd0.val[1] = srcVec;
    }
    if constexpr (CBLKLen > NUM_1) {
        float32x4_t srcVec = vdupq_n_f32(val);
        znd1.val[0] = srcVec;
        znd1.val[1] = srcVec;
    }
    if constexpr (CBLKLen > NUM_2) {
        float32x4_t srcVec = vdupq_n_f32(val);
        znd2.val[0] = srcVec;
        znd2.val[1] = srcVec;
    }
    if constexpr (CBLKLen > NUM_3) {
        float32x4_t srcVec = vdupq_n_f32(val);
        znd3.val[0] = srcVec;
        znd3.val[1] = srcVec;
    }
}

template<typename InType, typename OutType>
inline void EvaluateBatchNormCrossFWDNEON(SizeType loopRange, const InType *src, OutType *dst,
    const float32x4x2_t &svMean, const float32x4x2_t &svMF, const float32x4x2_t &svShift,
    std::uint8_t *ws, Propagation pKind, bool fuseRelu)
{
    SizeType loopIdx = 0;
    SizeType svStep0 = 0;
    SizeType svStep1 = NUM_2 * Vector::NeonLanes<InType>();
    SizeType svStep2 = NUM_2 * Vector::NeonLanes<InType>() * 2;
    SizeType svStep3 = NUM_2 * Vector::NeonLanes<InType>() * 3;
    SizeType svStep4 = NUM_2 * Vector::NeonLanes<InType>() * 4;
    float32x4x2_t data0;
    float32x4x2_t data1;
    float32x4x2_t data2;
    float32x4x2_t data3;
    SizeType loopEnd = loopRange - loopRange % svStep4;
    for (; loopIdx < loopEnd; loopIdx += svStep4) {
        LoadBNormNEON(src + loopIdx + svStep0, data0, loopIdx + svStep0, loopIdx + svStep1);
        LoadBNormNEON(src + loopIdx + svStep1, data1, loopIdx + svStep1, loopIdx + svStep2);
        LoadBNormNEON(src + loopIdx + svStep2, data2, loopIdx + svStep2, loopIdx + svStep3);
        LoadBNormNEON(src + loopIdx + svStep3, data3, loopIdx + svStep3, loopIdx + svStep4);

        data0.val[0] = vsubq_f32(data0.val[0], svMean.val[0]);
        data0.val[1] = vsubq_f32(data0.val[1], svMean.val[1]);
        data1.val[0] = vsubq_f32(data1.val[0], svMean.val[0]);
        data1.val[1] = vsubq_f32(data1.val[1], svMean.val[1]);
        data2.val[0] = vsubq_f32(data2.val[0], svMean.val[0]);
        data2.val[1] = vsubq_f32(data2.val[1], svMean.val[1]);
        data3.val[0] = vsubq_f32(data3.val[0], svMean.val[0]);
        data3.val[1] = vsubq_f32(data3.val[1], svMean.val[1]);

        data0.val[0] = vmlaq_f32(svShift.val[0], data0.val[0], svMF.val[0]);
        data0.val[1] = vmlaq_f32(svShift.val[1], data0.val[1], svMF.val[1]);
        data1.val[0] = vmlaq_f32(svShift.val[0], data1.val[0], svMF.val[0]);
        data1.val[1] = vmlaq_f32(svShift.val[1], data1.val[1], svMF.val[1]);
        data2.val[0] = vmlaq_f32(svShift.val[0], data2.val[0], svMF.val[0]);
        data2.val[1] = vmlaq_f32(svShift.val[1], data2.val[1], svMF.val[1]);
        data3.val[0] = vmlaq_f32(svShift.val[0], data3.val[0], svMF.val[0]);
        data3.val[1] = vmlaq_f32(svShift.val[1], data3.val[1], svMF.val[1]);

        Service::Unused(ws, pKind, fuseRelu);
        StoreBNormNEON(dst + loopIdx + svStep0, data0);
        StoreBNormNEON(dst + loopIdx + svStep1, data1);
        StoreBNormNEON(dst + loopIdx + svStep2, data2);
        StoreBNormNEON(dst + loopIdx + svStep3, data3);
    }
    float32x4x2_t data;
    for (; loopIdx < loopRange; loopIdx += svStep1) {
        LoadBNormNEON(src + loopIdx, data, loopIdx, loopRange);
        data.val[0] = vsubq_f32(data.val[0], svMean.val[0]);
        data.val[1] = vsubq_f32(data.val[1], svMean.val[1]);
        data.val[0] = vmlaq_f32(svShift.val[0], data.val[0], svMF.val[0]);
        data.val[1] = vmlaq_f32(svShift.val[1], data.val[1], svMF.val[1]);
        StoreBNormNEON(dst + loopIdx, data);
    }
}

template <typename InType>
inline void BatchMeanFWDNEON(SizeType loopRange, const InType *src, float32x4x2_t &svMean)
{
    SizeType loopIdx = 0;
    SizeType svStep = NUM_2 * Vector::NeonLanes<InType>();
    for (; loopIdx < loopRange; loopIdx += svStep) {
        float32x4x2_t data;
        LoadBNormNEON(src + loopIdx, data, loopIdx, loopRange);
        svMean.val[0] = vaddq_f32(svMean.val[0], data.val[0]);
        svMean.val[1] = vaddq_f32(svMean.val[1], data.val[1]);
    }
}

template <typename InType>
inline void BatchVarianceFWDNEON(SizeType loopRange, const InType *src,
    const float32x4x2_t &svMean, float32x4x2_t &svVar)
{
    SizeType loopIdx = 0;
    SizeType svStep = NUM_2 * Vector::NeonLanes<InType>();
    for (; loopIdx < loopRange; loopIdx += svStep) {
        float32x4x2_t data;
        LoadBNormNEON(src + loopIdx, data, loopIdx, loopRange);
        data.val[0] = vsubq_f32(data.val[0], svMean.val[0]);
        data.val[1] = vsubq_f32(data.val[1], svMean.val[1]);
        svVar.val[0] = vmlaq_f32(svVar.val[0], data.val[0], data.val[0]);
        svVar.val[1] = vmlaq_f32(svVar.val[1], data.val[1], data.val[1]);
    }
}

} // namespace BNorm
} // namespace Detail
} // namespace KuDNN

#endif // KUDNN_BNORM_CMD_LOOP_UNROL_HPP