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
#ifndef KUDNN_BNORM_CMD_LOOP_UNROL_HPP
#define KUDNN_BNORM_CMD_LOOP_UNROL_HPP

#include "kudnn.hpp"

namespace KuDNN {
namespace Detail {
namespace BNorm {

template <typename InType>
inline void LoadBNorm(const InType *src, svbool_t pg, svfloat32_t &out,
    const SizeType &loopIdx, const SizeType &loopRange)
{
    if constexpr (std::is_same_v<InType, float>) {
        out = svld1_f32(pg, src);
    } else if constexpr (std::is_same_v<InType, __fp16>) {
        svbool_t ldPg = svwhilelt_b16_u64(loopIdx, loopRange);
        svfloat16_t tmp = svld1_f16(ldPg, src);
        svfloat16_t zeros = svdup_n_f16(0);
        svfloat16_t tmp0 = svzip1(tmp, zeros);
        out = svcvt_f32_f16_x(pg, tmp0);
    } else if constexpr (std::is_same_v<InType, __bf16>) {
        svbool_t ldPg = svwhilelt_b16_u64(loopIdx, loopRange);
        svbfloat16_t tmp = svld1_bf16(ldPg, src);
        svbfloat16_t zeros = svdup_n_bf16(vcvth_bf16_f32(0.0f));
        svbfloat16_t tmp0 = svzip1(zeros, tmp);
        out = svreinterpret_f32_bf16(tmp0);
    } else if constexpr (std::is_same_v<InType, std::int8_t>) {
        out = svcvt_f32_s32_z(pg, svld1sb_s32(pg, src));
    }
}

template <typename InType>
inline svfloat32_t LoadBNormBwd(const svbool_t &pg, const InType *src)
{
    if constexpr (std::is_same_v<InType, float>) {
        return svld1_f32(pg, src);
    } else if constexpr (std::is_same_v<InType, __fp16>) {
        const int16_t *srcPtrS16 = static_cast<const int16_t *>(static_cast<const void *>(src));
        svint32_t s32Vals = svld1sh_s32(pg, srcPtrS16);
        return svcvt_f32_f16_x(pg, svreinterpret_f16_s32(s32Vals));
    } else if constexpr (std::is_same_v<InType, __bf16>) {
        const int16_t *srcPtrS16 = static_cast<const int16_t *>(static_cast<const void *>(src));
        svint32_t s32Vals = svld1sh_s32(pg, srcPtrS16);
        return svreinterpret_f32_s32(svlsl_n_s32_x(pg, s32Vals, static_cast<uint32_t>(NUM_16)));
    } else if constexpr (std::is_same_v<InType, std::uint8_t>) {
        return svcvt_f32_u32_x(pg, svld1ub_u32(pg, src));
    }
    return svdup_f32_x(svptrue_b32(), 0.0f);
}

template <typename OutType>
inline void StoreBNorm(OutType *dst, svbool_t pg, const svfloat32_t &svOut)
{
    if constexpr (std::is_same_v<OutType, float>) {
        svst1_f32(pg, dst, svOut);
    } else if constexpr (std::is_same_v<OutType, __fp16>) {
        svfloat16_t tmp0 = svcvt_f16_f32_z(pg, svOut);
        svint32_t tmp1 = svreinterpret_s32_f16(tmp0);
        svst1h_s32(pg, reinterpret_cast<int16_t*>(dst), tmp1);
    } else if constexpr (std::is_same_v<OutType, __bf16>) {
        svbfloat16_t tmp0 = svcvt_bf16_f32_z(pg, svOut);
        svint32_t tmp1 = svreinterpret_s32_bf16(tmp0);
        svst1h_s32(pg, reinterpret_cast<int16_t*>(dst), tmp1);
    } else if constexpr (std::is_same_v<OutType, std::int8_t>) {
        svint32_t tmp = svcvt_s32_z(pg, svrinti_f32_x(pg, svOut));
        tmp = svmax_n_s32_z(pg, tmp, std::numeric_limits<std::int8_t>::min());
        tmp = svmin_n_s32_z(pg, tmp, std::numeric_limits<std::int8_t>::max());
        svst1b_s32(pg, dst, tmp);
    }
}

template <SizeType CBLKLen>
void InitPgLoopUnrol(const SizeType &tailSize, svbool_t &pg0, svbool_t &pg1, svbool_t &pg2, svbool_t &pg3)
{
    if constexpr (CBLKLen > NUM_1) {
        pg0 = svptrue_b32();
    } else {
        pg0 = svwhilelt_b32_u64(NUM_0, tailSize);
    }
    if constexpr (CBLKLen > NUM_2) {
        pg1 = svptrue_b32();
    } else {
        pg1 = svwhilelt_b32_u64(NUM_0, tailSize);
    }
    if constexpr (CBLKLen > NUM_3) {
        pg2 = svptrue_b32();
    } else {
        pg2 = svwhilelt_b32_u64(NUM_0, tailSize);
    }
    if constexpr (CBLKLen == NUM_4) {
        pg3 = svwhilelt_b32_u64(NUM_0, tailSize);
    }
}

template <SizeType CBLKLen, typename InType>
void LoadLoopUnrol(const InType *src, svbool_t &pg0, svbool_t &pg1, svbool_t &pg2, svbool_t &pg3,
    svfloat32_t &res0, svfloat32_t &res1, svfloat32_t &res2, svfloat32_t &res3)
{
    const SizeType svStep = svcntw();
    svbool_t pgAll = svptrue_b32();
    if constexpr (CBLKLen > NUM_0) {
        LoadBNorm(src, pg0, res0, NUM_0, svcntp_b32(pgAll, pg0));
    }
    if constexpr (CBLKLen > NUM_1) {
        LoadBNorm(src + svStep, pg1, res1, NUM_0, svcntp_b32(pgAll, pg1));
    }
    if constexpr (CBLKLen > NUM_2) {
        LoadBNorm(src + svStep * NUM_2, pg2, res2, NUM_0, svcntp_b32(pgAll, pg2));
    }
    if constexpr (CBLKLen > NUM_3) {
        LoadBNorm(src + svStep * NUM_3, pg3, res3, NUM_0, svcntp_b32(pgAll, pg3));
    }
}

template <SizeType CBLKLen, typename OutType>
void StoreLoopUnrol(OutType *dst, svbool_t &pg0, svbool_t &pg1, svbool_t &pg2, svbool_t &pg3,
    svfloat32_t &out0, svfloat32_t &out1, svfloat32_t &out2, svfloat32_t &out3)
{
    const SizeType svStep = svcntw();
    if constexpr (CBLKLen > NUM_0) {
        StoreBNorm(dst, pg0, out0);
    }
    if constexpr (CBLKLen > NUM_1) {
        StoreBNorm(dst + svStep, pg1, out1);
    }
    if constexpr (CBLKLen > NUM_2) {
        StoreBNorm(dst + svStep * NUM_2, pg2, out2);
    }
    if constexpr (CBLKLen > NUM_3) {
        StoreBNorm(dst + svStep * NUM_3, pg3, out3);
    }
}

template <SizeType CBLKLen>
void AddXLoopUnrol(svbool_t &pg0, svbool_t &pg1, svbool_t &pg2, svbool_t &pg3,
    svfloat32_t &znd0, svfloat32_t &znd1, svfloat32_t &znd2, svfloat32_t &znd3,
    svfloat32_t &zm0, svfloat32_t &zm1, svfloat32_t &zm2, svfloat32_t &zm3)
{
    if constexpr (CBLKLen > NUM_0) {
        znd0 = svadd_x(pg0, znd0, zm0);
    }
    if constexpr (CBLKLen > NUM_1) {
        znd1 = svadd_x(pg1, znd1, zm1);
    }
    if constexpr (CBLKLen > NUM_2) {
        znd2 = svadd_x(pg2, znd2, zm2);
    }
    if constexpr (CBLKLen > NUM_3) {
        znd3 = svadd_x(pg3, znd3, zm3);
    }
}

template <SizeType CBLKLen>
void SubLoopUnrol(svbool_t &pg0, svbool_t &pg1, svbool_t &pg2, svbool_t &pg3,
    svfloat32_t &znd0, svfloat32_t &znd1, svfloat32_t &znd2, svfloat32_t &znd3,
    svfloat32_t &zm0, svfloat32_t &zm1, svfloat32_t &zm2, svfloat32_t &zm3)
{
    if constexpr (CBLKLen > NUM_0) {
        znd0 = svsub_f32_x(pg0, znd0, zm0);
    }
    if constexpr (CBLKLen > NUM_1) {
        znd1 = svsub_f32_x(pg1, znd1, zm1);
    }
    if constexpr (CBLKLen > NUM_2) {
        znd2 = svsub_f32_x(pg2, znd2, zm2);
    }
    if constexpr (CBLKLen > NUM_3) {
        znd3 = svsub_f32_x(pg3, znd3, zm3);
    }
}

template <SizeType CBLKLen>
void MadXLoopUnrol(svbool_t &pg0, svbool_t &pg1, svbool_t &pg2, svbool_t &pg3,
    svfloat32_t &znd0, svfloat32_t &znd1, svfloat32_t &znd2, svfloat32_t &znd3,
    svfloat32_t &zm0, svfloat32_t &zm1, svfloat32_t &zm2, svfloat32_t &zm3,
    svfloat32_t &za0, svfloat32_t &za1, svfloat32_t &za2, svfloat32_t &za3)
{
    if constexpr (CBLKLen > NUM_0) {
        znd0 = svmad_f32_x(pg0, znd0, zm0, za0);
    }
    if constexpr (CBLKLen > NUM_1) {
        znd1 = svmad_f32_x(pg1, znd1, zm1, za1);
    }
    if constexpr (CBLKLen > NUM_2) {
        znd2 = svmad_f32_x(pg2, znd2, zm2, za2);
    }
    if constexpr (CBLKLen > NUM_3) {
        znd3 = svmad_f32_x(pg3, znd3, zm3, za3);
    }
}

template <SizeType CBLKLen>
void MlaXLoopUnrol(svbool_t &pg0, svbool_t &pg1, svbool_t &pg2, svbool_t &pg3,
    svfloat32_t &zn0, svfloat32_t &zn1, svfloat32_t &zn2, svfloat32_t &zn3,
    svfloat32_t &zm0, svfloat32_t &zm1, svfloat32_t &zm2, svfloat32_t &zm3,
    svfloat32_t &zad0, svfloat32_t &zad1, svfloat32_t &zad2, svfloat32_t &zad3)
{
    if constexpr (CBLKLen > NUM_0) {
        zad0 = svmla_f32_x(pg0, zad0, zn0, zm0);
    }
    if constexpr (CBLKLen > NUM_1) {
        zad1 = svmla_f32_x(pg1, zad1, zn1, zm1);
    }
    if constexpr (CBLKLen > NUM_2) {
        zad2 = svmla_f32_x(pg2, zad2, zn2, zm2);
    }
    if constexpr (CBLKLen > NUM_3) {
        zad3 = svmla_f32_x(pg3, zad3, zn3, zm3);
    }
}

template <SizeType CBLKLen>
void DivnXLoopUnrol(const float &val, svbool_t &pg0, svbool_t &pg1, svbool_t &pg2, svbool_t &pg3,
    svfloat32_t &znd0, svfloat32_t &znd1, svfloat32_t &znd2, svfloat32_t &znd3)
{
    if constexpr (CBLKLen > NUM_0) {
        znd0 = svdiv_n_f32_x(pg0, znd0, val);
    }
    if constexpr (CBLKLen > NUM_1) {
        znd1 = svdiv_n_f32_x(pg1, znd1, val);
    }
    if constexpr (CBLKLen > NUM_2) {
        znd2 = svdiv_n_f32_x(pg2, znd2, val);
    }
    if constexpr (CBLKLen > NUM_3) {
        znd3 = svdiv_n_f32_x(pg3, znd3, val);
    }
}

template <SizeType CBLKLen>
void DupLoopUnrol(const float &val, svfloat32_t &znd0, svfloat32_t &znd1, svfloat32_t &znd2,
    svfloat32_t &znd3)
{
    if constexpr (CBLKLen > NUM_0) {
        znd0 = svdup_f32(val);
    }
    if constexpr (CBLKLen > NUM_1) {
        znd1 = svdup_f32(val);
    }
    if constexpr (CBLKLen > NUM_2) {
        znd2 = svdup_f32(val);
    }
    if constexpr (CBLKLen > NUM_3) {
        znd3 = svdup_f32(val);
    }
}

template<typename InType, typename OutType>
inline void EvaluateBatchNormCrossFWD(SizeType loopRange, const InType *src, OutType *dst,
    const svfloat32_t &svMean, const svfloat32_t &svMF, const svfloat32_t &svShift,
    std::uint8_t *ws, Propagation pKind, bool fuseRelu)
{
    SizeType loopIdx = 0;
    SizeType svStep0 = 0;
    SizeType svStep1 = svcntw();
    SizeType svStep2 = svcntw() * 2;
    SizeType svStep3 = svcntw() * 3;
    SizeType svStep4 = svcntw() * 4;
    svfloat32_t data0;
    svfloat32_t data1;
    svfloat32_t data2;
    svfloat32_t data3;
    SizeType loopEnd = loopRange - loopRange % svStep4;
    for (; loopIdx < loopEnd; loopIdx += svStep4) {
        svbool_t pg = svptrue_b32();
        LoadBNorm(src + loopIdx + svStep0, pg, data0, loopIdx + svStep0, loopIdx + svStep1);
        LoadBNorm(src + loopIdx + svStep1, pg, data1, loopIdx + svStep1, loopIdx + svStep2);
        LoadBNorm(src + loopIdx + svStep2, pg, data2, loopIdx + svStep2, loopIdx + svStep3);
        LoadBNorm(src + loopIdx + svStep3, pg, data3, loopIdx + svStep3, loopIdx + svStep4);

        data0 = svsub_f32_x(pg, data0, svMean);
        data1 = svsub_f32_x(pg, data1, svMean);
        data2 = svsub_f32_x(pg, data2, svMean);
        data3 = svsub_f32_x(pg, data3, svMean);

        data0 = svmad_f32_x(pg, data0, svMF, svShift);
        data1 = svmad_f32_x(pg, data1, svMF, svShift);
        data2 = svmad_f32_x(pg, data2, svMF, svShift);
        data3 = svmad_f32_x(pg, data3, svMF, svShift);

        if (fuseRelu) {
            svfloat32_t zero = svdup_n_f32(0.0f);
            svbool_t wsPg0 = svcmple(pg, data0, zero);
            svbool_t wsPg1 = svcmple(pg, data1, zero);
            svbool_t wsPg2 = svcmple(pg, data2, zero);
            svbool_t wsPg3 = svcmple(pg, data3, zero);
            data0 = svmul_m(wsPg0, data0, zero);
            data1 = svmul_m(wsPg1, data1, zero);
            data2 = svmul_m(wsPg2, data2, zero);
            data3 = svmul_m(wsPg3, data3, zero);
            if (pKind != Propagation::FORWARD_INFERENCE) {
                svuint32_t ones = svdup_n_u32(1);
                svuint32_t svWs0 = svdup_n_u32_m(ones, wsPg0, 0);
                svuint32_t svWs1 = svdup_n_u32_m(ones, wsPg1, 0);
                svuint32_t svWs2 = svdup_n_u32_m(ones, wsPg2, 0);
                svuint32_t svWs3 = svdup_n_u32_m(ones, wsPg3, 0);
                svst1b_u32(pg, ws + loopIdx + svStep0, svWs0);
                svst1b_u32(pg, ws + loopIdx + svStep1, svWs1);
                svst1b_u32(pg, ws + loopIdx + svStep2, svWs2);
                svst1b_u32(pg, ws + loopIdx + svStep3, svWs3);
            }
        }
        StoreBNorm(dst + loopIdx + svStep0, pg, data0);
        StoreBNorm(dst + loopIdx + svStep1, pg, data1);
        StoreBNorm(dst + loopIdx + svStep2, pg, data2);
        StoreBNorm(dst + loopIdx + svStep3, pg, data3);
    }
    svfloat32_t data;
    for (; loopIdx < loopRange; loopIdx += svStep1) {
        svbool_t pg = svwhilelt_b32_u64(loopIdx, loopRange);
        LoadBNorm(src + loopIdx, pg, data, loopIdx, loopRange);
        data = svsub_f32_x(pg, data, svMean);
        data = svmla_f32_x(pg, svShift, data, svMF);
        if (fuseRelu) {
            svfloat32_t zero = svdup_n_f32(0.0f);
            svbool_t wsPg = svcmple(pg, data, zero);
            data = svmul_m(wsPg, data, zero);
            if (pKind != Propagation::FORWARD_INFERENCE) {
                svuint32_t ones = svdup_n_u32(1);
                svuint32_t svWs = svdup_n_u32_m(ones, wsPg, 0);
                svst1b_u32(pg, ws + loopIdx, svWs);
            }
        }
        StoreBNorm(dst + loopIdx, pg, data);
    }
}

// Tool functions for bnorm fwd.
template <typename InType>
SizeType GetDataTypeId()
{
    if constexpr (std::is_same_v<InType, float>) {
        return IDX_1;
    } else if constexpr (std::is_same_v<InType, __fp16>) {
        return IDX_2;
    } else if constexpr (std::is_same_v<InType, __bf16>) {
        return IDX_3;
    } else if constexpr (std::is_same_v<InType, std::int8_t>) {
        return IDX_4;
    }
}

template <typename InType>
SizeType GetSvlen()
{
    if constexpr (std::is_same_v<InType, float>) {
        return svcntw();
    }
}

template <typename InType>
inline void BatchMeanFWD(SizeType loopRange, const InType *src, svfloat32_t &svMean)
{
    SizeType loopIdx = 0;
    SizeType svStep = svcntw();
    for (; loopIdx < loopRange; loopIdx += svStep) {
        svbool_t pg = svwhilelt_b32_u64(loopIdx, loopRange);
        svfloat32_t data;
        LoadBNorm(src + loopIdx, pg, data, loopIdx, loopRange);
        svMean = svadd_m(pg, svMean, data);
    }
}

template <typename InType>
inline void BatchVarianceFWD(SizeType loopRange, const InType *src, const svfloat32_t &svMean, svfloat32_t &svVar)
{
    SizeType loopIdx = 0;
    SizeType svStep = svcntw();
    for (; loopIdx < loopRange; loopIdx += svStep) {
        svbool_t pg = svwhilelt_b32_u64(loopIdx, loopRange);
        svfloat32_t data;
        LoadBNorm(src + loopIdx, pg, data, loopIdx, loopRange);
        data = svsub_x(pg, data, svMean);
        svVar = svmla_m(pg, svVar, data, data);
    }
}

constexpr SizeType TAG_SIZE = NUM_4; // one tag is 4 channel blocks

struct CBLKParams {
    SizeType cblkRange;
    SizeType tagTail;
    SizeType tailSize;
};

inline void CalCBLKParams(CBLKParams &cblkp, const SizeType &channelSize, const SizeType &svStep)
{
    cblkp.tailSize = channelSize % svStep;
    SizeType cblkSize = (channelSize + svStep - 1) / svStep;
    cblkp.tailSize = (cblkp.tailSize == 0) ? svStep : cblkp.tailSize;
    cblkp.cblkRange = (cblkSize / TAG_SIZE) * TAG_SIZE ;
    cblkp.tagTail = cblkSize - cblkp.cblkRange;
    if (cblkp.tagTail == 0 && cblkSize != 0) {
        cblkp.cblkRange -= TAG_SIZE;
        cblkp.tagTail = TAG_SIZE;
    }
}

template<typename InType>
constexpr bool IsExecute()
{
    return ((std::is_same_v<InType, float>) || (std::is_same_v<InType, __fp16>) ||
        (std::is_same_v<InType, __bf16>) || (std::is_same_v<InType, std::int8_t>));
}

} // namespace BNorm
} // namespace Detail
} // namespace KuDNN

#endif // KUDNN_BNORM_CMD_LOOP_UNROL_HPP