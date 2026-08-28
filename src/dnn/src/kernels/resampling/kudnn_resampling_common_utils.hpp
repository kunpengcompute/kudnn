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

#ifndef KUDNN_RESAMPLING_COMMON_UTILS_HPP
#define KUDNN_RESAMPLING_COMMON_UTILS_HPP

#include <arm_sve.h>
#include <arm_neon.h>

#include <cmath>

#include "kudnn.hpp"
#include "internal_funcs.hpp"

namespace KuDNN {
namespace Detail {
namespace Resampling {

// Float zeros.
constexpr float F32_ZERO = 0.0f;
constexpr __fp16 F16_ZERO = static_cast<__fp16>(F32_ZERO);
const __bf16 BF16_ZERO = vcvth_bf16_f32(F32_ZERO);
// Int zeros
constexpr int32_t S32_ZERO = 0;
constexpr int8_t S8_ZERO = static_cast<int8_t>(S32_ZERO);
constexpr int16_t S16_ZERO = static_cast<int16_t>(S32_ZERO);
// Uint zeros.
constexpr uint32_t U32_ZERO = 0u;
constexpr uint8_t U8_ZERO = static_cast<uint8_t>(U32_ZERO);
constexpr uint16_t U16_ZERO = static_cast<uint16_t>(S32_ZERO);

class StridesInfo {
public:
IntType strideIMB;
    IntType strideIC;
    IntType strideID;
    IntType strideIH;
    IntType strideIW;
    IntType strideOMB;
    IntType strideOC;
    IntType strideOD;
    IntType strideOH;
    IntType strideOW;

    StridesInfo(const Shape &stridesIn, const Shape &stridesOut)
        : strideIMB(static_cast<IntType>(stridesIn[IDX_0])),
        strideIC(static_cast<IntType>(stridesIn[IDX_1])),
        strideID(static_cast<IntType>(stridesIn[IDX_2])),
        strideIH(static_cast<IntType>(stridesIn[IDX_3])),
        strideIW(static_cast<IntType>(stridesIn[IDX_4])),
        strideOMB(static_cast<IntType>(stridesOut[IDX_0])),
        strideOC(static_cast<IntType>(stridesOut[IDX_1])),
        strideOD(static_cast<IntType>(stridesOut[IDX_2])),
        strideOH(static_cast<IntType>(stridesOut[IDX_3])),
        strideOW(static_cast<IntType>(stridesOut[IDX_4])) {}

    ~StridesInfo() noexcept {}
};

template <typename T>
class SplitDimInfo {
public:
    T mbStart;
    T mbEnd;
    T cStart;
    T cEnd;
    T dStart;
    T dEnd;
    T hStart;
    T hEnd;
    T wStart;
    T wEnd;

    SplitDimInfo(const Shape &batchSplitDims, const Shape &stride, bool isAxb = false)
    {
        if (!isAxb) {
            mbEnd = static_cast<T>(batchSplitDims[IDX_0] + stride[IDX_0]);
            cEnd = static_cast<T>(batchSplitDims[IDX_1] + stride[IDX_1]);
            dEnd = static_cast<T>(batchSplitDims[IDX_2] + stride[IDX_2]);
            hEnd = static_cast<T>(batchSplitDims[IDX_3] + stride[IDX_3]);
            wEnd = static_cast<T>(batchSplitDims[IDX_4] + stride[IDX_4]);
            mbStart = static_cast<T>(stride[IDX_0]);
            cStart = static_cast<T>(stride[IDX_1]);
            dStart = static_cast<T>(stride[IDX_2]);
            hStart = static_cast<T>(stride[IDX_3]);
            wStart = static_cast<T>(stride[IDX_4]);
        } else {
            mbEnd = static_cast<T>(batchSplitDims[IDX_0] + stride[IDX_0]);
            dEnd = static_cast<T>(batchSplitDims[IDX_1] + stride[IDX_1]);
            hEnd = static_cast<T>(batchSplitDims[IDX_2] + stride[IDX_2]);
            wEnd = static_cast<T>(batchSplitDims[IDX_3] + stride[IDX_3]);
            cEnd = static_cast<T>(batchSplitDims[IDX_4] + stride[IDX_4]);
            mbStart = static_cast<T>(stride[IDX_0]);
            dStart = static_cast<T>(stride[IDX_1]);
            hStart = static_cast<T>(stride[IDX_2]);
            wStart = static_cast<T>(stride[IDX_3]);
            cStart = static_cast<T>(stride[IDX_4]);
        }
    }

    ~SplitDimInfo() noexcept {}
};

// Round and Saturate from SrcType(f32/s64) value to DstType(s32/s8/u8).
template <typename SrcType, typename DstType>
inline DstType RoundAndSaturate(SrcType val)
{
    SrcType res = val;
    SrcType maxVal = static_cast<SrcType>(std::numeric_limits<DstType>::max());
    SrcType minVal = static_cast<SrcType>(std::numeric_limits<DstType>::min());
    if constexpr (std::is_same_v<SrcType, float>) {
        res = roundf(res);
    }
    res = std::min(res, maxVal);
    res = std::max(res, minVal);
    return static_cast<DstType>(res);
}

// Load a value from src, convert its type to f32.
template <typename SrcType>
inline float LoadF32FromAddress(const SrcType *src)
{
    if constexpr (std::is_same_v<SrcType, float>) {
        return static_cast<float>(*src);
    } else if constexpr (std::is_same_v<SrcType, __fp16>) {
        return static_cast<float>(*src);
    }  else if constexpr (std::is_same_v<SrcType, __bf16>) {
        return vcvtah_f32_bf16(*src);
    } else if constexpr (std::is_same_v<SrcType, int32_t>) {
        return static_cast<float>(*src);
    } else if constexpr (std::is_same_v<SrcType, int8_t>) {
        return static_cast<float>(*src);
    } else if constexpr (std::is_same_v<SrcType, uint8_t>) {
        return static_cast<float>(*src);
    } else {
        return static_cast<float>(*src);
    }
}

// Store a f32 value to dst.
template <typename DstType>
inline void StoreF32ToAddress(DstType *dst, float f32Val)
{
    if constexpr (std::is_same_v<DstType, float>) {
        (*dst) = f32Val;
    } else if constexpr (std::is_same_v<DstType, __fp16>) {
        (*dst) = static_cast<__fp16>(f32Val);
    }  else if constexpr (std::is_same_v<DstType, __bf16>) {
        (*dst) = vcvth_bf16_f32(f32Val);
    } else if constexpr (std::is_same_v<DstType, int32_t>) {
        (*dst) = RoundAndSaturate<float, int32_t>(f32Val);
    } else if constexpr (std::is_same_v<DstType, int8_t>) {
        (*dst) = RoundAndSaturate<float, int8_t>(f32Val);
    } else if constexpr (std::is_same_v<DstType, uint8_t>) {
        (*dst) = RoundAndSaturate<float, uint8_t>(f32Val);
    }
}

// Assign a value from SrcType to DstType.
template <typename SrcType, typename DstType>
inline void Assign(DstType *dst, const SrcType *src)
{
    if constexpr (std::is_same_v<SrcType, DstType>) {
        *dst = *src;
        return ;
    }
    float val = LoadF32FromAddress(src);
    StoreF32ToAddress(dst, val);
}

// Use gather to load values as a f32 vector from src.
template <typename SrcType>
inline svfloat32_t GatherLoadSvF32FromAddress(const svbool_t &pg, const SrcType *src, const svint32_t &idxs)
{
    if constexpr (std::is_same_v<SrcType, float>) {
        return svld1_gather_s32index_f32(pg, src, idxs);
    } else if constexpr (std::is_same_v<SrcType, __fp16>) {
        const int16_t *srcPtrS16 = static_cast<const int16_t *>(static_cast<const void *>(src));
        svint32_t s32Vals = svld1sh_gather_s32index_s32(pg, srcPtrS16, idxs);
        return svcvt_f32_f16_x(pg, svreinterpret_f16_s32(s32Vals));
    } else if constexpr (std::is_same_v<SrcType, __bf16>) {
        const int16_t *srcPtrS16 = static_cast<const int16_t *>(static_cast<const void *>(src));
        svint32_t s32Vals = svld1sh_gather_s32index_s32(pg, srcPtrS16, idxs);
        return svreinterpret_f32_s32(svlsl_n_s32_m(pg, s32Vals, static_cast<uint32_t>(NUM_16)));
    } else if constexpr (std::is_same_v<SrcType, int32_t>) {
        return svcvt_f32_s32_x(pg, svld1_gather_s32index_s32(pg, src, idxs));
    } else if constexpr (std::is_same_v<SrcType, int8_t>) {
        return svcvt_f32_s32_x(pg, svld1sb_gather_s32offset_s32(pg, src, idxs));
    } else if constexpr (std::is_same_v<SrcType, uint8_t>) {
        return svcvt_f32_u32_x(pg, svld1ub_gather_s32offset_u32(pg, src, idxs));
    }
}

// Load values as a f32 vector from src.
template <typename SrcType>
inline svfloat32_t LoadSvF32FromAddress(const svbool_t &pg, const SrcType *src)
{
    if constexpr (std::is_same_v<SrcType, float>) {
        return svld1_f32(pg, src);
    } else if constexpr (std::is_same_v<SrcType, __fp16>) {
        const int16_t *srcPtrS16 = static_cast<const int16_t *>(static_cast<const void *>(src));
        svint32_t s32Vals = svld1sh_s32(pg, srcPtrS16);
        return svcvt_f32_f16_x(pg, svreinterpret_f16_s32(s32Vals));
    }  else if constexpr (std::is_same_v<SrcType, __bf16>) {
        const int16_t *srcPtrS16 = static_cast<const int16_t *>(static_cast<const void *>(src));
        svint32_t s32Vals = svld1sh_s32(pg, srcPtrS16);
        return svreinterpret_f32_s32(svlsl_n_s32_m(pg, s32Vals, static_cast<uint32_t>(NUM_16)));
    } else if constexpr (std::is_same_v<SrcType, int32_t>) {
        return svcvt_f32_s32_x(pg, svld1_s32(pg, src));
    } else if constexpr (std::is_same_v<SrcType, int8_t>) {
        return svcvt_f32_s32_x(pg, svld1sb_s32(pg, src));
    } else if constexpr (std::is_same_v<SrcType, uint8_t>) {
        return svcvt_f32_u32_x(pg, svld1ub_u32(pg, src));
    }
}

// Store values in a f32 vectors to dst.
template <typename DstType>
inline void StoreSvF32ToAddress(const svbool_t &pg, DstType *dst, svfloat32_t f32Vals)
{
    if constexpr (std::is_same_v<DstType, float>) {
        svst1_f32(pg, dst, f32Vals);
    } else if constexpr (std::is_same_v<DstType, __fp16>) {
        svfloat16_t f16Vals = svcvt_f16_f32_x(pg, f32Vals);
        svint32_t s32Vals = svreinterpret_s32_f16(f16Vals);
        svst1h_s32(pg, static_cast<int16_t *>(static_cast<void *>(dst)), s32Vals);
    }  else if constexpr (std::is_same_v<DstType, __bf16>) {
        svbfloat16_t bf16Vals = svcvt_bf16_f32_x(pg, f32Vals);
        svint32_t s32Vals = svreinterpret_s32_bf16(bf16Vals);
        svst1h_s32(pg, static_cast<int16_t *>(static_cast<void *>(dst)), s32Vals);
    } else if constexpr (std::is_same_v<DstType, int32_t>) {
        svint32_t s32Vals = svcvt_s32_f32_x(pg, svrinta_f32_x(pg, f32Vals));
        s32Vals = svmin_n_s32_m(pg, s32Vals, std::numeric_limits<std::int32_t>::max());
        s32Vals = svmax_n_s32_m(pg, s32Vals, std::numeric_limits<std::int32_t>::min());
        svst1_s32(pg, dst, s32Vals);
    } else if constexpr (std::is_same_v<DstType, int8_t>) {
        svint32_t s32Vals = svcvt_s32_f32_x(pg, svrinta_f32_x(pg, f32Vals));
        s32Vals = svmin_n_s32_m(pg, s32Vals, static_cast<int32_t>(std::numeric_limits<std::int8_t>::max()));
        s32Vals = svmax_n_s32_m(pg, s32Vals, static_cast<int32_t>(std::numeric_limits<std::int8_t>::min()));
        svst1b_s32(pg, dst, s32Vals);
    } else if constexpr (std::is_same_v<DstType, uint8_t>) {
        svbool_t pgLtZero = svcmplt_n_f32(pg, f32Vals, F32_ZERO);
        svuint32_t u32Vals = svcvt_u32_f32_x(pg, svrinta_f32_x(pg, svdup_f32_m(f32Vals, pgLtZero, F32_ZERO)));
        u32Vals = svmin_n_u32_m(pg, u32Vals, static_cast<uint32_t>(std::numeric_limits<std::uint8_t>::max()));
        u32Vals = svmax_n_u32_m(pg, u32Vals, static_cast<uint32_t>(std::numeric_limits<std::uint8_t>::min()));
        svst1b_u32(pg, static_cast<uint8_t *>(static_cast<void *>(dst)), u32Vals);
    }
}

// Use gather to load values and store them as vector, src and dst are same type.
template <typename DataType>
inline void GatherLoadAndStoreSv(const svbool_t &pg, DataType *dst, const DataType *src, const svint32_t &idxs)
{
    if constexpr (std::is_same_v<DataType, float>) {
        svfloat32_t res = GatherLoadSvF32FromAddress(pg, src, idxs);
        StoreSvF32ToAddress(pg, dst, res);
    } else if constexpr (std::is_same_v<DataType, __fp16>) {
        svfloat32_t res = GatherLoadSvF32FromAddress(pg, src, idxs);
        StoreSvF32ToAddress(pg, dst, res);
    } else if constexpr (std::is_same_v<DataType, __bf16>) {
        svfloat32_t res = GatherLoadSvF32FromAddress(pg, src, idxs);
        StoreSvF32ToAddress(pg, dst, res);
    } else if constexpr (std::is_same_v<DataType, int32_t>) {
        svint32_t res = svld1_gather_s32index_s32(pg, src, idxs);
        svst1_s32(pg, dst, res);
    } else if constexpr (std::is_same_v<DataType, int8_t>) {
        svint32_t res = svld1sb_gather_s32offset_s32(pg, src, idxs);
        svst1b_s32(pg, dst, res);
    } else if constexpr (std::is_same_v<DataType, uint8_t>) {
        svuint32_t res = svld1ub_gather_s32offset_u32(pg, src, idxs);
        svst1b_u32(pg, dst, res);
    }
}

// Load values and store them as vector, src and dst are same type.
template <typename DataType>
inline void LoadAndStoreSv(const svbool_t &pg, DataType *dst, const DataType *src)
{
    if constexpr (std::is_same_v<DataType, float>) {
        svfloat32_t res = LoadSvF32FromAddress(pg, src);
        StoreSvF32ToAddress(pg, dst, res);
    } else if constexpr (std::is_same_v<DataType, __fp16>) {
        svfloat32_t res = LoadSvF32FromAddress(pg, src);
        StoreSvF32ToAddress(pg, dst, res);
    } else if constexpr (std::is_same_v<DataType, __bf16>) {
        svfloat32_t res = LoadSvF32FromAddress(pg, src);
        StoreSvF32ToAddress(pg, dst, res);
    } else if constexpr (std::is_same_v<DataType, int32_t>) {
        svint32_t res = svld1_s32(pg, src);
        svst1_s32(pg, dst, res);
    } else if constexpr (std::is_same_v<DataType, int8_t>) {
        svint32_t res = svld1sb_s32(pg, src);
        svst1b_s32(pg, dst, res);
    } else if constexpr (std::is_same_v<DataType, uint8_t>) {
        svuint32_t res = svld1ub_u32(pg, src);
        svst1b_u32(pg, dst, res);
    }
}

} // Resampling
} // Detail
} // KuDNN

#endif //  KUDNN_RESAMPLING_COMMON_UTILS_HPP
