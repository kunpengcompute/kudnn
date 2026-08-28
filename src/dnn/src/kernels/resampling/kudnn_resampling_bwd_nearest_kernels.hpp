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

#ifndef KUDNN_RESAMPLING_BWD_NEAREST_KERNELS_HPP
#define KUDNN_RESAMPLING_BWD_NEAREST_KERNELS_HPP

#include "kudnn_resampling_bwd_utils.hpp"

namespace KuDNN {
namespace Detail {
namespace Resampling {

template <typename DiffDstType, typename DiffSrcType>
struct NearestVector {
    using Type = svfloat32_t;
};
template <>
struct NearestVector<int32_t, int32_t> {
    using Type = svint32_t;
};
template <>
struct NearestVector<int8_t, int8_t> {
    using Type = svint8_t;
};
template <>
struct NearestVector<uint8_t, uint8_t> {
    using Type = svuint8_t;
};

template <typename DiffDstType, typename DiffSrcType>
struct NearestExecutor {
    static IntType SvStep()
    {
        if constexpr ((std::is_same_v<DiffDstType, int8_t> && std::is_same_v<DiffSrcType, int8_t>) ||
                      (std::is_same_v<DiffDstType, uint8_t> && std::is_same_v<DiffSrcType, uint8_t>)) {
            return svcntb();
        } else {
            return svcntw();
        }
    }

    static svbool_t Pg(IntType x, IntType xEnd)
    {
        if constexpr ((std::is_same_v<DiffDstType, int8_t> && std::is_same_v<DiffSrcType, int8_t>) ||
                      (std::is_same_v<DiffDstType, uint8_t> && std::is_same_v<DiffSrcType, uint8_t>)) {
            return svwhilelt_b8_s64(x, xEnd);
        } else {
            return svwhilelt_b32_s64(x, xEnd);
        }
    }

    static typename NearestVector<DiffDstType, DiffSrcType>::Type DupZero()
    {
        if constexpr (std::is_same_v<DiffDstType, int32_t> && std::is_same_v<DiffSrcType, int32_t>) {
            return svdup_n_s32(S32_ZERO);
        } else if constexpr (std::is_same_v<DiffDstType, int8_t> && std::is_same_v<DiffSrcType, int8_t>) {
            return svdup_n_s8(S8_ZERO);
        } else if constexpr (std::is_same_v<DiffDstType, uint8_t> && std::is_same_v<DiffSrcType, uint8_t>) {
            return svdup_n_u8(U8_ZERO);
        } else {
            return svdup_n_f32(F32_ZERO);
        }
    }

    static void LoadAndCalculate(IntType x, IntType xEnd, const DiffDstType *ddst,
                                 typename NearestVector<DiffDstType, DiffSrcType>::Type &weights)
    {
        if constexpr (std::is_same_v<DiffDstType, int32_t> && std::is_same_v<DiffSrcType, int32_t>) {
            svint32_t t1 = svld1_s32(svwhilelt_b32_s64(x, xEnd), ddst);
            weights = svqadd_s32(weights, t1);
        } else if constexpr (std::is_same_v<DiffDstType, int8_t> && std::is_same_v<DiffSrcType, int8_t>) {
            svint8_t t1 = svld1_s8(svwhilelt_b8_s64(x, xEnd), ddst);
            weights = svqadd_s8(weights, t1);
        } else if constexpr (std::is_same_v<DiffDstType, uint8_t> && std::is_same_v<DiffSrcType, uint8_t>) {
            svuint8_t t1 = svld1_u8(svwhilelt_b8_s64(x, xEnd), ddst);
            weights = svqadd_u8(weights, t1);
        } else {
            svbool_t pg = svwhilelt_b32_s64(x, xEnd);
            svfloat32_t t1 = LoadSvF32FromAddress(pg, ddst);
            weights = svadd_f32_m(pg, weights, t1);
        }
    }

    static void StoreAbx(DiffSrcType *dsrc, typename NearestVector<DiffDstType, DiffSrcType>::Type &weights)
    {
        if constexpr (std::is_same_v<DiffDstType, int32_t> && std::is_same_v<DiffSrcType, int32_t>) {
            int64_t weightsS64 = svaddv_s32(svptrue_b32(), weights);
            (*dsrc) = RoundAndSaturate<int64_t, int32_t>(weightsS64);
        } else if constexpr (std::is_same_v<DiffDstType, int8_t> && std::is_same_v<DiffSrcType, int8_t>) {
            int64_t weightsS64 = svaddv_s8(svptrue_b8(), weights);
            (*dsrc) = RoundAndSaturate<int64_t, int8_t>(weightsS64);
        } else if constexpr (std::is_same_v<DiffDstType, uint8_t> && std::is_same_v<DiffSrcType, uint8_t>) {
            uint64_t weightsU64 = svaddv_u8(svptrue_b8(), weights);
            (*dsrc) = RoundAndSaturate<uint64_t, uint8_t>(weightsU64);
        } else {
            StoreF32ToAddress(dsrc, svaddv_f32(svptrue_b32(), weights));
        }
    }

    static void StoreAxb(svbool_t pg, DiffSrcType *dsrc,
                         typename NearestVector<DiffDstType, DiffSrcType>::Type &weights)
    {
        if constexpr (std::is_same_v<DiffDstType, int32_t> && std::is_same_v<DiffSrcType, int32_t>) {
            svst1_s32(pg, dsrc, weights);
        } else if constexpr (std::is_same_v<DiffDstType, int8_t> && std::is_same_v<DiffSrcType, int8_t>) {
            svst1_s8(pg, dsrc, weights);
        } else if constexpr (std::is_same_v<DiffDstType, uint8_t> && std::is_same_v<DiffSrcType, uint8_t>) {
            svst1_u8(pg, dsrc, weights);
        } else {
            StoreSvF32ToAddress(pg, dsrc, weights);
        }
    }
};

template <typename DiffDstType, typename DiffSrcType>
inline void NearestBwdVCAbxKernel(IntType odLeft, IntType odRight, IntType ohLeft, IntType ohRight,
    IntType owLeft, IntType owRight, const DiffDstType *ddst, DiffSrcType *dsrc, StridesInfo &sInfo)
{
    IntType svStep = NearestExecutor<DiffDstType, DiffSrcType>::SvStep();
    typename NearestVector<DiffDstType, DiffSrcType>::Type weights =
        NearestExecutor<DiffDstType, DiffSrcType>::DupZero();
    for (IntType od = odLeft; od < odRight; od++) {
        for (IntType oh = ohLeft; oh < ohRight; oh++) {
            for (IntType ow = owLeft; ow < owRight; ow += svStep) {
                IntType ddstOffset = od * sInfo.strideOD + oh * sInfo.strideOH + ow * sInfo.strideOW;
                NearestExecutor<DiffDstType, DiffSrcType>::LoadAndCalculate(ow, owRight, ddst + ddstOffset, weights);
            }
        }
    }
    NearestExecutor<DiffDstType, DiffSrcType>::StoreAbx(dsrc, weights);
}

template <typename DiffDstType, typename DiffSrcType>
inline void NearestBwdVCAxbKernel(IntType odLeft, IntType odRight, IntType ohLeft, IntType ohRight, IntType owLeft,
    IntType owRight, const DiffDstType *ddst, DiffSrcType *dsrc, StridesInfo &sInfo, IntType cStart, IntType cEnd)
{
    IntType svStep = NearestExecutor<DiffDstType, DiffSrcType>::SvStep();
    for (IntType c = cStart; c < cEnd; c += svStep) {
        svbool_t pg = NearestExecutor<DiffDstType, DiffSrcType>::Pg(c, cEnd);
        typename NearestVector<DiffDstType, DiffSrcType>::Type weights =
            NearestExecutor<DiffDstType, DiffSrcType>::DupZero();
        for (IntType od = odLeft; od < odRight; od++) {
            for (IntType oh = ohLeft; oh < ohRight; oh++) {
                for (IntType ow = owLeft; ow < owRight; ow++) {
                    IntType ddstOffset =
                        od * sInfo.strideOD + oh * sInfo.strideOH + ow * sInfo.strideOW + c * sInfo.strideOC;
                    NearestExecutor<DiffDstType, DiffSrcType>::LoadAndCalculate(c, cEnd, ddst + ddstOffset, weights);
                }
            }
        }
        NearestExecutor<DiffDstType, DiffSrcType>::StoreAxb(pg, dsrc + c * sInfo.strideIC, weights);
    }
}

template <typename DiffDstType, typename DiffSrcType>
void NearestBwdVC(const Shape &batchSplitDims, const Shape &stride, const void *diffDst, void *diffSrc,
    const Shape &dimsDiffSrc, const Shape &stridesDiffDst, const Shape &stridesDiffSrc,
    const NearestData &data, bool isAxb)
{
    SplitDimInfo<IntType> sdInfo(batchSplitDims, stride, isAxb);
    StridesInfo sInfo(stridesDiffSrc, stridesDiffDst);
    IntType dID = dimsDiffSrc[IDX_2];
    IntType dIH = dimsDiffSrc[IDX_3];

    if (isAxb) {
        for (IntType mb = sdInfo.mbStart; mb < sdInfo.mbEnd; mb++) {
            for (IntType id = sdInfo.dStart; id < sdInfo.dEnd; id++) {
                for (IntType ih = sdInfo.hStart; ih < sdInfo.hEnd; ih++) {
                    for (IntType iw = sdInfo.wStart; iw < sdInfo.wEnd; iw++) {
                        IntType odLeft = data.leftIndices[id];
                        IntType odRight = data.rightIndices[id];
                        IntType ohLeft = data.leftIndices[dID + ih];
                        IntType ohRight = data.rightIndices[dID + ih];
                        IntType owLeft = data.leftIndices[dID + dIH + iw];
                        IntType owRight = data.rightIndices[dID + dIH + iw];
                        const DiffDstType *ddst = static_cast<const DiffDstType *>(diffDst) + mb * sInfo.strideOMB;
                        DiffSrcType *dsrc = static_cast<DiffSrcType *>(diffSrc)
                            + mb * sInfo.strideIMB + id * sInfo.strideID + ih * sInfo.strideIH + iw * sInfo.strideIW;
                        NearestBwdVCAxbKernel(odLeft, odRight, ohLeft, ohRight, owLeft, owRight,
                            ddst, dsrc, sInfo, sdInfo.cStart, sdInfo.cEnd);
                    }
                }
            }
        }
        return ;
    }

    for (IntType mb = sdInfo.mbStart; mb < sdInfo.mbEnd; mb++) {
        for (IntType c = sdInfo.cStart; c < sdInfo.cEnd; c++) {
            for (IntType id = sdInfo.dStart; id < sdInfo.dEnd; id++) {
                for (IntType ih = sdInfo.hStart; ih < sdInfo.hEnd; ih++) {
                    for (IntType iw = sdInfo.wStart; iw < sdInfo.wEnd; iw++) {
                        IntType odLeft = data.leftIndices[id];
                        IntType odRight = data.rightIndices[id];
                        IntType ohLeft = data.leftIndices[dID + ih];
                        IntType ohRight = data.rightIndices[dID + ih];
                        IntType owLeft = data.leftIndices[dID + dIH + iw];
                        IntType owRight = data.rightIndices[dID + dIH + iw];
                        const DiffDstType *ddst = static_cast<const DiffDstType *>(diffDst)
                            + mb * sInfo.strideOMB + c * sInfo.strideOC;
                        DiffSrcType *dsrc = static_cast<DiffSrcType *>(diffSrc)
                            + mb * sInfo.strideIMB + c * sInfo.strideIC
                            + id * sInfo.strideID + ih * sInfo.strideIH + iw * sInfo.strideIW;
                        NearestBwdVCAbxKernel(odLeft, odRight, ohLeft, ohRight, owLeft, owRight, ddst, dsrc, sInfo);
                    }
                }
            }
        }
    }
}

// std::function for NearestBwdVC().
using NearestBwdKerFunc = std::function<void(const Shape &, const Shape &, const void *, void *,
                                             const Shape &, const Shape &, const Shape &, const NearestData &, bool)>;

const std::unordered_map<BwdKerParas, NearestBwdKerFunc, BwdKerParasHash, BwdKerParasEqual> NEAREST_BWD_KER_MAP = {
    // F32
    {{Element::TypeT::F32, Element::TypeT::F32, DIM_5}, NearestBwdVC<float, float>},
    {{Element::TypeT::F32, Element::TypeT::F16, DIM_5}, NearestBwdVC<float, __fp16>},
    {{Element::TypeT::F32, Element::TypeT::BF16, DIM_5}, NearestBwdVC<float, __bf16>},
    {{Element::TypeT::F32, Element::TypeT::S32, DIM_5}, NearestBwdVC<float, int32_t>},
    {{Element::TypeT::F32, Element::TypeT::S8, DIM_5}, NearestBwdVC<float, int8_t>},
    {{Element::TypeT::F32, Element::TypeT::U8, DIM_5}, NearestBwdVC<float, uint8_t>},
    // F16
    {{Element::TypeT::F16, Element::TypeT::F32, DIM_5}, NearestBwdVC<__fp16, float>},
    {{Element::TypeT::F16, Element::TypeT::F16, DIM_5}, NearestBwdVC<__fp16, __fp16>},
    {{Element::TypeT::F16, Element::TypeT::BF16, DIM_5}, NearestBwdVC<__fp16, __bf16>},
    {{Element::TypeT::F16, Element::TypeT::S32, DIM_5}, NearestBwdVC<__fp16, int32_t>},
    {{Element::TypeT::F16, Element::TypeT::S8, DIM_5}, NearestBwdVC<__fp16, int8_t>},
    {{Element::TypeT::F16, Element::TypeT::U8, DIM_5}, NearestBwdVC<__fp16, uint8_t>},
    // BF16
    {{Element::TypeT::BF16, Element::TypeT::F32, DIM_5}, NearestBwdVC<__bf16, float>},
    {{Element::TypeT::BF16, Element::TypeT::F16, DIM_5}, NearestBwdVC<__bf16, __fp16>},
    {{Element::TypeT::BF16, Element::TypeT::BF16, DIM_5}, NearestBwdVC<__bf16, __bf16>},
    {{Element::TypeT::BF16, Element::TypeT::S32, DIM_5}, NearestBwdVC<__bf16, int32_t>},
    {{Element::TypeT::BF16, Element::TypeT::S8, DIM_5}, NearestBwdVC<__bf16, int8_t>},
    {{Element::TypeT::BF16, Element::TypeT::U8, DIM_5}, NearestBwdVC<__bf16, uint8_t>},
    // S32
    {{Element::TypeT::S32, Element::TypeT::F32, DIM_5}, NearestBwdVC<int32_t, float>},
    {{Element::TypeT::S32, Element::TypeT::F16, DIM_5}, NearestBwdVC<int32_t, __fp16>},
    {{Element::TypeT::S32, Element::TypeT::BF16, DIM_5}, NearestBwdVC<int32_t, __bf16>},
    {{Element::TypeT::S32, Element::TypeT::S32, DIM_5}, NearestBwdVC<int32_t, int32_t>},
    {{Element::TypeT::S32, Element::TypeT::S8, DIM_5}, NearestBwdVC<int32_t, int8_t>},
    {{Element::TypeT::S32, Element::TypeT::U8, DIM_5}, NearestBwdVC<int32_t, uint8_t>},
    // S8
    {{Element::TypeT::S8, Element::TypeT::F32, DIM_5}, NearestBwdVC<int8_t, float>},
    {{Element::TypeT::S8, Element::TypeT::F16, DIM_5}, NearestBwdVC<int8_t, __fp16>},
    {{Element::TypeT::S8, Element::TypeT::BF16, DIM_5}, NearestBwdVC<int8_t, __bf16>},
    {{Element::TypeT::S8, Element::TypeT::S32, DIM_5}, NearestBwdVC<int8_t, int32_t>},
    {{Element::TypeT::S8, Element::TypeT::S8, DIM_5}, NearestBwdVC<int8_t, int8_t>},
    {{Element::TypeT::S8, Element::TypeT::U8, DIM_5}, NearestBwdVC<int8_t, uint8_t>},
    // U8
    {{Element::TypeT::U8, Element::TypeT::F32, DIM_5}, NearestBwdVC<uint8_t, float>},
    {{Element::TypeT::U8, Element::TypeT::F16, DIM_5}, NearestBwdVC<uint8_t, __fp16>},
    {{Element::TypeT::U8, Element::TypeT::BF16, DIM_5}, NearestBwdVC<uint8_t, __bf16>},
    {{Element::TypeT::U8, Element::TypeT::S32, DIM_5}, NearestBwdVC<uint8_t, int32_t>},
    {{Element::TypeT::U8, Element::TypeT::S8, DIM_5}, NearestBwdVC<uint8_t, int8_t>},
    {{Element::TypeT::U8, Element::TypeT::U8, DIM_5}, NearestBwdVC<uint8_t, uint8_t>}
};

} // Resampling
} // Detail
} // KuDNN

#endif //  KUDNN_RESAMPLING_BWD_NEAREST_KERNELS_HPP
