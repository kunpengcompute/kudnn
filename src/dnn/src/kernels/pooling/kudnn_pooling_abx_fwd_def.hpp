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

#ifndef KUDNN_POOLING_ABX_FWD_DEF_HPP
#define KUDNN_POOLING_ABX_FWD_DEF_HPP

#include "kudnn.hpp"

#include "vector_wrappers.hpp"
#include "kernels/pooling/kudnn_pooling_kernels.hpp"

namespace KuDNN {
namespace Detail {
namespace Pooling {

template<PoolingFunction Func, typename CompType, typename InType, typename OutType, typename WsType, bool UpdWs,
    ARCH Arch = ARCH::VC>
void EvaluateAbxFWDInnerLoop(const Shape &start, const Shape &end, const InType *typedSrc, OutType *typedDst,
    WsType *typedWs, const Shape &initialDims, const Shape &outputDims, const Shape &stridesSrc,
    const Shape &stridesDst, const Shape &kernel, const Shape &strides, const Shape &dilation, const Shape &paddingL,
    SizeType ndims, CompType init, const std::vector<ReorderLayer> &reorders) noexcept
{
    IntType vl = static_cast<IntType>(Vector::UnrollLength<InType, CompType, OutType>());

    IntType isb = static_cast<IntType>(stridesSrc[NUM_5 - ndims]);
    IntType isc = static_cast<IntType>(stridesSrc[NUM_6 - ndims]);
    IntType osb = static_cast<IntType>(stridesDst[NUM_5 - ndims]);
    IntType osc = static_cast<IntType>(stridesDst[NUM_6 - ndims]);
    IntType iw = static_cast<IntType>(initialDims[IDX_4]);
    IntType kw = static_cast<IntType>(kernel[ndims - IDX_3]);
    IntType sw = static_cast<IntType>(strides[ndims - IDX_3]);
    IntType pw = static_cast<IntType>(paddingL[ndims - IDX_3]);
    IntType dw = static_cast<IntType>(dilation[ndims - IDX_3]) + 1;
    IntType mb = static_cast<IntType>(end[NUM_5 - ndims]);
    IntType oc = static_cast<IntType>(end[NUM_6 - ndims]);
    IntType ow = static_cast<IntType>(outputDims[IDX_4]);

    IntType oh = 1;
    IntType osh = 1;
    IntType ih = 1;
    IntType kh = 1;
    IntType sh = 1;
    IntType ph = 0;
    IntType dh = 1;
    IntType ish = 1;

    if (ndims >= DIM_4) {
        oh = static_cast<IntType>(outputDims[IDX_3]);
        osh = static_cast<IntType>(stridesDst[IDX_3]);
        ih = static_cast<IntType>(initialDims[IDX_3]);
        kh = static_cast<IntType>(kernel[ndims - IDX_4]);
        sh = static_cast<IntType>(strides[ndims - IDX_4]);
        ph = static_cast<IntType>(paddingL[ndims - IDX_4]);
        dh = static_cast<IntType>(dilation[ndims - IDX_4]) + 1;
        ish = static_cast<IntType>(stridesSrc[IDX_3]);
    }

    IntType od = 1;
    IntType osd = 1;
    IntType id = 1;
    IntType kd = 1;
    IntType sd = 1;
    IntType pd = 0;
    IntType dd = 1;
    IntType isd = 1;

    if (ndims == DIM_5) {
        od = static_cast<IntType>(outputDims[IDX_2]);
        osd = static_cast<IntType>(stridesDst[IDX_2]);
        id = static_cast<IntType>(initialDims[IDX_2]);
        kd = static_cast<IntType>(kernel[ndims - IDX_5]);
        sd = static_cast<IntType>(strides[ndims - IDX_5]);
        pd = static_cast<IntType>(paddingL[ndims - IDX_5]);
        dd = static_cast<IntType>(dilation[ndims - IDX_5]) + 1;
        isd = static_cast<IntType>(stridesSrc[IDX_2]);
    }

    IntType delimiter = kd * kh * kw;
    IntType spatialInDim = id * ih * iw;
    IntType spatialOutDim = od * oh * ow;

    auto tailInTranspose = [vl](SizeType n, SizeType m, const InType *src, InType *dst) {
        for (SizeType ni = 0; ni < n; ++ni) {
            for (SizeType mi = 0; mi < m; ++mi) {
                dst[mi * vl + ni] = src[ni * m + mi];
            }
        }
    };

    auto tailOutTranspose = [vl](SizeType n, SizeType m, OutType *src, OutType *dst) {
        for (SizeType ni = 0; ni < n; ++ni) {
            for (SizeType mi = 0; mi < m; ++mi) {
                dst[ni * m + mi] = src[mi * vl + ni];
            }
        }
    };

    auto tailWsTranspose = [vl](SizeType n, SizeType m, std::int32_t *src, WsType *dst) {
        for (SizeType ni = 0; ni < n; ++ni) {
            for (SizeType mi = 0; mi < m; ++mi) {
                dst[ni * m + mi] = src[mi * vl + ni];
            }
        }
    };

    const bool needWsPtr = UpdWs && Func == PoolingFunction::MAX;
    std::vector<InType, Service::AlignedAllocator<InType>> tmpSrc(vl * id * ih * iw,
        Vector::Convert<float, InType>::Do(0.f));
    std::vector<OutType, Service::AlignedAllocator<OutType>> tmpDst(vl * od * oh * ow,
        Vector::Convert<float, OutType>::Do(0.f));
    std::vector<std::int32_t, Service::AlignedAllocator<std::int32_t>> tmpWs(vl * od * oh * ow);

    for (IntType mbIter = static_cast<IntType>(start[NUM_5 - ndims]); mbIter < mb; ++mbIter) {
        for (IntType ocIter = static_cast<IntType>(start[NUM_6 - ndims]);
            ocIter < oc; ocIter += vl) {
            const InType *srcPtr = typedSrc + mbIter * isb + ocIter * isc;
            OutType *dstPtr = typedDst + mbIter * osb + ocIter * osc;
            WsType *wsPtr;
            if constexpr (needWsPtr) {
                wsPtr = typedWs + mbIter * osb + ocIter * osc;
            }

            svbool_t stPG0;
            svbool_t stPG1;
            svbool_t stPG2;
            svbool_t stPG3;
            svbool_t wsStPG0;
            svbool_t wsStPG1;
            svbool_t wsStPG2;
            svbool_t wsStPG3;
            svbool_t ldPG0;
            svbool_t ldPG1;
            svbool_t ldPG2;
            svbool_t ldPG3;
            svbool_t oppg0;
            svbool_t oppg1;
            svbool_t oppg2;
            svbool_t oppg3;

            Vector::InitPGN<OutType, Vector::GetUsedRegsN<OutType, CompType, InType>()>::Do(
                stPG0, stPG1, stPG2, stPG3, ocIter, oc);
            Vector::InitPGN<std::int32_t, Vector::GetUsedRegsN<CompType, InType, OutType>()>::Do(
                wsStPG0, wsStPG1, wsStPG2, wsStPG3, ocIter, oc);
            Vector::InitPGN<InType, Vector::GetUsedRegsN<InType, CompType, OutType>()>::Do(
                ldPG0, ldPG1, ldPG2, ldPG3, ocIter, oc);
            Vector::InitPGN<CompType, Vector::GetUsedRegsN<CompType, InType, OutType>()>::Do(
                oppg0, oppg1, oppg2, oppg3, ocIter, oc);

            if (ocIter + vl <= oc) {
                reorders[0].Run(srcPtr, tmpSrc.data());
            } else {
                tailInTranspose((oc - static_cast<IntType>(start[NUM_6 - ndims])) % vl, spatialInDim,
                    srcPtr, tmpSrc.data());
            }

            for (IntType odIter = 0; odIter < od; ++odIter) {
                IntType dStart = (ndims == DIM_5) ?
                    std::max<IntType>(0, (pd - odIter * sd + dd - 1) / dd) : 0;
                IntType dEnd = (ndims == DIM_5) ?
                    std::min<IntType>(kd, std::max<IntType>(0, (id + pd - odIter * sd + dd - 1) / dd)) : 1;
                for (IntType ohIter = 0; ohIter < oh; ++ohIter) {
                    IntType hStart = (ndims >= DIM_4) ?
                        std::max<IntType>(0, (ph - ohIter * sh + dh - 1) / dh) : 0;
                    IntType hEnd = (ndims >= DIM_4) ?
                        std::min<IntType>(kh, std::max<IntType>(0, (ih + ph - ohIter * sh + dh - 1) / dh)) : 1;
                    for (IntType owIter = 0; owIter < ow; ++owIter) {
                        IntType wStart = std::max<IntType>(0, (pw - owIter * sw + dw - 1) / dw);
                        IntType wEnd = std::min<IntType>
                            (kw, std::max<IntType>(0, (iw + pw - owIter * sw + dw - 1) / dw));

                        bool needToPadding = dStart >= dEnd || hStart >= hEnd || wStart >= wEnd;
                        if (needToPadding) {
                            typename Vector::Vec<CompType, Arch>::Type val0;
                            typename Vector::Vec<CompType, Arch>::Type val1;
                            typename Vector::Vec<CompType, Arch>::Type val2;
                            typename Vector::Vec<CompType, Arch>::Type val3;
                            Vector::DupFloatN<OutType, CompType,
                                Vector::GetUsedRegsN<OutType, CompType, InType>(), Arch>::Do(init,
                                val0, val1, val2, val3);
                            Vector::StoreN<OutType, CompType,
                                Vector::GetUsedRegsN<OutType, InType, CompType>(), Arch>::Do(stPG0,
                                stPG1, stPG2, stPG3, val0, val1, val2, val3,
                                tmpDst.data() + odIter * osd * vl + ohIter * osh * vl + owIter * vl);
                            if constexpr (needWsPtr) {
                                typename Vector::Vec<std::int32_t, Arch>::Type wsRes0;
                                typename Vector::Vec<std::int32_t, Arch>::Type wsRes1;
                                typename Vector::Vec<std::int32_t, Arch>::Type wsRes2;
                                typename Vector::Vec<std::int32_t, Arch>::Type wsRes3;
                                Vector::DupN<std::int32_t, std::int32_t,
                                    Vector::GetUsedRegsN<CompType, InType, OutType>(), Arch>::Do(0,
                                    wsRes0, wsRes1, wsRes2, wsRes3);
                                Vector::StoreN<std::int32_t, std::int32_t,
                                    Vector::GetUsedRegsN<CompType, InType, OutType>(), Arch>::Do(
                                    wsStPG0, wsStPG1, wsStPG2, wsStPG3,
                                    wsRes0, wsRes1, wsRes2, wsRes3,
                                    tmpWs.data() + odIter * osd *
                                    vl + ohIter * osh * vl + owIter * vl);
                            }
                            continue;
                        }

                        typename Vector::Vec<std::int32_t, Arch>::Type wsRes0;
                        typename Vector::Vec<std::int32_t, Arch>::Type wsRes1;
                        typename Vector::Vec<std::int32_t, Arch>::Type wsRes2;
                        typename Vector::Vec<std::int32_t, Arch>::Type wsRes3;

                        typename Vector::Vec<CompType, Arch>::Type res0;
                        typename Vector::Vec<CompType, Arch>::Type res1;
                        typename Vector::Vec<CompType, Arch>::Type res2;
                        typename Vector::Vec<CompType, Arch>::Type res3;

                        Vector::DupN<std::int32_t, std::int32_t,
                            Vector::GetUsedRegsN<CompType, InType, OutType>(), Arch>::Do(0,
                            wsRes0, wsRes1, wsRes2, wsRes3);
                        Vector::DupFloatN<InType, CompType, Vector::GetUsedRegsN
                            <InType, CompType, OutType>(), Arch>::Do(init, res0, res1, res2, res3);

                        for (IntType kdIter = dStart; kdIter < dEnd; ++kdIter) {
                            IntType d = (ndims == DIM_5) ? (odIter * sd + kdIter * dd - pd) : 0;
                            for (IntType khIter = hStart; khIter < hEnd; ++khIter) {
                                IntType h = (ndims >= DIM_4) ? (ohIter * sh + khIter * dh - ph) : 0;
                                for (IntType kwIter = wStart; kwIter < wEnd; ++kwIter) {
                                    IntType w = owIter * sw + kwIter * dw - pw;
                                    IntType idx = (d * isd + h * ish + w) * vl;
                                    typename Vector::Vec<CompType, Arch>::Type data0;
                                    typename Vector::Vec<CompType, Arch>::Type data1;
                                    typename Vector::Vec<CompType, Arch>::Type data2;
                                    typename Vector::Vec<CompType, Arch>::Type data3;
                                    Vector::LoadN<InType, CompType,
                                        Vector::GetUsedRegsN<InType, CompType, OutType>(), Arch>::Do(
                                        tmpSrc.data() + idx, ldPG0, ldPG1, ldPG2, ldPG3,
                                        data0, data1, data2, data3);
                                    if constexpr (Func == PoolingFunction::MAX) {
                                        if constexpr (UpdWs) {
                                            typename Vector::MaskVec<CompType, Arch>::Type mask0;
                                            typename Vector::MaskVec<CompType, Arch>::Type mask1;
                                            typename Vector::MaskVec<CompType, Arch>::Type mask2;
                                            typename Vector::MaskVec<CompType, Arch>::Type mask3;
                                            Vector::GtN<CompType,
                                                Vector::GetUsedRegsN<CompType, InType, OutType>(), Arch>::Do(
                                                oppg0, oppg1, oppg2, oppg3,
                                                data0, data1, data2, data3,
                                                res0, res1, res2, res3,
                                                mask0, mask1, mask2, mask3);
                                            typename Vector::Vec<std::int32_t, Arch>::Type curIdx0;
                                            typename Vector::Vec<std::int32_t, Arch>::Type curIdx1;
                                            typename Vector::Vec<std::int32_t, Arch>::Type curIdx2;
                                            typename Vector::Vec<std::int32_t, Arch>::Type curIdx3;
                                            Vector::DupN<std::int32_t, std::int32_t,
                                                Vector::GetUsedRegsN<CompType, InType,
                                                OutType>(), Arch>::Do((kdIter * kh + khIter) * kw + kwIter,
                                                curIdx0, curIdx1, curIdx2, curIdx3);
                                            Vector::SelN<CompType,
                                                Vector::GetUsedRegsN<CompType, InType, OutType>(), Arch>::Do(
                                                mask0, mask1, mask2, mask3,
                                                data0, data1, data2, data3,
                                                res0, res1, res2, res3,
                                                res0, res1, res2, res3);
                                            Vector::SelN<std::int32_t,
                                                Vector::GetUsedRegsN<CompType, InType, OutType>(), Arch>::Do(
                                                mask0, mask1, mask2, mask3,
                                                curIdx0, curIdx1, curIdx2, curIdx3,
                                                wsRes0, wsRes1, wsRes2, wsRes3,
                                                wsRes0, wsRes1, wsRes2, wsRes3);
                                        } else {
                                            Vector::MaxMN<CompType,
                                                Vector::GetUsedRegsN<CompType, InType, OutType>(), Arch>::Do(
                                                oppg0, oppg1, oppg2, oppg3,
                                                res0, res1, res2, res3,
                                                data0, data1, data2, data3,
                                                res0, res1, res2, res3);
                                        }
                                    } else {
                                        Vector::AddMN<CompType,
                                            Vector::GetUsedRegsN<CompType, InType, OutType>(), Arch>::Do(
                                            oppg0, oppg1, oppg2, oppg3,
                                            res0, res1, res2, res3,
                                            data0, data1, data2, data3,
                                            res0, res1, res2, res3);
                                    }
                                }
                            }
                        }
                        typename Vector::Vec<CompType, Arch>::Type del0;
                        typename Vector::Vec<CompType, Arch>::Type del1;
                        typename Vector::Vec<CompType, Arch>::Type del2;
                        typename Vector::Vec<CompType, Arch>::Type del3;
                        if constexpr (Func == PoolingFunction::AVG_EXCLUDE_PADDING) {
                            Vector::DupFloatN<InType, CompType,
                                Vector::GetUsedRegsN<InType, CompType, OutType>(), Arch>::Do(
                                ((wEnd - wStart)  * (hEnd - hStart) * (dEnd - dStart)),
                                del0, del1, del2, del3);
                            Vector::DivMN<CompType,
                                Vector::GetUsedRegsN<CompType, InType, OutType>(), Arch>::Do(
                                oppg0, oppg1, oppg2, oppg3,
                                res0, res1, res2, res3,
                                del0, del1, del2, del3,
                                res0, res1, res2, res3);
                        } else if constexpr (Func == PoolingFunction::AVG_INCLUDE_PADDING) {
                            Vector::DupFloatN<InType, CompType,
                                Vector::GetUsedRegsN<InType, CompType, OutType>(), Arch>::Do(delimiter,
                                del0, del1, del2, del3);
                            Vector::DivMN<CompType,
                                Vector::GetUsedRegsN<CompType, InType, OutType>(), Arch>::Do(
                                oppg0, oppg1, oppg2, oppg3,
                                res0, res1, res2, res3,
                                del0, del1, del2, del3,
                                res0, res1, res2, res3);
                        }
                        if constexpr (needWsPtr) {
                            Vector::StoreN<std::int32_t, std::int32_t,
                                Vector::GetUsedRegsN<CompType, InType, OutType>(), Arch>::Do(
                                wsStPG0, wsStPG1, wsStPG2, wsStPG3,
                                wsRes0, wsRes1, wsRes2, wsRes3,
                                tmpWs.data() + odIter * osd * vl + ohIter * osh * vl + owIter * vl);
                        }
                        Vector::StoreN<OutType, CompType,
                            Vector::GetUsedRegsN<OutType, InType, CompType>(), Arch>::Do(
                            stPG0, stPG1, stPG2, stPG3,
                            res0, res1, res2, res3,
                            tmpDst.data() + odIter * osd * vl + ohIter * osh * vl + owIter * vl);
                    }
                }
            }
            if (ocIter + vl <= oc) {
                    reorders[1].Run(tmpDst.data(), dstPtr);
                if constexpr (needWsPtr) {
                    tailWsTranspose(vl, spatialOutDim,
                        tmpWs.data(), wsPtr);
                }
            } else {
                tailOutTranspose((oc - static_cast<IntType>(start[NUM_6 - ndims])) % vl, spatialOutDim,
                    tmpDst.data(), dstPtr);
                if constexpr (needWsPtr) {
                    tailWsTranspose((oc - static_cast<IntType>(start[NUM_6 - ndims])) % vl, spatialOutDim,
                        tmpWs.data(), wsPtr);
                }
            }
        }
    }
}

template<PoolingFunction Func, typename CompType, typename InType, typename OutType, typename WsType, bool UpdWs>
void EvaluateAbxFWD(const Shape &start, const Shape &end, const InType *typedSrc, OutType *typedDst, WsType *typedWs,
    const Shape &initialDims, const Shape &outputDims, const Shape &stridesSrc, const Shape &stridesDst,
    const Shape &kernel, const Shape &strides, const Shape &dilation, const Shape &paddingL,
    SizeType ndims, CompType init, const std::vector<ReorderLayer> &reorders) noexcept
{
#if USE_NEON
    EvaluateAbxFWDInnerLoop<Func, CompType, InType, OutType, WsType, UpdWs, ARCH::NEON>(start, end, typedSrc, typedDst,
        typedWs, initialDims, outputDims, stridesSrc, stridesDst, kernel, strides, dilation, paddingL, ndims, init,
        reorders);
#else
    EvaluateAbxFWDInnerLoop<Func, CompType, InType, OutType, WsType, UpdWs>(start, end, typedSrc, typedDst, typedWs,
        initialDims, outputDims, stridesSrc, stridesDst, kernel, strides, dilation, paddingL, ndims, init, reorders);
#endif
}

} // Pooling
} // Detail
} // KuDNN

#endif //  KUDNN_POOLING_ABX_FWD_DEF_HPP