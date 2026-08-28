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

#ifndef KUDNN_POOLING_AXB_FWD_DEF_HPP
#define KUDNN_POOLING_AXB_FWD_DEF_HPP

#include "kudnn.hpp"

#include "vector_wrappers.hpp"
#include "kernels/pooling/kudnn_pooling_kernels.hpp"

namespace KuDNN {
namespace Detail {
namespace Pooling {
template<PoolingFunction Func, typename CompType, typename InType, typename OutType, typename WsType, bool UpdWs,
     ARCH Arch = ARCH::VC>
void EvaluateAxbFWDKernelLoop(IntType oc, IntType ows,
    IntType sw, IntType kw, IntType dw, IntType pw, IntType isb, IntType isw, IntType osb, IntType osw,
    IntType ohs, IntType sh, IntType kh, IntType dh, IntType ph, IntType ish,
    IntType osh, IntType ods,  IntType sd, IntType dd, IntType pd,
    IntType isd, IntType osd, IntType delimiter, std::int32_t *wsPtr, IntType mbIter, IntType odIter,
    IntType dStart, IntType dEnd, IntType ohIter, IntType hStart, IntType hEnd, IntType owIter,
    IntType wStart, IntType wEnd, IntType ocIter, CompType init, const InType *typedSrc, OutType *typedDst,
    SizeType ndims, const Shape &start)
{
    svbool_t ldPG0;
    svbool_t ldPG1;
    svbool_t ldPG2;
    svbool_t ldPG3;
    svbool_t oppg0;
    svbool_t oppg1;
    svbool_t oppg2;
    svbool_t oppg3;
    svbool_t stPG0;
    svbool_t stPG1;
    svbool_t stPG2;
    svbool_t stPG3;
    svbool_t wsStPG0;
    svbool_t wsStPG1;
    svbool_t wsStPG2;
    svbool_t wsStPG3;

    Vector::InitPGN<InType, Vector::GetUsedRegsN<InType, CompType, OutType>()>::Do(
        ldPG0, ldPG1, ldPG2, ldPG3, ocIter, oc);
    Vector::InitPGN<CompType, Vector::GetUsedRegsN<CompType, InType, OutType>()>::Do(
        oppg0, oppg1, oppg2, oppg3, ocIter, oc);
    Vector::InitPGN<OutType, Vector::GetUsedRegsN<OutType, CompType, InType>()>::Do(
        stPG0, stPG1, stPG2, stPG3, ocIter, oc);
    Vector::InitPGN<std::int32_t, Vector::GetUsedRegsN<CompType, InType, OutType>()>::Do(
        wsStPG0, wsStPG1, wsStPG2, wsStPG3, ocIter, oc);

    typename Vector::Vec<CompType, Arch>::Type res0;
    typename Vector::Vec<CompType, Arch>::Type res1;
    typename Vector::Vec<CompType, Arch>::Type res2;
    typename Vector::Vec<CompType, Arch>::Type res3;
    typename Vector::Vec<std::int32_t, Arch>::Type wsRes0;
    typename Vector::Vec<std::int32_t, Arch>::Type wsRes1;
    typename Vector::Vec<std::int32_t, Arch>::Type wsRes2;
    typename Vector::Vec<std::int32_t, Arch>::Type wsRes3;
    Vector::DupN<std::int32_t, std::int32_t,
        Vector::GetUsedRegsN<CompType, InType, OutType>(), Arch>::Do(
        0, wsRes0, wsRes1, wsRes2, wsRes3);
    Vector::DupFloatN<InType, CompType, Vector::GetUsedRegsN<InType, CompType, OutType>(), Arch>::Do(
        init, res0, res1, res2, res3);

    for (IntType kdIter = dStart; kdIter < dEnd; ++kdIter) {
        IntType d = (ndims == DIM_5) ? (odIter * sd + kdIter * dd - pd) : 0;
        for (IntType khIter = hStart; khIter < hEnd; ++khIter) {
            IntType h = (ndims >= DIM_4) ?  (ohIter * sh + khIter * dh - ph) : 0;
            for (IntType kwIter = wStart; kwIter < wEnd; ++kwIter) {
                typename Vector::Vec<CompType, Arch>::Type data0;
                typename Vector::Vec<CompType, Arch>::Type data1;
                typename Vector::Vec<CompType, Arch>::Type data2;
                typename Vector::Vec<CompType, Arch>::Type data3;
                Vector::LoadN<InType, CompType,
                    Vector::GetUsedRegsN<InType, CompType, OutType>(), Arch>::Do(
                    typedSrc + mbIter * isb + d * isd + h * ish +
                    (owIter * sw + kwIter * dw - pw) * isw + ocIter,
                    ldPG0, ldPG1, ldPG2, ldPG3, data0, data1, data2, data3);
                if constexpr (Func == PoolingFunction::MAX) {
                    if constexpr (UpdWs) {
                        typename Vector::MaskVec<CompType, Arch>::Type mask0;
                        typename Vector::MaskVec<CompType, Arch>::Type mask1;
                        typename Vector::MaskVec<CompType, Arch>::Type mask2;
                        typename Vector::MaskVec<CompType, Arch>::Type mask3;
                        Vector::GtN<CompType,
                            Vector::GetUsedRegsN<CompType, InType, OutType>(), Arch>::Do(
                            oppg0, oppg1, oppg2, oppg3, data0, data1, data2, data3,
                            res0, res1, res2, res3, mask0, mask1, mask2, mask3);
                        typename Vector::Vec<std::int32_t, Arch>::Type curIdx0;
                        typename Vector::Vec<std::int32_t, Arch>::Type curIdx1;
                        typename Vector::Vec<std::int32_t, Arch>::Type curIdx2;
                        typename Vector::Vec<std::int32_t, Arch>::Type curIdx3;
                        Vector::DupN<std::int32_t, std::int32_t,
                            Vector::GetUsedRegsN<CompType, InType, OutType>(), Arch>::Do(
                            (kdIter * kh * kw) + (khIter * kw) +
                            kwIter, curIdx0, curIdx1, curIdx2, curIdx3);
                        Vector::SelN<CompType,
                            Vector::GetUsedRegsN<CompType, InType, OutType>(), Arch>::Do(
                            mask0, mask1, mask2, mask3, data0, data1, data2, data3,
                            res0, res1, res2, res3, res0, res1, res2, res3);
                        Vector::SelN<std::int32_t,
                            Vector::GetUsedRegsN<CompType, InType, OutType>(), Arch>::Do(
                            mask0, mask1, mask2, mask3, curIdx0, curIdx1, curIdx2, curIdx3,
                            wsRes0, wsRes1, wsRes2, wsRes3, wsRes0, wsRes1, wsRes2, wsRes3);
                    } else {
                        Vector::MaxMN<CompType,
                            Vector::GetUsedRegsN<CompType, InType, OutType>(), Arch>::Do(
                            oppg0, oppg1, oppg2, oppg3, res0, res1, res2, res3,
                            data0, data1, data2, data3, res0, res1, res2, res3);
                    }
                } else {
                    Vector::AddMN<CompType,
                        Vector::GetUsedRegsN<CompType, InType, OutType>(), Arch>::Do(
                        oppg0, oppg1, oppg2, oppg3, res0, res1, res2, res3,
                        data0, data1, data2, data3, res0, res1, res2, res3);
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
            ((dEnd - dStart) * (hEnd - hStart) * (wEnd - wStart)), del0, del1, del2, del3);
        Vector::DivMN<CompType, Vector::GetUsedRegsN<CompType, InType, OutType>(), Arch>::Do(
            oppg0, oppg1, oppg2, oppg3, res0, res1, res2, res3,
            del0, del1, del2, del3, res0, res1, res2, res3);
    } else if constexpr (Func == PoolingFunction::AVG_INCLUDE_PADDING) {
        Vector::DupFloatN<InType, CompType,
            Vector::GetUsedRegsN<InType, CompType, OutType>(), Arch>::Do(
            delimiter, del0, del1, del2, del3);
        Vector::DivMN<CompType,
            Vector::GetUsedRegsN<CompType, InType, OutType>(), Arch>::Do(
            oppg0, oppg1, oppg2, oppg3, res0, res1, res2, res3,
            del0, del1, del2, del3, res0, res1, res2, res3);
    }
    const bool maxUpdWs = UpdWs && Func == PoolingFunction::MAX;
    if constexpr (maxUpdWs) {
        Vector::StoreN<std::int32_t, std::int32_t,
            Vector::GetUsedRegsN<CompType, InType, OutType>(), Arch>::Do(
            wsStPG0, wsStPG1, wsStPG2, wsStPG3, wsRes0, wsRes1, wsRes2, wsRes3,
            wsPtr + (mbIter - static_cast<IntType>(start[NUM_5 - ndims])) * osb +
            (odIter - ods) * osd + (ohIter - ohs) * osh + (owIter - ows) * osw +
            (ocIter - static_cast<IntType>(start[IDX_4])));
    }
    Vector::StoreN<OutType, CompType,
        Vector::GetUsedRegsN<OutType, InType, CompType>(), Arch>::Do(
        stPG0, stPG1, stPG2, stPG3, res0, res1, res2, res3,
        typedDst + mbIter * osb + odIter * osd + ohIter * osh + owIter * osw + ocIter);
}

template<PoolingFunction Func, typename CompType, typename InType, typename OutType, typename WsType, bool UpdWs,
    ARCH Arch = ARCH::VC>
void EvaluateAxbFWDInnerLoop(const Shape &start, const Shape &end, const InType *typedSrc, OutType *typedDst,
    WsType *typedWs, const Shape &initialDims, const Shape &stridesSrc, const Shape &stridesDst, const Shape &kernel,
    const Shape &strides, const Shape &dilation, const Shape &paddingL, SizeType ndims, CompType init) noexcept
{
    IntType vl = static_cast<IntType>(Vector::UnrollLength<InType, CompType, OutType>());

    IntType mb = static_cast<IntType>(end[NUM_5 - ndims]);
    IntType ow = static_cast<IntType>(end[IDX_3]);
    IntType oc = static_cast<IntType>(end[IDX_4]);
    IntType ows = static_cast<IntType>(start[IDX_3]);
    IntType iw = static_cast<IntType>(initialDims[IDX_4]);
    IntType sw = static_cast<IntType>(strides[ndims - IDX_3]);
    IntType kw = static_cast<IntType>(kernel[ndims - IDX_3]);
    IntType dw = static_cast<IntType>(dilation[ndims - IDX_3]) + 1;
    IntType pw = static_cast<IntType>(paddingL[ndims - IDX_3]);
    IntType isb = static_cast<IntType>(stridesSrc[NUM_5 - ndims]);
    IntType isw = static_cast<IntType>(stridesSrc[IDX_4]);
    IntType osb = static_cast<IntType>(stridesDst[NUM_5 - ndims]);
    IntType osw = static_cast<IntType>(stridesDst[IDX_4]);

    IntType oh = 1;
    IntType ohs = 0;
    IntType ih = 1;
    IntType sh = 1;
    IntType kh = 1;
    IntType dh = 1;
    IntType ph = 0;
    IntType ish = 1;
    IntType osh = 1;

    if (ndims >= DIM_4) {
        oh = static_cast<IntType>(end[IDX_2]);
        ohs = static_cast<IntType>(start[IDX_2]);
        ih = static_cast<IntType>(initialDims[IDX_3]);
        sh = static_cast<IntType>(strides[ndims - IDX_4]);
        kh = static_cast<IntType>(kernel[ndims - IDX_4]);
        dh = static_cast<IntType>(dilation[ndims - IDX_4]) + 1;
        ph = static_cast<IntType>(paddingL[ndims - IDX_4]);
        ish = static_cast<IntType>(stridesSrc[IDX_3]);
        osh = static_cast<IntType>(stridesDst[IDX_3]);
    }

    IntType id = 1;
    IntType od = 1;
    IntType ods = 0;
    IntType sd = 1;
    IntType pd = 0;
    IntType kd = 1;
    IntType dd = 1;
    IntType isd = 1;
    IntType osd = 1;

    if (ndims == DIM_5) {
        od = static_cast<IntType>(end[IDX_1]);
        ods = static_cast<IntType>(start[IDX_1]);
        id = static_cast<IntType>(initialDims[IDX_2]);
        sd = static_cast<IntType>(strides[ndims - IDX_5]);
        kd = static_cast<IntType>(kernel[ndims - IDX_5]);
        dd = static_cast<IntType>(dilation[ndims - IDX_5]) + 1;
        pd = static_cast<IntType>(paddingL[ndims - IDX_5]);
        isd = static_cast<IntType>(stridesSrc[IDX_2]);
        osd = static_cast<IntType>(stridesDst[IDX_2]);
    }

    IntType delimiter = kd * kh * kw;

    SizeType wsSize = (end[IDX_0] - start[IDX_0]) * (end[IDX_1] - start[IDX_1]) *
        (end[IDX_2] - start[IDX_2]) * (end[IDX_3] - start[IDX_3]) * (end[IDX_4] - start[IDX_4]);

    std::vector<std::int32_t> wsTmp(wsSize);
    std::int32_t *wsPtr;
    const bool int32UpdWs = UpdWs && std::is_same_v<WsType, std::int32_t>;
    if constexpr (int32UpdWs) {
        wsPtr = typedWs + (start[NUM_5 - ndims] * osb + ods * osd +
            ohs * osh + ows * osw + start[IDX_4]);
    } else {
        wsPtr = wsTmp.data();
    }

    for (IntType mbIter = static_cast<IntType>(start[NUM_5 - ndims]); mbIter < mb; ++mbIter) {
        for (IntType odIter = ods; odIter < od; ++odIter) {
                IntType dStart = (ndims == DIM_5) ? std::max<IntType>(0, (pd - odIter * sd + dd - 1) / dd) : 0;
                IntType dEnd = (ndims == DIM_5) ?
                    std::min<IntType>(kd, std::max<IntType>(0, (id + pd - odIter * sd + dd - 1) / dd)) : 1;
            for (IntType ohIter = ohs; ohIter < oh; ++ohIter) {
                IntType hStart = (ndims >= DIM_4) ? std::max<IntType>(0, (ph - ohIter * sh + dh - 1) / dh) : 0;
                IntType hEnd = (ndims >= DIM_4) ?
                    std::min<IntType>(kh, std::max<IntType>(0, (ih + ph - ohIter * sh + dh - 1) / dh)) : 1;
                for (IntType owIter = ows; owIter < ow; ++owIter) {
                    IntType wStart = std::max<IntType>(0, (pw - owIter * sw + dw - 1) / dw);
                    IntType wEnd = std::min<IntType>(kw, std::max<IntType>(0, (iw + pw - owIter * sw + dw - 1) / dw));
                    for (IntType ocIter = static_cast<IntType>(start[IDX_4]); ocIter < oc; ocIter += vl) {
#if USE_NEON
                        if (ocIter + vl <= oc) {
                            EvaluateAxbFWDKernelLoop<Func, CompType, InType, OutType, WsType, UpdWs, Arch>(
                                oc, ows, sw, kw, dw, pw, isb, isw, osb, osw, ohs,
                                sh, kh, dh, ph, ish, osh, ods, sd, dd, pd, isd, osd,
                                delimiter, wsPtr, mbIter, odIter, dStart, dEnd, ohIter, hStart, hEnd,
                                owIter, wStart, wEnd, ocIter, init, typedSrc, typedDst, ndims, start);
                        } else {
                            EvaluateAxbFWDKernelLoop<Func, CompType, InType, OutType, WsType, UpdWs>(
                                oc, ows, sw, kw, dw, pw, isb, isw, osb, osw, ohs,
                                sh, kh, dh, ph, ish, osh, ods, sd, dd, pd, isd, osd,
                                delimiter, wsPtr, mbIter, odIter, dStart, dEnd, ohIter, hStart, hEnd,
                                owIter, wStart, wEnd, ocIter, init, typedSrc, typedDst, ndims, start);
                        }
#else
                        EvaluateAxbFWDKernelLoop<Func, CompType, InType, OutType, WsType, UpdWs>(
                            oc, ows, sw, kw, dw, pw, isb, isw, osb, osw, ohs,
                            sh, kh, dh, ph, ish, osh, ods, sd, dd, pd, isd, osd,
                            delimiter, wsPtr, mbIter, odIter, dStart, dEnd, ohIter, hStart, hEnd,
                            owIter, wStart, wEnd, ocIter, init, typedSrc, typedDst, ndims, start);
#endif
                    }
                }
            }
        }
    }
    const bool notInt32UpdWs = UpdWs && !std::is_same_v<WsType, std::int32_t>;
    if constexpr (notInt32UpdWs) {
        for (SizeType i = 0; i < wsSize; ++i) {
            typedWs[(start[NUM_5 - ndims] * osb + ods * osd + ohs *
                osh + ows * osw + start[IDX_4]) + i] = wsTmp[i];
        }
    }
}

template<PoolingFunction Func, typename CompType, typename InType, typename OutType, typename WsType, bool UpdWs>
void EvaluateAxbFWD(const Shape &start, const Shape &end, const InType *typedSrc, OutType *typedDst, WsType *typedWs,
    const Shape &initialDims, const Shape &stridesSrc, const Shape &stridesDst, const Shape &kernel,
    const Shape &strides, const Shape &dilation, const Shape &paddingL, SizeType ndims, CompType init) noexcept
{
#if USE_NEON
    EvaluateAxbFWDInnerLoop<Func, CompType, InType, OutType, WsType, UpdWs, ARCH::NEON>(start, end, typedSrc, typedDst,
        typedWs, initialDims, stridesSrc, stridesDst, kernel, strides, dilation, paddingL, ndims, init);
#else
    EvaluateAxbFWDInnerLoop<Func, CompType, InType, OutType, WsType, UpdWs>(start, end, typedSrc, typedDst,
        typedWs, initialDims, stridesSrc, stridesDst, kernel, strides, dilation, paddingL, ndims, init);
#endif
}

} // Pooling
} // Detail
} // KuDNN

#endif //  KUDNN_POOLING_AXB_FWD_DEF_HPP