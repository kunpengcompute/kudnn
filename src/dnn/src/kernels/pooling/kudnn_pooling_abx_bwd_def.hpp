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

#ifndef KUDNN_POOLING_ABX_BWD_DEF_HPP
#define KUDNN_POOLING_ABX_BWD_DEF_HPP

#include <cassert>

#include "kudnn.hpp"

#include "vector_wrappers.hpp"
#include "kernels/pooling/kudnn_pooling_kernels.hpp"

namespace KuDNN {
namespace Detail {
namespace Pooling {

template<PoolingFunction Func, typename CompType, typename InType, typename OutType, typename WsType>
void EvaluateAbxBWD(const Shape &start, const Shape &end, InType *typedSrc, const OutType *typedDst,
    const WsType *typedWs, float *srcTmpBuf, const Shape &initialDims, const Shape &outputDims,
    const Shape &stridesSrc, const Shape &stridesDst, const Shape &kernel,
    const Shape &strides, const Shape &dilation, const Shape &paddingL, SizeType ndims) noexcept
{
    IntType dw = static_cast<IntType>(dilation[ndims - IDX_3]) + 1;
    IntType kw = static_cast<IntType>(kernel[ndims - IDX_3]);
    IntType isb = static_cast<IntType>(stridesSrc[NUM_5 - ndims]);
    IntType isc = static_cast<IntType>(stridesSrc[NUM_6 - ndims]);
    IntType osb = static_cast<IntType>(stridesDst[NUM_5 - ndims]);
    IntType osc = static_cast<IntType>(stridesDst[NUM_6 - ndims]);
    IntType mb = static_cast<IntType>(end[NUM_5 - ndims]);
    IntType oc = static_cast<IntType>(end[NUM_6 - ndims]);
    IntType iw = static_cast<IntType>(initialDims[IDX_4]);
    IntType ow = static_cast<IntType>(outputDims[IDX_4]);
    IntType pw = static_cast<IntType>(paddingL[ndims - IDX_3]);
    IntType sw = static_cast<IntType>(strides[ndims - IDX_3]);

    IntType kh = 1;
    IntType ph = 0;
    IntType osh = 1;
    IntType ish = 1;
    IntType ih = 1;
    IntType oh = 1;
    IntType dh = 1;
    IntType sh = 1;

    if (ndims >= DIM_4) {
        kh = static_cast<IntType>(kernel[ndims - IDX_4]);
        ph = static_cast<IntType>(paddingL[ndims - IDX_4]);
        osh = static_cast<IntType>(stridesDst[IDX_3]);
        ish = static_cast<IntType>(stridesSrc[IDX_3]);
        ih = static_cast<IntType>(initialDims[IDX_3]);
        oh = static_cast<IntType>(outputDims[IDX_3]);
        dh = static_cast<IntType>(dilation[ndims - IDX_4]) + 1;
        sh = static_cast<IntType>(strides[ndims - IDX_4]);
    }

    IntType kd = 1;
    IntType pd = 0;
    IntType osd = 1;
    IntType isd = 1;
    IntType id = 1;
    IntType od = 1;
    IntType dd = 1;
    IntType sd = 1;

    if (ndims == DIM_5) {
        kd = static_cast<IntType>(kernel[ndims - IDX_5]);
        pd = static_cast<IntType>(paddingL[ndims - IDX_5]);
        osd = static_cast<IntType>(stridesDst[IDX_2]);
        isd = static_cast<IntType>(stridesSrc[IDX_2]);
        id = static_cast<IntType>(initialDims[IDX_2]);
        od = static_cast<IntType>(outputDims[IDX_2]);
        dd = static_cast<IntType>(dilation[ndims - IDX_5]) + 1;
        sd = static_cast<IntType>(strides[ndims - IDX_5]);
    }

    float* srcTmp = nullptr;
    if constexpr (std::is_same_v<float, OutType>) {
        srcTmp = static_cast<float*>(typedSrc);
    } else {
        srcTmp = srcTmpBuf;
    }

    assert(srcTmp != nullptr);

    for (IntType mbIter = static_cast<IntType>(start[NUM_5 - ndims]); mbIter < mb; ++mbIter) {
        for (IntType ocIter = static_cast<IntType>(start[NUM_6 - ndims]);
            ocIter < oc; ++ocIter) {
            for (IntType odIter = 0; odIter < od; ++odIter) {
                for (IntType ohIter = 0; ohIter < oh; ++ohIter) {
                    for (IntType owIter = 0; owIter < ow; ++owIter) {
                        IntType indDst = mbIter * osb + ocIter * osc + odIter *
                            osd + ohIter * osh + owIter;
                        if constexpr (Func == PoolingFunction::MAX) {
                            WsType index = typedWs[indDst];
                            IntType idLocal = 0;
                            IntType ihLocal = 0;

                            if (ndims == DIM_5) {
                                const IntType kdLocal = (index / kw) / kh;
                                idLocal = odIter * sd - pd + kdLocal * dd;
                                bool outRangeD = (idLocal < 0) || (idLocal >= id);
                                if (outRangeD) {
                                    continue;
                                }
                            }

                            if (ndims >= DIM_4) {
                                const IntType khLocal = (index / kw) % kh;
                                ihLocal = ohIter * sh - ph + khLocal * dh;
                                bool outRangeH = (ihLocal < 0) || (ihLocal >= ih);
                                if (outRangeH) {
                                    continue;
                                }
                            }

                            IntType indSrc = mbIter * isb + ocIter * isc + idLocal * isd + ihLocal * ish;
                            Helpers::PoolingTraits<Func, InType, CompType, OutType, WsType, false>::
                                ComputeBWD(indDst, indSrc, srcTmp, typedDst, typedWs, iw, kw, dw,
                                sw, pw, owIter);
                        } else {
                            IntType dStart = (ndims == DIM_5) ?
                                (std::max<IntType>(0, (pd - odIter * sd + dd - 1) / dd)) : 0;
                            IntType dEnd = (ndims == DIM_5) ? (std::min<IntType>
                                (kd, std::max<IntType>(0, (id + pd - odIter * sd + dd - 1) / dd))) : 1;
                            IntType hStart = (ndims >= DIM_4) ?
                                (std::max<IntType>(0, (ph - ohIter * sh + dh - 1) / dh)) : 0;
                            IntType hEnd = (ndims >= DIM_4) ? (std::min<IntType>
                                (kh, std::max<IntType>(0, (ih + ph - ohIter * sh + dh - 1) / dh))) : 1;
                            IntType wStart = std::max<IntType>(0, (pw - owIter * sw + dw - 1) / dw);
                            IntType wEnd = std::min<IntType>(kw, std::max<IntType>
                                (0, (iw + pw - owIter * sw + dw - 1) / dw));
                            IntType delimeter = 1;
                            if constexpr (Func == PoolingFunction::AVG_INCLUDE_PADDING) {
                                delimeter = kd * kh * kw;
                            } else {
                                delimeter = (dEnd - dStart) * (hEnd - hStart) * (wEnd - wStart);
                            }
                            for (IntType kdIter = dStart; kdIter < dEnd; ++kdIter) {
                                const IntType idLocal2 = odIter * sd - pd + kdIter * dd;
                                for (IntType khIter = hStart; khIter < hEnd; ++khIter) {
                                    const IntType ihLocal2 = ohIter * sh - ph + khIter * dh;
                                    IntType indSrc = mbIter * isb + ocIter * isc + idLocal2 *
                                        isd + ihLocal2 * ish;
                                    Helpers::PoolingTraits<Func, InType, CompType,
                                        OutType, WsType, false>::ComputeBWD(indDst, indSrc,
                                        srcTmp, typedDst, typedWs, iw, kw, dw, sw, pw,
                                        owIter, delimeter, wStart, wEnd);
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    if constexpr (!std::is_same_v<float, OutType>) {
        for (IntType mbIter = static_cast<IntType>(start[NUM_5 - ndims]);
            mbIter < mb; ++mbIter) {
            SizeType idx0 = mbIter * initialDims[NUM_6 - ndims] * id * ih * iw;
            for (IntType icIter = static_cast<IntType>(start[NUM_6 - ndims]);
                icIter < oc; ++icIter) {
                SizeType idx1 = icIter * id * ih * iw;
                for (IntType idIter = 0; idIter < id; ++idIter) {
                    SizeType idx2 = idIter * ih * iw;
                    for (IntType ihIter = 0; ihIter < ih; ++ihIter) {
                        SizeType idx3 = ihIter * iw;
#ifdef KUDNNL_920F
                        if constexpr (std::is_same_v<__fp16, OutType>) {
                            IntType f32Vl = static_cast<IntType>(Vector::UnrollLength<float, float, float>());
                            IntType f16Vl = static_cast<IntType>(Vector::UnrollLength<__fp16, __fp16,
                                __fp16>());
                            IntType f16Tail = iw % f16Vl;
                            svbool_t pgAllf32 = svptrue_b32();
                            svbool_t pgAllf16 = svptrue_b16();
                            for (IntType iwIter = 0; iwIter < iw / f16Vl; ++iwIter) {
                                SizeType idx = idx0 + idx1 + idx2 + idx3 + iwIter * f16Vl;
                                svfloat32_t in0 = svld1_f32(pgAllf32, &srcTmp[idx]);
                                svfloat32_t in1 = svld1_f32(pgAllf32, &srcTmp[idx + f32Vl]);
                                svfloat16_t vec0 = svcvt_f16_f32_x(pgAllf32, in0);
                                svfloat16_t vec1 = svcvt_f16_f32_x(pgAllf32, in1);
                                svfloat16_t out0 = svuzp1_f16(vec0, vec1);
                                svst1_f16(pgAllf16, &typedSrc[idx], out0);
                            }
                            if (f16Tail != 0) {
                                for (IntType iwIter = iw / f16Vl * f16Vl; iwIter < iw; iwIter++) {
                                    SizeType idx = idx0 + idx1 + idx2 + idx3 + iwIter;
                                    typedSrc[idx] = Vector::Convert<float, InType>::Do(srcTmp[idx]);
                                }
                            }
                        } else {
                            for (IntType iwIter = 0; iwIter < iw; ++iwIter) {
                                SizeType idx = idx0 + idx1 + idx2 + idx3 + iwIter;
                                typedSrc[idx] = Vector::Convert<float, InType>::Do(srcTmp[idx]);
                            }
                        }
#else
                        for (IntType iwIter = 0; iwIter < iw; ++iwIter) {
                            SizeType idx = idx0 + idx1 + idx2 + idx3 + iwIter;
                            typedSrc[idx] = Vector::Convert<float, InType>::Do(srcTmp[idx]);
                        }
#endif
                    }
                }
            }
        }
    }
}

} // Pooling
} // Detail
} // KuDNN

#endif //  KUDNN_POOLING_ABX_BWD_DEF_HPP