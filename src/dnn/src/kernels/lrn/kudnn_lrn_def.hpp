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

#ifndef KUDNN_LRN_DEF_HPP
#define KUDNN_LRN_DEF_HPP

#include <cmath>

#include "kudnn.hpp"

#include "vector_wrappers.hpp"
#include "kernels/lrn/kudnn_lrn_kernels.hpp"

namespace KuDNN {
namespace Detail {
namespace LRN {
namespace Helpers {

template <typename CompType>
CompType ComputePow(CompType a, CompType beta)
{
    CompType y;
    if (std::fpclassify(beta - 0.75f) == FP_ZERO) {
        y = 1.0f / sqrtf((sqrtf(a) * a));
    } else {
        y = 1.0f / powf(a, beta);
    }
    return y;
};

inline void ComputeOffset(const Shape stridesSrc, SizeType iterN,
    SizeType iterC, SizeType iterD, SizeType iterH, SizeType iterW, SizeType &offset) noexcept
{
    offset = iterN * stridesSrc[IDX_0]  + iterC * stridesSrc[IDX_1]  + iterD * stridesSrc[IDX_2]  +
        iterH * stridesSrc[IDX_3]  + iterW * stridesSrc[IDX_4];
};

inline void ComputeSize(const SizeType &numDims, SizeType &size, bool acrossChannels,
    SizeType localSize) noexcept
{
    if (acrossChannels) {
        size = localSize;
    } else {
        size = 1;
        for (SizeType i = numDims - 2; i > 0; --i) {
            size *= localSize;
        }
    }
};

template <typename InType, typename ToType>
void ComputeSumAcross(const InType *src, ToType &sum,
    Shape strides, SizeType iterN, SizeType iterC, SizeType iterD, SizeType iterH, SizeType iterW,
    SizeType globalC, SizeType halfSize) noexcept
{
    SizeType startIndex = std::max<IntType>(iterC - halfSize, NUM_0);
    SizeType endIndex = std::min(iterC + halfSize + 1, globalC);
    for (SizeType cIter = startIndex; cIter < endIndex; ++cIter) {
        SizeType offset;
        Helpers::ComputeOffset(strides, iterN, cIter, iterD, iterH, iterW, offset);
        ToType s = Vector::Convert<InType, ToType>::Do(src[offset]);
        sum += s * s;
    }
};

template <typename InType, typename CompType>
void ComputeSumWithin(const InType *src, CompType &sum,
    Shape strides, SizeType iterN, SizeType iterC, SizeType iterD, SizeType iterH, SizeType iterW,
    SizeType d, SizeType h, SizeType w, SizeType halfSize) noexcept
{
    SizeType dstartIndex = std::max<IntType>(iterD - halfSize, NUM_0);
    SizeType dendIndex = std::min(iterD + halfSize + 1, d);
    SizeType hstartIndex = std::max<IntType>(iterH - halfSize, NUM_0);
    SizeType hendIndex = std::min(iterH + halfSize + 1, h);
    SizeType wstartIndex = std::max<IntType>(iterW - halfSize, NUM_0);
    SizeType wendIndex = std::min(iterW + halfSize + 1, w);
    for (SizeType dIter = dstartIndex; dIter < dendIndex; ++dIter) {
        for (SizeType hIter = hstartIndex; hIter < hendIndex; ++hIter) {
            for (SizeType wIter = wstartIndex; wIter < wendIndex; ++wIter) {
                SizeType offset;
                Helpers::ComputeOffset(strides, iterN, iterC, dIter, hIter, wIter, offset);
                CompType s = Vector::Convert<InType, CompType>::Do(src[offset]);
                sum += s * s;
            }
        }
    }
};

template <typename InType, typename CompType>
void SumOfSquares(const InType *src, CompType &sum, Shape strides, SizeType iterN, SizeType iterC, SizeType iterD,
    SizeType iterH, SizeType iterW, SizeType iter, SizeType iterEnd) noexcept
{
    svbool_t ldPG0;
    svbool_t ldPG1;
    svbool_t ldPG2;
    svbool_t ldPG3;
    svbool_t oppg0;
    svbool_t oppg1;
    svbool_t oppg2;
    svbool_t oppg3;
    Vector::InitPGN<InType, Vector::GetUsedRegsN<InType, CompType>()>::Do(
        ldPG0, ldPG1, ldPG2, ldPG3, iter, iterEnd);
    Vector::InitPGN<CompType, Vector::GetUsedRegsN<CompType, InType>()>::Do(
        oppg0, oppg1, oppg2, oppg3, iter, iterEnd);
    typename Vector::Vec<CompType>::Type data0;
    typename Vector::Vec<CompType>::Type data1;
    typename Vector::Vec<CompType>::Type data2;
    typename Vector::Vec<CompType>::Type data3;
    Vector::LoadN<InType, CompType,
        Vector::GetUsedRegsN<InType, CompType>()>::Do(
        src + iterN * strides[IDX_0]  + iterC * strides[IDX_1] + iterD * strides[IDX_2] +
        iterH * strides[IDX_3]  + iterW * strides[IDX_4],
        ldPG0, ldPG1, ldPG2, ldPG3, data0, data1, data2, data3);
    Vector::MulMN<CompType,
            Vector::GetUsedRegsN<CompType, InType>()>::Do(
        oppg0, oppg1, oppg2, oppg3, data0, data1, data2, data3,
        data0, data1, data2, data3, data0, data1, data2, data3);
    sum += Vector::AddvN<CompType,
        Vector::GetUsedRegsN<CompType, InType>()>::Do(
        oppg0, oppg1, oppg2, oppg3, data0, data1, data2, data3);
};
} // Helpers

// FWDAbx
template <typename CompType, typename InType, typename OutType>
void LocalResponseNormGenericAbxFWD(const InType *src, OutType *dst, float alpha, float beta, float k,
    SizeType localSize, const Shape &start, const Shape &end, const Shape &stridesSrc, const Shape &stridesDst,
    const Shape reorderedDimsSrc, const SizeType numDims,
    LRNormAlgorithmKind algorithm) noexcept
{
    SizeType vl = Vector::UnrollLength<InType, CompType, OutType>();

    SizeType c = reorderedDimsSrc[1];
    SizeType d = reorderedDimsSrc[2];
    SizeType h = reorderedDimsSrc[3];
    SizeType w = reorderedDimsSrc[4];

    SizeType size;
    bool acrossChannels = (algorithm == LRNormAlgorithmKind::ACROSS);
    Helpers::ComputeSize(numDims, size, acrossChannels, localSize);
    SizeType halfSize = (localSize - 1) / 2;

    for (SizeType iterN = start[IDX_0]; iterN < end[IDX_0]; ++iterN) {
        for (SizeType iterC = start[IDX_1]; iterC < end[IDX_1]; ++iterC) {
            for (SizeType iterD = start[IDX_2]; iterD < end[IDX_2]; ++iterD) {
                for (SizeType iterH = start[IDX_3]; iterH < end[IDX_3]; ++iterH) {
                    for (SizeType iterW = start[IDX_4]; iterW < end[IDX_4]; ++iterW) {
                        CompType sum = 0.0f;
                        SizeType srcOffset;
                        SizeType dstOffset;
                        Helpers::ComputeOffset(stridesSrc, iterN, iterC, iterD, iterH, iterW, srcOffset);
                        Helpers::ComputeOffset(stridesDst, iterN, iterC, iterD, iterH, iterW, dstOffset);
                        if (acrossChannels) {
                            Helpers::ComputeSumAcross(src, sum, stridesSrc, iterN, iterC, iterD, iterH, iterW,
                                c, halfSize);
                        } else {
                            SizeType dstartIndex = std::max<IntType>(iterD - halfSize, NUM_0);
                            SizeType dendIndex = std::min(iterD + halfSize + 1, d);
                            SizeType hstartIndex = std::max<IntType>(iterH - halfSize, NUM_0);
                            SizeType hendIndex = std::min(iterH + halfSize + 1, h);
                            SizeType wstartIndex = std::max<IntType>(iterW - halfSize, NUM_0);
                            SizeType wendIndex = std::min(iterW + halfSize + 1, w);
                            for (SizeType dIter = dstartIndex; dIter < dendIndex; ++dIter) {
                                for (SizeType hIter = hstartIndex; hIter < hendIndex; ++hIter) {
                                    for (SizeType wIter = wstartIndex; wIter < wendIndex; wIter += vl) {
                                        Helpers::SumOfSquares(src, sum, stridesSrc, iterN, iterC, dIter, hIter,
                                            wIter, wIter, wendIndex);
                                    }
                                }
                            }
                        }
                        sum = k + sum * alpha / static_cast<CompType>(size);
                        dst[dstOffset] = Vector::Convert<CompType, OutType>::Do(std::pow(sum, -beta) *
                            Vector::Convert <InType, CompType>::Do(src[srcOffset]));
                    }
                }
            }
        }
    }
};

// FWDAxb
template <typename CompType, typename InType, typename OutType>
void LocalResponseNormGenericAxbFWD(const InType *src, OutType *dst, float alpha, float beta, float k,
    SizeType localSize, const Shape &start, const Shape &end, const Shape &stridesSrc, const Shape &stridesDst,
    const Shape reorderedDimsSrc, const SizeType numDims,
    LRNormAlgorithmKind algorithm) noexcept
{
    SizeType vl = Vector::UnrollLength<InType, CompType, OutType>();

    bool acrossChannels = (algorithm == LRNormAlgorithmKind::ACROSS);
    SizeType c = reorderedDimsSrc[1];
    SizeType d = reorderedDimsSrc[2];
    SizeType h = reorderedDimsSrc[3];
    SizeType w = reorderedDimsSrc[4];

    SizeType size;

    Helpers::ComputeSize(numDims, size, acrossChannels, localSize);
    SizeType halfSize = (localSize - 1) / 2;

    for (SizeType iterN = start[IDX_0]; iterN < end[IDX_0]; ++iterN) {
        for (SizeType iterC = start[IDX_1]; iterC < end[IDX_1]; ++iterC) {
            for (SizeType iterD = start[IDX_2]; iterD < end[IDX_2]; ++iterD) {
                for (SizeType iterH = start[IDX_3]; iterH < end[IDX_3]; ++iterH) {
                    for (SizeType iterW = start[IDX_4]; iterW < end[IDX_4]; ++iterW) {
                        SizeType srcOffset;
                        SizeType dstOffset;
                        Helpers::ComputeOffset(stridesSrc, iterN, iterC, iterD, iterH, iterW, srcOffset);
                        Helpers::ComputeOffset(stridesDst, iterN, iterC, iterD, iterH, iterW, dstOffset);
                        CompType sum = 0.0f;
                        if (acrossChannels) {
                                SizeType startIndex = std::max<IntType>(iterC - halfSize, NUM_0);
                                SizeType endIndex = std::min(iterC + halfSize + 1, c);
                                for (SizeType cIter = startIndex; cIter < endIndex; cIter += vl) {
                                    Helpers::SumOfSquares(src, sum, stridesSrc, iterN, cIter, iterD, iterH,
                                        iterW, cIter, endIndex);
                                }
                        } else {
                                Helpers::ComputeSumWithin(src, sum, stridesSrc, iterN, iterC, iterD, iterH, iterW,
                                    d, h, w, halfSize);
                        }
                        sum = k + sum * alpha / static_cast<CompType>(size);
                        dst[dstOffset] = Vector::Convert<CompType, OutType>::Do(std::pow(sum, -beta) * Vector::Convert
                            <InType, CompType>::Do(src[srcOffset]));
                    }
                }
            }
        }
    }
};

// BWDAbx
template <typename CompType, typename InType, typename OutType>
void LocalResponseNormGenericAbxBWD(const InType *src, const OutType *diffDst, InType *diffSrc, float alpha, float beta,
    float k, SizeType localSize, const Shape &start, const Shape &end,
    const Shape &stridesSrc, const Shape &stridesDiffDst, const Shape &stridesDiffSrc,
    const Shape dimsSrc, const SizeType numDims, LRNormAlgorithmKind algorithm) noexcept
{
    SizeType c = dimsSrc[1];
    SizeType d = dimsSrc[2];
    SizeType h = dimsSrc[3];
    SizeType w = dimsSrc[4];
    SizeType size;
    bool acrossChannels = (algorithm == LRNormAlgorithmKind::ACROSS);
    SizeType vl = Vector::UnrollLength<InType, CompType, OutType>();
    Helpers::ComputeSize(numDims, size, acrossChannels, localSize);
    SizeType halfSize = (localSize - 1) / 2;
    for (SizeType iterN = start[IDX_0]; iterN < end[IDX_0]; ++iterN) {
        for (SizeType iterC = start[IDX_1]; iterC < end[IDX_1]; ++iterC) {
            for (SizeType iterD = start[IDX_2]; iterD < end[IDX_2]; ++iterD) {
                for (SizeType iterH = start[IDX_3]; iterH < end[IDX_3]; ++iterH) {
                    for (SizeType iterW = start[IDX_4]; iterW < end[IDX_4]; ++iterW) {
                        CompType a = 0.0f;
                        CompType b = 0.0f;
                        SizeType srcOffset;
                        SizeType diffDstOffset;
                        Helpers::ComputeOffset(stridesSrc, iterN, iterC, iterD, iterH, iterW, srcOffset);
                        Helpers::ComputeOffset(stridesDiffDst, iterN, iterC, iterD, iterH, iterW, diffDstOffset);
                        if (acrossChannels) {
                            SizeType startIndex = std::max<IntType>(iterC - halfSize, NUM_0);
                            SizeType endIndex = std::min(iterC + halfSize + 1, c);
                            for (SizeType i = startIndex; i < endIndex; ++i) {
                                SizeType diffDstOff;
                                SizeType srcOff;
                                CompType sum = 0.0f;
                                Helpers::ComputeOffset(stridesDiffDst, iterN, i, iterD, iterH, iterW, diffDstOff);
                                Helpers::ComputeOffset(stridesSrc, iterN, i, iterD, iterH, iterW, srcOff);
                                Helpers::ComputeSumAcross(src, sum, stridesSrc, iterN, i, iterD, iterH, iterW,
                                    c, halfSize);
                                CompType omega = k + alpha * sum / size;
                                CompType omegaToBeta = Helpers::ComputePow<CompType>(omega, beta);
                                CompType tmp = omegaToBeta * Vector::Convert<OutType, CompType>::
                                    Do(diffDst[diffDstOff]);
                                if (i == iterC) {
                                    a = tmp;
                                }
                                b += (tmp / omega * Vector::Convert<InType, CompType>::Do(src[srcOff]));
                            }
                        } else {
                            SizeType wstartIndex = std::max<IntType>(iterW - halfSize, NUM_0);
                            SizeType wendIndex = std::min(iterW + halfSize + 1, w);
                            SizeType dstartIndex = std::max<IntType>(iterD - halfSize, NUM_0);
                            SizeType dendIndex = std::min(iterD + halfSize + 1, d);
                            SizeType hstartIndex = std::max<IntType>(iterH - halfSize, NUM_0);
                            SizeType hendIndex = std::min(iterH + halfSize + 1, h);

                            for (SizeType dIter = dstartIndex; dIter < dendIndex; ++dIter) {
                                for (SizeType hIter = hstartIndex; hIter < hendIndex; ++hIter) {
                                    for (SizeType wIter = wstartIndex; wIter < wendIndex; ++wIter) {
                                        CompType sum = 0.0f;
                                        SizeType diffDstOff;
                                        SizeType srcOff;
                                        Helpers::ComputeOffset(stridesDiffDst, iterN, iterC, dIter, hIter,
                                            wIter, diffDstOff);
                                        Helpers::ComputeOffset(stridesSrc, iterN, iterC, dIter, hIter, wIter, srcOff);

                                        SizeType dstartIndex1 = std::max<IntType>(dIter - halfSize, NUM_0);
                                        SizeType dendIndex1 = std::min(dIter + halfSize + 1, d);
                                        SizeType hstartIndex1 = std::max<IntType>(hIter - halfSize, NUM_0);
                                        SizeType hendIndex1 = std::min(hIter + halfSize + 1, h);
                                        SizeType wstartIndex1 = std::max<IntType>(wIter - halfSize, NUM_0);
                                        SizeType wendIndex1 = std::min(wIter + halfSize + 1, w);
                                        for (SizeType dIter1 = dstartIndex1; dIter1 < dendIndex1; ++dIter1) {
                                            for (SizeType hIter1 = hstartIndex1; hIter1 < hendIndex1; ++hIter1) {
                                                for (SizeType wIter1 = wstartIndex1; wIter1 < wendIndex1;
                                                wIter1 += vl) {
                                                    Helpers::SumOfSquares(src, sum, stridesSrc, iterN, iterC,
                                                        dIter1, hIter1, wIter1, wIter1, wendIndex1);
                                                }
                                            }
                                        }
                                        CompType omega = k + alpha * sum / size;
                                        CompType omegaToBeta = Helpers::ComputePow<CompType>(omega, beta);
                                        CompType tmp = omegaToBeta * Vector::Convert<OutType, CompType>::
                                            Do(diffDst[diffDstOff]);
                                        if (dIter == iterD && hIter == iterH && wIter == iterW) {
                                            a = tmp;
                                        }
                                        b += (tmp / omega * Vector::Convert<InType, CompType>::
                                            Do(src[srcOff]));
                                    }
                                }
                            }
                        }
                        SizeType diffSrcOffset;
                        Helpers::ComputeOffset(stridesDiffSrc, iterN, iterC, iterD, iterH, iterW, diffSrcOffset);
                        b *= (2.0f * alpha * beta * Vector::Convert<InType, CompType>::
                            Do(src[srcOffset]) / size);
                        diffSrc[diffSrcOffset] = Vector::Convert<CompType, InType>::Do(a - b);
                    }
                }
            }
        }
    }
};

// BWDAxb
template <typename CompType, typename InType, typename OutType>
void LocalResponseNormGenericAxbBWD(const InType *src, const OutType *diffDst, InType *diffSrc, float alpha, float beta,
    float k, SizeType localSize, const Shape &start, const Shape &end,
    const Shape &stridesSrc, const Shape &stridesDiffDst, const Shape &stridesDiffSrc,
    const Shape dimsSrc, const SizeType numDims, LRNormAlgorithmKind algorithm) noexcept
{
    bool acrossChannels = (algorithm == LRNormAlgorithmKind::ACROSS);
    SizeType vl = Vector::UnrollLength<InType, CompType, OutType>();

    SizeType c = dimsSrc[1];
    SizeType d = dimsSrc[2];
    SizeType h = dimsSrc[3];
    SizeType w = dimsSrc[4];

    SizeType size;

    Helpers::ComputeSize(numDims, size, acrossChannels, localSize);
    SizeType halfSize = (localSize - 1) / 2;
    for (SizeType iterN = start[IDX_0]; iterN < end[IDX_0]; ++iterN) {
        for (SizeType iterC = start[IDX_1]; iterC < end[IDX_1]; ++iterC) {
            for (SizeType iterD = start[IDX_2]; iterD < end[IDX_2]; ++iterD) {
                for (SizeType iterH = start[IDX_3]; iterH < end[IDX_3]; ++iterH) {
                    for (SizeType iterW = start[IDX_4]; iterW < end[IDX_4]; ++iterW) {
                        SizeType srcOffset;
                        SizeType diffDstOffset;
                        Helpers::ComputeOffset(stridesSrc, iterN, iterC, iterD, iterH, iterW, srcOffset);
                        Helpers::ComputeOffset(stridesDiffDst, iterN, iterC, iterD, iterH, iterW, diffDstOffset);
                        CompType a = 0.0f;
                        CompType b = 0.0f;

                        if (acrossChannels) {
                            SizeType startIndex = std::max<IntType>(iterC - halfSize, NUM_0);
                            SizeType endIndex = std::min(iterC + halfSize + 1, c);
                            for (SizeType i = startIndex; i < endIndex; ++i) {
                                SizeType diffDstOff;
                                SizeType srcOff;
                                Helpers::ComputeOffset(stridesDiffDst, iterN, i, iterD, iterH, iterW, diffDstOff);
                                Helpers::ComputeOffset(stridesSrc, iterN, i, iterD, iterH, iterW, srcOff);
                                CompType sum = 0.0f;
      
                                SizeType startIndex1 = std::max<IntType>(i - halfSize, NUM_0);
                                SizeType endIndex1 = std::min(i + halfSize + 1, c);
                                for (SizeType cIter = startIndex1; cIter < endIndex1; cIter += vl) {
                                    Helpers::SumOfSquares(src, sum, stridesSrc, iterN, cIter, iterD, iterH,
                                        iterW, cIter, endIndex1);
                                }
                                CompType omega = k + alpha * sum / size;
                                CompType omegaToBeta = Helpers::ComputePow<CompType>(omega, beta);
                                CompType tmp = omegaToBeta *
                                    Vector::Convert<OutType, CompType>::Do(diffDst[diffDstOff]);
                                if (i == iterC) {
                                    a = tmp;
                                }
                                b += (tmp / omega * Vector::Convert<InType, CompType>::Do(src[srcOff]));
                            }
                        } else {
                            SizeType dstartIndex = std::max<IntType>(iterD - halfSize, NUM_0);
                            SizeType dendIndex = std::min(iterD + halfSize + 1, d);
                            SizeType hstartIndex = std::max<IntType>(iterH - halfSize, NUM_0);
                            SizeType hendIndex = std::min(iterH + halfSize + 1, h);
                            SizeType wstartIndex = std::max<IntType>(iterW - halfSize, NUM_0);
                            SizeType wendIndex = std::min(iterW + halfSize + 1, w);
                            for (SizeType dIter = dstartIndex; dIter < dendIndex; ++dIter) {
                                for (SizeType hIter = hstartIndex; hIter < hendIndex; ++hIter) {
                                    for (SizeType wIter = wstartIndex; wIter < wendIndex; ++wIter) {
                                        SizeType diffDstOff;
                                        SizeType srcOff;
                                        Helpers::ComputeOffset(stridesDiffDst, iterN, iterC, dIter, hIter,
                                            wIter, diffDstOff);
                                        Helpers::ComputeOffset(stridesSrc, iterN, iterC, dIter, hIter, wIter, srcOff);
                                        CompType sum = 0.0f;
                                        Helpers::ComputeSumWithin(src, sum, stridesSrc, iterN, iterC, dIter, hIter,
                                            wIter, d, h, w, halfSize);
                                        CompType omega = k + alpha * sum / size;
                                        CompType omegaToBeta = Helpers::ComputePow<CompType>(omega, beta);
                                        CompType tmp = omegaToBeta *
                                            Vector::Convert<OutType, CompType>::Do(diffDst[diffDstOff]);
                                        if (dIter == iterD && hIter == iterH && wIter == iterW) {
                                            a = tmp;
                                        }
                                        b += (tmp / omega * Vector::Convert<InType, CompType>::Do(src[srcOff]));
                                    }
                                }
                            }
                        }
                        SizeType diffSrcOffset;
                        Helpers::ComputeOffset(stridesDiffSrc, iterN, iterC, iterD, iterH, iterW, diffSrcOffset);
                        b *= (2.0f * alpha * beta * Vector::Convert<InType, CompType>::Do(src[srcOffset]) / size);
                        diffSrc[diffSrcOffset] = Vector::Convert<CompType, InType>::Do(a - b);
                    }
                }
            }
        }
    }
};

} // LRN
} // Detail
} // KuDNN

#endif //  KUDNN_LRN_DEF_HPP