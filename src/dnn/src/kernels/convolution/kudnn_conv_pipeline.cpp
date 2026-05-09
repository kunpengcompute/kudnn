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

#include <vector>

#ifdef TIMING
#include <iostream>
#include <chrono>
#include <iostream>
#include <iomanip>
#endif

#include <omp.h>

#include "kudnn.hpp"

#include "vector_wrappers.hpp"
#include "kernels/convolution/kudnn_conv_kernels.hpp"
#include "kudnn_conv_deconv_base.hpp"
#include "parallel_for.h"

namespace KuDNN {
namespace Detail {

static constexpr IntType CONV_THREAD_THREADS = 250'000;

constexpr IntType L1_SIZE = 65'536;    // in bytes
constexpr IntType L2_SIZE = 1'310'720; // in bytes
constexpr IntType L3_SIZE = 4'194'304; // in bytes

static std::vector<std::pair<Shape, Shape>> GetThreadPartition(IntType OptNumThreads, IntType ib, IntType oh,
                                                               IntType oc, IntType ohBlock, IntType ow, IntType ocBlock,
                                                               IntType kernelOWSize, IntType kernelOCSize)
{
    Service::Unused(OptNumThreads); // partition will be dependent on thread qty
    std::vector<std::pair<Shape, Shape>> partition;
    SizeType ocInc = static_cast<SizeType>(ocBlock);      // ocBlock oc
    SizeType ohInc = static_cast<SizeType>(NUM_1);        // NUM_1 ohBlock
    SizeType owInc = static_cast<SizeType>(kernelOWSize); // kernelOWSize ow
    assert(ohBlock != 0 && kernelOCSize != 0);
    partition.reserve(ib * ((oh + ohBlock - NUM_1) / ohBlock) * ((oc + ocBlock - NUM_1) / ocBlock) * ohBlock *
                      ((ow + kernelOWSize - NUM_1) / kernelOCSize));

    for (SizeType b = NUM_0; b < static_cast<SizeType>(ib); ++b) {
        for (SizeType ohB = NUM_0; ohB < static_cast<SizeType>(oh); ohB += static_cast<SizeType>(ohBlock)) {
            for (SizeType ocIter = NUM_0; ocIter < static_cast<SizeType>(oc); ocIter += ocInc) {
                for (SizeType ohIter = NUM_0; ohIter < static_cast<SizeType>(ohBlock); ohIter += ohInc) {
                    for (SizeType owIter = NUM_0; owIter < static_cast<SizeType>(ow); owIter += owInc) {
                        partition.push_back(
                            {{b, ohB, ocIter, ohIter, owIter},
                             {b + NUM_1, ohB + ohBlock, ocIter + ocInc, ohIter + ohInc, owIter + owInc}});
                    }
                }
            }
        }
    }
    return partition;
}

template <typename DstType>
static void FillDstKernelNCHW(DstType *dst, const DstType *bia, IntType ocBlock, IntType ohB, IntType ow, IntType oh,
                              IntType oc, IntType b, const std::pair<Shape, Shape> &curWork, IntType kernelOWSize,
                              IntType kernelOCSize, IntType residOHSize)
{
    for (IntType ocB = curWork.first[NUM_2]; ocB < static_cast<IntType>(curWork.second[NUM_2]); ocB += ocBlock) {
        IntType residOCBSize = std::min(oc - ocB, ocBlock);
        for (IntType ocSB = NUM_0; ocSB < residOCBSize; ocSB += kernelOCSize) {
            IntType residOCSize = std::min(residOCBSize - ocSB, kernelOCSize);
            for (IntType ocIter = NUM_0; ocIter < residOCSize; ++ocIter) {
                DstType fillValue = NUM_0;
                if (bia != nullptr) {
                    fillValue = bia[ocIter + ocSB + ocB];
                }
                for (IntType ohIter = curWork.first[NUM_3]; ohIter < residOHSize; ++ohIter) {
                    for (IntType owB = curWork.first[NUM_4]; owB < static_cast<IntType>(curWork.second[NUM_4]);
                         owB += kernelOWSize) {
                        IntType residOWSize = std::min(ow - owB, kernelOWSize);
                        for (IntType owIter = NUM_0; owIter < residOWSize; owIter += Vector::Length<DstType>()) {
                            svbool_t pg0;
                            Vector::WhileLt<DstType>::Do(pg0, owIter, residOWSize);
                            typename Vector::Vec<DstType>::Type dst0;
                            Vector::DupFloat<DstType>::Do(fillValue, dst0);
                            svst1(pg0,
                                  dst + (owIter + owB) + ow * (ohIter + ohB + oh * ((ocIter + ocSB + ocB) + oc * b)),
                                  dst0);
                        }
                    }
                }
            }
        }
    }
}

template <typename DstType>
static inline void FillDstKernelBias(DstType *dst, const DstType *bia, IntType ocBlock, IntType kernelOCSize,
                                     IntType owIter, IntType ohIter, IntType b, IntType ohB, IntType ow, IntType oh,
                                     IntType oc, IntType owB, const std::pair<Shape, Shape> &curWork)
{
    for (IntType ocB = curWork.first[NUM_2]; ocB < static_cast<IntType>(curWork.second[NUM_2]); ocB += ocBlock) {
        IntType residOCBSize = std::min(oc - ocB, ocBlock);
        for (IntType ocSB = NUM_0; ocSB < residOCBSize; ocSB += kernelOCSize) {
            IntType residOCSize = std::min(residOCBSize - ocSB, kernelOCSize);
            svbool_t pg0;
            Vector::WhileLt<DstType>::Do(pg0, NUM_0, residOCSize);
            svbool_t pg1;
            Vector::WhileLt<DstType>::Do(pg1, Vector::Length<DstType>(), residOCSize);
            svbool_t pg2;
            Vector::WhileLt<DstType>::Do(pg2, NUM_2 * Vector::Length<DstType>(), residOCSize);

            typename Vector::Vec<DstType>::Type dst0;
            typename Vector::Vec<DstType>::Type dst1;
            typename Vector::Vec<DstType>::Type dst2;

            dst0 = svld1(pg0, bia + ocSB + ocB);
            dst1 = svld1(pg1, bia + Vector::Length<DstType>() + ocSB + ocB);
            dst2 = svld1(pg2, bia + NUM_2 * Vector::Length<DstType>() + ocSB + ocB);

            svst1(pg0, dst + ocSB + ocB + oc * (owIter + owB + ow * (ohIter + ohB + oh * b)), dst0);
            svst1(pg1,
                  dst + Vector::Length<DstType>() + ocSB + ocB + oc * (owIter + owB + ow * (ohIter + ohB + oh * b)),
                  dst1);
            svst1(pg2,
                  dst + NUM_2 * Vector::Length<DstType>() + ocSB + ocB +
                      oc * (owIter + owB + ow * (ohIter + ohB + oh * b)),
                  dst2);
        }
    }
}

template <typename DstType>
static inline void FillDstKernel(DstType *dst, IntType ocBlock, IntType kernelOCSize, IntType owIter, IntType ohIter,
                                 IntType b, IntType ohB, IntType ow, IntType oh, IntType oc, IntType owB,
                                 const std::pair<Shape, Shape> &curWork)
{
    for (IntType ocB = curWork.first[NUM_2]; ocB < static_cast<IntType>(curWork.second[NUM_2]); ocB += ocBlock) {
        IntType residOCBSize = std::min(oc - ocB, ocBlock);
        for (IntType ocSB = NUM_0; ocSB < residOCBSize; ocSB += kernelOCSize) {
            IntType residOCSize = std::min(residOCBSize - ocSB, kernelOCSize);
            svbool_t pg0;
            Vector::WhileLt<DstType>::Do(pg0, NUM_0, residOCSize);
            svbool_t pg1;
            Vector::WhileLt<DstType>::Do(pg1, Vector::Length<DstType>(), residOCSize);
            svbool_t pg2;
            Vector::WhileLt<DstType>::Do(pg2, NUM_2 * Vector::Length<DstType>(), residOCSize);

            typename Vector::Vec<DstType>::Type dst0;
            typename Vector::Vec<DstType>::Type dst1;
            typename Vector::Vec<DstType>::Type dst2;

            Vector::DupFloat<DstType>::Do(0.0f, dst0);
            Vector::DupFloat<DstType>::Do(0.0f, dst1);
            Vector::DupFloat<DstType>::Do(0.0f, dst2);

            svst1(pg0, dst + ocSB + ocB + oc * (owIter + owB + ow * (ohIter + ohB + oh * b)), dst0);
            svst1(pg1,
                  dst + Vector::Length<DstType>() + ocSB + ocB + oc * (owIter + owB + ow * (ohIter + ohB + oh * b)),
                  dst1);
            svst1(pg2,
                  dst + NUM_2 * Vector::Length<DstType>() + ocSB + ocB +
                      oc * (owIter + owB + ow * (ohIter + ohB + oh * b)),
                  dst2);
        }
    }
}

template <typename DstType>
void FillDst(DstType *dst, const DstType *bia, const ConvolutionParams &params, const std::pair<Shape, Shape> &curWork,
             IntType ocBlock, IntType residOHSize, IntType kernelOCSize, IntType kernelOWSize, IntType ohB, IntType b)
{
    IntType oc = params.outputShape[NUM_1];
    IntType oh = params.outputShape[NUM_3];
    IntType ow = params.outputShape[NUM_4];
    if (params.outputLayout == Layout::NCHW) {
        FillDstKernelNCHW(dst, bia, ocBlock, ohB, ow, oh, oc, b, curWork, kernelOWSize, kernelOCSize, residOHSize);
    } else {
        for (IntType ohIter = curWork.first[NUM_3]; ohIter < residOHSize; ++ohIter) {
            for (IntType owB = curWork.first[NUM_4]; owB < static_cast<IntType>(curWork.second[NUM_4]);
                 owB += kernelOWSize) {
                IntType residOWSize = std::min(ow - owB, kernelOWSize);
                for (IntType owIter = NUM_0; owIter < residOWSize; ++owIter) {
                    if (bia != nullptr) {
                        FillDstKernelBias(dst, bia, ocBlock, kernelOCSize, owIter, ohIter, b, ohB, ow, oh, oc, owB,
                                          curWork);
                    } else {
                        FillDstKernel(dst, ocBlock, kernelOCSize, owIter, ohIter, b, ohB, ow, oh, oc, owB, curWork);
                    }
                }
            }
        }
    }
}

template <typename SrcType, typename ComputeType>
static void PackSrcKernelNCHW(const SrcType *src, ComputeType *buf, IntType icBlock, IntType ihSize, IntType kernelSize,
                              IntType iwSize, IntType ih, IntType iw, IntType ic, IntType iwBuf, IntType iwB,
                              IntType ihB, IntType icB, IntType b)
{
    src += iwB + iw * (ihB + ih * (icB + ic * b));
    for (IntType icIter = NUM_0; icIter < icBlock; ++icIter) {
        for (IntType ihIter = NUM_0; ihIter < ihSize; ++ihIter) {
            for (IntType iwIter = NUM_0; iwIter < iwSize; iwIter += Vector::Length<ComputeType>()) {
                svbool_t pg0;
                Vector::WhileLt<SrcType>::Do(pg0, iwIter, iwSize);
                typename Vector::Vec<ComputeType>::Type src0 = svld1(pg0, src + iwIter + iw * (ihIter + ih * icIter));
                svst1(pg0, buf + iwIter + iwBuf * (ihIter + kernelSize * icIter), src0);
            }
            for (IntType iwIter = iwSize; iwIter < iwBuf; iwIter += Vector::Length<ComputeType>()) {
                svbool_t pg0;
                Vector::WhileLt<SrcType>::Do(pg0, iwIter, iwBuf);
                typename Vector::Vec<ComputeType>::Type src0;
                Vector::DupFloat<ComputeType>::Do(0.0f, src0);
                svst1(pg0, buf + iwIter + iwBuf * (ihIter + kernelSize * icIter), src0);
            }
        }
        for (IntType ihIter = ihSize; ihIter < kernelSize; ++ihIter) {
            for (IntType iwIter = NUM_0; iwIter < iwBuf; iwIter += Vector::Length<ComputeType>()) {
                svbool_t pg0;
                Vector::WhileLt<SrcType>::Do(pg0, iwIter, iwBuf);
                typename Vector::Vec<ComputeType>::Type src0;
                Vector::DupFloat<ComputeType>::Do(0.0f, src0);
                svst1(pg0, buf + iwIter + iwBuf * (ihIter + kernelSize * icIter), src0);
            }
        }
    }
}
template <typename ComputeType, typename SrcType>
void PackSrc(Layout lt, const SrcType *src, ComputeType *buf, IntType icBlock, IntType ihSize, IntType kernelSize,
             IntType iwSize, IntType ih, IntType iw, IntType ic, IntType iwBuf, IntType iwB, IntType ihB, IntType icB,
             IntType b)
{
    if (lt == Layout::NCHW) {
        PackSrcKernelNCHW(src, buf, icBlock, ihSize, kernelSize, iwSize, ih, iw, ic, iwBuf, iwB, ihB, icB, b);
    } else {
        src += icB + ic * (iwB + iw * (ihB + ih * b));
        for (IntType ihIter = NUM_0; ihIter < ihSize; ++ihIter) {
            for (IntType iwIter = NUM_0; iwIter < iwSize; iwIter += Vector::Length<ComputeType>()) {
                svbool_t pg0;
                Vector::WhileLt<SrcType>::Do(pg0, iwIter, iwSize);
                for (IntType icIter = NUM_0; icIter < icBlock; ++icIter) {
                    typename Vector::Vec<ComputeType>::Type src0;
                    Vector::GatherLoad<SrcType>::Do(src + icIter + ic * (iwIter + iw * ihIter), pg0, ic, src0);
                    svst1(pg0, buf + iwIter + iwBuf * (ihIter + kernelSize * icIter), src0);
                }
            }
            for (IntType icIter = NUM_0; icIter < icBlock; ++icIter) {
                for (IntType iwIter = iwSize; iwIter < iwBuf; iwIter += Vector::Length<ComputeType>()) {
                    svbool_t pg0;
                    Vector::WhileLt<SrcType>::Do(pg0, iwIter, iwBuf);
                    typename Vector::Vec<ComputeType>::Type src0;
                    Vector::DupFloat<ComputeType>::Do(0.0f, src0);
                    svst1(pg0, buf + iwIter + iwBuf * (ihIter + kernelSize * icIter), src0);
                }
            }
        }
        for (IntType icIter = NUM_0; icIter < icBlock; ++icIter) {
            for (IntType ihIter = ihSize; ihIter < kernelSize; ++ihIter) {
                for (IntType iwIter = NUM_0; iwIter < iwBuf; iwIter += Vector::Length<ComputeType>()) {
                    svbool_t pg0;
                    Vector::WhileLt<SrcType>::Do(pg0, iwIter, iwBuf);
                    typename Vector::Vec<ComputeType>::Type src0;
                    Vector::DupFloat<ComputeType>::Do(0.0, src0);
                    svst1(pg0, buf + iwIter + iwBuf * (ihIter + kernelSize * icIter), src0);
                }
            }
        }
    }
}

template <typename ComputeType, typename SrcType>
static void PackSrcBordersKernelNCHW(const SrcType *src, ComputeType *buf, IntType icBlock, IntType ihStart,
                                     IntType iwStart, IntType kernelSize, IntType ihEnd, IntType iwEnd, IntType ih,
                                     IntType iw, IntType ic, IntType iwBuf, IntType iwB, IntType ihB, IntType icB,
                                     IntType b, IntType pw)
{
    src += iwB + iw * (ihB + ih * (icB + ic * b));
    for (IntType icIter = NUM_0; icIter < icBlock; ++icIter) {
        for (IntType ihIter = NUM_0; ihIter < ihStart; ++ihIter) {
            for (IntType iwIter = NUM_0; iwIter < iwBuf; iwIter += Vector::Length<ComputeType>()) {
                svbool_t pg0;
                Vector::WhileLt<SrcType>::Do(pg0, iwIter, iwBuf);
                typename Vector::Vec<ComputeType>::Type src0;
                Vector::DupFloat<ComputeType>::Do(0.0f, src0);
                svst1(pg0, buf + iwIter + iwBuf * (ihIter + kernelSize * icIter), src0);
            }
        }
        for (IntType ihIter = ihStart; ihIter < ihEnd; ++ihIter) {
            for (IntType iwIter = NUM_0; iwIter < iwBuf; iwIter += Vector::Length<ComputeType>()) {
                svbool_t pg0;
                Vector::WhileLt<SrcType>::Do(pg0, iwIter, iwBuf);
                typename Vector::Vec<ComputeType>::Type src0;
                Vector::DupFloat<ComputeType>::Do(0.0f, src0);
                svst1(pg0, buf + iwIter + iwBuf * (ihIter + kernelSize * icIter), src0);
            }
            IntType actIw = (pw >= (iwBuf - iw)) && ((iwBuf - iwEnd) == NUM_1) && (iwBuf > iw) ? iwBuf : iwEnd;
            actIw = (iwB + actIw) > iw ? iwEnd : actIw;
            for (IntType iwIter = iwStart; iwIter < iwBuf; iwIter += Vector::Length<ComputeType>()) {
                svbool_t pg0;
                Vector::WhileLt<SrcType>::Do(pg0, iwIter, actIw);
                typename Vector::Vec<ComputeType>::Type src0 = svld1(pg0, src + iwIter + iw * (ihIter + ih * icIter));
                svst1(pg0, buf + iwIter + iwBuf * (ihIter + kernelSize * icIter), src0);
            }
        }
        for (IntType ihIter = ihEnd; ihIter < kernelSize; ++ihIter) {
            for (IntType iwIter = NUM_0; iwIter < iwBuf; iwIter += Vector::Length<ComputeType>()) {
                svbool_t pg0;
                Vector::WhileLt<SrcType>::Do(pg0, iwIter, iwBuf);
                typename Vector::Vec<ComputeType>::Type src0;
                Vector::DupFloat<ComputeType>::Do(0.0f, src0);
                svst1(pg0, buf + iwIter + iwBuf * (ihIter + kernelSize * icIter), src0);
            }
        }
    }
}

template <typename ComputeType, typename SrcType>
static void PackSrcBordersKernel(const SrcType *src, ComputeType *buf, IntType icBlock, IntType ihStart,
                                 IntType iwStart, IntType kernelSize, IntType ihEnd, IntType iwEnd, IntType ih,
                                 IntType iw, IntType ic, IntType iwBuf, IntType iwB, IntType ihB, IntType icB,
                                 IntType b, IntType pw)
{
    src += icB + ic * (iwB + iw * (ihB + ih * b));
    for (IntType icIter = NUM_0; icIter < icBlock; ++icIter) {
        for (IntType ihIter = NUM_0; ihIter < ihStart; ++ihIter) {
            for (IntType iwIter = NUM_0; iwIter < iwBuf; iwIter += Vector::Length<ComputeType>()) {
                svbool_t pg0;
                Vector::WhileLt<SrcType>::Do(pg0, iwIter, iwBuf);
                typename Vector::Vec<ComputeType>::Type src0;
                Vector::DupFloat<ComputeType>::Do(0.0f, src0);
                svst1(pg0, buf + iwIter + iwBuf * (ihIter + kernelSize * icIter), src0);
            }
        }
        for (IntType ihIter = ihStart; ihIter < ihEnd; ++ihIter) {
            for (IntType iwIter = NUM_0; iwIter < iwBuf; iwIter += Vector::Length<ComputeType>()) {
                svbool_t pg0;
                Vector::WhileLt<SrcType>::Do(pg0, iwIter, iwBuf);
                typename Vector::Vec<ComputeType>::Type src0;
                Vector::DupFloat<ComputeType>::Do(0.0f, src0);
                svst1(pg0, buf + iwIter + iwBuf * (ihIter + kernelSize * icIter), src0);
            }
        }
    }
    for (IntType ihIter = ihStart; ihIter < ihEnd; ++ihIter) {
        for (IntType iwIter = iwStart; iwIter < iwEnd; iwIter += Vector::Length<ComputeType>()) {
            IntType actIw = (pw >= (iwBuf - iw)) && ((iwBuf - iwEnd) == NUM_1) && (iwBuf > iw) ? iwBuf : iwEnd;
            actIw = (iwB + actIw) > iw ? iwEnd : actIw;
            for (IntType icIter = NUM_0; icIter < icBlock; ++icIter) {
                svbool_t pg0;
                Vector::WhileLt<SrcType>::Do(pg0, iwIter, actIw);
                typename Vector::Vec<ComputeType>::Type src0;
                Vector::GatherLoad<SrcType>::Do(src + icIter + ic * (iwIter + iw * ihIter), pg0, ic, src0);
                svst1(pg0, buf + iwIter + iwBuf * (ihIter + kernelSize * icIter), src0);
            }
        }
    }
    for (IntType icIter = NUM_0; icIter < icBlock; ++icIter) {
        for (IntType ihIter = ihEnd; ihIter < kernelSize; ++ihIter) {
            for (IntType iwIter = NUM_0; iwIter < iwBuf; iwIter += Vector::Length<ComputeType>()) {
                svbool_t pg0;
                Vector::WhileLt<SrcType>::Do(pg0, iwIter, iwBuf);
                typename Vector::Vec<ComputeType>::Type src0;
                Vector::DupFloat<ComputeType>::Do(0.0f, src0);
                svst1(pg0, buf + iwIter + iwBuf * (ihIter + kernelSize * icIter), src0);
            }
        }
    }
}

template <typename ComputeType, typename SrcType>
void PackSrcBorders(Layout lt, const SrcType *src, ComputeType *buf, IntType icBlock, IntType ihExt, IntType iwExt,
                    IntType kernelSize, IntType iwSize, IntType ih, IntType iw, IntType ic, IntType iwBuf, IntType iwB,
                    IntType ihB, IntType icB, IntType b, IntType pw)
{
    IntType ihStart = (ihExt > ih) ? IntType{NUM_0} : std::max(IntType{NUM_0}, -ihExt);
    ihStart = std::min(ihStart, kernelSize);
    IntType ihEnd = ((ihExt + kernelSize) > ih) ? (kernelSize - (ihExt + kernelSize - ih)) : kernelSize;
    ihEnd = std::max(ihEnd, IntType{NUM_0});
    IntType iwStart = (iwExt > iw) ? IntType{NUM_0} : std::max(IntType{NUM_0}, -iwExt);
    iwStart = std::min(iwStart, iwBuf);
    IntType iwEnd = ((iwExt + iwBuf) > iw) ? (iwBuf - (iwExt + iwBuf - iw)) : iwSize;
    iwEnd = std::max(iwEnd, IntType{NUM_0});
    iwEnd = std::min(iwEnd, iwBuf);
    if (lt == Layout::NCHW) {
        PackSrcBordersKernelNCHW(src, buf, icBlock, ihStart, iwStart, kernelSize, ihEnd, iwEnd, ih, iw, ic, iwBuf, iwB,
                                 ihB, icB, b, pw);
    } else {
        PackSrcBordersKernel(src, buf, icBlock, ihStart, iwStart, kernelSize, ihEnd, iwEnd, ih, iw, ic, iwBuf, iwB, ihB,
                             icB, b, pw);
    }
}

template <typename WeiType, typename ComputeType>
static inline void PackWeiKernelNCHW(svbool_t pg0, svbool_t pg1, svbool_t pg2, SizeType offset, const WeiType *wei,
                                     ComputeType *buf, IntType icBlock, IntType ocBlock, IntType ic, IntType kernelSize,
                                     IntType kernelOCSize, IntType ocSB, IntType ocB)
{
    for (IntType icB = NUM_0; icB < ic; icB += icBlock) {
        IntType residICSize = std::min(ic - icB, icBlock);
        for (IntType icIter = NUM_0; icIter < residICSize; ++icIter) {
            for (IntType kh = NUM_0; kh < kernelSize; ++kh) {
                for (IntType kw = NUM_0; kw < kernelSize; ++kw) {
                    typename Vector::Vec<ComputeType>::Type wei0;
                    typename Vector::Vec<ComputeType>::Type wei1;
                    typename Vector::Vec<ComputeType>::Type wei2;

                    Vector::GatherLoad<WeiType>::Do(
                        wei + kw +
                            kernelSize *
                                (kh + kernelSize *
                                          (icIter + icB + ic * (NUM_0 * Vector::Length<ComputeType>() + ocSB + ocB))),
                        pg0, offset, wei0);
                    Vector::GatherLoad<WeiType>::Do(
                        wei + kw +
                            kernelSize *
                                (kh + kernelSize *
                                          (icIter + icB + ic * (NUM_1 * Vector::Length<ComputeType>() + ocSB + ocB))),
                        pg1, offset, wei1);
                    Vector::GatherLoad<WeiType>::Do(
                        wei + kw +
                            kernelSize *
                                (kh + kernelSize *
                                          (icIter + icB + ic * (NUM_2 * Vector::Length<ComputeType>() + ocSB + ocB))),
                        pg2, offset, wei2);

                    svst1(pg0,
                          buf + NUM_0 * Vector::Length<ComputeType>() +
                              kernelOCSize *
                                  (kw + kernelSize * (kh + kernelSize * (icIter + icBlock * ocSB / kernelOCSize))) +
                              kernelSize * kernelSize *
                                  (icB * ocBlock + ((ic + icBlock - NUM_1) / icBlock) * icBlock * ocB),
                          wei0);
                    svst1(pg1,
                          buf + NUM_1 * Vector::Length<ComputeType>() +
                              kernelOCSize *
                                  (kw + kernelSize * (kh + kernelSize * (icIter + icBlock * ocSB / kernelOCSize))) +
                              kernelSize * kernelSize *
                                  (icB * ocBlock + ((ic + icBlock - NUM_1) / icBlock) * icBlock * ocB),
                          wei1);
                    svst1(pg2,
                          buf + NUM_2 * Vector::Length<ComputeType>() +
                              kernelOCSize *
                                  (kw + kernelSize * (kh + kernelSize * (icIter + icBlock * ocSB / kernelOCSize))) +
                              kernelSize * kernelSize *
                                  (icB * ocBlock + ((ic + icBlock - NUM_1) / icBlock) * icBlock * ocB),
                          wei2);
                }
            }
        }
    }
}

template <typename WeiType, typename ComputeType>
static inline void PackWeiKernel(svbool_t pg0, svbool_t pg1, svbool_t pg2, SizeType offset, const WeiType *wei,
                                 ComputeType *buf, IntType icBlock, IntType ocBlock, IntType ic, IntType kernelSize,
                                 IntType kernelOCSize, IntType ocSB, IntType ocB)
{
    for (IntType kh = NUM_0; kh < kernelSize; ++kh) {
        for (IntType kw = NUM_0; kw < kernelSize; ++kw) {
            for (IntType icB = NUM_0; icB < ic; icB += icBlock) {
                IntType residICSize = std::min(ic - icB, icBlock);
                for (IntType icIter = NUM_0; icIter < residICSize; ++icIter) {
                    typename Vector::Vec<ComputeType>::Type wei0;
                    typename Vector::Vec<ComputeType>::Type wei1;
                    typename Vector::Vec<ComputeType>::Type wei2;

                    Vector::GatherLoad<WeiType>::Do(
                        wei + icIter + icB +
                            ic * (kw + kernelSize *
                                           (kh + kernelSize * (NUM_0 * Vector::Length<ComputeType>() + ocSB + ocB))),
                        pg0, offset, wei0);
                    Vector::GatherLoad<WeiType>::Do(
                        wei + icIter + icB +
                            ic * (kw + kernelSize *
                                           (kh + kernelSize * (NUM_1 * Vector::Length<ComputeType>() + ocSB + ocB))),
                        pg1, offset, wei1);
                    Vector::GatherLoad<WeiType>::Do(
                        wei + icIter + icB +
                            ic * (kw + kernelSize *
                                           (kh + kernelSize * (NUM_2 * Vector::Length<ComputeType>() + ocSB + ocB))),
                        pg2, offset, wei2);

                    svst1(pg0,
                          buf + NUM_0 * Vector::Length<ComputeType>() +
                              kernelOCSize *
                                  (kw + kernelSize * (kh + kernelSize * (icIter + icBlock * ocSB / kernelOCSize))) +
                              kernelSize * kernelSize *
                                  (icB * ocBlock + ((ic + icBlock - NUM_1) / icBlock) * icBlock * ocB),
                          wei0);
                    svst1(pg1,
                          buf + NUM_1 * Vector::Length<ComputeType>() +
                              kernelOCSize *
                                  (kw + kernelSize * (kh + kernelSize * (icIter + icBlock * ocSB / kernelOCSize))) +
                              kernelSize * kernelSize *
                                  (icB * ocBlock + ((ic + icBlock - NUM_1) / icBlock) * icBlock * ocB),
                          wei1);
                    svst1(pg2,
                          buf + NUM_2 * Vector::Length<ComputeType>() +
                              kernelOCSize *
                                  (kw + kernelSize * (kh + kernelSize * (icIter + icBlock * ocSB / kernelOCSize))) +
                              kernelSize * kernelSize *
                                  (icB * ocBlock + ((ic + icBlock - NUM_1) / icBlock) * icBlock * ocB),
                          wei2);
                }
            }
        }
    }
}

template <typename ComputeType, typename WeiType>
void PackWei(Layout lt, const WeiType *wei, ComputeType *buf, IntType icBlock, IntType ocBlock, IntType ic, IntType oc,
             IntType kernelSize, IntType kernelOCSize, IntType optNumThreads)
{
    std::vector<std::pair<Shape, Shape>> partition;
    {
        partition.reserve(oc * ic);
        for (SizeType ocB = NUM_0; ocB < static_cast<SizeType>(oc); ocB += ocBlock) {
            IntType residOCBSize = std::min(oc - static_cast<IntType>(ocB), ocBlock);
            for (SizeType ocSB = NUM_0; ocSB < static_cast<SizeType>(residOCBSize); ocSB += kernelOCSize) {
                IntType residOCSize = std::min(residOCBSize - static_cast<IntType>(ocSB), kernelOCSize);
                partition.push_back({{ocB, ocSB}, {ocB + residOCBSize, ocSB + residOCSize}});
            }
        }
    }

    Threading::ParallelFor::instance().run(
        [&](int64_t i) {
            std::pair<Shape, Shape> &curWork = partition[i];
            for (IntType ocB = curWork.first[NUM_0]; ocB < static_cast<IntType>(curWork.second[NUM_0]);
                 ocB += ocBlock) {
                IntType residOCBSize = std::min(oc - ocB, ocBlock);
                for (IntType ocSB = curWork.first[NUM_1]; ocSB < static_cast<IntType>(curWork.second[NUM_1]);
                     ocSB += kernelOCSize) {
                    IntType residOCSize = std::min(residOCBSize - ocSB, kernelOCSize);
                    svbool_t pg0;
                    Vector::WhileLt<WeiType>::Do(pg0, NUM_0, residOCSize);
                    svbool_t pg1;
                    Vector::WhileLt<WeiType>::Do(pg1, Vector::Length<ComputeType>(), residOCSize);
                    svbool_t pg2;
                    Vector::WhileLt<WeiType>::Do(pg2, NUM_2 * Vector::Length<ComputeType>(), residOCSize);
                    SizeType offset = kernelSize * kernelSize * ic;
                    if (lt == Layout::NCHW) {
                        PackWeiKernelNCHW(pg0, pg1, pg2, offset, wei, buf, icBlock, ocBlock, ic, kernelSize,
                                          kernelOCSize, ocSB, ocB);
                    } else {
                        PackWeiKernel(pg0, pg1, pg2, offset, wei, buf, icBlock, ocBlock, ic, kernelSize, kernelOCSize,
                                      ocSB, ocB);
                    }
                }
            }
        },
        static_cast<int64_t>(partition.size()), optNumThreads);
}

template <Layout Lt, IntType KernelSize, IntType ExecSize, typename CompType>
void KernelParser(IntType sw, const CompType *src, const CompType *wei, CompType *dst, IntType lastStride,
                  IntType lastIdx, IntType icBlock)
{
    switch (sw) {
        case NUM_1: {
            ConvKernel<CompType>::template Do<Lt, KernelSize, ExecSize, NUM_1>(src, wei, dst, lastStride, lastIdx,
                                                                               icBlock);
            break;
        }
        case NUM_2: {
            ConvKernel<CompType>::template Do<Lt, KernelSize, ExecSize, NUM_2>(src, wei, dst, lastStride, lastIdx,
                                                                               icBlock);
            break;
        }
        case NUM_3: {
            ConvKernel<CompType>::template Do<Lt, KernelSize, ExecSize, NUM_3>(src, wei, dst, lastStride, lastIdx,
                                                                               icBlock);
            break;
        }
        case NUM_4: {
            ConvKernel<CompType>::template Do<Lt, KernelSize, ExecSize, NUM_4>(src, wei, dst, lastStride, lastIdx,
                                                                               icBlock);
            break;
        }
        case NUM_5: {
            ConvKernel<CompType>::template Do<Lt, KernelSize, ExecSize, NUM_5>(src, wei, dst, lastStride, lastIdx,
                                                                               icBlock);
            break;
        }
        default: {
        }
    }
}

template <Layout Lt, IntType KernelSize, typename CompType>
void KernelParser(IntType execSize, IntType sw, const CompType *src, const CompType *wei, CompType *dst,
                  IntType lastStride, IntType lastIdx, IntType icBlock)
{
    if constexpr (std::is_same_v<CompType, float>) {
        switch (execSize) {
            case NUM_1: {
                KernelParser<Lt, KernelSize, NUM_1>(sw, src, wei, dst, lastStride, lastIdx, icBlock);
                break;
            }
            case NUM_2: {
                KernelParser<Lt, KernelSize, NUM_2>(sw, src, wei, dst, lastStride, lastIdx, icBlock);
                break;
            }
            case NUM_3: {
                KernelParser<Lt, KernelSize, NUM_3>(sw, src, wei, dst, lastStride, lastIdx, icBlock);
                break;
            }
            case NUM_4: {
                KernelParser<Lt, KernelSize, NUM_4>(sw, src, wei, dst, lastStride, lastIdx, icBlock);
                break;
            }
            case NUM_5: {
                KernelParser<Lt, KernelSize, NUM_5>(sw, src, wei, dst, lastStride, lastIdx, icBlock);
                break;
            }
            case NUM_6: {
                KernelParser<Lt, KernelSize, NUM_6>(sw, src, wei, dst, lastStride, lastIdx, icBlock);
                break;
            }
            case NUM_7: {
                KernelParser<Lt, KernelSize, NUM_7>(sw, src, wei, dst, lastStride, lastIdx, icBlock);
                break;
            }
            case NUM_8: {
                KernelParser<Lt, KernelSize, NUM_8>(sw, src, wei, dst, lastStride, lastIdx, icBlock);
                break;
            }
            default: {
            }
        }
    } else {
        switch (execSize) {
            case NUM_1: {
                KernelParser<Lt, KernelSize, NUM_1>(sw, src, wei, dst, lastStride, lastIdx, icBlock);
                break;
            }
            case NUM_2: {
                KernelParser<Lt, KernelSize, NUM_2>(sw, src, wei, dst, lastStride, lastIdx, icBlock);
                break;
            }
            case NUM_3: {
                KernelParser<Lt, KernelSize, NUM_3>(sw, src, wei, dst, lastStride, lastIdx, icBlock);
                break;
            }
            case NUM_4: {
                KernelParser<Lt, KernelSize, NUM_4>(sw, src, wei, dst, lastStride, lastIdx, icBlock);
                break;
            }
            case NUM_5: {
                KernelParser<Lt, KernelSize, NUM_5>(sw, src, wei, dst, lastStride, lastIdx, icBlock);
                break;
            }
            case NUM_6: {
                KernelParser<Lt, KernelSize, NUM_6>(sw, src, wei, dst, lastStride, lastIdx, icBlock);
                break;
            }
            case NUM_7: {
                KernelParser<Lt, KernelSize, NUM_7>(sw, src, wei, dst, lastStride, lastIdx, icBlock);
                break;
            }
            case NUM_8: {
                KernelParser<Lt, KernelSize, NUM_8>(sw, src, wei, dst, lastStride, lastIdx, icBlock);
                break;
            }
            case NUM_9: {
                KernelParser<Lt, KernelSize, NUM_9>(sw, src, wei, dst, lastStride, lastIdx, icBlock);
                break;
            }
            case NUM_10: {
                KernelParser<Lt, KernelSize, NUM_10>(sw, src, wei, dst, lastStride, lastIdx, icBlock);
                break;
            }
            case NUM_11: {
                KernelParser<Lt, KernelSize, NUM_11>(sw, src, wei, dst, lastStride, lastIdx, icBlock);
                break;
            }
            case NUM_12: {
                KernelParser<Lt, KernelSize, NUM_12>(sw, src, wei, dst, lastStride, lastIdx, icBlock);
                break;
            }
            case NUM_13: {
                KernelParser<Lt, KernelSize, NUM_13>(sw, src, wei, dst, lastStride, lastIdx, icBlock);
                break;
            }
            case NUM_14: {
                KernelParser<Lt, KernelSize, NUM_14>(sw, src, wei, dst, lastStride, lastIdx, icBlock);
                break;
            }
            case NUM_15: {
                KernelParser<Lt, KernelSize, NUM_15>(sw, src, wei, dst, lastStride, lastIdx, icBlock);
                break;
            }
            case NUM_16: {
                KernelParser<Lt, KernelSize, NUM_16>(sw, src, wei, dst, lastStride, lastIdx, icBlock);
                break;
            }
            default: {
            }
        }
    }
}

template <Layout Lt, typename CompType>
void KernelParser(IntType kernelSize, IntType execSize, IntType sw, const CompType *src, const CompType *wei,
                  CompType *dst, IntType lastStride, IntType lastIdx, IntType icBlock)
{
    switch (kernelSize) {
        case NUM_1: {
            KernelParser<Lt, NUM_1>(execSize, sw, src, wei, dst, lastStride, lastIdx, icBlock);
            break;
        }
        case NUM_2: {
            KernelParser<Lt, NUM_2>(execSize, sw, src, wei, dst, lastStride, lastIdx, icBlock);
            break;
        }
        case NUM_3: {
            KernelParser<Lt, NUM_3>(execSize, sw, src, wei, dst, lastStride, lastIdx, icBlock);
            break;
        }
        case NUM_4: {
            KernelParser<Lt, NUM_4>(execSize, sw, src, wei, dst, lastStride, lastIdx, icBlock);
            break;
        }
        case NUM_5: {
            KernelParser<Lt, NUM_5>(execSize, sw, src, wei, dst, lastStride, lastIdx, icBlock);
            break;
        }
        case NUM_6: {
            KernelParser<Lt, NUM_6>(execSize, sw, src, wei, dst, lastStride, lastIdx, icBlock);
            break;
        }
        case NUM_7: {
            KernelParser<Lt, NUM_7>(execSize, sw, src, wei, dst, lastStride, lastIdx, icBlock);
            break;
        }
        case NUM_8: {
            KernelParser<Lt, NUM_8>(execSize, sw, src, wei, dst, lastStride, lastIdx, icBlock);
            break;
        }
        case NUM_9: {
            KernelParser<Lt, NUM_9>(execSize, sw, src, wei, dst, lastStride, lastIdx, icBlock);
            break;
        }
        case NUM_10: {
            KernelParser<Lt, NUM_10>(execSize, sw, src, wei, dst, lastStride, lastIdx, icBlock);
            break;
        }
        case NUM_11: {
            KernelParser<Lt, NUM_11>(execSize, sw, src, wei, dst, lastStride, lastIdx, icBlock);
            break;
        }
        case NUM_12: {
            KernelParser<Lt, NUM_12>(execSize, sw, src, wei, dst, lastStride, lastIdx, icBlock);
            break;
        }
        case NUM_13: {
            KernelParser<Lt, NUM_13>(execSize, sw, src, wei, dst, lastStride, lastIdx, icBlock);
            break;
        }
        default: {
        }
    }
}

template <typename CompType>
void KernelParser(Layout lt, IntType kernelSize, IntType execSize, IntType sw, const CompType *src, const CompType *wei,
                  CompType *dst, IntType ow, IntType oh, IntType oc, IntType owB, IntType ohA, IntType ohB,
                  IntType ocSB, IntType ocB, IntType b, IntType residOCSBSize, IntType icBlock)
{
    if (lt == Layout::NCHW) {
        KernelParser<Layout::NCHW>(kernelSize, execSize, sw, src, wei,
                                   dst + owB + ow * (ohA + ohB + oh * (ocSB + ocB + oc * b)), ow * oh, residOCSBSize,
                                   icBlock);
    } else {
        KernelParser<Layout::NHWC>(kernelSize, execSize, sw, src, wei,
                                   dst + ocSB + ocB + oc * (owB + ow * (ohA + ohB + oh * b)), oc, residOCSBSize,
                                   icBlock);
    }
}

template <typename ComputeType, typename DataType>
void Conv2D(const ConvolutionParams &params, const DataType *src, const DataType *wei, DataType *dst,
            const DataType *bia) noexcept(false)
{
#ifdef TIMING
    double packWeiTime = NUM_0;
    double packSrcTime = NUM_0;
    double kernelTime = NUM_0;
    double fillDstTime = NUM_0;
#endif

    IntType ib = params.inputShape[NUM_0];
    IntType ic = params.inputShape[NUM_1];
    IntType ih = params.inputShape[NUM_3];
    IntType iw = params.inputShape[NUM_4];
    IntType oc = params.outputShape[NUM_1];
    IntType oh = params.outputShape[NUM_3];
    IntType ow = params.outputShape[NUM_4];
    IntType kh = params.kernelShape[NUM_3];
    IntType kw = params.kernelShape[NUM_4];
    IntType ph = params.paddingL[NUM_1];
    IntType pw = params.paddingL[NUM_2];
    IntType sh = params.stride[NUM_1];
    IntType sw = params.stride[NUM_2];
    constexpr IntType numElemsIn16Bits = (NUM_16 / sizeof(ComputeType));
    IntType kernelOCSize = ConvKernel<ComputeType>::OcSize();
    IntType kernelOWSize = ConvKernel<ComputeType>::OwSize();
    IntType kernelSize = kw;
    IntType kernelIWLength =
        (((kernelOWSize - NUM_1) * sw + kernelSize + numElemsIn16Bits - 1) / numElemsIn16Bits) * numElemsIn16Bits;
    IntType optNumThreads;
    IntType computationWorkload = std::max<IntType>(ic, oc) * ib * kh * kw * ih * iw + oh * ow;
    if (computationWorkload < CONV_THREAD_THREADS) {
        optNumThreads = 1;
    } else {
        optNumThreads = std::min<IntType>(NUM_20, Threading::GetMaxNumThreads());
    }

    IntType icBlock = std::min<IntType>(
        std::max<IntType>((L1_SIZE - kernelOCSize * kernelOWSize * sizeof(ComputeType)) /
                              ((kernelOCSize * kernelSize + kernelIWLength) * kernelSize * sizeof(ComputeType)),
                          1),
        ic);

    IntType ocBlock =
        ((optNumThreads > 1) ? 1 : 8) *
        std::min(
            std::max<IntType>((((L2_SIZE) / (icBlock * kernelOCSize * kernelSize * kernelSize * sizeof(ComputeType))) /
                               kernelOCSize) *
                                  kernelOCSize,
                              kernelOCSize),
            ((static_cast<IntType>(oc) + kernelOCSize - NUM_1) / kernelOCSize) * kernelOCSize);

    IntType ohBlock = std::min<IntType>(
        std::max<IntType>((L3_SIZE) / (((kernelOWSize + kernelIWLength) * kernelSize * icBlock) * sizeof(ComputeType)),
                          1),
        oh);

    std::vector<ComputeType, Service::AlignedAllocator<ComputeType>> weiBuf(kernelSize * kernelSize * icBlock *
                                                                            ocBlock * (ic + icBlock - NUM_1) / icBlock *
                                                                            (oc + ocBlock - NUM_1) / ocBlock);

#ifdef TIMING
    auto startPackWei = std::chrono::steady_clock::now();
#endif
    PackWei(params.kernelLayout, wei, weiBuf.data(), icBlock, ocBlock, ic, oc, kernelSize, kernelOCSize, optNumThreads);
#ifdef TIMING
    auto endPackWei = std::chrono::steady_clock::now();
    packWeiTime += (endPackWei - startPackWei).count();
#endif
    std::vector<std::pair<Shape, Shape>> partition =
        GetThreadPartition(optNumThreads, ib, oh, oc, ohBlock, ow, ocBlock, kernelOWSize, kernelOCSize);

#pragma omp parallel for num_threads(optNumThreads) schedule(static)
    for (auto &&curWork : partition) {
        std::vector<ComputeType,
                    Service::AlignedAllocator<ComputeType>>
            srcBuf(kernelIWLength * static_cast<IntType>(kh) * icBlock); // L3
        for (IntType b = curWork.first[NUM_0]; b < static_cast<IntType>(curWork.second[NUM_0]); ++b) {
            for (IntType ohbIter = curWork.first[NUM_1]; ohbIter < static_cast<IntType>(curWork.second[NUM_1]);
                 ohbIter += ohBlock) {
                IntType residOHSize = std::min(oh - ohbIter, static_cast<IntType>(curWork.second[NUM_3]));
#ifdef TIMING
                auto startFillDst = std::chrono::steady_clock::now();
#endif
                FillDst(dst, bia, params, curWork, ocBlock, residOHSize, kernelOCSize, kernelOWSize, ohbIter, b);
#ifdef TIMING
                auto endFillDst = std::chrono::steady_clock::now();
                fillDstTime += (endFillDst - startFillDst).count();
#endif
                for (IntType icB = NUM_0; icB < ic; icB += icBlock) {
                    IntType residICSize = std::min(ic - icB, icBlock);
                    for (IntType ocbIter = curWork.first[NUM_2]; ocbIter < static_cast<IntType>(curWork.second[NUM_2]);
                         ocbIter += ocBlock) {
                        IntType residOCSize = std::min(oc - ocbIter, ocBlock);
                        for (IntType ohIter = curWork.first[NUM_3]; ohIter < residOHSize; ++ohIter) {
                            for (IntType owbIter = curWork.first[NUM_4];
                                 owbIter < static_cast<IntType>(curWork.second[NUM_4]); owbIter += kernelOWSize) {
                                IntType residOWSize = std::min(ow - owbIter, kernelOWSize);
                                IntType ihB = (ohIter + ohbIter) * sh - ph;
                                IntType iwB = owbIter * sw - pw;
                                // pack src
#ifdef TIMING
                                auto startPackSrc = std::chrono::steady_clock::now();
#endif
                                if ((iwB < NUM_0) || (ihB < NUM_0)) {
                                    PackSrcBorders(params.inputLayout, src, srcBuf.data(), residICSize, ihB, iwB,
                                                   kernelSize, std::min((kernelOWSize - 1) * sw + kernelSize, iw), ih,
                                                   iw, ic,
                                                   (((residOWSize - NUM_1) * sw + kernelSize + numElemsIn16Bits - 1) /
                                                    numElemsIn16Bits) *
                                                       numElemsIn16Bits,
                                                   iwB, ihB, icB, b, pw);
                                } else {
                                    IntType kernelIWUnroll = (kernelOWSize - 1) * sw + kernelSize;
                                    IntType requiredIW = std::min(iw, (ow - 1) * sw + kw - pw);
                                    IntType limitIW = std::min(kernelIWUnroll, requiredIW - iwB);
                                    PackSrc(params.inputLayout, src, srcBuf.data(), residICSize,
                                            std::min(kernelSize, ih - ihB), kernelSize, limitIW, ih, iw, ic,
                                            (((residOWSize - NUM_1) * sw + kernelSize + numElemsIn16Bits - 1) /
                                             numElemsIn16Bits) *
                                                numElemsIn16Bits,
                                            iwB, ihB, icB, b);
                                }
#ifdef TIMING
                                auto endPackSrc = std::chrono::steady_clock::now();
                                packSrcTime += (endPackSrc - startPackSrc).count();
#endif
                                for (IntType ocSB = NUM_0; ocSB < residOCSize; ocSB += kernelOCSize) {
                                    IntType residOCSBSize = std::min(residOCSize - ocSB, kernelOCSize);
#ifdef TIMING
                                    auto startKernel = std::chrono::steady_clock::now();
#endif
                                    KernelParser(
                                        params.outputLayout, kernelSize, residOWSize, sw, srcBuf.data(),
                                        weiBuf.data() +
                                            kernelSize * kernelSize *
                                                (icB * ocBlock +
                                                 icBlock * (ocSB + ((ic + icBlock - NUM_1) / icBlock) * ocbIter)),
                                        dst, ow, oh, oc, owbIter, ohIter, ohbIter, ocSB, ocbIter, b, residOCSBSize,
                                        residICSize);
#ifdef TIMING
                                    auto endKernel = std::chrono::steady_clock::now();
                                    kernelTime += (endKernel - startKernel).count();
#endif
                                }
                            }
                        }
                    }
                }
            }
        }
#ifdef TIMING
        double totalTime = fillDstTime + packWeiTime + packSrcTime + kernelTime;
        std::cout << "fillDstTime = " << std::setw(NUM_14) << fillDstTime << ",     "
                  << (fillDstTime / totalTime * 100.0) << "%" << std::endl; // 100.0 - to output %
        std::cout << "packWeiTime = " << std::setw(NUM_14) << packWeiTime << ",     "
                  << (packWeiTime / totalTime * 100.0) << "%" << std::endl; // 100.0 - to output %
        std::cout << "packSrcTime = " << std::setw(NUM_14) << packSrcTime << ",     "
                  << (packSrcTime / totalTime * 100.0) << "%" << std::endl; // 100.0 - to output %
        std::cout << "kernelTime  = " << std::setw(NUM_14) << kernelTime << ",     " << (kernelTime / totalTime * 100.0)
                  << "%" << std::endl; // 100.0 - to output %

        std::cout << "kernelPerfScaler = " << (totalTime / kernelTime) << std::endl;
#endif
    }
}

template void Conv2D<float, float>(const ConvolutionParams &params, const float *src, const float *wei, float *dst,
                                   const float *bia) noexcept(false);

template void Conv2D<__fp16, __fp16>(const ConvolutionParams &params, const __fp16 *src, const __fp16 *wei, __fp16 *dst,
                                     const __fp16 *bia) noexcept(false);

} // namespace Detail
} // namespace KuDNN
