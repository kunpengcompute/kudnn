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

#ifndef KUDNN_SHUFFLE_DEF_HPP
#define KUDNN_SHUFFLE_DEF_HPP

#include "kudnn.hpp"

#include "vector_wrappers.hpp"
#include "kudnn_shuffle_kernels.hpp"

namespace KuDNN {
namespace Detail {
namespace Shuffle {

template <typename InType, ARCH Arch = ARCH::VC>
struct ShuffleNLoop final {
    static SizeType LoopInc() noexcept
    {
        if constexpr (Arch == ARCH::VC) {
            return Vector::UnrollLength<InType, InType, InType>();
        } else {
            return NUM_2 * Vector::NeonLanes<InType, InType, InType>();
        }
    }
    static void Do(SizeType loopIdx, SizeType loopRange, const InType *typedSrc,
        InType *typedDst, SizeType dstStride, SizeType srcStride, SizeType srcIdx, SizeType dstIdx,
        const SizeType axis, const SizeType *revTransposed,
        SizeType strideRevTransposed = 0) noexcept
    {
        if ((axis == IDX_4) || (dstStride != IDX_1)) {
            SizeType srcLoopIdx;
            for (SizeType i = 0; i < std::min(LoopInc(), loopRange - loopIdx); ++i) {
                srcLoopIdx = (axis == IDX_4) ? (revTransposed[loopIdx + i] - strideRevTransposed) : loopIdx + i;
                typedDst[dstIdx + (loopIdx + i) * dstStride] = typedSrc[srcIdx + srcLoopIdx * srcStride];
            }
        } else {
            svbool_t ldPG;
            Vector::InitPGN<InType, 1>::Do(
                ldPG, ldPG, ldPG, ldPG, loopIdx, loopRange);
            typename Vector::Vec<InType, Arch>::Type lhs;
            Vector::LoadN<InType, InType, 1, Arch>::Do(typedSrc + srcIdx + loopIdx, ldPG, ldPG,
                ldPG, ldPG, lhs, lhs, lhs, lhs);

            svbool_t stPG;
            Vector::InitPGN<InType, 1>::Do(
                stPG, stPG, stPG, stPG, loopIdx, loopRange);
            Vector::StoreN<InType, InType, 1, Arch>::Do(
                stPG, stPG, stPG, stPG, lhs, lhs, lhs, lhs,
                typedDst + dstIdx + loopIdx * dstStride);
        }
    }
};

template <typename InType>
inline void EvaluateShuffleIDX0(const Shape &batchSplitDims, const InType *typedSrc,
    InType *typedDst, const Shape &stridesSrc, const SizeType axis,
    const SizeType *revTransposed, SizeType strideRevTransposed) noexcept
{
#ifdef KUDNNL_920G
    SizeType neonLoop = batchSplitDims[IDX_4] - (batchSplitDims[IDX_4] % ShuffleNLoop<InType, ARCH::NEON>::LoopInc());
    SizeType tailLoop = batchSplitDims[IDX_4] % ShuffleNLoop<InType, ARCH::NEON>::LoopInc(); // Use VC
#endif
    for (SizeType a = 0; a < batchSplitDims[IDX_0]; a++) {
        SizeType srcIdx0 = (revTransposed[a] - strideRevTransposed) * stridesSrc[IDX_0];
        SizeType dstIdx0 = a * stridesSrc[IDX_0];
        for (SizeType b = 0; b < batchSplitDims[IDX_1]; b++) {
            SizeType idx1 = b * stridesSrc[IDX_1];
            for (SizeType c = 0; c < batchSplitDims[IDX_2]; c++) {
                SizeType idx2 = c * stridesSrc[IDX_2];
                for (SizeType d = 0; d < batchSplitDims[IDX_3]; d++) {
                    SizeType srcIdx = srcIdx0 + idx1 + idx2 + d * stridesSrc[IDX_3];
                    SizeType dstIdx = dstIdx0 + idx1 + idx2 + d * stridesSrc[IDX_3];
#ifdef KUDNNL_920G
                    Service::UnrollFor::Do<1>(neonLoop, ShuffleNLoop<InType, ARCH::NEON>{}, typedSrc, typedDst,
                        stridesSrc[IDX_4], stridesSrc[IDX_4], srcIdx, dstIdx, axis, revTransposed);
                    Service::UnrollFor::Do<1>(tailLoop, ShuffleNLoop<InType, ARCH::VC>{}, typedSrc + neonLoop,
                        typedDst + neonLoop, stridesSrc[IDX_4], stridesSrc[IDX_4], srcIdx, dstIdx, axis, revTransposed);
#else
                    Service::UnrollFor::Do<1>(batchSplitDims[IDX_4], ShuffleNLoop<InType, ARCH::VC>{}, typedSrc,
                        typedDst, stridesSrc[IDX_4], stridesSrc[IDX_4], srcIdx, dstIdx, axis, revTransposed);
#endif
                }
            }
        }
    }
}

template <typename InType>
inline void EvaluateShuffleIDX1(const Shape &batchSplitDims, const InType *typedSrc,
    InType *typedDst, const Shape &stridesSrc, const SizeType axis,
    const SizeType *revTransposed, SizeType strideRevTransposed) noexcept
{
#ifdef KUDNNL_920G
    SizeType neonLoop = batchSplitDims[IDX_4] - (batchSplitDims[IDX_4] % ShuffleNLoop<InType, ARCH::NEON>::LoopInc());
    SizeType tailLoop = batchSplitDims[IDX_4] % ShuffleNLoop<InType, ARCH::NEON>::LoopInc(); // Use VC
#endif
    for (SizeType a = 0; a < batchSplitDims[IDX_0]; a++) {
        SizeType idx0 = a * stridesSrc[IDX_0];
        for (SizeType b = 0; b < batchSplitDims[IDX_1]; b++) {
            SizeType srcIdx1 = (revTransposed[b] - strideRevTransposed) * stridesSrc[IDX_1];
            SizeType dstIdx1 = b * stridesSrc[IDX_1];
            for (SizeType c = 0; c < batchSplitDims[IDX_2]; c++) {
                SizeType idx2 = c * stridesSrc[IDX_2];
                for (SizeType d = 0; d < batchSplitDims[IDX_3]; d++) {
                    SizeType srcIdx = idx0 + srcIdx1 + idx2 + d * stridesSrc[IDX_3];
                    SizeType dstIdx = idx0 + dstIdx1 + idx2 + d * stridesSrc[IDX_3];
#ifdef KUDNNL_920G
                    Service::UnrollFor::Do<1>(neonLoop, ShuffleNLoop<InType, ARCH::NEON>{}, typedSrc, typedDst,
                        stridesSrc[IDX_4], stridesSrc[IDX_4], srcIdx, dstIdx, axis, revTransposed);
                    Service::UnrollFor::Do<1>(tailLoop, ShuffleNLoop<InType, ARCH::VC>{}, typedSrc + neonLoop,
                        typedDst + neonLoop, stridesSrc[IDX_4], stridesSrc[IDX_4], srcIdx, dstIdx, axis, revTransposed);
#else
                    Service::UnrollFor::Do<1>(batchSplitDims[IDX_4], ShuffleNLoop<InType, ARCH::VC>{}, typedSrc,
                        typedDst, stridesSrc[IDX_4], stridesSrc[IDX_4], srcIdx, dstIdx, axis, revTransposed);
#endif
                }
            }
        }
    }
}

template <typename InType>
inline void EvaluateShuffleIDX2(const Shape &batchSplitDims, const InType *typedSrc,
    InType *typedDst, const Shape &stridesSrc, const SizeType axis,
    const SizeType *revTransposed, SizeType strideRevTransposed) noexcept
{
#ifdef KUDNNL_920G
    SizeType neonLoop = batchSplitDims[IDX_4] - (batchSplitDims[IDX_4] % ShuffleNLoop<InType, ARCH::NEON>::LoopInc());
    SizeType tailLoop = batchSplitDims[IDX_4] % ShuffleNLoop<InType, ARCH::NEON>::LoopInc(); // Use VC
#endif
    for (SizeType a = 0; a < batchSplitDims[IDX_0]; a++) {
        SizeType idx0 = a * stridesSrc[IDX_0];
        for (SizeType b = 0; b < batchSplitDims[IDX_1]; b++) {
            SizeType idx1 = b * stridesSrc[IDX_1];
            for (SizeType c = 0; c < batchSplitDims[IDX_2]; c++) {
                SizeType srcIdx2 = (revTransposed[c] - strideRevTransposed) * stridesSrc[IDX_2];
                SizeType dstIdx2 = c * stridesSrc[IDX_2];
                for (SizeType d = 0; d < batchSplitDims[IDX_3]; d++) {
                    SizeType srcIdx = idx0 + idx1 + srcIdx2 + d * stridesSrc[IDX_3];
                    SizeType dstIdx = idx0 + idx1 + dstIdx2 + d * stridesSrc[IDX_3];
#ifdef KUDNNL_920G
                    Service::UnrollFor::Do<1>(neonLoop, ShuffleNLoop<InType, ARCH::NEON>{}, typedSrc, typedDst,
                        stridesSrc[IDX_4], stridesSrc[IDX_4], srcIdx, dstIdx, axis, revTransposed);
                    Service::UnrollFor::Do<1>(tailLoop, ShuffleNLoop<InType, ARCH::VC>{}, typedSrc + neonLoop,
                        typedDst + neonLoop, stridesSrc[IDX_4], stridesSrc[IDX_4], srcIdx, dstIdx, axis, revTransposed);
#else
                    Service::UnrollFor::Do<1>(batchSplitDims[IDX_4], ShuffleNLoop<InType, ARCH::VC>{}, typedSrc,
                        typedDst, stridesSrc[IDX_4], stridesSrc[IDX_4], srcIdx, dstIdx, axis, revTransposed);
#endif
                }
            }
        }
    }
}

template <typename InType>
inline void EvaluateShuffleIDX3(const Shape &batchSplitDims, const InType *typedSrc,
    InType *typedDst, const Shape &stridesSrc, const SizeType axis,
    const SizeType *revTransposed, SizeType strideRevTransposed) noexcept
{
#ifdef KUDNNL_920G
    SizeType neonLoop = batchSplitDims[IDX_4] - (batchSplitDims[IDX_4] % ShuffleNLoop<InType, ARCH::NEON>::LoopInc());
    SizeType tailLoop = batchSplitDims[IDX_4] % ShuffleNLoop<InType, ARCH::NEON>::LoopInc(); // Use VC
#endif
    for (SizeType a = 0; a < batchSplitDims[IDX_0]; a++) {
        SizeType idx0 = a * stridesSrc[IDX_0];
        for (SizeType b = 0; b < batchSplitDims[IDX_1]; b++) {
            SizeType idx1 = b * stridesSrc[IDX_1];
            for (SizeType c = 0; c < batchSplitDims[IDX_2]; c++) {
                SizeType idx2 = c * stridesSrc[IDX_2];
                for (SizeType d = 0; d < batchSplitDims[IDX_3]; d++) {
                    SizeType srcIdx =
                        idx0 + idx1 + idx2 + (revTransposed[d] - strideRevTransposed) * stridesSrc[IDX_3];
                    SizeType dstIdx = idx0 + idx1 + idx2 + d * stridesSrc[IDX_3];
#ifdef KUDNNL_920G
                    Service::UnrollFor::Do<1>(neonLoop, ShuffleNLoop<InType, ARCH::NEON>{}, typedSrc, typedDst,
                        stridesSrc[IDX_4], stridesSrc[IDX_4], srcIdx, dstIdx, axis, revTransposed);
                    Service::UnrollFor::Do<1>(tailLoop, ShuffleNLoop<InType, ARCH::VC>{}, typedSrc + neonLoop,
                        typedDst + neonLoop, stridesSrc[IDX_4], stridesSrc[IDX_4], srcIdx, dstIdx, axis, revTransposed);
#else
                    Service::UnrollFor::Do<1>(batchSplitDims[IDX_4], ShuffleNLoop<InType, ARCH::VC>{}, typedSrc,
                        typedDst, stridesSrc[IDX_4], stridesSrc[IDX_4], srcIdx, dstIdx, axis, revTransposed);
#endif
                }
            }
        }
    }
}

template <typename InType>
inline void EvaluateShuffleIDX4(const Shape &batchSplitDims, const InType *typedSrc,
    InType *typedDst, const Shape &stridesSrc, const SizeType axis,
    const SizeType *revTransposed, SizeType strideRevTransposed) noexcept
{
    for (SizeType a = 0; a < batchSplitDims[IDX_0]; a++) {
        SizeType idx0 = a * stridesSrc[IDX_0];
        for (SizeType b = 0; b < batchSplitDims[IDX_1]; b++) {
            SizeType idx1 = b * stridesSrc[IDX_1];
            for (SizeType c = 0; c < batchSplitDims[IDX_2]; c++) {
                SizeType idx2 = c * stridesSrc[IDX_2];
                for (SizeType d = 0; d < batchSplitDims[IDX_3]; d++) {
                    SizeType srcIdx = idx0 + idx1 + idx2 + d * stridesSrc[IDX_3];
                    Service::UnrollFor::Do<1>(batchSplitDims[IDX_4], ShuffleNLoop<InType, ARCH::VC>{},
                        typedSrc, typedDst, stridesSrc[IDX_4], stridesSrc[IDX_4],
                        srcIdx, srcIdx, axis, revTransposed, strideRevTransposed);
                }
            }
        }
    }
}

template <typename InType>
void EvaluateShuffle(const Shape &batchSplitDims, const InType *typedSrc,
    InType *typedDst, const Shape &stridesSrc, const SizeType axis,
    const SizeType *revTransposed, SizeType strideRevTransposed) noexcept
{
    if (axis == IDX_0) {
        EvaluateShuffleIDX0(batchSplitDims, typedSrc, typedDst, stridesSrc, axis, revTransposed, strideRevTransposed);
    } else if (axis == IDX_1) {
        EvaluateShuffleIDX1(batchSplitDims, typedSrc, typedDst, stridesSrc, axis, revTransposed, strideRevTransposed);
    } else if (axis == IDX_2) {
        EvaluateShuffleIDX2(batchSplitDims, typedSrc, typedDst, stridesSrc, axis, revTransposed, strideRevTransposed);
    } else if (axis == IDX_3) {
        EvaluateShuffleIDX3(batchSplitDims, typedSrc, typedDst, stridesSrc, axis, revTransposed, strideRevTransposed);
    } else {
        EvaluateShuffleIDX4(batchSplitDims, typedSrc, typedDst, stridesSrc, axis, revTransposed, strideRevTransposed);
    }
}

} // Shuffle
} // Detail
} // KuDNN

#endif // KUDNN_SHUFFLE_DEF_HPP
