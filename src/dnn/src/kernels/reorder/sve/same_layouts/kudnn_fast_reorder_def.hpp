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

#ifndef KUDNN_FAST_REORDER_DEF_HPP
#define KUDNN_FAST_REORDER_DEF_HPP

#include <type_traits>
#include "vector_wrappers.hpp"
#include "kudnn_fast_reorder_kernels.hpp"
#include "kernels/reorder/kudnn_reorder_helpers.hpp"

namespace KuDNN {
namespace Detail {
namespace Reorder {

template<typename InType, typename OutType, bool AlphaMul, ARCH Arch = ARCH::VC>
struct ReorderNLoopSameLayouts final {
    static SizeType LoopInc() noexcept
    {
        return Vector::UnrollLength<InType, float, OutType>();
    }
    static void Do(SizeType loopIdx, SizeType loopRange, const InType *typedSrc,
        OutType *typedDst, SizeType dstStride, SizeType srcIdx, SizeType dstIdx, float alpha) noexcept
    {
        Service::Unused(dstStride);
        if constexpr (std::is_same_v<InType, OutType> && !AlphaMul) {
            if constexpr (Arch == ARCH::NEON) {
                if constexpr (std::is_same_v<InType, float>) {
                    auto row = vld1q_f32_x2(typedSrc + srcIdx + loopIdx);
                    vst1q_f32_x2(typedDst + dstIdx + loopIdx, row);
                } else if constexpr (std::is_same_v<InType, __fp16>) {
                    auto row = vld1q_f16_x2(typedSrc + srcIdx + loopIdx);
                    vst1q_f16_x2(typedDst + dstIdx + loopIdx, row);
                } else if constexpr (std::is_same_v<InType, __bf16>) {
                    auto row = vld1q_bf16_x2(typedSrc + srcIdx + loopIdx);
                    vst1q_bf16_x2(typedDst + dstIdx + loopIdx, row);
                }
            } else {
                svbool_t srcPG0;
                Vector::WhileLt<InType>::Do(srcPG0, loopIdx, loopRange);
                auto row = svld1(srcPG0, typedSrc + srcIdx + loopIdx);
                svst1(srcPG0, typedDst + dstIdx + loopIdx, row);
            }
        } else {
            svbool_t srcPG0;
            svbool_t srcPG1;
            svbool_t srcPG2;
            svbool_t srcPG3;
            Vector::InitPGN<InType, Vector::GetUsedRegsN<InType, float, OutType>()>
                ::Do(srcPG0, srcPG1, srcPG2, srcPG3, loopIdx, loopRange);

            typename Vector::Vec<float, Arch>::Type dst0;
            typename Vector::Vec<float, Arch>::Type dst1;
            typename Vector::Vec<float, Arch>::Type dst2;
            typename Vector::Vec<float, Arch>::Type dst3;
            Vector::LoadN<InType, float, Vector::GetUsedRegsN<InType, float, OutType>(), Arch>
                ::Do(typedSrc + srcIdx + loopIdx, srcPG0, srcPG1, srcPG2, srcPG3, dst0, dst1, dst2, dst3);

            if constexpr (AlphaMul) {
                svbool_t comPG0;
                svbool_t comPG1;
                svbool_t comPG2;
                svbool_t comPG3;
                Vector::InitPGN<float, Vector::GetUsedRegsN<float, InType, OutType>()>
                    ::Do(comPG0, comPG1, comPG2, comPG3, loopIdx, loopRange);
                typename Vector::Vec<float, Arch>::Type alphaVec;
                typename Vector::Vec<float, Arch>::Type intMax;
                Vector::DupFloat<float, Arch>::Do(alpha, alphaVec);
                Vector::DupFloat<float, Arch>::Do(REORDER_S32_TO_F32_SAT_CONSTANT, intMax);
                Vector::MulnXN<float, Vector::GetUsedRegsN<float, InType, OutType>(), Arch>
                    ::Do(comPG0, comPG1, comPG2, comPG3, dst0, dst1, dst2, dst3,
                         alpha, alpha, alpha, alpha, dst0, dst1, dst2, dst3);
                if constexpr (std::is_same_v<OutType, std::int32_t>) {
                    Vector::MinXN<float, Vector::GetUsedRegsN<float, InType, OutType>(), Arch>
                    ::Do(comPG0, comPG1, comPG2, comPG3, dst0, dst1, dst2, dst3,
                         intMax, intMax, intMax, intMax, dst0, dst1, dst2, dst3);
                }
            }

            svbool_t dstPG0;
            svbool_t dstPG1;
            svbool_t dstPG2;
            svbool_t dstPG3;
            Vector::InitPGN<OutType, Vector::GetUsedRegsN<OutType, float, InType>()>
                ::Do(dstPG0, dstPG1, dstPG2, dstPG3, loopIdx, loopRange);

            Vector::StoreN<OutType, float, Vector::GetUsedRegsN<OutType, float, InType>(), Arch>
                ::Do(dstPG0, dstPG1, dstPG2, dstPG3, dst0, dst1, dst2, dst3, typedDst + dstIdx + loopIdx);
        }
    }
};

template<typename InType, typename OutType>
void ReorderSameLayouts(const Shape &start, const Shape &end, const InType *typedSrc, OutType *typedDst,
    const Shape &reorderedStridesSrc, const Shape &reorderedStridesDst, float alpha) noexcept
{
#if USE_NEON
        SizeType indexS = start[IDX_4];
        SizeType indexE = end[IDX_4];
        SizeType totalLen = indexE - indexS;
        SizeType loopLen = ReorderNLoopSameLayouts<InType, OutType, false, ARCH::NEON>::LoopInc();

        SizeType neonLoop = indexE - (totalLen % loopLen);

        Shape startNew = start;
        Shape endNew = end;
        endNew[IDX_4] = neonLoop;

        if (alpha == 1.0f) {
            ReorderCoreLoop(startNew, endNew, typedSrc, typedDst, reorderedStridesSrc, reorderedStridesDst,
                ReorderNLoopSameLayouts<InType, OutType, false, ARCH::NEON>{}, alpha);
        } else {
            ReorderCoreLoop(startNew, endNew, typedSrc, typedDst, reorderedStridesSrc, reorderedStridesDst,
                ReorderNLoopSameLayouts<InType, OutType, true, ARCH::NEON>{}, alpha);
        }

        startNew[IDX_4] = neonLoop;
        endNew[IDX_4] = indexE;
        if (alpha == 1.0f) {
            ReorderCoreLoop(startNew, endNew, typedSrc, typedDst, reorderedStridesSrc, reorderedStridesDst,
                ReorderNLoopSameLayouts<InType, OutType, false>{}, alpha);
        } else {
            ReorderCoreLoop(startNew, endNew, typedSrc, typedDst, reorderedStridesSrc, reorderedStridesDst,
                ReorderNLoopSameLayouts<InType, OutType, true>{}, alpha);
        }
#else
    if (alpha == 1.0f) {
        ReorderCoreLoop(start, end, typedSrc, typedDst, reorderedStridesSrc, reorderedStridesDst,
            ReorderNLoopSameLayouts<InType, OutType, false>{}, alpha);
    } else {
        ReorderCoreLoop(start, end, typedSrc, typedDst, reorderedStridesSrc, reorderedStridesDst,
            ReorderNLoopSameLayouts<InType, OutType, true>{}, alpha);
    }
#endif
}

} // Reorder
} // Detail
} // KuDNN

#endif // KUDNN_FAST_REORDER_DEF_HPP
