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

#ifndef KUDNN_SUM_DEF_HPP
#define KUDNN_SUM_DEF_HPP

#include <arm_sve.h>

#include "kudnn.hpp"

#include "vector_wrappers.hpp"
#include "kernels/sum/kudnn_sum_kernels.hpp"

namespace KuDNN {
namespace Detail {
namespace Sum {
namespace Helpers {

template <typename CompType, typename InType, typename OutType, ARCH Arch = ARCH::VC>
struct SumInternalKernel final {
    static SizeType LoopInc() noexcept
    {
        return 1;
    }
    static inline void Do(SizeType loopIdx, SizeType loopRange, const void **src, const float *scl,
        svbool_t ldpg0, svbool_t ldpg1, svbool_t ldpg2, svbool_t ldpg3,
        svbool_t oppg0, svbool_t oppg1, svbool_t oppg2, svbool_t oppg3,
        typename Vector::Vec<CompType, Arch>::Type& res0, typename Vector::Vec<CompType, Arch>::Type& res1,
        typename Vector::Vec<CompType, Arch>::Type& res2, typename Vector::Vec<CompType, Arch>::Type& res3,
        SizeType externalIdx) noexcept
    {
        Service::Unused(loopRange);
        using VecType = typename Vector::Vec<CompType, Arch>::Type;
        VecType data0;
        VecType data1;
        VecType data2;
        VecType data3;
        VecType scl0;
        VecType scl1;
        VecType scl2;
        VecType scl3;
        Vector::LoadN<InType, CompType, Vector::GetUsedRegsN<InType, CompType, OutType>(), Arch>::Do(
            static_cast<const InType *>(src[loopIdx]) + externalIdx, ldpg0, ldpg1, ldpg2, ldpg3,
            data0, data1, data2, data3);
        Vector::DupFloatN<InType, CompType, Vector::GetUsedRegsN<InType, CompType, OutType>(), Arch>::Do(
            scl[loopIdx], scl0, scl1, scl2, scl3);
        Vector::MlaXN<CompType, Vector::GetUsedRegsN<CompType, InType, OutType>(), Arch>::Do(
            oppg0, oppg1, oppg2, oppg3, res0, res1, res2, res3, data0,
            data1, data2, data3, scl0, scl1, scl2, scl3, res0, res1, res2, res3);
    }
};

} // Helpers

template <typename CompType, typename InType, typename OutType, SizeType N, ARCH Arch = ARCH::VC>
struct SumNLoop final {
    static SizeType LoopInc() noexcept
    {
        return Vector::UnrollLength<InType, CompType, OutType>();
    }
    static void Do(SizeType loopIdx, SizeType loopRange, const void **src,
        OutType *typedDst, const float *scl) noexcept
    {
        using VecType = typename Vector::Vec<CompType, Arch>::Type;
        VecType res0;
        VecType res1;
        VecType res2;
        VecType res3;
        Vector::DupFloatN<InType, CompType, Vector::GetUsedRegsN<InType, CompType, OutType>(), Arch>::Do(
            0.0, res0, res1, res2, res3);
        svbool_t ldpg0;
        svbool_t ldpg1;
        svbool_t ldpg2;
        svbool_t ldpg3;
        Vector::InitPGN<InType, Vector::GetUsedRegsN<InType, CompType, OutType>()>::Do(
            ldpg0, ldpg1, ldpg2, ldpg3, loopIdx, loopRange);
        svbool_t oppg0;
        svbool_t oppg1;
        svbool_t oppg2;
        svbool_t oppg3;
        Vector::InitPGN<CompType, Vector::GetUsedRegsN<CompType, InType, OutType>()>::Do(
            oppg0, oppg1, oppg2, oppg3, loopIdx, loopRange);
        Service::UnrollFor::Do<N>(N, Helpers::SumInternalKernel<CompType, InType, OutType, Arch>{}, src, scl,
            ldpg0, ldpg1, ldpg2, ldpg3, oppg0, oppg1, oppg2, oppg3, res0, res1, res2, res3, loopIdx);
        svbool_t stpg0;
        svbool_t stpg1;
        svbool_t stpg2;
        svbool_t stpg3;
        Vector::InitPGN<OutType, Vector::GetUsedRegsN<OutType, InType, CompType>()>::Do(
            stpg0, stpg1, stpg2, stpg3, loopIdx, loopRange);
        Vector::StoreN<OutType, CompType, Vector::GetUsedRegsN<OutType, InType, CompType>(), Arch>::Do(
            stpg0, stpg1, stpg2, stpg3, res0, res1, res2, res3, typedDst + loopIdx);
    }
};

template <typename CompType, typename InType, typename OutType, SizeType N>
void EvaluateSum(const Shape &batchSplitDims, const void **src,
    OutType *typedDst, const float *scl) noexcept
{
    SizeType size = Service::ArrayProduct(batchSplitDims, batchSplitDims.GetNumDims(), 0);
#if USE_NEON
    SizeType neonLoop = size - (size % SumNLoop<CompType, InType, OutType, N, ARCH::NEON>::LoopInc());

    Service::UnrollFor::Do<1>(0, neonLoop, SumNLoop<CompType, InType, OutType, N, ARCH::NEON>{}, src, typedDst, scl);
    Service::UnrollFor::Do<1>(neonLoop, size, SumNLoop<CompType, InType, OutType, N>{}, src, typedDst, scl);
#else
    Service::UnrollFor::Do<1>(size, SumNLoop<CompType, InType, OutType, N>{}, src, typedDst, scl);
#endif
}

template <typename CompType, typename InType, typename OutType>
void EvaluateSum(const Shape &batchSplitDims, const void **src,
    OutType *typedDst, const float *scl, const SizeType nSrc) noexcept
{
    using VecType = typename Vector::Vec<CompType>::Type;
    SizeType loopInc = Vector::UnrollLength<InType, CompType, OutType>();
    SizeType size = Service::ArrayProduct(batchSplitDims, batchSplitDims.GetNumDims(), 0);
    for (SizeType i = 0; i < size; i += loopInc) {
        VecType res0;
        VecType res1;
        VecType res2;
        VecType res3;
        Vector::DupFloatN<InType, CompType, Vector::GetUsedRegsN<InType, CompType, OutType>()>::Do(
            0.0, res0, res1, res2, res3);
        svbool_t ldpg0;
        svbool_t ldpg1;
        svbool_t ldpg2;
        svbool_t ldpg3;
        Vector::InitPGN<InType, Vector::GetUsedRegsN<InType, CompType, OutType>()>::Do(
            ldpg0, ldpg1, ldpg2, ldpg3, i, size);
        svbool_t oppg0;
        svbool_t oppg1;
        svbool_t oppg2;
        svbool_t oppg3;
        Vector::InitPGN<CompType, Vector::GetUsedRegsN<CompType, InType, OutType>()>::Do(
            oppg0, oppg1, oppg2, oppg3, i, size);
        for (SizeType j = 0; j < nSrc; ++j) {
            VecType data0;
            VecType data1;
            VecType data2;
            VecType data3;
            Vector::LoadN<InType, CompType, Vector::GetUsedRegsN<InType, CompType, OutType>()>::Do(
                static_cast<const InType *>(src[j]) + i, ldpg0, ldpg1, ldpg2, ldpg3, data0, data1, data2, data3);
            VecType scl0;
            VecType scl1;
            VecType scl2;
            VecType scl3;
            Vector::DupFloatN<InType, CompType, Vector::GetUsedRegsN<InType, CompType, OutType>()>::Do(
                scl[j], scl0, scl1, scl2, scl3);
            Vector::MlaXN<CompType, Vector::GetUsedRegsN<CompType, InType, OutType>()>::Do(
                oppg0, oppg1, oppg2, oppg3, res0, res1, res2, res3, data0,
                data1, data2, data3, scl0, scl1, scl2, scl3, res0, res1, res2, res3);
        }
        svbool_t stpg0;
        svbool_t stpg1;
        svbool_t stpg2;
        svbool_t stpg3;
        Vector::InitPGN<OutType, Vector::GetUsedRegsN<OutType, InType, CompType>()>::Do(
            stpg0, stpg1, stpg2, stpg3, i, size);
        Vector::StoreN<OutType, CompType, Vector::GetUsedRegsN<OutType, InType, CompType>()>::Do(
            stpg0, stpg1, stpg2, stpg3, res0, res1, res2, res3, typedDst + i);
    }
}

} // Sum
} // Detail
} // KuDNN

#endif // KUDNN_SUM_DEF_HPP
