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

#ifndef KUDNN_BINARY_DEF_LOOP_HPP
#define KUDNN_BINARY_DEF_LOOP_HPP

#include <arm_sve.h>

#include "kudnn.hpp"

#include "vector_wrappers.hpp"
#include "kernels/binary/kudnn_binary_kernels.hpp"

namespace KuDNN {
namespace Detail {
namespace Binary {

template <typename CompType, typename Src0Type, typename Src1Type, typename OutType,
    typename Operation, typename LLHS, typename LRHS, ARCH Arch = ARCH::VC>
struct BinaryInnerLoop final {
    static SizeType LoopInc() noexcept
    {
        if constexpr (Arch == ARCH::VC) {
            return Vector::UnrollLength<Src0Type, Src1Type, CompType, OutType>();
        } else {
            return NUM_2 * Vector::NeonLanes<Src0Type, Src1Type, OutType>();
        }
    }

    static void PerformLHS(svbool_t &opPG0, svbool_t &opPG1, svbool_t &opPG2, svbool_t &opPG3, const float scale0,
        typename Vector::Vec<CompType, Arch>::Type &lhs0, typename Vector::Vec<CompType, Arch>::Type &lhs1,
        typename Vector::Vec<CompType, Arch>::Type &lhs2, typename Vector::Vec<CompType, Arch>::Type &lhs3,
        typename Vector::Vec<CompType, Arch>::Type &zero0, typename Vector::Vec<CompType, Arch>::Type &zero1,
        typename Vector::Vec<CompType, Arch>::Type &zero2, typename Vector::Vec<CompType, Arch>::Type &zero3) noexcept
    {
        Vector::DupFloatN<CompType, CompType,
            Vector::GetUsedRegsN<CompType, Src0Type, Src1Type, OutType>(), Arch>::Do(
            0.0f, zero0, zero1, zero2, zero3);

        typename Vector::MaskVec<CompType, Arch>::Type pNotZeroL0;
        typename Vector::MaskVec<CompType, Arch>::Type pNotZeroL1;
        typename Vector::MaskVec<CompType, Arch>::Type pNotZeroL2;
        typename Vector::MaskVec<CompType, Arch>::Type pNotZeroL3;
        Vector::NeN<CompType, Vector::GetUsedRegsN<CompType, Src0Type, Src1Type, OutType>(), Arch>::Do(
            opPG0, opPG1, opPG2, opPG3, lhs0, lhs1, lhs2, lhs3, zero0, zero1, zero2, zero3,
            pNotZeroL0, pNotZeroL1, pNotZeroL2, pNotZeroL3);
        typename Vector::Vec<CompType, Arch>::Type lscale0;
        typename Vector::Vec<CompType, Arch>::Type lscale1;
        typename Vector::Vec<CompType, Arch>::Type lscale2;
        typename Vector::Vec<CompType, Arch>::Type lscale3;
        Vector::DupFloatN<Src0Type, CompType, Vector::GetUsedRegsN<Src0Type, Src1Type, CompType, OutType>(), Arch>::Do(
            scale0, lscale0, lscale1, lscale2, lscale3);
        Vector::MulMaskXN<CompType, Vector::GetUsedRegsN<CompType, Src0Type, Src1Type, OutType>(), Arch>::Do(
            pNotZeroL0, pNotZeroL1, pNotZeroL2, pNotZeroL3, lhs0, lhs1, lhs2, lhs3,
            lscale0, lscale1, lscale2, lscale3, lhs0, lhs1, lhs2, lhs3);
    }

    static void PerformRHS(svbool_t &opPG0, svbool_t &opPG1, svbool_t &opPG2, svbool_t &opPG3, const float scale1,
        typename Vector::Vec<CompType, Arch>::Type &rhs0, typename Vector::Vec<CompType, Arch>::Type &rhs1,
        typename Vector::Vec<CompType, Arch>::Type &rhs2, typename Vector::Vec<CompType, Arch>::Type &rhs3,
        typename Vector::Vec<CompType, Arch>::Type &zero0, typename Vector::Vec<CompType, Arch>::Type &zero1,
        typename Vector::Vec<CompType, Arch>::Type &zero2, typename Vector::Vec<CompType, Arch>::Type &zero3) noexcept
    {
        Vector::DupFloatN<CompType, CompType,
            Vector::GetUsedRegsN<CompType, Src0Type, Src1Type, OutType>(), Arch>::Do(
            0.0f, zero0, zero1, zero2, zero3);

        typename Vector::MaskVec<CompType, Arch>::Type pNotZeroR0;
        typename Vector::MaskVec<CompType, Arch>::Type pNotZeroR1;
        typename Vector::MaskVec<CompType, Arch>::Type pNotZeroR2;
        typename Vector::MaskVec<CompType, Arch>::Type pNotZeroR3;
        Vector::NeN<CompType, Vector::GetUsedRegsN<CompType, Src0Type, Src1Type, OutType>(), Arch>::Do(
            opPG0, opPG1, opPG2, opPG3, rhs0, rhs1, rhs2, rhs3, zero0, zero1, zero2, zero3,
            pNotZeroR0, pNotZeroR1, pNotZeroR2, pNotZeroR3);
        typename Vector::Vec<CompType, Arch>::Type rscale0;
        typename Vector::Vec<CompType, Arch>::Type rscale1;
        typename Vector::Vec<CompType, Arch>::Type rscale2;
        typename Vector::Vec<CompType, Arch>::Type rscale3;
        Vector::DupFloatN<Src1Type, CompType, Vector::GetUsedRegsN<Src1Type, Src0Type, CompType, OutType>(), Arch>::Do(
            scale1, rscale0, rscale1, rscale2, rscale3);
        Vector::MulMaskXN<CompType, Vector::GetUsedRegsN<CompType, Src0Type, Src1Type, OutType>(), Arch>::Do(
            pNotZeroR0, pNotZeroR1, pNotZeroR2, pNotZeroR3, rhs0, rhs1, rhs2, rhs3,
            rscale0, rscale1, rscale2, rscale3, rhs0, rhs1, rhs2, rhs3);
    }

    static void Do(SizeType loopIdx, SizeType loopRange, const Src0Type *lhs, const Src1Type *rhs, OutType *dst,
        const float scale0, const float scale1,
        SizeType higherStrideLhs, SizeType higherStrideRhs, SizeType higherStrideDst,
        SizeType lowerStrideLhs, SizeType lowerStrideRhs, SizeType lowerStrideDst) noexcept
    {
        typename Vector::Vec<CompType, Arch>::Type lhs0;
        typename Vector::Vec<CompType, Arch>::Type lhs1;
        typename Vector::Vec<CompType, Arch>::Type lhs2;
        typename Vector::Vec<CompType, Arch>::Type lhs3;
        // load LHS and lhs scale
        {
            svbool_t ldPG0;
            svbool_t ldPG1;
            svbool_t ldPG2;
            svbool_t ldPG3;
            Vector::InitPGN<Src0Type, Vector::GetUsedRegsN<Src0Type, Src1Type, CompType, OutType>()>::Do(
                ldPG0, ldPG1, ldPG2, ldPG3, loopIdx, loopRange);
            LLHS::Do(lhs, higherStrideLhs, loopIdx * lowerStrideLhs, ldPG0, ldPG1, ldPG2, ldPG3,
                lhs0, lhs1, lhs2, lhs3);
        }
        typename Vector::Vec<CompType, Arch>::Type rhs0;
        typename Vector::Vec<CompType, Arch>::Type rhs1;
        typename Vector::Vec<CompType, Arch>::Type rhs2;
        typename Vector::Vec<CompType, Arch>::Type rhs3;
        // load RHS and rhs scale
        {
            svbool_t ldPG0, ldPG1, ldPG2, ldPG3;
            Vector::InitPGN<Src1Type, Vector::GetUsedRegsN<Src1Type, Src0Type, CompType, OutType>()>::Do(
                ldPG0, ldPG1, ldPG2, ldPG3, loopIdx, loopRange);
            LRHS::Do(rhs, higherStrideRhs, loopIdx * lowerStrideRhs, ldPG0, ldPG1, ldPG2, ldPG3,
                rhs0, rhs1, rhs2, rhs3);
        }
        svbool_t opPG0;
        svbool_t opPG1;
        svbool_t opPG2;
        svbool_t opPG3;
        Vector::InitPGN<CompType, Vector::GetUsedRegsN<CompType, Src0Type, Src1Type, OutType>()>::Do(
            opPG0, opPG1, opPG2, opPG3, loopIdx, loopRange);

        typename Vector::Vec<CompType, Arch>::Type zero0;
        typename Vector::Vec<CompType, Arch>::Type zero1;
        typename Vector::Vec<CompType, Arch>::Type zero2;
        typename Vector::Vec<CompType, Arch>::Type zero3;
        // perform scaling of LHS
        if (scale0 != 1.0f) {
            PerformLHS(opPG0, opPG1, opPG2, opPG3, scale0, lhs0, lhs1, lhs2, lhs3, zero0, zero1, zero2, zero3);
        }
        // perform scaling of RHS
        if (scale1 != 1.0f) {
            PerformRHS(opPG0, opPG1, opPG2, opPG3, scale1, rhs0, rhs1, rhs2, rhs3, zero0, zero1, zero2, zero3);
        }

        // perform binary operation over scaled LHS and RHS
        Operation::Do(opPG0, opPG1, opPG2, opPG3, lhs0, lhs1, lhs2, lhs3, rhs0, rhs1, rhs2, rhs3,
            lhs0, lhs1, lhs2, lhs3);
        // store computed result
        svbool_t stPG0;
        svbool_t stPG1;
        svbool_t stPG2;
        svbool_t stPG3;
        Vector::InitPGN<OutType, Vector::GetUsedRegsN<OutType, Src0Type, Src1Type, CompType>()>::Do(
            stPG0, stPG1, stPG2, stPG3, loopIdx, loopRange);
        Vector::StoreN<OutType, CompType, Vector::GetUsedRegsN<OutType, Src0Type, Src1Type, CompType>(), Arch>::Do(
            stPG0, stPG1, stPG2, stPG3, lhs0, lhs1, lhs2, lhs3, dst + higherStrideDst + loopIdx * lowerStrideDst);
    }
};

template <typename CompType, ARCH Arch = ARCH::VC, typename Src0Type, typename Src1Type, typename OutType,
    typename Operation, typename LLHS, typename LRHS>
inline void Evaluate(const Src0Type *lhs, const Src1Type *rhs, OutType *dst,
    const float scale0, const float scale1, const Shape &dims, const Shape &stridesSrc0,
    const Shape &stridesSrc1, const Shape &stridesDst, Operation &&op, LLHS &&llhs, LRHS &&lrhs,
    const SizeType loopLength, const SizeType offset) noexcept
{
    Service::Unused(op, llhs, lrhs);
    for (SizeType i0 = 0; i0 < dims[IDX_0]; ++i0) {
        for (SizeType i1 = 0; i1 < dims[IDX_1]; ++i1) {
            for (SizeType i2 = 0; i2 < dims[IDX_2]; ++i2) {
                for (SizeType i3 = 0; i3 < dims[IDX_3]; ++i3) {
                    // To be evaluated
                    Service::UnrollFor::Do<1>(loopLength,
                                            BinaryInnerLoop<CompType, Src0Type, Src1Type, OutType, Operation,
                                            LLHS, LRHS, Arch>{},
                                            lhs, rhs, dst, scale0, scale1,
                                            i0 * stridesSrc0[IDX_0] + i1 * stridesSrc0[IDX_1] +
                                                i2 * stridesSrc0[IDX_2] + i3 * stridesSrc0[IDX_3]
                                                + offset * stridesSrc0[IDX_4],
                                            i0 * stridesSrc1[IDX_0] + i1 * stridesSrc1[IDX_1] +
                                                i2 * stridesSrc1[IDX_2] + i3 * stridesSrc1[IDX_3]
                                                + offset * stridesSrc1[IDX_4],
                                            i0 * stridesDst[IDX_0] + i1 * stridesDst[IDX_1] +
                                                i2 * stridesDst[IDX_2] + i3 * stridesDst[IDX_3]
                                                + offset * stridesDst[IDX_4],
                                            stridesSrc0[IDX_4], stridesSrc1[IDX_4], stridesDst[IDX_4]);
                }
            }
        }
    }
}

template <typename CompType, typename Src0Type, typename Src1Type, typename OutType>
inline void EvaluateSC(const Src0Type *lhs, const Src1Type *rhs, OutType *dst,
    const float scale0, const float scale1, const Shape &dims, const Shape &stridesSrc0,
    const Shape &stridesSrc1, const Shape &stridesDst, CompType (*op)(CompType, CompType)) noexcept
{
    for (SizeType i0 = 0; i0 < dims[IDX_0]; ++i0) {
        for (SizeType i1 = 0; i1 < dims[IDX_1]; ++i1) {
            for (SizeType i2 = 0; i2 < dims[IDX_2]; ++i2) {
                for (SizeType i3 = 0; i3 < dims[IDX_3]; ++i3) {
                    for (SizeType i4 = 0; i4 < dims[IDX_4]; ++i4) {
                        dst[ i0 * stridesDst[IDX_0]  + i1 * stridesDst[IDX_1]  + i2 * stridesDst[IDX_2]  +
                             i3 * stridesDst[IDX_3]  + i4 * stridesDst[IDX_4] ] =
                        Vector::Convert<CompType, OutType>::Do(op(
                            scale0 * Vector::Convert<Src0Type, CompType>::Do(lhs[i0 * stridesSrc0[IDX_0] +
                                i1 * stridesSrc0[IDX_1] + i2 * stridesSrc0[IDX_2] + i3 * stridesSrc0[IDX_3] +
                                i4 * stridesSrc0[IDX_4]]),
                            scale1 * Vector::Convert<Src1Type, CompType>::Do(rhs[i0 * stridesSrc1[IDX_0] +
                                i1 * stridesSrc1[IDX_1] + i2 * stridesSrc1[IDX_2] + i3 * stridesSrc1[IDX_3] +
                                i4 * stridesSrc1[IDX_4]])
                        ));
                    }
                }
            }
        }
    }
}

template <typename CompType, ARCH Arch = ARCH::VC, typename Src0Type, typename Src1Type, typename OutType,
typename Operation>
inline void EvaluateWrapper(const Src0Type *lhs, const Src1Type *rhs, OutType *dst,
    const float scale0, const float scale1, const Shape &dims, const Shape &stridesSrc0,
    const Shape &stridesSrc1, const Shape &stridesDst, Operation &&op, const SizeType loopLength,
    const SizeType offset) noexcept
{
    if ((stridesSrc0)[IDX_4] == 0 && ((stridesSrc1)[IDX_4] == 0)) {
        Evaluate<CompType, Arch>(lhs, rhs, dst, scale0, scale1, dims, stridesSrc0, stridesSrc1, stridesDst,
            std::forward<Operation>(op),
            Vector::DupWrapperN<Src0Type, CompType, Vector::GetUsedRegsN<Src0Type, Src1Type, CompType, OutType>(),
            Arch>{},
            Vector::DupWrapperN<Src1Type, CompType, Vector::GetUsedRegsN<Src1Type, Src0Type, CompType, OutType>(),
            Arch>{}, loopLength, offset);
    } else if ((stridesSrc0)[IDX_4] == 0) {
        Evaluate<CompType, Arch>(lhs, rhs, dst, scale0, scale1, dims, stridesSrc0, stridesSrc1, stridesDst,
            std::forward<Operation>(op),
            Vector::DupWrapperN<Src0Type, CompType, Vector::GetUsedRegsN<Src0Type, Src1Type, CompType, OutType>(),
            Arch>{},
            Vector::LoadWrapperN<Src1Type, CompType, Vector::GetUsedRegsN<Src1Type, Src0Type, CompType, OutType>(),
            Arch>{}, loopLength, offset);
    } else if ((stridesSrc1)[IDX_4] == 0) {
        Evaluate<CompType, Arch>(lhs, rhs, dst, scale0, scale1, dims, stridesSrc0, stridesSrc1, stridesDst,
            std::forward<Operation>(op),
            Vector::LoadWrapperN<Src0Type, CompType, Vector::GetUsedRegsN<Src0Type, Src1Type, CompType, OutType>(),
            Arch>{},
            Vector::DupWrapperN<Src1Type, CompType, Vector::GetUsedRegsN<Src1Type, Src0Type, CompType, OutType>(),
            Arch>{}, loopLength, offset);
    } else {
        Evaluate<CompType, Arch>(lhs, rhs, dst, scale0, scale1, dims, stridesSrc0, stridesSrc1, stridesDst,
            std::forward<Operation>(op),
            Vector::LoadWrapperN<Src0Type, CompType, Vector::GetUsedRegsN<Src0Type, Src1Type, CompType, OutType>(),
            Arch>{},
            Vector::LoadWrapperN<Src1Type, CompType, Vector::GetUsedRegsN<Src1Type, Src0Type, CompType, OutType>(),
            Arch>{}, loopLength, offset);
    }
}

#if USE_NEON
bool inline IsScalar(const Shape &stridesSrc)
{
    if (stridesSrc[IDX_0] == 0 && stridesSrc[IDX_1] == 0 && stridesSrc[IDX_2] == 0 && stridesSrc[IDX_3] == 0
        && stridesSrc[IDX_4] == 0) {
        return true;
    }
    return false;
}

bool inline IsSingleDimBroadcast(const Shape &stridesSrc0, const SizeType IDX_N)
{
    for (size_t dims = 0; dims < IDX_5; dims++) {
        if (dims == IDX_N) {
            if (stridesSrc0[dims] == 0) {
                return false;
            }
        } else {
            if (stridesSrc0[dims] != 0) {
                return false;
            }
        }
    }
    return true;
}

bool inline IsDimCombinationBroadcast(const Shape &stridesSrc0, const Shape &stridesSrc1, const SizeType IDX_N)
{
    if (IsSingleDimBroadcast(stridesSrc0, IDX_N)) {
        return true;
    }
    if (IsSingleDimBroadcast(stridesSrc1, IDX_N)) {
        return true;
    }
    return false;
}

bool inline IsNeonBest(const Shape &stridesSrc0, const Shape &stridesSrc1)
{
    // filter scalar broadcast
    if (IsScalar(stridesSrc0) || IsScalar(stridesSrc1)) {
        return true;
    }
    // filter channel broadcast
    if (IsDimCombinationBroadcast(stridesSrc0, stridesSrc1, IDX_1)) {
        return true;
    }
    // filter feature broadcast
    if (IsDimCombinationBroadcast(stridesSrc0, stridesSrc1, IDX_4)) {
        return true;
    }
    // neon dont support other broadcast
    return false;
}
#endif

} // Binary
} // Detail
} // KuDNN

#endif //  KUDNN_BINARY_DEF_LOOP_HPP