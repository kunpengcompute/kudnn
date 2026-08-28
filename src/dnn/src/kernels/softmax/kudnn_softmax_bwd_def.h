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

#ifndef KUDNN_SOFTMAX_BWD_DEF_H
#define KUDNN_SOFTMAX_BWD_DEF_H

#include "kudnn.hpp"

#include "vector_wrappers.hpp"
#include "kernels/softmax/kudnn_softmax_kernels.hpp"

namespace KuDNN {
namespace Detail {
namespace Softmax {

template <typename CompType, typename InType, typename OutType>
struct SoftmaxLastBWD final {
    static SizeType LoopInc() noexcept
    {
        return Vector::UnrollLength<InType, CompType, OutType>();
    }
    static void Do(SizeType loopIdx, SizeType loopRange, const OutType *dst, const OutType *dstDiff, bool isLogsoftmax,
        typename Vector::Vec<CompType>::Type &accumVar0, typename Vector::Vec<CompType>::Type &accumVar1,
        typename Vector::Vec<CompType>::Type &accumVar2, typename Vector::Vec<CompType>::Type &accumVar3) noexcept
    {
        svbool_t ldPG0;
        svbool_t ldPG1;
        svbool_t ldPG2;
        svbool_t ldPG3;
        typename Vector::Vec<CompType>::Type lhs0;
        typename Vector::Vec<CompType>::Type lhs1;
        typename Vector::Vec<CompType>::Type lhs2;
        typename Vector::Vec<CompType>::Type lhs3;
        typename Vector::Vec<CompType>::Type rhs0;
        typename Vector::Vec<CompType>::Type rhs1;
        typename Vector::Vec<CompType>::Type rhs2;
        typename Vector::Vec<CompType>::Type rhs3;
        Vector::InitPGN<OutType, Vector::GetUsedRegsN<OutType, InType, CompType>()>::Do(
            ldPG0, ldPG1, ldPG2, ldPG3, loopIdx, loopRange);
        Vector::LoadN<OutType, CompType, Vector::GetUsedRegsN<OutType, InType, CompType>()>::Do(
            dstDiff + loopIdx, ldPG0, ldPG1, ldPG2, ldPG3, rhs0, rhs1, rhs2, rhs3);
        svbool_t opPG0;
        svbool_t opPG1;
        svbool_t opPG2;
        svbool_t opPG3;
        Vector::InitPGN<CompType, Vector::GetUsedRegsN<CompType, InType, OutType>()>::Do(
            opPG0, opPG1, opPG2, opPG3, loopIdx, loopRange);
        if (!isLogsoftmax) {
            Vector::LoadN<OutType, CompType, Vector::GetUsedRegsN<OutType, InType, CompType>()>::Do(
                dst + loopIdx, ldPG0, ldPG1, ldPG2, ldPG3, lhs0, lhs1, lhs2, lhs3);
            Vector::MulZN<CompType, Vector::GetUsedRegsN<CompType, InType, OutType>()>::Do(
                opPG0, opPG1, opPG2, opPG3, lhs0, lhs1, lhs2, lhs3, rhs0, rhs1, rhs2, rhs3, rhs0, rhs1, rhs2, rhs3);
        }
        Vector::AddMN<CompType, Vector::GetUsedRegsN<CompType, InType, OutType>()>::Do(
            opPG0, opPG1, opPG2, opPG3, accumVar0, accumVar1, accumVar2, accumVar3,
            rhs0, rhs1, rhs2, rhs3, accumVar0, accumVar1, accumVar2, accumVar3);
    }
    static void Do(SizeType loopIdx, SizeType loopRange, const OutType *dst, bool isLogsoftmax,
        const OutType *dstDiff, CompType &sbr,
        InType *srcDiff) noexcept
    {
        svbool_t ldPG0;
        svbool_t ldPG1;
        svbool_t ldPG2;
        svbool_t ldPG3;
        Vector::InitPGN<OutType, Vector::GetUsedRegsN<OutType, InType, CompType>()>::Do(
            ldPG0, ldPG1, ldPG2, ldPG3, loopIdx, loopRange);
        typename Vector::Vec<CompType>::Type lhs0;
        typename Vector::Vec<CompType>::Type lhs1;
        typename Vector::Vec<CompType>::Type lhs2;
        typename Vector::Vec<CompType>::Type lhs3;
        Vector::LoadN<OutType, CompType, Vector::GetUsedRegsN<OutType, InType, CompType>()>::Do(
            dst + loopIdx, ldPG0, ldPG1, ldPG2, ldPG3, lhs0, lhs1, lhs2, lhs3);
        typename Vector::Vec<CompType>::Type rhs0;
        typename Vector::Vec<CompType>::Type rhs1;
        typename Vector::Vec<CompType>::Type rhs2;
        typename Vector::Vec<CompType>::Type rhs3;
        Vector::LoadN<OutType, CompType, Vector::GetUsedRegsN<OutType, InType, CompType>()>::Do(
            dstDiff + loopIdx, ldPG0, ldPG1, ldPG2, ldPG3, rhs0, rhs1, rhs2, rhs3);
        svbool_t opPG0;
        svbool_t opPG1;
        svbool_t opPG2;
        svbool_t opPG3;
        Vector::InitPGN<CompType, Vector::GetUsedRegsN<CompType, InType, OutType>()>::Do(
            opPG0, opPG1, opPG2, opPG3, loopIdx, loopRange);
        if (!isLogsoftmax) {
            Vector::SubnXN<CompType, Vector::GetUsedRegsN<CompType, InType, OutType>()>::Do(
                opPG0, opPG1, opPG2, opPG3, rhs0, rhs1, rhs2, rhs3, sbr, sbr, sbr, sbr, rhs0, rhs1, rhs2, rhs3);
            Vector::MulXN<CompType, Vector::GetUsedRegsN<CompType, InType, OutType>()>::Do(
                opPG0, opPG1, opPG2, opPG3, lhs0, lhs1, lhs2, lhs3, rhs0, rhs1, rhs2, rhs3, rhs0, rhs1, rhs2, rhs3);
        } else {
            Vector::ExpN<CompType, Vector::GetUsedRegsN<CompType, InType, OutType>()>::Do(
                opPG0, opPG1, opPG2, opPG3, lhs0, lhs1, lhs2, lhs3, lhs0, lhs1, lhs2, lhs3);
            Vector::MulnXN<CompType, Vector::GetUsedRegsN<CompType, InType, OutType>()>::Do(
                opPG0, opPG1, opPG2, opPG3, lhs0, lhs1, lhs2, lhs3, sbr, sbr, sbr, sbr, lhs0, lhs1, lhs2, lhs3);
            Vector::SubXN<CompType, Vector::GetUsedRegsN<CompType, InType, OutType>()>::Do(
                opPG0, opPG1, opPG2, opPG3, rhs0, rhs1, rhs2, rhs3, lhs0, lhs1, lhs2, lhs3, rhs0, rhs1, rhs2, rhs3);
        }
        svbool_t stPG0;
        svbool_t stPG1;
        svbool_t stPG2;
        svbool_t stPG3;
        Vector::InitPGN<InType, Vector::GetUsedRegsN<InType, OutType, CompType>()>::Do(
            stPG0, stPG1, stPG2, stPG3, loopIdx, loopRange);
        Vector::StoreN<InType, CompType, Vector::GetUsedRegsN<InType, OutType, CompType>()>::Do(
            stPG0, stPG1, stPG2, stPG3, rhs0, rhs1, rhs2, rhs3, srcDiff + loopIdx);
    }
};

template <typename CompType, typename InType, typename OutType>
void EvaluateLastAxisBWD(const OutType *dst, const OutType *dstDiff, InType *srcDiff,
    SizeType axisSize, SizeType outerSize, SizeType axisNum, AlgorithmKind algorithm) noexcept
{
    bool isLogsoftmax = (algorithm == AlgorithmKind::LOGSOFTMAX);
    const SizeType ouStride = axisNum > 0 ? axisSize : 1;
    for (SizeType ou = 0; ou < outerSize; ++ou) {
        SizeType offset = ou * ouStride;
        typename Vector::Vec<CompType>::Type accumVar0;
        typename Vector::Vec<CompType>::Type accumVar1;
        typename Vector::Vec<CompType>::Type accumVar2;
        typename Vector::Vec<CompType>::Type accumVar3;
        Vector::DupFloatN<OutType, CompType, Vector::GetUsedRegsN<OutType, InType, CompType>()>::Do(
            0.0f, accumVar0, accumVar1, accumVar2, accumVar3);
        Service::UnrollFor::Do<1>(axisSize, SoftmaxLastBWD<CompType, InType, OutType>{},
            dst + offset, dstDiff + offset, isLogsoftmax, accumVar0, accumVar1, accumVar2, accumVar3);
        svbool_t sbrPG0;
        svbool_t sbrPG1;
        svbool_t sbrPG2;
        svbool_t sbrPG3;
        Vector::InitPGN<CompType, Vector::GetUsedRegsN<CompType, InType, OutType>()>::Do(
            sbrPG0, sbrPG1, sbrPG2, sbrPG3, 0, axisSize);
        CompType sbr = Vector::AddvN<CompType, Vector::GetUsedRegsN<CompType, InType, OutType>()>::Do(
            sbrPG0, sbrPG1, sbrPG2, sbrPG3, accumVar0, accumVar1, accumVar2, accumVar3);
        Service::UnrollFor::Do<1>(axisSize, SoftmaxLastBWD<CompType, InType, OutType>{},
            dst + offset, isLogsoftmax, dstDiff + offset, sbr, srcDiff + offset);
    }
}

template <typename CompType, typename InType, typename OutType>
struct SoftmaxCrossBWD final {
    static SizeType LoopInc() noexcept
    {
        return 1;
    }
    static void Do(SizeType loopIdx, SizeType loopRange, const OutType *dst, const OutType *diffDst,
        bool isLogsoftmax, SizeType innerSize,
        svbool_t &ldPG0, svbool_t &ldPG1, svbool_t &ldPG2, svbool_t &ldPG3,
        typename Vector::Vec<CompType>::Type &data0, typename Vector::Vec<CompType>::Type &data1,
        typename Vector::Vec<CompType>::Type &data2, typename Vector::Vec<CompType>::Type &data3,
        typename Vector::Vec<CompType>::Type &diffData0, typename Vector::Vec<CompType>::Type &diffData1,
        typename Vector::Vec<CompType>::Type &diffData2, typename Vector::Vec<CompType>::Type &diffData3,
        svbool_t &opPG0, svbool_t &opPG1, svbool_t &opPG2, svbool_t &opPG3,
        typename Vector::Vec<CompType>::Type &sbr0, typename Vector::Vec<CompType>::Type &sbr1,
        typename Vector::Vec<CompType>::Type &sbr2, typename Vector::Vec<CompType>::Type &sbr3) noexcept
    {
        Service::Unused(loopRange);
        Vector::LoadN<OutType, CompType, Vector::GetUsedRegsN<OutType, InType, CompType>()>::Do(
            dst + loopIdx * innerSize, ldPG0, ldPG1, ldPG2, ldPG3, data0, data1, data2, data3);
        Vector::LoadN<OutType, CompType, Vector::GetUsedRegsN<OutType, InType, CompType>()>::Do(
            diffDst + loopIdx * innerSize, ldPG0, ldPG1, ldPG2, ldPG3, diffData0, diffData1, diffData2, diffData3);
        if (isLogsoftmax) {
            Vector::AddXN<CompType, Vector::GetUsedRegsN<CompType, InType, OutType>()>::Do(
                opPG0, opPG1, opPG2, opPG3, sbr0, sbr1, sbr2, sbr3, diffData0, diffData1, diffData2, diffData3,
                sbr0, sbr1, sbr2, sbr3);
        } else {
            Vector::MlaXN<CompType, Vector::GetUsedRegsN<CompType, InType, OutType>()>::Do(
                opPG0, opPG1, opPG2, opPG3, sbr0, sbr1, sbr2, sbr3, diffData0, diffData1, diffData2, diffData3,
                data0, data1, data2, data3, sbr0, sbr1, sbr2, sbr3);
        }
    }
    static void Do(SizeType loopIdx, SizeType loopRange, const OutType *dst, const OutType *diffDst,
        bool isLogsoftmax, SizeType innerSize,
        svbool_t &ldPG0, svbool_t &ldPG1, svbool_t &ldPG2, svbool_t &ldPG3,
        typename Vector::Vec<CompType>::Type &data0, typename Vector::Vec<CompType>::Type &data1,
        typename Vector::Vec<CompType>::Type &data2, typename Vector::Vec<CompType>::Type &data3,
        typename Vector::Vec<CompType>::Type &diffData0, typename Vector::Vec<CompType>::Type &diffData1,
        typename Vector::Vec<CompType>::Type &diffData2, typename Vector::Vec<CompType>::Type &diffData3,
        svbool_t &opPG0, svbool_t &opPG1, svbool_t &opPG2, svbool_t &opPG3,
        typename Vector::Vec<CompType>::Type &sbr0, typename Vector::Vec<CompType>::Type &sbr1,
        typename Vector::Vec<CompType>::Type &sbr2, typename Vector::Vec<CompType>::Type &sbr3,
        svbool_t &stPG0, svbool_t &stPG1, svbool_t &stPG2, svbool_t &stPG3, InType *diffSrc) noexcept
    {
        Service::Unused(loopRange);
        Vector::LoadN<OutType, CompType, Vector::GetUsedRegsN<OutType, InType, CompType>()>::Do(
            dst + loopIdx * innerSize, ldPG0, ldPG1, ldPG2, ldPG3, data0, data1, data2, data3);
        Vector::LoadN<OutType, CompType, Vector::GetUsedRegsN<OutType, InType, CompType>()>::Do(
            diffDst + loopIdx * innerSize, ldPG0, ldPG1, ldPG2, ldPG3, diffData0, diffData1, diffData2, diffData3);
        if (isLogsoftmax) {
            Vector::ExpN<CompType, Vector::GetUsedRegsN<CompType, InType, OutType>()>::Do(
                opPG0, opPG1, opPG2, opPG3, data0, data1, data2, data3, data0, data1, data2, data3);
            Vector::MulXN<CompType, Vector::GetUsedRegsN<CompType, InType, OutType>()>::Do(
                opPG0, opPG1, opPG2, opPG3, data0, data1, data2, data3, sbr0, sbr1, sbr2, sbr3,
                data0, data1, data2, data3);
            Vector::SubXN<CompType, Vector::GetUsedRegsN<CompType, InType, OutType>()>::Do(
                opPG0, opPG1, opPG2, opPG3, diffData0, diffData1, diffData2, diffData3,
                data0, data1, data2, data3, diffData0, diffData1, diffData2, diffData3);
        } else {
            Vector::SubXN<CompType, Vector::GetUsedRegsN<CompType, InType, OutType>()>::Do(
                opPG0, opPG1, opPG2, opPG3, diffData0, diffData1, diffData2, diffData3,
                sbr0, sbr1, sbr2, sbr3, diffData0, diffData1, diffData2, diffData3);
            Vector::MulXN<CompType, Vector::GetUsedRegsN<CompType, InType, OutType>()>::Do(
                opPG0, opPG1, opPG2, opPG3, diffData0, diffData1, diffData2, diffData3,
                data0, data1, data2, data3, diffData0, diffData1, diffData2, diffData3);
        }
        Vector::StoreN<InType, CompType, Vector::GetUsedRegsN<InType, OutType, CompType>()>::Do(
            stPG0, stPG1, stPG2, stPG3, diffData0, diffData1, diffData2, diffData3, diffSrc + loopIdx * innerSize);
    }
};

template <typename CompType, typename InType, typename OutType>
void EvaluateCrossAxisBWD(const OutType *dst, const OutType *dstDiff, InType *srcDiff,
    SizeType innerSize, SizeType axisSize, SizeType outerSize, AlgorithmKind algorithm) noexcept
{
    bool isLogsoftmax = (algorithm == AlgorithmKind::LOGSOFTMAX);
    SizeType loopInc = Vector::UnrollLength<InType, CompType, OutType>();
    for (SizeType ou = 0; ou < outerSize; ++ou) {
        for (SizeType in = 0; in < innerSize; in += loopInc) {
            SizeType offset = ou * axisSize * innerSize + in;
            typename Vector::Vec<CompType>::Type sbr0;
            typename Vector::Vec<CompType>::Type sbr1;
            typename Vector::Vec<CompType>::Type sbr2;
            typename Vector::Vec<CompType>::Type sbr3;
            Vector::DupFloatN<OutType, CompType, Vector::GetUsedRegsN<OutType, InType, CompType>()>::Do(
                0.0f, sbr0, sbr1, sbr2, sbr3);
            typename Vector::Vec<CompType>::Type data0;
            typename Vector::Vec<CompType>::Type data1;
            typename Vector::Vec<CompType>::Type data2;
            typename Vector::Vec<CompType>::Type data3;
            typename Vector::Vec<CompType>::Type diffData0;
            typename Vector::Vec<CompType>::Type diffData1;
            typename Vector::Vec<CompType>::Type diffData2;
            typename Vector::Vec<CompType>::Type diffData3;
            svbool_t ldPG0;
            svbool_t ldPG1;
            svbool_t ldPG2;
            svbool_t ldPG3;
            Vector::InitPGN<OutType, Vector::GetUsedRegsN<OutType, InType, CompType>()>::Do(
                ldPG0, ldPG1, ldPG2, ldPG3, in, innerSize);
            svbool_t opPG0;
            svbool_t opPG1;
            svbool_t opPG2;
            svbool_t opPG3;
            Vector::InitPGN<CompType, Vector::GetUsedRegsN<CompType, InType, OutType>()>::Do(
                opPG0, opPG1, opPG2, opPG3, in, innerSize);
            svbool_t stPG0;
            svbool_t stPG1;
            svbool_t stPG2;
            svbool_t stPG3;
            Vector::InitPGN<InType, Vector::GetUsedRegsN<InType, OutType, CompType>()>::Do(
                stPG0, stPG1, stPG2, stPG3, in, innerSize);
            Service::UnrollFor::Do<1>(axisSize, SoftmaxCrossBWD<CompType, InType, OutType>{},
                dst + offset, dstDiff + offset, isLogsoftmax, innerSize, ldPG0, ldPG1, ldPG2, ldPG3,
                data0, data1, data2, data3,
                diffData0, diffData1, diffData2, diffData3, opPG0, opPG1, opPG2, opPG3, sbr0, sbr1, sbr2, sbr3);
            Service::UnrollFor::Do<1>(axisSize, SoftmaxCrossBWD<CompType, InType, OutType>{},
                dst + offset, dstDiff + offset, isLogsoftmax, innerSize, ldPG0, ldPG1, ldPG2, ldPG3,
                data0, data1, data2, data3,
                diffData0, diffData1, diffData2, diffData3, opPG0, opPG1, opPG2, opPG3, sbr0, sbr1, sbr2, sbr3,
                stPG0, stPG1, stPG2, stPG3, srcDiff + offset);
        }
    }
}

} // Softmax
} // Detail
} // KuDNN

#endif //  KUDNN_SOFTMAX_BWD_DEF_H
