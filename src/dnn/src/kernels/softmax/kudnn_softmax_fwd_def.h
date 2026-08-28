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

#ifndef KUDNN_SOFTMAX_FWD_DEF_H
#define KUDNN_SOFTMAX_FWD_DEF_H

#include "kudnn.hpp"

#include "vector_wrappers.hpp"
#include "kernels/softmax/kudnn_softmax_kernels.hpp"

namespace KuDNN {
namespace Detail {
namespace Softmax {

namespace Helpers {

template<typename T>
inline constexpr T MIN_VAL = -std::numeric_limits<T>::infinity();

template<>
inline constexpr __fp16 MIN_VAL<__fp16> = -65504.0f;

} // Helpers

template <typename CompType, typename InType, typename OutType>
struct SoftmaxLastFWD final {
    static SizeType LoopInc() noexcept
    {
        return Vector::UnrollLength<InType, CompType, OutType>();
    }
    static void Do(SizeType loopIdx, SizeType loopRange, const InType *src,
        typename Vector::Vec<CompType>::Type &compVar0, typename Vector::Vec<CompType>::Type &compVar1,
        typename Vector::Vec<CompType>::Type &compVar2, typename Vector::Vec<CompType>::Type &compVar3) noexcept
    {
        svbool_t ldPG0;
        svbool_t ldPG1;
        svbool_t ldPG2;
        svbool_t ldPG3;
        Vector::InitPGN<InType, Vector::GetUsedRegsN<InType, OutType, CompType>()>::Do(
            ldPG0, ldPG1, ldPG2, ldPG3, loopIdx, loopRange);
        typename Vector::Vec<CompType>::Type data0;
        typename Vector::Vec<CompType>::Type data1;
        typename Vector::Vec<CompType>::Type data2;
        typename Vector::Vec<CompType>::Type data3;
        Vector::LoadN<InType, CompType, Vector::GetUsedRegsN<InType, OutType, CompType>()>::Do(
            src + loopIdx, ldPG0, ldPG1, ldPG2, ldPG3, data0, data1, data2, data3);
        svbool_t opPG0;
        svbool_t opPG1;
        svbool_t opPG2;
        svbool_t opPG3;
        Vector::InitPGN<CompType, Vector::GetUsedRegsN<CompType, InType, OutType>()>::Do(
            opPG0, opPG1, opPG2, opPG3, loopIdx, loopRange);
        Vector::MaxMN<CompType, Vector::GetUsedRegsN<CompType, InType, OutType>()>::Do(
            opPG0, opPG1, opPG2, opPG3, compVar0, compVar1, compVar2, compVar3,
            data0, data1, data2, data3, compVar0, compVar1, compVar2, compVar3);
    }
    static void Do(SizeType loopIdx, SizeType loopRange, const InType *src, bool isLogsoftmax, CompType &spaceMax,
        typename Vector::Vec<CompType>::Type &accumVar0, typename Vector::Vec<CompType>::Type &accumVar1,
        typename Vector::Vec<CompType>::Type &accumVar2, typename Vector::Vec<CompType>::Type &accumVar3,
        OutType *dst) noexcept
    {
        svbool_t ldPG0;
        svbool_t ldPG1;
        svbool_t ldPG2;
        svbool_t ldPG3;
        Vector::InitPGN<InType, Vector::GetUsedRegsN<InType, OutType, CompType>()>::Do(
            ldPG0, ldPG1, ldPG2, ldPG3, loopIdx, loopRange);
        typename Vector::Vec<CompType>::Type data0;
        typename Vector::Vec<CompType>::Type data1;
        typename Vector::Vec<CompType>::Type data2;
        typename Vector::Vec<CompType>::Type data3;
        Vector::LoadN<InType, CompType, Vector::GetUsedRegsN<InType, OutType, CompType>()>::Do(
            src + loopIdx, ldPG0, ldPG1, ldPG2, ldPG3, data0, data1, data2, data3);
        svbool_t opPG0;
        svbool_t opPG1;
        svbool_t opPG2;
        svbool_t opPG3;
        Vector::InitPGN<CompType, Vector::GetUsedRegsN<CompType, InType, OutType>()>::Do(
            opPG0, opPG1, opPG2, opPG3, loopIdx, loopRange);
        Vector::SubnXN<CompType, Vector::GetUsedRegsN<CompType, InType, OutType>()>::Do(
            opPG0, opPG1, opPG2, opPG3, data0, data1, data2, data3,
            spaceMax, spaceMax, spaceMax, spaceMax, data0, data1, data2, data3);
        svbool_t stPG0;
        svbool_t stPG1;
        svbool_t stPG2;
        svbool_t stPG3;
        Vector::InitPGN<OutType, Vector::GetUsedRegsN<OutType, InType, CompType>()>::Do(
            stPG0, stPG1, stPG2, stPG3, loopIdx, loopRange);
        if (isLogsoftmax) {
            Vector::StoreN<OutType, CompType, Vector::GetUsedRegsN<OutType, InType, CompType>()>::Do(
                stPG0, stPG1, stPG2, stPG3, data0, data1, data2, data3, dst + loopIdx);
        }
        Vector::ExpN<CompType, Vector::GetUsedRegsN<CompType, InType, OutType>()>::Do(
            opPG0, opPG1, opPG2, opPG3, data0, data1, data2, data3, data0, data1, data2, data3);
        if (!isLogsoftmax) {
            Vector::StoreN<OutType, CompType, Vector::GetUsedRegsN<OutType, InType, CompType>()>::Do(
                stPG0, stPG1, stPG2, stPG3, data0, data1, data2, data3, dst + loopIdx);
        }
        Vector::AddMN<CompType, Vector::GetUsedRegsN<CompType, InType, OutType>()>::Do(
            opPG0, opPG1, opPG2, opPG3, accumVar0, accumVar1, accumVar2, accumVar3,
            data0, data1, data2, data3, accumVar0, accumVar1, accumVar2, accumVar3);
    }
    static void Do(SizeType loopIdx, SizeType loopRange, CompType sum, bool isLogsoftmax,
        OutType *dst) noexcept
    {
        svbool_t stPG0;
        svbool_t stPG1;
        svbool_t stPG2;
        svbool_t stPG3;
        Vector::InitPGN<OutType, Vector::GetUsedRegsN<OutType, InType, CompType>()>::Do(
            stPG0, stPG1, stPG2, stPG3, loopIdx, loopRange);
        typename Vector::Vec<CompType>::Type data0;
        typename Vector::Vec<CompType>::Type data1;
        typename Vector::Vec<CompType>::Type data2;
        typename Vector::Vec<CompType>::Type data3;
        Vector::LoadN<OutType, CompType, Vector::GetUsedRegsN<OutType, InType, CompType>()>::Do(
            dst + loopIdx, stPG0, stPG1, stPG2, stPG3, data0, data1, data2, data3);
        svbool_t opPG0;
        svbool_t opPG1;
        svbool_t opPG2;
        svbool_t opPG3;
        Vector::InitPGN<CompType, Vector::GetUsedRegsN<CompType, InType, OutType>()>::Do(
            opPG0, opPG1, opPG2, opPG3, loopIdx, loopRange);
        if (isLogsoftmax) {
            Vector::AddnXN<CompType, Vector::GetUsedRegsN<CompType, InType, OutType>()>::Do(
                opPG0, opPG1, opPG2, opPG3, data0, data1, data2, data3,
                sum, sum, sum, sum, data0, data1, data2, data3);
        } else {
            Vector::MulnXN<CompType, Vector::GetUsedRegsN<CompType, InType, OutType>()>::Do(
                opPG0, opPG1, opPG2, opPG3, data0, data1, data2, data3,
                sum, sum, sum, sum, data0, data1, data2, data3);
        }
        Vector::StoreN<OutType, CompType, Vector::GetUsedRegsN<OutType, InType, CompType>()>::Do(
            stPG0, stPG1, stPG2, stPG3, data0, data1, data2, data3, dst + loopIdx);
    }
};

template <typename CompType, typename InType, typename OutType>
void EvaluateLastAxisFWD(const InType *src, OutType *dst, SizeType axisSize,
    SizeType outerSize, AlgorithmKind algorithm) noexcept
{
    bool isLogsoftmax = (algorithm == AlgorithmKind::LOGSOFTMAX);
    for (SizeType ou = 0; ou < outerSize; ++ou) {
        SizeType offset = ou * axisSize;
        typename Vector::Vec<CompType>::Type compVar0;
        typename Vector::Vec<CompType>::Type compVar1;
        typename Vector::Vec<CompType>::Type compVar2;
        typename Vector::Vec<CompType>::Type compVar3;
        Vector::DupFloatN<InType, CompType, Vector::GetUsedRegsN<InType, OutType, CompType>()>::Do(
            Helpers::MIN_VAL<CompType>, compVar0, compVar1, compVar2, compVar3);
        Service::UnrollFor::Do<1>(axisSize, SoftmaxLastFWD<CompType, InType, OutType>{},
            src + offset, compVar0, compVar1, compVar2, compVar3);
        svbool_t globalOpPG0;
        svbool_t globalOpPG1;
        svbool_t globalOpPG2;
        svbool_t globalOpPG3;
        Vector::InitPGN<CompType, Vector::GetUsedRegsN<CompType, InType, OutType>()>::Do(
            globalOpPG0, globalOpPG1, globalOpPG2, globalOpPG3, 0, axisSize);
        CompType spaceMax = Vector::MaxvN<CompType, Vector::GetUsedRegsN<CompType, InType, OutType>()>::Do(
            globalOpPG0, globalOpPG1, globalOpPG2, globalOpPG3, compVar0, compVar1, compVar2, compVar3);
        typename Vector::Vec<CompType>::Type accumVar0;
        typename Vector::Vec<CompType>::Type accumVar1;
        typename Vector::Vec<CompType>::Type accumVar2;
        typename Vector::Vec<CompType>::Type accumVar3;
        Vector::DupFloatN<InType, CompType, Vector::GetUsedRegsN<InType, OutType, CompType>()>::Do(
            0.0f, accumVar0, accumVar1, accumVar2, accumVar3);
        Service::UnrollFor::Do<1>(axisSize, SoftmaxLastFWD<CompType, InType, OutType>{},
            src + offset, isLogsoftmax, spaceMax, accumVar0, accumVar1, accumVar2, accumVar3, dst + offset);
        CompType sum = Vector::AddvN<CompType, Vector::GetUsedRegsN<CompType, InType, OutType>()>::Do(
            globalOpPG0, globalOpPG1, globalOpPG2, globalOpPG3, accumVar0, accumVar1, accumVar2, accumVar3);
        sum = (sum != static_cast<CompType>(0)) ? (1.0f / sum) : 1.0f;
        sum = isLogsoftmax ?  std::log(sum) : sum;
        Service::UnrollFor::Do<1>(axisSize, SoftmaxLastFWD<CompType, InType, OutType>{},
            sum, isLogsoftmax, dst + offset);
    }
}

template <typename CompType, typename InType, typename OutType>
struct SoftmaxCrossFWD final {
    static SizeType LoopInc() noexcept
    {
        return 1;
    }
    static void Do(SizeType loopIdx, SizeType loopRange, const InType *src, SizeType innerSize,
        svbool_t &ldPG0, svbool_t &ldPG1, svbool_t &ldPG2, svbool_t &ldPG3,
        typename Vector::Vec<CompType>::Type &data0, typename Vector::Vec<CompType>::Type &data1,
        typename Vector::Vec<CompType>::Type &data2, typename Vector::Vec<CompType>::Type &data3,
        svbool_t &opPG0, svbool_t &opPG1, svbool_t &opPG2, svbool_t &opPG3,
        typename Vector::Vec<CompType>::Type &spaceMax0, typename Vector::Vec<CompType>::Type &spaceMax1,
        typename Vector::Vec<CompType>::Type &spaceMax2, typename Vector::Vec<CompType>::Type &spaceMax3) noexcept
    {
        Service::Unused(loopRange);
        Vector::LoadN<InType, CompType, Vector::GetUsedRegsN<InType, OutType, CompType>()>::Do(
            src + loopIdx * innerSize, ldPG0, ldPG1, ldPG2, ldPG3, data0, data1, data2, data3);
        Vector::MaxXN<CompType, Vector::GetUsedRegsN<CompType, InType, OutType>()>::Do(
            opPG0, opPG1, opPG2, opPG3, spaceMax0, spaceMax1, spaceMax2, spaceMax3,
            data0, data1, data2, data3, spaceMax0, spaceMax1, spaceMax2, spaceMax3);
    }
    static void Do(SizeType loopIdx, SizeType loopRange, const InType *src,
        bool isLogsoftmax, SizeType innerSize, svbool_t &ldPG0, svbool_t &ldPG1, svbool_t &ldPG2, svbool_t &ldPG3,
        typename Vector::Vec<CompType>::Type &data0, typename Vector::Vec<CompType>::Type &data1,
        typename Vector::Vec<CompType>::Type &data2, typename Vector::Vec<CompType>::Type &data3,
        svbool_t &opPG0, svbool_t &opPG1, svbool_t &opPG2, svbool_t &opPG3,
        typename Vector::Vec<CompType>::Type &spaceMax0, typename Vector::Vec<CompType>::Type &spaceMax1,
        typename Vector::Vec<CompType>::Type &spaceMax2, typename Vector::Vec<CompType>::Type &spaceMax3,
        svbool_t &stPG0, svbool_t &stPG1, svbool_t &stPG2, svbool_t &stPG3,
        typename Vector::Vec<CompType>::Type &sum0, typename Vector::Vec<CompType>::Type &sum1,
        typename Vector::Vec<CompType>::Type &sum2, typename Vector::Vec<CompType>::Type &sum3, OutType *dst) noexcept
    {
        Service::Unused(loopRange);
        Vector::LoadN<InType, CompType, Vector::GetUsedRegsN<InType, OutType, CompType>()>::Do(
            src + loopIdx * innerSize, ldPG0, ldPG1, ldPG2, ldPG3, data0, data1, data2, data3);
        Vector::SubXN<CompType, Vector::GetUsedRegsN<CompType, InType, OutType>()>::Do(
            opPG0, opPG1, opPG2, opPG3, data0, data1, data2, data3,
            spaceMax0, spaceMax1, spaceMax2, spaceMax3, data0, data1, data2, data3);
        if (isLogsoftmax) {
            Vector::StoreN<OutType, CompType, Vector::GetUsedRegsN<OutType, InType, CompType>()>::Do(
                stPG0, stPG1, stPG2, stPG3, data0, data1, data2, data3, dst + loopIdx * innerSize);
        }
        Vector::ExpN<CompType, Vector::GetUsedRegsN<CompType, InType, OutType>()>::Do(
            opPG0, opPG1, opPG2, opPG3, data0, data1, data2, data3, data0, data1, data2, data3);
        if (!isLogsoftmax) {
            Vector::StoreN<OutType, CompType, Vector::GetUsedRegsN<OutType, InType, CompType>()>::Do(
                stPG0, stPG1, stPG2, stPG3, data0, data1, data2, data3, dst + loopIdx * innerSize);
        }
        Vector::AddXN<CompType, Vector::GetUsedRegsN<CompType, InType, OutType>()>::Do(
            opPG0, opPG1, opPG2, opPG3, sum0, sum1, sum2, sum3, data0, data1, data2, data3, sum0, sum1, sum2, sum3);
    }
    static void Do(SizeType loopIdx, SizeType loopRange, SizeType innerSize, bool isLogsoftmax,
        typename Vector::Vec<CompType>::Type &data0, typename Vector::Vec<CompType>::Type &data1,
        typename Vector::Vec<CompType>::Type &data2, typename Vector::Vec<CompType>::Type &data3,
        svbool_t &opPG0, svbool_t &opPG1, svbool_t &opPG2, svbool_t &opPG3,
        typename Vector::Vec<CompType>::Type &sum0, typename Vector::Vec<CompType>::Type &sum1,
        typename Vector::Vec<CompType>::Type &sum2, typename Vector::Vec<CompType>::Type &sum3,
        svbool_t &stPG0, svbool_t &stPG1, svbool_t &stPG2, svbool_t &stPG3, OutType *dst) noexcept
    {
        Service::Unused(loopRange);
        Vector::LoadN<OutType, CompType, Vector::GetUsedRegsN<OutType, InType, CompType>()>::Do(
            dst + loopIdx * innerSize, stPG0, stPG1, stPG2, stPG3, data0, data1, data2, data3);
        if (isLogsoftmax) {
            Vector::SubXN<CompType, Vector::GetUsedRegsN<CompType, InType, OutType>()>::Do(
                opPG0, opPG1, opPG2, opPG3, data0, data1, data2, data3,
                sum0, sum1, sum2, sum3, data0, data1, data2, data3);
        } else {
            Vector::DivXN<CompType, Vector::GetUsedRegsN<CompType, InType, OutType>()>::Do(
                opPG0, opPG1, opPG2, opPG3, data0, data1, data2, data3,
                sum0, sum1, sum2, sum3, data0, data1, data2, data3);
        }
        Vector::StoreN<OutType, CompType, Vector::GetUsedRegsN<OutType, InType, CompType>()>::Do(
            stPG0, stPG1, stPG2, stPG3, data0, data1, data2, data3, dst + loopIdx * innerSize);
    }
};

template <typename CompType, typename InType, typename OutType>
void EvaluateCrossAxisFWD(const InType *src, OutType *dst,
    SizeType innerSize, SizeType axisSize, SizeType outerSize, AlgorithmKind algorithm) noexcept
{
    bool isLogsoftmax = (algorithm == AlgorithmKind::LOGSOFTMAX);
    SizeType loopInc = Vector::UnrollLength<InType, CompType, OutType>();
    for (SizeType ou = 0; ou < outerSize; ++ou) {
        for (SizeType in = 0; in < innerSize; in += loopInc) {
            SizeType offset = ou * axisSize * innerSize + in;
            typename Vector::Vec<CompType>::Type spaceMax0;
            typename Vector::Vec<CompType>::Type spaceMax1;
            typename Vector::Vec<CompType>::Type spaceMax2;
            typename Vector::Vec<CompType>::Type spaceMax3;
            Vector::DupFloatN<InType, CompType, Vector::GetUsedRegsN<InType, OutType, CompType>()>::Do(
                Helpers::MIN_VAL<CompType>, spaceMax0, spaceMax1, spaceMax2, spaceMax3);
            typename Vector::Vec<CompType>::Type sum0;
            typename Vector::Vec<CompType>::Type sum1;
            typename Vector::Vec<CompType>::Type sum2;
            typename Vector::Vec<CompType>::Type sum3;
            Vector::DupFloatN<InType, CompType, Vector::GetUsedRegsN<InType, OutType, CompType>()>::Do(
                0.0f, sum0, sum1, sum2, sum3);
            typename Vector::Vec<CompType>::Type data0;
            typename Vector::Vec<CompType>::Type data1;
            typename Vector::Vec<CompType>::Type data2;
            typename Vector::Vec<CompType>::Type data3;
            svbool_t ldPG0;
            svbool_t ldPG1;
            svbool_t ldPG2;
            svbool_t ldPG3;
            Vector::InitPGN<InType, Vector::GetUsedRegsN<InType, CompType, OutType>()>::Do(
                ldPG0, ldPG1, ldPG2, ldPG3, in, innerSize);
            svbool_t opPG0;
            svbool_t opPG1;
            svbool_t opPG2;
            svbool_t opPG3;
            svbool_t stPG0;
            svbool_t stPG1;
            svbool_t stPG2;
            svbool_t stPG3;
            Vector::InitPGN<CompType, Vector::GetUsedRegsN<CompType, InType, OutType>()>::Do(
                opPG0, opPG1, opPG2, opPG3, in, innerSize);
            Vector::InitPGN<OutType, Vector::GetUsedRegsN<OutType, InType, CompType>()>::Do(
                stPG0, stPG1, stPG2, stPG3, in, innerSize);
            Service::UnrollFor::Do<1>(axisSize, SoftmaxCrossFWD<CompType, InType, OutType>{},
                src + offset, innerSize, ldPG0, ldPG1, ldPG2, ldPG3, data0, data1, data2, data3,
                opPG0, opPG1, opPG2, opPG3, spaceMax0, spaceMax1, spaceMax2, spaceMax3);
            Service::UnrollFor::Do<1>(axisSize, SoftmaxCrossFWD<CompType, InType, OutType>{},
                src + offset, isLogsoftmax, innerSize, ldPG0, ldPG1, ldPG2, ldPG3, data0, data1, data2, data3,
                opPG0, opPG1, opPG2, opPG3, spaceMax0, spaceMax1, spaceMax2, spaceMax3,
                stPG0, stPG1, stPG2, stPG3, sum0, sum1, sum2, sum3, dst + offset);
            if (isLogsoftmax) {
                Vector::LogN<CompType, Vector::GetUsedRegsN<CompType, InType, OutType>()>::Do(
                    opPG0, opPG1, opPG2, opPG3, sum0, sum1, sum2, sum3, sum0, sum1, sum2, sum3);
            }
            Service::UnrollFor::Do<1>(axisSize, SoftmaxCrossFWD<CompType, InType, OutType>{},
                innerSize, isLogsoftmax, data0, data1, data2, data3,
                opPG0, opPG1, opPG2, opPG3, sum0, sum1, sum2, sum3,
                stPG0, stPG1, stPG2, stPG3, dst + offset);
        }
    }
}

} // Softmax
} // Detail
} // KuDNN

#endif //  KUDNN_SOFTMAX_FWD_DEF_H
