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

#ifndef KUDNN_PRELU_IMPL_HPP
#define KUDNN_PRELU_IMPL_HPP

#include <algorithm>
#include <memory>

#include <arm_sve.h>
#include <arm_neon.h>

#include "kudnn.hpp"

#include "internal_funcs.hpp"
#include "threading_impl.hpp"
#include "vector_wrappers.hpp"

namespace KuDNN {
namespace Detail {
namespace PReLU {
namespace Helpers {

template <typename CompType, typename SrcType, typename WeightsType, typename OutType>
struct UpdateWrapper {
    KUDNN_INLINE static inline void Do(const svbool_t& opGP0, const svbool_t &opGP1, const svbool_t &opGP2,
        const svbool_t &opGP3, typename Vector::Vec<CompType>::Type &res0, typename Vector::Vec<CompType>::Type &res1,
        typename Vector::Vec<CompType>::Type &res2, typename Vector::Vec<CompType>::Type &res3,
        float *outFloat, SizeType higherIdx, SizeType lowerIdx, SizeType loopVar, SizeType loopMax)
    {
        outFloat[higherIdx] += Vector::AddvN<CompType,
            Vector::GetUsedRegsN<CompType, SrcType, WeightsType, OutType, float>()>::Do(
            opGP0, opGP1, opGP2, opGP3, res0, res1, res2, res3);
        Service::Unused(lowerIdx, loopVar, loopMax);
    }
};

template <typename CompType, typename SrcType, typename WeightsType, typename OutType>
struct StoreWrapper {
    KUDNN_INLINE static inline void Do(const svbool_t& opGP0, const svbool_t &opGP1, const svbool_t &opGP2,
        const svbool_t &opGP3, typename Vector::Vec<CompType>::Type &res0, typename Vector::Vec<CompType>::Type &res1,
        typename Vector::Vec<CompType>::Type &res2, typename Vector::Vec<CompType>::Type &res3,
        float *outFloat, SizeType higherIdx, SizeType lowerIdx, SizeType loopVar, SizeType loopMax)
    {
        typename Vector::Vec<CompType>::Type dst0;
        typename Vector::Vec<CompType>::Type dst1;
        typename Vector::Vec<CompType>::Type dst2;
        typename Vector::Vec<CompType>::Type dst3;
        svbool_t stpg0;
        svbool_t stpg1;
        svbool_t stpg2;
        svbool_t stpg3;
        Vector::InitPGN<float,
        Vector::GetUsedRegsN<float, CompType, SrcType, WeightsType, OutType>()>::Do(
            stpg0, stpg1, stpg2, stpg3, loopVar, loopMax);
        Vector::LoadN<float, CompType, Vector::GetUsedRegsN<float, CompType, SrcType, WeightsType, OutType>()>::Do(
            outFloat + higherIdx + lowerIdx, stpg0, stpg1, stpg2, stpg3, dst0, dst1, dst2, dst3);
        Vector::AddXN<CompType, Vector::GetUsedRegsN<CompType, SrcType, WeightsType, OutType, float>()>::Do(
            opGP0, opGP1, opGP2, opGP3, dst0, dst1, dst2, dst3, res0, res1, res2, res3, dst0, dst1, dst2, dst3);
        Vector::StoreN<float, CompType,
            Vector::GetUsedRegsN<float, CompType, SrcType, WeightsType, OutType>()>::Do(
            stpg0, stpg1, stpg2, stpg3, dst0, dst1, dst2, dst3, outFloat + higherIdx + lowerIdx);
    }
};

} // Helpers

enum struct BroadcastType {
    CHANNEL_SHARED = 0,
    WHOLE_TENSOR,
    SHARED_AXES
};

BroadcastType GetBCType(const TensorInfo &srcInfo, const TensorInfo &weightsInfo);
SizeType GetLastOneIdx(Shape reorderedWeiDims);

} // PReLU
} // Detail
} // KuDNN

#endif // KUDNN_PRELU_IMPL_HPP
