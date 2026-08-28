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

#ifndef KUDNN_BINARY_DEF_HPP
#define KUDNN_BINARY_DEF_HPP

#include <arm_sve.h>

#include "kudnn.hpp"

#include "vector_wrappers.hpp"
#include "kernels/binary/kudnn_binary_kernels.hpp"
#include "kernels/binary/kudnn_binary_def_loop.hpp"
#include "kernels/binary/kudnn_binary_def_helper.hpp"

namespace KuDNN {
namespace Detail {
namespace Binary {

template <typename CompType, typename Src0Type, typename Src1Type, typename OutType>
void EvaluateAdd(const Src0Type *lhs, const Src1Type *rhs, OutType *dst,
    const float scale0, const float scale1, const Shape &dims, const Shape &stridesSrc0,
    const Shape &stridesSrc1, const Shape &stridesDst, bool isSeqKernel) noexcept
{
    if (isSeqKernel) {
        Binary::EvaluateSC<CompType>(lhs, rhs, dst, scale0, scale1, dims, stridesSrc0, stridesSrc1, stridesDst,
            [](CompType a, CompType b) -> CompType { return a + b; });
    } else {
#if USE_NEON
    if (IsNeonBest(stridesSrc0, stridesSrc1)) {
        SizeType neonLoop = dims[IDX_4] - (dims[IDX_4] % (NUM_2 * Vector::NeonLanes<Src0Type, Src1Type,
            OutType>()));
        SizeType tailLoop = dims[IDX_4] % (NUM_2 * Vector::NeonLanes<Src0Type, Src1Type, OutType>()); // Use VC
        Binary::EvaluateWrapper<CompType, ARCH::NEON>(lhs, rhs, dst, scale0, scale1, dims, stridesSrc0, stridesSrc1,
            stridesDst, Vector::AddXN<CompType, Vector::GetUsedRegsN<CompType, Src0Type, Src1Type, OutType>(),
            ARCH::NEON>{}, neonLoop, 0);
        Binary::EvaluateWrapper<CompType, ARCH::VC>(lhs, rhs, dst, scale0, scale1, dims, stridesSrc0,
            stridesSrc1, stridesDst, Vector::AddXN<CompType, Vector::GetUsedRegsN<CompType, Src0Type, Src1Type,
            OutType>(), ARCH::VC>{}, tailLoop, neonLoop);
    } else {
        Binary::EvaluateWrapper<CompType, ARCH::VC>(lhs, rhs, dst, scale0, scale1, dims, stridesSrc0, stridesSrc1,
            stridesDst, Vector::AddXN<CompType, Vector::GetUsedRegsN<CompType, Src0Type, Src1Type, OutType>(),
            ARCH::VC>{}, dims[IDX_4], 0);
    }
#else
    Binary::EvaluateWrapper<CompType, ARCH::VC>(lhs, rhs, dst, scale0, scale1, dims, stridesSrc0, stridesSrc1,
        stridesDst, Vector::AddXN<CompType, Vector::GetUsedRegsN<CompType, Src0Type, Src1Type, OutType>(),
        ARCH::VC>{}, dims[IDX_4], 0);
#endif
    }
}

template <typename CompType, typename Src0Type, typename Src1Type, typename OutType>
void EvaluateDiv(const Src0Type *lhs, const Src1Type *rhs, OutType *dst,
    const float scale0, const float scale1, const Shape &dims, const Shape &stridesSrc0,
    const Shape &stridesSrc1, const Shape &stridesDst, bool isSeqKernel) noexcept
{
    if (isSeqKernel) {
        Binary::EvaluateSC<CompType>(lhs, rhs, dst, scale0, scale1, dims, stridesSrc0, stridesSrc1, stridesDst,
            [](CompType a, CompType b) -> CompType { return a / b; });
    } else {
#if USE_NEON
    if (IsNeonBest(stridesSrc0, stridesSrc1)) {
        SizeType neonLoop = dims[IDX_4] - (dims[IDX_4] % (NUM_2 * Vector::NeonLanes<Src0Type, Src1Type,
            OutType>()));
        SizeType tailLoop = dims[IDX_4] % (NUM_2 * Vector::NeonLanes<Src0Type, Src1Type, OutType>()); // Use VC
        Binary::EvaluateWrapper<CompType, ARCH::NEON>(lhs, rhs, dst, scale0, scale1, dims, stridesSrc0, stridesSrc1,
            stridesDst, Vector::DivXN<CompType, Vector::GetUsedRegsN<CompType, Src0Type, Src1Type, OutType>(),
            ARCH::NEON>{}, neonLoop, 0);
        Binary::EvaluateWrapper<CompType, ARCH::VC>(lhs, rhs, dst, scale0, scale1, dims, stridesSrc0,
            stridesSrc1, stridesDst, Vector::DivXN<CompType, Vector::GetUsedRegsN<CompType, Src0Type, Src1Type,
            OutType>(), ARCH::VC>{}, tailLoop, neonLoop);
    } else {
        Binary::EvaluateWrapper<CompType, ARCH::VC>(lhs, rhs, dst, scale0, scale1, dims, stridesSrc0, stridesSrc1,
            stridesDst, Vector::DivXN<CompType, Vector::GetUsedRegsN<CompType, Src0Type, Src1Type, OutType>(),
            ARCH::VC>{}, dims[IDX_4], 0);
    }
#else
    Binary::EvaluateWrapper<CompType, ARCH::VC>(lhs, rhs, dst, scale0, scale1, dims, stridesSrc0, stridesSrc1,
        stridesDst, Vector::DivXN<CompType, Vector::GetUsedRegsN<CompType, Src0Type, Src1Type, OutType>(),
        ARCH::VC>{}, dims[IDX_4], 0);
#endif
    }
}

template <typename CompType, typename Src0Type, typename Src1Type, typename OutType>
void EvaluateMax(const Src0Type *lhs, const Src1Type *rhs, OutType *dst,
    const float scale0, const float scale1, const Shape &dims, const Shape &stridesSrc0,
    const Shape &stridesSrc1, const Shape &stridesDst, bool isSeqKernel) noexcept
{
    if (isSeqKernel) {
        Binary::EvaluateSC<CompType>(lhs, rhs, dst, scale0, scale1, dims, stridesSrc0, stridesSrc1, stridesDst,
            [](CompType a, CompType b) -> CompType { return (a >= b) ? a : b; });
    } else {
#if USE_NEON
    if (IsNeonBest(stridesSrc0, stridesSrc1)) {
        SizeType neonLoop = dims[IDX_4] - (dims[IDX_4] % (NUM_2 * Vector::NeonLanes<Src0Type, Src1Type,
            OutType>()));
        SizeType tailLoop = dims[IDX_4] % (NUM_2 * Vector::NeonLanes<Src0Type, Src1Type, OutType>()); // Use VC
        Binary::EvaluateWrapper<CompType, ARCH::NEON>(lhs, rhs, dst, scale0, scale1, dims, stridesSrc0, stridesSrc1,
            stridesDst, Vector::MaxXN<CompType, Vector::GetUsedRegsN<CompType, Src0Type, Src1Type, OutType>(),
            ARCH::NEON>{}, neonLoop, 0);
        Binary::EvaluateWrapper<CompType, ARCH::VC>(lhs, rhs, dst, scale0, scale1, dims, stridesSrc0,
            stridesSrc1, stridesDst, Vector::MaxXN<CompType, Vector::GetUsedRegsN<CompType, Src0Type, Src1Type,
            OutType>(), ARCH::VC>{}, tailLoop, neonLoop);
    } else {
        Binary::EvaluateWrapper<CompType, ARCH::VC>(lhs, rhs, dst, scale0, scale1, dims, stridesSrc0, stridesSrc1,
            stridesDst, Vector::MaxXN<CompType, Vector::GetUsedRegsN<CompType, Src0Type, Src1Type, OutType>(),
            ARCH::VC>{}, dims[IDX_4], 0);
    }
#else
    Binary::EvaluateWrapper<CompType, ARCH::VC>(lhs, rhs, dst, scale0, scale1, dims, stridesSrc0, stridesSrc1,
        stridesDst, Vector::MaxXN<CompType, Vector::GetUsedRegsN<CompType, Src0Type, Src1Type, OutType>(),
        ARCH::VC>{}, dims[IDX_4], 0);
#endif
    }
}

template <typename CompType, typename Src0Type, typename Src1Type, typename OutType>
void EvaluateMin(const Src0Type *lhs, const Src1Type *rhs, OutType *dst,
    const float scale0, const float scale1, const Shape &dims, const Shape &stridesSrc0,
    const Shape &stridesSrc1, const Shape &stridesDst, bool isSeqKernel) noexcept
{
    if (isSeqKernel) {
        Binary::EvaluateSC<CompType>(lhs, rhs, dst, scale0, scale1, dims, stridesSrc0, stridesSrc1, stridesDst,
            [](CompType a, CompType b) -> CompType { return (a <= b) ? a : b; });
    } else {
#if USE_NEON
    if (IsNeonBest(stridesSrc0, stridesSrc1)) {
        SizeType neonLoop = dims[IDX_4] - (dims[IDX_4] % (NUM_2 * Vector::NeonLanes<Src0Type, Src1Type,
            OutType>()));
        SizeType tailLoop = dims[IDX_4] % (NUM_2 * Vector::NeonLanes<Src0Type, Src1Type, OutType>()); // Use VC
        Binary::EvaluateWrapper<CompType, ARCH::NEON>(lhs, rhs, dst, scale0, scale1, dims, stridesSrc0, stridesSrc1,
            stridesDst, Vector::MinXN<CompType, Vector::GetUsedRegsN<CompType, Src0Type, Src1Type, OutType>(),
            ARCH::NEON>{}, neonLoop, 0);
        Binary::EvaluateWrapper<CompType, ARCH::VC>(lhs, rhs, dst, scale0, scale1, dims, stridesSrc0,
            stridesSrc1, stridesDst, Vector::MinXN<CompType, Vector::GetUsedRegsN<CompType, Src0Type, Src1Type,
            OutType>(), ARCH::VC>{}, tailLoop, neonLoop);
    } else {
        Binary::EvaluateWrapper<CompType, ARCH::VC>(lhs, rhs, dst, scale0, scale1, dims, stridesSrc0, stridesSrc1,
            stridesDst, Vector::MinXN<CompType, Vector::GetUsedRegsN<CompType, Src0Type, Src1Type, OutType>(),
            ARCH::VC>{}, dims[IDX_4], 0);
    }
#else
    Binary::EvaluateWrapper<CompType, ARCH::VC>(lhs, rhs, dst, scale0, scale1, dims, stridesSrc0, stridesSrc1,
        stridesDst, Vector::MinXN<CompType, Vector::GetUsedRegsN<CompType, Src0Type, Src1Type, OutType>(),
        ARCH::VC>{}, dims[IDX_4], 0);
#endif
    }
}

template <typename CompType, typename Src0Type, typename Src1Type, typename OutType>
void EvaluateMul(const Src0Type *lhs, const Src1Type *rhs, OutType *dst,
    const float scale0, const float scale1, const Shape &dims, const Shape &stridesSrc0,
    const Shape &stridesSrc1, const Shape &stridesDst, bool isSeqKernel) noexcept
{
    if (isSeqKernel) {
        Binary::EvaluateSC<CompType>(lhs, rhs, dst, scale0, scale1, dims, stridesSrc0, stridesSrc1, stridesDst,
            [](CompType a, CompType b) -> CompType { return a * b; });
    } else {
#if USE_NEON
    if (IsNeonBest(stridesSrc0, stridesSrc1)) {
        SizeType neonLoop = dims[IDX_4] - (dims[IDX_4] % (NUM_2 * Vector::NeonLanes<Src0Type, Src1Type,
            OutType>()));
        SizeType tailLoop = dims[IDX_4] % (NUM_2 * Vector::NeonLanes<Src0Type, Src1Type, OutType>()); // Use VC
        Binary::EvaluateWrapper<CompType, ARCH::NEON>(lhs, rhs, dst, scale0, scale1, dims, stridesSrc0, stridesSrc1,
            stridesDst, Vector::MulXN<CompType, Vector::GetUsedRegsN<CompType, Src0Type, Src1Type, OutType>(),
            ARCH::NEON>{}, neonLoop, 0);
        Binary::EvaluateWrapper<CompType, ARCH::VC>(lhs, rhs, dst, scale0, scale1, dims, stridesSrc0,
            stridesSrc1, stridesDst, Vector::MulXN<CompType, Vector::GetUsedRegsN<CompType, Src0Type, Src1Type,
            OutType>(), ARCH::VC>{}, tailLoop, neonLoop);
    } else {
        Binary::EvaluateWrapper<CompType, ARCH::VC>(lhs, rhs, dst, scale0, scale1, dims, stridesSrc0, stridesSrc1,
            stridesDst, Vector::MulXN<CompType, Vector::GetUsedRegsN<CompType, Src0Type, Src1Type, OutType>(),
            ARCH::VC>{}, dims[IDX_4], 0);
    }
#else
    Binary::EvaluateWrapper<CompType, ARCH::VC>(lhs, rhs, dst, scale0, scale1, dims, stridesSrc0, stridesSrc1,
        stridesDst, Vector::MulXN<CompType, Vector::GetUsedRegsN<CompType, Src0Type, Src1Type, OutType>(),
        ARCH::VC>{}, dims[IDX_4], 0);
#endif
    }
}

template <typename CompType, typename Src0Type, typename Src1Type, typename OutType>
void EvaluateSub(const Src0Type *lhs, const Src1Type *rhs, OutType *dst,
    const float scale0, const float scale1, const Shape &dims, const Shape &stridesSrc0,
    const Shape &stridesSrc1, const Shape &stridesDst, bool isSeqKernel) noexcept
{
    if (isSeqKernel) {
        Binary::EvaluateSC<CompType>(lhs, rhs, dst, scale0, scale1, dims, stridesSrc0, stridesSrc1, stridesDst,
            [](CompType a, CompType b) -> CompType { return a - b; });
    } else {
#if USE_NEON
    if (IsNeonBest(stridesSrc0, stridesSrc1)) {
        SizeType neonLoop = dims[IDX_4] - (dims[IDX_4] % (NUM_2 * Vector::NeonLanes<Src0Type, Src1Type,
            OutType>()));
        SizeType tailLoop = dims[IDX_4] % (NUM_2 * Vector::NeonLanes<Src0Type, Src1Type, OutType>()); // Use VC
        Binary::EvaluateWrapper<CompType, ARCH::NEON>(lhs, rhs, dst, scale0, scale1, dims, stridesSrc0, stridesSrc1,
            stridesDst, Vector::SubXN<CompType, Vector::GetUsedRegsN<CompType, Src0Type, Src1Type, OutType>(),
            ARCH::NEON>{}, neonLoop, 0);
        Binary::EvaluateWrapper<CompType, ARCH::VC>(lhs, rhs, dst, scale0, scale1, dims, stridesSrc0,
            stridesSrc1, stridesDst, Vector::SubXN<CompType, Vector::GetUsedRegsN<CompType, Src0Type, Src1Type,
            OutType>(), ARCH::VC>{}, tailLoop, neonLoop);
    } else {
        Binary::EvaluateWrapper<CompType, ARCH::VC>(lhs, rhs, dst, scale0, scale1, dims, stridesSrc0, stridesSrc1,
            stridesDst, Vector::SubXN<CompType, Vector::GetUsedRegsN<CompType, Src0Type, Src1Type, OutType>(),
            ARCH::VC>{}, dims[IDX_4], 0);
    }
#else
    Binary::EvaluateWrapper<CompType, ARCH::VC>(lhs, rhs, dst, scale0, scale1, dims, stridesSrc0, stridesSrc1,
        stridesDst, Vector::SubXN<CompType, Vector::GetUsedRegsN<CompType, Src0Type, Src1Type, OutType>(),
        ARCH::VC>{}, dims[IDX_4], 0);
#endif
    }
}

template <typename CompType, typename Src0Type, typename Src1Type, typename OutType>
void EvaluateGe(const Src0Type *lhs, const Src1Type *rhs, OutType *dst,
    const float scale0, const float scale1, const Shape &dims, const Shape &stridesSrc0,
    const Shape &stridesSrc1, const Shape &stridesDst, bool isSeqKernel) noexcept
{
    if (isSeqKernel) {
        Binary::EvaluateSC<CompType>(lhs, rhs, dst, scale0, scale1, dims, stridesSrc0, stridesSrc1, stridesDst,
            [](CompType a, CompType b) -> CompType { return (a >= b) ? 1 : 0; });
    } else {
#if USE_NEON
    if (IsNeonBest(stridesSrc0, stridesSrc1)) {
        SizeType neonLoop = dims[IDX_4] - (dims[IDX_4] % (NUM_2 * Vector::NeonLanes<Src0Type, Src1Type,
            OutType>()));
        SizeType tailLoop = dims[IDX_4] % (NUM_2 * Vector::NeonLanes<Src0Type, Src1Type, OutType>()); // Use VC
        Binary::EvaluateWrapper<CompType, ARCH::NEON>(lhs, rhs, dst, scale0, scale1, dims, stridesSrc0, stridesSrc1,
            stridesDst, Helpers::ApeZN<Helpers::GeNWrapper<CompType,
            Vector::GetUsedRegsN<CompType, Src0Type, Src1Type, OutType>(), ARCH::NEON>,
            CompType, Vector::GetUsedRegsN<CompType, Src0Type, Src1Type, OutType>(), ARCH::NEON>{}, neonLoop, 0);
        Binary::EvaluateWrapper<CompType, ARCH::VC>(lhs, rhs, dst, scale0, scale1, dims, stridesSrc0, stridesSrc1,
            stridesDst, Helpers::ApeZN<Helpers::GeNWrapper<CompType,
            Vector::GetUsedRegsN<CompType, Src0Type, Src1Type, OutType>(), ARCH::VC>,
            CompType, Vector::GetUsedRegsN<CompType, Src0Type, Src1Type, OutType>(), ARCH::VC>{}, tailLoop, neonLoop);
    } else {
        Binary::EvaluateWrapper<CompType, ARCH::VC>(lhs, rhs, dst, scale0, scale1, dims, stridesSrc0, stridesSrc1,
            stridesDst, Helpers::ApeZN<Helpers::GeNWrapper<CompType,
            Vector::GetUsedRegsN<CompType, Src0Type, Src1Type, OutType>(), ARCH::VC>,
            CompType, Vector::GetUsedRegsN<CompType, Src0Type, Src1Type, OutType>(), ARCH::VC>{}, dims[IDX_4], 0);
    }
#else
    Binary::EvaluateWrapper<CompType, ARCH::VC>(lhs, rhs, dst, scale0, scale1, dims, stridesSrc0, stridesSrc1,
        stridesDst, Helpers::ApeZN<Helpers::GeNWrapper<CompType,
        Vector::GetUsedRegsN<CompType, Src0Type, Src1Type, OutType>(), ARCH::VC>,
        CompType, Vector::GetUsedRegsN<CompType, Src0Type, Src1Type, OutType>(), ARCH::VC>{}, dims[IDX_4], 0);
#endif
    }
}

template <typename CompType, typename Src0Type, typename Src1Type, typename OutType>
void EvaluateGt(const Src0Type *lhs, const Src1Type *rhs, OutType *dst,
    const float scale0, const float scale1, const Shape &dims, const Shape &stridesSrc0,
    const Shape &stridesSrc1, const Shape &stridesDst, bool isSeqKernel) noexcept
{
    if (isSeqKernel) {
        Binary::EvaluateSC<CompType>(lhs, rhs, dst, scale0, scale1, dims, stridesSrc0, stridesSrc1, stridesDst,
            [](CompType a, CompType b) -> CompType { return (a > b) ? 1 : 0; });
    } else {
#if USE_NEON
    if (IsNeonBest(stridesSrc0, stridesSrc1)) {
        SizeType neonLoop = dims[IDX_4] - (dims[IDX_4] % (NUM_2 * Vector::NeonLanes<Src0Type, Src1Type,
            OutType>()));
        SizeType tailLoop = dims[IDX_4] % (NUM_2 * Vector::NeonLanes<Src0Type, Src1Type, OutType>()); // Use VC
        Binary::EvaluateWrapper<CompType, ARCH::NEON>(lhs, rhs, dst, scale0, scale1, dims, stridesSrc0, stridesSrc1,
            stridesDst, Helpers::ApeZN<Helpers::GtNWrapper<CompType,
            Vector::GetUsedRegsN<CompType, Src0Type, Src1Type, OutType>(), ARCH::NEON>,
            CompType, Vector::GetUsedRegsN<CompType, Src0Type, Src1Type, OutType>(), ARCH::NEON>{}, neonLoop, 0);
        Binary::EvaluateWrapper<CompType, ARCH::VC>(lhs, rhs, dst, scale0, scale1, dims, stridesSrc0, stridesSrc1,
            stridesDst, Helpers::ApeZN<Helpers::GtNWrapper<CompType,
            Vector::GetUsedRegsN<CompType, Src0Type, Src1Type, OutType>(), ARCH::VC>,
            CompType, Vector::GetUsedRegsN<CompType, Src0Type, Src1Type, OutType>(), ARCH::VC>{}, tailLoop, neonLoop);
    } else {
        Binary::EvaluateWrapper<CompType, ARCH::VC>(lhs, rhs, dst, scale0, scale1, dims, stridesSrc0, stridesSrc1,
            stridesDst, Helpers::ApeZN<Helpers::GtNWrapper<CompType,
            Vector::GetUsedRegsN<CompType, Src0Type, Src1Type, OutType>(), ARCH::VC>,
            CompType, Vector::GetUsedRegsN<CompType, Src0Type, Src1Type, OutType>(), ARCH::VC>{}, dims[IDX_4], 0);
    }
#else
    Binary::EvaluateWrapper<CompType, ARCH::VC>(lhs, rhs, dst, scale0, scale1, dims, stridesSrc0, stridesSrc1,
        stridesDst, Helpers::ApeZN<Helpers::GtNWrapper<CompType,
        Vector::GetUsedRegsN<CompType, Src0Type, Src1Type, OutType>(), ARCH::VC>,
        CompType, Vector::GetUsedRegsN<CompType, Src0Type, Src1Type, OutType>(), ARCH::VC>{}, dims[IDX_4], 0);
#endif
    }
}

template <typename CompType, typename Src0Type, typename Src1Type, typename OutType>
void EvaluateLe(const Src0Type *lhs, const Src1Type *rhs, OutType *dst,
    const float scale0, const float scale1, const Shape &dims, const Shape &stridesSrc0,
    const Shape &stridesSrc1, const Shape &stridesDst, bool isSeqKernel) noexcept
{
    if (isSeqKernel) {
        Binary::EvaluateSC<CompType>(lhs, rhs, dst, scale0, scale1, dims, stridesSrc0, stridesSrc1, stridesDst,
            [](CompType a, CompType b) -> CompType { return (a <= b) ? 1 : 0; });
    } else {
#if USE_NEON
    if (IsNeonBest(stridesSrc0, stridesSrc1)) {
        SizeType neonLoop = dims[IDX_4] - (dims[IDX_4] % (NUM_2 * Vector::NeonLanes<Src0Type, Src1Type,
            OutType>()));
        SizeType tailLoop = dims[IDX_4] % (NUM_2 * Vector::NeonLanes<Src0Type, Src1Type, OutType>()); // Use VC
        Binary::EvaluateWrapper<CompType, ARCH::NEON>(lhs, rhs, dst, scale0, scale1, dims, stridesSrc0, stridesSrc1,
            stridesDst, Helpers::ApeZN<Helpers::LeNWrapper<CompType,
            Vector::GetUsedRegsN<CompType, Src0Type, Src1Type, OutType>(), ARCH::NEON>,
            CompType, Vector::GetUsedRegsN<CompType, Src0Type, Src1Type, OutType>(), ARCH::NEON>{}, neonLoop, 0);
        Binary::EvaluateWrapper<CompType, ARCH::VC>(lhs, rhs, dst, scale0, scale1, dims, stridesSrc0, stridesSrc1,
            stridesDst, Helpers::ApeZN<Helpers::LeNWrapper<CompType,
            Vector::GetUsedRegsN<CompType, Src0Type, Src1Type, OutType>(), ARCH::VC>,
            CompType, Vector::GetUsedRegsN<CompType, Src0Type, Src1Type, OutType>(), ARCH::VC>{}, tailLoop, neonLoop);
    } else {
        Binary::EvaluateWrapper<CompType, ARCH::VC>(lhs, rhs, dst, scale0, scale1, dims, stridesSrc0, stridesSrc1,
            stridesDst, Helpers::ApeZN<Helpers::LeNWrapper<CompType,
            Vector::GetUsedRegsN<CompType, Src0Type, Src1Type, OutType>(), ARCH::VC>,
            CompType, Vector::GetUsedRegsN<CompType, Src0Type, Src1Type, OutType>(), ARCH::VC>{}, dims[IDX_4], 0);
    }
#else
    Binary::EvaluateWrapper<CompType, ARCH::VC>(lhs, rhs, dst, scale0, scale1, dims, stridesSrc0, stridesSrc1,
        stridesDst, Helpers::ApeZN<Helpers::LeNWrapper<CompType,
        Vector::GetUsedRegsN<CompType, Src0Type, Src1Type, OutType>(), ARCH::VC>,
        CompType, Vector::GetUsedRegsN<CompType, Src0Type, Src1Type, OutType>(), ARCH::VC>{}, dims[IDX_4], 0);
#endif
    }
}

template <typename CompType, typename Src0Type, typename Src1Type, typename OutType>
void EvaluateLt(const Src0Type *lhs, const Src1Type *rhs, OutType *dst,
    const float scale0, const float scale1, const Shape &dims, const Shape &stridesSrc0,
    const Shape &stridesSrc1, const Shape &stridesDst, bool isSeqKernel) noexcept
{
    if (isSeqKernel) {
        Binary::EvaluateSC<CompType>(lhs, rhs, dst, scale0, scale1, dims, stridesSrc0, stridesSrc1, stridesDst,
            [](CompType a, CompType b) -> CompType { return (a < b) ? 1 : 0; });
    } else {
#if USE_NEON
    if (IsNeonBest(stridesSrc0, stridesSrc1)) {
        SizeType neonLoop = dims[IDX_4] - (dims[IDX_4] % (NUM_2 * Vector::NeonLanes<Src0Type, Src1Type,
            OutType>()));
        SizeType tailLoop = dims[IDX_4] % (NUM_2 * Vector::NeonLanes<Src0Type, Src1Type, OutType>()); // Use VC
        Binary::EvaluateWrapper<CompType, ARCH::NEON>(lhs, rhs, dst, scale0, scale1, dims, stridesSrc0, stridesSrc1,
            stridesDst, Helpers::ApeZN<Helpers::LtNWrapper<CompType,
            Vector::GetUsedRegsN<CompType, Src0Type, Src1Type, OutType>(), ARCH::NEON>,
            CompType, Vector::GetUsedRegsN<CompType, Src0Type, Src1Type, OutType>(), ARCH::NEON>{}, neonLoop, 0);
        Binary::EvaluateWrapper<CompType, ARCH::VC>(lhs, rhs, dst, scale0, scale1, dims, stridesSrc0, stridesSrc1,
            stridesDst, Helpers::ApeZN<Helpers::LtNWrapper<CompType,
            Vector::GetUsedRegsN<CompType, Src0Type, Src1Type, OutType>(), ARCH::VC>,
            CompType, Vector::GetUsedRegsN<CompType, Src0Type, Src1Type, OutType>(), ARCH::VC>{}, tailLoop, neonLoop);
    } else {
        Binary::EvaluateWrapper<CompType, ARCH::VC>(lhs, rhs, dst, scale0, scale1, dims, stridesSrc0, stridesSrc1,
            stridesDst, Helpers::ApeZN<Helpers::LtNWrapper<CompType,
            Vector::GetUsedRegsN<CompType, Src0Type, Src1Type, OutType>(), ARCH::VC>,
            CompType, Vector::GetUsedRegsN<CompType, Src0Type, Src1Type, OutType>(), ARCH::VC>{}, dims[IDX_4], 0);
    }
#else
    Binary::EvaluateWrapper<CompType, ARCH::VC>(lhs, rhs, dst, scale0, scale1, dims, stridesSrc0, stridesSrc1,
        stridesDst, Helpers::ApeZN<Helpers::LtNWrapper<CompType,
        Vector::GetUsedRegsN<CompType, Src0Type, Src1Type, OutType>(), ARCH::VC>,
        CompType, Vector::GetUsedRegsN<CompType, Src0Type, Src1Type, OutType>(), ARCH::VC>{}, dims[IDX_4], 0);
#endif
    }
}

template <typename CompType, typename Src0Type, typename Src1Type, typename OutType>
void EvaluateEq(const Src0Type *lhs, const Src1Type *rhs, OutType *dst,
    const float scale0, const float scale1, const Shape &dims, const Shape &stridesSrc0,
    const Shape &stridesSrc1, const Shape &stridesDst, bool isSeqKernel) noexcept
{
    if (isSeqKernel) {
        Binary::EvaluateSC<CompType>(lhs, rhs, dst, scale0, scale1, dims, stridesSrc0, stridesSrc1, stridesDst,
            [](CompType a, CompType b) -> CompType { return (a == b) ? 1 : 0; });
    } else {
#if USE_NEON
    if (IsNeonBest(stridesSrc0, stridesSrc1)) {
        SizeType neonLoop = dims[IDX_4] - (dims[IDX_4] % (NUM_2 * Vector::NeonLanes<Src0Type, Src1Type,
            OutType>()));
        SizeType tailLoop = dims[IDX_4] % (NUM_2 * Vector::NeonLanes<Src0Type, Src1Type, OutType>()); // Use VC
        Binary::EvaluateWrapper<CompType, ARCH::NEON>(lhs, rhs, dst, scale0, scale1, dims, stridesSrc0, stridesSrc1,
            stridesDst, Helpers::ApeZN<Helpers::EqNWrapper<CompType,
            Vector::GetUsedRegsN<CompType, Src0Type, Src1Type, OutType>(), ARCH::NEON>,
            CompType, Vector::GetUsedRegsN<CompType, Src0Type, Src1Type, OutType>(), ARCH::NEON>{}, neonLoop, 0);
        Binary::EvaluateWrapper<CompType, ARCH::VC>(lhs, rhs, dst, scale0, scale1, dims, stridesSrc0, stridesSrc1,
            stridesDst, Helpers::ApeZN<Helpers::EqNWrapper<CompType,
            Vector::GetUsedRegsN<CompType, Src0Type, Src1Type, OutType>(), ARCH::VC>,
            CompType, Vector::GetUsedRegsN<CompType, Src0Type, Src1Type, OutType>(), ARCH::VC>{}, tailLoop, neonLoop);
    } else {
        Binary::EvaluateWrapper<CompType, ARCH::VC>(lhs, rhs, dst, scale0, scale1, dims, stridesSrc0, stridesSrc1,
            stridesDst, Helpers::ApeZN<Helpers::EqNWrapper<CompType,
            Vector::GetUsedRegsN<CompType, Src0Type, Src1Type, OutType>(), ARCH::VC>,
            CompType, Vector::GetUsedRegsN<CompType, Src0Type, Src1Type, OutType>(), ARCH::VC>{}, dims[IDX_4], 0);
    }
#else
    Binary::EvaluateWrapper<CompType, ARCH::VC>(lhs, rhs, dst, scale0, scale1, dims, stridesSrc0, stridesSrc1,
        stridesDst, Helpers::ApeZN<Helpers::EqNWrapper<CompType,
        Vector::GetUsedRegsN<CompType, Src0Type, Src1Type, OutType>(), ARCH::VC>,
        CompType, Vector::GetUsedRegsN<CompType, Src0Type, Src1Type, OutType>(), ARCH::VC>{}, dims[IDX_4], 0);
#endif
    }
}

template <typename CompType, typename Src0Type, typename Src1Type, typename OutType>
void EvaluateNe(const Src0Type *lhs, const Src1Type *rhs, OutType *dst,
    const float scale0, const float scale1, const Shape &dims, const Shape &stridesSrc0,
    const Shape &stridesSrc1, const Shape &stridesDst, bool isSeqKernel) noexcept
{
    if (isSeqKernel) {
        Binary::EvaluateSC<CompType>(lhs, rhs, dst, scale0, scale1, dims, stridesSrc0, stridesSrc1, stridesDst,
            [](CompType a, CompType b) -> CompType { return (a != b) ? 1 : 0; });
    } else {
#if USE_NEON
    if (IsNeonBest(stridesSrc0, stridesSrc1)) {
        SizeType neonLoop = dims[IDX_4] - (dims[IDX_4] % (NUM_2 * Vector::NeonLanes<Src0Type, Src1Type,
            OutType>()));
        SizeType tailLoop = dims[IDX_4] % (NUM_2 * Vector::NeonLanes<Src0Type, Src1Type, OutType>()); // Use VC
        Binary::EvaluateWrapper<CompType, ARCH::NEON>(lhs, rhs, dst, scale0, scale1, dims, stridesSrc0, stridesSrc1,
            stridesDst, Helpers::ApeZN<Helpers::NeNWrapper<CompType,
            Vector::GetUsedRegsN<CompType, Src0Type, Src1Type, OutType>(), ARCH::NEON>,
            CompType, Vector::GetUsedRegsN<CompType, Src0Type, Src1Type, OutType>(), ARCH::NEON>{}, neonLoop, 0);
        Binary::EvaluateWrapper<CompType, ARCH::VC>(lhs, rhs, dst, scale0, scale1, dims, stridesSrc0, stridesSrc1,
            stridesDst, Helpers::ApeZN<Helpers::NeNWrapper<CompType,
            Vector::GetUsedRegsN<CompType, Src0Type, Src1Type, OutType>(), ARCH::VC>,
            CompType, Vector::GetUsedRegsN<CompType, Src0Type, Src1Type, OutType>(), ARCH::VC>{}, tailLoop, neonLoop);
    } else {
        Binary::EvaluateWrapper<CompType, ARCH::VC>(lhs, rhs, dst, scale0, scale1, dims, stridesSrc0, stridesSrc1,
            stridesDst, Helpers::ApeZN<Helpers::NeNWrapper<CompType,
            Vector::GetUsedRegsN<CompType, Src0Type, Src1Type, OutType>(), ARCH::VC>,
            CompType, Vector::GetUsedRegsN<CompType, Src0Type, Src1Type, OutType>(), ARCH::VC>{}, dims[IDX_4], 0);
    }
#else
    Binary::EvaluateWrapper<CompType, ARCH::VC>(lhs, rhs, dst, scale0, scale1, dims, stridesSrc0, stridesSrc1,
        stridesDst, Helpers::ApeZN<Helpers::NeNWrapper<CompType,
        Vector::GetUsedRegsN<CompType, Src0Type, Src1Type, OutType>(), ARCH::VC>,
        CompType, Vector::GetUsedRegsN<CompType, Src0Type, Src1Type, OutType>(), ARCH::VC>{}, dims[IDX_4], 0);
#endif
    }
}

} // Binary
} // Detail
} // KuDNN

#endif //  KUDNN_BINARY_DEF_HPP
