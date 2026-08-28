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

#include <algorithm>
#include <cassert>

#include "arm_sve.h"

#include "kudnn.hpp"

#include "internal_funcs.hpp"
#include "kudnn_jit.hpp"
#include "kudnn_wrapper_helpers.hpp"
#include "kudnn_jit_attributes.hpp"
#include "kernels/matmul_jit/kudnn_gemm_helpers.hpp"
#include "kernels/matmul_jit/kudnn_small_gemm.hpp"

namespace KuDNN {
namespace Detail {

constexpr static inline IntType ComputeIdx(IntType row, IntType col, IntType stride) noexcept
{
    return row * stride + col;
}

template <typename SrcType, typename WeiType, typename DstType, typename BiaType>
static void RefMatmul(const void *src, const void *wei, void *dst,
    const void *bia, MatMulDstFlags flags, float alpha, float beta, IntType m, IntType n, IntType k,
    IntType srcStride, IntType weiStride, IntType dstStride, IntType biaStride) noexcept
{
    const SrcType *typedSrc = static_cast<const SrcType *>(src);
    const WeiType *typedWei = static_cast<const WeiType *>(wei);
    DstType *typedDst = static_cast<DstType *>(dst);
    const BiaType *typedBia = static_cast<const BiaType *>(bia);
    for (IntType mIdx = 0; mIdx < m; ++mIdx) {
        for (IntType nIdx = 0; nIdx < n; ++nIdx) {
            DstType accumulator = 0;
            for (IntType kIdx = 0; kIdx < k; ++kIdx) {
                accumulator +=
                    typedSrc[ComputeIdx(mIdx, kIdx, srcStride)] * typedWei[ComputeIdx(kIdx, nIdx, weiStride)];
            }
            accumulator *= alpha;
            if (beta != 0.0f) {
                typedDst[ComputeIdx(mIdx, nIdx, dstStride)] =
                    accumulator + beta * typedDst[ComputeIdx(mIdx, nIdx, dstStride)];
            } else {
                typedDst[ComputeIdx(mIdx, nIdx, dstStride)] = accumulator;
            }
        }
    }
    Service::Unused(flags, beta, typedBia, biaStride);
}

template <typename SrcType, typename WeiType, typename DstType, typename BiaType>
static void RefMatmulOpt(const void *src, const void *wei, void *dst,
    const void *bia, MatMulDstFlags flags, float alpha, float beta, IntType m, IntType n, IntType k,
    IntType srcStride, IntType weiStride, IntType dstStride, IntType biaStride) noexcept
{
    const SrcType *typedSrc = static_cast<const SrcType *>(src);
    const WeiType *typedWei = static_cast<const WeiType *>(wei);
    DstType *typedDst = static_cast<DstType *>(dst);
    const BiaType *typedBia = static_cast<const BiaType *>(bia);
    constexpr IntType KERNEL_M_SIZE = NUM_4;
    constexpr IntType KERNEL_N_SIZE = NUM_4; // in VL
    constexpr IntType KERNEL_K_SIZE = NUM_4;
    const IntType vlw = svcntw();
    const IntType kernelNStep = vlw * KERNEL_N_SIZE;
    for (IntType mIdx = 0; mIdx < m; mIdx += KERNEL_M_SIZE) {
        IntType mPortion = std::min(KERNEL_M_SIZE, m - mIdx);
        for (IntType nIdx = 0; nIdx < n; nIdx += kernelNStep) {
            svbool_t weiPg0 = svwhilelt_b32(nIdx + NUM_0 * vlw, n);
            svbool_t weiPg1 = svwhilelt_b32(nIdx + NUM_1 * vlw, n);
            svbool_t weiPg2 = svwhilelt_b32(nIdx + NUM_2 * vlw, n);
            svbool_t weiPg3 = svwhilelt_b32(nIdx + NUM_3 * vlw, n);
            svfloat32_t acc00 = svdup_f32(0.0f);
            svfloat32_t acc01 = svdup_f32(0.0f);
            svfloat32_t acc02 = svdup_f32(0.0f);
            svfloat32_t acc03 = svdup_f32(0.0f);
            svfloat32_t acc10 = svdup_f32(0.0f);
            svfloat32_t acc11 = svdup_f32(0.0f);
            svfloat32_t acc12 = svdup_f32(0.0f);
            svfloat32_t acc13 = svdup_f32(0.0f);
            svfloat32_t acc20 = svdup_f32(0.0f);
            svfloat32_t acc21 = svdup_f32(0.0f);
            svfloat32_t acc22 = svdup_f32(0.0f);
            svfloat32_t acc23 = svdup_f32(0.0f);
            svfloat32_t acc30 = svdup_f32(0.0f);
            svfloat32_t acc31 = svdup_f32(0.0f);
            svfloat32_t acc32 = svdup_f32(0.0f);
            svfloat32_t acc33 = svdup_f32(0.0f);
            for (IntType kIdx = 0; kIdx < k; kIdx += KERNEL_K_SIZE) {
                IntType kPortion = std::min(KERNEL_K_SIZE, k - kIdx);
                svbool_t srcPg = svwhilelt_b32(kIdx, k);
                svfloat32_t src0 = svdup_f32(0.0f);
                if (mPortion > NUM_0) {
                    src0 = svld1rq(srcPg, typedSrc + srcStride * (mIdx + NUM_0) + kIdx);
                }
                svfloat32_t src1 = svdup_f32(0.0f);
                if (mPortion > NUM_1) {
                    src1 = svld1rq(srcPg, typedSrc + srcStride * (mIdx + NUM_1) + kIdx);
                }
                svfloat32_t src2 = svdup_f32(0.0f);
                if (mPortion > NUM_2) {
                    src2 = svld1rq(srcPg, typedSrc + srcStride * (mIdx + NUM_2) + kIdx);
                }
                svfloat32_t src3 = svdup_f32(0.0f);
                if (mPortion > NUM_3) {
                    src3 = svld1rq(srcPg, typedSrc + srcStride * (mIdx + NUM_3) + kIdx);
                }
                if (kPortion > NUM_0) {
                    svfloat32_t wei0 = svld1(weiPg0, typedWei + weiStride * (kIdx + NUM_0) + (nIdx + NUM_0 * vlw));
                    svfloat32_t wei1 = svld1(weiPg1, typedWei + weiStride * (kIdx + NUM_0) + (nIdx + NUM_1 * vlw));
                    svfloat32_t wei2 = svld1(weiPg2, typedWei + weiStride * (kIdx + NUM_0) + (nIdx + NUM_2 * vlw));
                    svfloat32_t wei3 = svld1(weiPg3, typedWei + weiStride * (kIdx + NUM_0) + (nIdx + NUM_3 * vlw));
                    acc00 = svmla_lane(acc00, wei0, src0, NUM_0);
                    acc01 = svmla_lane(acc01, wei1, src0, NUM_0);
                    acc02 = svmla_lane(acc02, wei2, src0, NUM_0);
                    acc03 = svmla_lane(acc03, wei3, src0, NUM_0);
                    acc10 = svmla_lane(acc10, wei0, src1, NUM_0);
                    acc11 = svmla_lane(acc11, wei1, src1, NUM_0);
                    acc12 = svmla_lane(acc12, wei2, src1, NUM_0);
                    acc13 = svmla_lane(acc13, wei3, src1, NUM_0);
                    acc20 = svmla_lane(acc20, wei0, src2, NUM_0);
                    acc21 = svmla_lane(acc21, wei1, src2, NUM_0);
                    acc22 = svmla_lane(acc22, wei2, src2, NUM_0);
                    acc23 = svmla_lane(acc23, wei3, src2, NUM_0);
                    acc30 = svmla_lane(acc30, wei0, src3, NUM_0);
                    acc31 = svmla_lane(acc31, wei1, src3, NUM_0);
                    acc32 = svmla_lane(acc32, wei2, src3, NUM_0);
                    acc33 = svmla_lane(acc33, wei3, src3, NUM_0);
                }
                if (kPortion > NUM_1) {
                    svfloat32_t wei0 = svld1(weiPg0, typedWei + weiStride * (kIdx + NUM_1) + (nIdx + NUM_0 * vlw));
                    svfloat32_t wei1 = svld1(weiPg1, typedWei + weiStride * (kIdx + NUM_1) + (nIdx + NUM_1 * vlw));
                    svfloat32_t wei2 = svld1(weiPg2, typedWei + weiStride * (kIdx + NUM_1) + (nIdx + NUM_2 * vlw));
                    svfloat32_t wei3 = svld1(weiPg3, typedWei + weiStride * (kIdx + NUM_1) + (nIdx + NUM_3 * vlw));
                    acc00 = svmla_lane(acc00, wei0, src0, NUM_1);
                    acc01 = svmla_lane(acc01, wei1, src0, NUM_1);
                    acc02 = svmla_lane(acc02, wei2, src0, NUM_1);
                    acc03 = svmla_lane(acc03, wei3, src0, NUM_1);
                    acc10 = svmla_lane(acc10, wei0, src1, NUM_1);
                    acc11 = svmla_lane(acc11, wei1, src1, NUM_1);
                    acc12 = svmla_lane(acc12, wei2, src1, NUM_1);
                    acc13 = svmla_lane(acc13, wei3, src1, NUM_1);
                    acc20 = svmla_lane(acc20, wei0, src2, NUM_1);
                    acc21 = svmla_lane(acc21, wei1, src2, NUM_1);
                    acc22 = svmla_lane(acc22, wei2, src2, NUM_1);
                    acc23 = svmla_lane(acc23, wei3, src2, NUM_1);
                    acc30 = svmla_lane(acc30, wei0, src3, NUM_1);
                    acc31 = svmla_lane(acc31, wei1, src3, NUM_1);
                    acc32 = svmla_lane(acc32, wei2, src3, NUM_1);
                    acc33 = svmla_lane(acc33, wei3, src3, NUM_1);
                }
                if (kPortion > NUM_2) {
                    svfloat32_t wei0 = svld1(weiPg0, typedWei + weiStride * (kIdx + NUM_2) + (nIdx + NUM_0 * vlw));
                    svfloat32_t wei1 = svld1(weiPg1, typedWei + weiStride * (kIdx + NUM_2) + (nIdx + NUM_1 * vlw));
                    svfloat32_t wei2 = svld1(weiPg2, typedWei + weiStride * (kIdx + NUM_2) + (nIdx + NUM_2 * vlw));
                    svfloat32_t wei3 = svld1(weiPg3, typedWei + weiStride * (kIdx + NUM_2) + (nIdx + NUM_3 * vlw));
                    acc00 = svmla_lane(acc00, wei0, src0, NUM_2);
                    acc01 = svmla_lane(acc01, wei1, src0, NUM_2);
                    acc02 = svmla_lane(acc02, wei2, src0, NUM_2);
                    acc03 = svmla_lane(acc03, wei3, src0, NUM_2);
                    acc10 = svmla_lane(acc10, wei0, src1, NUM_2);
                    acc11 = svmla_lane(acc11, wei1, src1, NUM_2);
                    acc12 = svmla_lane(acc12, wei2, src1, NUM_2);
                    acc13 = svmla_lane(acc13, wei3, src1, NUM_2);
                    acc20 = svmla_lane(acc20, wei0, src2, NUM_2);
                    acc21 = svmla_lane(acc21, wei1, src2, NUM_2);
                    acc22 = svmla_lane(acc22, wei2, src2, NUM_2);
                    acc23 = svmla_lane(acc23, wei3, src2, NUM_2);
                    acc30 = svmla_lane(acc30, wei0, src3, NUM_2);
                    acc31 = svmla_lane(acc31, wei1, src3, NUM_2);
                    acc32 = svmla_lane(acc32, wei2, src3, NUM_2);
                    acc33 = svmla_lane(acc33, wei3, src3, NUM_2);
                }
                if (kPortion > NUM_3) {
                    svfloat32_t wei0 = svld1(weiPg0, typedWei + weiStride * (kIdx + NUM_3) + (nIdx + NUM_0 * vlw));
                    svfloat32_t wei1 = svld1(weiPg1, typedWei + weiStride * (kIdx + NUM_3) + (nIdx + NUM_1 * vlw));
                    svfloat32_t wei2 = svld1(weiPg2, typedWei + weiStride * (kIdx + NUM_3) + (nIdx + NUM_2 * vlw));
                    svfloat32_t wei3 = svld1(weiPg3, typedWei + weiStride * (kIdx + NUM_3) + (nIdx + NUM_3 * vlw));
                    acc00 = svmla_lane(acc00, wei0, src0, NUM_3);
                    acc01 = svmla_lane(acc01, wei1, src0, NUM_3);
                    acc02 = svmla_lane(acc02, wei2, src0, NUM_3);
                    acc03 = svmla_lane(acc03, wei3, src0, NUM_3);
                    acc10 = svmla_lane(acc10, wei0, src1, NUM_3);
                    acc11 = svmla_lane(acc11, wei1, src1, NUM_3);
                    acc12 = svmla_lane(acc12, wei2, src1, NUM_3);
                    acc13 = svmla_lane(acc13, wei3, src1, NUM_3);
                    acc20 = svmla_lane(acc20, wei0, src2, NUM_3);
                    acc21 = svmla_lane(acc21, wei1, src2, NUM_3);
                    acc22 = svmla_lane(acc22, wei2, src2, NUM_3);
                    acc23 = svmla_lane(acc23, wei3, src2, NUM_3);
                    acc30 = svmla_lane(acc30, wei0, src3, NUM_3);
                    acc31 = svmla_lane(acc31, wei1, src3, NUM_3);
                    acc32 = svmla_lane(acc32, wei2, src3, NUM_3);
                    acc33 = svmla_lane(acc33, wei3, src3, NUM_3);
                }
            }
            if (alpha != 1.0f) {
                acc00 = svmulx_x(weiPg0, acc00, alpha);
                acc01 = svmulx_x(weiPg1, acc01, alpha);
                acc02 = svmulx_x(weiPg2, acc02, alpha);
                acc03 = svmulx_x(weiPg3, acc03, alpha);
                acc10 = svmulx_x(weiPg0, acc10, alpha);
                acc11 = svmulx_x(weiPg1, acc11, alpha);
                acc12 = svmulx_x(weiPg2, acc12, alpha);
                acc13 = svmulx_x(weiPg3, acc13, alpha);
                acc20 = svmulx_x(weiPg0, acc20, alpha);
                acc21 = svmulx_x(weiPg1, acc21, alpha);
                acc22 = svmulx_x(weiPg2, acc22, alpha);
                acc23 = svmulx_x(weiPg3, acc23, alpha);
                acc30 = svmulx_x(weiPg0, acc30, alpha);
                acc31 = svmulx_x(weiPg1, acc31, alpha);
                acc32 = svmulx_x(weiPg2, acc32, alpha);
                acc33 = svmulx_x(weiPg3, acc33, alpha);
            }
            if (mPortion > NUM_0) {
                svst1(weiPg0, typedDst + dstStride * (mIdx + NUM_0) + (nIdx + NUM_0 * vlw), acc00);
                svst1(weiPg1, typedDst + dstStride * (mIdx + NUM_0) + (nIdx + NUM_1 * vlw), acc01);
                svst1(weiPg2, typedDst + dstStride * (mIdx + NUM_0) + (nIdx + NUM_2 * vlw), acc02);
                svst1(weiPg3, typedDst + dstStride * (mIdx + NUM_0) + (nIdx + NUM_3 * vlw), acc03);
            }
            if (mPortion > NUM_1) {
                svst1(weiPg0, typedDst + dstStride * (mIdx + NUM_1) + (nIdx + NUM_0 * vlw), acc10);
                svst1(weiPg1, typedDst + dstStride * (mIdx + NUM_1) + (nIdx + NUM_1 * vlw), acc11);
                svst1(weiPg2, typedDst + dstStride * (mIdx + NUM_1) + (nIdx + NUM_2 * vlw), acc12);
                svst1(weiPg3, typedDst + dstStride * (mIdx + NUM_1) + (nIdx + NUM_3 * vlw), acc13);
            }
            if (mPortion > NUM_2) {
                svst1(weiPg0, typedDst + dstStride * (mIdx + NUM_2) + (nIdx + NUM_0 * vlw), acc20);
                svst1(weiPg1, typedDst + dstStride * (mIdx + NUM_2) + (nIdx + NUM_1 * vlw), acc21);
                svst1(weiPg2, typedDst + dstStride * (mIdx + NUM_2) + (nIdx + NUM_2 * vlw), acc22);
                svst1(weiPg3, typedDst + dstStride * (mIdx + NUM_2) + (nIdx + NUM_3 * vlw), acc23);
            }
            if (mPortion > NUM_3) {
                svst1(weiPg0, typedDst + dstStride * (mIdx + NUM_3) + (nIdx + NUM_0 * vlw), acc30);
                svst1(weiPg1, typedDst + dstStride * (mIdx + NUM_3) + (nIdx + NUM_1 * vlw), acc31);
                svst1(weiPg2, typedDst + dstStride * (mIdx + NUM_3) + (nIdx + NUM_2 * vlw), acc32);
                svst1(weiPg3, typedDst + dstStride * (mIdx + NUM_3) + (nIdx + NUM_3 * vlw), acc33);
            }
        }
    }
    Service::Unused(flags, beta, typedBia, biaStride);
}

template <typename SrcType, typename WeiType, typename DstType, typename BiaType>
static void RefMatmulM1(const void *src, const void *wei, void *dst,
    const void *bia, MatMulDstFlags flags, float alpha, float beta, IntType m, IntType n, IntType k,
    IntType srcStride, IntType weiStride, IntType dstStride, IntType biaStride) noexcept
{
    const SrcType *typedSrc = static_cast<const SrcType *>(src);
    const WeiType *typedWei = static_cast<const WeiType *>(wei);
    DstType *typedDst = static_cast<DstType *>(dst);
    const BiaType *typedBia = static_cast<const BiaType *>(bia);
    constexpr IntType KERNEL_N_SIZE = NUM_4; // in VL
    constexpr IntType KERNEL_K_SIZE = NUM_4;
    const IntType vlw = svcntw();
    const IntType kernelNStep = vlw * KERNEL_N_SIZE;
    for (IntType nIdx = 0; nIdx < n; nIdx += kernelNStep) {
        svbool_t weiPg0 = svwhilelt_b32(nIdx + NUM_0 * vlw, n);
        svbool_t weiPg1 = svwhilelt_b32(nIdx + NUM_1 * vlw, n);
        svbool_t weiPg2 = svwhilelt_b32(nIdx + NUM_2 * vlw, n);
        svbool_t weiPg3 = svwhilelt_b32(nIdx + NUM_3 * vlw, n);
        svfloat32_t acc00 = svdup_f32(0.0f);
        svfloat32_t acc01 = svdup_f32(0.0f);
        svfloat32_t acc02 = svdup_f32(0.0f);
        svfloat32_t acc03 = svdup_f32(0.0f);
        for (IntType kIdx = 0; kIdx < k; kIdx += KERNEL_K_SIZE) {
            IntType kPortion = std::min(KERNEL_K_SIZE, k - kIdx);
            svbool_t srcPg = svwhilelt_b32(kIdx, k);
            svfloat32_t src0 = svld1rq(srcPg, typedSrc + kIdx);
            if (kPortion > NUM_0) {
                svfloat32_t wei0 = svld1(weiPg0, typedWei + weiStride * (kIdx + NUM_0) + (nIdx + NUM_0 * vlw));
                svfloat32_t wei1 = svld1(weiPg1, typedWei + weiStride * (kIdx + NUM_0) + (nIdx + NUM_1 * vlw));
                svfloat32_t wei2 = svld1(weiPg2, typedWei + weiStride * (kIdx + NUM_0) + (nIdx + NUM_2 * vlw));
                svfloat32_t wei3 = svld1(weiPg3, typedWei + weiStride * (kIdx + NUM_0) + (nIdx + NUM_3 * vlw));
                acc00 = svmla_lane(acc00, wei0, src0, NUM_0);
                acc01 = svmla_lane(acc01, wei1, src0, NUM_0);
                acc02 = svmla_lane(acc02, wei2, src0, NUM_0);
                acc03 = svmla_lane(acc03, wei3, src0, NUM_0);
            }
            if (kPortion > NUM_1) {
                svfloat32_t wei0 = svld1(weiPg0, typedWei + weiStride * (kIdx + NUM_1) + (nIdx + NUM_0 * vlw));
                svfloat32_t wei1 = svld1(weiPg1, typedWei + weiStride * (kIdx + NUM_1) + (nIdx + NUM_1 * vlw));
                svfloat32_t wei2 = svld1(weiPg2, typedWei + weiStride * (kIdx + NUM_1) + (nIdx + NUM_2 * vlw));
                svfloat32_t wei3 = svld1(weiPg3, typedWei + weiStride * (kIdx + NUM_1) + (nIdx + NUM_3 * vlw));
                acc00 = svmla_lane(acc00, wei0, src0, NUM_1);
                acc01 = svmla_lane(acc01, wei1, src0, NUM_1);
                acc02 = svmla_lane(acc02, wei2, src0, NUM_1);
                acc03 = svmla_lane(acc03, wei3, src0, NUM_1);
            }
            if (kPortion > NUM_2) {
                svfloat32_t wei0 = svld1(weiPg0, typedWei + weiStride * (kIdx + NUM_2) + (nIdx + NUM_0 * vlw));
                svfloat32_t wei1 = svld1(weiPg1, typedWei + weiStride * (kIdx + NUM_2) + (nIdx + NUM_1 * vlw));
                svfloat32_t wei2 = svld1(weiPg2, typedWei + weiStride * (kIdx + NUM_2) + (nIdx + NUM_2 * vlw));
                svfloat32_t wei3 = svld1(weiPg3, typedWei + weiStride * (kIdx + NUM_2) + (nIdx + NUM_3 * vlw));
                acc00 = svmla_lane(acc00, wei0, src0, NUM_2);
                acc01 = svmla_lane(acc01, wei1, src0, NUM_2);
                acc02 = svmla_lane(acc02, wei2, src0, NUM_2);
                acc03 = svmla_lane(acc03, wei3, src0, NUM_2);
            }
            if (kPortion > NUM_3) {
                svfloat32_t wei0 = svld1(weiPg0, typedWei + weiStride * (kIdx + NUM_3) + (nIdx + NUM_0 * vlw));
                svfloat32_t wei1 = svld1(weiPg1, typedWei + weiStride * (kIdx + NUM_3) + (nIdx + NUM_1 * vlw));
                svfloat32_t wei2 = svld1(weiPg2, typedWei + weiStride * (kIdx + NUM_3) + (nIdx + NUM_2 * vlw));
                svfloat32_t wei3 = svld1(weiPg3, typedWei + weiStride * (kIdx + NUM_3) + (nIdx + NUM_3 * vlw));
                acc00 = svmla_lane(acc00, wei0, src0, NUM_3);
                acc01 = svmla_lane(acc01, wei1, src0, NUM_3);
                acc02 = svmla_lane(acc02, wei2, src0, NUM_3);
                acc03 = svmla_lane(acc03, wei3, src0, NUM_3);
            }
        }
        if (alpha != 1.0f) {
            acc00 = svmulx_x(weiPg0, acc00, alpha);
            acc01 = svmulx_x(weiPg1, acc01, alpha);
            acc02 = svmulx_x(weiPg2, acc02, alpha);
            acc03 = svmulx_x(weiPg3, acc03, alpha);
        }
            svst1(weiPg0, typedDst + (nIdx + NUM_0 * vlw), acc00);
            svst1(weiPg1, typedDst + (nIdx + NUM_1 * vlw), acc01);
            svst1(weiPg2, typedDst + (nIdx + NUM_2 * vlw), acc02);
            svst1(weiPg3, typedDst + (nIdx + NUM_3 * vlw), acc03);
    }
    Service::Unused(flags, beta, typedBia, biaStride, m, srcStride, dstStride);
}

template <typename SrcType, typename WeiType, typename DstType, typename BiaType>
static void RefMatmulN1VL(const void *src, const void *wei, void *dst,
    const void *bia, MatMulDstFlags flags, float alpha, float beta, IntType m, IntType n, IntType k,
    IntType srcStride, IntType weiStride, IntType dstStride, IntType biaStride) noexcept
{
    const SrcType *typedSrc = static_cast<const SrcType *>(src);
    const WeiType *typedWei = static_cast<const WeiType *>(wei);
    DstType *typedDst = static_cast<DstType *>(dst);
    const BiaType *typedBia = static_cast<const BiaType *>(bia);
    constexpr IntType KERNEL_M_SIZE = NUM_4;
    constexpr IntType KERNEL_K_SIZE = NUM_4;
    for (IntType mIdx = 0; mIdx < m; mIdx += KERNEL_M_SIZE) {
        IntType mPortion = std::min(KERNEL_M_SIZE, m - mIdx);
        svbool_t weiPg0 = svwhilelt_b32(NUM_0, n);
        svfloat32_t acc00 = svdup_f32(0.0f);
        svfloat32_t acc10 = svdup_f32(0.0f);
        svfloat32_t acc20 = svdup_f32(0.0f);
        svfloat32_t acc30 = svdup_f32(0.0f);
        for (IntType kIdx = 0; kIdx < k; kIdx += KERNEL_K_SIZE) {
            IntType kPortion = std::min(KERNEL_K_SIZE, k - kIdx);
            svbool_t srcPg = svwhilelt_b32(kIdx, k);
            svfloat32_t src0 = svdup_f32(0.0f);
            if (mPortion > NUM_0) {
                src0 = svld1rq(srcPg, typedSrc + srcStride * (mIdx + NUM_0) + kIdx);
            }
            svfloat32_t src1 = svdup_f32(0.0f);
            if (mPortion > NUM_1) {
                src1 = svld1rq(srcPg, typedSrc + srcStride * (mIdx + NUM_1) + kIdx);
            }
            svfloat32_t src2 = svdup_f32(0.0f);
            if (mPortion > NUM_2) {
                src2 = svld1rq(srcPg, typedSrc + srcStride * (mIdx + NUM_2) + kIdx);
            }
            svfloat32_t src3 = svdup_f32(0.0f);
            if (mPortion > NUM_3) {
                src3 = svld1rq(srcPg, typedSrc + srcStride * (mIdx + NUM_3) + kIdx);
            }
            if (kPortion > NUM_0) {
                svfloat32_t wei0 = svld1(weiPg0, typedWei + weiStride * (kIdx + NUM_0));
                acc00 = svmla_lane(acc00, wei0, src0, NUM_0);
                acc10 = svmla_lane(acc10, wei0, src1, NUM_0);
                acc20 = svmla_lane(acc20, wei0, src2, NUM_0);
                acc30 = svmla_lane(acc30, wei0, src3, NUM_0);
            }
            if (kPortion > NUM_1) {
                svfloat32_t wei0 = svld1(weiPg0, typedWei + weiStride * (kIdx + NUM_1));
                acc00 = svmla_lane(acc00, wei0, src0, NUM_1);
                acc10 = svmla_lane(acc10, wei0, src1, NUM_1);
                acc20 = svmla_lane(acc20, wei0, src2, NUM_1);
                acc30 = svmla_lane(acc30, wei0, src3, NUM_1);
            }
            if (kPortion > NUM_2) {
                svfloat32_t wei0 = svld1(weiPg0, typedWei + weiStride * (kIdx + NUM_2));
                acc00 = svmla_lane(acc00, wei0, src0, NUM_2);
                acc10 = svmla_lane(acc10, wei0, src1, NUM_2);
                acc20 = svmla_lane(acc20, wei0, src2, NUM_2);
                acc30 = svmla_lane(acc30, wei0, src3, NUM_2);
            }
            if (kPortion > NUM_3) {
                svfloat32_t wei0 = svld1(weiPg0, typedWei + weiStride * (kIdx + NUM_3));
                acc00 = svmla_lane(acc00, wei0, src0, NUM_3);
                acc10 = svmla_lane(acc10, wei0, src1, NUM_3);
                acc20 = svmla_lane(acc20, wei0, src2, NUM_3);
                acc30 = svmla_lane(acc30, wei0, src3, NUM_3);
            }
        }
        if (alpha != 1.0f) {
            acc00 = svmulx_x(weiPg0, acc00, alpha);
            acc10 = svmulx_x(weiPg0, acc10, alpha);
            acc20 = svmulx_x(weiPg0, acc20, alpha);
            acc30 = svmulx_x(weiPg0, acc30, alpha);
        }
        if (mPortion > NUM_0) {
            svst1(weiPg0, typedDst + dstStride * (mIdx + NUM_0), acc00);
        }
        if (mPortion > NUM_1) {
            svst1(weiPg0, typedDst + dstStride * (mIdx + NUM_1), acc10);
        }
        if (mPortion > NUM_2) {
            svst1(weiPg0, typedDst + dstStride * (mIdx + NUM_2), acc20);
        }
        if (mPortion > NUM_3) {
            svst1(weiPg0, typedDst + dstStride * (mIdx + NUM_3), acc30);
        }
    }
    Service::Unused(flags, beta, typedBia, biaStride);
}

struct SmallGEMMRefImpl final : SmallGEMMImplBase {
    SmallGEMMRefImpl(const GemmInfo &gInfo, IntType nThrs) : SmallGEMMImplBase(gInfo, nThrs) {}

    SmallGEMMParams GetSmallGEMMParams() const noexcept override
    {
        IntType numDims = gemmInfo.dstInfo.GetNumDims();
        MatMulDstFlags flags = MatMulDstFlags::FILL_C_WITH_ZEROZ;
        IntType m = gemmInfo.srcInfo.GetDims()[numDims - NUM_2];
        IntType n = gemmInfo.weiInfo.GetDims()[numDims - NUM_1];
        IntType k = gemmInfo.srcInfo.GetDims()[numDims - NUM_1];
        IntType srcStride = gemmInfo.srcInfo.GetStrides()[numDims - NUM_2];
        IntType weiStride = gemmInfo.weiInfo.GetStrides()[numDims - NUM_2];
        IntType dstStride = gemmInfo.dstInfo.GetStrides()[numDims - NUM_2];
        // gemmInfo.biaInfo.GetStrides()[numDims - NUM_2];
        IntType biaStride = 0;
        SmallGEMMParams ret{static_cast<SmallGEMMKernelFuncPtr>(nullptr), flags,
            m, n, k, srcStride, weiStride, dstStride, biaStride};
        if (m * n * k < NUM_512) {
            ret.kernelPtr = RefMatmul<float, float, float, float>;
        } else if (m == NUM_1) {
            ret.kernelPtr = RefMatmulM1<float, float, float, float>;
        } else if (n < static_cast<IntType>(svcntw())) {
            ret.kernelPtr = RefMatmulN1VL<float, float, float, float>;
        } else {
            ret.kernelPtr = RefMatmulOpt<float, float, float, float>;
        }
        return ret;
    }
};

struct SmallGEMMNotApplicableImpl final : SmallGEMMImplBase {
    SmallGEMMNotApplicableImpl(const GemmInfo &gInfo, IntType nThrs) : SmallGEMMImplBase(gInfo, nThrs) {}

    SmallGEMMParams GetSmallGEMMParams() const noexcept override
    {
        return {};
    }
};

struct SmallGEMMF32RefGenerator final : SmallGEMMGenerator {
    bool Supports(const GemmInfo &gemmInfo) const noexcept override
    {
        IntType numDims = gemmInfo.dstInfo.GetNumDims();
        if (numDims > NUM_2) {
            return false;
        }
        if (gemmInfo.srcInfo.GetType() != Element::TypeT::F32) {
            return false;
        }
        if (gemmInfo.weiInfo.GetType() != Element::TypeT::F32) {
            return false;
        }
        if (gemmInfo.dstInfo.GetType() != Element::TypeT::F32) {
            return false;
        }
        bool hasBias = gemmInfo.biaInfo.GetTotalTensorSize() != 0;
        if (hasBias && (gemmInfo.biaInfo.GetType() != Element::TypeT::F32)) {
            return false;
        }
        // will be supported in JIT kernel and condition will be removed
        if (hasBias) {
            return false;
        }
        // will be supported in JIT kernel and condition will be removed
        if (gemmInfo.attributes.GetPostOps().Len() > 0) {
            return false;
        }
        IntType m = gemmInfo.srcInfo.GetDims()[numDims - NUM_2];
        IntType n = gemmInfo.weiInfo.GetDims()[numDims - NUM_1];
        IntType k = gemmInfo.srcInfo.GetDims()[numDims - NUM_1];
        if (m * n * k > 4500) { // 4500 is a constant which enables small matrix opts
            return false;
        }
        // will be supported in JIT kernel and condition will be removed
        if (gemmInfo.srcInfo.GetLayout() != Layout::AB) {
            return false;
        }
        // will be supported in JIT kernel and condition will be removed
        if (gemmInfo.weiInfo.GetLayout() != Layout::AB) {
            return false;
        }
        // will be supported in JIT kernel and condition will be removed
        if (gemmInfo.dstInfo.GetLayout() != Layout::AB) {
            return false;
        }
        return true;
    }
    std::chrono::duration<float, std::milli> EstimateTime(const GemmInfo &gemmInfo,
        IntType nThreads) const noexcept override
    {
        Service::Unused(gemmInfo, nThreads);
        return std::chrono::duration<float, std::milli>{1.0};
    }
    std::shared_ptr<SmallGEMMImplBase> GetImpl(const GemmInfo &gemmInfo, IntType nThreads) const override
    {
        return std::make_shared<SmallGEMMRefImpl>(gemmInfo, nThreads);
    }
};

struct SmallGEMMNotApplicable final : SmallGEMMGenerator {
    bool Supports(const GemmInfo &gemmInfo) const noexcept override
    {
        Service::Unused(gemmInfo);
        return true;
    }
    std::chrono::duration<float, std::milli> EstimateTime(const GemmInfo &gemmInfo,
        IntType nThreads) const noexcept override
    {
        Service::Unused(gemmInfo, nThreads);
        return std::chrono::duration<float, std::milli>{std::numeric_limits<float>::max() / NUM_2};
    }
    std::shared_ptr<SmallGEMMImplBase> GetImpl(const GemmInfo &gemmInfo, IntType nThreads) const override
    {
        return std::make_shared<SmallGEMMNotApplicableImpl>(gemmInfo, nThreads);
    }
};

static const std::vector<std::shared_ptr<SmallGEMMGenerator>> g_SmallGEMMKinds = {
    std::make_shared<SmallGEMMF32RefGenerator>(),
    std::make_shared<SmallGEMMNotApplicable>()
};

static SmallTaskSolutionHandler g_smallGemmSolutionHandler;

const std::vector<std::shared_ptr<SmallGEMMGenerator>>& GetSmallGEMMGeneratorList() noexcept
{
    VERBOSE_PRINT;
    return g_SmallGEMMKinds;
}

SmallGEMMParams FindSmallGemmKernel(const GemmInfo &gemmInfo, IntType nThreads) noexcept(false)
{
    return g_smallGemmSolutionHandler.Find(gemmInfo, nThreads);
}

} // Detail
} // KuDNN