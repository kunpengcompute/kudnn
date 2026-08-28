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
#ifndef KUDNN_MATMUL_JIT_NEON_GEMV_IMPL_F32F32F32F32_HPP
#define KUDNN_MATMUL_JIT_NEON_GEMV_IMPL_F32F32F32F32_HPP

#include <memory>
#include <vector>
#include <array>
#include <type_traits>

#include "kudnn.hpp"

#include "cg_codegen.hpp"

#include "vector_wrappers.hpp"
#include "kudnn_jit.hpp"
#include "kernels/matmul_jit/kudnn_gemm_jit.hpp"
#include "kernels/matmul_jit/kudnn_gemm_helpers.hpp"
#include "kernels/matmul_jit/kudnn_gemm_threading_helpers.hpp"
#include "kernels/reorder/sve/bf16any32/kudnn_fast_jit_impl_bfp16any32_bfp16any32.hpp"
#include "kudnn_wrapper_helpers.hpp"
#include "internal_funcs.hpp"

namespace KuDNN {
namespace Detail {

template <>
struct MatMulKernelSkinnyABCodeGen<ARCH::NEON, float, float, float, float,
    MatMulImplKinds::F32F32F32F32_NEON> :
    MatMulKernelCodeGenBase<ARCH::NEON, float, float, float, float>
{
public:
    constexpr static const std::uint32_t KERNEL_M_SIZE = NUM_4;     // how much load from A
    constexpr static const std::uint32_t KERNEL_N_SIZE = NUM_1;
    constexpr static const std::uint32_t K_LOOP_UNROLL = NUM_4;     // in VL

    constexpr static const IntType NEON_VL = 4;
    IntType weiOffsetValue;

    // Kernel performs 4x4VL update of destination matrix via
    // outer product computations inside loops over M, K. Registers v1-v23 are used for
    // accumulation (then will be stored to destination matrix),
    // registerrs v1 - v4 are used to store matrix B values (reused to store 4VL),
    // registers v5 - v20 are used to store matrix A values.
    // The kernel as unroll by K variable ([m x k] * [k * n] -> [m * n]) by 4.

    MatMulKernelSkinnyABCodeGen(IntType m, IntType n, IntType k,
        IntType srcOffsetArg,
        IntType weiOffsetArg,
        IntType dstOffsetArg,
        IntType biaOffsetArg, const Attributes &attributesArg, bool hasAlphaArg);

protected:
    void InitializeDstRegisters(std::uint32_t mLim);

    void GenerateMLoopBody(std::uint32_t mLim, std::uint32_t k);

    void GenerateKLoopBody(std::uint32_t mLim, std::uint32_t k);

    void ApplyEltwise(const PostOps &postOps, SizeType poIdx, const std::vector<SizeType> &regIDs,
        Element::TypeT regType) override;

    static std::uint32_t GetWeiVRegIdx(std::uint32_t k) noexcept
    {
        return WEI_V_REGS_OFFSET + k % WEI_V_REGS_NUM;
    }
    static std::uint32_t GetSrcVRegIdx(std::uint32_t k, std::uint32_t m) noexcept
    {
        return SRC_V_REGS_OFFSET + (k + m * K_LOOP_UNROLL) % SRC_V_REGS_NUM;
    }
    static std::uint32_t GetDstVRegIdx(std::uint32_t m) noexcept
    {
        return DST_V_REGS_OFFSET + m % DST_V_REGS_NUM;
    }
    // pointer registers
    static std::uint32_t GetWeiXRegIdx(std::uint32_t k) noexcept
    {
        return WEI_V_REGS_OFFSET + k % WEI_V_REGS_NUM;
    }
    static std::uint32_t GetSrcXRegIdx(std::uint32_t m) noexcept
    {
        return SRC_X_REGS_OFFSET + m % SRC_V_REGS_NUM;
    }

    // V registers
    constexpr static const std::uint32_t WEI_V_REGS_NUM = K_LOOP_UNROLL; // wei V regs = v2-v5
    constexpr static const std::uint32_t WEI_V_REGS_OFFSET = NUM_2;      // wei V regs = v2-v5
    constexpr static const std::uint32_t SRC_V_REGS_NUM = KERNEL_M_SIZE * K_LOOP_UNROLL;         // src V regs = v6-v21
    constexpr static const std::uint32_t SRC_V_REGS_OFFSET = WEI_V_REGS_OFFSET + WEI_V_REGS_NUM; // src V regs = v6-v21
    constexpr static const std::uint32_t DST_V_REGS_NUM = KERNEL_M_SIZE; // dst V regs = v22-v26
    constexpr static const std::uint32_t DST_V_REGS_OFFSET = SRC_V_REGS_OFFSET +
        SRC_V_REGS_NUM; // dst V regs = v22-v26
    // X Registers x17-x26
    // Src - M Kernel registers
    // Wei & Dst - 1 register on each
    constexpr static const std::uint32_t SRC_X_REGS_NUM = KERNEL_M_SIZE;    // max 8
    constexpr static const std::uint32_t SRC_X_REGS_OFFSET = 17;
    constexpr static const std::uint32_t WEI_X_REGS_NUM = 1;
    constexpr static const std::uint32_t DST_X_REGS_NUM = 1;

    bool hasAlpha;

    // function input arguments
    ARMCG::XReg srcPtr = x0;
    ARMCG::XReg weiPtr = x1;
    ARMCG::XReg dstPtrLoad = x2;
    ARMCG::XReg biaPtr = x3;
    ARMCG::XReg flags = x4;     // for dst bias load
    ARMCG::SReg beta = s0;
    ARMCG::SReg alpha = s1;
    // other used registers
    ARMCG::XReg mVal = x5;
    ARMCG::XReg nVal = x6;
    ARMCG::XReg kVal = x7;
    ARMCG::XReg srcOffset = x8;
    ARMCG::XReg weiOffset = x9;
    ARMCG::XReg dstOffset = x10;
    ARMCG::XReg biaOffset = x11;
    ARMCG::XReg tmpMMinusMLoopCtr = x12;
    ARMCG::XReg tmp = x13;
    // the gap should be aligned with src, wei and dst load registers (x17-x26)
    ARMCG::XReg kLoopCtr = x14;
    ARMCG::XReg mLoopCtr = x15;

    ARMCG::XReg tmpWeiPtr = x27;        // tmp wei ptr
    ARMCG::XReg dstPtrStore = x28;        // tmp dst ptr
};

template <>
struct MatMulImplGenerator<MatMulKernelSkinnyABCodeGen<ARCH::NEON, float, float, float, float,
    MatMulImplKinds::F32F32F32F32_NEON>>:
    MatmulTypeHelper<float, float, float, float>, ImplGenerator<GemmInfo> {
    using CodeGen = MatMulKernelSkinnyABCodeGen<ARCH::NEON, float, float, float, float,
        MatMulImplKinds::F32F32F32F32_NEON>;
    struct DefaultImpl : MatMulImplBase {
        DefaultImpl(const GemmInfo &task, IntType nThreads);

        constexpr static std::uint32_t GetKernelMSize() noexcept;

        static std::uint32_t GetKernelNSize() noexcept;

        IntType GetMBlockSize() const noexcept override;

        IntType GetNBlockSize() const noexcept override;

        IntType GetKBlockSize() const noexcept override;

    protected:
        void InitializeCodeGens(IntType m, IntType n, IntType k,
            SizeType offsetSrc, SizeType offsetWei, SizeType offsetDst, SizeType offsetBia,
            bool wereSrcWeiSwapped, IntType mFull, IntType nFull, const Attributes &attributes) override;

        void InitializeSingleCodegen(IntType id, IntType m, IntType n, IntType k,
            IntType currentM, IntType currentN, IntType currentK,
            SizeType offsetSrc, SizeType offsetWei, SizeType offsetDst, SizeType offsetBia,
            bool wereSrcWeiSwapped, const Attributes &attributes, bool hasAlpha) override;

        void InitializeThreadsCountImpl() override;

        IntType GetThreadBufSize(IntType m, IntType n, IntType k,
            MatrixKind mKind) const noexcept override;

        void ReserveImplSpecificParams() override;

        std::vector<std::pair<Shape, Shape>> SplitWork() override;

        void RunSequential2D(IntType threadID, const void *src, const void *wei, void *dst, const void *bia,
            std::unique_ptr<PackStrategy> &srcPackStrategy, std::unique_ptr<PackStrategy> &weiPackStrategy,
            float alpha, float beta, const PostOpsDataPtrs &shiftedByBatchPOPtrs) const noexcept(false) override;

        void UpdateExecParams() noexcept override;

        // in case of M, N, K impl. depends on order before kernel call, M * K ~= L3 Cache per Thread (920B = 1.75 kB)
        constexpr static IntType M_BLOCK_SIZE = 1200; // M % KERNEL_M_SIZE == 0
        constexpr static IntType K_BLOCK_SIZE = 640;  // K % VL == 0
        constexpr static IntType N_BLOCK_SIZE = 1;  // N % (4 * VL) == 0

        std::array<CodeGenWrapper<CodeGen>, NUM_12> matmulCodegenKernels;
    };
    bool Supports(const GemmInfo &task) const noexcept override
    {
        VERBOSE_PRINT;
        if (((task.dstInfo.GetDims()[task.dstInfo.GetNumDims() - IDX_1] == NUM_1)
                && (IsLayoutAlikeAB(task.srcInfo.GetLayout()))) ||
            (((task.dstInfo.GetDims()[task.dstInfo.GetNumDims() - IDX_2] == NUM_1))
                && (IsLayoutAlikeBA(task.weiInfo.GetLayout())))) {
            return true;
        } else {
            return false;
        }
    }
    std::chrono::duration<float, std::milli> EstimateTime(const GemmInfo &task,
                                                          IntType nThreads) const noexcept override
    {
        VERBOSE_PRINT;
        // force use GEMV impl if it's supported
        float singleThreadPerformance = NUM_96;
        return CalculateMatmulTimeWithReorder<float, float, float, float>(
            singleThreadPerformance, task, nThreads);
    }
    std::shared_ptr<ImplBase<GemmInfo>> GetImpl(const GemmInfo &task, IntType nThreads) const override
    {
        return std::make_shared<DefaultImpl>(task, nThreads);
    }
};
} // Detail
} // KuDNN

#endif // KUDNN_MATMUL_JIT_NEON_GEMV_IMPL_F32F32F32F32_HPP
