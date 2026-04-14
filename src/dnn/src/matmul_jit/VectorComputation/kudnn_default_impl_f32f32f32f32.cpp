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

#include <memory>
#include <vector>
#include <array>
#include <algorithm>

#include <arm_sve.h>

#include "kudnn.hpp"

#include "cg_codegen.hpp"

#include "kudnn_jit.hpp"
#include "matmul_jit/kudnn_gemm_jit.hpp"
#include "matmul_jit/kudnn_gemm_helpers.hpp"
#include "matmul_jit/kudnn_gemm_threading_helpers.hpp"
#include "kudnn_wrapper_helpers.hpp"
#include "kernels/reorder/kudnn_reorder_def.hpp"
#include "matmul_jit/VectorComputation/kudnn_default_impl_f32f32f32f32.hpp"

namespace KuDNN {
namespace Detail {

MatMulKernelDefaultCodeGenImpl<ARCH::VectorComputation, float, float, float, float, MatmulDummyType>::MatMulKernelDefaultCodeGenImpl(
    IntType m, IntType n, IntType k,
    IntType srcOffsetArg,
    IntType weiOffsetArg,
    IntType dstOffsetArg,
    IntType biaOffsetArg)
{
    if ((m >= 0) && (n >= 0) && (k >= 0)) {
        Preamble();

        Mov(mVal, m);
        Mov(nVal, n);
        Mov(kVal, k);
        Mov(srcOffset, srcOffsetArg);
        Mov(weiOffset, weiOffsetArg);
        Mov(dstOffset, dstOffsetArg);
        Mov(biaOffset, biaOffsetArg);
        Ptrue(ARMCG::PRegS(5), ARMCG::Pattern::ALL); // p5 is true for this impl

        std::uint32_t mResid = m % KERNEL_M_SIZE;
        if (mResid == 0) {
            mResid = KERNEL_M_SIZE;
        }
        Mov(mLoopCtr, mVal);
        Sub(tmpMMinusMLoopCtr, mVal, mLoopCtr);
        if (m > KERNEL_M_SIZE) {
            ARMCG::Label mLoopHead;
            L(mLoopHead);
            {
                GenerateMLoopBody(KERNEL_M_SIZE, n, k);
                Sub(mLoopCtr, mLoopCtr, KERNEL_M_SIZE);
                Cmp(mLoopCtr, KERNEL_M_SIZE);
                Sub(tmpMMinusMLoopCtr, mVal, mLoopCtr);
            }
            B(ARMCG::Cond::GT, mLoopHead);
        }
        GenerateMLoopBody(mResid, n, k);

        Postamble();
    }
    Ret();
}

void MatMulKernelDefaultCodeGenImpl<ARCH::VectorComputation, float, float, float, float, MatmulDummyType>::InitializeDstRegisters(
    std::uint32_t mLim, std::uint32_t nLim)
{
    // can be reworked to reuse some code (at least nested loops)
    ARMCG::Label loadC;
    ARMCG::Label mullC;
    ARMCG::Label loadBias1;
    ARMCG::Label loadBiasM;
    ARMCG::Label loadBiasN;
    ARMCG::Label loadBiasMN;
    ARMCG::Label addBias1;
    ARMCG::Label addBiasM;
    ARMCG::Label addBiasN;
    ARMCG::Label addBiasMN;
    ARMCG::Label initDone;
    Tbz(flags, 0, loadBias1); // 0-bit correspond to FILL_C_WITH_ZEROZ
    // Fill with zeros by default
    for (std::uint32_t mIdx = 0; mIdx < mLim; ++mIdx) {
        for (std::uint32_t nIdx = 0; nIdx < nLim; ++nIdx) {
            Mov(ARMCG::ZRegB(GetDstZRegIdx(mIdx, nIdx)), 0);
        }
    }
    L(loadBias1);
    Tbz(flags, 3, loadBiasN); // 3-bit correspond to LOAD_BIAS_1
    Ldr(w26, ARMCG::AdrUimm(Ptr(biaPtr)));
    Cpy(ARMCG::ZRegS(7), p5, w26); // z7 is used as tmp register here
    for (std::uint32_t mIdx = 0; mIdx < mLim; ++mIdx) {
        for (std::uint32_t nIdx = 0; nIdx < nLim; ++nIdx) {
            Mov(ARMCG::ZRegD(GetDstZRegIdx(mIdx, nIdx)),
                ARMCG::ZRegD(7)); // z7 contains bias value
        }
    }
    L(loadBiasN);
    Tbz(flags, 5, loadBiasM); // 5-bit correspond to LOAD_BIAS_N
    Add(tmp, biaPtr, tmpNMinusNLoopCtr, ARMCG::ShMod::LSL, Math::Log2(sizeof(BiaType)));
    for (std::uint32_t nIdx = 0; nIdx < nLim; ++nIdx) {
        // z8-z11 registers are chosen for tmp values, could be any but dst Z registers
        Ld1w(ARMCG::ZRegS(8 + nIdx), ARMCG::PReg(GetLoadStorePRegIdx(nIdx)),
            Ptr(tmp, nIdx, ARMCG::ExtMod::MUL_VL));
    }
    for (std::uint32_t mIdx = 1; mIdx < mLim; ++mIdx) {
        for (std::uint32_t nIdx = 0; nIdx < nLim; ++nIdx) {
            Mov(ARMCG::ZRegD(GetDstZRegIdx(mIdx, nIdx)),
                ARMCG::ZRegD(8 + nIdx)); // z8-z11 are tmp registers used above
        }
    }
    L(loadBiasM);
    Tbz(flags, 4, loadBiasMN); // 4-bit correspond to LOAD_BIAS_M
    Add(tmp, biaPtr, tmpMMinusMLoopCtr, ARMCG::ShMod::LSL, Math::Log2(sizeof(BiaType)));
    for (std::uint32_t mIdx = 0; mIdx < mLim; ++mIdx) {
        Ldr(ARMCG::WReg(GetSrcXRegIdx(mIdx)), Ptr(tmp, static_cast<std::uint32_t>(sizeof(BiaType)) * mIdx));
        Cpy(ARMCG::ZRegS(GetSrcZRegIdx(mIdx)), p5,
            ARMCG::WReg(GetSrcXRegIdx(mIdx)));
    }
    for (std::uint32_t mIdx = 0; mIdx < mLim; ++mIdx) {
        for (std::uint32_t nIdx = 0; nIdx < nLim; ++nIdx) {
            Mov(ARMCG::ZRegD(GetDstZRegIdx(mIdx, nIdx)),
                ARMCG::ZRegD(GetSrcZRegIdx(mIdx)));
        }
    }
    L(loadBiasMN);
    Tbz(flags, 6, loadC); // 6-bit correspond to LOAD_BIAS_MN
    Mul(tmp, tmpMMinusMLoopCtr, biaOffset);
    Add(ARMCG::XReg(GetSrcXRegIdx(0)), biaPtr, tmp,
        ARMCG::ShMod::LSL, Math::Log2(sizeof(BiaType)));
    Add(ARMCG::XReg(GetSrcXRegIdx(0)), ARMCG::XReg(GetSrcXRegIdx(0)),
        tmpNMinusNLoopCtr, ARMCG::ShMod::LSL, Math::Log2(sizeof(BiaType)));
    for (std::uint32_t mIdx = 1; mIdx < mLim; ++mIdx) {
        Add(ARMCG::XReg(GetSrcXRegIdx(mIdx)), ARMCG::XReg(GetSrcXRegIdx(mIdx - 1)),
            biaOffset, ARMCG::ShMod::LSL, Math::Log2(sizeof(BiaType)));
    }
    for (std::uint32_t mIdx = 0; mIdx < mLim; ++mIdx) {
        for (std::uint32_t nIdx = 0; nIdx < nLim; ++nIdx) {
            Ld1w(ARMCG::ZRegS(GetDstZRegIdx(mIdx, nIdx)),
                ARMCG::PReg(GetLoadStorePRegIdx(nIdx)),
                Ptr(ARMCG::XReg(GetSrcXRegIdx(mIdx)), nIdx,
                    ARMCG::ExtMod::MUL_VL));
        }
    }
    L(loadC);
    Tbz(flags, 1, mullC); // 1-bit correspond to LOAD_C
    Mul(tmp, tmpMMinusMLoopCtr, dstOffset);
    Add(ARMCG::XReg(GetSrcXRegIdx(0)), dstPtr, tmp,
        ARMCG::ShMod::LSL, Math::Log2(sizeof(DstType)));
    Add(ARMCG::XReg(GetSrcXRegIdx(0)), ARMCG::XReg(GetSrcXRegIdx(0)),
        tmpNMinusNLoopCtr, ARMCG::ShMod::LSL, Math::Log2(sizeof(DstType)));
    for (std::uint32_t mIdx = 1; mIdx < mLim; ++mIdx) {
        Add(ARMCG::XReg(GetSrcXRegIdx(mIdx)),
            ARMCG::XReg(GetSrcXRegIdx(mIdx - 1)),
            dstOffset, ARMCG::ShMod::LSL, Math::Log2(sizeof(DstType)));
    }
    for (std::uint32_t mIdx = 0; mIdx < mLim; ++mIdx) {
        for (std::uint32_t nIdx = 0; nIdx < nLim; ++nIdx) {
            Ld1w(ARMCG::ZRegS(GetDstZRegIdx(mIdx, nIdx)),
                ARMCG::PReg(GetLoadStorePRegIdx(nIdx)),
                Ptr(ARMCG::XReg(GetSrcXRegIdx(mIdx)), nIdx,
                ARMCG::ExtMod::MUL_VL));
        }
    }
    L(mullC);
    Tbz(flags, 2, addBias1); // 2-bit correspond to MULL_C
    Mov(ARMCG::ZRegS(7), beta); // z7 is used as tmp register here
    for (std::uint32_t mIdx = 0; mIdx < mLim; ++mIdx) {
        for (std::uint32_t nIdx = 0; nIdx < nLim; ++nIdx) {
            Fmul(ARMCG::ZRegS(GetDstZRegIdx(mIdx, nIdx)),
                ARMCG::ZRegS(GetDstZRegIdx(mIdx, nIdx)),
                ARMCG::ZRegS(7)); // z7 contains beta value
        }
    }
    L(addBias1);
    Tbz(flags, 7, addBiasN); // 7-bit correspond to Add_BIAS_1
    Ldr(w26, ARMCG::AdrUimm(Ptr(biaPtr)));
    Cpy(ARMCG::ZRegS(7), p5, w26); // z7 is used as tmp register here
    for (std::uint32_t mIdx = 0; mIdx < mLim; ++mIdx) {
        for (std::uint32_t nIdx = 0; nIdx < nLim; ++nIdx) {
            Fadd(ARMCG::ZRegS(GetDstZRegIdx(mIdx, nIdx)),
                ARMCG::ZRegS(GetDstZRegIdx(mIdx, nIdx)),
                ARMCG::ZRegS(7)); // z7 contains bias value
        }
    }
    L(addBiasN);
    Tbz(flags, 9, addBiasM); // 9-bit correspond to Add_BIAS_N
    Add(tmp, biaPtr, tmpNMinusNLoopCtr, ARMCG::ShMod::LSL, Math::Log2(sizeof(BiaType)));
    for (std::uint32_t nIdx = 0; nIdx < nLim; ++nIdx) {
        // z4-z10 registers are chosen for tmp values, could be any but dst Z registers
        Ld1w(ARMCG::ZRegS(4 + nIdx), ARMCG::PReg(GetLoadStorePRegIdx(nIdx)),
            Ptr(tmp, nIdx, ARMCG::ExtMod::MUL_VL));
    }
    for (std::uint32_t mIdx = 0; mIdx < mLim; ++mIdx) {
        for (std::uint32_t nIdx = 0; nIdx < nLim; ++nIdx) {
            Fadd(ARMCG::ZRegS(GetDstZRegIdx(mIdx, nIdx)),
                ARMCG::ZRegS(GetDstZRegIdx(mIdx, nIdx)),
                ARMCG::ZRegS(4 + nIdx)); // z4-z10 registers are chosen for tmp values
        }
    }
    L(addBiasM);
    Tbz(flags, 8, addBiasMN); // 8-bit correspond to LOAD_BIAS_M
    Add(tmp, biaPtr, tmpMMinusMLoopCtr, ARMCG::ShMod::LSL, Math::Log2(sizeof(BiaType)));
    for (std::uint32_t mIdx = 0; mIdx < mLim; ++mIdx) {
        Ldr(ARMCG::WReg(GetSrcXRegIdx(mIdx)), Ptr(tmp, static_cast<std::uint32_t>(sizeof(BiaType)) * mIdx));
        Cpy(ARMCG::ZRegS(GetSrcZRegIdx(mIdx)), p5,
            ARMCG::WReg(GetSrcXRegIdx(mIdx)));
    }
    for (std::uint32_t mIdx = 0; mIdx < mLim; ++mIdx) {
        for (std::uint32_t nIdx = 0; nIdx < nLim; ++nIdx) {
            Fadd(ARMCG::ZRegS(GetDstZRegIdx(mIdx, nIdx)),
                ARMCG::ZRegS(GetDstZRegIdx(mIdx, nIdx)),
                ARMCG::ZRegS(GetSrcZRegIdx(mIdx)));
        }
    }
    L(addBiasMN);
    Tbz(flags, 10, initDone); // 10-bit correspond to LOAD_BIAS_MN
    Mul(tmp, tmpMMinusMLoopCtr, biaOffset);
    Add(ARMCG::XReg(GetSrcXRegIdx(0)), biaPtr, tmp,
        ARMCG::ShMod::LSL, Math::Log2(sizeof(BiaType)));
    Add(ARMCG::XReg(GetSrcXRegIdx(0)),
        ARMCG::XReg(GetSrcXRegIdx(0)),
        tmpNMinusNLoopCtr, ARMCG::ShMod::LSL, Math::Log2(sizeof(BiaType)));
    for (std::uint32_t mIdx = 1; mIdx < mLim; ++mIdx) {
        Add(ARMCG::XReg(GetSrcXRegIdx(mIdx)),
            ARMCG::XReg(GetSrcXRegIdx(mIdx - 1)),
            biaOffset, ARMCG::ShMod::LSL, Math::Log2(sizeof(BiaType)));
    }
    for (std::uint32_t mIdx = 0; mIdx < mLim; ++mIdx) {
        for (std::uint32_t nIdx = 0; nIdx < nLim; ++nIdx) {
            // z4-z10 registers are chosen for tmp values, could be any except dst registers
            Ld1w(ARMCG::ZRegS(4 + nIdx),
                ARMCG::PRegS(GetLoadStorePRegIdx(nIdx)),
                Ptr(ARMCG::XReg(GetSrcXRegIdx(mIdx)), nIdx,
                    ARMCG::ExtMod::MUL_VL));
            Fadd(ARMCG::ZRegS(GetDstZRegIdx(mIdx, nIdx)),
                ARMCG::ZRegS(GetDstZRegIdx(mIdx, nIdx)),
                ARMCG::ZRegS(4 + nIdx)); // z4-z10 registers are chosen for tmp values
        }
    }
    L(initDone);
}

void MatMulKernelDefaultCodeGenImpl<ARCH::VectorComputation, float, float, float, float, MatmulDummyType>::GenerateMLoopBody(
    std::uint32_t mLim, std::uint32_t nLim, std::uint32_t k)
{
    std::int64_t vectorLength = svcntw();
    std::uint32_t nResid = Math::CeilDiv(nLim % (KERNEL_N_SIZE * vectorLength), vectorLength);
    if (nResid == 0) {
        nResid = KERNEL_N_SIZE;
    }
    Mov(nLoopCtr, nVal);
    Sub(tmpNMinusNLoopCtr, nVal, nLoopCtr);
    if (nLim > KERNEL_N_SIZE * vectorLength) {
        ARMCG::Label nLoopHead;
        L(nLoopHead);
        {
            GenerateNLoopBody(mLim, KERNEL_N_SIZE, k);
            Sub(nLoopCtr, nLoopCtr, KERNEL_N_SIZE * vectorLength);
            Cmp(nLoopCtr, KERNEL_N_SIZE * vectorLength);
            Sub(tmpNMinusNLoopCtr, nVal, nLoopCtr);
        }
        B(ARMCG::Cond::GT, nLoopHead);
    }
    GenerateNLoopBody(mLim, nResid, k);
}

void MatMulKernelDefaultCodeGenImpl<ARCH::VectorComputation, float, float, float, float, MatmulDummyType>::GenerateNLoopBody(
    std::uint32_t mLim, std::uint32_t nLim, std::uint32_t k)
{
    std::uint32_t kResid = k % K_LOOP_UNROLL;
    if (kResid == 0) {
        kResid = K_LOOP_UNROLL;
    }
    Mov(tmp4VL, 0);
    for (std::uint32_t nIdx = 0; nIdx < nLim; ++nIdx) {
        Whilelt(ARMCG::PRegS(GetLoadStorePRegIdx(nIdx)), tmp4VL, nLoopCtr);
        Incw(tmp4VL);
    }
    InitializeDstRegisters(mLim, nLim);
    // Initialize src pointers
    Mul(tmp, tmpMMinusMLoopCtr, srcOffset);
    Add(ARMCG::XReg(GetSrcXRegIdx(0)), srcPtr, tmp,
        ARMCG::ShMod::LSL, Math::Log2(sizeof(SrcType)));
    for (std::uint32_t mIdx = 1; mIdx < mLim; ++mIdx) {
        Add(ARMCG::XReg(GetSrcXRegIdx(mIdx)),
            ARMCG::XReg(GetSrcXRegIdx(mIdx - 1)),
            srcOffset, ARMCG::ShMod::LSL, Math::Log2(sizeof(SrcType)));
    }
    // Initialize wei pointers
    Mul(tmp, tmpNMinusNLoopCtr, kVal);
    Add(ARMCG::XReg(GetWeiXRegIdx(0, nLim)), weiPtr, tmp,
        ARMCG::ShMod::LSL, Math::Log2(sizeof(WeiType)));
    for (std::uint32_t nIdx = 1; nIdx < nLim; ++nIdx) {
        Add(ARMCG::XReg(GetWeiXRegIdx(nIdx, nLim)),
            ARMCG::XReg(GetWeiXRegIdx(nIdx - 1, nLim)),
            weiOffset, ARMCG::ShMod::LSL, Math::Log2(sizeof(WeiType)));
    }
    // Main k loop
    Mov(kLoopCtr, kVal);
    if (k > K_LOOP_UNROLL) {
        ARMCG::Label kLoopHead;
        L(kLoopHead);
        {
            Whilelt(ARMCG::PRegS(0), xzr, kLoopCtr);
            // Preload wei registers z0, z1
            Ld1w(ARMCG::ZRegS(GetWeiZRegIdx(0)), p5,
                Ptr(ARMCG::XReg(GetWeiXRegIdx(0, nLim))));
            if (nLim > 1) {
                Ld1w(ARMCG::ZRegS(GetWeiZRegIdx(1)), p5,
                    Ptr(ARMCG::XReg(GetWeiXRegIdx(1, nLim))));
            }
            Sub(kLoopCtr, kLoopCtr, K_LOOP_UNROLL);
            Cmp(kLoopCtr, K_LOOP_UNROLL);
            // Preload src registers
            for (std::uint32_t mIdx = 0; mIdx < mLim; ++mIdx) {
                Ld1rqw(ARMCG::ZRegS(GetSrcZRegIdx(mIdx)), ARMCG::PRegS(0),
                    Ptr(ARMCG::XReg(GetSrcXRegIdx(mIdx))));
                Add(ARMCG::XReg(GetSrcXRegIdx(mIdx)),
                    ARMCG::XReg(GetSrcXRegIdx(mIdx)),
                    NUM_4 * sizeof(SrcType));
            }
            GenerateMLAPart(mLim, nLim, K_LOOP_UNROLL, false);
        }
        B(ARMCG::Cond::GT, kLoopHead);
    }
    // Process k tail
    Whilelt(ARMCG::PRegS(0), xzr, kLoopCtr);
    // preload wei registers z0, z1
    Ld1w(ARMCG::ZRegS(GetWeiZRegIdx(0)), p5,
        Ptr(ARMCG::XReg(GetWeiXRegIdx(0, nLim))));
    if (nLim > 1) {
        Ld1w(ARMCG::ZRegS(GetWeiZRegIdx(1)), p5,
            Ptr(ARMCG::XReg(GetWeiXRegIdx(1, nLim))));
    }
    // Preload src registers
    for (std::uint32_t mIdx = 0; mIdx < mLim; ++mIdx) {
        Ld1rqw(ARMCG::ZRegS(GetSrcZRegIdx(mIdx)), ARMCG::PRegS(0),
            Ptr(ARMCG::XReg(GetSrcXRegIdx(mIdx))));
    }
    GenerateMLAPart(mLim, nLim, kResid, true);

    // Initialize dst pointers
    Mul(tmp, tmpMMinusMLoopCtr, dstOffset);
    // Row-major format of dst
    Add(ARMCG::XReg(GetDstXRegIdx(0)), dstPtr, tmp,
        ARMCG::ShMod::LSL, Math::Log2(sizeof(DstType))); // Add stride for M
    Add(ARMCG::XReg(GetDstXRegIdx(0)),
        ARMCG::XReg(GetDstXRegIdx(0)),
        tmpNMinusNLoopCtr, ARMCG::ShMod::LSL, Math::Log2(sizeof(DstType))); // Add stride for N
    for (std::uint32_t mIdx = 1; mIdx < mLim; ++mIdx) {
        Add(ARMCG::XReg(GetDstXRegIdx(mIdx)),
            ARMCG::XReg(GetDstXRegIdx(mIdx - 1)),
            dstOffset, ARMCG::ShMod::LSL, Math::Log2(sizeof(DstType)));
    }
    // Store data to dst
    for (std::uint32_t mIdx = 0; mIdx < mLim; ++mIdx) {
        for (std::uint32_t nIdx = 0; nIdx < nLim; ++nIdx) {
            St1w(ARMCG::ZRegS(GetDstZRegIdx(mIdx, nIdx)),
                ARMCG::PRegS(GetLoadStorePRegIdx(nIdx)),
                Ptr(ARMCG::XReg(GetDstXRegIdx(mIdx)), nIdx,
                    ARMCG::ExtMod::MUL_VL));
        }
    }
}

void MatMulKernelDefaultCodeGenImpl<ARCH::VectorComputation, float, float, float, float, MatmulDummyType>::GenerateMLAPart(
    std::uint32_t mLim, std::uint32_t nLim, std::uint32_t kLim, bool isTail)
{
    for (std::uint32_t kIdx = 0; kIdx < kLim; ++kIdx) {
        // src (z2-z7) and weights (z0, z1) are preloaded
        for (std::uint32_t nIdx = 0; nIdx < nLim; ++nIdx) {
            for (std::uint32_t mIdx = 0; mIdx < mLim; ++mIdx) {
                Fmla(ARMCG::ZRegS(GetDstZRegIdx(mIdx, nIdx)),
                    ARMCG::ZRegS(GetWeiZRegIdx(nIdx)),
                    ARMCG::ZRegS(GetSrcZRegIdx(mIdx))[kIdx]);
            }
            std::uint32_t tmpN = (nLim == 1) ? 1 : 2;
            std::uint32_t kLoadOffset = kIdx + (nIdx + tmpN) / nLim; // +2 since 2 regs were already preloaded
            if (kLoadOffset != kLim) {
                // load wei reg right after usage
                Ld1w(ARMCG::ZRegS(GetWeiZRegIdx(nIdx)), p5,
                    Ptr(ARMCG::XReg(GetWeiXRegIdx(nIdx + tmpN, nLim)),
                        kLoadOffset, ARMCG::ExtMod::MUL_VL));
            }
            if ((kIdx == (kLim - 1)) && !isTail) {
                Addvl(ARMCG::XReg(GetWeiXRegIdx(nIdx, nLim)),
                    ARMCG::XReg(GetWeiXRegIdx(nIdx, nLim)), KERNEL_N_SIZE);
            }
        }
    }
}

using SveFloatGen = MatMulKernelDefaultCodeGen<ARCH::VectorComputation, float, float, float, float>;
MatMulImplGenerator<SveFloatGen>::DefaultImpl::DefaultImpl(
    const GemmInfo &task, IntType nThreads) : MatMulImplBase(task, nThreads)
{
    VERBOSE_PRINT;
    UpdateExecParams();
    Initialize();
}

constexpr std::uint32_t MatMulImplGenerator<SveFloatGen>::DefaultImpl::GetKernelMSize() noexcept
{
    return CodeGen::KERNEL_M_SIZE;
}

std::uint32_t MatMulImplGenerator<SveFloatGen>::DefaultImpl::GetKernelNSize() noexcept
{
    return CodeGen::KERNEL_N_SIZE * svcntw();
}

void MatMulImplGenerator<SveFloatGen>::DefaultImpl::InitializeCodeGens(
    IntType m, IntType n, IntType k, IntType vl,
    SizeType offsetSrc, SizeType offsetWei, SizeType offsetDst, SizeType offsetBia, bool wereSrcWeiSwapped)
{
    Service::Unused(offsetWei, wereSrcWeiSwapped);
    
    IntType mBlockExecFull =  M_BLOCK_SIZE;
    IntType mBlockExecResid = m % M_BLOCK_SIZE;
    IntType nBlockExecFull = N_BLOCK_SIZE;
    IntType nBlockExecResid = n % nBlockExecFull;
    IntType kBlockExecFull = K_BLOCK_SIZE;
    IntType kBlockExecResid = k % kBlockExecFull;
    CodeGenWrapper<CodeGen> cgw0;
    cgw0.Initialize(mBlockExecFull, nBlockExecFull, kBlockExecFull, static_cast<IntType>(offsetSrc),
        kBlockExecFull * vl, static_cast<IntType>(offsetDst), static_cast<IntType>(offsetBia));
    CodeGenWrapper<CodeGen> cgw1;
    cgw1.Initialize(mBlockExecFull, nBlockExecFull, kBlockExecResid, static_cast<IntType>(offsetSrc),
        kBlockExecResid * vl, static_cast<IntType>(offsetDst), static_cast<IntType>(offsetBia));
    CodeGenWrapper<CodeGen> cgw2;
    cgw2.Initialize(mBlockExecFull, nBlockExecResid, kBlockExecFull, static_cast<IntType>(offsetSrc),
        kBlockExecFull * vl, static_cast<IntType>(offsetDst), static_cast<IntType>(offsetBia));
    CodeGenWrapper<CodeGen> cgw3;
    cgw3.Initialize(mBlockExecFull, nBlockExecResid, kBlockExecResid, static_cast<IntType>(offsetSrc),
        kBlockExecResid * vl, static_cast<IntType>(offsetDst), static_cast<IntType>(offsetBia));

    CodeGenWrapper<CodeGen> cgw4;
    cgw4.Initialize(mBlockExecResid, nBlockExecFull, kBlockExecFull,
        static_cast<IntType>(offsetSrc), kBlockExecFull * vl, static_cast<IntType>(offsetDst),
        static_cast<IntType>(offsetBia));
    CodeGenWrapper<CodeGen> cgw5;
    cgw5.Initialize(mBlockExecResid, nBlockExecFull, kBlockExecResid,
        static_cast<IntType>(offsetSrc), kBlockExecResid * vl, static_cast<IntType>(offsetDst),
        static_cast<IntType>(offsetBia));
    CodeGenWrapper<CodeGen> cgw6;
    cgw6.Initialize(mBlockExecResid, nBlockExecResid, kBlockExecFull,
        static_cast<IntType>(offsetSrc), kBlockExecFull * vl, static_cast<IntType>(offsetDst),
        static_cast<IntType>(offsetBia));
    CodeGenWrapper<CodeGen> cgw7;
    cgw7.Initialize(mBlockExecResid, nBlockExecResid, kBlockExecResid,
        static_cast<IntType>(offsetSrc), kBlockExecResid * vl, static_cast<IntType>(offsetDst),
        static_cast<IntType>(offsetBia));
    
    perThreadImpls.push_back({std::move(cgw0), std::move(cgw1), std::move(cgw2), std::move(cgw3),
                                    std::move(cgw4), std::move(cgw5), std::move(cgw6), std::move(cgw7)});

    CodeGenWrapper<PackBRowDefaultSchemeCodegenF32> cgwPackBrowBNAlpha;
    cgwPackBrowBNAlpha.Initialize(nBlockExecFull, true);
    CodeGenWrapper<PackBRowDefaultSchemeCodegenF32> cgwPackBrowBNnoAlpha;
    cgwPackBrowBNnoAlpha.Initialize(nBlockExecFull, false);
    CodeGenWrapper<PackBRowDefaultSchemeCodegenF32> cgwPackBrowRNAlpha;
    cgwPackBrowRNAlpha.Initialize(nBlockExecResid, true);
    CodeGenWrapper<PackBRowDefaultSchemeCodegenF32> cgwPackBrowRNnoAlpha;
    cgwPackBrowRNnoAlpha.Initialize(nBlockExecResid, false);
    vectorPackBRow.push_back({std::move(cgwPackBrowBNAlpha), std::move(cgwPackBrowBNnoAlpha),
        std::move(cgwPackBrowRNAlpha), std::move(cgwPackBrowRNnoAlpha)});
}

IntType MatMulImplGenerator<SveFloatGen>::DefaultImpl::GetThreadBufSize(
    IntType m, IntType n, IntType k, IntType vl) noexcept
{
    Service::Unused(m);
    return static_cast<IntType>(sizeof(WeiType) * std::min<IntType>(K_BLOCK_SIZE, k) *
        Math::CeilDiv(std::min<IntType>(N_BLOCK_SIZE, n), CodeGen::KERNEL_N_SIZE * vl) * CodeGen::KERNEL_N_SIZE * vl);
}

void MatMulImplGenerator<SveFloatGen>::DefaultImpl::ReserveImplSpecificParams()
{
    perThreadImpls.reserve(gemmWorkSize);
    vectorPackBRow.reserve(gemmWorkSize);
}

std::vector<std::pair<Shape, Shape>> MatMulImplGenerator<SveFloatGen>::DefaultImpl::SplitWork()
{
    return CalculateWorkOnThread<DefaultImpl>(taskExec.dstInfo.GetDims(), implNThreads,
        dstLayout2D == Layout::BA, taskExec.srcInfo.GetDims()[taskExec.srcInfo.GetNumDims() - 1] > 256); // 256
        // is a number indicating that k is big enough to omit condition mWork % KERNEL_M_SIZE == 0 (same for n)
}

template <bool AlphaMul>
void MatMulImplGenerator<SveFloatGen>::DefaultImpl::PackBCol(
    IntType n, IntType k, const WeiType *bPtr, IntType bOffset, WeiType *bBuffer, float alpha)
{
    const IntType vl = svcntw();
    IntType idxOffset = 0;
    if constexpr (AlphaMul) {
        idxOffset = vl * vl; // TRANSPOSE_REORDER_FUNC_ARR values with alpha and w/o alpha
    }
    for (IntType nIdx = n; nIdx > 0; nIdx -= vl) {
        for (IntType kIdx = k; kIdx > 0; kIdx -= vl) {
            IntType idxN = std::min(vl, nIdx) - 1;
            IntType idxK = std::min(vl, kIdx) - 1;
            Reorder::TRANSPOSE_REORDER_FUNC_ARR[idxN * vl + idxK + idxOffset](
                bPtr + (n - nIdx) * bOffset + (k - kIdx),
                bBuffer + (k - kIdx) * vl + k * (n - nIdx), bOffset * sizeof(float),
                vl * sizeof(float), alpha);
        }
    }
}

void MatMulImplGenerator<SveFloatGen>::DefaultImpl::RunSequential2D(
    IntType threadId, const void *src, const void *wei, void *dst,
    const void *bia, void *buffer, float alpha, float beta) const noexcept(false)
{
    VERBOSE_PRINT;
    
    for (IntType mBlock = 0; mBlock < perThreadParams[threadId].m; mBlock += M_BLOCK_SIZE) {
        auto mBlockExec = std::min(M_BLOCK_SIZE,  perThreadParams[threadId].m - mBlock);
        IntType mIdx = (mBlockExec == M_BLOCK_SIZE) ? 0 : 1;
        for (IntType kBlock = 0; kBlock < perThreadParams[threadId].k; kBlock += K_BLOCK_SIZE) {
            auto kBlockExec = std::min(K_BLOCK_SIZE, perThreadParams[threadId].k - kBlock);
            int kIdx = (kBlockExec == K_BLOCK_SIZE) ? 0 : 1;
            for (IntType nBlock = 0; nBlock < perThreadParams[threadId].n; nBlock += N_BLOCK_SIZE) {
                auto nBlockExec = std::min(N_BLOCK_SIZE, perThreadParams[threadId].n - nBlock);
                int nIdx = (nBlockExec == N_BLOCK_SIZE) ? 0 : 1;
                DstType *dstExec = static_cast<DstType *>(dst) + nBlock +
                    mBlock * perThreadParams[threadId].dstOffset2D;
                const BiaType *biaExec = static_cast<const BiaType *>(bia) + nBlock * biaNOffset +
                    mBlock * perThreadParams[threadId].biaOffset2D;
                if (perThreadParams[threadId].weiLayout2D == Layout::AB) {
                    const WeiType *weiExec = static_cast<const WeiType *>(wei) +
                        nBlock + kBlock * perThreadParams[threadId].weiOffset2D;
                    if (alpha != 1) {
                        if (nBlockExec == N_BLOCK_SIZE) {
                            auto packB = vectorPackBRow[threadId][0].template GetCode<void(*)(
                                IntType, const WeiType *, IntType, const WeiType *, float)>();
                            packB(kBlockExec, weiExec, perThreadParams[threadId].weiOffset2D,
                                static_cast<WeiType *>(buffer), alpha);
                        } else {
                            auto packB = vectorPackBRow[threadId][2].template GetCode<void(*)(
                                IntType, const WeiType *, IntType, const WeiType *, float)>();
                            packB(kBlockExec, weiExec, perThreadParams[threadId].weiOffset2D,
                                static_cast<WeiType *>(buffer), alpha);
                        }
                    } else {
                        if (nBlockExec == N_BLOCK_SIZE) {
                            auto packB = vectorPackBRow[threadId][1].template GetCode<void(*)(
                                IntType, const WeiType *, IntType, const WeiType *)>();
                            packB(kBlockExec, weiExec, perThreadParams[threadId].weiOffset2D,
                                static_cast<WeiType *>(buffer));
                        } else {
                            auto packB = vectorPackBRow[threadId][3].template GetCode<void(*)(
                                IntType, const WeiType *, IntType, const WeiType *)>();
                            packB(kBlockExec, weiExec, perThreadParams[threadId].weiOffset2D,
                                static_cast<WeiType *>(buffer));
                        }
                    }
                } else {
                    const WeiType *weiExec = static_cast<const WeiType *>(wei) +
                        nBlock * perThreadParams[threadId].weiOffset2D + kBlock;
                    if (alpha != 1) {
                        PackBCol<true>(nBlockExec, kBlockExec, weiExec,
                            perThreadParams[threadId].weiOffset2D,
                            static_cast<WeiType *>(buffer), alpha);
                    } else {
                        PackBCol<false>(nBlockExec, kBlockExec, weiExec,
                            perThreadParams[threadId].weiOffset2D,
                            static_cast<WeiType *>(buffer));
                    }
                }
                const SrcType *srcExec = static_cast<const SrcType *>(src) + kBlock +
                    mBlock * perThreadParams[threadId].srcOffset2D;
                DstLoadFlags biaFlagsExec = (kBlock || (beta != 0.0f)) ? DstLoadFlags::LOAD_C : biaFlags;
                if ((beta != 1.0f) && (beta != 0.0f)) {
                    biaFlagsExec = biaFlagsExec | DstLoadFlags::MULL_C;
                }

                auto kernel = perThreadImpls[threadId][mIdx * 4 + nIdx * 2 + kIdx].template GetCode<void (*)(
                    const SrcType *, const WeiType *, DstType *, const BiaType *, DstLoadFlags, float)>();
                kernel(srcExec, static_cast<WeiType *>(buffer), dstExec, biaExec, biaFlagsExec, beta);
            }
        }
    }
}

void MatMulImplGenerator<SveFloatGen>::DefaultImpl::UpdateExecParams() noexcept
{
    VERBOSE_PRINT;
    Layout desiredSrcLayout = implTask.srcInfo.GetLayout();
    Layout desiredWeiLayout = implTask.weiInfo.GetLayout();
    Layout desiredDstLayout = implTask.dstInfo.GetLayout();
    Layout desiredBiaLayout = implTask.biaInfo.GetLayout();
    {
        if (IsLayoutAlikeAB(desiredDstLayout)) {
            if (!IsLayoutAlikeAB(desiredSrcLayout)) {
                desiredSrcLayout = GetPlaneLayoutAB(desiredSrcLayout);
            }
            if (!IsLayoutAlikeAB(desiredWeiLayout) && !IsLayoutAlikeBA(desiredWeiLayout)) {
                desiredWeiLayout = GetPlaneLayoutAB(desiredWeiLayout);
            }
        } else if (IsLayoutAlikeBA(desiredDstLayout)) {
            if (!IsLayoutAlikeBA(desiredWeiLayout)) {
                desiredWeiLayout = GetPlaneLayoutBA(desiredWeiLayout);
            }
            if (!IsLayoutAlikeAB(desiredSrcLayout) && !IsLayoutAlikeBA(desiredSrcLayout)) {
                desiredSrcLayout = GetPlaneLayoutAB(desiredSrcLayout);
            }
        } else {
            if (IsLayoutAlikeAB(desiredSrcLayout)) {
                desiredDstLayout = GetPlaneLayoutAB(desiredDstLayout);
                if (!IsLayoutAlikeAB(desiredWeiLayout) && !IsLayoutAlikeBA(desiredWeiLayout)) {
                    desiredWeiLayout = GetPlaneLayoutAB(desiredWeiLayout);
                }
            } else if (IsLayoutAlikeBA(desiredWeiLayout)) {
                desiredDstLayout = GetPlaneLayoutBA(desiredDstLayout);
                if (!IsLayoutAlikeAB(desiredSrcLayout) && !IsLayoutAlikeBA(desiredSrcLayout)) {
                    desiredSrcLayout = GetPlaneLayoutBA(desiredSrcLayout);
                }
            } else {
                desiredDstLayout = GetPlaneLayoutAB(desiredDstLayout);
                desiredSrcLayout = GetPlaneLayoutAB(desiredSrcLayout);
                if (!IsLayoutAlikeAB(desiredWeiLayout) && !IsLayoutAlikeBA(desiredWeiLayout)) {
                    desiredWeiLayout = GetPlaneLayoutAB(desiredWeiLayout);
                }
            }
        }
    }
    if (!IsLayoutAlikeAB(desiredBiaLayout) && !IsLayoutAlikeBA(desiredBiaLayout)) {
        if (IsLayoutAlikeAB(desiredDstLayout)) {
            desiredBiaLayout = GetPlaneLayoutAB(desiredBiaLayout);
        } else {
            desiredBiaLayout = GetPlaneLayoutBA(desiredBiaLayout);
        }
    }
    InitializeTaskExec(desiredSrcLayout, desiredWeiLayout, desiredDstLayout, desiredBiaLayout,
        GetSrcType(), GetWeiType(), GetDstType(),
        (implTask.biaInfo.GetTotalTensorSize() != 0) ? GetBiaType() :
            static_cast<Element::TypeT>(implTask.biaInfo.GetType()));
}

} // Detail
} // KuDNN
