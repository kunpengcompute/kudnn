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

#include "kernels/matmul_jit/sve/kudnn_default_impl_bf16bf16f32f32.hpp"
#include "kernels/matmul_jit/kudnn_gemm_threading_helpers.hpp"

#include <arm_sve.h>
#include <arm_neon.h>

namespace KuDNN {
namespace Detail {

using VCBF16GenBase = MatMulKernelCodeGenBase<ARCH::VC, __bf16, __bf16, float, float>;

VCBF16Gen::MatMulKernelDefaultCodeGen(
    IntType m, IntType n, IntType k,
    const Attributes &attributesArg) : VCBF16GenBase(attributesArg),
    vectorLengthByte(svcntb())
{
    if ((m > 0) && (n > 0) && (k > 0)) {
        if (this->attributes.GetPostOps().Len() > 0) {
            // set reserved dst registers
            for (std::uint32_t mIdx = 0; mIdx < KERNEL_M_SIZE; ++mIdx) {
                for (std::uint32_t nIdx = 0; nIdx < KERNEL_N_SIZE; ++nIdx) {
                    TakeVectorRegister(GetDstZRegIdx(mIdx, nIdx));
                }
            }
            // set reserved predicate register
            TakePredicateRegister(NUM_5);
            
            // set reserved scalar registers
            for (std::uint32_t idx = 0; idx < NUM_27; ++idx) {
                TakeScalarRegister(idx);
            }
        }
        Ldp(srcOffset, weiOffset, PostPtr(sp, NUM_16));
        Ldp(dstOffset, biaOffset, PostPtr(sp, NUM_0));
        Sub(sp, sp, NUM_16);
        Preamble();

        Sub(sp, sp, NUM_16);
        Str(alpha, ARMCG::AdrUimm(Ptr(sp))); // alpha will be overwritten while accumulation
        Sub(sp, sp, NUM_16);
        Str(beta, ARMCG::AdrUimm(Ptr(sp))); // beta will be overwritten while accumulation

        Ptrue(ARMCG::PRegH(NUM_5), ARMCG::Pattern::ALL); // p5 is true (half-word) for this impl

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

        Add(sp, sp, NUM_32);

        Postamble();
    }
    Ret();
}

void VCBF16Gen::StoreToDst(std::uint32_t mLim, std::uint32_t nLim)
{
    // Initialize dst matrix registers
    ARMCG::Label bf16VCfillZeros;
    ARMCG::Label bf16VCloadC;
    ARMCG::Label bf16VCmullC;
    ARMCG::Label bf16VCloadBias1;
    ARMCG::Label bf16VCloadBiasM;
    ARMCG::Label bf16VCloadBiasN;
    ARMCG::Label bf16VCloadBiasMN;
    ARMCG::Label bf16VCaddBias1;
    ARMCG::Label bf16VCaddBiasM;
    ARMCG::Label bf16VCaddBiasN;
    ARMCG::Label bf16VCaddBiasMN;
    ARMCG::Label bf16VCinitDone;
    ARMCG::Label skipPO;

    Ld1r(ARMCG::ZRegS(NUM_7), p5, Ptr(sp, static_cast<std::int32_t>(NUM_16)));
    Mov(tmpW, ARMCG::VReg4S(NUM_7)[0]);
    Mov(ARMCG::WReg(GetTmpXRegIdx(0)), 0x3f800000); // 0x3f800000 is a bit representation of 1.0f
    Cmp(tmp, ARMCG::XReg(GetTmpXRegIdx(0)));
    Beq(bf16VCfillZeros);
    for (std::uint32_t mIdx = 0; mIdx < mLim; ++mIdx) {
        for (std::uint32_t nIdx = 0; nIdx < nLim; ++nIdx) {
            Fmul(ARMCG::ZRegS(GetDstZRegIdx(mIdx, nIdx)), ARMCG::ZRegS(GetDstZRegIdx(mIdx, nIdx)),
                ARMCG::ZRegS(NUM_7));
        }
    }
    L(bf16VCfillZeros);
    Tbz(flags, 0, bf16VCloadBias1); // 0-bit correspond to FILL_C_WITH_ZEROZ
    for (std::uint32_t mIdx = 0; mIdx < mLim; ++mIdx) {
        for (std::uint32_t nIdx = 0; nIdx < nLim; ++nIdx) {
            Faddv(ARMCG::SReg(GetDstZRegIdx(mIdx, nIdx)), p5, ARMCG::ZRegS(GetDstZRegIdx(mIdx, nIdx)));
        }
    }
    L(bf16VCloadBias1);
    Tbz(flags, 3, bf16VCloadBiasN); // 3-bit correspond to LOAD_BIAS_1
    Ldr(ARMCG::SReg(0), ARMCG::AdrUimm(ARMCG::Ptr(biaPtr)));
    for (std::uint32_t mIdx = 0; mIdx < mLim; ++mIdx) {
        for (std::uint32_t nIdx = 0; nIdx < nLim; ++nIdx) {
            Faddv(ARMCG::SReg(GetDstZRegIdx(mIdx, nIdx)), p5, ARMCG::ZRegS(GetDstZRegIdx(mIdx, nIdx)));
            Fadd(ARMCG::SReg(GetDstZRegIdx(mIdx, nIdx)), ARMCG::SReg(GetDstZRegIdx(mIdx, nIdx)),
                ARMCG::SReg(0));
        }
    }
    L(bf16VCloadBiasN);
    Tbz(flags, 5, bf16VCloadBiasM); // 5-bit correspond to LOAD_BIAS_N
    for (std::uint32_t mIdx = 0; mIdx < mLim; ++mIdx) {
        for (std::uint32_t nIdx = 0; nIdx < nLim; ++nIdx) {
            Add(tmp, biaPtr, tmpNMinusNLoopCtr, ARMCG::ShMod::LSL, Math::Log2(sizeof(BiaType)));
            Ldr(ARMCG::SReg(nIdx), ARMCG::Ptr(tmp,
                static_cast<std::uint32_t>(nIdx * sizeof(BiaType))));
            Faddv(ARMCG::SReg(GetDstZRegIdx(mIdx, nIdx)), p5, ARMCG::ZRegS(GetDstZRegIdx(mIdx, nIdx)));
            Fadd(ARMCG::SReg(GetDstZRegIdx(mIdx, nIdx)), ARMCG::SReg(GetDstZRegIdx(mIdx, nIdx)),
                ARMCG::SReg(nIdx));
        }
    }
    L(bf16VCloadBiasM);
    Tbz(flags, 4, bf16VCloadBiasMN); // 4-bit correspond to LOAD_BIAS_M
    Add(tmp, biaPtr, tmpMMinusMLoopCtr, ARMCG::ShMod::LSL, Math::Log2(sizeof(BiaType)));
    for (std::uint32_t mIdx = 0; mIdx < mLim; ++mIdx) {
        Ldr(ARMCG::SReg(0), ARMCG::Ptr(tmp, static_cast<std::uint32_t>(sizeof(BiaType)) * mIdx));
        for (std::uint32_t nIdx = 0; nIdx < nLim; ++nIdx) {
            Faddv(ARMCG::SReg(GetDstZRegIdx(mIdx, nIdx)), p5, ARMCG::ZRegS(GetDstZRegIdx(mIdx, nIdx)));
            Fadd(ARMCG::SReg(GetDstZRegIdx(mIdx, nIdx)), ARMCG::SReg(GetDstZRegIdx(mIdx, nIdx)),
                ARMCG::SReg(0));
        }
    }
    L(bf16VCloadBiasMN);
    Tbz(flags, 6, bf16VCloadC); // 6-bit correspond to LOAD_BIAS_MN
    for (std::uint32_t mIdx = 0; mIdx < mLim; ++mIdx) {
        for (std::uint32_t nIdx = 0; nIdx < nLim; ++nIdx) {
            Mov(ARMCG::XReg(GetTmpXRegIdx(nIdx)), mIdx);
            Add(tmp, tmpMMinusMLoopCtr, ARMCG::XReg(GetTmpXRegIdx(nIdx)));
            Mul(ARMCG::XReg(GetTmpXRegIdx(nIdx)), tmp, biaOffset);
            Add(tmp, biaPtr, ARMCG::XReg(GetTmpXRegIdx(nIdx)), ARMCG::ShMod::LSL, Math::Log2(sizeof(BiaType)));
            Add(tmp, tmp, tmpNMinusNLoopCtr, ARMCG::ShMod::LSL, Math::Log2(sizeof(BiaType)));
            Ldr(ARMCG::SReg(nIdx), ARMCG::Ptr(tmp, static_cast<std::uint32_t>(sizeof(BiaType)) * nIdx));
            Faddv(ARMCG::SReg(GetDstZRegIdx(mIdx, nIdx)), p5, ARMCG::ZRegS(GetDstZRegIdx(mIdx, nIdx)));
            Fadd(ARMCG::SReg(GetDstZRegIdx(mIdx, nIdx)), ARMCG::SReg(GetDstZRegIdx(mIdx, nIdx)),
                ARMCG::SReg(nIdx));
        }
    }
    L(bf16VCloadC);
    Tbz(flags, 1, bf16VCmullC); // 1-bit correspond to LOAD_C
    for (std::uint32_t mIdx = 0; mIdx < mLim; ++mIdx) {
        for (std::uint32_t nIdx = 0; nIdx < nLim; ++nIdx) {
            Ldr(ARMCG::SReg(nIdx), ARMCG::Ptr(ARMCG::XReg(GetDstXRegIdx(mIdx)),
                static_cast<std::uint32_t>(nIdx * sizeof(DstType))));
            Faddv(ARMCG::SReg(GetDstZRegIdx(mIdx, nIdx)), p5, ARMCG::ZRegS(GetDstZRegIdx(mIdx, nIdx)));
            Fadd(ARMCG::SReg(GetDstZRegIdx(mIdx, nIdx)), ARMCG::SReg(GetDstZRegIdx(mIdx, nIdx)),
                ARMCG::SReg(nIdx));
        }
    }
    L(bf16VCmullC);
    Tbz(flags, 2, bf16VCaddBias1); // 2-bit correspond to MULL_C
    Ld1r(ARMCG::ZRegS(NUM_7), p5, Ptr(sp));
    for (std::uint32_t mIdx = 0; mIdx < mLim; ++mIdx) {
        for (std::uint32_t nIdx = 0; nIdx < nLim; ++nIdx) {
            Ldr(ARMCG::SReg(nIdx), ARMCG::Ptr(ARMCG::XReg(GetDstXRegIdx(mIdx)),
                static_cast<std::uint32_t>(nIdx * sizeof(DstType))));
            Fmul(ARMCG::ZRegS(nIdx),
                ARMCG::ZRegS(nIdx),
                ARMCG::ZRegS(NUM_7)); // z7 contains beta value
            Faddv(ARMCG::SReg(GetDstZRegIdx(mIdx, nIdx)), p5, ARMCG::ZRegS(GetDstZRegIdx(mIdx, nIdx)));
            Fadd(ARMCG::SReg(GetDstZRegIdx(mIdx, nIdx)), ARMCG::SReg(GetDstZRegIdx(mIdx, nIdx)),
                ARMCG::SReg(nIdx));
        }
    }
    L(bf16VCaddBias1);
    Tbz(flags, 7, bf16VCaddBiasN); // 7-bit correspond to Add_BIAS_1
    Ldr(ARMCG::SReg(nLim), ARMCG::AdrUimm(ARMCG::Ptr(biaPtr)));
    for (std::uint32_t mIdx = 0; mIdx < mLim; ++mIdx) {
        for (std::uint32_t nIdx = 0; nIdx < nLim; ++nIdx) {
            Fadd(ARMCG::SReg(GetDstZRegIdx(mIdx, nIdx)), ARMCG::SReg(GetDstZRegIdx(mIdx, nIdx)),
                ARMCG::SReg(nLim));
        }
    }
    L(bf16VCaddBiasN);
    Tbz(flags, 9, bf16VCaddBiasM); // 9-bit correspond to Add_BIAS_N
    for (std::uint32_t mIdx = 0; mIdx < mLim; ++mIdx) {
        for (std::uint32_t nIdx = 0; nIdx < nLim; ++nIdx) {
            Add(tmp, biaPtr, tmpNMinusNLoopCtr, ARMCG::ShMod::LSL, Math::Log2(sizeof(BiaType)));
            Ldr(ARMCG::SReg(nIdx), ARMCG::Ptr(tmp,
                static_cast<std::uint32_t>(nIdx * sizeof(BiaType))));
            Fadd(ARMCG::SReg(GetDstZRegIdx(mIdx, nIdx)), ARMCG::SReg(GetDstZRegIdx(mIdx, nIdx)),
                ARMCG::SReg(nIdx));
        }
    }
    L(bf16VCaddBiasM);
    Tbz(flags, 8, bf16VCaddBiasMN); // 8-bit correspond to Add_BIAS_M
    Add(tmp, biaPtr, tmpMMinusMLoopCtr, ARMCG::ShMod::LSL, Math::Log2(sizeof(BiaType)));
    for (std::uint32_t mIdx = 0; mIdx < mLim; ++mIdx) {
        Ldr(ARMCG::SReg(nLim), ARMCG::Ptr(tmp, static_cast<std::uint32_t>(sizeof(BiaType)) * mIdx));
        for (std::uint32_t nIdx = 0; nIdx < nLim; ++nIdx) {
            Fadd(ARMCG::SReg(GetDstZRegIdx(mIdx, nIdx)), ARMCG::SReg(GetDstZRegIdx(mIdx, nIdx)),
                ARMCG::SReg(nLim));
        }
    }
    L(bf16VCaddBiasMN);
    Tbz(flags, 10, bf16VCinitDone); // 10-bit correspond to ADD_BIAS_MN
    for (std::uint32_t mIdx = 0; mIdx < mLim; ++mIdx) {
        for (std::uint32_t nIdx = 0; nIdx < nLim; ++nIdx) {
            Mov(ARMCG::XReg(GetTmpXRegIdx(nIdx)), mIdx);
            Add(tmp, tmpMMinusMLoopCtr, ARMCG::XReg(GetTmpXRegIdx(nIdx)));
            Mul(ARMCG::XReg(GetTmpXRegIdx(nIdx)), tmp, biaOffset);
            Add(tmp, biaPtr, ARMCG::XReg(GetTmpXRegIdx(nIdx)), ARMCG::ShMod::LSL, Math::Log2(sizeof(BiaType)));
            Add(tmp, tmp, tmpNMinusNLoopCtr, ARMCG::ShMod::LSL, Math::Log2(sizeof(BiaType)));
            Ldr(ARMCG::SReg(nIdx), ARMCG::Ptr(tmp, static_cast<std::uint32_t>(sizeof(BiaType)) * nIdx));
            Fadd(ARMCG::SReg(GetDstZRegIdx(mIdx, nIdx)), ARMCG::SReg(GetDstZRegIdx(mIdx, nIdx)),
                ARMCG::SReg(nIdx));
        }
    }
    L(bf16VCinitDone);
    Tbz(flags, 11, skipPO); // 11-bit correspond to APPLY_POST_OPS
    std::vector<SizeType> dstRegsIDs(mLim * nLim);
    for (std::uint32_t mIdx = 0; mIdx < mLim; ++mIdx) {
        for (std::uint32_t nIdx = 0; nIdx < nLim; ++nIdx) {
            dstRegsIDs[mIdx * nLim + nIdx] = GetDstZRegIdx(mIdx, nIdx);
        }
    }
    // even though we use only 1 value in vector register fro dst
    // we can use common logic for post-ops application
    // since all the other values will just be ignored before store
    this->ApplyPostOps(dstRegsIDs, Element::MatchType<DstType>());
    L(skipPO);
    for (std::uint32_t mIdx = 0; mIdx < mLim; ++mIdx) {
        for (std::uint32_t nIdx = 0; nIdx < nLim; ++nIdx) {
            Str(ARMCG::SReg(GetDstZRegIdx(mIdx, nIdx)), ARMCG::Ptr(ARMCG::XReg(GetDstXRegIdx(mIdx)),
                static_cast<std::uint32_t>(nIdx * sizeof(DstType))));
        }
    }
}

void VCBF16Gen::GenerateMLoopBody(std::uint32_t mLim,
    std::uint32_t nLim, std::uint32_t k)
{
    std::uint32_t nResid = nLim % KERNEL_N_SIZE;
    if (nResid == 0) {
        nResid = KERNEL_N_SIZE;
    }
    Mov(nLoopCtr, nVal);
    Sub(tmpNMinusNLoopCtr, nVal, nLoopCtr);
    if (nLim > KERNEL_N_SIZE) {
        ARMCG::Label nLoopHead;
        L(nLoopHead);
        {
            GenerateNLoopBody(mLim, KERNEL_N_SIZE, k);
            Sub(nLoopCtr, nLoopCtr, KERNEL_N_SIZE);
            Cmp(nLoopCtr, KERNEL_N_SIZE);
            Sub(tmpNMinusNLoopCtr, nVal, nLoopCtr);
        }
        B(ARMCG::Cond::GT, nLoopHead);
    }
    GenerateNLoopBody(mLim, nResid, k);
}

void VCBF16Gen::GenerateNLoopBody(std::uint32_t mLim,
    std::uint32_t nLim, std::uint32_t k)
{
    for (std::uint32_t mIdx = 0; mIdx < KERNEL_M_SIZE; ++mIdx) {
        for (std::uint32_t nIdx = 0; nIdx < KERNEL_N_SIZE; ++nIdx) {
            Mov(ARMCG::ZRegH(GetDstZRegIdx(mIdx, nIdx)), 0);
        }
    }
    // Initialize src pointers
    Mul(tmp, tmpMMinusMLoopCtr, srcOffset);
    Lsl(tmp, tmp, NUM_1); // multiply by sizeof(__bf16) = 2
    Mov(ARMCG::XReg(GetSrcXRegIdx(0)), 0);
    Add(ARMCG::XReg(GetSrcXRegIdx(0)), srcPtr, tmp);
    for (std::uint32_t mIdx = 1; mIdx < mLim; ++mIdx) {
        Addvl(ARMCG::XReg(GetSrcXRegIdx(mIdx)), ARMCG::XReg(GetSrcXRegIdx(0)), mIdx);
    }
    // Initialize wei pointers
    Mul(tmp, tmpNMinusNLoopCtr, weiOffset);
    Lsl(tmp, tmp, NUM_1); // multiply by sizeof(__bf16) = 2
    Mov(ARMCG::XReg(GetWeiXRegIdx(0)), 0);
    Add(ARMCG::XReg(GetWeiXRegIdx(0)), weiPtr, tmp);
    for (std::uint32_t nIdx = 1; nIdx < nLim; ++nIdx) {
        Addvl(ARMCG::XReg(GetWeiXRegIdx(nIdx)), ARMCG::XReg(GetWeiXRegIdx(0)), nIdx);
    }
    // Main k loop
    Mov(kLoopCtr, kVal);
    if (k > (vectorLengthByte / NUM_2)) {
        ARMCG::Label kLoopHead;
        L(kLoopHead);
        {
            Sub(kLoopCtr, kLoopCtr, (vectorLengthByte / NUM_2));
            Cmp(kLoopCtr, (vectorLengthByte / NUM_2));
            GenerateAccumulation(mLim, nLim);
        }
        B(ARMCG::Cond::GT, kLoopHead);
    }

    GenerateAccumulation(mLim, nLim);

    // Initialize dst pointers
    Mul(tmp, tmpMMinusMLoopCtr, dstOffset);
    // Row-major format of dst
    Add(ARMCG::XReg(GetDstXRegIdx(0)), dstPtr, tmp,
        ARMCG::ShMod::LSL, 2); // Add stride for M, lsl 2 - mul by sizeof(float)
    Add(ARMCG::XReg(GetDstXRegIdx(0)),
        ARMCG::XReg(GetDstXRegIdx(0)),
        tmpNMinusNLoopCtr, ARMCG::ShMod::LSL, 2); // Add stride for N, lsl 2 - mul by sizeof(float)
    for (std::uint32_t mIdx = 1; mIdx < mLim; ++mIdx) {
        Add(ARMCG::XReg(GetDstXRegIdx(mIdx)),
            ARMCG::XReg(GetDstXRegIdx(mIdx - 1)),
            dstOffset, ARMCG::ShMod::LSL, 2); // lsl 2 - mul by sizeof(int32_t)
    }

    // Store data to dst
    StoreToDst(mLim, nLim);
}

void VCBF16Gen::GenerateAccumulation(std::uint32_t mLim, std::uint32_t nLim)
{
    Ld1h(ARMCG::ZRegH(GetWeiZRegIdx(0)), p5, Ptr(ARMCG::XReg(GetWeiXRegIdx(0))));
    Addvl(ARMCG::XReg(GetWeiXRegIdx(0)), ARMCG::XReg(GetWeiXRegIdx(0)), nLim);
    if (nLim > 1) {
        Ld1h(ARMCG::ZRegH(GetWeiZRegIdx(1)), p5, Ptr(ARMCG::XReg(GetWeiXRegIdx(1))));
        Addvl(ARMCG::XReg(GetWeiXRegIdx(1)), ARMCG::XReg(GetWeiXRegIdx(1)), nLim);
    }
    for (std::uint32_t mIdx = 0; mIdx < mLim; ++mIdx) {
        Ld1h(ARMCG::ZRegH(GetSrcZRegIdx(mIdx)), p5, Ptr(ARMCG::XReg(GetSrcXRegIdx(mIdx))));
        Addvl(ARMCG::XReg(GetSrcXRegIdx(mIdx)), ARMCG::XReg(GetSrcXRegIdx(mIdx)), mLim);
        // propogate ptr forward by (vlb x mLim) bytes
    }

    for (std::uint32_t nIdx = 0; nIdx < nLim; ++nIdx) {
        // propogate ptr forward by (vlb x nLim) bytes
        if (nIdx > 1) {
            Ld1h(ARMCG::ZRegH(GetWeiZRegIdx(nIdx)), p5, Ptr(ARMCG::XReg(GetWeiXRegIdx(nIdx))));
            Addvl(ARMCG::XReg(GetWeiXRegIdx(nIdx)), ARMCG::XReg(GetWeiXRegIdx(nIdx)), nLim);
        }
        for (std::uint32_t mIdx = 0; mIdx < mLim; ++mIdx) {
            Bfdot(ARMCG::ZRegS(GetDstZRegIdx(mIdx, nIdx)),
                  ARMCG::ZRegH(GetSrcZRegIdx(mIdx)),
                  ARMCG::ZRegH(GetWeiZRegIdx(nIdx)));
        }
    }
}

MatMulImplGenerator<VCBF16Gen>::DefaultImpl::DefaultImpl(
    const GemmInfo &task, IntType nThreads) : MatMulImplBase(task, nThreads)
{
    UpdateExecParams();
    Initialize();
}

constexpr std::uint32_t MatMulImplGenerator<VCBF16Gen>::DefaultImpl::GetKernelMSize() noexcept
{
    return VCBF16Gen::KERNEL_M_SIZE;
}

constexpr std::uint32_t MatMulImplGenerator<VCBF16Gen>::DefaultImpl::GetKernelNSize() noexcept
{
    return VCBF16Gen::KERNEL_N_SIZE;
}

std::uint32_t MatMulImplGenerator<VCBF16Gen>::DefaultImpl::GetKernelKSize() noexcept
{
    return svcnth();
}

IntType MatMulImplGenerator<VCBF16Gen>::DefaultImpl::GetMBlockSize() const noexcept
{
    // should be equally divisible by KERNEL_M_SIZE (6)
    return 2400; // is experimentally measured constant to obtain better perf
}
IntType MatMulImplGenerator<VCBF16Gen>::DefaultImpl::GetNBlockSize() const noexcept
{
    // should be equally divisible by KERNEL_N_SIZE (4)
    return 256; // is experimentally measured constant to obtain better perf
}
IntType MatMulImplGenerator<VCBF16Gen>::DefaultImpl::GetKBlockSize() const noexcept
{
    // should be equally divisible by vlh (16 for VC256, 32 for VC512)
    return 1024; // is experimentally measured constant to obtain better perf
}

void MatMulImplGenerator<VCBF16Gen>::DefaultImpl::InitializePackingPolicy()
{
    packingPolicy = PackingPolicy::SYNC_DST_DIM_SRC;
}

void MatMulImplGenerator<VCBF16Gen>::DefaultImpl::InitializeKernelK(IntType k, IntType kBlockSize,
    IntType &kBlockExecFull, IntType &kBlockExecResid) const noexcept
{
    IntType vlh = svcnth();
    kBlockExecFull = kBlockSize;
    kBlockExecResid = k % kBlockExecFull;
    kBlockExecResid = Math::CeilDiv(kBlockExecResid, vlh) * vlh;
}

void MatMulImplGenerator<VCBF16Gen>::DefaultImpl::InitializeSingleCodegen(
    IntType id, IntType m, IntType n, IntType k,
    IntType currentM, IntType currentN, IntType currentK,
    SizeType offsetSrc, SizeType offsetWei, SizeType offsetDst, SizeType offsetBia,
    bool wereSrcWeiSwapped, const Attributes &attributes, bool hasAlpha)
{
    Service::Unused(offsetSrc, offsetWei, offsetDst, offsetBia, wereSrcWeiSwapped, hasAlpha);
    if ((currentM == 0) || (currentN == 0) || (currentK == 0)) {
        return;
    }
    IntType vlh = svcnth();
    if ((currentM <= m) && (currentN <= n) && (currentK <= (Math::CeilDiv(k, vlh) * vlh))) {
        CodeGenParams<VCBF16Gen> currTask{currentM, currentN, currentK, false, attributes};
        if (codegens.find(currTask) == codegens.end()) {
            codegens.insert({currTask, std::make_shared<VCBF16Gen>(
                currentM, currentN, currentK, attributes)});
        }
        codegens[currTask]->Ready();
        matmulCodegenKernels[id] = codegens[currTask]->template GetCode<VCBF16CodegenFuncPtr>();
    }
}

void MatMulImplGenerator<VCBF16Gen>::DefaultImpl::InitializeThreadsCountImpl()
{
    IntType threadNumber;
    IntType size = GetRequiredMemSize();
    if (size < 7232) { // 7232 is practically measured number
        threadNumber = NUM_1;
    } else if (size < 51200) { // 51200 is practically measured number
        threadNumber = NUM_8;
    } else if (size < 139392) { // 139392 is practically measured number
        threadNumber = NUM_16;
    } else if (size < 359552) { // 359552 is practically measured number
        threadNumber = NUM_40;
    } else {
        threadNumber = InitThreadNum();
    }
    nThreadsExec = std::min(threadNumber, implNThreads);
}

IntType MatMulImplGenerator<VCBF16Gen>::DefaultImpl::GetThreadBufSize(
    IntType m, IntType n, IntType k, MatrixKind mKind) const noexcept
{
    // since vlw passed here, we should adjust
    IntType vlh = svcnth();
    if (mKind == MatrixKind::SRC) {
        return static_cast<IntType>(vlh * Math::CeilDiv(std::min<IntType>(GetKBlockSize(), k), vlh) *
            std::min<IntType>(GetMBlockSize(), m) * sizeof(SrcType));
    } else {
        return static_cast<IntType>(vlh * Math::CeilDiv(std::min<IntType>(GetKBlockSize(), k), vlh) *
             std::min<IntType>(GetNBlockSize(), n) * sizeof(WeiType));
    }
}

void MatMulImplGenerator<VCBF16Gen>::DefaultImpl::ReserveImplSpecificParams() {}

std::vector<std::pair<Shape, Shape>> MatMulImplGenerator<VCBF16Gen>::DefaultImpl::SplitWork()
{
    return CalculateWorkOnThread<DefaultImpl>(taskExec.dstInfo.GetDims(), nThreadsExec, dstLayout2D == Layout::BA);
}

template <IntType PACK_ROWS>
void MatMulImplGenerator<VCBF16Gen>::DefaultImpl::PackSimilar(IntType rowBlock, IntType colBlock,
    IntType kBlockExec, const void *src, IntType offset, void *buffer)
{
    IntType vlh = svcnth();
    svbool_t ptrue = svptrue_b8();
    __bf16 *to = static_cast<__bf16 *>(buffer);
    const __bf16 *from = static_cast<const __bf16 *>(src);
    IntType row = 0;

    for (; row + PACK_ROWS < rowBlock; row += PACK_ROWS) {
        for (IntType col = 0; col < colBlock; col += vlh) {
            svbool_t p0 = svwhilelt_b16(col, colBlock);
            for (IntType packRow = 0; packRow < PACK_ROWS; packRow++) {
                svbfloat16_t values = svld1(p0, from + (row + packRow) * offset + col);
                svst1(ptrue, to + packRow * vlh + row * kBlockExec + PACK_ROWS * col, values);
            }
        }
    }
    
    IntType remainingRows = rowBlock % PACK_ROWS;
    remainingRows = (remainingRows == 0) ? PACK_ROWS : remainingRows;
    to += row * kBlockExec;

    for (IntType col = 0; col < colBlock; col += vlh) {
        svbool_t p0 = svwhilelt_b16(col, colBlock);
        for (IntType packRow = 0; packRow < remainingRows; packRow++) {
            svbfloat16_t values = svld1(p0, from + (row + packRow) * offset + col);
            svst1(ptrue, to + packRow * vlh + col * remainingRows, values);
        }
    }
}

template <IntType PACK_COLS>
void MatMulImplGenerator<VCBF16Gen>::DefaultImpl::PackDifferent(IntType rowBlock, IntType colBlock,
    IntType kBlockExec, const void *src, IntType offset, void *buffer)
{
    IntType vlh = svcnth();
    IntType col = 0;
    __bf16 *to = static_cast<__bf16 *>(buffer);
    const __bf16 *from  = static_cast<const __bf16 *>(src);

    for (IntType row = 0; row < rowBlock; row += vlh) {
        for (col = 0; col + PACK_COLS < colBlock; col += PACK_COLS) {
            for (IntType packRow = 0; packRow < vlh; packRow++) {
                for (IntType packCol = 0; packCol < PACK_COLS; packCol++) {
                    to[packCol * vlh + packRow + PACK_COLS * row + col * kBlockExec] =
                        (row + packRow < rowBlock) ?
                         from[(row + packRow) * offset + col + packCol] : vcvth_bf16_f32(0);
                }
            }
        }
    }

    IntType remainingCols = colBlock % PACK_COLS;
    remainingCols = remainingCols == 0 ? PACK_COLS : remainingCols;
    to += (colBlock - remainingCols) * kBlockExec;

    for (IntType row = 0; row < rowBlock; row += vlh) {
        for (IntType packRow = 0; packRow < vlh; packRow++) {
            for (IntType packCol = 0; packCol < remainingCols; packCol++) {
                to[packCol * vlh + packRow] = (row + packRow < rowBlock) ?
                    from[(row + packRow) * offset + col + packCol] : vcvth_bf16_f32(0);
            }
        }
        to += remainingCols * vlh;
    }
}

void MatMulImplGenerator<VCBF16Gen>::DefaultImpl::Pack(
    const void *originalMatrix, void *buffer, IntType threadID, IntType dstDim, IntType accumDim, IntType dstDimOffset,
    IntType accumDimOffset, IntType bufferOffset, IntType bufferStride, float alpha, MatrixKind mKind) const
{
    IntType offset2D;
   
    if (mKind == MatrixKind::SRC) {
        offset2D = perThreadParams[threadID].srcOffset2D;
    } else {
        offset2D = perThreadParams[threadID].weiOffset2D;
    }

    Service::Unused(alpha);

    if (mKind == MatrixKind::SRC) {
        const SrcType *srcExec = static_cast<const SrcType *>(originalMatrix);
        if (perThreadParams[threadID].srcLayout2D == Layout::COL_MAJOR) {
            srcExec += dstDimOffset + accumDimOffset * offset2D;
            PackDifferent<GetKernelMSize()>(accumDim, dstDim, bufferStride, srcExec,
                perThreadParams[threadID].srcOffset2D,
                static_cast<SrcType *>(buffer) + bufferOffset);
        } else {
            srcExec += dstDimOffset * offset2D + accumDimOffset;
            PackSimilar<GetKernelMSize()>(dstDim, accumDim, bufferStride, srcExec,
                perThreadParams[threadID].srcOffset2D,
                static_cast<SrcType *>(buffer) + bufferOffset);
        }
    } else {
        const WeiType *weiExec = static_cast<const WeiType *>(originalMatrix);
        if (perThreadParams[threadID].weiLayout2D == Layout::ROW_MAJOR) {
            weiExec += dstDimOffset + accumDimOffset * offset2D;
            PackDifferent<GetKernelNSize()>(accumDim, dstDim, bufferStride, weiExec,
                perThreadParams[threadID].weiOffset2D,
                static_cast<WeiType *>(buffer) + bufferOffset);
        } else {
            weiExec += dstDimOffset * offset2D + accumDimOffset;
            PackSimilar<GetKernelNSize()>(dstDim, accumDim, bufferStride, weiExec,
                perThreadParams[threadID].weiOffset2D,
                static_cast<WeiType *>(buffer) + bufferOffset);
        }
    }
}


void MatMulImplGenerator<VCBF16Gen>::DefaultImpl::RunSequential2D(
    IntType threadID, const void *src, const void *wei, void *dst, const void *bia,
    std::unique_ptr<PackStrategy> &srcPackStrategy, std::unique_ptr<PackStrategy> &weiPackStrategy,
    float alpha, float beta, const PostOpsDataPtrs &shiftedByBatchPOPtrs) const noexcept(false)
{
    Service::Unused(shiftedByBatchPOPtrs);
    VERBOSE_PRINT;
    IntType m = perThreadParams[threadID].m;
    IntType n = perThreadParams[threadID].n;
    IntType k = perThreadParams[threadID].k;

    IntType tIDbatch;
    IntType tIDm;
    IntType tIDn;
    InitThreadIDs(threadID, tIDbatch, tIDm, tIDn);
    SrcType* srcBuffer = static_cast<SrcType *>(
        srcPackStrategy->GetPtr(tIDbatch * threadsM  + tIDm, tIDn, threadsN));
    WeiType* weiBuffer = static_cast<WeiType *>(
        weiPackStrategy->GetPtr(tIDbatch * threadsN  + tIDn, tIDm, threadsM));

    for (IntType mBlock = 0; mBlock < mFullBlockSizePerThread; mBlock += GetMBlockSize()) {
        auto mBlockExec = std::min(GetMBlockSize(),  m - mBlock);
        // mIdx, nIdx, kIdx -- stands for cache
        IntType mIdx = (mBlockExec == GetMBlockSize()) ? 0 : 1;
        IntType mIdxKernel = mIdx * (perThreadParams[threadID].isResidByM + 1);
        for (IntType kBlock = 0; kBlock < k; kBlock += GetKBlockSize()) {
            auto kBlockExec = std::min(GetKBlockSize(), k - kBlock);
            IntType kIdxKernel = (kBlockExec == GetKBlockSize()) ? 0 : 1;
            bool lastKIter = (k - kBlock) <= GetKBlockSize();
            MatMulDstFlags biaFlagsExec = SetBiaFlagsExec(biaFlags, beta, kBlock, lastKIter, alpha);
            IntType kBlockExecRoundedUp = Math::CeilDiv(kBlockExec, svcnth()) * svcnth();
            if (mBlock < m) {
                srcPackStrategy->Pack(*this, src, srcBuffer,
                    {mBlock, mBlockExec, GetKernelMSize()}, {kBlock, kBlockExec, GetKernelKSize()},
                    threadsN, tIDn, threadID, alpha);
            }
            srcPackStrategy->Barrier(tIDbatch * threadsM + tIDm, tIDn, threadsN);
            for (IntType nBlock = 0; nBlock < nFullBlockSizePerThread; nBlock += GetNBlockSize()) {
                if ((nBlock < n) && (mBlock < m)) {
                    auto nBlockExec = std::min(GetNBlockSize(), n - nBlock);
                    int nIdx = (nBlockExec == GetNBlockSize()) ? 0 : 1;
                    IntType nIdxKernel = nIdx * (perThreadParams[threadID].isResidByN + 1);
                    weiPackStrategy->Pack(*this, wei, weiBuffer,
                        {nBlock, nBlockExec, GetKernelNSize()}, {kBlock, kBlockExec, GetKernelKSize()},
                        threadsM, tIDm, threadID, alpha);
                    weiPackStrategy->Barrier(tIDbatch * threadsN + tIDn, tIDm, threadsM);
                    DstType *dstExec = static_cast<DstType *>(dst) + nBlock +
                        mBlock * perThreadParams[threadID].dstOffset2D;
                    const BiaType *biaExec = static_cast<const BiaType *>(bia) + nBlock * biaNOffset +
                            mBlock * perThreadParams[threadID].biaOffset2D;
                    auto kernel = matmulCodegenKernels[kIdxKernel * NUM_9 +
                            nIdxKernel * NUM_3 + mIdxKernel];
                    kernel(srcBuffer, weiBuffer, dstExec, biaExec, biaFlagsExec, alpha, beta,
                        mBlockExec, nBlockExec, kBlockExec, kBlockExecRoundedUp, kBlockExecRoundedUp,
                        perThreadParams[threadID].dstOffset2D, perThreadParams[threadID].biaOffset2D);
                    weiPackStrategy->Barrier(tIDbatch * threadsN + tIDn, tIDm, threadsM);
                }
            }
            srcPackStrategy->Barrier(tIDbatch * threadsM + tIDm, tIDn, threadsN);
        }
    }
}

void MatMulImplGenerator<VCBF16Gen>::DefaultImpl::UpdateExecParams() noexcept
{
    Layout desiredSrcLayout = implTask.srcInfo.GetLayout();
    Layout desiredWeiLayout = implTask.weiInfo.GetLayout();
    Layout desiredDstLayout = implTask.dstInfo.GetLayout();
    Layout desiredBiaLayout = implTask.biaInfo.GetLayout();
    if (!IsLayoutAlikeAB(desiredSrcLayout) && !IsLayoutAlikeBA(desiredSrcLayout)) {
        desiredSrcLayout = GetPlaneLayoutAB(desiredSrcLayout);
    }
    if (!IsLayoutAlikeAB(desiredWeiLayout) && !IsLayoutAlikeBA(desiredWeiLayout)) {
        desiredWeiLayout = GetPlaneLayoutBA(desiredWeiLayout);
    }
    if (!IsLayoutAlikeAB(desiredDstLayout) && !IsLayoutAlikeBA(desiredDstLayout)) {
        desiredDstLayout = GetPlaneLayoutAB(desiredDstLayout);
    }
    if (implTask.biaInfo.GetTotalTensorSize() != 0) {
        desiredBiaLayout = desiredDstLayout;
    }
    InitializeTaskExec(desiredSrcLayout, desiredWeiLayout, desiredDstLayout, desiredBiaLayout,
        VCBF16Gen::GetSrcType(), VCBF16Gen::GetWeiType(), VCBF16Gen::GetDstType(),
        (implTask.biaInfo.GetTotalTensorSize() != 0) ? VCBF16Gen::GetBiaType() :
            static_cast<Element::TypeT>(implTask.biaInfo.GetType()));
}

MatMulDstFlags MatMulImplGenerator<VCBF16Gen>::DefaultImpl::SetBiaFlagsExec(
    MatMulDstFlags biaF, float beta, bool notFirstKIter,
    bool lastKIter, float alpha) const noexcept
{
    Service::Unused(alpha);
    MatMulDstFlags ret;
    if (notFirstKIter) {
        ret = MatMulDstFlags::LOAD_C;
    } else { // first K iter, need to process bias and beta
        if (biaF == MatMulDstFlags::FILL_C_WITH_ZEROZ) { // No Bias
            if (beta == 0) {
                ret = MatMulDstFlags::FILL_C_WITH_ZEROZ;
            } else if (beta == 1) {
                ret = MatMulDstFlags::LOAD_C;
            } else {
                ret = MatMulDstFlags::MULL_C;
            }
        } else { // bias
            if (beta == 0) {
                ret = biaF;
            } else if (beta == 1) {
                ret = MatMulDstFlags::LOAD_C | (biaF << NUM_4);
            } else {
                ret = MatMulDstFlags::MULL_C | (biaF << NUM_4);
            }
        }
        ret = ret | MatMulDstFlags::FIRST_K_ITER;
    }
    if (lastKIter) {
        ret = ret | MatMulDstFlags::APPLY_POST_OPS;
    }
    return ret;
}


} // Detail
} // KuDNN
