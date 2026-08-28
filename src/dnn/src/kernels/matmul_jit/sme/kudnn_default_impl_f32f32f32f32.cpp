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

#include "kudnn.hpp"

#include "cg_codegen.hpp"

#include "kudnn_jit.hpp"
#include "kernels/matmul_jit/kudnn_gemm_jit.hpp"
#include "kernels/matmul_jit/kudnn_gemm_helpers.hpp"
#include "kernels/matmul_jit/kudnn_gemm_threading_helpers.hpp"
#include "kudnn_wrapper_helpers.hpp"
#include "kernels/reorder/sve/bf16any32/kudnn_fast_jit_impl_bfp16any32_bfp16any32.hpp"
#include "kernels/matmul_jit/sme/kudnn_default_impl_f32f32f32f32.hpp"

namespace KuDNN {
namespace Detail {

using MCF32GenBase = MatMulKernelCodeGenBase<ARCH::MC, float, float, float, float>;

MCF32Gen::MatMulKernelDefaultCodeGen(
    IntType m, IntType n, IntType k,
    const Attributes &attributesArg) : MCF32GenBase(attributesArg)
{
    if ((m > 0) && (n > 0) && (k > 0)) {
        if (this->attributes.GetPostOps().Len() > 0) {
            // set reserved predicate registers
            for (std::uint32_t idx = 0; idx < NUM_6; ++idx) {
                TakePredicateRegister(idx);
            }
            // set reserved scalar registers
            for (std::uint32_t idx = 0; idx < NUM_29; ++idx) {
                TakeScalarRegister(idx);
            }
        }
        Add(sp, sp, NUM_16);
        Ldp(dstOffset, biaOffset, PostPtr(sp, NUM_0));
        Sub(sp, sp, NUM_16);
        Preamble();

        // Move to scalar register before Smstart
        Fmov(alphaW, alpha);
        Fmov(betaW, beta);

        Smstart(ARMCG::StreamMode::SMZA);

        Mov(fOne, 0x3f800000); // 0x3f800000 is a bit representation of 1.0f
        // constants to perform continious stores from ZA tiles
        Mov(tmpLDStC0, NUM_0);
        Mov(tmpLDStC1, NUM_4);
        Mov(tmpLDStC2, NUM_8);
        Mov(tmpLDStC3, NUM_12);

        IntType vl = svcntw();
        IntType mResid = m % (KERNEL_M_SIZE * vl);
        if (mResid == 0) {
            mResid = KERNEL_M_SIZE * vl;
        }
        Mov(mLoopCtr, mVal);
        Sub(tmpMMinusMLoopCtr, mVal, mLoopCtr);
        if (m > (KERNEL_M_SIZE * vl)) {
            ARMCG::Label mLoopHead;
            L(mLoopHead);
            {
                GenerateMLoopBody(KERNEL_M_SIZE * vl, n, k);
                Sub(mLoopCtr, mLoopCtr, KERNEL_M_SIZE * vl);
                Cmp(mLoopCtr, KERNEL_M_SIZE * vl);
                Sub(tmpMMinusMLoopCtr, mVal, mLoopCtr);
            }
            B(ARMCG::Cond::GT, mLoopHead);
        }
        GenerateMLoopBody(mResid, n, k);

        Smstop(ARMCG::StreamMode::SMZA);

        Postamble();
    }
    Ret();
}

void MCF32Gen::GenerateMLoopBody(IntType mLim, IntType n, IntType k)
{
    IntType vl = svcntw();
    IntType nResid = n % (KERNEL_N_SIZE * vl);
    if (nResid == 0) {
        nResid = KERNEL_N_SIZE * vl;
    }
    Mov(nLoopCtr, nVal);
    Sub(tmpNMinusNLoopCtr, nVal, nLoopCtr);
    // initialize predicates for SRC loads
    Mov(tmp, 0);
    // p0, p2 used for SRC, p1, p3 used for WEI
    Whilelt(ARMCG::PRegS(NUM_0), tmp, mLoopCtr);
    if (mLim > vl) {
        Incw(tmp);
        Whilelt(ARMCG::PRegS(NUM_2), tmp, mLoopCtr);
    }
    if (n > (KERNEL_N_SIZE * vl)) {
        ARMCG::Label nLoopHead;
        L(nLoopHead);
        {
            GenerateNLoopBody(mLim, KERNEL_N_SIZE * vl, k);
            Sub(nLoopCtr, nLoopCtr, KERNEL_N_SIZE * vl);
            Cmp(nLoopCtr, KERNEL_N_SIZE * vl);
            Sub(tmpNMinusNLoopCtr, nVal, nLoopCtr);
        }
        B(ARMCG::Cond::GT, nLoopHead);
    }
    GenerateNLoopBody(mLim, nResid, k);
}
 
void MCF32Gen::GenerateNLoopBody(
    IntType mLim, IntType nLim, IntType k)
{
    IntType vl = svcntw();
    // Initialize src pointer
    Mul(tmp, tmpMMinusMLoopCtr, kVal);
    Add(srcPtrCopy, srcPtr, tmp,
        ARMCG::ShMod::LSL, Math::Log2(sizeof(SrcType)));
    // initialize predicates for WEI loads
    Mov(tmp, 0);
    // p0, p2 used for SRC, p1, p3 used for WEI
    Whilelt(ARMCG::PRegS(NUM_1), tmp, nLoopCtr);
    if (nLim > vl) {
        Incw(tmp);
        Whilelt(ARMCG::PRegS(NUM_3), tmp, nLoopCtr);
    }
    // Initialize wei pointer
    Mul(tmp, tmpNMinusNLoopCtr, kVal);
    Add(weiPtrCopy, weiPtr, tmp,
        ARMCG::ShMod::LSL, Math::Log2(sizeof(WeiType)));
 
    // Initialize dst pointer
    Mul(tmp, tmpMMinusMLoopCtr, dstOffset);
    // Row-major format of dst
    Add(dstPtrCopy, dstPtr, tmp); // Add stride for M
    Add(dstPtrCopy, dstPtrCopy, tmpNMinusNLoopCtr,
        ARMCG::ShMod::LSL, Math::Log2(sizeof(DstType))); // Add stride for N
    Mov(dstPtrLoopCopy, dstPtrCopy);
    // Initialize bia pointer
    Mul(tmp, tmpMMinusMLoopCtr, biaOffset);
    // Row-major format of bia
    Add(biaPtrCopy, biaPtr, tmp); // Add stride for M
    ARMCG::Label incrementBiasByN;
    ARMCG::Label notIncrementBiasByN;
    Tbnz(flags, 5, incrementBiasByN); // 4-bit correspond to LOAD_BIAS_N
    Tbnz(flags, 9, incrementBiasByN); // 8-bit correspond to ADD_BIAS_N
    Tbnz(flags, 6, incrementBiasByN); // 4-bit correspond to LOAD_BIAS_MN
    Tbnz(flags, 10, incrementBiasByN); // 8-bit correspond to ADD_BIAS_MN
    B(notIncrementBiasByN);
    L(incrementBiasByN);
    Add(biaPtrCopy, biaPtrCopy, tmpNMinusNLoopCtr,
        ARMCG::ShMod::LSL, Math::Log2(sizeof(BiaType))); // Add stride for N
    L(notIncrementBiasByN);
    Zero({ARMCG::ZARegS(NUM_0), ARMCG::ZARegS(NUM_1), ARMCG::ZARegS(NUM_2), ARMCG::ZARegS(NUM_3)});
    Ptrue(ARMCG::PRegS(NUM_5), ARMCG::Pattern::ALL);
    // Main k loop
    GenerateKLoop(mLim, nLim, k, vl);
    // Divide 2vl*2vl kernel by 4
    // 4 parts of:  m/2 * 2 vl
    // Maximum 8 Z register loaded in split part
    IntType Rows = std::min(mLim, vl);
    IntType Split = Math::CeilDiv(Rows, NUM_2);
    if (Split > 0) {
        MoveDstRegistersPartSplit(Split, nLim, vl, NUM_0, NUM_1, NUM_0);
        Mov(dstPtrCopy, dstPtrLoopCopy);
        BiasAndC(Split, nLim, vl);
        Mov(dstPtrCopy, dstPtrLoopCopy);
        StoreDstRegistersPartSplit(Split, nLim, vl);
    }

    IntType Split2 = Rows - Split;
    if (Split2 > 0) {
        Add(dstPtrCopy, dstPtrCopy, dstOffset);
        Mov(dstPtrStore, dstPtrCopy);
        MoveDstRegistersPartSplit(Split2, nLim, vl, NUM_0, NUM_1, Split);
        Mov(dstPtrCopy, dstPtrStore);
        Add(biaPtrCopy, biaPtrCopy, biaOffset);
        BiasAndC(Split2, nLim, vl);
        Mov(dstPtrCopy, dstPtrStore);
        StoreDstRegistersPartSplit(Split2, nLim, vl);
    }

    IntType RowsResid = std::min(mLim - vl, vl);
    IntType Split3 = Math::CeilDiv(RowsResid, NUM_2);
    if (Split3 > 0) {
        Add(dstPtrCopy, dstPtrCopy, dstOffset);
        Mov(dstPtrStore, dstPtrCopy);
        MoveDstRegistersPartSplit(Split3, nLim, vl, NUM_2, NUM_3, NUM_0);
        Mov(dstPtrCopy, dstPtrStore);
        Add(biaPtrCopy, biaPtrCopy, biaOffset);
        BiasAndC(Split3, nLim, vl);
        Mov(dstPtrCopy, dstPtrStore);
        StoreDstRegistersPartSplit(Split3, nLim, vl);
    }
    IntType Split4 = RowsResid - Split3;
    if (Split4 > 0) {
        Add(dstPtrCopy, dstPtrCopy, dstOffset);
        Mov(dstPtrStore, dstPtrCopy);
        MoveDstRegistersPartSplit(Split4, nLim, vl, NUM_2, NUM_3, Split3);
        Mov(dstPtrCopy, dstPtrStore);
        Add(biaPtrCopy, biaPtrCopy, biaOffset);
        BiasAndC(Split4, nLim, vl);
        Mov(dstPtrCopy, dstPtrStore);
        StoreDstRegistersPartSplit(Split4, nLim, vl);
    }
}

void MCF32Gen::MoveDstRegistersPartSplit(IntType SplitSize, IntType n, IntType vl,
    IntType ZARegIdx1, IntType ZARegIdx2, IntType ZARegOffset)
{
    for (IntType i = 0; i < SplitSize; ++i) {
        if (i != 0) {
            Add(dstPtrCopy, dstPtrCopy, dstOffset);
        }
        // 2 - use even numbered Z registers as tmp, 12 - number of first w register
        // 4 - max imm in MOVA instruction
        Mova(ARMCG::ZRegS(i * NUM_2), p1, ARMCG::ZARegHS(ZARegIdx1),
            ARMCG::WReg(NUM_12 + (i + ZARegOffset) / NUM_4), (i + ZARegOffset) % NUM_4);
        if (n > vl) {
            // 2 - use even numbered Z registers as tmp, 12 - number of first w register
            // 4 - max imm in MOVA instruction
            Mova(ARMCG::ZRegS(i * NUM_2 + NUM_1), p3, ARMCG::ZARegHS(ZARegIdx2),
                ARMCG::WReg(NUM_12 + (i + ZARegOffset) / NUM_4), (i + ZARegOffset) % NUM_4);
        }
    }
}

void MCF32Gen::MulAdd(IntType dstIdx, IntType argIdx,
    IntType pregIdx, IntType betaIdx)
{
    Fmul(ARMCG::ZRegS(argIdx), ARMCG::ZRegS(argIdx), ARMCG::ZRegS(betaIdx));
    Fadd(ARMCG::ZRegS(dstIdx), ARMCG::PRegB(pregIdx), ARMCG::ZRegS(argIdx));
}

void MCF32Gen::AddPair(IntType n, IntType vlw,
    IntType dstIdx1, IntType dstIdx2, IntType argIdx1, IntType argIdx2)
{
    Fadd(ARMCG::ZRegS(dstIdx1), p1, ARMCG::ZRegS(argIdx1));
    if (n > vlw) {
        Fadd(ARMCG::ZRegS(dstIdx2), p3, ARMCG::ZRegS(argIdx2));
    }
}

void MCF32Gen::BiasAndC(IntType SplitSize, IntType n, IntType vlw)
{
    // Initialize Dst matrix registers
    ARMCG::Label f32MCLoadC;
    ARMCG::Label f32MCMullC;
    ARMCG::Label f32MCLoadBias1;
    ARMCG::Label f32MCLoadBiasM;
    ARMCG::Label f32MCLoadBiasN;
    ARMCG::Label f32MCLoadBiasMN;
    ARMCG::Label f32MCAddBias1;
    ARMCG::Label f32MCAddBiasM;
    ARMCG::Label f32MCAddBiasN;
    ARMCG::Label f32MCAddBiasMN;
    ARMCG::Label f32MCInitDone;
    ARMCG::Label f32MCSkipPO;

    Cmp(alphaW, fOne);
    Beq(f32MCLoadBias1);
    Dup(ARMCG::ZRegS(NUM_30), alphaW);
    for (IntType i = 0; i < SplitSize; ++i) {
        Fmul(ARMCG::ZRegS(i * NUM_2), ARMCG::ZRegS(i * NUM_2), ARMCG::ZRegS(NUM_30));
        if (n > vlw) {
            Fmul(ARMCG::ZRegS(i * NUM_2 + NUM_1), ARMCG::ZRegS(i * NUM_2 + NUM_1), ARMCG::ZRegS(NUM_30));
        }
    }

    Tbz(flags, 0, f32MCLoadBias1); // 0-bit correspond to FILL_C_WITH_ZEROZ
    L(f32MCLoadBias1);
    Tbz(flags, 3, f32MCLoadBiasN); // 3-bit correspond to LOAD_BIAS_1
    Ld1rw(ARMCG::ZRegS(NUM_30), p5, ARMCG::AdrUimm(Ptr(biaPtr)));
    for (IntType i = 0; i < SplitSize; ++i) {
        AddPair(n, vlw, i * NUM_2, i * NUM_2 + NUM_1, NUM_30, NUM_30);
    }

    L(f32MCLoadBiasN);
    Tbz(flags, 5, f32MCLoadBiasM); // 5-bit correspond to LOAD_BIAS_N
    Ld1w(ARMCG::ZRegS(NUM_30), p1, Ptr(biaPtrCopy));
    if (n > vlw) {
        Ld1w(ARMCG::ZRegS(NUM_31), p3, Ptr(biaPtrCopy, NUM_1, ARMCG::ExtMod::MUL_VL));
    }
    for (IntType i = 0; i < SplitSize; ++i) {
        AddPair(n, vlw, i * NUM_2, i * NUM_2 + NUM_1, NUM_30, NUM_31);
    }

    L(f32MCLoadBiasM);
    Tbz(flags, 4, f32MCLoadBiasMN); // 4-bit correspond to LOAD_BIAS_M
    for (IntType i = 0; i < SplitSize; ++i) {
        if (i != 0) {
            Add(biaPtrCopy, biaPtrCopy, static_cast<std::uint32_t>(sizeof(BiaType)));
        }
        Ld1rw(ARMCG::ZRegS(NUM_30), p5, ARMCG::AdrUimm(Ptr(biaPtrCopy)));
        AddPair(n, vlw, i * NUM_2, i * NUM_2 + NUM_1, NUM_30, NUM_30);
    }

    L(f32MCLoadBiasMN);
    Tbz(flags, 6, f32MCLoadC); // 6-bit correspond to LOAD_BIAS_MN
    for (IntType i = 0; i < SplitSize; ++i) {
        if (i != 0) {
            Add(biaPtrCopy, biaPtrCopy, biaOffset);
        }
        Ld1w(ARMCG::ZRegS(NUM_30), p1, Ptr(biaPtrCopy));
        Fadd(ARMCG::ZRegS(i * NUM_2), p1, ARMCG::ZRegS(NUM_30));
        if (n > vlw) {
            Ld1w(ARMCG::ZRegS(NUM_31), p3, Ptr(biaPtrCopy, NUM_1, ARMCG::ExtMod::MUL_VL));
            Fadd(ARMCG::ZRegS(i * NUM_2 + NUM_1), p3, ARMCG::ZRegS(NUM_31));
        }
    }

    L(f32MCLoadC);
    Tbz(flags, 1, f32MCMullC); // 1-bit correspond to LOAD_C
    for (IntType i = 0; i < SplitSize; ++i) {
        if (i != 0) {
            Add(dstPtrCopy, dstPtrCopy, dstOffset);
        }
        Ld1w(ARMCG::ZRegS(NUM_30), p1, Ptr(dstPtrCopy));
        Fadd(ARMCG::ZRegS(i * NUM_2), p1, ARMCG::ZRegS(NUM_30));
        if (n > vlw) {
            Ld1w(ARMCG::ZRegS(NUM_31), p3, Ptr(dstPtrCopy, NUM_1, ARMCG::ExtMod::MUL_VL));
            Fadd(ARMCG::ZRegS(i * NUM_2 + NUM_1), p3, ARMCG::ZRegS(NUM_31));
        }
    }

    L(f32MCMullC);
    Tbz(flags, 2, f32MCAddBias1); // 2-bit correspond to MULL_C
    Dup(ARMCG::ZRegS(NUM_31), betaW);
    for (IntType i = 0; i < SplitSize; ++i) {
        if (i != 0) {
            Add(dstPtrCopy, dstPtrCopy, dstOffset);
        }
        Ld1w(ARMCG::ZRegS(NUM_29), p1, Ptr(dstPtrCopy));
        MulAdd(i * NUM_2, NUM_29, NUM_1, NUM_31);
        if (n > vlw) {
            Ld1w(ARMCG::ZRegS(NUM_30), p3, Ptr(dstPtrCopy, NUM_1, ARMCG::ExtMod::MUL_VL));
            MulAdd(i * NUM_2 + NUM_1, NUM_30, NUM_3, NUM_31);
        }
    }

    L(f32MCAddBias1);
    Tbz(flags, 7, f32MCAddBiasN); // 7-bit correspond to Add_BIAS_1
    Ld1rw(ARMCG::ZRegS(NUM_30), p5, ARMCG::AdrUimm(Ptr(biaPtr)));
    for (IntType i = 0; i < SplitSize; ++i) {
        AddPair(n, vlw, i * NUM_2, i * NUM_2 + NUM_1, NUM_30, NUM_30);
    }

    L(f32MCAddBiasN);
    Tbz(flags, 9, f32MCAddBiasM); // 9-bit correspond to Add_BIAS_N
    Ld1w(ARMCG::ZRegS(NUM_30), p1, Ptr(biaPtrCopy));
    if (n > vlw) {
        Ld1w(ARMCG::ZRegS(NUM_31), p3, Ptr(biaPtrCopy, NUM_1, ARMCG::ExtMod::MUL_VL));
    }
    for (IntType i = 0; i < SplitSize; ++i) {
        AddPair(n, vlw, i * NUM_2, i * NUM_2 + NUM_1, NUM_30, NUM_31);
    }

    L(f32MCAddBiasM);
    Tbz(flags, 8, f32MCAddBiasMN); // 8-bit correspond to ADD_BIAS_M
    for (IntType i = 0; i < SplitSize; ++i) {
        if (i != 0) {
            Add(biaPtrCopy, biaPtrCopy, static_cast<std::uint32_t>(sizeof(BiaType)));
        }

        Ld1rw(ARMCG::ZRegS(NUM_30), p5, ARMCG::AdrUimm(Ptr(biaPtrCopy)));
        AddPair(n, vlw, i * NUM_2, i * NUM_2 + NUM_1, NUM_30, NUM_30);
    }

    L(f32MCAddBiasMN);
    Tbz(flags, 10, f32MCInitDone); // 10-bit correspond to ADD_BIAS_MN
    for (IntType i = 0; i < SplitSize; ++i) {
        if (i != 0) {
            Add(biaPtrCopy, biaPtrCopy, biaOffset);
        }
        Ld1w(ARMCG::ZRegS(NUM_30), p1, Ptr(biaPtrCopy));
        Fadd(ARMCG::ZRegS(i * NUM_2), p1, ARMCG::ZRegS(NUM_30));
        if (n > vlw) {
            Ld1w(ARMCG::ZRegS(NUM_31), p3, Ptr(biaPtrCopy, NUM_1, ARMCG::ExtMod::MUL_VL));
            Fadd(ARMCG::ZRegS(i * NUM_2 + NUM_1), p3, ARMCG::ZRegS(NUM_31));
        }
    }

    L(f32MCInitDone);
    Tbz(flags, 11, f32MCSkipPO); // 11-bit correspond to APPLY_POST_OPS
    IntType size = SplitSize;
    if (n > vlw) {
        size += SplitSize;
    }
    std::vector<SizeType> dstRegsIDs(size);
    FillDstRegs(SplitSize, n, vlw, dstRegsIDs);
    for (std::uint32_t i = 0; i < size; ++i) {
        KUDNN_JIT_Generator_With_Attributes::TakeVectorRegister(dstRegsIDs[i]);
    }
    this->ApplyPostOps(dstRegsIDs, Element::MatchType<DstType>());
    for (std::uint32_t i = 0; i < size; ++i) {
        KUDNN_JIT_Generator_With_Attributes::FreeVectorRegister(dstRegsIDs[i]);
    }
    L(f32MCSkipPO);
}

void MCF32Gen::FillDstRegs(IntType sizeRegsIds, IntType n, IntType vl, std::vector<SizeType>& dstRegsIDs)
{
    for (IntType i = 0; i < sizeRegsIds; ++i) {
        if (n > vl) {
            dstRegsIDs[i * NUM_2] = i * NUM_2;
            dstRegsIDs[i * NUM_2 + NUM_1] = i * NUM_2 + NUM_1;
        } else {
            dstRegsIDs[i] = i * NUM_2;
        }
    }
}

void MCF32Gen::StoreDstRegistersPartSplit(IntType SplitSize, IntType n, IntType vl)
{
    for (IntType i = 0; i < SplitSize; ++i) {
        if (i != 0) {
            Add(dstPtrCopy, dstPtrCopy, dstOffset);
        }
        St1w(ARMCG::ZRegS(i * NUM_2), p1, Ptr(dstPtrCopy));
        if (n > vl) {
            St1w(ARMCG::ZRegS(i * NUM_2 + NUM_1), p3, Ptr(dstPtrCopy, NUM_1, ARMCG::ExtMod::MUL_VL));
        }
    }
}

void MCF32Gen::GenerateKLoop(
    IntType m, IntType n, IntType k, IntType vl)
{
    ARMCG::Label kLoopHead;
    Mov(kTmpCopy, kVal);
    Subs(kTmpCopy, kTmpCopy, K_LOOP_UNROLL);
    // p0, p2 - for SRC, p1, p3 - for WEI
    // z0, z2, z4, z6 - for SRC, z1, z3, z5, z7 - for WEI
    // SRC and WEI data is prepacked, so it's continious in memory.
    // K unroll by 2 requires 4 elements from SRC,  4 from WEI
    // Load src data
    Ld1w(ARMCG::ZRegS(NUM_0), p0, Ptr(srcPtrCopy));
    if (m > vl) {
        Ld1w(ARMCG::ZRegS(NUM_2), p2, Ptr(srcPtrCopy, NUM_1, ARMCG::ExtMod::MUL_VL));
    }
    // Load wei data
    Ld1w(ARMCG::ZRegS(NUM_1), p1, Ptr(weiPtrCopy));
    if (n > vl) {
        Ld1w(ARMCG::ZRegS(NUM_3), p3, Ptr(weiPtrCopy, NUM_1, ARMCG::ExtMod::MUL_VL));
    }
    if (k >= K_LOOP_UNROLL) {
        // Load src data for unrolled loop
        Ld1w(ARMCG::ZRegS(NUM_4), p0, Ptr(srcPtrCopy, NUM_2, ARMCG::ExtMod::MUL_VL));
        if (m > vl) {
            Ld1w(ARMCG::ZRegS(NUM_6), p2, Ptr(srcPtrCopy, NUM_3, ARMCG::ExtMod::MUL_VL));
        }
        // Load wei data unrolled loop
        Ld1w(ARMCG::ZRegS(NUM_5), p1, Ptr(weiPtrCopy, NUM_2, ARMCG::ExtMod::MUL_VL));
        if (n > vl) {
            Ld1w(ARMCG::ZRegS(NUM_7), p3, Ptr(weiPtrCopy, NUM_3, ARMCG::ExtMod::MUL_VL));
        }
        Addvl(srcPtrCopy, srcPtrCopy, KERNEL_M_SIZE * K_LOOP_UNROLL);
        Addvl(weiPtrCopy, weiPtrCopy, KERNEL_N_SIZE * K_LOOP_UNROLL);
        if (k > K_LOOP_UNROLL) {
            L(kLoopHead);
                Fmopa(ARMCG::ZARegS(NUM_0), p0, p1, ARMCG::ZRegS(NUM_0), ARMCG::ZRegS(NUM_1));
                Subs(kTmpCopy, kTmpCopy, K_LOOP_UNROLL);
                if (n > vl) {
                    Fmopa(ARMCG::ZARegS(NUM_1), p0, p3, ARMCG::ZRegS(NUM_0), ARMCG::ZRegS(NUM_3));
                }
                Ld1w(ARMCG::ZRegS(NUM_0), p0, Ptr(srcPtrCopy));
                if (m > vl) {
                    Fmopa(ARMCG::ZARegS(NUM_2), p2, p1, ARMCG::ZRegS(NUM_2), ARMCG::ZRegS(NUM_1));
                }
                Ld1w(ARMCG::ZRegS(NUM_1), p1, Ptr(weiPtrCopy));
                if ((m > vl) && (n > vl)) {
                    Fmopa(ARMCG::ZARegS(NUM_3), p2, p3, ARMCG::ZRegS(NUM_2), ARMCG::ZRegS(NUM_3));
                }
                if (m > vl) {
                    Ld1w(ARMCG::ZRegS(NUM_2), p2, Ptr(srcPtrCopy, NUM_1, ARMCG::ExtMod::MUL_VL));
                }
                if (n > vl) {
                    Ld1w(ARMCG::ZRegS(NUM_3), p3, Ptr(weiPtrCopy, NUM_1, ARMCG::ExtMod::MUL_VL));
                }

                Fmopa(ARMCG::ZARegS(NUM_0), p0, p1, ARMCG::ZRegS(NUM_4), ARMCG::ZRegS(NUM_5));
                if (n > vl) {
                    Fmopa(ARMCG::ZARegS(NUM_1), p0, p3, ARMCG::ZRegS(NUM_4), ARMCG::ZRegS(NUM_7));
                }
                Ld1w(ARMCG::ZRegS(NUM_4), p0, Ptr(srcPtrCopy, NUM_2, ARMCG::ExtMod::MUL_VL));
                if (m > vl) {
                    Fmopa(ARMCG::ZARegS(NUM_2), p2, p1, ARMCG::ZRegS(NUM_6), ARMCG::ZRegS(NUM_5));
                }
                Ld1w(ARMCG::ZRegS(NUM_5), p1, Ptr(weiPtrCopy, NUM_2, ARMCG::ExtMod::MUL_VL));
                if ((m > vl) && (n > vl)) {
                    Fmopa(ARMCG::ZARegS(NUM_3), p2, p3, ARMCG::ZRegS(NUM_6), ARMCG::ZRegS(NUM_7));
                }
                if (m > vl) {
                    Ld1w(ARMCG::ZRegS(NUM_6), p2, Ptr(srcPtrCopy, NUM_3, ARMCG::ExtMod::MUL_VL));
                }
                if (n > vl) {
                    Ld1w(ARMCG::ZRegS(NUM_7), p3, Ptr(weiPtrCopy, NUM_3, ARMCG::ExtMod::MUL_VL));
                }

                Addvl(srcPtrCopy, srcPtrCopy, KERNEL_M_SIZE * K_LOOP_UNROLL);
                Addvl(weiPtrCopy, weiPtrCopy, KERNEL_N_SIZE * K_LOOP_UNROLL);

            B(ARMCG::Cond::GT, kLoopHead);
        }
    }

    Fmopa(ARMCG::ZARegS(NUM_0), p0, p1, ARMCG::ZRegS(NUM_0), ARMCG::ZRegS(NUM_1));
    if (n > vl) {
        Fmopa(ARMCG::ZARegS(NUM_1), p0, p3, ARMCG::ZRegS(NUM_0), ARMCG::ZRegS(NUM_3));
    }
    if (m > vl) {
        Fmopa(ARMCG::ZARegS(NUM_2), p2, p1, ARMCG::ZRegS(NUM_2), ARMCG::ZRegS(NUM_1));
    }
    if ((m > vl) && (n > vl)) {
        Fmopa(ARMCG::ZARegS(NUM_3), p2, p3, ARMCG::ZRegS(NUM_2), ARMCG::ZRegS(NUM_3));
    }
    if (k % K_LOOP_UNROLL == 0) {
        Fmopa(ARMCG::ZARegS(NUM_0), p0, p1, ARMCG::ZRegS(NUM_4), ARMCG::ZRegS(NUM_5));
        if (n > vl) {
            Fmopa(ARMCG::ZARegS(NUM_1), p0, p3, ARMCG::ZRegS(NUM_4), ARMCG::ZRegS(NUM_7));
        }
        if (m > vl) {
            Fmopa(ARMCG::ZARegS(NUM_2), p2, p1, ARMCG::ZRegS(NUM_6), ARMCG::ZRegS(NUM_5));
        }
        if ((m > vl) && (n > vl)) {
            Fmopa(ARMCG::ZARegS(NUM_3), p2, p3, ARMCG::ZRegS(NUM_6), ARMCG::ZRegS(NUM_7));
        }
    }
}

MatMulImplGenerator<MCF32Gen>::DefaultImpl::DefaultImpl(
    const GemmInfo &task, IntType nThreads) : MatMulImplBase(task, nThreads)
{
    VERBOSE_PRINT;
    UpdateExecParams();
    Initialize();
}

std::uint32_t MatMulImplGenerator<MCF32Gen>::DefaultImpl::GetKernelMSize() noexcept
{
    return MCF32Gen::KERNEL_M_SIZE * svcntw();
}

std::uint32_t MatMulImplGenerator<MCF32Gen>::DefaultImpl::GetKernelNSize() noexcept
{
    return MCF32Gen::KERNEL_N_SIZE * svcntw();
}

IntType MatMulImplGenerator<MCF32Gen>::DefaultImpl::GetKBlockSize() const noexcept
{
    IntType k = taskExec.srcInfo.GetDims()[taskExec.srcInfo.GetNumDims() - NUM_1];
    IntType defKblock = 512; // 512 is experimentally measured constant to obtain better perf
    IntType numKblocks = Math::CeilDiv(k, defKblock);
    IntType residBlock = k % defKblock;
    IntType kBlock = k;
    if (k > NUM_3 * defKblock) {
        kBlock = defKblock + Math::CeilDiv(residBlock, numKblocks - NUM_1);
    } else if (k > defKblock) {
        kBlock = Math::CeilDiv(k, NUM_2);
    }
    kBlock = Math::CeilDiv(k, NUM_2) * NUM_2;
    return kBlock;
}

IntType MatMulImplGenerator<MCF32Gen>::DefaultImpl::GetMBlockSize() const noexcept
{
    return 2048; // 2048 is experimentally measured constant to obtain better perf
}

IntType MatMulImplGenerator<MCF32Gen>::DefaultImpl::GetNBlockSize() const noexcept
{
    // all the constants are experimentally measured to obtain better perf
    if (nFullBlockSizePerThread > 4096) { // 4096 is the largest thresholder
        return 160;        // 160 is the NB blocksize when n larger than 4096
    } else if (nFullBlockSizePerThread > 256) { // 256 is the second largest thresholder
        return 128;       // 128 is the NB blocksize when n larger than 256
    } else if (nFullBlockSizePerThread > 144) { // 144 is the third largest thresholder
        return 96;        // 96 is the NB blocksize when n larger than 144
    } else {
        return 64;        // 64 is the NB blocksize when n less or equal than 144
    }
}

void MatMulImplGenerator<MCF32Gen>::DefaultImpl::InitializePackingPolicy()
{
    packingPolicy = PackingPolicy::NO_SYNC;
    InitializePackingPolicyFromEnv();
    if ((packingPolicy == PackingPolicy::SYNC_ACC_DIM_SRC) ||
        (packingPolicy == PackingPolicy::SYNC_ACC_DIM_WEI) ||
        (packingPolicy == PackingPolicy::SYNC_ACC_DIM_SRC_WEI)) {
        throw Service::LogicError("Matmul: Current packing policy not supported for this precision");
    }
}

void MatMulImplGenerator<MCF32Gen>::DefaultImpl::InitializeSingleCodegen(
    IntType id, IntType m, IntType n, IntType k,
    IntType currentM, IntType currentN, IntType currentK,
    SizeType offsetSrc, SizeType offsetWei, SizeType offsetDst, SizeType offsetBia,
    bool wereSrcWeiSwapped, const Attributes &attributes, bool hasAlpha)
{
    Service::Unused(offsetSrc, offsetWei, offsetDst, offsetBia, wereSrcWeiSwapped, hasAlpha);
    if ((currentM == 0) || (currentN == 0) || (currentK == 0)) {
        return;
    }
    if ((currentM <= m) && (currentN <= n) && (currentK <= k)) {
        CodeGenParams<MCF32Gen> currTask{currentM, currentN, currentK, false, attributes};
        if (codegens.find(currTask) == codegens.end()) {
            codegens.insert({currTask, std::make_shared<MCF32Gen>(
                currentM, currentN, currentK, attributes)});
        }
        codegens[currTask]->Ready();
        matmulCodegenKernels[id] = codegens[currTask]->template GetCode<MCF32CodegenFuncPtr>();
    }
}


void MatMulImplGenerator<MCF32Gen>::DefaultImpl::InitializeReorderCodeGens(IntType m,
    IntType n, IntType k, IntType mFull, IntType nFull)
{
    IntType vl = svcntw();
    IntType mFullThreadBlock = mFull;
    IntType mResidThreadBlock = m % mFullThreadBlock;
    IntType nFullThreadBlock = nFull;
    IntType nResidThreadBlock = n % nFullThreadBlock;

    IntType mSubBlockExecFull = MCF32Gen::KERNEL_M_SIZE * vl;
    IntType mSubBlockExecResid1 = mFullThreadBlock % mSubBlockExecFull;
    IntType mSubBlockExecResid2 = mResidThreadBlock % mSubBlockExecFull;

    IntType nSubBlockExecFull = MCF32Gen::KERNEL_N_SIZE * vl;
    IntType nSubBlockExecResid1 = nFullThreadBlock % nSubBlockExecFull;
    IntType nSubBlockExecResid2 = nResidThreadBlock % nSubBlockExecFull;

    IntType kBlockExecFull = GetKBlockSize();
    IntType kBlockExecResid = k % GetKBlockSize();

    CodeGenWrapper<FastReorderKernelCodeGen<float, float>> reorderFBFKNA; // full Block full K NoAlpha
    reorderFBFKNA.Initialize(static_cast<SizeType>(vl),
                             static_cast<SizeType>(kBlockExecFull), false);

    CodeGenWrapper<FastReorderKernelCodeGen<float, float>> reorderRN1FKNA; // resid N 1 full K NoAlpha
    reorderRN1FKNA.Initialize(static_cast<SizeType>(nSubBlockExecResid1 % vl),
                              static_cast<SizeType>(kBlockExecFull), false);

    CodeGenWrapper<FastReorderKernelCodeGen<float, float>> reorderRN2FKNA; // resid N 2 full K NoAlpha
    reorderRN2FKNA.Initialize(static_cast<SizeType>(nSubBlockExecResid2 % vl),
                              static_cast<SizeType>(kBlockExecFull), false);

    CodeGenWrapper<FastReorderKernelCodeGen<float, float>> reorderRM1FKNA; // resid M 1 full K NoAlpha
    reorderRM1FKNA.Initialize(static_cast<SizeType>(mSubBlockExecResid1 % vl),
                              static_cast<SizeType>(kBlockExecFull), false);
    
    CodeGenWrapper<FastReorderKernelCodeGen<float, float>> reorderRM2FKNA; // resid M 2 full K NoAlpha
    reorderRM2FKNA.Initialize(static_cast<SizeType>(mSubBlockExecResid2 % vl),
                              static_cast<SizeType>(kBlockExecFull), false);

    CodeGenWrapper<FastReorderKernelCodeGen<float, float>> reorderFBRKNA; // full Block resid K NoAlpha
    reorderFBRKNA.Initialize(static_cast<SizeType>(vl),
                             static_cast<SizeType>(kBlockExecResid), false);
                            
    CodeGenWrapper<FastReorderKernelCodeGen<float, float>> reorderRN1RKNA; // resid N 1 resid K NoAlpha
    reorderRN1RKNA.Initialize(static_cast<SizeType>(nSubBlockExecResid1 % vl),
                              static_cast<SizeType>(kBlockExecResid), false);
    
    CodeGenWrapper<FastReorderKernelCodeGen<float, float>> reorderRN2RKNA; // resid N 2 resid K NoAlpha
    reorderRN2RKNA.Initialize(static_cast<SizeType>(nSubBlockExecResid2 % vl),
                              static_cast<SizeType>(kBlockExecResid), false);
                               
    CodeGenWrapper<FastReorderKernelCodeGen<float, float>> reorderRM1RKNA; // resid M 1 resid K NoAlpha
    reorderRM1RKNA.Initialize(static_cast<SizeType>(mSubBlockExecResid1 % vl),
                              static_cast<SizeType>(kBlockExecResid), false);
    
    CodeGenWrapper<FastReorderKernelCodeGen<float, float>> reorderRM2RKNA; // resid M 2 resid K NoAlpha
    reorderRM2RKNA.Initialize(static_cast<SizeType>(mSubBlockExecResid2 % vl),
                              static_cast<SizeType>(kBlockExecResid), false);

    reorderPackingCodegenKernels = {
        std::move(reorderFBFKNA), std::move(reorderRN1FKNA), std::move(reorderRN2FKNA),
        std::move(reorderRM1FKNA), std::move(reorderRM2FKNA),
        std::move(reorderFBRKNA), std::move(reorderRN1RKNA), std::move(reorderRN2RKNA),
        std::move(reorderRM1RKNA), std::move(reorderRM2RKNA)
    };
}

IntType MatMulImplGenerator<MCF32Gen>::DefaultImpl::GetThreadBufSize(
    IntType m, IntType n, IntType k, MatrixKind mKind) const noexcept
{
    IntType vlw = svcntw();
    IntType mRoundedUp = Math::CeilDiv(m, MCF32Gen::KERNEL_M_SIZE * vlw) * MCF32Gen::KERNEL_M_SIZE * vlw;
    IntType nRoundedUp = Math::CeilDiv(n, MCF32Gen::KERNEL_N_SIZE * vlw) * MCF32Gen::KERNEL_N_SIZE * vlw;
    IntType kRoundedUp = Math::CeilDiv(k, MCF32Gen::K_LOOP_UNROLL) * MCF32Gen::K_LOOP_UNROLL;
    if (mKind == MatrixKind::SRC) {
        return static_cast<IntType>(std::min(GetKBlockSize(), kRoundedUp) *
            std::min(GetMBlockSize(), mRoundedUp) * sizeof(SrcType));
    } else {
        return static_cast<IntType>(std::min(GetKBlockSize(), kRoundedUp) *
            std::min(GetNBlockSize(), nRoundedUp) * sizeof(WeiType));
    }
}

void MatMulImplGenerator<MCF32Gen>::DefaultImpl::ReserveImplSpecificParams() {}

std::vector<std::pair<Shape, Shape>> MatMulImplGenerator<MCF32Gen>::DefaultImpl::SplitWork()
{
    return CalculateWorkOnThread<DefaultImpl>(taskExec.dstInfo.GetDims(), nThreadsExec, dstLayout2D == Layout::BA);
}

// used for packing both Src and Wei matrices
void MatMulImplGenerator<MCF32Gen>::DefaultImpl::PackSimilarLt(IntType m, IntType k,
    const float *src, IntType offset, float *buffer, IntType bufferOffset)
{
    IntType vl = svcntw();
    svbool_t ptrue = svptrue_b32();
    for (IntType kIdx = 0; kIdx < k; ++kIdx) {
        for (IntType mIdx = 0; mIdx < m; mIdx += bufferOffset) {
            svbool_t p0 = svwhilelt_b32(mIdx + 0 * vl, m);
            svfloat32_t a0 = svld1(p0, src + 0 * vl + mIdx + offset * kIdx);
            svbool_t p1 = svwhilelt_b32(mIdx + 1 * vl, m);
            svfloat32_t a1 = svld1(p1, src + 1 * vl + mIdx + offset * kIdx);
            svst1(ptrue, buffer + 0 * vl + bufferOffset * kIdx + k * mIdx, a0);
            svst1(ptrue, buffer + 1 * vl + bufferOffset * kIdx + k * mIdx, a1);
        }
    }
}

// used for packing both Src and Wei matrices
void MatMulImplGenerator<MCF32Gen>::DefaultImpl::PackDiffLt(
    IntType threadID, IntType m, IntType k, const float *src, IntType offset, float *buffer, IntType bufferOffset,
    PackingType packingType, IntType kBlockSize) const
{
    const IntType vl = svcntw();
    IntType idxK = (k == kBlockSize) ? 0 : 1;
    bool isM = (packingType != PackingType::PACK_SRC) ? false : true;
    for (IntType mIdx = m; mIdx > 0; mIdx -= bufferOffset) {
        if (mIdx <= vl) {
            IntType idxM = ((mIdx < vl) ? 1 : 0);
            IntType mIdxKernel = (idxM) * (1 + isM * 2 +
                (isM ? perThreadParams[threadID].isResidByM : perThreadParams[threadID].isResidByN));
            IntType packBColIdx = idxK * NUM_5 +  mIdxKernel;
            auto kernel = reorderPackingCodegenKernels[packBColIdx]
                .template GetCode<void(*)(const float*, float*, IntType, IntType)>();
            kernel(src + (m - mIdx) * offset, buffer + k * (m - mIdx), offset * sizeof(float),
                   bufferOffset * sizeof(float));
        } else {
            IntType mIdxReorder1 = vl;
            IntType mIdxReorder2 = std::min(vl, mIdx - vl);
            IntType idxM1 = ((mIdxReorder1 < vl) ? 1 : 0);
            IntType idxM2 = ((mIdxReorder2 < vl) ? 1 : 0);
            IntType mIdxKernel1 = (idxM1) * (1 + isM * 2 +
                (isM ? perThreadParams[threadID].isResidByM : perThreadParams[threadID].isResidByN));
            IntType mIdxKernel2 = (idxM2) * (1 + isM * 2 +
                (isM ? perThreadParams[threadID].isResidByM : perThreadParams[threadID].isResidByN));
            IntType packBColIdx1 = idxK * NUM_5 + mIdxKernel1;
            IntType packBColIdx2 = idxK * NUM_5 + mIdxKernel2;
            auto kernel = reorderPackingCodegenKernels[packBColIdx1].
                template GetCode<void(*)(const float*, float*, IntType, IntType)>();
            kernel(src + (m - mIdx) * offset, buffer + k * (m - mIdx), offset * sizeof(float),
                   bufferOffset * sizeof(float));
            kernel = reorderPackingCodegenKernels[packBColIdx2].
                template GetCode<void(*)(const float*, float*, IntType, IntType)>();
            kernel(src + (m - mIdx + vl) * offset, buffer + k * (m - mIdx) + vl, offset * sizeof(float),
                   bufferOffset * sizeof(float));
        }
    }
}

void MatMulImplGenerator<MCF32Gen>::DefaultImpl::Pack(
    const void *originalMatrix, void *buffer, IntType threadID, IntType dstDim, IntType accumDim,
    IntType dstDimOffset, IntType accumDimOffset, IntType bufferOffset, IntType bufferStride, float alpha,
    MatrixKind mKind) const
{
    Service::Unused(bufferStride, alpha);
    auto vl = svcntw();
    Layout layout;
    IntType offset2D;
    auto initParams = [&] (Layout &layout_, IntType &offset2D_) {
        if (mKind == MatrixKind::SRC) {
            layout_ = perThreadParams[threadID].srcLayout2D;
            offset2D_ = perThreadParams[threadID].srcOffset2D;
        } else {
            layout_ = perThreadParams[threadID].weiLayout2D;
            offset2D_ = perThreadParams[threadID].weiOffset2D;
        }
    };
    initParams(layout, offset2D);
    if (mKind == MatrixKind::SRC) {
        const SrcType *srcExec = static_cast<const SrcType *>(originalMatrix);
        if (layout == Layout::COL_MAJOR) {
            srcExec += dstDimOffset + accumDimOffset * offset2D;
            PackSimilarLt(dstDim, accumDim, srcExec,
                offset2D, static_cast<SrcType *>(buffer) + bufferOffset,
                MCF32Gen::KERNEL_M_SIZE * vl);
        } else {
            srcExec += dstDimOffset * offset2D + accumDimOffset;
            PackDiffLt(threadID, dstDim, accumDim, srcExec,
                offset2D, static_cast<SrcType *>(buffer) + bufferOffset,
                MCF32Gen::KERNEL_M_SIZE * vl, PackingType::PACK_SRC,
                GetKBlockSize());
        }
    } else {
        const WeiType *weiExec = static_cast<const WeiType *>(originalMatrix);
        if (layout == Layout::ROW_MAJOR) {
            weiExec += dstDimOffset + accumDimOffset * offset2D;
            PackSimilarLt(dstDim, accumDim, weiExec,
                offset2D, static_cast<WeiType *>(buffer) + bufferOffset,
                MCF32Gen::KERNEL_N_SIZE * vl);
        } else {
            weiExec += dstDimOffset * offset2D + accumDimOffset;
            PackDiffLt(threadID, dstDim, accumDim, weiExec,
                offset2D, static_cast<WeiType *>(buffer) + bufferOffset,
                MCF32Gen::KERNEL_M_SIZE * vl, PackingType::PACK_WEI,
                GetKBlockSize());
        }
    }
}


void MatMulImplGenerator<MCF32Gen>::DefaultImpl::RunSequential2D(
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
    SrcType *srcBuffer = static_cast<SrcType *>(srcPackStrategy->GetPtr(tIDbatch * threadsM  + tIDm, tIDn, threadsN));
    WeiType *weiBuffer = static_cast<WeiType *>(weiPackStrategy->GetPtr(tIDbatch * threadsN  + tIDn, tIDm, threadsM));

    for (IntType mBlock = 0; mBlock < mFullBlockSizePerThread; mBlock += GetMBlockSize()) {
        auto mBlockExec = std::min(GetMBlockSize(), m - mBlock);
        IntType mIdx = (mBlockExec == GetMBlockSize()) ? 0 : 1;
        IntType mIdxKernel = mIdx * (perThreadParams[threadID].isResidByM + 1);
        for (IntType kBlock = 0; kBlock < k; kBlock += GetKBlockSize()) {
            auto kBlockExec = std::min(GetKBlockSize(), k - kBlock);
            IntType kIdxKernel = (kBlockExec == GetKBlockSize()) ? 0 : 1;
            bool lastKIter = (k - kBlock) <= GetKBlockSize();
            MatMulDstFlags biaFlagsExec = SetBiaFlagsExec(biaFlags, beta, kBlock, lastKIter);

            if (mBlock < m) {
                srcPackStrategy->Pack(*this, src, srcBuffer,
                    {mBlock, mBlockExec, GetKernelMSize()}, {kBlock, kBlockExec, 1},
                    threadsN, tIDn, threadID, alpha);
            }
            srcPackStrategy->Barrier(tIDbatch * threadsM + tIDm, tIDn, threadsN);
            for (IntType nBlock = 0; nBlock < nFullBlockSizePerThread; nBlock += GetNBlockSize()) {
                if ((nBlock < n) && (mBlock < m)) {
                    auto nBlockExec = std::min(GetNBlockSize(), n - nBlock);
                    IntType nIdx = (nBlockExec == GetNBlockSize()) ? 0 : 1;
                    IntType nIdxKernel = nIdx * (perThreadParams[threadID].isResidByN + 1);
                    weiPackStrategy->Pack(*this, wei, weiBuffer,
                        {nBlock, nBlockExec, GetKernelNSize()}, {kBlock, kBlockExec, 1},
                        threadsM, tIDm, threadID, alpha);
                    weiPackStrategy->Barrier(tIDbatch * threadsN + tIDn, tIDm, threadsM);
                    DstType *dstExec = static_cast<DstType *>(dst) + nBlock +
                        mBlock * perThreadParams[threadID].dstOffset2D;
                    const BiaType *biaExec = static_cast<const BiaType *>(bia) +
                        nBlock * biaNOffset + mBlock * perThreadParams[threadID].biaOffset2D;
                    auto kernel = matmulCodegenKernels[kIdxKernel * NUM_9 +
                        nIdxKernel * NUM_3 + mIdxKernel];
                    kernel(srcBuffer, weiBuffer, dstExec, biaExec, biaFlagsExec, alpha, beta,
                        mBlockExec, nBlockExec, kBlockExec, kBlockExec, kBlockExec,
                        perThreadParams[threadID].dstOffset2D * sizeof(DstType),
                        perThreadParams[threadID].biaOffset2D * sizeof(BiaType));
                    weiPackStrategy->Barrier(tIDbatch * threadsN + tIDn, tIDm, threadsM);
                }
            }
            srcPackStrategy->Barrier(tIDbatch * threadsM + tIDm, tIDn, threadsN);
        }
    }
}

void MatMulImplGenerator<MCF32Gen>::DefaultImpl::UpdateExecParams() noexcept
{
    VERBOSE_PRINT;
    Layout desiredSrcLayout = implTask.srcInfo.GetLayout();
    Layout desiredWeiLayout = implTask.weiInfo.GetLayout();
    Layout desiredDstLayout = implTask.dstInfo.GetLayout();
    Layout desiredBiaLayout = implTask.biaInfo.GetLayout();
    if (!IsLayoutAlikeAB(desiredSrcLayout) && !IsLayoutAlikeBA(desiredSrcLayout)) {
        desiredSrcLayout = GetPlaneLayoutBA(desiredSrcLayout);
    }
    if (!IsLayoutAlikeAB(desiredWeiLayout) && !IsLayoutAlikeBA(desiredWeiLayout)) {
        desiredWeiLayout = GetPlaneLayoutAB(desiredWeiLayout);
    }
    if (!IsLayoutAlikeAB(desiredDstLayout) && !IsLayoutAlikeBA(desiredDstLayout)) {
        desiredDstLayout = GetPlaneLayoutAB(desiredDstLayout);
    }
    if (implTask.biaInfo.GetTotalTensorSize() != 0) {
        desiredBiaLayout = desiredDstLayout;
    }
    InitializeTaskExec(desiredSrcLayout, desiredWeiLayout, desiredDstLayout, desiredBiaLayout,
        GetSrcType(), GetWeiType(), GetDstType(),
        (implTask.biaInfo.GetTotalTensorSize() != 0) ? GetBiaType() :
            static_cast<Element::TypeT>(implTask.biaInfo.GetType()));
}

MatMulDstFlags MatMulImplGenerator<MCF32Gen>::DefaultImpl::SetBiaFlagsExec(
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
            } else if (beta == 1.0f) {
                ret = MatMulDstFlags::LOAD_C;
            } else {
                ret = MatMulDstFlags::MULL_C;
            }
        } else { // bias
            if (beta == 0) {
                ret = biaF;
            } else if (beta == 1.0f) {
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
