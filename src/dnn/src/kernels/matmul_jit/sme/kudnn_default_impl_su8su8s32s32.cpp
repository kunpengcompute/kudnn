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
#include <type_traits>
#include <cstdint>

#include <arm_sve.h>

#include "kudnn.hpp"

#include "cg_codegen.hpp"

#include "kudnn_jit.hpp"
#include "kernels/matmul_jit/kudnn_gemm_jit.hpp"
#include "kernels/matmul_jit/kudnn_gemm_helpers.hpp"
#include "kernels/matmul_jit/kudnn_gemm_threading_helpers.hpp"
#include "kudnn_wrapper_helpers.hpp"
#include "kernels/reorder/sve/bf16any32/kudnn_fast_jit_impl_bfp16any32_bfp16any32.hpp"
#include "kernels/matmul_jit/sme/kudnn_default_impl_su8su8s32s32.hpp"

namespace KuDNN {
namespace Detail {

template <typename SrcType, typename WeiType, typename DstType, typename BiaType>
using MCSU8GenBase = MatMulKernelCodeGenBase<ARCH::MC, SrcType, WeiType, DstType, BiaType>;

template <typename SrcType, typename WeiType, typename DstType, typename BiaType>
MCSU8Gen<SrcType, WeiType, DstType, BiaType>::MatMulKernelDefaultCodeGen(
    IntType m, IntType n, IntType k,
    bool wereSrcWeiSwappedArg,
    const Attributes &attributesArg) : MCSU8GenBase<SrcType, WeiType, DstType, BiaType>(attributesArg),
    wereSrcWeiSwapped(wereSrcWeiSwappedArg)
{
    if ((m > 0) && (n > 0) && (k > 0)) {
        if (this->attributes.GetPostOps().Len() > 0) {
            // set reserved predicate registers
            for (std::uint32_t idx = 0; idx < NUM_6; ++idx) {
                KUDNN_JIT_Generator_With_Attributes::TakePredicateRegister(idx);
            }
            // set reserved scalar registers
            for (std::uint32_t idx = 0; idx < NUM_29; ++idx) {
                KUDNN_JIT_Generator_With_Attributes::TakeScalarRegister(idx);
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

        Lsl(mVal, mVal, NUM_2); // NUM_2 due to peculiarities of INT8 MOPA instruction
        Lsl(nVal, nVal, NUM_2); // NUM_2 due to peculiarities of INT8 MOPA instruction
        Mov(fOne, 0x3f800000); // 0x3f800000 is a bit representation of 1.0f
        // constants to perform continious stores from ZA tiles
        Mov(tmpLDStC0, NUM_0);
        Mov(tmpLDStC1, NUM_4);
        Mov(tmpLDStC2, NUM_8);
        Mov(tmpLDStC3, NUM_12);
        Ptrue(ARMCG::PRegS(NUM_5), ARMCG::Pattern::ALL);

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
                Sub(mLoopCtr, mLoopCtr, KERNEL_M_SIZE * vl * NUM_4); // NUM_4 due to peculiarities of INT8 MOPA instr
                Cmp(mLoopCtr, KERNEL_M_SIZE * vl * NUM_4); // NUM_4 due to peculiarities of INT8 MOPA instruction
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

template <typename SrcType, typename WeiType, typename DstType, typename BiaType>
void MCSU8Gen<SrcType, WeiType, DstType, BiaType>::GenerateMLoopBody(IntType mLim, IntType n, IntType k)
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
    Whilelt(ARMCG::PRegB(NUM_0), tmp, mLoopCtr);
    if (mLim > vl) {
        Incb(tmp);
        Whilelt(ARMCG::PRegB(NUM_2), tmp, mLoopCtr);
    }
    if (n > (KERNEL_N_SIZE * vl)) {
        ARMCG::Label nLoopHead;
        L(nLoopHead);
        {
            GenerateNLoopBody(mLim, KERNEL_N_SIZE * vl, k);
            Sub(nLoopCtr, nLoopCtr, KERNEL_N_SIZE * vl * NUM_4); // NUM_4 due to peculiarities of INT8 MOPA instruction
            Cmp(nLoopCtr, KERNEL_N_SIZE * vl * NUM_4); // NUM_4 due to peculiarities of INT8 MOPA instruction
            Sub(tmpNMinusNLoopCtr, nVal, nLoopCtr);
        }
        B(ARMCG::Cond::GT, nLoopHead);
    }
    GenerateNLoopBody(mLim, nResid, k);
}

template <typename SrcType, typename WeiType, typename DstType, typename BiaType>
void MCSU8Gen<SrcType, WeiType, DstType, BiaType>::GenerateNLoopBody(
    IntType mLim, IntType nLim, IntType k)
{
    IntType vl = svcntw();
    // Initialize src pointer
    Add(tmp, kVal, NUM_3);
    Lsr(tmp, tmp, NUM_2); // NUM_2 is due to FP16 MOPA peculiarities
    Lsl(tmp, tmp, NUM_2); // NUM_2 is due to FP16 MOPA peculiarities
    Mul(tmp, tmpMMinusMLoopCtr, tmp);
    Lsr(tmp, tmp, NUM_2); // NUM_2 is due to INT8 MOPA peculiarities
    Add(srcPtrCopy, srcPtr, tmp,
        ARMCG::ShMod::LSL, Math::Log2(sizeof(SrcType)));
    // initialize predicates for WEI loads
    Mov(tmp, 0);
    // p0, p2 used for SRC, p1, p3 used for WEI
    Whilelt(ARMCG::PRegB(NUM_1), tmp, nLoopCtr);
    if (nLim > vl) {
        Incb(tmp);
        Whilelt(ARMCG::PRegB(NUM_3), tmp, nLoopCtr);
    }
    // Initialize wei pointer
    Add(tmp, kVal, NUM_3);
    Lsr(tmp, tmp, NUM_2); // NUM_2 is due to FP16 MOPA peculiarities
    Lsl(tmp, tmp, NUM_2); // NUM_2 is due to FP16 MOPA peculiarities
    Mul(tmp, tmpNMinusNLoopCtr, tmp);
    Lsr(tmp, tmp, NUM_2); // NUM_2 is due to INT8 MOPA peculiarities
    Add(weiPtrCopy, weiPtr, tmp,
        ARMCG::ShMod::LSL, Math::Log2(sizeof(WeiType)));
 
    // Initialize dst pointers
    Mul(tmp, tmpMMinusMLoopCtr, dstOffset);
    Lsr(tmp, tmp, NUM_2); // NUM_2 is due to INT8 MOPA peculiarities
    // Row-major format of dst
    Add(dstPtrCopy, dstPtr, tmp); // Add stride for M
    Add(dstPtrCopy, dstPtrCopy, tmpNMinusNLoopCtr); // Add stride for N
    // tmpNMinusNLoopCtr is already multiplied by 4
    Mov(dstPtrLoopCopy, dstPtrCopy);
    // Initialize bia pointer
    Mul(tmp, tmpMMinusMLoopCtr, biaOffset);
    Lsr(tmp, tmp, NUM_2); // NUM_2 is due to INT8 MOPA peculiarities
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
    Add(biaPtrCopy, biaPtrCopy, tmpNMinusNLoopCtr); // Add stride for N
    // tmpNMinusNLoopCtr is already multiplied by 4
    L(notIncrementBiasByN);

    // fill dst registers with zeros
    Zero({ARMCG::ZARegS(NUM_0), ARMCG::ZARegS(NUM_1), ARMCG::ZARegS(NUM_2), ARMCG::ZARegS(NUM_3)});
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

template <typename SrcType, typename WeiType, typename DstType, typename BiaType>
void MCSU8Gen<SrcType, WeiType, DstType, BiaType>::FillDstRegs(
    IntType sizeRegsIds, IntType n, IntType vl, std::vector<SizeType>& dstRegsIDs)
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

template <typename SrcType, typename WeiType, typename DstType, typename BiaType>
void MCSU8Gen<SrcType, WeiType, DstType, BiaType>::MoveDstRegistersPartSplit(
    IntType SplitSize, IntType n, IntType vl, IntType ZARegIdx1, IntType ZARegIdx2, IntType ZARegOffset)
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

template <typename SrcType, typename WeiType, typename DstType, typename BiaType>
void MCSU8Gen<SrcType, WeiType, DstType, BiaType>::StoreDstRegistersPartSplit(
    IntType SplitSize, IntType n, IntType vl)
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

template <typename SrcType, typename WeiType, typename DstType, typename BiaType>
void MCSU8Gen<SrcType, WeiType, DstType, BiaType>::BiasAndC(IntType SplitSize, IntType n, IntType vlw)
{
    ARMCG::Label applyf32Conv;
    ARMCG::Label nof32Conv;
    ARMCG::Label ret;

    Mov(tmpW, 0x3f800000); // 0x3f800000 is a bit representation of 1.0f
    // alpha
    // 1.0f no convert
    Cmp(alphaW, tmpW);
    Bne(applyf32Conv);

    // beta
    // 1.0f, 0.0f no convert to f32
    Mov(tmpW, 0x00000000);
    Cmp(betaW, tmpW);
    Beq(nof32Conv);

    Mov(tmpW, 0x3f800000); // 0x3f800000 is a bit representation of 1.0f
    Cmp(betaW, tmpW);
    Beq(nof32Conv);

    L(applyf32Conv);
    CvtAndInit(true, SplitSize, n, vlw);
    B(ret);

    L(nof32Conv);
    CvtAndInit(false, SplitSize, n, vlw);
    L(ret);
}


template <typename SrcType, typename WeiType, typename DstType, typename BiaType>
void MCSU8Gen<SrcType, WeiType, DstType, BiaType>::CvtAdd(bool applyConvert, IntType dstIdx, IntType argIdx,
    IntType pregIdx)
{
    ARMCG::Label noConv;
    if (applyConvert) {
        // Convert to f32 on first K iter
        Tbz(flags, 12, noConv); // 12-bit correspond to FIRST_K_ITER
            Scvtf(ARMCG::ZRegS(argIdx), p5, ARMCG::ZRegS(argIdx));
        L(noConv);
        Fadd(ARMCG::ZRegS(dstIdx), ARMCG::PRegB(pregIdx), ARMCG::ZRegS(argIdx));
    } else {
        Add(ARMCG::ZRegS(dstIdx), ARMCG::ZRegS(dstIdx), ARMCG::ZRegS(argIdx));
    }
}

template <typename SrcType, typename WeiType, typename DstType, typename BiaType>
void MCSU8Gen<SrcType, WeiType, DstType, BiaType>::CvtMulAdd(IntType dstIdx, IntType argIdx,
    IntType pregIdx, IntType betaIdx)
{
    Scvtf(ARMCG::ZRegS(argIdx), p5, ARMCG::ZRegS(argIdx));
    Fmul(
        ARMCG::ZRegS(argIdx),
        ARMCG::ZRegS(argIdx),
        ARMCG::ZRegS(betaIdx)
    );
    Fadd(ARMCG::ZRegS(dstIdx), ARMCG::PRegB(pregIdx), ARMCG::ZRegS(argIdx));
}

template <typename SrcType, typename WeiType, typename DstType, typename BiaType>
void MCSU8Gen<SrcType, WeiType, DstType, BiaType>::AddPair(IntType n, IntType vlw, bool applyConvert,
    IntType dstIdx1, IntType dstIdx2, IntType argIdx1, IntType argIdx2)
{
    if (applyConvert) {
        Fadd(ARMCG::ZRegS(dstIdx1), p1, ARMCG::ZRegS(argIdx1));
        if (n > vlw) {
            Fadd(ARMCG::ZRegS(dstIdx2), p3, ARMCG::ZRegS(argIdx2));
        }
    } else {
        Add(ARMCG::ZRegS(dstIdx1), ARMCG::ZRegS(dstIdx1), ARMCG::ZRegS(argIdx1));
        if (n > vlw) {
            Add(ARMCG::ZRegS(dstIdx2), ARMCG::ZRegS(dstIdx2), ARMCG::ZRegS(argIdx2));
        }
    }
}

template <typename SrcType, typename WeiType, typename DstType, typename BiaType>
void MCSU8Gen<SrcType, WeiType, DstType, BiaType>::CvtAndInit(
    bool applyConvert, IntType SplitSize, IntType n, IntType vlw)
{
    // Initialize Dst matrix registers
    ARMCG::Label su8MCloadC;
    ARMCG::Label su8MCmullC;
    ARMCG::Label su8MCloadBias1;
    ARMCG::Label su8MCloadBiasM;
    ARMCG::Label su8MCloadBiasN;
    ARMCG::Label su8MCloadBiasMN;
    ARMCG::Label su8MCaddBias1;
    ARMCG::Label su8MCaddBiasM;
    ARMCG::Label su8MCaddBiasN;
    ARMCG::Label su8MCaddBiasMN;
    ARMCG::Label su8MCinitDone;
    ARMCG::Label skipPO;

    if (applyConvert) {
        for (IntType i = 0; i < SplitSize; ++i) {
            Scvtf(ARMCG::ZRegS(i * NUM_2), p5, ARMCG::ZRegS(i * NUM_2));
            if (n > vlw) {
                Scvtf(ARMCG::ZRegS(i * NUM_2 + NUM_1), p5, ARMCG::ZRegS(i * NUM_2 + NUM_1));
            }
        }
        // Load alpha
        Mov(tmpW, 0x3f800000); // 0x3f800000 is a bit representation of 1.0f

        Cmp(alphaW, tmpW);
        Beq(su8MCloadBias1);
        Dup(ARMCG::ZRegS(NUM_30), alphaW);
        for (IntType i = 0; i < SplitSize; ++i) {
            Fmul(ARMCG::ZRegS(i * NUM_2), ARMCG::ZRegS(i * NUM_2), ARMCG::ZRegS(NUM_30));
            if (n > vlw) {
                Fmul(ARMCG::ZRegS(i * NUM_2 + NUM_1), ARMCG::ZRegS(i * NUM_2 + NUM_1), ARMCG::ZRegS(NUM_30));
            }
        }
    }

    Tbz(flags, 0, su8MCloadBias1); // 0-bit correspond to FILL_C_WITH_ZEROZ
    L(su8MCloadBias1);
    Tbz(flags, 3, su8MCloadBiasN); // 3-bit correspond to LOAD_BIAS_1
    Ld1rw(ARMCG::ZRegS(NUM_30), p5, ARMCG::AdrUimm(Ptr(biaPtr)));
    if (applyConvert) {
        Scvtf(ARMCG::ZRegS(NUM_30), p5, ARMCG::ZRegS(NUM_30));
    }
    for (IntType i = 0; i < SplitSize; ++i) {
        AddPair(n, vlw, applyConvert, i * NUM_2, i * NUM_2 + NUM_1, NUM_30, NUM_30);
    }

    L(su8MCloadBiasN);
    Tbz(flags, 5, su8MCloadBiasM); // 5-bit correspond to LOAD_BIAS_N
    Ld1w(ARMCG::ZRegS(NUM_30), p1, Ptr(biaPtrCopy));
    if (n > vlw) {
        Ld1w(ARMCG::ZRegS(NUM_31), p3, Ptr(biaPtrCopy, NUM_1, ARMCG::ExtMod::MUL_VL));
    }
    if (applyConvert) {
        Scvtf(ARMCG::ZRegS(NUM_30), p5, ARMCG::ZRegS(NUM_30));
        if (n > vlw) {
            Scvtf(ARMCG::ZRegS(NUM_31), p5, ARMCG::ZRegS(NUM_31));
        }
    }
    for (IntType i = 0; i < SplitSize; ++i) {
        AddPair(n, vlw, applyConvert, i * NUM_2, i * NUM_2 + NUM_1, NUM_30, NUM_31);
    }

    L(su8MCloadBiasM);
    Tbz(flags, 4, su8MCloadBiasMN); // 4-bit correspond to LOAD_BIAS_M
    for (IntType i = 0; i < SplitSize; ++i) {
        if (i != 0) {
            Add(biaPtrCopy, biaPtrCopy, static_cast<std::uint32_t>(sizeof(BiaType)));
        }

        Ld1rw(ARMCG::ZRegS(NUM_30), p5, ARMCG::AdrUimm(Ptr(biaPtrCopy)));
        if (applyConvert) {
            Scvtf(ARMCG::ZRegS(NUM_30), p5, ARMCG::ZRegS(NUM_30));
        }

        AddPair(n, vlw, applyConvert, i * NUM_2, i * NUM_2 + NUM_1, NUM_30, NUM_30);
    }

    L(su8MCloadBiasMN);
    Tbz(flags, 6, su8MCloadC); // 6-bit correspond to LOAD_BIAS_MN
    for (IntType i = 0; i < SplitSize; ++i) {
        if (i != 0) {
            Add(biaPtrCopy, biaPtrCopy, biaOffset);
        }
        Ld1w(ARMCG::ZRegS(NUM_30), p1, Ptr(biaPtrCopy));
        CvtAdd(applyConvert, i * NUM_2, NUM_30, NUM_1);
        if (n > vlw) {
            Ld1w(ARMCG::ZRegS(NUM_31), p3, Ptr(biaPtrCopy, NUM_1, ARMCG::ExtMod::MUL_VL));
            CvtAdd(applyConvert, i * NUM_2 + NUM_1, NUM_31, NUM_3);
        }
    }

    L(su8MCloadC);
    Tbz(flags, 1, su8MCmullC); // 1-bit correspond to LOAD_C
    for (IntType i = 0; i < SplitSize; ++i) {
        if (i != 0) {
            Add(dstPtrCopy, dstPtrCopy, dstOffset);
        }
        Ld1w(ARMCG::ZRegS(NUM_30), p1, Ptr(dstPtrCopy));
        CvtAdd(applyConvert, i * NUM_2, NUM_30, NUM_1);
        if (n > vlw) {
            Ld1w(ARMCG::ZRegS(NUM_31), p3, Ptr(dstPtrCopy, NUM_1, ARMCG::ExtMod::MUL_VL));
            CvtAdd(applyConvert, i * NUM_2 + NUM_1, NUM_31, NUM_3);
        }
    }

    L(su8MCmullC);
    Tbz(flags, 2, su8MCaddBias1); // 2-bit correspond to MULL_C
    Dup(ARMCG::ZRegS(NUM_31), betaW);
    for (IntType i = 0; i < SplitSize; ++i) {
        if (i != 0) {
            Add(dstPtrCopy, dstPtrCopy, dstOffset);
        }
        Ld1w(ARMCG::ZRegS(NUM_29), p1, Ptr(dstPtrCopy));
        CvtMulAdd(i * NUM_2, NUM_29, NUM_1, NUM_31);
        if (n > vlw) {
            Ld1w(ARMCG::ZRegS(NUM_30), p3, Ptr(dstPtrCopy, NUM_1, ARMCG::ExtMod::MUL_VL));
            CvtMulAdd(i * NUM_2 + NUM_1, NUM_30, NUM_3, NUM_31);
        }
    }

    L(su8MCaddBias1);
    Tbz(flags, 7, su8MCaddBiasN); // 7-bit correspond to Add_BIAS_1
    Ld1rw(ARMCG::ZRegS(NUM_30), p5, ARMCG::AdrUimm(Ptr(biaPtr)));
    if (applyConvert) {
        Scvtf(ARMCG::ZRegS(NUM_30), p5, ARMCG::ZRegS(NUM_30));
    }
    for (IntType i = 0; i < SplitSize; ++i) {
        AddPair(n, vlw, applyConvert, i * NUM_2, i * NUM_2 + NUM_1, NUM_30, NUM_30);
    }

    L(su8MCaddBiasN);
    Tbz(flags, 9, su8MCaddBiasM); // 9-bit correspond to Add_BIAS_N
    Ld1w(ARMCG::ZRegS(NUM_30), p1, Ptr(biaPtrCopy));
    if (n > vlw) {
        Ld1w(ARMCG::ZRegS(NUM_31), p3, Ptr(biaPtrCopy, NUM_1, ARMCG::ExtMod::MUL_VL));
    }
    if (applyConvert) {
        Scvtf(ARMCG::ZRegS(NUM_30), p5, ARMCG::ZRegS(NUM_30));
        if (n > vlw) {
            Scvtf(ARMCG::ZRegS(NUM_31), p5, ARMCG::ZRegS(NUM_31));
        }
    }
    for (IntType i = 0; i < SplitSize; ++i) {
        AddPair(n, vlw, applyConvert, i * NUM_2, i * NUM_2 + NUM_1, NUM_30, NUM_31);
    }

    L(su8MCaddBiasM);
    Tbz(flags, 8, su8MCaddBiasMN); // 8-bit correspond to ADD_BIAS_M
    for (IntType i = 0; i < SplitSize; ++i) {
        if (i != 0) {
            Add(biaPtrCopy, biaPtrCopy, static_cast<std::uint32_t>(sizeof(BiaType)));
        }

        Ld1rw(ARMCG::ZRegS(NUM_30), p5, ARMCG::AdrUimm(Ptr(biaPtrCopy)));
        if (applyConvert) {
            Scvtf(ARMCG::ZRegS(NUM_30), p5, ARMCG::ZRegS(NUM_30));
        }

        AddPair(n, vlw, applyConvert, i * NUM_2, i * NUM_2 + NUM_1, NUM_30, NUM_30);
    }

    L(su8MCaddBiasMN);
    Tbz(flags, 10, su8MCinitDone); // 10-bit correspond to ADD_BIAS_MN
    for (IntType i = 0; i < SplitSize; ++i) {
        if (i != 0) {
            Add(biaPtrCopy, biaPtrCopy, biaOffset);
        }
        Ld1w(ARMCG::ZRegS(NUM_30), p1, Ptr(biaPtrCopy));
        CvtAdd(applyConvert, i * NUM_2, NUM_30, NUM_1);
        if (n > vlw) {
            Ld1w(ARMCG::ZRegS(NUM_31), p3, Ptr(biaPtrCopy, NUM_1, ARMCG::ExtMod::MUL_VL));
            CvtAdd(applyConvert, i * NUM_2 + NUM_1, NUM_31, NUM_3);
        }
    }

    L(su8MCinitDone);
    Tbz(flags, 11, skipPO); // 11-bit correspond to APPLY_POST_OPS
    IntType size = SplitSize;
    if (n > vlw) {
        size += SplitSize;
    }
    std::vector<SizeType> dstRegsIDs(size);
    FillDstRegs(SplitSize, n, vlw, dstRegsIDs);
    for (std::uint32_t i = 0; i < size; ++i) {
        KUDNN_JIT_Generator_With_Attributes::TakeVectorRegister(dstRegsIDs[i]);
    }
    if (applyConvert) {
        this->ApplyPostOps(dstRegsIDs, Element::MatchType<float>());
    } else {
        this->ApplyPostOps(dstRegsIDs, Element::MatchType<DstType>());
    }
    for (std::uint32_t i = 0; i < size; ++i) {
        KUDNN_JIT_Generator_With_Attributes::FreeVectorRegister(dstRegsIDs[i]);
    }
    // Apply convert on last K iter
    if (applyConvert) {
        for (IntType i = 0; i < SplitSize; ++i) {
            Frinti(ARMCG::ZRegS(i * NUM_2),
                p5, ARMCG::ZRegS(i * NUM_2));
            Fcvtzs(ARMCG::ZRegS(i * NUM_2),
                p5, ARMCG::ZRegS(i * NUM_2));
            if (n > vlw) {
                Frinti(ARMCG::ZRegS(i * NUM_2 + NUM_1),
                    p5, ARMCG::ZRegS(i * NUM_2 + NUM_1));
                Fcvtzs(ARMCG::ZRegS(i * NUM_2 + NUM_1),
                    p5, ARMCG::ZRegS(i * NUM_2 + NUM_1));
            }
        }
    }
    L(skipPO);
}

template <typename SrcType, typename WeiType, typename DstType, typename BiaType>
void MCSU8Gen<SrcType, WeiType, DstType, BiaType>::GenerateKLoop(
    IntType m, IntType n, IntType k, IntType vlw)
{
    // NUM_4 due to peculiarities of INT8 MOPA instruction
    IntType kRoundedUp = Math::CeilDiv(k, NUM_4 * K_LOOP_UNROLL) * NUM_4 * K_LOOP_UNROLL;
    ARMCG::Label kLoopHead;
    KUDNN_JIT_Generator::Mov(kTmpCopy, kVal);
    Subs(kTmpCopy, kTmpCopy, K_LOOP_UNROLL * NUM_4); // NUM_4 due to peculiarities of INT8 MOPA instruction
    // p0, p2 - for SRC, p1, p3 - for WEI
    // z0, z2, z4, z6 - for SRC, z1, z3, z5, z7 - for WEI
    // SRC and WEI data is prepacked, so it's continious in memory.
    // K unroll by 2 requires 4 elements from SRC,  4 from WEI
    // Load src data
    Ld1b(ARMCG::ZRegB(NUM_0), p0, Ptr(srcPtrCopy));
    if (m > vlw) {
        Ld1b(ARMCG::ZRegB(NUM_2), p2, Ptr(srcPtrCopy, NUM_1, ARMCG::ExtMod::MUL_VL));
    }
    // Load wei data
    Ld1b(ARMCG::ZRegB(NUM_1), p1, Ptr(weiPtrCopy));
    if (n > vlw) {
        Ld1b(ARMCG::ZRegB(NUM_3), p3, Ptr(weiPtrCopy, NUM_1, ARMCG::ExtMod::MUL_VL));
    }
    if (kRoundedUp >= (K_LOOP_UNROLL * NUM_4)) { // NUM_4 due to peculiarities of INT8 MOPA instruction
        // Load src data for unrolled loop
        Ld1b(ARMCG::ZRegB(NUM_4), p0, Ptr(srcPtrCopy, NUM_2, ARMCG::ExtMod::MUL_VL));
        if (m > vlw) {
            Ld1b(ARMCG::ZRegB(NUM_6), p2, Ptr(srcPtrCopy, NUM_3, ARMCG::ExtMod::MUL_VL));
        }
        // Load wei data unrolled loop
        Ld1b(ARMCG::ZRegB(NUM_5), p1, Ptr(weiPtrCopy, NUM_2, ARMCG::ExtMod::MUL_VL));
        if (n > vlw) {
            Ld1b(ARMCG::ZRegB(NUM_7), p3, Ptr(weiPtrCopy, NUM_3, ARMCG::ExtMod::MUL_VL));
        }
        Addvl(srcPtrCopy, srcPtrCopy, KERNEL_M_SIZE * K_LOOP_UNROLL);
        Addvl(weiPtrCopy, weiPtrCopy, KERNEL_N_SIZE * K_LOOP_UNROLL);
        if (kRoundedUp > (K_LOOP_UNROLL * NUM_4)) { // NUM_4 due to peculiarities of INT8 MOPA instruction
            L(kLoopHead);
                Int8Mopa(ARMCG::ZARegS(NUM_0), p0, p1, ARMCG::ZRegB(NUM_0), ARMCG::ZRegB(NUM_1));
                Subs(kTmpCopy, kTmpCopy, (K_LOOP_UNROLL * NUM_4)); // NUM_4 due to peculiarities of INT8 MOPA instr
                if (n > vlw) {
                    Int8Mopa(ARMCG::ZARegS(NUM_1), p0, p3, ARMCG::ZRegB(NUM_0), ARMCG::ZRegB(NUM_3));
                }
                Ld1b(ARMCG::ZRegB(NUM_0), p0, Ptr(srcPtrCopy));
                if (m > vlw) {
                    Int8Mopa(ARMCG::ZARegS(NUM_2), p2, p1, ARMCG::ZRegB(NUM_2), ARMCG::ZRegB(NUM_1));
                }
                Ld1b(ARMCG::ZRegB(NUM_1), p1, Ptr(weiPtrCopy));
                if ((m > vlw) && (n > vlw)) {
                    Int8Mopa(ARMCG::ZARegS(NUM_3), p2, p3, ARMCG::ZRegB(NUM_2), ARMCG::ZRegB(NUM_3));
                }
                if (m > vlw) {
                    Ld1b(ARMCG::ZRegB(NUM_2), p2, Ptr(srcPtrCopy, NUM_1, ARMCG::ExtMod::MUL_VL));
                }
                if (n > vlw) {
                    Ld1b(ARMCG::ZRegB(NUM_3), p3, Ptr(weiPtrCopy, NUM_1, ARMCG::ExtMod::MUL_VL));
                }

                Int8Mopa(ARMCG::ZARegS(NUM_0), p0, p1, ARMCG::ZRegB(NUM_4), ARMCG::ZRegB(NUM_5));
                if (n > vlw) {
                    Int8Mopa(ARMCG::ZARegS(NUM_1), p0, p3, ARMCG::ZRegB(NUM_4), ARMCG::ZRegB(NUM_7));
                }
                Ld1b(ARMCG::ZRegB(NUM_4), p0, Ptr(srcPtrCopy, NUM_2, ARMCG::ExtMod::MUL_VL));
                if (m > vlw) {
                    Int8Mopa(ARMCG::ZARegS(NUM_2), p2, p1, ARMCG::ZRegB(NUM_6), ARMCG::ZRegB(NUM_5));
                }
                Ld1b(ARMCG::ZRegB(NUM_5), p1, Ptr(weiPtrCopy, NUM_2, ARMCG::ExtMod::MUL_VL));
                if ((m > vlw) && (n > vlw)) {
                    Int8Mopa(ARMCG::ZARegS(NUM_3), p2, p3, ARMCG::ZRegB(NUM_6), ARMCG::ZRegB(NUM_7));
                }
                if (m > vlw) {
                    Ld1b(ARMCG::ZRegB(NUM_6), p2, Ptr(srcPtrCopy, NUM_3, ARMCG::ExtMod::MUL_VL));
                }
                if (n > vlw) {
                    Ld1b(ARMCG::ZRegB(NUM_7), p3, Ptr(weiPtrCopy, NUM_3, ARMCG::ExtMod::MUL_VL));
                }

                Addvl(srcPtrCopy, srcPtrCopy, KERNEL_M_SIZE * K_LOOP_UNROLL);
                Addvl(weiPtrCopy, weiPtrCopy, KERNEL_N_SIZE * K_LOOP_UNROLL);

            B(ARMCG::Cond::GT, kLoopHead);
        }
    }

    Int8Mopa(ARMCG::ZARegS(NUM_0), p0, p1, ARMCG::ZRegB(NUM_0), ARMCG::ZRegB(NUM_1));
    if (n > vlw) {
        Int8Mopa(ARMCG::ZARegS(NUM_1), p0, p3, ARMCG::ZRegB(NUM_0), ARMCG::ZRegB(NUM_3));
    }
    if (m > vlw) {
        Int8Mopa(ARMCG::ZARegS(NUM_2), p2, p1, ARMCG::ZRegB(NUM_2), ARMCG::ZRegB(NUM_1));
    }
    if ((m > vlw) && (n > vlw)) {
        Int8Mopa(ARMCG::ZARegS(NUM_3), p2, p3, ARMCG::ZRegB(NUM_2), ARMCG::ZRegB(NUM_3));
    }
    // NUM_4 due to peculiarities of INT8 MOPA instruction
    if ((k == kRoundedUp) || (k % (K_LOOP_UNROLL * NUM_4) > NUM_4)) {
        Int8Mopa(ARMCG::ZARegS(NUM_0), p0, p1, ARMCG::ZRegB(NUM_4), ARMCG::ZRegB(NUM_5));
        if (n > vlw) {
            Int8Mopa(ARMCG::ZARegS(NUM_1), p0, p3, ARMCG::ZRegB(NUM_4), ARMCG::ZRegB(NUM_7));
        }
        if (m > vlw) {
            Int8Mopa(ARMCG::ZARegS(NUM_2), p2, p1, ARMCG::ZRegB(NUM_6), ARMCG::ZRegB(NUM_5));
        }
        if ((m > vlw) && (n > vlw)) {
            Int8Mopa(ARMCG::ZARegS(NUM_3), p2, p3, ARMCG::ZRegB(NUM_6), ARMCG::ZRegB(NUM_7));
        }
    }
}

template <typename SrcType, typename WeiType, typename DstType, typename BiaType>
void MCSU8Gen<SrcType, WeiType, DstType, BiaType>::Int8Mopa(const ARMCG::ZARegS &zada, const ARMCG::PRegBase &pn,
    const ARMCG::PRegBase &pm, const ARMCG::ZRegB &zn, const ARMCG::ZRegB &zm)
{
    if constexpr (std::is_same_v<SrcType, WeiType>) {
        if constexpr (std::is_same_v<SrcType, std::int8_t>) {
            Smopa(zada, pn, pm, zn, zm);
        } else {
            Umopa(zada, pn, pm, zn, zm);
        }
    } else {
        if (std::is_same_v<SrcType, std::int8_t> == wereSrcWeiSwapped) {
            Usmopa(zada, pn, pm, zn, zm);
        } else {
            Sumopa(zada, pn, pm, zn, zm);
        }
    }
}

template <typename SrcType, typename WeiType, typename DstType, typename BiaType>
MatMulImplGenerator<MCSU8Gen<SrcType, WeiType, DstType, BiaType>>::DefaultImpl::DefaultImpl(
    const GemmInfo &task, IntType nThreads) : MatMulImplBase(task, nThreads)
{
    VERBOSE_PRINT;
    UpdateExecParams();
    Initialize();
}

template <typename SrcType, typename WeiType, typename DstType, typename BiaType>
std::uint32_t MatMulImplGenerator<MCSU8Gen<SrcType, WeiType, DstType, BiaType>>::DefaultImpl::GetKernelMSize()
    noexcept
{
    return MCSU8Gen<SrcType, WeiType, DstType, BiaType>::KERNEL_M_SIZE * svcntw();
}

template <typename SrcType, typename WeiType, typename DstType, typename BiaType>
std::uint32_t MatMulImplGenerator<MCSU8Gen<SrcType, WeiType, DstType, BiaType>>::DefaultImpl::GetKernelNSize()
    noexcept
{
    return MCSU8Gen<SrcType, WeiType, DstType, BiaType>::KERNEL_N_SIZE * svcntw();
}

template <typename SrcType, typename WeiType, typename DstType, typename BiaType>
IntType MatMulImplGenerator<MCSU8Gen<SrcType, WeiType, DstType, BiaType>>::DefaultImpl::GetMBlockSize()
    const noexcept
{
    return 256; // 256 is experimentally measured constant to obtain better perf
}

template <typename SrcType, typename WeiType, typename DstType, typename BiaType>
IntType MatMulImplGenerator<MCSU8Gen<SrcType, WeiType, DstType, BiaType>>::DefaultImpl::GetNBlockSize()
    const noexcept
{
    return 128; // 128 is experimentally measured constant to obtain better perf
}

template <typename SrcType, typename WeiType, typename DstType, typename BiaType>
IntType MatMulImplGenerator<MCSU8Gen<SrcType, WeiType, DstType, BiaType>>::DefaultImpl::GetKBlockSize()
    const noexcept
{
    IntType defKblock;
    // all the constants(1024,512,512,256) are experimentally measured and give optimal performance configuration
    if (MatMulImplBase::mFullBlockSizePerThread + MatMulImplBase::nFullBlockSizePerThread >= NUM_64) {
        if (MatMulImplBase::kFullBlockSizePerThread >= NUM_1024) {
            defKblock = NUM_1024;
        } else {
            defKblock = NUM_512;
        }
    } else {
        if (MatMulImplBase::kFullBlockSizePerThread >= NUM_512) {
            defKblock = NUM_512;
        } else {
            defKblock = NUM_256;
        }
    }
    IntType k = taskExec.srcInfo.GetDims()[taskExec.srcInfo.GetNumDims() - NUM_1];
    IntType numKblocks = Math::CeilDiv(k, defKblock);
    IntType residBlock = k % defKblock;
    IntType kBlock = k;
    if (k > NUM_3 * defKblock) {
        kBlock = defKblock + Math::CeilDiv(residBlock, numKblocks - NUM_1);
    } else if (k > defKblock) {
        kBlock = k / NUM_2;
    }
    return Math::CeilDiv(kBlock, NUM_4) * NUM_4;
}

template <typename SrcType, typename WeiType, typename DstType, typename BiaType>
void MatMulImplGenerator<MCSU8Gen<SrcType, WeiType, DstType, BiaType>>::DefaultImpl::InitializePackingPolicy()
{
    packingPolicy = PackingPolicy::NO_SYNC;
    InitializePackingPolicyFromEnv();
    if ((packingPolicy == PackingPolicy::SYNC_ACC_DIM_SRC) ||
        (packingPolicy == PackingPolicy::SYNC_ACC_DIM_WEI) ||
        (packingPolicy == PackingPolicy::SYNC_ACC_DIM_SRC_WEI)) {
        throw Service::LogicError("Matmul: Current packing policy not supported for this precision");
    }
}

template <typename SrcType, typename WeiType, typename DstType, typename BiaType>
void MatMulImplGenerator<MCSU8Gen<SrcType, WeiType, DstType, BiaType>>::DefaultImpl::InitializeSingleCodegen(
    IntType id, IntType m, IntType n, IntType k,
    IntType currentM, IntType currentN, IntType currentK,
    SizeType offsetSrc, SizeType offsetWei, SizeType offsetDst, SizeType offsetBia,
    bool wereSrcWeiSwapped, const Attributes &attributes, bool hasAlpha)
{
    Service::Unused(offsetSrc, offsetWei, offsetDst, offsetBia, hasAlpha);
    if ((currentM == 0) || (currentN == 0) || (currentK == 0)) {
        return;
    }
    if ((currentM <= m) && (currentN <= n) && (currentK <= k)) {
        CodeGenParams<MCSU8Gen<SrcType, WeiType, DstType, BiaType>> currTask{
            currentM, currentN, currentK, wereSrcWeiSwapped, attributes};
        if (codegens.find(currTask) == codegens.end()) {
            codegens.insert({currTask, std::make_shared<MCSU8Gen<SrcType, WeiType, DstType, BiaType>>(
                currentM, currentN, currentK, wereSrcWeiSwapped, attributes)});
        }
        codegens[currTask]->Ready();
        matmulCodegenKernels[id] =
            codegens[currTask]->template GetCode<MCSU8CodegenFuncPtr<SrcType, WeiType, DstType, BiaType>>();
    }
}

template <typename SrcType, typename WeiType, typename DstType, typename BiaType>
void MatMulImplGenerator<MCSU8Gen<SrcType, WeiType, DstType, BiaType>>::DefaultImpl::InitializeReorderCodeGens(
    IntType m, IntType n, IntType k, IntType mFull, IntType nFull)
{
    IntType vl = svcntw();
    IntType mFullThreadBlock = mFull;
    IntType mResidThreadBlock = m % mFullThreadBlock;
    IntType nFullThreadBlock = nFull;
    IntType nResidThreadBlock = n % nFullThreadBlock;

    IntType mSubBlockExecFull = MCSU8Gen<SrcType, WeiType, DstType, BiaType>::KERNEL_M_SIZE * vl;
    IntType mSubBlockExecResid1 = mFullThreadBlock % mSubBlockExecFull;
    IntType mSubBlockExecResid2 = mResidThreadBlock % mSubBlockExecFull;

    IntType nSubBlockExecFull = MCSU8Gen<SrcType, WeiType, DstType, BiaType>::KERNEL_N_SIZE * vl;
    IntType nSubBlockExecResid1 = nFullThreadBlock % nSubBlockExecFull;
    IntType nSubBlockExecResid2 = nResidThreadBlock % nSubBlockExecFull;

    IntType kBlockExecFull = GetKBlockSize();
    IntType kBlockExecResid = k % GetKBlockSize();
    
    // U8 specific ==> sizeof(float)
    CodeGenWrapper<FastReorderKernelCodeGen<float, float>> reorderFBFKNA; // full Block full K NoAlpha
    reorderFBFKNA.Initialize(static_cast<SizeType>(vl),
                             static_cast<SizeType>(kBlockExecFull / sizeof(float)), false);

    CodeGenWrapper<FastReorderKernelCodeGen<float, float>> reorderRN1FKNA; // resid N 1 full K NoAlpha
    reorderRN1FKNA.Initialize(static_cast<SizeType>(nSubBlockExecResid1 % vl),
                              static_cast<SizeType>(kBlockExecFull / sizeof(float)), false);

    CodeGenWrapper<FastReorderKernelCodeGen<float, float>> reorderRN2FKNA; // resid N 2 full K NoAlpha
    reorderRN2FKNA.Initialize(static_cast<SizeType>(nSubBlockExecResid2 % vl),
                              static_cast<SizeType>(kBlockExecFull / sizeof(float)), false);

    CodeGenWrapper<FastReorderKernelCodeGen<float, float>> reorderRM1FKNA; // resid M 1 full K NoAlpha
    reorderRM1FKNA.Initialize(static_cast<SizeType>(mSubBlockExecResid1 % vl),
                              static_cast<SizeType>(kBlockExecFull / sizeof(float)), false);
    
    CodeGenWrapper<FastReorderKernelCodeGen<float, float>> reorderRM2FKNA; // resid M 2 full K NoAlpha
    reorderRM2FKNA.Initialize(static_cast<SizeType>(mSubBlockExecResid2 % vl),
                              static_cast<SizeType>(kBlockExecFull / sizeof(float)), false);

    CodeGenWrapper<FastReorderKernelCodeGen<float, float>> reorderFBRKNA; // full Block resid K NoAlpha
    reorderFBRKNA.Initialize(static_cast<SizeType>(vl),
                             static_cast<SizeType>(kBlockExecResid / sizeof(float)), false);
                            
    CodeGenWrapper<FastReorderKernelCodeGen<float, float>> reorderRN1RKNA; // resid N 1 resid K NoAlpha
    reorderRN1RKNA.Initialize(static_cast<SizeType>(nSubBlockExecResid1 % vl),
                              static_cast<SizeType>(kBlockExecResid / sizeof(float)), false);
    
    CodeGenWrapper<FastReorderKernelCodeGen<float, float>> reorderRN2RKNA; // resid N 2 resid K NoAlpha
    reorderRN2RKNA.Initialize(static_cast<SizeType>(nSubBlockExecResid2 % vl),
                              static_cast<SizeType>(kBlockExecResid / sizeof(float)), false);
                            
    CodeGenWrapper<FastReorderKernelCodeGen<float, float>> reorderRM1RKNA; // resid M 1 resid K NoAlpha
    reorderRM1RKNA.Initialize(static_cast<SizeType>(mSubBlockExecResid1 % vl),
                              static_cast<SizeType>(kBlockExecResid / sizeof(float)), false);
    
    CodeGenWrapper<FastReorderKernelCodeGen<float, float>> reorderRM2RKNA; // resid M 2 resid K NoAlpha
    reorderRM2RKNA.Initialize(static_cast<SizeType>(mSubBlockExecResid2 % vl),
                              static_cast<SizeType>(kBlockExecResid / sizeof(float)), false);

    reorderPackingCodegenKernels = {
        std::move(reorderFBFKNA), std::move(reorderRN1FKNA), std::move(reorderRN2FKNA),
        std::move(reorderRM1FKNA), std::move(reorderRM2FKNA),
        std::move(reorderFBRKNA), std::move(reorderRN1RKNA), std::move(reorderRN2RKNA),
        std::move(reorderRM1RKNA), std::move(reorderRM2RKNA)
    };
}

template <typename SrcType, typename WeiType, typename DstType, typename BiaType>
IntType MatMulImplGenerator<MCSU8Gen<SrcType, WeiType, DstType, BiaType>>::DefaultImpl::GetThreadBufSize(
    IntType m, IntType n, IntType k, MatrixKind mKind) const noexcept
{
    IntType vlb = svcntb();
    IntType mRoundedUp = Math::CeilDiv(m, MCSU8Gen<SrcType, WeiType, DstType, BiaType>::KERNEL_M_SIZE * vlb) *
        MCSU8Gen<SrcType, WeiType, DstType, BiaType>::KERNEL_M_SIZE * vlb;
    IntType nRoundedUp = Math::CeilDiv(n, MCSU8Gen<SrcType, WeiType, DstType, BiaType>::KERNEL_N_SIZE * vlb) *
        MCSU8Gen<SrcType, WeiType, DstType, BiaType>::KERNEL_N_SIZE * vlb;
    // NUM_4 is due to peculiarities of MOPA instruction for INT8 data type
    IntType kRoundedUp = Math::CeilDiv(k, MCSU8Gen<SrcType, WeiType, DstType, BiaType>::K_LOOP_UNROLL * NUM_4) *
        MCSU8Gen<SrcType, WeiType, DstType, BiaType>::K_LOOP_UNROLL * NUM_4;
    if (mKind == MatrixKind::SRC) {
        return static_cast<IntType>(std::min(GetKBlockSize(), kRoundedUp) *
            std::min(GetMBlockSize(), mRoundedUp) * sizeof(SrcType));
    } else {
        return static_cast<IntType>(std::min(GetKBlockSize(), kRoundedUp) *
            std::min(GetNBlockSize(), nRoundedUp) * sizeof(WeiType));
    }
}

template <typename SrcType, typename WeiType, typename DstType, typename BiaType>
void MatMulImplGenerator<MCSU8Gen<SrcType, WeiType, DstType, BiaType>>::DefaultImpl::ReserveImplSpecificParams()
{}

using vepa = std::vector<std::pair<Shape, Shape>>;
template <typename SrcType, typename WeiType, typename DstType, typename BiaType>
vepa MatMulImplGenerator<MCSU8Gen<SrcType, WeiType, DstType, BiaType>>::DefaultImpl::SplitWork()
{
    return CalculateWorkOnThread<DefaultImpl>(taskExec.dstInfo.GetDims(), nThreadsExec, dstLayout2D == Layout::BA);
}

// used for packing both Src and Wei matrices
template <typename SrcType, typename WeiType, typename DstType, typename BiaType>
void MatMulImplGenerator<MCSU8Gen<SrcType, WeiType, DstType, BiaType>>::DefaultImpl::PackSimilarLt(
    IntType mBlock, IntType kBlock,
    const void *src, IntType offset, void *buffer, IntType bufferOffset)
{
    IntType vlb = svcntb();
    svbool_t ptrue = svptrue_b8();
    // NUM_4 is due to peculiarities of MOPA instruction for INT8 data type
    IntType kRoundedUp = ((kBlock + NUM_3) / NUM_4) * NUM_4;
    IntType k = 0;
    // NUM_4 is due to peculiarities of MOPA instruction for INT8 data type
    for (; k < kBlock - NUM_4; k += NUM_4) {
        for (IntType m = 0; m < mBlock; m += vlb) {
            svbool_t p0 = svwhilelt_b8(m + 0 * vlb, mBlock);
            svint8_t src0 = svld1(p0, static_cast<const std::int8_t*>(src) + 0 * vlb + m + offset * (k + 0));
            svint8_t src1 = svld1(p0, static_cast<const std::int8_t*>(src) + 0 * vlb + m + offset * (k + 1));
            svint8_t src2 = svld1(p0, static_cast<const std::int8_t*>(src) + 0 * vlb + m + offset * (k + 2));
            svint8_t src3 = svld1(p0, static_cast<const std::int8_t*>(src) + 0 * vlb + m + offset * (k + 3));
            svint8_t tmp0 = svzip1(src0, src1);
            svint8_t tmp1 = svzip2(src0, src1);
            svint8_t tmp2 = svzip1(src2, src3);
            svint8_t tmp3 = svzip2(src2, src3);
            svint8_t dst0 = svreinterpret_s8(svzip1(svreinterpret_s16(tmp0), svreinterpret_s16(tmp2)));
            svint8_t dst1 = svreinterpret_s8(svzip2(svreinterpret_s16(tmp0), svreinterpret_s16(tmp2)));
            svint8_t dst2 = svreinterpret_s8(svzip1(svreinterpret_s16(tmp1), svreinterpret_s16(tmp3)));
            svint8_t dst3 = svreinterpret_s8(svzip2(svreinterpret_s16(tmp1), svreinterpret_s16(tmp3)));
            svst1(ptrue, static_cast<std::int8_t *>(buffer)
                + 0 * vlb + (m + 0 * bufferOffset) * kRoundedUp + k * bufferOffset, dst0);
            svst1(ptrue, static_cast<std::int8_t *>(buffer)
                + 1 * vlb + (m + 0 * bufferOffset) * kRoundedUp + k * bufferOffset, dst1);
            svst1(ptrue, static_cast<std::int8_t *>(buffer)
                + 0 * vlb + (m + 1 * bufferOffset) * kRoundedUp + k * bufferOffset, dst2);
            svst1(ptrue, static_cast<std::int8_t *>(buffer)
                + 1 * vlb + (m + 1 * bufferOffset) * kRoundedUp + k * bufferOffset, dst3);
        }
    }
    IntType kResid = kBlock % NUM_4;
    if (kResid == 0) {
        kResid = NUM_4;
    }
    for (IntType m = 0; m < mBlock; m += vlb) {
        svbool_t p0 = svwhilelt_b8(m + 0 * vlb, mBlock);
        svint8_t src0 = svld1(p0, static_cast<const std::int8_t*>(src) + 0 * vlb + m + offset * (k + 0));
        svint8_t src1;
        if (kResid > NUM_1) {
            src1 = svld1(p0, static_cast<const std::int8_t*>(src) + 0 * vlb + m + offset * (k + 1));
        } else {
            src1 = svdup_s8(NUM_0);
        }
        svint8_t src2;
        if (kResid > NUM_2) {
            src2 = svld1(p0, static_cast<const std::int8_t*>(src) + 0 * vlb + m + offset * (k + 2));
        } else {
            src2 = svdup_s8(NUM_0);
        }
        svint8_t src3;
        if (kResid > NUM_3) {
            src3 = svld1(p0, static_cast<const std::int8_t*>(src) + 0 * vlb + m + offset * (k + 3));
        } else {
            src3 = svdup_s8(NUM_0);
        }
        svint8_t tmp0 = svzip1(src0, src1);
        svint8_t tmp1 = svzip2(src0, src1);
        svint8_t tmp2 = svzip1(src2, src3);
        svint8_t tmp3 = svzip2(src2, src3);
        svint8_t dst0 = svreinterpret_s8(svzip1(svreinterpret_s16(tmp0), svreinterpret_s16(tmp2)));
        svint8_t dst1 = svreinterpret_s8(svzip2(svreinterpret_s16(tmp0), svreinterpret_s16(tmp2)));
        svint8_t dst2 = svreinterpret_s8(svzip1(svreinterpret_s16(tmp1), svreinterpret_s16(tmp3)));
        svint8_t dst3 = svreinterpret_s8(svzip2(svreinterpret_s16(tmp1), svreinterpret_s16(tmp3)));
        svst1(ptrue, static_cast<std::int8_t *>(buffer)
            + 0 * vlb + (m + 0 * bufferOffset) * kRoundedUp + k * bufferOffset, dst0);
        svst1(ptrue, static_cast<std::int8_t *>(buffer)
            + 1 * vlb + (m + 0 * bufferOffset) * kRoundedUp + k * bufferOffset, dst1);
        svst1(ptrue, static_cast<std::int8_t *>(buffer)
            + 0 * vlb + (m + 1 * bufferOffset) * kRoundedUp + k * bufferOffset, dst2);
        svst1(ptrue, static_cast<std::int8_t *>(buffer)
            + 1 * vlb + (m + 1 * bufferOffset) * kRoundedUp + k * bufferOffset, dst3);
    }
}

// used for packing both Src and Wei matrices
template <typename SrcType, typename WeiType, typename DstType, typename BiaType>
void MatMulImplGenerator<MCSU8Gen<SrcType, WeiType, DstType, BiaType>>::DefaultImpl::PackDiffLt(
    IntType threadID, IntType mBlock, IntType kBlock,
    const void *src, IntType offset, void *buffer, IntType bufferOffset,
    PackingType packingType, IntType kBlockSize) const
{
    IntType kBlock32 = kBlock / sizeof(float);
    IntType kBlock32RoundedUp = Math::CeilDiv(kBlock, sizeof(float));
    IntType kResid = kBlock % sizeof(float);
    svbool_t pgResid = svwhilelt_b8_s64(0, kResid);
    svbool_t pg4 = svwhilelt_b8_s64(0, sizeof(float));
    IntType idxK = (kBlock == kBlockSize) ? 0 : 1;
    bool isM = (packingType != PackingType::PACK_SRC) ? false : true;
    const IntType vlw = svcntw();
    for (IntType mIdx = mBlock; mIdx > 0; mIdx -= bufferOffset) {
        if (mIdx <= vlw) {
            if (kBlock32 > 0) {
                IntType idxM = ((mIdx < vlw) ? 1 : 0);
                IntType mIdxKernel = (idxM) * (1 + isM * NUM_2 +
                    (isM ? perThreadParams[threadID].isResidByM : perThreadParams[threadID].isResidByN));
                IntType packBColIdx = idxK * NUM_5 +  mIdxKernel;
                auto kernel = reorderPackingCodegenKernels[packBColIdx]
                    .template GetCode<void(*)(const float*, float*, IntType, IntType, float)>();
                kernel(Service::AddOffset<const float *>(src, (mBlock - mIdx) * offset),
                    static_cast<float *>(buffer) + kBlock32RoundedUp * (mBlock - mIdx),
                    offset, bufferOffset * sizeof(float), 1.0f);
            }
            if (kResid) {
                for (IntType mSubIdx = mIdx; mSubIdx > 0; --mSubIdx) {
                    svint8_t srcReg = svld1(pgResid, static_cast<const std::int8_t *>(src) +
                                            (mBlock - mSubIdx) * offset + kBlock32 * sizeof(float));
                    svst1(pg4, static_cast<std::int8_t *>(buffer) +
                          kBlock32 * bufferOffset * sizeof(float) +
                          (kBlock32RoundedUp * (mBlock - mIdx) + (mIdx - mSubIdx)) * NUM_4, srcReg);
                }
            }
        } else {
            if (kBlock32 > 0) {
                IntType mIdxReorder1 = vlw;
                IntType mIdxReorder2 = std::min(vlw, mIdx - vlw);
                IntType idxM1 = ((mIdxReorder1 < vlw) ? 1 : 0);
                IntType idxM2 = ((mIdxReorder2 < vlw) ? 1 : 0);
                IntType mIdxKernel1 = (idxM1) * (1 + isM * NUM_2 +
                    (isM ? perThreadParams[threadID].isResidByM : perThreadParams[threadID].isResidByN));
                IntType mIdxKernel2 = (idxM2) * (1 + isM * NUM_2 +
                    (isM ? perThreadParams[threadID].isResidByM : perThreadParams[threadID].isResidByN));
                IntType packBColIdx1 = idxK * NUM_5 + mIdxKernel1;
                IntType packBColIdx2 = idxK * NUM_5 + mIdxKernel2;
                auto kernel = reorderPackingCodegenKernels[packBColIdx1]
                            .template GetCode<void(*)(const float*, float*, IntType, IntType, float)>();
                kernel(Service::AddOffset<const float *>(src, (mBlock - mIdx) * offset),
                       static_cast<float *>(buffer) + kBlock32RoundedUp * (mBlock - mIdx),
                       offset, bufferOffset * sizeof(float), 1.0f);

                kernel = reorderPackingCodegenKernels[packBColIdx2]
                            .template GetCode<void(*)(const float*, float*, IntType, IntType, float)>();
                kernel(reinterpret_cast<const float *>(static_cast<const char *>(src) + (mBlock - mIdx + vlw) * offset),
                       static_cast<float *>(buffer) + kBlock32RoundedUp * (mBlock - mIdx) + vlw,
                       offset, bufferOffset * sizeof(float), 1.0f);
            }
            if (kResid) {
                for (IntType mSubIdx = mIdx; mSubIdx > std::max(IntType{0}, mIdx - NUM_2 * vlw); --mSubIdx) {
                    svint8_t srcReg = svld1(pgResid, Service::AddOffset<const std::int8_t *>(src,
                        (mBlock - mSubIdx) * offset + kBlock32 * sizeof(float)));
                    svst1(pg4, static_cast<std::int8_t *>(buffer) +
                          kBlock32 * bufferOffset * sizeof(float) +
                          (kBlock32RoundedUp * (mBlock - mIdx) + (mIdx - mSubIdx)) * NUM_4, srcReg);
                }
            }
        }
    }
}

template <typename SrcType, typename WeiType, typename DstType, typename BiaType>
void MatMulImplGenerator<MCSU8Gen<SrcType, WeiType, DstType, BiaType>>::DefaultImpl::Pack(
    const void *originalMatrix, void *buffer, IntType threadID, IntType dstDim, IntType accumDim,
    IntType dstDimOffset, IntType accumDimOffset, IntType bufferOffset, IntType bufferStride, float alpha,
    MatrixKind mKind) const
{
    Service::Unused(bufferStride, alpha);
    auto vlw = svcntw();
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
                MCSU8Gen<SrcType, WeiType, DstType, BiaType>::KERNEL_M_SIZE * vlw);
        } else {
            srcExec += dstDimOffset * offset2D + accumDimOffset;
            PackDiffLt(threadID, dstDim, accumDim, srcExec,
                offset2D, static_cast<SrcType *>(buffer) + bufferOffset,
                MCSU8Gen<SrcType, WeiType, DstType, BiaType>::KERNEL_M_SIZE * vlw, PackingType::PACK_SRC,
                GetKBlockSize());
        }
    } else {
        const WeiType *weiExec = static_cast<const WeiType *>(originalMatrix);
        if (layout == Layout::ROW_MAJOR) {
            weiExec += dstDimOffset + accumDimOffset * offset2D;
            PackSimilarLt(dstDim, accumDim, weiExec,
                offset2D, static_cast<WeiType *>(buffer) + bufferOffset,
                MCSU8Gen<SrcType, WeiType, DstType, BiaType>::KERNEL_N_SIZE * vlw);
        } else {
            weiExec += dstDimOffset * offset2D + accumDimOffset;
            PackDiffLt(threadID, dstDim, accumDim, weiExec,
                offset2D, static_cast<WeiType *>(buffer) + bufferOffset,
                MCSU8Gen<SrcType, WeiType, DstType, BiaType>::KERNEL_N_SIZE * vlw, PackingType::PACK_WEI,
                GetKBlockSize());
        }
    }
}

template <typename SrcType, typename WeiType, typename DstType, typename BiaType>
void MatMulImplGenerator<MCSU8Gen<SrcType, WeiType, DstType, BiaType>>::DefaultImpl::RunSequential2D(
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
    SrcType* srcBuffer = static_cast<SrcType *>(srcPackStrategy->GetPtr(tIDbatch * threadsM  + tIDm, tIDn, threadsN));
    WeiType* weiBuffer = static_cast<WeiType *>(weiPackStrategy->GetPtr(tIDbatch * threadsN  + tIDn, tIDm, threadsM));

    for (IntType mBlock = 0; mBlock < mFullBlockSizePerThread; mBlock += GetMBlockSize()) {
        auto mBlockExec = std::min(GetMBlockSize(), m - mBlock);
        IntType mIdx = (mBlockExec == GetMBlockSize()) ? 0 : 1;
        IntType mIdxKernel = mIdx * (perThreadParams[threadID].isResidByM + 1);
        for (IntType kBlock = 0; kBlock < k; kBlock += GetKBlockSize()) {
            auto kBlockExec = std::min(GetKBlockSize(), k - kBlock);
            // NUM_4 is due to peculiarities of MOPA instruction for INT8 data type
            IntType kIdxKernel = (kBlockExec == GetKBlockSize()) ? 0 : 1;
            bool lastKIter = (k - kBlock) <= GetKBlockSize();
            MatMulDstFlags biaFlagsExec = SetBiaFlagsExec(biaFlags, beta, kBlock, lastKIter, alpha);
            IntType kBlockExecRoundedUp = Math::CeilDiv(kBlockExec, NUM_4) * NUM_4; // INT8 MOPA peculiarities
            if (mBlock < m) {
                srcPackStrategy->Pack(*this, src, srcBuffer,
                    {mBlock, mBlockExec, GetKernelMSize() * NUM_2}, {kBlock, kBlockExec, NUM_4},
                    threadsN, tIDn, threadID, alpha);
            }
            srcPackStrategy->Barrier(tIDbatch * threadsM + tIDm, tIDn, threadsN);
            for (IntType nBlock = 0; nBlock < nFullBlockSizePerThread; nBlock += GetNBlockSize()) {
                if ((nBlock < n) && (mBlock < m)) {
                    auto nBlockExec = std::min(GetNBlockSize(), n - nBlock);
                    IntType nIdx = (nBlockExec == GetNBlockSize()) ? 0 : 1;
                    IntType nIdxKernel = nIdx * (perThreadParams[threadID].isResidByN + 1);
                    weiPackStrategy->Pack(*this, wei, weiBuffer,
                        {nBlock, nBlockExec, GetKernelNSize() * NUM_2}, {kBlock, kBlockExec, NUM_4},
                        threadsM, tIDm, threadID, alpha);
                    weiPackStrategy->Barrier(tIDbatch * threadsN + tIDn, tIDm, threadsM);
                    DstType *dstExec = static_cast<DstType *>(dst) + nBlock +
                        mBlock * perThreadParams[threadID].dstOffset2D;
                    const BiaType *biaExec = static_cast<const BiaType *>(bia) +
                        nBlock * biaNOffset + mBlock * perThreadParams[threadID].biaOffset2D;
                    auto kernel = matmulCodegenKernels[kIdxKernel * NUM_9 +
                        nIdxKernel * NUM_3 + mIdxKernel];
                    kernel(srcBuffer, weiBuffer, dstExec, biaExec, biaFlagsExec,
                        alpha, beta, mBlockExec, nBlockExec, kBlockExec, kBlockExecRoundedUp, kBlockExecRoundedUp,
                        perThreadParams[threadID].dstOffset2D * sizeof(DstType),
                        perThreadParams[threadID].biaOffset2D * sizeof(DstType));
                    weiPackStrategy->Barrier(tIDbatch * threadsN + tIDn, tIDm, threadsM);
                }
            }
            srcPackStrategy->Barrier(tIDbatch * threadsM + tIDm, tIDn, threadsN);
        }
    }
}

template <typename SrcType, typename WeiType, typename DstType, typename BiaType>
void MatMulImplGenerator<MCSU8Gen<SrcType, WeiType, DstType, BiaType>>::DefaultImpl::UpdateExecParams() noexcept
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
        MCSU8Gen<SrcType, WeiType, DstType, BiaType>::GetSrcType(),
        MCSU8Gen<SrcType, WeiType, DstType, BiaType>::GetWeiType(),
        MCSU8Gen<SrcType, WeiType, DstType, BiaType>::GetDstType(),
        (implTask.biaInfo.GetTotalTensorSize() != 0) ? MCSU8Gen<SrcType, WeiType, DstType, BiaType>::GetBiaType() :
            static_cast<Element::TypeT>(implTask.biaInfo.GetType()));
}

template <typename SrcType, typename WeiType, typename DstType, typename BiaType>
MatMulDstFlags MatMulImplGenerator<MCSU8Gen<SrcType, WeiType, DstType, BiaType>>::DefaultImpl::SetBiaFlagsExec(
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

template struct MatMulImplGenerator<MatMulKernelDefaultCodeGen<ARCH::MC,
    std::int8_t, std::int8_t, std::int32_t, std::int32_t, MatMulImplKinds::SU8SU8S32S32_MC>>;

template struct MatMulImplGenerator<MatMulKernelDefaultCodeGen<ARCH::MC,
    std::int8_t, std::uint8_t, std::int32_t, std::int32_t, MatMulImplKinds::SU8SU8S32S32_MC>>;

template struct MatMulImplGenerator<MatMulKernelDefaultCodeGen<ARCH::MC,
    std::uint8_t, std::int8_t, std::int32_t, std::int32_t, MatMulImplKinds::SU8SU8S32S32_MC>>;

template struct MatMulImplGenerator<MatMulKernelDefaultCodeGen<ARCH::MC,
    std::uint8_t, std::uint8_t, std::int32_t, std::int32_t, MatMulImplKinds::SU8SU8S32S32_MC>>;

} // Detail
} // KuDNN
