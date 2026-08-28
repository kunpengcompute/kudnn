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
#include "kernels/matmul_jit/sme/kudnn_default_impl_bfp16bfp16bfp1632bfp1632.hpp"

namespace KuDNN {
namespace Detail {

template <typename SrcWeiType, typename DstBiaType>
using MCBFP16GenBase = MatMulKernelCodeGenBase<ARCH::MC, SrcWeiType, SrcWeiType, DstBiaType, DstBiaType>;

template <typename SrcWeiType, typename DstBiaType>
MCBFP16Gen<SrcWeiType, DstBiaType>::MatMulKernelDefaultCodeGen(
    IntType m, IntType n, IntType k,
    const Attributes &attributesArg)
    : MCBFP16GenBase<SrcWeiType, DstBiaType>(attributesArg)
{
    if ((m > 0) && (n > 0) && (k > 0)) {
        if (this->attributes.GetPostOps().Len() > 0) {
            // set reserved predicate registers
            for (std::uint32_t idx = 0; idx < NUM_5; ++idx) {
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

        Lsl(mVal, mVal, NUM_1); // NUM_1 due to peculiarities of FP16 MOPA instruction
        Lsl(nVal, nVal, NUM_1); // NUM_1 due to peculiarities of FP16 MOPA instruction
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
                Sub(mLoopCtr, mLoopCtr, KERNEL_M_SIZE * vl * NUM_2); // NUM_2 due to peculiarities of FP16 MOPA instr
                Cmp(mLoopCtr, KERNEL_M_SIZE * vl * NUM_2); // NUM_2 due to peculiarities of FP16 MOPA instruction
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

template <typename SrcWeiType, typename DstBiaType>
void MCBFP16Gen<SrcWeiType, DstBiaType>::GenerateMLoopBody(IntType mLim, IntType n, IntType k)
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
    Whilelt(ARMCG::PRegH(NUM_0), tmp, mLoopCtr);
    if (mLim > vl) {
        Inch(tmp);
        Whilelt(ARMCG::PRegH(NUM_2), tmp, mLoopCtr);
    }
    if (n > (KERNEL_N_SIZE * vl)) {
        ARMCG::Label nLoopHead;
        L(nLoopHead);
        {
            GenerateNLoopBody(mLim, KERNEL_N_SIZE * vl, k);
            Sub(nLoopCtr, nLoopCtr, KERNEL_N_SIZE * vl * NUM_2); // NUM_2 due to peculiarities of FP16 MOPA instruction
            Cmp(nLoopCtr, KERNEL_N_SIZE * vl * NUM_2); // NUM_2 due to peculiarities of FP16 MOPA instruction
            Sub(tmpNMinusNLoopCtr, nVal, nLoopCtr);
        }
        B(ARMCG::Cond::GT, nLoopHead);
    }
    GenerateNLoopBody(mLim, nResid, k);
}
 
template <typename SrcWeiType, typename DstBiaType>
void MCBFP16Gen<SrcWeiType, DstBiaType>::GenerateNLoopBody(
    IntType mLim, IntType nLim, IntType k)
{
    IntType vlw = svcntw();
    // Initialize src pointer
    Add(tmp, kVal, NUM_1);
    Lsr(tmp, tmp, NUM_1); // NUM_1 is due to FP16 MOPA peculiarities
    Lsl(tmp, tmp, NUM_1); // NUM_1 is due to FP16 MOPA peculiarities
    Mul(tmp, tmpMMinusMLoopCtr, tmp);
    Add(srcPtrCopy, srcPtr, tmp); // tmpMMinusMLoopCtr is already multiplied by 2 = sizeof(SrcWeiType)
    // initialize predicates for WEI loads
    Mov(tmp, 0);
    // p0, p2 used for SRC, p1, p3 used for WEI
    Whilelt(ARMCG::PRegH(NUM_1), tmp, nLoopCtr);
    if (nLim > vlw) {
        Inch(tmp);
        Whilelt(ARMCG::PRegH(NUM_3), tmp, nLoopCtr);
    }
    // Initialize wei pointer
    Add(tmp, kVal, NUM_1);
    Lsr(tmp, tmp, NUM_1); // NUM_1 is due to FP16 MOPA peculiarities
    Lsl(tmp, tmp, NUM_1); // NUM_1 is due to FP16 MOPA peculiarities
    Mul(tmp, tmpNMinusNLoopCtr, tmp);
    Add(weiPtrCopy, weiPtr, tmp); // tmpNMinusNLoopCtr is already multiplied by 2 = sizeof(SrcWeiType)
 
    // Initialize dst pointer
    Mul(tmp, tmpMMinusMLoopCtr, dstOffset);
    Lsr(tmp, tmp, NUM_1); // NUM_1 is due to FP16 MOPA peculiarities
    // Row-major format of dst
    Add(dstPtrCopy, dstPtr, tmp); // Add stride for M
    Add(dstPtrCopy, dstPtrCopy, tmpNMinusNLoopCtr,
        ARMCG::ShMod::LSL, Math::Log2(sizeof(DstBiaType)) - NUM_1); // Add stride for N
    // tmpNMinusNLoopCtr is already multiplied by 2
    Mov(dstPtrLoopCopy, dstPtrCopy);
    // Initialize bia pointer
    Mul(tmp, tmpMMinusMLoopCtr, biaOffset);
    Lsr(tmp, tmp, NUM_1); // NUM_1 is due to FP16 MOPA peculiarities
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
        ARMCG::ShMod::LSL, Math::Log2(sizeof(DstBiaType)) - NUM_1); // Add stride for N
    // tmpNMinusNLoopCtr is already multiplied by 2
    L(notIncrementBiasByN);
 
    // fill dst registers with zeros
    Zero({ARMCG::ZARegS(NUM_0), ARMCG::ZARegS(NUM_1), ARMCG::ZARegS(NUM_2), ARMCG::ZARegS(NUM_3)});
    // Main k loop
    GenerateKLoop(mLim, nLim, k, vlw);
 
    // Divide 2vl*2vl kernel by 4
    // 4 parts of:  m/2 * 2 vlw
    // Maximum 8 Z register loaded in split part
    IntType Rows = std::min(mLim, vlw);
    IntType Split = Math::CeilDiv(Rows, NUM_2);
    if (Split > 0) {
        MoveDstRegistersPartSplit(Split, nLim, vlw, NUM_0, NUM_1, NUM_0);
        Mov(dstPtrCopy, dstPtrLoopCopy);
        BiasAndC(Split, nLim, vlw);
        Mov(dstPtrCopy, dstPtrLoopCopy);
        StoreDstRegistersPartSplit(Split, nLim, vlw);
    }

    IntType Split2 = Rows - Split;
    if (Split2 > 0) {
        Add(dstPtrCopy, dstPtrCopy, dstOffset);
        Mov(dstPtrStore, dstPtrCopy);
        MoveDstRegistersPartSplit(Split2, nLim, vlw, NUM_0, NUM_1, Split);
        Mov(dstPtrCopy, dstPtrStore);
        Add(biaPtrCopy, biaPtrCopy, biaOffset);
        BiasAndC(Split2, nLim, vlw);
        Mov(dstPtrCopy, dstPtrStore);
        StoreDstRegistersPartSplit(Split2, nLim, vlw);
    }

    IntType RowsResid = std::min(mLim - vlw, vlw);
    IntType Split3 = Math::CeilDiv(RowsResid, NUM_2);
    if (Split3 > 0) {
        Add(dstPtrCopy, dstPtrCopy, dstOffset);
        Mov(dstPtrStore, dstPtrCopy);
        MoveDstRegistersPartSplit(Split3, nLim, vlw, NUM_2, NUM_3, NUM_0);
        Mov(dstPtrCopy, dstPtrStore);
        Add(biaPtrCopy, biaPtrCopy, biaOffset);
        BiasAndC(Split3, nLim, vlw);
        Mov(dstPtrCopy, dstPtrStore);
        StoreDstRegistersPartSplit(Split3, nLim, vlw);
    }

    IntType Split4 = RowsResid - Split3;
    if (Split4 > 0) {
        Add(dstPtrCopy, dstPtrCopy, dstOffset);
        Mov(dstPtrStore, dstPtrCopy);
        MoveDstRegistersPartSplit(Split4, nLim, vlw, NUM_2, NUM_3, Split3);
        Mov(dstPtrCopy, dstPtrStore);
        Add(biaPtrCopy, biaPtrCopy, biaOffset);
        BiasAndC(Split4, nLim, vlw);
        Mov(dstPtrCopy, dstPtrStore);
        StoreDstRegistersPartSplit(Split4, nLim, vlw);
    }
}

template <typename SrcWeiType, typename DstBiaType>
void MCBFP16Gen<SrcWeiType, DstBiaType>::MoveDstRegistersPartSplit(IntType SplitSize, IntType n, IntType vl,
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

template <typename SrcWeiType, typename DstBiaType>
void MCBFP16Gen<SrcWeiType, DstBiaType>::MulAdd(IntType dstIdx, IntType argIdx,
    IntType pregIdx, IntType betaIdx)
{
    Fmul(ARMCG::ZRegS(argIdx), ARMCG::ZRegS(argIdx), ARMCG::ZRegS(betaIdx));
    Fadd(ARMCG::ZRegS(dstIdx), ARMCG::PRegB(pregIdx), ARMCG::ZRegS(argIdx));
}

template <typename SrcWeiType, typename DstBiaType>
void MCBFP16Gen<SrcWeiType, DstBiaType>::AddPair(IntType n, IntType vlw,
    IntType dstIdx1, IntType dstIdx2, IntType argIdx1, IntType argIdx2)
{
    Fadd(ARMCG::ZRegS(dstIdx1), p1, ARMCG::ZRegS(argIdx1));
    if (n > vlw) {
        Fadd(ARMCG::ZRegS(dstIdx2), p3, ARMCG::ZRegS(argIdx2));
    }
}

template <typename SrcWeiType, typename DstBiaType>
void MCBFP16Gen<SrcWeiType, DstBiaType>::BiasAndC(IntType SplitSize, IntType n, IntType vlw)
{
    // Initialize Dst matrix registers
    ARMCG::Label bf16MCLoadC;
    ARMCG::Label bf16MCMullC;
    ARMCG::Label bf16MCLoadBias1;
    ARMCG::Label bf16MCLoadBiasM;
    ARMCG::Label bf16MCLoadBiasN;
    ARMCG::Label bf16MCLoadBiasMN;
    ARMCG::Label bf16MCAddBias1;
    ARMCG::Label bf16MCAddBiasM;
    ARMCG::Label bf16MCAddBiasN;
    ARMCG::Label bf16MCAddBiasMN;
    ARMCG::Label bf16MCInitDone;
    ARMCG::Label bf16MCSkipPO;

    KUDNN_JIT_Generator_With_Attributes::TakePredicateRegister(NUM_5);
    Ptrue(ARMCG::PRegS(NUM_5), ARMCG::Pattern::ALL);
    if constexpr (!std::is_same_v<float, DstBiaType>) {
        KUDNN_JIT_Generator_With_Attributes::TakePredicateRegister(NUM_6);
        KUDNN_JIT_Generator_With_Attributes::TakePredicateRegister(NUM_7);
        Ptrue(ARMCG::PRegH(NUM_7), ARMCG::Pattern::ALL);
        Mov(tmp, nLoopCtr);
        Lsr(tmp, tmp, NUM_1);
        Whilelt(ARMCG::PRegH(NUM_6), xzr, tmp);
    }

    Cmp(alphaW, fOne);
    Beq(bf16MCLoadBias1);
    Dup(ARMCG::ZRegS(NUM_30), alphaW);
    for (IntType i = 0; i < SplitSize; ++i) {
        Fmul(ARMCG::ZRegS(i * NUM_2), ARMCG::ZRegS(i * NUM_2), ARMCG::ZRegS(NUM_30));
        if (n > vlw) {
            Fmul(ARMCG::ZRegS(i * NUM_2 + NUM_1), ARMCG::ZRegS(i * NUM_2 + NUM_1), ARMCG::ZRegS(NUM_30));
        }
    }

    Tbz(flags, 0, bf16MCLoadBias1); // 0-bit correspond to FILL_C_WITH_ZEROZ
    L(bf16MCLoadBias1);
    Tbz(flags, 3, bf16MCLoadBiasN); // 3-bit correspond to LOAD_BIAS_1
    if constexpr (!std::is_same_v<float, DstBiaType>) {
        Ldrh(tmpW, ARMCG::AdrUimm(Ptr(biaPtr)));
        Mov(ARMCG::ZRegB(NUM_30), 0);
        Dup(ARMCG::ZRegH(NUM_31), tmpW);
        Bfp16Cvt(NUM_28, NUM_29, NUM_31, NUM_30, ARMCG::PRegH(NUM_7));
    } else {
        Ldr(tmpW, ARMCG::AdrUimm(Ptr(biaPtr)));
        Cpy(ARMCG::ZRegS(NUM_29), p5, tmpW); // z29 is used as tmp register here
    }
    for (IntType i = 0; i < SplitSize; ++i) {
        AddPair(n, vlw, i * NUM_2, i * NUM_2 + NUM_1, NUM_29, NUM_29);
    }

    L(bf16MCLoadBiasN);
    Tbz(flags, 5, bf16MCLoadBiasM); // 5-bit correspond to LOAD_BIAS_N
    if constexpr (!std::is_same_v<float, DstBiaType>) {
        Mov(ARMCG::ZRegB(NUM_30), 0);
        Ld1h(ARMCG::ZRegH(NUM_31), p6, Ptr(biaPtrCopy));
        Bfp16Cvt(NUM_28, NUM_29, NUM_31, NUM_30, ARMCG::PRegH(NUM_7));
    } else {
        Ld1w(ARMCG::ZRegS(NUM_28), p1, Ptr(biaPtrCopy)); // z28 is used as tmp register here
        if (n > vlw) {
            // z29 is used as tmp register here
            Ld1w(ARMCG::ZRegS(NUM_29), p3, Ptr(biaPtrCopy, NUM_1, ARMCG::ExtMod::MUL_VL));
        }
    }
    for (IntType i = 0; i < SplitSize; ++i) {
        AddPair(n, vlw, i * NUM_2, i * NUM_2 + NUM_1, NUM_28, NUM_29);
    }

    L(bf16MCLoadBiasM);
    Tbz(flags, 4, bf16MCLoadBiasMN); // 4-bit correspond to LOAD_BIAS_M
    for (IntType i = 0; i < SplitSize; ++i) {
        if (i != 0) {
            Add(biaPtrCopy, biaPtrCopy, static_cast<std::uint32_t>(sizeof(DstBiaType)));
        }
        if constexpr (!std::is_same_v<float, DstBiaType>) {
            Mov(ARMCG::ZRegB(NUM_30), 0);
            Ld1rh(ARMCG::ZRegH(NUM_31), p6, Ptr(biaPtrCopy));
            Bfp16Cvt(NUM_28, NUM_29, NUM_31, NUM_30, ARMCG::PRegH(NUM_7));
        } else {
            Ld1rw(ARMCG::ZRegS(NUM_28), p1, Ptr(biaPtrCopy)); // z28 is used as tmp register here
        }
        AddPair(n, vlw, i * NUM_2, i * NUM_2 + NUM_1, NUM_28, NUM_28);
    }

    L(bf16MCLoadBiasMN);
    Tbz(flags, 6, bf16MCLoadC); // 6-bit correspond to LOAD_BIAS_MN
    for (IntType i = 0; i < SplitSize; ++i) {
        if (i != 0) {
            Add(biaPtrCopy, biaPtrCopy, biaOffset);
        }
        if constexpr (!std::is_same_v<float, DstBiaType>) {
            Mov(ARMCG::ZRegB(NUM_30), 0);
            Ld1h(ARMCG::ZRegH(NUM_31), p6, Ptr(biaPtrCopy));
            Bfp16Cvt(NUM_28, NUM_29, NUM_31, NUM_30, ARMCG::PRegH(NUM_7));
        } else {
            Ld1w(ARMCG::ZRegS(NUM_28), p1, Ptr(biaPtrCopy)); // z28 is used as tmp register here
            if (n > vlw) {
                // z29 is used as tmp register here
                Ld1w(ARMCG::ZRegS(NUM_29), p3, Ptr(biaPtrCopy, NUM_1, ARMCG::ExtMod::MUL_VL));
            }
        }
        AddPair(n, vlw, i * NUM_2, i * NUM_2 + NUM_1, NUM_28, NUM_29);
    }

    L(bf16MCLoadC);
    Tbz(flags, 1, bf16MCMullC); // 1-bit correspond to LOAD_C
    for (IntType i = 0; i < SplitSize; ++i) {
        if (i != 0) {
            Add(dstPtrCopy, dstPtrCopy, dstOffset);
        }
        if constexpr (!std::is_same_v<float, DstBiaType>) {
            Mov(ARMCG::ZRegB(NUM_30), 0);
            Ld1h(ARMCG::ZRegH(NUM_31), p6, Ptr(dstPtrCopy));
            Bfp16Cvt(NUM_28, NUM_29, NUM_31, NUM_30, ARMCG::PRegH(NUM_7));
        } else {
            Ld1w(ARMCG::ZRegS(NUM_28), p1, Ptr(dstPtrCopy)); // z28 is used as tmp register here
            if (n > vlw) {
                // z29 is used as tmp register here
                Ld1w(ARMCG::ZRegS(NUM_29), p3, Ptr(dstPtrCopy, NUM_1, ARMCG::ExtMod::MUL_VL));
            }
        }
        Fadd(ARMCG::ZRegS(i * NUM_2), p1, ARMCG::ZRegS(NUM_28));
        if (n > vlw) {
            Fadd(ARMCG::ZRegS(i * NUM_2 + NUM_1), p3, ARMCG::ZRegS(NUM_29));
        }
    }

    L(bf16MCMullC);
    Tbz(flags, 2, bf16MCAddBias1); // 2-bit correspond to MULL_C
    Dup(ARMCG::ZRegS(NUM_31), betaW);
    for (IntType i = 0; i < SplitSize; ++i) {
        if (i != 0) {
            Add(dstPtrCopy, dstPtrCopy, dstOffset);
        }
        if constexpr (!std::is_same_v<float, DstBiaType>) {
            Mov(ARMCG::ZRegB(NUM_29), 0);
            Ld1h(ARMCG::ZRegH(NUM_30), p6, Ptr(dstPtrCopy));
            Bfp16Cvt(NUM_27, NUM_28, NUM_30, NUM_29, ARMCG::PRegH(NUM_7));
        } else {
            Ld1w(ARMCG::ZRegS(NUM_27), p1, Ptr(dstPtrCopy)); // z27 is used as tmp register here
            if (n > vlw) {
                // z28 is used as tmp register here
                Ld1w(ARMCG::ZRegS(NUM_28), p3, Ptr(dstPtrCopy, NUM_1, ARMCG::ExtMod::MUL_VL));
            }
        }
        MulAdd(i * NUM_2, NUM_27, NUM_1, NUM_31);
        if (n > vlw) {
            MulAdd(i * NUM_2 + NUM_1, NUM_28, NUM_3, NUM_31);
        }
    }

    L(bf16MCAddBias1);
    Tbz(flags, 7, bf16MCAddBiasN); // 7-bit correspond to Add_BIAS_1
    if constexpr (!std::is_same_v<float, DstBiaType>) {
        Ldrh(tmpW, ARMCG::AdrUimm(Ptr(biaPtr)));
        Mov(ARMCG::ZRegB(NUM_30), 0);
        Dup(ARMCG::ZRegH(NUM_31), tmpW);
        Bfp16Cvt(NUM_28, NUM_29, NUM_31, NUM_30, ARMCG::PRegH(NUM_7));
    } else {
        Ldr(tmpW, ARMCG::AdrUimm(Ptr(biaPtr)));
        Cpy(ARMCG::ZRegS(NUM_29), p5, tmpW); // z29 is used as tmp register here
    }
    for (IntType i = 0; i < SplitSize; ++i) {
        AddPair(n, vlw, i * NUM_2, i * NUM_2 + NUM_1, NUM_29, NUM_29);
    }

    L(bf16MCAddBiasN);
    Tbz(flags, 9, bf16MCAddBiasM); // 5-bit correspond to ADD_BIAS_N
    if constexpr (!std::is_same_v<float, DstBiaType>) {
        Mov(ARMCG::ZRegB(NUM_30), 0);
        Ld1h(ARMCG::ZRegH(NUM_31), p6, Ptr(biaPtrCopy));
        Bfp16Cvt(NUM_28, NUM_29, NUM_31, NUM_30, ARMCG::PRegH(NUM_7));
    } else {
        Ld1w(ARMCG::ZRegS(NUM_28), p1, Ptr(biaPtrCopy)); // z28 is used as tmp register here
        if (n > vlw) {
            // z29 is used as tmp register here
            Ld1w(ARMCG::ZRegS(NUM_29), p3, Ptr(biaPtrCopy, NUM_1, ARMCG::ExtMod::MUL_VL));
        }
    }
    for (IntType i = 0; i < SplitSize; ++i) {
        AddPair(n, vlw, i * NUM_2, i * NUM_2 + NUM_1, NUM_28, NUM_29);
    }

    L(bf16MCAddBiasM);
    Tbz(flags, 8, bf16MCAddBiasMN); // 8-bit correspond to ADD_BIAS_M
    for (IntType i = 0; i < SplitSize; ++i) {
        if (i != 0) {
            Add(biaPtrCopy, biaPtrCopy, static_cast<std::uint32_t>(sizeof(DstBiaType)));
        }
        if constexpr (!std::is_same_v<float, DstBiaType>) {
            Mov(ARMCG::ZRegB(NUM_30), 0);
            Ld1rh(ARMCG::ZRegH(NUM_31), p6, Ptr(biaPtrCopy));
            Bfp16Cvt(NUM_28, NUM_29, NUM_30, NUM_31, ARMCG::PRegH(NUM_7));
        } else {
            Ld1rw(ARMCG::ZRegS(NUM_28), p1, Ptr(biaPtrCopy)); // z28 is used as tmp register here
        }
        AddPair(n, vlw, i * NUM_2, i * NUM_2 + NUM_1, NUM_28, NUM_28);
    }

    L(bf16MCAddBiasMN);
    Tbz(flags, 10, bf16MCInitDone); // 10-bit correspond to ADD_BIAS_MN
    for (IntType i = 0; i < SplitSize; ++i) {
        if (i != 0) {
            Add(biaPtrCopy, biaPtrCopy, biaOffset);
        }
        if constexpr (!std::is_same_v<float, DstBiaType>) {
            Mov(ARMCG::ZRegB(NUM_30), 0);
            Ld1h(ARMCG::ZRegH(NUM_31), p6, Ptr(biaPtrCopy));
            Bfp16Cvt(NUM_28, NUM_29, NUM_31, NUM_30, ARMCG::PRegH(NUM_7));
        } else {
            Ld1w(ARMCG::ZRegS(NUM_28), p1, Ptr(biaPtrCopy)); // z28 is used as tmp register here
            if (n > vlw) {
                // z29 is used as tmp register here
                Ld1w(ARMCG::ZRegS(NUM_29), p3, Ptr(biaPtrCopy, NUM_1, ARMCG::ExtMod::MUL_VL));
            }
        }
        AddPair(n, vlw, i * NUM_2, i * NUM_2 + NUM_1, NUM_28, NUM_29);
    }

    L(bf16MCInitDone);
    Tbz(flags, 11, bf16MCSkipPO); // 11-bit correspond to APPLY_POST_OPS
    IntType size = SplitSize;
    if (n > vlw) {
        size += SplitSize;
    }
    KUDNN_JIT_Generator_With_Attributes::FreePredicateRegister(NUM_5);
    if constexpr (!std::is_same_v<float, DstBiaType>) {
        KUDNN_JIT_Generator_With_Attributes::FreePredicateRegister(NUM_6);
        KUDNN_JIT_Generator_With_Attributes::FreePredicateRegister(NUM_7);
    }
    std::vector<SizeType> dstRegsIDs(size);
    FillDstRegs(SplitSize, n, vlw, dstRegsIDs);
    for (std::uint32_t i = 0; i < size; ++i) {
        KUDNN_JIT_Generator_With_Attributes::TakeVectorRegister(dstRegsIDs[i]);
    }
    this->ApplyPostOps(dstRegsIDs, Element::MatchType<float>());
    for (std::uint32_t i = 0; i < size; ++i) {
        KUDNN_JIT_Generator_With_Attributes::FreeVectorRegister(dstRegsIDs[i]);
    }
    L(bf16MCSkipPO);
}

template <typename SrcWeiType, typename DstBiaType>
void MCBFP16Gen<SrcWeiType, DstBiaType>::FillDstRegs(
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

template <typename SrcWeiType, typename DstBiaType>
void MCBFP16Gen<SrcWeiType, DstBiaType>::StoreDstRegistersPartSplit(IntType SplitSize, IntType n, IntType vl)
{
    for (IntType i = 0; i < SplitSize; ++i) {
        if (i != 0) {
            Add(dstPtrCopy, dstPtrCopy, dstOffset);
        }
        if constexpr (std::is_same_v<float, DstBiaType>) {
            St1w(ARMCG::ZRegS(i * NUM_2), p1, Ptr(dstPtrCopy));
            if (n > vl) {
                St1w(ARMCG::ZRegS(i * NUM_2 + NUM_1), p3, Ptr(dstPtrCopy, NUM_1, ARMCG::ExtMod::MUL_VL));
            }
        } else {
            Mov(tmp, nLoopCtr);
            Lsr(tmp, tmp, NUM_1);
            Whilelt(ARMCG::PRegH(NUM_6), xzr, tmp);
            Bfp16Cvt(ARMCG::ZRegH(i * NUM_2), p1, ARMCG::ZRegS(i * NUM_2));
            Bfp16Cvt(ARMCG::ZRegH(i * NUM_2 + NUM_1), p3, ARMCG::ZRegS(i * NUM_2 + NUM_1));
            Uzp1(ARMCG::ZRegH(i * NUM_2), ARMCG::ZRegH(i * NUM_2), ARMCG::ZRegH(i * NUM_2 + NUM_1));
            St1h(ARMCG::ZRegH(i * NUM_2), p6, Ptr(dstPtrCopy));
        }
    }
}

template <typename SrcWeiType, typename DstBiaType>
void MCBFP16Gen<SrcWeiType, DstBiaType>::GenerateKLoop(
    IntType m, IntType n, IntType k, IntType vlw)
{
    // NUM_2 due to peculiarities of FP16 MOPA instruction
    IntType kRoundedUp = Math::CeilDiv(k, NUM_2 * K_LOOP_UNROLL) * NUM_2 * K_LOOP_UNROLL;
    ARMCG::Label kLoopHead;
    Mov(kTmpCopy, kVal);
    Subs(kTmpCopy, kTmpCopy, K_LOOP_UNROLL * NUM_2); // NUM_2 due to peculiarities of FP16 MOPA instruction
    // p0, p2 - for SRC, p1, p3 - for WEI
    // z0, z2, z4, z6 - for SRC, z1, z3, z5, z7 - for WEI
    // SRC and WEI data is prepacked, so it's continious in memory.
    // K unroll by 2 requires 4 elements from SRC,  4 from WEI
    // Load src data
    Ld1h(ARMCG::ZRegH(NUM_0), p0, Ptr(srcPtrCopy));
    if (m > vlw) {
        Ld1h(ARMCG::ZRegH(NUM_2), p2, Ptr(srcPtrCopy, NUM_1, ARMCG::ExtMod::MUL_VL));
    }
    // Load wei data
    Ld1h(ARMCG::ZRegH(NUM_1), p1, Ptr(weiPtrCopy));
    if (n > vlw) {
        Ld1h(ARMCG::ZRegH(NUM_3), p3, Ptr(weiPtrCopy, NUM_1, ARMCG::ExtMod::MUL_VL));
    }
    if (kRoundedUp >= K_LOOP_UNROLL * NUM_2) { // NUM_2 due to peculiarities of FP16 MOPA instruction
        // Load src data for unrolled loop
        Ld1h(ARMCG::ZRegH(NUM_4), p0, Ptr(srcPtrCopy, NUM_2, ARMCG::ExtMod::MUL_VL));
        if (m > vlw) {
            Ld1h(ARMCG::ZRegH(NUM_6), p2, Ptr(srcPtrCopy, NUM_3, ARMCG::ExtMod::MUL_VL));
        }
        // Load wei data unrolled loop
        Ld1h(ARMCG::ZRegH(NUM_5), p1, Ptr(weiPtrCopy, NUM_2, ARMCG::ExtMod::MUL_VL));
        if (n > vlw) {
            Ld1h(ARMCG::ZRegH(NUM_7), p3, Ptr(weiPtrCopy, NUM_3, ARMCG::ExtMod::MUL_VL));
        }
        Addvl(srcPtrCopy, srcPtrCopy, KERNEL_M_SIZE * K_LOOP_UNROLL); // KERNEL_M_SIZE * UnrollK = 4
        Addvl(weiPtrCopy, weiPtrCopy, KERNEL_N_SIZE * K_LOOP_UNROLL); // KERNEL_N_SIZE * UnrollK = 4
        if (kRoundedUp > (K_LOOP_UNROLL * NUM_2)) { // NUM_2 due to peculiarities of FP16 MOPA instruction
            L(kLoopHead);
                Bfp16Mopa(ARMCG::ZARegS(NUM_0), p0, p1, ARMCG::ZRegH(NUM_0), ARMCG::ZRegH(NUM_1));
                Subs(kTmpCopy, kTmpCopy, (K_LOOP_UNROLL * NUM_2)); // NUM_2 due to peculiarities of FP16 MOPA instr
                if (n > vlw) {
                    Bfp16Mopa(ARMCG::ZARegS(NUM_1), p0, p3, ARMCG::ZRegH(NUM_0), ARMCG::ZRegH(NUM_3));
                }
                Ld1h(ARMCG::ZRegH(NUM_0), p0, Ptr(srcPtrCopy));
                if (m > vlw) {
                    Bfp16Mopa(ARMCG::ZARegS(NUM_2), p2, p1, ARMCG::ZRegH(NUM_2), ARMCG::ZRegH(NUM_1));
                }
                Ld1h(ARMCG::ZRegH(NUM_1), p1, Ptr(weiPtrCopy));
                if ((m > vlw) && (n > vlw)) {
                    Bfp16Mopa(ARMCG::ZARegS(NUM_3), p2, p3, ARMCG::ZRegH(NUM_2), ARMCG::ZRegH(NUM_3));
                }
                if (m > vlw) {
                    Ld1h(ARMCG::ZRegH(NUM_2), p2, Ptr(srcPtrCopy, NUM_1, ARMCG::ExtMod::MUL_VL));
                }
                if (n > vlw) {
                    Ld1h(ARMCG::ZRegH(NUM_3), p3, Ptr(weiPtrCopy, NUM_1, ARMCG::ExtMod::MUL_VL));
                }

                Bfp16Mopa(ARMCG::ZARegS(NUM_0), p0, p1, ARMCG::ZRegH(NUM_4), ARMCG::ZRegH(NUM_5));
                if (n > vlw) {
                    Bfp16Mopa(ARMCG::ZARegS(NUM_1), p0, p3, ARMCG::ZRegH(NUM_4), ARMCG::ZRegH(NUM_7));
                }
                Ld1h(ARMCG::ZRegH(NUM_4), p0, Ptr(srcPtrCopy, NUM_2, ARMCG::ExtMod::MUL_VL));
                if (m > vlw) {
                    Bfp16Mopa(ARMCG::ZARegS(NUM_2), p2, p1, ARMCG::ZRegH(NUM_6), ARMCG::ZRegH(NUM_5));
                }
                Ld1h(ARMCG::ZRegH(NUM_5), p1, Ptr(weiPtrCopy, NUM_2, ARMCG::ExtMod::MUL_VL));
                if ((m > vlw) && (n > vlw)) {
                    Bfp16Mopa(ARMCG::ZARegS(NUM_3), p2, p3, ARMCG::ZRegH(NUM_6), ARMCG::ZRegH(NUM_7));
                }
                if (m > vlw) {
                    Ld1h(ARMCG::ZRegH(NUM_6), p2, Ptr(srcPtrCopy, NUM_3, ARMCG::ExtMod::MUL_VL));
                }
                if (n > vlw) {
                    Ld1h(ARMCG::ZRegH(NUM_7), p3, Ptr(weiPtrCopy, NUM_3, ARMCG::ExtMod::MUL_VL));
                }

                Addvl(srcPtrCopy, srcPtrCopy, KERNEL_M_SIZE * K_LOOP_UNROLL);
                Addvl(weiPtrCopy, weiPtrCopy, KERNEL_N_SIZE * K_LOOP_UNROLL);

            B(ARMCG::Cond::GT, kLoopHead);
        }
    }

    Bfp16Mopa(ARMCG::ZARegS(NUM_0), p0, p1, ARMCG::ZRegH(NUM_0), ARMCG::ZRegH(NUM_1));
    if (n > vlw) {
        Bfp16Mopa(ARMCG::ZARegS(NUM_1), p0, p3, ARMCG::ZRegH(NUM_0), ARMCG::ZRegH(NUM_3));
    }
    if (m > vlw) {
        Bfp16Mopa(ARMCG::ZARegS(NUM_2), p2, p1, ARMCG::ZRegH(NUM_2), ARMCG::ZRegH(NUM_1));
    }
    if ((m > vlw) && (n > vlw)) {
        Bfp16Mopa(ARMCG::ZARegS(NUM_3), p2, p3, ARMCG::ZRegH(NUM_2), ARMCG::ZRegH(NUM_3));
    }
    // NUM_2 due to peculiarities of FP16 MOPA instruction
    if ((k == kRoundedUp) || (k % (K_LOOP_UNROLL * NUM_2) > NUM_2)) {
        Bfp16Mopa(ARMCG::ZARegS(NUM_0), p0, p1, ARMCG::ZRegH(NUM_4), ARMCG::ZRegH(NUM_5));
        if (n > vlw) {
            Bfp16Mopa(ARMCG::ZARegS(NUM_1), p0, p3, ARMCG::ZRegH(NUM_4), ARMCG::ZRegH(NUM_7));
        }
        if (m > vlw) {
            Bfp16Mopa(ARMCG::ZARegS(NUM_2), p2, p1, ARMCG::ZRegH(NUM_6), ARMCG::ZRegH(NUM_5));
        }
        if ((m > vlw) && (n > vlw)) {
            Bfp16Mopa(ARMCG::ZARegS(NUM_3), p2, p3, ARMCG::ZRegH(NUM_6), ARMCG::ZRegH(NUM_7));
        }
    }
}

template <typename SrcWeiType, typename DstBiaType>
void MCBFP16Gen<SrcWeiType, DstBiaType>::StoreDstRegisterPartSplitBF16(IntType SplitSize)
{
    for (IntType i = 0; i < SplitSize; ++i) {
        if (i != 0) {
            Add(dstPtrCopy, dstPtrCopy, dstOffset);
        }

        Whilelt(ARMCG::PRegH(NUM_6), xzr, nVal);
        Bfp16Cvt(ARMCG::ZRegH(i * NUM_2), p1, ARMCG::ZRegS(i * NUM_2));
        Bfp16Cvt(ARMCG::ZRegH(i * NUM_2 + NUM_1), p3, ARMCG::ZRegS(i * NUM_2 + NUM_1));
        Uzp1(ARMCG::ZRegH(i * NUM_2), ARMCG::ZRegH(i * NUM_2), ARMCG::ZRegH(i * NUM_2 + NUM_1));
        St1h(ARMCG::ZRegH(i * NUM_2), p6, Ptr(dstPtrCopy));
    }
}

template <typename SrcWeiType, typename DstBiaType>
void MCBFP16Gen<SrcWeiType, DstBiaType>::StoreDstRegisters(
    IntType m, IntType n, IntType vlw)
{
    for (IntType i = 0; i < std::min(m, vlw); ++i) {
        if (i == 0) {
            Mov(dstPtrCopy, dstPtr);
        } else {
            Add(dstPtrCopy, dstPtrCopy, dstOffset);
        }

        Mova(ARMCG::ZRegS(i * NUM_2), p1, ARMCG::ZARegHS(0),
            ARMCG::WReg(NUM_12 + i / NUM_4), i % NUM_4);
        if (n > vlw) {
            Mova(ARMCG::ZRegS(i * NUM_2 + NUM_1), p3, ARMCG::ZARegHS(NUM_1),
                ARMCG::WReg(NUM_12 + i / NUM_4), i % NUM_4);
        }
        if constexpr (std::is_same_v<float, DstBiaType>) {
            St1w(ARMCG::ZRegS(i * NUM_2), p1, Ptr(dstPtrCopy));
            if (n > vlw) {
                St1w(ARMCG::ZRegS(i * NUM_2 + NUM_1), p3, Ptr(dstPtrCopy, NUM_1, ARMCG::ExtMod::MUL_VL));
            }
        } else {
            Whilelt(ARMCG::PRegH(NUM_6), xzr, nVal);
            Bfp16Cvt(ARMCG::ZRegH(i * NUM_2), p1, ARMCG::ZRegS(i * NUM_2));
            Bfp16Cvt(ARMCG::ZRegH(i * NUM_2 + NUM_1), p3, ARMCG::ZRegS(i * NUM_2 + NUM_1));
            Uzp1(ARMCG::ZRegH(i * NUM_2), ARMCG::ZRegH(i * NUM_2), ARMCG::ZRegH(i * NUM_2 + NUM_1));
            St1h(ARMCG::ZRegH(i * NUM_2), p6, Ptr(dstPtrCopy));
        }
    }

    for (IntType i = 0; i < std::min(m - vlw, vlw); ++i) {
        Add(dstPtrCopy, dstPtrCopy, dstOffset);

        Mova(ARMCG::ZRegS(i * NUM_2), p1, ARMCG::ZARegHS(NUM_2),
            ARMCG::WReg(NUM_12 + i / NUM_4), i % NUM_4);
        if (n > vlw) {
            Mova(ARMCG::ZRegS(i * NUM_2 + NUM_1), p3, ARMCG::ZARegHS(NUM_3),
                ARMCG::WReg(NUM_12 + i / NUM_4), i % NUM_4);
        }
        if constexpr (std::is_same_v<float, DstBiaType>) {
            St1w(ARMCG::ZRegS(i * NUM_2), p1, Ptr(dstPtrCopy));
            if (n > vlw) {
                St1w(ARMCG::ZRegS(i * NUM_2 + NUM_1), p3, Ptr(dstPtrCopy, NUM_1, ARMCG::ExtMod::MUL_VL));
            }
        } else {
            Whilelt(ARMCG::PRegH(NUM_6), xzr, nVal);
            Bfp16Cvt(ARMCG::ZRegH(i * NUM_2), p1, ARMCG::ZRegS(i * NUM_2));
            Bfp16Cvt(ARMCG::ZRegH(i * NUM_2 + NUM_1), p3, ARMCG::ZRegS(i * NUM_2 + NUM_1));
            Uzp1(ARMCG::ZRegH(i * NUM_2), ARMCG::ZRegH(i * NUM_2), ARMCG::ZRegH(i * NUM_2 + NUM_1));
            St1h(ARMCG::ZRegH(i * NUM_2), p6, Ptr(dstPtrCopy));
        }
    }
}

template <typename SrcWeiType, typename DstBiaType>
void MCBFP16Gen<SrcWeiType, DstBiaType>::Bfp16Cvt(const ARMCG::ZRegH &zd, const ARMCG::PRegBase &pg,
    const ARMCG::ZRegS &zn)
{
    if constexpr (std::is_same_v<SrcWeiType, __fp16>) {
        Fcvt(zd, pg, zn);
    } else {
        Bfcvt(zd, pg, zn);
    }
}

template <typename SrcWeiType, typename DstBiaType>
void MCBFP16Gen<SrcWeiType, DstBiaType>::Bfp16Cvt(IntType zd0Idx, IntType zd1Idx,
    IntType znDataIdx, IntType znZerosIdx, const ARMCG::PRegBase &pg)
{
    if constexpr (std::is_same_v<SrcWeiType, __fp16>) {
        Trn2(ARMCG::ZRegH(znZerosIdx), ARMCG::ZRegH(znDataIdx), ARMCG::ZRegH(znZerosIdx));
        Fcvt(ARMCG::ZRegS(znDataIdx), pg, ARMCG::ZRegH(znDataIdx));
        Fcvt(ARMCG::ZRegS(znZerosIdx), pg, ARMCG::ZRegH(znZerosIdx));
        Zip1(ARMCG::ZRegS(zd0Idx), ARMCG::ZRegS(znDataIdx), ARMCG::ZRegS(znZerosIdx));
        Zip2(ARMCG::ZRegS(zd1Idx), ARMCG::ZRegS(znDataIdx), ARMCG::ZRegS(znZerosIdx));
    } else {
        Zip1(ARMCG::ZRegH(zd0Idx), ARMCG::ZRegH(znZerosIdx), ARMCG::ZRegH(znDataIdx));
        Zip2(ARMCG::ZRegH(zd1Idx), ARMCG::ZRegH(znZerosIdx), ARMCG::ZRegH(znDataIdx));
    }
}

template <typename SrcWeiType, typename DstBiaType>
void MCBFP16Gen<SrcWeiType, DstBiaType>::Bfp16Mopa(const ARMCG::ZARegS &zada, const ARMCG::PRegBase &pn,
    const ARMCG::PRegBase &pm, const ARMCG::ZRegH &zn, const ARMCG::ZRegH &zm)
{
    if constexpr (std::is_same_v<SrcWeiType, __fp16>) {
        Fmopa(zada, pn, pm, zn, zm);
    } else {
        Bfmopa(zada, pn, pm, zn, zm);
    }
}

template <typename SrcWeiType, typename DstBiaType>
MatMulImplGenerator<MCBFP16Gen<SrcWeiType, DstBiaType>>::DefaultImpl::DefaultImpl(
    const GemmInfo &task, IntType nThreads) : MatMulImplBase(task, nThreads)
{
    VERBOSE_PRINT;
    UpdateExecParams();
    Initialize();
}

template <typename SrcWeiType, typename DstBiaType>
std::uint32_t MatMulImplGenerator<MCBFP16Gen<SrcWeiType, DstBiaType>>::DefaultImpl::GetKernelMSize() noexcept
{
    return MCBFP16Gen<SrcWeiType, DstBiaType>::KERNEL_M_SIZE * svcntw();
}

template <typename SrcWeiType, typename DstBiaType>
std::uint32_t MatMulImplGenerator<MCBFP16Gen<SrcWeiType, DstBiaType>>::DefaultImpl::GetKernelNSize() noexcept
{
    return MCBFP16Gen<SrcWeiType, DstBiaType>::KERNEL_N_SIZE * svcntw();
}

template <typename SrcWeiType, typename DstBiaType>
IntType MatMulImplGenerator<MCBFP16Gen<SrcWeiType, DstBiaType>>::DefaultImpl::GetMBlockSize() const noexcept
{
    return 224; // 224 experimentaly measured constants to obtain better perf
}

template <typename SrcWeiType, typename DstBiaType>
IntType MatMulImplGenerator<MCBFP16Gen<SrcWeiType, DstBiaType>>::DefaultImpl::GetNBlockSize() const noexcept
{
    return 64; // 64 is experimentaly measured  constants to obtain better perf
}

template <typename SrcWeiType, typename DstBiaType>
IntType MatMulImplGenerator<MCBFP16Gen<SrcWeiType, DstBiaType>>::DefaultImpl::GetKBlockSize() const noexcept
{
    IntType k = taskExec.srcInfo.GetDims()[taskExec.srcInfo.GetNumDims() - NUM_1];
    IntType defKblock = 512; // 512 is experimentally measured constant to obtain better perf
    IntType numKblocks = Math::CeilDiv(k, defKblock);
    IntType residBlock = k % defKblock;
    IntType kBlock = k;
    if (k > NUM_3 * defKblock) {
        kBlock = defKblock + Math::CeilDiv(residBlock, numKblocks - NUM_1);
    } else if (k > defKblock) {
        kBlock = k / NUM_2;
    }
    return Math::CeilDiv(kBlock, NUM_2) * NUM_2;
}

template <typename SrcWeiType, typename DstBiaType>
void MatMulImplGenerator<MCBFP16Gen<SrcWeiType, DstBiaType>>::DefaultImpl::InitializePackingPolicy()
{
    packingPolicy = PackingPolicy::NO_SYNC;
    InitializePackingPolicyFromEnv();
    if ((packingPolicy == PackingPolicy::SYNC_ACC_DIM_SRC) ||
        (packingPolicy == PackingPolicy::SYNC_ACC_DIM_WEI) ||
        (packingPolicy == PackingPolicy::SYNC_ACC_DIM_SRC_WEI)) {
        throw Service::LogicError("Matmul: Current packing policy not supported for this precision");
    }
}

template <typename SrcWeiType, typename DstBiaType>
void MatMulImplGenerator<MCBFP16Gen<SrcWeiType, DstBiaType>>::DefaultImpl::InitializeSingleCodegen(
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
        CodeGenParams<MCBFP16Gen<SrcWeiType, DstBiaType>> currTask{currentM, currentN, currentK, false, attributes};
        if (codegens.find(currTask) == codegens.end()) {
            codegens.insert({currTask, std::make_shared<MCBFP16Gen<SrcWeiType, DstBiaType>>(
                currentM, currentN, currentK, attributes)});
        }
        codegens[currTask]->Ready();
        matmulCodegenKernels[id] =
            codegens[currTask]->template GetCode<MCBFP16CodegenFuncPtr<SrcWeiType, DstBiaType>>();
    }
}

template <typename SrcWeiType, typename DstBiaType>
void MatMulImplGenerator<MCBFP16Gen<SrcWeiType, DstBiaType>>::DefaultImpl::InitializeReorderCodeGens(
    IntType m, IntType n, IntType k, IntType mFull, IntType nFull)
{
    IntType vlw = svcntw();
    IntType mFullThreadBlock = mFull;
    IntType mResidThreadBlock = m % mFullThreadBlock;
    IntType nFullThreadBlock = nFull;
    IntType nResidThreadBlock = n % nFullThreadBlock;

    IntType mSubBlockExecFull = MCBFP16Gen<SrcWeiType, DstBiaType>::KERNEL_M_SIZE * vlw;
    IntType mSubBlockExecResid1 = mFullThreadBlock % mSubBlockExecFull;
    IntType mSubBlockExecResid2 = mResidThreadBlock % mSubBlockExecFull;

    IntType nSubBlockExecFull = MCBFP16Gen<SrcWeiType, DstBiaType>::KERNEL_N_SIZE * vlw;
    IntType nSubBlockExecResid1 = nFullThreadBlock % nSubBlockExecFull;
    IntType nSubBlockExecResid2 = nResidThreadBlock % nSubBlockExecFull;

    IntType kBlockExecFull = GetKBlockSize();
    IntType kBlockExecResid = k % GetKBlockSize();

    // FP16 specific ==> NUM_2 = sizeof(float) / sizeof(__fp16)
    CodeGenWrapper<FastReorderKernelCodeGen<float, float>> reorderFBFKNA; // full Block full K NoAlpha
    reorderFBFKNA.Initialize(static_cast<SizeType>(vlw),
                             static_cast<SizeType>(kBlockExecFull / NUM_2), false);
    CodeGenWrapper<FastReorderKernelCodeGen<float, float>> reorderRN1FKNA; // resid N 1 full K NoAlpha
    reorderRN1FKNA.Initialize(static_cast<SizeType>(nSubBlockExecResid1 % vlw),
                              static_cast<SizeType>(kBlockExecFull / NUM_2), false);
    CodeGenWrapper<FastReorderKernelCodeGen<float, float>> reorderRN2FKNA; // resid N 2 full K NoAlpha
    reorderRN2FKNA.Initialize(static_cast<SizeType>(nSubBlockExecResid2 % vlw),
                              static_cast<SizeType>(kBlockExecFull / NUM_2), false);
    CodeGenWrapper<FastReorderKernelCodeGen<float, float>> reorderRM1FKNA; // resid M 1 full K NoAlpha
    reorderRM1FKNA.Initialize(static_cast<SizeType>(mSubBlockExecResid1 % vlw),
                              static_cast<SizeType>(kBlockExecFull / NUM_2), false);
    CodeGenWrapper<FastReorderKernelCodeGen<float, float>> reorderRM2FKNA; // resid N 2 full K NoAlpha
    reorderRM2FKNA.Initialize(static_cast<SizeType>(mSubBlockExecResid2 % vlw),
                              static_cast<SizeType>(kBlockExecFull / NUM_2), false);
    CodeGenWrapper<FastReorderKernelCodeGen<float, float>> reorderFBRKNA; // full Block resid K NoAlpha
    reorderFBRKNA.Initialize(static_cast<SizeType>(vlw),
                             static_cast<SizeType>(kBlockExecResid / NUM_2), false);
    CodeGenWrapper<FastReorderKernelCodeGen<float, float>> reorderRN1RKNA; // resid N 1 resid K NoAlpha
    reorderRN1RKNA.Initialize(static_cast<SizeType>(nSubBlockExecResid1 % vlw),
                              static_cast<SizeType>(kBlockExecResid / NUM_2), false);
    CodeGenWrapper<FastReorderKernelCodeGen<float, float>> reorderRN2RKNA; // resid N 2 resid K NoAlpha
    reorderRN2RKNA.Initialize(static_cast<SizeType>(nSubBlockExecResid2 % vlw),
                              static_cast<SizeType>(kBlockExecResid / NUM_2), false);
    CodeGenWrapper<FastReorderKernelCodeGen<float, float>> reorderRM1RKNA; // resid M 1 resid K NoAlpha
    reorderRM1RKNA.Initialize(static_cast<SizeType>(mSubBlockExecResid1 % vlw),
                              static_cast<SizeType>(kBlockExecResid / NUM_2), false);
    CodeGenWrapper<FastReorderKernelCodeGen<float, float>> reorderRM2RKNA; // resid N 2 resid K NoAlpha
    reorderRM2RKNA.Initialize(static_cast<SizeType>(mSubBlockExecResid2 % vlw),
                              static_cast<SizeType>(kBlockExecResid / NUM_2), false);

    reorderPackingCodegenKernels = {
        std::move(reorderFBFKNA), std::move(reorderRN1FKNA), std::move(reorderRN2FKNA),
        std::move(reorderRM1FKNA), std::move(reorderRM2FKNA),
        std::move(reorderFBRKNA), std::move(reorderRN1RKNA), std::move(reorderRN2RKNA),
        std::move(reorderRM1RKNA), std::move(reorderRM2RKNA)
    };
}

template <typename SrcWeiType, typename DstBiaType>
IntType MatMulImplGenerator<MCBFP16Gen<SrcWeiType, DstBiaType>>::DefaultImpl::GetThreadBufSize(
    IntType m, IntType n, IntType k, MatrixKind mKind) const noexcept
{
    IntType vlh = svcnth();
    IntType mRoundedUp = Math::CeilDiv(m, MCBFP16Gen<SrcWeiType, DstBiaType>::KERNEL_M_SIZE * vlh) *
        MCBFP16Gen<SrcWeiType, DstBiaType>::KERNEL_M_SIZE * vlh;
    IntType nRoundedUp = Math::CeilDiv(n, MCBFP16Gen<SrcWeiType, DstBiaType>::KERNEL_N_SIZE * vlh) *
        MCBFP16Gen<SrcWeiType, DstBiaType>::KERNEL_N_SIZE * vlh;
    IntType kRoundedUp = Math::CeilDiv(k, MCBFP16Gen<SrcWeiType, DstBiaType>::K_LOOP_UNROLL * NUM_2) *
        MCBFP16Gen<SrcWeiType, DstBiaType>::K_LOOP_UNROLL * NUM_2;
    if (mKind == MatrixKind::SRC) {
        return static_cast<IntType>(std::min(GetKBlockSize(), kRoundedUp) *
            std::min(GetMBlockSize(), mRoundedUp) * sizeof(SrcWeiType));
    } else {
        return static_cast<IntType>(std::min(GetKBlockSize(), kRoundedUp) *
            std::min(GetNBlockSize(), nRoundedUp) * sizeof(SrcWeiType));
    }
}

template <typename SrcWeiType, typename DstBiaType>
void MatMulImplGenerator<MCBFP16Gen<SrcWeiType, DstBiaType>>::DefaultImpl::ReserveImplSpecificParams() {}

using vape = std::vector<std::pair<Shape, Shape>>;

template <typename SrcWeiType, typename DstBiaType>
vape MatMulImplGenerator<MCBFP16Gen<SrcWeiType, DstBiaType>>::DefaultImpl::SplitWork()
{
    return CalculateWorkOnThread<DefaultImpl>(taskExec.dstInfo.GetDims(), nThreadsExec, dstLayout2D == Layout::BA);
}

template <typename SrcWeiType, typename DstBiaType>
void MatMulImplGenerator<MCBFP16Gen<SrcWeiType, DstBiaType>>::DefaultImpl::PackSimilarLt(IntType mBlock,
    IntType kBlock, const void *src, IntType offset, void *buffer, IntType bufferOffset)
{
    std::int64_t vlh = svcnth();
    svbool_t ptrue = svptrue_b16();
    std::int64_t kRoundedUp = ((kBlock + NUM_1) / NUM_2) * NUM_2;
    IntType k = 0;
    // NUM_2 is due to peculiarities of MOPA instruction for INT8 data type
    for (; k < kBlock - NUM_2; k += NUM_2) {
        for (std::int64_t m = 0; m < mBlock; m += vlh) {
            svbool_t p0 = svwhilelt_b16(m + 0 * vlh, mBlock);
            auto src00 = svld1(p0, static_cast<const SrcWeiType*>(src) + 0 * vlh + m + offset * (k + 0));
            auto src10 = svld1(p0, static_cast<const SrcWeiType*>(src) + 0 * vlh + m + offset * (k + 1));
            auto dst0 = svzip1(src00, src10);
            auto dst1 = svzip2(src00, src10);
            svst1(ptrue, static_cast<SrcWeiType*>(buffer)
                + 0 * vlh + (m + 0 * bufferOffset) * kRoundedUp + k * bufferOffset, dst0);
            svst1(ptrue, static_cast<SrcWeiType*>(buffer)
                + 1 * vlh + (m + 0 * bufferOffset) * kRoundedUp + k * bufferOffset, dst1);
        }
    }
    IntType kResid = kBlock % NUM_2;
    if (kResid == 0) {
        kResid = NUM_2;
    }
    for (IntType m = 0; m < mBlock; m += vlh) {
        svbool_t p0 = svwhilelt_b16(m + 0 * vlh, mBlock);
        typename Vector::Vec<SrcWeiType>::Type src00;
        typename Vector::Vec<SrcWeiType>::Type src10;
        src00 = svld1(p0, static_cast<const SrcWeiType*>(src) + 0 * vlh + m + offset * (k + 0));
        if (kResid > NUM_1) {
            src10 = svld1(p0, static_cast<const SrcWeiType*>(src) + 0 * vlh + m + offset * (k + 1));
        } else {
            Vector::DupFloat<SrcWeiType>::Do(NUM_0, src10);
        }
        auto dst0 = svzip1(src00, src10);
        auto dst1 = svzip2(src00, src10);
        svst1(ptrue, static_cast<SrcWeiType*>(buffer)
            + 0 * vlh + (m + 0 * bufferOffset) * kRoundedUp + k * bufferOffset, dst0);
        svst1(ptrue, static_cast<SrcWeiType*>(buffer)
            + 1 * vlh + (m + 0 * bufferOffset) * kRoundedUp + k * bufferOffset, dst1);
    }
}

// used for packing both Src and Wei matrices
template <typename SrcWeiType, typename DstBiaType>
void MatMulImplGenerator<MCBFP16Gen<SrcWeiType, DstBiaType>>::DefaultImpl::PackDiffLt(
    IntType threadID, IntType mBlock, IntType kBlock, const void *src, IntType offset, void *buffer,
    IntType bufferOffset, PackingType packingType, IntType kBlockSize) const
{
    const IntType vlw = svcntw();
    // BF16 / FP16 specific ==> NUM_2 = sizeof(float) / sizeof(bfloat16_t)
    IntType kBlock32 = kBlock / NUM_2;
    IntType kBlock32RoundedUp = Math::CeilDiv(kBlock, NUM_2);
    IntType kResid = kBlock % NUM_2;
    svbool_t pgResid = svwhilelt_b16_s64(0, kResid);
    svbool_t pg2 = svwhilelt_b16_s64(0, NUM_2);

    const IntType vl = svcntw();
    IntType idxK = (kBlock == kBlockSize) ? 0 : 1;
    bool isM = (packingType != PackingType::PACK_SRC) ? false : true;

    for (IntType mIdx = mBlock; mIdx > 0; mIdx -= bufferOffset) {
        if (mIdx <= vlw) {
            if (kBlock32 > 0) {
                IntType idxM = ((mIdx < vl) ? 1 : 0);
                IntType mIdxKernel = (idxM) * (1 + isM * 2 +
                    (isM ? perThreadParams[threadID].isResidByM : perThreadParams[threadID].isResidByN));
                IntType packBColIdx = idxK * NUM_5 + mIdxKernel;
                auto kernel = reorderPackingCodegenKernels[packBColIdx]
                                .template GetCode<void(*)(const float*, float*, IntType, IntType, float)>();
                kernel(Service::AddOffset<const float *>(src,
                    (mBlock - mIdx) * sizeof(__fp16) * offset),
                    static_cast<float *>(buffer) + kBlock32RoundedUp * (mBlock - mIdx),
                    offset * sizeof(__fp16), bufferOffset * sizeof(float), 1.0f);
            }
            if (kResid) {
                for (IntType mSubIdx = mIdx; mSubIdx > 0; --mSubIdx) {
                    auto srcReg = svld1(pgResid, static_cast<const SrcWeiType *>(src) +
                                        (mBlock - mSubIdx) * offset + kBlock32 * NUM_2);
                    svst1(pg2, static_cast<SrcWeiType *>(buffer) +
                          kBlock32 * bufferOffset * NUM_2 +
                          (kBlock32RoundedUp * (mBlock - mIdx) + (mIdx - mSubIdx)) * NUM_2, srcReg);
                }
            }
        } else {
            IntType mIdxReorder1 = vl;
            IntType mIdxReorder2 = std::min(vl, mIdx - vl);
            IntType idxM1 = ((mIdxReorder1 < vl) ? 1 : 0);
            IntType idxM2 = ((mIdxReorder2 < vl) ? 1 : 0);
            if (kBlock32) {
                IntType mIdxKernel1 = (idxM1) * (1 + isM * 2 +
                    (isM ? perThreadParams[threadID].isResidByM : perThreadParams[threadID].isResidByN));
                IntType mIdxKernel2 = (idxM2) * (1 + isM * 2 +
                    (isM ? perThreadParams[threadID].isResidByM : perThreadParams[threadID].isResidByN));
                IntType packBColIdx1 = idxK * NUM_5 + mIdxKernel1;
                IntType packBColIdx2 = idxK * NUM_5 + mIdxKernel2;
                auto kernel = reorderPackingCodegenKernels[packBColIdx1]
                                .template GetCode<void(*)(const float*, float*, IntType, IntType, float)>();
                kernel(Service::AddOffset<const float *>(src,
                    (mBlock - mIdx) * sizeof(__fp16) * offset),
                    static_cast<float *>(buffer) + kBlock32RoundedUp * (mBlock - mIdx),
                    offset * sizeof(__fp16), bufferOffset * sizeof(float), 1.0f);

                kernel = reorderPackingCodegenKernels[packBColIdx2]
                            .template GetCode<void(*)(const float*, float*, IntType, IntType, float)>();
                kernel(Service::AddOffset<const float *>(src,
                    (mBlock - mIdx + vlw) * sizeof(__fp16) * offset),
                    static_cast<float *>(buffer) + kBlock32RoundedUp * (mBlock - mIdx) + vlw,
                    offset * sizeof(__fp16), bufferOffset * sizeof(float), 1.0f);
            }
            if (kResid) {
                for (IntType mSubIdx = mIdx; mSubIdx > std::max(IntType{0}, mIdx - NUM_2 * vlw); --mSubIdx) {
                    auto srcReg = svld1(pgResid, static_cast<const SrcWeiType *>(src) +
                                        (mBlock - mSubIdx) * offset + kBlock32 * NUM_2);
                    svst1(pg2, static_cast<SrcWeiType *>(buffer) +
                          kBlock32 * bufferOffset * NUM_2 +
                          (kBlock32RoundedUp * (mBlock - mIdx) + (mIdx - mSubIdx)) * NUM_2, srcReg);
                }
            }
        }
    }
}

template <typename SrcWeiType, typename DstBiaType>
void MatMulImplGenerator<MCBFP16Gen<SrcWeiType, DstBiaType>>::DefaultImpl::Pack(
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
        const SrcWeiType *srcExec = static_cast<const SrcWeiType *>(originalMatrix);
        if (layout == Layout::COL_MAJOR) {
            srcExec += dstDimOffset + accumDimOffset * offset2D;
            PackSimilarLt(dstDim, accumDim, srcExec,
                offset2D, static_cast<SrcWeiType *>(buffer) + bufferOffset,
                MCBFP16Gen<SrcWeiType, DstBiaType>::KERNEL_M_SIZE * vlw);
        } else {
            srcExec += dstDimOffset * offset2D + accumDimOffset;
            PackDiffLt(threadID, dstDim, accumDim, srcExec,
                offset2D, static_cast<SrcWeiType *>(buffer) + bufferOffset,
                MCBFP16Gen<SrcWeiType, DstBiaType>::KERNEL_M_SIZE * vlw, PackingType::PACK_SRC,
                GetKBlockSize());
        }
    } else {
        const SrcWeiType *weiExec = static_cast<const SrcWeiType *>(originalMatrix);
        if (layout == Layout::ROW_MAJOR) {
            weiExec += dstDimOffset + accumDimOffset * offset2D;
            PackSimilarLt(dstDim, accumDim, weiExec,
                offset2D, static_cast<SrcWeiType *>(buffer) + bufferOffset,
                MCBFP16Gen<SrcWeiType, DstBiaType>::KERNEL_N_SIZE * vlw);
        } else {
            weiExec += dstDimOffset * offset2D + accumDimOffset;
            PackDiffLt(threadID, dstDim, accumDim, weiExec,
                offset2D, static_cast<SrcWeiType *>(buffer) + bufferOffset,
                MCBFP16Gen<SrcWeiType, DstBiaType>::KERNEL_N_SIZE * vlw, PackingType::PACK_WEI,
                GetKBlockSize());
        }
    }
}

template <typename SrcWeiType, typename DstBiaType>
void MatMulImplGenerator<MCBFP16Gen<SrcWeiType, DstBiaType>>::DefaultImpl::RunSequential2D(
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
    SrcWeiType *srcBuffer = static_cast<SrcWeiType *>(
        srcPackStrategy->GetPtr(tIDbatch * threadsM  + tIDm, tIDn, threadsN));
    SrcWeiType *weiBuffer = static_cast<SrcWeiType *>(
        weiPackStrategy->GetPtr(tIDbatch * threadsN  + tIDn, tIDm, threadsM));

    for (IntType mBlock = 0; mBlock < mFullBlockSizePerThread; mBlock += GetMBlockSize()) {
        auto mBlockExec = std::min(GetMBlockSize(), m - mBlock);
        IntType mIdx = (mBlockExec == GetMBlockSize()) ? 0 : 1;
        IntType mIdxKernel = mIdx * (perThreadParams[threadID].isResidByM + 1);
        for (IntType kBlock = 0; kBlock < k; kBlock += GetKBlockSize()) {
            auto kBlockExec = std::min(GetKBlockSize(), k - kBlock);
            IntType kIdxKernel = (kBlockExec == GetKBlockSize()) ? 0 : 1;
            bool lastKIter = (k - kBlock) <= GetKBlockSize();
            MatMulDstFlags biaFlagsExec = SetBiaFlagsExec(biaFlags, beta, kBlock, lastKIter);
            IntType kBlockExecRoundedUp = Math::CeilDiv(kBlockExec, NUM_2) * NUM_2; // FP16 MOPA peculiarities
            if (mBlock < m) {
                srcPackStrategy->Pack(*this, src, srcBuffer,
                    {mBlock, mBlockExec, GetKernelMSize()}, {kBlock, kBlockExec, NUM_2},
                    threadsN, tIDn, threadID, alpha);
            }
            srcPackStrategy->Barrier(tIDbatch * threadsM + tIDm, tIDn, threadsN);
            for (IntType nBlock = 0; nBlock < nFullBlockSizePerThread; nBlock += GetNBlockSize()) {
                if ((nBlock < n) && (mBlock < m)) {
                    auto nBlockExec = std::min(GetNBlockSize(), n - nBlock);
                    IntType nIdx = (nBlockExec == GetNBlockSize()) ? 0 : 1;
                    IntType nIdxKernel = nIdx * (perThreadParams[threadID].isResidByN + 1);
                    weiPackStrategy->Pack(*this, wei, weiBuffer,
                        {nBlock, nBlockExec, GetKernelNSize()}, {kBlock, kBlockExec, NUM_2},
                        threadsM, tIDm, threadID, alpha);
                    weiPackStrategy->Barrier(tIDbatch * threadsN + tIDn, tIDm, threadsM);
                    DstBiaType *dstExec = static_cast<DstBiaType *>(dst) + nBlock +
                        mBlock * perThreadParams[threadID].dstOffset2D;
                    const DstBiaType *biaExec = static_cast<const DstBiaType *>(bia) +
                        nBlock * biaNOffset + mBlock * perThreadParams[threadID].biaOffset2D;
                    auto kernel = matmulCodegenKernels[kIdxKernel * NUM_9 +
                        nIdxKernel * NUM_3 + mIdxKernel];
                    kernel(srcBuffer, weiBuffer, dstExec, biaExec, biaFlagsExec, alpha, beta,
                        mBlockExec, nBlockExec, kBlockExec, kBlockExecRoundedUp, kBlockExecRoundedUp,
                        perThreadParams[threadID].dstOffset2D * sizeof(DstBiaType),
                        perThreadParams[threadID].biaOffset2D * sizeof(DstBiaType));
                    weiPackStrategy->Barrier(tIDbatch * threadsN + tIDn, tIDm, threadsM);
                }
            }
            srcPackStrategy->Barrier(tIDbatch * threadsM + tIDm, tIDn, threadsN);
        }
    }
}

template <typename SrcWeiType, typename DstBiaType>
void MatMulImplGenerator<MCBFP16Gen<SrcWeiType, DstBiaType>>::DefaultImpl::UpdateExecParams() noexcept
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
        MCBFP16Gen<SrcWeiType, DstBiaType>::GetSrcType(),
        MCBFP16Gen<SrcWeiType, DstBiaType>::GetWeiType(),
        MCBFP16Gen<SrcWeiType, DstBiaType>::GetDstType(),
        (implTask.biaInfo.GetTotalTensorSize() != 0) ? MCBFP16Gen<SrcWeiType, DstBiaType>::GetBiaType() :
            static_cast<Element::TypeT>(implTask.biaInfo.GetType()));
}

template <typename SrcWeiType, typename DstBiaType>
MatMulDstFlags MatMulImplGenerator<MCBFP16Gen<SrcWeiType, DstBiaType>>::DefaultImpl::SetBiaFlagsExec(
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

template
struct MatMulImplGenerator<MatMulKernelDefaultCodeGen<ARCH::MC,
    __fp16, __fp16, float, float, MatMulImplKinds::BFP16BFP16BFP1632BFP1632_MC>>;
template
struct MatMulImplGenerator<MatMulKernelDefaultCodeGen<ARCH::MC,
    __bf16, __bf16, float, float, MatMulImplKinds::BFP16BFP16BFP1632BFP1632_MC>>;
template
struct MatMulImplGenerator<MatMulKernelDefaultCodeGen<ARCH::MC,
    __fp16, __fp16, __fp16, __fp16, MatMulImplKinds::BFP16BFP16BFP1632BFP1632_MC>>;
template
struct MatMulImplGenerator<MatMulKernelDefaultCodeGen<ARCH::MC,
    __bf16, __bf16, __bf16, __bf16, MatMulImplKinds::BFP16BFP16BFP1632BFP1632_MC>>;

} // Detail
} // KuDNN
