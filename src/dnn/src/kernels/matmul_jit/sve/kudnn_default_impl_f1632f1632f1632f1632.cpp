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
#include "kernels/matmul_jit/kudnn_gemm_jit.hpp"
#include "kernels/matmul_jit/kudnn_gemm_helpers.hpp"
#include "kernels/matmul_jit/kudnn_gemm_threading_helpers.hpp"
#include "kernels/reorder/sve/bf16any32/kudnn_fast_jit_impl_bfp16any32_bfp16any32.hpp"
#include "kudnn_wrapper_helpers.hpp"
#include "kernels/matmul_jit/sve/kudnn_default_impl_f1632f1632f1632f1632.hpp"

namespace KuDNN {
namespace Detail {

template <typename FP1632Type>
using VCF1632GenBase = MatMulKernelCodeGenBase<ARCH::VC, FP1632Type, FP1632Type, FP1632Type, FP1632Type>;

template <typename FP1632Type>
VCF1632Gen<FP1632Type>::MatMulKernelDefaultCodeGen(
    IntType m, IntType n, IntType k,
    const Attributes &attributesArg) : VCF1632GenBase<FP1632Type>(attributesArg)
{
    if ((m > 0) && (n > 0) && (k > 0)) {
        // this part is impl specific and indicates which registers can be used for post-ops
        if (this->attributes.GetPostOps().Len() > 0) {
            // set reserved dst registers
            for (std::uint32_t mIdx = 0; mIdx < KERNEL_M_SIZE; ++mIdx) {
                for (std::uint32_t nIdx = 0; nIdx < KERNEL_N_SIZE; ++nIdx) {
                    TakeVectorRegister(GetDstZRegIdx(mIdx, nIdx));
                }
            }
            // set reserved predicate registers
            for (std::uint32_t idx = 0; idx < NUM_6; ++idx) {
                TakePredicateRegister(idx);
            }
            // set reserved scalar registers
            for (std::uint32_t idx = 0; idx < NUM_28; ++idx) {
                TakeScalarRegister(idx);
            }
        }
        Ldp(srcOffset, weiOffset, PostPtr(sp, NUM_16));
        Ldp(dstOffset, biaOffset, PostPtr(sp, NUM_0));
        Sub(sp, sp, NUM_16);
        Preamble();

        if constexpr (std::is_same_v<float, FP1632Type>) {
            Sub(sp, sp, NUM_16);
            Str(alpha, ARMCG::AdrUimm(Ptr(sp)));
            Sub(sp, sp, NUM_16);
            Str(beta, ARMCG::AdrUimm(Ptr(sp))); // beta will be overwritten while accumulation
        } else {
            Sub(sp, sp, NUM_16);
            Fcvt(halpha, alpha);
            Str(halpha, ARMCG::AdrUimm(Ptr(sp)));
            Sub(sp, sp, NUM_16);
            Fcvt(hbeta, beta);
            Str(hbeta, ARMCG::AdrUimm(Ptr(sp))); // beta will be overwritten while accumulation
        }

        Ptrue(KUDNN_JIT_Generator::GetPReg<FP1632Type>(5), ARMCG::Pattern::ALL); // p5 is true for this impl

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

template <typename FP1632Type>
void VCF1632Gen<FP1632Type>::StoreToDst(std::uint32_t mLim, std::uint32_t nLim)
{
    // can be reworked to reuse some code (at least nested loops)
    ARMCG::Label fillZeros;
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
    ARMCG::Label skipPO;

    // z0, z1 registers are chosen for tmp values
    if constexpr (std::is_same_v<float, FP1632Type>) {
        KUDNN_JIT_Generator::Ld1r(ARMCG::ZRegS(NUM_1), p5, Ptr(sp, static_cast<std::int32_t>(NUM_16)));
        Mov(tmpW, ARMCG::VReg4S(NUM_1)[0]);
        Mov(ARMCG::WReg(GetTmpXRegIdx(NUM_0)), 0x3f800000); // 0x3f800000 is a bit representation of 1.0f
        Cmp(tmpW, ARMCG::WReg(GetTmpXRegIdx(NUM_0)));
    } else {
        KUDNN_JIT_Generator::Ld1rh(ARMCG::ZRegH(NUM_1), p5, ARMCG::AdrScImm(sp, static_cast<std::int32_t>(NUM_16)));
        KUDNN_JIT_Generator::Fcvt(ARMCG::ZRegS(NUM_0), p5, ARMCG::ZRegH(NUM_1));
        Mov(tmpW, ARMCG::VReg4S(NUM_0)[0]);
        Mov(ARMCG::WReg(GetTmpXRegIdx(NUM_0)), 0x3f800000); // 0x3f800000 is a bit representation of 1.0f
        Cmp(tmpW, ARMCG::WReg(GetTmpXRegIdx(NUM_0)));
    }
    Beq(loadBias1);
    for (std::uint32_t mIdx = 0; mIdx < mLim; ++mIdx) {
        for (std::uint32_t nIdx = 0; nIdx < nLim; ++nIdx) {
            Fmul(
                KUDNN_JIT_Generator::GetZReg<FP1632Type>(GetDstZRegIdx(mIdx, nIdx)),
                KUDNN_JIT_Generator::GetZReg<FP1632Type>(GetDstZRegIdx(mIdx, nIdx)),
                KUDNN_JIT_Generator::GetZReg<FP1632Type>(NUM_1)  // z1 contains alpha value
            );
        }
    }
    Tbz(flags, 0, loadBias1); // 0-bit correspond to FILL_C_WITH_ZEROZ
    L(loadBias1);
    Tbz(flags, 3, loadBiasN); // 3-bit correspond to LOAD_BIAS_1
    KUDNN_JIT_Generator::Ld1r(KUDNN_JIT_Generator::GetZReg<FP1632Type>(NUM_7), p5, Ptr(biaPtr));
    for (std::uint32_t mIdx = 0; mIdx < mLim; ++mIdx) {
        for (std::uint32_t nIdx = 0; nIdx < nLim; ++nIdx) {
            Fadd(KUDNN_JIT_Generator::GetZReg<FP1632Type>(GetDstZRegIdx(mIdx, nIdx)),
                KUDNN_JIT_Generator::GetZReg<FP1632Type>(GetDstZRegIdx(mIdx, nIdx)),
                KUDNN_JIT_Generator::GetZReg<FP1632Type>(NUM_7)); // z7 contains bias value
        }
    }
    L(loadBiasN);
    Tbz(flags, 5, loadBiasM); // 5-bit correspond to LOAD_BIAS_N
    Add(tmp, biaPtr, tmpNMinusNLoopCtr, ARMCG::ShMod::LSL, Math::Log2(sizeof(typename TypeHelper::BiaType)));
    for (std::uint32_t nIdx = 0; nIdx < nLim; ++nIdx) {
        // z4-z10 registers are chosen for tmp values, could be any but dst Z registers
        KUDNN_JIT_Generator::Ld1(KUDNN_JIT_Generator::GetZReg<FP1632Type>(NUM_4 + nIdx),
            ARMCG::PReg(GetLoadStorePRegIdx(nIdx)), Ptr(tmp, nIdx, ARMCG::ExtMod::MUL_VL));
    }
    for (std::uint32_t mIdx = 0; mIdx < mLim; ++mIdx) {
        for (std::uint32_t nIdx = 0; nIdx < nLim; ++nIdx) {
            Fadd(KUDNN_JIT_Generator::GetZReg<FP1632Type>(GetDstZRegIdx(mIdx, nIdx)),
                KUDNN_JIT_Generator::GetZReg<FP1632Type>(GetDstZRegIdx(mIdx, nIdx)),
                KUDNN_JIT_Generator::GetZReg<FP1632Type>(NUM_4 + nIdx)); // z4-z10 registers are chosen for tmp values
        }
    }
    L(loadBiasM);
    Tbz(flags, 4, loadBiasMN); // 4-bit correspond to LOAD_BIAS_M
    Add(tmp, biaPtr, tmpMMinusMLoopCtr, ARMCG::ShMod::LSL, Math::Log2(sizeof(typename TypeHelper::BiaType)));
    for (std::uint32_t mIdx = 0; mIdx < mLim; ++mIdx) {
        KUDNN_JIT_Generator::Ld1r(KUDNN_JIT_Generator::GetZReg<FP1632Type>(GetSrcZRegIdx(mIdx)), p5,
            Ptr(tmp, static_cast<std::uint32_t>(sizeof(typename TypeHelper::BiaType)) * mIdx));
    }
    for (std::uint32_t mIdx = 0; mIdx < mLim; ++mIdx) {
        for (std::uint32_t nIdx = 0; nIdx < nLim; ++nIdx) {
            Fadd(KUDNN_JIT_Generator::GetZReg<FP1632Type>(GetDstZRegIdx(mIdx, nIdx)),
                KUDNN_JIT_Generator::GetZReg<FP1632Type>(GetDstZRegIdx(mIdx, nIdx)),
                KUDNN_JIT_Generator::GetZReg<FP1632Type>(GetSrcZRegIdx(mIdx))); // z7 contains bias value
        }
    }
    L(loadBiasMN);
    Tbz(flags, 6, loadC); // 6-bit correspond to LOAD_BIAS_MN
    Mul(tmp, tmpMMinusMLoopCtr, biaOffset);
    Add(ARMCG::XReg(GetSrcXRegIdx(0)), biaPtr, tmp,
        ARMCG::ShMod::LSL, Math::Log2(sizeof(typename TypeHelper::BiaType)));
    Add(ARMCG::XReg(GetSrcXRegIdx(0)),
        ARMCG::XReg(GetSrcXRegIdx(0)),
        tmpNMinusNLoopCtr, ARMCG::ShMod::LSL, Math::Log2(sizeof(typename TypeHelper::BiaType)));
    for (std::uint32_t mIdx = 1; mIdx < mLim; ++mIdx) {
        Add(ARMCG::XReg(GetSrcXRegIdx(mIdx)),
            ARMCG::XReg(GetSrcXRegIdx(mIdx - 1)),
            biaOffset, ARMCG::ShMod::LSL, Math::Log2(sizeof(typename TypeHelper::BiaType)));
    }
    for (std::uint32_t mIdx = 0; mIdx < mLim; ++mIdx) {
        for (std::uint32_t nIdx = 0; nIdx < nLim; ++nIdx) {
            // z4-z10 registers are chosen for tmp values, could be any except dst registers
            KUDNN_JIT_Generator::Ld1(KUDNN_JIT_Generator::GetZReg<FP1632Type>(4 + nIdx),
                KUDNN_JIT_Generator::GetPReg<FP1632Type>(GetLoadStorePRegIdx(nIdx)),
                Ptr(ARMCG::XReg(GetSrcXRegIdx(mIdx)), nIdx,
                    ARMCG::ExtMod::MUL_VL));
            Fadd(KUDNN_JIT_Generator::GetZReg<FP1632Type>(GetDstZRegIdx(mIdx, nIdx)),
                KUDNN_JIT_Generator::GetZReg<FP1632Type>(GetDstZRegIdx(mIdx, nIdx)),
                KUDNN_JIT_Generator::GetZReg<FP1632Type>(4 + nIdx)); // z4-z10 registers are chosen for tmp values
        }
    }
    L(loadC);
    Tbz(flags, 1, mullC); // 1-bit correspond to LOAD_C
    Mul(tmp, tmpMMinusMLoopCtr, dstOffset);
    Add(ARMCG::XReg(GetSrcXRegIdx(0)), dstPtr, tmp,
        ARMCG::ShMod::LSL, Math::Log2(sizeof(typename TypeHelper::DstType)));
    Add(ARMCG::XReg(GetSrcXRegIdx(0)), ARMCG::XReg(GetSrcXRegIdx(0)),
        tmpNMinusNLoopCtr, ARMCG::ShMod::LSL, Math::Log2(sizeof(typename TypeHelper::DstType)));
    for (std::uint32_t mIdx = 1; mIdx < mLim; ++mIdx) {
        Add(ARMCG::XReg(GetSrcXRegIdx(mIdx)),
            ARMCG::XReg(GetSrcXRegIdx(mIdx - 1)),
            dstOffset, ARMCG::ShMod::LSL, Math::Log2(sizeof(typename TypeHelper::DstType)));
    }
    for (std::uint32_t mIdx = 0; mIdx < mLim; ++mIdx) {
        for (std::uint32_t nIdx = 0; nIdx < nLim; ++nIdx) {
            KUDNN_JIT_Generator::Ld1(KUDNN_JIT_Generator::GetZReg<FP1632Type>(GetSrcZRegIdx(mIdx)),
                ARMCG::PReg(GetLoadStorePRegIdx(nIdx)),
                Ptr(ARMCG::XReg(GetSrcXRegIdx(mIdx)), nIdx,
                ARMCG::ExtMod::MUL_VL));
            Fadd(KUDNN_JIT_Generator::GetZReg<FP1632Type>(GetDstZRegIdx(mIdx, nIdx)),
                KUDNN_JIT_Generator::GetZReg<FP1632Type>(GetDstZRegIdx(mIdx, nIdx)),
                KUDNN_JIT_Generator::GetZReg<FP1632Type>(GetSrcZRegIdx(mIdx)));
        }
    }
    L(mullC);
    Tbz(flags, 2, addBias1); // 2-bit correspond to MULL_C
    KUDNN_JIT_Generator::Ld1r(KUDNN_JIT_Generator::GetZReg<FP1632Type>(NUM_0), p5, Ptr(sp));
    Mul(tmp, tmpMMinusMLoopCtr, dstOffset);
    Add(ARMCG::XReg(GetSrcXRegIdx(0)), dstPtr, tmp,
        ARMCG::ShMod::LSL, Math::Log2(sizeof(typename TypeHelper::DstType)));
    Add(ARMCG::XReg(GetSrcXRegIdx(0)), ARMCG::XReg(GetSrcXRegIdx(0)),
        tmpNMinusNLoopCtr, ARMCG::ShMod::LSL, Math::Log2(sizeof(typename TypeHelper::DstType)));
    for (std::uint32_t mIdx = 1; mIdx < mLim; ++mIdx) {
        Add(ARMCG::XReg(GetSrcXRegIdx(mIdx)),
            ARMCG::XReg(GetSrcXRegIdx(mIdx - 1)),
            dstOffset, ARMCG::ShMod::LSL, Math::Log2(sizeof(typename TypeHelper::DstType)));
    }
    for (std::uint32_t mIdx = 0; mIdx < mLim; ++mIdx) {
        for (std::uint32_t nIdx = 0; nIdx < nLim; ++nIdx) {
            KUDNN_JIT_Generator::Ld1(KUDNN_JIT_Generator::GetZReg<FP1632Type>(GetSrcZRegIdx(mIdx)),
                ARMCG::PReg(GetLoadStorePRegIdx(nIdx)),
                Ptr(ARMCG::XReg(GetSrcXRegIdx(mIdx)), nIdx,
                ARMCG::ExtMod::MUL_VL));
            Fmul(KUDNN_JIT_Generator::GetZReg<FP1632Type>(GetSrcZRegIdx(mIdx)),
                KUDNN_JIT_Generator::GetZReg<FP1632Type>(GetSrcZRegIdx(mIdx)),
                KUDNN_JIT_Generator::GetZReg<FP1632Type>(NUM_0)); // z0 contains beta value
            Fadd(KUDNN_JIT_Generator::GetZReg<FP1632Type>(GetDstZRegIdx(mIdx, nIdx)),
                KUDNN_JIT_Generator::GetZReg<FP1632Type>(GetDstZRegIdx(mIdx, nIdx)),
                KUDNN_JIT_Generator::GetZReg<FP1632Type>(GetSrcZRegIdx(mIdx)));
        }
    }
    L(addBias1);
    Tbz(flags, 7, addBiasN); // 7-bit correspond to Add_BIAS_1
    KUDNN_JIT_Generator::Ld1r(KUDNN_JIT_Generator::GetZReg<FP1632Type>(NUM_7), p5, Ptr(biaPtr));
    for (std::uint32_t mIdx = 0; mIdx < mLim; ++mIdx) {
        for (std::uint32_t nIdx = 0; nIdx < nLim; ++nIdx) {
            Fadd(KUDNN_JIT_Generator::GetZReg<FP1632Type>(GetDstZRegIdx(mIdx, nIdx)),
                KUDNN_JIT_Generator::GetZReg<FP1632Type>(GetDstZRegIdx(mIdx, nIdx)),
                KUDNN_JIT_Generator::GetZReg<FP1632Type>(NUM_7)); // z7 contains bias value
        }
    }
    L(addBiasN);
    Tbz(flags, 9, addBiasM); // 9-bit correspond to Add_BIAS_N
    Add(tmp, biaPtr, tmpNMinusNLoopCtr, ARMCG::ShMod::LSL, Math::Log2(sizeof(typename TypeHelper::BiaType)));
    for (std::uint32_t nIdx = 0; nIdx < nLim; ++nIdx) {
        // z4-z10 registers are chosen for tmp values, could be any but dst Z registers
        KUDNN_JIT_Generator::Ld1(KUDNN_JIT_Generator::GetZReg<FP1632Type>(NUM_4 + nIdx),
            ARMCG::PReg(GetLoadStorePRegIdx(nIdx)), Ptr(tmp, nIdx, ARMCG::ExtMod::MUL_VL));
    }
    for (std::uint32_t mIdx = 0; mIdx < mLim; ++mIdx) {
        for (std::uint32_t nIdx = 0; nIdx < nLim; ++nIdx) {
            Fadd(KUDNN_JIT_Generator::GetZReg<FP1632Type>(GetDstZRegIdx(mIdx, nIdx)),
                KUDNN_JIT_Generator::GetZReg<FP1632Type>(GetDstZRegIdx(mIdx, nIdx)),
                KUDNN_JIT_Generator::GetZReg<FP1632Type>(NUM_4 + nIdx)); // z4-z10 registers are chosen for tmp values
        }
    }
    L(addBiasM);
    Tbz(flags, 8, addBiasMN); // 8-bit correspond to ADD_BIAS_M
    Add(tmp, biaPtr, tmpMMinusMLoopCtr, ARMCG::ShMod::LSL, Math::Log2(sizeof(typename TypeHelper::BiaType)));
    for (std::uint32_t mIdx = 0; mIdx < mLim; ++mIdx) {
        KUDNN_JIT_Generator::Ld1r(KUDNN_JIT_Generator::GetZReg<FP1632Type>(GetSrcZRegIdx(mIdx)), p5,
            Ptr(tmp, static_cast<std::uint32_t>(sizeof(typename TypeHelper::BiaType)) * mIdx));
    }
    for (std::uint32_t mIdx = 0; mIdx < mLim; ++mIdx) {
        for (std::uint32_t nIdx = 0; nIdx < nLim; ++nIdx) {
            Fadd(KUDNN_JIT_Generator::GetZReg<FP1632Type>(GetDstZRegIdx(mIdx, nIdx)),
                KUDNN_JIT_Generator::GetZReg<FP1632Type>(GetDstZRegIdx(mIdx, nIdx)),
                KUDNN_JIT_Generator::GetZReg<FP1632Type>(GetSrcZRegIdx(mIdx)));
        }
    }
    L(addBiasMN);
    Tbz(flags, 10, initDone); // 10-bit correspond to ADD_BIAS_MN
    Mul(tmp, tmpMMinusMLoopCtr, biaOffset);
    Add(ARMCG::XReg(GetSrcXRegIdx(0)), biaPtr, tmp,
        ARMCG::ShMod::LSL, Math::Log2(sizeof(typename TypeHelper::BiaType)));
    Add(ARMCG::XReg(GetSrcXRegIdx(0)),
        ARMCG::XReg(GetSrcXRegIdx(0)),
        tmpNMinusNLoopCtr, ARMCG::ShMod::LSL, Math::Log2(sizeof(typename TypeHelper::BiaType)));
    for (std::uint32_t mIdx = 1; mIdx < mLim; ++mIdx) {
        Add(ARMCG::XReg(GetSrcXRegIdx(mIdx)),
            ARMCG::XReg(GetSrcXRegIdx(mIdx - 1)),
            biaOffset, ARMCG::ShMod::LSL, Math::Log2(sizeof(typename TypeHelper::BiaType)));
    }
    for (std::uint32_t mIdx = 0; mIdx < mLim; ++mIdx) {
        for (std::uint32_t nIdx = 0; nIdx < nLim; ++nIdx) {
            // z4-z10 registers are chosen for tmp values, could be any except dst registers
            KUDNN_JIT_Generator::Ld1(KUDNN_JIT_Generator::GetZReg<FP1632Type>(4 + nIdx),
                KUDNN_JIT_Generator::GetPReg<FP1632Type>(GetLoadStorePRegIdx(nIdx)),
                Ptr(ARMCG::XReg(GetSrcXRegIdx(mIdx)), nIdx,
                    ARMCG::ExtMod::MUL_VL));
            Fadd(KUDNN_JIT_Generator::GetZReg<FP1632Type>(GetDstZRegIdx(mIdx, nIdx)),
                KUDNN_JIT_Generator::GetZReg<FP1632Type>(GetDstZRegIdx(mIdx, nIdx)),
                KUDNN_JIT_Generator::GetZReg<FP1632Type>(4 + nIdx)); // z4-z10 registers are chosen for tmp values
        }
    }
    L(initDone);
    Tbz(flags, 11, skipPO); // 11-bit correspond to APPLY_POST_OPS
    std::vector<SizeType> dstRegsIDs(mLim * nLim);
    for (std::uint32_t mIdx = 0; mIdx < mLim; ++mIdx) {
        for (std::uint32_t nIdx = 0; nIdx < nLim; ++nIdx) {
            dstRegsIDs[mIdx * nLim + nIdx] = GetDstZRegIdx(mIdx, nIdx);
        }
    }
    this->ApplyPostOps(dstRegsIDs, Element::MatchType<FP1632Type>());
    L(skipPO);
    // Initialize dst pointers
    Mul(tmp, tmpMMinusMLoopCtr, dstOffset);
    // Row-major format of dst
    Add(ARMCG::XReg(GetDstXRegIdx(0)), dstPtr, tmp,
        ARMCG::ShMod::LSL, Math::Log2(sizeof(typename TypeHelper::DstType))); // Add stride for M
    Add(ARMCG::XReg(GetDstXRegIdx(0)),
        ARMCG::XReg(GetDstXRegIdx(0)),
        tmpNMinusNLoopCtr, ARMCG::ShMod::LSL, Math::Log2(sizeof(typename TypeHelper::DstType))); // Add stride for N
    for (std::uint32_t mIdx = 1; mIdx < mLim; ++mIdx) {
        Add(ARMCG::XReg(GetDstXRegIdx(mIdx)),
            ARMCG::XReg(GetDstXRegIdx(mIdx - 1)),
            dstOffset, ARMCG::ShMod::LSL, Math::Log2(sizeof(typename TypeHelper::DstType)));
    }
    // Store data to dst
    for (std::uint32_t mIdx = 0; mIdx < mLim; ++mIdx) {
        for (std::uint32_t nIdx = 0; nIdx < nLim; ++nIdx) {
            KUDNN_JIT_Generator::St1(KUDNN_JIT_Generator::GetZReg<FP1632Type>(GetDstZRegIdx(mIdx, nIdx)),
                KUDNN_JIT_Generator::GetPReg<FP1632Type>(GetLoadStorePRegIdx(nIdx)),
                Ptr(ARMCG::XReg(GetDstXRegIdx(mIdx)), nIdx,
                    ARMCG::ExtMod::MUL_VL));
        }
    }
}

template <typename FP1632Type>
void VCF1632Gen<FP1632Type>::GenerateMLoopBody(
    std::uint32_t mLim, std::uint32_t nLim, std::uint32_t k)
{
    std::int64_t vectorLength = Vector::Length<FP1632Type>();
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

template <typename FP1632Type>
void VCF1632Gen<FP1632Type>::GenerateNLoopBody(
    std::uint32_t mLim, std::uint32_t nLim, std::uint32_t k)
{
    std::uint32_t kResid = k % K_LOOP_UNROLL;
    if (kResid == 0) {
        kResid = K_LOOP_UNROLL;
    }
    Mov(tmp4VL, 0);
    for (std::uint32_t nIdx = 0; nIdx < nLim; ++nIdx) {
        Whilelt(KUDNN_JIT_Generator::GetPReg<FP1632Type>(GetLoadStorePRegIdx(nIdx)), tmp4VL, nLoopCtr);
        KUDNN_JIT_Generator::Inc<FP1632Type>(tmp4VL);
    }

    // Fill with zeros by default
    for (std::uint32_t mIdx = 0; mIdx < mLim; ++mIdx) {
        for (std::uint32_t nIdx = 0; nIdx < nLim; ++nIdx) {
            Mov(ARMCG::ZRegB(GetDstZRegIdx(mIdx, nIdx)), 0);
        }
    }

    // Initialize src pointers
    Mul(tmp, tmpMMinusMLoopCtr, srcOffset);
    Add(ARMCG::XReg(GetSrcXRegIdx(0)), srcPtr, tmp,
        ARMCG::ShMod::LSL, Math::Log2(sizeof(typename TypeHelper::SrcType)));
    for (std::uint32_t mIdx = 1; mIdx < mLim; ++mIdx) {
        Add(ARMCG::XReg(GetSrcXRegIdx(mIdx)),
            ARMCG::XReg(GetSrcXRegIdx(mIdx - 1)),
            srcOffset, ARMCG::ShMod::LSL, Math::Log2(sizeof(typename TypeHelper::SrcType)));
    }
    // Initialize wei pointers
    Mul(tmp, tmpNMinusNLoopCtr, kVal);
    Add(ARMCG::XReg(GetWeiXRegIdx(0, nLim)), weiPtr, tmp,
        ARMCG::ShMod::LSL, Math::Log2(sizeof(typename TypeHelper::WeiType)));
    for (std::uint32_t nIdx = 1; nIdx < nLim; ++nIdx) {
        Add(ARMCG::XReg(GetWeiXRegIdx(nIdx, nLim)),
            ARMCG::XReg(GetWeiXRegIdx(nIdx - 1, nLim)),
            weiOffset, ARMCG::ShMod::LSL, Math::Log2(sizeof(typename TypeHelper::WeiType)));
    }
    // Main k loop
    Mov(kLoopCtr, kVal);
    if (k > K_LOOP_UNROLL) {
        ARMCG::Label kLoopHead;
        L(kLoopHead);
        {
            Whilelt(KUDNN_JIT_Generator::GetPReg<FP1632Type>(0), xzr, kLoopCtr);
            // Preload wei registers z0, z1
            KUDNN_JIT_Generator::Ld1(KUDNN_JIT_Generator::GetZReg<FP1632Type>(GetWeiZRegIdx(0)), p5,
                Ptr(ARMCG::XReg(GetWeiXRegIdx(0, nLim))));
            if (nLim > 1) {
                KUDNN_JIT_Generator::Ld1(KUDNN_JIT_Generator::GetZReg<FP1632Type>(GetWeiZRegIdx(1)), p5,
                    Ptr(ARMCG::XReg(GetWeiXRegIdx(1, nLim))));
            }
            Sub(kLoopCtr, kLoopCtr, K_LOOP_UNROLL);
            Cmp(kLoopCtr, K_LOOP_UNROLL);
            // Preload src registers
            for (std::uint32_t mIdx = 0; mIdx < mLim; ++mIdx) {
                KUDNN_JIT_Generator::Ld1rq(KUDNN_JIT_Generator::GetZReg<FP1632Type>(GetSrcZRegIdx(mIdx)),
                    KUDNN_JIT_Generator::GetPReg<FP1632Type>(0), Ptr(ARMCG::XReg(GetSrcXRegIdx(mIdx))));
                Add(ARMCG::XReg(GetSrcXRegIdx(mIdx)),
                    ARMCG::XReg(GetSrcXRegIdx(mIdx)),
                    K_LOOP_UNROLL * sizeof(typename TypeHelper::SrcType));
            }
            GenerateMLAPart(mLim, nLim, K_LOOP_UNROLL, false);
        }
        B(ARMCG::Cond::GT, kLoopHead);
    }
    // Process k tail
    Whilelt(KUDNN_JIT_Generator::GetPReg<FP1632Type>(0), xzr, kLoopCtr);
    // preload wei registers z0, z1
    KUDNN_JIT_Generator::Ld1(KUDNN_JIT_Generator::GetZReg<FP1632Type>(GetWeiZRegIdx(0)), p5,
        Ptr(ARMCG::XReg(GetWeiXRegIdx(0, nLim))));
    if (nLim > 1) {
        KUDNN_JIT_Generator::Ld1(KUDNN_JIT_Generator::GetZReg<FP1632Type>(GetWeiZRegIdx(1)), p5,
            Ptr(ARMCG::XReg(GetWeiXRegIdx(1, nLim))));
    }
    // Preload src registers
    for (std::uint32_t mIdx = 0; mIdx < mLim; ++mIdx) {
        KUDNN_JIT_Generator::Ld1rq(KUDNN_JIT_Generator::GetZReg<FP1632Type>(GetSrcZRegIdx(mIdx)),
            KUDNN_JIT_Generator::GetPReg<FP1632Type>(0), Ptr(ARMCG::XReg(GetSrcXRegIdx(mIdx))));
    }
    GenerateMLAPart(mLim, nLim, kResid, true);

    // Store data to dst
    StoreToDst(mLim, nLim);
}

template <typename FP1632Type>
void VCF1632Gen<FP1632Type>::GenerateMLAPart(
    std::uint32_t mLim, std::uint32_t nLim, std::uint32_t kLim, bool isTail)
{
    for (std::uint32_t kIdx = 0; kIdx < kLim; ++kIdx) {
        // src (z2-z7) and weights (z0, z1) are preloaded
        for (std::uint32_t nIdx = 0; nIdx < nLim; ++nIdx) {
            for (std::uint32_t mIdx = 0; mIdx < mLim; ++mIdx) {
                Fmla(KUDNN_JIT_Generator::GetZReg<FP1632Type>(GetDstZRegIdx(mIdx, nIdx)),
                    KUDNN_JIT_Generator::GetZReg<FP1632Type>(GetWeiZRegIdx(nIdx)),
                    KUDNN_JIT_Generator::GetZReg<FP1632Type>(GetSrcZRegIdx(mIdx))[kIdx]);
            }
            std::uint32_t tmpN = (nLim == 1) ? 1 : 2;
            std::uint32_t kLoadOffset = kIdx + (nIdx + tmpN) / nLim; // +2 since 2 regs were already preloaded
            if (kLoadOffset != kLim) {
                // load wei reg right after usage
                KUDNN_JIT_Generator::Ld1(KUDNN_JIT_Generator::GetZReg<FP1632Type>(GetWeiZRegIdx(nIdx)), p5,
                    Ptr(ARMCG::XReg(GetWeiXRegIdx(nIdx + tmpN, nLim)),
                        kLoadOffset, ARMCG::ExtMod::MUL_VL));
            }
            if ((kIdx == (kLim - 1)) && !isTail) {
                Addvl(ARMCG::XReg(GetWeiXRegIdx(nIdx, nLim)),
                    ARMCG::XReg(GetWeiXRegIdx(nIdx, nLim)), K_LOOP_UNROLL);
            }
        }
    }
}

template <typename FP1632Type>
MatMulImplGenerator<VCF1632Gen<FP1632Type>>::DefaultImpl::DefaultImpl(
    const GemmInfo &task, IntType nThreads) : MatMulImplBase(task, nThreads)
{
    VERBOSE_PRINT;
    UpdateExecParams();
    Initialize();
}

template <typename FP1632Type>
constexpr std::uint32_t MatMulImplGenerator<VCF1632Gen<FP1632Type>>::DefaultImpl::GetKernelMSize() noexcept
{
    return VCF1632Gen<FP1632Type>::KERNEL_M_SIZE;
}

template <typename FP1632Type>
std::uint32_t MatMulImplGenerator<VCF1632Gen<FP1632Type>>::DefaultImpl::GetKernelNSize() noexcept
{
    return VCF1632Gen<FP1632Type>::KERNEL_N_SIZE * Vector::Length<FP1632Type>();
}

template <typename FP1632Type>
IntType MatMulImplGenerator<VCF1632Gen<FP1632Type>>::DefaultImpl::GetMBlockSize()
    const noexcept
{
    // GetMBlockSize() % KERNEL_M_SIZE == 0 for kernel
    // GetMBlockSize() % VL == 0 for packing
    // 1152 and 1200 is experimentally measured constant to obtain better perf
    return std::is_same_v<float, FP1632Type> ? 1152 : 1200;
}

template <typename FP1632Type>
IntType MatMulImplGenerator<VCF1632Gen<FP1632Type>>::DefaultImpl::GetNBlockSize()
    const noexcept
{
    // GetNBlockSize() % KERNEL_N_SIZE == 0 for kernel
    // GetNBlockSize() % VL == 0 for packing
    return 256; // 256 is experimentally measured constant to obtain better perf
}

template <typename FP1632Type>
IntType MatMulImplGenerator<VCF1632Gen<FP1632Type>>::DefaultImpl::GetKBlockSize()
    const noexcept
{
    // GetKBlockSize() % VL == 0  for kernel and packing
    // 320 and 640 is experimentally measured constant to obtain better perf
    return std::is_same_v<float, FP1632Type> ? 320 : 640;
}

template <typename FP1632Type>
void MatMulImplGenerator<VCF1632Gen<FP1632Type>>::DefaultImpl::InitializePackingPolicy()
{
    if (((dstLayout2D == Layout::ROW_MAJOR) && (broadcastedStridesSrc[IDX_3] < broadcastedStridesSrc[IDX_4])) ||
        ((dstLayout2D == Layout::COL_MAJOR) && (broadcastedStridesWei[IDX_3] >= broadcastedStridesWei[IDX_4]))) {
        packingPolicy = PackingPolicy::SYNC_DST_DIM_SRC;
    } else {
        packingPolicy = PackingPolicy::NO_SYNC;
    }
    InitializePackingPolicyFromEnv();
    if ((packingPolicy == PackingPolicy::SYNC_ACC_DIM_SRC) ||
        (packingPolicy == PackingPolicy::SYNC_ACC_DIM_WEI) ||
        (packingPolicy == PackingPolicy::SYNC_ACC_DIM_SRC_WEI)) {
        throw Service::LogicError("Matmul: Current packing policy not supported for this precision");
    }
}

template <typename FP1632Type>
void MatMulImplGenerator<VCF1632Gen<FP1632Type>>::DefaultImpl::InitializeSingleCodegen(
    IntType id, IntType m, IntType n, IntType k,
    IntType currentM, IntType currentN, IntType currentK,
    SizeType offsetSrc, SizeType offsetWei, SizeType offsetDst, SizeType offsetBia,
    bool wereSrcWeiSwapped, const Attributes &attributes, bool hasAlpha)
{
    Service::Unused(offsetSrc, offsetWei, offsetDst, offsetBia, hasAlpha, wereSrcWeiSwapped);
    if ((currentM == 0) || (currentN == 0) || (currentK == 0)) {
        return;
    }
    if ((currentM <= m) && (currentN <= n) && (currentK <= k)) {
        CodeGenParams<VCF1632Gen<FP1632Type>> currTask{currentM, currentN, currentK, false, attributes};
        if (codegens.find(currTask) == codegens.end()) {
            codegens.insert({currTask, std::make_shared<VCF1632Gen<FP1632Type>>(
                currentM, currentN, currentK, attributes)});
        }
        codegens[currTask]->Ready();
        matmulCodegenKernels[id] = codegens[currTask]->template GetCode<VCF1632CodegenFuncPtr<FP1632Type>>();
    }
}

template <typename FP1632Type>
void MatMulImplGenerator<VCF1632Gen<FP1632Type>>::DefaultImpl::InitializeReorderCodeGens(
    IntType m, IntType n, IntType k, IntType mFull, IntType nFull)
{
    Service::Unused(mFull);
    IntType vl = Vector::Length<FP1632Type>();

    IntType kBlockExecFull = GetKBlockSize();
    IntType kBlockExecResid = k % kBlockExecFull;

    if (perThreadParams[0].weiLayout2D == Layout::COL_MAJOR) {
        IntType nFullThreadBlock = nFull;
        IntType nResidThreadBlock = n % nFullThreadBlock;
        IntType nBlockExecResid1 = nFullThreadBlock % vl;
        IntType nBlockExecResid2 = nResidThreadBlock % vl;

        CodeGenWrapper<FastReorderKernelCodeGen<FP1632Type, FP1632Type>> reorderFNFK; // full N full K
        CodeGenWrapper<FastReorderKernelCodeGen<FP1632Type, FP1632Type>> reorderFNRK; // full N resid K

        CodeGenWrapper<FastReorderKernelCodeGen<FP1632Type, FP1632Type>> reorderRN1FK; // Resid1 N full K
        CodeGenWrapper<FastReorderKernelCodeGen<FP1632Type, FP1632Type>> reorderRN1RK; // Resid1 N resid K

        CodeGenWrapper<FastReorderKernelCodeGen<FP1632Type, FP1632Type>> reorderRN2FK; // Resid2 N full K
        CodeGenWrapper<FastReorderKernelCodeGen<FP1632Type, FP1632Type>> reorderRN2RK; // Resid2 N resid K

        reorderFNFK.Initialize(static_cast<SizeType>(vl),
            static_cast<SizeType>(kBlockExecFull), false);
        reorderFNRK.Initialize(static_cast<SizeType>(vl),
            static_cast<SizeType>(kBlockExecResid), false);
        reorderRN1FK.Initialize(static_cast<SizeType>(nBlockExecResid1),
            static_cast<SizeType>(kBlockExecFull), false);
        reorderRN1RK.Initialize(static_cast<SizeType>(nBlockExecResid1),
            static_cast<SizeType>(kBlockExecResid), false);
        reorderRN2FK.Initialize(static_cast<SizeType>(nBlockExecResid2),
            static_cast<SizeType>(kBlockExecFull), false);
        reorderRN2RK.Initialize(static_cast<SizeType>(nBlockExecResid2),
            static_cast<SizeType>(kBlockExecResid), false);

        reorderPackingCodegenKernels = {
            std::move(reorderFNFK), std::move(reorderFNRK),
            std::move(reorderRN1FK), std::move(reorderRN1RK),
            std::move(reorderRN2FK), std::move(reorderRN2RK),
        };
    }

    if (perThreadParams[0].srcLayout2D == Layout::COL_MAJOR) {
        SizeType mPackPortion = GetPackPortion(MatrixKind::SRC);
        SizeType mBlockSize = GetMBlockSize();
        SizeType mPerThread = Math::CeilDiv(m, threadsM);
        SizeType mResid =  m % mPerThread;

        SizeType mPackSize = std::min(mPackPortion, mPerThread);
        SizeType mResidThread = (mPerThread % mBlockSize) % mPackSize;
        SizeType mResidFull = (mResid  % mBlockSize) % mPackSize;

        CodeGenWrapper<packAKernelCodeGenType> cgwPackMksKbs;
        CodeGenWrapper<packAKernelCodeGenType> cgwPackMksKbsResid;
        CodeGenWrapper<packAKernelCodeGenType> cgwPackMksResidKbs;
        CodeGenWrapper<packAKernelCodeGenType> cgwPackMksResidKbsResid;
        CodeGenWrapper<packAKernelCodeGenType> cgwPackMksResidFullKbs;
        CodeGenWrapper<packAKernelCodeGenType> cgwPackMksResidFullKbsResid;

        cgwPackMksKbs.Initialize(static_cast<SizeType>(kBlockExecFull), mPackSize, false);
        cgwPackMksKbsResid.Initialize(static_cast<SizeType>(kBlockExecResid), mPackSize, false);
        cgwPackMksResidKbs.Initialize(static_cast<SizeType>(kBlockExecFull), mResidThread, false);
        cgwPackMksResidKbsResid.Initialize(static_cast<SizeType>(kBlockExecResid), mResidThread, false);
        cgwPackMksResidFullKbs.Initialize(static_cast<SizeType>(kBlockExecFull), mResidFull, false);
        cgwPackMksResidFullKbsResid.Initialize(static_cast<SizeType>(kBlockExecResid), mResidFull, false);

        packAColCodegenKernels = {
            std::move(cgwPackMksKbs), std::move(cgwPackMksKbsResid),
            std::move(cgwPackMksResidKbs), std::move(cgwPackMksResidKbsResid),
            std::move(cgwPackMksResidFullKbs), std::move(cgwPackMksResidFullKbsResid)
        };
    }
}

template <typename FP1632Type>
void MatMulImplGenerator<VCF1632Gen<FP1632Type>>::DefaultImpl::InitializePackingCodeGens(
    IntType m, IntType n, IntType k, IntType mFull, IntType nFull)
{
    Service::Unused(m, k, mFull);
    if (perThreadParams[0].weiLayout2D == Layout::AB) {
        IntType nBlockExecFull;
        IntType nBlockExecResid1;
        IntType nBlockExecResid2;
        InitializeKernelMorN(n, nFull, GetPackPortion(MatrixKind::WEI),
            nBlockExecFull, nBlockExecResid1, nBlockExecResid2);
        for (IntType i = 0; i < NUM_3; ++i) {
            IntType currentN;
            InitializeCurrentMorN(i, nBlockExecFull, nBlockExecResid1, nBlockExecResid2, currentN);
            if ((currentN <= n) && (currentN != 0)) {
                PackVCF1632Params<FP1632Type> currTask{currentN};
                if (packingCodegens.find(currTask) == packingCodegens.end()) {
                    packingCodegens.insert({currTask, std::shared_ptr<PackBRowDefaultSchemeCodegenF16F32<FP1632Type>>{
                        new PackBRowDefaultSchemeCodegenF16F32<FP1632Type>{currentN}}});
                }
                packingCodegens[currTask]->Ready();
                vectorPackBRowCodegenKernels[i] =
                    packingCodegens[currTask]->template GetCode<VCF1632PackingCodegenFuncPtr<FP1632Type>>();
            }
        }
    }
}

template <typename FP1632Type>
void MatMulImplGenerator<VCF1632Gen<FP1632Type>>::DefaultImpl::InitializeThreadsCountImpl()
{
    IntType threadNumber = InitThreadNum();

    IntType size = GetRequiredMemSize();
    if constexpr (std::is_same_v<__fp16, FP1632Type>) {
        if (size < 6144) { // 6144 is practically measured number
            threadNumber = NUM_1;
        } else if (size < 38400) { // 38400 is practically measured number
            threadNumber = NUM_8;
        } else if (size < 104544) { // 104544 is practically measured number
            threadNumber = NUM_16;
        } else if (size < 269664) { // 269664 is practically measured number
            threadNumber = NUM_40;
        }
    } else {
        if (size < 21216) { // 21216 is practically measured number
            threadNumber = NUM_1;
        } else if (size < 76800) { // 76800 is practically measured number
            threadNumber = NUM_4;
        } else if (size < 209088) { // 209088 is practically measured number
            threadNumber = NUM_24;
        } else if (size < 1080768) { // 1080768 is practically measured number
            threadNumber = NUM_40;
        }
    }
    nThreadsExec = std::min(threadNumber, implNThreads);
}

template <typename FP1632Type>
IntType MatMulImplGenerator<VCF1632Gen<FP1632Type>>::DefaultImpl::GetThreadBufSize(
    IntType m, IntType n, IntType k, MatrixKind mKind) const noexcept
{
    IntType vl = Vector::Length<FP1632Type>();
    if (mKind == MatrixKind::SRC) {
        if (((dstLayout2D == Layout::ROW_MAJOR) && (broadcastedStridesSrc[IDX_3] < broadcastedStridesSrc[IDX_4])) ||
            ((dstLayout2D == Layout::COL_MAJOR) && (broadcastedStridesWei[IDX_3] >= broadcastedStridesWei[IDX_4]))) {
            return std::min(GetKBlockSize(), k) * std::min(GetMBlockSize(), m) * sizeof(FP1632Type);
        } else {
            return 0;
        }
    } else {
        return static_cast<IntType>(sizeof(FP1632Type) * std::min<IntType>(GetKBlockSize(), k) *
            Math::CeilDiv(std::min<IntType>(GetNBlockSize(), n), VCF1632Gen<FP1632Type>::KERNEL_N_SIZE * vl) *
            VCF1632Gen<FP1632Type>::KERNEL_N_SIZE * vl);
    }
}

template <typename FP1632Type>
void MatMulImplGenerator<VCF1632Gen<FP1632Type>>::DefaultImpl::ReserveImplSpecificParams() {}

template <typename FP1632Type>
std::vector<std::pair<Shape, Shape>> MatMulImplGenerator<VCF1632Gen<FP1632Type>>::DefaultImpl::SplitWork()
{
    return CalculateWorkOnThread<DefaultImpl>(taskExec.dstInfo.GetDims(), nThreadsExec, dstLayout2D == Layout::BA);
}

template <typename FP1632Type>
IntType MatMulImplGenerator<VCF1632Gen<FP1632Type>>::DefaultImpl::GetPackPortion(MatrixKind mKind) const
{
    const IntType vl = Vector::Length<FP1632Type>();
    IntType packPortion = 0;
    if (mKind == MatrixKind::SRC) {
        if (packingPolicy & PackingPolicy::SYNC_DST_DIM_SRC) {
            packPortion = vl;
        } else {
            packPortion = GetMBlockSize();
        }
    } else if (mKind == MatrixKind::WEI) {
        if (packingPolicy & PackingPolicy::SYNC_DST_DIM_WEI) {
            packPortion = vl;
        } else {
            packPortion = GetNBlockSize();
        }
    }

    return packPortion;
}

template <typename FP1632Type>
void MatMulImplGenerator<VCF1632Gen<FP1632Type>>::DefaultImpl::PackSrcBA(
    IntType threadID, IntType m, IntType k, const void *src, IntType offset, void *buffer) const
{
    IntType packPortion = GetPackPortion(MatrixKind::SRC);
    IntType kIdxKernel = (k == GetKBlockSize()) ? 0 : 1;
    IntType packSize = std::min(perThreadParams[NUM_0].m, packPortion);

    IntType mResid =  (m == packSize) ? 0 : 1;
    IntType mResidFull = mResid * (perThreadParams[threadID].isResidByM);

    IntType packIdx = mResid * NUM_2 + mResidFull * NUM_2 + kIdxKernel * NUM_1;
    IntType kOffset = std::min(GetKBlockSize(), perThreadParams[threadID].k);

    auto packA = packAColCodegenKernels[packIdx].
        template GetCode<void(*)(const FP1632Type *, FP1632Type *, IntType, IntType)>();
        packA(static_cast<const FP1632Type *>(src),
            static_cast<FP1632Type *>(buffer),
            offset * sizeof(FP1632Type),
            kOffset * sizeof(FP1632Type));
}

template <typename FP1632Type>
void MatMulImplGenerator<VCF1632Gen<FP1632Type>>::DefaultImpl::PackWeiAB(IntType threadID, IntType n, IntType k,
    const void *wei, IntType offset, void *buffer) const
{
    IntType packPortion = GetPackPortion(MatrixKind::WEI);

    IntType nIdx = (n == packPortion) ? 0 : 1;
    IntType nIdxKernel = nIdx * (perThreadParams[threadID].isResidByN + 1);

    vectorPackBRowCodegenKernels[nIdxKernel](n, k,
        static_cast<const typename VCF1632Gen<FP1632Type>::WeiType *>(wei),
        offset, static_cast<FP1632Type *>(buffer));
}

template <typename FP1632Type>
void MatMulImplGenerator<VCF1632Gen<FP1632Type>>::DefaultImpl::PackWeiBA(IntType threadID, IntType n, IntType k,
    const void *src, IntType offset, void *buffer) const
{
    const IntType vl = Vector::Length<FP1632Type>();
    IntType idxK = (k == GetKBlockSize()) ? 0 : 1;
    for (IntType nIdx = n; nIdx > 0; nIdx -= vl) {
        IntType idxN = (nIdx < vl) ? 1 : 0;
        IntType nIdxKernel = idxN * (perThreadParams[threadID].isResidByN + 1);
        IntType packIdx = nIdxKernel * NUM_2 + idxK;
        auto packBcol = reorderPackingCodegenKernels[packIdx]
                    .template GetCode<void(*)(const FP1632Type*, FP1632Type*, SizeType, SizeType)>();
        packBcol(static_cast<const typename VCF1632Gen<FP1632Type>::WeiType *>(src) + (n - nIdx) * offset,
            static_cast<FP1632Type *>(buffer) + k * (n - nIdx),
            static_cast<SizeType>(offset) * sizeof(FP1632Type),
            static_cast<SizeType>(vl) * sizeof(FP1632Type));
    }
}

template <typename FP1632Type>
void MatMulImplGenerator<VCF1632Gen<FP1632Type>>::DefaultImpl::Pack(const void *originalMatrix, void *buffer,
    IntType threadID, IntType dstDim, IntType accumDim, IntType dstDimOffset, IntType accumDimOffset,
    IntType bufferOffset, IntType bufferStride, float alpha, MatrixKind mKind) const
{
    Service::Unused(bufferStride, alpha);

    Layout layout;
    IntType offset2D;
    auto initParams = [&] () {
        if (mKind == MatrixKind::SRC) {
            layout = perThreadParams[threadID].srcLayout2D;
            offset2D = perThreadParams[threadID].srcOffset2D;
        } else {
            layout = perThreadParams[threadID].weiLayout2D;
            offset2D = perThreadParams[threadID].weiOffset2D;
        }
    };
    initParams();
    if (mKind == MatrixKind::SRC) {
        const FP1632Type *srcExec = static_cast<const FP1632Type *>(originalMatrix);
        if (layout == Layout::BA) {
            srcExec += dstDimOffset + accumDimOffset * offset2D;
            PackSrcBA(threadID, dstDim, accumDim, srcExec,
                offset2D, static_cast<FP1632Type *>(buffer) + bufferOffset);
        }
    } else {
        const FP1632Type *weiExec = static_cast<const FP1632Type *>(originalMatrix);
        if (layout == Layout::AB) {
            weiExec += dstDimOffset + accumDimOffset * offset2D;
            PackWeiAB(threadID, dstDim, accumDim, weiExec,
                offset2D, static_cast<FP1632Type *>(buffer) + bufferOffset);
        } else {
            weiExec += dstDimOffset * offset2D + accumDimOffset;
            PackWeiBA(threadID, dstDim, accumDim, weiExec,
                offset2D, static_cast<FP1632Type *>(buffer) + bufferOffset);
        }
    }
}

template <typename FP1632Type>
void MatMulImplGenerator<VCF1632Gen<FP1632Type>>::DefaultImpl::RunSequential2D(
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
    FP1632Type* srcBuffer = static_cast<FP1632Type *>(
        srcPackStrategy->GetPtr(tIDbatch * threadsM  + tIDm, tIDn, threadsN));
    FP1632Type* weiBuffer = static_cast<FP1632Type *>(
        weiPackStrategy->GetPtr(tIDbatch * threadsN  + tIDn, tIDm, threadsM));

    IntType offsetSrc = perThreadParams[threadID].srcOffset2D;
    if ((IsLayoutAlikeBA(taskExec.srcInfo.GetLayout()) && IsLayoutAlikeAB(taskExec.dstInfo.GetLayout())) ||
        (IsLayoutAlikeAB(taskExec.weiInfo.GetLayout()) && IsLayoutAlikeBA(taskExec.dstInfo.GetLayout()))) {
        offsetSrc =
            std::min<IntType>(taskExec.srcInfo.GetDims()[taskExec.srcInfo.GetNumDims() - NUM_1], GetKBlockSize());
    }

    for (IntType mBlock = 0; mBlock < mFullBlockSizePerThread; mBlock += GetMBlockSize()) {
        auto mBlockExec = std::min(GetMBlockSize(),  m - mBlock);

        IntType mIdx = (mBlockExec == GetMBlockSize()) ? 0 : 1;
        IntType mIdxKernel = mIdx * (perThreadParams[threadID].isResidByM + 1);
        for (IntType kBlock = 0; kBlock < k; kBlock += GetKBlockSize()) {
            auto kBlockExec = std::min(GetKBlockSize(), k - kBlock);
            IntType kIdxKernel = (kBlockExec == GetKBlockSize()) ? 0 : 1;
            bool lastKIter = (k - kBlock) <= GetKBlockSize();
            MatMulDstFlags biaFlagsExec = SetBiaFlagsExec(biaFlags, beta, kBlock, lastKIter);
            if (mBlock < m) {
                srcPackStrategy->Pack(*this, src, srcBuffer,
                    {mBlock, mBlockExec, GetPackPortion(MatrixKind::SRC)},
                    {kBlock, kBlockExec, std::min(GetKBlockSize(), k)},
                    threadsN, tIDn, threadID, alpha);
            }
            srcPackStrategy->Barrier(tIDbatch * threadsM + tIDm, tIDn, threadsN);
            const FP1632Type *srcExec = static_cast<const FP1632Type *>(src);
            if (perThreadParams[threadID].srcLayout2D == Layout::AB) {
                srcExec += kBlock + mBlock * perThreadParams[threadID].srcOffset2D;
            } else {
                srcExec = srcBuffer;
            }
            for (IntType nBlock = 0; nBlock < nFullBlockSizePerThread; nBlock += GetNBlockSize()) {
                if ((nBlock < n) && (mBlock < m)) {
                    auto nBlockExec = std::min(GetNBlockSize(), n - nBlock);
                    IntType nIdx = (nBlockExec == GetNBlockSize()) ? 0 : 1;
                    IntType nIdxKernel = nIdx * (perThreadParams[threadID].isResidByN + 1);
                    weiPackStrategy->Pack(*this, wei, weiBuffer,
                        {nBlock, nBlockExec, GetPackPortion(MatrixKind::WEI)}, {kBlock, kBlockExec, 1},
                        threadsM, tIDm, threadID, alpha);
                    weiPackStrategy->Barrier(tIDbatch * threadsN + tIDn, tIDm, threadsM);
                    FP1632Type *dstExec = static_cast<FP1632Type *>(dst) + nBlock +
                        mBlock * perThreadParams[threadID].dstOffset2D;
                    const FP1632Type *biaExec = static_cast<const FP1632Type *>(bia) +
                        nBlock * biaNOffset + mBlock * perThreadParams[threadID].biaOffset2D;
                    auto kernel = matmulCodegenKernels[kIdxKernel * NUM_9 +
                        nIdxKernel * NUM_3 + mIdxKernel];
                    kernel(srcExec, weiBuffer, dstExec, biaExec,
                        biaFlagsExec, alpha, beta, mBlockExec, nBlockExec, kBlockExec,
                        offsetSrc, kBlockExec * Vector::Length<FP1632Type>(), perThreadParams[threadID].dstOffset2D,
                        perThreadParams[threadID].biaOffset2D);
                    weiPackStrategy->Barrier(tIDbatch * threadsN + tIDn, tIDm, threadsM);
                }
            }
            if (perThreadParams[threadID].srcLayout2D == Layout::BA) {
                srcPackStrategy->Barrier(tIDbatch * threadsM + tIDm, tIDn, threadsN);
            }
        }
    }
}

template <typename FP1632Type>
void MatMulImplGenerator<VCF1632Gen<FP1632Type>>::DefaultImpl::UpdateExecParams() noexcept
{
    VERBOSE_PRINT;
    Layout desiredSrcLayout = implTask.srcInfo.GetLayout();
    Layout desiredWeiLayout = implTask.weiInfo.GetLayout();
    Layout desiredDstLayout = implTask.dstInfo.GetLayout();
    Layout desiredBiaLayout = implTask.biaInfo.GetLayout();
    {
        if (IsLayoutAlikeAB(desiredDstLayout)) {
            bool checkDesiredSrcLayout1 = !IsLayoutAlikeAB(desiredSrcLayout) && !IsLayoutAlikeBA(desiredSrcLayout);
            if (checkDesiredSrcLayout1) {
                desiredSrcLayout = GetPlaneLayoutAB(desiredSrcLayout);
            }
            bool checkDesiredWeiLayout1 = !IsLayoutAlikeAB(desiredWeiLayout) && !IsLayoutAlikeBA(desiredWeiLayout);
            if (checkDesiredWeiLayout1) {
                desiredWeiLayout = GetPlaneLayoutAB(desiredWeiLayout);
            }
        } else if (IsLayoutAlikeBA(desiredDstLayout)) {
            bool checkDesiredWeiLayout2 = !IsLayoutAlikeAB(desiredWeiLayout) && !IsLayoutAlikeBA(desiredWeiLayout);
            if (checkDesiredWeiLayout2) {
                desiredWeiLayout = GetPlaneLayoutBA(desiredWeiLayout);
            }
            bool checkDesiredSrcLayout2 = !IsLayoutAlikeAB(desiredSrcLayout) && !IsLayoutAlikeBA(desiredSrcLayout);
            if (checkDesiredSrcLayout2) {
                desiredSrcLayout = GetPlaneLayoutBA(desiredSrcLayout);
            }
        } else {
            if (IsLayoutAlikeAB(desiredSrcLayout)) {
                desiredDstLayout = GetPlaneLayoutAB(desiredDstLayout);
                bool checkDesiredWeiLayout3 = !IsLayoutAlikeAB(desiredWeiLayout) && !IsLayoutAlikeBA(desiredWeiLayout);
                if (checkDesiredWeiLayout3) {
                    desiredWeiLayout = GetPlaneLayoutAB(desiredWeiLayout);
                }
            } else if (IsLayoutAlikeBA(desiredWeiLayout)) {
                desiredDstLayout = GetPlaneLayoutBA(desiredDstLayout);
                bool checkDesiredSrcLayout3 = !IsLayoutAlikeAB(desiredSrcLayout) && !IsLayoutAlikeBA(desiredSrcLayout);
                if (checkDesiredSrcLayout3) {
                    desiredSrcLayout = GetPlaneLayoutBA(desiredSrcLayout);
                }
            } else {
                desiredDstLayout = GetPlaneLayoutAB(desiredDstLayout);
                desiredSrcLayout = GetPlaneLayoutAB(desiredSrcLayout);
                bool checkDesiredWeiLayout4 = !IsLayoutAlikeAB(desiredWeiLayout) && !IsLayoutAlikeBA(desiredWeiLayout);
                if (checkDesiredWeiLayout4) {
                    desiredWeiLayout = GetPlaneLayoutAB(desiredWeiLayout);
                }
            }
        }
    }
    if (implTask.biaInfo.GetTotalTensorSize() != 0) {
        desiredBiaLayout = desiredDstLayout;
    }
    InitializeTaskExec(desiredSrcLayout, desiredWeiLayout, desiredDstLayout, desiredBiaLayout,
        VCF1632Gen<FP1632Type>::GetSrcType(),
        VCF1632Gen<FP1632Type>::GetWeiType(),
        VCF1632Gen<FP1632Type>::GetDstType(),
        (implTask.biaInfo.GetTotalTensorSize() != 0) ? VCF1632Gen<FP1632Type>::GetBiaType() :
            static_cast<Element::TypeT>(implTask.biaInfo.GetType()));
}

template <typename FP1632Type>
MatMulDstFlags MatMulImplGenerator<VCF1632Gen<FP1632Type>>::DefaultImpl::SetBiaFlagsExec(
    MatMulDstFlags biaF, float beta, bool notFirstKIter, bool lastKIter, float alpha) const noexcept
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

template struct MatMulImplGenerator<MatMulKernelDefaultCodeGen
    <ARCH::VC, __fp16, __fp16, __fp16, __fp16, MatMulImplKinds::F1632F1632F1632F1632_VC>>;
template struct MatMulImplGenerator<MatMulKernelDefaultCodeGen
    <ARCH::VC, float, float, float, float, MatMulImplKinds::F1632F1632F1632F1632_VC>>;

} // Detail
} // KuDNN
