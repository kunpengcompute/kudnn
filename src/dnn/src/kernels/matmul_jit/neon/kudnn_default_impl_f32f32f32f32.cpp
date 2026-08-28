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
#include <algorithm>

#include "kudnn.hpp"

#include "cg_codegen.hpp"

#include "kudnn_jit.hpp"
#include "kernels/matmul_jit/kudnn_gemm_jit.hpp"
#include "kernels/matmul_jit/kudnn_gemm_helpers.hpp"
#include "kernels/matmul_jit/kudnn_gemm_threading_helpers.hpp"
#include "kernels/reorder/sve/bf16any32/kudnn_fast_jit_impl_bfp16any32_bfp16any32.hpp"
#include "kudnn_wrapper_helpers.hpp"
#include "kernels/matmul_jit/neon/kudnn_default_impl_f32f32f32f32.hpp"

namespace KuDNN {
namespace Detail {

using NeonF32GenBase = MatMulKernelCodeGenBase<ARCH::NEON, float, float, float, float>;

NeonF32Gen::MatMulKernelDefaultCodeGen(
    IntType m, IntType n, IntType k,
    const Attributes &attributesArg) : NeonF32GenBase(attributesArg)
{
    if ((m > 0) && (n > 0) && (k > 0)) {
        if (attributes.GetPostOps().Len() > 0) {
            // set reserved dst registers
            for (std::uint32_t mIdx = 0; mIdx < KERNEL_M_SIZE; ++mIdx) {
                for (std::uint32_t nIdx = 0; nIdx < KERNEL_N_SIZE; ++nIdx) {
                    TakeVectorRegister(GetDstVRegIdx(mIdx, nIdx));
                }
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

        Sub(sp, sp, NUM_16);
        Str(alpha, ARMCG::AdrUimm(Ptr(sp))); // alpha will be overwritten while accumulation
        Sub(sp, sp, NUM_16);
        Str(beta, ARMCG::AdrUimm(Ptr(sp))); // beta will be overwritten while accumulation

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


void NeonF32Gen::StoreToDst(
    std::uint32_t mLim, std::uint32_t nLim, std::uint32_t nResidScalar)
{
    // can be reworked to reuse some code (at least nested loops)
    ARMCG::Label loadCNeon;
    ARMCG::Label mullCNeon;
    ARMCG::Label loadBias1Neon;
    ARMCG::Label loadBiasMNeon;
    ARMCG::Label loadBiasNNeon;
    ARMCG::Label loadBiasMNNeon;
    ARMCG::Label addBias1Neon;
    ARMCG::Label addBiasMNeon;
    ARMCG::Label addBiasNNeon;
    ARMCG::Label addBiasMNNeon;
    ARMCG::Label initDoneNeon;
    ARMCG::Label skipPO;

    // Load alpha
    Add(tmp, sp, NUM_16);
    ARMCG::CodeGenerator::Ld1r(ARMCG::VReg4SList{ARMCG::VReg4S(ARMCG::VReg4S(NUM_0))}, ARMCG::AdrNoOfs(tmp));
    Mov(tmpW, ARMCG::VReg4S(ARMCG::VReg4S(NUM_0))[0]);
    Mov(ARMCG::WReg(GetTmpXRegIdx(0)), 0x3f800000); // 0x3f800000 is a bit representation of 1.0f
    Cmp(tmpW, ARMCG::WReg(GetTmpXRegIdx(0)));
    Beq(loadBias1Neon);
    for (std::uint32_t mIdx = 0; mIdx < mLim; ++mIdx) {
        for (std::uint32_t nIdx = 0; nIdx < nLim; ++nIdx) {
            Fmul(ARMCG::VReg4S(GetDstVRegIdx(mIdx, nIdx)),
                 ARMCG::VReg4S(GetDstVRegIdx(mIdx, nIdx)),
                 ARMCG::VReg4S(NUM_0)); // z7 contains bias value
        }
    }
    Tbz(flags, 0, loadBias1Neon); // 0-bit correspond to FILL_C_WITH_ZEROZ
    L(loadBias1Neon);
    Tbz(flags, 3, loadBiasNNeon); // 3-bit correspond to LOAD_BIAS_1
    CodeGenerator::Ld1r(ARMCG::VReg4SList(ARMCG::VReg4S(NUM_7)), Ptr(biaPtr));
    for (std::uint32_t mIdx = 0; mIdx < mLim; ++mIdx) {
        for (std::uint32_t nIdx = 0; nIdx < nLim; ++nIdx) {
            Fadd(ARMCG::VReg4S(GetDstVRegIdx(mIdx, nIdx)),
                 ARMCG::VReg4S(GetDstVRegIdx(mIdx, nIdx)),
                 ARMCG::VReg4S(NUM_7)); // z7 contains bias value
        }
    }
    L(loadBiasNNeon);
    Tbz(flags, 5, loadBiasMNeon); // 5-bit correspond to LOAD_BIAS_N
    Add(tmp, biaPtr, tmpNMinusNLoopCtr, ARMCG::ShMod::LSL, Math::Log2(sizeof(float)));
    for (std::uint32_t nIdx = 0; nIdx < nLim; ++nIdx) {
        // z4-z10 registers are chosen for tmp values, could be any but dst Z registers
        if (((nResidScalar != NEON_VL)) && (nIdx == (nLim - 1))) {
            for (IntType j = 0; j < nResidScalar; j++) {
                CodeGenerator::Ld1(ARMCG::VReg4SList(ARMCG::VReg4S(NUM_4 + nIdx))[j],
                    PostPtr(tmp, sizeof(float)));
            }
        } else {
            CodeGenerator::Ld1(ARMCG::VReg4SList(ARMCG::VReg4S(NUM_4 + nIdx)),
                PostPtr(tmp, NEON_VL * sizeof(float)));
        }
    }
    for (std::uint32_t mIdx = 0; mIdx < mLim; ++mIdx) {
        for (std::uint32_t nIdx = 0; nIdx < nLim; ++nIdx) {
            Fadd(ARMCG::VReg4S(GetDstVRegIdx(mIdx, nIdx)),
                 ARMCG::VReg4S(GetDstVRegIdx(mIdx, nIdx)),
                 ARMCG::VReg4S(NUM_4 + nIdx)); // z4-z10 registers are chosen for tmp values
        }
    }
    L(loadBiasMNeon);
    Tbz(flags, 4, loadBiasMNNeon); // 4-bit correspond to LOAD_BIAS_M
    Add(tmp, biaPtr, tmpMMinusMLoopCtr, ARMCG::ShMod::LSL, Math::Log2(sizeof(float)));
    for (std::uint32_t mIdx = 0; mIdx < mLim; ++mIdx) {
        CodeGenerator::Ld1r(ARMCG::VReg4SList(ARMCG::VReg4S(GetSrcVRegIdx(mIdx))), Ptr(tmp));
        CodeGenerator::Add(tmp, tmp, sizeof(float));
    }
    for (std::uint32_t mIdx = 0; mIdx < mLim; ++mIdx) {
        for (std::uint32_t nIdx = 0; nIdx < nLim; ++nIdx) {
            Fadd(ARMCG::VReg4S(GetDstVRegIdx(mIdx, nIdx)),
                ARMCG::VReg4S(GetDstVRegIdx(mIdx, nIdx)),
                ARMCG::VReg4S(GetSrcVRegIdx(mIdx)));
        }
    }
    L(loadBiasMNNeon);
    Tbz(flags, 6, loadCNeon); // 6-bit correspond to LOAD_BIAS_MN
    Mul(tmp, tmpMMinusMLoopCtr, biaOffset);
    Add(ARMCG::XReg(GetSrcXRegIdx(0)), biaPtr, tmp,
        ARMCG::ShMod::LSL, Math::Log2(sizeof(float)));
    Add(ARMCG::XReg(GetSrcXRegIdx(0)),
        ARMCG::XReg(GetSrcXRegIdx(0)),
        tmpNMinusNLoopCtr, ARMCG::ShMod::LSL, Math::Log2(sizeof(float)));
    for (std::uint32_t mIdx = 1; mIdx < mLim; ++mIdx) {
        Add(ARMCG::XReg(GetSrcXRegIdx(mIdx)),
            ARMCG::XReg(GetSrcXRegIdx(mIdx - 1)),
            biaOffset, ARMCG::ShMod::LSL, Math::Log2(sizeof(float)));
    }
    for (std::uint32_t mIdx = 0; mIdx < mLim; ++mIdx) {
        for (std::uint32_t nIdx = 0; nIdx < nLim; ++nIdx) {
            // z4-z10 registers are chosen for tmp values, could be any except dst registers
            if (((nResidScalar != NEON_VL)) && (nIdx == (nLim - 1))) {
                for (IntType j = 0; j < nResidScalar; j++) {
                    CodeGenerator::Ld1(ARMCG::VReg4S(NUM_4 + nIdx)[j],
                                       PostPtr(ARMCG::XReg(GetSrcXRegIdx(mIdx)), sizeof(float)));
                }
            } else {
                CodeGenerator::Ld1(ARMCG::VReg4SList(ARMCG::VReg4S(NUM_4 + nIdx)),
                                   PostPtr(ARMCG::XReg(GetSrcXRegIdx(mIdx)),
                                           NEON_VL * sizeof(float)));
            }
            Fadd(ARMCG::VReg4S(GetDstVRegIdx(mIdx, nIdx)),
                 ARMCG::VReg4S(GetDstVRegIdx(mIdx, nIdx)),
                 ARMCG::VReg4S(NUM_4 + nIdx)); // z4-z10 registers are chosen for tmp values
        }
    }
    L(loadCNeon);
    Tbz(flags, 1, mullCNeon); // 1-bit correspond to LOAD_C
    Mul(tmp, tmpMMinusMLoopCtr, dstOffset);
    Add(ARMCG::XReg(GetSrcXRegIdx(0)), dstPtr, tmp,
        ARMCG::ShMod::LSL, Math::Log2(sizeof(float)));
    Add(ARMCG::XReg(GetSrcXRegIdx(0)), ARMCG::XReg(GetSrcXRegIdx(0)),
        tmpNMinusNLoopCtr, ARMCG::ShMod::LSL, Math::Log2(sizeof(float)));
    for (std::uint32_t mIdx = 1; mIdx < mLim; ++mIdx) {
        Add(ARMCG::XReg(GetSrcXRegIdx(mIdx)),
            ARMCG::XReg(GetSrcXRegIdx(mIdx - 1)),
            dstOffset, ARMCG::ShMod::LSL, Math::Log2(sizeof(float)));
    }
    for (std::uint32_t mIdx = 0; mIdx < mLim; ++mIdx) {
        for (std::uint32_t nIdx = 0; nIdx < nLim; ++nIdx) {
            // handle N % 4 != 0
            if (((nResidScalar != NEON_VL)) && (nIdx == (nLim - 1))) {
                for (IntType j = 0; j < nResidScalar; j++) {
                    CodeGenerator::Ld1(ARMCG::VReg4S(GetSrcVRegIdx(mIdx))[j],
                                       PostPtr(ARMCG::XReg(GetSrcXRegIdx(mIdx)), sizeof(float)));
                }
            } else {
                CodeGenerator::Ld1(ARMCG::VReg4SList(ARMCG::VReg4S(GetSrcVRegIdx(mIdx))),
                                   PostPtr(ARMCG::XReg(GetSrcXRegIdx(mIdx)), NEON_VL * sizeof(float)));
            }
            Fadd(ARMCG::VReg4S(GetDstVRegIdx(mIdx, nIdx)),
                ARMCG::VReg4S(GetDstVRegIdx(mIdx, nIdx)),
                ARMCG::VReg4S(GetSrcVRegIdx(mIdx)));
        }
    }
    L(mullCNeon);
    Tbz(flags, 2, addBias1Neon); // 2-bit correspond to MULL_C
    CodeGenerator::Ld1r(ARMCG::VReg4SList(ARMCG::VReg4S(NUM_1)), Ptr(sp));
    Mul(tmp, tmpMMinusMLoopCtr, dstOffset);
    Add(ARMCG::XReg(GetSrcXRegIdx(0)), dstPtr, tmp,
        ARMCG::ShMod::LSL, Math::Log2(sizeof(float)));
    Add(ARMCG::XReg(GetSrcXRegIdx(0)), ARMCG::XReg(GetSrcXRegIdx(0)),
        tmpNMinusNLoopCtr, ARMCG::ShMod::LSL, Math::Log2(sizeof(float)));
    for (std::uint32_t mIdx = 1; mIdx < mLim; ++mIdx) {
        Add(ARMCG::XReg(GetSrcXRegIdx(mIdx)),
            ARMCG::XReg(GetSrcXRegIdx(mIdx - 1)),
            dstOffset, ARMCG::ShMod::LSL, Math::Log2(sizeof(float)));
    }
    for (std::uint32_t mIdx = 0; mIdx < mLim; ++mIdx) {
        for (std::uint32_t nIdx = 0; nIdx < nLim; ++nIdx) {
            // handle N % 4 != 0
            if (((nResidScalar != NEON_VL)) && (nIdx == (nLim - 1))) {
                for (IntType j = 0; j < nResidScalar; j++) {
                    CodeGenerator::Ld1(ARMCG::VReg4S(GetSrcVRegIdx(mIdx))[j],
                                       PostPtr(ARMCG::XReg(GetSrcXRegIdx(mIdx)), sizeof(float)));
                }
            } else {
                CodeGenerator::Ld1(ARMCG::VReg4SList(ARMCG::VReg4S(GetSrcVRegIdx(mIdx))),
                                   PostPtr(ARMCG::XReg(GetSrcXRegIdx(mIdx)), NEON_VL * sizeof(float)));
            }
            Fmul(ARMCG::VReg4S(GetSrcVRegIdx(mIdx)),
                 ARMCG::VReg4S(GetSrcVRegIdx(mIdx)),
                 ARMCG::VReg4S(NUM_1)); // z7 contains beta value
            Fadd(ARMCG::VReg4S(GetDstVRegIdx(mIdx, nIdx)),
                ARMCG::VReg4S(GetDstVRegIdx(mIdx, nIdx)),
                ARMCG::VReg4S(GetSrcVRegIdx(mIdx)));
        }
    }
    L(addBias1Neon);
    Tbz(flags, 7, addBiasNNeon); // 7-bit correspond to Add_BIAS_1
    CodeGenerator::Ld1r(ARMCG::VReg4SList(ARMCG::VReg4S(NUM_7)), Ptr(biaPtr));
    for (std::uint32_t mIdx = 0; mIdx < mLim; ++mIdx) {
        for (std::uint32_t nIdx = 0; nIdx < nLim; ++nIdx) {
            Fadd(ARMCG::VReg4S(GetDstVRegIdx(mIdx, nIdx)),
                 ARMCG::VReg4S(GetDstVRegIdx(mIdx, nIdx)),
                 ARMCG::VReg4S(NUM_7)); // z7 contains bias value
        }
    }
    L(addBiasNNeon);
    Tbz(flags, 9, addBiasMNeon); // 9-bit correspond to Add_BIAS_N
    Add(tmp, biaPtr, tmpNMinusNLoopCtr, ARMCG::ShMod::LSL, Math::Log2(sizeof(float)));
    for (std::uint32_t nIdx = 0; nIdx < nLim; ++nIdx) {
        // z4-z10 registers are chosen for tmp values, could be any but dst Z registers
        if (((nResidScalar != NEON_VL)) && (nIdx == (nLim - 1))) {
            for (IntType j = 0; j < nResidScalar; j++) {
                CodeGenerator::Ld1(ARMCG::VReg4SList(ARMCG::VReg4S(NUM_4 + nIdx))[j],
                    PostPtr(tmp, sizeof(float)));
            }
        } else {
            CodeGenerator::Ld1(ARMCG::VReg4SList(ARMCG::VReg4S(NUM_4 + nIdx)),
                PostPtr(tmp, NEON_VL * sizeof(float)));
        }
    }
    for (std::uint32_t mIdx = 0; mIdx < mLim; ++mIdx) {
        for (std::uint32_t nIdx = 0; nIdx < nLim; ++nIdx) {
            Fadd(ARMCG::VReg4S(GetDstVRegIdx(mIdx, nIdx)),
                 ARMCG::VReg4S(GetDstVRegIdx(mIdx, nIdx)),
                 ARMCG::VReg4S(NUM_4 + nIdx)); // z4-z10 registers are chosen for tmp values
        }
    }
    L(addBiasMNeon);
    Tbz(flags, 8, addBiasMNNeon); // 8-bit correspond to ADD_BIAS_M
    Add(tmp, biaPtr, tmpMMinusMLoopCtr, ARMCG::ShMod::LSL, Math::Log2(sizeof(float)));
    for (std::uint32_t mIdx = 0; mIdx < mLim; ++mIdx) {
        CodeGenerator::Ld1r(ARMCG::VReg4SList(ARMCG::VReg4S(GetSrcVRegIdx(mIdx))), Ptr(tmp));
        CodeGenerator::Add(tmp, tmp, sizeof(float));
    }
    for (std::uint32_t mIdx = 0; mIdx < mLim; ++mIdx) {
        for (std::uint32_t nIdx = 0; nIdx < nLim; ++nIdx) {
            Fadd(ARMCG::VReg4S(GetDstVRegIdx(mIdx, nIdx)),
                ARMCG::VReg4S(GetDstVRegIdx(mIdx, nIdx)),
                ARMCG::VReg4S(GetSrcVRegIdx(mIdx)));
        }
    }
    L(addBiasMNNeon);
    Tbz(flags, 10, initDoneNeon); // 10-bit correspond to ADD_BIAS_MN
    Mul(tmp, tmpMMinusMLoopCtr, biaOffset);
    Add(ARMCG::XReg(GetSrcXRegIdx(0)), biaPtr, tmp,
        ARMCG::ShMod::LSL, Math::Log2(sizeof(float)));
    Add(ARMCG::XReg(GetSrcXRegIdx(0)),
        ARMCG::XReg(GetSrcXRegIdx(0)),
        tmpNMinusNLoopCtr, ARMCG::ShMod::LSL, Math::Log2(sizeof(float)));
    for (std::uint32_t mIdx = 1; mIdx < mLim; ++mIdx) {
        Add(ARMCG::XReg(GetSrcXRegIdx(mIdx)),
            ARMCG::XReg(GetSrcXRegIdx(mIdx - 1)),
            biaOffset, ARMCG::ShMod::LSL, Math::Log2(sizeof(float)));
    }
    for (std::uint32_t mIdx = 0; mIdx < mLim; ++mIdx) {
        for (std::uint32_t nIdx = 0; nIdx < nLim; ++nIdx) {
            // z4-z10 registers are chosen for tmp values, could be any except dst registers
            if (((nResidScalar != NEON_VL)) && (nIdx == (nLim - 1))) {
                for (IntType j = 0; j < nResidScalar; j++) {
                    CodeGenerator::Ld1(ARMCG::VReg4S(NUM_4 + nIdx)[j],
                                       PostPtr(ARMCG::XReg(GetSrcXRegIdx(mIdx)), sizeof(float)));
                }
            } else {
                CodeGenerator::Ld1(ARMCG::VReg4SList(ARMCG::VReg4S(NUM_4 + nIdx)),
                                   PostPtr(ARMCG::XReg(GetSrcXRegIdx(mIdx)),
                                           NEON_VL * sizeof(float)));
            }
            Fadd(ARMCG::VReg4S(GetDstVRegIdx(mIdx, nIdx)),
                 ARMCG::VReg4S(GetDstVRegIdx(mIdx, nIdx)),
                 ARMCG::VReg4S(NUM_4 + nIdx)); // z4-z10 registers are chosen for tmp values
        }
    }
    L(initDoneNeon);
    Tbz(flags, 11, skipPO); // 11-bit correspond to APPLY_POST_OPS
    std::vector<SizeType> dstRegsIDs(mLim * nLim);
    for (std::uint32_t mIdx = 0; mIdx < mLim; ++mIdx) {
        for (std::uint32_t nIdx = 0; nIdx < nLim; ++nIdx) {
            dstRegsIDs[mIdx * nLim + nIdx] = GetDstVRegIdx(mIdx, nIdx);
        }
    }
    this->ApplyPostOps(dstRegsIDs, Element::MatchType<float>());
    L(skipPO);

    // Initialize dst pointers
    Mul(tmp, tmpMMinusMLoopCtr, dstOffset);
    // Row-major format of dst
    Add(ARMCG::XReg(GetDstXRegIdx(0)), dstPtr, tmp,
        ARMCG::ShMod::LSL, Math::Log2(sizeof(float))); // Add stride for M
    Add(ARMCG::XReg(GetDstXRegIdx(0)),
        ARMCG::XReg(GetDstXRegIdx(0)),
        tmpNMinusNLoopCtr, ARMCG::ShMod::LSL, Math::Log2(sizeof(float))); // Add stride for N
    for (std::uint32_t mIdx = 1; mIdx < mLim; ++mIdx) {
        Add(ARMCG::XReg(GetDstXRegIdx(mIdx)),
            ARMCG::XReg(GetDstXRegIdx(mIdx - 1)),
            dstOffset, ARMCG::ShMod::LSL, Math::Log2(sizeof(float)));
    }
    // Store data to dst
    for (std::uint32_t mIdx = 0; mIdx < mLim; ++mIdx) {
        for (std::uint32_t nIdx = 0; nIdx < nLim; ++nIdx) {
            if ((nResidScalar != NEON_VL) && (nIdx == (nLim - 1))) {
                for (IntType j = 0; j < nResidScalar; j++) {
                    CodeGenerator::St1(ARMCG::VReg4S(GetDstVRegIdx(mIdx, nIdx))[j],
                        PostPtr(ARMCG::XReg(GetDstXRegIdx(mIdx)), sizeof(float)));
                }
            } else {
                CodeGenerator::St1(ARMCG::VReg4SList(ARMCG::VReg4S(GetDstVRegIdx(mIdx, nIdx))),
                    PostPtr(ARMCG::XReg(GetDstXRegIdx(mIdx)), NEON_VL * (sizeof(float))));
            }
        }
    }
}

void NeonF32Gen::GenerateMLoopBody(
    std::uint32_t mLim, std::uint32_t nLim, std::uint32_t k)
{
    std::int64_t vectorLength = NEON_VL;
    std::uint32_t nResid = Math::CeilDiv(nLim % (KERNEL_N_SIZE * vectorLength), vectorLength);
    std::uint32_t nResidScalar = nLim % vectorLength;
    if (nResid == 0) {
        nResid = KERNEL_N_SIZE;
    }
    if (nResidScalar == 0) {
        nResidScalar = vectorLength;
    }
    Mov(nLoopCtr, nVal);
    Sub(tmpNMinusNLoopCtr, nVal, nLoopCtr);
    if (nLim > KERNEL_N_SIZE * vectorLength) {
        ARMCG::Label nLoopHead;
        L(nLoopHead);
        {
            GenerateNLoopBody(mLim, KERNEL_N_SIZE, k, vectorLength);   // nLim is original N task size
            Sub(nLoopCtr, nLoopCtr, KERNEL_N_SIZE * vectorLength);
            Cmp(nLoopCtr, KERNEL_N_SIZE * vectorLength);
            Sub(tmpNMinusNLoopCtr, nVal, nLoopCtr);
        }
        B(ARMCG::Cond::GT, nLoopHead);
    }
    GenerateNLoopBody(mLim, nResid, k, nResidScalar);
}

void NeonF32Gen::GenerateNLoopBody(
    std::uint32_t mLim, std::uint32_t nLim, std::uint32_t k, std::uint32_t nResidScalar)
{
    std::uint32_t kResid = k % K_LOOP_UNROLL;
    if (kResid == 0) {
        kResid = K_LOOP_UNROLL;
    }
    Mov(tmp4VL, 0);

    // Fill with zeros by default
    for (std::uint32_t mIdx = 0; mIdx < mLim; ++mIdx) {
        for (std::uint32_t nIdx = 0; nIdx < nLim; ++nIdx) {
            Movi(ARMCG::VReg4S(GetDstVRegIdx(mIdx, nIdx)), 0);
        }
    }

    // Initialize src pointers
    Mul(tmp, tmpMMinusMLoopCtr, srcOffset);
    Add(ARMCG::XReg(GetSrcXRegIdx(0)), srcPtr, tmp,
        ARMCG::ShMod::LSL, Math::Log2(sizeof(float)));
    for (std::uint32_t mIdx = 1; mIdx < mLim; ++mIdx) {
        Add(ARMCG::XReg(GetSrcXRegIdx(mIdx)),
            ARMCG::XReg(GetSrcXRegIdx(mIdx - 1)),
            srcOffset, ARMCG::ShMod::LSL, Math::Log2(sizeof(float)));
    }
    // Initialize wei pointers
    Mul(tmp, tmpNMinusNLoopCtr, kVal);
    Add(ARMCG::XReg(GetWeiXRegIdx(0, nLim)), weiPtr, tmp,
        ARMCG::ShMod::LSL, Math::Log2(sizeof(float)));
    for (std::uint32_t nIdx = 1; nIdx < nLim; ++nIdx) {
        Add(ARMCG::XReg(GetWeiXRegIdx(nIdx, nLim)),
            ARMCG::XReg(GetWeiXRegIdx(nIdx - 1, nLim)),
            weiOffset, ARMCG::ShMod::LSL, Math::Log2(sizeof(float)));
    }
    // Main k loop
    Mov(kLoopCtr, kVal);
    if (k > K_LOOP_UNROLL) {
        ARMCG::Label kLoopHead;
        L(kLoopHead);
        {
            CodeGenerator::Ld1(ARMCG::VReg4SList(ARMCG::VReg4S(GetWeiVRegIdx(0))),
                Ptr(ARMCG::XReg(GetWeiXRegIdx(0, nLim))));
            CodeGenerator::Add(ARMCG::XReg(GetWeiXRegIdx(0, nLim)),
                ARMCG::XReg(GetWeiXRegIdx(0, nLim)),
                static_cast<std::uint32_t>(sizeof(float) * NUM_4));
            if (nLim > 1) {
                CodeGenerator::Ld1(ARMCG::VReg4SList(ARMCG::VReg4S(GetWeiVRegIdx(1))),
                    Ptr(ARMCG::XReg(GetWeiXRegIdx(1, nLim))));
                CodeGenerator::Add(ARMCG::XReg(GetWeiXRegIdx(1, nLim)),
                    ARMCG::XReg(GetWeiXRegIdx(1, nLim)),
                    static_cast<std::uint32_t>(sizeof(float) * NUM_4));
            }
            Sub(kLoopCtr, kLoopCtr, K_LOOP_UNROLL);
            Cmp(kLoopCtr, K_LOOP_UNROLL);
            // Preload src registers
            for (std::uint32_t mIdx = 0; mIdx < mLim; ++mIdx) {
                CodeGenerator::Ld1(ARMCG::VReg4SList(ARMCG::VReg4S(GetSrcVRegIdx(mIdx))),
                    Ptr(ARMCG::XReg(GetSrcXRegIdx(mIdx))));
                Add(ARMCG::XReg(GetSrcXRegIdx(mIdx)),
                    ARMCG::XReg(GetSrcXRegIdx(mIdx)),
                    K_LOOP_UNROLL * sizeof(float));
            }
            GenerateMLAPart(mLim, nLim, K_LOOP_UNROLL, false);
        }
        B(ARMCG::Cond::GT, kLoopHead);
    }

    // Process k tail
    CodeGenerator::Ld1(ARMCG::VReg4SList(ARMCG::VReg4S(GetWeiVRegIdx(0))),
        Ptr(ARMCG::XReg(GetWeiXRegIdx(0, nLim))));
    CodeGenerator::Add(ARMCG::XReg(GetWeiXRegIdx(0, nLim)),
        ARMCG::XReg(GetWeiXRegIdx(0, nLim)),
        static_cast<std::uint32_t>(sizeof(float) * NUM_4));
    if (nLim > 1) {
        CodeGenerator::Ld1(ARMCG::VReg4SList(ARMCG::VReg4S(GetWeiVRegIdx(1))),
            Ptr(ARMCG::XReg(GetWeiXRegIdx(1, nLim))));
        CodeGenerator::Add(ARMCG::XReg(GetWeiXRegIdx(1, nLim)),
            ARMCG::XReg(GetWeiXRegIdx(1, nLim)),
            static_cast<std::uint32_t>(sizeof(float) * NUM_4));
    }
    // Preload src registers
    for (std::uint32_t mIdx = 0; mIdx < mLim; ++mIdx) {
        if (kResid == NUM_4) {
            CodeGenerator::Ld1(ARMCG::VReg4SList(ARMCG::VReg4S(GetSrcVRegIdx(mIdx))),
                Ptr(ARMCG::XReg(GetSrcXRegIdx(mIdx))));
        } else {
            CodeGenerator::Movi(ARMCG::VReg4S(GetSrcVRegIdx(mIdx)), 0);
            for (std::uint32_t kIdx = 0; kIdx < kResid; ++kIdx) {
                CodeGenerator::Ld1(ARMCG::VReg4S(GetSrcVRegIdx(mIdx))[kIdx],
                    PostPtr(ARMCG::XReg(GetSrcXRegIdx(mIdx)), sizeof(float)));
            }
        }
    }
    GenerateMLAPart(mLim, nLim, kResid, true);
    StoreToDst(mLim, nLim, nResidScalar);
}

void NeonF32Gen::ApplyEltwise(const PostOps &postOps, SizeType poIdx, const std::vector<SizeType> &regIDs,
    Element::TypeT regType)
{
    Service::Unused(regType); // this imple operates only on F32 data type
    ActivationFunction actEltwise = ActivationFunction::UNIMPLEMENTED;
    float alphaEltwise = 1.0f;
    float betaEltwise = 0.0f;
    postOps.GetParamsEltwise(poIdx, actEltwise, alphaEltwise, betaEltwise);
    if (actEltwise != ActivationFunction::RELU) {
        throw KuDNN::Service::LogicError {"Unsupported post ops for F1632 impl"};
    } else {
        // betaEltwise isn't used for relu
        auto tmpVRegID = FindFreeVectorReg();
        TakeVectorRegister(tmpVRegID);
        auto tmpRegID = FindFreeScalarReg();
        TakeScalarRegister(tmpRegID);
        std::int32_t alphaBits = ARMCG::Util::Float2Int(alphaEltwise);
        Mov(ARMCG::XReg(tmpRegID), alphaBits);
        Dup(ARMCG::VReg4S(tmpVRegID), ARMCG::WReg(tmpRegID));
        for (auto &&regID : regIDs) {
            auto maskVRegID = FindFreeVectorReg();
            TakeVectorRegister(maskVRegID);
            Fcmge(ARMCG::VReg4S(maskVRegID),
                ARMCG::VReg4S(regID), 0.0);
            auto tmp1VRegID = FindFreeVectorReg();
            TakeVectorRegister(tmp1VRegID);
            Fmul(ARMCG::VReg4S(tmp1VRegID), ARMCG::VReg4S(regID),
                ARMCG::VReg4S(tmpVRegID));
            Bif(ARMCG::VReg16B(regID), ARMCG::VReg16B(tmp1VRegID),
                ARMCG::VReg16B(maskVRegID));
            FreeVectorRegister(tmp1VRegID);
            FreeVectorRegister(maskVRegID);
        }
        FreeVectorRegister(tmpVRegID);
        FreeScalarRegister(tmpRegID);
    }
}

void NeonF32Gen::GenerateMLAPart(
    std::uint32_t mLim, std::uint32_t nLim, std::uint32_t kLim, bool isTail)
{
    Service::Unused(isTail);
    for (std::uint32_t kIdx = 0; kIdx < kLim; ++kIdx) {
        // src (z2-z7) and weights (z0, z1) are preloaded
        for (std::uint32_t nIdx = 0; nIdx < nLim; ++nIdx) {
            for (std::uint32_t mIdx = 0; mIdx < mLim; ++mIdx) {
                Fmla(ARMCG::VReg4S(GetDstVRegIdx(mIdx, nIdx)),
                    ARMCG::VReg4S(GetWeiVRegIdx(nIdx)),
                    ARMCG::VReg4S(GetSrcVRegIdx(mIdx))[kIdx]);
            }
            std::uint32_t tmpN = (nLim == 1) ? 1 : 2;
            std::uint32_t kLoadOffset = kIdx + (nIdx + tmpN) / nLim; // +2 since 2 regs were already preloaded
            if (kLoadOffset != kLim) {
                // load wei reg right after usage
                CodeGenerator::Ld1(ARMCG::VReg4SList(ARMCG::VReg4S(GetWeiVRegIdx(nIdx))),
                    Ptr(ARMCG::XReg(GetWeiXRegIdx(nIdx + tmpN, nLim))));
                CodeGenerator::Add(ARMCG::XReg(GetWeiXRegIdx(nIdx + tmpN, nLim)),
                    ARMCG::XReg(GetWeiXRegIdx(nIdx + tmpN, nLim)),
                    static_cast<std::uint32_t>(sizeof(float) * NUM_4));
            }
        }
    }
}

MatMulImplGenerator<NeonF32Gen>::DefaultImpl::DefaultImpl(
    const GemmInfo &task, IntType nThreads) : MatMulImplBase(task, nThreads)
{
    VERBOSE_PRINT;
    UpdateExecParams();
    Initialize();
}

constexpr std::uint32_t MatMulImplGenerator<NeonF32Gen>::DefaultImpl::GetKernelMSize() noexcept
{
    return NeonF32Gen::KERNEL_M_SIZE;
}

std::uint32_t MatMulImplGenerator<NeonF32Gen>::DefaultImpl::GetKernelNSize() noexcept
{
    return NeonF32Gen::KERNEL_N_SIZE * NeonF32Gen::NEON_VL;
}

IntType MatMulImplGenerator<NeonF32Gen>::DefaultImpl::GetMBlockSize() const noexcept
{
    // should be GetMBlockSize() % GetKernelMSize() == 0
    return 1200; // 1200 is experimentally measured constant to obtain better perf
}

IntType MatMulImplGenerator<NeonF32Gen>::DefaultImpl::GetNBlockSize() const noexcept
{
    // should be GetNBlockSize() % GetKernelNSize() == 0
    return 256; // 256 is experimentally measured constant to obtain better perf
}

IntType MatMulImplGenerator<NeonF32Gen>::DefaultImpl::GetKBlockSize() const noexcept
{
    // should be GetKBlockSize() % K_LOOP_UNROLL == 0
    return 640; // 640 is experimentally measured constant to obtain better perf
}

void MatMulImplGenerator<NeonF32Gen>::DefaultImpl::InitializeSingleCodegen(
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
        CodeGenParams<NeonF32Gen> currTask{currentM, currentN, currentK, false, attributes};
        if (codegens.find(currTask) == codegens.end()) {
            codegens.insert({currTask, std::make_shared<NeonF32Gen>(currentM, currentN, currentK, attributes)});
        }
        codegens[currTask]->Ready();
        matmulCodegenKernels[id] = codegens[currTask]->GetCode<NeonF32GenFuncPtr>();
    }
}

void MatMulImplGenerator<NeonF32Gen>::DefaultImpl::InitializePackingCodeGens(
    IntType m, IntType n, IntType k, IntType mFull, IntType nFull)
{
    Service::Unused(m, k, mFull);
    if (perThreadParams[0].weiLayout2D == Layout::AB) {
        IntType nBlockExecFull;
        IntType nBlockExecResid1;
        IntType nBlockExecResid2;
        InitializeKernelMorN(n, nFull, GetNBlockSize(), nBlockExecFull, nBlockExecResid1, nBlockExecResid2);
        for (IntType i = 0; i < NUM_3; ++i) {
            IntType currentN;
            InitializeCurrentMorN(i, nBlockExecFull, nBlockExecResid1, nBlockExecResid2, currentN);
            if ((currentN <= n) && (currentN != 0)) {
                PackNeonF32GenParams currTask{currentN};
                if (packingCodegens.find(currTask) == packingCodegens.end()) {
                    packingCodegens.insert({currTask, std::shared_ptr<PackBRowDefaultSchemeNeonF32GenF32>{
                        new PackBRowDefaultSchemeNeonF32GenF32{currentN}}});
                }
                packingCodegens[currTask]->Ready();
                vectorPackBRowCodegenKernels[i] = packingCodegens[currTask]->GetCode<PackingNeonF32GenFuncPtr>();
            }
        }
    }
}

void MatMulImplGenerator<NeonF32Gen>::DefaultImpl::InitializeThreadsCountImpl()
{
    IntType threadNumber = InitThreadNum();

    IntType size = GetRequiredMemSize();
    if (size < 21216) { // 21216 is practically measured number
        threadNumber = NUM_1;
    } else if (size < 58858) { // 58858 is practically measured number
        threadNumber = NUM_8;
    } else if (size < 115296) { // 115296 is practically measured number
        threadNumber = NUM_24;
    } else if (size < 355776) { // 355776 is practically measured number
        threadNumber = NUM_40;
    }
    nThreadsExec = std::min(threadNumber, implNThreads);
}
IntType MatMulImplGenerator<NeonF32Gen>::DefaultImpl::GetThreadBufSize(
    IntType m, IntType n, IntType k, MatrixKind mKind) const noexcept
{
    IntType vlw = NUM_4; // NEON_VLW
    Service::Unused(m);
    if (mKind == MatrixKind::SRC) {
        return static_cast<IntType>(0);
    } else {
        return static_cast<IntType>(sizeof(typename NeonF32Gen::WeiType) * std::min<IntType>(GetKBlockSize(), k) *
            Math::CeilDiv(std::min<IntType>(GetNBlockSize(), n), NeonF32Gen::KERNEL_N_SIZE *
            vlw) * NeonF32Gen::KERNEL_N_SIZE * vlw);
    }
}

void MatMulImplGenerator<NeonF32Gen>::DefaultImpl::ReserveImplSpecificParams() {}

std::vector<std::pair<Shape, Shape>> MatMulImplGenerator<NeonF32Gen>::DefaultImpl::SplitWork()
{
    return CalculateWorkOnThread<DefaultImpl>(taskExec.dstInfo.GetDims(), nThreadsExec, dstLayout2D == Layout::BA);
}

void MatMulImplGenerator<NeonF32Gen>::DefaultImpl::PackBCol(
    IntType threadID, IntType n, IntType k, const typename NeonF32Gen::WeiType *bPtr, IntType bOffset,
    typename NeonF32Gen::WeiType *bBuffer) const
{
    // threadID was added to align APIs with other impls
    Service::Unused(threadID);
    constexpr IntType vl = NeonF32Gen::NEON_VL;
    for (IntType nIdx = 0; nIdx < n; nIdx += vl) {
        IntType nSubLim = std::min(n - nIdx, vl);
        for (IntType kIdx = 0; kIdx < k; kIdx += vl) {
            IntType kSubLim = std::min(k - kIdx, vl);
            for (IntType nSubIdx = 0; nSubIdx < nSubLim; ++nSubIdx) {
                for (IntType kSubIdx = 0; kSubIdx < kSubLim; ++kSubIdx) {
                    bBuffer[nIdx * k + (kIdx + kSubIdx) * vl + nSubIdx] =
                        bPtr[kIdx + kSubIdx + bOffset * (nIdx + nSubIdx)];
                }
            }
        }
    }
}

void MatMulImplGenerator<NeonF32Gen>::DefaultImpl::RunSequential2D(
    IntType threadID, const void *src, const void *wei, void *dst, const void *bia,
    std::unique_ptr<PackStrategy> &srcPackStrategy, std::unique_ptr<PackStrategy> &weiPackStrategy,
    float alpha, float beta, const PostOpsDataPtrs &shiftedByBatchPOPtrs) const noexcept(false)
{
    Service::Unused(srcPackStrategy, shiftedByBatchPOPtrs);
    VERBOSE_PRINT;

    IntType tIDbatch;
    IntType tIDm;
    IntType tIDn;
    InitThreadIDs(threadID, tIDbatch, tIDm, tIDn);
    WeiType* weiBuffer = static_cast<WeiType *>(weiPackStrategy->GetPtr(tIDbatch * threadsN  + tIDn, tIDm, threadsM));

    for (IntType mBlock = 0; mBlock < perThreadParams[threadID].m; mBlock += GetMBlockSize()) {
        auto mBlockExec = std::min(GetMBlockSize(),  perThreadParams[threadID].m - mBlock);
        IntType mIdx = (mBlockExec == GetMBlockSize()) ? 0 : 1;
        IntType mIdxKernel = mIdx * (perThreadParams[threadID].isResidByM + 1);
        for (IntType kBlock = 0; kBlock < perThreadParams[threadID].k; kBlock += GetKBlockSize()) {
            auto kBlockExec = std::min(GetKBlockSize(), perThreadParams[threadID].k - kBlock);
            IntType kIdxKernel = (kBlockExec == GetKBlockSize()) ? 0 : 1;
            bool lastKIter = (perThreadParams[threadID].k - kBlock) <= GetKBlockSize();
            MatMulDstFlags biaFlagsExec = SetBiaFlagsExec(biaFlags, beta, kBlock, lastKIter);
            for (IntType nBlock = 0; nBlock < perThreadParams[threadID].n; nBlock += GetNBlockSize()) {
                auto nBlockExec = std::min(GetNBlockSize(), perThreadParams[threadID].n - nBlock);
                IntType nIdx = (nBlockExec == GetNBlockSize()) ? 0 : 1;
                IntType nIdxKernel = nIdx * (perThreadParams[threadID].isResidByN + 1);
                typename NeonF32Gen::DstType *dstExec = static_cast<typename NeonF32Gen::DstType *>(dst) + nBlock +
                    mBlock * perThreadParams[threadID].dstOffset2D;
                const typename NeonF32Gen::BiaType *biaExec = static_cast<const typename NeonF32Gen::BiaType *>(bia) +
                    nBlock * biaNOffset + mBlock * perThreadParams[threadID].biaOffset2D;
                if (perThreadParams[threadID].weiLayout2D == Layout::AB) {
                    const typename NeonF32Gen::WeiType *weiExec =
                        static_cast<const typename NeonF32Gen::WeiType *>(wei) +
                        nBlock + kBlock * perThreadParams[threadID].weiOffset2D;
                    auto packB = vectorPackBRowCodegenKernels[nIdxKernel];
                    packB(nBlockExec, kBlockExec, weiExec,
                        perThreadParams[threadID].weiOffset2D, weiBuffer);
                } else {
                    const typename NeonF32Gen::WeiType *weiExec =
                        static_cast<const typename NeonF32Gen::WeiType *>(wei) +
                        nBlock * perThreadParams[threadID].weiOffset2D + kBlock;
                    PackBCol(threadID, nBlockExec, kBlockExec, weiExec,
                        perThreadParams[threadID].weiOffset2D, weiBuffer);
                }
                const typename NeonF32Gen::SrcType *srcExec = static_cast<const typename NeonF32Gen::SrcType *>(src) +
                    kBlock + mBlock * perThreadParams[threadID].srcOffset2D;

                auto kernel = matmulCodegenKernels[kIdxKernel * NUM_9 +
                    nIdxKernel * NUM_3 + mIdxKernel];
                kernel(srcExec, weiBuffer, dstExec, biaExec,
                    biaFlagsExec, alpha, beta, mBlockExec, nBlockExec, kBlockExec,
                    perThreadParams[threadID].srcOffset2D, kBlockExec * NeonF32Gen::NEON_VL,
                    perThreadParams[threadID].dstOffset2D, perThreadParams[threadID].biaOffset2D);
            }
        }
    }
}

void MatMulImplGenerator<NeonF32Gen>::DefaultImpl::UpdateExecParams() noexcept
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
            bool checkDesiredWeiLayout1 = (!IsLayoutAlikeAB(desiredWeiLayout) && !IsLayoutAlikeBA(desiredWeiLayout));
            if (checkDesiredWeiLayout1) {
                desiredWeiLayout = GetPlaneLayoutAB(desiredWeiLayout);
            }
        } else if (IsLayoutAlikeBA(desiredDstLayout)) {
            if (!IsLayoutAlikeBA(desiredWeiLayout)) {
                desiredWeiLayout = GetPlaneLayoutBA(desiredWeiLayout);
            }
            bool checkDesiredSrcLayout1 = (!IsLayoutAlikeAB(desiredSrcLayout) && !IsLayoutAlikeBA(desiredSrcLayout));
            if (checkDesiredSrcLayout1) {
                desiredSrcLayout = GetPlaneLayoutBA(desiredSrcLayout);
            }
        } else {
            if (IsLayoutAlikeAB(desiredSrcLayout)) {
                desiredDstLayout = GetPlaneLayoutAB(desiredDstLayout);
                bool checkDesiredWeiLayout2 =
                    (!IsLayoutAlikeAB(desiredWeiLayout) && !IsLayoutAlikeBA(desiredWeiLayout));
                if (checkDesiredWeiLayout2) {
                    desiredWeiLayout = GetPlaneLayoutAB(desiredWeiLayout);
                }
            } else if (IsLayoutAlikeBA(desiredWeiLayout)) {
                desiredDstLayout = GetPlaneLayoutBA(desiredDstLayout);
                bool checkDesiredSrcLayout2 =
                    (!IsLayoutAlikeAB(desiredSrcLayout) && !IsLayoutAlikeBA(desiredSrcLayout));
                if (checkDesiredSrcLayout2) {
                    desiredSrcLayout = GetPlaneLayoutBA(desiredSrcLayout);
                }
            } else {
                desiredDstLayout = GetPlaneLayoutAB(desiredDstLayout);
                desiredSrcLayout = GetPlaneLayoutAB(desiredSrcLayout);
                bool checkDesiredWeiLayout3 =
                    (!IsLayoutAlikeAB(desiredWeiLayout) && !IsLayoutAlikeBA(desiredWeiLayout));
                if (checkDesiredWeiLayout3) {
                    desiredWeiLayout = GetPlaneLayoutAB(desiredWeiLayout);
                }
            }
        }
    }
    if (implTask.biaInfo.GetTotalTensorSize() != 0) {
        desiredBiaLayout = desiredDstLayout;
    }
    InitializeTaskExec(desiredSrcLayout, desiredWeiLayout, desiredDstLayout, desiredBiaLayout,
        NeonF32Gen::GetSrcType(), NeonF32Gen::GetWeiType(), NeonF32Gen::GetDstType(),
        (implTask.biaInfo.GetTotalTensorSize() != 0) ? NeonF32Gen::GetBiaType() :
            static_cast<Element::TypeT>(implTask.biaInfo.GetType()));
}

MatMulDstFlags MatMulImplGenerator<NeonF32Gen>::DefaultImpl::SetBiaFlagsExec(
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

std::unordered_map<PackNeonF32GenParams, std::shared_ptr<PackBRowDefaultSchemeNeonF32GenF32>>
    MatMulImplGenerator<NeonF32Gen>::DefaultImpl::packingCodegens;

} // Detail
} // KuDNN
