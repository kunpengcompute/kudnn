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

#include "kernels/matmul_jit/neon/kudnn_default_impl_su8su8s32s32_mmla.hpp"
#include "kernels/matmul_jit/kudnn_gemm_threading_helpers.hpp"

#include <arm_sve.h>

namespace KuDNN {
namespace Detail {

template <typename SrcType, typename WeiType, typename DstType, typename BiaType>
using NeonSU8MmlaGenBase = MatMulKernelCodeGenBase<ARCH::NEON, SrcType, WeiType, DstType, BiaType>;

template <typename SrcType, typename WeiType, typename DstType, typename BiaType>
NeonSU8MmlaGen<SrcType, WeiType, DstType, BiaType>::MatMulKernelDefaultCodeGen(
    IntType m, IntType n, IntType k,
    bool wereSrcWeiSwappedArg, const Attributes &attributesArg)
    : NeonSU8MmlaGenBase<SrcType, WeiType, DstType, BiaType>(attributesArg),
    wereSrcWeiSwapped(wereSrcWeiSwappedArg), vectorLengthByte(svcntb())
{
    if ((m > 0) && (n > 0) && (k > 0)) {
        if (this->attributes.GetPostOps().Len() > 0) {
            // set reserved dst registers
            for (std::uint32_t mIdx = 0; mIdx < Math::CeilDiv(KERNEL_M_SIZE, NUM_2); ++mIdx) {
                for (std::uint32_t nIdx = 0; nIdx < KERNEL_N_SIZE; ++nIdx) {
                    KUDNN_JIT_Generator_With_Attributes::TakeVectorRegister(
                        GetTmpDstVRegIdx(mIdx, nIdx * NUM_2));
                    KUDNN_JIT_Generator_With_Attributes::TakeVectorRegister(
                        GetTmpDstVRegIdx(mIdx, nIdx * NUM_2 + NUM_1));
                }
            }

            // set reserved scalar registers
            for (std::uint32_t idx = 0; idx < NUM_24; ++idx) {
                KUDNN_JIT_Generator_With_Attributes::TakeScalarRegister(idx);
            }
        }
        Ldp(srcOffset, weiOffset, PostPtr(sp, NUM_16));
        Ldp(dstOffset, biaOffset, PostPtr(sp, NUM_0));
        Sub(sp, sp, NUM_16);
        Preamble();

        Sub(sp, sp, NUM_16);
        Str(alpha, ARMCG::AdrUimm(Ptr(sp)));
        Sub(sp, sp, NUM_16);
        Str(beta, ARMCG::AdrUimm(Ptr(sp)));

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

template <typename SrcType, typename WeiType, typename DstType, typename BiaType>
void NeonSU8MmlaGen<SrcType, WeiType, DstType, BiaType>::CvtAndAdd(
    bool applyConvert, int tmpRegIdx, int mRegIdx, int nRegIdx)
{
    ARMCG::Label noConv;
    if (applyConvert) {
        Tbz(flags, 12, noConv); // 12-bit correspond to FIRST_K_ITER
        Scvtf(ARMCG::VReg4S(GetTmpVRegIdx(tmpRegIdx)),
            ARMCG::VReg4S(GetTmpVRegIdx(tmpRegIdx)));
        L(noConv);
        Fadd(ARMCG::VReg4S(GetStDstVRegIdx(mRegIdx, nRegIdx)),
            ARMCG::VReg4S(GetStDstVRegIdx(mRegIdx, nRegIdx)),
            ARMCG::VReg4S(GetTmpVRegIdx(tmpRegIdx)));
    } else {
        Add(ARMCG::VReg4S(GetStDstVRegIdx(mRegIdx, nRegIdx)),
            ARMCG::VReg4S(GetStDstVRegIdx(mRegIdx, nRegIdx)),
            ARMCG::VReg4S(GetTmpVRegIdx(tmpRegIdx)));
    }
}

template <typename SrcType, typename WeiType, typename DstType, typename BiaType>
void NeonSU8MmlaGen<SrcType, WeiType, DstType, BiaType>::CvtAndStore(bool applyConvert,
    std::uint32_t mLim, std::uint32_t nLim, std::uint32_t nResidScalar)
{
    nLim /= NUM_2;
    ARMCG::Label su8NEONloadC;
    ARMCG::Label su8NEONmullC;
    ARMCG::Label su8NEONloadBias1;
    ARMCG::Label su8NEONloadBiasM;
    ARMCG::Label su8NEONloadBiasN;
    ARMCG::Label su8NEONloadBiasMN;
    ARMCG::Label su8NEONaddBias1;
    ARMCG::Label su8NEONaddBiasM;
    ARMCG::Label su8NEONaddBiasN;
    ARMCG::Label su8NEONaddBiasMN;
    ARMCG::Label su8NEONaddinitDone;
    ARMCG::Label su8NEONfillZeros;
    ARMCG::Label skipPO;

    if (!(std::is_same_v<SrcType, WeiType>) && !(std::is_same_v<SrcType, std::int8_t> == wereSrcWeiSwapped)) {
        // dst in transposed format
        for (std::uint32_t mIdx = 0; mIdx < Math::CeilDiv(mLim, NUM_2); ++mIdx) {
            for (std::uint32_t nIdx = 0; nIdx < nLim; ++nIdx) {
                Uzp1(ARMCG::VReg4S(GetTmpDstVRegIdx(mIdx, nIdx * NUM_2)),
                    ARMCG::VReg4S(GetDstVRegIdx(mIdx, nIdx * NUM_2)),
                    ARMCG::VReg4S(GetDstVRegIdx(mIdx, nIdx * NUM_2 + NUM_1)));
                Uzp2(ARMCG::VReg4S(GetTmpDstVRegIdx(mIdx, nIdx * NUM_2 + NUM_1)),
                    ARMCG::VReg4S(GetDstVRegIdx(mIdx, nIdx * NUM_2)),
                    ARMCG::VReg4S(GetDstVRegIdx(mIdx, nIdx * NUM_2 + NUM_1)));
            }
        }
    } else {
        for (std::uint32_t mIdx = 0; mIdx < Math::CeilDiv(mLim, NUM_2); ++mIdx) {
            for (std::uint32_t nIdx = 0; nIdx < nLim; ++nIdx) {
                Uzp1(ARMCG::VReg2D(GetTmpDstVRegIdx(mIdx, nIdx * NUM_2)),
                    ARMCG::VReg2D(GetDstVRegIdx(mIdx, nIdx * NUM_2)),
                    ARMCG::VReg2D(GetDstVRegIdx(mIdx, nIdx * NUM_2 + NUM_1)));
                Uzp2(ARMCG::VReg2D(GetTmpDstVRegIdx(mIdx, nIdx * NUM_2 + NUM_1)),
                    ARMCG::VReg2D(GetDstVRegIdx(mIdx, nIdx * NUM_2)),
                    ARMCG::VReg2D(GetDstVRegIdx(mIdx, nIdx * NUM_2 + NUM_1)));
            }
        }
    }
    // now we've unzipped dst regs and can work with them normally
    if (applyConvert) {
        for (std::uint32_t mIdx = 0; mIdx < mLim; ++mIdx) {
            for (std::uint32_t nIdx = 0; nIdx < nLim; ++nIdx) {
                Scvtf(ARMCG::VReg4S(GetStDstVRegIdx(mIdx, nIdx)),
                    ARMCG::VReg4S(GetStDstVRegIdx(mIdx, nIdx)));
            }
        }
        // Load alpha
        Add(tmp, sp, NUM_16);
        ARMCG::CodeGenerator::Ld1r(ARMCG::VReg4SList{ARMCG::VReg4S(GetTmpVRegIdx(0))}, ARMCG::AdrNoOfs(tmp));
        Mov(tmpW, ARMCG::VReg4S(GetTmpVRegIdx(0))[0]);
        Mov(ARMCG::WReg(GetTmpXRegIdx(0)), 0x3f800000); // 0x3f800000 is a bit representation of 1.0f
        Cmp(tmpW, ARMCG::WReg(GetTmpXRegIdx(0)));
        Beq(su8NEONfillZeros);
        for (std::uint32_t mIdx = 0; mIdx < mLim; ++mIdx) {
            for (std::uint32_t nIdx = 0; nIdx < nLim; ++nIdx) {
                Fmul(
                    ARMCG::VReg4S(GetStDstVRegIdx(mIdx, nIdx)),
                    ARMCG::VReg4S(GetStDstVRegIdx(mIdx, nIdx)),
                    ARMCG::VReg4S(GetTmpVRegIdx(0))
                );
            }
        }
    }
    L(su8NEONfillZeros);
    Tbz(flags, 0, su8NEONloadBias1); // 0-bit correspond to FILL_C_WITH_ZEROS
    L(su8NEONloadBias1);
    Tbz(flags, 3, su8NEONloadBiasN); // 3-bit correspond to LOAD_BIAS_1
    ARMCG::CodeGenerator::Ld1r(ARMCG::VReg4SList{ARMCG::VReg4S(GetTmpVRegIdx(NUM_7))}, ARMCG::AdrNoOfs(biaPtr));
    for (std::uint32_t mIdx = 0; mIdx < mLim; ++mIdx) {
        for (std::uint32_t nIdx = 0; nIdx < nLim; ++nIdx) {
            Mov(ARMCG::VReg16B(GetTmpVRegIdx(nIdx)), ARMCG::VReg16B(GetTmpVRegIdx(NUM_7)));
            CvtAndAdd(applyConvert, nIdx, mIdx, nIdx);
        }
    }
    L(su8NEONloadBiasN);
    Tbz(flags, 5, su8NEONloadBiasM); // 5-bit correspond to LOAD_BIAS_N
    for (std::uint32_t mIdx = 0; mIdx < mLim; ++mIdx) {
        Add(tmp, biaPtr, tmpNMinusNLoopCtr, ARMCG::ShMod::LSL, Math::Log2(sizeof(BiaType)));
        for (std::uint32_t nIdx = 0; nIdx < nLim; ++nIdx) {
            if (((nResidScalar != NEON_VLW)) && (nIdx == (nLim - 1))) {
                Movi(ARMCG::VReg4S(GetTmpVRegIdx(nIdx)), 0);
                for (IntType j = 0; j < nResidScalar; j++) {
                    ARMCG::CodeGenerator::Ld1(ARMCG::VReg4S(GetTmpVRegIdx(nIdx))[j],
                        ARMCG::PostPtr(tmp, sizeof(DstType)));
                }
            } else {
                ARMCG::CodeGenerator::Ld1(ARMCG::VReg4SList{ARMCG::VReg4S(GetTmpVRegIdx(nIdx))},
                    ARMCG::PostPtr(tmp, NUM_16));
            }
            CvtAndAdd(applyConvert, nIdx, mIdx, nIdx);
        }
    }
    L(su8NEONloadBiasM);
    Tbz(flags, 4, su8NEONloadBiasMN); // 4-bit correspond to LOAD_BIAS_M
    Add(tmp, biaPtr, tmpMMinusMLoopCtr, ARMCG::ShMod::LSL, Math::Log2(sizeof(BiaType)));
    for (std::uint32_t mIdx = 0; mIdx < mLim; ++mIdx) {
        ARMCG::CodeGenerator::Ld1r(ARMCG::VReg4SList{ARMCG::VReg4S(GetTmpVRegIdx(mIdx))},
            ARMCG::AdrNoOfs(tmp));
        Add(tmp, tmp, sizeof(BiaType));
        for (std::uint32_t nIdx = 0; nIdx < nLim; ++nIdx) {
            Mov(ARMCG::VReg16B(GetTmpVRegIdx(mLim)), ARMCG::VReg16B(GetTmpVRegIdx(mIdx)));
            CvtAndAdd(applyConvert, mLim, mIdx, nIdx);
        }
    }
    L(su8NEONloadBiasMN);
    Tbz(flags, 6, su8NEONloadC); // 6-bit correspond to LOAD_BIAS_MN
    for (std::uint32_t mIdx = 0; mIdx < mLim; ++mIdx) {
        for (std::uint32_t nIdx = 0; nIdx < nLim; ++nIdx) {
            Add(ARMCG::XReg(GetTmpXRegIdx(mIdx)), tmpMMinusMLoopCtr, mIdx);
            Mul(ARMCG::XReg(GetTmpXRegIdx(mIdx)), ARMCG::XReg(GetTmpXRegIdx(mIdx)), biaOffset);
            Add(ARMCG::XReg(GetTmpXRegIdx(mIdx)), biaPtr, ARMCG::XReg(GetTmpXRegIdx(mIdx)),
                ARMCG::ShMod::LSL, Math::Log2(sizeof(BiaType)));
            Add(ARMCG::XReg(GetTmpXRegIdx(mIdx)), ARMCG::XReg(GetTmpXRegIdx(mIdx)), tmpNMinusNLoopCtr,
                ARMCG::ShMod::LSL, Math::Log2(sizeof(BiaType)));
            Add(ARMCG::XReg(GetTmpXRegIdx(mIdx)), ARMCG::XReg(GetTmpXRegIdx(mIdx)), nIdx * NUM_16);
            if (((nResidScalar != NEON_VLW)) && (nIdx == (nLim - 1))) {
                Movi(ARMCG::VReg4S(GetTmpVRegIdx(mIdx)), 0);
                for (IntType j = 0; j < nResidScalar; j++) {
                    ARMCG::CodeGenerator::Ld1(ARMCG::VReg4S(GetTmpVRegIdx(mIdx))[j],
                        ARMCG::PostPtr(ARMCG::XReg(GetTmpXRegIdx(mIdx)), sizeof(DstType)));
                }
            } else {
                ARMCG::CodeGenerator::Ld1(ARMCG::VReg4SList{ARMCG::VReg4S(GetTmpVRegIdx(mIdx))},
                    ARMCG::AdrNoOfs(ARMCG::XReg(GetTmpXRegIdx(mIdx))));
            }
            CvtAndAdd(applyConvert, mIdx, mIdx, nIdx);
        }
    }
    L(su8NEONloadC);
    Tbz(flags, 1, su8NEONmullC); // 1-bit correspond to LOAD_C
    for (std::uint32_t mIdx = 0; mIdx < mLim; ++mIdx) {
        for (std::uint32_t nIdx = 0; nIdx < nLim; ++nIdx) {
            Add(ARMCG::XReg(GetTmpXRegIdx(mIdx)), ARMCG::XReg(GetDstXRegIdx(mIdx)), nIdx * NUM_16);
            if (((nResidScalar != NEON_VLW)) && (nIdx == (nLim - 1))) {
                Movi(ARMCG::VReg4S(GetTmpVRegIdx(mIdx)), 0);
                for (IntType j = 0; j < nResidScalar; j++) {
                    ARMCG::CodeGenerator::Ld1(ARMCG::VReg4S(GetTmpVRegIdx(mIdx))[j],
                        ARMCG::PostPtr(ARMCG::XReg(GetTmpXRegIdx(mIdx)), sizeof(DstType)));
                }
            } else {
                ARMCG::CodeGenerator::Ld1(ARMCG::VReg4SList{ARMCG::VReg4S(GetTmpVRegIdx(mIdx))},
                    ARMCG::AdrNoOfs(ARMCG::XReg(GetTmpXRegIdx(mIdx))));
            }
            CvtAndAdd(applyConvert, mIdx, mIdx, nIdx);
        }
    }
    L(su8NEONmullC);
    Tbz(flags, 2, su8NEONaddBias1); // 2-bit correspond to MULL_C
    ARMCG::CodeGenerator::Ld1r(ARMCG::VReg4SList{ARMCG::VReg4S(GetTmpVRegIdx(NUM_7))}, ARMCG::AdrNoOfs(sp));
    for (std::uint32_t mIdx = 0; mIdx < mLim; ++mIdx) {
        for (std::uint32_t nIdx = 0; nIdx < nLim; ++nIdx) {
            Add(ARMCG::XReg(GetTmpXRegIdx(mIdx)), ARMCG::XReg(GetDstXRegIdx(mIdx)), nIdx * NUM_16);
            if (((nResidScalar != NEON_VLW)) && (nIdx == (nLim - 1))) {
                Movi(ARMCG::VReg4S(GetTmpVRegIdx(mIdx)), 0);
                for (IntType j = 0; j < nResidScalar; j++) {
                    ARMCG::CodeGenerator::Ld1(ARMCG::VReg4S(GetTmpVRegIdx(mIdx))[j],
                        ARMCG::PostPtr(ARMCG::XReg(GetTmpXRegIdx(mIdx)), sizeof(DstType)));
                }
            } else {
                ARMCG::CodeGenerator::Ld1(ARMCG::VReg4SList{ARMCG::VReg4S(GetTmpVRegIdx(mIdx))},
                    ARMCG::AdrNoOfs(ARMCG::XReg(GetTmpXRegIdx(mIdx))));
            }
            Scvtf(ARMCG::VReg4S(GetTmpVRegIdx(mIdx)),
                ARMCG::VReg4S(GetTmpVRegIdx(mIdx)));
            Fmul(
                ARMCG::VReg4S(GetTmpVRegIdx(mIdx)),
                ARMCG::VReg4S(GetTmpVRegIdx(mIdx)),
                ARMCG::VReg4S(GetTmpVRegIdx(NUM_7))
            );
            Fadd(ARMCG::VReg4S(GetStDstVRegIdx(mIdx, nIdx)),
                ARMCG::VReg4S(GetStDstVRegIdx(mIdx, nIdx)),
                ARMCG::VReg4S(GetTmpVRegIdx(mIdx)));
        }
    }
    L(su8NEONaddBias1);
    Tbz(flags, 7, su8NEONaddBiasN); // 7-bit correspond to ADD_BIAS_1
    ARMCG::CodeGenerator::Ld1r(ARMCG::VReg4SList{ARMCG::VReg4S(GetTmpVRegIdx(NUM_7))}, ARMCG::AdrNoOfs(biaPtr));
    for (std::uint32_t mIdx = 0; mIdx < mLim; ++mIdx) {
        for (std::uint32_t nIdx = 0; nIdx < nLim; ++nIdx) {
            Mov(ARMCG::VReg16B(GetTmpVRegIdx(nIdx)), ARMCG::VReg16B(GetTmpVRegIdx(NUM_7)));
            CvtAndAdd(applyConvert, nIdx, mIdx, nIdx);
        }
    }
    L(su8NEONaddBiasN);
    Tbz(flags, 9, su8NEONaddBiasM); // 9-bit correspond to ADD_BIAS_N
    for (std::uint32_t mIdx = 0; mIdx < mLim; ++mIdx) {
        Add(tmp, biaPtr, tmpNMinusNLoopCtr, ARMCG::ShMod::LSL, Math::Log2(sizeof(BiaType)));
        for (std::uint32_t nIdx = 0; nIdx < nLim; ++nIdx) {
            if (((nResidScalar != NEON_VLW)) && (nIdx == (nLim - 1))) {
                Movi(ARMCG::VReg4S(GetTmpVRegIdx(nIdx)), 0);
                for (IntType j = 0; j < nResidScalar; j++) {
                    ARMCG::CodeGenerator::Ld1(ARMCG::VReg4S(GetTmpVRegIdx(nIdx))[j],
                        ARMCG::PostPtr(tmp, sizeof(DstType)));
                }
            } else {
                ARMCG::CodeGenerator::Ld1(ARMCG::VReg4SList{ARMCG::VReg4S(GetTmpVRegIdx(nIdx))},
                    ARMCG::PostPtr(tmp, NUM_16));
            }
            CvtAndAdd(applyConvert, nIdx, mIdx, nIdx);
        }
    }
    L(su8NEONaddBiasM);
    Tbz(flags, 8, su8NEONaddBiasMN); // 8-bit correspond to ADD_BIAS_M
    Add(tmp, biaPtr, tmpMMinusMLoopCtr, ARMCG::ShMod::LSL, Math::Log2(sizeof(BiaType)));
    for (std::uint32_t mIdx = 0; mIdx < mLim; ++mIdx) {
        ARMCG::CodeGenerator::Ld1r(ARMCG::VReg4SList{ARMCG::VReg4S(GetTmpVRegIdx(mIdx))}, ARMCG::AdrNoOfs(tmp));
        Add(tmp, tmp, sizeof(BiaType));
        for (std::uint32_t nIdx = 0; nIdx < nLim; ++nIdx) {
            Mov(ARMCG::VReg16B(GetTmpVRegIdx(mLim)), ARMCG::VReg16B(GetTmpVRegIdx(mIdx)));
            CvtAndAdd(applyConvert, mLim, mIdx, nIdx);
        }
    }
    L(su8NEONaddBiasMN);
    Tbz(flags, 10, su8NEONaddinitDone); // 10-bit correspond to ADD_BIAS_MN
    for (std::uint32_t mIdx = 0; mIdx < mLim; ++mIdx) {
        for (std::uint32_t nIdx = 0; nIdx < nLim; ++nIdx) {
            Add(ARMCG::XReg(GetTmpXRegIdx(mIdx)), tmpMMinusMLoopCtr, mIdx);
            Mul(ARMCG::XReg(GetTmpXRegIdx(mIdx)), ARMCG::XReg(GetTmpXRegIdx(mIdx)), biaOffset);
            Add(ARMCG::XReg(GetTmpXRegIdx(mIdx)), biaPtr, ARMCG::XReg(GetTmpXRegIdx(mIdx)),
                ARMCG::ShMod::LSL, Math::Log2(sizeof(BiaType)));
            Add(ARMCG::XReg(GetTmpXRegIdx(mIdx)), ARMCG::XReg(GetTmpXRegIdx(mIdx)), tmpNMinusNLoopCtr,
                ARMCG::ShMod::LSL, Math::Log2(sizeof(BiaType)));
            Add(ARMCG::XReg(GetTmpXRegIdx(mIdx)), ARMCG::XReg(GetTmpXRegIdx(mIdx)), nIdx * NUM_16);
            if (((nResidScalar != NEON_VLW)) && (nIdx == (nLim - 1))) {
                Movi(ARMCG::VReg4S(GetTmpVRegIdx(mIdx)), 0);
                for (IntType j = 0; j < nResidScalar; j++) {
                    ARMCG::CodeGenerator::Ld1(ARMCG::VReg4S(GetTmpVRegIdx(mIdx))[j],
                        ARMCG::PostPtr(ARMCG::XReg(GetTmpXRegIdx(mIdx)), sizeof(DstType)));
                }
            } else {
                ARMCG::CodeGenerator::Ld1(ARMCG::VReg4SList{ARMCG::VReg4S(GetTmpVRegIdx(mIdx))},
                    ARMCG::AdrNoOfs(ARMCG::XReg(GetTmpXRegIdx(mIdx))));
            }
            CvtAndAdd(applyConvert, mIdx, mIdx, nIdx);
        }
    }
    L(su8NEONaddinitDone);
    Tbz(flags, 11, skipPO); // 11-bit correspond to APPLY_POST_OPS
    std::vector<SizeType> dstRegsIDs(mLim * nLim);
    for (std::uint32_t mIdx = 0; mIdx < mLim; ++mIdx) {
        for (std::uint32_t nIdx = 0; nIdx < nLim; ++nIdx) {
            dstRegsIDs[mIdx * nLim + nIdx] = GetStDstVRegIdx(mIdx, nIdx);
        }
    }
    if (applyConvert) {
        this->ApplyPostOps(dstRegsIDs, Element::MatchType<float>());
    } else {
        this->ApplyPostOps(dstRegsIDs, Element::MatchType<DstType>());
    }
    // Apply convert on last K iter
    if (applyConvert) {
        for (std::uint32_t mIdx = 0; mIdx < mLim; ++mIdx) {
            for (std::uint32_t nIdx = 0; nIdx < nLim; ++nIdx) {
                Frinti(ARMCG::VReg4S(GetStDstVRegIdx(mIdx, nIdx)),
                    ARMCG::VReg4S(GetStDstVRegIdx(mIdx, nIdx)));
                Fcvtzs(ARMCG::VReg4S(GetStDstVRegIdx(mIdx, nIdx)),
                    ARMCG::VReg4S(GetStDstVRegIdx(mIdx, nIdx)));
            }
        }
    }
    L(skipPO);
    for (std::uint32_t mIdx = 0; mIdx < mLim; ++mIdx) {
        for (std::uint32_t nIdx = 0; nIdx < nLim; ++nIdx) {
            if (((nResidScalar != NEON_VLW)) && (nIdx == (nLim - 1))) {
                Movi(ARMCG::VReg4S(GetTmpVRegIdx(mIdx)), 0);
                for (IntType j = 0; j < nResidScalar; j++) {
                    ARMCG::CodeGenerator::St1(ARMCG::VReg4S(GetStDstVRegIdx(mIdx, nIdx))[j],
                        ARMCG::PostPtr(ARMCG::XReg(GetDstXRegIdx(mIdx)), sizeof(DstType)));
                }
            } else {
                ARMCG::CodeGenerator::St1(ARMCG::VReg4SList{ARMCG::VReg4S(GetStDstVRegIdx(mIdx, nIdx))},
                    ARMCG::PostPtr(ARMCG::XReg(GetDstXRegIdx(mIdx)), NUM_16));
            }
        }
    }
}

template <typename SrcType, typename WeiType, typename DstType, typename BiaType>
void NeonSU8MmlaGen<SrcType, WeiType, DstType, BiaType>::StoreToDst(std::uint32_t mLim, std::uint32_t nLim,
    std::uint32_t nResidScalar)
{
    ARMCG::Label applyf32Conv;
    ARMCG::Label nof32Conv;
    ARMCG::Label ret;

    Mov(ARMCG::WReg(GetTmpXRegIdx(0)), 0x3f800000); // 0x3f800000 is a bit representation of 1.0f

    // alpha
    // 1.0f no convert
    Ldr(tmpW, Ptr(sp, static_cast<std::int32_t>(NUM_16)));
    Cmp(tmpW, ARMCG::WReg(GetTmpXRegIdx(0)));
    Bne(applyf32Conv);

    // beta
    // 1.0f, 0.0f no convert to f32
    Ldr(tmp, Ptr(sp));
    Cmp(tmp, 0);
    Beq(nof32Conv);

    Cmp(tmp, ARMCG::XReg(GetTmpXRegIdx(0)));
    Beq(nof32Conv);

    L(applyf32Conv);
    CvtAndStore(true, mLim, nLim, nResidScalar);
    B(ret);

    L(nof32Conv);
    CvtAndStore(false, mLim, nLim, nResidScalar);
    L(ret);
}

template <typename SrcType, typename WeiType, typename DstType, typename BiaType>
void NeonSU8MmlaGen<SrcType, WeiType, DstType, BiaType>::GenerateMLoopBody(std::uint32_t mLim,
    std::uint32_t nLim, std::uint32_t k)
{
    std::uint32_t vectorLength = NEON_VLW; // 4xs32 elements in NEON vector
    std::uint32_t nResid = Math::CeilDiv(nLim % (KERNEL_N_SIZE * vectorLength), vectorLength) * NUM_2;
    if (nResid == 0) {
        nResid = KERNEL_N_SIZE * NUM_2;
    }
    std::uint32_t nResidScalar = nLim % vectorLength;
    if (nResidScalar == 0) {
        nResidScalar = vectorLength;
    }
    Mov(nLoopCtr, nVal);
    Sub(tmpNMinusNLoopCtr, nVal, nLoopCtr);
    if (nLim > (KERNEL_N_SIZE * vectorLength)) {
        ARMCG::Label nLoopHead;
        L(nLoopHead);
        {
            GenerateNLoopBody(mLim, KERNEL_N_SIZE * NUM_2, k, vectorLength);
            Sub(nLoopCtr, nLoopCtr, KERNEL_N_SIZE * vectorLength);
            Cmp(nLoopCtr, KERNEL_N_SIZE * vectorLength);
            Sub(tmpNMinusNLoopCtr, nVal, nLoopCtr);
        }
        B(ARMCG::Cond::GT, nLoopHead);
    }
    GenerateNLoopBody(mLim, nResid, k, nResidScalar);
}

template <typename SrcType, typename WeiType, typename DstType, typename BiaType>
void NeonSU8MmlaGen<SrcType, WeiType, DstType, BiaType>::GenerateNLoopBody(std::uint32_t mLim,
    std::uint32_t nLim, std::uint32_t k, std::uint32_t nResidScalar)
{
    for (std::uint32_t mIdx = 0; mIdx < Math::CeilDiv(mLim, NUM_2); ++mIdx) {
        for (std::uint32_t nIdx = 0; nIdx < nLim; ++nIdx) {
            Movi(ARMCG::VReg16B(GetDstVRegIdx(mIdx, nIdx)), 0);
        }
    }
    // Initialize wei pointer
    Mul(tmp, tmpNMinusNLoopCtr, weiOffset);
    Add(ARMCG::XReg(GetWeiXRegIdx()), weiPtr, tmp);

    // Initialize src pointers
    Mul(tmp, tmpMMinusMLoopCtr, srcOffset);
    Add(ARMCG::XReg(GetSrcXRegIdx(0)), srcPtr, tmp,
        ARMCG::ShMod::LSL, Math::Log2(sizeof(SrcType)));
    for (std::uint32_t mIdx = 1; mIdx < mLim; ++mIdx) {
        Add(ARMCG::XReg(GetSrcXRegIdx(mIdx)),
            ARMCG::XReg(GetSrcXRegIdx(mIdx - 1)),
            srcOffset, ARMCG::ShMod::LSL, Math::Log2(sizeof(SrcType)));
    }
    // Main k loop
    Mov(kLoopCtr, kVal);
    if (k > KERNEL_K_STEP) {
        ARMCG::Label kLoopHead;
        L(kLoopHead);
        {
            Sub(kLoopCtr, kLoopCtr, KERNEL_K_STEP);
            Cmp(kLoopCtr, KERNEL_K_STEP);
            GenerateAccumulation(mLim, nLim, KERNEL_K_STEP);
        }
        B(ARMCG::Cond::GT, kLoopHead);
    }
    std::uint32_t kResid = k % KERNEL_K_STEP;
    if (kResid == 0) {
        kResid = KERNEL_K_STEP;
    }
    GenerateAccumulation(mLim, nLim, kResid);

    // Initialize dst pointers
    Mul(tmp, tmpMMinusMLoopCtr, dstOffset);
    // Row-major format of dst
    Add(ARMCG::XReg(GetDstXRegIdx(0)), dstPtr, tmp,
        ARMCG::ShMod::LSL, Math::Log2(sizeof(DstType)));
    Add(ARMCG::XReg(GetDstXRegIdx(0)),
        ARMCG::XReg(GetDstXRegIdx(0)),
        tmpNMinusNLoopCtr, ARMCG::ShMod::LSL, Math::Log2(sizeof(DstType)));
    for (std::uint32_t mIdx = 1; mIdx < mLim; ++mIdx) {
        Add(ARMCG::XReg(GetDstXRegIdx(mIdx)),
            ARMCG::XReg(GetDstXRegIdx(mIdx - 1)),
            dstOffset, ARMCG::ShMod::LSL, Math::Log2(sizeof(DstType)));
    }

    // Store data to dst
    StoreToDst(mLim, nLim, nResidScalar);
}

template <typename SrcType, typename WeiType, typename DstType, typename BiaType>
void NeonSU8MmlaGen<SrcType, WeiType, DstType, BiaType>::Mmla(const ARMCG::VReg16B &vn,
    const ARMCG::VReg16B &vm, const ARMCG::VReg4S &vd)
{
    if constexpr (std::is_same_v<SrcType, WeiType>) {
        if constexpr (std::is_same_v<SrcType, std::int8_t>) {
            Smmla(vd, vn, vm);
        } else {
            Ummla(vd, vn, vm);
        }
    } else {
        if (std::is_same_v<SrcType, std::int8_t> == wereSrcWeiSwapped) {
            Usmmla(vd, vn, vm);
        } else {
            // this gives dst in transposed format
            Usmmla(vd, vm, vn);
        }
    }
}

template <typename SrcType, typename WeiType, typename DstType, typename BiaType>
void NeonSU8MmlaGen<SrcType, WeiType, DstType, BiaType>::GenerateAccumulation(
    std::uint32_t mLim, std::uint32_t nLim, std::uint32_t kLim)
{
    // division by 2 isdue to peculiarities of MMLA instruciton (two lines are processed at once)
    for (std::uint32_t mIdx = 0; mIdx < Math::CeilDiv(mLim, NUM_2); ++mIdx) {
        if (kLim < KERNEL_K_STEP) {
            for (IntType j = 0; j < kLim; ++j) {
                ARMCG::CodeGenerator::Ld1(ARMCG::VReg16B(GetTmpSrcVRegIdx(mIdx, NUM_0))[j],
                    ARMCG::PostPtr(ARMCG::XReg(GetSrcXRegIdx(mIdx * NUM_2 + NUM_0)), NUM_1));
            }
        } else {
            ARMCG::CodeGenerator::Ld1(ARMCG::VReg16BList{ARMCG::VReg16B(GetTmpSrcVRegIdx(mIdx, NUM_0))},
                ARMCG::PostPtr(ARMCG::XReg(GetSrcXRegIdx(mIdx * NUM_2 + NUM_0)), NUM_16));
        }
        if ((mIdx * NUM_2 + NUM_1) < mLim) {
            if (kLim < KERNEL_K_STEP) {
                for (IntType j = 0; j < kLim; ++j) {
                    ARMCG::CodeGenerator::Ld1(ARMCG::VReg16B(GetTmpSrcVRegIdx(mIdx, NUM_1))[j],
                        ARMCG::PostPtr(ARMCG::XReg(GetSrcXRegIdx(mIdx * NUM_2 + NUM_1)), NUM_1));
                }
            } else {
                ARMCG::CodeGenerator::Ld1(ARMCG::VReg16BList{ARMCG::VReg16B(GetTmpSrcVRegIdx(mIdx, NUM_1))},
                    ARMCG::PostPtr(ARMCG::XReg(GetSrcXRegIdx(mIdx * NUM_2 + NUM_1)), NUM_16));
            }
        }
        Trn1(ARMCG::VReg2D(GetSrcVRegIdx(mIdx, NUM_0)), ARMCG::VReg2D(GetTmpSrcVRegIdx(mIdx, NUM_0)),
            ARMCG::VReg2D(GetTmpSrcVRegIdx(mIdx, NUM_1)));
        Trn2(ARMCG::VReg2D(GetSrcVRegIdx(mIdx, NUM_1)), ARMCG::VReg2D(GetTmpSrcVRegIdx(mIdx, NUM_0)),
            ARMCG::VReg2D(GetTmpSrcVRegIdx(mIdx, NUM_1)));
    }

    for (std::uint32_t kIdx = 0; kIdx < Math::CeilDiv(kLim, KERNEL_K_STEP / K_LOOP_UNROLL); ++kIdx) {
        for (std::uint32_t nIdx = 0; nIdx < nLim; ++nIdx) {
            ARMCG::CodeGenerator::Ld1(ARMCG::VReg16BList{ARMCG::VReg16B(GetWeiVRegIdx(nIdx))},
                ARMCG::PostPtr(ARMCG::XReg(GetWeiXRegIdx()), NUM_16));
            for (std::uint32_t mIdx = 0; mIdx < Math::CeilDiv(mLim, NUM_2); ++mIdx) {
                Mmla(ARMCG::VReg16B(GetSrcVRegIdx(mIdx, kIdx)),
                    ARMCG::VReg16B(GetWeiVRegIdx(nIdx)),
                    ARMCG::VReg4S(GetDstVRegIdx(mIdx, nIdx)));
            }
        }
        Mov(tmp, NUM_16 * (KERNEL_N_SIZE * NEON_VLW / NUM_2 - nLim));
        Add(ARMCG::XReg(GetWeiXRegIdx()), ARMCG::XReg(GetWeiXRegIdx()), tmp);
    }
}

template <typename SrcType, typename WeiType, typename DstType, typename BiaType>
MatMulImplGenerator<NeonSU8MmlaGen<SrcType, WeiType, DstType, BiaType>>::DefaultImpl::DefaultImpl(
    const GemmInfo &task, IntType nThreads) : MatMulImplBase(task, nThreads)
{
    UpdateExecParams();
    Initialize();
}

template <typename SrcTy, typename WeiTy, typename DstTy, typename BiaTy>
constexpr std::uint32_t MatMulImplGenerator<NeonSU8MmlaGen<SrcTy, WeiTy, DstTy, BiaTy>>::DefaultImpl::GetKernelMSize()
    noexcept
{
    return NeonSU8MmlaGen<SrcTy, WeiTy, DstTy, BiaTy>::KERNEL_M_SIZE;
}

template <typename SrcTy, typename WeiTy, typename DstTy, typename BiaTy>
constexpr std::uint32_t MatMulImplGenerator<NeonSU8MmlaGen<SrcTy, WeiTy, DstTy, BiaTy>>::DefaultImpl::GetKernelNSize()
    noexcept
{
    return NeonSU8MmlaGen<SrcTy, WeiTy, DstTy, BiaTy>::KERNEL_N_SIZE;
}

template <typename SrcTy, typename WeiTy, typename DstTy, typename BiaTy>
constexpr std::uint32_t MatMulImplGenerator<NeonSU8MmlaGen<SrcTy, WeiTy, DstTy, BiaTy>>::DefaultImpl::GetKernelKSize()
    noexcept
{
    return NeonSU8MmlaGen<SrcTy, WeiTy, DstTy, BiaTy>::KERNEL_K_STEP /
        NeonSU8MmlaGen<SrcTy, WeiTy, DstTy, BiaTy>::K_LOOP_UNROLL ;
}

template <typename SrcType, typename WeiType, typename DstType, typename BiaType>
void MatMulImplGenerator<NeonSU8MmlaGen<SrcType, WeiType, DstType, BiaType>>::DefaultImpl::InitializePackingPolicy()
{
    if (((dstLayout2D == Layout::ROW_MAJOR) && (broadcastedStridesSrc[IDX_3] < broadcastedStridesSrc[IDX_4])) ||
        ((dstLayout2D == Layout::COL_MAJOR) && (broadcastedStridesWei[IDX_3] >= broadcastedStridesWei[IDX_4]))) {
        packingPolicy = PackingPolicy::SYNC_DST_DIM_SRC;
    } else {
        packingPolicy = PackingPolicy::NO_SYNC;
    }
}

template <typename SrcType, typename WeiType, typename DstType, typename BiaType>
IntType MatMulImplGenerator<NeonSU8MmlaGen<SrcType, WeiType, DstType, BiaType>>::DefaultImpl::GetMBlockSize()
    const noexcept
{
    return 5100; // is experimentally measured constant to obtain better perf
}

template <typename SrcType, typename WeiType, typename DstType, typename BiaType>
IntType MatMulImplGenerator<NeonSU8MmlaGen<SrcType, WeiType, DstType, BiaType>>::DefaultImpl::GetNBlockSize()
    const noexcept
{
    return 128; // is experimentally measured constant to obtain better perf
}

template <typename SrcType, typename WeiType, typename DstType, typename BiaType>
IntType MatMulImplGenerator<NeonSU8MmlaGen<SrcType, WeiType, DstType, BiaType>>::DefaultImpl::GetKBlockSize()
    const noexcept
{
    return 8192; // is experimentally measured constant to obtain better perf
}

template <typename SrcType, typename WeiType, typename DstType, typename BiaType>
void MatMulImplGenerator<NeonSU8MmlaGen<SrcType, WeiType, DstType, BiaType>>::DefaultImpl::InitializeSingleCodegen(
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
        CodeGenParams<NeonSU8MmlaGen<SrcType, WeiType, DstType, BiaType>> currTask{
            currentM, currentN, currentK, wereSrcWeiSwapped, attributes};
        if (codegens.find(currTask) == codegens.end()) {
            codegens.insert({currTask, std::make_shared<NeonSU8MmlaGen<SrcType, WeiType, DstType, BiaType>>(
                currentM, currentN, currentK, wereSrcWeiSwapped, attributes)});
        }
        codegens[currTask]->Ready();
        matmulCodegenKernels[id] =
            codegens[currTask]->template GetCode<NeonSU8MmlaCodegenFuncPtr<SrcType, WeiType, DstType, BiaType>>();
    }
}

template <typename SrcType, typename WeiType, typename DstType, typename BiaType>
void MatMulImplGenerator<NeonSU8MmlaGen<SrcType, WeiType, DstType, BiaType>>::DefaultImpl::InitializeReorderCodeGens(
    IntType m, IntType n, IntType k, IntType mFull, IntType nFull)
{
    Service::Unused(n, nFull);

    IntType kBlockExecFull = GetKBlockSize();
    IntType kBlockExecResid = k % kBlockExecFull;

    if (perThreadParams[0].srcLayout2D == Layout::COL_MAJOR) {
        // naming convention: F - full block, R(1,2) - resid (1,2),
        // K, N, M - dimensions to which full block or resid is applied.
        // N - means partition of M by threadsN (see scheme in RunSequential2D)
        SizeType mFullThreadBlock = mFull;
        SizeType mResidThreadBlock = m % mFullThreadBlock;
        if (mResidThreadBlock == 0) {
            mResidThreadBlock = mFullThreadBlock;
        }
        SizeType FNFMBlock = Math::CeilDiv(GetMBlockSize(), threadsN);
        SizeType usedThreads = Math::CeilDiv(GetMBlockSize(), FNFMBlock);
        SizeType RNFMBlock = GetMBlockSize() - FNFMBlock * (usedThreads - 1);
        SizeType FNR1MBlock = Math::CeilDiv(mFullThreadBlock % GetMBlockSize(), threadsN);
        usedThreads = Math::CeilDiv(mFullThreadBlock % GetMBlockSize(), FNR1MBlock);
        SizeType RNR1MBlock = (mFullThreadBlock % GetMBlockSize()) - FNR1MBlock * (usedThreads - 1);
        SizeType FNR2MBlock = Math::CeilDiv(mResidThreadBlock % GetMBlockSize(), threadsN);
        usedThreads = Math::CeilDiv(mResidThreadBlock % GetMBlockSize(), FNR2MBlock);
        SizeType RNR2MBlock = (mResidThreadBlock % GetMBlockSize()) - FNR2MBlock * (usedThreads - 1);

        CodeGenWrapper<FastReorderSU8SU8DiffLtKernelCodeGen<SrcType, SrcType>> cgwPackAcolFKFNFM;
        cgwPackAcolFKFNFM.Initialize(static_cast<SizeType>(kBlockExecFull), FNFMBlock, false);
        CodeGenWrapper<FastReorderSU8SU8DiffLtKernelCodeGen<SrcType, SrcType>> cgwPackAcolFKFNR1M;
        cgwPackAcolFKFNR1M.Initialize(static_cast<SizeType>(kBlockExecFull), FNR1MBlock, false);
        CodeGenWrapper<FastReorderSU8SU8DiffLtKernelCodeGen<SrcType, SrcType>> cgwPackAcolFKFNR2M;
        cgwPackAcolFKFNR2M.Initialize(static_cast<SizeType>(kBlockExecFull), FNR2MBlock, false);

        CodeGenWrapper<FastReorderSU8SU8DiffLtKernelCodeGen<SrcType, SrcType>> cgwPackAcolFKRNFM;
        cgwPackAcolFKRNFM.Initialize(static_cast<SizeType>(kBlockExecFull), RNFMBlock, false);
        CodeGenWrapper<FastReorderSU8SU8DiffLtKernelCodeGen<SrcType, SrcType>> cgwPackAcolFKRNRM1;
        cgwPackAcolFKRNRM1.Initialize(static_cast<SizeType>(kBlockExecFull), RNR1MBlock, false);
        CodeGenWrapper<FastReorderSU8SU8DiffLtKernelCodeGen<SrcType, SrcType>> cgwPackAcolFKRNR2M;
        cgwPackAcolFKRNR2M.Initialize(static_cast<SizeType>(kBlockExecFull), RNR2MBlock, false);

        CodeGenWrapper<FastReorderSU8SU8DiffLtKernelCodeGen<SrcType, SrcType>> cgwPackAcolRKFNFM;
        cgwPackAcolRKFNFM.Initialize(static_cast<SizeType>(kBlockExecResid), FNFMBlock, false);
        CodeGenWrapper<FastReorderSU8SU8DiffLtKernelCodeGen<SrcType, SrcType>> cgwPackAcolRKFNR1M;
        cgwPackAcolRKFNR1M.Initialize(static_cast<SizeType>(kBlockExecResid), FNR1MBlock, false);
        CodeGenWrapper<FastReorderSU8SU8DiffLtKernelCodeGen<SrcType, SrcType>> cgwPackAcolRKFNR2M;
        cgwPackAcolRKFNR2M.Initialize(static_cast<SizeType>(kBlockExecResid), FNR2MBlock, false);

        CodeGenWrapper<FastReorderSU8SU8DiffLtKernelCodeGen<SrcType, SrcType>> cgwPackAcolRKRNFM;
        cgwPackAcolRKRNFM.Initialize(static_cast<SizeType>(kBlockExecResid), RNFMBlock, false);
        CodeGenWrapper<FastReorderSU8SU8DiffLtKernelCodeGen<SrcType, SrcType>> cgwPackAcolRKRNRM1;
        cgwPackAcolRKRNRM1.Initialize(static_cast<SizeType>(kBlockExecResid), RNR1MBlock, false);
        CodeGenWrapper<FastReorderSU8SU8DiffLtKernelCodeGen<SrcType, SrcType>> cgwPackAcolRKRNR2M;
        cgwPackAcolRKRNR2M.Initialize(static_cast<SizeType>(kBlockExecResid), RNR2MBlock, false);

        packAColCodegenKernels = {
            std::move(cgwPackAcolFKFNFM), std::move(cgwPackAcolFKFNR1M), std::move(cgwPackAcolFKFNR2M),
            std::move(cgwPackAcolFKRNFM), std::move(cgwPackAcolFKRNRM1), std::move(cgwPackAcolFKRNR2M),
            std::move(cgwPackAcolRKFNFM), std::move(cgwPackAcolRKFNR1M), std::move(cgwPackAcolRKFNR2M),
            std::move(cgwPackAcolRKRNFM), std::move(cgwPackAcolRKRNRM1), std::move(cgwPackAcolRKRNR2M)
        };
    }
}

template <typename SrcType, typename WeiType, typename DstType, typename BiaType>
void MatMulImplGenerator<NeonSU8MmlaGen<SrcType, WeiType, DstType, BiaType>>::DefaultImpl::InitializeThreadsCountImpl()
{
    IntType threadNumber;
    IntType size = GetRequiredMemSize();
    if (size < 6144) { // 6144 is practically measured number
        threadNumber = NUM_1;
    } else if (size < 38400) { // 38400 is practically measured number
        threadNumber = NUM_8;
    } else if (size < 104544) { // 104544 is practically measured number
        threadNumber = NUM_16;
    } else if (size < 269664) { // 269664 is practically measured number
        threadNumber = NUM_40;
    } else {
        threadNumber = NUM_80;
    }
    nThreadsExec = std::min(threadNumber, implNThreads);
}

template <typename SrcType, typename WeiType, typename DstType, typename BiaType>
IntType MatMulImplGenerator<NeonSU8MmlaGen<SrcType, WeiType, DstType, BiaType>>::DefaultImpl::GetThreadBufSize(
    IntType m, IntType n, IntType k, MatrixKind mKind) const noexcept
{
    Service::Unused(m);
    IntType vlb = NUM_16; // 16x8 bit elements in 1 NEON register
    if (mKind == MatrixKind::SRC) {
        if (((dstLayout2D == Layout::ROW_MAJOR) && (broadcastedStridesSrc[IDX_3] < broadcastedStridesSrc[IDX_4])) ||
            ((dstLayout2D == Layout::COL_MAJOR) && (broadcastedStridesWei[IDX_3] >= broadcastedStridesWei[IDX_4]))) {
            return std::min(GetKBlockSize(), k) * std::min(GetMBlockSize(), m) * sizeof(SrcType);
        } else {
            return 0;
        }
    } else {
        return static_cast<IntType>(sizeof(WeiType) *
            Math::CeilDiv(std::min<IntType>(GetNBlockSize(), n), vlb) * vlb *
            Math::CeilDiv(std::min<IntType>(GetKBlockSize(), k), GetKernelKSize() * NUM_2) * GetKernelKSize() * NUM_2);
    }
}

template <typename SrcType, typename WeiType, typename DstType, typename BiaType>
void MatMulImplGenerator<NeonSU8MmlaGen<SrcType, WeiType, DstType, BiaType>>::DefaultImpl::ReserveImplSpecificParams()
{}

using vepa = std::vector<std::pair<Shape, Shape>>;
template <typename SrcType, typename WeiType, typename DstType, typename BiaType>
vepa MatMulImplGenerator<NeonSU8MmlaGen<SrcType, WeiType, DstType, BiaType>>::DefaultImpl::SplitWork()
{
    return CalculateWorkOnThread<DefaultImpl>(taskExec.dstInfo.GetDims(), nThreadsExec, dstLayout2D == Layout::BA);
}

template <typename SrcType, typename WeiType, typename DstType, typename BiaType>
void MatMulImplGenerator<NeonSU8MmlaGen<SrcType, WeiType, DstType, BiaType>>::DefaultImpl::PackBCol(
    const WeiType* B_ptr, WeiType* B_buffer, IntType N, IntType K, IntType ldb)
{
    IntType vlb = NUM_16; // 16x8 bit elements in 1 NEON register
    IntType kRoundedUp = Math::CeilDiv(K, NUM_8) * NUM_8;
    for (IntType kBlock = 0; kBlock < K; kBlock += NUM_8) {
        IntType kBlockLim = std::min(K - kBlock, NUM_8);
        for (IntType nBlock = 0; nBlock < N; nBlock += vlb) {
            IntType nBlockLim = std::min(N - nBlock, vlb);
            for (IntType n = 0; n < nBlockLim; ++n) {
                for (IntType k = 0; k < kBlockLim; ++k) {
                    B_buffer[k + n * NUM_8 + kBlock * vlb + nBlock * kRoundedUp] = B_ptr[kBlock + k +
                        ldb * (nBlock + n)];
                }
                for (IntType k = kBlockLim; k < NUM_8; ++k) {
                    B_buffer[k + n * NUM_8 + kBlock * vlb + nBlock * kRoundedUp] = 0;
                }
            }
            for (IntType n = nBlockLim; n < vlb; ++n) {
                for (IntType k = 0; k < NUM_8; ++k) {
                    B_buffer[k + n * NUM_8 + kBlock * vlb + nBlock * kRoundedUp] = 0;
                }
            }
        }
    }
}

template <typename SrcType, typename WeiType, typename DstType, typename BiaType>
void MatMulImplGenerator<NeonSU8MmlaGen<SrcType, WeiType, DstType, BiaType>>::DefaultImpl::PackBRow(
    const WeiType* B_ptr, WeiType* B_buffer, IntType N, IntType K, IntType ldb)
{
    IntType vlb = NUM_16; // 16x8 bit elements in 1 NEON register
    IntType kRoundedUp = Math::CeilDiv(K, NUM_8) * NUM_8;
    for (IntType kBlock = 0; kBlock < K; kBlock += NUM_8) {
        IntType kBlockLim = std::min(K - kBlock, NUM_8);
        for (IntType nBlock = 0; nBlock < N; nBlock += vlb) {
            IntType nBlockLim = std::min(N - nBlock, vlb);
            for (IntType k = 0; k < kBlockLim; ++k) {
                for (IntType n = 0; n < nBlockLim; ++n) {
                    B_buffer[k + n * NUM_8 + kBlock * vlb + nBlock * kRoundedUp] = B_ptr[(kBlock + k) * ldb +
                        (nBlock + n)];
                }
                for (IntType n = nBlockLim; n < vlb; ++n) {
                    B_buffer[k + n * NUM_8 + kBlock * vlb + nBlock * kRoundedUp] = 0;
                }
            }
            for (IntType k = kBlockLim; k < NUM_8; ++k) {
                for (IntType n = 0; n < vlb; ++n) {
                    B_buffer[k + n * NUM_8 + kBlock * vlb + nBlock * kRoundedUp] = 0;
                }
            }
        }
    }
}

template <typename SrcType, typename WeiType, typename DstType, typename BiaType>
void MatMulImplGenerator<NeonSU8MmlaGen<SrcType, WeiType, DstType, BiaType>>::DefaultImpl::RunSequential2D(
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

    IntType offsetSrc = perThreadParams[threadID].srcOffset2D;
    if ((IsLayoutAlikeBA(taskExec.srcInfo.GetLayout()) && IsLayoutAlikeAB(taskExec.dstInfo.GetLayout())) ||
        (IsLayoutAlikeAB(taskExec.weiInfo.GetLayout()) && IsLayoutAlikeBA(taskExec.dstInfo.GetLayout()))) {
        offsetSrc = std::min(k, GetKBlockSize());
    }

    for (IntType mBlock = 0; mBlock < mFullBlockSizePerThread; mBlock += GetMBlockSize()) {
        auto mBlockExec = std::min(GetMBlockSize(),  m - mBlock);
        IntType mIdx = (mBlockExec == GetMBlockSize()) ? 0 : 1;
        IntType mIdxKernel = mIdx * (perThreadParams[threadID].isResidByM + 1);
        IntType packMPortionCommon = Math::CeilDiv(mBlockExec, threadsN);
        IntType numPackThreads = Math::CeilDiv(mBlockExec, packMPortionCommon);
        IntType residNThread = tIDn == (numPackThreads - 1);
        for (IntType kBlock = 0; kBlock < k; kBlock += GetKBlockSize()) {
            auto kBlockExec = std::min(GetKBlockSize(), k - kBlock);
            IntType kIdxKernel = (kBlockExec == GetKBlockSize()) ? 0 : 1;
            bool lastKIter = (k - kBlock) <= GetKBlockSize();
            MatMulDstFlags biaFlagsExec = SetBiaFlagsExec(biaFlags, beta, kBlock, lastKIter, alpha);
            IntType kBlockExecRoundedUp = Math::CeilDiv(kBlockExec, NUM_8) * NUM_8;
            const SrcType *srcExec = static_cast<const SrcType *>(src);
            if (perThreadParams[threadID].srcLayout2D == Layout::BA) {
                const SrcType *srcPtr = static_cast<const SrcType *>(src) +
                    mBlock + packMPortionCommon * tIDn + kBlock * perThreadParams[threadID].srcOffset2D;
                if ((tIDn < numPackThreads) && (mBlock < m)) {
                    auto packA = packAColCodegenKernels[kIdxKernel * NUM_6 + residNThread * NUM_3 + mIdxKernel].
                        template GetCode<void(*)(const SrcType *, SrcType *, IntType, IntType)>();
                    packA(srcPtr, srcBuffer + (packMPortionCommon * tIDn) * std::min(GetKBlockSize(), k),
                        perThreadParams[threadID].srcOffset2D * sizeof(SrcType),
                        std::min(GetKBlockSize(), k) * sizeof(SrcType));
                }
                srcPackStrategy->Barrier(tIDbatch * threadsM + tIDm, tIDn, threadsN);
                srcExec = srcBuffer;
            } else {
                srcExec += kBlock + mBlock * perThreadParams[threadID].srcOffset2D;
            }
            for (IntType nBlock = 0; nBlock < nFullBlockSizePerThread; nBlock += GetNBlockSize()) {
                if ((nBlock < n) && (mBlock < m)) {
                    auto nBlockExec = std::min(GetNBlockSize(), n - nBlock);
                    int nIdx = (nBlockExec == GetNBlockSize()) ? 0 : 1;
                    IntType nIdxKernel = nIdx * (perThreadParams[threadID].isResidByN + 1);
                    if (perThreadParams[threadID].weiLayout2D == Layout::ROW_MAJOR) {
                        const WeiType *weiExec = static_cast<const WeiType *>(wei) + nBlock +
                            kBlock * perThreadParams[threadID].weiOffset2D;
                        PackBRow(weiExec, weiBuffer, nBlockExec, kBlockExec, perThreadParams[threadID].weiOffset2D);
                    } else {
                        const WeiType *weiExec = static_cast<const WeiType *>(wei) +
                            nBlock * perThreadParams[threadID].weiOffset2D + kBlock;
                        PackBCol(weiExec, weiBuffer, nBlockExec, kBlockExec, perThreadParams[threadID].weiOffset2D);
                    }
                    DstType *dstExec = static_cast<DstType *>(dst) + nBlock +
                        mBlock * perThreadParams[threadID].dstOffset2D;
                    const BiaType *biaExec = static_cast<const BiaType *>(bia) + nBlock * biaNOffset +
                        mBlock * perThreadParams[threadID].biaOffset2D;
                    auto kernel = matmulCodegenKernels[kIdxKernel * NUM_9 +
                        nIdxKernel * NUM_3 + mIdxKernel];
                    kernel(srcExec, weiBuffer, dstExec, biaExec, biaFlagsExec,
                        alpha, beta, mBlockExec, nBlockExec, kBlockExec,
                        offsetSrc, kBlockExecRoundedUp,
                        perThreadParams[threadID].dstOffset2D, perThreadParams[threadID].biaOffset2D);
                }
            }
            if (perThreadParams[threadID].srcLayout2D == Layout::BA) {
                srcPackStrategy->Barrier(tIDbatch * threadsM + tIDm, tIDn, threadsN);
            }
        }
    }
}

template <typename SrcType, typename WeiType, typename DstType, typename BiaType>
void MatMulImplGenerator<NeonSU8MmlaGen<SrcType, WeiType, DstType, BiaType>>::DefaultImpl::UpdateExecParams() noexcept
{
    Layout desiredSrcLayout = implTask.srcInfo.GetLayout();
    Layout desiredWeiLayout = implTask.weiInfo.GetLayout();
    Layout desiredDstLayout = implTask.dstInfo.GetLayout();
    Layout desiredBiaLayout = implTask.biaInfo.GetLayout();

    {
        if (IsLayoutAlikeAB(desiredDstLayout)) {
            if (!IsLayoutAlikeAB(desiredSrcLayout) && !IsLayoutAlikeBA(desiredSrcLayout)) {
                desiredSrcLayout = GetPlaneLayoutAB(desiredSrcLayout);
            }
            if (!IsLayoutAlikeAB(desiredWeiLayout) && !IsLayoutAlikeBA(desiredWeiLayout)) {
                desiredWeiLayout = GetPlaneLayoutAB(desiredWeiLayout);
            }
        } else if (IsLayoutAlikeBA(desiredDstLayout)) {
            if (!IsLayoutAlikeAB(desiredWeiLayout) && !IsLayoutAlikeBA(desiredWeiLayout)) {
                desiredWeiLayout = GetPlaneLayoutBA(desiredWeiLayout);
            }
            if (!IsLayoutAlikeAB(desiredSrcLayout) && !IsLayoutAlikeBA(desiredSrcLayout)) {
                desiredSrcLayout = GetPlaneLayoutBA(desiredSrcLayout);
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

    if (implTask.biaInfo.GetTotalTensorSize() != 0) {
        desiredBiaLayout = desiredDstLayout;
    }
    InitializeTaskExec(desiredSrcLayout, desiredWeiLayout, desiredDstLayout, desiredBiaLayout,
        NeonSU8MmlaGen<SrcType, WeiType, DstType, BiaType>::GetSrcType(),
        NeonSU8MmlaGen<SrcType, WeiType, DstType, BiaType>::GetWeiType(),
        NeonSU8MmlaGen<SrcType, WeiType, DstType, BiaType>::GetDstType(),
        (implTask.biaInfo.GetTotalTensorSize() != 0) ?
            NeonSU8MmlaGen<SrcType, WeiType, DstType, BiaType>::GetBiaType() :
            static_cast<Element::TypeT>(implTask.biaInfo.GetType()));
}

template <typename SrcType, typename WeiType, typename DstType, typename BiaType>
MatMulDstFlags MatMulImplGenerator<NeonSU8MmlaGen<SrcType, WeiType, DstType, BiaType>>::DefaultImpl::SetBiaFlagsExec(
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

template struct MatMulImplGenerator<MatMulKernelDefaultCodeGen
    <ARCH::NEON, std::int8_t, std::int8_t, std::int32_t, std::int32_t, MatMulImplKinds::SU8SU8S32S32_NEON_MMLA>>;
template struct MatMulImplGenerator<MatMulKernelDefaultCodeGen
    <ARCH::NEON, std::int8_t, std::uint8_t, std::int32_t, std::int32_t, MatMulImplKinds::SU8SU8S32S32_NEON_MMLA>>;
template struct MatMulImplGenerator<MatMulKernelDefaultCodeGen
    <ARCH::NEON, std::uint8_t, std::int8_t, std::int32_t, std::int32_t, MatMulImplKinds::SU8SU8S32S32_NEON_MMLA>>;
template struct MatMulImplGenerator<MatMulKernelDefaultCodeGen
    <ARCH::NEON, std::uint8_t, std::uint8_t, std::int32_t, std::int32_t, MatMulImplKinds::SU8SU8S32S32_NEON_MMLA>>;

} // Detail
} // KuDNN
