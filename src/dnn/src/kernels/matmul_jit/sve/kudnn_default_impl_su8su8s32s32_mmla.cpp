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

#include "kernels/matmul_jit/sve/kudnn_default_impl_su8su8s32s32_mmla.hpp"
#include "kernels/matmul_jit/kudnn_gemm_threading_helpers.hpp"

#include <arm_sve.h>

namespace KuDNN {
namespace Detail {

template <typename SrcType, typename WeiType, typename DstType, typename BiaType>
using VCSU8MmlaGenBase = MatMulKernelCodeGenBase<ARCH::VC, SrcType, WeiType, DstType, BiaType>;

template <typename SrcType, typename WeiType, typename DstType, typename BiaType>
VCSU8MmlaGen<SrcType, WeiType, DstType, BiaType>::MatMulKernelDefaultCodeGen(
    IntType m, IntType n, IntType k,
    bool wereSrcWeiSwappedArg, const Attributes &attributesArg)
    : VCSU8MmlaGenBase<SrcType, WeiType, DstType, BiaType>(attributesArg),
    wereSrcWeiSwapped(wereSrcWeiSwappedArg), vectorLengthByte(svcntb())
{
    if ((m > 0) && (n > 0) && (k > 0)) {
        if (this->attributes.GetPostOps().Len() > 0) {
            // set reserved dst registers
            for (std::uint32_t mIdx = 0; mIdx < Math::CeilDiv(KERNEL_M_SIZE, NUM_2); ++mIdx) {
                for (std::uint32_t nIdx = 0; nIdx < KERNEL_N_SIZE; ++nIdx) {
                    KUDNN_JIT_Generator_With_Attributes::TakeVectorRegister(
                        GetTmpDstZRegIdx(mIdx, nIdx * NUM_2));
                    KUDNN_JIT_Generator_With_Attributes::TakeVectorRegister(
                        GetTmpDstZRegIdx(mIdx, nIdx * NUM_2 + NUM_1));
                }
            }
            // set reserved predicate register
            for (std::uint32_t i = 0; i < NUM_6; ++i) {
                KUDNN_JIT_Generator_With_Attributes::TakePredicateRegister(i);
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

        Ptrue(ARMCG::PRegB(NUM_5), ARMCG::Pattern::ALL); // p5 is true (byte) for this impl

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
void VCSU8MmlaGen<SrcType, WeiType, DstType, BiaType>::CvtAndAdd(
    bool applyConvert, int tmpRegIdx, int mRegIdx, int nRegIdx)
{
    ARMCG::Label noConv;
    if (applyConvert) {
        Tbz(flags, 12, noConv); // 12-bit correspond to FIRST_K_ITER
        Scvtf(ARMCG::ZRegS(GetTmpZRegIdx(tmpRegIdx)),
            p5, ARMCG::ZRegS(GetTmpZRegIdx(tmpRegIdx)));
        L(noConv);
        Fadd(ARMCG::ZRegS(GetStDstZRegIdx(mRegIdx, nRegIdx)),
            p5, ARMCG::ZRegS(GetTmpZRegIdx(tmpRegIdx)));
    } else {
        Add(ARMCG::ZRegS(GetStDstZRegIdx(mRegIdx, nRegIdx)),
            ARMCG::ZRegS(GetStDstZRegIdx(mRegIdx, nRegIdx)),
            ARMCG::ZRegS(GetTmpZRegIdx(tmpRegIdx)));
    }
}

template <typename SrcType, typename WeiType, typename DstType, typename BiaType>
void VCSU8MmlaGen<SrcType, WeiType, DstType, BiaType>::CvtAndStore(bool applyConvert,
    std::uint32_t mLim, std::uint32_t nLim)
{
    nLim /= NUM_2;
    ARMCG::Label su8VCloadC;
    ARMCG::Label su8VCmullC;
    ARMCG::Label su8VCloadBias1;
    ARMCG::Label su8VCloadBiasM;
    ARMCG::Label su8VCloadBiasN;
    ARMCG::Label su8VCloadBiasMN;
    ARMCG::Label su8VCaddBias1;
    ARMCG::Label su8VCaddBiasM;
    ARMCG::Label su8VCaddBiasN;
    ARMCG::Label su8VCaddBiasMN;
    ARMCG::Label su8VCaddinitDone;
    ARMCG::Label su8VCfillZeros;
    ARMCG::Label skipPO;

    if (!(std::is_same_v<SrcType, WeiType>) && !(std::is_same_v<SrcType, std::int8_t> == wereSrcWeiSwapped)) {
        // dst in transposed format
        for (std::uint32_t mIdx = 0; mIdx < Math::CeilDiv(mLim, NUM_2); ++mIdx) {
            for (std::uint32_t nIdx = 0; nIdx < nLim; ++nIdx) {
                Uzp1(ARMCG::ZRegS(GetTmpDstZRegIdx(mIdx, nIdx * NUM_2)),
                    ARMCG::ZRegS(GetDstZRegIdx(mIdx, nIdx * NUM_2)),
                    ARMCG::ZRegS(GetDstZRegIdx(mIdx, nIdx * NUM_2 + NUM_1)));
                Uzp2(ARMCG::ZRegS(GetTmpDstZRegIdx(mIdx, nIdx * NUM_2 + NUM_1)),
                    ARMCG::ZRegS(GetDstZRegIdx(mIdx, nIdx * NUM_2)),
                    ARMCG::ZRegS(GetDstZRegIdx(mIdx, nIdx * NUM_2 + NUM_1)));
            }
        }
    } else {
        for (std::uint32_t mIdx = 0; mIdx < Math::CeilDiv(mLim, NUM_2); ++mIdx) {
            for (std::uint32_t nIdx = 0; nIdx < nLim; ++nIdx) {
                Uzp1(ARMCG::ZRegD(GetTmpDstZRegIdx(mIdx, nIdx * NUM_2)),
                    ARMCG::ZRegD(GetDstZRegIdx(mIdx, nIdx * NUM_2)),
                    ARMCG::ZRegD(GetDstZRegIdx(mIdx, nIdx * NUM_2 + NUM_1)));
                Uzp2(ARMCG::ZRegD(GetTmpDstZRegIdx(mIdx, nIdx * NUM_2 + NUM_1)),
                    ARMCG::ZRegD(GetDstZRegIdx(mIdx, nIdx * NUM_2)),
                    ARMCG::ZRegD(GetDstZRegIdx(mIdx, nIdx * NUM_2 + NUM_1)));
            }
        }
    }
    // now we've unzipped dst regs and can work with them normally
    if (applyConvert) {
        for (std::uint32_t mIdx = 0; mIdx < mLim; ++mIdx) {
            for (std::uint32_t nIdx = 0; nIdx < nLim; ++nIdx) {
                Scvtf(ARMCG::ZRegS(GetStDstZRegIdx(mIdx, nIdx)),
                    p5, ARMCG::ZRegS(GetStDstZRegIdx(mIdx, nIdx)));
            }
        }
        // Load alpha
        Ld1rw(ARMCG::ZRegS(GetTmpZRegIdx(0)), p5, Ptr(sp, static_cast<std::int32_t>(NUM_16)));
        Mov(tmpW, ARMCG::VReg4S(GetTmpZRegIdx(0))[0]);
        Mov(ARMCG::WReg(GetTmpXRegIdx(0)), 0x3f800000); // 0x3f800000 is a bit representation of 1.0f
        Cmp(tmpW, ARMCG::WReg(GetTmpXRegIdx(0)));
        Beq(su8VCfillZeros);
        for (std::uint32_t mIdx = 0; mIdx < mLim; ++mIdx) {
            for (std::uint32_t nIdx = 0; nIdx < nLim; ++nIdx) {
                Fmul(
                    ARMCG::ZRegS(GetStDstZRegIdx(mIdx, nIdx)),
                    ARMCG::ZRegS(GetStDstZRegIdx(mIdx, nIdx)),
                    ARMCG::ZRegS(GetTmpZRegIdx(0))
                );
            }
        }
    }
    L(su8VCfillZeros);
    Tbz(flags, 0, su8VCloadBias1); // 0-bit correspond to FILL_C_WITH_ZEROS
    L(su8VCloadBias1);
    Tbz(flags, 3, su8VCloadBiasN); // 3-bit correspond to LOAD_BIAS_1
    Ld1rw(ARMCG::ZRegS(GetTmpZRegIdx(NUM_7)), p5, ARMCG::Ptr(biaPtr));
    for (std::uint32_t mIdx = 0; mIdx < mLim; ++mIdx) {
        for (std::uint32_t nIdx = 0; nIdx < nLim; ++nIdx) {
            Mov(ARMCG::ZRegS(GetTmpZRegIdx(nIdx)), p5, ARMCG::ZRegS(GetTmpZRegIdx(NUM_7)));
            CvtAndAdd(applyConvert, nIdx, mIdx, nIdx);
        }
    }
    L(su8VCloadBiasN);
    Tbz(flags, 5, su8VCloadBiasM); // 5-bit correspond to LOAD_BIAS_N
    Add(tmp, biaPtr, tmpNMinusNLoopCtr, ARMCG::ShMod::LSL, Math::Log2(sizeof(BiaType)));
    for (std::uint32_t mIdx = 0; mIdx < mLim; ++mIdx) {
        for (std::uint32_t nIdx = 0; nIdx < nLim; ++nIdx) {
            Ld1w(ARMCG::ZRegS(GetTmpZRegIdx(nIdx)), ARMCG::PRegS(GetLoadStorePRegIdx(nIdx)), ARMCG::Ptr(tmp,
                nIdx, ARMCG::ExtMod::MUL_VL));
            CvtAndAdd(applyConvert, nIdx, mIdx, nIdx);
        }
    }
    L(su8VCloadBiasM);
    Tbz(flags, 4, su8VCloadBiasMN); // 4-bit correspond to LOAD_BIAS_M
    Add(tmp, biaPtr, tmpMMinusMLoopCtr, ARMCG::ShMod::LSL, Math::Log2(sizeof(BiaType)));
    for (std::uint32_t mIdx = 0; mIdx < mLim; ++mIdx) {
        Ld1rw(ARMCG::ZRegS(GetTmpZRegIdx(mIdx)), p5, ARMCG::Ptr(tmp,
            static_cast<std::uint32_t>(sizeof(BiaType) * mIdx)));
        for (std::uint32_t nIdx = 0; nIdx < nLim; ++nIdx) {
            Mov(ARMCG::ZRegS(GetTmpZRegIdx(mLim)), p5, ARMCG::ZRegS(GetTmpZRegIdx(mIdx)));
            CvtAndAdd(applyConvert, mLim, mIdx, nIdx);
        }
    }
    L(su8VCloadBiasMN);
    Tbz(flags, 6, su8VCloadC); // 6-bit correspond to LOAD_BIAS_MN
    for (std::uint32_t mIdx = 0; mIdx < mLim; ++mIdx) {
        for (std::uint32_t nIdx = 0; nIdx < nLim; ++nIdx) {
            Add(ARMCG::XReg(GetTmpXRegIdx(mIdx)), tmpMMinusMLoopCtr, mIdx);
            Mul(ARMCG::XReg(GetTmpXRegIdx(mIdx)), ARMCG::XReg(GetTmpXRegIdx(mIdx)), biaOffset);
            Add(ARMCG::XReg(GetTmpXRegIdx(mIdx)), biaPtr, ARMCG::XReg(GetTmpXRegIdx(mIdx)),
                ARMCG::ShMod::LSL, Math::Log2(sizeof(BiaType)));
            Add(ARMCG::XReg(GetTmpXRegIdx(mIdx)), ARMCG::XReg(GetTmpXRegIdx(mIdx)), tmpNMinusNLoopCtr,
                ARMCG::ShMod::LSL, Math::Log2(sizeof(BiaType)));
            Ld1w(ARMCG::ZRegS(GetTmpZRegIdx(mIdx)), ARMCG::PRegS(GetLoadStorePRegIdx(nIdx)),
                ARMCG::Ptr(ARMCG::XReg(GetTmpXRegIdx(mIdx)),
                nIdx, ARMCG::ExtMod::MUL_VL));
            CvtAndAdd(applyConvert, mIdx, mIdx, nIdx);
        }
    }
    L(su8VCloadC);
    Tbz(flags, 1, su8VCmullC); // 1-bit correspond to LOAD_C
    for (std::uint32_t mIdx = 0; mIdx < mLim; ++mIdx) {
        for (std::uint32_t nIdx = 0; nIdx < nLim; ++nIdx) {
            Ld1w(ARMCG::ZRegS(GetTmpZRegIdx(mIdx)), ARMCG::PRegS(GetLoadStorePRegIdx(nIdx)),
                ARMCG::Ptr(ARMCG::XReg(GetDstXRegIdx(mIdx)),
                nIdx, ARMCG::ExtMod::MUL_VL));
            CvtAndAdd(applyConvert, mIdx, mIdx, nIdx);
        }
    }
    L(su8VCmullC);
    Tbz(flags, 2, su8VCaddBias1); // 2-bit correspond to MULL_C
    Ld1rw(ARMCG::ZRegS(GetTmpZRegIdx(NUM_7)), p5, Ptr(sp));
    for (std::uint32_t mIdx = 0; mIdx < mLim; ++mIdx) {
        for (std::uint32_t nIdx = 0; nIdx < nLim; ++nIdx) {
            Ld1w(ARMCG::ZRegS(GetTmpZRegIdx(mIdx)), ARMCG::PRegS(GetLoadStorePRegIdx(nIdx)),
                ARMCG::Ptr(ARMCG::XReg(GetDstXRegIdx(mIdx)),
                nIdx, ARMCG::ExtMod::MUL_VL));
            Scvtf(ARMCG::ZRegS(GetTmpZRegIdx(mIdx)),
                p5, ARMCG::ZRegS(GetTmpZRegIdx(mIdx)));
            Fmul(
                ARMCG::ZRegS(GetTmpZRegIdx(mIdx)),
                ARMCG::ZRegS(GetTmpZRegIdx(mIdx)),
                ARMCG::ZRegS(GetTmpZRegIdx(NUM_7))
            );
            Fadd(ARMCG::ZRegS(GetStDstZRegIdx(mIdx, nIdx)), p5,
                ARMCG::ZRegS(GetTmpZRegIdx(mIdx)));
        }
    }
    L(su8VCaddBias1);
    Tbz(flags, 7, su8VCaddBiasN); // 7-bit correspond to ADD_BIAS_1
    Ld1rw(ARMCG::ZRegS(GetTmpZRegIdx(NUM_7)), p5, ARMCG::Ptr(biaPtr));
    for (std::uint32_t mIdx = 0; mIdx < mLim; ++mIdx) {
        for (std::uint32_t nIdx = 0; nIdx < nLim; ++nIdx) {
            Mov(ARMCG::ZRegS(GetTmpZRegIdx(nIdx)), p5, ARMCG::ZRegS(GetTmpZRegIdx(NUM_7)));
            CvtAndAdd(applyConvert, nIdx, mIdx, nIdx);
        }
    }
    L(su8VCaddBiasN);
    Tbz(flags, 9, su8VCaddBiasM); // 9-bit correspond to ADD_BIAS_N
    Add(tmp, biaPtr, tmpNMinusNLoopCtr, ARMCG::ShMod::LSL, Math::Log2(sizeof(BiaType)));
    for (std::uint32_t mIdx = 0; mIdx < mLim; ++mIdx) {
        for (std::uint32_t nIdx = 0; nIdx < nLim; ++nIdx) {
            Ld1w(ARMCG::ZRegS(GetTmpZRegIdx(nIdx)), ARMCG::PRegS(GetLoadStorePRegIdx(nIdx)),
                ARMCG::Ptr(tmp, nIdx, ARMCG::ExtMod::MUL_VL));
            CvtAndAdd(applyConvert, nIdx, mIdx, nIdx);
        }
    }
    L(su8VCaddBiasM);
    Tbz(flags, 8, su8VCaddBiasMN); // 8-bit correspond to ADD_BIAS_M
    Add(tmp, biaPtr, tmpMMinusMLoopCtr, ARMCG::ShMod::LSL, Math::Log2(sizeof(BiaType)));
    for (std::uint32_t mIdx = 0; mIdx < mLim; ++mIdx) {
        Ld1rw(ARMCG::ZRegS(GetTmpZRegIdx(mIdx)), p5, ARMCG::Ptr(tmp,
            static_cast<std::uint32_t>(sizeof(BiaType) * mIdx)));
        for (std::uint32_t nIdx = 0; nIdx < nLim; ++nIdx) {
            Mov(ARMCG::ZRegS(GetTmpZRegIdx(mLim)), p5, ARMCG::ZRegS(GetTmpZRegIdx(mIdx)));
            CvtAndAdd(applyConvert, mLim, mIdx, nIdx);
        }
    }
    L(su8VCaddBiasMN);
    Tbz(flags, 10, su8VCaddinitDone); // 10-bit correspond to ADD_BIAS_MN
    for (std::uint32_t mIdx = 0; mIdx < mLim; ++mIdx) {
        for (std::uint32_t nIdx = 0; nIdx < nLim; ++nIdx) {
            Add(ARMCG::XReg(GetTmpXRegIdx(mIdx)), tmpMMinusMLoopCtr, mIdx);
            Mul(ARMCG::XReg(GetTmpXRegIdx(mIdx)), ARMCG::XReg(GetTmpXRegIdx(mIdx)), biaOffset);
            Add(ARMCG::XReg(GetTmpXRegIdx(mIdx)), biaPtr, ARMCG::XReg(GetTmpXRegIdx(mIdx)),
                ARMCG::ShMod::LSL, Math::Log2(sizeof(BiaType)));
            Add(ARMCG::XReg(GetTmpXRegIdx(mIdx)), ARMCG::XReg(GetTmpXRegIdx(mIdx)), tmpNMinusNLoopCtr,
                ARMCG::ShMod::LSL, Math::Log2(sizeof(BiaType)));
            Ld1w(ARMCG::ZRegS(GetTmpZRegIdx(mIdx)), ARMCG::PRegS(GetLoadStorePRegIdx(nIdx)),
                ARMCG::Ptr(ARMCG::XReg(GetTmpXRegIdx(mIdx)),
                nIdx, ARMCG::ExtMod::MUL_VL));
            CvtAndAdd(applyConvert, mIdx, mIdx, nIdx);
        }
    }
    L(su8VCaddinitDone);
    Tbz(flags, 11, skipPO); // 11-bit correspond to APPLY_POST_OPS
    std::vector<SizeType> dstRegsIDs(mLim * nLim);
    for (std::uint32_t mIdx = 0; mIdx < mLim; ++mIdx) {
        for (std::uint32_t nIdx = 0; nIdx < nLim; ++nIdx) {
            dstRegsIDs[mIdx * nLim + nIdx] = GetStDstZRegIdx(mIdx, nIdx);
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
                Frinti(ARMCG::ZRegS(GetStDstZRegIdx(mIdx, nIdx)),
                    p5, ARMCG::ZRegS(GetStDstZRegIdx(mIdx, nIdx)));
                Fcvtzs(ARMCG::ZRegS(GetStDstZRegIdx(mIdx, nIdx)),
                    p5, ARMCG::ZRegS(GetStDstZRegIdx(mIdx, nIdx)));
            }
        }
    }
    L(skipPO);
    for (std::uint32_t mIdx = 0; mIdx < mLim; ++mIdx) {
        for (std::uint32_t nIdx = 0; nIdx < nLim; ++nIdx) {
            St1w(ARMCG::ZRegS(GetStDstZRegIdx(mIdx, nIdx)), ARMCG::PRegS(GetLoadStorePRegIdx(nIdx)),
                Ptr(ARMCG::XReg(GetDstXRegIdx(mIdx)), nIdx, ARMCG::ExtMod::MUL_VL));
        }
    }
}

template <typename SrcType, typename WeiType, typename DstType, typename BiaType>
void VCSU8MmlaGen<SrcType, WeiType, DstType, BiaType>::StoreToDst(std::uint32_t mLim, std::uint32_t nLim)
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
    CvtAndStore(true, mLim, nLim);
    B(ret);

    L(nof32Conv);
    CvtAndStore(false, mLim, nLim);
    L(ret);
}

template <typename SrcType, typename WeiType, typename DstType, typename BiaType>
void VCSU8MmlaGen<SrcType, WeiType, DstType, BiaType>::GenerateMLoopBody(std::uint32_t mLim,
    std::uint32_t nLim, std::uint32_t k)
{
    std::uint32_t vectorLength = svcntw();
    std::uint32_t nResid = Math::CeilDiv(nLim % (KERNEL_N_SIZE * vectorLength), vectorLength) * NUM_2;
    if (nResid == 0) {
        nResid = KERNEL_N_SIZE * NUM_2;
    }
    Mov(nLoopCtr, nVal);
    Sub(tmpNMinusNLoopCtr, nVal, nLoopCtr);
    if (nLim > (KERNEL_N_SIZE * vectorLength)) {
        ARMCG::Label nLoopHead;
        L(nLoopHead);
        {
            GenerateNLoopBody(mLim, KERNEL_N_SIZE * NUM_2, k);
            Sub(nLoopCtr, nLoopCtr, KERNEL_N_SIZE * vectorLength);
            Cmp(nLoopCtr, KERNEL_N_SIZE * vectorLength);
            Sub(tmpNMinusNLoopCtr, nVal, nLoopCtr);
        }
        B(ARMCG::Cond::GT, nLoopHead);
    }
    GenerateNLoopBody(mLim, nResid, k);
}

template <typename SrcType, typename WeiType, typename DstType, typename BiaType>
void VCSU8MmlaGen<SrcType, WeiType, DstType, BiaType>::GenerateNLoopBody(std::uint32_t mLim,
    std::uint32_t nLim, std::uint32_t k)
{
    for (std::uint32_t mIdx = 0; mIdx < Math::CeilDiv(mLim, NUM_2); ++mIdx) {
        for (std::uint32_t nIdx = 0; nIdx < nLim; ++nIdx) {
            Mov(ARMCG::ZRegB(GetDstZRegIdx(mIdx, nIdx)), 0);
        }
    }
    Mov(tmp, 0);
    for (std::uint32_t nIdx = 0; nIdx < Math::CeilDiv(nLim, NUM_2); ++nIdx) {
        Whilelt(ARMCG::PRegS(GetLoadStorePRegIdx(nIdx)), tmp, nLoopCtr);
        Incw(tmp);
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
        Ptrue(ARMCG::PRegB(NUM_0), ARMCG::Pattern::ALL);
        ARMCG::Label kLoopHead;
        L(kLoopHead);
        {
            Sub(kLoopCtr, kLoopCtr, KERNEL_K_STEP);
            Cmp(kLoopCtr, KERNEL_K_STEP);
            GenerateAccumulation(mLim, nLim, KERNEL_K_STEP);
        }
        B(ARMCG::Cond::GT, kLoopHead);
    }
    Whilelt(ARMCG::PRegB(NUM_0), xzr, kLoopCtr);
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
    StoreToDst(mLim, nLim);
}

template <typename SrcType, typename WeiType, typename DstType, typename BiaType>
void VCSU8MmlaGen<SrcType, WeiType, DstType, BiaType>::Mmla(const ARMCG::ZRegB &zsource,
    const ARMCG::ZRegB &zweight, const ARMCG::ZRegS &zdest)
{
    if constexpr (std::is_same_v<SrcType, WeiType>) {
        if constexpr (std::is_same_v<SrcType, std::int8_t>) {
            Smmla(zdest, zsource, zweight);
        } else {
            Ummla(zdest, zsource, zweight);
        }
    } else {
        if (std::is_same_v<SrcType, std::int8_t> == wereSrcWeiSwapped) {
            Usmmla(zdest, zsource, zweight);
        } else {
            // this gives dst in transposed format
            Usmmla(zdest, zweight, zsource);
        }
    }
}

template <typename SrcType, typename WeiType, typename DstType, typename BiaType>
void VCSU8MmlaGen<SrcType, WeiType, DstType, BiaType>::GenerateAccumulation(
    std::uint32_t mLim, std::uint32_t nLim, std::uint32_t kLim)
{
    // division by 2 isdue to peculiarities of MMLA instruciton (two lines are processed at once)
    for (std::uint32_t mIdx = 0; mIdx < Math::CeilDiv(mLim, NUM_2); ++mIdx) {
        Ld1rqb(ARMCG::ZRegB(GetTmpSrcZRegIdx(mIdx, NUM_0)), ARMCG::PRegB(NUM_0),
            Ptr(ARMCG::XReg(GetSrcXRegIdx(mIdx * NUM_2 + NUM_0))));
        // propagate by size of quadword
        Add(ARMCG::XReg(GetSrcXRegIdx(mIdx * NUM_2 + NUM_0)),
            ARMCG::XReg(GetSrcXRegIdx(mIdx * NUM_2 + NUM_0)), NUM_16);
        if ((mIdx * NUM_2 + NUM_1) < mLim) {
            Ld1rqb(ARMCG::ZRegB(GetTmpSrcZRegIdx(mIdx, NUM_1)), ARMCG::PRegB(NUM_0),
                Ptr(ARMCG::XReg(GetSrcXRegIdx(mIdx * NUM_2 + NUM_1))));
            // propagate by size of quadword
            Add(ARMCG::XReg(GetSrcXRegIdx(mIdx * NUM_2 + NUM_1)),
                ARMCG::XReg(GetSrcXRegIdx(mIdx * NUM_2 + NUM_1)), NUM_16);
        }
        Trn1(ARMCG::ZRegD(GetSrcZRegIdx(mIdx, NUM_0)), ARMCG::ZRegD(GetTmpSrcZRegIdx(mIdx, NUM_0)),
            ARMCG::ZRegD(GetTmpSrcZRegIdx(mIdx, NUM_1)));
        Trn2(ARMCG::ZRegD(GetSrcZRegIdx(mIdx, NUM_1)), ARMCG::ZRegD(GetTmpSrcZRegIdx(mIdx, NUM_0)),
            ARMCG::ZRegD(GetTmpSrcZRegIdx(mIdx, NUM_1)));
    }

    for (std::uint32_t kIdx = 0; kIdx < Math::CeilDiv(kLim, KERNEL_K_STEP / K_LOOP_UNROLL); ++kIdx) {
        for (std::uint32_t nIdx = 0; nIdx < nLim; ++nIdx) {
            Ld1b(ARMCG::ZRegB(GetWeiZRegIdx(nIdx)), p5,
                Ptr(ARMCG::XReg(GetWeiXRegIdx()), nIdx, ARMCG::ExtMod::MUL_VL));
            for (std::uint32_t mIdx = 0; mIdx < Math::CeilDiv(mLim, NUM_2); ++mIdx) {
                Mmla(ARMCG::ZRegB(GetSrcZRegIdx(mIdx, kIdx)),
                    ARMCG::ZRegB(GetWeiZRegIdx(nIdx)),
                    ARMCG::ZRegS(GetDstZRegIdx(mIdx, nIdx)));
            }
        }
        Addvl(ARMCG::XReg(GetWeiXRegIdx()), ARMCG::XReg(GetWeiXRegIdx()), NUM_8);
    }
}

template <typename SrcType, typename WeiType, typename DstType, typename BiaType>
MatMulImplGenerator<VCSU8MmlaGen<SrcType, WeiType, DstType, BiaType>>::DefaultImpl::DefaultImpl(
    const GemmInfo &task, IntType nThreads) : MatMulImplBase(task, nThreads)
{
    UpdateExecParams();
    Initialize();
}

template <typename SrcTy, typename WeiTy, typename DstTy, typename BiaTy>
constexpr std::uint32_t MatMulImplGenerator<VCSU8MmlaGen<SrcTy, WeiTy, DstTy, BiaTy>>::DefaultImpl::GetKernelMSize()
    noexcept
{
    return VCSU8MmlaGen<SrcTy, WeiTy, DstTy, BiaTy>::KERNEL_M_SIZE;
}

template <typename SrcTy, typename WeiTy, typename DstTy, typename BiaTy>
constexpr std::uint32_t MatMulImplGenerator<VCSU8MmlaGen<SrcTy, WeiTy, DstTy, BiaTy>>::DefaultImpl::GetKernelNSize()
    noexcept
{
    return VCSU8MmlaGen<SrcTy, WeiTy, DstTy, BiaTy>::KERNEL_N_SIZE;
}

template <typename SrcTy, typename WeiTy, typename DstTy, typename BiaTy>
constexpr std::uint32_t MatMulImplGenerator<VCSU8MmlaGen<SrcTy, WeiTy, DstTy, BiaTy>>::DefaultImpl::GetKernelKSize()
    noexcept
{
    return VCSU8MmlaGen<SrcTy, WeiTy, DstTy, BiaTy>::KERNEL_K_STEP /
        VCSU8MmlaGen<SrcTy, WeiTy, DstTy, BiaTy>::K_LOOP_UNROLL;
}

template <typename SrcType, typename WeiType, typename DstType, typename BiaType>
void MatMulImplGenerator<VCSU8MmlaGen<SrcType, WeiType, DstType, BiaType>>::DefaultImpl::InitializePackingPolicy()
{
    if (((dstLayout2D == Layout::ROW_MAJOR) && (broadcastedStridesSrc[IDX_3] < broadcastedStridesSrc[IDX_4])) ||
        ((dstLayout2D == Layout::COL_MAJOR) && (broadcastedStridesWei[IDX_3] >= broadcastedStridesWei[IDX_4]))) {
        packingPolicy = PackingPolicy::SYNC_DST_DIM_SRC;
    } else {
        packingPolicy = PackingPolicy::NO_SYNC;
    }
}

template <typename SrcType, typename WeiType, typename DstType, typename BiaType>
IntType MatMulImplGenerator<VCSU8MmlaGen<SrcType, WeiType, DstType, BiaType>>::DefaultImpl::GetMBlockSize()
    const noexcept
{
    return 5100; // is experimentally measured constant to obtain better perf
}

template <typename SrcType, typename WeiType, typename DstType, typename BiaType>
IntType MatMulImplGenerator<VCSU8MmlaGen<SrcType, WeiType, DstType, BiaType>>::DefaultImpl::GetNBlockSize()
    const noexcept
{
    return 128; // is experimentally measured constant to obtain better perf
}

template <typename SrcType, typename WeiType, typename DstType, typename BiaType>
IntType MatMulImplGenerator<VCSU8MmlaGen<SrcType, WeiType, DstType, BiaType>>::DefaultImpl::GetKBlockSize()
    const noexcept
{
    return 8192; // is experimentally measured constant to obtain better perf
}

template <typename SrcType, typename WeiType, typename DstType, typename BiaType>
void MatMulImplGenerator<VCSU8MmlaGen<SrcType, WeiType, DstType, BiaType>>::DefaultImpl::InitializeSingleCodegen(
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
        CodeGenParams<VCSU8MmlaGen<SrcType, WeiType, DstType, BiaType>> currTask{
            currentM, currentN, currentK, wereSrcWeiSwapped, attributes};
        if (codegens.find(currTask) == codegens.end()) {
            codegens.insert({currTask, std::make_shared<VCSU8MmlaGen<SrcType, WeiType, DstType, BiaType>>(
                currentM, currentN, currentK, wereSrcWeiSwapped, attributes)});
        }
        codegens[currTask]->Ready();
        matmulCodegenKernels[id] =
            codegens[currTask]->template GetCode<VCSU8MmlaCodegenFuncPtr<SrcType, WeiType, DstType, BiaType>>();
    }
}

template <typename SrcType, typename WeiType, typename DstType, typename BiaType>
void MatMulImplGenerator<VCSU8MmlaGen<SrcType, WeiType, DstType, BiaType>>::DefaultImpl::InitializeReorderCodeGens(
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
void MatMulImplGenerator<VCSU8MmlaGen<SrcType, WeiType, DstType, BiaType>>::DefaultImpl::InitializeThreadsCountImpl()
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
        threadNumber = InitThreadNum();
    }
    nThreadsExec = std::min(threadNumber, implNThreads);
}

template <typename SrcType, typename WeiType, typename DstType, typename BiaType>
IntType MatMulImplGenerator<VCSU8MmlaGen<SrcType, WeiType, DstType, BiaType>>::DefaultImpl::GetThreadBufSize(
    IntType m, IntType n, IntType k, MatrixKind mKind) const noexcept
{
    Service::Unused(m);
    // since vlw passed here, we should adjust
    IntType vlb = svcntb();
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
void MatMulImplGenerator<VCSU8MmlaGen<SrcType, WeiType, DstType, BiaType>>::DefaultImpl::ReserveImplSpecificParams()
{}

using vepa = std::vector<std::pair<Shape, Shape>>;
template <typename SrcType, typename WeiType, typename DstType, typename BiaType>
vepa MatMulImplGenerator<VCSU8MmlaGen<SrcType, WeiType, DstType, BiaType>>::DefaultImpl::SplitWork()
{
    return CalculateWorkOnThread<DefaultImpl>(taskExec.dstInfo.GetDims(), nThreadsExec, dstLayout2D == Layout::BA);
}

template <typename SrcType, typename WeiType, typename DstType, typename BiaType>
void MatMulImplGenerator<VCSU8MmlaGen<SrcType, WeiType, DstType, BiaType>>::DefaultImpl::PackBCol(
    const WeiType* B_ptr, WeiType* B_buffer, IntType N, IntType K, IntType ldb)
{
    IntType vlb = svcntb();
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
void MatMulImplGenerator<VCSU8MmlaGen<SrcType, WeiType, DstType, BiaType>>::DefaultImpl::PackBRow(
    const WeiType* B_ptr, WeiType* B_buffer, IntType N, IntType K, IntType ldb)
{
    IntType vlb = svcntb();
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
void MatMulImplGenerator<VCSU8MmlaGen<SrcType, WeiType, DstType, BiaType>>::DefaultImpl::RunSequential2D(
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
void MatMulImplGenerator<VCSU8MmlaGen<SrcType, WeiType, DstType, BiaType>>::DefaultImpl::UpdateExecParams() noexcept
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
        VCSU8MmlaGen<SrcType, WeiType, DstType, BiaType>::GetSrcType(),
        VCSU8MmlaGen<SrcType, WeiType, DstType, BiaType>::GetWeiType(),
        VCSU8MmlaGen<SrcType, WeiType, DstType, BiaType>::GetDstType(),
        (implTask.biaInfo.GetTotalTensorSize() != 0) ? VCSU8MmlaGen<SrcType, WeiType, DstType, BiaType>::GetBiaType() :
            static_cast<Element::TypeT>(implTask.biaInfo.GetType()));
}

template <typename SrcType, typename WeiType, typename DstType, typename BiaType>
MatMulDstFlags MatMulImplGenerator<VCSU8MmlaGen<SrcType, WeiType, DstType, BiaType>>::DefaultImpl::SetBiaFlagsExec(
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
    <ARCH::VC, std::int8_t, std::int8_t, std::int32_t, std::int32_t, MatMulImplKinds::SU8SU8S32S32_VC_MMLA>>;
template struct MatMulImplGenerator<MatMulKernelDefaultCodeGen
    <ARCH::VC, std::int8_t, std::uint8_t, std::int32_t, std::int32_t, MatMulImplKinds::SU8SU8S32S32_VC_MMLA>>;
template struct MatMulImplGenerator<MatMulKernelDefaultCodeGen
    <ARCH::VC, std::uint8_t, std::int8_t, std::int32_t, std::int32_t, MatMulImplKinds::SU8SU8S32S32_VC_MMLA>>;
template struct MatMulImplGenerator<MatMulKernelDefaultCodeGen
    <ARCH::VC, std::uint8_t, std::uint8_t, std::int32_t, std::int32_t, MatMulImplKinds::SU8SU8S32S32_VC_MMLA>>;

} // Detail
} // KuDNN
