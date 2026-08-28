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
#include "kernels/reorder/sve/bf16any32/kudnn_fast_jit_impl_bfp16any32_bfp16any32.hpp"
#include "kudnn_wrapper_helpers.hpp"
#include "kernels/matmul_jit/neon/kudnn_gemv_impl_f32f32f32f32_ba.hpp"


namespace KuDNN {
namespace Detail {

using F32NeonKernelSkinnyBAGen = MatMulKernelSkinnyBACodeGen<ARCH::NEON, float, float, float, float,
    MatMulImplKinds::F32F32F32F32_NEON>;

using F32NeonKernelSkinnyBAGenBase = MatMulKernelCodeGenBase<ARCH::NEON, float, float, float, float>;

F32NeonKernelSkinnyBAGen::MatMulKernelSkinnyBACodeGen(
    IntType m, IntType n, IntType k,
    IntType srcOffsetArg,
    IntType weiOffsetArg,
    IntType dstOffsetArg,
    IntType biaOffsetArg, const Attributes &attributesArg, bool hasAlphaArg)
    : F32NeonKernelSkinnyBAGenBase(attributesArg), hasAlpha(hasAlphaArg)
{
    weiOffsetValue = weiOffsetArg;
    if ((m > 0) && (n > 0) && (k > 0)) {
        if (attributes.GetPostOps().Len() > 0) {
            // set reserved dst registers
            for (std::uint32_t idx = SRC_V_REGS_OFFSET + SRC_V_REGS_NUM;
                idx < (DST_V_REGS_OFFSET + DST_V_REGS_NUM); ++idx) {
                TakeVectorRegister(idx);
            }
            // set reserved scalar registers
            for (std::uint32_t idx = 0; idx < NUM_29; ++idx) {
                TakeScalarRegister(idx);
            }
        }
        Preamble();
        Mov(alpha, ARMCG::VReg4S(NUM_1)[0]);  // alpha in s1 by ABI, store it in W16
        Mov(beta, ARMCG::VReg4S(NUM_0)[0]);  // beta in s0 by ABI, store it in W16
        Mov(mVal, m);
        Mov(kVal, k);
        Mov(srcOffset, srcOffsetArg * sizeof(float));   // in bytes
        Mov(weiOffset, weiOffsetArg * sizeof(float));   // in bytes
        Mov(dstOffset, dstOffsetArg * sizeof(float));   // in bytes
        Mov(biaOffset, biaOffsetArg * sizeof(float));   // in bytes
        Mov(dstPtrStore, dstPtrLoad);

        std::uint32_t mResid = Math::CeilDiv(m % (KERNEL_M_SIZE * NEON_VL), NEON_VL);
        std::uint32_t mResidScalar = m % NEON_VL;
        if (mResidScalar == 0) {
            mResidScalar = NEON_VL;
        }
        if (mResid == 0) {
            mResid = KERNEL_M_SIZE;
        }

        Mov(mLoopCtr, mVal);
        if (m > KERNEL_M_SIZE * NEON_VL) {
            ARMCG::Label mLoopHead;
            L(mLoopHead);
            {
                GenerateMLoopBody(KERNEL_M_SIZE, k, NEON_VL);
                Add(srcPtr, srcPtr, KERNEL_M_SIZE * NEON_VL * sizeof(float));
                Sub(mLoopCtr, mLoopCtr, KERNEL_M_SIZE * NEON_VL);
                Cmp(mLoopCtr, KERNEL_M_SIZE * NEON_VL);
            }
            B(ARMCG::Cond::GT, mLoopHead);
        }
        GenerateMLoopBody(mResid, k, mResidScalar);

        Postamble();
    }
    Ret();
}

void F32NeonKernelSkinnyBAGen::InitializeDstRegisters(
    std::uint32_t mLim, std::uint32_t mResidScalar)
{
    ARMCG::Label loadCNeonSkinny;
    ARMCG::Label mullCNeonSkinny;
    ARMCG::Label loadBias1NeonSkinny;
    ARMCG::Label loadBiasMNeonSkinny;
    ARMCG::Label loadBiasNNeonSkinny;
    ARMCG::Label loadBiasMNNeonSkinny;
    ARMCG::Label addBias1NeonSkinny;
    ARMCG::Label addBiasMNeonSkinny;
    ARMCG::Label addBiasNNeonSkinny;
    ARMCG::Label addBiasMNNeonSkinny;
    ARMCG::Label initDoneNeonSkinny;
    ARMCG::Label accumulateWithDstSkinny;
    ARMCG::Label init1N;
    ARMCG::Label initMN;
    ARMCG::Label add1N;
    ARMCG::Label addMN;

    Tbz(flags, 0, loadBias1NeonSkinny); // 0-bit correspond to FILL_C_WITH_ZEROZ
    // Fill Accum with zeros by default
    for (std::uint32_t mIdx = 0; mIdx < mLim; ++mIdx) {
        Movi(ARMCG::VReg4S(GetDstVRegIdx(mIdx)), 0);
    }
    L(loadBias1NeonSkinny);
    Tbz(flags, 3, loadBiasNNeonSkinny); // 3-bit correspond to LOAD_BIAS_1
    B(init1N);
    L(loadBiasNNeonSkinny);
    Tbz(flags, 5, loadBiasMNeonSkinny); // 5-bit correspond to LOAD_BIAS_N
    L(init1N);
    for (std::uint32_t mIdx = 0; mIdx < mLim; ++mIdx) {
        CodeGenerator::Ld1r(ARMCG::VReg4SList(ARMCG::VReg4S(GetDstVRegIdx(mIdx))), Ptr(biaPtr));
    }
    B(initDoneNeonSkinny);
    L(loadBiasMNeonSkinny);
    Tbz(flags, 4, loadBiasMNNeonSkinny); // 4-bit correspond to LOAD_BIAS_M
    B(initMN);
    L(loadBiasMNNeonSkinny);
    Tbz(flags, 6, loadCNeonSkinny); // 6-bit correspond to LOAD_BIAS_MN
    L(initMN);
    Mov(tmp, NEON_VL);
    Mul(tmp, biaOffset, tmp);
    for (std::uint32_t mIdx = 0; mIdx < mLim; ++mIdx) {
        if ((mResidScalar != NEON_VL) && (mIdx == (mLim - 1))) {
            for (IntType j = 0; j < mResidScalar; j++) {
                CodeGenerator::Ld1(ARMCG::VReg4S(GetDstVRegIdx(mIdx))[j], PostPtr(biaPtr, biaOffset));
            }
        } else {
            CodeGenerator::Ld1(ARMCG::VReg4SList(ARMCG::VReg4S(GetDstVRegIdx(mIdx))),
                PostPtr(biaPtr, tmp));
        }
    }
    B(accumulateWithDstSkinny);
    L(loadCNeonSkinny);
    Tbz(flags, 1, mullCNeonSkinny); // 1-bit correspond to LOAD_C
    Mov(tmp, NEON_VL);
    Mul(tmp, dstOffset, tmp);
    for (std::uint32_t mIdx = 0; mIdx < mLim; ++mIdx) {
        if ((mResidScalar != NEON_VL) && (mIdx == (mLim - 1))) {
            for (IntType j = 0; j < mResidScalar; j++) {
                CodeGenerator::Ld1(ARMCG::VReg4S(GetDstVRegIdx(mIdx))[j], PostPtr(dstPtrLoad, dstOffset));
            }
        } else {
            CodeGenerator::Ld1(ARMCG::VReg4SList(ARMCG::VReg4S(GetDstVRegIdx(mIdx))),
                PostPtr(dstPtrLoad, tmp));
        }
    }
    L(mullCNeonSkinny);
    Tbz(flags, 2, addBias1NeonSkinny); // 2-bit correspond to MULL_C
    Dup(ARMCG::VReg4S(NUM_0), beta); // mov beta from W to V4S
    for (std::uint32_t mIdx = 0; mIdx < mLim; ++mIdx) {
        Fmul(ARMCG::VReg4S(GetDstVRegIdx(mIdx)),
            ARMCG::VReg4S(GetDstVRegIdx(mIdx)),
            ARMCG::VReg4S(NUM_0)); // s0 contains beta value
    }
    L(addBias1NeonSkinny);
    Tbz(flags, 7, addBiasNNeonSkinny); // 7-bit correspond to Add_BIAS_1
    B(add1N);
    L(addBiasNNeonSkinny);
    Tbz(flags, 9, addBiasMNeonSkinny); // 9-bit correspond to Add_BIAS_N
    L(add1N);
    CodeGenerator::Ld1r(ARMCG::VReg4SList(ARMCG::VReg4S(NUM_3)), Ptr(biaPtr));
    for (std::uint32_t mIdx = 0; mIdx < mLim; ++mIdx) {
        Fadd(ARMCG::VReg4S(GetDstVRegIdx(mIdx)),
             ARMCG::VReg4S(GetDstVRegIdx(mIdx)),
             ARMCG::VReg4S((NUM_3)));
    }
    L(addBiasMNeonSkinny);
    Tbz(flags, 8, addBiasMNNeonSkinny); // 8-bit correspond to ADD_BIAS_M
    B(addMN);
    L(addBiasMNNeonSkinny);
    Tbz(flags, 10, accumulateWithDstSkinny); // 10-bit correspond to ADD_BIAS_MN
    L(addMN);
    Mov(tmp, NEON_VL);
    Mul(tmp, biaOffset, tmp);
    // dst and acum are busy, load to src as tmp
    for (std::uint32_t mIdx = 0; mIdx < mLim; ++mIdx) {
        if ((mResidScalar != NEON_VL) && (mIdx == (mLim - 1))) {
            for (IntType j = 0; j < mResidScalar; j++) {
                CodeGenerator::Ld1(ARMCG::VReg4S(GetSrcVRegIdx(0, mIdx))[j], PostPtr(biaPtr, biaOffset));
            }
        } else {
            CodeGenerator::Ld1(ARMCG::VReg4SList(ARMCG::VReg4S(GetSrcVRegIdx(0, mIdx))),
                PostPtr(biaPtr, tmp));
        }
        Fadd(ARMCG::VReg4S(GetDstVRegIdx(mIdx)), ARMCG::VReg4S(GetDstVRegIdx(mIdx)),
            ARMCG::VReg4S(GetSrcVRegIdx(0, mIdx)));
    }
    L(accumulateWithDstSkinny);
    L(initDoneNeonSkinny);

    // accumulate Dst regs with Accum regs
    for (std::uint32_t mIdx = 0; mIdx < mLim; ++mIdx) {
        CodeGenerator::Fadd(ARMCG::VReg4S(GetAccumVRegIdx(mIdx)),
                            ARMCG::VReg4S(GetAccumVRegIdx(mIdx)),
                            ARMCG::VReg4S(GetDstVRegIdx(mIdx)));
    }
}

void F32NeonKernelSkinnyBAGen::ApplyEltwise(const PostOps &postOps, SizeType poIdx, const std::vector<SizeType> &regIDs,
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
        // all vector elements will used to store
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

void F32NeonKernelSkinnyBAGen::GenerateMLoopBody(std::uint32_t mLim, std::uint32_t k, std::uint32_t mResidScalar)
{
    Mov(tmpWeiPtr, weiPtr);

    std::uint32_t kResid = k % K_LOOP_UNROLL;
    if (kResid == 0) {
        kResid = K_LOOP_UNROLL;
    }
    std::uint32_t kResidScalar = k % NEON_VL;
    if (kResidScalar == 0) {
        kResidScalar = NEON_VL;
    }

    // fill accum regs with zeros by default
    for (std::uint32_t mIdx = 0; mIdx < mLim; ++mIdx) {
        Movi(ARMCG::VReg4S(GetAccumVRegIdx(mIdx)), 0);
    }

    // Main m loop
    Mov(kLoopCtr, kVal);
    Mov(tmpKMinusKLoopCtr, 0);
    if (k > K_LOOP_UNROLL) {
        ARMCG::Label kLoopHead;
        L(kLoopHead);
        {
            Sub(kLoopCtr, kLoopCtr, K_LOOP_UNROLL);
            Cmp(kLoopCtr, K_LOOP_UNROLL);
            GenerateKLoopBody(mLim, K_LOOP_UNROLL, mResidScalar, kResidScalar);
            Sub(tmpKMinusKLoopCtr, kVal, kLoopCtr);
        }
        B(ARMCG::Cond::GT, kLoopHead);
    }
    GenerateKLoopBody(mLim, kResid, mResidScalar, kResidScalar);

    // accum mul on alpha
    if (hasAlpha) {
        for (std::uint32_t mIdx = 0; mIdx < mLim; ++mIdx) {
            Mov(ARMCG::VReg4S(NUM_1)[0], alpha);
            Fmul(ARMCG::VReg4S(GetAccumVRegIdx(mIdx)), ARMCG::VReg4S(GetAccumVRegIdx(mIdx)),
                ARMCG::VReg4S(NUM_1)[0]); // s1 is alpha
        }
    }

    InitializeDstRegisters(mLim, mResidScalar);

    ARMCG::Label skipPO;
    Tbz(flags, 11, skipPO); // 11-bit correspond to APPLY_POST_OPS
    std::vector<SizeType> dstRegsIDs(mLim);
    for (std::uint32_t mIdx = 0; mIdx < mLim; ++mIdx) {
        dstRegsIDs[mIdx] = GetAccumVRegIdx(mIdx);
    }
    this->ApplyPostOps(dstRegsIDs, Element::MatchType<float>());
    L(skipPO);

    // store dst
    Mov(tmp, NEON_VL);
    Mul(tmp, dstOffset, tmp);
    for (std::uint32_t mIdx = 0 ; mIdx < mLim; ++mIdx) {
        // store scalar resid by M
        if ((mResidScalar != NEON_VL) && (mIdx == (mLim - 1))) {
            for (IntType j = 0; j < mResidScalar; j++) {
                CodeGenerator::St1(ARMCG::VReg4S(GetAccumVRegIdx(mIdx))[j],
                    PostPtr(dstPtrStore, dstOffset));
            }
        } else {
            CodeGenerator::St1(ARMCG::VReg4SList(ARMCG::VReg4S(GetAccumVRegIdx(mIdx))),
                PostPtr(dstPtrStore, tmp));
        }
    }
}

void F32NeonKernelSkinnyBAGen::GenerateKLoopBody(std::uint32_t mLim, std::uint32_t kLim,
    std::uint32_t mResidScalar, std::uint32_t kResidScalar)
{
    Service::Unused(mResidScalar);
    // Src: initialize pointers
    Mul(tmp, tmpKMinusKLoopCtr, srcOffset);
    Add(ARMCG::XReg(GetSrcXRegIdx(0)), srcPtr, tmp);
    for (std::uint32_t kIdx = 1; kIdx < kLim; ++kIdx) {
        Add(ARMCG::XReg(GetSrcXRegIdx(kIdx)),
            ARMCG::XReg(GetSrcXRegIdx(kIdx - 1)),
            srcOffset);
    }

    // load B , twice
    for (std::uint32_t kIdx = 0; kIdx < kLim / NEON_VL; ++kIdx) {
        if (weiOffsetValue > NUM_1) {
            CodeGenerator::Movi(ARMCG::VReg4S(GetWeiVRegIdx(kIdx)), 0);
            for (IntType j = 0; j < NEON_VL; j++) {
                CodeGenerator::Ld1(ARMCG::VReg4S(GetWeiVRegIdx(kIdx))[j],
                    PostPtr(tmpWeiPtr, weiOffset));
            }
        } else {
            CodeGenerator::Ld1(ARMCG::VReg4SList(ARMCG::VReg4S(GetWeiVRegIdx(kIdx))),
                PostPtr(tmpWeiPtr, NEON_VL * sizeof(float)));
        }
    }
    // process tail by K
    if ((kResidScalar != NEON_VL) && ((kLim / NEON_VL) < NUM_2)) {
        CodeGenerator::Movi(ARMCG::VReg4S(GetWeiVRegIdx(((kLim / NEON_VL == 0) ? 0 : 1))), 0);
        for (IntType j = 0; j < kResidScalar; j++) {
            CodeGenerator::Ld1(ARMCG::VReg4S(GetWeiVRegIdx(((kLim / NEON_VL == 0) ? 0 : 1)))[j],
                PostPtr(tmpWeiPtr, weiOffset));     // use 1 Pointer, load and propagate
        }
    }
    // load A,  fmla
    for (std::uint32_t kIdx = 0; kIdx < kLim; ++kIdx) {
        for (std::uint32_t mIdx = 0; mIdx < mLim ; ++mIdx) {
            if ((mResidScalar != NEON_VL) && (mIdx == (mLim - 1))) {
                for (IntType j = 0; j < mResidScalar; j++) {
                    CodeGenerator::Ld1(ARMCG::VReg4S(GetSrcVRegIdx(mIdx, kIdx))[j],
                        PostPtr(ARMCG::XReg(GetSrcXRegIdx(kIdx)), sizeof(float)));
                }
            } else {
                CodeGenerator::Ld1(ARMCG::VReg4SList(ARMCG::VReg4S(GetSrcVRegIdx(mIdx, kIdx))),
                    PostPtr(ARMCG::XReg(GetSrcXRegIdx(kIdx)), NEON_VL * sizeof(float)));
            }

            Fmla(ARMCG::VReg4S(GetAccumVRegIdx(mIdx)),
                ARMCG::VReg4S(GetSrcVRegIdx(mIdx, kIdx)),
                ARMCG::VReg4S(GetWeiVRegIdx(kIdx / NEON_VL))[kIdx % NEON_VL]);
        }
    }
}

MatMulImplGenerator<F32NeonKernelSkinnyBAGen>::DefaultImpl::DefaultImpl(
    const GemmInfo &task, IntType nThreads) : MatMulImplBase(task, nThreads)
{
    VERBOSE_PRINT;
    UpdateExecParams();
    Initialize();
}

constexpr std::uint32_t MatMulImplGenerator<F32NeonKernelSkinnyBAGen>::DefaultImpl::GetKernelMSize() noexcept
{
    return CodeGen::KERNEL_M_SIZE;
}

std::uint32_t MatMulImplGenerator<F32NeonKernelSkinnyBAGen>::DefaultImpl::GetKernelNSize() noexcept
{
    return 1;
}

IntType MatMulImplGenerator<F32NeonKernelSkinnyBAGen>::DefaultImpl::GetMBlockSize()
    const noexcept
{
    return M_BLOCK_SIZE; // is experimentally measured constant to obtain better perf
}

IntType MatMulImplGenerator<F32NeonKernelSkinnyBAGen>::DefaultImpl::GetNBlockSize()
    const noexcept
{
    return 1;
}

IntType MatMulImplGenerator<F32NeonKernelSkinnyBAGen>::DefaultImpl::GetKBlockSize()
    const noexcept
{
    return K_BLOCK_SIZE; // is experimentally measured constant to obtain better perf
}

void MatMulImplGenerator<F32NeonKernelSkinnyBAGen>::DefaultImpl::InitializeCodeGens(
    IntType m, IntType n, IntType k,
    SizeType offsetSrc, SizeType offsetWei, SizeType offsetDst, SizeType offsetBia,
    bool wereSrcWeiSwapped, IntType mFull, IntType nFull, const Attributes &attributes)
{
    Service::Unused(nFull);
    IntType mBlockExecFull;
    IntType mBlockExecResid1;
    IntType mBlockExecResid2;
    InitializeKernelMorN(m, mFull, M_BLOCK_SIZE, mBlockExecFull, mBlockExecResid1, mBlockExecResid2);

    IntType kBlockExecFull;
    IntType kBlockExecResid;
    InitializeKernelK(k, K_BLOCK_SIZE, kBlockExecFull, kBlockExecResid);

    for (IntType i = 0; i < (NUM_3 * NUM_2 * NUM_2); ++i) {
        IntType currentM;
        InitializeCurrentMorN(i, mBlockExecFull, mBlockExecResid1, mBlockExecResid2, currentM);
        IntType currentN = NUM_1;
        IntType currentK = kBlockExecFull;
        if ((i / NUM_3) % NUM_2 == NUM_1) {
            currentK = kBlockExecResid;
        }
        bool hasAlpha = false;
        if ((i / NUM_3 / NUM_2) % NUM_2 == NUM_1) {
            hasAlpha = true;
        }
        InitializeSingleCodegen(i, m, n, k, currentM, currentN, currentK, offsetSrc, offsetWei, offsetDst, offsetBia,
            wereSrcWeiSwapped, attributes, hasAlpha);
    }
}

void MatMulImplGenerator<F32NeonKernelSkinnyBAGen>::DefaultImpl::InitializeSingleCodegen(
    IntType id, IntType m, IntType n, IntType k,
    IntType currentM, IntType currentN, IntType currentK,
    SizeType offsetSrc, SizeType offsetWei, SizeType offsetDst, SizeType offsetBia,
    bool wereSrcWeiSwapped, const Attributes &attributes, bool hasAlpha)
{
    Service::Unused(wereSrcWeiSwapped);
    if ((currentM <= m) && (currentN <= n) && (currentK <= k)) {
        matmulCodegenKernels[id].Initialize(currentM, currentN, currentK,
            static_cast<IntType>(offsetSrc), static_cast<IntType>(offsetWei),
            static_cast<IntType>(offsetDst), static_cast<IntType>(offsetBia),
            attributes, hasAlpha);
    }
}

void MatMulImplGenerator<F32NeonKernelSkinnyBAGen>::DefaultImpl::InitializeThreadsCountImpl()
{
    IntType threadNumber = InitThreadNum();

    IntType size = GetRequiredMemSize();
    if (size < 130000) { // 130000 is practically measured number
        threadNumber = NUM_1;
    } else if (size < 200000) { // 200000 is practically measured number
        threadNumber = NUM_2;
    } else if (size < 250000) { // 250000 is practically measured number
        threadNumber = NUM_4;
    } else if (size < 300000) { // 300000 is practically measured number
        threadNumber = NUM_8;
    } else if (size < 400000) { // 400000 is practically measured number
        threadNumber = NUM_16;
    } else if (size < 8000000) { // 8000000 is practically measured number
        threadNumber = NUM_40;
    }
    nThreadsExec = std::min(threadNumber, implNThreads);
}
IntType MatMulImplGenerator<F32NeonKernelSkinnyBAGen>::DefaultImpl::GetThreadBufSize(
    IntType m, IntType n, IntType k, MatrixKind mKind) const noexcept
{
    Service::Unused(m, n, k, mKind);
    return 0;
}

void MatMulImplGenerator<F32NeonKernelSkinnyBAGen>::DefaultImpl::ReserveImplSpecificParams() {}

std::vector<std::pair<Shape, Shape>> MatMulImplGenerator<F32NeonKernelSkinnyBAGen>::DefaultImpl::SplitWork()
{
    return CalculateWorkOnThread<DefaultImpl>(taskExec.dstInfo.GetDims(), nThreadsExec, dstLayout2D == Layout::BA);
}

void MatMulImplGenerator<F32NeonKernelSkinnyBAGen>::DefaultImpl::RunSequential2D(
    IntType threadID, const void *src, const void *wei, void *dst, const void *bia,
    std::unique_ptr<PackStrategy> &srcPackStrategy, std::unique_ptr<PackStrategy> &weiPackStrategy,
    float alpha, float beta, const PostOpsDataPtrs &shiftedByBatchPOPtrs) const noexcept(false)
{
    VERBOSE_PRINT;
    Service::Unused(srcPackStrategy, weiPackStrategy, shiftedByBatchPOPtrs, alpha);
    IntType alphaIdxKernel = (alpha == 1.0f) ? 0 : 1;
    for (IntType mBlock = 0; mBlock < perThreadParams[threadID].m; mBlock += M_BLOCK_SIZE) {
        auto mBlockExec = std::min(M_BLOCK_SIZE,  perThreadParams[threadID].m - mBlock);
        IntType mIdxKernel = ((mBlockExec == M_BLOCK_SIZE) ? 0 : 1) * (perThreadParams[threadID].isResidByM + 1);
        for (IntType kBlock = 0; kBlock < perThreadParams[threadID].k; kBlock += K_BLOCK_SIZE) {
            auto kBlockExec = std::min(K_BLOCK_SIZE, perThreadParams[threadID].k - kBlock);
            IntType kIdxKernel = (kBlockExec == K_BLOCK_SIZE) ? 0 : 1;
            bool lastKIter = (perThreadParams[threadID].k - kBlock) <= K_BLOCK_SIZE;
            for (IntType nBlock = 0; nBlock < perThreadParams[threadID].n; nBlock += NUM_1) {
                typename CodeGen::DstType *dstExec = static_cast<typename CodeGen::DstType *>(dst) + nBlock +
                    mBlock * perThreadParams[threadID].dstOffset2D;
                const typename CodeGen::BiaType *biaExec = static_cast<const typename CodeGen::BiaType *>(bia) +
                    nBlock * biaNOffset + mBlock * perThreadParams[threadID].biaOffset2D;
                const typename CodeGen::SrcType *srcExec = static_cast<const typename CodeGen::SrcType *>(src) +
                    (kBlock * perThreadParams[threadID].srcOffset2D) + mBlock;
                MatMulDstFlags biaFlagsExec = SetBiaFlagsExec(biaFlags, beta, kBlock, lastKIter);
                const typename CodeGen::WeiType *weiExec = static_cast<const typename CodeGen::WeiType *>(wei) +
                    nBlock * perThreadParams[threadID].weiOffset2D + kBlock;
                if (perThreadParams[threadID].weiLayout2D == Layout::AB) {
                    weiExec = static_cast<const typename CodeGen::WeiType *>(wei) +
                        nBlock + kBlock * perThreadParams[threadID].weiOffset2D;
                }

                auto kernel = matmulCodegenKernels[
                    alphaIdxKernel * NUM_6 + kIdxKernel * NUM_3 + mIdxKernel].template GetCode<void (*)(
                    const typename CodeGen::SrcType *, const typename CodeGen::WeiType *, typename CodeGen::DstType *,
                    const typename CodeGen::BiaType *, MatMulDstFlags, float, float)>();
                kernel(srcExec, weiExec, dstExec, biaExec, biaFlagsExec, beta, alpha);
            }
        }
    }
}

void MatMulImplGenerator<F32NeonKernelSkinnyBAGen>::DefaultImpl::UpdateExecParams() noexcept
{
    VERBOSE_PRINT;
    Layout desiredSrcLayout = implTask.srcInfo.GetLayout();
    Layout desiredWeiLayout = implTask.weiInfo.GetLayout();
    Layout desiredDstLayout = implTask.dstInfo.GetLayout();
    Layout desiredBiaLayout = implTask.biaInfo.GetLayout();

    // when one dim == 1, kernel will operate on SRC with AB layout
    if (implTask.dstInfo.GetDims()[implTask.dstInfo.GetNumDims() - IDX_1] == NUM_1) {
        desiredSrcLayout = GetPlaneLayoutBA(desiredSrcLayout);
        desiredDstLayout = GetPlaneLayoutAB(desiredDstLayout);
    } else {
        // src and wei matricies swap
        desiredWeiLayout = GetPlaneLayoutAB(desiredWeiLayout);
        desiredDstLayout = GetPlaneLayoutBA(desiredDstLayout);
    }

    if (!IsLayoutAlikeAB(desiredBiaLayout) && !IsLayoutAlikeBA(desiredBiaLayout)) {
        if (IsLayoutAlikeAB(desiredDstLayout)) {
            desiredBiaLayout = GetPlaneLayoutBA(desiredBiaLayout);
        } else {
            desiredBiaLayout = GetPlaneLayoutAB(desiredBiaLayout);
        }
    }
    InitializeTaskExec(desiredSrcLayout, desiredWeiLayout, desiredDstLayout, desiredBiaLayout,
        CodeGen::GetSrcType(), CodeGen::GetWeiType(), CodeGen::GetDstType(),
        (implTask.biaInfo.GetTotalTensorSize() != 0) ? CodeGen::GetBiaType() :
            static_cast<Element::TypeT>(implTask.biaInfo.GetType()));
}

} // Detail
} // KuDNN
