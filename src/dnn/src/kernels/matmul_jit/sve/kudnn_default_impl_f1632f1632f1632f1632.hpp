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

#ifndef KUDNN_MATMUL_JIT_VC_DEFAUT_IMPL_F1632F1632F1632F1632_HPP
#define KUDNN_MATMUL_JIT_VC_DEFAUT_IMPL_F1632F1632F1632F1632_HPP

#include <memory>
#include <vector>
#include <array>
#include <type_traits>
#include <unordered_map>

#include <arm_sve.h>

#include "kudnn.hpp"

#include "cg_codegen.hpp"

#include "vector_wrappers.hpp"
#include "kudnn_jit.hpp"
#include "kernels/matmul_jit/kudnn_gemm_jit.hpp"
#include "kernels/matmul_jit/kudnn_gemm_helpers.hpp"
#include "kernels/matmul_jit/kudnn_gemm_threading_helpers.hpp"
#include "kernels/reorder/sve/bf16any32/kudnn_fast_jit_impl_bfp16any32_bfp16any32.hpp"
#include "kudnn_wrapper_helpers.hpp"
#include "internal_funcs.hpp"

#ifdef KUDNNL_920F
    #include "kernels/reorder/sme/bfp16/kudnn_fast_jit_impl_bfp16_bfp16.hpp"
#endif // KUDNNL_920F

namespace KuDNN {
namespace Detail {

template <typename FP1632Type>
struct MatMulKernelDefaultCodeGen<ARCH::VC, FP1632Type, FP1632Type, FP1632Type, FP1632Type,
    MatMulImplKinds::F1632F1632F1632F1632_VC> :
    MatMulKernelCodeGenBase<ARCH::VC, FP1632Type, FP1632Type, FP1632Type, FP1632Type> {
public:
    using TypeHelper = MatmulTypeHelper<FP1632Type, FP1632Type, FP1632Type, FP1632Type>;
    constexpr static const std::uint32_t KERNEL_M_SIZE = 6;
    constexpr static const std::uint32_t KERNEL_N_SIZE = 4; // in VL
    // indexed FMLA operates on indices inside 128 bit (16 bytes) of register
    constexpr static const std::uint32_t K_LOOP_UNROLL = NUM_16 / sizeof(FP1632Type);
    // Kernel performs 6x4VL update of destination matrix via
    // outer product computations inside loops over M, N, K. Registers z8-z31 are used for
    // accumulation (then will be stored to destination matrix),
    // registerrs z1, z0 are used to store matrix B values (reused to store 4VL),
    // registers z2 - z7 are used to store matrix A values.
    // The kernel has unroll by K variable ([m x k] * [k * n] -> [m * n]) by 4.
    //                                                 x17     x18     x19     x20
    //                                                  |       |       |       |
    //                                                  v       v       v       v
    //                                                  +-------+-------+-------+-------+
    //                     1st K iteration              | z1.s  | z0.s  | z1.s  | z0.s  |
    //                                                  +-------+-------+-------+-------+
    //          Rhs ->                                  |  ...  |  ...  |  ...  |  ...  |
    //                                                  +-------+-------+-------+-------|
    //                   4th or 8th K iteration         | z1.s  | z0.s  | z1.s  | z0.s  |
    //                                                  +-------+-------+-------+-------+
    //                   Lhs                            |       |       |       |       |
    //                    |
    //                    v                             |       |       |       |       |
    //            1st K           4th K
    //          +---------+-----+--------------+  -  -  +-------+-------+-------+-------+
    //  x26 ->  | z7.s[0] | ... | z7.s[3 or 7] |        | z8.s  | z9.s  | z10.s | z11.s |
    //  x25 ->  | z6.s[0] | ... | z6.s[3 or 7] |        | z12.s | z13.s | z14.s | z15.s |
    //  x24 ->  | z5.s[0] | ... | z5.s[3 or 7] |        | z16.s | z17.s | z18.s | z19.s |
    //  x23 ->  | z4.s[0] | ... | z4.s[3 or 7] |        | z20.s | z21.s | z22.s | z23.s |
    //  x22 ->  | z3.s[0] | ... | z3.s[3 or 7] |        | z24.s | z25.s | z26.s | z27.s |
    //  x21 ->  | z2.s[0] | ... | z2.s[3 or 7] |        | z28.s | z29.s | z30.s | z31.s |
    //          +---------+-----+--------------+  -  -  +-------+-------+-------+-------+
    // There are also kernels of smaller sizes to process borders of matrix (5x4VL, 4x4VL, 3x4VL, 2x4VL, 1x4VL).
    // They are very similar to main kernel.

    MatMulKernelDefaultCodeGen(IntType m, IntType n, IntType k,
        const Attributes &attributesArg);

    static bool CompareMSize(IntType lhs, IntType rhs) noexcept
    {
        if ((lhs <= KERNEL_M_SIZE) || (rhs <= KERNEL_M_SIZE)) {
            return lhs == rhs;
        } else {
            return ProcessMSize(lhs) == ProcessMSize(rhs);
        }
    }

    static IntType ProcessMSize(IntType m) noexcept
    {
        return m % KERNEL_M_SIZE;
    }

    static bool CompareNSize(IntType lhs, IntType rhs) noexcept
    {
        if ((lhs <= static_cast<IntType>(KERNEL_N_SIZE * Vector::Length<FP1632Type>())) ||
            (rhs <= static_cast<IntType>(KERNEL_N_SIZE * Vector::Length<FP1632Type>()))) {
            return Math::CeilDiv(lhs, Vector::Length<FP1632Type>()) == Math::CeilDiv(rhs, Vector::Length<FP1632Type>());
        } else {
            return ProcessNSize(lhs) == ProcessNSize(rhs);
        }
    }

    static IntType ProcessNSize(IntType n) noexcept
    {
        return Math::CeilDiv(n, Vector::Length<FP1632Type>()) % KERNEL_N_SIZE;
    }

    static bool CompareKSize(IntType lhs, IntType rhs) noexcept
    {
        if ((lhs <= K_LOOP_UNROLL) || (rhs <= K_LOOP_UNROLL)) {
            return lhs == rhs;
        } else {
            return ProcessKSize(lhs) == ProcessKSize(rhs);
        }
    }

    static IntType ProcessKSize(IntType k) noexcept
    {
        return k % K_LOOP_UNROLL;
    }

protected:

    void InitializeDstRegisters(std::uint32_t mLim, std::uint32_t nLim);
    void StoreToDst(std::uint32_t mLim, std::uint32_t nLim);

    void GenerateMLoopBody(std::uint32_t mLim, std::uint32_t nLim, std::uint32_t k);

    void GenerateNLoopBody(std::uint32_t mLim, std::uint32_t nLim, std::uint32_t k);

    void GenerateMLAPart(std::uint32_t mLim, std::uint32_t nLim, std::uint32_t kLim, bool isTail);

    static std::uint32_t GetDstZRegIdx(std::uint32_t m, std::uint32_t n) noexcept
    {
        return DST_Z_REG_OFFSET + (n + m * KERNEL_N_SIZE) % NUM_DST_Z_REGS;
    }

    static std::uint32_t GetDstXRegIdx(std::uint32_t m) noexcept
    {
        return DST_X_REG_OFFSET - m % NUM_DST_X_REGS;
    }

    static std::uint32_t GetSrcZRegIdx(std::uint32_t m) noexcept
    {
        return SRC_Z_REG_OFFSET - m % NUM_SRC_Z_REGS;
    }

    static std::uint32_t GetSrcXRegIdx(std::uint32_t m) noexcept
    {
        return SRC_X_REG_OFFSET - m % NUM_SRC_X_REGS;
    }

    static std::uint32_t GetWeiZRegIdx(std::uint32_t n) noexcept
    {
        return WEI_Z_REG_OFFSET - n % NUM_WEI_Z_REGS;
    }

    static std::uint32_t GetWeiXRegIdx(std::uint32_t n, std::uint32_t numUsedWeiRegs) noexcept
    {
        if (numUsedWeiRegs != (KERNEL_N_SIZE - 1)) {
            return WEI_X_REG_OFFSET + n % numUsedWeiRegs;
        } else {
            return WEI_X_REG_OFFSET + ((n < 2) ? n : (KERNEL_N_SIZE - n)); // separate logic for n < 2
        }
    }

    static std::uint32_t GetTmpXRegIdx(std::uint32_t n) noexcept
    {
        return TMP_X_REG_OFFSET + n % NUM_TMP_X_REGS;
    }

    static std::uint32_t GetLoadStorePRegIdx(std::uint32_t n) noexcept
    {
        return LOAD_STORE_P_REG_OFFSET + n % NUM_LOAD_STORE_P_REGS;
    }

    constexpr static const std::uint32_t DST_Z_REG_OFFSET = 8; // dst Z registers are z8 - z31
    constexpr static const std::uint32_t NUM_DST_Z_REGS = 24; // dst Z registers are z8 - z31
    constexpr static const std::uint32_t SRC_Z_REG_OFFSET = 7; // src Z registers are z7 - z2
    constexpr static const std::uint32_t NUM_SRC_Z_REGS = 6; // src Z registers are z7 - z2
    constexpr static const std::uint32_t WEI_Z_REG_OFFSET = 1; // wei Z registers are z0, z1
    constexpr static const std::uint32_t NUM_WEI_Z_REGS = 2; // wei Z registers are z0, z1

    constexpr static const std::uint32_t DST_X_REG_OFFSET = 26; // src X registers are x21-x26
    constexpr static const std::uint32_t NUM_DST_X_REGS = 6; // src X registers are x21-x26
    constexpr static const std::uint32_t SRC_X_REG_OFFSET = 26; // src X registers are x21-x26
    constexpr static const std::uint32_t NUM_SRC_X_REGS = 6; // src X registers are x21-x26
    constexpr static const std::uint32_t WEI_X_REG_OFFSET = 17; // wei X registers are x17-x20
    constexpr static const std::uint32_t NUM_WEI_X_REGS = 4; // wei X registers are x17-x20

    constexpr static const std::uint32_t TMP_X_REG_OFFSET = 27; // tmp X registers are x27-x28
    constexpr static const std::uint32_t NUM_TMP_X_REGS = 2;

    constexpr static const std::uint32_t LOAD_STORE_P_REG_OFFSET = 1; // load/store P registers are p1-p4
    constexpr static const std::uint32_t NUM_LOAD_STORE_P_REGS = 4; // load/store P registers are p1-p4

    // function input arguments
    ARMCG::XReg srcPtr = x0;
    ARMCG::XReg weiPtr = x1;
    ARMCG::XReg dstPtr = x2;
    ARMCG::XReg biaPtr = x3;
    ARMCG::XReg flags = x4;
    ARMCG::SReg alpha = s0;
    ARMCG::HReg halpha = h0;
    ARMCG::SReg beta = s1;
    ARMCG::HReg hbeta = h1;
    ARMCG::XReg mVal = x5;
    ARMCG::XReg nVal = x6;
    ARMCG::XReg kVal = x7;
    ARMCG::XReg srcOffset = x8;
    ARMCG::XReg weiOffset = x9;
    ARMCG::XReg dstOffset = x10;
    ARMCG::XReg biaOffset = x11;
    ARMCG::XReg tmpMMinusMLoopCtr = x12;
    ARMCG::XReg tmpNMinusNLoopCtr = x13;
    ARMCG::XReg kLoopCtr = x14;
    ARMCG::XReg nLoopCtr = x15;
    ARMCG::XReg mLoopCtr = x16;
    // the gap should be aligned with src, wei and dst load registers (x17-x26)
    ARMCG::XReg tmp4VL = x27;
    ARMCG::XReg tmp = x28;
    ARMCG::WReg tmpW = w28;
    ARMCG::HReg tmpH = h28;

    using KUDNN_JIT_Generator::sp;
    using KUDNN_JIT_Generator::x0;
    using KUDNN_JIT_Generator::x1;
    using KUDNN_JIT_Generator::x2;
    using KUDNN_JIT_Generator::x3;
    using KUDNN_JIT_Generator::x4;
    using KUDNN_JIT_Generator::s0;
    using KUDNN_JIT_Generator::s1;
    using KUDNN_JIT_Generator::h0;
    using KUDNN_JIT_Generator::h1;
    using KUDNN_JIT_Generator::x5;
    using KUDNN_JIT_Generator::x6;
    using KUDNN_JIT_Generator::x7;
    using KUDNN_JIT_Generator::x8;
    using KUDNN_JIT_Generator::x9;
    using KUDNN_JIT_Generator::x10;
    using KUDNN_JIT_Generator::x11;
    using KUDNN_JIT_Generator::x12;
    using KUDNN_JIT_Generator::x13;
    using KUDNN_JIT_Generator::x14;
    using KUDNN_JIT_Generator::x15;
    using KUDNN_JIT_Generator::x16;
    using KUDNN_JIT_Generator::x27;
    using KUDNN_JIT_Generator::x28;
    using KUDNN_JIT_Generator::w28;
    using KUDNN_JIT_Generator::h28;
    using KUDNN_JIT_Generator::xzr;
    using KUDNN_JIT_Generator::p5;
    using KUDNN_JIT_Generator::Preamble;
    using KUDNN_JIT_Generator::Str;
    using KUDNN_JIT_Generator::Ldp;
    using KUDNN_JIT_Generator::Whilelt;
    using KUDNN_JIT_Generator::Mov;
    using KUDNN_JIT_Generator::Sub;
    using KUDNN_JIT_Generator::L;
    using KUDNN_JIT_Generator::Cmp;
    using KUDNN_JIT_Generator::B;
    using KUDNN_JIT_Generator::Postamble;
    using KUDNN_JIT_Generator::Ret;
    using KUDNN_JIT_Generator::Ptrue;
    using KUDNN_JIT_Generator::Tbz;
    using KUDNN_JIT_Generator::Ldr;
    using KUDNN_JIT_Generator::Cpy;
    using KUDNN_JIT_Generator::Add;
    using KUDNN_JIT_Generator::Mul;
    using KUDNN_JIT_Generator::Fmul;
    using KUDNN_JIT_Generator::Fmov;
    using KUDNN_JIT_Generator::Fadd;
    using KUDNN_JIT_Generator::Fmla;
    using KUDNN_JIT_Generator::Addvl;
    using KUDNN_JIT_Generator::Fcvt;
    using KUDNN_JIT_Generator::Fcmle;
    using KUDNN_JIT_Generator::Beq;
    using KUDNN_JIT_Generator_With_Attributes::TakeVectorRegister;
    using KUDNN_JIT_Generator_With_Attributes::TakePredicateRegister;
    using KUDNN_JIT_Generator_With_Attributes::TakeScalarRegister;
    using KUDNN_JIT_Generator_With_Attributes::FreeScalarRegister;
    using KUDNN_JIT_Generator_With_Attributes::FindFreeVectorReg;
    using KUDNN_JIT_Generator_With_Attributes::FindFreePredicateReg;
    using KUDNN_JIT_Generator_With_Attributes::FreePredicateRegister;
    using KUDNN_JIT_Generator_With_Attributes::FreeVectorRegister;
};

template <typename FP1632Type>
using VCF1632Gen = MatMulKernelDefaultCodeGen<ARCH::VC, FP1632Type, FP1632Type, FP1632Type, FP1632Type,
    MatMulImplKinds::F1632F1632F1632F1632_VC>;
template <typename FP1632Type>
using VCF1632CodegenFuncPtr = void (*)(
    const typename VCF1632Gen<FP1632Type>::SrcType *, const typename VCF1632Gen<FP1632Type>::WeiType *,
    typename VCF1632Gen<FP1632Type>::DstType *,
    const typename VCF1632Gen<FP1632Type>::BiaType *, MatMulDstFlags, float, float, IntType, IntType, IntType,
    IntType, IntType, IntType, IntType);
template <typename FP1632Type>
using VCF1632PackingCodegenFuncPtr = void(*)(
    IntType, IntType, const typename VCF1632Gen<FP1632Type>::WeiType *, IntType,
    const typename VCF1632Gen<FP1632Type>::WeiType *);

template <typename FP1632Type>
struct PackBRowDefaultSchemeCodegenF16F32 : public KUDNN_JIT_Generator {
    explicit PackBRowDefaultSchemeCodegenF16F32(IntType nBlockExec)
    {
        IntType svlen = Vector::Length<FP1632Type>();
        IntType iterCount = Math::CeilDiv(nBlockExec, svlen);
        IntType iterRemainder;

        if (iterCount % VCF1632Gen<FP1632Type>::KERNEL_N_SIZE == 0) {
            iterRemainder = VCF1632Gen<FP1632Type>::KERNEL_N_SIZE;
        } else {
            iterRemainder = (iterCount % VCF1632Gen<FP1632Type>::KERNEL_N_SIZE);
        }

        Mov(nOriginalVal, n);
        Mov(kbOffset, 0);
        Mov(kvlOffset, 0);
        Mov(storeStride, svlen * sizeof(FP1632Type));
        Mul(storeStride, storeStride, kCtr);

        IntType decrement = VCF1632Gen<FP1632Type>::KERNEL_N_SIZE * svlen;
        if ((nBlockExec % decrement) != 0) {
            decrement = nBlockExec % (VCF1632Gen<FP1632Type>::KERNEL_N_SIZE * svlen);
        }

        Sub(n, n, decrement);

        ARMCG::Label kLoopHead;
        L(kLoopHead);

            Add(currLoadPtr, bPtr, kbOffset, ARMCG::ShMod::LSL, Math::Log2(sizeof(FP1632Type)));
            Add(currStorePtr, bBuffer, kvlOffset, ARMCG::ShMod::LSL, Math::Log2(sizeof(FP1632Type)));

            Mov(nCtr, 0);
            if (iterCount > VCF1632Gen<FP1632Type>::KERNEL_N_SIZE) {
                ARMCG::Label nLoopHead;
                L(nLoopHead);
                    for (std::uint32_t i = 0; i < VCF1632Gen<FP1632Type>::KERNEL_N_SIZE; ++i) {
                        GenerateOneIter(i);
                        Add(currStorePtr, currStorePtr, storeStride);
                        Add(nCtr, nCtr, svlen);
                    }
                    Add(currLoadPtr, currLoadPtr, sizeof(FP1632Type) * VCF1632Gen<FP1632Type>::KERNEL_N_SIZE * svlen);
                    Cmp(nCtr, n);
                B(ARMCG::Cond::LT, nLoopHead);
            }
            for (std::uint32_t i = 0; i < iterRemainder; ++i) {
                GenerateOneIter(i);
                Add(currStorePtr, currStorePtr, storeStride);
                Add(nCtr, nCtr, svlen);
            }

            Add(kbOffset, kbOffset, bOffset);
            Add(kvlOffset, kvlOffset, svlen);
            Subs(kCtr, kCtr, 1);
        B(ARMCG::Cond::NE, kLoopHead);

        Ret();
    }

protected:
    void GenerateOneIter(IntType i)
    {
        Whilelt(KUDNN_JIT_Generator::GetPReg<FP1632Type>(i + 1), nCtr, nOriginalVal);
        Ld1(KUDNN_JIT_Generator::GetZReg<FP1632Type>(i + 1), ARMCG::PReg(i + 1),
            Ptr(currLoadPtr, i, ARMCG::ExtMod::MUL_VL));
        KUDNN_JIT_Generator::St1(KUDNN_JIT_Generator::GetZReg<FP1632Type>(i + 1), ARMCG::PReg(i + 1), Ptr(currStorePtr));
    }

    ARMCG::XReg n = x0;
    ARMCG::XReg kCtr = x1;
    ARMCG::XReg bPtr = x2;
    ARMCG::XReg bOffset = x3;
    ARMCG::XReg bBuffer = x4;
    ARMCG::XReg storeStride = x5;
    ARMCG::XReg currStorePtr = x6;
    ARMCG::XReg currLoadPtr = x7;
    ARMCG::XReg kbOffset = x8;
    ARMCG::XReg kvlOffset = x9;
    ARMCG::XReg nCtr = x10;
    ARMCG::XReg nOriginalVal = x11;
};

template <typename FP1632Type>
struct PackATypeHelper {
#ifdef KUDNNL_920F
    using Type = FastReorderKernelCodeGenMC<FP1632Type, FP1632Type>;
#else // KUDNNL_920F
    using Type = FastReorderKernelCodeGen<FP1632Type, FP1632Type>;
#endif // KUDNNL_920F
};

template <>
struct PackATypeHelper<float> {
    using Type = FastReorderKernelCodeGen<float, float>;
};

template <typename FP1632Type>
struct PackVCF1632Params {
    IntType n;
};

template <typename FP1632Type>
inline bool operator==(const PackVCF1632Params<FP1632Type> &lhs, const PackVCF1632Params<FP1632Type> &rhs) noexcept
{
    return VCF1632Gen<FP1632Type>::CompareNSize(lhs.n, rhs.n);
}

template <typename FP1632Type>
struct MatMulImplGenerator<VCF1632Gen<FP1632Type>> :
    MatmulTypeHelper<FP1632Type, FP1632Type, FP1632Type, FP1632Type>, ImplGenerator<GemmInfo> {
    using packAKernelCodeGenType = typename PackATypeHelper<FP1632Type>::Type;
    struct DefaultImpl : MatMulImplBase {
        DefaultImpl(const GemmInfo &task, IntType nThreads);

        constexpr static std::uint32_t GetKernelMSize() noexcept;

        static std::uint32_t GetKernelNSize() noexcept;

        IntType GetMBlockSize() const noexcept override;

        IntType GetNBlockSize() const noexcept override;

        IntType GetKBlockSize() const noexcept override;

    protected:

        void InitializeSingleCodegen(IntType id, IntType m, IntType n, IntType k,
            IntType currentM, IntType currentN, IntType currentK,
            SizeType offsetSrc, SizeType offsetWei, SizeType offsetDst, SizeType offsetBia,
            bool wereSrcWeiSwapped, const Attributes &attributes, bool hasAlpha) override;

        void InitializePackingCodeGens(IntType m, IntType n, IntType k,
            IntType mFull, IntType nFull) override;

        void InitializeReorderCodeGens(IntType m, IntType n, IntType k,
            IntType mFull, IntType nFull) override;

        void InitializeThreadsCountImpl() override;

        void InitializePackingPolicy() override;

        IntType GetThreadBufSize(IntType m, IntType n, IntType k, MatrixKind mKind) const noexcept override;

        void ReserveImplSpecificParams() override;

        std::vector<std::pair<Shape, Shape>> SplitWork() override;

        IntType GetPackPortion(MatrixKind mKind) const;

        void RunSequential2D(IntType threadID, const void *src, const void *wei, void *dst, const void *bia,
            std::unique_ptr<PackStrategy> &srcPackStrategy, std::unique_ptr<PackStrategy> &weiPackStrategy,
            float alpha, float beta, const PostOpsDataPtrs &shiftedByBatchPOPtrs) const noexcept(false) override;

        void UpdateExecParams() noexcept override;

        MatMulDstFlags SetBiaFlagsExec(MatMulDstFlags biaF, float beta, bool notFirstKIter,
            bool lastKIter, float alpha = 1.0f) const noexcept override;

        void PackSrcBA(IntType threadId, IntType m, IntType k, const void *src, IntType offset, void *buffer) const;
        void PackWeiAB(IntType threadId, IntType n, IntType k, const void *src,
            IntType offset, void *buffer) const;
        void PackWeiBA(IntType threadId, IntType n, IntType k, const void *src,
            IntType offset, void *buffer) const;

        void Pack(const void *originalMatrix, void *buffer, IntType threadId, IntType dstDim, IntType accumDim,
            IntType dstDimOffset, IntType accumDimOffset, IntType bufferOffset, IntType bufferStride, float alpha,
            MatrixKind mKind) const override;

        std::array<VCF1632CodegenFuncPtr<FP1632Type>, NUM_18> matmulCodegenKernels;
        std::array<VCF1632PackingCodegenFuncPtr<FP1632Type>, NUM_3> vectorPackBRowCodegenKernels;
        std::array<CodeGenWrapper<FastReorderKernelCodeGen<FP1632Type, FP1632Type>>, NUM_6>
            reorderPackingCodegenKernels;
        std::array<CodeGenWrapper<packAKernelCodeGenType>, NUM_6> packAColCodegenKernels;
        // these static variables are modified at impl construction stage which is performed
        // synchroniously, that's why we don't need extra syncs to prevent form writing to these vars
        inline static std::unordered_map<PackVCF1632Params<FP1632Type>,
            std::shared_ptr<PackBRowDefaultSchemeCodegenF16F32<FP1632Type>>> packingCodegens;
        inline static std::unordered_map<CodeGenParams<VCF1632Gen<FP1632Type>>,
            std::shared_ptr<VCF1632Gen<FP1632Type>>> codegens;
    };
    bool Supports(const GemmInfo &task) const noexcept override
    {
        VERBOSE_PRINT;
        bool srcOK = true;
        bool weiOK = true;
        bool dstOK = true;
        bool biaOK = true;
        bool fp16check = true;
        if constexpr (std::is_same_v<FP1632Type, __fp16>) {
            srcOK = (sizeof(FP1632Type) >= task.srcInfo.GetType().GetSize()) &&
                (task.srcInfo.GetType() != Element::MatchType<__bf16>());
            weiOK = (sizeof(FP1632Type) >= task.weiInfo.GetType().GetSize()) &&
                (task.weiInfo.GetType() != Element::MatchType<__bf16>());
            dstOK = (sizeof(FP1632Type) >= task.dstInfo.GetType().GetSize()) &&
                (task.dstInfo.GetType() != Element::MatchType<__bf16>());
            if (task.biaInfo.GetTotalTensorSize() != 0) {
                biaOK = (sizeof(FP1632Type) >= task.biaInfo.GetType().GetSize()) &&
                    (task.biaInfo.GetType() != Element::MatchType<__bf16>());
            }
            IntType vl = Vector::Length<FP1632Type>();
            fp16check = (vl <= NUM_16);
        }
        return fp16check && srcOK && weiOK && dstOK && biaOK;
    }
    std::chrono::duration<float, std::milli> EstimateTime(const GemmInfo &task,
                                                            IntType nThreads) const noexcept override
    {
        VERBOSE_PRINT;
        float singleThreadPerformance = 1.0;
#ifdef KUDNNL_920F
        // experementaly measured performance in GFlops/s
        if constexpr (std::is_same_v<FP1632Type, float>) {
            singleThreadPerformance = 93;  // 93 is the estimated performance in GFlops/s for 920F
        } else {
            singleThreadPerformance = 1.0; // won't be executed on 920F since register length is too much
        }
#else
        if constexpr (std::is_same_v<FP1632Type, float>) {
            singleThreadPerformance = 85; // 85 is the float type performance in GFlops/s
        } else {
            singleThreadPerformance = 150; // 150 is the f16/bf16 type performance in GFlops/s
        }
#endif
        return CalculateMatmulTimeWithReorder<FP1632Type, FP1632Type, FP1632Type, FP1632Type>(
            singleThreadPerformance, task, nThreads);
    }
    std::shared_ptr<ImplBase<GemmInfo>> GetImpl(const GemmInfo &task, IntType nThreads) const override
    {
        VERBOSE_PRINT;
        return std::make_shared<DefaultImpl>(task, nThreads);
    }
};
} // Detail
} // KuDNN

template <typename FP1632Type>
struct std::hash<KuDNN::Detail::PackVCF1632Params<FP1632Type>> {
    std::size_t operator()(const KuDNN::Detail::PackVCF1632Params<FP1632Type>& packCodeGenParams) const noexcept
    {
        return std::hash<KuDNN::IntType>{}(KuDNN::Detail::VCF1632Gen<FP1632Type>::ProcessNSize(packCodeGenParams.n));
    }
};

#endif // KUDNN_MATMUL_JIT_VC_DEFAUT_IMPL_F1632F1632F1632F1632_HPP
