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
#ifndef KUDNN_MATMUL_JIT_VC_DEFAUT_IMPL_F32F32F32F32_HPP
#define KUDNN_MATMUL_JIT_VC_DEFAUT_IMPL_F32F32F32F32_HPP

#include <memory>
#include <vector>
#include <array>
#include <type_traits>
#include <unordered_map>

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

namespace KuDNN {
namespace Detail {

template <>
struct MatMulKernelDefaultCodeGen<ARCH::NEON, float, float, float, float,
    MatMulImplKinds::F32F32F32F32_NEON> :
    MatMulKernelCodeGenBase<ARCH::NEON, float, float, float, float>
{
public:
    constexpr static const std::uint32_t KERNEL_M_SIZE = 6;
    constexpr static const std::uint32_t KERNEL_N_SIZE = 4; // in VL
    // indexed FMLA operates on indices inside 128 bit (16 bytes) of register
    constexpr static const std::uint32_t K_LOOP_UNROLL = NUM_16 / sizeof(float);

    constexpr static const IntType NEON_VL = 4;
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
        if ((lhs <= (KERNEL_N_SIZE * NEON_VL)) || (rhs <= (KERNEL_N_SIZE * NEON_VL))) {
            return lhs == rhs;
        } else {
            return ProcessNSize(lhs) == ProcessNSize(rhs);
        }
    }

    static IntType ProcessNSize(IntType n) noexcept
    {
        return n % (KERNEL_N_SIZE * NEON_VL);
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
    void StoreToDst(std::uint32_t mLim, std::uint32_t nLim, std::uint32_t nResidScalar);

    void GenerateMLoopBody(std::uint32_t mLim, std::uint32_t nLim, std::uint32_t k);

    void GenerateNLoopBody(std::uint32_t mLim, std::uint32_t nLim, std::uint32_t k, std::uint32_t nResidScalar);

    void GenerateMLAPart(std::uint32_t mLim, std::uint32_t nLim, std::uint32_t kLim, bool isTail);

    void ApplyEltwise(const PostOps &postOps, SizeType poIdx, const std::vector<SizeType> &regIDs,
        Element::TypeT regType) override;

    static std::uint32_t GetDstVRegIdx(std::uint32_t m, std::uint32_t n) noexcept
    {
        return DST_V_REG_OFFSET + (n + m * KERNEL_N_SIZE) % NUM_DST_V_REGS;
    }

    static std::uint32_t GetDstXRegIdx(std::uint32_t m) noexcept
    {
        return DST_X_REG_OFFSET - m % NUM_DST_X_REGS;
    }

    static std::uint32_t GetSrcVRegIdx(std::uint32_t m) noexcept
    {
        return SRC_V_REG_OFFSET - m % NUM_SRC_V_REGS;
    }

    static std::uint32_t GetSrcXRegIdx(std::uint32_t m) noexcept
    {
        return SRC_X_REG_OFFSET - m % NUM_SRC_X_REGS;
    }

    static std::uint32_t GetWeiVRegIdx(std::uint32_t n) noexcept
    {
        return WEI_V_REG_OFFSET - n % NUM_WEI_V_REGS;
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

    constexpr static const std::uint32_t DST_V_REG_OFFSET = 8; // dst Z registers are z8 - z31
    constexpr static const std::uint32_t NUM_DST_V_REGS = 24; // dst Z registers are z8 - z31
    constexpr static const std::uint32_t SRC_V_REG_OFFSET = 7; // src Z registers are z7 - z2
    constexpr static const std::uint32_t NUM_SRC_V_REGS = 6; // src Z registers are z7 - z2
    constexpr static const std::uint32_t WEI_V_REG_OFFSET = 1; // wei Z registers are z0, z1
    constexpr static const std::uint32_t NUM_WEI_V_REGS = 2; // wei Z registers are z0, z1

    constexpr static const std::uint32_t DST_X_REG_OFFSET = 26; // src X registers are x21-x26
    constexpr static const std::uint32_t NUM_DST_X_REGS = 6; // src X registers are x21-x26
    constexpr static const std::uint32_t SRC_X_REG_OFFSET = 26; // src X registers are x21-x26
    constexpr static const std::uint32_t NUM_SRC_X_REGS = 6; // src X registers are x21-x26
    constexpr static const std::uint32_t WEI_X_REG_OFFSET = 17; // wei X registers are x17-x20
    constexpr static const std::uint32_t NUM_WEI_X_REGS = 4; // wei X registers are x17-x20

    constexpr static const std::uint32_t TMP_X_REG_OFFSET = 27; // tmp X registers are x30-x31
    constexpr static const std::uint32_t NUM_TMP_X_REGS = 2;

    constexpr static const std::uint32_t LOAD_STORE_P_REG_OFFSET = 1; // wei X registers are x17-x20
    constexpr static const std::uint32_t NUM_LOAD_STORE_P_REGS = 4; // wei X registers are x17-x20

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
};

using NeonF32Gen = MatMulKernelDefaultCodeGen<ARCH::NEON, float, float, float, float,
    MatMulImplKinds::F32F32F32F32_NEON>;
using NeonF32GenFuncPtr = void (*)(
    const typename NeonF32Gen::SrcType *, const typename NeonF32Gen::WeiType *, typename NeonF32Gen::DstType *,
    const typename NeonF32Gen::BiaType *, MatMulDstFlags, float, float, IntType, IntType, IntType,
    IntType, IntType, IntType, IntType);
using PackingNeonF32GenFuncPtr = void(*)(
    IntType, IntType, const typename NeonF32Gen::WeiType *, IntType,
    const typename NeonF32Gen::WeiType *);

struct PackBRowDefaultSchemeNeonF32GenF32 : public KUDNN_JIT_Generator {
    explicit PackBRowDefaultSchemeNeonF32GenF32(IntType nBlockExec)
    {
        IntType iterCount = Math::CeilDiv(nBlockExec, NeonF32Gen::NEON_VL);
        IntType nBlockRemainder;
        IntType iterRemainder;

        if (nBlockExec % NeonF32Gen::NEON_VL == 0) {
            nBlockRemainder = NeonF32Gen::NEON_VL;
        } else {
            nBlockRemainder = (nBlockExec % NeonF32Gen::NEON_VL);
        }
        if (iterCount % NeonF32Gen::KERNEL_N_SIZE == 0) {
            iterRemainder = NeonF32Gen::KERNEL_N_SIZE;
        } else {
            iterRemainder = (iterCount % NeonF32Gen::KERNEL_N_SIZE);
        }
        Mov(kbOffset, 0);
        Mov(kvlOffset, 0);
        Mov(storeStride, NeonF32Gen::NEON_VL * sizeof(float));
        Mul(storeStride, storeStride, kCtr);

        IntType decrement = NeonF32Gen::KERNEL_N_SIZE * NeonF32Gen::NEON_VL;
        if ((nBlockExec % decrement) != 0) {
            decrement = nBlockExec % (NeonF32Gen::KERNEL_N_SIZE * NeonF32Gen::NEON_VL);
        }

        Sub(n, n, decrement);

        ARMCG::Label kLoopHead;
        L(kLoopHead);

            Add(currLoadPtr, bPtr, kbOffset, ARMCG::ShMod::LSL, Math::Log2(sizeof(float)));
            Add(currStorePtr, bBuffer, kvlOffset, ARMCG::ShMod::LSL, Math::Log2(sizeof(float)));

            Mov(nCtr, 0);
            if (iterCount > NeonF32Gen::KERNEL_N_SIZE) {
                ARMCG::Label nLoopHead;
                L(nLoopHead);
                    for (std::uint32_t i = 0; i < NeonF32Gen::KERNEL_N_SIZE; ++i) {
                        GenerateOneIter(i, NeonF32Gen::NEON_VL, nBlockRemainder, false);
                        Add(currStorePtr, currStorePtr, storeStride);
                        Add(nCtr, nCtr, NeonF32Gen::NEON_VL);
                    }
                    Cmp(nCtr, n);
                B(ARMCG::Cond::LT, nLoopHead);
            }
            // Here we need to process corner cases
            for (std::uint32_t i = 0; i < iterRemainder; ++i) {
                GenerateOneIter(i, iterRemainder, nBlockRemainder, true);
                Add(currStorePtr, currStorePtr, storeStride);
                Add(nCtr, nCtr, NeonF32Gen::NEON_VL);
            }
            Add(kbOffset, kbOffset, bOffset);
            Add(kvlOffset, kvlOffset, NeonF32Gen::NEON_VL);
            Subs(kCtr, kCtr, 1);
        B(ARMCG::Cond::NE, kLoopHead);

        Ret();
    }

protected:
    void GenerateOneIter(IntType i, IntType remainderQuad,
        IntType remainderElem, bool isRemainder)
    {
        if ((remainderElem < NUM_4) && (isRemainder) && (i == remainderQuad - 1)) {
            // last quad, <4 elems left
            CodeGenerator::Movi(ARMCG::VReg4S(i + 1), 0);
            for (IntType j = 0; j < remainderElem; j++) {
                CodeGenerator::Ld1(ARMCG::VReg4S(i + 1)[j], PostPtr(currLoadPtr, sizeof(float)));
            }
            CodeGenerator::St1(ARMCG::VReg4SList(ARMCG::VReg4S(i + 1)), Ptr(currStorePtr));
        } else {
            CodeGenerator::Ld1(ARMCG::VReg4SList(ARMCG::VReg4S(i + 1)),
                PostPtr(currLoadPtr, sizeof(float) * NUM_4));
            CodeGenerator::St1(ARMCG::VReg4SList(ARMCG::VReg4S(i + 1)), Ptr(currStorePtr));
        }
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
};

struct PackNeonF32GenParams {
    IntType n;
};

inline bool operator==(const PackNeonF32GenParams &lhs, const PackNeonF32GenParams &rhs) noexcept
{
    return NeonF32Gen::CompareNSize(lhs.n, rhs.n);
}

template <>
struct MatMulImplGenerator<NeonF32Gen>:
    MatmulTypeHelper<float, float, float, float>, ImplGenerator<GemmInfo> {
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

        void InitializeThreadsCountImpl() override;

        IntType GetThreadBufSize(IntType m, IntType n, IntType k,
            MatrixKind mKind) const noexcept override;

        void ReserveImplSpecificParams() override;

        std::vector<std::pair<Shape, Shape>> SplitWork() override;

        MatMulDstFlags SetBiaFlagsExec(MatMulDstFlags biaF, float beta, bool notFirstKIter,
            bool lastKIter, float alpha = 1.0f) const noexcept override;

        void PackBCol(IntType threadID, IntType n, IntType k, const typename NeonF32Gen::WeiType *bPtr,
            IntType bOffset, typename NeonF32Gen::WeiType *bBuffer) const;

        void RunSequential2D(IntType threadID, const void *src, const void *wei, void *dst, const void *bia,
            std::unique_ptr<PackStrategy> &srcPackStrategy, std::unique_ptr<PackStrategy> &weiPackStrategy,
            float alpha, float beta, const PostOpsDataPtrs &shiftedByBatchPOPtrs) const noexcept(false) override;

        void UpdateExecParams() noexcept override;

        std::array<NeonF32GenFuncPtr, NUM_18> matmulCodegenKernels;
        std::array<PackingNeonF32GenFuncPtr, NUM_3> vectorPackBRowCodegenKernels;
        // these static variables are modified at impl construction stage which is performed
        // synchroniously, that's why we don't need extra syncs to prevent form writing to these vars
        static std::unordered_map<PackNeonF32GenParams, std::shared_ptr<PackBRowDefaultSchemeNeonF32GenF32>>
            packingCodegens;
        inline static std::unordered_map<CodeGenParams<NeonF32Gen>, std::shared_ptr<NeonF32Gen>> codegens;
    };
    bool Supports(const GemmInfo &task) const noexcept override
    {
        VERBOSE_PRINT;
        Service::Unused(task);
        return true;
    }
    std::chrono::duration<float, std::milli> EstimateTime(const GemmInfo &task,
                                                          IntType nThreads) const noexcept override
    {
        VERBOSE_PRINT;
        float singleThreadPerformance = NUM_77;
        if (std::getenv("KUDNN_FORCE_NEON")) {
#ifdef KUDNNL_920F
            singleThreadPerformance = 512;   // 512 is the neon theor peak
#else
            singleThreadPerformance = 96;   // 96 is the neon theor peak
#endif
        }
        return CalculateMatmulTimeWithReorder<float, float, float, float>(
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

template <>
struct std::hash<KuDNN::Detail::PackNeonF32GenParams> {
    std::size_t operator()(const KuDNN::Detail::PackNeonF32GenParams& packCodeGenParams) const noexcept
    {
        return std::hash<KuDNN::IntType>{}(KuDNN::Detail::NeonF32Gen::ProcessNSize(packCodeGenParams.n));
    }
};

#endif // KUDNN_MATMUL_JIT_VC_DEFAUT_IMPL_F32F32F32F32_HPP
