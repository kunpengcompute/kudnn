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

#ifndef KUDNN_MATMUL_JIT_VectorComputation_DEFAUT_IMPL_F32F32F32F32_HPP
#define KUDNN_MATMUL_JIT_VectorComputation_DEFAUT_IMPL_F32F32F32F32_HPP

#include <memory>
#include <vector>
#include <array>

#include <arm_sve.h>

#include "kudnn.hpp"

#include "cg_codegen.hpp"

#include "kudnn_jit.hpp"
#include "matmul_jit/kudnn_gemm_jit.hpp"

namespace KuDNN {
namespace Detail {

template <>
struct MatMulKernelDefaultCodeGenImpl<ARCH::VectorComputation, float, float, float, float, MatmulDummyType>
    : MatMulKernelCodeGenBase<ARCH::VectorComputation, float, float, float, float> {
public:
    constexpr static const std::uint32_t KERNEL_M_SIZE = 6;
    constexpr static const std::uint32_t KERNEL_N_SIZE = 4; // in VL
    // Kernel performs 6x4VL update of destination matrix via
    // outer product computations inside loops over M, N, K. Registers z8-z31 are used for
    // accumulation (then will be stored to destination matrix),
    // registerrs z1, z0 are used to store matrix B values (reused to store 4VL),
    // registers z2 - z7 are used to store matrix A values.
    // The kernel has unroll by K variable ([m x k] * [k * n] -> [m * n]) by 4.
    //                                            x17     x18     x19     x20
    //                                             |       |       |       |
    //                                             v       v       v       v
    //                                             +-------+-------+-------+-------+
    //                     1st K iteration         | z1.s  | z0.s  | z1.s  | z0.s  |
    //                                             +-------+-------+-------+-------+
    //          Rhs ->                             |  ...  |  ...  |  ...  |  ...  |
    //                                             +-------+-------+-------+-------|
    //                     4th K iteration         | z1.s  | z0.s  | z1.s  | z0.s  |
    //                                             +-------+-------+-------+-------+
    //                   Lhs                       |       |       |       |       |
    //                    |
    //                    v                        |       |       |       |       |
    //            1st K           4th K
    //          +---------+-----+---------+  -  -  +-------+-------+-------+-------+
    //  x26 ->  | z7.s[0] | ... | z7.s[3] |        | z8.s  | z9.s  | z10.s | z11.s |
    //  x25 ->  | z6.s[0] | ... | z6.s[3] |        | z12.s | z13.s | z14.s | z15.s |
    //  x24 ->  | z5.s[0] | ... | z5.s[3] |        | z16.s | z17.s | z18.s | z19.s |
    //  x23 ->  | z4.s[0] | ... | z4.s[3] |        | z20.s | z21.s | z22.s | z23.s |
    //  x22 ->  | z3.s[0] | ... | z3.s[3] |        | z24.s | z25.s | z26.s | z27.s |
    //  x21 ->  | z2.s[0] | ... | z2.s[3] |        | z28.s | z29.s | z30.s | z31.s |
    //          +---------+-----+---------+  -  -  +-------+-------+-------+-------+

    // There are also kernels of smaller sizes to process borders of matrix (5x4VL, 4x4VL, 3x4VL, 2x4VL, 1x4VL).
    // They are very similar to main kernel.

    MatMulKernelDefaultCodeGenImpl(IntType m, IntType n, IntType k, IntType srcOffsetArg, IntType weiOffsetArg,
                                   IntType dstOffsetArg, IntType biaOffsetArg);

protected:
    void InitializeDstRegisters(std::uint32_t mLim, std::uint32_t nLim);

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

    static std::uint32_t GetLoadStorePRegIdx(std::uint32_t n) noexcept
    {
        return LOAD_STORE_P_REG_OFFSET + n % NUM_LOAD_STORE_P_REGS;
    }

    constexpr static const std::uint32_t K_LOOP_UNROLL = 4;

    constexpr static const std::uint32_t DST_Z_REG_OFFSET = 8; // dst Z registers are z8 - z31
    constexpr static const std::uint32_t NUM_DST_Z_REGS = 24;  // dst Z registers are z8 - z31
    constexpr static const std::uint32_t SRC_Z_REG_OFFSET = 7; // src Z registers are z7 - z2
    constexpr static const std::uint32_t NUM_SRC_Z_REGS = 6;   // src Z registers are z7 - z2
    constexpr static const std::uint32_t WEI_Z_REG_OFFSET = 1; // wei Z registers are z0, z1
    constexpr static const std::uint32_t NUM_WEI_Z_REGS = 2;   // wei Z registers are z0, z1

    constexpr static const std::uint32_t DST_X_REG_OFFSET = 26; // src X registers are x21-x26
    constexpr static const std::uint32_t NUM_DST_X_REGS = 6;    // src X registers are x21-x26
    constexpr static const std::uint32_t SRC_X_REG_OFFSET = 26; // src X registers are x21-x26
    constexpr static const std::uint32_t NUM_SRC_X_REGS = 6;    // src X registers are x21-x26
    constexpr static const std::uint32_t WEI_X_REG_OFFSET = 17; // wei X registers are x17-x20
    constexpr static const std::uint32_t NUM_WEI_X_REGS = 4;    // wei X registers are x17-x20

    constexpr static const std::uint32_t LOAD_STORE_P_REG_OFFSET = 1; // wei X registers are x17-x20
    constexpr static const std::uint32_t NUM_LOAD_STORE_P_REGS = 4;   // wei X registers are x17-x20

    // function input arguments
    ARMCG::XReg srcPtr = x0;
    ARMCG::XReg weiPtr = x1;
    ARMCG::XReg dstPtr = x2;
    ARMCG::XReg biaPtr = x3;
    ARMCG::XReg flags = x4;
    ARMCG::SReg beta = s0;
    // other used registers
    ARMCG::XReg mVal = x6;
    ARMCG::XReg nVal = x7;
    ARMCG::XReg kVal = x8;
    ARMCG::XReg srcOffset = x9;
    ARMCG::XReg weiOffset = x10;
    ARMCG::XReg dstOffset = x11;
    ARMCG::XReg biaOffset = x12;
    ARMCG::XReg tmpMMinusMLoopCtr = x13;
    ARMCG::XReg tmp = x14;
    ARMCG::XReg tmpNMinusNLoopCtr = x15;
    ARMCG::XReg tmp4VL = x16;
    // the gap should be aligned with src, wei and dst load registers (x17-x26)
    ARMCG::XReg kLoopCtr = x27;
    ARMCG::XReg nLoopCtr = x28;
    ARMCG::XReg mLoopCtr = x29;
};

template <>
struct MatMulKernelDefaultCodeGen<ARCH::VectorComputation, float, float, float, float>
    : MatMulKernelDefaultCodeGenImpl<ARCH::VectorComputation, float, float, float, float, MatmulDummyType> {
    MatMulKernelDefaultCodeGen(IntType m, IntType n, IntType k, IntType srcOffsetArg, IntType weiOffsetArg,
                               IntType dstOffsetArg, IntType biaOffsetArg)
        : MatMulKernelDefaultCodeGenImpl(m, n, k, srcOffsetArg, weiOffsetArg, dstOffsetArg, biaOffsetArg)
    {
        VERBOSE_PRINT;
    }
};

struct PackBRowDefaultSchemeCodegenF32 : public KUDNN_JIT_Generator {
    using CodeGen = MatMulKernelDefaultCodeGen<ARCH::VectorComputation, float, float, float, float>;
    PackBRowDefaultSchemeCodegenF32(IntType nBlockExec, bool hasAlpha) : KUDNN_JIT_Generator()
    {
        IntType svlen = svcntw();
        IntType iterCount = (nBlockExec + svlen - 1) / svlen;
        IntType iterRemainder;

        if (iterCount % CodeGen::KERNEL_N_SIZE == 0) {
            iterRemainder = CodeGen::KERNEL_N_SIZE;
        } else {
            iterRemainder = (iterCount % CodeGen::KERNEL_N_SIZE);
        }

        if (hasAlpha) {
            Mov(ARMCG::ZRegS(0), alpha);
        }

        Mov(n, nBlockExec);
        Mov(kbOffset, 0);
        Mov(kvlOffset, 0);
        Cntw(storeStride, ARMCG::Pattern::ALL, ARMCG::ExtMod::MUL, sizeof(float));
        Mul(storeStride, storeStride, kCtr);

        if (nBlockExec % (CodeGen::KERNEL_N_SIZE * svlen) == 0) {
            nBlockExec = nBlockExec - (CodeGen::KERNEL_N_SIZE * svlen);
        } else {
            nBlockExec = nBlockExec - (nBlockExec % (CodeGen::KERNEL_N_SIZE * svlen));
        }

        ARMCG::Label kLoopHead;

        L(kLoopHead);

        Add(currLoadPtr, bPtr, kbOffset, ARMCG::ShMod::LSL, NUM_2);
        Add(currStorePtr, bBuffer, kvlOffset, ARMCG::ShMod::LSL, NUM_2);

        Mov(nCtr, 0);
        if (iterCount > CodeGen::KERNEL_N_SIZE) {
            ARMCG::Label nLoopHead;
            L(nLoopHead);
            for (std::uint32_t i = 0; i < CodeGen::KERNEL_N_SIZE; ++i) {
                GenerateOneIter(hasAlpha, i);
                Add(currStorePtr, currStorePtr, storeStride);
                Add(nCtr, nCtr, svlen);
            }
            Add(currLoadPtr, currLoadPtr, sizeof(float) * CodeGen::KERNEL_N_SIZE * svlen);
            Cmp(nCtr, nBlockExec);
            B(ARMCG::Cond::NE, nLoopHead);
        }

        for (std::uint32_t i = 0; i < iterRemainder; ++i) {
            GenerateOneIter(hasAlpha, i);
            Add(currStorePtr, currStorePtr, storeStride);
            Add(nCtr, nCtr, svlen);
        }

        Add(kbOffset, kbOffset, bOffset);
        Add(kvlOffset, kvlOffset, svlen);
        Subs(kCtr, kCtr, 1);
        B(ARMCG::Cond::NE, kLoopHead);

        Ret();
    }

    void GenerateOneIter(bool hasAlpha, IntType i)
    {
        Whilelt(ARMCG::PRegS(i + 1), nCtr, n);
        Ld1w(ARMCG::ZRegS(i + 1), ARMCG::PReg(i + 1), Ptr(currLoadPtr, i, ARMCG::ExtMod::MUL_VL));
        if (hasAlpha) {
            Fmul(ARMCG::ZRegS(i + 1), ARMCG::PReg(i + 1), ARMCG::ZRegS(0));
        }
        St1w(ARMCG::ZRegS(i + 1), ARMCG::PReg(i + 1), Ptr(currStorePtr));
    }

    ARMCG::SReg alpha = s0;
    ARMCG::XReg kCtr = x0;
    ARMCG::XReg bPtr = x1;
    ARMCG::XReg bOffset = x2;
    ARMCG::XReg bBuffer = x3;
    ARMCG::XReg storeStride = x4;
    ARMCG::XReg currStorePtr = x5;
    ARMCG::XReg currLoadPtr = x6;
    ARMCG::XReg kbOffset = x7;
    ARMCG::XReg kvlOffset = x8;
    ARMCG::XReg n = x9;
    ARMCG::XReg nCtr = x10;
};

template <>
struct MatMulImplGenerator<MatMulKernelDefaultCodeGen<ARCH::VectorComputation, float, float, float, float>>
    : MatmulTypeHelper<float, float, float, float>
    , ImplGenerator<GemmInfo> {
    using CodeGen = MatMulKernelDefaultCodeGen<ARCH::VectorComputation, float, float, float, float>;
    struct DefaultImpl : MatMulImplBase {
        DefaultImpl(const GemmInfo &task, IntType nThreads);

        constexpr static std::uint32_t GetKernelMSize() noexcept;

        static std::uint32_t GetKernelNSize() noexcept;

    protected:
        void InitializeCodeGens(IntType m, IntType n, IntType k, IntType vl, SizeType offsetSrc, SizeType offsetWei,
                                SizeType offsetDst, SizeType offsetBia, bool wereSrcWeiSwapped) override;

        IntType GetThreadBufSize(IntType m, IntType n, IntType k, IntType vl) noexcept override;

        void ReserveImplSpecificParams() override;

        std::vector<std::pair<Shape, Shape>> SplitWork() override;

        template <bool AlphaMul = false>
        static void PackBCol(IntType n, IntType k, const WeiType *bPtr, IntType bOffset, WeiType *bBuffer,
                             float alpha = 1.0f);

        void RunSequential2D(IntType threadId, const void *src, const void *wei, void *dst, const void *bia,
                             void *buffer, float alpha, float beta) const noexcept(false) override;

        void UpdateExecParams() noexcept override;

        // in case of M, N, K impl. depends on order beffore kernel call, M * K ~= L3 Cache per Thread . 929B = 1.75 kB
        constexpr static IntType M_BLOCK_SIZE = 1440; // M % 6 == 0
        constexpr static IntType K_BLOCK_SIZE = 316;  // K % 4 == 0
        constexpr static IntType N_BLOCK_SIZE = 256;  // N % 4* VL == 0

        std::vector<std::array<CodeGenWrapper<CodeGen>, NUM_8>> perThreadImpls;
        std::vector<std::array<CodeGenWrapper<PackBRowDefaultSchemeCodegenF32>, NUM_4>> vectorPackBRow;
    };
    bool Supports(const GemmInfo &task) const noexcept override
    {
        VERBOSE_PRINT;
        Service::Unused(task);
        return true;
    }
    float EstimateTime(const GemmInfo &task, IntType nThreads) const noexcept override
    {
        VERBOSE_PRINT;
        Service::Unused(task, nThreads);

        return 1.0f;
    }
    std::shared_ptr<ImplBase<GemmInfo>> GetImpl(const GemmInfo &task, IntType nThreads) const override
    {
        VERBOSE_PRINT;
        return std::make_shared<DefaultImpl>(task, nThreads);
    }
};

} // namespace Detail
} // namespace KuDNN

#endif // KUDNN_MATMUL_JIT_VectorComputation_DEFAUT_IMPL_F32F32F32F32_HPP
