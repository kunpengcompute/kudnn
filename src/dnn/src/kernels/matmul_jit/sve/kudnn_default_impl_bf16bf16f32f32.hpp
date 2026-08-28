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

#ifndef KUDNN_MATMUL_JIT_VC_DEFAULT_IMPL_BF16BF16F32F32_HPP
#define KUDNN_MATMUL_JIT_VC_DEFAULT_IMPL_BF16BF16F32F32_HPP

#include <memory>
#include <vector>
#include <array>
 
#include <arm_sve.h>

#include "kudnn.hpp"
#include "cg_codegen.hpp"
#include "kudnn_jit.hpp"
#include "kernels/matmul_jit/kudnn_gemm_jit.hpp"
#include "internal_funcs.hpp"
 
namespace KuDNN {
namespace Detail {

template <>
struct MatMulKernelDefaultCodeGen<ARCH::VC, __bf16, __bf16, float, float,
    MatMulImplKinds::BF16BF16F32F32_VC> :
    MatMulKernelCodeGenBase<ARCH::VC, __bf16, __bf16, float, float>
{
public:
    constexpr static const std::uint32_t KERNEL_M_SIZE = 6;
    constexpr static const std::uint32_t KERNEL_N_SIZE = 4;
    // Kernel performs 6x4 update of destination matrix
    // The kernel has unrolled by K variable ([m x k] * [k * n] -> [m * n]) by vector length.
    //                                          x17     x18     x19     x20
    //                                          |       |       |       |
    //                                          v       v       v       v
    //                                          +-------+-------+-------+-------+
    //          Rhs ->                          | z1.h  | z0.h  | z1.h  | z0.h  |
    //                                          +-------+-------+-------+-------+
    //         Lhs                              |       |       |       |       |
    //          |
    //          v                               |       |       |       |       |
    //          +-------+  -  -  -  -  -  -  -  +-------+-------+-------+-------+
    //  x26 ->  | z7.h  |                       | z8.s  | z9.s  | z10.s | z11.s |
    //  x25 ->  | z6.h  |                       | z12.s | z13.s | z14.s | z15.s |
    //  x24 ->  | z5.h  |                       | z16.s | z17.s | z18.s | z19.s |
    //  x23 ->  | z4.h  |                       | z20.s | z21.s | z22.s | z23.s |
    //  x22 ->  | z3.h  |                       | z24.s | z25.s | z26.s | z27.s |
    //  x21 ->  | z2.h  |                       | z28.s | z29.s | z30.s | z31.s |
    //          +-------+  -  -  -  -  -  -  -  +-------+-------+-------+-------+
    // There are also kernels of smaller sizes to process borders of matrix by different (1-6):m,(1-4):n
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
        if ((lhs <= KERNEL_N_SIZE) || (rhs <= KERNEL_N_SIZE)) {
            return lhs == rhs;
        } else {
            return ProcessNSize(lhs) == ProcessNSize(rhs);
        }
    }

    static IntType ProcessNSize(IntType n) noexcept
    {
        return n % KERNEL_N_SIZE;
    }

    static bool CompareKSize(IntType lhs, IntType rhs) noexcept
    {
        return ProcessKSize(lhs) == ProcessKSize(rhs);
    }

    static IntType ProcessKSize(IntType k) noexcept
    {
        return k > static_cast<IntType>(svcnth());
    }

protected:
    void StoreToDst(std::uint32_t mLim, std::uint32_t nLim);

    void GenerateMLoopBody(std::uint32_t mLim, std::uint32_t nLim, std::uint32_t k);
 
    void GenerateNLoopBody(std::uint32_t mLim, std::uint32_t nLim, std::uint32_t k);
 
    void GenerateAccumulation(std::uint32_t mLim, std::uint32_t nLim);

    static std::uint32_t GetDstZRegIdx(std::uint32_t m, std::uint32_t n) noexcept
    {
        return DST_Z_REG_OFFSET + (n + m * KERNEL_N_SIZE) % NUM_DST_Z_REGS;
    }

    static std::uint32_t GetDstXRegIdx(std::uint32_t m) noexcept
    {
        return dstXRegOffset - m % numDstXRegs;
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
        return WEI_Z_REG_OFFSEt - n % NUM_WEI_Z_REGS;
    }

    static std::uint32_t GetWeiXRegIdx(std::uint32_t n) noexcept
    {
        return WEI_X_REG_OFFSET + n % NUM_WEI_X_REGS;
    }

    static std::uint32_t GetTmpXRegIdx(std::uint32_t n) noexcept
    {
        return TMP_X_REG_OFFSET + n % NUM_TMP_X_REGS;
    }

    constexpr static const std::uint32_t DST_Z_REG_OFFSET = 8; // dst Z registers are z8 - z31
    constexpr static const std::uint32_t NUM_DST_Z_REGS = KERNEL_M_SIZE * KERNEL_N_SIZE; // dst Z registers are z8 - z31
    constexpr static const std::uint32_t SRC_Z_REG_OFFSET = 7; // src Z registers are z7 - z2
    constexpr static const std::uint32_t NUM_SRC_Z_REGS = KERNEL_M_SIZE; // src Z registers are z7 - z2
    constexpr static const std::uint32_t WEI_Z_REG_OFFSEt = 1; // wei Z registers are z0, z1
    constexpr static const std::uint32_t NUM_WEI_Z_REGS = 2; // wei Z registers are z0, z1

    constexpr static const std::uint32_t dstXRegOffset = 26; // src X registers are x21-x26
    constexpr static const std::uint32_t numDstXRegs = KERNEL_M_SIZE; // src X registers are x21-x26
    constexpr static const std::uint32_t SRC_X_REG_OFFSET = 26; // src X registers are x21-x26
    constexpr static const std::uint32_t NUM_SRC_X_REGS = KERNEL_M_SIZE; // src X registers are x21-x26
    constexpr static const std::uint32_t WEI_X_REG_OFFSET = 17; // wei X registers are x17-x20
    constexpr static const std::uint32_t NUM_WEI_X_REGS = KERNEL_N_SIZE; // wei X registers are x17-x20

    constexpr static const std::uint32_t TMP_X_REG_OFFSET = WEI_X_REG_OFFSET; // tmp X registers are x17-x20
    constexpr static const std::uint32_t NUM_TMP_X_REGS = KERNEL_N_SIZE; // tmp X registers are x17-x20

    constexpr static const std::uint32_t TMP_REG_NUM = 27;

    std::uint64_t vectorLengthByte;

    // function input arguments
    ARMCG::XReg srcPtr = x0;
    ARMCG::XReg weiPtr = x1;
    ARMCG::XReg dstPtr = x2;
    ARMCG::XReg biaPtr = x3;
    ARMCG::XReg flags = x4;
    ARMCG::SReg alpha = s0;
    ARMCG::SReg beta = s1;
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
    ARMCG::XReg tmp = x27;
    ARMCG::WReg tmpW = w27;
};

using VCBF16Gen = MatMulKernelDefaultCodeGen<ARCH::VC, __bf16, __bf16, float, float,
    MatMulImplKinds::BF16BF16F32F32_VC>;
using VCBF16CodegenFuncPtr = void (*)(
    const __bf16 *, const __bf16 *, float *,
    const float *, MatMulDstFlags, float, float, IntType, IntType, IntType,
    IntType, IntType, IntType, IntType);

template <>
struct MatMulImplGenerator<VCBF16Gen> :
    MatmulTypeHelper<__bf16, __bf16, float, float>, ImplGenerator<GemmInfo> {
    struct DefaultImpl : MatMulImplBase {
        DefaultImpl(const GemmInfo &task, IntType nThreads);

        constexpr static std::uint32_t GetKernelMSize() noexcept;

        constexpr static std::uint32_t GetKernelNSize() noexcept;

        static std::uint32_t GetKernelKSize() noexcept;

        IntType GetMBlockSize() const noexcept override;

        IntType GetNBlockSize() const noexcept override;

        IntType GetKBlockSize() const noexcept override;

        void Pack(const void *originalMatrix, void *buffer, IntType threadID, IntType dstDim, IntType accumDim,
            IntType dstDimOffset, IntType accumDimOffset, IntType bufferOffset, IntType bufferStride, float alpha,
            MatrixKind mKind) const override;

    protected:
        void InitializeKernelK(IntType k, IntType kBlockSize,
            IntType &kBlockExecFull, IntType &kBlockExecResid) const noexcept override;

        void InitializeSingleCodegen(IntType id, IntType m, IntType n, IntType k,
            IntType currentM, IntType currentN, IntType currentK,
            SizeType offsetSrc, SizeType offsetWei, SizeType offsetDst, SizeType offsetBia,
            bool wereSrcWeiSwapped, const Attributes &attributes, bool hasAlpha) override;

        void InitializeThreadsCountImpl() override;

        void InitializePackingPolicy() override;

        IntType GetThreadBufSize(IntType m, IntType n, IntType k,
            MatrixKind mKind) const noexcept override;

        void ReserveImplSpecificParams() override;

        std::vector<std::pair<Shape, Shape>> SplitWork() override;

        template <IntType PACK_ROWS>
        static void PackSimilar(IntType rowBlock, IntType colBlock,
            IntType kBlockExec, const void *src, IntType offset, void *buffer);

        template <IntType PACK_COLS>
        static void PackDifferent(IntType rowBlock, IntType colBlock,
            IntType kBlockExec, const void *src, IntType offset, void *buffer);

        void RunSequential2D(IntType threadID, const void *src, const void *wei, void *dst, const void *bia,
            std::unique_ptr<PackStrategy> &srcPackStrategy, std::unique_ptr<PackStrategy> &weiPackStrategy,
            float alpha, float beta, const PostOpsDataPtrs &shiftedByBatchPOPtrs) const noexcept(false) override;

        void UpdateExecParams() noexcept override;

        MatMulDstFlags SetBiaFlagsExec(MatMulDstFlags biaF, float beta, bool notFirstKIter,
            bool lastKIter, float alpha = 1.0f) const noexcept override;

        std::array<VCBF16CodegenFuncPtr, NUM_18> matmulCodegenKernels;
        // these static variables are modified at impl construction stage which is performed
        // synchroniously, that's why we don't need extra syncs to prevent form writing to these vars
        inline static std::unordered_map<CodeGenParams<VCBF16Gen>,
            std::shared_ptr<VCBF16Gen>> codegens;
    };

    bool Supports(const GemmInfo &task) const noexcept override
    {
        VERBOSE_PRINT;
        bool srcMatched = (task.srcInfo.GetType().GetSize() <= sizeof(SrcType)) &&
            (task.srcInfo.GetType() != Element::TypeT::F16);
        bool weiMatched = (task.weiInfo.GetType().GetSize() <= sizeof(WeiType)) &&
            (task.weiInfo.GetType() != Element::TypeT::F16);
        return (srcMatched && weiMatched);
    }

    std::chrono::duration<float, std::milli> EstimateTime(const GemmInfo &task,
                                                            IntType nThreads) const noexcept override
    {
#ifdef KUDNNL_920F
        const float singleThreadPerformance = 75;    // experementaly measured performance in GFlops/s
#else
        const float singleThreadPerformance = 89;    // experementaly measured performance in GFlops/s
#endif
        return CalculateMatmulTimeWithReorder<__bf16, __bf16, float, float>(
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

#endif // KUDNN_MATMUL_JIT_VC_DEFAULT_IMPL_BF16BF16F32F32_HPP
