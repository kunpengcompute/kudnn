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

#ifndef KUDNN_MATMUL_JIT_MatrixComputation_DEFAUT_IMPL_FP16FP16F32F32_HPP
#define KUDNN_MATMUL_JIT_MatrixComputation_DEFAUT_IMPL_FP16FP16F32F32_HPP

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

template<>
struct MatMulKernelDefaultCodeGenImpl<ARCH::MatrixComputation, __fp16, __fp16, float, float, MatmulDummyType> :
    MatMulKernelCodeGenBase<ARCH::MatrixComputation, __fp16, __fp16, float, float> {
public:
    constexpr static const std::uint32_t kernelMSize = 2;   // in VL
    constexpr static const std::uint32_t kernelNSize = 2 ;  // in VL
    constexpr static const std::uint32_t kLoopUnroll = 2;

    constexpr static IntType GetMBlockSize(IntType m, IntType n, IntType k) noexcept
    {
        Service::Unused(m, n, k);
        return 224; // 224 experimentaly measured constants to obtain better perf
    }
    
    constexpr static IntType GetNBlockSize(IntType m, IntType n, IntType k) noexcept
    {
        Service::Unused(m, n, k);
        return 64; // 64 is experimentaly measured  constants to obtain better perf
    }

    constexpr static IntType GetKBlockSize(IntType m, IntType n, IntType k) noexcept
    {
        Service::Unused(m, n, k);
        return 512; // 512 is experimentaly measured  constants to obtain better perf
    }

    // Kernel performs 2VLx2VL update of destination matrix via
    // outer product computations inside loops over K.
    // Registers za0-3.s are used for accumulation (then will be stored to destination matrix),
    // registerrs z0, z1, z4, z5 are used to store source matrix values,
    // registers z2, z3, z6, z7 are used to store weights matrix values.
    // The kernel has unroll by K variable ([m x k] * [k * n] -> [m * n]) by 2.
    //                                   +-------+-------+
    //                 1st K iteration   | z2.s  | z3.s  |
    //     Weights ->                    +-------+-------+
    //                 2nd K iteration   | z6.s  | z7.s  |
    //                                   +-------+-------+
    //                  Source           |       |       |
    //                    |
    //                    v              |       |       |
    //            1st K   2nd K
    //          +-------+-------+  -  -  +-------+-------+
    //          | z0.s  | z4.s  |        | za0.s | za1.s |
    //          +-------+-------+        +-------|-------+
    //          | z1.s  | z5.s  |        | za2.s | za3.s |
    //          +--- ---+-------+  -  -  +-------+-------+
    MatMulKernelDefaultCodeGenImpl(IntType m, IntType n, IntType k,
        IntType srcOffsetArg,
        IntType weiOffsetArg,
        IntType dstOffsetArg,
        IntType biaOffsetArg);
protected:

    void InitializeDstRegisters(IntType m, IntType n, IntType vl);

    void GenerateKLoop(IntType m, IntType n, IntType k, IntType vl);

    void StoreDstRegisters(IntType m, IntType n, IntType vl);

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
    ARMCG::WReg tmpLDStC0 = w12;
    ARMCG::WReg tmpLDStC1 = w13;
    ARMCG::WReg tmpLDStC2 = w14;
    ARMCG::WReg tmpLDStC3 = w15;
    ARMCG::XReg biaOffset = x16;
    ARMCG::XReg srcPtrCopy = x17;
    ARMCG::XReg weiPtrCopy = x18;
    ARMCG::XReg dstPtrCopy = x19;
    ARMCG::XReg biaPtrCopy = x20;
    ARMCG::XReg tmp = x21;
    ARMCG::XReg mTmpCopy = x22;
    ARMCG::XReg nTmpCopy = x23;
    ARMCG::XReg kTmpCopy = x24;
    ARMCG::WReg tmpW = w25;
};

template <>
struct MatMulKernelDefaultCodeGen<ARCH::MatrixComputation, __fp16, __fp16, float, float> :
    MatMulKernelDefaultCodeGenImpl<ARCH::MatrixComputation, __fp16, __fp16, float, float, MatmulDummyType> {
    MatMulKernelDefaultCodeGen(IntType m, IntType n, IntType k,
        IntType srcOffsetArg,
        IntType weiOffsetArg,
        IntType dstOffsetArg,
        IntType biaOffsetArg) : MatMulKernelDefaultCodeGenImpl(m, n, k, srcOffsetArg,
            weiOffsetArg, dstOffsetArg, biaOffsetArg)
        {
            VERBOSE_PRINT;
        }
};

template <>
struct MatMulImplGenerator<MatMulKernelDefaultCodeGen<ARCH::MatrixComputation, __fp16, __fp16, float, float>> :
    MatmulTypeHelper<__fp16, __fp16, float, float>, ImplGenerator<GemmInfo> {
    using CodeGen = MatMulKernelDefaultCodeGen<ARCH::MatrixComputation, __fp16, __fp16, float, float>;
    struct DefaultImpl : MatMulImplBase {
        DefaultImpl(const GemmInfo &task, IntType nThreads);

        static std::uint32_t GetKernelMSize() noexcept;

        static std::uint32_t GetKernelNSize() noexcept;
    protected:
        void InitializeCodeGens(IntType m, IntType n, IntType k, IntType vl,
            SizeType offsetSrc, SizeType offsetWei, SizeType offsetDst, SizeType offsetBia,
            bool wereSrcWeiSwapped) override;

        IntType GetThreadBufSize(IntType m, IntType n, IntType k, IntType vl) noexcept override;

        void ReserveImplSpecificParams() override;

        std::vector<std::pair<Shape, Shape>> SplitWork() override;

        // used for packing both Src and Wei matrices
        template <bool AlphaMul = false>
        static void PackSimilarLt(IntType m, IntType k,
            const void *src, IntType offset, void *buffer, IntType bufferOffset, float alpha = 1.0f);

        // used for packing both Src and Wei matrices
        template <bool AlphaMul = false>
        static void PackDiffLt(IntType mBlock, IntType kBlock,
            const void *src, IntType offset, void *buffer, IntType bufferOffset, float alpha = 1.0f);

        void RunSequential2D(IntType threadId, const void *src, const void *wei, void *dst,
            const void *bia, void *buffer, float alpha, float beta) const noexcept(false) override;

        void UpdateExecParams() noexcept override;
        std::vector<std::array<CodeGenWrapper<CodeGen>, NUM_8>> perThreadImpls;
    };
    bool Supports(const GemmInfo &task) const noexcept override
    {
        VERBOSE_PRINT;
        // Supported cases: src & wei < fp16, exclude fb16
        bool srcMatched = (task.srcInfo.GetType().GetSize() <= sizeof(__fp16)) &&
            (task.srcInfo.GetType() != Element::TypeT::BF16);
        bool weiMatched = (task.weiInfo.GetType().GetSize() <= sizeof(__fp16)) &&
            (task.weiInfo.GetType() != Element::TypeT::BF16);
        return (srcMatched && weiMatched);
    }
    float EstimateTime(const GemmInfo &task, IntType nThreads) const noexcept override
    {
        VERBOSE_PRINT;
        Service::Unused(task, nThreads);
        return 0.26f; // currently return lower then float
    }
    std::shared_ptr<ImplBase<GemmInfo>> GetImpl(const GemmInfo &task, IntType nThreads) const override
    {
        VERBOSE_PRINT;
        return std::make_shared<DefaultImpl>(task, nThreads);
    }
};

} // Detail
} // KuDNN


#endif // KUDNN_MATMUL_JIT_MatrixComputation_DEFAUT_IMPL_F32F32F32F32_HPP
