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

#ifndef KUDNN_MATMUL_JIT_MC_DEFAUT_IMPL_F32F32F32F32_HPP
#define KUDNN_MATMUL_JIT_MC_DEFAUT_IMPL_F32F32F32F32_HPP

#include <memory>
#include <vector>
#include <array>

#include "kudnn.hpp"

#include "kudnn_jit.hpp"
#include "internal_funcs.hpp"
#include "cg_codegen.hpp"
#include "kernels/matmul_jit/kudnn_gemm_jit.hpp"

namespace KuDNN {
namespace Detail {

template <>
struct MatMulKernelDefaultCodeGen<ARCH::MC, float, float, float, float, MatMulImplKinds::F32F32F32F32_MC> :
    MatMulKernelCodeGenBase<ARCH::MC, float, float, float, float> {
public:
    constexpr static const std::uint32_t KERNEL_M_SIZE = NUM_2; // in VL
    constexpr static const std::uint32_t KERNEL_N_SIZE = NUM_2; // in VL
    constexpr static const std::uint32_t K_LOOP_UNROLL = NUM_2;

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
    //                Source             |       |       |
    //                  |
    //                  v                |       |       |
    //            1st K   2nd K
    //          +-------+-------+  -  -  +-------+-------+
    //          | z0.s  | z4.s  |        | za0.s | za1.s |
    //          +-------+-------+        +-------|-------+
    //          | z1.s  | z5.s  |        | za2.s | za3.s |
    //          +--- ---+-------+  -  -  +-------+-------+

    MatMulKernelDefaultCodeGen(IntType m, IntType n, IntType k,
        const Attributes &attributesArg);

    static bool CompareMSize(IntType lhs, IntType rhs) noexcept
    {
        if ((lhs <= static_cast<IntType>(KERNEL_M_SIZE * Vector::Length<float>())) ||
            (rhs <= static_cast<IntType>(KERNEL_M_SIZE * Vector::Length<float>()))) {
            return lhs == rhs;
        } else {
            return ProcessMSize(lhs) == ProcessMSize(rhs);
        }
    }

    static IntType ProcessMSize(IntType m) noexcept
    {
        return m % (KERNEL_M_SIZE * Vector::Length<float>());
    }

    static bool CompareNSize(IntType lhs, IntType rhs) noexcept
    {
        if ((lhs <= static_cast<IntType>(KERNEL_N_SIZE * Vector::Length<float>())) ||
            (rhs <= static_cast<IntType>(KERNEL_N_SIZE * Vector::Length<float>()))) {
            return Math::CeilDiv(lhs, Vector::Length<float>()) == Math::CeilDiv(rhs, Vector::Length<float>());
        } else {
            return ProcessNSize(lhs) == ProcessNSize(rhs);
        }
    }

    static IntType ProcessNSize(IntType n) noexcept
    {
        return Math::CeilDiv(n, Vector::Length<float>()) % KERNEL_N_SIZE;
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

    void BiasAndC(IntType SplitSize, IntType n, IntType vlw);
    void GenerateMLoopBody(IntType mLim, IntType n, IntType k);
    void GenerateNLoopBody(IntType mLim, IntType nLim, IntType k);
    void GenerateKLoop(IntType mLim, IntType nLim, IntType k, IntType vl);

    void MulAdd(IntType dstIdx, IntType argIdx, IntType pregIdx, IntType betaIdx);
    void MoveDstRegistersPartSplit(IntType m, IntType n, IntType vl, IntType regInd1, IntType regInd2, IntType RedImm);
    void FillDstRegs(IntType m, IntType n, IntType vl, std::vector<SizeType>& dstRegsIDs);
    void StoreDstRegistersPartSplit(IntType m, IntType n, IntType vl);
    void AddPair(IntType n, IntType vlw, IntType dstIdx1, IntType dstIdx2, IntType argIdx1, IntType argIdx2);

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
    ARMCG::XReg dstOffset = x8;
    ARMCG::XReg biaOffset = x9;
    ARMCG::XReg srcPtrCopy = x10;
    ARMCG::XReg weiPtrCopy = x11;
    ARMCG::WReg tmpLDStC0 = w12;
    ARMCG::WReg tmpLDStC1 = w13;
    ARMCG::WReg tmpLDStC2 = w14;
    ARMCG::WReg tmpLDStC3 = w15;
    ARMCG::XReg dstPtrCopy = x16;
    ARMCG::XReg biaPtrCopy = x17;
    ARMCG::XReg mLoopCtr = x18;
    ARMCG::XReg tmpMMinusMLoopCtr = x19;
    ARMCG::XReg nLoopCtr = x20;
    ARMCG::XReg tmpNMinusNLoopCtr = x21;
    ARMCG::XReg dstPtrLoopCopy = x22;
    ARMCG::XReg dstPtrStore = x23;
    ARMCG::XReg dstPtrSave = x24;
    ARMCG::XReg kTmpCopy = x25;
    ARMCG::WReg alphaW = w26;
    ARMCG::WReg betaW = w27;
    ARMCG::WReg fOne = w28;
    ARMCG::XReg tmp = x29;
};

using MCF32Gen = MatMulKernelDefaultCodeGen<ARCH::MC, float, float, float, float,
    MatMulImplKinds::F32F32F32F32_MC>;
using MCF32CodegenFuncPtr = void (*)(
    const float *, const float *, float *,
    const float *, MatMulDstFlags, float, float, IntType, IntType, IntType,
    IntType, IntType, IntType, IntType);

template <>
struct MatMulImplGenerator<MatMulKernelDefaultCodeGen<ARCH::MC, float, float, float, float,
    MatMulImplKinds::F32F32F32F32_MC>> :
    MatmulTypeHelper<float, float, float, float>, ImplGenerator<GemmInfo> {
    struct DefaultImpl : MatMulImplBase {
        DefaultImpl(const GemmInfo &task, IntType nThreads);

        static std::uint32_t GetKernelMSize() noexcept;

        static std::uint32_t GetKernelNSize() noexcept;

        IntType GetMBlockSize() const noexcept override;

        IntType GetNBlockSize() const noexcept override;

        IntType GetKBlockSize() const noexcept override;

        void Pack(const void *originalMatrix, void *buffer, IntType threadID, IntType dstDim, IntType accumDim,
            IntType dstDimOffset, IntType accumDimOffset, IntType bufferOffset, IntType bufferStride, float alpha,
            MatrixKind mKind) const override;

    protected:

        void InitializeSingleCodegen(IntType id, IntType m, IntType n, IntType k,
            IntType currentM, IntType currentN, IntType currentK,
            SizeType offsetSrc, SizeType offsetWei, SizeType offsetDst, SizeType offsetBia,
            bool wereSrcWeiSwapped, const Attributes &attributes, bool hasAlpha) override;

        void InitializeReorderCodeGens(IntType m, IntType n, IntType k,
            IntType mFull, IntType nFull) override;

        void InitializePackingPolicy() override;
        
        IntType GetThreadBufSize(IntType m, IntType n, IntType k,
            MatrixKind mKind) const noexcept override;

        void ReserveImplSpecificParams() override;

        std::vector<std::pair<Shape, Shape>> SplitWork() override;

        // used for packing both Src and Wei matrices
        static void PackSimilarLt(IntType m, IntType k,
            const float *src, IntType offset, float *buffer, IntType bufferOffset);

        // used for packing both Src and Wei matrices
        void PackDiffLt(IntType threadID, IntType m, IntType k,
            const float *src, IntType offset, float *buffer, IntType bufferOffset,
            PackingType packingType, IntType kBlockSize) const;

        void RunSequential2D(IntType threadID, const void *src, const void *wei, void *dst, const void *bia,
            std::unique_ptr<PackStrategy> &srcPackStrategy, std::unique_ptr<PackStrategy> &weiPackStrategy,
            float alpha, float beta, const PostOpsDataPtrs &shiftedByBatchPOPtrs) const noexcept(false) override;

        void UpdateExecParams() noexcept override;

        MatMulDstFlags SetBiaFlagsExec(MatMulDstFlags biaF, float beta, bool notFirstKIter,
            bool lastKIter, float alpha = 1.0f) const noexcept override;

        std::array<MCF32CodegenFuncPtr, NUM_18> matmulCodegenKernels;
        std::array<CodeGenWrapper<FastReorderKernelCodeGen<float, float>>, NUM_20> reorderPackingCodegenKernels;
        // these static variables are modified at impl construction stage which is performed
        // synchroniously, that's why we don't need extra syncs to prevent form writing to these vars
        inline static std::unordered_map<CodeGenParams<MCF32Gen>,
            std::shared_ptr<MCF32Gen>> codegens;
    };
    bool Supports(const GemmInfo &task) const noexcept override
    {
        Service::Unused(task);
        VERBOSE_PRINT;
        return true;
    }
    std::chrono::duration<float, std::milli> EstimateTime(const GemmInfo &task,
                                                            IntType nThreads) const noexcept override
    {
        VERBOSE_PRINT;
        const float SINGLE_THREAD_PERFORMANCE = 300;    // experementaly measured performance in GFlops/s
        return CalculateMatmulTimeWithReorder<SrcType, WeiType, DstType, BiaType>(
            SINGLE_THREAD_PERFORMANCE, task, nThreads);
    }
    std::shared_ptr<ImplBase<GemmInfo>> GetImpl(const GemmInfo &task, IntType nThreads) const override
    {
        VERBOSE_PRINT;
        return std::make_shared<DefaultImpl>(task, nThreads);
    }
};

} // Detail
} // KuDNN

#endif // KUDNN_MATMUL_JIT_MC_DEFAUT_IMPL_F32F32F32F32_HPP
