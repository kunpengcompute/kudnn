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

#ifndef KUDNN_MATMUL_JIT_MC_DEFAUT_IMPL_BFP16BFP16F32F32_HPP
#define KUDNN_MATMUL_JIT_MC_DEFAUT_IMPL_BFP16BFP16F32F32_HPP


#include <memory>
#include <vector>
#include <array>
#include <type_traits>

#include <arm_sve.h>

#include "kudnn.hpp"

#include "cg_codegen.hpp"

#include "kudnn_jit.hpp"
#include "kernels/matmul_jit/kudnn_gemm_jit.hpp"
#include "internal_funcs.hpp"

namespace KuDNN {
namespace Detail {

template <typename SrcWeiType, typename DstBiaType>
struct MatMulKernelDefaultCodeGen<ARCH::MC, SrcWeiType, SrcWeiType, DstBiaType, DstBiaType,
    MatMulImplKinds::BFP16BFP16BFP1632BFP1632_MC> :
    MatMulKernelCodeGenBase<ARCH::MC, SrcWeiType, SrcWeiType, DstBiaType, DstBiaType> {
public:
    constexpr static const std::uint32_t KERNEL_M_SIZE = 2;   // in VL
    constexpr static const std::uint32_t KERNEL_N_SIZE = 2 ;  // in VL
    constexpr static const std::uint32_t K_LOOP_UNROLL = 2;

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
        if ((lhs <= (K_LOOP_UNROLL * NUM_2)) || (rhs <= (K_LOOP_UNROLL * NUM_2))) {
            return lhs == rhs;
        } else {
            return ProcessKSize(lhs) == ProcessKSize(rhs);
        }
    }

    static IntType ProcessKSize(IntType k) noexcept
    {
        return k % (K_LOOP_UNROLL * NUM_2);
    }

protected:

    void BiasAndC(IntType SplitSize, IntType n, IntType vlw);

    void GenerateMLoopBody(IntType mLim, IntType n, IntType k);

    void GenerateNLoopBody(IntType mLim, IntType nLim, IntType k);

    void GenerateKLoop(IntType m, IntType n, IntType k, IntType vl);

    void StoreDstRegisters(IntType m, IntType n, IntType vl);

    void StoreDstRegisterPartSplitBF16(IntType SplitSize);

    // to convert f32 to (b)f16
    void Bfp16Cvt(const ARMCG::ZRegH &zd, const ARMCG::PRegBase &pg, const ARMCG::ZRegS &zn);

    // to convert (b)f16 to f32, overrides znData, znZeros registers
    void Bfp16Cvt(IntType zd0Idx, IntType zd1Idx, IntType znDataIdx,
        IntType znZerosIdx, const ARMCG::PRegBase &pg);

    void Bfp16Mopa(const ARMCG::ZARegS &zada, const ARMCG::PRegBase &pn,
        const ARMCG::PRegBase &pm, const ARMCG::ZRegH &zn, const ARMCG::ZRegH &zm);

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
    ARMCG::WReg tmpW = w29;

    using KUDNN_JIT_Generator::wzr;
    using KUDNN_JIT_Generator::s0;
    using KUDNN_JIT_Generator::s1;
    using KUDNN_JIT_Generator::x0;
    using KUDNN_JIT_Generator::x1;
    using KUDNN_JIT_Generator::x2;
    using KUDNN_JIT_Generator::x3;
    using KUDNN_JIT_Generator::x4;
    using KUDNN_JIT_Generator::x5;
    using KUDNN_JIT_Generator::x6;
    using KUDNN_JIT_Generator::x7;
    using KUDNN_JIT_Generator::x8;
    using KUDNN_JIT_Generator::x9;
    using KUDNN_JIT_Generator::x10;
    using KUDNN_JIT_Generator::x11;
    using KUDNN_JIT_Generator::w12;
    using KUDNN_JIT_Generator::w13;
    using KUDNN_JIT_Generator::w14;
    using KUDNN_JIT_Generator::w15;
    using KUDNN_JIT_Generator::x16;
    using KUDNN_JIT_Generator::x17;
    using KUDNN_JIT_Generator::x18;
    using KUDNN_JIT_Generator::x19;
    using KUDNN_JIT_Generator::x20;
    using KUDNN_JIT_Generator::x21;
    using KUDNN_JIT_Generator::x22;
    using KUDNN_JIT_Generator::x23;
    using KUDNN_JIT_Generator::x24;
    using KUDNN_JIT_Generator::x25;
    using KUDNN_JIT_Generator::w26;
    using KUDNN_JIT_Generator::w27;
    using KUDNN_JIT_Generator::w28;
    using KUDNN_JIT_Generator::x29;
    using KUDNN_JIT_Generator::w29;
    using KUDNN_JIT_Generator::xzr;
    using KUDNN_JIT_Generator::sp;
    using KUDNN_JIT_Generator::p0;
    using KUDNN_JIT_Generator::p1;
    using KUDNN_JIT_Generator::p2;
    using KUDNN_JIT_Generator::p3;
    using KUDNN_JIT_Generator::p4;
    using KUDNN_JIT_Generator::p5;
    using KUDNN_JIT_Generator::p6;
    using KUDNN_JIT_Generator::p7;
    using KUDNN_JIT_Generator::Preamble;
    using KUDNN_JIT_Generator::Smstart;
    using KUDNN_JIT_Generator::Whilelt;
    using KUDNN_JIT_Generator::Incb;
    using KUDNN_JIT_Generator::Inch;
    using KUDNN_JIT_Generator::Smstop;
    using KUDNN_JIT_Generator::Postamble;
    using KUDNN_JIT_Generator::Ptrue;
    using KUDNN_JIT_Generator::Tbz;
    using KUDNN_JIT_Generator::Tbnz;
    using KUDNN_JIT_Generator::Zero;
    using KUDNN_JIT_Generator::L;
    using KUDNN_JIT_Generator::Ldr;
    using KUDNN_JIT_Generator::Ldp;
    using KUDNN_JIT_Generator::Ldrh;
    using KUDNN_JIT_Generator::Cpy;
    using KUDNN_JIT_Generator::Dup;
    using KUDNN_JIT_Generator::Mova;
    using KUDNN_JIT_Generator::Ld1b;
    using KUDNN_JIT_Generator::Ld1h;
    using KUDNN_JIT_Generator::Ld1w;
    using KUDNN_JIT_Generator::Fmul;
    using KUDNN_JIT_Generator::Sub;
    using KUDNN_JIT_Generator::Incw;
    using KUDNN_JIT_Generator::Mul;
    using KUDNN_JIT_Generator::Lsr;
    using KUDNN_JIT_Generator::Add;
    using KUDNN_JIT_Generator::Lsl;
    using KUDNN_JIT_Generator::Subs;
    using KUDNN_JIT_Generator::Addvl;
    using KUDNN_JIT_Generator::Fmopa;
    using KUDNN_JIT_Generator::Bfmopa;
    using KUDNN_JIT_Generator::B;
    using KUDNN_JIT_Generator::St1w;
    using KUDNN_JIT_Generator::St1h;
    using KUDNN_JIT_Generator::Fadd;
    using KUDNN_JIT_Generator::Uzp1;
    using KUDNN_JIT_Generator::Trn2;
    using KUDNN_JIT_Generator::Zip1;
    using KUDNN_JIT_Generator::Zip2;
    using KUDNN_JIT_Generator::Fcvt;
    using KUDNN_JIT_Generator::Bfcvt;
    using KUDNN_JIT_Generator::Ret;
    using KUDNN_JIT_Generator::Mov;
    using KUDNN_JIT_Generator::Fmov;
    using KUDNN_JIT_Generator::Ld1rw;
    using KUDNN_JIT_Generator::Ld1rh;
    using KUDNN_JIT_Generator::Cmp;
    using KUDNN_JIT_Generator::Beq;
};

template <typename SrcWeiType, typename DstBiaType>
using MCBFP16Gen = MatMulKernelDefaultCodeGen<ARCH::MC, SrcWeiType, SrcWeiType, DstBiaType, DstBiaType,
    MatMulImplKinds::BFP16BFP16BFP1632BFP1632_MC>;
template <typename SrcWeiType, typename DstBiaType>
using MCBFP16CodegenFuncPtr = void (*)(
    const SrcWeiType *, const SrcWeiType *, DstBiaType *,
    const DstBiaType *, MatMulDstFlags, float, float, IntType, IntType, IntType,
    IntType, IntType, IntType, IntType);

template <typename SrcWeiType, typename DstBiaType>
struct MatMulImplGenerator<MCBFP16Gen<SrcWeiType, DstBiaType>> :
    MatmulTypeHelper<SrcWeiType, SrcWeiType, DstBiaType, DstBiaType>, ImplGenerator<GemmInfo> {
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
        static void PackSimilarLt(IntType mBlock, IntType kBlock,
            const void *src, IntType offset, void *buffer, IntType bufferOffset);

        // used for packing both Src and Wei matrices
        void PackDiffLt(IntType threadID, IntType mBlock, IntType kBlock,
            const void *src, IntType offset, void *buffer, IntType bufferOffset, PackingType packingType,
            IntType kBlockSize) const;

        void RunSequential2D(IntType threadID, const void *src, const void *wei, void *dst, const void *bia,
            std::unique_ptr<PackStrategy> &srcPackStrategy, std::unique_ptr<PackStrategy> &weiPackStrategy,
            float alpha, float beta, const PostOpsDataPtrs &shiftedByBatchPOPtrs) const noexcept(false) override;

        void UpdateExecParams() noexcept override;

        MatMulDstFlags SetBiaFlagsExec(MatMulDstFlags biaF, float beta, bool notFirstKIter,
            bool lastKIter, float alpha = 1.0f) const noexcept override;

        template<typename F16Type, typename Dummy = void>
        struct Calculation;

        template<typename Dummy>
        struct Calculation<__fp16, Dummy> {
            using Type = __fp16;
        };

        template<typename Dummy>
        struct Calculation<__bf16, Dummy> {
            using Type = float;
        };

        std::array<MCBFP16CodegenFuncPtr<SrcWeiType, DstBiaType>, NUM_18> matmulCodegenKernels;
        std::array<CodeGenWrapper<FastReorderKernelCodeGen<float, float>>, NUM_10> reorderPackingCodegenKernels;
        // these static variables are modified at impl construction stage which is performed
        // synchroniously, that's why we don't need extra syncs to prevent form writing to these vars
        inline static std::unordered_map<CodeGenParams<MCBFP16Gen<SrcWeiType, DstBiaType>>,
            std::shared_ptr<MCBFP16Gen<SrcWeiType, DstBiaType>>> codegens;
    };
    bool Supports(const GemmInfo &task) const noexcept override
    {
        VERBOSE_PRINT;
        bool srcMatched = (task.srcInfo.GetType() == Element::MatchType<SrcWeiType>()) ||
            (std::is_same_v<DstBiaType, float> && (task.srcInfo.GetType().GetSize() < sizeof(SrcWeiType)));
        bool weiMatched = (task.weiInfo.GetType() == Element::MatchType<SrcWeiType>()) ||
            (std::is_same_v<DstBiaType, float> && (task.srcInfo.GetType().GetSize() < sizeof(SrcWeiType)));
        bool dstMatched = (task.dstInfo.GetType() == Element::MatchType<DstBiaType>()) ||
            (std::is_same_v<DstBiaType, float> && (task.dstInfo.GetType().GetSize() < sizeof(DstBiaType)));
        bool biaMatched = true;
        if (task.biaInfo.GetTotalTensorSize() != 0) {
            biaMatched = (task.biaInfo.GetType() == Element::MatchType<DstBiaType>()) ||
                (std::is_same_v<DstBiaType, float> && (task.biaInfo.GetType().GetSize() < sizeof(DstBiaType)));
        }
        return (srcMatched && weiMatched && dstMatched && biaMatched);
    }
    std::chrono::duration<float, std::milli> EstimateTime(const GemmInfo &task,
                                                            IntType nThreads) const noexcept override
    {
        VERBOSE_PRINT;
        const float SINGLE_THREAD_PERFORMANCE = 530;    // experementaly measured performance in GFlops/s
        return CalculateMatmulTimeWithReorder<SrcWeiType, SrcWeiType, DstBiaType, DstBiaType>(
            SINGLE_THREAD_PERFORMANCE, task, nThreads);
    }
    std::shared_ptr<ImplBase<GemmInfo>> GetImpl(const GemmInfo &task, IntType nThreads) const override
    {
        VERBOSE_PRINT;
        return std::make_shared<DefaultImpl>(task, nThreads);
    }
};
}
} // KuDNN

#endif // KUDNN_MATMUL_JIT_MC_DEFAUT_IMPL_BFP16BFP16F32F32_HPP
