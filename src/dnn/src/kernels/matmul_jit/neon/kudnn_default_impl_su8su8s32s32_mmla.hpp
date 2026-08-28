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

#ifndef KUDNN_MATMUL_JIT_NEON_DEFAULT_IMPL_SU8SU8S32S32_MMLA_HPP
#define KUDNN_MATMUL_JIT_NEON_DEFAULT_IMPL_SU8SU8S32S32_MMLA_HPP

#include <memory>
#include <vector>
#include <array>
 
#include <arm_sve.h>

#include "kudnn.hpp"
#include "cg_codegen.hpp"
#include "kudnn_jit.hpp"
#include "kernels/matmul_jit/kudnn_gemm_jit.hpp"
#include "kernels/reorder/neon/su8_diff_lt/kudnn_fast_jit_impl_su8_su8_diff_lt.hpp"
#include "internal_funcs.hpp"

namespace KuDNN {
namespace Detail {

template <typename SrcType, typename WeiType, typename DstType, typename BiaType>
struct MatMulKernelDefaultCodeGen<ARCH::NEON, SrcType, WeiType, DstType, BiaType,
    MatMulImplKinds::SU8SU8S32S32_NEON_MMLA> :
    MatMulKernelCodeGenBase<ARCH::NEON, SrcType, WeiType, DstType, BiaType>
{
public:
    constexpr static const std::uint32_t KERNEL_M_SIZE  = NUM_6;
    constexpr static const std::uint32_t KERNEL_N_SIZE  = NUM_4; // in VL
    constexpr static const std::uint32_t KERNEL_K_STEP = NUM_16;
    constexpr static const std::uint32_t K_LOOP_UNROLL  = NUM_2;
    constexpr static const std::uint32_t NEON_VLW = NUM_4;

    // Kernel performs 6x4VL update of destination matrix
    // Due to peculiarities of MMLA instruction, each dst register contains multiple cols and rows of dst
    // and each src register contains multiple cols and rows of src. To handle this extra
    // in-register transpositions/zips are performed
    // The kernel has unrolled by K variable ([m x k] * [k * n] -> [m * n]) by vector length.
    //                                             x26
    //                                              |
    //                                              v
    //                                              +-------+-------+-------+-------+-------+-------+-------+-------+
    //                     Rhs ->                   | v6.s  | v7.s  | v6.s  | v7.s  | v6.s  | v7.s  | v6.s  | v7.s  |
    //                                              +-------+-------+-------+-------+-------+-------+-------+-------+
    //                   Lhs                        |       |       |       |       |       |       |       |       |
    //                    |
    //                    v                         |       |       |       |       |       |       |       |       |
    //            1st K           8th K
    //          +---------+-----+----------+  -  -  +-------+-------+-------+-------+-------+-------+-------+-------+
    //  x25 ->  | v0.s[0] | ... | v0.s[7]  |        | v8.s  | v9.s  | v10.s | v11.s | v12.s | v13.s | v14.s | v15.s |
    //  x24 ->  | v0.s[8] | ... | v0.s[15] |        | v8.s  | v9.s  | v10.s | v11.s | v12.s | v13.s | v14.s | v15.s |
    //  x23 ->  | v2.s[0] | ... | v2.s[7]  |        | v16.s | v17.s | v18.s | v19.s | v20.s | v21.s | v22.s | v23.s |
    //  x22 ->  | v2.s[8] | ... | v2.s[15] |        | v16.s | v17.s | v18.s | v19.s | v20.s | v21.s | v22.s | v23.s |
    //  x21 ->  | v4.s[0] | ... | v4.s[7]  |        | v24.s | v25.s | v26.s | v27.s | v28.s | v29.s | v30.s | v31.s |
    //  x20 ->  | v4.s[8] | ... | v4.s[15] |        | v24.s | v25.s | v26.s | v27.s | v28.s | v29.s | v30.s | v31.s |
    //          +---------+-----+----------+  -  -  +-------+-------+-------+-------+-------+-------+-------+-------+

    MatMulKernelDefaultCodeGen(IntType m, IntType n, IntType k,
        bool wereSrcWeiSwappedArg, const Attributes &attributesArg);

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
        if ((lhs <= (KERNEL_N_SIZE * NEON_VLW)) || (rhs <= (KERNEL_N_SIZE * NEON_VLW))) {
            return lhs == rhs;
        } else {
            return ProcessNSize(lhs) == ProcessNSize(rhs);
        }
    }

    static IntType ProcessNSize(IntType n) noexcept
    {
        return n % (KERNEL_N_SIZE * NEON_VLW);
    }

    static bool CompareKSize(IntType lhs, IntType rhs) noexcept
    {
        if ((lhs <= KERNEL_K_STEP) || (rhs <= KERNEL_K_STEP)) {
            return lhs == rhs;
        } else {
            return ProcessKSize(lhs) == ProcessKSize(rhs);
        }
    }

    static IntType ProcessKSize(IntType k) noexcept
    {
        return k % KERNEL_K_STEP;
    }

protected:
    void CvtAndAdd(bool applyConvert, int tmpRegIdx, int mRegIdx, int nRegIdx);

    void CvtAndStore(bool applyConvert, std::uint32_t mLim, std::uint32_t nLim, std::uint32_t nResidScalar);

    void StoreToDst(std::uint32_t mLim, std::uint32_t nLim, std::uint32_t nResidScalar);

    void GenerateMLoopBody(std::uint32_t mLim, std::uint32_t nLim, std::uint32_t k);
 
    void GenerateNLoopBody(std::uint32_t mLim, std::uint32_t nLim, std::uint32_t k, std::uint32_t nResidScalar);
 
    void GenerateAccumulation(std::uint32_t mLim, std::uint32_t nLim, std::uint32_t k);

    void Mmla(const ARMCG::VReg16B &vn, const ARMCG::VReg16B &vm,
        const ARMCG::VReg4S &vd);

    // m in range 0-2, n in range 0-7
    static std::uint32_t GetDstVRegIdx(std::uint32_t m, std::uint32_t n) noexcept
    {
        // due to peculiarities of MMLA instruction original 6x4VL data is
        // placed into 3x8 registers, so register offset is multiplied by 2
        return DST_V_REG_OFFSET + (n + m * KERNEL_N_SIZE * NUM_2) % NUM_DST_V_REGS;
    }

    // m in range 0-2, n in range 0-7
    static std::uint32_t GetTmpDstVRegIdx(std::uint32_t m, std::uint32_t n) noexcept
    {
        return m * NUM_8 + n;
    }

    // m in range 0-5, n in range 0-3
    static std::uint32_t GetStDstVRegIdx(std::uint32_t m, std::uint32_t n) noexcept
    {
        return n * NUM_2 + m % NUM_2 + m / NUM_2 * NUM_8;
    }

    // idx in range 0-7
    static std::uint32_t GetTmpVRegIdx(std::uint32_t idx) noexcept
    {
        return GetTmpDstVRegIdx(KERNEL_M_SIZE  / NUM_2 - NUM_1, KERNEL_N_SIZE  * NUM_2 - NUM_1) + NUM_1 + idx % NUM_8;
    }

    // m in range 0-5
    static std::uint32_t GetDstXRegIdx(std::uint32_t m) noexcept
    {
        return GetSrcXRegIdx(m);
    }

    // m in range 0-2, k in range 0-1
    static std::uint32_t GetSrcVRegIdx(std::uint32_t m, std::uint32_t k = 0) noexcept
    {
        // due to peculiarities of MMLA instruction we have two registers in K direction
        return SRC_V_REG_OFFSET + (m * K_LOOP_UNROLL + k) % NUM_SRC_V_REGS;
    }

    // m in range 0-2, k in range 0-1
    static std::uint32_t GetTmpSrcVRegIdx(std::uint32_t m, std::uint32_t k = 0) noexcept
    {
        return GetSrcVRegIdx(m, k) + NUM_1;
    }

    static std::uint32_t GetSrcXRegIdx(std::uint32_t m) noexcept
    {
        return SRC_X_REG_OFFSET - m % NUM_SRC_X_REGS;
    }

    // n in range 0-7
    static std::uint32_t GetWeiVRegIdx(std::uint32_t n) noexcept
    {
        return WEi_V_REG_OFFSET - n % NUM_WEI_V_REGS;
    }

    static std::uint32_t GetWeiXRegIdx() noexcept
    {
        return WEI_X_REG_OFFSET;
    }

    static std::uint32_t GetTmpXRegIdx(std::uint32_t n) noexcept
    {
        return TMP_X_REG_OFFSET + n % NUM_TMP_X_REGS;
    }

    constexpr static const std::uint32_t DST_V_REG_OFFSET = 8; // dst V registers are v8 - v31
    constexpr static const std::uint32_t NUM_DST_V_REGS = KERNEL_M_SIZE * KERNEL_N_SIZE; // dst V registers are v8 - v31
    constexpr static const std::uint32_t SRC_V_REG_OFFSET = 0; // src V registers are v0 - v5
    constexpr static const std::uint32_t NUM_SRC_V_REGS = KERNEL_M_SIZE; // src V registers are v0 - v5
    constexpr static const std::uint32_t WEi_V_REG_OFFSET = 7; // wei V registers are v7, v6
    constexpr static const std::uint32_t NUM_WEI_V_REGS = 2; // wei V registers are v7, v6

    constexpr static const std::uint32_t SRC_X_REG_OFFSET = 22; // src X registers are x17-x22
    constexpr static const std::uint32_t NUM_SRC_X_REGS = KERNEL_M_SIZE ; // src X registers are x17-x22
    constexpr static const std::uint32_t WEI_X_REG_OFFSET = 23; // wei X register is x23
    constexpr static const std::uint32_t NUM_WEI_X_REGS = 1; // wei X register is x23

    constexpr static const std::uint32_t TMP_X_REG_OFFSET = 25; // tmp X registers are x25-x29
    constexpr static const std::uint32_t NUM_TMP_X_REGS = KERNEL_N_SIZE; // tmp X registers are x25-x29

    bool wereSrcWeiSwapped;
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
    ARMCG::XReg tmp = x24;
    ARMCG::WReg tmpW = w24;

    using KUDNN_JIT_Generator::sp;
    using KUDNN_JIT_Generator::xzr;
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
    using KUDNN_JIT_Generator::x12;
    using KUDNN_JIT_Generator::x13;
    using KUDNN_JIT_Generator::x14;
    using KUDNN_JIT_Generator::x15;
    using KUDNN_JIT_Generator::x16;
    using KUDNN_JIT_Generator::x24;
    using KUDNN_JIT_Generator::w24;
    using KUDNN_JIT_Generator::s0;
    using KUDNN_JIT_Generator::s1;
    using KUDNN_JIT_Generator::Preamble;
    using KUDNN_JIT_Generator::Postamble;
    using KUDNN_JIT_Generator::Tbz;
    using KUDNN_JIT_Generator::L;
    using KUDNN_JIT_Generator::Mov;
    using KUDNN_JIT_Generator::Movi;
    using KUDNN_JIT_Generator::And;
    using KUDNN_JIT_Generator::Cmp;
    using KUDNN_JIT_Generator::Ldr;
    using KUDNN_JIT_Generator::Ldp;
    using KUDNN_JIT_Generator::Trn1;
    using KUDNN_JIT_Generator::Trn2;
    using KUDNN_JIT_Generator::Uzp1;
    using KUDNN_JIT_Generator::Uzp2;
    using KUDNN_JIT_Generator::Str;
    using KUDNN_JIT_Generator::Fmov;
    using KUDNN_JIT_Generator::Add;
    using KUDNN_JIT_Generator::Sub;
    using KUDNN_JIT_Generator::Mul;
    using KUDNN_JIT_Generator::Smmla;
    using KUDNN_JIT_Generator::Ummla;
    using KUDNN_JIT_Generator::Usmmla;
    using KUDNN_JIT_Generator::B;
    using KUDNN_JIT_Generator::Ret;
    using KUDNN_JIT_Generator::Fmul;
    using KUDNN_JIT_Generator::Fadd;
    using KUDNN_JIT_Generator::Fcvtzs;
    using KUDNN_JIT_Generator::Frinti;
    using KUDNN_JIT_Generator::Scvtf;
    using KUDNN_JIT_Generator::Beq;
    using KUDNN_JIT_Generator::Bne;
};

template <typename SrcType, typename WeiType, typename DstType, typename BiaType>
using NeonSU8MmlaGen = MatMulKernelDefaultCodeGen<ARCH::NEON, SrcType, WeiType, DstType, BiaType,
    MatMulImplKinds::SU8SU8S32S32_NEON_MMLA>;
template <typename SrcType, typename WeiType, typename DstType, typename BiaType>
using NeonSU8MmlaCodegenFuncPtr = void (*)(
    const SrcType *, const WeiType *, DstType *,
    const BiaType *, MatMulDstFlags, float, float, IntType, IntType, IntType,
    IntType, IntType, IntType, IntType);

template <typename SrcType, typename WeiType, typename DstType, typename BiaType>
struct MatMulImplGenerator<NeonSU8MmlaGen<SrcType, WeiType, DstType, BiaType>> :
    MatmulTypeHelper<SrcType, WeiType, DstType, BiaType>, ImplGenerator<GemmInfo> {
    struct DefaultImpl : MatMulImplBase {
        DefaultImpl(const GemmInfo &task, IntType nThreads);

        constexpr static std::uint32_t GetKernelMSize() noexcept;

        constexpr static std::uint32_t GetKernelNSize() noexcept;

        constexpr static std::uint32_t GetKernelKSize() noexcept;

        IntType GetMBlockSize() const noexcept override;

        IntType GetNBlockSize() const noexcept override;

        IntType GetKBlockSize() const noexcept override;

    protected:

        void InitializeSingleCodegen(IntType id, IntType m, IntType n, IntType k,
            IntType currentM, IntType currentN, IntType currentK,
            SizeType offsetSrc, SizeType offsetWei, SizeType offsetDst, SizeType offsetBia,
            bool wereSrcWeiSwapped, const Attributes &attributes, bool hasAlpha) override;

        void InitializeReorderCodeGens(IntType m, IntType n, IntType k,
            IntType mFull, IntType nFull) override;

        void InitializeThreadsCountImpl() override;

        void InitializePackingPolicy() override;

        IntType GetThreadBufSize(IntType m, IntType n, IntType k,
            MatrixKind mKind) const noexcept override;

        void ReserveImplSpecificParams() override;

        std::vector<std::pair<Shape, Shape>> SplitWork() override;

        static void PackBCol(const WeiType* B_ptr, WeiType* B_buffer, IntType N, IntType K, IntType ldb);

        static void PackBRow(const WeiType* B_ptr, WeiType* B_buffer, IntType N, IntType K, IntType ldb);

        void RunSequential2D(IntType threadID, const void *src, const void *wei, void *dst, const void *bia,
            std::unique_ptr<PackStrategy> &srcPackStrategy, std::unique_ptr<PackStrategy> &weiPackStrategy,
            float alpha, float beta, const PostOpsDataPtrs &shiftedByBatchPOPtrs) const noexcept(false) override;

        void UpdateExecParams() noexcept override;

        MatMulDstFlags SetBiaFlagsExec(MatMulDstFlags biaF, float beta, bool notFirstKIter,
            bool lastKIter, float alpha = 1.0f) const noexcept override;

        std::array<NeonSU8MmlaCodegenFuncPtr<SrcType, WeiType, DstType, BiaType>, NUM_18> matmulCodegenKernels;
        std::array<CodeGenWrapper<FastReorderSU8SU8DiffLtKernelCodeGen<SrcType, SrcType>>,
            NUM_12> packAColCodegenKernels;
        // these static variables are modified at impl construction stage which is performed
        // synchroniously, that's why we don't need extra syncs to prevent form writing to these vars
        inline static std::unordered_map<CodeGenParams<NeonSU8MmlaGen<SrcType, WeiType, DstType, BiaType>>,
            std::shared_ptr<NeonSU8MmlaGen<SrcType, WeiType, DstType, BiaType>>> codegens;
    };

    bool Supports(const GemmInfo &task) const noexcept override
    {
        VERBOSE_PRINT;
        bool srcMatched = task.srcInfo.GetType() == Element::MatchType<SrcType>();
        bool weiMatched = task.weiInfo.GetType() == Element::MatchType<WeiType>();
        bool dstWiderImpl = (task.dstInfo.GetType().GetSize() <= sizeof(DstType)) &&
            task.dstInfo.GetType().IsIntegral();
        bool biaWiderImpl = (task.biaInfo.GetTotalTensorSize() != 0) ? (
            (task.biaInfo.GetType().GetSize() <= sizeof(BiaType)) && task.biaInfo.GetType().IsIntegral()) : true;
        return (srcMatched && weiMatched && dstWiderImpl && biaWiderImpl);
    }

    std::chrono::duration<float, std::milli> EstimateTime(const GemmInfo &task,
                                                            IntType nThreads) const noexcept override
    {
#ifdef KUDNNL_920F
        const float singleThreadPerformance = 150;    // experementaly measured performance in GFlops/s
#else
        const float singleThreadPerformance = 380;    // experementaly measured performance in GFlops/s
#endif
        return CalculateMatmulTimeWithReorder<SrcType, WeiType, BiaType, DstType>(
            singleThreadPerformance, task, nThreads);
    }
    std::shared_ptr<ImplBase<GemmInfo>> GetImpl(const GemmInfo &task, IntType nThreads) const override
    {
        return std::make_shared<DefaultImpl>(task, nThreads);
    }
};

} // Detail
} // KuDNN

#endif // KUDNN_MATMUL_JIT_NEON_DEFAULT_IMPL_SU8SU8S32S32_MMLA_HPP
