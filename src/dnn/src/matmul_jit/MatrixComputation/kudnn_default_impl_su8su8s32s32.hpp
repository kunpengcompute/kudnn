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

#ifndef KUDNN_MATMUL_JIT_MatrixComputation_DEFAUT_IMPL_SU8SU8S32S32_HPP
#define KUDNN_MATMUL_JIT_MatrixComputation_DEFAUT_IMPL_SU8SU8S32S32_HPP

#include <memory>
#include <vector>
#include <array>
#include <type_traits>
#include <arm_sve.h>

#include "kudnn.hpp"
#include "cg_codegen.hpp"
#include "kudnn_jit.hpp"
#include "matmul_jit/kudnn_gemm_jit.hpp"

namespace KuDNN {
namespace Detail {

template <typename SrcType, typename WeiType, typename DstType, typename BiaType>
constexpr bool IsIntegerMatmul() noexcept
{
    constexpr bool isSrcInteger8 = std::is_same_v<SrcType, std::int8_t> || std::is_same_v<SrcType, std::uint8_t>;
    constexpr bool isWeiInteger8 = std::is_same_v<WeiType, std::int8_t> || std::is_same_v<WeiType, std::uint8_t>;
    constexpr bool isDstInteger32 = std::is_same_v<DstType, std::int32_t>;
    constexpr bool isBiaInteger32 = std::is_same_v<BiaType, std::int32_t>;
    return (isSrcInteger8 && isWeiInteger8 && isDstInteger32 && isBiaInteger32);
}

template <typename SrcType, typename WeiType, typename DstType, typename BiaType>
struct MatMulKernelDefaultCodeGenImpl<
    ARCH::MatrixComputation, SrcType, WeiType, DstType, BiaType,
    typename std::enable_if<IsIntegerMatmul<SrcType, WeiType, DstType, BiaType>(), MatmulDummyType>::type>
    : MatMulKernelCodeGenBase<ARCH::MatrixComputation, SrcType, WeiType, DstType, BiaType> {
public:
    constexpr static const std::uint32_t kernelMSize = 2; // in VL
    constexpr static const std::uint32_t kernelNSize = 2; // in VL
    constexpr static const std::uint32_t kLoopUnroll = 2;
    constexpr static IntType GetMBlockSize(IntType m, IntType n, IntType k) noexcept
    {
        Service::Unused(m, n, k);
        return 192; // 192 is experimentally measured constant to obtain better perf
    }
    constexpr static IntType GetNBlockSize(IntType m, IntType n, IntType k) noexcept
    {
        Service::Unused(m, n, k);
        return 128; // 128 is experimentally measured constant to obtain better perf
    }
    constexpr static IntType GetKBlockSize(IntType m, IntType n, IntType k) noexcept
    {
        Service::Unused(m, n, k);
        // all the constants(1024,512,256,64) are experimentally measured and give optimal performance configuration
        return (m + n >= 64) ? ((k >= 1024) ? 1024 : 512) : ((k >= 512) ? 512 : 256);
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

    MatMulKernelDefaultCodeGenImpl(IntType m, IntType n, IntType k, IntType biaOffsetArg, bool wereSrcWeiSwappedArg);

protected:
    void InitializeDstRegisters(IntType m, IntType n, IntType vl);

    void GenerateKLoop(IntType m, IntType n, IntType k, IntType vl);

    void StoreDstRegisters(IntType m, IntType n, IntType vl);

    void Int8Mopa(const ARMCG::ZARegS &zada, const ARMCG::PRegBase &pn, const ARMCG::PRegBase &pm,
                  const ARMCG::ZRegB &zn, const ARMCG::ZRegB &zm);
    bool wereSrcWeiSwapped;
    // function input arguments
    ARMCG::XReg srcPtr = x0;
    ARMCG::XReg weiPtr = x1;
    ARMCG::XReg dstPtr = x2;
    ARMCG::XReg biaPtr = x3;
    ARMCG::XReg flags = x4;
    ARMCG::XReg dstOffset = x5;
    // other used registers
    ARMCG::XReg mVal = x6;
    ARMCG::XReg nVal = x7;
    ARMCG::XReg kVal = x8;
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
    using KUDNN_JIT_Generator::x0;
    using KUDNN_JIT_Generator::x1;
    using KUDNN_JIT_Generator::x2;
    using KUDNN_JIT_Generator::x3;
    using KUDNN_JIT_Generator::x4;
    using KUDNN_JIT_Generator::x5;
    using KUDNN_JIT_Generator::s0;
    using KUDNN_JIT_Generator::x6;
    using KUDNN_JIT_Generator::x7;
    using KUDNN_JIT_Generator::x8;
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
    using KUDNN_JIT_Generator::w25;
    using KUDNN_JIT_Generator::p0;
    using KUDNN_JIT_Generator::p1;
    using KUDNN_JIT_Generator::p2;
    using KUDNN_JIT_Generator::p3;
    using KUDNN_JIT_Generator::p4;
    using KUDNN_JIT_Generator::p5;
    using KUDNN_JIT_Generator::Preamble;
    using KUDNN_JIT_Generator::Smstart;
    using KUDNN_JIT_Generator::Whilelt;
    using KUDNN_JIT_Generator::Incb;
    using KUDNN_JIT_Generator::Smstop;
    using KUDNN_JIT_Generator::Postamble;
    using KUDNN_JIT_Generator::Ptrue;
    using KUDNN_JIT_Generator::Tbz;
    using KUDNN_JIT_Generator::Zero;
    using KUDNN_JIT_Generator::L;
    using KUDNN_JIT_Generator::Ldr;
    using KUDNN_JIT_Generator::Cpy;
    using KUDNN_JIT_Generator::Mova;
    using KUDNN_JIT_Generator::Ld1b;
    using KUDNN_JIT_Generator::Ld1w;
    using KUDNN_JIT_Generator::Fmul;
    using KUDNN_JIT_Generator::Add;
    using KUDNN_JIT_Generator::Subs;
    using KUDNN_JIT_Generator::Addvl;
    using KUDNN_JIT_Generator::Smopa;
    using KUDNN_JIT_Generator::Sumopa;
    using KUDNN_JIT_Generator::Usmopa;
    using KUDNN_JIT_Generator::Umopa;
    using KUDNN_JIT_Generator::B;
    using KUDNN_JIT_Generator::St1w;
    using KUDNN_JIT_Generator::Fadd;
    using KUDNN_JIT_Generator::Ret;
};

template <typename SrcType, typename WeiType, typename DstType, typename BiaType>
struct MatMulKernelDefaultCodeGen<ARCH::MatrixComputation, SrcType, WeiType, DstType, BiaType>
    : MatMulKernelDefaultCodeGenImpl<
          ARCH::MatrixComputation, SrcType, WeiType, DstType, BiaType,
          typename std::enable_if<IsIntegerMatmul<SrcType, WeiType, DstType, BiaType>(), MatmulDummyType>::type> {
    MatMulKernelDefaultCodeGen(IntType m, IntType n, IntType k, IntType biaOffsetArg, bool wereSrcWeiSwapped)
        : MatMulKernelDefaultCodeGenImpl<
              ARCH::MatrixComputation, SrcType, WeiType, DstType, BiaType,
              typename std::enable_if<IsIntegerMatmul<SrcType, WeiType, DstType, BiaType>(), MatmulDummyType>::type>(
              m, n, k, biaOffsetArg, wereSrcWeiSwapped)
    {
        VERBOSE_PRINT;
    }
};

template <typename SrcType, typename WeiType, typename DstType, typename BiaType>
struct MatMulImplGenerator<MatMulKernelDefaultCodeGen<ARCH::MatrixComputation, SrcType, WeiType, DstType, BiaType>>
    : MatmulTypeHelper<SrcType, WeiType, DstType, BiaType>
    , ImplGenerator<GemmInfo> {
    using CodeGen = MatMulKernelDefaultCodeGen<ARCH::MatrixComputation, SrcType, WeiType, DstType, BiaType>;
    struct DefaultImpl : MatMulImplBase {
        DefaultImpl(const GemmInfo &task, IntType nThreads);

        static std::uint32_t GetKernelMSize() noexcept;

        static std::uint32_t GetKernelNSize() noexcept;

    protected:
        void InitializeCodeGens(IntType m, IntType n, IntType k, IntType vl, SizeType offsetSrc, SizeType offsetWei,
                                SizeType offsetDst, SizeType offsetBia, bool wereSrcWeiSwapped) override;

        IntType GetThreadBufSize(IntType m, IntType n, IntType k, IntType vlb) noexcept override;

        void ReserveImplSpecificParams() override;

        std::vector<std::pair<Shape, Shape>> SplitWork() override;

        // used for packing both Src and Wei matrices
        static void PackSimilarLt(IntType mBlock, IntType kBlock, const void *src, IntType offset, void *buffer,
                                  IntType bufferOffset);

        // used for packing both Src and Wei matrices
        static void PackDiffLt(IntType mBlock, IntType kBlock, const void *src, IntType offset, void *buffer,
                               IntType bufferOffset);

        void RunSequential2D(IntType threadId, const void *src, const void *wei, void *dst, const void *bia,
                             void *buffer, float alpha, float beta) const noexcept(false) override;

        void UpdateExecParams() noexcept override;
        std::vector<std::array<CodeGenWrapper<CodeGen>, NUM_8>> perThreadImpls;
    };
    bool Supports(const GemmInfo &task) const noexcept override
    {
        VERBOSE_PRINT;
        bool srcMatched = task.srcInfo.GetType() == Element::MatchType<SrcType>();
        bool weiMatched = task.weiInfo.GetType() == Element::MatchType<WeiType>();
        bool dstWiderImpl = (task.dstInfo.GetType().GetSize() <= sizeof(DstType)) &&
                            task.dstInfo.GetType().IsIntegral();
        bool biaWiderImpl =
            (task.biaInfo.GetTotalTensorSize() != 0) ?
                ((task.biaInfo.GetType().GetSize() <= sizeof(BiaType)) && task.biaInfo.GetType().IsIntegral()) :
                true;
        return (srcMatched && weiMatched && dstWiderImpl && biaWiderImpl);
    }
    float EstimateTime(const GemmInfo &task, IntType nThreads) const noexcept override
    {
        VERBOSE_PRINT;
        Service::Unused(task, nThreads);

        return 0.25f;
    }
    std::shared_ptr<ImplBase<GemmInfo>> GetImpl(const GemmInfo &task, IntType nThreads) const override
    {
        VERBOSE_PRINT;
        return std::make_shared<DefaultImpl>(task, nThreads);
    }
};

} // namespace Detail
} // namespace KuDNN

#endif // KUDNN_MATMUL_JIT_MatrixComputation_DEFAUT_IMPL_SU8SU8S32S32_HPP
