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

#ifndef KUDNN_GEMM_JIT_HPP
#define KUDNN_GEMM_JIT_HPP

#include <algorithm>
#include <memory>
#include <atomic>

#include "arm_sve.h"

#include <omp.h>

#include "kudnn.hpp"

#include "internal_funcs.hpp"
#include "kudnn_jit.hpp"
#include "kudnn_jit_attributes.hpp"
#include "kudnn_wrapper_helpers.hpp"
#include "kudnn_memory_buffer.hpp"
#include "kernels/matmul_jit/kudnn_gemm_helpers.hpp"
#include "threading_impl.hpp"

namespace KuDNN {
namespace Detail {

enum class MatMulImplKinds {
    F32F32F32F32_NEON,
    SU8SU8S32S32_NEON_MMLA,
    F1632F1632F1632F1632_VC,
    SU8SU8S32S32_VC,
    SU8SU8S32S32_VC_MMLA,
    BF16BF16F32F32_VC,
    F32F32F32F32_MC,
    SU8SU8S32S32_MC,
    BFP16BFP16BFP1632BFP1632_MC
};

enum class MatrixKind {
    SRC,
    WEI,
    DST,
    BIA
};

enum class PackingPolicy : std::uint32_t {
    NO_SYNC = 0,
    SYNC_DST_DIM_SRC = 1,
    SYNC_DST_DIM_WEI = 2,
    SYNC_DST_DIM_SRC_WEI = 3,
    SYNC_ACC_DIM_SRC = 4,
    SYNC_ACC_DIM_WEI = 8,
    SYNC_ACC_DIM_SRC_WEI = 12
};

inline bool operator & (PackingPolicy lhs, PackingPolicy rhs)
{
    return static_cast<std::uint32_t>(lhs) & static_cast<std::uint32_t>(rhs);
}

struct LoopCtrHelper {
    IntType start;
    IntType size;
    IntType step;
};

template <ARCH Arch, typename SrcType, typename WeiType, typename DstType, typename BiaType>
struct MatMulKernelCodeGenBase : KUDNN_JIT_Generator_With_Attributes,
    MatmulTypeHelper<SrcType, WeiType, DstType, BiaType> {
    explicit MatMulKernelCodeGenBase(const Attributes &attributesArg)
        : KUDNN_JIT_Generator_With_Attributes(attributesArg) {}
    void ApplySum(const PostOps &postOps, SizeType poIdx,
        const std::vector<SizeType> &regIDs, Element::TypeT regType) override
    {
        Service::Unused(postOps, poIdx, regIDs, regType);
        // nothing to do for matmul, since SUM is supported via beta parameter
    }
};

// Kernel Code Generator with "default" scheme (optimized for big matrices, not small or skinny)
template <ARCH Arch, typename SrcType, typename WeiType, typename DstType, typename BiaType, MatMulImplKinds ImplKind>
struct MatMulKernelDefaultCodeGen {};

// Kernel Code Generator with "GEMV" scheme (optimized for Skinny)
template <ARCH Arch, typename SrcType, typename WeiType, typename DstType, typename BiaType, MatMulImplKinds ImplKind>
struct MatMulKernelSkinnyABCodeGen {};

// Kernel Code Generator with "GEMV" scheme (optimized for Skinny BA layout)
template <ARCH Arch, typename SrcType, typename WeiType, typename DstType, typename BiaType, MatMulImplKinds ImplKind>
struct MatMulKernelSkinnyBACodeGen {};

template <typename CodeGenType>
struct CodeGenParams {
    using CodeGen = CodeGenType;
    IntType m;
    IntType n;
    IntType k;
    // this can be eliminated in future
    bool extraCheckForSrcWeiSwapped;
    const Attributes attributes;
};

template <typename CodeGenType>
inline bool operator==(const CodeGenParams<CodeGenType> &lhs, const CodeGenParams<CodeGenType> &rhs) noexcept
{
    return (CodeGenType::CompareMSize(lhs.m, rhs.m) &&
        CodeGenType::CompareNSize(lhs.n, rhs.n) &&
        CodeGenType::CompareKSize(lhs.k, rhs.k) &&
        (lhs.extraCheckForSrcWeiSwapped == rhs.extraCheckForSrcWeiSwapped) &&
        (lhs.attributes == rhs.attributes));
}

template <typename MatMulKernelCodeGen>
struct MatMulImplGenerator {};
struct PackStrategy;

struct MatMulImplBase : ImplBase<GemmInfo> {
    struct UniqueParams final {
        // 5D Matmul Parameters
        IntType parallelStrideSrc;
        IntType parallelStrideWei;
        IntType parallelStrideDst;
        IntType parallelStrideBia;
        Shape broadcastedDims; // not reordered
        IntType SrcBufferSize;
        IntType WeiBufferSize;
        // // 2D reorder wrappers
        ReorderWrapper biaRW2D;
        // // 2D GEMM parameters (overlap with 5D Matmul parameters)
        IntType m; // swapped with original n in case of BA-like layout of dst
        IntType n; // swapped with original m in case of BA-like layout of dst
        IntType k;
        IntType srcOffset2D; // swapped with original weiOffset2D in case of BA-like layout of dst
        IntType weiOffset2D; // swapped with original srcOffset2D in case of BA-like layout of dst
        IntType dstOffset2D;
        IntType biaOffset2D;
        Layout srcLayout2D; // swapped with original weiLayout2D in case of BA-like layout of dst
        Layout weiLayout2D; // swapped with original srcLayout2D in case of BA-like layout of dst
        // takes into account BA-like layout of dst
        bool isResidByM; // is M block size - resid on thread
        bool isResidByN; // is N block size - resid on thread
    };

    MatMulImplBase(const GemmInfo &task, IntType nThreads);

    void Run(typename GemmInfo::RunTimeParams params) noexcept(false) override;

    // will be pure virtual and overriden inside all implementations later
    virtual void Pack(const void *originalMatrix, void *buffer, IntType threadID, IntType dstDim, IntType accumDim,
        IntType dstDimOffset, IntType accumDimOffset, IntType bufferOffset, IntType bufferStride,
        float alpha, MatrixKind mKind) const
    {
        Service::Unused(originalMatrix, buffer, threadID, dstDim, accumDim,
            dstDimOffset, accumDimOffset, bufferOffset, bufferStride, alpha, mKind);
    }

protected:

    virtual IntType GetMBlockSize() const noexcept = 0;

    virtual IntType GetNBlockSize() const noexcept = 0;

    virtual IntType GetKBlockSize() const noexcept = 0;

    void InitializeTaskExec(Layout desiredSrcLayout, Layout desiredWeiLayout,
        Layout desiredDstLayout, Layout desiredBiaLayout, Element::TypeT desiredSrcPrec,
        Element::TypeT desiredWeiPrec, Element::TypeT desiredDstPrec, Element::TypeT desiredBiaPrec);

    void InitializeThreadsCount();

    void InitializeKernelMorN(IntType size, IntType fullThreadSize, IntType blockSize,
        IntType &blockExecFull, IntType &blockExecResid1, IntType &blockExecResid2) const noexcept;
    
    virtual void InitializeKernelK(IntType k, IntType kBlockSize,
        IntType &kBlockExecFull, IntType &kBlockExecResid) const noexcept;

    void InitializeCurrentMorN(IntType id, IntType blockExecFull, IntType blockExecResid1, IntType blockExecResid2,
        IntType &currentSize) const noexcept;

    virtual void InitializeCodeGens(IntType m, IntType n, IntType k,
        SizeType offsetSrc, SizeType offsetWei, SizeType offsetDst, SizeType offsetBia,
        bool wereSrcWeiSwapped, IntType mFull, IntType nFull, const Attributes &attributes);
    
    virtual void InitializeSingleCodegen(IntType id, IntType m, IntType n, IntType k,
        IntType currentM, IntType currentN, IntType currentK,
        SizeType offsetSrc, SizeType offsetWei, SizeType offsetDst, SizeType offsetBia,
        bool wereSrcWeiSwapped, const Attributes &attributes, bool hasAlpha) = 0;

    virtual void InitializePackingCodeGens(IntType m, IntType n, IntType k, IntType mFull, IntType nFull);

    virtual void InitializeReorderCodeGens(IntType m, IntType n, IntType k, IntType mFull, IntType nFull);

    virtual void InitializeThreadsCountImpl();

    IntType GetRequiredMemSize();

    virtual IntType GetThreadBufSize(IntType m, IntType n, IntType k, MatrixKind mKind) const noexcept = 0;

    virtual void ReserveImplSpecificParams() = 0;

    virtual std::vector<std::pair<Shape, Shape>> SplitWork() = 0;

    virtual MatMulDstFlags SetBiaFlagsExec(MatMulDstFlags biaF, float beta, bool notFirstKIter,
        bool lastKIter, float alpha = 1.0f) const noexcept;

    void Initialize();

    void InitializePerThreadParams();

    virtual void InitializePackingPolicy();

    void InitializePackingPolicyFromEnv();

    static MatMulDstFlags InitializeLoadFlags(Layout dstLayout, bool hasBiaMBroadcast, bool hasBiaNBroadcast) noexcept;

    // updates taskExec if needed
    virtual void UpdateExecParams() noexcept = 0;

    void RunSequential(IntType threadID,
        const void *src, const void *wei, void *dst, const void *bia,
        std::unique_ptr<PackStrategy> &srcPackStrategy, std::unique_ptr<PackStrategy> &weiPackStrategy,
        const PostOpsDataPtrs &postOpsPtrs, float alpha, float beta) const noexcept(false);

    virtual void RunSequential2D(IntType threadID, const void *src, const void *wei,
        void *dst, const void *bia,
        std::unique_ptr<PackStrategy> &srcPackStrategy, std::unique_ptr<PackStrategy> &weiPackStrategy,
        float alpha, float beta, const PostOpsDataPtrs &shiftedByBatchPOPtrs) const noexcept(false) = 0;

    PostOpsDataPtrs ShiftPODPbyThreadID(IntType threadID,
        const PostOpsDataPtrs &poPtrs) const;

    PostOpsDataPtrs ShiftPODPbyBatch(SizeType b0, SizeType b1, SizeType b2,
        const PostOpsDataPtrs &poPtrs) const;

    void InitThreadIDs(IntType threadID, IntType &tIDbatch, IntType &tIDm, IntType &tIDn) const;

    bool isInitialized;
    GemmInfo taskExec;
    IntType nThreadsExec;

    IntType mFullBlockSizePerThread;     // size of first M Full dimension
    IntType nFullBlockSizePerThread;     // size of first N Full dimension
    IntType kFullBlockSizePerThread;     // size of first k Full dimension

    std::vector<UniqueParams> perThreadParams;
    // Multidimensional reorder wrappers
    ReorderWrapper srcRWM;
    ReorderWrapper weiRWM;
    ReorderWrapper dstRWM;
    ReorderWrapper biaRWM;
    Shape broadcastedStridesSrc;
    Shape broadcastedStridesWei;
    Shape broadcastedStridesDst;
    Shape broadcastedStridesBia;
    Layout dstLayout2D;
    MatMulDstFlags biaFlags;
    IntType gemmWorkSize;
    IntType biaNOffset;

    IntType threadsBatched;
    IntType threadsM;
    IntType threadsN;
    // once we want to move it to constructor parameters (not only env control)
    // we need to move this field to `GemmInfo`
    PackingPolicy packingPolicy;
};

struct PackStrategy {
    PackStrategy(IntType threadsBatched, IntType numGroups, IntType groupSize, MatrixKind matrixKind);

    virtual void Pack(const MatMulImplBase &matmulImpl, const void *originalMatrix, void *buffer,
        LoopCtrHelper dstDimension, LoopCtrHelper accumulationDimension,
        IntType threadGroupSize, IntType threadIDInsideGroup, IntType threadID, float alpha) const = 0;

    virtual void Barrier(IntType groupID, IntType threadIDInsideGroup, IntType groupSize) = 0;

    virtual void AllocateBuffers(IntType bufferSize, IntType groupID,
        IntType threadIDInsideGroup, IntType groupSize) = 0;

    virtual void DeallocateBuffers(IntType groupID,
        IntType threadIDInsideGroup, IntType groupSize) = 0;

    void* GetPtr(IntType groupID, IntType threadIDInsideGroup, IntType groupSize);

    virtual ~PackStrategy() = default;
protected:
    MatrixKind mKind;
    std::unique_ptr<Service::MemoryBuffer[]> bufferPtrs;
    std::unique_ptr<std::atomic<std::uint64_t>[]> fast;
    std::unique_ptr<std::atomic<std::uint64_t>[]> slow;
};

struct PrivateBufferPack : PackStrategy {
    PrivateBufferPack(IntType threadsBatched, IntType numGroups, IntType groupSize, MatrixKind matrixKind);

    void Pack(const MatMulImplBase &matmulImpl, const void *originalMatrix, void *buffer, LoopCtrHelper dstDimension,
        LoopCtrHelper accumulationDimension, IntType threadGroupSize, IntType threadIDInsideGroup,
        IntType threadID, float alpha) const override;

    void Barrier(IntType groupID, IntType threadIDInsideGroup, IntType groupSize) override;

    void AllocateBuffers(IntType bufferSize, IntType groupID,
        IntType threadIDInsideGroup, IntType groupSize) override;

    void DeallocateBuffers(IntType groupID,
        IntType threadIDInsideGroup, IntType groupSize) override;
};

struct SharedBufferPack : PackStrategy {
    SharedBufferPack(IntType threadsBatched, IntType numGroups, IntType groupSize, MatrixKind matrixKind);

    void Barrier(IntType groupID, IntType threadIDInsideGroup, IntType groupSize) override;

    void AllocateBuffers(IntType bufferSize, IntType groupID,
        IntType threadIDInsideGroup, IntType groupSize) override;

    void DeallocateBuffers(IntType groupID,
        IntType threadIDInsideGroup, IntType groupSize) override;
};

struct SharedBufferPackSplitDstDim : SharedBufferPack {
    SharedBufferPackSplitDstDim(IntType threadsBatched, IntType numGroups, IntType groupSize, MatrixKind matrixKind);

    void Pack(const MatMulImplBase &matmulImpl, const void *originalMatrix, void *buffer, LoopCtrHelper dstDimension,
        LoopCtrHelper accumulationDimension, IntType threadGroupSize, IntType threadIDInsideGroup,
        IntType threadID, float alpha) const override;
};

struct SharedBufferPackSplitAccDim : SharedBufferPack {
    SharedBufferPackSplitAccDim(IntType threadsBatched, IntType numGroups, IntType groupSize, MatrixKind matrixKind);

    void Pack(const MatMulImplBase &matmulImpl, const void *originalMatrix, void *buffer, LoopCtrHelper dstDimension,
        LoopCtrHelper accumulationDimension, IntType threadGroupSize, IntType threadIDInsideGroup,
        IntType threadID, float alpha) const override;
};

} // Detail
} // KuDNN

template <typename CodeGenType>
struct std::hash<KuDNN::Detail::CodeGenParams<CodeGenType>> {
    std::size_t operator()(const KuDNN::Detail::CodeGenParams<CodeGenType>& codeGenParams) const noexcept
    {
        return std::hash<KuDNN::IntType>{}(CodeGenType::ProcessMSize(codeGenParams.m)) ^
            (std::hash<KuDNN::IntType>{}(CodeGenType::ProcessNSize(codeGenParams.n)) << 1) ^
            (std::hash<KuDNN::IntType>{}(CodeGenType::ProcessKSize(codeGenParams.k)) << 2) ^
            (std::hash<KuDNN::IntType>{}(codeGenParams.attributes.GetPostOps().Len()));
    }
};

#endif // KUDNN_GEMM_JIT_HPP
