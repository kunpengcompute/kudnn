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

#include <omp.h>
#include <arm_sve.h>
#include <kupl.h>
#include "kudnn.hpp"

#include "internal_funcs.hpp"
#include "kudnn_jit.hpp"
#include "kudnn_wrapper_helpers.hpp"
#include "matmul_jit/kudnn_gemm_helpers.hpp"
#include "threading_impl.hpp"

namespace KuDNN {
namespace Detail {

template <typename HelperSrcType, typename HelperWeiType, typename HelperDstType, typename HelperBiaType>
struct MatmulTypeHelper {
    using SrcType = HelperSrcType;
    using WeiType = HelperWeiType;
    using DstType = HelperDstType;
    using BiaType = HelperBiaType;
    constexpr static SizeType GetSrcSize() noexcept
    {
        return sizeof(SrcType);
    }
    constexpr static Element::TypeT GetSrcType() noexcept
    {
        return Element::MatchType<SrcType>();
    }
    constexpr static SizeType GetWeiSize() noexcept
    {
        return sizeof(WeiType);
    }
    constexpr static Element::TypeT GetWeiType() noexcept
    {
        return Element::MatchType<WeiType>();
    }
    constexpr static SizeType GetDstSize() noexcept
    {
        return sizeof(DstType);
    }
    constexpr static Element::TypeT GetDstType() noexcept
    {
        return Element::MatchType<DstType>();
    }
    constexpr static SizeType GetBiaSize() noexcept
    {
        return sizeof(BiaType);
    }
    constexpr static Element::TypeT GetBiaType() noexcept
    {
        return Element::MatchType<BiaType>();
    }
};

#ifdef __cplusplus
extern "C" {
#endif
extern kupl_egroup_h *g_egroups;

#ifdef __cplusplus
}
#endif

// only decl to be used inside template context
// needed to avoid code duplication for templated implementations,
// e.g. s8s8s32s32 and s8u8s32s32
struct MatmulDummyType {};

template <ARCH Arch, typename SrcType, typename WeiType, typename DstType, typename BiaType>
struct MatMulKernelCodeGenBase : KUDNN_JIT_Generator, MatmulTypeHelper<SrcType, WeiType, DstType, BiaType> {};

// implementation of MatMulKernelDefaultCodeGen
template <ARCH Arch, typename SrcType, typename WeiType, typename DstType, typename BiaType, typename Dummy>
struct MatMulKernelDefaultCodeGenImpl {};

// Kernel Code Generator with "default" scheme (optimized for big matrices, not small or skinny)
template <ARCH Arch, typename SrcType, typename WeiType, typename DstType, typename BiaType>
struct MatMulKernelDefaultCodeGen {};

template <typename MatMulKernelCodeGen>
struct MatMulImplGenerator {};

struct MatMulImplBase : ImplBase<GemmInfo> {
    struct UniqueParams final {
        // 5D Matmul Parameters
        IntType parallelStrideSrc;
        IntType parallelStrideWei;
        IntType parallelStrideDst;
        IntType parallelStrideBia;
        Shape broadcastedDims;
        IntType bufferSize;
        // // 2D reorder wrappers
        ReorderWrapper biaRW2D;
        // // 2D GEMM parameters (overlap with 5D Matmul parameters)
        IntType m;
        IntType n;
        IntType k;
        IntType srcOffset2D;
        IntType weiOffset2D;
        IntType dstOffset2D;
        IntType biaOffset2D;
        Layout srcLayout2D;
        Layout weiLayout2D;
    };

    MatMulImplBase(const GemmInfo &task, IntType nThreads) : ImplBase<GemmInfo>(task, nThreads),
        isInitialized(false), taskExec(task), perThreadParams(0)
    {
        VERBOSE_PRINT;
    }

    void Run(typename GemmInfo::RunTimeParams params) noexcept(false) override
    {
        VERBOSE_PRINT;
        if (!isInitialized || (perThreadParams.size() == 0)) {
            throw Service::LogicError{"Internal Error"};
        }
        const void *srcExec = srcRWM.Get(params.src);
        const void *weiExec = weiRWM.Get(params.wei);
        void *dstExec = dstRWM.Get(params.dst);
        const void *biaExec = biaRWM.Get(params.bia);

        if ((implNThreads > 1) && (gemmWorkSize > 1)) {
            if (g_egroups == NULL) {
                auto &&activeLevel = omp_get_active_level();
#pragma omp parallel for num_threads(gemmWorkSize) schedule(static) \
                    if ((activeLevel < 1) || \
                    (Threading::GetThreadingControlStatus() == Threading::ThreadingControl::KUDNN_MANUAL))
                for (IntType i = 0; i < gemmWorkSize; ++i) {
                    RunSequential(i, srcExec, weiExec, dstExec, biaExec, params.alpha, params.beta);
                }
            } else {
                int num = gemmWorkSize;
                kupl_nd_range_t range;
                KUPL_1D_RANGE_INIT(range, 0, num);

                kupl_parallel_for_desc_t desc = {.field_mask = KUPL_PARALLEL_FOR_DESC_FIELD_DEFAULT,
                    .range = &range,
                    .egroup = nullptr,
                    .concurrency = num,
                    .policy = KUPL_LOOP_POLICY_STATIC};
                
                int ret = kupl::parallel_for(&desc, [&](const kupl_nd_range_t *nd_range, int tid, int tnum) {
                    Service::Unused(nd_range);
                    Service::Unused(tnum);
                    RunSequential(tid, srcExec, weiExec, dstExec, biaExec, params.alpha, params.beta);
                });
                Service::Unused(ret);
            }
        } else {
            RunSequential(0, srcExec, weiExec, dstExec, biaExec, params.alpha, params.beta);
        }
        if (dstExec != params.dst) {
            ReorderLayer r(taskExec.dstInfo, implTask.dstInfo);
            r.Run(dstExec, params.dst);
        }
    }

protected:

    void InitializeTaskExec(Layout desiredSrcLayout, Layout desiredWeiLayout,
        Layout desiredDstLayout, Layout desiredBiaLayout, Element::TypeT desiredSrcPrec,
        Element::TypeT desiredWeiPrec, Element::TypeT desiredDstPrec, Element::TypeT desiredBiaPrec)
    {
        VERBOSE_PRINT;
        if ((desiredSrcLayout == implTask.srcInfo.GetLayout()) &&
            (implTask.srcInfo.GetType() == desiredSrcPrec)) {
            taskExec.srcInfo = implTask.srcInfo;
        } else {
            taskExec.srcInfo = {implTask.srcInfo.GetDims(), desiredSrcPrec, desiredSrcLayout};
        }
        if ((desiredWeiLayout == implTask.weiInfo.GetLayout()) &&
            (implTask.weiInfo.GetType() == desiredWeiPrec)) {
            taskExec.weiInfo = implTask.weiInfo;
        } else {
            taskExec.weiInfo = {implTask.weiInfo.GetDims(), desiredWeiPrec, desiredWeiLayout};
        }
        if ((desiredDstLayout == implTask.dstInfo.GetLayout()) &&
            (implTask.dstInfo.GetType() == desiredDstPrec)) {
            taskExec.dstInfo = implTask.dstInfo;
        } else {
            taskExec.dstInfo = {implTask.dstInfo.GetDims(), desiredDstPrec, desiredDstLayout};
        }
        if ((desiredBiaLayout == implTask.biaInfo.GetLayout()) &&
            (implTask.biaInfo.GetType() == desiredBiaPrec)) {
            taskExec.biaInfo = implTask.biaInfo;
        } else {
            taskExec.biaInfo = {implTask.biaInfo.GetDims(), desiredBiaPrec, desiredBiaLayout};
        }
    }

    virtual void InitializeCodeGens(IntType m, IntType n, IntType k, IntType vl,
        SizeType offsetSrc, SizeType offsetWei, SizeType offsetDst, SizeType offsetBia,
        bool wereSrcWeiSwapped) = 0;

    virtual IntType GetThreadBufSize(IntType m, IntType n, IntType k, IntType vl) noexcept = 0;

    virtual void ReserveImplSpecificParams() = 0;

    virtual std::vector<std::pair<Shape, Shape>> SplitWork() = 0;

    void Initialize()
    {
        VERBOSE_PRINT;
        srcRWM.Initialize(implTask.srcInfo, taskExec.srcInfo);
        weiRWM.Initialize(implTask.weiInfo, taskExec.weiInfo);
        dstRWM.Initialize(implTask.dstInfo, taskExec.dstInfo);
        biaRWM.Initialize(implTask.biaInfo, taskExec.biaInfo);
        broadcastedStridesSrc = Service::BroadcastTo5D(taskExec.srcInfo.GetStrides(), false);
        broadcastedStridesWei = Service::BroadcastTo5D(taskExec.weiInfo.GetStrides(), false);
        broadcastedStridesDst = Service::BroadcastTo5D(taskExec.dstInfo.GetStrides(), false);
        broadcastedStridesBia = Service::BroadcastTo5D(taskExec.biaInfo.GetStrides(), false);

        if (IsLayoutAlikeBA(taskExec.dstInfo.GetLayout())) {
            dstLayout2D = Layout::BA;
        } else {
            dstLayout2D = Layout::AB;
        }
        biaFlags = DstLoadFlags::FILL_C_WITH_ZEROZ;
        if (taskExec.biaInfo.GetTotalTensorSize() != 0) {
            IntType m = taskExec.dstInfo.GetDims()[taskExec.dstInfo.GetNumDims() - 2];
            IntType n = taskExec.dstInfo.GetDims()[taskExec.dstInfo.GetNumDims() - 1];
            IntType biaM = std::min<IntType>(m,
                taskExec.biaInfo.GetDims()[taskExec.biaInfo.GetNumDims() - 2]);
            IntType biaN = std::min<IntType>(n,
                taskExec.biaInfo.GetDims()[taskExec.biaInfo.GetNumDims() - 1]);
            biaFlags = InitializeLoadFlags(dstLayout2D, biaM < m, biaN < n);
        }
        biaNOffset = (static_cast<bool>(biaFlags & DstLoadFlags::LOAD_BIAS_N) ||
            static_cast<bool>(biaFlags & DstLoadFlags::LOAD_BIAS_MN)) ? 1 : 0;
        InitializePerThreadParams();
        isInitialized = true;
    }

    void InitializePerThreadParams()
    {
        // 2D GEMM parameters (overlap with 5D Matmul parameters)
        SizeType offsetSrc = broadcastedStridesSrc[IDX_3];
        Layout srcLayout2D = Layout::AB;
        Layout srcTLayout2D = Layout::BA;
        if (offsetSrc < broadcastedStridesSrc[IDX_4]) {
            offsetSrc = broadcastedStridesSrc[IDX_4];
            srcLayout2D = Layout::BA;
            srcTLayout2D = Layout::AB;
        }
        SizeType offsetWei = broadcastedStridesWei[IDX_3];
        Layout weiLayout2D = Layout::AB;
        Layout weiTLayout2D = Layout::BA;
        if (offsetWei < broadcastedStridesWei[IDX_4]) {
            offsetWei = broadcastedStridesWei[IDX_4];
            weiLayout2D = Layout::BA;
            weiTLayout2D = Layout::AB;
        }
        SizeType offsetDst = broadcastedStridesDst[IDX_3];
        if (dstLayout2D == Layout::BA) {
            offsetDst = broadcastedStridesDst[IDX_4];
            srcLayout2D = weiTLayout2D;
            weiLayout2D = srcTLayout2D;
            std::swap(offsetSrc, offsetWei);
        }

        Shape srcStrides = {offsetSrc, 1};
        if (srcLayout2D == Layout::BA) {
            srcStrides = {1, offsetSrc};
        }
        Shape weiStrides = {offsetWei, 1};
        if (weiLayout2D == Layout::BA) {
            weiStrides = {1, offsetWei};
        }

        SizeType offsetBia = broadcastedStridesBia[IDX_3];
        Shape biaStrides = {offsetBia, 1};
        Layout biaLayout2D = Layout::AB;
        if (offsetBia < broadcastedStridesBia[IDX_4]) {
            offsetBia = broadcastedStridesBia[IDX_4];
            biaStrides = {1, offsetBia};
            biaLayout2D = Layout::BA;
        }
        if ((broadcastedStridesBia[IDX_3] == 0) || (broadcastedStridesBia[IDX_4] == 0)) {
            biaLayout2D = dstLayout2D;
            if (biaLayout2D == Layout::AB) {
                offsetBia = broadcastedStridesBia[IDX_3];
                biaStrides = {offsetBia, 1};
            } else {
                offsetBia = broadcastedStridesBia[IDX_4];
                biaStrides = {1, offsetBia};
            }
        }

        IntType k = taskExec.srcInfo.GetDims()[taskExec.srcInfo.GetNumDims() - 1];

        std::vector<std::pair<Shape, Shape>> gemmWork = SplitWork();
        gemmWorkSize = gemmWork.size();
        perThreadParams.reserve(gemmWorkSize);
        ReserveImplSpecificParams();

        for (SizeType i = 0; i < gemmWork.size(); ++i) {
            IntType vl = svcntw();
            IntType m = gemmWork[i].first[taskExec.dstInfo.GetNumDims() - NUM_2];
            IntType n = gemmWork[i].first[taskExec.dstInfo.GetNumDims() - NUM_1];
            IntType biaM = m;
            IntType biaN = n;
            if (taskExec.biaInfo.GetTotalTensorSize() != 0) {
                biaM = std::min<IntType>(m,
                    taskExec.biaInfo.GetDims()[taskExec.biaInfo.GetNumDims() - NUM_2]);
                biaN = std::min<IntType>(n,
                    taskExec.biaInfo.GetDims()[taskExec.biaInfo.GetNumDims() - NUM_1]);
            }
            if (dstLayout2D == Layout::BA) {
                std::swap(m, n);
            }

            ReorderWrapper biaRW2D;
            if (taskExec.biaInfo.GetTotalTensorSize() != 0) {
                TensorInfo biaInfo2D{Shape{static_cast<SizeType>(biaM),
                    static_cast<SizeType>(biaN)}, taskExec.biaInfo.GetType(), biaLayout2D, biaStrides};
                TensorInfo desiredBiaInfo2D{Shape{static_cast<SizeType>(biaM),
                    static_cast<SizeType>(biaN)}, taskExec.biaInfo.GetType(), dstLayout2D};
                biaRW2D.Initialize(biaInfo2D, desiredBiaInfo2D);
                if (biaRW2D.WillReorder(biaInfo2D, desiredBiaInfo2D)) {
                    offsetBia = n;
                }
            }
            // Batch Stride --> Memory Stride
            SizeType strideSrc = 0;
            SizeType strideWei = 0;
            SizeType strideDst = 0;
            SizeType strideBia = 0;
            SizeType srcSize = taskExec.srcInfo.GetType().GetSize();
            SizeType weiSize = taskExec.weiInfo.GetType().GetSize();
            SizeType dstSize = taskExec.dstInfo.GetType().GetSize();
            SizeType biaSize = taskExec.biaInfo.GetType().GetSize();
            for (SizeType j = 0; j < gemmWork[i].second.GetNumDims() - NUM_2; ++j) {
                strideSrc += gemmWork[i].second[j] * taskExec.srcInfo.GetStrides()[j] * srcSize;
                strideWei += gemmWork[i].second[j] * taskExec.weiInfo.GetStrides()[j] * weiSize;
                strideDst += gemmWork[i].second[j] * taskExec.dstInfo.GetStrides()[j] * dstSize;
            }
            strideSrc += gemmWork[i].second[gemmWork[i].second.GetNumDims() - NUM_2] *
                taskExec.srcInfo.GetStrides()[gemmWork[i].second.GetNumDims() - NUM_2] * srcSize;
            strideWei += gemmWork[i].second[gemmWork[i].second.GetNumDims() - NUM_1] *
                taskExec.weiInfo.GetStrides()[gemmWork[i].second.GetNumDims() - NUM_1] * weiSize;
            strideDst += gemmWork[i].second[gemmWork[i].second.GetNumDims() - NUM_2] *
                taskExec.dstInfo.GetStrides()[gemmWork[i].second.GetNumDims() - NUM_2] * dstSize;
            strideDst += gemmWork[i].second[gemmWork[i].second.GetNumDims() - NUM_1] *
                taskExec.dstInfo.GetStrides()[gemmWork[i].second.GetNumDims() - NUM_1] * dstSize;
            if (taskExec.biaInfo.GetTotalTensorSize() != 0) {
                for (SizeType j = 0; j < gemmWork[i].second.GetNumDims(); ++j) {
                    strideBia += gemmWork[i].second[j] * taskExec.biaInfo.GetStrides()[j] * biaSize;
                }
            }

            perThreadParams.push_back({static_cast<IntType>(strideSrc), static_cast<IntType>(strideWei),
                static_cast<IntType>(strideDst), static_cast<IntType>(strideBia),
                Service::BroadcastTo5D(gemmWork[i].first),
                GetThreadBufSize(m, n, k, vl),
                std::move(biaRW2D),
                m, n, k, static_cast<IntType>(offsetSrc), static_cast<IntType>(offsetWei),
                static_cast<IntType>(offsetDst), static_cast<IntType>(offsetBia),
                srcLayout2D, weiLayout2D});

            InitializeCodeGens(m, n, k, vl, offsetSrc, offsetWei, offsetDst, offsetBia,
                IsLayoutAlikeBA(taskExec.dstInfo.GetLayout()));
        }
    }

    static DstLoadFlags InitializeLoadFlags(Layout dstLayout, bool hasBiaMBroadcast, bool hasBiaNBroadcast) noexcept
    {
        if (hasBiaMBroadcast && hasBiaNBroadcast) {
            return DstLoadFlags::LOAD_BIAS_1;
        } else if ((hasBiaMBroadcast && IsLayoutAlikeAB(dstLayout)) ||
            (hasBiaNBroadcast && !IsLayoutAlikeAB(dstLayout))) {
            return DstLoadFlags::LOAD_BIAS_N;
        } else if ((hasBiaNBroadcast && IsLayoutAlikeAB(dstLayout)) ||
            (hasBiaMBroadcast && !IsLayoutAlikeAB(dstLayout))) {
            return DstLoadFlags::LOAD_BIAS_M;
        } else {
            return DstLoadFlags::LOAD_BIAS_MN;
        }
    }

    // updates taskExec if needed
    virtual void UpdateExecParams() noexcept = 0;

    void RunSequential(IntType threadId,
        const void *src, const void *wei, void *dst, const void *bia, float alpha, float beta) const noexcept(false)
    {
        VERBOSE_PRINT;
        float* buffer = (float*) Service::AlignedAlloc(perThreadParams[threadId].bufferSize, 128);
        if (!buffer) {
            throw Service::BadAlloc{};
        }

        // reorders will be single-threaded, since there's parallelization over it
        auto oldLocalNThreads = Threading::SetNumThreadsLocalInternal(1);
        // Loop over batched dimensions
        for (SizeType i0 = 0; i0 < perThreadParams[threadId].broadcastedDims[IDX_0]; ++i0) {
            for (SizeType i1 = 0; i1 < perThreadParams[threadId].broadcastedDims[IDX_1]; ++i1) {
                for (SizeType i2 = 0; i2 < perThreadParams[threadId].broadcastedDims[IDX_2]; ++i2) {
                    const void *srcPtr = static_cast<const void *>(
                        static_cast<const std::byte *>(src) + perThreadParams[threadId].parallelStrideSrc +
                        (i0 * broadcastedStridesSrc[IDX_0] + i1 * broadcastedStridesSrc[IDX_1] +
                        i2 * broadcastedStridesSrc[IDX_2]) * taskExec.srcInfo.GetType().GetSize());
                    const void *weiPtr = static_cast<const void *>(
                        static_cast<const std::byte *>(wei) + perThreadParams[threadId].parallelStrideWei +
                        (i0 * broadcastedStridesWei[IDX_0] + i1 * broadcastedStridesWei[IDX_1] +
                        i2 * broadcastedStridesWei[IDX_2]) * taskExec.weiInfo.GetType().GetSize());
                    void *dstPtr = static_cast<void *>(
                        static_cast<std::byte *>(dst) + perThreadParams[threadId].parallelStrideDst +
                        (i0 * broadcastedStridesDst[IDX_0] + i1 * broadcastedStridesDst[IDX_1] +
                        i2 * broadcastedStridesDst[IDX_2]) * taskExec.dstInfo.GetType().GetSize());
                    const void *biaPtr = bia;
                    if (biaPtr) {
                        biaPtr = static_cast<const void *>(
                            static_cast<const std::byte *>(biaPtr) + perThreadParams[threadId].parallelStrideBia +
                            (i0 * broadcastedStridesBia[IDX_0] + i1 * broadcastedStridesBia[IDX_1] +
                            i2 * broadcastedStridesBia[IDX_2]) * taskExec.biaInfo.GetType().GetSize());
                    }
                    const void *biaExec = perThreadParams[threadId].biaRW2D.Get(biaPtr);
                    if (dstLayout2D == Layout::AB) {
                        RunSequential2D(threadId, srcPtr, weiPtr, dstPtr, biaExec, buffer, alpha, beta);
                    } else {
                        RunSequential2D(threadId, weiPtr, srcPtr, dstPtr, biaExec, buffer, alpha, beta);
                    }
                }
            }
        }
        Threading::SetNumThreadsLocalInternal(oldLocalNThreads);
        Service::Deallocate(buffer);
    }

    virtual void RunSequential2D(IntType threadId, const void *src, const void *wei,
        void *dst, const void *bia, void *buffer, float alpha, float beta) const noexcept(false) = 0;

    bool isInitialized;
    GemmInfo taskExec;
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
    DstLoadFlags biaFlags;
    IntType gemmWorkSize;
    IntType biaNOffset;
};

} // Detail
} // KuDNN

#endif // KUDNN_GEMM_JIT_HPP
