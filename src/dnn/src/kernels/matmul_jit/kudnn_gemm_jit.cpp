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

#include <algorithm>
#include <cassert>

#include <arm_sve.h>

#include <omp.h>

#include "kudnn.hpp"

#include "internal_funcs.hpp"
#include "kudnn_jit.hpp"
#include "kudnn_wrapper_helpers.hpp"
#include "kernels/matmul_jit/kudnn_gemm_helpers.hpp"
#include "threading_impl.hpp"
#include "kudnn_gemm_jit.hpp"

namespace KuDNN {
namespace Detail {

MatMulImplBase::MatMulImplBase(const GemmInfo &task, IntType nThreads) : ImplBase<GemmInfo>(task, nThreads),
    isInitialized(false), taskExec(task), perThreadParams(0)
{
    VERBOSE_PRINT;
}

#if defined(KUDNN_CPU_THREADING_RUNTIME) && (KUDNN_CPU_THREADING_RUNTIME == KUDNN_RUNTIME_OMP)
struct NestedOMPHelper {
    NestedOMPHelper() : oldNestedState(omp_get_max_active_levels())
    {
        omp_set_max_active_levels(std::numeric_limits<int>::max()); // this value is used by GCC OMP by default
    }
    ~NestedOMPHelper()
    {
        omp_set_max_active_levels(oldNestedState);
    }
    int oldNestedState;
};
#endif

void MatMulImplBase::Run(typename GemmInfo::RunTimeParams params) noexcept(false)
{
    VERBOSE_PRINT;
    if (!isInitialized || (perThreadParams.size() == 0)) {
        throw Service::LogicError{"Internal Error"};
    }
    const void *srcExec = srcRWM.Get(params.src);
    const void *weiExec = weiRWM.Get(params.wei);
    void *dstExec = dstRWM.Get(params.dst, (params.beta == 0.0f));
    const void *biaExec = biaRWM.Get(params.bia);
    PostOpsDataPtrs postOpsPtrs = params.postOpsPtrs;

    std::unique_ptr<PackStrategy> srcPackStrategy;
    if (packingPolicy & PackingPolicy::SYNC_DST_DIM_SRC) {
        srcPackStrategy.reset(new SharedBufferPackSplitDstDim{threadsBatched, threadsM, threadsN, MatrixKind::SRC});
    } else if (packingPolicy & PackingPolicy::SYNC_ACC_DIM_SRC) {
        srcPackStrategy.reset(new SharedBufferPackSplitAccDim{threadsBatched, threadsM, threadsN, MatrixKind::SRC});
    } else {
        srcPackStrategy.reset(new PrivateBufferPack{threadsBatched, threadsM, threadsN, MatrixKind::SRC});
    }
    std::unique_ptr<PackStrategy> weiPackStrategy;
    if (packingPolicy & PackingPolicy::SYNC_DST_DIM_WEI) {
        weiPackStrategy.reset(new SharedBufferPackSplitDstDim{threadsBatched, threadsN, threadsM, MatrixKind::WEI});
    } else if (packingPolicy & PackingPolicy::SYNC_ACC_DIM_WEI) {
        weiPackStrategy.reset(new SharedBufferPackSplitAccDim{threadsBatched, threadsN, threadsM, MatrixKind::WEI});
    } else {
        weiPackStrategy.reset(new PrivateBufferPack{threadsBatched, threadsN, threadsM, MatrixKind::WEI});
    }

    if ((nThreadsExec > 1) && (gemmWorkSize > 1)) {
#if defined(KUDNN_CPU_THREADING_RUNTIME) && (KUDNN_CPU_THREADING_RUNTIME == KUDNN_RUNTIME_OMP)
        auto &&activeLevel = omp_get_active_level();
        NestedOMPHelper helper; // RAI object responsible for setting OMP Nested flags
        Service::Unused(helper);
#pragma omp parallel for num_threads(gemmWorkSize) \
            if ((activeLevel < 1) || \
            (Threading::GetThreadingControlStatus() == Threading::ThreadingControl::KUDNN_MANUAL))
        for (IntType i = 0; i < gemmWorkSize; ++i) {
            RunSequential(i, srcExec, weiExec, dstExec, biaExec, srcPackStrategy, weiPackStrategy,
                postOpsPtrs, params.alpha, params.beta);
        }
#elif defined(KUDNN_CPU_THREADING_RUNTIME) && (KUDNN_CPU_THREADING_RUNTIME == KUDNN_RUNTIME_THREADPOOL)
        KuDNN::Threading::Parallel(gemmWorkSize, [&](IntType i) {
            RunSequential(i, srcExec, weiExec, dstExec, biaExec, srcPackStrategy, weiPackStrategy,
                postOpsPtrs, params.alpha, params.beta);
        });
#endif
    } else {
        RunSequential(0, srcExec, weiExec, dstExec, biaExec, srcPackStrategy, weiPackStrategy,
            postOpsPtrs, params.alpha, params.beta);
    }
    if (dstExec != params.dst) {
        ReorderLayer r(taskExec.dstInfo, implTask.dstInfo);
        r.Run(dstExec, params.dst);
    }
}
void MatMulImplBase::InitializeTaskExec(Layout desiredSrcLayout, Layout desiredWeiLayout,
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

void MatMulImplBase::InitializeThreadsCount()
{
    if (std::getenv("KUDNN_FIXED_NUM_THREADS") != nullptr) {
        nThreadsExec = implNThreads;
    } else {
        InitializeThreadsCountImpl();
    }
}

void MatMulImplBase::InitializeKernelMorN(IntType size, IntType fullThreadSize, IntType blockSize,
    IntType &blockExecFull, IntType &blockExecResid1, IntType &blockExecResid2) const noexcept
{
    IntType residThreadBlock = size % fullThreadSize;
    blockExecFull = blockSize;
    blockExecResid1 = fullThreadSize % blockSize;
    blockExecResid2 = residThreadBlock % blockSize;
}

void MatMulImplBase::InitializeKernelK(IntType k, IntType kBlockSize,
    IntType &kBlockExecFull, IntType &kBlockExecResid) const noexcept
{
    kBlockExecFull = kBlockSize;
    kBlockExecResid = k % kBlockExecFull;
}

void MatMulImplBase::InitializeCurrentMorN(IntType id,
    IntType blockExecFull, IntType blockExecResid1, IntType blockExecResid2,
    IntType &currentSize) const noexcept
{
    currentSize =  blockExecFull;
    if ((id % NUM_3) == NUM_1) {
        currentSize = blockExecResid1;
    } else if ((id % NUM_3) == NUM_2)  {
        currentSize = blockExecResid2;
    }
}

void MatMulImplBase::InitializeCodeGens(
    IntType m, IntType n, IntType k,
    SizeType offsetSrc, SizeType offsetWei, SizeType offsetDst, SizeType offsetBia,
    bool wereSrcWeiSwapped, IntType mFull, IntType nFull, const Attributes &attributes)
{
    IntType mBlockExecFull;
    IntType mBlockExecResid1;
    IntType mBlockExecResid2;
    InitializeKernelMorN(m, mFull, GetMBlockSize(), mBlockExecFull, mBlockExecResid1, mBlockExecResid2);

    IntType nBlockExecFull;
    IntType nBlockExecResid1;
    IntType nBlockExecResid2;
    InitializeKernelMorN(n, nFull, GetNBlockSize(), nBlockExecFull, nBlockExecResid1, nBlockExecResid2);

    IntType kBlockExecFull;
    IntType kBlockExecResid;
    InitializeKernelK(k, GetKBlockSize(), kBlockExecFull, kBlockExecResid);

    for (IntType i = 0; i < (NUM_3 * NUM_3 * NUM_2); ++i) {
        IntType currentM;
        InitializeCurrentMorN(i, mBlockExecFull, mBlockExecResid1, mBlockExecResid2, currentM);
        IntType currentN;
        InitializeCurrentMorN(i / NUM_3, nBlockExecFull, nBlockExecResid1, nBlockExecResid2, currentN);
        IntType currentK = kBlockExecFull;
        if ((i / NUM_3 / NUM_3) % NUM_2 == NUM_1) {
            currentK = kBlockExecResid;
        }
        InitializeSingleCodegen(i, m, n, k, currentM, currentN, currentK, offsetSrc, offsetWei, offsetDst, offsetBia,
            wereSrcWeiSwapped, attributes, false);
    }
}

void MatMulImplBase::InitializePackingCodeGens(IntType m, IntType n, IntType k,
    IntType mFull, IntType nFull)
{
    Service::Unused(m, n, k, mFull, nFull);
}

void MatMulImplBase::InitializeReorderCodeGens(IntType m, IntType n, IntType k,
    IntType mFull, IntType nFull)
{
    Service::Unused(m, n, k, mFull, nFull);
}

void MatMulImplBase::InitializeThreadsCountImpl()
{
    nThreadsExec = implNThreads;
}

IntType MatMulImplBase::GetRequiredMemSize()
{
    IntType batchSize = NUM_1;
    for (SizeType i = 0; i < taskExec.srcInfo.GetNumDims() - NUM_2; ++i) {
        batchSize *= taskExec.dstInfo.GetDims()[i];
    }
    IntType mSize = taskExec.srcInfo.GetDims()[taskExec.srcInfo.GetNumDims() - NUM_2];
    IntType nSize = taskExec.dstInfo.GetDims()[taskExec.dstInfo.GetNumDims() - NUM_1];
    IntType kSize = taskExec.srcInfo.GetDims()[taskExec.srcInfo.GetNumDims() - NUM_1];

    return  batchSize * (taskExec.srcInfo.GetType().GetSize() * mSize * kSize) +
                        (taskExec.dstInfo.GetType().GetSize() * kSize * nSize) +
                        (taskExec.srcInfo.GetType().GetSize() * mSize * nSize);
}

MatMulDstFlags MatMulImplBase::SetBiaFlagsExec(MatMulDstFlags biaF, float beta, bool notFirstKIter,
    bool lastKIter, float alpha) const noexcept
{
    Service::Unused(alpha);
    MatMulDstFlags ret;
    if (notFirstKIter) {
        ret = MatMulDstFlags::LOAD_C;
    } else { // first K iter, need to process bias and beta
        if (biaF == MatMulDstFlags::FILL_C_WITH_ZEROZ) { // No Bias
            if (beta == 0) {
                ret = MatMulDstFlags::FILL_C_WITH_ZEROZ;
            } else if (beta == 1) {
                ret = MatMulDstFlags::LOAD_C;
            } else {
                ret = MatMulDstFlags::LOAD_C | MatMulDstFlags::MULL_C;
            }
        } else { // bias
            if (beta == 0) {
                ret = biaF;
            } else if (beta == 1) {
                ret = MatMulDstFlags::LOAD_C | (biaF << NUM_4);
            } else {
                ret = MatMulDstFlags::LOAD_C | MatMulDstFlags::MULL_C | (biaF << NUM_4);
            }
        }
        ret = ret | MatMulDstFlags::FIRST_K_ITER;
    }
    if (lastKIter) {
        ret = ret | MatMulDstFlags::APPLY_POST_OPS;
    }
    return ret;
}

void MatMulImplBase::InitializePackingPolicy()
{
    packingPolicy = PackingPolicy::NO_SYNC;
}

void MatMulImplBase::InitializePackingPolicyFromEnv()
{
    auto packPolicy = std::getenv("KUDNN_PACKING_POLICY");
    if (packPolicy != nullptr) {
        std::uint32_t pPolicy = std::stoi(packPolicy);
        // Current pPolicy numbers are invalid due to their bit representations
        if (pPolicy > static_cast<std::uint32_t>(PackingPolicy::SYNC_ACC_DIM_SRC_WEI) ||
            ((pPolicy == NUM_5) || (pPolicy == NUM_6) || (pPolicy == NUM_7) ||
            (pPolicy == NUM_9) || (pPolicy == NUM_10) || (pPolicy == NUM_11))) {
            packingPolicy = PackingPolicy::NO_SYNC;
        } else {
            packingPolicy = static_cast<PackingPolicy>(pPolicy);
        }
    }
}

void MatMulImplBase::Initialize()
{
    if (srcRWM.WillReorder(implTask.srcInfo, taskExec.srcInfo)) {
        srcRWM.Initialize(implTask.srcInfo, taskExec.srcInfo);
    }
    if (weiRWM.WillReorder(implTask.weiInfo, taskExec.weiInfo)) {
        weiRWM.Initialize(implTask.weiInfo, taskExec.weiInfo);
    }
    if (dstRWM.WillReorder(implTask.dstInfo, taskExec.dstInfo)) {
        dstRWM.Initialize(implTask.dstInfo, taskExec.dstInfo);
    }
    if (biaRWM.WillReorder(implTask.biaInfo, taskExec.biaInfo)) {
        biaRWM.Initialize(implTask.biaInfo, taskExec.biaInfo);
    }

    broadcastedStridesSrc = Service::BroadcastTo5D(taskExec.srcInfo.GetStrides(), false);
    broadcastedStridesWei = Service::BroadcastTo5D(taskExec.weiInfo.GetStrides(), false);
    broadcastedStridesDst = Service::BroadcastTo5D(taskExec.dstInfo.GetStrides(), false);
    broadcastedStridesBia = Service::BroadcastTo5D(taskExec.biaInfo.GetStrides(), false);

    if (IsLayoutAlikeBA(taskExec.dstInfo.GetLayout())) {
        dstLayout2D = Layout::BA;
    } else {
        dstLayout2D = Layout::AB;
    }
    // placing this function should be done carefully
    InitializePackingPolicy(); // for some impls it uses broadcastedStridesSrc, broadcastedStridesWei and dstLayout2D
    biaFlags = MatMulDstFlags::FILL_C_WITH_ZEROZ;
    if (taskExec.biaInfo.GetTotalTensorSize() != 0) {
        IntType m = taskExec.dstInfo.GetDims()[taskExec.dstInfo.GetNumDims() - 2];
        IntType n = taskExec.dstInfo.GetDims()[taskExec.dstInfo.GetNumDims() - 1];
        IntType biaM = std::min<IntType>(m,
            taskExec.biaInfo.GetDims()[taskExec.biaInfo.GetNumDims() - 2]);
        IntType biaN = std::min<IntType>(n,
            taskExec.biaInfo.GetDims()[taskExec.biaInfo.GetNumDims() - 1]);
        biaFlags = InitializeLoadFlags(dstLayout2D, biaM < m, biaN < n);
    }
    biaNOffset = (static_cast<bool>(biaFlags & MatMulDstFlags::LOAD_BIAS_N) ||
        static_cast<bool>(biaFlags & MatMulDstFlags::LOAD_BIAS_MN)) ? 1 : 0;

    InitializeThreadsCount();
    InitializePerThreadParams();
    isInitialized = true;
}

void MatMulImplBase::InitializePerThreadParams()
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
    offsetSrc = std::max(SizeType{1}, offsetSrc);
    offsetWei = std::max(SizeType{1}, offsetWei);

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

    // gemmWork has non reordered data partition to be aligned with tensorInfo and compute strides
    std::vector<std::pair<Shape, Shape>> gemmWork = SplitWork();
    gemmWorkSize = gemmWork.size();
    perThreadParams.reserve(gemmWorkSize);
    ReserveImplSpecificParams();

    IntType mFull = taskExec.dstInfo.GetDims()[taskExec.dstInfo.GetNumDims() - 2];
    IntType nFull = taskExec.dstInfo.GetDims()[taskExec.dstInfo.GetNumDims() - 1];
    IntType kFull = taskExec.srcInfo.GetDims()[taskExec.srcInfo.GetNumDims() - 1];
    mFullBlockSizePerThread = gemmWork[0].first[taskExec.dstInfo.GetNumDims() - NUM_2];
    nFullBlockSizePerThread = gemmWork[0].first[taskExec.dstInfo.GetNumDims() - NUM_1];
    kFullBlockSizePerThread = taskExec.srcInfo.GetDims()[taskExec.srcInfo.GetNumDims() - 1];
    threadsM = Math::CeilDiv(mFull, mFullBlockSizePerThread);
    threadsN = Math::CeilDiv(nFull, nFullBlockSizePerThread);
    threadsBatched = gemmWorkSize / (threadsM * threadsN);
    // the kernel will operate on swapped m,n sizes
    if (dstLayout2D == Layout::BA) {
        std::swap(mFull, nFull);
        std::swap(mFullBlockSizePerThread, nFullBlockSizePerThread);
        std::swap(threadsM, threadsN);
    }

    for (SizeType i = 0; i < gemmWork.size(); ++i) {
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
                offsetBia = n; // going to check setting value
            }
        }
        // Batch Stride --> Memory Stride
        SizeType strideSrc = 0;
        SizeType strideWei = 0;
        SizeType strideDst = 0;
        SizeType strideBia = 0;
        {
            SizeType srcSize = taskExec.srcInfo.GetType().GetSize();
            SizeType weiSize = taskExec.weiInfo.GetType().GetSize();
            SizeType dstSize = taskExec.dstInfo.GetType().GetSize();
            SizeType biaSize = taskExec.biaInfo.GetType().GetSize();
            // gemmWork has non-rerodered dims and strides, thus can be directly multiplied by tensorInfo
            // with corresponding index
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
        }
        // perThreadParams expect swapped m and n
        perThreadParams.push_back({static_cast<IntType>(strideSrc), static_cast<IntType>(strideWei),
            static_cast<IntType>(strideDst), static_cast<IntType>(strideBia),
            Service::BroadcastTo5D(gemmWork[i].first),
            GetThreadBufSize(m, n, k, MatrixKind::SRC),
            GetThreadBufSize(m, n, k, MatrixKind::WEI),
            std::move(biaRW2D),
            m, n, k, static_cast<IntType>(offsetSrc), static_cast<IntType>(offsetWei),
            static_cast<IntType>(offsetDst), static_cast<IntType>(offsetBia),
            srcLayout2D, weiLayout2D,
            m != mFullBlockSizePerThread, n != nFullBlockSizePerThread});
    }
    InitializeReorderCodeGens(mFull, nFull, kFull, mFullBlockSizePerThread, nFullBlockSizePerThread);
    InitializePackingCodeGens(mFull, nFull, kFull, mFullBlockSizePerThread, nFullBlockSizePerThread);
    InitializeCodeGens(mFull, nFull, kFull, offsetSrc, offsetWei, offsetDst, offsetBia,
        IsLayoutAlikeBA(taskExec.dstInfo.GetLayout()), mFullBlockSizePerThread, nFullBlockSizePerThread,
        implTask.attributes);
}

MatMulDstFlags MatMulImplBase::InitializeLoadFlags(Layout dstLayout, bool hasBiaMBroadcast,
    bool hasBiaNBroadcast) noexcept
{
    if (hasBiaMBroadcast && hasBiaNBroadcast) {
        return MatMulDstFlags::LOAD_BIAS_1;
    } else if ((hasBiaMBroadcast && IsLayoutAlikeAB(dstLayout)) ||
        (hasBiaNBroadcast && !IsLayoutAlikeAB(dstLayout))) {
        return MatMulDstFlags::LOAD_BIAS_N;
    } else if ((hasBiaNBroadcast && IsLayoutAlikeAB(dstLayout)) ||
        (hasBiaMBroadcast && !IsLayoutAlikeAB(dstLayout))) {
        return MatMulDstFlags::LOAD_BIAS_M;
    } else {
        return MatMulDstFlags::LOAD_BIAS_MN;
    }
}

void MatMulImplBase::RunSequential(IntType threadID,
    const void *src, const void *wei, void *dst, const void *bia,
    std::unique_ptr<PackStrategy> &srcPackStrategy, std::unique_ptr<PackStrategy> &weiPackStrategy,
    const PostOpsDataPtrs &postOpsPtrs, float alpha, float beta) const noexcept(false)
{
    VERBOSE_PRINT;
    PostOpsDataPtrs shiftedByTIDPOPtrs = ShiftPODPbyThreadID(threadID, postOpsPtrs);

    IntType tIDbatch;
    IntType tIDm;
    IntType tIDn;
    InitThreadIDs(threadID, tIDbatch, tIDm, tIDn);

    srcPackStrategy->AllocateBuffers(perThreadParams[threadID].SrcBufferSize,
        tIDbatch * threadsM + tIDm, tIDn, threadsN);
    srcPackStrategy->Barrier(tIDbatch * threadsM + tIDm, tIDn, threadsN);

    weiPackStrategy->AllocateBuffers(perThreadParams[threadID].WeiBufferSize,
        tIDbatch * threadsN + tIDn, tIDm, threadsM);
    weiPackStrategy->Barrier(tIDbatch * threadsN + tIDn, tIDm, threadsM);

    // reorders will be single-threaded, since there's parallelization over it
    auto oldLocalNThreads = Threading::SetNumThreadsLocalInternal(1);
    // Loop over batched dimensions
    for (SizeType i0 = 0; i0 < perThreadParams[threadID].broadcastedDims[IDX_0]; ++i0) {
        for (SizeType i1 = 0; i1 < perThreadParams[threadID].broadcastedDims[IDX_1]; ++i1) {
            for (SizeType i2 = 0; i2 < perThreadParams[threadID].broadcastedDims[IDX_2]; ++i2) {
                const void *srcPtr = static_cast<const void *>(
                    static_cast<const std::byte *>(src) + perThreadParams[threadID].parallelStrideSrc +
                    (i0 * broadcastedStridesSrc[IDX_0] + i1 * broadcastedStridesSrc[IDX_1] +
                    i2 * broadcastedStridesSrc[IDX_2]) * taskExec.srcInfo.GetType().GetSize());
                const void *weiPtr = static_cast<const void *>(
                    static_cast<const std::byte *>(wei) + perThreadParams[threadID].parallelStrideWei +
                    (i0 * broadcastedStridesWei[IDX_0] + i1 * broadcastedStridesWei[IDX_1] +
                    i2 * broadcastedStridesWei[IDX_2]) * taskExec.weiInfo.GetType().GetSize());
                void *dstPtr = static_cast<void *>(
                    static_cast<std::byte *>(dst) + perThreadParams[threadID].parallelStrideDst +
                    (i0 * broadcastedStridesDst[IDX_0] + i1 * broadcastedStridesDst[IDX_1] +
                    i2 * broadcastedStridesDst[IDX_2]) * taskExec.dstInfo.GetType().GetSize());
                const void *biaPtr = bia;
                PostOpsDataPtrs shiftedByBatchPOPtrs = ShiftPODPbyBatch(i0, i1, i2, shiftedByTIDPOPtrs);
                if (biaPtr) {
                    biaPtr = static_cast<const void *>(
                        static_cast<const std::byte *>(biaPtr) + perThreadParams[threadID].parallelStrideBia +
                        (i0 * broadcastedStridesBia[IDX_0] + i1 * broadcastedStridesBia[IDX_1] +
                        i2 * broadcastedStridesBia[IDX_2]) * taskExec.biaInfo.GetType().GetSize());
                }
                const void *biaExec = perThreadParams[threadID].biaRW2D.Get(biaPtr);
                if (dstLayout2D == Layout::AB) {
                    RunSequential2D(threadID, srcPtr, weiPtr, dstPtr, biaExec,
                        srcPackStrategy, weiPackStrategy, alpha, beta, shiftedByBatchPOPtrs);
                } else {
                    // no need to swap srcPackStrategy and weiPackStrategy because
                    // they are consturcted using swapped parameters.
                    RunSequential2D(threadID, weiPtr, srcPtr, dstPtr, biaExec,
                        srcPackStrategy, weiPackStrategy, alpha, beta, shiftedByBatchPOPtrs);
                }
            }
        }
    }
    Threading::SetNumThreadsLocalInternal(oldLocalNThreads);
    srcPackStrategy->DeallocateBuffers(tIDbatch * threadsM + tIDm, tIDn, threadsN);
    weiPackStrategy->DeallocateBuffers(tIDbatch * threadsN + tIDn, tIDm, threadsM);
}

PostOpsDataPtrs MatMulImplBase::ShiftPODPbyThreadID(IntType threadID,
    const PostOpsDataPtrs &poPtrs) const
{
    // Will be done in next commit
    Service::Unused(threadID);
    PostOpsDataPtrs ret = poPtrs;
    return ret;
}

PostOpsDataPtrs MatMulImplBase::ShiftPODPbyBatch(SizeType b0, SizeType b1, SizeType b2,
    const PostOpsDataPtrs &poPtrs) const
{
    // Will be done in next commit
    Service::Unused(b0, b1, b2);
    PostOpsDataPtrs ret = poPtrs;
    return ret;
}

PackStrategy::PackStrategy(IntType threadsBatched, IntType numGroups, IntType groupSize,
    MatrixKind matrixKind) : mKind(matrixKind)
{
    bufferPtrs = std::unique_ptr<Service::MemoryBuffer[]>{new Service::MemoryBuffer[threadsBatched *
        numGroups * groupSize]};
    fast = std::unique_ptr<std::atomic<std::uint64_t>[]>{new std::atomic<std::uint64_t>[threadsBatched * numGroups]};
    slow = std::unique_ptr<std::atomic<std::uint64_t>[]>{new std::atomic<std::uint64_t>[threadsBatched * numGroups]};
    for (IntType i = 0; i < (threadsBatched * numGroups); ++i) {
        fast[i] = 0;
        slow[i] = 0;
    }
}

PrivateBufferPack::PrivateBufferPack(IntType threadsBatched, IntType numGroups, IntType groupSize,
    MatrixKind matrixKind) : PackStrategy(threadsBatched, numGroups, groupSize, matrixKind) {}

SharedBufferPack::SharedBufferPack(IntType threadsBatched, IntType numGroups, IntType groupSize,
    MatrixKind matrixKind) : PackStrategy(threadsBatched, numGroups, groupSize, matrixKind) {}

SharedBufferPackSplitDstDim::SharedBufferPackSplitDstDim(IntType threadsBatched, IntType numGroups, IntType groupSize,
    MatrixKind matrixKind) : SharedBufferPack(threadsBatched, numGroups, groupSize, matrixKind) {}
 
SharedBufferPackSplitAccDim::SharedBufferPackSplitAccDim(IntType threadsBatched, IntType numGroups, IntType groupSize,
    MatrixKind matrixKind) : SharedBufferPack(threadsBatched, numGroups, groupSize, matrixKind) {}

void PrivateBufferPack::Pack(const MatMulImplBase &matmulImpl, const void *originalMatrix, void *buffer,
    LoopCtrHelper dstDimension, LoopCtrHelper accumulationDimension,
    IntType threadGroupSize, IntType threadIDInsideGroup, IntType threadID, float alpha) const
{
    Service::Unused(threadGroupSize, threadIDInsideGroup);
    if ((dstDimension.size > 0) && (accumulationDimension.size > 0)) {
        IntType bufferStride = Math::CeilDiv(accumulationDimension.size, accumulationDimension.step) *
            accumulationDimension.step;
        matmulImpl.Pack(originalMatrix, buffer, threadID, dstDimension.size, accumulationDimension.size,
            dstDimension.start, accumulationDimension.start, 0, bufferStride, alpha, mKind);
    }
}

void SharedBufferPackSplitDstDim::Pack(const MatMulImplBase &matmulImpl, const void *originalMatrix, void *buffer,
    LoopCtrHelper dstDimension, LoopCtrHelper accumulationDimension,
    IntType threadGroupSize, IntType threadIDInsideGroup, IntType threadID, float alpha) const
{
    if ((dstDimension.size > 0) && (accumulationDimension.size > 0)) {
        IntType numBlocks = Math::CeilDiv(dstDimension.size, dstDimension.step);
        IntType numBlocksPerThreadCommon = Math::CeilDiv(numBlocks, threadGroupSize);
        IntType packPortionCommon = std::min(dstDimension.size, numBlocksPerThreadCommon * dstDimension.step);
        IntType numPackThreads = Math::CeilDiv(dstDimension.size, packPortionCommon);
        IntType packPortionResid = dstDimension.size - packPortionCommon * (numPackThreads - NUM_1);
        IntType numBlocksPerThreadResid = numBlocks - numBlocksPerThreadCommon * (numPackThreads - NUM_1);

        IntType numBlocksPerThread = numBlocksPerThreadCommon;
        if (threadIDInsideGroup == (numPackThreads - NUM_1)) {
            numBlocksPerThread = numBlocksPerThreadResid;
        } else if (threadIDInsideGroup > (numPackThreads - NUM_1)) {
            numBlocksPerThread = 0;
        }
        IntType threadOffset = packPortionCommon * threadIDInsideGroup;
        for (IntType packBlockIdx = 0; packBlockIdx < numBlocksPerThread; ++packBlockIdx) {
            IntType packSize = dstDimension.step;
            if ((threadIDInsideGroup == (numPackThreads - NUM_1)) && (packBlockIdx == (numBlocksPerThread - NUM_1))) {
                packSize = packPortionResid - packSize * (numBlocksPerThread - NUM_1);
            }
            IntType dstDimOffset = threadOffset + packBlockIdx * dstDimension.step;
            IntType bufferStride = Math::CeilDiv(accumulationDimension.size, accumulationDimension.step) *
                accumulationDimension.step;
            IntType bufferOffset = dstDimOffset * bufferStride;
            matmulImpl.Pack(originalMatrix, buffer, threadID, packSize, accumulationDimension.size,
                dstDimension.start + dstDimOffset, accumulationDimension.start, bufferOffset, bufferStride,
                alpha, mKind);
        }
    }
}

void SharedBufferPackSplitAccDim::Pack(const MatMulImplBase &matmulImpl, const void *originalMatrix, void *buffer,
    LoopCtrHelper dstDimension, LoopCtrHelper accumulationDimension,
    IntType threadGroupSize, IntType threadIDInsideGroup, IntType threadID, float alpha) const
{
    if ((dstDimension.size > 0) && (accumulationDimension.size > 0)) {
        IntType numBlocks = Math::CeilDiv(accumulationDimension.size, accumulationDimension.step);
        IntType numBlocksPerThreadCommon = Math::CeilDiv(numBlocks, threadGroupSize);
        IntType packPortionCommon = std::min(accumulationDimension.size, numBlocksPerThreadCommon *
            accumulationDimension.step);
        IntType numPackThreads = Math::CeilDiv(accumulationDimension.size, packPortionCommon);
        IntType packPortionResid = accumulationDimension.size - packPortionCommon * (numPackThreads - 1);
        IntType numKBlocksPerThreadResid = numBlocks - numBlocksPerThreadCommon * (numPackThreads - 1);

        IntType numBlocksPerThread = numBlocksPerThreadCommon;
        if (threadIDInsideGroup == (numPackThreads - NUM_1)) {
            numBlocksPerThread = numKBlocksPerThreadResid;
        } else if (threadIDInsideGroup > (numPackThreads - NUM_1)) {
            numBlocksPerThread = 0;
        }

        IntType threadOffset = packPortionCommon * threadIDInsideGroup;

        for (IntType packBlockIdx = 0; packBlockIdx < numBlocksPerThread; ++packBlockIdx) {
            IntType packSize = accumulationDimension.step;
            if ((threadIDInsideGroup == (numPackThreads - NUM_1)) && (packBlockIdx == (numBlocksPerThread - NUM_1))) {
                packSize = packPortionResid - packSize * (numBlocksPerThread - NUM_1);
            }

            IntType accDimOffset = threadOffset + packBlockIdx * accumulationDimension.step;
            IntType bufferStride = Math::CeilDiv(accumulationDimension.size, accumulationDimension.step) *
                accumulationDimension.step;

            if (dstDimension.size > dstDimension.step) {
                IntType currDstDimBlock = (dstDimension.size % dstDimension.step == 0) ?
                    (dstDimension.size - dstDimension.step)
                    : (dstDimension.size - (dstDimension.size % dstDimension.step));
                IntType bufferOffset1 = accDimOffset * dstDimension.step;
                // (dstDimension.size - currDstDimBlock) - is resid part of dst size
                IntType bufferOffset2 = currDstDimBlock * bufferStride + accDimOffset *
                    (dstDimension.size - currDstDimBlock);
                matmulImpl.Pack(originalMatrix, buffer, threadID, currDstDimBlock, packSize,
                    dstDimension.start, accumulationDimension.start + accDimOffset, bufferOffset1,
                    bufferStride, alpha, mKind);
                matmulImpl.Pack(originalMatrix, buffer, threadID, dstDimension.size - currDstDimBlock, packSize,
                    dstDimension.start + currDstDimBlock, accumulationDimension.start + accDimOffset, bufferOffset2,
                    bufferStride, alpha, mKind);
            } else {
                IntType bufferOffset = accDimOffset * dstDimension.size;
                matmulImpl.Pack(originalMatrix, buffer, threadID, dstDimension.size, packSize,
                    dstDimension.start, accumulationDimension.start + accDimOffset, bufferOffset,
                    bufferStride, alpha, mKind);
            }
        }
    }
}

void PrivateBufferPack::Barrier(IntType groupID, IntType threadIDInsideGroup, IntType groupSize)
{
    Service::Unused(groupID, threadIDInsideGroup, groupSize);
}

void SharedBufferPack::Barrier(IntType groupID, IntType threadIDInsideGroup, IntType groupSize)
{
    assert(groupSize > 0);
    if (groupSize != NUM_1) {
        std::uint64_t newSlow = slow[groupID] + static_cast<std::uint64_t>(groupSize);
        ++fast[groupID];
        if (threadIDInsideGroup == 0) {
            while (fast[groupID] != newSlow) {}
            // notify other threads inside group
            slow[groupID] = newSlow;
        } else {
            // wait until master thread in the group notifies
            while (slow[groupID] != newSlow) {}
        }
    }
}

void PrivateBufferPack::AllocateBuffers(IntType bufferSize, IntType groupID,
    IntType threadIDInsideGroup, IntType groupSize)
{
    bufferPtrs[groupID * groupSize + threadIDInsideGroup] =
        Service::MemoryBuffer(bufferSize, Service::MemoryType::FAST);
}

void SharedBufferPack::AllocateBuffers(IntType bufferSize, IntType groupID,
    IntType threadIDInsideGroup, IntType groupSize)
{
    if (threadIDInsideGroup == 0) {
        bufferPtrs[groupID * groupSize] = Service::MemoryBuffer(bufferSize, Service::MemoryType::FAST);
        for (IntType i = NUM_1; i < groupSize; ++i) {
            bufferPtrs[groupID * groupSize + i] = bufferPtrs[groupID * groupSize];
        }
    }
}

void PrivateBufferPack::DeallocateBuffers(IntType groupID,
    IntType threadIDInsideGroup, IntType groupSize)
{
    bufferPtrs[groupID * groupSize + threadIDInsideGroup].Free();
}

void SharedBufferPack::DeallocateBuffers(IntType groupID,
    IntType threadIDInsideGroup, IntType groupSize)
{
    if (threadIDInsideGroup == 0) {
        for (IntType i = 0; i < groupSize; ++i) {
            bufferPtrs[groupID * groupSize + i].Free();
        }
    }
}

void* PackStrategy::GetPtr(IntType groupID, IntType threadIDInsideGroup, IntType groupSize)
{
    return bufferPtrs[groupID * groupSize + threadIDInsideGroup].ptr;
}

void MatMulImplBase::InitThreadIDs(IntType threadID, IntType &tIDbatch, IntType &tIDm, IntType &tIDn) const
{
    tIDbatch = threadID / (threadsM * threadsN);
    IntType tIDInsideBatch = threadID % (threadsM * threadsN);
    // Thread ordering is always in AB-like format for m and n dimensions
    // this is done to have thread partition info aligned with
    // tensorInfo. So, in case of dst in BA-like layout we
    // need to compute thread IDs differently since threadsM
    // and threadsN are already swapped.
    tIDm = tIDInsideBatch / threadsN;
    tIDn = tIDInsideBatch % threadsN;
    if (dstLayout2D == Layout::COL_MAJOR) {
        tIDm = tIDInsideBatch % threadsM;
        tIDn = tIDInsideBatch / threadsM;
    }
}

} // Detail
} // KuDNN

