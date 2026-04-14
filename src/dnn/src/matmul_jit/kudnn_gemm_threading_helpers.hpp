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

#ifndef KUDNN_GEMM_THREADING_HELPERS_HPP
#define KUDNN_GEMM_THREADING_HELPERS_HPP

#include <cstdint>
#include <cmath>
#include <vector>

#include "kudnn.hpp"

namespace KuDNN {
namespace Detail {

struct MatMul2DPartition final {
    IntType m;
    IntType n;
};

// calculate thread usage for GEMM 2d
IntType CalculateThreadWorkSize(IntType m, IntType n, IntType threadsM, IntType threadsN) noexcept;
// Update Stride value for each from dimension right to left itteravly
Shape UpdateDims(const Shape &sizeInitial, const Shape &stridePrev, Shape increment) noexcept(false);
// Update Dims value for each from dimension right to left itteravly
Shape UpdateStride(const Shape &sizeInitial, const Shape &stridePrev, const Shape &dimsPrev);
// Calculate balanced thread partition for 2D GEMM based on MxN sizes. Main idea: miminize work on thread
MatMul2DPartition GetMatMul2DPartitionBalanced(IntType m, IntType n, IntType nThreads);

struct MatMulThreadPartitioner {
    static void Do(const SizeType multiDimLoopCounter[], std::pair<Shape, Shape>* &prevWork,
        std::vector<std::pair<Shape, Shape>> &gemmWork, const Shape &inrements, const Shape &batchedDims) noexcept
    {
        Shape strideCur;
        Shape dimsCur;
        // first iteration differs from other iterations
        if ((multiDimLoopCounter[IDX_0] == 0) && (multiDimLoopCounter[IDX_1] == 0) &&
            (multiDimLoopCounter[IDX_2] == 0) && (multiDimLoopCounter[IDX_3] == 0) &&
            (multiDimLoopCounter[IDX_4] == 0)) {
            strideCur = Shape(multiDimLoopCounter, batchedDims.GetNumDims());
            dimsCur = inrements;
        } else {
            strideCur = UpdateStride(batchedDims, prevWork->second, prevWork->first);
            dimsCur = UpdateDims(batchedDims, strideCur, inrements);
        }
        gemmWork.push_back({dimsCur, strideCur});
        prevWork = &gemmWork.back();
    }
};

// calculate data dims and strides on each thread
template <typename KernelInfo>
std::vector<std::pair<Shape, Shape>> CalculateWorkOnThread(const Shape &batchedDims, IntType nThreads,
    bool swapMNBlocks, bool isKBig)
{
    auto &&[lastDimSplitIdx, prod, splitDims] = Threading::GetSplitDimInfo(batchedDims,
        static_cast<SizeType>(nThreads));
    Service::Unused(splitDims);

    std::array <SizeType, MAX_DIMS> realThreadsUsageSet{{1, 1, 1, 1, 1}};
    Shape increments = batchedDims;

    IntType mBlock = KernelInfo::GetKernelMSize();
    IntType nBlock = KernelInfo::GetKernelNSize();
    IntType m = batchedDims[batchedDims.GetNumDims() - NUM_2];
    IntType n = batchedDims[batchedDims.GetNumDims() - NUM_1];
    if (swapMNBlocks) {
        std::swap(mBlock, nBlock);
        std::swap(m, n);
    }
    IntType mPart = Math::CeilDiv(m, mBlock);
    IntType nPart = Math::CeilDiv(n, nBlock);
    // restore available thread quantity for the last 2 dimensions
    if (lastDimSplitIdx == (batchedDims.GetNumDims() - NUM_1)) {
        prod /= batchedDims[lastDimSplitIdx - 1];
    }
    IntType thResid = nThreads / prod;
    for (SizeType i = 0; i < batchedDims.GetNumDims() - NUM_2; i++) {
        if (i < lastDimSplitIdx) {
            increments[i] = NUM_1;
            realThreadsUsageSet[i] = batchedDims[i];
        } else if (i == lastDimSplitIdx) {
            increments[i] = Math::CeilDiv(batchedDims[i], thResid);
            realThreadsUsageSet[i] = Math::CeilDiv(batchedDims[i], increments[i]);
        }
    }
    if (lastDimSplitIdx >= (batchedDims.GetNumDims() - NUM_2)) {
        SizeType localMCommon = m;
        SizeType localNCommon = n;
        MatMul2DPartition tp;
        // all the cases are checked explicitly since it allows to avoid very complicated logic
        // this leads to some code duplication, but otherwise code logic becomes hardly understandable
        Service::Unused(isKBig);
        if (isKBig) {
            if (n < (NUM_2 * m)) {
                if (m > (NUM_8 * thResid)) {
                    tp = GetMatMul2DPartitionBalanced(m, 1, thResid);
                } else {
                    tp = GetMatMul2DPartitionBalanced(m, n, thResid);
                }
            } else {
                if (nPart > (NUM_8 * thResid)) {
                    tp = GetMatMul2DPartitionBalanced(1, n, thResid);
                } else {
                    tp = GetMatMul2DPartitionBalanced(m, n, thResid);
                }
            }
        } else {
            // split over two dimensions anyways
            if ((mPart * nPart) > (NUM_8 * thResid)) {
                tp = GetMatMul2DPartitionBalanced(mPart, nPart, thResid);
            } else {
                tp = GetMatMul2DPartitionBalanced(m, n, thResid);
            }
        }
        if (isKBig && (mPart > NUM_8)) { // sequential pipeline gives ~4x worse perf for
        // `1xKernelNSize` kernel than for `KernelMSizexKernelNSize`,
        // so having at least 4 kernels inside thread will will
        // make the last piece negligible (1xKERNEL_OPS/1xKERNEL_PERF << 4*FULL_KERNEL_OPS/FULL_KERNEL_PERF)
            localMCommon = Math::CeilDiv(m, tp.m);
        } else {
            localMCommon = std::min(Math::CeilDiv(mPart, tp.m) * mBlock, m);
        }
        if (isKBig && (nPart > NUM_4)) { // sequential pipeline gives ~8x worse perf for
        // `KernelMSizex1` kernel than for `KernelMSizexKernelNSize`,
        // so having at least 4 kernels inside thread will will
        // make the last piece negligible (1xKERNEL_OPS/1xKERNEL_PERF << 4*FULL_KERNEL_OPS/FULL_KERNEL_PERF)
            localNCommon = Math::CeilDiv(n, tp.n);
        } else {
            localNCommon = std::min(Math::CeilDiv(nPart, tp.n) * nBlock, n);
        }

        SizeType mThreadRealUsage = Math::CeilDiv(m, localMCommon);
        SizeType nThreadRealUsage = Math::CeilDiv(n, localNCommon);

        if (swapMNBlocks) {
            std::swap(mThreadRealUsage, nThreadRealUsage);
            std::swap(localMCommon, localNCommon);
        }

        realThreadsUsageSet[batchedDims.GetNumDims() - NUM_2] = mThreadRealUsage;
        realThreadsUsageSet[batchedDims.GetNumDims() - NUM_1] = nThreadRealUsage;

        increments[batchedDims.GetNumDims() - NUM_2] = localMCommon;
        increments[batchedDims.GetNumDims() - NUM_1] = localNCommon;
    }
    Shape realThreadsUsageDims(realThreadsUsageSet.data(), batchedDims.GetNumDims());
    // ==== gemm work on thread data  ===
    std::vector<std::pair<Shape, Shape>> gemmWork;
    std::pair<Shape, Shape>* prevStrideDim = nullptr;

    Service::ForCaller::Do(realThreadsUsageDims, MatMulThreadPartitioner{}, prevStrideDim,
        gemmWork, increments, batchedDims);

    return gemmWork;
}

} // Detail
} // KuDNN

#endif // KUDNN_GEMM_THREADING_HELPERS_HPP
