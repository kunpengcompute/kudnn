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

struct MatmulThrPartSolverIFace {
    virtual double EstimateEffectiveness(IntType m, IntType n, IntType threadsM, IntType threadsN,
        IntType nThreads) const noexcept = 0;
    virtual ~MatmulThrPartSolverIFace() = default;
};

struct MatmulThrPartSolverAdvanced : MatmulThrPartSolverIFace {
    double EstimateEffectiveness(IntType m, IntType n, IntType threadsM, IntType threadsN,
        IntType nThreads) const noexcept override
    {
        return CalculateMetric1(threadsM, threadsN, nThreads) + CalculateMetric2(m, n, threadsM, threadsN);
    }
protected:
    double CalculateMetric1(IntType threadsM, IntType threadsN, IntType nThreads) const
    {
        double thrsM = threadsM;
        double thrsN = threadsN;
        double nThrs = nThreads;

        double metric1 = (nThrs - (thrsM * thrsN)) / nThrs;
        return metric1;
    }
    double CalculateMetric2(IntType m, IntType n, IntType threadsM, IntType threadsN) const
    {
        double mSize = m;
        double nSize = n;
        double metricImportanceRatio = NUM_1;
        double skewness = NUM_1;
        double mPortion = Math::CeilDiv(m, threadsM);
        double nPortion = Math::CeilDiv(n, threadsN);
#ifndef KUDNNL_920F
        metricImportanceRatio = NUM_2;
        skewness = nSize / mSize;
        skewness = std::max(2.0, skewness); // 2.0 is a practically measured number
        skewness = std::min(3.0, skewness); // 3.0 is a practically measured number
#endif
        double metric2 = std::abs((mPortion - skewness * nPortion) /
            (std::sqrt(NUM_1 + skewness * skewness) * (mSize + nSize)));
        return metricImportanceRatio * metric2;
    }
};

struct MatmulThrPartSolverSimple : MatmulThrPartSolverIFace {
    double EstimateEffectiveness(IntType m, IntType n, IntType threadsM, IntType threadsN,
        IntType nThreads) const noexcept override
    {
        Service::Unused(nThreads);
        IntType mPortion = Math::CeilDiv(m, threadsM);
        IntType nPortion = Math::CeilDiv(n, threadsN);
        return mPortion * nPortion;
    }
};

// Update Stride value for each from dimension right to left itteravly
Shape UpdateDims(const Shape &sizeInitial, const Shape &stridePrev, Shape increment) noexcept(false);
// Update Dims value for each from dimension right to left itteravly
Shape UpdateStride(const Shape &sizeInitial, const Shape &stridePrev, const Shape &dimsPrev);
// Calculate balanced thread partition for 2D GEMM based on MxN sizes. Main idea: miminize work on thread
MatMul2DPartition GetMatMul2DPartitionBalanced(IntType m, IntType n, IntType nThreads,
    std::unique_ptr<MatmulThrPartSolverIFace>& metric);

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

// calculate data dims and strides on each thread. Returns the partition which is not affected by dimension swap
template <typename KernelInfo>
std::vector<std::pair<Shape, Shape>> CalculateWorkOnThread(const Shape &batchedDims, IntType nThreads,
    bool swapMNBlocks)
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
    // swap dimensions since kernel will operate in swapped m and n in case of BA-like layout
    if (swapMNBlocks) {
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

        std::unique_ptr<MatmulThrPartSolverIFace> metric;

        if (std::getenv("KUDNN_FORCE_NEON")) {
            metric = std::make_unique<MatmulThrPartSolverSimple>();
        } else {
            metric = std::make_unique<MatmulThrPartSolverAdvanced>();
        }

        if ((m >= (NUM_8 * mBlock)) && (n <= (NUM_3 * nBlock))) {
            tp = GetMatMul2DPartitionBalanced(m, nPart, thResid, metric);
            localMCommon = Math::CeilDiv(m, tp.m);
            localNCommon = std::min(Math::CeilDiv(nPart, tp.n) * nBlock, n);
        } else if ((m <= (NUM_3 * mBlock)) && (n >= (NUM_8 * nBlock))) {
            tp = GetMatMul2DPartitionBalanced(mPart, n, thResid, metric);
            localMCommon = std::min(Math::CeilDiv(mPart, tp.m) * mBlock, m);
            localNCommon = Math::CeilDiv(n, tp.n);
        } else {
            tp = GetMatMul2DPartitionBalanced(m, n, thResid, metric);
            localMCommon = Math::CeilDiv(m, tp.m);
            localNCommon = Math::CeilDiv(n, tp.n);
        }

        SizeType mThreadRealUsage = Math::CeilDiv(m, localMCommon);
        SizeType nThreadRealUsage = Math::CeilDiv(n, localNCommon);

        // swap back to be consistent with batchedDims
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

IntType InitThreadNum();

} // Detail
} // KuDNN

#endif // KUDNN_GEMM_THREADING_HELPERS_HPP
