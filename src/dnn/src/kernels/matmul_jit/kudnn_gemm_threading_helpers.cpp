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
#include <climits>
#include <cmath>

#include "kudnn.hpp"

#include "threading_impl.hpp"
#include "kernels/matmul_jit/kudnn_gemm_threading_helpers.hpp"

namespace KuDNN {
namespace Detail {

IntType InitThreadNum()
{
#ifdef KUDNNL_920G
    return NUM_96;
#else
    return NUM_80;
#endif
}

Shape UpdateStride(const Shape &sizeInitial, const Shape &stridePrev, const Shape &dimsPrev) noexcept(false)
{
    Shape outStride = stridePrev;
    for (IntType i = static_cast<IntType>(stridePrev.GetNumDims()) - 1; i >= 0; --i) {
        outStride[i] = stridePrev[i] + dimsPrev[i];
        if (outStride[i] < sizeInitial[i]) {
            break;
        } else {
            if (i == 0) {
                throw Service::LogicError("Shape: 0-idx stride cannot be equal 0");
            }
            outStride[i] = 0;
        }
    }
    return outStride;
}

Shape UpdateDims(const Shape &sizeInitial, const Shape &stridePrev, Shape increment)
{
    for (SizeType i = 0; i < stridePrev.GetNumDims(); ++i) {
        if (stridePrev[i] + increment[i] > sizeInitial[i]) {
            increment[i] = sizeInitial[i] - stridePrev[i];
        }
    }
    return increment;
}

MatMul2DPartition GetMatMul2DPartitionBalanced(IntType m, IntType n, IntType nThreads,
    std::unique_ptr<MatmulThrPartSolverIFace>& metric)
{
    MatMul2DPartition tp = {NUM_1, NUM_1};
    double maxMetric = std::numeric_limits<double>::max();
    // Checking all possbile pairs of thread kernels
    // Thread kernel with the lowest currMetric number is the best for this task

    for (IntType i = NUM_1; i <= nThreads; ++i) {
        IntType j =  nThreads / i;
        IntType usedThreadsM =  std::min(m, i);
        IntType usedThreadsN = std::min(n, j);
        double currMetric = metric->EstimateEffectiveness(m, n, usedThreadsM, usedThreadsN, nThreads);
        if ((currMetric < maxMetric)) {
            maxMetric = currMetric;
            tp = {i, j};
        }
    }
    return tp;
}

} // Detail
} // KuDNN
