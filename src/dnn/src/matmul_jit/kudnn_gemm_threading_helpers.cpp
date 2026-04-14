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

#include "kudnn.hpp"

#include "threading_impl.hpp"
#include "matmul_jit/kudnn_gemm_threading_helpers.hpp"

namespace KuDNN {
namespace Detail {

IntType CalculateThreadWorkSize(IntType m, IntType n, IntType threadsM, IntType threadsN) noexcept
{
    IntType mPortion = Math::CeilDiv(m, threadsM);
    IntType nPortion = Math::CeilDiv(n, threadsN);
    return mPortion * nPortion;
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

MatMul2DPartition GetMatMul2DPartitionBalanced(IntType m, IntType n, IntType nThreads)
{
    MatMul2DPartition tp = {1, 1};
    // minimize  work per thread in scope of available threads
    IntType maxTh = INT_MAX;
    for (IntType i = 1; i <= nThreads; ++i) {
        IntType j =  nThreads / i;
        IntType activeTh = CalculateThreadWorkSize(m, n, i, j);
        if (activeTh < maxTh) {
            maxTh = activeTh;
            tp = {i, j};
        }
    }
    return tp;
}

} // Detail
} // KuDNN
