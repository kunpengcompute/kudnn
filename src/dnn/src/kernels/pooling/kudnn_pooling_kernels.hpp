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

#ifndef KUDNN_POOLING_KERNELS_HPP
#define KUDNN_POOLING_KERNELS_HPP

#include "kudnn.hpp"
#include "vector_wrappers.hpp"

namespace KuDNN {
namespace Detail {
namespace Pooling {
namespace Helpers {
    constexpr SizeType U8_MAX = 255;

    template<typename Type>
    inline constexpr float MIN_VAL = std::numeric_limits<Type>::lowest();

    template<>
    inline constexpr float MIN_VAL<__fp16> = -65504.0f;

    template<>
    inline constexpr float MIN_VAL<__bf16> = -3.38953139e38f;

    template<PoolingFunction Func, typename InType, typename CompType, typename OutType, typename WsType, bool UpdWs>
    struct PoolingTraits {
        static void ComputeBWD(IntType indDst, IntType indSrc, CompType *typedSrc, const OutType *typedDst,
            const WsType *typedWs, IntType iw, IntType kw, IntType dw, IntType sw, IntType pw, IntType ow,
            IntType delimeter = 1, IntType wStart = 0, IntType wEnd = 0)
        {
            IntType iwStride;
            if constexpr (Func == PoolingFunction::MAX) {
                WsType index = typedWs[indDst];
                IntType kwMod = index % kw;
                iwStride = ow * sw - pw + kwMod * dw;
                if ((iwStride < 0) || (iwStride >= iw)) {
                    return;
                }
                typedSrc[indSrc + iwStride] +=
                    Vector::Convert<OutType, CompType>::Do(typedDst[indDst]);
            } else {
                for (IntType kwIter = wStart; kwIter < wEnd; ++kwIter) {
                    iwStride = ow * sw - pw + kwIter * dw;
                    typedSrc[indSrc + iwStride] +=
                        Vector::Convert<OutType, CompType>::Do(typedDst[indDst]) / delimeter;
                }
            }
        }
    };

} // namespace Helpers

template<PoolingFunction Func, typename CompType, typename InType, typename OutType, typename WsType, bool UpdWs>
void EvaluateAxbFWD(const Shape &start, const Shape &end, const InType *typedSrc, OutType *typedDst, WsType *typedWs,
    const Shape &initialDims, const Shape &stridesSrc, const Shape &stridesDst, const Shape &kernel,
    const Shape &strides, const Shape &dilation, const Shape &paddingL, SizeType ndims, CompType init) noexcept;

template<PoolingFunction Func, typename CompType, typename InType, typename OutType, typename WsType, bool UpdWs>
void EvaluateAbxFWD(const Shape &start, const Shape &end, const InType *typedSrc, OutType *typedDst, WsType *typedWs,
    const Shape &initialDims, const Shape &outputDims, const Shape &stridesSrc, const Shape &stridesDst,
    const Shape &kernel, const Shape &strides, const Shape &dilation, const Shape &paddingL,
    SizeType ndims, CompType init, const std::vector<ReorderLayer> &reorders) noexcept;

template<PoolingFunction Func, typename CompType, typename InType, typename OutType, typename WsType>
void EvaluateAxbBWD(const Shape &start, const Shape &end, InType *typedSrc, const OutType *typedDst,
    const WsType *typedWs, float *srcTmpBuf, const Shape &initialDims, const Shape &outputDims,
    const Shape &stridesSrc, const Shape &stridesDst, const Shape &kernel,
    const Shape &strides, const Shape &dilation, const Shape &paddingL, SizeType ndims) noexcept;

template<PoolingFunction Func, typename CompType, typename InType, typename OutType, typename WsType>
void EvaluateAbxBWD(const Shape &start, const Shape &end, InType *typedSrc, const OutType *typedDst,
    const WsType *typedWs, float *srcTmpBuf, const Shape &initialDims, const Shape &outputDims,
    const Shape &stridesSrc, const Shape &stridesDst, const Shape &kernel,
    const Shape &strides, const Shape &dilation, const Shape &paddingL, SizeType ndims) noexcept;

} // Pooling
} // Detail
} // KuDNN

#endif //  KUDNN_POOLING_KERNELS_HPP
