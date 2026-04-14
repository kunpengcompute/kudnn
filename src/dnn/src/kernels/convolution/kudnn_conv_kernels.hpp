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

#ifndef KUDNN_CONV_KERNELS_HPP
#define KUDNN_CONV_KERNELS_HPP

#include "kudnn.hpp"

#include "vector_wrappers.hpp"
#include "kudnn_conv_deconv_base.hpp"

namespace KuDNN {
namespace Detail {

template <typename ComputeType>
struct ConvKernel final {
    using CompType = ComputeType;
    static IntType OcSize() noexcept
    {
        return NUM_3 * Vector::Length<CompType>();
    }
    static constexpr IntType OwSize() noexcept
    {
        return NUM_2 * (NUM_16 / sizeof(CompType));
    }
    template <Layout Lt, IntType KernelSize, IntType ExecSize, IntType sw>
    static void Do(const CompType *src, const CompType *wei, CompType *dst,
        IntType dstStride, IntType dstIdx, IntType icBlock) noexcept;
};

template <typename ComputeType, typename DataType>
void Conv2D(const ConvolutionParams &params, const DataType *src,
    const DataType *wei, DataType *dst,
    const DataType *bia) noexcept(false);

} // Detail
} // KuDNN

#endif // KUDNN_CONV_KERNELS_HPP
