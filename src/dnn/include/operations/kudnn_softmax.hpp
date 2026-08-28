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

#ifndef KUDNN_SOFTMAX_HPP
#define KUDNN_SOFTMAX_HPP

#include <memory>

#include "types/kudnn_tensor_info.hpp"
#include "service/kudnn_err_codes.hpp"

namespace KuDNN {

enum class AlgorithmKind : std::uint32_t {
    SOFTMAX         = 0x0U,
    LOGSOFTMAX      = 0x1U,
};

namespace Detail {

class SoftmaxLayerFWDImpl;
class SoftmaxLayerBWDImpl;

} // Detail

class SoftmaxLayerFWD final {
public:
    using SizeType = ::KuDNN::SizeType;
    static Status ValidateInput(const TensorInfo &srcInfo, const TensorInfo &dstInfo, SizeType axis,
        AlgorithmKind algorithm) noexcept;
    SoftmaxLayerFWD(const TensorInfo &srcInfo, const TensorInfo &dstInfo, SizeType axis,
        AlgorithmKind algorithm) noexcept(false);
    SoftmaxLayerFWD(const SoftmaxLayerFWD &other) noexcept(false);
    SoftmaxLayerFWD(SoftmaxLayerFWD &&other) noexcept;
    SoftmaxLayerFWD& operator=(const SoftmaxLayerFWD &other) noexcept(false);
    SoftmaxLayerFWD& operator=(SoftmaxLayerFWD &&other) noexcept;
    void Run(const void *src, void *dst) const noexcept(false);
    ~SoftmaxLayerFWD() noexcept;
private:
    std::unique_ptr<Detail::SoftmaxLayerFWDImpl> pImpl;
};

class SoftmaxLayerBWD final {
public:
    using SizeType = ::KuDNN::SizeType;
    static Status ValidateInput(const TensorInfo &dstInfo, const TensorInfo &dstDiffInfo,
                                const TensorInfo &srcDiffInfo, SizeType axis, AlgorithmKind algorithm) noexcept;
    SoftmaxLayerBWD(const TensorInfo &dstInfo, const TensorInfo &dstDiffInfo, const TensorInfo &srcDiffInfo,
	                SizeType axis, AlgorithmKind algorithm) noexcept(false);
    SoftmaxLayerBWD(const SoftmaxLayerBWD &other) noexcept(false);
    SoftmaxLayerBWD(SoftmaxLayerBWD &&other) noexcept;
    SoftmaxLayerBWD& operator=(const SoftmaxLayerBWD &other) noexcept(false);
    SoftmaxLayerBWD& operator=(SoftmaxLayerBWD &&other) noexcept;
    void Run(const void *dst, const void *dstDiff, void *srcDiff) const noexcept(false);
    ~SoftmaxLayerBWD() noexcept;
private:
    std::unique_ptr<Detail::SoftmaxLayerBWDImpl> pImpl;
};

} // KuDNN

#endif // KUDNN_SOFTMAX_HPP
