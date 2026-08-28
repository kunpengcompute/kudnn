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

#ifndef KUDNN_POOLING_HPP
#define KUDNN_POOLING_HPP

#include <memory>

#include "types/kudnn_tensor_info.hpp"
#include "service/kudnn_err_codes.hpp"
#include "types/kudnn_propagation_type.hpp"

namespace KuDNN {

enum class PoolingFunction {
    UNDEFINED,
    MAX,
    AVG_EXCLUDE_PADDING,
    AVG_INCLUDE_PADDING,
};

namespace Detail {

class PoolingLayerFWDImpl;
class PoolingLayerBWDImpl;

} // Detail

class PoolingLayerFWD final {
public:
    using SizeType = ::KuDNN::SizeType;
    static Status ValidateInput(const Propagation &propKind, const PoolingFunction &poolingAlg,
        const TensorInfo &srcInfo, const TensorInfo &dstInfo, const Shape &strides,
        const Shape &kernel, const Shape &dilation, const Shape &paddingL, const Shape &paddingR) noexcept;
    PoolingLayerFWD(const Propagation &propKind, const PoolingFunction &poolingAlg,
        const TensorInfo &srcInfo, const TensorInfo &dstInfo, const Shape &strides,
        const Shape &kernel, const Shape &dilation, const Shape &paddingL, const Shape &paddingR) noexcept(false);
    PoolingLayerFWD(const PoolingLayerFWD &other) noexcept(false);
    PoolingLayerFWD(PoolingLayerFWD &&other) noexcept;
    PoolingLayerFWD& operator=(const PoolingLayerFWD &other) noexcept(false);
    PoolingLayerFWD& operator=(PoolingLayerFWD &&other) noexcept;
    void Run(const void *src, void *dst, void *ws = nullptr) const noexcept(false);
    void SetPadType(Element::Type newPadType) noexcept;
    ~PoolingLayerFWD() noexcept;
private:
    std::unique_ptr<Detail::PoolingLayerFWDImpl> pImpl;
};

class PoolingLayerBWD final {
public:
    using SizeType = ::KuDNN::SizeType;
    static Status ValidateInput(const PoolingFunction &poolingAlg,
        const TensorInfo &srcDiffInfo, const TensorInfo &dstDiffInfo, const Shape &strides,
        const Shape &kernel, const Shape &dilation, const Shape &paddingL, const Shape &paddingR) noexcept;
    PoolingLayerBWD(const PoolingFunction &poolingAlg,
        const TensorInfo &srcDiffInfo, const TensorInfo &dstDiffInfo, const Shape &strides,
        const Shape &kernel, const Shape &dilation, const Shape &paddingL, const Shape &paddingR) noexcept(false);
    PoolingLayerBWD(const PoolingLayerBWD &other) noexcept(false);
    PoolingLayerBWD(PoolingLayerBWD &&other) noexcept;
    PoolingLayerBWD& operator=(const PoolingLayerBWD &other) noexcept(false);
    PoolingLayerBWD& operator=(PoolingLayerBWD &&other) noexcept;
    void Run(void *diffSrc, const void *diffDst, const void *ws = nullptr) const noexcept(false);
    ~PoolingLayerBWD() noexcept;
private:
    std::unique_ptr<Detail::PoolingLayerBWDImpl> pImpl;
};

} // KuDNN

#endif // KUDNN_POOLING_HPP
