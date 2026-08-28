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

#ifndef KUDNN_INNER_PRODUCT_HPP
#define KUDNN_INNER_PRODUCT_HPP

#include <memory>

#include "types/kudnn_tensor_info.hpp"
#include "service/kudnn_err_codes.hpp"

namespace KuDNN {

namespace Detail {

class InnerProductLayerFWDImpl;
class InnerProductLayerBWDDataImpl;
class InnerProductLayerBWDWeightsImpl;

} // Detail

class InnerProductLayerFWD final {
public:
    using SizeType = ::KuDNN::SizeType;
    static Status ValidateInput(const TensorInfo &src, const TensorInfo &weights,
        const TensorInfo &dst, const TensorInfo &bias) noexcept;
    static Status ValidateInput(const TensorInfo &src, const TensorInfo &weights,
        const TensorInfo &dst, const TensorInfo &bias, const Attributes &attributes) noexcept;
    InnerProductLayerFWD(const TensorInfo &src, const TensorInfo &weights,
        const TensorInfo &dst, const TensorInfo &bias) noexcept(false);
    InnerProductLayerFWD(const TensorInfo &src, const TensorInfo &weights,
        const TensorInfo &dst, const TensorInfo &bias, const Attributes &attributes) noexcept(false);
    InnerProductLayerFWD(const InnerProductLayerFWD &other) noexcept(false);
    InnerProductLayerFWD(InnerProductLayerFWD &&other) noexcept;
    InnerProductLayerFWD& operator=(const InnerProductLayerFWD &other) noexcept(false);
    InnerProductLayerFWD& operator=(InnerProductLayerFWD &&other) noexcept;
    void Run(const void *src, const void *wei, void *dst,  const void *bia = nullptr,
        float alpha = 1.0f, float beta = 0.0f) const noexcept(false);
    void Run(const void *src, const void *wei, void *dst, const void *bia, const PostOpsDataPtrs &postOpsPtrs,
        float alpha = 1.0f, float beta = 0.0f) const noexcept(false);
    ~InnerProductLayerFWD() noexcept;
private:
    std::unique_ptr<Detail::InnerProductLayerFWDImpl> pImpl;
};

class InnerProductLayerBWDData final {
public:
    using SizeType = ::KuDNN::SizeType;
    static Status ValidateInput(const TensorInfo &diffDst, const TensorInfo &weights,
        const TensorInfo &diffSrc) noexcept;
    InnerProductLayerBWDData(const TensorInfo &diffDst, const TensorInfo &weights,
        const TensorInfo &diffSrc) noexcept(false);
    InnerProductLayerBWDData(const InnerProductLayerBWDData &other) noexcept(false);
    InnerProductLayerBWDData(InnerProductLayerBWDData &&other) noexcept;
    InnerProductLayerBWDData& operator=(const InnerProductLayerBWDData &other) noexcept(false);
    InnerProductLayerBWDData& operator=(InnerProductLayerBWDData &&other) noexcept;
    void Run(const void *diffDst, const void *wei, void *diffSrc) const noexcept(false);
    ~InnerProductLayerBWDData() noexcept;
private:
    std::unique_ptr<Detail::InnerProductLayerBWDDataImpl> pImpl;
};

class InnerProductLayerBWDWeights final {
public:
    using SizeType = ::KuDNN::SizeType;
    static Status ValidateInput(const TensorInfo &diffDst, const TensorInfo &src,
        const TensorInfo &diffWeights, const TensorInfo &diffBias) noexcept;
    InnerProductLayerBWDWeights(const TensorInfo &diffDst, const TensorInfo &src,
        const TensorInfo &diffWeights, const TensorInfo &diffBias) noexcept(false);
    InnerProductLayerBWDWeights(const InnerProductLayerBWDWeights &other) noexcept(false);
    InnerProductLayerBWDWeights(InnerProductLayerBWDWeights &&other) noexcept;
    InnerProductLayerBWDWeights& operator=(const InnerProductLayerBWDWeights &other) noexcept(false);
    InnerProductLayerBWDWeights& operator=(InnerProductLayerBWDWeights &&other) noexcept;
    void Run(const void *diffDst, const void *src, void *diffWeights, void *diffBias = nullptr) const noexcept(false);
    ~InnerProductLayerBWDWeights() noexcept;
private:
    std::unique_ptr<Detail::InnerProductLayerBWDWeightsImpl> pImpl;
};

} // KuDNN

#endif // KUDNN_INNER_PRODUCT_HPP
