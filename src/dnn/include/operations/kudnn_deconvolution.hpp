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

#ifndef KUDNN_DECONVOLUTION_HPP
#define KUDNN_DECONVOLUTION_HPP

#include <memory>

#include "types/kudnn_tensor_info.hpp"
#include "service/kudnn_err_codes.hpp"

namespace KuDNN {

enum class DeconvolutionAlgorithm {
    UNIMPLEMENTED,
    DIRECT,
    WINOGRAD
};

namespace Detail {

class DeconvolutionLayerFWDImpl;
class DeconvolutionLayerBWDDataImpl;
class DeconvolutionLayerBWDWeightsImpl;

} // Detail

class DeconvolutionLayerFWD final {
public:
    using SizeType = ::KuDNN::SizeType;
    static Status ValidateInput(const TensorInfo &src, const TensorInfo &weights,
        const TensorInfo &dst, const TensorInfo &bias,
        const Shape &strides, const Shape &dilates,
        const Shape &paddingL, const Shape &paddingR,
        DeconvolutionAlgorithm alg) noexcept;
    DeconvolutionLayerFWD(const TensorInfo &src, const TensorInfo &weights,
        const TensorInfo &dst, const TensorInfo &bias,
        const Shape &strides, const Shape &paddingL, const Shape &paddingR,
        DeconvolutionAlgorithm alg) noexcept(false);
    DeconvolutionLayerFWD(const TensorInfo &src, const TensorInfo &weights,
        const TensorInfo &dst, const TensorInfo &bias,
        const Shape &strides, const Shape &dilates,
        const Shape &paddingL, const Shape &paddingR,
        DeconvolutionAlgorithm alg) noexcept(false);
    DeconvolutionLayerFWD(const DeconvolutionLayerFWD &other) noexcept(false);
    DeconvolutionLayerFWD(DeconvolutionLayerFWD &&other) noexcept;
    DeconvolutionLayerFWD& operator=(const DeconvolutionLayerFWD &other) noexcept(false);
    DeconvolutionLayerFWD& operator=(DeconvolutionLayerFWD &&other) noexcept;
    void Run(const void *src, const void *wei, void *dst, const void *bia) const noexcept(false);
    ~DeconvolutionLayerFWD() noexcept;
private:
    std::unique_ptr<Detail::DeconvolutionLayerFWDImpl> pImpl;
};

class DeconvolutionLayerBWDData final {
public:
    using SizeType = ::KuDNN::SizeType;
    static Status ValidateInput(const TensorInfo &diffDst, const TensorInfo &weights,
        const TensorInfo &diffSrc, const Shape &strides, const Shape &dilates,
        const Shape &paddingL, const Shape &paddingR, DeconvolutionAlgorithm alg) noexcept;
    DeconvolutionLayerBWDData(const TensorInfo &diffDst, const TensorInfo &weights,
        const TensorInfo &diffSrc, const Shape &strides, const Shape &paddingL,
        const Shape &paddingR, DeconvolutionAlgorithm alg) noexcept(false);
    DeconvolutionLayerBWDData(const TensorInfo &diffDst, const TensorInfo &weights,
        const TensorInfo &diffSrc, const Shape &strides, const Shape &dilates,
        const Shape &paddingL, const Shape &paddingR, DeconvolutionAlgorithm alg) noexcept(false);
    DeconvolutionLayerBWDData(const DeconvolutionLayerBWDData &other) noexcept(false);
    DeconvolutionLayerBWDData(DeconvolutionLayerBWDData &&other) noexcept;
    DeconvolutionLayerBWDData& operator=(const DeconvolutionLayerBWDData &other) noexcept(false);
    DeconvolutionLayerBWDData& operator=(DeconvolutionLayerBWDData &&other) noexcept;
    void Run(const void *diffDst, const void *wei, void *diffSrc) const noexcept(false);
    ~DeconvolutionLayerBWDData() noexcept;
private:
    std::unique_ptr<Detail::DeconvolutionLayerBWDDataImpl> pImpl;
};

class DeconvolutionLayerBWDWeights final {
public:
    using SizeType = ::KuDNN::SizeType;
    static Status ValidateInput(const TensorInfo &diffDst, const TensorInfo &src,
        const TensorInfo &diffWeights, const TensorInfo &diffBias,
        const Shape &strides, const Shape &dilates,
        const Shape &paddingL, const Shape &paddingR,
        DeconvolutionAlgorithm alg) noexcept;
    DeconvolutionLayerBWDWeights(const TensorInfo &diffDst, const TensorInfo &src,
        const TensorInfo &diffWeights, const TensorInfo &diffBias,
        const Shape &strides, const Shape &paddingL, const Shape &paddingR,
        DeconvolutionAlgorithm alg) noexcept(false);
    DeconvolutionLayerBWDWeights(const TensorInfo &diffDst, const TensorInfo &src,
        const TensorInfo &diffWeights, const TensorInfo &diffBias,
        const Shape &strides, const Shape &dilates,
        const Shape &paddingL, const Shape &paddingR,
        DeconvolutionAlgorithm alg) noexcept(false);
    DeconvolutionLayerBWDWeights(const DeconvolutionLayerBWDWeights &other) noexcept(false);
    DeconvolutionLayerBWDWeights(DeconvolutionLayerBWDWeights &&other) noexcept;
    DeconvolutionLayerBWDWeights& operator=(const DeconvolutionLayerBWDWeights &other) noexcept(false);
    DeconvolutionLayerBWDWeights& operator=(DeconvolutionLayerBWDWeights &&other) noexcept;
    void Run(const void *diffDst, const void *src, void *diffWei, void *diffBias) const noexcept(false);
    ~DeconvolutionLayerBWDWeights() noexcept;
private:
    std::unique_ptr<Detail::DeconvolutionLayerBWDWeightsImpl> pImpl;
};

} // KuDNN

#endif // KUDNN_DECONVOLUTION_HPP
