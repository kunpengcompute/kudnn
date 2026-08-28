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

#ifndef KUDNN_CONVOLUTION_HPP
#define KUDNN_CONVOLUTION_HPP

#include <memory>
#include "types/kudnn_tensor_info.hpp"
#include "service/kudnn_err_codes.hpp"

namespace KuDNN {

enum class ConvolutionAlgorithm {
    UNIMPLEMENTED,
    AUTO,
    DIRECT,
    WINOGRAD
};

namespace Detail {
class ConvolutionFWDImplBase;
class ConvolutionLayerBWDDataImpl;
class ConvolutionLayerBWDWeightsImpl;
} // Detail

class ConvolutionLayerFWD final {
public:
    static Status ValidateInput(const TensorInfo &src, const TensorInfo &weights,
        const TensorInfo &dst, const TensorInfo &bias, const Shape &strides,
        const Shape &dilates, const Shape &paddingL, int kCpuIsa,
        bool isLayoutBlock, bool &isJitConv, bool bUseWino) noexcept;
    static Status ValidateInput(const TensorInfo &src, const TensorInfo &weights,
        const TensorInfo &dst, const TensorInfo &bias, const Shape &strides,
        const Shape &dilates, const Shape &paddingL, int kCpuIsa,
        bool isLayoutBlock, bool &isJitConv, bool bUseWino,
        int &brgBlock, int nthreads) noexcept;
    explicit ConvolutionLayerFWD(int isa, const TensorInfo &src, const TensorInfo &weights,
        const TensorInfo &dst, const TensorInfo &bias, KuDNN::Shape strides, KuDNN::Shape paddingL,
        KuDNN::Shape dilates, KuDNN::Propagation prop, int nthreads, bool isLayoutBlock,
        bool bUseWino) noexcept(false);
    ConvolutionLayerFWD(const TensorInfo &src, const TensorInfo &weights,
        const TensorInfo &dst, const TensorInfo &bias,
        const Shape &strides, const Shape &paddingL, const Shape &paddingR,
        ConvolutionAlgorithm alg) noexcept(false);
    ConvolutionLayerFWD(const TensorInfo &src, const TensorInfo &weights,
        const TensorInfo &dst, const TensorInfo &bias,
        const Shape &strides, const Shape &dilates,
        const Shape &paddingL, const Shape &paddingR,
        ConvolutionAlgorithm alg) noexcept(false);
    void Run(const void *src, const void *wei,
        void *dst, const void *bias) noexcept(false);
    ConvolutionLayerFWD(const ConvolutionLayerFWD &other) noexcept(false);
    ConvolutionLayerFWD(ConvolutionLayerFWD &&other) noexcept;
    ConvolutionLayerFWD& operator=(const ConvolutionLayerFWD &other) noexcept(false);
    ConvolutionLayerFWD& operator=(ConvolutionLayerFWD &&other) noexcept;
    ~ConvolutionLayerFWD() noexcept;
private:
    std::unique_ptr<Detail::ConvolutionFWDImplBase> pImpl;
};

class ConvolutionLayerBWDData final {
public:
    using SizeType = ::KuDNN::SizeType;
    static Status ValidateInput(const TensorInfo &diffDst, const TensorInfo &weights,
        const TensorInfo &diffSrc,
        const Shape &strides, const Shape &dilates,
        const Shape &paddingL, const Shape &paddingR,
        ConvolutionAlgorithm alg) noexcept;
    ConvolutionLayerBWDData(const TensorInfo &diffDst, const TensorInfo &weights,
        const TensorInfo &diffSrc,
        const Shape &strides, const Shape &paddingL, const Shape &paddingR,
        ConvolutionAlgorithm alg) noexcept(false);
    ConvolutionLayerBWDData(const TensorInfo &diffDst, const TensorInfo &weights,
        const TensorInfo &diffSrc,
        const Shape &strides, const Shape &dilates,
        const Shape &paddingL, const Shape &paddingR,
        ConvolutionAlgorithm alg) noexcept(false);
    ConvolutionLayerBWDData(const ConvolutionLayerBWDData &other) noexcept(false);
    ConvolutionLayerBWDData(ConvolutionLayerBWDData &&other) noexcept;
    ConvolutionLayerBWDData& operator=(const ConvolutionLayerBWDData &other) noexcept(false);
    ConvolutionLayerBWDData& operator=(ConvolutionLayerBWDData &&other) noexcept;
    void Run(const void *diffDst, const void *wei, void *diffSrc) const noexcept(false);
    ~ConvolutionLayerBWDData() noexcept;
private:
    std::unique_ptr<Detail::ConvolutionLayerBWDDataImpl> pImpl;
};

class ConvolutionLayerBWDWeights final {
public:
    using SizeType = ::KuDNN::SizeType;
    static Status ValidateInput(const TensorInfo &diffDst, const TensorInfo &src,
        const TensorInfo &diffWeights, const TensorInfo &diffBias,
        const Shape &strides, const Shape &dilates,
        const Shape &paddingL, const Shape &paddingR,
        ConvolutionAlgorithm alg) noexcept;
    ConvolutionLayerBWDWeights(const TensorInfo &diffDst, const TensorInfo &src,
        const TensorInfo &diffWeights, const TensorInfo &diffBias,
        const Shape &strides, const Shape &paddingL, const Shape &paddingR,
        ConvolutionAlgorithm alg) noexcept(false);
    ConvolutionLayerBWDWeights(const TensorInfo &diffDst, const TensorInfo &src,
        const TensorInfo &diffWeights, const TensorInfo &diffBias,
        const Shape &strides, const Shape &dilates,
        const Shape &paddingL, const Shape &paddingR,
        ConvolutionAlgorithm alg) noexcept(false);
    ConvolutionLayerBWDWeights(const ConvolutionLayerBWDWeights &other) noexcept(false);
    ConvolutionLayerBWDWeights(ConvolutionLayerBWDWeights &&other) noexcept;
    ConvolutionLayerBWDWeights& operator=(const ConvolutionLayerBWDWeights &other) noexcept(false);
    ConvolutionLayerBWDWeights& operator=(ConvolutionLayerBWDWeights &&other) noexcept;
    void Run(const void *diffDst, const void *src, void *diffWei, void *diffBias) const noexcept(false);
    ~ConvolutionLayerBWDWeights() noexcept;
private:
    std::unique_ptr<Detail::ConvolutionLayerBWDWeightsImpl> pImpl;
};
} // KuDNN

#endif // KUDNN_CONVOLUTION_HPP
