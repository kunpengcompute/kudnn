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

#ifndef KUDNN_RESAMPLING_HPP
#define KUDNN_RESAMPLING_HPP

#include <memory>

#include "types/kudnn_tensor_info.hpp"
#include "service/kudnn_err_codes.hpp"

namespace KuDNN {

enum class ResamplingAlg {
    UNIMPLEMENTED,
    NEAREST_NEIGHBOR,
    LINEAR,
};

namespace Detail {
namespace Resampling {
    class ResamplingLayerImplFWD;
    class ResamplingLayerImplBWD;
} // Resampling
} // Detail

class ResamplingLayerFWD final {
public:
    using SizeType = ::KuDNN::SizeType;
    static Status ValidateInput(const TensorInfo &srcInfo, const TensorInfo &dstInfo,
        ResamplingAlg algKind) noexcept;
    ResamplingLayerFWD(const TensorInfo &srcInfo, const TensorInfo &dstInfo,
        ResamplingAlg algKind) noexcept(false);
    ResamplingLayerFWD(const ResamplingLayerFWD &other) noexcept(false);
    ResamplingLayerFWD(ResamplingLayerFWD &&other) noexcept;
    ResamplingLayerFWD& operator=(const ResamplingLayerFWD &other) noexcept(false);
    ResamplingLayerFWD& operator=(ResamplingLayerFWD &&other) noexcept;
    void Run(const void *src, void *dst) const noexcept(false);
    ~ResamplingLayerFWD() noexcept;
private:
    std::unique_ptr<Detail::Resampling::ResamplingLayerImplFWD> pImpl;
};

class ResamplingLayerBWD final {
public:
    using SizeType = ::KuDNN::SizeType;
    ResamplingLayerBWD(const TensorInfo &diffDstInfo, const TensorInfo &diffSrcInfo,
        ResamplingAlg algKind) noexcept(false);

    ResamplingLayerBWD(const ResamplingLayerBWD &other) noexcept(false);
    ResamplingLayerBWD(ResamplingLayerBWD &&other) noexcept;
    ResamplingLayerBWD& operator=(const ResamplingLayerBWD &other) noexcept(false);
    ResamplingLayerBWD& operator=(ResamplingLayerBWD &&other) noexcept;

    ~ResamplingLayerBWD() noexcept;

    static Status ValidateInput(const TensorInfo &diffDstInfo, const TensorInfo &diffSrcInfo,
        ResamplingAlg algKind) noexcept;

    void Run(const void *diffDst, void *diffSrc) const noexcept(false);

private:
    std::unique_ptr<Detail::Resampling::ResamplingLayerImplBWD> pImpl;
};

} // KuDNN

#endif
