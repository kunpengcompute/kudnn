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

#ifndef KUDNN_LRN_HPP
#define KUDNN_LRN_HPP

#include <memory>

#include "types/kudnn_tensor_info.hpp"
#include "service/kudnn_err_codes.hpp"

namespace KuDNN {

enum class LRNormAlgorithmKind : std::uint32_t {
    WITHIN         = 0x0U,
    ACROSS         = 0x1U,
};

namespace Detail {

class LocalResponseNormalizationLayerFWDImpl;
class LocalResponseNormalizationLayerBWDImpl;

} // Detail

class LocalResponseNormalizationLayerFWD final {
public:
    using SizeType = ::KuDNN::SizeType;
    static Status ValidateInput(const TensorInfo &srcInfo, const TensorInfo &dstInfo, float alpha, float beta, float k,
        SizeType localSize, LRNormAlgorithmKind algorithm) noexcept;
    LocalResponseNormalizationLayerFWD(const TensorInfo &srcInfo, const TensorInfo &dstInfo, float alpha, float beta,
        float k, SizeType localSize, LRNormAlgorithmKind algorithm) noexcept(false);
    LocalResponseNormalizationLayerFWD(const LocalResponseNormalizationLayerFWD &other) noexcept(false);
    LocalResponseNormalizationLayerFWD(LocalResponseNormalizationLayerFWD &&other) noexcept;
    LocalResponseNormalizationLayerFWD& operator=(const LocalResponseNormalizationLayerFWD &other) noexcept(false);
    LocalResponseNormalizationLayerFWD& operator=(LocalResponseNormalizationLayerFWD &&other) noexcept;
    void Run(const void *src, void *dst) const noexcept(false);
    ~LocalResponseNormalizationLayerFWD() noexcept;
private:
    std::unique_ptr<Detail::LocalResponseNormalizationLayerFWDImpl> pImpl;
};

class LocalResponseNormalizationLayerBWD final {
public:
    using SizeType = ::KuDNN::SizeType;
    static Status ValidateInput(const TensorInfo &srcInfo,
        const TensorInfo &diffDstInfo, const TensorInfo &diffSrcInfo,
        float alpha, float beta, float k,
        SizeType localSize, LRNormAlgorithmKind algorithm) noexcept;
    LocalResponseNormalizationLayerBWD(const TensorInfo &srcInfo,
        const TensorInfo &diffDstInfo, const TensorInfo &diffSrcInfo,
        float alpha, float beta, float k,
        SizeType localSize, LRNormAlgorithmKind algorithm) noexcept(false);
    LocalResponseNormalizationLayerBWD(const LocalResponseNormalizationLayerBWD &other) noexcept(false);
    LocalResponseNormalizationLayerBWD(LocalResponseNormalizationLayerBWD &&other) noexcept;
    LocalResponseNormalizationLayerBWD& operator=(const LocalResponseNormalizationLayerBWD &other) noexcept(false);
    LocalResponseNormalizationLayerBWD& operator=(LocalResponseNormalizationLayerBWD &&other) noexcept;
    void Run(const void *src, const void *diffDst,
            void *diffSrc) const noexcept(false);
    ~LocalResponseNormalizationLayerBWD() noexcept;
private:
    std::unique_ptr<Detail::LocalResponseNormalizationLayerBWDImpl> pImpl;
};

} // KuDNN

#endif // KUDNN_LRN_HPP
