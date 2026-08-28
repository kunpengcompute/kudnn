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

#ifndef KUDNN_PRELU_HPP
#define KUDNN_PRELU_HPP

#include <memory>

#include "types/kudnn_tensor_info.hpp"
#include "service/kudnn_err_codes.hpp"

namespace KuDNN {

namespace Detail {

class PReLULayerFWDImpl;
class PReLULayerBWDImpl;

} // Detail

class PReLULayerFWD final {
public:
    using SizeType = ::KuDNN::SizeType;
    static Status ValidateInput(const TensorInfo &srcInfo, const TensorInfo &weightsInfo,
        const TensorInfo &dstInfo) noexcept;
    PReLULayerFWD(const TensorInfo &srcInfo, const TensorInfo &weightsInfo,
        const TensorInfo &dstInfo) noexcept(false);
    PReLULayerFWD(const PReLULayerFWD &other) noexcept(false);
    PReLULayerFWD(PReLULayerFWD &&other) noexcept;
    PReLULayerFWD& operator=(const PReLULayerFWD &other) noexcept(false);
    PReLULayerFWD& operator=(PReLULayerFWD &&other) noexcept;
    void Run(const void *src, const void *wei, void *dst) const noexcept(false);
    ~PReLULayerFWD() noexcept;
private:
    std::unique_ptr<Detail::PReLULayerFWDImpl> pImpl;
};

class PReLULayerBWD final {
public:
    using SizeType = ::KuDNN::SizeType;
    static Status ValidateInput(const TensorInfo &srcInfo, const TensorInfo &diffSrcInfo,
        const TensorInfo &weightsInfo, const TensorInfo &diffWeightsInfo,
        const TensorInfo &diffDstInfo) noexcept;
    PReLULayerBWD(const TensorInfo &srcInfo, const TensorInfo &diffSrcInfo,
        const TensorInfo &weightsInfo, const TensorInfo &diffWeightsInfo,
        const TensorInfo &diffDstInfo) noexcept(false);
    PReLULayerBWD(const PReLULayerBWD &other) noexcept(false);
    PReLULayerBWD(PReLULayerBWD &&other) noexcept;
    PReLULayerBWD& operator=(const PReLULayerBWD &other) noexcept(false);
    PReLULayerBWD& operator=(PReLULayerBWD &&other) noexcept;
    void Run(const void *src, void *diffSrc, const void *wei,
        void *diffWeights, const void *diffDst) const noexcept(false);
    ~PReLULayerBWD() noexcept;
private:
    std::unique_ptr<Detail::PReLULayerBWDImpl> pImpl;
};

} // KuDNN

#endif // KUDNN_PRELU_HPP
