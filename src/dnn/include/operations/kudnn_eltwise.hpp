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

#ifndef KUDNN_ELTWISE_HPP
#define KUDNN_ELTWISE_HPP

#include <memory>

#include "types/kudnn_tensor_info.hpp"
#include "service/kudnn_err_codes.hpp"

namespace KuDNN {

// supported functions
enum class ActivationFunction {
    UNIMPLEMENTED,
    ABS,
    EXP,
    LINEAR,
    LOG,
    RELU,
    ROUND,
    SQRT,
    SQUARE,
    TANH,
    SIGMOID,
    POW,
    CLIP,
    CLIP_V2,
    HARDSIGMOID,
    HARDSWISH,
    EXP_DST,
    LOGISTIC_DST,
    TANH_DST,
    ELU,
    GELU_TANH,
    CLIP_V2_DST,
    ELU_DST,
    RELU_DST,
    SQRT_DST,
    SWISH,
    SRELU,
    GELU_ERF,
    MISH
};

namespace Detail {

class ActivationLayerImplFWD;
class ActivationLayerImplBWD;

} // namespace Detail

class KUDNN_API_PUBLIC ActivationLayerFWD final {
public:
    using SizeType = ::KuDNN::SizeType;
    static Status ValidateInput(const TensorInfo &srcInfo, const TensorInfo &dstInfo, ActivationFunction kind,
                                float alpha = 0.0f, float beta = 0.0f) noexcept(false);
    ActivationLayerFWD(const TensorInfo &srcInfo, const TensorInfo &dstInfo, ActivationFunction kind,
                       float alpha = 0.0f, float beta = 0.0f) noexcept(false);
    ActivationLayerFWD(const ActivationLayerFWD &other) noexcept(false);
    ActivationLayerFWD(ActivationLayerFWD &&other) noexcept;
    ActivationLayerFWD &operator=(const ActivationLayerFWD &other) noexcept(false);
    ActivationLayerFWD &operator=(ActivationLayerFWD &&other) noexcept;
    void Run(const void *src, void *dst) const noexcept(false);
    ~ActivationLayerFWD() noexcept;

private:
    std::unique_ptr<Detail::ActivationLayerImplFWD> pImpl;
};

class KUDNN_API_PUBLIC ActivationLayerBWD final {
public:
    using SizeType = ::KuDNN::SizeType;
    static Status ValidateInput(const TensorInfo &dsInfo, const TensorInfo &ddInfo, const TensorInfo &srcInfo,
                                ActivationFunction kind, float alpha = 0.0f, float beta = 0.0f) noexcept;
    ActivationLayerBWD(const TensorInfo &dsInfo, const TensorInfo &ddInfo, const TensorInfo &srcInfo,
                       ActivationFunction kind, float alpha = 0.0f, float beta = 0.0f) noexcept(false);
    ActivationLayerBWD(const ActivationLayerBWD &other) noexcept(false);
    ActivationLayerBWD(ActivationLayerBWD &&other) noexcept;
    ActivationLayerBWD &operator=(const ActivationLayerBWD &other) noexcept(false);
    ActivationLayerBWD &operator=(ActivationLayerBWD &&other) noexcept;
    void Run(void *ds, const void *dd, const void *src) const noexcept(false);
    ~ActivationLayerBWD() noexcept;

private:
    std::unique_ptr<Detail::ActivationLayerImplBWD> pImpl;
};

} // namespace KuDNN

#endif // KUDNN_ELTWISE_HPP
