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

#ifndef KUDNN_LINEAR_ACTIVATIONLAYER_HPP
#define KUDNN_LINEAR_ACTIVATIONLAYER_HPP

#include <memory>

#include "types/kudnn_tensor_info.hpp"
#include "service/kudnn_err_codes.hpp"
#include "operations/kudnn_eltwise.hpp"
#include "operations/kudnn_gemm.hpp"

namespace KuDNN {

namespace Detail {

class LinearActivationLayerFWDImpl;

} // namespace Detail

class KUDNN_API_PUBLIC LinearActivationLayerFWD final {
public:
    using SizeType = ::KuDNN::SizeType;
    static Status ValidateInput(const TensorInfo &aInfo, const TensorInfo &bInfo, const TensorInfo &cInfo,
                                const TensorInfo &biasInfo, float alpha = 1.0f, float beta = 0.0f,
                                ActivationFunction kind = KuDNN::ActivationFunction::RELU,
                                int numThreads = 0) noexcept(false);

    static Status ValidateInput(const TensorInfo &aInfo, const TensorInfo &bInfo, const TensorInfo &cInfo,
                                float alpha = 1.0f, float beta = 0.0f,
                                ActivationFunction kind = KuDNN::ActivationFunction::RELU,
                                int numThreads = 0) noexcept(false);

    LinearActivationLayerFWD(const TensorInfo &aInfo, const TensorInfo &bInfo, const TensorInfo &cInfo,
                             const TensorInfo &biasInfo, float alpha = 1.0f, float beta = 0.0f,
                             ActivationFunction kind = KuDNN::ActivationFunction::RELU,
                             int numThreads = 0) noexcept(false);

    LinearActivationLayerFWD(const TensorInfo &aInfo, const TensorInfo &bInfo, const TensorInfo &cInfo,
                             float alpha = 1.0f, float beta = 0.0f,
                             ActivationFunction kind = KuDNN::ActivationFunction::RELU,
                             int numThreads = 0) noexcept(false);
    LinearActivationLayerFWD(const LinearActivationLayerFWD &other) noexcept(false);
    LinearActivationLayerFWD(LinearActivationLayerFWD &&other) noexcept;
    LinearActivationLayerFWD &operator=(const LinearActivationLayerFWD &other) noexcept(false);
    LinearActivationLayerFWD &operator=(LinearActivationLayerFWD &&other) noexcept;
    void Run(const void *a, const void *b, void *c, const void *bias, int numThreads = 0) const noexcept(false);
    void Run(const void *a, const void *b, void *c, int numThreads = 0) const noexcept(false);
    ~LinearActivationLayerFWD() noexcept;

private:
    std::unique_ptr<Detail::LinearActivationLayerFWDImpl> pImpl;
};

} // namespace KuDNN

#endif // KUDNN_LINEAR_ACTIVATIONLAYER_HPP
