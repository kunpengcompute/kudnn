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

#ifndef KUDNN_GEMM_HPP
#define KUDNN_GEMM_HPP

#include <memory>

#include "types/kudnn_tensor_info.hpp"
#include "service/kudnn_err_codes.hpp"

#include "operations/kudnn_attributes.hpp"
#include "operations/kudnn_post_ops.hpp"

namespace KuDNN {

namespace Detail {

class GemmImpl;

} // Detail

class Gemm final {
public:
    using SizeType = ::KuDNN::SizeType;
    static Status ValidateInput(const TensorInfo &aInfo, const TensorInfo &bInfo,
        const TensorInfo &cInfo, const TensorInfo &biasInfo) noexcept;
    static Status ValidateInput(const TensorInfo &aInfo, const TensorInfo &bInfo,
        const TensorInfo &cInfo) noexcept;
    static Status ValidateInput(const TensorInfo &aInfo, const TensorInfo &bInfo,
        const TensorInfo &cInfo, const TensorInfo &biasInfo, const Attributes &attributes) noexcept;
    static Status ValidateInput(const TensorInfo &aInfo, const TensorInfo &bInfo,
            const TensorInfo &cInfo, const Attributes &attributes) noexcept;
    Gemm(const TensorInfo &aInfo, const TensorInfo &bInfo, const TensorInfo &cInfo,
        const TensorInfo &biasInfo) noexcept(false);
    Gemm(const TensorInfo &aInfo, const TensorInfo &bInfo, const TensorInfo &cInfo) noexcept(false);
    Gemm(const TensorInfo &aInfo, const TensorInfo &bInfo, const TensorInfo &cInfo,
        const TensorInfo &biasInfo, const Attributes &attributes) noexcept(false);
    Gemm(const TensorInfo &aInfo, const TensorInfo &bInfo, const TensorInfo &cInfo,
        const Attributes &attributes) noexcept(false);
    Gemm(const Gemm &other) noexcept(false);
    Gemm(Gemm &&other) noexcept;
    Gemm& operator=(const Gemm &other) noexcept(false);
    Gemm& operator=(Gemm &&other) noexcept;
    void Run(const void *a, const void *b, void *c, const void *bias,
        float alpha = 1.0f, float beta = 0.0f) const noexcept(false);
    void Run(const void *a, const void *b, void *c, const void *bias, const PostOpsDataPtrs &postOpsPtrs,
        float alpha = 1.0f, float beta = 0.0f) const noexcept(false);
    void Run(const void *a, const void *b, void *c,
        float alpha = 1.0f, float beta = 0.0f) const noexcept(false);
    void Run(const void *a, const void *b, void *c, const PostOpsDataPtrs &postOpsPtrs,
        float alpha = 1.0f, float beta = 0.0f) const noexcept(false);
    ~Gemm() noexcept;
private:
    std::unique_ptr<Detail::GemmImpl> pImpl;
};

} // KuDNN

#endif // KUDNN_GEMM_HPP
