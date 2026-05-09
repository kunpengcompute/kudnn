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

#ifndef KUDNN_LINEAR_RES_H
#define KUDNN_LINEAR_RES_H

#include <memory>

#include "types/kudnn_tensor_info.hpp"
#include "service/kudnn_err_codes.hpp"
#include "operations/kudnn_gemm.hpp"

namespace KuDNN {

// supported functions
enum class ResOpsFunction {
    RES_EXT,
    RES_MUL,
    RES_IDENTIAL
};

namespace Detail {

class LinearResFWDImpl;

} // namespace Detail

class KUDNN_API_PUBLIC LinearResFWD final {
public:
    using SizeType = ::KuDNN::SizeType;
    static Status ValidateInput(const TensorInfo &aInfo, const TensorInfo &bInfo, const TensorInfo &cInfo,
                                const TensorInfo &biasInfo, const TensorInfo &resInfo, const TensorInfo &linearResInfo,
                                float alpha = 1.0f, float beta = 0.0f, float gamma = 1.0f,
                                ResOpsFunction kind = ResOpsFunction::RES_IDENTIAL, int numThreads = 0) noexcept(false);

    static Status ValidateInput(const TensorInfo &aInfo, const TensorInfo &bInfo, const TensorInfo &cInfo,
                                const TensorInfo &res, const TensorInfo &linearRes, float alpha = 1.0f,
                                float beta = 0.0f, float gamma = 1.0f,
                                ResOpsFunction kind = ResOpsFunction::RES_IDENTIAL, int numThreads = 0) noexcept(false);

    LinearResFWD(const TensorInfo &aInfo, const TensorInfo &bInfo, const TensorInfo &cInfo, const TensorInfo &biasInfo,
                 const TensorInfo &res, const TensorInfo &linearRes, float alpha = 1.0f, float beta = 0.0f,
                 float gamma = 1.0f, ResOpsFunction kind = ResOpsFunction::RES_IDENTIAL,
                 int numThreads = 0) noexcept(false);

    LinearResFWD(const TensorInfo &aInfo, const TensorInfo &bInfo, const TensorInfo &cInfo, const TensorInfo &res,
                 const TensorInfo &linearRes, float alpha = 1.0f, float beta = 0.0f, float gamma = 1.0f,
                 ResOpsFunction kind = ResOpsFunction::RES_IDENTIAL, int numThreads = 0) noexcept(false);

    LinearResFWD(const LinearResFWD &other) noexcept(false);

    LinearResFWD(LinearResFWD &&other) noexcept;

    LinearResFWD &operator=(const LinearResFWD &other) noexcept(false);

    LinearResFWD &operator=(LinearResFWD &&other) noexcept;

    void Run(const void *a, const void *b, void *c, const void *bias, const void *res, void *linearRes,
             float gamma = 1.0f, int numThreads = 0) const noexcept(false);

    void Run(const void *a, const void *b, void *c, const void *res, void *linearRes, float gamma = 1.0f,
             int numThreads = 0) const noexcept(false);

    ~LinearResFWD() noexcept;

private:
    std::unique_ptr<Detail::LinearResFWDImpl> pImpl;
};

} // namespace KuDNN

#endif // KUDNN_LINEAR_RES_H
