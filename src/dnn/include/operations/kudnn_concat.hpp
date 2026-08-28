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

#ifndef KUDNN_CONCAT_HPP
#define KUDNN_CONCAT_HPP

#include <memory>
#include <vector>

#include "types/kudnn_tensor_info.hpp"
#include "service/kudnn_err_codes.hpp"

namespace KuDNN {

namespace Detail {

class ConcatLayerImpl;

} // Detail

class ConcatLayer final {
public:
    using SizeType = ::KuDNN::SizeType;
    static Status ValidateInput(const std::vector<TensorInfo> &srcInfo,
        const int concatDim, const TensorInfo &dstInfo) noexcept;
    ConcatLayer(const std::vector<TensorInfo> &srcInfo, const int concatDim,
        const TensorInfo &dstInfo) noexcept(false);
    ConcatLayer(const ConcatLayer &other) noexcept(false);
    ConcatLayer(ConcatLayer &&other) noexcept;
    ConcatLayer& operator=(const ConcatLayer &other) noexcept(false);
    ConcatLayer& operator=(ConcatLayer &&other) noexcept;
    void Run(const void **src, void *dst) const noexcept(false);
    ~ConcatLayer() noexcept;
private:
    std::unique_ptr<Detail::ConcatLayerImpl> pImpl;
};

} // KuDNN

#endif // KUDNN_CONCAT_HPP
