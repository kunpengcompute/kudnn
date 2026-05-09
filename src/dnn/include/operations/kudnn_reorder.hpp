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

#ifndef KUDNN_REORDER_HPP
#define KUDNN_REORDER_HPP

#include <memory>

#include "types/kudnn_tensor_info.hpp"
#include "service/kudnn_err_codes.hpp"

namespace KuDNN {
namespace Detail {

class ReorderLayerImpl;

} // namespace Detail

class KUDNN_API_PUBLIC ReorderLayer final {
public:
    using SizeType = ::KuDNN::SizeType;
    static Status ValidateInput(const TensorInfo &srcInfo, const TensorInfo &dstInfo) noexcept;
    ReorderLayer(const TensorInfo &srcInfo, const TensorInfo &dstInfo) noexcept(false);
    ReorderLayer(const ReorderLayer &other) noexcept(false);
    ReorderLayer(ReorderLayer &&other) noexcept;
    ReorderLayer &operator=(const ReorderLayer &other) noexcept(false);
    ReorderLayer &operator=(ReorderLayer &&other) noexcept;
    void Run(const void *src, void *dst) const noexcept(false);
    ~ReorderLayer() noexcept;

private:
    std::unique_ptr<Detail::ReorderLayerImpl> pImpl;
};

} // namespace KuDNN

#endif // KUDNN_REORDER_HPP
