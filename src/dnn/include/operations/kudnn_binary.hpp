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

#ifndef KUDNN_BINARY_HPP
#define KUDNN_BINARY_HPP

#include <memory>

#include "types/kudnn_tensor_info.hpp"
#include "service/kudnn_err_codes.hpp"
#include "types/kudnn_binary_kinds.hpp"

namespace KuDNN {

namespace Detail {

class BinaryLayerImpl;

} // Detail

class BinaryLayer final {
public:
    using SizeType = ::KuDNN::SizeType;
    static Status ValidateInput(const TensorInfo &src0Info, const TensorInfo &src1Info,
        const TensorInfo &dstInfo, BinaryFunction op) noexcept;
    BinaryLayer(const TensorInfo &src0Info, const TensorInfo &src1Info,
        const TensorInfo &dstInfo, BinaryFunction op) noexcept(false);
    BinaryLayer(const BinaryLayer &other) noexcept(false);
    BinaryLayer(BinaryLayer &&other) noexcept;
    BinaryLayer& operator=(const BinaryLayer &other) noexcept(false);
    BinaryLayer& operator=(BinaryLayer &&other) noexcept;
    void Run(const void *src0, const void *src1, void *dst,
        const float scale0 = 1.0f, const float scale1 = 1.0f) const noexcept(false);
    ~BinaryLayer() noexcept;
private:
    std::unique_ptr<Detail::BinaryLayerImpl> pImpl;
};

} // KuDNN

#endif // KUDNN_BINARY_HPP
