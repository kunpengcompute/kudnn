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

#ifndef KUDNN_SHUFFLE_HPP
#define KUDNN_SHUFFLE_HPP

#include <memory>

#include "types/kudnn_tensor_info.hpp"
#include "service/kudnn_err_codes.hpp"

namespace KuDNN {

namespace Detail {

class ShuffleLayerImpl;

} // Detail

class ShuffleLayer final {
public:
    using SizeType = ::KuDNN::SizeType;
    static Status ValidateInput(const TensorInfo &srcInfo, const TensorInfo &dstInfo, SizeType axis,
        SizeType groupSize) noexcept;
    ShuffleLayer(const TensorInfo &srcInfo, const TensorInfo &dstInfo, SizeType axis,
        SizeType group) noexcept(false);
    ShuffleLayer(const ShuffleLayer &other) noexcept(false);
    ShuffleLayer(ShuffleLayer &&other) noexcept;
    ShuffleLayer& operator=(const ShuffleLayer &other) noexcept(false);
    ShuffleLayer& operator=(ShuffleLayer &&other) noexcept;
    void Run(const void *src, void *dst) const noexcept(false);
    ~ShuffleLayer() noexcept;
private:
    std::unique_ptr<Detail::ShuffleLayerImpl> pImpl;
};

} // KuDNN

#endif // KUDNN_SHUFFLE_HPP
