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

#ifndef KUDNN_ATTRIBUTES_HPP
#define KUDNN_ATTRIBUTES_HPP

#include <memory>

#include "types/kudnn_tensor_info.hpp"
#include "operations/kudnn_post_ops.hpp"
#include "service/kudnn_err_codes.hpp"
namespace KuDNN {

namespace Detail {

class AttributesImpl;

} // Detail

class Attributes final {
public:
    using SizeType = ::KuDNN::SizeType;
    Attributes() noexcept(false);
    Attributes(const Attributes &other) noexcept(false);
    Attributes(Attributes &&other) noexcept;
    Attributes& operator=(const Attributes &other) noexcept(false);
    Attributes& operator=(Attributes &&other) noexcept;
    ~Attributes() noexcept;
    PostOps GetPostOps() const noexcept(false);
    void SetPostOps(const PostOps &po) noexcept(false);
private:
    std::unique_ptr<Detail::AttributesImpl> pImpl;
    friend bool operator==(const Attributes &lhs, const Attributes &rhs) noexcept;
};

bool operator==(const Attributes &lhs, const Attributes &rhs) noexcept;

} // KuDNN

#endif // KUDNN_ATTRIBUTES_HPP
