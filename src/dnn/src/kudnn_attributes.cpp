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

#include <cstddef>
#include <memory>
#include <utility>

#include "kudnn.hpp"

namespace KuDNN {
namespace Detail {

class AttributesImpl final {
public:
    AttributesImpl() noexcept(true) {}
    PostOps GetPostOps() const
    {
        return postOps;
    }
    void SetPostOps(const PostOps &po)
    {
        postOps = po;
    }
    PostOps postOps;
    friend bool operator==(const AttributesImpl &lhs, const AttributesImpl &rhs) noexcept;
};

inline bool operator==(const AttributesImpl &lhs, const AttributesImpl &rhs) noexcept
{
    return lhs.postOps == rhs.postOps;
}

} // Detail

Attributes::~Attributes() noexcept {}

// all public methods just call corresponding pImpl implemnentation
Attributes::Attributes() noexcept(false)
    : pImpl(new Detail::AttributesImpl{}) {}

Attributes::Attributes(const Attributes &other) noexcept(false)
    : pImpl(new Detail::AttributesImpl{*(other.pImpl.get())}) {}

Attributes::Attributes(Attributes &&other) noexcept
    : pImpl(std::move(other.pImpl)) {}

Attributes& Attributes::operator=(const Attributes &other) noexcept(false)
{
    if (this != &other) {
        this->pImpl.reset(new Detail::AttributesImpl{*(other.pImpl.get())});
    }
    return *this;
}

Attributes& Attributes::operator=(Attributes &&other) noexcept
{
    if (this != &other) {
        this->pImpl = std::move(other.pImpl);
    }
    return *this;
}

PostOps Attributes::GetPostOps() const noexcept(false)
{
    return pImpl->GetPostOps();
}

void Attributes::SetPostOps(const PostOps &po) noexcept(false)
{
    return pImpl->SetPostOps(po);
}

bool operator==(const Attributes &lhs, const Attributes &rhs) noexcept
{
    return (*(lhs.pImpl.get())) == (*(rhs.pImpl.get()));
}

} // KuDNN
