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

#ifndef KUDNN_POST_OPS_HPP
#define KUDNN_POST_OPS_HPP

#include <memory>
#include <vector>

#include "types/kudnn_tensor_info.hpp"
#include "service/kudnn_err_codes.hpp"
#include "types/kudnn_eltwise_kinds.hpp"
#include "types/kudnn_binary_kinds.hpp"
#include "types/kudnn_primitive_kind.hpp"

namespace KuDNN {

namespace Detail {

class PostOpsImpl;

} // Detail

using PostOpsDataPtrs = std::vector<const void *>;

enum class PReLUMask {
    COMMON     = 0,
    PER_DIM_0  = (1 << 0),
    PER_DIM_1  = (1 << 1),
    PER_DIM_01 = (1 << 0) + (1 << 1),
    PER_DIM_2  = (1 << 2),
    PER_DIM_3  = (1 << 3),
    PER_DIM_4  = (1 << 4),
    PER_TENSOR = (1 << MAX_DIMS) - 1,
    PER_OC     = PER_DIM_1,
    PER_OCIC   = PER_DIM_01
};

class PostOps final {
public:
    using SizeType = ::KuDNN::SizeType;
    PostOps() noexcept(false);
    PostOps(const PostOps &other) noexcept(false);
    PostOps(PostOps &&other) noexcept;
    PostOps& operator=(const PostOps &other) noexcept(false);
    PostOps& operator=(PostOps &&other) noexcept;
    ~PostOps() noexcept;
    SizeType Len() const noexcept;
    PrimitiveKind Kind(SizeType index) const noexcept;
    Status AppendSum(float scale = 1.0f) noexcept(false);
    void GetParamsSum(SizeType index, float &scale) const noexcept(false);
    Status AppendEltwise(ActivationFunction kind, float alpha = 0.0f, float beta = 0.0f) noexcept(false);
    void GetParamsEltwise(SizeType index, ActivationFunction &kind, float &alpha, float &beta) const noexcept(false);
    Status AppendDW(Element::Type weiType, Element::Type biaType, Element::Type dstType,
            SizeType kernelSize, SizeType strides, SizeType paddingL) noexcept(false);
    void GetParamsDW(SizeType index, Element::Type &weiType, Element::Type &biaType, Element::Type &dstType,
            SizeType &kernelSize, SizeType &strides, SizeType &paddingL) const noexcept(false);
    Status AppendBinary(BinaryFunction op, const TensorInfo &src1) noexcept(false);
    void GetParamsBinary(SizeType index, BinaryFunction &op, TensorInfo &src1) const noexcept(false);
    Status AppendPReLU(PReLUMask mask) noexcept(false);
    void GetParamsPReLU(SizeType index, PReLUMask &mask) const noexcept(false);
private:
    std::unique_ptr<Detail::PostOpsImpl> pImpl;
    friend bool operator==(const PostOps &lhs, const PostOps &rhs) noexcept;
};
bool operator==(const PostOps &lhs, const PostOps &rhs) noexcept;

} // KuDNN

#endif // KUDNN_POST_OPS_HPP
