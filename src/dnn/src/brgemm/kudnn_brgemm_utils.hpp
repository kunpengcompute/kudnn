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

#ifndef KUDNN_BRGEMM_UTILS_HPP
#define KUDNN_BRGEMM_UTILS_HPP

#include "kudnn_brgemm_types.hpp"
#include "internal_funcs.hpp"
namespace KuDNN {

template <typename base, typename derived>
inline Status SafePtrAssign(base *&lhs, derived *rhs)
{
    if (rhs == nullptr) {
        return Status::BAD_ARGUMENTS;
    }
    lhs = rhs;
    return Status::SUCCESS;
}

Status InitBrgemmConf(BrgemmT *brg, ARMCG::CPUISAT isa, BrgemmBatchKindT kind, Element::TypeT dtA, Element::TypeT dtB,
    BrgemmLayoutT layout, float alpha, float beta, IntType ldA, IntType ldB, IntType ldC, IntType m, IntType n,
    IntType k, const BrgemmStridesT *strides = nullptr, bool isBF32 = false);

Status BrgemmBlocking(BrgemmT *brg);

} // namespace KuDNN

#endif // KUDNN_BRGEMM_UTILS_HPP
