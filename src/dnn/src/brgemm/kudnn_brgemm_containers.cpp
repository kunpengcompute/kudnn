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

#include "kudnn_brgemm_containers.hpp"

#include <memory>

#include "service/kudnn_err_codes.hpp"
#include "service/kudnn_exception.hpp"
#include "types/kudnn_data_type.hpp"

#include "kudnn_brgemm.hpp"
#include "kudnn_brgemm_types.hpp"

namespace KuDNN {

bool BrgemmDescContainerT::Insert(IntType idx, BrgemmT &brg)
{
    const auto insret = brgSet.insert(brg);
    if (insret.second) {
        refs[idx] = &(*insret.first);
    }

    return insret.second;
}

Status BrgemmKernelContainerT::Insert(IntType idx, const BrgemmT *brg)
{
    auto itr = brgMap.find(brg);
    bool check = itr != brgMap.end();
    if (!check) {
        BrgemmBaseKernelT *kernel = nullptr;
        check = BrgemmKernelCreate(&kernel, *brg) == Status::SUCCESS;
        std::shared_ptr<BrgemmBaseKernelT> sptr(kernel);
        refs[idx] = kernel;
        check &= brgMap.insert({brg, sptr}).second;
        if (!check) {
            throw Service::LogicError("");
        }
    } else {
        refs[idx] = itr->second.get();
    }
    return Status::SUCCESS;
}

} // namespace KuDNN
