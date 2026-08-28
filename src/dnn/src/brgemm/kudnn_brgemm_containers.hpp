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
#ifndef KUDNN_BRGEMM_CONTAINERS_HPP
#define KUDNN_BRGEMM_CONTAINERS_HPP

#include <map>
#include <memory>
#include <set>
#include <vector>

#include "service/kudnn_err_codes.hpp"
#include "types/kudnn_data_type.hpp"

#include "kudnn_brgemm_types.hpp"

namespace KuDNN {

struct BrgemmDescContainerT {
public:
    BrgemmDescContainerT() {}
    explicit BrgemmDescContainerT(SizeType size)
    {
        refs.resize(size);
    }
    inline const BrgemmT *operator[](IntType idx) const
    {
        return refs[idx];
    }

    bool Insert(IntType idx, BrgemmT &brg);

private:
    std::vector<const BrgemmT *> refs;
    std::set<BrgemmT> brgSet;
};

struct BrgemmKernelContainerT {
public:
    BrgemmKernelContainerT() {}
    explicit BrgemmKernelContainerT(SizeType size)
    {
        refs.resize(size);
    }
    inline const BrgemmBaseKernelT *operator[](IntType idx) const
    {
        return refs[idx];
    }
    Status Insert(IntType idx, const BrgemmT *brg);
private:
    std::vector<const BrgemmBaseKernelT *> refs;
    std::unordered_map<const BrgemmT *, std::shared_ptr<BrgemmBaseKernelT>> brgMap;
};

} // namespace KuDNN

#endif // KUDNN_BRGEMM_CONTAINERS_HPP
