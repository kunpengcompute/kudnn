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

#include <memory>
#include <vector>

#include "kudnn.hpp"
#include "kudnn_jit.hpp"
#include "kernels/reorder/kudnn_reorder_helpers.hpp"

namespace KuDNN {
namespace Detail {

// For hash function we want pretty unique value, main info about current reorder task
// are src and dst dims, so we want to make our hash with multiplication adjacent dimensions
// We safe with this loop since lowest GetNumDims() = 2 for src and dst;
template<>
SizeType GetTaskHash<ReorderInfo>(const ReorderInfo& task)
{
    SizeType intHash = 0;
    for (SizeType i = task.srcInfo.GetNumDims() - 1; i > 0; --i) {
        intHash += task.srcInfo.GetDims()[i] * task.srcInfo.GetDims()[i - 1];
        intHash += task.dstInfo.GetDims()[i] * task.dstInfo.GetDims()[i - 1];
    }
    return intHash;
}

} // Detail
} // KuDNN
