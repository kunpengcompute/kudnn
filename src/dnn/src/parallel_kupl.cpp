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

#include <kupl.h>
#include "parallel_for.h"
#include "internal_funcs.hpp"

namespace KuDNN {
namespace Threading {

void KuplBackend ::parallel_for(int64_t total, int numThreads, const std::function<void(int64_t)> &fn)
{
    kupl_nd_range_t range;
    KUPL_1D_RANGE_INIT(range, 0, total);

    kupl_parallel_for_desc_t desc = {
        .field_mask = KUPL_PARALLEL_FOR_DESC_FIELD_DEFAULT,
        .range = &range,
        .egroup = nullptr,
        .concurrency = numThreads,
        .policy = KUPL_LOOP_POLICY_STATIC,
    };

    kupl::parallel_for(&desc, [&](const kupl_nd_range_t *nd_range, int tid, int tnum) {
        Service::Unused(tid);
        Service::Unused(tnum);
        // kupl 的 range 是切分后的范围
        for (int64_t i = nd_range->nd_range[0].lower; i < nd_range->nd_range[0].upper;
            i += nd_range->nd_range[0].step) {
            fn(i);
        }
    });
}

int KuplBackend ::get_default_num_threads()
{
    return kupl_get_num_executors();
}
}
}