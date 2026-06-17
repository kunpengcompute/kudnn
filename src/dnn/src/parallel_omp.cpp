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

#include <omp.h>
#include "parallel_for.h"

namespace KuDNN {
namespace Threading {

void OpenMPBackend ::parallel_for(int64_t total, int numThreads, const std::function<void(int64_t)> &fn)
{
#pragma omp parallel for num_threads(numThreads)
    for (int64_t i = 0; i < total; i++) {
        fn(i);
    }
}

int OpenMPBackend ::get_default_num_threads()
{
    return omp_get_max_threads();
}

} // namespace Threading
} // namespace KuDNN
