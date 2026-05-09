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

#include "parallel_for.h"

namespace KuDNN {
namespace Threading {

class SerialBackend : public ParallelBackend {
public:
    void parallel_for(int64_t total, int, const std::function<void(int64_t)> &fn)
    {
        for (int i = 0; i < total; ++i) {
            fn(i);
        }
    }

    static int get_default_num_threads()
    {
        return 1;
    }
};

int g_defaultNumThreads;

std::unique_ptr<ParallelBackend> create_parallel_backend()
{
    const char *env = std::getenv("KUDNN_THREAD_TYPE");

    if (!env || strcmp(env, "OMP") == 0) {
        g_defaultNumThreads = OpenMPBackend::get_default_num_threads();
        return std::make_unique<OpenMPBackend>();
    }

    if (strcmp(env, "KUPL") == 0) {
        g_defaultNumThreads = KuplBackend::get_default_num_threads();
        return std::make_unique<KuplBackend>();
    }

    if (strcmp(env, "SERIAL") == 0) {
        g_defaultNumThreads = SerialBackend::get_default_num_threads();
        return std::make_unique<SerialBackend>();
    }

    g_defaultNumThreads = OpenMPBackend::get_default_num_threads();
    return std::make_unique<OpenMPBackend>();
}

} // namespace Threading
} // namespace KuDNN
