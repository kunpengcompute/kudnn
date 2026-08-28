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

#ifndef KUDNN_THREADING_HPP
#define KUDNN_THREADING_HPP

#include <cstdint>
#include <functional>

namespace KuDNN {
namespace Threading {

enum class EnvMode {
    KUDNN_THREAD_USE_ENV,
    KUDNN_THREAD_IGNORE_ENV
};

enum class ThreadingControl {
    KUDNN_DEFAULT,
    KUDNN_MANUAL
};

bool ShouldCalculateOptThreads() noexcept;
int SetMaxNumThreads(int nThreads) noexcept;
int SetFixedNumThreads(int nThreads) noexcept;
int GetMaxNumThreads() noexcept;
int SetNumThreadsLocal(int nThreads) noexcept;
EnvMode SetEnvMode(EnvMode mode) noexcept;
ThreadingControl GetThreadingControlStatus() noexcept;

struct ThreadpoolIface {
    virtual int GetNumThreads() const = 0;
    virtual bool IsInParallel() const = 0;
    virtual void ParallelFor(int n, const std::function<void(int, int)> &fn) = 0;
    virtual uint64_t GetFlags() const = 0;
    static constexpr uint64_t KUDNN_ASYNCHRONOUS = 1;
    virtual ~ThreadpoolIface() {};
};

void ActivateThreadpool(ThreadpoolIface *tp);
void DeactivateThreadpool();

} // Threading
} // KuDNN

#endif // KUDNN_THREADING_HPP
