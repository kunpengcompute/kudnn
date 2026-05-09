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

KUDNN_API_PUBLIC bool ShouldCalculateOptThreads() noexcept;
KUDNN_API_PUBLIC int SetMaxNumThreads(int nThreads) noexcept;
KUDNN_API_PUBLIC int SetFixedNumThreads(int nThreads) noexcept;
KUDNN_API_PUBLIC int GetMaxNumThreads() noexcept;
KUDNN_API_PUBLIC int SetNumThreadsLocal(int nThreads) noexcept;
KUDNN_API_PUBLIC EnvMode SetEnvMode(EnvMode mode) noexcept;
KUDNN_API_PUBLIC ThreadingControl GetThreadingControlStatus() noexcept;

} // namespace Threading
} // namespace KuDNN

#endif // KUDNN_THREADING_HPP
