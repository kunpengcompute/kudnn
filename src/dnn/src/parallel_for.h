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

#pragma once
#include <functional>
#include <cstdint>
#include <memory>

#include <cstring>

#include "service/kudnn_err_codes.hpp"
#include "types/kudnn_tensor_info.hpp"

namespace KuDNN {
namespace Threading {

extern int g_defaultNumThreads;

class ParallelBackend {
public:
    virtual ~ParallelBackend() = default;

    // 1D 区间并行
    virtual void parallel_for(int64_t total, int numThreads, const std::function<void(int64_t)> &fn) = 0;
};

class KuplBackend : public ParallelBackend {
public:
    void parallel_for(int64_t total,
                      int numThreads,
                      const std::function<void(int64_t)> &fn);
    static int get_default_num_threads();
};

class OpenMPBackend : public ParallelBackend {
public:
    void parallel_for(int64_t total,
                      int numThreads,
                      const std::function<void(int64_t)> &fn);
    static int get_default_num_threads();
};

std::unique_ptr<ParallelBackend> create_parallel_backend();

class ParallelFor {
public:
    static ParallelFor &instance()
    {
        static ParallelFor inst;
        return inst;
    }

    void run(const std::function<void(int64_t)> &fn, int64_t total, int numThreads = 0)
    {
        if (numThreads == 0) {
            numThreads = g_defaultNumThreads;
        }
        backend_->parallel_for(total, numThreads, fn);
    }

private:
    ParallelFor() : backend_(create_parallel_backend()) {}
    std::unique_ptr<ParallelBackend> backend_;
};

template <size_t N>
class ParallelForND {
public:
    using Shape = KuDNN::Shape;

    template <typename Func>
    static void run(Func &&func, const Shape &shape, int numThreads = 0)
    {
        int64_t total = 1;
        for (size_t i = 0; i < N; ++i) {
            total *= shape[i];
        }

        ParallelFor::instance().run(
            [&](int64_t linear) {
                std::array<int64_t, N> idx;
                decode(linear, shape, idx);
                call_func(func, idx, std::make_index_sequence<N>{});
            }, total, numThreads);
    }

private:
    // 把线性 index 解码成 N 维坐标
    static inline void decode(int64_t linear, const Shape &shape, std::array<int64_t, N> &idx)
    {
        for (int i = N - 1; i >= 0; --i) {
            idx[i] = linear % shape[i];
            linear /= shape[i];
        }
    }

    // 把 array 展开成 func(a,b,c,...) 调用
    template <typename Func, size_t... I>
    static inline void call_func(Func &func, const std::array<int64_t, N> &idx, std::index_sequence<I...>)
    {
        func(idx[I]...);
    }
};

}  // namespace Threading
}  // namespace KuDNN