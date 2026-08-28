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

#ifndef KUDNN_THREADING_IMPL_HPP
#define KUDNN_THREADING_IMPL_HPP

#include <omp.h>
#include <vector>
#include <type_traits>
#include <utility>

#include "kudnn.hpp"
#include "internal_funcs.hpp"

namespace KuDNN {
namespace Threading {

int SetNumThreadsLocalInternal(int nThreads) noexcept;

namespace Helpers {

template <typename Void, typename FuncImpl, typename ... Args>
struct HasExtraParamsImpl : public  std::false_type {
};

template<typename FuncImpl, typename ... Args>
struct HasExtraParamsImpl<std::void_t<decltype(std::declval<FuncImpl>().RunSequential(Shape{0}, Shape{0},
    SizeType{}, SizeType{}, std::declval<Args>()...))>, FuncImpl, Args ...> : public std::true_type {
};

struct Additive final {
    Additive(const Shape &dimsPara, SizeType lastDimSplitIdx, SizeType incr)
        : dims(dimsPara), ldsi(lastDimSplitIdx), inc(incr)
    {
        for (SizeType i = 0; i < MAX_DIMS; ++i) {
            data[i] = 0;
        }
    }
    // this is not performant, but it's executed inside the loop over numThreads, so it won't be hotspot.
    Additive& operator++() noexcept
    {
        for (SizeType i = 0; i <= ldsi; ++i) {
            if (i != ldsi) {
                ++(data[i]);
            } else {
                data[i] += inc;
            }
            if (data[i] < dims[i]) {
                break;
            } else {
                data[i] = 0;
            }
        }
        return *this;
    }
    const SizeType* GetData() const noexcept
    {
        return data.data();
    }
    SizeType GetNumDims() const noexcept
    {
        return dims.GetNumDims();
    }
private:
    Shape dims;
    SizeType ldsi; // lastDimSplitIdx
    SizeType inc;
    std::array<SizeType, MAX_DIMS> data;
};

} // Helpers

std::vector<std::pair<Shape, Shape>> SplitDims(const Shape &batchedDims, SizeType& optNumThreads) noexcept;

struct SplitDimInfo final {
    SizeType lastDimSplitIdx;
    SizeType prod;
    std::array<SizeType, MAX_DIMS> splitDims;
};

SplitDimInfo GetSplitDimInfo(const Shape &batchedDims, SizeType optNumThreads);

template <typename FuncImpl, typename ... Args>
constexpr auto HAS_EXTRA_PARAMS = Helpers::HasExtraParamsImpl<void, FuncImpl, Args...>::value;

template <typename FuncImpl, typename ... Args>
void BatchThreader(FuncImpl&& funcObj, Args&& ... args) noexcept
{
    auto &&needGetOptNumThreads = ShouldCalculateOptThreads();
    auto &&maxThreads = GetMaxNumThreads();
    auto &&batchedDims = funcObj.GetBatchedDims();
    if (maxThreads > 1) {
        auto&& optNumThreads = needGetOptNumThreads ?
            funcObj.Parallelizable::GetOptNumThreads(maxThreads) : maxThreads;
        if (optNumThreads == 1) {
            funcObj.RunSequential(batchedDims, args...);
        } else {
            auto &&splitDimsAndStrides = SplitDims(batchedDims, optNumThreads);
            auto &&activeLevel = omp_get_active_level();

#pragma omp parallel for num_threads(splitDimsAndStrides.size()) \
                if ((activeLevel < 1) || (GetThreadingControlStatus() == ThreadingControl::KUDNN_MANUAL))
            for (SizeType i = 0; i < splitDimsAndStrides.size(); ++i) {
                if constexpr (HAS_EXTRA_PARAMS<FuncImpl, Args...>) {
                    funcObj.RunSequential(splitDimsAndStrides[i].first, splitDimsAndStrides[i].second, i,
                        splitDimsAndStrides.size(), args...);
                } else {
                    funcObj.RunSequential(splitDimsAndStrides[i].first, splitDimsAndStrides[i].second, args...);
                }
            }
        }
    } else {
        funcObj.RunSequential(batchedDims, args...);
    }
}
} // Threading
} // KuDNN

#if defined(KUDNN_CPU_THREADING_RUNTIME) && (KUDNN_CPU_THREADING_RUNTIME == KUDNN_RUNTIME_THREADPOOL)
#include <atomic>
#include <cassert>
#include <chrono>
#include <mutex>
#include <thread>
#include <condition_variable>

namespace KuDNN {
namespace Threading {
// internal threadpool impl
ThreadpoolIface *GetActiveThreadpool();
int GetThreadpoolNumThreads();
bool IsInParallel();
void Parallel(int nthr, const std::function<void(int)> &f);

struct CountingBarrier {
    CountingBarrier(unsigned size = 0)
    {
        Init(size);
    }

    void Init(unsigned size)
    {
        assert(size < KUDNNWaiterMask);
        notified = false;
        state = size;
    }

    void Notify()
    {
        auto s = state.fetch_sub(1) - 1;
        if (s != KUDNNWaiterMask) {
            assert(((s + 1) & ~KUDNNWaiterMask) != 0);
            return;
        }
        std::unique_lock<std::mutex> l(mutex);
        notified = true;
        cv.notify_all();
    }

    void Wait()
    {
        auto s = state.fetch_or(KUDNNWaiterMask);
        if (s == 0) {
            return;
        }
        std::unique_lock<std::mutex> l(mutex);
        cv.wait(l, [this]() { return notified; });
    }

private:
    static constexpr unsigned KUDNNWaiterMask = 1u << (sizeof(unsigned) * 8 - 1);

    std::atomic<unsigned> state;
    bool notified;

    std::condition_variable cv;
    std::mutex mutex;
};
} // Threading
} // KuDNN
#endif

#endif // KUDNN_THREADING_IMPL_HPP
