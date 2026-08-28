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

#ifndef ONEDNN_THREADPOOL_HPP
#define ONEDNN_THREADPOOL_HPP

#include <list>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#define EIGEN_USE_THREADS

#include "dnnl.hpp"
#include "dnnl_threadpool.hpp"

using dnnl::threadpool_interop::threadpool_iface;

// Divide 'n' units of work equally among 'teams' threads. If 'n' is not
// divisible by 'teams' and has a remainder 'r', the first 'r' teams have one
// unit of work more than the rest. Returns the range of work that belongs to
// the team 'tid'.
// Parameters
//   n        Total number of jobs.
//   team     Number of workers.
//   tid      Current thread_id.
//   n_start  start of range operated by the thread.
//   n_end    end of the range operated by the thread.

template <typename T, typename U>
inline void balance211(T n, U team, U tid, T *n_start, T *n_end)
{
    if (team <= 1 || n == 0) {
        *n_start = 0;
        *n_end = n;
        return;
    }
    T min_per_team = n / team;
    T remainder = n - min_per_team * team;  // i.e., n % teams.
    *n_start = tid * min_per_team + std::min(tid, remainder);
    *n_end = *n_start + min_per_team + (tid < remainder);
}

inline void run_jobs(bool balance, int i, int n, int njobs, const std::function<void(int, int)> &fn)
{
    if (balance) {
        int start, end;
        balance211(n, njobs, i, &start, &end);
        for (int j = start; j < end; j++)
            fn(j, n);
    } else {
        fn(i, n);
    }
}

class OneDnnThreadPool : public threadpool_iface {
public:
    OneDnnThreadPool() = default;

    OneDnnThreadPool(Eigen::ThreadPoolInterface *eigen_interface, int num_threads = -1)
        : eigen_interface_(eigen_interface)
    {
        set_num_and_max_threads(num_threads);
    }

    virtual int get_num_threads() const override
    {
        return num_threads_;
    }
    virtual bool get_in_parallel() const override
    {
        return (eigen_interface_->CurrentThreadId() != -1) ? true : false;
    }
    virtual uint64_t get_flags() const override
    {
        return ASYNCHRONOUS;
    }
    virtual void parallel_for(int n, const std::function<void(int, int)> &fn) override
    {
        // Should never happen (handled by DNNL)
        if (n == 0)
            return;

        // Should never happen (handled by DNNL)
        if (n == 1) {
            fn(0, 1);
            return;
        }

        int nthr = get_num_threads();
        int njobs = std::min(n, nthr);
        bool balance = (nthr < n);

        // If use_caller_thread, schedule njobs-1 jobs to thread pool and run last
        // job directly.
        const int njobs_to_schedule = njobs - 1;

        for (int i = 0; i < njobs_to_schedule; i++) {
            eigen_interface_->ScheduleWithHint(
                [balance, i, n, njobs, fn]() { run_jobs(balance, i, n, njobs, fn); }, i, i + 1);
        }
        run_jobs(balance, njobs_to_schedule, n, njobs, fn);
    }

    ~OneDnnThreadPool()
    {}

    static void set_onednn_max_threads(int num_threads)
    {
#if DNNL_VERSION_MAJOR >= 3 || (DNNL_VERSION_MAJOR == 2 && DNNL_VERSION_MINOR >= 7)
#ifndef DNNL_AARCH64_USE_ACL
        dnnl_threadpool_interop_set_max_concurrency(num_threads);
#endif  // DNNL_AARCH64_USE_ACL
#endif  // DNNL_VERSION_MAJOR >= 3 ||
        // (DNNL_VERSION_MAJOR == 2 && DNNL_VERSION_MINOR >= 7)
    }

private:
    Eigen::ThreadPoolInterface *eigen_interface_ = nullptr;
    int num_threads_ = 1;  // Execute in caller thread.
    inline void set_num_and_max_threads(int num_threads)
    {
        num_threads_ = num_threads == -1 ? eigen_interface_->NumThreads() : num_threads;
        set_onednn_max_threads(num_threads_);
    }
};

#endif