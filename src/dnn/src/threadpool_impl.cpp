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

#include <cstring>
#include <dlfcn.h>
#include "threading_impl.hpp"
#include "kupl.h"
#include "omp.h"

namespace KuDNN {
namespace Threading {

static void KuplFunc(kupl_nd_range_t* nd_range, void* args, int tid, int tnum)
{
    Service::Unused(tid);
    std::function<void(int, int)>* f = (std::function<void(int, int)>*)args;
    for (int64_t i = nd_range->nd_range[0].lower; i < nd_range->nd_range[0].upper; i += nd_range->nd_range[0].step) {
        (*f)(i, tnum);
    }
}

class KuplThreadPool : public KuDNN::Threading::ThreadpoolIface {
public:
    // 使用 using 定义类型别名
    using kupl_parallel_for_t = int (*)(kupl_parallel_for_desc_t*, kupl_pf_func_t, void*);
    using kupl_get_num_executors_t = int (*)();
    using kupl_in_parallel_t = bool (*)();

    explicit KuplThreadPool(int num = 0)
        : numThreads(num),
          kupl_parallel_for(nullptr),
          kupl_get_num_executors(nullptr),
          kupl_in_parallel(nullptr)
    {
        LoadKuplLibrary();
    }

    int GetNumThreads() const override
    {
        return kupl_get_num_executors();
    }

    uint64_t GetFlags() const override
    {
        return KUDNN_ASYNCHRONOUS;
    }

    void ParallelFor(int n, const std::function<void(int, int)>& fn) override
    {
        kupl_nd_range_t range;
        KUPL_1D_RANGE_INIT(range, 0, n);
        kupl_parallel_for_desc_t desc = {
            .field_mask = KUPL_PARALLEL_FOR_DESC_FIELD_DEFAULT,
            .range = &range,
            .egroup = nullptr,
            .concurrency = this->numThreads,
            .policy = KUPL_LOOP_POLICY_STATIC,
        };
        kupl_parallel_for(&desc, KuplFunc, (void*)&fn);
    };

    bool IsInParallel() const override
    {
        return kupl_in_parallel();
    }

private:
    int numThreads;
    // 使用类型别名
    kupl_parallel_for_t kupl_parallel_for;
    kupl_get_num_executors_t kupl_get_num_executors;
    kupl_in_parallel_t kupl_in_parallel;

    void LoadKuplLibrary()
    {
        const char* kuplLibPathEnv  = std::getenv("KUPL_LIBRARY_PATH");
        std::string kuplLibPath;
        std::string kuplPathSuffix;
#if defined(__clang__)
    kuplPathSuffix = "/../../../kupl/bisheng/release/lib/libkupl.so";
#elif defined(__GNUC__)
    kuplPathSuffix = "/../../../kupl/gcc/release/lib/libkupl.so";
#else
    printf("Using an unknown compiler\n");
#endif
        if (!kuplLibPathEnv) {
            Dl_info dlInfo;
            if (dladdr((void *)KuplFunc, &dlInfo) != 0 && dlInfo.dli_fname != nullptr) {

                std::string currentSoPath = dlInfo.dli_fname;
                size_t lastSlash = currentSoPath.find_last_of('/');
                if (lastSlash != std::string::npos) {
                    std::string dirPath = currentSoPath.substr(0, lastSlash);
                    
                    // 拼接路径
                    kuplLibPath = dirPath + kuplPathSuffix;
                }
            }
        } else {
            kuplLibPath = kuplLibPathEnv;
        }

        void *libHandle = dlopen(kuplLibPath.c_str(), RTLD_LAZY);
        if (!libHandle) {
            throw std::runtime_error("Failed to load libkupl.so");
        }

        dlerror();
        kupl_parallel_for = (kupl_parallel_for_t)dlsym(libHandle, "kupl_parallel_for");
        const char* error = dlerror();
        if (error) {
            dlclose(libHandle);
            libHandle = nullptr;
            throw std::runtime_error("Failed to load kupl_parallel_for: " + std::string(error));
        }

        // 加载 kupl_get_num_executors
        kupl_get_num_executors = (kupl_get_num_executors_t)dlsym(libHandle, "kupl_get_num_executors");
        error = dlerror();
        if (error) {
            kupl_get_num_executors = nullptr;
        }

        kupl_in_parallel = (kupl_in_parallel_t)dlsym(libHandle, "kupl_in_parallel");
        error = dlerror();
        if (error) {
            kupl_in_parallel = nullptr;
        }
    }
};

class OmpThreadPool : public KuDNN::Threading::ThreadpoolIface {
public:
    explicit OmpThreadPool(int num = 0)
    {
        this->numThreads = num;
        omp_set_num_threads(this->numThreads);
    }

    int GetNumThreads() const override
    {
        return this->numThreads;
    }

    uint64_t GetFlags() const override
    {
        return KUDNN_ASYNCHRONOUS;
    }

    void ParallelFor(int n, const std::function<void(int, int)>& fn) override
    {
#pragma omp parallel for
        for (int i = 0; i < n; i++) {
            fn(i, this->numThreads);
        }
    };

    bool IsInParallel() const override
    {
        return omp_in_parallel();
    }

private:
    int numThreads;
};

static CONSTRUCTOR void KudnnParallelInit() noexcept
{
    char *pkdnnThreadType, *pkdnnThreadNum;
    int threadNum = 0;
    if ((pkdnnThreadNum = std::getenv("OMP_NUM_THREADS")) != nullptr) {
        threadNum = std::stoi(pkdnnThreadNum);
    } else {
        threadNum = omp_get_max_threads();
    }
    if ((pkdnnThreadType = std::getenv("KUDNN_THREAD_TYPE")) != nullptr) {
        if (strcmp(pkdnnThreadType, "KUPL") == 0) {
            KuplThreadPool* kuplThreadPool = new KuplThreadPool(threadNum);
            ActivateThreadpool(kuplThreadPool);
        } else {  // 默认OMP
            OmpThreadPool* ompThreadPool = new OmpThreadPool(threadNum);
            ActivateThreadpool(ompThreadPool);
        }
    } else {
        OmpThreadPool* ompThreadPool = new OmpThreadPool(threadNum);
        ActivateThreadpool(ompThreadPool);
    }
}

namespace {
    thread_local ThreadpoolIface* kdnnThreadPool = nullptr;
}

void ActivateThreadpool(KuDNN::Threading::ThreadpoolIface *tp)
{
    if (!kdnnThreadPool) {
        kdnnThreadPool = tp;
    }
}

void DeactivateThreadpool()
{
    kdnnThreadPool = nullptr;
}

ThreadpoolIface *GetActiveThreadpool()
{
    return kdnnThreadPool;
}

bool IsInParallel()
{
    ThreadpoolIface *tp = GetActiveThreadpool();
    return tp ? tp->IsInParallel() : false;
}

int GetThreadpoolNumThreads()
{
    ThreadpoolIface *tp = GetActiveThreadpool();
    return tp ? tp->GetNumThreads() : 1;
}

void Parallel(int nthr, const std::function<void(int)> &f)
{
    ThreadpoolIface *tp = GetActiveThreadpool();
    if (!tp || IsInParallel()) {
        DeactivateThreadpool();
        for (int ithr = 0; ithr < nthr; ithr++) {
            f(ithr);
        }
        ActivateThreadpool(tp);
    } else {
        bool async = tp->GetFlags() & KuDNN::Threading::ThreadpoolIface::KUDNN_ASYNCHRONOUS;
        CountingBarrier b;
        if (async) b.Init(nthr);
        tp->ParallelFor(nthr, [&](int ithread, int nthread) {
            KuDNN::Service::Unused(nthread);
            bool isMaster = GetActiveThreadpool() == tp;
            if (!isMaster) {
                ActivateThreadpool(tp);
            }
            f(ithread);
            if (!isMaster) {
                DeactivateThreadpool();
            }
            if (async) {
                b.Notify();
            }
        });
        if (async) {
            b.Wait();
        }
    }
}
} // Threading
} // KuDNN