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

#include <cstdlib>
#include <omp.h>
#include <vector>
#include <utility>

#include <unistd.h>

#include "kudnn.hpp"
#include "kblas.h"

#include "internal_funcs.hpp"
#include "threading_impl.hpp"

namespace KuDNN {
namespace Threading {

static int g_kdnnMaxNumThreads = 0;
static int g_kdnnFixedNumThreads = 0;
static int g_availableCores = 0;

#ifdef __GNUG__
static __thread int g_kdnnMaxNumThreadsLocal = 0;
#else
static __declspec(__thread) int g_kdnnMaxNumThreadsLocal = 0;
#endif

/* 0 - Do nothing, default; non-zero value - Ignore environment settings specific to the library. */
static EnvMode g_modeType = EnvMode::KUDNN_THREAD_USE_ENV;
static ThreadingControl g_useKUDNNThreading = ThreadingControl::KUDNN_DEFAULT;
static bool g_fixedNumThreads = false;

static CONSTRUCTOR void KudnnParallelParamInit() noexcept
{
    g_availableCores = sysconf(_SC_NPROCESSORS_ONLN);
    int maxThreadFromCaller = std::min(g_availableCores, omp_get_max_threads());
    if (g_modeType == EnvMode::KUDNN_THREAD_USE_ENV) {
        char *pkdnnNumThreads;
        if ((pkdnnNumThreads = std::getenv("KUDNN_FIXED_NUM_THREADS")) != nullptr) {
            g_kdnnFixedNumThreads = std::atol(pkdnnNumThreads);
            if (g_kdnnFixedNumThreads > 0) {
                g_fixedNumThreads = true;
                g_kdnnFixedNumThreads = std::min(g_kdnnFixedNumThreads, g_availableCores);
                g_useKUDNNThreading = ThreadingControl::KUDNN_MANUAL;
                BlasSetNumThreads(g_kdnnFixedNumThreads);
                return;
            }
        }
    }
    if (g_modeType == EnvMode::KUDNN_THREAD_USE_ENV) {
        char *pkdnnNumThreads;
        if ((pkdnnNumThreads = std::getenv("KUDNN_MAX_NUM_THREADS")) != nullptr) {
            g_kdnnMaxNumThreads = std::atol(pkdnnNumThreads);
            if (g_kdnnMaxNumThreads > 0) {
                g_kdnnMaxNumThreads = std::min(g_kdnnMaxNumThreads, g_availableCores);
                g_useKUDNNThreading = ThreadingControl::KUDNN_MANUAL;
                BlasSetNumThreads(g_kdnnMaxNumThreads);
                return;
            }
        }
    }

    g_useKUDNNThreading = ThreadingControl::KUDNN_DEFAULT;
    g_kdnnMaxNumThreads = maxThreadFromCaller;
    BlasSetNumThreads(g_kdnnMaxNumThreads);
}

EnvMode SetEnvMode(EnvMode mode) noexcept
{
    g_modeType = mode;
    return g_modeType;
}

int SetMaxNumThreads(int nThreads) noexcept
{
    int oldkdnnMaxNumThreads = g_kdnnMaxNumThreads;
    if ((nThreads > 0) && (nThreads <= g_availableCores)) {
        g_kdnnMaxNumThreads = nThreads;
        g_useKUDNNThreading = ThreadingControl::KUDNN_MANUAL;
        BlasSetNumThreads(g_kdnnMaxNumThreads);
    }
    return oldkdnnMaxNumThreads;
}

int SetFixedNumThreads(int nThreads) noexcept
{
    int oldkdnnFixedNumThreads = g_kdnnFixedNumThreads;
    if ((nThreads > 0) && (nThreads <= g_availableCores)) {
        g_fixedNumThreads = true;
        g_kdnnFixedNumThreads = nThreads;
        g_useKUDNNThreading = ThreadingControl::KUDNN_MANUAL;
        BlasSetNumThreads(g_kdnnFixedNumThreads);
    }
    return oldkdnnFixedNumThreads;
}

bool ShouldCalculateOptThreads() noexcept
{
    return !g_fixedNumThreads;
}

int GetMaxNumThreads() noexcept
{
    if (g_kdnnMaxNumThreadsLocal > 0) {
        return g_kdnnMaxNumThreadsLocal;
    } else if (g_fixedNumThreads) {
        return g_kdnnFixedNumThreads;
    } else {
        return g_kdnnMaxNumThreads;
    }
}

int SetNumThreadsLocal(int nThreads) noexcept
{
    int oldkdnnMaxNumThreadsLocal = g_kdnnMaxNumThreadsLocal;
    if ((nThreads >= 1) && (nThreads <= g_availableCores)) {
        g_kdnnMaxNumThreadsLocal = nThreads;
        g_useKUDNNThreading = ThreadingControl::KUDNN_MANUAL;
    }
    return oldkdnnMaxNumThreadsLocal;
}

int SetNumThreadsLocalInternal(int nThreads) noexcept
{
    int oldkdnnMaxNumThreadsLocal = g_kdnnMaxNumThreadsLocal;
    if ((nThreads >= 0) && (nThreads <= g_availableCores)) {
        g_kdnnMaxNumThreadsLocal = nThreads;
    }
    return oldkdnnMaxNumThreadsLocal;
}

ThreadingControl GetThreadingControlStatus() noexcept
{
    return g_useKUDNNThreading;
}

SplitDimInfo GetSplitDimInfo(const Shape &batchedDims, SizeType optNumThreads)
{
    SplitDimInfo info {0, 1, {0, 0, 0, 0, 0}};

    // find d (lastDimSplitIdx)
    while ((info.lastDimSplitIdx < batchedDims.GetNumDims()) && (info.prod <= optNumThreads)) {
        info.prod *= batchedDims[info.lastDimSplitIdx];
        info.splitDims[info.lastDimSplitIdx] = 1;
        ++info.lastDimSplitIdx;
    }
    // one iteration is extra
    --info.lastDimSplitIdx; // in while loop there will be at least one iteration, so it's safe to decrement
    info.prod /= batchedDims[info.lastDimSplitIdx];

    return info;
}

// This function is complex, but it's not recommended to split it into separate functions since
// partition logic is closely related to strides computations. Splitting these two logical points
// may lead to unaligned behavior between them, thus errors.
// each vector element represents splitDims and splitStrides for the i-th thread.
std::vector<std::pair<Shape, Shape>> SplitDims(const Shape &batchedDims, SizeType& optNumThreads) noexcept
{
    SplitDimInfo res = GetSplitDimInfo(batchedDims, optNumThreads);
    SizeType lastDimSplitIdx = res.lastDimSplitIdx;
    SizeType prod = res.prod;
    std::array<SizeType, MAX_DIMS> splitDims = res.splitDims;

    SizeType numThreadsLastSplitDim = optNumThreads / prod; // numThreadsLastSplitDim >= 1 so it's safe to do subtract 1

    if (numThreadsLastSplitDim > batchedDims[lastDimSplitIdx]) {
        numThreadsLastSplitDim = batchedDims[lastDimSplitIdx];
    }

    splitDims[lastDimSplitIdx] = (batchedDims[lastDimSplitIdx] + numThreadsLastSplitDim - 1) / numThreadsLastSplitDim;

    // to avoid cases such as 7 threads over 17 numbers (we will actually have 5 threads for 3 and 1 thread for 2,
    // total 6 threads in use)
    while ((numThreadsLastSplitDim - 1) * splitDims[lastDimSplitIdx] > batchedDims[lastDimSplitIdx]) {
        --numThreadsLastSplitDim;
    }

    std::array<SizeType, MAX_DIMS> splitDimsResid = splitDims;
    splitDimsResid[lastDimSplitIdx] = batchedDims[lastDimSplitIdx] -
        (numThreadsLastSplitDim - 1) * splitDims[lastDimSplitIdx];
    // fill other dimensions unchanged
    SizeType ctr = lastDimSplitIdx + 1;
    while (ctr < batchedDims.GetNumDims()) {
        splitDimsResid[ctr] = splitDims[ctr] = batchedDims[ctr];
        ctr++;
    }

    // actual quantity of threads will be (prod * numThreadsLastSplitDim),
    // so we reserve memory to push back w/0 overhead on reallocations
    std::vector<std::pair<Shape, Shape>> out;
    optNumThreads = prod * numThreadsLastSplitDim;
    out.reserve(optNumThreads);

    Helpers::Additive add(batchedDims, lastDimSplitIdx, splitDims[lastDimSplitIdx]);
    for (SizeType lastSplidIdx = 0; lastSplidIdx < numThreadsLastSplitDim - 1; ++lastSplidIdx) {
        for (SizeType i = 0; i < prod; ++i) {
            out.push_back({{splitDims.data(), batchedDims.GetNumDims()}, {add.GetData(), add.GetNumDims()}});
            ++add;
        }
    }
    for (SizeType i = 0; i < prod; ++i) {
        out.push_back({{splitDimsResid.data(), batchedDims.GetNumDims()}, {add.GetData(), add.GetNumDims()}});
        ++add;
    }

    return out;
}

} // Threading
} // KuDNN
