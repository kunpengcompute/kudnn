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

#ifndef KUDNN_REORDER_JIT_HPP
#define KUDNN_REORDER_JIT_HPP

#include <algorithm>

#include <omp.h>

#include "kudnn.hpp"

#include "internal_funcs.hpp"
#include "kudnn_jit.hpp"
#include "kudnn_wrapper_helpers.hpp"
#include "kernels/reorder/kudnn_reorder_helpers.hpp"
#include "threading_impl.hpp"

namespace KuDNN {
namespace Detail {

template <typename FastReorderKernelCodeGen>
struct FastReorderImplGenerator {};

struct ReorderImplBase : ImplBase<ReorderInfo> {
    struct UniqueParams final {
    public:
        Shape threadStart;
        Shape threadEnd;
        SizeType kernelIdx;
    };

    ReorderImplBase(const ReorderInfo &task, IntType nThreads) : ImplBase<ReorderInfo>(task, nThreads),
        perThreadParams(0)
    {
        VERBOSE_PRINT;
    }

    void Run(typename ReorderInfo::RunTimeParams params) noexcept(false) override
    {
        VERBOSE_PRINT;
        if ((perThreadParams.size() == 0)) {
            throw Service::LogicError{"Internal Error"};
        }

        if ((implNThreads > 1) && (reorderWorkSize > 1)) {
#if defined(KUDNN_CPU_THREADING_RUNTIME) && (KUDNN_CPU_THREADING_RUNTIME == KUDNN_RUNTIME_OMP)
            auto &&activeLevel = omp_get_active_level();
#pragma omp parallel for num_threads(reorderWorkSize) \
                if ((activeLevel < 1) || \
                (Threading::GetThreadingControlStatus() == Threading::ThreadingControl::KUDNN_MANUAL))
            for (IntType i = 0; i < reorderWorkSize; ++i) {
                RunSequential(i, params.src, params.dst, params.alpha);
            }
#elif defined(KUDNN_CPU_THREADING_RUNTIME) && (KUDNN_CPU_THREADING_RUNTIME == KUDNN_RUNTIME_THREADPOOL)
        KuDNN::Threading::Parallel(reorderWorkSize, [&](IntType i) {
            RunSequential(i, params.src, params.dst, params.alpha);
        });
#endif
        } else {
            RunSequential(0, params.src, params.dst, params.alpha);
        }
    }

    virtual void RunSequential(IntType threadID, const void* src, void* dst, float alpha) const noexcept(false) = 0;

protected:
    std::vector<UniqueParams> perThreadParams;
    IntType reorderWorkSize;

    virtual SizeType GetOptNumThreads() const noexcept = 0;
    
    template <typename Precision>
    static SizeType GetSpecifiedThreadCount(SizeType size)
    {
        SizeType sizeTh2 = 524288;  // 524288 elements count for F32 920F
        SizeType sizeTh3 = 1048576; // 1048576 elements count for F32 920F
        if constexpr (sizeof(Precision) == sizeof(__bf16)) {
            sizeTh2 = 262144; // 262144 experimentally measured for BF16/F16 920F
            sizeTh3 = 524288; // 524288 measured for BF16/F16 920F
        }
        if (size < sizeTh2) {
            return NUM_8;
        } else if ((size >= sizeTh2) && (size <= sizeTh3)) {
            return NUM_16;
        } else {
            return NUM_38;
        }
    }
};

} // Detail
} // KuDNN

#endif // KUDNN_REORDER_JIT_HPP
