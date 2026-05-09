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

#include <kupl.h>
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

template <typename Void, typename FuncImpl, typename... Args>
struct HasExtraParamsImpl : public std::false_type {};

template <typename FuncImpl, typename... Args>
struct HasExtraParamsImpl<std::void_t<decltype(std::declval<FuncImpl>().RunSequential(
                              Shape{0}, Shape{0}, SizeType{}, SizeType{}, std::declval<Args>()...))>,
                          FuncImpl, Args...> : public std::true_type {};

struct Additive final {
    Additive(const Shape &dimsPara, SizeType lastDimSplitIdx, SizeType incr)
        : dims(dimsPara), ldsi(lastDimSplitIdx), inc(incr)
    {
        for (SizeType i = 0; i < MAX_DIMS; ++i) {
            data[i] = 0;
        }
    }
    // this is not performant, but it's executed inside the loop over numThreads, so it won't be hotspot.
    Additive &operator++() noexcept
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
    const SizeType *GetData() const noexcept
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

} // namespace Helpers

#ifdef __cplusplus
extern "C" {
#endif
extern kupl_egroup_h *g_egroups;

#ifdef __cplusplus
}
#endif

std::vector<std::pair<Shape, Shape>> SplitDims(const Shape &batchedDims, SizeType &optNumThreads) noexcept;

struct SplitDimInfo final {
    SizeType lastDimSplitIdx;
    SizeType prod;
    std::array<SizeType, MAX_DIMS> splitDims;
};

SplitDimInfo GetSplitDimInfo(const Shape &batchedDims, SizeType optNumThreads);

template <typename FuncImpl, typename... Args>
constexpr auto HAS_EXTRA_PARAMS = Helpers::HasExtraParamsImpl<void, FuncImpl, Args...>::value;

template <typename FuncImpl, typename... Args>
void BatchThreader(FuncImpl &&funcObj, Args &&...args) noexcept
{
    auto &&needGetOptNumThreads = ShouldCalculateOptThreads();
    auto &&maxThreads = GetMaxNumThreads();
    auto &&batchedDims = funcObj.GetBatchedDims();
    if (maxThreads > 1) {
        auto &&optNumThreads = needGetOptNumThreads ? funcObj.Parallelizable::GetOptNumThreads(maxThreads) : maxThreads;
        if (optNumThreads == 1) {
            funcObj.RunSequential(batchedDims, args...);
        } else {
            auto &&splitDimsAndStrides = SplitDims(batchedDims, optNumThreads);

            if (g_egroups == NULL) {
                auto &&activeLevel = omp_get_active_level();
#pragma omp parallel for num_threads(splitDimsAndStrides.size()) \
    schedule(static) if ((activeLevel < 1) || (GetThreadingControlStatus() == ThreadingControl::KUDNN_MANUAL))
                for (SizeType i = 0; i < splitDimsAndStrides.size(); ++i) {
                    if constexpr (HAS_EXTRA_PARAMS<FuncImpl, Args...>) {
                        funcObj.RunSequential(splitDimsAndStrides[i].first, splitDimsAndStrides[i].second, i,
                                              splitDimsAndStrides.size(), args...);
                    } else {
                        funcObj.RunSequential(splitDimsAndStrides[i].first, splitDimsAndStrides[i].second, args...);
                    }
                }
            } else {
                int num = splitDimsAndStrides.size();
                kupl_nd_range_t range;
                KUPL_1D_RANGE_INIT(range, 0, num);

                kupl_parallel_for_desc_t desc = {.field_mask = KUPL_PARALLEL_FOR_DESC_FIELD_DEFAULT,
                                                 .range = &range,
                                                 .egroup = nullptr,
                                                 .concurrency = (int)optNumThreads,
                                                 .policy = KUPL_LOOP_POLICY_STATIC};

                int ret = kupl::parallel_for(&desc, [&](const kupl_nd_range_t *nd_range, int tid, int tnum) {
                    Service::Unused(nd_range);
                    Service::Unused(tnum);
                    if constexpr (HAS_EXTRA_PARAMS<FuncImpl, Args...>) {
                        funcObj.RunSequential(splitDimsAndStrides[tid].first, splitDimsAndStrides[tid].second, tid,
                                              splitDimsAndStrides.size(), args...);
                    } else {
                        funcObj.RunSequential(splitDimsAndStrides[tid].first, splitDimsAndStrides[tid].second, args...);
                    }
                });
                Service::Unused(ret);
            }
        }
    } else {
        funcObj.RunSequential(batchedDims, args...);
    }
}

} // namespace Threading
} // namespace KuDNN

#endif // KUDNN_THREADING_IMPL_HPP
