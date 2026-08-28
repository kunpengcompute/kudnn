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

#ifndef KUDNN_FAST_REORDER_SAME_LAYOUTS
#define KUDNN_FAST_REORDER_SAME_LAYOUTS

#include <memory>
#include <vector>
#include <algorithm>
#include <cmath>

#include <arm_sve.h>

#include "kudnn.hpp"

#include "kernels/reorder/sve/same_layouts/kudnn_fast_reorder_kernels.hpp"
#include "kernels/reorder/kudnn_reorder_helpers.hpp"
#include "kernels/reorder/kudnn_reorder_support_functions.hpp"
#include "kernels/reorder/kudnn_reorder_jit.hpp"
#include "kudnn_jit.hpp"
#include "kudnn_wrapper_helpers.hpp"
#include "threading_impl.hpp"
#include "vector_wrappers.hpp"

namespace KuDNN {
namespace Detail {

template <typename InType, typename OutType>
struct SameLayoutsFastReorder : ImplGenerator<ReorderInfo> {
    struct ImplSameLayoutsFastReorder : ReorderImplBase {
        ImplSameLayoutsFastReorder(const ReorderInfo &task, SizeType nThreads) : ReorderImplBase(task, nThreads)
        {
            VERBOSE_PRINT;

            Shape dims = Service::GetShapeAccordingToLayout(task.dstInfo.GetDims(), task.srcInfo.GetLayout());
            Shape srcStrides = Service::GetShapeAccordingToLayout(task.srcInfo.GetStrides(), task.srcInfo.GetLayout());
            Shape dstStrides = Service::GetShapeAccordingToLayout(task.dstInfo.GetStrides(), task.srcInfo.GetLayout());
            reorderedDims = Service::BroadcastTo5D(dims);
            reorderedStridesSrc = Service::BroadcastTo5D(srcStrides, false);
            reorderedStridesDst = Service::BroadcastTo5D(dstStrides, false);

            for (SizeType i = reorderedDims.GetNumDims() - 1; i > 0; --i) {
                ReorderHelpers::ReduceDimsAndStrides(reorderedDims, reorderedStridesSrc, reorderedStridesDst);
            }
            for (SizeType i = reorderedDims.GetNumDims() - 1; i > 0; --i) {
                ReorderHelpers::CompressDimsAndStrides(reorderedDims, reorderedStridesSrc, reorderedStridesDst);
            }

            SizeType optNumThreads =
                std::min(nThreads, Threading::ShouldCalculateOptThreads() ? GetOptNumThreads() : nThreads);
            if (nThreads > 1 && dims.GetTotalDimsSize() != 0) {
                std::vector<std::pair<Shape, Shape>> reorderWork = Threading::SplitDims(reorderedDims,
                                                                                        optNumThreads);
                perThreadParams.reserve(reorderWork.size());
                reorderWorkSize = reorderWork.size();
                for (SizeType i = 0; i < reorderWork.size(); ++i) {
                    const Shape threadStart = Service::BroadcastTo5D(reorderWork[i].second, false);
                    const Shape threadEnd = threadStart + Service::BroadcastTo5D(reorderWork[i].first, true);
                    perThreadParams.push_back({threadStart, threadEnd, 0});
                }
            } else {
                const Shape threadStart = {0, 0, 0, 0, 0};
                const Shape threadEnd = reorderedDims;
                perThreadParams.push_back({threadStart, threadEnd, 0});
            }
        }
    protected:
        void RunSequential(IntType threadID, const void *src, void *dst, float alpha) const noexcept(false) override
        {
            const InType *typedSrc = static_cast<const InType *>(src);
            OutType *typedDst = static_cast<OutType *>(dst);
            KuDNN::Detail::Reorder::ReorderSameLayouts(perThreadParams[threadID].threadStart,
                perThreadParams[threadID].threadEnd, typedSrc, typedDst,
                reorderedStridesSrc, reorderedStridesDst, alpha);
        }

        SizeType GetOptNumThreads() const noexcept override
        {
            SizeType numElements = reorderedDims.GetTotalDimsSize();
#ifdef KUDNNL_920F
            SizeType l2UpperPart = 131072;
            if (L2_CACHE_SIZE_BYTES / sizeof(InType) > SizeType(numElements)) {
                if (SizeType(numElements) >= l2UpperPart) {
                    return NUM_4;
                }
                return NUM_1;
            } else {
                return GetSpecifiedThreadCount<InType>(SizeType(L3_CACHE_SIZE_BYTES / sizeof(InType)));
            }
#else
            SizeType l2UpperPart = 160000;
            SizeType l3UpperPart = 524288;
            if (L1_CACHE_SIZE_BYTES / sizeof(InType) > SizeType(numElements)) {
                return NUM_1;
            } else if (numElements < L2_CACHE_SIZE_BYTES / sizeof(InType)) {
                if (numElements <= l2UpperPart) {
                    return NUM_8;
                }
                return NUM_16;
            } else if (numElements < L3_CACHE_SIZE_BYTES / sizeof(InType)) {
                if (numElements <= l3UpperPart) {
                    return NUM_16;
                }
                return NUM_40;
            } else {
                return NUM_40;
            }
#endif
        }

        Shape reorderedDims;
        Shape reorderedStridesSrc;
        Shape reorderedStridesDst;
    };

    bool Supports(const ReorderInfo &task) const noexcept override
    {
        VERBOSE_PRINT;
        Shape srcStrides = Service::GetShapeAccordingToLayout(task.srcInfo.GetStrides(), task.srcInfo.GetLayout());
        Shape dstStrides = Service::GetShapeAccordingToLayout(task.dstInfo.GetStrides(), task.srcInfo.GetLayout());
        srcStrides = Service::BroadcastTo5D(srcStrides, false);
        dstStrides = Service::BroadcastTo5D(dstStrides, false);

        bool isContinuousRow = (srcStrides[IDX_4] == NUM_1 && dstStrides[IDX_4] == NUM_1);
        bool srcMatched = (task.srcInfo.GetType() == Element::MatchType<InType>());
        bool dstMatched = (task.dstInfo.GetType() == Element::MatchType<OutType>());
        return isContinuousRow && (srcMatched && dstMatched);
    }
    std::chrono::duration<float, std::milli> EstimateTime(const ReorderInfo  &task,
                                                            IntType nThreads) const noexcept override
    {
        VERBOSE_PRINT;
        Service::Unused(task, nThreads);
        return std::chrono::duration<float, std::milli>{0.1f};
    }
    std::shared_ptr<ImplBase<ReorderInfo>> GetImpl(const ReorderInfo &task, IntType nThreads) const override
    {
        VERBOSE_PRINT;
        return std::make_shared<ImplSameLayoutsFastReorder>(task, nThreads);
    }
};

} // Detail
} // KuDNN

#endif // KUDNN_FAST_REORDER_SAME_LAYOUTS
