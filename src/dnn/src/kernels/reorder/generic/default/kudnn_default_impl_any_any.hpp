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

#ifndef KUDNN_DEFAULT_IMPL_ANY_ANY_HPP
#define KUDNN_DEFAULT_IMPL_ANY_ANY_HPP

#include <memory>
#include <vector>
#include <algorithm>
#include <cmath>
#include <functional>
#include <unordered_map>

#include <arm_sve.h>

#include "kudnn.hpp"

#include "kernels/reorder/kudnn_reorder_helpers.hpp"
#include "kudnn_jit.hpp"
#include "kudnn_reorder_kernels.hpp"
#include "kudnn_wrapper_helpers.hpp"
#include "threading_impl.hpp"
#include "vector_wrappers.hpp"

namespace KuDNN {
namespace Detail {

struct MainSchemeSimpleReorder : ImplGenerator<ReorderInfo> {
    struct ImplMainSimpleReorder : ReorderImplBase {
        ImplMainSimpleReorder(const ReorderInfo &task, SizeType nThreads) : ReorderImplBase(task, nThreads),
            srcType(task.srcInfo.GetType()), dstType(task.dstInfo.GetType()),
            reorderedDims(Service::BroadcastTo5D(Service::GetShapeAccordingToLayout(
                task.dstInfo.GetDims(), task.srcInfo.GetLayout()))),
            reorderedStridesSrc(Service::BroadcastTo5D(Service::GetShapeAccordingToLayout(
                task.srcInfo.GetStrides(), task.srcInfo.GetLayout()), false)),
            reorderedStridesDst(Service::BroadcastTo5D(Service::GetShapeAccordingToLayout(
                task.dstInfo.GetStrides(), task.srcInfo.GetLayout()), false))
        {
            VERBOSE_PRINT;

            SizeType optNumThreads =
                std::min(nThreads, Threading::ShouldCalculateOptThreads() ? GetOptNumThreads() : nThreads);

            if (nThreads > 1 && reorderedDims.GetTotalDimsSize() != 0) {
                std::vector<std::pair<Shape, Shape>> reorderWork = Threading::SplitDims(reorderedDims,
                                                                                        optNumThreads);
                perThreadParams.reserve(reorderWork.size());
                reorderWorkSize = reorderWork.size();
                for (SizeType i = 0; i < reorderWork.size(); ++i) {
                    Shape start = Service::BroadcastTo5D(reorderWork[i].second, false);
                    Shape end = start + Service::BroadcastTo5D(reorderWork[i].first, true);
                    perThreadParams.push_back({start, end, 0});
                }
            } else {
                Shape start = {0, 0, 0, 0, 0};
                Shape end = reorderedDims;
                perThreadParams.push_back({start, end, 0});
            }
        }
    protected:

        void RunSequential(IntType threadID, const void *src, void *dst, float alpha) const noexcept(false) override
        {
            TypesPara typesPair{srcType, dstType};

            if (EvaluateMap.find(typesPair) != EvaluateMap.end()) {
                auto kerFunc = EvaluateMap.at(typesPair);
                kerFunc(perThreadParams[threadID].threadStart, perThreadParams[threadID].threadEnd, src,
                    dst, reorderedStridesSrc, reorderedStridesDst, alpha);
            }
        }

        using ReorderDefaultFunc = std::function<void(const Shape &, const Shape &, const void *,
            void *, const Shape &, const Shape &, float)>;

        const std::unordered_map<TypesPara, ReorderDefaultFunc, TypesParaHash, TypesParaEqual> EvaluateMap = {
            {{Element::TypeT::F32, Element::TypeT::F32}, Reorder::ReorderDefault<float, float>},
            {{Element::TypeT::F32, Element::TypeT::S32}, Reorder::ReorderDefault<float, std::int32_t>},
            {{Element::TypeT::F32, Element::TypeT::F16}, Reorder::ReorderDefault<float, __fp16>},
            {{Element::TypeT::F32, Element::TypeT::BF16}, Reorder::ReorderDefault<float, __bf16>},
            {{Element::TypeT::F32, Element::TypeT::S8}, Reorder::ReorderDefault<float, int8_t>},
            {{Element::TypeT::F32, Element::TypeT::U8}, Reorder::ReorderDefault<float, uint8_t>},

            {{Element::TypeT::S32, Element::TypeT::F32}, Reorder::ReorderDefault<std::int32_t, float>},
            {{Element::TypeT::S32, Element::TypeT::S32}, Reorder::ReorderDefault<std::int32_t, std::int32_t>},
            {{Element::TypeT::S32, Element::TypeT::F16}, Reorder::ReorderDefault<std::int32_t, __fp16>},
            {{Element::TypeT::S32, Element::TypeT::BF16}, Reorder::ReorderDefault<std::int32_t, __bf16>},
            {{Element::TypeT::S32, Element::TypeT::S8}, Reorder::ReorderDefault<std::int32_t, int8_t>},
            {{Element::TypeT::S32, Element::TypeT::U8}, Reorder::ReorderDefault<std::int32_t, uint8_t>},

            {{Element::TypeT::F16, Element::TypeT::F32}, Reorder::ReorderDefault<__fp16, float>},
            {{Element::TypeT::F16, Element::TypeT::S32}, Reorder::ReorderDefault<__fp16, std::int32_t>},
            {{Element::TypeT::F16, Element::TypeT::F16}, Reorder::ReorderDefault<__fp16, __fp16>},
            {{Element::TypeT::F16, Element::TypeT::BF16}, Reorder::ReorderDefault<__fp16, __bf16>},
            {{Element::TypeT::F16, Element::TypeT::S8}, Reorder::ReorderDefault<__fp16, int8_t>},
            {{Element::TypeT::F16, Element::TypeT::U8}, Reorder::ReorderDefault<__fp16, uint8_t>},

            {{Element::TypeT::BF16, Element::TypeT::F32}, Reorder::ReorderDefault<__bf16, float>},
            {{Element::TypeT::BF16, Element::TypeT::S32}, Reorder::ReorderDefault<__bf16, std::int32_t>},
            {{Element::TypeT::BF16, Element::TypeT::F16}, Reorder::ReorderDefault<__bf16, __fp16>},
            {{Element::TypeT::BF16, Element::TypeT::BF16}, Reorder::ReorderDefault<__bf16, __bf16>},
            {{Element::TypeT::BF16, Element::TypeT::S8}, Reorder::ReorderDefault<__bf16, int8_t>},
            {{Element::TypeT::BF16, Element::TypeT::U8}, Reorder::ReorderDefault<__bf16, uint8_t>},

            {{Element::TypeT::S8, Element::TypeT::F32}, Reorder::ReorderDefault<std::int8_t, float>},
            {{Element::TypeT::S8, Element::TypeT::S32}, Reorder::ReorderDefault<std::int8_t, std::int32_t>},
            {{Element::TypeT::S8, Element::TypeT::F16}, Reorder::ReorderDefault<std::int8_t, __fp16>},
            {{Element::TypeT::S8, Element::TypeT::BF16}, Reorder::ReorderDefault<std::int8_t, __bf16>},
            {{Element::TypeT::S8, Element::TypeT::S8}, Reorder::ReorderDefault<std::int8_t, int8_t>},
            {{Element::TypeT::S8, Element::TypeT::U8}, Reorder::ReorderDefault<std::int8_t, uint8_t>},

            {{Element::TypeT::U8, Element::TypeT::F32}, Reorder::ReorderDefault<std::uint8_t, float>},
            {{Element::TypeT::U8, Element::TypeT::S32}, Reorder::ReorderDefault<std::uint8_t, std::int32_t>},
            {{Element::TypeT::U8, Element::TypeT::F16}, Reorder::ReorderDefault<std::uint8_t, __fp16>},
            {{Element::TypeT::U8, Element::TypeT::BF16}, Reorder::ReorderDefault<std::uint8_t, __bf16>},
            {{Element::TypeT::U8, Element::TypeT::S8}, Reorder::ReorderDefault<std::uint8_t, int8_t>},
            {{Element::TypeT::U8, Element::TypeT::U8}, Reorder::ReorderDefault<std::uint8_t, uint8_t>}
        };

        Element::TypeT srcType;
        Element::TypeT dstType;
        Shape reorderedDims;
        Shape reorderedStridesSrc;
        Shape reorderedStridesDst;

    private:

        SizeType GetOptNumThreads() const noexcept override
        {
            SizeType numElements = reorderedDims.GetTotalDimsSize();
            if (reorderedStridesSrc == reorderedStridesDst) {
                return std::max(SizeType{1}, numElements / 20000); // 20000 is a constant which will be changed
                //                                                    after perf measurements
            } else {
                return std::max(SizeType{1}, numElements / 15000); // 15000 is a constant which will be changed
                //                                                    after perf measurements
            }
        }
    };

    bool Supports(const ReorderInfo &task) const noexcept override
    {
        VERBOSE_PRINT;
        Service::Unused(task);
        // going to Add logic
        return true;
    }
    std::chrono::duration<float, std::milli> EstimateTime(const ReorderInfo  &task,
                                                            IntType nThreads) const noexcept override
    {
        VERBOSE_PRINT;
        Service::Unused(task, nThreads);
        // going to Add logic to estimate execution time of the task using nThreads
        return std::chrono::duration<float, std::milli>{1.0f};
    }
    std::shared_ptr<ImplBase<ReorderInfo>> GetImpl(const ReorderInfo &task, IntType nThreads) const override
    {
        VERBOSE_PRINT;
        return std::make_shared<ImplMainSimpleReorder>(task, nThreads);
    }
};

} // Detail
} // KuDNN

#endif // KUDNN_DEFAULT_IMPL_ANY_ANY_HPP
