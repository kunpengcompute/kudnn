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

#ifndef KUDNN_FAST_JIT_IMPL_SU8_SU8_512_HPP
#define KUDNN_FAST_JIT_IMPL_SU8_SU8_512_HPP

#include <memory>
#include <vector>
#include <array>
#include <algorithm>

#include <arm_sve.h>

#include "kudnn.hpp"

#include "kudnn_jit.hpp"
#include "kernels/reorder/kudnn_reorder_support_functions.hpp"
#include "kernels/reorder/kudnn_reorder_jit.hpp"
#include "kernels/reorder/kudnn_reorder_helpers.hpp"
#include "kernels/reorder/sve/su8/kudnn_fast_impl_su8_su8_512_s8_s8_transpose_kernels_alpha.hpp"
#include "kernels/reorder/sve/su8/kudnn_fast_impl_su8_su8_512_s8_s8_transpose_kernels.hpp"
#include "kernels/reorder/sve/su8/kudnn_fast_impl_su8_su8_512_s8_u8_transpose_kernels_alpha.hpp"
#include "kernels/reorder/sve/su8/kudnn_fast_impl_su8_su8_512_s8_u8_transpose_kernels.hpp"
#include "kernels/reorder/sve/su8/kudnn_fast_impl_su8_su8_512_u8_s8_transpose_kernels_alpha.hpp"
#include "kernels/reorder/sve/su8/kudnn_fast_impl_su8_su8_512_u8_s8_transpose_kernels.hpp"
#include "kernels/reorder/sve/su8/kudnn_fast_impl_su8_su8_512_u8_u8_transpose_kernels_alpha.hpp"
#include "kernels/reorder/sve/su8/kudnn_fast_impl_su8_su8_512_u8_u8_transpose_kernels.hpp"

#include "kudnn_wrapper_helpers.hpp"
#include "threading_impl.hpp"

namespace KuDNN {
namespace Detail {

template <typename SrcType, typename DstType>
struct MainSchemeFastReorderSU8 : ImplGenerator<ReorderInfo> {
    struct ImplMainFastReorderSU8 : ReorderImplBase {
        ImplMainFastReorderSU8(const ReorderInfo &task, SizeType nThreads) : ReorderImplBase(task, nThreads)
        {
            VERBOSE_PRINT;

            Shape srcStrides = Service::GetShapeAccordingToLayout(task.srcInfo.GetStrides(),
                                                                  task.srcInfo.GetLayout());
            Shape dstStrides = Service::GetShapeAccordingToLayout(task.dstInfo.GetStrides(),
                                                                  task.srcInfo.GetLayout());

            reorderedDims =  Service::BroadcastTo5D(Service::GetShapeAccordingToLayout(task.dstInfo.GetDims(),
                                                                                       task.srcInfo.GetLayout()));
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
            std::vector<std::pair<Shape, Shape>> reorderWork = Threading::SplitDims(reorderedDims, optNumThreads);

            perThreadParams.reserve(reorderWork.size());
            reorderWorkSize = reorderWork.size();

            for (SizeType i = 0; i < reorderWork.size(); ++i) {
                Shape start = Service::BroadcastTo5D(reorderWork[i].second, false);
                Shape end = start + Service::BroadcastTo5D(reorderWork[i].first, true);
                perThreadParams.push_back({start, end, 0});
            }
        }
    protected:
        void RunSequential(IntType threadID, const void *src, void *dst, float alpha) const noexcept(false) override
        {
            SizeType block_size = 128;
            SizeType vl = svcntw();
            VERBOSE_PRINT;
            const SrcType* typedSrc = static_cast<const SrcType*>(src);
            DstType* typedDst = static_cast<DstType*>(dst);
            for (SizeType i0 = perThreadParams[threadID].threadStart[IDX_0];
                i0 < perThreadParams[threadID].threadEnd[IDX_0]; ++i0) {
                for (SizeType i1 = perThreadParams[threadID].threadStart[IDX_1];
                    i1 < perThreadParams[threadID].threadEnd[IDX_1]; ++i1) {
                    for (SizeType i2 = perThreadParams[threadID].threadStart[IDX_2];
                        i2 < perThreadParams[threadID].threadEnd[IDX_2]; ++i2) {
                        for (SizeType ih = perThreadParams[threadID].threadStart[IDX_3];
                            ih < perThreadParams[threadID].threadEnd[IDX_3]; ih += block_size) {
                            SizeType maxIh = std::min(ih + block_size, perThreadParams[threadID].threadEnd[IDX_3]);
                            for (SizeType iw = perThreadParams[threadID].threadStart[IDX_4];
                                iw < perThreadParams[threadID].threadEnd[IDX_4]; iw += block_size) {
                                SizeType maxIw = std::min(iw + block_size, perThreadParams[threadID].threadEnd[IDX_4]);
                                for (SizeType localIh = ih; localIh < maxIh; localIh += vl) {
                                    SizeType residIh = std::min(vl, maxIh - localIh) - 1;
                                    for (SizeType localIw = iw; localIw < maxIw; localIw += vl) {
                                        SizeType residIw = std::min(vl, maxIw - localIw) - 1;
                                        if constexpr (std::is_same_v<SrcType, std::int8_t>) {
                                            if constexpr (std::is_same_v<DstType, std::int8_t>) {
                                                if (alpha == 1.0f) {
                                                    Reorder::TRANSPOSE_REORDER_FUNC_ARR_INT8_INT8[
                                                        + residIh * vl + residIw](
                                                        typedSrc + ((i0 + i1 + i2) * reorderedStridesDst[NUM_4]
                                                            + localIh) * reorderedStridesSrc[NUM_3] + localIw,
                                                        typedDst + ((i0 + i1 + i2) * reorderedStridesSrc[NUM_3]
                                                            + localIw) * reorderedStridesDst[NUM_4] + localIh,
                                                        reorderedStridesSrc[NUM_3], reorderedStridesDst[NUM_4], alpha);
                                                } else {
                                                    Reorder::TRANSPOSE_REORDER_FUNC_ARR_INT8_INT8_ALPHA[
                                                        residIh * vl + residIw](
                                                        typedSrc + ((i0 + i1 + i2) * reorderedStridesDst[NUM_4]
                                                            + localIh) * reorderedStridesSrc[NUM_3] + localIw,
                                                        typedDst + ((i0 + i1 + i2) * reorderedStridesSrc[NUM_3]
                                                            + localIw) * reorderedStridesDst[NUM_4] + localIh,
                                                        reorderedStridesSrc[NUM_3], reorderedStridesDst[NUM_4], alpha);
                                                }
                                            } else {
                                                if (alpha == 1.0f) {
                                                    Reorder::TRANSPOSE_REORDER_FUNC_ARR_INT8_UINT8[
                                                        + residIh * vl + residIw](
                                                        typedSrc + ((i0 + i1 + i2) * reorderedStridesDst[NUM_4]
                                                            + localIh) * reorderedStridesSrc[NUM_3] + localIw,
                                                        typedDst + ((i0 + i1 + i2) * reorderedStridesSrc[NUM_3]
                                                            + localIw) * reorderedStridesDst[NUM_4] + localIh,
                                                        reorderedStridesSrc[NUM_3], reorderedStridesDst[NUM_4], alpha);
                                                } else {
                                                    Reorder::TRANSPOSE_REORDER_FUNC_ARR_INT8_UINT8_ALPHA[
                                                        + residIh * vl + residIw](
                                                        typedSrc + ((i0 + i1 + i2) * reorderedStridesDst[NUM_4]
                                                            + localIh) * reorderedStridesSrc[NUM_3] + localIw,
                                                        typedDst + ((i0 + i1 + i2) * reorderedStridesSrc[NUM_3]
                                                            + localIw) * reorderedStridesDst[NUM_4] + localIh,
                                                        reorderedStridesSrc[NUM_3], reorderedStridesDst[NUM_4], alpha);
                                                }
                                            }
                                        } else {
                                            if constexpr (std::is_same_v<DstType, std::int8_t>) {
                                                if (alpha == 1.0f) {
                                                    Reorder::TRANSPOSE_REORDER_FUNC_ARR_UINT8_INT8[
                                                        + residIh * vl + residIw](
                                                        typedSrc + ((i0 + i1 + i2) * reorderedStridesDst[NUM_4]
                                                            + localIh) * reorderedStridesSrc[NUM_3] + localIw,
                                                        typedDst + ((i0 + i1 + i2) * reorderedStridesSrc[NUM_3]
                                                            + localIw) * reorderedStridesDst[NUM_4] + localIh,
                                                        reorderedStridesSrc[NUM_3], reorderedStridesDst[NUM_4], alpha);
                                                } else {
                                                    Reorder::TRANSPOSE_REORDER_FUNC_ARR_UINT8_INT8_ALPHA[
                                                        + residIh * vl + residIw](
                                                        typedSrc + ((i0 + i1 + i2) * reorderedStridesDst[NUM_4]
                                                            + localIh) * reorderedStridesSrc[NUM_3] + localIw,
                                                        typedDst + ((i0 + i1 + i2) * reorderedStridesSrc[NUM_3]
                                                            + localIw) * reorderedStridesDst[NUM_4] + localIh,
                                                        reorderedStridesSrc[NUM_3], reorderedStridesDst[NUM_4], alpha);
                                                }
                                            } else {
                                                if (alpha == 1.0f) {
                                                    Reorder::TRANSPOSE_REORDER_FUNC_ARR_UINT8_UINT8[
                                                        + residIh * vl + residIw](
                                                        typedSrc + ((i0 + i1 + i2) * reorderedStridesDst[NUM_4]
                                                            + localIh) * reorderedStridesSrc[NUM_3] + localIw,
                                                        typedDst + ((i0 + i1 + i2) * reorderedStridesSrc[NUM_3]
                                                            + localIw) * reorderedStridesDst[NUM_4] + localIh,
                                                        reorderedStridesSrc[NUM_3], reorderedStridesDst[NUM_4], alpha);
                                                } else {
                                                    Reorder::TRANSPOSE_REORDER_FUNC_ARR_UINT8_UINT8_ALPHA[
                                                        + residIh * vl + residIw](
                                                        typedSrc + ((i0 + i1 + i2) * reorderedStridesDst[NUM_4]
                                                            + localIh) * reorderedStridesSrc[NUM_3] + localIw,
                                                        typedDst + ((i0 + i1 + i2) * reorderedStridesSrc[NUM_3]
                                                            + localIw) * reorderedStridesDst[NUM_4] + localIh,
                                                        reorderedStridesSrc[NUM_3], reorderedStridesDst[NUM_4], alpha);
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
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
        Shape reorderedDims;
        Shape reorderedStridesSrc;
        Shape reorderedStridesDst;
    };
    bool Supports(const ReorderInfo &task) const noexcept override
    {
        VERBOSE_PRINT;
#ifdef KUDNNL_920F
            bool srcMatched = (task.srcInfo.GetType() == Element::MatchType<SrcType>());
            bool dstMatched = (task.dstInfo.GetType() == Element::MatchType<DstType>());
            bool isAXBToABX = (task.srcInfo.GetStandardAXBLayout() == task.srcInfo.GetLayout()) &&
                (task.dstInfo.GetStandardABXLayout() == task.dstInfo.GetLayout()) &&
                (task.srcInfo.GetNumDims() > DIM_2) &&
                task.srcInfo.IsDense() && task.dstInfo.IsDense();
            bool isABXToAXB = (task.srcInfo.GetStandardABXLayout() == task.srcInfo.GetLayout()) &&
                (task.dstInfo.GetStandardAXBLayout() == task.dstInfo.GetLayout()) &&
                (task.srcInfo.GetNumDims() > DIM_2) &&
                task.srcInfo.IsDense() && task.dstInfo.IsDense();
            bool isABToBA = (task.srcInfo.GetLayout() == Layout::AB) &&
                (task.dstInfo.GetLayout() == Layout::BA) &&
                task.srcInfo.IsDense() && task.dstInfo.IsDense();
            bool isBAToAB = (task.srcInfo.GetLayout() == Layout::BA) &&
                (task.dstInfo.GetLayout() == Layout::AB) &&
                task.srcInfo.IsDense() && task.dstInfo.IsDense();
            if ((srcMatched && dstMatched) && (isAXBToABX || isABXToAXB || isABToBA || isBAToAB)) {
                return true;
            } else {
                return false;
            }
#else // KUDNNL_920F
            Service::Unused(task);
            return false;
#endif
    }
    std::chrono::duration<float, std::milli> EstimateTime(const ReorderInfo &task,
                                                          IntType nThreads) const noexcept override
    {
        Service::Unused(task, nThreads);
        // going to Add logic to estimate execution time of the task using nThreads
        return std::chrono::duration<float, std::milli>{0.2f};
    }
    std::shared_ptr<ImplBase<ReorderInfo>> GetImpl(const ReorderInfo &task, IntType nThreads) const override
    {
        return std::make_shared<ImplMainFastReorderSU8>(task, nThreads);
    }
};

} // Detail
} // KuDNN

#endif // KUDNN_FAST_JIT_IMPL_SU8_SU8_512_HPP
