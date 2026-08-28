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

#ifndef KUDNN_REORDER_DEF_HPP
#define KUDNN_REORDER_DEF_HPP

#include "vector_wrappers.hpp"
#include "kudnn_reorder_kernels.hpp"
#include "kernels/reorder/kudnn_reorder_helpers.hpp"

namespace KuDNN {
namespace Detail {
namespace Reorder {

template <typename InType, typename OutType, bool AlphaMul>
struct ReorderNLoopDefault final {
    static SizeType LoopInc() noexcept
    {
        return 1;
    }
    static void Do(SizeType loopIdx, SizeType loopRange, const InType *typedSrc,
        OutType *typedDst, SizeType dstStride, SizeType srcIdx, SizeType dstIdx, float alpha) noexcept
    {
        Service::Unused(loopRange);
        if constexpr (AlphaMul) {
            float tmp = Vector::Convert<InType, float>::Do(typedSrc[srcIdx + loopIdx]);
            tmp *= alpha;
            if constexpr (std::is_same_v<OutType, std::int32_t>) {
                tmp = std::min(tmp, REORDER_S32_TO_F32_SAT_CONSTANT);
            }
            typedDst[dstIdx + loopIdx * dstStride] =
                Vector::Convert<float, OutType>::Do(tmp);
        } else {
            typedDst[dstIdx + loopIdx * dstStride] =
                Vector::Convert<InType, OutType>::Do(typedSrc[srcIdx + loopIdx]);
        }
    }
};

template<typename InType, typename OutType>
void ReorderDefault(const Shape &start, const Shape &end, const void *typedSrcDefault, void *typedDstDefault,
    const Shape &reorderedStridesSrc, const Shape &reorderedStridesDst, float alpha) noexcept
{
    const InType *typedSrc = static_cast<const InType *>(typedSrcDefault);
    OutType *typedDst = static_cast<OutType *>(typedDstDefault);
    if (alpha == 1.0f) {
        ReorderCoreLoop(start, end, typedSrc, typedDst, reorderedStridesSrc, reorderedStridesDst,
            ReorderNLoopDefault<InType, OutType, false>{}, alpha);
    } else {
        ReorderCoreLoop(start, end, typedSrc, typedDst, reorderedStridesSrc, reorderedStridesDst,
            ReorderNLoopDefault<InType, OutType, true>{}, alpha);
    }
}

} // Reorder
} // Detail
} // KuDNN

#endif // KUDNN_REORDER_DEF_HPP
