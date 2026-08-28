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

#ifndef KUDNN_REORDER_HELPERS_HPP
#define KUDNN_REORDER_HELPERS_HPP

#include <cstdint>

#include "kudnn.hpp"
#include "vector_wrappers.hpp"

namespace KuDNN {
namespace Detail {

// This constant is needed to prevent ONEDNN saturation features
constexpr float REORDER_S32_TO_F32_SAT_CONSTANT = 2147483520.f;
// Hex representation on upper constant
constexpr std::uint32_t REORDER_S32_TO_F32_SAT_CONSTANT_HEX = 0x4effffff;

struct TypesPara {
    Element::TypeT srcType;
    Element::TypeT dstType;
};

struct TypesParaHash {
    SizeType operator()(const TypesPara &paras) const
    {
        std::size_t h1 = std::hash<SizeType>{}(static_cast<SizeType>(paras.srcType));
        std::size_t h2 = std::hash<SizeType>{}(static_cast<SizeType>(paras.dstType));
        return h1 ^ (h2 << 1);
    }
};

struct TypesParaEqual {
    bool operator()(const TypesPara &paras0, const TypesPara &paras1) const
    {
        return paras0.srcType == paras1.srcType &&
               paras0.dstType == paras1.dstType;
    }
};

struct ReorderInfo final {
    struct RunTimeParams final {
        const void* src;
        void* dst;
        float alpha;
    };
    ReorderInfo(const TensorInfo &srcInfoPara, const TensorInfo &dstInfoPara) noexcept(false)
        : srcInfo(srcInfoPara), dstInfo(dstInfoPara){ }
    TensorInfo srcInfo;
    TensorInfo dstInfo;
};

inline bool operator==(const ReorderInfo &lhs, const ReorderInfo &rhs) noexcept
{
    return ((lhs.srcInfo == rhs.srcInfo) && (lhs.dstInfo == rhs.dstInfo));
}

template <typename InType, typename OutType, typename ReorderNLoop>
void ReorderCoreLoop(const Shape &start, const Shape &end, const InType *typedSrc, OutType *typedDst,
    const Shape &reorderedStridesSrc, const Shape &reorderedStridesDst, ReorderNLoop &&reorderParams,
    float alpha) noexcept
    {
    for (SizeType a = start[IDX_0]; a < end[IDX_0]; a++) {
        SizeType srcIdx0 = a * reorderedStridesSrc[IDX_0];
        SizeType dstIdx0 = a * reorderedStridesDst[IDX_0];
        for (SizeType b = start[IDX_1]; b < end[IDX_1]; b++) {
            SizeType srcIdx1 = b * reorderedStridesSrc[IDX_1];
            SizeType dstIdx1 = b * reorderedStridesDst[IDX_1];
            for (SizeType c = start[IDX_2]; c < end[IDX_2]; c++) {
                SizeType srcIdx2 = c * reorderedStridesSrc[IDX_2];
                SizeType dstIdx2 = c * reorderedStridesDst[IDX_2];
                for (SizeType d = start[IDX_3]; d < end[IDX_3]; d++) {
                    SizeType srcIdx = srcIdx0 + srcIdx1 + srcIdx2 + d * reorderedStridesSrc[IDX_3];
                    SizeType dstIdx = dstIdx0 + dstIdx1 + dstIdx2 + d * reorderedStridesDst[IDX_3];
                    Service::UnrollFor::Do<NUM_1>(start[IDX_4], end[IDX_4], std::forward<ReorderNLoop>(reorderParams),
                        typedSrc, typedDst, reorderedStridesDst[IDX_4], srcIdx, dstIdx, alpha);
                }
            }
        }
    }
}

} // Detail
} // KuDNN

#endif // KUDNN_REORDER_HELPERS_HPP
