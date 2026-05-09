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

#ifndef KUDNN_GEMM_HELPERS_HPP
#define KUDNN_GEMM_HELPERS_HPP

#include <cstdint>

#include "kudnn.hpp"

namespace KuDNN {
namespace Detail {

enum class DstLoadFlags : std::int64_t {
    FILL_C_WITH_ZEROZ = 0x1uLL,
    LOAD_C = 0x2uLL,
    MULL_C = 0x4uLL,
    LOAD_BIAS_1 = 0x8uLL,
    LOAD_BIAS_M = 0x10uLL,
    LOAD_BIAS_N = 0x20uLL,
    LOAD_BIAS_MN = 0x40uLL,
    ADD_BIAS_1 = 0x80uLL,
    ADD_BIAS_M = 0x100uLL,
    ADD_BIAS_N = 0x200uLL,
    ADD_BIAS_MN = 0x400uLL
};

inline DstLoadFlags operator|(const DstLoadFlags lhs, const DstLoadFlags rhs)
{
    return static_cast<DstLoadFlags>(static_cast<std::uint64_t>(lhs) | static_cast<std::uint64_t>(rhs));
}

inline DstLoadFlags operator&(const DstLoadFlags lhs, const DstLoadFlags rhs)
{
    return static_cast<DstLoadFlags>(static_cast<std::uint64_t>(lhs) & static_cast<std::uint64_t>(rhs));
}

inline DstLoadFlags operator<<(const DstLoadFlags lhs, IntType shift)
{
    return static_cast<DstLoadFlags>(static_cast<std::uint64_t>(lhs) << shift);
}

// checks if layout has last logical dimension with fastest changing index
bool IsLayoutAlikeAB(Layout layout) noexcept;

// checks if layout has last but one logical dimension with fastest changing index
bool IsLayoutAlikeBA(Layout layout) noexcept;

SizeType GetLayoutDiemensionality(Layout layout) noexcept;

Layout GetPlaneLayoutAB(Layout layout) noexcept;

Layout GetPlaneLayoutBA(Layout layout) noexcept;

struct GemmInfo final {
    struct RunTimeParams final {
        const void *src;
        const void *wei;
        void *dst;
        const void *bia;
        float alpha;
        float beta;
    };
    GemmInfo(const TensorInfo &srcInfoPara, const TensorInfo &weiInfoPara, const TensorInfo &dstInfoPara,
             const TensorInfo &biaInfoPara) noexcept(false)
        : srcInfo(srcInfoPara), weiInfo(weiInfoPara), dstInfo(dstInfoPara), biaInfo(biaInfoPara)
    {
    }
    TensorInfo srcInfo;
    TensorInfo weiInfo;
    TensorInfo dstInfo;
    TensorInfo biaInfo;
};

inline bool operator==(const GemmInfo &lhs, const GemmInfo &rhs) noexcept
{
    return ((lhs.srcInfo == rhs.srcInfo) && (lhs.weiInfo == rhs.weiInfo) && (lhs.dstInfo == rhs.dstInfo) &&
            (lhs.biaInfo == rhs.biaInfo));
}

} // namespace Detail
} // namespace KuDNN

#endif // KUDNN_GEMM_HELPERS_HPP
