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
#include <ratio>
#include <chrono>

#include "kudnn.hpp"

namespace KuDNN {
namespace Detail {

enum class PackingType {
    PACK_SRC,
    PACK_WEI
};

enum class MatMulDstFlags : std::int64_t {
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
    ADD_BIAS_MN = 0x400uLL,
    APPLY_POST_OPS = 0x800uLL,
    FIRST_K_ITER = 0x1000uLL
};

inline MatMulDstFlags operator | (const MatMulDstFlags lhs, const MatMulDstFlags rhs)
{
    return static_cast<MatMulDstFlags>(static_cast<std::uint64_t>(lhs) | static_cast<std::uint64_t>(rhs));
}

inline MatMulDstFlags operator & (const MatMulDstFlags lhs, const MatMulDstFlags rhs)
{
    return static_cast<MatMulDstFlags>(static_cast<std::uint64_t>(lhs) & static_cast<std::uint64_t>(rhs));
}

inline MatMulDstFlags operator << (const MatMulDstFlags lhs, IntType shift)
{
    return static_cast<MatMulDstFlags>(static_cast<std::uint64_t>(lhs) << shift);
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
        PostOpsDataPtrs postOpsPtrs;
        float alpha;
        float beta;
    };
    GemmInfo(const TensorInfo &srcInfoPara, const TensorInfo &weiInfoPara, const TensorInfo &dstInfoPara,
        const TensorInfo &biaInfoPara, const Attributes &attributesPara) noexcept(false)
        : srcInfo(srcInfoPara), weiInfo(weiInfoPara), dstInfo(dstInfoPara), biaInfo(biaInfoPara),
        attributes(attributesPara) {}
    TensorInfo srcInfo;
    TensorInfo weiInfo;
    TensorInfo dstInfo;
    TensorInfo biaInfo;
    Attributes attributes;
};

inline bool operator==(const GemmInfo &lhs, const GemmInfo &rhs) noexcept
{
    return ((lhs.srcInfo == rhs.srcInfo) && (lhs.weiInfo == rhs.weiInfo) &&
        (lhs.dstInfo == rhs.dstInfo) && (lhs.biaInfo == rhs.biaInfo) &&
        (lhs.attributes == rhs.attributes));
}

constexpr const float MEM_COPY_BANDWIDTH = 2.5e6;    // Bytes/ms

// estimates reorder time on single thread
template <typename MatrixType>
std::chrono::duration<float, std::milli> CalculateReorderTime(const KuDNN::TensorInfo &tensorInfo)
{
    return std::chrono::duration<float, std::milli>{tensorInfo.GetTotalTensorSize() *
        sizeof(MatrixType) / MEM_COPY_BANDWIDTH};
};

// estimates reorder time on single thread
template <typename SrcType, typename WeiType, typename DstType, typename BiaType>
std::chrono::duration<float, std::milli> CalculateReorderTimeForMatmul(const GemmInfo &task)
{
    std::chrono::duration<float, std::milli> reorderTime = std::chrono::milliseconds(0);
    if (task.srcInfo.GetType() != Element::MatchType<SrcType>()) {
        reorderTime += CalculateReorderTime<SrcType>(task.srcInfo);
    }
    if (task.weiInfo.GetType() != Element::MatchType<WeiType>()) {
        reorderTime += CalculateReorderTime<WeiType>(task.weiInfo);
    }
    if (task.dstInfo.GetType() != Element::MatchType<DstType>()) {
        reorderTime += CalculateReorderTime<DstType>(task.dstInfo) * NUM_2; // Dst reorders 2 times
    }
    if (task.biaInfo.GetType() != Element::MatchType<BiaType>()) {
        reorderTime += CalculateReorderTime<BiaType>(task.biaInfo);
    }
    return reorderTime;
}

inline double CalculateWorkSize(const GemmInfo &task)
{
    SizeType batchDims = task.dstInfo.GetNumDims() - NUM_2; // 2 dimensions of matmul
    double batchSize = 1;
    for (SizeType i = 0; i < batchDims; ++i) {
        batchSize *= task.dstInfo.GetDims()[i];
    }
    return batchSize * task.srcInfo.GetDims()[task.srcInfo.GetNumDims() - NUM_2] *
            task.weiInfo.GetDims()[task.weiInfo.GetNumDims() - NUM_1] *
            (NUM_2 * task.srcInfo.GetDims()[task.srcInfo.GetNumDims() - NUM_1] - NUM_1);
}

template <typename SrcType, typename WeiType, typename DstType, typename BiaType>
std::chrono::duration<float, std::milli> CalculateMatmulTimeWithReorder(
    float singleThreadGFlops, const GemmInfo &task, SizeType nThreads)
{
    std::chrono::duration<float, std::milli> reorderTime =
        CalculateReorderTimeForMatmul<SrcType, WeiType, DstType, BiaType>(task);
    // expect linear thread scalability
    return (std::chrono::duration<float, std::milli>{CalculateWorkSize(task) /
        (singleThreadGFlops) / std::mega::num} + reorderTime) / nThreads;
}

template <typename HelperSrcType, typename HelperWeiType, typename HelperDstType, typename HelperBiaType>
struct MatmulTypeHelper {
    using SrcType = HelperSrcType;
    using WeiType = HelperWeiType;
    using DstType = HelperDstType;
    using BiaType = HelperBiaType;
    constexpr static SizeType GetSrcSize() noexcept
    {
        return sizeof(SrcType);
    }
    constexpr static Element::TypeT GetSrcType() noexcept
    {
        return Element::MatchType<SrcType>();
    }
    constexpr static SizeType GetWeiSize() noexcept
    {
        return sizeof(WeiType);
    }
    constexpr static Element::TypeT GetWeiType() noexcept
    {
        return Element::MatchType<WeiType>();
    }
    constexpr static SizeType GetDstSize() noexcept
    {
        return sizeof(DstType);
    }
    constexpr static Element::TypeT GetDstType() noexcept
    {
        return Element::MatchType<DstType>();
    }
    constexpr static SizeType GetBiaSize() noexcept
    {
        return sizeof(BiaType);
    }
    constexpr static Element::TypeT GetBiaType() noexcept
    {
        return Element::MatchType<BiaType>();
    }
};

} // Detail
} // KuDNN

#endif // KUDNN_GEMM_HELPERS_HPP
