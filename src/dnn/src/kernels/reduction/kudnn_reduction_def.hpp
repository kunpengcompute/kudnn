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

#ifndef KUDNN_REDUCTION_DEF_HPP
#define KUDNN_REDUCTION_DEF_HPP

#include <algorithm>
#include <arm_sve.h>
#include <cmath>

#include "kudnn.hpp"

#include "vector_wrappers.hpp"
#include "kernels/reduction/kudnn_reduction_def_ones.hpp"
#include "kernels/reduction/kudnn_reduction_kernels.hpp"
#include "kernels/reduction/kudnn_reduction_sve512_norm.hpp"

namespace KuDNN {
namespace Detail {
namespace Reduction {

template <typename OutType>
static void EvaluateRefStore(const Shape &start, const Shape &end, OutType *dst, ReductionFunction kind,
    Shape &stridesDst, SizeType reduceSize, float* dstTmp) noexcept
{
    for (SizeType j0 = start[IDX_0]; j0 < end[IDX_0]; ++j0) {
        SizeType idxDst0 = j0 * stridesDst[IDX_0];
        for (SizeType j1 = start[IDX_1]; j1 < end[IDX_1]; ++j1) {
            SizeType idxDst1 = j1 * stridesDst[IDX_1];
            for (SizeType j2 = start[IDX_2]; j2 < end[IDX_2]; ++j2) {
                SizeType idxDst2 = j2 * stridesDst[IDX_2];
                for (SizeType j3 = start[IDX_3]; j3 < end[IDX_3]; ++j3) {
                    SizeType idxDst3 = j3 * stridesDst[IDX_3];
                    for (SizeType j4 = start[IDX_4]; j4 < end[IDX_4]; ++j4) {
                        SizeType dstIdx = idxDst0 + idxDst1 + idxDst2 + idxDst3 + j4 * stridesDst[IDX_4];
                        if (kind == ReductionFunction::MEAN) {
                            dst[dstIdx] = Vector::Convert<float, OutType>::Do(dstTmp[dstIdx] / reduceSize);
                        } else if constexpr (!std::is_same_v<float, OutType>) {
                            dst[dstIdx] = Vector::Convert<float, OutType>::Do(dstTmp[dstIdx]);
                        }
                    }
                }
            }
        }
    }
}

template <typename CompType, typename InType, typename OutType, typename ScalarOp>
KUDNN_INLINE inline void EvaluateRef(const Shape &start, const Shape &end, const InType *src, OutType *dst,
    float* dstBuf, ScalarOp &&scOp, CompType acc, ReductionFunction kind, SizeType reduceSize,
    Shape &dimsSrc, Shape &dimsDst, Shape &stridesSrc, Shape &stridesDst) noexcept
{
    Service::Unused(acc);
    float* dstTmp = nullptr;
    if constexpr (std::is_same_v<float, OutType>) {
        dstTmp = static_cast<float*>(dst);
    } else {
        dstTmp = dstBuf;
    }
    if (dstTmp == nullptr) {
        return;
    }

    std::array<SizeType, MAX_DIMS> dims;
    for (SizeType i = 0; i < dimsDst.GetNumDims(); ++i) {
        dims[i] = (dimsDst[i] == 1) ? dimsSrc[i] : end[i];
    }

    for (SizeType j0 = start[IDX_0]; j0 < dims[IDX_0]; ++j0) {
        SizeType idxSrc0 = j0 * stridesSrc[IDX_0];
        SizeType idxDst0 = j0 * stridesDst[IDX_0];
        for (SizeType j1 = start[IDX_1]; j1 < dims[IDX_1]; ++j1) {
            SizeType idxSrc1 = j1 * stridesSrc[IDX_1];
            SizeType idxDst1 = j1 * stridesDst[IDX_1];
            for (SizeType j2 = start[IDX_2]; j2 < dims[IDX_2]; ++j2) {
                SizeType idxSrc2 = j2 * stridesSrc[IDX_2];
                SizeType idxDst2 = j2 * stridesDst[IDX_2];
                for (SizeType j3 = start[IDX_3]; j3 < dims[IDX_3]; ++j3) {
                    SizeType idxSrc3 = j3 * stridesSrc[IDX_3];
                    SizeType idxDst3 = j3 * stridesDst[IDX_3];
                    for (SizeType j4 = start[IDX_4]; j4 < dims[IDX_4]; ++j4) {
                        SizeType idxSrc = idxSrc0 + idxSrc1 + idxSrc2 + idxSrc3 + j4 * stridesSrc[IDX_4];
                        SizeType idxDst = idxDst0 + idxDst1 + idxDst2 + idxDst3 + j4 * stridesDst[IDX_4];
                        dstTmp[idxDst] = scOp(Vector::Convert<InType, float>::Do(src[idxSrc]), dstTmp[idxDst]);
                    }
                }
            }
        }
    }

    EvaluateRefStore(start, end, dst, kind, stridesDst, reduceSize, dstTmp);
}

template <typename OutType>
static void EvaluateNormStore(const Shape &start, const Shape &end, OutType *dst, ReductionFunction kind,
    Shape &stridesDst, float p, float eps, float* dstTmp) noexcept
{
    for (SizeType i0 = start[IDX_0]; i0 < end[IDX_0]; ++i0) {
        SizeType idxDst0 = i0 * stridesDst[IDX_0];
        for (SizeType i1 = start[IDX_1]; i1 < end[IDX_1]; ++i1) {
            SizeType idxDst1 = i1 * stridesDst[IDX_1];
            for (SizeType i2 = start[IDX_2]; i2 < end[IDX_2]; ++i2) {
                SizeType idxDst2 = i2 * stridesDst[IDX_2];
                for (SizeType i3 = start[IDX_3]; i3 < end[IDX_3]; ++i3) {
                    SizeType idxDst3 = i3 * stridesDst[IDX_3];
                    for (SizeType i4 = start[IDX_4]; i4 < end[IDX_4]; ++i4) {
                        SizeType idxDst = idxDst0 + idxDst1 + idxDst2 + idxDst3 + i4 * stridesDst[IDX_4];
                        if (kind == ReductionFunction::NORM_LP_SUM) {
                            dst[idxDst] = Vector::Convert<float, OutType>::Do(
                                std::pow(dstTmp[idxDst] + eps, 1.0f / p));
                        } else if (kind == ReductionFunction::NORM_LP_MAX) {
                            dst[idxDst] = Vector::Convert<float, OutType>::Do(
                                std::pow(std::max(dstTmp[idxDst], eps), 1.0f / p));
                        } else if (kind == ReductionFunction::NORM_LP_POWER_P_SUM) {
                            dst[idxDst] = Vector::Convert<float, OutType>::Do(dstTmp[idxDst] + eps);
                        } else if (kind == ReductionFunction::NORM_LP_POWER_P_MAX) {
                            dst[idxDst] = Vector::Convert<float, OutType>::Do(std::max(dstTmp[idxDst], eps));
                        }
                    }
                }
            }
        }
    }
}

template <typename CompType, typename InType, typename OutType>
KUDNN_INLINE inline void EvaluateNorm(const Shape &start, const Shape &end, const InType *src, OutType *dst,
    float* dstBuf, CompType acc, ReductionFunction kind, Shape &dimsSrc, Shape &dimsDst,
    Shape &stridesSrc, Shape &stridesDst, float p, float eps) noexcept
{
    Service::Unused(acc);
    std::array<SizeType, MAX_DIMS> dims;
    for (SizeType i = 0; i < dimsDst.GetNumDims(); ++i) {
        dims[i] = (dimsDst[i] == 1) ? dimsSrc[i] : end[i];
    }

    float* dstTmp = nullptr;
    if constexpr (std::is_same_v<float, OutType>) {
        dstTmp = static_cast<float*>(dst);
    } else {
        dstTmp = dstBuf;
    }

    for (SizeType i0 = start[IDX_0]; i0 < dims[IDX_0]; ++i0) {
        SizeType idxSrc0 = i0 * stridesSrc[IDX_0];
        SizeType idxDst0 = i0 * stridesDst[IDX_0];
        for (SizeType i1 = start[IDX_1]; i1 < dims[IDX_1]; ++i1) {
            SizeType idxSrc1 = i1 * stridesSrc[IDX_1];
            SizeType idxDst1 = i1 * stridesDst[IDX_1];
            for (SizeType i2 = start[IDX_2]; i2 < dims[IDX_2]; ++i2) {
                SizeType idxSrc2 = i2 * stridesSrc[IDX_2];
                SizeType idxDst2 = i2 * stridesDst[IDX_2];
                for (SizeType i3 = start[IDX_3]; i3 < dims[IDX_3]; ++i3) {
                    SizeType idxSrc3 = i3 * stridesSrc[IDX_3];
                    SizeType idxDst3 = i3 * stridesDst[IDX_3];
                    for (SizeType i4 = start[IDX_4]; i4 < dims[IDX_4]; ++i4) {
                        SizeType idxSrc = idxSrc0 + idxSrc1 + idxSrc2 + idxSrc3 + i4 * stridesSrc[IDX_4];
                        SizeType idxDst = idxDst0 + idxDst1 + idxDst2 + idxDst3 + i4 * stridesDst[IDX_4];
                        dstTmp[idxDst] += std::pow(std::abs(Vector::Convert<InType, float>::Do(src[idxSrc])), p);
                    }
                }
            }
        }
    }

    EvaluateNormStore(start, end, dst, kind, stridesDst, p, eps, dstTmp);
}

template <typename CompType, typename InType, typename OutType>
KUDNN_INLINE inline void EvaluateWrapper(const Shape &start, const Shape &end,
    const InType *src, OutType *dst, float* dstBuf,
    CompType acc, ReductionFunction kind,
    Shape &dimsSrc, Shape &dimsDst, Shape &stridesSrc, Shape &stridesDst, float p = 1, float eps = 0) noexcept
{
#ifdef KUDNNL_920F
    if (dimsDst[dimsDst.GetNumDims() - 1] == 1) {
        EvaluateNormOnes(start, end, src, dst, dstBuf, acc, kind, p, eps, dimsSrc, dimsDst, stridesSrc, stridesDst);
    } else {
        EvaluateNorm(start, end, src, dst, dstBuf, acc, kind, dimsSrc, dimsDst, stridesSrc, stridesDst, p, eps);
    }
#else
    EvaluateNorm(start, end, src, dst, dstBuf, acc, kind, dimsSrc, dimsDst, stridesSrc, stridesDst, p, eps);
#endif
}


template <typename CompType, typename InType, typename OutType, typename ScalarOp>
KUDNN_INLINE inline void EvaluateWrapper(const Shape &start, const Shape &end,
    const InType *src, OutType *dst, float* dstBuf, ScalarOp &&scOp, ReductionOp op,
    CompType acc, ReductionFunction kind, SizeType reduceSize, SizeType idleSize,
    Shape &dimsSrc, Shape &dimsDst, Shape &stridesSrc, Shape &stridesDst) noexcept
{
    if (dimsDst[dimsDst.GetNumDims() - 1] == 1) {
        if (idleSize == 1) {
            EvaluateScalar(start, end, src, dst, dstBuf, std::forward<ScalarOp>(scOp),
                op, acc, kind, reduceSize);
        } else {
            EvaluateOnes(start, end, src, dst, dstBuf, std::forward<ScalarOp>(scOp),
                op, acc, kind, reduceSize,
                dimsSrc, dimsDst, stridesSrc, stridesDst);
        }
    } else {
        EvaluateRef(start, end, src, dst, dstBuf, std::forward<ScalarOp>(scOp), acc, kind, reduceSize,
                    dimsSrc, dimsDst, stridesSrc, stridesDst);
    }
}

template <typename CompType, typename InType, typename OutType>
void EvaluateMin(const Shape &start, const Shape &end, const InType *src, OutType *dst, float* dstBuf,
    CompType acc, ReductionFunction kind, SizeType reduceSize, SizeType idleSize,
    Shape &reorderedDimsSrc, Shape &reorderedDimsDst,
    Shape &reorderedStridesSrc, Shape &reorderedStridesDst, float p, float eps) noexcept
{
    Service::Unused(p, eps);
    Reduction::EvaluateWrapper(start, end, src, dst, dstBuf, [] (float a, float b) {return std::min(a, b);},
        ReductionOp::MIN, acc, kind, reduceSize,
        idleSize, reorderedDimsSrc, reorderedDimsDst, reorderedStridesSrc, reorderedStridesDst);
};

template <typename CompType, typename InType, typename OutType>
void EvaluateMax(const Shape &start, const Shape &end, const InType *src, OutType *dst, float* dstBuf,
    CompType acc, ReductionFunction kind, SizeType reduceSize, SizeType idleSize,
    Shape &reorderedDimsSrc, Shape &reorderedDimsDst,
    Shape &reorderedStridesSrc, Shape &reorderedStridesDst, float p, float eps) noexcept
{
    Service::Unused(p, eps);
    Reduction::EvaluateWrapper(start, end, src, dst, dstBuf, [] (float a, float b) {return std::max(a, b);},
        ReductionOp::MAX, acc, kind, reduceSize,
        idleSize, reorderedDimsSrc, reorderedDimsDst, reorderedStridesSrc, reorderedStridesDst);
};

template <typename CompType, typename InType, typename OutType>
void EvaluateMean(const Shape &start, const Shape &end, const InType *src, OutType *dst, float* dstBuf,
    CompType acc, ReductionFunction kind, SizeType reduceSize, SizeType idleSize, Shape &reorderedDimsSrc,
    Shape &reorderedDimsDst, Shape &reorderedStridesSrc, Shape &reorderedStridesDst, float p, float eps) noexcept
{
    Service::Unused(p, eps);
    Reduction::EvaluateWrapper(start, end, src, dst, dstBuf, std::plus<float>(),
        ReductionOp::MEAN, acc, kind, reduceSize,
        idleSize, reorderedDimsSrc, reorderedDimsDst, reorderedStridesSrc, reorderedStridesDst);
};

template <typename CompType, typename InType, typename OutType>
void EvaluateSum(const Shape &start, const Shape &end, const InType *src, OutType *dst, float* dstBuf,
    CompType acc, ReductionFunction kind, SizeType reduceSize, SizeType idleSize, Shape &reorderedDimsSrc,
    Shape &reorderedDimsDst, Shape &reorderedStridesSrc, Shape &reorderedStridesDst, float p, float eps) noexcept
{
    Service::Unused(p, eps);
    Reduction::EvaluateWrapper(start, end, src, dst, dstBuf, std::plus<float>(),
        ReductionOp::SUM, acc, kind, reduceSize,
        idleSize, reorderedDimsSrc, reorderedDimsDst, reorderedStridesSrc, reorderedStridesDst);
};

template <typename CompType, typename InType, typename OutType>
void EvaluateMul(const Shape &start, const Shape &end, const InType *src, OutType *dst, float* dstBuf,
    CompType acc, ReductionFunction kind, SizeType reduceSize, SizeType idleSize,
    Shape &reorderedDimsSrc, Shape &reorderedDimsDst, Shape &reorderedStridesSrc,
    Shape &reorderedStridesDst, float p, float eps) noexcept
{
    Service::Unused(p, eps);
    Reduction::EvaluateWrapper(start, end, src, dst, dstBuf, std::multiplies<float>(),
        ReductionOp::MUL, acc, kind, reduceSize,
        idleSize, reorderedDimsSrc, reorderedDimsDst, reorderedStridesSrc, reorderedStridesDst);
};

template <typename CompType, typename InType, typename OutType>
void EvaluateNormLp(const Shape &start, const Shape &end, const InType *src, OutType *dst, float* dstBuf,
    CompType acc, ReductionFunction kind, SizeType reduceSize, SizeType idleSize,
    Shape &reorderedDimsSrc, Shape &reorderedDimsDst,
    Shape &reorderedStridesSrc, Shape &reorderedStridesDst, float p, float eps) noexcept
{
    Service::Unused(reduceSize);
    Service::Unused(idleSize);
    Reduction::EvaluateWrapper(start, end, src, dst, dstBuf, acc, kind,
        reorderedDimsSrc, reorderedDimsDst,
        reorderedStridesSrc, reorderedStridesDst, p, eps);
};

} // Reduction
} // Detail
} // KuDNN

#endif //  KUDNN_REDUCTION_DEF_HPP
