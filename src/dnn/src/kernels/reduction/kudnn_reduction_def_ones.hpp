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

#ifndef KUDNN_REDUCTION_DEF_ONES_HPP
#define KUDNN_REDUCTION_DEF_ONES_HPP

#include <algorithm>
#include <arm_sve.h>
#include <cmath>

#include "kudnn.hpp"

#include "vector_wrappers.hpp"
#include "kernels/reduction/kudnn_reduction_kernels.hpp"
#include "kernels/reduction/kudnn_reduction_sve512_norm.hpp"

namespace KuDNN {
namespace Detail {
namespace Reduction {

enum class ReductionOp {
    MAX,
    MIN,
    SUM,
    MEAN,
    MUL
};

template <typename CompType, typename InType, typename OutType>
KUDNN_INLINE inline void ApplyReductionOpNeon(
    svbool_t oppg0, svbool_t oppg1, svbool_t oppg2, svbool_t oppg3,
    typename Vector::Vec<CompType, ARCH::NEON>::Type& res0, typename Vector::Vec<CompType, ARCH::NEON>::Type& res1,
    typename Vector::Vec<CompType, ARCH::NEON>::Type& res2, typename Vector::Vec<CompType, ARCH::NEON>::Type& res3,
    typename Vector::Vec<CompType, ARCH::NEON>::Type& data0, typename Vector::Vec<CompType, ARCH::NEON>::Type& data1,
    typename Vector::Vec<CompType, ARCH::NEON>::Type& data2, typename Vector::Vec<CompType, ARCH::NEON>::Type& data3,
    ReductionOp op
) {
    switch (op) {
        case ReductionOp::MAX:
            Vector::MaxMN<CompType, Vector::GetUsedRegsN<CompType, InType, OutType>(), ARCH::NEON>::Do(
                oppg0, oppg1, oppg2, oppg3, res0, res1, res2, res3,
                data0, data1, data2, data3, res0, res1, res2, res3);
            break;
        case ReductionOp::MIN:
            Vector::MinMN<CompType, Vector::GetUsedRegsN<CompType, InType, OutType>(), ARCH::NEON>::Do(
                oppg0, oppg1, oppg2, oppg3, res0, res1, res2, res3,
                data0, data1, data2, data3, res0, res1, res2, res3);
            break;
        case ReductionOp::MUL:
            Vector::MulMN<CompType, Vector::GetUsedRegsN<CompType, InType, OutType>(), ARCH::NEON>::Do(
                oppg0, oppg1, oppg2, oppg3, res0, res1, res2, res3,
                data0, data1, data2, data3, res0, res1, res2, res3);
            break;
        default:
            Vector::AddMN<CompType, Vector::GetUsedRegsN<CompType, InType, OutType>(), ARCH::NEON>::Do(
                oppg0, oppg1, oppg2, oppg3, res0, res1, res2, res3,
                data0, data1, data2, data3, res0, res1, res2, res3);
            break;
    }
}

template <typename CompType, typename InType, typename OutType>
KUDNN_INLINE inline void ApplyReductionOpVC(
    svbool_t oppg0, svbool_t oppg1, svbool_t oppg2, svbool_t oppg3,
    typename Vector::Vec<CompType, ARCH::VC>::Type& res0, typename Vector::Vec<CompType, ARCH::VC>::Type& res1,
    typename Vector::Vec<CompType, ARCH::VC>::Type& res2, typename Vector::Vec<CompType, ARCH::VC>::Type& res3,
    typename Vector::Vec<CompType, ARCH::VC>::Type& data0, typename Vector::Vec<CompType, ARCH::VC>::Type& data1,
    typename Vector::Vec<CompType, ARCH::VC>::Type& data2, typename Vector::Vec<CompType, ARCH::VC>::Type& data3,
    ReductionOp op
) {
    switch (op) {
        case ReductionOp::MAX:
            Vector::MaxMN<CompType, Vector::GetUsedRegsN<CompType, InType, OutType>(), ARCH::VC>::Do(
                oppg0, oppg1, oppg2, oppg3, res0, res1, res2, res3,
                data0, data1, data2, data3, res0, res1, res2, res3);
            break;
        case ReductionOp::MIN:
            Vector::MinMN<CompType, Vector::GetUsedRegsN<CompType, InType, OutType>(), ARCH::VC>::Do(
                oppg0, oppg1, oppg2, oppg3, res0, res1, res2, res3,
                data0, data1, data2, data3, res0, res1, res2, res3);
            break;
        case ReductionOp::MUL:
            Vector::MulMN<CompType, Vector::GetUsedRegsN<CompType, InType, OutType>(), ARCH::VC>::Do(
                oppg0, oppg1, oppg2, oppg3, res0, res1, res2, res3,
                data0, data1, data2, data3, res0, res1, res2, res3);
            break;
        default:
            Vector::AddMN<CompType, Vector::GetUsedRegsN<CompType, InType, OutType>(), ARCH::VC>::Do(
                oppg0, oppg1, oppg2, oppg3, res0, res1, res2, res3,
                data0, data1, data2, data3, res0, res1, res2, res3);
            break;
    }
}

template <typename CompType, typename InType, typename OutType, typename ScalarOp>
KUDNN_INLINE inline void EvaluateOnes(const Shape &start, const Shape &end, const InType *src, OutType *dst,
    float* dstBuf, ScalarOp &&scOp, ReductionOp op, CompType acc, ReductionFunction kind,
    SizeType reduceSize, Shape &dimsSrc, Shape &dimsDst, Shape &stridesSrc, Shape &stridesDst) noexcept
{
    float* dstTmp = nullptr;
    if constexpr (!std::is_same_v<float, OutType>) {
        dstTmp = dstBuf;
    } else {
        dstTmp = static_cast<float*>(dst);
    }

    std::array<SizeType, MAX_DIMS> dims;
    for (SizeType i = 0; i < dimsDst.GetNumDims(); ++i) {
        dims[i] = (dimsDst[i] == 1) ? dimsSrc[i] : end[i];
    }

#if USE_NEON
    printf("EvaluateOnes USE_NEON!!\n");
    SizeType loopInc = NUM_2 * Vector::NeonLanes<CompType>();
#else
    SizeType loopInc = Vector::UnrollLength<InType, CompType, OutType>();
#endif

    for (SizeType k0 = start[IDX_0]; k0 < dims[IDX_0]; ++k0) {
        SizeType idxSrc0 = k0 * stridesSrc[IDX_0];
        SizeType idxDst0 = k0 * stridesDst[IDX_0];
        for (SizeType k1 = start[IDX_1]; k1 < dims[IDX_1]; ++k1) {
            SizeType idxSrc1 = k1 * stridesSrc[IDX_1];
            SizeType idxDst1 = k1 * stridesDst[IDX_1];
            for (SizeType k2 = start[IDX_2]; k2 < dims[IDX_2]; ++k2) {
                SizeType idxSrc2 = k2 * stridesSrc[IDX_2];
                SizeType idxDst2 = k2 * stridesDst[IDX_2];
                for (SizeType k3 = start[IDX_3]; k3 < dims[IDX_3]; ++k3) {
                    SizeType idxSrc = idxSrc0 + idxSrc1 + idxSrc2 + k3 * stridesSrc[IDX_3];
                    SizeType idxDst = idxDst0 + idxDst1 + idxDst2 + k3 * stridesDst[IDX_3];

#if USE_NEON
                    typename Vector::Vec<CompType, ARCH::NEON>::Type res0, res1, res2, res3;
                    typename Vector::Vec<CompType, ARCH::VC>::Type res4, res5, res6, res7;
                    Vector::DupFloatN<InType, CompType, Vector::GetUsedRegsN<InType, CompType, OutType>(),
                        ARCH::NEON>::Do(acc, res0, res1, res2, res3);
                    Vector::DupFloatN<InType, CompType, Vector::GetUsedRegsN<InType, CompType, OutType>(),
                        ARCH::VC>::Do(acc, res4, res5, res6, res7);

                    SizeType innerSize = dims[IDX_4] - start[IDX_4];
                    SizeType neonLoop = innerSize - (innerSize % loopInc);
                    SizeType tailLoop = innerSize % loopInc;

                    for (SizeType idx = start[IDX_4]; idx < start[IDX_4] + neonLoop; idx += loopInc) {
                        svbool_t ldpg0, ldpg1, ldpg2, ldpg3;
                        Vector::InitPGN<InType, Vector::GetUsedRegsN<InType, CompType, OutType>()>::Do(
                            ldpg0, ldpg1, ldpg2, ldpg3, idx, dims[IDX_4]);
                        typename Vector::Vec<CompType, ARCH::NEON>::Type data0, data1, data2, data3;
                        Vector::LoadN<InType, CompType, Vector::GetUsedRegsN<InType, CompType, OutType>(),
                            ARCH::NEON>::Do(
                            src + idxSrc + idx, ldpg0, ldpg1, ldpg2, ldpg3, data0, data1, data2, data3);
                        svbool_t oppg0, oppg1, oppg2, oppg3;
                        Vector::InitPGN<CompType, Vector::GetUsedRegsN<CompType, InType, OutType>()>::Do(
                            oppg0, oppg1, oppg2, oppg3, idx, dims[IDX_4]);
                        ApplyReductionOpNeon<CompType, InType, OutType>(
                            oppg0, oppg1, oppg2, oppg3,
                            res0, res1, res2, res3,
                            data0, data1, data2, data3,
                            op
                        );
                    }

                    if (tailLoop != 0) {
                        svbool_t ldpg0, ldpg1, ldpg2, ldpg3;
                        Vector::InitPGN<InType, Vector::GetUsedRegsN<InType, CompType, OutType>()>::Do(
                            ldpg0, ldpg1, ldpg2, ldpg3, start[IDX_4] + neonLoop, dims[IDX_4]);
                        typename Vector::Vec<CompType>::Type data0, data1, data2, data3;
                        Vector::LoadN<InType, CompType, Vector::GetUsedRegsN<InType, CompType, OutType>(),
                            ARCH::VC>::Do(
                            src + idxSrc + neonLoop, ldpg0, ldpg1, ldpg2, ldpg3, data0, data1, data2, data3);
                        svbool_t oppg0, oppg1, oppg2, oppg3;
                        Vector::InitPGN<CompType, Vector::GetUsedRegsN<CompType, InType, OutType>()>::Do(
                            oppg0, oppg1, oppg2, oppg3, start[IDX_4] + neonLoop, dims[IDX_4]);
                        ApplyReductionOpVC<CompType, InType, OutType>(
                            oppg0, oppg1, oppg2, oppg3,
                            res4, res5, res6, res7,
                            data0, data1, data2, data3,
                            op
                        );
                    }
#else
                    typename Vector::Vec<CompType>::Type res0, res1, res2, res3;
                    Vector::DupFloatN<InType, CompType, Vector::GetUsedRegsN<InType, CompType, OutType>()>::Do(
                        acc, res0, res1, res2, res3);
                    for (SizeType idx = start[IDX_4]; idx < dims[IDX_4]; idx += loopInc) {
                        svbool_t ldpg0, ldpg1, ldpg2, ldpg3;
                        Vector::InitPGN<InType, Vector::GetUsedRegsN<InType, CompType, OutType>()>::Do(
                            ldpg0, ldpg1, ldpg2, ldpg3, idx, dims[IDX_4]);
                        typename Vector::Vec<CompType>::Type data0, data1, data2, data3;
                        Vector::LoadN<InType, CompType, Vector::GetUsedRegsN<InType, CompType, OutType>()>::Do(
                            src + idxSrc + idx, ldpg0, ldpg1, ldpg2, ldpg3, data0, data1, data2, data3);
                        svbool_t oppg0, oppg1, oppg2, oppg3;
                        Vector::InitPGN<CompType, Vector::GetUsedRegsN<CompType, InType, OutType>()>::Do(
                            oppg0, oppg1, oppg2, oppg3, idx, dims[IDX_4]);
                        ApplyReductionOpVC<CompType, InType, OutType>(
                            oppg0, oppg1, oppg2, oppg3,
                            res0, res1, res2, res3,
                            data0, data1, data2, data3,
                            op
                        );
                    }
#endif

#if USE_NEON
    std::array<CompType, NUM_4 * Vector::VC_MAX_BYTE_LEN_920G / sizeof(CompType)> tmp, tmp1;
#else
    #ifdef KUDNNL_920F
                        std::array<CompType, NUM_4 * Vector::VC_MAX_BYTE_LEN_920F / sizeof(CompType)> tmp;
    #else
                        std::array<CompType, NUM_4 * Vector::VC_MAX_BYTE_LEN_920B / sizeof(CompType)> tmp;
    #endif
#endif

                    svbool_t stpg0, stpg1, stpg2, stpg3;
                    Vector::InitPGN<CompType,
                        Vector::GetUsedRegsN<CompType, InType, OutType>()>::Do(
                        stpg0, stpg1, stpg2, stpg3, 0, Vector::Length<CompType>() *
                        Vector::GetUsedRegsN<CompType, InType, OutType>());
#if USE_NEON
                    Vector::StoreN<CompType, CompType,
                        Vector::GetUsedRegsN<CompType, InType, OutType>(), ARCH::NEON>::Do(
                        stpg0, stpg1, stpg2, stpg3, res0, res1, res2, res3, tmp.data());
                    Vector::StoreN<CompType, CompType,
                        Vector::GetUsedRegsN<CompType, InType, OutType>(), ARCH::VC>::Do(
                        stpg0, stpg1, stpg2, stpg3, res4, res5, res6, res7, tmp1.data());
#else
                    Vector::StoreN<CompType, CompType,
                        Vector::GetUsedRegsN<CompType, InType, OutType>()>::Do(
                        stpg0, stpg1, stpg2, stpg3, res0, res1, res2, res3, tmp.data());
#endif

                    for (SizeType idx = start[IDX_4]; idx < std::min(dims[IDX_4],
                        Vector::Length<CompType>() * Vector::GetUsedRegsN<CompType, InType, OutType>()); idx++) {
#if USE_NEON
                        tmp[idx] = scOp(tmp[idx], tmp1[idx]);
#endif
                        dstTmp[idxDst] = scOp(dstTmp[idxDst], tmp[idx]);
                    }
                }
            }
        }
    }

    for (SizeType k0 = start[IDX_0]; k0 < end[IDX_0]; ++k0) {
        SizeType idxDst0 = k0 * stridesDst[IDX_0];
        for (SizeType k1 = start[IDX_1]; k1 < end[IDX_1]; ++k1) {
            SizeType idxDst1 = k1 * stridesDst[IDX_1];
            for (SizeType k2 = start[IDX_2]; k2 < end[IDX_2]; ++k2) {
                SizeType idxDst2 = k2 * stridesDst[IDX_2];
                for (SizeType k3 = start[IDX_3]; k3 < end[IDX_3]; ++k3) {
                    SizeType idxDst = idxDst0 + idxDst1 + idxDst2 + k3 * stridesDst[IDX_3];
                    if (kind == ReductionFunction::MEAN) {
                        dst[idxDst] = Vector::Convert<float, OutType>::Do(dstTmp[idxDst] / reduceSize);
                    } else if constexpr (!std::is_same_v<float, OutType>) {
                        dst[idxDst] = Vector::Convert<float, OutType>::Do(dstTmp[idxDst]);
                    }
                }
            }
        }
    }
}

template <typename CompType, typename InType, typename OutType, typename ScalarOp>
KUDNN_INLINE inline void EvaluateScalar(const Shape &start, const Shape &end, const InType *src, OutType *dst,
    float* dstBuf, ScalarOp &&scOp, ReductionOp op,
    CompType acc, ReductionFunction kind, SizeType reduceSize) noexcept
{
    Service::Unused(start, end);
    float* dstTmp = nullptr;

    if constexpr (!std::is_same_v<float, OutType>) {
        dstTmp = dstBuf;
    } else {
        dstTmp = static_cast<float*>(dst);
        dstTmp[IDX_0] = acc;
    }

#if USE_NEON
    printf("EvaluateScalar USE_NEON!!\n");
    typename Vector::Vec<CompType, ARCH::NEON>::Type res0, res1, res2, res3;
    typename Vector::Vec<CompType, ARCH::VC>::Type res4, res5, res6, res7;
    Vector::DupFloatN<InType, CompType, Vector::GetUsedRegsN<InType, CompType, OutType>(), ARCH::NEON>::Do(
        acc, res0, res1, res2, res3);
    Vector::DupFloatN<InType, CompType, Vector::GetUsedRegsN<InType, CompType, OutType>(), ARCH::VC>::Do(
        acc, res4, res5, res6, res7);
    SizeType loopInc = NUM_2 * Vector::NeonLanes<CompType>();

    SizeType neonLoop = reduceSize - (reduceSize % loopInc);
    SizeType tailLoop = reduceSize % loopInc;

    for (SizeType idx = 0; idx < neonLoop; idx += loopInc) {
        svbool_t ldpg0, ldpg1, ldpg2, ldpg3;
        Vector::InitPGN<InType, Vector::GetUsedRegsN<InType, CompType, OutType>()>::Do(
            ldpg0, ldpg1, ldpg2, ldpg3, idx, reduceSize);
        typename Vector::Vec<CompType, ARCH::NEON>::Type data0, data1, data2, data3;
        Vector::LoadN<InType, CompType, Vector::GetUsedRegsN<InType, CompType, OutType>(), ARCH::NEON>::Do(
            src + idx, ldpg0, ldpg1, ldpg2, ldpg3, data0, data1, data2, data3);
        svbool_t oppg0, oppg1, oppg2, oppg3;
        Vector::InitPGN<CompType, Vector::GetUsedRegsN<CompType, InType, OutType>()>::Do(
            oppg0, oppg1, oppg2, oppg3, idx, reduceSize);
        ApplyReductionOpNeon<CompType, InType, OutType>(
            oppg0, oppg1, oppg2, oppg3,
            res0, res1, res2, res3,
            data0, data1, data2, data3,
            op
        );
    }

    if (tailLoop != 0) {
        svbool_t ldpg0, ldpg1, ldpg2, ldpg3;
        Vector::InitPGN<InType, Vector::GetUsedRegsN<InType, CompType, OutType>()>::Do(
            ldpg0, ldpg1, ldpg2, ldpg3, neonLoop, reduceSize);
        typename Vector::Vec<CompType>::Type data0, data1, data2, data3;
        Vector::LoadN<InType, CompType, Vector::GetUsedRegsN<InType, CompType, OutType>(), ARCH::VC>::Do(
            src + neonLoop, ldpg0, ldpg1, ldpg2, ldpg3, data0, data1, data2, data3);
        svbool_t oppg0, oppg1, oppg2, oppg3;
        Vector::InitPGN<CompType, Vector::GetUsedRegsN<CompType, InType, OutType>()>::Do(
            oppg0, oppg1, oppg2, oppg3, neonLoop, reduceSize);
        ApplyReductionOpVC<CompType, InType, OutType>(
            oppg0, oppg1, oppg2, oppg3,
            res4, res5, res6, res7,
            data0, data1, data2, data3,
            op
        );
    }
#else
    typename Vector::Vec<CompType>::Type res0, res1, res2, res3;
    Vector::DupFloatN<InType, CompType, Vector::GetUsedRegsN<InType, CompType, OutType>()>::Do(
        acc, res0, res1, res2, res3);
    SizeType loopInc = Vector::UnrollLength<InType, CompType, OutType>();

    for (SizeType idx = 0; idx < reduceSize; idx += loopInc) {
        svbool_t ldpg0, ldpg1, ldpg2, ldpg3;
        Vector::InitPGN<InType, Vector::GetUsedRegsN<InType, CompType, OutType>()>::Do(
            ldpg0, ldpg1, ldpg2, ldpg3, idx, reduceSize);
        typename Vector::Vec<CompType>::Type data0, data1, data2, data3;
        Vector::LoadN<InType, CompType, Vector::GetUsedRegsN<InType, CompType, OutType>()>::Do(
            src + idx, ldpg0, ldpg1, ldpg2, ldpg3, data0, data1, data2, data3);
        svbool_t oppg0, oppg1, oppg2, oppg3;
        Vector::InitPGN<CompType, Vector::GetUsedRegsN<CompType, InType, OutType>()>::Do(
            oppg0, oppg1, oppg2, oppg3, idx, reduceSize);
        ApplyReductionOpVC<CompType, InType, OutType>(
            oppg0, oppg1, oppg2, oppg3,
            res0, res1, res2, res3,
            data0, data1, data2, data3,
            op
        );
    }
#endif

#if USE_NEON
    std::array<CompType, NUM_4 * Vector::VC_MAX_BYTE_LEN_920G / sizeof(CompType)> temp, temp1;
#else
    #ifdef KUDNNL_920F
        std::array<CompType, NUM_4 * Vector::VC_MAX_BYTE_LEN_920F / sizeof(CompType)> temp;
    #else
        std::array<CompType, NUM_4 * Vector::VC_MAX_BYTE_LEN_920B / sizeof(CompType)> temp;
    #endif
#endif

    svbool_t stpg0, stpg1, stpg2, stpg3;
    Vector::InitPGN<CompType,
        Vector::GetUsedRegsN<CompType, InType, OutType>()>::Do(
        stpg0, stpg1, stpg2, stpg3, 0, Vector::Length<CompType>() *
        Vector::GetUsedRegsN<CompType, InType, OutType>());
#if USE_NEON
    Vector::StoreN<CompType, CompType,
        Vector::GetUsedRegsN<CompType, InType, OutType>(), ARCH::NEON>::Do(
        stpg0, stpg1, stpg2, stpg3, res0, res1, res2, res3, temp.data());
    Vector::StoreN<CompType, CompType,
        Vector::GetUsedRegsN<CompType, InType, OutType>(), ARCH::VC>::Do(
        stpg0, stpg1, stpg2, stpg3, res4, res5, res6, res7, temp1.data());
#else
    Vector::StoreN<CompType, CompType,
        Vector::GetUsedRegsN<CompType, InType, OutType>()>::Do(
        stpg0, stpg1, stpg2, stpg3, res0, res1, res2, res3, temp.data());
#endif

    for (SizeType idx = 0; idx < std::min(reduceSize,
        Vector::Length<CompType>() * Vector::GetUsedRegsN<CompType, InType, OutType>()); idx++) {
#if USE_NEON
        temp[idx] = scOp(temp[idx], temp1[idx]);
#endif
        dstTmp[IDX_0] = scOp(dstTmp[IDX_0], temp[idx]);
    }

    if (kind == ReductionFunction::MEAN) {
        dst[IDX_0] = Vector::Convert<float, OutType>::Do(dstTmp[IDX_0] / reduceSize);
    } else if constexpr (!std::is_same_v<float, OutType>) {
        dst[IDX_0] = Vector::Convert<float, OutType>::Do(dstTmp[IDX_0]);
    }
}

} // Reduction
} // Detail
} // KuDNN

#endif //  KUDNN_REDUCTION_DEF_HPP
