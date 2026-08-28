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

#ifndef KUDNN_CONCAT_DEF_HPP
#define KUDNN_CONCAT_DEF_HPP

#include <vector>

#include <arm_sve.h>
#include <omp.h>

#include "kudnn.hpp"

#include "vector_wrappers.hpp"

namespace KuDNN {
namespace Detail {
namespace Concat {

constexpr SizeType CONCAT_ELE_NUM_1 = 0x1000000;

constexpr SizeType CONCAT_THREADS_64 = 64;

template <typename CompType, typename InType, typename OutType>
void EvaluateConcat(const std::vector<Shape> &concatSrcDims, const Shape &concatDstDim,
    const void **src, OutType *typedDst, const SizeType dim, const SizeType nSrc) noexcept
{
    SizeType threadNum = omp_get_max_threads();
    if (concatDstDim.GetTotalDimsSize() <= CONCAT_ELE_NUM_1 && threadNum > CONCAT_THREADS_64) {
        threadNum = CONCAT_THREADS_64;
    }

    SizeType blockNum = 1;
    SizeType blockSizeCom = 1;
    for (SizeType i = 0; i < concatSrcDims[0].GetNumDims(); i++) {
        if (i < dim) {
            blockNum *= concatSrcDims[0][i];
        } else if (i > dim) {
            blockSizeCom *= concatSrcDims[0][i];
        }
    }
    std::vector<SizeType> blockSize;
    SizeType blockSizeAll = 0;
    for (SizeType j = 0; j < nSrc; j++) {
        SizeType blockSizeEle = concatSrcDims[j][dim] * blockSizeCom;
        blockSize.push_back(blockSizeEle);
        blockSizeAll += blockSizeEle;
    }
#if USE_NEON
    SizeType loopInc = NUM_2 * Vector::NeonLanes<CompType>();
    SizeType loopInc1 = Vector::UnrollLength<InType, OutType, OutType>();
    #pragma omp parallel for num_threads(threadNum)
    for (SizeType k = 0; k < blockNum; k++) {
        OutType *dstTmp = typedDst + blockSizeAll * k;
        for (SizeType j = 0; j < nSrc; j++) {
            const InType *srcTmp = static_cast<const InType *>(src[j]) + k * blockSize[j];
            SizeType neonLoop = blockSize[j] - (blockSize[j] % loopInc);
            SizeType tailLoop = blockSize[j] % loopInc;
            for (SizeType i = 0; i < neonLoop; i += loopInc) {
                svbool_t pg0;
                svbool_t pg1;
                svbool_t pg2;
                svbool_t pg3;
                Vector::InitPGN<InType, Vector::GetUsedRegsN<InType, OutType, OutType>()>::Do(
                    pg0, pg1, pg2, pg3, i, neonLoop);
                typename Vector::Vec<OutType, ARCH::NEON>::Type data0;
                typename Vector::Vec<OutType, ARCH::NEON>::Type data1;
                typename Vector::Vec<OutType, ARCH::NEON>::Type data2;
                typename Vector::Vec<OutType, ARCH::NEON>::Type data3;
                Vector::LoadN<InType, OutType, Vector::GetUsedRegsN<InType, OutType, OutType>(), ARCH::NEON>::Do(
                    srcTmp + i, pg0, pg1, pg2, pg3, data0, data1, data2, data3);
                Vector::StoreN<OutType, OutType, Vector::GetUsedRegsN<OutType, InType, OutType>(), ARCH::NEON>::Do(
                    pg0, pg1, pg2, pg3, data0, data1, data2, data3, dstTmp + i);
            }
            dstTmp += neonLoop;
            srcTmp += neonLoop;

            for (SizeType i = 0; i < tailLoop; i += loopInc1) {
                svbool_t pg4;
                svbool_t pg5;
                svbool_t pg6;
                svbool_t pg7;
                Vector::InitPGN<InType, Vector::GetUsedRegsN<InType, OutType, OutType>()>::Do(
                    pg4, pg5, pg6, pg7, i, tailLoop);
                typename Vector::Vec<OutType>::Type data4;
                typename Vector::Vec<OutType>::Type data5;
                typename Vector::Vec<OutType>::Type data6;
                typename Vector::Vec<OutType>::Type data7;
                Vector::LoadN<InType, OutType, Vector::GetUsedRegsN<InType, OutType, OutType>()>::Do(
                    srcTmp + i, pg4, pg5, pg6, pg7, data4, data5, data6, data7);
                Vector::StoreN<OutType, OutType, Vector::GetUsedRegsN<OutType, InType, OutType>()>::Do(
                    pg4, pg5, pg6, pg7, data4, data5, data6, data7, dstTmp + i);
            }
            dstTmp += tailLoop;
        }
    }
#else
    SizeType loopInc = Vector::UnrollLength<InType, OutType, OutType>();
    #pragma omp parallel for num_threads(threadNum)
    for (SizeType k = 0; k < blockNum; k++) {
        OutType *dstTmp = typedDst + blockSizeAll * k;
        for (SizeType j = 0; j < nSrc; j++) {
            const InType *srcTmp = static_cast<const InType *>(src[j]) + k * blockSize[j];
            for (SizeType i = 0; i < blockSize[j]; i += loopInc) {
                svbool_t pg0;
                svbool_t pg1;
                svbool_t pg2;
                svbool_t pg3;
                Vector::InitPGN<InType, Vector::GetUsedRegsN<InType, OutType, OutType>()>::Do(
                    pg0, pg1, pg2, pg3, i, blockSize[j]);
                typename Vector::Vec<OutType>::Type data0;
                typename Vector::Vec<OutType>::Type data1;
                typename Vector::Vec<OutType>::Type data2;
                typename Vector::Vec<OutType>::Type data3;
                Vector::LoadN<InType, OutType, Vector::GetUsedRegsN<InType, OutType, OutType>()>::Do(
                    srcTmp + i, pg0, pg1, pg2, pg3, data0, data1, data2, data3);
                Vector::StoreN<OutType, OutType, Vector::GetUsedRegsN<OutType, InType, OutType>()>::Do(
                    pg0, pg1, pg2, pg3, data0, data1, data2, data3, dstTmp + i);
            }
            dstTmp += blockSize[j];
        }
    }
#endif
}

} // Concat
} // Detail
} // KuDNN

#endif // KUDNN_CONCAT_DEF_HPP