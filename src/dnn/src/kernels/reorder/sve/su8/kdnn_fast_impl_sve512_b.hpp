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

#ifndef KUDNN_FAST_REORDER_VC512_B_HPP
#define KUDNN_FAST_REORDER_VC512_B_HPP

#include <arm_sve.h>
#include <typeinfo>

#include "kudnn.hpp"

#include "internal_funcs.hpp"

namespace KuDNN {
namespace Detail {
namespace Reorder {

template <typename DstType>
svint32_t MinMaxCore(svint32_t row, svbool_t pgtrue)
{
    row = svmax_n_s32_x(pgtrue, row, std::numeric_limits<DstType>::min());
    row = svmin_n_s32_x(pgtrue, row, std::numeric_limits<DstType>::max());
    return row;
}

static svint32_t MulConvertCore(svint32_t row, svbool_t pgtrue, svfloat32_t alphaVec)
{
    svfloat32_t tmp = svcvt_f32_s32_x(pgtrue, row);
    tmp = svmul_f32_x(pgtrue, tmp, alphaVec);
    row = svcvt_s32_x(pgtrue, svrinti_f32_x(pgtrue, tmp));
    return row;
}

template <typename SrcType>
svint32_t LoadCore(svbool_t pgLoad, const void *src, SizeType strideIn)
{
    svint32_t row;
    if constexpr (std::is_same_v<SrcType, std::int8_t>) {
        row = svld1sb_s32(pgLoad, Service::AddOffset<const SrcType *>(src, strideIn));
    } else {
        row = svld1ub_s32(pgLoad, Service::AddOffset<const SrcType *>(src, strideIn));
    }
    return row;
}

template <typename SrcType, typename DstType, bool AlphaMul>
void StoreCore(svbool_t pgStore, svint32_t row, void *dst, SizeType strideOut)
{
    if constexpr (AlphaMul || !(std::is_same_v<SrcType, DstType>)) {
        row = MinMaxCore<DstType>(row, pgStore);
    }
    // Svst1b_s32 do truncate instead of convert from s32 to s8/u8, so it is safe to use
    svst1b_s32(pgStore, Service::AddOffset<std::int8_t *>(dst, strideOut), row);
}

template <SizeType InHeight, SizeType InWidth, bool AlphaMul, typename SrcType, typename DstType>
static inline void ReorderInt8Core(const void *src, void *dst, SizeType strideIn, SizeType strideOut,
                                   float alpha)
{
    svbool_t pgLoad = svwhilelt_b32_s64(NUM_0, InWidth);

    svint32_t row1;
    svint32_t row2;
    svint32_t row3;
    svint32_t row4;
    svint32_t row5;
    svint32_t row6;
    svint32_t row7;
    svint32_t row8;
    svint32_t row9;
    svint32_t row10;
    svint32_t row11;
    svint32_t row12;
    svint32_t row13;
    svint32_t row14;
    svint32_t row15;
    svint32_t row16;

    row1 = LoadCore<SrcType>(pgLoad, src, NUM_0 * strideIn);
    if constexpr (InHeight > NUM_1) {
        row2 = LoadCore<SrcType>(pgLoad, src, NUM_1 * strideIn);
    } else {
        row2 = svdup_s32(NUM_0);
    }
    if constexpr (InHeight > NUM_2) {
        row3 = LoadCore<SrcType>(pgLoad, src, NUM_2 * strideIn);
    } else {
        row3 = svdup_s32(NUM_0);
    }
    if constexpr (InHeight > NUM_3) {
        row4 = LoadCore<SrcType>(pgLoad, src, NUM_3 * strideIn);
    } else {
        row4 = svdup_s32(NUM_0);
    }
    if constexpr (InHeight > NUM_4) {
        row5 = LoadCore<SrcType>(pgLoad, src, NUM_4 * strideIn);
    } else {
        row5 = svdup_s32(NUM_0);
    }
    if constexpr (InHeight > NUM_5) {
        row6 = LoadCore<SrcType>(pgLoad, src, NUM_5 * strideIn);
    } else {
        row6 = svdup_s32(NUM_0);
    }
    if constexpr (InHeight > NUM_6) {
        row7 = LoadCore<SrcType>(pgLoad, src, NUM_6 * strideIn);
    } else {
        row7 = svdup_s32(NUM_0);
    }
    if constexpr (InHeight > NUM_7) {
        row8 = LoadCore<SrcType>(pgLoad, src, NUM_7 * strideIn);
    } else {
        row8 = svdup_s32(NUM_0);
    }
    if constexpr (InHeight > NUM_8) {
        row9 = LoadCore<SrcType>(pgLoad, src, NUM_8 * strideIn);
    } else {
        row9 = svdup_s32(NUM_0);
    }
    if constexpr (InHeight > NUM_9) {
        row10 = LoadCore<SrcType>(pgLoad, src, NUM_9 * strideIn);
    } else {
        row10 = svdup_s32(NUM_0);
    }
    if constexpr (InHeight > NUM_10) {
        row11 = LoadCore<SrcType>(pgLoad, src, NUM_10 * strideIn);
    } else {
        row11 = svdup_s32(NUM_0);
    }
    if constexpr (InHeight > NUM_11) {
        row12 = LoadCore<SrcType>(pgLoad, src, NUM_11 * strideIn);
    } else {
        row12 = svdup_s32(NUM_0);
    }
    if constexpr (InHeight > NUM_12) {
        row13 = LoadCore<SrcType>(pgLoad, src, NUM_12 * strideIn);
    } else {
        row13 = svdup_s32(NUM_0);
    }
    if constexpr (InHeight > NUM_13) {
        row14 = LoadCore<SrcType>(pgLoad, src, NUM_13 * strideIn);
    } else {
        row14 = svdup_s32(NUM_0);
    }
    if constexpr (InHeight > NUM_14) {
        row15 = LoadCore<SrcType>(pgLoad, src, NUM_14 * strideIn);
    } else {
        row15 = svdup_s32(NUM_0);
    }
    if constexpr (InHeight > NUM_15) {
        row16 = LoadCore<SrcType>(pgLoad, src, NUM_15 * strideIn);
    } else {
        row16 = svdup_s32(NUM_0);
    }

    svint32_t tmp1;
    svint32_t tmp2;
    svint32_t tmp3;
    svint32_t tmp4;
    svint32_t tmp5;
    svint32_t tmp6;
    svint32_t tmp7;
    svint32_t tmp8;
    svint32_t tmp9;
    svint32_t tmp10;
    svint32_t tmp11;
    svint32_t tmp12;
    svint32_t tmp13;
    svint32_t tmp14;
    svint32_t tmp15;
    svint32_t tmp16;

    tmp1 = svzip1_s32(row1, row9);
    tmp2 = svzip2_s32(row1, row9);
    tmp3 = svzip1_s32(row2, row10);
    tmp4 = svzip2_s32(row2, row10);
    tmp5 = svzip1_s32(row3, row11);
    tmp6 = svzip2_s32(row3, row11);
    tmp7 = svzip1_s32(row4, row12);
    tmp8 = svzip2_s32(row4, row12);
    tmp9 = svzip1_s32(row5, row13);
    tmp10 = svzip2_s32(row5, row13);
    tmp11 = svzip1_s32(row6, row14);
    tmp12 = svzip2_s32(row6, row14);
    tmp13 = svzip1_s32(row7, row15);
    tmp14 = svzip2_s32(row7, row15);
    tmp15 = svzip1_s32(row8, row16);
    tmp16 = svzip2_s32(row8, row16);

    row1 = svzip1_s32(tmp1, tmp9);
    row2 = svzip2_s32(tmp1, tmp9);
    row3 = svzip1_s32(tmp2, tmp10);
    row4 = svzip2_s32(tmp2, tmp10);
    row5 = svzip1_s32(tmp3, tmp11);
    row6 = svzip2_s32(tmp3, tmp11);
    row7 = svzip1_s32(tmp4, tmp12);
    row8 = svzip2_s32(tmp4, tmp12);
    row9 = svzip1_s32(tmp5, tmp13);
    row10 = svzip2_s32(tmp5, tmp13);
    row11 = svzip1_s32(tmp6, tmp14);
    row12 = svzip2_s32(tmp6, tmp14);
    row13 = svzip1_s32(tmp7, tmp15);
    row14 = svzip2_s32(tmp7, tmp15);
    row15 = svzip1_s32(tmp8, tmp16);
    row16 = svzip2_s32(tmp8, tmp16);

    tmp1 = svzip1_s32(row1, row9);
    tmp2 = svzip2_s32(row1, row9);
    tmp3 = svzip1_s32(row2, row10);
    tmp4 = svzip2_s32(row2, row10);
    tmp5 = svzip1_s32(row3, row11);
    tmp6 = svzip2_s32(row3, row11);
    tmp7 = svzip1_s32(row4, row12);
    tmp8 = svzip2_s32(row4, row12);
    tmp9 = svzip1_s32(row5, row13);
    tmp10 = svzip2_s32(row5, row13);
    tmp11 = svzip1_s32(row6, row14);
    tmp12 = svzip2_s32(row6, row14);
    tmp13 = svzip1_s32(row7, row15);
    tmp14 = svzip2_s32(row7, row15);
    tmp15 = svzip1_s32(row8, row16);
    tmp16 = svzip2_s32(row8, row16);

    row1 = svzip1_s32(tmp1, tmp9);
    row2 = svzip2_s32(tmp1, tmp9);
    row3 = svzip1_s32(tmp2, tmp10);
    row4 = svzip2_s32(tmp2, tmp10);
    row5 = svzip1_s32(tmp3, tmp11);
    row6 = svzip2_s32(tmp3, tmp11);
    row7 = svzip1_s32(tmp4, tmp12);
    row8 = svzip2_s32(tmp4, tmp12);
    row9 = svzip1_s32(tmp5, tmp13);
    row10 = svzip2_s32(tmp5, tmp13);
    row11 = svzip1_s32(tmp6, tmp14);
    row12 = svzip2_s32(tmp6, tmp14);
    row13 = svzip1_s32(tmp7, tmp15);
    row14 = svzip2_s32(tmp7, tmp15);
    row15 = svzip1_s32(tmp8, tmp16);
    row16 = svzip2_s32(tmp8, tmp16);

    if constexpr (AlphaMul) {
        svbool_t pgMul = svwhilelt_b32_s64(NUM_0, InHeight);
        svfloat32_t alphaVec = svdup_f32(alpha);
            row1 = MulConvertCore(row1, pgMul, alphaVec);
        if constexpr (InWidth > NUM_1) {
            row2 = MulConvertCore(row2, pgMul, alphaVec);
        }
        if constexpr (InWidth > NUM_2) {
            row3 = MulConvertCore(row3, pgMul, alphaVec);
        }
        if constexpr (InWidth > NUM_3) {
            row4 = MulConvertCore(row4, pgMul, alphaVec);
        }
        if constexpr (InWidth > NUM_4) {
            row5 = MulConvertCore(row5, pgMul, alphaVec);
        }
        if constexpr (InWidth > NUM_5) {
            row6 = MulConvertCore(row6, pgMul, alphaVec);
        }
        if constexpr (InWidth > NUM_6) {
            row7 = MulConvertCore(row7, pgMul, alphaVec);
        }
        if constexpr (InWidth > NUM_7) {
            row8 = MulConvertCore(row8, pgMul, alphaVec);
        }
        if constexpr (InWidth > NUM_8) {
            row9 = MulConvertCore(row9, pgMul, alphaVec);
        }
        if constexpr (InWidth > NUM_9) {
            row10 = MulConvertCore(row10, pgMul, alphaVec);
        }
        if constexpr (InWidth > NUM_10) {
            row11 = MulConvertCore(row11, pgMul, alphaVec);
        }
        if constexpr (InWidth > NUM_11) {
            row12 = MulConvertCore(row12, pgMul, alphaVec);
        }
        if constexpr (InWidth > NUM_12) {
            row13 = MulConvertCore(row13, pgMul, alphaVec);
        }
        if constexpr (InWidth > NUM_13) {
            row14 = MulConvertCore(row14, pgMul, alphaVec);
        }
        if constexpr (InWidth > NUM_14) {
            row15 = MulConvertCore(row15, pgMul, alphaVec);
        }
        if constexpr (InWidth > NUM_15) {
            row16 = MulConvertCore(row16, pgMul, alphaVec);
        }
    }

    svbool_t pgStore = svwhilelt_b32_s64(NUM_0, InHeight);
    StoreCore<SrcType, DstType, AlphaMul>(pgStore, row1, dst, NUM_0 * strideOut);
    if constexpr (InWidth > NUM_1) {
        StoreCore<SrcType, DstType, AlphaMul>(pgStore, row2, dst, NUM_1 * strideOut);
    }
    if constexpr (InWidth > NUM_2) {
        StoreCore<SrcType, DstType, AlphaMul>(pgStore, row3, dst, NUM_2 * strideOut);
    }
    if constexpr (InWidth > NUM_3) {
        StoreCore<SrcType, DstType, AlphaMul>(pgStore, row4, dst, NUM_3 * strideOut);
    }
    if constexpr (InWidth > NUM_4) {
        StoreCore<SrcType, DstType, AlphaMul>(pgStore, row5, dst, NUM_4 * strideOut);
    }
    if constexpr (InWidth > NUM_5) {
        StoreCore<SrcType, DstType, AlphaMul>(pgStore, row6, dst, NUM_5 * strideOut);
    }
    if constexpr (InWidth > NUM_6) {
        StoreCore<SrcType, DstType, AlphaMul>(pgStore, row7, dst, NUM_6 * strideOut);
    }
    if constexpr (InWidth > NUM_7) {
        StoreCore<SrcType, DstType, AlphaMul>(pgStore, row8, dst, NUM_7 * strideOut);
    }
    if constexpr (InWidth > NUM_8) {
        StoreCore<SrcType, DstType, AlphaMul>(pgStore, row9, dst, NUM_8 * strideOut);
    }
    if constexpr (InWidth > NUM_9) {
        StoreCore<SrcType, DstType, AlphaMul>(pgStore, row10, dst, NUM_9 * strideOut);
    }
    if constexpr (InWidth > NUM_10) {
        StoreCore<SrcType, DstType, AlphaMul>(pgStore, row11, dst, NUM_10 * strideOut);
    }
    if constexpr (InWidth > NUM_11) {
        StoreCore<SrcType, DstType, AlphaMul>(pgStore, row12, dst, NUM_11 * strideOut);
    }
    if constexpr (InWidth > NUM_12) {
        StoreCore<SrcType, DstType, AlphaMul>(pgStore, row13, dst, NUM_12 * strideOut);
    }
    if constexpr (InWidth > NUM_13) {
        StoreCore<SrcType, DstType, AlphaMul>(pgStore, row14, dst, NUM_13 * strideOut);
    }
    if constexpr (InWidth > NUM_14) {
        StoreCore<SrcType, DstType, AlphaMul>(pgStore, row15, dst, NUM_14 * strideOut);
    }
    if constexpr (InWidth > NUM_15) {
        StoreCore<SrcType, DstType, AlphaMul>(pgStore, row16, dst, NUM_15 * strideOut);
    }
}

} // Reorder
} // Detail
} // KuDNN

#endif // KUDNN_FAST_REORDER_VC512_B_HPP
