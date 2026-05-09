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

#ifndef KUDNN_FAST_REORDER_VectorComputation512_B_HPP
#define KUDNN_FAST_REORDER_VectorComputation512_B_HPP

#include <arm_sve.h>

#include "kudnn.hpp"

#include "internal_funcs.hpp"

namespace KuDNN {
namespace Detail {
namespace Reorder {

// strides in bytes
template <SizeType InHeight, SizeType InWidth, bool AlphaMul = false>
static inline void TranspKernelB(const void *src, void *dst, SizeType strideIn, SizeType strideOut, float alpha = 1.0f)
{
    static_assert((InHeight > NUM_0) && (InHeight < NUM_17), "TranspKernelB: InHeight <= 0 || InHeight > 16");
    static_assert((InWidth > NUM_0) && (InWidth < NUM_17), "TranspKernelB: InWidth <= 0 || InWidth > 16");

    Service::Unused(alpha);

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

    row1 = svld1sb_s32(pgLoad, Service::AddOffset<const std::int8_t *>(src, NUM_0 * strideIn));

    if constexpr (InHeight > NUM_1) {
        row2 = svld1sb_s32(pgLoad, Service::AddOffset<const std::int8_t *>(src, NUM_1 * strideIn));
    } else {
        row2 = svdup_s32(NUM_0);
    }
    if constexpr (InHeight > NUM_2) {
        row3 = svld1sb_s32(pgLoad, Service::AddOffset<const std::int8_t *>(src, NUM_2 * strideIn));
    } else {
        row3 = svdup_s32(NUM_0);
    }
    if constexpr (InHeight > NUM_3) {
        row4 = svld1sb_s32(pgLoad, Service::AddOffset<const std::int8_t *>(src, NUM_3 * strideIn));
    } else {
        row4 = svdup_s32(NUM_0);
    }
    if constexpr (InHeight > NUM_4) {
        row5 = svld1sb_s32(pgLoad, Service::AddOffset<const std::int8_t *>(src, NUM_4 * strideIn));
    } else {
        row5 = svdup_s32(NUM_0);
    }
    if constexpr (InHeight > NUM_5) {
        row6 = svld1sb_s32(pgLoad, Service::AddOffset<const std::int8_t *>(src, NUM_5 * strideIn));
    } else {
        row6 = svdup_s32(NUM_0);
    }
    if constexpr (InHeight > NUM_6) {
        row7 = svld1sb_s32(pgLoad, Service::AddOffset<const std::int8_t *>(src, NUM_6 * strideIn));
    } else {
        row7 = svdup_s32(NUM_0);
    }
    if constexpr (InHeight > NUM_7) {
        row8 = svld1sb_s32(pgLoad, Service::AddOffset<const std::int8_t *>(src, NUM_7 * strideIn));
    } else {
        row8 = svdup_s32(NUM_0);
    }
    if constexpr (InHeight > NUM_8) {
        row9 = svld1sb_s32(pgLoad, Service::AddOffset<const std::int8_t *>(src, NUM_8 * strideIn));
    } else {
        row9 = svdup_s32(NUM_0);
    }
    if constexpr (InHeight > NUM_9) {
        row10 = svld1sb_s32(pgLoad, Service::AddOffset<const std::int8_t *>(src, NUM_9 * strideIn));
    } else {
        row10 = svdup_s32(NUM_0);
    }
    if constexpr (InHeight > NUM_10) {
        row11 = svld1sb_s32(pgLoad, Service::AddOffset<const std::int8_t *>(src, NUM_10 * strideIn));
    } else {
        row11 = svdup_s32(NUM_0);
    }
    if constexpr (InHeight > NUM_11) {
        row12 = svld1sb_s32(pgLoad, Service::AddOffset<const std::int8_t *>(src, NUM_11 * strideIn));
    } else {
        row12 = svdup_s32(NUM_0);
    }
    if constexpr (InHeight > NUM_12) {
        row13 = svld1sb_s32(pgLoad, Service::AddOffset<const std::int8_t *>(src, NUM_12 * strideIn));
    } else {
        row13 = svdup_s32(NUM_0);
    }
    if constexpr (InHeight > NUM_13) {
        row14 = svld1sb_s32(pgLoad, Service::AddOffset<const std::int8_t *>(src, NUM_13 * strideIn));
    } else {
        row14 = svdup_s32(NUM_0);
    }
    if constexpr (InHeight > NUM_14) {
        row15 = svld1sb_s32(pgLoad, Service::AddOffset<const std::int8_t *>(src, NUM_14 * strideIn));
    } else {
        row15 = svdup_s32(NUM_0);
    }
    if constexpr (InHeight > NUM_15) {
        row16 = svld1sb_s32(pgLoad, Service::AddOffset<const std::int8_t *>(src, NUM_15 * strideIn));
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

    svbool_t pgStore = svwhilelt_b32_s64(NUM_0, InHeight);

    svst1b_s32(pgStore, Service::AddOffset<std::int8_t *>(dst, NUM_0 * strideOut), row1);
    if constexpr (InWidth > NUM_1) {
        svst1b_s32(pgStore, Service::AddOffset<std::int8_t *>(dst, NUM_1 * strideOut), row2);
    }
    if constexpr (InWidth > NUM_2) {
        svst1b_s32(pgStore, Service::AddOffset<std::int8_t *>(dst, NUM_2 * strideOut), row3);
    }
    if constexpr (InWidth > NUM_3) {
        svst1b_s32(pgStore, Service::AddOffset<std::int8_t *>(dst, NUM_3 * strideOut), row4);
    }
    if constexpr (InWidth > NUM_4) {
        svst1b_s32(pgStore, Service::AddOffset<std::int8_t *>(dst, NUM_4 * strideOut), row5);
    }
    if constexpr (InWidth > NUM_5) {
        svst1b_s32(pgStore, Service::AddOffset<std::int8_t *>(dst, NUM_5 * strideOut), row6);
    }
    if constexpr (InWidth > NUM_6) {
        svst1b_s32(pgStore, Service::AddOffset<std::int8_t *>(dst, NUM_6 * strideOut), row7);
    }
    if constexpr (InWidth > NUM_7) {
        svst1b_s32(pgStore, Service::AddOffset<std::int8_t *>(dst, NUM_7 * strideOut), row8);
    }
    if constexpr (InWidth > NUM_8) {
        svst1b_s32(pgStore, Service::AddOffset<std::int8_t *>(dst, NUM_8 * strideOut), row9);
    }
    if constexpr (InWidth > NUM_9) {
        svst1b_s32(pgStore, Service::AddOffset<std::int8_t *>(dst, NUM_9 * strideOut), row10);
    }
    if constexpr (InWidth > NUM_10) {
        svst1b_s32(pgStore, Service::AddOffset<std::int8_t *>(dst, NUM_10 * strideOut), row11);
    }
    if constexpr (InWidth > NUM_11) {
        svst1b_s32(pgStore, Service::AddOffset<std::int8_t *>(dst, NUM_11 * strideOut), row12);
    }
    if constexpr (InWidth > NUM_12) {
        svst1b_s32(pgStore, Service::AddOffset<std::int8_t *>(dst, NUM_12 * strideOut), row13);
    }
    if constexpr (InWidth > NUM_13) {
        svst1b_s32(pgStore, Service::AddOffset<std::int8_t *>(dst, NUM_13 * strideOut), row14);
    }
    if constexpr (InWidth > NUM_14) {
        svst1b_s32(pgStore, Service::AddOffset<std::int8_t *>(dst, NUM_14 * strideOut), row15);
    }
    if constexpr (InWidth > NUM_15) {
        svst1b_s32(pgStore, Service::AddOffset<std::int8_t *>(dst, NUM_15 * strideOut), row16);
    }
}

} // namespace Reorder
} // namespace Detail
} // namespace KuDNN

#endif // KUDNN_FAST_REORDER_VectorComputation512_B_HPP
