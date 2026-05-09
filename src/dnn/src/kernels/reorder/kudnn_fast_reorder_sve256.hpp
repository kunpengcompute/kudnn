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

#ifndef KUDNN_FAST_REORDER_VectorComputation256_HPP
#define KUDNN_FAST_REORDER_VectorComputation256_HPP

#include <arm_sve.h>

#include "kudnn.hpp"

#include "internal_funcs.hpp"

namespace KuDNN {
namespace Detail {
namespace Reorder {

// strides in bytes
template <SizeType InHeight, SizeType InWidth, bool AlphaMul = false>
inline void TranspKernel(const void *src, void *dst, SizeType strideIn, SizeType strideOut, float alpha = 1.0f)
{
    static_assert((InHeight > NUM_0) && (InHeight < NUM_9), "TranspKernel: InHeight <= 0 || InHeight > 8");
    static_assert((InWidth > NUM_0) && (InWidth < NUM_9), "TranspKernel: InWidth <= 0 || InWidth > 8");

    svbool_t pgLoad = svwhilelt_b32_s64(NUM_0, InWidth);
    svfloat32_t row1;
    svfloat32_t row2;
    svfloat32_t row3;
    svfloat32_t row4;
    svfloat32_t row5;
    svfloat32_t row6;
    svfloat32_t row7;
    svfloat32_t row8;

    row1 = svld1_f32(pgLoad, Service::AddOffset<const float *>(src, NUM_0 * strideIn));

    if constexpr (InHeight > NUM_1) {
        row2 = svld1_f32(pgLoad, Service::AddOffset<const float *>(src, NUM_1 * strideIn));
    } else {
        row2 = svdup_f32(0.0f);
    }
    if constexpr (InHeight > NUM_2) {
        row3 = svld1_f32(pgLoad, Service::AddOffset<const float *>(src, NUM_2 * strideIn));
    } else {
        row3 = svdup_f32(0.0f);
    }
    if constexpr (InHeight > NUM_3) {
        row4 = svld1_f32(pgLoad, Service::AddOffset<const float *>(src, NUM_3 * strideIn));
    } else {
        row4 = svdup_f32(0.0f);
    }
    if constexpr (InHeight > NUM_4) {
        row5 = svld1_f32(pgLoad, Service::AddOffset<const float *>(src, NUM_4 * strideIn));
    } else {
        row5 = svdup_f32(0.0f);
    }
    if constexpr (InHeight > NUM_5) {
        row6 = svld1_f32(pgLoad, Service::AddOffset<const float *>(src, NUM_5 * strideIn));
    } else {
        row6 = svdup_f32(0.0f);
    }
    if constexpr (InHeight > NUM_6) {
        row7 = svld1_f32(pgLoad, Service::AddOffset<const float *>(src, NUM_6 * strideIn));
    } else {
        row7 = svdup_f32(0.0f);
    }
    if constexpr (InHeight > NUM_7) {
        row8 = svld1_f32(pgLoad, Service::AddOffset<const float *>(src, NUM_7 * strideIn));
    } else {
        row8 = svdup_f32(0.0f);
    }

    svfloat32_t tmp1;
    svfloat32_t tmp2;
    svfloat32_t tmp3;
    svfloat32_t tmp4;
    svfloat32_t tmp5;
    svfloat32_t tmp6;
    svfloat32_t tmp7;
    svfloat32_t tmp8;

    tmp1 = svzip1_f32(row1, row5);
    tmp2 = svzip2_f32(row1, row5);
    tmp3 = svzip1_f32(row2, row6);
    tmp4 = svzip2_f32(row2, row6);
    tmp5 = svzip1_f32(row3, row7);
    tmp6 = svzip2_f32(row3, row7);
    tmp7 = svzip1_f32(row4, row8);
    tmp8 = svzip2_f32(row4, row8);

    row1 = svzip1_f32(tmp1, tmp5);
    row2 = svzip2_f32(tmp1, tmp5);
    row3 = svzip1_f32(tmp2, tmp6);
    row4 = svzip2_f32(tmp2, tmp6);
    row5 = svzip1_f32(tmp3, tmp7);
    row6 = svzip2_f32(tmp3, tmp7);
    row7 = svzip1_f32(tmp4, tmp8);
    row8 = svzip2_f32(tmp4, tmp8);

    tmp1 = svzip1_f32(row1, row5);
    tmp2 = svzip2_f32(row1, row5);
    tmp3 = svzip1_f32(row2, row6);
    tmp4 = svzip2_f32(row2, row6);
    tmp5 = svzip1_f32(row3, row7);
    tmp6 = svzip2_f32(row3, row7);
    tmp7 = svzip1_f32(row4, row8);
    tmp8 = svzip2_f32(row4, row8);

    svbool_t pgStore = svwhilelt_b32_s64(NUM_0, InHeight);

    if constexpr (AlphaMul) {
        svfloat32_t alpha_v = svdup_n_f32(alpha);
        tmp1 = svmul_f32_x(pgStore, tmp1, alpha_v);
        tmp2 = svmul_f32_x(pgStore, tmp2, alpha_v);
        tmp3 = svmul_f32_x(pgStore, tmp3, alpha_v);
        tmp4 = svmul_f32_x(pgStore, tmp4, alpha_v);
        tmp5 = svmul_f32_x(pgStore, tmp5, alpha_v);
        tmp6 = svmul_f32_x(pgStore, tmp6, alpha_v);
        tmp7 = svmul_f32_x(pgStore, tmp7, alpha_v);
        tmp8 = svmul_f32_x(pgStore, tmp8, alpha_v);
    }

    svst1_f32(pgStore, Service::AddOffset<float *>(dst, NUM_0 * strideOut), tmp1);
    if constexpr (InWidth > NUM_1) {
        svst1_f32(pgStore, Service::AddOffset<float *>(dst, NUM_1 * strideOut), tmp2);
    }
    if constexpr (InWidth > NUM_2) {
        svst1_f32(pgStore, Service::AddOffset<float *>(dst, NUM_2 * strideOut), tmp3);
    }
    if constexpr (InWidth > NUM_3) {
        svst1_f32(pgStore, Service::AddOffset<float *>(dst, NUM_3 * strideOut), tmp4);
    }
    if constexpr (InWidth > NUM_4) {
        svst1_f32(pgStore, Service::AddOffset<float *>(dst, NUM_4 * strideOut), tmp5);
    }
    if constexpr (InWidth > NUM_5) {
        svst1_f32(pgStore, Service::AddOffset<float *>(dst, NUM_5 * strideOut), tmp6);
    }
    if constexpr (InWidth > NUM_6) {
        svst1_f32(pgStore, Service::AddOffset<float *>(dst, NUM_6 * strideOut), tmp7);
    }
    if constexpr (InWidth > NUM_7) {
        svst1_f32(pgStore, Service::AddOffset<float *>(dst, NUM_7 * strideOut), tmp8);
    }
}

} // namespace Reorder
} // namespace Detail
} // namespace KuDNN

#endif // KUDNN_FAST_REORDER_VectorComputation256_HPP
