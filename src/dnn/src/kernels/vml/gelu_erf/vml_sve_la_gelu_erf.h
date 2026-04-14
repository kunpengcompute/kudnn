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

#ifndef VML_VectorComputation_LA_GELU_ERF_H
#define VML_VectorComputation_LA_GELU_ERF_H

#include "kernels/vml/vml_sve_la_erf.h"

#define GELU_ERF_INV_SQRT_PI    svdup_f32(0x1.20dd74p-1f)

__attribute__((always_inline)) inline svfloat32_t SvGeluErfF32X(svfloat32_t x, svbool_t pg)
{
    svfloat32_t in = svmul_f32_x(pg, x, VML_VectorComputation_F32_INVSQRT2);
    svfloat32_t erfIn = SvErfF32XNew(in, pg);
    svfloat32_t s = svmul_f32_x(pg, x, VectorComputation_FLOAT_HALF);
    svfloat32_t res = svmad_f32_x(pg, s, erfIn, s);

    return res;
}

__attribute__((always_inline)) inline svfloat32_t SvGeluErfBwdF32X(svfloat32_t x, svfloat32_t dd, svbool_t pg)
{
    svfloat32_t in = svmul_f32_x(pg, x, VML_VectorComputation_F32_INVSQRT2);
    svfloat32_t erfIn = SvErfF32X(in, pg);

    svfloat32_t in2 = svmls_f32_x(pg, VectorComputation_FLOAT_ZERO, in, in);
    svfloat32_t expIn2 = SvExpF32X(in2, pg);

    svfloat32_t res = svmul_f32_x(pg, in, GELU_ERF_INV_SQRT_PI);
    res = svmul_f32_x(pg, res, expIn2);
    res = svadd_f32_x(pg, res, VectorComputation_FLOAT_HALF);
    res = svmad_f32_x(pg, VectorComputation_FLOAT_HALF, erfIn, res);
    res = svmul_f32_x(pg, dd, res);

    return res;
}

#endif
