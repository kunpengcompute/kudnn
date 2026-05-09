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

#ifndef VML_VectorComputation_LA_MISH_H
#define VML_VectorComputation_LA_MISH_H

#include "kernels/vml/vml_sve_la_exp.h"

#define FWD_MISH_MAX_X_FOR_EQUATION_F 0x1.62e42ep5f // 0x42317217
#define BWD_MISH_MAX_X_FOR_EQUATION_F 0x1.62e42ep4f // 0x41b17217

inline svfloat32_t SveMishFwdF32X(svfloat32_t x, svbool_t pg)
{
    // mish(x) = x * ((e^x + 1)^2 - 1)/((e^x + 1)^2 + 1)
    svfloat32_t nx = svmin_n_f32_x(pg, x, FWD_MISH_MAX_X_FOR_EQUATION_F);
    svfloat32_t e = SvExpF32X(nx, pg);

    svfloat32_t ea1 = svadd_f32_x(pg, e, VectorComputation_FLOAT_ONE);
    svfloat32_t e2 = svmul_f32_x(pg, ea1, ea1);
    svfloat32_t e2a1 = svadd_f32_x(pg, e2, VectorComputation_FLOAT_ONE);
    svfloat32_t e2m1 = svsub_f32_x(pg, e2, VectorComputation_FLOAT_ONE);

    svfloat32_t res = svdiv_f32_x(pg, e2m1, e2a1);
    res = svmul_f32_x(pg, res, x);
    return res;
}

inline svfloat32_t SveMishBwdF32XPart(svfloat32_t x, svfloat32_t e, svbool_t pg)
{
    // e^3x + 4*e^2x
    svfloat32_t e2 = svmul_f32_x(pg, e, e);
    svfloat32_t g1 = svmad_f32_x(
        pg, e2, e, svmul_f32_x(pg, svmul_f32_x(pg, e2, VectorComputation_FLOAT_TWO), VectorComputation_FLOAT_TWO));

    // e^3x + 4*e^2x + 4*e^x*(x+1.5)
    svfloat32_t xa1 = svadd_f32_x(pg, x, VectorComputation_FLOAT_ONE);
    svfloat32_t g2 = svadd_f32_x(pg, xa1, VectorComputation_FLOAT_HALF);
    g2 = svmad_f32_x(pg, svmul_f32_x(pg, svmul_f32_x(pg, g2, VectorComputation_FLOAT_TWO), VectorComputation_FLOAT_TWO),
                     e, g1);

    // omega = e^3x + 4*e^2x + 4*e^x*(x+1.5) + 4*(x+1)
    svfloat32_t res =
        svmad_f32_x(pg, svmul_f32_x(pg, xa1, VectorComputation_FLOAT_TWO), VectorComputation_FLOAT_TWO, g2);
    return res;
}

inline svfloat32_t SveMishBwdF32X(svfloat32_t x, svbool_t pg)
{
    svfloat32_t nx = svmin_n_f32_x(pg, x, BWD_MISH_MAX_X_FOR_EQUATION_F);
    svfloat32_t e = SvExpF32X(nx, pg);

    svfloat32_t p1 = SveMishBwdF32XPart(x, e, pg);

    // delta = (e^x+1)^2 + 1
    svfloat32_t ea1 = svadd_f32_x(pg, e, VectorComputation_FLOAT_ONE);
    svfloat32_t ea1Sqr = svmul_f32_x(pg, ea1, ea1);
    svfloat32_t ea1SqrAddOne = svadd_f32_x(pg, ea1Sqr, VectorComputation_FLOAT_ONE);
    svfloat32_t p2 = svmul_f32_x(pg, ea1SqrAddOne, ea1SqrAddOne);

    // e^x * omega / delta^2
    svfloat32_t res = svmul_f32_x(pg, e, p1);
    res = svdiv_f32_x(pg, res, p2);
    return res;
}

#endif
