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

#ifndef VML_VectorComputation_LA_SIGMOID_H
#define VML_VectorComputation_LA_SIGMOID_H

#include "vml_sve_base.h"

#define VectorComputation_SIGMOID_LN_MIN_F32 svdup_f32(-0x1.5cp6f) // -87
#define VectorComputation_SIGMOID_LN_MAX_F32 svdup_f32(0x1.5cp6f)  // 87
#define VectorComputation_SIGMOID_INVLN2_F32 svdup_n_f32(0x1.715476p+0f)
#define VectorComputation_SIGMOID_SHIFTER_F32 svdup_n_f32(0x1.803f8p17f)
#define VectorComputation_SIGMOID_POLY_C0_F32 svdup_n_f32(0x1.62e584p-1f)
#define VectorComputation_SIGMOID_POLY_C1_F32 svdup_n_f32(0x1.ebfc28p-3f)

__attribute__((always_inline)) inline svfloat32_t SveSigmoidF32Exp(svfloat32_t x, svbool_t pg, svbool_t npg)
{
    svfloat32_t ki = svmad_f32_z(npg, x, VectorComputation_SIGMOID_INVLN2_F32, VectorComputation_SIGMOID_SHIFTER_F32);
    svfloat32_t kd = svsub_f32_z(npg, ki, VectorComputation_SIGMOID_SHIFTER_F32);
    svuint32_t k = svreinterpret_u32_f32(ki);
    svfloat32_t expk = svexpa_f32(k);
    svfloat32_t r = svnmls_f32_z(npg, kd, x, VectorComputation_SIGMOID_INVLN2_F32);
    svfloat32_t poly =
        svmla_f32_z(npg, VectorComputation_SIGMOID_POLY_C0_F32, r, VectorComputation_SIGMOID_POLY_C1_F32);
    poly = svmla_f32_z(npg, VML_VectorComputation_F32_ONE, poly, r);
    svfloat32_t expx = svmla_f32_z(pg, VML_VectorComputation_F32_ONE, expk, poly);
    return expx;
}

__attribute__((always_inline)) inline svfloat32_t SvSigmoidF32X(svfloat32_t x, svbool_t pg)
{
    svbool_t npg = svcmpge_f32(pg, x, VectorComputation_SIGMOID_LN_MIN_F32);
    svbool_t npg1 = svcmple_f32(pg, x, VectorComputation_SIGMOID_LN_MAX_F32);
    svfloat32_t expx = SveSigmoidF32Exp(x, pg, npg);
    svfloat32_t res = svdiv_f32_x(pg, VML_VectorComputation_F32_ONE, expx);
    res = svadd_f32_m(npg1, VML_VectorComputation_F32_ZERO, res);
    res = svmul_f32_m(npg, VML_VectorComputation_F32_ONE, res);
    return res;
}

#endif
