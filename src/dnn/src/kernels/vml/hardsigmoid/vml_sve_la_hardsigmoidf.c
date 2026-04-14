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

#include "kernels/vml/vml_sve_base.h"

void HardSigmoidLaSveFWDF32(const int64_t len, const float alpha, const float beta, const float *src, float *dst)
{
    int64_t i;
    int loopHardSigmoidf = svcntw();
    svfloat32_t alphaVec = svdup_n_f32(alpha);
    svfloat32_t betaVec = svdup_n_f32(beta);
    for (i = 0; i < len; i += loopHardSigmoidf) {
        svbool_t pg = svwhilelt_b32_s64(i, len);
        svfloat32_t x = svld1_f32(pg, src + i);
        svfloat32_t y = svmad_f32_x(pg, x, alphaVec, betaVec);
        svfloat32_t res = svminnm_f32_x(pg, y, VectorComputation_FLOAT_ONE);
        res = svmaxnm_f32_x(pg, res, VectorComputation_FLOAT_ZERO);
        svst1_f32(pg, dst + i, res);
    }
}

void HardSigmoidLaSveBWDF32(const int64_t len, const float alpha, const float beta,
    const float *src, const float *dd, float *ds)
{
    int64_t i;
    int loopHardSigmoidf = svcntw();
    svfloat32_t alphaVec = svdup_n_f32(alpha);
    svfloat32_t betaVec = svdup_n_f32(beta);
    for (i = 0; i < len; i += loopHardSigmoidf) {
        svbool_t pg = svwhilelt_b32_s64(i, len);
        svfloat32_t x = svld1_f32(pg, src + i);
        svfloat32_t ddX = svld1_f32(pg, dd + i);

        svfloat32_t y = svmad_f32_x(pg, x, alphaVec, betaVec);
        svfloat32_t z = svmul_f32_x(pg, ddX, alphaVec);

        svbool_t cmpLeZero = svcmple_f32(pg, y, VectorComputation_FLOAT_ZERO);
        svbool_t cmpGeOne = svcmpge_f32(pg, y, VectorComputation_FLOAT_ONE);

        svfloat32_t res = svdup_n_f32_m(z, cmpLeZero, FLOAT_ZERO);
        res = svdup_n_f32_m(res, cmpGeOne, FLOAT_ZERO);
        svst1_f32(pg, ds + i, res);
    }
}
