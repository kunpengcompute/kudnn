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

#include "vml_sve_la_mish.h"

void MishLaVCFWDF32(const int64_t len, const float *src, float *dst)
{
    int64_t i;
    int loopMishf = svcntw();
    for (i = 0; i < len; i += loopMishf) {
        svbool_t pg = svwhilelt_b32_s64(i, len);
        svfloat32_t x = svld1_f32(pg, src + i);
        svfloat32_t res = VCMishFwdF32X(x, pg);
        svst1_f32(pg, dst + i, res);
    }
}

void MishLaVCBWDF32(const int64_t len, const float *src, const float *dd, float *ds)
{
    int64_t i;
    int loopMishf = svcntw();
    for (i = 0; i < len; i += loopMishf) {
        svbool_t pg = svwhilelt_b32_s64(i, len);
        svfloat32_t x = svld1_f32(pg, src + i);
        svfloat32_t dx = svld1_f32(pg, dd + i);
        svfloat32_t res = VCMishBwdF32X(x, pg);
        res = svmul_f32_x(pg, res, dx);
        svst1_f32(pg, ds + i, res);
    }
}
