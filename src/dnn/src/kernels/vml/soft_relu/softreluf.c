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

#include "vml_sve_la_softrelu.h"

void SoftReluLaVCFWDF32(const int64_t len, const float alpha, const float *src, float *dst)
{
    int64_t i;
    int loopSoftReluf = svcntw();
    svfloat32_t va = svdup_n_f32(alpha);
    for (i = 0; i < len; i += loopSoftReluf) {
        svbool_t pg = svwhilelt_b32_s64(i, len);
        svfloat32_t x = svld1_f32(pg, src + i);
        svfloat32_t res = SvSoftReluF32X(va, x, pg);

        svst1_f32(pg, dst + i, res);
    }
}

void SoftReluLaVCBWDF32(const int64_t len, const float alpha, const float *src, const float *dd, float *ds)
{
    int64_t i;
    int loopSoftReluf = svcntw();
    svfloat32_t va = svdup_n_f32(-alpha); /* sigmoid takes -ax as input */
    for (i = 0; i < len; i += loopSoftReluf) {
        svbool_t pg = svwhilelt_b32_s64(i, len);
        svfloat32_t srcX = svld1_f32(pg, src + i);
        svfloat32_t ddX = svld1_f32(pg, dd + i);
        svfloat32_t res = SvSoftReluBwdF32X(va, srcX, ddX, pg);

        svst1_f32(pg, ds + i, res);
    }
}
