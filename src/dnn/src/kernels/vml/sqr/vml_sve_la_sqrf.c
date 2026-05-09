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

void SqrLaSveFWDF32(const int64_t len, const float *src, float *dst)
{
    int loopCnt = svcntw();
    int64_t i;
    for (i = 0; i < len; i += loopCnt) {
        svbool_t pg = svwhilelt_b32_s64(i, len);
        svfloat32_t a = svld1_f32(pg, &src[i]);
        svfloat32_t c = svmul_f32_x(pg, a, a);
        svst1_f32(pg, &dst[i], c);
    }
}

void SqrLaSveBWDF32(const int64_t len, const float *src, const float *dd, float *ds)
{
    int loopCnt = svcntw();
    int64_t i;
    for (i = 0; i < len; i += loopCnt) {
        svbool_t pg = svwhilelt_b32_s64(i, len);
        svfloat32_t srcX = svld1_f32(pg, src + i);
        svfloat32_t res = svld1_f32(pg, dd + i);
        srcX = svmul_n_f32_x(pg, srcX, FLOAT_TWO); /* particalX */
        res = svmul_f32_x(pg, res, srcX);

        svst1_f32(pg, ds + i, res);
    }
}
