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

void SqrLaSveFWDF16(const int64_t len, const __fp16 *src, __fp16 *dst)
{
    int64_t i;
    int loopSqrh = svcnth();
    for (i = 0; i < len; i += loopSqrh) {
        svbool_t pg = svwhilelt_b16_s64(i, len);
        svfloat16_t x = svld1_f16(pg, src + i);
        svfloat16_t res = svmul_f16_x(pg, x, x);
        svst1_f16(pg, dst + i, res);
    }
}

void SqrLaSveBWDF16(const int64_t len, const __fp16 *src, const __fp16 *dd, __fp16 *ds)
{
    int loopSqrh = svcnth();
    int64_t i;
    for (i = 0; i < len; i += loopSqrh) {
        svbool_t pg = svwhilelt_b16_s64(i, len);
        svfloat16_t srcX = svld1_f16(pg, src + i);
        svfloat16_t res = svld1_f16(pg, dd + i);
        srcX = svmul_n_f16_x(pg, srcX, FLOAT_TWO); /* particalX */
        res = svmul_f16_x(pg, res, srcX);

        svst1_f16(pg, ds + i, res);
    }
}
