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

void SoftReluLaSveFWDF16(const int64_t len, const float alpha, const __fp16 *src, __fp16 *dst)
{
    int64_t i;
    int loopSoftReluh = svcnth();
    svfloat32_t va = svdup_n_f32(alpha);
    for (i = 0; i < len; i += loopSoftReluh) {
        svbool_t pgB16 = svwhilelt_b16_s64(i, len);
        svbool_t pgB32 = svwhilelt_b32_s64(i, len);

        // convert to fp32
        svfloat16_t x = svld1_f16(pgB16, src + i);
        svfloat32_t xEven = svcvt_f32_f16_x(pgB16, x);
        svfloat32_t xOdd = CvtOddF32F16(x, pgB16, pgB32);

        // soft_relu compute
        svfloat32_t resEven = SvSoftReluF32X(va, xEven, pgB32);
        svfloat32_t resOdd = SvSoftReluF32X(va, xOdd, pgB32);

        // convert back to fp16
        svfloat16_t res = svcvt_f16_f32_x(pgB32, resEven);
        x = CvtOddF16F32(resOdd, pgB32); /* resOdd(f16) */
        res = svsel_f16(pgB32, res, x);

        svst1_f16(pgB16, dst + i, res);
    }
}

void SoftReluLaSveBWDF16(const int64_t len, const float alpha, const __fp16 *src, const __fp16 *dd, __fp16 *ds)
{
    int64_t i;
    int loopSoftReluh = svcnth();
    svfloat32_t va = svdup_n_f32(-alpha);
    for (i = 0; i < len; i += loopSoftReluh) {
        svbool_t pgB16 = svwhilelt_b16_s64(i, len);
        svbool_t pgB32 = svwhilelt_b32_s64(i, len);

        // convert to fp32
        svfloat16_t x = svld1_f16(pgB16, src + i);
        svfloat32_t xEven = svcvt_f32_f16_x(pgB16, x);
        svfloat32_t xOdd = CvtOddF32F16(x, pgB16, pgB32);

        svfloat16_t ddX = svld1_f16(pgB16, dd + i);
        svfloat32_t ddEven = svcvt_f32_f16_x(pgB16, ddX);
        svfloat32_t ddOdd = CvtOddF32F16(ddX, pgB16, pgB32);

        // soft_relu compute
        svfloat32_t resEven = SvSoftReluBwdF32X(va, xEven, ddEven, pgB32);
        svfloat32_t resOdd = SvSoftReluBwdF32X(va, xOdd, ddOdd, pgB32);

        // convert back to fp16
        svfloat16_t res = svcvt_f16_f32_x(pgB32, resEven); /* resEven(f16) */
        x = CvtOddF16F32(resOdd, pgB32);                   /* resOdd(f16) */
        res = svsel_f16(pgB32, res, x);

        svst1_f16(pgB16, ds + i, res);
    }
}
