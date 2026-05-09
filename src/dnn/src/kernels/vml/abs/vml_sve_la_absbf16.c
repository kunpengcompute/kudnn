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

static inline void AbsMainKernelBf16(svbool_t pg, const uint16_t *src, uint16_t *dst)
{
    svuint16_t ux = svld1_u16(pg, src);
    svuint16_t res = svand_u16_x(pg, ux, VML_VectorComputation_U16_ABS);
    svst1_u16(pg, dst, res);
}

void AbsLaSveFWDBF16(const int64_t len, const __bf16 *src, __bf16 *dst)
{
    int64_t i = 0;
    int loopAbsh = svcnth();
    svbool_t pgB16 = svptrue_b16();
    for (; i <= len - loopAbsh; i += loopAbsh) {
        AbsMainKernelBf16(pgB16, (const uint16_t *)(src + i), (uint16_t *)(dst + i));
    }
    for (; i < len; i += loopAbsh) {
        pgB16 = svwhilelt_b16_s64(i, len);
        AbsMainKernelBf16(pgB16, (const uint16_t *)(src + i), (uint16_t *)(dst + i));
    }
}

void AbsLaSveBWDBF16(const int64_t len, const __bf16 *src, const __bf16 *dd, __bf16 *ds)
{
    int64_t i;
    int loopAbsh = svcnth();
    for (i = 0; i < len; i += loopAbsh) {
        svbool_t pg = svwhilelt_b16_s64(i, len);
        svuint16_t ux = svld1_u16(pg, (const uint16_t *)src + i);
        svuint16_t uAbsSrcX = svand_u16_x(pg, ux, VML_VectorComputation_U16_ABS);
        svuint16_t uDdX = svld1_u16(pg, (const uint16_t *)dd + i);

        svbool_t cmpEqZero = svcmpeq_u16(pg, uAbsSrcX, VML_VectorComputation_U16_ZERO);
        svbool_t cmpLtZero = svcmplt_u16(pg, VML_VectorComputation_U16_SIGN, ux);

        svuint16_t res = sveor_u16_m(cmpLtZero, uDdX, VML_VectorComputation_U16_SIGN);
        res = svsel_u16(cmpEqZero, VML_VectorComputation_U16_ZERO, res);

        svst1_u16(pg, (uint16_t *)ds + i, res);
    }
}
