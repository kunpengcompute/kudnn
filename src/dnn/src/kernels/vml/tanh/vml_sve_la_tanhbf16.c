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

#include "kernels/vml/vml_sve_la_tanh.h"

void TanhLaSveFWDBF16(const int64_t len, const __bf16 *src, __bf16 *dst)
{
    int64_t i;
    int loopTanhh = svcnth();
    for (i = 0; i < len; i += loopTanhh) {
        svbool_t pgB16 = svwhilelt_b16_s64(i, len);
        svbool_t pgB32 = svwhilelt_b32_s64(i, len);

        svbfloat16_t x = svld1_bf16(pgB16, src + i);

        svfloat32_t xBot = SvCvtF32BF16BottomX(x);
        svfloat32_t xTop = SvCvtF32BF16TopX(x);

        svfloat32_t resBot = SvTanhF32X(xBot, pgB32);
        svfloat32_t resTop = SvTanhF32X(xTop, pgB32);

        svbfloat16_t res = SvCvtBF16F32FullX(resBot, resTop);
        svst1_bf16(pgB16, dst + i, res);
    }
}

void TanhLaSveBWDBF16(const int64_t len, const __bf16 *src, const __bf16 *dd, __bf16 *ds)
{
    int64_t i;
    int loopTanhh = svcnth();
    for (i = 0; i < len; i += loopTanhh) {
        svbool_t pgB16 = svwhilelt_b16_s64(i, len);
        svbool_t pgB32 = svwhilelt_b32_s64(i, len);

        svbfloat16_t srcX = svld1_bf16(pgB16, src + i);
        svbfloat16_t ddX = svld1_bf16(pgB16, dd + i);

        svfloat32_t srcBot = SvCvtF32BF16BottomX(srcX);
        svfloat32_t srcTop = SvCvtF32BF16TopX(srcX);

        svfloat32_t ddBot = SvCvtF32BF16BottomX(ddX);
        svfloat32_t ddTop = SvCvtF32BF16TopX(ddX);

        svfloat32_t srcBotTanh = SvTanhF32X(srcBot, pgB32);
        svfloat32_t srcTopTanh = SvTanhF32X(srcTop, pgB32);

        svfloat32_t ddBotRes = svmls_f32_x(pgB32, VectorComputation_FLOAT_ONE, srcBotTanh, srcBotTanh);
        svfloat32_t ddTopRes = svmls_f32_x(pgB32, VectorComputation_FLOAT_ONE, srcTopTanh, srcTopTanh);
        ddBotRes = svmul_f32_x(pgB32, ddBot, ddBotRes);
        ddTopRes = svmul_f32_x(pgB32, ddTop, ddTopRes);

        svbfloat16_t res = SvCvtBF16F32FullX(ddBotRes, ddTopRes);
        svst1_bf16(pgB16, ds + i, res);
    }
}
