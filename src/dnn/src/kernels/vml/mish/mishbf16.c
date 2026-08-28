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

void MishLaVCFWDBF16(const int64_t len, const __bf16 *src, __bf16 *dst)
{
    int64_t i;
    int loopMishh = svcnth();
    for (i = 0; i < len; i += loopMishh) {
        svbool_t pgB16 = svwhilelt_b16_s64(i, len);
        svbool_t pgB32 = svwhilelt_b32_s64(i, len);

        svbfloat16_t x = svld1_bf16(pgB16, src + i);
        svfloat32_t xBot = SvCvtF32BF16BottomX(x);
        svfloat32_t xTop = SvCvtF32BF16TopX(x);

        svfloat32_t resBot = VCMishFwdF32X(xBot, pgB32);
        svfloat32_t resTop = VCMishFwdF32X(xTop, pgB32);

        svbfloat16_t res = SvCvtBF16F32FullX(resBot, resTop);
        svst1_bf16(pgB16, dst + i, res);
    }
}

void MishLaVCBWDBF16(const int64_t len, const __bf16 *src, const __bf16 *dd, __bf16 *ds)
{
    int64_t i;
    int loopMishh = svcnth();
    for (i = 0; i < len; i += loopMishh) {
        svbool_t pgB16 = svwhilelt_b16_s64(i, len);
        svbool_t pgB32 = svwhilelt_b32_s64(i, len);

        svbfloat16_t srcX = svld1_bf16(pgB16, src + i);
        svbfloat16_t ddX = svld1_bf16(pgB16, dd + i);

        svfloat32_t srcBot = SvCvtF32BF16BottomX(srcX);
        svfloat32_t srcTop = SvCvtF32BF16TopX(srcX);

        svfloat32_t resBot = VCMishBwdF32X(srcBot, pgB32);
        svfloat32_t resTop = VCMishBwdF32X(srcTop, pgB32);

        svfloat32_t ddBot = SvCvtF32BF16BottomX(ddX);
        svfloat32_t ddTop = SvCvtF32BF16TopX(ddX);

        resBot = svmul_f32_x(pgB32, resBot, ddBot);
        resTop = svmul_f32_x(pgB32, resTop, ddTop);

        resTop = SvF32CvtBf16InfSpecial(pgB32, resTop);
        resBot = SvF32CvtBf16InfSpecial(pgB32, resBot);

        svbfloat16_t res = SvCvtBF16F32FullX(resBot, resTop);
        svst1_bf16(pgB16, ds + i, res);
    }
}
