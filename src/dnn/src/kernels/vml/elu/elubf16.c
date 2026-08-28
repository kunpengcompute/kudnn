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

#include "kernels/vml/vml_sve_la_exp.h"

void EluLaVCFWDBF16(const int64_t len, const float alpha, const __bf16 *src, __bf16 *dst)
{
    int64_t i;
    int loopEluh = svcnth();
    int numF32 = svcntw();
    svfloat32_t va = svdup_n_f32(alpha);
    for (i = 0; i < len; i += loopEluh) {
        svbool_t pgB16 = svwhilelt_b16_s64(i, len);
        svbool_t pgB32 = svwhilelt_b32_s64(i, len);
        svbool_t pgF32Bot = pgB32;
        svbool_t pgF32Top = svwhilelt_b32_s64(i + numF32, len);

        svbfloat16_t x = svld1_bf16(pgB16, src + i);

        // convert to fp32
        svfloat32_t xBot = SvCvtF32BF16BottomX(x);
        svfloat32_t xTop = SvCvtF32BF16TopX(x);

        // elu compute
        svbool_t pgRetX = svcmpgt(pgB32, xBot, VC_FLOAT_ZERO);
        svfloat32_t resBot = SvExpF32X(xBot, pgB32);
        resBot = svsub_f32_x(pgB32, resBot, VC_FLOAT_ONE);
        resBot = svmul_f32_x(pgB32, resBot, va);
        resBot = svsel_f32(pgRetX, xBot, resBot);

        pgRetX = svcmpgt(pgB32, xTop, VC_FLOAT_ZERO);
        svfloat32_t resTop = SvExpF32X(xTop, pgB32);
        resTop = svsub_f32_x(pgB32, resTop, VC_FLOAT_ONE);
        resTop = svmul_f32_x(pgB32, resTop, va);
        resTop = svsel_f32(pgRetX, xTop, resTop);

        resBot = SvF32CvtBf16InfSpecial(pgB32, resBot);
        resTop = SvF32CvtBf16InfSpecial(pgB32, resTop);

        // convert back to bf16
        svbfloat16_t res = SvCvtBF16F32X(pgF32Bot, resBot, pgF32Top, resTop);
        svst1_bf16(pgB16, dst + i, res);
    }
}

void EluLaVCBWDBF16(const int64_t len, const float alpha, const __bf16 *src, const __bf16 *dd, __bf16 *ds)
{
    int64_t i;
    int loopEluh = svcnth();
    svfloat32_t va = svdup_n_f32(alpha);
    for (i = 0; i < len; i += loopEluh) {
        svbool_t pgB16 = svwhilelt_b16_s64(i, len);
        svbool_t pgB32 = svwhilelt_b32_s64(i, len);

        svbfloat16_t x = svld1_bf16(pgB16, src + i);
        svbfloat16_t ddX = svld1_bf16(pgB16, dd + i);

        // convert to fp32
        svfloat32_t xBot = SvCvtF32BF16BottomX(x);
        svfloat32_t xTop = SvCvtF32BF16TopX(x);

        svfloat32_t ddBot = SvCvtF32BF16BottomX(ddX);
        svfloat32_t ddTop = SvCvtF32BF16TopX(ddX);

        // elu compute
        svfloat32_t partialX = SvExpF32X(xBot, pgB32);
        svfloat32_t resBot = svmul_f32_x(pgB32, partialX, va);
        resBot = svmul_f32_x(pgB32, resBot, ddBot);
        svbool_t pgRetDD = svcmpgt_f32(pgB32, xBot, VC_FLOAT_ZERO);
        resBot = svsel_f32(pgRetDD, ddBot, resBot);

        partialX = SvExpF32X(xTop, pgB32);
        svfloat32_t resTop = svmul_f32_x(pgB32, partialX, va);
        resTop = svmul_f32_x(pgB32, resTop, ddTop);
        pgRetDD = svcmpgt_f32(pgB32, xTop, VC_FLOAT_ZERO);
        resTop = svsel_f32(pgRetDD, ddTop, resTop);

        resBot = SvF32CvtBf16InfSpecial(pgB32, resBot);
        resTop = SvF32CvtBf16InfSpecial(pgB32, resTop);

        // convert back to bf16
        svbfloat16_t res = SvCvtBF16F32FullX(resBot, resTop);
        svst1_bf16(pgB16, ds + i, res);
    }
}
