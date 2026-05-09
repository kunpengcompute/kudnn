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
#include "kernels/vml/common_macro.h"

static void LogisticMainKernelBF16(const __bf16 *src, const __bf16 *dd, int step, __bf16 *ds)
{
    svbool_t pgB16 = svptrue_b16();
    svbool_t pgB32 = svptrue_b32();

    VML_VectorComputation_DECLARE(svfloat32_t, xBot);
    VML_VectorComputation_DECLARE(svfloat32_t, xTop);
    VML_VectorComputation_DECLARE(svfloat32_t, ddBot);
    VML_VectorComputation_DECLARE(svfloat32_t, ddTop);

    LOAD_CONVERT_BF16_TO_FP32_2(src, dd, pgB16, xBot, xTop, ddBot, ddTop, step);

    // logistic_dst compute
    VML_VectorComputation_DECLARE(svfloat32_t, resBot);
    VML_VectorComputation_FUNC_PG_N1_2PARA(resBot, svsub_f32_x, pgB32, VectorComputation_FLOAT_ONE, xBot);
    VML_VectorComputation_FUNC_PG_2PARA(resBot, svmul_f32_x, pgB32, resBot, xBot);
    VML_VectorComputation_FUNC_PG_2PARA(resBot, svmul_f32_x, pgB32, resBot, ddBot);

    VML_VectorComputation_DECLARE(svfloat32_t, resTop);
    VML_VectorComputation_FUNC_PG_N1_2PARA(resTop, svsub_f32_x, pgB32, VectorComputation_FLOAT_ONE, xTop);
    VML_VectorComputation_FUNC_PG_2PARA(resTop, svmul_f32_x, pgB32, resTop, xTop);
    VML_VectorComputation_FUNC_PG_2PARA(resTop, svmul_f32_x, pgB32, resTop, ddTop);

    VML_VectorComputation_FUNC_PG_1PARA(resBot, SvF32CvtBf16InfSpecial, pgB32, resBot);
    VML_VectorComputation_FUNC_PG_1PARA(resTop, SvF32CvtBf16InfSpecial, pgB32, resTop);

    CONVERT_STORE_FP32_TO_BF16(ds, pgB16, resBot, resTop, step);
}

void LogisticDstSveBWDBF16(const int64_t len, const __bf16 *src, const __bf16 *dd, __bf16 *ds)
{
    int size = svcnth();
    int64_t i = 0;
    for (; i <= len - size * VML_VectorComputation_UNFOLD_N; i += size * VML_VectorComputation_UNFOLD_N) {
        LogisticMainKernelBF16(src + i, dd + i, size, ds + i);
    }

    for (; i < len; i += size) {
        svbool_t pgB16 = svwhilelt_b16_s32(i, len);
        svbool_t pgB32 = svwhilelt_b32_s32(i, len);

        svbfloat16_t srcX = svld1_bf16(pgB16, src + i);
        svbfloat16_t ddX = svld1_bf16(pgB16, dd + i);

        // convert to fp32
        svfloat32_t xBot = SvCvtF32BF16BottomX(srcX);
        svfloat32_t xTop = SvCvtF32BF16TopX(srcX);

        svfloat32_t resBot = SvCvtF32BF16BottomX(ddX);
        svfloat32_t oneSubXBot = svsub_f32_x(pgB32, VectorComputation_FLOAT_ONE, xBot);
        xBot = svmul_f32_m(pgB32, xBot, oneSubXBot);
        xBot = svmul_f32_m(pgB32, xBot, resBot);

        svfloat32_t resTop = SvCvtF32BF16TopX(ddX);
        svfloat32_t oneSubXTop = svsub_f32_x(pgB32, VectorComputation_FLOAT_ONE, xTop);
        xTop = svmul_f32_m(pgB32, xTop, oneSubXTop);
        xTop = svmul_f32_m(pgB32, xTop, resTop);

        xBot = SvF32CvtBf16InfSpecial(pgB32, xBot);
        xTop = SvF32CvtBf16InfSpecial(pgB32, xTop);

        svbfloat16_t res = SvCvtBF16F32FullX(xBot, xTop);
        svst1_bf16(pgB16, ds + i, res);
    }
}
