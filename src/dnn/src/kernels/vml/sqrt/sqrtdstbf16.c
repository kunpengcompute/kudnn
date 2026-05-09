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
#define VML_VectorComputation_UNFOLD_N 7
#include "kernels/vml/common_macro.h"

static void SqrtMainKernelBwdBF16(const __bf16 *src, const __bf16 *dd, __bf16 *ds, int step)
{
    svbool_t pgB16 = svptrue_b16();
    svbool_t pgB32 = svptrue_b32();

    VML_VectorComputation_DECLARE(svfloat32_t, xBot);
    VML_VectorComputation_DECLARE(svfloat32_t, xTop);
    VML_VectorComputation_DECLARE(svfloat32_t, ddBot);
    VML_VectorComputation_DECLARE(svfloat32_t, ddTop);

    LOAD_CONVERT_BF16_TO_FP32_2(src, dd, pgB16, xBot, xTop, ddBot, ddTop, step);

    // sqrt compute
    VML_VectorComputation_DECLARE(svfloat32_t, resBot);
    VML_VectorComputation_DECLARE(svfloat32_t, resTop);
    VML_VectorComputation_FUNC_PG_N2_2PARA(resBot, svmul_f32_x, pgB32, xBot, VectorComputation_FLOAT_TWO);
    VML_VectorComputation_FUNC_PG_N2_2PARA(resTop, svmul_f32_x, pgB32, xTop, VectorComputation_FLOAT_TWO);
    VML_VectorComputation_FUNC_PG_2PARA(resBot, svdiv_f32_x, pgB32, ddBot, resBot);
    VML_VectorComputation_FUNC_PG_2PARA(resTop, svdiv_f32_x, pgB32, ddTop, resTop);

    CONVERT_STORE_FP32_TO_BF16(ds, pgB16, resBot, resTop, step);
}

void SqrtDstSveBWDBF16(const int64_t len, const __bf16 *src, const __bf16 *dd, __bf16 *ds)
{
    int64_t i = 0;
    int64_t loopSqrtDsth = svcnth();

    for (; i <= len - loopSqrtDsth * VML_VectorComputation_UNFOLD_N;
         i += loopSqrtDsth * VML_VectorComputation_UNFOLD_N) {
        SqrtMainKernelBwdBF16(src + i, dd + i, ds + i, loopSqrtDsth);
    }
    for (; i < len; i += loopSqrtDsth) {
        svbool_t pgB16 = svwhilelt_b16_s64(i, len);
        svbool_t pgB32 = svwhilelt_b32_s64(i, len);

        svbfloat16_t srcX = svld1_bf16(pgB16, src + i);
        svbfloat16_t ddX = svld1_bf16(pgB16, dd + i);

        // convert to fp32
        svfloat32_t xBot = SvCvtF32BF16BottomX(srcX);
        svfloat32_t xTop = SvCvtF32BF16TopX(srcX);

        svfloat32_t resBot = SvCvtF32BF16BottomX(ddX);
        svfloat32_t resTop = SvCvtF32BF16TopX(ddX);

        resBot = svdiv_f32_x(pgB32, resBot, svmul_f32_x(pgB32, xBot, VectorComputation_FLOAT_TWO));
        resTop = svdiv_f32_x(pgB32, resTop, svmul_f32_x(pgB32, xTop, VectorComputation_FLOAT_TWO));

        // convert back to bf16
        svbfloat16_t res = SvCvtBF16F32FullX(resBot, resTop);
        svst1_bf16(pgB16, ds + i, res);
    }
}
