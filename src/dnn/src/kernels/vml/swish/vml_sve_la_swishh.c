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

#include "kernels/vml/vml_sve_la_sigmoid.h"
#include "kernels/vml/common_macro.h"

void SwishLaSveFWDF16(const int64_t len, const float alpha, const __fp16 *src, __fp16 *dst)
{
    int64_t i;
    int loopSwishh = svcnth();
    svfloat32_t va = svdup_n_f32(alpha);
    for (i = 0; i < len; i += loopSwishh) {
        svbool_t pgB16 = svwhilelt_b16_s64(i, len);
        svbool_t pgB32 = svwhilelt_b32_s64(i, len);

        svfloat16_t x = svld1_f16(pgB16, src + i);

        svfloat32_t xfEven = svcvt_f32_f16_x(pgB16, x);
        svfloat32_t xfOdd = CvtOddF32F16(x, pgB16, pgB32);
        svfloat32_t axfEven = svmul_f32_x(pgB32, xfEven, va);
        svfloat32_t axfOdd = svmul_f32_x(pgB32, xfOdd, va);
        svfloat32_t resEven = SvSigmoidF32X(svneg_f32_x(pgB32, axfEven), pgB32);
        svfloat32_t resOdd = SvSigmoidF32X(svneg_f32_x(pgB32, axfOdd), pgB32);
        resEven = svmul_f32_x(pgB32, resEven, xfEven);
        resOdd = svmul_f32_x(pgB32, resOdd, xfOdd);

        svfloat16_t yEven = svcvt_f16_f32_x(pgB32, resEven);
        svfloat16_t yOdd = CvtOddF16F32(resOdd, pgB32);
        svfloat16_t y = svadd_f16_x(pgB16, yEven, yOdd);
        svst1_f16(pgB16, dst + i, y);
    }
}

#undef VML_VectorComputation_UNFOLD_N
#define VML_VectorComputation_UNFOLD_N 4

static void SwishMainKernelBwdF16(svfloat32_t va, const __fp16 *src, const __fp16 *dd, __fp16 *ds, int step)
{
    svbool_t pgB16 = svptrue_b16();
    svbool_t pgB32 = svptrue_b32();
    VML_VectorComputation_DECLARE(svfloat16_t, srcX);
    VML_VectorComputation_DECLARE(svfloat16_t, ddX);

    VML_VectorComputation_LD_STEP(srcX, svld1_f16, pgB16, src, step);
    VML_VectorComputation_LD_STEP(ddX, svld1_f16, pgB16, dd, step);

    VML_VectorComputation_DECLARE(svfloat32_t, srcXEven);
    VML_VectorComputation_FUNC_PG_1PARA(srcXEven, svcvt_f32_f16_x, pgB16, srcX);
    VML_VectorComputation_DECLARE(svfloat32_t, srcXOdd);
    VML_VectorComputation_FUNC_NOPG_N23_3PARA(srcXOdd, CvtOddF32F16, srcX, pgB16, pgB32);
    VML_VectorComputation_DECLARE(svfloat32_t, aSrcXEven);
    VML_VectorComputation_FUNC_PG_N2_2PARA(aSrcXEven, svmul_f32_x, pgB32, srcXEven, va);
    VML_VectorComputation_DECLARE(svfloat32_t, aSrcXOdd);
    VML_VectorComputation_FUNC_PG_N2_2PARA(aSrcXOdd, svmul_f32_x, pgB32, srcXOdd, va);
    VML_VectorComputation_DECLARE(svfloat32_t, negEven);
    VML_VectorComputation_DECLARE(svfloat32_t, negOdd);
    VML_VectorComputation_FUNC_PG_1PARA(negEven, svneg_f32_x, pgB32, aSrcXEven);
    VML_VectorComputation_FUNC_PG_1PARA(negOdd, svneg_f32_x, pgB32, aSrcXOdd);
    VML_VectorComputation_DECLARE(svfloat32_t, sigmoidResEven);
    VML_VectorComputation_DECLARE(svfloat32_t, sigmoidResOdd);
    VML_VectorComputation_FUNC_NOPG_N2_2PARA(sigmoidResEven, SvSigmoidF32X, negEven, pgB32);
    VML_VectorComputation_FUNC_NOPG_N2_2PARA(sigmoidResOdd, SvSigmoidF32X, negOdd, pgB32);

    VML_VectorComputation_DECLARE(svfloat32_t, resEven);
    VML_VectorComputation_DECLARE(svfloat32_t, resOdd);
    VML_VectorComputation_FUNC_PG_3PARA(resEven, svmls_f32_x, pgB32, aSrcXEven, aSrcXEven, sigmoidResEven);
    VML_VectorComputation_FUNC_PG_3PARA(resOdd, svmls_f32_x, pgB32, aSrcXOdd, aSrcXOdd, sigmoidResOdd);
    VML_VectorComputation_FUNC_PG_3PARA(resEven, svmla_f32_x, pgB32, sigmoidResEven, sigmoidResEven, resEven);
    VML_VectorComputation_FUNC_PG_3PARA(resOdd, svmla_f32_x, pgB32, sigmoidResOdd, sigmoidResOdd, resOdd);

    VML_VectorComputation_DECLARE(svfloat32_t, ddXEven);
    VML_VectorComputation_DECLARE(svfloat32_t, ddXOdd);
    VML_VectorComputation_FUNC_PG_1PARA(ddXEven, svcvt_f32_f16_x, pgB16, ddX);
    VML_VectorComputation_FUNC_NOPG_N23_3PARA(ddXOdd, CvtOddF32F16, ddX, pgB16, pgB32);
    VML_VectorComputation_FUNC_PG_2PARA(resEven, svmul_f32_x, pgB32, resEven, ddXEven);
    VML_VectorComputation_FUNC_PG_2PARA(resOdd, svmul_f32_x, pgB32, resOdd, ddXOdd);

    VML_VectorComputation_DECLARE(svfloat16_t, yEven);
    VML_VectorComputation_DECLARE(svfloat16_t, yOdd);
    VML_VectorComputation_DECLARE(svfloat16_t, y);
    VML_VectorComputation_FUNC_PG_1PARA(yEven, svcvt_f16_f32_x, pgB32, resEven);
    VML_VectorComputation_FUNC_NOPG_N2_2PARA(yOdd, CvtOddF16F32, resOdd, pgB32);
    VML_VectorComputation_FUNC_PG_2PARA(y, svadd_f16_x, pgB16, yEven, yOdd);
    VML_VectorComputation_ST_STEP(svst1_f16, pgB16, ds, step, y);
}

void SwishLaSveBWDF16(const int64_t len, const float alpha, const __fp16 *src, const __fp16 *dd, __fp16 *ds)
{
    int64_t i = 0;
    int loopSwishh = svcnth();
    svfloat32_t va = svdup_n_f32(alpha);
    for (; i <= len - loopSwishh * VML_VectorComputation_UNFOLD_N; i += loopSwishh * VML_VectorComputation_UNFOLD_N) {
        SwishMainKernelBwdF16(va, src + i, dd + i, ds + i, loopSwishh);
    }
    for (; i < len; i += loopSwishh) {
        svbool_t pgB16 = svwhilelt_b16_s64(i, len);
        svbool_t pgB32 = svwhilelt_b32_s64(i, len);

        svfloat16_t srcX = svld1_f16(pgB16, src + i);
        svfloat16_t ddX = svld1_f16(pgB16, dd + i);

        svfloat32_t srcXEven = svcvt_f32_f16_x(pgB16, srcX);
        svfloat32_t srcXOdd = CvtOddF32F16(srcX, pgB16, pgB32);
        svfloat32_t aSrcXEven = svmul_f32_x(pgB32, srcXEven, va);
        svfloat32_t aSrcXOdd = svmul_f32_x(pgB32, srcXOdd, va);
        svfloat32_t sigmoidResEven = SvSigmoidF32X(svneg_f32_x(pgB32, aSrcXEven), pgB32);
        svfloat32_t sigmoidResOdd = SvSigmoidF32X(svneg_f32_x(pgB32, aSrcXOdd), pgB32);
        svfloat32_t resEven = svmls_f32_x(pgB32, aSrcXEven, aSrcXEven, sigmoidResEven);
        svfloat32_t resOdd = svmls_f32_x(pgB32, aSrcXOdd, aSrcXOdd, sigmoidResOdd);
        resEven = svmla_f32_x(pgB32, sigmoidResEven, sigmoidResEven, resEven);
        resOdd = svmla_f32_x(pgB32, sigmoidResOdd, sigmoidResOdd, resOdd);

        svfloat32_t ddXEven = svcvt_f32_f16_x(pgB16, ddX);
        svfloat32_t ddXOdd = CvtOddF32F16(ddX, pgB16, pgB32);
        resEven = svmul_f32_x(pgB32, resEven, ddXEven);
        resOdd = svmul_f32_x(pgB32, resOdd, ddXOdd);

        svfloat16_t yEven = svcvt_f16_f32_x(pgB32, resEven);
        svfloat16_t yOdd = CvtOddF16F32(resOdd, pgB32);
        svfloat16_t y = svadd_f16_x(pgB16, yEven, yOdd);
        svst1_f16(pgB16, ds + i, y);
    }
}
