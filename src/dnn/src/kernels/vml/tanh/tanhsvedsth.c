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

static void TanhMainKernelF16(const __fp16 *src, const __fp16 *dd, int step, __fp16 *ds)
{
    svbool_t pgB16 = svptrue_b16();
    svbool_t pgB32 = svptrue_b32();
    VML_VectorComputation_DECLARE(svfloat16_t, srcX);
    VML_VectorComputation_DECLARE(svfloat32_t, xEven);
    VML_VectorComputation_DECLARE(svfloat32_t, xOdd);
    VML_VectorComputation_DECLARE(svfloat16_t, ddX);
    VML_VectorComputation_DECLARE(svfloat32_t, ddEven);
    VML_VectorComputation_DECLARE(svfloat32_t, ddOdd);
    VML_VectorComputation_DECLARE(svfloat32_t, tmp);
    VML_VectorComputation_LD_STEP(srcX, svld1_f16, pgB16, src, step);
    VML_VectorComputation_LD_STEP(ddX, svld1_f16, pgB16, dd, step);
    // convert src(even)
    VML_VectorComputation_FUNC_PG_1PARA(xEven, svcvt_f32_f16_x, pgB16, srcX);
    // convert src(odd)
    VML_VectorComputation_DECLARE(svuint32_t, shiftedSrc);
    VML_VectorComputation_FUNC_NOPG_1PARA(shiftedSrc, svreinterpret_u32_f16, srcX);
    VML_VectorComputation_FUNC_PG_N2_2PARA(shiftedSrc, svlsr_n_u32_x, pgB32, shiftedSrc, F16_SHIFT);
    VML_VectorComputation_FUNC_NOPG_1PARA(srcX, svreinterpret_f16_u32, shiftedSrc);
    VML_VectorComputation_FUNC_PG_1PARA(xOdd, svcvt_f32_f16_x, pgB16, srcX);
    // convert dd(even)
    VML_VectorComputation_FUNC_PG_1PARA(ddEven, svcvt_f32_f16_x, pgB16, ddX);
    // convert dd(odd)
    VML_VectorComputation_FUNC_NOPG_1PARA(shiftedSrc, svreinterpret_u32_f16, ddX);
    VML_VectorComputation_FUNC_PG_N2_2PARA(shiftedSrc, svlsr_n_u32_x, pgB32, shiftedSrc, F16_SHIFT);
    VML_VectorComputation_FUNC_NOPG_1PARA(ddX, svreinterpret_f16_u32, shiftedSrc);
    VML_VectorComputation_FUNC_PG_1PARA(ddOdd, svcvt_f32_f16_x, pgB16, ddX);

    VML_VectorComputation_FUNC_PG_N1_2PARA(tmp, svadd_f32_x, pgB32, VectorComputation_FLOAT_ONE, xEven);
    VML_VectorComputation_FUNC_PG_N1_2PARA(xEven, svsub_f32_x, pgB32, VectorComputation_FLOAT_ONE, xEven);
    VML_VectorComputation_FUNC_PG_2PARA(xEven, svmul_f32_x, pgB32, tmp, xEven);
    VML_VectorComputation_FUNC_PG_2PARA(xEven, svmul_f32_x, pgB32, ddEven, xEven);

    VML_VectorComputation_FUNC_PG_N1_2PARA(tmp, svadd_f32_x, pgB32, VectorComputation_FLOAT_ONE, xOdd);
    VML_VectorComputation_FUNC_PG_N1_2PARA(xOdd, svsub_f32_x, pgB32, VectorComputation_FLOAT_ONE, xOdd);
    VML_VectorComputation_FUNC_PG_2PARA(xOdd, svmul_f32_x, pgB32, tmp, xOdd);
    VML_VectorComputation_FUNC_PG_2PARA(xOdd, svmul_f32_x, pgB32, ddOdd, xOdd);

    // even values convert back
    VML_VectorComputation_FUNC_PG_1PARA(srcX, svcvt_f16_f32_x, pgB32, xEven);
    // odd values convert back (reuse of shiftedSrc registers)
    VML_VectorComputation_FUNC_PG_1PARA(ddX, svcvt_f16_f32_x, pgB32, xOdd);
    VML_VectorComputation_FUNC_NOPG_1PARA(shiftedSrc, svreinterpret_u32_f16, ddX);
    VML_VectorComputation_FUNC_PG_N2_2PARA(shiftedSrc, svlsl_n_u32_x, pgB32, shiftedSrc, F16_SHIFT);
    VML_VectorComputation_FUNC_NOPG_1PARA(ddX, svreinterpret_f16_u32, shiftedSrc);
    VML_VectorComputation_FUNC_PG_2PARA(srcX, svadd_f16_x, pgB16, srcX, ddX);
    VML_VectorComputation_ST_STEP(svst1_f16, pgB16, ds, step, srcX);
}

void TanhDstSveBWDF16(const int64_t len, const __fp16 *src, const __fp16 *dd, __fp16 *ds)
{
    int size = svcnth();
    int64_t i = 0;
    for (; i <= len - size * VML_VectorComputation_UNFOLD_N; i += size * VML_VectorComputation_UNFOLD_N) {
        TanhMainKernelF16(src + i, dd + i, size, ds + i);
    }

    for (; i < len; i += size) {
        svbool_t pgB16 = svwhilelt_b16_s32(i, len);
        svbool_t pgB32 = svwhilelt_b32_s32(i, len);

        svfloat16_t ddX = svld1_f16(pgB16, dd + i);
        svfloat32_t ddEven = svcvt_f32_f16_x(pgB16, ddX);
        svfloat32_t ddOdd = CvtOddF32F16(ddX, pgB16, pgB32);

        svfloat16_t srcX = svld1_f16(pgB16, src + i);
        svfloat32_t xEven = svcvt_f32_f16_x(pgB16, srcX);
        svfloat32_t xOdd = CvtOddF32F16(srcX, pgB16, pgB32);

        svfloat32_t vecEven = svsub_f32_x(pgB32, VectorComputation_FLOAT_ONE, xEven);
        svfloat32_t tmpEven = svadd_f32_x(pgB32, VectorComputation_FLOAT_ONE, xEven);
        vecEven = svmul_f32_m(pgB32, vecEven, tmpEven);
        vecEven = svmul_f32_m(pgB32, vecEven, ddEven);

        svfloat32_t vecOdd = svsub_f32_x(pgB32, VectorComputation_FLOAT_ONE, xOdd);
        svfloat32_t tmpOdd = svadd_f32_x(pgB32, VectorComputation_FLOAT_ONE, xOdd);
        vecOdd = svmul_f32_m(pgB32, vecOdd, tmpOdd);
        vecOdd = svmul_f32_m(pgB32, vecOdd, ddOdd);

        svfloat16_t res = svcvt_f16_f32_x(pgB32, vecEven);
        srcX = CvtOddF16F32(vecOdd, pgB32);
        res = svsel_f16(pgB32, res, srcX);
        svst1_f16(pgB16, ds + i, res);
    }
}