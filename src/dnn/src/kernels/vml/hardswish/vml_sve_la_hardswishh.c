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
#define VML_VectorComputation_UNFOLD_N 6
#include "kernels/vml/common_macro.h"

static void HardSwishMainKernelFwdF16(svfloat32_t va, svfloat32_t vb, const __fp16 *src, __fp16 *dst, int step)
{
    svbool_t pgB16 = svptrue_b16();
    svbool_t pgB32 = svptrue_b32();

    VML_VectorComputation_DECLARE(svfloat32_t, xEven);
    VML_VectorComputation_DECLARE(svfloat32_t, xOdd);

    LOAD_CONVERT_FP16_TO_FP32_1(src, pgB16, xEven, xOdd, step);

    // hardswish compute
    VML_VectorComputation_DECLARE(svfloat32_t, resEven);
    VML_VectorComputation_DECLARE(svfloat32_t, resOdd);
    VML_VectorComputation_FUNC_PG_N13_3PARA(resEven, svmad_f32_x, pgB32, va, xEven, vb);
    VML_VectorComputation_FUNC_PG_N13_3PARA(resOdd, svmad_f32_x, pgB32, va, xOdd, vb);
    VML_VectorComputation_FUNC_PG_N1_2PARA(resEven, svmin_f32_x, pgB32, VectorComputation_FLOAT_ONE, resEven);
    VML_VectorComputation_FUNC_PG_N1_2PARA(resOdd, svmin_f32_x, pgB32, VectorComputation_FLOAT_ONE, resOdd);
    VML_VectorComputation_FUNC_PG_N1_2PARA(resEven, svmax_f32_x, pgB32, VectorComputation_FLOAT_ZERO, resEven);
    VML_VectorComputation_FUNC_PG_N1_2PARA(resOdd, svmax_f32_x, pgB32, VectorComputation_FLOAT_ZERO, resOdd);
    VML_VectorComputation_FUNC_PG_2PARA(resEven, svmul_f32_x, pgB32, resEven, xEven);
    VML_VectorComputation_FUNC_PG_2PARA(resOdd, svmul_f32_x, pgB32, resOdd, xOdd);

    CONVERT_STORE_FP32_TO_FP16(dst, pgB16, resEven, resOdd, step);
}

void HardSwishSveFWDF16(const int64_t len, const float alpha, const float beta, const __fp16 *src, __fp16 *dst)
{
    int64_t i;
    int64_t loopHardSwishh = svcnth();
    svfloat32_t va = svdup_n_f32(alpha);
    svfloat32_t vb = svdup_n_f32(beta);

    for (i = 0; i <= len - loopHardSwishh * VML_VectorComputation_UNFOLD_N;
         i += loopHardSwishh * VML_VectorComputation_UNFOLD_N) {
        HardSwishMainKernelFwdF16(va, vb, src + i, dst + i, loopHardSwishh);
    }
    for (; i < len; i += loopHardSwishh) {
        svbool_t pgB16 = svwhilelt_b16_s64(i, len);
        svbool_t pgB32 = svwhilelt_b32_s64(i, len);

        // convert to fp32
        svfloat16_t x = svld1_f16(pgB16, src + i);
        svfloat32_t xEven = svcvt_f32_f16_x(pgB16, x);
        svfloat32_t xOdd = CvtOddF32F16(x, pgB16, pgB32);

        // hardswish compute
        svfloat32_t resEven = svmad_f32_x(pgB32, va, xEven, vb); /* alpha * s + beta */
        svfloat32_t resOdd = svmad_f32_x(pgB32, va, xOdd, vb);
        resEven = svmin_f32_x(pgB32, VectorComputation_FLOAT_ONE, resEven); /* min(1, alpha * s + beta) */
        resOdd = svmin_f32_x(pgB32, VectorComputation_FLOAT_ONE, resOdd);
        resEven = svmax_f32_x(pgB32, VectorComputation_FLOAT_ZERO, resEven); /* max(0, min(1, alpha * s + beta)) */
        resOdd = svmax_f32_x(pgB32, VectorComputation_FLOAT_ZERO, resOdd);
        resEven = svmul_f32_x(pgB32, resEven, xEven); /* s * max(0, min(1, alpha * s + beta)) */
        resOdd = svmul_f32_x(pgB32, resOdd, xOdd);

        // convert back to fp16
        svfloat16_t res = svcvt_f16_f32_x(pgB32, resEven);
        x = CvtOddF16F32(resOdd, pgB32); /* resOdd(f16) */
        res = svsel_f16(pgB32, res, x);

        svst1_f16(pgB16, dst + i, res);
    }
}

static void HardSwishMainKernelBwdF16(svfloat32_t va, svfloat32_t vb, svfloat32_t halfVb, const __fp16 *src,
                                      const __fp16 *dd, __fp16 *ds, int step)
{
    svbool_t pgB32 = svptrue_b32();
    svbool_t pgB16 = svptrue_b16();
    VML_VectorComputation_DECLARE(svfloat32_t, xEven);
    VML_VectorComputation_DECLARE(svfloat32_t, xOdd);
    VML_VectorComputation_DECLARE(svfloat32_t, ddOdd);
    VML_VectorComputation_DECLARE(svfloat32_t, ddEven);
    VML_VectorComputation_DECLARE(svbool_t, pgRetDD);
    VML_VectorComputation_DECLARE(svbool_t, pgMain);
    // hardswish compute
    VML_VectorComputation_DECLARE(svfloat32_t, resEven);
    VML_VectorComputation_DECLARE(svfloat32_t, resOdd);
    LOAD_CONVERT_FP16_TO_FP32_2(src, dd, pgB16, xEven, xOdd, ddEven, ddOdd, step);

    VML_VectorComputation_FUNC_PG_N13_3PARA(resEven, svmad_f32_x, pgB32, va, xEven, vb);
    VML_VectorComputation_FUNC_PG_N13_3PARA(resOdd, svmad_f32_x, pgB32, va, xOdd, vb);
    VML_VectorComputation_FUNC_PG_N2_2PARA(pgRetDD, svcmpge, pgB32, resEven, VectorComputation_FLOAT_ONE);
    VML_VectorComputation_FUNC_PG_N2_2PARA(pgMain, svcmpgt, pgB32, resEven, VectorComputation_FLOAT_ZERO);
    VML_VectorComputation_FUNC_NOPG_N3_3PARA(pgMain, svcmplt, pgMain, resEven, VectorComputation_FLOAT_ONE);
    VML_VectorComputation_FUNC_PG_N23_3PARA(resEven, svmad_f32_x, pgB32, xEven, va, halfVb);
    VML_VectorComputation_FUNC_PG_N2_2PARA(resEven, svmul_f32_x, pgB32, resEven, VectorComputation_FLOAT_TWO);
    VML_VectorComputation_FUNC_NOPG_3PARA(resEven, svmul_f32_z, pgMain, resEven, ddEven);
    VML_VectorComputation_FUNC_NOPG_3PARA(resEven, svsel_f32, pgRetDD, ddEven, resEven);

    VML_VectorComputation_FUNC_PG_N2_2PARA(pgRetDD, svcmpge, pgB32, resOdd, VectorComputation_FLOAT_ONE);
    VML_VectorComputation_FUNC_PG_N2_2PARA(pgMain, svcmpgt, pgB32, resOdd, VectorComputation_FLOAT_ZERO);
    VML_VectorComputation_FUNC_NOPG_N3_3PARA(pgMain, svcmplt, pgMain, resOdd, VectorComputation_FLOAT_ONE);
    VML_VectorComputation_FUNC_PG_N23_3PARA(resOdd, svmad_f32_x, pgB32, xOdd, va, halfVb);
    VML_VectorComputation_FUNC_PG_N2_2PARA(resOdd, svmul_f32_x, pgB32, resOdd, VectorComputation_FLOAT_TWO);
    VML_VectorComputation_FUNC_NOPG_3PARA(resOdd, svmul_f32_z, pgMain, resOdd, ddOdd);
    VML_VectorComputation_FUNC_NOPG_3PARA(resOdd, svsel_f32, pgRetDD, ddOdd, resOdd);

    CONVERT_STORE_FP32_TO_FP16(ds, pgB16, resEven, resOdd, step);
}

void HardSwishSveBWDF16(const int64_t len, const float alpha, const float beta, const __fp16 *src, const __fp16 *dd,
                        __fp16 *ds)
{
    int64_t i;
    int64_t loopHardSwishh = svcnth();
    svfloat32_t va = svdup_n_f32(alpha);
    svfloat32_t vb = svdup_n_f32(beta);
    svfloat32_t halfVb = svdup_n_f32(beta * FLOAT_HALF);

    for (i = 0; i <= len - loopHardSwishh * VML_VectorComputation_UNFOLD_N;
         i += loopHardSwishh * VML_VectorComputation_UNFOLD_N) {
        HardSwishMainKernelBwdF16(va, vb, halfVb, src + i, dd + i, ds + i, loopHardSwishh);
    }
    for (; i < len; i += loopHardSwishh) {
        svbool_t pgB16 = svwhilelt_b16_s64(i, len);
        svbool_t pgB32 = svwhilelt_b32_s64(i, len);

        svfloat16_t x = svld1_f16(pgB16, src + i);
        svfloat32_t xEven = svcvt_f32_f16_x(pgB16, x);
        svfloat32_t xOdd = CvtOddF32F16(x, pgB16, pgB32);

        svfloat16_t ddX = svld1_f16(pgB16, dd + i);
        svfloat32_t ddEven = svcvt_f32_f16_x(pgB16, ddX);
        svfloat32_t ddOdd = CvtOddF32F16(ddX, pgB16, pgB32);

        // hardswish compute
        svfloat32_t resEven = svmad_f32_x(pgB32, va, xEven, vb); /* alpha * s + beta */
        svfloat32_t resOdd = svmad_f32_x(pgB32, va, xOdd, vb);

        svbool_t pgRetDD = svcmpge(pgB32, resEven, VectorComputation_FLOAT_ONE); /* alpha * s + beta >= 1 */
        svbool_t pgMain = svcmpgt(pgB32, resEven, VectorComputation_FLOAT_ZERO); /* 0 < alpha * s + beta < 1 */
        pgMain = svcmplt(pgMain, resEven, VectorComputation_FLOAT_ONE);          /* 0 < alpha * s + beta < 1 */
        resEven = svmad_f32_x(pgB32, xEven, va, halfVb);                         /* alpha * s + beta / 2 */
        resEven = svmul_f32_x(pgB32, resEven, VectorComputation_FLOAT_TWO);      /* 2 * alpha * s + beta */
        resEven = svmul_f32_z(pgMain, resEven, ddEven); /* dd * (2 * alpha * s + beta) for pgMain, zero for others */
        resEven = svsel_f32(pgRetDD, ddEven, resEven);

        pgRetDD = svcmpge(pgB32, resOdd, VectorComputation_FLOAT_ONE);    /* alpha * s + beta >= 1 */
        pgMain = svcmpgt(pgB32, resOdd, VectorComputation_FLOAT_ZERO);    /* 0 < alpha * s + beta < 1 */
        pgMain = svcmplt(pgMain, resOdd, VectorComputation_FLOAT_ONE);    /* 0 < alpha * s + beta < 1 */
        resOdd = svmad_f32_x(pgB32, xOdd, va, halfVb);                    /* alpha * s + beta / 2 */
        resOdd = svmul_f32_x(pgB32, resOdd, VectorComputation_FLOAT_TWO); /* 2 * alpha * s + beta */
        resOdd = svmul_f32_z(pgMain, resOdd, ddOdd); /* dd * (2 * alpha * s + beta) for pgMain, zero for others */
        resOdd = svsel_f32(pgRetDD, ddOdd, resOdd);

        // convert back to fp16
        svfloat16_t res = svcvt_f16_f32_x(pgB32, resEven); /* resEven(f16) */
        x = CvtOddF16F32(resOdd, pgB32);                   /* resOdd(f16) */
        res = svsel_f16(pgB32, res, x);

        svst1_f16(pgB16, ds + i, res);
    }
}
