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
#define VML_VectorComputation_UNFOLD_N 4
#include "kernels/vml/common_macro.h"

static void HardSwishMainKernelFwdBF16(svfloat32_t va, svfloat32_t vb, const __bf16 *src, __bf16 *dst, int step)
{
    svbool_t pgB16 = svptrue_b16();
    svbool_t pgB32 = svptrue_b32();

    VML_VectorComputation_DECLARE(svfloat32_t, xBot);
    VML_VectorComputation_DECLARE(svfloat32_t, xTop);

    LOAD_CONVERT_BF16_TO_FP32_1(src, pgB16, xBot, xTop, step);

    // hardswish compute
    VML_VectorComputation_DECLARE(svfloat32_t, resBot);
    VML_VectorComputation_DECLARE(svfloat32_t, resTop);
    VML_VectorComputation_FUNC_PG_N13_3PARA(resBot, svmad_f32_x, pgB32, va, xBot, vb);
    VML_VectorComputation_FUNC_PG_N13_3PARA(resTop, svmad_f32_x, pgB32, va, xTop, vb);
    VML_VectorComputation_FUNC_PG_N1_2PARA(resBot, svmin_f32_x, pgB32, VectorComputation_FLOAT_ONE, resBot);
    VML_VectorComputation_FUNC_PG_N1_2PARA(resTop, svmin_f32_x, pgB32, VectorComputation_FLOAT_ONE, resTop);
    VML_VectorComputation_FUNC_PG_N1_2PARA(resBot, svmax_f32_x, pgB32, VectorComputation_FLOAT_ZERO, resBot);
    VML_VectorComputation_FUNC_PG_N1_2PARA(resTop, svmax_f32_x, pgB32, VectorComputation_FLOAT_ZERO, resTop);
    VML_VectorComputation_FUNC_PG_2PARA(resBot, svmul_f32_x, pgB32, resBot, xBot);
    VML_VectorComputation_FUNC_PG_2PARA(resTop, svmul_f32_x, pgB32, resTop, xTop);

    CONVERT_STORE_FP32_TO_BF16(dst, pgB16, resBot, resTop, step);
}

void HardSwishSveFWDBF16(const int64_t len, const float alpha, const float beta, const __bf16 *src, __bf16 *dst)
{
    int64_t i;
    int64_t loopHardSwishh = svcnth();
    svfloat32_t va = svdup_n_f32(alpha);
    svfloat32_t vb = svdup_n_f32(beta);

    for (i = 0; i <= len - loopHardSwishh * VML_VectorComputation_UNFOLD_N;
         i += loopHardSwishh * VML_VectorComputation_UNFOLD_N) {
        HardSwishMainKernelFwdBF16(va, vb, src + i, dst + i, loopHardSwishh);
    }
    for (; i < len; i += loopHardSwishh) {
        svbool_t pgB16 = svwhilelt_b16_s64(i, len);
        svbool_t pgB32 = svwhilelt_b32_s64(i, len);

        svbfloat16_t x = svld1_bf16(pgB16, src + i);

        // convert to fp32
        svfloat32_t xBot = SvCvtF32BF16BottomX(x);
        svfloat32_t xTop = SvCvtF32BF16TopX(x);

        // hardswish compute
        svfloat32_t resBot = svmad_f32_x(pgB32, va, xBot, vb); /* alpha * s + beta */
        svfloat32_t resTop = svmad_f32_x(pgB32, va, xTop, vb);
        resBot = svmin_f32_x(pgB32, VectorComputation_FLOAT_ONE, resBot); /* min(1, alpha * s + beta) */
        resTop = svmin_f32_x(pgB32, VectorComputation_FLOAT_ONE, resTop);
        resBot = svmax_f32_x(pgB32, VectorComputation_FLOAT_ZERO, resBot); /* max(0, min(1, alpha * s + beta)) */
        resTop = svmax_f32_x(pgB32, VectorComputation_FLOAT_ZERO, resTop);
        resBot = svmul_f32_x(pgB32, resBot, xBot); /* s * max(0, min(1, alpha * s + beta)) */
        resTop = svmul_f32_x(pgB32, resTop, xTop);

        // convert back to bf16
        svbfloat16_t res = SvCvtBF16F32FullX(resBot, resTop);
        svst1_bf16(pgB16, dst + i, res);
    }
}

static void HardSwishMainKernelBwdBF16(svfloat32_t va, svfloat32_t vb, svfloat32_t halfVb, const __bf16 *src,
                                       const __bf16 *dd, __bf16 *ds, int step)
{
    svbool_t pgB16 = svptrue_b16();
    svbool_t pgB32 = svptrue_b32();

    VML_VectorComputation_DECLARE(svfloat32_t, xBot);
    VML_VectorComputation_DECLARE(svfloat32_t, xTop);
    VML_VectorComputation_DECLARE(svfloat32_t, ddBot);
    VML_VectorComputation_DECLARE(svfloat32_t, ddTop);

    LOAD_CONVERT_BF16_TO_FP32_2(src, dd, pgB16, xBot, xTop, ddBot, ddTop, step);

    // hardswish compute
    VML_VectorComputation_DECLARE(svfloat32_t, resBot);
    VML_VectorComputation_DECLARE(svfloat32_t, resTop);
    VML_VectorComputation_FUNC_PG_N13_3PARA(resBot, svmad_f32_x, pgB32, va, xBot, vb);
    VML_VectorComputation_FUNC_PG_N13_3PARA(resTop, svmad_f32_x, pgB32, va, xTop, vb);

    VML_VectorComputation_DECLARE(svbool_t, pgRetDD);
    VML_VectorComputation_DECLARE(svbool_t, pgMain);
    VML_VectorComputation_FUNC_PG_N2_2PARA(pgRetDD, svcmpge, pgB32, resBot, VectorComputation_FLOAT_ONE);
    VML_VectorComputation_FUNC_PG_N2_2PARA(pgMain, svcmpgt, pgB32, resBot, VectorComputation_FLOAT_ZERO);
    VML_VectorComputation_FUNC_NOPG_N3_3PARA(pgMain, svcmplt, pgMain, resBot, VectorComputation_FLOAT_ONE);
    VML_VectorComputation_FUNC_PG_N23_3PARA(resBot, svmad_f32_x, pgB32, xBot, va, halfVb);
    VML_VectorComputation_FUNC_PG_N2_2PARA(resBot, svmul_f32_x, pgB32, resBot, VectorComputation_FLOAT_TWO);
    VML_VectorComputation_FUNC_NOPG_3PARA(resBot, svmul_f32_z, pgMain, resBot, ddBot);
    VML_VectorComputation_FUNC_NOPG_3PARA(resBot, svsel_f32, pgRetDD, ddBot, resBot);

    VML_VectorComputation_FUNC_PG_N2_2PARA(pgRetDD, svcmpge, pgB32, resTop, VectorComputation_FLOAT_ONE);
    VML_VectorComputation_FUNC_PG_N2_2PARA(pgMain, svcmpgt, pgB32, resTop, VectorComputation_FLOAT_ZERO);
    VML_VectorComputation_FUNC_NOPG_N3_3PARA(pgMain, svcmplt, pgMain, resTop, VectorComputation_FLOAT_ONE);
    VML_VectorComputation_FUNC_PG_N23_3PARA(resTop, svmad_f32_x, pgB32, xTop, va, halfVb);
    VML_VectorComputation_FUNC_PG_N2_2PARA(resTop, svmul_f32_x, pgB32, resTop, VectorComputation_FLOAT_TWO);
    VML_VectorComputation_FUNC_NOPG_3PARA(resTop, svmul_f32_z, pgMain, resTop, ddTop);
    VML_VectorComputation_FUNC_NOPG_3PARA(resTop, svsel_f32, pgRetDD, ddTop, resTop);

    VML_VectorComputation_FUNC_PG_1PARA(resBot, SvF32CvtBf16InfSpecial, pgB32, resBot);
    VML_VectorComputation_FUNC_PG_1PARA(resTop, SvF32CvtBf16InfSpecial, pgB32, resTop);

    CONVERT_STORE_FP32_TO_BF16(ds, pgB16, resBot, resTop, step);
}

void HardSwishSveBWDBF16(const int64_t len, const float alpha, const float beta, const __bf16 *src, const __bf16 *dd,
                         __bf16 *ds)
{
    int64_t i;
    int64_t loopHardSwishh = svcnth();
    svfloat32_t va = svdup_n_f32(alpha);
    svfloat32_t vb = svdup_n_f32(beta);
    svfloat32_t halfVb = svdup_n_f32(beta * FLOAT_HALF);

    for (i = 0; i <= len - loopHardSwishh * VML_VectorComputation_UNFOLD_N;
         i += loopHardSwishh * VML_VectorComputation_UNFOLD_N) {
        HardSwishMainKernelBwdBF16(va, vb, halfVb, src + i, dd + i, ds + i, loopHardSwishh);
    }
    for (; i < len; i += loopHardSwishh) {
        svbool_t pgB16 = svwhilelt_b16_s64(i, len);
        svbool_t pgB32 = svwhilelt_b32_s64(i, len);

        svbfloat16_t x = svld1_bf16(pgB16, src + i);
        svbfloat16_t ddX = svld1_bf16(pgB16, dd + i);

        // convert to fp32
        svfloat32_t xBot = SvCvtF32BF16BottomX(x);
        svfloat32_t xTop = SvCvtF32BF16TopX(x);

        svfloat32_t ddBot = SvCvtF32BF16BottomX(ddX);
        svfloat32_t ddTop = SvCvtF32BF16TopX(ddX);

        // hardswish compute
        svfloat32_t resBot = svmad_f32_x(pgB32, va, xBot, vb); /* alpha * s + beta */
        svfloat32_t resTop = svmad_f32_x(pgB32, va, xTop, vb);

        svbool_t pgRetDD = svcmpge(pgB32, resBot, VectorComputation_FLOAT_ONE); /* alpha * s + beta >= 1 */
        svbool_t pgMain = svcmpgt(pgB32, resBot, VectorComputation_FLOAT_ZERO); /* 0 < alpha * s + beta < 1 */
        pgMain = svcmplt(pgMain, resBot, VectorComputation_FLOAT_ONE);          /* 0 < alpha * s + beta < 1 */
        resBot = svmad_f32_x(pgB32, xBot, va, halfVb);                          /* alpha * s + beta / 2 */
        resBot = svmul_f32_x(pgB32, resBot, VectorComputation_FLOAT_TWO);       /* 2 * alpha * s + beta */
        resBot = svmul_f32_z(pgMain, resBot, ddBot); /* dd * (2 * alpha * s + beta) for pgMain, zero for others */
        resBot = svsel_f32(pgRetDD, ddBot, resBot);

        pgRetDD = svcmpge(pgB32, resTop, VectorComputation_FLOAT_ONE);    /* alpha * s + beta >= 1 */
        pgMain = svcmpgt(pgB32, resTop, VectorComputation_FLOAT_ZERO);    /* 0 < alpha * s + beta < 1 */
        pgMain = svcmplt(pgMain, resTop, VectorComputation_FLOAT_ONE);    /* 0 < alpha * s + beta < 1 */
        resTop = svmad_f32_x(pgB32, xTop, va, halfVb);                    /* alpha * s + beta / 2 */
        resTop = svmul_f32_x(pgB32, resTop, VectorComputation_FLOAT_TWO); /* 2 * alpha * s + beta */
        resTop = svmul_f32_z(pgMain, resTop, ddTop); /* dd * (2 * alpha * s + beta) for pgMain, zero for others */
        resTop = svsel_f32(pgRetDD, ddTop, resTop);

        resBot = SvF32CvtBf16InfSpecial(pgB32, resBot);
        resTop = SvF32CvtBf16InfSpecial(pgB32, resTop);

        // convert back to bf16
        svbfloat16_t res = SvCvtBF16F32FullX(resBot, resTop);
        svst1_bf16(pgB16, ds + i, res);
    }
}
