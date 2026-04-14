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

static void ClipV2DstMainKernelBwdF16(const float alpha, const float beta, const __fp16 *src, const __fp16 *dd,
    __fp16 *ds, int step)
{
    svbool_t pgB32 = svptrue_b32();
    svbool_t pgB16 = svptrue_b16();

    VML_VectorComputation_DECLARE(svfloat32_t, xEven);
    VML_VectorComputation_DECLARE(svfloat32_t, xOdd);
    VML_VectorComputation_DECLARE(svfloat32_t, ddEven);
    VML_VectorComputation_DECLARE(svfloat32_t, ddOdd);

    LOAD_CONVERT_FP16_TO_FP32_2(src, dd, pgB16, xEven, xOdd, ddEven, ddOdd, step);

    // clip_v2_dst compute
    VML_VectorComputation_DECLARE(svbool_t, pgRetZeroEven);
    VML_VectorComputation_DECLARE(svfloat32_t, resEven);
    VML_VectorComputation_FUNC_PG_N2_2PARA(pgRetZeroEven, svcmpge_n_f32, pgB32, xEven, beta);
    VML_VectorComputation_FUNC_NOPG_N3_3PARA(resEven, svdup_f32_m, ddEven, pgRetZeroEven, FLOAT_ZERO);

    VML_VectorComputation_DECLARE(svbool_t, pgRetZeroOdd);
    VML_VectorComputation_DECLARE(svfloat32_t, resOdd);
    VML_VectorComputation_FUNC_PG_N2_2PARA(pgRetZeroOdd, svcmpge_n_f32, pgB32, xOdd, beta);
    VML_VectorComputation_FUNC_NOPG_N3_3PARA(resOdd, svdup_f32_m, ddOdd, pgRetZeroOdd, FLOAT_ZERO);

    VML_VectorComputation_FUNC_PG_N2_2PARA(pgRetZeroEven, svcmple_n_f32, pgB32, xEven, alpha);
    VML_VectorComputation_FUNC_NOPG_N3_3PARA(resEven, svdup_f32_m, resEven, pgRetZeroEven, FLOAT_ZERO);
    VML_VectorComputation_FUNC_PG_N2_2PARA(pgRetZeroOdd, svcmple_n_f32, pgB32, xOdd, alpha);
    VML_VectorComputation_FUNC_NOPG_N3_3PARA(resOdd, svdup_f32_m, resOdd, pgRetZeroOdd, FLOAT_ZERO);

    CONVERT_STORE_FP32_TO_FP16(ds, pgB16, resEven, resOdd, step);
}

void ClipV2DstSveBWDF16(const int64_t len, const float alpha, const float beta, const __fp16 *src, const __fp16 *dd,
    __fp16 *ds)
{
    int64_t i = 0;
    int64_t loopClipDstV2h = svcnth();

    for (; i <= len - loopClipDstV2h * VML_VectorComputation_UNFOLD_N; i += loopClipDstV2h * VML_VectorComputation_UNFOLD_N) {
        ClipV2DstMainKernelBwdF16(alpha, beta, src + i, dd + i, ds + i, loopClipDstV2h);
    }
    for (; i < len; i += loopClipDstV2h) {
        svbool_t pgB16 = svwhilelt_b16_s32(i, len);
        svbool_t pgB32 = svwhilelt_b32_s32(i, len);

        svfloat16_t srcX = svld1_f16(pgB16, src + i);
        svfloat32_t xEven = svcvt_f32_f16_x(pgB16, srcX);
        svfloat32_t xOdd = CvtOddF32F16(srcX, pgB16, pgB32);

        svfloat16_t ddX = svld1_f16(pgB16, dd + i);
        svfloat32_t ddEven = svcvt_f32_f16_x(pgB16, ddX);
        svfloat32_t ddOdd = CvtOddF32F16(ddX, pgB16, pgB32);

        svfloat32_t resEven = svdup_f32_m(ddEven, svcmpge_n_f32(pgB32, xEven, beta), FLOAT_ZERO);
        svfloat32_t resOdd = svdup_f32_m(ddOdd, svcmpge_n_f32(pgB32, xOdd, beta), FLOAT_ZERO);

        resEven = svdup_f32_m(resEven, svcmple_n_f32(pgB32, xEven, alpha), FLOAT_ZERO);
        resOdd = svdup_f32_m(resOdd, svcmple_n_f32(pgB32, xOdd, alpha), FLOAT_ZERO);

        svfloat16_t res = svcvt_f16_f32_x(pgB32, resEven);

        srcX = CvtOddF16F32(resOdd, pgB32);
        res = svsel_f16(pgB32, res, srcX);

        svst1_f16(pgB16, ds + i, res);
    }
}