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

static void LogisticMainKernelF16(const __fp16 *src, const __fp16 *dd, int step, __fp16 *ds)
{
    svbool_t pgB32 = svptrue_b32();
    svbool_t pgB16 = svptrue_b16();

    VML_VC_DECLARE(svfloat32_t, xEven);
    VML_VC_DECLARE(svfloat32_t, xOdd);
    VML_VC_DECLARE(svfloat32_t, ddEven);
    VML_VC_DECLARE(svfloat32_t, ddOdd);

    LOAD_CONVERT_FP16_TO_FP32_2(src, dd, pgB16, xEven, xOdd, ddEven, ddOdd, step);

    // logistic_dst compute
    VML_VC_DECLARE(svfloat32_t, resEven);
    VML_VC_FUNC_PG_N1_2PARA(resEven, svsub_f32_x, pgB32, VC_FLOAT_ONE, xEven);
    VML_VC_FUNC_PG_2PARA(resEven, svmul_f32_x, pgB32, resEven, xEven);
    VML_VC_FUNC_PG_2PARA(resEven, svmul_f32_x, pgB32, resEven, ddEven);

    VML_VC_DECLARE(svfloat32_t, resOdd);
    VML_VC_FUNC_PG_N1_2PARA(resOdd, svsub_f32_x, pgB32, VC_FLOAT_ONE, xOdd);
    VML_VC_FUNC_PG_2PARA(resOdd, svmul_f32_x, pgB32, resOdd, xOdd);
    VML_VC_FUNC_PG_2PARA(resOdd, svmul_f32_x, pgB32, resOdd, ddOdd);

    CONVERT_STORE_FP32_TO_FP16(ds, pgB16, resEven, resOdd, step);
}

void LogisticDstVCBWDF16(const int64_t len, const __fp16 *src, const __fp16 *dd, __fp16 *ds)
{
    int size = svcnth();
    int64_t i = 0;
    for (; i <= len - size * VML_VC_UNFOLD_N; i += size * VML_VC_UNFOLD_N) {
        LogisticMainKernelF16(src + i, dd + i, size, ds + i);
    }

    for (; i < len; i += size) {
        svbool_t pgB16 = svwhilelt_b16_s32(i, len);
        svbool_t pgB32 = svwhilelt_b32_s32(i, len);

        svfloat16_t srcX = svld1_f16(pgB16, src + i);
        svfloat32_t xEven = svcvt_f32_f16_x(pgB16, srcX);
        svfloat32_t xOdd = CvtOddF32F16(srcX, pgB16, pgB32);

        svfloat16_t ddX = svld1_f16(pgB16, dd + i);
        svfloat32_t ddEven = svcvt_f32_f16_x(pgB16, ddX);
        svfloat32_t ddOdd = CvtOddF32F16(ddX, pgB16, pgB32);
        svfloat32_t tmp = svsub_f32_x(pgB32, VC_FLOAT_ONE, xEven);
        xEven = svmul_f32_m(pgB32, xEven, tmp);
        xEven = svmul_f32_m(pgB32, xEven, ddEven);

        tmp = svsub_f32_x(pgB32, VC_FLOAT_ONE, xOdd);
        xOdd = svmul_f32_m(pgB32, xOdd, tmp);
        xOdd = svmul_f32_m(pgB32, xOdd, ddOdd);

        svfloat16_t res = svcvt_f16_f32_x(pgB32, xEven);
        srcX = CvtOddF16F32(xOdd, pgB32);
        res = svsel_f16(pgB32, res, srcX);
        svst1_f16(pgB16, ds + i, res);
    }
}