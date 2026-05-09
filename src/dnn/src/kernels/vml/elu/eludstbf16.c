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
#define VML_VectorComputation_UNFOLD_N 2
#include "kernels/vml/common_macro.h"

void EluDstSveFWDBF16(const int64_t len, const float alpha, const __bf16 *src, __bf16 *dst)
{
    int64_t i = 0;
    for (; i < len; i++) {
        uint16_t xU16 = AsUint16(src[i]);
        // x > 0
        if (xU16 <= F16_ABS_MASK) {
            dst[i] = src[i];
            continue;
        }
        uint32_t s = SveCvtU16ToU32(xU16);
        float x = AsFloat(s);
        uint32_t bits = AsUint(alpha * expm1f(x));
        uint16_t resU16 = SveCvtU32ToU16(bits);

        dst[i] = AsBF16(resU16);
    }
}

static void EluMainKernelBwdBF16(svfloat32_t a, const __bf16 *src, const __bf16 *dd, __bf16 *ds, int step)
{
    svbool_t pgB16 = svptrue_b16();
    svbool_t pgB32 = svptrue_b32();

    VML_VectorComputation_DECLARE(svfloat32_t, xBot);
    VML_VectorComputation_DECLARE(svfloat32_t, xTop);
    VML_VectorComputation_DECLARE(svfloat32_t, ddBot);
    VML_VectorComputation_DECLARE(svfloat32_t, ddTop);

    LOAD_CONVERT_BF16_TO_FP32_2(src, dd, pgB16, xBot, xTop, ddBot, ddTop, step);

    // elu compute
    VML_VectorComputation_DECLARE(svbool_t, pgBot);
    VML_VectorComputation_DECLARE(svbool_t, pgTop);
    VML_VectorComputation_FUNC_PG_N2_2PARA(pgBot, svcmple_n_f32, pgB32, xBot, FLOAT_ZERO);
    VML_VectorComputation_FUNC_PG_N2_2PARA(pgTop, svcmple_n_f32, pgB32, xTop, FLOAT_ZERO);

    VML_VectorComputation_DECLARE(svfloat32_t, addBot);
    VML_VectorComputation_DECLARE(svfloat32_t, addTop);
    VML_VectorComputation_FUNC_UPG_N2_2PARA(addBot, svadd_f32_m, pgBot, xBot, a);
    VML_VectorComputation_FUNC_UPG_N2_2PARA(addTop, svadd_f32_m, pgTop, xTop, a);

    VML_VectorComputation_DECLARE(svfloat32_t, resBot);
    VML_VectorComputation_DECLARE(svfloat32_t, resTop);
    VML_VectorComputation_FUNC_NOPG_3PARA(resBot, svmul_f32_m, pgBot, ddBot, addBot);
    VML_VectorComputation_FUNC_NOPG_3PARA(resTop, svmul_f32_m, pgTop, ddTop, addTop);

    VML_VectorComputation_FUNC_PG_1PARA(resBot, SvF32CvtBf16InfSpecial, pgB32, resBot);
    VML_VectorComputation_FUNC_PG_1PARA(resTop, SvF32CvtBf16InfSpecial, pgB32, resTop);

    CONVERT_STORE_FP32_TO_BF16(ds, pgB16, resBot, resTop, step);
}

void EluDstSveBWDBF16(const int64_t len, const float alpha, const __bf16 *src, const __bf16 *dd, __bf16 *ds)
{
    int64_t i = 0;
    int64_t loopEluDsth = svcnth();
    svfloat32_t va = svdup_n_f32(alpha);

    for (; i <= len - loopEluDsth * VML_VectorComputation_UNFOLD_N; i += loopEluDsth * VML_VectorComputation_UNFOLD_N) {
        EluMainKernelBwdBF16(va, src + i, dd + i, ds + i, loopEluDsth);
    }
    for (; i < len; i += loopEluDsth) {
        svbool_t pgB16 = svwhilelt_b16_s64(i, len);
        svbool_t pgB32 = svwhilelt_b32_s64(i, len);

        svbfloat16_t srcX = svld1_bf16(pgB16, src + i);
        svbfloat16_t ddX = svld1_bf16(pgB16, dd + i);

        svfloat32_t xBot = SvCvtF32BF16BottomX(srcX);
        svfloat32_t xTop = SvCvtF32BF16TopX(srcX);

        svfloat32_t resBot = SvCvtF32BF16BottomX(ddX);
        svfloat32_t resTop = SvCvtF32BF16TopX(ddX);

        svbool_t pgBot = svcmple_n_f32(pgB32, xBot, FLOAT_ZERO);
        svbool_t pgTop = svcmple_n_f32(pgB32, xTop, FLOAT_ZERO);

        resBot = svmul_f32_m(pgBot, resBot, svadd_f32_m(pgBot, xBot, va));
        resTop = svmul_f32_m(pgTop, resTop, svadd_f32_m(pgTop, xTop, va));

        resBot = SvF32CvtBf16InfSpecial(pgB32, resBot);
        resTop = SvF32CvtBf16InfSpecial(pgB32, resTop);

        // convert back to bf16
        svbfloat16_t res = SvCvtBF16F32FullX(resBot, resTop);
        svst1_bf16(pgB16, ds + i, res);
    }
}
