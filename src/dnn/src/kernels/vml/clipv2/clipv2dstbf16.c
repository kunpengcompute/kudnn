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
#define VML_VC_UNFOLD_N 6
#include "kernels/vml/common_macro.h"

static void ClipV2DstMainKernelBwdBF16(float alpha, float beta, const __bf16 *src, const __bf16 *dd, __bf16 *ds,
    int step)
{
    svbool_t pgB16 = svptrue_b16();
    svbool_t pgB32 = svptrue_b32();

    VML_VC_DECLARE(svfloat32_t, xBot);
    VML_VC_DECLARE(svfloat32_t, xTop);
    VML_VC_DECLARE(svfloat32_t, ddBot);
    VML_VC_DECLARE(svfloat32_t, ddTop);

    LOAD_CONVERT_BF16_TO_FP32_2(src, dd, pgB16, xBot, xTop, ddBot, ddTop, step);

    // clip_v2_dst compute
    VML_VC_DECLARE(svbool_t, pgRetZeroUp);
    VML_VC_DECLARE(svbool_t, pgRetZeroDown);
    VML_VC_DECLARE(svfloat32_t, resBot);
    VML_VC_DECLARE(svfloat32_t, resTop);

    VML_VC_FUNC_PG_N2_2PARA(pgRetZeroUp, svcmpge_n_f32, pgB32, xBot, beta);
    VML_VC_FUNC_NOPG_N3_3PARA(resBot, svdup_f32_m, ddBot, pgRetZeroUp, FLOAT_ZERO);
    VML_VC_FUNC_PG_N2_2PARA(pgRetZeroDown, svcmple_n_f32, pgB32, xBot, alpha);
    VML_VC_FUNC_NOPG_N3_3PARA(resBot, svdup_f32_m, resBot, pgRetZeroDown, FLOAT_ZERO);
    VML_VC_FUNC_PG_N2_2PARA(pgRetZeroUp, svcmpge_n_f32, pgB32, xTop, beta);
    VML_VC_FUNC_NOPG_N3_3PARA(resTop, svdup_f32_m, ddTop, pgRetZeroUp, FLOAT_ZERO);
    VML_VC_FUNC_PG_N2_2PARA(pgRetZeroDown, svcmple_n_f32, pgB32, xTop, alpha);
    VML_VC_FUNC_NOPG_N3_3PARA(resTop, svdup_f32_m, resTop, pgRetZeroDown, FLOAT_ZERO);

    CONVERT_STORE_FP32_TO_BF16(ds, pgB32, pgB16, resBot, resTop, step);
}

void ClipV2DstVCBWDBF16(const int64_t len, const float alpha, const float beta, const __bf16 *src, const __bf16 *dd,
    __bf16 *ds)
{
    int64_t i = 0;
    int64_t loopClipDstV2h = svcnth();
    int numF32 = svcntw();
    for (; i <= len - loopClipDstV2h * VML_VC_UNFOLD_N; i += loopClipDstV2h * VML_VC_UNFOLD_N) {
        ClipV2DstMainKernelBwdBF16(alpha, beta, src + i, dd + i, ds + i, loopClipDstV2h);
    }
    for (; i < len; i += loopClipDstV2h) {
        svbool_t pgB16 = svwhilelt_b16_s64(i, len);
        svbool_t pgB32 = svwhilelt_b32_s64(i, len);
        svbool_t pgF32Bot = pgB32;
        svbool_t pgF32Top = svwhilelt_b32_s64(i + numF32, len);

        svbfloat16_t srcX = svld1_bf16(pgB16, src + i);
        svbfloat16_t ddX = svld1_bf16(pgB16, dd + i);

        // convert to fp32
        svfloat32_t xBot = SvCvtF32BF16BottomX(srcX);
        svfloat32_t xTop = SvCvtF32BF16TopX(srcX);

        svfloat32_t resBot = SvCvtF32BF16BottomX(ddX);
        svfloat32_t resTop = SvCvtF32BF16TopX(ddX);

        resBot = svdup_f32_m(resBot, svcmpge_n_f32(pgB32, xBot, beta), FLOAT_ZERO);
        resBot = svdup_f32_m(resBot, svcmple_n_f32(pgB32, xBot, alpha), FLOAT_ZERO);
        resTop = svdup_f32_m(resTop, svcmpge_n_f32(pgB32, xTop, beta), FLOAT_ZERO);
        resTop = svdup_f32_m(resTop, svcmple_n_f32(pgB32, xTop, alpha), FLOAT_ZERO);

        // convert back to bf16
        svbfloat16_t res = SvCvtBF16F32X(pgF32Bot, resBot, pgF32Top, resTop);
        svst1_bf16(pgB16, ds + i, res);
    }
}
