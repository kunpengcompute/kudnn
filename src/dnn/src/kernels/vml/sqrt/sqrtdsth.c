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

static void SqrtMainKernelBwdF16(const __fp16 *src, const __fp16 *dd, __fp16 *ds, int step)
{
    svbool_t pg = svptrue_b16();
    VML_VC_DECLARE(svfloat16_t, src_x);
    VML_VC_DECLARE(svfloat16_t, dd_x);
    VML_VC_DECLARE(svfloat16_t, res);

    VML_VC_LD_STEP(src_x, svld1_f16, pg, src, step);

    VML_VC_LD_STEP(dd_x, svld1_f16, pg, dd, step);

    VML_VC_FUNC_PG_N2_2PARA(res, svmul_f16_x, pg, src_x, VML_VC_F16_TWO);
    VML_VC_FUNC_PG_2PARA(res, svdiv_f16_x, pg, dd_x, res);
    VML_VC_ST_STEP(svst1_f16, pg, ds, step, res);
}

void SqrtDstVCBWDF16(const int64_t len, const __fp16 *src, const __fp16 *dd, __fp16 *ds)
{
    int64_t i = 0;
    int64_t loopSqrtDsth = svcnth();

    for (; i <= len - loopSqrtDsth * VML_VC_UNFOLD_N; i += loopSqrtDsth * VML_VC_UNFOLD_N) {
        SqrtMainKernelBwdF16(src + i, dd + i, ds + i, loopSqrtDsth);
    }
    for (; i < len; i += loopSqrtDsth) {
        svbool_t pg = svwhilelt_b16_s64(i, len);
        svfloat16_t srcX = svld1_f16(pg, src + i);
        svfloat16_t ddX = svld1_f16(pg, dd + i);

        svfloat16_t res = svmul_f16_x(pg, srcX, VML_VC_F16_TWO);
        res = svdiv_f16_x(pg, ddX, res);

        svst1_f16(pg, ds + i, res);
    }
}