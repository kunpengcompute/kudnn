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

#include "kernels/vml/common_macro.h"
#include "kernels/vml/vml_sve_la_sigmoid.h"

static void SwishLaVCKernelFWDF32(svfloat32_t va, const float *src, float *dst, int step)
{
    svbool_t pg = svptrue_b32();
    VML_VC_DECLARE(svfloat32_t, x);
    VML_VC_LD_STEP(x, svld1_f32, pg, src, step);
    VML_VC_DECLARE(svfloat32_t, ax);
    VML_VC_FUNC_PG_N2_2PARA(ax, svmul_f32_x, pg, x, va);
    VML_VC_DECLARE(svfloat32_t, t);
    VML_VC_FUNC_PG_1PARA(t, svneg_f32_x, pg, ax);
    VML_VC_DECLARE(svfloat32_t, res);
    VML_VC_FUNC_NOPG_N2_2PARA(res, SvSigmoidF32X, t, pg);
    VML_VC_FUNC_PG_2PARA(res, svmul_f32_x, pg, x, res);
    VML_VC_ST_STEP(svst1_f32, pg, dst, step, res);
}

void SwishLaVCFWDF32(const int64_t len, const float alpha, const float *src, float *dst)
{
    int64_t i = 0;
    int64_t loopClipSwishf = svcntw();
    svfloat32_t va = svdup_n_f32(alpha);
    for (; i <= len - loopClipSwishf * VML_VC_UNFOLD_N; i += loopClipSwishf * VML_VC_UNFOLD_N) {
        SwishLaVCKernelFWDF32(va, src + i, dst + i, loopClipSwishf);
    }
    for (; i < len; i += loopClipSwishf) {
        svbool_t pg = svwhilelt_b32_s64(i, len);
        svfloat32_t x = svld1_f32(pg, src + i);
        svfloat32_t ax = svmul_f32_x(pg, x, va);
        svfloat32_t res = SvSigmoidF32X(svneg_f32_x(pg, ax), pg);
        res = svmul_f32_x(pg, x, res);
        svst1_f32(pg, dst + i, res);
    }
}

#undef VML_VC_UNFOLD_N
#define VML_VC_UNFOLD_N 6

static void SwishMainKernelBwdF32(svfloat32_t va, const float *src, const float *dd, float *ds, int step)
{
    svbool_t pg = svptrue_b32();
    VML_VC_DECLARE(svfloat32_t, srcX);
    VML_VC_DECLARE(svfloat32_t, ddX);

    VML_VC_LD_STEP(srcX, svld1_f32, pg, src, step);

    VML_VC_LD_STEP(ddX, svld1_f32, pg, dd, step);

    VML_VC_DECLARE(svfloat32_t, aSrcX);
    VML_VC_FUNC_PG_N2_2PARA(aSrcX, svmul_f32_x, pg, srcX, va);
    VML_VC_DECLARE(svfloat32_t, sigmoidRes);
    VML_VC_DECLARE(svfloat32_t, aSrcXNeg);
    VML_VC_FUNC_PG_1PARA(aSrcXNeg, svneg_f32_x, pg, aSrcX);
    VML_VC_FUNC_NOPG_N2_2PARA(sigmoidRes, SvSigmoidF32X, aSrcXNeg, pg);

    VML_VC_DECLARE(svfloat32_t, res);
    VML_VC_FUNC_PG_3PARA(res, svmls_f32_x, pg, aSrcX, aSrcX, sigmoidRes);
    VML_VC_FUNC_PG_3PARA(res, svmla_f32_x, pg, sigmoidRes, sigmoidRes, res);
    VML_VC_FUNC_PG_2PARA(res, svmul_f32_x, pg, res, ddX);
    VML_VC_ST_STEP(svst1_f32, pg, ds, step, res);
}

void SwishLaVCBWDF32(const int64_t len, const float alpha, const float *src, const float *dd, float *ds)
{
    int64_t i = 0;
    int loopSwishf = svcntw();
    svfloat32_t va = svdup_n_f32(alpha);
    for (; i <= len - loopSwishf * VML_VC_UNFOLD_N; i += loopSwishf * VML_VC_UNFOLD_N) {
        SwishMainKernelBwdF32(va, src + i, dd + i, ds + i, loopSwishf);
    }
    for (; i < len; i += loopSwishf) {
        svbool_t pg = svwhilelt_b32_s64(i, len);
        svfloat32_t srcX = svld1_f32(pg, src + i);
        svfloat32_t ddX = svld1_f32(pg, dd + i);

        // R = alpha * s
        // Q = sigmoid(-alpha * s)
        svfloat32_t aSrcX = svmul_f32_x(pg, srcX, va);
        svfloat32_t sigmoidRes = SvSigmoidF32X(svneg_f32_x(pg, aSrcX), pg);

        // compute Q * (1 + R * (1 - Q))
        // T = R * (1 - Q) = R - R * Q
        // Q * (1 + T) = Q + Q * T
        svfloat32_t res = svmls_f32_x(pg, aSrcX, aSrcX, sigmoidRes);
        res = svmla_f32_x(pg, sigmoidRes, sigmoidRes, res);
        res = svmul_f32_x(pg, res, ddX);

        svst1_f32(pg, ds + i, res);
    }
}
