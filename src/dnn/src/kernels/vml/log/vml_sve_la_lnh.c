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

#define VectorComputation_VHLN_INF_BIT svdup_n_u16(0x7c00U)
#define VectorComputation_VHLN_SUBNORMAL_BIT svdup_n_u16(F16_SUBNORMAL_BIT)
#define VectorComputation_VHLN_OFFSET svdup_n_u16(0x3940U) // 0x1.5p-1
#define FP16_MANTISSA_BIT_LEN 10
#define FP16_MANTISSA_MASK (0x3ffU)
#define VectorComputation_VHLN_LN2 svdup_n_f16(0x1.62cp-1f)
#define VectorComputation_VHLN_LN2LOW svdup_n_f16(0x1.218p-12f)

#define VectorComputation_VHLN_C2 svdup_n_f16(-0x1.ff8p-2f)
#define VectorComputation_VHLN_C3 svdup_n_f16(0x1.67p-2f)
#define VectorComputation_VHLN_C4 svdup_n_f16(-0x1.1dp-2f)

static inline __fp16 Logh(__fp16 x)
{
    return (__fp16)logf((float)x);
}

/*
 * p=fpminimax(log(1+x)/x-1, [|1,2,3|], [|10...|], [0x1.5p-1 - 1, 0x1.5p0 - 1], relative);
 * supnorm((p+1)*x,log(x+1),[0x1.5p-1 - 1, 0x1.5p0 - 1],relative,0x1p-12);
 * ulperror: [0x1.8519052ap-11;0x1.8530942dd00bp-11]
 */
__attribute__((always_inline)) static inline svfloat16_t VhLnLaSvePoly(svbool_t pg, svfloat16_t r, svfloat16_t n)
{
    svfloat16_t r2 = svmul_f16_x(pg, r, r);
    svfloat16_t poly = svmla_f16_x(pg, VectorComputation_VHLN_C3, r, VectorComputation_VHLN_C4);
    poly = svmla_f16_x(pg, VectorComputation_VHLN_C2, r, poly);
    poly = svmla_f16_x(pg, r, r2, poly);

    svfloat16_t res = svmla_f16_x(pg, poly, n, VectorComputation_VHLN_LN2LOW);
    res = svmla_f16_x(pg, res, n, VectorComputation_VHLN_LN2);
    return res;
}

static svfloat16_t SvLnF16X(svfloat16_t x, svbool_t pg)
{
    svuint16_t ux = svreinterpret_u16_f16(x);
    svbool_t pgSp = svcmpgt_u16(pg, svsub_u16_x(pg, ux, VectorComputation_VHLN_SUBNORMAL_BIT),
                                svsub_u16_x(pg, VML_VectorComputation_F16_INF, VectorComputation_VHLN_SUBNORMAL_BIT));
    svbool_t pgRetNan = svcmplt_n_f16(pg, x, FLOAT_ZERO);
    svbool_t pgRetInf = svcmpeq_n_f16(pg, x, INFINITY);
    svbool_t pgRetNegInf = svcmpeq_n_f16(pg, x, FLOAT_ZERO);
    pgSp = svbic_b_z(pg, pgSp, pgRetNan);
    pgSp = svbic_b_z(pg, pgSp, pgRetNegInf);

    // x = m * 2^n, 0x1.5p-1 <= m < 0x1.5p0
    svuint16_t uxSubLnfOffset = svsub_u16_x(pg, ux, VectorComputation_VHLN_OFFSET);
    svint16_t sn = svasr_n_s16_x(pg, svreinterpret_s16_u16(uxSubLnfOffset), FP16_MANTISSA_BIT_LEN);
    svfloat16_t n = svcvt_f16_x(pg, sn);
    svuint16_t uxSubLnfOffsetMant = svand_n_u16_x(pg, uxSubLnfOffset, FP16_MANTISSA_MASK);

    // r = m - 1
    svuint16_t uxAbsScale = svadd_u16_x(pg, uxSubLnfOffsetMant, VectorComputation_VHLN_OFFSET);
    svfloat16_t r = svsub_f16_x(pg, svreinterpret_f16_u16(uxAbsScale), VML_VectorComputation_F16_ONE);
    svfloat16_t res = VhLnLaSvePoly(pg, r, n);

    res = svsel_f16(pgRetNan, svreinterpret_f16_u16(svdup_n_u16(F16_NAN_BIT)), res);
    res = svsel_f16(pgRetInf, svreinterpret_f16_u16(svdup_n_u16(F16_INF_BIT)), res);
    res = svsel_f16(pgRetNegInf, svreinterpret_f16_u16(svdup_n_u16(F16_NEG_INF_BIT)), res);

    // only f16 has special handle because subnorm x in bf16/f32 is ignored by fpcs
    if (unlikely(svptest_any(pg, pgSp))) {
        res = VhFunc1f1Special(pgSp, x, res, Logh);
    }
    return res;
}

static void KplVmlVhLnLaSveMainN(const __fp16 *src, __fp16 *dst, int step)
{
    svbool_t pg = svptrue_b16();
    svbool_t pgSp;
    VML_VectorComputation_DECLARE(svfloat16_t, x);
    VML_VectorComputation_LD_STEP(x, svld1_f16, pg, src, step);

    VML_VectorComputation_DECLARE(svuint16_t, ux);
    VML_VectorComputation_DECLARE(svuint16_t, uxTmp);
    VML_VectorComputation_DECLARE(svbool_t, pgSp);
    VML_VectorComputation_DECLARE(svbool_t, pgRetNan);
    VML_VectorComputation_DECLARE(svbool_t, pgRetNegInf);
    VML_VectorComputation_DECLARE(svbool_t, pgRetInf);
    VML_VectorComputation_FUNC_NOPG_1PARA(ux, svreinterpret_u16_f16, x);
    VML_VectorComputation_FUNC_PG_N2_2PARA(uxTmp, svsub_n_u16_x, pg, ux, F16_SUBNORMAL_BIT);
    VML_VectorComputation_FUNC_PG_N2_2PARA(pgSp, svcmpgt_n_u16, pg, uxTmp, F16_INF_BIT - F16_SUBNORMAL_BIT);
    VML_VectorComputation_FUNC_PG_N2_2PARA(pgRetNan, svcmplt_n_f16, pg, x, FLOAT_ZERO);
    VML_VectorComputation_FUNC_PG_N2_2PARA(pgRetNegInf, svcmpeq_n_f16, pg, x, FLOAT_ZERO);
    VML_VectorComputation_FUNC_PG_N2_2PARA(pgRetInf, svcmpeq_n_f16, pg, x, INFINITY);
    VML_VectorComputation_FUNC_PG_2PARA(pgSp, svbic_b_z, pg, pgSp, pgRetNan);
    VML_VectorComputation_FUNC_PG_2PARA(pgSp, svbic_b_z, pg, pgSp, pgRetNegInf);
    VML_VectorComputation_REDUCE(pgSp, svorr_b_z, pg, pgSp);

    VML_VectorComputation_DECLARE(svuint16_t, uxSubLnfOffset);
    VML_VectorComputation_DECLARE(svint16_t, sxSubLnfOffset);
    VML_VectorComputation_DECLARE(svint16_t, sn);
    VML_VectorComputation_DECLARE(svfloat16_t, n);
    VML_VectorComputation_FUNC_PG_N2_2PARA(uxSubLnfOffset, svsub_u16_x, pg, ux, VectorComputation_VHLN_OFFSET);
    VML_VectorComputation_FUNC_NOPG_1PARA(sxSubLnfOffset, svreinterpret_s16_u16, uxSubLnfOffset);
    VML_VectorComputation_FUNC_PG_N2_2PARA(sn, svasr_n_s16_x, pg, sxSubLnfOffset, FP16_MANTISSA_BIT_LEN);
    VML_VectorComputation_FUNC_PG_1PARA(n, svcvt_f16_x, pg, sn);

    VML_VectorComputation_DECLARE(svuint16_t, uxSubLnfOffsetMant);
    VML_VectorComputation_DECLARE(svuint16_t, uxAbsScale);
    VML_VectorComputation_DECLARE(svfloat16_t, xAbsScale);
    VML_VectorComputation_DECLARE(svfloat16_t, r);
    VML_VectorComputation_FUNC_PG_N2_2PARA(uxSubLnfOffsetMant, svand_n_u16_x, pg, uxSubLnfOffset, FP16_MANTISSA_MASK);
    VML_VectorComputation_FUNC_PG_N2_2PARA(uxAbsScale, svadd_u16_x, pg, uxSubLnfOffsetMant,
                                           VectorComputation_VHLN_OFFSET);
    VML_VectorComputation_FUNC_NOPG_1PARA(xAbsScale, svreinterpret_f16_u16, uxAbsScale);
    VML_VectorComputation_FUNC_PG_N2_2PARA(r, svsub_f16_x, pg, xAbsScale, VML_VectorComputation_F16_ONE);

    VML_VectorComputation_DECLARE(svfloat16_t, r2);
    VML_VectorComputation_DECLARE(svfloat16_t, poly);
    VML_VectorComputation_DECLARE(svfloat16_t, res);
    VML_VectorComputation_FUNC_PG_2PARA(r2, svmul_f16_x, pg, r, r);
    VML_VectorComputation_FUNC_PG_N13_3PARA(poly, svmla_f16_x, pg, VectorComputation_VHLN_C3, r,
                                            VectorComputation_VHLN_C4);
    VML_VectorComputation_FUNC_PG_N1_3PARA(poly, svmla_f16_x, pg, VectorComputation_VHLN_C2, r, poly);
    VML_VectorComputation_FUNC_PG_3PARA(poly, svmla_f16_x, pg, r, r2, poly);
    VML_VectorComputation_FUNC_PG_N3_3PARA(res, svmla_f16_x, pg, poly, n, VectorComputation_VHLN_LN2LOW);
    VML_VectorComputation_FUNC_PG_N3_3PARA(res, svmla_f16_x, pg, res, n, VectorComputation_VHLN_LN2);

    VML_VectorComputation_FUNC_NOPG_N2_3PARA(res, svsel_f16, pgRetNan, svreinterpret_f16_u16(svdup_n_u16(F16_NAN_BIT)),
                                             res);
    VML_VectorComputation_FUNC_NOPG_N2_3PARA(res, svsel_f16, pgRetNegInf,
                                             svreinterpret_f16_u16(svdup_n_u16(F16_NEG_INF_BIT)), res);
    VML_VectorComputation_FUNC_NOPG_N2_3PARA(res, svsel_f16, pgRetInf, svreinterpret_f16_u16(svdup_n_u16(F16_INF_BIT)),
                                             res);
    if (unlikely(svptest_any(pg, pgSp))) {
        for (int j = 0; j < step * VML_VectorComputation_UNFOLD_N; j++) {
            dst[j] = Logh(src[j]);
        }
    } else {
        VML_VectorComputation_ST_STEP(svst1_f16, pg, dst, step, res);
    }
}

void LnLaSveFWDF16(const int64_t len, const __fp16 *src, __fp16 *dst)
{
    int64_t i;
    int loopLnh = svcnth();
    for (i = 0; i <= len - loopLnh * VML_VectorComputation_UNFOLD_N; i += loopLnh * VML_VectorComputation_UNFOLD_N) {
        KplVmlVhLnLaSveMainN(src + i, dst + i, loopLnh);
    }
    for (; i < len; i += loopLnh) {
        svbool_t pg = svwhilelt_b16_s64(i, len);
        svfloat16_t x = svld1_f16(pg, src + i);
        svfloat16_t res = SvLnF16X(x, pg);
        svst1_f16(pg, dst + i, res);
    }
}

#undef VML_VectorComputation_UNFOLD_N
#define VML_VectorComputation_UNFOLD_N 5

static void LnMainKernelBwdF16(const __fp16 *src, const __fp16 *dd, __fp16 *ds, int step)
{
    svbool_t pg = svptrue_b16();
    VML_VectorComputation_DECLARE(svfloat16_t, x);
    VML_VectorComputation_DECLARE(svfloat16_t, ddX);

    VML_VectorComputation_LD_STEP(x, svld1_f16, pg, src, step);
    VML_VectorComputation_LD_STEP(ddX, svld1_f16, pg, dd, step);

    VML_VectorComputation_FUNC_PG_2PARA(ddX, svdiv_f16_x, pg, ddX, x);
    VML_VectorComputation_ST_STEP(svst1_f16, pg, ds, step, ddX);
}

void LnLaSveBWDF16(const int64_t len, const __fp16 *src, const __fp16 *dd, __fp16 *ds)
{
    int loopLnh = svcnth();
    int64_t i = 0;
    for (; i <= len - loopLnh * VML_VectorComputation_UNFOLD_N; i += loopLnh * VML_VectorComputation_UNFOLD_N) {
        LnMainKernelBwdF16(src + i, dd + i, ds + i, loopLnh);
    }
    for (; i < len; i += loopLnh) {
        svbool_t pg = svwhilelt_b16_s64(i, len);
        svfloat16_t x = svld1_f16(pg, src + i);
        svfloat16_t ddX = svld1_f16(pg, dd + i);

        svfloat16_t res = svdiv_f16_x(pg, ddX, x);
        svst1_f16(pg, ds + i, res);
    }
}
