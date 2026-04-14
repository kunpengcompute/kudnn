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

__attribute__((always_inline)) static inline svfloat16_t FmaF16CvtF32(svfloat16_t x,
    svfloat32_t va, svfloat32_t vb, svbool_t pgB16, svbool_t pgB32)
{
    svfloat32_t xfEven = svcvt_f32_f16_x(pgB16, x);
    svfloat32_t xfOdd = CvtOddF32F16(x, pgB16, pgB32);
    svfloat32_t resEven = svmad_f32_x(pgB32, va, xfEven, vb);
    svfloat32_t resOdd = svmad_f32_x(pgB32, va, xfOdd, vb);
    svfloat16_t yEven = svcvt_f16_f32_x(pgB32, resEven);
    svfloat16_t yOdd = CvtOddF16F32(resOdd, pgB32);
    svfloat16_t y = svadd_f16_x(pgB16, yEven, yOdd);
    return y;
}

void HardSigmoidLaSveFWDF16(const int64_t len, const float alpha, const float beta, const __fp16 *src, __fp16 *dst)
{
    int64_t i;
    int loopHardSigmoidh = svcnth();
    svfloat32_t alphaVec = svdup_n_f32(alpha);
    svfloat32_t betaVec = svdup_n_f32(beta);
    for (i = 0; i < len; i += loopHardSigmoidh) {
        svbool_t pgB16 = svwhilelt_b16_s64(i, len);
        svbool_t pgB32 = svwhilelt_b32_s64(i, len);

        svfloat16_t x = svld1_f16(pgB16, src + i);
        svfloat16_t y = FmaF16CvtF32(x, alphaVec, betaVec, pgB16, pgB32);

        svfloat16_t res = svminnm_f16_x(pgB16, y, VML_VectorComputation_F16_ONE);
        res = svmaxnm_f16_x(pgB16, res, VML_VectorComputation_F16_ZERO);
        svst1_f16(pgB16, dst + i, res);
    }
}

void HardSigmoidLaSveBWDF16(const int64_t len, const float alpha, const float beta,
    const __fp16 *src, const __fp16 *dd, __fp16 *ds)
{
    int64_t i;
    int loopHardSigmoidh = svcnth();
    svfloat32_t alphaVec = svdup_n_f32(alpha);
    svfloat32_t betaVec = svdup_n_f32(beta);
    for (i = 0; i < len; i += loopHardSigmoidh) {
        svbool_t pgB16 = svwhilelt_b16_s64(i, len);
        svbool_t pgB32 = svwhilelt_b32_s64(i, len);

        svfloat16_t srcX = svld1_f16(pgB16, src + i);
        svfloat16_t ddX = svld1_f16(pgB16, dd + i);

        svfloat32_t srcXfEven = svcvt_f32_f16_x(pgB16, srcX);
        svfloat32_t srcXfOdd = CvtOddF32F16(srcX, pgB16, pgB32);

        svfloat32_t ddXfEven = svcvt_f32_f16_x(pgB16, ddX);
        svfloat32_t ddXfOdd = CvtOddF32F16(ddX, pgB16, pgB32);

        svfloat32_t srcXEvenRes = svmad_f32_x(pgB32, srcXfEven, alphaVec, betaVec);
        svfloat32_t srcXOddRes = svmad_f32_x(pgB32, srcXfOdd, alphaVec, betaVec);

        svbool_t cmpEven = svcmplt_f32(pgB32, srcXEvenRes, VectorComputation_FLOAT_ONE);
        svbool_t cmpOdd = svcmplt_f32(pgB32, srcXOddRes, VectorComputation_FLOAT_ONE);
        cmpEven = svcmpgt_f32(cmpEven, srcXEvenRes, VectorComputation_FLOAT_ZERO);
        cmpOdd = svcmpgt_f32(cmpOdd, srcXOddRes, VectorComputation_FLOAT_ZERO);

        svfloat32_t ddXEvenRes = svmul_f32_z(cmpEven, ddXfEven, alphaVec);
        svfloat32_t ddXOddRes = svmul_f32_z(cmpOdd, ddXfOdd, alphaVec);

        svfloat16_t resEven = svcvt_f16_f32_x(pgB32, ddXEvenRes);
        svfloat16_t resOdd = CvtOddF16F32(ddXOddRes, pgB32);
        svfloat16_t res = svadd_f16_x(pgB16, resEven, resOdd);

        svst1_f16(pgB16, ds + i, res);
    }
}
