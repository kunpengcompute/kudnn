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

#include "fused_ops.h"
#include "fast_exp.h"

void VecAddMul(int64_t numel, float *in1, float p, float *in2, float *out)
{
    int64_t vl = svcntw();
    for (int64_t i = 0; i < numel; i += vl) {
        svbool_t pg = svwhilelt_b32(i, numel);
        svfloat32_t values = svld1_f32(pg, in1 + i);
        svfloat32_t add = svld1_f32(pg, in2 + i);
        svfloat32_t res1 = svadd_x(pg, values, add);
        svfloat32_t res2 = svmul_x(pg, res1, svdup_f32(p));
        svst1_f32(pg, out + i, res2);
    }
}

void Layernorm(float* input, float* weights, float *bias, float eps, int64_t normShape, float *output)
{
    const int64_t vl = svcntw();
    svfloat32_t meanSum = svdup_f32(0.0);
    svfloat32_t varSum = svdup_f32(0.0);

    for (int64_t i = 0; i < normShape; i += vl) {
        svbool_t pg = svwhilelt_b32(i, normShape);
        svfloat32_t values = svld1_f32(pg, input + i);
        meanSum = svadd_m(pg, meanSum, values);
    }

    float mean = svaddv(svptrue_b32(), meanSum);
    mean = mean / normShape;

    for (int64_t i = 0; i < normShape; i += vl) {
        svbool_t pg = svwhilelt_b32(i, normShape);
        svfloat32_t values = svld1_f32(pg, input + i);
        values = svsub_x(pg, values, mean);
        varSum = svmla_m(pg, varSum, values, values);
    }

    float variance = svaddv(svptrue_b32(), varSum);
    variance = variance / normShape;

    variance = 1 / sqrtf(variance + eps);
    svfloat32_t part1 = svdup_f32(-variance * mean);

    for (int64_t i = 0; i < normShape; i += vl) {
        svbool_t pg = svwhilelt_b32(i, normShape);
        svfloat32_t values = svld1_f32(pg, input + i);
        svfloat32_t weightsValue = svld1_f32(pg, weights + i);
        svfloat32_t biasValue = svld1_f32(pg, bias + i);

        values = svmla_x(pg, part1, values, variance);
        values = svmad_x(pg, values, weightsValue, biasValue);
        svst1_f32(pg, output + i, values);
    }
}

void SigMoid(float *in, int64_t numel, float *out)
{
    const int64_t vl = svcntw();
    for (int64_t i = 0; i < numel; i += vl) {
        svbool_t pg = svwhilelt_b32(i, numel);
        svfloat32_t values = svld1_f32(pg, &in[i]);
        values = svneg_x(pg, values);
        svfloat32_t expNeg = FastExp(pg, values);
        svfloat32_t res = svdiv_x(pg, svdup_f32(1), svadd_x(pg, expNeg, svdup_f32(1)));
        svst1_f32(pg, &out[i], res);
    }
}

void SigMoidMul(int64_t rows, int64_t cols, int64_t ldi, int64_t ldo, float *in, float *in2, float *out)
{
    const int64_t vl = svcntw();
    for (int64_t i = 0; i < rows; i++) {
        for (int64_t j = 0; j < cols; j += vl) {
            svbool_t pg = svwhilelt_b32(j, cols);
            svfloat32_t values1 = svld1_f32(pg, &in[j + i * cols]);
            svfloat32_t values2 = svld1_f32(pg, &in2[j + i * cols]);
            values1 = svneg_x(pg, values1);
            svfloat32_t expNeg = FastExp(pg, values1);
            svfloat32_t sigmoid = svdiv_x(pg, svdup_f32(1), svadd_x(pg, expNeg, svdup_f32(1)));
            svfloat32_t res = svmul_x(pg, values2, sigmoid);
            svst1_f32(pg, &out[i * cols + j], res);
        }
    }
}

void SoftMax(int64_t s1, int64_t s2, int64_t s3, float *in, float *out)
{
    const int64_t vl = svcntw();
    for (int64_t k = 0; k < s2; k++) {
        svfloat32_t vecMax = svdup_f32(-1e9);
        svfloat32_t reduce = svdup_f32(0.f);
        // find max values
        for (int64_t i = 0; i < s1; i++) {
            for (int64_t j = 0; j < s3; j += vl) {
                svbool_t pg = svwhilelt_b32(j, s3);
                svfloat32_t values = svld1_f32(pg, &in[j + k * s3]);
                vecMax = svmax_m(pg, vecMax, values);
            }
        }
        float32_t scaMax = svmaxv(svptrue_b32(), vecMax);
        // cal softmax
        float buf[s3];
        for (int64_t i = 0; i < s1; i++) {
            for (int64_t j = 0; j < s3; j += vl) {
                svbool_t pg = svwhilelt_b32(j, s3);
                svfloat32_t values = svld1_f32(pg, &in[j + k * s3]);
                values = FastExp(pg, svsub_x(pg, values, scaMax));
                reduce = svadd_m(pg, reduce, values);
                svst1_f32(pg, (buf + j), values);
            }
            float sumInv = 1 / svaddv(svptrue_b32(), reduce);
            for (int64_t j = 0; j < s3; j += vl) {
                svbool_t pg = svwhilelt_b32(j, s3);
                svfloat32_t values = svld1_f32(pg, (buf + j));
                values = svmul_x(pg, values, svdup_f32(sumInv));
                svst1_f32(pg, (out + k * s3 + j), values);
            }
        }
    }
}

void FillData(int64_t size, float *dst, float data)
{
    const int64_t vl = svcntw();
    svfloat32_t value = svdup_f32(data);
    for (int64_t i = 0; i < size; i += vl) {
        svbool_t pg = svwhilelt_b32(i, size);
        svst1_f32(pg, dst + i, value);
    }
}

void FillData(int64_t size, float16_t *dst, float16_t data)
{
    const int64_t vl = svcnth();
    svfloat16_t value = svdup_f16(data);
    for (int64_t i = 0; i < size; i += vl) {
        svbool_t pg = svwhilelt_b16(i, size);
        svst1_f16(pg, dst + i, value);
    }
}

float ReducedMax(int64_t size, float *data)
{
    const int64_t vl = svcntw();
    svfloat32_t vecMax = svdup_f32(-1e9);
    for (int64_t i = 0; i < size; i += vl) {
        svbool_t pg = svwhilelt_b32(i, size);
        svfloat32_t values = svld1(pg, data + i);
        vecMax = svmax_m(pg, vecMax, values);
    }
    return svmaxv(svptrue_b32(), vecMax);
}

void VecLinearFusionKernel(int64_t size, float *in1, float p, float *bias, float *out)
{
    int64_t vl = svcntw();
    for (int64_t i = 0; i < size; i += vl) {
        svbool_t pg = svwhilelt_b32(i, size);
        svfloat32_t values = svld1(pg, in1 + i);
        svfloat32_t res = svmul_x(pg, values, svdup_f32(p));
        if (bias != nullptr) {
            svfloat32_t add = svld1(pg, bias + i);
            res = svadd_x(pg, res, add);
        }
        svst1_f32(pg, out + i, res);
    }
}

void VecLinearFusionKernel(int64_t size, float *in1, float p, float *bias, float16_t *out)
{
    const int64_t vlf = svcntw();
    const int64_t vlh = svcnth();
    for (int64_t i = 0; i < size; i += vlh) {
        svbool_t pgFloat0 = svwhilelt_b32(i, size);
        svbool_t pgFloat1 = svwhilelt_b32(i + vlf, size);
        svbool_t pgHalf = svwhilelt_b16(i, size);
        svfloat32_t inputFloat0 = svld1_f32(pgFloat0, in1 + i);
        svfloat32_t inputFloat1 = svld1_f32(pgFloat1, in1 + i + vlf);
        svfloat32_t res0 = svmul_n_f32_x(pgFloat0, inputFloat0, p);
        svfloat32_t res1 = svmul_n_f32_x(pgFloat1, inputFloat1, p);
        if (bias != nullptr) {
            svfloat32_t add0 = svld1_f32(pgFloat0, bias + i);
            svfloat32_t add1 = svld1_f32(pgFloat1, bias + i + vlf);
            res0 = svadd_f32_x(pgFloat0, inputFloat0, add0);
            res1 = svadd_f32_x(pgFloat1, inputFloat1, add1);
        }
        svfloat16_t resHalf0 = svcvt_f16_f32_x(svptrue_b32(), res0);
        svfloat16_t resHalf1 = svcvt_f16_f32_x(svptrue_b32(), res1);
        svfloat16_t res = svuzp1_f16(resHalf0, resHalf1);
        svst1_f16(pgHalf, out + i, res);
    }
}