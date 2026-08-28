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

void MatCopy(int64_t rows, int64_t cols, int64_t ldi, int64_t ldo, float *in, float *out)
{
    const int64_t vl = svcntw();
    for (int64_t r = 0; r < rows; r += 1) {
        float *out1 = out + r * ldo;
        float *in1 = in + r * ldi;
        for (int64_t c = 0; c < cols; c += vl) {
            svbool_t pg = svwhilelt_b32(c, cols);
            svfloat32_t values = svld1_f32(pg, in1 + c);
            svst1_f32(pg, (out1 + c), values);
        }
    }
}

void MatAddVec(int64_t rows, int64_t cols, int64_t ld, float *in, float *out)
{
    const int64_t vl = svcntw();
    for (int64_t r = 0; r < rows; r++) {
        float *out1 = out + r * ld;
        for (int64_t c = 0; c < cols; c += vl) {
            svbool_t pg = svwhilelt_b32(c, cols);
            svfloat32_t add0 = svld1_f32(pg, in + c);
            svfloat32_t add1 = svld1_f32(pg, out1 + c);
            svfloat32_t res = svadd_x(pg, add0, add1);
            svst1_f32(pg, (out1 + c), res);
        }
    }
}

void MatAddMat(int64_t rows, int64_t cols, int64_t ldi0, int64_t ldi1,
    int64_t ldo, float *in0, float *in1, float *out, bool hasLastStride)
{
    const int64_t vl = svcntw();
    for (int64_t r = 0; r < rows; r++) {
        float *out1 = out + r * ldo;
        float *in01 = in0 + r * ldi0;
        float *in11 = in1 + r * ldi1;
        for (int64_t c = 0; c < cols; c += vl) {
            svbool_t pg = svwhilelt_b32(c, cols);
            svfloat32_t add0 = svld1_f32(pg, in01 + c);
            svfloat32_t add1 = hasLastStride ? svld1_f32(pg, in11 + c) : svdup_n_f32(*(in11));
            svfloat32_t res = svadd_f32_x(pg, add0, add1);
            svst1_f32(pg, (out1 + c), res);
        }
    }
}

void MatAddMat(int64_t rows, int64_t cols, int64_t ldi0, int64_t ldi1,
    int64_t ldo, float *in0, float16_t *in1, float *out, bool hasLastStride)
{
    const int64_t vlf = svcntw();
    const int64_t vlh = svcnth();
    for (int64_t r = 0; r < rows; r++) {
        float *out1 = out + r * ldo;
        float *in01 = in0 + r * ldi0;
        float16_t *in11 = in1 + r * ldi1;
        for (int64_t c = 0; c < cols; c += vlh) {
            svbool_t pgFloat0 = svwhilelt_b32(c, cols);
            svbool_t pgFloat1 = svwhilelt_b32(c + vlf, cols);
            svbool_t pgHalf = svwhilelt_b16(c, cols);
            svfloat32_t inputFloat0 = svld1_f32(pgFloat0, in01 + c);
            svfloat32_t inputFloat1 = svld1_f32(pgFloat1, in01 + c + vlf);
            svfloat16_t inputHalf = hasLastStride ? svld1_f16(pgHalf, in11 + c) : svdup_n_f16(*(in11));
            svfloat32_t cvtEven = svcvt_f32_f16_x(svptrue_b16(), inputHalf);
#ifdef __ARM_FEATURE_SVE2
            svfloat32_t cvtOdd = svcvtlt_f32_f16_x(svptrue_b16(), inputHalf);
#else
            svfloat16_t interval = svtrn2_f16(inputHalf, inputHalf);
            svfloat32_t cvtOdd = svcvt_f32_f16_x(svptrue_b16(), interval);
#endif
            svfloat32_t addFloat0 = svzip1_f32(cvtEven, cvtOdd);
            svfloat32_t addFloat1 = svzip2_f32(cvtEven, cvtOdd);
            svfloat32_t res0 = svadd_f32_x(pgFloat0, inputFloat0, addFloat0);
            svst1_f32(pgFloat0, (out1 + c), res0);
            svfloat32_t res1 = svadd_f32_x(pgFloat1, inputFloat1, addFloat1);
            svst1_f32(pgFloat1, (out1 + c + vlf), res1);
        }
    }
}

void MatAddMat(int64_t rows, int64_t cols, int64_t ldi0, int64_t ldi1,
    int64_t ldo, float *in0, bfloat16_t *in1, float *out, bool hasLastStride)
{
    const int64_t vlf = svcntw();
    const int64_t vlh = svcnth();
    for (int64_t r = 0; r < rows; r++) {
        float *out1 = out + r * ldo;
        float *in01 = in0 + r * ldi0;
        bfloat16_t *in11 = in1 + r * ldi1;
        for (int64_t c = 0; c < cols; c += vlh) {
            svbool_t pgFloat0 = svwhilelt_b32(c, cols);
            svbool_t pgFloat1 = svwhilelt_b32(c + vlf, cols);
            svbool_t pgHalf = svwhilelt_b16(c, cols);
            svfloat32_t inputFloat0 = svld1_f32(pgFloat0, in01 + c);
            svfloat32_t inputFloat1 = svld1_f32(pgFloat1, in01 + c + vlf);
            svbfloat16_t inputHalf = hasLastStride ? svld1_bf16(pgHalf, in11 + c) : svdup_n_bf16(*(in11));
            svfloat32_t inputEven = svuzp1_f32(inputFloat0, inputFloat1);
            svfloat32_t inputOdd = svuzp2_f32(inputFloat0, inputFloat1);
            svfloat32_t sumEven = svbfmlalb_n_f32(inputEven, inputHalf, 1.0);
            svfloat32_t sumOdd = svbfmlalt_n_f32(inputOdd, inputHalf, 1.0);
            svfloat32_t sum0 = svzip1_f32(sumEven, sumOdd);
            svfloat32_t sum1 = svzip2_f32(sumEven, sumOdd);
            svst1_f32(pgFloat0, (out1 + c), sum0);
            svst1_f32(pgFloat1, (out1 + c + vlf), sum1);
        }
    }
}

void MatTrans(int64_t rows, int64_t cols, int64_t ldi, int64_t ldo, float *in, float *out)
{
    for (int64_t i = 0; i < rows; i++) {
        for (int64_t j = 0; j < cols; j++) {
            out[i * ldo + j] = in[j * ldi + i];
        }
    }
}

void MatMulVec(int64_t rows, int64_t cols, int64_t ldi, int64_t ldo, float *in0, float *in1, float *out)
{
    const int64_t vl = svcntw();
    for (int64_t r = 0; r < rows; r++) {
        svfloat32_t mul0 = svdup_f32(in0[r]);
        for (int64_t c = 0; c < cols; c += vl) {
            svbool_t pg = svwhilelt_b32(c, cols);
            svfloat32_t mul1 = svld1_f32(pg, &in1[c + r * cols]);
            svfloat32_t res = svmul_m(pg, mul0, mul1);
            svst1_f32(pg, &out[r * cols + c], res);
        }
    }
}