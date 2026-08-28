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

#pragma once
#include <arm_sve.h>
#include <cmath>

void MatTrans(int64_t rows, int64_t cols, int64_t ldi, int64_t ldo, float *in, float *out);
void MatMulVec(int64_t rows, int64_t cols, int64_t ldi, int64_t ldo, float *in0, float *in1, float *out);
void MatCopy(int64_t rows, int64_t cols, int64_t ldi, int64_t ldo, float *in, float *out);
void Layernorm(float* input, float* weights, float *bias, float eps, int64_t norm_shape, float *output);
void MatAddVec(int64_t rows, int64_t cols, int64_t ld, float *in, float *out);
void MatAddMat(int64_t rows, int64_t cols, int64_t ldi0, int64_t ldi1,
    int64_t ldo, float *in0, float *in1, float *out, bool hasLastStride = true);
void MatAddMat(int64_t rows, int64_t cols, int64_t ldi0, int64_t ldi1,
    int64_t ldo, float *in0, float16_t *in1, float *out, bool hasLastStride = true);
void MatAddMat(int64_t rows, int64_t cols, int64_t ldi0, int64_t ldi1,
    int64_t ldo, float *in0, bfloat16_t *in1, float *out, bool hasLastStride = true);
void SigMoid(float *in, int64_t numel, float *out);
void SigMoidMul(int64_t rows, int64_t cols, int64_t ldi, int64_t ldo, float *in, float *in2, float *out);
void VecAddMul(int64_t numel, float *in1, float p, float *in2, float *out);
void SoftMax(int64_t s1, int64_t s2, int64_t s3, float *in, float *out);
void FillData(int64_t size, float *dst, float data);
void FillData(int64_t size, float16_t *dst, float16_t data);
float ReducedMax(int64_t size, float *data);
void VecLinearFusionKernel(int64_t size, float *in1, float p, float *bias, float *out);
void VecLinearFusionKernel(int64_t size, float *in1, float p, float *bias, float16_t *out);