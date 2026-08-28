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

#define MAX_HYPER_PARAMETER 256

const int PRIMARY_M = 16;
const int PRIMARY_N = 64;

const int NO_LOAD_C = 0x0;
const int ONLY_LOAD_C = 0x1;
const int LOAD_C_WITH_BETA = 0x2;

const int PART_M = 512;           // m在LLC上的切分
const int PART_N = PRIMARY_N * 2; // 双Buffer
const int PART_K = 256;           // k在LLC级别的切分，一级part

enum {
    PA_FIRST,
    PB_FIRST,
    PA_ONLY,
    PB_ONLY,
    NO_PACK,
};

struct unigemm_hyper_parameter {
    /* Locate the specific shape */
    int m, n, k;
    bool transA, transB, rowMajor;

    /* hyper parameter */
    int m_thd;
    int n_thd;
    int k_thd;
    int kernelMode;
};

struct unigemm_hyper_parameter_set {
    int num; /* Only accept 256 shape */
    void *cBuf;
    int cBufLen;
    struct unigemm_hyper_parameter *params;
    struct unigemm_hyper_parameter localParams;
};

void unigemm_set_hyper_parameter(struct unigemm_hyper_parameter_set *hps);

void unigemm_cc(float *A, float *B, float *C, int m, int n, int k,
    int lda, int ldb, int ldc, bool acc, int thread_count);
void unigemm_cc(float16_t *A, float16_t *B, float *C, int m, int n, int k,
    int lda, int ldb, int ldc, bool acc, int thread_count);

void unigemm_fp32(bool rowMajor, bool transA, bool transB,
    const int M, const int N, const int K, const float alpha,
    const float *A, const int lda, const float *B, const int ldb,
    const float beta, float *C, const int ldc, int thread_count);
void unigemm_fp16(bool rowMajor, bool transA, bool transB,
    const int M, const int N, const int K, const float alpha,
    const float16_t *A, const int lda, const float16_t *B, const int ldb,
    const float beta, float *C, const int ldc, int thread_count);

void unigemm_attention_fp32(bool rowMajor, bool transA, bool transB, const int M,
    const int N, const int K, const float alpha, const float *A, const int lda,
    const float *B, const int ldb, const float beta, float *C, const int ldc,
    float *qkMaxData, float *qkSumData, const float *V, int ldv, float *E, int lde, int thread_count);
