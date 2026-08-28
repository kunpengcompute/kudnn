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

#include "blas_wrapper.h"
#include "unigemm.h"
#include <iostream>

namespace common {
void BlasGemm(
    enum CBLAS_TRANSPOSE transa, enum CBLAS_TRANSPOSE transb,
    int64_t m, int64_t n, int64_t k,
    const float alpha,
    const float *a, int64_t lda,
    const float *b, int64_t ldb,
    const float beta,
    float *c, int64_t ldc)
{
    cblas_sgemm(
        CblasRowMajor,
        transa, transb,
        m, n, k,
        alpha,
        a, lda,
        b, ldb,
        beta,
        c, ldc);
}

void BlasGemm(
    enum CBLAS_TRANSPOSE transa, enum CBLAS_TRANSPOSE transb,
    int64_t m, int64_t n, int64_t k,
    const float alpha,
    const float16_t *a, int64_t lda,
    const float16_t *b, int64_t ldb,
    const float beta,
    float *c, int64_t ldc)
{
    cblas_shgemm(
        CblasRowMajor,
        transa, transb,
        m, n, k,
        alpha,
        a, lda,
        b, ldb,
        beta,
        c, ldc);
}

void BlasGemm(
    enum CBLAS_TRANSPOSE transa, enum CBLAS_TRANSPOSE transb,
    int64_t m, int64_t n, int64_t k,
    const float alpha,
    const bfloat16_t *a, int64_t lda,
    const bfloat16_t *b, int64_t ldb,
    const float beta,
    float *c, int64_t ldc)
{
    cblas_sbgemm(
        CblasRowMajor,
        transa, transb,
        m, n, k,
        alpha,
        a, lda,
        b, ldb,
        beta,
        c, ldc);
}

void BlasGemm(
    enum CBLAS_TRANSPOSE transa, enum CBLAS_TRANSPOSE transb,
    int64_t m, int64_t n, int64_t k,
    const float16_t alpha,
    const float16_t *a, int64_t lda,
    const float16_t *b, int64_t ldb,
    const float16_t beta,
    float16_t *c, int64_t ldc)
{
    cblas_hgemm(
        CblasRowMajor,
        transa, transb,
        m, n, k,
        alpha,
        a, lda,
        b, ldb,
        beta,
        c, ldc);
}

void BlasGemm(
    enum CBLAS_TRANSPOSE transa, enum CBLAS_TRANSPOSE transb,
    int64_t m, int64_t n, int64_t k,
    const bfloat16_t alpha,
    const bfloat16_t *a, int64_t lda,
    const bfloat16_t *b, int64_t ldb,
    const bfloat16_t beta,
    bfloat16_t *c, int64_t ldc)
{
    cblas_bgemm(
        CblasRowMajor,
        transa, transb,
        m, n, k,
        alpha,
        a, lda,
        b, ldb,
        beta,
        c, ldc);
}

void UniGemm(bool transA, bool transB,
    int64_t m, int64_t n, int64_t k,
    const float alpha,
    const float *a, int64_t lda,
    const float *b, int64_t ldb,
    const float beta,
    float *c, int64_t ldc, int thread_count)
{
    unigemm_fp32(true, transA, transB,
        m, n, k,
        alpha,
        a, lda, b, ldb,
        beta, c, ldc, thread_count);
}

void UniGemm(bool transA, bool transB,
    int64_t m, int64_t n, int64_t k,
    const float alpha,
    const float16_t *a, int64_t lda,
    const float16_t *b, int64_t ldb,
    const float beta,
    float *c, int64_t ldc, int thread_count)
{
    unigemm_fp16(true, transA, transB,
        m, n, k,
        alpha,
        a, lda, b, ldb,
        beta, c, ldc, thread_count);
}
}