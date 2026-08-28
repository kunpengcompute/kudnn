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

#include <cstdlib>
#include <iostream>
#include <cmath>
#include <ctime>
#include <random>

#include "gtest/gtest.h"
#include "kudnn.hpp"
#include "kudnn_tensor_data.hpp"

// forward
static bool ForwardFunc1(int max_position = 10000, int dim = 128,
    int batchSize = 2, int seqLen = 1, int qHeads = 2, int kHeads = 2)
{
    KuDNN::RopeLayerFWD layer(max_position, dim);

    int qStride = qHeads * dim, kStride = kHeads * dim;
    int qSize = batchSize * seqLen * qHeads * dim;
    int kSize = batchSize * seqLen * kHeads * dim;
    __fp16 *query = (__fp16 *)malloc(qSize * sizeof(__fp16));
    __fp16 *key = (__fp16 *)malloc(kSize * sizeof(__fp16));
    int *positionIds = (int *)malloc(seqLen * sizeof(int));
    if (query == nullptr || key == nullptr || positionIds == nullptr) {
        free(query);
        free(key);
        free(positionIds);
        std::cerr << "Memory allocation failed" << std::endl;
        return false;
    }
    // generate random test data
    std::uniform_real_distribution<float> u(-1, 1);
    std::default_random_engine e(time(NULL));
#pragma omp parallel for
    for (int i = 0; i < qSize; ++i) {
        *(query + i) = (__fp16)u(e);
    }
#pragma omp parallel for
    for (int i = 0; i < kSize; ++i) {
        *(key + i) = (__fp16)u(e);
    }
#pragma omp parallel for
    for (int i = 0; i < seqLen; ++i) {
        *(positionIds + i) = i;
    }
    layer.Run(query, key, qStride, kStride, batchSize, seqLen, qHeads, kHeads, positionIds, KuDNN::RopeKind::DeepSeek);
    layer.Run(query, key, qStride, kStride, batchSize, seqLen, qHeads, kHeads, positionIds, KuDNN::RopeKind::OpenSora);
    free(query);
    free(key);
    free(positionIds);
    return true;
}

static void QueryKeyNull(int max_position = 10000, int dim = 4,
    int batchSize = 1, int seqLen = 1, int qHeads = 1, int kHeads = 1)
{
    KuDNN::RopeLayerFWD layer(max_position, dim);

    int qStride = qHeads * dim, kStride = kHeads * dim;
    __fp16 *query = nullptr;
    __fp16 *key = nullptr;
    int *positionIds = (int *)malloc(seqLen * sizeof(int));
    if (positionIds == nullptr) {
        free(positionIds);
        std::cerr << "Memory allocation failed" << std::endl;
        return;
    }
#pragma omp parallel for
    for (int i = 0; i < seqLen; ++i) {
        *(positionIds + i) = i;
    }

    try {
        layer.Run(query, key, qStride, kStride, batchSize, seqLen, qHeads, kHeads, positionIds);
    } catch(...) {
        free(positionIds);
        throw;
    }

    free(positionIds);
    return;
}

static void NegativeArgs(int max_position = 10000, int dim = 4,
    int batchSize = -1, int seqLen = 1, int qHeads = 1, int kHeads = 1)
{
    KuDNN::RopeLayerFWD layer(max_position, dim);

    int qStride = qHeads * dim, kStride = kHeads * dim;
    __fp16 *query = (__fp16 *)malloc(4);
    __fp16 *key = (__fp16 *)malloc(4);
    int *positionIds = (int *)malloc(seqLen * sizeof(int));
    if (query == nullptr || key == nullptr || positionIds == nullptr) {
        free(query);
        free(key);
        free(positionIds);
        std::cerr << "Memory allocation failed" << std::endl;
        return;
    }
#pragma omp parallel for
    for (int i = 0; i < seqLen; ++i) {
        *(positionIds + i) = i;
    }

    try {
        layer.Run(query, key, qStride, kStride, batchSize, seqLen, qHeads, kHeads, positionIds);
    } catch(...) {
        free(query);
        free(key);
        free(positionIds);
        throw;
    }

    free(query);
    free(key);
    free(positionIds);
    return;
}

static void NegativePids(int max_position = 10000, int dim = 4,
    int batchSize = 1, int seqLen = 1, int qHeads = 1, int kHeads = 1)
{
    KuDNN::RopeLayerFWD layer(max_position, dim);

    int qStride = qHeads * dim, kStride = kHeads * dim;
    int qSize = batchSize * seqLen * qHeads * dim;
    int kSize = batchSize * seqLen * kHeads * dim;
    __fp16 *query = (__fp16 *)malloc(qSize * sizeof(__fp16));
    __fp16 *key = (__fp16 *)malloc(kSize * sizeof(__fp16));
    int *positionIds = (int *)malloc(seqLen * sizeof(int));
    if (query == nullptr || key == nullptr || positionIds == nullptr) {
        free(query);
        free(key);
        free(positionIds);
        std::cerr << "Memory allocation failed" << std::endl;
        return;
    }
#pragma omp parallel for
    for (int i = 0; i < seqLen; ++i) {
        *(positionIds + i) = -1;
    }

    try {
        layer.Run(query, key, qStride, kStride, batchSize, seqLen, qHeads, kHeads, positionIds);
    } catch(...) {
        free(query);
        free(key);
        free(positionIds);
        throw;
    }

    free(query);
    free(key);
    free(positionIds);
    return;
}

static void ForwardConstructorFailed()
{
    int max_position = 10000, dim = 99;
    KuDNN::RopeLayerFWD layer(max_position, dim);
}

static void ForwardCopyConstructor1()
{
    int max_position = 10000, dim = 128;
    KuDNN::RopeLayerFWD layer1(max_position, dim);
    KuDNN::RopeLayerFWD layer2(layer1);
}

static void ForwardCopyConstructor2()
{
    int max_position = 10000, dim = 128;
    KuDNN::RopeLayerFWD layer1(max_position, dim);
    KuDNN::RopeLayerFWD layer2(max_position, dim);
    layer1 = layer2;
}

static void ForwardMoveConstructor1()
{
    int max_position = 10000, dim = 128;
    KuDNN::RopeLayerFWD layer1(max_position, dim);
    KuDNN::RopeLayerFWD layer2(std::move(layer1));
}

static void ForwardMoveConstructor2()
{
    int max_position = 10000, dim = 128;
    KuDNN::RopeLayerFWD layer1(max_position, dim);
    KuDNN::RopeLayerFWD layer2(max_position, dim);
    layer2 = std::move(layer1);
}

static void NegativeDims()
{
    int max_position = 10000, dim = -1;
    KuDNN::RopeLayerFWD layer1(max_position, dim);
}

static void NegativeMaxPos()
{
    int max_position = -1, dim = 128;
    KuDNN::RopeLayerFWD layer1(max_position, dim);
}

TEST(RopeGTests, NegativeDims)
{
    EXPECT_ANY_THROW(NegativeDims());
}

TEST(RopeGTests, NegativeMaxPos)
{
    EXPECT_ANY_THROW(NegativeMaxPos());
}

TEST(RopeGTests, ForwardFunc1)
{
    int max_position = 10000, dim = 128;
    int batchSize = 2, seqLen = 128, qHeads = 28, kHeads = 28;
    GTEST_EXPECT_TRUE(ForwardFunc1(max_position, dim,
        batchSize, seqLen, qHeads, kHeads));
}

TEST(RopeGTests, QueryKeyNull)
{
    EXPECT_ANY_THROW(QueryKeyNull());
}

TEST(RopeGTests, NegativeArgs)
{
    EXPECT_ANY_THROW(NegativeArgs());
}

TEST(RopeGTests, NegativePids)
{
    EXPECT_ANY_THROW(NegativePids());
}

TEST(RopeGTests, ForwardConstructorFailed)
{
    EXPECT_ANY_THROW(ForwardConstructorFailed());
}

TEST(RopeGTests, FWDCopyConstructor1)
{
    EXPECT_NO_THROW(ForwardCopyConstructor1());
}

TEST(RopeGTests, FWDCopyConstructor2)
{
    EXPECT_NO_THROW(ForwardCopyConstructor2());
}

TEST(RopeGTests, FWDMoveConstructor1)
{
    EXPECT_NO_THROW(ForwardMoveConstructor1());
}

TEST(RopeGTests, FWDMoveConstructor2)
{
    EXPECT_NO_THROW(ForwardMoveConstructor2());
}
