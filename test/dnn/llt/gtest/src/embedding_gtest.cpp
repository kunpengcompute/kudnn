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

#include "gtest/gtest.h"
#include "kudnn.hpp"
#include "kudnn_tensor_data.hpp"

__fp16 g_weight[10][3] = {
    {0.1, 0.12, 0.15}, {0.21, 0.22, 0.25}, {0.11, 0.21, 0.32}, {0.77, 0.02, 0.05}, {0.54, 0.32, 0.08},
    {0.14, 0.42, 0.25}, {0.31, 0.2, 0.07}, {0.22, 0.14, 0.03}, {0.17, 0.28, 0.32}, {0.44, 0.26, 0.23}
};

__fp16* EmbeddingRef(__fp16 *weight, int m, int n, int *indice, int dim, int padding_idx = -1)
{
    __fp16* out = (__fp16*)malloc(dim * n * sizeof(__fp16));
    __fp16 *ptr = out;
    for (int i = 0; i < dim; i++) {
        if (indice[i] < 0 || indice[i] >= m) {
            free(out);
            return nullptr;
        }
        if (padding_idx == indice[i]) {
            memset(&weight[indice[i] * n], 0, n * sizeof(__fp16));
        }
        memcpy(ptr, &weight[indice[i] * n], n * sizeof(__fp16));
        ptr = ptr + n;
    }
    return out;
}

// test cases
TEST(EmbeddingGTests, FWDRun1)
{
    int num_embeddings = 10;
    int embedding_dim = 3;
    int num = 4;
    KuDNN::EmbeddingLayerFWD embedding(num_embeddings, embedding_dim, (__fp16*)g_weight);
    int indice[4] = {1, 2, 3, 7};
    __fp16 *out = embedding.Run(indice, num);
    __fp16 *ref = EmbeddingRef((__fp16*)g_weight, num_embeddings, embedding_dim, indice, num);
    for (int i = 0; i < embedding_dim * num; i++) {
        EXPECT_EQ(out[i], ref[i]);
    }
    free(out);
    free(ref);
}

TEST(EmbeddingGTests, FWDRun2)
{
    int num_embeddings = 10;
    int embedding_dim = 3;
    int num = 4;
    int padding_idx = 2;
    KuDNN::EmbeddingLayerFWD embedding(num_embeddings, embedding_dim, (__fp16*)g_weight, padding_idx);
    int indice[4] = {1, 2, 3, 7};
    __fp16 *out = embedding.Run(indice, num);
    __fp16 *ref = EmbeddingRef((__fp16*)g_weight, num_embeddings, embedding_dim, indice, num, padding_idx);
    for (int i = 0; i < embedding_dim * num; i++) {
        EXPECT_EQ(out[i], ref[i]);
    }
    free(out);
    free(ref);
}


// test cases
TEST(EmbeddingGTests, FWDCopyConstructor1)
{
    int num_embeddings = 10;
    int embedding_dim = 3;
    KuDNN::EmbeddingLayerFWD embedding1(num_embeddings, embedding_dim, (__fp16*)g_weight);
    KuDNN::EmbeddingLayerFWD embedding2(num_embeddings, embedding_dim, (__fp16*)g_weight);
    embedding1 = embedding2;
}

TEST(EmbeddingGTests, FWDCopyConstructor2)
{
    int num_embeddings = 10;
    int embedding_dim = 3;
    KuDNN::EmbeddingLayerFWD embedding1(num_embeddings, embedding_dim, (__fp16*)g_weight);
    KuDNN::EmbeddingLayerFWD embedding2(embedding1);
}

TEST(EmbeddingGTests, FWDMoveConstructor1)
{
    int num_embeddings = 10;
    int embedding_dim = 3;
    KuDNN::EmbeddingLayerFWD embedding1(num_embeddings, embedding_dim, (__fp16*)g_weight);
    KuDNN::EmbeddingLayerFWD embedding2(std::move(embedding1));
}

TEST(EmbeddingGTests, FWDMoveConstructor2)
{
    int num_embeddings = 10;
    int embedding_dim = 3;
    KuDNN::EmbeddingLayerFWD embedding1(num_embeddings, embedding_dim, (__fp16*)g_weight);
    KuDNN::EmbeddingLayerFWD embedding2(num_embeddings, embedding_dim, (__fp16*)g_weight);
    embedding1 = std::move(embedding2);
}

TEST(EmbeddingGTests, EXP_CASE1)
{
    EXPECT_THROW(KuDNN::EmbeddingLayerFWD embedding(10, 3, nullptr), KuDNN::Service::LogicError);
}

TEST(EmbeddingGTests, EXP_CASE2)
{
    EXPECT_THROW(KuDNN::EmbeddingLayerFWD embedding(-1, 3, (__fp16*)g_weight, 2), KuDNN::Service::LogicError);
}

TEST(EmbeddingGTests, EXP_CASE3)
{
    EXPECT_THROW(KuDNN::EmbeddingLayerFWD embedding(10, -1, (__fp16*)g_weight, 2), KuDNN::Service::LogicError);
}

TEST(EmbeddingGTests, EXP_CASE4)
{
    EXPECT_THROW(KuDNN::EmbeddingLayerFWD embedding(10, 3, (__fp16*)g_weight, 15), KuDNN::Service::LogicError);
}

TEST(EmbeddingGTests, EXP_CASE5)
{
    int num_embeddings = 10;
    int embedding_dim = 3;
    int num = -1;
    int padding_idx = 2;
    KuDNN::EmbeddingLayerFWD embedding(num_embeddings, embedding_dim, (__fp16*)g_weight, padding_idx);
    int indice[4] = {1, 2, 3, 7};
    __fp16 *out = embedding.Run(indice, num);
    EXPECT_EQ(out, nullptr);
    free(out);
}

TEST(EmbeddingGTests, EXP_CASE6)
{
    int num_embeddings = 10;
    int embedding_dim = 3;
    int num = 4;
    int padding_idx = 2;
    KuDNN::EmbeddingLayerFWD embedding(num_embeddings, embedding_dim, (__fp16*)g_weight, padding_idx);
    int indice[4] = {1, 2, 3, -7};
    __fp16 *out = embedding.Run(indice, num);
    EXPECT_EQ(out, nullptr);
    free(out);
}

TEST(EmbeddingGTests, EXP_ZeroEmbeddingDim)
{
    EXPECT_THROW(KuDNN::EmbeddingLayerFWD embedding(10, 0, (__fp16*)g_weight, 2), KuDNN::Service::LogicError);
}