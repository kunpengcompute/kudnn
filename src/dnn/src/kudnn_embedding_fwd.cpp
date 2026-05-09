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

#include <cstddef>
#include <cmath>
#include <cstring>
#include <limits>

#include "kudnn.hpp"

#include "internal_funcs.hpp"
#include "threading_impl.hpp"

namespace KuDNN {
namespace Detail {

class EmbeddingLayerFWDImpl final {
public:
    EmbeddingLayerFWDImpl(const int num_embeddings, const int embedding_dim, __fp16 *_weight, const int padding_idx,
                          const float max_norm, const float norm_type) noexcept(false)
        : num_embeddings(num_embeddings),
          embedding_dim(embedding_dim),
          weight(_weight),
          padding_idx(padding_idx),
          max_norm(max_norm),
          norm_type(norm_type)
    {
        Service::ThrowOnStatus(
            EmbeddingLayerFWD::ValidateInput(num_embeddings, embedding_dim, weight, padding_idx, max_norm, norm_type),
            "Embedding layer");
        if (padding_idx != -1) {
            memset(&weight[embedding_dim * padding_idx], 0, sizeof(__fp16) * embedding_dim);
        }
    }

    __fp16 *Run(int *indice, int dims) const noexcept(false)
    {
        if (dims <= 0) {
            return nullptr;
        }
        (void)max_norm;
        (void)norm_type;
        (void)padding_idx;

        __fp16 *out = (__fp16 *)malloc(sizeof(__fp16) * dims * embedding_dim);
        if (out == nullptr) {
            return nullptr;
        }
        __fp16 *ptr = out;
        for (int i = 0; i < dims; i++) {
            if (indice[i] < 0 || indice[i] >= num_embeddings) {
                free(out);
                return nullptr;
            }

            memcpy(ptr, &weight[embedding_dim * indice[i]], sizeof(__fp16) * embedding_dim);
            ptr += embedding_dim;
        }
        return out;
    }

private:
    int num_embeddings;
    int embedding_dim;
    __fp16 *weight;
    int padding_idx;
    float max_norm;
    float norm_type;
};

} // namespace Detail

EmbeddingLayerFWD::~EmbeddingLayerFWD() noexcept {}

EmbeddingLayerFWD::EmbeddingLayerFWD(const int num_embeddings, const int embedding_dim, __fp16 *_weight,
                                     const int padding_idx, const float max_norm, const float norm_type) noexcept(false)
    : pImpl(new Detail::EmbeddingLayerFWDImpl(num_embeddings, embedding_dim, _weight, padding_idx, max_norm, norm_type))
{
}

EmbeddingLayerFWD::EmbeddingLayerFWD(const EmbeddingLayerFWD &other) noexcept(false)
    : pImpl(new Detail::EmbeddingLayerFWDImpl{*(other.pImpl.get())})
{
}

EmbeddingLayerFWD::EmbeddingLayerFWD(EmbeddingLayerFWD &&other) noexcept : pImpl(std::move(other.pImpl)) {}

EmbeddingLayerFWD &EmbeddingLayerFWD::operator=(const EmbeddingLayerFWD &other) noexcept(false)
{
    if (this != &other) {
        this->pImpl.reset(new Detail::EmbeddingLayerFWDImpl{*(other.pImpl.get())});
    }
    return *this;
}

EmbeddingLayerFWD &EmbeddingLayerFWD::operator=(EmbeddingLayerFWD &&other) noexcept
{
    if (this != &other) {
        this->pImpl = std::move(other.pImpl);
    }
    return *this;
}

__fp16 *EmbeddingLayerFWD::Run(int *indice, int dims) const noexcept(false)
{
    return pImpl->Run(indice, dims);
}

Status EmbeddingLayerFWD::ValidateInput(const int num_embeddings, const int embedding_dim, __fp16 *_weight,
                                        const int padding_idx, const float max_norm, const float norm_type) noexcept
{
    (void)max_norm;
    (void)norm_type;
    if (_weight == nullptr) {
        return Status::BAD_ARGUMENTS;
    }
    if (num_embeddings <= 0) {
        return Status::BAD_ARGUMENTS;
    }
    if (embedding_dim <= 0) {
        return Status::BAD_ARGUMENTS;
    }
    if (padding_idx >= num_embeddings || padding_idx < -1) {
        return Status::BAD_ARGUMENTS;
    }

    return Status::SUCCESS;
}

} // namespace KuDNN
