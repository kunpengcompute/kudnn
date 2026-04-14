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


#ifndef KUDNN_EMBEDDING_HPP
#define KUDNN_EMBEDDING_HPP

#include <memory>

#include "types/kudnn_tensor_info.hpp"
#include "service/kudnn_err_codes.hpp"

namespace KuDNN {

namespace Detail {

class EmbeddingLayerFWDImpl;

} // Detail

class KUDNN_API_PUBLIC EmbeddingLayerFWD final {
public:
    using SizeType = ::KuDNN::SizeType;
    static Status ValidateInput(const int num_embeddings, const int embedding_dim, __fp16 *_weight,
        const int padding_idx, const float max_norm, const float norm_type) noexcept;
    EmbeddingLayerFWD(const int num_embeddings, const int embedding_dim, __fp16 *_weight,
        const int padding_idx = -1, const float max_norm = 1.0, const float norm_type = 2.0) noexcept(false);
    EmbeddingLayerFWD(const EmbeddingLayerFWD &other) noexcept(false);
    EmbeddingLayerFWD(EmbeddingLayerFWD &&other) noexcept;
    EmbeddingLayerFWD& operator=(const EmbeddingLayerFWD &other) noexcept(false);
    EmbeddingLayerFWD& operator=(EmbeddingLayerFWD &&other) noexcept;
    __fp16* Run(int *indice, int dims) const noexcept(false);
    ~EmbeddingLayerFWD() noexcept;
private:
    std::unique_ptr<Detail::EmbeddingLayerFWDImpl> pImpl;
};

} // KuDNN

#endif // KUDNN_SOFTMAX_HPP
