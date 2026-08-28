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

#ifndef KUDNN_ROPE_HPP
#define KUDNN_ROPE_HPP

#include <memory>

#include "types/kudnn_tensor_info.hpp"
#include "service/kudnn_err_codes.hpp"

namespace KuDNN {

enum class RopeKind : std::uint32_t {
    OpenSora = 0x0U,
    DeepSeek = 0x1U,
};

namespace Detail {

class RopeLayerFWDImpl;

} // namespace Detail

class RopeLayerFWD final {
public:
    using SizeType = ::KuDNN::SizeType;
    static Status ValidateInput(const int maxPosition, const int dim, const float base,
                                const float scalingFactor) noexcept;
    RopeLayerFWD(const int maxPosition, const int dim, const float base = 10000,
                 const float scalingFactor = 1.0f) noexcept(false);
    RopeLayerFWD(const RopeLayerFWD &other) noexcept(false);
    RopeLayerFWD(RopeLayerFWD &&other) noexcept;
    RopeLayerFWD &operator=(const RopeLayerFWD &other) noexcept(false);
    RopeLayerFWD &operator=(RopeLayerFWD &&other) noexcept;
    void Run(void *query, void *key, const int qStride, const int kStride, const int batchSize, const int seqLen,
             const int qHeads, const int kHeads, const int *positionIds, RopeKind kind = RopeKind::OpenSora) const
        noexcept(false);
    ~RopeLayerFWD() noexcept;

private:
    std::unique_ptr<Detail::RopeLayerFWDImpl> pImpl;
};

} // namespace KuDNN

#endif // KUDNN_ROPE_HPP
