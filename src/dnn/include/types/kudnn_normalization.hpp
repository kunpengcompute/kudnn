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

#ifndef KUDNN_NORMALIZATION_HPP
#define KUDNN_NORMALIZATION_HPP

namespace KuDNN {

enum class KUDNN_API_PUBLIC NormalizationFlags : std::uint32_t {
    NONE = 0x0U,
    USE_GLOBAL_STATS = 0x1U,
    USE_SCALE = 0x2U,
    USE_SHIFT = 0x4U,
    FUSE_NORM_RELU = 0x8U
};

KUDNN_API_PUBLIC NormalizationFlags operator&(const NormalizationFlags &lhs, const NormalizationFlags &rhs);
KUDNN_API_PUBLIC NormalizationFlags operator|(const NormalizationFlags &lhs, const NormalizationFlags &rhs);
KUDNN_API_PUBLIC NormalizationFlags &operator&=(NormalizationFlags &lhs, const NormalizationFlags &rhs);
KUDNN_API_PUBLIC NormalizationFlags &operator|=(NormalizationFlags &lhs, const NormalizationFlags &rhs);

} // namespace KuDNN

#endif // KUDNN_NORMALIZATION_HPP
