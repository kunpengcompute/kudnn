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

#include <memory>
#include <cmath>
#include <cstddef>

#include <arm_sve.h>

#include "kudnn.hpp"

#include "kudnn_group_normalization_impl.hpp"
#include "internal_funcs.hpp"
#include "threading_impl.hpp"
#include "vector_wrappers.hpp"

namespace KuDNN {

inline NormalizationFlags operator & (const NormalizationFlags &lhs, const NormalizationFlags &rhs)
{
    return static_cast<NormalizationFlags>(static_cast<std::uint32_t>(lhs) & static_cast<std::uint32_t>(rhs));
}

inline NormalizationFlags operator | (const NormalizationFlags &lhs, const NormalizationFlags &rhs)
{
    return static_cast<NormalizationFlags>(static_cast<std::uint32_t>(lhs) | static_cast<std::uint32_t>(rhs));
}

inline NormalizationFlags &operator |= (NormalizationFlags &lhs, const NormalizationFlags &rhs)
{
    lhs = lhs | rhs;
    return lhs;
}

inline NormalizationFlags &operator &= (NormalizationFlags &lhs, const NormalizationFlags &rhs)
{
    lhs = lhs & rhs;
    return lhs;
}

} // KuDNN
