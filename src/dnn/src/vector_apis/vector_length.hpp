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

#ifndef KUDNN_VECTOR_LENGTH
#define KUDNN_VECTOR_LENGTH

#include <arm_sve.h>

#include "kudnn.hpp"

namespace KuDNN {
namespace Detail {
namespace Vector {

template <>
KUDNN_INLINE inline SizeType Length<float>()
{
    return svcntw();
}
template <>
KUDNN_INLINE inline SizeType Length<__fp16>()
{
    return svcnth();
}
template <>
KUDNN_INLINE inline SizeType Length<__bf16>()
{
    return svcnth();
}

template <>
KUDNN_INLINE inline SizeType Length<std::int8_t>()
{
    return svcntb();
}

template <>
KUDNN_INLINE inline SizeType Length<std::uint8_t>()
{
    return svcntb();
}

template <>
KUDNN_INLINE inline SizeType Length<std::int32_t>()
{
    return svcntw();
}

} // namespace Vector
} // namespace Detail
} // namespace KuDNN

#endif // KUDNN_VECTOR_LENGTH
