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

#ifndef CG_INNER_HPP
#define CG_INNER_HPP

#include <cstdint>
#include <unistd.h>

#include "cg_error.hpp"

namespace ARMCG {

enum class MaxCodeSize { DEFAULT = 4096 };

namespace Inner {

inline size_t GetPageSize()
{
    static const size_t PAGE_SIZE = static_cast<size_t>(sysconf(_SC_PAGESIZE));
    return PAGE_SIZE;
}

inline bool IsInDisp8(uint32_t x)
{
    return x >= 0xFFFFFF80 || x <= 0x7F;
}

inline bool IsInInt32(uint64_t x)
{
    return x >= ~uint64_t(0x7fffffffu) || x <= 0x7FFFFFFFU;
}

inline uint32_t VerifyInInt32(uint64_t x)
{
    if (!IsInInt32(x)) {
        throw Error(Err::OFFSET_IS_TOO_BIG);
    }
    return static_cast<uint32_t>(x);
}

constexpr uint32_t GenSysInstOp(uint32_t op1, uint32_t crn, uint32_t crm, uint32_t op2)
{
    return (op1 << BITFIELD_11) | (crn << BITFIELD_7) | (crm << BITFIELD_3) | op2;
}

} // namespace Inner
} // namespace ARMCG

#endif // CG_INNER_HPP
