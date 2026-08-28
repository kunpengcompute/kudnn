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

#ifndef CG_CPU_ISA_TYPE_HPP
#define CG_CPU_ISA_TYPE_HPP

namespace ARMCG {
/// CPU instruction set flags

enum class CPUISABIT : unsigned {
    ASIMD_BIT = 1u << 0,
    VC_128_BIT = 1u << 1,
    VC_256_BIT = 1u << 2,
    VC_384_BIT = 1u << 3,
    VC_512_BIT = 1u << 4,
};

enum class CPUISAT : unsigned {
    ISA_UNDEF = 0u,
    ASIMD = static_cast<unsigned>(CPUISABIT::ASIMD_BIT),
    VC_128 = static_cast<unsigned>(CPUISABIT::VC_128_BIT) | ASIMD,
    VC_256 = static_cast<unsigned>(CPUISABIT::VC_256_BIT) | VC_128,
    VC_384 = static_cast<unsigned>(CPUISABIT::VC_384_BIT) | VC_256,
    VC_512 = static_cast<unsigned>(CPUISABIT::VC_512_BIT) | VC_384,
    ISA_ALL = ~0u,
};

} // ARMCG

#endif  // CG_CPU_ISA_TYPE_HPP
