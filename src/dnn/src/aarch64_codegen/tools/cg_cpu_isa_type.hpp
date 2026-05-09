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
    VectorComputation_128_BIT = 1u << 1,
    VectorComputation_256_BIT = 1u << 2,
    VectorComputation_384_BIT = 1u << 3,
    VectorComputation_512_BIT = 1u << 4,
};

enum class CPUISAT : unsigned {
    ISA_UNDEF = 0u,
    ASIMD = static_cast<unsigned>(CPUISABIT::ASIMD_BIT),
    VectorComputation_128 = static_cast<unsigned>(CPUISABIT::VectorComputation_128_BIT) | ASIMD,
    VectorComputation_256 = static_cast<unsigned>(CPUISABIT::VectorComputation_256_BIT) | VectorComputation_128,
    VectorComputation_384 = static_cast<unsigned>(CPUISABIT::VectorComputation_384_BIT) | VectorComputation_256,
    VectorComputation_512 = static_cast<unsigned>(CPUISABIT::VectorComputation_512_BIT) | VectorComputation_384,
    ISA_ALL = ~0u,
};

} // namespace ARMCG

#endif // CG_CPU_ISA_TYPE_HPP
