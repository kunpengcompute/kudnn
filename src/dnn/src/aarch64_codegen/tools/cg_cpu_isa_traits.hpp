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

#ifndef CG_CPU_ISA_TRAITS_HPP
#define CG_CPU_ISA_TRAITS_HPP

#include <type_traits>
#include <cassert>
/* in order to make selinux happy memory that would be marked with X-bit should
 * be obtained with mmap */
#ifdef __GNUC__
#define CG_USE_MMAP_ALLOCATOR
#endif

#include "cg_cpu_isa_type.hpp"
#include "tools/cg_util.hpp"
#include "src/cg_reg.hpp"
#include "src/cg_reg_v.hpp"

namespace ARMCG {

enum class CpuIsaCmpT {
    // List of infix comparison relations between two  CPUISAT
    // where we take isa_1 and isa_2 to be two  CPUISAT instances.

    // isa_1 SUBSET isa_2 if all feature flags supported by isa_1
    // are supported by isa_2 as well (equality allowed)
    SUBSET,

    // isa_1 SUPERSET isa_2 if all feature flags supported by isa_2
    // are supported by isa_1 as well (equality allowed)
    SUPERSET,

    // Few more options that (depending upon need) can be enabled in future

    // 1. PROPER_SUBSET: isa_1 SUBSET isa_2 and isa_1 != isa_2
    // 2. PROPER_SUPERSET: isa_1 SUPERSET isa_2 and isa_1 != isa_2
};

inline bool CompareIsa(CPUISAT isa1, CpuIsaCmpT cmp, CPUISAT isa2)
{
    // By default, comparison between ISA ignores ISA specific hints
    unsigned mask1 = static_cast<unsigned>(isa1);
    unsigned mask2 = static_cast<unsigned>(isa2);
    unsigned maskcommon = mask1 & mask2;

    switch (cmp) {
        case CpuIsaCmpT::SUBSET:
            return mask1 == maskcommon;
        case CpuIsaCmpT::SUPERSET:
            return mask2 == maskcommon;
        default:
            assert(!"unsupported comparison of isa");
            return false;
    }
}

inline bool IsSubset(CPUISAT isa1, CPUISAT isa2)
{
    return CompareIsa(isa1, CpuIsaCmpT::SUBSET, isa2);
}

inline bool IsSuperset(CPUISAT isa1, CPUISAT isa2)
{
    return CompareIsa(isa1, CpuIsaCmpT::SUPERSET, isa2);
}

inline const ARMCG::Util::Cpu &GetCpu()
{
    const static ARMCG::Util::Cpu DEV_CPU;
    return DEV_CPU;
}

template <CPUISAT>
struct CpuIsaTraits {};

template <>
struct CpuIsaTraits<CPUISAT::ASIMD> {
    using TReg = VReg;
    using TRegB = VReg16B;
    using TRegH = VReg8H;
    using TRegS = VReg4S;
    using TRegD = VReg2D;
    static constexpr int VLEN_SHIFT = 4;
    static constexpr int VLEN = 16;
    static constexpr int N_VREGS = 32;
    static constexpr const char *USER_OPTION_ENV = "advanced_simd";
};

template <>
struct CpuIsaTraits<CPUISAT::VectorComputation_128> {
    using TReg = ZReg;
    using TRegB = ZRegB;
    using TRegH = ZRegH;
    using TRegS = ZRegS;
    using TRegD = ZRegD;
    static constexpr int VLEN_SHIFT = 4;
    static constexpr int VLEN = 16;
    static constexpr int N_VREGS = 32;
    static constexpr const char *USER_OPTION_ENV = "VectorComputation_ ## bits";
};

template <>
struct CpuIsaTraits<CPUISAT::VectorComputation_256> {
    using TReg = ZReg;
    using TRegB = ZRegB;
    using TRegH = ZRegH;
    using TRegS = ZRegS;
    using TRegD = ZRegD;
    static constexpr int VLEN_SHIFT = 5;
    static constexpr int VLEN = 32;
    static constexpr int N_VREGS = 32;
    static constexpr const char *USER_OPTION_ENV = "VectorComputation_ ## bits";
};

template <>
struct CpuIsaTraits<CPUISAT::VectorComputation_512> {
    using TReg = ZReg;
    using TRegB = ZRegB;
    using TRegH = ZRegH;
    using TRegS = ZRegS;
    using TRegD = ZRegD;
    static constexpr int VLEN_SHIFT = 6;
    static constexpr int VLEN = 64;
    static constexpr int N_VREGS = 32;
    static constexpr const char *USER_OPTION_ENV = "VectorComputation_ ## bits";
};

inline bool MayIUse(const CPUISAT cpuIsa)
{
    using namespace ARMCG::Util;
    switch (cpuIsa) {
        case CPUISAT::ASIMD:
            return ARMCG::GetCpu().Has(HwCap::CG_HWCAP_VectorComputation);
        case CPUISAT::VectorComputation_128:
            return ARMCG::GetCpu().Has(HwCap::CG_HWCAP_VectorComputation) &&
                   ARMCG::GetCpu().GetSveLen() >= SveLen::VectorComputation_128;
        case CPUISAT::VectorComputation_256:
            return ARMCG::GetCpu().Has(HwCap::CG_HWCAP_VectorComputation) &&
                   ARMCG::GetCpu().GetSveLen() >= SveLen::VectorComputation_256;
        case CPUISAT::VectorComputation_384:
            return ARMCG::GetCpu().Has(HwCap::CG_HWCAP_VectorComputation) &&
                   ARMCG::GetCpu().GetSveLen() >= SveLen::VectorComputation_384;
        case CPUISAT::VectorComputation_512:
            return ARMCG::GetCpu().Has(HwCap::CG_HWCAP_VectorComputation) &&
                   ARMCG::GetCpu().GetSveLen() >= SveLen::VectorComputation_512;
        case CPUISAT::ISA_UNDEF:
            return true;
        case CPUISAT::ISA_ALL:
            return false;
        default:
            return false;
    }
    return false;
}

inline Util::SveLen GetSveLength()
{
    return ARMCG::GetCpu().GetSveLen();
}

inline bool MayIUseAtomic()
{
    return ARMCG::GetCpu().IsAtomicSupported();
}

inline bool MayIUseBF16()
{
    return ARMCG::GetCpu().IsBf16Supported();
}

} // namespace ARMCG

#endif // CG_CPU_ISA_TRAITS_HPP
