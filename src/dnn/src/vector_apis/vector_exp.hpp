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

#ifndef KUDNN_VECTOR_EXP
#define KUDNN_VECTOR_EXP

#include <arm_sve.h>

#include "internal_funcs.hpp"

namespace KuDNN {
namespace Detail {
namespace Vector {

template<typename T>
inline constexpr T LN_MIN = -87.0f;

template<typename T>
inline constexpr T INVLN2 = 0x1.715476p+0f;

template<typename T>
inline constexpr T SHIFTER = 0x1.803f8p17f;

template<>
inline constexpr __fp16 SHIFTER<__fp16> = +0x1.78000000p+0005;

template<typename T>
inline constexpr T INVLN2_LOW = 0x1.4ae0cp-26f;

template<typename T>
inline constexpr T POLY_C0 = 0x1.62e486p-1f;

template<typename T>
inline constexpr T POLY_C1 = 0x1.ebfbe0p-3f;

template <typename ComputeType, SizeType N>
struct ExpaN final {
    static constexpr auto Reinterpret(const typename Vec<ComputeType>::Type &in) noexcept
    {
        if constexpr (sizeof(ComputeType) == 4) {
            return svreinterpret_u32(in);
        } else {
            return svreinterpret_u16(in);
        }
    }
    static void Do(const typename Vec<ComputeType>::Type &in0,
        const typename Vec<ComputeType>::Type &in1, const typename Vec<ComputeType>::Type &in2,
        const typename Vec<ComputeType>::Type &in3, typename Vec<ComputeType>::Type &out0,
        typename Vec<ComputeType>::Type &out1, typename Vec<ComputeType>::Type &out2,
        typename Vec<ComputeType>::Type &out3) noexcept
    {
        static_assert(N <= 4, "Pseudo unroll by 4 is maximum");
        auto &&k0 = Reinterpret(in0);
        out0 = svexpa(k0);
        if constexpr (N > 1) {
            auto &&k1 = Reinterpret(in1);
            out1 = svexpa(k1);
            if constexpr (N > 2) {
                auto &&k2 = Reinterpret(in2);
                out2 = svexpa(k2);
                if constexpr (N > 3) {
                    auto &&k3 = Reinterpret(in3);
                    out3 = svexpa(k3);
                }
            }
        }
    }
};

template <typename ComputeType, SizeType N>
struct AccuracyIncreaseExpN final {
    static void Do(const svbool_t& pg0, const svbool_t &pg1,
        const svbool_t &pg2, const svbool_t &pg3,
        typename Vector::Vec<ComputeType>::Type &x0, typename Vector::Vec<ComputeType>::Type &x1,
        typename Vector::Vec<ComputeType>::Type &x2, typename Vector::Vec<ComputeType>::Type &x3,
        typename Vector::Vec<ComputeType>::Type &y0, typename Vector::Vec<ComputeType>::Type &y1,
        typename Vector::Vec<ComputeType>::Type &y2, typename Vector::Vec<ComputeType>::Type &y3,
        typename Vector::Vec<ComputeType>::Type &z0, typename Vector::Vec<ComputeType>::Type &z1,
        typename Vector::Vec<ComputeType>::Type &z2, typename Vector::Vec<ComputeType>::Type &z3,
        typename Vector::Vec<ComputeType>::Type &out0, typename Vector::Vec<ComputeType>::Type &out1,
        typename Vector::Vec<ComputeType>::Type &out2, typename Vector::Vec<ComputeType>::Type &out3) noexcept
    {
        if constexpr (std::is_same_v<ComputeType, float>) {
            Vector::MlaZN<ComputeType, N>::Do(pg0, pg1, pg2, pg3,
                x0, x1, x2, x3, y0, y1, y2, y3, z0, z1, z2, z3, out0, out1, out2, out3);
        }
    }
};

template <typename ComputeType, SizeType N>
void ExpN<ComputeType, N>::Do(const svbool_t& pg0,
    const svbool_t &pg1, const svbool_t &pg2, const svbool_t &pg3,
    typename Vector::Vec<ComputeType>::Type &in0, typename Vector::Vec<ComputeType>::Type &in1,
    typename Vector::Vec<ComputeType>::Type &in2, typename Vector::Vec<ComputeType>::Type &in3,
    typename Vector::Vec<ComputeType>::Type &out0, typename Vector::Vec<ComputeType>::Type &out1,
    typename Vector::Vec<ComputeType>::Type &out2, typename Vector::Vec<ComputeType>::Type &out3) noexcept
{
    typename Vector::Vec<ComputeType>::Type lnMinConst0;
    typename Vector::Vec<ComputeType>::Type lnMinConst1;
    typename Vector::Vec<ComputeType>::Type lnMinConst2;
    typename Vector::Vec<ComputeType>::Type lnMinConst3;
    Vector::DupFloatN<ComputeType, ComputeType, N>::Do(LN_MIN<ComputeType>,
        lnMinConst0, lnMinConst1, lnMinConst2, lnMinConst3);
    typename Vector::Vec<ComputeType>::Type invLn2Const0;
    typename Vector::Vec<ComputeType>::Type invLn2Const1;
    typename Vector::Vec<ComputeType>::Type invLn2Const2;
    typename Vector::Vec<ComputeType>::Type invLn2Const3;
    Vector::DupFloatN<ComputeType, ComputeType, N>::Do(INVLN2<ComputeType>,
        invLn2Const0, invLn2Const1, invLn2Const2, invLn2Const3);
    typename Vector::Vec<ComputeType>::Type invLn2LowConst0;
    typename Vector::Vec<ComputeType>::Type invLn2LowConst1;
    typename Vector::Vec<ComputeType>::Type invLn2LowConst2;
    typename Vector::Vec<ComputeType>::Type invLn2LowConst3;
    Vector::DupFloatN<ComputeType, ComputeType, N>::Do(INVLN2_LOW<ComputeType>,
        invLn2LowConst0, invLn2LowConst1, invLn2LowConst2, invLn2LowConst3);
    typename Vector::Vec<ComputeType>::Type shifterConst0;
    typename Vector::Vec<ComputeType>::Type shifterConst1;
    typename Vector::Vec<ComputeType>::Type shifterConst2;
    typename Vector::Vec<ComputeType>::Type shifterConst3;
    Vector::DupFloatN<ComputeType, ComputeType, N>::Do(SHIFTER<ComputeType>,
        shifterConst0, shifterConst1, shifterConst2, shifterConst3);
    typename Vector::Vec<ComputeType>::Type polyC0Const0;
    typename Vector::Vec<ComputeType>::Type polyC0Const1;
    typename Vector::Vec<ComputeType>::Type polyC0Const2;
    typename Vector::Vec<ComputeType>::Type polyC0Const3;
    Vector::DupFloatN<ComputeType, ComputeType, N>::Do(POLY_C0<ComputeType>,
        polyC0Const0, polyC0Const1, polyC0Const2, polyC0Const3);
    typename Vector::Vec<ComputeType>::Type polyC1Const0;
    typename Vector::Vec<ComputeType>::Type polyC1Const1;
    typename Vector::Vec<ComputeType>::Type polyC1Const2;
    typename Vector::Vec<ComputeType>::Type polyC1Const3;
    Vector::DupFloatN<ComputeType, ComputeType, N>::Do(POLY_C1<ComputeType>,
        polyC1Const0, polyC1Const1, polyC1Const2, polyC1Const3);
    typename Vector::Vec<ComputeType>::Type allOneConst0;
    typename Vector::Vec<ComputeType>::Type allOneConst1;
    typename Vector::Vec<ComputeType>::Type allOneConst2;
    typename Vector::Vec<ComputeType>::Type allOneConst3;
    Vector::DupFloatN<ComputeType, ComputeType, N>::Do(1.0f,
        allOneConst0, allOneConst1, allOneConst2, allOneConst3);

    svbool_t npgPG0;
    svbool_t npgPG1;
    svbool_t npgPG2;
    svbool_t npgPG3;
    Vector::GeN<ComputeType, N>::Do(pg0, pg1, pg2, pg3, in0, in1, in2, in3,
        lnMinConst0, lnMinConst1, lnMinConst2, lnMinConst3, npgPG0, npgPG1, npgPG2, npgPG3);
    typename Vector::Vec<ComputeType>::Type ki0;
    typename Vector::Vec<ComputeType>::Type ki1;
    typename Vector::Vec<ComputeType>::Type ki2;
    typename Vector::Vec<ComputeType>::Type ki3;
    Vector::MadZN<ComputeType, N>::Do(npgPG0, npgPG1, npgPG2, npgPG3, in0, in1, in2, in3,
        invLn2Const0, invLn2Const1, invLn2Const2, invLn2Const3,
        shifterConst0, shifterConst1, shifterConst2, shifterConst3,
        ki0, ki1, ki2, ki3);
    typename Vector::Vec<ComputeType>::Type kd0;
    typename Vector::Vec<ComputeType>::Type kd1;
    typename Vector::Vec<ComputeType>::Type kd2;
    typename Vector::Vec<ComputeType>::Type kd3;
    Vector::SubZN<ComputeType, N>::Do(npgPG0, npgPG1, npgPG2, npgPG3,
        ki0, ki1, ki2, ki3, shifterConst0, shifterConst1, shifterConst2, shifterConst3,
        kd0, kd1, kd2, kd3);
    typename Vector::Vec<ComputeType>::Type expk0;
    typename Vector::Vec<ComputeType>::Type expk1;
    typename Vector::Vec<ComputeType>::Type expk2;
    typename Vector::Vec<ComputeType>::Type expk3;
    Vector::ExpaN<ComputeType, N>::Do(ki0, ki1, ki2, ki3, expk0, expk1, expk2, expk3);
    typename Vector::Vec<ComputeType>::Type r0;
    typename Vector::Vec<ComputeType>::Type r1;
    typename Vector::Vec<ComputeType>::Type r2;
    typename Vector::Vec<ComputeType>::Type r3;
    Vector::NmlsZN<ComputeType, N>::Do(npgPG0, npgPG1, npgPG2, npgPG3,
        kd0, kd1, kd2, kd3, in0, in1, in2, in3, invLn2Const0, invLn2Const1, invLn2Const2, invLn2Const3,
        r0, r1, r2, r3);
    Vector::AccuracyIncreaseExpN<ComputeType, N>::Do(npgPG0, npgPG1, npgPG2, npgPG3,
        r0, r1, r2, r3, in0, in1, in2, in3, invLn2LowConst0, invLn2LowConst1, invLn2LowConst2, invLn2LowConst3,
        r0, r1, r2, r3);
    typename Vector::Vec<ComputeType>::Type poly0;
    typename Vector::Vec<ComputeType>::Type poly1;
    typename Vector::Vec<ComputeType>::Type poly2;
    typename Vector::Vec<ComputeType>::Type poly3;
    Vector::MlaZN<ComputeType, N>::Do(npgPG0, npgPG1, npgPG2, npgPG3,
        polyC0Const0, polyC0Const1, polyC0Const2, polyC0Const3, r0, r1, r2, r3,
        polyC1Const0, polyC1Const1, polyC1Const2, polyC1Const3, poly0, poly1, poly2, poly3);
    Vector::MlaZN<ComputeType, N>::Do(npgPG0, npgPG1, npgPG2, npgPG3,
        allOneConst0, allOneConst1, allOneConst2, allOneConst3, poly0, poly1, poly2, poly3,
        r0, r1, r2, r3, poly0, poly1, poly2, poly3);
    Vector::MulZN<ComputeType, N>::Do(npgPG0, npgPG1, npgPG2, npgPG3,
        expk0, expk1, expk2, expk3, poly0, poly1, poly2, poly3,
        out0, out1, out2, out3);
}

} // Vector
} // Detail
} // KuDNN

#endif // KUDNN_VECTOR_EXP
