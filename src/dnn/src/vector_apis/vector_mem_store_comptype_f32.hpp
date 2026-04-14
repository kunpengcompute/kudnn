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

#ifndef KUDNN_VECTOR_MEM_STORE_COMPTYPE_F32_HPP
#define KUDNN_VECTOR_MEM_STORE_COMPTYPE_F32_HPP

#include <cstdint>
#include <limits>
#include <arm_neon.h>
#include <arm_sve.h>

#include "kudnn.hpp"

#include "internal_funcs.hpp"
#include "vector_length.hpp"

namespace KuDNN {
namespace Detail {
namespace Vector {

// CompType is float

template <>
struct StoreN<float, float, 1> final {
    static void Do(const svbool_t& pg0, const svbool_t &pg1, const svbool_t &pg2,
        const svbool_t &pg3, const typename Vec<float>::Type &out0, const typename Vec<float>::Type &out1,
        const typename Vec<float>::Type &out2, const typename Vec<float>::Type &out3, float *dst) noexcept
    {
        svst1_f32(pg0, dst, out0);
        Service::Unused(pg1, out1, pg2, out2, pg3, out3);
    }
};

template <>
struct StoreN<float, float, 2> final { // 2 means that store two float regs from float regs
    static void Do(const svbool_t& pg0, const svbool_t &pg1, const svbool_t &pg2,
        const svbool_t &pg3, const typename Vec<float>::Type &out0, const typename Vec<float>::Type &out1,
        const typename Vec<float>::Type &out2, const typename Vec<float>::Type &out3, float *dst) noexcept
    {
        svst1_f32(pg0, dst, out0);
        svst1_f32(pg1, dst + Vector::Length<float>(), out1);
        Service::Unused(pg2, out2, pg3, out3);
    }
};

template <>
struct StoreN<float, float, 4> final { // 4 means that store four float regs from float regs
    static void Do(const svbool_t& pg0, const svbool_t &pg1, const svbool_t &pg2,
        const svbool_t &pg3, const typename Vec<float>::Type &out0, const typename Vec<float>::Type &out1,
        const typename Vec<float>::Type &out2, const typename Vec<float>::Type &out3, float *dst) noexcept
    {
        svst1_f32(pg0, dst, out0);
        dst += Vector::Length<float>();
        svst1_f32(pg1, dst, out1);
        dst += Vector::Length<float>();
        svst1_f32(pg2, dst, out2);
        dst += Vector::Length<float>();
        svst1_f32(pg3, dst, out3);
    }
};
 
template <>
struct StoreN<__bf16, __bf16, 1> final {
    static void Do(const svbool_t& pg0, const svbool_t &pg1, const svbool_t &pg2,
        const svbool_t &pg3, const typename Vec<__bf16>::Type &out0, const typename Vec<__bf16>::Type &out1,
        const typename Vec<__bf16>::Type &out2, const typename Vec<__bf16>::Type &out3, __bf16 *dst) noexcept
    {
        svst1_bf16(pg0, dst, out0);
        Service::Unused(pg1, out1, pg2, out2, pg3, out3);
    }
};

template <>
struct StoreN<std::int8_t, std::int8_t, 1> final {
    static void Do(const svbool_t& pg0, const svbool_t &pg1, const svbool_t &pg2,
        const svbool_t &pg3, const typename Vec<std::int8_t>::Type &out0,
        const typename Vec<std::int8_t>::Type &out1, const typename Vec<std::int8_t>::Type &out2,
        const typename Vec<std::int8_t>::Type &out3, std::int8_t *dst) noexcept
    {
        svst1_s8(pg0, dst, out0);
        Service::Unused(pg1, out1, pg2, out2, pg3, out3);
    }
};

template <>
struct StoreN<std::uint8_t, std::uint8_t, 1> final {
    static void Do(const svbool_t& pg0, const svbool_t &pg1, const svbool_t &pg2,
        const svbool_t &pg3, const typename Vec<std::uint8_t>::Type &out0,
        const typename Vec<std::uint8_t>::Type &out1, const typename Vec<std::uint8_t>::Type &out2,
        const typename Vec<std::uint8_t>::Type &out3, std::uint8_t *dst) noexcept
    {
        svst1_u8(pg0, dst, out0);
        Service::Unused(pg1, out1, pg2, out2, pg3, out3);
    }
};

template <>
struct StoreN<std::int32_t, std::int32_t, 1> final {
    static void Do(const svbool_t& pg0, const svbool_t &pg1, const svbool_t &pg2,
        const svbool_t &pg3, const typename Vec<std::int32_t>::Type &out0,
        const typename Vec<std::int32_t>::Type &out1, const typename Vec<std::int32_t>::Type &out2,
        const typename Vec<std::int32_t>::Type &out3, std::int32_t *dst) noexcept
    {
        svst1_s32(pg0, dst, out0);
        Service::Unused(pg1, out1, pg2, out2, pg3, out3);
    }
};

template <>
struct StoreN<std::int32_t, std::int32_t, 2> final { // 2 means that store two time for one src
    static void Do(const svbool_t& pg0, const svbool_t &pg1, const svbool_t &pg2,
        const svbool_t &pg3, const typename Vec<std::int32_t>::Type &out0,
        const typename Vec<std::int32_t>::Type &out1, const typename Vec<std::int32_t>::Type &out2,
        const typename Vec<std::int32_t>::Type &out3, std::int32_t *dst) noexcept
    {
        svst1_s32(pg0, dst, out0);
        svst1_s32(pg1, dst + Vector::Length<std::int32_t>(), out1);
        Service::Unused(pg2, out2, pg3, out3);
    }
};

template <>
struct StoreN<std::int32_t, std::int32_t, 4> final { // 4 means that store four time for one src
    static void Do(const svbool_t& pg0, const svbool_t &pg1, const svbool_t &pg2,
        const svbool_t &pg3, const typename Vec<std::int32_t>::Type &out0,
        const typename Vec<std::int32_t>::Type &out1, const typename Vec<std::int32_t>::Type &out2,
        const typename Vec<std::int32_t>::Type &out3, std::int32_t *dst) noexcept
    {
        svst1_s32(pg0, dst, out0);
        dst += Vector::Length<std::int32_t>();
        svst1_s32(pg1, dst, out1);
        dst += Vector::Length<std::int32_t>();
        svst1_s32(pg2, dst, out2);
        dst += Vector::Length<std::int32_t>();
        svst1_s32(pg3, dst, out3);
    }
};

template <>
struct StoreN<__bf16, float, 1> final {
    static void Do(const svbool_t& pg0, const svbool_t &pg1, const svbool_t &pg2,
        const svbool_t &pg3, const typename Vec<float>::Type &out0, const typename Vec<float>::Type &out1,
        const typename Vec<float>::Type &out2, const typename Vec<float>::Type &out3, __bf16 *dst) noexcept
    {
        svbool_t pgTrue = svptrue_b32();
        svuint32_t uout0 = svreinterpret_u32_f32(out0);
        svuint32_t uout1 = svreinterpret_u32_f32(out1);
        uout0 = svadd_x(pgTrue, uout0, (1u << 15)); // 15 shift gives the bit after the last significant in bf16
        uout1 = svadd_x(pgTrue, uout1, (1u << 15)); // 15 shift gives the bit after the last significant in bf16
        svbfloat16_t tmp0 = svreinterpret_bf16_u32(uout0);
        svbfloat16_t tmp1 = svreinterpret_bf16_u32(uout1);
        svbfloat16_t res = svuzp2(tmp0, tmp1);
        svst1_bf16(pg0, dst, res);
        Service::Unused(pg1, pg2, out2, pg3, out3);
    }
};

template <>
struct StoreN<__bf16, float, 2> final { // 2 means that store two __bf16 regs from float regs
    static void Do(const svbool_t& pg0, const svbool_t &pg1, const svbool_t &pg2,
        const svbool_t &pg3, const typename Vec<float>::Type &out0, const typename Vec<float>::Type &out1,
        const typename Vec<float>::Type &out2, const typename Vec<float>::Type &out3, __bf16 *dst) noexcept
    {
        svbool_t pgTrue = svptrue_b32();
        {
            svuint32_t uout0 = svreinterpret_u32_f32(out0);
            svuint32_t uout1 = svreinterpret_u32_f32(out1);
            uout0 = svadd_x(pgTrue, uout0, (1u << 15)); // 15 shift gives the bit after the last significant in bf16
            uout1 = svadd_x(pgTrue, uout1, (1u << 15)); // 15 shift gives the bit after the last significant in bf16
            svbfloat16_t tmp0 = svreinterpret_bf16_u32(uout0);
            svbfloat16_t tmp1 = svreinterpret_bf16_u32(uout1);
            svbfloat16_t res = svuzp2(tmp0, tmp1);
            svst1_bf16(pg0, dst, res);
        }
        {
            svuint32_t uout0 = svreinterpret_u32_f32(out2);
            svuint32_t uout1 = svreinterpret_u32_f32(out3);
            uout0 = svadd_x(pgTrue, uout0, (1u << 15)); // 15 shift gives the bit after the last significant in bf16
            uout1 = svadd_x(pgTrue, uout1, (1u << 15)); // 15 shift gives the bit after the last significant in bf16
            svbfloat16_t tmp0 = svreinterpret_bf16_u32(uout0);
            svbfloat16_t tmp1 = svreinterpret_bf16_u32(uout1);
            svbfloat16_t res = svuzp2(tmp0, tmp1);
            svst1_bf16(pg1, dst + Vector::Length<__bf16>(), res);
        }
        Service::Unused(pg2, pg3);
    }
};

template <>
struct StoreN<__fp16, float, 1> final {
    static void Do(const svbool_t& pg0, const svbool_t &pg1, const svbool_t &pg2,
        const svbool_t &pg3, const typename Vec<float>::Type &out0, const typename Vec<float>::Type &out1,
        const typename Vec<float>::Type &out2, const typename Vec<float>::Type &out3, __fp16 *dst) noexcept
    {
        svbool_t pgFalse = svpfalse_b();
        svbool_t pgEven = svzip1_b16(pg0, pgFalse);
        svbool_t pgOdd = svzip2_b16(pg0, pgFalse);
        svfloat16_t vec0 = svcvt_f16_f32_x(pgEven, out0);
        svfloat16_t vec1 = svcvt_f16_f32_x(pgOdd, out1);
        svfloat16_t res = svuzp1_f16(vec0, vec1);
        svst1_f16(pg0, dst, res);
        Service::Unused(pg1, pg2, out2, pg3, out3);
    }
};

template <>
struct StoreN<__fp16, float, 2> final { // 2 means that store two __fp16 regs from float regs
    static void Do(const svbool_t& pg0, const svbool_t &pg1, const svbool_t &pg2,
        const svbool_t &pg3, const typename Vec<float>::Type &out0, const typename Vec<float>::Type &out1,
        const typename Vec<float>::Type &out2, const typename Vec<float>::Type &out3, __fp16 *dst) noexcept
    {
        svbool_t pgFalse = svpfalse_b();
        {
            svbool_t pgEven = svzip1_b16(pg0, pgFalse);
            svbool_t pgOdd = svzip2_b16(pg0, pgFalse);
            svfloat16_t vec0 = svcvt_f16_f32_x(pgEven, out0);
            svfloat16_t vec1 = svcvt_f16_f32_x(pgOdd, out1);
            svfloat16_t res = svuzp1_f16(vec0, vec1);
            svst1_f16(pg0, dst, res);
        }
        {
            svbool_t pgEven = svzip1_b16(pg1, pgFalse);
            svbool_t pgOdd = svzip2_b16(pg1, pgFalse);
            svfloat16_t vec0 = svcvt_f16_f32_x(pgEven, out2);
            svfloat16_t vec1 = svcvt_f16_f32_x(pgOdd, out3);
            svfloat16_t res = svuzp1_f16(vec0, vec1);
            svst1_f16(pg1, dst + Vector::Length<__fp16>(), res);
        }
        Service::Unused(pg2, pg3);
    }
};

template <>
struct StoreN<std::int8_t, float, 1> final {
    static void Do(const svbool_t& pg0, const svbool_t &pg1, const svbool_t &pg2,
        const svbool_t &pg3, const typename Vec<float>::Type &out0, const typename Vec<float>::Type &out1,
        const typename Vec<float>::Type &out2, const typename Vec<float>::Type &out3, std::int8_t *dst) noexcept
    {
        svbool_t pgtrue = svptrue_b32();
        svint32_t stmp0 = svcvt_s32_x(pgtrue, svrinti_f32_x(pgtrue, out0));
        stmp0 = svmax_n_s32_x(pgtrue, stmp0, std::numeric_limits<std::int8_t>::min());
        stmp0 = svmin_n_s32_x(pgtrue, stmp0, std::numeric_limits<std::int8_t>::max());
        svint32_t stmp1 = svcvt_s32_x(pgtrue, svrinti_f32_x(pgtrue, out1));
        stmp1 = svmax_n_s32_x(pgtrue, stmp1, std::numeric_limits<std::int8_t>::min());
        stmp1 = svmin_n_s32_x(pgtrue, stmp1, std::numeric_limits<std::int8_t>::max());
        svint32_t stmp2 = svcvt_s32_x(pgtrue, svrinti_f32_x(pgtrue, out2));
        stmp2 = svmax_n_s32_x(pgtrue, stmp2, std::numeric_limits<std::int8_t>::min());
        stmp2 = svmin_n_s32_x(pgtrue, stmp2, std::numeric_limits<std::int8_t>::max());
        svint32_t stmp3 = svcvt_s32_x(pgtrue, svrinti_f32_x(pgtrue, out3));
        stmp3 = svmax_n_s32_x(pgtrue, stmp3, std::numeric_limits<std::int8_t>::min());
        stmp3 = svmin_n_s32_x(pgtrue, stmp3, std::numeric_limits<std::int8_t>::max());

        svint32_t t0 = svuzp1_s32(stmp0, stmp1);
        svint32_t t1 = svuzp2_s32(stmp0, stmp1);
        svint32_t t2 = svuzp1_s32(stmp2, stmp3);
        svint32_t t3 = svuzp2_s32(stmp2, stmp3);

        stmp0 = svuzp1_s32(t0, t2);
        stmp1 = svuzp1_s32(t1, t3);
        stmp2 = svuzp2_s32(t0, t2);
        stmp3 = svuzp2_s32(t1, t3);

        stmp0 = svlsl_n_s32_x(pgtrue, stmp0, 24); // 24 bits is a size of 3 std::int8_t elements
        svuint32_t tmp0 = svreinterpret_u32(stmp0);
        stmp1 = svlsl_n_s32_x(pgtrue, stmp1, 24); // 24 bits is a size of 3 std::int8_t elements
        svuint32_t tmp1 = svreinterpret_u32(stmp1);
        stmp2 = svlsl_n_s32_x(pgtrue, stmp2, 24); // 24 bits is a size of 3 std::int8_t elements
        svuint32_t tmp2 = svreinterpret_u32(stmp2);
        stmp3 = svlsl_n_s32_x(pgtrue, stmp3, 24); // 24 bits is a size of 3 std::int8_t elements
        svuint32_t tmp3 = svreinterpret_u32(stmp3);

        tmp0 = svlsr_n_u32_x(pgtrue, tmp0, 24); // 24 bits is a size of 3 std::int8_t elements
        tmp1 = svlsr_n_u32_x(pgtrue, tmp1, 16); // 16 bits is a size of 2 std::int8_t elements
        tmp2 = svlsr_n_u32_x(pgtrue, tmp2, 8);  // 8 bits is a size of 1 std::int8_t element

        tmp0 = svorr_u32_x(pgtrue, tmp0, tmp1);
        tmp2 = svorr_u32_x(pgtrue, tmp2, tmp3);
        tmp0 = svorr_u32_x(pgtrue, tmp0, tmp2);

        svint8_t outintreg = svreinterpret_s8(tmp0);

        svst1_s8(pg0, dst, outintreg);
        Service::Unused(pg1, pg2, pg3);
    }
};

template <>
struct StoreN<std::uint8_t, float, 1> final {
    static void Do(const svbool_t& pg0, const svbool_t &pg1, const svbool_t &pg2,
        const svbool_t &pg3, const typename Vec<float>::Type &out0, const typename Vec<float>::Type &out1,
        const typename Vec<float>::Type &out2, const typename Vec<float>::Type &out3, std::uint8_t *dst) noexcept
    {
        svbool_t pgtrue = svptrue_b32();

        svuint32_t stmp0 = svcvt_u32_x(pgtrue, svrinti_f32_x(pgtrue, out0));
        svuint32_t stmp1 = svcvt_u32_x(pgtrue, svrinti_f32_x(pgtrue, out1));
        svuint32_t stmp2 = svcvt_u32_x(pgtrue, svrinti_f32_x(pgtrue, out2));
        svuint32_t stmp3 = svcvt_u32_x(pgtrue, svrinti_f32_x(pgtrue, out3));
        
        stmp0 = svmin_n_u32_x(pgtrue, stmp0, std::numeric_limits<std::uint8_t>::max());
        stmp1 = svmin_n_u32_x(pgtrue, stmp1, std::numeric_limits<std::uint8_t>::max());
        stmp2 = svmin_n_u32_x(pgtrue, stmp2, std::numeric_limits<std::uint8_t>::max());
        stmp3 = svmin_n_u32_x(pgtrue, stmp3, std::numeric_limits<std::uint8_t>::max());

        svuint32_t t0 = svuzp1_u32(stmp0, stmp1);
        svuint32_t t1 = svuzp2_u32(stmp0, stmp1);
        svuint32_t t2 = svuzp1_u32(stmp2, stmp3);
        svuint32_t t3 = svuzp2_u32(stmp2, stmp3);

        stmp0 = svuzp1_u32(t0, t2);
        stmp1 = svuzp1_u32(t1, t3);
        stmp2 = svuzp2_u32(t0, t2);
        stmp3 = svuzp2_u32(t1, t3);

        stmp0 = svlsl_n_u32_x(pgtrue, stmp0, 24); // 24 bits is a size of 3 std::uint8_t elements
        stmp1 = svlsl_n_u32_x(pgtrue, stmp1, 24); // 24 bits is a size of 3 std::uint8_t elements
        stmp2 = svlsl_n_u32_x(pgtrue, stmp2, 24); // 24 bits is a size of 3 std::uint8_t elements
        stmp3 = svlsl_n_u32_x(pgtrue, stmp3, 24); // 24 bits is a size of 3 std::uint8_t elements

        stmp0 = svlsr_n_u32_x(pgtrue, stmp0, 24); // 24 bits is a size of 3 std::uint8_t elements
        stmp1 = svlsr_n_u32_x(pgtrue, stmp1, 16); // 16 bits is a size of 2 std::uint8_t elements
        stmp2 = svlsr_n_u32_x(pgtrue, stmp2, 8);  // 8 bits is a size of 1 std::uint8_t element

        stmp0 = svorr_u32_x(pgtrue, stmp0, stmp1);
        stmp2 = svorr_u32_x(pgtrue, stmp2, stmp3);
        stmp0 = svorr_u32_x(pgtrue, stmp0, stmp2);

        svuint8_t outuintreg = svreinterpret_u8(stmp0);

        svst1_u8(pg0, dst, outuintreg);
        Service::Unused(pg1, pg2, pg3);
    }
};

template <>
struct StoreN<std::int32_t, float, 1> final {
    static void Do(const svbool_t& pg0, const svbool_t &pg1, const svbool_t &pg2,
        const svbool_t &pg3, const typename Vec<float>::Type &out0, const typename Vec<float>::Type &out1,
        const typename Vec<float>::Type &out2, const typename Vec<float>::Type &out3, std::int32_t *dst) noexcept
    {
        svst1_s32(pg0, dst, svcvt_s32_x(pg0, svrinti_f32_x(pg0, out0)));
        Service::Unused(pg1, out1, pg2, out2, pg3, out3);
    }
};

template <>
struct StoreN<std::int32_t, float, 2> final { // 2 means that store two float regs from float regs
    static void Do(const svbool_t& pg0, const svbool_t &pg1, const svbool_t &pg2,
        const svbool_t &pg3, const typename Vec<float>::Type &out0, const typename Vec<float>::Type &out1,
        const typename Vec<float>::Type &out2, const typename Vec<float>::Type &out3, std::int32_t *dst) noexcept
    {
        svst1_s32(pg0, dst, svcvt_s32_x(pg0, svrinti_f32_x(pg0, out0)));
        svst1_s32(pg1, dst + Vector::Length<std::int32_t>(), svcvt_s32_x(pg1, svrinti_f32_x(pg1, out1)));
        Service::Unused(pg2, out2, pg3, out3);
    }
};

template <>
struct StoreN<std::int32_t, float, 4> final { // 4 means that store four float regs from float regs
    static void Do(const svbool_t& pg0, const svbool_t &pg1, const svbool_t &pg2,
        const svbool_t &pg3, const typename Vec<float>::Type &out0, const typename Vec<float>::Type &out1,
        const typename Vec<float>::Type &out2, const typename Vec<float>::Type &out3, std::int32_t *dst) noexcept
    {
        svst1_s32(pg0, dst, svcvt_s32_x(pg0, svrinti_f32_x(pg0, out0)));
        svst1_s32(pg1, dst + Vector::Length<std::int32_t>(), svcvt_s32_x(pg1, svrinti_f32_x(pg1, out1)));
        // 2 means that store from second offset in the float regs
        svst1_s32(pg2, dst + 2 * Vector::Length<std::int32_t>(), svcvt_s32_x(pg2, svrinti_f32_x(pg2, out2)));
        // 3 means that store from third offset in the float regs
        svst1_s32(pg3, dst + 3 * Vector::Length<std::int32_t>(), svcvt_s32_x(pg3, svrinti_f32_x(pg3, out3)));
    }
};

} // Vector
} // Detail
} // KuDNN

#endif // KUDNN_VECTOR_MEM_STORE_COMPTYPE_F32_HPP