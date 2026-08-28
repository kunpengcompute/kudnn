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

#ifndef KUDNN_VECTOR_MEM_STORE_COMPTYPE_F16_HPP
#define KUDNN_VECTOR_MEM_STORE_COMPTYPE_F16_HPP

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

// CompType is __fp16

template <>
struct StoreN<float, __fp16, 2> final { // 2 means that store two float regs from __fp16 regs
    static void Do(const svbool_t& pg0, const svbool_t &pg1, const svbool_t &pg2,
        const svbool_t &pg3, const typename Vec<__fp16>::Type &out0, const typename Vec<__fp16>::Type &out1,
        const typename Vec<__fp16>::Type &out2, const typename Vec<__fp16>::Type &out3, float *dst) noexcept
    {
        svbool_t pgFalse = svpfalse_b();
        svbool_t pgTrue = svptrue_b16();
        svbool_t pgEven = svtrn1_b16(pgTrue, pgFalse);
        svbool_t pgOdd = svtrn2_b16(pgTrue, pgFalse);
        svfloat16_t zeros = svdup_n_f16(0.0f);
        svfloat32_t vec0 = svcvt_f32_f16_x(pgEven, out0);
        svfloat16_t tmp = svtrn2(out0, zeros);
        svfloat32_t vec1 = svcvt_f32_f16_x(pgOdd, tmp);
        svfloat32_t st0 = svzip1_f32(vec0, vec1);
        svfloat32_t st1 = svzip2_f32(vec0, vec1);
        svst1_f32(pg0, dst, st0);
        svst1_f32(pg1, dst + Vector::Length<float>(), st1);
        Service::Unused(out1, pg2, out2, pg3, out3);
    }
};

template <>
struct StoreN<float, __fp16, 4> final { // 4 means that store four float regs from __fp16 regs
    static void Do(const svbool_t& pg0, const svbool_t &pg1, const svbool_t &pg2,
        const svbool_t &pg3, const typename Vec<__fp16>::Type &out0, const typename Vec<__fp16>::Type &out1,
        const typename Vec<__fp16>::Type &out2, const typename Vec<__fp16>::Type &out3, float *dst) noexcept
    {
        svbool_t pgFalse = svpfalse_b();
        svbool_t pgTrue = svptrue_b16();
        svfloat16_t zeros = svdup_n_f16(0.0f);
        svbool_t pgEven = svtrn1_b16(pgTrue, pgFalse);
        svbool_t pgOdd = svtrn2_b16(pgTrue, pgFalse);
        {
            svfloat32_t vec0 = svcvt_f32_f16_x(pgEven, out0);
            svfloat16_t tmp = svtrn2(out0, zeros);
            svfloat32_t vec1 = svcvt_f32_f16_x(pgOdd, tmp);
            svfloat32_t st0 = svzip1_f32(vec0, vec1);
            svfloat32_t st1 = svzip2_f32(vec0, vec1);
            svst1_f32(pg0, dst, st0);
            svst1_f32(pg1, dst + Vector::Length<float>(), st1);
        }
        {
            svfloat32_t vec0 = svcvt_f32_f16_x(pgEven, out1);
            svfloat16_t tmp = svtrn2(out1, zeros);
            svfloat32_t vec1 = svcvt_f32_f16_x(pgOdd, tmp);
            svfloat32_t st0 = svzip1_f32(vec0, vec1);
            svfloat32_t st1 = svzip2_f32(vec0, vec1);
            // 2 means that store from second offset in the float regs
            svst1_f32(pg0, dst + 2 * Vector::Length<float>(), st0);
            // 3 means that store from third offset in the float regs
            svst1_f32(pg1, dst + 3 * Vector::Length<float>(), st1);
        }
        Service::Unused(pg2, out2, pg3, out3);
    }
};

//

template <>
struct StoreN<std::int32_t, __fp16, 2> final { // 2 means that store two std::int32_t regs from __fp16 regs
    static void Do(const svbool_t& pg0, const svbool_t &pg1, const svbool_t &pg2,
        const svbool_t &pg3, const typename Vec<__fp16>::Type &out0, const typename Vec<__fp16>::Type &out1,
        const typename Vec<__fp16>::Type &out2, const typename Vec<__fp16>::Type &out3, std::int32_t *dst) noexcept
    {
        svbool_t pgFalse = svpfalse_b();
        svbool_t pgTrue = svptrue_b16();
        svbool_t pgEven = svtrn1_b16(pgTrue, pgFalse);
        svbool_t pgOdd = svtrn2_b16(pgTrue, pgFalse);
        svfloat16_t zeros = svdup_n_f16(0.0f);
        svfloat32_t vec0 = svcvt_f32_f16_x(pgEven, out0);
        svfloat16_t tmp = svtrn2(out0, zeros);
        svfloat32_t vec1 = svcvt_f32_f16_x(pgOdd, tmp);
        svfloat32_t st0 = svzip1_f32(vec0, vec1);
        svfloat32_t st1 = svzip2_f32(vec0, vec1);
        svst1_s32(pg0, dst, svcvt_s32_x(pg0, svrinti_f32_x(pg0, st0)));
        svst1_s32(pg1, dst + Vector::Length<std::int32_t>(), svcvt_s32_x(pg1, svrinti_f32_x(pg1, st1)));
        Service::Unused(out1, pg2, out2, pg3, out3);
    }
};

template <>
struct StoreN<std::int32_t, __fp16, 4> final { // 4 means that store four std::int32_t regs from __fp16 regs
    static void Do(const svbool_t& pg0, const svbool_t &pg1, const svbool_t &pg2,
        const svbool_t &pg3, const typename Vec<__fp16>::Type &out0, const typename Vec<__fp16>::Type &out1,
        const typename Vec<__fp16>::Type &out2, const typename Vec<__fp16>::Type &out3, std::int32_t *dst) noexcept
    {
        svfloat16_t zeros = svdup_n_f16(0.0f);
        svbool_t pgFalse = svpfalse_b();
        svbool_t pgTrue = svptrue_b16();
        svbool_t pgEven = svtrn1_b16(pgTrue, pgFalse);
        svbool_t pgOdd = svtrn2_b16(pgTrue, pgFalse);
        {
            svfloat32_t vec0 = svcvt_f32_f16_x(pgEven, out0);
            svfloat16_t tmp = svtrn2(out0, zeros);
            svfloat32_t vec1 = svcvt_f32_f16_x(pgOdd, tmp);
            svfloat32_t st0 = svzip1_f32(vec0, vec1);
            svfloat32_t st1 = svzip2_f32(vec0, vec1);
            svst1_s32(pg0, dst, svcvt_s32_x(pg0, svrinti_f32_x(pg0, st0)));
            svst1_s32(pg1, dst + Vector::Length<std::int32_t>(),  svcvt_s32_x(pg1, svrinti_f32_x(pg1, st1)));
        }
        {
            svfloat32_t vec0 = svcvt_f32_f16_x(pgEven, out1);
            svfloat16_t tmp = svtrn2(out1, zeros);
            svfloat32_t vec1 = svcvt_f32_f16_x(pgOdd, tmp);
            svfloat32_t st0 = svzip1_f32(vec0, vec1);
            svfloat32_t st1 = svzip2_f32(vec0, vec1);
            // 2 means that store from second offset in the float regs
            svst1_s32(pg0, dst + 2 * Vector::Length<std::int32_t>(), svcvt_s32_x(pg0, svrinti_f32_x(pg0, st0)));
            // 3 means that store from third offset in the float regs
            svst1_s32(pg1, dst + 3 * Vector::Length<std::int32_t>(), svcvt_s32_x(pg1, svrinti_f32_x(pg1, st1)));
        }
        Service::Unused(pg2, out2, pg3, out3);
    }
};

//

template <>
struct StoreN<__bf16, __fp16, 1> final {
    static void Do(const svbool_t& pg0, const svbool_t &pg1, const svbool_t &pg2,
        const svbool_t &pg3, const typename Vec<__fp16>::Type &out0, const typename Vec<__fp16>::Type &out1,
        const typename Vec<__fp16>::Type &out2, const typename Vec<__fp16>::Type &out3, __bf16 *dst) noexcept
    {
        svbool_t pgFalse = svpfalse_b();
        svbool_t pgTrue = svptrue_b16();
        svbool_t pgEven = svtrn1_b16(pgTrue, pgFalse);
        svbool_t pgOdd = svtrn2_b16(pgTrue, pgFalse);
        svfloat16_t zeros = svdup_n_f16(0.0f);
        svfloat32_t vec0 = svcvt_f32_f16_x(pgEven, out0);
        svfloat16_t tmp = svtrn2(out0, zeros);
        svfloat32_t vec1 = svcvt_f32_f16_x(pgOdd, tmp);
        svfloat32_t st0 = svzip1_f32(vec0, vec1);
        svfloat32_t st1 = svzip2_f32(vec0, vec1);
        svuint32_t uout0 = svreinterpret_u32_f32(st0);
        svuint32_t uout1 = svreinterpret_u32_f32(st1);
        uout0 = svadd_x(pgTrue, uout0, (1u << 15)); // 15 shift gives the bit after the last significant in bf16
        uout1 = svadd_x(pgTrue, uout1, (1u << 15)); // 15 shift gives the bit after the last significant in bf16
        svbfloat16_t tmp1 = svreinterpret_bf16_u32(uout1);
        svbfloat16_t tmp0 = svreinterpret_bf16_u32(uout0);
        svbfloat16_t res = svuzp2(tmp0, tmp1);
        svst1_bf16(pg0, dst, res);
        Service::Unused(pg1, out1, pg2, out2, pg3, out3);
    }
};

template <>
struct StoreN<__bf16, __fp16, 2> final { // 2 means that store two __bf16 regs from __fp16 regs
    static void Do(const svbool_t& pg0, const svbool_t &pg1, const svbool_t &pg2,
        const svbool_t &pg3, const typename Vec<__fp16>::Type &out0, const typename Vec<__fp16>::Type &out1,
        const typename Vec<__fp16>::Type &out2, const typename Vec<__fp16>::Type &out3, __bf16 *dst) noexcept
    {
        svbool_t pgFalse = svpfalse_b();
        svbool_t pgTrue = svptrue_b16();
        svbool_t pgEven = svtrn1_b16(pgTrue, pgFalse);
        svbool_t pgOdd = svtrn2_b16(pgTrue, pgFalse);
        svfloat16_t zeros = svdup_n_f16(0.0f);
        {
            svfloat32_t vec0 = svcvt_f32_f16_x(pgEven, out0);
            svfloat16_t tmp = svtrn2(out0, zeros);
            svfloat32_t vec1 = svcvt_f32_f16_x(pgOdd, tmp);
            svfloat32_t st0 = svzip1_f32(vec0, vec1);
            svfloat32_t st1 = svzip2_f32(vec0, vec1);
            svuint32_t uout0 = svreinterpret_u32_f32(st0);
            svuint32_t uout1 = svreinterpret_u32_f32(st1);
            uout0 = svadd_x(pgTrue, uout0, (1u << 15)); // 15 shift gives the bit after the last significant in bf16
            uout1 = svadd_x(pgTrue, uout1, (1u << 15)); // 15 shift gives the bit after the last significant in bf16
            svbfloat16_t tmp0 = svreinterpret_bf16_u32(uout0);
            svbfloat16_t tmp1 = svreinterpret_bf16_u32(uout1);
            svbfloat16_t res = svuzp2(tmp0, tmp1);
            svst1_bf16(pg0, dst, res);
        }
        {
            svfloat32_t vec0 = svcvt_f32_f16_x(pgEven, out1);
            svfloat16_t tmp = svtrn2(out1, zeros);
            svfloat32_t vec1 = svcvt_f32_f16_x(pgOdd, tmp);
            svfloat32_t st0 = svzip1_f32(vec0, vec1);
            svfloat32_t st1 = svzip2_f32(vec0, vec1);
            svuint32_t uout0 = svreinterpret_u32_f32(st0);
            svuint32_t uout1 = svreinterpret_u32_f32(st1);
            uout0 = svadd_x(pgTrue, uout0, (1u << 15)); // 15 shift gives the bit after the last significant in bf16
            uout1 = svadd_x(pgTrue, uout1, (1u << 15)); // 15 shift gives the bit after the last significant in bf16
            svbfloat16_t tmp0 = svreinterpret_bf16_u32(uout0);
            svbfloat16_t tmp1 = svreinterpret_bf16_u32(uout1);
            svbfloat16_t res = svuzp2(tmp0, tmp1);
            svst1_bf16(pg1, dst + Vector::Length<__bf16>(), res);
        }
        Service::Unused(pg2, out2, pg3, out3);
    }
};

template <>
struct StoreN<__fp16, __fp16, 1> final {
    static void Do(const svbool_t& pg0, const svbool_t &pg1, const svbool_t &pg2,
        const svbool_t &pg3, const typename Vec<__fp16>::Type &out0, const typename Vec<__fp16>::Type &out1,
        const typename Vec<__fp16>::Type &out2, const typename Vec<__fp16>::Type &out3, __fp16 *dst) noexcept
    {
        svst1_f16(pg0, dst, out0);
        Service::Unused(pg1, out1, pg2, out2, pg3, out3);
    }
};

template <>
struct StoreN<__fp16, __fp16, 2> final { // 2 means that store two __fp16 regs from __fp16 regs
    static void Do(const svbool_t& pg0, const svbool_t &pg1, const svbool_t &pg2,
        const svbool_t &pg3, const typename Vec<__fp16>::Type &out0, const typename Vec<__fp16>::Type &out1,
        const typename Vec<__fp16>::Type &out2, const typename Vec<__fp16>::Type &out3, __fp16 *dst) noexcept
    {
        svst1_f16(pg0, dst, out0);
        svst1_f16(pg1, dst + Vector::Length<__fp16>(), out1);
        Service::Unused(pg2, out2, pg3, out3);
    }
};

template <>
struct StoreN<std::int8_t, __fp16, 1> final {
    static void Do(const svbool_t& pg0, const svbool_t &pg1, const svbool_t &pg2,
        const svbool_t &pg3, const typename Vec<__fp16>::Type &out0, const typename Vec<__fp16>::Type &out1,
        const typename Vec<__fp16>::Type &out2, const typename Vec<__fp16>::Type &out3, std::int8_t *dst) noexcept
    {
        svbool_t pgtrue = svptrue_b16();
        svint16_t stmp0 = svcvt_s16_x(pgtrue, svrinti_f16_x(pgtrue, out0));
        stmp0 = svmax_n_s16_x(pgtrue, stmp0, std::numeric_limits<std::int8_t>::min());
        stmp0 = svmin_n_s16_x(pgtrue, stmp0, std::numeric_limits<std::int8_t>::max());
        svint16_t stmp1 = svcvt_s16_x(pgtrue, svrinti_f16_x(pgtrue, out1));
        stmp1 = svmax_n_s16_x(pgtrue, stmp1, std::numeric_limits<std::int8_t>::min());
        stmp1 = svmin_n_s16_x(pgtrue, stmp1, std::numeric_limits<std::int8_t>::max());

        svint16_t t0 = svuzp1_s16(stmp0, stmp1);
        svint16_t t1 = svuzp2_s16(stmp0, stmp1);

        t0 = svlsl_n_s16_x(pgtrue, t0, 8); // 8 bits is a size of 1 std::int8_t element
        svuint16_t tmp0 = svreinterpret_u16(t0);
        t1 = svlsl_n_s16_x(pgtrue, t1, 8); // 8 bits is a size of 1 std::int8_t element
        svuint16_t tmp1 = svreinterpret_u16(t1);

        tmp0 = svlsr_n_u16_x(pgtrue, tmp0, 8); // 8 bits is a size of 1 std::int8_t element

        tmp0 = svorr_u16_x(pgtrue, tmp0, tmp1);

        svint8_t outintreg = svreinterpret_s8(tmp0);

        svst1_s8(pg0, dst, outintreg);
        Service::Unused(pg1, out2, pg2, out3, pg3);
    }
};

template <>
struct StoreN<std::uint8_t, __fp16, 1> final {
    static void Do(const svbool_t& pg0, const svbool_t &pg1, const svbool_t &pg2,
        const svbool_t &pg3, const typename Vec<__fp16>::Type &out0, const typename Vec<__fp16>::Type &out1,
        const typename Vec<__fp16>::Type &out2, const typename Vec<__fp16>::Type &out3, std::uint8_t *dst) noexcept
    {
        svbool_t pgtrue = svptrue_b16();

        svuint16_t stmp0 = svcvt_u16_x(pgtrue, svrinti_f16_x(pgtrue, out0));
        svuint16_t stmp1 = svcvt_u16_x(pgtrue, svrinti_f16_x(pgtrue, out1));
        
        stmp0 = svmin_n_u16_x(pgtrue, stmp0, std::numeric_limits<std::uint8_t>::max());
        stmp1 = svmin_n_u16_x(pgtrue, stmp1, std::numeric_limits<std::uint8_t>::max());

        svuint16_t t0 = svuzp1_u16(stmp0, stmp1);
        svuint16_t t1 = svuzp2_u16(stmp0, stmp1);

        t0 = svlsl_n_u16_x(pgtrue, t0, 8); // 8 bits is a size of 1 std::uint8_t elements
        t1 = svlsl_n_u16_x(pgtrue, t1, 8); // 8 bits is a size of 1 std::uint8_t elements

        t0 = svlsr_n_u16_x(pgtrue, t0, 8); // 8 bits is a size of 1 std::uint8_t elements
        
        t0 = svorr_u16_x(pgtrue, t0, t1);

        svuint8_t outintreg = svreinterpret_u8(stmp0);

        svst1_u8(pg0, dst, outintreg);
        Service::Unused(pg1, out2, pg2, out3, pg3);
    }
};

} // Vector
} // Detail
} // KuDNN

#endif // KUDNN_VECTOR_MEM_STORE_COMPTYPE_F16_HPP