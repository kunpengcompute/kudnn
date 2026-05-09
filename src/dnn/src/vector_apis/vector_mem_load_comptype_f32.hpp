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

#ifndef KUDNN_VECTOR_MEM_LOAD_COMPTYPE_F32_HPP
#define KUDNN_VECTOR_MEM_LOAD_COMPTYPE_F32_HPP

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
struct LoadN<float, float, 1> final {
    static void Do(const float *src, const svbool_t &pg0, const svbool_t &pg1, const svbool_t &pg2, const svbool_t &pg3,
                   typename Vec<float>::Type &in0, typename Vec<float>::Type &in1, typename Vec<float>::Type &in2,
                   typename Vec<float>::Type &in3) noexcept
    {
        in0 = svld1_f32(pg0, src);
        Service::Unused(pg1, in1, pg2, in2, pg3, in3);
    }
};

template <>
struct LoadN<float, float, 2> final { // 2 means that load two float regs into float regs
    static void Do(const float *src, const svbool_t &pg0, const svbool_t &pg1, const svbool_t &pg2, const svbool_t &pg3,
                   typename Vec<float>::Type &in0, typename Vec<float>::Type &in1, typename Vec<float>::Type &in2,
                   typename Vec<float>::Type &in3) noexcept
    {
        in0 = svld1_f32(pg0, src);
        in1 = svld1_f32(pg1, src + Vector::Length<float>());
        Service::Unused(pg2, in2, pg3, in3);
    }
};

template <>
struct LoadN<float, float, 4> final { // 4 means that load four float regs into float regs
    static void Do(const float *src, const svbool_t &pg0, const svbool_t &pg1, const svbool_t &pg2, const svbool_t &pg3,
                   typename Vec<float>::Type &in0, typename Vec<float>::Type &in1, typename Vec<float>::Type &in2,
                   typename Vec<float>::Type &in3) noexcept
    {
        in0 = svld1_f32(pg0, src);
        src += Vector::Length<float>();
        in1 = svld1_f32(pg1, src);
        src += Vector::Length<float>();
        in2 = svld1_f32(pg2, src);
        src += Vector::Length<float>();
        in3 = svld1_f32(pg3, src);
    }
};

template <>
struct LoadN<__bf16, __bf16, 1> final {
    static void Do(const __bf16 *src, const svbool_t &pg0, const svbool_t &pg1, const svbool_t &pg2,
                   const svbool_t &pg3, typename Vec<__bf16>::Type &in0, typename Vec<__bf16>::Type &in1,
                   typename Vec<__bf16>::Type &in2, typename Vec<__bf16>::Type &in3) noexcept
    {
        in0 = svld1_bf16(pg0, src);
        Service::Unused(pg1, in1, pg2, in2, pg3, in3);
    }
};

template <>
struct LoadN<std::int8_t, std::int8_t, 1> final {
    static void Do(const std::int8_t *src, const svbool_t &pg0, const svbool_t &pg1, const svbool_t &pg2,
                   const svbool_t &pg3, typename Vec<std::int8_t>::Type &in0, typename Vec<std::int8_t>::Type &in1,
                   typename Vec<std::int8_t>::Type &in2, typename Vec<std::int8_t>::Type &in3) noexcept
    {
        in0 = svld1_s8(pg0, src);
        Service::Unused(pg1, in1, pg2, in2, pg3, in3);
    }
};

template <>
struct LoadN<std::uint8_t, std::uint8_t, 1> final {
    static void Do(const std::uint8_t *src, const svbool_t &pg0, const svbool_t &pg1, const svbool_t &pg2,
                   const svbool_t &pg3, typename Vec<std::uint8_t>::Type &in0, typename Vec<std::uint8_t>::Type &in1,
                   typename Vec<std::uint8_t>::Type &in2, typename Vec<std::uint8_t>::Type &in3) noexcept
    {
        in0 = svld1_u8(pg0, src);
        Service::Unused(pg1, in1, pg2, in2, pg3, in3);
    }
};

template <>
struct LoadN<std::int32_t, std::int32_t, 1> final {
    static void Do(const std::int32_t *src, const svbool_t &pg0, const svbool_t &pg1, const svbool_t &pg2,
                   const svbool_t &pg3, typename Vec<std::int32_t>::Type &in0, typename Vec<std::int32_t>::Type &in1,
                   typename Vec<std::int32_t>::Type &in2, typename Vec<std::int32_t>::Type &in3) noexcept
    {
        in0 = svld1_s32(pg0, src);
        Service::Unused(pg1, in1, pg2, in2, pg3, in3);
    }
};

template <>
struct LoadN<__bf16, float, 1> final {
    static void Do(const __bf16 *src, const svbool_t &pg0, const svbool_t &pg1, const svbool_t &pg2,
                   const svbool_t &pg3, typename Vec<float>::Type &in0, typename Vec<float>::Type &in1,
                   typename Vec<float>::Type &in2, typename Vec<float>::Type &in3) noexcept
    {
        svbfloat16_t tmp = svld1_bf16(pg0, src);
        svbfloat16_t zeros = svdup_n_bf16(vcvth_bf16_f32(0.0f));
        svbfloat16_t tmp0 = svzip1(zeros, tmp);
        svbfloat16_t tmp1 = svzip2(zeros, tmp);
        in0 = svreinterpret_f32_bf16(tmp0);
        in1 = svreinterpret_f32_bf16(tmp1);
        Service::Unused(pg1, pg2, in2, pg3, in3);
    }
};

template <>
struct LoadN<__bf16, float, 2> final { // 2 means that load __bf16 float regs into float regs
    static void Do(const __bf16 *src, const svbool_t &pg0, const svbool_t &pg1, const svbool_t &pg2,
                   const svbool_t &pg3, typename Vec<float>::Type &in0, typename Vec<float>::Type &in1,
                   typename Vec<float>::Type &in2, typename Vec<float>::Type &in3) noexcept
    {
        svbfloat16_t zeros = svdup_n_bf16(vcvth_bf16_f32(0.0f));
        {
            svbfloat16_t tmp = svld1_bf16(pg0, src);
            svbfloat16_t tmp0 = svzip1(zeros, tmp);
            svbfloat16_t tmp1 = svzip2(zeros, tmp);
            in0 = svreinterpret_f32_bf16(tmp0);
            in1 = svreinterpret_f32_bf16(tmp1);
        }
        {
            svbfloat16_t tmp = svld1_bf16(pg1, src + Vector::Length<__bf16>());
            svbfloat16_t tmp0 = svzip1(zeros, tmp);
            svbfloat16_t tmp1 = svzip2(zeros, tmp);
            in2 = svreinterpret_f32_bf16(tmp0);
            in3 = svreinterpret_f32_bf16(tmp1);
        }
        Service::Unused(pg2, pg3);
    }
};

template <>
struct LoadN<__fp16, float, 1> final {
    static void Do(const __fp16 *src, const svbool_t &pg0, const svbool_t &pg1, const svbool_t &pg2,
                   const svbool_t &pg3, typename Vec<float>::Type &in0, typename Vec<float>::Type &in1,
                   typename Vec<float>::Type &in2, typename Vec<float>::Type &in3) noexcept
    {
        svbool_t pgFalse = svpfalse_b();
        svbool_t pgEven = svtrn1_b16(pg0, pgFalse);
        svbool_t pgOdd = svtrn2_b16(pg0, pgFalse);
        svfloat16_t tmp = svld1_f16(pg0, src);
        svfloat16_t zeros = svdup_n_f16(0.0f);
        svfloat32_t vec0 = svcvt_f32_f16_x(pgEven, tmp);
        svfloat16_t tmp1 = svtrn2(tmp, zeros);
        svfloat32_t vec1 = svcvt_f32_f16_x(pgOdd, tmp1);
        in0 = svzip1_f32(vec0, vec1);
        in1 = svzip2_f32(vec0, vec1);
        Service::Unused(pg1, pg2, in2, pg3, in3);
    }
};

template <>
struct LoadN<__fp16, float, 2> final { // 2 means that load two __fp16 regs into float regs
    static void Do(const __fp16 *src, const svbool_t &pg0, const svbool_t &pg1, const svbool_t &pg2,
                   const svbool_t &pg3, typename Vec<float>::Type &in0, typename Vec<float>::Type &in1,
                   typename Vec<float>::Type &in2, typename Vec<float>::Type &in3) noexcept
    {
        svbool_t pgFalse = svpfalse_b();
        svfloat16_t zeros = svdup_n_f16(0.0f);
        {
            svbool_t pgEven = svtrn1_b16(pg0, pgFalse);
            svbool_t pgOdd = svtrn2_b16(pg0, pgFalse);
            svfloat16_t tmp = svld1_f16(pg0, src);
            svfloat32_t vec0 = svcvt_f32_f16_x(pgEven, tmp);
            svfloat16_t tmp1 = svtrn2(tmp, zeros);
            svfloat32_t vec1 = svcvt_f32_f16_x(pgOdd, tmp1);
            in0 = svzip1_f32(vec0, vec1);
            in1 = svzip2_f32(vec0, vec1);
        }
        {
            svbool_t pgEven = svtrn1_b16(pg1, pgFalse);
            svbool_t pgOdd = svtrn2_b16(pg1, pgFalse);
            svfloat16_t tmp = svld1_f16(pg1, src + Vector::Length<__fp16>());
            svfloat32_t vec0 = svcvt_f32_f16_x(pgEven, tmp);
            svfloat16_t tmp1 = svtrn2(tmp, zeros);
            svfloat32_t vec1 = svcvt_f32_f16_x(pgOdd, tmp1);
            in2 = svzip1_f32(vec0, vec1);
            in3 = svzip2_f32(vec0, vec1);
        }
        Service::Unused(pg2, pg3);
    }
};

template <>
struct LoadN<std::int8_t, float, 1> final {
    static void Do(const std::int8_t *src, const svbool_t &pg0, const svbool_t &pg1, const svbool_t &pg2,
                   const svbool_t &pg3, typename Vec<float>::Type &in0, typename Vec<float>::Type &in1,
                   typename Vec<float>::Type &in2, typename Vec<float>::Type &in3) noexcept
    {
        svint8_t inreg = svld1_s8(pg0, src);
        svuint32_t inreg32 = svreinterpret_u32(inreg);

        svbool_t pgtrue = svptrue_b32();

        svuint32_t tmp0 = svand_n_u32_x(pgtrue, inreg32, 0xff000000u);
        svuint32_t tmp1 = svand_n_u32_x(pgtrue, inreg32, 0x00ff0000u);
        tmp1 = svlsl_n_u32_x(pgtrue, tmp1, 8); // 8 bits is a size of 1 std::int8_t element
        svuint32_t tmp2 = svand_n_u32_x(pgtrue, inreg32, 0x0000ff00u);
        tmp2 = svlsl_n_u32_x(pgtrue, tmp2, 16); // 16 bits is a size of 2 std::int8_t elements
        svuint32_t tmp3 = svand_n_u32_x(pgtrue, inreg32, 0x000000ffu);
        tmp3 = svlsl_n_u32_x(pgtrue, tmp3, 24); // 24 bits is a size of 3 std::int8_t elements
        svint32_t stmp0 = svreinterpret_s32(tmp0);
        stmp0 = svasr_n_s32_x(pgtrue, stmp0, 24); // 24 bits is a size of 3 std::int8_t elements
        svint32_t stmp1 = svreinterpret_s32(tmp1);
        stmp1 = svasr_n_s32_x(pgtrue, stmp1, 24); // 24 bits is a size of 3 std::int8_t elements
        svint32_t stmp2 = svreinterpret_s32(tmp2);
        stmp2 = svasr_n_s32_x(pgtrue, stmp2, 24); // 24 bits is a size of 3 std::int8_t elements
        svint32_t stmp3 = svreinterpret_s32(tmp3);
        stmp3 = svasr_n_s32_x(pgtrue, stmp3, 24); // 24 bits is a size of 3 std::int8_t elements

        svint32_t t0 = svzip1_s32(stmp3, stmp1);
        svint32_t t1 = svzip2_s32(stmp3, stmp1);
        svint32_t t2 = svzip1_s32(stmp2, stmp0);
        svint32_t t3 = svzip2_s32(stmp2, stmp0);

        stmp0 = svzip1_s32(t0, t2);
        stmp1 = svzip2_s32(t0, t2);
        stmp2 = svzip1_s32(t1, t3);
        stmp3 = svzip2_s32(t1, t3);

        in0 = svcvt_f32_x(pgtrue, stmp0);
        in1 = svcvt_f32_x(pgtrue, stmp1);
        in2 = svcvt_f32_x(pgtrue, stmp2);
        in3 = svcvt_f32_x(pgtrue, stmp3);
        Service::Unused(pg1, pg2, pg3);
    }
};

template <>
struct LoadN<std::uint8_t, float, 1> final {
    static void Do(const std::uint8_t *src, const svbool_t &pg0, const svbool_t &pg1, const svbool_t &pg2,
                   const svbool_t &pg3, typename Vec<float>::Type &in0, typename Vec<float>::Type &in1,
                   typename Vec<float>::Type &in2, typename Vec<float>::Type &in3) noexcept
    {
        svuint8_t inreg = svld1_u8(pg0, src);
        svuint32_t inreg32 = svreinterpret_u32(inreg);

        svbool_t pgtrue = svptrue_b32();

        svuint32_t tmp0 = svand_n_u32_x(pgtrue, inreg32, 0xff000000u);
        svuint32_t tmp1 = svand_n_u32_x(pgtrue, inreg32, 0x00ff0000u);
        svuint32_t tmp2 = svand_n_u32_x(pgtrue, inreg32, 0x0000ff00u);
        svuint32_t tmp3 = svand_n_u32_x(pgtrue, inreg32, 0x000000ffu);

        tmp1 = svlsl_n_u32_x(pgtrue, tmp1, 8);  // 8 bits is a size of 1 std::uint8_t element
        tmp2 = svlsl_n_u32_x(pgtrue, tmp2, 16); // 16 bits is a size of 2 std::uint8_t elements
        tmp3 = svlsl_n_u32_x(pgtrue, tmp3, 24); // 24 bits is a size of 3 std::uint8_t elements

        tmp0 = svlsr_n_u32_x(pgtrue, tmp0, 24); // 24 bits is a size of 3 std::uint8_t elements
        tmp1 = svlsr_n_u32_x(pgtrue, tmp1, 24); // 24 bits is a size of 3 std::uint8_t elements
        tmp2 = svlsr_n_u32_x(pgtrue, tmp2, 24); // 24 bits is a size of 3 std::uint8_t elements
        tmp3 = svlsr_n_u32_x(pgtrue, tmp3, 24); // 24 bits is a size of 3 std::uint8_t elements

        svuint32_t t0 = svzip1_u32(tmp3, tmp1);
        svuint32_t t1 = svzip2_u32(tmp3, tmp1);
        svuint32_t t2 = svzip1_u32(tmp2, tmp0);
        svuint32_t t3 = svzip2_u32(tmp2, tmp0);

        tmp0 = svzip1_u32(t0, t2);
        tmp1 = svzip2_u32(t0, t2);
        tmp2 = svzip1_u32(t1, t3);
        tmp3 = svzip2_u32(t1, t3);

        in0 = svcvt_f32_x(pgtrue, tmp0);
        in1 = svcvt_f32_x(pgtrue, tmp1);
        in2 = svcvt_f32_x(pgtrue, tmp2);
        in3 = svcvt_f32_x(pgtrue, tmp3);
        Service::Unused(pg1, pg2, pg3);
    }
};

template <>
struct LoadN<std::int32_t, float, 1> final {
    static void Do(const std::int32_t *src, const svbool_t &pg0, const svbool_t &pg1, const svbool_t &pg2,
                   const svbool_t &pg3, typename Vec<float>::Type &in0, typename Vec<float>::Type &in1,
                   typename Vec<float>::Type &in2, typename Vec<float>::Type &in3) noexcept
    {
        in0 = svcvt_f32_x(pg0, svld1_s32(pg0, src));
        Service::Unused(pg1, in1, pg2, in2, pg3, in3);
    }
};

template <>
struct LoadN<std::int32_t, float, 2> final { // 2 means that load two float regs into float regs
    static void Do(const std::int32_t *src, const svbool_t &pg0, const svbool_t &pg1, const svbool_t &pg2,
                   const svbool_t &pg3, typename Vec<float>::Type &in0, typename Vec<float>::Type &in1,
                   typename Vec<float>::Type &in2, typename Vec<float>::Type &in3) noexcept
    {
        in0 = svcvt_f32_x(pg0, svld1_s32(pg0, src));
        src += Vector::Length<std::int32_t>();
        in1 = svcvt_f32_x(pg1, svld1_s32(pg1, src));
        Service::Unused(pg2, in2, pg3, in3);
    }
};

template <>
struct LoadN<std::int32_t, float, 4> final { // 4 means that load four float regs into float regs
    static void Do(const std::int32_t *src, const svbool_t &pg0, const svbool_t &pg1, const svbool_t &pg2,
                   const svbool_t &pg3, typename Vec<float>::Type &in0, typename Vec<float>::Type &in1,
                   typename Vec<float>::Type &in2, typename Vec<float>::Type &in3) noexcept
    {
        in0 = svcvt_f32_x(pg0, svld1_s32(pg0, src));
        src += Vector::Length<std::int32_t>();
        in1 = svcvt_f32_x(pg1, svld1_s32(pg1, src));
        src += Vector::Length<std::int32_t>();
        in2 = svcvt_f32_x(pg2, svld1_s32(pg2, src));
        src += Vector::Length<std::int32_t>();
        in3 = svcvt_f32_x(pg3, svld1_s32(pg3, src));
    }
};

} // namespace Vector
} // namespace Detail
} // namespace KuDNN

#endif // KUDNN_VECTOR_MEM_LOAD_COMPTYPE_F32_HPP
