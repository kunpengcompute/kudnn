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

#ifndef KUDNN_VECTOR_MEM_LOAD_COMPTYPE_F16_HPP
#define KUDNN_VECTOR_MEM_LOAD_COMPTYPE_F16_HPP

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

// CompType is f16

template <>
struct LoadN<float, __fp16, 2> final { // 2 means that load two float regs into __fp16 regs
    static void Do(const float *src,
        const svbool_t& pg0, const svbool_t &pg1, const svbool_t &pg2, const svbool_t &pg3,
        typename Vec<__fp16>::Type &in0, typename Vec<__fp16>::Type &in1,
        typename Vec<__fp16>::Type &in2, typename Vec<__fp16>::Type &in3) noexcept
    {
        svfloat32_t src0 = svld1_f32(pg0, src);
        svfloat32_t src1 = svld1_f32(pg1, src + Vector::Length<float>());
        svfloat16_t vec0 = svcvt_f16_f32_x(pg0, src0);
        svfloat16_t vec1 = svcvt_f16_f32_x(pg1, src1);
        in0 = svuzp1_f16(vec0, vec1);
        Service::Unused(in1, pg2, in2, pg3, in3);
    }
};

template <>
struct LoadN<float, __fp16, 4> final { // 4 means that load four float regs into __fp16 regs
    static void Do(const float *src,
        const svbool_t& pg0, const svbool_t &pg1, const svbool_t &pg2, const svbool_t &pg3,
        typename Vec<__fp16>::Type &in0, typename Vec<__fp16>::Type &in1,
        typename Vec<__fp16>::Type &in2, typename Vec<__fp16>::Type &in3) noexcept
    {
        {
            svfloat32_t src0 = svld1_f32(pg0, src);
            svfloat32_t src1 = svld1_f32(pg1, src + Vector::Length<float>());
            svfloat16_t vec0 = svcvt_f16_f32_x(pg0, src0);
            svfloat16_t vec1 = svcvt_f16_f32_x(pg1, src1);
            in0 = svuzp1_f16(vec0, vec1);
        }
        {
            svfloat32_t src2 = svld1_f32(pg2, src + 2 * Vector::Length<float>());
            svfloat32_t src3 = svld1_f32(pg3, src + 3 * Vector::Length<float>());
            svfloat16_t vec2 = svcvt_f16_f32_x(pg2, src2);
            svfloat16_t vec3 = svcvt_f16_f32_x(pg3, src3);
            in1 = svuzp1_f16(vec2, vec3);
        }
        Service::Unused(in2, in3);
    }
};

template <>
struct LoadN<std::int32_t, __fp16, 2> final { // 2 means that load two float regs into __fp16 regs
    static void Do(const std::int32_t *src,
        const svbool_t& pg0, const svbool_t &pg1, const svbool_t &pg2, const svbool_t &pg3,
        typename Vec<__fp16>::Type &in0, typename Vec<__fp16>::Type &in1,
        typename Vec<__fp16>::Type &in2, typename Vec<__fp16>::Type &in3) noexcept
    {
        svfloat32_t src0 = svcvt_f32_x(pg0, svld1_s32(pg0, src));
        svfloat32_t src1 = svcvt_f32_x(pg1, svld1_s32(pg1, src + Vector::Length<std::int32_t>()));
        svfloat16_t vec0 = svcvt_f16_f32_x(pg0, src0);
        svfloat16_t vec1 = svcvt_f16_f32_x(pg1, src1);
        in0 = svuzp1_f16(vec0, vec1);
        Service::Unused(in1, pg2, in2, pg3, in3);
    }
};

template <>
struct LoadN<std::int32_t, __fp16, 4> final { // 4 means that load four float regs into __fp16 regs
    static void Do(const std::int32_t *src,
        const svbool_t& pg0, const svbool_t &pg1, const svbool_t &pg2, const svbool_t &pg3,
        typename Vec<__fp16>::Type &in0, typename Vec<__fp16>::Type &in1,
        typename Vec<__fp16>::Type &in2, typename Vec<__fp16>::Type &in3) noexcept
    {
        {
            svfloat32_t src0 = svcvt_f32_x(pg0, svld1_s32(pg0, src));
            svfloat32_t src1 = svcvt_f32_x(pg1, svld1_s32(pg1, src + Vector::Length<std::int32_t>()));
            svfloat16_t vec0 = svcvt_f16_f32_x(pg0, src0);
            svfloat16_t vec1 = svcvt_f16_f32_x(pg1, src1);
            in0 = svuzp1_f16(vec0, vec1);
        }
        {
            svfloat32_t src2 = svcvt_f32_x(pg2, svld1_s32(pg2, src + 2 * Vector::Length<std::int32_t>()));
            svfloat32_t src3 = svcvt_f32_x(pg3, svld1_s32(pg3, src + 3 * Vector::Length<std::int32_t>()));
            svfloat16_t vec2 = svcvt_f16_f32_x(pg2, src2);
            svfloat16_t vec3 = svcvt_f16_f32_x(pg3, src3);
            in1 = svuzp1_f16(vec2, vec3);
        }
        Service::Unused(in2, in3);
    }
};

template <>
struct LoadN<__bf16, __fp16, 1> final {
    static void Do(const __bf16 *src,
        const svbool_t& pg0, const svbool_t &pg1, const svbool_t &pg2, const svbool_t &pg3,
        typename Vec<__fp16>::Type &in0, typename Vec<__fp16>::Type &in1,
        typename Vec<__fp16>::Type &in2, typename Vec<__fp16>::Type &in3) noexcept
    {
        svbool_t pgTrue = svptrue_b32();
        svbfloat16_t tmp = svld1_bf16(pg0, src);
        svbfloat16_t zeros = svdup_n_bf16(vcvth_bf16_f32(0.0f));
        svbfloat16_t tmp0 = svzip1(zeros, tmp);
        svbfloat16_t tmp1 = svzip2(zeros, tmp);
        svfloat32_t src0 = svreinterpret_f32_bf16(tmp0);
        svfloat32_t src1 = svreinterpret_f32_bf16(tmp1);
        svfloat16_t vec0 = svcvt_f16_f32_x(pgTrue, src0);
        svfloat16_t vec1 = svcvt_f16_f32_x(pgTrue, src1);
        in0 = svuzp1_f16(vec0, vec1);
        Service::Unused(pg1, in1, pg2, in2, pg3, in3);
    }
};

template <>
struct LoadN<__bf16, __fp16, 2> final { // 2 means that load two __bf16 regs into __fp16 regs
    static void Do(const __bf16 *src,
        const svbool_t& pg0, const svbool_t &pg1, const svbool_t &pg2, const svbool_t &pg3,
        typename Vec<__fp16>::Type &in0, typename Vec<__fp16>::Type &in1,
        typename Vec<__fp16>::Type &in2, typename Vec<__fp16>::Type &in3) noexcept
    {
        svbool_t pgTrue = svptrue_b32();
        svbfloat16_t zeros = svdup_n_bf16(vcvth_bf16_f32(0.0f));
        {
            svbfloat16_t tmp = svld1_bf16(pg0, src);
            svbfloat16_t tmp0 = svzip1(zeros, tmp);
            svbfloat16_t tmp1 = svzip2(zeros, tmp);
            svfloat32_t src0 = svreinterpret_f32_bf16(tmp0);
            svfloat32_t src1 = svreinterpret_f32_bf16(tmp1);
            svfloat16_t vec0 = svcvt_f16_f32_x(pgTrue, src0);
            svfloat16_t vec1 = svcvt_f16_f32_x(pgTrue, src1);
            in0 = svuzp1_f16(vec0, vec1);
        }
        {
            svbfloat16_t tmp = svld1_bf16(pg1, src + Vector::Length<__bf16>());
            svbfloat16_t tmp0 = svzip1(zeros, tmp);
            svbfloat16_t tmp1 = svzip2(zeros, tmp);
            svfloat32_t src0 = svreinterpret_f32_bf16(tmp0);
            svfloat32_t src1 = svreinterpret_f32_bf16(tmp1);
            svfloat16_t vec0 = svcvt_f16_f32_x(pgTrue, src0);
            svfloat16_t vec1 = svcvt_f16_f32_x(pgTrue, src1);
            in1 = svuzp1_f16(vec0, vec1);
        }
        Service::Unused(pg2, in2, pg3, in3);
    }
};

template <>
struct LoadN<__fp16, __fp16, 1> final {
    static void Do(const __fp16 *src,
        const svbool_t& pg0, const svbool_t &pg1, const svbool_t &pg2, const svbool_t &pg3,
        typename Vec<__fp16>::Type &in0, typename Vec<__fp16>::Type &in1,
        typename Vec<__fp16>::Type &in2, typename Vec<__fp16>::Type &in3) noexcept
    {
        in0 = svld1_f16(pg0, src);
        Service::Unused(pg1, in1, pg2, in2, pg3, in3);
    }
};

template <>
struct LoadN<__fp16, __fp16, 2> final { // 2 means that load two __fp16 regs into __fp16 regs
    static void Do(const __fp16 *src,
        const svbool_t& pg0, const svbool_t &pg1, const svbool_t &pg2, const svbool_t &pg3,
        typename Vec<__fp16>::Type &in0, typename Vec<__fp16>::Type &in1,
        typename Vec<__fp16>::Type &in2, typename Vec<__fp16>::Type &in3) noexcept
    {
        in0 = svld1_f16(pg0, src);
        in1 = svld1_f16(pg1, src + Vector::Length<__fp16>());
        Service::Unused(pg2, in2, pg3, in3);
    }
};

template <>
struct LoadN<std::int8_t, __fp16, 1> final {
    static void Do(const std::int8_t *src, const svbool_t& pg0, const svbool_t &pg1,
        const svbool_t &pg2, const svbool_t &pg3, typename Vec<__fp16>::Type &in0, typename Vec<__fp16>::Type &in1,
        typename Vec<__fp16>::Type &in2, typename Vec<__fp16>::Type &in3) noexcept
    {
        svint8_t inreg = svld1_s8(pg0, src);
        svuint16_t inreg16 = svreinterpret_u16(inreg);

        svbool_t pgtrue = svptrue_b16();

        svuint16_t tmp0 = svand_n_u16_x(pgtrue, inreg16, 0xff00u);
        svuint16_t tmp1 = svand_n_u16_x(pgtrue, inreg16, 0x00ffu);
        tmp1 = svlsl_n_u16_x(pgtrue, tmp1, 8); // 8 bits is a size of 1 std::int8_t element
        svint16_t stmp0 = svreinterpret_s16(tmp0);
        stmp0 = svasr_n_s16_x(pgtrue, stmp0, 8); // 8 bits is a size of 1 std::int8_t element
        svint16_t stmp1 = svreinterpret_s16(tmp1);
        stmp1 = svasr_n_s16_x(pgtrue, stmp1, 8); // 8 bits is a size of 1 std::int8_t element

        svint16_t t0 = svzip1_s16(stmp1, stmp0);
        svint16_t t1 = svzip2_s16(stmp1, stmp0);

        in0 = svcvt_f16_x(pgtrue, t0);
        in1 = svcvt_f16_x(pgtrue, t1);
        Service::Unused(pg1, pg2, pg3, in2, in3);
    }
};

template <>
struct LoadN<std::uint8_t, __fp16, 1> final {
    static void Do(const std::uint8_t *src, const svbool_t& pg0, const svbool_t &pg1,
        const svbool_t &pg2, const svbool_t &pg3, typename Vec<__fp16>::Type &in0, typename Vec<__fp16>::Type &in1,
        typename Vec<__fp16>::Type &in2, typename Vec<__fp16>::Type &in3) noexcept
    {
        svuint8_t inreg = svld1_u8(pg0, src);
        svuint16_t inreg16 = svreinterpret_u16(inreg);

        svbool_t pgtrue = svptrue_b16();

        svuint16_t tmp0 = svand_n_u16_x(pgtrue, inreg16, 0xff00u);
        svuint16_t tmp1 = svand_n_u16_x(pgtrue, inreg16, 0x00ffu);
    
        tmp1 = svlsl_n_u16_x(pgtrue, tmp1, 8); // 8 bits is a size of 1 std::uint8_t element
        
        tmp0 = svlsr_n_u16_x(pgtrue, tmp0, 8); // 8 bits is a size of 1 std::uint8_t elements
        tmp1 = svlsr_n_u16_x(pgtrue, tmp1, 8); // 8 bits is a size of 1 std::uint8_t elements

        svuint16_t t0 = svzip1_u16(tmp1, tmp0);
        svuint16_t t1 = svzip2_u16(tmp1, tmp0);

        in0 = svcvt_f16_x(pgtrue, t0);
        in1 = svcvt_f16_x(pgtrue, t1);
        Service::Unused(pg1, pg2, pg3, in2, in3);
    }
};

} // Vector
} // Detail
} // KuDNN

#endif // KUDNN_VECTOR_MEM_LOAD_COMPTYPE_F16_HPP