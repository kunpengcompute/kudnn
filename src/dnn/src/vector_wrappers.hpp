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

#ifndef KUDNN_VECTOR_WRAPPERS
#define KUDNN_VECTOR_WRAPPERS

#include <arm_sve.h>

#if defined(__clang__) || defined(KUDNNL_920G)
#include <arm_neon.h>
#endif

#ifdef KUDNN_DEBUG
#include <cstdint>
#include <iomanip>
#include <iostream>
#endif

#include "kudnn.hpp"
#include "internal_funcs.hpp"

#define CONCAT(a, b) a##b
#define CONCAT_MACRO(a, b) CONCAT(a, b)

namespace KuDNN {
namespace Detail {
namespace Vector {

// Here's a list of abstactions, needed to avoid code duplication for different data types and
// their combinations.
// We support s8, u8, s32, fp16, bf16, f32 data types and their combinations as input/compute/output
// Note, compute types could be only f32 and f16.
// So there could be huge amount of kernels if we don't consider how to avoid code duplication.
// Since max single element data type size ratio is 4 we will have pseudo unroll by 4.
// We suppose that 8 is too much, because even if we have suppoort of i4, fp4 or want to add fp64 data type,
// we don't expect to have kernels with ratio 8. e.g. if in/out data types are i8, there's no need to compute in fp64
// because f32 is precise enough. So 4 is maximum pseudo-unroll factor and all functions take quad arguemnts instead
// of a single one.

constexpr SizeType VC_MAX_BYTE_LEN_920B = 32;
constexpr SizeType VC_MAX_BYTE_LEN_920F = 64;
constexpr SizeType VC_MAX_BYTE_LEN_920G = 64;

template <typename ScalarType, ARCH Arch = ARCH::VC>
struct Vec {
};

template <>
struct Vec<float> {
    using Type = svfloat32_t;
};

template <>
struct Vec<__fp16> {
    using Type = svfloat16_t;
};

template <>
struct Vec<__bf16> {
    using Type = svbfloat16_t;
};

template <>
struct Vec<std::int32_t> {
    using Type = svint32_t;
};

template <>
struct Vec<std::int8_t> {
    using Type = svint8_t;
};

template <>
struct Vec<std::uint8_t> {
    using Type = svuint8_t;
};

#ifdef KUDNNL_920G
template <>
struct Vec<float, ARCH::NEON> {
    using Type = float32x4x2_t;
};

template <>
struct Vec<__fp16, ARCH::NEON> {
    using Type = float16x8x2_t;
};

template <>
struct Vec<__bf16, ARCH::NEON> {
    using Type = bfloat16x8x2_t;
};

template <>
struct Vec<std::int32_t, ARCH::NEON> {
    using Type = int32x4x2_t;
};

template <>
struct Vec<std::int8_t, ARCH::NEON> {
    using Type = int8x16x2_t;
};

template <>
struct Vec<std::uint8_t, ARCH::NEON> {
    using Type = uint8x16x2_t;
};

#endif

template <typename ScalarType, ARCH Arch = ARCH::VC>
struct MaskVec {
};

template <typename ScalarType>
struct MaskVec<ScalarType, ARCH::VC> {
    using Type = svbool_t;
};

#ifdef KUDNNL_920G
template <>
struct MaskVec<float, ARCH::NEON> {
    using Type = uint32x4x2_t;
};

template <>
struct MaskVec<__fp16, ARCH::NEON> {
    using Type = uint16x8x2_t;
};

template <>
struct MaskVec<__bf16, ARCH::NEON> {
    using Type = uint16x8x2_t;
};

template <>
struct MaskVec<std::int32_t, ARCH::NEON> {
    using Type = uint32x4x2_t;
};

template <>
struct MaskVec<std::int8_t, ARCH::NEON> {
    using Type = uint8x16x2_t;
};

template <>
struct MaskVec<std::uint8_t, ARCH::NEON> {
    using Type = uint8x16x2_t;
};
#endif

template <typename T>
KUDNN_INLINE constexpr inline SizeType GetMinSize()
{
    return sizeof(T);
}

template <typename T1, typename T2, typename ... Rest>
KUDNN_INLINE constexpr inline SizeType GetMinSize()
{
    return (sizeof(T1) > GetMinSize<T2, Rest ...>()) ? GetMinSize<T2, Rest ...>() : sizeof(T1);
}

template <typename ScalarType, ARCH Arch = ARCH::VC>
KUDNN_INLINE inline SizeType Length();

template <typename ... Args>
KUDNN_INLINE inline SizeType UnrollLength()
{
    return svcntb() / GetMinSize<Args ...>();
}

template <typename ... Args>
constexpr SizeType NeonLanes()
{
    return NUM_16 / GetMinSize<Args ...>();   // 128 bits = 16 bytes
}

// Computes how many of T-type registers will be used
template <typename T, typename ... Rest>
KUDNN_INLINE constexpr inline SizeType GetUsedRegsN()
{
    return sizeof(T) / GetMinSize<T, Rest ...>();
}

// Single Predicate operations
template <typename Type>
struct WhileLt final {
    static void Do(svbool_t &pg, SizeType loopVar, SizeType loopMax) noexcept
    {
        if constexpr (sizeof(Type) == 1) {
            pg = svwhilelt_b8_s64(loopVar, loopMax);
        } else if constexpr (sizeof(Type) == 2) {
            pg = svwhilelt_b16_s64(loopVar, loopMax);
        } else if constexpr (sizeof(Type) == 4) {
            pg = svwhilelt_b32_s64(loopVar, loopMax);
        } else if constexpr (sizeof(Type) == 8) {
            pg = svwhilelt_b64_s64(loopVar, loopMax);
        }
        // no other options
    }
};

template <typename Type, ARCH Arch = ARCH::VC>
struct DupFloat final {
    static void Do(const float &src, typename Vec<Type>::Type &in) noexcept;
};

template <typename Type>
struct DupFloat<Type, ARCH::VC> final {
    static void Do(const float &src, typename Vec<Type>::Type &in) noexcept
    {
        if constexpr (std::is_same_v<Type, double>) {
            in = svdup_n_f64(src);
        } else if constexpr (std::is_same_v<Type, float>) {
            in = svdup_n_f32(src);
        } else if constexpr (std::is_same_v<Type, __fp16>) {
            in = svdup_n_f16(src);
        } else if constexpr (std::is_same_v<Type, __bf16>) {
            in = svdup_n_bf16(vcvth_bf16_f32(src));
        } else if constexpr (std::is_same_v<Type, std::int64_t>) {
            in = svdup_n_s64(src);
        } else if constexpr (std::is_same_v<Type, std::int32_t>) {
            in = svdup_n_s32(src);
        } else if constexpr (std::is_same_v<Type, std::int16_t>) {
            in = svdup_n_s16(src);
        } else if constexpr (std::is_same_v<Type, std::int8_t>) {
            in = svdup_n_s8(src);
        }
        // no other options
    }
};

#ifdef KUDNNL_920G
template <typename Type>
struct DupFloat<Type, ARCH::NEON> final {
    static void Do(const float &src, typename Vec<Type, ARCH::NEON>::Type &in) noexcept
    {
        if constexpr (std::is_same_v<Type, float>) {
            in.val[0] = vdupq_n_f32(src);
            in.val[1] = vdupq_n_f32(src);
        }
        // add other options if needed
    }
};
#endif // KUDNNL_920G

template <typename Type>
struct GatherLoad final {
    static_assert(std::is_same_v<Type, float> || std::is_same_v<Type, __fp16> ||
        std::is_same_v<Type, __bf16>, "Only f32, f16, bf16 are supported for gather load");
    static void Do(const Type *src, SizeType loopVar, SizeType loopMax,
        SizeType offset, typename Vec<Type>::Type &in) noexcept
    {
        if constexpr (std::is_same_v<Type, float>) {
            svbool_t pg = svwhilelt_b32(loopVar, loopMax);
            Do(src, pg, offset, in);
        } else {
            const Type *src0 = src;
            const Type *src1 = src + offset;
            svbool_t pg0 = svwhilelt_b32_s64(0, (loopMax - loopVar) - (loopMax - loopVar) / 2);
            svbool_t pg1 = svwhilelt_b32_s64(0, (loopMax - loopVar) / 2);
            svuint32_t offsets = svindex_u32(0, 2 * offset * sizeof(Type));
            // there's no gather-load intrinsics for fp16 and bf16, so we use assembly w/a
            asm volatile(
                "ld1h {z0.s}, %[pg0]/z, [%[src0], %[offsets].s, uxtw]\n"
                "ld1h {z1.s}, %[pg1]/z, [%[src1], %[offsets].s, uxtw]\n"
                "trn1 %[in].h, z0.h, z1.h\n"
                :  // outputs
                [in] "=w"(in)
                :  // inputs
                [src0] "r"(src0),
                [src1] "r"(src1),
                [offsets] "w"(offsets),
                [pg0] "Upl"(pg0),
                [pg1] "Upl"(pg1)
                :  // clobbers
                "cc", "memory",
                "z0", "z1"
            );
        }
    }

    static void Do(const Type *src, svbool_t pg, SizeType offset,
        typename Vec<Type>::Type &in) noexcept
    {
        if constexpr (std::is_same_v<Type, float>) {
            svuint32_t offsets = svindex_u32(0, offset * sizeof(Type));
            in = svld1_gather_offset(pg, src, offsets);
        } else {
            auto n = svcntp_b16(pg, pg);
            Do(src, 0, n, offset, in);
        }
    }
};

template <typename Type>
struct ScatterStore final {
    static_assert(std::is_same_v<Type, float> || std::is_same_v<Type, __fp16> ||
        std::is_same_v<Type, __bf16>, "Only f32, f16, bf16 are supported for gather load");
    static void Do(SizeType loopVar, SizeType loopMax,
        SizeType offset, typename Vec<Type>::Type &out, Type *dst) noexcept
    {
        if constexpr (std::is_same_v<Type, float>) {
            svbool_t pg = svwhilelt_b32(loopVar, loopMax);
            Do(pg, offset, out, dst);
        } else {
            const Type *dst0 = dst;
            const Type *dst1 = dst + offset;
            svbool_t pg0 = svwhilelt_b32_s64(0, (loopMax - loopVar) - (loopMax - loopVar) / 2);
            svbool_t pg1 = svwhilelt_b32_s64(0, (loopMax - loopVar) / 2);
            svuint32_t offsets = svindex_u32(0, 2 * offset * sizeof(Type));
            // there's no scatter-store intrinsics for fp16 and bf16, so we use assembly w/a
            asm volatile(
                "trn1 z0.h, %[out].h, %[out].h\n"
                "trn2 z1.h, %[out].h, %[out].h\n"
                "st1h {z0.s}, %[pg0], [%[dst0], %[offsets].s, uxtw]\n"
                "st1h {z1.s}, %[pg1], [%[dst1], %[offsets].s, uxtw]\n"
                :  // outputs
                [dst0] "+r"(dst0),
                [dst1] "+r"(dst1)
                :  // inputs
                [out] "w"(out),
                [offsets] "w"(offsets),
                [pg0] "Upl"(pg0),
                [pg1] "Upl"(pg1)
                :  // clobbers
                "cc", "memory",
                "z0", "z1"
            );
        }
    }

    static void Do(svbool_t pg, SizeType offset,
        typename Vec<Type>::Type &out, Type *dst) noexcept
    {
        if constexpr (std::is_same_v<Type, float>) {
            svuint32_t offsets = svindex_u32(0, offset * sizeof(Type));
            svst1_scatter_offset(pg, dst, offsets, out);
        } else {
            auto n = svcntp_b16(pg, pg);
            Do(0, n, offset, out, dst);
        }
    }
};

// Predicate operations

template <typename Type, SizeType N, ARCH Arch = ARCH::VC>
struct InitPGN final {
    static void Do(typename Vector::MaskVec<Type, Arch>::Type &pg0, typename Vector::MaskVec<Type, Arch>::Type &pg1,
    typename Vector::MaskVec<Type, Arch>::Type &pg2, typename Vector::MaskVec<Type, Arch>::Type &pg3,
    SizeType loopVar, SizeType loopMax) noexcept;
};

// Memory operations

template <typename InType, typename ComputeType, SizeType N, ARCH Arch = ARCH::VC>
struct LoadN final {
    static void Do(const InType *src,
        const svbool_t& pg0, const svbool_t &pg1, const svbool_t &pg2, const svbool_t &pg3,
        typename Vec<ComputeType, Arch>::Type &in0, typename Vec<ComputeType, Arch>::Type &in1,
        typename Vec<ComputeType, Arch>::Type &in2, typename Vec<ComputeType, Arch>::Type &in3) noexcept;
};

template <typename InType, typename ComputeType, SizeType N, ARCH Arch = ARCH::VC>
struct DupN {
    static void Do(const InType &src,
        typename Vec<ComputeType, Arch>::Type &in0, typename Vec<ComputeType, Arch>::Type &in1,
        typename Vec<ComputeType, Arch>::Type &in2, typename Vec<ComputeType, Arch>::Type &in3) noexcept;
};

template <typename InType, typename ComputeType, SizeType N, ARCH Arch = ARCH::VC>
struct DupFloatN final {
    static void Do(const float &src,
        typename Vec<ComputeType, Arch>::Type &in0, typename Vec<ComputeType, Arch>::Type &in1,
        typename Vec<ComputeType, Arch>::Type &in2, typename Vec<ComputeType, Arch>::Type &in3) noexcept;
};

template <typename OutType, typename ComputeType, SizeType N, ARCH Arch = ARCH::VC>
struct StoreN final {
    static void Do(const svbool_t& pg0, const svbool_t &pg1, const svbool_t &pg2,
        const svbool_t &pg3,
        const typename Vec<ComputeType, Arch>::Type &out0,
        const typename Vec<ComputeType, Arch>::Type &out1,
        const typename Vec<ComputeType, Arch>::Type &out2,
        const typename Vec<ComputeType, Arch>::Type &out3,
        OutType *dst) noexcept;
};

template <typename FromType, typename ToType>
struct Convert final {
    static ToType Do(FromType val) noexcept;
};

template <typename FromType, typename ToType, SizeType N>
struct ConvertN final {
    static void Do(const svbool_t& pg0, const svbool_t &pg1, const svbool_t &pg2,
        const svbool_t &pg3, const typename Vec<FromType>::Type &in0, const typename Vec<FromType>::Type &in1,
        const typename Vec<FromType>::Type &in2, const typename Vec<FromType>::Type &in3,
        typename Vec<ToType>::Type &out0, typename Vec<ToType>::Type &out1,
        typename Vec<ToType>::Type &out2, typename Vec<ToType>::Type &out3) noexcept;
};

template <typename InType, typename ComputeType, SizeType N, ARCH Arch = ARCH::VC>
struct LoadWrapperN final {
    static void Do(const InType *src, SizeType higherIdx, SizeType lowerIdx,
        const svbool_t& pg0, const svbool_t &pg1, const svbool_t &pg2, const svbool_t &pg3,
        typename Vector::Vec<ComputeType, Arch>::Type &in0, typename Vector::Vec<ComputeType, Arch>::Type &in1,
        typename Vector::Vec<ComputeType, Arch>::Type &in2, typename Vector::Vec<ComputeType, Arch>::Type &in3) noexcept
    {
        Vector::LoadN<InType, ComputeType, N, Arch>::Do(src + higherIdx + lowerIdx, pg0, pg1, pg2, pg3,
            in0, in1, in2, in3);
    }
};

template <typename InType, typename ComputeType, SizeType N, ARCH Arch = ARCH::VC>
struct DupWrapperN final {
    static void Do(const InType *src, SizeType higherIdx, SizeType lowerIdx,
        const svbool_t& pg0, const svbool_t &pg1, const svbool_t &pg2, const svbool_t &pg3,
        typename Vector::Vec<ComputeType, Arch>::Type &in0, typename Vector::Vec<ComputeType, Arch>::Type &in1,
        typename Vector::Vec<ComputeType, Arch>::Type &in2, typename Vector::Vec<ComputeType, Arch>::Type &in3) noexcept
    {
        Vector::DupN<InType, ComputeType, N, Arch>::Do(*(src + higherIdx), in0, in1, in2, in3);
        Service::Unused(lowerIdx, pg0, pg1, pg2, pg3);
    }
};

// Element selection
template <typename ComputeType, SizeType N, ARCH Arch = ARCH::VC>
struct SelN final {
    static void Do(const typename MaskVec<ComputeType>::Type &mask0, const typename MaskVec<ComputeType>::Type &mask1,
        const typename MaskVec<ComputeType>::Type  &mask2, const typename MaskVec<ComputeType>::Type  &mask3,
        const typename Vec<ComputeType>::Type &lhs0, const typename Vec<ComputeType>::Type &lhs1,
        const typename Vec<ComputeType>::Type &lhs2, const typename Vec<ComputeType>::Type &lhs3,
        const typename Vec<ComputeType>::Type &rhs0, const typename Vec<ComputeType>::Type &rhs1,
        const typename Vec<ComputeType>::Type &rhs2, const typename Vec<ComputeType>::Type &rhs3,
        typename Vec<ComputeType>::Type &out0, typename Vec<ComputeType>::Type &out1,
        typename Vec<ComputeType>::Type &out2, typename Vec<ComputeType>::Type &out3) noexcept;
};

template <typename ComputeType, SizeType N>
struct SelN<ComputeType, N, ARCH::VC> final {
    static void Do(const svbool_t& mask0, const svbool_t &mask1,
        const svbool_t &mask2, const svbool_t &mask3,
        const typename Vec<ComputeType>::Type &lhs0, const typename Vec<ComputeType>::Type &lhs1,
        const typename Vec<ComputeType>::Type &lhs2, const typename Vec<ComputeType>::Type &lhs3,
        const typename Vec<ComputeType>::Type &rhs0, const typename Vec<ComputeType>::Type &rhs1,
        const typename Vec<ComputeType>::Type &rhs2, const typename Vec<ComputeType>::Type &rhs3,
        typename Vec<ComputeType>::Type &out0, typename Vec<ComputeType>::Type &out1,
        typename Vec<ComputeType>::Type &out2, typename Vec<ComputeType>::Type &out3) noexcept
    {
        static_assert(N <= 4, "Pseudo unroll by 4 is maximum");
        out0 = svsel(mask0, lhs0, rhs0);
        if constexpr (N > 1) {
            out1 = svsel(mask1, lhs1, rhs1);
            if constexpr (N > 2) {
                out2 = svsel(mask2, lhs2, rhs2);
                if constexpr (N > 3) {
                    out3 = svsel(mask3, lhs3, rhs3);
                }
            }
        }
    }
};

#ifdef KUDNNL_920G
template <typename ComputeType, SizeType N>
struct SelN<ComputeType, N, ARCH::NEON> final {
    static inline void  Operation(const typename MaskVec<float, ARCH::NEON>::Type &mask,
        const typename Vec<float, ARCH::NEON>::Type &lhs, const typename Vec<float, ARCH::NEON>::Type &rhs,
        typename Vec<float, ARCH::NEON>::Type &out)
    {
        out.val[0] = vbslq_f32(mask.val[0], lhs.val[0], rhs.val[0]);
        out.val[1] = vbslq_f32(mask.val[1], lhs.val[1], rhs.val[1]);
    }
    static inline void Operation(const typename MaskVec<float16_t, ARCH::NEON>::Type &mask,
        const typename Vec<float16_t, ARCH::NEON>::Type &lhs, const typename Vec<float16_t, ARCH::NEON>::Type &rhs,
        typename Vec<float16_t, ARCH::NEON>::Type &out)
    {
        out.val[0] = vbslq_f16(mask.val[0], lhs.val[0], rhs.val[0]);
        out.val[1] = vbslq_f16(mask.val[1], lhs.val[1], rhs.val[1]);
    }
    static inline void  Operation(const typename MaskVec<int32_t, ARCH::NEON>::Type &mask,
        const typename Vec<int32_t, ARCH::NEON>::Type &lhs, const typename Vec<int32_t, ARCH::NEON>::Type &rhs,
        typename Vec<int32_t, ARCH::NEON>::Type &out)
    {
        out.val[0] = vbslq_s32(mask.val[0], lhs.val[0], rhs.val[0]);
        out.val[1] = vbslq_s32(mask.val[1], lhs.val[1], rhs.val[1]);
    }

    static void Do(
        const typename MaskVec<ComputeType, ARCH::NEON>::Type &mask0,
        const typename MaskVec<ComputeType, ARCH::NEON>::Type &mask1,
        const typename MaskVec<ComputeType, ARCH::NEON>::Type &mask2,
        const typename MaskVec<ComputeType, ARCH::NEON>::Type &mask3,
        const typename Vec<ComputeType, ARCH::NEON>::Type &lhs0,
        const typename Vec<ComputeType, ARCH::NEON>::Type &lhs1,
        const typename Vec<ComputeType, ARCH::NEON>::Type &lhs2,
        const typename Vec<ComputeType, ARCH::NEON>::Type &lhs3,
        const typename Vec<ComputeType, ARCH::NEON>::Type &rhs0,
        const typename Vec<ComputeType, ARCH::NEON>::Type &rhs1,
        const typename Vec<ComputeType, ARCH::NEON>::Type &rhs2,
        const typename Vec<ComputeType, ARCH::NEON>::Type &rhs3,
        typename Vec<ComputeType, ARCH::NEON>::Type &out0,
        typename Vec<ComputeType, ARCH::NEON>::Type &out1,
        typename Vec<ComputeType, ARCH::NEON>::Type &out2,
        typename Vec<ComputeType, ARCH::NEON>::Type &out3) noexcept
    {
        static_assert(N <= 4, "Pseudo unroll by 4 is maximum");
        Operation(mask0, lhs0, rhs0, out0);
        if constexpr (N > 1) {
            Operation(mask1, lhs1, rhs1, out1);
            if constexpr (N > 2) {
                Operation(mask2, lhs2, rhs2, out2);
                if constexpr (N > 3) {
                    Operation(mask3, lhs3, rhs3, out3);
                }
            }
        }
    }
};
#endif

// Across vector math operations

template <typename ComputeType, SizeType N, ARCH Arch = ARCH::VC>
struct AddvN final {
    static ComputeType Do(const svbool_t& pg0, const svbool_t &pg1, const svbool_t &pg2, const svbool_t &pg3,
        const typename Vec<ComputeType, Arch>::Type &lhs0, const typename Vec<ComputeType, Arch>::Type &lhs1,
        const typename Vec<ComputeType, Arch>::Type &lhs2, const typename Vec<ComputeType, Arch>::Type &lhs3) noexcept;
};

template <typename ComputeType, SizeType N>
struct AddvN<ComputeType, N, ARCH::VC> final {
    static ComputeType Do(const svbool_t& pg0, const svbool_t &pg1, const svbool_t &pg2, const svbool_t &pg3,
        const typename Vec<ComputeType>::Type &lhs0, const typename Vec<ComputeType>::Type &lhs1,
        const typename Vec<ComputeType>::Type &lhs2, const typename Vec<ComputeType>::Type &lhs3) noexcept
    {
        static_assert(N <= 4, "Pseudo unroll by 4 is maximum");
        ComputeType result = svaddv(pg0, lhs0);
        if constexpr (N > 1) {
            result += svaddv(pg1, lhs1);
            if constexpr (N > 2) {
                result += svaddv(pg2, lhs2);
                if constexpr (N > 3) {
                    result += svaddv(pg3, lhs3);
                }
            }
        }
        return result;
    }
};

#ifdef KUDNNL_920G
template <typename ComputeType, SizeType N>
struct AddvN<ComputeType, N, ARCH::NEON> final {
    static ComputeType Do(const svbool_t& pg0, const svbool_t &pg1, const svbool_t &pg2, const svbool_t &pg3,
        const typename Vec<ComputeType, ARCH::NEON>::Type &lhs0,
        const typename Vec<ComputeType, ARCH::NEON>::Type &lhs1,
        const typename Vec<ComputeType, ARCH::NEON>::Type &lhs2,
        const typename Vec<ComputeType, ARCH::NEON>::Type &lhs3) noexcept
    {
        Service::Unused(pg0, pg1, pg2, pg3);
        static_assert(N <= 4, "Pseudo unroll by 4 is maximum");
        if constexpr (std::is_same_v<ComputeType, float>) {
            ComputeType result = vaddvq_f32(lhs0.val[0]) + vaddvq_f32(lhs0.val[1]);
            if constexpr (N > 1) {
                result += vaddvq_f32(lhs1.val[0]) + vaddvq_f32(lhs1.val[1]);
                if constexpr (N > 2) {
                    result += vaddvq_f32(lhs2.val[0]) + vaddvq_f32(lhs2.val[1]);
                    if constexpr (N > 3) {
                        result += vaddvq_f32(lhs3.val[0]) + vaddvq_f32(lhs3.val[1]);
                    }
                }
            }
            return result;
        }
    }
};
#endif

template <>
struct AddvN<int32_t, 1> final {
    static int32_t Do(const svbool_t& pg0, const svbool_t &pg1, const svbool_t &pg2,
        const svbool_t &pg3, const typename Vec<int32_t>::Type &lhs0, const typename Vec<int32_t>::Type &lhs1,
        const typename Vec<int32_t>::Type &lhs2, const typename Vec<int32_t>::Type &lhs3) noexcept
    {
        int32_t result = svaddv_s32(pg0, lhs0);
        Service::Unused(pg1, pg2, pg3, lhs1, lhs2, lhs3);
        return result;
    }
};

template <typename ComputeType, SizeType N>
struct MaxvN final {
    static ComputeType Do(const svbool_t& pg0, const svbool_t &pg1, const svbool_t &pg2,
        const svbool_t &pg3, const typename Vec<ComputeType>::Type &lhs0, const typename Vec<ComputeType>::Type &lhs1,
        const typename Vec<ComputeType>::Type &lhs2, const typename Vec<ComputeType>::Type &lhs3) noexcept
    {
        static_assert(N <= 4, "Pseudo unroll by 4 is maximum");
        ComputeType max = svmaxv(pg0, lhs0);
        if constexpr (N > 1) {
            max = std::max(svmaxv(pg1, lhs1), max);
            if constexpr (N > 2) {
                max = std::max(svmaxv(pg2, lhs2), max);
                if constexpr (N > 3) {
                    max = std::max(svmaxv(pg3, lhs3), max);
                }
            }
        }
        return max;
    }
};

// Custom operations used in kernels

template <typename ComputeType>
struct Log final {
    static void Do(const svbool_t &pg, typename Vector::Vec<ComputeType>::Type &in,
        typename Vector::Vec<ComputeType>::Type &out) noexcept;
    static void DoStore(const svbool_t &pg, svbool_t &pgRetNan, svbool_t &pgRetInf, svbool_t &pgRetNegInf,
        typename Vector::Vec<ComputeType>::Type &poly, typename Vector::Vec<ComputeType>::Type &out,
        typename Vector::Vec<ComputeType>::Type &n) noexcept;
};

template <typename ComputeType, SizeType N>
struct LogN final {
    static void Do(const svbool_t& pg0,
        const svbool_t &pg1, const svbool_t &pg2, const svbool_t &pg3,
        typename Vector::Vec<ComputeType>::Type &in0, typename Vector::Vec<ComputeType>::Type &in1,
        typename Vector::Vec<ComputeType>::Type &in2, typename Vector::Vec<ComputeType>::Type &in3,
        typename Vector::Vec<ComputeType>::Type &out0, typename Vector::Vec<ComputeType>::Type &out1,
        typename Vector::Vec<ComputeType>::Type &out2, typename Vector::Vec<ComputeType>::Type &out3) noexcept
    {
        static_assert(N <= 4, "Pseudo unroll by 4 is maximum");
        Log<ComputeType>::Do(pg0, in0, out0);
        if constexpr (N > 1) {
            Log<ComputeType>::Do(pg1, in1, out1);
            if constexpr (N > 2) {
                Log<ComputeType>::Do(pg2, in2, out2);
                if constexpr (N > 3) {
                    Log<ComputeType>::Do(pg3, in3, out3);
                }
            }
        }
    }
};

template <typename ComputeType, SizeType N>
struct ExpN final {
    static void Do(const svbool_t& pg0,
        const svbool_t &pg1, const svbool_t &pg2, const svbool_t &pg3,
        typename Vector::Vec<ComputeType>::Type &in0, typename Vector::Vec<ComputeType>::Type &in1,
        typename Vector::Vec<ComputeType>::Type &in2, typename Vector::Vec<ComputeType>::Type &in3,
        typename Vector::Vec<ComputeType>::Type &out0, typename Vector::Vec<ComputeType>::Type &out1,
        typename Vector::Vec<ComputeType>::Type &out2, typename Vector::Vec<ComputeType>::Type &out3) noexcept;
};

#ifdef KUDNN_DEBUG
template <typename Type>
inline void PrintRegister(const svbool_t& pg, typename Vector::Vec<Type>::Type &data) noexcept
{
#ifdef KUDNNL_920F
    Type buffer[VC_MAX_BYTE_LEN_920F / sizeof(Type)];
#else
    Type buffer[VC_MAX_BYTE_LEN_920B / sizeof(Type)];
#endif

    svst1(pg, buffer, data);

    SizeType activeCount = svcntp_b8(svptrue_b8(), pg);
    std::cout << "[Active elements = " << activeCount << "]: ";
    for (SizeType i = 0; i < activeCount; ++i) {
        if constexpr (std::is_same_v<Type, float> || std::is_same_v<Type, __fp16>) {
            // 12 means that print 12 effective digits
            std::cout << std::fixed << std::setprecision(12) << static_cast<float>(buffer[i]) << "  ";
        } else if constexpr (std::is_same_v<Type, __bf16>) {
            // 12 means that print 12 effective digits
            std::cout << std::fixed << std::setprecision(12) << vcvth_bf16_f32(buffer[i]) << "  ";
        } else {
            std::cout << static_cast<std::int32_t>(buffer[i]) << "  ";
        }
    }
    std::cout << std::endl;
}

template <typename Type, SizeType N>
inline void PrintRegister(const svbool_t& pg0,
    const svbool_t& pg1, const svbool_t& pg2, const svbool_t& pg3,
    typename Vector::Vec<Type>::Type &data0, typename Vector::Vec<Type>::Type &data1,
    typename Vector::Vec<Type>::Type &data2, typename Vector::Vec<Type>::Type &data3) noexcept
{
    std::cout << "====================" << std::endl;
    std::cout << "[VReg0]";
    PrintRegister<Type>(pg0, data0);
    if constexpr (N > 1) {
        std::cout << "[VReg1]";
        PrintRegister<Type>(pg1, data1);
        if constexpr (N > 2) {
            std::cout << "[VReg2]";
            PrintRegister<Type>(pg2, data2);
            if constexpr (N > 3) {
                std::cout << "[VReg3]";
                PrintRegister<Type>(pg3, data3);
            }
        }
    }
    std::cout << "====================" << std::endl;
}
#endif

} // Vector
} // Detail
} // KuDNN

#include "vector_apis/vector_compare.hpp"
#include "vector_apis/vector_compare_sve.hpp"
#include "vector_apis/vector_compare_neon.hpp"
#include "vector_apis/vector_arithmetic.hpp"
#include "vector_apis/vector_arithmetic_neon.hpp"
#include "vector_apis/vector_arithmetic_sve.hpp"
#include "vector_apis/vector_arithmetic_sve_mul_div.hpp"
#include "vector_apis/vector_length.hpp"
#include "vector_apis/vector_init_predicate.hpp"
#include "vector_apis/vector_mem_operations.hpp"
#include "vector_apis/vector_convert.hpp"
#include "vector_apis/vector_log.hpp"
#include "vector_apis/vector_exp.hpp"

#endif // KUDNN_VECTOR_WRAPPERS
