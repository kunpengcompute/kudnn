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

#ifndef KUDNN_VECTOR_LOG
#define KUDNN_VECTOR_LOG

#include <arm_sve.h>

#include "kudnn.hpp"

#include "internal_funcs.hpp"

namespace KuDNN {
namespace Detail {
namespace Vector {
namespace Helpers {

template <typename ScalarType>
struct FpToVecIntTypeMap {
};

template <>
struct FpToVecIntTypeMap<float> {
    using UnsignedIntScalarType = std::uint32_t;
    using SignedIntVecType = svint32_t;
    using UnsignedIntVecType = svuint32_t;
};

template <>
struct FpToVecIntTypeMap<__fp16> {
    using UnsignedIntScalarType = std::uint16_t;
    using SignedIntVecType = svint16_t;
    using UnsignedIntVecType = svuint16_t;
};

template <typename FromType, typename ToType>
struct Reinterpret final {
    static ToType Do(const FromType &in) noexcept;
};

template <> // reinterpret float types to unsigned integer
struct Reinterpret<Vector::Vec<float>::Type, FpToVecIntTypeMap<float>::UnsignedIntVecType> final {
    static typename FpToVecIntTypeMap<float>::UnsignedIntVecType
        Do(const typename Vector::Vec<float>::Type &in) noexcept
        {
            return svreinterpret_u32(in);
        }
};

template <> // reinterpret float types to unsigned integer
struct Reinterpret<Vector::Vec<__fp16>::Type, FpToVecIntTypeMap<__fp16>::UnsignedIntVecType> final {
    static typename FpToVecIntTypeMap<__fp16>::UnsignedIntVecType
        Do(const typename Vector::Vec<__fp16>::Type &in) noexcept
        {
            return svreinterpret_u16(in);
        }
};

template <> // reinterpret unsigned integer types to signed integer
struct Reinterpret<Helpers::FpToVecIntTypeMap<float>::UnsignedIntVecType,
    FpToVecIntTypeMap<float>::SignedIntVecType> final {
    static typename FpToVecIntTypeMap<float>::SignedIntVecType Do(
        const typename Helpers::FpToVecIntTypeMap<float>::UnsignedIntVecType &in) noexcept
        {
            return svreinterpret_s32(in);
        }
};

template <> // reinterpret unsigned integer types to signed integer
struct Reinterpret<Helpers::FpToVecIntTypeMap<__fp16>::UnsignedIntVecType,
    FpToVecIntTypeMap<__fp16>::SignedIntVecType> final {
    static typename FpToVecIntTypeMap<__fp16>::SignedIntVecType Do(
        const typename Helpers::FpToVecIntTypeMap<__fp16>::UnsignedIntVecType &in) noexcept
        {
            return svreinterpret_s16(in);
        }
};

template <> // reinterpret unsigned integer types to float
struct Reinterpret<FpToVecIntTypeMap<float>::UnsignedIntVecType, Vector::Vec<float>::Type> final {
    static typename Vector::Vec<float>::Type
        Do(const typename FpToVecIntTypeMap<float>::UnsignedIntVecType &in) noexcept
        {
            return svreinterpret_f32(in);
        }
};

template <> // reinterpret unsigned integer types to float
struct Reinterpret<FpToVecIntTypeMap<__fp16>::UnsignedIntVecType, Vector::Vec<__fp16>::Type> final {
    static typename Vector::Vec<__fp16>::Type
        Do(const typename FpToVecIntTypeMap<__fp16>::UnsignedIntVecType &in) noexcept
        {
            return svreinterpret_f16(in);
        }
};

template <typename FromType, typename ToType>
struct Convert final {
    static ToType Do(svbool_t pg, const FromType &in) noexcept;
};

template <> // converts signed integer types to float
struct Convert<typename FpToVecIntTypeMap<float>::SignedIntVecType, typename Vector::Vec<float>::Type> final {
    static typename Vector::Vec<float>::Type
        Do(svbool_t pg, const typename FpToVecIntTypeMap<float>::SignedIntVecType &in) noexcept
        {
            return svcvt_f32_x(pg, in);
        }
};

template <> // converts signed integer types to float
struct Convert<typename FpToVecIntTypeMap<__fp16>::SignedIntVecType, typename Vector::Vec<__fp16>::Type> final {
    static typename Vector::Vec<__fp16>::Type
        Do(svbool_t pg, const typename FpToVecIntTypeMap<__fp16>::SignedIntVecType &in) noexcept
        {
            return svcvt_f16_x(pg, in);
        }
};

template <typename ScalarType, typename VectorType>
struct Dup final {
    static VectorType Do(const ScalarType &in) noexcept;
};

template <> // dups float values
struct Dup<float, typename Vector::Vec<float>::Type> final {
    static typename Vector::Vec<float>::Type Do(const float &in) noexcept
        {
            return svdup_f32(in);
        }
};

template <> // dups float values
struct Dup<__fp16, typename Vector::Vec<__fp16>::Type> final {
    static typename Vector::Vec<__fp16>::Type Do(const __fp16 &in) noexcept
        {
            return svdup_f16(in);
        }
};

template <> // dups unsigned integer 32-bit values
struct Dup<typename FpToVecIntTypeMap<float>::UnsignedIntScalarType,
    typename FpToVecIntTypeMap<float>::UnsignedIntVecType> final {
    static typename FpToVecIntTypeMap<float>::UnsignedIntVecType Do(
        const typename FpToVecIntTypeMap<float>::UnsignedIntScalarType &in) noexcept
        {
            return svdup_u32(in);
        }
};

template <> // dups unsigned integer 16-bit values
struct Dup<typename FpToVecIntTypeMap<__fp16>::UnsignedIntScalarType,
    typename FpToVecIntTypeMap<__fp16>::UnsignedIntVecType> final {
    static typename FpToVecIntTypeMap<__fp16>::UnsignedIntVecType Do(
        const typename FpToVecIntTypeMap<__fp16>::UnsignedIntScalarType &in) noexcept
        {
            return svdup_u16(in);
        }
};

template<typename T>
inline constexpr typename FpToVecIntTypeMap<T>::UnsignedIntScalarType LN_SUBNORMAL_BIT = 0x00800000U;
template<>
inline constexpr typename FpToVecIntTypeMap<__fp16>::UnsignedIntScalarType LN_SUBNORMAL_BIT<__fp16> = 0x400U;

template<typename T>
inline constexpr T LN_ZERO = 0.0f;

template<typename T>
inline constexpr typename FpToVecIntTypeMap<T>::UnsignedIntScalarType LN_OFFSET = 0x3f2aaaabU;
template<>
inline constexpr typename FpToVecIntTypeMap<__fp16>::UnsignedIntScalarType LN_OFFSET<__fp16> = 0x3940U;

template<typename T>
inline constexpr typename FpToVecIntTypeMap<T>::UnsignedIntScalarType LN_MANTISSA_BIT_LEN = 23U;
template<>
inline constexpr typename FpToVecIntTypeMap<__fp16>::UnsignedIntScalarType LN_MANTISSA_BIT_LEN<__fp16> = 10U;

template<typename T>
inline constexpr typename FpToVecIntTypeMap<T>::UnsignedIntScalarType LN_MANTISSA_MASK = 0x7fffffU;
template<>
inline constexpr typename FpToVecIntTypeMap<__fp16>::UnsignedIntScalarType LN_MANTISSA_MASK<__fp16> = 0x3ffU;

template<typename T>
inline constexpr T LN_ONE = 1.0f;

template<typename T>
inline constexpr typename FpToVecIntTypeMap<T>::UnsignedIntScalarType LN_NAN_BIT = 0x7fc00000U;
template<>
inline constexpr typename FpToVecIntTypeMap<__fp16>::UnsignedIntScalarType LN_NAN_BIT<__fp16> = 0x7e00U;

template<typename T>
inline constexpr typename FpToVecIntTypeMap<T>::UnsignedIntScalarType LN_INF_BIT = 0x7f800000U;
template<>
inline constexpr typename FpToVecIntTypeMap<__fp16>::UnsignedIntScalarType LN_INF_BIT<__fp16> = 0x7c00U;

template<typename T>
inline constexpr typename FpToVecIntTypeMap<T>::UnsignedIntScalarType LN_NEG_INF_BIT = 0xff800000U;
template<>
inline constexpr typename FpToVecIntTypeMap<__fp16>::UnsignedIntScalarType
    LN_NEG_INF_BIT<__fp16> = static_cast<uint16_t>(0xfc00U);

template<typename T>
inline constexpr T LN_LN2 = 0x1.62cp-1f;

template<typename T>
inline constexpr T LN_LN2LOW = 0x1.218p-12f;

template<typename T>
inline constexpr T LN_C2 = -0x1.ffffc8p-2f;
template<>
inline constexpr __fp16 LN_C2<__fp16> = -0x1.ff8p-2f;

template<typename T>
inline constexpr T LN_C3 = 0x1.555d7cp-2f;
template<>
inline constexpr __fp16 LN_C3<__fp16> = 0x1.67p-2f;

template<typename T>
inline constexpr T LN_C4 = -0x1.00187cp-2f;
template<>
inline constexpr __fp16 LN_C4<__fp16> = -0x1.1dp-2f;

template<typename T>
inline constexpr T LN_C5 = 0x1.961348p-3f;
template<>
inline constexpr __fp16 LN_C5<__fp16> = 0.0;

template<typename T>
inline constexpr T LN_C6 = -0x1.4f9934p-3f;
template<>
inline constexpr __fp16 LN_C6<__fp16> = 0.0;

template<typename T>
inline constexpr T LN_C7 = 0x1.5a9aa2p-3f;
template<>
inline constexpr __fp16 LN_C7<__fp16> = 0.0;

template<typename T>
inline constexpr T LN_C8 = -0x1.3e737cp-3f;
template<>
inline constexpr __fp16 LN_C8<__fp16> = 0.0;

} // Helpers

template <typename ComputeType>
void Log<ComputeType>::DoStore(const svbool_t &pg, svbool_t &pgRetNan, svbool_t &pgRetInf, svbool_t &pgRetNegInf,
    typename Vector::Vec<ComputeType>::Type &poly, typename Vector::Vec<ComputeType>::Type &out,
    typename Vector::Vec<ComputeType>::Type &n) noexcept
{
    typename Vector::Vec<ComputeType>::Type res = svmla_x(pg, poly, n,
        Helpers::Dup<ComputeType, typename Vector::Vec<ComputeType>::Type>::Do(Helpers::LN_LN2LOW<ComputeType>));
    res = svmla_x(pg, res, n, Helpers::Dup<ComputeType,
        typename Vector::Vec<ComputeType>::Type>::Do(Helpers::LN_LN2<ComputeType>));
    
    res = svsel(pgRetNan,
        Helpers::Reinterpret<typename Helpers::FpToVecIntTypeMap<ComputeType>::UnsignedIntVecType,
        typename Vector::Vec<ComputeType>::Type>::Do(
            Helpers::Dup<typename Helpers::FpToVecIntTypeMap<ComputeType>::UnsignedIntScalarType,
            typename Helpers::FpToVecIntTypeMap<ComputeType>::UnsignedIntVecType>::Do(
                Helpers::LN_NAN_BIT<ComputeType>)),
        res);
    res = svsel(pgRetInf,
        Helpers::Reinterpret<typename Helpers::FpToVecIntTypeMap<ComputeType>::UnsignedIntVecType,
        typename Vector::Vec<ComputeType>::Type>::Do(
            Helpers::Dup<typename Helpers::FpToVecIntTypeMap<ComputeType>::UnsignedIntScalarType,
            typename Helpers::FpToVecIntTypeMap<ComputeType>::UnsignedIntVecType>::Do(
                Helpers::LN_INF_BIT<ComputeType>)),
        res);
    res = svsel(pgRetNegInf,
        Helpers::Reinterpret<typename Helpers::FpToVecIntTypeMap<ComputeType>::UnsignedIntVecType,
        typename Vector::Vec<ComputeType>::Type>::Do(
            Helpers::Dup<typename Helpers::FpToVecIntTypeMap<ComputeType>::UnsignedIntScalarType,
            typename Helpers::FpToVecIntTypeMap<ComputeType>::UnsignedIntVecType>::Do(
                Helpers::LN_NEG_INF_BIT<ComputeType>)),
        res);
    out = res;
}

template <typename ComputeType>
void Log<ComputeType>::Do(const svbool_t &pg, typename Vector::Vec<ComputeType>::Type &in,
    typename Vector::Vec<ComputeType>::Type &out) noexcept
{
    typename Helpers::FpToVecIntTypeMap<ComputeType>::UnsignedIntVecType ux =
        Helpers::Reinterpret<typename Vector::Vec<ComputeType>::Type,
        typename Helpers::FpToVecIntTypeMap<ComputeType>::UnsignedIntVecType>::Do(in);
    svbool_t pgRetNan = svcmplt(pg, in, Helpers::LN_ZERO<ComputeType>);
    svbool_t pgRetInf = svcmpeq(pg, in, Helpers::LN_INF_BIT<ComputeType>);
    svbool_t pgRetNegInf = svcmpeq(pg, in, Helpers::LN_ZERO<ComputeType>);

    // x = m * 2^n, 0x1.5p-1 <= m < 0x1.5p0
    typename Helpers::FpToVecIntTypeMap<ComputeType>::UnsignedIntVecType uxSubLnfOffset =
        svsub_x(pg, ux, Helpers::LN_OFFSET<ComputeType>);
    typename Helpers::FpToVecIntTypeMap<ComputeType>::SignedIntVecType sn =
        svasr_x(pg, Helpers::Reinterpret<typename Helpers::FpToVecIntTypeMap<ComputeType>::UnsignedIntVecType,
        typename Helpers::FpToVecIntTypeMap<ComputeType>::SignedIntVecType>::Do(uxSubLnfOffset),
        Helpers::LN_MANTISSA_BIT_LEN<ComputeType>);
    typename Vector::Vec<ComputeType>::Type n =
        Helpers::Convert<typename Helpers::FpToVecIntTypeMap<ComputeType>::SignedIntVecType,
        typename Vector::Vec<ComputeType>::Type>::Do(pg, sn);
    typename Helpers::FpToVecIntTypeMap<ComputeType>::UnsignedIntVecType uxSubLnfOffsetMant =
        svand_x(pg, uxSubLnfOffset, Helpers::LN_MANTISSA_MASK<ComputeType>);

    // r = m - 1
    typename Helpers::FpToVecIntTypeMap<ComputeType>::UnsignedIntVecType uxAbsScale =
        svadd_x(pg, uxSubLnfOffsetMant, Helpers::LN_OFFSET<ComputeType>);
    typename Vector::Vec<ComputeType>::Type r = svsub_x(pg,
        Helpers::Reinterpret<typename Helpers::FpToVecIntTypeMap<ComputeType>::UnsignedIntVecType,
        typename Vector::Vec<ComputeType>::Type>::Do(uxAbsScale), Helpers::LN_ONE<ComputeType>);

    typename Vector::Vec<ComputeType>::Type r2 = svmul_x(pg, r, r);
    typename Vector::Vec<ComputeType>::Type poly = svmla_x(pg,
        Helpers::Dup<ComputeType, typename Vector::Vec<ComputeType>::Type>::Do(Helpers::LN_C7<ComputeType>),
        r, Helpers::Dup<ComputeType, typename Vector::Vec<ComputeType>::Type>::Do(Helpers::LN_C8<ComputeType>));
    poly = svmla_x(pg, Helpers::Dup<ComputeType,
        typename Vector::Vec<ComputeType>::Type>::Do(Helpers::LN_C6<ComputeType>), r, poly);
    poly = svmla_x(pg, Helpers::Dup<ComputeType,
        typename Vector::Vec<ComputeType>::Type>::Do(Helpers::LN_C5<ComputeType>), r, poly);
    poly = svmla_x(pg, Helpers::Dup<ComputeType,
        typename Vector::Vec<ComputeType>::Type>::Do(Helpers::LN_C4<ComputeType>), r, poly);
    poly = svmla_x(pg, Helpers::Dup<ComputeType,
        typename Vector::Vec<ComputeType>::Type>::Do(Helpers::LN_C3<ComputeType>), r, poly);
    poly = svmla_x(pg, Helpers::Dup<ComputeType,
        typename Vector::Vec<ComputeType>::Type>::Do(Helpers::LN_C2<ComputeType>), r, poly);
    poly = svmla_x(pg, r, r2, poly);
    DoStore(pg, pgRetNan, pgRetInf, pgRetNegInf, poly, out, n);
}

} // Vector
} // Detail
} // KuDNN

#endif // KUDNN_VECTOR_LOG
