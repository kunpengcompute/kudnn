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

#ifndef KUDNN_VECTOR_INIT_PREDICATE
#define KUDNN_VECTOR_INIT_PREDICATE

#if !defined(__clang__) && (defined(KUDNN_ASAN) || defined(KUDNN_DEBUG))
#pragma GCC optimize("O0")
#endif

#include <cstdint>
#include <arm_sve.h>

#include "kudnn.hpp"

#include "internal_funcs.hpp"
#include "vector_length.hpp"

namespace KuDNN {
namespace Detail {
namespace Vector {

template <>
struct InitPGN<float, 1> final {
    static void Do(svbool_t &pg0, svbool_t &pg1, svbool_t &pg2, svbool_t &pg3, SizeType loopVar,
                   SizeType loopMax) noexcept
    {
        pg0 = svwhilelt_b32_s64(loopVar, loopMax);
        Service::Unused(pg1, pg2, pg3);
    }
};

template <>
struct InitPGN<float, 2> final { // 2 means that initializing two time for one loopVar
    static void Do(svbool_t &pg0, svbool_t &pg1, svbool_t &pg2, svbool_t &pg3, SizeType loopVar,
                   SizeType loopMax) noexcept
    {
        pg0 = svwhilelt_b32_s64(loopVar, loopMax);
        pg1 = svwhilelt_b32_s64(loopVar + Vector::Length<float>(), loopMax);
        Service::Unused(pg2, pg3);
    }
};

template <>
struct InitPGN<float, 4> final { // 4 means that initializing four time for one loopVar
    static void Do(svbool_t &pg0, svbool_t &pg1, svbool_t &pg2, svbool_t &pg3, SizeType loopVar,
                   SizeType loopMax) noexcept
    {
        pg0 = svwhilelt_b32_s64(loopVar, loopMax);
        loopVar += Vector::Length<float>();
        pg1 = svwhilelt_b32_s64(loopVar, loopMax);
        loopVar += Vector::Length<float>();
        pg2 = svwhilelt_b32_s64(loopVar, loopMax);
        loopVar += Vector::Length<float>();
        pg3 = svwhilelt_b32_s64(loopVar, loopMax);
    }
};

template <>
struct InitPGN<__fp16, 1> final {
    static void Do(svbool_t &pg0, svbool_t &pg1, svbool_t &pg2, svbool_t &pg3, SizeType loopVar,
                   SizeType loopMax) noexcept
    {
        pg0 = svwhilelt_b16_s64(loopVar, loopMax);
        Service::Unused(pg1, pg2, pg3);
    }
};

template <>
struct InitPGN<__fp16, 2> final { // 2 means that initializing two time for one loopVar
    static void Do(svbool_t &pg0, svbool_t &pg1, svbool_t &pg2, svbool_t &pg3, SizeType loopVar,
                   SizeType loopMax) noexcept
    {
        pg0 = svwhilelt_b16_s64(loopVar, loopMax);
        loopVar += Vector::Length<__fp16>();
        pg1 = svwhilelt_b16_s64(loopVar, loopMax);
        Service::Unused(pg2, pg3);
    }
};

template <>
struct InitPGN<__bf16, 1> final {
    static void Do(svbool_t &pg0, svbool_t &pg1, svbool_t &pg2, svbool_t &pg3, SizeType loopVar,
                   SizeType loopMax) noexcept
    {
        pg0 = svwhilelt_b16_s64(loopVar, loopMax);
        Service::Unused(pg1, pg2, pg3);
    }
};

template <>
struct InitPGN<__bf16, 2> final { // 2 means that initializing two time for one loopVar
    static void Do(svbool_t &pg0, svbool_t &pg1, svbool_t &pg2, svbool_t &pg3, SizeType loopVar,
                   SizeType loopMax) noexcept
    {
        pg0 = svwhilelt_b16_s64(loopVar, loopMax);
        loopVar += Vector::Length<__bf16>();
        pg1 = svwhilelt_b16_s64(loopVar, loopMax);
        Service::Unused(pg2, pg3);
    }
};

template <>
struct InitPGN<std::int8_t, 1> final {
    static void Do(svbool_t &pg0, svbool_t &pg1, svbool_t &pg2, svbool_t &pg3, SizeType loopVar,
                   SizeType loopMax) noexcept
    {
        pg0 = svwhilelt_b8_s64(loopVar, loopMax);
        Service::Unused(pg1, pg2, pg3);
    }
};

template <>
struct InitPGN<std::uint8_t, 1> final {
    static void Do(svbool_t &pg0, svbool_t &pg1, svbool_t &pg2, svbool_t &pg3, SizeType loopVar,
                   SizeType loopMax) noexcept
    {
        pg0 = svwhilelt_b8_s64(loopVar, loopMax);
        Service::Unused(pg1, pg2, pg3);
    }
};

template <>
struct InitPGN<std::int32_t, 1> final {
    static void Do(svbool_t &pg0, svbool_t &pg1, svbool_t &pg2, svbool_t &pg3, SizeType loopVar,
                   SizeType loopMax) noexcept
    {
        pg0 = svwhilelt_b32_s64(loopVar, loopMax);
        Service::Unused(pg1, pg2, pg3);
    }
};

template <>
struct InitPGN<std::int32_t, 2> final { // 2 means that initializing two time for one loopVar
    static void Do(svbool_t &pg0, svbool_t &pg1, svbool_t &pg2, svbool_t &pg3, SizeType loopVar,
                   SizeType loopMax) noexcept
    {
        pg0 = svwhilelt_b32_s64(loopVar, loopMax);
        pg1 = svwhilelt_b32_s64(loopVar + Vector::Length<std::int32_t>(), loopMax);
        Service::Unused(pg2, pg3);
    }
};

template <>
struct InitPGN<std::int32_t, 4> final { // 4 means that initializing four time for one loopVar
    static void Do(svbool_t &pg0, svbool_t &pg1, svbool_t &pg2, svbool_t &pg3, SizeType loopVar,
                   SizeType loopMax) noexcept
    {
        pg0 = svwhilelt_b32_s64(loopVar, loopMax);
        loopVar += Vector::Length<std::int32_t>();
        pg1 = svwhilelt_b32_s64(loopVar, loopMax);
        loopVar += Vector::Length<std::int32_t>();
        pg2 = svwhilelt_b32_s64(loopVar, loopMax);
        loopVar += Vector::Length<std::int32_t>();
        pg3 = svwhilelt_b32_s64(loopVar, loopMax);
    }
};

} // namespace Vector
} // namespace Detail
} // namespace KuDNN

#endif // KUDNN_VECTOR_INIT_PREDICATE
