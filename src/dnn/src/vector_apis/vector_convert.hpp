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

#ifndef KUDNN_VECTOR_CONVERT
#define KUDNN_VECTOR_CONVERT

#include <limits>
#include <cstdint>
#include <algorithm>
#include <cmath>
#include <arm_sve.h>
#include <arm_neon.h>

#include "kudnn.hpp"

#include "internal_funcs.hpp"

namespace KuDNN {
namespace Detail {
namespace Vector {

template<>
struct ConvertN<float, float, 1> final {
    static void Do(const svbool_t& pg0, const svbool_t &pg1, const svbool_t &pg2,
        const svbool_t &pg3, const typename Vec<float>::Type &in0, const typename Vec<float>::Type &in1,
        const typename Vec<float>::Type &in2, const typename Vec<float>::Type &in3,
        typename Vec<float>::Type &out0, typename Vec<float>::Type &out1,
        typename Vec<float>::Type &out2, typename Vec<float>::Type &out3) noexcept
    {
        out0 = in0;
        Service::Unused(pg0, pg1, pg2, pg3, in1, in2, in3, out1, out2, out3);
    }
};

template<>
struct ConvertN<float, float, 2> final { // 2 means that converts two inputs one time
    static void Do(const svbool_t& pg0, const svbool_t &pg1, const svbool_t &pg2,
        const svbool_t &pg3, const typename Vec<float>::Type &in0, const typename Vec<float>::Type &in1,
        const typename Vec<float>::Type &in2, const typename Vec<float>::Type &in3,
        typename Vec<float>::Type &out0, typename Vec<float>::Type &out1,
        typename Vec<float>::Type &out2, typename Vec<float>::Type &out3) noexcept
    {
        out0 = in0;
        out1 = in1;
        Service::Unused(pg0, pg1, pg2, pg3, in2, in3, out2, out3);
    }
};

template<>
struct ConvertN<float, float, 4> final { // 4 means that converts four inputs one time
    static void Do(const svbool_t& pg0, const svbool_t &pg1, const svbool_t &pg2,
        const svbool_t &pg3, const typename Vec<float>::Type &in0, const typename Vec<float>::Type &in1,
        const typename Vec<float>::Type &in2, const typename Vec<float>::Type &in3,
        typename Vec<float>::Type &out0, typename Vec<float>::Type &out1,
        typename Vec<float>::Type &out2, typename Vec<float>::Type &out3) noexcept
    {
        out0 = in0;
        out1 = in1;
        out2 = in2;
        out3 = in3;
        Service::Unused(pg0, pg1, pg2, pg3);
    }
};

template<>
struct ConvertN<float, __fp16, 2> final { // 2 means that converts two inputs one time
    static void Do(const svbool_t& pg0, const svbool_t &pg1, const svbool_t &pg2,
        const svbool_t &pg3, const typename Vec<float>::Type &in0, const typename Vec<float>::Type &in1,
        const typename Vec<float>::Type &in2, const typename Vec<float>::Type &in3,
        typename Vec<__fp16>::Type &out0, typename Vec<__fp16>::Type &out1,
        typename Vec<__fp16>::Type &out2, typename Vec<__fp16>::Type &out3) noexcept
    {
        svfloat16_t vec0 = svcvt_f16_f32_x(pg0, in0);
        svfloat16_t vec1 = svcvt_f16_f32_x(pg1, in1);
        out0 = svuzp1_f16(vec0, vec1);
        Service::Unused(pg2, pg3, in2, in3, out1, out2, out3);
    }
};

template<>
struct ConvertN<float, std::int32_t, 1> final { // 4 means that converts four inputs one time
    static void Do(const svbool_t& pg0, const svbool_t &pg1, const svbool_t &pg2,
        const svbool_t &pg3, const typename Vec<float>::Type &in0, const typename Vec<float>::Type &in1,
        const typename Vec<float>::Type &in2, const typename Vec<float>::Type &in3,
        typename Vec<std::int32_t>::Type &out0, typename Vec<std::int32_t>::Type &out1,
        typename Vec<std::int32_t>::Type &out2, typename Vec<std::int32_t>::Type &out3) noexcept
    {
        out0 = svcvt_s32_f32_x(pg0, in0);
        Service::Unused(pg0, pg1, pg2, pg3, in1, in2, in3, out1, out2, out3);
    }
};
// double -> *

template <>
struct Convert<double, float> final {
    static float Do(double val) noexcept
    {
        return val;
    }
};

template <>
struct Convert<double, __fp16> final {
    static __fp16 Do(double val) noexcept
    {
        return val;
    }
};

template <>
struct Convert<double, __bf16> final {
    static __bf16 Do(double val) noexcept
    {
        return vcvth_bf16_f32(val);
    }
};

// fp32 -> *

template <>
struct Convert<float, float> final {
    static float Do(const float val) noexcept
    {
        return val;
    }
};

template <>
struct Convert<float, __fp16> final {
    static __fp16 Do(const float val) noexcept
    {
        return static_cast<__fp16>(val);
    }
};

template <>
struct Convert<float, __bf16> final {
    static __bf16 Do(const float val) noexcept
    {
        return vcvth_bf16_f32(val);
    }
};

template <>
struct Convert<float, std::int8_t> final {
    static std::int8_t Do(const float val) noexcept
    {
        return static_cast<std::int8_t>(
            std::nearbyint(std::max<float>(std::min<float>(val, std::numeric_limits<std::int8_t>::max()),
            std::numeric_limits<std::int8_t>::min())));
    }
};

template <>
struct Convert<float, std::uint8_t> final {
    static std::uint8_t Do(const float val) noexcept
    {
        return static_cast<std::uint8_t>(
            std::nearbyint(std::max<float>(std::min<float>(val, std::numeric_limits<std::uint8_t>::max()),
            std::numeric_limits<std::uint8_t>::min())));
    }
};

template <>
struct Convert<float, std::int32_t> final {
    static std::int32_t Do(const float val) noexcept
    {
        return static_cast<std::int32_t>(
            std::nearbyint(std::max<float>(std::min<float>(val,
            static_cast<float>(std::numeric_limits<std::int32_t>::max())),
            static_cast<float>(std::numeric_limits<std::int32_t>::min()))));
    }
};

// fp16 -> *

template <>
struct Convert<__fp16, float> final {
    static float Do(const __fp16 val) noexcept
    {
        return static_cast<float>(val);
    }
};

template <>
struct Convert<__fp16, __fp16> final {
    static __fp16 Do(const __fp16 val) noexcept
    {
        return val;
    }
};

template <>
struct Convert<__fp16, __bf16> final {
    static __bf16 Do(const __fp16 val) noexcept
    {
        return vcvth_bf16_f32(val);
    }
};

template <>
struct Convert<__fp16, std::int8_t> final {
    static std::int8_t Do(const __fp16 val) noexcept
    {
        return static_cast<std::int8_t>(
            std::nearbyint(std::max<__fp16>(std::min<__fp16>(val, std::numeric_limits<std::int8_t>::max()),
            std::numeric_limits<std::int8_t>::min())));
    }
};

template <>
struct Convert<__fp16, std::uint8_t> final {
    static std::uint8_t Do(const __fp16 val) noexcept
    {
        return static_cast<std::uint8_t>(
            std::nearbyint(std::max<__fp16>(std::min<__fp16>(val, std::numeric_limits<std::uint8_t>::max()),
            std::numeric_limits<std::uint8_t>::min())));
    }
};

template <>
struct Convert<__fp16, std::int32_t> final {
    static std::int32_t Do(const __fp16 val) noexcept
    {
        return std::nearbyint(val);
    }
};

// bf16 -> *

template <>
struct Convert<__bf16, float> final {
    static float Do(const __bf16 val) noexcept
    {
        return vcvtah_f32_bf16(val);
    }
};

template <>
struct Convert<__bf16, __fp16> final {
    static __fp16 Do(const __bf16 val) noexcept
    {
        return vcvtah_f32_bf16(val);
    }
};

template <>
struct Convert<__bf16, __bf16> final {
    static __bf16 Do(const __bf16 val) noexcept
    {
        return val;
    }
};

template <>
struct Convert<__bf16, std::int8_t> final {
    static std::int8_t Do(const __bf16 val) noexcept
    {
        return static_cast<std::int8_t>(
            std::nearbyint(std::max<float>(std::min<float>(vcvtah_f32_bf16(val),
            std::numeric_limits<std::int8_t>::max()),
            std::numeric_limits<std::int8_t>::min())));
    }
};

template <>
struct Convert<__bf16, std::uint8_t> final {
    static std::uint8_t Do(const __bf16 val) noexcept
    {
        return static_cast<std::uint8_t>(
            std::nearbyint(std::max<float>(std::min<float>(vcvtah_f32_bf16(val),
            std::numeric_limits<std::uint8_t>::max()),
            std::numeric_limits<std::uint8_t>::min())));
    }
};

template <>
struct Convert<__bf16, std::int32_t> final {
    static std::int32_t Do(const __bf16 val) noexcept
    {
        return std::nearbyint(vcvtah_f32_bf16(val));
    }
};

// s8 -> *

template <>
struct Convert<std::int8_t, float> final {
    static float Do(const std::int8_t val) noexcept
    {
        return static_cast<float>(val);
    }
};

template <>
struct Convert<std::int8_t, __fp16> final {
    static __fp16 Do(const std::int8_t val) noexcept
    {
        return val;
    }
};

template <>
struct Convert<std::int8_t, __bf16> final {
    static __bf16 Do(const std::int8_t val) noexcept
    {
        return vcvth_bf16_f32(val);
    }
};

template <>
struct Convert<std::int8_t, std::int8_t> final {
    static std::int8_t Do(const std::int8_t val) noexcept
    {
        return val;
    }
};

template <>
struct Convert<std::int8_t, std::uint8_t> final {
    static std::uint8_t Do(const std::int8_t val) noexcept
    {
        return (val < 0) ? 0 : val;
    }
};

template <>
struct Convert<std::int8_t, std::int32_t> final {
    static std::int32_t Do(const std::int8_t val) noexcept
    {
        return val;
    }
};

// u8 -> *

template <>
struct Convert<std::uint8_t, float> final {
    static float Do(const std::uint8_t val) noexcept
    {
        return static_cast<float>(val);
    }
};

template <>
struct Convert<std::uint8_t, __fp16> final {
    static __fp16 Do(const std::uint8_t val) noexcept
    {
        return val;
    }
};

template <>
struct Convert<std::uint8_t, __bf16> final {
    static __bf16 Do(const std::uint8_t val) noexcept
    {
        return vcvth_bf16_f32(val);
    }
};

template <>
struct Convert<std::uint8_t, std::int8_t> final {
    static std::int8_t Do(const std::uint8_t val) noexcept
    {
        return std::min(val, static_cast<std::uint8_t>(INT8_MAX));
    }
};

template <>
struct Convert<std::uint8_t, std::uint8_t> final {
    static std::uint8_t Do(const std::uint8_t val) noexcept
    {
        return val;
    }
};

template <>
struct Convert<std::uint8_t, std::int32_t> final {
    static std::int32_t Do(const std::uint8_t val) noexcept
    {
        return val;
    }
};

// s32 -> *

template <>
struct Convert<std::int32_t, float> final {
    static float Do(const std::int32_t val) noexcept
    {
        return static_cast<float>(val);
    }
};

template <>
struct Convert<std::int32_t, __fp16> final {
    static __fp16 Do(const std::int32_t val) noexcept
    {
        return static_cast<__fp16>(val);
    }
};

template <>
struct Convert<std::int32_t, __bf16> final {
    static __bf16 Do(const std::int32_t val) noexcept
    {
        return vcvth_bf16_f32(val);
    }
};

template <>
struct Convert<std::int32_t, std::int8_t> final {
    static std::int8_t Do(const std::int32_t val) noexcept
    {
        return std::max<std::int32_t>(std::min<std::int32_t>(val, std::numeric_limits<std::int8_t>::max()),
            std::numeric_limits<std::int8_t>::min());
    }
};

template <>
struct Convert<std::int32_t, std::uint8_t> final {
    static std::uint8_t Do(const std::int32_t val) noexcept
    {
        return std::max<std::int32_t>(std::min<std::int32_t>(val, std::numeric_limits<std::uint8_t>::max()),
            std::numeric_limits<std::uint8_t>::min());
    }
};

template <>
struct Convert<std::int32_t, std::int32_t> final {
    static std::int32_t Do(const std::int32_t val) noexcept
    {
        return val;
    }
};

} // Vector
} // Detail
} // KuDNN

#endif // KUDNN_VECTOR_CONVERT
