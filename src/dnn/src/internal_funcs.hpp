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

#ifndef KUDNN_INTERNAL_FUNCS
#define KUDNN_INTERNAL_FUNCS

#include <string>
#include <ostream>
#include <algorithm>

#include "kudnn.hpp"

#define KUDNN_MODULO_4(n) ((n) & 0x3)

#define CONSTRUCTOR __attribute__((constructor))
#define DESTRUCTOR __attribute__((destructor))
// in case we have template declaration and full specialization CMake option isn't enough to hide names
#define KUDNN_FORCE_INTERNAL_API __attribute__ ((visibility("hidden")))
#ifndef KUDNN_DEBUG
#define KUDNN_INLINE __attribute__((always_inline))
#else
#define KUDNN_INLINE
#endif

namespace KuDNN {

namespace Math {

template <typename T1, typename T2 = T1>
constexpr std::enable_if_t<std::is_integral_v<T1> && std::is_integral_v<T2>, T1> CeilDiv(T1 a, T2 b) noexcept
{
    static_assert(std::is_integral_v<T1> && std::is_integral_v<T2>, "CeilDiv: T1 and T2 should be integral data types");
    return (a + b - 1) / b;
}

template <typename T>
constexpr T Log2(T in) noexcept(false)
{
    static_assert(std::is_integral_v<T>, "T should be integral data type");
    if (in <= 0) {
        throw Service::LogicError{"Log2 of non-positive value can't be computed"};
    }
    T ret = 0;
    while ((in / NUM_2) > 0) {
        in /= NUM_2;
        ++ret;
    }
    return ret;
}

} // namespace Math

enum class ARCH {
    GENERIC = 0,
    NEON,
    VectorComputation,
    MatrixComputation
};

constexpr SizeType DEFAULT_MAX_NUM_THREADS = 128;
::std::ostream& operator << (::std::ostream &os, const Shape &shape);
::std::ostream& operator << (::std::ostream &os, const TensorInfo &info);
::std::ostream& operator << (::std::ostream &os, const Element::TypeT &type);
::std::ostream& operator << (::std::ostream &os, const Element::Type &type);
::std::ostream& operator << (::std::ostream &os, const Layout &layout);

struct Parallelizable {
    Parallelizable() : isInitialized(false)
    {}
    Parallelizable& Initialize(SizeType ont, Shape bd)
    {
        isInitialized = true;
        optNumThreads = ont;
        batchedDims = bd;
        return *this;
    }
    Shape GetBatchedDims() const noexcept(false)
    {
        if (!isInitialized) {
            throw KuDNN::Service::LogicError{"Parallel part wasn't initialized"};
        } else {
            return batchedDims;
        }
    }
    SizeType GetOptNumThreads(SizeType maxThreads) const noexcept(false)
    {
        if (!isInitialized) {
            throw KuDNN::Service::LogicError{"Parallel part wasn't initialized"};
        } else {
            return std::min(optNumThreads, maxThreads);
        }
    }
protected:
    void ResetDimsThreads(const Shape& dims, SizeType threads) noexcept(false)
    {
        batchedDims = dims;
        optNumThreads = threads;
    }
private:
    bool isInitialized;
    SizeType optNumThreads;
    Shape batchedDims;
};

namespace Service {

template <typename T>
struct Bridge {};
 
template <typename T>
struct Bridge<T *> {
    using VoidType = void *;
    using ByteType = std::byte *;
};
 
template <typename T>
struct Bridge<const T *> {
    using VoidType = const void *;
    using ByteType = const std::byte *;
};
 
// offset in bytes
template <typename TargetType, typename InitialType>
constexpr std::enable_if_t<std::is_pointer_v<InitialType> && std::is_pointer_v<TargetType>, TargetType> AddOffset(
    InitialType ptr, int offset) noexcept
{
    return static_cast<TargetType>(
        static_cast<typename Bridge<InitialType>::VoidType>(
            static_cast<typename Bridge<InitialType>::ByteType>(
                static_cast<typename Bridge<InitialType>::VoidType>(ptr)
            ) + offset
        )
    );
}

template <typename T>
KUDNN_INLINE inline void Unused(T &&t) noexcept
{
    static_cast<void>(t);
}

template <typename T, typename ... Args>
KUDNN_INLINE constexpr inline void Unused(T &&t, Args && ... args) noexcept
{
    Unused(std::forward<T>(t));
    Unused(std::forward<Args>(args)...);
}

Shape BroadcastTo5D(const Shape &sh, bool isDims = true) noexcept(false);
Shape BroadcastToNCDHW(const Shape &sh, bool isDims = true) noexcept(false);
void ThrowOnStatus(Status s, const std::string& funcName) noexcept(false);
SizeType ArrayProduct(const Shape &arr, SizeType end, SizeType start) noexcept;
void ConvertFp16ToFp32(const __fp16 *src, float *dst, SizeType size);
void ConvertBf16ToFp32(const __bf16 *src, float *dst, SizeType size);
void ConvertS32ToFp32(const int32_t *src, float *dst, SizeType size);
void ConvertS8ToFp32(const int8_t *src, float *dst, SizeType size);
void ConvertU8ToFp32(const uint8_t *src, float *dst, SizeType size);
void ConvertFp32ToFp16(const float *src, __fp16 *dst, SizeType size);
void ConvertFp32ToBf16(const float *src, __bf16 *dst, SizeType size);
void ConvertFp32ToS32(const float *src, int32_t *dst, SizeType size);
void ConvertFp32ToS8(const float *src, int8_t *dst, SizeType size);
void ConvertFp32ToU8(const float *src, uint8_t *dst, SizeType size);

void ConvertFp16ToS32(const __fp16 *src, std::int32_t *dst, SizeType size);
void ConvertBf16ToS32(const __bf16 *src, std::int32_t *dst, SizeType size);
void ConvertS8ToS32(const int8_t *src, std::int32_t *dst, SizeType size);
void ConvertU8ToS32(const uint8_t *src, std::int32_t *dst, SizeType size);
void ConvertS32ToFp16(const std::int32_t *src, __fp16 *dst, SizeType size);
void ConvertS32ToBf16(const std::int32_t *src, __bf16 *dst, SizeType size);
void ConvertS32ToS8(const std::int32_t *src, int8_t *dst, SizeType size);
void ConvertS32ToU8(const std::int32_t *src, uint8_t *dst, SizeType size);

inline constexpr SizeType UNROLL_1 = 1;
inline constexpr SizeType UNROLL_2 = 2;
inline constexpr SizeType UNROLL_3 = 3;
inline constexpr SizeType UNROLL_4 = 4;
inline constexpr SizeType UNROLL_5 = 5;
inline constexpr SizeType UNROLL_6 = 6;
inline constexpr SizeType UNROLL_7 = 7;
inline constexpr SizeType MAX_UNROLL = 8;

struct Unroll final {
    template <SizeType UnrollFactor, typename KernelFunctor, typename ... Args>
    static void Do(SizeType idx, SizeType loopLim, KernelFunctor &&functor, Args && ... args) noexcept
    {
        Unused(functor);
        static_assert((UnrollFactor >= 0) && (UnrollFactor <= MAX_UNROLL), "Unroll factor should be in range [1, 8]");
        KernelFunctor::Do(idx, loopLim, std::forward<Args>(args)...);
        if constexpr (UnrollFactor > UNROLL_1) {
            KernelFunctor::Do(idx + KernelFunctor::LoopInc(), loopLim, std::forward<Args>(args)...);
            if constexpr (UnrollFactor > UNROLL_2) {
                KernelFunctor::Do(idx + UNROLL_2 * KernelFunctor::LoopInc(), loopLim, std::forward<Args>(args)...);
                if constexpr (UnrollFactor > UNROLL_3) {
                    KernelFunctor::Do(idx + UNROLL_3 * KernelFunctor::LoopInc(), loopLim, std::forward<Args>(args)...);
                    if constexpr (UnrollFactor > UNROLL_4) {
                        KernelFunctor::Do(idx +
                            UNROLL_4 * KernelFunctor::LoopInc(), loopLim, std::forward<Args>(args)...);
                        if constexpr (UnrollFactor > UNROLL_5) {
                            KernelFunctor::Do(idx +
                                UNROLL_5 * KernelFunctor::LoopInc(), loopLim, std::forward<Args>(args)...);
                            if constexpr (UnrollFactor > UNROLL_6) {
                                KernelFunctor::Do(idx +
                                    UNROLL_6 * KernelFunctor::LoopInc(), loopLim, std::forward<Args>(args)...);
                                if constexpr (UnrollFactor > UNROLL_7) {
                                    KernelFunctor::Do(idx +
                                        UNROLL_7 * KernelFunctor::LoopInc(), loopLim, std::forward<Args>(args)...);
                                }
                            }
                        }
                    }
                }
            }
        }
    }
};

struct UnrollFor final {
    template <SizeType UnrollFactor, typename KernelFunctor, typename ... Args>
    static void Do(SizeType loopStart, SizeType loopLim, KernelFunctor &&functor, Args && ... args) noexcept
    {
        SizeType i = loopStart;
        for (i = loopStart; (i + UnrollFactor * KernelFunctor::LoopInc()) <= loopLim;
            i += UnrollFactor * KernelFunctor::LoopInc()) {
            Unroll::Do<UnrollFactor>(i, loopLim, std::forward<KernelFunctor>(functor), std::forward<Args>(args)...);
        }
        for (; i < loopLim; i += KernelFunctor::LoopInc()) {
            KernelFunctor::Do(i, loopLim, std::forward<Args>(args)...);
        }
    }
    template <SizeType UnrollFactor, typename KernelFunctor, typename ... Args>
    static void Do(SizeType loopLim, KernelFunctor &&functor, Args && ... args) noexcept
    {
        Do<UnrollFactor>(0, loopLim, std::forward<KernelFunctor>(functor), std::forward<Args>(args)...);
    }
};

template <SizeType N>
struct Forer {
    static_assert((N > 0) && (N <= MAX_DIMS), "N should be in range [1, MAX_DIMS]");
    template <typename KernelFunctor, typename... Args>
    static void Do(const Shape &loopEnd, SizeType multiDimLoopCounter[], KernelFunctor &&func, Args &&... args)
    {
        if constexpr (N == 1) {
            for (SizeType i = 0; i < loopEnd[loopEnd.GetNumDims() - N]; ++i) {
                multiDimLoopCounter[loopEnd.GetNumDims() - N] = i;
                KernelFunctor::Do(multiDimLoopCounter, std::forward<Args>(args)...);
            }
        } else {
            for (SizeType i = 0; i < loopEnd[loopEnd.GetNumDims() - N]; ++i) {
                multiDimLoopCounter[loopEnd.GetNumDims() - N] = i;
                Forer<N - 1>::Do(loopEnd, multiDimLoopCounter,
                    std::forward<KernelFunctor>(func), std::forward<Args>(args)...);
            }
        }
    }
};

struct ForCaller {
    template <typename KernelFunctor, typename... Args>
    static void Do(const Shape &loopEnd, KernelFunctor &&func, Args &&... args) noexcept
    {
        SizeType  multiDimLoopCounter[MAX_DIMS] = {0, 0, 0, 0, 0};
        switch (loopEnd.GetNumDims()) {
            case NUM_1: Forer<NUM_1>::Do(loopEnd, multiDimLoopCounter,
                std::forward<KernelFunctor>(func), std::forward<Args>(args)...); break;
            case NUM_2: Forer<NUM_2>::Do(loopEnd, multiDimLoopCounter,
                std::forward<KernelFunctor>(func), std::forward<Args>(args)...); break;
            case NUM_3: Forer<NUM_3>::Do(loopEnd, multiDimLoopCounter,
                std::forward<KernelFunctor>(func), std::forward<Args>(args)...); break;
            case NUM_4: Forer<NUM_4>::Do(loopEnd, multiDimLoopCounter,
                std::forward<KernelFunctor>(func), std::forward<Args>(args)...); break;
            case NUM_5: Forer<NUM_5>::Do(loopEnd, multiDimLoopCounter,
                std::forward<KernelFunctor>(func), std::forward<Args>(args)...); break;
            default: break;
        }
    }
};

template <typename CompType, typename SecondType, typename ThirdType, typename Runnable, typename ... Args>
inline void ParseType(Runnable &&runnable, Args && ... args)
    noexcept(noexcept(std::declval<Runnable>().template Evaluate<CompType, SecondType, ThirdType>(args ...)))
{
    runnable.template Evaluate<CompType, SecondType, ThirdType>(args ...);
}

template <bool IsSupportedInt, bool HasFP16CompType, typename Runnable, typename ... Args>
inline void ParseType(Element::TypeT compType, Element::TypeT secondType, Element::TypeT thirdType,
    Runnable &&runnable, Args && ... args) noexcept(false)
{
    Unused(compType, secondType);
    using CompType = float;
    if constexpr (IsSupportedInt) {
        if (thirdType == Element::TypeT::F32) {
            ParseType<CompType, float, float>(std::forward<Runnable>(runnable),
                std::forward<Args>(args) ...);
        } else if (thirdType == Element::TypeT::F16) {
            if constexpr (HasFP16CompType) {
                ParseType<__fp16, __fp16, __fp16>(std::forward<Runnable>(runnable),
                    std::forward<Args>(args) ...);
            } else {
                ParseType<CompType, __fp16, __fp16>(std::forward<Runnable>(runnable),
                    std::forward<Args>(args) ...);
            }
        } else if (thirdType == Element::TypeT::BF16) {
            ParseType<CompType, __bf16, __bf16>(std::forward<Runnable>(runnable),
                std::forward<Args>(args) ...);
        }  else if (thirdType == Element::TypeT::S8) {
            ParseType<CompType, std::int8_t, std::int8_t>(std::forward<Runnable>(runnable),
                std::forward<Args>(args) ...);
        } else if (thirdType == Element::TypeT::U8) {
            ParseType<CompType, std::uint8_t, std::uint8_t>(std::forward<Runnable>(runnable),
                std::forward<Args>(args) ...);
        } else if (thirdType == Element::TypeT::S32) {
            ParseType<CompType, std::int32_t, std::int32_t>(std::forward<Runnable>(runnable),
                std::forward<Args>(args) ...);
        } else {
            throw Service::Unsupported {"Unsupported data type"};
        }
    } else {
        if (thirdType == Element::TypeT::F32) {
            ParseType<CompType, float, float>(std::forward<Runnable>(runnable),
                std::forward<Args>(args) ...);
        } else if (thirdType == Element::TypeT::F16) {
            if constexpr (HasFP16CompType) {
                ParseType<__fp16, __fp16, __fp16>(std::forward<Runnable>(runnable),
                    std::forward<Args>(args) ...);
            } else {
                ParseType<CompType, __fp16, __fp16>(std::forward<Runnable>(runnable),
                    std::forward<Args>(args) ...);
            }
        } else if (thirdType == Element::TypeT::BF16) {
            ParseType<CompType, __bf16, __bf16>(std::forward<Runnable>(runnable),
                std::forward<Args>(args) ...);
        } else {
            throw Service::Unsupported {"Unsupported data type"};
        }
    }
}

template <typename CompType, typename SecondType, typename ThirdType, typename ForthType,
    typename Runnable, typename ... Args>
inline void ParseTypeAdvanced(Runnable &&runnable, Args && ... args)
    noexcept(noexcept(std::declval<Runnable>().template Evaluate<CompType, SecondType,
        ThirdType, ForthType>(args ...)))
{
    runnable.template Evaluate<CompType, SecondType, ThirdType, ForthType>(args ...);
}

template <bool HasFP16CompType, typename Runnable, typename ... Args>
inline void ParseTypeAdvanced(Element::TypeT compType, Element::TypeT secondType, Element::TypeT thirdType,
    Element::TypeT forthType, Runnable &&runnable, Args && ... args) noexcept(false)
{
    Unused(compType, secondType, thirdType);
    using CompType = float;
    if (forthType == Element::TypeT::F32) {
        ParseTypeAdvanced<CompType, float, float, float>(std::forward<Runnable>(runnable),
            std::forward<Args>(args) ...);
    } else if (forthType == Element::TypeT::F16) {
        if constexpr (HasFP16CompType) {
            ParseTypeAdvanced<__fp16, __fp16, __fp16, __fp16>(std::forward<Runnable>(runnable),
                std::forward<Args>(args) ...);
        } else {
            ParseTypeAdvanced<CompType, __fp16, __fp16, __fp16>(std::forward<Runnable>(runnable),
                std::forward<Args>(args) ...);
        }
    } else if (forthType == Element::TypeT::BF16) {
        ParseTypeAdvanced<CompType, __bf16, __bf16, __bf16>(std::forward<Runnable>(runnable),
            std::forward<Args>(args) ...);
    } else if (forthType == Element::TypeT::S8) {
        ParseTypeAdvanced<CompType, std::int8_t, std::int8_t, std::int8_t>(std::forward<Runnable>(runnable),
            std::forward<Args>(args) ...);
    } else if (forthType == Element::TypeT::U8) {
        ParseTypeAdvanced<CompType, std::uint8_t, std::uint8_t, std::uint8_t>(std::forward<Runnable>(runnable),
            std::forward<Args>(args) ...);
    } else if (forthType == Element::TypeT::S32) {
        ParseTypeAdvanced<CompType, std::int32_t, std::int32_t, std::int32_t>(std::forward<Runnable>(runnable),
            std::forward<Args>(args) ...);
    } else {
        throw Service::Unsupported {"Unsupported data type"};
    }
}

template <typename T, typename P>
constexpr bool OneOf(const T& val, const P& item)
{
    return val == item;
}
template <typename T, typename P, typename... Args>
constexpr bool OneOf(const T& val, const P& item, const Args& ... itemOthers)
{
    return val == item || OneOf(val, itemOthers...);
}

} // Service

} // KuDNN

#endif // KUDNN_INTERNAL_FUNCS
