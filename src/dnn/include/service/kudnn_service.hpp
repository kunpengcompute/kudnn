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

#ifndef KUDNN_SERVICE_HPP
#define KUDNN_SERVICE_HPP

#include <type_traits>
#include <limits>
#include <cstddef>

#include "types/kudnn_types.hpp"
#include "types/kudnn_data_type.hpp"
#include "types/kudnn_layout.hpp"
#include "service/kudnn_exception.hpp"

namespace KuDNN {
namespace Service {

constexpr SizeType ALIGNMENT = 128;

KUDNN_API_PUBLIC SizeType GetIdxForUserLayout(Layout l, SizeType idx) noexcept(false);

KUDNN_API_PUBLIC Shape GetShapeAccordingToLayout(const Shape &sh, Layout parLayout) noexcept;

KUDNN_API_PUBLIC Shape FlushStrides(const Shape &dims, const Shape &strides) noexcept(false);

template <typename...>
struct IsAllSame : std::false_type {};

template <>
struct IsAllSame<> : std::true_type {};

template <typename T>
struct IsAllSame<T> : std::true_type {};

template <typename T>
struct IsAllSame<T, T> : std::true_type {};

template <typename T, typename... U>
struct IsAllSame<T, T, U...> : IsAllSame<T, U...> {};

template <
    typename IntType,
    typename std::enable_if<std::is_integral<typename std::remove_reference<IntType>::type>::value, bool>::type = true>
bool WillIntMultOverflow(IntType a, IntType b) noexcept
{
    if ((a == 0) || (b == 0)) {
        return false;
    } else {
        volatile IntType mul = a * b;
        return (mul / a) != b;
    }
}

template <typename IntType, typename... Args,
          typename std::enable_if<IsAllSame<IntType, Args...>::value, bool>::type = true>
bool WillIntMultOverflow(IntType a, IntType b, Args... args) noexcept
{
    if (WillIntMultOverflow(a, b)) {
        return true;
    } else {
        return WillIntMultOverflow(a * b, args...);
    }
}

template <typename ForwardIt,
          typename std::enable_if<std::is_integral<typename std::iterator_traits<ForwardIt>::value_type>::value,
                                  bool>::type = true>
bool WillIntMultOverflow(ForwardIt begin, ForwardIt end) noexcept
{
    typename std::iterator_traits<ForwardIt>::value_type mult = 1;
    for (auto &&it = begin; it != end; ++it) {
        if (WillIntMultOverflow(mult, *it)) {
            return true;
        }
        mult *= *it;
    }
    return false;
}

// User may use allocation/deallocation functions directly or via allocator class
KUDNN_API_PUBLIC void *AlignedAlloc(SizeType n, SizeType alignment = ALIGNMENT) noexcept(false);
KUDNN_API_PUBLIC void Deallocate(void *p, SizeType n = 0) noexcept;

// This class represents an allocator which user may use in order to obtain aligned memory
// and pass it inside KuDNN functions. To be compatible with std::allocator all member names are in snake_case
template <typename T, SizeType alignment = ALIGNMENT>
struct AlignedAllocator {
    using value_type = T;
    using pointer = T *;
    using const_pointer = const T *;
    using reference = T &;
    using const_reference = const T &;
    using size_type = ::KuDNN::SizeType;
    using difference_type = std::ptrdiff_t;

    template <typename U>
    struct rebind {
        using other = AlignedAllocator<U, alignment>;
    };

    T *allocate(SizeType n) const noexcept(false)
    {
        if (n > std::numeric_limits<SizeType>::max() / sizeof(T)) {
            throw BadArrayNewLength();
        }
        return static_cast<T *>(AlignedAlloc(n * sizeof(T), alignment));
    }

    void deallocate(T *p, SizeType n = 0) const noexcept
    {
        ::KuDNN::Service::Deallocate(p, n);
    }
    virtual ~AlignedAllocator() = default;
};

template <typename T>
struct Deallocator {
    void operator()(T *p, ::KuDNN::SizeType n = 0) const noexcept
    {
        ::KuDNN::Service::Deallocate(p, n);
    }
};

template <typename T, SizeType alignment_1, typename U, SizeType alignment_2>
constexpr bool operator==(const AlignedAllocator<T, alignment_1> &, const AlignedAllocator<U, alignment_2> &) noexcept
{
    return (alignment_1 == alignment_2) && std::is_same<T, U>::value;
}
template <typename T, SizeType alignment_1, typename U, SizeType alignment_2>
constexpr bool operator!=(const AlignedAllocator<T, alignment_1> &lhs,
                          const AlignedAllocator<U, alignment_2> &rhs) noexcept
{
    return !(lhs == rhs);
}

} // namespace Service
} // namespace KuDNN

#endif // KUDNN_SERVICE_HPP
