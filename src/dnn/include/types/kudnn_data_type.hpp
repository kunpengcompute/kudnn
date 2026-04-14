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

#ifndef KUDNN_DATATYPE_HPP
#define KUDNN_DATATYPE_HPP

#include <cstddef>
#include <cstdint>

#include "service/kudnn_exception.hpp"

#define KUDNN_API_PUBLIC __attribute__((visibility("default")))

namespace KuDNN {

using SizeType = std::size_t;
using IntType = std::int64_t;

namespace Element {

// enumeration of KuDNN supported data types
enum class TypeT {
    UNDEFINED = 0,
    F32,
    F16,
    BF16,
    S32,
    S8,
    U8
};

// This class is designed to provide runtime info about data type.
struct Type final {
public:
    using SizeType = ::KuDNN::SizeType;
    Type(const TypeT& t) noexcept(false) : type{t}, size(0)
    {
        switch (type) {
            case TypeT::F32: {
                size = sizeof(float);
                break;
            }
            case TypeT::F16: {
                size = sizeof(__fp16);
                break;
            }
            case TypeT::BF16: {
                size = sizeof(__bf16);
                break;
            }
            case TypeT::S32: {
                size = sizeof(std::int32_t);
                break;
            }
            case TypeT::S8: {
                size = sizeof(std::int8_t);
                break;
            }
            case TypeT::U8: {
                size = sizeof(std::uint8_t);
                break;
            }
            default: {}
        }
        if (type == TypeT::UNDEFINED) {
            throw Service::LogicError {"Type: unsupported data type"};
        }
    }
    SizeType GetSize() const noexcept
    {
        return size;
    }
    SizeType GetBitwidth() const noexcept
    {
        return (GetSize() * 8uLL);
    }
    operator TypeT() const noexcept
    {
        return type;
    }
    bool IsFP() const noexcept
    {
        if ((type == TypeT::F32) ||
            (type == TypeT::F16) ||
            (type == TypeT::BF16)) {
            return true;
        } else {
            return false;
        }
    }
    bool IsIntegral() const noexcept
    {
        return !IsFP();
    }
    bool IsSigned() const noexcept
    {
        if (type == TypeT::U8) {
            return false;
        } else {
            return true;
        }
    }
    bool IsUnsigned() const noexcept
    {
        return !IsSigned();
    }
private:
    TypeT type;
    SizeType size;
};

template <typename ElementType>
inline TypeT MatchType()
{
    return TypeT::UNDEFINED;
}

template <>
inline TypeT MatchType<float>()
{
    return TypeT::F32;
}

template <>
inline TypeT MatchType<__fp16>()
{
    return TypeT::F16;
}

template <>
inline TypeT MatchType<__bf16>()
{
    return TypeT::BF16;
}

template <>
inline TypeT MatchType<std::int32_t>()
{
    return TypeT::S32;
}

template <>
inline TypeT MatchType<std::int8_t>()
{
    return TypeT::S8;
}

template <>
inline TypeT MatchType<std::uint8_t>()
{
    return TypeT::U8;
}

inline bool operator==(const Type &lhs, const TypeT &rhs) noexcept
{
    return (static_cast<TypeT>(lhs) == rhs);
}

inline bool operator==(const TypeT &lhs, const Type &rhs) noexcept
{
    return (lhs == static_cast<TypeT>(rhs));
}

inline bool operator==(const Type &lhs, const Type &rhs) noexcept
{
    return (lhs.GetSize() == rhs.GetSize()) &&
           (static_cast<TypeT>(lhs) == (static_cast<TypeT>(rhs)));
}

inline bool operator!=(const Type &lhs, const TypeT &rhs) noexcept
{
    return !(lhs == rhs);
}
inline bool operator!=(const TypeT &lhs, const Type &rhs) noexcept
{
    return !(lhs == rhs);
}
inline bool operator!=(const Type &lhs, const Type &rhs) noexcept
{
    return !(lhs == rhs);
}

} // Element
} // KuDNN

#endif // KUDNN_DATATYPE_HPP
