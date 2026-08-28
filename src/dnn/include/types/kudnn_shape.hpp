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

#ifndef KUDNN_SHAPE_HPP
#define KUDNN_SHAPE_HPP

#include <array>
#include <initializer_list>

#include "service/kudnn_service.hpp"
#include "types/kudnn_types.hpp"
#include "service/kudnn_exception.hpp"

namespace KuDNN {

constexpr SizeType MAX_DIMS = 5;

// This class designed to specify multidimensional data shape
struct Shape final {
    using SizeType = ::KuDNN::SizeType;
    static constexpr SizeType NUM_MAX_DIMENSIONS = MAX_DIMS;

    Shape() noexcept : dimsArray{}, numDims{0}
    {
        for (SizeType i = 0; i < NUM_MAX_DIMENSIONS; ++i) {
            dimsArray[i] = 0;
        }
    }

    template <typename T>
    Shape(T *ptr, const SizeType size) noexcept(false) : numDims(size)
    {
        if (ptr == nullptr) {
            throw Service::LogicError("Shape: ptr is nullptr");
        }
        CheckNumDims(numDims);
        for (SizeType i = 0; i < numDims; ++i) {
            dimsArray[i] = ptr[i];
        }
        for (SizeType i = numDims; i < NUM_MAX_DIMENSIONS; ++i) {
            dimsArray[i] = 0;
        }
    }
    template <typename... Ts>
    Shape(Ts... dims) noexcept(false) : dimsArray{{static_cast<SizeType>(dims)...}}, numDims{sizeof...(dims)}
    {
        CheckNumDims(numDims);
    }
    Shape(const std::initializer_list<SizeType>& list) noexcept(false)
    {
        ResetShape(list);
    }

    template <typename... Ts>
    Shape& ResetShape(Ts... dims) noexcept(false)
    {
        CheckNumDims(sizeof...(dims));
        numDims = sizeof...(dims);
        dimsArray = {static_cast<SizeType>(dims)...};
        for (SizeType i = numDims; i < NUM_MAX_DIMENSIONS; ++i) {
            dimsArray[i] = 0;
        }
        return *this;
    }
    Shape& ResetShape(const std::initializer_list<SizeType>& list) noexcept(false)
    {
        CheckNumDims(list.size());
        numDims = list.size();
        for (SizeType i = 0; i < numDims; ++i) {
            dimsArray[i] = *(list.begin() + i);
        }
        for (SizeType i = numDims; i < NUM_MAX_DIMENSIONS; ++i) {
            dimsArray[i] = 0;
        }
        return *this;
    }
    template <typename T>
    Shape& ResetShape(T *ptr, const SizeType size) noexcept(false)
    {
        if (ptr == nullptr) {
            throw Service::LogicError("Shape: ptr is nullptr");
        }
        CheckNumDims(size);
        numDims = size;
        for (SizeType i = 0; i < numDims; ++i) {
            dimsArray[i] = ptr[i];
        }
        for (SizeType i = numDims; i < NUM_MAX_DIMENSIONS; ++i) {
            dimsArray[i] = 0;
        }
        return *this;
    }

    Shape& operator+=(const Shape &adder) noexcept(false)
    {
        if (adder.GetNumDims() !=  this->GetNumDims()) {
            throw Service::LogicError("Shape: different size of base and adder shapes");
        }
        for (SizeType i = 0; i < adder.GetNumDims(); ++i) {
            this->operator[](i) += adder[i];
        }
        return *this;
    }

    Shape operator+(const Shape &adder) const noexcept(false)
    {
        if (adder.GetNumDims() != this->GetNumDims()) {
            throw Service::LogicError("Shape: different size of base and adder shapes");
        }
        std::array<SizeType, NUM_MAX_DIMENSIONS> tmp;
        for (SizeType i = 0; i < adder.GetNumDims(); ++i) {
            tmp[i] = this->operator[](i) + adder[i];
        }
        return Shape(tmp.data(), this->GetNumDims());
    }

    SizeType operator[](SizeType id) const noexcept(false)
    {
        if (id >= numDims) {
            throw Service::LogicError("Shape: index >= num_dims");
        }
        return dimsArray[id];
    }

    SizeType& operator[](SizeType id) noexcept(false)
    {
        if (id >= numDims) {
            throw Service::LogicError("Shape: index >= num_dims");
        }
        return dimsArray[id];
    }

    SizeType GetNumDims() const noexcept
    {
        return numDims;
    }

    SizeType GetTotalDimsSize() const noexcept(false)
    {
        if (Service::WillIntMultOverflow(dimsArray.begin(), dimsArray.begin() + numDims)) {
            throw Service::LogicError("Shape: computing total size will cause overflow");
        }
        SizeType accum = 1;
        for (SizeType i = 0; i < numDims; ++i) {
            accum *= dimsArray[i];
        }
        return accum;
    }

private:
    void CheckNumDims(SizeType nDims) const noexcept(false)
    {
        if (nDims > NUM_MAX_DIMENSIONS) {
            throw Service::LogicError("Shape: dims is greater than NUM_MAX_DIMENSIONS");
        }
    }
    std::array<SizeType, NUM_MAX_DIMENSIONS> dimsArray;
    SizeType numDims;
};

inline bool operator==(const Shape &lhs, const Shape &rhs) noexcept
{
    if (lhs.GetNumDims() == rhs.GetNumDims()) {
        for (SizeType i = 0; i < lhs.GetNumDims(); ++i) {
            if (lhs[i] != rhs[i]) {
                return false;
            }
        }
        return true;
    }
    return false;
}

inline bool operator!=(const Shape &lhs, const Shape &rhs) noexcept
{
    return !(lhs == rhs);
}

} // KuDNN

#endif // KUDNN_SHAPE_HPP
