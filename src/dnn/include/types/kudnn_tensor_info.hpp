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

#ifndef KUDNN_TENSORINFO_HPP
#define KUDNN_TENSORINFO_HPP

#include "types/kudnn_types.hpp"
#include "types/kudnn_data_type.hpp"
#include "types/kudnn_layout.hpp"
#include "types/kudnn_shape.hpp"
#include "service/kudnn_service.hpp"

namespace KuDNN {

constexpr SizeType DIM_1 = 1;
constexpr SizeType DIM_2 = 2;
constexpr SizeType DIM_3 = 3;
constexpr SizeType DIM_4 = 4;
constexpr SizeType DIM_5 = 5;

constexpr SizeType IDX_0 = 0;
constexpr SizeType IDX_1 = 1;
constexpr SizeType IDX_2 = 2;
constexpr SizeType IDX_3 = 3;
constexpr SizeType IDX_4 = 4;
constexpr SizeType IDX_5 = 5;

constexpr IntType NUM_0 = 0;
constexpr IntType NUM_1 = 1;
constexpr IntType NUM_2 = 2;
constexpr IntType NUM_3 = 3;
constexpr IntType NUM_4 = 4;
constexpr IntType NUM_5 = 5;
constexpr IntType NUM_6 = 6;
constexpr IntType NUM_7 = 7;
constexpr IntType NUM_8 = 8;
constexpr IntType NUM_9 = 9;
constexpr IntType NUM_10 = 10;
constexpr IntType NUM_11 = 11;
constexpr IntType NUM_12 = 12;
constexpr IntType NUM_13 = 13;
constexpr IntType NUM_14 = 14;
constexpr IntType NUM_15 = 15;
constexpr IntType NUM_16 = 16;
constexpr IntType NUM_17 = 17;
constexpr IntType NUM_18 = 18;
constexpr IntType NUM_19 = 19;
constexpr IntType NUM_20 = 20;
constexpr IntType NUM_21 = 21;
constexpr IntType NUM_22 = 22;
constexpr IntType NUM_23 = 23;
constexpr IntType NUM_24 = 24;
constexpr IntType NUM_25 = 25;
constexpr IntType NUM_26 = 26;
constexpr IntType NUM_27 = 27;
constexpr IntType NUM_28 = 28;
constexpr IntType NUM_29 = 29;
constexpr IntType NUM_30 = 30;
constexpr IntType NUM_31 = 31;
constexpr IntType NUM_32 = 32;
constexpr IntType NUM_33 = 33;
constexpr IntType NUM_34 = 34;
constexpr IntType NUM_35 = 35;
constexpr IntType NUM_36 = 36;
constexpr IntType NUM_37 = 37;
constexpr IntType NUM_38 = 38;
constexpr IntType NUM_39 = 39;
constexpr IntType NUM_40 = 40;
constexpr IntType NUM_41 = 41;
constexpr IntType NUM_42 = 42;
constexpr IntType NUM_43 = 43;
constexpr IntType NUM_44 = 44;
constexpr IntType NUM_45 = 45;
constexpr IntType NUM_46 = 46;
constexpr IntType NUM_47 = 47;
constexpr IntType NUM_48 = 48;
constexpr IntType NUM_49 = 49;
constexpr IntType NUM_50 = 50;
constexpr IntType NUM_51 = 51;
constexpr IntType NUM_52 = 52;
constexpr IntType NUM_53 = 53;
constexpr IntType NUM_54 = 54;
constexpr IntType NUM_55 = 55;
constexpr IntType NUM_56 = 56;
constexpr IntType NUM_57 = 57;
constexpr IntType NUM_58 = 58;
constexpr IntType NUM_59 = 59;
constexpr IntType NUM_60 = 60;
constexpr IntType NUM_61 = 61;
constexpr IntType NUM_62 = 62;
constexpr IntType NUM_63 = 63;
constexpr IntType NUM_64 = 64;
constexpr IntType NUM_65 = 65;
constexpr IntType NUM_66 = 66;
constexpr IntType NUM_67 = 67;
constexpr IntType NUM_68 = 68;
constexpr IntType NUM_69 = 69;
constexpr IntType NUM_70 = 70;
constexpr IntType NUM_71 = 71;
constexpr IntType NUM_72 = 72;
constexpr IntType NUM_73 = 73;
constexpr IntType NUM_74 = 74;
constexpr IntType NUM_75 = 75;
constexpr IntType NUM_76 = 76;
constexpr IntType NUM_77 = 77;
constexpr IntType NUM_78 = 78;
constexpr IntType NUM_79 = 79;
constexpr IntType NUM_80 = 80;
constexpr IntType NUM_81 = 81;
constexpr IntType NUM_82 = 82;
constexpr IntType NUM_83 = 83;
constexpr IntType NUM_84 = 84;
constexpr IntType NUM_85 = 85;
constexpr IntType NUM_86 = 86;
constexpr IntType NUM_87 = 87;
constexpr IntType NUM_88 = 88;
constexpr IntType NUM_89 = 89;
constexpr IntType NUM_120 = 120;
constexpr IntType NUM_132 = 132;
constexpr IntType NUM_1024 = 1024;

// This class is designed to describe multidimensional data: it's type, Layout, Shape and strides
struct KUDNN_API_PUBLIC TensorInfo final {
    using SizeType = ::KuDNN::SizeType;
    TensorInfo(const Shape &shape, const Element::Type &t, const Layout &l) noexcept(false)
        : dims(shape), type(t), layout(l), stridesInfo()
    {
        ValidateElementType(type);
        ValidateLayout(dims, layout);
        SetMinimumRequiredStrides(dims, layout, stridesInfo);
    }

    TensorInfo(const Shape &shape, const Element::Type &t, const Layout &l, const Shape &strides) noexcept(false)
        : dims(shape), type(t), layout(l), stridesInfo(strides)
    {
        ValidateElementType(type);
        // We use 0 strides for dimensions == 1,
        // this done to natively support implicit dimension broadcasting
        stridesInfo = Service::FlushStrides(dims, stridesInfo);
        ValidateDimsAndStrides(dims, layout, strides);
    }

    Shape GetDims() const noexcept
    {
        return dims;
    }

    SizeType GetNumDims() const noexcept
    {
        return dims.GetNumDims();
    }

    SizeType GetTotalTensorSize() const noexcept
    {
        return dims.GetTotalDimsSize();
    }

    Shape GetStrides() const noexcept
    {
        return stridesInfo;
    }

    Element::Type GetType() const noexcept
    {
        return type;
    }

    Layout GetLayout() const noexcept
    {
        return layout;
    }

    bool IsDense() const noexcept
    {
        Shape minStrides = dims;
        SetMinimumRequiredStrides(dims, layout, minStrides);
        return minStrides == stridesInfo;
    }

    bool HasStrideOverFastestDim() const noexcept
    {
        return (stridesInfo[Service::GetIdxForUserLayout(layout, dims.GetNumDims() - 1)] > 1);
    }

    Layout GetStandardABXLayout() const
    {
        switch (dims.GetNumDims()) {
            case DIM_1: {
                return Layout::A;
            }
            case DIM_2: {
                return Layout::AB;
            }
            case DIM_3: {
                return Layout::ABC;
            }
            case DIM_4: {
                return Layout::ABCD;
            }
            case DIM_5: {
                return Layout::ABCDE;
            }
            default: {
                throw Service::LogicError{"Tensor Info: tensor dimensionality is incorrect"};
            }
        }
    }

    Layout GetStandardAXBLayout() const
    {
        switch (dims.GetNumDims()) {
            case DIM_1: {
                return Layout::A;
            }
            case DIM_2: {
                return Layout::AB;
            }
            case DIM_3: {
                return Layout::ACB;
            }
            case DIM_4: {
                return Layout::ACDB;
            }
            case DIM_5: {
                return Layout::ACDEB;
            }
            default: {
                throw Service::LogicError{"Tensor Info: tensor dimensionality is incorrect"};
            }
        }
    }

    Layout GetStandardBXALayout() const
    {
        switch (dims.GetNumDims()) {
            case DIM_1: {
                return Layout::A;
            }
            case DIM_2: {
                return Layout::BA;
            }
            case DIM_3: {
                return Layout::BCA;
            }
            case DIM_4: {
                return Layout::BCDA;
            }
            case DIM_5: {
                return Layout::BCDEA;
            }
            default: {
                throw Service::LogicError{"Tensor Info: tensor dimensionality is incorrect"};
            }
        }
    }

    SizeType GetAxisNumAccordingToLayout(SizeType axisNum) const noexcept(false);

    template <bool areExcetionsDeclined = false>
    static typename std::conditional<areExcetionsDeclined, Status, void>::type
    ValidateElementType(Element::Type parType) noexcept(areExcetionsDeclined);

    template <bool areExcetionsDeclined = false>
    static typename std::conditional<areExcetionsDeclined, Status, void>::type
    ValidateLayout(Shape parDims, Layout parLayout) noexcept(areExcetionsDeclined);

    template <bool areExcetionsDeclined = false>
    static typename std::conditional<areExcetionsDeclined, Status, void>::type
    SetMinimumRequiredStrides(Shape parDims, Layout parLayout, Shape &parStrides) noexcept(areExcetionsDeclined);

    template <bool areExcetionsDeclined = false>
    static typename std::conditional<areExcetionsDeclined, Status, void>::type
    ValidateDimsAndStrides(Shape parDims, Layout parLayout, Shape parStrides) noexcept(areExcetionsDeclined);

private:
    Shape dims;
    Element::Type type;
    Layout layout;
    Shape stridesInfo;
};

inline bool operator==(const TensorInfo &lhs, const TensorInfo &rhs) noexcept
{
    return (lhs.GetDims() == rhs.GetDims()) &&
           (static_cast<Element::TypeT>(lhs.GetType()) == static_cast<Element::TypeT>(rhs.GetType())) &&
           (lhs.GetLayout() == rhs.GetLayout()) && (lhs.GetStrides() == rhs.GetStrides());
}

inline bool operator!=(const TensorInfo &lhs, const TensorInfo &rhs) noexcept
{
    return !(lhs == rhs);
}

} // namespace KuDNN

#endif // KUDNN_TENSORINFO_HPP
