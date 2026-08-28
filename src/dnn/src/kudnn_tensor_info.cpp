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

#include <array>

#include "service/kudnn_err_codes.hpp"
#include "types/kudnn_tensor_info.hpp"

namespace KuDNN {

SizeType TensorInfo::GetAxisNumAccordingToLayout(SizeType axisNum) const noexcept(false)
{
    if (axisNum > dims.GetNumDims()) {
        throw Service::LogicError {"Tensor Info: axisNum > tensor dimensionality"};
    }
    SizeType idxArr[] = {IDX_0, IDX_1, IDX_2, IDX_3, IDX_4};
    Shape idx = Service::GetShapeAccordingToLayout(Shape(idxArr, dims.GetNumDims()), layout);
    SizeType num = 0;
    while (idx[num] != axisNum) { num++; }
    return num;
}

template <bool areExcetionsDeclined>
ReturnType<areExcetionsDeclined> TensorInfo::ValidateElementType(Element::Type
    parType) noexcept(areExcetionsDeclined)
{
    if (parType == Element::TypeT::UNDEFINED) {
        if constexpr (areExcetionsDeclined) {
            return Status::BAD_ARGUMENTS;
        } else {
            throw Service::LogicError {"Tensor Info: Undefined data type of tensor"};
        }
    }

    if constexpr (areExcetionsDeclined) {
        return Status::SUCCESS;
    }
}

template
ReturnType<false> TensorInfo::ValidateElementType<false>(Element::Type
    parType) noexcept(false);

template
ReturnType<true> TensorInfo::ValidateElementType<true>(Element::Type
    parType) noexcept(true);


template <bool areExcetionsDeclined>
ReturnType<areExcetionsDeclined> TensorInfo::ValidateLayout(const Shape& parDims,
    Layout parLayout) noexcept(areExcetionsDeclined)
{
    /*
        KuDNN supports tensors up to 5D, below we consider each dimensionality
        as a separate case, since layout is dependent on dimensionality
    */
    bool isInvalidLayout = false;
    switch (parDims.GetNumDims()) {
        case DIM_1: {
            if (parLayout != Layout::A) {
                isInvalidLayout = true;
            }
            break;
        }
        case DIM_2: {
            if ((parLayout != Layout::AB) && (parLayout != Layout::BA)) {
                isInvalidLayout = true;
            }
            break;
        }
        case DIM_3: {
            bool layoutNotSupported = (parLayout != Layout::ABC) && (parLayout != Layout::ACB) &&
                (parLayout != Layout::BAC) && (parLayout != Layout::BCA) &&
                (parLayout != Layout::CAB) && (parLayout != Layout::CBA);
            if (layoutNotSupported) {
                isInvalidLayout = true;
            }
            break;
        }
        case DIM_4: {
            bool layoutNotSupported = (parLayout != Layout::ABCD) && (parLayout != Layout::ABDC) &&
                (parLayout != Layout::ACBD) && (parLayout != Layout::ACDB) &&
                (parLayout != Layout::ADBC) && (parLayout != Layout::ADCB) &&
                (parLayout != Layout::BACD) && (parLayout != Layout::BCDA) &&
                (parLayout != Layout::CDAB) && (parLayout != Layout::CDBA) &&
                (parLayout != Layout::DCAB);
            if (layoutNotSupported) {
                isInvalidLayout = true;
            }
            break;
        }
        case DIM_5: {
            bool layoutNotSupported = (parLayout != Layout::ABCDE) && (parLayout != Layout::ABCED) &&
                (parLayout != Layout::ABDEC) && (parLayout != Layout::ACBDE) &&
                (parLayout != Layout::ACDEB) && (parLayout != Layout::ADECB) &&
                (parLayout != Layout::BACDE) && (parLayout != Layout::BCDEA) &&
                (parLayout != Layout::CDEAB) && (parLayout != Layout::CDEBA) &&
                (parLayout != Layout::DECAB);
            if (layoutNotSupported) {
                isInvalidLayout = true;
            }
            break;
        }
        default: {
            if constexpr (areExcetionsDeclined) {
                return Status::BAD_ARGUMENTS;
            } else {
                throw Service::LogicError {"Tensor Info: tensor dimensionality is incorrect"};
            }
        }
    }

    if (isInvalidLayout) {
        if constexpr (areExcetionsDeclined) {
            return Status::BAD_ARGUMENTS;
        } else {
            throw Service::LogicError {"Tensor Info: incorrect layout for " + std::to_string(parDims.GetNumDims())
                + "D tensor"};
        }
    }

    if constexpr (areExcetionsDeclined) {
        return Status::SUCCESS;
    }
}

template
ReturnType<false> TensorInfo::ValidateLayout<false>(const Shape& parDims,
    Layout parLayout) noexcept(false);

template
ReturnType<true> TensorInfo::ValidateLayout<true>(const Shape& parDims,
    Layout parLayout) noexcept(true);


template <bool areExcetionsDeclined>
ReturnType<areExcetionsDeclined> TensorInfo::SetMinimumRequiredStrides(const Shape&  parDims,
    Layout parLayout, Shape& parStrides) noexcept(areExcetionsDeclined)
{
    /*
        KuDNN supports tensors up to 5D, below we consider each layout
        as a separate case, since strides is dependent on layout
    */
    switch (parLayout) {
        case Layout::A: {
            parStrides.ResetShape(1);
            break;
        }
        case Layout::AB: {
            parStrides.ResetShape(parDims[IDX_1], 1);
            break;
        }
        case Layout::BA: {
            parStrides.ResetShape(1, parDims[IDX_0]);
            break;
        }
        case Layout::ABC: {
            parStrides.ResetShape(parDims[IDX_1] * parDims[IDX_2], parDims[IDX_2], 1);
            break;
        }
        case Layout::ACB: {
            parStrides.ResetShape(parDims[IDX_1] * parDims[IDX_2], 1, parDims[IDX_1]);
            break;
        }
        case Layout::BAC: {
            parStrides.ResetShape(parDims[IDX_2], parDims[IDX_0] * parDims[IDX_2], 1);
            break;
        }
        case Layout::BCA: {
            parStrides.ResetShape(1, parDims[IDX_0] * parDims[IDX_2], parDims[IDX_0]);
            break;
        }
        case Layout::CAB: {
            parStrides.ResetShape(parDims[IDX_1], 1, parDims[IDX_0] * parDims[IDX_1]);
            break;
        }
        case Layout::CBA: {
            parStrides.ResetShape(1, parDims[IDX_0], parDims[IDX_0] * parDims[IDX_1]);
            break;
        }
        case Layout::ABCD: {
            parStrides.ResetShape(parDims[IDX_1] * parDims[IDX_2] * parDims[IDX_3], parDims[IDX_2] * parDims[IDX_3],
                                  parDims[IDX_3], 1);
            break;
        }
        case Layout::ABDC: {
            parStrides.ResetShape(parDims[IDX_1] * parDims[IDX_2] * parDims[IDX_3], parDims[IDX_2] * parDims[IDX_3], 1,
                                  parDims[IDX_2]);
            break;
        }
        case Layout::ACBD: {
            parStrides.ResetShape(parDims[IDX_1] * parDims[IDX_2] * parDims[IDX_3], parDims[IDX_3],
                                  parDims[IDX_1] * parDims[IDX_3], 1);
            break;
        }
        case Layout::ACDB: {
            parStrides.ResetShape(parDims[IDX_1] * parDims[IDX_2] * parDims[IDX_3], 1, parDims[IDX_1] * parDims[IDX_3],
                                  parDims[IDX_1]);
            break;
        }
        case Layout::ADBC: {
            parStrides.ResetShape(parDims[IDX_1] * parDims[IDX_2] * parDims[IDX_3], parDims[IDX_2], 1,
                                  parDims[IDX_1] * parDims[IDX_2]);
            break;
        }
        case Layout::ADCB: {
            parStrides.ResetShape(parDims[IDX_1] * parDims[IDX_2] * parDims[IDX_3], 1, parDims[IDX_1],
                                  parDims[IDX_1] * parDims[IDX_2]);
            break;
        }
        case Layout::BACD: {
            parStrides.ResetShape(parDims[IDX_2] * parDims[IDX_3], parDims[IDX_0] * parDims[IDX_2] * parDims[IDX_3],
                                  parDims[IDX_3], 1);
            break;
        }
        case Layout::BCDA: {
            parStrides.ResetShape(1, parDims[IDX_0] * parDims[IDX_2] * parDims[IDX_3], parDims[IDX_0] * parDims[IDX_3],
                                  parDims[IDX_0]);
            break;
        }
        case Layout::CDAB: {
            parStrides.ResetShape(parDims[IDX_1], 1, parDims[IDX_0] * parDims[IDX_1] * parDims[IDX_3],
                                  parDims[IDX_0] * parDims[IDX_1]);
            break;
        }
        case Layout::CDBA: {
            parStrides.ResetShape(1, parDims[IDX_0], parDims[IDX_0] * parDims[IDX_1] * parDims[IDX_3],
                                  parDims[IDX_0] * parDims[IDX_1]);
            break;
        }
        case Layout::DCAB: {
            parStrides.ResetShape(parDims[IDX_1], 1, parDims[IDX_0] * parDims[IDX_1],
                                  parDims[IDX_0] * parDims[IDX_1] * parDims[IDX_2]);
            break;
        }
        case Layout::ABCDE: {
            parStrides.ResetShape(parDims[IDX_1] * parDims[IDX_2] * parDims[IDX_3] * parDims[IDX_4],
                                  parDims[IDX_2] * parDims[IDX_3] * parDims[IDX_4], parDims[IDX_3] * parDims[IDX_4],
                                  parDims[IDX_4], 1);
            break;
        }
        case Layout::ABDEC: {
            parStrides.ResetShape(parDims[IDX_1] * parDims[IDX_2] * parDims[IDX_3] * parDims[IDX_4],
                                  parDims[IDX_2] * parDims[IDX_3] * parDims[IDX_4], 1, parDims[IDX_2] * parDims[IDX_4],
                                  parDims[IDX_2]);
            break;
        }
        case Layout::ACBDE: {
            parStrides.ResetShape(parDims[IDX_1] * parDims[IDX_2] * parDims[IDX_3] * parDims[IDX_4],
                                  parDims[IDX_3] * parDims[IDX_4], parDims[IDX_1] * parDims[IDX_3] * parDims[IDX_4],
                                  parDims[IDX_4], 1);
            break;
        }
        case Layout::ACDEB: {
            parStrides.ResetShape(parDims[IDX_1] * parDims[IDX_2] * parDims[IDX_3] * parDims[IDX_4], 1,
                                  parDims[IDX_1] * parDims[IDX_3] * parDims[IDX_4], parDims[IDX_1] * parDims[IDX_4],
                                  parDims[IDX_1]);
            break;
        }
        case Layout::ADECB: {
            parStrides.ResetShape(parDims[IDX_1] * parDims[IDX_2] * parDims[IDX_3] * parDims[IDX_4], 1,
                                  parDims[IDX_1], parDims[IDX_1] * parDims[IDX_2] * parDims[IDX_4],
                                  parDims[IDX_1] * parDims[IDX_2]);
            break;
        }
        case Layout::BACDE: {
            parStrides.ResetShape(parDims[IDX_2] * parDims[IDX_3] * parDims[IDX_4],
                                  parDims[IDX_0] * parDims[IDX_2] * parDims[IDX_3] * parDims[IDX_4],
                                  parDims[IDX_3] * parDims[IDX_4], parDims[IDX_4], 1);
            break;
        }
        case Layout::BCDEA: {
            parStrides.ResetShape(1, parDims[IDX_0] * parDims[IDX_2] * parDims[IDX_3] * parDims[IDX_4],
                                  parDims[IDX_0] * parDims[IDX_3] * parDims[IDX_4], parDims[IDX_0] * parDims[IDX_4],
                                  parDims[IDX_0]);
            break;
        }
        case Layout::CDEAB: {
            parStrides.ResetShape(parDims[IDX_1], 1, parDims[IDX_0] * parDims[IDX_1] * parDims[IDX_3] * parDims[IDX_4],
                                  parDims[IDX_0] * parDims[IDX_1] * parDims[IDX_4], parDims[IDX_0] * parDims[IDX_1]);
            break;
        }
        case Layout::CDEBA: {
            parStrides.ResetShape(1, parDims[IDX_0], parDims[IDX_0] * parDims[IDX_1] * parDims[IDX_3] * parDims[IDX_4],
                                  parDims[IDX_0] * parDims[IDX_1] * parDims[IDX_4], parDims[IDX_0] * parDims[IDX_1]);
            break;
        }
        case Layout::DECAB: {
            parStrides.ResetShape(parDims[IDX_1], 1, parDims[IDX_0] * parDims[IDX_1],
                                  parDims[IDX_0] * parDims[IDX_1] * parDims[IDX_2] * parDims[IDX_4],
                                  parDims[IDX_0] * parDims[IDX_1] * parDims[IDX_2]);
            break;
        }
        case Layout::ABCED: {
            parStrides.ResetShape(parDims[IDX_1] * parDims[IDX_2] * parDims[IDX_3] * parDims[IDX_4],
                                  parDims[IDX_2] * parDims[IDX_3] * parDims[IDX_4], parDims[IDX_3] * parDims[IDX_4], 1,
                                  parDims[IDX_3]);
            break;
        }
        default: {
            if constexpr (areExcetionsDeclined) {
                return Status::BAD_ARGUMENTS;
            } else {
                throw Service::LogicError {"Tensor Info: layout wasn't processed"};
            }
            break;
        }
    }
    parStrides = Service::FlushStrides(parDims, parStrides);

    if constexpr (areExcetionsDeclined) {
        return Status::SUCCESS;
    }
}

template
ReturnType<false> TensorInfo::SetMinimumRequiredStrides<false>(const Shape& parDims,
    Layout parLayout, Shape& parStrides) noexcept(false);

template
ReturnType<true> TensorInfo::SetMinimumRequiredStrides<true>(const Shape& parDims,
    Layout parLayout, Shape& parStrides) noexcept(true);


template <bool areExcetionsDeclined>
ReturnType<areExcetionsDeclined> TensorInfo::ValidateDimsAndStrides(const Shape& parDims,
    Layout parLayout, const Shape& parStrides) noexcept(areExcetionsDeclined)
{
    if (parDims.GetNumDims() != parStrides.GetNumDims()) {
        if constexpr (areExcetionsDeclined) {
            return Status::BAD_ARGUMENTS;
        } else {
            throw Service::LogicError {"Tensor Info: dims and strides have different dimensionality"};
        }
    }
    if constexpr (areExcetionsDeclined) {
        KuDNN::Status s = ValidateLayout<true>(parDims, parLayout);
        if (s != Status::SUCCESS) {
            return s;
        }
    } else {
        ValidateLayout<false>(parDims, parLayout);
    }
    std::array<SizeType, MAX_DIMS> dimsArr = {1, 1, 1, 1, 1};
    for (SizeType i = 0; i < parDims.GetNumDims(); ++i) {
        dimsArr[i] = parDims[i];
    }
    if (Service::WillIntMultOverflow(dimsArr.cbegin(), dimsArr.cbegin() + parDims.GetNumDims())) {
        if constexpr (areExcetionsDeclined) {
            return Status::BAD_ARGUMENTS;
        } else {
            throw Service::LogicError {"TensorInfo: total size doesn't fit in SizeType"};
        }
    }
    Shape minStrides = parDims;
    if constexpr (areExcetionsDeclined) {
        KuDNN::Status s = SetMinimumRequiredStrides<true>(parDims, parLayout, minStrides);
        if (s != Status::SUCCESS) {
            return s;
        }
    } else {
        SetMinimumRequiredStrides<false>(parDims, parLayout, minStrides);
    }
    for (SizeType i = 0; i < parDims.GetNumDims(); ++i) {
        if (parStrides[i] < minStrides[i]) {
            if constexpr (areExcetionsDeclined) {
                return Status::BAD_ARGUMENTS;
            } else {
                throw Service::LogicError {"Tensor Info: incorrect strides"};
            }
        }
    }

    if constexpr (areExcetionsDeclined) {
        return Status::SUCCESS;
    }
}

template
ReturnType<false> TensorInfo::ValidateDimsAndStrides<false>(const Shape&  parDims,
    Layout parLayout, const Shape&  parStrides) noexcept(false);

template
ReturnType<true> TensorInfo::ValidateDimsAndStrides<true>(const Shape&  parDims,
    Layout parLayout, const Shape&  parStrides) noexcept(true);


} // KuDNN
