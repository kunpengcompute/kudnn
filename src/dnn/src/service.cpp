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

#include <cstdlib>
#include <array>
#include <memory>
#include <string>
#include <arm_sve.h>
#include <cmath>
#ifdef __clang__
#include <arm_neon.h> // Needed to build with clang
#endif

#include "kudnn.hpp"

#include "internal_funcs.hpp"
#include "kernels/kernel_funcs.h"

namespace KuDNN {


namespace Service {

void ThrowOnStatus(Status s, const std::string& funcName) noexcept(false)
{
    switch (s) {
        case Status::UNSUPPORTED: {
            throw Service::Unsupported {funcName + ": unsupported configuration of parameters"};
            break;
        }
        case Status::BAD_ARGUMENTS: {
            throw Service::LogicError {funcName + ": bad arguments' Shapes"};
            break;
        }
        default: {}
    }
}

void ConvertFp32ToBf16(const float *src, __bf16 *dst, SizeType size)
{
    SizeType vl = svcntw();
    SizeType vl2 = IDX_2 * vl;
    SizeType nResid = size % (vl2);
    for (SizeType i = 0; i < nResid; ++i) {
        dst[i] = vcvth_bf16_f32(src[i]);
    }
    svuint32_t idx = svindex_u32(IDX_0, IDX_2);
    for (SizeType i = nResid; i < size; i += vl2) {
        svbool_t pg = svptrue_b32();

        svfloat32_t vecFloat1 = svld1_gather_u32index_f32(pg, src + i, idx);
        svfloat32_t vecFloat2 = svld1_gather_u32index_f32(pg, src + IDX_1 + i, idx);

        svbfloat16_t vecBfloat = svcvt_bf16_f32_x(pg, vecFloat1);
        vecBfloat = svcvtnt_bf16_f32_x(vecBfloat, pg, vecFloat2);

        svst1(svptrue_b16(), static_cast<__bf16 *>(dst) + i, vecBfloat);
    }
}

void ConvertFp32ToFp16(const float *src, __fp16 *dst, SizeType size)
{
    for (SizeType i = 0; i < size; ++i) {
        dst[i] = src[i];
    }
}


void ConvertFp32ToS32(const float *src, int32_t *dst, SizeType size)
{
    for (SizeType i = 0; i < size; ++i) {
        if (src[i] < static_cast<float>(std::numeric_limits<int32_t>::min())) {
            dst[i] = std::numeric_limits<int32_t>::min();
        } else if (src[i] > static_cast<float>(std::numeric_limits<int32_t>::max())) {
            dst[i] = std::numeric_limits<int32_t>::max();
        } else {
            dst[i] = static_cast<int32_t>(std::nearbyintf(src[i]));
        }
    }
}

void ConvertFp32ToS8(const float *src, int8_t *dst, SizeType size)
{
    for (SizeType i = 0; i < size; ++i) {
        if (src[i] < std::numeric_limits<int8_t>::min()) {
            dst[i] = std::numeric_limits<int8_t>::min();
        } else if (src[i] > std::numeric_limits<int8_t>::max()) {
            dst[i] = std::numeric_limits<int8_t>::max();
        } else {
            dst[i] = static_cast<int8_t>(std::nearbyintf(src[i]));
        }
    }
}

void ConvertFp32ToU8(const float *src, uint8_t *dst, SizeType size)
{
    for (SizeType i = 0; i < size; ++i) {
        if (src[i] < 0.0f) {
            dst[i] = 0U;
        } else if (src[i] > std::numeric_limits<uint8_t>::max()) {
            dst[i] = std::numeric_limits<uint8_t>::max();
        } else {
            dst[i] = static_cast<uint8_t>(std::nearbyintf(src[i]));
        }
    }
}


SizeType ArrayProduct(const Shape &arr, SizeType end, SizeType start) noexcept
{
    SizeType prod = 1;
    for (SizeType i = start; i < end; ++i) {
        prod *= arr[i];
    }
    return prod;
}

SizeType GetIdxForUserLayout(Layout l, SizeType idx) noexcept(false)
{
    Shape sh;
    switch (l) {
        case Layout::A: {
            if (idx > IDX_0) {
                throw Service::LogicError("Shape: index >= layout size");
            }
            sh = {IDX_0};
            break;
        }
        case Layout::AB: {
            if (idx > IDX_1) {
                throw Service::LogicError("Shape: index >= layout size");
            }
            sh = {IDX_0, IDX_1};
            break;
        }
        case Layout::BA: {
            if (idx > IDX_1) {
                throw Service::LogicError("Shape: index >= layout size");
            }
            sh = {IDX_1, IDX_0};
            break;
        }
        case Layout::ABC: {
            if (idx > IDX_2) {
                throw Service::LogicError("Shape: index >= layout size");
            }
            sh = {IDX_0, IDX_1, IDX_2};
            break;
        }
        case Layout::ACB: {
            if (idx > IDX_2) {
                throw Service::LogicError("Shape: index >= layout size");
            }
            sh = {IDX_0, IDX_2, IDX_1};
            break;
        }
        case Layout::BAC: {
            if (idx > IDX_2) {
                throw Service::LogicError("Shape: index >= layout size");
            }
            sh = {IDX_1, IDX_0, IDX_2};
            break;
        }
        case Layout::BCA: {
            if (idx > IDX_2) {
                throw Service::LogicError("Shape: index >= layout size");
            }
            sh = {IDX_1, IDX_2, IDX_0};
            break;
        }
        case Layout::CAB: {
            if (idx > IDX_2) {
                throw Service::LogicError("Shape: index >= layout size");
            }
            sh = {IDX_2, IDX_0, IDX_1};
            break;
        }
        case Layout::CBA: {
            if (idx > IDX_2) {
                throw Service::LogicError("Shape: index >= layout size");
            }
            sh = {IDX_2, IDX_1, IDX_0};
            break;
        }
        case Layout::ABCD: {
            if (idx > IDX_3) {
                throw Service::LogicError("Shape: index >= layout size");
            }
            sh = {IDX_0, IDX_1, IDX_2, IDX_3};
            break;
        }
        case Layout::ABDC: {
            if (idx > IDX_3) {
                throw Service::LogicError("Shape: index >= layout size");
            }
            sh = {IDX_0, IDX_1, IDX_3, IDX_2};
            break;
        }
        case Layout::ACBD: {
            if (idx > IDX_3) {
                throw Service::LogicError("Shape: index >= layout size");
            }
            sh = {IDX_0, IDX_2, IDX_1, IDX_3};
            break;
        }
        case Layout::ACDB: {
            if (idx > IDX_3) {
                throw Service::LogicError("Shape: index >= layout size");
            }
            sh = {IDX_0, IDX_2, IDX_3, IDX_1};
            break;
        }
        case Layout::ADBC: {
            if (idx > IDX_3) {
                throw Service::LogicError("Shape: index >= layout size");
            }
            sh = {IDX_0, IDX_3, IDX_1, IDX_2};
            break;
        }
        case Layout::ADCB: {
            if (idx > IDX_3) {
                throw Service::LogicError("Shape: index >= layout size");
            }
            sh = {IDX_0, IDX_3, IDX_2, IDX_1};
            break;
        }
        case Layout::BACD: {
            if (idx > IDX_3) {
                throw Service::LogicError("Shape: index >= layout size");
            }
            sh = {IDX_1, IDX_0, IDX_2, IDX_3};
            break;
        }
        case Layout::BCDA: {
            if (idx > IDX_3) {
                throw Service::LogicError("Shape: index >= layout size");
            }
            sh = {IDX_1, IDX_2, IDX_3, IDX_0};
            break;
        }
        case Layout::CDAB: {
            if (idx > IDX_3) {
                throw Service::LogicError("Shape: index >= layout size");
            }
            sh = {IDX_2, IDX_3, IDX_0, IDX_1};
            break;
        }
        case Layout::CDBA: {
            if (idx > IDX_3) {
                throw Service::LogicError("Shape: index >= layout size");
            }
            sh = {IDX_2, IDX_3, IDX_1, IDX_0};
            break;
        }
        case Layout::DCAB: {
            if (idx > IDX_3) {
                throw Service::LogicError("Shape: index >= layout size");
            }
            sh = {IDX_3, IDX_2, IDX_0, IDX_1};
            break;
        }
        case Layout::ABCDE: {
            if (idx > IDX_4) {
                throw Service::LogicError("Shape: index >= layout size");
            }
            sh = {IDX_0, IDX_1, IDX_2, IDX_3, IDX_4};
            break;
        }
        case Layout::ABCED: {
            if (idx > IDX_4) {
                throw Service::LogicError("Shape: index >= layout size");
            }
            sh = {IDX_0, IDX_1, IDX_2, IDX_4, IDX_3};
            break;
        }
        case Layout::ABDEC: {
            if (idx > IDX_4) {
                throw Service::LogicError("Shape: index >= layout size");
            }
            sh = {IDX_0, IDX_1, IDX_3, IDX_4, IDX_2};
            break;
        }
        case Layout::ACBDE: {
            if (idx > IDX_4) {
                throw Service::LogicError("Shape: index >= layout size");
            }
            sh = {IDX_0, IDX_2, IDX_1, IDX_3, IDX_4};
            break;
        }
        case Layout::ACDEB: {
            if (idx > IDX_4) {
                throw Service::LogicError("Shape: index >= layout size");
            }
            sh = {IDX_0, IDX_2, IDX_3, IDX_4, IDX_1};
            break;
        }
        case Layout::ADECB: {
            if (idx > IDX_4) {
                throw Service::LogicError("Shape: index >= layout size");
            }
            sh = {IDX_0, IDX_3, IDX_4, IDX_2, IDX_1};
            break;
        }
        case Layout::BACDE: {
            if (idx > IDX_4) {
                throw Service::LogicError("Shape: index >= layout size");
            }
            sh = {IDX_1, IDX_0, IDX_2, IDX_3, IDX_4};
            break;
        }
        case Layout::BCDEA: {
            if (idx > IDX_4) {
                throw Service::LogicError("Shape: index >= layout size");
            }
            sh = {IDX_1, IDX_2, IDX_3, IDX_4, IDX_0};
            break;
        }
        case Layout::CDEAB: {
            if (idx > IDX_4) {
                throw Service::LogicError("Shape: index >= layout size");
            }
            sh = {IDX_2, IDX_3, IDX_4, IDX_0, IDX_1};
            break;
        }
        case Layout::CDEBA: {
            if (idx > IDX_4) {
                throw Service::LogicError("Shape: index >= layout size");
            }
            sh = {IDX_2, IDX_3, IDX_4, IDX_1, IDX_0};
            break;
        }
        case Layout::DECAB: {
            if (idx > IDX_4) {
                throw Service::LogicError("Shape: index >= layout size");
            }
            sh = {IDX_3, IDX_4, IDX_2, IDX_0, IDX_1};
            break;
        }
        default: {
            throw Service::LogicError("Shape: incorrect layout");
            break;
        }
    }
    return sh[idx];
}

Shape GetShapeAccordingToLayout(const Shape &sh, Layout parLayout) noexcept
{
    Shape result;
    std::array<SizeType, MAX_DIMS> arr;
    for (SizeType i = 0; i < sh.GetNumDims(); ++i) {
        arr[i] = sh[Service::GetIdxForUserLayout(parLayout, i)];
    }
    result.ResetShape(arr.data(), sh.GetNumDims());

    return result;
}

// We use 0 strides for dimensions == 1,
// this done to natively support implicit dimension broadcasting
Shape FlushStrides(const Shape &dims, const Shape &strides) noexcept(false)
{
    if (dims.GetNumDims() != strides.GetNumDims()) {
        throw Service::LogicError {"FlushStrides: dims and strides have different dimensionality"};
    }
    Shape result;
    std::array<SizeType, MAX_DIMS> arr;
    for (SizeType i = 0; i < strides.GetNumDims(); ++i) {
        if (dims[i] == 1) {
            arr[i] = 0;
        } else {
            arr[i] = strides[i];
        }
    }
    result.ResetShape(arr.data(), strides.GetNumDims());
    return result;
}

Shape BroadcastTo5D(const Shape &sh, bool isDims) noexcept(false)
{
    /*  We cast everything to the `abcde` format
        1d: a -> 1111a
        2d: ab -> 111ab
        3d: abc -> 11abc
        4d: abcd -> 1abcd
        5d: abcde -> abcde
    */
    SizeType fill = isDims ? 1 : 0;
    switch (sh.GetNumDims()) {
        case DIM_1: {
            return Shape{fill, fill, fill, fill, sh[IDX_0]};
        }
        case DIM_2: {
            return Shape{fill, fill, fill, sh[IDX_0], sh[IDX_1]};
        }
        case DIM_3: {
            return Shape{fill, fill, sh[IDX_0], sh[IDX_1], sh[IDX_2]};
        }
        case DIM_4: {
            return Shape{fill, sh[IDX_0], sh[IDX_1], sh[IDX_2], sh[IDX_3]};
        }
        case DIM_5: {
            return Shape{sh[IDX_0], sh[IDX_1], sh[IDX_2], sh[IDX_3], sh[IDX_4]};
        }
        default: {
            throw Service::LogicError {"BroadcastTo5D: nothing to broadcast"};
        }
    }
}


void *AlignedAlloc(SizeType n, SizeType alignment) noexcept(false)
{
    if ((alignment == 0) || ((alignment & (alignment - 1)) != 0)) {
        throw Service::LogicError("KuDNN alignment isn't a power of 2");
    }
    if (n > (std::numeric_limits<std::size_t>::max() - alignment)) {
        throw BadArrayNewLength();
    }
    n = ((n + alignment - 1) / alignment) * alignment;
    if (auto p = ::aligned_alloc(alignment, n)) {
        return p;
    }
    throw BadAlloc();
}

void Deallocate(void *p, SizeType n) noexcept
{
    Service::Unused(n);
    free(p);
}

} // Service
} // KuDNN
