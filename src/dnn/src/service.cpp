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

#ifdef USE_HBM
    #include <memkind.h>
    #include <hbwmalloc.h>
#endif

#include "kudnn.hpp"

#include "internal_funcs.hpp"
#include "kernels/kernel_funcs.h"

#include "kernels/matmul_jit/kudnn_gemm_helpers.hpp"
#include "kernels/matmul_jit/kudnn_memory_buffer.hpp"
#include "kernels/reorder/kudnn_reorder_helpers.hpp"

namespace KuDNN {

namespace Detail {

template <typename Task>
void ClearImpls() noexcept(false);

// Function is for internal use only. Thread safety is not guaranteed.
// that's why it's decalred only in internal headers.
// To use it, it is required to declare it user code space.
void ClearImpls();

// clean JIT impls of reorder and matmul
void ClearImpls()
{
    ClearImpls<ReorderInfo>();
    ClearImpls<GemmInfo>();
}

} // Detail

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

Shape GetShapeAccordingToLayout(const Shape &sh, Layout parLayout) noexcept(false)
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

Shape BroadcastToNCDHW(const Shape &sh, bool isDims) noexcept(false)
{
    /*  We cast everything to the `NCDHW` format */
    SizeType fill = isDims ? 1 : 0;
    switch (sh.GetNumDims()) {
        case DIM_1: {
            return Shape{fill, sh[IDX_0], fill, fill, fill}; // C to nCdhw
        }
        case DIM_2: {
            return Shape{sh[IDX_0], sh[IDX_1], fill, fill, fill}; // NC to NCdhw
        }
        case DIM_3: {
            return Shape{sh[IDX_0], sh[IDX_1], fill, fill, sh[IDX_2]}; // NCW to NCdhW
        }
        case DIM_4: {
            return Shape{sh[IDX_0], sh[IDX_1], fill, sh[IDX_2], sh[IDX_3]}; // NCHW to NCdHW
        }
        case DIM_5: {
            return Shape{sh[IDX_0], sh[IDX_1], sh[IDX_2], sh[IDX_3], sh[IDX_4]}; // full NCDHW
        }
        default: {
            throw Service::LogicError {"BroadcastToNCDHW: nothing to broadcast"};
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

void Deallocate(const void *p, SizeType n) noexcept
{
    Service::Unused(n);
    free(const_cast<void *>(p));
}

MemoryBuffer::MemoryBuffer()
    : ptr(nullptr),
    type(MemoryType::DEFAULT),
    owner(false)
{}

MemoryBuffer::MemoryBuffer(MemoryBuffer&& other)
    : ptr(other.ptr),
    type(other.type),
    owner(other.owner)
{
    other.ptr = nullptr;
    other.owner = false;
}

MemoryBuffer& MemoryBuffer::operator=(MemoryBuffer&& other)
{
    Free();
    ptr = other.ptr;
    type = other.type;
    owner = other.owner;
    other.ptr = nullptr;
    other.owner = false;

    return *this;
}

MemoryBuffer::MemoryBuffer(const MemoryBuffer& other)
    : ptr(other.ptr),
    type(other.type),
    owner(false)
{
}

MemoryBuffer& MemoryBuffer::operator=(const MemoryBuffer& other)
{
    Free();
    ptr = other.ptr;
    type = other.type;
    owner = false;
    return *this;
}

MemoryBuffer::MemoryBuffer(SizeType size, MemoryType requestedType, SizeType alignment)
{
    owner = true;
#ifdef USE_HBM
    bool sizeIsOkForHBM = true;
#ifdef __clang__
    sizeIsOkForHBM = size > 14336; // 14336 is actually a magic number
    // for some reason when compiled with clang HBM alloc work inproperly
    // it causes segfaults after all the work is done (when application end the work)
    // even if there's no operations with allocated memory
#endif // __clang__
    if (sizeIsOkForHBM && (requestedType == MemoryType::FAST) &&
        (memkind_check_available(MEMKIND_HBW) == 0) &&
        (MEMKIND_SUCCESS == memkind_posix_memalign(MEMKIND_HBW, &ptr, alignment, size))) {
        type = MemoryType::FAST;
        return;
    }
#else
    Unused(requestedType);
#endif
    ptr = AlignedAlloc(size, alignment);
    type = MemoryType::DEFAULT;
}

void MemoryBuffer::Free()
{
    if (!owner || ptr == nullptr) {
        ptr = nullptr;
        owner = false;
        return;
    }
#ifdef USE_HBM
    if (type == MemoryType::FAST) {
        memkind_free(MEMKIND_HBW, ptr);
    } else
#endif
    Deallocate(ptr, 0);
    
    ptr = nullptr;
    owner = false;
}

MemoryBuffer::~MemoryBuffer()
{
    Free();
}

} // Service
} // KuDNN
