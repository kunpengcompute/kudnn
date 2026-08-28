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

#ifndef KUDNN_REORDER_SUPPORT_FUNCTIONS_HPP
#define KUDNN_REORDER_SUPPORT_FUNCTIONS_HPP

#include "kudnn.hpp"

namespace KuDNN {
namespace Detail {

class ReorderHelpers {
public:
    template<typename... Strides>
    static void CompressDimsAndStrides(Shape &dims, Strides && ...strides)
    {
        for (SizeType i = dims.GetNumDims() - 1; i > 0; --i) {
            if (CompressDimsAndStridesCondition(i, dims, std::forward<Strides>(strides)...)) {
                dims[i] = dims[i] * dims[i - 1];
                for (SizeType j = i - 1; j > 0; --j) {
                    ShiftDimsAndStrides(j, dims, std::forward<Strides>(strides)...);
                }
                RemoveDimsAndStridesUnnecessaryValues(dims, std::forward<Strides>(strides)...);
                break;
            }
        }
    }

    template<typename... Strides>
    static void ReduceDimsAndStrides(Shape &dims, Strides && ...strides)
    {
        for (SizeType i = dims.GetNumDims() - 1; i > 0; --i) {
            if (dims[i] == 1) {
                for (SizeType j = i; j > 0; --j) {
                    ShiftDimsAndStrides(j, dims, std::forward<Strides>(strides)...);
                }
                RemoveDimsAndStridesUnnecessaryValues(dims, std::forward<Strides>(strides)...);
                break;
            }
        }
    }

private:
    template<typename... Strides>
    static bool CompressDimsAndStridesCondition(const SizeType i, const Shape &dims,
                                                const Shape &stride, Strides && ... strides)
    {
        return (stride[i - 1] == stride[i] * dims[i]) &&
                CompressDimsAndStridesCondition(i, dims, std::forward<Strides>(strides)...);
    }
    static bool CompressDimsAndStridesCondition(const SizeType i, const Shape &dims, const Shape &stride)
    {
        return (stride[i - 1] == stride[i] * dims[i]);
    }

    template<typename... Strides>
    static void ShiftDimsAndStrides(SizeType j, Shape &stride, Strides &&... strides)
    {
        stride[j] = stride[j - 1];
        return ShiftDimsAndStrides(j, std::forward<Strides>(strides)...);
    }

    static void ShiftDimsAndStrides(SizeType j, Shape &stride)
    {
        stride[j] = stride[j - 1];
    }

    template<typename... Strides>
    static void RemoveDimsAndStridesUnnecessaryValues(Shape &dims, Shape &stride, Strides && ... strides)
    {
        stride[0] = 0;
        return RemoveDimsAndStridesUnnecessaryValues(dims, std::forward<Strides>(strides)...);
    }

    static void RemoveDimsAndStridesUnnecessaryValues(Shape &dims)
    {
        dims[0] = 1;
    }
};

} // Detail
} // KuDNN

#endif // KUDNN_REORDER_SUPPORT_FUNCTIONS_HPP
