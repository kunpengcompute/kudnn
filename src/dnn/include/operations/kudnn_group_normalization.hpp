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

#ifndef KUDNN_GROUP_NORMALIZATION_HPP
#define KUDNN_GROUP_NORMALIZATION_HPP

#include <memory>

#include "types/kudnn_tensor_info.hpp"
#include "service/kudnn_err_codes.hpp"

namespace KuDNN {

namespace Detail {

class GroupNormalizationLayerFWDImpl;
class GroupNormalizationLayerBWDImpl;

} // Detail

class GroupNormalizationLayerFWD final {
public:
    using SizeType = ::KuDNN::SizeType;
    static Status ValidateInput(const TensorInfo &srcInfo, const TensorInfo &scaleshiftInfo, SizeType groupInfo,
        const TensorInfo &dstInfo, NormalizationFlags flags) noexcept;
    GroupNormalizationLayerFWD(const TensorInfo &srcInfo, const TensorInfo &scaleshiftInfo, SizeType groupSize,
                          const TensorInfo &dstInfo, NormalizationFlags flags) noexcept(false);
    GroupNormalizationLayerFWD(const GroupNormalizationLayerFWD &other) noexcept(false);
    GroupNormalizationLayerFWD(GroupNormalizationLayerFWD &&other) noexcept;
    GroupNormalizationLayerFWD& operator=(const GroupNormalizationLayerFWD &other) noexcept(false);
    GroupNormalizationLayerFWD& operator=(GroupNormalizationLayerFWD &&other) noexcept;
    void Run(const void *src, void *dst, const void *scale, const void *shift, float *mean, float *variance,
             bool saveStats, const float eps) const noexcept(false);
    ~GroupNormalizationLayerFWD() noexcept;
private:
    std::unique_ptr<Detail::GroupNormalizationLayerFWDImpl> pImpl;
};

class GroupNormalizationLayerBWD final {
public:
    using SizeType = ::KuDNN::SizeType;
    static Status ValidateInput(const TensorInfo &srcInfo, const TensorInfo &statInfo,
        const TensorInfo &diffSrcInfo, const TensorInfo &diffDstInfo,
        const TensorInfo &scaleShiftInfo, const TensorInfo &diffscaleShiftInfo,
        NormalizationFlags flags) noexcept;
    GroupNormalizationLayerBWD(const TensorInfo &srcInfo, const TensorInfo &statInfo,
        const TensorInfo &diffSrcInfo, const TensorInfo &diffDstInfo,
        const TensorInfo &scaleShiftInfo, const TensorInfo &diffscaleShiftInfo,
        NormalizationFlags flags) noexcept(false);
    GroupNormalizationLayerBWD(const GroupNormalizationLayerBWD &other) noexcept(false);
    GroupNormalizationLayerBWD(GroupNormalizationLayerBWD &&other) noexcept;
    GroupNormalizationLayerBWD& operator=(const GroupNormalizationLayerBWD &other) noexcept(false);
    GroupNormalizationLayerBWD& operator=(GroupNormalizationLayerBWD &&other) noexcept;
    void Run(const void *src, const float *mean, const float *variance, const void *diffDst,
            const void *scale, void *diffSrc, void *diffScale,
            void *diffShift, float eps) const noexcept(false);
    ~GroupNormalizationLayerBWD() noexcept;
private:
    std::unique_ptr<Detail::GroupNormalizationLayerBWDImpl> pImpl;
};

} // KuDNN

#endif // KUDNN_GROUP_NORMALIZATION_HPP