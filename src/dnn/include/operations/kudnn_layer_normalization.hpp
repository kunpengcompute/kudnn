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

#ifndef KUDNN_LAYER_NORMALIZATION_HPP
#define KUDNN_LAYER_NORMALIZATION_HPP

#include <memory>

#include "types/kudnn_tensor_info.hpp"
#include "service/kudnn_err_codes.hpp"

namespace KuDNN {

namespace Detail {

class NormalizationLayerFWDImpl;
class NormalizationLayerBWDImpl;

} // namespace Detail

class KUDNN_API_PUBLIC NormalizationLayerFWD final {
public:
    using SizeType = ::KuDNN::SizeType;
    static Status ValidateInput(const TensorInfo &srcInfo, const TensorInfo &statsInfo,
                                const TensorInfo &scaleshiftInfo, const TensorInfo &dstInfo,
                                NormalizationFlags flags) noexcept(false);
    NormalizationLayerFWD(const TensorInfo &srcInfo, const TensorInfo &statsInfo, const TensorInfo &scaleshiftInfo,
                          const TensorInfo &dstInfo, NormalizationFlags flags) noexcept(false);
    NormalizationLayerFWD(const NormalizationLayerFWD &other) noexcept(false);
    NormalizationLayerFWD(NormalizationLayerFWD &&other) noexcept;
    NormalizationLayerFWD &operator=(const NormalizationLayerFWD &other) noexcept(false);
    NormalizationLayerFWD &operator=(NormalizationLayerFWD &&other) noexcept;
    void Run(const void *src, void *dst, const void *scale, const void *shift, float *mean, float *variance,
             bool saveStats, const float eps) const noexcept(false);
    ~NormalizationLayerFWD() noexcept;

private:
    std::unique_ptr<Detail::NormalizationLayerFWDImpl> pImpl;
};

class KUDNN_API_PUBLIC NormalizationLayerBWD final {
public:
    using SizeType = ::KuDNN::SizeType;
    static Status ValidateInput(const TensorInfo &srcInfo, const TensorInfo &statInfo, const TensorInfo &diffSrcInfo,
                                const TensorInfo &diffDstInfo, const TensorInfo &scaleShiftInfo,
                                const TensorInfo &diffscaleShiftInfo, NormalizationFlags flags) noexcept;
    NormalizationLayerBWD(const TensorInfo &srcInfo, const TensorInfo &statInfo, const TensorInfo &diffSrcInfo,
                          const TensorInfo &diffDstInfo, const TensorInfo &scaleShiftInfo,
                          const TensorInfo &diffscaleShiftInfo, NormalizationFlags flags) noexcept(false);
    NormalizationLayerBWD(const NormalizationLayerBWD &other) noexcept(false);
    NormalizationLayerBWD(NormalizationLayerBWD &&other) noexcept;
    NormalizationLayerBWD &operator=(const NormalizationLayerBWD &other) noexcept(false);
    NormalizationLayerBWD &operator=(NormalizationLayerBWD &&other) noexcept;
    void Run(const void *src, const float *mean, const float *variance, const void *diffDst, const void *scale,
             void *diffSrc, void *diffScale, void *diffShift, float eps) const noexcept(false);
    ~NormalizationLayerBWD() noexcept;

private:
    std::unique_ptr<Detail::NormalizationLayerBWDImpl> pImpl;
};

} // namespace KuDNN

#endif // KUDNN_LAYER_NORMALIZATION_HPP
