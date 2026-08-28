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

#ifndef KUDNN_BATCH_NORMALIZATION_HPP
#define KUDNN_BATCH_NORMALIZATION_HPP

#include <memory>

#include "types/kudnn_tensor_info.hpp"
#include "types/kudnn_normalization.hpp"
#include "service/kudnn_err_codes.hpp"
#include "types/kudnn_propagation_type.hpp"

namespace KuDNN {

namespace Detail {

class BatchNormalizationLayerFWDImpl;
class BatchNormalizationLayerBWDImpl;

} // Detail

class BatchNormalizationLayerFWD final {
public:
    using SizeType = ::KuDNN::SizeType;
    static Status ValidateInput(const Propagation &propKind, const TensorInfo &srcInfo, const TensorInfo &statsInfo,
        const TensorInfo &scaleShiftInfo, const TensorInfo &dstInfo,
        NormalizationFlags flags) noexcept;
    BatchNormalizationLayerFWD(const Propagation &propKind, const TensorInfo &srcInfo, const TensorInfo &statsInfo,
        const TensorInfo &scaleShiftInfo, const TensorInfo &dstInfo, NormalizationFlags flags) noexcept(false);
    BatchNormalizationLayerFWD(const BatchNormalizationLayerFWD &other) noexcept(false);
    BatchNormalizationLayerFWD(BatchNormalizationLayerFWD &&other) noexcept;
    BatchNormalizationLayerFWD& operator=(const BatchNormalizationLayerFWD &other) noexcept(false);
    BatchNormalizationLayerFWD& operator=(BatchNormalizationLayerFWD &&other) noexcept;
    void Run(const void *src, void *dst, const float *scale, const float *shift, float *mean, float *variance,
        bool saveStats, const float eps, void *ws = nullptr) const noexcept(false);
    ~BatchNormalizationLayerFWD() noexcept;
private:
    std::unique_ptr<Detail::BatchNormalizationLayerFWDImpl> pImpl;
};

class BatchNormalizationLayerBWD final {
public:
    using SizeType = ::KuDNN::SizeType;
    static Status ValidateInput(const Propagation &propKind, const TensorInfo &srcInfo, const TensorInfo &diffDstInfo,
        const TensorInfo &statsInfo, const TensorInfo &scaleShiftInfo, const TensorInfo &diffSrcInfo,
        const TensorInfo &diffScaleShiftInfo, NormalizationFlags flags) noexcept;
    BatchNormalizationLayerBWD(const Propagation &propKind, const TensorInfo &srcInfo, const TensorInfo &diffDstInfo,
        const TensorInfo &statsInfo, const TensorInfo &scaleShiftInfo, const TensorInfo &diffSrcInfo,
        const TensorInfo &diffScaleShiftInfo, NormalizationFlags flags) noexcept(false);
    BatchNormalizationLayerBWD(const BatchNormalizationLayerBWD &other) noexcept(false);
    BatchNormalizationLayerBWD(BatchNormalizationLayerBWD &&other) noexcept;
    BatchNormalizationLayerBWD& operator=(const BatchNormalizationLayerBWD &other) noexcept(false);
    BatchNormalizationLayerBWD& operator=(BatchNormalizationLayerBWD &&other) noexcept;
    void Run(const void *src, const void *diffDst, const float *mean, const float *variance, const float *scale,
        void *diffSrc, float *diffScale, float *diffShift, const float eps,
        const void *ws = nullptr) const noexcept(false);
    ~BatchNormalizationLayerBWD() noexcept;
private:
    std::unique_ptr<Detail::BatchNormalizationLayerBWDImpl> pImpl;
};

} // KuDNN

#endif // KUDNN_BATCH_NORMALIZATION_HPP
