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


#ifndef KUDNN_RMS_NORMALIZATION_HPP
#define KUDNN_RMS_NORMALIZATION_HPP

#include "types/kudnn_tensor_info.hpp"
#include "service/kudnn_err_codes.hpp"

namespace KuDNN {

namespace Detail {
class RMSNormalizationLayerFWDImpl;
} // Detail

class KUDNN_API_PUBLIC RMSNormalizationLayerFWD final {
public:
    using SizeType = ::KuDNN::SizeType;
    static Status ValidateInput(const TensorInfo &srcInfo, const TensorInfo &statsInfo,
                                const TensorInfo &scaleInfo, const TensorInfo &dstInfo,
                                NormalizationFlags flags) noexcept(false);
    RMSNormalizationLayerFWD(const TensorInfo &srcInfo, const TensorInfo &statsInfo, const TensorInfo &scaleInfo,
                        const TensorInfo &dstInfo, NormalizationFlags flags) noexcept(false);
    RMSNormalizationLayerFWD(const RMSNormalizationLayerFWD &other) noexcept(false);
    RMSNormalizationLayerFWD(RMSNormalizationLayerFWD &&other) noexcept;
    RMSNormalizationLayerFWD& operator=(const RMSNormalizationLayerFWD &other) noexcept(false);
    RMSNormalizationLayerFWD& operator=(RMSNormalizationLayerFWD &&other) noexcept;
    void Run(const void *src, void *dst, const void *scale, float *variance,
            bool saveStats, const float eps) const noexcept(false);
    ~RMSNormalizationLayerFWD() noexcept;
private:
    std::unique_ptr<Detail::RMSNormalizationLayerFWDImpl> pImpl;
};

} // KuDNN

#endif // KUDNN_RMS_NORMALIZATION_HPP
