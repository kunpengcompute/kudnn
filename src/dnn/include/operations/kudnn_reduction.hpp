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

#ifndef KUDNN_REDUCTION_HPP
#define KUDNN_REDUCTION_HPP

#include <memory>

#include "types/kudnn_tensor_info.hpp"
#include "service/kudnn_err_codes.hpp"

namespace KuDNN {

// supported functions
enum class ReductionFunction {
    UNIMPLEMENTED,
    MAX,
    MIN,
    SUM,
    MUL,
    MEAN,
    NORM_LP_MAX,
    NORM_LP_SUM,
    NORM_LP_POWER_P_MAX,
    NORM_LP_POWER_P_SUM,
};

namespace Detail {

class ReductionLayerImpl;

} // Detail

class ReductionLayer final {
public:
    using SizeType = ::KuDNN::SizeType;
    static Status ValidateInput(const TensorInfo &srcInfo, const TensorInfo &dstInfo,
	    const ReductionFunction& reductionAlg, float p, float eps) noexcept;
    ReductionLayer(const TensorInfo &srcInfo, const TensorInfo &dstInfo, ReductionFunction kind,
        float p = 1, float eps = 0) noexcept(false);
    ReductionLayer(const ReductionLayer &other) noexcept(false);
    ReductionLayer(ReductionLayer &&other) noexcept;
    ReductionLayer& operator=(const ReductionLayer &other) noexcept(false);
    ReductionLayer& operator=(ReductionLayer &&other) noexcept;
    void Run(const void *src, void *dst) const noexcept(false);
    ~ReductionLayer() noexcept;
private:
    std::unique_ptr<Detail::ReductionLayerImpl> pImpl;
};

} // KuDNN

#endif // KUDNN_REDUCTION_HPP
