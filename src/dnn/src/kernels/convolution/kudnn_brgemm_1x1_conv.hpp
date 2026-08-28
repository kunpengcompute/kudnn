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

#ifndef KUDNN_BRGEMM_1X1_CONV_HPP
#define KUDNN_BRGEMM_1X1_CONV_HPP

#include "service/kudnn_err_codes.hpp"
#include "tools/cg_cpu_isa_traits.hpp"
#include "types/kudnn_data_type.hpp"
#include "types/kudnn_propagation_type.hpp"
#include "types/kudnn_shape.hpp"
#include "types/kudnn_tensor_info.hpp"

#include "kudnn_brgemm_primitive_conf.hpp"

namespace KuDNN {

Status KdnnBrgemm1x1ConvInitConf(KdnnBrgconvDataT &bcd, ARMCG::CPUISAT isa, const TensorInfo &src,
    const TensorInfo &weights, const TensorInfo &dst, const TensorInfo &bias, Shape strides,
    Shape dilates, Shape paddingL, Propagation prop, IntType nthreads);

Status KdnnBrgemm1x1ConvCreateKernel(KdnnBrgconvDataT &bcd);

void KdnnBrgemm1x1ConvExecute(const KdnnBrgconvDataT &bcd, const float *src, const float *wei,
    float *dst, const float *bias);

// kdnn brgconv interface -------------------------------------------------------------------------
Status KdnnBrgconvInitConf(KdnnBrgconvDataT &bcd, ARMCG::CPUISAT isa, const TensorInfo &src,
    const TensorInfo &weights, const TensorInfo &dst, const TensorInfo &bias, Shape strides,
    Shape dilates, Shape paddingL, Propagation prop, IntType nthreads);

Status KdnnBrgconvCreateKernel(KdnnBrgconvDataT &bcd);

void KdnnBrgconvExecute(const KdnnBrgconvDataT &bcd, const float *src, const float *wei,
    float *dst, const float *bias);

} // namespace KuDNN

#endif // KUDNN_BRGEMM_1X1_CONV_HPP
