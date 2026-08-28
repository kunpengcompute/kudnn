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

#include "kudnn.hpp"

#include "cpu/aarch64/kudnn/kudnn_utils.hpp"
#include "common/type_helpers.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace aarch64 {

namespace kdnn_utils {

using namespace dnnl::impl::alg_kind;
using namespace data_type;

std::pair<bool, KuDNN::ConvolutionLayerBWDData*> convert_to_kdnn_conv_bwd_data(const memory_desc_wrapper& mem_desc_diff_dst,
        const memory_desc_wrapper& mem_desc_wei, const memory_desc_wrapper& mem_desc_diff_src,
        const convolution_desc_t &cd, const alg_kind_t &alg) noexcept(false) {
    if (!common_tensor_checks(mem_desc_diff_dst, mem_desc_wei, mem_desc_diff_src)) {
        return {false, nullptr};
    }
    KuDNN::TensorInfo diff_dst = get_kdnn_tensor_info(mem_desc_diff_dst);
    KuDNN::TensorInfo wei = get_kdnn_tensor_info(mem_desc_wei);
    KuDNN::TensorInfo diff_src = get_kdnn_tensor_info(mem_desc_diff_src);
    KuDNN::Shape strides = get_kdnn_shape(cd.strides, mem_desc_diff_dst.ndims() - 2);
    KuDNN::Shape paddingL = get_kdnn_shape(cd.padding[0], mem_desc_diff_dst.ndims() - 2);
    KuDNN::Shape paddingR = get_kdnn_shape(cd.padding[1], mem_desc_diff_dst.ndims() - 2);
    KuDNN::Shape dilates = get_kdnn_shape(cd.dilates, mem_desc_diff_dst.ndims() - 2);
    if (KuDNN::Status::SUCCESS != KuDNN::ConvolutionLayerBWDData::ValidateInput(diff_dst, wei, diff_src,
        strides, dilates, paddingL, paddingR, get_kdnn_conv_alg(alg))) {
        return {false, nullptr};
    } else {
        return {true, new KuDNN::ConvolutionLayerBWDData{diff_dst, wei, diff_src,
        strides, dilates, paddingL, paddingR, get_kdnn_conv_alg(alg)}};
    }
}

std::pair<bool, KuDNN::ConvolutionLayerBWDWeights*> convert_to_kdnn_conv_bwd_weights(const memory_desc_wrapper& mem_desc_diff_dst,
        const memory_desc_wrapper& mem_desc_src, const memory_desc_wrapper& mem_desc_diff_wei,
        const memory_desc_wrapper& mem_desc_diff_bia, const convolution_desc_t &cd, const alg_kind_t &alg) noexcept(false)
{
    if (!common_tensor_checks(mem_desc_diff_dst, mem_desc_src, mem_desc_diff_wei)) {
        return {false, nullptr};
    }

    KuDNN::TensorInfo diff_dst = get_kdnn_tensor_info(mem_desc_diff_dst);
    KuDNN::TensorInfo src = get_kdnn_tensor_info(mem_desc_src);
    KuDNN::TensorInfo diff_wei = get_kdnn_tensor_info(mem_desc_diff_wei);
    KuDNN::TensorInfo diff_bias = KuDNN::TensorInfo({0}, diff_wei.GetType(), KuDNN::Layout::A);
    if (mem_desc_diff_bia != &glob_zero_md) {
        if (!common_tensor_checks(mem_desc_diff_bia)) {
            return {false, nullptr};
        }
        diff_bias = get_kdnn_tensor_info(mem_desc_diff_bia);
    }
    KuDNN::Shape strides = get_kdnn_shape(cd.strides, mem_desc_src.ndims() - 2);
    KuDNN::Shape paddingL = get_kdnn_shape(cd.padding[0], mem_desc_src.ndims() - 2);
    KuDNN::Shape paddingR = get_kdnn_shape(cd.padding[1], mem_desc_src.ndims() - 2);
    KuDNN::Shape dilates = get_kdnn_shape(cd.dilates, mem_desc_src.ndims() - 2);
    if (KuDNN::Status::SUCCESS != KuDNN::ConvolutionLayerBWDWeights::ValidateInput(diff_dst, src, diff_wei, diff_bias,
        strides, dilates, paddingL, paddingR, get_kdnn_conv_alg(alg))) {
        return {false, nullptr};
    } else {
        return {true, new KuDNN::ConvolutionLayerBWDWeights{diff_dst, src, diff_wei, diff_bias,
            strides, dilates, paddingL, paddingR, get_kdnn_conv_alg(alg)}};
    }
}

} // namespace kdnn_utils

} // namespace aarch64
} // namespace cpu
} // namespace impl
} // namespace dnnl
