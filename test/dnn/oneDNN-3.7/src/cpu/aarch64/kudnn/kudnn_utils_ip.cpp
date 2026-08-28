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

std::pair<bool, KuDNN::InnerProductLayerFWD*> convert_to_kdnn_ip_fwd(const memory_desc_wrapper& mem_desc_src,
        const memory_desc_wrapper& mem_desc_wei, const memory_desc_wrapper& mem_desc_dst,
        const memory_desc_wrapper& mem_desc_bia, const primitive_attr_t& attributes) noexcept(false) {
    if (!common_tensor_checks(mem_desc_src, mem_desc_wei, mem_desc_dst)) {
        return {false, nullptr};
    }
    KuDNN::TensorInfo src = get_kdnn_tensor_info(mem_desc_src);
    KuDNN::TensorInfo wei = get_kdnn_tensor_info(mem_desc_wei);
    KuDNN::TensorInfo dst = get_kdnn_tensor_info(mem_desc_dst);
    KuDNN::TensorInfo bias = src;
    KuDNN::Element::TypeT biasType = static_cast<KuDNN::Element::TypeT>(wei.GetType()) == KuDNN::Element::TypeT::S8 ?
        KuDNN::Element::TypeT::S32 : static_cast<KuDNN::Element::TypeT>(src.GetType());
    if (mem_desc_bia == &glob_zero_md) {
        bias = KuDNN::TensorInfo({0, 0}, biasType, KuDNN::Layout::AB);
    } else {
        if (!common_tensor_checks(mem_desc_bia)) {
            return {false, nullptr};
        }
        bias = get_kdnn_tensor_info(mem_desc_bia);
    }
    if (KuDNN::Status::SUCCESS != KuDNN::InnerProductLayerFWD::ValidateInput(src,wei, dst, bias, get_kdnn_attributes(attributes))) {
        return {false, nullptr};
    } else {
        return {true, new KuDNN::InnerProductLayerFWD{src, wei, dst, bias, get_kdnn_attributes(attributes)}};
    }
}

std::pair<bool, KuDNN::InnerProductLayerBWDData*> convert_to_kdnn_ip_bwd_d(const memory_desc_wrapper& mem_desc_diff_dst,
        const memory_desc_wrapper& mem_desc_wei, const memory_desc_wrapper& mem_desc_diff_src) noexcept(false) {
    if (!common_tensor_checks(mem_desc_diff_dst, mem_desc_wei, mem_desc_diff_src)) {
        return {false, nullptr};
    }
    KuDNN::TensorInfo diff_dst = get_kdnn_tensor_info(mem_desc_diff_dst);
    KuDNN::TensorInfo wei = get_kdnn_tensor_info(mem_desc_wei);
    KuDNN::TensorInfo diff_src = get_kdnn_tensor_info(mem_desc_diff_src);
    if (KuDNN::Status::SUCCESS != KuDNN::InnerProductLayerBWDData::ValidateInput(diff_dst,
        wei, diff_src)) {
        return {false, nullptr};
    } else {
        return {true, new KuDNN::InnerProductLayerBWDData{diff_dst, wei, diff_src}};
    }
}

std::pair<bool, KuDNN::InnerProductLayerBWDWeights*> convert_to_kdnn_ip_bwd_w(const memory_desc_wrapper& mem_desc_diff_dst,
        const memory_desc_wrapper& mem_desc_src, const memory_desc_wrapper& mem_desc_diff_wei,
        const memory_desc_wrapper& mem_desc_diff_bia) noexcept(false) {
    if (!common_tensor_checks(mem_desc_diff_dst, mem_desc_src, mem_desc_diff_wei)) {
        return {false, nullptr};
    }
    KuDNN::TensorInfo diff_dst = get_kdnn_tensor_info(mem_desc_diff_dst);
    KuDNN::TensorInfo src = get_kdnn_tensor_info(mem_desc_src);
    KuDNN::TensorInfo diff_wei = get_kdnn_tensor_info(mem_desc_diff_wei);
    KuDNN::TensorInfo diff_bia = diff_dst;
    if (mem_desc_diff_bia == &glob_zero_md) {
        diff_bia = KuDNN::TensorInfo({0, 0}, src.GetType(), KuDNN::Layout::AB);
    } else {
        if (!common_tensor_checks(mem_desc_diff_bia)) {
            return {false, nullptr};
        }
        diff_bia = get_kdnn_tensor_info(mem_desc_diff_bia);
    }
    if (KuDNN::Status::SUCCESS != KuDNN::InnerProductLayerBWDWeights::ValidateInput(diff_dst, src, diff_wei, diff_bia)) {
        return {false, nullptr};
    } else {
        return {true, new KuDNN::InnerProductLayerBWDWeights{diff_dst, src, diff_wei, diff_bia}};
    }
}

} // namespace kdnn_utils

} // namespace aarch64
} // namespace cpu
} // namespace impl
} // namespace dnnl
