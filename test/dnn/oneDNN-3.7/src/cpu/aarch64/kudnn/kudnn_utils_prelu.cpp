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

std::pair<bool, KuDNN::PReLULayerFWD*> convert_to_kdnn_prelu(const memory_desc_wrapper& mem_desc_src,
        const memory_desc_wrapper& mem_desc_wei, const memory_desc_wrapper& mem_desc_dst) noexcept(false) {
    using namespace format_tag;
    if (utils::one_of(format_tag::undef, get_layout(mem_desc_src)) ||
        utils::one_of(format_tag::undef, get_layout(mem_desc_wei))) {
        return {false, nullptr};
    }
    if (!common_tensor_checks(mem_desc_src, mem_desc_wei, mem_desc_dst)) {
        return {false, nullptr};
    }
    auto src_tag = get_layout(mem_desc_src, mem_desc_wei);
    KuDNN::TensorInfo src = get_kdnn_tensor_info(mem_desc_src);
    KuDNN::TensorInfo wei = get_kdnn_tensor_info(mem_desc_wei);
    KuDNN::TensorInfo dst = get_kdnn_tensor_info(mem_desc_dst);
    if (!utils::one_of(format_tag::undef, src_tag)) {
        wei = KuDNN::TensorInfo(get_kdnn_tensor_info(mem_desc_wei).GetDims(),
            get_kdnn_data_t(mem_desc_wei.data_type()), src.GetLayout());
    }
    if (KuDNN::Status::SUCCESS != KuDNN::PReLULayerFWD::ValidateInput(src, wei, dst)) {
        return {false, nullptr};
    } else {
        return {true, new KuDNN::PReLULayerFWD{src, wei, dst}};
    }
}

std::pair<bool, KuDNN::PReLULayerBWD*> convert_to_kdnn_prelu(const memory_desc_wrapper& mem_desc_src,
    const memory_desc_wrapper& mem_desc_diff_src, const memory_desc_wrapper& mem_desc_wei,
    const memory_desc_wrapper& mem_desc_diff_wei, const memory_desc_wrapper& mem_desc_diff_dst) noexcept(false) {
    using namespace format_tag;
    if (utils::one_of(format_tag::undef, get_layout(mem_desc_src)) ||
        utils::one_of(format_tag::undef, get_layout(mem_desc_wei))) {
        return {false, nullptr};
    }
    if (!common_tensor_checks(mem_desc_src, mem_desc_diff_src, mem_desc_wei, mem_desc_diff_wei, mem_desc_diff_dst)) {
        return {false, nullptr};
    }
    auto src_tag = get_layout(mem_desc_src, mem_desc_wei);
    KuDNN::TensorInfo src = get_kdnn_tensor_info(mem_desc_src);
    KuDNN::TensorInfo diff_src = get_kdnn_tensor_info(mem_desc_diff_src);
    KuDNN::TensorInfo wei = get_kdnn_tensor_info(mem_desc_wei);
    KuDNN::TensorInfo diff_wei = get_kdnn_tensor_info(mem_desc_diff_wei);
    KuDNN::TensorInfo diff_dst = get_kdnn_tensor_info(mem_desc_diff_dst);
    if (!utils::one_of(format_tag::undef, src_tag)) {
        wei = KuDNN::TensorInfo(get_kdnn_tensor_info(mem_desc_wei).GetDims(),
            get_kdnn_data_t(mem_desc_wei.data_type()), src.GetLayout());
        diff_wei = KuDNN::TensorInfo(get_kdnn_tensor_info(mem_desc_diff_wei).GetDims(),
            get_kdnn_data_t(mem_desc_diff_wei.data_type()), src.GetLayout());
    }
    if (KuDNN::Status::SUCCESS != KuDNN::PReLULayerBWD::ValidateInput(src, diff_src, wei, diff_wei, diff_dst)) {
        return {false, nullptr};
    } else {
        return {true, new KuDNN::PReLULayerBWD{src, diff_src, wei, diff_wei, diff_dst}};
    }
}

} // namespace kdnn_utils

} // namespace aarch64
} // namespace cpu
} // namespace impl
} // namespace dnnl
