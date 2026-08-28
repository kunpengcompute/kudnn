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

std::pair<bool, KuDNN::ReductionLayer*> convert_to_kdnn_reduction(const memory_desc_wrapper& mem_desc_src,
    const memory_desc_wrapper& mem_desc_dst, const alg_kind_t& reduction_alg, float p, float eps) noexcept(false) {
    using namespace format_tag;
    auto src_tag = get_layout(mem_desc_src, mem_desc_dst);
    auto dst_tag = get_layout(mem_desc_src, mem_desc_dst);
    if (utils::one_of(format_tag::undef, src_tag, dst_tag)) {
        src_tag = get_layout(mem_desc_src);
        dst_tag = get_layout(mem_desc_dst);
    }
    if (!common_tensor_checks(mem_desc_src, mem_desc_dst)) {
        return {false, nullptr};
    }
    KuDNN::TensorInfo src = KuDNN::TensorInfo(get_kdnn_tensor_info(mem_desc_src).GetDims(),
        get_kdnn_data_t(mem_desc_src.data_type()), get_kdnn_layout(src_tag));
    KuDNN::TensorInfo dst = KuDNN::TensorInfo(get_kdnn_tensor_info(mem_desc_dst).GetDims(),
        get_kdnn_data_t(mem_desc_dst.data_type()), get_kdnn_layout(dst_tag));
    if (KuDNN::Status::SUCCESS != KuDNN::ReductionLayer::ValidateInput(src, dst, get_kdnn_reduction_alg(reduction_alg), p, eps)) {
        return {false, nullptr};
    } else {
        return {true, new KuDNN::ReductionLayer{src, dst, get_kdnn_reduction_alg(reduction_alg), p, eps}};
    }
}

} // namespace kdnn_utils

} // namespace aarch64
} // namespace cpu
} // namespace impl
} // namespace dnnl
