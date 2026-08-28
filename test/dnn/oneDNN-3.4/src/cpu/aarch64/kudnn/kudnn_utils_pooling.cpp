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

std::pair<bool, KuDNN::PoolingLayerFWD*> convert_to_kdnn_pooling_fwd(const memory_desc_wrapper& mem_desc_src,
    const memory_desc_wrapper& mem_desc_dst, const pooling_desc_t& cd) noexcept(false)
{
    using namespace format_tag;
    auto src_tag = get_layout(mem_desc_src, mem_desc_dst);
    if (utils::one_of(format_tag::undef, src_tag)) {
        return {false, nullptr};
    }
    if (!common_tensor_checks(mem_desc_src, mem_desc_dst)) {
        return {false, nullptr};
    }

    KuDNN::Shape strides = get_kdnn_shape(cd.strides, mem_desc_src.ndims() - 2);
    KuDNN::Shape kernel = get_kdnn_shape(cd.kernel, mem_desc_src.ndims() - 2);
    KuDNN::Shape dilation = get_kdnn_shape(cd.dilation, mem_desc_src.ndims() - 2);
    KuDNN::Shape paddingL = get_kdnn_shape(cd.padding[0], mem_desc_src.ndims() - 2);
    KuDNN::Shape paddingR = get_kdnn_shape(cd.padding[1], mem_desc_src.ndims() - 2);

    KuDNN::TensorInfo src = KuDNN::TensorInfo(get_kdnn_tensor_info(mem_desc_src).GetDims(),
        get_kdnn_data_t(mem_desc_src.data_type()), get_kdnn_layout(src_tag));
    KuDNN::TensorInfo dst = KuDNN::TensorInfo(get_kdnn_tensor_info(mem_desc_dst).GetDims(),
        get_kdnn_data_t(mem_desc_dst.data_type()), get_kdnn_layout(src_tag));
    
    if (KuDNN::Status::SUCCESS != KuDNN::PoolingLayerFWD::ValidateInput(get_kdnn_prop_kind_t(cd.prop_kind),
        get_kdnn_pooling_alg(cd.alg_kind), src, dst, strides, kernel, dilation, paddingL, paddingR)) {
        return {false, nullptr};
    } else {
        return {true, new KuDNN::PoolingLayerFWD{get_kdnn_prop_kind_t(cd.prop_kind),
            get_kdnn_pooling_alg(cd.alg_kind), src, dst, strides, kernel, dilation, paddingL, paddingR}};
    }

}

std::pair<bool, KuDNN::PoolingLayerBWD*> convert_to_kdnn_pooling_bwd(const memory_desc_wrapper& mem_desc_diff_src,
    const memory_desc_wrapper& mem_desc_diff_dst, const pooling_desc_t& cd) noexcept(false)
{
    using namespace format_tag;
    auto src_tag = get_layout(mem_desc_diff_src, mem_desc_diff_dst);
    if (utils::one_of(format_tag::undef, src_tag)) {
        return {false, nullptr};
    }
    if (!common_tensor_checks(mem_desc_diff_src, mem_desc_diff_dst)) {
        return {false, nullptr};
    }

    KuDNN::Shape strides = get_kdnn_shape(cd.strides, mem_desc_diff_src.ndims() - 2);
    KuDNN::Shape kernel = get_kdnn_shape(cd.kernel, mem_desc_diff_src.ndims() - 2);
    KuDNN::Shape dilation = get_kdnn_shape(cd.dilation, mem_desc_diff_src.ndims() - 2);
    KuDNN::Shape paddingL = get_kdnn_shape(cd.padding[0], mem_desc_diff_src.ndims() - 2);
    KuDNN::Shape paddingR = get_kdnn_shape(cd.padding[1], mem_desc_diff_src.ndims() - 2);

    KuDNN::TensorInfo src = KuDNN::TensorInfo(get_kdnn_tensor_info(mem_desc_diff_src).GetDims(),
        get_kdnn_data_t(mem_desc_diff_src.data_type()), get_kdnn_layout(src_tag));
    KuDNN::TensorInfo dst = KuDNN::TensorInfo(get_kdnn_tensor_info(mem_desc_diff_dst).GetDims(),
        get_kdnn_data_t(mem_desc_diff_dst.data_type()), get_kdnn_layout(src_tag));
    
    if (KuDNN::Status::SUCCESS != KuDNN::PoolingLayerBWD::ValidateInput(
        get_kdnn_pooling_alg(cd.alg_kind), src, dst, strides, kernel, dilation, paddingL, paddingR)) {
        return {false, nullptr};
    } else {
        return {true, new KuDNN::PoolingLayerBWD{
            get_kdnn_pooling_alg(cd.alg_kind), src, dst, strides, kernel, dilation, paddingL, paddingR}};
    }

}

} // namespace kdnn_utils

} // namespace aarch64
} // namespace cpu
} // namespace impl
} // namespace dnnl
