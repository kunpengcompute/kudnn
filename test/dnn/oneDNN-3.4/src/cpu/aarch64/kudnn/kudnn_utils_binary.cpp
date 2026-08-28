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

std::pair<bool, KuDNN::BinaryLayer*> convert_to_kdnn_binary(const memory_desc_wrapper& mem_desc_src0,
                                                           const memory_desc_wrapper& mem_desc_src1,
                                                           const memory_desc_wrapper& mem_desc_dst,
                                                           const alg_kind_t& binary_op) noexcept(false)
{
    using namespace format_tag;

    if (!common_tensor_checks(mem_desc_src0, mem_desc_src1, mem_desc_dst)) {
        return {false, nullptr};
    }
    KuDNN::TensorInfo src0 = KuDNN::TensorInfo(get_kdnn_tensor_info(mem_desc_src0).GetDims(),
                                             get_kdnn_data_t(mem_desc_src0.data_type()),
                                             get_kdnn_layout(mem_desc_src0));
    KuDNN::TensorInfo src1 = KuDNN::TensorInfo(get_kdnn_tensor_info(mem_desc_src1).GetDims(),
                                             get_kdnn_data_t(mem_desc_src1.data_type()),
                                             get_kdnn_layout(mem_desc_src1));
    KuDNN::TensorInfo dst = KuDNN::TensorInfo(get_kdnn_tensor_info(mem_desc_dst).GetDims(),
                                            get_kdnn_data_t(mem_desc_dst.data_type()),
                                            get_kdnn_layout(mem_desc_dst));
    if (KuDNN::Status::SUCCESS != KuDNN::BinaryLayer::ValidateInput(src0, src1, dst, get_kdnn_op(binary_op))) {
        return {false, nullptr};
    }

    return {true, new KuDNN::BinaryLayer{src0, src1, dst, get_kdnn_op(binary_op)}};
}

} // namespace kdnn_utils

} // namespace aarch64
} // namespace cpu
} // namespace impl
} // namespace dnnl
