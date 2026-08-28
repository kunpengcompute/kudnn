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

std::pair<bool, KuDNN::SumLayer*> convert_to_kdnn_sum(const std::vector<memory_desc_wrapper>& mem_desc_src, const float *scales,
        const memory_desc_wrapper& mem_desc_dst) noexcept(false) {
    if (!common_tensor_checks(mem_desc_dst)) {
        return {false, nullptr};
    }
    std::vector<KuDNN::TensorInfo> mem_desc_src_kdnn;
    for (size_t ind = 0; ind < mem_desc_src.size(); ++ind) {
        if (!common_tensor_checks(mem_desc_src[ind])) {
            return {false, nullptr};
        }
        mem_desc_src_kdnn.push_back(get_kdnn_tensor_info(mem_desc_src[ind]));
    }
    if (KuDNN::Status::SUCCESS != KuDNN::SumLayer::ValidateInput(mem_desc_src_kdnn, scales,
        get_kdnn_tensor_info(mem_desc_dst))) {
        return {false, nullptr};
    } else {
        return {true, new KuDNN::SumLayer{mem_desc_src_kdnn, scales, get_kdnn_tensor_info(mem_desc_dst)}};
    }
}

} // namespace kdnn_utils

} // namespace aarch64
} // namespace cpu
} // namespace impl
} // namespace dnnl
