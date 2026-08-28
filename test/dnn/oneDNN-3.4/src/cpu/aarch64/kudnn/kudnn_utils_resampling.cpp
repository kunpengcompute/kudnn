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

std::pair<bool, KuDNN::ResamplingLayerFWD*> convert_to_kdnn_resampling(const memory_desc_wrapper& mem_desc_src,
        const memory_desc_wrapper& mem_desc_dst, const alg_kind_t& resampling_alg) noexcept(false) {
    if (!common_tensor_checks(mem_desc_src, mem_desc_dst)) {
        return {false, nullptr};
    }
    if (KuDNN::Status::SUCCESS != KuDNN::ResamplingLayerFWD::ValidateInput(get_kdnn_tensor_info(mem_desc_src),
        get_kdnn_tensor_info(mem_desc_dst), get_kdnn_resampling_alg(resampling_alg))) {
        return {false, nullptr};
    } else {
        return {true, new KuDNN::ResamplingLayerFWD{get_kdnn_tensor_info(mem_desc_src),
            get_kdnn_tensor_info(mem_desc_dst), get_kdnn_resampling_alg(resampling_alg)}};
    }
}

std::pair<bool, KuDNN::ResamplingLayerBWD*> convert_to_kdnn_resampling_bwd(
    const memory_desc_wrapper& mem_desc_diff_dst, const memory_desc_wrapper& mem_desc_diff_src,
    const alg_kind_t& resampling_alg) noexcept(false) {
    if (!common_tensor_checks(mem_desc_diff_dst, mem_desc_diff_src)) {
        return {false, nullptr};
    }
    if (KuDNN::Status::SUCCESS != KuDNN::ResamplingLayerBWD::ValidateInput(get_kdnn_tensor_info(mem_desc_diff_dst),
        get_kdnn_tensor_info(mem_desc_diff_src), get_kdnn_resampling_alg(resampling_alg))) {
        return {false, nullptr};
    } else {
        return {true, new KuDNN::ResamplingLayerBWD{get_kdnn_tensor_info(mem_desc_diff_dst),
            get_kdnn_tensor_info(mem_desc_diff_src), get_kdnn_resampling_alg(resampling_alg)}};
    }
}

} // namespace kdnn_utils

} // namespace aarch64
} // namespace cpu
} // namespace impl
} // namespace dnnl
