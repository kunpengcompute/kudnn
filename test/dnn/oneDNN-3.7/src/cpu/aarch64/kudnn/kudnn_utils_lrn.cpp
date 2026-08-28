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

std::pair<bool, KuDNN::LocalResponseNormalizationLayerFWD*> convert_to_kdnn_lrn_fwd(const memory_desc_wrapper& mem_desc_src,
        const memory_desc_wrapper& mem_desc_dst, float alpha, float beta, float k,
        std::size_t local_size, bool across_channels) noexcept(false) {
    KuDNN::LRNormAlgorithmKind algorithm = across_channels ?
        KuDNN::LRNormAlgorithmKind::ACROSS : KuDNN::LRNormAlgorithmKind::WITHIN;
    if (!common_tensor_checks(mem_desc_src, mem_desc_dst)) {
        return {false, nullptr};
    }
    KuDNN::TensorInfo src = get_kdnn_tensor_info(mem_desc_src);
    KuDNN::TensorInfo dst = get_kdnn_tensor_info(mem_desc_dst);

    if (KuDNN::Status::SUCCESS != KuDNN::LocalResponseNormalizationLayerFWD::ValidateInput(src,
        dst, alpha, beta, k, local_size, algorithm)) {
        return {false, nullptr};
    } else {
        return {true, new KuDNN::LocalResponseNormalizationLayerFWD{src, dst, alpha, beta, k,
            local_size, algorithm}};
    }
}

std::pair<bool, KuDNN::LocalResponseNormalizationLayerBWD*> convert_to_kdnn_lrn_bwd(const memory_desc_wrapper& src_d,
    const memory_desc_wrapper& diff_dst_d, const memory_desc_wrapper& diff_src_d,
    float alpha, float beta, float k, std::size_t local_size, bool across_channels) noexcept(false) {
    KuDNN::LRNormAlgorithmKind algorithm = across_channels ?
        KuDNN::LRNormAlgorithmKind::ACROSS : KuDNN::LRNormAlgorithmKind::WITHIN;
    if (!common_tensor_checks(src_d, diff_dst_d, diff_src_d)) {
        return {false, nullptr};
    }

    KuDNN::TensorInfo src = get_kdnn_tensor_info(src_d);
    KuDNN::TensorInfo diff_src = get_kdnn_tensor_info(diff_src_d);
    KuDNN::TensorInfo diff_dst = get_kdnn_tensor_info(diff_dst_d);

    if (KuDNN::Status::SUCCESS != KuDNN::LocalResponseNormalizationLayerBWD::ValidateInput(src,
        diff_dst, diff_src, alpha, beta, k, local_size, algorithm)) {
        return {false, nullptr};
    } else {
        return {true, new KuDNN::LocalResponseNormalizationLayerBWD{src,
            diff_dst, diff_src, alpha, beta, k, local_size, algorithm}};
    }
}

} // namespace kdnn_utils

} // namespace aarch64
} // namespace cpu
} // namespace impl
} // namespace dnnl
