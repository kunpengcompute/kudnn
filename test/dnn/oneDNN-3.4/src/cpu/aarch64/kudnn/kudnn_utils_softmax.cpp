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

std::pair<bool, KuDNN::SoftmaxLayerFWD*> convert_to_kdnn_softmax(const memory_desc_wrapper& mem_desc_src,
    const memory_desc_wrapper& mem_desc_dst, std::size_t axis, bool is_logsoftmax) noexcept(false) {
    if (!common_tensor_checks(mem_desc_src, mem_desc_dst)) {
        return {false, nullptr};
    }
    KuDNN::AlgorithmKind algorithm = is_logsoftmax ? KuDNN::AlgorithmKind::LOGSOFTMAX : KuDNN::AlgorithmKind::SOFTMAX;
    if (KuDNN::Status::SUCCESS != KuDNN::SoftmaxLayerFWD::ValidateInput(get_kdnn_tensor_info(mem_desc_src),
        get_kdnn_tensor_info(mem_desc_dst), axis, algorithm)) {
        return {false, nullptr};
    } else {
        return {true, new KuDNN::SoftmaxLayerFWD{get_kdnn_tensor_info(mem_desc_src),
            get_kdnn_tensor_info(mem_desc_dst), axis, algorithm}};
    }
}

std::pair<bool, KuDNN::SoftmaxLayerBWD*> convert_to_kdnn_softmax(const memory_desc_wrapper& mem_desc_dst,
    const memory_desc_wrapper& mem_desc_dst_diff, const memory_desc_wrapper& mem_desc_src_diff,
    std::size_t axis, bool is_logsoftmax) noexcept(false) {
    if (!common_tensor_checks(mem_desc_dst, mem_desc_dst_diff, mem_desc_src_diff)) {
        return {false, nullptr};
    }

    KuDNN::AlgorithmKind algorithm = is_logsoftmax ? KuDNN::AlgorithmKind::LOGSOFTMAX : KuDNN::AlgorithmKind::SOFTMAX;
    if (KuDNN::Status::SUCCESS != KuDNN::SoftmaxLayerBWD::ValidateInput(get_kdnn_tensor_info(mem_desc_dst),
                                                                      get_kdnn_tensor_info(mem_desc_dst_diff),
                                                                      get_kdnn_tensor_info(mem_desc_src_diff), axis, algorithm)) {
        return {false, nullptr};
    } else {
        return {true, new KuDNN::SoftmaxLayerBWD{get_kdnn_tensor_info(mem_desc_dst),
            get_kdnn_tensor_info(mem_desc_dst_diff), get_kdnn_tensor_info(mem_desc_src_diff), axis, algorithm}};
    }
}

} // namespace kdnn_utils

} // namespace aarch64
} // namespace cpu
} // namespace impl
} // namespace dnnl
