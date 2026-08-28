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

std::pair<bool, KuDNN::ActivationLayerFWD*> convert_to_kdnn_act_fwd(const memory_desc_wrapper& mem_desc_src,
        const memory_desc_wrapper& mem_desc_dst, const alg_kind_t& eltwise_alg, float alpha, float beta) noexcept(false) {
    if (!common_tensor_checks(mem_desc_src, mem_desc_dst)) {
        return {false, nullptr};
    }
    if (KuDNN::Status::SUCCESS != KuDNN::ActivationLayerFWD::ValidateInput(get_kdnn_tensor_info(mem_desc_src),
        get_kdnn_tensor_info(mem_desc_dst), get_kdnn_alg(eltwise_alg), alpha, beta)) {
        return {false, nullptr};
    } else {
        return {true, new KuDNN::ActivationLayerFWD{get_kdnn_tensor_info(mem_desc_src),
            get_kdnn_tensor_info(mem_desc_dst), get_kdnn_alg(eltwise_alg), alpha, beta}};
    }
}

std::pair<bool, KuDNN::ActivationLayerBWD*> convert_to_kdnn_act_bwd(const memory_desc_wrapper& mem_desc_ds,
        const memory_desc_wrapper& mem_desc_dd, const memory_desc_wrapper& mem_desc_src,
        const alg_kind_t& eltwise_alg, float alpha, float beta) noexcept(false) {
    if (!common_tensor_checks(mem_desc_ds, mem_desc_dd, mem_desc_src)) {
        return {false, nullptr};
    }
    if (KuDNN::Status::SUCCESS != KuDNN::ActivationLayerBWD::ValidateInput(get_kdnn_tensor_info(mem_desc_ds),
        get_kdnn_tensor_info(mem_desc_dd), get_kdnn_tensor_info(mem_desc_src), get_kdnn_alg(eltwise_alg), alpha, beta)) {
        return {false, nullptr};
    } else {
        return {true, new KuDNN::ActivationLayerBWD{get_kdnn_tensor_info(mem_desc_ds), get_kdnn_tensor_info(mem_desc_dd),
        get_kdnn_tensor_info(mem_desc_src), get_kdnn_alg(eltwise_alg), alpha, beta}};
    }
}

std::pair<bool, KuDNN::ActivationLayerBWD*> convert_to_kdnn_act_bwd_with_dst(const memory_desc_wrapper& mem_desc_ds,
        const memory_desc_wrapper& mem_desc_dd, const memory_desc_wrapper& mem_desc_src,
        const alg_kind_t& eltwise_alg, float alpha, float beta) noexcept(false) {
    if (!common_tensor_checks(mem_desc_ds, mem_desc_dd, mem_desc_src)) {
        return {false, nullptr};
    }
    if (KuDNN::Status::SUCCESS != KuDNN::ActivationLayerBWD::ValidateInput(get_kdnn_tensor_info(mem_desc_ds),
        get_kdnn_tensor_info(mem_desc_dd), get_kdnn_tensor_info(mem_desc_src), get_kdnn_with_dst_alg(eltwise_alg), alpha, beta)) {
        return {false, nullptr};
    } else {
        return {true, new KuDNN::ActivationLayerBWD{get_kdnn_tensor_info(mem_desc_ds), get_kdnn_tensor_info(mem_desc_dd),
        get_kdnn_tensor_info(mem_desc_src), get_kdnn_with_dst_alg(eltwise_alg), alpha, beta}};
    }
}

} // namespace kdnn_utils

} // namespace aarch64
} // namespace cpu
} // namespace impl
} // namespace dnnl
