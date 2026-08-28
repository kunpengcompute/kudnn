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

std::pair<bool, KuDNN::BatchNormalizationLayerFWD*> convert_to_kdnn_batch_normalization_fwd(
    const memory_desc_wrapper& mem_desc_src, const memory_desc_wrapper& mem_desc_dst,
    const memory_desc_wrapper& mem_desc_stats, const memory_desc_wrapper& mem_desc_scaleshift,
    bool use_global_stats, bool use_scale, bool use_shift, const dim_t channel_size,
    bool fuse_relu, bool is_training) noexcept(false) {
    if (!common_tensor_checks(mem_desc_src)) {
        return {false, nullptr};
    }
    KuDNN::TensorInfo src = get_kdnn_tensor_info(mem_desc_src);
    KuDNN::TensorInfo dst = get_kdnn_tensor_info(mem_desc_dst);
    KuDNN::TensorInfo stats = KuDNN::TensorInfo({{static_cast<KuDNN::SizeType>(channel_size)}, KuDNN::Element::TypeT::F32, KuDNN::Layout::A});
    KuDNN::TensorInfo scale_shift = KuDNN::TensorInfo({{static_cast<KuDNN::SizeType>(channel_size)}, KuDNN::Element::TypeT::F32, KuDNN::Layout::A});

    if (mem_desc_stats != &glob_zero_md) {
        if (!common_tensor_checks(mem_desc_stats)) {
            return {false, nullptr};
        }
        stats = get_kdnn_tensor_info(mem_desc_stats);
    }

    if (mem_desc_scaleshift != &glob_zero_md) {
        if (!common_tensor_checks(mem_desc_scaleshift)) {
            return {false, nullptr};
        }
        scale_shift = get_kdnn_tensor_info(mem_desc_scaleshift);
    }
    KuDNN::NormalizationFlags flags =  KuDNN::NormalizationFlags::NONE;
    flags |= use_global_stats ? KuDNN::NormalizationFlags::USE_GLOBAL_STATS : KuDNN::NormalizationFlags::NONE;
    flags |= use_scale ? KuDNN::NormalizationFlags::USE_SCALE : KuDNN::NormalizationFlags::NONE;
    flags |= use_shift ? KuDNN::NormalizationFlags::USE_SHIFT : KuDNN::NormalizationFlags::NONE;
    flags |= fuse_relu ? KuDNN::NormalizationFlags::FUSE_NORM_RELU : KuDNN::NormalizationFlags::NONE;
    KuDNN::Propagation p_kind = is_training ? KuDNN::Propagation::FORWARD_TRAINING : KuDNN::Propagation::FORWARD_INFERENCE;
    if (KuDNN::Status::SUCCESS != KuDNN::BatchNormalizationLayerFWD::ValidateInput(p_kind, src,
        stats, scale_shift, dst, flags)) {
        return {false, nullptr};
    } else {
        return {true, new KuDNN::BatchNormalizationLayerFWD{p_kind, src, stats, scale_shift, dst, flags}};
    }
}

std::pair<bool, KuDNN::BatchNormalizationLayerBWD*> convert_to_kdnn_batch_normalization_bwd(const memory_desc_wrapper& mem_desc_src,
        const memory_desc_wrapper& mem_desc_diff_dst, const memory_desc_wrapper& mem_desc_stats,
        const memory_desc_wrapper& mem_desc_scaleshift, const memory_desc_wrapper& mem_desc_diff_src,
        const memory_desc_wrapper& mem_desc_diff_ss, bool use_global_stats, bool use_scale,
        bool use_shift, const dim_t channel_size, bool fuse_relu, bool is_data) noexcept(false) {

    if (!common_tensor_checks(mem_desc_src, mem_desc_diff_dst, mem_desc_diff_src)) {
        return {false, nullptr};
    }
    KuDNN::TensorInfo src = get_kdnn_tensor_info(mem_desc_src);
    KuDNN::TensorInfo diff_dst = get_kdnn_tensor_info(mem_desc_diff_dst);
    KuDNN::TensorInfo diff_src = get_kdnn_tensor_info(mem_desc_diff_src);
    KuDNN::TensorInfo stats = KuDNN::TensorInfo({{static_cast<KuDNN::SizeType>(channel_size)}, KuDNN::Element::TypeT::F32, KuDNN::Layout::A});
    KuDNN::TensorInfo scale_shift = KuDNN::TensorInfo({{static_cast<KuDNN::SizeType>(channel_size)}, KuDNN::Element::TypeT::F32, KuDNN::Layout::A});
    KuDNN::TensorInfo diff_scale_shift = KuDNN::TensorInfo({{static_cast<KuDNN::SizeType>(channel_size)}, KuDNN::Element::TypeT::F32, KuDNN::Layout::A});

    if (mem_desc_stats != &glob_zero_md) {
        if (!common_tensor_checks(mem_desc_stats)) {
            return {false, nullptr};
        }
        stats = get_kdnn_tensor_info(mem_desc_stats);
    }
    if (mem_desc_scaleshift != &glob_zero_md) {
        if (!common_tensor_checks(mem_desc_scaleshift)) {
            return {false, nullptr};
        }
        scale_shift = get_kdnn_tensor_info(mem_desc_scaleshift);
    }
    if (mem_desc_diff_ss != &glob_zero_md) {
        if (!common_tensor_checks(mem_desc_diff_ss)) {
            return {false, nullptr};
        }
        diff_scale_shift = get_kdnn_tensor_info(mem_desc_diff_ss);
    }

    KuDNN::NormalizationFlags flags =  KuDNN::NormalizationFlags::NONE;
    flags |= use_global_stats ? KuDNN::NormalizationFlags::USE_GLOBAL_STATS : KuDNN::NormalizationFlags::NONE;
    flags |= use_scale ? KuDNN::NormalizationFlags::USE_SCALE : KuDNN::NormalizationFlags::NONE;
    flags |= use_shift ? KuDNN::NormalizationFlags::USE_SHIFT : KuDNN::NormalizationFlags::NONE;
    flags |= fuse_relu ? KuDNN::NormalizationFlags::FUSE_NORM_RELU : KuDNN::NormalizationFlags::NONE;
    KuDNN::Propagation p_kind = is_data ? KuDNN::Propagation::BACKWARD_DATA : KuDNN::Propagation::BACKWARD;
    if (KuDNN::Status::SUCCESS != KuDNN::BatchNormalizationLayerBWD::ValidateInput(p_kind, src,
        diff_dst, stats, scale_shift, diff_src, diff_scale_shift, flags)) {
        return {false, nullptr};
    } else {
        return {true, new KuDNN::BatchNormalizationLayerBWD{p_kind, src, diff_dst, stats,
            scale_shift, diff_src, diff_scale_shift, flags}};
    }
}

} // namespace kdnn_utils
} // namespace aarch64
} // namespace cpu
} // namespace impl
} // namespace dnnl
