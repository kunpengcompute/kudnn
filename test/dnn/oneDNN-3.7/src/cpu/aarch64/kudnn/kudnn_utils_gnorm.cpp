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

std::pair<bool, KuDNN::GroupNormalizationLayerFWD*> convert_to_kdnn_group_normalization_fwd(
        const memory_desc_wrapper& mem_desc_src, const memory_desc_wrapper& mem_desc_scaleshift,
         std::size_t groupInfo, const memory_desc_wrapper& mem_desc_dst, bool use_global_stats,
        bool use_scale, bool use_shift) noexcept(false) {
    using namespace format_tag;
    auto src_tag = get_layout(mem_desc_src, mem_desc_dst);
    if (utils::one_of(format_tag::undef, src_tag)) {
        return {false, nullptr};
    }
    if (!common_tensor_checks(mem_desc_src, mem_desc_dst)) {
        return {false, nullptr};
    }
    KuDNN::TensorInfo src = KuDNN::TensorInfo(get_kdnn_tensor_info(mem_desc_src).GetDims(),
        get_kdnn_data_t(mem_desc_src.data_type()), get_kdnn_layout(src_tag));
    KuDNN::TensorInfo dst = KuDNN::TensorInfo(get_kdnn_tensor_info(mem_desc_dst).GetDims(),
        get_kdnn_data_t(mem_desc_dst.data_type()), get_kdnn_layout(src_tag));
    KuDNN::TensorInfo scaleshift = KuDNN::TensorInfo({src.GetDims()[src.GetNumDims() - 1]},
        src.GetType(), KuDNN::Layout::A);

    if (mem_desc_scaleshift != &glob_zero_md) {
        if (!common_tensor_checks(mem_desc_scaleshift)) {
            return {false, nullptr};
        }
        scaleshift = get_kdnn_tensor_info(mem_desc_scaleshift);
    }
    KuDNN::NormalizationFlags flags = KuDNN::NormalizationFlags::NONE;
    flags |= use_global_stats ? KuDNN::NormalizationFlags::USE_GLOBAL_STATS : KuDNN::NormalizationFlags::NONE;
    flags |= use_scale ? KuDNN::NormalizationFlags::USE_SCALE : KuDNN::NormalizationFlags::NONE;
    flags |= use_shift ? KuDNN::NormalizationFlags::USE_SHIFT : KuDNN::NormalizationFlags::NONE;

    if (KuDNN::Status::SUCCESS != KuDNN::GroupNormalizationLayerFWD::ValidateInput(src,
        scaleshift, groupInfo, dst, flags)) {
        return {false, nullptr};
    } else {
        return {true, new KuDNN::GroupNormalizationLayerFWD(src, scaleshift, groupInfo, dst, flags)};
    }
}

std::pair<bool, KuDNN::GroupNormalizationLayerBWD*> convert_to_kdnn_group_normalization_bwd(const memory_desc_wrapper& src_d,
    const memory_desc_wrapper& stat_d, const memory_desc_wrapper& diff_src_d, const memory_desc_wrapper& diff_dst_d,
        const memory_desc_wrapper& scale_shift_d, const memory_desc_wrapper& diff_scale_shift_d,
        bool use_global_stats, bool use_scale, bool use_shift) noexcept(false) {
    using namespace format_tag;
    auto src_tag = get_layout(src_d, diff_dst_d, diff_src_d);
    if (utils::one_of(format_tag::undef, src_tag)) {
        return {false, nullptr};
    }
    if (!common_tensor_checks(src_d, diff_dst_d, diff_src_d, stat_d)) {
        return {false, nullptr};
    }

    KuDNN::TensorInfo src = KuDNN::TensorInfo(get_kdnn_tensor_info(src_d).GetDims(),
        get_kdnn_data_t(src_d.data_type()), get_kdnn_layout(src_tag));
    KuDNN::TensorInfo diff_src = KuDNN::TensorInfo(get_kdnn_tensor_info(diff_src_d).GetDims(),
        get_kdnn_data_t(diff_src_d.data_type()), get_kdnn_layout(src_tag));
    KuDNN::TensorInfo diff_dst = KuDNN::TensorInfo(get_kdnn_tensor_info(diff_dst_d).GetDims(),
        get_kdnn_data_t(diff_dst_d.data_type()), get_kdnn_layout(src_tag));
    KuDNN::TensorInfo stat = KuDNN::TensorInfo(get_kdnn_tensor_info(stat_d).GetDims(),
            get_kdnn_data_t(stat_d.data_type()),
            get_kdnn_layout(stat_d));
    KuDNN::TensorInfo scale_shift = KuDNN::TensorInfo({src.GetDims()[src.GetNumDims() - 1]}, src.GetType(), KuDNN::Layout::A);
    KuDNN::TensorInfo diff_scale_shift = KuDNN::TensorInfo({src.GetDims()[src.GetNumDims() - 1]}, src.GetType(), KuDNN::Layout::A);

    if (scale_shift_d != &glob_zero_md) {
        if (!common_tensor_checks(scale_shift_d)) {
            return {false, nullptr};
        }
        scale_shift = get_kdnn_tensor_info(scale_shift_d);
    }

    if (diff_scale_shift_d != &glob_zero_md) {
        if (!common_tensor_checks(diff_scale_shift_d)) {
            return {false, nullptr};
        }
        diff_scale_shift = get_kdnn_tensor_info(diff_scale_shift_d);
    }

    KuDNN::NormalizationFlags flags =  KuDNN::NormalizationFlags::NONE;
    flags |= use_global_stats ? KuDNN::NormalizationFlags::USE_GLOBAL_STATS : KuDNN::NormalizationFlags::NONE;
    flags |= use_scale ? KuDNN::NormalizationFlags::USE_SCALE : KuDNN::NormalizationFlags::NONE;
    flags |= use_shift ? KuDNN::NormalizationFlags::USE_SHIFT : KuDNN::NormalizationFlags::NONE;

    if (KuDNN::Status::SUCCESS != KuDNN::GroupNormalizationLayerBWD::ValidateInput(src, stat, diff_src,
        diff_dst, scale_shift, diff_scale_shift, flags)) {
        return {false, nullptr};
    } else {
        return {true, new KuDNN::GroupNormalizationLayerBWD{src, stat, diff_src,
            diff_dst, scale_shift, diff_scale_shift, flags}};
    }
}

} // namespace kdnn_utils

} // namespace aarch64
} // namespace cpu
} // namespace impl
} // namespace dnnl
