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

void reserve_postops_data(KuDNN::PostOpsDataPtrs& poPtrs, const std::vector<dnnl_post_ops::entry_t> &post_ops_entry,
        const exec_ctx_t &ctx) {
    int post_op_index = 0;
    for (auto &&post_op : post_ops_entry) {
        if (post_op.is_sum()) {
            poPtrs.push_back(nullptr);
        } else if (post_op.is_eltwise()) {
            poPtrs.push_back(nullptr);
        } else if (post_op.is_convolution()) {
            poPtrs.push_back(nullptr); // TODO: need to check
        } else if (post_op.is_binary()) {
            const void *src1 = CTX_IN_MEM(const void *,
            (DNNL_ARG_ATTR_MULTIPLE_POST_OP(post_op_index)
            | DNNL_ARG_SRC_1));
            poPtrs.push_back(src1);
        } else if (post_op.is_prelu()) {
            const void *wei = CTX_IN_MEM(const void *,
            (DNNL_ARG_ATTR_MULTIPLE_POST_OP(post_op_index)
            | DNNL_ARG_WEIGHTS));
            poPtrs.push_back(wei);
        }
        ++post_op_index;
    }
}

bool is_data_type_supported_by_kdnn(const dnnl_data_type_t& dt) noexcept {
    if (get_kdnn_data_t(dt) == KuDNN::Element::TypeT::UNDEFINED) {
        return false;
    } else {
        return true;
    }
}

bool is_data_layout_supported_by_kdnn(const memory_desc_wrapper& mem_desc) noexcept {
    return get_layout(mem_desc) != format_tag::undef;
}

KuDNN::PReLUMask get_kdnn_prelu_mask(int mask) noexcept {
    if (mask > static_cast<int>(KuDNN::PReLUMask::PER_TENSOR)) {
        return KuDNN::PReLUMask::PER_TENSOR;
    } else {
        return static_cast<KuDNN::PReLUMask>(mask);
    }
}

KuDNN::Propagation get_kdnn_prop_kind_t(const dnnl_prop_kind_t& pk) noexcept {
    switch (pk) {
        case dnnl_forward_training: return KuDNN::Propagation::FORWARD_TRAINING;
        case dnnl_forward_inference: return KuDNN::Propagation::FORWARD_INFERENCE;
        case dnnl_backward_data: return KuDNN::Propagation::BACKWARD_DATA;
        case dnnl_backward_weights: return KuDNN::Propagation::BACKWARD_WEIGHTS;
        case dnnl_backward_bias: return KuDNN::Propagation::BACKWARD_BIAS;
        case dnnl_backward: return KuDNN::Propagation::BACKWARD;
        default: return KuDNN::Propagation::UNDEFINED;
    }
}

KuDNN::Element::TypeT get_kdnn_data_t(const dnnl_data_type_t& dt) noexcept {
    switch (dt) {
        case dnnl_f32: return KuDNN::Element::TypeT::F32;
        case dnnl_f16: return KuDNN::Element::TypeT::F16;
        case dnnl_bf16: return KuDNN::Element::TypeT::BF16;
        case dnnl_s8: return KuDNN::Element::TypeT::S8;
        case dnnl_u8: return KuDNN::Element::TypeT::U8;
        case dnnl_s32: return KuDNN::Element::TypeT::S32;
        default: return KuDNN::Element::TypeT::UNDEFINED;
    }
}

KuDNN::ActivationFunction get_kdnn_alg(const alg_kind_t& eltwise_alg) noexcept {
    switch (eltwise_alg) {
        case eltwise_abs : return KuDNN::ActivationFunction::ABS;
        case eltwise_exp: return KuDNN::ActivationFunction::EXP;
        case eltwise_linear: return KuDNN::ActivationFunction::LINEAR;
        case eltwise_log : return KuDNN::ActivationFunction::LOG;
        case eltwise_relu : return KuDNN::ActivationFunction::RELU;
        case eltwise_round : return KuDNN::ActivationFunction::ROUND;
        case eltwise_sqrt : return KuDNN::ActivationFunction::SQRT;
        case eltwise_square : return KuDNN::ActivationFunction::SQUARE;
        case eltwise_tanh: return KuDNN::ActivationFunction::TANH;
        case eltwise_logistic: return KuDNN::ActivationFunction::SIGMOID;
        case eltwise_pow: return KuDNN::ActivationFunction::POW;
        case eltwise_clip: return KuDNN::ActivationFunction::CLIP;
        case eltwise_clip_v2: return KuDNN::ActivationFunction::CLIP_V2;
        case eltwise_hardsigmoid: return KuDNN::ActivationFunction::HARDSIGMOID;
        case eltwise_hardswish: return KuDNN::ActivationFunction::HARDSWISH;
        case eltwise_elu: return KuDNN::ActivationFunction::ELU;
        case eltwise_gelu_tanh: return KuDNN::ActivationFunction::GELU_TANH;
        case eltwise_swish: return KuDNN::ActivationFunction::SWISH;
        case eltwise_soft_relu: return KuDNN::ActivationFunction::SRELU;
        case eltwise_gelu_erf: return KuDNN::ActivationFunction::GELU_ERF;
        case eltwise_mish: return KuDNN::ActivationFunction::MISH;
        case eltwise_exp_use_dst_for_bwd: return KuDNN::ActivationFunction::EXP;
        case eltwise_logistic_use_dst_for_bwd: return KuDNN::ActivationFunction::SIGMOID;
        case eltwise_tanh_use_dst_for_bwd: return KuDNN::ActivationFunction::TANH;
        case eltwise_clip_v2_use_dst_for_bwd: return KuDNN::ActivationFunction::CLIP_V2;
        case eltwise_elu_use_dst_for_bwd: return KuDNN::ActivationFunction::ELU;
        case eltwise_relu_use_dst_for_bwd: return KuDNN::ActivationFunction::RELU;
        case eltwise_sqrt_use_dst_for_bwd: return KuDNN::ActivationFunction::SQRT;
        default: return KuDNN::ActivationFunction::UNIMPLEMENTED;
    }
}

KuDNN::ActivationFunction get_kdnn_with_dst_alg(const alg_kind_t& eltwise_alg) noexcept {
    switch (eltwise_alg) {
        case eltwise_exp_use_dst_for_bwd: return KuDNN::ActivationFunction::EXP_DST;
        case eltwise_logistic_use_dst_for_bwd: return KuDNN::ActivationFunction::LOGISTIC_DST;
        case eltwise_tanh_use_dst_for_bwd: return KuDNN::ActivationFunction::TANH_DST;
        case eltwise_clip_v2_use_dst_for_bwd: return KuDNN::ActivationFunction::CLIP_V2_DST;
        case eltwise_elu_use_dst_for_bwd: return KuDNN::ActivationFunction::ELU_DST;
        case eltwise_relu_use_dst_for_bwd: return KuDNN::ActivationFunction::RELU_DST;
        case eltwise_sqrt_use_dst_for_bwd: return KuDNN::ActivationFunction::SQRT_DST;
        default: return KuDNN::ActivationFunction::UNIMPLEMENTED;
    }
}

KuDNN::ResamplingAlg get_kdnn_resampling_alg(const alg_kind_t& resampling_alg) noexcept {
    switch (resampling_alg) {
        case resampling_nearest : return KuDNN::ResamplingAlg::NEAREST_NEIGHBOR;
        case resampling_linear : return KuDNN::ResamplingAlg::LINEAR;
        default: return KuDNN::ResamplingAlg::UNIMPLEMENTED;
    }
}

KuDNN::ConvolutionAlgorithm get_kdnn_conv_alg(const alg_kind_t& conv_alg) noexcept {
    switch (conv_alg) {
        case convolution_auto: return KuDNN::ConvolutionAlgorithm::AUTO;
        case convolution_direct: return KuDNN::ConvolutionAlgorithm::DIRECT;
        case convolution_winograd : return KuDNN::ConvolutionAlgorithm::WINOGRAD;
        default: return KuDNN::ConvolutionAlgorithm::UNIMPLEMENTED;
    }
}

KuDNN::DeconvolutionAlgorithm get_kdnn_deconv_alg(const alg_kind_t& deconv_alg) noexcept {
    switch (deconv_alg) {
        case deconvolution_direct: return KuDNN::DeconvolutionAlgorithm::DIRECT;
        case deconvolution_winograd: return KuDNN::DeconvolutionAlgorithm::WINOGRAD;
        default: return KuDNN::DeconvolutionAlgorithm::UNIMPLEMENTED;
    }
}

KuDNN::ReductionFunction get_kdnn_reduction_alg(const alg_kind_t& reduction_alg) noexcept {
    switch (reduction_alg) {
        case reduction_max : return KuDNN::ReductionFunction::MAX;
        case reduction_min: return KuDNN::ReductionFunction::MIN;
        case reduction_sum : return KuDNN::ReductionFunction::SUM;
        case reduction_mul : return KuDNN::ReductionFunction::MUL;
        case reduction_mean : return KuDNN::ReductionFunction::MEAN;
        case reduction_norm_lp_max: return KuDNN::ReductionFunction::NORM_LP_MAX;
        case reduction_norm_lp_sum: return KuDNN::ReductionFunction::NORM_LP_SUM;
        case reduction_norm_lp_power_p_max: return KuDNN::ReductionFunction::NORM_LP_POWER_P_MAX;
        case reduction_norm_lp_power_p_sum: return KuDNN::ReductionFunction::NORM_LP_POWER_P_SUM;
        default: return KuDNN::ReductionFunction::UNIMPLEMENTED;
    }
}

KuDNN::PoolingFunction get_kdnn_pooling_alg(const alg_kind_t& pooling_alg) noexcept {
    switch (pooling_alg) {
        case pooling_max : return KuDNN::PoolingFunction::MAX;
        case pooling_avg_include_padding: return KuDNN::PoolingFunction::AVG_INCLUDE_PADDING;
        case pooling_avg_exclude_padding : return KuDNN::PoolingFunction::AVG_EXCLUDE_PADDING;
        default: return KuDNN::PoolingFunction::UNDEFINED;
    }
}

KuDNN::RnnAlgorithm get_kdnn_rnn_alg(const alg_kind_t& rnn_alg) noexcept {
    switch (rnn_alg) {
        case vanilla_rnn: return KuDNN::RnnAlgorithm::RNN;
        case vanilla_lstm: return KuDNN::RnnAlgorithm::LSTM;
        case vanilla_gru : return KuDNN::RnnAlgorithm::GRU;
        case lbr_gru: return KuDNN::RnnAlgorithm::LBR_GRU;
        case vanilla_augru: return KuDNN::RnnAlgorithm::AUGRU;
        case lbr_augru : return KuDNN::RnnAlgorithm::LBR_AUGRU;
        default: return KuDNN::RnnAlgorithm::UNIMPLEMENTED;
    }
}

KuDNN::ActivateFunctionRNN get_kdnn_rnn_activation_alg(const alg_kind_t& rnn_activation_alg) noexcept {
    switch (rnn_activation_alg) {
        case eltwise_relu: return KuDNN::ActivateFunctionRNN::RELU;
        case eltwise_tanh: return KuDNN::ActivateFunctionRNN::TANH;
        case eltwise_logistic: return KuDNN::ActivateFunctionRNN::LOGISTIC;
        default: return KuDNN::ActivateFunctionRNN::UNIMPLEMENTED;
    }
}

KuDNN::BinaryFunction get_kdnn_op(const alg_kind_t& binary_op) noexcept {
    switch (binary_op) {
        case binary_add : return KuDNN::BinaryFunction::ADD;
        case binary_mul : return KuDNN::BinaryFunction::MUL;
        case binary_max : return KuDNN::BinaryFunction::MAX;
        case binary_min : return KuDNN::BinaryFunction::MIN;
        case binary_div : return KuDNN::BinaryFunction::DIV;
        case binary_sub : return KuDNN::BinaryFunction::SUB;
        case binary_ge  : return KuDNN::BinaryFunction::GE;
        case binary_gt  : return KuDNN::BinaryFunction::GT;
        case binary_le  : return KuDNN::BinaryFunction::LE;
        case binary_lt  : return KuDNN::BinaryFunction::LT;
        case binary_eq  : return KuDNN::BinaryFunction::EQ;
        case binary_ne  : return KuDNN::BinaryFunction::NE;
        default         : return KuDNN::BinaryFunction::UNIMPLEMENTED;
    }
}

KuDNN::Layout get_kdnn_layout(format_tag_t tag) noexcept {
    KuDNN::Layout l = KuDNN::Layout::UNDEFINED;
    switch (tag) {
        case(dnnl_a): {
            l = KuDNN::Layout::A;
        } break;
        case(dnnl_ab): {
            l = KuDNN::Layout::AB;
        } break;
        case(dnnl_ba): {
            l = KuDNN::Layout::BA;
        } break;
        case(dnnl_abc): {
            l = KuDNN::Layout::ABC;
        } break;
        case(dnnl_acb): {
            l = KuDNN::Layout::ACB;
        } break;
        case(dnnl_bac): {
            l = KuDNN::Layout::BAC;
        } break;
        case(dnnl_bca): {
            l = KuDNN::Layout::BCA;
        } break;
        case(dnnl_cab): {
            l = KuDNN::Layout::CAB;
        } break;
        case(dnnl_cba): {
            l = KuDNN::Layout::CBA;
        } break;
        case(dnnl_abcd): {
            l = KuDNN::Layout::ABCD;
        } break;
        case(dnnl_abdc): {
            l = KuDNN::Layout::ABDC;
        } break;
        case(dnnl_acbd): {
            l = KuDNN::Layout::ACBD;
        } break;
        case(dnnl_acdb): {
            l = KuDNN::Layout::ACDB;
        } break;
        case(dnnl_adbc): {
            l = KuDNN::Layout::ADBC;
        } break;
        case(dnnl_adcb): {
            l = KuDNN::Layout::ADCB;
        } break;
        case(dnnl_bacd): {
            l = KuDNN::Layout::BACD;
        } break;
        case(dnnl_bcda): {
            l = KuDNN::Layout::BCDA;
        } break;
        case(dnnl_cdab): {
            l = KuDNN::Layout::CDAB;
        } break;
        case(dnnl_cdba): {
            l = KuDNN::Layout::CDBA;
        } break;
        case(dnnl_dcab): {
            l = KuDNN::Layout::DCAB;
        } break;
        case(dnnl_abcde): {
            l = KuDNN::Layout::ABCDE;
        } break;
        case(dnnl_abced): {
            l = KuDNN::Layout::ABCED;
        } break;
        case(dnnl_abdec): {
            l = KuDNN::Layout::ABDEC;
        } break;
        case(dnnl_acbde): {
            l = KuDNN::Layout::ACBDE;
        } break;
        case(dnnl_acdeb): {
            l = KuDNN::Layout::ACDEB;
        } break;
        case(dnnl_adecb): {
            l = KuDNN::Layout::ADECB;
        } break;
        case(dnnl_bacde): {
            l = KuDNN::Layout::BACDE;
        } break;
        case(dnnl_bcdea): {
            l = KuDNN::Layout::BCDEA;
        } break;
        case(dnnl_cdeab): {
            l = KuDNN::Layout::CDEAB;
        } break;
        case(dnnl_cdeba): {
            l = KuDNN::Layout::CDEBA;
        } break;
        case(dnnl_decab): {
            l = KuDNN::Layout::DECAB;
        } break;
        default: {} break;
    }
    return l;
}

KuDNN::Layout get_kdnn_layout(const memory_desc_wrapper& mem_desc) noexcept {
    return get_kdnn_layout(get_layout(mem_desc));
}

format_tag_t get_layout_without_last_dim(format_tag_t tag) noexcept {
    switch (tag) {
        case(dnnl_a): {
            return format_tag_t::dnnl_format_tag_undef;
        } break;
        case(dnnl_ab): {
            return format_tag_t::dnnl_a;
        } break;
        case(dnnl_ba): {
            return format_tag_t::dnnl_a;
        } break;
        case(dnnl_abc): {
            return format_tag_t::dnnl_ab;
        } break;
        case(dnnl_acb): {
            return format_tag_t::dnnl_ab;
        } break;
        case(dnnl_bac): {
            return format_tag_t::dnnl_ba;
        } break;
        case(dnnl_bca): {
            return format_tag_t::dnnl_ba;
        } break;
        case(dnnl_cab): {
            return format_tag_t::dnnl_ab;
        } break;
        case(dnnl_cba): {
            return format_tag_t::dnnl_ba;
        } break;
        case(dnnl_abcd): {
            return format_tag_t::dnnl_abc;
        } break;
        case(dnnl_abdc): {
            return format_tag_t::dnnl_abc;
        } break;
        case(dnnl_acbd): {
            return format_tag_t::dnnl_acb;
        } break;
        case(dnnl_acdb): {
            return format_tag_t::dnnl_acb;
        } break;
        case(dnnl_adbc): {
            return format_tag_t::dnnl_abc;
        } break;
        case(dnnl_adcb): {
            return format_tag_t::dnnl_acb;
        } break;
        case(dnnl_bacd): {
            return format_tag_t::dnnl_bac;
        } break;
        case(dnnl_bcda): {
            return format_tag_t::dnnl_bca;
        } break;
        case(dnnl_cdab): {
            return format_tag_t::dnnl_cab;
        } break;
        case(dnnl_cdba): {
            return format_tag_t::dnnl_cba;
        } break;
        case(dnnl_dcab): {
            return format_tag_t::dnnl_cab;
        } break;
        case(dnnl_abcde): {
            return format_tag_t::dnnl_abcd;
        } break;
        case(dnnl_abced): {
            return format_tag_t::dnnl_abcd;
        } break;
        case(dnnl_abdec): {
            return format_tag_t::dnnl_abdc;
        } break;
        case(dnnl_acbde): {
            return format_tag_t::dnnl_acbd;
        } break;
        case(dnnl_acdeb): {
            return format_tag_t::dnnl_acdb;
        } break;
        case(dnnl_adecb): {
            return format_tag_t::dnnl_adcb;
        } break;
        case(dnnl_bacde): {
            return format_tag_t::dnnl_bacd;
        } break;
        case(dnnl_bcdea): {
            return format_tag_t::dnnl_bcda;
        } break;
        case(dnnl_cdeab): {
            return format_tag_t::dnnl_cdab;
        } break;
        case(dnnl_cdeba): {
            return format_tag_t::dnnl_cdba;
        } break;
        case(dnnl_decab): {
            return format_tag_t::dnnl_dcab;
        } break;
        default: {
            return format_tag_t::dnnl_format_tag_undef;
        } break;
    }
}

KuDNN::Shape get_kdnn_shape(const dims_t& shape, std::size_t num_dims) noexcept(false) {
    KuDNN::Shape result;
    switch (num_dims) {
        case 1: {
            result = {static_cast<KuDNN::SizeType>(shape[0])};
        } break;
        case 2: {
            result = {static_cast<KuDNN::SizeType>(shape[0]),
                      static_cast<KuDNN::SizeType>(shape[1])};
        } break;
        case 3: {
            result = {static_cast<KuDNN::SizeType>(shape[0]),
                      static_cast<KuDNN::SizeType>(shape[1]),
                      static_cast<KuDNN::SizeType>(shape[2])};
        } break;
        case 4: {
            result = {static_cast<KuDNN::SizeType>(shape[0]),
                      static_cast<KuDNN::SizeType>(shape[1]),
                      static_cast<KuDNN::SizeType>(shape[2]),
                      static_cast<KuDNN::SizeType>(shape[3])};
        } break;
        case 5: {
            result = {static_cast<KuDNN::SizeType>(shape[0]),
                      static_cast<KuDNN::SizeType>(shape[1]),
                      static_cast<KuDNN::SizeType>(shape[2]),
                      static_cast<KuDNN::SizeType>(shape[3]),
                      static_cast<KuDNN::SizeType>(shape[4])};
        } break;
        default: {
            result = {0};
        } break;
    }
    return result;
}

bool common_tensor_checks_without_layout(const memory_desc_wrapper& mem_desc) {
    return ((kdnn_utils::is_data_type_supported_by_kdnn(mem_desc.data_type())) &&
        ((mem_desc.ndims() >= 1) && (mem_desc.ndims() <= 5)));
}

bool common_tensor_checks(const memory_desc_wrapper& mem_desc) {
    return (common_tensor_checks_without_layout(mem_desc) &&
        kdnn_utils::is_data_layout_supported_by_kdnn(mem_desc));
}

KuDNN::TensorInfo get_kdnn_tensor_info(const memory_desc_wrapper& mem_desc, KuDNN::Layout lt) noexcept(false) {
    std::size_t num_dims = mem_desc.ndims();
    KuDNN::Shape dimensions = get_kdnn_shape(mem_desc.dims(), num_dims);
    KuDNN::Shape strides = get_kdnn_shape(mem_desc.strides(), num_dims);
    strides = KuDNN::Service::FlushStrides(dimensions, strides);
    return KuDNN::TensorInfo(dimensions, get_kdnn_data_t(mem_desc.data_type()), lt, strides);
}

KuDNN::TensorInfo get_kdnn_tensor_info(const memory_desc_wrapper& mem_desc) noexcept(false) {
    KuDNN::Layout lt = get_kdnn_layout(mem_desc);
    return get_kdnn_tensor_info(mem_desc, lt);
}

KuDNN::Attributes get_kdnn_attributes(const primitive_attr_t& attributes) noexcept(false) {
    KuDNN::PostOps post_ops;
    for (int i = 0; i < attributes.post_ops_.len(); i++) {
        auto &po = attributes.post_ops_.entry_[i];
        if (po.is_sum(false)) {
            post_ops.AppendSum(po.sum.scale);
        } else if (po.is_eltwise()) {
            post_ops.AppendEltwise(get_kdnn_alg(po.eltwise.alg), po.eltwise.alpha, po.eltwise.beta);
        } else if (po.is_convolution()) {
            post_ops.AppendDW(get_kdnn_data_t(po.depthwise_conv.wei_dt), get_kdnn_data_t(po.depthwise_conv.bias_dt),
                get_kdnn_data_t(po.depthwise_conv.dst_dt), po.depthwise_conv.kernel,
                po.depthwise_conv.stride, po.depthwise_conv.padding);
        } else if (po.is_binary()) {
            post_ops.AppendBinary(get_kdnn_op(po.binary.alg), get_kdnn_tensor_info(po.binary.src1_desc));
        } else if (po.is_prelu()) {
            post_ops.AppendPReLU(get_kdnn_prelu_mask(po.prelu.mask));
        }
    }
    KuDNN::Attributes attrs;
    attrs.SetPostOps(post_ops);
    return attrs;
}

} // namespace kdnn_utils

} // namespace aarch64
} // namespace cpu
} // namespace impl
} // namespace dnnl
