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
#include "cpu/rnn/rnn_utils.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace aarch64 {

namespace kdnn_utils {

using namespace dnnl::impl::alg_kind;
using namespace data_type;

std::pair<bool, KuDNN::RnnFWD *> convert_to_kdnn_rnn_fwd(void *rnn_ptr,
    const rnn_desc_t &rd, const primitive_attr_t &attr,
    const memory_desc_wrapper &src_layer_d, const memory_desc_wrapper &src_iter_d,
    const memory_desc_wrapper &src_iter_c_d, const memory_desc_wrapper &weights_layer_d,
    const memory_desc_wrapper &weights_iter_d, const memory_desc_wrapper &weights_projection_d,
    const memory_desc_wrapper &dst_layer_d, const memory_desc_wrapper &dst_iter_d,
    const memory_desc_wrapper &dst_iter_c_d, const memory_desc_wrapper &bias_d, size_t &workspaceSize) noexcept(false)
{
    if (!common_tensor_checks_without_layout(src_layer_d, src_iter_d,
        weights_layer_d, weights_iter_d,
        dst_layer_d, dst_iter_d,
        bias_d)) {
        return {false, nullptr};
    }

    if (rd.cell_kind == alg_kind::vanilla_lstm) {
        if (!common_tensor_checks_without_layout(src_iter_c_d, dst_iter_c_d)) {
            return {false, nullptr};
        }
    }

    rnn_utils::rnn_conf_t &rnn = *(rnn_utils::rnn_conf_t *)rnn_ptr;
    KuDNN::RnnCfg kdnnRnn;
    kdnnRnn.nDir = rnn.n_dir;
    kdnnRnn.nLayer = rnn.n_layer;
    kdnnRnn.nIter = rnn.n_iter;
    kdnnRnn.nGates = rnn.n_gates;
    kdnnRnn.isLbr = rnn.is_lbr;
    kdnnRnn.dhc = rnn.dhc;
    kdnnRnn.dic = rnn.dic;
    kdnnRnn.dlc = rnn.dlc;
    kdnnRnn.mb = rnn.mb;
    kdnnRnn.slc = rnn.slc;
    kdnnRnn.weightsLayerLd = rnn.weights_layer_ld;
    kdnnRnn.scratchGatesLd = rnn.scratch_gates_ld;
    kdnnRnn.sic = rnn.sic;
    kdnnRnn.weightsIterLd = rnn.weights_iter_ld;
    kdnnRnn.srcLayerLd = rnn.src_layer_ld_;
    kdnnRnn.dstIterLd = rnn.dst_iter_ld_;
    kdnnRnn.wsStatesLayerLd = rnn.ws_states_layer_ld;
    kdnnRnn.srcIterLd = rnn.src_iter_ld_;
    kdnnRnn.dstLayerLd = rnn.dst_layer_ld_;
    kdnnRnn.wsStatesIterLd = rnn.ws_states_iter_ld;
    kdnnRnn.execDir = static_cast<KuDNN::Detail::ExecutionDirectionT>(rnn.exec_dir);

    kdnnRnn.projHtLd = rnn.proj_ht_ld;
    kdnnRnn.wsStatesLayerSize = rnn.ws_states_layer_size;
    kdnnRnn.scratchGatesNld = rnn.scratch_gates_nld;
    kdnnRnn.srcIterCLd = rnn.src_iter_c_ld_;
    kdnnRnn.dstIterCLd = rnn.dst_iter_c_ld_;
    kdnnRnn.weightsProjectionLd = rnn.weights_projection_ld;
    kdnnRnn.wsGatesLd = rnn.ws_gates_ld;
    kdnnRnn.wsGatesNld = rnn.ws_gates_nld;
    kdnnRnn.isTraining = rnn.is_training;
    kdnnRnn.wsPerCell = rnn.ws_per_cell;
    kdnnRnn.wsDiffStatesIterCLd = rnn.ws_diff_states_iter_c_ld;
    kdnnRnn.wsHtLd = rnn.ws_ht_ld;
    kdnnRnn.wsHtNld = rnn.ws_ht_nld;
    kdnnRnn.gatesFuncLinearMode = attr.rnn_tparams_.test_mode_;
    kdnnRnn.alpha = rd.alpha;
    if (attr.rnn_tparams_.scales_ != nullptr) {
        kdnnRnn.isScale = true;
        kdnnRnn.scales[0] = attr.rnn_tparams_.scales_[0];
        kdnnRnn.scales[1] = attr.rnn_tparams_.scales_[1];
        kdnnRnn.scales[2] = attr.rnn_tparams_.scales_[2];
        kdnnRnn.scales[3] = attr.rnn_tparams_.scales_[3];
    }
    kdnnRnn.cscale = attr.rnn_tparams_.cscale_;

    size_t scratchpadSize;
    rnn_utils::set_offsets(rnn, kdnnRnn.wsGatesOffset, kdnnRnn.wsHtOffset,
        kdnnRnn.wsStatesLayerOffset, kdnnRnn.wsStatesIterOffset,
        kdnnRnn.wsStatesIterCOffset, kdnnRnn.wsDiffStatesLayerOffset,
        kdnnRnn.wsDiffStatesIterOffset, kdnnRnn.wsDiffStatesIterCOffset,
        kdnnRnn.wsGridCompOffset, kdnnRnn.wsBiasOffset, kdnnRnn.scratchGatesOffset,
        kdnnRnn.scratchHtOffset, kdnnRnn.scratchDiffHtOffset,
        kdnnRnn.scratchCellOffset, scratchpadSize, workspaceSize);

    std::vector<KuDNN::TensorInfo> mem_desc_src_kdnn;
    mem_desc_src_kdnn.push_back(get_kdnn_tensor_info(src_layer_d, KuDNN::Layout::ABC));
    mem_desc_src_kdnn.push_back(get_kdnn_tensor_info(src_iter_d, KuDNN::Layout::ABCD));

    std::vector<KuDNN::TensorInfo> mem_desc_weights_kdnn;
    mem_desc_weights_kdnn.push_back(get_kdnn_tensor_info(weights_layer_d, KuDNN::Layout::ABCDE));
    mem_desc_weights_kdnn.push_back(get_kdnn_tensor_info(weights_iter_d, KuDNN::Layout::ABCDE));

    std::vector<KuDNN::TensorInfo> mem_desc_dst_kdnn;
    mem_desc_dst_kdnn.push_back(get_kdnn_tensor_info(dst_layer_d, KuDNN::Layout::ABC));
    mem_desc_dst_kdnn.push_back(get_kdnn_tensor_info(dst_iter_d, KuDNN::Layout::ABCD));
    if (rd.cell_kind == alg_kind::vanilla_lstm) {
        mem_desc_src_kdnn.push_back(get_kdnn_tensor_info(src_iter_c_d, KuDNN::Layout::ABCD));
        if (rnn.is_lstm_projection) {
            if (!common_tensor_checks_without_layout(weights_projection_d)) {
                return {false, nullptr};
            }
            mem_desc_weights_kdnn.push_back(get_kdnn_tensor_info(weights_projection_d, KuDNN::Layout::ABCD));
        }
        mem_desc_dst_kdnn.push_back(get_kdnn_tensor_info(dst_iter_c_d, KuDNN::Layout::ABCD));
    }
    KuDNN::TensorInfo mem_desc_bias_kdnn = get_kdnn_tensor_info(bias_d);
    if (KuDNN::Status::SUCCESS !=
        KuDNN::RnnFWD::ValidateInput(mem_desc_src_kdnn, mem_desc_weights_kdnn, mem_desc_dst_kdnn, mem_desc_bias_kdnn,
            kdnnRnn, get_kdnn_rnn_alg(rd.cell_kind), get_kdnn_rnn_activation_alg(rd.activation_kind))) {
        return {false, nullptr};
    } else {
        return {true, new KuDNN::RnnFWD{mem_desc_src_kdnn, mem_desc_weights_kdnn, mem_desc_dst_kdnn, mem_desc_bias_kdnn,
        kdnnRnn, get_kdnn_rnn_alg(rd.cell_kind), get_kdnn_rnn_activation_alg(rd.activation_kind)}};
    }
}

std::pair<bool, KuDNN::RnnBWD *> convert_to_kdnn_rnn_bwd(void *rnn_ptr,
    const rnn_desc_t &rd, const primitive_attr_t &attr,
    const memory_desc_wrapper &src_layer_d, const memory_desc_wrapper &src_iter_d,
    const memory_desc_wrapper &src_iter_c_d, const memory_desc_wrapper &weights_layer_d,
    const memory_desc_wrapper &weights_iter_d, const memory_desc_wrapper &weights_projection_d,
    const memory_desc_wrapper &dst_layer_d, const memory_desc_wrapper &dst_iter_d,
    const memory_desc_wrapper &dst_iter_c_d, const memory_desc_wrapper &bias_d,
    const memory_desc_wrapper &diff_dst_layer_d, const memory_desc_wrapper &diff_dst_iter_d,
    const memory_desc_wrapper &diff_dst_iter_c_d,
    const memory_desc_wrapper &diff_src_layer_d, const memory_desc_wrapper &diff_src_iter_d,
    const memory_desc_wrapper &diff_src_iter_c_d,
    const memory_desc_wrapper &diff_weights_layer_d, const memory_desc_wrapper &diff_weights_iter_d,
    const memory_desc_wrapper &diff_bias_d) noexcept(false)
{
    if (!common_tensor_checks_without_layout(src_layer_d, src_iter_d,
        weights_layer_d, weights_iter_d,
        dst_layer_d, dst_iter_d,
        bias_d)) {
        return {false, nullptr};
    }

    if (rd.cell_kind == alg_kind::vanilla_lstm) {
        if (!common_tensor_checks_without_layout(src_iter_c_d, dst_iter_c_d)) {
            return {false, nullptr};
        }
    }

    rnn_utils::rnn_conf_t &rnn = *static_cast<rnn_utils::rnn_conf_t *>(rnn_ptr);
    KuDNN::RnnCfg kdnnRnn;
    kdnnRnn.nDir = rnn.n_dir;
    kdnnRnn.nLayer = rnn.n_layer;
    kdnnRnn.nIter = rnn.n_iter;
    kdnnRnn.nGates = rnn.n_gates;
    kdnnRnn.nBias = rnn.n_bias;
    kdnnRnn.dhc = rnn.dhc;
    kdnnRnn.dlc = rnn.dlc;
    kdnnRnn.dic = rnn.dic;
    kdnnRnn.mb = rnn.mb;
    kdnnRnn.slc = rnn.slc;
    kdnnRnn.weightsLayerLd = rnn.weights_layer_ld;
    kdnnRnn.scratchGatesLd = rnn.scratch_gates_ld;
    kdnnRnn.sic = rnn.sic;
    kdnnRnn.weightsIterLd = rnn.weights_iter_ld;
    kdnnRnn.srcLayerLd = rnn.src_layer_ld_;
    kdnnRnn.dstIterLd = rnn.dst_iter_ld_;
    kdnnRnn.wsStatesLayerLd = rnn.ws_states_layer_ld;
    kdnnRnn.srcIterLd = rnn.src_iter_ld_;
    kdnnRnn.dstLayerLd = rnn.dst_layer_ld_;
    kdnnRnn.wsStatesIterLd = rnn.ws_states_iter_ld;
    kdnnRnn.execDir = static_cast<KuDNN::Detail::ExecutionDirectionT>(rnn.exec_dir);

    kdnnRnn.wsDiffStatesLayerLd = rnn.ws_diff_states_layer_ld;
    kdnnRnn.wsDiffStatesIterLd = rnn.ws_diff_states_iter_ld;
    kdnnRnn.wsDiffStatesIterCLd = rnn.ws_diff_states_iter_c_ld;
    kdnnRnn.wsDiffStatesLayerSize = rnn.ws_diff_states_layer_size;
    kdnnRnn.wsDiffStatesIterSize = rnn.ws_diff_states_iter_size;
    kdnnRnn.wsDiffStatesIterCSize = rnn.ws_diff_states_iter_c_size;
    kdnnRnn.diffWeightsLayerLd = rnn.diff_weights_layer_ld;
    kdnnRnn.diffWeightsIterLd = rnn.diff_weights_iter_ld;

    kdnnRnn.scratchGatesSize = rnn.scratch_gates_size;
    kdnnRnn.scratchGatesNld = rnn.scratch_gates_nld;
    kdnnRnn.wsGatesLd = rnn.ws_gates_ld;
    kdnnRnn.wsGatesNld = rnn.ws_gates_nld;
    kdnnRnn.srcIterCLd = rnn.src_iter_c_ld_;
    kdnnRnn.dstIterCLd = rnn.dst_iter_c_ld_;
    kdnnRnn.scratchCellSize = rnn.scratch_cell_size;

    kdnnRnn.scratchDiffHtLd = rnn.scratch_diff_ht_ld;
    kdnnRnn.wsHtLd = rnn.ws_ht_ld;
    kdnnRnn.wsHtNld = rnn.ws_ht_nld;
    kdnnRnn.diffWeightsProjectionLd = rnn.diff_weights_projection_ld;
    kdnnRnn.diffWeightsProjectionNld = rnn.diff_weights_projection_nld;
    kdnnRnn.weightsProjectionLd = rnn.weights_projection_ld;
    kdnnRnn.isTraining = rnn.is_training;
    kdnnRnn.wsPerCell = rnn.ws_per_cell;
    kdnnRnn.diffWeightsOverwrite = rnn.diff_weights_overwrite;
    kdnnRnn.gatesFuncLinearMode = attr.rnn_tparams_.test_mode_;
    kdnnRnn.alpha = rd.alpha;
    if (attr.rnn_tparams_.scales_ != nullptr) {
        kdnnRnn.isScale = true;
        kdnnRnn.scales[0] = attr.rnn_tparams_.scales_[0];
        kdnnRnn.scales[1] = attr.rnn_tparams_.scales_[1];
        kdnnRnn.scales[2] = attr.rnn_tparams_.scales_[2];
        kdnnRnn.scales[3] = attr.rnn_tparams_.scales_[3];
    }
    kdnnRnn.cscale = attr.rnn_tparams_.cscale_;

    size_t scratchpadSize, workspaceSize;
    rnn_utils::set_offsets(rnn, kdnnRnn.wsGatesOffset, kdnnRnn.wsHtOffset,
        kdnnRnn.wsStatesLayerOffset, kdnnRnn.wsStatesIterOffset,
        kdnnRnn.wsStatesIterCOffset, kdnnRnn.wsDiffStatesLayerOffset,
        kdnnRnn.wsDiffStatesIterOffset, kdnnRnn.wsDiffStatesIterCOffset,
        kdnnRnn.wsGridCompOffset, kdnnRnn.wsBiasOffset, kdnnRnn.scratchGatesOffset,
        kdnnRnn.scratchHtOffset, kdnnRnn.scratchDiffHtOffset,
        kdnnRnn.scratchCellOffset, scratchpadSize, workspaceSize);

    std::vector<KuDNN::TensorInfo> mem_desc_src_kdnn;
    mem_desc_src_kdnn.push_back(get_kdnn_tensor_info(src_layer_d, KuDNN::Layout::ABC));
    mem_desc_src_kdnn.push_back(get_kdnn_tensor_info(src_iter_d, KuDNN::Layout::ABCD));
    if (rd.cell_kind == alg_kind::vanilla_lstm) {
        mem_desc_src_kdnn.push_back(get_kdnn_tensor_info(src_iter_c_d, KuDNN::Layout::ABCD));
    }

    std::vector<KuDNN::TensorInfo> mem_desc_weights_kdnn;
    mem_desc_weights_kdnn.push_back(get_kdnn_tensor_info(weights_layer_d, KuDNN::Layout::ABDEC));
    mem_desc_weights_kdnn.push_back(get_kdnn_tensor_info(weights_iter_d, KuDNN::Layout::ABDEC));

    std::vector<KuDNN::TensorInfo> mem_desc_dst_kdnn;
    mem_desc_dst_kdnn.push_back(get_kdnn_tensor_info(dst_layer_d, KuDNN::Layout::ABC));
    mem_desc_dst_kdnn.push_back(get_kdnn_tensor_info(dst_iter_d, KuDNN::Layout::ABCD));
    if (rd.cell_kind == alg_kind::vanilla_lstm) {
        mem_desc_dst_kdnn.push_back(get_kdnn_tensor_info(dst_iter_c_d, KuDNN::Layout::ABCD));
    }
    KuDNN::TensorInfo mem_desc_bias_kdnn = get_kdnn_tensor_info(bias_d);

    std::vector<KuDNN::TensorInfo> mem_desc_diff_src_kdnn;
    mem_desc_diff_src_kdnn.push_back(get_kdnn_tensor_info(diff_src_layer_d, KuDNN::Layout::ABC));
    mem_desc_diff_src_kdnn.push_back(get_kdnn_tensor_info(diff_src_iter_d, KuDNN::Layout::ABCD));
    if (rd.cell_kind == alg_kind::vanilla_lstm) {
        mem_desc_diff_src_kdnn.push_back(get_kdnn_tensor_info(diff_src_iter_c_d, KuDNN::Layout::ABCD));
    }

    std::vector<KuDNN::TensorInfo> mem_desc_diff_weights_kdnn;
    mem_desc_diff_weights_kdnn.push_back(get_kdnn_tensor_info(diff_weights_layer_d, KuDNN::Layout::ABCDE));
    mem_desc_diff_weights_kdnn.push_back(get_kdnn_tensor_info(diff_weights_iter_d, KuDNN::Layout::ABCDE));

    std::vector<KuDNN::TensorInfo> mem_desc_diff_dst_kdnn;
    mem_desc_diff_dst_kdnn.push_back(get_kdnn_tensor_info(diff_dst_layer_d, KuDNN::Layout::ABC));
    mem_desc_diff_dst_kdnn.push_back(get_kdnn_tensor_info(diff_dst_iter_d, KuDNN::Layout::ABCD));
    if (rd.cell_kind == alg_kind::vanilla_lstm) {
        mem_desc_diff_dst_kdnn.push_back(get_kdnn_tensor_info(diff_dst_iter_c_d, KuDNN::Layout::ABCD));
    }

    KuDNN::TensorInfo mem_desc_diff_bias_kdnn = get_kdnn_tensor_info(diff_bias_d);
    if (KuDNN::Status::SUCCESS !=
        KuDNN::RnnBWD::ValidateInput(mem_desc_src_kdnn, mem_desc_weights_kdnn, mem_desc_dst_kdnn, mem_desc_bias_kdnn,
            mem_desc_diff_src_kdnn, mem_desc_diff_weights_kdnn, mem_desc_diff_dst_kdnn, mem_desc_diff_bias_kdnn,
            get_kdnn_rnn_alg(rd.cell_kind), get_kdnn_rnn_activation_alg(rd.activation_kind))) {
        return {false, nullptr};
    } else {
        return {true, new KuDNN::RnnBWD{mem_desc_src_kdnn, mem_desc_weights_kdnn, mem_desc_dst_kdnn, mem_desc_bias_kdnn,
        mem_desc_diff_src_kdnn, mem_desc_diff_weights_kdnn, mem_desc_diff_dst_kdnn, mem_desc_diff_bias_kdnn,
        kdnnRnn, get_kdnn_rnn_alg(rd.cell_kind), get_kdnn_rnn_activation_alg(rd.activation_kind)}};
    }
}

} // namespace kdnn_utils
} // namespace aarch64
} // namespace cpu
} // namespace impl
} // namespace dnnl
