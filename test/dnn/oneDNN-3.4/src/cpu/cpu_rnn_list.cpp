/*******************************************************************************
 * Copyright 2019-2023 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *******************************************************************************/
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

#include "cpu/cpu_engine.hpp"

#include "cpu/rnn/ref_rnn.hpp"

#if DNNL_AARCH64 && DNNL_AARCH64_USE_KUDNN
#include "cpu/aarch64/kudnn/kudnn_rnn.hpp"
using namespace dnnl::impl::cpu::aarch64;
#endif  // DNNL_AARCH64_USE_KUDNN

namespace dnnl {
namespace impl {
namespace cpu {

namespace {
using namespace dnnl::impl::data_type;
using namespace dnnl::impl::prop_kind;

const std::map<pk_impl_key_t, std::vector<impl_list_item_t>> &impl_list_map()
{
    // clang-format off
    static std::map<pk_impl_key_t, std::vector<impl_list_item_t>> the_map =  REG_RNN_P({
        {{forward}, {
            CPU_INSTANCE_AARCH64_KUDNN(kdnn_rnn_fwd_t)
            CPU_INSTANCE(ref_rnn_fwd_bf16_t)
            CPU_INSTANCE(ref_rnn_fwd_f16_t)
            CPU_INSTANCE(ref_rnn_fwd_f32_t)
            CPU_INSTANCE(ref_rnn_fwd_s8s8_t)
            CPU_INSTANCE(ref_rnn_fwd_u8s8_t)
            nullptr,
        }},
        {{backward}, REG_BWD_PK({
            CPU_INSTANCE_AARCH64_KUDNN(kdnn_rnn_bwd_t)
            CPU_INSTANCE(ref_rnn_bwd_f32_t)
            CPU_INSTANCE(ref_rnn_bwd_bf16_t)
            CPU_INSTANCE(ref_rnn_bwd_f16_t)
            nullptr,
        })},
    });
    // clang-format on
    return the_map;
}
} // namespace

const impl_list_item_t *get_rnn_impl_list(const rnn_desc_t *desc) {
    static const impl_list_item_t empty_list[] = {nullptr};

    const bool is_fwd = utils::one_of(
        desc->prop_kind, forward_training, forward_inference);
    prop_kind_t prop_kind = is_fwd ? forward : backward;

    pk_impl_key_t key {prop_kind};

    const auto impl_list_it = impl_list_map().find(key);
    return impl_list_it != impl_list_map().cend() ? impl_list_it->second.data()
                                                  : empty_list;
}

} // namespace cpu
} // namespace impl
} // namespace dnnl
