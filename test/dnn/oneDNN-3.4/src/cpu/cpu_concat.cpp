/*******************************************************************************
* Copyright 2017-2022 Intel Corporation
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

#include "common/impl_list_item.hpp"
#include "cpu/ref_concat.hpp"
#include "cpu/simple_concat.hpp"

#if DNNL_AARCH64_USE_KUDNN
#include "cpu/aarch64/kudnn/kudnn_concat.hpp"
#endif
using namespace dnnl::impl::cpu::aarch64;

namespace dnnl {
namespace impl {
namespace cpu {

namespace {
using namespace dnnl::impl::data_type;
#define INSTANCE(...) \
    impl_list_item_t(impl_list_item_t::concat_type_deduction_helper_t< \
            __VA_ARGS__::pd_t>()),
#define CONCAT_INSTANCE_AARCH64_KUDNN(...) DNNL_AARCH64_KUDNN_ONLY(INSTANCE(__VA_ARGS__))
// clang-format off
constexpr impl_list_item_t cpu_concat_impl_list[] = REG_CONCAT_P({
        CONCAT_INSTANCE_AARCH64_KUDNN(kdnn_concat_t)
        INSTANCE(simple_concat_t<f32>)
        INSTANCE(simple_concat_t<u8>)
        INSTANCE(simple_concat_t<s8>)
        INSTANCE(simple_concat_t<s32>)
        INSTANCE(simple_concat_t<bf16>)
        INSTANCE(simple_concat_t<f16>)
        INSTANCE(ref_concat_t)
        nullptr,
});
// clang-format on
#undef INSTANCE
} // namespace

const impl_list_item_t *
cpu_engine_impl_list_t::get_concat_implementation_list() {
    return cpu_concat_impl_list;
}

} // namespace cpu
} // namespace impl
} // namespace dnnl
