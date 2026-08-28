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

void set_kdnn_threading() noexcept {
    if (KuDNN::Threading::GetThreadingControlStatus() == KuDNN::Threading::ThreadingControl::KUDNN_DEFAULT) {
        KuDNN::Threading::SetMaxNumThreads(dnnl_get_current_num_threads());
    }
}

} // namespace kdnn_utils

} // namespace aarch64
} // namespace cpu
} // namespace impl
} // namespace dnnl
