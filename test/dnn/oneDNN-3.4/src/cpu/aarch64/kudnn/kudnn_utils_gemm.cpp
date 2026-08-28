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

std::pair<bool, KuDNN::Gemm*> convert_to_kdnn_gemm(const memory_desc_wrapper& mem_desc_src,
    const memory_desc_wrapper& mem_desc_wei, const memory_desc_wrapper& mem_desc_dst,
    const memory_desc_wrapper& mem_desc_bias, const primitive_attr_t& attributes) noexcept(false) {
    if (!common_tensor_checks(mem_desc_src, mem_desc_wei, mem_desc_dst)) {
        return {false, nullptr};
    }

#if DNNL_CPU_THREADING_RUNTIME == DNNL_RUNTIME_THREADPOOL
    dnnl::threadpool_interop::threadpool_iface *tp = dnnl::impl::threadpool_utils::get_active_threadpool();
    KuDNN::Threading::ThreadpoolIface* threadpool =
        reinterpret_cast<KuDNN::Threading::ThreadpoolIface *>(reinterpret_cast<void *>(tp));
    KuDNN::Threading::DeactivateThreadpool();
    KuDNN::Threading::ActivateThreadpool(threadpool);
#endif

    if (mem_desc_bias == &glob_zero_md) {
        if (KuDNN::Status::SUCCESS != KuDNN::Gemm::ValidateInput(get_kdnn_tensor_info(mem_desc_src),
            get_kdnn_tensor_info(mem_desc_wei), get_kdnn_tensor_info(mem_desc_dst),
            get_kdnn_attributes(attributes))) {
            return {false, nullptr};
        } else {
            return {true, new KuDNN::Gemm{get_kdnn_tensor_info(mem_desc_src), get_kdnn_tensor_info(mem_desc_wei),
                get_kdnn_tensor_info(mem_desc_dst), get_kdnn_attributes(attributes)}};
        }
    } else {
        if (!common_tensor_checks(mem_desc_bias)) {
            return {false, nullptr};
        }
        if (KuDNN::Status::SUCCESS != KuDNN::Gemm::ValidateInput(get_kdnn_tensor_info(mem_desc_src),
            get_kdnn_tensor_info(mem_desc_wei), get_kdnn_tensor_info(mem_desc_dst),
            get_kdnn_tensor_info(mem_desc_bias), get_kdnn_attributes(attributes))) {
            return {false, nullptr};
        } else {
            return {true, new KuDNN::Gemm{get_kdnn_tensor_info(mem_desc_src), get_kdnn_tensor_info(mem_desc_wei),
                get_kdnn_tensor_info(mem_desc_dst), get_kdnn_tensor_info(mem_desc_bias),
                get_kdnn_attributes(attributes)}};
        }
    }
}

} // namespace kdnn_utils

} // namespace aarch64
} // namespace cpu
} // namespace impl
} // namespace dnnl
