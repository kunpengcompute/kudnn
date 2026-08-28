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

std::pair<bool, KuDNN::ReorderLayer*> convert_to_kdnn_reorder(const memory_desc_wrapper& mem_desc_src,
    const memory_desc_wrapper& mem_desc_dst) noexcept(false)
{

#if DNNL_CPU_THREADING_RUNTIME == DNNL_RUNTIME_THREADPOOL
    dnnl::threadpool_interop::threadpool_iface *tp = dnnl::impl::threadpool_utils::get_active_threadpool();
    KuDNN::Threading::ThreadpoolIface* threadpool =
        reinterpret_cast<KuDNN::Threading::ThreadpoolIface *>(reinterpret_cast<void *>(tp));
    KuDNN::Threading::DeactivateThreadpool();
    KuDNN::Threading::ActivateThreadpool(threadpool);
#endif

    using namespace format_tag;
    auto src_tag = get_layout(mem_desc_src);
    if (utils::one_of(format_tag::undef, src_tag)) {
        return {false, nullptr};
    }
    auto dst_tag = get_layout(mem_desc_dst);
    if (utils::one_of(format_tag::undef, dst_tag)) {
        return {false, nullptr};
    }
    if (!common_tensor_checks(mem_desc_src, mem_desc_dst)) {
        return {false, nullptr};
    }
    KuDNN::TensorInfo src = KuDNN::TensorInfo(get_kdnn_tensor_info(mem_desc_src).GetDims(),
        get_kdnn_data_t(mem_desc_src.data_type()), get_kdnn_layout(src_tag));
    KuDNN::TensorInfo dst = KuDNN::TensorInfo(get_kdnn_tensor_info(mem_desc_dst).GetDims(),
        get_kdnn_data_t(mem_desc_dst.data_type()), get_kdnn_layout(dst_tag));
    if (KuDNN::Status::SUCCESS != KuDNN::ReorderLayer::ValidateInput(src, dst)) {
        return {false, nullptr};
    } else {
        return {true, new KuDNN::ReorderLayer{src, dst}};
    }
}

} // namespace kdnn_utils
} // namespace aarch64
} // namespace cpu
} // namespace impl
} // namespace dnnl
