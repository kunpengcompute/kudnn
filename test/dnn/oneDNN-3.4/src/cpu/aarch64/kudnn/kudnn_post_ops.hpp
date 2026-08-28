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

#ifndef CPU_AARCH64_KUDNN_POST_OPS_HPP
#define CPU_AARCH64_KUDNN_POST_OPS_HPP

#include <memory>

#include "kudnn.hpp"

#include "cpu/cpu_primitive.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace aarch64 {

struct kdnn_post_ops_t {

    kdnn_post_ops_t() = default;

    status_t init(engine_t *engine, post_ops_t &post_ops,
            const memory_desc_t &dst_md);

    bool has_sum() const;

    status_t create_resource(
            engine_t *engine, resource_mapper_t &mapper) const;

    status_t execute(const exec_ctx_t &ctx, void *src_orig, void *dst = nullptr) const;

    size_t len() const;

    const std::vector<std::shared_ptr<primitive_t>>& get_post_op_primitives() const;

private:
    // Index of the sum post op if there is one, < 0 means no sum
    int sum_index = -1;
    float sum_scale = 1.f;
    data_type_t dst_data_type;
    // Vector of primitives used to execute the post ops. They are constructed
    // in init to be either kdnn_binary_t (for sum, add, sub, div, mul, min and
    // max) or kdnn_eltwise_fwd_t (for relu, elu, tanh, square, abs etc)
    std::vector<std::shared_ptr<primitive_t>> post_op_primitives;
};

} // namespace aarch64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif // CPU_AARCH64_KUDNN_POST_OPS_HPP
