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

#include "cpu/aarch64/kudnn/kudnn_binary.hpp"
#include "cpu/aarch64/kudnn/kudnn_eltwise.hpp"
#include "cpu/aarch64/kudnn/kudnn_prelu.hpp"
#include "kudnn_post_ops.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace aarch64 {

status_t aarch64::kdnn_post_ops_t::init(engine_t *engine, post_ops_t &post_ops, const memory_desc_t &dst_md)
{
    CHECK(post_ops.set_default_formats(&dst_md));
    dst_data_type = dst_md.data_type;

    sum_index = -1;
    post_op_primitives = {};

    for (int i = 0; i < post_ops.len(); i++) {
        auto &po = post_ops.entry_[i];

        if (po.is_binary()) {
            binary_desc_t po_desc;
            po_desc.primitive_kind = primitive_kind::binary;
            po_desc.alg_kind = po.binary.alg;
            memory_desc_t src0_md;
            memory_desc_init_by_md_and_dt(src0_md, dst_md, data_type_t::dnnl_f32); // Change the input precision to f32
            po_desc.src_desc[0] = src0_md;
            po_desc.src_desc[1] = po.binary.src1_desc;
            po_desc.dst_desc = dst_md;
            auto attr = dnnl_primitive_attr();
            typename kdnn_binary_t::pd_t kdnn_binary_pd(
                &po_desc, &attr, nullptr);
            CHECK(kdnn_binary_pd.init(engine));

            auto kdnn_binary
                = std::make_shared<kdnn_binary_t>(&kdnn_binary_pd);
            CHECK(kdnn_binary->init(engine));
            post_op_primitives.push_back(kdnn_binary);
        } else if (po.is_eltwise()) {
            // Eltwise post op scale must be 1 (no scale)
            if (po.eltwise.scale != 1.0f) {
                return dnnl::impl::status::unimplemented;
            }

            eltwise_desc_t eltwise_desc;
            eltwise_desc.primitive_kind = primitive_kind::eltwise;
            switch (po.eltwise.alg) {
                case dnnl_eltwise_relu_use_dst_for_bwd: eltwise_desc.alg_kind = dnnl_eltwise_relu; break;
                case dnnl_eltwise_tanh_use_dst_for_bwd: eltwise_desc.alg_kind = dnnl_eltwise_tanh; break;
                case dnnl_eltwise_elu_use_dst_for_bwd: eltwise_desc.alg_kind = dnnl_eltwise_elu; break;
                case dnnl_eltwise_sqrt_use_dst_for_bwd: eltwise_desc.alg_kind = dnnl_eltwise_sqrt; break;
                case dnnl_eltwise_logistic_use_dst_for_bwd: eltwise_desc.alg_kind = dnnl_eltwise_logistic; break;
                case dnnl_eltwise_exp_use_dst_for_bwd: eltwise_desc.alg_kind = dnnl_eltwise_exp; break;
                case dnnl_eltwise_clip_v2_use_dst_for_bwd: eltwise_desc.alg_kind = dnnl_eltwise_clip_v2; break;
                default:
                    eltwise_desc.alg_kind = po.eltwise.alg;
                    break;
            }
            eltwise_desc.alpha = po.eltwise.alpha;
            eltwise_desc.beta = po.eltwise.beta;
            // When will we support fp16: pass eltwise a desc with
            // f32 datatype to perform the operation in fp32 rather than fp16
            memory_desc_t src0_md;
            memory_desc_init_by_md_and_dt(src0_md, dst_md, data_type_t::dnnl_f32); // Change the input precision to f32
            eltwise_desc.src_desc = src0_md;
            eltwise_desc.dst_desc = dst_md;
            eltwise_desc.prop_kind = prop_kind_t::dnnl_forward;
            auto empty_attr = dnnl_primitive_attr();
            typename kdnn_eltwise_fwd_t::pd_t kdnn_eltwise_pd(
                    &eltwise_desc, &empty_attr, nullptr);
            CHECK(kdnn_eltwise_pd.init(engine));
            auto kdnn_eltwise
                    = std::make_shared<kdnn_eltwise_fwd_t>(&kdnn_eltwise_pd);
            CHECK(kdnn_eltwise->init(engine));
            post_op_primitives.push_back(kdnn_eltwise);
        } else if (po.is_prelu()) {
            prelu_desc_t po_desc;
            po_desc.primitive_kind = primitive_kind::prelu;
            memory_desc_t src0_md;
            memory_desc_init_by_md_and_dt(src0_md, dst_md, data_type_t::dnnl_f32);
            po_desc.src_desc = src0_md;

            memory_desc_t wei_md;
            dnnl_dim_t dims[dst_md.ndims];
            for (int i = 0; i < dst_md.ndims; ++i) {
                dims[i] = 1;
            }
            switch (kdnn_utils::get_kdnn_prelu_mask(po.prelu.mask)) {
                case KuDNN::PReLUMask::COMMON: {
                    break;
                }
                case KuDNN::PReLUMask::PER_DIM_0: {
                    dims[0] = dst_md.dims[0];
                    break;
                }
                case KuDNN::PReLUMask::PER_DIM_1: {
                    if (dst_md.ndims > 1) {
                        dims[1] = dst_md.dims[1];
                    }
                    break;
                }
                case KuDNN::PReLUMask::PER_DIM_01: {
                    dims[0] = dst_md.dims[0];
                    if (dst_md.ndims > 1) {
                        dims[1] = dst_md.dims[1];
                    }
                    break;
                }
                case KuDNN::PReLUMask::PER_DIM_2: {
                    if (dst_md.ndims > 2) {
                        dims[2] = dst_md.dims[2];
                    }
                    break;
                }
                case KuDNN::PReLUMask::PER_DIM_3: {
                    if (dst_md.ndims > 3) {
                        dims[3] = dst_md.dims[3];
                    }
                    break;
                }
                case KuDNN::PReLUMask::PER_DIM_4: {
                    if (dst_md.ndims > 4) {
                        dims[4] = dst_md.dims[4];
                    }
                    break;
                }
                case KuDNN::PReLUMask::PER_TENSOR: {
                    for (int i = 0; i < dst_md.ndims; ++i) {
                        dims[i] = dst_md.dims[i];
                    }
                    break;
                }
                default: {
                    break;
                }
            }
            dnnl::impl::format_tag_t tag;
            switch (dst_md.ndims) {
                case 1: {
                    tag = dnnl_a;
                    break;
                }
                case 2: {
                    tag = dnnl_ab;
                    break;
                }
                case 3: {
                    tag = dnnl_acb;
                    break;
                }
                case 4: {
                    tag = dnnl_acdb;
                    break;
                }
                case 5: {
                    tag = dnnl_acdeb;
                    break;
                }
                default: {
                    tag = dnnl_format_tag_undef;
                    break;
                }
            }
            memory_desc_init_by_tag(wei_md, dst_md.ndims,
                dims, data_type::f32, tag);

            po_desc.weights_desc = wei_md;
            po_desc.dst_desc = dst_md;
            po_desc.prop_kind = prop_kind_t::dnnl_forward;
            auto empty_attr = dnnl_primitive_attr();
            typename kdnn_prelu_fwd_t::pd_t kdnn_prelu_pd(
                    &po_desc, &empty_attr, nullptr);
            CHECK(kdnn_prelu_pd.init(engine));

            auto kdnn_prelu
                    = std::make_shared<kdnn_prelu_fwd_t>(&kdnn_prelu_pd);
            CHECK(kdnn_prelu->init(engine));
            post_op_primitives.push_back(kdnn_prelu);
        } else if (po.is_sum(false)) {
            if (sum_index >= 0) {
                return status::unimplemented;
            }

            sum_index = i;

            binary_desc_t po_desc;
            po_desc.primitive_kind = primitive_kind::binary;
            po_desc.alg_kind = alg_kind::binary_add;

            memory_desc_t src0_md;
            memory_desc_init_by_md_and_dt(src0_md, dst_md, data_type_t::dnnl_f32);

            po_desc.src_desc[0] = src0_md;
            po_desc.src_desc[1] = dst_md;
            po_desc.dst_desc = dst_md;
            auto attr = dnnl_primitive_attr();
            sum_scale = po.sum.scale;
            typename kdnn_binary_t::pd_t kdnn_binary_pd(
                &po_desc, &attr, nullptr);
            CHECK(kdnn_binary_pd.init(engine));

            auto kdnn_binary
                = std::make_shared<kdnn_binary_t>(&kdnn_binary_pd);
            CHECK(kdnn_binary->init(engine));
            post_op_primitives.push_back(kdnn_binary);
        } else {
            return status::unimplemented;
        }
    }

    return status::success;
}

bool aarch64::kdnn_post_ops_t::has_sum() const
{
    return sum_index >= 0;
}

status_t aarch64::kdnn_post_ops_t::create_resource(
    engine_t *engine, resource_mapper_t &mapper) const
{
    for (const auto &post_op : post_op_primitives) {
        CHECK(post_op->create_resource(engine, mapper));
    }
    return status::success;
}

size_t aarch64::kdnn_post_ops_t::len() const
{
    return post_op_primitives.size();
}

const std::vector<std::shared_ptr<primitive_t>>& aarch64::kdnn_post_ops_t::get_post_op_primitives() const
{
    return post_op_primitives;
}

status_t aarch64::kdnn_post_ops_t::execute(const exec_ctx_t &ctx, void *src_orig, void *dst) const
{
    int post_op_index = 0;
    void *src = src_orig; // f32 buffer

    if (dst == nullptr) {
        dst = src_orig;
    }

    for (auto &post_op : post_op_primitives) {
        if (post_op->kind() == primitive_kind::binary) {
            auto binary_post_op
                = static_cast<kdnn_binary_t *>(post_op.get());

            if (post_op_index == sum_index) {
                dst = CTX_OUT_MEM(void *, DNNL_ARG_DST);
                CHECK(binary_post_op->execute_forward(ctx, src,
                    dst, dst, sum_scale));
            } else {
                const void *src1 = CTX_IN_MEM(const void *,
                    (DNNL_ARG_ATTR_MULTIPLE_POST_OP(post_op_index)
                            | DNNL_ARG_SRC_1));
                CHECK(binary_post_op->execute_forward(ctx, src,
                    src1, dst));
            }
        } else if (post_op->kind() == primitive_kind::eltwise) {
            auto eltwise_post_op
                = static_cast<kdnn_eltwise_fwd_t *>(post_op.get());

            CHECK(eltwise_post_op->execute_forward(ctx, src, dst));
        } else if (post_op->kind() == primitive_kind::prelu) {
            auto prelu_post_op
                = static_cast<kdnn_prelu_fwd_t *>(post_op.get());

            const void *wei = CTX_IN_MEM(const void *,
                    (DNNL_ARG_ATTR_MULTIPLE_POST_OP(post_op_index)
                            | DNNL_ARG_WEIGHTS));
            CHECK(prelu_post_op->execute_forward(ctx, src, wei, dst));

        } else {
            return status::runtime_error;
        }

        ++post_op_index;
    }

    return status::success;
}

} // namespace aarch64
} // namespace cpu
} // namespace impl
} // namespace dnnl
