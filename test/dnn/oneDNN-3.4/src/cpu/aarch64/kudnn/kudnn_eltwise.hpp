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

#ifndef CPU_AARCH64_KUDNN_ELTWISE_HPP
#define CPU_AARCH64_KUDNN_ELTWISE_HPP

#include <memory>

#include "kudnn.hpp"

#include "cpu/cpu_eltwise_pd.hpp"
#include "cpu/aarch64/kudnn/kudnn_utils.hpp"
#include "cpu/aarch64/kudnn/kudnn_post_ops.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace aarch64 {

struct kdnn_eltwise_fwd_resource_t : public resource_t {
    kdnn_eltwise_fwd_resource_t(const std::unique_ptr<KuDNN::ActivationLayerFWD> &kdnn_eltwise_prim) noexcept
        : kdnn_eltwise_obj_(new KuDNN::ActivationLayerFWD{*(kdnn_eltwise_prim.get())}) {}

    KuDNN::ActivationLayerFWD &get_kdnn_obj() const noexcept { return *kdnn_eltwise_obj_; }

    DNNL_DISALLOW_COPY_AND_ASSIGN(kdnn_eltwise_fwd_resource_t);

private:
    std::unique_ptr<KuDNN::ActivationLayerFWD> kdnn_eltwise_obj_;
}; // kdnn_eltwise_fwd_resource_t

struct kdnn_eltwise_fwd_t : public primitive_t {
    struct pd_t : public cpu_eltwise_fwd_pd_t {
        using cpu_eltwise_fwd_pd_t::cpu_eltwise_fwd_pd_t;

        // Need copy constructor because `unique_ptr` is a member of the class
        pd_t(const pd_t& p) : cpu_eltwise_fwd_pd_t(p) {
            if (p.kdnn_eltwise_prim_) {
                this->kdnn_eltwise_prim_.reset(new KuDNN::ActivationLayerFWD{*(p.kdnn_eltwise_prim_.get())});
                this->post_ops_ = p.post_ops_;
                this->need_po_wa_ = p.need_po_wa_;
                this->dst_size_ = p.dst_size_;
            }
        }

        DECLARE_COMMON_PD_T("kdnn", kdnn_eltwise_fwd_t);

        status_t init(engine_t *engine) {
            const memory_desc_wrapper src_d(src_md());
            data_type_t dst_dt = dst_md_.data_type;
            if (attr_.post_ops_.len() > 1) {
                return status::unimplemented; // Currently, only one post-ops is supported.
            }
            bool has_post_ops = (attr_.post_ops_.len() >= 1);
            if (has_post_ops) {
                dst_md_.data_type = data_type_t::dnnl_f32; // Temporarily change the output data type to f32
            }
            const memory_desc_wrapper dst_d(dst_md());

            bool ok = is_fwd() && !has_zero_dim_memory() && set_default_formats_common() &&
                attr()->has_default_values(primitive_attr_t::skip_mask_t::post_ops) && (attr_.post_ops_.len() <= 1);
            if (!ok) return status::unimplemented;
            need_po_wa_ = false;
            dst_size_ = 0;
            auto&& eltwise_fwd = kdnn_utils::convert_to_kdnn_act_fwd(src_d, dst_d, desc_.alg_kind, desc_.alpha, desc_.beta);
            if (!eltwise_fwd.first) {
                return status::unimplemented;
            } else {
                kdnn_eltwise_prim_.reset(eltwise_fwd.second);
                if (has_post_ops) {
                    dst_md_.data_type = dst_dt;
                    need_po_wa_ = true;
                    if ((dst_dt != data_type_t::dnnl_f32) || attr_.post_ops_.entry_[0].is_sum(false)) {
                        dst_size_ = dst_d.nelems() * types::data_type_size(data_type_t::dnnl_f32);
                    }
                }
                CHECK(post_ops_.init(engine, attr_.post_ops_, dst_md_));
                return status::success;
            }
        }

        // We use `unique_ptr` because `activation_layer_info` doesn't have default constructor
        std::unique_ptr<KuDNN::ActivationLayerFWD> kdnn_eltwise_prim_;

        friend struct kdnn_post_ops_t;
        bool need_po_wa_;
        size_t dst_size_;
        kdnn_post_ops_t post_ops_;
    }; // pd_t

    kdnn_eltwise_fwd_t(const pd_t *kpd) : primitive_t(kpd) {}

    status_t execute(const exec_ctx_t &ctx) const override {
        return execute_forward(ctx);
    }

    status_t create_resource(
            engine_t *engine, resource_mapper_t &mapper) const override {
        if (mapper.has_resource(this)) return status::success;

        mapper.add(this, std::make_unique<kdnn_eltwise_fwd_resource_t>(pd()->kdnn_eltwise_prim_));
        CHECK(pd()->post_ops_.create_resource(engine, mapper));

        return status::success;
    }

private:
    // execute_forward has to be const thus mutability of mtx
    mutable std::mutex mtx;

    status_t execute_forward(const exec_ctx_t &ctx) const {
        const void *src = CTX_IN_MEM(const void *, DNNL_ARG_SRC);
        void *dst = CTX_OUT_MEM(void *, DNNL_ARG_DST);

        return execute_forward(ctx, src, dst);
    }

    // Execute forward with arbitrary src and dst
    status_t execute_forward(
            const exec_ctx_t &ctx, const void *src, void *dst) const {
        // Lock here is needed because resource_mapper does not support concurrent access.
        std::lock_guard<std::mutex> _lock {this->mtx};
 
        KuDNN::ActivationLayerFWD &kdnn_obj = (ctx.get_resource_mapper()->get<kdnn_eltwise_fwd_resource_t>(this))->get_kdnn_obj();

        if (pd()->need_po_wa_) {
            try {
                if (pd()->dst_size_ != 0) {
                    std::unique_ptr<float, KuDNN::Service::Deallocator<float>> tmp_buffer(
                        static_cast<float*>(KuDNN::Service::AlignedAlloc(pd()->dst_size_)));
                    kdnn_obj.Run(src, tmp_buffer.get());
                    pd()->post_ops_.execute(ctx, tmp_buffer.get(), dst);
                } else {
                    kdnn_obj.Run(src, dst);
                    pd()->post_ops_.execute(ctx, dst, dst);
                }
            } catch (const std::exception& e) {
                return status::runtime_error;
            }
        } else {
            try {
                kdnn_obj.Run(src, dst);
            } catch (const std::exception& e) {
                return status::runtime_error;
            }
        }

        return status::success;
    }

    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }

    friend struct kdnn_post_ops_t;
}; // kdnn_eltwise_fwd_t

struct kdnn_eltwise_bwd_resource_t : public resource_t {
    kdnn_eltwise_bwd_resource_t(const std::unique_ptr<KuDNN::ActivationLayerBWD> &kdnn_eltwise_prim) noexcept
        : kdnn_eltwise_obj_(new KuDNN::ActivationLayerBWD{*(kdnn_eltwise_prim.get())}) {}

    KuDNN::ActivationLayerBWD &get_kdnn_obj() const noexcept { return *kdnn_eltwise_obj_; }

    DNNL_DISALLOW_COPY_AND_ASSIGN(kdnn_eltwise_bwd_resource_t);

private:
    std::unique_ptr<KuDNN::ActivationLayerBWD> kdnn_eltwise_obj_;
}; // kdnn_eltwise_bwd_resource_t

struct kdnn_eltwise_bwd_t : public primitive_t { 
    struct pd_t : public cpu_eltwise_bwd_pd_t {
        using cpu_eltwise_bwd_pd_t::cpu_eltwise_bwd_pd_t;

        // Need copy constructor because `unique_ptr` is a member of the class
        pd_t(const pd_t& p) : cpu_eltwise_bwd_pd_t(p) {
            if (p.kdnn_eltwise_prim_) {
                this->kdnn_eltwise_prim_.reset(new KuDNN::ActivationLayerBWD{*(p.kdnn_eltwise_prim_.get())});
            }
        }

        DECLARE_COMMON_PD_T("kdnn", kdnn_eltwise_bwd_t);

        status_t init(engine_t *engine) {
            const memory_desc_wrapper diff_src_d = diff_src_md();
            const memory_desc_wrapper diff_dst_d = diff_dst_md();
            const memory_desc_wrapper src_d = (utils::one_of(desc_.alg_kind, alg_kind::eltwise_linear))
                ? diff_dst_md()   // Real src will not be used by algorithm
                : ((!is_fwd() && use_dst()) ? dst_md() : src_md());
            bool ok = !is_fwd() && !has_zero_dim_memory() && attr()->has_default_values()
                    && set_default_formats_common();
            if (!ok) return status::unimplemented;
            if (!is_fwd() && use_dst()) {
                auto&& eltwise_bwd = kdnn_utils::convert_to_kdnn_act_bwd_with_dst(diff_src_d, diff_dst_d, src_d, desc_.alg_kind, desc_.alpha, desc_.beta);
                if (!eltwise_bwd.first) {
                    return status::unimplemented;
                } else {
                    kdnn_eltwise_prim_.reset(eltwise_bwd.second);
                    return status::success;
                }
            } else {
                auto&& eltwise_bwd = kdnn_utils::convert_to_kdnn_act_bwd(diff_src_d, diff_dst_d, src_d, desc_.alg_kind, desc_.alpha, desc_.beta);
                if (!eltwise_bwd.first) {
                    return status::unimplemented;
                } else {
                    kdnn_eltwise_prim_.reset(eltwise_bwd.second);
                    return status::success;
                }
            }
            return status::success;
        }

        // We use `unique_ptr` because `activation_layer_info` doesn't have default constructor
        std::unique_ptr<KuDNN::ActivationLayerBWD> kdnn_eltwise_prim_;

        friend struct kdnn_post_ops_t;
    }; // pd_t

    kdnn_eltwise_bwd_t(const pd_t *kpd) : primitive_t(kpd) {}

    status_t execute(const exec_ctx_t &ctx) const override {
        const void *src = pd()->use_dst() ? CTX_IN_MEM(const void *, DNNL_ARG_DST)
                                          : CTX_IN_MEM(const void *, DNNL_ARG_SRC);
        const void *diff_dst = CTX_IN_MEM(const void *, DNNL_ARG_DIFF_DST);
        void *diff_src = CTX_OUT_MEM(void *, DNNL_ARG_DIFF_SRC);
        return execute(ctx, diff_src, diff_dst, src);
    }

    status_t create_resource(
            engine_t *engine, resource_mapper_t &mapper) const override {
        if (mapper.has_resource(this)) return status::success;

        mapper.add(this, std::make_unique<kdnn_eltwise_bwd_resource_t>(pd()->kdnn_eltwise_prim_));

        return status::success;
    }

private:
    // execute_backward has to be const thus mutability of mtx
    mutable std::mutex mtx;

    // Execute backward with arbitrary src and dst
    status_t execute(
            const exec_ctx_t &ctx, void *diff_src, const void *diff_dst, const void *src) const {
        // Lock here is needed because resource_mapper does not support concurrent access.
        std::lock_guard<std::mutex> _lock {this->mtx};

        KuDNN::ActivationLayerBWD &kdnn_obj = (ctx.get_resource_mapper()->get<kdnn_eltwise_bwd_resource_t>(this))->get_kdnn_obj();
        try {
            kdnn_obj.Run(diff_src, diff_dst, src);
        } catch (const std::exception& e) {
            return status::runtime_error;
        }

        return status::success;
    }

    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }

    friend struct kdnn_post_ops_t;
}; // kdnn_eltwise_bwd_t

} // namespace aarch64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif // CPU_AARCH64_KUDNN_ELTWISE_HPP
