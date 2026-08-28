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

#ifndef CPU_AARCH64_KUDNN_INNER_PRODUCT_HPP
#define CPU_AARCH64_KUDNN_INNER_PRODUCT_HPP

#include "kudnn.hpp"

#include "cpu/cpu_inner_product_pd.hpp"
#include "cpu/aarch64/kudnn/kudnn_utils.hpp"
#include "cpu/aarch64/kudnn/kudnn_post_ops.hpp"
#include "cpu/cpu_primitive.hpp"
#include "cpu/scale_utils.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace aarch64 {

struct kdnn_ip_fwd_resource_t : public resource_t {
    kdnn_ip_fwd_resource_t(const std::unique_ptr<KuDNN::InnerProductLayerFWD> &kdnn_ip_prim) noexcept
        : kdnn_ip_obj_(new KuDNN::InnerProductLayerFWD{*(kdnn_ip_prim.get())}) {}

    KuDNN::InnerProductLayerFWD &get_kdnn_obj() const noexcept { return *kdnn_ip_obj_; }

    DNNL_DISALLOW_COPY_AND_ASSIGN(kdnn_ip_fwd_resource_t);

private:
    std::unique_ptr<KuDNN::InnerProductLayerFWD> kdnn_ip_obj_;
}; // kdnn_ip_fwd_resource_t

struct kdnn_inner_product_fwd_t : public primitive_t {
    struct pd_t : public cpu_inner_product_fwd_pd_t {
        using cpu_inner_product_fwd_pd_t::cpu_inner_product_fwd_pd_t;

        // Need copy constructor because `unique_ptr` is a member of the class
        pd_t(const pd_t& p) : cpu_inner_product_fwd_pd_t(p) {
            if (p.kdnn_ip_prim_) {
                this->kdnn_ip_prim_.reset(new KuDNN::InnerProductLayerFWD{*(p.kdnn_ip_prim_.get())});
                this->post_ops_ = p.post_ops_;
                this->need_po_wa_ = p.need_po_wa_;
                this->dst_size_ = p.dst_size_;
            }
        }

        DECLARE_COMMON_PD_T("kdnn", kdnn_inner_product_fwd_t);

        status_t init(engine_t *engine) {
            const bool ok = is_fwd() && !has_zero_dim_memory() &&
                        (set_default_params() == status::success) &&
                        attr()->has_default_values(primitive_attr_t::skip_mask_t::post_ops |
                        dnnl_primitive_attr::skip_mask_t::scales_runtime) &&
                        (attr()->output_scales_.mask_ == 0) &&
                        attr_scales_ok();

            if (!ok) return status::unimplemented;

            using namespace format_tag;
            auto dst_tag = memory_desc_matches_one_of_tag(dst_md_, nc, cn);

            if (utils::one_of(format_tag::undef, dst_tag)) {
                // Unsupported memory layout
                return dnnl::impl::status::unimplemented;
            }

            if ((src_md_.format_kind == format_kind_t::dnnl_format_kind_any) ||
                (src_md_.format_kind == format_kind_t::dnnl_format_kind_undef)) {
                CHECK(memory_desc_init_by_tag(src_md_, utils::pick(ndims() - 3, ncw, nchw, ncdhw)));
            }
            if ((weights_md_.format_kind == format_kind_t::dnnl_format_kind_any) ||
                (weights_md_.format_kind == format_kind_t::dnnl_format_kind_undef)) {
                CHECK(memory_desc_init_by_tag(weights_md_, utils::pick(ndims() - 3, ncw, nchw, ncdhw)));
            }

            const memory_desc_wrapper src_d(src_md());
            const memory_desc_wrapper weights_d(weights_md(0));
            if (attr_.post_ops_.len() > 1) {
                return status::unimplemented; // Currently, only one post-ops is supported.
            }
            const memory_desc_wrapper dst_d(dst_md());
            const memory_desc_wrapper bias_d(weights_md(1));
            if (attr_.post_ops_.len() > 0) {
                auto &po = attr_.post_ops_.entry_[0];
                if (po.is_binary()) {
                    auto &src1_desc = po.binary.src1_desc;
                    if (src1_desc.format_kind == dnnl_format_kind_any) {
                        src1_desc.format_kind = dnnl_blocked;
                        memory_desc_wrapper::compute_blocking(src1_desc,
                            dnnl::impl::cpu::aarch64::kdnn_utils::get_layout(dst_md()));
                    }
                }
            }
            need_po_wa_ = false;
            dst_size_ = 0;
            auto&& ip_fwd = kdnn_utils::convert_to_kdnn_ip_fwd(src_d,
                weights_d, dst_d, bias_d, attr_);
            if (!ip_fwd.first) {
                // This part planned to be removed when post-ops are supported in KuDNN
                data_type_t dst_dt = dst_md_.data_type;
                bool has_post_ops = (attr_.post_ops_.len() >= 1);
                if (has_post_ops) {
                    need_po_wa_ = true;
                    dst_md_.data_type = data_type_t::dnnl_f32; // Temporarily change the output data type to f32
                }
                auto&& ip_fwd_no_attr = kdnn_utils::convert_to_kdnn_ip_fwd(src_d, weights_d, dst_d, bias_d, {});
                if (!ip_fwd_no_attr.first) {
                    return status::unimplemented;
                } else {
                    kdnn_ip_prim_.reset(ip_fwd_no_attr.second);
                    if (has_post_ops) {
                        dst_md_.data_type = dst_dt;
                        if (dst_dt != data_type_t::dnnl_f32) {
                            dst_size_ = dst_d.nelems() * types::data_type_size(data_type_t::dnnl_f32);
                        }
                    }
                    CHECK(post_ops_.init(engine, attr_.post_ops_, dst_md_));
                }
            }
            else {
                kdnn_ip_prim_.reset(ip_fwd.second);
            }
            auto scratchpad = scratchpad_registry().registrar();
            book_precomputed_scales(scratchpad, attr()->scales_, 1);
            return status::success;
        }

        bool attr_scales_ok(const std::vector<int> &supported_args
                = {DNNL_ARG_SRC, DNNL_ARG_WEIGHTS}) const {
            bool ok = attr()->scales_.has_default_values(supported_args);
            for (auto arg : supported_args) {
                int mask = attr()->scales_.get(arg).mask_;
                ok = ok && (mask == 0);
            }
            return ok;
        }

        bool need_po_wa_;
        size_t dst_size_;
        kdnn_post_ops_t post_ops_;

        std::unique_ptr<KuDNN::InnerProductLayerFWD> kdnn_ip_prim_;

    }; // pd_t

    kdnn_inner_product_fwd_t(const pd_t *kpd) : primitive_t(kpd) {}

    status_t create_resource(
            engine_t *engine, resource_mapper_t &mapper) const override {
        if (mapper.has_resource(this)) return status::success;

        mapper.add(this, std::make_unique<kdnn_ip_fwd_resource_t>(pd()->kdnn_ip_prim_));
        CHECK(pd()->post_ops_.create_resource(engine, mapper));

        return status::success;
    }

    status_t execute(const exec_ctx_t &ctx) const override {
        return execute_forward(ctx);
    }

private:
    mutable std::mutex mtx;

    status_t execute_forward(const exec_ctx_t &ctx) const {
        const void *src = CTX_IN_MEM(const void *, DNNL_ARG_SRC);
        const void *wei = CTX_IN_MEM(const void *, DNNL_ARG_WEIGHTS);
        void *bia = CTX_OUT_MEM(void *, DNNL_ARG_BIAS);
        void *dst = CTX_OUT_MEM(void *, DNNL_ARG_DST);

        return execute_forward(ctx, src, wei, dst, bia);
    }

    // Execute forward with arbitrary src, wei, bias and dst
    status_t execute_forward(const exec_ctx_t &ctx, const void *src,
            const void *wei, void *dst, void *bias) const {
        // Lock here is needed because resource_mapper does not support concurrent access.
        std::lock_guard<std::mutex> _lock {this->mtx};

        KuDNN::InnerProductLayerFWD &kdnn_obj =
            (ctx.get_resource_mapper()->get<kdnn_ip_fwd_resource_t>(this))->get_kdnn_obj();

        const float *scales;
        DEFINE_ARG_SCALES_BUFFER(src_scales, DNNL_ARG_SRC);
        DEFINE_ARG_SCALES_BUFFER(wei_scales, DNNL_ARG_WEIGHTS);

        auto &&scratchpad = ctx.get_scratchpad_grantor();

        const int ndims = pd()->ndims();
        const auto dst_d = ctx.memory_mdw(DNNL_ARG_DST, pd()->dst_md());
        scales = precompute_scales(scratchpad, src_scales, wei_scales,
                dst_d.dims()[ndims - 1], pd()->attr());
#if DNNL_CPU_THREADING_RUNTIME == DNNL_RUNTIME_THREADPOOL
        dnnl::threadpool_interop::threadpool_iface *tp = dnnl::impl::threadpool_utils::get_active_threadpool();
        KuDNN::Threading::ThreadpoolIface* threadpool =
            reinterpret_cast<KuDNN::Threading::ThreadpoolIface *>(reinterpret_cast<void *>(tp));
        KuDNN::Threading::DeactivateThreadpool();
        KuDNN::Threading::ActivateThreadpool(threadpool);
#endif

        if (pd()->need_po_wa_) {
            try {
                if (pd()->dst_size_ != 0) {
                    std::unique_ptr<float, KuDNN::Service::Deallocator<float>> tmp_buffer(
                        static_cast<float*>(KuDNN::Service::AlignedAlloc(pd()->dst_size_)));
                    kdnn_obj.Run(src, wei, tmp_buffer.get(), bias, scales[0], 0.0f);
                    pd()->post_ops_.execute(ctx, tmp_buffer.get(), dst);
                } else {
                    kdnn_obj.Run(src, wei, dst, bias, scales[0], 0.0f);
                    pd()->post_ops_.execute(ctx, dst, dst);
                }
            } catch (const std::exception& e) {
                return status::runtime_error;
            }
        } else {
            try {
                KuDNN::PostOpsDataPtrs poPtrs;
                poPtrs.reserve(pd()->attr()->post_ops_.len());
                const std::vector<dnnl_post_ops::entry_t> &post_ops_entry =
                    pd()->attr()->post_ops_.entry_;
                kdnn_utils::reserve_postops_data(poPtrs, post_ops_entry, ctx);
                kdnn_obj.Run(src, wei, dst, bias, poPtrs, scales[0], 0.0f);
            } catch (const std::exception& e) {
                return status::runtime_error;
            }
        }

        return status::success;
    }

    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }
}; // kdnn_inner_product_fwd_t

struct kdnn_ip_bwd_d_resource_t : public resource_t {
    kdnn_ip_bwd_d_resource_t(const std::unique_ptr<KuDNN::InnerProductLayerBWDData> &kdnn_ip_prim) noexcept
        : kdnn_ip_obj_(new KuDNN::InnerProductLayerBWDData{*(kdnn_ip_prim.get())}) {}

    KuDNN::InnerProductLayerBWDData &get_kdnn_obj() const noexcept { return *kdnn_ip_obj_; }

    DNNL_DISALLOW_COPY_AND_ASSIGN(kdnn_ip_bwd_d_resource_t);

private:
    std::unique_ptr<KuDNN::InnerProductLayerBWDData> kdnn_ip_obj_;
}; // kdnn_ip_bwd_d_resource_t

struct kdnn_inner_product_bwd_data_t : public primitive_t {
    struct pd_t : public cpu_inner_product_bwd_data_pd_t {
        using cpu_inner_product_bwd_data_pd_t::
                cpu_inner_product_bwd_data_pd_t;

        pd_t(const pd_t& p) : cpu_inner_product_bwd_data_pd_t(p) {
            if (p.kdnn_ip_prim_) {
                this->kdnn_ip_prim_.reset(new KuDNN::InnerProductLayerBWDData{*(p.kdnn_ip_prim_.get())});
            }
        }

        DECLARE_COMMON_PD_T("kdnn", kdnn_inner_product_bwd_data_t);

        status_t init(engine_t *engine) {

            const bool ok = desc()->prop_kind == prop_kind::backward_data
                    && !has_zero_dim_memory()
                    && attr()->has_default_values()
                    && set_default_params() == status::success;
            if (!ok) return status::unimplemented;

            using namespace format_tag;
            auto dst_tag = memory_desc_matches_one_of_tag(diff_dst_md_, nc, cn);

            if (utils::one_of(format_tag::undef, dst_tag)) {
                // Unsupported memory layout
                return dnnl::impl::status::unimplemented;
            }

            if ((diff_src_md_.format_kind == format_kind_t::dnnl_format_kind_any) ||
                (diff_src_md_.format_kind == format_kind_t::dnnl_format_kind_undef)) {
                CHECK(memory_desc_init_by_tag(diff_src_md_, utils::pick(ndims() - 3, ncw, nchw, ncdhw)));
            }
            if ((weights_md_.format_kind == format_kind_t::dnnl_format_kind_any) ||
                (weights_md_.format_kind == format_kind_t::dnnl_format_kind_undef)) {
                CHECK(memory_desc_init_by_tag(weights_md_, utils::pick(ndims() - 3, ncw, nchw, ncdhw)));
            }

            const memory_desc_wrapper diff_dst_d(diff_dst_md(0));
            const memory_desc_wrapper weights_d(weights_md(0));
            const memory_desc_wrapper diff_src_d(diff_src_md(0));
            auto&& ip_bwdd = kdnn_utils::convert_to_kdnn_ip_bwd_d(diff_dst_d,
                weights_d, diff_src_d);
            if (!ip_bwdd.first) {
                return status::unimplemented;
            } else {
                kdnn_ip_prim_.reset(ip_bwdd.second);
                return status::success;
            }
        }

        std::unique_ptr<KuDNN::InnerProductLayerBWDData> kdnn_ip_prim_;

    }; // pd_t

    kdnn_inner_product_bwd_data_t(const pd_t *kpd) : primitive_t(kpd) {}

    status_t create_resource(
            engine_t *engine, resource_mapper_t &mapper) const override {
        if (mapper.has_resource(this)) return status::success;

        mapper.add(this, std::make_unique<kdnn_ip_bwd_d_resource_t>(pd()->kdnn_ip_prim_));

        return status::success;
    }

    status_t execute(const exec_ctx_t &ctx) const override {
        return execute_forward(ctx);
    }

private:
    mutable std::mutex mtx;

    status_t execute_forward(const exec_ctx_t &ctx) const {
        const void *diff_dst = CTX_IN_MEM(const void *, DNNL_ARG_DIFF_DST);
        const void *weights  = CTX_IN_MEM(const void *, DNNL_ARG_WEIGHTS);
        void *diff_src       = CTX_OUT_MEM(void *, DNNL_ARG_DIFF_SRC);

        return execute_forward(ctx, diff_dst, weights, diff_src);
    }

    status_t execute_forward(const exec_ctx_t &ctx, const void *diff_dst,
            const void *weights, void *diff_src) const {
        // Lock here is needed because resource_mapper does not support concurrent access.
        std::lock_guard<std::mutex> _lock {this->mtx};

        KuDNN::InnerProductLayerBWDData &kdnn_obj =
            (ctx.get_resource_mapper()->get<kdnn_ip_bwd_d_resource_t>(this))->get_kdnn_obj();

        try {
            kdnn_obj.Run(diff_dst, weights, diff_src);
        } catch (const std::exception &e) {
            return status::runtime_error;
        }

        return status::success;
    }

    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }
}; // kdnn_inner_product_bwd_data_t

struct kdnn_ip_bwd_w_resource_t : public resource_t {
    kdnn_ip_bwd_w_resource_t(const std::unique_ptr<KuDNN::InnerProductLayerBWDWeights> &kdnn_ip_prim) noexcept
        : kdnn_ip_obj_(new KuDNN::InnerProductLayerBWDWeights{*(kdnn_ip_prim.get())}) {}

    KuDNN::InnerProductLayerBWDWeights &get_kdnn_obj() const noexcept { return *kdnn_ip_obj_; }

    DNNL_DISALLOW_COPY_AND_ASSIGN(kdnn_ip_bwd_w_resource_t);

private:
    std::unique_ptr<KuDNN::InnerProductLayerBWDWeights> kdnn_ip_obj_;
}; // kdnn_ip_bwd_w_resource_t

struct kdnn_inner_product_bwd_weights_t : public primitive_t {
    struct pd_t : public cpu_inner_product_bwd_weights_pd_t {
        using cpu_inner_product_bwd_weights_pd_t::
                cpu_inner_product_bwd_weights_pd_t;

        pd_t(const pd_t& p) : cpu_inner_product_bwd_weights_pd_t(p) {
            if (p.kdnn_ip_prim_) {
                this->kdnn_ip_prim_.reset(new KuDNN::InnerProductLayerBWDWeights{*(p.kdnn_ip_prim_.get())});
            }
        }

        DECLARE_COMMON_PD_T("kdnn", kdnn_inner_product_bwd_weights_t);

        status_t init(engine_t *engine) {

            const bool ok = desc()->prop_kind == prop_kind::backward_weights
                    && !has_zero_dim_memory()
                    && attr()->has_default_values()
                    && set_default_params() == status::success;
            if (!ok) return status::unimplemented;

            using namespace format_tag;
            auto dst_tag = memory_desc_matches_one_of_tag(diff_dst_md_, nc, cn);

            if (utils::one_of(format_tag::undef, dst_tag)) {
                // Unsupported memory layout
                return dnnl::impl::status::unimplemented;
            }

            if ((src_md_.format_kind == format_kind_t::dnnl_format_kind_any) ||
                (src_md_.format_kind == format_kind_t::dnnl_format_kind_undef)) {
                CHECK(memory_desc_init_by_tag(src_md_, utils::pick(ndims() - 3, ncw, nchw, ncdhw)));
            }
            if ((diff_weights_md_.format_kind == format_kind_t::dnnl_format_kind_any) ||
                (diff_weights_md_.format_kind == format_kind_t::dnnl_format_kind_undef)) {
                CHECK(memory_desc_init_by_tag(diff_weights_md_, utils::pick(ndims() - 3, ncw, nchw, ncdhw)));
            }

            const memory_desc_wrapper diff_dst_d(diff_dst_md(0));
            const memory_desc_wrapper diff_weights_d(diff_weights_md(0));
            const memory_desc_wrapper src_d(src_md(0));
            const memory_desc_wrapper diff_bias_d(diff_weights_md(1));
            auto&& ip_bwdw = kdnn_utils::convert_to_kdnn_ip_bwd_w(diff_dst_d,
                src_d, diff_weights_d, diff_bias_d);
            if (!ip_bwdw.first) {
                return status::unimplemented;
            } else {
                kdnn_ip_prim_.reset(ip_bwdw.second);
                return status::success;
            }
        }

        std::unique_ptr<KuDNN::InnerProductLayerBWDWeights> kdnn_ip_prim_;

    }; // pd_t

    kdnn_inner_product_bwd_weights_t(const pd_t *kpd) : primitive_t(kpd) {}

    status_t create_resource(
            engine_t *engine, resource_mapper_t &mapper) const override {
        if (mapper.has_resource(this)) return status::success;

        mapper.add(this, std::make_unique<kdnn_ip_bwd_w_resource_t>(pd()->kdnn_ip_prim_));

        return status::success;
    }

    status_t execute(const exec_ctx_t &ctx) const override {
        return execute_forward(ctx);
    }

private:
    mutable std::mutex mtx;

    status_t execute_forward(const exec_ctx_t &ctx) const {
        const void *diff_dst = CTX_IN_MEM(const void *, DNNL_ARG_DIFF_DST);
        const void *src      = CTX_IN_MEM(const void *, DNNL_ARG_SRC);
        void *diff_weights   = CTX_OUT_MEM(void *, DNNL_ARG_DIFF_WEIGHTS);
        void *diff_bias      = CTX_OUT_MEM(void *, DNNL_ARG_DIFF_BIAS);

        return execute_forward(ctx, diff_dst, src, diff_weights, diff_bias);
    }

    status_t execute_forward(const exec_ctx_t &ctx, const void *diff_dst,
            const void *src, void *diff_weights, void *diff_bias) const {
        // Lock here is needed because resource_mapper does not support concurrent access.
        std::lock_guard<std::mutex> _lock {this->mtx};

        KuDNN::InnerProductLayerBWDWeights &kdnn_obj =
            (ctx.get_resource_mapper()->get<kdnn_ip_bwd_w_resource_t>(this))->get_kdnn_obj();

        try {
            kdnn_obj.Run(diff_dst, src, diff_weights, diff_bias);
        } catch (const std::exception &e) {
            return status::runtime_error;
        }

        return status::success;
    }

    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }
}; // kdnn_inner_product_bwd_weights_t

} // namespace aarch64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif // CPU_AARCH64_KUDNN_INNER_PRODUCT_HPP
