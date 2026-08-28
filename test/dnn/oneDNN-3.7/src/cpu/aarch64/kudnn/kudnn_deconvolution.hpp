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

#ifndef CPU_AARCH64_KUDNN_DECONVOLUTION_HPP
#define CPU_AARCH64_KUDNN_DECONVOLUTION_HPP

#include <memory>

#include "kudnn.hpp"

#include "cpu/cpu_deconvolution_pd.hpp"
#include "cpu/aarch64/kudnn/kudnn_utils.hpp"
#include "cpu/aarch64/kudnn/kudnn_post_ops.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace aarch64 {

struct kdnn_deconvolution_fwd_resource_t : public resource_t {
    kdnn_deconvolution_fwd_resource_t(const std::unique_ptr<KuDNN::DeconvolutionLayerFWD> &kdnn_deconvolution_prim) noexcept
        : kdnn_deconvolution_fwd_obj_(new KuDNN::DeconvolutionLayerFWD{*(kdnn_deconvolution_prim.get())}) {}

    KuDNN::DeconvolutionLayerFWD &get_kdnn_obj() const noexcept { return *kdnn_deconvolution_fwd_obj_; }

    DNNL_DISALLOW_COPY_AND_ASSIGN(kdnn_deconvolution_fwd_resource_t);

private:
    std::unique_ptr<KuDNN::DeconvolutionLayerFWD> kdnn_deconvolution_fwd_obj_;
}; // kdnn_deconvolution_fwd_resource_t

struct kdnn_deconvolution_fwd_t : public primitive_t {
    struct pd_t : public cpu_deconvolution_fwd_pd_t {
        using cpu_deconvolution_fwd_pd_t::cpu_deconvolution_fwd_pd_t;

        pd_t(const pd_t& p) : cpu_deconvolution_fwd_pd_t(p) {
            if (p.kdnn_deconvolution_prim_) {
                this->kdnn_deconvolution_prim_.reset(new KuDNN::DeconvolutionLayerFWD{*(p.kdnn_deconvolution_prim_.get())});
                this->post_ops_ = p.post_ops_;
                this->need_po_wa_ = p.need_po_wa_;
                this->dst_size_ = p.dst_size_;
            }
        }

        DECLARE_COMMON_PD_T("kdnn", kdnn_deconvolution_fwd_t);

        status_t init(engine_t *engine) {
            const bool ok = is_fwd()
                && attr()->has_default_values(primitive_attr_t::skip_mask_t::post_ops)
                && attr_.set_default_formats(dst_md(0)) == status::success;
            if (!ok) return status::unimplemented;

            // Set memory formats
            const memory_desc_wrapper check_src_d(src_md());
            const memory_desc_wrapper check_weights_d(weights_md(0));
            const memory_desc_wrapper check_dst_d(dst_md());
            const memory_desc_wrapper check_bias_d(weights_md(1));
            using namespace format_tag;

            auto src_tag = check_src_d.format_kind() == format_kind::any
                ? utils::pick(ndims() - 3, ncw, nchw, ncdhw)
                : kdnn_utils::get_layout(src_md_);
            auto wei_tag = with_groups()
                ? (check_weights_d.format_kind() == format_kind::any
                    ? utils::pick(ndims() - 3, goiw, goihw, goidhw)
                    : kdnn_utils::get_layout(weights_md_))
                : (check_weights_d.format_kind() == format_kind::any
                    ? utils::pick(ndims() - 3, oiw, oihw, oidhw)
                    : kdnn_utils::get_layout(weights_md_));
            auto dst_tag = check_dst_d.format_kind() == format_kind::any
                ? utils::pick(ndims() - 3, ncw, nchw, ncdhw)
                : kdnn_utils::get_layout(dst_md_);

            CHECK(memory_desc_init_by_tag(src_md_, src_tag));
            CHECK(memory_desc_init_by_tag(dst_md_, dst_tag));
            CHECK(memory_desc_init_by_tag(weights_md_, wei_tag));
            if (check_bias_d != &glob_zero_md) {
                CHECK(memory_desc_init_by_tag(bias_md_, x));
            }

            const memory_desc_wrapper src_d(&src_md_);
            const memory_desc_wrapper weights_d(&weights_md_);
            data_type_t dst_dt = dst_md_.data_type;
            if (attr_.post_ops_.len() > 1) {
                return status::unimplemented; // Currently, only one post-ops is supported.
            }
            bool has_post_ops = (attr_.post_ops_.len() >= 1);
            if (has_post_ops) {
                dst_md_.data_type = data_type_t::dnnl_f32; // Temporarily change the output data type to f32
            }
            const memory_desc_wrapper dst_d(&dst_md_);
            const memory_desc_wrapper bias_d(&bias_md_);
            need_po_wa_ = false;
            dst_size_ = 0;
            auto&& deconv_fwd = kdnn_utils::convert_to_kdnn_deconv_fwd(src_d,
                weights_d, dst_d, bias_d, *desc(), desc_.alg_kind);
            if (!deconv_fwd.first) {
                return status::unimplemented;
            } else {
                kdnn_deconvolution_prim_.reset(deconv_fwd.second);
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

        bool need_po_wa_;
        size_t dst_size_;
        kdnn_post_ops_t post_ops_;

        std::unique_ptr<KuDNN::DeconvolutionLayerFWD> kdnn_deconvolution_prim_;

    }; // pd_t

    kdnn_deconvolution_fwd_t(const pd_t *kpd) : primitive_t(kpd) {}

    status_t create_resource(
        engine_t *engine, resource_mapper_t &mapper) const override {
        if (mapper.has_resource(this)) return status::success;

        mapper.add(this, std::make_unique<kdnn_deconvolution_fwd_resource_t>(pd()->kdnn_deconvolution_prim_));
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
        const void *bia = CTX_IN_MEM(const void *, DNNL_ARG_BIAS);
        void *dst = CTX_OUT_MEM(void *, DNNL_ARG_DST);

        return execute_forward(ctx, src, wei, dst, bia);
    }

    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }

    status_t execute_forward(const exec_ctx_t &ctx, const void *src,
            const void *wei, void *dst, const void *bias) const {
        // Lock here is needed because resource_mapper does not support concurrent access.
        std::lock_guard<std::mutex> _lock {this->mtx};

        KuDNN::DeconvolutionLayerFWD &kdnn_obj =
            (ctx.get_resource_mapper()->get<kdnn_deconvolution_fwd_resource_t>(this))->get_kdnn_obj();

        if (pd()->need_po_wa_) {
            try {
                if (pd()->dst_size_ != 0) {
                    std::unique_ptr<float, KuDNN::Service::Deallocator<float>> tmp_buffer(
                        static_cast<float*>(KuDNN::Service::AlignedAlloc(pd()->dst_size_)));
                    kdnn_obj.Run(src, wei, tmp_buffer.get(), bias);
                    pd()->post_ops_.execute(ctx, tmp_buffer.get(), dst);
                } else {
                    kdnn_obj.Run(src, wei, dst, bias);
                    pd()->post_ops_.execute(ctx, dst, dst);
                }
            } catch (const std::exception& e) {
                return status::runtime_error;
            }
        } else {
            try {
                kdnn_obj.Run(src, wei, dst, bias);
            } catch (const std::exception& e) {
                return status::runtime_error;
            }
        }
 
        return status::success;
    }

}; // kdnn_deconvolution_fwd_t

struct kdnn_deconvolution_bwd_data_resource_t : public resource_t {
    kdnn_deconvolution_bwd_data_resource_t(const std::unique_ptr<KuDNN::DeconvolutionLayerBWDData> &kdnn_deconvolution_prim) noexcept
        : kdnn_deconvolution_bwd_data_obj_(new KuDNN::DeconvolutionLayerBWDData{*(kdnn_deconvolution_prim.get())}) {}

    KuDNN::DeconvolutionLayerBWDData &get_kdnn_obj() const noexcept { return *kdnn_deconvolution_bwd_data_obj_; }

    DNNL_DISALLOW_COPY_AND_ASSIGN(kdnn_deconvolution_bwd_data_resource_t);

private:
    std::unique_ptr<KuDNN::DeconvolutionLayerBWDData> kdnn_deconvolution_bwd_data_obj_;
}; // kdnn_deconvolution_bwd_data_resource_t

struct kdnn_deconvolution_bwd_data_t : public primitive_t {
    struct pd_t : public cpu_deconvolution_bwd_data_pd_t {
        using cpu_deconvolution_bwd_data_pd_t::cpu_deconvolution_bwd_data_pd_t;

        pd_t(const pd_t& p) : cpu_deconvolution_bwd_data_pd_t(p) {
            if (p.kdnn_deconvolution_prim_) {
                this->kdnn_deconvolution_prim_.reset(new KuDNN::DeconvolutionLayerBWDData{*(p.kdnn_deconvolution_prim_.get())});
            }
        }

        DECLARE_COMMON_PD_T("kdnn", kdnn_deconvolution_bwd_data_t);

        status_t init(engine_t *engine) {
            const bool ok = desc()->prop_kind == prop_kind::backward_data
                && attr()->has_default_values();
            if (!ok) return status::unimplemented;

            // Set memory formats
            const memory_desc_wrapper check_diff_dst_d(diff_dst_md());
            const memory_desc_wrapper check_weights_d(weights_md());
            const memory_desc_wrapper check_diff_src_d(diff_src_md());
            using namespace format_tag;

            auto diff_dst_tag = check_diff_dst_d.format_kind() == format_kind::any
                ? utils::pick(ndims() - 3, ncw, nchw, ncdhw)
                : kdnn_utils::get_layout(diff_dst_md_);
            auto wei_tag = with_groups()
                ? (check_weights_d.format_kind() == format_kind::any
                    ? utils::pick(ndims() - 3, goiw, goihw, goidhw)
                    : kdnn_utils::get_layout(weights_md_))
                : (check_weights_d.format_kind() == format_kind::any
                    ? utils::pick(ndims() - 3, oiw, oihw, oidhw)
                    : kdnn_utils::get_layout(weights_md_));
            auto diff_src_tag = check_diff_src_d.format_kind() == format_kind::any
                ? utils::pick(ndims() - 3, ncw, nchw, ncdhw)
                : kdnn_utils::get_layout(diff_src_md_);

            CHECK(memory_desc_init_by_tag(diff_dst_md_, diff_dst_tag));
            CHECK(memory_desc_init_by_tag(weights_md_, wei_tag));
            CHECK(memory_desc_init_by_tag(diff_src_md_, diff_src_tag));

            const memory_desc_wrapper diff_dst_d(diff_dst_md());
            const memory_desc_wrapper weights_d(weights_md());
            const memory_desc_wrapper diff_src_d(diff_src_md());
            auto&& deconv_bwdd = kdnn_utils::convert_to_kdnn_deconv_bwd_data(diff_dst_d, weights_d, diff_src_d,
                    *desc(), desc_.alg_kind);
            if (!deconv_bwdd.first) {
                return status::unimplemented;
            } else {
                kdnn_deconvolution_prim_.reset(deconv_bwdd.second);
                return status::success;
            }
        }

        std::unique_ptr<KuDNN::DeconvolutionLayerBWDData> kdnn_deconvolution_prim_;

    }; // pd_t

    kdnn_deconvolution_bwd_data_t(const pd_t *kpd) : primitive_t(kpd) {}

    status_t create_resource(
        engine_t *engine, resource_mapper_t &mapper) const override {
        if (mapper.has_resource(this)) return status::success;

        mapper.add(this, std::make_unique<kdnn_deconvolution_bwd_data_resource_t>(pd()->kdnn_deconvolution_prim_));

        return status::success;
    }

    status_t execute(const exec_ctx_t &ctx) const override {
        return execute_backward_data(ctx);
    }

private:
    mutable std::mutex mtx;

    status_t execute_backward_data(const exec_ctx_t &ctx) const {
        const void *diff_dst = CTX_IN_MEM(const void *, DNNL_ARG_DIFF_DST);
        const void *wei = CTX_IN_MEM(const void *, DNNL_ARG_WEIGHTS);
        void *diff_src = CTX_OUT_MEM(void *, DNNL_ARG_DIFF_SRC);

        return execute_backward_data(ctx, diff_dst, wei, diff_src);
    }

    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }

    status_t execute_backward_data(const exec_ctx_t &ctx, const void *diff_dst,
            const void *wei, void *diff_src) const {
        // Lock here is needed because resource_mapper does not support concurrent access.
        std::lock_guard<std::mutex> _lock {this->mtx};

        KuDNN::DeconvolutionLayerBWDData &kdnn_obj =
            (ctx.get_resource_mapper()->get<kdnn_deconvolution_bwd_data_resource_t>(this))->get_kdnn_obj();

        try {
            kdnn_obj.Run(diff_dst, wei, diff_src);
        } catch (const std::exception &e) {
            return status::runtime_error;
        }

        return status::success;
    }

}; // kdnn_deconvolution_bwd_data_t

struct kdnn_deconvolution_bwd_weights_resource_t : public resource_t {
    kdnn_deconvolution_bwd_weights_resource_t(const std::unique_ptr<KuDNN::DeconvolutionLayerBWDWeights> &kdnn_deconvolution_prim) noexcept
        : kdnn_deconvolution_bwd_weights_obj_(new KuDNN::DeconvolutionLayerBWDWeights{*(kdnn_deconvolution_prim.get())}) {}

    KuDNN::DeconvolutionLayerBWDWeights &get_kdnn_obj() const noexcept { return *kdnn_deconvolution_bwd_weights_obj_; }

    DNNL_DISALLOW_COPY_AND_ASSIGN(kdnn_deconvolution_bwd_weights_resource_t);

private:
    std::unique_ptr<KuDNN::DeconvolutionLayerBWDWeights> kdnn_deconvolution_bwd_weights_obj_;
}; // kdnn_deconvolution_bwd_weights_resource_t

struct kdnn_deconvolution_bwd_weights_t : public primitive_t {
    struct pd_t : public cpu_deconvolution_bwd_weights_pd_t {
        using cpu_deconvolution_bwd_weights_pd_t::cpu_deconvolution_bwd_weights_pd_t;

        pd_t(const pd_t& p) : cpu_deconvolution_bwd_weights_pd_t(p) {
            if (p.kdnn_deconvolution_prim_) {
                this->kdnn_deconvolution_prim_.reset(new KuDNN::DeconvolutionLayerBWDWeights{*(p.kdnn_deconvolution_prim_.get())});
            }
        }

        DECLARE_COMMON_PD_T("kdnn", kdnn_deconvolution_bwd_weights_t);

        status_t init(engine_t *engine) {
            const bool ok = desc()->prop_kind == prop_kind::backward_weights
                && attr()->has_default_values();
            if (!ok) return status::unimplemented;

            // Set memory formats
            const memory_desc_wrapper check_diff_dst_d(diff_dst_md());
            const memory_desc_wrapper check_src_d(src_md());
            const memory_desc_wrapper check_diff_wei_d(diff_weights_md(0));
            const memory_desc_wrapper check_diff_bias_d(diff_weights_md(1));
            using namespace format_tag;

            auto diff_dst_tag = check_diff_dst_d.format_kind() == format_kind::any
                ? utils::pick(ndims() - 3, ncw, nchw, ncdhw)
                : kdnn_utils::get_layout(diff_dst_md_);
            auto src_tag = check_src_d.format_kind() == format_kind::any
                ? utils::pick(ndims() - 3, ncw, nchw, ncdhw)
                : kdnn_utils::get_layout(src_md_);
            auto diff_wei_tag = with_groups()
                ? (check_diff_wei_d.format_kind() == format_kind::any
                    ? utils::pick(ndims() - 3, goiw, goihw, goidhw)
                    : kdnn_utils::get_layout(diff_weights_md_))
                : (check_diff_wei_d.format_kind() == format_kind::any
                    ? utils::pick(ndims() - 3, oiw, oihw, oidhw)
                    : kdnn_utils::get_layout(diff_weights_md_));

            CHECK(memory_desc_init_by_tag(diff_dst_md_, diff_dst_tag));
            CHECK(memory_desc_init_by_tag(src_md_, src_tag));
            CHECK(memory_desc_init_by_tag(diff_weights_md_, diff_wei_tag));
            if (check_diff_bias_d != &glob_zero_md) {
                CHECK(memory_desc_init_by_tag(diff_bias_md_, x));
            }

            const memory_desc_wrapper diff_dst_d(diff_dst_md());
            const memory_desc_wrapper src_d(src_md());
            const memory_desc_wrapper diff_wei_d(diff_weights_md(0));
            const memory_desc_wrapper diff_bias_d(diff_weights_md(1));
            auto&& deconv_bwdw = kdnn_utils::convert_to_kdnn_deconv_bwd_weights(diff_dst_d,
                    src_d, diff_wei_d, diff_bias_d, *desc(), desc_.alg_kind);
            if (!deconv_bwdw.first) {
                return status::unimplemented;
            } else {
                kdnn_deconvolution_prim_.reset(deconv_bwdw.second);
                return status::success;
            }
        }

        std::unique_ptr<KuDNN::DeconvolutionLayerBWDWeights> kdnn_deconvolution_prim_;

    }; // pd_t

    kdnn_deconvolution_bwd_weights_t(const pd_t *kpd) : primitive_t(kpd) {}

    status_t create_resource(
        engine_t *engine, resource_mapper_t &mapper) const override {
        if (mapper.has_resource(this)) return status::success;

        mapper.add(this, std::make_unique<kdnn_deconvolution_bwd_weights_resource_t>(pd()->kdnn_deconvolution_prim_));

        return status::success;
    }

    status_t execute(const exec_ctx_t &ctx) const override {
        return execute_backward_weights(ctx);
    }

private:
    mutable std::mutex mtx;

    status_t execute_backward_weights(const exec_ctx_t &ctx) const {
        const void *diff_dst = CTX_IN_MEM(const void *, DNNL_ARG_DIFF_DST);
        const void *src = CTX_IN_MEM(const void *, DNNL_ARG_SRC);
        void *diff_wei = CTX_OUT_MEM(void *, DNNL_ARG_DIFF_WEIGHTS);
        void *diff_bia = CTX_OUT_MEM(void *, DNNL_ARG_DIFF_BIAS);

        return execute_backward_weights(ctx, diff_dst, src, diff_wei, diff_bia);
    }

    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }

    status_t execute_backward_weights(const exec_ctx_t &ctx, const void *diff_dst,
            const void *src, void *diff_wei, void *diff_bia) const {
        // Lock here is needed because resource_mapper does not support concurrent access.
        std::lock_guard<std::mutex> _lock {this->mtx};

        KuDNN::DeconvolutionLayerBWDWeights &kdnn_obj =
            (ctx.get_resource_mapper()->get<kdnn_deconvolution_bwd_weights_resource_t>(this))->get_kdnn_obj();

        try {
            kdnn_obj.Run(diff_dst, src, diff_wei, diff_bia);
        } catch (const std::exception &e) {
            return status::runtime_error;
        }

        return status::success;
    }

}; // kdnn_deconvolution_bwd_weights_t

} // namespace aarch64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif // CPU_AARCH64_KUDNN_DECONVOLUTION_HPP
