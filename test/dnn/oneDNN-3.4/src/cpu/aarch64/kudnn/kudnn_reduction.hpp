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

#ifndef CPU_AARCH64_KUDNN_REDUCTION_HPP
#define CPU_AARCH64_KUDNN_REDUCTION_HPP

#include "kudnn.hpp"

#include "cpu/cpu_reduction_pd.hpp"
#include "cpu/aarch64/kudnn/kudnn_utils.hpp"
#include "cpu/aarch64/kudnn/kudnn_post_ops.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace aarch64 {

struct kdnn_reduction_resource_t : public resource_t {
    kdnn_reduction_resource_t(const std::unique_ptr<KuDNN::ReductionLayer> &kdnn_reduction_prim) noexcept
        : kdnn_reduction_obj_(new KuDNN::ReductionLayer{*(kdnn_reduction_prim.get())}) {}

    KuDNN::ReductionLayer &get_kdnn_obj() const noexcept { return *kdnn_reduction_obj_; }

    DNNL_DISALLOW_COPY_AND_ASSIGN(kdnn_reduction_resource_t);

private:
    std::unique_ptr<KuDNN::ReductionLayer> kdnn_reduction_obj_;
}; // kdnn_reduction_resource_t

struct kdnn_reduction_t : public primitive_t {
    struct pd_t : public cpu_reduction_pd_t {
        using cpu_reduction_pd_t::cpu_reduction_pd_t;

        // Need copy constructor because `unique_ptr` is a member of the class
        pd_t(const pd_t& p) : cpu_reduction_pd_t(p) {
            if (p.kdnn_reduction_prim_) {
                this->kdnn_reduction_prim_.reset(new KuDNN::ReductionLayer{*(p.kdnn_reduction_prim_.get())});
                this->post_ops_ = p.post_ops_;
                this->need_po_wa_ = p.need_po_wa_;
                this->dst_size_ = p.dst_size_;
            }
        }

        DECLARE_COMMON_PD_T("kdnn", kdnn_reduction_t);
        
        status_t init(engine_t *engine) {
            const bool ok = set_default_params() == status::success
                    && attr()->has_default_values(primitive_attr_t::skip_mask_t::post_ops)
                    && attr_.set_default_formats(dst_md(0)) == status::success;
            if (!ok) return status::unimplemented;

            const memory_desc_wrapper check_src_d(src_md());
            const memory_desc_wrapper check_dst_d(dst_md());
            using namespace format_tag;

            auto src_tag = check_src_d.format_kind() == format_kind::any
                ? utils::pick(check_src_d.ndims() - 1, a, ab, abc, abcd, abcde)
                : kdnn_utils::get_layout(src_md_);
            auto dst_tag = check_dst_d.format_kind() == format_kind::any
                ? utils::pick(check_dst_d.ndims() - 1, a, ab, abc, abcd, abcde)
                : kdnn_utils::get_layout(dst_md_);

            CHECK(memory_desc_init_by_tag(src_md_, src_tag));
            CHECK(memory_desc_init_by_tag(src_md_, dst_tag));

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
            need_po_wa_ = false;
            dst_size_ = 0;
            auto&& reduction = kdnn_utils::convert_to_kdnn_reduction(src_d, dst_d, desc_.alg_kind, desc_.p, desc_.eps);
            if (!reduction.first) {
                return status::unimplemented;
            } else {
                kdnn_reduction_prim_.reset(reduction.second);
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

        std::unique_ptr<KuDNN::ReductionLayer> kdnn_reduction_prim_;

    }; // pd_t

    kdnn_reduction_t(const pd_t *kpd) : primitive_t(kpd) {}

    status_t create_resource(
            engine_t *engine, resource_mapper_t &mapper) const override {
        if (mapper.has_resource(this)) return status::success;

        mapper.add(this, std::make_unique<kdnn_reduction_resource_t>(pd()->kdnn_reduction_prim_));
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
        void *dst = CTX_OUT_MEM(void *, DNNL_ARG_DST);

        return execute_forward(ctx, src, dst);
    }

    // Execute forward with arbitrary src and dst
    status_t execute_forward(const exec_ctx_t &ctx, const void *src, void *dst) const {
        // Lock here is needed because resource_mapper does not support concurrent access.
        std::lock_guard<std::mutex> _lock {this->mtx};
 
        KuDNN::ReductionLayer &kdnn_obj =
            (ctx.get_resource_mapper()->get<kdnn_reduction_resource_t>(this))->get_kdnn_obj();

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
}; // kdnn_reduction_t

} // namespace aarch64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif // CPU_AARCH64_KUDNN_REDUCTION_HPP
