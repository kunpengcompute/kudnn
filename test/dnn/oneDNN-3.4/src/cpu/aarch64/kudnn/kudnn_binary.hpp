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

#ifndef CPU_AARCH64_KUDNN_BINARY_HPP
#define CPU_AARCH64_KUDNN_BINARY_HPP

#include <memory>

#include "kudnn.hpp"

#include "cpu/cpu_binary_pd.hpp"
#include "cpu/aarch64/kudnn/kudnn_utils.hpp"
#include "cpu/aarch64/kudnn/kudnn_post_ops.hpp"
#include "cpu/cpu_primitive.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace aarch64 {

struct kdnn_binary_resource_t : public resource_t {
    kdnn_binary_resource_t(const std::unique_ptr<KuDNN::BinaryLayer> &kdnn_binary_prim) noexcept
        : kdnn_binary_obj_(new KuDNN::BinaryLayer{*(kdnn_binary_prim.get())}) {}

    KuDNN::BinaryLayer &get_kdnn_obj() const noexcept { return *kdnn_binary_obj_; }

    DNNL_DISALLOW_COPY_AND_ASSIGN(kdnn_binary_resource_t);

private:
    std::unique_ptr<KuDNN::BinaryLayer> kdnn_binary_obj_;
}; // kdnn_binary_resource_t

struct kdnn_binary_t : public primitive_t {
    struct pd_t : public cpu_binary_pd_t {
        using cpu_binary_pd_t::cpu_binary_pd_t;

        pd_t(const pd_t& p) : cpu_binary_pd_t(p) {
            if (p.kdnn_binary_prim_) {
                this->kdnn_binary_prim_.reset(new KuDNN::BinaryLayer{*(p.kdnn_binary_prim_.get())});
                this->post_ops_ = p.post_ops_;
                this->need_po_wa_ = p.need_po_wa_;
                this->dst_size_ = p.dst_size_;
            }
        }

        DECLARE_COMMON_PD_T("kdnn", kdnn_binary_t);

        status_t init(engine_t *engine) {
            auto&& first_non_any_md = kdnn_utils::get_first_non_any_format_kind(src0_md_, src1_md_, dst_md_);
            auto&& layout = kdnn_utils::get_layout(first_non_any_md);
            if (src0_md_.format_kind == format_kind::any) {
                CHECK(memory_desc_init_by_tag(src0_md_, layout));
            }
            if (src1_md_.format_kind == format_kind::any) {
                CHECK(memory_desc_init_by_tag(src1_md_, layout));
            }
            if (dst_md_.format_kind == format_kind::any) {
                CHECK(memory_desc_init_by_tag(dst_md_, layout));
            }

            if (!attr()->has_default_values(primitive_attr_t::skip_mask_t::scales_runtime |
                    primitive_attr_t::skip_mask_t::post_ops)
                || (!(attr()->scales_.get(DNNL_ARG_SRC_0).mask_ == 0))
                || (!(attr()->scales_.get(DNNL_ARG_SRC_1).mask_ == 0))) {
                return status::unimplemented;
            }

            const memory_desc_wrapper src0_d(src_md(0));
            const memory_desc_wrapper src1_d(src_md(1));
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
            auto&& binary = kdnn_utils::convert_to_kdnn_binary(src0_d, src1_d, dst_d, desc_.alg_kind);
            if (!binary.first) {
                return status::unimplemented;
            } else {
                kdnn_binary_prim_.reset(binary.second);
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

        std::unique_ptr<KuDNN::BinaryLayer> kdnn_binary_prim_;

        friend struct kdnn_post_ops_t;

        bool need_po_wa_;
        size_t dst_size_;
        kdnn_post_ops_t post_ops_;
    }; // pd_t

    kdnn_binary_t(const pd_t *kpd) : primitive_t(kpd) {}

    status_t create_resource(
            engine_t *engine, resource_mapper_t &mapper) const override {
        if (mapper.has_resource(this)) return status::success;

        mapper.add(this, std::make_unique<kdnn_binary_resource_t>(pd()->kdnn_binary_prim_));
        CHECK(pd()->post_ops_.create_resource(engine, mapper));

        return status::success;
    }

    status_t execute(const exec_ctx_t &ctx) const override {
        return execute_forward(ctx);
    }

private:
    // execute_forward has to be const thus mutability of mtx
    mutable std::mutex mtx;

    status_t execute_forward(const exec_ctx_t &ctx) const {
        auto src0 = CTX_IN_MEM(const void *, DNNL_ARG_SRC_0);
        auto src1 = CTX_IN_MEM(const void *, DNNL_ARG_SRC_1);
        void *dst = CTX_OUT_MEM(void *, DNNL_ARG_DST);

        return execute_forward(ctx, src0, src1, dst);
    }

    // Execute forward with arbitrary src and dst
    status_t execute_forward(const exec_ctx_t &ctx, const void *src0,
            const void *src1, void *dst, float sum_scale = 1.f) const {
        // Lock here is needed because resource_mapper does not support concurrent access.
        std::lock_guard<std::mutex> _lock {this->mtx};
 
        KuDNN::BinaryLayer &kdnn_obj = (
            ctx.get_resource_mapper()->get<kdnn_binary_resource_t>(this))->get_kdnn_obj();

        const float *scales[2];
        ASSIGN_ARG_SCALE_VALUE(scales[0], DNNL_ARG_SRC_0);
        ASSIGN_ARG_SCALE_VALUE(scales[1], DNNL_ARG_SRC_1);
        if (sum_scale != 1.f) {
            const_cast<float *>(scales[1])[0] = sum_scale;
        }

        if (pd()->need_po_wa_) {
            try {
                if (pd()->dst_size_ != 0) {
                    std::unique_ptr<float, KuDNN::Service::Deallocator<float>> tmp_buffer(
                    static_cast<float*>(KuDNN::Service::AlignedAlloc(pd()->dst_size_)));
                    kdnn_obj.Run(src0, src1, tmp_buffer.get(), scales[0][0], scales[1][0]);
                    pd()->post_ops_.execute(ctx, tmp_buffer.get(), dst);
                } else {
                    kdnn_obj.Run(src0, src1, dst, scales[0][0], scales[1][0]);
                    pd()->post_ops_.execute(ctx, dst, dst);
                }
            } catch (const std::exception& e) {
                return status::runtime_error;
            }
        } else {
            try {
                kdnn_obj.Run(src0, src1, dst, scales[0][0], scales[1][0]);
            } catch (const std::exception& e) {
                return status::runtime_error;
            }
        }

        return status::success;
    }

    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }

    friend struct kdnn_post_ops_t;
}; // kdnn_binary_t

} // namespace aarch64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif // CPU_AARCH64_KUDNN_BINARY_HPP
