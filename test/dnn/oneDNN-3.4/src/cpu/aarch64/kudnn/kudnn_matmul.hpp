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

#ifndef CPU_AARCH64_KUDNN_MATMUL_HPP
#define CPU_AARCH64_KUDNN_MATMUL_HPP

#include "kudnn.hpp"

#include "common/utils.hpp"
#include "cpu/aarch64/kudnn/kudnn_utils.hpp"
#include "cpu/aarch64/kudnn/kudnn_post_ops.hpp"
#include "cpu/cpu_primitive.hpp"
#include "cpu/scale_utils.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace aarch64 {
namespace matmul {

using namespace dnnl::impl::utils;
using namespace dnnl::impl::format_tag;

struct kdnn_matmul_resource_t : public resource_t {
    kdnn_matmul_resource_t(const std::unique_ptr<KuDNN::Gemm> &kdnn_matmul_prim) noexcept
        : kdnn_matmul_obj_(new KuDNN::Gemm{*(kdnn_matmul_prim.get())}) {}

    KuDNN::Gemm &get_kdnn_obj() const noexcept { return *kdnn_matmul_obj_; }

    DNNL_DISALLOW_COPY_AND_ASSIGN(kdnn_matmul_resource_t);

private:
    std::unique_ptr<KuDNN::Gemm> kdnn_matmul_obj_;
}; // kdnn_matmul_resource_t

struct kdnn_matmul_t : public primitive_t {
    struct pd_t : public dnnl::impl::cpu::matmul::cpu_matmul_pd_t {
        using cpu_matmul_pd_t::cpu_matmul_pd_t;

        // Need copy constructor because `unique_ptr` is a member of the class
        pd_t(const pd_t& p) : cpu_matmul_pd_t(p) {
            if (p.kdnn_matmul_prim_) {
                this->kdnn_matmul_prim_.reset(new KuDNN::Gemm{*(p.kdnn_matmul_prim_.get())});
                this->post_ops_ = p.post_ops_;
                this->need_po_wa_ = p.need_po_wa_;
                this->dst_size_ = p.dst_size_;
            }
        }

        DECLARE_COMMON_PD_T("kdnn", kdnn_matmul_t);

        status_t init(engine_t *engine) {

            bool ok = !has_zero_dim_memory() &&
                attr()->has_default_values(dnnl_primitive_attr::skip_mask_t::post_ops |
                    dnnl_primitive_attr::skip_mask_t::scales_runtime) &&
                attr_scales_ok() &&
                (attr()->output_scales_.mask_ == 0) &&
                (attr()->scales_.get(DNNL_ARG_WEIGHTS).mask_ == 0) &&
                !has_runtime_dims_or_strides();
            if (!ok) return status::unimplemented;

            if (weights_md_.format_kind == dnnl_format_kind_t::dnnl_format_kind_any) {
                // Default to 'N T' form to reduce reorder calls in the model.
                const auto wei_tag = pick(weights_md_.ndims - 2, ba, acb, abdc, abced);
                memory_desc_init_by_tag(weights_md_, wei_tag);
            }
            VDISPATCH_MATMUL(set_default_formats(), VERBOSE_UNSUPPORTED_TAG);
            const memory_desc_wrapper src_d(&src_md_);
            const memory_desc_wrapper wei_d(&weights_md_);
            if (attr_.post_ops_.len() > 1) {
                return status::unimplemented; // Currently, only one post-ops is supported.
            }
            const memory_desc_wrapper dst_d(dst_md());
            const memory_desc_wrapper bias_d(&bias_md_);
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
            auto&& matmul = kdnn_utils::convert_to_kdnn_gemm(src_d, wei_d, dst_d, bias_d, attr_);
            if (!matmul.first) {
                // This part planned to be removed when post-ops are supported in KuDNN
                data_type_t dst_dt = dst_md_.data_type;
                bool has_post_ops = (attr_.post_ops_.len() >= 1);
                if (has_post_ops) {
                    need_po_wa_ = true;
                    dst_md_.data_type = data_type_t::dnnl_f32; // Temporarily change the output data type to f32
                }
                auto&& matmul_no_attr = kdnn_utils::convert_to_kdnn_gemm(src_d, wei_d, dst_d, bias_d, {});
                if (!matmul_no_attr.first) {
                    return status::unimplemented;
                } else {
                    kdnn_matmul_prim_.reset(matmul_no_attr.second);
                    if (has_post_ops) {
                        dst_md_.data_type = dst_dt;
                        if (dst_dt != data_type_t::dnnl_f32) {
                            dst_size_ = dst_d.nelems() * types::data_type_size(data_type_t::dnnl_f32);
                        }
                    }
                    CHECK(post_ops_.init(engine, attr_.post_ops_, dst_md_));
                }
            } else {
                kdnn_matmul_prim_.reset(matmul.second);
            }
            auto scratchpad = scratchpad_registry().registrar();
            book_precomputed_scales(scratchpad, attr()->scales_, N());
            return status::success;
        }

        bool attr_scales_ok(
                const std::vector<int> &supported_args = {DNNL_ARG_SRC,
                        DNNL_ARG_WEIGHTS}) const override {
            if (attr()->scales_.has_default_values()) return true;

            bool ok = attr()->scales_.has_default_values(supported_args);
            for (int arg : supported_args) {
                const auto &sc = attr()->scales_.get(arg);
                const auto &mask = sc.mask_;
                if (!sc.has_default_values()) {
                    if (arg == DNNL_ARG_WEIGHTS) {
                        ok = ok
                                && utils::one_of(mask, 0, wei_qmask_N(),
                                        wei_qmask_N() + wei_qmask_K());
                        ok = ok && utils::one_of(sc.ndims_, 0, 2)
                                && IMPLICATION(sc.ndims_ == 2,
                                        sc.group_dims_[1] == 1
                                                && K() % sc.group_dims_[0]
                                                        == 0);
                    } else
                        ok = ok && (mask == 0);
                }
            }
            return ok;
        }

        bool need_po_wa_;
        size_t dst_size_;
        kdnn_post_ops_t post_ops_;

        std::unique_ptr<KuDNN::Gemm> kdnn_matmul_prim_;

    }; // pd_t

    kdnn_matmul_t(const pd_t *kpd) : primitive_t(kpd) {}

    status_t create_resource(
            engine_t *engine, resource_mapper_t &mapper) const override {
        if (mapper.has_resource(this)) return status::success;

        mapper.add(this, std::make_unique<kdnn_matmul_resource_t>(pd()->kdnn_matmul_prim_));
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
        const void *bias = CTX_IN_MEM(const void *, DNNL_ARG_BIAS);
        void *dst = CTX_OUT_MEM(void *, DNNL_ARG_DST);

        return execute_forward(ctx, src, wei, dst, bias);
    }

    // Execute forward with arbitrary src, wei, bias and dst
    status_t execute_forward(const exec_ctx_t &ctx, const void *src,
            const void *wei, void *dst, const void *bias) const {
        // Lock here is needed because resource_mapper does not support concurrent access.
        std::lock_guard<std::mutex> _lock {this->mtx};

        KuDNN::Gemm &kdnn_obj =
            (ctx.get_resource_mapper()->get<kdnn_matmul_resource_t>(this))->get_kdnn_obj();

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
            } catch (const std::exception &e) {
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
            } catch (const std::exception &e) {
                return status::runtime_error;
            }
        }

        return status::success;
    }

    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }
}; // kdnn_matmul_t

} // namespace matmul
} // namespace aarch64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif // CPU_AARCH64_KUDNN_MATMUL_HPP
