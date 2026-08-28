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

#ifndef CPU_AARCH64_KUDNN_RNN_HPP
#define CPU_AARCH64_KUDNN_RNN_HPP
#include <map>

#include "kudnn.hpp"

#include "cpu/rnn/cpu_rnn_pd.hpp"
#include "cpu/aarch64/kudnn/kudnn_utils.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace aarch64 {

using namespace dnnl::impl::utils;
using namespace dnnl::impl::cpu::rnn_utils;
using namespace dnnl::impl::data_type;

template <impl::data_type_t src_type, impl::data_type_t weights_type, impl::data_type_t acc_type,
    impl::data_type_t scratch_type>
struct kdnn_rnn_common_t {
    /* These types are defined for each element in the cell execution */
    typedef typename prec_traits<src_type>::type src_layer_t;
    typedef typename prec_traits<src_type>::type src_iter_t;
    typedef typename prec_traits<src_type>::type dst_layer_t;
    typedef typename prec_traits<src_type>::type dst_iter_t;
    typedef typename prec_traits<weights_type>::type weights_t;
    typedef typename prec_traits<src_type>::type gemm_data_t;
    typedef typename prec_traits<acc_type>::type gemm_acc_t;
    typedef typename prec_traits<scratch_type>::type scratch_t;
    typedef typename prec_traits<src_type>::type ht_t;
    typedef typename prec_traits<src_type>::type gates_t;
};

using init_conf_func_type = bool(*)(rnn_conf_t &rnn, const rnn_desc_t &rd,
    const primitive_attr_t &attr, const memory_desc_wrapper &src_layer_d,
    const memory_desc_wrapper &src_iter_d,
    const memory_desc_wrapper &src_iter_c_d,
    const memory_desc_wrapper &weights_layer_d,
    const memory_desc_wrapper &weights_iter_d,
    const memory_desc_wrapper &weights_projection_d,
    const memory_desc_wrapper &dst_layer_d,
    const memory_desc_wrapper &dst_iter_d,
    const memory_desc_wrapper &dst_iter_c_d,
    const memory_desc_wrapper &bias_d);
using set_conf_func_type = void(*)(rnn_conf_t &rnn, const rnn_desc_t &rd,
    const memory_desc_wrapper &weights_layer_d,
    const memory_desc_wrapper &weights_iter_d,
    const memory_desc_wrapper &weights_projection_d,
    const memory_desc_wrapper &diff_weights_layer_d,
    const memory_desc_wrapper &diff_weights_iter_d,
    const memory_desc_wrapper &diff_weights_projection_d);
using set_workspace_sizes_func_type = void(*)(rnn_conf_t &rnn, const rnn_desc_t &rd);

using kdnn_rnn_common_f32_t = kdnn_rnn_common_t<f32, f32, f32, f32>;
init_conf_func_type init_conf_f32 = init_conf<kdnn_rnn_common_f32_t>;
set_conf_func_type set_conf_f32 = set_conf<kdnn_rnn_common_f32_t>;
set_workspace_sizes_func_type set_workspace_sizes_f32 = set_workspace_sizes<kdnn_rnn_common_f32_t>;

using kdnn_rnn_common_bf16_t = kdnn_rnn_common_t<bf16, bf16, f32, bf16>;
init_conf_func_type init_conf_bf16 = init_conf<kdnn_rnn_common_bf16_t>;
set_conf_func_type set_conf_bf16 = set_conf<kdnn_rnn_common_bf16_t>;
set_workspace_sizes_func_type set_workspace_sizes_bf16 = set_workspace_sizes<kdnn_rnn_common_bf16_t>;

using kdnn_rnn_common_bf16f32_t = kdnn_rnn_common_t<bf16, bf16, f32, f32>;
init_conf_func_type init_conf_bf16f32 = init_conf<kdnn_rnn_common_bf16f32_t>;
set_conf_func_type set_conf_bf16f32 = set_conf<kdnn_rnn_common_bf16f32_t>;
set_workspace_sizes_func_type set_workspace_sizes_bf16f32 = set_workspace_sizes<kdnn_rnn_common_bf16f32_t>;

struct kdnn_rnn_fwd_resource_t : public resource_t {
    kdnn_rnn_fwd_resource_t(const std::unique_ptr<KuDNN::RnnFWD> &kdnn_rnn_fwd_prim) noexcept
        : kdnn_rnn_fwd_obj_(new KuDNN::RnnFWD{*(kdnn_rnn_fwd_prim.get())})
    {}

    KuDNN::RnnFWD &get_kdnn_obj() const noexcept
    {
        return *kdnn_rnn_fwd_obj_;
    }

    DNNL_DISALLOW_COPY_AND_ASSIGN(kdnn_rnn_fwd_resource_t);

private:
    std::unique_ptr<KuDNN::RnnFWD> kdnn_rnn_fwd_obj_;
};  // kdnn_rnn_fwd_resource_t

struct kdnn_rnn_fwd_t : public primitive_t {
    struct pd_t : public cpu_rnn_fwd_pd_t {
        using cpu_rnn_fwd_pd_t::cpu_rnn_fwd_pd_t;

        // Need copy constructor because `unique_ptr` is a member of the class
        pd_t(const pd_t &p) : cpu_rnn_fwd_pd_t(p)
        {
            if (p.kdnn_rnn_fwd_prim_) {
                this->kdnn_rnn_fwd_prim_.reset(new KuDNN::RnnFWD{*(p.kdnn_rnn_fwd_prim_.get())});
            }
        }

        DECLARE_COMMON_PD_T("kdnn", kdnn_rnn_fwd_t);

        status_t init(engine_t *engine)
        {
            init_conf_func_type init_conf_func;
            set_conf_func_type set_conf_func;
            set_workspace_sizes_func_type set_workspace_sizes_func;
            const data_type_t src_layer_dt = this->desc()->src_layer_desc.data_type;
            const data_type_t weights_iter_dt = this->desc()->weights_iter_desc.data_type;
            const data_type_t weights_layer_dt = this->desc()->weights_layer_desc.data_type;
            if (src_layer_dt == f32 && everyone_is(f32, weights_iter_dt, weights_layer_dt)) {
                init_conf_func = init_conf_f32;
                set_conf_func = set_conf_f32;
                set_workspace_sizes_func = set_workspace_sizes_f32;
            } else if (src_layer_dt == bf16 && everyone_is(bf16, weights_iter_dt, weights_layer_dt)) {
                init_conf_func = init_conf_bf16f32;
                set_conf_func = set_conf_bf16f32;
                set_workspace_sizes_func = set_workspace_sizes_bf16f32;
            } else {
                return status::unimplemented;
            }
            bool ok = one_of(this->desc()->prop_kind, prop_kind::forward_training, prop_kind::forward_inference) &&
                desc()->weights_iter_desc.data_type == desc()->weights_layer_desc.data_type &&
                one_of(desc()->cell_kind, alg_kind::vanilla_rnn, alg_kind::vanilla_lstm, alg_kind::vanilla_gru,
                    alg_kind::lbr_gru, alg_kind::vanilla_augru, alg_kind::lbr_augru) &&
                set_default_params() == status::success && with_bias();
            if (!ok)
                return status::unimplemented;
            if (desc()->cell_kind == alg_kind::vanilla_rnn) {
                if (!one_of(desc()->activation_kind,
                    alg_kind::eltwise_relu, alg_kind::eltwise_tanh,
                    alg_kind::eltwise_logistic)) {
                        return status::unimplemented;
                    }
            }
            rnn_ = zero<decltype(rnn_)>();
            rnn_.is_brgemm = false;
            ok = init_conf_func(rnn_, *this->desc(), *this->attr(),
                this->src_md(0), this->src_md(1), this->src_md(0x2),
                this->weights_md(0), this->weights_md(1),
                this->arg_md(DNNL_ARG_WEIGHTS_PROJECTION), this->dst_md(0),
                this->dst_md(1), this->dst_md(0x2),
                this->arg_md(DNNL_ARG_BIAS));
            if (!ok)
                return status::unimplemented;

            memory_desc_t new_weights_layer_md = *this->weights_md(0);
            CHECK(set_expected_desc(rnn_, new_weights_layer_md,
                rnn_utils::weights_type_t::layer));
            if (this->weights_layer_md_.format_kind == format_kind::any) {
                this->weights_layer_md_ = new_weights_layer_md;
            } else if (this->weights_layer_md_.format_kind
                    == format_kind::rnn_packed) {
                if (this->weights_layer_md_ != new_weights_layer_md)
                    return status::unimplemented;
            }

            memory_desc_t new_weights_iter_md = *this->weights_md(1);
            CHECK(set_expected_desc(rnn_, new_weights_iter_md,
                rnn_utils::weights_type_t::iter));
            if (this->weights_iter_md_.format_kind == format_kind::any) {
                this->weights_iter_md_ = new_weights_iter_md;
            } else if (this->weights_iter_md_.format_kind
                    == format_kind::rnn_packed) {
                if (this->weights_iter_md_ != new_weights_iter_md)
                    return status::unimplemented;
            }

            if (rnn_.is_lstm_projection) {
                memory_desc_t new_weights_projection_md
                        = *this->arg_md(DNNL_ARG_WEIGHTS_PROJECTION);
                CHECK(set_expected_desc(rnn_, new_weights_projection_md,
                        rnn_utils::weights_type_t::projection));
                if (this->weights_projection_md_.format_kind
                        == format_kind::any) {
                    this->weights_projection_md_ = new_weights_projection_md;
                } else if (this->weights_projection_md_.format_kind
                        == format_kind::rnn_packed) {
                    if (this->weights_projection_md_
                            != new_weights_projection_md)
                        return status::unimplemented;
                }
            }

            CHECK(this->check_layout_consistency(false)); /* is_brgemm */

            set_conf_func(rnn_, *this->desc(), this->weights_md(0),
                this->weights_md(1),
                this->arg_md(DNNL_ARG_WEIGHTS_PROJECTION),
                this->diff_weights_md(0), this->diff_weights_md(1),
                this->arg_md(DNNL_ARG_DIFF_WEIGHTS_PROJECTION));
            set_workspace_sizes_func(rnn_, *this->desc());

            size_t workspaceSize = 0;
            auto &&rnn_fwd = kdnn_utils::convert_to_kdnn_rnn_fwd(&rnn_, *this->desc(), *this->attr(), this->src_md(0),
                this->src_md(1), this->src_md(2), this->weights_md(0), this->weights_md(1),
                this->arg_md(DNNL_ARG_WEIGHTS_PROJECTION), this->dst_md(0), this->dst_md(1),
                this->dst_md(2), this->arg_md(DNNL_ARG_BIAS), workspaceSize);
            if (rnn_.is_training) {
                dims_t ws_dims = {(dim_t)workspaceSize};
                CHECK(memory_desc_init_by_tag(this->ws_md_, 1, ws_dims,
                        data_type::u8, format_tag::x));
            }
            if (!rnn_fwd.first) {
                return status::unimplemented;
            } else {
                kdnn_rnn_fwd_prim_.reset(rnn_fwd.second);
                return status::success;
            }
        }
        rnn_utils::rnn_conf_t rnn_;
        std::unique_ptr<KuDNN::RnnFWD> kdnn_rnn_fwd_prim_;
    }; // pd_t

    kdnn_rnn_fwd_t(const pd_t *kpd) : primitive_t(kpd)
    {
    }

    status_t create_resource(engine_t *engine, resource_mapper_t &mapper) const override
    {
        if (mapper.has_resource(this))
            return status::success;

        mapper.add(this, std::make_unique<kdnn_rnn_fwd_resource_t>(pd()->kdnn_rnn_fwd_prim_));

        return status::success;
    }

    status_t execute(const exec_ctx_t &ctx) const override
    {
        return execute_forward(ctx);
    }

private:
    mutable std::mutex mtx;

    status_t execute_forward(const exec_ctx_t &ctx) const
    {
        std::vector<const void *> srcVec;
        srcVec.push_back(CTX_IN_MEM(const void *, DNNL_ARG_SRC_LAYER));
        srcVec.push_back(CTX_IN_MEM(const void *, DNNL_ARG_SRC_ITER));
        srcVec.push_back(CTX_IN_MEM(const void *, DNNL_ARG_SRC_ITER_C));
        std::vector<const void *> weightVec;
        weightVec.push_back(CTX_IN_MEM(const void *, DNNL_ARG_WEIGHTS_LAYER));
        weightVec.push_back(CTX_IN_MEM(const void *, DNNL_ARG_WEIGHTS_ITER));
        weightVec.push_back(CTX_IN_MEM(const void *, DNNL_ARG_WEIGHTS_PEEPHOLE));
        weightVec.push_back(CTX_IN_MEM(const void *, DNNL_ARG_WEIGHTS_PROJECTION));
        std::vector<void *> dstVec;
        dstVec.push_back(CTX_OUT_MEM(void *, DNNL_ARG_DST_LAYER));
        dstVec.push_back(CTX_OUT_MEM(void *, DNNL_ARG_DST_ITER));
        dstVec.push_back(CTX_OUT_MEM(void *, DNNL_ARG_DST_ITER_C));
        const void *bias = CTX_IN_MEM(const void *, DNNL_ARG_BIAS);
        const void *augruAttention = CTX_IN_MEM(const void *, DNNL_ARG_AUGRU_ATTENTION);
        void *wsBase = CTX_OUT_MEM(void *, DNNL_ARG_WORKSPACE);

        return execute_forward(ctx, srcVec.data(), weightVec.data(), dstVec.data(), bias, wsBase, augruAttention);
    }

    // Execute forward with arbitrary src and dst
    status_t execute_forward(const exec_ctx_t &ctx, const void **src, const void **weight, void **dst,
        const void *bias, void *wsBase, const void *augruAttentions) const
    {
        // Lock here is needed because resource_mapper does not support concurrent access.
        std::lock_guard<std::mutex> _lock{this->mtx};

        KuDNN::RnnFWD &kdnn_obj = (ctx.get_resource_mapper()->get<kdnn_rnn_fwd_resource_t>(this))->get_kdnn_obj();

        try {
            kdnn_obj.Run(src, weight, dst, bias, wsBase, augruAttentions);
        } catch (const std::exception &e) {
            return status::runtime_error;
        }

        return status::success;
    }

    const pd_t *pd() const
    {
        return (const pd_t *)primitive_t::pd().get();
    }
};  // kdnn_rnn_fwd_t

struct kdnn_rnn_bwd_resource_t : public resource_t {
    kdnn_rnn_bwd_resource_t(const std::unique_ptr<KuDNN::RnnBWD> &kdnn_rnn_bwd_prim) noexcept
        : kdnn_rnn_bwd_obj_(new KuDNN::RnnBWD{*(kdnn_rnn_bwd_prim.get())})
    {}

    KuDNN::RnnBWD &get_kdnn_obj() const noexcept
    {
        return *kdnn_rnn_bwd_obj_;
    }

    DNNL_DISALLOW_COPY_AND_ASSIGN(kdnn_rnn_bwd_resource_t);

private:
    std::unique_ptr<KuDNN::RnnBWD> kdnn_rnn_bwd_obj_;
};  // kdnn_rnn_bwd_resource_t

struct kdnn_rnn_bwd_t : public primitive_t {
    struct pd_t : public cpu_rnn_bwd_pd_t {
        using cpu_rnn_bwd_pd_t::cpu_rnn_bwd_pd_t;

        // Need copy constructor because `unique_ptr` is a member of the class
        pd_t(const pd_t &p) : cpu_rnn_bwd_pd_t(p)
        {
            if (p.kdnn_rnn_bwd_prim_) {
                this->kdnn_rnn_bwd_prim_.reset(new KuDNN::RnnBWD{*(p.kdnn_rnn_bwd_prim_.get())});
            }
        }

        DECLARE_COMMON_PD_T("kdnn", kdnn_rnn_bwd_t);

        status_t init(engine_t *engine)
        {
            init_conf_func_type init_conf_func;
            set_conf_func_type set_conf_func;
            set_workspace_sizes_func_type set_workspace_sizes_func;
            const data_type_t src_layer_dt = this->desc()->src_layer_desc.data_type;
            const data_type_t weights_iter_dt = this->desc()->weights_iter_desc.data_type;
            const data_type_t weights_layer_dt = this->desc()->weights_layer_desc.data_type;
            if (src_layer_dt == f32 && everyone_is(f32, weights_iter_dt, weights_layer_dt)) {
                init_conf_func = init_conf_f32;
                set_conf_func = set_conf_f32;
                set_workspace_sizes_func = set_workspace_sizes_f32;
            } else if (src_layer_dt == bf16 && everyone_is(bf16, weights_iter_dt, weights_layer_dt)) {
                init_conf_func = init_conf_bf16;
                set_conf_func = set_conf_bf16;
                set_workspace_sizes_func = set_workspace_sizes_bf16;
            } else {
                return status::unimplemented;
            }
            bool ok = desc_.prop_kind == prop_kind::backward &&
                desc()->weights_iter_desc.data_type == desc()->weights_layer_desc.data_type &&
                one_of(desc()->cell_kind, alg_kind::vanilla_rnn, alg_kind::vanilla_lstm, alg_kind::vanilla_gru,
                    alg_kind::lbr_gru, alg_kind::vanilla_augru, alg_kind::lbr_augru) &&
                set_default_params() == status::success && with_bias();
            if (!ok)
                return status::unimplemented;

            rnn_ = zero<decltype(rnn_)>();
            rnn_.is_brgemm = false;
            ok = init_conf_func(rnn_, *this->desc(), *this->attr(),
                this->src_md(0), this->src_md(1), this->src_md(0x2),
                this->weights_md(0), this->weights_md(1),
                this->arg_md(DNNL_ARG_WEIGHTS_PROJECTION), this->dst_md(0),
                this->dst_md(1), this->dst_md(0x2),
                this->arg_md(DNNL_ARG_BIAS));

            if (rnn_.is_bf16_conf()) {
                if (!utils::one_of(
                            rnn_.bias_dt, data_type::bf16, data_type::f32)
                        || rnn_.src_iter_c_dt != rnn_.dst_iter_c_dt
                        || !utils::one_of(rnn_.src_iter_c_dt, data_type::undef,
                                data_type::bf16, data_type::f32))
                    return status::unimplemented;
            } else if (rnn_.bias_dt != data_type::f32
                    || !utils::one_of(rnn_.src_iter_c_dt, data_type::undef,
                            data_type::f32)
                    || rnn_.src_iter_c_dt != rnn_.dst_iter_c_dt)
                return status::unimplemented;

            memory_desc_t new_weights_layer_md = *this->weights_md(0);
            CHECK(set_expected_desc(rnn_, new_weights_layer_md,
                rnn_utils::weights_type_t::layer));
            if (this->weights_layer_md_.format_kind == format_kind::any) {
                this->weights_layer_md_ = new_weights_layer_md;
            } else if (this->weights_layer_md_.format_kind
                    == format_kind::rnn_packed) {
                if (this->weights_layer_md_ != new_weights_layer_md)
                    return status::unimplemented;
            }

            memory_desc_t new_weights_iter_md = *this->weights_md(1);
            CHECK(set_expected_desc(rnn_, new_weights_iter_md,
                rnn_utils::weights_type_t::iter));
            if (this->weights_iter_md_.format_kind == format_kind::any) {
                this->weights_iter_md_ = new_weights_iter_md;
            } else if (this->weights_iter_md_.format_kind
                    == format_kind::rnn_packed) {
                if (this->weights_iter_md_ != new_weights_iter_md)
                    return status::unimplemented;
            }

            CHECK(this->check_layout_consistency(false)); /* is_brgemm */

            set_conf_func(rnn_, *this->desc(), this->weights_md(0),
                    this->weights_md(1),
                    this->arg_md(DNNL_ARG_WEIGHTS_PROJECTION),
                    this->diff_weights_md(0), this->diff_weights_md(1),
                    this->arg_md(DNNL_ARG_DIFF_WEIGHTS_PROJECTION));
            set_workspace_sizes_func(rnn_, *this->desc());

            auto &&rnn_bwd = kdnn_utils::convert_to_kdnn_rnn_bwd(&rnn_, *this->desc(), *this->attr(), this->src_md(0),
                this->src_md(1), this->src_md(2), this->weights_md(0), this->weights_md(1),
                this->arg_md(DNNL_ARG_WEIGHTS_PROJECTION), this->dst_md(0), this->dst_md(1),
                this->dst_md(2), this->arg_md(DNNL_ARG_BIAS),
                this->diff_dst_md(0), this->diff_dst_md(1), this->diff_dst_md(2),
                this->diff_src_md(0), this->diff_src_md(1), this->diff_src_md(2),
                this->diff_weights_md(0), this->diff_weights_md(0), this->arg_md(DNNL_ARG_DIFF_BIAS));
            if (!rnn_bwd.first) {
                return status::unimplemented;
            } else {
                kdnn_rnn_bwd_prim_.reset(rnn_bwd.second);
                return status::success;
            }
        }
        rnn_utils::rnn_conf_t rnn_;
        std::unique_ptr<KuDNN::RnnBWD> kdnn_rnn_bwd_prim_;
    }; // pd_t

    kdnn_rnn_bwd_t(const pd_t *kpd) : primitive_t(kpd)
    {
    }

    status_t create_resource(engine_t *engine, resource_mapper_t &mapper) const override
    {
        if (mapper.has_resource(this))
            return status::success;

        mapper.add(this, std::make_unique<kdnn_rnn_bwd_resource_t>(pd()->kdnn_rnn_bwd_prim_));

        return status::success;
    }

    status_t execute(const exec_ctx_t &ctx) const override
    {
        return execute_backward(ctx);
    }

private:
    mutable std::mutex mtx;

    status_t execute_backward(const exec_ctx_t &ctx) const
    {
        std::vector<const void *> srcVec;
        srcVec.push_back(CTX_IN_MEM(const void *, DNNL_ARG_SRC_LAYER));
        srcVec.push_back(CTX_IN_MEM(const void *, DNNL_ARG_SRC_ITER));
        srcVec.push_back(CTX_IN_MEM(const void *, DNNL_ARG_SRC_ITER_C));
        std::vector<const void *> weightVec;
        weightVec.push_back(CTX_IN_MEM(const void *, DNNL_ARG_WEIGHTS_LAYER));
        weightVec.push_back(CTX_IN_MEM(const void *, DNNL_ARG_WEIGHTS_ITER));
        weightVec.push_back(CTX_IN_MEM(const void *, DNNL_ARG_WEIGHTS_PEEPHOLE));
        weightVec.push_back(CTX_IN_MEM(const void *, DNNL_ARG_WEIGHTS_PROJECTION));
        std::vector<const void *> dstVec;
        dstVec.push_back(CTX_IN_MEM(const void *, DNNL_ARG_DST_LAYER));
        dstVec.push_back(CTX_IN_MEM(const void *, DNNL_ARG_DST_ITER));
        dstVec.push_back(const_cast<const void *>(CTX_OUT_MEM(void *, DNNL_ARG_DST_ITER_C)));
        const void *bias = CTX_IN_MEM(const void *, DNNL_ARG_BIAS);
        std::vector<const void *> diffDstVec;
        diffDstVec.push_back(CTX_IN_MEM(const void *, DNNL_ARG_DIFF_DST_LAYER));
        diffDstVec.push_back(CTX_IN_MEM(const void *, DNNL_ARG_DIFF_DST_ITER));
        diffDstVec.push_back(CTX_IN_MEM(const void *, DNNL_ARG_DIFF_DST_ITER_C));

        std::vector<void *> diffSrcVec;
        diffSrcVec.push_back(CTX_OUT_MEM(void *, DNNL_ARG_DIFF_SRC_LAYER));
        diffSrcVec.push_back(CTX_OUT_MEM(void *, DNNL_ARG_DIFF_SRC_ITER));
        diffSrcVec.push_back(CTX_OUT_MEM(void *, DNNL_ARG_DIFF_SRC_ITER_C));
        std::vector<void *> diffWeightVec;
        diffWeightVec.push_back(CTX_OUT_MEM(void *, DNNL_ARG_DIFF_WEIGHTS_LAYER));
        diffWeightVec.push_back(CTX_OUT_MEM(void *, DNNL_ARG_DIFF_WEIGHTS_ITER));
        diffWeightVec.push_back(CTX_OUT_MEM(float *, DNNL_ARG_DIFF_WEIGHTS_PEEPHOLE));
        diffWeightVec.push_back(CTX_OUT_MEM(float *, DNNL_ARG_DIFF_WEIGHTS_PROJECTION));

        void *wsBase = CTX_OUT_MEM(void *, DNNL_ARG_WORKSPACE);
        void *diffBias = CTX_OUT_MEM(void *, DNNL_ARG_DIFF_BIAS);

        const void *augruAttention = CTX_IN_MEM(const void *, DNNL_ARG_AUGRU_ATTENTION);
        void *diffAugruAttention = CTX_OUT_MEM(void *, DNNL_ARG_DIFF_AUGRU_ATTENTION);

        return execute_backward(ctx, srcVec.data(), weightVec.data(), dstVec.data(), bias, wsBase,
                                diffDstVec.data(), diffSrcVec.data(), diffWeightVec.data(), diffBias,
                                augruAttention, diffAugruAttention);
    }

    // Execute forward with arbitrary src and dst
    status_t execute_backward(const exec_ctx_t &ctx,
        const void **src, const void **weight, const void **dst, const void *bias, void *wsBase, const void **diffDstVec,
        void **diffSrcVec, void **diffWeightVec, void *diffBias, const void *augruAttention,
            void *diffAugruAttention) const
    {
        // Lock here is needed because resource_mapper does not support concurrent access.
        std::lock_guard<std::mutex> _lock{this->mtx};

        KuDNN::RnnBWD &kdnn_obj = (ctx.get_resource_mapper()->get<kdnn_rnn_bwd_resource_t>(this))->get_kdnn_obj();

        try {
            kdnn_obj.Run(src, weight, dst, bias, wsBase, diffDstVec, diffSrcVec, diffWeightVec, diffBias,
                augruAttention, diffAugruAttention);
        } catch (const std::exception &e) {
            return status::runtime_error;
        }

        return status::success;
    }

    const pd_t *pd() const
    {
        return (const pd_t *)primitive_t::pd().get();
    }
}; // kdnn_rnn_bwd_t

} // namespace aarch64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif // CPU_AARCH64_KUDNN_RNN_HPP
