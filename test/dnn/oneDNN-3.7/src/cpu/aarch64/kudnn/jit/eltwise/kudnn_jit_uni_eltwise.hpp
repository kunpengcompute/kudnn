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

#ifndef CPU_AARCH64_KUDNN_JIT_UNI_ELTWISE_HPP
#define CPU_AARCH64_KUDNN_JIT_UNI_ELTWISE_HPP

#include <assert.h>

#include "common/c_types_map.hpp"
#include "common/primitive.hpp"
#include "common/type_helpers.hpp"
#include "common/utils.hpp"

#include "cpu/cpu_eltwise_pd.hpp"

#include "cpu/aarch64/cpu_isa_traits.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace aarch64 {

struct kdnn_jit_uni_eltwise_kernel;

template <cpu_isa_t isa, impl::data_type_t d_type>
struct kdnn_jit_uni_eltwise_fwd_t : public primitive_t {
    struct pd_t : public cpu_eltwise_fwd_pd_t {
        using cpu_eltwise_fwd_pd_t::cpu_eltwise_fwd_pd_t;

        DECLARE_COMMON_PD_T("kdnn_jit_eltwise", kdnn_jit_uni_eltwise_fwd_t);

        status_t init(engine_t *engine);
    };

    kdnn_jit_uni_eltwise_fwd_t(const pd_t *apd);
    virtual ~kdnn_jit_uni_eltwise_fwd_t();

    typedef typename prec_traits<d_type>::type data_t;

    status_t init(engine_t *engine) override;

    status_t execute(const exec_ctx_t &ctx) const override;

private:
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }
    std::unique_ptr<kdnn_jit_uni_eltwise_kernel> kernel_;
};

template <cpu_isa_t isa, impl::data_type_t d_type>
struct kdnn_jit_uni_eltwise_bwd_t : public primitive_t {
    struct pd_t : public cpu_eltwise_bwd_pd_t {
        using cpu_eltwise_bwd_pd_t::cpu_eltwise_bwd_pd_t;

        DECLARE_COMMON_PD_T("kdnn_jit_eltwise", kdnn_jit_uni_eltwise_bwd_t);

        status_t init(engine_t *engine);
    };

    kdnn_jit_uni_eltwise_bwd_t(const pd_t *apd);
    virtual ~kdnn_jit_uni_eltwise_bwd_t();

    typedef typename prec_traits<d_type>::type data_t;

    status_t init(engine_t *engine) override;

    status_t execute(const exec_ctx_t &ctx) const override;

private:
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }
    std::unique_ptr<kdnn_jit_uni_eltwise_kernel> kernel_;
};

} // namespace aarch64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif
