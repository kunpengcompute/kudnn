/*******************************************************************************
* Copyright 2017-2024 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

#include <algorithm>
#include <cstring>
#include <random>
#include <vector>
#include <omp.h>
#include <arm_sve.h>
#include "oneapi/dnnl/dnnl.h"

#include "utils/fill.hpp"
#include "utils/parallel.hpp"

#include "dnnl_common.hpp"
#include "dnnl_memory.hpp"

#include "ip/ip.hpp"

static constexpr float FLOAT16_ULP = 0.009766f;
static constexpr float BFLOAT16_ULP = 0.078125f;
static constexpr float FLOAT32_ULP = 0.000000119f;

namespace ip {

dnnl_status_t init_pd(init_pd_args_t<prb_t> &init_pd_args) {
    const prb_t *prb = init_pd_args.prb;
    res_t *res = init_pd_args.res;

    auto src_d = dnn_mem_t::init_md(
            prb->ndims, prb->src_dims().data(), prb->get_dt(SRC), prb->stag);
    auto wei_d = dnn_mem_t::init_md(
            prb->ndims, prb->wei_dims().data(), prb->get_dt(WEI), prb->wtag);
    auto bia_d = dnn_mem_t::init_md(
            1, prb->bia_dims().data(), prb->get_dt(BIA), tag::any);
    auto dst_d = dnn_mem_t::init_md(
            2, prb->dst_dims().data(), prb->get_dt(DST), prb->dtag);

    attr_args_t attr_args;
    attr_args.prepare_post_ops_mds(prb->attr, 2, prb->dst_dims().data());
    auto wei_scale = prb->attr.scales.get(DNNL_ARG_WEIGHTS);
    if (wei_scale.policy == policy_t::PER_OC) {
        attr_args.prepare_scales(prb->attr, DNNL_ARG_WEIGHTS, 1);
    }
    auto dnnl_attr = make_benchdnn_dnnl_wrapper(
            create_dnnl_attr(prb->attr, attr_args));

    switch (prb->dir) {
        case FWD_D:
        case FWD_B:
        case FWD_I:
            if (prb->dir != FWD_B) bia_d.reset(nullptr);
            TIME_C_PD(DNN_SAFE_STATUS(
                    dnnl_inner_product_forward_primitive_desc_create(
                            &init_pd_args.pd, init_pd_args.engine,
                            prb->dir == FWD_I ? dnnl_forward_inference
                                              : dnnl_forward_training,
                            init_pd_args.src_md ? init_pd_args.src_md : src_d,
                            wei_d, bia_d, dst_d, dnnl_attr)));
            break;
        case BWD_D:
            TIME_C_PD(DNN_SAFE_STATUS(
                    dnnl_inner_product_backward_data_primitive_desc_create(
                            &init_pd_args.pd, init_pd_args.engine, src_d, wei_d,
                            dst_d, init_pd_args.hint, dnnl_attr)));
            break;
        case BWD_W:
        case BWD_WB:
            if (prb->dir == BWD_W) bia_d.reset(nullptr);
            TIME_C_PD(DNN_SAFE_STATUS(
                    dnnl_inner_product_backward_weights_primitive_desc_create(
                            &init_pd_args.pd, init_pd_args.engine, src_d, wei_d,
                            bia_d, dst_d, init_pd_args.hint, dnnl_attr)));
            break;
        default: DNN_SAFE_STATUS(dnnl_invalid_arguments);
    }

    return dnnl_success;
}

int init_prim_ref(benchdnn_dnnl_wrapper_t<dnnl_primitive_t> &prim_ref,
        const prb_t *prb, res_t *res) {
    if (!(has_bench_mode_bit(mode_bit_t::corr) && fast_ref)) return OK;
    // Create prim_ref if only original prim was successfully created.
    if (res->state != INITIALIZED) return OK;

    // f32 cases should go through reference no matter what.
    if (is_cpu() && (prb->src_dt() == dnnl_f32 && prb->wei_dt() == dnnl_f32))
        return OK;

    // Create a new copy of prb to avoid potentially corrupting the test by
    // modifying prb in place.
    auto cpu_attr = prb->attr;
    update_cpu_ref_attrs(cpu_attr);
    std::vector<std::vector<dnnl_data_type_t>> prim_ref_dt {
            prb->dt, {dnnl_f32}};
    if (is_cpu()) prim_ref_dt.erase(prim_ref_dt.begin());
    dnnl_primitive_t prim_ref_ {};

    for (const auto &prim_ref_dt_i : prim_ref_dt) {
        prb_t prb_cpu {*prb, prb->mb, prb->dir, prim_ref_dt_i, tag::any,
                tag::any, tag::any, cpu_attr, prb->ctx_init, prb->ctx_exe};

        init_pd_args_t<prb_t> init_pd_args(
                /* res = */ nullptr, get_cpu_engine(), &prb_cpu, prb->dir,
                /* hint = */ nullptr, /* src_md = */ nullptr);
        init_pd(init_pd_args);

        benchdnn_dnnl_wrapper_t<dnnl_primitive_desc_t> pdw;
        fetch_impl(pdw, init_pd_args, /* res = */ nullptr,
                /* is_service_prim = */ true);

        if (pdw) {
            if (query_impl_info(pdw) == "ref:any") return OK;

            auto st = dnnl_primitive_create(&prim_ref_, pdw);
            if (st != dnnl_success) continue;

            BENCHDNN_PRINT(5, "CPU reference oneDNN implementation: %s\n",
                    query_impl_info(pdw).c_str());
            res->prim_ref_repro = prb_cpu.str();
            break;
        }
    }

    prim_ref.reset(prim_ref_);
    return OK;
}

int check_reorder_presence(
        const prb_t *prb, dnn_mem_t &mem_dt, dnn_mem_t &mem_fp, res_t *res) {
    if (!is_cpu()) return OK;

    dnnl_data_type_t dt_check = dnnl_s8;
#if defined(DNNL_AARCH64) && (DNNL_AARCH64 == 1)
    /* Note for x64:
    Both data types of src and weight are s8, oneDNN addds 128 to one of the s8
    input to make it of type u8 instead, as explained in
    https://oneapi-src.github.io/oneDNN/dev_guide_int8_computations.html or
    doc/advanced/int8_computations.md
    It is because `VPDPBUSD` instruction uses the combination of s8 and u8 as
    input.

    Note for AArch64:
    Because dot product instructions of AArch64 "SDOT" receives s8 input
    for both src and weight, the addition (and its counterpart of subtraction)
    is not required for AArch64.
    */
    if (res->impl_name.find("jit", 0) == 0) dt_check = dnnl_u8;
#endif

    const bool wei_x8x8
            = prb->get_dt(WEI) == dnnl_s8 && prb->get_dt(SRC) == dt_check;
    const bool is_def_zp = prb->attr.zero_points.is_def(DNNL_ARG_SRC);
    if (wei_x8x8 || !is_def_zp) {
        // Check that s8 -> s8_comp exists in the library since users may have
        // already quantized data.
        dnn_mem_t mem_fp_s8(mem_fp.md_, dnnl_s8, tag::abx, get_cpu_engine());
        dnn_mem_t mem_dt_s8(mem_dt.md_, get_test_engine());
        SAFE(mem_fp_s8.reorder(mem_fp), WARN);
        SAFE(mem_dt_s8.reorder(mem_fp_s8), WARN);
        SAFE(mem_dt.size() == mem_dt_s8.size() ? OK : FAIL, WARN);
        int rc = std::memcmp((void *)mem_dt, (void *)mem_dt_s8, mem_dt.size());
        SAFE(rc == 0 ? OK : FAIL, WARN);
    }

    return OK;
}

int fill_data(data_kind_t kind, const prb_t *prb, const cfg_t &cfg,
        dnn_mem_t &mem_dt, dnn_mem_t &mem_fp, res_t *res) {
    const auto nelems = mem_fp.nelems();
    if (nelems == 0) return OK;

    // Refer to modes documentation for filling principles.
    if (has_bench_mode_bit(mode_bit_t::bitwise)) {
        return fill_random_real(mem_dt, mem_fp, res);
    }

    cfg_t::density_args_t density_args;
    density_args.data_kind = kind;
    density_args.n_acc = prb->count_n_acc();
    const auto density = cfg.get_density(density_args);

    /* Do fixed partitioning to have same filling for any number of threads */
    const int64_t chunk_size = 64;
    const int64_t n_chunks = div_up(nelems, chunk_size);

    benchdnn_parallel_nd(n_chunks, [&](int64_t idx_chunk) {
        int64_t idx_start = idx_chunk * chunk_size;
        int64_t idx_end = MIN2(idx_start + chunk_size, nelems);
        // Note: we use a different seed for each chunk to avoid
        // repeating patterns. We could use discard(idx_start) too but
        // it has a complexity in O(idx_start). We also add 1 to avoid
        // seeding with 0.
        std::minstd_rand int_seed(kind * nelems + idx_start + 1);
        int_seed.discard(1);
        std::minstd_rand b_seed(kind * nelems + idx_start + 1);
        b_seed.discard(10);

        std::uniform_int_distribution<> gen(
                cfg.get_range_min(kind), cfg.get_range_max(kind));
        std::bernoulli_distribution b_dist(density);

        // make sure the first element is positive
        if (idx_start == 0) {
            float val = 0;
            while (val <= 0)
                val = gen(int_seed);
            mem_fp.set_elem(
                    0, round_to_nearest_representable(cfg.get_dt(kind), val));
            idx_start += 1;
        }

        for (int64_t idx = idx_start; idx < idx_end; ++idx) {
            bool is_one = density == 1.f ? true : b_dist(b_seed);
            float val = is_one * gen(int_seed);
            mem_fp.set_elem(
                    idx, round_to_nearest_representable(cfg.get_dt(kind), val));
        }
    });

    SAFE(mem_dt.reorder(mem_fp, cfg.get_swapped_dt(kind)), WARN);

    if (kind == WEI)
        SAFE(check_reorder_presence(prb, mem_dt, mem_fp, res), WARN);

    return OK;
}

void skip_unimplemented_prb(const prb_t *prb, res_t *res) {
    skip_unimplemented_data_type(
            {prb->get_dt(SRC), prb->get_dt(WEI), prb->get_dt(DST)}, prb->dir,
            res);
    skip_unimplemented_sum_po(prb->attr, res, dnnl_inner_product,
            prb->get_dt(SRC), prb->get_dt(DST));
    skip_unimplemented_prelu_po(prb->attr, res, dnnl_inner_product);

    if (is_cpu()) {
        auto is_dt_f16_or_f32 = [&](dnnl_data_type_t dt) {
            return dt == dnnl_f16 || dt == dnnl_f32;
        };

        if (!IMPLICATION(prb->get_dt(SRC) == dnnl_f16
                            || prb->get_dt(WEI) == dnnl_f16
                            || prb->get_dt(DST) == dnnl_f16,
                    is_dt_f16_or_f32(prb->get_dt(SRC))
                            && is_dt_f16_or_f32(prb->get_dt(WEI))
                            && is_dt_f16_or_f32(prb->get_dt(DST)))) {
            res->state = SKIPPED, res->reason = CASE_NOT_SUPPORTED;
        }
    }
}

void skip_invalid_prb(const prb_t *prb, res_t *res) {}

void compute_pointwise_trh_fwd(const prb_t *prb, const args_t &args, std::vector<float>& pointwise_trh) {
    const dnn_mem_t &wei_m = args.find(DNNL_ARG_WEIGHTS);
    const dnn_mem_t &src_m = args.find(DNNL_ARG_SRC);
    const dnn_mem_t &dst_m = args.find(DNNL_ARG_DST);
    const dnn_mem_t &bia_m = args.find(DNNL_ARG_BIAS);

    int64_t M = prb->mb;
    int64_t N = prb->oc;
    int64_t K = prb->ic * prb->id * prb->ih * prb->iw;
    pointwise_trh.resize(M * N);
    float ulp = BFLOAT16_ULP;
    if (prb->dt[0] == dnnl_f16 && prb->dt[1] == dnnl_f16) ulp = FLOAT16_ULP;
    if (prb->dt[0] == dnnl_f32 && prb->dt[1] == dnnl_f32) ulp = FLOAT32_ULP;
    double int_ulp = 0.0f;
    bool has_po = false;
    if (prb->attr.post_ops.len() > 0) {
        has_po = true;
    }
    if (((prb->dt[2] != dnnl_f16) && (prb->dt[2] != dnnl_bf16) && (prb->dt[2] != dnnl_f32))) {
        // in case dst is integral and any of inputs is fp then computations are performed in fp
        // so any minor error may lead to incorrect rounding, thus int_ulp is increased corresponding fp ulp
        if ((prb->dt[0] == dnnl_f16) || (prb->dt[1] == dnnl_f16) || (prb->dt[3] == dnnl_f16)) {
            int_ulp = 1.0f;
        }
        if ((prb->dt[0] == dnnl_bf16) || (prb->dt[1] == dnnl_bf16) || (prb->dt[3] == dnnl_bf16)) {
            int_ulp = 1.0f;
        }
        if ((prb->dt[0] == dnnl_f32) || (prb->dt[1] == dnnl_f32) || (prb->dt[3] == dnnl_f32)) {
            int_ulp = 1.0f;
        }
        if (has_po) {
            int_ulp = 1.0f;
        }
        if (!prb->attr.scales.get(DNNL_ARG_SRC).is_def() || !prb->attr.scales.get(DNNL_ARG_WEIGHTS).is_def()) {
            int_ulp = 1.0f;
        }
    }

    const dnn_mem_t &src_scales
            = args.find(DNNL_ARG_ATTR_SCALES | DNNL_ARG_SRC);
    const dnn_mem_t &wei_scales
            = args.find(DNNL_ARG_ATTR_SCALES | DNNL_ARG_WEIGHTS);
    const bool has_src_scale = !prb->attr.scales.get(DNNL_ARG_SRC).is_def();
    const bool has_wei_scale = !prb->attr.scales.get(DNNL_ARG_WEIGHTS).is_def();
    const int wei_scale_mask
            = prb->attr.scales.get_mask(DNNL_ARG_WEIGHTS, dnnl_inner_product);
    float src_scale = has_src_scale ? src_scales.get_elem(0) : 1.f;

    #pragma omp parallel for
    for (int64_t m = 0; m < M; m++) {
        for (int64_t n = 0; n < N; n++) {
            double P_mn = 0;
            double matmul_res = 0.;
            for (int64_t k = 0; k < K; k++) {
                double s = ((float *)src_m)[k + m * K];
                double wei = ((float *)wei_m)[k + n * K];
                double wei_scale = has_wei_scale ? wei_scales.get_elem(wei_scale_mask > 0 ? n : 0) : 1.f;
                double tmp = s * wei * src_scale * wei_scale;
                matmul_res += tmp;
                P_mn = P_mn < std::abs(tmp) ? std::abs(tmp) : P_mn;
            }
            if (prb->dir & FLAG_BIA) {
                int64_t bia_off = bia_off_f(prb, n);
                float *bia_ptr = (float *)bia_m;
                matmul_res += bia_ptr[bia_off];
            }
            pointwise_trh[n + m * N] = std::max(K * P_mn * ulp, int_ulp);
            if (has_po) {
                // when we have post-ops the error estimation becomes more complex:
                // supposing we have f(g(y)) where g(y) is matmul operation
                // f - is post-ops
                // let's consider x = g(y). For single matmul we can estimate error
                // [x - dx, x + dx].
                // For standalone f(x) we also know acceptable error [f - df, f + df]
                // Not let's take into account the fact that f is complex function
                // then interval becomes [f(x) - df - f'(x)*dx, f(x) + df + f'(x)*dx]
                // (considered the case when f'(x) is monotonic and positive)
                // the above computed threshold is actually dx
                // for each kind of post-ops we compute the whole expression (df + abs(f'(x))*dx)
                // as acceptable error. df for eltwise functions is defined in eltwise.cpp source file
                if (prb->attr.post_ops.entry[0].is_sum_kind()) {
                    float scale = prb->attr.post_ops.entry[0].sum.scale;
                    float *dst_ptr = (float *)dst_m;
                        matmul_res += scale * dst_ptr[n + m * N];
                    // compare.cpp for thr values
                    float trh = 4e-5f;
                    if (prb->dt[2] == dnnl_f16) {
                        trh = 5e-3;
                    }
                    pointwise_trh[n + m * N] = std::abs(scale * matmul_res) * trh +  std::abs(scale * pointwise_trh[n + m * N]);
                }
                if (prb->attr.post_ops.entry[0].is_eltwise_kind()) {
                    float alpha = prb->attr.post_ops.entry[0].eltwise.alpha;
                    float beta = prb->attr.post_ops.entry[0].eltwise.beta;
                    float trh = (prb->dt[2] == dnnl_f32 || prb->dt[2] == dnnl_f64) ? 4e-6f : epsilon_dt(prb->dt[2]);
                    // Tolerate bigger compute errors for complex algorithms.
                    const bool alg_has_higher_tolerance = prb->attr.post_ops.entry[0].kind == attr_t::post_ops_t::kind_t::GELU_TANH ||
                        prb->attr.post_ops.entry[0].kind == attr_t::post_ops_t::kind_t::ELU ||
                        prb->attr.post_ops.entry[0].kind == attr_t::post_ops_t::kind_t::SWISH ||
                        prb->attr.post_ops.entry[0].kind == attr_t::post_ops_t::kind_t::TANH ||
                        prb->attr.post_ops.entry[0].kind == attr_t::post_ops_t::kind_t::SRELU ||
                        prb->attr.post_ops.entry[0].kind == attr_t::post_ops_t::kind_t::MISH ||
                        prb->attr.post_ops.entry[0].kind == attr_t::post_ops_t::kind_t::LOG ||
                            (is_nvidia_gpu() && prb->attr.post_ops.entry[0].kind == attr_t::post_ops_t::kind_t::POW)
                                || ((prb->attr.post_ops.entry[0].kind == attr_t::post_ops_t::kind_t::ELU_DST ||
                                    prb->attr.post_ops.entry[0].kind == attr_t::post_ops_t::kind_t::TANH_DST));
                    if ((prb->dt[2] == dnnl_f32 || prb->dt[2] == dnnl_f64) && alg_has_higher_tolerance)
                        trh = 4e-5f;
                    // for latter fomulas refer to oneDNN doc
                    // for all relative eerror multiplicators eltwise.cpp could serve as referrence
                        switch (prb->attr.post_ops.entry[0].kind) {
                            case attr_t::post_ops_t::kind_t::ABS: {
                                pointwise_trh[n + m * N] = std::abs(matmul_res) * trh + pointwise_trh[n + m * N];
                                break;
                            }
                            case attr_t::post_ops_t::kind_t::CLIP: {
                                double r = matmul_res;
                                r = std::min<double>(r, beta);
                                r = std::max<double>(r, alpha);
                                pointwise_trh[n + m * N] = std::abs(r) * trh + pointwise_trh[n + m * N];
                                break;
                            }
                            case attr_t::post_ops_t::kind_t::CLIP_V2: {
                                double r = matmul_res;
                                r = std::min<double>(r, beta);
                                r = std::max<double>(r, alpha);
                                pointwise_trh[n + m * N] = std::abs(r) * trh + pointwise_trh[n + m * N];
                                break;
                            }
                            case attr_t::post_ops_t::kind_t::CLIP_V2_DST: {
                                double r = matmul_res;
                                r = std::min<double>(r, beta);
                                r = std::max<double>(r, alpha);
                                pointwise_trh[n + m * N] = std::abs(r) * trh + pointwise_trh[n + m * N];
                                break;
                            }
                            case attr_t::post_ops_t::kind_t::ELU: {
                                if (matmul_res < 2e-5f) {
                                    // When the left and right derivative values ​​are different, choose the larger side to calculate the threshold.
                                    pointwise_trh[n + m * N] = std::max(matmul_res * trh + pointwise_trh[n + m * N], std::abs(alpha * (std::exp(matmul_res) - 1) * trh) + std::abs(alpha * (std::exp(matmul_res)) * pointwise_trh[n + m * N]));
                                }
                                break;
                            }
                            case attr_t::post_ops_t::kind_t::LINEAR: {
                                pointwise_trh[n + m * N] = std::abs((alpha * matmul_res + beta) * trh) + std::abs(alpha) * pointwise_trh[n + m * N];
                                break;
                            }
                            case attr_t::post_ops_t::kind_t::ROUND: {
                                pointwise_trh[n + m * N] = std::abs(std::round(matmul_res) * trh) + pointwise_trh[n + m * N];
                                break;
                            }
                            case attr_t::post_ops_t::kind_t::ELU_DST: {
                                if (matmul_res < 0) {
                                    pointwise_trh[n + m * N] = std::abs(alpha * (std::exp(matmul_res) - 1) * trh) + std::abs(alpha * (std::exp(matmul_res)) * pointwise_trh[n + m * N]);
                                }
                                break;
                            }
                            case attr_t::post_ops_t::kind_t::EXP: {
                                pointwise_trh[n + m * N] = std::exp(matmul_res) * trh + std::exp(matmul_res) * pointwise_trh[n + m * N];
                                break;
                            }
                            case attr_t::post_ops_t::kind_t::EXP_DST: {
                                pointwise_trh[n + m * N] = std::exp(matmul_res) * trh + std::exp(matmul_res) * pointwise_trh[n + m * N];
                                break;
                            }
                            case attr_t::post_ops_t::kind_t::HARDSIGMOID: {
                                pointwise_trh[n + m * N] = std::abs(std::max(0.0, std::min(1.0, alpha * matmul_res + beta)) * trh) + std::abs(alpha) * pointwise_trh[n + m * N];
                                break;
                            }
                            case attr_t::post_ops_t::kind_t::HARDSWISH: {
                                pointwise_trh[n + m * N] = std::abs(matmul_res * std::max(0.0, std::min(1.0, alpha * matmul_res + beta)) * trh) + (std::abs(alpha * matmul_res) + std::max(0.0, std::min(1.0, alpha * matmul_res + beta))) * pointwise_trh[n + m * N];
                                break;
                            }
                            case attr_t::post_ops_t::kind_t::LOG: {
                                if (std::abs(matmul_res) < 2e-5f) {
                                    pointwise_trh[n + m * N] = -1;
                                } else {
                                    pointwise_trh[n + m * N] = std::abs(std::log(matmul_res) * trh) + 1. / std::abs(matmul_res) * pointwise_trh[n + m * N];
                                }
                                break;
                            }
                            case attr_t::post_ops_t::kind_t::LOGISTIC: {
                                pointwise_trh[n + m * N] = 1. / (1. + std::exp(-matmul_res)) * trh + std::exp(-matmul_res) / ((1. + std::exp(-matmul_res)) * (1. + std::exp(-matmul_res))) * pointwise_trh[n + m * N];
                                break;
                            }
                            case attr_t::post_ops_t::kind_t::LOGISTIC_DST: {
                                pointwise_trh[n + m * N] = 1. / (1. + std::exp(-matmul_res)) * trh + std::exp(-matmul_res) / ((1. + std::exp(-matmul_res)) * (1. + std::exp(-matmul_res))) * pointwise_trh[n + m * N];
                                break;
                            }
                            case attr_t::post_ops_t::kind_t::POW: {
                                if ((std::abs(matmul_res - 1) < 2e-5f || std::abs(matmul_res + 1) < 2e-5f) && beta > 1e6f) {
                                    pointwise_trh[n + m * N] = -1;
                                } else if (std::abs(matmul_res) < 2e-5f && std::fabs(beta - std::round(beta)) != 0) {
                                    pointwise_trh[n + m * N] = -1;
                                } else {
                                    pointwise_trh[n + m * N] = std::abs(alpha * std::pow(matmul_res, beta) * trh) + std::abs(alpha * (beta - 1.) *  std::pow(matmul_res, beta - 1.)) * pointwise_trh[n + m * N];
                                }
                                break;
                            }
                            case attr_t::post_ops_t::kind_t::RELU: {
                                if (matmul_res > 0) {
                                    pointwise_trh[n + m * N] = std::abs(matmul_res * trh) + pointwise_trh[n + m * N];
                                } else {
                                    pointwise_trh[n + m * N] = std::abs(alpha * matmul_res * trh) + std::abs(alpha) * pointwise_trh[n + m * N];
                                }
                                break;
                            }
                            case attr_t::post_ops_t::kind_t::RELU_DST: {
                                if (matmul_res > 0) {
                                    pointwise_trh[n + m * N] = std::abs(matmul_res * trh) + pointwise_trh[n + m * N];
                                } else {
                                    pointwise_trh[n + m * N] = std::abs(alpha * matmul_res * trh) + std::abs(alpha) * pointwise_trh[n + m * N];
                                }
                                break;
                            }
                            case attr_t::post_ops_t::kind_t::SRELU: {
                                pointwise_trh[n + m * N] = std::abs(1. / alpha * std::log(1. + std::exp(alpha * matmul_res)) * trh) + 1. / (1. + std::exp(-alpha * matmul_res)) * pointwise_trh[n + m * N];
                                break;
                            }
                            case attr_t::post_ops_t::kind_t::SQRT: {
                                if (std::abs(matmul_res) < 2e-5f) {
                                    pointwise_trh[n + m * N] = -1;
                                } else {
                                    pointwise_trh[n + m * N] = std::sqrt(std::abs(matmul_res)) * trh + 0.5 / std::sqrt(std::abs(matmul_res)) * pointwise_trh[n + m * N];
                                }
                                break;
                            }
                            case attr_t::post_ops_t::kind_t::SQRT_DST: {
                                if (std::abs(matmul_res) < 2e-5f) {
                                    pointwise_trh[n + m * N] = -1;
                                } else {
                                    pointwise_trh[n + m * N] = std::sqrt(std::abs(matmul_res)) * trh + 0.5 / std::sqrt(std::abs(matmul_res)) * pointwise_trh[n + m * N];
                                }
                                break;
                            }
                            case attr_t::post_ops_t::kind_t::SQUARE: {
                                pointwise_trh[n + m * N] = std::abs(matmul_res * matmul_res * trh) + 2. * std::abs(matmul_res) * pointwise_trh[n + m * N];
                                break;
                            }
                            case attr_t::post_ops_t::kind_t::TANH: {
                                pointwise_trh[n + m * N] = std::abs(std::tanh(matmul_res) * trh) + 1. / (std::cosh(matmul_res) * std::cosh(matmul_res)) * pointwise_trh[n + m * N];
                                break;
                            }
                            case attr_t::post_ops_t::kind_t::TANH_DST: {
                                pointwise_trh[n + m * N] = std::abs(std::tanh(matmul_res) * trh) + 1. / (std::cosh(matmul_res) * std::cosh(matmul_res)) * pointwise_trh[n + m * N];
                                break;
                            }
                            case attr_t::post_ops_t::kind_t::GELU_TANH: {
                                pointwise_trh[n + m * N] = std::abs(0.5 * matmul_res * (1. + std::tanh(std::sqrt(2. / 3.1415) * (matmul_res + 0.044715 * matmul_res * matmul_res * matmul_res))) * trh) +
                                    std::abs(0.5 * (1. + std::tanh(std::sqrt(2. / 3.1415) * (matmul_res + 0.044715 * matmul_res * matmul_res * matmul_res))) +
                                        0.5 * matmul_res * std::sqrt(2. / 3.1415) * (1. + 3. * 0.044715 * matmul_res * matmul_res) / 
                                            (std::cosh(std::sqrt(2. / 3.1415) * (matmul_res + 0.044715 * matmul_res * matmul_res * matmul_res)) * std::cosh(std::sqrt(2. / 3.1415) * (matmul_res + 0.044715 * matmul_res * matmul_res * matmul_res))))
                                    * pointwise_trh[n + m * N];
                                break;
                            }
                            case attr_t::post_ops_t::kind_t::GELU_ERF: {
                                pointwise_trh[n + m * N] = std::abs(0.5 * matmul_res * (1. + std::erf(matmul_res / std::sqrt(2.))) * trh) +
                                    std::abs(0.5 * (1. + std::erf(matmul_res / std::sqrt(2.))) + 0.5 * matmul_res * (std::exp(- (matmul_res * matmul_res) / 2.) / std::sqrt(2.))) * pointwise_trh[n + m * N];
                                break;
                            }
                            case attr_t::post_ops_t::kind_t::MISH: {
                                pointwise_trh[n + m * N] = std::abs(matmul_res * std::tanh(std::log(1. + std::exp(matmul_res))) * trh) +
                                    std::abs(std::tanh(std::log(1. + std::exp(matmul_res))) + matmul_res / (std::cosh(std::log(1. + std::exp(matmul_res))) * std::cosh(std::log(1. + std::exp(matmul_res))) * (1. + std::exp(-matmul_res)))) * pointwise_trh[n + m * N];
                                break;
                            }
                            case attr_t::post_ops_t::kind_t::SWISH: {
                                pointwise_trh[n + m * N] = std::abs((matmul_res) / (1. + std::exp(-alpha * matmul_res)) * trh) +
                                    std::abs(1. / (1. + std::exp(-alpha * matmul_res)) - alpha * matmul_res * std::exp(- alpha * matmul_res) / ((1. + std::exp(- alpha * matmul_res)) * (1. + std::exp(- alpha * matmul_res)))) * pointwise_trh[n + m * N];
                                break;
                            }
                            default: {
                                throw std::logic_error{"Unsuppoorted testing configuration"};
                            }
                    }
                }
            }
        }
    }
}

void compute_pointwise_trh_bwd_d(const prb_t *prb, const args_t &args, std::vector<float>& pointwise_trh) {
    const dnn_mem_t &wei_m = args.find(DNNL_ARG_WEIGHTS);
    const dnn_mem_t &diff_dst_m = args.find(DNNL_ARG_DIFF_DST);
 
    int64_t M = prb->mb;
    int64_t N = prb->ic * prb->id * prb->ih * prb->iw;
    int64_t K = prb->oc;
    pointwise_trh.resize(M * N);
    float ulp = BFLOAT16_ULP;
    if (prb->dt[0] == dnnl_f16 && prb->dt[1] == dnnl_f16) ulp = FLOAT16_ULP;
    if (prb->dt[0] == dnnl_f32 && prb->dt[1] == dnnl_f32) ulp = FLOAT32_ULP;

    #pragma omp parallel for
    for (int64_t m = 0; m < M; m++) {
        for (int64_t n = 0; n < N; n++) {
            float P_mn = 0;
            for (int64_t k = 0; k < K; k++) {
                float dd = ((float *)diff_dst_m)[k + m * K];
                float wei = ((float *)wei_m)[n + k * N];
                P_mn = P_mn < std::abs(dd * wei) ? std::abs(dd * wei) : P_mn;
            }
            pointwise_trh[m * N + n] = K * P_mn * ulp;
        }
    }
}

void compute_pointwise_trh_bwd_w(const prb_t *prb, const args_t &args, std::vector<float>& pointwise_trh) {
    const dnn_mem_t &src_m = args.find(DNNL_ARG_SRC);
    const dnn_mem_t &diff_dst_m = args.find(DNNL_ARG_DIFF_DST);
    int64_t M = prb->oc;
    int64_t N = prb->ic * prb->id * prb->ih * prb->iw;
    int64_t K = prb->mb;
    pointwise_trh.resize(M * N);
    float ulp = BFLOAT16_ULP;
    if (prb->dt[0] == dnnl_f16 && prb->dt[1] == dnnl_f16) ulp = FLOAT16_ULP;
    if (prb->dt[0] == dnnl_f32 && prb->dt[1] == dnnl_f32) ulp = FLOAT32_ULP;

    #pragma omp parallel for
    for (int64_t m = 0; m < M; m++) {
        for (int64_t n = 0; n < N; n++) {
            float P_mn = 0;
            for (int64_t k = 0; k < K; k++) {
                float dd = ((float *)diff_dst_m)[m + k * M];
                float s = ((float *)src_m)[n + k * N];
                P_mn = P_mn < std::abs(dd * s) ? std::abs(dd * s) : P_mn;
            }
            pointwise_trh[m * N + n] = K * P_mn * ulp;
        }
    }
}

void compute_pointwise_trh(
        const prb_t *prb, const args_t &args, std::vector<float>& pointwise_trh) {
    if (prb->dir & FLAG_FWD)
        compute_pointwise_trh_fwd(prb, args, pointwise_trh);
    if (prb->dir == BWD_D)
        compute_pointwise_trh_bwd_d(prb, args, pointwise_trh);
    else if (prb->dir & FLAG_BWD && prb->dir & FLAG_WEI)
        compute_pointwise_trh_bwd_w(prb, args, pointwise_trh);
}

void setup_cmp(compare::compare_t &cmp, const prb_t *prb, data_kind_t kind,
        const args_t &ref_args) {
    const char* enable_pointwise_trh = std::getenv("ENABGLE_POINTWISE_TRH");
    // Using ENABGLE_POINTWISE_TRH=off can disable the ponitwise thresholding
    if (!((enable_pointwise_trh != nullptr) && (std::string(enable_pointwise_trh) == "off")) &&
        (((prb->dt[0] == dnnl_f16 && prb->dt[1] == dnnl_f16 && prb->dt[2] == dnnl_f16) ||
        (prb->dt[0] == dnnl_bf16 && prb->dt[1] == dnnl_bf16 && prb->dt[2] == dnnl_bf16)) ||
         !prb->attr.scales.get(DNNL_ARG_SRC).is_def() || !prb->attr.scales.get(DNNL_ARG_WEIGHTS).is_def() || (prb->attr.post_ops.len() > 0))) {
        cmp.set_has_pointwise_trh(true);
        compute_pointwise_trh(prb, ref_args, cmp.pointwise_trh);
    }
    float trh = prb->dt[2] == dnnl_f32 ? 1e-6f : epsilon_dt(prb->dt[2]);
    if (prb->attr.post_ops.len() > 0) {
        if (prb->attr.post_ops.entry[0].is_eltwise_kind()) {
            trh = trh;
        }
    }
    cmp.set_threshold(trh);
}


std::vector<int> supported_exec_args(dir_t dir) {
    static const std::vector<int> exec_fwd_args = {
            DNNL_ARG_SRC,
            DNNL_ARG_WEIGHTS,
            DNNL_ARG_BIAS,
            DNNL_ARG_DST,
    };
    static const std::vector<int> exec_bwd_d_args = {
            DNNL_ARG_DIFF_SRC,
            DNNL_ARG_WEIGHTS,
            DNNL_ARG_BIAS,
            DNNL_ARG_DIFF_DST,
    };
    static const std::vector<int> exec_bwd_w_args = {
            DNNL_ARG_SRC,
            DNNL_ARG_DIFF_WEIGHTS,
            DNNL_ARG_DIFF_BIAS,
            DNNL_ARG_DIFF_DST,
    };
    return (dir & FLAG_FWD)    ? exec_fwd_args
            : (dir & FLAG_WEI) ? exec_bwd_w_args
                               : exec_bwd_d_args;
};

int init_ref_memory_args(dnn_mem_map_t &ref_mem_map, dnn_mem_map_t &mem_map,
        dnnl_primitive_t prim, const prb_t *prb, res_t *res, dir_t dir,
        dnnl_primitive_t prim_ref) {
    if (has_bench_mode_modifier(mode_modifier_t::no_host_memory)) return OK;

    const auto &ref_engine = get_cpu_engine();

    // Move cfg out of filling since its creation is not free.
    cfg_t cfg(prb, {SRC, WEI, BIA, DST});

    for (auto &entry : mem_map) {
        const int exec_arg = entry.first;
        // The function targets regular exec_args that are positive.
        // Negative args are used by bitwise and are broken in the `default`
        // branch due to `&` always returns `true`.
        if (exec_arg <= 0) continue;

        auto &mem = entry.second; // `mem` is modified by filler (reorder).

        // Scratchpad memory relates to a primitive. If reference needs it,
        // use switch below to define a memory desc for it.
        if (exec_arg != DNNL_ARG_SCRATCHPAD) {
            ref_mem_map.emplace(exec_arg,
                    dnn_mem_t(mem.md_, dnnl_f32, tag::abx, ref_engine));
        }
        auto &ref_mem = ref_mem_map[exec_arg];

        switch (exec_arg) {
            case DNNL_ARG_SRC:
                SAFE(fill_data(SRC, prb, cfg, mem, ref_mem, res), WARN);
                break;
            case DNNL_ARG_WEIGHTS:
                SAFE(fill_data(WEI, prb, cfg, mem, ref_mem, res), WARN);
                break;
            case DNNL_ARG_BIAS:
                SAFE(fill_data(BIA, prb, cfg, mem, ref_mem, res), WARN);
                break;
            case DNNL_ARG_DST:
                if (prb->attr.post_ops.find(attr_t::post_ops_t::kind_t::SUM)
                        >= 0) {
                    SAFE(fill_data(DST, prb, cfg, mem, ref_mem, res), WARN);
                    // Bitwise mode for sum requires a copy due to data for
                    // post-op will be overwritten and it must be refreshed.
                    if (has_bench_mode_bit(mode_bit_t::bitwise)) {
                        SAFE(mem_map.at(-exec_arg).reorder(ref_mem), WARN);
                    }
                }
                break;
            case DNNL_ARG_DIFF_DST:
                SAFE(fill_data(DST, prb, cfg, mem, ref_mem, res), WARN);
                break;
            default:
                SAFE(init_ref_memory_args_default_case(
                             exec_arg, mem, ref_mem, prb->attr, res),
                        WARN);
                break;
        }

        update_ref_mem_map_from_prim(prim_ref, mem, ref_mem_map, exec_arg,
                cfg.get_swapped_dt(exec_arg2data_kind(exec_arg)));

        // Don't keep reference memory if it is not used further.
        if (!has_bench_mode_bit(mode_bit_t::corr)) ref_mem_map.clear();
    }

    return OK;
}

std::vector<data_kind_t> get_kinds_to_check(const prb_t *prb) {
    std::vector<data_kind_t> check_kinds;
    if (prb->dir & FLAG_FWD) {
        check_kinds = {DST};
    } else if (prb->dir == BWD_D) {
        check_kinds = {SRC};
    } else if (prb->dir & FLAG_BWD && prb->dir & FLAG_WEI) {
        check_kinds = {WEI};
        if (prb->dir & FLAG_BIA) check_kinds.push_back(BIA);
    } else {
        assert(!"unexpected!");
        SAFE_V(FAIL);
    }
    assert(!check_kinds.empty());
    return check_kinds;
}

int createit(std::vector<benchdnn_dnnl_wrapper_t<dnnl_primitive_t>> &v_prim,
        const prb_t *prb, res_t *res) {
    v_prim.resize(2); // regular + cpu_ref
    SAFE(init_prim(prb->ctx_init, v_prim[0], init_pd, prb, res), WARN);
    // Use CPU prim as the reference in GPU testing to reduce testing time.
    SAFE(init_prim_ref(v_prim[1], prb, res), WARN);
    return OK;
}

int check_cacheit(
        std::vector<benchdnn_dnnl_wrapper_t<dnnl_primitive_t>> &v_prim,
        const prb_t *prb, res_t *res) {
    SAFE(check_caches(v_prim[0], prb, res), WARN);
    // Don't check caches for CPU prim as the reference.
    return OK;
}

int doit(const std::vector<benchdnn_dnnl_wrapper_t<dnnl_primitive_t>> &v_prim,
        const prb_t *prb, res_t *res) {
    const auto &prim = v_prim[0];
    const auto &prim_ref = v_prim[1];

    dnn_mem_map_t mem_map, ref_mem_map;
    init_memory_args<prb_t>(mem_map, prb, prim, supported_exec_args(prb->dir));
    TIME_FILL(SAFE(init_ref_memory_args(ref_mem_map, mem_map, prim, prb, res,
                           prb->dir, prim_ref),
            WARN));

    args_t args(mem_map), ref_args(ref_mem_map);

    SAFE(execute_and_wait(prim, args, res), WARN);

    check_correctness(prb, get_kinds_to_check(prb), args, ref_args, setup_cmp,
            res, prim_ref);
    SAFE(check_bitwise(prim, get_kinds_to_check(prb), args, prb->inplace, res),
            WARN);

    return measure_perf(prb->ctx_exe, res, prim, args);
}

} // namespace ip
