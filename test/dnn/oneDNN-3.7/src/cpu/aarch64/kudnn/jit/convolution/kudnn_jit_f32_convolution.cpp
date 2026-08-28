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

#include "common/c_types_map.hpp"
#include "common/dnnl_thread.hpp"
#include "common/type_helpers.hpp"
#include "common/utils.hpp"

#include "kudnn_jit_f32_convolution.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace aarch64 {

using namespace dnnl::impl::status;
using namespace dnnl::impl::memory_tracking::names;
using namespace dnnl::impl::utils;

using namespace nstl;

using kdnn_jit_conv_ker_t = void (*)(kdnn_jit_conv_call_s *);

#define kdnn_PIPELINE(field) \
    do { \
        p.field = p.field##_prf; \
        p.field##_prf = field; \
    } while (0)

inline void kdnn_jit_conv_ker_pipeline(kdnn_jit_conv_ker_t ker, kdnn_jit_conv_call_s &p,
        const void *src, const void *dst, const void *filt, const void *bias,
        int channel, int kh_padding, int reduce_work, int load_work) {
    kdnn_PIPELINE(src);
    kdnn_PIPELINE(dst);
    kdnn_PIPELINE(filt);
    kdnn_PIPELINE(bias);
    kdnn_PIPELINE(channel);
    // non-positive value of kh_padding is allowed, in this case kernel must
    // skip computation part and initialize output by zeroes
    kdnn_PIPELINE(kh_padding);
    kdnn_PIPELINE(reduce_work);
    kdnn_PIPELINE(load_work);

    if (p.src) ker(&p);
}
// The special case for the driver with ow-parallelization (FWD)
inline void kdnn_jit_conv_ker_pipeline_ow_thr(kdnn_jit_conv_ker_t ker, kdnn_jit_conv_call_s &p,
        const void *src, const void *dst, const void *filt, const void *bias,
        int channel, int kh_padding, int owb, int reduce_work, int load_work,
        int flags) {
    kdnn_PIPELINE(owb);
    kdnn_PIPELINE(flags);
    kdnn_jit_conv_ker_pipeline(ker, p, src, dst, filt, bias, channel, kh_padding,
            reduce_work, load_work);
}
// The special case for the driver with iw-parallelization (BWD)
inline void kdnn_jit_conv_ker_pipeline_iw_thr(kdnn_jit_conv_ker_t ker, kdnn_jit_conv_call_s &p,
        const void *src, const void *dst, const void *filt, const void *bias,
        int channel, int kh_padding, int iwb, int reduce_work, int load_work) {
    kdnn_PIPELINE(iwb);

    kdnn_jit_conv_ker_pipeline(ker, p, src, dst, filt, bias, channel, kh_padding,
            reduce_work, load_work);
}

inline void kdnn_jit_sve_conv_3d_ker_pipeline(kdnn_jit_conv_ker_t ker, kdnn_jit_conv_call_s &p,
        const void *src, const void *dst, const void *filt, const void *bias,
        int channel, int kh_padding, int kd_padding, int reduce_work,
        int load_work) {
    kdnn_PIPELINE(src);
    kdnn_PIPELINE(dst);
    kdnn_PIPELINE(filt);
    kdnn_PIPELINE(bias);
    kdnn_PIPELINE(channel);
    // non-positive value of both kd_padding and kh_padding is allowed, in this
    // case kernel must skip computation part and initialize output by zeroes
    kdnn_PIPELINE(kh_padding);
    kdnn_PIPELINE(kd_padding);
    kdnn_PIPELINE(reduce_work);
    kdnn_PIPELINE(load_work);

    if (p.src) ker(&p);
}
// The special case for the driver with ow-parallelization (FWD)
// TODO: implement it for BWD_D and BWD_W too
inline void kdnn_jit_sve_conv_3d_ker_pipeline_ow_thr(kdnn_jit_conv_ker_t ker,
        kdnn_jit_conv_call_s &p, const void *src, const void *dst, const void *filt,
        const void *bias, int channel, int kh_padding, int kd_padding, int owb,
        int reduce_work, int load_work, int flags) {
    kdnn_PIPELINE(owb);
    kdnn_PIPELINE(flags);

    kdnn_jit_sve_conv_3d_ker_pipeline(ker, p, src, dst, filt, bias, channel,
            kh_padding, kd_padding, reduce_work, load_work);
}

#define kdnn_wht_blk_off(d, g, ...) \
    (pd()->with_groups() ? (d).blk_off((g), __VA_ARGS__) \
                         : (d).blk_off(__VA_ARGS__))

template <cpu_isa_t isa>
void kdnn_jit_sve_convolution_fwd_t<isa>::prepare_padded_bias(
        const float *&bias,
        const memory_tracking::grantor_t &scratchpad) const {
    if (!pd()->wants_padded_bias()) return;

    auto padded_bias
            = scratchpad.template get<float>(key_conv_padded_bias);
    utils::array_copy(padded_bias, bias, pd()->jcp_.oc_without_padding);
    utils::array_set(padded_bias + pd()->jcp_.oc_without_padding, (float)0,
            pd()->jcp_.oc - pd()->jcp_.oc_without_padding);
    bias = padded_bias;
}

template <cpu_isa_t isa>
void kdnn_jit_sve_convolution_fwd_t<isa>::execute_forward_1d(const exec_ctx_t &ctx) const {
    auto src = CTX_IN_MEM(const float *, DNNL_ARG_SRC);
    auto weights = CTX_IN_MEM(const float *, DNNL_ARG_WEIGHTS);
    auto bias = CTX_IN_MEM(const float *, DNNL_ARG_BIAS);
    auto dst = CTX_OUT_MEM(float *, DNNL_ARG_DST);

    prepare_padded_bias(bias, ctx.get_scratchpad_grantor());

    const memory_desc_wrapper src_d(pd()->src_md());
    const memory_desc_wrapper dst_d(pd()->dst_md());
    const memory_desc_wrapper weights_d(pd()->weights_md(0));

    const auto &jcp = pd()->jcp_;
    const kdnn_jit_conv_ker_t jit_ker = (decltype(jit_ker))kernel_->jit_ker();
    assert(jcp.nb_oc % jcp.nb_oc_blocking == 0);

    int oc_chunks = jcp.nb_oc / jcp.nb_oc_blocking;
    int g_blocking = 1;
    int nb_groups = jcp.ngroups / g_blocking;
    int work_amount = jcp.mb * nb_groups * oc_chunks * jcp.nb_ow;
    int nthr = jcp.aligned_threads;

    parallel(nthr, [&](const int ithr, const int nthr) {
        int start {0}, end {0}, start_copy;
        balance211(work_amount, nthr, ithr, start, end);
        start_copy = start;

        auto par_conv = kdnn_jit_conv_call_s();
        size_t src_c_stride = src_d.blk_off(0, 1);
        size_t wht_ic_stride = kdnn_wht_blk_off(weights_d, 0, 0, 1);

        for (int icb_l2 = 0; icb_l2 < jcp.nb_ic; icb_l2 += jcp.nb_ic_L2) {
            start = start_copy;
            int n {0}, gg {0}, occ {0}, owb {0};

            if (jcp.loop_order == kdnn_loop_cwgn) {
                int dummy {0};
                nd_iterator_init(start, occ, oc_chunks, owb, jcp.nb_ow, gg,
                        nb_groups, n, jcp.mb, dummy, 1);
            } else if (jcp.loop_order == kdnn_loop_gncw) {
                int dummy {0};
                nd_iterator_init(start, gg, nb_groups, n, jcp.mb, occ,
                        oc_chunks, owb, jcp.nb_ow, dummy, 1);
            } else if (jcp.loop_order == kdnn_loop_nhwcg) {
                nd_iterator_init(start, n, jcp.mb, owb, jcp.nb_ow, occ,
                        oc_chunks, gg, nb_groups);
            } else {
                assert(!"unsupported loop order");
            }

            while (start < end) {
                int ocb = occ * jcp.nb_oc_blocking;
                int g = gg * g_blocking;
                int g_ocb = g * jcp.nb_oc + ocb;
                int g_icb = g * jcp.nb_ic * jcp.nonblk_group_off;

                int ow_s = owb * jcp.ow_block;
                int iw_s = ow_s * jcp.stride_w;
                const bool is_dst_layout_nxc = jcp.dst_tag == format_tag::nwc;
                const int oc_off_idx = is_dst_layout_nxc
                        ? g * jcp.oc + ocb * jcp.oc_block
                        : g_ocb;
                auto dst_w = dst + dst_d.blk_off(n, oc_off_idx, ow_s);
                const bool is_src_layout_nxc = jcp.src_tag == format_tag::nwc;
                const int ic_off_idx = is_src_layout_nxc
                        ? g * jcp.ic + icb_l2 * jcp.ic_block
                        : g_icb + icb_l2;
                auto src_w = src + src_d.blk_off(n, ic_off_idx, iw_s);
                auto wht_w = weights + kdnn_wht_blk_off(weights_d, g, ocb, icb_l2);
                auto bias_w = bias ? bias
                                + oc_off_idx
                                        * (is_dst_layout_nxc ? 1 : jcp.oc_block)
                                   : nullptr;

                int icb_step = is_src_layout_nxc ? jcp.nb_ic_L2 : 1;
                int icb_end = min(jcp.nb_ic, icb_l2 + jcp.nb_ic_L2);
                const int oc_work = utils::this_block_size(ocb * jcp.oc_block,
                        jcp.oc, jcp.nb_oc_blocking * jcp.oc_block);
                int ic_work = icb_step * jcp.ic_block;
                for (int icb = icb_l2; icb < icb_end; icb += icb_step) {
                    int curr_nb_ic = nstl::min(icb_step, icb_end - icb);
                    int flags = 0;
                    if (icb == 0) flags |= kdnn_FLAG_IC_FIRST;
                    if (icb + curr_nb_ic >= jcp.nb_ic) {
                        flags |= kdnn_FLAG_IC_LAST;
                        ic_work = utils::this_block_size(icb * jcp.ic_block,
                                jcp.ic, icb_step * jcp.ic_block);
                    }
                    kdnn_jit_conv_ker_pipeline_ow_thr(jit_ker, par_conv, src_w,
                            dst_w, wht_w, bias_w, icb, 1, owb, ic_work, oc_work,
                            flags);

                    src_w += src_c_stride;
                    wht_w += wht_ic_stride;
                }
                if (jcp.loop_order == kdnn_loop_cwgn) {
                    int dummy {0};
                    nd_iterator_jump(start, end, occ, oc_chunks, owb, jcp.nb_ow,
                            gg, nb_groups, n, jcp.mb, dummy, 1);
                } else if (jcp.loop_order == kdnn_loop_gncw) {
                    int dummy {0};
                    nd_iterator_jump(start, end, gg, nb_groups, n, jcp.mb, occ,
                            oc_chunks, owb, jcp.nb_ow, dummy, 1);
                } else if (jcp.loop_order == kdnn_loop_nhwcg) {
                    ++start;
                    nd_iterator_step(n, jcp.mb, owb, jcp.nb_ow, occ, oc_chunks,
                            gg, nb_groups);
                } else {
                    assert(!"unsupported loop order");
                }
            }
        }

        // This call is required only to finalize pipeline with paramaters set
        // on the last iteration of loop above. Only valid pointers make sense
        // here as call parameters to avoid execution of prefetch instructions
        // with nullptr, other parameters are not used in real jit call here
        kdnn_jit_conv_ker_pipeline_ow_thr(
                jit_ker, par_conv, src, dst, weights, bias, 0, 0, 0, 0, 0, 0);
    });
}

template <cpu_isa_t isa>
void kdnn_jit_sve_convolution_fwd_t<isa>::execute_forward_2d(const exec_ctx_t &ctx) const {
    auto src = CTX_IN_MEM(const float *, DNNL_ARG_SRC);
    auto weights = CTX_IN_MEM(const float *, DNNL_ARG_WEIGHTS);
    auto bias = CTX_IN_MEM(const float *, DNNL_ARG_BIAS);
    auto dst = CTX_OUT_MEM(float *, DNNL_ARG_DST);

    prepare_padded_bias(bias, ctx.get_scratchpad_grantor());

    const memory_desc_wrapper src_d(pd()->src_md());
    const memory_desc_wrapper dst_d(pd()->dst_md());
    const memory_desc_wrapper weights_d(pd()->weights_md(0));

    const auto &jcp = pd()->jcp_;
    const kdnn_jit_conv_ker_t jit_ker = (decltype(jit_ker))kernel_->jit_ker();
    assert(jcp.nb_oc % jcp.nb_oc_blocking == 0);

    int oc_chunks = jcp.nb_oc / jcp.nb_oc_blocking;
    int g_blocking = 1;
    int nb_groups = jcp.ngroups / g_blocking;
    int work_amount = jcp.mb * nb_groups * oc_chunks * jcp.oh * jcp.nb_ow;
    int nthr = jcp.aligned_threads;

    parallel(nthr, [&](const int ithr, const int nthr) {
        int start {0}, end {0}, start_copy;
        balance211(work_amount, nthr, ithr, start, end);
        start_copy = start;

        auto par_conv = kdnn_jit_conv_call_s();
        size_t src_h_stride = src_d.blk_off(0, 0, 1);
        size_t src_c_stride = src_d.blk_off(0, 1);
        size_t dst_h_stride = dst_d.blk_off(0, 0, 1);
        size_t wht_h_stride = kdnn_wht_blk_off(weights_d, 0, 0, 0, 1);
        size_t wht_ic_stride = kdnn_wht_blk_off(weights_d, 0, 0, 1);

        for (int icb_l2 = 0; icb_l2 < jcp.nb_ic; icb_l2 += jcp.nb_ic_L2) {
            start = start_copy;
            int n {0}, gg {0}, occ {0}, oh_s {0}, owb {0};

            if (jcp.loop_order == kdnn_loop_cwgn)
                nd_iterator_init(start, occ, oc_chunks, owb, jcp.nb_ow, gg,
                        nb_groups, n, jcp.mb, oh_s, jcp.oh);
            else if (jcp.loop_order == kdnn_loop_gncw)
                nd_iterator_init(start, gg, nb_groups, n, jcp.mb, occ,
                        oc_chunks, owb, jcp.nb_ow, oh_s, jcp.oh);
            else if (jcp.loop_order == kdnn_loop_nhwcg)
                nd_iterator_init(start, n, jcp.mb, oh_s, jcp.oh, owb, jcp.nb_ow,
                        occ, oc_chunks, gg, nb_groups);
            else
                assert(!"unsupported loop order");

            while (start < end) {
                int ocb = occ * jcp.nb_oc_blocking;
                int g = gg * g_blocking;
                int g_ocb = g * jcp.nb_oc + ocb;
                int g_icb = g * jcp.nb_ic * jcp.nonblk_group_off;

                int work_rem = end - start;

                int ow_s = owb * jcp.ow_block;
                int iw_s = ow_s * jcp.stride_w;
                int oh_e = oh_s + work_rem > jcp.oh ? jcp.oh : oh_s + work_rem;
                if (jcp.loop_order == kdnn_loop_nhwcg)
                    oh_e = oh_s + 1; //step instead

                for (int oh_b = oh_s; oh_b < oh_e; oh_b += jcp.h_blocking) {
                    int ih_b = -jcp.t_pad + oh_b * jcp.stride_h;
                    const bool is_dst_layout_nxc
                            = jcp.dst_tag == format_tag::nhwc;
                    const int oc_off_idx = is_dst_layout_nxc
                            ? g * jcp.oc + ocb * jcp.oc_block
                            : g_ocb;
                    auto dst_w = dst + dst_d.blk_off(n, oc_off_idx, oh_b, ow_s);
                    const bool is_src_layout_nxc
                            = jcp.src_tag == format_tag::nhwc;
                    const int ic_off_idx = is_src_layout_nxc
                            ? g * jcp.ic + icb_l2 * jcp.ic_block
                            : g_icb + icb_l2;
                    auto src_w = src + src_d.blk_off(n, ic_off_idx, ih_b, iw_s);
                    auto wht_w
                            = weights + kdnn_wht_blk_off(weights_d, g, ocb, icb_l2);

                    int icb_step = is_src_layout_nxc ? jcp.nb_ic_L2 : 1;
                    int icb_end = min(jcp.nb_ic, icb_l2 + jcp.nb_ic_L2);
                    auto bias_w = bias ? bias
                                    + oc_off_idx
                                            * (is_dst_layout_nxc ? 1
                                                                 : jcp.oc_block)
                                       : nullptr;
                    const int oc_work
                            = utils::this_block_size(ocb * jcp.oc_block, jcp.oc,
                                    jcp.nb_oc_blocking * jcp.oc_block);
                    int ic_work = icb_step * jcp.ic_block;
                    for (int icb = icb_l2; icb < icb_end; icb += icb_step) {
                        int curr_nb_ic = nstl::min(icb_step, icb_end - icb);
                        int flags = 0;
                        if (icb == 0) flags |= kdnn_FLAG_IC_FIRST;
                        if (icb + curr_nb_ic >= jcp.nb_ic) {
                            flags |= kdnn_FLAG_IC_LAST;
                            ic_work = utils::this_block_size(icb * jcp.ic_block,
                                    jcp.ic, icb_step * jcp.ic_block);
                        }
                        auto src_c = src_w;
                        auto dst_c = dst_w;
                        for (int oj = oh_b, ij = ih_b;
                                oj < min(oh_e, oh_b + jcp.h_blocking);
                                ++oj, ij += jcp.stride_h) {
                            int dilate_h = jcp.dilate_h + 1;
                            int i_t_overflow = div_up(max(0, -ij), dilate_h);
                            int i_b_overflow = div_up(
                                    max(0,
                                            ij - jcp.ih
                                                    + (jcp.kh - 1) * dilate_h
                                                    + 1),
                                    dilate_h);
                            int kh_padding = nstl::max(
                                    0, jcp.kh - i_t_overflow - i_b_overflow);

                            auto aux_src = src_c
                                    + i_t_overflow * dilate_h * src_h_stride;
                            auto aux_wht = wht_w + i_t_overflow * wht_h_stride;

                            kdnn_jit_conv_ker_pipeline_ow_thr(jit_ker, par_conv,
                                    aux_src, dst_c, aux_wht, bias_w, icb,
                                    kh_padding, owb, ic_work, oc_work, flags);

                            src_c += src_h_stride * jcp.stride_h;
                            dst_c += dst_h_stride;
                        }
                        src_w += src_c_stride;
                        wht_w += wht_ic_stride;
                    }
                }

                if (jcp.loop_order == kdnn_loop_cwgn)
                    nd_iterator_jump(start, end, occ, oc_chunks, owb, jcp.nb_ow,
                            gg, nb_groups, n, jcp.mb, oh_s, jcp.oh);
                else if (jcp.loop_order == kdnn_loop_gncw)
                    nd_iterator_jump(start, end, gg, nb_groups, n, jcp.mb, occ,
                            oc_chunks, owb, jcp.nb_ow, oh_s, jcp.oh);
                else if (jcp.loop_order == kdnn_loop_nhwcg) {
                    ++start;
                    nd_iterator_step(n, jcp.mb, oh_s, jcp.oh, owb, jcp.nb_ow,
                            occ, oc_chunks, gg, nb_groups);
                } else
                    assert(!"unsupported loop order");
            }
        }

        // This call is required only to finalize pipeline with paramaters set
        // on the last iteration of loop above. Only valid pointers make sense
        // here as call parameters to avoid execution of prefetch instructions
        // with nullptr, other parameters are not used in real jit call here
        kdnn_jit_conv_ker_pipeline_ow_thr(
                jit_ker, par_conv, src, dst, weights, bias, 0, 0, 0, 0, 0, 0);
    });
}

template <cpu_isa_t isa>
void kdnn_jit_sve_convolution_fwd_t<isa>::execute_forward_3d(const exec_ctx_t &ctx) const {
    auto src = CTX_IN_MEM(const float *, DNNL_ARG_SRC);
    auto weights = CTX_IN_MEM(const float *, DNNL_ARG_WEIGHTS);
    auto bias = CTX_IN_MEM(const float *, DNNL_ARG_BIAS);
    auto dst = CTX_OUT_MEM(float *, DNNL_ARG_DST);

    prepare_padded_bias(bias, ctx.get_scratchpad_grantor());

    const memory_desc_wrapper src_d(pd()->src_md());
    const memory_desc_wrapper dst_d(pd()->dst_md());
    const memory_desc_wrapper weights_d(pd()->weights_md(0));

    const auto &jcp = pd()->jcp_;
    const kdnn_jit_conv_ker_t jit_ker = (decltype(jit_ker))kernel_->jit_ker();
    assert(jcp.nb_oc % jcp.nb_oc_blocking == 0);

    int oc_chunks = jcp.nb_oc / jcp.nb_oc_blocking;
    int g_blocking = 1;
    int nb_groups = jcp.ngroups / g_blocking;
    int work_amount
            = jcp.mb * nb_groups * oc_chunks * jcp.od * jcp.oh * jcp.nb_ow;
    int nthr = jcp.nthr;

    parallel(nthr, [&](const int ithr, const int nthr) {
        int start {0}, end {0}, start_copy;
        balance211(work_amount, nthr, ithr, start, end);
        start_copy = start;

        auto par_conv = kdnn_jit_conv_call_s();
        size_t src_d_stride = src_d.blk_off(0, 0, 1);
        size_t src_h_stride = src_d.blk_off(0, 0, 0, 1);
        size_t src_c_stride = src_d.blk_off(0, 1);
        size_t dst_h_stride = dst_d.blk_off(0, 0, 0, 1);
        size_t wht_d_stride = kdnn_wht_blk_off(weights_d, 0, 0, 0, 1);
        size_t wht_h_stride = kdnn_wht_blk_off(weights_d, 0, 0, 0, 0, 1);
        size_t wht_ic_stride = kdnn_wht_blk_off(weights_d, 0, 0, 1);

        for (int icb_l2 = 0; icb_l2 < jcp.nb_ic; icb_l2 += jcp.nb_ic_L2) {
            start = start_copy;
            int n {0}, gg {0}, occ {0}, oh_s {0}, od_s {0}, owb {0};

            if (jcp.loop_order == kdnn_loop_cwgn)
                nd_iterator_init(start, occ, oc_chunks, owb, jcp.nb_ow, gg,
                        nb_groups, n, jcp.mb, od_s, jcp.od, oh_s, jcp.oh);
            else if (jcp.loop_order == kdnn_loop_gncw)
                nd_iterator_init(start, gg, nb_groups, n, jcp.mb, occ,
                        oc_chunks, owb, jcp.nb_ow, od_s, jcp.od, oh_s, jcp.oh);
            else if (jcp.loop_order == kdnn_loop_nhwcg)
                nd_iterator_init(start, n, jcp.mb, od_s, jcp.od, oh_s, jcp.oh,
                        owb, jcp.nb_ow, occ, oc_chunks, gg, nb_groups);
            else
                assert(!"unsupported loop order");

            while (start < end) {
                int ocb = occ * jcp.nb_oc_blocking;
                int g = gg * g_blocking;
                int g_ocb = g * jcp.nb_oc + ocb;
                int g_icb = g * jcp.nb_ic * jcp.nonblk_group_off;

                int work_rem = end - start;
                int ih_s = -jcp.t_pad + oh_s * jcp.stride_h;
                int ow_s = owb * jcp.ow_block;
                int iw_s = ow_s * jcp.stride_w;
                int oh_e = oh_s + work_rem > jcp.oh ? jcp.oh : oh_s + work_rem;
                if (jcp.loop_order == kdnn_loop_nhwcg)
                    oh_e = oh_s + 1; //step instead

                int id_s = -jcp.f_pad + od_s * jcp.stride_d;

                int dilate_d = jcp.dilate_d + 1;
                int d_t_overflow = div_up(max(0, -id_s), dilate_d);
                int d_b_overflow = div_up(
                        max(0, id_s - jcp.id + (jcp.kd - 1) * dilate_d + 1),
                        dilate_d);
                int kd_padding
                        = nstl::max(0, jcp.kd - d_t_overflow - d_b_overflow);
                const bool is_dst_layout_nxc = jcp.dst_tag == format_tag::ndhwc;
                const int oc_off_idx = is_dst_layout_nxc
                        ? g * jcp.oc + ocb * jcp.oc_block
                        : g_ocb;
                auto dst_w
                        = dst + dst_d.blk_off(n, oc_off_idx, od_s, oh_s, ow_s);
                const bool is_src_layout_nxc = jcp.src_tag == format_tag::ndhwc;
                const int ic_off_idx = is_src_layout_nxc
                        ? g * jcp.ic + icb_l2 * jcp.ic_block
                        : g_icb + icb_l2;
                auto src_w = src
                        + src_d.blk_off(n, ic_off_idx, id_s, ih_s, iw_s)
                        + d_t_overflow * dilate_d * src_d_stride;
                auto wht_w = weights + kdnn_wht_blk_off(weights_d, g, ocb, icb_l2)
                        + d_t_overflow * wht_d_stride;
                auto bias_w = bias ? bias
                                + oc_off_idx
                                        * (is_dst_layout_nxc ? 1 : jcp.oc_block)
                                   : nullptr;

                const int icb_step = is_src_layout_nxc ? jcp.nb_ic_L2 : 1;
                int icb_end = min(jcp.nb_ic, icb_l2 + jcp.nb_ic_L2);
                const int oc_work = utils::this_block_size(ocb * jcp.oc_block,
                        jcp.oc, jcp.nb_oc_blocking * jcp.oc_block);
                int ic_work = icb_step * jcp.ic_block;
                for (int icb = icb_l2; icb < icb_end; icb += icb_step) {
                    int curr_nb_ic = nstl::min(icb_step, icb_end - icb);
                    int flags = 0;
                    if (icb == 0) flags |= kdnn_FLAG_IC_FIRST;
                    if (icb + curr_nb_ic >= jcp.nb_ic) {
                        flags |= kdnn_FLAG_IC_LAST;
                        ic_work = utils::this_block_size(icb * jcp.ic_block,
                                jcp.ic, icb_step * jcp.ic_block);
                    }
                    auto src_c = src_w;
                    auto dst_c = dst_w;
                    for (int oj = oh_s, ij = ih_s; oj < oh_e;
                            ++oj, ij += jcp.stride_h) {
                        int dilate_h = jcp.dilate_h + 1;
                        int i_t_overflow = div_up(max(0, -ij), dilate_h);
                        int i_b_overflow = div_up(
                                max(0,
                                        ij - jcp.ih + (jcp.kh - 1) * dilate_h
                                                + 1),
                                dilate_h);
                        int kh_padding = nstl::max(
                                0, jcp.kh - i_t_overflow - i_b_overflow);
                        kdnn_jit_sve_conv_3d_ker_pipeline_ow_thr(jit_ker, par_conv,
                                src_c + i_t_overflow * dilate_h * src_h_stride,
                                dst_c, wht_w + i_t_overflow * wht_h_stride,
                                bias_w, icb, kh_padding, kd_padding, owb,
                                ic_work, oc_work, flags);

                        src_c += src_h_stride * jcp.stride_h;
                        dst_c += dst_h_stride;
                    }
                    src_w += src_c_stride;
                    wht_w += wht_ic_stride;
                }

                if (jcp.loop_order == kdnn_loop_cwgn)
                    nd_iterator_jump(start, end, occ, oc_chunks, owb, jcp.nb_ow,
                            gg, nb_groups, n, jcp.mb, od_s, jcp.od, oh_s,
                            jcp.oh);
                else if (jcp.loop_order == kdnn_loop_gncw)
                    nd_iterator_jump(start, end, gg, nb_groups, n, jcp.mb, occ,
                            oc_chunks, owb, jcp.nb_ow, od_s, jcp.od, oh_s,
                            jcp.oh);
                else if (jcp.loop_order == kdnn_loop_nhwcg) {
                    ++start;
                    nd_iterator_step(n, jcp.mb, od_s, jcp.od, oh_s, jcp.oh, owb,
                            jcp.nb_ow, occ, oc_chunks, gg, nb_groups);
                } else
                    assert(!"unsupported loop order");
            }
        }

        // This call is required only to finalize pipeline with paramaters set
        // on the last iteration of loop above. Only valid pointers make sense
        // here as call parameters to avoid execution of prefetch instructions
        // with nullptr, other parameters are not used in real jit call here
        kdnn_jit_sve_conv_3d_ker_pipeline_ow_thr(jit_ker, par_conv, src, dst,
                weights, bias, 0, 0, 0, 0, 0, 0, 0);
    });
}

template struct kdnn_jit_sve_convolution_fwd_t<sve_256>;

template <cpu_isa_t isa>
void kdnn_jit_sve_convolution_bwd_data_t<isa>::execute_backward_data_1d(const exec_ctx_t &ctx) const {
    auto diff_dst = CTX_IN_MEM(const float *, DNNL_ARG_DIFF_DST);
    auto weights = CTX_IN_MEM(const float *, DNNL_ARG_WEIGHTS);
    auto diff_src = CTX_OUT_MEM(float *, DNNL_ARG_DIFF_SRC);

    const memory_desc_wrapper diff_dst_d(pd()->diff_dst_md());
    const memory_desc_wrapper diff_src_d(pd()->diff_src_md());
    const memory_desc_wrapper weights_d(pd()->weights_md(0));

    const auto &jcp = pd()->jcp_;
    const kdnn_jit_conv_ker_t jit_ker = (decltype(jit_ker))kernel_->jit_ker();

    int ic_chunks = jcp.nb_ic / jcp.nb_ic_blocking;
    int g_blocking = 1;
    int nb_groups = jcp.ngroups / g_blocking;
    int work_amount = nb_groups * jcp.mb * ic_chunks * jcp.nb_iw;
    int nthr = jcp.nthr;

    parallel(nthr, [&](const int ithr, const int nthr) {
        int start {0}, end {0}, start_copy;
        balance211(work_amount, nthr, ithr, start, end);
        start_copy = start;

        auto par_conv = kdnn_jit_conv_call_s();
        size_t diff_dst_c_stride = diff_dst_d.blk_off(0, 1);
        size_t wht_oc_stride = kdnn_wht_blk_off(weights_d, 0, 1);

        for (int ocb_l2 = 0; ocb_l2 < jcp.nb_oc; ocb_l2 += jcp.nb_oc_L2) {
            start = start_copy;
            int n {0}, gg {0}, icc {0}, iwb {0};
            if (jcp.loop_order == kdnn_loop_cwgn) {
                int dummy {0};
                nd_iterator_init(start, icc, ic_chunks, iwb, jcp.nb_iw, gg,
                        nb_groups, n, jcp.mb, dummy, 1);
            } else if (jcp.loop_order == kdnn_loop_gncw) {
                int dummy {0};
                nd_iterator_init(start, gg, nb_groups, n, jcp.mb, icc,
                        ic_chunks, iwb, jcp.nb_iw, dummy, 1);
            } else if (jcp.loop_order == kdnn_loop_nhwcg) {
                nd_iterator_init(start, n, jcp.mb, iwb, jcp.nb_iw, icc,
                        ic_chunks, gg, nb_groups);
            } else {
                assert(!"unsupported loop order");
            }

            while (start < end) {
                int icb = icc * jcp.nb_ic_blocking;
                int g = gg * g_blocking;
                int g_icb = g * jcp.nb_ic + icb;
                int g_ocb = g * jcp.nb_oc;
                int iw_s = iwb * jcp.iw_block;
                int ow_s = iw_s / jcp.stride_w;

                const bool is_dsrc_layout_nxc = jcp.src_tag == format_tag::nwc;
                const int ic_off_idx = is_dsrc_layout_nxc
                        ? g * jcp.ic + icb * jcp.ic_block
                        : g_icb;
                auto diff_src_w
                        = diff_src + diff_src_d.blk_off(n, ic_off_idx, iw_s);
                const bool is_ddst_layout_nxc = jcp.dst_tag == format_tag::nwc;
                const int oc_off_idx = is_ddst_layout_nxc
                        ? g * jcp.oc + ocb_l2 * jcp.oc_block
                        : g_ocb + ocb_l2;
                auto diff_dst_w
                        = diff_dst + diff_dst_d.blk_off(n, oc_off_idx, ow_s);
                auto wht_w = weights + kdnn_wht_blk_off(weights_d, g, ocb_l2, icb);

                int ocb_step = is_ddst_layout_nxc ? jcp.nb_oc_L2 : 1;
                int ocb_end = min(jcp.nb_oc, ocb_l2 + jcp.nb_oc_L2);
                const int load_work = utils::this_block_size(icb * jcp.ic_block,
                        jcp.ic, jcp.nb_ic_blocking * jcp.ic_block);
                int reduce_work = ocb_step * jcp.oc_block;
                for (int ocb = ocb_l2; ocb < ocb_end; ocb += ocb_step) {
                    int curr_nb_oc = nstl::min(ocb_step, ocb_end - ocb);
                    if (ocb + curr_nb_oc >= jcp.nb_oc) {
                        reduce_work = utils::this_block_size(ocb * jcp.oc_block,
                                jcp.oc, ocb_step * jcp.oc_block);
                    }

                    kdnn_jit_conv_ker_pipeline_iw_thr(jit_ker, par_conv, diff_src_w,
                            diff_dst_w, wht_w, 0, ocb, 1, iwb, reduce_work,
                            load_work);
                    diff_dst_w += diff_dst_c_stride;
                    wht_w += wht_oc_stride;
                }

                if (jcp.loop_order == kdnn_loop_cwgn) {
                    int dummy {0};
                    nd_iterator_jump(start, end, icc, ic_chunks, iwb, jcp.nb_iw,
                            gg, nb_groups, n, jcp.mb, dummy, 1);
                } else if (jcp.loop_order == kdnn_loop_gncw) {
                    int dummy {0};
                    nd_iterator_jump(start, end, gg, nb_groups, n, jcp.mb, icc,
                            ic_chunks, iwb, jcp.nb_iw, dummy, 1);
                } else if (jcp.loop_order == kdnn_loop_nhwcg) {
                    ++start;
                    nd_iterator_step(n, jcp.mb, iwb, jcp.nb_iw, icc, ic_chunks,
                            gg, nb_groups);
                } else {
                    assert(!"unsupported loop order");
                }
            }
        }

        // This call is required only to finalize pipeline with paramaters set
        // on the last iteration of loop above. Only valid pointers make sense
        // here as call parameters to avoid execution of prefetch instructions
        // with nullptr, other parameters are not used in real jit call here
        kdnn_jit_conv_ker_pipeline_iw_thr(jit_ker, par_conv, diff_src, diff_dst,
                weights, 0, 0, 0, 0, 0, 0);
    });
}

template <cpu_isa_t isa>
void kdnn_jit_sve_convolution_bwd_data_t<isa>::execute_backward_data_2d(const exec_ctx_t &ctx) const {
    auto diff_dst = CTX_IN_MEM(const float *, DNNL_ARG_DIFF_DST);
    auto weights = CTX_IN_MEM(const float *, DNNL_ARG_WEIGHTS);
    auto diff_src = CTX_OUT_MEM(float *, DNNL_ARG_DIFF_SRC);

    const memory_desc_wrapper diff_dst_d(pd()->diff_dst_md());
    const memory_desc_wrapper diff_src_d(pd()->diff_src_md());
    const memory_desc_wrapper weights_d(pd()->weights_md(0));

    const auto &jcp = pd()->jcp_;
    const kdnn_jit_conv_ker_t jit_ker = (decltype(jit_ker))kernel_->jit_ker();

    int ic_chunks = jcp.nb_ic / jcp.nb_ic_blocking;
    int g_blocking = 1;
    int nb_groups = jcp.ngroups / g_blocking;
    int work_amount = nb_groups * jcp.mb * ic_chunks * jcp.ih * jcp.nb_iw;
    int nthr = jcp.nthr;

    parallel(nthr, [&](const int ithr, const int nthr) {
        int start {0}, end {0}, start_copy;
        balance211(work_amount, nthr, ithr, start, end);
        start_copy = start;

        auto par_conv = kdnn_jit_conv_call_s();
        size_t diff_src_h_stride = diff_src_d.blk_off(0, 0, 1);
        size_t diff_dst_h_stride = diff_dst_d.blk_off(0, 0, 1);
        size_t diff_dst_c_stride = diff_dst_d.blk_off(0, 1);
        size_t wht_h_stride = kdnn_wht_blk_off(weights_d, 0, 0, 0, 1);
        size_t wht_oc_stride = kdnn_wht_blk_off(weights_d, 0, 1);

        bool is_fast_path = jcp.dilate_h == 0 && jcp.stride_h == 1;

        for (int ocb_l2 = 0; ocb_l2 < jcp.nb_oc; ocb_l2 += jcp.nb_oc_L2) {
            start = start_copy;
            int n {0}, gg {0}, icc {0}, ih_s {0}, iwb {0};

            if (jcp.loop_order == kdnn_loop_cwgn) {
                nd_iterator_init(start, icc, ic_chunks, iwb, jcp.nb_iw, gg,
                        nb_groups, n, jcp.mb, ih_s, jcp.ih);
            } else if (jcp.loop_order == kdnn_loop_gncw) {
                nd_iterator_init(start, gg, nb_groups, n, jcp.mb, icc,
                        ic_chunks, iwb, jcp.nb_iw, ih_s, jcp.ih);
            } else if (jcp.loop_order == kdnn_loop_nhwcg) {
                nd_iterator_init(start, n, jcp.mb, ih_s, jcp.ih, iwb, jcp.nb_iw,
                        icc, ic_chunks, gg, nb_groups);
            } else
                assert(!"unsupported loop order");

            while (start < end) {
                int icb = icc * jcp.nb_ic_blocking;
                int g = gg * g_blocking;
                int g_icb = g * jcp.nb_ic + icb;
                int g_ocb = g * jcp.nb_oc;

                int work_rem = end - start;
                int ih_e = ih_s + work_rem > jcp.ih ? jcp.ih : ih_s + work_rem;
                if (jcp.loop_order == kdnn_loop_nhwcg)
                    ih_e = ih_s + 1; //step instead
                int iw_s = iwb * jcp.iw_block;
                int ow_s = iw_s / jcp.stride_w;
                const bool is_dsrc_layout_nxc = jcp.src_tag == format_tag::nhwc;
                const int ic_off_idx = is_dsrc_layout_nxc
                        ? g * jcp.ic + icb * jcp.ic_block
                        : g_icb;
                auto diff_src_w
                        = diff_src + diff_src_d.blk_off(n, ic_off_idx, 0, iw_s);
                const bool is_ddst_layout_nxc = jcp.dst_tag == format_tag::nhwc;
                const int oc_off_idx = is_ddst_layout_nxc
                        ? g * jcp.oc + ocb_l2 * jcp.oc_block
                        : g_ocb + ocb_l2;
                auto diff_dst_w
                        = diff_dst + diff_dst_d.blk_off(n, oc_off_idx, 0, ow_s);
                auto wht_w = weights + kdnn_wht_blk_off(weights_d, g, ocb_l2, icb);

                int ocb_step = is_ddst_layout_nxc ? jcp.nb_oc_L2 : 1;
                int ocb_end = min(jcp.nb_oc, ocb_l2 + jcp.nb_oc_L2);
                const int load_work = utils::this_block_size(icb * jcp.ic_block,
                        jcp.ic, jcp.nb_ic_blocking * jcp.ic_block);
                int reduce_work = ocb_step * jcp.oc_block;
                for (int ocb = ocb_l2; ocb < ocb_end; ocb += ocb_step) {
                    int curr_nb_oc = nstl::min(ocb_step, ocb_end - ocb);
                    if (ocb + curr_nb_oc >= jcp.nb_oc) {
                        reduce_work = utils::this_block_size(ocb * jcp.oc_block,
                                jcp.oc, ocb_step * jcp.oc_block);
                    }
                    for (int ij = ih_s; ij < ih_e; ++ij) {
                        int oj, k_len, k_lo;
                        if (is_fast_path) { // dilate == 0 && stride == 1
                            int i_t_overflow
                                    = max(0, jcp.kh - 1 - ij - jcp.t_pad);
                            int i_b_overflow
                                    = max(0, jcp.kh - jcp.ih + ij - jcp.b_pad);
                            k_len = jcp.kh - i_t_overflow - i_b_overflow;
                            k_lo = i_b_overflow;
                            oj = ij + jcp.t_pad - i_b_overflow;
                        } else if (jcp.dilate_h != 0) { // stride == 1
                            int dilate_h = jcp.dilate_h + 1;
                            // Note: use div_up to account for "holes" in filter
                            int i_t_overflow
                                    = div_up(max(0,
                                                     (jcp.kh - 1) * dilate_h
                                                             - ij - jcp.t_pad),
                                            dilate_h);
                            int i_b_overflow = div_up(
                                    max(0,
                                            (jcp.kh - 1) * dilate_h + 1 - jcp.ih
                                                    + ij - jcp.b_pad),
                                    dilate_h);
                            k_len = jcp.kh - i_t_overflow - i_b_overflow;
                            k_lo = i_b_overflow;
                            oj = ij + jcp.t_pad - i_b_overflow * dilate_h;
                        } else { // dilate == 0
                            int i_t_overflow = max(0,
                                    (jcp.kh - 1 - ij - jcp.t_pad)
                                            / jcp.stride_h);
                            int i_b_overflow = max(0,
                                    (jcp.kh - jcp.ih + ij - jcp.b_pad)
                                            / jcp.stride_h);
                            int overflow_kh_hi = jcp.kh - 1
                                    - modulo(jcp.ih - 1 + jcp.b_pad - ij,
                                            jcp.stride_h);
                            int overflow_kh_lo
                                    = (ij + jcp.t_pad) % jcp.stride_h;

                            k_len = (overflow_kh_hi - overflow_kh_lo)
                                            / jcp.stride_h
                                    + 1 - i_t_overflow - i_b_overflow;
                            k_lo = overflow_kh_lo + i_b_overflow * jcp.stride_h;
                            oj = (ij + jcp.t_pad - k_lo) / jcp.stride_h;
                        }

                        kdnn_jit_conv_ker_pipeline_iw_thr(jit_ker, par_conv,
                                diff_src_w + ij * diff_src_h_stride,
                                diff_dst_w + oj * diff_dst_h_stride,
                                wht_w + k_lo * wht_h_stride, 0, ocb, k_len, iwb,
                                reduce_work, load_work);
                    }
                    diff_dst_w += diff_dst_c_stride;
                    wht_w += wht_oc_stride;
                }

                if (jcp.loop_order == kdnn_loop_cwgn) {
                    nd_iterator_jump(start, end, icc, ic_chunks, iwb, jcp.nb_iw,
                            gg, nb_groups, n, jcp.mb, ih_s, jcp.ih);
                } else if (jcp.loop_order == kdnn_loop_gncw) {
                    nd_iterator_jump(start, end, gg, nb_groups, n, jcp.mb, icc,
                            ic_chunks, iwb, jcp.nb_iw, ih_s, jcp.ih);
                } else if (jcp.loop_order == kdnn_loop_nhwcg) {
                    ++start;
                    nd_iterator_step(n, jcp.mb, ih_s, jcp.ih, iwb, jcp.nb_iw,
                            icc, ic_chunks, gg, nb_groups);
                } else
                    assert(!"unsupported loop order");
            }
        }

        // This call is required only to finalize pipeline with paramaters set
        // on the last iteration of loop above. Only valid pointers make sense
        // here as call parameters to avoid execution of prefetch instructions
        // with nullptr, other parameters are not used in real jit call here
        kdnn_jit_conv_ker_pipeline_iw_thr(jit_ker, par_conv, diff_src, diff_dst,
                weights, 0, 0, 0, 0, 0, 0);
    });
}

template <cpu_isa_t isa>
void kdnn_jit_sve_convolution_bwd_data_t<isa>::execute_backward_data_3d(const exec_ctx_t &ctx) const {
    auto diff_dst = CTX_IN_MEM(const float *, DNNL_ARG_DIFF_DST);
    auto weights = CTX_IN_MEM(const float *, DNNL_ARG_WEIGHTS);
    auto diff_src = CTX_OUT_MEM(float *, DNNL_ARG_DIFF_SRC);

    const memory_desc_wrapper diff_dst_d(pd()->diff_dst_md());
    const memory_desc_wrapper diff_src_d(pd()->diff_src_md());
    const memory_desc_wrapper weights_d(pd()->weights_md(0));

    const auto &jcp = pd()->jcp_;
    const kdnn_jit_conv_ker_t jit_ker = (decltype(jit_ker))kernel_->jit_ker();

    int ic_chunks = jcp.nb_ic / jcp.nb_ic_blocking;
    int g_blocking = 1;
    int nb_groups = jcp.ngroups / g_blocking;
    int work_amount = nb_groups * jcp.mb * ic_chunks * jcp.id * jcp.ih;
    int nthr = jcp.nthr;

    parallel(nthr, [&](const int ithr, const int nthr) {
        int start {0}, end {0}, start_copy;
        balance211(work_amount, nthr, ithr, start, end);
        start_copy = start;

        auto par_conv = kdnn_jit_conv_call_s();
        size_t diff_src_h_stride = diff_src_d.blk_off(0, 0, 0, 1);
        size_t diff_src_d_stride = diff_src_d.blk_off(0, 0, 1);
        size_t diff_dst_h_stride = diff_dst_d.blk_off(0, 0, 0, 1);
        size_t diff_dst_d_stride = diff_dst_d.blk_off(0, 0, 1);
        size_t diff_dst_c_stride = diff_dst_d.blk_off(0, 1);
        size_t wht_h_stride = kdnn_wht_blk_off(weights_d, 0, 0, 0, 0, 1);
        size_t wht_d_stride = kdnn_wht_blk_off(weights_d, 0, 0, 0, 1);
        size_t wht_oc_stride = kdnn_wht_blk_off(weights_d, 0, 1);

        bool is_fast_path_d = jcp.dilate_d == 0 && jcp.stride_d == 1;
        bool is_fast_path_h = jcp.dilate_h == 0 && jcp.stride_h == 1;

        for (int ocb_l2 = 0; ocb_l2 < jcp.nb_oc; ocb_l2 += jcp.nb_oc_L2) {
            start = start_copy;
            int n {0}, gg {0}, icc {0}, ih_s {0}, id_s {0};
            // Input width threading is not currently implemented for 3d, so it
            // is not included in the iterator.
            if (jcp.loop_order == kdnn_loop_cwgn)
                nd_iterator_init(start, icc, ic_chunks, gg, nb_groups, n,
                        jcp.mb, id_s, jcp.id, ih_s, jcp.ih);
            else if (jcp.loop_order == kdnn_loop_gncw)
                nd_iterator_init(start, gg, nb_groups, n, jcp.mb, icc,
                        ic_chunks, id_s, jcp.id, ih_s, jcp.ih);
            else if (jcp.loop_order == kdnn_loop_nhwcg)
                nd_iterator_init(start, n, jcp.mb, id_s, jcp.id, ih_s, jcp.ih,
                        icc, ic_chunks, gg, nb_groups);
            else
                assert(!"unsupported loop order");

            while (start < end) {
                int icb = icc * jcp.nb_ic_blocking;
                int g = gg * g_blocking;
                int g_icb = g * jcp.nb_ic + icb;
                int g_ocb = g * jcp.nb_oc;

                int work_rem = end - start;
                int ih_e = ih_s + work_rem > jcp.ih ? jcp.ih : ih_s + work_rem;
                if (jcp.loop_order == kdnn_loop_nhwcg)
                    ih_e = ih_s + 1; //step instead
                int d_len = 0, d_lo = 0, d_oj = 0;
                if (is_fast_path_d) { // dilate == 0 && stride == 1
                    int d_t_overflow = max(0, jcp.kd - 1 - id_s - jcp.f_pad);
                    int d_b_overflow
                            = max(0, jcp.kd - jcp.id + id_s - jcp.back_pad);
                    d_len = jcp.kd - d_t_overflow - d_b_overflow;
                    d_lo = d_b_overflow;
                    d_oj = id_s + jcp.f_pad - d_b_overflow;
                } else if (jcp.dilate_d != 0) { // stride == 1
                    int dilate_d = jcp.dilate_d + 1;
                    // Note: use div_up to account for "holes" in filter
                    int d_t_overflow = div_up(
                            max(0, (jcp.kd - 1) * dilate_d - id_s - jcp.f_pad),
                            dilate_d);
                    int d_b_overflow = div_up(
                            max(0,
                                    (jcp.kd - 1) * dilate_d + 1 - jcp.id + id_s
                                            - jcp.back_pad),
                            dilate_d);
                    d_len = jcp.kd - d_t_overflow - d_b_overflow;
                    d_lo = d_b_overflow;
                    d_oj = id_s + jcp.f_pad - d_b_overflow * dilate_d;
                } else { // dilate == 0
                    int d_t_overflow = max(
                            0, (jcp.kd - 1 - id_s - jcp.f_pad) / jcp.stride_d);
                    int d_b_overflow = max(0,
                            (jcp.kd - jcp.id + id_s - jcp.back_pad)
                                    / jcp.stride_d);
                    int overflow_kd_hi = jcp.kd - 1
                            - modulo(jcp.id - 1 + jcp.back_pad - id_s,
                                    jcp.stride_d);
                    int overflow_kd_lo = (id_s + jcp.f_pad) % jcp.stride_d;

                    d_len = (overflow_kd_hi - overflow_kd_lo) / jcp.stride_d + 1
                            - d_t_overflow - d_b_overflow;
                    d_lo = overflow_kd_lo + d_b_overflow * jcp.stride_d;
                    d_oj = (id_s + jcp.f_pad - d_lo) / jcp.stride_d;
                }

                const bool is_dsrc_layout_nxc
                        = jcp.src_tag == format_tag::ndhwc;
                const int ic_off_idx = is_dsrc_layout_nxc
                        ? g * jcp.ic + icb * jcp.ic_block
                        : g_icb;
                auto diff_src_w = diff_src + diff_src_d.blk_off(n, ic_off_idx)
                        + id_s * diff_src_d_stride;
                const bool is_ddst_layout_nxc
                        = jcp.dst_tag == format_tag::ndhwc;
                const int oc_off_idx = is_ddst_layout_nxc
                        ? g * jcp.oc + ocb_l2 * jcp.oc_block
                        : g_ocb + ocb_l2;
                auto diff_dst_w = diff_dst + diff_dst_d.blk_off(n, oc_off_idx)
                        + d_oj * diff_dst_d_stride;
                auto wht_w = weights + kdnn_wht_blk_off(weights_d, g, ocb_l2, icb)
                        + d_lo * wht_d_stride;

                int ocb_step = is_ddst_layout_nxc ? jcp.nb_oc_L2 : 1;
                int ocb_end = min(jcp.nb_oc, ocb_l2 + jcp.nb_oc_L2);
                const int load_work = utils::this_block_size(icb * jcp.ic_block,
                        jcp.ic, jcp.nb_ic_blocking * jcp.ic_block);
                int reduce_work = ocb_step * jcp.oc_block;
                for (int ocb = ocb_l2; ocb < ocb_end; ocb += ocb_step) {
                    int curr_nb_oc = nstl::min(ocb_step, ocb_end - ocb);
                    if (ocb + curr_nb_oc >= jcp.nb_oc) {
                        reduce_work = utils::this_block_size(ocb * jcp.oc_block,
                                jcp.oc, ocb_step * jcp.oc_block);
                    }
                    for (int ij = ih_s; ij < ih_e; ++ij) {
                        int oj, k_len, k_lo;
                        if (is_fast_path_h) { // dilate == 0 && stride == 1
                            int i_t_overflow
                                    = max(0, jcp.kh - 1 - ij - jcp.t_pad);
                            int i_b_overflow
                                    = max(0, jcp.kh - jcp.ih + ij - jcp.b_pad);
                            k_len = jcp.kh - i_t_overflow - i_b_overflow;
                            k_lo = i_b_overflow;
                            oj = ij + jcp.t_pad - i_b_overflow;
                        } else if (jcp.dilate_h != 0) { // stride == 1
                            int dilate_h = jcp.dilate_h + 1;
                            // Note: use div_up to account for "holes" in filter
                            int i_t_overflow
                                    = div_up(max(0,
                                                     (jcp.kh - 1) * dilate_h
                                                             - ij - jcp.t_pad),
                                            dilate_h);
                            int i_b_overflow = div_up(
                                    max(0,
                                            (jcp.kh - 1) * dilate_h + 1 - jcp.ih
                                                    + ij - jcp.b_pad),
                                    dilate_h);
                            k_len = jcp.kh - i_t_overflow - i_b_overflow;
                            k_lo = i_b_overflow;
                            oj = ij + jcp.t_pad - i_b_overflow * dilate_h;
                        } else { // dilate == 0
                            int i_t_overflow = max(0,
                                    (jcp.kh - 1 - ij - jcp.t_pad)
                                            / jcp.stride_h);
                            int i_b_overflow = max(0,
                                    (jcp.kh - jcp.ih + ij - jcp.b_pad)
                                            / jcp.stride_h);
                            int overflow_kh_hi = jcp.kh - 1
                                    - modulo(jcp.ih - 1 + jcp.b_pad - ij,
                                            jcp.stride_h);
                            int overflow_kh_lo
                                    = (ij + jcp.t_pad) % jcp.stride_h;

                            k_len = (overflow_kh_hi - overflow_kh_lo)
                                            / jcp.stride_h
                                    + 1 - i_t_overflow - i_b_overflow;
                            k_lo = overflow_kh_lo + i_b_overflow * jcp.stride_h;
                            oj = (ij + jcp.t_pad - k_lo) / jcp.stride_h;
                        }
                        assert(k_len >= 0);

                        kdnn_jit_sve_conv_3d_ker_pipeline(jit_ker, par_conv,
                                diff_src_w + ij * diff_src_h_stride,
                                diff_dst_w + oj * diff_dst_h_stride,
                                wht_w + k_lo * wht_h_stride, 0, ocb, k_len,
                                d_len, reduce_work, load_work);
                    }
                    diff_dst_w += diff_dst_c_stride;
                    wht_w += wht_oc_stride;
                }

                if (jcp.loop_order == kdnn_loop_cwgn)
                    nd_iterator_jump(start, end, icc, ic_chunks, gg, nb_groups,
                            n, jcp.mb, id_s, jcp.id, ih_s, jcp.ih);
                else if (jcp.loop_order == kdnn_loop_gncw)
                    nd_iterator_jump(start, end, gg, nb_groups, n, jcp.mb, icc,
                            ic_chunks, id_s, jcp.id, ih_s, jcp.ih);
                else if (jcp.loop_order == kdnn_loop_nhwcg) {
                    ++start;
                    nd_iterator_step(n, jcp.mb, id_s, jcp.id, ih_s, jcp.ih, icc,
                            ic_chunks, gg, nb_groups);
                } else
                    assert(!"unsupported loop order");
            }
        }

        // This call is required only to finalize pipeline with paramaters set
        // on the last iteration of loop above. Only valid pointers make sense
        // here as call parameters to avoid execution of prefetch instructions
        // with nullptr, other parameters are not used in real jit call here
        kdnn_jit_sve_conv_3d_ker_pipeline(jit_ker, par_conv, diff_src, diff_dst,
                weights, 0, 0, 1, 1, 0, 0);
    });
}

template struct kdnn_jit_sve_convolution_bwd_data_t<sve_256>;

} // namespace aarch64
} // namespace cpu
} // namespace impl
} // namespace dnnl

