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
#include "common/nstl.hpp"
#include "common/type_helpers.hpp"
#include "common/utils.hpp"

#include "cpu/aarch64/cpu_barrier.hpp"

#include "kudnn_jit_f32_conv_kernel.hpp"
#include "cpu/platform.hpp"

#define KUDNN_GET_OFF(field) static_cast<int32_t>(offsetof(kdnn_jit_conv_call_s, field))
#define KUDNN_A64FX_L2_EFFECTIVE_CAPACITY ((666 - 128) * 1024)

namespace dnnl {
namespace impl {
namespace cpu {
namespace aarch64 {

using namespace dnnl::impl::format_tag;
using namespace dnnl::impl::memory_tracking::names;
using namespace dnnl::impl::utils;

namespace {

constexpr auto small_spatial = 14;
unsigned int L2_cache_size = platform::get_per_core_cache_size(2);

// calculates filter size taking into account dilation
inline int calculate_extended_filter_size(int filter_size, int dilation) {
    return (filter_size - 1) * (dilation + 1) + 1;
}

inline int calculate_end_padding(int start_padding, int dst_size, int src_size,
        int spatial_stride, int dilated_filter_size) {
    return (dst_size - 1) * spatial_stride + dilated_filter_size
            - (src_size + start_padding);
}

inline void pick_loop_order(kdnn_jit_conv_conf_t &jcp) {
    using namespace prop_kind;
    assert(one_of(
            jcp.prop_kind, forward_training, forward_inference, backward_data));
    auto w = (jcp.prop_kind == backward_data) ? jcp.iw : jcp.ow;
    auto h = (jcp.prop_kind == backward_data) ? jcp.ih : jcp.oh;

    // The w in the loop order is currently ignored by 3D BWD_D
    jcp.loop_order = (w <= small_spatial && h <= small_spatial) ? kdnn_loop_cwgn
                                                                : kdnn_loop_gncw;
    if (utils::one_of(jcp.src_tag, format_tag::ndhwc, format_tag::nhwc,
                format_tag::nwc)
            && jcp.ngroups > 1 && jcp.oc < 8)
        jcp.loop_order = kdnn_loop_nhwcg;
}

inline status_t init_tag(format_tag_t &tag, memory_desc_t &md,
        const memory_desc_wrapper &mdw, const format_tag_t tag_value) {
    if (mdw.format_kind() == format_kind::any) {
        CHECK(memory_desc_init_by_tag(md, tag_value));
        tag = tag_value;
    } else {
        tag = mdw.matches_one_of_tag(tag_value);
    }

    if (tag != tag_value) return status::unimplemented;

    return status::success;
}

inline bool is_1stconv(const kdnn_jit_conv_conf_t &jcp) {
    if (mayiuse(sve_256))
        return (jcp.ic < 8 && jcp.ngroups == 1);
    else
        return one_of(jcp.ic, 1, 3);
}

inline bool is_ow_threading_on(const kdnn_jit_conv_conf_t &jcp) {
    return (jcp.nb_ow > 1);
}

inline bool is_owb_prefetching(const kdnn_jit_conv_conf_t &jcp) {
    return false;
}

inline bool is_iw_threading_on(const kdnn_jit_conv_conf_t &jcp) {
    return (jcp.nb_iw > 1);
}

} // namespace

template <cpu_isa_t isa>
void kdnn_jit_sve_conv_fwd_kernel<isa>::prepare_output(int ur_w) {

    auto zreg_out_s = [=](int i_ur, int i_oc) {
        if (jcp.kernel_kind == kdnn_cont_bcast) {
            int idx = 29 - (i_ur + i_oc * jcp.ur_w);
            return ZRegS(idx);
        }
        int idx = i_ur + i_oc * jcp.ur_w;
        assert(idx < ker_reg_base_idx);
        return ZRegS(idx);
    };

    int prev_out_ofs = -1;
    for (int k = 0; k < jcp.nb_oc_blocking; k++)
        for (int j = 0; j < ur_w; j++) {
            fmov(zreg_out_s(j, k));
            if (!is_owb_prefetching(jcp)) {
                size_t aux_output_offset = get_output_offset(j, k);
                std::string op = "LD";
                if (j == 0) {
                    prefetch(op, 2, reg_out_prf, aux_output_offset);
                    add_imm(reg_tmp_addr, reg_out_prf, aux_output_offset,
                            reg_tmp_imm);
                } else {
                    add_imm(reg_tmp_addr, reg_tmp_addr,
                            aux_output_offset - prev_out_ofs, reg_tmp_imm);
                    prefetch(op, 2, reg_tmp_addr, 0);
                }
                prev_out_ofs = aux_output_offset;
            }
        }
}

template <cpu_isa_t isa>
void kdnn_jit_sve_conv_fwd_kernel<isa>::store_output(int ur_w) {

    Label no_update_label, store_label, eltwise_label;

    auto _test = [&](const int cond) { return tst(reg_channel, cond); };

    auto zreg_tmp = [=](int idx) { return ZReg(idx); };
    auto zreg_tmp_s = [=](int idx) { return ZRegS(idx); };

    auto zreg_out = [=](int i_ur, int i_oc) {
        if (jcp.kernel_kind == kdnn_cont_bcast) {
            int idx = 29 - (i_ur + i_oc * jcp.ur_w);
            return ZReg(idx);
        }
        int idx = i_ur + i_oc * jcp.ur_w;
        assert(idx < ker_reg_base_idx);
        return ZReg(idx);
    };
    auto zreg_out_s = [=](int i_ur, int i_oc) {
        if (jcp.kernel_kind == kdnn_cont_bcast) {
            int idx = 29 - (i_ur + i_oc * jcp.ur_w);
            return ZRegS(idx);
        }
        int idx = i_ur + i_oc * jcp.ur_w;
        assert(idx < ker_reg_base_idx);
        return ZRegS(idx);
    };

    ldr(reg_channel, ptr(kdnn_abi_param1, KUDNN_GET_OFF(flags)));

    if (jcp.with_bias) { ldr(reg_bias, ptr(kdnn_abi_param1, KUDNN_GET_OFF(bias))); }

    if (!jcp.with_sum) {
        auto _jmp = [&](const Label &l) { return b(NE, l); };

        _test(kdnn_FLAG_IC_FIRST);
        _jmp(no_update_label);
    }

    int reg_ofs = jcp.ur_w * jcp.nb_oc_blocking;
    int num_regs = 32 - reg_ofs;
    int prev_out_ofs = -1;

    for (int k = 0; k < jcp.nb_oc_blocking; k++) {
        for (int j = 0; j < ur_w; j++) {
            size_t aux_output_offset = get_output_offset(j, k);
            int idx = jcp.kernel_kind == kdnn_cont_bcast
                      ? j
                      : reg_ofs + ((j + k * ur_w) % num_regs);
            if (j == 0) {
                add_imm(reg_out_ofs, reg_out, aux_output_offset, reg_tmp_imm);
                prev_out_ofs = aux_output_offset;
                ld1w(zreg_tmp(idx).s, P_ALL_ONE / T_z, ptr(reg_out_ofs));
            } else if (kdnn_ldr_imm_check<size_t, isa>(
                               aux_output_offset - prev_out_ofs)) {
                add_imm(X_DEFAULT_ADDR, reg_out_ofs,
                        (aux_output_offset - prev_out_ofs), X_TMP_0);
                ld1w(zreg_tmp(idx).s, P_ALL_ONE / T_z, ptr(X_DEFAULT_ADDR));
            } else {
                add_imm(reg_out_ofs, reg_out_ofs,
                        aux_output_offset - prev_out_ofs, reg_tmp_imm);
                prev_out_ofs = aux_output_offset;
                ld1w(zreg_tmp(idx).s, P_ALL_ONE / T_z, ptr(reg_out_ofs));
            }
        }
        for (int j = 0; j < ur_w; j++) {
            int idx = jcp.kernel_kind == kdnn_cont_bcast
                      ? j
                      : reg_ofs + ((j + k * ur_w) % num_regs);
            fadd(zreg_out_s(j, k), zreg_out_s(j, k), zreg_tmp_s(idx));
        }
    }

    if (!jcp.with_sum) {
        b(eltwise_label);
    } else {
        auto _jmp = [&](const Label &l) { return b(EQ, l); };

        // *Note 1
        _test(kdnn_FLAG_IC_FIRST);
        _jmp(eltwise_label);
    }

    auto bias_load = [=](int bias_offset, int idx) {
        int ofs = bias_offset;

        if ((kdnn_VL_OFS(ofs, isa) < LDRMAX) && (kdnn_VL_OFS(ofs, isa) >= (-1 * LDRMAX))
                && ((ofs & 0x3f) == 0)) {
            add_imm(X_DEFAULT_ADDR, reg_bias, ofs, X_TMP_0);
            ld1w(zreg_tmp(idx).s, P_ALL_ONE / T_z, ptr(X_DEFAULT_ADDR));
        } else {
            add_imm(reg_tmp_addr, reg_bias, ofs, reg_tmp_imm);
            ld1w(zreg_tmp(idx).s, P_ALL_ONE / T_z, ptr(reg_tmp_addr));
        }
    };

    L(no_update_label);
    if (jcp.with_bias) {
        for (int k = 0; k < jcp.nb_oc_blocking; k++) {
            int bias_offset = jcp.typesize_out * k * jcp.oc_block;
            int idx = jcp.kernel_kind == kdnn_cont_bcast ? k : reg_ofs + (k % num_regs);
            bias_load(bias_offset, idx);
            for (int j = 0; j < ur_w; j++) {
                fadd(zreg_out_s(j, k), zreg_out_s(j, k), zreg_tmp_s(idx));
            }
            int ofs = bias_offset + 256; // cache line size ?
            std::string op = "LD";
            prefetch(op, 2, reg_bias, ofs);
        }
    }

    L(eltwise_label);
    if (jcp.with_eltwise) {
       tst(reg_channel, kdnn_FLAG_IC_LAST);
       b(EQ, store_label);

        if (ur_w == jcp.ur_w) {
            if (jcp.kernel_kind == kdnn_cont_bcast) {
                eltwise_injector_->compute_vector_range(
                    30 - jcp.nb_oc_blocking * jcp.ur_w, 30);
            } else {
                eltwise_injector_->compute_vector_range(
                    0, jcp.nb_oc_blocking * jcp.ur_w);
            }
        } else {
            for (int k = 0; k < jcp.nb_oc_blocking; k++)
                if (jcp.kernel_kind == kdnn_cont_bcast) {
                    eltwise_injector_->compute_vector_range(
                        30 - k * jcp.ur_w - ur_w, 30 - k * jcp.ur_w);
                } else {
                    eltwise_injector_->compute_vector_range(
                        k * jcp.ur_w, k * jcp.ur_w + ur_w);
                }
        }
    }
    auto out_str = [=](int j, int k, int aux_output_offset, int prev_out_ofs) {
        int ofs = aux_output_offset;

        if (kdnn_str_imm_check<int, isa>(ofs)) {
            add_imm(X_DEFAULT_ADDR, reg_out, ofs, X_TMP_0);
            st1w(zreg_out(j, k).s, P_ALL_ONE / T_z, ptr(X_DEFAULT_ADDR));
        } else if ((prev_out_ofs != -1)
                && kdnn_str_imm_check<int, isa>(ofs - prev_out_ofs)) {
            add_imm(X_DEFAULT_ADDR, reg_tmp_addr, (ofs - prev_out_ofs),
                    X_TMP_0);
            st1w(zreg_out(j, k).s, P_ALL_ONE / T_z, ptr(X_DEFAULT_ADDR));
        } else {
            if (prev_out_ofs == -1)
                add_imm(reg_tmp_addr, reg_out, ofs, reg_tmp_imm);
            else
                add_imm(reg_tmp_addr, reg_tmp_addr, ofs - prev_out_ofs,
                        reg_tmp_imm);
            st1w(zreg_out(j, k).s, P_ALL_ONE / T_z, ptr(reg_tmp_addr));
            prev_out_ofs = aux_output_offset;
        }
        return prev_out_ofs;
    };

    L(store_label);
    prev_out_ofs = -1;
    for (int k = 0; k < jcp.nb_oc_blocking; k++) {
        for (int j = 0; j < ur_w; j++) {
            size_t aux_output_offset = get_output_offset(j, k);

            prev_out_ofs = out_str(j, k, aux_output_offset,
                    prev_out_ofs); // <- reg_tmp_addr
        }
    }
}

template <cpu_isa_t isa>
void kdnn_jit_sve_conv_fwd_kernel<isa>::compute_loop_fma(
    int ur_w, int pad_l, int pad_r) {
    const bool is_source_layout_nxc = is_src_layout_nxc();
    const bool icb_loop_in_compute_function = is_source_layout_nxc;
    const int ic_tail = jcp.ic_tail;
    int iw = jcp.iw;
    int kw = jcp.kw;
    int ic_block = jcp.ic_block;
    int oc_block = jcp.oc_block;
    int nb_oc_block = jcp.nb_oc_blocking;
    Label kh_label, kd_label;
    std::vector<Label> ic_tail_jmp(kw);

    // It seems that this compute_loop currently only handles one block of oc.
    // assert if it is extended in future to catch unpadded_oc_tail.
    assert(nb_oc_block == 1);

    int num_ker_loads = ic_block * kw;
    int ker_pipeline_depth = std::min(jcp.ic, 2);

    int inp_mul = is_source_layout_nxc ? jcp.ngroups * jcp.ic
                                       : (!jcp.is_1stconv ? ic_block : 1);

    if (one_of(jcp.ndims, 3, 4)) {
        mov(aux_reg_inp, reg_inp);
        add_imm(aux_reg_inp2, aux_reg_inp, 0x100, reg_tmp_imm);
        add_imm(aux_reg_inp3, aux_reg_inp2, 0x100, reg_tmp_imm);
        mov(aux_reg_ker, reg_ker);
    }
    if (jcp.ndims == 5) {
        mov(reg_out_org, reg_out);
        ldr(reg_ki, ptr(kdnn_abi_param1, KUDNN_GET_OFF(kd_padding)));
        if (icb_loop_in_compute_function) {
            // need to continue with the same kernel pointer, but as
            // aux_reg_ker_d == reg_ker we need to save its value and restore
            // it after kd loop
            mov(aux_reg_ker_d_org, aux_reg_ker_d);
        } else {
            mov(aux_reg_ker_d, aux_reg_ker_d_org);
        }
        mov(aux_reg_inp_d, reg_inp);

        L(kd_label);
        ldr(reg_kj, ptr(kdnn_abi_param1, KUDNN_GET_OFF(kh_padding)));
    } else {
        mov(reg_kj, reg_kh);
    }

    if (jcp.ndims == 5) {
        mov(aux_reg_inp, aux_reg_inp_d);
        add_imm(aux_reg_inp2, aux_reg_inp, 0x100, reg_tmp_imm);
        add_imm(aux_reg_inp3, aux_reg_inp2, 0x100, reg_tmp_imm);
        mov(aux_reg_ker, aux_reg_ker_d);
    }

    auto zreg_inp_s = [=](int i_ic, int nb_x_blocking) {
        int idx = i_ic + nb_x_blocking * jcp.ur_w;
        assert(idx < 31);
        return ZRegS(idx);
    };
    auto zreg_out_s = [=](int i_ur, int i_oc) {
        int idx = i_ur + i_oc * jcp.ur_w;
        assert(idx < ker_reg_base_idx);
        return ZRegS(idx);
    };
    auto zreg_wei = [=](int idx) {
        assert(idx < 32);
        return ZReg(idx);
    };
    auto zreg_wei_s = [=](int idx) {
        assert(idx < 32);
        return ZRegS(idx);
    };

    auto bcast_load = [&](int jj, int nb_oc_block, int aux_input_offset,
                              int prev_ofs) {
        if (kdnn_ld1rw_imm_check(aux_input_offset)) {
            ld1rw(zreg_inp_s(jj, nb_oc_block), P_ALL_ONE,
                    ptr(aux_reg_inp, static_cast<int32_t>(aux_input_offset)));
        } else if (kdnn_ld1rw_imm_check(aux_input_offset - 0x100)) {
            ld1rw(zreg_inp_s(jj, nb_oc_block), P_ALL_ONE,
                    ptr(aux_reg_inp2,
                            static_cast<int32_t>(aux_input_offset - 0x100)));
        } else if (kdnn_ld1rw_imm_check(aux_input_offset - 0x200)) {
            ld1rw(zreg_inp_s(jj, nb_oc_block), P_ALL_ONE,
                    ptr(aux_reg_inp3,
                            static_cast<int32_t>(aux_input_offset - 0x200)));
        } else {
            if ((prev_ofs != -1)
                    && kdnn_ld1rw_imm_check(aux_input_offset - prev_ofs)) {

                ld1rw(zreg_inp_s(jj, nb_oc_block), P_ALL_ONE,
                        ptr(reg_prev_bcast_addr,
                                static_cast<int32_t>(
                                        aux_input_offset - prev_ofs)));
            } else {
                int ofs;
                if ((prev_ofs != -1) && ((aux_input_offset - prev_ofs) > 0)) {
                    ofs = aux_input_offset - prev_ofs;
                    add_imm(reg_prev_bcast_addr, reg_prev_bcast_addr, ofs,
                            reg_tmp_imm);
                } else {
                    ofs = aux_input_offset;
                    add_imm(reg_prev_bcast_addr, aux_reg_inp, ofs, reg_tmp_imm);
                }

                ld1rw(zreg_inp_s(jj, nb_oc_block), P_ALL_ONE,
                        ptr(reg_prev_bcast_addr));
                prev_ofs = aux_input_offset;
            }
        }

        return prev_ofs;
    };

    auto wei_load = [=](int aux_kernel_offset, int reg_idx, int prev_ofs) {
        int ofs = aux_kernel_offset;

        if (kdnn_ldr_imm_check<int, isa>(ofs)) {
            add_imm(X_DEFAULT_ADDR, aux_reg_ker, ofs, X_TMP_0);
            ld1w(zreg_wei(reg_idx).s, P_ALL_ONE / T_z, ptr(X_DEFAULT_ADDR));
        } else {
            int ofs_tmp = ofs - prev_ofs;
            if ((prev_ofs != -1) && kdnn_ldr_imm_check<int, isa>(ofs_tmp)) {
                add_imm(X_DEFAULT_ADDR, reg_prev_wei_addr, ofs_tmp, X_TMP_0);
                ld1w(zreg_wei(reg_idx).s, P_ALL_ONE / T_z, ptr(X_DEFAULT_ADDR));
            } else {
                if ((prev_ofs != -1) && (ofs_tmp > 0)) {
                    ofs_tmp = aux_kernel_offset - prev_ofs;
                    add_imm(reg_prev_wei_addr, reg_prev_wei_addr, ofs_tmp,
                            reg_tmp_imm);
                } else {
                    add_imm(reg_prev_wei_addr, aux_reg_ker, ofs, reg_tmp_imm);
                }

                ld1w(zreg_wei(reg_idx).s, P_ALL_ONE / T_z,
                        ptr(reg_prev_wei_addr));
                prev_ofs = ofs;
            }
        }
        return prev_ofs;
    };

    align(32);
    L(kh_label);
    {
        int prev_bcast_ofs = -1;
        int prev_wei_ofs = -1;
        int wei_reg_ofs = 30;
        int step = 0;
        for (int ki = 0; ki < kw; ki++) {
            for (int ic = 0; ic < ic_block; ic++) {
                if (ic_tail && ic >= ic_tail) {
                    // if src has only tails to compute, skip early
                    if (jcp.ic == ic_tail)
                        break;
                    else if (ic == ic_tail) {
                        cmp_imm(reg_channel, ic_tail, reg_tmp_imm);
                        b(EQ, ic_tail_jmp[ki]);
                    }
                }
                int aux_kernel_offset = 0;
                if (step == 0) {
                    for (int i = 0; i < ker_pipeline_depth; i++) {
                        aux_kernel_offset = jcp.typesize_in
                            * (ki * ic_block * oc_block + (ic + i) * oc_block);
                        prev_wei_ofs = wei_load(aux_kernel_offset,
                            wei_reg_ofs + i,
                            prev_wei_ofs);
                    }
                } else if (step < num_ker_loads - ker_pipeline_depth + 1) {
                    int load_offset = ker_pipeline_depth - 1;
                    int ker_load_reg_idx
                            = (step + load_offset) % ker_pipeline_depth;
                    aux_kernel_offset = jcp.typesize_in
                            * (ki * ic_block * oc_block + (ic + load_offset) * oc_block);
                    prev_wei_ofs = wei_load(aux_kernel_offset,
                            wei_reg_ofs + ker_load_reg_idx,
                            prev_wei_ofs);
                }

                int j_start = get_ow_start(ki, pad_l);
                int j_end = get_ow_end(ur_w, ki, pad_r);
                for (int j = j_start; j < j_end; j++) {
                    size_t aux_input_offset
                            = get_input_offset(ki, ic, j, pad_l);
                    prev_bcast_ofs = bcast_load(j, nb_oc_block,
                            aux_input_offset, prev_bcast_ofs);
                }
                for (int j = j_start; j < j_end; j++) {
                    // size_t aux_input_offset
                    //         = get_input_offset(ki, ic, j, pad_l);
                    // prev_bcast_ofs = bcast_load(0, nb_oc_block, aux_input_offset, prev_bcast_ofs);
                    fmla(zreg_out_s(j, 0), P_ALL_ONE,
                        zreg_inp_s(j, nb_oc_block),
                        zreg_wei_s(wei_reg_ofs + step % ker_pipeline_depth));
                }
                step++;
            }
            L(ic_tail_jmp[ki]);
        }
        int shift_kernel_ptr = jcp.typesize_in * kw * oc_block * ic_block;
        add_imm(aux_reg_ker, aux_reg_ker, shift_kernel_ptr, reg_tmp_imm);
        int shift_input_ptr = jcp.typesize_in * (jcp.dilate_h + 1) * iw * inp_mul;
        add_imm(aux_reg_inp, aux_reg_inp, shift_input_ptr, reg_tmp_imm);
        add_imm(aux_reg_inp2, aux_reg_inp, 0x100, reg_tmp_imm);
        add_imm(aux_reg_inp3, aux_reg_inp2, 0x100, reg_tmp_imm);
        sub(reg_kj, reg_kj, 1);
        cmp(reg_kj, 0);
        b(GT, kh_label);
    }

    if (jcp.ndims == 5) {
        add_imm(aux_reg_inp_d, aux_reg_inp_d,
                typesize * (jcp.dilate_d + 1) * jcp.ih * jcp.iw * inp_mul,
                reg_tmp_imm);
        const int ker_shift
                = typesize * jcp.kw * jcp.kh * jcp.oc_block * jcp.ic_block;
        add_imm(aux_reg_ker_d, aux_reg_ker_d, ker_shift, reg_tmp_imm);

        sub(reg_ki, reg_ki, 1); //dec(reg_ki);
        cmp(reg_ki, 0);
        b(GT, kd_label);

        if (icb_loop_in_compute_function) mov(aux_reg_ker_d, aux_reg_ker_d_org);
        mov(reg_out, reg_out_org);
    }
}

template <cpu_isa_t isa>
void kdnn_jit_sve_conv_fwd_kernel<isa>::compute_loop_fma_cont(
        int ur_w, int pad_l, int pad_r) {
    int kw = jcp.kw;
    int ic_block = jcp.ic_block;
    int oc_block = jcp.oc_block;
    int nb_oc_block = jcp.nb_oc_blocking;
    const bool is_source_layout_nxc = is_src_layout_nxc();
    const bool icb_loop_in_compute_function = is_source_layout_nxc;
    const int ic_tail = jcp.ic_tail;

    Label kh_label, kd_label;

    std::vector<Label> ic_tail_jmp(kw);
    int shift_kernel_ptr
            = jcp.typesize_in * jcp.kw * jcp.oc_block * jcp.ic_block;
    int inp_mul = is_source_layout_nxc ? jcp.ngroups * jcp.ic
                                       : (!jcp.is_1stconv ? ic_block : 1);

    int shift_input_ptr
            = jcp.typesize_in * (jcp.dilate_h + 1) * jcp.iw * inp_mul;

    if (one_of(jcp.ndims, 3, 4)) {
        mov(aux_reg_inp, reg_inp);
        add_imm(aux_reg_inp2, aux_reg_inp, 0x100, reg_tmp_imm);
        add_imm(aux_reg_inp3, aux_reg_inp2, 0x100, reg_tmp_imm);
        mov(aux_reg_ker, reg_ker);
    }

    if (jcp.ndims == 5) {
        mov(reg_out_org, reg_out);
        ldr(reg_ki, ptr(kdnn_abi_param1, KUDNN_GET_OFF(kd_padding)));
        if (icb_loop_in_compute_function) {
            // need to continue with the same kernel pointer, but as
            // aux_reg_ker_d == reg_ker we need to save its value and restore
            // it after kd loop
            mov(aux_reg_ker_d_org, aux_reg_ker_d);
        } else {
            mov(aux_reg_ker_d, aux_reg_ker_d_org);
        }
        mov(aux_reg_inp_d, reg_inp);

        L(kd_label);
        ldr(reg_kj, ptr(kdnn_abi_param1, KUDNN_GET_OFF(kh_padding)));
    } else {
        mov(reg_kj, reg_kh);
    }

    if (jcp.ndims == 5) {
        mov(aux_reg_inp, aux_reg_inp_d);
        add_imm(aux_reg_inp2, aux_reg_inp, 0x100, reg_tmp_imm);
        add_imm(aux_reg_inp3, aux_reg_inp2, 0x100, reg_tmp_imm);
        mov(aux_reg_ker, aux_reg_ker_d);
    }

    auto zreg_inp_s = [=](int i_ic) {
        return ZRegS(i_ic);
    };
    auto zreg_inp_idx = [=](int i_ic, int idx) {
        assert(idx < 32);
        return ZRegS(i_ic)[idx];
    };
    auto zreg_out_s = [=](int i_ur, int i_oc) {
        int idx = 29 - (i_ur + i_oc * jcp.ur_w);
        assert(idx >= 0);
        return ZRegS(idx);
    };
    auto zreg_wei = [=](int idx) {
        assert(idx < 32);
        return ZReg(idx);
    };
    auto zreg_wei_s = [=](int idx) {
        assert(idx < 32);
        return ZRegS(idx);
    };

    auto bcast_load = [&](int jj, int aux_input_offset, int prev_ofs) {
        int ofs;
        if ((prev_ofs != -1) && ((aux_input_offset - prev_ofs) > 0)) {
            ofs = aux_input_offset - prev_ofs;
            add_imm(reg_prev_bcast_addr, reg_prev_bcast_addr, ofs,
                    reg_tmp_imm);
        } else {
            ofs = aux_input_offset;
            add_imm(reg_prev_bcast_addr, aux_reg_inp, ofs, reg_tmp_imm);
        }

        ld1rqw(zreg_inp_s(jj), P_ALL_ONE, ptr(reg_prev_bcast_addr));
        prev_ofs = aux_input_offset;
        return prev_ofs;
    };

    auto wei_load = [=](int aux_kernel_offset, int reg_idx, int prev_ofs) {
        int ofs = aux_kernel_offset;

        if (kdnn_ldr_imm_check<int, isa>(ofs)) {
            add_imm(X_DEFAULT_ADDR, aux_reg_ker, ofs, X_TMP_0);
            ld1w(zreg_wei(reg_idx).s, P_ALL_ONE / T_z, ptr(X_DEFAULT_ADDR));
        } else {
            int ofs_tmp = ofs - prev_ofs;
            if ((prev_ofs != -1) && kdnn_ldr_imm_check<int, isa>(ofs_tmp)) {
                add_imm(X_DEFAULT_ADDR, reg_prev_wei_addr, ofs_tmp, X_TMP_0);
                ld1w(zreg_wei(reg_idx).s, P_ALL_ONE / T_z, ptr(X_DEFAULT_ADDR));
            } else {
                if ((prev_ofs != -1) && (ofs_tmp > 0)) {
                    ofs_tmp = aux_kernel_offset - prev_ofs;
                    add_imm(reg_prev_wei_addr, reg_prev_wei_addr, ofs_tmp,
                            reg_tmp_imm);
                } else {
                    add_imm(reg_prev_wei_addr, aux_reg_ker, ofs, reg_tmp_imm);
                }

                ld1w(zreg_wei(reg_idx).s, P_ALL_ONE / T_z,
                        ptr(reg_prev_wei_addr));
                prev_ofs = ofs;
            }
        }
        return prev_ofs;
    };

    align(32);
    L(kh_label);
    {
        int prev_bcast_ofs = -1;
        int prev_wei_ofs = -1;
        for (int ki = 0; ki < kw; ki++) {

            int jj_start = get_ow_start(ki, pad_l);
            int jj_end = get_ow_end(ur_w, ki, pad_r);

            int wei_reg_ofs = 30;
            for (int ic = 0; ic < ic_block; ic++) {
                if (ic_tail && ic >= ic_tail) {
                    // if src has only tails to compute, skip early
                    if (jcp.ic == ic_tail) {
                        break;
                    } else if (ic == ic_tail) {
                        cmp_imm(reg_channel, ic_tail, reg_tmp_imm);
                        b(EQ, ic_tail_jmp[ki]);
                    }
                }
                int wei_count = 0;
                for (int ii = 0; ii < nb_oc_block; ii++) {
                    int reg_idx = wei_reg_ofs + ii;
                    if (reg_idx >= 32) break;
                    int aux_kernel_offset = jcp.typesize_in
                            * (ii * jcp.nb_ic * jcp.kh * jcp.kw * jcp.kd
                                            * ic_block * oc_block
                                    + ki * ic_block * oc_block + ic * oc_block);

                    wei_count++;
                    if (jj_end - jj_start > 0) {
                        prev_wei_ofs = wei_load(aux_kernel_offset,
                                wei_reg_ofs + (ii % 2),
                                prev_wei_ofs);
                    }
                }

                if (ic % 4 == 0) {
                    for (int jj = jj_start; jj < jj_end; jj++) {
                        size_t aux_input_offset
                            = get_input_offset(ki, ic, jj, pad_l);
                        prev_bcast_ofs = bcast_load(jj,
                            aux_input_offset, prev_bcast_ofs);
                    }
                }

                for (int ii = 0; ii < nb_oc_block; ii++) {
                    int aux_kernel_offset = jcp.typesize_in
                            * ((ii + wei_count) * jcp.nb_ic * jcp.kh * jcp.kw
                                            * jcp.kd * ic_block * oc_block
                                    + ki * ic_block * oc_block + ic * oc_block);

                    for (int jj = jj_start; jj < jj_end; jj++)
                        fmla(zreg_out_s(jj, ii),
                            zreg_wei_s(wei_reg_ofs + (ii % 2)),
                            zreg_inp_idx(jj, ic % 4));

                    if ((jj_end - jj_start > 0)
                            && ((wei_count + ii) < nb_oc_block)) {
                        prev_wei_ofs = wei_load(aux_kernel_offset,
                                wei_reg_ofs + ((ii + wei_count) % 2),
                                prev_wei_ofs);
                    }
                }
            }
            L(ic_tail_jmp[ki]);
        }

        add_imm(aux_reg_ker, aux_reg_ker, shift_kernel_ptr, reg_tmp_imm);
        add_imm(aux_reg_inp, aux_reg_inp, shift_input_ptr, reg_tmp_imm);
        add_imm(aux_reg_inp2, aux_reg_inp, 0x100, reg_tmp_imm);
        add_imm(aux_reg_inp3, aux_reg_inp2, 0x100, reg_tmp_imm);
        sub(reg_kj, reg_kj, 1); //dec(reg_kj);
        cmp(reg_kj, 0);
        b(GT, kh_label);
    }

    if (jcp.ndims == 5) {
        add_imm(aux_reg_inp_d, aux_reg_inp_d,
                typesize * (jcp.dilate_d + 1) * jcp.ih * jcp.iw * inp_mul,
                reg_tmp_imm);
        const int ker_shift
                = typesize * jcp.kw * jcp.kh * jcp.oc_block * jcp.ic_block;
        add_imm(aux_reg_ker_d, aux_reg_ker_d, ker_shift, reg_tmp_imm);

        sub(reg_ki, reg_ki, 1); //dec(reg_ki);
        cmp(reg_ki, 0);
        b(GT, kd_label);

        if (icb_loop_in_compute_function) mov(aux_reg_ker_d, aux_reg_ker_d_org);
        mov(reg_out, reg_out_org);
    }
}

template <cpu_isa_t isa>
void kdnn_jit_sve_conv_fwd_kernel<isa>::compute_loop_fma_core(
        int ur_w, int pad_l, int pad_r) {
    int kw = jcp.kw;
    int ic_block = jcp.ic_block;
    int oc_block = jcp.oc_block;
    int nb_oc_block = jcp.nb_oc_blocking;
    const bool is_source_layout_nxc = is_src_layout_nxc();
    const bool icb_loop_in_compute_function = is_source_layout_nxc;
    const int ic_tail = jcp.ic_tail;

    Label kh_label, kd_label;

    std::vector<Label> ic_tail_jmp(kw);
    int shift_kernel_ptr
            = jcp.typesize_in * jcp.kw * jcp.oc_block * jcp.ic_block;
    int inp_mul = is_source_layout_nxc ? jcp.ngroups * jcp.ic
                                       : (!jcp.is_1stconv ? ic_block : 1);

    int shift_input_ptr
            = jcp.typesize_in * (jcp.dilate_h + 1) * jcp.iw * inp_mul;

    if (one_of(jcp.ndims, 3, 4)) {
        mov(aux_reg_inp, reg_inp);
        add_imm(aux_reg_inp2, aux_reg_inp, 0x100, reg_tmp_imm);
        add_imm(aux_reg_inp3, aux_reg_inp2, 0x100, reg_tmp_imm);
        mov(aux_reg_ker, reg_ker);
    }

    if (jcp.ndims == 5) {
        mov(reg_out_org, reg_out);
        ldr(reg_ki, ptr(kdnn_abi_param1, KUDNN_GET_OFF(kd_padding)));
        if (icb_loop_in_compute_function) {
            // need to continue with the same kernel pointer, but as
            // aux_reg_ker_d == reg_ker we need to save its value and restore
            // it after kd loop
            mov(aux_reg_ker_d_org, aux_reg_ker_d);
        } else {
            mov(aux_reg_ker_d, aux_reg_ker_d_org);
        }
        mov(aux_reg_inp_d, reg_inp);

        L(kd_label);
        ldr(reg_kj, ptr(kdnn_abi_param1, KUDNN_GET_OFF(kh_padding)));
    } else {
        mov(reg_kj, reg_kh);
    }

    if (jcp.ndims == 5) {
        mov(aux_reg_inp, aux_reg_inp_d);
        add_imm(aux_reg_inp2, aux_reg_inp, 0x100, reg_tmp_imm);
        add_imm(aux_reg_inp3, aux_reg_inp2, 0x100, reg_tmp_imm);
        mov(aux_reg_ker, aux_reg_ker_d);
    }

    auto zreg_inp_s = [=](int i_ic, int nb_x_blocking) {
        int idx = i_ic + nb_x_blocking * jcp.ur_w;
        assert(idx < 31);
        return ZRegS(idx);
    };
    auto zreg_out_s = [=](int i_ur, int i_oc) {
        int idx = i_ur + i_oc * jcp.ur_w;
        assert(idx < ker_reg_base_idx);
        return ZRegS(idx);
    };
    auto zreg_wei = [=](int idx) {
        assert(idx < 32);
        return ZReg(idx);
    };
    auto zreg_wei_s = [=](int idx) {
        assert(idx < 32);
        return ZRegS(idx);
    };

    auto bcast_load = [&](int jj, int nb_oc_block, int aux_input_offset,
                              int prev_ofs) {
        if (kdnn_ld1rw_imm_check(aux_input_offset)) {
            ld1rw(zreg_inp_s(jj, nb_oc_block), P_ALL_ONE,
                    ptr(aux_reg_inp, static_cast<int32_t>(aux_input_offset)));
        } else if (kdnn_ld1rw_imm_check(aux_input_offset - 0x100)) {
            ld1rw(zreg_inp_s(jj, nb_oc_block), P_ALL_ONE,
                    ptr(aux_reg_inp2,
                            static_cast<int32_t>(aux_input_offset - 0x100)));
        } else if (kdnn_ld1rw_imm_check(aux_input_offset - 0x200)) {
            ld1rw(zreg_inp_s(jj, nb_oc_block), P_ALL_ONE,
                    ptr(aux_reg_inp3,
                            static_cast<int32_t>(aux_input_offset - 0x200)));
        } else {
            if ((prev_ofs != -1)
                    && kdnn_ld1rw_imm_check(aux_input_offset - prev_ofs)) {

                ld1rw(zreg_inp_s(jj, nb_oc_block), P_ALL_ONE,
                        ptr(reg_prev_bcast_addr,
                                static_cast<int32_t>(
                                        aux_input_offset - prev_ofs)));
            } else {
                int ofs;
                if ((prev_ofs != -1) && ((aux_input_offset - prev_ofs) > 0)) {
                    ofs = aux_input_offset - prev_ofs;
                    add_imm(reg_prev_bcast_addr, reg_prev_bcast_addr, ofs,
                            reg_tmp_imm);
                } else {
                    ofs = aux_input_offset;
                    add_imm(reg_prev_bcast_addr, aux_reg_inp, ofs, reg_tmp_imm);
                }

                ld1rw(zreg_inp_s(jj, nb_oc_block), P_ALL_ONE,
                        ptr(reg_prev_bcast_addr));
                prev_ofs = aux_input_offset;
            }
        }

        return prev_ofs;
    };

    auto wei_load = [=](int aux_kernel_offset, int reg_idx, int prev_ofs) {
        int ofs = aux_kernel_offset;

        if (kdnn_ldr_imm_check<int, isa>(ofs)) {
            add_imm(X_DEFAULT_ADDR, aux_reg_ker, ofs, X_TMP_0);
            ld1w(zreg_wei(reg_idx).s, P_ALL_ONE / T_z, ptr(X_DEFAULT_ADDR));
        } else {
            int ofs_tmp = ofs - prev_ofs;
            if ((prev_ofs != -1) && kdnn_ldr_imm_check<int, isa>(ofs_tmp)) {
                add_imm(X_DEFAULT_ADDR, reg_prev_wei_addr, ofs_tmp, X_TMP_0);
                ld1w(zreg_wei(reg_idx).s, P_ALL_ONE / T_z, ptr(X_DEFAULT_ADDR));
            } else {
                if ((prev_ofs != -1) && (ofs_tmp > 0)) {
                    ofs_tmp = aux_kernel_offset - prev_ofs;
                    add_imm(reg_prev_wei_addr, reg_prev_wei_addr, ofs_tmp,
                            reg_tmp_imm);
                } else {
                    add_imm(reg_prev_wei_addr, aux_reg_ker, ofs, reg_tmp_imm);
                }

                ld1w(zreg_wei(reg_idx).s, P_ALL_ONE / T_z,
                        ptr(reg_prev_wei_addr));
                prev_ofs = ofs;
            }
        }
        return prev_ofs;
    };

    align(32);
    L(kh_label);
    {
        int prev_bcast_ofs = -1;
        int prev_wei_ofs = -1;
        for (int ki = 0; ki < kw; ki++) {

            int jj_start = get_ow_start(ki, pad_l);
            int jj_end = get_ow_end(ur_w, ki, pad_r);

            int wei_reg_ofs = nb_oc_block * jcp.ur_w;
            wei_reg_ofs += ur_w >= 16 ? 1 : jj_end;
            int num_regs4wei = 32 - wei_reg_ofs;
            for (int ic = 0; ic < ic_block; ic++) {
                if (ic_tail && ic >= ic_tail) {
                    // if src has only tails to compute, skip early
                    if (jcp.ic == ic_tail) {
                        break;
                    } else if (ic == ic_tail) {
                        cmp_imm(reg_channel, ic_tail, reg_tmp_imm);
                        b(EQ, ic_tail_jmp[ki]);
                    }
                }
                int wei_count = 0;
                for (int ii = 0; ii < nb_oc_block; ii++) {
                    int reg_idx = wei_reg_ofs + ii;
                    if (reg_idx >= 32) break;
                    int aux_kernel_offset = jcp.typesize_in
                            * (ii * jcp.nb_ic * jcp.kh * jcp.kw * jcp.kd
                                            * ic_block * oc_block
                                    + ki * ic_block * oc_block + ic * oc_block);

                    wei_count++;
                    if (jj_end - jj_start > 0) {
                        prev_wei_ofs = wei_load(aux_kernel_offset,
                                wei_reg_ofs + (ii % num_regs4wei),
                                prev_wei_ofs);
                    }
                }

                for (int jj = jj_start; jj < jj_end; jj++) {
                    size_t aux_input_offset
                            = get_input_offset(ki, ic, jj, pad_l);
                    prev_bcast_ofs = bcast_load(jj, nb_oc_block,
                            aux_input_offset, prev_bcast_ofs);
                }

                for (int ii = 0; ii < nb_oc_block; ii++) {
                    int aux_kernel_offset = jcp.typesize_in
                            * ((ii + wei_count) * jcp.nb_ic * jcp.kh * jcp.kw
                                            * jcp.kd * ic_block * oc_block
                                    + ki * ic_block * oc_block + ic * oc_block);

                    for (int jj = jj_start; jj < jj_end; jj++)
                        if (jcp.kernel_kind == kdnn_expl_bcast || jcp.kernel_kind == kdnn_embd_bcast) {
                            fmla(zreg_out_s(jj, ii), P_ALL_ONE,
                                zreg_inp_s(jj, nb_oc_block),
                                zreg_wei_s(wei_reg_ofs + (ii % num_regs4wei)));
                        } else {
                            assert(NULL);
                        }

                    if ((jj_end - jj_start > 0)
                            && ((wei_count + ii) < nb_oc_block)) {
                        prev_wei_ofs = wei_load(aux_kernel_offset,
                                wei_reg_ofs + ((ii + wei_count) % num_regs4wei),
                                prev_wei_ofs);
                    }
                }
            }
            L(ic_tail_jmp[ki]);
        }

        add_imm(aux_reg_ker, aux_reg_ker, shift_kernel_ptr, reg_tmp_imm);
        add_imm(aux_reg_inp, aux_reg_inp, shift_input_ptr, reg_tmp_imm);
        add_imm(aux_reg_inp2, aux_reg_inp, 0x100, reg_tmp_imm);
        add_imm(aux_reg_inp3, aux_reg_inp2, 0x100, reg_tmp_imm);
        sub(reg_kj, reg_kj, 1); //dec(reg_kj);
        cmp(reg_kj, 0);
        b(GT, kh_label);
    }

    if (jcp.ndims == 5) {
        add_imm(aux_reg_inp_d, aux_reg_inp_d,
                typesize * (jcp.dilate_d + 1) * jcp.ih * jcp.iw * inp_mul,
                reg_tmp_imm);
        const int ker_shift
                = typesize * jcp.kw * jcp.kh * jcp.oc_block * jcp.ic_block;
        add_imm(aux_reg_ker_d, aux_reg_ker_d, ker_shift, reg_tmp_imm);

        sub(reg_ki, reg_ki, 1); //dec(reg_ki);
        cmp(reg_ki, 0);
        b(GT, kd_label);

        if (icb_loop_in_compute_function) mov(aux_reg_ker_d, aux_reg_ker_d_org);
        mov(reg_out, reg_out_org);
    }
}

template <cpu_isa_t isa>
void kdnn_jit_sve_conv_fwd_kernel<isa>::compute_loop(
        int ur_w, int pad_l, int pad_r) {

    if (jcp.ndims == 5) mov(reg_oi_org, reg_oi);

    prepare_output(ur_w);

    Label skip_compute_loop;
    if (jcp.ndims == 5) {
        if ((jcp.dilate_d >= jcp.id)
                || (jcp.kd - 1) * (jcp.dilate_d + 1)
                        < nstl::max(jcp.f_pad, jcp.back_pad)) {
            ldr(reg_kj, ptr(kdnn_abi_param1, KUDNN_GET_OFF(kd_padding)));
            cmp(reg_kj, 0);
            b(LE, skip_compute_loop);
        }
    }
    if ((jcp.dilate_h >= jcp.ih)
            || (jcp.kh - 1) * (jcp.dilate_h + 1)
                    < nstl::max(jcp.t_pad, jcp.b_pad)) {
        ldr(reg_kj, ptr(kdnn_abi_param1, KUDNN_GET_OFF(kh_padding)));
        cmp(reg_kj, 0);
        b(LE, skip_compute_loop);
    }

    Label ic_loop;
    const bool generate_icb_loop = jcp.nb_ic > 1 && is_src_layout_nxc();
    if (generate_icb_loop) {
        mov(reg_inp_org, reg_inp);
        mov(reg_ker_org, reg_ker);

        ldr(reg_channel, ptr(kdnn_abi_param1, KUDNN_GET_OFF(reduce_work)));
        L(ic_loop);
    }

    if (jcp.ver == kdnn_ver_fma)
        if (jcp.is_1stconv && jcp.kernel_kind != kdnn_expl_bcast) {
            compute_loop_fma(ur_w, pad_l, pad_r);
        } else if (jcp.kernel_kind == kdnn_embd_bcast && jcp.nb_oc_blocking == 1) {
            compute_loop_fma(ur_w, pad_l, pad_r);
        } else if (jcp.kernel_kind == kdnn_cont_bcast) {
            compute_loop_fma_cont(ur_w, pad_l, pad_r);
        } else {
            compute_loop_fma_core(ur_w, pad_l, pad_r);
        }
    else
        assert(!"unknown convolution version");

    if (generate_icb_loop) {
        assert(is_src_layout_nxc());
        const int inp_shift = jcp.ic_block * jcp.typesize_in;
        add_imm(reg_inp, reg_inp, inp_shift, reg_tmp_imm);
        const int ker_shift = jcp.kd * jcp.kh * jcp.kw * jcp.ic_block
                * jcp.oc_block * jcp.typesize_in;
        add_imm(reg_ker, reg_ker, ker_shift, reg_tmp_imm);
        sub_imm(reg_channel, reg_channel, jcp.ic_block, reg_tmp_imm);
        cmp(reg_channel, 0);
        b(GT, ic_loop);
        mov(reg_ker, reg_ker_org);
        mov(reg_inp, reg_inp_org);
    }

    L(skip_compute_loop);
    store_output(ur_w);
    if (jcp.ndims == 5) mov(reg_oi, reg_oi_org);
}

template <cpu_isa_t isa>
void kdnn_jit_sve_conv_fwd_kernel<isa>::generate() {
    int iw = jcp.iw;
    int ow = jcp.ow;
    int ow_block = jcp.ow_block;
    int nb_ow = jcp.nb_ow;
    int kw = jcp.kw;
    int l_pad = jcp.l_pad;
    int ur_w = jcp.ur_w;
    int ur_w_tail = jcp.ur_w_tail;
    int stride_w = jcp.stride_w;

    int inp_mult = is_src_layout_nxc() ? jcp.ngroups * jcp.ic
                                       : (jcp.is_1stconv ? 1 : jcp.ic_block);
    int inp_shift_pad = jcp.typesize_in * (ur_w * stride_w - l_pad) * inp_mult;
    int inp_shift = jcp.typesize_in * ur_w * stride_w * inp_mult;
    int inp_shift_pad_second_block = -1 * jcp.typesize_in * l_pad * inp_mult;
    int out_shift = jcp.typesize_out * ur_w
            * (is_dst_layout_nxc() ? jcp.ngroups * jcp.oc : jcp.oc_block);
    
    const int simd_w_ = cpu_isa_traits<isa>::vlen / sizeof(float);

    preamble();

    //TO DO : renaming predicate register (P_ALL_ONE)
    if (simd_w_ != cpu_sveLen / sizeof(float))
        set_preg(P_ALL_ONE.s, simd_w_, X_TMP_0, X_TMP_1);

    ldr(reg_inp, ptr(kdnn_abi_param1, KUDNN_GET_OFF(src)));
    ldr(reg_out, ptr(kdnn_abi_param1, KUDNN_GET_OFF(dst)));
    ldr(reg_ker, ptr(kdnn_abi_param1, KUDNN_GET_OFF(filt)));
    ldr(reg_kh, ptr(kdnn_abi_param1, KUDNN_GET_OFF(kh_padding)));
    if (jcp.ndims == 5) mov(aux_reg_ker_d_org, reg_ker);

    int r_pad = nstl::max(0, jcp.r_pad);
    int n_oi = ow / ur_w;
    int r_pad1 = calculate_end_padding(l_pad, ur_w * n_oi, iw, stride_w,
            calculate_extended_filter_size(kw, jcp.dilate_w));
    int n_oi_last2 = n_oi - 1 > 0 ? n_oi - 1 : 0;
    int r_pad_last2 = calculate_end_padding(l_pad, ur_w * n_oi_last2, iw, stride_w,
        calculate_extended_filter_size(kw, jcp.dilate_w));
    if (!is_ow_threading_on(jcp)) { // nb_ow <= 1
        // nb_ow is # of output width blocks ??

        // ow is being processed as a whole - with left and right paddings
        // n_oi is # of output width blocks ??
        if (r_pad1 > 0) n_oi--;
        if (ow == ur_w) {
            ldr(reg_out_prf, ptr(kdnn_abi_param1, KUDNN_GET_OFF(dst_prf)));
            compute_loop(ur_w, l_pad, r_pad);
        } else {
            mov(reg_out_prf, reg_out);
            if (n_oi == 0) {
                add_imm(reg_out_prf, reg_out_prf, out_shift, reg_tmp_imm);
                compute_loop(ur_w, l_pad, r_pad1);
                add_imm(reg_out, reg_out, out_shift, reg_tmp_imm);
                if (ur_w_tail != 0) {
                    add_imm(reg_out_prf, reg_out_prf, out_shift, reg_tmp_imm);
                    int l_pad1 = l_pad - ur_w * stride_w;
                    if (l_pad1 > 0) {
                        compute_loop(ur_w_tail, l_pad1, r_pad);
                    } else {
                        add_imm(reg_inp, reg_inp, inp_shift_pad, reg_tmp_imm);
                        compute_loop(ur_w_tail, 0, r_pad);
                    }
                }
            } else if (l_pad <= ur_w && r_pad_last2 == 0) {
                mov(reg_oi, 0);
                if (l_pad > 0) {
                    add_imm(reg_out_prf, reg_out_prf, out_shift, reg_tmp_imm);
                    compute_loop(ur_w, l_pad, 0);
                    add_imm(reg_inp, reg_inp, inp_shift_pad, reg_tmp_imm);
                    add_imm(reg_out, reg_out, out_shift, reg_tmp_imm);
                    add_imm(reg_oi, reg_oi, 1, reg_tmp_imm); // increment
                }
                if ((l_pad <= 0 && n_oi > 0) || (l_pad > 0 && n_oi > 1)) {
                    Label ow_loop_label;
                    L(ow_loop_label);
                    {
                        add_imm(reg_out_prf, reg_out_prf, out_shift,
                                reg_tmp_imm);
                        compute_loop(ur_w, 0, 0);

                        add_imm(reg_inp, reg_inp, inp_shift, reg_tmp_imm);
                        add_imm(reg_out, reg_out, out_shift, reg_tmp_imm);
                        add_imm(reg_oi, reg_oi, 1, reg_tmp_imm); //inc(reg_oi);
                        cmp_imm(reg_oi, n_oi, reg_tmp_imm);

                        b(LT, ow_loop_label);
                    }
                }
                if (r_pad1 > 0) {
                    add_imm(reg_out_prf, reg_out_prf, out_shift, reg_tmp_imm);
                    compute_loop(ur_w, 0, r_pad1);
                    add_imm(reg_inp, reg_inp, inp_shift, reg_tmp_imm);
                    add_imm(reg_out, reg_out, out_shift, reg_tmp_imm);
                }
                if (ur_w_tail != 0) {
                    add_imm(reg_out_prf, reg_out_prf, out_shift, reg_tmp_imm);
                    compute_loop(ur_w_tail, 0, r_pad);
                }
            }
            else {
                n_oi = r_pad1 > 0 ? n_oi : n_oi - 1;
                int inp_shift_pad1 = 0;
                int pre_inp_shift_pad1 = 0;
                for (int oi = 1; oi <= n_oi + 1; oi++) {
                    add_imm(reg_out_prf, reg_out_prf, out_shift, reg_tmp_imm);
                    int l_pad_l = l_pad - (oi - 1) * ur_w * stride_w;
                    int l_pad_r = l_pad_l > 0 ? l_pad_l : 0;
                    int r_pad_l = calculate_end_padding(l_pad, oi * ur_w, iw, stride_w,
                        calculate_extended_filter_size(kw, jcp.dilate_w));
                    int r_pad_r = r_pad_l > 0 ? r_pad_l : 0;
                    compute_loop(ur_w, l_pad_r, r_pad_r);
                    if (l_pad - oi * ur_w * stride_w < 0) {
                        inp_shift_pad1 = jcp.typesize_in * (oi * ur_w * stride_w - l_pad) * inp_mult;
                        add_imm(reg_inp, reg_inp, inp_shift_pad1 - pre_inp_shift_pad1, reg_tmp_imm);
                        pre_inp_shift_pad1 = inp_shift_pad1;
                    }
                    add_imm(reg_out, reg_out, out_shift, reg_tmp_imm);
                    add_imm(reg_oi, reg_oi, 1, reg_tmp_imm); // increment
                }
                if (ur_w_tail != 0) {
                    add_imm(reg_out_prf, reg_out_prf, out_shift, reg_tmp_imm);
                    int l_pad_m = l_pad - (n_oi + 1) * ur_w * stride_w;
                    int l_pad_n = l_pad_m > 0 ? l_pad_m : 0;
                    compute_loop(ur_w_tail, l_pad_n, r_pad);
                }
            }
        }
    } else {
        // ow block is only processed.
        // Number of block is passed as parameter owb,
        // and padding processing depends on this number.

        Label end_label, last_oi_label, middle_ow_blocks_label, tail_label;
        Label oi_loop_label, oi_loop_start_label, oi_loop_end_label;

        assert(ow_block % ur_w == 0);
        int n_oi_not_last_ow_block = ow_block / ur_w;
        // to simplify code (and general regs usage),
        // size of ow block must be >= 2 * ur_w
        assert(n_oi_not_last_ow_block > 1);
        int n_oi_next_last_ow_block = n_oi_not_last_ow_block;
        int n_oi_first_ow_block = n_oi_not_last_ow_block;

        int n_oi_last_ow_block = (ow - ow_block * (nb_ow - 1)) / ur_w;

        // prepare right padding
        bool next_last_ow_block_padded = r_pad1 > 0 && n_oi_last_ow_block == 0;
        bool first_ow_block_padded
                = next_last_ow_block_padded && jcp.nb_ow == 2;
        bool last_ow_block_padded = r_pad1 > 0 && n_oi_last_ow_block > 0;

        if (last_ow_block_padded)
            n_oi_last_ow_block--;
        else if (first_ow_block_padded)
            n_oi_first_ow_block--;
        else if (next_last_ow_block_padded)
            n_oi_next_last_ow_block--;

        ldr(reg_owb, ptr(kdnn_abi_param1, KUDNN_GET_OFF(owb)));
        cmp(reg_owb, 0); // is that the first ow-block ?
        b(GT, middle_ow_blocks_label);

        // the first ow block, compute left padding

        mov(reg_oi, n_oi_first_ow_block);
        mov(reg_out_prf, reg_out);

        if (l_pad > 0) {
            add_imm(reg_out_prf, reg_out_prf, out_shift, reg_tmp_imm);
            compute_loop(ur_w, l_pad, 0);
            add_imm(reg_inp, reg_inp, inp_shift_pad, reg_tmp_imm);
            add_imm(reg_out, reg_out, out_shift, reg_tmp_imm);
            sub(reg_oi, reg_oi, 1); // decrement
            cmp(reg_oi, 0);
        }
        b(oi_loop_label);

        // middle or last ow block entry

        L(middle_ow_blocks_label);

        if (l_pad > 0) {
            // just to consider left padding, not compute
            add_imm(reg_inp, reg_inp, inp_shift_pad_second_block, reg_tmp_imm);
        }

        // set number of iteration for oi-loop
        cmp_imm(reg_owb, jcp.nb_ow - 1, reg_tmp_imm); // last ow-block ?
        mov(reg_oi, n_oi_last_ow_block);
        b(EQ, oi_loop_label);
        cmp_imm(reg_owb, jcp.nb_ow - 2, reg_tmp_imm); // next to last ow-block ?
        mov(reg_oi, n_oi_next_last_ow_block);
        b(EQ, oi_loop_label);
        mov(reg_oi, n_oi_not_last_ow_block); // other middle ow-blocks

        // oi loop w/o padding
        L(oi_loop_label);
        L(oi_loop_start_label);
        cmp(reg_oi, 0);
        b(LE, oi_loop_end_label);

        add_imm(reg_out_prf, reg_out_prf, out_shift, reg_tmp_imm);
        compute_loop(ur_w, 0, 0);
        add_imm(reg_inp, reg_inp, inp_shift, reg_tmp_imm);
        add_imm(reg_out, reg_out, out_shift, reg_tmp_imm);
        sub(reg_oi, reg_oi, 1); // dec(reg_oi);
        cmp(reg_oi, 0);
        b(oi_loop_start_label);
        L(oi_loop_end_label);

        ldr(reg_owb, ptr(kdnn_abi_param1, KUDNN_GET_OFF(owb)));

        cmp(reg_owb, 0); // first ow-block ?
        if (first_ow_block_padded) {
            b(EQ, last_oi_label);
        } else {
            b(EQ, end_label);
        }
        cmp_imm(reg_owb, jcp.nb_ow - 2, reg_tmp_imm); // next to last ow-block ?
        b(LT, end_label);
        if (next_last_ow_block_padded) {
            b(EQ, last_oi_label);
        } else {
            b(EQ, end_label);
        }
        // that is last block
        if (!last_ow_block_padded) { b(tail_label); }

        // last oi block with right padding
        L(last_oi_label);
        add_imm(reg_out_prf, reg_out_prf, out_shift, reg_tmp_imm);
        compute_loop(ur_w, 0, r_pad1);
        add_imm(reg_inp, reg_inp, inp_shift, reg_tmp_imm);
        add_imm(reg_out, reg_out, out_shift, reg_tmp_imm);

        ldr(reg_owb, ptr(kdnn_abi_param1, KUDNN_GET_OFF(owb)));
        cmp_imm(reg_owb, jcp.nb_ow - 1, reg_tmp_imm); // last ow_block?
        b(LT, end_label);

        L(tail_label);
        if (ur_w_tail != 0) {
            add_imm(reg_out_prf, reg_out_prf, out_shift, reg_tmp_imm);
            compute_loop(ur_w_tail, 0, r_pad);
        }
        L(end_label);
    }
    postamble();

    if (jcp.with_eltwise) { eltwise_injector_->prepare_table(); }

}

template <cpu_isa_t isa>
bool kdnn_jit_sve_conv_fwd_kernel<isa>::post_ops_ok(
        kdnn_jit_conv_conf_t &jcp, const primitive_attr_t &attr) {
    const auto &p = attr.post_ops_;

    auto is_eltwise = [&](int idx) { return p.entry_[idx].is_eltwise(); };
    auto is_sum = [&](int idx) { return p.entry_[idx].is_sum(); };

    switch (p.len()) {
        case 0: return true; // no post_ops
        case 1: return is_eltwise(0) || is_sum(0); // sum OR eltwise
        case 2: return is_sum(0) && is_eltwise(1); // sum -> eltwise
        default: return false;
    }

    return false;
}

template <cpu_isa_t isa>
status_t kdnn_jit_sve_conv_fwd_kernel<isa>::init_conf(kdnn_jit_conv_conf_t &jcp,
        const convolution_desc_t &cd, memory_desc_t &src_md,
        memory_desc_t &weights_md, memory_desc_t &dst_md,
        memory_desc_t &bias_md, const primitive_attr_t &attr, int nthreads) {
    using namespace prop_kind;

    if (!mayiuse(isa)) { return status::unimplemented; }

    const memory_desc_wrapper src_d(&src_md);
    const memory_desc_wrapper weights_d(&weights_md);
    const memory_desc_wrapper dst_d(&dst_md);
    const memory_desc_wrapper bias_d(&bias_md);

    const int regs = 28;
    const bool with_groups = weights_d.ndims() == src_d.ndims() + 1;
    int ndims = src_d.ndims();

    jcp = zero<decltype(jcp)>();
    jcp.nthr = jcp.aligned_threads = nthreads;
    jcp.ndims = ndims;
    jcp.prop_kind = cd.prop_kind;
    jcp.ngroups = with_groups ? weights_d.dims()[0] : 1;
    jcp.mb = src_d.dims()[0];
    jcp.oc = dst_d.dims()[1] / jcp.ngroups;
    jcp.oc_without_padding = jcp.oc;
    jcp.ic = src_d.dims()[1] / jcp.ngroups;
    jcp.ic_without_padding = jcp.ic;
    jcp.id = (ndims == 5) ? src_d.dims()[2] : 1;
    jcp.ih = (ndims == 3) ? 1 : src_d.dims()[ndims - 2];
    jcp.iw = src_d.dims()[ndims - 1];
    jcp.od = (ndims == 5) ? dst_d.dims()[2] : 1;
    jcp.oh = (ndims == 3) ? 1 : dst_d.dims()[ndims - 2];
    jcp.ow = dst_d.dims()[ndims - 1];
    jcp.kd = (ndims == 5) ? weights_d.dims()[with_groups + 2] : 1;
    jcp.kh = (ndims == 3) ? 1 : weights_d.dims()[with_groups + ndims - 2];
    jcp.kw = weights_d.dims()[with_groups + ndims - 1];
    jcp.f_pad = (ndims == 5) ? cd.padding[0][0] : 0;
    jcp.t_pad = (ndims == 3) ? 0 : cd.padding[0][ndims - 4];
    jcp.l_pad = cd.padding[0][ndims - 3];
    jcp.stride_d = (ndims == 5) ? cd.strides[0] : 1;
    jcp.stride_h = (ndims == 3) ? 1 : cd.strides[ndims - 4];
    jcp.stride_w = cd.strides[ndims - 3];

    jcp.dilate_d = (ndims == 5) ? cd.dilates[0] : 0;
    jcp.dilate_h = (ndims == 3) ? 0 : cd.dilates[ndims - 4];
    jcp.dilate_w = cd.dilates[ndims - 3];

    int ext_kw = calculate_extended_filter_size(jcp.kw, jcp.dilate_w);
    int ext_kh = calculate_extended_filter_size(jcp.kh, jcp.dilate_h);
    int ext_kd = calculate_extended_filter_size(jcp.kd, jcp.dilate_d);
    jcp.r_pad = calculate_end_padding(
            jcp.l_pad, jcp.ow, jcp.iw, jcp.stride_w, ext_kw);
    jcp.b_pad = calculate_end_padding(
            jcp.t_pad, jcp.oh, jcp.ih, jcp.stride_h, ext_kh);
    jcp.back_pad = calculate_end_padding(
            jcp.f_pad, jcp.od, jcp.id, jcp.stride_d, ext_kd);

    const auto dat_tag_nxc = pick(ndims - 3, nwc, nhwc, ndhwc);
    const auto dat_tag_ncx = pick(ndims - 3, ncw, nchw, ncdhw);
    const auto dat_tag_nCx8c = pick(ndims - 3, nCw8c, nChw8c, nCdhw8c);
    const auto dat_tag_nCx16c = pick(ndims - 3, nCw16c, nChw16c, nCdhw16c);
    auto curr_src_tag = src_d.matches_one_of_tag(
            dat_tag_nxc, dat_tag_nCx16c, dat_tag_nCx8c, dat_tag_ncx);
    auto curr_dst_tag = dst_d.matches_one_of_tag(
            dat_tag_nxc, dat_tag_nCx16c, dat_tag_nCx8c);
    bool is_data_layout_nxc
            = utils::everyone_is(dat_tag_nxc, curr_src_tag, curr_dst_tag);

    /* 1st convolution check */
    jcp.is_1stconv = is_1stconv(jcp);

    /* Padding check (Channel) */
    bool ok_to_pad_channels
            = true && jcp.ngroups == 1 && src_d.data_type() == data_type::f32;

    int full_simd_w = cpu_isa_traits<isa>::vlen / typesize;
    jcp.simd_w = full_simd_w;
    jcp.oc_block = jcp.simd_w;
    jcp.ic_block = jcp.is_1stconv ? jcp.ic : jcp.simd_w;

    /* Channel padding */
    if (ok_to_pad_channels) {
        jcp.oc = rnd_up(jcp.oc, jcp.oc_block);
        jcp.ic = rnd_up(jcp.ic, jcp.ic_block);
    }

    /* Input and output channels must be multiples of simd_w */
    if (!(jcp.oc % jcp.oc_block == 0 && jcp.ic % jcp.ic_block == 0)) {
        return status::unimplemented;
    }
    jcp.ic_tail = 0;
    jcp.oc_tail = 0;

    /* Post operation check */
    if (!post_ops_ok(jcp, attr)) { return status::unimplemented; }

    /* Eltwise operation check */
    const auto &p = attr.post_ops_;
    jcp.with_sum = p.find(primitive_kind::sum) != -1;
    const int eltwise_ind = p.find(primitive_kind::eltwise);
    jcp.with_eltwise = eltwise_ind != -1;
    if (jcp.with_eltwise) {
       jcp.eltwise = p.entry_[eltwise_ind].eltwise;
       if (!kdnn_eltwise_injector::is_supported(isa, jcp.eltwise.alg))
           return status::unimplemented;
       if (dst_d.data_type() == data_type::s32) return status::unimplemented;
    }

    format_tag_t src_tag, dst_tag, wei_tag;

    switch (isa) {
        case sve_512:
            dst_tag = dat_tag_nCx16c;
            src_tag = jcp.is_1stconv ? dat_tag_ncx : dat_tag_nCx16c;
            wei_tag = pick(2 * ndims - 6 + with_groups, OIw16i16o, gOIw16i16o,
                    OIhw16i16o, gOIhw16i16o, OIdhw16i16o, gOIdhw16i16o);
            break;
        case sve_256:
            // dst_tag = is_data_layout_nxc ? dat_tag_nxc : dat_tag_nCx8c;
            // src_tag = is_data_layout_nxc
            //         ? dat_tag_nxc
            //         : jcp.is_1stconv ? dat_tag_ncx : dat_tag_nCx8c;
            dst_tag = dat_tag_nxc;
            src_tag = dat_tag_nxc;
            wei_tag = pick(2 * ndims - 6 + with_groups, OIw8i8o, gOIw8i8o,
                    OIhw8i8o, gOIhw8i8o, OIdhw8i8o, gOIdhw8i8o);
            break;
        default: break;
    }

    if (src_md.format_kind == format_kind::any)
        CHECK(memory_desc_init_by_tag(src_md, src_tag));
    else if (curr_src_tag != src_tag)
        return status::unimplemented;
    jcp.src_tag = src_tag;

    if (dst_md.format_kind == format_kind::any)
        CHECK(memory_desc_init_by_tag(dst_md, dst_tag));
    else if (curr_dst_tag != dst_tag)
        return status::unimplemented;
    jcp.dst_tag = dst_tag;

    jcp.with_bias = cd.bias_desc.format_kind != format_kind::undef;
    if (jcp.with_bias) {
        if (bias_d.format_kind() == format_kind::any)
            CHECK(memory_desc_init_by_tag(bias_md, x));
    }

    if (mayiuse(isa) && src_d.data_type() == data_type::f32
            && weights_d.data_type() == data_type::f32
            && dst_d.data_type() == data_type::f32) {
        jcp.ver = kdnn_ver_fma;
        jcp.typesize_in = typesize;
        jcp.typesize_out = typesize;

        if (jcp.is_1stconv) {
            switch (isa) {
                case sve_512:
                    wei_tag = with_groups
                            ? pick(ndims - 3, gOwi16o, gOhwi16o, gOdhwi16o)
                            : pick(ndims - 3, Owi16o, Ohwi16o, Odhwi16o);
                    break;
                case sve_256:
                    wei_tag = with_groups
                            ? pick(ndims - 3, gOwi8o, gOhwi8o, gOdhwi8o)
                            : pick(ndims - 3, Owi8o, Ohwi8o, Odhwi8o);
                    break;
                default: break;
            }
        }
    } else {
        return status::unimplemented;
    }

    if (init_tag(jcp.wei_tag, weights_md, weights_d, wei_tag)
            != status::success)
        return status::unimplemented;

    jcp.ur_w = nstl::min(jcp.ow, regs); // ur_w is min(output width, regs=28)

    int n_oi = (jcp.ow / jcp.ur_w);
    int r_pad = calculate_end_padding(
            jcp.l_pad, jcp.ur_w * n_oi, jcp.iw, jcp.stride_w, ext_kw);
    if (jcp.l_pad > 0 && r_pad > 0) n_oi--;

    /* Grouped channel offset to support 'non-blocked data' format for
     * convolution sizes with '(input_channel / ngroups) < simd' */
    jcp.nonblk_group_off
            = (jcp.ngroups > 1 && one_of(jcp.src_tag, ncw, nchw, ncdhw))
            ? jcp.ic
            : 1;

    jcp.nb_ic = div_up(jcp.ic, jcp.ic_block);
    jcp.nb_oc = div_up(jcp.oc, jcp.oc_block);
    jcp.nb_ic_blocking = jcp.nb_oc_blocking = 1;

    jcp.ow_block = jcp.ow;

    auto get_thr_eff = [=](int nb_oc_blocking, int ow_block) {
        int nb_ow = div_up(jcp.ow, ow_block);
        int nb_oc_chunks = div_up(jcp.nb_oc, nb_oc_blocking);
        int work_amount = jcp.mb * jcp.oh * nb_oc_chunks * nb_ow;
        float disbalance = (float)jcp.ow / rnd_up(jcp.ow, ow_block);
        float thr_eff = disbalance * (float)work_amount
                / rnd_up(work_amount, jcp.nthr);
        return thr_eff;
    };

    auto get_ow_block = [=](int nb_oc_blocking, int ur_w, float &eff) {
        int res_ow_block = jcp.ow;
        eff = get_thr_eff(nb_oc_blocking, res_ow_block);

        return res_ow_block;
    };

    const unsigned int L1_cache_size = platform::get_per_core_cache_size(1);
    if (jcp.ver == kdnn_ver_fma && mayiuse(isa)) {
        int try_nb_oc_blocking = 2;
        unsigned int ker_inp_size = typesize * div_up(jcp.iw, jcp.stride_w)
                * jcp.ic_block * jcp.kh * jcp.kd;
        unsigned int ker_out_size
                = typesize * jcp.ow * jcp.oc_block * try_nb_oc_blocking;
        unsigned int ker_wei_size = typesize * jcp.kh * jcp.kw * jcp.ic_block
                * jcp.oc_block * try_nb_oc_blocking * jcp.kd;
        unsigned int ker_total_size
                = ker_inp_size + ker_out_size + ker_wei_size;

        bool embd_bcast_condition_base = true
                && (jcp.kw == 3 && jcp.ow <= 28
                        && ker_total_size < L1_cache_size)
                && !(jcp.kw == 3 && jcp.ow == 13 && jcp.ic >= 192)
                && !(jcp.kw == 3 && jcp.ow == 28 && jcp.ic >= 512);

        // These conditions define a set of shapes with 'ow = 1' which
        // have a very limited optimization space for performance. Try
        // to optimize by using a larger 'nb_oc_blocking' size.
        bool expl_bcast_condition
                = everyone_is(1, jcp.ngroups, jcp.mb, jcp.stride_h, jcp.ow,
                          jcp.stride_w, jcp.id, jcp.od, jcp.kd, jcp.stride_d)
                && jcp.iw == jcp.kw && jcp.nb_oc > 1
                && everyone_is(0, jcp.l_pad, jcp.r_pad, jcp.dilate_w, jcp.f_pad,
                        jcp.back_pad, jcp.dilate_d)
                && jcp.oh >= 60 && jcp.kh >= 3;

        bool embd_bcast_condition = !expl_bcast_condition
                && (jcp.kw > 3
                        || (jcp.stride_w == 1 && jcp.stride_h == 1
                                && embd_bcast_condition_base)
                        || ((jcp.stride_w != 1 || jcp.stride_h != 1)
                                && ((jcp.mb <= 16
                                        && (jcp.oc <= 192 || jcp.oh <= 10)
                                        && embd_bcast_condition_base)))
                        || (jcp.mb == 1
                                && (jcp.ur_w >= jcp.ow || jcp.is_1stconv
                                        || (jcp.ow <= 147 && jcp.oc <= 96))));
        // unused now
        (void)(embd_bcast_condition);

        if (jcp.mb == 1) {
            unsigned int inp_size = jcp.mb * div_up(jcp.ih, jcp.stride_h)
                    * div_up(jcp.iw, jcp.stride_w) * jcp.ic;
            unsigned int wei_size = jcp.ic * jcp.oc * jcp.kh * jcp.kw;

            // Estimate whether we need to limit the number of threads
            // and calculate this number. Includes some heuristic.
            int oc_chunks = jcp.nb_oc / jcp.nb_oc_blocking;
            int work_amount = jcp.mb * jcp.ngroups * oc_chunks * jcp.oh;
            int job_size_min = work_amount / nthreads;
            int job_size_max = div_up(work_amount, nthreads);
            int ch_max = rnd_up(jcp.oh, job_size_max);
            int ch_min = (job_size_min == 0) ? jcp.oh
                                             : rnd_up(jcp.oh, job_size_min);
            bool not_aligned_max = ch_max % jcp.oh != 0 && ch_max / jcp.oh < 2
                    && (jcp.oh != 8 || ch_max / jcp.oh > 1);
            bool not_aligned_min = ch_min % jcp.oh != 0 && ch_min / jcp.oh < 2
                    && (jcp.oh != 8 || ch_min / jcp.oh > 1);
            bool eligible_case = (jcp.stride_h == 1 && jcp.stride_w == 1)
                    || nthreads > oc_chunks;
            if (jcp.loop_order == kdnn_loop_cgn && oc_chunks > 1 && nthreads > 1
                    && wei_size / inp_size > 24
                    && (not_aligned_max || not_aligned_min) && eligible_case) {
                // Try to find number of threads > nthreads / 2 such that
                // oc_chunks is a multiple of nthreads, or nthreads is a
                // multiple of oc_chunks. Otherwise, keep default value.
                // TODO: implement a task-based alternative without throttling.
                jcp.aligned_threads = jcp.nthr;
                for (int i = jcp.nthr; i > jcp.nthr / 2; i--) {
                    if (oc_chunks % i == 0 || i % oc_chunks == 0) {
                        jcp.aligned_threads = i;
                        break;
                    }
                }
            }
        }

        int max_nb_oc = 5;
        // Multi oc block performance is better
        // if (embd_bcast_condition) {
        //     jcp.kernel_kind = kdnn_embd_bcast;
        //     jcp.ur_w = nstl::min(jcp.ow, 15);
        //     jcp.nb_ic_blocking = jcp.nb_oc_blocking = 1;
        //     const unsigned int L1_cache_size
        //             = platform::get_per_core_cache_size(1);
        //     if (ker_total_size < L1_cache_size && jcp.ow <= 8 && jcp.kh <= 3
        //             && jcp.kw <= 3 && jcp.nb_oc % try_nb_oc_blocking == 0
        //             && IMPLICATION(jcp.is_1stconv, jcp.mb == 1)
        //             && IMPLICATION(jcp.mb == 1, jcp.ur_w < jcp.ow)) {
        //         jcp.nb_oc_blocking = try_nb_oc_blocking;
        //         jcp.ur_w = nstl::min(jcp.ow, 31 / (jcp.nb_oc_blocking + 1));
        //     }
        // } else {
            if (jcp.id > 1) {
                max_nb_oc = 2;
            }
            jcp.kernel_kind = kdnn_expl_bcast;
            jcp.nb_ic_blocking = 1;
            if (IMPLICATION(jcp.is_1stconv, jcp.mb >= 1)
                    || expl_bcast_condition) {
                int best_nb_oc_blocking = 1;
                for (int i = nstl::min(jcp.nb_oc, max_nb_oc); i > 0; i--) {
                    if (jcp.nb_oc % i == 0) {
                        best_nb_oc_blocking = i;
                        break;
                    }
                }
                jcp.nb_oc_blocking = best_nb_oc_blocking;
                jcp.ur_w = nstl::min(jcp.ow, 31 / (jcp.nb_oc_blocking + 1));
                if (jcp.l_pad > jcp.ur_w) {
                    jcp.nb_oc_blocking = 1;
                    jcp.ur_w = nstl::min(jcp.ow, 31 / (jcp.nb_oc_blocking + 1));
                }
                if (!jcp.is_1stconv && jcp.ur_w <= 8) { jcp.kernel_kind = kdnn_cont_bcast; }
            // }
        }
    }
    jcp.ur_w_tail = jcp.ow % jcp.ur_w;

    bool args_ok = true
            && jcp.ic <= src_d.padded_dims()[1]
            && jcp.oc <= dst_d.padded_dims()[1]
            && jcp.ic <= weights_d.padded_dims()[with_groups + 1]
            && jcp.oc <= weights_d.padded_dims()[with_groups + 0];
    if (!args_ok) return status::unimplemented;

    pick_loop_order(jcp);

    jcp.nb_ic_L2 = jcp.nb_ic;

    float thr_eff;
    jcp.ow_block = get_ow_block(jcp.nb_oc_blocking, jcp.ur_w, thr_eff);
    jcp.nb_ow = div_up(jcp.ow, jcp.ow_block);

    const int L2_size = platform::get_per_core_cache_size(2) / sizeof(float);

    // Source and output data needs to fit in L2,
    // leaving some space for weights and prefetching.
    int h_L2 = int(((0.6f * L2_size) / jcp.simd_w
                           - nstl::min(0, jcp.kh - jcp.stride_h) * jcp.iw)
            / (jcp.stride_h * jcp.iw + jcp.ow));
    jcp.h_blocking = nstl::max(1, nstl::min(jcp.oh, h_L2));

    if (is_data_layout_nxc) {
        // TODO: improve L2 blocking for large IC
        const int nb_ic_theshold_L2 = 32;
        if (jcp.nb_ic > nb_ic_theshold_L2 && jcp.nb_ic < 2 * nb_ic_theshold_L2)
            jcp.nb_ic_L2 = div_up(jcp.nb_ic, 2);
        else
            jcp.nb_ic_L2 = nstl::min(nb_ic_theshold_L2, jcp.nb_ic);
    }
    return status::success;
}

template <cpu_isa_t isa>
void kdnn_jit_sve_conv_fwd_kernel<isa>::init_scratchpad(
        memory_tracking::registrar_t &scratchpad, const kdnn_jit_conv_conf_t &jcp) {

    if (jcp.with_bias && jcp.oc != jcp.oc_without_padding)
        scratchpad.book(key_conv_padded_bias, jcp.oc, jcp.typesize_out);
}

/*struct instantiation*/
template struct kdnn_jit_sve_conv_fwd_kernel<sve_256>;

template <cpu_isa_t isa>
void kdnn_jit_sve_conv_bwd_data_kernel<isa>::prepare_output(int ur_w) {
    auto zreg_out_s = [=](int i_ur, int i_oc) {
        int idx = i_ur + i_oc * jcp.ur_w;
        assert(idx < ker_reg_base_idx);
        return ZRegS(idx);
    };

    long long int prev_ofs = 0;
    for (int k = 0; k < jcp.nb_ic_blocking; k++) {
        for (int j = 0; j < ur_w; j++) {
            ZRegS zreg = zreg_out_s(j, k);
            fmov(zreg);
            size_t aux_src_offset = (size_t)typesize
                    * ((size_t)k * jcp.ih * jcp.iw * jcp.id + j) * jcp.ic_block;

            std::string op = "LD";
            prev_ofs = prefetch(op, 2, reg_src_prf, aux_src_offset, prev_ofs);
        }
    }
}

template <cpu_isa_t isa>
void kdnn_jit_sve_conv_bwd_data_kernel<isa>::store_output(int ur_w) {

    int num_used_zreg = 32 - ker_reg_base_idx;

    auto zreg_tmp = [=](int idx) {
        int zreg_idx = (idx % num_used_zreg) + ker_reg_base_idx;
        return ZReg(zreg_idx);
    };

    auto zreg_tmp_s = [=](int idx) {
        int zreg_idx = (idx % num_used_zreg) + ker_reg_base_idx;
        return ZRegS(zreg_idx);
    };

    auto zreg_out = [=](int i_ur, int i_oc) {
        int idx = i_ur + i_oc * jcp.ur_w;
        assert(idx < ker_reg_base_idx);
        return ZReg(idx);
    };

    auto zreg_out_s = [=](int i_ur, int i_oc) {
        int idx = i_ur + i_oc * jcp.ur_w;
        assert(idx < ker_reg_base_idx);
        return ZRegS(idx);
    };

    auto out_load = [=](int aux_output_offset, int idx, int prev_ofs) {
        int ofs = aux_output_offset;
        if ((kdnn_VL_OFS(ofs, isa) < LDRMAX) && (kdnn_VL_OFS(ofs, isa) >= (-1 * LDRMAX))
                && ((ofs & 0x3f) == 0)) {
            ldr(zreg_tmp(idx), ptr(reg_src, static_cast<int32_t>(kdnn_VL_OFS(ofs, isa))));
        } else {
            int tmp_ofs = aux_output_offset - prev_ofs;

            if (((tmp_ofs & 0x3f) == 0) && (kdnn_VL_OFS(tmp_ofs, isa) < LDRWMAX)
                    && (tmp_ofs >= 0)) {
                ldr(zreg_tmp(idx),
                        ptr(reg_tmp_addr,
                                static_cast<int32_t>(kdnn_VL_OFS(tmp_ofs, isa))));
            } else {
                add_imm(reg_tmp_addr, reg_src, ofs, reg_tmp_imm);
                ldr(zreg_tmp(idx), ptr(reg_tmp_addr));
                prev_ofs = ofs;
            }
        }
        return prev_ofs;
    };

    auto out_str = [=](int j, int k, int aux_output_offset, int prev_ofs) {
        int ofs = aux_output_offset;

        if ((kdnn_VL_OFS(ofs, isa) < LDRMAX) && (kdnn_VL_OFS(ofs, isa) >= (-1 * LDRMAX))
                && ((ofs & 0x3f) == 0)) {
            str(zreg_out(j, k),
                    ptr(reg_src, static_cast<int32_t>(kdnn_VL_OFS(ofs, isa))));
        } else {
            int tmp_ofs = aux_output_offset - prev_ofs;

            if (((tmp_ofs & 0x3f) == 0) && (kdnn_VL_OFS(tmp_ofs, isa) < LDRWMAX)
                    && (tmp_ofs >= 0)) {
                str(zreg_out(j, k),
                        ptr(reg_tmp_addr,
                                static_cast<int32_t>(kdnn_VL_OFS(tmp_ofs, isa))));
            } else {
                add_imm(reg_tmp_addr, reg_src, ofs, reg_tmp_imm);
                str(zreg_out(j, k), ptr(reg_tmp_addr));
                prev_ofs = ofs;
            }
        }
        return prev_ofs;
    };

    Label no_update_label;

    ldr(reg_channel, ptr(kdnn_abi_param1, KUDNN_GET_OFF(channel)));
    cmp(reg_channel, 0);
    b(EQ, no_update_label);
    int prev_ofs = 0;
    for (int k = 0; k < jcp.nb_ic_blocking; k++) {
        for (int j = 0; j < ur_w; j++) {
            int num_ldr = nstl::min(ur_w, num_used_zreg);
            if (j == 0) {
                for (int t = 0; t < num_ldr; t++) {
                    size_t aux_src_offset = (size_t)typesize
                            * ((size_t)k * jcp.ih * jcp.iw * jcp.id + j + t)
                            * jcp.ic_block;
                    prev_ofs = out_load(aux_src_offset, t, prev_ofs);
                }
            } else if (j < ur_w - num_ldr + 1) {
                size_t aux_src_offset = (size_t)typesize
                        * ((size_t)k * jcp.ih * jcp.iw * jcp.id + j + num_ldr
                                - 1)
                        * jcp.ic_block;
                prev_ofs = out_load(aux_src_offset, j + num_ldr - 1, prev_ofs);
            }
            fadd(zreg_out_s(j, k), zreg_out_s(j, k), zreg_tmp_s(j));
        }
    }

    L(no_update_label);
    prev_ofs = 0;
    for (int k = 0; k < jcp.nb_ic_blocking; k++) {
        for (int j = 0; j < ur_w; j++) {
            size_t aux_src_offset = (size_t)typesize
                    * ((size_t)k * jcp.ih * jcp.iw * jcp.id + j) * jcp.ic_block;

            prev_ofs = out_str(j, k, aux_src_offset, prev_ofs);
        }
    }
}

template <cpu_isa_t isa>
void kdnn_jit_sve_conv_bwd_data_kernel<isa>::compute_loop_fma(
        int ur_w, int l_overflow, int r_overflow) {
    Label kh_label, kd_label;
    int kw = jcp.kw;
    int ow = jcp.ow;

    int ic_block = jcp.ic_block;
    int oc_block = jcp.oc_block;
    int l_pad = jcp.l_pad;
    int dilate_w = jcp.dilate_w + 1;
    int stride_w = jcp.stride_w;
    int stride_h = jcp.stride_h;

    int ker_pipeline_depth = 2;
    assert(ker_reg_base_idx + ker_pipeline_depth <= 31);
    assert(oc_block >= ker_pipeline_depth);

    int num_ker_loads = oc_block * kw;

    auto zreg_ker = [=](int i_ic) {
        assert(i_ic < ker_pipeline_depth);
        assert((ker_reg_base_idx + i_ic) < 31);
        return ZReg(ker_reg_base_idx + i_ic);
    };
    auto zreg_ker_s = [=](int i_ic) {
        assert(i_ic < ker_pipeline_depth);
        assert((ker_reg_base_idx + i_ic) < 31);
        return ZRegS(ker_reg_base_idx + i_ic);
    };
    auto zreg_out_s = [=](int i_ur, int i_oc) {
        int idx = (i_ur + i_oc * jcp.ur_w);
        assert(idx < ker_reg_base_idx);
        return ZRegS(idx);
    };
    auto zreg_in_s = [=](int idx) {
        int num_used_zreg = ker_reg_base_idx + ker_pipeline_depth;
        int zreg_idx = (idx % (32 - num_used_zreg)) + num_used_zreg;
        assert(zreg_idx <= 31);
        return ZRegS(zreg_idx);
    };

    auto bcast_load = [&](int aux_output_offset, int prev_ofs, int idx) {
        int num_used_zreg = ker_reg_base_idx + ker_pipeline_depth;
        int zreg_idx = (idx % (32 - num_used_zreg)) + num_used_zreg;
        if (((aux_output_offset & 0x3) == 0) && (aux_output_offset < LDRWMAX)
                && (aux_output_offset >= 0)) {
            ld1rw(ZRegS(zreg_idx), P_ALL_ONE,
                    ptr(aux_reg_dst, static_cast<int32_t>(aux_output_offset)));
        } else {
            int ofs = aux_output_offset - prev_ofs;
            int ofs2 = aux_output_offset - (prev_ofs + 0x100);
            int ofs3 = aux_output_offset - (prev_ofs + 0x200);
            if (((ofs & 0x3) == 0) && (ofs < LDRWMAX) && (ofs >= 0)) {
                ld1rw(ZRegS(zreg_idx), P_ALL_ONE,
                        ptr(reg_prev_bcast_addr, static_cast<int32_t>(ofs)));
            } else if (((ofs2 & 0x3) == 0) && (ofs2 < LDRWMAX) && (ofs2 >= 0)
                    && (prev_ofs != 0)) {
                ld1rw(ZRegS(zreg_idx), P_ALL_ONE,
                        ptr(reg_prev_bcast_addr2, static_cast<int32_t>(ofs2)));
            } else if (((ofs3 & 0x3) == 0) && (ofs3 < LDRWMAX) && (ofs3 >= 0)
                    && (prev_ofs != 0)) {
                ld1rw(ZRegS(zreg_idx), P_ALL_ONE,
                        ptr(reg_prev_bcast_addr3, static_cast<int32_t>(ofs3)));
            } else {
                ofs = aux_output_offset;
                add_imm(reg_prev_bcast_addr, aux_reg_dst, ofs, reg_tmp_imm);
                add_imm(reg_prev_bcast_addr2, aux_reg_dst, ofs + 0x100,
                        reg_tmp_imm);
                add_imm(reg_prev_bcast_addr3, aux_reg_dst, ofs + 0x200,
                        reg_tmp_imm);

                ld1rw(ZRegS(zreg_idx), P_ALL_ONE, ptr(reg_prev_bcast_addr));
                prev_ofs = ofs;
            }
        }
        return prev_ofs;
    };
    auto ker_load = [=](int i, int aux_kernel_offset) {
        int ofs = aux_kernel_offset;

        if ((kdnn_VL_OFS(ofs, isa) < LDRMAX) && (kdnn_VL_OFS(ofs, isa) >= (-1 * LDRMAX))
                && ((ofs & 0x3f) == 0)) {
            ldr(zreg_ker(i),
                    ptr(aux_reg_ker, static_cast<int32_t>(kdnn_VL_OFS(ofs, isa))));
        } else {
            add_imm(reg_tmp_addr, aux_reg_ker, ofs, reg_tmp_imm);
            ldr(zreg_ker(i), ptr(reg_tmp_addr));
        }
    };

    if (one_of(jcp.ndims, 3, 4)) {
        mov(aux_reg_dst, reg_dst);
        mov(aux_reg_ker, reg_ker);

        mov(aux_reg_dst_prf, reg_dst_prf);
        mov(aux_reg_ker_prf, reg_ker_prf);
    }

    if (jcp.ndims == 5) {
        mov(reg_src_prf_org, reg_src_prf);
        mov(reg_src_org, reg_src);

        ldr(reg_ki, ptr(kdnn_abi_param1, KUDNN_GET_OFF(kd_padding)));
        mov(aux_reg_dst_d, reg_dst);
        ldr(aux_reg_ker_d, ptr(kdnn_abi_param1, KUDNN_GET_OFF(filt)));
        mov(aux_reg_dst_d_prf, reg_dst_prf);
        mov(aux_reg_ker_d_prf, reg_ker_prf);

        L(kd_label);
        ldr(reg_kj, ptr(kdnn_abi_param1, KUDNN_GET_OFF(kh_padding)));
    } else {
        mov(reg_kj, reg_kh);
    }

    if (jcp.ndims == 5) {
        mov(aux_reg_dst, aux_reg_dst_d);
        mov(aux_reg_ker, aux_reg_ker_d);
        mov(aux_reg_dst_prf, aux_reg_dst_d_prf);
        mov(aux_reg_ker_prf, aux_reg_ker_d_prf);
    }
    int prev_ofs = 0;
    L(kh_label);
    {
        int step = 0;
        for (int ki = 0; ki < kw; ki++) {
            for (int oc = 0; oc < oc_block; oc++) {
                if (step == 0) {
                    for (int i = 0; i < ker_pipeline_depth; i++) {
                        int aux_kernel_offset = typesize
                                * ((oc + i) * oc_block
                                        + ki * ic_block * oc_block);
                        ker_load(i, aux_kernel_offset);
                    }
                } else if (step < num_ker_loads - ker_pipeline_depth + 1) {
                    int load_offset = ker_pipeline_depth - 1;
                    int ker_load_reg_idx
                            = (step + load_offset) % ker_pipeline_depth;
                    int aux_kernel_offset = typesize
                            * ((oc + load_offset) * oc_block
                                    + ki * ic_block * oc_block);
                    ker_load(ker_load_reg_idx, aux_kernel_offset);
                }

                auto zreg_kernel_s = zreg_ker_s(step % ker_pipeline_depth);

                int jj_start = get_iw_start(ki, l_overflow);
                int jj_end = get_iw_end(ur_w, ki, r_overflow);
                assert(stride_w != 1
                        || jj_start
                                == nstl::max(0,
                                        l_overflow - (kw - 1 - ki) * dilate_w));
                assert(stride_w != 1
                        || jj_end
                                == ur_w
                                        - nstl::max(
                                                0, r_overflow - ki * dilate_w));

                int bcast_idx = 0;
                int bcast_pipeline_depth
                        = 32 - (ker_reg_base_idx + ker_pipeline_depth);
                int num_bcast_pipeline = nstl::min(
                        ((jj_end - jj_start) / stride_w), bcast_pipeline_depth);
                for (int jj = jj_start; jj < jj_end; jj += stride_w) {
                    assert((jj + l_pad - ki * dilate_w) % stride_w == 0);
                    if (num_bcast_pipeline > 1) {
                        if (jj == jj_start) {
                            for (int i = 0; i < num_bcast_pipeline; i++) {
                                int jj_skip = jj + stride_w * i;
                                int aux_dst_offset = typesize
                                        * (((jj_skip + l_pad - ki * dilate_w)
                                                   / stride_w)
                                                        * jcp.oc_block
                                                + oc);
                                prev_ofs = bcast_load(aux_dst_offset, prev_ofs,
                                        bcast_idx + i);
                            }
                        } else if (jj < jj_end
                                        - (num_bcast_pipeline - 1) * stride_w) {
                            int jj_skip
                                    = jj + (num_bcast_pipeline - 1) * stride_w;
                            int aux_dst_offset = typesize
                                    * (((jj_skip + l_pad - ki * dilate_w)
                                               / stride_w)
                                                    * jcp.oc_block
                                            + oc);
                            prev_ofs = bcast_load(aux_dst_offset, prev_ofs,
                                    bcast_idx + (num_bcast_pipeline - 1));
                        }
                    } else {
                        int aux_dst_offset = typesize
                                * (((jj + l_pad - ki * dilate_w) / stride_w)
                                                * jcp.oc_block
                                        + oc);
                        prev_ofs = bcast_load(
                                aux_dst_offset, prev_ofs, bcast_idx);
                    }
                    fmla(zreg_out_s(jj, 0), P_ALL_ONE, zreg_kernel_s,
                            zreg_in_s(bcast_idx));
                    bcast_idx++;
                }
                step++;
            }
        }

        add_imm(aux_reg_ker, aux_reg_ker,
                typesize * stride_h * kw * oc_block * ic_block, reg_tmp_imm);
        sub_imm(aux_reg_dst, aux_reg_dst,
                typesize * (jcp.dilate_h + 1) * ow * oc_block, reg_tmp_imm);
        add_imm(aux_reg_ker_prf, aux_reg_ker_prf,
                typesize * stride_h * kw * oc_block * ic_block, reg_tmp_imm);
        sub_imm(aux_reg_dst_prf, aux_reg_dst_prf,
                typesize * (jcp.dilate_h + 1) * ow * oc_block, reg_tmp_imm);
        sub(reg_kj, reg_kj, 1);
        cmp(reg_kj, 0);
        b(GT, kh_label); //jg(kh_label, T_NEAR);
    }
    if (jcp.ndims == 5) {
        sub_imm(aux_reg_dst_d, aux_reg_dst_d,
                typesize * (jcp.dilate_d + 1) * jcp.oh * ow * ic_block,
                reg_tmp_imm);
        add_imm(aux_reg_ker_d, aux_reg_ker_d,
                typesize * jcp.stride_d * jcp.kw * jcp.kh * oc_block * ic_block,
                reg_tmp_imm);
        sub_imm(aux_reg_dst_d_prf, aux_reg_dst_d_prf,
                typesize * (jcp.dilate_d + 1) * jcp.oh * ow * ic_block,
                reg_tmp_imm);
        add_imm(aux_reg_ker_d_prf, aux_reg_ker_d_prf,
                typesize * jcp.stride_d * jcp.kw * jcp.kh * oc_block * ic_block,
                reg_tmp_imm);

        sub(reg_ki, reg_ki, 1);
        cmp(reg_ki, 0);
        b(GT, kd_label); //jg(kd_label, T_NEAR);
    }

    if (jcp.ndims == 5) {
        mov(reg_src, reg_src_org);
        mov(reg_src_prf, reg_src_prf_org);
    }
}

template <cpu_isa_t isa>
void kdnn_jit_sve_conv_bwd_data_kernel<isa>::compute_loop_fma_core(
        int ur_w, int l_overflow, int r_overflow, int k_offset) {
    int kw = jcp.kw;
    int ow = jcp.ow;
    int stride_w = jcp.stride_w;
    int ic_block = jcp.ic_block;
    int oc_block = jcp.oc_block;
    int nb_ic_block = jcp.nb_ic_blocking;
    Label kh_label, kd_label;

    const bool ddst_layout_nxc = is_ddst_layout_nxc();
    int shift_ker_ptr = typesize * kw * oc_block * ic_block;
    int oc_mult = ddst_layout_nxc ? jcp.ngroups * jcp.oc : oc_block;
    int shift_dst_ptr = typesize * (jcp.dilate_h + 1) * ow * oc_mult;

    const int oc_tail = jcp.oc_tail;
    const int max_filter_size = 20;
    Label oc_tail_jmp[max_filter_size];

    auto kernel_offset = [=](int icb, int oc, int ki) {
        int blk_idx = icb * jcp.kh * jcp.kw * jcp.kd + ki;
        int blk_offset = blk_idx * jcp.oc_block * jcp.ic_block;
        int oc_offset = oc * jcp.oc_block;
        return typesize * (blk_offset + oc_offset);
    };

    auto zreg_inp_s = [=](int i_ic, int nb_x_blocking) {
        int idx = (i_ic + nb_x_blocking * jcp.ur_w);
        assert(idx < 31);
        return ZRegS(idx);
    };
    auto zreg_out_s = [=](int i_ur, int i_oc) {
        int idx = (i_ur + i_oc * jcp.ur_w);
        assert(idx < ker_reg_base_idx);
        return ZRegS(idx);
    };
    const int num_wei_reg = 2;
    auto zreg_wei = [=](int idx) { return ZReg(31 - (idx % num_wei_reg)); };
    auto zreg_wei_s = [=](int idx) { return ZRegS(31 - (idx % num_wei_reg)); };

    auto bcast_load = [&](int jj, int nb_oc_block, int aux_output_offset,
                              int prev_ofs, int jj_end) {
        if (((aux_output_offset & 0x3) == 0) && (aux_output_offset < LDRWMAX)
                && (aux_output_offset >= 0)) {
            ld1rw(zreg_inp_s(jj, nb_oc_block), P_ALL_ONE,
                    ptr(aux_reg_dst, static_cast<int32_t>(aux_output_offset)));
        } else {
            if ((prev_ofs > -1) && ((aux_output_offset - prev_ofs) > 0)
                    && ((aux_output_offset - prev_ofs) < LDRWMAX)
                    && (((aux_output_offset - prev_ofs) & 0x3) == 0)) {

                ld1rw(zreg_inp_s(jj, nb_oc_block), P_ALL_ONE,
                        ptr(reg_prev_bcast_addr,
                                static_cast<int32_t>(
                                        aux_output_offset - prev_ofs)));

            } else {
                int ofs;
                if ((prev_ofs > -1) && ((aux_output_offset - prev_ofs) > 0)) {
                    ofs = aux_output_offset - prev_ofs;
                    add_imm(reg_prev_bcast_addr, reg_prev_bcast_addr, ofs,
                            reg_tmp_imm);

                } else {
                    ofs = aux_output_offset;
                    add_imm(reg_prev_bcast_addr, aux_reg_dst, ofs, reg_tmp_imm);
                }

                ld1rw(zreg_inp_s(jj, nb_oc_block), P_ALL_ONE,
                        ptr(reg_prev_bcast_addr));
                prev_ofs = aux_output_offset;
            }
        }
        return prev_ofs;
    };

    auto bcast_load_sw = [&](int jj, int nb_oc_block, int aux_output_offset,
                                 int prev_ofs, int jj_end) {
        if (((aux_output_offset & 0x3) == 0) && (aux_output_offset < LDRWMAX)
                && (aux_output_offset >= 0)) {
            ld1rw(zreg_inp_s(jj % stride_w, nb_oc_block), P_ALL_ONE,
                    ptr(aux_reg_dst, static_cast<int32_t>(aux_output_offset)));
        } else {
            if ((prev_ofs > -1) && ((aux_output_offset - prev_ofs) > 0)
                    && ((aux_output_offset - prev_ofs) < LDRWMAX)
                    && (((aux_output_offset - prev_ofs) & 0x3) == 0)) {

                ld1rw(zreg_inp_s(jj % stride_w, nb_oc_block), P_ALL_ONE,
                        ptr(reg_prev_bcast_addr,
                                static_cast<int32_t>(
                                        aux_output_offset - prev_ofs)));

            } else {
                int ofs;
                if ((prev_ofs > -1) && ((aux_output_offset - prev_ofs) > 0)) {
                    ofs = aux_output_offset - prev_ofs;
                    add_imm(reg_prev_bcast_addr, reg_prev_bcast_addr, ofs,
                            reg_tmp_imm);

                } else {
                    ofs = aux_output_offset;
                    add_imm(reg_prev_bcast_addr, aux_reg_dst, ofs, reg_tmp_imm);
                }

                ld1rw(zreg_inp_s(jj % stride_w, nb_oc_block), P_ALL_ONE,
                        ptr(reg_prev_bcast_addr));
                prev_ofs = aux_output_offset;
            }
        }
        return prev_ofs;
    };

    auto wei_load = [=](int aux_kernel_offset, int idx) {
        int ofs = aux_kernel_offset;

        if ((kdnn_VL_OFS(ofs, isa) < LDRMAX) && (kdnn_VL_OFS(ofs, isa) >= (-1 * LDRMAX))) {
            ldr(zreg_wei(idx),
                    ptr(aux_reg_ker, static_cast<int32_t>(kdnn_VL_OFS(ofs, isa))));
        } else {
            add_imm(reg_tmp_addr, aux_reg_ker, ofs, reg_tmp_imm);
            ldr(zreg_wei(idx), ptr(reg_tmp_addr));
        }
    };

    if (one_of(jcp.ndims, 3, 4)) {
        mov(aux_reg_dst, reg_dst);
        mov(aux_reg_ker, reg_ker);
    }

    if (jcp.ndims == 5) {
        mov(reg_src_prf_org, reg_src_prf);
        mov(reg_src_org, reg_src);

        ldr(reg_ki, ptr(kdnn_abi_param1, KUDNN_GET_OFF(kd_padding)));
        mov(aux_reg_dst_d, reg_dst);
        ldr(aux_reg_ker_d, ptr(kdnn_abi_param1, KUDNN_GET_OFF(filt)));

        L(kd_label);
        ldr(reg_kj, ptr(kdnn_abi_param1, KUDNN_GET_OFF(kh_padding)));
    } else {
        mov(reg_kj, reg_kh);
    }

    if (jcp.ndims == 5) {
        mov(aux_reg_dst, aux_reg_dst_d);
        mov(aux_reg_ker, aux_reg_ker_d);
    }

    L(kh_label);
    {
        for (int ki = 0; ki < kw; ki++) { // kernel width
            int prev_ofs = -1;

            int jj_start = get_iw_start(ki, l_overflow);
            int jj_end = get_iw_end(ur_w, ki, r_overflow);
            for (int oc = 0; oc < oc_block; oc++) {
                if (oc_tail && oc >= oc_tail) {
                    // if src has only tails to compute, skip early
                    if (jcp.oc == oc_tail)
                        break;
                    else if (oc == oc_tail) {
                        cmp_imm(reg_channel, oc_tail, reg_tmp_imm);
                        b(NE, oc_tail_jmp[ki]);
                    }
                }
                if (stride_w == 1) {
                    for (int jj = jj_start; jj < jj_end; jj += 1) {
                        int aux_output_offset = get_dst_offset(jj, oc, ki);
                        prev_ofs = bcast_load(jj, nb_ic_block,
                                aux_output_offset, prev_ofs, jj_end);
                    }
                }
                int wei_count = 0;
                for (int ii = 0; ii < nb_ic_block; ii++) {
                    if (jj_end - jj_start > 0) {
                        if (nb_ic_block > 1) {
                            if (ii == 0) {
                                for (int t = 0; t < num_wei_reg; t++) {
                                    int aux_kernel_offset = kernel_offset(
                                            ii + t, oc, ki + k_offset);
                                    wei_load(aux_kernel_offset, t);
                                }
                            } else if (ii < nb_ic_block - num_wei_reg + 1) {
                                int aux_kernel_offset
                                        = kernel_offset(ii + num_wei_reg - 1,
                                                oc, ki + k_offset);
                                wei_load(aux_kernel_offset,
                                        wei_count + num_wei_reg - 1);
                            }
                        } else {
                            int aux_kernel_offset
                                    = kernel_offset(ii, oc, ki + k_offset);
                            wei_load(aux_kernel_offset, wei_count);
                        }
                    }
                    for (int jj = jj_start; jj < jj_end; jj += stride_w) {
                        if (stride_w == 1) {
                            fmla(zreg_out_s(jj, ii), P_ALL_ONE,
                                    zreg_inp_s(jj, nb_ic_block),
                                    zreg_wei_s(wei_count));
                        } else {
                            int aux_output_offset = get_dst_offset(jj, oc, ki);
                            prev_ofs = bcast_load_sw(jj, nb_ic_block,
                                    aux_output_offset, prev_ofs, jj_end);

                            fmla(zreg_out_s(jj, ii), P_ALL_ONE,
                                    zreg_inp_s(jj % stride_w, nb_ic_block),
                                    zreg_wei_s(wei_count));
                        }
                    }
                    wei_count++;
                }
            }
            L(oc_tail_jmp[ki]);
        }
        add_imm(aux_reg_ker, aux_reg_ker, shift_ker_ptr, reg_tmp_imm);
        assert(shift_dst_ptr >= 0);
        sub_imm(aux_reg_dst, aux_reg_dst, shift_dst_ptr, reg_tmp_imm);
        sub(reg_kj, reg_kj, 1);
        cmp(reg_kj, 0);
        b(GT, kh_label);
    }

    if (jcp.ndims == 5) {
        sub_imm(aux_reg_dst_d, aux_reg_dst_d,
                typesize * (jcp.dilate_d + 1) * jcp.oh * ow * ic_block,
                reg_tmp_imm);
        add_imm(aux_reg_ker_d, aux_reg_ker_d,
                typesize * jcp.kw * jcp.kh * oc_block * ic_block, reg_tmp_imm);

        sub(reg_ki, reg_ki, 1);
        cmp(reg_ki, 0);
        b(GT, kd_label);

        mov(reg_src, reg_src_org);
        mov(reg_src_prf, reg_src_prf_org);
    }
}

template <cpu_isa_t isa>
void kdnn_jit_sve_conv_bwd_data_kernel<isa>::compute_loop(
        int ur_w, int l_overflow, int r_overflow, int k_offset) {
    if (jcp.ndims == 5) mov(reg_oi_org, reg_oi);

    prepare_output(ur_w);

    Label skip_compute_loop;
    if (jcp.ndims == 5) {
        ldr(reg_kj, ptr(kdnn_abi_param1, KUDNN_GET_OFF(kd_padding)));
        cmp(reg_kj, 0);
        b(LE, skip_compute_loop);
    }
    ldr(reg_kj, ptr(kdnn_abi_param1, KUDNN_GET_OFF(kh_padding)));
    cmp(reg_kj, 0);
    b(LE, skip_compute_loop);

    const bool generate_ocb_loop = jcp.nb_oc > 1 && is_ddst_layout_nxc();
    Label oc_loop;
    if (generate_ocb_loop) {
        mov(reg_dst_org, reg_dst);
        mov(reg_ker_org, reg_ker);

        ldr(reg_channel, ptr(kdnn_abi_param1, KUDNN_GET_OFF(reduce_work)));
        L(oc_loop);
    }

    if (jcp.ver == kdnn_ver_fma)
        if (jcp.nb_ic_blocking == 1)
            compute_loop_fma(ur_w, l_overflow, r_overflow);
        else
            compute_loop_fma_core(ur_w, l_overflow, r_overflow, k_offset);

    else
        assert("!unknown convolution version");

    if (generate_ocb_loop) {
        add_imm(reg_dst, reg_dst, jcp.oc_block * typesize, reg_tmp_imm);
        const int ker_shift = jcp.nb_ic * jcp.kd * jcp.kh * jcp.kw
                * jcp.ic_block * jcp.oc_block * typesize;
        add_imm(reg_ker, reg_ker, ker_shift, reg_tmp_imm);
        sub_imm(reg_channel, reg_channel, jcp.oc_block, reg_tmp_imm);
        cmp(reg_channel, 0);
        b(GT, oc_loop);

        mov(reg_ker, reg_ker_org);
        mov(reg_dst, reg_dst_org);
    }

    L(skip_compute_loop);
    store_output(ur_w);
    if (jcp.ndims == 5) mov(reg_oi, reg_oi_org);
}

template <cpu_isa_t isa>
void kdnn_jit_sve_conv_bwd_data_kernel<isa>::generate() {
    int iw = jcp.iw;
    int kw = jcp.kw;
    int ur_w = jcp.ur_w;
    int ic_block = jcp.ic_block;
    int oc_block = jcp.oc_block;
    int nb_iw = jcp.nb_iw;
    int iw_block = jcp.iw_block;
    int ur_w_tail = jcp.ur_w_tail;
    int dilate_w = jcp.dilate_w + 1;
    int stride_w = jcp.stride_w;

    int dst_shift = jcp.typesize_in * (ur_w / stride_w)
            * (is_ddst_layout_nxc() ? jcp.ngroups * jcp.oc : oc_block);
    int src_shift = jcp.typesize_out * ur_w
            * (is_dsrc_layout_nxc() ? jcp.ngroups * jcp.ic : ic_block);

    preamble();

    ldr(reg_src, ptr(kdnn_abi_param1, KUDNN_GET_OFF(src)));
    ldr(reg_dst, ptr(kdnn_abi_param1, KUDNN_GET_OFF(dst)));
    ldr(reg_ker, ptr(kdnn_abi_param1, KUDNN_GET_OFF(filt)));

    ldr(reg_kh, ptr(kdnn_abi_param1, KUDNN_GET_OFF(kh_padding)));
    ldr(reg_src_prf, ptr(kdnn_abi_param1, KUDNN_GET_OFF(src_prf)));
    ldr(reg_dst_prf, ptr(kdnn_abi_param1, KUDNN_GET_OFF(dst_prf)));
    ldr(reg_ker_prf, ptr(kdnn_abi_param1, KUDNN_GET_OFF(filt_prf)));

    int l_overflow = nstl::max(0, ((kw - 1) * dilate_w - jcp.l_pad) / stride_w);
    int r_overflow = nstl::max(
            0, ((kw - 1) * dilate_w - nstl::max(0, jcp.r_pad)) / stride_w);
    int r_overflow_no_tail = nstl::max(0,
            ((kw - 1) * dilate_w - nstl::max(0, jcp.r_pad + ur_w_tail))
                    / stride_w);

    int body_l_overflow = 0, body_r_overflow = 0;
    int n_oi = iw / ur_w;
    int head_n_oi = 0, body_n_oi = 0, pretail_n_oi = 0, tail_n_oi = 0;
    int head_thread = 0, pretail_thread = 0, tail_thread = 0;
    bool threaded = is_iw_threading_on(jcp);
    Label head_label, body_label, pretail_label, tail_label, end_label;
    assert(n_oi > 0);
    if (r_overflow_no_tail > 0) n_oi--;
    if (l_overflow > 0) n_oi--;
    if (n_oi < 0) {
        // l_overflow and r_overflow_no_tail are handled in the same
        // compute_loop. Perform one iteration of body handling l_overflow and
        // r_overflow_no_tail.
        // TODO: Align other convolution kernels with this kernel. This version
        // now uses r_overflow_no_tail instead of r_overflow in compute loop,
        // this was done since when iw == ur_w, ur_w_tail == 0 and thus
        // r_overflow_no_tail seems more appropriate
        body_l_overflow = l_overflow;
        body_r_overflow = r_overflow_no_tail;
        n_oi = 1;
        l_overflow = 0;
        r_overflow_no_tail = 0;
    }

    if (!threaded) {
        if (n_oi > 1) { mov_imm(reg_oi, n_oi); }
    } else {
        // Setup for threaded code generation, and jump into the correct
        // portion of code for execution.
        head_thread = 0;
        tail_thread = nb_iw - 1;
        pretail_thread = tail_thread;

        int base_n_oi = iw_block / ur_w;
        head_n_oi = l_overflow > 0 ? base_n_oi - 1 : base_n_oi;
        tail_n_oi = (iw - iw_block * (nb_iw - 1)) / ur_w;
        pretail_n_oi = tail_n_oi;
        if (r_overflow_no_tail > 0) {
            if (tail_n_oi > 0) {
                pretail_n_oi--;
                tail_n_oi = pretail_n_oi;
            } else {
                // pretail_thread and tail_thread are different
                pretail_n_oi = base_n_oi - 1;
                pretail_thread = tail_thread - 1;
            }
            if (head_thread == pretail_thread) {
                head_n_oi--;
                pretail_n_oi = 0;
                tail_n_oi = 0;
            }
        }
        body_n_oi = (head_thread < pretail_thread - 1) ? base_n_oi : 0;

        // n_oi is used to determine how much control flow in the body portion
        // of the code needs generated. As such, n_oi needs to be set to the
        // maximum number of iterations it will be used the body code section.
        n_oi = nstl::max(body_n_oi, head_n_oi);
        n_oi = nstl::max(n_oi, pretail_n_oi);

        assert(iw_block % ur_w == 0);
        ldr(reg_iwb, ptr(kdnn_abi_param1, KUDNN_GET_OFF(iwb)));

        if (head_n_oi != 0) mov_imm(reg_oi, head_n_oi);
        cmp_imm(reg_iwb, head_thread, reg_tmp_imm);
        b(EQ, head_label);

        cmp_imm(reg_iwb, pretail_thread, reg_tmp_imm);
        if (pretail_n_oi == 0) {
            b(EQ, pretail_label);
        } else {
            mov_imm(reg_oi, pretail_n_oi);
            b(EQ, body_label);
        }
        if (pretail_thread != tail_thread) {
            cmp_imm(reg_iwb, tail_thread, reg_tmp_imm);
            b(EQ, tail_label);
        }
        if (body_n_oi != 0) {
            mov_imm(reg_oi, body_n_oi);
            b(body_label);
        } else {
            b(end_label);
        }
    }

    L(head_label);
    if (l_overflow > 0) {
        compute_loop(ur_w, l_overflow, 0);
        if (threaded && head_n_oi == 0 && head_thread != pretail_thread)
            b(end_label);
        else {
            add_imm(reg_src, reg_src, src_shift, reg_tmp_imm);
            add_imm(reg_dst, reg_dst, dst_shift, reg_tmp_imm);
            add_imm(reg_src_prf, reg_src_prf, src_shift, reg_tmp_imm);
            add_imm(reg_dst_prf, reg_dst_prf, dst_shift, reg_tmp_imm);
        }
    }

    L(body_label);
    if (n_oi > 0) {
        Label ow_loop_label;
        L(ow_loop_label);
        {
            compute_loop(ur_w, body_l_overflow, body_r_overflow);
            if (n_oi > 1 || r_overflow_no_tail > 0 || ur_w_tail != 0) {
                add_imm(reg_src, reg_src, src_shift, reg_tmp_imm);
                add_imm(reg_src_prf, reg_src_prf, src_shift, reg_tmp_imm);
                if (!jcp.large_w_filter) {
                    add_imm(reg_dst, reg_dst, dst_shift, reg_tmp_imm);
                    add_imm(reg_dst_prf, reg_dst_prf, dst_shift, reg_tmp_imm);
                }
            }
            if (n_oi > 1) {
                sub(reg_oi, reg_oi, 1);
                cmp(reg_oi, 0);
                b(GT, ow_loop_label);
            }
        }
    }
    if (threaded) {
        ldr(reg_iwb, ptr(kdnn_abi_param1, KUDNN_GET_OFF(iwb)));
        cmp_imm(reg_iwb, pretail_thread, reg_tmp_imm);
        b(NE, end_label);
    }

    L(pretail_label);
    if (r_overflow_no_tail > 0) {
        compute_loop(ur_w, 0, r_overflow_no_tail);
        if (ur_w_tail != 0) {
            if (threaded && tail_thread != pretail_thread) b(end_label);
            add_imm(reg_src, reg_src, src_shift, reg_tmp_imm);
            add_imm(reg_dst, reg_dst, dst_shift, reg_tmp_imm);
            add_imm(reg_src_prf, reg_src_prf, src_shift, reg_tmp_imm);
            add_imm(reg_dst_prf, reg_dst_prf, dst_shift, reg_tmp_imm);
        }
    }

    L(tail_label);
    if (ur_w_tail != 0) {
        /* if 'filter-width > ur_w' then the main loop only partially computes
         * width, ur_w_tail needs to offset the initial ur_w from the filter
         * address. */
        if (jcp.large_w_filter)
            compute_loop(ur_w_tail, body_l_overflow, r_overflow - ur_w, ur_w);
        else
            compute_loop(ur_w_tail, 0, r_overflow);
    }

    L(end_label);

    postamble();
}

template <cpu_isa_t isa>
status_t kdnn_jit_sve_conv_bwd_data_kernel<isa>::init_conf(kdnn_jit_conv_conf_t &jcp,
        const convolution_desc_t &cd, memory_desc_t &diff_src_md,
        memory_desc_t &weights_md, memory_desc_t &diff_dst_md, int nthreads) {
    if (!mayiuse(sve_256)) return status::unimplemented;
    // SVE512 not supported
    if (cpu().getSveLen() > Xbyak_aarch64::util::SVE_256) return status::unimplemented;

    const memory_desc_wrapper diff_src_d(&diff_src_md);
    const memory_desc_wrapper weights_d(&weights_md);
    const memory_desc_wrapper diff_dst_d(&diff_dst_md);
    jcp = zero<decltype(jcp)>();

    const bool with_groups = weights_d.ndims() == diff_src_d.ndims() + 1;
    int ndims = diff_src_d.ndims();

    jcp.nthr = jcp.aligned_threads = nthreads;
    jcp.ndims = ndims;
    jcp.prop_kind = cd.prop_kind;

    jcp.ngroups = with_groups ? weights_d.dims()[0] : 1;
    jcp.mb = diff_src_d.dims()[0];

    jcp.oc = diff_dst_d.dims()[1] / jcp.ngroups;
    jcp.oc_without_padding = jcp.oc;
    jcp.ic = diff_src_d.dims()[1] / jcp.ngroups;
    jcp.ic_without_padding = jcp.ic;

    jcp.id = (ndims == 5) ? diff_src_d.dims()[2] : 1;
    jcp.ih = (ndims == 3) ? 1 : diff_src_d.dims()[ndims - 2];
    jcp.iw = diff_src_d.dims()[ndims - 1];
    jcp.od = (ndims == 5) ? diff_dst_d.dims()[2] : 1;
    jcp.oh = (ndims == 3) ? 1 : diff_dst_d.dims()[ndims - 2];
    jcp.ow = diff_dst_d.dims()[ndims - 1];

    jcp.kd = (ndims == 5) ? weights_d.dims()[with_groups + 2] : 1;
    jcp.kh = (ndims == 3) ? 1 : weights_d.dims()[with_groups + ndims - 2];
    jcp.kw = weights_d.dims()[with_groups + ndims - 1];

    jcp.f_pad = (ndims == 5) ? cd.padding[0][0] : 0;
    jcp.t_pad = (ndims == 3) ? 0 : cd.padding[0][ndims - 4];
    jcp.l_pad = cd.padding[0][ndims - 3];

    jcp.stride_d = (ndims == 5) ? cd.strides[0] : 1;
    jcp.stride_h = (ndims == 3) ? 1 : cd.strides[ndims - 4];
    jcp.stride_w = cd.strides[ndims - 3];

    jcp.dilate_d = (ndims == 5) ? cd.dilates[0] : 0;
    jcp.dilate_h = (ndims == 3) ? 0 : cd.dilates[ndims - 4];
    jcp.dilate_w = cd.dilates[ndims - 3];
    if ((jcp.dilate_w != 0 && jcp.stride_w != 1)
            || (jcp.dilate_d != 0 && jcp.stride_d != 1)
            || (jcp.dilate_h != 0 && jcp.stride_h != 1))
        return status::unimplemented;

    int ext_kw = calculate_extended_filter_size(jcp.kw, jcp.dilate_w);
    int ext_kh = calculate_extended_filter_size(jcp.kh, jcp.dilate_h);
    int ext_kd = calculate_extended_filter_size(jcp.kd, jcp.dilate_d);
    jcp.r_pad = calculate_end_padding(
            jcp.l_pad, jcp.ow, jcp.iw, jcp.stride_w, ext_kw);
    jcp.b_pad = calculate_end_padding(
            jcp.t_pad, jcp.oh, jcp.ih, jcp.stride_h, ext_kh);
    jcp.back_pad = calculate_end_padding(
            jcp.f_pad, jcp.od, jcp.id, jcp.stride_d, ext_kd);
    bool kernel_outside_src = false || ext_kw <= jcp.l_pad
            || ext_kw <= jcp.r_pad || ext_kh <= jcp.t_pad || ext_kh <= jcp.b_pad
            || ext_kd <= jcp.f_pad || ext_kd <= jcp.back_pad;
    if (kernel_outside_src) return status::unimplemented;

    jcp.aligned_threads = 0;
    const auto dat_tag_nxc = pick(ndims - 3, nwc, nhwc, ndhwc);
    const auto dat_tag_nCx4c = pick(ndims - 3, nCw4c, nChw4c, nCdhw4c);
    const auto dat_tag_nCx8c = pick(ndims - 3, nCw8c, nChw8c, nCdhw8c);
    const auto dat_tag_nCx16c = pick(ndims - 3, nCw16c, nChw16c, nCdhw16c);
    auto curr_src_tag = diff_src_d.matches_one_of_tag(
            dat_tag_nxc, dat_tag_nCx16c, dat_tag_nCx8c, dat_tag_nCx4c);
    auto curr_dst_tag = diff_dst_d.matches_one_of_tag(
            dat_tag_nxc, dat_tag_nCx16c, dat_tag_nCx8c, dat_tag_nCx4c);
    bool is_data_layout_nxc
            = utils::everyone_is(dat_tag_nxc, curr_src_tag, curr_dst_tag);
    if (mayiuse(sve_256) && is_data_layout_nxc) return status::unimplemented;

    jcp.is_1stconv = false;

    bool ok_to_pad_channels = true && !is_data_layout_nxc && jcp.ngroups == 1
            && diff_src_d.data_type() == data_type::f32;

    const int full_simd_w = cpu_isa_traits<sve_256>::vlen / typesize;
    jcp.simd_w = full_simd_w;

    jcp.oc_block = jcp.simd_w;
    jcp.ic_block = jcp.is_1stconv ? jcp.ic : jcp.simd_w;

    if (ok_to_pad_channels) {
        jcp.oc = rnd_up(jcp.oc, jcp.oc_block);
        jcp.ic = rnd_up(jcp.ic, jcp.ic_block);
    }

    if (!IMPLICATION(!is_data_layout_nxc,
                jcp.oc % jcp.oc_block == 0 && jcp.ic % jcp.ic_block == 0))
        return status::unimplemented;
    jcp.ic_tail = is_data_layout_nxc ? jcp.ic % jcp.simd_w : 0;
    jcp.oc_tail = is_data_layout_nxc ? jcp.oc % jcp.simd_w : 0;

    format_tag_t dat_tag, wei_tag;
    const auto nxc_tag = pick(ndims - 3, nwc, nhwc, ndhwc);

    if (jcp.simd_w == 8) {
        dat_tag = is_data_layout_nxc ? nxc_tag
                                     : pick(ndims - 3, nCw8c, nChw8c, nCdhw8c);
        wei_tag = pick(2 * ndims - 6 + with_groups, OIw8o8i, gOIw8o8i,
                    OIhw8o8i, gOIhw8o8i, OIdhw8o8i, gOIdhw8o8i);
    } else if (jcp.simd_w == 4) {
        assert(with_groups);
        dat_tag = is_data_layout_nxc ? nxc_tag
                                     : pick(ndims - 3, nCw4c, nChw4c, nCdhw4c);
        wei_tag = pick(ndims - 3, gOIw4o4i, gOIhw4o4i, gOIdhw4o4i);
    } else {
        dat_tag = is_data_layout_nxc
                ? pick(ndims - 3, nwc, nhwc, ndhwc)
                : pick(ndims - 3, nCw16c, nChw16c, nCdhw16c);
        wei_tag = pick(2 * ndims - 6 + with_groups, OIw16o16i, gOIw16o16i,
                OIhw16o16i, gOIhw16o16i, OIdhw16o16i, gOIdhw16o16i);
    }

    if (diff_src_md.format_kind == format_kind::any) {
        CHECK(memory_desc_init_by_tag(diff_src_md, dat_tag));
    } else if (curr_src_tag != dat_tag)
        return status::unimplemented;
    jcp.src_tag = dat_tag;

    if (diff_dst_md.format_kind == format_kind::any) {
        CHECK(memory_desc_init_by_tag(diff_dst_md, dat_tag));
    } else if (curr_dst_tag != dat_tag)
        return status::unimplemented;
    jcp.dst_tag = dat_tag;

    if (init_tag(jcp.wei_tag, weights_md, weights_d, wei_tag)
            != status::success)
        return status::unimplemented;

    jcp.nb_ic = div_up(jcp.ic, jcp.ic_block);
    jcp.nb_oc = div_up(jcp.oc, jcp.oc_block);

    jcp.ur_w = jcp.stride_w;

    int regs = 24;
    if (jcp.iw <= regs)
        jcp.ur_w = jcp.iw;
    else {
        for (int ur_w = regs; ur_w > 0; --ur_w)
            if (ur_w % jcp.stride_w == 0) {
                jcp.ur_w = ur_w;
                break;
            }
    }
    int l_overflow = nstl::max(
            0, ((jcp.kw - 1) * (jcp.dilate_w + 1) - jcp.l_pad) / jcp.stride_w);
    int r_overflow_no_tail = nstl::max(0,
            ((jcp.kw - 1) * (jcp.dilate_w + 1)
                    - nstl::max(0, jcp.r_pad + jcp.iw % jcp.ur_w))
                    / jcp.stride_w);
    int n_oi = jcp.iw / jcp.ur_w;
    if (r_overflow_no_tail > 0) n_oi--;

    if (mayiuse(sve_256) && diff_dst_d.data_type() == data_type::f32
            && weights_d.data_type() == data_type::f32
            && diff_src_d.data_type() == data_type::f32) {
        jcp.ver = kdnn_ver_fma;
        jcp.typesize_in = typesize;
        jcp.typesize_out = typesize;
    } else {
        return status::unimplemented;
    }

    jcp.nb_ic_blocking = jcp.nb_oc_blocking = 1;

    bool large_code_size = (jcp.ur_w != jcp.ow)
            && ((l_overflow <= 0 && n_oi > 0) || (l_overflow > 0 && n_oi > 1))
            && (r_overflow_no_tail > 0) && (l_overflow > 0);
    if (large_code_size) {
        const int max_code_size = 24 * 1024;
        const int num_ops_per_reg = 6 + jcp.oc_block * jcp.kw;
        int mult = 1;
        if (l_overflow > 0) mult += 1;
        if (r_overflow_no_tail > 0) mult += 1;
        for (int ur_w = jcp.ur_w; ur_w > regs / 2; --ur_w) {
            if ((ur_w / jcp.stride_w) * mult * num_ops_per_reg * 9.2
                    < max_code_size) {
                if (ur_w % jcp.stride_w == 0) {
                    jcp.ur_w = ur_w;
                    break;
                }
            }
        }
    }

    /* Support for large filter 'kw > 14' is only possible when ur_w is small
     * (e.g ur_w = 1) because of register allocation (max_reg = 31) */
    const int min_filter_size = 14;
    /* Don't let JIT generate too big of a code which might result in an
     * out-of-memory crash. */
    const int max_filter_size = 20;

    /* These conditions define a set of shapes with 'ow = 1' which
     * have a very limited optimization space for performance.
     * Optimize by using a targeted 'jcp.nb_ic_blocking' value. */
    jcp.large_w_filter = jcp.kw >= min_filter_size && jcp.kw < max_filter_size
            && jcp.ow == 1 && jcp.nb_ic > 1 && jcp.kw == jcp.iw
            && jcp.stride_w == 1
            && utils::everyone_is(0, jcp.dilate_d, jcp.dilate_h, jcp.dilate_w);

    if (jcp.ver == kdnn_ver_fma && mayiuse(sve_256)) {
        int try_nb_ic_blocking = 2;
        bool use_expl_bcast
                = !(jcp.kw == 1 || (jcp.kw == 5 && jcp.iw < 8)
                          || (jcp.kw < 5
                                  && ((jcp.iw <= 5
                                          || (jcp.iw > 8 && jcp.iw <= 13)))))
                || jcp.stride_h > 1 || jcp.stride_d > 1;
        if (use_expl_bcast && !jcp.large_w_filter) {
            jcp.kernel_kind = kdnn_embd_bcast;
            jcp.ur_w = nstl::min(jcp.iw, 16);
            jcp.nb_ic_blocking = jcp.nb_oc_blocking = 1;
            if (!(jcp.kw > 3 || (jcp.kw == 3 && jcp.ow > 8))
                    && jcp.stride_h == 1 && jcp.stride_d == 1)
                if (jcp.nb_ic % try_nb_ic_blocking == 0) {
                    jcp.nb_ic_blocking = try_nb_ic_blocking;
                    jcp.ur_w = 30 / (jcp.nb_ic_blocking + 1);
                    if (jcp.iw < jcp.ur_w) jcp.ur_w = jcp.iw;
                }
        } else {
            jcp.kernel_kind = kdnn_expl_bcast;
            jcp.nb_oc_blocking = 1;
            jcp.nb_ic_blocking = jcp.large_w_filter ? 2 : 4;
            if (jcp.nb_ic < jcp.nb_ic_blocking) jcp.nb_ic_blocking = jcp.nb_ic;
            if (jcp.nb_ic % jcp.nb_ic_blocking != 0)
                for (int i = jcp.nb_ic_blocking; i > 0; i--)
                    if (jcp.nb_ic % i == 0) {
                        jcp.nb_ic_blocking = i;
                        break;
                    }
            if (jcp.stride_w > 1) {
                jcp.ur_w = 30 / (jcp.nb_ic_blocking + 1);
            } else {
                jcp.ur_w = 31 / (jcp.nb_ic_blocking + 1);
            }
            if (jcp.iw < jcp.ur_w) jcp.ur_w = jcp.iw;
        }
    }
    jcp.ur_w_tail = jcp.iw % jcp.ur_w;

    auto is_iw_threading_applicable = [=]() { return one_of(jcp.ndims, 3, 4); };

    auto get_thr_eff = [=](int nb_ic_blocking, int iw_block) {
        // Cost heuristic for threading overhead. Determined using OMP.
        const float iw_block_cost = 32.0;

        int nb_iw = div_up(jcp.iw, iw_block);
        int nb_ic_chunks = div_up(jcp.nb_ic, nb_ic_blocking);
        int work_amount = jcp.mb * jcp.ih * nb_ic_chunks * nb_iw;
        float disbalance = (float)jcp.iw / rnd_up(jcp.iw, iw_block);
        float block_overhead = nstl::max(0.0f, 1.0f - iw_block_cost / iw_block);
        float thr_eff = block_overhead * disbalance
                * ((float)work_amount / rnd_up(work_amount, nthreads));
        return thr_eff;
    };

    auto get_iw_block = [=](int nb_ic_blocking, int ur_w) {
        int res_iw_block = jcp.iw;
        if (!is_iw_threading_applicable()) return res_iw_block;

        int max_nb_iw = div_up(jcp.iw, 2 * ur_w);
        int iw_block_thr;
        float eff;

        if (jcp.ndims == 3) {
            // Blocking optimization to prevent data from leaving cache This
            // blocking optimization does not handle height blocking, so it does
            // not apply to higher dimensions.
            // TODO: Implement a more general optimization taking into account
            // the height dimension.
            int L2_part
                    = (platform::get_per_core_cache_size(2) * 7 / 8) / typesize;
            int size_diff_src_chunk = jcp.ic_block * nb_ic_blocking * ur_w;
            int size_diff_dst_chunk = jcp.oc_block * ur_w;
            int size_wei_chunk
                    = jcp.ic_block * nb_ic_blocking * jcp.oc_block * jcp.kw;
            int nurw_cache = (L2_part - 2 * size_wei_chunk)
                    / (2 * size_diff_dst_chunk + 2 * size_diff_src_chunk);
            // current design of generate() requires iw_block >= 2 * ur_w
            int iw_block_cache = ur_w * nstl::max(2, nurw_cache);

            iw_block_thr = iw_block_cache;
        } else
            iw_block_thr = jcp.iw;
        eff = get_thr_eff(nb_ic_blocking, iw_block_thr);

        // Search for most efficient threading over iw_blocks.
        int start_nb_iw = div_up(jcp.iw, iw_block_thr);
        for (int nb_iw = start_nb_iw; nb_iw <= max_nb_iw; nb_iw++) {
            float eff_threshold = 0.98f;
            if (eff > eff_threshold) break;
            int iw_block
                    = nstl::min(rnd_up(div_up(jcp.iw, nb_iw), ur_w), jcp.iw);
            if (div_up(jcp.iw, iw_block) != nb_iw) continue;
            float thr_eff = get_thr_eff(nb_ic_blocking, iw_block);
            if (iw_block >= 2 * ur_w && thr_eff > eff) {
                iw_block_thr = iw_block;
                eff = thr_eff;
            }
        }
        res_iw_block = nstl::min(jcp.iw, nstl::max(2 * ur_w, iw_block_thr));
        return res_iw_block;
    };

    jcp.iw_block = get_iw_block(jcp.nb_ic_blocking, jcp.ur_w);
    jcp.nb_iw = div_up(jcp.iw, jcp.iw_block);

    if (l_overflow * jcp.stride_w > jcp.ur_w && !jcp.large_w_filter)
        return status::unimplemented;
    r_overflow_no_tail = nstl::max(0,
            ((jcp.kw - 1) * (jcp.dilate_w + 1)
                    - nstl::max(0, jcp.r_pad + jcp.ur_w_tail))
                    / jcp.stride_w);
    bool tails_not_ok = false
            /* maximum 1 ur_w block with r_overflow so far */
            || r_overflow_no_tail * jcp.stride_w > jcp.ur_w
            /* ur_w must be a multiple of stride */
            || ((jcp.iw > jcp.ur_w) && (jcp.ur_w % jcp.stride_w != 0))
            /* r_pad must not extend beyond ur_w_tail */
            || ((jcp.iw > jcp.ur_w) && (jcp.r_pad + jcp.ur_w_tail < 0));
    if (tails_not_ok) return status::unimplemented;

    pick_loop_order(jcp);

    jcp.nb_oc_L2 = jcp.nb_oc;

    if (is_data_layout_nxc) {
        // TODO: improve L2 blocking for large OC
        const int nb_oc_theshold_L2 = 32;
        if (jcp.nb_oc > nb_oc_theshold_L2 && jcp.nb_oc < 2 * nb_oc_theshold_L2)
            jcp.nb_oc_L2 = div_up(jcp.nb_oc, 2);
        else
            jcp.nb_oc_L2 = nstl::min(nb_oc_theshold_L2, jcp.nb_oc);
    }

    bool args_ok = true && jcp.ic <= diff_src_d.padded_dims()[1]
            && jcp.oc <= diff_dst_d.padded_dims()[1]
            && jcp.ic <= weights_d.padded_dims()[with_groups + 1]
            && jcp.oc <= weights_d.padded_dims()[with_groups + 0];
    if (!args_ok) return status::unimplemented;

    // A rough check on code size
    // TODO: come up with a tighter bound
    {
        const int max_code_size = 256 * 1024; // default size of jit generator
        int mult = 1 + (l_overflow > 0) + (r_overflow_no_tail > 0);
        const float max_instruction_size = 15;
        float ur_fac
                = (float)jcp.kw * jcp.oc_block * jcp.nb_ic_blocking * jcp.ur_w;
        float code_size = mult * ur_fac * max_instruction_size;
        if (code_size > max_code_size && !jcp.large_w_filter)
            return status::unimplemented;
    }

    return status::success;
}

template <cpu_isa_t isa>
void kdnn_jit_sve_conv_bwd_data_kernel<isa>::init_scratchpad(
        memory_tracking::registrar_t &scratchpad, const kdnn_jit_conv_conf_t &jcp) {
    UNUSED(scratchpad);
    UNUSED(jcp);
}

/*struct instantiation*/
template struct kdnn_jit_sve_conv_bwd_data_kernel<sve_256>;

} // namespace aarch64
} // namespace cpu
} // namespace impl
} // namespace dnnl

