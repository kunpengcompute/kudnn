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

#ifndef KUDNN_JIT_F32_CONV_KERNEL_HPP
#define KUDNN_JIT_F32_CONV_KERNEL_HPP

#include "common/c_types_map.hpp"
#include "common/memory_tracking.hpp"

#include "cpu/aarch64/kudnn/jit/utils/kudnn_jit_generator.hpp"
#include "cpu/aarch64/kudnn/jit/utils/kudnn_jit_primitive_conf.hpp"
#include "cpu/aarch64/kudnn/jit/utils/kudnn_jit_op_imm_check.hpp"
#include "cpu/aarch64/kudnn/jit/eltwise/kudnn_jit_uni_eltwise_injector.hpp"

#define LDRWMAX 252
#define kdnn_VL_OFS(ofs, isa) (ofs >> cpu_isa_traits<isa>::vlen_shift)

using namespace Xbyak_aarch64;

namespace dnnl {
namespace impl {
namespace cpu {
namespace aarch64 {

template <cpu_isa_t isa = isa_undef>
struct kdnn_jit_sve_conv_fwd_kernel : public kdnn_jit_generator {

    kdnn_jit_sve_conv_fwd_kernel(
            const kdnn_jit_conv_conf_t &ajcp, const primitive_attr_t &attr)
        : jcp(ajcp), attr_(attr), eltwise_injector_(nullptr) {

        if (jcp.with_eltwise)
            eltwise_injector_
                    = new kdnn_jit_uni_eltwise_injector_f32<isa>(this, jcp.eltwise);
        }

    ~kdnn_jit_sve_conv_fwd_kernel() {
        delete eltwise_injector_;
    }

    DECLARE_KUDNN_JIT_AUX_FUNCTIONS(kdnn_jit_sve_conv_fwd_kernel)

    kdnn_jit_conv_conf_t jcp;
    const primitive_attr_t &attr_;

    static bool post_ops_ok(kdnn_jit_conv_conf_t &jcp, const primitive_attr_t &attr);
    static status_t init_conf(kdnn_jit_conv_conf_t &jcp,
            const convolution_desc_t &cd, memory_desc_t &src_pd,
            memory_desc_t &weights_pd, memory_desc_t &dst_pd,
            memory_desc_t &bias_pd, const primitive_attr_t &attr, int nthreads);
    static void init_scratchpad(memory_tracking::registrar_t &scratchpad,
            const kdnn_jit_conv_conf_t &jcp);

private:
    using reg64_t = const XReg;
    enum {
        typesize = sizeof(float),
        ker_reg_base_idx = 28,
    };

    reg64_t param = kdnn_abi_param1;
    reg64_t reg_inp = x1; // src base addr (2d)
    reg64_t reg_ker = x2; // ker base addr (2d)
    reg64_t aux_reg_ker_d = x2; // ker addr (3d)
    reg64_t reg_out = x3; // dst base addr (2d)
    reg64_t reg_ki = x3; // d-dim loop var? (3d)
    reg64_t reg_owb = x5; // num of ow-block
    reg64_t reg_out_prf = x6; // addr for prefetch

    reg64_t aux_reg_inp = x7; // src addr (main loop)
    reg64_t aux_reg_inp2 = x24; // src addr (main loop)
    reg64_t aux_reg_inp3 = x25; // src addr (main loop)
    reg64_t reg_out_ofs = x7; // dst addr (store_output)
    reg64_t aux_reg_ker = x8; // ker addr (main loop)
    reg64_t reg_channel = x9; // reduce workload
    reg64_t reg_bias = x10; // bias addr (prepare_out)

    reg64_t aux_reg_inp_d = x11; // src addr (3d)
    reg64_t reg_oi = x11;

    reg64_t reg_kh = x12; // ker h size
    reg64_t reg_kj = x13; // ker h workload

    /* Temporary registers for ARM insts */
    reg64_t reg_tmp_addr = x14;
    reg64_t reg_prev_bcast_addr = x15;
    reg64_t reg_prev_wei_addr = x16;
    reg64_t reg_tmp_imm = x17;

    reg64_t reg_out_org = x27; // dst base addr (3d)
    reg64_t reg_oi_org = x19; // base oi (3d)
    reg64_t aux_reg_ker_d_org = x20;
    // register conflict with post-op eltwise
    reg64_t reg_ker_org = x22; // ker base addr (3d)
    reg64_t reg_inp_org = x29; // src base addr (3d)

    void prefetch(
            const std::string prfop, int level, reg64_t in, long long int ofs) {
        bool for_load = false;
        if (prfop == "LD") {
            for_load = true;
        } else if (prfop == "ST") {
            for_load = false;
        } else {
            assert(!"invalid prfop");
        }

        bool cacheline_aligned = ((ofs & 0xFF) == 0) ? true : false;
        if (cacheline_aligned == true) {
            Prfop op = PLDL1KEEP;
            switch (level) {
                case 1: op = (for_load == true) ? PLDL1KEEP : PSTL1KEEP; break;
                case 2: op = (for_load == true) ? PLDL2KEEP : PSTL2KEEP; break;
                case 3: op = (for_load == true) ? PLDL3KEEP : PSTL3KEEP; break;
                default: assert(!"invalid prfop"); break;
            }

            if ((ofs <= PRFMMAX) && (ofs >= 0)) {
                prfm(op, ptr(in, static_cast<int32_t>(ofs)));
            } else {
                add_imm(reg_tmp_addr, in, ofs, reg_tmp_imm);
                prfm(op, ptr(reg_tmp_addr));
            }
        } else {
            PrfopSve op_sve = PLDL1KEEP_SVE;
            switch (level) {
                case 1:
                    op_sve = (for_load == true) ? PLDL1KEEP_SVE : PSTL1KEEP_SVE;
                    break;
                case 2:
                    op_sve = (for_load == true) ? PLDL2KEEP_SVE : PSTL2KEEP_SVE;
                    break;
                case 3:
                    op_sve = (for_load == true) ? PLDL3KEEP_SVE : PSTL3KEEP_SVE;
                    break;
                default: assert(!"invalid level"); break;
            }

            if ((kdnn_VL_OFS(ofs, isa) < PRFWMAX)
                    && (kdnn_VL_OFS(ofs, isa) >= (-1 * PRFWMAX))) {
                prfw(op_sve, P_ALL_ONE,
                        ptr(in, static_cast<int32_t>(kdnn_VL_OFS(ofs, isa))));
            } else {
                add_imm(reg_tmp_addr, in, ofs, reg_tmp_imm);
                prfw(op_sve, P_ALL_ONE, ptr(reg_tmp_addr));
            }
        }
    }

    kdnn_jit_uni_eltwise_injector_f32<isa> *eltwise_injector_;

    inline void prepare_output(int ur_w);
    inline void store_output(int ur_w);
    inline void compute_loop_fma(int ur_w, int pad_l, int pad_r);
    inline void compute_loop_fma_cont(int ur_w, int pad_l, int pad_r);
    inline void compute_loop_fma_core(int ur_w, int pad_l, int pad_r);
    inline void compute_loop(int ur_w, int pad_l, int pad_r);

    void generate() override;

    inline size_t get_output_offset(int oi, int n_oc_block) {
        const bool is_nxc_layout = is_dst_layout_nxc();
        size_t ow_str = is_nxc_layout ? jcp.ngroups * jcp.oc : jcp.oc_block;
        size_t ocb_str = is_nxc_layout
                ? jcp.oc_block
                : (size_t)jcp.od * jcp.oh * jcp.ow * jcp.oc_block;

        return jcp.typesize_out * (n_oc_block * ocb_str + oi * ow_str);
    }

    inline size_t get_input_offset(int ki, int ic, int oi, int pad_l) {
        const bool is_nxc_layout = is_src_layout_nxc();
        size_t iw_str = is_nxc_layout ? jcp.ngroups * jcp.ic
                                      : (!jcp.is_1stconv ? jcp.ic_block : 1);
        size_t ic_str = !jcp.is_1stconv || is_nxc_layout
                ? 1
                : (size_t)jcp.iw * jcp.ih * jcp.id;
        size_t iw_idx = ki * (jcp.dilate_w + 1) + oi * jcp.stride_w - pad_l;

        return jcp.typesize_in * (iw_idx * iw_str + ic * ic_str);
    }

    inline int get_kernel_offset(
            int ki, int ic, int n_oc_block, int ker_number) {
        return jcp.typesize_in * jcp.oc_block
                * (n_oc_block * jcp.nb_ic * jcp.ic_block * jcp.kh * jcp.kw
                                * jcp.kd
                        + (ic + ker_number) + ki * jcp.ic_block);
    }

    inline int get_ow_start(int ki, int pad_l) {
        return nstl::max(0,
                utils::div_up(pad_l - ki * (jcp.dilate_w + 1), jcp.stride_w));
    }

    inline int get_ow_end(int ur_w, int ki, int pad_r) {
        return ur_w
                - nstl::max(0,
                        utils::div_up(
                                pad_r - (jcp.kw - 1 - ki) * (jcp.dilate_w + 1),
                                jcp.stride_w));
    }
    inline bool is_src_layout_nxc() {
        return utils::one_of(jcp.src_tag, format_tag::ndhwc, format_tag::nhwc,
                format_tag::nwc);
    }
    inline bool is_dst_layout_nxc() {
        return utils::one_of(jcp.dst_tag, format_tag::ndhwc, format_tag::nhwc,
                format_tag::nwc);
    }
};

template <cpu_isa_t isa = isa_undef>
struct kdnn_jit_sve_conv_bwd_data_kernel : public kdnn_jit_generator {

    kdnn_jit_sve_conv_bwd_data_kernel(const kdnn_jit_conv_conf_t &ajcp)
        : jcp(ajcp) {}

    DECLARE_KUDNN_JIT_AUX_FUNCTIONS(kdnn_jit_sve_conv_bwd_data_kernel)
    kdnn_jit_conv_conf_t jcp;
    void (*jit_ker_)(kdnn_jit_conv_call_s *);

    static status_t init_conf(kdnn_jit_conv_conf_t &jcp,
            const convolution_desc_t &cd, memory_desc_t &diff_src_d,
            memory_desc_t &weights_d, memory_desc_t &diff_dst_d, int nthreads);
    static void init_scratchpad(memory_tracking::registrar_t &scratchpad,
            const kdnn_jit_conv_conf_t &jcp);;

private:
    using reg64_t = const XReg;
    enum {
        typesize = sizeof(float),
    };
    int ker_reg_base_idx = (jcp.nb_ic_blocking == 1) ? 16 : 24;

    reg64_t param = kdnn_abi_param1;
    reg64_t reg_dst = x1;
    reg64_t reg_ker = x2;
    reg64_t reg_src = x3;

    reg64_t reg_dst_prf = x23;
    reg64_t reg_ker_prf = x5;
    reg64_t reg_src_prf = x6;
    reg64_t reg_iwb = x24;

    reg64_t aux_reg_dst = x7;
    reg64_t aux_reg_ker = x8;

    reg64_t aux_reg_dst_prf = x9;
    reg64_t aux_reg_ker_prf = x10;

    reg64_t aux_reg_dst_d_prf = x6;
    reg64_t aux_reg_dst_d = x11;
    reg64_t aux_reg_ker_d_prf = x12;
    reg64_t aux_reg_ker_d = x2;
    reg64_t reg_ki = x3;

    reg64_t reg_kj = x13;
    reg64_t reg_oi = x11;
    reg64_t reg_kh = x12;

    reg64_t reg_channel = x9;

    reg64_t reg_tmp = x14;
    reg64_t reg_long_offt = x7;

    /* Temporary registers for ARM insts */
    reg64_t reg_prev_bcast_addr = x15;
    reg64_t reg_prev_bcast_addr2 = x17;
    reg64_t reg_prev_bcast_addr3 = x21;
    reg64_t reg_tmp_imm = x16;
    reg64_t reg_tmp_addr = x27;

    reg64_t reg_src_prf_org = x19;
    reg64_t reg_src_org = x20;
    reg64_t reg_oi_org = x25;
    reg64_t reg_dst_org = x22;
    reg64_t reg_ker_org = x26;
    reg64_t reg_input_org = x22;
    reg64_t reg_kernel_org = x26;

    long long int prefetch(const std::string prfop, int level, reg64_t in,
            long long int ofs, long long int prev_ofs) {
        bool for_load = false;
        if (prfop == "LD") {
            for_load = true;
        } else if (prfop == "ST") {
            for_load = false;
        } else {
            assert(!"invalid prfop");
        }

        bool cacheline_aligned = ((ofs & 0xFF) == 0) ? true : false;
        if (cacheline_aligned == true) {
            Prfop op = PLDL1KEEP;
            switch (level) {
                case 1: op = (for_load == true) ? PLDL1KEEP : PSTL1KEEP; break;
                case 2: op = (for_load == true) ? PLDL2KEEP : PSTL2KEEP; break;
                case 3: op = (for_load == true) ? PLDL3KEEP : PSTL3KEEP; break;
                default: assert(!"invalid prfop"); break;
            }

            long long int tmp_ofs = ofs - prev_ofs;
            if ((ofs <= PRFMMAX) && (ofs >= 0)) {
                prfm(op, ptr(in, static_cast<int32_t>(ofs)));
            } else if ((tmp_ofs <= PRFMMAX) && (tmp_ofs >= 0)) {
                prfm(op, ptr(reg_tmp_addr, static_cast<int32_t>(tmp_ofs)));
            } else {
                add_imm(reg_tmp_addr, in, ofs, reg_tmp_imm);
                prfm(op, ptr(reg_tmp_addr));
                prev_ofs = ofs;
            }
        } else {
            PrfopSve op_sve = PLDL1KEEP_SVE;
            switch (level) {
                case 1:
                    op_sve = (for_load == true) ? PLDL1KEEP_SVE : PSTL1KEEP_SVE;
                    break;
                case 2:
                    op_sve = (for_load == true) ? PLDL2KEEP_SVE : PSTL2KEEP_SVE;
                    break;
                case 3:
                    op_sve = (for_load == true) ? PLDL3KEEP_SVE : PSTL3KEEP_SVE;
                    break;
                default: assert(!"invalid prfop"); break;
            }

            long long int tmp_ofs = ofs - prev_ofs;
            if ((kdnn_VL_OFS(ofs, isa) <= PRFWMAX)
                    && (kdnn_VL_OFS(ofs, isa) >= (-1 * PRFWMAX - 1))) {
                prfw(op_sve, P_ALL_ONE,
                        ptr(in, static_cast<int32_t>(kdnn_VL_OFS(ofs, isa))));
            } else if ((kdnn_VL_OFS(tmp_ofs, isa) <= PRFWMAX)
                    && (kdnn_VL_OFS(tmp_ofs, isa) >= (-1 * PRFWMAX - 1))) {
                prfw(op_sve, P_ALL_ONE,
                        ptr(reg_tmp_addr,
                                static_cast<int32_t>(kdnn_VL_OFS(tmp_ofs, isa))));
            } else {
                add_imm(reg_tmp_addr, in, ofs, reg_tmp_imm);
                prfw(op_sve, P_ALL_ONE, ptr(reg_tmp_addr));
                prev_ofs = ofs;
            }
        }
        return prev_ofs;
    }

    ZReg reg_wei = ZReg(31);

    inline void prepare_output(int ur_w);
    inline void store_output(int ur_w);
    inline void compute_loop_fma(int ur_w, int l_overflow, int r_overflow);
    inline void compute_loop_fma_core(
            int ur_w, int l_overflow, int r_overflow, int k_offset);
    inline void compute_loop(
            int ur_w, int l_overflow, int r_overflow, int k_offset = 0);
    void generate() override;

    inline int get_iw_start(int ki, int l_overflow) {
        int res = (jcp.iw - 1 + jcp.r_pad) % jcp.stride_w
                + l_overflow * jcp.stride_w
                - (jcp.kw - 1 - ki) * (jcp.dilate_w + 1);
        while (res < 0)
            res += jcp.stride_w;

        return res;
    }

    inline int get_iw_end(int ur_w, int ki, int r_overflow) {
        if (utils::one_of(ur_w, jcp.iw, jcp.ur_w_tail))
            ur_w += nstl::min(0, jcp.r_pad); // remove negative padding
        int res = (ur_w - 1 + jcp.l_pad) % jcp.stride_w
                + r_overflow * jcp.stride_w - ki * (jcp.dilate_w + 1);
        while (res < 0)
            res += jcp.stride_w;

        return ur_w - res;
    }

    inline size_t get_diff_src_offset(int iw, int icb) {
        const bool is_nxc_layout = is_dsrc_layout_nxc();
        size_t iw_str = is_nxc_layout ? jcp.ngroups * jcp.ic : jcp.ic_block;
        size_t icb_str = is_nxc_layout
                ? jcp.ic_block
                : (size_t)jcp.id * jcp.ih * jcp.iw * jcp.ic_block;

        return typesize * (icb * icb_str + iw * iw_str);
    }

    inline ptrdiff_t get_dst_offset(int iw, int oc, int kw) {
        ptrdiff_t ow
                = (iw + jcp.l_pad - kw * (jcp.dilate_w + 1)) / jcp.stride_w;
        ptrdiff_t ow_str
                = is_ddst_layout_nxc() ? jcp.ngroups * jcp.oc : jcp.oc_block;

        return typesize * (ow * ow_str + oc);
    };

    inline bool is_dsrc_layout_nxc() {
        return utils::one_of(jcp.src_tag, format_tag::ndhwc, format_tag::nhwc,
                format_tag::nwc);
    }
    inline bool is_ddst_layout_nxc() {
        return utils::one_of(jcp.dst_tag, format_tag::ndhwc, format_tag::nhwc,
                format_tag::nwc);
    }
};

} // namespace aarch64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif

