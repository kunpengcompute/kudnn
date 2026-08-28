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


#ifndef KUDNN_JIT_SVE_CONVOLUTION_HPP
#define KUDNN_JIT_SVE_CONVOLUTION_HPP

#include <memory>

#include "kudnn.hpp"

#include "cpu/cpu_convolution_pd.hpp"
#include "cpu/aarch64/kudnn/kudnn_utils.hpp"
#include "cpu/aarch64/cpu_isa_traits.hpp"
#include "cpu/aarch64/kudnn/kudnn_post_ops.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace aarch64 {

using namespace dnnl::impl::status;
using namespace dnnl::impl::memory_tracking::names;
using namespace dnnl::impl::utils;
using namespace dnnl::impl::format_tag;
using namespace dnnl::impl::cpu::aarch64::kdnn_utils;
struct kdnn_convolution_fwd_resource_t : public resource_t {
    kdnn_convolution_fwd_resource_t(const std::unique_ptr<KuDNN::ConvolutionLayerFWD> &kdnn_convolution_prim) noexcept
        : kdnn_convolution_fwd_obj_(new KuDNN::ConvolutionLayerFWD{*(kdnn_convolution_prim.get())}) {}
 
    KuDNN::ConvolutionLayerFWD &get_kdnn_obj() const noexcept { return *kdnn_convolution_fwd_obj_; }
 
    DNNL_DISALLOW_COPY_AND_ASSIGN(kdnn_convolution_fwd_resource_t);
 
private:
    std::unique_ptr<KuDNN::ConvolutionLayerFWD> kdnn_convolution_fwd_obj_;
}; // kdnn_convolution_fwd_resource_t

template <cpu_isa_t isa = isa_undef>
struct kdnn_convolution_fwd_t : public primitive_t {
    struct pd_t : public cpu_convolution_fwd_pd_t {
        using cpu_convolution_fwd_pd_t::cpu_convolution_fwd_pd_t;

        DECLARE_COMMON_PD_T("kdnn", kdnn_convolution_fwd_t);
        status_t init(engine_t *engine) {
            using namespace data_type;
            const auto dst_type = dst_md(0)->data_type;
            memory_desc_t tmp_src_md_jit;
            memory_desc_t tmp_weights_md_jit;
            memory_desc_t tmp_bias_md_jit;
            memory_desc_t tmp_dst_md_jit;
            status_t status;
            bool ok = true && mayiuse(isa) && is_fwd()
                    && set_default_alg_kind(alg_kind::convolution_direct)
                    && attr()->has_default_values(
                            primitive_attr_t::skip_mask_t::post_ops, dst_type)
                    && attr()->has_default_values(primitive_attr_t::skip_mask_t::post_ops)
                    && !has_zero_dim_memory();
            if (!ok) return status::unimplemented;
            const char* value = ::getenv("OMP_NUM_THREADS");
            nthreads = value == NULL ? dnnl_get_max_threads() : atoi(value);
            if (nthreads <= 0) {
                return status::unimplemented;
            }
            const convolution_desc_t &cd = *desc();
            tmp_src_md_jit = src_md_;
            tmp_weights_md_jit = weights_md_;
            tmp_dst_md_jit = dst_md_;
            tmp_bias_md_jit= bias_md_;
            bool is_src_nxc = false;
            bool is_dst_nxc = false;
            bool is_data_nCxc = false;
            bool is_data_layout_nxc = false;
            kdnn_convolution_fwd_t<isa>::get_lt_sta(tmp_src_md_jit, tmp_dst_md_jit,
                tmp_weights_md_jit, is_data_nCxc, is_src_nxc,
                is_dst_nxc, is_data_layout_block);
            if (is_data_nCxc) return status::unimplemented;
            is_data_layout_nxc = is_src_nxc && is_dst_nxc;
            status = kdnn_convolution_fwd_t<isa>::init_conf(
                    cd, tmp_src_md_jit, tmp_weights_md_jit, tmp_dst_md_jit,
                    tmp_bias_md_jit, *attr(), is_data_layout_nxc);
            set_default_formats();
            const memory_desc_wrapper src_d(&src_md_);
            const memory_desc_wrapper weights_d(&weights_md_);
            need_po_wa_ = false;
            dst_size_ = 0;
            data_type_t dst_dt = dst_md_.data_type;
            bool has_post_ops = (attr_.post_ops_.len() >= 1);
            if (attr_.post_ops_.len() > 1) {
                return status::unimplemented; // Currently, only one post-ops is supported.
            }
            if (has_post_ops) {
                dst_md_.data_type = data_type_t::dnnl_f32;
                is_data_layout_block = false;
                is_use_wino_ = false;
            }
            const memory_desc_wrapper dst_d(&dst_md_);
            const memory_desc_wrapper bias_d(&bias_md_);
            if (weights_d.ndims() >= 6) return status::unimplemented;
            KuDNN::Layout src_lt = is_src_nxc
                                ? pick(src_d.ndims() - 3, KuDNN::Layout::ACB, KuDNN::Layout::ACDB, KuDNN::Layout::ACDEB)
                                : get_kdnn_layout(src_d);
            KuDNN::Layout wei_lt = get_kdnn_layout(weights_d);
            KuDNN::Layout dst_lt = is_dst_nxc
                                ? pick(src_d.ndims() - 3, KuDNN::Layout::ACB, KuDNN::Layout::ACDB, KuDNN::Layout::ACDEB)
                                : get_kdnn_layout(dst_d);
            if (wei_lt == KuDNN::Layout::UNDEFINED) {
                return status::unimplemented;
            }
            src_tensor = KuDNN::TensorInfo(get_kdnn_shape(src_d.dims(), src_d.ndims()), get_kdnn_data_t(src_d.data_type()), src_lt);
            wei_tensor = KuDNN::TensorInfo(get_kdnn_shape(weights_d.dims(), weights_d.ndims()), get_kdnn_data_t(weights_d.data_type()), wei_lt);
            dst_tensor = KuDNN::TensorInfo(get_kdnn_shape(dst_d.dims(), dst_d.ndims()), get_kdnn_data_t(dst_d.data_type()), dst_lt);
            bias_tensor = KuDNN::TensorInfo({0}, wei_tensor.GetType(), KuDNN::Layout::A);
            if (bias_d != &glob_zero_md) {
                if (!common_tensor_checks(bias_d)) {
                    return status::unimplemented;
                }
                bias_tensor = get_kdnn_tensor_info(bias_d);
            }
            // get isa
            k_cpu_isa = isa == sve_512 ? 0x1f : 0x07;
            strides = get_kdnn_shape(cd.strides, src_d.ndims() - 2);
            paddingL = get_kdnn_shape(cd.padding[0], src_d.ndims() - 2);
            dilates = get_kdnn_shape(cd.dilates, src_d.ndims() - 2);
            prop = get_kdnn_prop_kind_t(cd.prop_kind);
            bool isJitConv = false;
            if (KuDNN::Status::SUCCESS != KuDNN::ConvolutionLayerFWD::ValidateInput(src_tensor, wei_tensor,
                dst_tensor, bias_tensor, strides, dilates, paddingL, k_cpu_isa, is_data_layout_block, isJitConv,
                is_use_wino_, brg_block, nthreads)) {
                return status::unimplemented;
            }
            const bool withGroups = wei_tensor.GetNumDims() == src_tensor.GetNumDims() + 1;
            int64_t ndims = src_tensor.GetNumDims();
            int64_t kh = (ndims == 3) ? 1 : wei_tensor.GetDims()[withGroups + ndims - 2];
            int64_t kw = wei_tensor.GetDims()[withGroups + ndims - 1];
            int64_t ih = (ndims == 3) ? 1 : src_tensor.GetDims()[ndims - 2];
            int64_t iw = src_tensor.GetDims()[ndims - 1];
            int64_t oh = (ndims == 3) ? 1 : dst_tensor.GetDims()[ndims - 2];
            int64_t ow = dst_tensor.GetDims()[ndims - 1];
            int64_t strideH = (ndims == 3) ? 1 : strides[ndims - 4];
            int64_t strideW = strides[ndims - 3];
            int64_t dilateH = (ndims == 3) ? 0 : dilates[ndims - 4];
            int64_t dilateW = dilates[ndims - 3];
            int64_t tPad = (ndims == 3) ? 0 : paddingL[ndims - 4];
            int64_t lPad = paddingL[ndims - 3];
            int64_t ext_kw = (kw - 1) * (dilateW + 1) + 1;
            int64_t ext_kh = (kh - 1) * (dilateH + 1) + 1;
            int64_t rPad = (ow - 1) * strideW + ext_kw - (iw + lPad);
            int64_t bPad = (oh - 1) * strideH + ext_kh - (ih + tPad);

            if (!has_post_ops && src_tensor.GetLayout() == src_tensor.GetStandardAXBLayout() && dst_tensor.GetLayout() == dst_tensor.GetStandardAXBLayout() &&
                (wei_tensor.GetLayout() == wei_tensor.GetStandardAXBLayout() || wei_tensor.GetLayout() == wei_tensor.GetStandardABXLayout()) &&
                !withGroups && isa == sve_256 && ndims == 4 && kw == kh && kh == 3 && ih == iw && oh == ow &&
                strideH == strideW && strideW == 1 && dilateH == dilateW && dilateW == 0 && lPad == rPad && tPad == bPad &&
                src_d.data_type() == data_type_t::dnnl_f32 && dst_d.data_type() == data_type_t::dnnl_f32 &&
                weights_d.data_type() == data_type_t::dnnl_f32) {
                // use kdnn wino
                is_use_wino_ = true;
                isJitConv = false;
            } else {
                is_use_wino_ = false;
            }
            if (has_post_ops) {
                dst_md_.data_type = dst_dt;
            }
            CHECK(post_ops_.init(engine, attr_.post_ops_, dst_md_));
            if (has_post_ops) {
                isJitConv = false;
                need_po_wa_ = true;
                if ((dst_dt != data_type_t::dnnl_f32) || attr_.post_ops_.entry_[0].is_sum(false)) {
                    dst_size_ = dst_d.nelems() * types::data_type_size(data_type_t::dnnl_f32);
                }
            }
            if (isJitConv) {
                if (kdnn_convolution_fwd_t<isa>::dispatch_to_brgconv(cd, tmp_src_md_jit, tmp_weights_md_jit,
                    tmp_dst_md_jit)) {
                    if (brg_block != 0) {
                        status = kdnn_convolution_fwd_t<isa>::pick_tags(brg_block, tmp_src_md_jit, tmp_weights_md_jit,
                            tmp_dst_md_jit);
                        if (status != status::success) {
                            return status::unimplemented;
                        }
                    } else {
                        return status::unimplemented;
                    }
                } else {
                    if (brg_block != 0) {
                        return status::unimplemented;
                    }
                }
                src_md_ = tmp_src_md_jit;
                weights_md_ = tmp_weights_md_jit;
                bias_md_ = tmp_bias_md_jit;
                dst_md_ = tmp_dst_md_jit;
            }
            auto scratchpad = scratchpad_registry().registrar();
            kdnn_convolution_fwd_t<isa>::init_scratchpad(scratchpad,
                *desc(), src_md_, weights_md_, dst_md_, bias_md_);
            return status;
        }

        KuDNN::TensorInfo src_tensor{KuDNN::TensorInfo({0}, KuDNN::Element::TypeT::F32, KuDNN::Layout::A)};
        KuDNN::TensorInfo dst_tensor{KuDNN::TensorInfo({0}, KuDNN::Element::TypeT::F32, KuDNN::Layout::A)};
        KuDNN::TensorInfo wei_tensor{KuDNN::TensorInfo({0}, KuDNN::Element::TypeT::F32, KuDNN::Layout::A)};
        KuDNN::TensorInfo bias_tensor{KuDNN::TensorInfo({0}, KuDNN::Element::TypeT::F32, KuDNN::Layout::A)};
        KuDNN::Shape strides;
        KuDNN::Shape paddingL;
        KuDNN::Shape dilates;
        KuDNN::Propagation prop;
        int nthreads;
        int brg_block;
        bool is_data_layout_block;
        int k_cpu_isa;
        kdnn_post_ops_t post_ops_;
        bool need_po_wa_;
        bool is_use_wino_ = false;
        size_t dst_size_;
    protected:
        bool set_default_formats() {
            using namespace format_tag;
            auto src_tag = utils::pick(ndims() - 3, ncw, nchw, ncdhw);
            auto wei_tag = with_groups()
                    ? utils::pick(ndims() - 3, goiw, goihw, goidhw)
                    : utils::pick(ndims() - 3, oiw, oihw, oidhw);
            auto dst_tag = utils::pick(ndims() - 3, ncw, nchw, ncdhw);
            return set_default_formats_common(src_tag, wei_tag, dst_tag);
        }
    }; // pd_t

    kdnn_convolution_fwd_t(const pd_t *kpd) : primitive_t(kpd) {}

    status_t init(engine_t *engine) override {
        CHECK(safe_ptr_assign(kernel_,
                new KuDNN::ConvolutionLayerFWD(pd()->k_cpu_isa, pd()->src_tensor, pd()->wei_tensor,
                    pd()->dst_tensor, pd()->bias_tensor, pd()->strides, pd()->paddingL, pd()->dilates,
                    pd()->prop, pd()->nthreads, pd()->is_data_layout_block, pd()->is_use_wino_)));
        return status::success;
    }

    static bool dispatch_to_brgconv(const convolution_desc_t &cd, const memory_desc_wrapper &src_d,
            const memory_desc_wrapper &weights_d, const memory_desc_wrapper &dst_d);
    
    static status_t pick_tags(int oc_block, memory_desc_t &src_md, memory_desc_t &weights_md,
            memory_desc_t &dst_md);

    static status_t init_conf(
            const convolution_desc_t &cd, memory_desc_t &src_pd,
            memory_desc_t &weights_pd, memory_desc_t &dst_pd,
            memory_desc_t &bias_pd, const primitive_attr_t &attr,
            bool &is_data_layout_nxc);

    static void get_lt_sta(const memory_desc_t &src_pd, const memory_desc_t &dst_pd,
        const memory_desc_t &wei_pd, bool& is_data_nCxc,
        bool &is_src_layout_nxc, bool &is_dst_layout_nxc,
        bool &is_data_block);
    
    static void init_scratchpad(memory_tracking::registrar_t &scratchpad, const convolution_desc_t &cd,
        memory_desc_t &src_md, memory_desc_t &weights_md, memory_desc_t &dst_md, memory_desc_t &bias_md);

    status_t create_resource(
            engine_t *engine, resource_mapper_t &mapper) const override {
        if (mapper.has_resource(this)) return status::success;

        mapper.add(this, std::make_unique<kdnn_convolution_fwd_resource_t>(kernel_));
        CHECK(pd()->post_ops_.create_resource(engine, mapper));
 
        return status::success;
    }

    status_t execute(const exec_ctx_t &ctx) const override {
        execute_forward(ctx);
        if (pd()->wants_zero_pad_dst()) ctx.zero_pad_output(DNNL_ARG_DST);
        return status::success;
    }
private:
    // execute_forward has to be const thus mutability of mtx
    mutable std::mutex mtx;

    void prepare_padded_bias(const void *&bias,
        const memory_tracking::grantor_t &scratchpad, const memory_desc_wrapper &src_md,
        const memory_desc_wrapper &dst_md, const memory_desc_wrapper &weights_md) const;
    status_t execute_forward(const exec_ctx_t &ctx) const;

    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }

    std::unique_ptr<KuDNN::ConvolutionLayerFWD> kernel_;
}; // kdnn_convolution_fwd_t

} // namespace aarch64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif // KUDNN_JIT_SVE_CONVOLUTION_HPP
