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

#include "kudnn_jit_sve_convolution.hpp"
#include "cpu/aarch64/jit_primitive_conf.hpp"
#include "cpu/aarch64/kudnn/kudnn_utils.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace aarch64 {

using namespace dnnl::impl::status;
using namespace dnnl::impl::memory_tracking::names;
using namespace dnnl::impl::utils;
using namespace dnnl::impl::format_tag;
using namespace dnnl::impl::cpu::aarch64::kdnn_utils;

using namespace nstl;

namespace {

template <cpu_isa_t isa>
inline bool check_dw_conv(const memory_desc_wrapper &src_d,
        const memory_desc_wrapper &weights_d, const memory_desc_wrapper &dst_d)
{
    if (src_d.ndims() != weights_d.ndims() - 1) {
        return false;
    }
    if (src_d.dims()[1] != weights_d.dims()[0] ||
        dst_d.dims()[1] != weights_d.dims()[0]) {
        return false;
    }
    if (src_d.ndims() != 4) {
        return false;
    }
    bool is_data_layout_nxc =
        src_d.matches_one_of_tag(format_tag::nhwc) != format_tag::undef;
    int simd_w = cpu_isa_traits<isa>::vlen / sizeof(float);
    if (is_data_layout_nxc && src_d.dims()[1] % simd_w != 0) {
        return false;
    }
    return true;
}



inline status_t init_tag(format_tag_t &tag, memory_desc_t &md,
        const memory_desc_wrapper &mdw, const format_tag_t tag_value) {
    if (mdw.format_kind() == format_kind::any) {
        CHECK(memory_desc_init_by_tag(md, tag_value));
        tag = tag_value;
    } else {
        tag = mdw.matches_one_of_tag(tag_value);
    }
    return status::success;
}

} // namespace

template <cpu_isa_t isa>
bool kdnn_convolution_fwd_t<isa>::dispatch_to_brgconv(const convolution_desc_t &cd,
    const memory_desc_wrapper &src_d, const memory_desc_wrapper &weights_d, const memory_desc_wrapper &dst_d)
{
    // check input
    int ndims = src_d.ndims();
    format_tag_t data_tag_nxc = pick(ndims - 3, nwc, nhwc, ndhwc);
    bool is_src_nxc = src_d.matches_one_of_tag(data_tag_nxc) == data_tag_nxc;
    bool is_dst_nxc = dst_d.matches_one_of_tag(data_tag_nxc) == data_tag_nxc;
    bool input_ok = true
        // check data type
        && src_d.data_type() == data_type::f32
        && weights_d.data_type() == data_type::f32
        && dst_d.data_type() == data_type::f32
        // check layout
        && (src_d.format_any() || is_src_nxc)
        && weights_d.format_any()
        && (dst_d.format_any() || is_dst_nxc)
        // check prop kind
        && cd.prop_kind != prop_kind::backward;
    if (!input_ok) {
        return false;
    }

    // skip case
    bool with_groups = src_d.ndims() == weights_d.ndims() - 1;
    dim_t ngroups = with_groups ? weights_d.dims()[0] : 1;
    dim_t ic = src_d.dims()[1] / ngroups;
    dim_t iw = src_d.dims()[ndims - 1];
    dim_t oc = dst_d.dims()[1] / ngroups;
    dim_t ow = dst_d.dims()[ndims - 1];
    bool skip_case = false
        // dw conv, direct is better
        || (with_groups && ngroups > 1 && ic == 1 && oc == 1)
        // grouped small ic, direct is better
        || (with_groups && ngroups > 1
            && IMPLICATION(one_of(src_d.data_type(), data_type::bf16, data_type::u8, data_type::s8), ic % 4 == 0 && oc % 4 == 0)
            && ic < static_cast<dim_t>(isa_max_vlen(isa)) / static_cast<dim_t>(types::data_type_size(src_d.data_type())))
        // some case can be better than direct sme/sve
        || (isa == sve_512 && (!(ic <= 256 && iw <= 56 && oc <= 512 && ow <= 28) && !(ic >= 384 && iw >= 300 && oc <= 1024)));
    if (skip_case) {
        return false;
    }

    // brdgmm_dw: unimplemented
    // brgconv_1x1: supported
    // brgconv: unimplemented
    dim_t id = (ndims == 5) ? src_d.dims()[2] : 1;
    dim_t ih = (ndims == 3) ? 1 : src_d.dims()[ndims - 2];
    dim_t od = (ndims == 5) ? dst_d.dims()[2] : 1;
    dim_t oh = (ndims == 3) ? 1 : dst_d.dims()[ndims - 2];
    dim_t kd = (ndims == 5) ? weights_d.dims()[with_groups + 2] : 1;
    dim_t kh = (ndims == 3) ? 1 : weights_d.dims()[with_groups + ndims - 2];
    dim_t kw = weights_d.dims()[with_groups + ndims - 1];
    dim_t f_pad = (ndims == 5) ? cd.padding[0][0] : 0;
    dim_t t_pad = (ndims == 3) ? 0 : cd.padding[0][ndims - 4];
    dim_t l_pad = cd.padding[0][ndims - 3];
    dim_t stride_d = (ndims == 5) ? cd.strides[0] : 1;
    dim_t stride_h = (ndims == 3) ? 1 : cd.strides[ndims - 4];
    dim_t stride_w = cd.strides[ndims - 3];
    dim_t dilate_d = (ndims == 5) ? cd.dilates[0] : 0;
    dim_t dilate_h = (ndims == 3) ? 0 : cd.dilates[ndims - 4];
    dim_t dilate_w = cd.dilates[ndims - 3];
    dim_t ext_kd = (kd - 1) * (dilate_d + 1) + 1;
    dim_t ext_kh = (kh - 1) * (dilate_h + 1) + 1;
    dim_t ext_kw = (kw - 1) * (dilate_w + 1) + 1;
    dim_t back_pad = (od - 1) * stride_d + ext_kd - (id + f_pad);
    dim_t b_pad = (oh - 1) * stride_h + ext_kh - (ih + t_pad);
    dim_t r_pad = (ow - 1) * stride_w + ext_kw - (iw + l_pad);
    bool is_1x1 = f_pad <= 0 && back_pad <= 0 && t_pad <= 0 && b_pad <= 0 && l_pad <= 0 && r_pad <= 0
        && utils::everyone_is(1, kd, kh, kw);
    if (!is_1x1) {
        return false;
    }

    return true;
}

template <cpu_isa_t isa>
status_t kdnn_convolution_fwd_t<isa>::pick_tags(int oc_block,
    memory_desc_t &src_md, memory_desc_t &weights_md, memory_desc_t &dst_md)
{
    const memory_desc_wrapper src_d(&src_md);
    const memory_desc_wrapper weights_d(&weights_md);
    const memory_desc_wrapper dst_d(&dst_md);

    const bool wei_plain = false
        && is_superset(isa, cpu_isa_t::sve_512)
        && weights_d.data_type() == data_type::f32
        && weights_d.is_plain();

    int ndims = src_md.ndims;
    const bool is_1d = ndims == 3;
    const bool is_2d = ndims == 4;
    const bool is_3d = ndims == 5;
    const int vnni_granularity = data_type_vnni_granularity(weights_d.data_type());
    format_tag_t src_tag, wei_tag, dst_tag;
    dst_tag = pick(ndims - 3, nwc, nhwc, ndhwc);
    const bool with_groups = src_d.ndims() == weights_d.ndims() - 1;
    const bool is_ic_padded = false;

    if (wei_plain) {
        if (is_3d) {
            switch (vnni_granularity) {
                case 1: wei_tag = with_groups ? dhwigo : dhwio; break;
                case 2: wei_tag = with_groups ? gdhwIo2i : dhwIo2i; break;
                case 4: wei_tag = with_groups ? gdhwIo4i : dhwIo4i; break;
                default: return status::unimplemented;
            }
        } else if (is_2d) {
            switch (vnni_granularity) {
                case 1: wei_tag = with_groups ? hwigo : hwio; break;
                case 2: wei_tag = with_groups ? ghwIo2i : hwIo2i; break;
                case 4: wei_tag = with_groups ? ghwIo4i : hwIo4i; break;
                default: return status::unimplemented;
            }
        } else {
            assert(is_1d);
            UNUSED(is_1d);
            switch (vnni_granularity) {
                case 1: wei_tag = with_groups ? wigo : wio; break;
                case 2: wei_tag = with_groups ? gwIo2i : wIo2i; break;
                case 4: wei_tag = with_groups ? gwIo4i : wIo4i; break;
                default: return status::unimplemented;
            }
        }
    } else {
        if (oc_block == 64) {
            if (is_3d) {
                switch (vnni_granularity) {
                    case 1: wei_tag = with_groups ? gOdhwi64o : Odhwi64o; break;
                    case 2: wei_tag = is_ic_padded
                        ? (with_groups ? gOdhwI16i64o2i : OdhwI16i64o2i)
                        : (with_groups ? gOdhwI64o2i : OdhwI64o2i); break;
                    case 4: wei_tag = is_ic_padded
                        ? (with_groups ? gOdhwI16i64o4i : OdhwI16i64o4i)
                        : (with_groups ? gOdhwI64o4i : OdhwI64o4i); break;
                    default: return status::unimplemented;
                }
            } else if (is_2d) {
                switch (vnni_granularity) {
                    case 1: wei_tag = with_groups ? gOhwi64o : Ohwi64o; break;
                    case 2: wei_tag = is_ic_padded
                        ? (with_groups ? gOhwI16i64o2i : OhwI16i64o2i)
                        : (with_groups ? gOhwI64o2i : OhwI64o2i); break;
                    case 4: wei_tag = is_ic_padded
                        ? (with_groups ? gOhwI16i64o4i : OhwI16i64o4i)
                        : (with_groups ? gOhwI64o4i : OhwI64o4i); break;
                    default: return status::unimplemented;
                }
            } else {
                assert(is_1d);
                UNUSED(is_1d);
                switch (vnni_granularity) {
                    case 1: wei_tag = with_groups ? gOwi64o : Owi64o; break;
                    case 2: wei_tag = is_ic_padded
                        ? (with_groups ? gOwI16i64o2i : OwI16i64o2i)
                        : (with_groups ? gOwI64o2i : OwI64o2i); break;
                    case 4: wei_tag = is_ic_padded
                        ? (with_groups ? gOwI16i64o4i : OwI16i64o4i)
                        : (with_groups ? gOwI64o4i : OwI64o4i); break;
                    default: return status::unimplemented;
                }
            }
        } else if (oc_block == 48) {
            if (is_3d) {
                switch (vnni_granularity) {
                    case 1: wei_tag = with_groups ? gOdhwi48o : Odhwi48o; break;
                    case 2: wei_tag = is_ic_padded
                        ? (with_groups ? gOdhwI16i48o2i : OdhwI16i48o2i)
                        : (with_groups ? gOdhwI48o2i : OdhwI48o2i); break;
                    case 4: wei_tag = is_ic_padded
                        ? (with_groups ? gOdhwI16i48o4i : OdhwI16i48o4i)
                        : (with_groups ? gOdhwI48o4i : OdhwI48o4i); break;
                    default: return status::unimplemented;
                }
            } else if (is_2d) {
                switch (vnni_granularity) {
                    case 1: wei_tag = with_groups ? gOhwi48o : Ohwi48o; break;
                    case 2: wei_tag = is_ic_padded
                        ? (with_groups ? gOhwI16i48o2i : OhwI16i48o2i)
                        : (with_groups ? gOhwI48o2i : OhwI48o2i); break;
                    case 4: wei_tag = is_ic_padded
                        ? (with_groups ? gOhwI16i48o4i : OhwI16i48o4i)
                        : (with_groups ? gOhwI48o4i : OhwI48o4i); break;
                    default: return status::unimplemented;
                }
            } else {
                assert(is_1d);
                UNUSED(is_1d);
                switch (vnni_granularity) {
                    case 1: wei_tag = with_groups ? gOwi48o : Owi48o; break;
                    case 2: wei_tag = is_ic_padded
                        ? (with_groups ? gOwI16i48o2i : OwI16i48o2i)
                        : (with_groups ? gOwI48o2i : OwI48o2i); break;
                    case 4: wei_tag = is_ic_padded
                        ? (with_groups ? gOwI16i48o4i : OwI16i48o4i)
                        : (with_groups ? gOwI48o4i : OwI48o4i); break;
                    default: return status::unimplemented;
                }
            }
        } else if (oc_block == 32) {
            if (is_3d) {
                switch (vnni_granularity) {
                    case 1: wei_tag = with_groups ? gOdhwi32o : Odhwi32o; break;
                    case 2: wei_tag = is_ic_padded
                        ? (with_groups ? gOdhwI16i32o2i : OdhwI16i32o2i)
                        : (with_groups ? gOdhwI32o2i : OdhwI32o2i); break;
                    case 4: wei_tag = is_ic_padded
                        ? (with_groups ? gOdhwI16i32o4i : OdhwI16i32o4i)
                        : (with_groups ? gOdhwI32o4i : OdhwI32o4i); break;
                    default: return status::unimplemented;
                }
            } else if (is_2d) {
                switch (vnni_granularity) {
                    case 1: wei_tag = with_groups ? gOhwi32o : Ohwi32o; break;
                    case 2: wei_tag = is_ic_padded
                        ? (with_groups ? gOhwI16i32o2i : OhwI16i32o2i)
                        : (with_groups ? gOhwI32o2i : OhwI32o2i); break;
                    case 4: wei_tag = is_ic_padded
                        ? (with_groups ? gOhwI16i32o4i : OhwI16i32o4i)
                        : (with_groups ? gOhwI32o4i : OhwI32o4i); break;
                    default: return status::unimplemented;
                }
            } else {
                assert(is_1d);
                UNUSED(is_1d);
                switch (vnni_granularity) {
                    case 1: wei_tag = with_groups ? gOwi32o : Owi32o; break;
                    case 2: wei_tag = is_ic_padded
                        ? (with_groups ? gOwI16i32o2i : OwI16i32o2i)
                        : (with_groups ? gOwI32o2i : OwI32o2i); break;
                    case 4: wei_tag = is_ic_padded
                        ? (with_groups ? gOwI16i32o4i : OwI16i32o4i)
                        : (with_groups ? gOwI32o4i : OwI32o4i); break;
                    default: return status::unimplemented;
                }
            }
        } else if (oc_block == 24) {
            if (is_3d) {
                switch (vnni_granularity) {
                    case 1: wei_tag = with_groups ? gOdhwi24o : Odhwi24o; break;
                    case 2: wei_tag = with_groups ? gOdhwI24o2i : OdhwI24o2i; break;
                    case 4: wei_tag = with_groups ? gOdhwI24o4i : OdhwI24o4i; break;
                    default: return status::unimplemented;
                }
            } else if (is_2d) {
                switch (vnni_granularity) {
                    case 1: wei_tag = with_groups ? gOhwi24o : Ohwi24o; break;
                    case 2: wei_tag = with_groups ? gOhwI24o2i : OhwI24o2i; break;
                    case 4: wei_tag = with_groups ? gOhwI24o4i : OhwI24o4i; break;
                    default: return status::unimplemented;
                }
            } else {
                assert(is_1d);
                UNUSED(is_1d);
                switch (vnni_granularity) {
                    case 1: wei_tag = with_groups ? gOwi24o : Owi24o; break;
                    case 2: wei_tag = with_groups ? gOwI24o2i : OwI24o2i; break;
                    case 4: wei_tag = with_groups ? gOwI24o4i : OwI24o4i; break;
                    default: return status::unimplemented;
                }
            }
        } else if (oc_block == 16) {
            if (is_3d) {
                switch (vnni_granularity) {
                    case 1: wei_tag = with_groups ? gOdhwi16o : Odhwi16o; break;
                    case 2: wei_tag = is_ic_padded
                        ? (with_groups ? gOdhwI16i16o2i : OdhwI16i16o2i)
                        : (with_groups ? gOdhwI16o2i : OdhwI16o2i); break;
                    case 4: wei_tag = is_ic_padded
                        ? (with_groups ? gOdhwI16i16o4i : OdhwI16i16o4i)
                        : (with_groups ? gOdhwI16o4i : OdhwI16o4i); break;
                    default: return status::unimplemented;
                }
            } else if (is_2d) {
                switch (vnni_granularity) {
                    case 1: wei_tag = with_groups ? gOhwi16o : Ohwi16o; break;
                    case 2: wei_tag = is_ic_padded
                        ? (with_groups ? gOhwI16i16o2i : OhwI16i16o2i)
                        : (with_groups ? gOhwI16o2i : OhwI16o2i); break;
                    case 4: wei_tag = is_ic_padded
                        ? (with_groups ? gOhwI16i16o4i : OhwI16i16o4i)
                        : (with_groups ? gOhwI16o4i : OhwI16o4i); break;
                    default: return status::unimplemented;
                }
            } else {
                assert(is_1d);
                UNUSED(is_1d);
                switch (vnni_granularity) {
                    case 1: wei_tag = with_groups ? gOwi16o : Owi16o; break;
                    case 2: wei_tag = is_ic_padded
                        ? (with_groups ? gOwI16i16o2i : OwI16i16o2i)
                        : (with_groups ? gOwI16o2i : OwI16o2i); break;
                    case 4: wei_tag = is_ic_padded
                        ? (with_groups ? gOwI16i16o4i : OwI16i16o4i)
                        : (with_groups ? gOwI16o4i : OwI16o4i); break;
                    default: return status::unimplemented;
                }
            }
        } else if (oc_block == 8) {
            if (is_3d) {
                switch (vnni_granularity) {
                    case 1: wei_tag = with_groups ? gOdhwi8o : Odhwi8o; break;
                    case 2: wei_tag = with_groups ? gOdhwI8o2i : OdhwI8o2i; break;
                    case 4: wei_tag = with_groups ? gOdhwI8o4i : OdhwI8o4i; break;
                    default: return status::unimplemented;
                }
            } else if (is_2d) {
                switch (vnni_granularity) {
                    case 1: wei_tag = with_groups ? gOhwi8o : Ohwi8o; break;
                    case 2: wei_tag = with_groups ? gOhwI8o2i : OhwI8o2i; break;
                    case 4: wei_tag = with_groups ? gOhwI8o4i : OhwI8o4i; break;
                    default: return status::unimplemented;
                }
            } else {
                assert(is_1d);
                UNUSED(is_1d);
                switch (vnni_granularity) {
                    case 1: wei_tag = with_groups ? gOwi8o : Owi8o; break;
                    case 2: wei_tag = with_groups ? gOwI8o2i : OwI8o2i; break;
                    case 4: wei_tag = with_groups ? gOwI8o4i : OwI8o4i; break;
                    default: return status::unimplemented;
                }
            }
        } else {
            return status::unimplemented;
        }
    }

    src_tag = dst_tag;

    CHECK(init_tag(src_tag, src_md, src_d, src_tag));
    CHECK(init_tag(wei_tag, weights_md, weights_d, wei_tag));
    CHECK(init_tag(dst_tag, dst_md, dst_d, dst_tag));

    return status::success;
}

template <cpu_isa_t isa>
void kdnn_convolution_fwd_t<isa>::prepare_padded_bias(
        const void *&bias,
        const memory_tracking::grantor_t &scratchpad, const memory_desc_wrapper &src_md,
        const memory_desc_wrapper &dst_md, const memory_desc_wrapper &weights_md) const {
    if (!pd()->wants_padded_bias()) return;
    bool with_groups = weights_md.ndims() == src_md.ndims() + 1;
    int ngroups = with_groups ? weights_md.dims()[0] : 1;
    bool is_dw_conv = check_dw_conv<isa>(src_md, weights_md, dst_md);
    int oc = is_dw_conv ? dst_md.dims()[1] : dst_md.dims()[1] / ngroups;
    int ocWithoutPadding = oc;
    auto bias_type = pd()->bias_tensor.GetType();
    if (ngroups == 1 || is_dw_conv) {
        if (bias_type == KuDNN::Element::TypeT::F32) {
            int simd_w = cpu_isa_traits<isa>::vlen / sizeof(float);
            oc = (oc + simd_w - 1) / simd_w * simd_w;
        } else if (bias_type == KuDNN::Element::TypeT::F16) {
            int simd_w = cpu_isa_traits<isa>::vlen / sizeof(float16_t);
            oc = (oc + simd_w - 1) / simd_w * simd_w;
        } else if (bias_type == KuDNN::Element::TypeT::BF16) {
            int simd_w = cpu_isa_traits<isa>::vlen / sizeof(bfloat16_t);
            oc = (oc + simd_w - 1) / simd_w * simd_w;
        }
    }
    if (bias_type == KuDNN::Element::TypeT::F32) {
        auto padded_bias
            = scratchpad.template get<float>(key_conv_padded_bias);
        utils::array_copy(padded_bias, static_cast<const float *>(bias), ocWithoutPadding);
        utils::array_set(padded_bias + ocWithoutPadding, (float)0,
            oc - ocWithoutPadding);
        bias = padded_bias;
    } else if (bias_type == KuDNN::Element::TypeT::F16) {
        auto padded_bias
            = scratchpad.template get<float16_t>(key_conv_padded_bias);
        utils::array_copy(padded_bias, static_cast<const float16_t *>(bias), ocWithoutPadding);
        utils::array_set(padded_bias + ocWithoutPadding, (float16_t)0,
            oc - ocWithoutPadding);
        bias = padded_bias;
    } else if (bias_type == KuDNN::Element::TypeT::BF16) {
        auto padded_bias
            = scratchpad.template get<bfloat16_t>(key_conv_padded_bias);
        utils::array_copy(padded_bias, static_cast<const bfloat16_t *>(bias), ocWithoutPadding);
        utils::array_set(padded_bias + ocWithoutPadding, (bfloat16_t)0,
            oc - ocWithoutPadding);
        bias = padded_bias;
    }
}

template <cpu_isa_t isa>
status_t kdnn_convolution_fwd_t<isa>::execute_forward(const exec_ctx_t &ctx) const {
    auto src = CTX_IN_MEM(const void *, DNNL_ARG_SRC);
    auto weights = CTX_IN_MEM(const void *, DNNL_ARG_WEIGHTS);
    auto bias = CTX_IN_MEM(const void *, DNNL_ARG_BIAS);
    auto dst = CTX_OUT_MEM(void *, DNNL_ARG_DST);
    const memory_desc_wrapper src_d(pd()->src_md());
    const memory_desc_wrapper dst_d(pd()->dst_md());
    const memory_desc_wrapper weights_d(pd()->weights_md(0));
    if (pd()->is_use_wino_) {
        setenv("USE_KUDNN_WINO", "on", 1);
    } else {
        setenv("USE_KUDNN_WINO", "off", 1);
    }
    prepare_padded_bias(bias, ctx.get_scratchpad_grantor(), src_d, dst_d, weights_d);
    if (pd()->need_po_wa_) {
        try {
            if (pd()->dst_size_ != 0) {
                std::unique_ptr<float, KuDNN::Service::Deallocator<float>> tmp_buffer(
                    static_cast<float*>(KuDNN::Service::AlignedAlloc(pd()->dst_size_)));
                kernel_->Run(src, weights, tmp_buffer.get(), bias);
                pd()->post_ops_.execute(ctx, tmp_buffer.get(), dst);
            } else {
                kernel_->Run(src, weights, dst, bias);
                pd()->post_ops_.execute(ctx, dst, dst);
            }
        } catch (const std::exception& e) {
            return status::runtime_error;
        }
    } else {
        try {
            kernel_->Run(src, weights, dst, bias);
        } catch (const std::exception& e) {
            return status::runtime_error;
        }
    }
    return status::success;
}

template <cpu_isa_t isa>
void kdnn_convolution_fwd_t<isa>::get_lt_sta(const memory_desc_t &src_md, const memory_desc_t &dst_md,
        const memory_desc_t &wei_md, bool& is_data_nCxc,
        bool &is_src_layout_nxc, bool &is_dst_layout_nxc,
        bool &is_data_block) {
    using namespace prop_kind;
    if (!mayiuse(isa)) { return; }
    const memory_desc_wrapper src_d(&src_md);
    const memory_desc_wrapper dst_d(&dst_md);
    const memory_desc_wrapper weights_d(&wei_md);
    int ndims = src_d.ndims();
    const auto dat_tag_nxc = pick(ndims - 3, nwc, nhwc, ndhwc);
    const auto dat_tag_ncx = pick(ndims - 3, ncw, nchw, ncdhw);
    const auto data_tag_cxn = pick(ndims - 3, bca, bcda, bcdea);
    const auto data_tag_xcn = pick(ndims - 3, cba, cdba, cdeba);
    const auto dat_tag_nCx8c = pick(ndims - 3, nCw8c, nChw8c, nCdhw8c);
    const auto dat_tag_nCx16c = pick(ndims - 3, nCw16c, nChw16c, nCdhw16c);
    auto curr_src_tag = src_d.matches_one_of_tag(
            dat_tag_nxc, dat_tag_nCx16c, dat_tag_nCx8c, dat_tag_ncx, data_tag_xcn, data_tag_cxn);
    auto curr_dst_tag = dst_d.matches_one_of_tag(
            dat_tag_nxc, dat_tag_nCx16c, dat_tag_nCx8c, dat_tag_ncx, data_tag_xcn, data_tag_cxn);
    is_data_nCxc
        = (dat_tag_nCx8c == curr_src_tag) || (dat_tag_nCx16c == curr_src_tag) ||
          (dat_tag_nCx8c == curr_dst_tag) || (dat_tag_nCx16c == curr_dst_tag);
    is_src_layout_nxc = curr_src_tag == dat_tag_nxc ? true : false;
    is_dst_layout_nxc = curr_dst_tag == dat_tag_nxc ? true : false;
    bool is_wei_any = weights_d.format_any();
    bool cur_layout_any = src_d.format_any() && dst_d.format_any();
    bool cur_layout_axb = curr_src_tag == dat_tag_nxc && curr_dst_tag == dat_tag_nxc;
    is_data_block = is_wei_any && (cur_layout_any || cur_layout_axb);
}

template <cpu_isa_t isa>
status_t kdnn_convolution_fwd_t<isa>::init_conf(
        const convolution_desc_t &cd, memory_desc_t &src_md,
        memory_desc_t &weights_md, memory_desc_t &dst_md,
        memory_desc_t &bias_md, const primitive_attr_t &attr,
        bool &is_data_layout_nxc) {
    using namespace prop_kind;
    if (!mayiuse(isa)) { return status::unimplemented; }
    const memory_desc_wrapper src_d(&src_md);
    const memory_desc_wrapper weights_d(&weights_md);
    const memory_desc_wrapper dst_d(&dst_md);
    const memory_desc_wrapper bias_d(&bias_md);

    if (kdnn_convolution_fwd_t<isa>::dispatch_to_brgconv(cd, src_d, weights_d, dst_d)) {
        if (cd.bias_desc.format_kind != format_kind::undef) {
            if (bias_d.format_kind() == format_kind::any)
                CHECK(memory_desc_init_by_tag(bias_md, x));
        }
        return status::success;
    }

    int ndims = src_d.ndims();
    const bool is_1d = ndims == 3;
    const bool is_3d = ndims == 5;
    const bool with_groups = weights_d.ndims() == src_d.ndims() + 1;
    if (isa == sve_512 && with_groups && src_d.data_type() == data_type::f16 && dst_d.data_type() == data_type::f16) {
        return status::unimplemented;
    }
    int ngroups = with_groups ? weights_d.dims()[0] : 1;
    int ic = src_d.dims()[1] / ngroups;
    int oc = dst_d.dims()[1] / ngroups;
    bool is_depthwise = with_groups && everyone_is(1, ic, oc);
    /* 1st convolution check */
    int simd_w = cpu_isa_traits<isa>::vlen / sizeof(float);
    const bool is_dw_conv = check_dw_conv<isa>(src_d, weights_d, dst_d);
    if (src_d.data_type() == data_type::f16 && dst_d.data_type() == data_type::f16) {
        simd_w = cpu_isa_traits<isa>::vlen / sizeof(float16_t);
    }
    bool is1stConv = (ic < simd_w && ngroups == 1 && !is_dw_conv);

    const auto dat_tag_nxc = pick(ndims - 3, nwc, nhwc, ndhwc);
    const auto dat_tag_ncx = pick(ndims - 3, ncw, nchw, ncdhw);
    const auto dat_tag_nCx4c = pick(ndims - 3, nCw4c, nChw4c, nCdhw4c);
    const auto dat_tag_nCx8c = pick(ndims - 3, nCw8c, nChw8c, nCdhw8c);
    const auto dat_tag_nCx16c = pick(ndims - 3, nCw16c, nChw16c, nCdhw16c);
    const auto dat_tag_nCx32c = pick(ndims - 3, nCw32c, nChw32c, nCdhw32c);
    auto curr_src_tag = src_d.matches_one_of_tag(
            dat_tag_nxc, dat_tag_nCx16c, dat_tag_nCx8c, dat_tag_ncx);
    auto curr_dst_tag = dst_d.matches_one_of_tag(
            dat_tag_nxc, dat_tag_nCx16c, dat_tag_nCx8c, dat_tag_ncx);
    is_data_layout_nxc
        = utils::everyone_is(dat_tag_nxc, curr_src_tag, curr_dst_tag);
    int dilate_h = (ndims == 3) ? 0 : cd.dilates[ndims - 4];
    int dilate_w = cd.dilates[ndims - 3];
    bool isIntType = src_d.data_type() == data_type::s8 || src_d.data_type() == data_type::u8;
    int ic_block = 16;
    int oc_block = 16;
    if (isa == sve_256) {
        ic_block = 8;
        oc_block = 8;
    }
    if (ngroups == 1) {
        ic = rnd_up(ic, ic_block);
        oc = rnd_up(oc, oc_block);
    }
    if (isIntType && ngroups != 1 && ((ic % ic_block != 0) || (oc % oc_block != 0))) {
        ic_block = (ic % 8 == 0) && (oc % 8 == 0) ? 8 : 4;
        oc_block = ic_block;
    }
    format_tag_t src_tag, dst_tag, wei_tag;
    switch (isa) {
        case sve_512:
            if (src_d.data_type() == data_type::f32 && !is_data_layout_nxc && ngroups == 1 &&
                (ndims == 4 || ndims == 5) && oc % 16 == 0 && dilate_h == 0 && dilate_w == 0) {
                // USE SME
                dst_tag = dat_tag_ncx;
                src_tag = dat_tag_ncx;
                wei_tag = pick(2 * ndims - 6 + with_groups, Owi16o, gOwi16o,
                    Ohwi16o, gOhwi16o, Odhwi16o, gOdhwi16o);
            } else {
                // USE SVE
                dst_tag = is_data_layout_nxc ? dat_tag_nxc : dat_tag_nCx16c;
                //src_tag = is_data_layout_nxc ? dat_tag_nxc : dat_tag_nCx16c;
                src_tag = is_data_layout_nxc ? dat_tag_nxc : (is1stConv ? dat_tag_ncx : dat_tag_nCx16c);
                wei_tag = pick(2 * ndims - 6 + with_groups, OIw16i16o, gOIw16i16o,
                        OIhw16i16o, gOIhw16i16o, OIdhw16i16o, gOIdhw16i16o);
                if (src_d.data_type() == data_type::f16) {
                    dst_tag = is_data_layout_nxc ? dat_tag_nxc : dat_tag_nCx32c;
                    src_tag = is_data_layout_nxc ? dat_tag_nxc : (is1stConv ? dat_tag_ncx : dat_tag_nCx16c);
                    wei_tag = pick(ndims - 3, OIw16i32o, OIhw16i32o, OIdhw16i32o);
                }
            }
            break;
        case sve_256:
            dst_tag = is_data_layout_nxc ? dat_tag_nxc : dat_tag_nCx8c;
            //src_tag = is_data_layout_nxc ? dat_tag_nxc : dat_tag_nCx8c;
            src_tag = is_data_layout_nxc ? dat_tag_nxc : (is1stConv ? dat_tag_ncx : dat_tag_nCx8c);
            wei_tag = pick(2 * ndims - 6 + with_groups, OIw8i8o, gOIw8i8o,
                    OIhw8i8o, gOIhw8i8o, OIdhw8i8o, gOIdhw8i8o);
            if (ngroups > 1 && ((ic % 8 != 0 && ic % 4 == 0) || (oc % 8 != 0 && oc % 4 == 0))) {
                dst_tag = is_data_layout_nxc ? dat_tag_nxc : dat_tag_nCx4c;
                src_tag = is_data_layout_nxc ? dat_tag_nxc : dat_tag_nCx4c;
                wei_tag = pick(ndims - 3, gOIw4i4o, gOIhw4i4o, gOIdhw4i4o);
            }
            if (src_d.data_type() == data_type::f16) {
                dst_tag = is_data_layout_nxc ? dat_tag_nxc : dat_tag_nCx16c;
                src_tag = is_data_layout_nxc ? dat_tag_nxc : (is1stConv ? dat_tag_ncx : dat_tag_nCx16c);
                wei_tag = pick(2 * ndims - 6 + with_groups, OIw16i16o, gOIw16i16o,
                    OIhw16i16o, gOIhw16i16o, OIdhw16i16o, gOIdhw16i16o);
            }
            break;
        default: break;
    }
    bool isNeedSetIntLayout = (ic % ic_block == 0 && oc % oc_block == 0) && isIntType;
    if (isNeedSetIntLayout && ic_block == 16) {
        dst_tag = dat_tag_nxc;
        src_tag = dat_tag_nxc;
        if (is_3d) {
            wei_tag = with_groups ? gOIdhw4i16o4i : OIdhw4i16o4i; //OIdhw4i8o4i
        } else if (is_1d) {
            wei_tag = with_groups ? gOIw4i16o4i : OIw4i16o4i;
        } else {
            wei_tag = with_groups ? gOIhw4i16o4i : OIhw4i16o4i;
        }
    } else if (isNeedSetIntLayout && ic_block == 8) {
        dst_tag = dat_tag_nxc;
        src_tag = dat_tag_nxc;
        if (is_3d) {
            wei_tag = with_groups ? gOIdhw2i8o4i : OIdhw2i8o4i; //OIdhw4i8o4i
        } else if (is_1d) {
            wei_tag = with_groups ? gOIw2i8o4i : OIw2i8o4i;
        } else {
            wei_tag = with_groups ? gOIhw2i8o4i : OIhw2i8o4i;
        }
    } else if (isNeedSetIntLayout && ic_block == 4) {
        dst_tag = dat_tag_nxc;
        src_tag = dat_tag_nxc;
        if (is_3d) {
            wei_tag = with_groups ? gOIdhw4o4i : OIdhw4o4i; //OIdhw4i8o4i
        } else if (is_1d) {
            wei_tag = with_groups ? gOIw4o4i : OIw4o4i;
        } else {
            wei_tag = with_groups ? gOIhw4o4i : OIhw4o4i;
        }
    }
    if (src_d.format_kind() == format_kind::any)
        CHECK(memory_desc_init_by_tag(src_md, src_tag));

    if (dst_d.format_kind() == format_kind::any)
        CHECK(memory_desc_init_by_tag(dst_md, dst_tag));

    bool withBias = cd.bias_desc.format_kind != format_kind::undef;
    if (withBias) {
        if (bias_d.format_kind() == format_kind::any)
            CHECK(memory_desc_init_by_tag(bias_md, x));
    }
    if (is1stConv) {
        switch (isa) {
            case sve_512:
                wei_tag = with_groups
                        ? pick(ndims - 3, gOwi16o, gOhwi16o, gOdhwi16o)
                        : pick(ndims - 3, Owi16o, Ohwi16o, Odhwi16o);
                if (src_d.data_type() == data_type::f16) {
                    wei_tag = pick(ndims - 3, Owi32o, Ohwi32o, Odhwi32o);
                }
                break;
            case sve_256:
                wei_tag = with_groups
                        ? pick(ndims - 3, gOwi8o, gOhwi8o, gOdhwi8o)
                        : pick(ndims - 3, Owi8o, Ohwi8o, Odhwi8o);
                if (src_d.data_type() == data_type::f16) {
                    wei_tag = with_groups
                        ? pick(ndims - 3, gOwi16o, gOhwi16o, gOdhwi16o)
                        : pick(ndims - 3, Owi16o, Ohwi16o, Odhwi16o);
                }
                break;
            default: break;
        }
    }
    if (src_d.data_type() == data_type::f32 && is_dw_conv) {
        wei_tag = isa == sve_512 ? Goihw16g : Goihw8g;
    }
    format_tag_t w_tag;
    init_tag(w_tag, weights_md, weights_d, wei_tag);
    if (src_d.data_type() == data_type::u8) {
        weights_md.extra.flags = 0
                | memory_extra_flags::compensation_conv_s8s8
                | memory_extra_flags::scale_adjust;
        weights_md.extra.compensation_mask = (1 << 0) +
            (with_groups && !is_depthwise ? (1 << 1) : 0);
        weights_md.extra.scale_adjust = 1.f;
    }
    return status::success;
}

template <cpu_isa_t isa>
void kdnn_convolution_fwd_t<isa>::init_scratchpad(
        memory_tracking::registrar_t &scratchpad, const convolution_desc_t &cd, memory_desc_t &src_md,
        memory_desc_t &weights_md, memory_desc_t &dst_md, memory_desc_t &bias_md) {

    const memory_desc_wrapper src_d(&src_md);
    const memory_desc_wrapper weights_d(&weights_md);
    const memory_desc_wrapper dst_d(&dst_md);
    const memory_desc_wrapper bias_d(&bias_md);
    bool with_groups = weights_d.ndims() == src_d.ndims() + 1;
    int ngroups = with_groups ? weights_d.dims()[0] : 1;
    bool is_dw_conv = check_dw_conv<isa>(src_d, weights_d, dst_d);
    int oc = is_dw_conv ? dst_d.dims()[1] : dst_d.dims()[1] / ngroups;
    int ocWithoutPadding = oc;
    if (ngroups == 1 || is_dw_conv) {
        if (src_d.data_type() == data_type::f32) {
            int simd_w = cpu_isa_traits<isa>::vlen / sizeof(float);
            oc = (oc + simd_w - 1) / simd_w * simd_w;
        } else if (src_d.data_type() == data_type::f16) {
            int simd_w = cpu_isa_traits<isa>::vlen / sizeof(float16_t);
            oc = (oc + simd_w - 1) / simd_w * simd_w;
        } else if (src_d.data_type() == data_type::bf16) {
            int simd_w = cpu_isa_traits<isa>::vlen / sizeof(bfloat16_t);
            oc = (oc + simd_w - 1) / simd_w * simd_w;
        }
    }
    bool withBias = cd.bias_desc.format_kind != format_kind::undef;
    if (withBias && ocWithoutPadding != oc) {
        if (src_d.data_type() == data_type::f32) {
            scratchpad.book(key_conv_padded_bias, oc, sizeof(float));
        } else if (src_d.data_type() == data_type::f16) {
            scratchpad.book(key_conv_padded_bias, oc, sizeof(float16_t));
        } else if (src_d.data_type() == data_type::bf16) {
            scratchpad.book(key_conv_padded_bias, oc, sizeof(bfloat16_t));
        }
    }
}

template struct kdnn_convolution_fwd_t<sve_256>;
template struct kdnn_convolution_fwd_t<sve_512>;

} // namespace aarch64
} // namespace cpu
} // namespace impl
} // namespace dnnl
