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

#ifndef CPU_AARCH64_KUDNN_JIT_INJECTOR_UTILS_HPP
#define CPU_AARCH64_KUDNN_JIT_INJECTOR_UTILS_HPP

#include <array>
#include <cstddef>
#include <set>
#include <stack>

#include "kudnn_jit_generator.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace aarch64 {
namespace kdnn_injector_utils {

using vmm_index_set_t = typename std::set<size_t>;
using vmm_index_set_iterator_t = typename std::set<size_t>::iterator;
template <cpu_isa_t isa>
struct vmm_size_t;

template <>
struct vmm_size_t<sve_256> {
    static constexpr std::size_t bytes = 32u;
};

enum class layout_t { ncsp, c_blocked, nspc, cspn, unsupported };

inline layout_t get_layout_type(const memory_desc_wrapper &dst_d) {
    const auto strides = dst_d.blocking_desc().strides;
    if (!dst_d.is_plain()) return layout_t::c_blocked;
    if (strides[0] >= strides[1]
            && IMPLICATION(dst_d.ndims() >= 3, strides[1] >= strides[2]))
        return layout_t::ncsp;
    if (strides[1] == 1) return layout_t::nspc;
    if (strides[0] == 1) return layout_t::cspn;
    return layout_t::unsupported;
}

/*
 * Scope guard for general purpose register and vector registers preservation.
 * Pushes registers to stack during construction and pops during destruction.
 */
template <cpu_isa_t isa>
class kdnn_register_preserve_guard_t {

public:
    kdnn_register_preserve_guard_t(kdnn_jit_generator *host,
            std::initializer_list<Xbyak_aarch64::XReg> reg64_to_preserve,
            std::initializer_list<Xbyak_aarch64::VReg> vmm_to_preserve = {});
    kdnn_register_preserve_guard_t(kdnn_register_preserve_guard_t &&other) = default;
    kdnn_register_preserve_guard_t &operator=(kdnn_register_preserve_guard_t &&other)
            = default;
    DNNL_DISALLOW_COPY_AND_ASSIGN(kdnn_register_preserve_guard_t);
    ~kdnn_register_preserve_guard_t();
    size_t calc_vmm_to_preserve_size_bytes(
            const std::initializer_list<Xbyak_aarch64::VReg> &vmm_to_preserve)
            const;
    size_t stack_space_occupied() const;

private:
    kdnn_jit_generator *host_;
    std::stack<Xbyak_aarch64::XReg> reg64_stack_;
    std::stack<Xbyak_aarch64::VReg> vmm_stack_;
    const uint64_t cpu_sveLen_ = get_sve_length();
    size_t vmm_to_preserve_size_bytes_;
};

template <cpu_isa_t isa>
class kdnn_conditional_register_preserve_guard_t
    : public kdnn_register_preserve_guard_t<isa> {
public:
    kdnn_conditional_register_preserve_guard_t(bool condition_to_be_met,
            kdnn_jit_generator *host,
            std::initializer_list<Xbyak_aarch64::XReg> reg64_to_preserve,
            std::initializer_list<Xbyak_aarch64::VReg> vmm_to_preserve = {});
    DNNL_DISALLOW_COPY_AND_ASSIGN(kdnn_conditional_register_preserve_guard_t);
};

} // namespace injector_utils
} // namespace aarch64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif
