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

#ifndef KUDNN_JIT_CONV_KERNEL_HPP
#define KUDNN_JIT_CONV_KERNEL_HPP
#include <memory>
#include "cg_jit_generator.hpp"
#include "kudnn_conv_const.hpp"
#include "kudnn_primitive_conf.hpp"


namespace KuDNN {

inline int GetVlenOfs(int ofs, const ARMCG::CPUISAT& isa)
{
    static_cast<void>(isa);
#ifdef KUDNNL_920Pro
    return ofs >> ARMCG::CpuIsaTraits<ARMCG::CPUISAT::VectorComputation_512>::VLEN_SHIFT;
#else
    return ofs >> ARMCG::CpuIsaTraits<ARMCG::CPUISAT::VectorComputation_256>::VLEN_SHIFT;
#endif
}

// Check the immediate value for LDR(vector) instruction
// The imm9 in the LDR instruction is the optional signed immediate vector
// offset, in the range -256 to 255, defaulting to 0.
template <typename T, ARMCG::CPUISAT isa = ARMCG::CPUISAT::VectorComputation_512>
bool KudnnLdrImmCheck(T ofs)
{
    int vlen = ARMCG::CpuIsaTraits<isa>::VLEN;
    int vlenShift = ARMCG::CpuIsaTraits<isa>::VLEN_SHIFT;
    int shiftedOfs = ofs >> vlenShift;
    return ((shiftedOfs) <= LDRMAX) && (shiftedOfs >= LDRMIN)
            && ((ofs % vlen) == 0);
}

// Check the immediate value for SDR(vector) instruction
// The imm9 in the STR instruction is the optional signed immediate vector
// offset, in the range -256 to 255, defaulting to 0.
template <typename T, ARMCG::CPUISAT isa = ARMCG::CPUISAT::VectorComputation_512>
bool KudnnStrImmCheck(T ofs)
{
    int vlen = ARMCG::CpuIsaTraits<isa>::VLEN;
    int vlenShift = ARMCG::CpuIsaTraits<isa>::VLEN_SHIFT;
    int shiftedOfs = ofs >> vlenShift;
    return ((shiftedOfs) <= STRMAX) && (shiftedOfs >= STRMIN)
            && ((ofs % vlen) == 0);
}

// Check the immediate value for LD1RW instruction
// The imm6 in the LD1RW instruction is the optional unsigned immediate byte
// offset, a multiple of 4 in the range 0 to 252, defaulting to 0.
template <typename T>
bool KudnnLd1rwImmCheck(T ofs)
{
    return ((ofs & 0x3) == 0) && (ofs <= LD1RWMAX) && (ofs >= 0);
}

// Check the immediate value for PRFM(immediate) instruction
//
// The pimm in the PRFM instruction is the optional positive immediate byte
// offset, a multiple of 8 in the range 0 to 32760, defaulting to 0.
template <typename T>
bool KudnnPrfmImmCheck(T ofs)
{
    return (ofs <= PRFMMAX) && (ofs >= PRFMMIN) && ((ofs & 0x7) == 0);
}

// Check the immediate value for PRFW(scalar plus immediate) instruction
//
// The imm6 in the PRFW instruction is the optional signed immediate vector
// offset, in the range -32 to 31, defaulting to 0.
template <typename T, ARMCG::CPUISAT isa = ARMCG::CPUISAT::VectorComputation_512>
bool KudnnPrfwImmCheck(T ofs)
{
    int vlen = ARMCG::CpuIsaTraits<isa>::VLEN;
    int vlenShift = ARMCG::CpuIsaTraits<isa>::VLEN_SHIFT;
    int shiftedOfs = ofs >> vlenShift;
    return (shiftedOfs <= PRFWMAX) && (shiftedOfs >= PRFWMIN)
            && ((ofs % vlen) == 0);
}

class ZRegFactory {
public:
    ZRegFactory() {}
    ~ZRegFactory() {}
public:
    static std::unique_ptr<ARMCG::ZRegBase> GetZRegAcdTp4Fmov(const Element::TypeT& dtp, const int idx)
    {
        assert(idx >= 0);
        switch (dtp) {
            case Element::TypeT::F32: {
                std::unique_ptr<ARMCG::ZRegBase> pZreg = std::make_unique<ARMCG::ZRegH>(idx);
                return pZreg;
            }
            case Element::TypeT::F16:
            case Element::TypeT::BF16: {
                std::unique_ptr<ARMCG::ZRegBase> pZreg = std::make_unique<ARMCG::ZRegH>(idx);
                return pZreg;
            }
            default: {
                std::unique_ptr<ARMCG::ZRegBase> pZreg(nullptr);
                return pZreg;
            }
        }
    }

    static std::unique_ptr<ARMCG::ZRegBase> GetZRegAcdTp(const Element::TypeT& dtp, const int idx)
    {
        assert(idx >= 0);
        switch (dtp) {
            case Element::TypeT::F32:
            case Element::TypeT::BF16: {
                std::unique_ptr<ARMCG::ZRegBase> pZreg = std::make_unique<ARMCG::ZRegS>(idx);
                return pZreg;
            }
            case Element::TypeT::F16: {
                std::unique_ptr<ARMCG::ZRegBase> pZreg = std::make_unique<ARMCG::ZRegH>(idx);
                return pZreg;
            }
            default: {
                std::unique_ptr<ARMCG::ZRegBase> pZreg(nullptr);
                return pZreg;
            }
        }
    }
};

template <ARMCG::CPUISAT isa = ARMCG::CPUISAT::ISA_UNDEF, Element::TypeT dtp = Element::TypeT::F32>
struct KudnnJitConvFwdKernel : public ARMCG::JitGenerator {
    explicit KudnnJitConvFwdKernel(const KudnnConvJitConfT &ajcp) : jcp(ajcp) {}
    KudnnConvJitConfT jcp;
public:
    const char *Name() const override
    {
        return "KudnnJitConvFwdKernel";
    }
    const char *SourceFile() const override
    {
        return __FILE__;
    }
private:
    using Reg64T = const ARMCG::XReg;

    Reg64T param = ARMCG::ABI_PARAM1;
    Reg64T regInp = x1; // src base addr (2d)
    Reg64T regKer = x2; // ker base addr (2d)
    Reg64T auxRegKerD = x2; // ker addr (3d)
    Reg64T regOut = x3; // dst base addr (2d)
    Reg64T regKi = x3; // d-dim loop var (3d)
    Reg64T regOutPrf = x6; // addr for Prefetch

    Reg64T auxRegInp = x7; // src addr (main loop)
    Reg64T auxRegInp2 = x24; // src addr (main loop)
    Reg64T auxRegInp3 = x25; // src addr (main loop)
    Reg64T regOutOfs = x7; // dst addr (StoreOutput)
    Reg64T auxRegKer = x8; // ker addr (main loop)
    Reg64T regChannel = x9; // reduce workload
    Reg64T regBias = x10; // bias addr (prepare_out)

    Reg64T auxRegInpD = x11; // src addr (3d)
    Reg64T regOi = x11;

    Reg64T regKh = x12; // ker h size
    Reg64T regKj = x13; // ker h workload

    /* Temporary registers for ARM insts */
    Reg64T regTmpAddr = x14;
    Reg64T regPrevBcastAddr = x15;
    Reg64T regPrevWeiAddr = x16;
    Reg64T regTmpImm = x17;

    Reg64T regOutOrg = x27; // dst base addr (3d)
    Reg64T regOiOrg = x19; // base oi (3d)
    Reg64T auxRegKerOrg = x20;
    // register conflict with post-op eltwise
    Reg64T regKerOrg = x22; // ker base addr (3d)
    Reg64T regInpOrg = x29; // src base addr (3d)

    inline void PrepareOutput(int urW);
    inline void StoreOutput(int urW);
    inline void ComputeLoopFma(int urW, int padL, int padR);
    inline void ComputeLoopFmaCore(int urW, int padL, int padR);
    inline void ComputeLoop(int urW, int padL, int padR);

    void Generate() override;

    inline size_t GetOutputOffset(int oi, int nOcBlock)
    {
        const bool isNxcLayout = IsDstLayoutNxc();
        size_t owStr = isNxcLayout ? jcp.ngroups * jcp.oc : jcp.ocBlock;
        size_t ocbStr = isNxcLayout
                        ? jcp.ocBlock
                        : static_cast<size_t>(jcp.od) * jcp.oh * jcp.ow * jcp.ocBlock;
        return jcp.typeSizeOut * (nOcBlock * ocbStr + oi * owStr);
    }

    inline size_t GetInputOffset(int ki, int ic, int oi, int padL)
    {
        const bool isNxcLayout = IsSrcLayoutNxc();
        size_t iwStr = isNxcLayout ? jcp.ngroups * jcp.ic
                                   : (!jcp.is1stConv ? jcp.icBlock : 1);
        size_t icStr = !jcp.is1stConv || isNxcLayout
                       ? 1
                       : static_cast<size_t>(jcp.iw) * jcp.ih * jcp.id;
        size_t iwIdx = ki * (jcp.dilateW + 1) + oi * jcp.strideW - padL;
        return jcp.typeSizeIn * (iwIdx * iwStr + ic * icStr);
    }

    inline int DivUp(int value, int divisor)
    {
        return (value + divisor - 1) / divisor;
    }

    inline int GetOwStart(int ki, int padL)
    {
        return std::max(0, DivUp(padL - ki * (jcp.dilateW + 1), jcp.strideW));
    }

    inline int GetOwEnd(int urW, int ki, int padR)
    {
        return urW - std::max(0, DivUp(padR - (jcp.kw - 1 - ki) * (jcp.dilateW + 1), jcp.strideW));
    }

    inline bool IsSrcLayoutNxc()
    {
        return jcp.srcTag == Layout::NDHWC
                || jcp.srcTag == Layout::NHWC
                || jcp.srcTag == Layout::ACB;
    }

    inline bool IsDstLayoutNxc()
    {
        return jcp.dstTag == Layout::NDHWC
                || jcp.dstTag == Layout::NHWC
                || jcp.dstTag == Layout::ACB;
    }

    inline void FmovZReg(const Element::TypeT& tp, std::unique_ptr<ARMCG::ZRegBase> pZReg)
    {
        assert(pZReg.get() != nullptr);
        switch (tp) {
            case Element::TypeT::F32: {
                ARMCG::ZRegS* pZRegS = static_cast<ARMCG::ZRegS*>(pZReg.get());
                Fmov(*pZRegS);
                break;
            }
            case Element::TypeT::F16:
            case Element::TypeT::BF16: {
                ARMCG::ZRegH* pZRegH = static_cast<ARMCG::ZRegH*>(pZReg.get());
                Fmov(*pZRegH);
                break;
            }
            default: {
                break;
            }
        }
    }
};
} // namespace KuDNN

#endif
