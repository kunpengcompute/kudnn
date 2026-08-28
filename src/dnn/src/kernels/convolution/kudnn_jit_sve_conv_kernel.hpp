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
#include "internal_funcs.hpp"

namespace KuDNN {

inline IntType GetVlenOfs(IntType ofs, const ARMCG::CPUISAT& isa)
{
    static_cast<void>(isa);
#ifdef KUDNNL_920F
    return ofs >> ARMCG::CpuIsaTraits<ARMCG::CPUISAT::VC_512>::VLEN_SHIFT;
#else
    return ofs >> ARMCG::CpuIsaTraits<ARMCG::CPUISAT::VC_256>::VLEN_SHIFT;
#endif
}

// Check the immediate value for LDR(vector) instruction
// The imm9 in the LDR instruction is the optional signed immediate vector
// offset, in the range -256 to 255, defaulting to 0.
template <typename T, ARMCG::CPUISAT isa = ARMCG::CPUISAT::VC_512>
bool KdnnLdrImmCheck(T ofs)
{
    IntType vlen = ARMCG::CpuIsaTraits<isa>::VLEN;
    IntType vlenShift = ARMCG::CpuIsaTraits<isa>::VLEN_SHIFT;
    IntType shiftedOfs = ofs >> vlenShift;
    return ((shiftedOfs) <= LDRMAX) && (shiftedOfs >= LDRMIN)
            && ((ofs % vlen) == 0);
}

// Check the immediate value for SDR(vector) instruction
// The imm9 in the STR instruction is the optional signed immediate vector
// offset, in the range -256 to 255, defaulting to 0.
template <typename T, ARMCG::CPUISAT isa = ARMCG::CPUISAT::VC_512>
bool KdnnStrImmCheck(T ofs)
{
    IntType vlen = ARMCG::CpuIsaTraits<isa>::VLEN;
    IntType vlenShift = ARMCG::CpuIsaTraits<isa>::VLEN_SHIFT;
    IntType shiftedOfs = ofs >> vlenShift;
    return ((shiftedOfs) <= STRMAX) && (shiftedOfs >= STRMIN)
            && ((ofs % vlen) == 0);
}

// Check the immediate value for LD1RW instruction
// The imm6 in the LD1RW instruction is the optional unsigned immediate byte
// offset, a multiple of 4 in the range 0 to 252, defaulting to 0.
template <typename T>
bool KdnnLd1rwImmCheck(T ofs)
{
    return ((ofs & 0x3) == 0) && (ofs <= LD1RWMAX) && (ofs >= 0);
}

// Check the immediate value for PRFM(immediate) instruction
//
// The pimm in the PRFM instruction is the optional positive immediate byte
// offset, a multiple of 8 in the range 0 to 32760, defaulting to 0.
template <typename T>
bool KdnnPrfmImmCheck(T ofs)
{
    return (ofs <= PRFMMAX) && (ofs >= PRFMMIN) && ((ofs & 0x7) == 0);
}

// Check the immediate value for PRFW(scalar plus immediate) instruction
//
// The imm6 in the PRFW instruction is the optional signed immediate vector
// offset, in the range -32 to 31, defaulting to 0.
template <typename T, ARMCG::CPUISAT isa = ARMCG::CPUISAT::VC_512>
bool KdnnPrfwImmCheck(T ofs)
{
    IntType vlen = ARMCG::CpuIsaTraits<isa>::VLEN;
    IntType vlenShift = ARMCG::CpuIsaTraits<isa>::VLEN_SHIFT;
    IntType shiftedOfs = ofs >> vlenShift;
    return (shiftedOfs <= PRFWMAX) && (shiftedOfs >= PRFWMIN)
            && ((ofs % vlen) == 0);
}

class ZRegFactory {
public:
    ZRegFactory() {}
    ~ZRegFactory() {}
public:
    static std::unique_ptr<ARMCG::ZRegBase> GetZRegAcdTp4Fmov(const Element::TypeT& dtp, const IntType& idx)
    {
        switch (dtp) {
            case Element::TypeT::F32: {
                std::unique_ptr<ARMCG::ZRegBase> pZreg = std::make_unique<ARMCG::ZRegS>(idx);
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

    static std::unique_ptr<ARMCG::ZRegBase> GetZRegAcdTp(const Element::TypeT& dtp, const IntType& idx)
    {
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
struct KdnnJitConvFwdKernel : public ARMCG::JitGenerator {
    explicit KdnnJitConvFwdKernel(const KdnnConvJitConfT &ajcp) : jcp(ajcp) {}
    KdnnConvJitConfT jcp;
public:
    const char *Name() const override
    {
        return "KdnnJitConvFwdKernel";
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

    inline void PrepareOutput(IntType urW);
    inline void StoreOutput(IntType urW);
    inline void StoreOutputFadd(IntType rOfs, IntType nRegs, IntType& preOutOfs, IntType urW);
#ifdef KUDNNL_920B
    inline void StoreOutputForVReg(IntType urW);
#endif
    inline void ComputeLoopCommon(IntType urW, IntType padL, IntType padR);
    inline void ComputeLoopFma(IntType urW, IntType padL, IntType padR);
    inline void ComputeLoopFmaCore(IntType urW, IntType padL, IntType padR);
    inline void ComputeLoop(IntType urW, IntType padL, IntType padR);

    void Generate() override;
    void DealNoiIsZero(IntType oShift, IntType urw, IntType leftPad,
        IntType rightPad, IntType rightPad1, SizeType inShiftPad, IntType sw, IntType urwTail);
    void DealLpadLessThanUrw(IntType oShift, SizeType iShift, IntType urw, IntType leftPad,
        IntType rightPad, IntType rightPad1, IntType inShiftPad, IntType urwTail, IntType noi);
    void DealGeneralLoop(IntType oShift, IntType inMult, IntType urw, IntType leftPad, IntType rightPad,
        IntType rightPad1, IntType urwTail, IntType sw, IntType iw, IntType kw, IntType& noi);

    inline SizeType GetOutputOffset(IntType oi, IntType nOcBlock)
    {
        const bool isNxcLayout = IsDstLayoutNxc();
        size_t owStr = isNxcLayout ? jcp.ngroups * jcp.oc : jcp.ocBlock;
        SizeType ocbStr = isNxcLayout
                        ? jcp.ocBlock
                        : static_cast<SizeType>(jcp.od) * jcp.oh * jcp.ow * jcp.ocBlock;
        return jcp.typeSizeOut * (nOcBlock * ocbStr + oi * owStr);
    }

    inline SizeType GetInputOffset(IntType ki, IntType ic, IntType oi, IntType padL)
    {
        const bool isNxcLayout = IsSrcLayoutNxc();
        size_t iwStr = isNxcLayout ? jcp.ngroups * jcp.ic
                                   : (!jcp.is1stConv ? jcp.icBlock : 1);
        SizeType icStr = !jcp.is1stConv || isNxcLayout
                       ? 1
                       : static_cast<SizeType>(jcp.iw) * jcp.ih * jcp.id;
        SizeType iwIdx = ki * (jcp.dilateW + 1) + oi * jcp.strideW - padL;
        return jcp.typeSizeIn * (iwIdx * iwStr + ic * icStr);
    }

    inline IntType GetOwStart(IntType ki, IntType padL)
    {
        IntType owStart = Math::CeilDiv(padL - ki * (jcp.dilateW + 1), jcp.strideW);
        return owStart > 0 ? owStart : 0;
    }

    inline IntType GetOwEnd(IntType urW, IntType ki, IntType padR)
    {
        IntType resOwEnd = Math::CeilDiv(padR - (jcp.kw - 1 - ki) * (jcp.dilateW + 1), jcp.strideW);
        return urW - (resOwEnd > 0 ? resOwEnd : 0);
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

    inline IntType WeiLoadVC(SizeType auxKernelOffset, IntType regIdx, IntType prevOfs)
    {
        SizeType ofs = auxKernelOffset;
        if (KdnnLdrImmCheck<SizeType, isa>(ofs)) {
            Ldr(ARMCG::ZRegS(regIdx), ARMCG::AdrScImm(Ptr(auxRegKer, static_cast<int32_t>(GetVlenOfs(ofs, isa)))));
        } else {
            SizeType ofsTmp = ofs - prevOfs;
            if ((prevOfs != -1) && KdnnLdrImmCheck<SizeType, isa>(ofsTmp)) {
                Ldr(ARMCG::ZRegS(regIdx), ARMCG::AdrScImm(Ptr(regPrevWeiAddr,
                    static_cast<int32_t>(GetVlenOfs(ofsTmp, isa)))));
            } else {
                if ((prevOfs != -1) && (ofsTmp > 0)) {
                    ofsTmp = auxKernelOffset - prevOfs;
                    AddImm(regPrevWeiAddr, regPrevWeiAddr, ofsTmp,
                        regTmpImm);
                } else {
                    AddImm(regPrevWeiAddr, auxRegKer, ofs, regTmpImm);
                }
                Ld1w(ARMCG::ZRegS(regIdx), pAllOne / ARMCG::PredType::T_Z,
                    Ptr(regPrevWeiAddr));
                prevOfs = static_cast<IntType>(ofs);
            }
        }
        return prevOfs;
    };
};
} // namespace KuDNN

#endif
