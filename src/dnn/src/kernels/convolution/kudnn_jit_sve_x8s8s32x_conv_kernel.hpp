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
#ifndef KUDNN_JIT_VC_X8S8S32X_CONV_KERNEL_HPP
#define KUDNN_JIT_VC_X8S8S32X_CONV_KERNEL_HPP
#include <memory>
#include <climits>
#include <cstdint>

#include "cg_jit_generator.hpp"
#include "kudnn_conv_const.hpp"
#include "kudnn_primitive_conf.hpp"
#include "internal_funcs.hpp"
#include "tools/cg_util.hpp"

namespace KuDNN {
template <typename T>
struct NumericLimits {};

template <>
struct NumericLimits<float> : public std::numeric_limits<float> {};

template <>
struct NumericLimits<int32_t> : public std::numeric_limits<int32_t> {};

template <>
struct NumericLimits<int16_t> : public std::numeric_limits<int16_t> {};

template <>
struct NumericLimits<uint16_t> : public std::numeric_limits<uint16_t> {};

template <>
struct NumericLimits<int8_t> : public std::numeric_limits<int8_t> {};

template <>
struct NumericLimits<uint8_t> : public std::numeric_limits<uint8_t> {};

template <typename T>
inline T MaxValue(Element::TypeT dataType)
{
    if (dataType == Element::TypeT::S32) {
        return static_cast<T>(NumericLimits<int32_t>::max());
    } else if (dataType == Element::TypeT::S8) {
        return static_cast<T>(NumericLimits<int8_t>::max());
    } else {
        return static_cast<T>(NumericLimits<uint8_t>::max());
    }
}

struct KdnnJitX8S8S32ConvFwdKernel : public ARMCG::JitGenerator {
    explicit KdnnJitX8S8S32ConvFwdKernel(const KdnnConvJitConfT &ajcp) : jcp(ajcp)
    {
        IntType chBlock = jcp.icBlock;
#ifdef KUDNNL_920F
        if (chBlock == IC_LEN16) {
            sveLen_ = VC_LEN_64;
        } else if (chBlock == IC_LEN8) {
            sveLen_ = VC_LEN_32;
        } else {
            sveLen_ = VC_LEN_16;
        }
#else
        if (chBlock == IC_LEN8) {
            sveLen_ = VC_LEN_32;
        } else {
            sveLen_ = VC_LEN_16;
        }
#endif
    }
    KdnnConvJitConfT jcp;
public:
    const char *Name() const override
    {
        return "KdnnJitConvIntFwdKernel";
    }
    const char *SourceFile() const override
    {
        return __FILE__;
    }
private:
    using Reg64T = const ARMCG::XReg;
    enum class IcBlockT {
        NO_LAST_BLOCK,
        LAST_IC_BLOCK,
        LAST_SP_BLOCK,
    };
    /* data regs */
    Reg64T regPtrScales = x7;
    Reg64T auxRegSaturation = x7; // src base addr (2d)
    Reg64T regInp = x8;
    Reg64T regKer = x9; // ker base addr (2d)
    Reg64T regOut = x10; // dst base addr (2d)
    Reg64T auxRegInp = x11; // src addr (main loop)
    Reg64T regPtrSumScale = x11;
    Reg64T auxRegKer = x12;
    Reg64T auxRegInpD = x13;
    Reg64T regCompensation = x14;
    Reg64T auxRegKerD = x15;
    Reg64T regInpBufferPtr = auxRegInpD;
    Reg64T auxRegInpBufferPtr = auxRegKerD;

    /* counter regs */
    Reg64T regBiasAlpha = x1;
    Reg64T regParam1 = x0;
    Reg64T regOi = x3;
    Reg64T regBias = x2;
    Reg64T regOcBlocks = x6;
    Reg64T regOwb = auxRegKer;
    Reg64T regScratch = regCompensation;
    Reg64T regKj = regPtrScales;
    Reg64T regKi = regCompensation;
    Reg64T regOverflow = regPtrScales;
    Reg64T regIcb = regBias;
    Reg64T regJmpTblBase = regKj;
    ARMCG::XReg regStack = x22;
    ARMCG::XReg regTmp0Imm = x23;
    ARMCG::XReg regTmp1Imm = x24;
    ARMCG::XReg regTmp0Adr = x25;
    const ARMCG::PReg kTailMask = p2;
    const ARMCG::PReg kBlendMask = p8;
    const ARMCG::PReg maskTmp = p3;
    const ARMCG::PReg maskTmp2 = p9;
    const ARMCG::PReg maskAllOne = p4;
    const ARMCG::ZReg vmmWei = ARMCG::ZReg(REG_INDEX31);
    const ARMCG::ZReg vmmComp = ARMCG::ZReg(REG_INDEX30);
    const ARMCG::ZReg vmmBias = ARMCG::ZReg(REG_INDEX31);
    const ARMCG::ZReg vmmPreLoad = ARMCG::ZReg(REG_INDEX30);
    const ARMCG::ZReg vmmPrevDst = ARMCG::ZReg(REG_INDEX31);
    const ARMCG::ZReg vmmSaturation = ARMCG::ZReg(REG_INDEX30);
    const ARMCG::ZReg vmmZero = ARMCG::ZReg(REG_INDEX31);
    const ARMCG::ZReg vmmShift = vmmComp;
    const ARMCG::ZReg vmmTmp = ARMCG::ZReg(REG_INDEX28);
    const ARMCG::ZReg vmmOne = ARMCG::ZReg(REG_INDEX29);
    const ARMCG::ZReg zmmWei = ARMCG::ZReg(REG_INDEX31);
    ARMCG::ZReg zmmSrc = ARMCG::ZReg(REG_INDEX0);
    ARMCG::ZReg zmmShiftZero = ARMCG::ZReg(REG_INDEX0);
    ARMCG::ZReg zmmPermute = ARMCG::ZReg(REG_INDEX0);
    bool maskGFlag = false;
    IntType sveLen_;
private:
    ARMCG::ZReg VmmOut(const IntType& iUr, const IntType& iOc);
    ARMCG::ZReg VmmInp(const IntType& iIC, const IntType& nbXBlocking);
    void PrepareOutput(const IntType& urW);
    void StoreOutput(const IntType& urW, bool bLastOcBlockFlag);
    void ComputeKer(IntType urW, IntType padL, IntType padR,
        IcBlockT lastIcBlockFlag, bool hPadded = false);
    void KhLoop(IntType urW, IntType padL, IntType padR, IcBlockT lastIcBlockFlag);
    void IcbLoop(IntType urW, IntType padL, IntType padR, bool bVal);
    void Generate() override;
    void Cvt2Ps(Element::TypeT typeIn, ARMCG::ZReg vmmIn, const ARMCG::XReg regBase,
        const IntType offset, bool maskFlag);
    IntType GetOffset(IntType rawOfft);
    ARMCG::XReg GetCompAddrReg(ARMCG::XReg base, IntType offset = 0);
    void VmmMaskAllOne();
};
} // namespace KuDNN

#endif