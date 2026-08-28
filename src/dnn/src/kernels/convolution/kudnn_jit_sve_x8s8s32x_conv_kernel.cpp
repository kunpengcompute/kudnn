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
#include "kudnn_jit_sve_x8s8s32x_conv_kernel.hpp"
namespace KuDNN {
using namespace ARMCG;
using namespace Element;

void KdnnJitX8S8S32ConvFwdKernel::PrepareOutput(const IntType& urW)
{
    IntType nbOcBlock = jcp.nbOcBlocking;
    for (IntType k = 0; k < nbOcBlock; k++) {
        for (IntType j = 0; j < urW; j++) {
            auto vmm = VmmOut(j, k);
            Eor(vmm.d, vmm.d, vmm.d);
        }
    }
    if (!jcp.signedInput) {
        Eor(regScratch, regScratch, regScratch);
        Dup(vmmShift.b, S8_MIN_VAL);
    }
}

void KdnnJitX8S8S32ConvFwdKernel::Cvt2Ps(TypeT typeIn, ZReg vmmIn,
    const XReg regBase, const IntType offset, bool maskFlag)
{
    auto vmm = vmmIn;
    auto regAddr = GetCompAddrReg(regBase, offset);
    switch (typeIn) {
        case Element::TypeT::F32:
        case Element::TypeT::S32: {
            if (maskFlag) {
                Ld1w(vmm.s, kTailMask / PredType::T_Z, Ptr(regAddr));
            } else {
                // Because regAddr maybe disaligned,
                // `Ldr(vmm, Ptr(regAddr))` can't be used.
                Ld1w(vmm.s, maskAllOne / PredType::T_Z, Ptr(regAddr));
            }
            break;
        }
        default: {
            throw KuDNN::Service::Unsupported{"unsupported data type"};
        }
    }
    if (typeIn != Element::TypeT::F32) {
        Scvtf(vmmIn.s, maskAllOne, vmmIn.s);
    }
}

void KdnnJitX8S8S32ConvFwdKernel::StoreOutput(const IntType& urW, bool bLastOcBlockFlag)
{
    IntType nbOcBlock = jcp.nbOcBlocking;
    IntType ocBlock = jcp.ocBlock;
    Ldr(regBias, Ptr(regParam1, static_cast<int32_t>(offsetof(KdnnConvJitCallS, bias))));
    Ldr(regPtrScales, Ptr(regParam1, static_cast<int32_t>(offsetof(KdnnConvJitCallS, scales))));
    if (!jcp.signedInput) {
        Ldr(regCompensation, Ptr(regParam1, static_cast<int32_t>(offsetof(KdnnConvJitCallS, compensation))));
    }
    for (IntType k = 0; k < nbOcBlock; k++) {
        const bool maskFlag = bLastOcBlockFlag && k == nbOcBlock - 1 && maskGFlag;
        IntType scaleOffset = jcp.isOcScale * (sizeof(float) * k * ocBlock);
        ARMCG::PReg mask = maskFlag ? kTailMask : maskAllOne;
        if (jcp.withBias) {
            IntType biasOffset = jcp.typeSizeBia * k * ocBlock;
            Cvt2Ps(jcp.biaDt, vmmBias, regBias, biasOffset, maskFlag);
        }
        if (!jcp.signedInput) {
            IntType compOffset = sizeof(int32_t) * k * ocBlock;
            Cvt2Ps(Element::TypeT::S32, vmmComp, regCompensation, compOffset, maskFlag);
        }
        /* optimization under specific conditions: preload scaleOffset data */
        if (jcp.signedInput) {
            auto regAddr = GetCompAddrReg(regPtrScales, scaleOffset);
            Ld1w(vmmPreLoad.s, mask, Ptr(regAddr));
        }
        /* Add to accum: compensation, bias and permute */
        for (IntType j = 0; j < urW; j++) {
            auto vmm = VmmOut(j, k);
            Scvtf(vmm.s, maskAllOne, vmm.s);
            if (!jcp.signedInput) {
                Fsub(vmm.s, vmm.s, vmmComp.s);
            }
            if (jcp.withBias) {
                Fadd(vmm.s, vmm.s, vmmBias.s);
            }
            if (jcp.signedInput) {
                /* optimization under specific conditions: optimize using preloaded scaleOffset data */
                Fmul(vmm.s, vmm.s, vmmPreLoad.s);
            } else {
                auto regAddr = GetCompAddrReg(regPtrScales, scaleOffset);
                St1w(vmmTmp.s, maskAllOne, Ptr(regStack, -1, ExtMod::MUL_VL));
                Ld1w(vmmTmp.s, mask, Ptr(regAddr));
                Fmul(vmm.s, vmm.s, vmmTmp.s);
                Ld1w(vmmTmp.s, maskAllOne, Ptr(regStack, -1, ExtMod::MUL_VL));
            }
        }
    }

    // Properly saturate the accumulators for integer datatypes
    if (Service::OneOf(jcp.dstDt, Element::TypeT::U8, Element::TypeT::S8, Element::TypeT::S32)) {
        if (jcp.dstDt == Element::TypeT::U8) {
            Eor(vmmZero.d, vmmZero.d, vmmZero.d);
        }
        float saturationUbound = MaxValue<float>(jcp.dstDt);
        MovImm(auxRegSaturation, ARMCG::Util::Float2Int(saturationUbound));
        Dup(vmmSaturation.s, WReg(auxRegSaturation.GetIdx()));

        for (IntType k = 0; k < nbOcBlock; k++) {
            for (IntType j = 0; j < urW; j++) {
                auto vmm = VmmOut(j, k);
                if (jcp.dstDt == Element::TypeT::U8) {
                    Fmaxnm(vmm.s, maskAllOne, vmmZero.s);
                    Fmax(vmm.s, maskAllOne, vmmZero.s);
                }
                Fminnm(vmm.s, maskAllOne, vmmSaturation.s);
                Fmin(vmm.s, maskAllOne, vmmSaturation.s);
                Frintn(vmm.s, maskAllOne, vmm.s);
                Fcvtzs(vmm.s, maskAllOne, vmm.s);
            }
        }
    }

    /* write out register to output_addr */
    for (IntType k = 0; k < nbOcBlock; k++) {
        const bool maskFlag = bLastOcBlockFlag && k == nbOcBlock - 1 && maskGFlag;
        for (IntType j = 0; j < urW; j++) {
            IntType auxOutputOffset = jcp.typeSizeOut
                    * (k * ocBlock + j * jcp.ocWithoutPadding * jcp.ngroups);
            auto base = regOut;
            auto re = GetOffset(auxOutputOffset);

            AddImm(regTmp0Adr, base, re, regTmp0Imm);
            auto vmm = VmmOut(j, k);
            auto _mask = maskFlag ? kTailMask : maskAllOne;
            switch (jcp.dstDt) {
                case Element::TypeT::F32:
                case Element::TypeT::S32: {
                    St1w(vmm.s, _mask, Ptr(regTmp0Adr));
                    break;
                }
                case Element::TypeT::S8: {
                    Smin(vmm.s, S8_MAX_VAL);
                    Smax(vmm.s, S8_MIN_VAL);
                    St1b(vmm.s, _mask, Ptr(regTmp0Adr));
                    break;
                }
                case Element::TypeT::U8: {
                    Umin(vmm.s, U8_MAX_VAL);
                    St1b(vmm.s, _mask, Ptr(regTmp0Adr));
                    break;
                }
                default: throw KuDNN::Service::LogicError{"unknown dstDt"};
            }
        }
    }
}

void KdnnJitX8S8S32ConvFwdKernel::ComputeKer(IntType urW, IntType padL, IntType padR,
    IcBlockT lastIcBlockFlag, bool hPadded)
{
    IntType kw = jcp.kw;
    IntType strideW = jcp.strideW;
    IntType icBlock = jcp.icBlock;
    IntType ocBlock = jcp.ocBlock;
    IntType chBlockAll = jcp.chBlock * icBlock * ocBlock;
    IntType nbOcBlock = jcp.nbOcBlocking;
    auto inputOffset = [this, strideW, padL](IntType oi, IntType ic, IntType ki) {
        return jcp.typeSizeIn
                * ((ki * (jcp.dilateW + 1) + oi * strideW - padL)
                * jcp.icWithoutPadding * jcp.ngroups + IC_LEN4 * ic);
    };
    auto kernelOffset = [this, chBlockAll, ocBlock](IntType ii, IntType ic, IntType ki) {
        return jcp.typeSizeIn * ((ii * jcp.nbIc * jcp.kd * jcp.kh * jcp.kw + ki) *
            chBlockAll + IC_LEN4 * ic * ocBlock);
    };
    auto compute = [this](ZReg vregAcc, ZReg vregWei, ZReg vregSrc) {
        Sdot(ZRegS(vregAcc.GetIdx()), ZRegB(vregSrc.GetIdx()), ZRegB(vregWei.GetIdx()));
    };

    for (IntType ki = 0; ki < kw; ki++) {
        IntType jjStart = std::max<IntType>(Math::CeilDiv(padL - ki * (jcp.dilateW + 1), jcp.strideW), 0);
        IntType jjEnd = urW - std::max<IntType>(Math::CeilDiv(padR - (jcp.kw - 1 - ki) *
            (jcp.dilateW + 1), jcp.strideW), 0);
        IntType icTailSize = jcp.icWithoutPadding % IC_LEN4;
        IntType start = (!jcp.signedInput) ? 0 : jjStart;
        IntType end = (!jcp.signedInput) ? urW : jjEnd;
        /* Skip the last loads of input if (ic%16)/4 < icBlock/4 */
        IntType icb = (lastIcBlockFlag != IcBlockT::NO_LAST_BLOCK)
                ? Math::CeilDiv((jcp.icWithoutPadding % icBlock), IC_LEN4)
                : icBlock / IC_LEN4;
        for (IntType ic = 0; ic < icb; ic++) {
            if (hPadded) {
                /* fill padded area with shifted values */
                auto inp = VmmInp(0, nbOcBlock);
                Eor(inp.d, inp.d, inp.d);
                Sub(inp.b, inp.b, vmmShift.b);
            } else {
                for (IntType jj = start; jj < end; jj++) {
                    IntType auxInputOffset = inputOffset(jj, ic, ki);
                    if (jj >= jjStart && jj < jjEnd) {
                        if ((lastIcBlockFlag == IcBlockT::LAST_SP_BLOCK) &&
                            (icTailSize != 0) && (ic == icb - 1)) {
#ifdef KUDNNL_920F
                            auto xmmTmp = VReg16B(VmmInp(jj, nbOcBlock).GetIdx());
#else
                            auto xmmTmp = VReg8B(VmmInp(jj, nbOcBlock).GetIdx());
#endif
                            for (IntType r = 0; r < icTailSize; ++r) {
                                AddImm(regTmp0Adr, auxRegInp, (auxInputOffset + r), regTmp0Imm);
                                Ldrb(WReg(regTmp1Imm.GetIdx()), AdrUimm(Ptr(regTmp0Adr)));
#ifdef KUDNNL_920F
                                Ins(VReg16B(xmmTmp.GetIdx())[r], WReg(regTmp1Imm.GetIdx()));
#else
                                Ins(VReg8B(xmmTmp.GetIdx())[r], WReg(regTmp1Imm.GetIdx()));
#endif
                            }
                            Dup(VmmInp(jj, nbOcBlock).s, ZRegS(xmmTmp.GetIdx())[0]);
                        } else {
                            auto base = auxRegInp;
                            auto re = GetOffset(auxInputOffset);
                            if ((-0x40 <= re) && (re < 0x40) && ((re % IC_LEN4) == 0)) {
                                Ld1rw(VmmInp(jj, nbOcBlock).s, maskAllOne, Ptr(base, static_cast<int32_t>(re)));
                            } else {
                                AddImm(regTmp0Adr, base, re, regTmp0Imm);
                                Ld1rw(VmmInp(jj, nbOcBlock).s, maskAllOne, Ptr(regTmp0Adr));
                            }
                        }
                        if (!jcp.signedInput) {
                            Sub(VmmInp(jj, nbOcBlock).b, VmmInp(jj, nbOcBlock).b, vmmShift.b);
                        }
                    } else {
                        /* fill padded area with shifted values */
                        if (!jcp.signedInput) {
                            auto inp = VmmInp(jj, nbOcBlock);
                            Eor(inp.d, inp.d, inp.d);
                            Sub(inp.b, inp.b, vmmShift.b);
                        }
                    }
                }
            }
            for (IntType ii = 0; ii < nbOcBlock; ii++) {
                if (!jcp.signedInput) {
                    IntType auxKernelOffset = kernelOffset(ii, ic, ki);
                    auto regAddr = GetCompAddrReg(auxRegKer, auxKernelOffset);
                    Ld1w(vmmWei.s, maskAllOne, Ptr(regAddr));
                    for (IntType jj = start; jj < end; jj++) {
                        auto inp = (hPadded == true) ? VmmInp(0, nbOcBlock) : VmmInp(jj, nbOcBlock);
                        compute(VmmOut(jj, ii), vmmWei, inp);
                    }
                } else {
                    if (ii == 0) {
                        IntType auxKernelOffset = kernelOffset(ii, ic, ki);
                        auto regAddr = GetCompAddrReg(auxRegKer, auxKernelOffset);
                        Ld1w(vmmWei.s, maskAllOne, Ptr(regAddr));
                    }
                    if ((ii + 1) < nbOcBlock) {
                        IntType auxKernelOffset = kernelOffset((ii + 1), ic, ki);
                        auto vmmWei2 = ((ii % IC_LEN2) == 0) ? vmmComp : vmmWei;
                        auto regAddr = GetCompAddrReg(auxRegKer, auxKernelOffset);
                        Ld1w(vmmWei2.s, maskAllOne, Ptr(regAddr));
                    }
                    for (IntType jj = start; jj < end; jj++) {
                        auto vmmWei2 = ((ii % IC_LEN2) == 0) ? vmmWei : vmmComp;
                        auto inp = (hPadded == true) ? VmmInp(0, nbOcBlock) : VmmInp(jj, nbOcBlock);
                        compute(VmmOut(jj, ii), vmmWei2, inp);
                    }
                }
            }
        }
    }
}

void KdnnJitX8S8S32ConvFwdKernel::KhLoop(IntType urW, IntType padL, IntType padR,
    IcBlockT lastIcBlockFlag)
{
    Label kdLabel;
    Label khLabel;
    Label skipKdLoop;
    Label skipKhLoop;
    Label fOverflowLabel;
    Label noFoverflowLabel;
    Label dHFOverflowLabel;
    Label tOverflowLabel;
    Label noTOverflowLabel;
    Label bOverflowLabel;
    Label noBOverflowLabel;
    Label backOverflowLabel;
    Label noBackOverflowLabel;
    Label dHBackOverflowLabel;

    IntType chBlockAll = jcp.chBlock * jcp.icBlock * jcp.ocBlock;
    IntType shiftKernelPtr = jcp.typeSizeIn * jcp.kw * chBlockAll;
    IntType shiftInputPtr = jcp.typeSizeIn * jcp.iw * jcp.icWithoutPadding * jcp.ngroups;
    if (jcp.ndims == KERNEL_DIMS_5) {
        Mov(auxRegKerD, regKer);
        Mov(auxRegInpD, regInp);
        if (!jcp.signedInput) {
            Ldr(regKi, Ptr(regParam1, static_cast<int32_t>(offsetof(KdnnConvJitCallS, fOverflow))));
            Cmp(regKi, 0);
            B(Cond::EQ, noFoverflowLabel);
            L(fOverflowLabel);
            {
                Mov(auxRegKer, auxRegKerD);
                MovImm(regKj, jcp.kh);
                L(dHFOverflowLabel);
                {
                    ComputeKer(urW, padL, padR, lastIcBlockFlag, true);
                    AddsImm(auxRegKer, auxRegKer, shiftKernelPtr, regTmp0Imm);
                    Subs(regKj, regKj, 1);
                    B(Cond::NE, dHFOverflowLabel);
                }
                AddImm(auxRegKerD, auxRegKerD, shiftKernelPtr * jcp.kh, regTmp0Imm);
                Subs(regKi, regKi, 1);
                B(Cond::NE, fOverflowLabel);
            }
            L(noFoverflowLabel);
        }

        Ldr(regKi, Ptr(regParam1, static_cast<int32_t>(offsetof(KdnnConvJitCallS, kdPadding))));
        if ((!jcp.signedInput) || (jcp.dilateD >= jcp.id) || (jcp.signedInput && (jcp.kd - 1) *
            (jcp.dilateD + 1) < std::max<IntType>(jcp.fPad, jcp.backPad))) {
            Cmp(regKi, 0);
            B(Cond::EQ, skipKdLoop);
        }
        L(kdLabel);
        Mov(auxRegInp, auxRegInpD);
        Mov(auxRegKer, auxRegKerD);
    } else {
        Mov(auxRegInp, regInp);
        Mov(auxRegKer, regKer);
    }

    if (!jcp.signedInput && jcp.ndims > KERNEL_DIMS_3) {
        Ldr(regOverflow, Ptr(regParam1, static_cast<int32_t>(offsetof(KdnnConvJitCallS, tOverflow))));
        Cmp(regOverflow, 0);
        B(Cond::EQ, noTOverflowLabel);
        L(tOverflowLabel);
        {
            ComputeKer(urW, padL, padR, lastIcBlockFlag, true);
            AddsImm(auxRegKer, auxRegKer, shiftKernelPtr, regTmp0Imm);
            Subs(regOverflow, regOverflow, 1);
            Cmp(regOverflow, 0);
            B(Cond::GT, tOverflowLabel);
        }
        L(noTOverflowLabel);
    }
    Ldr(regKj, Ptr(regParam1, static_cast<int32_t>(offsetof(KdnnConvJitCallS, khPadding))));
    if ((!jcp.signedInput) || (jcp.dilateH >= jcp.ih) || (jcp.signedInput && (jcp.kh - 1) *
        (jcp.dilateH + 1) < std::max<IntType>(jcp.tPad, jcp.bPad))) {
        Cmp(regKj, 0);
        B(Cond::EQ, skipKhLoop);
    }
    L(khLabel);
    {
        ComputeKer(urW, padL, padR, lastIcBlockFlag, false);
        AddsImm(auxRegKer, auxRegKer, shiftKernelPtr, regTmp0Imm);
        AddsImm(auxRegInp, auxRegInp, shiftInputPtr * (jcp.dilateH + 1), regTmp0Imm);
        Subs(regKj, regKj, 1);
        Cmp(regKj, 0);
        B(Cond::GT, khLabel);
    }
    L(skipKhLoop);
    if (!jcp.signedInput && jcp.ndims > KERNEL_DIMS_3) {
        Ldr(regOverflow, Ptr(regParam1, static_cast<int32_t>(offsetof(KdnnConvJitCallS, bOverflow))));
        Cmp(regOverflow, 0);
        B(Cond::EQ, noBOverflowLabel);
        L(bOverflowLabel);
        {
            ComputeKer(urW, padL, padR, lastIcBlockFlag, true);
            AddsImm(auxRegKer, auxRegKer, shiftKernelPtr, regTmp0Imm);
            Subs(regOverflow, regOverflow, 1);
            Cmp(regOverflow, 0);
            B(Cond::GT, bOverflowLabel);
        }
        L(noBOverflowLabel);
    }

    if (jcp.ndims == KERNEL_DIMS_5) {
        AddsImm(auxRegInpD, auxRegInpD, shiftInputPtr * jcp.ih * (jcp.dilateD + 1), regTmp0Imm);
        AddsImm(auxRegKerD, auxRegKerD, shiftKernelPtr * jcp.kh, regTmp0Imm);
        Subs(regKi, regKi, 1);
        B(Cond::NE, kdLabel);

        L(skipKdLoop);
        if (!jcp.signedInput) {
            Ldr(regKi, Ptr(regParam1, static_cast<int32_t>(offsetof(KdnnConvJitCallS, backOverflow))));
            Cmp(regKi, 0);
            B(Cond::EQ, noBackOverflowLabel);
            L(backOverflowLabel);
            {
                Mov(auxRegKer, auxRegKerD);
                Mov(regKj, jcp.kh);
                L(dHBackOverflowLabel);
                {
                    ComputeKer(urW, padL, padR, lastIcBlockFlag, true);
                    AddsImm(auxRegKer, auxRegKer, shiftKernelPtr, regTmp0Imm);
                    Subs(regKj, regKj, 1);
                    B(Cond::NE, dHBackOverflowLabel);
                }
                AddsImm(auxRegKerD, auxRegKerD, shiftKernelPtr * jcp.kh, regTmp0Imm);
                Subs(regKi, regKi, 1);
                B(Cond::NE, backOverflowLabel);
            }
            L(noBackOverflowLabel);
        }
    }
}

void KdnnJitX8S8S32ConvFwdKernel::IcbLoop(IntType urW, IntType padL,
    IntType padR, bool isLastSpBlock)
{
    PrepareOutput(urW);
    Label icbLabel;
    MovImm(regIcb, jcp.nbIc);
    L(icbLabel);
    if (jcp.ngroups % jcp.chBlock != 0 || jcp.icWithoutPadding != jcp.ic) {
        Label commonKer;
        Label endKer;
        Cmp(regIcb, 1);
        B(Cond::NE, commonKer);
        KhLoop(urW, padL, padR, isLastSpBlock ? IcBlockT::LAST_SP_BLOCK : IcBlockT::LAST_IC_BLOCK);
        B(endKer);
        L(commonKer);
        KhLoop(urW, padL, padR, IcBlockT::NO_LAST_BLOCK);
        L(endKer);
    } else {
        KhLoop(urW, padL, padR, IcBlockT::NO_LAST_BLOCK);
    }
    // End of IC Loop
    IntType inpStep = jcp.icBlock;
    IntType kerStep = jcp.kd * jcp.kh * jcp.kw * jcp.ocBlock * jcp.icBlock;
    AddsImm(regInp, regInp, jcp.typeSizeIn * inpStep, regTmp0Imm);
    AddsImm(regKer, regKer, jcp.typeSizeIn * kerStep, regTmp0Imm);

    Subs(regIcb, regIcb, 1);
    Cmp(regIcb, 0);
    B(Cond::GT, icbLabel);

    SubsImm(regInp, regInp, jcp.typeSizeIn * inpStep * jcp.nbIc,
            regTmp0Imm);
    SubsImm(regKer, regKer, jcp.typeSizeIn * kerStep * jcp.nbIc,
            regTmp0Imm);
    if (jcp.ngroups % jcp.chBlock != 0 || jcp.ocWithoutPadding != jcp.oc) {
        Label commonStore;
        Label endStore;
        Cmp(regOcBlocks, jcp.nbOc - jcp.nbOcBlocking);
        B(Cond::NE, commonStore);
        StoreOutput(urW, true); // last oc block
        B(endStore);
        L(commonStore);
        StoreOutput(urW, false);
        L(endStore);
    } else {
        StoreOutput(urW, false);
    }
}

void KdnnJitX8S8S32ConvFwdKernel::VmmMaskAllOne()
{
    maskGFlag = false;
#ifdef KUDNNL_920F
    if (sveLen_ == VC_LEN_64) {
        maskGFlag = true;
        Ptrue(maskAllOne.b);
    } else if (sveLen_ == VC_LEN_32) {
        Ptrue(maskAllOne.b, Pattern::VL32);
    } else if (sveLen_ == VC_LEN_16) {
        Ptrue(maskAllOne.b, Pattern::VL16);
    } else {
        throw KuDNN::Service::LogicError{"unreachable"};
    }
#else
    if (sveLen_ == VC_LEN_32) {
        maskGFlag = true;
        Ptrue(maskAllOne.b);
    } else if (sveLen_ == VC_LEN_16) {
        Ptrue(maskAllOne.b, Pattern::VL16);
    } else {
        throw KuDNN::Service::LogicError{"unreachable"};
    }
#endif
}

ARMCG::ZReg KdnnJitX8S8S32ConvFwdKernel::VmmOut(const IntType& iUr, const IntType& iOc)
{
    IntType nbXBlocking = jcp.nbOcBlocking;
    IntType idx = iUr * nbXBlocking + iOc;
    return ARMCG::ZReg(idx);
}

ARMCG::ZReg KdnnJitX8S8S32ConvFwdKernel::VmmInp(const IntType& iIC, const IntType& nbXBlocking)
{
    IntType idx = iIC + nbXBlocking * jcp.urW;
    return ARMCG::ZReg(idx);
}

IntType KdnnJitX8S8S32ConvFwdKernel::GetOffset(IntType rawOfft)
{
    auto offt = rawOfft;
    IntType scale = 0;
    const IntType EVEXMax8bOfft = 0x200;
    if (EVEXMax8bOfft <= offt && offt < REG_OFFSET_3 * EVEXMax8bOfft) {
        offt = offt - REG_OFFSET_2 * EVEXMax8bOfft;
        scale = 1;
    } else if (REG_OFFSET_3 * EVEXMax8bOfft <= offt && offt < REG_OFFSET_5 * EVEXMax8bOfft) {
        offt = offt - REG_OFFSET_4 * EVEXMax8bOfft;
        scale = REG_OFFSET_2;
    }
    auto re = offt;
    if (scale) {
        re = re + (REG_OFFSET_2 * EVEXMax8bOfft) * scale;
    }
    return re;
}

ARMCG::XReg KdnnJitX8S8S32ConvFwdKernel::GetCompAddrReg(ARMCG::XReg base, IntType offset)
{
    auto offt = GetOffset(offset);
    if (offt == 0) {
        return base;
    }
    auto regTmpAdr = regTmp0Adr;
    auto regTmpImm = regTmp0Imm;
    AddImm(regTmpAdr, base, offt, regTmpImm);
    return regTmpAdr;
}

void KdnnJitX8S8S32ConvFwdKernel::Generate()
{
    IntType inIcShift = jcp.icWithoutPadding * jcp.ngroups;
    IntType inpShiftPad = jcp.typeSizeIn * (jcp.urW * jcp.strideW - jcp.lPad) * inIcShift;
    IntType inpShift = jcp.typeSizeIn * (jcp.urW * jcp.strideW * inIcShift);
    IntType outShift = jcp.typeSizeOut * (jcp.urW * jcp.ocWithoutPadding * jcp.ngroups);
    Preamble();
    VmmMaskAllOne();
    Ldr(regInp, Ptr(regParam1, static_cast<int32_t>(offsetof(KdnnConvJitCallS, src))));
    Ldr(regOut, Ptr(regParam1, static_cast<int32_t>(offsetof(KdnnConvJitCallS, dst))));
    Ldr(regKer, Ptr(regParam1, static_cast<int32_t>(offsetof(KdnnConvJitCallS, filt))));

    if (jcp.ngroups % jcp.chBlock != 0 || jcp.ocWithoutPadding != jcp.oc) {
        int tailSize = jcp.ocWithoutPadding % jcp.ocBlock;
        int mask = (1 << tailSize) - 1;
        Ldr(regOcBlocks, Ptr(regParam1, static_cast<int32_t>(offsetof(KdnnConvJitCallS, ocBlocks))));
        auto regwTmp = regOi;
        Mov(regwTmp, mask);
        auto vmmTmp1 = ZReg(REG_INDEX31);
        auto vmmTmp2 = ZReg(REG_INDEX30);
        Index(vmmTmp1.s, 0, 1);
        Mov(vmmTmp2.s, 1);
        Lsl(vmmTmp2.s, maskAllOne, vmmTmp1.s);
        Dup(vmmTmp1.s, WReg(regwTmp.GetIdx()));
        And(vmmTmp1.d, vmmTmp1.d, vmmTmp2.d);
        Cmpne(kTailMask.s, maskAllOne, vmmTmp1.s, 0);
    }

    IntType rPad = jcp.rPad > 0 ? jcp.rPad : 0;
    IntType nOi = jcp.ow / jcp.urW;
    IntType rPad1 = CalculateEndPadding(jcp.lPad, jcp.urW * nOi, jcp.iw,
        jcp.strideW, CalculateExtendedFilterSize(jcp.kw, jcp.dilateW));
    if (rPad1 > 0 || jcp.urWTail == 0) {
        nOi--;
    }
    Eor(regOi, regOi, regOi);
    if (jcp.ow == jcp.urW) {
        IcbLoop(jcp.urW, jcp.lPad, rPad, true);
    } else {
        if (nOi == 0) {
            IcbLoop(jcp.urW, jcp.lPad, rPad1, jcp.urWTail == 0);
            AddsImm(regInp, regInp, inpShiftPad, regTmp0Imm);
            AddsImm(regOut, regOut, outShift, regTmp0Imm);
            if (jcp.urWTail != 0) {
                IcbLoop(jcp.urWTail, 0, rPad, true);
            }
        } else {
            if (jcp.lPad > 0) {
                IcbLoop(jcp.urW, jcp.lPad, 0, false);
                AddsImm(regInp, regInp, inpShiftPad, regTmp0Imm);
                AddsImm(regOut, regOut, outShift, regTmp0Imm);
                Adds(regOi, regOi, 1);
            }
            if ((jcp.lPad <= 0 && nOi > 0) || (jcp.lPad > 0 && nOi > 1)) {
                Label owLoopLabel;
                L(owLoopLabel);
                {
                    IcbLoop(jcp.urW, 0, 0, false);
                    AddsImm(regInp, regInp, inpShift, regTmp0Imm);
                    AddsImm(regOut, regOut, outShift, regTmp0Imm);

                    Adds(regOi, regOi, 1);
                    MovImm(regTmp0Imm, nOi);
                    Cmp(regOi, regTmp0Imm);
                    B(Cond::LT, owLoopLabel);
                }
            }
            if (rPad1 > 0 || jcp.urWTail == 0) {
                IcbLoop(jcp.urW, 0, rPad1, jcp.urWTail == 0);
                AddsImm(regInp, regInp, inpShift, regTmp0Imm);
                AddsImm(regOut, regOut, outShift, regTmp0Imm);
            }
            if (jcp.urWTail != 0) {
                IcbLoop(jcp.urWTail, 0, rPad, true);
            }
        }
    }
    Postamble();
}
} // namespace KuDNN