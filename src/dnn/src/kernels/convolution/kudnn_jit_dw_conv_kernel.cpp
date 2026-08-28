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

#include "kudnn_jit_dw_conv_kernel.hpp"

namespace KuDNN {
using namespace ARMCG;
template<CPUISAT isa>
void KdnnJitDwConvFwdKernel<isa>::LoadSrc(IntType urChBlock, IntType urW)
{
    const auto isDstLayoutNXC = IsDstLayoutNXC();
    const auto chBlock = jcp.chBlock;
    const auto ocbStride = isDstLayoutNXC ? chBlock : jcp.oh * jcp.ow * chBlock;
    const auto owStride = isDstLayoutNXC ? jcp.ngroups : chBlock;

    for (IntType ch = 0; ch < urChBlock; ch++) {
        for (IntType ow = 0; ow < urW; ow++) {
            ZRegS zregsAcc = GetAccRegS(ch * urW + ow);

            SizeType bOff = ch * chBlock;
            if (this->jcp.withBias) {
                AddImm(regTmpAdr, regBia, bOff * jcp.typeSizeBia, regTmpImm);
                Ld1w(zregsAcc, pAllOne, Ptr(regTmpAdr));
            } else {
                Fmov(zregsAcc);
            }

            SizeType oOff = ch * ocbStride + ow * owStride;
            if (this->jcp.withSum) {
                AddImm(regTmpAdr, regOut, oOff * jcp.typeSizeOut, regTmpImm);
                Ld1w(ZRegS(0), pAllOne, Ptr(regTmpAdr));
                Fadd(zregsAcc, zregsAcc, ZRegS(0));
            }
        }
    }
}

template<CPUISAT isa>
void KdnnJitDwConvFwdKernel<isa>::ApplyFilterUnrolled(IntType urChBlock, IntType urW, IntType padL, IntType padR)
{
    IntType chBlock = jcp.chBlock;
    IntType dilateH = jcp.dilateH + 1;
    IntType dilateW = jcp.dilateW + 1;
    IntType strideW = jcp.strideW;
    const auto isSrcLayoutNXC = IsSrcLayoutNXC();
    const auto iwStride = isSrcLayoutNXC ? jcp.ngroups : chBlock;
    const auto ihStride = jcp.iw * iwStride;
    const auto icbStride = isSrcLayoutNXC ? chBlock : jcp.ih * jcp.iw * chBlock;

    Cmp(regKH, 0);
    Label iterExitLabel;
    B(Cond::EQ, iterExitLabel);

    Mov(regIterKH, regKH);
    Label khLabel;
    L(khLabel);
    for (IntType ch = 0; ch < urChBlock; ch++) {
        for (IntType kw = 0; kw < jcp.kw; kw++) {
            SizeType kerOff = ch * jcp.kh * jcp.kw * chBlock + kw * chBlock;

            ZRegS zregsKer = GetKerRegS(0);
            AddImm(regTmpAdr, regKerAux, kerOff * jcp.typeSizeIn, regTmpImm);
            Ld1w(zregsKer, pAllOne, Ptr(regTmpAdr));

            IntType owStart = GetOWStart(kw, padL);
            IntType owEnd = GetOWEnd(urW, kw, padR);
            for (IntType ow = owStart; ow < owEnd; ow++) {
                SizeType inpOff = ch * icbStride + (ow * strideW - padL) * iwStride + kw * dilateW * iwStride;
                ZRegS zregsSrc = GetSrcRegS(0);
                AddImm(regTmpAdr, regInpAux, inpOff * jcp.typeSizeIn, regTmpImm);
                Ld1w(zregsSrc, pAllOne, Ptr(regTmpAdr));

                ZRegS zregsAcc = GetAccRegS(ch * urW + ow);
                Fmla(zregsAcc, pAllOne, zregsSrc, zregsKer);
            }
        }
    }

    AddImm(regKerAux, regKerAux, jcp.kw * chBlock * jcp.typeSizeIn, regTmpImm);
    AddImm(regInpAux, regInpAux, ihStride * dilateH * jcp.typeSizeIn, regTmpImm);

    Sub(regIterKH, regIterKH, 1);
    Cmp(regIterKH, 0);
    B(Cond::GT, khLabel);

    L(iterExitLabel);
}

template<CPUISAT isa>
void KdnnJitDwConvFwdKernel<isa>::StoreDst(IntType urChBlock, IntType urW)
{
    const auto isDstLayoutNXC = IsDstLayoutNXC();
    const auto chBlock = jcp.chBlock;
    const auto ocbStride = isDstLayoutNXC ? chBlock : jcp.oh * jcp.ow * chBlock;
    const auto owStride = isDstLayoutNXC ? jcp.ngroups : chBlock;

    for (IntType ch = 0; ch < urChBlock; ch++) {
        for (IntType ow = 0; ow < urW; ow++) {
            const IntType oOff = ch * ocbStride + ow * owStride;
            ZRegS zregsDst = GetAccRegS(ch * urW + ow);
            AddImm(regTmpAdr, regOut, oOff * jcp.typeSizeOut, regTmpImm);
            St1w(zregsDst, pAllOne, Ptr(regTmpAdr));
        }
    }
}

template<CPUISAT isa>
void KdnnJitDwConvFwdKernel<isa>::ComputeLoop(IntType urW, IntType urChBlock, IntType padL, IntType padR)
{
    const bool chLoop = urChBlock >= jcp.nbChBlocking;
    const SizeType inpChStride =
        static_cast<SizeType>(jcp.nbChBlocking) * jcp.chBlock * jcp.typeSizeIn;
    const SizeType weiChStride =
        static_cast<SizeType>(jcp.nbChBlocking) * jcp.kh * jcp.kw * jcp.chBlock * jcp.typeSizeIn;
    const SizeType outChStride =
        static_cast<SizeType>(jcp.nbChBlocking) * jcp.chBlock * jcp.typeSizeOut;
    const SizeType biaStride =
        static_cast<SizeType>(jcp.nbChBlocking) * jcp.chBlock * jcp.typeSizeBia;

    auto compute = [this, urW, padL, padR](IntType urCB) {
        Mov(regInpAux, regInp);

        Mov(regKerAux, regKer);
        LoadSrc(urCB, urW);
        ApplyFilterUnrolled(urCB, urW, padL, padR);
        StoreDst(urCB, urW);
    };

    if (chLoop) {
        const IntType chTail = jcp.nbCh % jcp.nbChBlocking;
    
        Mov(regChBlockAux, regChBlock);
        Mov(regInpStack, regInp);
        Mov(regKerStack, regKer);
        Mov(regOutStack, regOut);
        if (jcp.withBias) {
            Mov(regBiaStack, regBia);
        }

        Label chTailLabel;
        if (chTail) {
            Cmp(regChBlockAux, jcp.nbChBlocking);
            B(Cond::LT, chTailLabel);
        }

        Label chLoopLabel;
        L(chLoopLabel);
        compute(jcp.nbChBlocking);
        AddImm(regInp, regInp, inpChStride, regTmpImm);
        AddImm(regKer, regKer, weiChStride, regTmpImm);
        AddImm(regOut, regOut, outChStride, regTmpImm);
        if (jcp.withBias) {
            AddImm(regBia, regBia, biaStride, regTmpImm);
        }
        SubImm(regChBlockAux, regChBlockAux, jcp.nbChBlocking, regTmpImm);
        Cmp(regChBlockAux, jcp.nbChBlocking);
        B(Cond::GE, chLoopLabel);

        Label skipChTailLabel;
        if (chTail) {
            L(chTailLabel);
            Cmp(regChBlockAux, 0);
            B(Cond::LE, skipChTailLabel);
            compute(chTail);
            L(skipChTailLabel);
        }

        if (jcp.withBias) {
            Mov(regBia, regBiaStack);
        }
        Mov(regInp, regInpStack);
        Mov(regKer, regKerStack);
        Mov(regOut, regOutStack);
    } else {
        compute(urChBlock);
    }
}

template <CPUISAT isa>
void KdnnJitDwConvFwdKernel<isa>::OWLoop(IntType urChBlock)
{
    IntType iw = jcp.iw;
    IntType kw = jcp.kw;
    IntType ow = jcp.ow;
    IntType padL = jcp.lPad;
    IntType urW = jcp.urW;
    IntType urWTail = jcp.urWTail;
    IntType strideW = jcp.strideW;

    const auto isSrcLayoutNXC = IsSrcLayoutNXC();
    const auto datCStride = isSrcLayoutNXC ? jcp.ngroups : jcp.chBlock;
    SizeType inpShift = static_cast<SizeType>(jcp.typeSizeIn) * urW * strideW * datCStride;
    SizeType outShift = static_cast<SizeType>(jcp.typeSizeOut) * urW * datCStride;

    SizeType inpShiftPad = jcp.typeSizeIn * (urW * strideW - padL) * datCStride;
    IntType padR = jcp.rPad > 0 ? jcp.rPad : 0;
    IntType nOI = ow / urW;
    IntType padR1 = CalculateEndPadding(padL, urW * nOI, iw, strideW, CalculateExtendedFilterSize(kw, jcp.dilateW));
    if (padR1 > 0) {
        nOI--;
    }

    Mov(regOI, 0);
    if (ow == urW) {
        ComputeLoop(urW, urChBlock, padL, padR);
    } else {
        if (nOI == 0) {
            ComputeLoop(urW, urChBlock, padL, padR1);
            AddImm(regInp, regInp, inpShiftPad, regTmpImm);
            AddImm(regOut, regOut, outShift, regTmpImm);
            if (urWTail != 0) {
                ComputeLoop(urWTail, urChBlock, 0, padR);
            }
        } else {
            if (padL > 0) {
                ComputeLoop(urW, urChBlock, padL, 0);
                AddImm(regInp, regInp, inpShiftPad, regTmpImm);
                AddImm(regOut, regOut, outShift, regTmpImm);
                Add(regOI, regOI, 1);
            }
            if ((padL <= 0 && nOI > 0) || (padL > 0 && nOI > 1)) {
                Label owLoopLabel;
                L(owLoopLabel);

                ComputeLoop(urW, urChBlock, 0, 0);
                AddImm(regInp, regInp, inpShift, regTmpImm);
                AddImm(regOut, regOut, outShift, regTmpImm);
                Add(regOI, regOI, 1);
                Cmp(regOI, nOI);
                B(Cond::LT, owLoopLabel);
            }
            if (padR1 > 0) {
                ComputeLoop(urW, urChBlock, 0, padR1);
                AddImm(regInp, regInp, inpShift, regTmpImm);
                AddImm(regOut, regOut, outShift, regTmpImm);
            }
            if (urWTail != 0) {
                ComputeLoop(urWTail, urChBlock, 0, padR);
            }
        }
    }
}

template <CPUISAT isa>
void KdnnJitDwConvFwdKernel<isa>::Generate()
{
    this->Preamble();
    Ldr(regInp, Ptr(ABI_PARAM1, static_cast<uint32_t>(offsetof(KdnnConvJitCallS, src))));
    Ldr(regKer, Ptr(ABI_PARAM1, static_cast<uint32_t>(offsetof(KdnnConvJitCallS, filt))));
    Ldr(regOut, Ptr(ABI_PARAM1, static_cast<uint32_t>(offsetof(KdnnConvJitCallS, dst))));
    if (jcp.withBias) {
        Ldr(regBia, Ptr(ABI_PARAM1, static_cast<uint32_t>(offsetof(KdnnConvJitCallS, bias))));
    }
    Ldr(regKH, Ptr(ABI_PARAM1, static_cast<uint32_t>(offsetof(KdnnConvJitCallS, khPadding))));
    Ldr(regChBlock, Ptr(ABI_PARAM1, static_cast<uint32_t>(offsetof(KdnnConvJitCallS, chBlocks))));

    IntType chBlockTail = jcp.nbCh % jcp.nbChBlocking;

    Label chBlockTailLabel;
    Label exitLabel;
    if (IsSrcLayoutNXC()) {
        OWLoop(jcp.nbCh);
    } else {
        Cmp(regChBlock, jcp.nbChBlocking);
        B(Cond::NE, chBlockTail ? chBlockTailLabel : exitLabel);

        OWLoop(jcp.nbChBlocking);

        if (chBlockTail) {
            L(chBlockTailLabel);

            Cmp(regChBlock, chBlockTail);
            B(Cond::NE, exitLabel);

            OWLoop(chBlockTail);
        }

        L(exitLabel);
    }

    this->Postamble();
}

#ifdef KUDNNL_920F
template struct KdnnJitDwConvFwdKernel<CPUISAT::VC_512>;
#else
template struct KdnnJitDwConvFwdKernel<CPUISAT::VC_256>;
#endif

} // namespace KuDNN
