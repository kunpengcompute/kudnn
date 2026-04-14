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

inline int CalculateExtendedFilterSize(int filterSize, int dilation)
{
    return (filterSize - 1) * (dilation + 1) + 1;
}

inline int CalculateEndPadding(int startPadding, int dstSize, int srcSize,
    int spatialStride, int dilatedFilterSize)
{
    return (dstSize - 1) * spatialStride + dilatedFilterSize - (srcSize + startPadding);
}

template<CPUISAT isa>
void KudnnJitDwConvFwdKernel<isa>::LoadSrc(int urChBlock, int urW)
{
    const auto isDstLayoutNXC = IsDstLayoutNXC();
    const auto chBlock = jcp.chBlock;
    const auto ocbStride = isDstLayoutNXC ? chBlock : jcp.oh * jcp.ow * chBlock;
    const auto owStride = isDstLayoutNXC ? jcp.ngroups : chBlock;

    for (int ch = 0; ch < urChBlock; ch++) {
        for (int ow = 0; ow < urW; ow++) {
            ZRegS zregsAcc = GetAccRegS(ch * urW + ow);

            int bOff = ch * chBlock;
            if (this->jcp.withBias) {
                AddImm(regTmpAdr, regBia, bOff * jcp.typeSizeBia, regTmpImm);
                Ld1w(zregsAcc, pAllOne, Ptr(regTmpAdr));
            } else {
                Fmov(zregsAcc);
            }

            int oOff = ch * ocbStride + ow * owStride;
            if (this->jcp.withSum) {
                AddImm(regTmpAdr, regOut, oOff * jcp.typeSizeOut, regTmpImm);
                Ld1w(ZRegS(0), pAllOne, Ptr(regTmpAdr));
                Fadd(zregsAcc, zregsAcc, ZRegS(0));
            }
        }
    }
}

template<CPUISAT isa>
void KudnnJitDwConvFwdKernel<isa>::ApplyFilterUnrolled(int urChBlock, int urW, int padL, int padR)
{
    int chBlock = jcp.chBlock;
    int dilateH = jcp.dilateH + 1;
    int dilateW = jcp.dilateW + 1;
    int strideW = jcp.strideW;
    const auto isSrcLayoutNXC = IsSrcLayoutNXC();
    const auto iwStride = isSrcLayoutNXC ? jcp.ngroups : chBlock;
    const auto ihStride = jcp.iw * iwStride;
    const auto icbStride = isSrcLayoutNXC ? chBlock : (jcp.isFusedConv ? 1 : jcp.ih) * jcp.iw * chBlock;

    Cmp(regKH, 0);
    Label iterExitLabel;
    B(Cond::EQ, iterExitLabel);

    Mov(regIterKH, regKH);
    Label khLabel;
    L(khLabel);
    if (jcp.isFusedConv) {
        Ldr(regInpAux, AdrUimm(Ptr(regInpBufPtrAux)));
        Add(regInpAux, regInpAux, regIWOffset);
    }
    for (int ch = 0; ch < urChBlock; ch++) {
        for (int kw = 0; kw < jcp.kw; kw++) {
            int kerOff = ch * jcp.kh * jcp.kw * chBlock + kw * chBlock;

            ZRegS zregsKer = GetKerRegS(0);
            AddImm(regTmpAdr, regKerAux, kerOff * jcp.typeSizeIn, regTmpImm);
            Ld1w(zregsKer, pAllOne, Ptr(regTmpAdr));

            int owStart = GetOWStart(kw, padL);
            int owEnd = GetOWEnd(urW, kw, padR);
            for (int ow = owStart; ow < owEnd; ow++) {
                int inpOff = ch * icbStride + (ow * strideW - padL) * iwStride + kw * dilateW * iwStride;
                ZRegS zregsSrc = GetSrcRegS(0);
                AddImm(regTmpAdr, regInpAux, inpOff * jcp.typeSizeIn, regTmpImm);
                Ld1w(zregsSrc, pAllOne, Ptr(regTmpAdr));

                ZRegS zregsAcc = GetAccRegS(ch * urW + ow);
                Fmla(zregsAcc, pAllOne, zregsSrc, zregsKer);
            }
        }
    }

    AddImm(regKerAux, regKerAux, jcp.kw * chBlock * jcp.typeSizeIn, regTmpImm);
    if (jcp.isFusedConv) {
        AddImm(regInpBufPtrAux, regInpBufPtrAux, sizeof(void *), regTmpImm);
    } else {
        AddImm(regInpAux, regInpAux, ihStride * dilateH * jcp.typeSizeIn, regTmpImm);
    }

    Sub(regIterKH, regIterKH, 1);
    Cmp(regIterKH, 0);
    B(Cond::GT, khLabel);

    L(iterExitLabel);
}

template<CPUISAT isa>
void KudnnJitDwConvFwdKernel<isa>::StoreDst(int urChBlock, int urW)
{
    const auto isDstLayoutNXC = IsDstLayoutNXC();
    const auto chBlock = jcp.chBlock;
    const auto ocbStride = isDstLayoutNXC ? chBlock : jcp.oh * jcp.ow * chBlock;
    const auto owStride = isDstLayoutNXC ? jcp.ngroups : chBlock;

    for (int ch = 0; ch < urChBlock; ch++) {
        for (int ow = 0; ow < urW; ow++) {
            const int oOff = ch * ocbStride + ow * owStride;
            ZRegS zregsDst = GetAccRegS(ch * urW + ow);
            AddImm(regTmpAdr, regOut, oOff * jcp.typeSizeOut, regTmpImm);
            St1w(zregsDst, pAllOne, Ptr(regTmpAdr));
        }
    }
}

template<CPUISAT isa>
void KudnnJitDwConvFwdKernel<isa>::ComputeLoop(int urW, int urChBlock, int padL, int padR)
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

    auto compute = [&](int urCB) {
        if (jcp.isFusedConv) {
            Mov(regInpBufPtrAux, regInpBufPtr);
        } else {
            Mov(regInpAux, regInp);
        }

        Mov(regKerAux, regKer);
        LoadSrc(urCB, urW);
        ApplyFilterUnrolled(urCB, urW, padL, padR);
        StoreDst(urCB, urW);
    };

    if (chLoop) {
        const int chTail = jcp.nbCh % jcp.nbChBlocking;
    
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
void KudnnJitDwConvFwdKernel<isa>::OWLoop(int urChBlock)
{
    int iw = jcp.iw;
    int kw = jcp.kw;
    int ow = jcp.ow;
    int padL = jcp.lPad;
    int urW = jcp.urW;
    int urWTail = jcp.urWTail;
    int strideW = jcp.strideW;

    const auto isSrcLayoutNXC = IsSrcLayoutNXC();
    const auto datCStride = isSrcLayoutNXC ? jcp.ngroups : jcp.chBlock;
    SizeType inpShift = static_cast<SizeType>(jcp.typeSizeIn) * urW * strideW * datCStride;
    SizeType outShift = static_cast<SizeType>(jcp.typeSizeOut) * urW * datCStride;

    int inpShiftPad = jcp.typeSizeIn * (urW * strideW - padL) * datCStride;
    int padR = jcp.rPad > 0 ? jcp.rPad : 0;
    int nOI = ow / urW;
    int padR1 = CalculateEndPadding(padL, urW * nOI, iw, strideW, CalculateExtendedFilterSize(kw, jcp.dilateW));
    if (padR1 > 0) {
        nOI--;
    }

    assert(jcp.nbOw <= 1);

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
void KudnnJitDwConvFwdKernel<isa>::Generate()
{
    this->Preamble();

    if (jcp.isFusedConv) {
        Ldr(regInpBufPtr, Ptr(ABI_PARAM1, static_cast<uint32_t>(offsetof(KudnnConvJitCallS, src))));
        Mov(regIWOffset, 0);
    } else {
        Ldr(regInp, Ptr(ABI_PARAM1, static_cast<uint32_t>(offsetof(KudnnConvJitCallS, src))));
    }
    Ldr(regKer, Ptr(ABI_PARAM1, static_cast<uint32_t>(offsetof(KudnnConvJitCallS, filt))));
    Ldr(regOut, Ptr(ABI_PARAM1, static_cast<uint32_t>(offsetof(KudnnConvJitCallS, dst))));
    if (jcp.withBias) {
        Ldr(regBia, Ptr(ABI_PARAM1, static_cast<uint32_t>(offsetof(KudnnConvJitCallS, bias))));
    }
    Ldr(regKH, Ptr(ABI_PARAM1, static_cast<uint32_t>(offsetof(KudnnConvJitCallS, khPadding))));
    Ldr(regChBlock, Ptr(ABI_PARAM1, static_cast<uint32_t>(offsetof(KudnnConvJitCallS, chBlocks))));

    int chBlockTail = jcp.nbCh % jcp.nbChBlocking;

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

#ifdef KUDNNL_920Pro
template struct KudnnJitDwConvFwdKernel<CPUISAT::VectorComputation_512>;
#else
template struct KudnnJitDwConvFwdKernel<CPUISAT::VectorComputation_256>;
#endif

} // namespace KuDNN
