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

#include "kudnn_jit_sve_conv_kernel.hpp"
#include "kudnn_conv_const.hpp"
namespace KuDNN {
namespace {
// calculates filter size taking into account dilation
inline int CalculateExtendedFilterSize(int filterSize, int dilation)
{
    return (filterSize - 1) * (dilation + 1) + 1;
}

inline int CalculateEndPadding(int startPadding, int dstDize, int srcSize,
    int spatialStride, int dilatedFilterSize)
{
    return (dstDize - 1) * spatialStride + dilatedFilterSize
            - (srcSize + startPadding);
}
}

using namespace ARMCG;
using namespace Element;

template <CPUISAT isa, TypeT dtp>
void KudnnJitConvFwdKernel<isa, dtp>::PrepareOutput(int urW)
{
    auto zregOut = [this](int iUr, int iOc) {
        if (jcp.kernelKind == KudnnConvKernelKindT::KUDNN_CONT_BCAST) {
            int idx = REG_INDEX29 - (iUr + iOc * jcp.urW);
            return ZRegFactory::GetZRegAcdTp4Fmov(dtp, idx);
        }
        int idx = iUr + iOc * jcp.urW;
        return ZRegFactory::GetZRegAcdTp4Fmov(dtp, idx);
    };

    for (int k = 0; k < jcp.nbOcBlocking; k++) {
        for (int j = 0; j < urW; j++) {
            FmovZReg(dtp, zregOut(j, k));
        }
    }
}

template <CPUISAT isa, TypeT dtp>
void KudnnJitConvFwdKernel<isa, dtp>::StoreOutput(int urW)
{
    Label noUpdateLabel;
    Label storeLabel;
    Label eltwiseLabel;
    auto tTest = [this](const int cond) { return Tst(regChannel, cond); };
    auto zregOut = [this](int iUr, int iOc) {
        int idx = jcp.kernelKind == KudnnConvKernelKindT::KUDNN_CONT_BCAST
                  ? REG_INDEX29 - (iUr + iOc * jcp.urW)
                  : iUr + iOc * jcp.urW;
        return ZRegFactory::GetZRegAcdTp(dtp, idx);
    };

    Ldr(regChannel, Ptr(ABI_PARAM1, static_cast<uint32_t>(
        offsetof(KudnnConvJitCallS, flags))));

    if (jcp.withBias) {
        Ldr(regBias, Ptr(ABI_PARAM1, static_cast<uint32_t>(
            offsetof(KudnnConvJitCallS, bias))));
    }

    if (!jcp.withSum) {
        auto tJmp = [this](const Label &l) { return B(Cond::NE, l); };
        tTest(static_cast<int32_t>(KudnnConvFlag::KUDNN_FLAG_IC_FIRST));
        tJmp(noUpdateLabel);
    }

    int regOfs = jcp.urW * jcp.nbOcBlocking;
    int nRegs = REG_INDEX32 - regOfs;
    if (dtp == TypeT::BF16) {
        nRegs = REG_INDEX32 - regOfs - REG_INDEX2;
        Fmov(ZReg(REG_INDEX31).s, pAllOne, 1.f);
        Bfcvt(ZReg(REG_INDEX30).h, pAllOne, ZReg(REG_INDEX31).s);
    }
    int prevOutOfs = -1;
    for (int k = 0; k < jcp.nbOcBlocking; k++) {
        for (int j = 0; j < urW; j++) {
            int tmpIdx = jcp.kernelKind == KudnnConvKernelKindT::KUDNN_CONT_BCAST
                         ? j
                         : regOfs + ((j + k * urW) % nRegs);
            size_t auxOutputOffset = GetOutputOffset(j, k);
            if (j == 0) {
                AddImm(regOutOfs, regOut, auxOutputOffset, regTmpImm);
                prevOutOfs = static_cast<int>(auxOutputOffset);
                dtp == TypeT::F32 ? Ld1w(ZReg(tmpIdx).s, pAllOne / PredType::T_Z, Ptr(regOutOfs)) :
                    (dtp == TypeT::F16 ? Ld1h(ZReg(tmpIdx).h, pAllOne / PredType::T_Z, Ptr(regOutOfs)) :
                    Ld1h(ZReg(tmpIdx).s, pAllOne / PredType::T_Z, Ptr(regOutOfs)));
            } else if (KudnnLdrImmCheck<size_t, isa>(auxOutputOffset - prevOutOfs)) {
                if (dtp == TypeT::BF16) {
                    AddImm(xDefaultAddr, regOutOfs, (auxOutputOffset - prevOutOfs), X_TMP_0);
                } else if (dtp == TypeT::F16) {
                    AddImm(regOutOfs, regOutOfs,
                    auxOutputOffset - prevOutOfs, regTmpImm);
                    prevOutOfs = auxOutputOffset;
                }
                dtp == TypeT::F32 ? Ldr(ZRegS(tmpIdx), AdrScImm(Ptr(regOutOfs,
                    static_cast<int32_t>(GetVlenOfs(auxOutputOffset - prevOutOfs, isa))))) :
                    (dtp == TypeT::F16 ? Ld1h(ZReg(tmpIdx).h, pAllOne / PredType::T_Z, Ptr(regOutOfs)) :
                        Ld1h(ZReg(tmpIdx).s, pAllOne / PredType::T_Z, Ptr(xDefaultAddr)));
            } else {
                AddImm(regOutOfs, regOutOfs,
                    auxOutputOffset - prevOutOfs, regTmpImm);
                prevOutOfs = static_cast<int>(auxOutputOffset);
                dtp == TypeT::F32 ? Ld1w(ZReg(tmpIdx).s, pAllOne / PredType::T_Z, Ptr(regOutOfs)) : (
                    dtp == TypeT::F16 ? Ld1h(ZReg(tmpIdx).h, pAllOne / PredType::T_Z, Ptr(regOutOfs)) :
                    Ld1h(ZReg(tmpIdx).s, pAllOne / PredType::T_Z, Ptr(regOutOfs)));
            }
        }
        for (int j = 0; j < urW; j++) {
            int tmpIdx = jcp.kernelKind == KudnnConvKernelKindT::KUDNN_CONT_BCAST
                         ? j
                         : regOfs + ((j + k * urW) % nRegs);
            auto pZreg = zregOut(j, k);
            assert(pZreg.get() != nullptr);
            if (dtp == TypeT::F32) {
                ARMCG::ZRegS* pZRegS = static_cast<ARMCG::ZRegS*>(pZreg.get());
                Fadd(*pZRegS, *pZRegS, ZRegS(tmpIdx));
            } else if (dtp == TypeT::F16) {
                ARMCG::ZRegH* pZRegH = static_cast<ARMCG::ZRegH*>(pZreg.get());
                Fadd(*pZRegH, *pZRegH, ZRegH(tmpIdx));
            } else if (dtp == TypeT::BF16) {
                ARMCG::ZRegS* pZRegS = static_cast<ARMCG::ZRegS*>(pZreg.get());
                Bfmlalb(*pZRegS, ZReg(REG_INDEX30).h, ZReg(tmpIdx).h);
            }
        }
    }

    B(eltwiseLabel);

    auto biasLoad = [this](int biasOffset, int idx) {
        int ofs = biasOffset;
        if ((GetVlenOfs(ofs, isa) < LDRMAX) && (GetVlenOfs(ofs, isa) >= (-1 * LDRMAX))
                && ((ofs & 0x3f) == 0)) {
            AddImm(xDefaultAddr, regBias, ofs, X_TMP_0);
            dtp == TypeT::F32 ? Ld1w(ZReg(idx).s, pAllOne / PredType::T_Z, Ptr(xDefaultAddr)) :
                (dtp == TypeT::F16 ? Ld1h(ZReg(idx).h, pAllOne / PredType::T_Z, Ptr(xDefaultAddr)) :
                Ld1h(ZReg(idx).s, pAllOne / PredType::T_Z, Ptr(xDefaultAddr)));
        } else {
            AddImm(regTmpAddr, regBias, ofs, regTmpImm);
            dtp == TypeT::F32 ? Ld1w(ZReg(idx).s, pAllOne / PredType::T_Z, Ptr(regTmpAddr)) :
                (dtp == TypeT::F16 ? Ld1h(ZReg(idx).h, pAllOne / PredType::T_Z, Ptr(regTmpAddr)) :
                Ld1h(ZReg(idx).s, pAllOne / PredType::T_Z, Ptr(regTmpAddr)));
        }
    };

    L(noUpdateLabel);
    if (jcp.withBias) {
        for (int k = 0; k < jcp.nbOcBlocking; k++) {
            int biasOffset = jcp.typeSizeOut * k * jcp.ocBlock;
            int idx = jcp.kernelKind == KudnnConvKernelKindT::KUDNN_CONT_BCAST
                      ? k : regOfs + (k % nRegs);
            biasLoad(biasOffset, idx);
            if (dtp == TypeT::BF16) {
                Fmov(ZReg(REG_INDEX31).s, pAllOne, 1.f);
                Bfcvt(ZReg(REG_INDEX30).h, pAllOne, ZReg(REG_INDEX31).s);
            }
            
            for (int j = 0; j < urW; j++) {
                auto pZreg = zregOut(j, k);
                assert(pZreg.get() != nullptr);
                if (dtp == TypeT::F32) {
                    ARMCG::ZRegS* pZRegS = static_cast<ARMCG::ZRegS*>(pZreg.get());
                    Fadd(*pZRegS, *pZRegS, ZRegS(idx));
                } else if (dtp == TypeT::F16) {
                    ARMCG::ZRegH* pZRegH = static_cast<ARMCG::ZRegH*>(pZreg.get());
                    Fadd(*pZRegH, *pZRegH, ZRegH(idx));
                } else if (dtp == TypeT::BF16) {
                    ARMCG::ZRegS* pZRegS = static_cast<ARMCG::ZRegS*>(pZreg.get());
                    Bfmlalb(*pZRegS, ZReg(REG_INDEX30).h, ZReg(idx).h);
                }
            }
        }
    }

    L(eltwiseLabel);
    auto outStrF32 = [this, zregOut](int j, int k, int auxOutputOffset, int tPrevOutOfs) {
        auto pZreg = zregOut(j, k);
        assert(pZreg.get() != nullptr);
        ARMCG::ZRegS* pZRegS = static_cast<ARMCG::ZRegS*>(pZreg.get());
        int ofs = auxOutputOffset;
        if (KudnnStrImmCheck<int, isa>(ofs)) {
            AddImm(xDefaultAddr, regOut, ofs, X_TMP_0);
            St1w(*pZRegS, pAllOne / PredType::T_Z, Ptr(xDefaultAddr));
        } else if ((tPrevOutOfs != -1)
                && KudnnStrImmCheck<int, isa>(ofs - tPrevOutOfs)) {
            AddImm(xDefaultAddr, regTmpAddr, (ofs - tPrevOutOfs),
                X_TMP_0);
            St1w(*pZRegS, pAllOne / PredType::T_Z, Ptr(xDefaultAddr));
        } else {
            if (tPrevOutOfs == -1) {
                AddImm(regTmpAddr, regOut, ofs, regTmpImm);
            } else {
                AddImm(regTmpAddr, regTmpAddr, ofs - tPrevOutOfs, regTmpImm);
            }
            St1w(*pZRegS, pAllOne / PredType::T_Z, Ptr(regTmpAddr));
            tPrevOutOfs = auxOutputOffset;
        }
        return tPrevOutOfs;
    };

    auto outStrF16 = [this, zregOut](int j, int k, int auxOutputOffset, int tPrevOutOfs) {
        auto pZreg = zregOut(j, k);
        assert(pZreg.get() != nullptr);
        ARMCG::ZRegH* pZRegH = static_cast<ARMCG::ZRegH*>(pZreg.get());
        int ofs = auxOutputOffset;
        if (tPrevOutOfs == -1) {
            AddImm(regTmpAddr, regOut, ofs, regTmpImm);
        } else {
            AddImm(regTmpAddr, regTmpAddr, ofs - tPrevOutOfs, regTmpImm);
        }
        St1h(*pZRegH, pAllOne / PredType::T_Z, Ptr(regTmpAddr));
        tPrevOutOfs = auxOutputOffset;
        return tPrevOutOfs;
    };

    auto outStrBF16 = [this](int auxOutputOffset, int tPrevOutOfs) {
        int ofs = auxOutputOffset;
        if (tPrevOutOfs == -1) {
            AddImm(regTmpAddr, regOut, ofs, regTmpImm);
        } else {
            AddImm(regTmpAddr, regTmpAddr, ofs - tPrevOutOfs, regTmpImm);
        }
        St1h(ZReg(REG_INDEX31).s, pAllOne / PredType::T_Z, Ptr(regTmpAddr));
        tPrevOutOfs = auxOutputOffset;
        return tPrevOutOfs;
    };

    L(storeLabel);
    prevOutOfs = -1;
    for (int k = 0; k < jcp.nbOcBlocking; k++) {
        for (int j = 0; j < urW; j++) {
            size_t auxOutputOffset = GetOutputOffset(j, k);
            if (dtp == TypeT::F32) {
                prevOutOfs = outStrF32(j, k, auxOutputOffset, prevOutOfs);
            } else if (dtp == TypeT::F16) {
                prevOutOfs = outStrF16(j, k, auxOutputOffset, prevOutOfs);
            } else if (dtp == TypeT::BF16) {
                auto pZreg = zregOut(j, k);
                assert(pZreg.get() != nullptr);
                ARMCG::ZRegS* pZRegS = static_cast<ARMCG::ZRegS*>(pZreg.get());
                Bfcvt(ZReg(REG_INDEX31).h, pAllOne, *pZRegS);
                prevOutOfs = outStrBF16(auxOutputOffset, prevOutOfs);
            }
        }
    }
}

template <CPUISAT isa, TypeT dtp>
void KudnnJitConvFwdKernel<isa, dtp>::ComputeLoop(int urW, int padL, int padR)
{
    if (jcp.ndims == KERNEL_DIMS_5) {
        Mov(regOiOrg, regOi);
    }
    PrepareOutput(urW);
    Label skipComputeLoop;
    if (jcp.ndims == KERNEL_DIMS_5) {
        if ((jcp.dilateD >= jcp.id)
                || (jcp.kd - 1) * (jcp.dilateD + 1)
                        < std::max(static_cast<int>(jcp.fPad), static_cast<int>(jcp.backPad))) {
            Ldr(regKj, Ptr(ABI_PARAM1, static_cast<uint32_t>(offsetof(KudnnConvJitCallS, kdPadding))));
            Cmp(regKj, 0);
            B(Cond::LE, skipComputeLoop);
        }
    }
    if ((jcp.dilateH >= jcp.ih)
            || (jcp.kh - 1) * (jcp.dilateH + 1)
                    < std::max(static_cast<int>(jcp.tPad), static_cast<int>(jcp.bPad))) {
        Ldr(regKj, Ptr(ABI_PARAM1, static_cast<uint32_t>(offsetof(KudnnConvJitCallS, khPadding))));
        Cmp(regKj, 0);
        B(Cond::LE, skipComputeLoop);
    }

    Label icLoop;
    const bool generateIcbLoop = jcp.nbIc > 1 && IsSrcLayoutNxc();
    if (generateIcbLoop) {
        Mov(regInpOrg, regInp);
        Mov(regKerOrg, regKer);
        Ldr(regChannel, Ptr(ABI_PARAM1, static_cast<uint32_t>(offsetof(KudnnConvJitCallS, reduceWork))));
        L(icLoop);
    }

    if (jcp.ver == KudnnConvVersionT::KUDNN_VER_FMA) {
        if (jcp.kernelKind == KudnnConvKernelKindT::KUDNN_CONT_BCAST) {
            ComputeLoopFma(urW, padL, padR);
        } else {
            ComputeLoopFmaCore(urW, padL, padR);
        }
    } else {
        assert(!"unknown convolution version");
    }
    if (generateIcbLoop) {
        const int inpShift = jcp.icBlock * jcp.typeSizeIn;
        AddImm(regInp, regInp, inpShift, regTmpImm);
        const int kerShift = jcp.kd * jcp.kh * jcp.kw * jcp.icBlock
                * jcp.ocBlock * jcp.typeSizeIn;
        AddImm(regKer, regKer, kerShift, regTmpImm);
        SubImm(regChannel, regChannel, jcp.icBlock, regTmpImm);
        Cmp(regChannel, 0);
        B(Cond::GT, icLoop);
        Mov(regKer, regKerOrg);
        Mov(regInp, regInpOrg);
    }

    L(skipComputeLoop);
    StoreOutput(urW);
    if (jcp.ndims == KERNEL_DIMS_5) {
        Mov(regOi, regOiOrg);
    }
}

template <CPUISAT isa, TypeT dtp>
void KudnnJitConvFwdKernel<isa, dtp>::ComputeLoopFma(int urW, int padL, int padR)
{
    int kw = jcp.kw;
    int icBlock = jcp.icBlock;
    int ocBlock = jcp.ocBlock;
    int nbOcBlock = jcp.nbOcBlocking;
    const bool isSourceLayoutNxc = IsSrcLayoutNxc();
    const bool icbLoopInComputeFunction = isSourceLayoutNxc;

    Label khLabel;
    Label kdLabel;

    std::vector<Label> icTailJmp(kw);
    int shiftKernelPtr = jcp.typeSizeIn * jcp.kw * jcp.ocBlock * jcp.icBlock;
    int inpMul = isSourceLayoutNxc ? jcp.ngroups * jcp.ic
                                   : (!jcp.is1stConv ? icBlock : 1);

    int shiftInputPtr = jcp.typeSizeIn * (jcp.dilateH + 1) * jcp.iw * inpMul;

    if (jcp.ndims == KERNEL_DIMS_3 || jcp.ndims == KERNEL_DIMS_4) {
        Mov(auxRegInp, regInp);
        AddImm(auxRegInp2, auxRegInp, 0x100, regTmpImm);
        AddImm(auxRegInp3, auxRegInp2, 0x100, regTmpImm);
        Mov(auxRegKer, regKer);
    }

    if (jcp.ndims == KERNEL_DIMS_5) {
        Mov(regOutOrg, regOut);
        Ldr(regKi, Ptr(ABI_PARAM1, static_cast<uint32_t>(
            offsetof(KudnnConvJitCallS, kdPadding))));
        if (icbLoopInComputeFunction) {
            // need to continue with the same kernel pointer, but as
            // auxRegKerD == regKer we need to save its value and restore
            // it after kd loop
            Mov(auxRegKerOrg, auxRegKerD);
        } else {
            Mov(auxRegKerD, auxRegKerOrg);
        }
        Mov(auxRegInpD, regInp);

        L(kdLabel);
        Ldr(regKj, Ptr(ABI_PARAM1, static_cast<uint32_t>(
            offsetof(KudnnConvJitCallS, khPadding))));
    } else {
        Mov(regKj, regKh);
    }

    if (jcp.ndims == KERNEL_DIMS_5) {
        Mov(auxRegInp, auxRegInpD);
        AddImm(auxRegInp2, auxRegInp, 0x100, regTmpImm);
        AddImm(auxRegInp3, auxRegInp2, 0x100, regTmpImm);
        Mov(auxRegKer, auxRegKerD);
    }

    auto zregInpIdxF32 = [](int iIc, int idx) {
        return ZRegS(iIc)[idx];
    };
    auto zregInpIdxHalfF = [](int iIc, int idx) {
        return ZRegH(iIc)[idx];
    };
    auto zregOutS = [this](int iUr, int iOc) {
        int idx = REG_INDEX29 - (iUr + iOc * jcp.urW);
        return ZRegS(idx);
    };
    auto zregOutH = [this](int iUr, int iOc) {
        int idx = REG_INDEX29 - (iUr + iOc * jcp.urW);
        return ZRegH(idx);
    };

    auto bcastLoad = [this](int jj, int auxInputOffset, int prevOfs) {
        int ofs;
        if ((prevOfs != -1) && ((auxInputOffset - prevOfs) > 0)) {
            ofs = auxInputOffset - prevOfs;
            AddImm(regPrevBcastAddr, regPrevBcastAddr, ofs,
                regTmpImm);
        } else {
            ofs = auxInputOffset;
            AddImm(regPrevBcastAddr, auxRegInp, ofs, regTmpImm);
        }
        dtp == TypeT::F32 ? Ld1rqw(ZRegS(jj), pAllOne, Ptr(regPrevBcastAddr)) :
            Ld1rqh(ZRegH(jj), pAllOne, Ptr(regPrevBcastAddr));
        prevOfs = auxInputOffset;
        return prevOfs;
    };

    auto weiLoadF32 = [this](int auxKernelOffset, int regIdx, int prevOfs) {
        int ofs = auxKernelOffset;
        if (KudnnLdrImmCheck<int, isa>(ofs)) {
            Ldr(ZRegS(regIdx), AdrScImm(Ptr(auxRegKer, static_cast<int32_t>(GetVlenOfs(ofs, isa)))));
        } else {
            int ofsTmp = ofs - prevOfs;
            if ((prevOfs != -1) && KudnnLdrImmCheck<int, isa>(ofsTmp)) {
                Ldr(ZRegS(regIdx), AdrScImm(Ptr(regPrevWeiAddr, static_cast<int32_t>(GetVlenOfs(ofsTmp, isa)))));
            } else {
                if ((prevOfs != -1) && (ofsTmp > 0)) {
                    ofsTmp = auxKernelOffset - prevOfs;
                    AddImm(regPrevWeiAddr, regPrevWeiAddr, ofsTmp,
                        regTmpImm);
                } else {
                    AddImm(regPrevWeiAddr, auxRegKer, ofs, regTmpImm);
                }
                Ld1w(ZRegS(regIdx), pAllOne / PredType::T_Z,
                    Ptr(regPrevWeiAddr));
                prevOfs = ofs;
            }
        }
        return prevOfs;
    };

    auto weiLoadF16OrBF16 = [this](int auxKernelOffset, int regIdx, int prevOfs) {
        int ofs;
        if ((prevOfs != -1) && (auxKernelOffset - prevOfs > 0)) {
            ofs = auxKernelOffset - prevOfs;
            AddImm(regPrevWeiAddr, regPrevWeiAddr, ofs,
                regTmpImm);
        } else {
            ofs = auxKernelOffset;
            AddImm(regPrevWeiAddr, auxRegKer, ofs, regTmpImm);
        }
        dtp == TypeT::F16 ? Ld1h(ZRegH(regIdx), pAllOne / PredType::T_Z,
            Ptr(regPrevWeiAddr)) : Ld1h(ZRegS(regIdx), pAllOne / PredType::T_Z, Ptr(regPrevWeiAddr));
        prevOfs = auxKernelOffset;
        return prevOfs;
    };

    if (dtp == TypeT::F32 || dtp == TypeT::F16) {
        Align(REG_INDEX32);
    } else if (dtp == TypeT::BF16) {
        Align(REG_INDEX16);
    }
    L(khLabel);
    {
        int prevBcastOfs = -1;
        int prevWeiOfs = -1;
        for (int ki = 0; ki < kw; ki++) {
            int jjStart = GetOwStart(ki, padL);
            int jjEnd = GetOwEnd(urW, ki, padR);
            int weiRegOfs = REG_INDEX30;
            for (int ic = 0; ic < icBlock; ic++) {
                int weiCount = 0;
                for (int ii = 0; ii < nbOcBlock; ii++) {
                    int regIdx = weiRegOfs + ii;
                    if (regIdx >= REG_INDEX32) {
                        break;
                    }
                    int auxKernelOffset = jcp.typeSizeIn
                            * (ii * jcp.nbIc * jcp.kh * jcp.kw * jcp.kd
                                            * icBlock * ocBlock
                                    + ki * icBlock * ocBlock + ic * ocBlock);
                    weiCount++;
                    if (jjEnd - jjStart > 0) {
                        prevWeiOfs = dtp == TypeT::F32 ? weiLoadF32(auxKernelOffset,
                            weiRegOfs + (ii % KERNEL_C_LEN2),
                            prevWeiOfs) : weiLoadF16OrBF16(auxKernelOffset, weiRegOfs + (ii % KERNEL_C_LEN2),
                            prevWeiOfs);
                    }
                }

                int tmpKernelClen = dtp == TypeT::F32 ? KERNEL_C_LEN4 : KERNEL_C_LEN8;
                if (ic % tmpKernelClen == 0) {
                    for (int jj = jjStart; jj < jjEnd; jj++) {
                        size_t auxInputOffset
                            = GetInputOffset(ki, ic, jj, padL);
                        prevBcastOfs = bcastLoad(jj,
                            auxInputOffset, prevBcastOfs);
                    }
                }

                for (int ii = 0; ii < nbOcBlock; ii++) {
                    int auxKernelOffset = jcp.typeSizeIn
                            * ((ii + weiCount) * jcp.nbIc * jcp.kh * jcp.kw
                                            * jcp.kd * icBlock * ocBlock
                                    + ki * icBlock * ocBlock + ic * ocBlock);

                    for (int jj = jjStart; jj < jjEnd; jj++) {
                        if (dtp == TypeT::F32) {
                            Fmla(zregOutS(jj, ii),
                                ZRegS(weiRegOfs + (ii % KERNEL_C_LEN2)),
                                zregInpIdxF32(jj, ic % tmpKernelClen));
                        } else if (dtp == TypeT::F16) {
                            Fmla(zregOutH(jj, ii),
                                ZRegH(weiRegOfs + (ii % KERNEL_C_LEN2)),
                                zregInpIdxHalfF(jj, ic % tmpKernelClen));
                        }  else if (dtp == TypeT::BF16) {
                            Bfmlalb(zregOutS(jj, ii),
                                ZRegH(weiRegOfs + (ii % KERNEL_C_LEN2)),
                                zregInpIdxHalfF(jj, ic % IC_LEN8));
                        }
                    }
                    if ((jjEnd - jjStart > 0)
                            && ((weiCount + ii) < nbOcBlock)) {
                        prevWeiOfs = dtp == TypeT::F32 ? weiLoadF32(auxKernelOffset,
                            weiRegOfs + ((ii + weiCount) % KERNEL_C_LEN2), prevWeiOfs) :
                            weiLoadF16OrBF16(auxKernelOffset, weiRegOfs + ((ii + weiCount) % KERNEL_C_LEN2),
                            prevWeiOfs);
                    }
                }
            }
            L(icTailJmp[ki]);
        }

        AddImm(auxRegKer, auxRegKer, shiftKernelPtr, regTmpImm);
        AddImm(auxRegInp, auxRegInp, shiftInputPtr, regTmpImm);
        AddImm(auxRegInp2, auxRegInp, 0x100, regTmpImm);
        AddImm(auxRegInp3, auxRegInp2, 0x100, regTmpImm);
        Sub(regKj, regKj, 1);
        Cmp(regKj, 0);
        B(Cond::GT, khLabel);
    }

    if (jcp.ndims == KERNEL_DIMS_5) {
        int dtpSize = dtp == TypeT::F32 ? TYPE_SIZE : HALF_TYPE_SIZE;
        AddImm(auxRegInpD, auxRegInpD,
            dtpSize * (jcp.dilateD + 1) * jcp.ih * jcp.iw * inpMul,
            regTmpImm);
        const int kerShift
                = dtpSize * jcp.kw * jcp.kh * jcp.ocBlock * jcp.icBlock;
        AddImm(auxRegKerD, auxRegKerD, kerShift, regTmpImm);

        Sub(regKi, regKi, 1);
        Cmp(regKi, 0);
        B(Cond::GT, kdLabel);

        if (icbLoopInComputeFunction) {
            Mov(auxRegKerD, auxRegKerOrg);
        }
        Mov(regOut, regOutOrg);
    }
}

template <CPUISAT isa, TypeT dtp>
void KudnnJitConvFwdKernel<isa, dtp>::ComputeLoopFmaCore(
    int urW, int padL, int padR)
{
    int kw = jcp.kw;
    int icBlock = jcp.icBlock;
    int ocBlock = jcp.ocBlock;
    int nbOcBlock = jcp.nbOcBlocking;
    const bool isSourceLayoutNxc = IsSrcLayoutNxc();
    const bool icbLoopInComputeFunction = isSourceLayoutNxc;

    Label khLabel;
    Label kdLabel;

    std::vector<Label> icTailJmp(kw);
    int shiftKernelPtr
            = jcp.typeSizeIn * jcp.kw * jcp.ocBlock * jcp.icBlock;
    int inpMul = isSourceLayoutNxc ? jcp.ngroups * jcp.ic
                                       : (!jcp.is1stConv ? icBlock : 1);

    int shiftInputPtr
            = jcp.typeSizeIn * (jcp.dilateH + 1) * jcp.iw * inpMul;

    if (jcp.ndims == KERNEL_DIMS_3 || jcp.ndims == KERNEL_DIMS_4) {
        Mov(auxRegInp, regInp);
        AddImm(auxRegInp2, auxRegInp, 0x100, regTmpImm);
        AddImm(auxRegInp3, auxRegInp2, 0x100, regTmpImm);
        Mov(auxRegKer, regKer);
    }

    if (jcp.ndims == KERNEL_DIMS_5) {
        Mov(regOutOrg, regOut);
        Ldr(regKi, Ptr(ABI_PARAM1, static_cast<uint32_t>(offsetof(KudnnConvJitCallS, kdPadding))));
        if (icbLoopInComputeFunction) {
            // need to continue with the same kernel pointer, but as
            // auxRegKerD == regKer we need to save its value and restore
            // it after kd loop
            Mov(auxRegKerOrg, auxRegKerD);
        } else {
            Mov(auxRegKerD, auxRegKerOrg);
        }
        Mov(auxRegInpD, regInp);

        L(kdLabel);
        Ldr(regKj, Ptr(ABI_PARAM1, static_cast<uint32_t>(offsetof(KudnnConvJitCallS, khPadding))));
    } else {
        Mov(regKj, regKh);
    }

    if (jcp.ndims == KERNEL_DIMS_5) {
        Mov(auxRegInp, auxRegInpD);
        AddImm(auxRegInp2, auxRegInp, 0x100, regTmpImm);
        AddImm(auxRegInp3, auxRegInp2, 0x100, regTmpImm);
        Mov(auxRegKer, auxRegKerD);
    }

    auto zregInp = [this](int iIc, int nbXBlocking) {
        return ZReg(iIc + nbXBlocking * jcp.urW);
    };
    auto zregInpS = [this](int iIc, int nbXBlocking) {
        return ZRegS(iIc + nbXBlocking * jcp.urW);
    };
    auto zregInpH = [this](int iIc, int nbXBlocking) {
        return ZRegH(iIc + nbXBlocking * jcp.urW);
    };
    auto zregOutS = [this](int iUr, int iOc) {
        return ZRegS(iUr + iOc * jcp.urW);
    };
    auto zregOutH = [this](int iUr, int iOc) {
        return ZRegH(iUr + iOc * jcp.urW);
    };

    auto bcastLoadF32 = [this, zregInpS](int jj, int tNbOcBlock, int auxInputOffset,
                              int prevOfs) {
        if (KudnnLd1rwImmCheck(auxInputOffset)) {
            Ld1rw(zregInpS(jj, tNbOcBlock), pAllOne,
                AdrScImm(Ptr(auxRegInp, static_cast<int32_t>(auxInputOffset))));
        } else if (KudnnLd1rwImmCheck(auxInputOffset - 0x100)) {
            Ld1rw(zregInpS(jj, tNbOcBlock), pAllOne,
                AdrScImm(Ptr(auxRegInp2,
                    static_cast<int32_t>(auxInputOffset - 0x100))));
        } else if (KudnnLd1rwImmCheck(auxInputOffset - 0x200)) {
            Ld1rw(zregInpS(jj, tNbOcBlock), pAllOne,
                AdrScImm(Ptr(auxRegInp3,
                    static_cast<int32_t>(auxInputOffset - 0x200))));
        } else {
            if ((prevOfs != -1) &&
                KudnnLd1rwImmCheck(auxInputOffset - prevOfs)) {
                Ld1rw(zregInpS(jj, tNbOcBlock), pAllOne,
                    AdrScImm(Ptr(regPrevBcastAddr,
                        static_cast<int32_t>(
                            auxInputOffset - prevOfs))));
            } else {
                int ofs;
                if ((prevOfs != -1) && ((auxInputOffset - prevOfs) > 0)) {
                    ofs = auxInputOffset - prevOfs;
                    AddImm(regPrevBcastAddr, regPrevBcastAddr, ofs, regTmpImm);
                } else {
                    ofs = auxInputOffset;
                    AddImm(regPrevBcastAddr, auxRegInp, ofs, regTmpImm);
                }
                Ld1rw(zregInpS(jj, tNbOcBlock), pAllOne,
                    Ptr(regPrevBcastAddr));
                prevOfs = auxInputOffset;
            }
        }

        return prevOfs;
    };

    auto bcastLoad = [this, zregInpH, zregInp](int jj, int tNbOcBlock, int auxInputOffset,
                              int prevOfs) {
        int ofs;
        if ((prevOfs != -1) && ((auxInputOffset - prevOfs) > 0)) {
            ofs = auxInputOffset - prevOfs;
            AddImm(regPrevBcastAddr, regPrevBcastAddr, ofs, regTmpImm);
        } else {
            ofs = auxInputOffset;
            AddImm(regPrevBcastAddr, auxRegInp, ofs, regTmpImm);
        }
        dtp == TypeT::F16 ? Ld1rh(zregInpH(jj, tNbOcBlock), pAllOne,
            Ptr(regPrevBcastAddr)) : Ld1rh(zregInp(jj, tNbOcBlock).s, pAllOne, Ptr(regPrevBcastAddr));
        prevOfs = auxInputOffset;
        return prevOfs;
    };

    auto weiLoadF32 = [this](int auxKernelOffset, int regIdx, int prevOfs) {
        int ofs = auxKernelOffset;

        if (KudnnLdrImmCheck<int, isa>(ofs)) {
            Ldr(ZRegS(regIdx), AdrScImm(Ptr(auxRegKer, static_cast<int32_t>(GetVlenOfs(ofs, isa)))));
        } else {
            int ofsTmp = ofs - prevOfs;
            if ((prevOfs != -1) && KudnnLdrImmCheck<int, isa>(ofsTmp)) {
                Ldr(ZRegS(regIdx), AdrScImm(Ptr(regPrevWeiAddr, static_cast<int32_t>(GetVlenOfs(ofsTmp, isa)))));
            } else {
                if ((prevOfs != -1) && (ofsTmp > 0)) {
                    ofsTmp = auxKernelOffset - prevOfs;
                    AddImm(regPrevWeiAddr, regPrevWeiAddr, ofsTmp,
                            regTmpImm);
                } else {
                    AddImm(regPrevWeiAddr, auxRegKer, ofs, regTmpImm);
                }
                Ld1w(ZRegS(regIdx), pAllOne / PredType::T_Z,
                    Ptr(regPrevWeiAddr));
                prevOfs = ofs;
            }
        }
        return prevOfs;
    };

    auto weiLoadF16OrF16 = [this](int auxKernelOffset, int regIdx, int prevOfs) {
        int ofs;
        if ((prevOfs != -1) && (auxKernelOffset - prevOfs > 0)) {
            ofs = auxKernelOffset - prevOfs;
            AddImm(regPrevWeiAddr, regPrevWeiAddr, ofs,
                    regTmpImm);
        } else {
            ofs = auxKernelOffset;
            AddImm(regPrevWeiAddr, auxRegKer, ofs, regTmpImm);
        }
        dtp == TypeT::F16 ? Ld1h(ZRegH(regIdx), pAllOne / PredType::T_Z,
            Ptr(regPrevWeiAddr)) : Ld1h(ZRegS(regIdx), pAllOne / PredType::T_Z, Ptr(regPrevWeiAddr));
        prevOfs = auxKernelOffset;
        return prevOfs;
    };

    Align(REG_INDEX32);
    L(khLabel);
    {
        int prevBcastOfs = -1;
        int prevWeiOfs = -1;
        for (int ki = 0; ki < kw; ki++) {
            int jjStart = GetOwStart(ki, padL);
            int jjEnd = GetOwEnd(urW, ki, padR);

            int weiRegOfs = nbOcBlock * jcp.urW;
            weiRegOfs += urW >= KERNEL_UR_W_LEN16 ? KERNEL_UR_W_LEN1 : jjEnd;
            int numRegs4wei = REG_INDEX32 - weiRegOfs;
            for (int ic = 0; ic < icBlock; ic++) {
                int weiCount = 0;
                for (int ii = 0; ii < nbOcBlock; ii++) {
                    int regIdx = weiRegOfs + ii;
                    if (regIdx >= REG_INDEX32) {
                        break;
                    }
                    int auxKernelOffset = jcp.typeSizeIn
                            * (ii * jcp.nbIc * jcp.kh * jcp.kw * jcp.kd
                                            * icBlock * ocBlock
                                    + ki * icBlock * ocBlock + ic * ocBlock);
                    weiCount++;
                    if (jjEnd - jjStart > 0) {
                        prevWeiOfs = dtp == TypeT::F32 ? weiLoadF32(auxKernelOffset,
                            weiRegOfs + (ii % numRegs4wei), prevWeiOfs) :
                            weiLoadF16OrF16(auxKernelOffset, weiRegOfs + (ii % numRegs4wei), prevWeiOfs);
                    }
                }
                for (int jj = jjStart; jj < jjEnd; jj++) {
                    size_t auxInputOffset = GetInputOffset(ki, ic, jj, padL);
                    prevBcastOfs = dtp == TypeT::F32 ? bcastLoadF32(jj, nbOcBlock,
                        auxInputOffset, prevBcastOfs) : bcastLoad(jj, nbOcBlock,
                        auxInputOffset, prevBcastOfs);
                }

                for (int ii = 0; ii < nbOcBlock; ii++) {
                    int auxKernelOffset = jcp.typeSizeIn
                            * ((ii + weiCount) * jcp.nbIc * jcp.kh * jcp.kw * jcp.kd
                                            * icBlock * ocBlock
                                    + ki * icBlock * ocBlock + ic * ocBlock);
                    for (int jj = jjStart; jj < jjEnd; jj++) {
                        if (jcp.kernelKind == KudnnConvKernelKindT::KUDNN_EXPL_BCAST) {
                            if (dtp == TypeT::F32) {
                                Fmla(zregOutS(jj, ii), pAllOne, zregInpS(jj, nbOcBlock),
                                    ZRegS(weiRegOfs + (ii % numRegs4wei)));
                            } else if (dtp == TypeT::F16) {
                                Fmla(zregOutH(jj, ii), pAllOne, zregInpH(jj, nbOcBlock),
                                    ZRegH(weiRegOfs + (ii % numRegs4wei)));
                            } else if (dtp == TypeT::BF16) {
                                Bfmlalb(zregOutS(jj, ii),
                                    zregInp(jj, nbOcBlock).h,
                                    ZRegH(weiRegOfs + (ii % numRegs4wei)));
                            }
                        }
                    }
                    if ((jjEnd - jjStart > 0)
                           && ((ii + weiCount) < nbOcBlock)) {
                        prevWeiOfs = dtp == TypeT::F32 ? weiLoadF32(auxKernelOffset,
                            weiRegOfs + ((ii + weiCount) % numRegs4wei), prevWeiOfs) :
                            weiLoadF16OrF16(auxKernelOffset, weiRegOfs + ((ii + weiCount) % numRegs4wei), prevWeiOfs);
                    }
                }
            }
            L(icTailJmp[ki]);
        }

        AddImm(auxRegKer, auxRegKer, shiftKernelPtr, regTmpImm);
        AddImm(auxRegInp, auxRegInp, shiftInputPtr, regTmpImm);
        AddImm(auxRegInp2, auxRegInp, 0x100, regTmpImm);
        AddImm(auxRegInp3, auxRegInp2, 0x100, regTmpImm);
        Sub(regKj, regKj, 1);
        Cmp(regKj, 0);
        B(Cond::GT, khLabel);
    }

    if (jcp.ndims == KERNEL_DIMS_5) {
        int dtpSize = dtp == TypeT::F32 ? TYPE_SIZE : HALF_TYPE_SIZE;
        AddImm(auxRegInpD, auxRegInpD,
            dtpSize * (jcp.dilateD + 1) * jcp.ih * jcp.iw * inpMul,
            regTmpImm);
        const int kerShift = dtpSize * jcp.kw * jcp.kh * jcp.ocBlock * jcp.icBlock;
        AddImm(auxRegKerD, auxRegKerD, kerShift, regTmpImm);

        Sub(regKi, regKi, 1);
        Cmp(regKi, 0);
        B(Cond::GT, kdLabel);

        if (icbLoopInComputeFunction) {
            Mov(auxRegKerD, auxRegKerOrg);
        }
        Mov(regOut, regOutOrg);
    }
}

template <CPUISAT isa, TypeT dtp>
void KudnnJitConvFwdKernel<isa, dtp>::Generate()
{
    int iw = jcp.iw;
    int ow = jcp.ow;
    int owBlock = jcp.owBlock;
    int nbOw = jcp.nbOw;
    assert(owBlock == jcp.ow && nbOw == 1);
    int kw = jcp.kw;
    int lPad = jcp.lPad;
    int urW = jcp.urW;
    int urWTail = jcp.urWTail;
    int strideW = jcp.strideW;

    int inpMult = IsSrcLayoutNxc() ? jcp.ngroups * jcp.ic
                                   : (jcp.is1stConv ? 1 : jcp.icBlock);
    int inpShiftPad = jcp.typeSizeIn * (urW * strideW - lPad) * inpMult;
    int inpShift = jcp.typeSizeIn * urW * strideW * inpMult;
    int outShift = jcp.typeSizeOut * urW
            * (IsDstLayoutNxc() ? jcp.ngroups * jcp.oc : jcp.ocBlock);

    Preamble();

    Ldr(regInp, Ptr(ABI_PARAM1, static_cast<uint32_t>(
        offsetof(KudnnConvJitCallS, src))));
    Ldr(regOut, Ptr(ABI_PARAM1, static_cast<uint32_t>(
        offsetof(KudnnConvJitCallS, dst))));
    Ldr(regKer, Ptr(ABI_PARAM1, static_cast<uint32_t>(
        offsetof(KudnnConvJitCallS, filt))));
    Ldr(regKh, Ptr(ABI_PARAM1, static_cast<uint32_t>(
        offsetof(KudnnConvJitCallS, khPadding))));
    if (jcp.ndims == KERNEL_DIMS_5) {
        Mov(auxRegKerOrg, regKer);
    }

    int rPad = std::max(0, static_cast<int>(jcp.rPad));
    int nOi = ow / urW;
    int rPad1 = CalculateEndPadding(lPad, urW * nOi, iw, strideW,
        CalculateExtendedFilterSize(kw, jcp.dilateW));
    int nOiLast2 = nOi - 1 > 0 ? nOi - 1 : 0;
    int rPadLast2 = CalculateEndPadding(lPad, urW * nOiLast2, iw, strideW,
        CalculateExtendedFilterSize(kw, jcp.dilateW));
    // ow is being processed as a whole - with left and right paddings
    // nOi is # of output width blocks
    if (rPad1 > 0) {
        nOi--;
    }

    if (ow == urW) {
        Ldr(regOutPrf, Ptr(ABI_PARAM1, static_cast<uint32_t>(
            offsetof(KudnnConvJitCallS, dstPrf))));
        ComputeLoop(urW, lPad, rPad);
    } else {
        Mov(regOutPrf, regOut);
        if (nOi == 0) {
            AddImm(regOutPrf, regOutPrf, outShift, regTmpImm);
            ComputeLoop(urW, lPad, rPad1);
            AddImm(regOut, regOut, outShift, regTmpImm);
            if (urWTail != 0) {
                AddImm(regOutPrf, regOutPrf, outShift, regTmpImm);
                int lPad1 = lPad - urW * strideW;
                if (lPad1 > 0) {
                    ComputeLoop(urWTail, lPad1, rPad);
                } else {
                    AddImm(regInp, regInp, inpShiftPad, regTmpImm);
                    ComputeLoop(urWTail, 0, rPad);
                }
            }
        } else if (lPad <= urW && rPadLast2 <= 0) {
            // generate loop for simplify when lpad <= ur_w
            Mov(regOi, 0);
            if (lPad > 0) {
                AddImm(regOutPrf, regOutPrf, outShift, regTmpImm);
                ComputeLoop(urW, lPad, 0);
                AddImm(regInp, regInp, inpShiftPad, regTmpImm);
                AddImm(regOut, regOut, outShift, regTmpImm);
                AddImm(regOi, regOi, 1, regTmpImm);
            }
            if ((lPad <= 0 && nOi > 0) || (lPad > 0 && nOi > 1)) {
                Label owLoopLabel;
                L(owLoopLabel);
                {
                    AddImm(regOutPrf, regOutPrf, outShift,
                            regTmpImm);
                    ComputeLoop(urW, 0, 0);

                    AddImm(regInp, regInp, inpShift, regTmpImm);
                    AddImm(regOut, regOut, outShift, regTmpImm);
                    AddImm(regOi, regOi, 1, regTmpImm);
                    CmpImm(regOi, nOi, regTmpImm);

                    B(Cond::LT, owLoopLabel);
                }
            }
            if (rPad1 > 0) {
                AddImm(regOutPrf, regOutPrf, outShift, regTmpImm);
                ComputeLoop(urW, 0, rPad1);
                AddImm(regInp, regInp, inpShift, regTmpImm);
                AddImm(regOut, regOut, outShift, regTmpImm);
            }
            if (urWTail != 0) {
                AddImm(regOutPrf, regOutPrf, outShift, regTmpImm);
                ComputeLoop(urWTail, 0, rPad);
            }
        } else {
            nOi = rPad1 > 0 ? nOi : nOi - 1;
            int inpShiftPadl = 0;
            int preInpShiftPadl = 0;
            for (int oi = 1; oi <= nOi + 1; oi++) {
                AddImm(regOutPrf, regOutPrf, outShift, regTmpImm);
                int lPadL = lPad - (oi - 1) * urW * strideW;
                int lPadR = lPadL > 0 ? lPadL : 0;
                int rPadL = CalculateEndPadding(lPad, oi * urW, iw, strideW,
                    CalculateExtendedFilterSize(kw, jcp.dilateW));
                int rPadR = rPadL > 0 ? rPadL : 0;
                ComputeLoop(urW, lPadR, rPadR);
                if (lPad - oi * urW * strideW < 0) {
                    inpShiftPadl = jcp.typeSizeIn * (oi * urW * strideW - lPad) * inpMult;
                    AddImm(regInp, regInp, inpShiftPadl - preInpShiftPadl, regTmpImm);
                    preInpShiftPadl = inpShiftPadl;
                }
                AddImm(regOut, regOut, outShift, regTmpImm);
                AddImm(regOi, regOi, 1, regTmpImm);
            }
            if (urWTail != 0) {
                AddImm(regOutPrf, regOutPrf, outShift, regTmpImm);
                int lPadM = lPad - (nOi + 1) * urW * strideW;
                int lPadN = lPadM > 0 ? lPadM : 0;
                ComputeLoop(urWTail, lPadN, rPad);
            }
        }
    }
    Postamble();
}

/* struct instantiation */
#ifdef KUDNNL_920Pro
template struct KudnnJitConvFwdKernel<CPUISAT::VectorComputation_512, Element::TypeT::F32>;
template struct KudnnJitConvFwdKernel<CPUISAT::VectorComputation_512, Element::TypeT::F16>;
template struct KudnnJitConvFwdKernel<CPUISAT::VectorComputation_512, Element::TypeT::BF16>;
#else
template struct KudnnJitConvFwdKernel<CPUISAT::VectorComputation_256, Element::TypeT::F32>;
template struct KudnnJitConvFwdKernel<CPUISAT::VectorComputation_256, Element::TypeT::F16>;
template struct KudnnJitConvFwdKernel<CPUISAT::VectorComputation_256, Element::TypeT::BF16>;
#endif

} // namespace KuDNN
