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
#include "internal_funcs.hpp"
namespace KuDNN {
using namespace ARMCG;
using namespace Element;

template <CPUISAT isa, TypeT dtp>
void KdnnJitConvFwdKernel<isa, dtp>::PrepareOutput(IntType urW)
{
#ifdef KUDNNL_920B
    if (jcp.kernelKind == KdnnConvKernelKindT::KUDNN_EMBD_BCAST) {
        for (IntType k = 0; k < jcp.nbOcBlocking; k++) {
            for (IntType j = 0; j < urW; j++) {
                Movi(VReg(j + k * jcp.urW).s, 0);
            }
        }
    } else {
#endif
        auto zregOut = [this](IntType iUr, IntType iOc) {
            if (jcp.kernelKind == KdnnConvKernelKindT::KUDNN_CONT_BCAST) {
                IntType idx = REG_INDEX29 - (iUr + iOc * jcp.urW);
                return ZRegFactory::GetZRegAcdTp4Fmov(dtp, idx);
            }
            IntType idx = iUr + iOc * jcp.urW;
            return ZRegFactory::GetZRegAcdTp4Fmov(dtp, idx);
        };

        for (IntType k = 0; k < jcp.nbOcBlocking; k++) {
            for (IntType j = 0; j < urW; j++) {
                FmovZReg(dtp, zregOut(j, k));
            }
        }
#ifdef KUDNNL_920B
    }
#endif
}

#ifdef KUDNNL_920B
template <CPUISAT isa, TypeT dtp>
void KdnnJitConvFwdKernel<isa, dtp>::StoreOutputForVReg(IntType urW)
{
    Label noUpdateLabel;
    Label storeLabel;
    Label eltwiseLabel;
    auto tTest = [this](const IntType cond) { return Tst(regChannel, cond); };

    Ldr(regChannel, Ptr(ABI_PARAM1, static_cast<uint32_t>(offsetof(KdnnConvJitCallS, flags))));

    if (jcp.withBias) {
        Ldr(regBias, Ptr(ABI_PARAM1, static_cast<uint32_t>(offsetof(KdnnConvJitCallS, bias))));
    }

    if (!jcp.withSum) {
        auto tJmp = [this](const Label &l) { return B(Cond::NE, l); };
        tTest(static_cast<int32_t>(KdnnConvFlag::KUDNN_FLAG_IC_FIRST));
        tJmp(noUpdateLabel);
    }

    IntType regOfs = jcp.urW * jcp.nbOcBlocking;
    IntType nRegs = REG_INDEX32 - regOfs;

    IntType prevOutOfs = -1;
    for (IntType k = 0; k < jcp.nbOcBlocking; k++) {
        for (IntType j = 0; j < urW; j++) {
            IntType tmpIdx = regOfs + ((j + k * urW) % nRegs);
            SizeType auxOutputOffset = GetOutputOffset(j, k);
            if (j == 0) {
                AddImm(regOutOfs, regOut, auxOutputOffset, regTmpImm);
                prevOutOfs = auxOutputOffset;
                Ld1(VReg4SList(VReg(tmpIdx).s), Ptr(regOutOfs));
            } else {
                AddImm(regOutOfs, regOutOfs, auxOutputOffset - prevOutOfs, regTmpImm);
                prevOutOfs = auxOutputOffset;
                Ld1(VReg4SList(VReg(tmpIdx).s), Ptr(regOutOfs));
            }
            Fadd(VReg(j + k * jcp.urW).s, VReg(j + k * jcp.urW).s, VReg(tmpIdx).s);
        }
    }

    B(eltwiseLabel);

    auto biasLoad = [this](IntType biasOffset, IntType idx) {
        IntType ofs = biasOffset;
        if ((GetVlenOfs(ofs, isa) < LDRMAX) && (GetVlenOfs(ofs, isa) >= (-1 * LDRMAX))
                && ((ofs & 0x3f) == 0)) {
            AddImm(xDefaultAddr, regBias, ofs, X_TMP_0);
            Ld1(VReg4SList(VReg(idx).s), Ptr(xDefaultAddr));
        } else {
            AddImm(regTmpAddr, regBias, ofs, regTmpImm);
            Ld1(VReg4SList(VReg(idx).s), Ptr(regTmpAddr));
        }
    };

    L(noUpdateLabel);
    if (jcp.withBias) {
        for (IntType k = 0; k < jcp.nbOcBlocking; k++) {
            SizeType biasOffset = jcp.typeSizeOut * k * jcp.ocBlock;
            IntType idx = regOfs + (k % nRegs);
            biasLoad(biasOffset, idx);
            for (IntType j = 0; j < urW; j++) {
                Fadd(VReg(j + k * jcp.urW).s, VReg(j + k * jcp.urW).s, VReg(idx).s);
            }
        }
    }

    L(eltwiseLabel);
    auto outStr = [this](IntType j, IntType k, IntType auxOutputOffset, IntType tPrevOutOfs) {
        IntType ofs = auxOutputOffset;
        if (KdnnStrImmCheck<IntType, isa>(ofs)) {
            AddImm(xDefaultAddr, regOut, ofs, X_TMP_0);
            St1(VReg4SList(VReg(j + k * jcp.urW).s), Ptr(xDefaultAddr));
        } else if ((tPrevOutOfs != -1)
                && KdnnStrImmCheck<IntType, isa>(ofs - tPrevOutOfs)) {
            AddImm(xDefaultAddr, regTmpAddr, (ofs - tPrevOutOfs), X_TMP_0);
            St1(VReg4SList(VReg(j + k * jcp.urW).s), Ptr(xDefaultAddr));
        } else {
            if (tPrevOutOfs == -1) {
                AddImm(regTmpAddr, regOut, ofs, regTmpImm);
            } else {
                AddImm(regTmpAddr, regTmpAddr, ofs - tPrevOutOfs, regTmpImm);
            }
            St1(VReg4SList(VReg(j + k * jcp.urW).s), Ptr(regTmpAddr));
            tPrevOutOfs = auxOutputOffset;
        }
        return tPrevOutOfs;
    };

    L(storeLabel);
    prevOutOfs = -1;
    for (IntType k = 0; k < jcp.nbOcBlocking; k++) {
        for (IntType j = 0; j < urW; j++) {
            SizeType auxOutputOffset = GetOutputOffset(j, k);
            prevOutOfs = outStr(j, k, auxOutputOffset, prevOutOfs);
        }
    }
}
#endif

template <CPUISAT isa, TypeT dtp>
void KdnnJitConvFwdKernel<isa, dtp>::StoreOutputFadd(IntType rOfs, IntType nRegs,
    IntType& preOutOfs, IntType urW)
{
    auto zregOut = [this](IntType iUr, IntType iOc) {
        IntType idx = jcp.kernelKind == KdnnConvKernelKindT::KUDNN_CONT_BCAST
                  ? REG_INDEX29 - (iUr + iOc * jcp.urW)
                  : iUr + iOc * jcp.urW;
        return ZRegFactory::GetZRegAcdTp(dtp, idx);
    };
    for (IntType k = 0; k < jcp.nbOcBlocking; k++) {
        for (IntType j = 0; j < urW; j++) {
            IntType tmpIdx = jcp.kernelKind == KdnnConvKernelKindT::KUDNN_CONT_BCAST
                         ? j
                         : rOfs + ((j + k * urW) % nRegs);
            SizeType auxOutputOffset = GetOutputOffset(j, k);
            if (j == 0) {
                AddImm(regOutOfs, regOut, auxOutputOffset, regTmpImm);
                preOutOfs = auxOutputOffset;
                dtp == TypeT::F32 ? Ld1w(ZReg(tmpIdx).s, pAllOne / PredType::T_Z, Ptr(regOutOfs)) :
                    (dtp == TypeT::F16 ? Ld1h(ZReg(tmpIdx).h, pAllOne / PredType::T_Z, Ptr(regOutOfs)) :
                    Ld1h(ZReg(tmpIdx).s, pAllOne / PredType::T_Z, Ptr(regOutOfs)));
            } else if (KdnnLdrImmCheck<SizeType, isa>(auxOutputOffset - preOutOfs)) {
                if (dtp == TypeT::BF16) {
                    AddImm(xDefaultAddr, regOutOfs, (auxOutputOffset - preOutOfs), X_TMP_0);
                } else if (dtp == TypeT::F16) {
                    AddImm(regOutOfs, regOutOfs, auxOutputOffset - preOutOfs, regTmpImm);
                    preOutOfs = auxOutputOffset;
                }
                dtp == TypeT::F32 ? Ldr(ZRegS(tmpIdx), AdrScImm(Ptr(regOutOfs,
                    static_cast<int32_t>(GetVlenOfs(auxOutputOffset - preOutOfs, isa))))) :
                    (dtp == TypeT::F16 ? Ld1h(ZReg(tmpIdx).h, pAllOne / PredType::T_Z, Ptr(regOutOfs)) :
                        Ld1h(ZReg(tmpIdx).s, pAllOne / PredType::T_Z, Ptr(xDefaultAddr)));
            } else {
                AddImm(regOutOfs, regOutOfs,
                    auxOutputOffset - preOutOfs, regTmpImm);
                preOutOfs = auxOutputOffset;
                dtp == TypeT::F32 ? Ld1w(ZReg(tmpIdx).s, pAllOne / PredType::T_Z, Ptr(regOutOfs)) : (
                    dtp == TypeT::F16 ? Ld1h(ZReg(tmpIdx).h, pAllOne / PredType::T_Z, Ptr(regOutOfs)) :
                    Ld1h(ZReg(tmpIdx).s, pAllOne / PredType::T_Z, Ptr(regOutOfs)));
            }
        }
        for (IntType j = 0; j < urW; j++) {
            IntType tmpIdx = jcp.kernelKind == KdnnConvKernelKindT::KUDNN_CONT_BCAST
                         ? j
                         : rOfs + ((j + k * urW) % nRegs);
            auto pZreg = zregOut(j, k);
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
}

template <CPUISAT isa, TypeT dtp>
void KdnnJitConvFwdKernel<isa, dtp>::StoreOutput(IntType urW)
{
#ifdef KUDNNL_920B
    if (jcp.kernelKind == KdnnConvKernelKindT::KUDNN_EMBD_BCAST) {
        StoreOutputForVReg(urW);
        return;
    }
#endif
    Label noUpdateLabel;
    Label storeLabel;
    Label eltwiseLabel;
    auto tTest = [this](const IntType cond) { return Tst(regChannel, cond); };
    auto zregOut = [this](IntType iUr, IntType iOc) {
        IntType idx = jcp.kernelKind == KdnnConvKernelKindT::KUDNN_CONT_BCAST
                  ? REG_INDEX29 - (iUr + iOc * jcp.urW)
                  : iUr + iOc * jcp.urW;
        return ZRegFactory::GetZRegAcdTp(dtp, idx);
    };

    Ldr(regChannel, Ptr(ABI_PARAM1, static_cast<uint32_t>(
        offsetof(KdnnConvJitCallS, flags))));

    if (jcp.withBias) {
        Ldr(regBias, Ptr(ABI_PARAM1, static_cast<uint32_t>(
            offsetof(KdnnConvJitCallS, bias))));
    }

    if (!jcp.withSum) {
        auto tJmp = [this](const Label &l) { return B(Cond::NE, l); };
        tTest(static_cast<int32_t>(KdnnConvFlag::KUDNN_FLAG_IC_FIRST));
        tJmp(noUpdateLabel);
    }

    IntType regOfs = jcp.urW * jcp.nbOcBlocking;
    IntType nRegs = REG_INDEX32 - regOfs;
    if (dtp == TypeT::BF16) {
        nRegs = REG_INDEX32 - regOfs - REG_INDEX2;
        Fmov(ZReg(REG_INDEX31).s, pAllOne, 1.f);
        Bfcvt(ZReg(REG_INDEX30).h, pAllOne, ZReg(REG_INDEX31).s);
    }
    IntType prevOutOfs = -1;
    StoreOutputFadd(regOfs, nRegs, prevOutOfs, urW);
    B(eltwiseLabel);

    auto biasLoad = [this](SizeType biasOffset, IntType idx) {
        SizeType ofs = biasOffset;
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
        for (IntType k = 0; k < jcp.nbOcBlocking; k++) {
            SizeType biasOffset = jcp.typeSizeOut * k * jcp.ocBlock;
            IntType idx = jcp.kernelKind == KdnnConvKernelKindT::KUDNN_CONT_BCAST
                      ? k : regOfs + (k % nRegs);
            biasLoad(biasOffset, idx);
            if (dtp == TypeT::BF16) {
                Fmov(ZReg(REG_INDEX31).s, pAllOne, 1.f);
                Bfcvt(ZReg(REG_INDEX30).h, pAllOne, ZReg(REG_INDEX31).s);
            }
            
            for (IntType j = 0; j < urW; j++) {
                auto pZreg = zregOut(j, k);
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
    auto outStrF32 = [this, zregOut](IntType j, IntType k, IntType auxOutputOffset, IntType tPrevOutOfs) {
        auto pZreg = zregOut(j, k);
        ARMCG::ZRegS* pZRegS = static_cast<ARMCG::ZRegS*>(pZreg.get());
        IntType ofs = auxOutputOffset;
        if (KdnnStrImmCheck<IntType, isa>(ofs)) {
            AddImm(xDefaultAddr, regOut, ofs, X_TMP_0);
            St1w(*pZRegS, pAllOne / PredType::T_Z, Ptr(xDefaultAddr));
        } else if ((tPrevOutOfs != -1)
                && KdnnStrImmCheck<IntType, isa>(ofs - tPrevOutOfs)) {
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

    auto outStrF16 = [this, zregOut](IntType j, IntType k, SizeType auxOutputOffset, IntType tPrevOutOfs) {
        auto pZreg = zregOut(j, k);
        ARMCG::ZRegH* pZRegH = static_cast<ARMCG::ZRegH*>(pZreg.get());
        SizeType ofs = auxOutputOffset;
        if (tPrevOutOfs == -1) {
            AddImm(regTmpAddr, regOut, ofs, regTmpImm);
        } else {
            AddImm(regTmpAddr, regTmpAddr, ofs - tPrevOutOfs, regTmpImm);
        }
        St1h(*pZRegH, pAllOne / PredType::T_Z, Ptr(regTmpAddr));
        tPrevOutOfs = auxOutputOffset;
        return tPrevOutOfs;
    };

    auto outStrBF16 = [this](SizeType auxOutputOffset, IntType tPrevOutOfs) {
        SizeType ofs = auxOutputOffset;
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
    for (IntType k = 0; k < jcp.nbOcBlocking; k++) {
        for (IntType j = 0; j < urW; j++) {
            SizeType auxOutputOffset = GetOutputOffset(j, k);
            if (dtp == TypeT::F32) {
                prevOutOfs = outStrF32(j, k, auxOutputOffset, prevOutOfs);
            } else if (dtp == TypeT::F16) {
                prevOutOfs = outStrF16(j, k, auxOutputOffset, prevOutOfs);
            } else if (dtp == TypeT::BF16) {
                auto pZreg = zregOut(j, k);
                ARMCG::ZRegS* pZRegS = static_cast<ARMCG::ZRegS*>(pZreg.get());
                Bfcvt(ZReg(REG_INDEX31).h, pAllOne, *pZRegS);
                prevOutOfs = outStrBF16(auxOutputOffset, prevOutOfs);
            }
        }
    }
}

template <CPUISAT isa, TypeT dtp>
void KdnnJitConvFwdKernel<isa, dtp>::ComputeLoopCommon(IntType urW, IntType padL, IntType padR)
{
    Label kdLabel;
    const bool isSourceLayoutNxc = IsSrcLayoutNxc();
    IntType inpMul = isSourceLayoutNxc ? jcp.ngroups * jcp.ic
                                   : (!jcp.is1stConv ? jcp.icBlock : 1);

    if (jcp.ndims == KERNEL_DIMS_3 || jcp.ndims == KERNEL_DIMS_4) {
        Mov(auxRegInp, regInp);
        AddImm(auxRegInp2, auxRegInp, 0x100, regTmpImm);
        AddImm(auxRegInp3, auxRegInp2, 0x100, regTmpImm);
        Mov(auxRegKer, regKer);
    }

    if (jcp.ndims == KERNEL_DIMS_5) {
        Mov(regOutOrg, regOut);
        Ldr(regKi, Ptr(ABI_PARAM1, static_cast<uint32_t>(offsetof(KdnnConvJitCallS, kdPadding))));
        if (isSourceLayoutNxc) {
            // need to continue with the same kernel pointer, but as
            // auxRegKerD == regKer we need to save its value and restore
            // it after kd loop
            Mov(auxRegKerOrg, auxRegKerD);
        } else {
            Mov(auxRegKerD, auxRegKerOrg);
        }
        Mov(auxRegInpD, regInp);

        L(kdLabel);
        Ldr(regKj, Ptr(ABI_PARAM1, static_cast<uint32_t>(offsetof(KdnnConvJitCallS, khPadding))));
    } else {
        Mov(regKj, regKh);
    }

    if (jcp.ndims == KERNEL_DIMS_5) {
        Mov(auxRegInp, auxRegInpD);
        AddImm(auxRegInp2, auxRegInp, 0x100, regTmpImm);
        AddImm(auxRegInp3, auxRegInp2, 0x100, regTmpImm);
        Mov(auxRegKer, auxRegKerD);
    }

    if (jcp.kernelKind == KdnnConvKernelKindT::KUDNN_EXPL_BCAST
        || jcp.kernelKind == KdnnConvKernelKindT::KUDNN_EMBD_BCAST) {
        ComputeLoopFmaCore(urW, padL, padR);
    } else {
        ComputeLoopFma(urW, padL, padR);
    }

    if (jcp.ndims == KERNEL_DIMS_5) {
        IntType dtpSize = dtp == TypeT::F32 ? TYPE_SIZE : HALF_TYPE_SIZE;
        AddImm(auxRegInpD, auxRegInpD,
            dtpSize * (jcp.dilateD + 1) * jcp.ih * jcp.iw * inpMul,
            regTmpImm);
        const SizeType kerShift = dtpSize * jcp.kw * jcp.kh * jcp.ocBlock * jcp.icBlock;
        AddImm(auxRegKerD, auxRegKerD, kerShift, regTmpImm);

        Sub(regKi, regKi, 1);
        Cmp(regKi, 0);
        B(Cond::GT, kdLabel);

        if (isSourceLayoutNxc) {
            Mov(auxRegKerD, auxRegKerOrg);
        }
        Mov(regOut, regOutOrg);
    }
}

template <CPUISAT isa, TypeT dtp>
void KdnnJitConvFwdKernel<isa, dtp>::ComputeLoop(IntType urW, IntType padL, IntType padR)
{
    if (jcp.ndims == KERNEL_DIMS_5) {
        Mov(regOiOrg, regOi);
    }
    PrepareOutput(urW);
    Label skipComputeLoop;
    if (jcp.ndims == KERNEL_DIMS_5) {
        if ((jcp.dilateD >= jcp.id)
                || (jcp.kd - 1) * (jcp.dilateD + 1)
                        < (jcp.fPad > jcp.backPad ? jcp.fPad : jcp.backPad)) {
            Ldr(regKj, Ptr(ABI_PARAM1, static_cast<uint32_t>(offsetof(KdnnConvJitCallS, kdPadding))));
            Cmp(regKj, 0);
            B(Cond::LE, skipComputeLoop);
        }
    }
    if ((jcp.dilateH >= jcp.ih)
            || (jcp.kh - 1) * (jcp.dilateH + 1)
                    < (jcp.tPad > jcp.bPad ? jcp.tPad : jcp.bPad)) {
        Ldr(regKj, Ptr(ABI_PARAM1, static_cast<uint32_t>(offsetof(KdnnConvJitCallS, khPadding))));
        Cmp(regKj, 0);
        B(Cond::LE, skipComputeLoop);
    }

    Label icLoop;
    const bool generateIcbLoop = jcp.nbIc > 1 && IsSrcLayoutNxc();
    if (generateIcbLoop) {
        Mov(regInpOrg, regInp);
        Mov(regKerOrg, regKer);
        Ldr(regChannel, Ptr(ABI_PARAM1, static_cast<uint32_t>(offsetof(KdnnConvJitCallS, reduceWork))));
        L(icLoop);
    }

    if (jcp.ver == KdnnConvVersionT::KUDNN_VER_FMA) {
        ComputeLoopCommon(urW, padL, padR);
    } else {
        printf("unknown convolution version\n");
        return;
    }
    if (generateIcbLoop) {
        const IntType inpShift = jcp.icBlock * jcp.typeSizeIn;
        AddImm(regInp, regInp, inpShift, regTmpImm);
        const IntType kerShift = jcp.kd * jcp.kh * jcp.kw * jcp.icBlock
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
void KdnnJitConvFwdKernel<isa, dtp>::ComputeLoopFma(IntType urW, IntType padL, IntType padR)
{
    IntType kw = jcp.kw;
    IntType icBlock = jcp.icBlock;
    IntType ocBlock = jcp.ocBlock;
    IntType nbOcBlock = jcp.nbOcBlocking;
    Label khLabel;

    std::vector<Label> icTailJmp(kw);
    SizeType shiftKernelPtr = jcp.typeSizeIn * jcp.kw * jcp.ocBlock * jcp.icBlock;
    SizeType inpMul = IsSrcLayoutNxc() ? jcp.ngroups * jcp.ic
                                   : (!jcp.is1stConv ? icBlock : 1);

    SizeType shiftInputPtr = jcp.typeSizeIn * (jcp.dilateH + 1) * jcp.iw * inpMul;

    auto zregInpIdxF32 = [](IntType iIc, IntType idx) {
        return ZRegS(iIc)[idx];
    };
    auto zregInpIdxHalfF = [](IntType iIc, IntType idx) {
        return ZRegH(iIc)[idx];
    };
    auto zregOutS = [this](IntType iUr, IntType iOc) {
        IntType idx = REG_INDEX29 - (iUr + iOc * jcp.urW);
        return ZRegS(idx);
    };
    auto zregOutH = [this](IntType iUr, IntType iOc) {
        IntType idx = REG_INDEX29 - (iUr + iOc * jcp.urW);
        return ZRegH(idx);
    };

    auto bcastLoad = [this](IntType jj, SizeType auxInputOffset, IntType prevOfs) {
        SizeType ofs;
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
        prevOfs = static_cast<IntType>(auxInputOffset);
        return prevOfs;
    };

    auto weiLoadF16OrBF16 = [this](SizeType auxKernelOffset, IntType regIdx, IntType prevOfs) {
        SizeType ofs;
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
        prevOfs = static_cast<IntType>(auxKernelOffset);
        return prevOfs;
    };

    if (dtp == TypeT::F32 || dtp == TypeT::F16) {
        Align(REG_INDEX32);
    } else if (dtp == TypeT::BF16) {
        Align(REG_INDEX16);
    }
    L(khLabel);
    {
        IntType prevBcastOfs = -1;
        IntType prevWeiOfs = -1;
        for (IntType ki = 0; ki < kw; ki++) {
            IntType jjStart = GetOwStart(ki, padL);
            IntType jjEnd = GetOwEnd(urW, ki, padR);
            IntType weiRegOfs = REG_INDEX30;
            for (IntType ic = 0; ic < icBlock; ic++) {
                IntType weiCount = 0;
                for (IntType ii = 0; ii < nbOcBlock; ii++) {
                    IntType regIdx = weiRegOfs + ii;
                    if (regIdx >= REG_INDEX32) {
                        break;
                    }
                    SizeType auxKernelOffset = jcp.typeSizeIn
                            * (ii * jcp.nbIc * jcp.kh * jcp.kw * jcp.kd
                                            * icBlock * ocBlock
                                    + ki * icBlock * ocBlock + ic * ocBlock);
                    weiCount++;
                    if (jjEnd - jjStart > 0) {
                        prevWeiOfs = dtp == TypeT::F32 ? WeiLoadVC(auxKernelOffset,
                            weiRegOfs + (ii % KERNEL_C_LEN2),
                            prevWeiOfs) : weiLoadF16OrBF16(auxKernelOffset, weiRegOfs + (ii % KERNEL_C_LEN2),
                            prevWeiOfs);
                    }
                }

                IntType tmpKernelClen = dtp == TypeT::F32 ? KERNEL_C_LEN4 : KERNEL_C_LEN8;
                if (ic % tmpKernelClen == 0) {
                    for (IntType jj = jjStart; jj < jjEnd; jj++) {
                        SizeType auxInputOffset
                            = GetInputOffset(ki, ic, jj, padL);
                        prevBcastOfs = bcastLoad(jj,
                            auxInputOffset, prevBcastOfs);
                    }
                }

                for (IntType ii = 0; ii < nbOcBlock; ii++) {
                    SizeType auxKernelOffset = jcp.typeSizeIn
                            * ((ii + weiCount) * jcp.nbIc * jcp.kh * jcp.kw
                                            * jcp.kd * icBlock * ocBlock
                                    + ki * icBlock * ocBlock + ic * ocBlock);

                    for (IntType jj = jjStart; jj < jjEnd; jj++) {
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
                        prevWeiOfs = dtp == TypeT::F32 ? WeiLoadVC(auxKernelOffset,
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
}

template <CPUISAT isa, TypeT dtp>
void KdnnJitConvFwdKernel<isa, dtp>::ComputeLoopFmaCore(
    IntType urW, IntType padL, IntType padR)
{
    IntType kw = jcp.kw;
    IntType icBlock = jcp.icBlock;
    IntType ocBlock = jcp.ocBlock;
    IntType nbOcBlock = jcp.nbOcBlocking;
    const bool isSourceLayoutNxc = IsSrcLayoutNxc();

    Label khLabel;

    std::vector<Label> icTailJmp(kw);
    SizeType kernelShift = jcp.typeSizeIn * kw * ocBlock * icBlock;
    SizeType inpMul = isSourceLayoutNxc ? jcp.ngroups * jcp.ic
                                   : (!jcp.is1stConv ? icBlock : 1);

    SizeType inputShift = jcp.typeSizeIn * (jcp.dilateH + 1) * jcp.iw * inpMul;

    auto zregInp = [this](IntType iIc, IntType nbXBlocking) {
        return ZReg(iIc + nbXBlocking * jcp.urW);
    };
    auto zregInpS = [this](IntType iIc, IntType nbXBlocking) {
        return ZRegS(iIc + nbXBlocking * jcp.urW);
    };
    auto zregInpH = [this](IntType iIc, IntType nbXBlocking) {
        return ZRegH(iIc + nbXBlocking * jcp.urW);
    };
    auto zregOutS = [this](IntType iUr, IntType iOc) {
        return ZRegS(iUr + iOc * jcp.urW);
    };
    auto zregOutH = [this](IntType iUr, IntType iOc) {
        return ZRegH(iUr + iOc * jcp.urW);
    };

    auto bcastLoadF32 = [this, zregInpS](IntType jj, IntType tNbOcBlock, SizeType auxInputOffset,
                              IntType prevOfs) {
#ifdef KUDNNL_920B
        auto bcastLoadVReg = [this](IntType subjj, IntType subTNbOcBlock,
            IntType subAuxInputOffset, IntType subPrevOfs) {
            SizeType ofs;
            if ((subPrevOfs != -1) && ((subAuxInputOffset - subPrevOfs) > 0)) {
                ofs = subAuxInputOffset - subPrevOfs;
                AddImm(regPrevBcastAddr, regPrevBcastAddr, ofs, regTmpImm);
            } else {
                ofs = subAuxInputOffset;
                AddImm(regPrevBcastAddr, auxRegInp, ofs, regTmpImm);
            }
            Ld1r(VReg4SList(VReg(subjj + subTNbOcBlock * jcp.urW).s), Ptr(regPrevBcastAddr));
            subPrevOfs = subAuxInputOffset;
            return subPrevOfs;
        };
        if (jcp.icBlock == 4 && jcp.ocBlock == 4) {
            return bcastLoadVReg(jj, tNbOcBlock, auxInputOffset, prevOfs);
        }
#endif
        if (KdnnLd1rwImmCheck(auxInputOffset)) {
            Ld1rw(zregInpS(jj, tNbOcBlock), pAllOne,
                AdrScImm(Ptr(auxRegInp, static_cast<int32_t>(auxInputOffset))));
        } else if (KdnnLd1rwImmCheck(auxInputOffset - 0x100)) {
            Ld1rw(zregInpS(jj, tNbOcBlock), pAllOne,
                AdrScImm(Ptr(auxRegInp2,
                    static_cast<int32_t>(auxInputOffset - 0x100))));
        } else if (KdnnLd1rwImmCheck(auxInputOffset - 0x200)) {
            Ld1rw(zregInpS(jj, tNbOcBlock), pAllOne,
                AdrScImm(Ptr(auxRegInp3,
                    static_cast<int32_t>(auxInputOffset - 0x200))));
        } else {
            if ((prevOfs != -1) &&
                KdnnLd1rwImmCheck(auxInputOffset - prevOfs)) {
                Ld1rw(zregInpS(jj, tNbOcBlock), pAllOne,
                    AdrScImm(Ptr(regPrevBcastAddr,
                        static_cast<int32_t>(
                            auxInputOffset - prevOfs))));
            } else {
                SizeType ofs;
                if ((prevOfs != -1) && ((auxInputOffset - prevOfs) > 0)) {
                    ofs = auxInputOffset - prevOfs;
                    AddImm(regPrevBcastAddr, regPrevBcastAddr, ofs, regTmpImm);
                } else {
                    ofs = auxInputOffset;
                    AddImm(regPrevBcastAddr, auxRegInp, ofs, regTmpImm);
                }
                Ld1rw(zregInpS(jj, tNbOcBlock), pAllOne,
                    Ptr(regPrevBcastAddr));
                prevOfs = static_cast<IntType>(auxInputOffset);
            }
        }
        return prevOfs;
    };

    auto bcastLoad = [this, zregInpH, zregInp](IntType jj, IntType tNbOcBlock, SizeType auxInputOffset,
                              IntType prevOfs) {
        SizeType ofs;
        if ((prevOfs != -1) && ((auxInputOffset - prevOfs) > 0)) {
            ofs = auxInputOffset - prevOfs;
            AddImm(regPrevBcastAddr, regPrevBcastAddr, ofs, regTmpImm);
        } else {
            ofs = auxInputOffset;
            AddImm(regPrevBcastAddr, auxRegInp, ofs, regTmpImm);
        }
        dtp == TypeT::F16 ? Ld1rh(zregInpH(jj, tNbOcBlock), pAllOne,
            Ptr(regPrevBcastAddr)) : Ld1rh(zregInp(jj, tNbOcBlock).s, pAllOne, Ptr(regPrevBcastAddr));
        prevOfs = static_cast<IntType>(auxInputOffset);
        return prevOfs;
    };

    auto weiLoadF32 = [this](SizeType auxKernelOffset, IntType regIdx, IntType prevOfs) {
#ifdef KUDNNL_920B
        auto weiLoadVreg = [this](SizeType subAuxKernelOffset, IntType subRegIdx, IntType subPrevOfs) {
            SizeType ofs;
            if ((subPrevOfs != -1) && (subAuxKernelOffset - subPrevOfs > 0)) {
                ofs = subAuxKernelOffset - subPrevOfs;
                AddImm(regPrevWeiAddr, regPrevWeiAddr, ofs, regTmpImm);
            } else {
                ofs = subAuxKernelOffset;
                AddImm(regPrevWeiAddr, auxRegKer, ofs, regTmpImm);
            }
            Ld1(VReg4SList(VReg(subRegIdx).s), Ptr(regPrevWeiAddr));
            subPrevOfs = subAuxKernelOffset;
            return subPrevOfs;
        };
        if (jcp.icBlock == 4 && jcp.ocBlock == 4) {
            return weiLoadVreg(auxKernelOffset, regIdx, prevOfs);
        }
#endif
        return WeiLoadVC(auxKernelOffset, regIdx, prevOfs);
    };

    auto weiLoadF16OrF16 = [this](SizeType auxKernelOffset, IntType regIdx, IntType prevOfs) {
        SizeType ofs;
        if ((prevOfs != -1) && (auxKernelOffset - prevOfs > 0)) {
            ofs = auxKernelOffset - prevOfs;
            AddImm(regPrevWeiAddr, regPrevWeiAddr, ofs,
                    regTmpImm);
        } else {
            ofs = static_cast<IntType>(auxKernelOffset);
            AddImm(regPrevWeiAddr, auxRegKer, ofs, regTmpImm);
        }
        dtp == TypeT::F16 ? Ld1h(ZRegH(regIdx), pAllOne / PredType::T_Z,
            Ptr(regPrevWeiAddr)) : Ld1h(ZRegS(regIdx), pAllOne / PredType::T_Z, Ptr(regPrevWeiAddr));
        prevOfs = static_cast<IntType>(auxKernelOffset);
        return prevOfs;
    };

    Align(REG_INDEX32);
    L(khLabel);
    {
        IntType prevBcastOfs = -1;
        IntType prevWeiOfs = -1;
        for (IntType ki = 0; ki < kw; ki++) {
            IntType jjStart = GetOwStart(ki, padL);
            IntType jjEnd = GetOwEnd(urW, ki, padR);

            IntType weiRegOfs = nbOcBlock * jcp.urW;
            weiRegOfs += urW >= KERNEL_UR_W_LEN16 ? KERNEL_UR_W_LEN1 : jjEnd;
            IntType numRegs4wei = REG_INDEX32 - weiRegOfs;
            for (IntType ic = 0; ic < icBlock; ic++) {
                IntType weiCount = 0;
                for (IntType nb = 0; nb < nbOcBlock; nb++) {
                    IntType regIdx = weiRegOfs + nb;
                    if (regIdx >= REG_INDEX32) {
                        break;
                    }
                    SizeType auxKernelOffset = jcp.typeSizeIn
                            * (nb * jcp.nbIc * jcp.kh * jcp.kw * jcp.kd
                                            * icBlock * ocBlock
                                    + ki * icBlock * ocBlock + ic * ocBlock);
                    weiCount++;
                    if (jjEnd - jjStart > 0) {
                        prevWeiOfs = dtp == TypeT::F32 ? weiLoadF32(auxKernelOffset,
                            weiRegOfs + (nb % numRegs4wei), prevWeiOfs) :
                            weiLoadF16OrF16(auxKernelOffset, weiRegOfs + (nb % numRegs4wei), prevWeiOfs);
                    }
                }
                for (IntType jj = jjStart; jj < jjEnd; jj++) {
                    SizeType auxInputOffset = GetInputOffset(ki, ic, jj, padL);
                    prevBcastOfs = dtp == TypeT::F32 ? bcastLoadF32(jj, nbOcBlock,
                        auxInputOffset, prevBcastOfs) : bcastLoad(jj, nbOcBlock,
                        auxInputOffset, prevBcastOfs);
                }

                for (IntType nb = 0; nb < nbOcBlock; nb++) {
                    SizeType auxKernelOffset = jcp.typeSizeIn
                            * ((nb + weiCount) * jcp.nbIc * jcp.kh * jcp.kw * jcp.kd
                                            * icBlock * ocBlock
                                    + ki * icBlock * ocBlock + ic * ocBlock);
                    for (IntType jj = jjStart; jj < jjEnd; jj++) {
                        if (jcp.kernelKind == KdnnConvKernelKindT::KUDNN_EXPL_BCAST) {
                            if (dtp == TypeT::F32) {
                                Fmla(zregOutS(jj, nb), pAllOne, zregInpS(jj, nbOcBlock),
                                    ZRegS(weiRegOfs + (nb % numRegs4wei)));
                            } else if (dtp == TypeT::F16) {
                                Fmla(zregOutH(jj, nb), pAllOne, zregInpH(jj, nbOcBlock),
                                    ZRegH(weiRegOfs + (nb % numRegs4wei)));
                            } else if (dtp == TypeT::BF16) {
                                Bfmlalb(zregOutS(jj, nb),
                                    zregInp(jj, nbOcBlock).h,
                                    ZRegH(weiRegOfs + (nb % numRegs4wei)));
                            }
                        } else if (jcp.kernelKind == KdnnConvKernelKindT::KUDNN_EMBD_BCAST) {
                            Fmla(VReg(jj + nb * jcp.urW).s, VReg(jj + nbOcBlock * jcp.urW).s,
                                VReg(weiRegOfs + (nb % numRegs4wei)).s);
                        }
                    }
                    if ((jjEnd - jjStart > 0)
                           && ((nb + weiCount) < nbOcBlock)) {
                        prevWeiOfs = dtp == TypeT::F32 ? weiLoadF32(auxKernelOffset,
                            weiRegOfs + ((nb + weiCount) % numRegs4wei), prevWeiOfs) :
                            weiLoadF16OrF16(auxKernelOffset, weiRegOfs + ((nb + weiCount) % numRegs4wei), prevWeiOfs);
                    }
                }
            }
            L(icTailJmp[ki]);
        }

        AddImm(auxRegKer, auxRegKer, kernelShift, regTmpImm);
        AddImm(auxRegInp, auxRegInp, inputShift, regTmpImm);
        AddImm(auxRegInp2, auxRegInp, 0x100, regTmpImm);
        AddImm(auxRegInp3, auxRegInp2, 0x100, regTmpImm);
        Sub(regKj, regKj, 1);
        Cmp(regKj, 0);
        B(Cond::GT, khLabel);
    }
}

template <CPUISAT isa, TypeT dtp>
void KdnnJitConvFwdKernel<isa, dtp>::DealNoiIsZero(IntType oShift, IntType urw, IntType leftPad,
    IntType rightPad, IntType rightPad1, SizeType inShiftPad, IntType sw, IntType urwTail)
{
    AddImm(regOutPrf, regOutPrf, oShift, regTmpImm);
    ComputeLoop(urw, leftPad, rightPad1);
    AddImm(regOut, regOut, oShift, regTmpImm);
    if (urwTail != 0) {
        AddImm(regOutPrf, regOutPrf, oShift, regTmpImm);
        IntType lPad1 = leftPad - urw * sw;
        if (lPad1 > 0) {
            ComputeLoop(urwTail, lPad1, rightPad);
        } else {
            AddImm(regInp, regInp, inShiftPad, regTmpImm);
            ComputeLoop(urwTail, 0, rightPad);
        }
    }
}
 
template <CPUISAT isa, TypeT dtp>
void KdnnJitConvFwdKernel<isa, dtp>::DealLpadLessThanUrw(IntType oShift, SizeType iShift, IntType urw,
    IntType leftPad, IntType rightPad, IntType rightPad1, IntType inShiftPad,
    IntType urwTail, IntType noi)
{
    Mov(regOi, 0);
    if (leftPad > 0) {
        AddImm(regOutPrf, regOutPrf, oShift, regTmpImm);
        ComputeLoop(urw, leftPad, 0);
        AddImm(regInp, regInp, inShiftPad, regTmpImm);
        AddImm(regOut, regOut, oShift, regTmpImm);
        AddImm(regOi, regOi, 1, regTmpImm);
    }
    if ((leftPad <= 0 && noi > 0) || (leftPad > 0 && noi > 1)) {
        Label owLoopLabel;
        L(owLoopLabel);
        {
            AddImm(regOutPrf, regOutPrf, oShift,
                    regTmpImm);
            ComputeLoop(urw, 0, 0);
 
            AddImm(regInp, regInp, iShift, regTmpImm);
            AddImm(regOut, regOut, oShift, regTmpImm);
            AddImm(regOi, regOi, 1, regTmpImm);
            CmpImm(regOi, noi, regTmpImm);
 
            B(Cond::LT, owLoopLabel);
        }
    }
    if (rightPad1 > 0) {
        AddImm(regOutPrf, regOutPrf, oShift, regTmpImm);
        ComputeLoop(urw, 0, rightPad1);
        AddImm(regInp, regInp, iShift, regTmpImm);
        AddImm(regOut, regOut, oShift, regTmpImm);
    }
    if (urwTail != 0) {
        AddImm(regOutPrf, regOutPrf, oShift, regTmpImm);
        ComputeLoop(urwTail, 0, rightPad);
    }
}
 
template <CPUISAT isa, TypeT dtp>
void KdnnJitConvFwdKernel<isa, dtp>::DealGeneralLoop(IntType oShift, IntType inMult, IntType urw, IntType leftPad,
    IntType rightPad, IntType rightPad1, IntType urwTail, IntType sw, IntType iw, IntType kw, IntType& noi)
{
    noi = rightPad1 > 0 ? noi : noi - 1;
    SizeType inpShiftPadl = 0;
    SizeType preInpShiftPadl = 0;
    for (IntType oi = 1; oi <= noi + 1; oi++) {
        AddImm(regOutPrf, regOutPrf, oShift, regTmpImm);
        IntType lPadL = leftPad - (oi - 1) * urw * sw;
        IntType lPadR = lPadL > 0 ? lPadL : 0;
        IntType rPadL = CalculateEndPadding(leftPad, oi * urw, iw, sw,
            CalculateExtendedFilterSize(kw, jcp.dilateW));
        IntType rPadR = rPadL > 0 ? rPadL : 0;
        ComputeLoop(urw, lPadR, rPadR);
        if (leftPad - oi * urw * sw < 0) {
            inpShiftPadl = jcp.typeSizeIn * (oi * urw * sw - leftPad) * inMult;
            AddImm(regInp, regInp, inpShiftPadl - preInpShiftPadl, regTmpImm);
            preInpShiftPadl = inpShiftPadl;
        }
        AddImm(regOut, regOut, oShift, regTmpImm);
        AddImm(regOi, regOi, 1, regTmpImm);
    }
    if (urwTail != 0) {
        AddImm(regOutPrf, regOutPrf, oShift, regTmpImm);
        IntType lPadM = leftPad - (noi + 1) * urw * sw;
        IntType lPadN = lPadM > 0 ? lPadM : 0;
        ComputeLoop(urwTail, lPadN, rightPad);
    }
}

template <CPUISAT isa, TypeT dtp>
void KdnnJitConvFwdKernel<isa, dtp>::Generate()
{
    IntType iw = jcp.iw;
    IntType ow = jcp.ow;
    IntType kw = jcp.kw;
    IntType lPad = jcp.lPad;
    IntType urW = jcp.urW;
    IntType urWTail = jcp.urWTail;
    IntType strideW = jcp.strideW;

    IntType inpMult = IsSrcLayoutNxc() ? jcp.ngroups * jcp.ic
                                   : (jcp.is1stConv ? 1 : jcp.icBlock);
    SizeType inpShiftPad = jcp.typeSizeIn * (urW * strideW - lPad) * inpMult;
    SizeType inpShift = jcp.typeSizeIn * urW * strideW * inpMult;
    IntType outShift = jcp.typeSizeOut * urW
            * (IsDstLayoutNxc() ? jcp.ngroups * jcp.oc : jcp.ocBlock);

    Preamble();

    Ldr(regInp, Ptr(ABI_PARAM1, static_cast<uint32_t>(
        offsetof(KdnnConvJitCallS, src))));
    Ldr(regOut, Ptr(ABI_PARAM1, static_cast<uint32_t>(
        offsetof(KdnnConvJitCallS, dst))));
    Ldr(regKer, Ptr(ABI_PARAM1, static_cast<uint32_t>(
        offsetof(KdnnConvJitCallS, filt))));
    Ldr(regKh, Ptr(ABI_PARAM1, static_cast<uint32_t>(
        offsetof(KdnnConvJitCallS, khPadding))));
    if (jcp.ndims == KERNEL_DIMS_5) {
        Mov(auxRegKerOrg, regKer);
    }

    IntType rPad = jcp.rPad > 0 ? jcp.rPad : 0;
    IntType nOi = ow / urW;
    IntType rPad1 = CalculateEndPadding(lPad, urW * nOi, iw, strideW,
        CalculateExtendedFilterSize(kw, jcp.dilateW));
    IntType nOiLast2 = nOi - 1 > 0 ? nOi - 1 : 0;
    IntType rPadLast2 = CalculateEndPadding(lPad, urW * nOiLast2, iw, strideW,
        CalculateExtendedFilterSize(kw, jcp.dilateW));
    // ow is being processed as a whole - with left and right paddings
    // nOi is # of output width blocks
    if (rPad1 > 0) {
        nOi--;
    }

    if (ow == urW) {
        Ldr(regOutPrf, Ptr(ABI_PARAM1, static_cast<uint32_t>(
            offsetof(KdnnConvJitCallS, dstPrf))));
        ComputeLoop(urW, lPad, rPad);
    } else {
        Mov(regOutPrf, regOut);
        if (nOi == 0) {
            DealNoiIsZero(outShift, urW, lPad, rPad, rPad1, inpShiftPad, strideW, urWTail);
        } else if (lPad <= urW && rPadLast2 <= 0) {
            // generate loop for simplify when lpad <= ur_w
            DealLpadLessThanUrw(outShift, inpShift, urW, lPad, rPad, rPad1,
                inpShiftPad, urWTail, nOi);
        } else {
            DealGeneralLoop(outShift, inpMult, urW, lPad,
                rPad, rPad1, urWTail, strideW, iw, kw, nOi);
        }
    }
    Postamble();
}

/* struct instantiation */
#ifdef KUDNNL_920F
template struct KdnnJitConvFwdKernel<CPUISAT::VC_512, Element::TypeT::F32>;
template struct KdnnJitConvFwdKernel<CPUISAT::VC_512, Element::TypeT::F16>;
template struct KdnnJitConvFwdKernel<CPUISAT::VC_512, Element::TypeT::BF16>;
#else
template struct KdnnJitConvFwdKernel<CPUISAT::VC_256, Element::TypeT::F32>;
template struct KdnnJitConvFwdKernel<CPUISAT::VC_256, Element::TypeT::F16>;
template struct KdnnJitConvFwdKernel<CPUISAT::VC_256, Element::TypeT::BF16>;
#endif

} // namespace KuDNN
