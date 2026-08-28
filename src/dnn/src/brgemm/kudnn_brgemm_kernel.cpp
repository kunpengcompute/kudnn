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

#include "kudnn_brgemm_kernel.hpp"

#include <algorithm>
#include <cstdint>

#include "cg_jit_generator.hpp"
#include "internal_funcs.hpp"
#include "service/kudnn_exception.hpp"
#include "types/kudnn_data_type.hpp"

#include "kudnn_brgemm_types.hpp"

namespace KuDNN {
using namespace ARMCG;

namespace {

constexpr uint64_t ARMCG_BITMASK_12 = ~static_cast<uint64_t>(0xfff);

} // namespace

void KdnnBrgemmKernelT::Cvt2ps(Element::TypeT typeIn, const ZReg zmmIn, const XReg &addr, bool maskFlag,
    bool store, PReg kTailMask, const IntType offset, const IntType baseOffset)
{
    const PReg mask = maskFlag ? kTailMask : pAllOne;
    const IntType len = (cpuVCLen / VAL_4) * VAL_4;
    const IntType mul = (offset - baseOffset) / len;
    const IntType mod = (offset - baseOffset) % len;
    switch (typeIn) {
        case Element::TypeT::F32:
            if (mul >= VC_MULTILDR_IMM_RANGE_LOW && mul <= VC_MULTILDR_IMM_RANGE_HIGH && mod == 0) {
                Ld1w(ZTmp1().s, mask / PredType::T_Z, Ptr(addr, static_cast<int>(mul), ExtMod::MUL_VL));
            } else {
                Ld1w(ZTmp1().s, mask / PredType::T_Z, Ptr(AddrOff(addr, offset - baseOffset, xDefaultAddr, X_TMP_0)));
            }
            if (store) {
                Mov(zmmIn.s, kTailMask / PredType::T_M, ZTmp1().s);
            }
            break;
        default: throw Service::Unsupported(""); break;
    }
}

void KdnnBrgemmKernelT::LdbRegsShift(IntType ldBlock2, bool isTail)
{
    IntType BOffset = isTail ? LdbBOffset(1, true) : LdbBOffset(ldBlock2);
    IntType COffset = isTail ? LdbCOffset(1, true) : LdbCOffset(ldBlock2);
    IntType DOffset = isTail ? LdbDOffset(1, true) : LdbDOffset(ldBlock2);
    AddImm(regBOffset, regBOffset, BOffset, X_TMP_0);
    AddImm(regAuxC, regAuxC, COffset, X_TMP_0);
    AddImm(regAuxD, regAuxD, DOffset, X_TMP_0);
    if (brg.withBias) {
        if ((regAuxBiasOffs & ARMCG_BITMASK_12) == 0) {
            Ldr(regAuxBias, Ptr(xSP, regAuxBiasOffs));
            AddImm(regAuxBias, regAuxBias, isTail ? BiasOffset(1, true) : BiasOffset(ldBlock2), X_TMP_0);
            Str(regAuxBias, Ptr(xSP, regAuxBiasOffs));
        } else {
            AddImm(xDefaultAddr, xSP, regAuxBiasOffs, X_TMP_0);
            Ldr(regAuxBias, Ptr(xDefaultAddr));
            AddImm(regAuxBias, regAuxBias, isTail ? BiasOffset(1, true) : BiasOffset(ldBlock2), X_TMP_0);
            Str(regAuxBias, Ptr(xDefaultAddr));
        }
    }
}

void KdnnBrgemmKernelT::CopyPostOpsStackValuesToAux(bool isRegTail)
{
    if (!isRegTail) {
        Mov(regAuxC, regC);
        Mov(regAuxD, regD);
        Eor(regBOffset, regBOffset, regBOffset);
        if (brg.withBias) {
            if ((regBiasOffs & ARMCG_BITMASK_12) == 0) {
                Ldr(regBias, Ptr(xSP, regBiasOffs));
            } else {
                AddImm(xDefaultAddr, xSP, regBiasOffs, X_TMP_0);
                Ldr(regBias, Ptr(xDefaultAddr));
            }
            if ((regAuxBiasOffs & ARMCG_BITMASK_12) == 0) {
                Str(regBias, Ptr(xSP, regAuxBiasOffs));
            } else {
                AddImm(xDefaultAddr, xSP, regAuxBiasOffs, X_TMP_0);
                Str(regBias, Ptr(xDefaultAddr));
            }
        }
    }
}

void KdnnBrgemmKernelT::AdvanceBdBlock2PostOpRegs(IntType bdBlock2)
{
    Service::Unused(bdBlock2);
}

void KdnnBrgemmKernelT::ReadParams()
{
    Label labelDone;
    if (brg.type == BrgemmBatchKindT::ADDRESS) {
        uint32_t batchOffset = offsetof(BrgemmKernelParamsT, batch);
        if ((batchOffset & ARMCG_BITMASK_12) == 0) {
            Ldr(regAddrBatch, Ptr(param1, batchOffset));
        } else {
            AddImm(xDefaultAddr, param1, batchOffset, X_TMP_0);
            Ldr(regAddrBatch, Ptr(xDefaultAddr));
        }
    }
    uint32_t ptrCOffset = offsetof(BrgemmKernelParamsT, ptrC);
    uint32_t ptrDOffset = offsetof(BrgemmKernelParamsT, ptrD);
    uint32_t bsOffset = offsetof(BrgemmKernelParamsT, bs);
    Ldr(regC, Ptr(param1, ptrCOffset));
    Ldr(regD, Ptr(param1, ptrDOffset));
    Ldr(regBs, Ptr(param1, bsOffset));
    if (brg.withBias) {
        uint32_t ptrBiasOffset = offsetof(BrgemmKernelParamsT, ptrBias);
        Ldr(regBias, Ptr(param1, ptrBiasOffset));
        Str(regBias, Ptr(xSP, regBiasOffs));
    }
    uint32_t doPostOpsOffset = offsetof(BrgemmKernelParamsT, doPostOps);
    Ldr(regDoPostOps, Ptr(param1, doPostOpsOffset));
    Str(regDoPostOps, Ptr(xSP, regDoPostOpsOffs));
    uint32_t skipAccmOffset = offsetof(BrgemmKernelParamsT, skipAccm);
    Ldr(regSkipAccm, Ptr(param1, skipAccmOffset));
    Str(regSkipAccm, Ptr(xSP, regSkipAccmOffs));
    uint32_t zpAValOffset = offsetof(BrgemmKernelParamsT, zpAVal);
    Ldr(regZpAVal, Ptr(param1, zpAValOffset));
    Str(regZpAVal, Ptr(xSP, regZpAValOffs));
    uint32_t doApplyCompOffset = offsetof(BrgemmKernelParamsT, doApplyComp);
    Ldr(regDoComp, Ptr(param1, doApplyCompOffset));
    Str(regDoComp, Ptr(xSP, regDoCompOffs));
}

void KdnnBrgemmKernelT::ZeroAccumulators(IntType bdBlock2, bool isBdbTail, IntType ldBlock2, bool isLdTail,
    bool skipAccumulation)
{
    Service::Unused(bdBlock2);
    Service::Unused(skipAccumulation);

    IntType bdBlock = isBdbTail ? brg.bdbTail : brg.bdBlock;
    const bool needToApplyBeta = brg.beta != 0.0f;
    if constexpr (BRGEMM_MC_AVAILABLE) {
#ifdef KUDNNL_920F
        Mov(X_TMP_2, x15);
        if (needToApplyBeta) {
            for (IntType bd = 0; bd < bdBlock; bd++) {
                for (IntType ld = 0; ld < ldBlock2; ld++) {
                    if (brg.beta != 1.0f) {
                        throw Service::Unsupported("");
                    }
                    if (ld == 0 && bd % VAL_4 == 0) {
                        Mov(x15, static_cast<uint64_t>(bd));
                    }
                    const bool isTail = isLdTail && ld + 1 == ldBlock2;
                    const auto kMask = isTail ? ldTailMask : ldFullMask;
                    const IntType offset = COffset(bd, ld);
                    Mov(regTmp, static_cast<uint64_t>(offset / brg.typeSizeC));
                    Ld1w(ZARegVX(ZARegIndex(ld, brg.dtA)).s, w15, static_cast<uint32_t>(bd % VAL_4),
                        kMask / PredType::T_Z, AdrScSc(regAuxC, regTmp, ShMod::LSL, static_cast<uint32_t>(VAL_2)));
                }
            }
        } else {
            Zero({za0, za1, za2, za3});
        }
        Mov(x15, X_TMP_2);
#endif
    } else {
        for (IntType bd = 0; bd < bdBlock; bd++) {
            for (IntType ld = 0; ld < ldBlock2; ld++) {
                auto zmm = Accm(ldBlock2, bd, ld);
                // avoid fadd instruction when k is blocked
                if (needToApplyBeta) {
                    const bool isTail = isLdTail && ld + 1 == ldBlock2;
                    const auto kMask = isTail ? ldTailMask : ldFullMask;
                    const IntType offset = COffset(bd, ld);
                    IntType baseOffset = 0;
                    auto xAddr = regAuxC;
                    if (static_cast<uint64_t>(offset - baseOffset) > cpuVCLen * VC_MULTILDR_IMM_RANGE_HIGH) {
                        AddImm(regTmp, regAuxC, offset, X_TMP_0);
                        baseOffset = offset;
                        xAddr = regTmp;
                    }
                    const IntType len = (cpuVCLen / VAL_4) * VAL_4;
                    const IntType mul = (offset - baseOffset) / len;
                    const IntType mod = (offset - baseOffset) % len;
                    if (mul >= VC_MULTILDR_IMM_RANGE_LOW && mul <= VC_MULTILDR_IMM_RANGE_HIGH && mod == 0) {
                        Ld1w(zmm.s, kMask / PredType::T_Z,
                            Ptr(xAddr, static_cast<int>(mul), ExtMod::MUL_VL));
                    } else {
                        Ld1w(zmm.s, kMask / PredType::T_Z,
                            Ptr(AddrOff(xAddr, offset - baseOffset, xDefaultAddr, X_TMP_0)));
                    }
                    const bool needInitBetaVmm = brg.beta != 1.0f;
                    auto vmmBeta = ZTailMask();
                    if (needInitBetaVmm) {
                        auto wregTmp = WReg(regTmpGpr.GetIdx());
                        uint32_t *pBrgBeta = (decltype(pBrgBeta))(&brg.beta);
                        MovImm(wregTmp, *pBrgBeta);
                        Dup(vmmBeta.s, wregTmp);
                        Fmul(zmm.s, zmm.s, vmmBeta.s);
                    }
                } else {
                    Eor(zmm.d, zmm.d, zmm.d);
                }
            }
        }
    }
}

void KdnnBrgemmKernelT::StoreAccumulators(IntType bdBlock2, bool isBdbTail, IntType ldBlock2, bool isLdTail,
    bool skipAccumulation)
{
    Service::Unused(bdBlock2);
    Service::Unused(skipAccumulation);

    const bool hasZeroPoints = brg.zpTypeA != BrgemmBroadcastT::UNDEFINED
        || brg.zpTypeB != BrgemmBroadcastT::UNDEFINED || brg.zpTypeC != BrgemmBroadcastT::UNDEFINED;
    const bool arePostOpsApplicable = brg.withEltwise || brg.withBinary || brg.withScales || brg.withBias
        || brg.withSum || brg.dtC != brg.dtD || brg.reqS8S8Compensation || hasZeroPoints;
    const bool needToApplyAlphaBeta = brg.alpha != 1.0f || brg.beta != 0.0f;
    IntType bdBlock = isBdbTail ? brg.bdbTail : brg.bdBlock;
    if (needToApplyAlphaBeta) {
        ApplyAlphaBeta(bdBlock, ldBlock2, isLdTail);
    }
    Label labelDone;
    if (arePostOpsApplicable) {
        Label labelStoreWithoutPostOps;
        if ((regDoPostOpsOffs & ARMCG_BITMASK_12) == 0) {
            Ldr(regDoPostOps, Ptr(xSP, regDoPostOpsOffs));
        } else {
            AddImm(xDefaultAddr, xSP, regDoPostOpsOffs, X_TMP_0);
            Ldr(regDoPostOps, Ptr(xDefaultAddr));
        }
        CmpImm(regDoPostOps, 0, X_TMP_0);
        B(Cond::EQ, labelStoreWithoutPostOps);
        StoreAccumulatorsApplyPostOps(bdBlock, ldBlock2, 0, isLdTail);
        Bl(labelDone);
        LAligned(labelStoreWithoutPostOps);
    }
    StoreAccumulatorsWithoutPostOps(bdBlock, ldBlock2, isLdTail);
    LAligned(labelDone);
}

void KdnnBrgemmKernelT::StoreAccumulatorsWithoutPostOps(IntType bdBlock, IntType ldBlock2, bool isLdTail)
{
    auto xAddr = regAuxC;
    IntType baseOffset = 0;
    if constexpr (BRGEMM_MC_AVAILABLE) {
#ifdef KUDNNL_920F
        Mov(X_TMP_2, x15);
        for (IntType bd = 0; bd < bdBlock; bd++) {
            for (IntType ld = 0; ld < ldBlock2; ld++) {
                if (ld == 0 && bd % VAL_4 == 0) {
                    Mov(x15, static_cast<uint64_t>(bd));
                }
                const IntType offset = COffset(bd, ld);
                Mov(regTmp, static_cast<uint64_t>(offset / brg.typeSizeC));
                St1w(ZARegVX(ZARegIndex(ld, brg.dtA)).s, w15, static_cast<uint32_t>(bd % VAL_4),
                    !isLdTail ? ldFullMask : ldTailMask,
                    AdrScSc(regAuxC, regTmp, ShMod::LSL, static_cast<uint32_t>(VAL_2)));
            }
        }
        Mov(x15, X_TMP_2);
#endif
    } else {
        for (IntType bd = 0; bd < bdBlock; bd++) {
            for (IntType ld = 0; ld < ldBlock2; ld++) {
                const IntType offset = COffset(bd, ld);
                if (static_cast<uint64_t>(offset - baseOffset) > cpuVCLen * VC_MULTILDR_IMM_RANGE_HIGH) {
                    AddImm(regTmp, regAuxC, offset, X_TMP_0);
                    baseOffset = offset;
                    xAddr = regTmp;
                }
                auto zmm = Accm(ldBlock2, bd, ld);
                const auto mask = isLdTail ? ldTailMask : pAllOne;
                const IntType len = (cpuVCLen / VAL_4) * VAL_4;
                const IntType mul = (offset - baseOffset) / len;
                const IntType mod = (offset - baseOffset) % len;
                if (mul >= VC_MULTILDR_IMM_RANGE_LOW && mul <= VC_MULTILDR_IMM_RANGE_HIGH && mod == 0) {
                    St1w(zmm.s, mask, Ptr(xAddr, mul, ExtMod::MUL_VL));
                } else {
                    St1w(zmm.s, mask, Ptr(AddrOff(xAddr, offset - baseOffset, xDefaultAddr, X_TMP_0)));
                }
            }
        }
    }
}

void KdnnBrgemmKernelT::StoreAccumulatorsApplyPostOps(IntType bdBlock, IntType ldBlock2, IntType ldbAndBdbOffset,
    bool isLdTail)
{
    Service::Unused(ldbAndBdbOffset);

    if (brg.withBias) {
        if ((regAuxBiasOffs & ARMCG_BITMASK_12) == 0) {
            Ldr(regAuxBias, Ptr(xSP, regAuxBiasOffs));
        } else {
            AddImm(xDefaultAddr, xSP, regAuxBiasOffs, X_TMP_0);
            Ldr(regAuxBias, Ptr(xDefaultAddr));
        }
    }
    auto kMask = !isLdTail ? ldFullMask : ldTailMask;
    auto xAddr = regAuxBias;
    IntType baseOffset = 0;
    for (IntType ld = 0; ld < ldBlock2; ld++) {
        auto zmmBias = ZTmp1();
        if (brg.withBias) {
            const IntType offset = BiasOffset(ld);
            if (static_cast<uint64_t>(offset - baseOffset) > cpuVCLen * VC_MULTILDR_IMM_RANGE_HIGH) {
                AddImm(regTmp, regAuxBias, offset, X_TMP_0);
                baseOffset = offset;
                xAddr = regTmp;
            }
            Cvt2ps(brg.dtBias, zmmBias, xAddr, true, false, kMask, offset, baseOffset);
        }
        if constexpr (BRGEMM_MC_AVAILABLE) {
#ifdef KUDNNL_920F
            Fdup(z8.s, 1.0);
            Fmopa(ZAReg(ZARegIndex(ld, brg.dtA)).s, kMask / PredType::T_M, pAllOne / PredType::T_M, zmmBias.s, z8.s);
#endif
        } else {
            for (IntType bd = 0; bd < bdBlock; bd++) {
                auto zmm = Accm(ldBlock2, bd, ld);
                if (brg.withBias) {
                    Fadd(zmm.s, zmm.s, zmmBias.s);
                }
            }
        }
    }
    // [unimplemented] postops
    xAddr = regAuxD;
    baseOffset = 0;
    if constexpr (BRGEMM_MC_AVAILABLE) {
#ifdef KUDNNL_920F
        Mov(X_TMP_2, x15);
        for (IntType bd = 0; bd < bdBlock; bd++) {
            for (IntType ld = 0; ld < ldBlock2; ld++) {
                if (ld == 0 && bd % VAL_4 == 0) {
                    Mov(x15, static_cast<uint64_t>(bd));
                }
                const IntType offset = DOffset(bd, ld);
                Mov(regTmp, static_cast<uint64_t>(offset / brg.typeSizeD));
                St1w(ZARegVX(ZARegIndex(ld, brg.dtA)).s, w15, static_cast<uint32_t>(bd % VAL_4), kMask,
                    AdrScSc(regAuxD, regTmp, ShMod::LSL, static_cast<uint32_t>(VAL_2)));
            }
        }
        Mov(x15, X_TMP_2);
#endif
    } else {
        for (IntType bd = 0; bd < bdBlock; bd++) {
            for (IntType ld = 0; ld < ldBlock2; ld++) {
                const IntType offset = DOffset(bd, ld);
                if (static_cast<uint64_t>(offset - baseOffset) > cpuVCLen * VC_MULTILDR_IMM_RANGE_HIGH) {
                    AddImm(regTmp, regAuxD, offset, X_TMP_0);
                    baseOffset = offset;
                    xAddr = regTmp;
                }
                auto zmm = Accm(ldBlock2, bd, ld);
                const IntType len = (cpuVCLen / VAL_4) * VAL_4;
                const IntType mul = (offset - baseOffset) / len;
                const IntType mod = (offset - baseOffset) % len;
                switch (brg.dtD) {
                    case Element::TypeT::F32:
                        if (mul >= VC_MULTILDR_IMM_RANGE_LOW && mul <= VC_MULTILDR_IMM_RANGE_HIGH && mod == 0) {
                            St1w(zmm.s, kMask, Ptr(xAddr, static_cast<int>(mul), ExtMod::MUL_VL));
                        } else {
                            St1w(zmm.s, kMask, Ptr(AddrOff(xAddr, offset - baseOffset, xDefaultAddr, X_TMP_0)));
                        }
                        break;
                    default: throw Service::Unsupported(""); break;
                }
            }
        }
    }
}

void KdnnBrgemmKernelT::ApplyAlphaBeta(IntType bdBlock, IntType ldBlock2, bool isLdTail)
{
    Service::Unused(bdBlock);
    Service::Unused(ldBlock2);
    Service::Unused(isLdTail);
}

void KdnnBrgemmKernelT::SetABMatrices()
{
    if (brg.type == BrgemmBatchKindT::ADDRESS) {
        if (brg.brgattr.maxBs > 1) {
            uint32_t ptraOffset = offsetof(BrgemmBatchElementT, ptr.a);
            uint32_t ptrbOffset = offsetof(BrgemmBatchElementT, ptr.b);
            Ldr(regAuxA, Ptr(regAux1Batch, brg.layout == BrgemmLayoutT::ROW_MAJOR ? ptraOffset : ptrbOffset));
            Ldr(regAuxB, Ptr(regAux1Batch, brg.layout == BrgemmLayoutT::ROW_MAJOR ? ptrbOffset : ptraOffset));
            AddImm(regAux1Batch, regAux1Batch, sizeof(BrgemmBatchElementT), X_TMP_0);
            Prfm(Prfop::PLDL1_KEEP, AdrUimm(Ptr(regAux1Batch)));
        }
    }
    Add(regAuxA, regAuxA, regAOffset);
    Prfm(Prfop::PLDL1_KEEP, AdrUimm(Ptr(regAuxA)));
    Add(regAuxB, regAuxB, regBOffset);
    Prfm(Prfop::PLDL1_KEEP, AdrUimm(Ptr(regAuxB)));
}

void KdnnBrgemmKernelT::RestoreABMatrices()
{
    bool restoreRegBatch = brg.brgattr.maxBs > 1 || vpadExist;
    if (brg.type == BrgemmBatchKindT::ADDRESS) {
        if (restoreRegBatch) {
            Mov(regAux1Batch, regAddrBatch);
        }
    }
}

void KdnnBrgemmKernelT::DotProduct(ZReg zd, ZReg zn, ZReg zm, IntType offset)
{
    Fmla(zd.s, zn.s, zm.s[offset]);
}

void KdnnBrgemmKernelT::GemmMicrokernelVC(IntType bdBlock2, bool isBdbTail, IntType ldBlock2, bool isRdTail,
    bool isLdTail, IntType vpad, IntType rowsForRdTail)
{
    Service::Unused(bdBlock2);

    IntType bdBlock = isBdbTail ? brg.bdbTail : brg.bdBlock;
    const IntType bdB = std::max(vpad, static_cast<IntType>(0));
    const IntType bdE = std::min(bdBlock, bdBlock + vpad);
    const bool isValidBd = needCompPads && vpad != 0 ? bdB <= bdE : bdB < bdE;
    if (!isValidBd) {
        return;
    }

    bool isEmdbd = brg.embdBcst;
    IntType rdLoop = isRdTail ? brg.rdbTail : brg.rdBlock;
    IntType rdTailSize = 0;
    if (isRdTail) {
        SetPreg(rdTailMask.s, brg.rdbTail, X_TMP_0, X_TMP_1);
    }
    auto broadcast = [this, isRdTail](const ZReg &zd, IntType offset, bool isTail, Element::TypeT dt) {
        if (isTail) {
            throw Service::LogicError("");
        }
        if (dt == Element::TypeT::F32) {
            if (VC_BCLDR_IMM_RANGE_LOW <= offset && offset <= VC_BCLDR_IMM_RANGE_HIGH
                && offset % VC_BCLDR_IMM_MODULUS == 0) {
                Ld1rqw(zd.s, (isRdTail ? rdTailMask : pAllOne) / PredType::T_Z,
                    Ptr(regAuxA, static_cast<int32_t>(offset)));
            } else {
                AddImm(xDefaultAddr, regAuxA, offset, X_TMP_0);
                Ld1rqw(zd.s, (isRdTail ? rdTailMask : pAllOne) / PredType::T_Z, Ptr(xDefaultAddr));
            }
        } else {
            throw Service::Unsupported("");
        }
    };
    auto load = [this, isLdTail](const ZReg &zd, XReg &addr, IntType offset, IntType &baseOffsetIn, Element::TypeT dt) {
        if (offset - baseOffsetIn > static_cast<IntType>(cpuVCLen) * VC_CONTILDR_IMM_RANGE_HIGH) {
            AddImm(regTmp, regAuxB, offset, X_TMP_0);
            baseOffsetIn = offset;
            addr = regTmp;
        }
        const IntType len = (cpuVCLen / VAL_4) * VAL_4;
        const IntType mul = (offset - baseOffsetIn) / len;
        const IntType mod = (offset - baseOffsetIn) % len;
        if (dt == Element::TypeT::F32) {
            if (VC_CONTILDR_IMM_RANGE_LOW <= mul && mul <= VC_CONTILDR_IMM_RANGE_HIGH && mod == 0) {
                Ld1w(zd.s, (isLdTail ? ldTailMask : pAllOne) / PredType::T_Z, Ptr(addr, mul, ExtMod::MUL_VL));
            } else {
                AddImm(xDefaultAddr, addr, offset - baseOffsetIn, X_TMP_0);
                Ld1w(zd.s, (isLdTail ? ldTailMask : pAllOne) / PredType::T_Z, Ptr(xDefaultAddr));
            }
        } else {
            throw Service::Unsupported("");
        }
    };
    const bool compVpad = vpad != 0 && (brg.reqS8S8Compensation || brg.zpTypeA != BrgemmBroadcastT::UNDEFINED);
    if (brg.reqCalCompPads || compVpad) {
        throw Service::Unsupported("");
    }
    bool maybeLoadBytes = (rowsForRdTail > 0 || brg.brgattr.waryAKTailRead) && isRdTail && rdTailSize != 0
        && (brg.isBF16 || brg.isInt8);
    auto xAddr = regAuxB;
    IntType baseOffset = 0;
    if constexpr (BRGEMM_MC_AVAILABLE) {
#ifdef KUDNNL_920F
        Mov(X_TMP_2, x15);
        if (bdB != 0) {
            Mov(x15, static_cast<uint64_t>((bdB / VAL_4) * VAL_4));
        }
        for (IntType bd = bdB; bd < bdE; bd++) {
            if (bd % VAL_4 == 0) {
                Mov(x15, static_cast<uint64_t>(bd));
            }
            const auto mask = isRdTail ? rdTailMask : rdFullMask;
            const IntType offset = AOffset(bd, 0);
            Mov(regTmp, static_cast<uint64_t>(offset / brg.typeSizeA));
            Ld1w(za0v.s, w15, static_cast<uint32_t>(bd % VAL_4), mask / PredType::T_Z,
                AdrScSc(regAuxA, regTmp, ShMod::LSL, static_cast<uint32_t>(VAL_2)));
        }
        const auto bdmask = isBdbTail ? bdTailMask : bdFullMask;
        const auto ldmask = isLdTail ? ldTailMask : ldFullMask;
        for (IntType rd = 0; rd < rdLoop; rd += brg.rdStep) {
            for (IntType ld = 0; ld < ldBlock2; ld++) {
                if (ld == 0) {
                    if (rd % VAL_4 == 0) {
                        Mov(x15, static_cast<uint64_t>(rd));
                    }
                    Mova(ZReg(maxEffectiveVregs - rd - 1).s, bdmask / PredType::T_M, za0h.s, w15,
                        static_cast<uint32_t>(rd % VAL_4));
                }
                load(ZReg(ld), xAddr, BOffset(ld, rd), baseOffset, brg.dtB);
                Fmopa(ZAReg(ZARegIndex(ld, brg.dtA)).s, ldmask / PredType::T_M, bdmask / PredType::T_M, ZReg(ld).s,
                    ZReg(maxEffectiveVregs - 1 - rd).s);
            }
        }
        Mov(x15, X_TMP_2);
#endif
    } else {
        for (IntType rd = 0; rd < rdLoop; rd += brg.rdStep) {
            for (IntType ld = 0; ld < ldBlock2; ld++) {
                load(Load(ld), xAddr, BOffset(ld, rd), baseOffset, brg.dtB);
            }
            for (IntType bd = bdB; bd < bdE; bd++) {
                if (rd == 0) {
                    bool haveToLoadBytes = maybeLoadBytes && (rd == rdLoop - brg.rdStep);
                    IntType rowsByLoadBytes = haveToLoadBytes ? rowsForRdTail : 0;
                    const bool bdByLoadBytes = bd >= bdE - rowsByLoadBytes || brg.brgattr.waryAKTailRead;
                    const bool tailBroadcast = isEmdbd ? false : haveToLoadBytes && bdByLoadBytes;
                    if (bd == bdB) {
                        broadcast(Bcst(bd - bdB), AOffset(bd, rd), tailBroadcast, brg.dtA);
                    }
                    if (bd != bdE - 1) {
                        broadcast(Bcst(bd + 1 - bdB), AOffset(bd + 1, rd), tailBroadcast, brg.dtA);
                    } else if (rd == 0) {
                        AddImm(xDefaultAddr, regAuxB, BOffset(0, rd + brg.rdStep), X_TMP_0);
                        Prfm(Prfop::PLDL1_KEEP, AdrUimm(Ptr(xDefaultAddr)));
                    }
                }
                for (IntType ld = 0; ld < ldBlock2; ld++) {
                    DotProduct(Accm(ldBlock2, bd, ld), Load(ld), Bcst(bd - bdB), rd);
                }
            }
        }
    }
}

void KdnnBrgemmKernelT::LdbLoop(IntType bdBlock2, bool bBdbTail, IntType ldBlock2, IntType ldbLoopLength,
    bool isRegTail, bool bLdTail, bool checkTopVpad, bool checkBottomVpad, IntType rowsForRdTail,
    bool skipAccumulation)
{
    Label ldbLoopLabel;
    Label bsLoopLabel;
    CopyPostOpsStackValuesToAux(isRegTail);
    auto ldLoopBody = [this](IntType vpad, IntType isBdbTail, IntType ibdBlock2,
        IntType ildBlock2, bool isLdTail, IntType iRowsForRdTail) {
        SetABMatrices();
        IntType bdBlock = isBdbTail ? brg.bdbTail : brg.bdBlock;
        const auto bdB = std::max(vpad, static_cast<IntType>(0));
        const auto bdE = std::min(bdBlock, bdBlock + vpad);
        const auto isValidBd = (needCompPads && vpad != 0) ? bdB <= bdE : bdB < bdE;
        if (!isValidBd) {
            return;
        }
        if (brg.rdb > 0) {
            Mov(regRdbLoop, brg.rdb);
            Label rdbLoopLabel;
            LAligned(rdbLoopLabel, VAL_64);
            const bool isRdTail = false;
            GemmMicrokernelVC(ibdBlock2, isBdbTail, ildBlock2, isRdTail, isLdTail, vpad, iRowsForRdTail);
            AddImm(regAuxA, regAuxA, RdbAOffset(), X_TMP_0);
            AddImm(regAuxB, regAuxB, RdbBOffset(), X_TMP_0);
            Sub(regRdbLoop, regRdbLoop, 1);
            CmpImm(regRdbLoop, 0, X_TMP_0);
            B(Cond::GT, rdbLoopLabel);
        }
        if (brg.rdbTail != 0) {
            const bool isRdTail = true;
            GemmMicrokernelVC(ibdBlock2, isBdbTail, ildBlock2, isRdTail, isLdTail, vpad, iRowsForRdTail);
        }
    };
    if (isLdbLoop) {
        MovImm(regLdbLoop, ldbLoopLength);
    }
    LAligned(ldbLoopLabel, VAL_64);
    // ldbLoop begin
    ZeroAccumulators(bdBlock2, bBdbTail, ldBlock2, bLdTail, skipAccumulation);
    if (isLdbLoop) {
        if ((regDOffs & ARMCG_BITMASK_12) == 0) {
            Str(regD, Ptr(xSP, regDOffs));
        } else {
            AddImm(xDefaultAddr, xSP, regDOffs, X_TMP_0);
            Str(regD, Ptr(xDefaultAddr));
        }
    } else {
        Mov(regLdbLoop, regD);
    }
    if (brg.brgattr.maxBs > 1) {
        if ((regAuxDOffs & ARMCG_BITMASK_12) == 0) {
            Str(regAuxD, Ptr(xSP, regAuxDOffs));
        } else {
            AddImm(xDefaultAddr, xSP, regAuxDOffs, X_TMP_0);
            Str(regAuxD, Ptr(xDefaultAddr));
        }
    }
    if (brg.alpha != 0.0f && !skipAccumulation) {
        RestoreABMatrices();
        if (brg.brgattr.maxBs > 1) {
            Mov(regBsLoop, regBs);
        }
        LAligned(bsLoopLabel, VAL_64);
        // bsLoop begin
        if (checkTopVpad || checkBottomVpad) {
            throw Service::Unsupported("");
        } else {
            ldLoopBody(0, bBdbTail, bdBlock2, ldBlock2, bLdTail, rowsForRdTail);
        }
        if (brg.brgattr.maxBs > 1) {
            Sub(regBsLoop, regBsLoop, 1);
            CmpImm(regBsLoop, 0, X_TMP_0);
            B(Cond::GT, bsLoopLabel);
        }
        // bsLoop end
    }
    if (isLdbLoop) {
        if ((regDOffs & ARMCG_BITMASK_12) == 0) {
            Ldr(regD, Ptr(xSP, regDOffs));
        } else {
            AddImm(xDefaultAddr, xSP, regDOffs, X_TMP_0);
            Ldr(regD, Ptr(xDefaultAddr));
        }
    } else {
        Mov(regD, regLdbLoop);
    }
    if (brg.brgattr.maxBs > 1) {
        if ((regAuxDOffs & ARMCG_BITMASK_12) == 0) {
            Ldr(regAuxD, Ptr(xSP, regAuxDOffs));
        } else {
            AddImm(xDefaultAddr, xSP, regAuxDOffs, X_TMP_0);
            Ldr(regAuxD, Ptr(xDefaultAddr));
        }
    }
    StoreAccumulators(bdBlock2, bBdbTail, ldBlock2, bLdTail, skipAccumulation);
    if (isLdbLoop) {
        if (!bLdTail) {
            LdbRegsShift(ldBlock2);
        } else {
            LdbRegsShift(1, true);
        }
        Sub(regLdbLoop, regLdbLoop, 1);
        CmpImm(regLdbLoop, 0, X_TMP_0);
        B(Cond::GT, ldbLoopLabel);
    }
    // ldbLoop end
}

void KdnnBrgemmKernelT::BdbLoop()
{
    auto doLdbLoop = [this](IntType bdBlock2, bool isBdbTail, bool checkTopVpad, bool checkBottomVpad,
        IntType rowsForRdTail, bool skipAccumulation) {
        if (brg.ldb2 > 0) {
            const bool isLdRegTail = false;
            const bool isLdTail = false;
            LdbLoop(bdBlock2, isBdbTail, brg.ldBlock2, brg.ldb2,
                isLdRegTail, isLdTail, checkTopVpad, checkBottomVpad, rowsForRdTail, skipAccumulation);
        }
        if (brg.ldb2Tail > 0) {
            const bool isLdRegTail = brg.ldb2 == 0 ? false : true;
            const bool isLdTail = false;
            LdbLoop(bdBlock2, isBdbTail, brg.ldb2Tail, 1,
                isLdRegTail, isLdTail, checkTopVpad, checkBottomVpad, rowsForRdTail, skipAccumulation);
        }
        if (brg.ldbTail > 0) {
            const bool isLdRegTail = (brg.ldb2 == 0 && brg.ldb2Tail == 0) ? false : true;
            const bool isLdTail = true;
            LdbLoop(bdBlock2, isBdbTail, 1, 1,
                isLdRegTail, isLdTail, checkTopVpad, checkBottomVpad, rowsForRdTail, skipAccumulation);
        }
    };
    auto bdbLoopBody = [this, &doLdbLoop](IntType bdBlock2, bool isBdbTail, bool checkTopVpad, bool checkBottomVpad,
        IntType rowsForRdTail, bool skipAccumulation) {
        doLdbLoop(bdBlock2, isBdbTail, checkTopVpad, checkBottomVpad, rowsForRdTail, skipAccumulation);
        AddImm(regC, regC, BdbCOffset(bdBlock2), X_TMP_0);
        AddImm(regD, regD, BdbDOffset(bdBlock2), X_TMP_0);
        AddImm(regAOffset, regAOffset, BdbAOffset(bdBlock2), X_TMP_0);
        AdvanceBdBlock2PostOpRegs(bdBlock2);
    };
    IntType rowsForRdTail = 0;
    IntType bdBlocksForRdTail = 0;
    IntType blocks = std::max(rowsForRdTail - brg.bdbTail + brg.brgattr.maxBottomVpad, static_cast<IntType>(0));
    if (brg.bdBlock == 0) {
        throw Service::LogicError("");
    }
    bdBlocksForRdTail = (blocks + brg.bdBlock - 1) / brg.bdBlock;
    auto ldBlock2 = brg.ldb2 > 0 ? brg.ldBlock2 : (brg.ldb2Tail > 0 ? brg.ldb2Tail : 1);
    const IntType freeVregs = maxEffectiveVregs - brg.reqS8S8Compensation;
    nBcast1Load = brg.isInt8 && brg.bdBlock * (ldBlock2 + 1) < freeVregs && bdBlocksForRdTail == 0
        && rowsForRdTail == 0;
    // loop order may be specified in brgemm attributes
    if (brg.brgattr.hintLoopOrder != BrgemmKernelLoopOrderT::UNDEFINED) {
        nBcast1Load = brg.brgattr.hintLoopOrder == BrgemmKernelLoopOrderT::LOAD;
    }
    auto bdbLoopVC = [this, rowsForRdTail, &bdbLoopBody, bdBlocksForRdTail, &doLdbLoop](bool skipAccumulation) {
        Label noVpadLabel;
        Label bdbLoopEndLabel;
        // batch may be null for BrgemmBatchKindT::STRIDE
        if (!vpadExist || brg.type == BrgemmBatchKindT::STRIDE) {
            LAligned(noVpadLabel, VAL_64);
            if (brg.bdb > 0) {
                MovImm(regBdbLoop, brg.bdb);
                if (brg.bdb > (rowsForRdTail ? 1 : 0)) {
                    Label bdbLoopLabel;
                    LAligned(bdbLoopLabel, VAL_64);
                    bdbLoopBody(1, false, false, false, bdBlocksForRdTail <= 1 ? 0 : rowsForRdTail, skipAccumulation);
                    Sub(regBdbLoop, regBdbLoop, 1);
                    CmpImm(regBdbLoop, rowsForRdTail ? 1 : 0, X_TMP_0);
                    B(Cond::GT, bdbLoopLabel);
                }
                if (rowsForRdTail) {
                    bdbLoopBody(1, false, false, true, bdBlocksForRdTail == 0 ? 0 : rowsForRdTail, skipAccumulation);
                }
            }
            if (brg.bdbTail > 0) {
                doLdbLoop(1, true, false, false, rowsForRdTail, skipAccumulation);
            }
        }
        LAligned(bdbLoopEndLabel, VAL_64);
    };
    auto bdbLoopGeneral = [this, &bdbLoopVC](bool skipAccumulation) {
        if (brg.type == BrgemmBatchKindT::ADDRESS && brg.brgattr.maxBs == 1 && !vpadExist && !skipAccumulation) {
            uint32_t ptraOffset = offsetof(BrgemmBatchElementT, ptr.a);
            uint32_t ptrbOffset = offsetof(BrgemmBatchElementT, ptr.b);
            Ldr(regAux1A, Ptr(regAddrBatch, ptraOffset));
            Ldr(regAux1B, Ptr(regAddrBatch, ptrbOffset));
        }
        Eor(regAOffset, regAOffset, regAOffset);
        bdbLoopVC(skipAccumulation);
    };
    if (!brg.brgattr.generateSkipAccumulation) {
        bdbLoopGeneral(false);
    }
}

void KdnnBrgemmKernelT::Generate()
{
    // make sure isa impl is consistent to cpu sve length
    bool isConsistent = false;
    switch (brg.isaImpl) {
        case CPUISAT::VC_512: isConsistent = CpuIsaTraits<CPUISAT::VC_512>::VLEN == cpuVCLen; break;
        case CPUISAT::VC_256: isConsistent = CpuIsaTraits<CPUISAT::VC_256>::VLEN == cpuVCLen; break;
        default: break;
    }
    if (!isConsistent) {
        throw Service::Unsupported("");
    }
    // make sure some code is unused and delete them
    vpadExist = brg.brgattr.maxTopVpad > 0 || brg.brgattr.maxBottomVpad > 0;
    needCompPads = (!(brg.zpTypeA == BrgemmBroadcastT::UNDEFINED) || !!(brg.reqS8S8Compensation))
        && (!(!vpadExist) || !!(brg.reqCalCompPads));
    bool isAssertSuccess = brg.alpha == 1.0f && brg.type == BrgemmBatchKindT::ADDRESS && !brg.withBinary
        && !brg.withScales && brg.zpTypeA == BrgemmBroadcastT::UNDEFINED && brg.zpTypeB == BrgemmBroadcastT::UNDEFINED
        && brg.zpTypeC == BrgemmBroadcastT::UNDEFINED && !brg.withDstScales && brg.brgattr.maxBs > 1
        && !brg.brgattr.generateSkipAccumulation && (brg.isF32) && !brg.isBF16 && !brg.isInt8
        && !brg.reqS8S8Compensation && !nBcast1Load && !vpadExist;
    if (!isAssertSuccess) {
        throw Service::Unsupported("");
    }

    Preamble();
    SubImm(xSP, xSP, stackSpaceNeeded, X_TMP_0);
#ifdef KUDNNL_920F
    if constexpr (BRGEMM_MC_AVAILABLE) {
        Smstart(StreamMode::SMZA);
        Ptrue(pAllOne.b);
        SetPreg(bdFullMask.s, brg.bdBlock, X_TMP_0, X_TMP_1);
        SetPreg(bdTailMask.s, brg.bdbTail, X_TMP_0, X_TMP_1);
        SetPreg(rdFullMask.s, brg.rdBlock, X_TMP_0, X_TMP_1);
        SetPreg(rdTailMask.s, brg.rdbTail, X_TMP_0, X_TMP_1);
    }
#endif
    SetPreg(ldFullMask.s, brg.ldBlock, X_TMP_0, X_TMP_1);
    SetPreg(ldTailMask.s, brg.ldbTail, X_TMP_0, X_TMP_1);
    Mov(x7, x0);
    Mov(x6, x1);
    Mov(x2, x2);
    Mov(x1, x3);
    Mov(x8, x4);
    Mov(x9, x5);
    ReadParams();
    BdbLoop();
#ifdef KUDNNL_920F
    if constexpr (BRGEMM_MC_AVAILABLE) {
        Smstop(StreamMode::SMZA);
    }
#endif
    AddImm(xSP, xSP, stackSpaceNeeded, X_TMP_0);
    Postamble();
    // [unimplemented] postops
}

} // namespace KuDNN
