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

#ifndef KUDNN_BRGEMM_KERNEL_HPP
#define KUDNN_BRGEMM_KERNEL_HPP

#include <cstdint>

#include "cg_jit_generator.hpp"
#include "service/kudnn_err_codes.hpp"
#include "service/kudnn_exception.hpp"
#include "types/kudnn_data_type.hpp"

#include "kudnn_brgemm_types.hpp"

namespace KuDNN {
using namespace ARMCG;

constexpr IntType BRGEMM_KERNEL_NUM_2 = 2;
#ifdef KUDNNL_920F
constexpr bool BRGEMM_MC_AVAILABLE = true;
#else
constexpr bool BRGEMM_MC_AVAILABLE = false;
#endif

struct KdnnBrgemmKernelT : public JitGenerator {
public:
    explicit KdnnBrgemmKernelT(const BrgemmT &brgIn) : JitGenerator(nullptr, CODE_SIZE_256_1024, true), brg(brgIn),
        maxEffectiveVregs(maxVregs - (brg.isInt8 && !brg.hasInt8Vnni ? BRGEMM_KERNEL_NUM_2 : static_cast<IntType>(0)))
    {
        if (brg.isaImpl == CPUISAT::ISA_UNDEF || brg.isaImpl == CPUISAT::ISA_ALL) {
            throw Service::Unsupported("");
        }

        const IntType isLdbTail = brg.ldbTail ? 1 : 0;
        const IntType isLdb2Tail = brg.ldb2Tail ? 1 : 0;
        isLdbLoop = brg.ldb2 + isLdb2Tail + isLdbTail > 1;
        // [unimplemented] postops
    }

    const char *Name() const override
    {
        return "kdnn_brgemm_kernel_t";
    }
    const char *SourceFile() const override
    {
        return __FILE__;
    }

    BrgemmT brg;

private:
    const XReg param1 = XReg(7);
    const XReg regC = x15;
    const XReg regAuxC = x14;
    const XReg regAddrBatch = x13;
    const XReg regA = x13;
    const XReg regB = x12;
    const XReg regAuxA = x11;
    const XReg regAuxB = x10;
    const XReg regAuxAVpad = regAuxA;
    const XReg regBdbLoop = x9;
    const XReg regLdbLoop = x8;
    const XReg regStrideLda = regBdbLoop;
    const XReg regStrideLdb = regLdbLoop;
    const XReg regStrideLdBlock = regLdbLoop;
    const XReg regS8InputShift = regBdbLoop;
    const XReg regZpAInputShift = regBdbLoop;
    const XReg regBsLoop = x0;
    const XReg regRdbLoop = x3;
    const XReg regBs = x1;
    const XReg regAOffset = x2;
    const XReg regBOffset = x6;
    const XReg regAux1Batch = x5;
    const XReg regAux1A = x5;
    const XReg regAux1B = x7;
    const XReg regOffsetBatch = regAux1A;
    const XReg regStrideBatch = regRdbLoop;
    const XReg regBias = regRdbLoop;
    const XReg regScales = regRdbLoop;
    const XReg regAuxBias = regRdbLoop;
    const XReg regDstScales = regRdbLoop;
    const XReg regZpCompA = regRdbLoop;
    const XReg regAuxZpCompA = regRdbLoop;
    const XReg regZpCompB = regRdbLoop;
    const XReg regAuxZpCompB = regRdbLoop;
    const XReg regZpCValues = regRdbLoop;
    const XReg regAuxZpCValues = regRdbLoop;
    const XReg regAuxScales = regAuxB;
    const XReg regAuxDstScales = regAuxB;
    const XReg regDoPostOps = regRdbLoop;
    const XReg regDoComp = regRdbLoop;
    const XReg regSkipAccm = regRdbLoop;
    const XReg regTmpGpr = regRdbLoop;
    const XReg regPtrSumScale = regRdbLoop;
    const XReg regPtrSumZp = regBdbLoop;
    const XReg regZpAVal = regRdbLoop;
    const XReg regBuf = regRdbLoop;
    const XReg regCompensation = regBias;
    const XReg regAuxCompensation = regAuxBias;
    const XReg regD = regAuxA;
    const XReg regAuxD = regBsLoop;
    const XReg regTmp = x16;

    constexpr static int originOffsetBatchOffs = 0;
    constexpr static int originStrideBatchOffs = 0;
    constexpr static int regBiasOffs = 8;
    constexpr static int regAuxBiasOffs = 16;
    constexpr static int regDoPostOpsOffs = 24;
    constexpr static int regDOffs = 32;
    constexpr static int regAuxDOffs = 40;
    constexpr static int regScalesOffs = 48;
    constexpr static int regAuxScalesOffs = 56;
    constexpr static int regBdbLoopOffs = 64;
    constexpr static int regLdbLoopOffs = 72;
    constexpr static int regBufOffs = 80;
    constexpr static int regCompOffs = regBufOffs;
    constexpr static int regAuxCompOffs = 88;
    constexpr static int abiParam1Offs = 96;
    constexpr static int regZpCompAOffs = 104;
    constexpr static int regAuxZpCompAOffs = 112;
    constexpr static int regZpCompBOffs = 120;
    constexpr static int regAuxZpCompBOffs = 128;
    constexpr static int regZpCValuesOffs = 136;
    constexpr static int regAuxZpCValuesOffs = 144;
    constexpr static int regDataCPtr = 152;
    constexpr static int regSkipAccmOffs = 160;
    constexpr static int regZpAValOffs = 168;
    constexpr static int regDoCompOffs = 176;
    constexpr static int regDstScalesOffs = 184;
    constexpr static int stackSpaceNeeded = 192;

    bool isLdbLoop = false;
    bool withBinaryNonScalarBcast = false;
    constexpr static IntType maxVregs = 32;
    const IntType maxEffectiveVregs;

    PReg bdFullMask = PReg(1);
    PReg bdTailMask = PReg(2);
    PReg ldFullMask = PReg(3);
    PReg ldTailMask = PReg(4);
    PReg rdFullMask = PReg(5);
    PReg rdTailMask = PReg(6);

    ZReg Accm(IntType ldBlock, IntType bd, IntType ld)
    {
        uint32_t idx = maxEffectiveVregs - 1 - (bd * ldBlock + ld);

        return ZReg(idx);
    }
    ZReg Bcst(IntType bd = 0)
    {
        uint32_t idx = bd;

        return ZReg(idx);
    }
    ZReg Load(IntType ld = 0)
    {
        uint32_t idx = maxEffectiveVregs - 1 - (brg.ldBlock2 * brg.bdBlock) - ld;

        return ZReg(idx);
    }
    uint32_t ZARegIndex(IntType ld, Element::TypeT dt)
    {
        uint32_t idx = Element::Type(dt).GetSize() - 1 - ld;
        if (idx > Element::Type(dt).GetSize() - 1) {
            throw Service::LogicError("");
        }

        return idx;
    }
    const ZReg &ZTmp1() const noexcept
    {
        return this->z0;
    }
    const ZReg &ZTmp2() const noexcept
    {
        return this->z1;
    }
    const ZReg &ZTmp3() const noexcept
    {
        return this->z2;
    }
    const ZReg &ZTailMask() const noexcept
    {
        return this->z1;
    }
    const ZReg &ZOneBytes() const noexcept
    {
        return this->z3;
    }
    const ZReg &ZZpAShift() const noexcept
    {
        return this->z2;
    }
    const ZReg &ZInpShift() const noexcept
    {
        return this->z1;
    }
    ZReg Int8OnesWords() const noexcept
    {
        uint32_t idx = maxVregs - 1;

        return ZReg(idx);
    }
    ZReg Int8DotProductTemp() const noexcept
    {
        uint32_t idx = maxVregs - BRGEMM_KERNEL_NUM_2;

        return ZReg(idx);
    }

    void LoadData(Element::TypeT typeIn, ZReg &vmm, XReg &regAddr, IntType loadSize);
    void Cvt2ps(Element::TypeT typeIn, const ZReg zmmIn, const XReg &addr, bool maskFlag, bool store, PReg ktailMask,
        const IntType offset, const IntType baseOffset);
    void AdvanceLdbPostOpRegs();
    void RestoreLdbPostOpRegs(IntType ldBlock2);
    void AdvanceBdbPostOpRegs(IntType adjBdBlock);
    void RestoreBdbPostOpRegs(IntType bdBlock2);
    void LdbRegsShift(IntType ldBlock2, bool isTail = false);
    void AdvanceBdBlock2PostOpRegs(IntType bdBlock2);
    void CopyPostOpsStackValuesToAux(bool isRegTail);
    void ReadParams();
    void ZeroAccumulators(IntType bdBlock2, bool isBdbTail, IntType ldBlock2, bool isLdTail, bool skipAccumulation);
    void StoreAccumulators(IntType bdBlock2, bool isBdbTail, IntType ldBlock2, bool isLdTail, bool skipAccumulation);
    void StoreAccumulatorsWithoutPostOps(IntType bdBlock, IntType ldBlock2, bool isLdTail);
    void StoreAccumulatorsApplyPostOps(IntType bdBlock, IntType ldBlock2, IntType ldbAndBdbOffset, bool isLdTail);
    void ApplyCompensation(IntType bdBlock, IntType ldBlock2, bool isLdTail);
    void ApplyAlphaBeta(IntType bdBlock, IntType ldBlock2, bool isLdTail);
    void ApplyPostOps(IntType bdBlock, IntType ldBlock2, IntType ldbAndBdbOffset, bool isLdTail);
    void SetABMatrices();
    void RestoreABMatrices();
    void ComputeInt8Compensation(IntType rdLoop, IntType bdB, IntType bdE, IntType bdBlock, IntType ldBlock2,
        bool isLdTail, IntType vpad);
    void DotProduct(ZReg z1, ZReg z2, ZReg z3);
    void DotProduct(ZReg z1, ZReg z2, ZReg z3, IntType offset);
    void GemmMicrokernelVC(IntType bdBlock2, bool isBdbTail, IntType ldBlock2, bool isRdTail, bool isLdTail,
        IntType vpad, IntType rowsForRdTail);
    void LdbLoop(IntType bdBlock2, bool isBdbTail, IntType ldBlock2, IntType ldbLoopLength, bool isRegTail,
        bool isLdTail, bool checkTopVpad, bool checkBottomVpad, IntType rowsForRdTail, bool skipAccumulation);
    void BdbLoop();
    void Generate() override;

    IntType AOffset(IntType bd, IntType rd) const noexcept
    {
        return brg.typeSizeA * (bd * brg.ldA + rd);
    }
    IntType BOffset(IntType ld, IntType rd) const noexcept
    {
        return brg.typeSizeB * ((rd / brg.ldStep) * brg.ldStep * brg.ldB + brg.ldStep * ld * brg.ldBlock);
    }
    IntType COffset(IntType bd, IntType ld) const noexcept
    {
        return brg.typeSizeC * (bd * brg.ldC + ld * brg.ldBlock);
    }
    IntType DOffset(IntType bd, IntType ld) const noexcept
    {
        return brg.typeSizeD * (bd * brg.ldD + ld * brg.ldBlock);
    }
    IntType BiasOffset(IntType ld, bool isTail = false) const noexcept
    {
        return isTail ? brg.typeSizeBias * brg.ldbTail : brg.typeSizeBias * ld * brg.ldBlock;
    }
    IntType PoOffset(IntType bd, IntType ld) const noexcept
    {
        return bd * brg.ldD + ld * brg.ldBlock;
    }
    IntType ScalesOffset(IntType ld, bool isTail = false) const noexcept;
    IntType BdbAOffset(IntType bdBlock2) const noexcept
    {
        return brg.typeSizeA * bdBlock2 * brg.bdBlock * brg.ldA;
    }
    IntType BdbCOffset(IntType bdBlock2) const noexcept
    {
        return brg.typeSizeC * bdBlock2 * brg.bdBlock * brg.ldC;
    }
    IntType BdbDOffset(IntType bdBlock2) const noexcept
    {
        return brg.typeSizeD * bdBlock2 * brg.bdBlock * brg.ldD;
    }
    IntType BdbPoOffset(IntType bdBlock2) const noexcept
    {
        return bdBlock2 * brg.bdBlock * brg.ldD;
    }
    IntType BdbCompensationOffset(IntType bdBlock2) const noexcept
    {
        return bdBlock2 * brg.bdBlock * brg.ldB * sizeof(int32_t);
    }
    IntType BdbZpCompAOffset(IntType bdBlock2) const noexcept
    {
        return bdBlock2 * brg.bdBlock * brg.ldB * sizeof(int32_t);
    }
    IntType BdbZpCompBOffset(IntType bdBlock2) const noexcept
    {
        return ZpCompBOffset(bdBlock2 * brg.bdBlock);
    }
    IntType LdbBOffset(IntType ldBlock2, bool isTail = false) const noexcept
    {
        return isTail
            ? brg.typeSizeB * brg.ldbTail * brg.ldStep
            : brg.typeSizeB * ldBlock2 * brg.ldBlock * brg.ldStep;
    }
    IntType LdbCOffset(IntType ldBlock2, bool isTail = false) const noexcept
    {
        return isTail ? brg.typeSizeC * brg.ldbTail : brg.typeSizeC * ldBlock2 * brg.ldBlock;
    }
    IntType LdbDOffset(IntType ldBlock2, bool isTail = false) const noexcept
    {
        return isTail ? brg.typeSizeD * brg.ldbTail : brg.typeSizeD * ldBlock2 * brg.ldBlock;
    }
    IntType LdbPoOffset(IntType ldBlock2, bool isTail = false) const noexcept
    {
        return isTail ? brg.ldbTail : ldBlock2 * brg.ldBlock;
    }
    IntType RdbAOffset() const noexcept
    {
        return brg.typeSizeA * brg.rdBlock;
    }
    IntType RdbBOffset() const noexcept
    {
        return brg.typeSizeB * brg.rdBlock * brg.ldB;
    }

    IntType OcLogicalOffset(IntType ld, bool isTail = false) const noexcept
    {
        return isTail ? brg.ldbTail : ld * brg.ldBlock;
    }
    IntType CompensationsOffset(IntType ld, bool isTail = false) const noexcept;
    IntType CompensationVpadOffset(IntType ld, IntType bd) const noexcept
    {
        return (ld * brg.ldBlock + bd * brg.ldB) * sizeof(int32_t);
    }
    IntType ZpCompAOffset(IntType ld, bool isTail = false) const noexcept;
    IntType ZpCompAVpadOffset(IntType ld, IntType bd) const noexcept
    {
        return (ld * brg.ldBlock + bd * brg.ldB) * sizeof(int32_t);
    }
    IntType ZpCompBOffset(IntType bd) const noexcept
    {
        return bd * sizeof(int32_t);
    }
    IntType ZpCValuesOffset(IntType ld, bool isTail = false) const noexcept;

    bool nBcast1Load = false;
    bool vpadExist = false;
    bool needCompPads = false;
};

struct BrgemmKernelT : public BrgemmBaseKernelT {
public:
    explicit BrgemmKernelT(BrgemmT brg) : kernel(new KdnnBrgemmKernelT(brg)) {}
    ~BrgemmKernelT() override
    {
        delete kernel;
    }
    void operator()(BrgemmKernelParamsT *params) const override
    {
        using CallBrgemmKernelT = void (*)(BrgemmKernelParamsT *);

        CallBrgemmKernelT func = (decltype(func))(kernel->JitKer());
        func(params);
    }

    Status CreateKernel() override
    {
        return kernel->CreateKernel() ? Status::SUCCESS : Status::BAD_ARGUMENTS;
    }

private:
    BrgemmKernelT(const BrgemmKernelT &) = delete;
    void operator=(const BrgemmKernelT &) = delete;

    KdnnBrgemmKernelT *kernel = nullptr;
};

} // namespace KuDNN

#endif // KUDNN_BRGEMM_KERNEL_HPP
