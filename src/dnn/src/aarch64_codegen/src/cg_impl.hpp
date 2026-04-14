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


#ifndef CG_IMPL_HPP
#define CG_IMPL_HPP

// if imm is like "00..011..100..0" or "11..100..011..1", returns TRUE, otherwise FALSE
template <typename T>
bool IsBitMask(T imm)
{
    uint64_t bitptn = static_cast<uint64_t>(imm);
    int curr;
    int prev = 0;
    uint64_t invcount = 0;
    prev = (bitptn & 0x1) ? 1 : 0;
    for (size_t i = 1; i < DATA_SIZE_8_BYTES * sizeof(T); i++) {
        curr = (bitptn & (uint64_t(1) << i)) ? 1 : 0;
        if (prev != curr) {
            invcount++;
        }
        prev = curr;
    }
    // invcount == 0 means all 0 or all 1
    if (VAL_1 <= invcount && invcount <= VAL_2) {
        return true;
    }
    return false;
}

template<typename T1, typename T2>
void MoveData(T1 &a, T2 &b)
{
    int size = sizeof(b);
    void* pA = static_cast<void*>(&a);
    void* pB = static_cast<void*>(&b);

    uint8_t *pOne = static_cast<uint8_t *>(pA);
    uint8_t *pTwo = static_cast<uint8_t *>(pB);
    for (int i = 0; i < size; ++i) {
        pOne[i] = pTwo[i];
    }
}

template <typename T>
bool CheckValue(T val, const std::initializer_list<T> &list)
{
    return std::any_of(list.begin(), list.end(), [=](T x) { return x == val; });
}

template <typename T>
bool CheckValue(T val, T min, T max)
{
    return (min <= val && val <= max);
}

template <typename T>
bool CheckValue(T val, const std::function<bool(T)> &func)
{
    return func(val);
}

template <typename T>
void VerifyEqual(T valSrc, T valStd, Err errType)
{
    if (valSrc != valStd) {
        throw Error(errType);
    }
}

void VerifyIncRange(uint64_t val, uint64_t min, uint64_t max, Err errType, bool toI = false)
{
    if (toI && !CheckValue(static_cast<int64_t>(val), static_cast<int64_t>(min), static_cast<int64_t>(max))) {
        throw Error(errType);
    } else if (!toI && !CheckValue(val, min, max)) {
        throw Error(errType);
    }
}

void VerifyNotIncRange(uint64_t val, uint64_t min, uint64_t max, Err errType, bool toI = false)
{
    if (toI && CheckValue(static_cast<uint64_t>(val), static_cast<uint64_t>(min), static_cast<uint64_t>(max))) {
        throw Error(errType);
    } else if (!toI && CheckValue(val, min, max)) {
        throw Error(errType);
    }
}

void VerifyIncList(uint64_t val, const std::initializer_list<uint64_t> &list, Err errType)
{
    if (!CheckValue(val, list)) {
        throw Error(errType);
    }
}

void VerifyNotIncList(uint64_t val, const std::initializer_list<uint64_t> &list, Err errType)
{
    if (CheckValue(val, list)) {
        throw Error(errType);
    }
}

void VerifyCond(uint64_t val, const std::function<bool(uint64_t)> &func, Err errType)
{
    if (!CheckValue(val, func)) {
        throw Error(errType);
    }
}

void VerifyNotCond(uint64_t val, const std::function<bool(uint64_t)> &func, Err errType)
{
    if (CheckValue(val, func)) {
        throw Error(errType);
    }
}

constexpr uint32_t LeftShift(uint32_t val, uint32_t pos)
{
    return val << pos;
}

uint64_t Lsb(uint64_t v)
{
    return v & 0x1;
}

uint32_t Concat(const std::initializer_list<uint32_t> list)
{
    uint32_t result = 0;
    for (auto x : list) {
        result |= x;
    }
    return result;
}

uint64_t Msb(uint64_t v, uint32_t size)
{
    uint32_t shift = (size == 0) ? 0 : size - 1;
    return (v >> shift) & 0x1;
}

uint64_t Ones(uint32_t size)
{
    return (size == VAL_64) ? 0xffffffffffffffff : (1UL << size) - 1;
}

uint32_t Field(uint64_t v, uint32_t mpos, uint32_t lpos)
{
    if (mpos < lpos) {
        throw Error(Err::ILLEGAL_IMM_RANGE);
    }
    return static_cast<uint32_t>((v >> lpos) & Ones(mpos - lpos + 1));
}

uint64_t RRotate(uint64_t v, uint32_t size, uint32_t num)
{
    uint32_t shift = (size == 0) ? 0 : (num % size);
    v &= Ones(size);
    return (v << shift) | ((v & Ones(shift)) << (size - shift));
}

uint64_t LRotate(uint64_t v, uint32_t size, uint32_t num)
{
    uint32_t shift = (size == 0) ? 0 : (num % size);
    uint32_t detaShift = size - shift;
    v &= Ones(size);
    if (detaShift >= VAL_64) {
        return (v << shift) & Ones(size);
    }
    return ((v << shift) | (v >> detaShift)) & Ones(size);
}

uint64_t Replicate(uint64_t v, uint32_t esize, uint32_t size)
{
    uint64_t result = 0;
    for (uint32_t i = 0; i < VAL_64 / esize; ++i) {
        result |= v << (esize * i);
    }
    return result & Ones(size);
}

bool CheckPtn(uint64_t v, uint32_t esize, uint32_t size);
uint32_t GetPtnSize(uint64_t v, uint32_t size);
uint32_t GetPtnRotateNum(uint64_t ptn, uint32_t ptnSize);
uint32_t CountOneBit(uint64_t v, uint32_t size);
uint32_t CountSeqOneBit(uint64_t v, uint32_t size);
uint32_t Code8bitFloat(double x);
Cond Invert(Cond cond)
{
    uint32_t invVal = static_cast<uint32_t>(cond) ^ 1;
    return static_cast<Cond>(invVal & Ones(BITFIELD_4));
}
uint32_t GenSf(const RReg &reg)
{
    return (reg.GetBit() == BITFIELD_64) ? 1 : 0;
}
uint32_t GenQ(const VRegVec &reg)
{
    return (reg.GetBit() * reg.GetLane() == VAL_128) ? 1 : 0;
}
uint32_t GenQ(const VRegElem &reg);
uint32_t GenSize(const Reg &reg);
uint32_t GenSizeEnc(const VRegElem &reg);
uint32_t GenSize(uint32_t dtype)
{
    uint32_t size = (dtype == 0xf) ? VAL_3 :
        (dtype == 0x4 || dtype == 0xa || dtype == 0xb) ? 2 : (VAL_5 <= dtype && dtype <= VAL_9) ? 1 : 0;
    return size;
}
uint32_t GenS(const VRegElem &reg);

// generate encoded imm
uint32_t GenNImmrImms(uint64_t imm, uint32_t size);
// generate relative address for label offset
uint64_t GenLabelOffset(const Label &label, const JmpLabel &jmpL);

template <class T> void PutLInner(T &label)
{
    if (IsAutoGrow() && size_ >= maxSize_) {
        GrowMemory();
    }
    UncondBrImm(0, label);
}

// base
uint32_t PCrelAddrEnc(uint32_t op, const XReg &rd, int64_t labelOffset);
void PCrelAddr(uint32_t op, const XReg &rd, const Label &label);
void AddSubImm(uint32_t op, uint32_t s, const RReg &rd, const RReg &rn, uint32_t imm, uint32_t sh);
void LogicalImm(uint32_t opc, const RReg &rd, const RReg &rn, uint64_t imm, bool alias = false);
void MvWideImm(uint32_t opc, const RReg &rd, uint32_t imm, uint32_t sh);
void MvImm(const RReg &rd, uint64_t imm);
bool CheckMovAliasMovz(uint64_t imm, uint32_t rdBit, uint32_t &hw, uint32_t &fieldCount, uint32_t &imm16,
    uint32_t (&validField)[VAL_4]);
bool CheckMovAliasMovn(uint64_t invImm, uint32_t rdBit, uint32_t &invHw, uint32_t &invFieldCount, uint32_t &invImm16);
bool CheckMovAliasOrr(uint64_t imm, uint32_t rdBit);
void MovzPlusSuccessiveMovk(uint64_t imm, const RReg &rd, uint32_t (&validField)[VAL_4]);
void Bitfield(uint32_t opc, const RReg &rd, const RReg &rn, uint32_t immr, uint32_t imms, bool rnChk = true);
uint32_t CondBrImmEnc(uint32_t cond, int64_t labelOffset);
void CondBrImm(Cond cond, const Label &label);
void Hints(uint32_t crm, uint32_t op2);
void BarriersOpt(uint32_t op2, BarOpt opt, uint32_t rt);
void UncondBrNoReg(uint32_t opc, uint32_t op2, uint32_t op3, uint32_t rn, uint32_t op4);
void UncondBr1Reg(uint32_t opc, uint32_t op2, uint32_t op3, const RReg &rn, uint32_t op4);
uint32_t UncondBrImmEnc(uint32_t op, int64_t labelOffset);
void UncondBrImm(uint32_t op, const Label &label);
uint32_t CompareBrEnc(uint32_t op, const RReg &rt, int64_t labelOffset);
uint32_t TestBrEnc(uint32_t op, const RReg &rt, uint32_t imm, int64_t labelOffset);
void CompareBr(uint32_t op, const RReg &rt, const Label &label);
void StExclusive(uint32_t size, uint32_t o0, const WReg ws, const RReg &rt, const AdrImm &adr);
void LdExclusive(uint32_t size, uint32_t o0, const RReg &rt, const AdrImm &adr);
void LdStRegPairPostImm(uint32_t opc, uint32_t l, const RReg &rt1, const RReg &rt2, const AdrPostImm &adr);
void LdStRegPair(uint32_t opc, uint32_t l, const RReg &rt1, const RReg &rt2, const AdrImm &adr);
void LdStRegPairPre(uint32_t opc, uint32_t l, const RReg &rt1, const RReg &rt2, const AdrPreImm &adr);
void LdStRegUnsImm(uint32_t size, uint32_t opc, const RReg &rt, const AdrImm &adr);
void LdStRegPostImm(uint32_t size, uint32_t opc, const RReg &rt, const AdrPostImm &adr);
void LdStRegPre(uint32_t size, uint32_t opc, const RReg &rt, const AdrPreImm &adr);
void AtomicMemOp(uint32_t size, uint32_t v, uint32_t a, uint32_t r, uint32_t o3, uint32_t opc,
    const RReg &rs, const RReg &rt, const AdrNoOfs &adr);
void LdStRegUnImm(uint32_t size, uint32_t opc, const RReg &rt, const AdrUimm &adr);
void PfRegImm(Prfop prfop, const AdrUimm &adr);
void DataProc2Src(uint32_t opcode, const RReg &rd, const RReg &rn, const RReg &rm);
void LogicalShiftReg(uint32_t opc, uint32_t n, const RReg &rd, const RReg &rn,
    const RReg &rm, ShMod shmod, uint32_t sh);
void MvReg(const RReg &rd, const RReg &rn);
void AddSubShiftReg(uint32_t opc, uint32_t s, const RReg &rd, const RReg &rn, const RReg &rm,
    ShMod shmod, uint32_t sh, bool alias = false);
void AddSubExtReg(uint32_t opc, uint32_t s, const RReg &rd, const RReg &rn,
    const RReg &rm, ExtMod extmod, uint32_t sh);
void DataProc3Reg(uint32_t op54, uint32_t op31, uint32_t o0, const RReg &rd,
    const RReg &rn, const RReg &rm, const RReg &ra);

// simd
void AdvSimdLdStMultiStructExceptLd1St1(uint32_t l, uint32_t opc, const VRegList &vt, const AdrNoOfs &adr);
void AdvSimdLdStMultiStructForLd1St1(uint32_t l, uint32_t opc, const VRegList &vt, const AdrNoOfs &adr);
void AdvSimdLdStSingleStruct(uint32_t l, uint32_t r, uint32_t num, const VRegElem &vt, const AdrNoOfs &adr);
void AdvSimdLdRepSingleStruct(uint32_t l, uint32_t r, uint32_t opcode,
    uint32_t s, const VRegVec &vt, const AdrNoOfs &adr);
void AdvSimdLdStSingleStructPostReg(uint32_t l, uint32_t r, uint32_t num, const VRegElem &vt, const AdrPostReg &adr);
void AdvSimdLdStSingleStructPostImm(uint32_t l, uint32_t r, uint32_t num, const VRegElem &vt, const AdrPostImm &adr);
void LdStSimdFpPairPostImm(uint32_t l, const VRegSc &vt1, const VRegSc &vt2, const AdrPostImm &adr);
void LdStSimdFpPair(uint32_t l, const VRegSc &vt1, const VRegSc &vt2, const AdrImm &adr);
void LdStSimdFpPairPre(uint32_t l, const VRegSc &vt1, const VRegSc &vt2, const AdrPreImm &adr);
void LdStSimdFpUnImm(uint32_t opc, const VRegSc &vt, const AdrUimm &adr);
void AdvSimdPermute(uint32_t opcode, const VRegVec &vd, const VRegVec &vn, const VRegVec &vm);
void AdvSimdExtract(uint32_t op2, const VRegVec &vd, const VRegVec &vn, const VRegVec &vm, uint32_t index);
void AdvSimdCopyDupElem(uint32_t op, uint32_t imm4, const VRegVec &vd, const VRegElem &vn);
void AdvSimdCopyDupGen(uint32_t op, uint32_t imm4, const VRegVec &vd, const RReg &rn);
void AdvSimdCopyMov(uint32_t op, uint32_t imm4, const RReg &rd, const VRegElem &vn);
void AdvSimdCopyInsGen(uint32_t op, uint32_t imm4, const VRegElem &vd, const RReg &rn);
void AdvSimdCopyElemIns(uint32_t op, const VRegElem &vd, const VRegElem &vn);
void AdvSimd2RegMisc(uint32_t q, uint32_t u, uint32_t opcode, const VRegVec &vd, const VRegVec &vn);
void AdvSimd2RegMiscSz0x(uint32_t u, uint32_t opcode, const VRegVec &vd, const VRegVec &vn);
void AdvSimd2RegMiscSz1x(uint32_t u, uint32_t opcode, const VRegVec &vd, const VRegVec &vn);
void AdvSimdAcrossLanes(uint32_t u, uint32_t opcode, const VRegSc &vd, const VRegVec &vn);
void AdvSimd3Same(uint32_t u, uint32_t opcode, const VRegVec &vd, const VRegVec &vn, const VRegVec &vm);
void AdvSimd3SameSz0x(uint32_t u, uint32_t opcode, const VRegVec &vd, const VRegVec &vn, const VRegVec &vm);
void AdvSimd3SameSz1x(uint32_t u, uint32_t opcode, const VRegVec &vd, const VRegVec &vn, const VRegVec &vm);
void AdvSimd3SameSz(uint32_t u, uint32_t size, uint32_t opcode, const VRegVec &vd,
    const VRegVec &vn, const VRegVec &vm);
void AdvSimdModiImmMoviMvni(uint32_t op, uint32_t o2, const VRegVec &vd,
    uint32_t imm, ShMod shmod, uint32_t sh);
void AdvSimdShImm(uint32_t u, uint32_t opcode, const VRegVec &vd, const VRegVec &vn, uint32_t sh);
void FpDataProc2Reg(uint32_t m, uint32_t s, uint32_t type, uint32_t opcode,
    const VRegSc &vd, const VRegSc &vn, const VRegSc &vm);
void ConversionFpInt(uint32_t sf, uint32_t s, uint32_t type, uint32_t rmode,
    uint32_t opcode, const RReg &rd, const VRegSc &vn);
void ConversionFpInt(uint32_t sf, uint32_t s, uint32_t type, uint32_t rmode,
    uint32_t opcode, const VRegSc &vd, const RReg &rn);

// sve
void SveIntBinArPred(uint32_t opc, uint32_t type, const ZRegBase &zd,
    const PRegBase &pg, const ZRegBase &zn);
void SveBitwiseLOpPred(uint32_t opc, const ZRegBase &zd, const PRegBase &pg, const ZRegBase &zn);
void SveIntMinMaxDiffPred(uint32_t opc, uint32_t u, const ZRegBase &zd,
    const PRegBase &pg, const ZRegBase &zn);
void SveIntMultDivVecPred(uint32_t opc, uint32_t u, const ZRegBase &zd,
    const PRegBase &pg, const ZRegBase &zn);
void SveIntReduction(uint32_t opc, uint32_t type, const Reg &rd, const PRegBase &pg, const Reg &rn);
void SveIntAddReductPred(uint32_t opc, uint32_t u, const VRegSc &vd, const PRegBase &pg, const ZRegBase &zn);
void SveConstPrefPred(uint32_t opc, const ZRegBase &zd, const PRegBase &pg, const ZRegBase &zn);
void SveBitShPred(uint32_t opc, uint32_t type, const ZRegBase &zdn, const PRegBase &pg, const ZRegBase &zm);
void SveBitwiseShByImmPred(uint32_t opc, uint32_t l, uint32_t u, const ZRegBase &zdn,
    const PRegBase &pg, uint32_t amount);
void SveBitwiseShVecPred(uint32_t opc, const ZRegBase &zdn, const PRegBase &pg, const ZRegBase &zm);
void SveIntUnaryArPred(uint32_t opc, uint32_t type, const ZRegBase &zd,
    const PRegBase &pg, const ZRegBase &zn);
void SveBitwiseUnaryOpPred(uint32_t opc, const ZRegBase &zdn, const PRegBase &pg, const ZRegBase &zm);
void SveIntUnaryOpPred(uint32_t opc, const ZRegBase &zdn, const PRegBase &pg, const ZRegBase &zm);
void SveIntAddSubUnpred(uint32_t opc, const ZRegBase &zd, const ZRegBase &zn, const ZRegBase &zm);
void SveBitwiseLOpUnpred(uint32_t opc, const ZRegBase &zd, const ZRegBase &zn, const ZRegBase &zm);
void SveIndexGenImmImmInc(const ZRegBase &zd, int32_t imm1, int32_t imm2);
void SveStackFrameAdjust(uint32_t op, const XReg &xd, const XReg &xn, int32_t imm);
void SveBitwiseShByImmUnpred(uint32_t opc, const ZRegBase &zd,
    const ZRegBase &zn, uint32_t amount);
void SveIntMiscUnpred(uint32_t size, uint32_t opc, uint32_t type,
    const ZRegBase &zd, const ZRegBase &zn);
void SveFpExpAccel(uint32_t opc, const ZRegBase &zd, const ZRegBase &zn);
void SveElemCountGrp(uint32_t size, uint32_t op, uint32_t type1, uint32_t type2,
    const Reg &rd, Pattern pat, ExtMod mod, uint32_t imm);
void SveElemCount(uint32_t size, uint32_t op, const XReg &xd, Pattern pat, ExtMod mod, uint32_t imm);
void SveIncDecRegByElemCount(uint32_t size, uint32_t d, const XReg &xd, Pattern pat,
    ExtMod mod, uint32_t imm);
void SveBitwiseImm(uint32_t opc, const ZRegBase &zd, uint64_t imm);
void SveBitwiseLogicalImmUnpred(uint32_t opc, const ZRegBase &zdn, uint64_t imm);
void SveBcBitmaskImm(const ZRegBase &zdn, uint64_t imm);
void SveCopyFpImmPred(const ZRegBase &zd, const PRegBase &pg, double imm);
void SveCopyIntImmPred(const ZRegBase &zd, const PRegBase &pg, uint32_t imm, ShMod mod, uint32_t sh);
void SveExtVec(const ZRegBase &zdn, const ZRegBase &zm, uint32_t imm);
void SvePerVecUnpred(uint32_t size, uint32_t type1, uint32_t type2, const ZRegBase &zd, const Reg &rn);
void SveBcGeneralReg(const ZRegBase &zd, const RReg &rn);
void SveBcIndexedElem(const ZRegBase &zd, const ZRegElem &zn);
void SvePermutePredElem(uint32_t opc, uint32_t h, const PRegBase &pd,
    const PRegBase &pn, const PRegBase &pm);
void SvePermuteVecElem(uint32_t opc, const ZRegBase &zd, const ZRegBase &zn, const ZRegBase &zm);
void SveCopyGeneralRegToVecPred(const ZRegBase &zd, const PRegBase &pg, const RReg &rn);
void SveSelVecSplice(const ZRegBase &zd, const PRegBase &pg, const ZRegBase &zn);
void SveSelVecElemPred(const ZRegBase &zd, const PRegBase &pg, const ZRegBase &zn, const ZRegBase &zm);
void SveIntCompVecGrp(uint32_t opc, uint32_t ne, const PRegBase &pd, const PRegBase &pg,
    const ZRegBase &zn, const ZRegBase &zm);
void SveIntCompVec(uint32_t op, uint32_t o2, uint32_t ne, const PRegBase &pd,
    const PRegBase &pg, const ZRegBase &zn, const ZRegBase &zm);
void SvePredLOp(uint32_t op, uint32_t s, uint32_t o2, uint32_t o3, const PRegBase &pd,
    const PRegBase &pg, const PRegBase &pn, const PRegBase &pm);
void SvePredInit(uint32_t s, const PRegBase &pd, Pattern pat);
void SvePredZero(uint32_t op, uint32_t s, const PRegBase &pd);
void SveIntCompSImm(uint32_t op, uint32_t o2, uint32_t ne, const PRegBase &pd,
    const PRegBase &pg, const ZRegBase &zn, int32_t imm);
void SveIntCompScalarCountAndLimit(uint32_t u, uint32_t lt, uint32_t eq,
    const PRegBase &pd, const RReg &rn, const RReg &rm);
void SveBcFpImmUnpred(uint32_t opc, uint32_t o2, const ZRegBase &zd, double imm);
void SveBcIntImmUnpred(uint32_t opc, const ZRegBase &zd, int32_t imm, ShMod mod, uint32_t sh);
void SveIntMinMaxImmUnpred(uint32_t opc, uint32_t o2, const ZRegBase &zdn, int32_t imm);
void SveIntMultAddUnpredGroup(uint32_t op0, const ZRegBase &zda, const ZRegBase &zn, const ZRegBase &zm, uint32_t rot);
void SveFpMultAddIndexed(uint32_t op, const ZRegBase &zda, const ZRegBase &zn, const ZRegElem &zm);
void SveFpReciproEstUnPred(uint32_t opc, const ZRegBase &zd, const ZRegBase &zn);
void SveFpCompWithZero(uint32_t eq, uint32_t lt, uint32_t ne, const PRegBase &pd,
    const PRegBase &pg, const ZRegBase &zn, double zero);
void SveFpArithmeticUnpred(uint32_t opc, const ZRegBase &zd, const ZRegBase &zn, const ZRegBase &zm);
void SveFpArithmeticPred(uint32_t opc, const ZRegBase &zdn, const PRegBase &pg, const ZRegBase &zm);
void SveFpArithmeticImmPred(uint32_t opc, const ZRegBase &zdn, const PRegBase &pg, float ct);
void SveFpCvtPrecision(uint32_t opc, uint32_t opc2, const ZRegBase &zd, const PRegBase &pg, const ZRegBase &zn);
void SveFpCvtToInt(uint32_t opc, uint32_t opc2, uint32_t u, const ZRegBase &zd, const PRegBase &pg, const ZRegBase &zn);
void SveFpRoundToIntegral(uint32_t opc, const ZRegBase &zd, const PRegBase &pg, const ZRegBase &zn);
void SveFpUnaryOp(uint32_t opc, const ZRegBase &zd, const PRegBase &pg, const ZRegBase &zn);
void SveIntCvtToFp(uint32_t opc, uint32_t opc2, uint32_t u, const ZRegBase &zd, const PRegBase &pg, const ZRegBase &zn);
void SveFpCompVec(uint32_t op, uint32_t o2, uint32_t o3, const PRegBase &pd, const PRegBase &pg,
    const ZRegBase &zn, const ZRegBase &zm);
void SveFpMultAccumAddend(uint32_t opc, const ZRegBase &zda, const PRegBase &pg,
    const ZRegBase &zn, const ZRegBase &zm);
void SveFpMultAccumMulti(uint32_t opc, const ZRegBase &zdn, const PRegBase &pg, const ZRegBase &zm, const ZRegBase &za);
void Sve32GatherLdSc32U(uint32_t msz, uint32_t u, uint32_t ff, const ZRegBase &zt,
    const PRegBase &pg, const AdrSc32U &adr);
void Sve32GatherLdWSc32S(uint32_t u, uint32_t ff, const ZRegBase &zt, const PRegBase &pg, const AdrSc32S &adr);
void Sve32ContiPfScImm(PrfopSve prfopSve, uint32_t msz, const PRegBase &pg, const AdrScImm &adr);
void Sve32ContiPfScImm(PrfopSve prfopSve, uint32_t msz, const PRegBase &pg, const AdrNoOfs &adr);
void SveLoadAndBcElem(uint32_t dtypeh, uint32_t dtypel, const ZRegBase &zt, const PRegBase &pg, const AdrScImm &adr);
void SveLoadAndBcElem(uint32_t dtypeh, uint32_t dtypel, const ZRegBase &zt, const PRegBase &pg, const AdrNoOfs &adr);
void SveLoadPredReg(const PRegBase &pt, const AdrScImm &adr);
void SveLoadPredReg(const PRegBase &pt, const AdrNoOfs &adr);
void SveLoadPredVec(const ZRegBase &zt, const AdrScImm &adr);
void SveLoadPredVec(const ZRegBase &zt, const AdrNoOfs &adr);
void SveContiLdScImm(uint32_t dtype, const ZRegBase &zt, const PRegBase &pg, const AdrScImm &adr);
void SveContiLdScImm(uint32_t dtype, const ZRegBase &zt, const PRegBase &pg, const AdrNoOfs &adr);
void SveContiLdScSc(uint32_t dtype, const ZRegBase &zt, const PRegBase &pg, const AdrScSc &adr);
void SveLdBcQuadScImm(uint32_t msz, uint32_t num, const ZRegBase &zt, const PRegBase &pg, const AdrNoOfs &adr);
void SveContiNTStScImm(uint32_t msz, const ZRegBase &zt, const PRegBase &pg, const AdrNoOfs &adr);
void SveContiStScImm(uint32_t msz, const ZRegBase &zt, const PRegBase &pg, const AdrScImm &adr);
void SveContiStScImm(uint32_t msz, const ZRegBase &zt, const PRegBase &pg, const AdrNoOfs &adr);
void SveContiStScSc(uint32_t msz, const ZRegBase &zt, const PRegBase &pg, const AdrScSc &adr);
void SveStorePredReg(const PRegBase &pt, const AdrScImm &adr);
void SveStorePredReg(const PRegBase &pt, const AdrNoOfs &adr);
void SveStorePredVec(const ZRegBase &zt, const AdrScImm &adr);
void SveStorePredVec(const ZRegBase &zt, const AdrNoOfs &adr);
void TestBr(uint32_t op, const RReg &rt, uint32_t imm, const Label &label);
void SveFpWideMultAddGroup(uint32_t bit10, const ZRegBase &zda, const ZRegBase &zn, const ZRegBase &zm);
void SveFpWideMultAddIndexedGroup(uint32_t bit10, const ZRegBase &zda, const ZRegBase &zn, const ZRegElem &zm);

// MatrixComputation
#ifdef KUDNNL_920Pro
void MatrixComputationStart(StreamMode mod);
void MatrixComputationStop(StreamMode mod);
void MatrixComputationZero(const std::initializer_list<ZARegBase> &list);
uint32_t EncodeSourcesMOP(const ZRegBase &zm, const PRegBase &pm, const PRegBase &pn, const ZRegBase &zn);
void MatrixComputationOuterProductAccNonWidening(const ZARegS &zada, const PRegBase &pn, const PRegBase &pm,
    const ZRegS &zn, const ZRegS &zm);
void MatrixComputationOuterProductAcc2Way(const ZARegS &zada, const PRegBase &pn, const PRegBase &pm,
    const ZRegH &zn, const ZRegH &zm);
void MatrixComputationOuterProductAccSigned4Way(const ZARegS &zada, const PRegBase &pn, const PRegBase &pm,
    const ZRegB &zn, const ZRegB &zm);
void MatrixComputationOuterProductSubSigned4Way(const ZARegS &zada, const PRegBase &pn, const PRegBase &pm,
    const ZRegB &zn, const ZRegB &zm);
void MatrixComputationOuterProductAccSignedUnsigned(const ZARegS &zada, const PRegBase &pn, const PRegBase &pm,
    const ZRegB &zn, const ZRegB &zm);
void MatrixComputationOuterProductSubSignedUnsigned(const ZARegS &zada, const PRegBase &pn, const PRegBase &pm,
    const ZRegB &zn, const ZRegB &zm);
void MatrixComputationOuterProductAccUnsigned4Way(const ZARegS &zada, const PRegBase &pn, const PRegBase &pm,
    const ZRegB &zn, const ZRegB &zm);
void MatrixComputationOuterProductSubUnsigned4Way(const ZARegS &zada, const PRegBase &pn, const PRegBase &pm,
    const ZRegB &zn, const ZRegB &zm);
void MatrixComputationOuterProductAccUnsignedSigned(const ZARegS &zada, const PRegBase &pn, const PRegBase &pm,
    const ZRegB &zn, const ZRegB &zm);
void MatrixComputationOuterProductSubUnsignedSigned(const ZARegS &zada, const PRegBase &pn, const PRegBase &pm,
    const ZRegB &zn, const ZRegB &zm);
void MatrixComputationMovTile2Vec(const ZRegB &zd, const PRegBase &pg, const ZARegB &zan, bool v, const WReg &ws, uint32_t offs);
void MatrixComputationMovTile2Vec(const ZRegS &zd, const PRegBase &pg, const ZARegS &zan, bool v, const WReg &ws, uint32_t offs);
void MatrixComputationMovVec2Tile(const ZARegB &zad, bool v, const WReg &ws, uint32_t offs, const PRegBase &pg, const ZRegB &zn);
void MatrixComputationMovVec2Tile(const ZARegS &zad, bool v, const WReg &ws, uint32_t offs, const PRegBase &pg, const ZRegS &zn);
void MatrixComputationStoreTile(const ZARegB &zat, bool v, const WReg &ws, uint32_t offs, const PRegBase &pg, const AdrScSc &adr);
void MatrixComputationLoadTile(const ZARegB &zat, bool v, const WReg &ws, uint32_t offs, const PRegBase &pg, const AdrScSc &adr);
void MatrixComputationStoreTile(const ZARegS &zat, bool v, const WReg &ws, uint32_t offs, const PRegBase &pg, const AdrScSc &adr);
void MatrixComputationAddVec2Tile(const ZARegS &zada, bool v, const PRegBase &pn, const PRegBase &pm, const ZRegS &zn);
#endif

#endif // CG_IMPL_HPP
