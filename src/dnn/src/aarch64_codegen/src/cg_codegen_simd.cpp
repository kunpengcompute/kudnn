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
#include "cg_codegen.hpp"

#include "src/cg_error.hpp"
#include "src/cg_memory.hpp"
#include "src/cg_reg.hpp"
#include "src/cg_inner.hpp"
#include "tools/cg_util.hpp"

namespace ARMCG {

// Advanced SIMD load/store multipule structure
void CodeGenerator::AdvSimdLdStMultiStructExceptLd1St1(
    uint32_t l, uint32_t opc, const VRegList &vt, const AdrNoOfs &adr)
{
    uint32_t q = GenQ(vt);
    uint32_t size = GenSize(vt);
    uint32_t len = vt.GetLen();

    VerifyIncRange(len, BITFIELD_1, BITFIELD_4, Err::ILLEGAL_REG_IDX);

    opc = (opc == 0x2 && len == 1) ? 0x7 : (opc == 0x2 && len == VAL_2) ? 0xa :
        (opc == 0x2 && len == VAL_3) ? 0x6 : opc;
    uint32_t code = Concat({LeftShift(q, BITFIELD_30), LeftShift(0x18, BITFIELD_23), LeftShift(l, BITFIELD_22),
        LeftShift(opc, BITFIELD_12), LeftShift(size, BITFIELD_10), LeftShift(adr.GetXn().GetIdx(), BITFIELD_5),
        LeftShift(vt.GetIdx(), 0)});
    WriteData(code);
}

void CodeGenerator::AdvSimdLdStMultiStructForLd1St1(uint32_t l, uint32_t opc, const VRegList &vt, const AdrNoOfs &adr)
{
    AdvSimdLdStMultiStructExceptLd1St1(l, opc, vt, adr);
}

// Advanced SIMD load/store single structures
void CodeGenerator::AdvSimdLdStSingleStruct(
    uint32_t l, uint32_t r, uint32_t num, const VRegElem &vt, const AdrNoOfs &adr)
{
    uint32_t q = GenQ(vt);
    uint32_t s = GenS(vt);
    uint32_t size = GenSizeEnc(vt);
    uint32_t opc = (vt.GetBit() == VAL_8) ? Field(num - 1, 1, 1) : (vt.GetBit() == VAL_16) ?
        Field(num - 1, BITFIELD_1, BITFIELD_1) + VAL_2 : Field(num - 1, BITFIELD_1, BITFIELD_1) + VAL_4;
    uint32_t code = Concat({LeftShift(q, BITFIELD_30), LeftShift(0x1a, BITFIELD_23), LeftShift(l, BITFIELD_22),
        LeftShift(r, BITFIELD_21), LeftShift(opc, BITFIELD_13), LeftShift(s, BITFIELD_12), LeftShift(size, BITFIELD_10),
        LeftShift(adr.GetXn().GetIdx(), BITFIELD_5), LeftShift(vt.GetIdx(), 0)});
    WriteData(code);
}

// Advanced SIMD load replication single structures
void CodeGenerator::AdvSimdLdRepSingleStruct(uint32_t l, uint32_t r, uint32_t opcode, uint32_t s,
    const VRegVec &vt, const AdrNoOfs &adr)
{
    uint32_t q = GenQ(vt);
    uint32_t size = GenSize(vt);
    uint32_t code = Concat({LeftShift(q, BITFIELD_30), LeftShift(0x1a, BITFIELD_23), LeftShift(l, BITFIELD_22),
        LeftShift(r, BITFIELD_21), LeftShift(opcode, BITFIELD_13), LeftShift(s, BITFIELD_12),
        LeftShift(size, BITFIELD_10), LeftShift(adr.GetXn().GetIdx(), BITFIELD_5), LeftShift(vt.GetIdx(), 0)});
    WriteData(code);
}

// Advanced SIMD load/store single structures (post-indexed register)
void CodeGenerator::AdvSimdLdStSingleStructPostReg(uint32_t l, uint32_t r,
    uint32_t num, const VRegElem &vt, const AdrPostReg &adr)
{
    uint32_t q = GenQ(vt);
    uint32_t s = GenS(vt);
    uint32_t size = GenSizeEnc(vt);
    uint32_t opc = (vt.GetBit() == VAL_8) ? Field(num - 1, BITFIELD_1, BITFIELD_1) :
        (vt.GetBit() == VAL_16) ? Field(num - 1, BITFIELD_1, BITFIELD_1) + VAL_2 : Field(num - 1, 1, 1) + VAL_4;
    uint32_t code = Concat({LeftShift(q, BITFIELD_30), LeftShift(0x1b, BITFIELD_23), LeftShift(l, BITFIELD_22),
        LeftShift(r, BITFIELD_21), LeftShift(adr.GetXm().GetIdx(), BITFIELD_16), LeftShift(opc, BITFIELD_13),
        LeftShift(s, BITFIELD_12), LeftShift(size, BITFIELD_10), LeftShift(adr.GetXn().GetIdx(), BITFIELD_5),
        LeftShift(vt.GetIdx(), 0)});
    WriteData(code);
}

// Advanced SIMD load/store single structures (post-indexed immediate)
void CodeGenerator::AdvSimdLdStSingleStructPostImm(
    uint32_t l, uint32_t r, uint32_t num, const VRegElem &vt, const AdrPostImm &adr)
{
    uint32_t q = GenQ(vt);
    uint32_t s = GenS(vt);
    uint32_t size = GenSizeEnc(vt);
    uint32_t opc = (vt.GetBit() == VAL_8) ? Field(num - 1, BITFIELD_1, BITFIELD_1) : (vt.GetBit() == VAL_16) ?
        Field(num - 1, BITFIELD_1, BITFIELD_1) + VAL_2 : Field(num - 1, BITFIELD_1, BITFIELD_1) + VAL_4;

    VerifyIncList(adr.GetImm(), {num * vt.GetBit() / VAL_8}, Err::ILLEGAL_IMM_VALUE);

    uint32_t code = Concat({LeftShift(q, BITFIELD_30), LeftShift(0x1b, BITFIELD_23), LeftShift(l, BITFIELD_22),
        LeftShift(r, BITFIELD_21), LeftShift(0x1f, BITFIELD_16), LeftShift(opc, BITFIELD_13), LeftShift(s, BITFIELD_12),
        LeftShift(size, BITFIELD_10), LeftShift(adr.GetXn().GetIdx(), BITFIELD_5), LeftShift(vt.GetIdx(), 0)});
    WriteData(code);
}

// Advanced SIMD load/store multple structures (post-indexed register offset)
void CodeGenerator::AdvSimdLdStMultiStructPostRegExceptLd1St1(
    uint32_t l, uint32_t opc, const VRegList &vt, const AdrPostReg &adr)
{
    uint32_t q = GenQ(vt);
    uint32_t size = GenSize(vt);

    VerifyIncRange(adr.GetXm().GetIdx(), 0, SP_IDX - 1, Err::ILLEGAL_REG_IDX);

    uint32_t len = vt.GetLen();
    VerifyIncRange(len, VAL_1, VAL_4, Err::ILLEGAL_REG_IDX);
    opc = (opc == 0x2 && len == VAL_1) ? 0x7 : (opc == 0x2 && len == VAL_2)
                                       ? 0xa : (opc == 0x2 && len == VAL_3) ? 0x6 : opc;
    uint32_t code = Concat({LeftShift(q, BITFIELD_30), LeftShift(0x19, BITFIELD_23), LeftShift(l, BITFIELD_22),
        LeftShift(adr.GetXm().GetIdx(), BITFIELD_16), LeftShift(opc, BITFIELD_12), LeftShift(size, BITFIELD_10),
        LeftShift(adr.GetXn().GetIdx(), BITFIELD_5), LeftShift(vt.GetIdx(), BITFIELD_0)});
    WriteData(code);
}

void CodeGenerator::AdvSimdLdStMultiStructPostRegForLd1St1(
    uint32_t l, uint32_t opc, const VRegList &vt, const AdrPostReg &adr)
{
    AdvSimdLdStMultiStructPostRegExceptLd1St1(l, opc, vt, adr);
}

// Advanced SIMD load/store multple structures (post-indexed immediate offset)
void CodeGenerator::AdvSimdLdStMultiStructPostImmExceptLd1St1(
    uint32_t l, uint32_t opc, const VRegList &vt, const AdrPostImm &adr)
{
    uint32_t q = GenQ(vt);
    uint32_t size = GenSize(vt);
    uint32_t len = vt.GetLen();

    VerifyIncRange(adr.GetImm(), 0, ((VAL_8 * len) << q), Err::ILLEGAL_IMM_RANGE);
    VerifyIncRange(len, VAL_1, VAL_4, Err::ILLEGAL_REG_IDX);

    opc = (opc == 0x2 && len == VAL_1) ? 0x7 : (opc == 0x2 && len == VAL_2)
                                       ? 0xa : (opc == 0x2 && len == VAL_3) ? 0x6 : opc;
    uint32_t code = Concat({LeftShift(q, BITFIELD_30), LeftShift(0x19, BITFIELD_23), LeftShift(l, BITFIELD_22),
        LeftShift(0x1f, BITFIELD_16), LeftShift(opc, BITFIELD_12), LeftShift(size, BITFIELD_10),
        LeftShift(adr.GetXn().GetIdx(), BITFIELD_5), LeftShift(vt.GetIdx(), BITFIELD_0)});
    WriteData(code);
}

// Advanced SIMD load/store multple structures (post-indexed immediate offset)
void CodeGenerator::AdvSimdLdStMultiStructPostImmForLd1St1(
    uint32_t l, uint32_t opc, const VRegList &vt, const AdrPostImm &adr)
{
    AdvSimdLdStMultiStructPostImmExceptLd1St1(l, opc, vt, adr);
}

// Load/store pair (offset)
void CodeGenerator::LdStSimdFpPair(uint32_t l, const VRegSc &vt1, const VRegSc &vt2, const AdrImm &adr)
{
    int32_t imm = adr.GetImm();
    int32_t times = vt1.GetBit() / 32;

    VerifyIncRange(imm, (SIMD_LDRSTR_IMM_RANGE_LOW * times), (SIMD_LDRSTR_IMM_RANGE_HIGH * times),
        Err::ILLEGAL_IMM_RANGE, true);
    VerifyCond(
        imm, [=](uint64_t x) { return ((x % (OFFSET_FOUR * times)) == 0); }, Err::ILLEGAL_IMM_COND);

    uint32_t opc = (vt1.GetBit() == 32) ? 0 : (vt1.GetBit() == 64) ? 1 : 2;
    uint32_t sh = static_cast<uint32_t>(std::log2(4 * times));
    uint32_t imm7 = (imm >> sh) & Ones(IMM7_BIT_LENGTH);
    uint32_t v = 1;
    uint32_t code = Concat({LeftShift(opc, BITFIELD_30), LeftShift(0x5, BITFIELD_27), LeftShift(v, BITFIELD_26),
        LeftShift(0x2, BITFIELD_23), LeftShift(l, BITFIELD_22), LeftShift(imm7, BITFIELD_15),
        LeftShift(vt2.GetIdx(), BITFIELD_10), LeftShift(adr.GetXn().GetIdx(), BITFIELD_5),
        LeftShift(vt1.GetIdx(), BITFIELD_0)});
    WriteData(code);
}

// Load/store pair (pre-indexed)
void CodeGenerator::LdStSimdFpPairPre(uint32_t l, const VRegSc &vt1, const VRegSc &vt2, const AdrPreImm &adr)
{
    int32_t imm = adr.GetImm();
    int32_t times = vt1.GetBit() / 32;

    VerifyIncRange(imm, (SIMD_LDRSTR_IMM_RANGE_LOW * times), (SIMD_LDRSTR_IMM_RANGE_HIGH * times),
        Err::ILLEGAL_IMM_RANGE, true);
    VerifyCond(
        imm, [=](uint64_t x) { return ((x % (OFFSET_FOUR * times)) == 0); }, Err::ILLEGAL_IMM_COND);

    uint32_t opc = (vt1.GetBit() == 32) ? 0 : (vt1.GetBit() == 64) ? 1 : 2;
    uint32_t sh = static_cast<uint32_t>(std::log2(OFFSET_FOUR * times));
    uint32_t imm7 = (imm >> sh) & Ones(IMM7_BIT_LENGTH);
    uint32_t v = 1;
    uint32_t code = Concat({LeftShift(opc, BITFIELD_30), LeftShift(0x5, BITFIELD_27), LeftShift(v, BITFIELD_26),
        LeftShift(0x3, BITFIELD_23), LeftShift(l, BITFIELD_22), LeftShift(imm7, BITFIELD_15),
        LeftShift(vt2.GetIdx(), BITFIELD_10), LeftShift(adr.GetXn().GetIdx(), BITFIELD_5),
        LeftShift(vt1.GetIdx(), BITFIELD_0)});
    WriteData(code);
}

// Load/store pair (post-indexed)
void CodeGenerator::LdStSimdFpPairPostImm(uint32_t l, const VRegSc &vt1, const VRegSc &vt2, const AdrPostImm &adr)
{
    int32_t imm = adr.GetImm();
    int32_t times = vt1.GetBit() / 32;

    VerifyIncRange(imm, (SIMD_LDRSTR_IMM_RANGE_LOW * times), (SIMD_LDRSTR_IMM_RANGE_HIGH * times),
        Err::ILLEGAL_IMM_RANGE, true);
    VerifyCond(
        imm, [times](uint64_t x) { return ((x % (OFFSET_FOUR * times)) == 0); }, Err::ILLEGAL_IMM_COND);

    uint32_t opc = (vt1.GetBit() == 32) ? 0 : (vt1.GetBit() == 64) ? 1 : 2;
    uint32_t sh = static_cast<uint32_t>(std::log2(OFFSET_FOUR * times));
    uint32_t imm7 = (imm >> sh) & Ones(IMM7_BIT_LENGTH);
    uint32_t v = 1;
    uint32_t code = Concat({LeftShift(opc, BITFIELD_30), LeftShift(0x5, BITFIELD_27), LeftShift(v, BITFIELD_26),
        LeftShift(0x1, BITFIELD_23), LeftShift(l, BITFIELD_22), LeftShift(imm7, BITFIELD_15),
        LeftShift(vt2.GetIdx(), BITFIELD_10), LeftShift(adr.GetXn().GetIdx(), BITFIELD_5),
        LeftShift(vt1.GetIdx(), BITFIELD_0)});
    WriteData(code);
}

// loat/store register (unsigned immediate)
void CodeGenerator::LdStSimdFpUnImm(uint32_t opc, const VRegSc &vt, const AdrUimm &adr)
{
    uint32_t imm = adr.GetImm();
    uint32_t times = vt.GetBit() / VC_ELT_BITLEN_B;
    uint32_t sh = static_cast<uint32_t>(std::log2(times));
    uint32_t imm12 = (imm >> sh) & Ones(IMM12_BIT_LENGTH);

    VerifyIncRange(imm, 0, IMM13_RANGE_HIGH * times, Err::ILLEGAL_IMM_RANGE);
    VerifyCond(imm, [=](uint64_t x) {return ((x & Ones(sh)) == 0);}, Err::ILLEGAL_IMM_COND);

    uint32_t size = GenSize(vt);
    uint32_t code = Concat({LeftShift(size, BITFIELD_30), LeftShift(0x7, BITFIELD_27), LeftShift(0x1, BITFIELD_26),
        LeftShift(0x1, BITFIELD_24), LeftShift(opc, BITFIELD_22), LeftShift(imm12, BITFIELD_10),
        LeftShift(adr.GetXn().GetIdx(), BITFIELD_5), LeftShift(vt.GetIdx(), BITFIELD_0)});
    WriteData(code);
}

// Load/store register (SIMD&FP, unscaled immediate)
void CodeGenerator::LdStSimdFpRegUnsImm(uint32_t opc, const VRegSc &vt, const AdrImm &adr)
{
    uint32_t size = (vt.GetBit() == 16) ? 1 : (vt.GetBit() == 32) ? 2 : (vt.GetBit() == 64) ? 3 : 0;

    int imm = adr.GetImm();
    uint32_t imm9 = adr.GetImm() & Ones(IMM9_BIT_LENGTH);

    VerifyIncRange(imm, IMM9_RANGE_LOW, IMM9_RANGE_HIGH, Err::ILLEGAL_IMM_RANGE, true);

    uint32_t v = 1;
    uint32_t code = Concat({LeftShift(size, BITFIELD_30), LeftShift(0x7, BITFIELD_27), LeftShift(v, BITFIELD_26),
        LeftShift(opc, BITFIELD_22), LeftShift(imm9, BITFIELD_12), LeftShift(adr.GetXn().GetIdx(), BITFIELD_5),
        LeftShift(vt.GetIdx(), BITFIELD_0)});
    WriteData(code);
}

// Advanced SIMD permute
void CodeGenerator::AdvSimdPermute(uint32_t opcode, const VRegVec &vd, const VRegVec &vn, const VRegVec &vm)
{
    uint32_t q = GenQ(vd);
    uint32_t size = GenSize(vd);
    uint32_t code = Concat({LeftShift(q, BITFIELD_30), LeftShift(0xe, BITFIELD_24), LeftShift(size, BITFIELD_22),
        LeftShift(vm.GetIdx(), BITFIELD_16), LeftShift(opcode, BITFIELD_12), LeftShift(0x2, BITFIELD_10),
        LeftShift(vn.GetIdx(), BITFIELD_5), LeftShift(vd.GetIdx(), BITFIELD_0)});
    WriteData(code);
}

// Advanced SIMD extract
void CodeGenerator::AdvSimdExtract(
    uint32_t op2, const VRegVec &vd, const VRegVec &vn, const VRegVec &vm, uint32_t index)
{
    uint32_t q = GenQ(vd);
    uint32_t imm4 = static_cast<uint32_t>(index) & Ones(IMM4_BIT_LENGTH);

    VerifyIncRange(index, 0, IMM5_RANGE_HIGH, Err::ILLEGAL_CONST_RANGE);
    if (q == 0) {
        VerifyCond(imm4, [](int64_t x) { return (static_cast<uint64_t>(x) >> OFFSET_THREE) == 0; },
            Err::ILLEGAL_CONST_COND);
    }

    uint32_t code = Concat({LeftShift(q, BITFIELD_30), LeftShift(0x2e, BITFIELD_24), LeftShift(op2, BITFIELD_22),
        LeftShift(vm.GetIdx(), BITFIELD_16), LeftShift(imm4, BITFIELD_11),
        LeftShift(vn.GetIdx(), BITFIELD_5), LeftShift(vd.GetIdx(), BITFIELD_0)});
    WriteData(code);
}

// Advanced SIMD copy
void CodeGenerator::AdvSimdCopyDupElem(uint32_t op, uint32_t imm4, const VRegVec &vd, const VRegElem &vn)
{
    uint32_t q = GenQ(vd);
    uint32_t size = GenSize(vd);
    uint32_t imm5 = (0x1 << size) | (vn.GetElemIdx() << (size + OFFSET_ONE));
    uint32_t code = Concat({LeftShift(q, BITFIELD_30), LeftShift(op, BITFIELD_29), LeftShift(0xe, BITFIELD_24),
        LeftShift(imm5, BITFIELD_16), LeftShift(imm4, BITFIELD_11),
        LeftShift(0x1, BITFIELD_10), LeftShift(vn.GetIdx(), BITFIELD_5), LeftShift(vd.GetIdx(), BITFIELD_0)});
    WriteData(code);
}

// Advanced SIMD copy
void CodeGenerator::AdvSimdCopyDupGen(uint32_t op, uint32_t imm4, const VRegVec &vd, const RReg &rn)
{
    uint32_t q = GenQ(vd);
    uint32_t size = GenSize(vd);
    uint32_t imm5 = 0x1 << size;
    uint32_t code = Concat({LeftShift(q, BITFIELD_30), LeftShift(op, BITFIELD_29), LeftShift(0xe, BITFIELD_24),
        LeftShift(imm5, BITFIELD_16), LeftShift(imm4, BITFIELD_11), LeftShift(0x3, BITFIELD_10),
        LeftShift(rn.GetIdx(), BITFIELD_5), LeftShift(vd.GetIdx(), BITFIELD_0)});
    WriteData(code);
}

// Advanced SIMD copy
void CodeGenerator::AdvSimdCopyMov(uint32_t op, uint32_t imm4, const RReg &rd, const VRegElem &vn)
{
    uint32_t q = GenSf(rd);
    uint32_t size = GenSize(vn);
    uint32_t imm5 = ((0x1 << size) | (vn.GetElemIdx() << (size + OFFSET_ONE))) & Ones(IMM5_BIT_LENGTH);
    uint32_t code = Concat({LeftShift(q, BITFIELD_30), LeftShift(op, BITFIELD_29), LeftShift(0xe, BITFIELD_24),
        LeftShift(imm5, BITFIELD_16), LeftShift(imm4, BITFIELD_11), LeftShift(0x1, BITFIELD_10),
        LeftShift(vn.GetIdx(), BITFIELD_5), LeftShift(rd.GetIdx(), BITFIELD_0)});
    WriteData(code);
}

// Advanced SIMD copy
void CodeGenerator::AdvSimdCopyInsGen(uint32_t op, uint32_t imm4, const VRegElem &vd, const RReg &rn)
{
    uint32_t q = 0x1;
    uint32_t size = GenSize(vd);
    uint32_t imm5 = ((0x1 << size) | (vd.GetElemIdx() << (size + OFFSET_ONE))) & Ones(IMM5_BIT_LENGTH);
    uint32_t code = Concat({LeftShift(q, BITFIELD_30), LeftShift(op, BITFIELD_29), LeftShift(0xe, BITFIELD_24),
        LeftShift(imm5, BITFIELD_16), LeftShift(imm4, BITFIELD_11), LeftShift(0x1, BITFIELD_10),
        LeftShift(rn.GetIdx(), BITFIELD_5), LeftShift(vd.GetIdx(), BITFIELD_0)});
    WriteData(code);
}

// Advanced SIMD copy
void CodeGenerator::AdvSimdCopyElemIns(uint32_t op, const VRegElem &vd, const VRegElem &vn)
{
    uint32_t q = 0x1;
    uint32_t size = GenSize(vd);
    uint32_t imm5 = ((0x1 << size) | (vd.GetElemIdx() << (size + OFFSET_ONE))) & Ones(IMM5_BIT_LENGTH);
    uint32_t imm4 = (vn.GetElemIdx() << size) & Ones(IMM4_BIT_LENGTH);
    uint32_t code = Concat({LeftShift(q, BITFIELD_30), LeftShift(op, BITFIELD_29), LeftShift(0xe, BITFIELD_24),
        LeftShift(imm5, BITFIELD_16), LeftShift(imm4, BITFIELD_11), LeftShift(0x1, BITFIELD_10),
        LeftShift(vn.GetIdx(), BITFIELD_5), LeftShift(vd.GetIdx(), BITFIELD_0)});
    WriteData(code);
}

// Advanced SIMD two-register miscellaneous
void CodeGenerator::AdvSimd2RegMisc(uint32_t q, uint32_t u, uint32_t opcode, const VRegVec &vd, const VRegVec &vn)
{
    uint32_t size = GenSize(vn);

    uint32_t code = Concat({LeftShift(q, BITFIELD_30), LeftShift(u, BITFIELD_29), LeftShift(0xe, BITFIELD_24),
        LeftShift(size, BITFIELD_22), LeftShift(0x1, BITFIELD_21), LeftShift(opcode, BITFIELD_12),
        LeftShift(0x2, BITFIELD_10), LeftShift(vn.GetIdx(), BITFIELD_5), LeftShift(vd.GetIdx(), BITFIELD_0)});
    WriteData(code);
}

// Advanced SIMD two-register miscellaneous
void CodeGenerator::AdvSimd2RegMiscSz0x(uint32_t u, uint32_t opcode, const VRegVec &vd, const VRegVec &vn)
{
    bool selVd = (opcode == 0x17);
    uint32_t q = (!selVd) ? GenQ(vd) : GenQ(vn);
    uint32_t size = (selVd) ? GenSize(vd) : GenSize(vn);
    uint32_t code = Concat({LeftShift(q, BITFIELD_30), LeftShift(u, BITFIELD_29), LeftShift(0xe, BITFIELD_24),
        LeftShift((size & 0x1), BITFIELD_22), LeftShift(0x1, BITFIELD_21), LeftShift(opcode, BITFIELD_12),
        LeftShift(2, BITFIELD_10), LeftShift(vn.GetIdx(), BITFIELD_5), LeftShift(vd.GetIdx(), BITFIELD_0)});
    WriteData(code);
}

// Advanced SIMD two-register miscellaneous
void CodeGenerator::AdvSimd2RegMiscSz1x(uint32_t u, uint32_t opcode, const VRegVec &vd, const VRegVec &vn,
    double zero)
{
    VerifyIncList(std::lround(zero * VAL_10), {0}, Err::ILLEGAL_CONST_VALUE);
    AdvSimd2RegMiscSz1x(u, opcode, vd, vn);
}
void CodeGenerator::AdvSimd2RegMiscSz1x(uint32_t u, uint32_t opcode, const VRegVec &vd, const VRegVec &vn)
{
    uint32_t q = GenQ(vd);
    uint32_t size = GenSize(vd);
    uint32_t code = Concat({LeftShift(q, BITFIELD_30), LeftShift(u, BITFIELD_29), LeftShift(0xe, BITFIELD_24),
        LeftShift(size, BITFIELD_22), LeftShift(0x1, BITFIELD_21), LeftShift(opcode, BITFIELD_12),
        LeftShift(0x2, BITFIELD_10), LeftShift(vn.GetIdx(), BITFIELD_5), LeftShift(vd.GetIdx(), BITFIELD_0)});
    WriteData(code);
}

// Advanced SIMD across lanes
void CodeGenerator::AdvSimdAcrossLanes(uint32_t u, uint32_t opcode, const VRegSc &vd, const VRegVec &vn)
{
    uint32_t q = GenQ(vn);
    uint32_t size = GenSize(vn);
    uint32_t code = Concat({LeftShift(q, BITFIELD_30), LeftShift(u, BITFIELD_29), LeftShift(0xe, BITFIELD_24),
        LeftShift(size, BITFIELD_22), LeftShift(0x3, BITFIELD_20), LeftShift(opcode, BITFIELD_12),
        LeftShift(0x2, BITFIELD_10), LeftShift(vn.GetIdx(), BITFIELD_5), LeftShift(vd.GetIdx(), BITFIELD_0)});
    WriteData(code);
}

// Advanced SIMD three same
void CodeGenerator::AdvSimd3Same(uint32_t u, uint32_t opcode, const VRegVec &vd, const VRegVec &vn, const VRegVec &vm)
{
    uint32_t q = GenQ(vd);
    uint32_t size = GenSize(vd);
    uint32_t code = Concat({LeftShift(q, BITFIELD_30), LeftShift(u, BITFIELD_29), LeftShift(0xe, BITFIELD_24),
        LeftShift(size, BITFIELD_22), LeftShift(0x1, BITFIELD_21), LeftShift(vm.GetIdx(), BITFIELD_16),
        LeftShift(opcode, BITFIELD_11), LeftShift(0x1, BITFIELD_10), LeftShift(vn.GetIdx(), BITFIELD_5),
        LeftShift(vd.GetIdx(), BITFIELD_0)});
    WriteData(code);
}

// Advanced SIMD three same
void CodeGenerator::AdvSimd3Same(uint32_t opc2, const VRegVec &vd, const VRegVec &vn, const VRegVec &vm)
{
    uint32_t q = GenQ(vd);
    uint32_t code = Concat({LeftShift(q, BITFIELD_30), LeftShift(0x1, BITFIELD_29), LeftShift(0xe, BITFIELD_24),
        LeftShift(opc2, BITFIELD_21), LeftShift(vm.GetIdx(), BITFIELD_16),
        LeftShift(0x7, BITFIELD_10), LeftShift(vn.GetIdx(), BITFIELD_5), LeftShift(vd.GetIdx(), BITFIELD_0)});
    WriteData(code);
}
 
void CodeGenerator::AdvSimd2RegMVN(uint32_t u, uint32_t opcode,
    const VRegVec &vd, const VRegVec &vn)
{
    uint32_t q = GenQ(vd);
    uint32_t code = Concat({LeftShift(q, BITFIELD_30), LeftShift(u, BITFIELD_29), LeftShift(0xe, BITFIELD_24),
        LeftShift(opcode, BITFIELD_21), LeftShift(0x1, BITFIELD_14),
        LeftShift(0x6, BITFIELD_10), LeftShift(vn.GetIdx(), BITFIELD_5),
        LeftShift(vd.GetIdx(), BITFIELD_0)});
    WriteData(code);
}

// Advanced SIMD three same
void CodeGenerator::AdvSimd3SameSz0x(
    uint32_t u, uint32_t opcode, const VRegVec &vd, const VRegVec &vn, const VRegVec &vm)
{
    uint32_t q = GenQ(vd);
    uint32_t size = GenSize(vd) & 1;
    uint32_t code = Concat({LeftShift(q, BITFIELD_30), LeftShift(u, BITFIELD_29), LeftShift(0xe, BITFIELD_24),
        LeftShift(size, BITFIELD_22), LeftShift(0x1, BITFIELD_21), LeftShift(vm.GetIdx(), BITFIELD_16),
        LeftShift(opcode, BITFIELD_11), LeftShift(0x1, BITFIELD_10), LeftShift(vn.GetIdx(), BITFIELD_5),
        LeftShift(vd.GetIdx(), BITFIELD_0)});
    WriteData(code);
}

// Advanced SIMD three same
void CodeGenerator::AdvSimd3SameSz1x(uint32_t u, uint32_t opcode, const VRegVec &vd,
    const VRegVec &vn, const VRegVec &vm)
{
    uint32_t q = GenQ(vd);
    uint32_t size = GenSize(vd);
    uint32_t code = Concat({LeftShift(q, BITFIELD_30), LeftShift(u, BITFIELD_29), LeftShift(0xe, BITFIELD_24),
        LeftShift(size, BITFIELD_22), LeftShift(0x1, BITFIELD_21), LeftShift(vm.GetIdx(), BITFIELD_16),
        LeftShift(opcode, BITFIELD_11), LeftShift(0x1, BITFIELD_10), LeftShift(vn.GetIdx(), BITFIELD_5),
        LeftShift(vd.GetIdx(), BITFIELD_0)});
    WriteData(code);
}

// Advanced SIMD three same
void CodeGenerator::AdvSimd3SameSz(
    uint32_t u, uint32_t size, uint32_t opcode, const VRegVec &vd, const VRegVec &vn, const VRegVec &vm)
{
    uint32_t q = GenQ(vd);
    uint32_t code = Concat({LeftShift(q, BITFIELD_30), LeftShift(u, BITFIELD_29), LeftShift(0xe, BITFIELD_24),
        LeftShift(size, BITFIELD_22), LeftShift(0x1, BITFIELD_21), LeftShift(vm.GetIdx(), BITFIELD_16),
        LeftShift(opcode, BITFIELD_11), LeftShift(0x1, BITFIELD_10), LeftShift(vn.GetIdx(), BITFIELD_5),
        LeftShift(vd.GetIdx(), BITFIELD_0)});
    WriteData(code);
}

// Advanced SIMD modified immediate (vector)
void CodeGenerator::AdvSimdModiImmMoviMvni(
    uint32_t op, uint32_t o2, const VRegVec &vd, uint32_t imm, ShMod shmod, uint32_t sh)
{
    uint32_t q = GenQ(vd);
    uint32_t crmode = (vd.GetBit() == VC_ELT_BITLEN_B) ? 0xe :
        (vd.GetBit() == VC_ELT_BITLEN_H) ? 0x8 | (sh >> OFFSET_TWO) :
        (vd.GetBit() == VC_ELT_BITLEN_S && shmod == ShMod::LSL) ? (sh >> OFFSET_TWO) :
        (vd.GetBit() == VC_ELT_BITLEN_S && shmod == ShMod::MSL) ? 0xc | (sh >> OFFSET_FOUR) : 0xe;

    if (vd.GetBit() == VC_ELT_BITLEN_B) {
        VerifyIncList(sh, {0}, Err::ILLEGAL_CONST_VALUE);
    } else if (vd.GetBit() == VC_ELT_BITLEN_H) {
        VerifyIncList(sh, {VC_ELT_BITLEN_B * Field(crmode, BITFIELD_1, BITFIELD_1)}, Err::ILLEGAL_CONST_VALUE);
    } else if (vd.GetBit() == VC_ELT_BITLEN_S && shmod == ShMod::LSL) {
        VerifyIncList(sh, {VC_ELT_BITLEN_B * Field(crmode, BITFIELD_2, BITFIELD_1)}, Err::ILLEGAL_CONST_VALUE);
    } else if (vd.GetBit() == VC_ELT_BITLEN_S && shmod == ShMod::MSL) {
        VerifyIncList(sh, {VC_ELT_BITLEN_B * Field(crmode, 0, 0) + VC_ELT_BITLEN_B}, Err::ILLEGAL_CONST_VALUE);
    }

    uint32_t abc = Field(imm, BITFIELD_7, BITFIELD_5);
    uint32_t defgh = Field(imm, BITFIELD_4, BITFIELD_0);
    uint32_t code = Concat({LeftShift(q, BITFIELD_30), LeftShift(op, BITFIELD_29), LeftShift(0xf, BITFIELD_24),
        LeftShift(abc, BITFIELD_16), LeftShift(crmode, BITFIELD_12), LeftShift(o2, BITFIELD_11),
        LeftShift(0x1, BITFIELD_10), LeftShift(defgh, BITFIELD_5), LeftShift(vd.GetIdx(), BITFIELD_0)});
    WriteData(code);
}

// Advanced SIMD modified immediate
void CodeGenerator::AdvSimdModiImmFmov(uint32_t op, uint32_t o2, const VRegVec &vd, double imm)
{
    uint32_t q = GenQ(vd);
    uint32_t crmode = 0xf;
    uint32_t imm8 = Code8bitFloat(imm);
    uint32_t abc = Field(imm8, BITFIELD_7, BITFIELD_5);
    uint32_t defgh = Field(imm8, BITFIELD_4, BITFIELD_0);
    uint32_t code = Concat({LeftShift(q, BITFIELD_30), LeftShift(op, BITFIELD_29), LeftShift(0xf, BITFIELD_24),
        LeftShift(abc, BITFIELD_16), LeftShift(crmode, BITFIELD_12), LeftShift(o2, BITFIELD_11),
        LeftShift(0x1, BITFIELD_10), LeftShift(defgh, BITFIELD_5), LeftShift(vd.GetIdx(), BITFIELD_0)});
    WriteData(code);
}

// Advanced SIMD Scalar two-register miscellaneous
void CodeGenerator::AdvSimdSc2RegMiscSz1x(uint32_t u, uint32_t opcode, const VRegSc &vd, const VRegSc &vn)
{
    uint32_t size = GenSize(vd);
    uint32_t code = Concat({LeftShift(1, BITFIELD_30), LeftShift(u, BITFIELD_29), LeftShift(0xf, BITFIELD_25),
        LeftShift(size, BITFIELD_22), LeftShift(1, BITFIELD_21), LeftShift(opcode, BITFIELD_12),
        LeftShift(VAL_2, BITFIELD_10), LeftShift(vn.GetIdx(), BITFIELD_5), LeftShift(vd.GetIdx(), 0)});
    WriteData(code);
}
void CodeGenerator::AdvSimdSc2RegMiscSz1x(uint32_t u, uint32_t opcode, const VRegSc &vd, const VRegSc &vn,
    double zero)
{
    VerifyIncList(std::lround(zero * VAL_10), {0}, Err::ILLEGAL_CONST_VALUE);
    AdvSimdSc2RegMiscSz1x(u, opcode, vd, vn);
}

// advanced simd three same fp16
void CodeGenerator::AdvSimd3SameFp16(uint32_t u, uint32_t a, uint32_t opcode, const VRegVec &vd, const VRegVec &vn,
    const VRegVec &vm)
{
    uint32_t q = GenQ(vd);
    uint32_t code = Concat({LeftShift(q, BITFIELD_30), LeftShift(u, BITFIELD_29), LeftShift(0xe, BITFIELD_24),
        LeftShift(a, BITFIELD_23), LeftShift(0x2, BITFIELD_21), LeftShift(vm.GetIdx(), BITFIELD_16),
        LeftShift(opcode, BITFIELD_11), LeftShift(1, BITFIELD_10), LeftShift(vn.GetIdx(), BITFIELD_5),
        LeftShift(vd.GetIdx(), 0)});
    WriteData(code);
}

// Advanced SIMD scalar three same
void CodeGenerator::AdvSimdSc3SameSz0x(uint32_t u, uint32_t opcode, const VRegSc &vd, const VRegSc &vn,
    const VRegSc &vm)
{
    uint32_t size = GenSize(vd) & 1;
    uint32_t code = Concat({LeftShift(1, BITFIELD_30), LeftShift(u, BITFIELD_29), LeftShift(0xf, BITFIELD_25),
        LeftShift(size, BITFIELD_22), LeftShift(1, BITFIELD_21), LeftShift(vm.GetIdx(), BITFIELD_16),
        LeftShift(opcode, BITFIELD_11), LeftShift(1, BITFIELD_10), LeftShift(vn.GetIdx(), BITFIELD_5),
        LeftShift(vd.GetIdx(), 0)});
    WriteData(code);
}

// advanced simd scalar pairwize
void CodeGenerator::AdvSimdScPairwise(uint32_t u, uint32_t size, uint32_t opcode, const VRegSc &vd, const VRegVec &vn)
{
    uint32_t code = Concat({LeftShift(1, BITFIELD_30), LeftShift(u, BITFIELD_29), LeftShift(0xf, BITFIELD_25),
        LeftShift(size, BITFIELD_22), LeftShift(0x3, BITFIELD_20), LeftShift(opcode, BITFIELD_12),
        LeftShift(0x2, BITFIELD_10), LeftShift(vn.GetIdx(), BITFIELD_5), LeftShift(vd.GetIdx(), 0)});
    WriteData(code);
}

// Advanced SIMD shift by immediate
void CodeGenerator::AdvSimdShImm(uint32_t u, uint32_t opcode, const VRegVec &vd, const VRegVec &vn, uint32_t sh)
{
    bool vdSel = (opcode != 0x14);
    uint32_t q = (vdSel) ? GenQ(vd) : GenQ(vn);
    uint32_t size = (vdSel) ? GenSize(vd) : GenSize(vn);

    bool lsh = (opcode == 0xa || opcode == 0xc || opcode == 0xe || opcode == 0x14); // left shift
    uint32_t base = (vdSel) ? vd.GetBit() : vn.GetBit();
    uint32_t imm = (lsh) ? (sh + base) : ((base << 1) - sh);
    uint32_t immh = (0x1 << size) | Field(imm, size + OFFSET_TWO, BITFIELD_3);
    uint32_t immb = Field(imm, BITFIELD_2, BITFIELD_0);

    VerifyIncRange(sh, (OFFSET_ONE - lsh), (base - lsh), Err::ILLEGAL_CONST_RANGE);

    uint32_t code = Concat({LeftShift(q, BITFIELD_30), LeftShift(u, BITFIELD_29), LeftShift(0xf, BITFIELD_24),
        LeftShift(immh, BITFIELD_19), LeftShift(immb, BITFIELD_16), LeftShift(opcode, BITFIELD_11),
        LeftShift(0x1, BITFIELD_10), LeftShift(vn.GetIdx(), BITFIELD_5), LeftShift(vd.GetIdx(), BITFIELD_0)});
    WriteData(code);
}

// Floating-piont data-processing (1 source)
void CodeGenerator::FpDataProc1Reg(uint32_t m, uint32_t s, uint32_t type, uint32_t opcode,
    const VRegSc &vd, const VRegSc &vn)
{
    uint32_t code = Concat({LeftShift(m, BITFIELD_31), LeftShift(s, BITFIELD_29), LeftShift(0xf, BITFIELD_25),
        LeftShift(type, BITFIELD_22), LeftShift(0x1, BITFIELD_21), LeftShift(opcode, BITFIELD_15),
        LeftShift(0x1, BITFIELD_14), LeftShift(vn.GetIdx(), BITFIELD_5), LeftShift(vd.GetIdx(), BITFIELD_0)});
    WriteData(code);
}

// Floating-piont data-processing (2 source)
void CodeGenerator::FpDataProc2Reg(
    uint32_t m, uint32_t s, uint32_t type, uint32_t opcode, const VRegSc &vd, const VRegSc &vn, const VRegSc &vm)
{
    uint32_t code = Concat({LeftShift(m, BITFIELD_31), LeftShift(s, BITFIELD_29), LeftShift(0xf, BITFIELD_25),
        LeftShift(type, BITFIELD_22), LeftShift(0x1, BITFIELD_21), LeftShift(vm.GetIdx(), BITFIELD_16),
        LeftShift(opcode, BITFIELD_12), LeftShift(0x2, BITFIELD_10), LeftShift(vn.GetIdx(), BITFIELD_5),
        LeftShift(vd.GetIdx(), BITFIELD_0)});
    WriteData(code);
}

// Advanced SIMD vector x indexed element
void CodeGenerator::AdvSimdVecXindElemEnc(uint32_t q, uint32_t u, uint32_t size, uint32_t opcode,
    const VRegVec &vd, const VRegVec &vn, const VRegElem &vm)
{
    bool ucmla = (u == 1 && (opcode & 0x9) == 1);
    uint32_t bits = (vm.GetBit() == BITFIELD_8) ? BITFIELD_32 : (ucmla) ? vm.GetBit() * 2 : vm.GetBit();
    uint32_t eidx = vm.GetElemIdx();
    uint32_t h = (bits == BITFIELD_16) ? Field(eidx, BITFIELD_2, BITFIELD_2)
                                       : (bits == 32) ? Field(eidx, BITFIELD_1, BITFIELD_1)
                                                      : Field(eidx, BITFIELD_0, BITFIELD_0);
    uint32_t l = (bits == BITFIELD_16) ? Field(eidx, BITFIELD_1, BITFIELD_1)
                                       : (bits == BITFIELD_32) ? Field(eidx, BITFIELD_0, BITFIELD_0) : 0;
    uint32_t m = (bits == BITFIELD_16) ? Field(eidx, BITFIELD_0, BITFIELD_0)
                                       : Field(vm.GetIdx(), BITFIELD_4, BITFIELD_4);
    uint32_t vmidx = vm.GetIdx() & Ones(BITFIELD_4);

    if (bits == BITFIELD_16) {
        VerifyIncRange(vm.GetIdx(), BITFIELD_0, BITFIELD_15, Err::ILLEGAL_REG_IDX);
    } else {
        VerifyIncRange(vm.GetIdx(), BITFIELD_0, BITFIELD_31, Err::ILLEGAL_REG_IDX);
    }

    uint32_t code = Concat({LeftShift(q, BITFIELD_30), LeftShift(u, BITFIELD_29), LeftShift(0xf, BITFIELD_24),
        LeftShift(size, BITFIELD_22), LeftShift(l, BITFIELD_21), LeftShift(m, BITFIELD_20),
        LeftShift(vmidx, BITFIELD_16), LeftShift(opcode, BITFIELD_12), LeftShift(h, BITFIELD_11),
        LeftShift(vn.GetIdx(), BITFIELD_5), LeftShift(vd.GetIdx(), BITFIELD_0)});
    WriteData(code);
}

void CodeGenerator::AdvSimdVecXindElemSz(uint32_t u, uint32_t size, uint32_t opcode, const VRegVec &vd,
    const VRegVec &vn, const VRegElem &vm)
{
    uint32_t q = GenQ(vd);
    AdvSimdVecXindElemEnc(q, u, size, opcode, vd, vn, vm);
}

// Floating-piont immediate
void CodeGenerator::FpImm(uint32_t m, uint32_t s, uint32_t type, const VRegSc &vd, double imm)
{
    uint32_t imm8 = Code8bitFloat(imm);
    uint32_t code = Concat({LeftShift(m, BITFIELD_31), LeftShift(s, BITFIELD_29), LeftShift(0xf, BITFIELD_25),
        LeftShift(type, BITFIELD_22), LeftShift(0x1, BITFIELD_21), LeftShift(imm8, BITFIELD_13),
        LeftShift(0x1, BITFIELD_12), LeftShift(vd.GetIdx(), BITFIELD_0)});
    WriteData(code);
}

// Floating-piont compare
void CodeGenerator::FpComp(uint32_t m, uint32_t s, uint32_t type, uint32_t op, uint32_t opcode2,
    const VRegSc &vn, const VRegSc &vm)
{
    uint32_t code = Concat({LeftShift(m, BITFIELD_31), LeftShift(s, BITFIELD_29), LeftShift(0xf, BITFIELD_25),
        LeftShift(type, BITFIELD_22), LeftShift(0x1, BITFIELD_21), LeftShift(vm.GetIdx(), BITFIELD_16),
        LeftShift(op, BITFIELD_14), LeftShift(0x1, BITFIELD_13), LeftShift(vn.GetIdx(), BITFIELD_5),
        LeftShift(opcode2, BITFIELD_0)});
    WriteData(code);
}

void CodeGenerator::FpComp(uint32_t m, uint32_t s, uint32_t type, uint32_t op, uint32_t opcode2,
    const VRegSc &vn, double imm)
{
    VerifyIncList(std::lround(imm), {0}, Err::ILLEGAL_CONST_VALUE);
    uint32_t code = Concat({LeftShift(m, BITFIELD_31), LeftShift(s, BITFIELD_29), LeftShift(0xf, BITFIELD_25),
        LeftShift(type, BITFIELD_22), LeftShift(0x1, BITFIELD_21), LeftShift(op, BITFIELD_14),
        LeftShift(0x1, BITFIELD_13), LeftShift(vn.GetIdx(), BITFIELD_5), LeftShift(opcode2, BITFIELD_0)});
    WriteData(code);
}

void CodeGenerator::Ld1(const VReg16BList &vt, const AdrNoOfs &adr)
{
    AdvSimdLdStMultiStructForLd1St1(VAL_1, VAL_2, vt, adr);
}

void CodeGenerator::Ld1(const VReg16BList &vt, const AdrPostReg &adr)
{
    AdvSimdLdStMultiStructPostRegForLd1St1(VAL_1, VAL_2, vt, adr);
}

void CodeGenerator::Ld1(const VReg16BList &vt, const AdrPostImm &adr)
{
    AdvSimdLdStMultiStructPostImmForLd1St1(VAL_1, VAL_2, vt, adr);
}

void CodeGenerator::Ld1(const VReg8HList &vt, const AdrNoOfs &adr)
{
    AdvSimdLdStMultiStructForLd1St1(VAL_1, VAL_2, vt, adr);
}

void CodeGenerator::Ld1(const VReg8HList &vt, const AdrPostReg &adr)
{
    AdvSimdLdStMultiStructPostRegForLd1St1(VAL_1, VAL_2, vt, adr);
}

void CodeGenerator::Ld1(const VReg8HList &vt, const AdrPostImm &adr)
{
    AdvSimdLdStMultiStructPostImmForLd1St1(VAL_1, VAL_2, vt, adr);
}

void CodeGenerator::Ld1(const VReg4SList &vt, const AdrNoOfs &adr)
{
    AdvSimdLdStMultiStructForLd1St1(VAL_1, VAL_2, vt, adr);
}

void CodeGenerator::Ld1(const VReg4SList &vt, const AdrPostReg &adr)
{
    AdvSimdLdStMultiStructPostRegForLd1St1(VAL_1, VAL_2, vt, adr);
}

void CodeGenerator::Ld1(const VReg4SList &vt, const AdrPostImm &adr)
{
    AdvSimdLdStMultiStructPostImmForLd1St1(VAL_1, VAL_2, vt, adr);
}

void CodeGenerator::St1(const VReg16BList &vt, const AdrNoOfs &adr)
{
    AdvSimdLdStMultiStructForLd1St1(VAL_0, VAL_2, vt, adr);
}

void CodeGenerator::St1(const VReg16BList &vt, const AdrPostReg &adr)
{
    AdvSimdLdStMultiStructPostRegForLd1St1(VAL_0, VAL_2, vt, adr);
}

void CodeGenerator::St1(const VReg16BList &vt, const AdrPostImm &adr)
{
    AdvSimdLdStMultiStructPostImmForLd1St1(VAL_0, VAL_2, vt, adr);
}

void CodeGenerator::St1(const VReg8HList &vt, const AdrNoOfs &adr)
{
    AdvSimdLdStMultiStructForLd1St1(VAL_0, VAL_2, vt, adr);
}

void CodeGenerator::St1(const VReg8HList &vt, const AdrPostReg &adr)
{
    AdvSimdLdStMultiStructPostRegForLd1St1(VAL_0, VAL_2, vt, adr);
}

void CodeGenerator::St1(const VReg8HList &vt, const AdrPostImm &adr)
{
    AdvSimdLdStMultiStructPostImmForLd1St1(VAL_0, VAL_2, vt, adr);
}

void CodeGenerator::St1(const VReg4SList &vt, const AdrNoOfs &adr)
{
    AdvSimdLdStMultiStructForLd1St1(VAL_0, VAL_2, vt, adr);
}

void CodeGenerator::St1(const VReg4SList &vt, const AdrPostReg &adr)
{
    AdvSimdLdStMultiStructPostRegForLd1St1(VAL_0, VAL_2, vt, adr);
}

void CodeGenerator::St1(const VReg4SList &vt, const AdrPostImm &adr)
{
    AdvSimdLdStMultiStructPostImmForLd1St1(VAL_0, VAL_2, vt, adr);
}

void CodeGenerator::St1(const VRegBElem &vt, const AdrNoOfs &adr)
{
    AdvSimdLdStSingleStruct(VAL_0, VAL_0, VAL_1, vt, adr);
}

void CodeGenerator::St1(const VRegBElem &vt, const AdrPostReg &adr)
{
    AdvSimdLdStSingleStructPostReg(VAL_0, VAL_0, VAL_1, vt, adr);
}

void CodeGenerator::St1(const VRegBElem &vt, const AdrPostImm &adr)
{
    AdvSimdLdStSingleStructPostImm(VAL_0, VAL_0, VAL_1, vt, adr);
}

void CodeGenerator::St1(const VRegHElem &vt, const AdrNoOfs &adr)
{
    AdvSimdLdStSingleStruct(VAL_0, VAL_0, VAL_1, vt, adr);
}

void CodeGenerator::St1(const VRegHElem &vt, const AdrPostReg &adr)
{
    AdvSimdLdStSingleStructPostReg(VAL_0, VAL_0, VAL_1, vt, adr);
}

void CodeGenerator::St1(const VRegHElem &vt, const AdrPostImm &adr)
{
    AdvSimdLdStSingleStructPostImm(VAL_0, VAL_0, VAL_1, vt, adr);
}

void CodeGenerator::St1(const VRegSElem &vt, const AdrNoOfs &adr)
{
    AdvSimdLdStSingleStruct(VAL_0, VAL_0, VAL_1, vt, adr);
}

void CodeGenerator::St1(const VRegSElem &vt, const AdrPostReg &adr)
{
    AdvSimdLdStSingleStructPostReg(VAL_0, VAL_0, VAL_1, vt, adr);
}

void CodeGenerator::St1(const VRegSElem &vt, const AdrPostImm &adr)
{
    AdvSimdLdStSingleStructPostImm(VAL_0, VAL_0, VAL_1, vt, adr);
}

void CodeGenerator::Ld4(const VRegDElem &vt, const AdrNoOfs &adr)
{
    AdvSimdLdStSingleStruct(VAL_1, VAL_1, VAL_4, vt, adr);
}

void CodeGenerator::Ld1(const VRegBElem &vt, const AdrNoOfs &adr)
{
    AdvSimdLdStSingleStruct(VAL_1, VAL_0, VAL_1, vt, adr);
}

void CodeGenerator::Ld1(const VRegBElem &vt, const AdrPostReg &adr)
{
    AdvSimdLdStSingleStructPostReg(VAL_1, VAL_0, VAL_1, vt, adr);
}

void CodeGenerator::Ld1(const VRegBElem &vt, const AdrPostImm &adr)
{
    AdvSimdLdStSingleStructPostImm(VAL_1, VAL_0, VAL_1, vt, adr);
}

void CodeGenerator::Ld1(const VRegHElem &vt, const AdrNoOfs &adr)
{
    AdvSimdLdStSingleStruct(VAL_1, VAL_0, VAL_1, vt, adr);
}

void CodeGenerator::Ld1(const VRegHElem &vt, const AdrPostReg &adr)
{
    AdvSimdLdStSingleStructPostReg(VAL_1, VAL_0, VAL_1, vt, adr);
}

void CodeGenerator::Ld1(const VRegHElem &vt, const AdrPostImm &adr)
{
    AdvSimdLdStSingleStructPostImm(VAL_1, VAL_0, VAL_1, vt, adr);
}

void CodeGenerator::Ld1(const VRegSElem &vt, const AdrNoOfs &adr)
{
    AdvSimdLdStSingleStruct(VAL_1, VAL_0, VAL_1, vt, adr);
}

void CodeGenerator::Ld1(const VRegSElem &vt, const AdrPostReg &adr)
{
    AdvSimdLdStSingleStructPostReg(VAL_1, VAL_0, VAL_1, vt, adr);
}

void CodeGenerator::Ld1(const VRegSElem &vt, const AdrPostImm &adr)
{
    AdvSimdLdStSingleStructPostImm(VAL_1, VAL_0, VAL_1, vt, adr);
}

void CodeGenerator::Ld1r(const VReg4SList &vt, const AdrNoOfs &adr)
{
    AdvSimdLdRepSingleStruct(VAL_1, VAL_0, VAL_6, VAL_0, vt, adr);
}

void CodeGenerator::St4(const VRegDElem &vt, const AdrPostReg &adr)
{
    AdvSimdLdStSingleStructPostReg(VAL_0, VAL_1, VAL_4, vt, adr);
}

void CodeGenerator::St4(const VRegDElem &vt, const AdrPostImm &adr)
{
    AdvSimdLdStSingleStructPostImm(VAL_0, VAL_1, VAL_4, vt, adr);
}

void CodeGenerator::Ld4(const VRegDElem &vt, const AdrPostImm &adr)
{
    AdvSimdLdStSingleStructPostImm(VAL_1, VAL_1, VAL_4, vt, adr);
}

void CodeGenerator::Stp(const SReg &vt1, const SReg &vt2, const AdrPostImm &adr)
{
    LdStSimdFpPairPostImm(0, vt1, vt2, adr);
}

void CodeGenerator::Stp(const DReg &vt1, const DReg &vt2, const AdrPostImm &adr)
{
    LdStSimdFpPairPostImm(0, vt1, vt2, adr);
}

void CodeGenerator::Stp(const QReg &vt1, const QReg &vt2, const AdrPostImm &adr)
{
    LdStSimdFpPairPostImm(0, vt1, vt2, adr);
}

void CodeGenerator::Stp(const SReg &vt1, const SReg &vt2, const AdrImm &adr)
{
    LdStSimdFpPair(0, vt1, vt2, adr);
}

void CodeGenerator::Stp(const DReg &vt1, const DReg &vt2, const AdrImm &adr)
{
    LdStSimdFpPair(0, vt1, vt2, adr);
}

void CodeGenerator::Stp(const QReg &vt1, const QReg &vt2, const AdrImm &adr)
{
    LdStSimdFpPair(0, vt1, vt2, adr);
}

void CodeGenerator::Stp(const DReg &vt1, const DReg &vt2, const AdrPreImm &adr)
{
    LdStSimdFpPairPre(0, vt1, vt2, adr);
}

void CodeGenerator::Ldp(const SReg &vt1, const SReg &vt2, const AdrPostImm &adr)
{
    LdStSimdFpPairPostImm(VAL_1, vt1, vt2, adr);
}

void CodeGenerator::Ldp(const DReg &vt1, const DReg &vt2, const AdrPostImm &adr)
{
    LdStSimdFpPairPostImm(VAL_1, vt1, vt2, adr);
}

void CodeGenerator::Ldp(const QReg &vt1, const QReg &vt2, const AdrPostImm &adr)
{
    LdStSimdFpPairPostImm(VAL_1, vt1, vt2, adr);
}

void CodeGenerator::Ldp(const SReg &vt1, const SReg &vt2, const AdrImm &adr)
{
    LdStSimdFpPair(VAL_1, vt1, vt2, adr);
}

void CodeGenerator::Ldp(const DReg &vt1, const DReg &vt2, const AdrImm &adr)
{
    LdStSimdFpPair(VAL_1, vt1, vt2, adr);
}

void CodeGenerator::Ldp(const QReg &vt1, const QReg &vt2, const AdrImm &adr)
{
    LdStSimdFpPair(VAL_1, vt1, vt2, adr);
}

void CodeGenerator::Ldp(const DReg &vt1, const DReg &vt2, const AdrPreImm &adr)
{
    LdStSimdFpPairPre(VAL_1, vt1, vt2, adr);
}

void CodeGenerator::Str(const BReg &vt, const AdrUimm &adr)
{
    LdStSimdFpUnImm((vt.GetBit() != VAL_128) ? 0 : VAL_2, vt, adr);
}

void CodeGenerator::Str(const HReg &vt, const AdrUimm &adr)
{
    LdStSimdFpUnImm((vt.GetBit() != VAL_128) ? 0 : VAL_2, vt, adr);
}

void CodeGenerator::Str(const SReg &vt, const AdrUimm &adr)
{
    LdStSimdFpUnImm((vt.GetBit() != VAL_128) ? 0 : VAL_2, vt, adr);
}

void CodeGenerator::Str(const DReg &vt, const AdrUimm &adr)
{
    LdStSimdFpUnImm((vt.GetBit() != VAL_128) ? 0 : VAL_2, vt, adr);
}

void CodeGenerator::Str(const QReg &vt, const AdrUimm &adr)
{
    LdStSimdFpUnImm((vt.GetBit() != VAL_128) ? 0 : VAL_2, vt, adr);
}

void CodeGenerator::Ldr(const BReg &vt, const AdrUimm &adr)
{
    LdStSimdFpUnImm((vt.GetBit() != VAL_128) ? VAL_1 : VAL_3, vt, adr);
}

void CodeGenerator::Ldr(const HReg &vt, const AdrUimm &adr)
{
    LdStSimdFpUnImm((vt.GetBit() != VAL_128) ? VAL_1 : VAL_3, vt, adr);
}

void CodeGenerator::Ldr(const SReg &vt, const AdrUimm &adr)
{
    LdStSimdFpUnImm((vt.GetBit() != VAL_128) ? VAL_1 : VAL_3, vt, adr);
}

void CodeGenerator::Ldr(const DReg &vt, const AdrUimm &adr)
{
    LdStSimdFpUnImm((vt.GetBit() != VAL_128) ? VAL_1 : VAL_3, vt, adr);
}

void CodeGenerator::Ldr(const QReg &vt, const AdrUimm &adr)
{
    LdStSimdFpUnImm((vt.GetBit() != VAL_128) ? VAL_1 : VAL_3, vt, adr);
}

void CodeGenerator::Ldur(const QReg &vt, const AdrImm &adr)
{
    LdStSimdFpRegUnsImm((vt.GetBit() != VAL_128) ? VAL_1 : VAL_3, vt, adr);
}

void CodeGenerator::Zip1(const VReg16B &vd, const VReg16B &vn, const VReg16B &vm)
{
    AdvSimdPermute(VAL_3, vd, vn, vm);
}

void CodeGenerator::Zip2(const VReg16B &vd, const VReg16B &vn, const VReg16B &vm)
{
    AdvSimdPermute(VAL_7, vd, vn, vm);
}

void CodeGenerator::Uzp1(const VReg16B &vd, const VReg16B &vn, const VReg16B &vm)
{
    AdvSimdPermute(VAL_1, vd, vn, vm);
}

void CodeGenerator::Uzp1(const VReg8H &vd, const VReg8H &vn, const VReg8H &vm)
{
    AdvSimdPermute(VAL_1, vd, vn, vm);
}

void CodeGenerator::Uzp1(const VReg2S &vd, const VReg2S &vn, const VReg2S &vm)
{
    AdvSimdPermute(VAL_1, vd, vn, vm);
}

void CodeGenerator::Uzp1(const VReg4S &vd, const VReg4S &vn, const VReg4S &vm)
{
    AdvSimdPermute(VAL_1, vd, vn, vm);
}

void CodeGenerator::Uzp1(const VReg2D &vd, const VReg2D &vn, const VReg2D &vm)
{
    AdvSimdPermute(VAL_1, vd, vn, vm);
}

void CodeGenerator::Trn1(const VReg2D &vd, const VReg2D &vn, const VReg2D &vm)
{
    AdvSimdPermute(VAL_2, vd, vn, vm);
}

void CodeGenerator::Uzp2(const VReg2S &vd, const VReg2S &vn, const VReg2S &vm)
{
    AdvSimdPermute(VAL_5, vd, vn, vm);
}

void CodeGenerator::Uzp2(const VReg4S &vd, const VReg4S &vn, const VReg4S &vm)
{
    AdvSimdPermute(VAL_5, vd, vn, vm);
}

void CodeGenerator::Uzp2(const VReg2D &vd, const VReg2D &vn, const VReg2D &vm)
{
    AdvSimdPermute(VAL_5, vd, vn, vm);
}

void CodeGenerator::Trn2(const VReg2D &vd, const VReg2D &vn, const VReg2D &vm)
{
    AdvSimdPermute(VAL_6, vd, vn, vm);
}

void CodeGenerator::Ext(const VReg16B &vd, const VReg16B &vn, const VReg16B &vm, const uint32_t index)
{
    AdvSimdExtract(0, vd, vn, vm, index);
}

void CodeGenerator::Dup(const VReg16B &vd, const VRegBElem &vn)
{
    AdvSimdCopyDupElem(0, 0, vd, vn);
}

void CodeGenerator::Dup(const VReg4S &vd, const VRegSElem &vn)
{
    AdvSimdCopyDupElem(0, 0, vd, vn);
}

void CodeGenerator::Dup(const VReg16B &vd, const WReg &rn)
{
    AdvSimdCopyDupGen(0, 0, vd, rn);
}

void CodeGenerator::Dup(const VReg4S &vd, const WReg &rn)
{
    AdvSimdCopyDupGen(0, 0, vd, rn);
}

void CodeGenerator::Mov(const WReg &rd, const VRegSElem &vn)
{
    AdvSimdCopyMov(0, VAL_7, rd, vn);
}

void CodeGenerator::Ins(const VRegBElem &vd, const WReg &rn)
{
    AdvSimdCopyInsGen(0, VAL_3, vd, rn);
}

void CodeGenerator::Mov(const VRegDElem &vd, const XReg &rn)
{
    AdvSimdCopyInsGen(0, VAL_3, vd, rn);
}

void CodeGenerator::Mov(const VRegSElem &vd, const WReg &rn)
{
    AdvSimdCopyInsGen(0, VAL_3, vd, rn);
}

void CodeGenerator::Ins(const VRegSElem &vd, const VRegSElem &vn)
{
    AdvSimdCopyElemIns(VAL_1, vd, vn);
}

void CodeGenerator::Ins(const VRegDElem &vd, const VRegDElem &vn)
{
    AdvSimdCopyElemIns(VAL_1, vd, vn);
}

void CodeGenerator::Mov(const VRegSElem &vd, const VRegSElem &vn)
{
    AdvSimdCopyElemIns(VAL_1, vd, vn);
}

void CodeGenerator::Bfcvtn(const VReg4H &vd, const VReg4S &vn)
{
    AdvSimd2RegMisc(0, 0, VAL_22, vd, vn);
}

void CodeGenerator::Scvtf(const VReg4S &vd, const VReg4S &vn)
{
    AdvSimd2RegMiscSz0x(0, VAL_29, vd, vn);
}

void CodeGenerator::Fcvtzs(const VReg4S &vd, const VReg4S &vn)
{
    AdvSimd2RegMiscSz1x(0, VAL_27, vd, vn);
}

void CodeGenerator::Frinti(const VReg4S &vd, const VReg4S &vn)
{
    AdvSimd2RegMiscSz1x(VAL_1, VAL_25, vd, vn);
}

void CodeGenerator::Fsqrt(const VReg4S &vd, const VReg4S &vn)
{
    AdvSimd2RegMiscSz1x(VAL_1, VAL_31, vd, vn);
}

void CodeGenerator::Fmov(const VReg4S &vd, const double imm)
{
    AdvSimdModiImmFmov(0, 0, vd, imm);
}

void CodeGenerator::Fmov(const SReg &vd, const double imm)
{
    FpImm(VAL_0, VAL_0, VAL_0, vd, imm);
}

void CodeGenerator::Fmov(const DReg &vd, const double imm)
{
    FpImm(VAL_0, VAL_0, VAL_1, vd, imm);
}

void CodeGenerator::Fmov(const HReg &vd, const double imm)
{
    FpImm(VAL_0, VAL_0, VAL_3, vd, imm);
}

void CodeGenerator::Addv(const SReg &vd, const VReg4S &vn)
{
    AdvSimdAcrossLanes(0, VAL_27, vd, vn);
}

void CodeGenerator::Smax(const VReg16B &vd, const VReg16B &vn, const VReg16B &vm)
{
    AdvSimd3Same(0, VAL_12, vd, vn, vm);
}

void CodeGenerator::Smax(const VReg4S &vd, const VReg4S &vn, const VReg4S &vm)
{
    AdvSimd3Same(0, VAL_12, vd, vn, vm);
}

void CodeGenerator::Smin(const VReg4S &vd, const VReg4S &vn, const VReg4S &vm)
{
    AdvSimd3Same(0, VAL_13, vd, vn, vm);
}

void CodeGenerator::Add(const VReg4S &vd, const VReg4S &vn, const VReg4S &vm)
{
    AdvSimd3Same(0, VAL_16, vd, vn, vm);
}

void CodeGenerator::Umin(const VReg16B &vd, const VReg16B &vn, const VReg16B &vm)
{
    AdvSimd3Same(VAL_1, VAL_13, vd, vn, vm);
}

void CodeGenerator::Sub(const VReg4S &vd, const VReg4S &vn, const VReg4S &vm)
{
    AdvSimd3Same(VAL_1, VAL_16, vd, vn, vm);
}

void CodeGenerator::Fmaxnm(const VReg4S &vd, const VReg4S &vn, const VReg4S &vm)
{
    AdvSimd3SameSz0x(0, VAL_24, vd, vn, vm);
}

void CodeGenerator::Fmla(const VReg4S &vd, const VReg4S &vn, const VReg4S &vm)
{
    AdvSimd3SameSz0x(0, VAL_25, vd, vn, vm);
}

void CodeGenerator::Fmla(const VReg4S &vd, const VReg4S &vn, const VRegSElem &vm)
{
    AdvSimdVecXindElemSz(0, VAL_2, VAL_1, vd, vn, vm);
}

void CodeGenerator::Smmla(const VReg4S &vd, const VReg16B &vn, const VReg16B &vm)
{
    uint32_t code = Concat({LeftShift(VAL_1, BITFIELD_30), LeftShift(VAL_0, BITFIELD_29),
        LeftShift(0x74, BITFIELD_21), LeftShift(vm.GetIdx(), BITFIELD_16),
        LeftShift(0xa, BITFIELD_12), LeftShift(0x0, BITFIELD_11), LeftShift(VAL_1, BITFIELD_10),
        LeftShift(vn.GetIdx(), BITFIELD_5), LeftShift(vd.GetIdx(), BITFIELD_0)});
    WriteData(code);
}

void CodeGenerator::Usmmla(const VReg4S &vd, const VReg16B &vn, const VReg16B &vm)
{
    uint32_t code = Concat({LeftShift(VAL_1, BITFIELD_30), LeftShift(VAL_0, BITFIELD_29),
        LeftShift(0x74, BITFIELD_21), LeftShift(vm.GetIdx(), BITFIELD_16),
        LeftShift(VAL_10, BITFIELD_12), LeftShift(VAL_1, BITFIELD_11), LeftShift(VAL_1, BITFIELD_10),
        LeftShift(vn.GetIdx(), BITFIELD_5), LeftShift(vd.GetIdx(), BITFIELD_0)});
    WriteData(code);
}

void CodeGenerator::Ummla(const VReg4S &vd, const VReg16B &vn, const VReg16B &vm)
{
    uint32_t code = Concat({LeftShift(VAL_1, BITFIELD_30), LeftShift(VAL_1, BITFIELD_29),
        LeftShift(0x74, BITFIELD_21), LeftShift(vm.GetIdx(), BITFIELD_16),
        LeftShift(0xa, BITFIELD_12), LeftShift(VAL_0, BITFIELD_11), LeftShift(VAL_1, BITFIELD_10),
        LeftShift(vn.GetIdx(), BITFIELD_5), LeftShift(vd.GetIdx(), BITFIELD_0)});
    WriteData(code);
}

void CodeGenerator::Fadd(const VReg4S &vd, const VReg4S &vn, const VReg4S &vm)
{
    AdvSimd3SameSz0x(0, VAL_26, vd, vn, vm);
}

void CodeGenerator::Fmax(const VReg4S &vd, const VReg4S &vn, const VReg4S &vm)
{
    AdvSimd3SameSz0x(0, VAL_30, vd, vn, vm);
}

void CodeGenerator::Fmul(const VReg4S &vd, const VReg4S &vn, const VReg4S &vm)
{
    AdvSimd3SameSz0x(1, VAL_27, vd, vn, vm);
}

void CodeGenerator::Fmul(const VReg4S &vd, const VReg4S &vn, const VRegSElem &vm)
{
    AdvSimdVecXindElemSz(0, VAL_2, VAL_9, vd, vn, vm);
}

void CodeGenerator::Fdiv(const VReg4S &vd, const VReg4S &vn, const VReg4S &vm)
{
    AdvSimd3SameSz0x(1, VAL_31, vd, vn, vm);
}

void CodeGenerator::Fsub(const VReg4S &vd, const VReg4S &vn, const VReg4S &vm)
{
    AdvSimd3SameSz1x(0, VAL_26, vd, vn, vm);
}

void CodeGenerator::Fmin(const VReg4S &vd, const VReg4S &vn, const VReg4S &vm)
{
    AdvSimd3SameSz1x(0, VAL_30, vd, vn, vm);
}

void CodeGenerator::Bic(const VReg16B &vd, const VReg16B &vn, const VReg16B &vm)
{
    AdvSimd3SameSz(0, VAL_1, VAL_3, vd, vn, vm);
}

void CodeGenerator::Orr(const VReg16B &vd, const VReg16B &vn, const VReg16B &vm)
{
    AdvSimd3SameSz(0, VAL_2, VAL_3, vd, vn, vm);
}

void CodeGenerator::Mov(const VReg8B &vd, const VReg8B &vn)
{
    AdvSimd3SameSz(0, VAL_2, VAL_3, vd, vn, vn);
}

void CodeGenerator::Mov(const VReg16B &vd, const VReg16B &vn)
{
    AdvSimd3SameSz(0, VAL_2, VAL_3, vd, vn, vn);
}

void CodeGenerator::Eor(const VReg16B &vd, const VReg16B &vn, const VReg16B &vm)
{
    AdvSimd3SameSz(VAL_1, 0, VAL_3, vd, vn, vm);
}

void CodeGenerator::Movi(const VReg4S &vd, const uint32_t imm8, const ShMod mod, const uint32_t sh)
{
    AdvSimdModiImmMoviMvni(0, 0, vd, imm8, mod, sh);
}

void CodeGenerator::Movi(const VReg16B &vd, const uint32_t imm8, const ShMod mod, const uint32_t sh)
{
    AdvSimdModiImmMoviMvni(0, 0, vd, imm8, mod, sh);
}

void CodeGenerator::Sxtl(const VReg8H &vd, const VReg8B &vn)
{
    AdvSimdShImm(0, VAL_20, vd, vn, 0);
}

void CodeGenerator::Sxtl(const VReg4S &vd, const VReg4H &vn)
{
    AdvSimdShImm(0, VAL_20, vd, vn, 0);
}

void CodeGenerator::Scvtf(const VReg4S &vd, const VReg4S &vn, const uint32_t sh)
{
    AdvSimdShImm(0, VAL_28, vd, vn, sh);
}

void CodeGenerator::Uxtl(const VReg8H &vd, const VReg8B &vn)
{
    AdvSimdShImm(VAL_1, VAL_20, vd, vn, 0);
}

void CodeGenerator::Uxtl(const VReg4S &vd, const VReg4H &vn)
{
    AdvSimdShImm(VAL_1, VAL_20, vd, vn, 0);
}

void CodeGenerator::Fadd(const SReg &vd, const SReg &vn, const SReg &vm)
{
    FpDataProc2Reg(0, 0, 0, VAL_2, vd, vn, vm);
}

void CodeGenerator::Fcvt(const HReg &vd, const SReg &vn)
{
    FpDataProc1Reg(0, 0, 0, VAL_7, vd, vn);
}

void CodeGenerator::Sshr(const VReg16B &vd, const VReg16B &vn, const uint32_t sh)
{
    AdvSimdShImm(VAL_0, VAL_0, vd, vn, sh);
}

void CodeGenerator::Sshr(const VReg4S &vd, const VReg4S &vn, const uint32_t sh)
{
    AdvSimdShImm(VAL_0, VAL_0, vd, vn, sh);
}

void CodeGenerator::Ushr(const VReg16B &vd, const VReg16B &vn, const uint32_t sh)
{
    AdvSimdShImm(VAL_1, VAL_0, vd, vn, sh);
}

void CodeGenerator::Ushr(const VReg4S &vd, const VReg4S &vn, const uint32_t sh)
{
    AdvSimdShImm(VAL_1, VAL_0, vd, vn, sh);
}

void CodeGenerator::Shl(const VReg16B &vd, const VReg16B &vn, const uint32_t sh)
{
    AdvSimdShImm(VAL_0, VAL_10, vd, vn, sh);
}

void CodeGenerator::Shl(const VReg4S &vd, const VReg4S &vn, const uint32_t sh)
{
    AdvSimdShImm(VAL_0, VAL_10, vd, vn, sh);
}

void CodeGenerator::Sshl(const VReg16B &vd, const VReg16B &vn, const VReg16B &vm)
{
    AdvSimd3Same(VAL_0, VAL_8, vd, vn, vm);
}

void CodeGenerator::Sshl(const VReg4S &vd, const VReg4S &vn, const VReg4S &vm)
{
    AdvSimd3Same(VAL_0, VAL_8, vd, vn, vm);
}

void CodeGenerator::Ushl(const VReg16B &vd, const VReg16B &vn, const VReg16B &vm)
{
    AdvSimd3Same(VAL_1, VAL_8, vd, vn, vm);
}

void CodeGenerator::Ushl(const VReg4S &vd, const VReg4S &vn, const VReg4S &vm)
{
    AdvSimd3Same(VAL_1, VAL_8, vd, vn, vm);
}

void CodeGenerator::Bif(const VReg8B &vd, const VReg8B &vn, const VReg8B &vm)
{
    AdvSimd3Same(VAL_7, vd, vn, vm);
}

void CodeGenerator::Bif(const VReg16B &vd, const VReg16B &vn, const VReg16B &vm)
{
    AdvSimd3Same(VAL_7, vd, vn, vm);
}

void CodeGenerator::Bit(const VReg8B &vd, const VReg8B &vn, const VReg8B &vm)
{
    AdvSimd3Same(VAL_5, vd, vn, vm);
}

void CodeGenerator::Bit(const VReg16B &vd, const VReg16B &vn, const VReg16B &vm)
{
    AdvSimd3Same(VAL_5, vd, vn, vm);
}

void CodeGenerator::Mvn(const VReg8B &vd, const VReg8B &vn)
{
    AdvSimd2RegMVN(VAL_1, VAL_1, vd, vn);
}

void CodeGenerator::Mvn(const VReg16B &vd, const VReg16B &vn)
{
    AdvSimd2RegMVN(VAL_1, VAL_1, vd, vn);
}

void CodeGenerator::Faddp(const HReg &vd, const VReg2H &vn)
{
    AdvSimdScPairwise(0, 0, 0xd, vd, vn);
}
void CodeGenerator::Faddp(const SReg &vd, const VReg2S &vn)
{
    AdvSimdScPairwise(1, (vd.GetBit() == 0x20) ? 0 : 1, 0xd, vd, vn);
}
void CodeGenerator::Faddp(const VReg4H &vd, const VReg4H &vn, const VReg4H &vm)
{
    AdvSimd3SameFp16(1, 0, 0x2, vd, vn, vm);
}
void CodeGenerator::Faddp(const VReg8H &vd, const VReg8H &vn, const VReg8H &vm)
{
    AdvSimd3SameFp16(1, 0, 0x2, vd, vn, vm);
}
void CodeGenerator::Faddp(const VReg2S &vd, const VReg2S &vn, const VReg2S &vm)
{
    AdvSimd3SameSz0x(1, 0x1a, vd, vn, vm);
}
void CodeGenerator::Faddp(const VReg4S &vd, const VReg4S &vn, const VReg4S &vm)
{
    AdvSimd3SameSz0x(1, 0x1a, vd, vn, vm);
}

void CodeGenerator::Fcmge(const VReg2S &vd, const VReg2S &vn, const double zero)
{
    AdvSimd2RegMiscSz1x(1, 0xc, vd, vn, zero);
}
void CodeGenerator::Fcmge(const VReg4S &vd, const VReg4S &vn, const double zero)
{
    AdvSimd2RegMiscSz1x(1, 0xc, vd, vn, zero);
}
void CodeGenerator::Fcmge(const VReg2S &vd, const VReg2S &vn, const VReg2S &vm)
{
    AdvSimd3SameSz0x(1, 0x1c, vd, vn, vm);
}
void CodeGenerator::Fcmge(const VReg4S &vd, const VReg4S &vn, const VReg4S &vm)
{
    AdvSimd3SameSz0x(1, 0x1c, vd, vn, vm);
}
void CodeGenerator::Fcmge(const SReg &vd, const SReg &vn, const double zero)
{
    AdvSimdSc2RegMiscSz1x(1, 0xc, vd, vn, zero);
}
void CodeGenerator::Fcmge(const SReg &vd, const SReg &vn, const SReg &vm)
{
    AdvSimdSc3SameSz0x(1, 0x1c, vd, vn, vm);
}

void CodeGenerator::Fcmp(const SReg &vn, const SReg &vm)
{
    FpComp(VAL_0, VAL_0, VAL_0, VAL_0, VAL_0, vn, vm);
}

void CodeGenerator::Fcmp(const SReg &vn, const double imm)
{
    FpComp(VAL_0, VAL_0, VAL_0, VAL_0, VAL_8, vn, imm);
}

void CodeGenerator::Fcmp(const DReg &vn, const DReg &vm)
{
    FpComp(VAL_0, VAL_0, VAL_1, VAL_0, VAL_0, vn, vm);
}

void CodeGenerator::Fcmp(const DReg &vn, const double imm)
{
    FpComp(VAL_0, VAL_0, VAL_1, VAL_0, VAL_8, vn, imm);
}

void CodeGenerator::Fcmp(const HReg &vn, const HReg &vm)
{
    FpComp(VAL_0, VAL_0, VAL_3, VAL_0, VAL_0, vn, vm);
}

void CodeGenerator::Fcmp(const HReg &vn, const double imm)
{
    FpComp(VAL_0, VAL_0, VAL_3, VAL_0, VAL_8, vn, imm);
}

void CodeGenerator::Fcmpe(const SReg &vn, const SReg &vm)
{
    FpComp(VAL_0, VAL_0, VAL_0, VAL_0, VAL_16, vn, vm);
}

void CodeGenerator::Fcmpe(const SReg &vn, const double imm)
{
    FpComp(VAL_0, VAL_0, VAL_0, VAL_0, VAL_24, vn, imm);
}

void CodeGenerator::Fcmpe(const DReg &vn, const DReg &vm)
{
    FpComp(VAL_0, VAL_0, VAL_1, VAL_0, VAL_16, vn, vm);
}

void CodeGenerator::Fcmpe(const DReg &vn, const double imm)
{
    FpComp(VAL_0, VAL_0, VAL_1, VAL_0, VAL_24, vn, imm);
}

void CodeGenerator::Fcmpe(const HReg &vn, const HReg &vm)
{
    FpComp(VAL_0, VAL_0, VAL_3, VAL_0, VAL_16, vn, vm);
}

void CodeGenerator::Fcmpe(const HReg &vn, const double imm)
{
    FpComp(VAL_0, VAL_0, VAL_3, VAL_0, VAL_24, vn, imm);
}

} // ARMCG
