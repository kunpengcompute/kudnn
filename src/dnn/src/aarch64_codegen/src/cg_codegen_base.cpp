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

uint32_t CodeGenerator::PCrelAddrEnc(uint32_t op, const XReg &rd, int64_t labelOffset)
{
    int32_t imm = static_cast<int32_t>((op == 1) ? static_cast<uint64_t>(labelOffset) >> BITFIELD_12 : labelOffset);
    uint32_t immlo = Field(imm, BITFIELD_1, 0);
    uint32_t immhi = Field(imm, BITFIELD_20, 2);
    VerifyIncRange(rd.GetIdx(), 0, SP_IDX - 1, Err::ILLEGAL_REG_IDX);
    VerifyIncRange(imm, -1 * (1 << BITFIELD_20), Ones(BITFIELD_20), Err::ILLEGAL_IMM_RANGE, true);
    return Concat({LeftShift(op, BITFIELD_31), LeftShift(immlo, BITFIELD_29), LeftShift(0x10, BITFIELD_24),
                   LeftShift(immhi, BITFIELD_5), LeftShift(rd.GetIdx(), 0)});
}

void CodeGenerator::PCrelAddr(uint32_t op, const XReg &rd, const Label &label)
{
    auto encFunc = [this, op, rd](int64_t labelOffset) {
        return PCrelAddrEnc(op, rd, labelOffset);
    };
    JmpLabel jmpL = JmpLabel(encFunc, size_);
    uint32_t code = PCrelAddrEnc(op, rd, GenLabelOffset(label, jmpL));
    WriteData(code);
}

// Add/subtract (immediate)
void CodeGenerator::AddSubImm(uint32_t op, uint32_t s, const RReg &rd, const RReg &rn, uint32_t imm, uint32_t sh)
{
    uint32_t sf = GenSf(rd);
    uint32_t imm12 = imm & Ones(BITFIELD_12);
    uint32_t shF = (sh == BITFIELD_12) ? 1 : 0;

    VerifyIncRange(imm, 0, Ones(BITFIELD_12), Err::ILLEGAL_IMM_RANGE);
    VerifyIncList(sh, {0, BITFIELD_12}, Err::ILLEGAL_CONST_VALUE);

    uint32_t code = Concat({LeftShift(sf, BITFIELD_31), LeftShift(op, BITFIELD_30), LeftShift(s, BITFIELD_29),
                            LeftShift(0x11, BITFIELD_24), LeftShift(shF, BITFIELD_22), LeftShift(imm12, BITFIELD_10),
                            LeftShift(rn.GetIdx(), BITFIELD_5), LeftShift(rd.GetIdx(), 0)});
    WriteData(code);
}

// Logical (immediate)
void CodeGenerator::LogicalImm(uint32_t opc, const RReg &rd, const RReg &rn, uint64_t imm, bool alias)
{
    uint32_t sf = GenSf(rd);
    uint32_t nImmrImms = GenNImmrImms(imm, rd.GetBit());

    if (!alias && opc == VAL_3) {
        VerifyIncRange(rd.GetIdx(), 0, SP_IDX - 1, Err::ILLEGAL_REG_IDX);
    }
    if (!alias && opc == 1) {
        VerifyIncRange(rn.GetIdx(), 0, SP_IDX - 1, Err::ILLEGAL_REG_IDX);
    }

    uint32_t code =
        Concat({LeftShift(sf, BITFIELD_31), LeftShift(opc, BITFIELD_29), LeftShift(0x24, BITFIELD_23),
                LeftShift(nImmrImms, BITFIELD_10), LeftShift(rn.GetIdx(), BITFIELD_5), LeftShift(rd.GetIdx(), 0)});
    WriteData(code);
}

// Move wide(immediate)
void CodeGenerator::MvWideImm(uint32_t opc, const RReg &rd, uint32_t imm, uint32_t sh)
{
    uint32_t sf = GenSf(rd);
    uint32_t hw = Field(sh, BITFIELD_5, BITFIELD_4);
    uint32_t imm16 = imm & 0xffff;

    if (sf == 0) {
        VerifyIncList(sh, {0, BITFIELD_16}, Err::ILLEGAL_CONST_VALUE);
    } else {
        VerifyIncList(sh, {0, BITFIELD_16, BITFIELD_32, BITFIELD_48}, Err::ILLEGAL_CONST_VALUE);
    }
    VerifyIncRange(imm, 0, Ones(BITFIELD_16), Err::ILLEGAL_IMM_RANGE);
    VerifyIncRange(rd.GetIdx(), 0, SP_IDX - 1, Err::ILLEGAL_REG_IDX);

    uint32_t code = Concat({LeftShift(sf, BITFIELD_31), LeftShift(opc, BITFIELD_29), LeftShift(0x25, BITFIELD_23),
                            LeftShift(hw, BITFIELD_21), LeftShift(imm16, BITFIELD_5), LeftShift(rd.GetIdx(), 0)});
    WriteData(code);
}

// Move (immediate) can be encoded to three different instructions
// MOV (bitmask immediate), which is an alias of ORR
// MOV (inverted wide immediate), which is an alias of MOVN
// MOV (wide immediate), which is an alias of MOVZ
void CodeGenerator::MvImm(const RReg &rd, uint64_t imm)
{
    uint32_t rdBit = rd.GetBit();
    uint32_t hw = 0;
    uint32_t invHw = 0;
    uint32_t validField[OFFSET_FOUR] = {0};
    uint32_t imm16 = 0;
    uint32_t invImm16 = 0;
    uint32_t fieldCount = 0;
    uint32_t invFieldCount = 0;

    if (imm == 0) {
        MvWideImm(BITFIELD_2, rd, 0, 0);
    } else if ((rdBit == BITFIELD_64 && imm == ~uint64_t(0)) ||
               (rdBit == BITFIELD_32 && ((imm & uint64_t(0xffffffff)) == uint64_t(0xffffffff)))) {
        MvWideImm(0, rd, 0, 0);
    } else if (CheckMovAliasMovz(imm, rdBit, hw, fieldCount, imm16, validField)) {
        MvWideImm(BITFIELD_2, rd, imm16, hw << BITFIELD_4);
    } else if (CheckMovAliasMovn(~imm, rdBit, invHw, invFieldCount, invImm16)) {
        MvWideImm(0, rd, invImm16, invHw << BITFIELD_4);
    } else if (CheckMovAliasOrr(imm, rdBit)) {
        LogicalImm(1, rd, RReg(BITFIELD_31, rdBit), imm, true);
    } else {
        MovzPlusSuccessiveMovk(imm, rd, validField);
    }
}

bool CodeGenerator::CheckMovAliasMovz(uint64_t imm, uint32_t rdBit, uint32_t &hw, uint32_t &fieldCount, uint32_t &imm16,
                                      uint32_t (&validField)[VAL_4])
{
    for (uint32_t i = 0; i < rdBit / BITFIELD_16; ++i) {
        if (Field(imm, BITFIELD_15 + i * BITFIELD_16, i * BITFIELD_16)) {
            validField[i] = 1;
            ++fieldCount;
            hw = i;
            imm16 = Field(imm, BITFIELD_15 + BITFIELD_16 * i, BITFIELD_16 * i);
        }
    }
    return fieldCount < VAL_2 && !(imm16 == 0 && hw != 0);
}

bool CodeGenerator::CheckMovAliasMovn(uint64_t invImm, uint32_t rdBit, uint32_t &invHw, uint32_t &invFieldCount,
                                      uint32_t &invImm16)
{
    for (uint32_t i = 0; i < rdBit / BITFIELD_16; ++i) {
        if (Field(invImm, BITFIELD_15 + i * BITFIELD_16, i * BITFIELD_16)) {
            ++invFieldCount;
            invImm16 = Field(invImm, BITFIELD_15 + BITFIELD_16 * i, BITFIELD_16 * i);
            invHw = i;
        }
    }
    return invFieldCount == 1 &&
           ((!(invImm16 == 0 && invHw != 0) && invImm16 != Ones(BITFIELD_16) && rdBit == BITFIELD_32) ||
            (!(invImm16 == 0 && invHw != 0) && rdBit == BITFIELD_64));
}

bool CodeGenerator::CheckMovAliasOrr(uint64_t imm, uint32_t rdBit)
{
    auto ptnSize = GetPtnSize(imm, rdBit);
    auto ptn = imm & Ones(ptnSize);
    auto rotateNum = GetPtnRotateNum(ptn, ptnSize);
    auto rotatePtn = LRotate(ptn, ptnSize, rotateNum);
    auto oneBitNum = CountOneBit(rotatePtn, ptnSize);
    auto seqOneBitNum = CountSeqOneBit(rotatePtn, ptnSize);
    return oneBitNum == seqOneBitNum;
}

void CodeGenerator::MovzPlusSuccessiveMovk(uint64_t imm, const RReg &rd, uint32_t (&validField)[VAL_4])
{
    bool isFirst = true;
    for (uint32_t i = 0; i < rd.GetBit() / BITFIELD_16; ++i) {
        if (validField[i]) {
            if (isFirst) {
                MvWideImm(BITFIELD_2, rd, Field(imm, BITFIELD_15 + BITFIELD_16 * i, BITFIELD_16 * i), BITFIELD_16 * i);
                isFirst = false;
            } else {
                MvWideImm(BITFIELD_3, rd, Field(imm, BITFIELD_15 + BITFIELD_16 * i, BITFIELD_16 * i), BITFIELD_16 * i);
            }
        }
    }
}

// Bitfield
void CodeGenerator::Bitfield(uint32_t opc, const RReg &rd, const RReg &rn, uint32_t immr, uint32_t imms, bool rnChk)
{
    uint32_t sf = GenSf(rd);
    uint32_t n = sf;

    VerifyIncRange(rd.GetIdx(), 0, SP_IDX - 1, Err::ILLEGAL_REG_IDX);
    if (rnChk) {
        VerifyIncRange(rn.GetIdx(), 0, SP_IDX - 1, Err::ILLEGAL_REG_IDX);
    }
    VerifyIncRange(immr, 0, rd.GetBit() - 1, Err::ILLEGAL_IMM_RANGE);
    VerifyIncRange(imms, 0, rd.GetBit() - 1, Err::ILLEGAL_IMM_RANGE);

    uint32_t code = Concat({LeftShift(sf, BITFIELD_31), LeftShift(opc, BITFIELD_29), LeftShift(0x26, BITFIELD_23),
                            LeftShift(n, BITFIELD_22), LeftShift(immr, BITFIELD_16), LeftShift(imms, BITFIELD_10),
                            LeftShift(rn.GetIdx(), BITFIELD_5), LeftShift(rd.GetIdx(), 0)});
    WriteData(code);
}

// Conditional branch (immediate)
uint32_t CodeGenerator::CondBrImmEnc(uint32_t cond, int64_t labelOffset)
{
    uint32_t imm19 = static_cast<uint32_t>((static_cast<uint64_t>(labelOffset) >> 2) & Ones(BITFIELD_19));
    VerifyIncRange(labelOffset, -1 * (1 << BITFIELD_20), Ones(BITFIELD_20), Err::LABEL_IS_TOO_FAR, true);
    return Concat({LeftShift(0x2a, BITFIELD_25), LeftShift(imm19, BITFIELD_5), LeftShift(cond, 0)});
}

void CodeGenerator::CondBrImm(Cond cond, const Label &label)
{
    auto encFunc = [this, cond](int64_t labelOffset) {
        return CondBrImmEnc(static_cast<uint32_t>(cond), labelOffset);
    };
    JmpLabel jmpL = JmpLabel(encFunc, size_);
    uint32_t code = CondBrImmEnc(static_cast<uint32_t>(cond), GenLabelOffset(label, jmpL));
    WriteData(code);
}

// Hints
void CodeGenerator::Hints(uint32_t crm, uint32_t op2)
{
    uint32_t code = Concat(
        {LeftShift(0xd5032, BITFIELD_12), LeftShift(crm, BITFIELD_8), LeftShift(op2, BITFIELD_5), LeftShift(0x1f, 0)});
    WriteData(code);
}

// Barriers (option)
void CodeGenerator::BarriersOpt(uint32_t op2, BarOpt opt, uint32_t rt)
{
    if (op2 == VAL_6) {
        VerifyIncList(static_cast<uint64_t>(opt), {static_cast<uint64_t>(BarOpt::SY)}, Err::ILLEGAL_BARRIER_OPT);
    }

    uint32_t code = Concat({LeftShift(0xd5033, BITFIELD_12), LeftShift(static_cast<uint32_t>(opt), BITFIELD_8),
                            LeftShift(op2, BITFIELD_5), LeftShift(rt, 0)});
    WriteData(code);
}

// Unconditional branch
void CodeGenerator::UncondBrNoReg(uint32_t opc, uint32_t op2, uint32_t op3, uint32_t rn, uint32_t op4)
{
    uint32_t code = Concat({LeftShift(0x6b, BITFIELD_25), LeftShift(opc, BITFIELD_21), LeftShift(op2, BITFIELD_16),
                            LeftShift(op3, BITFIELD_10), LeftShift(rn, BITFIELD_5), LeftShift(op4, 0)});
    WriteData(code);
}

void CodeGenerator::UncondBr1Reg(uint32_t opc, uint32_t op2, uint32_t op3, const RReg &rn, uint32_t op4)
{
    uint32_t code = Concat({LeftShift(0x6b, BITFIELD_25), LeftShift(opc, BITFIELD_21), LeftShift(op2, BITFIELD_16),
                            LeftShift(op3, BITFIELD_10), LeftShift(rn.GetIdx(), BITFIELD_5), LeftShift(op4, 0)});
    WriteData(code);
}

// Unconditional branch (immediate)
uint32_t CodeGenerator::UncondBrImmEnc(uint32_t op, int64_t labelOffset)
{
    VerifyIncRange(labelOffset, -1 * (BITFIELD_1 << BITFIELD_27), Ones(BITFIELD_27), Err::LABEL_IS_TOO_FAR, true);
    uint32_t imm26 = static_cast<uint32_t>((static_cast<uint64_t>(labelOffset) >> BITFIELD_2) & Ones(BITFIELD_26));
    return Concat({LeftShift(op, BITFIELD_31), LeftShift(BITFIELD_5, BITFIELD_26), LeftShift(imm26, 0)});
}

void CodeGenerator::UncondBrImm(uint32_t op, const Label &label)
{
    auto encFunc = [this, op](int64_t labelOffset) {
        return UncondBrImmEnc(op, labelOffset);
    };
    JmpLabel jmpL = JmpLabel(encFunc, size_);
    uint32_t code = UncondBrImmEnc(op, GenLabelOffset(label, jmpL));
    WriteData(code);
}

// Compare and branch (immediate)
uint32_t CodeGenerator::CompareBrEnc(uint32_t op, const RReg &rt, int64_t labelOffset)
{
    VerifyIncRange(labelOffset, -1 * (1 << BITFIELD_20), Ones(BITFIELD_20), Err::LABEL_IS_TOO_FAR, true);
    uint32_t sf = GenSf(rt);
    uint32_t imm19 = (static_cast<uint64_t>(labelOffset) >> BITFIELD_2) & Ones(BITFIELD_19);
    return Concat({LeftShift(sf, BITFIELD_31), LeftShift(0x1a, BITFIELD_25), LeftShift(op, BITFIELD_24),
                   LeftShift(imm19, BITFIELD_5), LeftShift(rt.GetIdx(), 0)});
}

void CodeGenerator::CompareBr(uint32_t op, const RReg &rt, const Label &label)
{
    auto encFunc = [this, op, rt](int64_t labelOffset) {
        return CompareBrEnc(op, rt, labelOffset);
    };
    JmpLabel jmpL = JmpLabel(encFunc, size_);
    uint32_t code = CompareBrEnc(op, rt, GenLabelOffset(label, jmpL));
    WriteData(code);
}

void CodeGenerator::MrsNZCV(const XReg &rd)
{
    int64_t code = Concat({LeftShift(0x6A9DA10, BITFIELD_5), LeftShift(rd.GetIdx(), BITFIELD_0)});
    WriteData(code);
}

// store exclusive
void CodeGenerator::StExclusive(uint32_t size, uint32_t o0, const WReg ws, const RReg &rt, const AdrImm &adr)
{
    uint32_t l = 0;
    uint32_t o2 = 0;
    uint32_t o1 = 0;
    VerifyIncList(adr.GetImm(), {0}, Err::ILLEGAL_IMM_VALUE);
    VerifyIncRange(rt.GetIdx(), 0, SP_IDX, Err::ILLEGAL_REG_IDX);
    VerifyIncRange(ws.GetIdx(), 0, SP_IDX, Err::ILLEGAL_REG_IDX);
    uint32_t code = Concat({LeftShift(size, BITFIELD_30), LeftShift(0x8, BITFIELD_24), LeftShift(o2, BITFIELD_23),
                            LeftShift(l, BITFIELD_22), LeftShift(o1, BITFIELD_21), LeftShift(ws.GetIdx(), BITFIELD_16),
                            LeftShift(o0, BITFIELD_15), LeftShift(0x1f, BITFIELD_10),
                            LeftShift(adr.GetXn().GetIdx(), BITFIELD_5), LeftShift(rt.GetIdx(), 0)});
    WriteData(code);
}

// load exclusive
void CodeGenerator::LdExclusive(uint32_t size, uint32_t o0, const RReg &rt, const AdrImm &adr)
{
    uint32_t l = 1;
    uint32_t o2 = 0;
    uint32_t o1 = 0;

    VerifyIncRange(rt.GetIdx(), 0, SP_IDX, Err::ILLEGAL_REG_IDX);
    VerifyIncList(adr.GetImm(), {0}, Err::ILLEGAL_IMM_VALUE);

    uint32_t code = Concat({LeftShift(size, BITFIELD_30), LeftShift(0x8, BITFIELD_24), LeftShift(o2, BITFIELD_23),
                            LeftShift(l, BITFIELD_22), LeftShift(o1, BITFIELD_21), LeftShift(0x1f, BITFIELD_16),
                            LeftShift(o0, BITFIELD_15), LeftShift(0x1f, BITFIELD_10),
                            LeftShift(adr.GetXn().GetIdx(), BITFIELD_5), LeftShift(rt.GetIdx(), 0)});
    WriteData(code);
}

// Load/store pair (post-indexed)
void CodeGenerator::LdStRegPairPostImm(uint32_t opc, uint32_t l, const RReg &rt1, const RReg &rt2,
                                       const AdrPostImm &adr)
{
    int32_t imm = adr.GetImm();
    int32_t times = (opc == 0x2) ? OFFSET_TWO : OFFSET_ONE;

    VerifyIncRange(imm, (SIMD_LDRSTR_IMM_RANGE_LOW * times), (SIMD_LDRSTR_IMM_RANGE_HIGH * times),
                   Err::ILLEGAL_IMM_RANGE, true);
    VerifyCond(imm, [=](uint64_t x) { return ((x % (OFFSET_FOUR * times)) == 0); }, Err::ILLEGAL_IMM_COND);

    uint32_t imm7 = (static_cast<uint32_t>(imm) >> (times + OFFSET_ONE)) & Ones(IMM7_BIT_LENGTH);
    uint32_t v = 0x0;

    VerifyIncRange(rt1.GetIdx(), 0, SP_IDX, Err::ILLEGAL_REG_IDX);
    VerifyIncRange(rt2.GetIdx(), 0, SP_IDX, Err::ILLEGAL_REG_IDX);

    uint32_t code = Concat({LeftShift(opc, BITFIELD_30), LeftShift(0x5, BITFIELD_27), LeftShift(v, BITFIELD_26),
                            LeftShift(0x1, BITFIELD_23), LeftShift(l, BITFIELD_22), LeftShift(imm7, BITFIELD_15),
                            LeftShift(rt2.GetIdx(), BITFIELD_10), LeftShift(adr.GetXn().GetIdx(), BITFIELD_5),
                            LeftShift(rt1.GetIdx(), BITFIELD_0)});
    WriteData(code);
}

// Load/store pair (offset)
void CodeGenerator::LdStRegPair(uint32_t opc, uint32_t l, const RReg &rt1, const RReg &rt2, const AdrImm &adr)
{
    int32_t imm = adr.GetImm();
    int32_t times = (opc == 0x2) ? OFFSET_TWO : OFFSET_ONE;

    VerifyIncRange(imm, (SIMD_LDRSTR_IMM_RANGE_LOW * times), (SIMD_LDRSTR_IMM_RANGE_HIGH * times),
                   Err::ILLEGAL_IMM_RANGE, true);
    VerifyCond(imm, [=](uint64_t x) { return ((x % (OFFSET_FOUR * times)) == 0); }, Err::ILLEGAL_IMM_COND);

    uint32_t imm7 = (static_cast<uint32_t>(imm) >> (times + OFFSET_ONE)) & Ones(IMM7_BIT_LENGTH);
    uint32_t v = 0x0;

    VerifyIncRange(rt1.GetIdx(), 0, SP_IDX, Err::ILLEGAL_REG_IDX);
    VerifyIncRange(rt2.GetIdx(), 0, SP_IDX, Err::ILLEGAL_REG_IDX);

    uint32_t code = Concat({LeftShift(opc, BITFIELD_30), LeftShift(0x5, BITFIELD_27), LeftShift(v, BITFIELD_26),
                            LeftShift(0x2, BITFIELD_23), LeftShift(l, BITFIELD_22), LeftShift(imm7, BITFIELD_15),
                            LeftShift(rt2.GetIdx(), BITFIELD_10), LeftShift(adr.GetXn().GetIdx(), BITFIELD_5),
                            LeftShift(rt1.GetIdx(), BITFIELD_0)});
    WriteData(code);
}

// Load/store pair (pre-indexed)
void CodeGenerator::LdStRegPairPre(uint32_t opc, uint32_t l, const RReg &rt1, const RReg &rt2, const AdrPreImm &adr)
{
    int32_t imm = adr.GetImm();
    int32_t times = (opc == 0x2) ? OFFSET_TWO : OFFSET_ONE;

    VerifyIncRange(imm, (SIMD_LDRSTR_IMM_RANGE_LOW * times), (SIMD_LDRSTR_IMM_RANGE_HIGH * times),
                   Err::ILLEGAL_IMM_RANGE, true);
    VerifyCond(imm, [=](uint64_t x) { return ((x % (OFFSET_FOUR * times)) == 0); }, Err::ILLEGAL_IMM_COND);

    uint32_t imm7 = (static_cast<uint32_t>(imm) >> (times + OFFSET_ONE)) & Ones(IMM7_BIT_LENGTH);
    uint32_t v = 0x0;

    VerifyIncRange(rt1.GetIdx(), 0, SP_IDX, Err::ILLEGAL_REG_IDX);
    VerifyIncRange(rt2.GetIdx(), 0, SP_IDX, Err::ILLEGAL_REG_IDX);

    uint32_t code = Concat({LeftShift(opc, BITFIELD_30), LeftShift(0x5, BITFIELD_27), LeftShift(v, BITFIELD_26),
                            LeftShift(0x3, BITFIELD_23), LeftShift(l, BITFIELD_22), LeftShift(imm7, BITFIELD_15),
                            LeftShift(rt2.GetIdx(), BITFIELD_10), LeftShift(adr.GetXn().GetIdx(), BITFIELD_5),
                            LeftShift(rt1.GetIdx(), BITFIELD_0)});
    WriteData(code);
}

// Load/store register (unscaled immediate)
void CodeGenerator::LdStRegUnsImm(uint32_t size, uint32_t opc, const RReg &rt, const AdrImm &adr)
{
    int imm = adr.GetImm();
    uint32_t imm9 = static_cast<uint32_t>(imm) & Ones(IMM9_BIT_LENGTH);

    VerifyIncRange(imm, IMM9_RANGE_LOW, IMM9_RANGE_HIGH, Err::ILLEGAL_IMM_RANGE, true);
    VerifyIncRange(rt.GetIdx(), 0, SP_IDX, Err::ILLEGAL_REG_IDX);

    uint32_t v = 0x0;
    uint32_t code = Concat({LeftShift(size, BITFIELD_30), LeftShift(0x7, BITFIELD_27), LeftShift(v, BITFIELD_26),
                            LeftShift(opc, BITFIELD_22), LeftShift(imm9, BITFIELD_12),
                            LeftShift(adr.GetXn().GetIdx(), BITFIELD_5), LeftShift(rt.GetIdx(), BITFIELD_0)});
    WriteData(code);
}

// Load/store register (immediate post-indexed)
void CodeGenerator::LdStRegPostImm(uint32_t size, uint32_t opc, const RReg &rt, const AdrPostImm &adr)
{
    int imm = adr.GetImm();
    uint32_t imm9 = static_cast<uint32_t>(imm) & Ones(IMM9_BIT_LENGTH);

    VerifyIncRange(imm, IMM9_RANGE_LOW, IMM9_RANGE_HIGH, Err::ILLEGAL_IMM_RANGE, true);
    VerifyIncRange(rt.GetIdx(), 0, SP_IDX, Err::ILLEGAL_REG_IDX);

    uint32_t v = 0x0;
    uint32_t code = Concat({LeftShift(size, BITFIELD_30), LeftShift(0x7, BITFIELD_27), LeftShift(v, BITFIELD_26),
                            LeftShift(opc, BITFIELD_22), LeftShift(imm9, BITFIELD_12), LeftShift(0x1, BITFIELD_10),
                            LeftShift(adr.GetXn().GetIdx(), BITFIELD_5), LeftShift(rt.GetIdx(), BITFIELD_0)});
    WriteData(code);
}

// Load/store register (immediate pre-indexed)
void CodeGenerator::LdStRegPre(uint32_t size, uint32_t opc, const RReg &rt, const AdrPreImm &adr)
{
    int imm = adr.GetImm();
    uint32_t imm9 = static_cast<uint32_t>(imm) & Ones(IMM9_BIT_LENGTH);

    VerifyIncRange(imm, IMM9_RANGE_LOW, IMM9_RANGE_HIGH, Err::ILLEGAL_IMM_RANGE, true);
    VerifyIncRange(rt.GetIdx(), 0, SP_IDX, Err::ILLEGAL_REG_IDX);

    uint32_t v = 0x0;
    uint32_t code = Concat({LeftShift(size, BITFIELD_30), LeftShift(0x7, BITFIELD_27), LeftShift(v, BITFIELD_26),
                            LeftShift(opc, BITFIELD_22), LeftShift(imm9, BITFIELD_12), LeftShift(0x3, BITFIELD_10),
                            LeftShift(adr.GetXn().GetIdx(), BITFIELD_5), LeftShift(rt.GetIdx(), BITFIELD_0)});
    WriteData(code);
}

void CodeGenerator::AtomicMemOp(uint32_t size, uint32_t v, uint32_t a, uint32_t r, uint32_t o3, uint32_t opc,
                                const RReg &rs, const RReg &rt, const AdrNoOfs &adr)
{
    uint32_t code =
        Concat({LeftShift(size, BITFIELD_30), LeftShift(0x7, BITFIELD_27), LeftShift(v, BITFIELD_26),
                LeftShift(a, BITFIELD_23), LeftShift(r, BITFIELD_22), LeftShift(0x1, BITFIELD_21),
                LeftShift(rs.GetIdx(), BITFIELD_16), LeftShift(o3, BITFIELD_15), LeftShift(opc, BITFIELD_12),
                LeftShift(adr.GetXn().GetIdx(), BITFIELD_5), LeftShift(rt.GetIdx(), BITFIELD_0)});
    WriteData(code);
}

// loat/store register (unsigned immediate)
void CodeGenerator::LdStRegUnImm(uint32_t size, uint32_t opc, const RReg &rt, const AdrUimm &adr)
{
    uint32_t imm = adr.GetImm();
    int32_t times = 0x1 << size;
    uint32_t imm12 = (imm >> size) & Ones(IMM12_BIT_LENGTH);

    VerifyIncRange(imm, 0, IMM13_RANGE_HIGH * times, Err::ILLEGAL_IMM_RANGE);
    VerifyCond(imm, [=](uint64_t x) { return ((x & Ones(size)) == 0); }, Err::ILLEGAL_IMM_COND);
    VerifyIncRange(rt.GetIdx(), 0, SP_IDX, Err::ILLEGAL_REG_IDX);

    uint32_t code = Concat({LeftShift(size, BITFIELD_30), LeftShift(0x7, BITFIELD_27), LeftShift(0x0, BITFIELD_26),
                            LeftShift(0x1, BITFIELD_24), LeftShift(opc, BITFIELD_22), LeftShift(imm12, BITFIELD_10),
                            LeftShift(adr.GetXn().GetIdx(), BITFIELD_5), LeftShift(rt.GetIdx(), BITFIELD_0)});

    WriteData(code);
}

// loat/store register (unsigned immediate)
void CodeGenerator::PfRegImm(Prfop prfop, const AdrUimm &adr)
{
    uint32_t imm = adr.GetImm();
    int32_t times = VectorComputation_ELT_BITLEN_B;
    uint32_t imm12 = (imm >> OFFSET_THREE) & Ones(IMM12_BIT_LENGTH);

    VerifyIncRange(imm, 0, IMM13_RANGE_HIGH * times, Err::ILLEGAL_IMM_RANGE);
    VerifyCond(imm, [=](uint64_t x) { return ((x & Ones(IMM3_BIT_LENGTH)) == 0); }, Err::ILLEGAL_IMM_COND);

    uint32_t code =
        Concat({LeftShift(0x3, BITFIELD_30), LeftShift(0x7, BITFIELD_27), LeftShift(0x0, BITFIELD_26),
                LeftShift(0x1, BITFIELD_24), LeftShift(0x2, BITFIELD_22), LeftShift(imm12, BITFIELD_10),
                LeftShift(adr.GetXn().GetIdx(), BITFIELD_5), LeftShift(static_cast<uint32_t>(prfop), BITFIELD_0)});
    WriteData(code);
}

// Data processing (2 source)
void CodeGenerator::DataProc2Src(uint32_t opcode, const RReg &rd, const RReg &rn, const RReg &rm)
{
    uint32_t sf = GenSf(rm);
    uint32_t s = 0;

    VerifyCond(
        SP_IDX, [=](uint64_t x) { return rd.GetIdx() < x || rn.GetIdx() < x || rm.GetIdx() < x; },
        Err::ILLEGAL_REG_IDX);

    uint32_t code = Concat({LeftShift(sf, BITFIELD_31), LeftShift(s, BITFIELD_29), LeftShift(0xd6, BITFIELD_21),
                            LeftShift(rm.GetIdx(), BITFIELD_16), LeftShift(opcode, BITFIELD_10),
                            LeftShift(rn.GetIdx(), BITFIELD_5), LeftShift(rd.GetIdx(), BITFIELD_0)});
    WriteData(code);
}

// Logical (shifted register)
void CodeGenerator::LogicalShiftReg(uint32_t opc, uint32_t n, const RReg &rd, const RReg &rn, const RReg &rm,
                                    ShMod shmod, uint32_t sh)
{
    uint32_t sf = GenSf(rd);
    uint32_t imm6 = sh & Ones(IMM6_BIT_LENGTH);

    VerifyIncRange(sh, 0, (VectorComputation_ELT_BITLEN_S << sf) - OFFSET_ONE, Err::ILLEGAL_CONST_RANGE);
    VerifyCond(
        SP_IDX, [=](uint64_t x) { return rd.GetIdx() < x || rn.GetIdx() < x || rm.GetIdx() < x; },
        Err::ILLEGAL_REG_IDX);

    uint32_t code = Concat({LeftShift(sf, BITFIELD_31), LeftShift(opc, BITFIELD_29), LeftShift(0xa, BITFIELD_24),
                            LeftShift(static_cast<uint32_t>(shmod), BITFIELD_22), LeftShift(n, BITFIELD_21),
                            LeftShift(rm.GetIdx(), BITFIELD_16), LeftShift(imm6, BITFIELD_10),
                            LeftShift(rn.GetIdx(), BITFIELD_5), LeftShift(rd.GetIdx(), BITFIELD_0)});
    WriteData(code);
}

// Move (register) alias of ADD,ORR
void CodeGenerator::MvReg(const RReg &rd, const RReg &rn)
{
    if (rd.GetIdx() == SP_IDX || rn.GetIdx() == SP_IDX) {
        // alias of ADD
        AddSubImm(0, 0, rd, rn, 0, 0);
    } else {
        // alias of ORR
        LogicalShiftReg(0x1, 0, rd, RReg(SP_IDX, rd.GetBit()), rn, ShMod::LSL, 0);
    }
}

// Add/subtract (shifted register)
void CodeGenerator::AddSubShiftReg(uint32_t opc, uint32_t s, const RReg &rd, const RReg &rn, const RReg &rm,
                                   ShMod shmod, uint32_t sh, bool alias)
{
    uint32_t rdSp = (rd.GetIdx() == SP_IDX);
    uint32_t rnSp = (rn.GetIdx() == SP_IDX);
    if (((rdSp + rnSp) >= OFFSET_ONE + static_cast<uint32_t>(alias)) && shmod == ShMod::LSL) {
        AddSubExtReg(opc, s, rd, rn, rm, ExtMod::EXT_LSL, sh);
        return;
    }

    if (shmod == ShMod::NONE) {
        shmod = ShMod::LSL;
    }

    uint32_t sf = GenSf(rd);
    uint32_t imm6 = sh & Ones(IMM6_BIT_LENGTH);

    VerifyIncRange(sh, 0, (VectorComputation_ELT_BITLEN_S << sf) - OFFSET_ONE, Err::ILLEGAL_CONST_RANGE);
    if (!(alias && s == 0x1)) {
        VerifyIncRange(rd.GetIdx(), 0, SP_IDX - OFFSET_ONE, Err::ILLEGAL_REG_IDX);
    }
    if (!(alias && opc == 0x1)) {
        VerifyIncRange(rn.GetIdx(), 0, SP_IDX - OFFSET_ONE, Err::ILLEGAL_REG_IDX);
    }

    VerifyIncRange(rm.GetIdx(), 0, SP_IDX - OFFSET_ONE, Err::ILLEGAL_REG_IDX);
    uint32_t code = Concat({LeftShift(sf, BITFIELD_31), LeftShift(opc, BITFIELD_30), LeftShift(s, BITFIELD_29),
                            LeftShift(0xb, BITFIELD_24), LeftShift(static_cast<uint32_t>(shmod), BITFIELD_22),
                            LeftShift(rm.GetIdx(), BITFIELD_16), LeftShift(imm6, BITFIELD_10),
                            LeftShift(rn.GetIdx(), BITFIELD_5), LeftShift(rd.GetIdx(), BITFIELD_0)});
    WriteData(code);
}

// Add/subtract (extended register)
void CodeGenerator::AddSubExtReg(uint32_t opc, uint32_t s, const RReg &rd, const RReg &rn, const RReg &rm,
                                 ExtMod extmod, uint32_t sh)
{
    uint32_t sf = GenSf(rd);
    uint32_t opt = 0;
    uint32_t imm3 = sh & Ones(IMM3_BIT_LENGTH);

    VerifyIncRange(sh, 0, UIMM2_RANGE_HIGH, Err::ILLEGAL_CONST_RANGE);

    uint32_t option = (extmod == ExtMod::EXT_LSL && sf == 0) ? 2 :
                      (extmod == ExtMod::EXT_LSL && sf == 1) ? 3 :
                                                               static_cast<int>(extmod);
    uint32_t code =
        Concat({LeftShift(sf, BITFIELD_31), LeftShift(opc, BITFIELD_30), LeftShift(s, BITFIELD_29),
                LeftShift(0xb, BITFIELD_24), LeftShift(opt, BITFIELD_22), LeftShift(0x1, BITFIELD_21),
                LeftShift(rm.GetIdx(), BITFIELD_16), LeftShift(option, BITFIELD_13), LeftShift(imm3, BITFIELD_10),
                LeftShift(rn.GetIdx(), BITFIELD_5), LeftShift(rd.GetIdx(), BITFIELD_0)});
    WriteData(code);
}

// Conditional select
void CodeGenerator::DataProc3Reg(uint32_t op54, uint32_t op31, uint32_t o0, const RReg &rd, const RReg &rn,
                                 const RReg &rm, const RReg &ra)
{
    uint32_t sf = GenSf(rd);
    VerifyCond(
        SP_IDX, [=](uint64_t x) { return rd.GetIdx() < x || rn.GetIdx() < x || rm.GetIdx() < x || ra.GetIdx() < x; },
        Err::ILLEGAL_REG_IDX);

    uint32_t code = Concat({LeftShift(sf, BITFIELD_31), LeftShift(op54, BITFIELD_29), LeftShift(0x1b, BITFIELD_24),
                            LeftShift(op31, BITFIELD_21), LeftShift(rm.GetIdx(), BITFIELD_16),
                            LeftShift(o0, BITFIELD_15), LeftShift(ra.GetIdx(), BITFIELD_10),
                            LeftShift(rn.GetIdx(), BITFIELD_5), LeftShift(rd.GetIdx(), BITFIELD_0)});
    WriteData(code);
}

void CodeGenerator::Adr(const XReg &xd, const Label &label)
{
    PCrelAddr(0, xd, label);
}

void CodeGenerator::Add(const XReg &rd, const XReg &rn, const uint32_t imm, const uint32_t sh)
{
    AddSubImm(0, 0, rd, rn, imm, sh);
}

void CodeGenerator::Adds(const XReg &rd, const XReg &rn, const uint32_t imm, const uint32_t sh)
{
    AddSubImm(0, VAL_1, rd, rn, imm, sh);
}

void CodeGenerator::Subs(const XReg &rd, const XReg &rn, const uint32_t imm, const uint32_t sh)
{
    AddSubImm(VAL_1, VAL_1, rd, rn, imm, sh);
}

void CodeGenerator::Sub(const WReg &rd, const WReg &rn, const uint32_t imm, const uint32_t sh)
{
    AddSubImm(VAL_1, 0, rd, rn, imm, sh);
}

void CodeGenerator::Sub(const XReg &rd, const XReg &rn, const uint32_t imm, const uint32_t sh)
{
    AddSubImm(VAL_1, 0, rd, rn, imm, sh);
}

void CodeGenerator::Cmp(const WReg &rn, const uint32_t imm, const uint32_t sh)
{
    AddSubImm(VAL_1, VAL_1, WReg(RINDEX31), rn, imm, sh);
}

void CodeGenerator::Cmp(const XReg &rn, const uint32_t imm, const uint32_t sh)
{
    AddSubImm(VAL_1, VAL_1, XReg(RINDEX31), rn, imm, sh);
}

void CodeGenerator::And(const XReg &rd, const XReg &rn, const uint64_t imm)
{
    LogicalImm(0, rd, rn, imm);
}

void CodeGenerator::Tst(const XReg &rn, const uint64_t imm)
{
    LogicalImm(VAL_3, XReg(RINDEX31), rn, imm, true);
}

void CodeGenerator::Movn(const XReg &rd, const uint32_t imm, const uint32_t sh)
{
    MvWideImm(0, rd, imm, sh);
}

void CodeGenerator::Movz(const WReg &rd, const uint32_t imm, const uint32_t sh)
{
    MvWideImm(VAL_2, rd, imm, sh);
}

void CodeGenerator::Movz(const XReg &rd, const uint32_t imm, const uint32_t sh)
{
    MvWideImm(VAL_2, rd, imm, sh);
}

void CodeGenerator::Movk(const WReg &rd, const uint32_t imm, const uint32_t sh)
{
    MvWideImm(VAL_3, rd, imm, sh);
}

void CodeGenerator::Movk(const XReg &rd, const uint32_t imm, const uint32_t sh)
{
    MvWideImm(VAL_3, rd, imm, sh);
}

void CodeGenerator::Mov(const WReg &rd, const uint64_t imm)
{
    MvImm(rd, imm);
}

void CodeGenerator::Mov(const XReg &rd, const uint64_t imm)
{
    MvImm(rd, imm);
}

void CodeGenerator::Sxtb(const WReg &rd, const WReg &rn)
{
    Bitfield(0, rd, rn, 0, VAL_7);
}

void CodeGenerator::Lsl(const XReg &rd, const XReg &rn, const uint32_t sh)
{
    Bitfield(VAL_2, rd, rn, (((-1) * sh) % VAL_64) & Ones(VAL_6), VAL_63 - sh);
}

void CodeGenerator::Lsr(const XReg &rd, const XReg &rn, const uint32_t sh)
{
    Bitfield(VAL_2, rd, rn, sh, VAL_63);
}

void CodeGenerator::Uxtb(const WReg &rd, const WReg &rn)
{
    Bitfield(VAL_2, rd, rn, 0, VAL_7);
}

void CodeGenerator::B(const Cond cond, const Label &label)
{
    CondBrImm(cond, label);
}

void CodeGenerator::Nop()
{
    Hints(0, 0);
}

void CodeGenerator::Yield()
{
    Hints(0, VAL_1);
}

void CodeGenerator::Dmb(const BarOpt opt)
{
    BarriersOpt(VAL_5, opt, VAL_31);
}

void CodeGenerator::Ret()
{
    UncondBrNoReg(VAL_2, VAL_31, 0, VAL_30, 0);
}

void CodeGenerator::Br(const XReg &rn)
{
    UncondBr1Reg(0, VAL_31, 0, rn, 0);
}

void CodeGenerator::B(const Label &label)
{
    UncondBrImm(0, label);
}

void CodeGenerator::Bl(const Label &label)
{
    UncondBrImm(VAL_1, label);
}

void CodeGenerator::Cbz(const XReg &rt, const Label &label)
{
    CompareBr(0, rt, label);
}

void CodeGenerator::Cbnz(const WReg &rt, const Label &label)
{
    CompareBr(VAL_1, rt, label);
}

void CodeGenerator::Cbnz(const XReg &rt, const Label &label)
{
    CompareBr(VAL_1, rt, label);
}

void CodeGenerator::Stlxr(const WReg &ws, const WReg &rt, const AdrImm &adr)
{
    StExclusive(VAL_2, VAL_1, ws, rt, adr);
}

void CodeGenerator::Stlxr(const WReg &ws, const XReg &rt, const AdrImm &adr)
{
    StExclusive(VAL_3, VAL_1, ws, rt, adr);
}

void CodeGenerator::Ldaxr(const XReg &rt, const AdrImm &adr)
{
    LdExclusive(VAL_3, VAL_1, rt, adr);
}

void CodeGenerator::Stp(const XReg &rt1, const XReg &rt2, const AdrPostImm &adr)
{
    LdStRegPairPostImm((rt1.GetBit() == VAL_32) ? 0 : VAL_2, 0, rt1, rt2, adr);
}

void CodeGenerator::Ldp(const XReg &rt1, const XReg &rt2, const AdrPostImm &adr)
{
    LdStRegPairPostImm((rt1.GetBit() == VAL_32) ? 0 : VAL_2, VAL_1, rt1, rt2, adr);
}

void CodeGenerator::Stp(const XReg &rt1, const XReg &rt2, const AdrImm &adr)
{
    LdStRegPair((rt1.GetBit() == VAL_32) ? 0 : VAL_2, 0, rt1, rt2, adr);
}

void CodeGenerator::Stp(const XReg &rt1, const XReg &rt2, const AdrPreImm &adr)
{
    LdStRegPairPre((rt1.GetBit() == VAL_32) ? 0 : VAL_2, 0, rt1, rt2, adr);
}

void CodeGenerator::Ldurb(const WReg &rt, const AdrImm &adr)
{
    LdStRegUnsImm(0, VAL_1, rt, adr);
}

void CodeGenerator::Ldr(const WReg &rt, const AdrPostImm &adr)
{
    LdStRegPostImm(VAL_2, VAL_1, rt, adr);
}

void CodeGenerator::Ldrsw(const XReg &rt, const AdrPostImm &adr)
{
    LdStRegPostImm(VAL_2, VAL_2, rt, adr);
}

void CodeGenerator::Str(const XReg &rt, const AdrPostImm &adr)
{
    LdStRegPostImm(VAL_3, 0, rt, adr);
}

void CodeGenerator::Ldr(const XReg &rt, const AdrPostImm &adr)
{
    LdStRegPostImm(VAL_3, VAL_1, rt, adr);
}

void CodeGenerator::Str(const XReg &rt, const AdrPreImm &adr)
{
    LdStRegPre(VAL_3, 0, rt, adr);
}

void CodeGenerator::Ldaddal(const XReg &rs, const XReg &rt, const AdrNoOfs &adr)
{
    AtomicMemOp(VAL_3, 0, VAL_1, VAL_1, 0, 0, rs, rt, adr);
}

void CodeGenerator::Strb(const WReg &rt, const AdrUimm &adr)
{
    LdStRegUnImm(0, 0, rt, adr);
}

void CodeGenerator::Ldrb(const WReg &rt, const AdrUimm &adr)
{
    LdStRegUnImm(0, VAL_1, rt, adr);
}

void CodeGenerator::Strh(const WReg &rt, const AdrUimm &adr)
{
    LdStRegUnImm(VAL_1, 0, rt, adr);
}

void CodeGenerator::Ldrh(const WReg &rt, const AdrUimm &adr)
{
    LdStRegUnImm(VAL_1, VAL_1, rt, adr);
}

void CodeGenerator::Str(const WReg &rt, const AdrUimm &adr)
{
    LdStRegUnImm(VAL_2, 0, rt, adr);
}

void CodeGenerator::Ldr(const WReg &rt, const AdrUimm &adr)
{
    LdStRegUnImm(VAL_2, VAL_1, rt, adr);
}

void CodeGenerator::Ldrsw(const XReg &rt, const AdrUimm &adr)
{
    LdStRegUnImm(VAL_2, VAL_2, rt, adr);
}

void CodeGenerator::Str(const XReg &rt, const AdrUimm &adr)
{
    LdStRegUnImm(VAL_3, 0, rt, adr);
}

void CodeGenerator::Ldr(const XReg &rt, const AdrUimm &adr)
{
    LdStRegUnImm(VAL_3, 1, rt, adr);
}

void CodeGenerator::Prfm(const Prfop prfop, const AdrUimm &adr)
{
    PfRegImm(prfop, adr);
}

void CodeGenerator::Udiv(const XReg &rd, const XReg &rn, const XReg &rm)
{
    DataProc2Src(VAL_2, rd, rn, rm);
}

void CodeGenerator::Mvn(const XReg &rd, const XReg &rm, const ShMod shmod, const uint32_t sh)
{
    LogicalShiftReg(VAL_1, VAL_1, rd, XReg(RINDEX31), rm, shmod, sh);
}

void CodeGenerator::Eor(const XReg &rd, const XReg &rn, const XReg &rm, const ShMod shmod, const uint32_t sh)
{
    LogicalShiftReg(VAL_2, 0, rd, rn, rm, shmod, sh);
}

void CodeGenerator::Tst(const XReg &rn, const XReg &rm, const ShMod shmod, const uint32_t sh)
{
    LogicalShiftReg(VAL_3, 0, XReg(RINDEX31), rn, rm, shmod, sh);
}

void CodeGenerator::Mov(const XReg &rd, const XReg &rn)
{
    MvReg(rd, rn);
}

void CodeGenerator::Add(const WReg &rd, const WReg &rn, const WReg &rm, const ShMod shmod, const uint32_t sh)
{
    AddSubShiftReg(0, 0, rd, rn, rm, shmod, sh);
}

void CodeGenerator::Add(const XReg &rd, const XReg &rn, const XReg &rm, const ShMod shmod, const uint32_t sh)
{
    AddSubShiftReg(0, 0, rd, rn, rm, shmod, sh);
}

void CodeGenerator::Adds(const XReg &rd, const XReg &rn, const XReg &rm, const ShMod shmod, const uint32_t sh)
{
    AddSubShiftReg(0, VAL_1, rd, rn, rm, shmod, sh);
}

void CodeGenerator::Sub(const XReg &rd, const XReg &rn, const XReg &rm, const ShMod shmod, const uint32_t sh)
{
    AddSubShiftReg(VAL_1, 0, rd, rn, rm, shmod, sh);
}

void CodeGenerator::Subs(const WReg &rd, const WReg &rn, const WReg &rm, const ShMod shmod, const uint32_t sh)
{
    AddSubShiftReg(VAL_1, VAL_1, rd, rn, rm, shmod, sh);
}

void CodeGenerator::Subs(const XReg &rd, const XReg &rn, const XReg &rm, const ShMod shmod, const uint32_t sh)
{
    AddSubShiftReg(VAL_1, VAL_1, rd, rn, rm, shmod, sh);
}

void CodeGenerator::Subs(const WReg &rd, const WReg &rn, const WReg &rm, const ExtMod extmod, const uint32_t sh)
{
    AddSubExtReg(VAL_1, VAL_1, rd, rn, rm, extmod, sh);
}

void CodeGenerator::Subs(const XReg &rd, const XReg &rn, const XReg &rm, const ExtMod extmod, const uint32_t sh)
{
    AddSubExtReg(VAL_1, VAL_1, rd, rn, rm, extmod, sh);
}

void CodeGenerator::Subs(const XReg &rd, const XReg &rn, const WReg &rm, const ExtMod extmod, const uint32_t sh)
{
    AddSubExtReg(VAL_1, VAL_1, rd, rn, rm, extmod, sh);
}

void CodeGenerator::Cmp(const XReg &rn, const XReg &rm, const ShMod shmod, const uint32_t sh)
{
    AddSubShiftReg(VAL_1, VAL_1, XReg(RINDEX31), rn, rm, shmod, sh, true);
}

void CodeGenerator::Add(const XReg &rd, const XReg &rn, const XReg &rm, const ExtMod extmod, const uint32_t sh)
{
    AddSubExtReg(0, 0, rd, rn, rm, extmod, sh);
}

void CodeGenerator::Madd(const XReg &rd, const XReg &rn, const XReg &rm, const XReg &ra)
{
    DataProc3Reg(0, 0, 0, rd, rn, rm, ra);
}

void CodeGenerator::Mul(const XReg &rd, const XReg &rn, const XReg &rm)
{
    DataProc3Reg(0, 0, 0, rd, rn, rm, XReg(RINDEX31));
}

void CodeGenerator::Beq(const Label &label)
{
    B(Cond::EQ, label);
}

void CodeGenerator::Bne(const Label &label)
{
    B(Cond::NE, label);
}

void CodeGenerator::Bmi(const Label &label)
{
    B(Cond::MI, label);
}

void CodeGenerator::Bge(const Label &label)
{
    B(Cond::GE, label);
}

void CodeGenerator::Bgt(const Label &label)
{
    B(Cond::GT, label);
}

void CodeGenerator::Ble(const Label &label)
{
    B(Cond::LE, label);
}

void CodeGenerator::Blt(const Label &label)
{
    B(Cond::LT, label);
}

void CodeGenerator::BFirst(const Label &label)
{
    Bmi(label);
}

} // namespace ARMCG
