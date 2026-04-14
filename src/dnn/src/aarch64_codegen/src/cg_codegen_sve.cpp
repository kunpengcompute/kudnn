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

// VectorComputation Integer Binary Arithmetic - Predicated Group
void CodeGenerator::SveIntBinArPred(uint32_t opc, uint32_t type, const ZRegBase &zd,
    const PRegBase &pg, const ZRegBase &zn)
{
    uint32_t size = GenSize(zd);
    VerifyIncRange(pg.GetIdx(), 0, VectorComputation_HALF_PG_RANGE, Err::ILLEGAL_REG_IDX);
    uint32_t code = Concat({LeftShift(0x4, BITFIELD_24), LeftShift(size, BITFIELD_22), LeftShift(type, BITFIELD_19),
        LeftShift(opc, BITFIELD_16), LeftShift(pg.GetIdx(), BITFIELD_10), LeftShift(zn.GetIdx(), BITFIELD_5),
        LeftShift(zd.GetIdx(), BITFIELD_0)});
    WriteData(code);
}

// VectorComputation bitwize Logical Operation (predicated)
void CodeGenerator::SveBitwiseLOpPred(uint32_t opc, const ZRegBase &zd, const PRegBase &pg, const ZRegBase &zn)
{
    SveIntBinArPred(opc, 0x3, zd, pg, zn);
}

// VectorComputation Integer min/max/diffrence (predicated)
void CodeGenerator::SveIntMinMaxDiffPred(uint32_t opc, uint32_t u, const ZRegBase &zd,
    const PRegBase &pg, const ZRegBase &zn)
{
    SveIntBinArPred(((opc << OFFSET_ONE) | u), 0x1, zd, pg, zn);
}

// VectorComputation Integer multiply/divide vectors (predicated)
void CodeGenerator::SveIntMultDivVecPred(uint32_t opc, uint32_t u, const ZRegBase &zd,
    const PRegBase &pg, const ZRegBase &zn)
{
    SveIntBinArPred(((opc << OFFSET_ONE) | u), 0x2, zd, pg, zn);
}

// VectorComputation Integer Reduction Group
void CodeGenerator::SveIntReduction(uint32_t opc, uint32_t type, const Reg &rd, const PRegBase &pg, const Reg &rn)
{
    uint32_t size = GenSize(rn);
    VerifyIncRange(pg.GetIdx(), 0, VectorComputation_HALF_PG_RANGE, Err::ILLEGAL_REG_IDX);
    uint32_t code = Concat({LeftShift(0x4, BITFIELD_24), LeftShift(size, BITFIELD_22), LeftShift(type, BITFIELD_19),
        LeftShift(opc, BITFIELD_16), LeftShift(0x1, BITFIELD_13), LeftShift(pg.GetIdx(), BITFIELD_10),
        LeftShift(rn.GetIdx(), BITFIELD_5), LeftShift(rd.GetIdx(), BITFIELD_0)});
    WriteData(code);
}

// VectorComputation integer add reduction (predicated)
void CodeGenerator::SveIntAddReductPred(uint32_t opc, uint32_t u, const VRegSc &vd,
    const PRegBase &pg, const ZRegBase &zn)
{
    SveIntReduction(((opc << 1) | u), 0, vd, pg, zn);
}

// VectorComputation constructive prefix (predicated)
void CodeGenerator::SveConstPrefPred(uint32_t opc, const ZRegBase &zd, const PRegBase &pg, const ZRegBase &zn)
{
    SveIntReduction((opc << OFFSET_ONE | (static_cast<uint32_t>(pg.IsM()))), 0x2, zd, pg, zn);
}

// VectorComputation Bitwise Shift - Predicate Group
void CodeGenerator::SveBitShPred(uint32_t opc, uint32_t type, const ZRegBase &zdn,
    const PRegBase &pg, const ZRegBase &zm)
{
    uint32_t size = GenSize(zdn);
    VerifyIncRange(pg.GetIdx(), 0, VectorComputation_HALF_PG_RANGE, Err::ILLEGAL_REG_IDX);
    uint32_t code = Concat({LeftShift(0x4, BITFIELD_24), LeftShift(size, BITFIELD_22), LeftShift(type, BITFIELD_19),
        LeftShift(opc, BITFIELD_16), LeftShift(0x4, BITFIELD_13), LeftShift(pg.GetIdx(), BITFIELD_10),
        LeftShift(zm.GetIdx(), BITFIELD_5), LeftShift(zdn.GetIdx(), BITFIELD_0)});
    WriteData(code);
}

// VectorComputation bitwise shift by immediate (predicated)
void CodeGenerator::SveBitwiseShByImmPred(
    uint32_t opc, uint32_t l, uint32_t u, const ZRegBase &zdn, const PRegBase &pg, uint32_t amount)
{
    bool lsl = (l == 0x1);
    uint32_t size = GenSize(zdn);
    uint32_t imm = (lsl) ? (amount + zdn.GetBit()) : (OFFSET_TWO * zdn.GetBit() - amount);
    uint32_t imm3 = imm & Ones(IMM3_BIT_LENGTH);
    uint32_t tsz = (0x1 << size) | Field(imm, size + OFFSET_TWO, BITFIELD_3);
    uint32_t tszh = Field(tsz, BITFIELD_3, BITFIELD_2);
    uint32_t tszl = Field(tsz, BITFIELD_1, BITFIELD_0);

    VerifyIncRange(pg.GetIdx(), 0, VectorComputation_HALF_PG_RANGE, Err::ILLEGAL_REG_IDX);
    VerifyIncRange(amount, (OFFSET_ONE - lsl), (zdn.GetBit() - lsl), Err::ILLEGAL_CONST_RANGE);

    uint32_t code = Concat({LeftShift(0x4, BITFIELD_24), LeftShift(tszh, BITFIELD_22), LeftShift(opc, BITFIELD_18),
        LeftShift(l, BITFIELD_17), LeftShift(u, BITFIELD_16), LeftShift(0x4, BITFIELD_13),
        LeftShift(pg.GetIdx(), BITFIELD_10), LeftShift(tszl, BITFIELD_8), LeftShift(imm3, BITFIELD_5),
        LeftShift(zdn.GetIdx(), BITFIELD_0)});
    WriteData(code);
}

// VectorComputation bitwise shift by vector (predicated)
void CodeGenerator::SveBitwiseShVecPred(uint32_t opc, const ZRegBase &zdn, const PRegBase &pg, const ZRegBase &zm)
{
    SveBitShPred(opc, 0x2, zdn, pg, zm);
}

// VectorComputation Integer Unary Arithmetic - Predicated Group
void CodeGenerator::SveIntUnaryArPred(uint32_t opc, uint32_t type, const ZRegBase &zd,
    const PRegBase &pg, const ZRegBase &zn)
{
    uint32_t size = GenSize(zd);
    VerifyIncRange(pg.GetIdx(), 0, VectorComputation_HALF_PG_RANGE, Err::ILLEGAL_REG_IDX);
    uint32_t code = Concat({LeftShift(0x4, BITFIELD_24), LeftShift(size, BITFIELD_22), LeftShift(type, BITFIELD_19),
        LeftShift(opc, BITFIELD_16), LeftShift(0x5, BITFIELD_13), LeftShift(pg.GetIdx(), BITFIELD_10),
        LeftShift(zn.GetIdx(), BITFIELD_5), LeftShift(zd.GetIdx(), BITFIELD_0)});
    WriteData(code);
}

// VectorComputation bitwise unary operations (predicated)
void CodeGenerator::SveBitwiseUnaryOpPred(uint32_t opc, const ZRegBase &zdn, const PRegBase &pg, const ZRegBase &zm)
{
    SveIntUnaryArPred(opc, 0x3, zdn, pg, zm);
}

// VectorComputation integer unary operations (predicated)
void CodeGenerator::SveIntUnaryOpPred(uint32_t opc, const ZRegBase &zdn, const PRegBase &pg, const ZRegBase &zm)
{
    SveIntUnaryArPred(opc, 0x2, zdn, pg, zm);
}

// VectorComputation integer add/subtract vectors (unpredicated)
void CodeGenerator::SveIntAddSubUnpred(uint32_t opc, const ZRegBase &zd, const ZRegBase &zn, const ZRegBase &zm)
{
    uint32_t size = GenSize(zd);
    uint32_t code = Concat({LeftShift(0x4, BITFIELD_24), LeftShift(size, BITFIELD_22), LeftShift(0x1, BITFIELD_21),
        LeftShift(zm.GetIdx(), BITFIELD_16), LeftShift(opc, BITFIELD_10),
        LeftShift(zn.GetIdx(), BITFIELD_5), LeftShift(zd.GetIdx(), BITFIELD_0)});
    WriteData(code);
}

// VectorComputation bitwise logical operations (unpredicated)
void CodeGenerator::SveBitwiseLOpUnpred(uint32_t opc, const ZRegBase &zd, const ZRegBase &zn, const ZRegBase &zm)
{
    uint32_t code = Concat({LeftShift(0x4, BITFIELD_24), LeftShift(opc, BITFIELD_22), LeftShift(0x1, BITFIELD_21),
        LeftShift(zm.GetIdx(), BITFIELD_16), LeftShift(0xc, BITFIELD_10),
        LeftShift(zn.GetIdx(), BITFIELD_5), LeftShift(zd.GetIdx(), BITFIELD_0)});
    WriteData(code);
}

// VectorComputation index generation (immediate start, immediate increment)
void CodeGenerator::SveIndexGenImmImmInc(const ZRegBase &zd, int32_t imm1, int32_t imm2)
{
    uint32_t size = GenSize(zd);
    uint32_t imm5b = static_cast<uint32_t>(imm2) & Ones(IMM5_BIT_LENGTH);
    uint32_t imm5 = static_cast<uint32_t>(imm1) & Ones(IMM5_BIT_LENGTH);

    VerifyIncRange(imm1, IMM5_RANGE_LOW, IMM5_RANGE_HIGH, Err::ILLEGAL_IMM_RANGE, true);
    VerifyIncRange(imm2, IMM5_RANGE_LOW, IMM5_RANGE_HIGH, Err::ILLEGAL_IMM_RANGE, true);

    uint32_t code = Concat({LeftShift(0x4, BITFIELD_24), LeftShift(size, BITFIELD_22), LeftShift(0x1, BITFIELD_21),
        LeftShift(imm5b, BITFIELD_16),
        LeftShift(0x10, BITFIELD_10), LeftShift(imm5, BITFIELD_5), LeftShift(zd.GetIdx(), BITFIELD_0)});
    WriteData(code);
}

// VectorComputation stack frame adjustment
void CodeGenerator::SveStackFrameAdjust(uint32_t op, const XReg &xd, const XReg &xn, int32_t imm)
{
    uint32_t imm6 = static_cast<uint32_t>(imm) & Ones(IMM6_BIT_LENGTH);
    VerifyIncRange(imm, IMM6_RANGE_LOW, IMM6_RANGE_HIGH, Err::ILLEGAL_IMM_RANGE, true);
    uint32_t code = Concat({LeftShift(0x8, BITFIELD_23), LeftShift(op, BITFIELD_22), LeftShift(0x1, BITFIELD_21),
        LeftShift(xn.GetIdx(), BITFIELD_16),
        LeftShift(0xa, BITFIELD_11), LeftShift(imm6, BITFIELD_5), LeftShift(xd.GetIdx(), BITFIELD_0)});
    WriteData(code);
}

// VectorComputation bitwise shift by immediate (unpredicated)
void CodeGenerator::SveBitwiseShByImmUnpred(uint32_t opc, const ZRegBase &zd, const ZRegBase &zn, uint32_t amount)
{
    bool lsl = (opc == 0x3);
    uint32_t size = GenSize(zd);
    uint32_t imm = (lsl) ? (amount + zd.GetBit()) : (OFFSET_TWO * zd.GetBit() - amount);
    uint32_t imm3 = imm & Ones(IMM3_BIT_LENGTH);
    uint32_t tsz = (0x1 << size) | Field(imm, size + BITFIELD_2, BITFIELD_3);
    uint32_t tszh = Field(tsz, BITFIELD_3, BITFIELD_2);
    uint32_t tszl = Field(tsz, BITFIELD_1, BITFIELD_0);

    VerifyIncRange(amount, (1 - lsl), (zd.GetBit() - lsl), Err::ILLEGAL_CONST_RANGE);

    uint32_t code = Concat({LeftShift(0x4, BITFIELD_24), LeftShift(tszh, BITFIELD_22), LeftShift(0x1, BITFIELD_21),
        LeftShift(tszl, BITFIELD_19), LeftShift(imm3, BITFIELD_16),
        LeftShift(0x9, BITFIELD_12), LeftShift(opc, BITFIELD_10), LeftShift(zn.GetIdx(), BITFIELD_5),
        LeftShift(zd.GetIdx(), BITFIELD_0)});
    WriteData(code);
}

// VectorComputation Integer Misc - Unpredicated Group
void CodeGenerator::SveIntMiscUnpred(uint32_t size, uint32_t opc, uint32_t type,
    const ZRegBase &zd, const ZRegBase &zn)
{
    uint32_t code = Concat({LeftShift(0x4, BITFIELD_24), LeftShift(size, BITFIELD_22), LeftShift(0x1, BITFIELD_21),
        LeftShift(opc, BITFIELD_16), LeftShift(0xb, BITFIELD_12),
        LeftShift(type, BITFIELD_10), LeftShift(zn.GetIdx(), BITFIELD_5), LeftShift(zd.GetIdx(), BITFIELD_0)});
    WriteData(code);
}

// VectorComputation floating-point exponential accelerator
void CodeGenerator::SveFpExpAccel(uint32_t opc, const ZRegBase &zd, const ZRegBase &zn)
{
    uint32_t size = GenSize(zd);
    SveIntMiscUnpred(size, opc, 0x2, zd, zn);
}

// VectorComputation Element Count Group
void CodeGenerator::SveElemCountGrp(
    uint32_t size, uint32_t op, uint32_t type1, uint32_t type2, const Reg &rd, Pattern pat, ExtMod mod, uint32_t imm)
{
    uint32_t imm4 = (imm - OFFSET_ONE) & Ones(IMM4_BIT_LENGTH);
    VerifyIncList(static_cast<uint64_t>(mod), {static_cast<uint64_t>(ExtMod::MUL)}, Err::ILLEGAL_EXTMOD);
    VerifyIncRange(imm, OFFSET_ONE, UIMM4_RANGE_HIGH, Err::ILLEGAL_IMM_RANGE);
    uint32_t code = Concat({LeftShift(0x4, BITFIELD_24), LeftShift(size, BITFIELD_22), LeftShift(type1, BITFIELD_20),
        LeftShift(imm4, BITFIELD_16), LeftShift(type2, BITFIELD_11), LeftShift(op, BITFIELD_10),
        LeftShift(static_cast<uint32_t>(pat), BITFIELD_5), LeftShift(rd.GetIdx(), BITFIELD_0)});
    WriteData(code);
}

// VectorComputation element count
void CodeGenerator::SveElemCount(
    uint32_t size, uint32_t op, const XReg &xd, Pattern pat, ExtMod mod, uint32_t imm)
{
    SveElemCountGrp(size, op, 0x2, 0x1c, xd, pat, mod, imm);
}

// VectorComputation inc/dec register by element count
void CodeGenerator::SveIncDecRegByElemCount(
    uint32_t size, uint32_t d, const XReg &xd, Pattern pat, ExtMod mod, uint32_t imm)
{
    SveElemCountGrp(size, d, 0x3, 0x1c, xd, pat, mod, imm);
}

// VectorComputation Bitwise Immeidate Group
void CodeGenerator::SveBitwiseImm(uint32_t opc, const ZRegBase &zd, uint64_t imm)
{
    uint32_t imm13 = GenNImmrImms(imm, zd.GetBit());
    uint32_t code = Concat({LeftShift(0x5, BITFIELD_24), LeftShift(opc, BITFIELD_22), LeftShift(imm13, BITFIELD_5),
        LeftShift(zd.GetIdx(), BITFIELD_0)});
    WriteData(code);
}

// VectorComputation bitwise logical with immediate (unpredicated)
void CodeGenerator::SveBitwiseLogicalImmUnpred(uint32_t opc, const ZRegBase &zdn, uint64_t imm)
{
    SveBitwiseImm(opc, zdn, imm);
}

// VectorComputation broadcast bitmask immediate
void CodeGenerator::SveBcBitmaskImm(const ZRegBase &zdn, uint64_t imm)
{
    SveBitwiseImm(0x3, zdn, imm);
}

// VectorComputation copy floating-point immediate (predicated)
void CodeGenerator::SveCopyFpImmPred(const ZRegBase &zd, const PRegBase &pg, double imm)
{
    uint32_t size = GenSize(zd);
    uint32_t imm8 = Code8bitFloat(imm);
    uint32_t code = Concat({LeftShift(0x5, BITFIELD_24), LeftShift(size, BITFIELD_22), LeftShift(0x1, BITFIELD_20),
        LeftShift(pg.GetIdx(), BITFIELD_16), LeftShift(0x6, BITFIELD_13), LeftShift(imm8, BITFIELD_5),
        LeftShift(zd.GetIdx(), BITFIELD_0)});
    WriteData(code);
}

// VectorComputation copy integer immediate (predicated)
void CodeGenerator::SveCopyIntImmPred(const ZRegBase &zd, const PRegBase &pg, uint32_t imm, ShMod mod, uint32_t sh)
{
    VerifyIncList(static_cast<uint64_t>(mod), {static_cast<uint64_t>(ShMod::LSL)}, Err::ILLEGAL_SHMOD);
    VerifyIncList(sh, {0, IMM8_BIT_LENGTH}, Err::ILLEGAL_CONST_VALUE);
    uint32_t size = GenSize(zd);
    uint32_t imm8 = imm & Ones(IMM8_BIT_LENGTH);
    uint32_t type = (pg.IsM() << 0x1) | (sh == IMM8_BIT_LENGTH);
    uint32_t code = Concat({LeftShift(0x5, BITFIELD_24), LeftShift(size, BITFIELD_22), LeftShift(1, BITFIELD_20),
        LeftShift(pg.GetIdx(), BITFIELD_16), LeftShift(type, BITFIELD_13),
        LeftShift(imm8, BITFIELD_5), LeftShift(zd.GetIdx(), BITFIELD_0)});
    WriteData(code);
}

// VectorComputation extract vector (immediate offset)
void CodeGenerator::SveExtVec(const ZRegBase &zdn, const ZRegBase &zm, uint32_t imm)
{
    uint32_t imm8h = Field(imm, BITFIELD_7, BITFIELD_3);
    uint32_t imm8l = Field(imm, BITFIELD_2, BITFIELD_0);
    VerifyIncRange(imm, 0, UIMM8_RANGE_HIGH, Err::ILLEGAL_IMM_RANGE);
    uint32_t code = Concat({LeftShift(0x5, BITFIELD_24), LeftShift(0x1, BITFIELD_21), LeftShift(imm8h, BITFIELD_16),
        LeftShift(imm8l, BITFIELD_10), LeftShift(zm.GetIdx(), BITFIELD_5), LeftShift(zdn.GetIdx(), BITFIELD_0)});
    WriteData(code);
}

// VectorComputation Permute Vector - Unpredicate Group
void CodeGenerator::SvePerVecUnpred(uint32_t size, uint32_t type1, uint32_t type2, const ZRegBase &zd, const Reg &rn)
{
    uint32_t code = Concat({LeftShift(0x5, BITFIELD_24), LeftShift(size, BITFIELD_22), LeftShift(0x1, BITFIELD_21),
        LeftShift(type1, BITFIELD_16), LeftShift(type2, BITFIELD_10),
        LeftShift(rn.GetIdx(), BITFIELD_5), LeftShift(zd.GetIdx(), BITFIELD_0)});
    WriteData(code);
}

// VectorComputation broadcast general register
void CodeGenerator::SveBcGeneralReg(const ZRegBase &zd, const RReg &rn)
{
    uint32_t size = GenSize(zd);
    SvePerVecUnpred(size, 0, 0xe, zd, rn);
}

// VectorComputation broadcast indexed element
void CodeGenerator::SveBcIndexedElem(const ZRegBase &zd, const ZRegElem &zn)
{
    uint32_t eidx = zn.GetElemIdx();
    uint32_t pos = static_cast<uint32_t>(std::log2(zn.GetBit()) - 0x2);
    uint32_t imm = (eidx << pos) | (0x1 << (pos - 0x1));
    uint32_t imm2 = Field(imm, BITFIELD_6, BITFIELD_5);
    uint32_t tsz = Field(imm, BITFIELD_4, BITFIELD_0);

    if (zd.GetBit() == VectorComputation_ELT_BITLEN_Q) {
        VerifyIncList(Field(tsz, BITFIELD_4, BITFIELD_0), {0x10}, Err::ILLEGAL_IMM_COND);
    } else if (zd.GetBit() == VectorComputation_ELT_BITLEN_D) {
        VerifyIncList(Field(tsz, BITFIELD_3, BITFIELD_0), {0x8}, Err::ILLEGAL_IMM_COND);
    } else if (zd.GetBit() == VectorComputation_ELT_BITLEN_S) {
        VerifyIncList(Field(tsz, BITFIELD_2, BITFIELD_0), {0x4}, Err::ILLEGAL_IMM_COND);
    } else if (zd.GetBit() == VectorComputation_ELT_BITLEN_H) {
        VerifyIncList(Field(tsz, BITFIELD_1, BITFIELD_0), {0x2}, Err::ILLEGAL_IMM_COND);
    } else if (zd.GetBit() == VectorComputation_ELT_BITLEN_B) {
        VerifyIncList(Field(tsz, BITFIELD_0, BITFIELD_0), {0x1}, Err::ILLEGAL_IMM_COND);
    }

    SvePerVecUnpred(imm2, tsz, 0x8, zd, zn);
}

// VectorComputation permute predicate elements
void CodeGenerator::SvePermutePredElem(uint32_t opc, uint32_t h, const PRegBase &pd,
    const PRegBase &pn, const PRegBase &pm)
{
    uint32_t size = GenSize(pd);
    uint32_t code = Concat({LeftShift(0x5, BITFIELD_24), LeftShift(size, BITFIELD_22), LeftShift(0x1, BITFIELD_21),
        LeftShift(pm.GetIdx(), BITFIELD_16), LeftShift(0x2, BITFIELD_13), LeftShift(opc, BITFIELD_11),
        LeftShift(h, BITFIELD_10), LeftShift(pn.GetIdx(), BITFIELD_5), LeftShift(pd.GetIdx(), BITFIELD_0)});
    WriteData(code);
}

// VectorComputation permute vector elements
void CodeGenerator::SvePermuteVecElem(uint32_t opc, const ZRegBase &zd, const ZRegBase &zn, const ZRegBase &zm)
{
    uint32_t size = GenSize(zd);
    uint32_t code = Concat({LeftShift(0x5, BITFIELD_24), LeftShift(size, BITFIELD_22), LeftShift(0x1, BITFIELD_21),
        LeftShift(zm.GetIdx(), BITFIELD_16), LeftShift(0x3, BITFIELD_13), LeftShift(opc, BITFIELD_10),
        LeftShift(zn.GetIdx(), BITFIELD_5), LeftShift(zd.GetIdx(), BITFIELD_0)});
    WriteData(code);
}

// VectorComputation copy general register to vector (predicated)
void CodeGenerator::SveCopyGeneralRegToVecPred(const ZRegBase &zd, const PRegBase &pg, const RReg &rn)
{
    uint32_t size = GenSize(zd);
    VerifyIncRange(pg.GetIdx(), 0, VectorComputation_HALF_PG_RANGE, Err::ILLEGAL_REG_IDX);
    uint32_t code = Concat({LeftShift(0x5, BITFIELD_24), LeftShift(size, BITFIELD_22), LeftShift(0x14, BITFIELD_17),
        LeftShift(0x5, BITFIELD_13), LeftShift(pg.GetIdx(), BITFIELD_10), LeftShift(rn.GetIdx(), BITFIELD_5),
        LeftShift(zd.GetIdx(), BITFIELD_0)});
    WriteData(code);
}

// VectorComputation vector splice
void CodeGenerator::SveSelVecSplice(const ZRegBase &zd, const PRegBase &pg, const ZRegBase &zn)
{
    uint32_t size = GenSize(zd);
    VerifyIncRange(pg.GetIdx(), 0, VectorComputation_HALF_PG_RANGE, Err::ILLEGAL_REG_IDX);
    uint32_t code = Concat({LeftShift(0x5, BITFIELD_24), LeftShift(size, BITFIELD_22), LeftShift(0xb, BITFIELD_18),
        LeftShift(0x4, BITFIELD_13), LeftShift(pg.GetIdx(), BITFIELD_10), LeftShift(zn.GetIdx(), BITFIELD_5),
        LeftShift(zd.GetIdx(), BITFIELD_0)});
    WriteData(code);
}

// VectorComputation select vector elements (predicated)
void CodeGenerator::SveSelVecElemPred(const ZRegBase &zd, const PRegBase &pg, const ZRegBase &zn, const ZRegBase &zm)
{
    uint32_t size = GenSize(zd);
    uint32_t code = Concat({LeftShift(0x5, BITFIELD_24), LeftShift(size, 22), LeftShift(0x1, BITFIELD_21),
        LeftShift(zm.GetIdx(), BITFIELD_16), LeftShift(0x3, BITFIELD_14), LeftShift(pg.GetIdx(), BITFIELD_10),
        LeftShift(zn.GetIdx(), BITFIELD_5), LeftShift(zd.GetIdx(), BITFIELD_0)});
    WriteData(code);
}

// VectorComputation Integer Compare - Vector Group
void CodeGenerator::SveIntCompVecGrp(uint32_t opc, uint32_t ne, const PRegBase &pd,
    const PRegBase &pg, const ZRegBase &zn, const ZRegBase &zm)
{
    uint32_t size = GenSize(pd);
    uint32_t code = Concat({LeftShift(0x24, BITFIELD_24), LeftShift(size, BITFIELD_22),
        LeftShift(zm.GetIdx(), BITFIELD_16), LeftShift(opc, BITFIELD_13), LeftShift(pg.GetIdx(), BITFIELD_10),
        LeftShift(zn.GetIdx(), BITFIELD_5), LeftShift(ne, BITFIELD_4), LeftShift(pd.GetIdx(), BITFIELD_0)});
    WriteData(code);
}

// VectorComputation integer compare vectors
void CodeGenerator::SveIntCompVec(uint32_t op, uint32_t o2, uint32_t ne, const PRegBase &pd,
    const PRegBase &pg, const ZRegBase &zn, const ZRegBase &zm)
{
    uint32_t opc = (op << VectorComputation_COMPVEC_2) | o2;
    SveIntCompVecGrp(opc, ne, pd, pg, zn, zm);
}

// VectorComputation predicate logical operations
void CodeGenerator::SvePredLOp(uint32_t op, uint32_t s, uint32_t o2, uint32_t o3, const PRegBase &pd,
    const PRegBase &pg, const PRegBase &pn, const PRegBase &pm)
{
    uint32_t code = Concat({LeftShift(0x25, BITFIELD_24), LeftShift(op, BITFIELD_23), LeftShift(s, BITFIELD_22),
        LeftShift(pm.GetIdx(), BITFIELD_16), LeftShift(0x1, BITFIELD_14), LeftShift(pg.GetIdx(), BITFIELD_10),
        LeftShift(o2, BITFIELD_9), LeftShift(pn.GetIdx(), BITFIELD_5), LeftShift(o3, BITFIELD_4),
        LeftShift(pd.GetIdx(), BITFIELD_0)});
    WriteData(code);
}

// VectorComputation predicate initialize
void CodeGenerator::SvePredInit(uint32_t s, const PRegBase &pd, Pattern pat)
{
    uint32_t size = GenSize(pd);
    uint32_t code = Concat({LeftShift(0x25, BITFIELD_24), LeftShift(size, BITFIELD_22), LeftShift(0x3, BITFIELD_19),
        LeftShift(s, BITFIELD_16), LeftShift(0x7, BITFIELD_13),
        LeftShift(static_cast<uint32_t>(pat), BITFIELD_5), LeftShift(pd.GetIdx(), BITFIELD_0)});
    WriteData(code);
}

// VectorComputation predicate zero
void CodeGenerator::SvePredZero(uint32_t op, uint32_t s, const PRegBase &pd)
{
    uint32_t code = Concat({LeftShift(0x25, BITFIELD_24), LeftShift(op, BITFIELD_23), LeftShift(s, BITFIELD_22),
        LeftShift(0x3, BITFIELD_19), LeftShift(0x7, BITFIELD_13), LeftShift(0x1, BITFIELD_10),
        LeftShift(pd.GetIdx(), BITFIELD_0)});
    WriteData(code);
}

// VectorComputation integer compare with signed immediate
void CodeGenerator::SveIntCompSImm(
    uint32_t op, uint32_t o2, uint32_t ne, const PRegBase &pd, const PRegBase &pg, const ZRegBase &zn, int32_t imm)
{
    uint32_t size = GenSize(pd);
    uint32_t imm5 = static_cast<uint32_t>(imm) & Ones(IMM5_BIT_LENGTH);
    VerifyIncRange(pg.GetIdx(), 0, VectorComputation_HALF_PG_RANGE, Err::ILLEGAL_REG_IDX);
    VerifyIncRange(imm, IMM5_RANGE_LOW, IMM5_RANGE_HIGH, Err::ILLEGAL_IMM_RANGE, true);
    uint32_t code = Concat({LeftShift(0x25, BITFIELD_24), LeftShift(size, BITFIELD_22), LeftShift(imm5, BITFIELD_16),
        LeftShift(op, BITFIELD_15), LeftShift(o2, BITFIELD_13), LeftShift(pg.GetIdx(), BITFIELD_10),
        LeftShift(zn.GetIdx(), BITFIELD_5), LeftShift(ne, BITFIELD_4), LeftShift(pd.GetIdx(), BITFIELD_0)});
    WriteData(code);
}

// VectorComputation integer compare scalar count and limit
void CodeGenerator::SveIntCompScalarCountAndLimit(
    uint32_t u, uint32_t lt, uint32_t eq, const PRegBase &pd, const RReg &rn, const RReg &rm)
{
    uint32_t size = GenSize(pd);
    uint32_t sf = GenSf(rn);
    uint32_t code = Concat({LeftShift(0x25, BITFIELD_24), LeftShift(size, BITFIELD_22), LeftShift(0x1, BITFIELD_21),
        LeftShift(rm.GetIdx(), BITFIELD_16), LeftShift(sf, BITFIELD_12), LeftShift(u, BITFIELD_11),
        LeftShift(lt, BITFIELD_10), LeftShift(rn.GetIdx(), BITFIELD_5), LeftShift(eq, BITFIELD_4),
        LeftShift(pd.GetIdx(), BITFIELD_0)});
    WriteData(code);
}

// VectorComputation broadcast floating-point immediate (unpredicated)
void CodeGenerator::SveBcFpImmUnpred(uint32_t opc, uint32_t o2, const ZRegBase &zd, double imm)
{
    uint32_t size = GenSize(zd);
    uint32_t imm8 = Code8bitFloat(imm);
    uint32_t code = Concat({LeftShift(0x25, BITFIELD_24), LeftShift(size, BITFIELD_22), LeftShift(0x7, BITFIELD_19),
        LeftShift(opc, BITFIELD_17), LeftShift(0x7, BITFIELD_14), LeftShift(o2, BITFIELD_13),
        LeftShift(imm8, BITFIELD_5), LeftShift(zd.GetIdx(), BITFIELD_0)});
    WriteData(code);
}

// VectorComputation broadcast integer immediate (unpredicated)
void CodeGenerator::SveBcIntImmUnpred(uint32_t opc, const ZRegBase &zd, int32_t imm, ShMod mod, uint32_t sh)
{
    VerifyIncList(static_cast<uint64_t>(mod), {static_cast<uint64_t>(ShMod::LSL)}, Err::ILLEGAL_SHMOD);
    VerifyIncList(sh, {0, IMM8_BIT_LENGTH}, Err::ILLEGAL_CONST_VALUE);
    VerifyIncRange(imm, IMM8_RANGE_LOW, IMM8_RANGE_HIGH, Err::ILLEGAL_IMM_RANGE, true);

    uint32_t size = GenSize(zd);
    uint32_t imm8 = static_cast<uint32_t>(imm) & Ones(IMM8_BIT_LENGTH);
    uint32_t code = Concat({LeftShift(0x25, BITFIELD_24), LeftShift(size, BITFIELD_22), LeftShift(0x7, BITFIELD_19),
        LeftShift(opc, BITFIELD_17), LeftShift(0x3, BITFIELD_14), LeftShift((sh == IMM8_BIT_LENGTH), BITFIELD_13),
        LeftShift(imm8, BITFIELD_5), LeftShift(zd.GetIdx(), BITFIELD_0)});
    WriteData(code);
}

// VectorComputation integer min/max immediate (unpredicated)
void CodeGenerator::SveIntMinMaxImmUnpred(uint32_t opc, uint32_t o2, const ZRegBase &zdn, int32_t imm)
{
    if ((opc & 0x1)) {
        VerifyIncRange(imm, 0, UIMM8_RANGE_HIGH, Err::ILLEGAL_IMM_RANGE);
    } else {
        VerifyIncRange(imm, IMM8_RANGE_LOW, IMM8_RANGE_HIGH, Err::ILLEGAL_IMM_RANGE, true);
    }

    uint32_t size = GenSize(zdn);
    uint32_t imm8 = static_cast<uint32_t>(imm) & Ones(IMM8_BIT_LENGTH);
    uint32_t code = Concat({LeftShift(0x25, BITFIELD_24), LeftShift(size, BITFIELD_22), LeftShift(0x5, BITFIELD_19),
        LeftShift(opc, BITFIELD_16), LeftShift(0x3, BITFIELD_14), LeftShift(o2, BITFIELD_13),
        LeftShift(imm8, BITFIELD_5), LeftShift(zdn.GetIdx(), BITFIELD_0)});
    WriteData(code);
}

// VectorComputation Integer Multiply-Add - Unpredicated
void CodeGenerator::SveIntMultAddUnpredGroup(
    uint32_t op0, const ZRegBase &zda, const ZRegBase &zn, const ZRegBase &zm, uint32_t rot)
{
    uint32_t size = GenSize(zda);

    // If instruction has no "rot" opernad, 0 is passed.
    VerifyIncList(rot, {VectorComputation_ROTATION_0, VectorComputation_ROTATION_90, VectorComputation_ROTATION_180, VectorComputation_ROTATION_270},
        Err::ILLEGAL_CONST_VALUE);
    uint32_t code = Concat({LeftShift(0x44, BITFIELD_24), LeftShift(size, BITFIELD_22),
        LeftShift(zm.GetIdx(), BITFIELD_16), LeftShift(op0 | (rot / VectorComputation_ROTATION_90), BITFIELD_10),
        LeftShift(zn.GetIdx(), BITFIELD_5), LeftShift(zda.GetIdx(), BITFIELD_0)});
    WriteData(code);
}

// VectorComputation floating-point multiply-add (indexed)
void CodeGenerator::SveFpMultAddIndexed(uint32_t op, const ZRegBase &zda, const ZRegBase &zn, const ZRegElem &zm)
{
    uint32_t zmIdx = zm.GetIdx();
    uint32_t zmBit = zm.GetBit();
    uint32_t zmEidx = zm.GetElemIdx();
    uint32_t size = (zmBit == VectorComputation_ELT_BITLEN_H) ? (0 | Field(zmEidx, BITFIELD_2, BITFIELD_2)) : GenSize(zda);
    uint32_t opc = (zmBit == VectorComputation_ELT_BITLEN_D) ?
        (((zmEidx & Ones(VectorComputation_ELT_BITLEN_D_MASK)) << VectorComputation_ELT_BITLEN_D_SHIFT) | zmIdx) :
        (((zmEidx & Ones(VectorComputation_ELT_BITLEN_S_MASK)) << VectorComputation_ELT_BITLEN_S_SHIFT) | zmIdx);

    VerifyIncRange(zmEidx, 0, ((zmBit == VectorComputation_ELT_BITLEN_H) ? VectorComputation_ELT_BITLEN_H_RANGE :
        (zmBit == VectorComputation_ELT_BITLEN_S) ? VectorComputation_ELT_BITLEN_S_RANGE : VectorComputation_ELT_BITLEN_B_RANGE),
        Err::ILLEGAL_REG_ELEM_IDX);
    VerifyIncRange(zmEidx, 0, ((zmBit == VectorComputation_ELT_BITLEN_D) ?
        VectorComputation_ELT_BITLEN_D_RANGE : VectorComputation_ELT_BITLEN_H_RANGE), Err::ILLEGAL_REG_ELEM_IDX);

    uint32_t code = Concat({LeftShift(0x64, BITFIELD_24), LeftShift(size, BITFIELD_22), LeftShift(0x1, BITFIELD_21),
        LeftShift(opc, BITFIELD_16), LeftShift(op, BITFIELD_10),
        LeftShift(zn.GetIdx(), BITFIELD_5), LeftShift(zda.GetIdx(), BITFIELD_0)});
    WriteData(code);
}

// VectorComputation floating-point reciprocal estimate unpredicated
void CodeGenerator::SveFpReciproEstUnPred(uint32_t opc, const ZRegBase &zd, const ZRegBase &zn)
{
    uint32_t size = GenSize(zd);
    uint32_t code = Concat({LeftShift(0x65, BITFIELD_24), LeftShift(size, BITFIELD_22), LeftShift(0x1, BITFIELD_19),
        LeftShift(opc, BITFIELD_16), LeftShift(0x3, BITFIELD_12), LeftShift(zn.GetIdx(), BITFIELD_5),
        LeftShift(zd.GetIdx(), BITFIELD_0)});
    WriteData(code);
}

// VectorComputation floating-point compare with zero
void CodeGenerator::SveFpCompWithZero(
    uint32_t eq, uint32_t lt, uint32_t ne, const PRegBase &pd, const PRegBase &pg, const ZRegBase &zn, double zero)
{
    uint32_t size = GenSize(pd);
    VerifyIncList(std::lround(zero * VectorComputation_COMPZERO_FACTOR), {0}, Err::ILLEGAL_CONST_VALUE);
    VerifyIncRange(pg.GetIdx(), 0, VectorComputation_HALF_PG_RANGE, Err::ILLEGAL_REG_IDX);
    uint32_t code = Concat({LeftShift(0x65, BITFIELD_24), LeftShift(size, BITFIELD_22), LeftShift(0x1, BITFIELD_20),
        LeftShift(eq, BITFIELD_17), LeftShift(lt, BITFIELD_16), LeftShift(0x1, BITFIELD_13),
        LeftShift(pg.GetIdx(), BITFIELD_10), LeftShift(zn.GetIdx(), BITFIELD_5), LeftShift(ne, BITFIELD_4),
        LeftShift(pd.GetIdx(), BITFIELD_0)});
    WriteData(code);
}

// VectorComputation floating-point arithmetic (unpredicated)
void CodeGenerator::SveFpArithmeticUnpred(uint32_t opc, const ZRegBase &zd, const ZRegBase &zn, const ZRegBase &zm)
{
    uint32_t size = GenSize(zd);
    uint32_t code = Concat({LeftShift(0x65, BITFIELD_24), LeftShift(size, BITFIELD_22),
        LeftShift(zm.GetIdx(), BITFIELD_16), LeftShift(opc, BITFIELD_10), LeftShift(zn.GetIdx(), BITFIELD_5),
        LeftShift(zd.GetIdx(), BITFIELD_0)});
    WriteData(code);
}

// VectorComputation floating-point arithmetic (predicated)
void CodeGenerator::SveFpArithmeticPred(uint32_t opc, const ZRegBase &zdn, const PRegBase &pg, const ZRegBase &zm)
{
    uint32_t size = GenSize(zdn);
    VerifyIncRange(pg.GetIdx(), 0, VectorComputation_HALF_PG_RANGE, Err::ILLEGAL_REG_IDX);
    uint32_t code = Concat({LeftShift(0x65, BITFIELD_24), LeftShift(size, BITFIELD_22), LeftShift(opc, BITFIELD_16),
        LeftShift(0x4, BITFIELD_13), LeftShift(pg.GetIdx(), BITFIELD_10), LeftShift(zm.GetIdx(), BITFIELD_5),
        LeftShift(zdn.GetIdx(), BITFIELD_0)});
    WriteData(code);
}

// VectorComputation floating-point arithmetic with immediate (predicated)
void CodeGenerator::SveFpArithmeticImmPred(uint32_t opc, const ZRegBase &zdn, const PRegBase &pg, float ct)
{
    uint32_t size = GenSize(zdn);
    uint32_t i1 = (std::lround(ct * VectorComputation_ARITH_IMM_STEP_10) < VectorComputation_ARITH_IMM_STEP_10) ? 0 : MODE_ENCODE_ONE;
    VerifyIncRange(pg.GetIdx(), 0, VectorComputation_HALF_PG_RANGE, Err::ILLEGAL_REG_IDX);

    if (opc == VectorComputation_ARITH_IMM_OPC_0 || opc == VectorComputation_ARITH_IMM_OPC_1 || opc == VectorComputation_ARITH_IMM_OPC_3) {
        VerifyIncList(std::lround(ct * VectorComputation_ARITH_IMM_STEP_10), {VectorComputation_ARITH_IMM_STEP_5, VectorComputation_ARITH_IMM_STEP_10},
            Err::ILLEGAL_CONST_VALUE);
    } else if (opc == VectorComputation_ARITH_IMM_OPC_2) {
        VerifyIncList(std::lround(ct * VectorComputation_ARITH_IMM_STEP_10), {VectorComputation_ARITH_IMM_STEP_5, VectorComputation_ARITH_IMM_STEP_20},
            Err::ILLEGAL_CONST_VALUE);
    } else {
        VerifyIncList(std::lround(ct * VectorComputation_ARITH_IMM_STEP_10), {0, VectorComputation_ARITH_IMM_STEP_10},
            Err::ILLEGAL_CONST_VALUE);
    }

    uint32_t code = Concat({LeftShift(0x65, BITFIELD_24), LeftShift(size, BITFIELD_22), LeftShift(0x3, BITFIELD_19),
        LeftShift(opc, BITFIELD_16), LeftShift(0x4, BITFIELD_13), LeftShift(pg.GetIdx(), BITFIELD_10),
        LeftShift(i1, BITFIELD_5), LeftShift(zdn.GetIdx(), BITFIELD_0)});
    WriteData(code);
}

// VectorComputation floating-point convert precision
void CodeGenerator::SveFpCvtPrecision(uint32_t opc, uint32_t opc2, const ZRegBase &zd,
    const PRegBase &pg, const ZRegBase &zn)
{
    uint32_t code = Concat({LeftShift(0x65, BITFIELD_24), LeftShift(opc, BITFIELD_22), LeftShift(0x1, BITFIELD_19),
        LeftShift(opc2, BITFIELD_16), LeftShift(0x5, BITFIELD_13),
        LeftShift(pg.GetIdx(), BITFIELD_10), LeftShift(zn.GetIdx(), BITFIELD_5), LeftShift(zd.GetIdx(), BITFIELD_0)});
    WriteData(code);
}

// VectorComputation floating-point convert to integer
void CodeGenerator::SveFpCvtToInt(
    uint32_t opc, uint32_t opc2, uint32_t u, const ZRegBase &zd, const PRegBase &pg, const ZRegBase &zn)
{
    uint32_t code = Concat({LeftShift(0x65, BITFIELD_24), LeftShift(opc, BITFIELD_22), LeftShift(0x3, BITFIELD_19),
        LeftShift(opc2, BITFIELD_17), LeftShift(u, BITFIELD_16), LeftShift(0x5, BITFIELD_13),
        LeftShift(pg.GetIdx(), BITFIELD_10), LeftShift(zn.GetIdx(), BITFIELD_5), LeftShift(zd.GetIdx(), BITFIELD_0)});
    WriteData(code);
}

// VectorComputation floating-point round to integral value
void CodeGenerator::SveFpRoundToIntegral(
    uint32_t opc, const ZRegBase &zd, const PRegBase &pg, const ZRegBase &zn)
{
    uint32_t size = GenSize(zd);
    VerifyIncRange(pg.GetIdx(), 0, VectorComputation_HALF_PG_RANGE, Err::ILLEGAL_REG_IDX);
    uint32_t code = Concat({LeftShift(0x65, BITFIELD_24), LeftShift(size, BITFIELD_22), LeftShift(opc, BITFIELD_16),
        LeftShift(0x5, BITFIELD_13), LeftShift(pg.GetIdx(), BITFIELD_10),
        LeftShift(zn.GetIdx(), BITFIELD_5), LeftShift(zd.GetIdx(), BITFIELD_0)});
    WriteData(code);
}

// VectorComputation floating-point unary operations
void CodeGenerator::SveFpUnaryOp(
    uint32_t opc, const ZRegBase &zd, const PRegBase &pg, const ZRegBase &zn)
{
    uint32_t size = GenSize(zd);
    VerifyIncRange(pg.GetIdx(), 0, VectorComputation_HALF_PG_RANGE, Err::ILLEGAL_REG_IDX);
    uint32_t code = Concat({LeftShift(0x65, BITFIELD_24), LeftShift(size, BITFIELD_22), LeftShift(0x3, BITFIELD_18),
        LeftShift(opc, BITFIELD_16), LeftShift(0x5, BITFIELD_13), LeftShift(pg.GetIdx(), BITFIELD_10),
        LeftShift(zn.GetIdx(), BITFIELD_5), LeftShift(zd.GetIdx(), BITFIELD_0)});
    WriteData(code);
}

// VectorComputation integer convert to floationg-point
void CodeGenerator::SveIntCvtToFp(
    uint32_t opc, uint32_t opc2, uint32_t u, const ZRegBase &zd, const PRegBase &pg, const ZRegBase &zn)
{
    uint32_t code = Concat({LeftShift(0x65, BITFIELD_24), LeftShift(opc, BITFIELD_22), LeftShift(0x2, BITFIELD_19),
        LeftShift(opc2, BITFIELD_17), LeftShift(u, BITFIELD_16), LeftShift(0x5, BITFIELD_13),
        LeftShift(pg.GetIdx(), BITFIELD_10), LeftShift(zn.GetIdx(), BITFIELD_5), LeftShift(zd.GetIdx(), BITFIELD_0)});
    WriteData(code);
}

// VectorComputation floationg-point compare vectors
void CodeGenerator::SveFpCompVec(uint32_t op, uint32_t o2, uint32_t o3, const PRegBase &pd,
    const PRegBase &pg, const ZRegBase &zn, const ZRegBase &zm)
{
    uint32_t size = GenSize(pd);
    VerifyIncRange(pg.GetIdx(), 0, VectorComputation_COMPVEC_PG_RANGE, Err::ILLEGAL_REG_IDX);
    uint32_t code = Concat({LeftShift(0x65, BITFIELD_24), LeftShift(size, BITFIELD_22),
        LeftShift(zm.GetIdx(), BITFIELD_16), LeftShift(op, BITFIELD_15), LeftShift(0x1, BITFIELD_14),
        LeftShift(o2, BITFIELD_13), LeftShift(pg.GetIdx(), BITFIELD_10), LeftShift(zn.GetIdx(), BITFIELD_5),
        LeftShift(o3, BITFIELD_4), LeftShift(pd.GetIdx(), BITFIELD_0)});
    WriteData(code);
}

// VectorComputation floationg-point multiply-accumulate writing addend
void CodeGenerator::SveFpMultAccumAddend(
    uint32_t opc, const ZRegBase &zda, const PRegBase &pg, const ZRegBase &zn, const ZRegBase &zm)
{
    uint32_t size = GenSize(zda);
    uint32_t code = Concat({LeftShift(0x65, BITFIELD_24), LeftShift(size, BITFIELD_22), LeftShift(0x1, BITFIELD_21),
        LeftShift(zm.GetIdx(), BITFIELD_16), LeftShift(opc, BITFIELD_13), LeftShift(pg.GetIdx(), BITFIELD_10),
        LeftShift(zn.GetIdx(), BITFIELD_5), LeftShift(zda.GetIdx(), BITFIELD_0)});
    WriteData(code);
}

// VectorComputation floationg-point multiply-accumulate writing multiplicand
void CodeGenerator::SveFpMultAccumMulti(
    uint32_t opc, const ZRegBase &zdn, const PRegBase &pg, const ZRegBase &zm, const ZRegBase &za)
{
    uint32_t size = GenSize(zdn);
    uint32_t code = Concat({LeftShift(0x65, BITFIELD_24), LeftShift(size, BITFIELD_22), LeftShift(0x1, BITFIELD_21),
        LeftShift(za.GetIdx(), BITFIELD_16), LeftShift(0x1, BITFIELD_15), LeftShift(opc, BITFIELD_13),
        LeftShift(pg.GetIdx(), BITFIELD_10), LeftShift(zm.GetIdx(), BITFIELD_5), LeftShift(zdn.GetIdx(), BITFIELD_0)});
    WriteData(code);
}

// VectorComputation 32-bit gather load (scalar plus 32-bit unscaled offsets)
void CodeGenerator::Sve32GatherLdSc32U(
    uint32_t msz, uint32_t u, uint32_t ff, const ZRegBase &zt, const PRegBase &pg, const AdrSc32U &adr)
{
    uint32_t xs = (adr.GetMod() == ExtMod::SXTW) ? MODE_ENCODE_ONE : 0;
    VerifyIncRange(pg.GetIdx(), 0, VectorComputation_LDR_PG_RANGE, Err::ILLEGAL_REG_IDX);
    uint32_t code = Concat({LeftShift(0x42, BITFIELD_25), LeftShift(msz, BITFIELD_23), LeftShift(xs, BITFIELD_22),
        LeftShift(adr.GetZm().GetIdx(), BITFIELD_16), LeftShift(u, BITFIELD_14), LeftShift(ff, BITFIELD_13),
        LeftShift(pg.GetIdx(), BITFIELD_10), LeftShift(adr.GetXn().GetIdx(), BITFIELD_5),
        LeftShift(zt.GetIdx(), BITFIELD_0)});
    WriteData(code);
}

// VectorComputation 32-bit gather load words (scalar plus 32-bit scaled offsets)
void CodeGenerator::Sve32GatherLdWSc32S(uint32_t u, uint32_t ff, const ZRegBase &zt,
    const PRegBase &pg, const AdrSc32S &adr)
{
    uint32_t xs = (adr.GetMod() == ExtMod::SXTW) ? MODE_ENCODE_ONE : 0;
    VerifyIncRange(pg.GetIdx(), 0, VectorComputation_LDR_PG_RANGE, Err::ILLEGAL_REG_IDX);
    uint32_t code = Concat({LeftShift(0x42, BITFIELD_25), LeftShift(0x2, BITFIELD_23), LeftShift(xs, BITFIELD_22),
        LeftShift(0x1, BITFIELD_21), LeftShift(adr.GetZm().GetIdx(), BITFIELD_16), LeftShift(u, BITFIELD_14),
        LeftShift(ff, BITFIELD_13), LeftShift(pg.GetIdx(), BITFIELD_10), LeftShift(adr.GetXn().GetIdx(), BITFIELD_5),
        LeftShift(zt.GetIdx(), BITFIELD_0)});
    WriteData(code);
}

// VectorComputation 32-bit contiguous prefetch (scalar plus immediate)
void CodeGenerator::Sve32ContiPfScImm(PrfopSve prfopSve, uint32_t msz, const PRegBase &pg, const AdrScImm &adr)
{
    int32_t simm = adr.GetSimm();
    uint32_t imm6 = static_cast<uint32_t>(simm) & Ones(VectorComputation_CONTIPRF_IMM_BIT_LENGTH);
    VerifyIncRange(simm, VectorComputation_CONTIPRF_IMM_RANGE_LOW, VectorComputation_CONTIPRF_IMM_RANGE_HIGH,
        Err::ILLEGAL_IMM_RANGE, true);
    VerifyIncRange(pg.GetIdx(), 0, VectorComputation_PREFETCH_PG_RANGE, Err::ILLEGAL_REG_IDX);
    uint32_t code = Concat({LeftShift(0x42, BITFIELD_25), LeftShift(0x7, BITFIELD_22), LeftShift(imm6, BITFIELD_16),
        LeftShift(msz, BITFIELD_13), LeftShift(pg.GetIdx(), BITFIELD_10),
        LeftShift(adr.GetXn().GetIdx(), BITFIELD_5), LeftShift(static_cast<uint32_t>(prfopSve), BITFIELD_0)});
    WriteData(code);
}

void CodeGenerator::Sve32ContiPfScImm(PrfopSve prfopSve, uint32_t msz, const PRegBase &pg, const AdrNoOfs &adr)
{
    VerifyIncRange(pg.GetIdx(), 0, VectorComputation_PREFETCH_PG_RANGE, Err::ILLEGAL_REG_IDX);
    uint32_t code = Concat({LeftShift(0x42, BITFIELD_25), LeftShift(0x7, BITFIELD_22), LeftShift(0x0, BITFIELD_16),
        LeftShift(msz, BITFIELD_13), LeftShift(pg.GetIdx(), BITFIELD_10), LeftShift(adr.GetXn().GetIdx(), BITFIELD_5),
        LeftShift(static_cast<uint32_t>(prfopSve), BITFIELD_0)});
    WriteData(code);
}

// VectorComputation load and broadcast element
void CodeGenerator::SveLoadAndBcElem(
    uint32_t dtypeh, uint32_t dtypel, const ZRegBase &zt, const PRegBase &pg, const AdrScImm &adr)
{
    int32_t uimm = adr.GetSimm();
    uint32_t dtype = (dtypeh << VectorComputation_BCLDR_DTYPE_SHIFT) | dtypel;
    uint32_t size = GenSize(dtype);
    uint32_t imm6 = (static_cast<uint32_t>(uimm) >> size) & Ones(VectorComputation_BCLDR_UIMM_BIT_LENGTH);

    VerifyIncRange(uimm, 0, VectorComputation_BCLDR_IMM_BLOCK * (0x1 << size), Err::ILLEGAL_IMM_RANGE);
    VerifyCond(
        uimm, [=](uint64_t x) {return (x % ((static_cast<uint64_t>(0x1)) << size)) == 0;}, Err::ILLEGAL_IMM_COND);
    VerifyIncRange(pg.GetIdx(), 0, VectorComputation_LDR_PG_RANGE, Err::ILLEGAL_REG_IDX);

    uint32_t code = Concat({LeftShift(0x42, BITFIELD_25), LeftShift(dtypeh, BITFIELD_23), LeftShift(0x1, BITFIELD_22),
        LeftShift(imm6, BITFIELD_16), LeftShift(0x1, BITFIELD_15), LeftShift(dtypel, BITFIELD_13),
        LeftShift(pg.GetIdx(), BITFIELD_10), LeftShift(adr.GetXn().GetIdx(), BITFIELD_5),
        LeftShift(zt.GetIdx(), BITFIELD_0)});
    WriteData(code);
}

void CodeGenerator::SveLoadAndBcElem(
    uint32_t dtypeh, uint32_t dtypel, const ZRegBase &zt, const PRegBase &pg, const AdrNoOfs &adr)
{
    VerifyIncRange(pg.GetIdx(), 0, VectorComputation_LDR_PG_RANGE, Err::ILLEGAL_REG_IDX);
    uint32_t code = Concat({LeftShift(0x42, BITFIELD_25), LeftShift(dtypeh, BITFIELD_23), LeftShift(0x1, BITFIELD_22),
        LeftShift(0x0, BITFIELD_16), LeftShift(0x1, BITFIELD_15), LeftShift(dtypel, BITFIELD_13),
        LeftShift(pg.GetIdx(), BITFIELD_10), LeftShift(adr.GetXn().GetIdx(), BITFIELD_5),
        LeftShift(zt.GetIdx(), BITFIELD_0)});
    WriteData(code);
}

// VectorComputation load predicate register
void CodeGenerator::SveLoadPredReg(const PRegBase &pt, const AdrScImm &adr)
{
    int32_t imm = adr.GetSimm();
    uint32_t imm9h = Field(imm, BITFIELD_8, BITFIELD_3);
    uint32_t imm9l = Field(imm, BITFIELD_2, BITFIELD_0);
    VerifyIncRange(imm, VectorComputation_PREDLDR_IMM_RANGE_LOW, VectorComputation_PREDLDR_IMM_RANGE_HIGH,
        Err::ILLEGAL_IMM_RANGE, true);
    uint32_t code = Concat({LeftShift(0x42, BITFIELD_25), LeftShift(0x3, BITFIELD_23), LeftShift(imm9h, BITFIELD_16),
        LeftShift(imm9l, BITFIELD_10), LeftShift(adr.GetXn().GetIdx(), BITFIELD_5),
        LeftShift(pt.GetIdx(), BITFIELD_0)});
    WriteData(code);
}

void CodeGenerator::SveLoadPredReg(const PRegBase &pt, const AdrNoOfs &adr)
{
    uint32_t code = Concat({LeftShift(0x42, BITFIELD_25), LeftShift(0x3, BITFIELD_23), LeftShift(0x0, BITFIELD_16),
        LeftShift(0x0, BITFIELD_10), LeftShift(adr.GetXn().GetIdx(), BITFIELD_5), LeftShift(pt.GetIdx(), BITFIELD_0)});
    WriteData(code);
}

// VectorComputation load predicate vector
void CodeGenerator::SveLoadPredVec(const ZRegBase &zt, const AdrScImm &adr)
{
    int32_t imm = adr.GetSimm();
    uint32_t imm9h = Field(imm,  BITFIELD_8,  BITFIELD_3);
    uint32_t imm9l = Field(imm,  BITFIELD_2,  BITFIELD_0);

    VerifyIncRange(imm, VectorComputation_PREDLDR_IMM_RANGE_LOW, VectorComputation_PREDLDR_IMM_RANGE_HIGH,
        Err::ILLEGAL_IMM_RANGE, true);

    uint32_t code = Concat({LeftShift(0x42,  BITFIELD_25), LeftShift(0x3,  BITFIELD_23), LeftShift(imm9h,  BITFIELD_16),
        LeftShift(0x1,  BITFIELD_14), LeftShift(imm9l,  BITFIELD_10),
        LeftShift(adr.GetXn().GetIdx(),  BITFIELD_5), LeftShift(zt.GetIdx(),  BITFIELD_0)});
    WriteData(code);
}

void CodeGenerator::SveLoadPredVec(const ZRegBase &zt, const AdrNoOfs &adr)
{
    uint32_t code = Concat({LeftShift(0x42,  BITFIELD_25), LeftShift(0x3,  BITFIELD_23), LeftShift(0x0,  BITFIELD_16),
        LeftShift(0x1,  BITFIELD_14), LeftShift(0x0,  BITFIELD_10),
        LeftShift(adr.GetXn().GetIdx(),  BITFIELD_5), LeftShift(zt.GetIdx(),  BITFIELD_0)});
    WriteData(code);
}

// VectorComputation contiguous load (scalar plus immediate)
void CodeGenerator::SveContiLdScImm(uint32_t dtype, const ZRegBase &zt, const PRegBase &pg, const AdrScImm &adr)
{
    int32_t imm = adr.GetSimm();
    uint32_t imm4 = static_cast<uint32_t>(imm) & Ones(VectorComputation_CONTILDR_IMM_BIT_LENGTH);
    VerifyIncRange(imm, VectorComputation_CONTILDR_IMM_RANGE_LOW, VectorComputation_CONTILDR_IMM_RANGE_HIGH,
        Err::ILLEGAL_IMM_RANGE, true);
    VerifyIncRange(pg.GetIdx(), 0, VectorComputation_LDR_PG_RANGE, Err::ILLEGAL_REG_IDX);
    uint32_t code = Concat({LeftShift(0x52, BITFIELD_25), LeftShift(dtype, BITFIELD_21), LeftShift(imm4, BITFIELD_16),
        LeftShift(0x5, BITFIELD_13), LeftShift(pg.GetIdx(), BITFIELD_10),
        LeftShift(adr.GetXn().GetIdx(), BITFIELD_5), LeftShift(zt.GetIdx(), BITFIELD_0)});
    WriteData(code);
}

void CodeGenerator::SveContiLdScImm(uint32_t dtype, const ZRegBase &zt, const PRegBase &pg, const AdrNoOfs &adr)
{
    VerifyIncRange(pg.GetIdx(), 0, VectorComputation_LDR_PG_RANGE, Err::ILLEGAL_REG_IDX);
    uint32_t code = Concat({LeftShift(0x52, BITFIELD_25), LeftShift(dtype, BITFIELD_21), LeftShift(0x0, BITFIELD_16),
        LeftShift(0x5, BITFIELD_13), LeftShift(pg.GetIdx(), BITFIELD_10), LeftShift(adr.GetXn().GetIdx(), BITFIELD_5),
        LeftShift(zt.GetIdx(), BITFIELD_0)});
    WriteData(code);
}

// VectorComputation contiguous load (scalar plus scalar)
void CodeGenerator::SveContiLdScSc(uint32_t dtype, const ZRegBase &zt, const PRegBase &pg, const AdrScSc &adr)
{
    VerifyIncList(adr.GetSh(), {GenSize(dtype)}, Err::ILLEGAL_CONST_VALUE);
    VerifyIncRange(pg.GetIdx(), 0, VectorComputation_LDR_PG_RANGE, Err::ILLEGAL_REG_IDX);
    uint32_t code = Concat({LeftShift(0x52, BITFIELD_25), LeftShift(dtype, BITFIELD_21),
        LeftShift(adr.GetXm().GetIdx(), BITFIELD_16), LeftShift(0x2, BITFIELD_13), LeftShift(pg.GetIdx(), BITFIELD_10),
        LeftShift(adr.GetXn().GetIdx(), BITFIELD_5), LeftShift(zt.GetIdx(), BITFIELD_0)});
    WriteData(code);
}

void CodeGenerator::SveLdBcQuadScImm(uint32_t msz, uint32_t num, const ZRegBase &zt,
    const PRegBase &pg, const AdrNoOfs &adr)
{
    VerifyIncRange(pg.GetIdx(), 0, VectorComputation_LDR_PG_RANGE, Err::ILLEGAL_REG_IDX);
    uint32_t code = Concat({LeftShift(0x52, BITFIELD_25), LeftShift(msz, BITFIELD_23), LeftShift(num, BITFIELD_21),
        LeftShift(0x0, BITFIELD_16), LeftShift(0x1, BITFIELD_13), LeftShift(pg.GetIdx(), BITFIELD_10),
        LeftShift(adr.GetXn().GetIdx(), BITFIELD_5), LeftShift(zt.GetIdx(), BITFIELD_0)});
    WriteData(code);
}

void CodeGenerator::SveContiNTStScImm(uint32_t msz, const ZRegBase &zt, const PRegBase &pg, const AdrNoOfs &adr)
{
    VerifyIncRange(pg.GetIdx(), 0, VectorComputation_CONTISTR_PG_RANGE, Err::ILLEGAL_REG_IDX);
    uint32_t code = Concat({LeftShift(0x72, BITFIELD_25), LeftShift(msz, BITFIELD_23), LeftShift(0x1, BITFIELD_20),
        LeftShift(0x0, BITFIELD_16), LeftShift(0x7, BITFIELD_13), LeftShift(pg.GetIdx(), BITFIELD_10),
        LeftShift(adr.GetXn().GetIdx(), BITFIELD_5), LeftShift(zt.GetIdx(), BITFIELD_0)});
    WriteData(code);
}

// VectorComputation contiguous store (scalar plus immediate)
void CodeGenerator::SveContiStScImm(uint32_t msz, const ZRegBase &zt, const PRegBase &pg, const AdrScImm &adr)
{
    uint32_t size = GenSize(zt);
    int32_t imm = adr.GetSimm();
    uint32_t imm4 = static_cast<uint32_t>(imm) & Ones(VectorComputation_CONTISTR_IMM_BIT_LENGTH);
    VerifyIncRange(imm, VectorComputation_CONTISTR_IMM_RANGE_LOW, VectorComputation_CONTISTR_IMM_RANGE_HIGH,
        Err::ILLEGAL_IMM_RANGE, true);
    VerifyIncRange(pg.GetIdx(), 0, VectorComputation_CONTISTR_PG_RANGE, Err::ILLEGAL_REG_IDX);
    uint32_t code = Concat({LeftShift(0x72, BITFIELD_25), LeftShift(msz, BITFIELD_23), LeftShift(size, BITFIELD_21),
        LeftShift(imm4, BITFIELD_16), LeftShift(0x7, BITFIELD_13), LeftShift(pg.GetIdx(), BITFIELD_10),
        LeftShift(adr.GetXn().GetIdx(), BITFIELD_5), LeftShift(zt.GetIdx(), BITFIELD_0)});
    WriteData(code);
}

void CodeGenerator::SveContiStScImm(uint32_t msz, const ZRegBase &zt, const PRegBase &pg, const AdrNoOfs &adr)
{
    uint32_t size = GenSize(zt);
    VerifyIncRange(pg.GetIdx(), 0, VectorComputation_CONTISTR_PG_RANGE, Err::ILLEGAL_REG_IDX);
    uint32_t code = Concat({LeftShift(0x72, BITFIELD_25), LeftShift(msz, BITFIELD_23), LeftShift(size, BITFIELD_21),
        LeftShift(0x0, BITFIELD_16), LeftShift(0x7, BITFIELD_13), LeftShift(pg.GetIdx(), BITFIELD_10),
        LeftShift(adr.GetXn().GetIdx(), BITFIELD_5), LeftShift(zt.GetIdx(), BITFIELD_0)});
    WriteData(code);
}

// VectorComputation contiguous store (scalar plus scalar)
void CodeGenerator::SveContiStScSc(uint32_t msz, const ZRegBase &zt, const PRegBase &pg, const AdrScSc &adr)
{
    VerifyIncList(adr.GetSh(), {msz}, Err::ILLEGAL_CONST_VALUE);
    uint32_t size = GenSize(zt);
    VerifyIncRange(pg.GetIdx(), 0, VectorComputation_CONTISTR_PG_RANGE, Err::ILLEGAL_REG_IDX);
    uint32_t code = Concat({LeftShift(0x72, BITFIELD_25), LeftShift(msz, BITFIELD_23), LeftShift(size, BITFIELD_21),
        LeftShift(adr.GetXm().GetIdx(), BITFIELD_16), LeftShift(2, BITFIELD_13), LeftShift(pg.GetIdx(), BITFIELD_10),
        LeftShift(adr.GetXn().GetIdx(), BITFIELD_5), LeftShift(zt.GetIdx(), BITFIELD_0)});
    WriteData(code);
}

// VectorComputation store predicate register
void CodeGenerator::SveStorePredReg(const PRegBase &pt, const AdrScImm &adr)
{
    int32_t imm = adr.GetSimm();
    uint32_t imm9h = Field(imm, BITFIELD_8, BITFIELD_3);
    uint32_t imm9l = Field(imm, BITFIELD_2, BITFIELD_0);
    VerifyIncRange(imm, VectorComputation_STR_IMM_RANGE_LOW, VectorComputation_STR_IMM_RANGE_HIGH,
        Err::ILLEGAL_IMM_RANGE, true);
    uint32_t code = Concat({LeftShift(0x72, BITFIELD_25), LeftShift(0x3, BITFIELD_23), LeftShift(imm9h, BITFIELD_16),
        LeftShift(imm9l, BITFIELD_10), LeftShift(adr.GetXn().GetIdx(), BITFIELD_5),
        LeftShift(pt.GetIdx(), BITFIELD_0)});
    WriteData(code);
}

void CodeGenerator::SveStorePredReg(const PRegBase &pt, const AdrNoOfs &adr)
{
    uint32_t code = Concat({LeftShift(0x72, BITFIELD_25), LeftShift(0x3, BITFIELD_23), LeftShift(0x0, BITFIELD_16),
        LeftShift(0x0, BITFIELD_10), LeftShift(adr.GetXn().GetIdx(), BITFIELD_5), LeftShift(pt.GetIdx(), BITFIELD_0)});
    WriteData(code);
}

// VectorComputation store predicate vector
void CodeGenerator::SveStorePredVec(const ZRegBase &zt, const AdrScImm &adr)
{
    int32_t imm = adr.GetSimm();
    uint32_t imm9h = Field(imm, BITFIELD_8, BITFIELD_3);
    uint32_t imm9l = Field(imm, BITFIELD_2, BITFIELD_0);
    VerifyIncRange(imm, VectorComputation_STR_IMM_RANGE_LOW, VectorComputation_STR_IMM_RANGE_HIGH,
        Err::ILLEGAL_IMM_RANGE, true);
    uint32_t code = Concat({LeftShift(0x72, BITFIELD_25), LeftShift(0x3, BITFIELD_23), LeftShift(imm9h, BITFIELD_16),
        LeftShift(0x2, BITFIELD_13), LeftShift(imm9l, BITFIELD_10), LeftShift(adr.GetXn().GetIdx(), BITFIELD_5),
        LeftShift(zt.GetIdx(), BITFIELD_0)});
    WriteData(code);
}

void CodeGenerator::SveStorePredVec(const ZRegBase &zt, const AdrNoOfs &adr)
{
    uint32_t code = Concat({LeftShift(0x72, BITFIELD_25), LeftShift(0x3, BITFIELD_23), LeftShift(0x0, BITFIELD_16),
        LeftShift(0x2, BITFIELD_13), LeftShift(0x0, BITFIELD_10), LeftShift(adr.GetXn().GetIdx(), BITFIELD_5),
        LeftShift(zt.GetIdx(), BITFIELD_0)});
    WriteData(code);
}

// Test and branch (immediate)
uint32_t CodeGenerator::TestBrEnc(uint32_t op, const RReg &rt, uint32_t imm, int64_t labelOffset)
{
    VerifyIncRange(labelOffset, -1 * (1 << BITFIELD_15), Ones(BITFIELD_15), Err::LABEL_IS_TOO_FAR, true);
    VerifyIncRange(imm, 0, Ones(BITFIELD_6), Err::ILLEGAL_IMM_RANGE);

    uint32_t bit5 = Field(imm, BITFIELD_5, BITFIELD_5);
    uint32_t bit40 = Field(imm, BITFIELD_4, BITFIELD_0);
    uint32_t imm14 = (static_cast<uint32_t>(labelOffset >> BITFIELD_2)) & Ones(BITFIELD_14);

    if (bit5 == 1) {
        VerifyIncList(rt.GetBit(), {BITFIELD_64}, Err::ILLEGAL_IMM_VALUE);
    }

    return Concat({LeftShift(bit5, BITFIELD_31), LeftShift(0x1b, BITFIELD_25), LeftShift(op, BITFIELD_24),
        LeftShift(bit40, BITFIELD_19), LeftShift(imm14, BITFIELD_5), LeftShift(rt.GetIdx(), BITFIELD_0)});
}

void CodeGenerator::TestBr(uint32_t op, const RReg &rt, uint32_t imm, const Label &label)
{
    auto encFunc = [this, op, rt, imm](int64_t labelOffset) {
        return TestBrEnc(op, rt, imm, labelOffset);
    };
    JmpLabel jmpL = JmpLabel(encFunc, size_);
    uint32_t code = TestBrEnc(op, rt, imm, GenLabelOffset(label, jmpL));
    WriteData(code);
}

// conversion between floating-point and integer
void CodeGenerator::ConversionFpInt(uint32_t sf, uint32_t s, uint32_t type, uint32_t rmode,
    uint32_t opcode, const RReg &rd, const VRegSc &vn)
{
    uint32_t code = Concat({LeftShift(sf, BITFIELD_31), LeftShift(s, BITFIELD_29), LeftShift(0xf, BITFIELD_25),
        LeftShift(type, BITFIELD_22), LeftShift(1, BITFIELD_21), LeftShift(rmode, BITFIELD_19),
        LeftShift(opcode, BITFIELD_16), LeftShift(vn.GetIdx(), BITFIELD_5), LeftShift(rd.GetIdx(), 0)});
    WriteData(code);
}

// conversion between floating-point and integer
void CodeGenerator::ConversionFpInt(uint32_t sf, uint32_t s, uint32_t type, uint32_t rmode,
    uint32_t opcode, const VRegSc &vd, const RReg &rn)
{
    uint32_t code = Concat({LeftShift(sf, BITFIELD_31), LeftShift(s, BITFIELD_29), LeftShift(0xf, BITFIELD_25),
        LeftShift(type, BITFIELD_22), LeftShift(1, BITFIELD_21), LeftShift(rmode, BITFIELD_19),
        LeftShift(opcode, BITFIELD_16), LeftShift(rn.GetIdx(), BITFIELD_5), LeftShift(vd.GetIdx(), 0)});
    WriteData(code);
}

void CodeGenerator::SveFpWideMultAddGroup(uint32_t bit10, const ZRegBase &zda,
    const ZRegBase &zn, const ZRegBase &zm)
{
    uint32_t code = Concat({LeftShift(0x64, BITFIELD_24), LeftShift(zm.GetIdx(), BITFIELD_16),
        LeftShift(bit10, BITFIELD_10), LeftShift(zn.GetIdx(), BITFIELD_5), LeftShift(zda.GetIdx(), BITFIELD_0)});
    WriteData(code);
}

void CodeGenerator::SveFpWideMultAddIndexedGroup(uint32_t bit10, const ZRegBase &zda,
    const ZRegBase &zn, const ZRegElem &zm)
{
    uint32_t zmEidx = zm.GetElemIdx();
    uint32_t i2 = 0;
    uint32_t i3h = 0;
    uint32_t i3l = 0;

    VerifyIncRange(zm.GetIdx(), 0, RINDEX7, Err::ILLEGAL_REG_IDX);

    if (bit10 == 0x1810 /* BFDOT */) {
        VerifyIncRange(zmEidx, 0, RINDEX3, Err::ILLEGAL_REG_IDX);
        i2 = zmEidx;
    } else {
        VerifyIncRange(zmEidx, 0, RINDEX7, Err::ILLEGAL_REG_IDX);
        i3h = Field(zmEidx, RINDEX2, 1);
        i3l = Field(zmEidx, 0, 0);
    }

    uint32_t code = Concat({LeftShift(0x64, BITFIELD_24), LeftShift((i2 | i3h), BITFIELD_19),
        LeftShift(zm.GetIdx(), BITFIELD_16), LeftShift(i3l, BITFIELD_11), LeftShift(bit10, BITFIELD_10),
        LeftShift(zn.GetIdx(), BITFIELD_5), LeftShift(zda.GetIdx(), BITFIELD_0)});
    WriteData(code);
}

void CodeGenerator::Orr(const ZRegS &zdn, const PRegBase &pg, const ZRegS &zm)
{
    SveBitwiseLOpPred(0, zdn, pg, zm);
}

void CodeGenerator::Eor(const ZRegS &zdn, const PRegBase &pg, const ZRegS &zm)
{
    SveBitwiseLOpPred(VAL_1, zdn, pg, zm);
}

void CodeGenerator::And(const ZRegB &zdn, const PRegBase &pg, const ZRegB &zm)
{
    SveBitwiseLOpPred(VAL_2, zdn, pg, zm);
}

void CodeGenerator::And(const ZRegS &zdn, const PRegBase &pg, const ZRegS &zm)
{
    SveBitwiseLOpPred(VAL_2, zdn, pg, zm);
}

void CodeGenerator::Smin(const ZRegS &zdn, const PRegBase &pg, const ZRegS &zm)
{
    SveIntMinMaxDiffPred(VAL_1, 0, zdn, pg, zm);
}

void CodeGenerator::Mul(const ZRegS &zdn, const PRegBase &pg, const ZRegS &zm)
{
    SveIntMultDivVecPred(0, 0, zdn, pg, zm);
}

void CodeGenerator::Movprfx(const ZRegS &zd, const PRegBase &pg, const ZRegS &zn)
{
    SveConstPrefPred(0, zd, pg, zn);
}

void CodeGenerator::Asr(const ZRegS &zdn, const PRegBase &pg, const uint32_t amount)
{
    SveBitwiseShByImmPred(0x0, 0, 0, zdn, pg, amount);
}

void CodeGenerator::Lsl(const ZRegS &zdn, const PRegBase &pg, const uint32_t amount)
{
    SveBitwiseShByImmPred(0x0, VAL_1, VAL_1, zdn, pg, amount);
}

void CodeGenerator::Lsl(const ZRegH &zdn, const PRegBase &pg, const ZRegH &zm)
{
    SveBitwiseShVecPred(VAL_3, zdn, pg, zm);
}

void CodeGenerator::Lsl(const ZRegS &zdn, const PRegBase &pg, const ZRegS &zm)
{
    SveBitwiseShVecPred(VAL_3, zdn, pg, zm);
}

void CodeGenerator::Fabs(const ZRegS &zd, const PRegBase &pg, const ZRegS &zn)
{
    SveBitwiseUnaryOpPred(VAL_4, zd, pg, zn);
}

void CodeGenerator::Fneg(const ZRegS &zd, const PRegBase &pg, const ZRegS &zn)
{
    SveBitwiseUnaryOpPred(VAL_5, zd, pg, zn);
}

void CodeGenerator::Sxtb(const ZRegS &zd, const PRegBase &pg, const ZRegS &zn)
{
    SveIntUnaryOpPred(0, zd, pg, zn);
}

void CodeGenerator::Uxtb(const ZRegS &zd, const PRegBase &pg, const ZRegS &zn)
{
    SveIntUnaryOpPred(VAL_1, zd, pg, zn);
}

void CodeGenerator::Add(const ZRegS &zd, const ZRegS &zn, const ZRegS &zm)
{
    SveIntAddSubUnpred(0, zd, zn, zm);
}

void CodeGenerator::Sub(const ZRegB &zd, const ZRegB &zn, const ZRegB &zm)
{
    SveIntAddSubUnpred(VAL_1, zd, zn, zm);
}

void CodeGenerator::Sub(const ZRegS &zd, const ZRegS &zn, const ZRegS &zm)
{
    SveIntAddSubUnpred(VAL_1, zd, zn, zm);
}

void CodeGenerator::And(const ZRegD &zd, const ZRegD &zn, const ZRegD &zm)
{
    SveBitwiseLOpUnpred(0, zd, zn, zm);
}

void CodeGenerator::Orr(const ZRegD &zd, const ZRegD &zn, const ZRegD &zm)
{
    SveBitwiseLOpUnpred(VAL_1, zd, zn, zm);
}

void CodeGenerator::Mov(const ZRegD &zd, const ZRegD &zn)
{
    SveBitwiseLOpUnpred(VAL_1, zd, zn, zn);
}

void CodeGenerator::Eor(const ZRegD &zd, const ZRegD &zn, const ZRegD &zm)
{
    SveBitwiseLOpUnpred(VAL_2, zd, zn, zm);
}

void CodeGenerator::Index(const ZRegH &zd, const int32_t imm1, const int32_t imm2)
{
    SveIndexGenImmImmInc(zd, imm1, imm2);
}

void CodeGenerator::Index(const ZRegS &zd, const int32_t imm1, const int32_t imm2)
{
    SveIndexGenImmImmInc(zd, imm1, imm2);
}

void CodeGenerator::Addvl(const XReg &xd, const XReg &xn, const int32_t imm)
{
    SveStackFrameAdjust(0, xd, xn, imm);
}

void CodeGenerator::Asr(const ZRegS &zd, const ZRegS &zn, const uint32_t amount)
{
    SveBitwiseShByImmUnpred(0, zd, zn, amount);
}

void CodeGenerator::Lsr(const ZRegS &zd, const ZRegS &zn, const uint32_t amount)
{
    SveBitwiseShByImmUnpred(VAL_1, zd, zn, amount);
}

void CodeGenerator::Lsl(const ZRegS &zd, const ZRegS &zn, const uint32_t amount)
{
    SveBitwiseShByImmUnpred(VAL_3, zd, zn, amount);
}

void CodeGenerator::Fexpa(const ZRegS &zd, const ZRegS &zn)
{
    SveFpExpAccel(0, zd, zn);
}

void CodeGenerator::Cntw(const XReg &xd, const Pattern pat, const ExtMod mod, const uint32_t imm)
{
    SveElemCount(VAL_2, 0, xd, pat, mod, imm);
}

void CodeGenerator::Incb(const XReg &xd, const Pattern pat, const ExtMod mod, const uint32_t imm)
{
    SveIncDecRegByElemCount(0, 0, xd, pat, mod, imm);
}

void CodeGenerator::Inch(const XReg &xd, const Pattern pat, const ExtMod mod, const uint32_t imm)
{
    SveIncDecRegByElemCount(VAL_1, 0, xd, pat, mod, imm);
}

void CodeGenerator::Incw(const XReg &xd, const Pattern pat, const ExtMod mod, const uint32_t imm)
{
    SveIncDecRegByElemCount(VAL_2, 0, xd, pat, mod, imm);
}

void CodeGenerator::Decw(const XReg &xd, const Pattern pat, const ExtMod mod, const uint32_t imm)
{
    SveIncDecRegByElemCount(VAL_2, VAL_1, xd, pat, mod, imm);
}

void CodeGenerator::Eor(const ZRegS &zdn, const uint64_t imm)
{
    SveBitwiseLogicalImmUnpred(VAL_1, zdn, imm);
}

void CodeGenerator::And(const ZRegS &zdn, const uint64_t imm)
{
    SveBitwiseLogicalImmUnpred(VAL_2, zdn, imm);
}

void CodeGenerator::Dupm(const ZRegS &zd, const uint64_t imm)
{
    SveBcBitmaskImm(zd, imm);
}

void CodeGenerator::Fcpy(const ZRegS &zd, const PRegBase &pg, const double imm)
{
    SveCopyFpImmPred(zd, pg, imm);
}

void CodeGenerator::Fmov(const ZRegS &zd, const PRegBase &pg, const double imm)
{
    SveCopyFpImmPred(zd, pg, imm);
}

void CodeGenerator::Mov(const ZRegB &zd, const PRegBase &pg, const uint32_t imm, const ShMod mod, const uint32_t sh)
{
    SveCopyIntImmPred(zd, pg, imm, mod, sh);
}

void CodeGenerator::Mov(const ZRegS &zd, const PRegBase &pg, const uint32_t imm, const ShMod mod, const uint32_t sh)
{
    SveCopyIntImmPred(zd, pg, imm, mod, sh);
}

void CodeGenerator::Ext(const ZRegB &zdn, const ZRegB &zm, const uint32_t imm)
{
    SveExtVec(zdn, zm, imm);
}

void CodeGenerator::Dup(const ZRegH &zd, const WReg &rn)
{
    SveBcGeneralReg(zd, rn);
}

void CodeGenerator::Dup(const ZRegS &zd, const WReg &rn)
{
    SveBcGeneralReg(zd, rn);
}

void CodeGenerator::Mov(const ZRegS &zd, const WReg &rn)
{
    SveBcGeneralReg(zd, rn);
}

void CodeGenerator::Mov(const ZRegS &zd, const SReg &vn)
{
    SveBcIndexedElem(zd, ZRegElem(vn.GetIdx(), 0, vn.GetBit()));
}

void CodeGenerator::Dup(const ZRegB &zd, const ZRegBElem &zn)
{
    SveBcIndexedElem(zd, zn);
}

void CodeGenerator::Dup(const ZRegS &zd, const ZRegSElem &zn)
{
    SveBcIndexedElem(zd, zn);
}

void CodeGenerator::Zip1(const PRegB &pd, const PRegB &pn, const PRegB &pm)
{
    SvePermutePredElem(0, 0, pd, pn, pm);
}

void CodeGenerator::Zip1(const PRegH &pd, const PRegH &pn, const PRegH &pm)
{
    SvePermutePredElem(0, 0, pd, pn, pm);
}

void CodeGenerator::Uzp1(const PRegB &pd, const PRegB &pn, const PRegB &pm)
{
    SvePermutePredElem(VAL_1, 0, pd, pn, pm);
}

void CodeGenerator::Uzp1(const PRegH &pd, const PRegH &pn, const PRegH &pm)
{
    SvePermutePredElem(VAL_1, 0, pd, pn, pm);
}

void CodeGenerator::Trn1(const PRegS &pd, const PRegS &pn, const PRegS &pm)
{
    SvePermutePredElem(VAL_2, 0, pd, pn, pm);
}

void CodeGenerator::Trn1(const PRegD &pd, const PRegD &pn, const PRegD &pm)
{
    SvePermutePredElem(VAL_2, 0, pd, pn, pm);
}

void CodeGenerator::Zip1(const ZRegB &zd, const ZRegB &zn, const ZRegB &zm)
{
    SvePermuteVecElem(0, zd, zn, zm);
}

void CodeGenerator::Zip1(const ZRegH &zd, const ZRegH &zn, const ZRegH &zm)
{
    SvePermuteVecElem(0, zd, zn, zm);
}

void CodeGenerator::Zip1(const ZRegS &zd, const ZRegS &zn, const ZRegS &zm)
{
    SvePermuteVecElem(0, zd, zn, zm);
}

void CodeGenerator::Zip2(const ZRegB &zd, const ZRegB &zn, const ZRegB &zm)
{
    SvePermuteVecElem(VAL_1, zd, zn, zm);
}

void CodeGenerator::Zip2(const ZRegH &zd, const ZRegH &zn, const ZRegH &zm)
{
    SvePermuteVecElem(VAL_1, zd, zn, zm);
}

void CodeGenerator::Zip2(const ZRegS &zd, const ZRegS &zn, const ZRegS &zm)
{
    SvePermuteVecElem(VAL_1, zd, zn, zm);
}

void CodeGenerator::Uzp1(const ZRegB &zd, const ZRegB &zn, const ZRegB &zm)
{
    SvePermuteVecElem(VAL_2, zd, zn, zm);
}

void CodeGenerator::Uzp1(const ZRegH &zd, const ZRegH &zn, const ZRegH &zm)
{
    SvePermuteVecElem(VAL_2, zd, zn, zm);
}

void CodeGenerator::Uzp2(const ZRegD &zd, const ZRegD &zn, const ZRegD &zm)
{
    SvePermuteVecElem(VAL_3, zd, zn, zm);
}

void CodeGenerator::Trn1(const ZRegS &zd, const ZRegS &zn, const ZRegS &zm)
{
    SvePermuteVecElem(VAL_4, zd, zn, zm);
}

void CodeGenerator::Trn1(const ZRegD &zd, const ZRegD &zn, const ZRegD &zm)
{
    SvePermuteVecElem(VAL_4, zd, zn, zm);
}

void CodeGenerator::Trn2(const ZRegS &zd, const ZRegS &zn, const ZRegS &zm)
{
    SvePermuteVecElem(VAL_5, zd, zn, zm);
}

void CodeGenerator::Trn2(const ZRegD &zd, const ZRegD &zn, const ZRegD &zm)
{
    SvePermuteVecElem(VAL_5, zd, zn, zm);
}

void CodeGenerator::Cpy(const ZRegS &zd, const PRegBase &pg, const WReg &rn)
{
    SveCopyGeneralRegToVecPred(zd, pg, rn);
}

void CodeGenerator::Splice(const ZRegD &zdn, const PRegBase &pg, const ZRegD &zm)
{
    SveSelVecSplice(zdn, pg, zm);
}

void CodeGenerator::Sel(const ZRegB &zd, const PRegBase &pg, const ZRegB &zn, const ZRegB &zm)
{
    SveSelVecElemPred(zd, pg, zn, zm);
}

void CodeGenerator::Sel(const ZRegS &zd, const PRegBase &pg, const ZRegS &zn, const ZRegS &zm)
{
    SveSelVecElemPred(zd, pg, zn, zm);
}

void CodeGenerator::Mov(const ZRegB &zd, const PRegBase &pg, const ZRegB &zn)
{
    SveSelVecElemPred(zd, pg, zn, zd);
}

void CodeGenerator::Mov(const ZRegS &zd, const PRegBase &pg, const ZRegS &zn)
{
    SveSelVecElemPred(zd, pg, zn, zd);
}

void CodeGenerator::Mov(const ZRegD &zd, const PRegBase &pg, const ZRegD &zn)
{
    SveSelVecElemPred(zd, pg, zn, zd);
}

void CodeGenerator::Cmpeq(const PRegS &pd, const PRegBase &pg, const ZRegS &zn, const ZRegS &zm)
{
    SveIntCompVec(VAL_1, VAL_1, 0, pd, pg, zn, zm);
}

void CodeGenerator::Cmpls(const PRegB &pd, const PRegBase &pg, const ZRegB &zn, const ZRegB &zm)
{
    SveIntCompVec(0, 0, 0, pd, pg, zm, zn);
}

void CodeGenerator::Cmplt(const PRegB &pd, const PRegBase &pg, const ZRegB &zn, const ZRegB &zm)
{
    SveIntCompVec(VAL_1, 0, VAL_1, pd, pg, zm, zn);
}

void CodeGenerator::Cmplt(const PRegS &pd, const PRegBase &pg, const ZRegS &zn, const ZRegS &zm)
{
    SveIntCompVec(VAL_1, 0, VAL_1, pd, pg, zm, zn);
}

void CodeGenerator::And(const PRegB &pd, const PRegBase &pg, const PRegB &pn, const PRegB &pm)
{
    SvePredLOp(0, 0, 0, 0, pd, pg, pn, pm);
}

void CodeGenerator::Mov(const PRegB &pd, const PRegBase &pg, const PRegB &pn)
{
    SvePredLOp(0, 0, pg.IsM(), pg.IsM(), pd, pg, pn, (pg.IsZ()) ? pn : pd);
}

void CodeGenerator::Eor(const PRegB &pd, const PRegBase &pg, const PRegB &pn, const PRegB &pm)
{
    SvePredLOp(0, 0, VAL_1, 0, pd, pg, pn, pm);
}

void CodeGenerator::Not(const PRegB &pd, const PRegBase &pg, const PRegB &pn)
{
    SvePredLOp(0, 0, VAL_1, 0, pd, pg, pn, pg);
}

void CodeGenerator::Mov(const PRegB &pd, const PRegB &pn)
{
    SvePredLOp(VAL_1, 0, 0, 0, pd, pn, pn, pn);
}

void CodeGenerator::Ptrue(const PRegB &pd, const Pattern pat)
{
    SvePredInit(0, pd, pat);
}

void CodeGenerator::Ptrue(const PRegH &pd, const Pattern pat)
{
    SvePredInit(0, pd, pat);
}

void CodeGenerator::Ptrue(const PRegS &pd, const Pattern pat)
{
    SvePredInit(0, pd, pat);
}

void CodeGenerator::Ptrue(const PRegD &pd, const Pattern pat)
{
    SvePredInit(0, pd, pat);
}

void CodeGenerator::Pfalse(const PRegB &pd)
{
    SvePredZero(0, 0, pd);
}

void CodeGenerator::Cmplt(const PRegS &pd, const PRegBase &pg, const ZRegS &zn, const int32_t imm)
{
    SveIntCompSImm(0, VAL_1, 0, pd, pg, zn, imm);
}

void CodeGenerator::Cmple(const PRegS &pd, const PRegBase &pg, const ZRegS &zn, const int32_t imm)
{
    SveIntCompSImm(0, VAL_1, VAL_1, pd, pg, zn, imm);
}

void CodeGenerator::Cmpeq(const PRegS &pd, const PRegBase &pg, const ZRegS &zn, const int32_t imm)
{
    SveIntCompSImm(VAL_1, 0, 0, pd, pg, zn, imm);
}

void CodeGenerator::Cmpne(const PRegH &pd, const PRegBase &pg, const ZRegH &zn, const int32_t imm)
{
    SveIntCompSImm(VAL_1, 0, VAL_1, pd, pg, zn, imm);
}

void CodeGenerator::Cmpne(const PRegS &pd, const PRegBase &pg, const ZRegS &zn, const int32_t imm)
{
    SveIntCompSImm(VAL_1, 0, VAL_1, pd, pg, zn, imm);
}

void CodeGenerator::Whilelt(const PRegB &pd, const XReg &rn, const XReg &rm)
{
    SveIntCompScalarCountAndLimit(0, VAL_1, 0, pd, rn, rm);
}

void CodeGenerator::Whilelt(const PRegH &pd, const XReg &rn, const XReg &rm)
{
    SveIntCompScalarCountAndLimit(0, VAL_1, 0, pd, rn, rm);
}

void CodeGenerator::Whilelt(const PRegS &pd, const XReg &rn, const XReg &rm)
{
    SveIntCompScalarCountAndLimit(0, VAL_1, 0, pd, rn, rm);
}

void CodeGenerator::Fdup(const ZRegS &zd, const double imm)
{
    SveBcFpImmUnpred(0, 0, zd, imm);
}

void CodeGenerator::Fmov(const ZRegS &zd, const double imm)
{
    SveBcFpImmUnpred(0, 0, zd, imm);
}

void CodeGenerator::Dup(const ZRegB &zd, const int32_t imm, const ShMod mod, const uint32_t sh)
{
    SveBcIntImmUnpred(0, zd, imm, mod, sh);
}

void CodeGenerator::Dup(const ZRegS &zd, const int32_t imm, const ShMod mod, const uint32_t sh)
{
    SveBcIntImmUnpred(0, zd, imm, mod, sh);
}

void CodeGenerator::Dup(const ZRegD &zd, const int32_t imm, const ShMod mod, const uint32_t sh)
{
    SveBcIntImmUnpred(0, zd, imm, mod, sh);
}

void CodeGenerator::Mov(const ZRegB &zd, const int32_t imm, const ShMod mod, const uint32_t sh)
{
    SveBcIntImmUnpred(0, zd, imm, mod, sh);
}

void CodeGenerator::Mov(const ZRegH &zd, const int32_t imm, const ShMod mod, const uint32_t sh)
{
    SveBcIntImmUnpred(0, zd, imm, mod, sh);
}

void CodeGenerator::Mov(const ZRegS &zd, const int32_t imm, const ShMod mod, const uint32_t sh)
{
    SveBcIntImmUnpred(0, zd, imm, mod, sh);
}

void CodeGenerator::Fmov(const ZRegH &zd, const float imm)
{
    SveBcIntImmUnpred(0, zd, static_cast<uint32_t>(imm), ShMod::LSL, 0);
}

void CodeGenerator::Fmov(const ZRegS &zd, const float imm)
{
    SveBcIntImmUnpred(0, zd, static_cast<uint32_t>(imm), ShMod::LSL, 0);
}

void CodeGenerator::Smax(const ZRegS &zdn, const int32_t imm)
{
    SveIntMinMaxImmUnpred(0, 0, zdn, imm);
}

void CodeGenerator::Smin(const ZRegS &zdn, const int32_t imm)
{
    SveIntMinMaxImmUnpred(VAL_2, 0, zdn, imm);
}

void CodeGenerator::Umin(const ZRegS &zdn, const int32_t imm)
{
    SveIntMinMaxImmUnpred(VAL_3, 0, zdn, imm);
}

void CodeGenerator::Sdot(const ZRegS &zda, const ZRegB &zn, const ZRegB &zm)
{
    SveIntMultAddUnpredGroup(0, zda, zn, zm, 0);
}

void CodeGenerator::Saddv(const DReg &vd, const PRegBase &pg, const ZRegS &zn)
{
    SveIntAddReductPred(0, 0, vd, pg, zn);
}

void CodeGenerator::Uaddv(const DReg &vd, const PRegBase &pg, const ZRegS &zn)
{
    SveIntAddReductPred(0, VAL_1, vd, pg, zn);
}

void CodeGenerator::Fmla(const ZRegS &zda, const ZRegS &zn, const ZRegSElem &zm)
{
    SveFpMultAddIndexed(0, zda, zn, zm);
}

void CodeGenerator::Fmla(const ZRegH &zda, const ZRegH &zn, const ZRegHElem &zm)
{
    SveFpMultAddIndexed(0, zda, zn, zm);
}

void CodeGenerator::Frecpe(const ZRegS &zd, const ZRegS &zn)
{
    SveFpReciproEstUnPred(VAL_6, zd, zn);
}

void CodeGenerator::Fcmge(const PRegS &pd, const PRegBase &pg, const ZRegS &zn, const double zero)
{
    SveFpCompWithZero(0, 0, 0, pd, pg, zn, zero);
}

void CodeGenerator::Fcmgt(const PRegS &pd, const PRegBase &pg, const ZRegS &zn, const double zero)
{
    SveFpCompWithZero(0, 0, VAL_1, pd, pg, zn, zero);
}

void CodeGenerator::Fcmlt(const PRegS &pd, const PRegBase &pg, const ZRegS &zn, const double zero)
{
    SveFpCompWithZero(0, VAL_1, 0, pd, pg, zn, zero);
}

void CodeGenerator::Fcmle(const PRegS &pd, const PRegBase &pg, const ZRegS &zn, const double zero)
{
    SveFpCompWithZero(0, VAL_1, VAL_1, pd, pg, zn, zero);
}

void CodeGenerator::Fcmeq(const PRegS &pd, const PRegBase &pg, const ZRegS &zn, const double zero)
{
    SveFpCompWithZero(VAL_1, 0, 0, pd, pg, zn, zero);
}

void CodeGenerator::Fadd(const ZRegH &zd, const ZRegH &zn, const ZRegH &zm)
{
    SveFpArithmeticUnpred(0, zd, zn, zm);
}

void CodeGenerator::Fadd(const ZRegS &zd, const ZRegS &zn, const ZRegS &zm)
{
    SveFpArithmeticUnpred(0, zd, zn, zm);
}

void CodeGenerator::Fsub(const ZRegS &zd, const ZRegS &zn, const ZRegS &zm)
{
    SveFpArithmeticUnpred(VAL_1, zd, zn, zm);
}

void CodeGenerator::Fmul(const ZRegS &zd, const ZRegS &zn, const ZRegS &zm)
{
    SveFpArithmeticUnpred(VAL_2, zd, zn, zm);
}

void CodeGenerator::Frecps(const ZRegS &zd, const ZRegS &zn, const ZRegS &zm)
{
    SveFpArithmeticUnpred(VAL_6, zd, zn, zm);
}

void CodeGenerator::Fadd(const ZRegS &zdn, const PRegBase &pg, const ZRegS &zm)
{
    SveFpArithmeticPred(0, zdn, pg, zm);
}

void CodeGenerator::Fmul(const ZRegH &zdn, const PRegBase &pg, const ZRegH &zm)
{
    SveFpArithmeticPred(VAL_2, zdn, pg, zm);
}

void CodeGenerator::Fmul(const ZRegS &zdn, const PRegBase &pg, const ZRegS &zm)
{
    SveFpArithmeticPred(VAL_2, zdn, pg, zm);
}

void CodeGenerator::Fmaxnm(const ZRegS &zdn, const PRegBase &pg, const ZRegS &zm)
{
    SveFpArithmeticPred(VAL_4, zdn, pg, zm);
}

void CodeGenerator::Fminnm(const ZRegS &zdn, const PRegBase &pg, const ZRegS &zm)
{
    SveFpArithmeticPred(VAL_5, zdn, pg, zm);
}

void CodeGenerator::Fmax(const ZRegS &zdn, const PRegBase &pg, const ZRegS &zm)
{
    SveFpArithmeticPred(VAL_6, zdn, pg, zm);
}

void CodeGenerator::Fmin(const ZRegS &zdn, const PRegBase &pg, const ZRegS &zm)
{
    SveFpArithmeticPred(VAL_7, zdn, pg, zm);
}

void CodeGenerator::Fscale(const ZRegS &zdn, const PRegBase &pg, const ZRegS &zm)
{
    SveFpArithmeticPred(VAL_9, zdn, pg, zm);
}

void CodeGenerator::Fdiv(const ZRegS &zdn, const PRegBase &pg, const ZRegS &zm)
{
    SveFpArithmeticPred(VAL_13, zdn, pg, zm);
}

void CodeGenerator::Fadd(const ZRegS &zdn, const PRegBase &pg, const float ct)
{
    SveFpArithmeticImmPred(0, zdn, pg, ct);
}

void CodeGenerator::Fsub(const ZRegS &zdn, const PRegBase &pg, const float ct)
{
    SveFpArithmeticImmPred(VAL_1, zdn, pg, ct);
}

void CodeGenerator::Fmul(const ZRegS &zdn, const PRegBase &pg, const float ct)
{
    SveFpArithmeticImmPred(VAL_2, zdn, pg, ct);
}

void CodeGenerator::Fmaxnm(const ZRegS &zdn, const PRegBase &pg, const float ct)
{
    SveFpArithmeticImmPred(VAL_4, zdn, pg, ct);
}

void CodeGenerator::Fminnm(const ZRegS &zdn, const PRegBase &pg, const float ct)
{
    SveFpArithmeticImmPred(VAL_5, zdn, pg, ct);
}

void CodeGenerator::Fmax(const ZRegS &zdn, const PRegBase &pg, const float ct)
{
    SveFpArithmeticImmPred(VAL_6, zdn, pg, ct);
}

void CodeGenerator::Fcvt(const ZRegH &zd, const PRegBase &pg, const ZRegS &zn)
{
    SveFpCvtPrecision(VAL_2, 0, zd, pg, zn);
}

void CodeGenerator::Fcvt(const ZRegS &zd, const PRegBase &pg, const ZRegH &zn)
{
    SveFpCvtPrecision(VAL_2, VAL_1, zd, pg, zn);
}

void CodeGenerator::Bfcvt(const ZRegH &zd, const PRegBase &pg, const ZRegS &zn)
{
    SveFpCvtPrecision(VAL_2, VAL_2, zd, pg, zn);
}

void CodeGenerator::Fcvtzs(const ZRegS &zd, const PRegBase &pg, const ZRegS &zn)
{
    SveFpCvtToInt(VAL_2, VAL_2, 0, zd, pg, zn);
}

void CodeGenerator::Fcvtzu(const ZRegS &zd, const PRegBase &pg, const ZRegS &zn)
{
    SveFpCvtToInt(VAL_2, VAL_2, VAL_1, zd, pg, zn);
}

void CodeGenerator::Frintn(const ZRegS &zd, const PRegBase &pg, const ZRegS &zn)
{
    SveFpRoundToIntegral(0, zd, pg, zn);
}

void CodeGenerator::Frintm(const ZRegS &zd, const PRegBase &pg, const ZRegS &zn)
{
    SveFpRoundToIntegral(VAL_2, zd, pg, zn);
}

void CodeGenerator::Frinti(const ZRegS &zd, const PRegBase &pg, const ZRegS &zn)
{
    SveFpRoundToIntegral(VAL_7, zd, pg, zn);
}

void CodeGenerator::Fsqrt(const ZRegS &zd, const PRegBase &pg, const ZRegS &zn)
{
    SveFpUnaryOp(VAL_1, zd, pg, zn);
}

void CodeGenerator::Scvtf(const ZRegS &zd, const PRegBase &pg, const ZRegS &zn)
{
    SveIntCvtToFp(VAL_2, VAL_2, 0, zd, pg, zn);
}

void CodeGenerator::Fcmge(const PRegS &pd, const PRegBase &pg, const ZRegS &zn, const ZRegS &zm)
{
    SveFpCompVec(0, 0, 0, pd, pg, zn, zm);
}

void CodeGenerator::Fcmgt(const PRegS &pd, const PRegBase &pg, const ZRegS &zn, const ZRegS &zm)
{
    SveFpCompVec(0, 0, VAL_1, pd, pg, zn, zm);
}

void CodeGenerator::Fcmle(const PRegS &pd, const PRegBase &pg, const ZRegS &zn, const ZRegS &zm)
{
    SveFpCompVec(0, 0, 0, pd, pg, zm, zn);
}

void CodeGenerator::Fcmlt(const PRegS &pd, const PRegBase &pg, const ZRegS &zn, const ZRegS &zm)
{
    SveFpCompVec(0, 0, VAL_1, pd, pg, zm, zn);
}

void CodeGenerator::Fcmeq(const PRegS &pd, const PRegBase &pg, const ZRegS &zn, const ZRegS &zm)
{
    SveFpCompVec(0, VAL_1, 0, pd, pg, zn, zm);
}

void CodeGenerator::Fcmne(const PRegS &pd, const PRegBase &pg, const ZRegS &zn, const ZRegS &zm)
{
    SveFpCompVec(0, VAL_1, VAL_1, pd, pg, zn, zm);
}

void CodeGenerator::Facge(const PRegS &pd, const PRegBase &pg, const ZRegS &zn, const ZRegS &zm)
{
    SveFpCompVec(1, 0, VAL_1, pd, pg, zn, zm);
}

void CodeGenerator::Fmla(const ZRegH &zda, const PRegBase &pg, const ZRegH &zn, const ZRegH &zm)
{
    SveFpMultAccumAddend(0, zda, pg, zn, zm);
}

void CodeGenerator::Fmla(const ZRegS &zda, const PRegBase &pg, const ZRegS &zn, const ZRegS &zm)
{
    SveFpMultAccumAddend(0, zda, pg, zn, zm);
}

void CodeGenerator::Fmls(const ZRegS &zda, const PRegBase &pg, const ZRegS &zn, const ZRegS &zm)
{
    SveFpMultAccumAddend(VAL_1, zda, pg, zn, zm);
}

void CodeGenerator::Fmad(const ZRegS &zdn, const PRegBase &pg, const ZRegS &zm, const ZRegS &za)
{
    SveFpMultAccumMulti(0, zdn, pg, zm, za);
}

void CodeGenerator::Fnmsb(const ZRegS &zdn, const PRegBase &pg, const ZRegS &zm, const ZRegS &za)
{
    SveFpMultAccumMulti(VAL_3, zdn, pg, zm, za);
}

void CodeGenerator::Ld1sb(const ZRegS &zt, const PRegBase &pg, const AdrSc32U &adr)
{
    Sve32GatherLdSc32U(0, 0, 0, zt, pg, adr);
}

void CodeGenerator::Ld1b(const ZRegS &zt, const PRegBase &pg, const AdrSc32U &adr)
{
    Sve32GatherLdSc32U(0, VAL_1, 0, zt, pg, adr);
}

void CodeGenerator::Ld1h(const ZRegS &zt, const PRegBase &pg, const AdrSc32U &adr)
{
    Sve32GatherLdSc32U(VAL_1, VAL_1, 0, zt, pg, adr);
}

void CodeGenerator::Ld1w(const ZRegS &zt, const PRegBase &pg, const AdrSc32U &adr)
{
    Sve32GatherLdSc32U(VAL_2, VAL_1, 0, zt, pg, adr);
}

void CodeGenerator::Ld1w(const ZRegS &zt, const PRegBase &pg, const AdrSc32S &adr)
{
    Sve32GatherLdWSc32S(VAL_1, 0, zt, pg, adr);
}

void CodeGenerator::Prfh(const PrfopSve prfopSve, const PRegBase &pg, const AdrNoOfs &adr)
{
    Sve32ContiPfScImm(prfopSve, VAL_1, pg, adr);
}

void CodeGenerator::Prfw(const PrfopSve prfopSve, const PRegBase &pg, const AdrScImm &adr)
{
    Sve32ContiPfScImm(prfopSve, VAL_2, pg, adr);
}

void CodeGenerator::Prfw(const PrfopSve prfopSve, const PRegBase &pg, const AdrNoOfs &adr)
{
    Sve32ContiPfScImm(prfopSve, VAL_2, pg, adr);
}

void CodeGenerator::Ld1rb(const ZRegS &zt, const PRegBase &pg, const AdrNoOfs &adr)
{
    SveLoadAndBcElem(0, VAL_2, zt, pg, adr);
}

void CodeGenerator::Ld1rh(const ZRegH &zt, const PRegBase &pg, const AdrNoOfs &adr)
{
    SveLoadAndBcElem(VAL_1, VAL_1, zt, pg, adr);
}

void CodeGenerator::Ld1rh(const ZRegH &zt, const PRegBase &pg, const AdrScImm &adr)
{
    SveLoadAndBcElem(VAL_1, VAL_1, zt, pg, adr);
}

void CodeGenerator::Ld1rh(const ZRegS &zt, const PRegBase &pg, const AdrNoOfs &adr)
{
    SveLoadAndBcElem(VAL_1, VAL_2, zt, pg, adr);
}

void CodeGenerator::Ld1rh(const ZRegS &zt, const PRegBase &pg, const AdrScImm &adr)
{
    SveLoadAndBcElem(VAL_1, VAL_2, zt, pg, adr);
}

void CodeGenerator::Ld1rw(const ZRegS &zt, const PRegBase &pg, const AdrScImm &adr)
{
    SveLoadAndBcElem(VAL_2, VAL_2, zt, pg, adr);
}

void CodeGenerator::Ld1rw(const ZRegS &zt, const PRegBase &pg, const AdrNoOfs &adr)
{
    SveLoadAndBcElem(VAL_2, VAL_2, zt, pg, adr);
}

void CodeGenerator::Ld1rsb(const ZRegS &zt, const PRegBase &pg, const AdrNoOfs &adr)
{
    SveLoadAndBcElem(VAL_3, VAL_1, zt, pg, adr);
}

void CodeGenerator::Ldr(const PRegBase &pt, const AdrScImm &adr)
{
    SveLoadPredReg(pt, adr);
}

void CodeGenerator::Ldr(const PRegBase &pt, const AdrNoOfs &adr)
{
    SveLoadPredReg(pt, adr);
}

void CodeGenerator::Ldr(const ZRegBase &zt, const AdrScImm &adr)
{
    SveLoadPredVec(zt, adr);
}

void CodeGenerator::Ldr(const ZRegBase &zt, const AdrNoOfs &adr)
{
    SveLoadPredVec(zt, adr);
}

void CodeGenerator::Ld1b(const ZRegB &zt, const PRegBase &pg, const AdrNoOfs &adr)
{
    SveContiLdScImm(0, zt, pg, adr);
}

void CodeGenerator::Ld1b(const ZRegB &zt, const PRegBase &pg, const AdrScImm &adr)
{
    SveContiLdScImm(0, zt, pg, adr);
}

void CodeGenerator::Ld1b(const ZRegS &zt, const PRegBase &pg, const AdrScImm &adr)
{
    SveContiLdScImm(VAL_2, zt, pg, adr);
}

void CodeGenerator::Ld1b(const ZRegS &zt, const PRegBase &pg, const AdrNoOfs &adr)
{
    SveContiLdScImm(VAL_2, zt, pg, adr);
}

void CodeGenerator::Ld1h(const ZRegH &zt, const PRegBase &pg, const AdrNoOfs &adr)
{
    SveContiLdScImm(VAL_5, zt, pg, adr);
}

void CodeGenerator::Ld1h(const ZRegH &zt, const PRegBase &pg, const AdrScImm &adr)
{
    SveContiLdScImm(VAL_5, zt, pg, adr);
}

void CodeGenerator::Ld1h(const ZRegH &zt, const PRegBase &pg, const AdrScSc &adr)
{
    SveContiLdScSc(VAL_5, zt, pg, adr);
}

void CodeGenerator::Ld1h(const ZRegS &zt, const PRegBase &pg, const AdrNoOfs &adr)
{
    SveContiLdScImm(VAL_6, zt, pg, adr);
}

void CodeGenerator::Ld1w(const ZRegS &zt, const PRegBase &pg, const AdrScImm &adr)
{
    SveContiLdScImm(VAL_10, zt, pg, adr);
}

void CodeGenerator::Ld1w(const ZRegS &zt, const PRegBase &pg, const AdrNoOfs &adr)
{
    SveContiLdScImm(VAL_10, zt, pg, adr);
}

void CodeGenerator::Ld1w(const ZRegD &zt, const PRegBase &pg, const AdrNoOfs &adr)
{
    SveContiLdScImm(VAL_11, zt, pg, adr);
}

void CodeGenerator::Ld1sb(const ZRegS &zt, const PRegBase &pg, const AdrScImm &adr)
{
    SveContiLdScImm(VAL_13, zt, pg, adr);
}

void CodeGenerator::Ld1sb(const ZRegS &zt, const PRegBase &pg, const AdrNoOfs &adr)
{
    SveContiLdScImm(VAL_13, zt, pg, adr);
}

void CodeGenerator::Ld1d(const ZRegD &zt, const PRegBase &pg, const AdrNoOfs &adr)
{
    SveContiLdScImm(VAL_15, zt, pg, adr);
}

void CodeGenerator::Ld1w(const ZRegS &zt, const PRegBase &pg, const AdrScSc &adr)
{
    SveContiLdScSc(VAL_10, zt, pg, adr);
}

void CodeGenerator::Ld1rqw(const ZRegS &zt, const PRegBase &pg, const AdrNoOfs &adr)
{
    SveLdBcQuadScImm(VAL_2, 0, zt, pg, adr);
}

void CodeGenerator::Ld1rqh(const ZRegH &zt, const PRegBase &pg, const AdrNoOfs &adr)
{
    SveLdBcQuadScImm(VAL_1, 0, zt, pg, adr);
}

void CodeGenerator::Stnt1w(const ZRegS &zt, const PRegBase &pg, const AdrNoOfs &adr)
{
    SveContiNTStScImm(VAL_2, zt, pg, adr);
}

void CodeGenerator::Stnt1d(const ZRegD &zt, const PRegBase &pg, const AdrNoOfs &adr)
{
    SveContiNTStScImm(VAL_3, zt, pg, adr);
}

void CodeGenerator::St1b(const ZRegB &zt, const PRegBase &pg, const AdrScImm &adr)
{
    SveContiStScImm(0, zt, pg, adr);
}

void CodeGenerator::St1b(const ZRegS &zt, const PRegBase &pg, const AdrScImm &adr)
{
    SveContiStScImm(0, zt, pg, adr);
}

void CodeGenerator::St1b(const ZRegB &zt, const PRegBase &pg, const AdrNoOfs &adr)
{
    SveContiStScImm(0, zt, pg, adr);
}

void CodeGenerator::St1b(const ZRegS &zt, const PRegBase &pg, const AdrNoOfs &adr)
{
    SveContiStScImm(0, zt, pg, adr);
}

void CodeGenerator::St1h(const ZRegH &zt, const PRegBase &pg, const AdrNoOfs &adr)
{
    SveContiStScImm(VAL_1, zt, pg, adr);
}

void CodeGenerator::St1h(const ZRegS &zt, const PRegBase &pg, const AdrNoOfs &adr)
{
    SveContiStScImm(VAL_1, zt, pg, adr);
}

void CodeGenerator::St1w(const ZRegS &zt, const PRegBase &pg, const AdrScImm &adr)
{
    SveContiStScImm(VAL_2, zt, pg, adr);
}

void CodeGenerator::St1w(const ZRegS &zt, const PRegBase &pg, const AdrNoOfs &adr)
{
    SveContiStScImm(VAL_2, zt, pg, adr);
}

void CodeGenerator::St1w(const ZRegS &zt, const PRegBase &pg, const AdrScSc &adr)
{
    SveContiStScSc(VAL_2, zt, pg, adr);
}

void CodeGenerator::Str(const PRegBase &pt, const AdrScImm &adr)
{
    SveStorePredReg(pt, adr);
}

void CodeGenerator::Str(const PRegBase &pt, const AdrNoOfs &adr)
{
    SveStorePredReg(pt, adr);
}

void CodeGenerator::Str(const ZRegBase &zt, const AdrScImm &adr)
{
    SveStorePredVec(zt, adr);
}

void CodeGenerator::Str(const ZRegBase &zt, const AdrNoOfs &adr)
{
    SveStorePredVec(zt, adr);
}

void CodeGenerator::Fcvtns(const VReg4S &vd, const VReg4S &vn)
{
    AdvSimd2RegMiscSz0x(0, VAL_26, vd, vn);
}

void CodeGenerator::Tbz(const WReg &rt, const uint32_t imm, const Label &label)
{
    TestBr(0, rt, imm, label);
}

void CodeGenerator::Tbz(const XReg &rt, const uint32_t imm, const Label &label)
{
    TestBr(0, rt, imm, label);
}

void CodeGenerator::Fmov(const SReg &d, const WReg &n)
{
    ConversionFpInt(0, 0, 0, 0, VAL_7, d, n);
}

void CodeGenerator::Fmov(const WReg &d, const SReg &n)
{
    ConversionFpInt(0, 0, 0, 0, VAL_6, d, n);
}

void CodeGenerator::Fmov(const XReg &d, const DReg &n)
{
    ConversionFpInt(VAL_1, 0, VAL_1, 0, VAL_6, d, n);
}

void CodeGenerator::Bfmlalb(const ZRegS &zda, const ZRegH &zn, const ZRegH &zm)
{
    SveFpWideMultAddGroup(0x3820, zda, zn, zm);
}

void CodeGenerator::Bfmlalb(const ZRegS &zda, const ZRegH &zn, const ZRegHElem &zm)
{
    SveFpWideMultAddIndexedGroup(0x3810, zda, zn, zm);
}

} // ARMCG