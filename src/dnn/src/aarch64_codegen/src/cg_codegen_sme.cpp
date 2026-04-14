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

#ifdef KUDNNL_920Pro
void CodeGenerator::MatrixComputationStart(StreamMode mod)
{
    uint32_t mask = static_cast<uint32_t>(mod);
    VerifyIncList(mask, {static_cast<uint32_t>(StreamMode::SM), static_cast<uint32_t>(StreamMode::ZA),
        static_cast<uint32_t>(StreamMode::SMZA)}, Err::ILLEGAL_CONST_VALUE);
    uint32_t code = Concat({LeftShift(0x1AA068, BITFIELD_11), LeftShift(mask, BITFIELD_9),
        LeftShift(0x17F, BITFIELD_0)});
    WriteData(code);
}

void CodeGenerator::MatrixComputationStop(StreamMode mod)
{
    uint32_t mask = static_cast<uint32_t>(mod);
    VerifyIncList(mask, {static_cast<uint32_t>(StreamMode::SM), static_cast<uint32_t>(StreamMode::ZA),
        static_cast<uint32_t>(StreamMode::SMZA)}, Err::ILLEGAL_CONST_VALUE);
    uint32_t code = Concat({LeftShift(0x1AA068, BITFIELD_11), LeftShift(mask, BITFIELD_9),
        LeftShift(0x7F, BITFIELD_0)});
    WriteData(code);
}

void CodeGenerator::MatrixComputationZero(const std::initializer_list<ZARegBase> &list)
{
    if (list.size() == 0) {
        // zeroes nothing, for the sake of instruction completeness
        uint32_t code = 0xC0080000;
        WriteData(code);
    } else {
        // currently only supports clearing the entire za storage
        uint32_t code = 0xC00800FF;
        WriteData(code);
    }
}

uint32_t CodeGenerator::EncodeSourcesMOP(const ZRegBase &zm, const PRegBase &pm, const PRegBase &pn, const ZRegBase &zn)
{
    VerifyIncRange(pn.GetIdx(), 0, MatrixComputation_PG_RANGE, Err::ILLEGAL_REG_IDX);
    VerifyIncRange(pm.GetIdx(), 0, MatrixComputation_PG_RANGE, Err::ILLEGAL_REG_IDX);
    uint32_t encode = Concat({LeftShift(zm.GetIdx(), BITFIELD_11), LeftShift(pm.GetIdx(), BITFIELD_8),
        LeftShift(pn.GetIdx(), BITFIELD_5), LeftShift(zn.GetIdx(), BITFIELD_0)});
    return encode;
}

void CodeGenerator::MatrixComputationOuterProductAccNonWidening(const ZARegS &zada, const PRegBase &pn, const PRegBase &pm,
    const ZRegS &zn, const ZRegS &zm)
{
    VerifyIncRange(zada.GetIdx(), 0, MatrixComputation_ZASLICE_OFFS_S_RANGE, Err::ILLEGAL_REG_IDX);
    uint32_t code = Concat({LeftShift(0x404, BITFIELD_21), LeftShift(EncodeSourcesMOP(zm, pm, pn, zn), BITFIELD_5),
        LeftShift(zada.GetIdx(), BITFIELD_0)});
    WriteData(code);
}

void CodeGenerator::MatrixComputationOuterProductAcc2Way(const ZARegS &zada, const PRegBase &pn, const PRegBase &pm,
    const ZRegH &zn, const ZRegH &zm)
{
    VerifyIncRange(zada.GetIdx(), 0, MatrixComputation_ZASLICE_OFFS_S_RANGE, Err::ILLEGAL_REG_IDX);
    uint32_t code = Concat({LeftShift(0x40d, BITFIELD_21), LeftShift(EncodeSourcesMOP(zm, pm, pn, zn), BITFIELD_5),
    LeftShift(zada.GetIdx(), BITFIELD_0)});
    WriteData(code);
}

void CodeGenerator::MatrixComputationOuterProductAccSigned4Way(const ZARegS &zada, const PRegBase &pn, const PRegBase &pm,
    const ZRegB &zn, const ZRegB &zm)
{
    VerifyIncRange(zada.GetIdx(), 0, MatrixComputation_ZASLICE_OFFS_S_RANGE, Err::ILLEGAL_REG_IDX);
    uint32_t code = Concat({LeftShift(0x504, BITFIELD_21), LeftShift(EncodeSourcesMOP(zm, pm, pn, zn), BITFIELD_5),
    LeftShift(zada.GetIdx(), BITFIELD_0)});
    WriteData(code);
}

void CodeGenerator::MatrixComputationOuterProductSubSigned4Way(const ZARegS &zada, const PRegBase &pn, const PRegBase &pm,
    const ZRegB &zn, const ZRegB &zm)
{
    VerifyIncRange(zada.GetIdx(), 0, MatrixComputation_ZASLICE_OFFS_S_RANGE, Err::ILLEGAL_REG_IDX);
    uint32_t code = Concat({LeftShift(0x504, BITFIELD_21), LeftShift(EncodeSourcesMOP(zm, pm, pn, zn), BITFIELD_5),
        LeftShift(0x4, BITFIELD_2), LeftShift(zada.GetIdx(), BITFIELD_0)});
    WriteData(code);
}

void CodeGenerator::MatrixComputationOuterProductAccSignedUnsigned(const ZARegS &zada, const PRegBase &pn, const PRegBase &pm,
    const ZRegB &zn, const ZRegB &zm)
{
    VerifyIncRange(zada.GetIdx(), 0, MatrixComputation_ZASLICE_OFFS_S_RANGE, Err::ILLEGAL_REG_IDX);
    uint32_t code = Concat({LeftShift(0x505, BITFIELD_21), LeftShift(EncodeSourcesMOP(zm, pm, pn, zn), BITFIELD_5),
    LeftShift(zada.GetIdx(), BITFIELD_0)});
    WriteData(code);
}

void CodeGenerator::MatrixComputationOuterProductSubSignedUnsigned(const ZARegS &zada, const PRegBase &pn, const PRegBase &pm,
    const ZRegB &zn, const ZRegB &zm)
{
    VerifyIncRange(zada.GetIdx(), 0, MatrixComputation_ZASLICE_OFFS_S_RANGE, Err::ILLEGAL_REG_IDX);
    uint32_t code = Concat({LeftShift(0x505, BITFIELD_21), LeftShift(EncodeSourcesMOP(zm, pm, pn, zn), BITFIELD_5),
    LeftShift(0x4, BITFIELD_2), LeftShift(zada.GetIdx(), BITFIELD_0)});
    WriteData(code);
}

void CodeGenerator::MatrixComputationOuterProductAccUnsigned4Way(const ZARegS &zada, const PRegBase &pn, const PRegBase &pm,
    const ZRegB &zn, const ZRegB &zm)
{
    VerifyIncRange(zada.GetIdx(), 0, MatrixComputation_ZASLICE_OFFS_S_RANGE, Err::ILLEGAL_REG_IDX);
    uint32_t code = Concat({LeftShift(0x50D, BITFIELD_21), LeftShift(EncodeSourcesMOP(zm, pm, pn, zn), BITFIELD_5),
    LeftShift(zada.GetIdx(), BITFIELD_0)});
    WriteData(code);
}

void CodeGenerator::MatrixComputationOuterProductSubUnsigned4Way(const ZARegS &zada, const PRegBase &pn, const PRegBase &pm,
    const ZRegB &zn, const ZRegB &zm)
{
    VerifyIncRange(zada.GetIdx(), 0, MatrixComputation_ZASLICE_OFFS_S_RANGE, Err::ILLEGAL_REG_IDX);
    uint32_t code = Concat({LeftShift(0x50D, BITFIELD_21), LeftShift(EncodeSourcesMOP(zm, pm, pn, zn), BITFIELD_5),
    LeftShift(0x4, BITFIELD_2), LeftShift(zada.GetIdx(), BITFIELD_0)});
    WriteData(code);
}

void CodeGenerator::MatrixComputationOuterProductAccUnsignedSigned(const ZARegS &zada, const PRegBase &pn, const PRegBase &pm,
    const ZRegB &zn, const ZRegB &zm)
{
    VerifyIncRange(zada.GetIdx(), 0, MatrixComputation_ZASLICE_OFFS_S_RANGE, Err::ILLEGAL_REG_IDX);
    uint32_t code = Concat({LeftShift(0x50C, BITFIELD_21), LeftShift(EncodeSourcesMOP(zm, pm, pn, zn), BITFIELD_5),
    LeftShift(zada.GetIdx(), BITFIELD_0)});
    WriteData(code);
}

void CodeGenerator::MatrixComputationOuterProductSubUnsignedSigned(const ZARegS &zada, const PRegBase &pn, const PRegBase &pm,
    const ZRegB &zn, const ZRegB &zm)
{
    VerifyIncRange(zada.GetIdx(), 0, MatrixComputation_ZASLICE_OFFS_S_RANGE, Err::ILLEGAL_REG_IDX);
    uint32_t code = Concat({LeftShift(0x50C, BITFIELD_21), LeftShift(EncodeSourcesMOP(zm, pm, pn, zn), BITFIELD_5),
    LeftShift(0x4, BITFIELD_2), LeftShift(zada.GetIdx(), BITFIELD_0)});
    WriteData(code);
}

void CodeGenerator::MatrixComputationMovTile2Vec(const ZRegB &zd, const PRegBase &pg, const ZARegB &zan,
    bool v, const WReg &ws, uint32_t offs)
{
    VerifyIncRange(zan.GetIdx(), 0, MatrixComputation_ZATILE_B_RANGE, Err::ILLEGAL_REG_IDX);
    VerifyIncRange(ws.GetIdx(), MatrixComputation_ZASLICE_INDEX_RANGE_BEGIN, MatrixComputation_ZASLICE_INDEX_RANGE_END, Err::ILLEGAL_REG_IDX);
    VerifyIncRange(offs, 0, MatrixComputation_ZASLICE_OFFS_B_RANGE, Err::ILLEGAL_IMM_RANGE);
    VerifyIncRange(pg.GetIdx(), 0, MatrixComputation_PG_RANGE, Err::ILLEGAL_REG_IDX);
    uint32_t rsField = ws.GetIdx() - MatrixComputation_ZASLICE_INDEX_RANGE_BEGIN;
    uint32_t code = Concat({LeftShift(0xC002, BITFIELD_16), LeftShift(v ? 0x1 : 0x0, BITFIELD_15),
        LeftShift(rsField, BITFIELD_13), LeftShift(pg.GetIdx(), BITFIELD_10), LeftShift(0, BITFIELD_9),
        LeftShift(offs, BITFIELD_5), LeftShift(zd.GetIdx(), BITFIELD_0)});
    WriteData(code);
}

void CodeGenerator::MatrixComputationMovTile2Vec(const ZRegS &zd, const PRegBase &pg, const ZARegS &zan,
    bool v, const WReg &ws, uint32_t offs)
{
    VerifyIncRange(zan.GetIdx(), 0, MatrixComputation_ZATILE_S_RANGE, Err::ILLEGAL_REG_IDX);
    VerifyIncRange(ws.GetIdx(), MatrixComputation_ZASLICE_INDEX_RANGE_BEGIN, MatrixComputation_ZASLICE_INDEX_RANGE_END, Err::ILLEGAL_REG_IDX);
    VerifyIncRange(offs, 0, MatrixComputation_ZASLICE_OFFS_S_RANGE, Err::ILLEGAL_IMM_RANGE);
    VerifyIncRange(pg.GetIdx(), 0, MatrixComputation_PG_RANGE, Err::ILLEGAL_REG_IDX);
    uint32_t rsField = ws.GetIdx() - MatrixComputation_ZASLICE_INDEX_RANGE_BEGIN;
    uint32_t code = Concat({LeftShift(0xC082, BITFIELD_16), LeftShift(v ? 0x1 : 0x0, BITFIELD_15),
        LeftShift(rsField, BITFIELD_13), LeftShift(pg.GetIdx(), BITFIELD_10), LeftShift(zan.GetIdx(), BITFIELD_7),
        LeftShift(offs, BITFIELD_5), LeftShift(zd.GetIdx(), BITFIELD_0)});
    WriteData(code);
}

void CodeGenerator::MatrixComputationMovVec2Tile(const ZARegB &zad, bool v, const WReg &ws, uint32_t offs,
    const PRegBase &pg, const ZRegB &zn)
{
    VerifyIncRange(zad.GetIdx(), 0, MatrixComputation_ZATILE_B_RANGE, Err::ILLEGAL_REG_IDX);
    VerifyIncRange(ws.GetIdx(), MatrixComputation_ZASLICE_INDEX_RANGE_BEGIN, MatrixComputation_ZASLICE_INDEX_RANGE_END, Err::ILLEGAL_REG_IDX);
    VerifyIncRange(offs, 0, MatrixComputation_ZASLICE_OFFS_B_RANGE, Err::ILLEGAL_IMM_RANGE);
    VerifyIncRange(pg.GetIdx(), 0, MatrixComputation_PG_RANGE, Err::ILLEGAL_REG_IDX);
    uint32_t rsField = ws.GetIdx() - MatrixComputation_ZASLICE_INDEX_RANGE_BEGIN;
    uint32_t code = Concat({LeftShift(0xC000, BITFIELD_16), LeftShift(v ? 0x1 : 0x0, BITFIELD_15),
        LeftShift(rsField, BITFIELD_13), LeftShift(pg.GetIdx(), BITFIELD_10), LeftShift(zn.GetIdx(), BITFIELD_5),
        LeftShift(offs, BITFIELD_0)});
    WriteData(code);
}

void CodeGenerator::MatrixComputationMovVec2Tile(const ZARegS &zad, bool v, const WReg &ws, uint32_t offs,
    const PRegBase &pg, const ZRegS &zn)
{
    VerifyIncRange(zad.GetIdx(), 0, MatrixComputation_ZATILE_S_RANGE, Err::ILLEGAL_REG_IDX);
    VerifyIncRange(ws.GetIdx(), MatrixComputation_ZASLICE_INDEX_RANGE_BEGIN, MatrixComputation_ZASLICE_INDEX_RANGE_END, Err::ILLEGAL_REG_IDX);
    VerifyIncRange(offs, 0, MatrixComputation_ZASLICE_OFFS_S_RANGE, Err::ILLEGAL_IMM_RANGE);
    VerifyIncRange(pg.GetIdx(), 0, MatrixComputation_PG_RANGE, Err::ILLEGAL_REG_IDX);
    uint32_t rsField = ws.GetIdx() - MatrixComputation_ZASLICE_INDEX_RANGE_BEGIN;
    uint32_t code = Concat({LeftShift(0xC080, BITFIELD_16), LeftShift(v ? 0x1 : 0x0, BITFIELD_15),
        LeftShift(rsField, BITFIELD_13), LeftShift(pg.GetIdx(), BITFIELD_10), LeftShift(zn.GetIdx(), BITFIELD_5),
        LeftShift(zad.GetIdx(), BITFIELD_2), LeftShift(offs, BITFIELD_0)});
    WriteData(code);
}

void CodeGenerator::MatrixComputationStoreTile(const ZARegB &zat, bool v, const WReg &ws, uint32_t offs,
    const PRegBase &pg, const AdrScSc &adr)
{
    VerifyIncRange(zat.GetIdx(), 0, MatrixComputation_ZATILE_B_RANGE, Err::ILLEGAL_REG_IDX);
    VerifyIncRange(ws.GetIdx(), MatrixComputation_ZASLICE_INDEX_RANGE_BEGIN, MatrixComputation_ZASLICE_INDEX_RANGE_END, Err::ILLEGAL_REG_IDX);
    VerifyIncRange(offs, 0, MatrixComputation_ZASLICE_OFFS_B_RANGE, Err::ILLEGAL_IMM_RANGE);
    VerifyIncRange(pg.GetIdx(), 0, MatrixComputation_PG_RANGE, Err::ILLEGAL_REG_IDX);
    VerifyEqual(adr.GetMod(), ShMod::LSL, Err::ILLEGAL_SHMOD);
    VerifyIncRange(adr.GetSh(), VAL_0, VAL_0, Err::ILLEGAL_IMM_RANGE);
    uint32_t rsField = ws.GetIdx() - MatrixComputation_ZASLICE_INDEX_RANGE_BEGIN;
    uint32_t code = Concat({LeftShift(0x701, BITFIELD_21),  LeftShift(adr.GetXm().GetIdx(), BITFIELD_16),
        LeftShift(v ? 0x1 : 0x0, BITFIELD_15), LeftShift(rsField, BITFIELD_13), LeftShift(pg.GetIdx(), BITFIELD_10),
        LeftShift(adr.GetXn().GetIdx(), BITFIELD_5), LeftShift(0x0, BITFIELD_4), LeftShift(offs, BITFIELD_0)});
    WriteData(code);
}

void CodeGenerator::MatrixComputationLoadTile(const ZARegB &zat, bool v, const WReg &ws, uint32_t offs,
    const PRegBase &pg, const AdrScSc &adr)
{
    VerifyIncRange(zat.GetIdx(), 0, MatrixComputation_ZATILE_B_RANGE, Err::ILLEGAL_REG_IDX);
    VerifyIncRange(ws.GetIdx(), MatrixComputation_ZASLICE_INDEX_RANGE_BEGIN, MatrixComputation_ZASLICE_INDEX_RANGE_END, Err::ILLEGAL_REG_IDX);
    VerifyIncRange(offs, 0, MatrixComputation_ZASLICE_OFFS_B_RANGE, Err::ILLEGAL_IMM_RANGE);
    VerifyIncRange(pg.GetIdx(), 0, MatrixComputation_PG_RANGE, Err::ILLEGAL_REG_IDX);
    VerifyEqual(adr.GetMod(), ShMod::LSL, Err::ILLEGAL_SHMOD);
    VerifyIncRange(adr.GetSh(), VAL_0, VAL_0, Err::ILLEGAL_IMM_RANGE);
    uint32_t rsField = ws.GetIdx() - MatrixComputation_ZASLICE_INDEX_RANGE_BEGIN;
    uint32_t code = Concat({LeftShift(0x700, BITFIELD_21),  LeftShift(adr.GetXm().GetIdx(), BITFIELD_16),
        LeftShift(v ? 0x1 : 0x0, BITFIELD_15), LeftShift(rsField, BITFIELD_13), LeftShift(pg.GetIdx(), BITFIELD_10),
        LeftShift(adr.GetXn().GetIdx(), BITFIELD_5), LeftShift(0x0, BITFIELD_4), LeftShift(offs, BITFIELD_0)});
    WriteData(code);
}

void CodeGenerator::MatrixComputationStoreTile(const ZARegS &zat, bool v, const WReg &ws, uint32_t offs,
    const PRegBase &pg, const AdrScSc &adr)
{
    VerifyIncRange(zat.GetIdx(), 0, MatrixComputation_ZATILE_S_RANGE, Err::ILLEGAL_REG_IDX);
    VerifyIncRange(ws.GetIdx(), MatrixComputation_ZASLICE_INDEX_RANGE_BEGIN, MatrixComputation_ZASLICE_INDEX_RANGE_END, Err::ILLEGAL_REG_IDX);
    VerifyIncRange(offs, 0, MatrixComputation_ZASLICE_OFFS_S_RANGE, Err::ILLEGAL_IMM_RANGE);
    VerifyIncRange(pg.GetIdx(), 0, MatrixComputation_PG_RANGE, Err::ILLEGAL_REG_IDX);
    VerifyEqual(adr.GetMod(), ShMod::LSL, Err::ILLEGAL_SHMOD);
    VerifyIncRange(adr.GetSh(), VAL_2, VAL_2, Err::ILLEGAL_IMM_RANGE);
    uint32_t rsField = ws.GetIdx() - MatrixComputation_ZASLICE_INDEX_RANGE_BEGIN;
    uint32_t code = Concat({LeftShift(0x705, BITFIELD_21),  LeftShift(adr.GetXm().GetIdx(), BITFIELD_16),
        LeftShift(v ? 0x1 : 0x0, BITFIELD_15), LeftShift(rsField, BITFIELD_13), LeftShift(pg.GetIdx(), BITFIELD_10),
        LeftShift(adr.GetXn().GetIdx(), BITFIELD_5), LeftShift(zat.GetIdx(), BITFIELD_2), LeftShift(offs, BITFIELD_0)});
    WriteData(code);
}

void CodeGenerator::MatrixComputationAddVec2Tile(const ZARegS &zada, bool v, const PRegBase &pn,
    const PRegBase &pm, const ZRegS &zn)
{
    VerifyIncRange(zada.GetIdx(), 0, MatrixComputation_ZATILE_S_RANGE, Err::ILLEGAL_REG_IDX);
    VerifyIncRange(pn.GetIdx(), 0, MatrixComputation_PG_RANGE, Err::ILLEGAL_REG_IDX);
    VerifyIncRange(pm.GetIdx(), 0, MatrixComputation_PG_RANGE, Err::ILLEGAL_REG_IDX);
    uint32_t code = Concat({LeftShift(0x6048, BITFIELD_17), LeftShift(v ? 0x1 : 0x0, BITFIELD_16),
        LeftShift(pm.GetIdx(), BITFIELD_13), LeftShift(pn.GetIdx(), BITFIELD_10),
        LeftShift(zn.GetIdx(), BITFIELD_5), LeftShift(zada.GetIdx(), BITFIELD_0)});
    WriteData(code);
}

void CodeGenerator::Smstart(StreamMode mod)
{
    MatrixComputationStart(mod);
}

void CodeGenerator::Smstop(StreamMode mod)
{
    MatrixComputationStop(mod);
}

void CodeGenerator::Zero(const std::initializer_list<ZARegBase> &list)
{
    MatrixComputationZero(list);
}

void CodeGenerator::Fmopa(const ZARegS &zada, const PRegBase &pn, const PRegBase &pm, const ZRegS &zn, const ZRegS &zm)
{
    MatrixComputationOuterProductAccNonWidening(zada, pn, pm, zn, zm);
}

void CodeGenerator::Fmopa(const ZARegS &zada, const PRegBase &pn, const PRegBase &pm, const ZRegH &zn, const ZRegH &zm)
{
    MatrixComputationOuterProductAcc2Way(zada, pn, pm, zn, zm);
}

void CodeGenerator::Smopa(const ZARegS &zada, const PRegBase &pn, const PRegBase &pm, const ZRegB &zn, const ZRegB &zm)
{
    MatrixComputationOuterProductAccSigned4Way(zada, pn, pm, zn, zm);
}

void CodeGenerator::Smops(const ZARegS &zada, const PRegBase &pn, const PRegBase &pm, const ZRegB &zn, const ZRegB &zm)
{
    MatrixComputationOuterProductSubSigned4Way(zada, pn, pm, zn, zm);
}

void CodeGenerator::Sumopa(const ZARegS &zada, const PRegBase &pn, const PRegBase &pm, const ZRegB &zn, const ZRegB &zm)
{
    MatrixComputationOuterProductAccSignedUnsigned(zada, pn, pm, zn, zm);
}

void CodeGenerator::Sumops(const ZARegS &zada, const PRegBase &pn, const PRegBase &pm, const ZRegB &zn, const ZRegB &zm)
{
    MatrixComputationOuterProductSubSignedUnsigned(zada, pn, pm, zn, zm);
}

void CodeGenerator::Umopa(const ZARegS &zada, const PRegBase &pn, const PRegBase &pm, const ZRegB &zn, const ZRegB &zm)
{
    MatrixComputationOuterProductAccUnsigned4Way(zada, pn, pm, zn, zm);
}

void CodeGenerator::Umops(const ZARegS &zada, const PRegBase &pn, const PRegBase &pm, const ZRegB &zn, const ZRegB &zm)
{
    MatrixComputationOuterProductSubUnsigned4Way(zada, pn, pm, zn, zm);
}

void CodeGenerator::Usmopa(const ZARegS &zada, const PRegBase &pn, const PRegBase &pm, const ZRegB &zn, const ZRegB &zm)
{
    MatrixComputationOuterProductAccUnsignedSigned(zada, pn, pm, zn, zm);
}

void CodeGenerator::Usmops(const ZARegS &zada, const PRegBase &pn, const PRegBase &pm, const ZRegB &zn, const ZRegB &zm)
{
    MatrixComputationOuterProductSubUnsignedSigned(zada, pn, pm, zn, zm);
}

void CodeGenerator::Mova(const ZRegB &zd, const PRegBase &pg, const ZARegHB &zan, const WReg &ws, uint32_t offs)
{
    MatrixComputationMovTile2Vec(zd, pg, ZARegB(zan.GetIdx()), false, ws, offs);
}

void CodeGenerator::Mova(const ZRegS &zd, const PRegBase &pg, const ZARegHS &zan, const WReg &ws, uint32_t offs)
{
    MatrixComputationMovTile2Vec(zd, pg, ZARegS(zan.GetIdx()), false, ws, offs);
}

void CodeGenerator::Mova(const ZRegB &zd, const PRegBase &pg, const ZARegVB &zan, const WReg &ws, uint32_t offs)
{
    MatrixComputationMovTile2Vec(zd, pg, ZARegB(zan.GetIdx()), true, ws, offs);
}

void CodeGenerator::Mova(const ZRegS &zd, const PRegBase &pg, const ZARegVS &zan, const WReg &ws, uint32_t offs)
{
    MatrixComputationMovTile2Vec(zd, pg, ZARegS(zan.GetIdx()), true, ws, offs);
}

void CodeGenerator::Mova(const ZARegHB &zad, const WReg &ws, uint32_t offs, const PRegBase &pg, const ZRegB &zn)
{
    MatrixComputationMovVec2Tile(ZARegB(zad.GetIdx()), false, ws, offs, pg, zn);
}

void CodeGenerator::Mova(const ZARegHS &zad, const WReg &ws, uint32_t offs, const PRegBase &pg, const ZRegS &zn)
{
    MatrixComputationMovVec2Tile(ZARegS(zad.GetIdx()), false, ws, offs, pg, zn);
}

void CodeGenerator::Mova(const ZARegVB &zad, const WReg &ws, uint32_t offs, const PRegBase &pg, const ZRegB &zn)
{
    MatrixComputationMovVec2Tile(ZARegB(zad.GetIdx()), true, ws, offs, pg, zn);
}

void CodeGenerator::Mova(const ZARegVS &zad, const WReg &ws, uint32_t offs, const PRegBase &pg, const ZRegS &zn)
{
    MatrixComputationMovVec2Tile(ZARegS(zad.GetIdx()), true, ws, offs, pg, zn);
}

void CodeGenerator::Mov(const ZRegB &zd, const PRegBase &pg, const ZARegHB &zan, const WReg &ws, uint32_t offs)
{
    Mova(zd, pg, zan, ws, offs);
}

void CodeGenerator::Mov(const ZRegB &zd, const PRegBase &pg, const ZARegVB &zan, const WReg &ws, uint32_t offs)
{
    Mova(zd, pg, zan, ws, offs);
}

void CodeGenerator::Mov(const ZARegHB &zad, const WReg &ws, uint32_t offs, const PRegBase &pg, const ZRegB &zn)
{
    Mova(zad, ws, offs, pg, zn);
}

void CodeGenerator::Mov(const ZARegVB &zad, const WReg &ws, uint32_t offs, const PRegBase &pg, const ZRegB &zn)
{
    Mova(zad, ws, offs, pg, zn);
}

void CodeGenerator::St1b(const ZARegHB &zat, const WReg &ws, uint32_t offs, const PRegBase &pg, const AdrScSc &adr)
{
    MatrixComputationStoreTile(ZARegB(zat.GetIdx()), false, ws, offs, pg, adr);
}

void CodeGenerator::St1b(const ZARegVB &zat, const WReg &ws, uint32_t offs, const PRegBase &pg, const AdrScSc &adr)
{
    MatrixComputationStoreTile(ZARegB(zat.GetIdx()), true, ws, offs, pg, adr);
}

void CodeGenerator::Ld1b(const ZARegHB &zat, const WReg &ws, uint32_t offs, const PRegBase &pg, const AdrScSc &adr)
{
    MatrixComputationLoadTile(ZARegB(zat.GetIdx()), false, ws, offs, pg, adr);
}

void CodeGenerator::Ld1b(const ZARegVB &zat, const WReg &ws, uint32_t offs, const PRegBase &pg, const AdrScSc &adr)
{
    MatrixComputationLoadTile(ZARegB(zat.GetIdx()), true, ws, offs, pg, adr);
}

void CodeGenerator::St1w(const ZARegHS &zat, const WReg &ws, uint32_t offs, const PRegBase &pg, const AdrScSc &adr)
{
    MatrixComputationStoreTile(ZARegS(zat.GetIdx()), false, ws, offs, pg, adr);
}

void CodeGenerator::St1w(const ZARegVS &zat, const WReg &ws, uint32_t offs, const PRegBase &pg, const AdrScSc &adr)
{
    MatrixComputationStoreTile(ZARegS(zat.GetIdx()), true, ws, offs, pg, adr);
}

void CodeGenerator::Addha(const ZARegS &zada, const PRegBase &pn, const PRegBase &pm, const ZRegS &zn)
{
    MatrixComputationAddVec2Tile(zada, false, pn, pm, zn);
}

void CodeGenerator::Addva(const ZARegS &zada, const PRegBase &pn, const PRegBase &pm, const ZRegS &zn)
{
    MatrixComputationAddVec2Tile(zada, true, pn, pm, zn);
}

#endif

} // ARMCG