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

void CodeGenerator::Ready(ProtectMode mode)
{
    if (HasUndefinedLabel()) {
        throw Error(Err::LABEL_IS_NOT_FOUND);
    }
    if (IsAutoGrow()) {
        CalcJmpAddress();
    }
    if (UseProtect()) {
        SetProtectMode(mode);
    }
    ClearCache(static_cast<void *>(GetCode()), static_cast<void *>(GetCurr()));
}

void CodeGenerator::ClearCache(void *begin, void *end)
{
    __builtin___clear_cache(static_cast<char *>(begin), static_cast<char *>(end));
}

// armv8/sve psuedo code function
bool CodeGenerator::CheckPtn(uint64_t v, uint32_t esize, uint32_t size)
{
    assert(esize != 0);
    uint32_t maxNum = size / esize;
    std::vector<uint64_t> ptns;
    for (uint32_t i = 0; i < maxNum; ++i) {
        ptns.push_back((v >> (esize * i)) & Ones(esize));
    }
    return std::all_of(ptns.begin(), ptns.end(), [&ptns](uint64_t x) { return x == ptns[0]; });
}

uint32_t CodeGenerator::GetPtnSize(uint64_t v, uint32_t size)
{
    uint32_t esize;
    for (esize = VAL_2; esize <= size; esize <<= 1) {
        if (CheckPtn(v, esize, size)) {
            break;
        }
    }
    return esize;
}

uint32_t CodeGenerator::GetPtnRotateNum(uint64_t ptn, uint32_t ptnSize)
{
    assert(ptn != 0 && (ptn & Ones(ptnSize)) != Ones(ptnSize));
    uint32_t num;
    for (num = 0; Msb(ptn, ptnSize) || !Lsb(ptn); ++num) {
        ptn = LRotate(ptn, ptnSize, VAL_1);
    }
    return num;
}

uint32_t CodeGenerator::CountOneBit(uint64_t v, uint32_t size)
{
    uint64_t num = 0;
    for (uint32_t i = 0; i < size; ++i) {
        num += Lsb(v);
        v >>= BITFIELD_1;
    };
    return static_cast<uint32_t>(num);
}

uint32_t CodeGenerator::CountSeqOneBit(uint64_t v, uint32_t size)
{
    uint32_t num;
    for (num = 0; num < size && Lsb(v); ++num) {
        v >>= BITFIELD_1;
    };
    return num;
}

// 8bitFloat = 1(sign) + 3(exponent) + 4(mantissa)
uint32_t CodeGenerator::Code8bitFloat(double x)
{
    uint64_t u;
    MoveData(u, x);
    uint32_t sign = (u >> BIT_63) & 1;
    int e = int((u >> BITFIELD_52) & Ones(BITFIELD_11)) - VAL_1023;
    if (e < SIZE_NEG_3 || e > VAL_4) {
        throw Error(Err::ILLEGAL_IMM_VALUE);
    }
    int e1 = e + VAL_7;
    uint32_t exp = static_cast<uint32_t>(e1) & VAL_7;
    uint64_t m = u & Ones(BITFIELD_52);
    if (m & Ones(BITFIELD_48)) {
        throw Error(Err::ILLEGAL_IMM_VALUE);
    }
    return uint32_t((sign << BITFIELD_7) | (exp << BITFIELD_4) | (m >> BITFIELD_48));
}

uint32_t CodeGenerator::GenQ(const VRegElem &reg)
{
    uint32_t pos = 0;
    switch (reg.GetBit()) {
        case VAL_8:
            pos = VAL_3;
            break;
        case VAL_16:
            pos = VAL_2;
            break;
        case VAL_32:
            pos = VAL_1;
            break;
        case VAL_64:
            pos = 0;
            break;
        default:
            pos = 0;
    }
    return Field(reg.GetElemIdx(), pos, pos);
}

uint32_t CodeGenerator::GenSize(const Reg &reg)
{
    uint32_t size = 0;
    switch (reg.GetBit()) {
        case VAL_8:
            size = 0;
            break;
        case VAL_16:
            size = VAL_1;
            break;
        case VAL_32:
            size = VAL_2;
            break;
        case VAL_64:
            size = VAL_3;
            break;
        case VAL_128:
            size = VAL_4;
            break;
        default:
            size = 0;
    }
    return size;
}

uint32_t CodeGenerator::GenSizeEnc(const VRegElem &reg)
{
    uint32_t size = 0;
    switch (reg.GetBit()) {
        case VAL_8:
            size = Field(reg.GetElemIdx(), BITFIELD_1, 0);
            break;
        case VAL_16:
            size = Field(reg.GetElemIdx(), 0, 0) << 1;
            break;
        case VAL_32:
            size = 0;
            break;
        case VAL_64:
            size = VAL_1;
            break;
        default:
            size = 0;
    }
    return size;
}

uint32_t CodeGenerator::GenS(const VRegElem &reg)
{
    uint32_t s = 0;
    switch (reg.GetBit()) {
        case VAL_8:
            s = Field(reg.GetElemIdx(), VAL_2, VAL_2);
            break;
        case VAL_16:
            s = Field(reg.GetElemIdx(), BITFIELD_1, BITFIELD_1);
            break;
        case VAL_32:
            s = Field(reg.GetElemIdx(), 0, 0);
            break;
        case VAL_64:
            s = 0;
            break;
        default:
            s = 0;
    }
    return s;
}

uint32_t CodeGenerator::GenNImmrImms(uint64_t imm, uint32_t size)
{
    // check imm
    if (imm == 0 || imm == Ones(size)) {
        throw Error(Err::ILLEGAL_IMM_VALUE);
    }
    auto ptnSize = GetPtnSize(imm, size);
    auto ptn = imm & Ones(ptnSize);
    auto rotateNum = GetPtnRotateNum(ptn, ptnSize);
    auto rotatePtn = LRotate(ptn, ptnSize, rotateNum);
    auto oneBitNum = CountOneBit(rotatePtn, ptnSize);
    auto seqOneBitNum = CountSeqOneBit(rotatePtn, ptnSize);
    // check ptn
    if (oneBitNum != seqOneBitNum) {
        throw Error(Err::ILLEGAL_IMM_VALUE);
    }
    uint32_t n = (ptnSize > VAL_32) ? 1 : 0;
    uint32_t immr = rotateNum;
    uint32_t imms =
        static_cast<uint32_t>((Ones(VAL_6) & ~Ones(static_cast<uint32_t>(std::log2(ptnSize)) + 1)) | (oneBitNum - 1));
    return (n << VAL_12) | (immr << VAL_6) | imms;
}

// generate relative address for label offset
uint64_t CodeGenerator::GenLabelOffset(const Label &label, const JmpLabel &jmpL)
{
    size_t offset = 0;
    uint64_t labelOffset = 0;
    if (labelMgr_.GetOffset(offset, label)) {
        labelOffset = (offset - size_) * CSIZE;
    } else {
        labelMgr_.AddUndefinedLabel(label, jmpL);
    }
    return labelOffset;
}

} // namespace ARMCG
