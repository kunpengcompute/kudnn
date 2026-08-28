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

#ifndef CG_MNEMONIC_TPL_HPP
#define CG_MNEMONIC_TPL_HPP

template <typename T, typename std::enable_if<std::is_unsigned<T>::value, std::nullptr_t>::type = nullptr>
void AddImm(const XReg &dst, const XReg &src, T imm, const XReg &tmp)
{
    // this AddImm function allows dst == src
    // but tmp must be different from src
    uint64_t bitPtn = static_cast<uint64_t>(imm);
    // Add(immediate) supports unsigned imm12
    const uint64_t imM12Mask = ~uint64_t(0xfff);
    // <= 4095
    if ((bitPtn & imM12Mask) == 0) {
        Add(dst, src, static_cast<uint32_t>(imm & 0xfff));
        return;
    }
    MovImm(tmp, imm);
    Add(dst, src, tmp);
    return;
}

template <typename T, typename std::enable_if<std::is_signed<T>::value, std::nullptr_t>::type = nullptr>
void AddImm(const XReg &dst, const XReg &src, T imm, const XReg &tmp)
{
    // this AddImm function allows dst == src
    // but tmp must be different from src
    // sign bit must be extended
    int64_t bitPtn = static_cast<int64_t>(imm);
    // Add(immediate) supports unsigned imm12
    if (imm >= 0) {
        const uint64_t imM12Mask = ~uint64_t(0xfff);
        // <= 4095
        if ((bitPtn & imM12Mask) == 0) {
            Add(dst, src, static_cast<uint32_t>(imm & 0xfff));
            return;
        }
    }
    MovImm(tmp, imm);
    Add(dst, src, tmp);
    return;
}

template <typename T, typename std::enable_if<std::is_unsigned<T>::value, std::nullptr_t>::type = nullptr>
void SubImm(const XReg &dst, const XReg &src, T imm, const XReg &tmp)
{
    // this SubImm function allows dst == src
    // but tmp must be different from src
    uint64_t bitPtn = static_cast<uint64_t>(imm);
    // Add(immediate) supports unsigned imm12
    const uint64_t imM12Mask = ~uint64_t(0xfff);
    // <= 4095
    if ((bitPtn & imM12Mask) == 0) {
        Sub(dst, src, static_cast<uint32_t>(imm & 0xfff));
        return;
    }
    MovImm(tmp, imm);
    Sub(dst, src, tmp);
    return;
}

template <typename T, typename std::enable_if<std::is_signed<T>::value, std::nullptr_t>::type = nullptr>
void SubImm(const XReg &dst, const XReg &src, T imm, const XReg &tmp)
{
    // this SubImm function allows dst == src
    // but tmp must be different from src
    // sign bit must be extended
    int64_t bitPtn = static_cast<int64_t>(imm);
    // Add(immediate) supports unsigned imm12
    if (imm >= 0) {
        const uint64_t imM12Mask = ~uint64_t(0xfff);
        // <= 4095
        if ((bitPtn & imM12Mask) == 0) {
            Sub(dst, src, static_cast<uint32_t>(imm & 0xfff));
            return;
        }
    }
    MovImm(tmp, imm);
    Sub(dst, src, tmp);
    return;
}

template <typename T, typename std::enable_if<std::is_unsigned<T>::value, std::nullptr_t>::type = nullptr>
void AddsImm(const XReg &dst, const XReg &src, T imm, const XReg &tmp)
{
    // this AddImm function allows dst == src
    // but tmp must be different from src
    uint64_t bitPtn = static_cast<uint64_t>(imm);
    // Add(immediate) supports unsigned imm12
    const uint64_t imM12Mask = ~uint64_t(0xfff);
    // <= 4095
    if ((bitPtn & imM12Mask) == 0) {
        Adds(dst, src, static_cast<uint32_t>(imm & 0xfff));
        return;
    }
    MovImm(tmp, imm);
    Adds(dst, src, tmp);
    return;
}

template <typename T, typename std::enable_if<std::is_signed<T>::value, std::nullptr_t>::type = nullptr>
void AddsImm(const XReg &dst, const XReg &src, T imm, const XReg &tmp)
{
    // this AddImm function allows dst == src
    // but tmp must be different from src
    // sign bit must be extended
    int64_t bitPtn = static_cast<int64_t>(imm);
    // Add(immediate) supports unsigned imm12
    if (imm >= 0) {
        const uint64_t imM12Mask = ~uint64_t(0xfff);
        // <= 4095
        if ((bitPtn & imM12Mask) == 0) {
            Adds(dst, src, static_cast<uint32_t>(imm & 0xfff));
            return;
        }
    }
    MovImm(tmp, imm);
    Adds(dst, src, tmp);
    return;
}

template <typename T, typename std::enable_if<std::is_unsigned<T>::value, std::nullptr_t>::type = nullptr>
void SubsImm(const XReg &dst, const XReg &src, T imm, const XReg &tmp)
{
    // this SubImm function allows dst == src
    // but tmp must be different from src
    uint64_t bitPtn = static_cast<uint64_t>(imm);
    // Add(immediate) supports unsigned imm12
    const uint64_t imM12Mask = ~uint64_t(0xfff);
    // <= 4095
    if ((bitPtn & imM12Mask) == 0) {
        Subs(dst, src, static_cast<uint32_t>(imm & 0xfff));
        return;
    }
    MovImm(tmp, imm);
    Subs(dst, src, tmp);
    return;
}

template <typename T, typename std::enable_if<std::is_signed<T>::value, std::nullptr_t>::type = nullptr>
void SubsImm(const XReg &dst, const XReg &src, T imm, const XReg &tmp)
{
    // this SubImm function allows dst == src
    // but tmp must be different from src
    // sign bit must be extended
    int64_t bitPtn = static_cast<int64_t>(imm);
    // Add(immediate) supports unsigned imm12
    if (imm >= 0) {
        const uint64_t imM12Mask = ~uint64_t(0xfff);
        // <= 4095
        if ((bitPtn & imM12Mask) == 0) {
            Subs(dst, src, static_cast<uint32_t>(imm & 0xfff));
            return;
        }
    }
    MovImm(tmp, imm);
    Subs(dst, src, tmp);
    return;
}

template <typename T> void MovImm(const XReg &dst, T imm)
{
    bool flag = false;
    uint64_t bitPtn = static_cast<uint64_t>(imm);
    uint64_t invBitPtn = ~bitPtn;
    int count = 0;
    int pos = 0;
    uint32_t ptn = 0;
    if (imm == 0) {
        Mov(dst, 0);
        return;
    }
    if (bitPtn == ~uint64_t(0)) {
        Movn(dst, 0);
        return;
    }
    if (IsBitMask(bitPtn)) {
        Mov(dst, imm);
        return;
    }
    // check if movn is applicable
    for (int i = 0; i < STEP_4; i++) {
        uint32_t tagBit = static_cast<uint32_t>((invBitPtn >> (BITFIELD_16 * i)) & 0xFFFF);
        if (tagBit) {
            ptn = tagBit;
            pos = i;
            count++;
        }
    }
    if (count == 1) {
        Movn(dst, ptn, BITFIELD_16 * pos);
        return;
    }
    for (int i = 0; i < STEP_4; i++) {
        uint64_t tagBit = (bitPtn >> (BITFIELD_16 * i)) & 0xFFFF;
        if (tagBit) {
            if (!flag) {
                Movz(dst, tagBit, BITFIELD_16 * i);
                flag = true;
            } else {
                Movk(dst, tagBit, BITFIELD_16 * i);
            }
        }
    }
    return;
}

template <typename T, typename std::enable_if<std::is_unsigned<T>::value, std::nullptr_t>::type = nullptr>
void MovImm(const WReg &dst, T imm)
{
    bool flag = false;
    uint32_t bitPtn = static_cast<uint32_t>(imm);
    if (imm == 0) {
        Mov(dst, 0);
        return;
    }
    if (uint64_t(0xFFFFFFFF) < imm && imm < uint64_t(0xFFFFFFFF80000000)) {
        throw Error(Err::ILLEGAL_IMM_RANGE);
    }
    if (IsBitMask(imm)) {
        Mov(dst, imm);
        return;
    }
    MovImmBySeg(dst, bitPtn, flag);
}

template <typename T, typename std::enable_if<std::is_signed<T>::value, std::nullptr_t>::type = nullptr>
void MovImm(const WReg &dst, T imm)
{
    bool flag = false;
    // T may be int32_t or int64_t
    uint32_t bitPtn = static_cast<uint32_t>(imm);
    if (imm == 0) {
        Mov(dst, 0);
        return;
    }
    if (imm < std::numeric_limits<int32_t>::min()) {
        throw Error(Err::ILLEGAL_IMM_RANGE);
    }
    if (IsBitMask(bitPtn)) {
        Mov(dst, bitPtn);
        return;
    }
    MovImmBySeg(dst, bitPtn, flag);
}

template <typename T>
void CmpImm(const XReg &a, T imm, const XReg &b)
{
    if ((imm >> BITFIELD_12) == 0) {
        Cmp(a, imm);
    } else {
        MovImm(b, imm);
        Cmp(a, b);
    }
    return;
}

#endif // CG_MNEMONIC_TPL_HPP
