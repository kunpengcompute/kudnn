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

#ifndef CG_ADR_VectorComputation_HPP
#define CG_ADR_VectorComputation_HPP

#include "cg_adr.hpp"

namespace ARMCG {

// scalar plus vector (unscaled 64-bit offset)
class AdrSc64U : public Adr {
public:
    explicit AdrSc64U(const XReg &xn, const ZRegD &zm) : Adr(AdrKind::SC_64VEC), xn_(xn), zm_(zm) {}

    const XReg &GetXn() const
    {
        return xn_;
    }
    const ZRegD &GetZm() const
    {
        return zm_;
    }

private:
    XReg xn_;
    ZRegD zm_;
};

// scalar plus vector (scalar 64-bit offset)
class AdrSc64S : public Adr {
public:
    explicit AdrSc64S(const XReg &xn, const ZRegD &zm, ShMod mod, uint32_t sh)
        : Adr(AdrKind::SC_64VEC), xn_(xn), zm_(zm), mod_(mod), sh_(sh)
    {
    }
    explicit AdrSc64S(const AdrSc64U &a)
        : Adr(AdrKind::SC_64VEC), xn_(a.GetXn()), zm_(a.GetZm()), mod_(ShMod::LSL), sh_(0)
    {
    }

    const XReg &GetXn() const
    {
        return xn_;
    }
    const ZRegD &GetZm() const
    {
        return zm_;
    }
    ShMod GetMod() const
    {
        return mod_;
    }
    uint32_t GetSh() const
    {
        return sh_;
    }

private:
    XReg xn_;
    ZRegD zm_;
    ShMod mod_;
    uint32_t sh_;
};

// vector plus scalar (64-bit unscaled sclar register offset, 64-bit element)
class AdrVecSc64 : public Adr {
public:
    explicit AdrVecSc64(const ZRegD &zn, const XReg &xm) : Adr(AdrKind::VEC_SC64E), zn_(zn), xm_(xm) {}

    const ZRegD &GetZn() const
    {
        return zn_;
    }
    const XReg &GetXm() const
    {
        return xm_;
    }

private:
    ZRegD zn_;
    XReg xm_;
};

// scalar plus vector (unscaled 32-bit offset)
class AdrSc32U : public Adr {
public:
    explicit AdrSc32U(const XReg &xn, const ZRegS &zm, ExtMod mod)
        : Adr(AdrKind::SC_32VEC32E), xn_(xn), zm_(zm), mod_(Trans(mod))
    {
    }

    const XReg &GetXn() const
    {
        return xn_;
    }
    const ZRegS &GetZm() const
    {
        return zm_;
    }
    ExtMod GetMod() const
    {
        return mod_;
    }

private:
    XReg xn_;
    ZRegS zm_;
    ExtMod mod_;
};

// scalar plus vector (scaled 32-bit offset)
class AdrSc32S : public Adr {
public:
    explicit AdrSc32S(const XReg &xn, const ZRegS &zm, ExtMod mod, uint32_t sh)
        : Adr(AdrKind::SC_32VEC32E), xn_(xn), zm_(zm), mod_(Trans(mod)), sh_(sh)
    {
    }
    explicit AdrSc32S(const AdrSc32U &a)
        : Adr(AdrKind::SC_32VEC32E), xn_(a.GetXn()), zm_(a.GetZm()), mod_(a.GetMod()), sh_(0)
    {
    }

    const XReg &GetXn() const
    {
        return xn_;
    }
    const ZRegS &GetZm() const
    {
        return zm_;
    }
    ExtMod GetMod() const
    {
        return mod_;
    }
    uint32_t GetSh() const
    {
        return sh_;
    }

private:
    XReg xn_;
    ZRegS zm_;
    ExtMod mod_;
    uint32_t sh_;
};

// vector plus scalar (64-bit unscaled sclar register offset, 32-bit element)
class AdrVecSc32 : public Adr {
public:
    explicit AdrVecSc32(const ZRegS &zn, const XReg &xm) : Adr(AdrKind::VEC_SC32E), zn_(zn), xm_(xm) {}

    const ZRegS &GetZn() const
    {
        return zn_;
    }
    const XReg &GetXm() const
    {
        return xm_;
    }

private:
    ZRegS zn_;
    XReg xm_;
};

// scalar plus vector (unpacked unscaled 32-bit offset)
class AdrSc32UU : public Adr {
public:
    explicit AdrSc32UU(const XReg &xn, const ZRegD &zm, ExtMod mod)
        : Adr(AdrKind::SC_32VEC64E), xn_(xn), zm_(zm), mod_(Trans(mod))
    {
    }

    const XReg &GetXn() const
    {
        return xn_;
    }
    const ZRegD &GetZm() const
    {
        return zm_;
    }
    ExtMod GetMod() const
    {
        return mod_;
    }

private:
    XReg xn_;
    ZRegD zm_;
    ExtMod mod_;
};

// scalar plus vector (unpacked scaled 32-bit offset)
class AdrSc32US : public Adr {
public:
    explicit AdrSc32US(const XReg &xn, const ZRegD &zm, ExtMod mod, uint32_t sh)
        : Adr(AdrKind::SC_32VEC64E), xn_(xn), zm_(zm), mod_(Trans(mod)), sh_(sh)
    {
    }
    explicit AdrSc32US(const AdrSc32UU &a)
        : Adr(AdrKind::SC_32VEC64E), xn_(a.GetXn()), zm_(a.GetZm()), mod_(a.GetMod()), sh_(0)
    {
    }

    const XReg &GetXn() const
    {
        return xn_;
    }
    const ZRegD &GetZm() const
    {
        return zm_;
    }
    ExtMod GetMod() const
    {
        return mod_;
    }
    uint32_t GetSh() const
    {
        return sh_;
    }

private:
    XReg xn_;
    ZRegD zm_;
    ExtMod mod_;
    uint32_t sh_;
};

// vector plus immediate 64-bit element
class AdrVecImm64 : public Adr {
public:
    explicit AdrVecImm64(const ZRegD &zn, uint32_t imm = 0) : Adr(AdrKind::VEC_IMM64E), zn_(zn), imm_(imm) {}

    const ZRegD &GetZn() const
    {
        return zn_;
    }
    uint32_t GetImm() const
    {
        return imm_;
    }

private:
    ZRegD zn_;
    uint32_t imm_;
};

// vector plus immediate 32-bit element
class AdrVecImm32 : public Adr {
public:
    explicit AdrVecImm32(const ZRegS &zn, uint32_t imm = 0) : Adr(AdrKind::VEC_IMM32E), zn_(zn), imm_(imm) {}

    const ZRegS &GetZn() const
    {
        return zn_;
    }
    uint32_t GetImm() const
    {
        return imm_;
    }

private:
    ZRegS zn_;
    uint32_t imm_;
};

// vector Address (packed offset)
class AdrVec : public Adr {
public:
    explicit AdrVec(const ZRegBase &zn, const ZRegBase &zm, ShMod mod = ShMod::NONE, uint32_t sh = 0)
        : Adr(AdrKind::VEC_PACK), zn_(zn), zm_(zm), mod_(mod), sh_(sh)
    {
    }

    const ZRegBase &GetZn() const
    {
        return zn_;
    }
    const ZRegBase &GetZm() const
    {
        return zm_;
    }
    ShMod GetMod() const
    {
        return mod_;
    }
    uint32_t GetSh() const
    {
        return sh_;
    }

private:
    ZRegBase zn_;
    ZRegBase zm_;
    ShMod mod_;
    uint32_t sh_;
};

// vector Address (unpacked offset)
class AdrVecU : public Adr {
public:
    explicit AdrVecU(const ZRegBase &zn, const ZRegBase &zm, ExtMod mod, uint32_t sh = 0)
        : Adr(AdrKind::VEC_UNPACK), zn_(zn), zm_(zm), mod_(Trans(mod)), sh_(sh)
    {
    }

    const ZRegBase &GetZn() const
    {
        return zn_;
    }
    const ZRegBase &GetZm() const
    {
        return zm_;
    }
    ExtMod GetMod() const
    {
        return mod_;
    }
    uint32_t GetSh() const
    {
        return sh_;
    }

private:
    ZRegBase zn_;
    ZRegBase zm_;
    ExtMod mod_;
    uint32_t sh_;
};

inline AdrSc64U Ptr(const XReg &xn, const ZRegD &zm)
{
    return AdrSc64U(xn, zm);
}
inline AdrSc64S Ptr(const XReg &xn, const ZRegD &zm, ShMod mod, uint32_t sh)
{
    return AdrSc64S(xn, zm, mod, sh);
}
inline AdrVecSc64 Ptr(const ZRegD &zn, const XReg &xm)
{
    return AdrVecSc64(zn, xm);
}
inline AdrSc32U Ptr(const XReg &xn, const ZRegS &zm, ExtMod mod)
{
    return AdrSc32U(xn, zm, mod);
}
inline AdrSc32S Ptr(const XReg &xn, const ZRegS &zm, ExtMod mod, uint32_t sh)
{
    return AdrSc32S(xn, zm, mod, sh);
}
inline AdrVecSc32 Ptr(const ZRegS &zn, const XReg &xm)
{
    return AdrVecSc32(zn, xm);
}
inline AdrSc32UU Ptr(const XReg &xn, const ZRegD &zm, ExtMod mod)
{
    return AdrSc32UU(xn, zm, mod);
}
inline AdrSc32US Ptr(const XReg &xn, const ZRegD &zm, ExtMod mod, uint32_t sh)
{
    return AdrSc32US(xn, zm, mod, sh);
}
inline AdrVecImm64 Ptr(const ZRegD &zn, uint32_t imm)
{
    return AdrVecImm64(zn, imm);
}
inline AdrVecImm32 Ptr(const ZRegS &zn, uint32_t imm)
{
    return AdrVecImm32(zn, imm);
}
inline AdrVec Ptr(const ZRegS &zn, const ZRegS &zm, ShMod mod, uint32_t sh)
{
    return AdrVec(zn, zm, mod, sh);
}
inline AdrVec Ptr(const ZRegD &zn, const ZRegD &zm, ShMod mod, uint32_t sh)
{
    return AdrVec(zn, zm, mod, sh);
}
inline AdrVecU Ptr(const ZRegD &zn, const ZRegD &zm, ExtMod mod, uint32_t sh)
{
    return AdrVecU(zn, zm, mod, sh);
}

} // namespace ARMCG

#endif // CG_ADR_VectorComputation_HPP
