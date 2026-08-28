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

#ifndef CG_ADR_VC_HPP
#define CG_ADR_VC_HPP

#include "cg_adr.hpp"

namespace ARMCG {

// many addressing modes take the form [Xn, Zm] so we define an intermediate class to avoid code repetition
class AdrScalarPlusVectorD : public Adr {
public:
    explicit AdrScalarPlusVectorD(AdrKind kind, const XReg &xn, const ZRegD &zm) : Adr(kind), xn_(xn), zm_(zm) {}

    const XReg &GetXn() const
    {
        return xn_;
    }
    const ZRegD &GetZm() const
    {
        return zm_;
    }

protected:
    XReg xn_;
    ZRegD zm_;
};

class AdrScalarPlusVectorS : public Adr {
public:
    explicit AdrScalarPlusVectorS(AdrKind kind, const XReg &xn, const ZRegS &zm) : Adr(kind), xn_(xn), zm_(zm) {}

    const XReg &GetXn() const
    {
        return xn_;
    }
    const ZRegS &GetZm() const
    {
        return zm_;
    }

protected:
    XReg xn_;
    ZRegS zm_;
};

class AdrVectorPlusVector : public Adr {
public:
    explicit AdrVectorPlusVector(AdrKind kind, const ZRegBase &zn, const ZRegBase &zm)
        : Adr(kind), zn_(zn), zm_(zm) {}

    const ZRegBase &GetZn() const
    {
        return zn_;
    }
    const ZRegBase &GetZm() const
    {
        return zm_;
    }

protected:
    ZRegBase zn_;
    ZRegBase zm_;
};

// scalar plus vector (unscaled 64-bit offset)
class AdrSc64U : public AdrScalarPlusVectorD {
public:
    explicit AdrSc64U(const XReg &xn, const ZRegD &zm) : AdrScalarPlusVectorD(AdrKind::SC_64VEC, xn, zm) {}
};

// scalar plus vector (scalar 64-bit offset)
class AdrSc64S : public AdrScalarPlusVectorD {
public:
    explicit AdrSc64S(const XReg &xn, const ZRegD &zm, ShMod mod, uint32_t sh)
        : AdrScalarPlusVectorD(AdrKind::SC_64VEC, xn, zm), modSc64_(mod), shSc64_(sh) {}
    explicit AdrSc64S(const AdrSc64U &a)
        : AdrScalarPlusVectorD(AdrKind::SC_64VEC, a.GetXn(), a.GetZm()), modSc64_(ShMod::LSL), shSc64_(0) {}

    ShMod GetMod() const
    {
        return modSc64_;
    }
    uint32_t GetSh() const
    {
        return shSc64_;
    }

private:
    ShMod modSc64_;
    uint32_t shSc64_;
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
class AdrSc32U : public AdrScalarPlusVectorS {
public:
    explicit AdrSc32U(const XReg &xn, const ZRegS &zm, ExtMod mod)
        : AdrScalarPlusVectorS(AdrKind::SC_32VEC32E, xn, zm), modSc32U_(Trans(mod)) {}

    ExtMod GetMod() const
    {
        return modSc32U_;
    }

private:
    ExtMod modSc32U_;
};

// scalar plus vector (scaled 32-bit offset)
class AdrSc32S : public AdrScalarPlusVectorS {
public:
    explicit AdrSc32S(const XReg &xn, const ZRegS &zm, ExtMod mod, uint32_t sh)
        : AdrScalarPlusVectorS(AdrKind::SC_32VEC32E, xn, zm), modSc32S_(Trans(mod)), shSc32S_(sh) {}
    explicit AdrSc32S(const AdrSc32U &a)
        : AdrScalarPlusVectorS(AdrKind::SC_32VEC32E, a.GetXn(), a.GetZm()), modSc32S_(a.GetMod()), shSc32S_(0) {}

    ExtMod GetMod() const
    {
        return modSc32S_;
    }
    uint32_t GetSh() const
    {
        return shSc32S_;
    }

private:
    ExtMod modSc32S_;
    uint32_t shSc32S_;
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
class AdrSc32UU : public AdrScalarPlusVectorD {
public:
    explicit AdrSc32UU(const XReg &xn, const ZRegD &zm, ExtMod mod)
        : AdrScalarPlusVectorD(AdrKind::SC_32VEC64E, xn, zm), modSc32UU_(Trans(mod)) {}

    ExtMod GetMod() const
    {
        return modSc32UU_;
    }

private:
    ExtMod modSc32UU_;
};

// scalar plus vector (unpacked scaled 32-bit offset)
class AdrSc32US : public AdrScalarPlusVectorD {
public:
    explicit AdrSc32US(const XReg &xn, const ZRegD &zm, ExtMod mod, uint32_t sh)
        : AdrScalarPlusVectorD(AdrKind::SC_32VEC64E, xn, zm), modSc32US_(Trans(mod)), shSc32US_(sh) {}
    explicit AdrSc32US(const AdrSc32UU &a)
        : AdrScalarPlusVectorD(AdrKind::SC_32VEC64E, a.GetXn(), a.GetZm()), modSc32US_(a.GetMod()), shSc32US_(0) {}

    ExtMod GetMod() const
    {
        return modSc32US_;
    }
    uint32_t GetSh() const
    {
        return shSc32US_;
    }

private:
    ExtMod modSc32US_;
    uint32_t shSc32US_;
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
class AdrVec : public AdrVectorPlusVector {
public:
    explicit AdrVec(const ZRegBase &zn, const ZRegBase &zm, ShMod mod = ShMod::NONE, uint32_t sh = 0)
        : AdrVectorPlusVector(AdrKind::VEC_PACK, zn, zm), modVec_(mod), shVec_(sh) {}

    ShMod GetMod() const
    {
        return modVec_;
    }
    uint32_t GetSh() const
    {
        return shVec_;
    }

private:
    ShMod modVec_;
    uint32_t shVec_;
};

// vector Address (unpacked offset)
class AdrVecU : public AdrVectorPlusVector {
public:
    explicit AdrVecU(const ZRegBase &zn, const ZRegBase &zm, ExtMod mod, uint32_t sh = 0)
        : AdrVectorPlusVector(AdrKind::VEC_UNPACK, zn, zm), modVecU_(Trans(mod)), shVecU_(sh) {}

    ExtMod GetMod() const
    {
        return modVecU_;
    }
    uint32_t GetSh() const
    {
        return shVecU_;
    }

private:
    ExtMod modVecU_;
    uint32_t shVecU_;
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

} // ARMCG

#endif  // CG_ADR_VC_HPP