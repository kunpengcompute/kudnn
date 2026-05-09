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

#ifndef CG_ADR_HPP
#define CG_ADR_HPP

#include "cg_reg_v.hpp"

namespace ARMCG {

enum class ShMod {
    LSL = 0, // logical shift left
    LSR = 1, // logical shift right
    ASR = 2, // arithmetic shift right
    ROR = 3, // rotate right
    MSL = 4, // masking shift left, aka "shifting ones", which shifts to the left with 1 as padding
    NONE = 5
};

enum class ExtMod {
    UXTB = 0, // unsigned extend to byte
    UXTH = 1, // unsigned extend to half-word
    UXTW = 2, // unsigned extend to word
    UXTX = 3, // unsigned extend to double-word
    SXTB = 4, // signed extend to byte
    SXTH = 5, // signed extend to half-word
    SXTW = 6, // signed extend to word
    SXTX = 7, // signed extend to double-word
    UXT = 8,
    SXT = 9,
    MUL = 10,    // multiply
    MUL_VL = 11, // multipy by vector length
    EXT_LSL = 12 // logical shift left
};

// Pre-indexed: first update base register with offset and then use it to do addressing
// Post-indexed: first use base register to do addressing and then update it with offset
enum class AdrKind {
    BASE_ONLY = 1,         // no offset: [Rn]
    BASE_IMM = 1 << 1,     // immidiate offset: [Rn, #<offset_12>]
    BASE_REG = 1 << 2,     // scaled register offset: [Rn, Rm, <shift>#<offset_12>]
    BASE_EXT = 1 << 3,     // for use with load SIMD and FP Register: [Xn|SP, (Wm|Xm), extend {amount}]
    PRE = 1 << 4,          // immediate pre-indexed: [Rn, #<offset_12>]!
    POST_IMM = 1 << 5,     // immediate post-indexed: [Rn], #<offset_12>
    POST_REG = 1 << 6,     // scaled register post-indexed: [Rn], Rm, <shift>#<offset_12>
    SC_SC = 1 << 7,        // scalar plus immediate index: [<Xn|SP>{, #<imm>, MUL VL}]
    SC_IMM = 1 << 8,       // scalar plus scalar index: [<Xn|SP>, <Xm>, LSL #<elt byte>]
    SC_64VEC = 1 << 9,     // scalar base plus 64-bit vector index
    SC_32VEC64E = 1 << 10, // scalar base plus 32-bit vecotr index (64-bit element)
    SC_32VEC32E = 1 << 11, // scalar base plus 32-bit vecotr index (32-bit element)
    VEC_IMM64E = 1 << 12,  // vector base plus immediate offset (64-bit element)
    VEC_IMM32E = 1 << 13,  // vector base plus immediate offset (32-bit element)
    VEC_SC32E = 1 << 14,   // vector base pulus 64-bit scalar offset (32-bit element)
    VEC_SC64E = 1 << 15,   // vector base pulus 64-bit scalar offset (32-bit element)
    VEC_PACK = 1 << 16,    // vector (packed) // for VectorComputation address generator
    VEC_UNPACK = 1 << 17,  // vector (unpacked)
};

const int REG_LEN_64 = 64;

class Adr {
public:
    explicit Adr(AdrKind kind) : kind_(kind) {}

    AdrKind GetKind() const
    {
        return kind_;
    }

protected:
    ExtMod Trans(ExtMod org)
    {
        return ((org == ExtMod::UXT) ? ExtMod::UXTW : (org == ExtMod::SXT) ? ExtMod::SXTW : org);
    }
    ExtMod Trans(const RReg &rm, ExtMod org)
    {
        if (org == ExtMod::SXT) {
            return (rm.GetBit() == REG_LEN_64) ? ExtMod::SXTX : ExtMod::SXTW;
        } else if (org == ExtMod::UXT) {
            return (rm.GetBit() == REG_LEN_64) ? ExtMod::UXTX : ExtMod::UXTW;
        }
        return org;
    }

private:
    AdrKind kind_;
};

// pre-indexed
class AdrPreImm : public Adr {
public:
    explicit AdrPreImm(const XReg &xn, int32_t imm) : Adr(AdrKind::PRE), xn_(xn), imm_(imm) {}

    const XReg &GetXn() const
    {
        return xn_;
    }
    int32_t GetImm() const
    {
        return imm_;
    }

private:
    XReg xn_;
    int32_t imm_;
};

// pos-indexed (immediate offset)
class AdrPostImm : public Adr {
public:
    explicit AdrPostImm(const XReg &xn, int32_t imm) : Adr(AdrKind::POST_IMM), xn_(xn), imm_(imm) {}

    const XReg &GetXn() const
    {
        return xn_;
    }
    int32_t GetImm() const
    {
        return imm_;
    }

private:
    XReg xn_;
    int32_t imm_;
};

// pos-indexed (register offset)
class AdrPostReg : public Adr {
public:
    explicit AdrPostReg(const XReg &xn, const XReg &xm) : Adr(AdrKind::POST_REG), xn_(xn), xm_(xm) {}

    const XReg &GetXn() const
    {
        return xn_;
    }
    const XReg &GetXm() const
    {
        return xm_;
    }

private:
    XReg xn_;
    XReg xm_;
};

// base only
class AdrNoOfs : public Adr {
public:
    explicit AdrNoOfs(const XReg &xn) : Adr(AdrKind::BASE_ONLY), xn_(xn) {}
    const XReg &GetXn() const
    {
        return xn_;
    }

private:
    XReg xn_;
};

// base plus offset (signed immediate)
class AdrImm : public Adr {
public:
    explicit AdrImm(const XReg &xn, int32_t imm) : Adr(AdrKind::BASE_IMM), xn_(xn), imm_(imm) {}
    explicit AdrImm(const AdrNoOfs &a) : Adr(AdrKind::BASE_IMM), xn_(a.GetXn()), imm_(0) {}

    const XReg &GetXn() const
    {
        return xn_;
    }
    int32_t GetImm() const
    {
        return imm_;
    }

private:
    XReg xn_;
    int32_t imm_;
};

// base plus offset (unsigned immediate)
class AdrUimm : public Adr {
public:
    explicit AdrUimm(const XReg &xn, uint32_t uimm) : Adr(AdrKind::BASE_IMM), xn_(xn), uimm_(uimm) {}
    explicit AdrUimm(const AdrNoOfs &a) : Adr(AdrKind::BASE_IMM), xn_(a.GetXn()), uimm_(0) {}
    explicit AdrUimm(const AdrImm &a) : Adr(AdrKind::BASE_IMM), xn_(a.GetXn()), uimm_(a.GetImm()) {}

    const XReg &GetXn() const
    {
        return xn_;
    }
    uint32_t GetImm() const
    {
        return uimm_;
    }

private:
    XReg xn_;
    uint32_t uimm_;
};

// base plus offset (register)
class AdrReg : public Adr {
public:
    explicit AdrReg(const XReg &xn, const XReg &xm, ShMod mod, uint32_t sh)
        : Adr(AdrKind::BASE_REG), xn_(xn), xm_(xm), mod_(mod), sh_(sh), init_(true)
    {
    }
    explicit AdrReg(const XReg &xn, const XReg &xm, ShMod mod = ShMod::LSL)
        : Adr(AdrKind::BASE_REG), xn_(xn), xm_(xm), mod_(mod), sh_(0), init_(false)
    {
    }

    const XReg &GetXn() const
    {
        return xn_;
    }
    const XReg &GetXm() const
    {
        return xm_;
    }
    ShMod GetMod() const
    {
        return mod_;
    }
    uint32_t GetSh() const
    {
        return sh_;
    }
    bool GetInitSh() const
    {
        return init_;
    }

private:
    XReg xn_;
    XReg xm_;
    ShMod mod_;
    uint32_t sh_;
    bool init_;
};

// base plus offset (extended register)
class AdrExt : public Adr {
public:
    explicit AdrExt(const XReg &xn, const RReg &rm, ExtMod mod, uint32_t sh)
        : Adr(AdrKind::BASE_EXT), xn_(xn), rm_(rm), mod_(Trans(rm, mod)), sh_(sh), init_(true)
    {
    }
    explicit AdrExt(const XReg &xn, const RReg &rm, ExtMod mod)
        : Adr(AdrKind::BASE_EXT), xn_(xn), rm_(rm), mod_(Trans(rm, mod)), sh_(0), init_(false)
    {
    }

    const XReg &GetXn() const
    {
        return xn_;
    }
    const RReg &GetRm() const
    {
        return rm_;
    }
    ExtMod GetMod() const
    {
        return mod_;
    }
    uint32_t GetSh() const
    {
        return sh_;
    }
    bool GetInitSh() const
    {
        return init_;
    }

private:
    XReg xn_;
    RReg rm_;
    ExtMod mod_;
    uint32_t sh_;
    bool init_;
};

// scalar plus scalar
class AdrScSc : public Adr {
public:
    explicit AdrScSc(const XReg &xn, const XReg &xm, ShMod mod, uint32_t sh)
        : Adr(AdrKind::SC_SC), xn_(xn), xm_(xm), mod_(mod), sh_(sh), init_(true)
    {
    }
    explicit AdrScSc(const XReg &xn, const XReg &xm)
        : Adr(AdrKind::SC_SC), xn_(xn), xm_(xm), mod_(ShMod::LSL), sh_(0), init_(false)
    {
    }

    const XReg &GetXn() const
    {
        return xn_;
    }
    const XReg &GetXm() const
    {
        return xm_;
    }
    uint32_t GetSh() const
    {
        return sh_;
    }
    ShMod GetMod() const
    {
        return mod_;
    }
    bool GetInitMod() const
    {
        return init_;
    }

private:
    XReg xn_;
    XReg xm_;
    ShMod mod_;
    uint32_t sh_;
    bool init_;
};

// scalar plus immediate
class AdrScImm : public Adr {
public:
    explicit AdrScImm(const XReg &xn, int32_t simm = 0, ExtMod mod = ExtMod::MUL_VL)
        : Adr(AdrKind::SC_IMM), xn_(xn), simm_(simm), mod_(Trans(mod))
    {
    }
    explicit AdrScImm(const AdrImm &a) : Adr(AdrKind::SC_IMM), xn_(a.GetXn()), simm_(a.GetImm()), mod_(ExtMod::MUL_VL)
    {
    }

    const XReg &GetXn() const
    {
        return xn_;
    }
    int32_t GetSimm() const
    {
        return simm_;
    }
    ExtMod GetMod() const
    {
        return mod_;
    }

private:
    XReg xn_;
    int32_t simm_;
    ExtMod mod_;
};

inline AdrNoOfs Ptr(const XReg &xn)
{
    return AdrNoOfs(xn);
}
inline AdrImm Ptr(const XReg &xn, int32_t imm)
{
    return AdrImm(xn, imm);
}
inline AdrUimm Ptr(const XReg &xn, uint32_t uimm)
{
    return AdrUimm(xn, uimm);
}
inline AdrReg Ptr(const XReg &xn, const XReg &xm)
{
    return AdrReg(xn, xm);
}
inline AdrReg Ptr(const XReg &xn, const XReg &xm, ShMod mod, uint32_t sh)
{
    return AdrReg(xn, xm, mod, sh);
}
inline AdrReg Ptr(const XReg &xn, const XReg &xm, ShMod mod)
{
    return AdrReg(xn, xm, mod);
}
inline AdrExt Ptr(const XReg &xn, const RReg &rm, ExtMod mod, uint32_t sh)
{
    return AdrExt(xn, rm, mod, sh);
}
inline AdrExt Ptr(const XReg &xn, const RReg &rm, ExtMod mod)
{
    return AdrExt(xn, rm, mod);
}
inline AdrPreImm PrePtr(const XReg &xn, int32_t imm)
{
    return AdrPreImm(xn, imm);
}
inline AdrPostImm PostPtr(const XReg &xn, int32_t imm)
{
    return AdrPostImm(xn, imm);
}
inline AdrPostReg PostPtr(const XReg &xn, XReg xm)
{
    return AdrPostReg(xn, xm);
}
inline AdrScImm Ptr(const XReg &xn, int32_t simm, ExtMod mod)
{
    return AdrScImm(xn, simm, mod);
}

} // namespace ARMCG

#endif // CG_ADR_HPP
