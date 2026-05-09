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

#ifndef CG_REG_HPP
#define CG_REG_HPP

#include <cassert>
#include "cg_arm_const.hpp"

namespace ARMCG {

class Operand {
public:
    enum class Kind {
        NONE,
        RREG,
        VREG_SC,
        VREG_VEC,
        ZREG,
        PREG_Z,
        PREG_M,
        ZAREG,
        OPMASK
    };
    enum class Code {
        X0 = 0,
        X1,
        X2,
        X3,
        X4,
        X5,
        X6,
        X7,
        X8,
        X9,
        X10,
        X11,
        X12,
        X13,
        X14,
        X15,
        X16,
        X17,
        X18,
        X19,
        X20,
        X21,
        X22,
        X23,
        X24,
        X25,
        X26,
        X27,
        X28,
        X29,
        X30,
        X31 = 31,
    };

public:
    explicit Operand(Kind kind, uint32_t bit) noexcept : kind_(kind), bit_(bit) {}

    static const int vl = 4;  // vector length
    static const int svl = 4; // streaming vector length. svl >= vl

    uint32_t GetBit() const
    {
        return bit_;
    }
    bool IsRReg() const
    {
        return Is(Kind::RREG);
    }
    bool IsVRegSc() const
    {
        return Is(Kind::VREG_SC);
    }
    bool IsVRegVec() const
    {
        return Is(Kind::VREG_VEC);
    }
    bool IsZReg() const
    {
        return Is(Kind::ZREG);
    }
    bool IsPRegZ() const
    {
        return Is(Kind::PREG_Z);
    }
    bool IsPRegM() const
    {
        return Is(Kind::PREG_M);
    }

private:
    Kind kind_;
    uint32_t bit_;

    bool Is(Kind kind) const
    {
        return (kind_ == kind);
    }
};

class Reg : public Operand {
public:
    explicit Reg(uint32_t index, Kind kind, uint32_t bit) noexcept : Operand(kind, bit), index_(index) {}

    uint32_t GetIdx() const
    {
        return index_;
    }

private:
    uint32_t index_;
};

// general purpose register
class RReg : public Reg {
public:
    explicit RReg(uint32_t index, uint32_t bit) noexcept : Reg(index, Kind::RREG, bit) {}
};

class XReg : public RReg {
public:
    explicit XReg(uint32_t index) noexcept : RReg(index, BIT_64) {}
};

class WReg : public RReg {
public:
    explicit WReg(uint32_t index) noexcept : RReg(index, BIT_32) {}
};

// SIMD & FP scalar register
class VRegSc : public Reg {
public:
    explicit VRegSc(uint32_t index, uint32_t bit) noexcept : Reg(index, Kind::VREG_SC, bit) {}
};

class BReg : public VRegSc {
public:
    explicit BReg(uint32_t index) noexcept : VRegSc(index, BIT_8) {}
};
class HReg : public VRegSc {
public:
    explicit HReg(uint32_t index) noexcept : VRegSc(index, BIT_16) {}
};
class SReg : public VRegSc {
public:
    explicit SReg(uint32_t index) noexcept : VRegSc(index, BIT_32) {}
};
class DReg : public VRegSc {
public:
    explicit DReg(uint32_t index) noexcept : VRegSc(index, BIT_64) {}
};
class QReg : public VRegSc {
public:
    explicit QReg(uint32_t index) noexcept : VRegSc(index, BIT_128) {}
};

// VectorComputation SIMD Vector Register Base
class ZRegBase : public Reg {
public:
    explicit ZRegBase(uint32_t index, uint32_t bits = VAL_128 * vl) noexcept : Reg(index, Kind::ZREG, bits) {}
};

// VectorComputation SIMD Vector Register Element
class ZRegElem : public ZRegBase {
public:
    explicit ZRegElem(uint32_t index, uint32_t eidx, uint32_t bit) noexcept : ZRegBase(index, bit), elemIdx_(eidx) {}

    uint32_t GetElemIdx() const
    {
        return elemIdx_;
    }

private:
    uint32_t elemIdx_;
};

// base for VectorComputation SIMD Vector Register List
class ZRegList : public ZRegBase {
public:
    explicit ZRegList(const ZRegBase &s) noexcept : ZRegBase(s.GetIdx(), s.GetBit()), len_(1) {}
    explicit ZRegList(const ZRegBase &s, const ZRegBase &e) noexcept
        : ZRegBase(s.GetIdx(), s.GetBit()), len_(e.GetIdx() - s.GetIdx() + 1)
    {
    }

    uint32_t GetLen() const
    {
        return len_;
    }

private:
    uint32_t len_;
};

class ZRegB;
class ZRegH;
class ZRegS;
class ZRegD;
class ZRegQ;

class ZRegBElem : public ZRegElem {
public:
    explicit ZRegBElem(uint32_t index, uint32_t eidx) noexcept : ZRegElem(index, eidx, VAL_8) {}
};
class ZRegHElem : public ZRegElem {
public:
    explicit ZRegHElem(uint32_t index, uint32_t eidx) noexcept : ZRegElem(index, eidx, VAL_16) {}
};
class ZRegSElem : public ZRegElem {
public:
    explicit ZRegSElem(uint32_t index, uint32_t eidx) noexcept : ZRegElem(index, eidx, VAL_32) {}
};
class ZRegDElem : public ZRegElem {
public:
    explicit ZRegDElem(uint32_t index, uint32_t eidx) noexcept : ZRegElem(index, eidx, VAL_64) {}
};
class ZRegQElem : public ZRegElem {
public:
    explicit ZRegQElem(uint32_t index, uint32_t eidx) noexcept : ZRegElem(index, eidx, VAL_128) {}
};

class ZRegBList : public ZRegList {
public:
    explicit ZRegBList(const ZRegB &s);
    explicit ZRegBList(const ZRegBase &s, const ZRegBase &e) noexcept : ZRegList(s, e) {}
    ZRegBElem operator[](uint32_t i) const
    {
        return ZRegBElem(GetIdx(), i);
    }
};
class ZRegHList : public ZRegList {
public:
    explicit ZRegHList(const ZRegH &s) noexcept;
    explicit ZRegHList(const ZRegBase &s, const ZRegBase &e) noexcept : ZRegList(s, e) {}
    ZRegHElem operator[](uint32_t i) const
    {
        return ZRegHElem(GetIdx(), i);
    }
};
class ZRegSList : public ZRegList {
public:
    explicit ZRegSList(const ZRegS &s) noexcept;
    explicit ZRegSList(const ZRegBase &s, const ZRegBase &e) noexcept : ZRegList(s, e) {}
    ZRegSElem operator[](uint32_t i) const
    {
        return ZRegSElem(GetIdx(), i);
    }
};
class ZRegDList : public ZRegList {
public:
    explicit ZRegDList(const ZRegD &s) noexcept;
    explicit ZRegDList(const ZRegBase &s, const ZRegBase &e) noexcept : ZRegList(s, e) {}
    ZRegDElem operator[](uint32_t i) const
    {
        return ZRegDElem(GetIdx(), i);
    }
};
class ZRegQList : public ZRegList {
public:
    explicit ZRegQList(const ZRegQ &s) noexcept;
    explicit ZRegQList(const ZRegBase &s, const ZRegBase &e) noexcept : ZRegList(s, e) {}
    ZRegQElem operator[](uint32_t i) const
    {
        return ZRegQElem(GetIdx(), i);
    }
};

class ZRegB : public ZRegBase {
public:
    explicit ZRegB(uint32_t index) noexcept : ZRegBase(index, VAL_8) {}
    ZRegBElem operator[](uint32_t i) const
    {
        return ZRegBElem(GetIdx(), i);
    }
    ZRegBList operator-(const ZRegB &other) const
    {
        return ZRegBList(*this, other);
    }
};
class ZRegH : public ZRegBase {
public:
    explicit ZRegH(uint32_t index) noexcept : ZRegBase(index, VAL_16) {}
    ZRegHElem operator[](uint32_t i) const
    {
        return ZRegHElem(GetIdx(), i);
    }
    ZRegHList operator-(const ZRegH &other) const
    {
        return ZRegHList(*this, other);
    }
};
class ZRegS : public ZRegBase {
public:
    explicit ZRegS(uint32_t index) noexcept : ZRegBase(index, VAL_32) {}
    ZRegSElem operator[](uint32_t i) const
    {
        return ZRegSElem(GetIdx(), i);
    }
    ZRegSList operator-(const ZRegS &other) const
    {
        return ZRegSList(*this, other);
    }
};
class ZRegD : public ZRegBase {
public:
    explicit ZRegD(uint32_t index) noexcept : ZRegBase(index, VAL_64) {}
    ZRegDElem operator[](uint32_t i) const
    {
        return ZRegDElem(GetIdx(), i);
    }
    ZRegDList operator-(const ZRegD &other) const
    {
        return ZRegDList(*this, other);
    }
};
class ZRegQ : public ZRegBase {
public:
    explicit ZRegQ(uint32_t index) noexcept : ZRegBase(index, VAL_128) {}
    ZRegQElem operator[](uint32_t i) const
    {
        return ZRegQElem(GetIdx(), i);
    }
    ZRegQList operator-(const ZRegQ &other) const
    {
        return ZRegQList(*this, other);
    }
};

// SIMD Vector Register for VectorComputation
class ZReg : public ZRegBase {
public:
    explicit ZReg(uint32_t index) noexcept : ZRegBase(index), b(index), h(index), s(index), d(index), q(index) {}

    ZRegB b;
    ZRegH h;
    ZRegS s;
    ZRegD d;
    ZRegQ q;
};

class PRegBase : public Reg {
public:
    explicit PRegBase(uint32_t index, bool m = false, uint32_t bits = VAL_16 * vl) noexcept
        : Reg(index, ((m == 0) ? Kind::PREG_Z : Kind::PREG_M), bits)
    {
    }

    bool IsM() const
    {
        return IsPRegM();
    }
    bool IsZ() const
    {
        return IsPRegZ();
    }
};

class PRegB : public PRegBase {
public:
    explicit PRegB(uint32_t index) noexcept : PRegBase(index, false, VAL_8) {}
};
class PRegH : public PRegBase {
public:
    explicit PRegH(uint32_t index) noexcept : PRegBase(index, false, VAL_16) {}
};
class PRegS : public PRegBase {
public:
    explicit PRegS(uint32_t index) noexcept : PRegBase(index, false, VAL_32) {}
};
class PRegD : public PRegBase {
public:
    explicit PRegD(uint32_t index) noexcept : PRegBase(index, false, VAL_64) {}
};

enum class PredType {
    T_Z, // Zeroing predication
    T_M  // Merging predication
};

class PReg : public PRegBase {
public:
    explicit PReg(uint32_t index, bool m = false) noexcept : PRegBase(index, m), b(index), h(index), s(index), d(index)
    {
    }
    PRegBase operator/(PredType t) const
    {
        return (t == PredType::T_Z) ? PRegBase(GetIdx(), false) : PRegBase(GetIdx(), true);
    }

    PRegB b;
    PRegH h;
    PRegS s;
    PRegD d;
};

// MatrixComputation Storage
class ZARegBase : public Reg {
protected:
    explicit ZARegBase(uint32_t index, uint32_t bits = VAL_128 * svl) noexcept : Reg(index, Kind::ZAREG, bits) {}
};

class ZARegB : public ZARegBase {
public:
    explicit ZARegB(uint32_t index) noexcept : ZARegBase(index) {}
};
class ZARegS : public ZARegBase {
public:
    explicit ZARegS(uint32_t index) noexcept : ZARegBase(index) {}
};
class ZAReg : public ZARegBase {
public:
    explicit ZAReg(uint32_t index) noexcept : ZARegBase(index), b(index), s(index) {}
    ZARegB b;
    ZARegS s;
};

class ZARegHB : public ZARegBase {
public:
    explicit ZARegHB(uint32_t index) noexcept : ZARegBase(index) {}
};

class ZARegHS : public ZARegBase {
public:
    explicit ZARegHS(uint32_t index) noexcept : ZARegBase(index) {}
};

class ZARegH : public ZARegBase {
public:
    explicit ZARegH(uint32_t index) noexcept : ZARegBase(index), s(index), b(index) {}
    ZARegHS s;
    ZARegHB b;
};

class ZARegVB : public ZARegBase {
public:
    explicit ZARegVB(uint32_t index) noexcept : ZARegBase(index) {}
};

class ZARegVS : public ZARegBase {
public:
    explicit ZARegVS(uint32_t index) noexcept : ZARegBase(index) {}
};
class ZARegV : public ZARegBase {
public:
    explicit ZARegV(uint32_t index) noexcept : ZARegBase(index), s(index), b(index) {}
    ZARegVS s;
    ZARegVB b;
};

} // namespace ARMCG

#endif // CG_REG_HPP
