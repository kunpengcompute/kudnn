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

#ifndef CG_REG_V_HPP
#define CG_REG_V_HPP

#include "cg_reg.hpp"

namespace ARMCG {

// base for SIMD vector register
class VRegVec : public Reg {
public:
    explicit VRegVec(uint32_t index, uint32_t bit, uint32_t lane) : Reg(index, Kind::VREG_VEC, bit), lane_(lane) {};

    uint32_t GetLane() const
    {
        return lane_;
    }

private:
    uint32_t lane_;
};

// SIMD vector register element
class VRegElem : public VRegVec {
public:
    explicit VRegElem(uint32_t index, uint32_t eidx, uint32_t bit, uint32_t lane)
        : VRegVec(index, bit, lane), elemIdx_(eidx)
    {
    }

    uint32_t GetElemIdx() const
    {
        return elemIdx_;
    }

private:
    uint32_t elemIdx_;
};

// base for SIMD Vector Register List
class VRegList : public VRegVec {
public:
    explicit VRegList(const VRegVec &s) : VRegVec(s.GetIdx(), s.GetBit(), s.GetLane()), len_(1) {}
    explicit VRegList(const VRegVec &s, const VRegVec &e)
        : VRegVec(s.GetIdx(), s.GetBit(), s.GetLane()), len_(((e.GetIdx() + VAL_32 - s.GetIdx()) % VAL_32) + 1)
    {
    }

    uint32_t GetLen() const
    {
        return len_;
    }

private:
    uint32_t len_;
};

class VReg4B;
class VReg8B;
class VReg16B;
class VReg2H;
class VReg4H;
class VReg8H;
class VReg2S;
class VReg4S;
class VReg1D;
class VReg2D;
class VReg1Q;

class VRegBElem : public VRegElem {
public:
    explicit VRegBElem(uint32_t index, uint32_t eidx, uint32_t lane) : VRegElem(index, eidx, BIT_8, lane) {}
};
class VRegHElem : public VRegElem {
public:
    explicit VRegHElem(uint32_t index, uint32_t eidx, uint32_t lane) : VRegElem(index, eidx, BIT_16, lane) {}
};
class VRegSElem : public VRegElem {
public:
    explicit VRegSElem(uint32_t index, uint32_t eidx, uint32_t lane) : VRegElem(index, eidx, BIT_32, lane) {}
};
class VRegDElem : public VRegElem {
public:
    explicit VRegDElem(uint32_t index, uint32_t eidx, uint32_t lane) : VRegElem(index, eidx, BIT_64, lane) {}
};
class VRegQElem : public VRegElem {
public:
    explicit VRegQElem(uint32_t index, uint32_t eidx, uint32_t lane) : VRegElem(index, eidx, BIT_128, lane) {}
};

class VReg4BList : public VRegList {
public:
    explicit VReg4BList(const VReg4B &s);
    VReg4BList(const VRegVec &s, const VRegVec &e) : VRegList(s, e) {}
    VRegBElem operator[](uint32_t i) const
    {
        assert(GetLane() > i);
        return VRegBElem(GetIdx(), i, GetLane());
    }
};
class VReg8BList : public VRegList {
public:
    explicit VReg8BList(const VReg8B &s);
    VReg8BList(const VRegVec &s, const VRegVec &e) : VRegList(s, e) {}
    VRegBElem operator[](uint32_t i) const
    {
        assert(GetLane() > i);
        return VRegBElem(GetIdx(), i, GetLane());
    }
};
class VReg16BList : public VRegList {
public:
    explicit VReg16BList(const VReg16B &s);
    VReg16BList(const VRegVec &s, const VRegVec &e) : VRegList(s, e) {}
    VRegBElem operator[](uint32_t i) const
    {
        assert(GetLane() > i);
        return VRegBElem(GetIdx(), i, GetLane());
    }
};
class VReg2HList : public VRegList {
public:
    explicit VReg2HList(const VReg2H &s);
    VReg2HList(const VRegVec &s, const VRegVec &e) : VRegList(s, e) {}
    VRegHElem operator[](uint32_t i) const
    {
        assert(GetLane() > i);
        return VRegHElem(GetIdx(), i, GetLane());
    }
};
class VReg4HList : public VRegList {
public:
    explicit VReg4HList(const VReg4H &s);
    VReg4HList(const VRegVec &s, const VRegVec &e) : VRegList(s, e) {}
    VRegHElem operator[](uint32_t i) const
    {
        assert(GetLane() > i);
        return VRegHElem(GetIdx(), i, GetLane());
    }
};
class VReg8HList : public VRegList {
public:
    explicit VReg8HList(const VReg8H &s);
    VReg8HList(const VRegVec &s, const VRegVec &e) : VRegList(s, e) {}
    VRegHElem operator[](uint32_t i) const
    {
        assert(GetLane() > i);
        return VRegHElem(GetIdx(), i, GetLane());
    }
};
class VReg2SList : public VRegList {
public:
    explicit VReg2SList(const VReg2S &s);
    VReg2SList(const VRegVec &s, const VRegVec &e) : VRegList(s, e) {}
    VRegSElem operator[](uint32_t i) const
    {
        assert(GetLane() > i);
        return VRegSElem(GetIdx(), i, GetLane());
    }
};
class VReg4SList : public VRegList {
public:
    explicit VReg4SList(const VReg4S &s);
    VReg4SList(const VRegVec &s, const VRegVec &e) : VRegList(s, e) {}
    VRegSElem operator[](uint32_t i) const
    {
        assert(GetLane() > i);
        return VRegSElem(GetIdx(), i, GetLane());
    }
};
class VReg1DList : public VRegList {
public:
    explicit VReg1DList(const VReg1D &s);
    VReg1DList(const VRegVec &s, const VRegVec &e) : VRegList(s, e) {}
    VRegDElem operator[](uint32_t i) const
    {
        assert(GetLane() > i);
        return VRegDElem(GetIdx(), i, GetLane());
    }
};
class VReg2DList : public VRegList {
public:
    explicit VReg2DList(const VReg2D &s);
    VReg2DList(const VRegVec &s, const VRegVec &e) : VRegList(s, e) {}
    VRegDElem operator[](uint32_t i) const
    {
        assert(GetLane() > i);
        return VRegDElem(GetIdx(), i, GetLane());
    }
};
class VReg1QList : public VRegList {
public:
    explicit VReg1QList(const VReg1Q &s);
    VReg1QList(const VRegVec &s, const VRegVec &e) : VRegList(s, e) {}
    VRegQElem operator[](uint32_t i) const
    {
        assert(GetLane() > i);
        return VRegQElem(GetIdx(), i, GetLane());
    }
};

class VReg4B : public VRegVec {
public:
    explicit VReg4B(uint32_t index) : VRegVec(index, VAL_8, LANE_4) {}
    VRegBElem operator[](uint32_t i) const
    {
        assert(GetLane() > i);
        return VRegBElem(GetIdx(), i, GetLane());
    }
    VReg4BList operator-(const VReg4B &other) const
    {
        return VReg4BList(*this, other);
    }
};
class VReg8B : public VRegVec {
public:
    explicit VReg8B(uint32_t index) : VRegVec(index, VAL_8, LANE_8) {}
    VRegBElem operator[](uint32_t i) const
    {
        assert(GetLane() > i);
        return VRegBElem(GetIdx(), i, GetLane());
    }
    VReg8BList operator-(const VReg8B &other) const
    {
        return VReg8BList(*this, other);
    }
};
class VReg16B : public VRegVec {
public:
    explicit VReg16B(uint32_t index) : VRegVec(index, VAL_8, LANE_16) {}
    VRegBElem operator[](uint32_t i) const
    {
        assert(GetLane() > i);
        return VRegBElem(GetIdx(), i, GetLane());
    }
    VReg16BList operator-(const VReg16B &other) const
    {
        return VReg16BList(*this, other);
    }
};
class VReg2H : public VRegVec {
public:
    explicit VReg2H(uint32_t index) : VRegVec(index, VAL_16, LANE_2) {}
    VRegHElem operator[](uint32_t i) const
    {
        assert(GetLane() > i);
        return VRegHElem(GetIdx(), i, GetLane());
    }
    VReg2HList operator-(const VReg2H &other) const
    {
        return VReg2HList(*this, other);
    }
};
class VReg4H : public VRegVec {
public:
    explicit VReg4H(uint32_t index) : VRegVec(index, VAL_16, LANE_4) {}
    VRegHElem operator[](uint32_t i) const
    {
        assert(GetLane() > i);
        return VRegHElem(GetIdx(), i, GetLane());
    }
    VReg4HList operator-(const VReg4H &other) const
    {
        return VReg4HList(*this, other);
    }
};
class VReg8H : public VRegVec {
public:
    explicit VReg8H(uint32_t index) : VRegVec(index, VAL_16, LANE_8) {}
    VRegHElem operator[](uint32_t i) const
    {
        assert(GetLane() > i);
        return VRegHElem(GetIdx(), i, GetLane());
    }
    VReg8HList operator-(const VReg8H &other) const
    {
        return VReg8HList(*this, other);
    }
};
class VReg2S : public VRegVec {
public:
    explicit VReg2S(uint32_t index) : VRegVec(index, VAL_32, LANE_2) {}
    VRegSElem operator[](uint32_t i) const
    {
        assert(GetLane() > i);
        return VRegSElem(GetIdx(), i, GetLane());
    }
    VReg2SList operator-(const VReg2S &other) const
    {
        return VReg2SList(*this, other);
    }
};

class VReg4S : public VRegVec {
public:
    explicit VReg4S(uint32_t index) : VRegVec(index, VAL_32, LANE_4) {}
    VRegSElem operator[](uint32_t i) const
    {
        assert(GetLane() > i);
        return VRegSElem(GetIdx(), i, GetLane());
    }
    VReg4SList operator-(const VReg4S &other) const
    {
        return VReg4SList(*this, other);
    }
};
class VReg1D : public VRegVec {
public:
    explicit VReg1D(uint32_t index) : VRegVec(index, VAL_64, LANE_1) {}
    VRegDElem operator[](uint32_t i) const
    {
        assert(GetLane() > i);
        return VRegDElem(GetIdx(), i, GetLane());
    }
    VReg1DList operator-(const VReg1D &other) const
    {
        return VReg1DList(*this, other);
    }
};
class VReg2D : public VRegVec {
public:
    explicit VReg2D(uint32_t index) : VRegVec(index, VAL_64, LANE_2) {}
    VRegDElem operator[](uint32_t i) const
    {
        assert(GetLane() > i);
        return VRegDElem(GetIdx(), i, GetLane());
    }
    VReg2DList operator-(const VReg2D &other) const
    {
        return VReg2DList(*this, other);
    }
};
class VReg1Q : public VRegVec {
public:
    explicit VReg1Q(uint32_t index) : VRegVec(index, VAL_128, LANE_1) {}
    VRegQElem operator[](uint32_t i) const
    {
        assert(GetLane() > i);
        return VRegQElem(GetIdx(), i, GetLane());
    }
    VReg1QList operator-(const VReg1Q &other) const
    {
        return VReg1QList(*this, other);
    }
};

inline VReg4BList::VReg4BList(const VReg4B &s) : VRegList(s, s) {}
inline VReg8BList::VReg8BList(const VReg8B &s) : VRegList(s, s) {}
inline VReg16BList::VReg16BList(const VReg16B &s) : VRegList(s, s) {}
inline VReg2HList::VReg2HList(const VReg2H &s) : VRegList(s, s) {}
inline VReg4HList::VReg4HList(const VReg4H &s) : VRegList(s, s) {}
inline VReg8HList::VReg8HList(const VReg8H &s) : VRegList(s, s) {}
inline VReg2SList::VReg2SList(const VReg2S &s) : VRegList(s, s) {}
inline VReg4SList::VReg4SList(const VReg4S &s) : VRegList(s, s) {}
inline VReg1DList::VReg1DList(const VReg1D &s) : VRegList(s, s) {}
inline VReg2DList::VReg2DList(const VReg2D &s) : VRegList(s, s) {}
inline VReg1QList::VReg1QList(const VReg1Q &s) : VRegList(s, s) {}

// SIMD vector register
class VReg : public VRegVec {
public:
    explicit VReg(uint32_t index)
        : VRegVec(index, BIT_128, LANE_1),
          b4(index),
          b8(index),
          b16(index),
          b(index),
          h2(index),
          h4(index),
          h8(index),
          h(index),
          s2(index),
          s4(index),
          s(index),
          d1(index),
          d2(index),
          d(index),
          q1(index),
          q(index)
    {
    }

    VReg4B b4;
    VReg8B b8;
    VReg16B b16;
    VReg16B b;
    VReg2H h2;
    VReg4H h4;
    VReg8H h8;
    VReg8H h;
    VReg2S s2;
    VReg4S s4;
    VReg4S s;
    VReg1D d1;
    VReg2D d2;
    VReg2D d;
    VReg1Q q1;
    VReg1Q q;
};

} // namespace ARMCG

#endif // CG_REG_V_HPP
