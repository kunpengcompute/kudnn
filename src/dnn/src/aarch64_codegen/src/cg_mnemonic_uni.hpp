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

#ifndef CG_MNEMONIC_UNI_HPP
#define CG_MNEMONIC_UNI_HPP

template <typename T>
void UniAdd(const T &regX1, const T &regX2, const T &op)
{
    Add(regX1, regX2, op);
}

void UniAdd(const VReg4S &regX1, const VReg4S &regX2, const VReg4S &op)
{
    Add(regX1, regX2, op);
}

void UniAdd(const ZReg &regX1, const ZReg &regX2, const ZReg &op)
{
    Add(ZRegS(regX1.GetIdx()), ZRegS(regX2.GetIdx()), ZRegS(op.GetIdx()));
}

template <typename T>
void UdivMod(const T &q, const T &r, const T &divend, const T &divisor)
{
    assert(q.getIdx() != divisor.getIdx());
    assert(q.getIdx() != divend.getIdx());
    assert(r.getIdx() != divend.getIdx());

    Udiv(q, divend, divisor);
    Mul(r, q, divisor);
    Sub(r, divend, r);
}

template <typename T>
void Umod(const T &r, const T &divend, const T &divisor)
{
    assert(r.getIdx() != divend.getIdx());
    assert(r.getIdx() != divisor.getIdx());

    Udiv(r, divend, divisor);
    Mul(r, r, divisor);
    Sub(r, divend, r);
}

void UniClear(const VReg &dst)
{
    Eor(dst.b, dst.b, dst.b);
}

void UniClear(const ZReg &dst)
{
    Eor(dst.d, dst.d, dst.d);
}

template <typename T>
void UniFadd(const T &dst, const T &src, const T &src2)
{
    Fadd(dst, src, src2);
}

void UniFcvtzs(const VReg4S &d, const VReg4S &s)
{
    Fcvtzs(d, s);
}

void UniFcvtzs(const ZRegS &d, const ZRegS &s)
{
    Fcvtzs(d, pAllOne / PredType::T_Z, s);
}

template <typename TReg>
void UniFdiv(const TReg &dst, const TReg &src, const TReg &src2, const TReg &tmp, const PReg &pred)
{
    uint32_t dstIdx = dst.getIdx();
    uint32_t srcIdx = src.getIdx();
    uint32_t src2Idx = src2.getIdx();
    uint32_t tmpIdx = tmp.getIdx();

    if (dstIdx == src2Idx) {
        assert(tmpIdx != srcIdx && tmpIdx != src2Idx);

        Mov(ZRegD(tmpIdx), ZRegD(src2Idx));
        Mov(dst, pred / PredType::T_M, src);
        Fdiv(dst, pred / PredType::T_M, tmp);
    } else if (dstIdx == srcIdx) {
        Fdiv(dst, pred / PredType::T_M, src2);
    } else {
        Mov(dst, pAllOne / PredType::T_M, src);
        Fdiv(dst, pred / PredType::T_M, src2);
    }
}

template <typename TReg>
void UniFdiv(const TReg &dst, const TReg &src, const TReg &src2)
{
    Fdiv(dst, src, src2);
}

void UniFdiv(const VReg4S &dst, const VReg4S &src, const VReg4S &src2, const VReg4S &tmp, const PReg &pred)
{
    (void)(tmp);
    (void)(pred);
    Fdiv(dst, src, src2);
}

template <typename T>
void UniFmad(const T &dst, const T &src, const T &src2)
{
    Fmad(dst, pAllOne / PredType::T_M, src, src2);
}

void UniFmad(const VReg4S &dst, const VReg4S &src, const VReg4S &src2)
{
    Fmul(dst, dst, src);
    Fadd(dst, dst, src2);
}

template <typename T>
void UniFmax(const T &dst, const T &src, const T &src2)
{
    uint32_t dstIdx = dst.getIdx();
    uint32_t srcIdx = src.getIdx();
    if (dstIdx != srcIdx) {
        Mov(ZRegD(dstIdx), ZRegD(srcIdx));
    }
    Fmax(dst, pAllOne / PredType::T_M, src2);
}

template <typename T>
void UniFmaxnm(const T &dst, const T &src, const T &src2)
{
    uint32_t dstIdx = dst.getIdx();
    uint32_t srcIdx = src.getIdx();
    if (dstIdx != srcIdx) {
        Mov(ZRegD(dstIdx), ZRegD(srcIdx));
    }
    Fmaxnm(dst, pAllOne / PredType::T_M, src2);
}

void UniFmaxnm(const VReg4S &dst, const VReg4S &src, const VReg4S &src2)
{
    Fmaxnm(dst, src, src2);
}

template <typename T>
void UniFmin(const T &dst, const T &src, const T &src2)
{
    uint32_t dstIdx = dst.getIdx();
    uint32_t srcIdx = src.getIdx();
    if (dstIdx != srcIdx) {
        Mov(ZRegD(dstIdx), ZRegD(srcIdx));
    }
    Fmin(dst, pAllOne / PredType::T_M, src2);
}

template <typename T>
void UniFmul(const T &dst, const T &src, const T &src2)
{
    Fmul(dst, src, src2);
}

void UniFrinti(const VReg4S &d, const VReg4S &s)
{
    Frinti(d, s);
}

void UniFrinti(const ZRegS &d, const ZRegS &s)
{
    Frinti(d, pAllOne / PredType::T_M, s);
}

template <typename T>
void UniFsqrt(const T &dst, const T &src)
{
    Fsqrt(dst, pAllOne / PredType::T_M, src);
}

void UniFsqrt(const VReg4S &dst, const VReg4S &src)
{
    Fsqrt(dst, src);
}

void UniFsub(const VReg4S &regV1, const VReg4S &regV2, const VReg4S &regV3)
{
    Fsub(regV1, regV2, regV3);
}

template <typename T>
void UniFsub(const T &dst, const T &src, const T &src2)
{
    Fsub(dst, src, src2);
}

void UniFsub(const ZRegS &regZ1, const ZRegS &regZ2, const ZRegS &regZ3)
{
    Fsub(regZ1, regZ2, regZ3);
}

void UniEor(const VReg &regV1, const VReg &regV2, const VReg &regV3)
{
    Eor(VReg16B(regV1.GetIdx()), VReg16B(regV2.GetIdx()), VReg16B(regV3.GetIdx()));
}

void UniEor(const ZReg &regZ1, const ZReg &regZ2, const ZReg &regZ3)
{
    Eor(ZRegD(regZ1.GetIdx()), ZRegD(regZ2.GetIdx()), ZRegD(regZ3.GetIdx()));
}

void UniLd1rw(const VReg4S &dst, const XReg &base, const int64_t off)
{
    if (off == 0) {
        Ld1r(VReg4SList(dst), Ptr(base));
    } else {
        AddImm(xDefaultAddr, base, off, X_TMP_0);
        Ld1r(VReg4SList(dst), Ptr(xDefaultAddr));
    }
}

void UniLd1rw(const ZRegS &dst, const XReg &base, const int64_t off)
{
    if (SIZE_NEG_32 <= off && off < VAL_32) {
        Ld1rw(dst, pAllOne / PredType::T_Z, AdrScImm(Ptr(base, static_cast<int>(off))));
    } else {
        AddImm(xDefaultAddr, base, off, X_TMP_0);
        Ld1rw(dst, pAllOne / PredType::T_Z, Ptr(xDefaultAddr));
    }
}

void UniLdr(const VReg &dst, const XReg &addr)
{
    Ldr(QReg(dst.GetIdx()), AdrUimm(Ptr(addr)));
}

void UniLdr(const ZReg &dst, const XReg &addr)
{
    Ldr(dst, Ptr(addr));
}

template <typename T>
void UniLdr(const ZReg &r, const XReg &base, const T off)
{
    assert(cpuSveLen != 0);
    const int offMod = off % cpuSveLen;
    const int offMulVl = off / cpuSveLen;

    if (offMod == 0 && offMulVl >= SIZE_NEG_256 && offMulVl <= VAL_255) {
        Ldr(r, Ptr(base, offMulVl, ExtMod::MUL_VL));
    } else {
        const int offset = offMod * 0x10 * (cpuSveLen / VAL_16);
        AddImm(xDefaultAddr, base, offset, X_TMP_0);
        Ldr(r, Ptr(xDefaultAddr));
    }
}

template <typename T>
void UniLdr(const VReg &r, const XReg &base, const T off)
{
    const int offMod = off % VAL_16;
    const int offMulVl = off / VAL_16;

    if (offMod == 0 && offMulVl >= 0 && offMulVl <= VAL_65520) {
        Ldr(QReg(r.GetIdx()), ptr(base, off));
    } else {
        AddImm(xDefaultAddr, base, offMod * VAL_4, X_TMP_0);
        Ldr(QReg(r.GetIdx()), AdrUimm(Ptr(xDefaultAddr)));
    }
}

void UniOrr(const VReg &d, const VReg &regS0, const VReg &regS1)
{
    Orr(d.b16, regS0.b16, regS1.b16);
}

void UniOrr(const ZReg &d, const ZReg &regS0, const ZReg &regS1)
{
    Orr(d.d, regS0.d, regS1.d);
}

void UniScvtf(const VReg4S &regT0, const VReg4S &regT1)
{
    Scvtf(regT0, regT1);
}

void UniScvtf(const ZRegS &regT0, const ZRegS &regT1)
{
    Scvtf(regT0, pAllOne / PredType::T_M, regT1);
}

void UniStr(const VReg &src, const XReg &addr)
{
    Str(QReg(src.GetIdx()), AdrUimm(Ptr(addr)));
}

void UniStr(const ZReg &src, const XReg &addr)
{
    Str(src, AdrNoOfs(Ptr(addr)));
}

template <typename T>
void UniSub(const T &regX1, const T &regX2, const T &op)
{
    sub(regX1, regX2, op);
}

#endif // CG_MNEMONIC_UNI_HPP
