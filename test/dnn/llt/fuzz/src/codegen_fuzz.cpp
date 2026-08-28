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

#include "fuzz_test_common.h"
#include "cg_codegen.hpp"
#include "cg_jit_generator.hpp"
#include "tools/cg_util.hpp"
#include "tools/cg_cpu_isa_traits.hpp"
#include "tools/cg_cpu_isa_type.hpp"

using namespace ARMCG;

class JitGenFuzz : public ARMCG::JitGenerator {
public:
    const char *Name() const override
    {
        return "JitGenFuzz";
    }
    const char *SourceFile() const override
    {
        return "codegen_fuzz.cpp";
    }
    template <class F>
    const F GetCode() const
    {
        return reinterpret_cast<F>(top_);
    }
    void Generate() override {};
};

class SubCodeArray : public CodeArray {
    public:
        SubCodeArray() : CodeArray(128) {
            UseProtect();
            ResetSize();
            SetSize(64);
            GrowMemory();
            Save(24, 8, [](int64_t x) { return static_cast<uint32_t>(x + 10); });
            CalcJmpAddress();
            addrInfoList_.begin() -> GetVal();
        }
        ~SubCodeArray() {}
    };

int main()
{
    DT_Enable_Leak_Check(0, 0);
    printf("codegen fuzz test start\n");
    // Verify inst generate, not actually execute
    JitGenFuzz cg;
    ARMCG::Label b1, b2, b3, b4, b5, b6, b7, b8, b9, b10;
    ARMCG::Label b11, b12, b13, b14;

    // add & sub
    cg.Mov(cg.x9, 2);
    cg.Mov(cg.x10, 1);
    cg.Add(cg.w11, cg.w10, cg.w9);
    cg.Adds(cg.x11, cg.x10, 10);
    cg.Adds(cg.x11, cg.x10, cg.x9);
    cg.Add(cg.x11, cg.x9, cg.x10, ARMCG::ExtMod::UXTW);
    cg.Madd(cg.x11, cg.x11, cg.x9, cg.x10);
    cg.Sub(cg.w12, cg.w9, 1);
    cg.Subs(cg.x12, cg.x9, cg.x10);
    cg.Subs(cg.w12, cg.w9, cg.w10);
    cg.Subs(cg.w12, cg.w9, cg.w10, ARMCG::ExtMod::UXTW);
    cg.Subs(cg.x12, cg.x9, cg.x10, ARMCG::ExtMod::UXTW);
    cg.Subs(cg.x12, cg.x9, cg.w10, ARMCG::ExtMod::UXTW);

    cg.Cmp(cg.w9, 5);
    cg.Beq(b1);
    cg.Mov(cg.w9, 2);
    cg.L(b1);

    cg.And(cg.x9, cg.x9, 0x80000000);
    cg.Movz(cg.w11, 65535);
    cg.Movk(cg.w12, 65535);
    cg.Mov(cg.w13, 65535);
    cg.Sxtb(cg.w9, cg.w9);
    cg.Lsl(cg.x10, cg.x10, 4);
    cg.Lsr(cg.x11, cg.x11, 4);
    cg.Uxtb(cg.w9, cg.w9);

    cg.Yield();

    cg.Mov(cg.x9, 4);
    cg.Mov(cg.x10, 0);
    cg.Whilelt(cg.p0.s, cg.x9, cg.x10);
    cg.Prfw(PrfopVC::PLDL1STRM_VC, cg.p0, AdrScImm(cg.x0));
    cg.Prfm(Prfop::PLDL1_STRM, AdrUimm(cg.x0, 0));
    cg.Dmb(BarOpt::LD);

    // label & jump
    cg.Adr(cg.x12, b2);
    cg.Br(cg.x12);
    cg.Mov(cg.x13, 0);
    cg.L(b2);
    cg.Cbz(cg.x9, b3);
    cg.Mov(cg.x9, 0);
    cg.L(b3);
    cg.Cbnz(cg.x9, b4);
    cg.Mov(cg.x9, 4);
    cg.L(b4);
    cg.Cbnz(cg.w9, b5);
    cg.Mov(cg.w9, 0);
    cg.L(b5);
    cg.Cmp(cg.x9, cg.x10);
    cg.Bne(b6);
    cg.L(b6);
    cg.Bmi(b7);
    cg.L(b7);
    cg.Bge(b8);
    cg.L(b8);
    cg.Bgt(b9);
    cg.L(b9);
    cg.Ble(b10);
    cg.L(b10);
    cg.Blt(b11);
    cg.L(b11);
    cg.BFirst(b12);
    cg.L(b12);
    cg.Bl(b13);
    cg.L(b13);

    // load & store
    cg.Ldr(cg.x9, AdrUimm(cg.x0, 0));
    cg.Ldaxr(cg.x9, AdrImm(cg.x0, 0));
    cg.Stlxr(cg.w9, cg.x10, AdrImm(cg.x4, 0));
    cg.Stlxr(cg.w9, cg.w10, AdrImm(cg.x4, 0));
    cg.Stp(cg.x9, cg.x10, AdrImm(cg.x4, 0));
    cg.Ldurb(cg.w11, AdrImm(cg.x0, 0));
    cg.Ldr(cg.w12, AdrPostImm(cg.x0, 0));
    cg.Ldr(cg.x13, AdrPostImm(cg.x0, 0));
    cg.Str(cg.x13, AdrPostImm(cg.x7, 0));
    cg.Ldr(cg.x14, AdrPostImm(cg.x0, 0));
    cg.Str(cg.x14, AdrPreImm(cg.x7, 0));
    cg.Str(cg.w14, AdrUimm(cg.x4, 0));
    cg.Str(cg.x14, AdrUimm(cg.x7, 0));
    cg.Ldaddal(cg.x15, cg.x14, AdrNoOfs(cg.x0));
    cg.Ldrb(cg.w15, AdrUimm(cg.x1, 0));
    cg.Strb(cg.w15, AdrUimm(cg.x5, 0));
    cg.Ldrh(cg.w15, AdrUimm(cg.x2, 0));
    cg.Strh(cg.w15, AdrUimm(cg.x6, 0));
    cg.Ldrsw(cg.x16, AdrUimm(cg.x0, 0));
    cg.Ldrsw(cg.x16, AdrPostImm (cg.x0, 0));

    // bit op
    cg.Mov(cg.x9, 4);
    cg.Mov(cg.x10, 8);
    cg.Udiv(cg.x11, cg.x10, cg.x9);
    cg.Mvn(cg.x9, cg.x9);
    cg.Eor(cg.x11, cg.x10, cg.x10);
    cg.Tst(cg.x10, cg.x10);

    // simd
    cg.Ld1(cg.v0.b[0], AdrNoOfs(cg.x1));
    cg.St1(cg.v0.b[0], AdrNoOfs(cg.x4));
    cg.Ld1(cg.v1.h[0], AdrNoOfs(cg.x2));
    cg.St1(cg.v1.h[0], AdrNoOfs(cg.x5));
    cg.Ld1(cg.v2.s[0], AdrNoOfs(cg.x2));
    cg.St1(cg.v2.s[0], AdrNoOfs(cg.x5));
    cg.Mov(cg.x15, 0);
    cg.St1(cg.v1.h[0], AdrPostReg(cg.x5, cg.x15));
    cg.St1(cg.v2.s[0], AdrPostReg(cg.x5, cg.x15));
    cg.Ld4((cg.v0.d2 - cg.v3.d2)[0], AdrNoOfs(cg.x3));
    cg.St4((cg.v0.d2 - cg.v3.d2)[0], AdrPostReg(cg.x7, cg.x15));
    cg.Ldp(cg.d9, cg.d10, AdrImm(cg.x3, 0));
    cg.Stp(cg.d9, cg.d10, AdrImm(cg.x7, 0));
    cg.Ldp(cg.d11, cg.d12, AdrPreImm(cg.x3, 0));
    cg.Stp(cg.d11, cg.d12, AdrPostImm(cg.x3, 0));
    cg.Ldr(cg.b0, AdrUimm(cg.x1, 0));
    cg.Str(cg.b0, AdrUimm(cg.x4, 0));
    cg.Ldr(cg.h1, AdrUimm(cg.x2, 0));
    cg.Str(cg.h1, AdrUimm(cg.x5, 0));
    cg.Ldr(cg.s2, AdrUimm(cg.x2, 0));
    cg.Str(cg.s2, AdrUimm(cg.x5, 0));
    cg.Ldr(cg.d3, AdrUimm(cg.x3, 0));
    cg.Str(cg.d3, AdrUimm(cg.x7, 0));
    cg.Ldr(cg.q4, AdrUimm(cg.x3, 0));
    cg.Str(cg.q4, AdrUimm(cg.x7, 0));

    cg.Fadd(cg.s2, cg.s2, cg.s1);

    cg.Dup(cg.v0.s4, cg.w9);
    cg.Dup(cg.v1.s4, cg.w10);
    cg.Dup(cg.v2.b16, cg.w10);
    cg.Uzp1(cg.v3.b16, cg.v0.b16, cg.v1.b16);
    cg.Uzp1(cg.v4.h8, cg.v0.h8, cg.v1.h8);
    cg.Ext(cg.v5.b16, cg.v1.b16, cg.v2.b16, 4);
    cg.Dup(cg.v6.b16, cg.v1.b[0]);
    cg.Mov(cg.w15, cg.v0.s[0]);
    cg.Mov(cg.v7.d[0], cg.x15);
    cg.Mov(cg.v8.s[0], cg.v7.s[0]);
    cg.Ins(cg.v1.b[0], cg.w9);
    cg.Ins(cg.v1.s[0], cg.v0.s[0]);
    cg.Ins(cg.v2.d[0], cg.v0.d[0]);

    cg.Bfcvtn(cg.v1.h4, cg.v0.s4);
    cg.Scvtf(cg.v2.s4, cg.v0.s4);
    cg.Fcvtzs(cg.v3.s4, cg.v0.s4);
    cg.Frinti(cg.v4.s4, cg.v0.s4);
    cg.Fsqrt(cg.v5.s4, cg.v0.s4);

    cg.Addv(cg.s1, cg.v0.s4);
    cg.Smax(cg.v2.b16, cg.v0.b16, cg.v1.b16);
    cg.Smax(cg.v3.s4, cg.v0.s4, cg.v1.s4);
    cg.Smin(cg.v4.s4, cg.v0.s4, cg.v1.s4);
    cg.Umin(cg.v5.b16, cg.v0.b16, cg.v1.b16);
    cg.Add(cg.v6.s4, cg.v0.s4, cg.v1.s4);
    cg.Sub(cg.v7.s4, cg.v0.s4, cg.v1.s4);
    cg.Fmaxnm(cg.v8.s4, cg.v0.s4, cg.v1.s4);
    cg.Fmax(cg.v8.s4, cg.v0.s4, cg.v1.s4);
    cg.Fmul(cg.v8.s4, cg.v0.s4, cg.v1.s4);
    cg.Fdiv(cg.v8.s4, cg.v0.s4, cg.v1.s4);
    cg.Fsub(cg.v8.s4, cg.v0.s4, cg.v1.s4);
    cg.Fmin(cg.v8.s4, cg.v0.s4, cg.v1.s4);
    cg.Bic(cg.v9.b16, cg.v0.b16, cg.v1.b16);
    cg.Orr(cg.v9.b16, cg.v0.b16, cg.v1.b16);
    cg.Eor(cg.v9.b16, cg.v0.b16, cg.v1.b16);
    cg.Mov(cg.v10.b16, cg.v0.b16);
    cg.Movi(cg.v1.b16, 1);
    // cg.Sxtl(cg.v2.h8, cg.v0.b8);
    cg.Sxtl(cg.v2.s4, cg.v0.h4);
    cg.Scvtf(cg.v2.s4, cg.v0.s4);
    // cg.Uxtl(cg.v2.h8, cg.v0.b8);
    cg.Uxtl(cg.v2.s4, cg.v0.h4);
    cg.Fcvtns(cg.v0.s4, cg.v1.s4);
    cg.Fmov(cg.s1, cg.w1);
    cg.Fmov(cg.w1, cg.s1);
    cg.Fmov(cg.x2, cg.d1);

    // sve
    cg.Ptrue(cg.p0.b, Pattern::ALL);
    cg.Pfalse(cg.p1.b);
    cg.Orr(cg.z1.s, cg.p0.s, cg.z0.s);
    cg.Orr(cg.z2.d, cg.z1.d, cg.z0.d);
    cg.Eor(cg.z1.s, cg.p0.s, cg.z0.s);
    cg.Eor(cg.z1.d, cg.z1.d, cg.z0.d);
    cg.And(cg.z1.b, cg.p0.b, cg.z0.b);
    cg.And(cg.z1.s, cg.p0.s, cg.z0.s);
    cg.And(cg.z1.d, cg.z1.d, cg.z0.d);
    cg.Smin(cg.z1.s, cg.p0.s, cg.z0.s);
    cg.Mul(cg.z1.s, cg.p0.s, cg.z0.s);
    cg.Movprfx(cg.z1.s, cg.p0.s, cg.z0.s);
    cg.Asr(cg.z1.s, cg.p0.s, 1);
    cg.Lsl(cg.z1.s, cg.p0.s, 1);
    cg.Lsl(cg.z1.h, cg.p0.h, cg.z0.h);
    cg.Lsl(cg.z1.s, cg.p0.h, cg.z0.s);
    cg.Fabs(cg.z1.s, cg.p0.s, cg.z0.s);
    cg.Fneg(cg.z1.s, cg.p0.s, cg.z0.s);
    cg.Sxtb(cg.z1.s, cg.p0.s, cg.z0.s);
    cg.Uxtb(cg.z1.s, cg.p0.s, cg.z0.s);
    cg.Add(cg.z2.s, cg.z0.s, cg.z1.s);
    cg.Sub(cg.z2.b, cg.z0.b, cg.z1.b);
    cg.Sub(cg.z2.s, cg.z0.s, cg.z1.s);
    cg.Index(cg.z1.h, 0, 1);
    cg.Index(cg.z1.s, 0, 1);
    cg.Asr(cg.z2.s, cg.z0.s, 4);
    cg.Lsr(cg.z2.s, cg.z0.s, 4);
    cg.Lsl(cg.z2.s, cg.z0.s, 4);
    cg.Fexpa(cg.z2.s, cg.z0.s);
    cg.Incb(cg.x9);
    cg.Inch(cg.x10);
    cg.Decw(cg.x11);
    cg.Eor(cg.z0.s, 1);
    cg.And(cg.z0.s, 1);
    cg.Dupm(cg.z0.s, 1);
    cg.Fcpy(cg.z0.s, cg.p0.s, 1.0);
    cg.Mov(cg.z0.b, cg.p0.b, 1);
    cg.Mov(cg.z0.s, cg.p0.s, 1);
    cg.Mov(cg.z0.s, cg.w10);
    cg.Ext(cg.z1.b, cg.z0.b, 1);
    cg.Dup(cg.z0.h, cg.w9);
    cg.Dup(cg.z1.s, cg.w9);
    cg.Dup(cg.z1.b, cg.z0.b[0]);
    cg.Dup(cg.z1.s, cg.z0.s[0]);
    cg.Zip1(cg.p2.b, cg.p1.b, cg.p0.b);
    cg.Zip1(cg.p2.h, cg.p1.h, cg.p0.h);
    cg.Uzp1(cg.p2.b, cg.p1.b, cg.p0.b);
    cg.Uzp1(cg.p2.h, cg.p1.h, cg.p0.h);
    cg.Trn1(cg.p2.s, cg.p1.s, cg.p0.s);
    cg.Trn1(cg.p2.d, cg.p1.d, cg.p0.d);
    cg.Zip1(cg.z2.b, cg.z0.b, cg.z1.b);
    cg.Zip1(cg.z2.h, cg.z0.h, cg.z1.h);
    cg.Zip1(cg.z2.s, cg.z0.s, cg.z1.s);
    cg.Zip2(cg.z2.b, cg.z0.b, cg.z1.b);
    cg.Zip2(cg.z2.h, cg.z0.h, cg.z1.h);
    cg.Zip2(cg.z2.s, cg.z0.s, cg.z1.s);
    cg.Uzp1(cg.z2.b, cg.z0.b, cg.z1.b);
    cg.Uzp1(cg.z2.h, cg.z0.h, cg.z1.h);
    cg.Uzp2(cg.z2.d, cg.z0.d, cg.z1.d);
    cg.Trn1(cg.z2.s, cg.z0.s, cg.z1.s);
    cg.Trn1(cg.z2.d, cg.z0.d, cg.z1.d);
    cg.Trn2(cg.z2.s, cg.z0.s, cg.z1.s);
    cg.Trn2(cg.z2.d, cg.z0.d, cg.z1.d);
    cg.Splice(cg.z1.d, cg.p0.d, cg.z0.d);
    cg.Sel(cg.z2.b, cg.p0.b, cg.z0.b, cg.z1.b);
    cg.Sel(cg.z2.s, cg.p0.s, cg.z0.s, cg.z1.s);
    cg.Mov(cg.z1.b, cg.p0.b, cg.z0.b);
    cg.Mov(cg.z1.s, cg.p0.s, cg.z0.s);
    cg.Mov(cg.z1.d, cg.p0.d, cg.z0.d);
    cg.Cmpeq(cg.p2.s, cg.p0.s, cg.z0.s, cg.z1.s);
    cg.Cmpls(cg.p2.b, cg.p0.b, cg.z0.b, cg.z1.b);
    cg.Cmplt(cg.p2.b, cg.p0.b, cg.z0.b, cg.z1.b);
    cg.Cmplt(cg.p2.s, cg.p0.s, cg.z0.s, cg.z1.s);
    cg.And(cg.p2.b, cg.p0.b, cg.p0.b, cg.p1.b);
    cg.Mov(cg.p2.b, cg.p1.b, cg.p0.b);
    cg.Eor(cg.p3.b, cg.p0.b, cg.p1.b, cg.p2.b);
    cg.Mov(cg.p1.b, cg.p0.b);
    cg.Ptrue(cg.p4.h, Pattern::ALL);
    cg.Ptrue(cg.p5.d, Pattern::ALL);
    cg.Cmplt(cg.p2.s, cg.p0.s, cg.z0.s, 1);
    cg.Cmple(cg.p2.s, cg.p0.s, cg.z0.s, 1);
    cg.Cmpeq(cg.p2.s, cg.p0.s, cg.z0.s, 1);
    cg.Cmpne(cg.p2.s, cg.p0.s, cg.z0.s, 1);
    cg.Mov(cg.x9, 4);
    cg.Mov(cg.x10, 0);
    cg.Whilelt(cg.p0.b, cg.x9, cg.x10);
    cg.Whilelt(cg.p0.h, cg.x9, cg.x10);
    cg.Fdup(cg.z1.s, 1.0);
    cg.Fmov(cg.z1.s, 1.0);
    cg.Dup(cg.z1.b, 1);
    cg.Dup(cg.z1.s, 1);
    cg.Dup(cg.z1.d, 1);
    cg.Mov(cg.z1.h, 1);
    cg.Mov(cg.z1.s, 1);
    cg.Smax(cg.z1.s, 1);
    cg.Smin(cg.z1.s, 1);
    cg.Umin(cg.z1.s, 1);
    cg.Udot(cg.z2.s, cg.z1.b, cg.z0.b);
    cg.Usdot(cg.z2.s, cg.z1.b, cg.z0.b);
    cg.Sdot(cg.z2.s, cg.z1.b, cg.z0.b);
    cg.Saddv(cg.d2, cg.p0, cg.z0.s);
    cg.Uaddv(cg.d2, cg.p0, cg.z0.s);
    cg.Frecpe(cg.z1.s, cg.z0.s);
    cg.Fcmge(cg.p2.s, cg.p0.s, cg.z0.s, 0.0);
    cg.Fcmgt(cg.p2.s, cg.p0.s, cg.z0.s, 0.0);
    cg.Fcmlt(cg.p2.s, cg.p0.s, cg.z0.s, 0.0);
    cg.Fcmle(cg.p2.s, cg.p0.s, cg.z0.s, 0.0);
    cg.Fcmeq(cg.p2.s, cg.p0.s, cg.z0.s, 0.0);
    cg.Fsub(cg.z2.s, cg.z0.s, cg.z1.s);
    cg.Frecps(cg.z2.s, cg.z0.s, cg.z1.s);
    cg.Fadd(cg.z2.s, cg.p0.s, cg.z1.s);
    cg.Fmul(cg.z2.h, cg.p0.h, cg.z1.h);
    cg.Fmaxnm(cg.z2.s, cg.p0.s, cg.z1.s);
    cg.Fminnm(cg.z2.s, cg.p0.s, cg.z1.s);
    cg.Fmax(cg.z2.s, cg.p0.s, cg.z1.s);
    cg.Fmin(cg.z2.s, cg.p0.s, cg.z1.s);
    cg.Fscale(cg.z2.s, cg.p0.s, cg.z1.s);
    cg.Fdiv(cg.z2.s, cg.p0.s, cg.z1.s);
    cg.Fadd(cg.z2.s, cg.p0.s, 1);
    cg.Fsub(cg.z2.s, cg.p0.s, 1);
    cg.Fmul(cg.z2.s, cg.p0.s, 2);
    cg.Fmaxnm(cg.z2.s, cg.p0.s, 1);
    cg.Fminnm(cg.z2.s, cg.p0.s, 1);
    cg.Fmax(cg.z2.s, cg.p0.s, 1);
    cg.Fcvt(cg.z2.h, cg.p0, cg.z1.s);
    cg.Fcvt(cg.z2.s, cg.p0, cg.z1.h);
    cg.Fcvtzs(cg.z2.s, cg.p0.s, cg.z1.s);
    cg.Fcvtzu(cg.z2.s, cg.p0.s, cg.z1.s);
    cg.Frintn(cg.z2.s, cg.p0.s, cg.z1.s);
    cg.Frintm(cg.z2.s, cg.p0.s, cg.z1.s);
    cg.Frinti(cg.z2.s, cg.p0.s, cg.z1.s);
    cg.Fsqrt(cg.z2.s, cg.p0.s, cg.z1.s);
    cg.Scvtf(cg.z2.s, cg.p0.s, cg.z1.s);
    cg.Fcmge(cg.p2.s, cg.p0.s, cg.z0.s, cg.z1.s);
    cg.Fcmgt(cg.p2.s, cg.p0.s, cg.z0.s, cg.z1.s);
    cg.Fcmle(cg.p2.s, cg.p0.s, cg.z0.s, cg.z1.s);
    cg.Fcmlt(cg.p2.s, cg.p0.s, cg.z0.s, cg.z1.s);
    cg.Fcmeq(cg.p2.s, cg.p0.s, cg.z0.s, cg.z1.s);
    cg.Fcmne(cg.p2.s, cg.p0.s, cg.z0.s, cg.z1.s);
    cg.Facge(cg.p2.s, cg.p0.s, cg.z0.s, cg.z1.s);
    cg.Fmls(cg.z2.s, cg.p0.s, cg.z1.s, cg.z0.s);
    cg.Fmad(cg.z2.s, cg.p0.s, cg.z1.s, cg.z0.s);
    cg.Fnmsb(cg.z2.s, cg.p0.s, cg.z1.s, cg.z0.s);

    cg.Ld1sb(cg.z1.s, cg.p0.s, AdrSc32U(cg.x0, cg.z0.s, ExtMod::UXTW));
    cg.Ld1b(cg.z1.s, cg.p0.s, AdrSc32U(cg.x0, cg.z0.s, ExtMod::UXTW));
    cg.Ld1h(cg.z1.s, cg.p0.s, AdrSc32U(cg.x0, cg.z0.s, ExtMod::UXTW));
    cg.Ld1w(cg.z1.s, cg.p0.s, AdrSc32U(cg.x0, cg.z0.s, ExtMod::UXTW));
    cg.Ld1w(cg.z1.s, cg.p0.s, AdrSc32S(cg.x0, cg.z0.s, ExtMod::UXTW, 0));
    cg.Prfh(PrfopVC::PLDL1STRM_VC, cg.p0, AdrNoOfs(cg.x0));
    cg.Prfw(PrfopVC::PLDL1STRM_VC, cg.p0, AdrNoOfs(cg.x0));
    cg.Ld1rb(cg.z1.s, cg.p0.s, AdrNoOfs(cg.x0));
    cg.Ld1rh(cg.z1.s, cg.p0.s, AdrNoOfs(cg.x0));
    cg.Ld1rh(cg.z1.h, cg.p0.s, AdrScImm(cg.x0, 0));
    cg.Ld1rh(cg.z1.s, cg.p0.s, AdrScImm(cg.x0, 0));
    cg.Ld1rsb(cg.z1.s, cg.p0.s, AdrNoOfs(cg.x0));
    cg.Ldr(cg.p0.s, AdrScImm(cg.x0, 0));
    cg.Ldr(cg.p0.s, AdrNoOfs(cg.x0));
    cg.Ldr(cg.z0.s, AdrNoOfs(cg.x0));
    cg.Mov(cg.x11, 0);
    cg.Ld1b(cg.z1.b, cg.p0.b, AdrNoOfs(cg.x0));
    cg.Ld1b(cg.z1.b, cg.p0.b, AdrScImm(cg.x0, 0));
    cg.Ld1b(cg.z1.s, cg.p0.b, AdrNoOfs(cg.x0));
    cg.Ld1b(cg.z1.s, cg.p0.b, AdrScImm(cg.x0, 0));
    // cg.Ld1h(cg.z1.h, cg.p0, AdrScSc(cg.x0, cg.x11, ShMod::LSL, 1));
    cg.Ld1h(cg.z1.h, cg.p0.b, AdrScImm(cg.x0, 0));
    cg.Ld1w(cg.z1.d, cg.p0.d, AdrNoOfs(cg.x0));
    cg.Ld1sb(cg.z1.s, cg.p0.s, AdrScImm(cg.x0, 0));
    cg.Ld1sb(cg.z1.s, cg.p0.s, AdrNoOfs(cg.x0));
    cg.Ld1d(cg.z1.d, cg.p0.s, AdrNoOfs(cg.x0));
    // cg.Ld1w(cg.z1.s, cg.p0, AdrScSc(cg.x0, cg.x11, ShMod::LSL, 1));
    cg.Stnt1w(cg.z1.s, cg.p0.s, AdrNoOfs(cg.x0));
    cg.Stnt1d(cg.z1.d, cg.p0.d, AdrNoOfs(cg.x0));
    cg.St1b(cg.z1.b, cg.p0.b, AdrScImm(cg.x0, 0));
    cg.St1b(cg.z1.s, cg.p0.s, AdrScImm(cg.x0, 0));
    cg.St1b(cg.z1.b, cg.p0.b, AdrNoOfs(cg.x0));
    cg.St1b(cg.z1.s, cg.p0.s, AdrNoOfs(cg.x0));
    // cg.St1w(cg.z1.s, cg.p0, AdrScSc(cg.x0, cg.x11, ShMod::LSL, 1));
    cg.Str(cg.p1, AdrScImm(cg.x0, 0));
    cg.Str(cg.p1, AdrNoOfs(cg.x0));
    cg.Str(cg.z1.b, AdrScImm(cg.x0, 0));
    cg.Str(cg.z1.b, AdrNoOfs(cg.x0));
    cg.Tbz(cg.w9, 1, b14);
    cg.L(b14);

    //vreg load and store
    int regIdx = 0;
    cg.Ld1(VReg4SList(VReg(regIdx).s), Ptr(cg.x16));
    cg.St1(VReg4SList(VReg(regIdx).s), AdrNoOfs(cg.x0));
    cg.Ld1(VReg4SList(VReg(regIdx).s), AdrNoOfs(cg.x1));
    cg.Fmla(VReg(0).s, VReg(1).s, VReg(2).s);
    cg.Fmla(VReg(3).s, VReg(4).s, VReg(5).s);
    cg.Movi(VReg(6).s, 0);
    cg.Scvtf(VReg(0).s, VReg(1).s, 1);
    //util
    GetCpu().GetCacheType(Util::Arm64CacheLevel::L1);
    Util::CpuInfo info;
    Util::Cpu cpu;
    cpu.GetLastDataCacheLevel();
    cpu.GetType();
    cpu.IsBf16Supported();
    GetCpu().Has(Util::HwCap::CG_HWCAP_VC);
    GetCpu().DumpCacheInfo();
    info.GetCacheSize(Util::CacheType::ICACHE, 6);
    info.GetCacheSize(Util::CacheType::DCACHE, 6);
    info.GetCacheSize(Util::CacheType::UCACHE, 6);
    info.GetCodeCacheSize(5);
    info.GetLastDataCacheLevel();
    info.GetNumCores(Util::Arm64CpuTopologyLevel::SMT);
    info.GetNumCores(Util::Arm64CpuTopologyLevel::CORE);
    info.GetUnifiedCacheSize(1);
    info.Put();
    info.GetType();
    GetCpu().IsAtomicSupported();
    cg.Align(4);
    SubCodeArray *sca = new SubCodeArray();
    delete sca; 

#ifdef KUDNNL_920F
    //sme 
    cg.Smstart(StreamMode::SMZA);
    cg.Zero({cg.za0});
    cg.Smops(cg.za0.s, cg.p0, cg.p1, cg.z0.b, cg.z1.b);
    cg.Smopa(cg.za0.s, cg.p0, cg.p1, cg.z0.b, cg.z1.b);
    cg.Sumopa(cg.za0.s, cg.p0, cg.p1, cg.z0.b, cg.z1.b);
    cg.Sumops(cg.za0.s, cg.p0, cg.p1, cg.z0.b, cg.z1.b);
    cg.Umopa(cg.za0.s, cg.p0, cg.p1, cg.z0.b, cg.z1.b);
    cg.Umops(cg.za0.s, cg.p0, cg.p1, cg.z0.b, cg.z1.b);
    cg.Usmopa(cg.za0.s, cg.p0, cg.p1, cg.z0.b, cg.z1.b);
    cg.Usmops(cg.za0.s, cg.p0, cg.p1, cg.z0.b, cg.z1.b);
    cg.Mova(cg.za0h.b, cg.w12, 0, cg.p0, cg.z0.b);
    cg.Mov(cg.za0h.b, cg.w12, 1, cg.p0, cg.z0.b);
    cg.Mova(cg.za0v.b, cg.w12, 0, cg.p0, cg.z0.b);
    cg.Mov(cg.za0v.b, cg.w12, 1, cg.p0, cg.z0.b);
    cg.Mova(cg.z1.b, cg.p0, cg.za0h.b, cg.w12, 0);
    cg.Mov(cg.z2.b, cg.p0, cg.za0h.b, cg.w12, 1);
    cg.Mov(cg.x11, 0);
    cg.St1b(cg.z1.b, cg.p0, AdrNoOfs(cg.x1));
    cg.Mov(cg.w12, 0);
    cg.Mov(cg.w12, 0);
    cg.Ld1b(cg.za0v.b, cg.w12, 0, cg.p0, AdrScSc(cg.x0, cg.x9));
    cg.St1b(cg.za0v.b, cg.w12, 0, cg.p0, AdrScSc(cg.x1, cg.x9));
    cg.St1b(cg.za0h.b, cg.w12, 0, cg.p0, AdrScSc(cg.x1, cg.x9));
    cg.Addha(cg.za0.s, cg.p1, cg.p0, cg.z0.s);
    cg.Addva(cg.za1.s, cg.p0, cg.p1, cg.z0.s);
    cg.Add(cg.x8, cg.x8, 4);
    cg.St1w(cg.za0h.s, cg.w12, 1, cg.p0, AdrScSc(cg.x1, cg.x8, ShMod::LSL, 2));
    cg.St1w(cg.za1v.s, cg.w12, 1, cg.p0, AdrScSc(cg.x2, cg.x8, ShMod::LSL, 2));
    cg.Smstop(StreamMode::SMZA);
#endif
    cg.Ret();
    cg.Ready();
    cg.GetCode<int (*)(int *, uint8_t *, float *, uint64_t*, int *, uint8_t *, float *, uint64_t *)>();
    printf("codegen fuzz test end\n");
    return 0;
}
