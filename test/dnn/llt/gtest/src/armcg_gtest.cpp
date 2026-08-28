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
#include "gtest/gtest.h"
#include "cg_codegen.hpp"
#include "cg_jit_generator.hpp"
#include "tools/cg_util.hpp"
#include "tools/cg_cpu_isa_traits.hpp"
#include "tools/cg_cpu_isa_type.hpp"

using namespace ARMCG;
namespace dnnl {
class armcg_test_t : public ::testing::Test {
public:
    void SetUp() override {
    }

    void TearDown() override { 
    }
};

class GTestJitGen : public ARMCG::JitGenerator {
public:
    const char *Name() const override
    {
        return "GTestJitGen";
    }
    const char *SourceFile() const override
    {
        return "armcg_gtest.cpp";
    }
    template <class F>
    const F GetCode() const
    {
        return reinterpret_cast<F>(top_);
    }
    void Generate() override {};
};

float F16toF32(const uint16_t x) { // IEEE-754 16-bit floating-point format (without infinity): 1-5-10, exp-15, +-131008.0, +-6.1035156E-5, +-5.9604645E-8, 3.311 digits
    const unsigned e = (x&0x7C00)>>10; // exponent
    const unsigned m = (x&0x03FF)<<13; // mantissa
    const unsigned v = Util::BitCast<unsigned>((float)m)>>23; // evil log2 bit hack to count leading zeros in denormalized format
    return Util::BitCast<float>((x&0x8000)<<16 | (e!=0)*((e+112)<<23|m) | ((e==0)&(m!=0))*((v-37)<<23|((m<<(150-v))&0x007FE000))); // sign : normalized : denormalized
}

float DotProduct2WayF16(uint16_t* matA, int startIdA, uint16_t* matB, int startIdB)
{
    return F16toF32(matA[startIdA]) * F16toF32(matB[startIdB]) + F16toF32(matA[startIdA + 1]) * F16toF32(matB[startIdB + 1]);
}

int DotProduct4Way(int8_t* matA, int startIdA, int8_t* matB, int startIdB)
{
    return matA[startIdA] * matB[startIdB] + matA[startIdA + 1] * matB[startIdB + 1] +
        matA[startIdA+ 2] * matB[startIdB + 2] + matA[startIdA + 3] * matB[startIdB + 3];
}

int DotProduct4Way(int8_t* matA, int startIdA, uint8_t* matB, int startIdB)
{
    return matA[startIdA] * matB[startIdB] + matA[startIdA + 1] * matB[startIdB + 1] +
        matA[startIdA+ 2] * matB[startIdB + 2] + matA[startIdA + 3] * matB[startIdB + 3];
}

int DotProduct4Way(uint8_t* matA, int startIdA, uint8_t* matB, int startIdB)
{
    return matA[startIdA] * matB[startIdB] + matA[startIdA + 1] * matB[startIdB + 1] +
        matA[startIdA+ 2] * matB[startIdB + 2] + matA[startIdA + 3] * matB[startIdB + 3];
}

int DotProduct4Way(uint8_t* matA, int startIdA, int8_t* matB, int startIdB)
{
    return matA[startIdA] * matB[startIdB] + matA[startIdA + 1] * matB[startIdB + 1] +
        matA[startIdA+ 2] * matB[startIdB + 2] + matA[startIdA + 3] * matB[startIdB + 3];
}

TEST(armcg_test_t, TestAdd)
{
    CodeGenerator gen;
    gen.Ldr(gen.w9, AdrUimm(gen.x0, 0));
    gen.Ldr(gen.w10, AdrUimm(gen.x0, 4));
    gen.Add(gen.w0, gen.w9, gen.w10);
    gen.Add(gen.w0, gen.w0, gen.w1);
    gen.Ret();
    gen.Ready();
 
    auto f = gen.GetCode<int (*)(int*, int)>();
    int arr[] = {3, 4};
    int scale = 5;
    ASSERT_EQ(f(arr, scale), 12);
}

TEST(armcg_test_t, TestSqrt)
{
    CodeGenerator cg;
    cg.Mov(cg.x9, 0);
    cg.Whilelt(cg.p0.s, cg.x9, cg.x1);
    cg.Ld1w(cg.z0.s, cg.p0, AdrNoOfs(cg.x0));
    cg.Fsqrt(cg.z0.s, cg.p0, cg.z0.s);
    cg.St1w(cg.z0.s, cg.p0, AdrNoOfs(cg.x0));
    cg.Ret();
    cg.Ready();

    auto F = cg.GetCode<void (*)(float*, int)>();
    float arr1[] = {0.0f, 9.0f, 25.0f, 0.046f, 40.0f, 20.0f, 0.0805929987f, 0.524f};
    float arr2[] = {0.0f, 9.0f, 25.0f, 0.046f, 40.0f, 20.0f, 0.0805929987f, 0.524f};
    F(arr1, 8);
    ASSERT_EQ((float)sqrt(arr2[0]), arr1[0]);
    ASSERT_EQ((float)sqrt(arr2[1]), arr1[1]);
    ASSERT_EQ((float)sqrt(arr2[2]), arr1[2]);
    ASSERT_EQ((float)sqrt(arr2[3]), arr1[3]);
    ASSERT_EQ((float)sqrt(arr2[4]), arr1[4]);
    ASSERT_EQ((float)sqrt(arr2[5]), arr1[5]);
    ASSERT_EQ((float)sqrt(arr2[6]), arr1[6]);
    ASSERT_EQ((float)sqrt(arr2[7]), arr1[7]);
}

TEST(armcg_test_t, TestSub)
{
    GTestJitGen cg;
    cg.Ldr(cg.w9, AdrUimm(cg.x0, 0));
    cg.Ldr(cg.w10, AdrUimm(cg.x0, 4));
    cg.Subs(cg.w0, cg.w9, cg.w10);
    cg.Subs(cg.w0, cg.w0, cg.w1);
    cg.Ret();
    cg.Ready();
    
    auto f = cg.GetCode<int(*)(int*, int)>();
    int arr[] = {9, 2};
    int scale = 1;
    ASSERT_EQ(f(arr, scale), 6);
    
    GTestJitGen *cg1 = new GTestJitGen();
    cg1->Ldr(cg.w9, AdrUimm(cg.x0, 0));
    cg1->Ldr(cg.w10, AdrUimm(cg.x0, 4));
    cg1->Subs(cg.w0, cg.w9, cg.w10);
    cg1->Subs(cg.w0, cg.w0, cg.w1);
    cg1->Preamble();
    cg1->Postamble();
    bool ret = cg1->CreateKernel();
    delete cg1;
    ASSERT_EQ(ret, true);
}

TEST(armcg_test_t, TestMul)
{
    GTestJitGen cg;
    cg.Ldr(cg.x9, AdrUimm(cg.x0, 0));
    cg.Ldr(cg.x10, AdrUimm(cg.x0, 8));
    cg.Mul(cg.x0, cg.x9, cg.x10);
    cg.Mul(cg.x0, cg.x0, cg.x1);
    cg.Ret();
    cg.Ready();
    auto f = cg.GetCode<uint64_t(*)(uint64_t*, uint64_t)>();
    uint64_t arr[] = {9, 2};
    uint64_t scale = 5;
    ASSERT_EQ(f(arr, scale), 90);
}

TEST(armcg_test_t, TestDiv)
{
    GTestJitGen cg;
    cg.Ldr(cg.x9, AdrUimm(cg.x0, 0));
    cg.Ldr(cg.x10, AdrUimm(cg.x0, 8));
    cg.Udiv(cg.x0, cg.x9, cg.x10);
    cg.Udiv(cg.x0, cg.x0, cg.x1);
    cg.Ret();
    cg.Ready();
    auto f = cg.GetCode<uint64_t(*)(uint64_t*, uint64_t)>();
    uint64_t arr[] = {1000, 2};
    uint64_t scale = 5;
    ASSERT_EQ(f(arr, scale), 100);
}

// basic test
TEST(armcg_test_t, TestAdr)
{
    GTestJitGen cg;
    Label b;
    cg.Adr(cg.x1, b);
    cg.Br(cg.x1);
    cg.Mov(cg.x0, 0);
    cg.L(b);
    cg.Ret();
    cg.Ready();
    auto f = cg.GetCode<int (*)(int)>();
    ASSERT_EQ(f(5), 5);

    ARMCG::XReg rd(100);
    EXPECT_THROW(cg.Adr(rd, b), std::exception);
    EXPECT_THROW(cg.Mov(rd, 31), std::exception);
}

TEST(armcg_test_t, TestB1)
{
    GTestJitGen cg;
    Label b;
    b.Clear();
    cg.B(b);
    cg.Mov(cg.x0, 0);
    cg.Bl(b);
    cg.L(b);
    cg.Ret();
    cg.Ready();
    auto f = cg.GetCode<int (*)(int)>();
    auto value = f(5);
    ASSERT_EQ(value, 5);
}

TEST(armcg_test_t, TestB2)
{
    GTestJitGen cg;
    Label b;
    b.Clear();
    cg.Cmp(cg.x0, 5);
    cg.B(Cond::EQ, b);
    cg.Mov(cg.x0, 0);
    cg.L(b);
    cg.Ret();
    cg.Ready();
    auto f = cg.GetCode<int (*)(int)>();
    auto value = f(5);
    ASSERT_EQ(value, 5);
}

TEST(armcg_test_t, TestBeq)
{
    GTestJitGen cg;
    Label b;
    b.Clear();
    cg.Cmp(cg.x0, 5);
    cg.Beq(b);
    cg.Mov(cg.x0, 0);
    cg.L(b);
    cg.Ret();
    cg.Ready();

    auto f = cg.GetCode<int (*)(int)>();
    auto value = f(5);
    ASSERT_EQ(value, 5);
}

TEST(armcg_test_t, TestBge)
{
    GTestJitGen cg;
    Label b;
    b.Clear();
    cg.Cmp(cg.x0, 5);
    cg.Bge(b);
    cg.Mov(cg.x0, 0);
    cg.L(b);
    cg.Ret();
    cg.Ready();

    auto f = cg.GetCode<int (*)(int)>();
    auto value = f(5);
    ASSERT_EQ(value, 5);
}

TEST(armcg_test_t, TestBgt)
{
    GTestJitGen cg;
    Label b;
    b.Clear();
    cg.Cmp(cg.x0, 5);
    cg.Bgt(b);
    cg.Mov(cg.x0, 0);
    cg.L(b);
    cg.Ret();
    cg.Ready();

    auto f = cg.GetCode<int (*)(int)>();
    auto value = f(5);
    ASSERT_EQ(value, 0);
}

TEST(armcg_test_t, TestBle)
{
    GTestJitGen cg;
    Label b;
    cg.Cmp(cg.x0, 5);
    cg.Ble(b);
    cg.Mov(cg.x0, 0);
    cg.L(b);
    cg.Ret();
    cg.Ready();

    auto f = cg.GetCode<int (*)(int)>();
    auto value = f(5);
    ASSERT_EQ(value, 5);
}

TEST(armcg_test_t, TestBlt)
{
    GTestJitGen cg;
    Label b;
    cg.Cmp(cg.x0, 5);
    cg.Blt(b);
    cg.Mov(cg.x0, 0);
    cg.L(b);
    cg.Ret();
    cg.Ready();

    auto f = cg.GetCode<int (*)(int)>();
    auto value = f(5);
    ASSERT_EQ(value, 0);
}

TEST(armcg_test_t, TestBne)
{
    GTestJitGen cg;
    Label b;
    cg.Cmp(cg.x0, 5);
    cg.Bne(b);
    cg.Mov(cg.x0, 0);
    cg.L(b);
    cg.Ret();
    cg.Ready();

    auto f = cg.GetCode<int (*)(int)>();
    auto value = f(5);
    ASSERT_EQ(value, 0);
}

TEST(armcg_test_t, TestBmi)
{
    GTestJitGen cg;
    Label b;
    cg.Cmp(cg.x0, 5);
    cg.Bmi(b);
    cg.Mov(cg.x0, 0);
    cg.L(b);
    cg.Ret();
    cg.Ready();

    auto f = cg.GetCode<int (*)(int)>();
    auto value = f(-5);
    ASSERT_EQ(value, 0);
}

TEST(armcg_test_t, TestBfirst)
{
    GTestJitGen cg;
    Label b;
    cg.Cmp(cg.x0, 5);
    cg.BFirst(b);
    cg.Mov(cg.x0, 0);
    cg.L(b);
    cg.Ret();
    cg.Ready();

    auto f = cg.GetCode<int (*)(int)>();
    ASSERT_EQ(f(-5), 0);
}

TEST(armcg_test_t, TestCbnz1)
{
    GTestJitGen cg;
    Label b;
    cg.Cbnz(cg.x0, b);
    cg.Mov(cg.x0, 0);
    cg.L(b);
    cg.Ret();
    cg.Ready();

    auto f = cg.GetCode<int (*)(int)>();
    ASSERT_EQ(f(5), 5);
}

TEST(armcg_test_t, TestCbnz2)
{
    GTestJitGen cg;
    Label b;
    cg.Cbnz(cg.w0, b);
    cg.Mov(cg.w0, 0);
    cg.L(b);
    cg.Ret();
    cg.Ready();

    auto f = cg.GetCode<int (*)(int)>();
    ASSERT_EQ(f(5), 5);
}

TEST(armcg_test_t, TestCbz)
{
    GTestJitGen cg;
    Label b;
    cg.Cbz(cg.x0, b);
    cg.Mov(cg.x0, 0);
    cg.L(b);
    cg.Ret();
    cg.Ready();

    auto f = cg.GetCode<int (*)(int)>();
    ASSERT_EQ(f(5), 0);
}

TEST(armcg_test_t, TestTbz)
{
    GTestJitGen cg;
    Label b1, b2;
    cg.Tbz(cg.x0, 0x0, b1);
    cg.Tbz(cg.w0, 0x1, b2);
    cg.Add(cg.x0, cg.x0, 2);
    cg.L(b1);
    cg.Add(cg.x0, cg.x0, 2);
    cg.L(b2);
    cg.Add(cg.x0, cg.x0, 2);
    cg.Ret();
    cg.Ready();

    auto f = cg.GetCode<int (*)(int)>();
    ASSERT_EQ(f(3), 9);
    ASSERT_EQ(f(4), 8);
    ASSERT_EQ(f(5), 7);
}

TEST(armcg_test_t, TestTbnz)
{
    GTestJitGen cg;
    Label b1, b2;
    cg.Tbnz(cg.x0, 0x0, b1);
    cg.Tbnz(cg.w0, 0x1, b2);
    cg.Add(cg.x0, cg.x0, 2);
    cg.L(b1);
    cg.Add(cg.x0, cg.x0, 2);
    cg.L(b2);
    cg.Add(cg.x0, cg.x0, 2);
    cg.Ret();
    cg.Ready();

    auto f = cg.GetCode<int (*)(int)>();
    ASSERT_EQ(f(3), 7);
    ASSERT_EQ(f(4), 10);
    ASSERT_EQ(f(5), 9);
}

TEST(armcg_test_t, TestPrf)
{
    GTestJitGen cg;
    cg.Mov(cg.x9, 0);
    cg.Whilelt(cg.p0.s, cg.x9, cg.x0);
    cg.Prfh(PrfopVC::PLDL1STRM_VC, cg.p0, AdrNoOfs(cg.x1));
    cg.Nop();
    cg.Prfw(PrfopVC::PLDL1STRM_VC, cg.p0, AdrNoOfs(cg.x1));
    cg.Yield();
    cg.Prfw(PrfopVC::PLDL1STRM_VC, cg.p0, AdrScImm(cg.x1));
    cg.Dmb(BarOpt::LD);
    cg.Prfm(Prfop::PLDL1_STRM, AdrUimm(cg.x1, 0));
    cg.Ld1w(cg.z0.s, cg.p0, AdrNoOfs(cg.x1));
    cg.Ret();
    cg.Ready();

    float arr1[] = {0.0f, 9.0f, 25.0f, 0.046f, 40.0f, 20.0f, 0.0805929987f, 0.524f};
    auto f = cg.GetCode<int (*)(int, float*)>();
    ASSERT_EQ(f(8, arr1), 8);
}

//compare test
TEST(armcg_test_t, TestCmp1)
{
    GTestJitGen cg;
    cg.Cmp(cg.w0, 10);
    cg.MrsNZCV(cg.x0);
    cg.And(cg.x0, cg.x0, 0x80000000);
    cg.Ret();
    cg.Ready();

    auto f = cg.GetCode<int (*)(int)>();
    ASSERT_EQ(f(5), 0x80000000);
}

TEST(armcg_test_t, TestCmp2)
{
    GTestJitGen cg;
    cg.Cmp(cg.x0, 10);
    cg.MrsNZCV(cg.x0);
    cg.And(cg.x0, cg.x0, 0x80000000);
    cg.Ret();
    cg.Ready();

    auto f = cg.GetCode<int (*)(int)>();
    ASSERT_EQ(f(15), 0);
}

TEST(armcg_test_t, TestCmp3)
{
    GTestJitGen cg;
    cg.Cmp(cg.x0, cg.x1);
    cg.MrsNZCV(cg.x0);
    cg.And(cg.x0, cg.x0, 0x80000000);
    cg.Ret();
    cg.Ready();

    auto f = cg.GetCode<int (*)(int, int)>();
    ASSERT_EQ(f(15, 5), 0);
}

TEST(armcg_test_t, TestCmp4)
{
    GTestJitGen cg;
    Label b;
    b.Clear();
    cg.Cmp(cg.w0, cg.w1, ShMod::LSL, 2);
    cg.B(Cond::EQ, b);
    cg.Mov(cg.w0, 4);
    cg.L(b);
    cg.Ret();
    cg.Ready();

    auto f = cg.GetCode<int (*)(int, int)>();
    ASSERT_EQ(f(8, 2), 8);
}

TEST(armcg_test_t, TestBfmlalb1)
{
    GTestJitGen cg;
    cg.Mov(cg.x9, 0);
    cg.Mov(cg.x10, 8);
    cg.Whilelt(cg.p0.s, cg.x9, cg.x10);
    cg.Ld1w(cg.z0.s, cg.p0, AdrNoOfs(cg.x0));

    cg.Mov(cg.x11, 16);
    cg.Whilelt(cg.p1.h, cg.x9, cg.x11);
    cg.Ld1h(cg.z1.h, cg.p1, AdrNoOfs(cg.x1));
    cg.Ld1h(cg.z2.h, cg.p1, AdrNoOfs(cg.x1));

    cg.Bfmlalb(cg.z0.s, cg.z1.h, cg.z2.h);
    cg.St1w(cg.z0.s, cg.p0, AdrNoOfs(cg.x0));
    cg.Ret();
    cg.Ready();

    uint16_t arrBf16[] = {0x0, 0xc0c0, 0x3db2, 0xbd3e, 0x4220, 0xc1a0, 0x3da5, 0xbf06,
        0x42b1, 0x41b6, 0x4233, 0x0, 0x4120, 0xc040, 0x3da6};
    float arrF32[8] = {0.0f, -6.0f, 0.087116f, -0.046347f, 40.0f, -20.0f, 0.080593f, -0.524f};
    auto f = cg.GetCode<void (*)(float*, uint16_t*)>();
    f(arrF32, arrBf16);
    // f32 + bf16(even slot) * bf16(even slot)
    ASSERT_FLOAT_EQ(arrF32[0], 0.0f);
    ASSERT_FLOAT_EQ(arrF32[1], -5.9924459f);
    ASSERT_FLOAT_EQ(arrF32[2], 1600.0872f);
    ASSERT_FLOAT_EQ(arrF32[3], -0.039856054f);
    ASSERT_FLOAT_EQ(arrF32[4], 7872.25f);
    ASSERT_FLOAT_EQ(arrF32[5], 1982.5625f);
    ASSERT_FLOAT_EQ(arrF32[6], 100.08059f);
    ASSERT_FLOAT_EQ(arrF32[7], -0.51743013f);
}

TEST(armcg_test_t, TestBfmlalb2)
{
    GTestJitGen cg;
    cg.Mov(cg.x9, 0);
    cg.Mov(cg.x10, 8);
    cg.Whilelt(cg.p0.s, cg.x9, cg.x10);
    cg.Ld1w(cg.z0.s, cg.p0, AdrNoOfs(cg.x0));

    cg.Mov(cg.x11, 16);
    cg.Whilelt(cg.p1.h, cg.x9, cg.x11);
    cg.Ld1h(cg.z1.h, cg.p1, AdrNoOfs(cg.x1));
    cg.Ld1h(cg.z2.h, cg.p1, AdrNoOfs(cg.x1));

    cg.Bfmlalb(cg.z0.s, cg.z1.h, cg.z2.h[4]);
    cg.St1w(cg.z0.s, cg.p0, AdrNoOfs(cg.x0));
    cg.Ret();
    cg.Ready();

    uint16_t arrBf16[] = {0x0, 0xc0c0, 0x3db2, 0xbd3e, 0x4220, 0xc1a0, 0x3da5, 0xbf06,
        0x42b1, 0x41b6, 0x4233, 0x0, 0x4120, 0xc040, 0x3da6};
    float arrF32[8] = {0.0f, -6.0f, 0.087116f, -0.046347f, 40.0f, -20.0f, 0.080593f, -0.524f};
    auto f = cg.GetCode<void (*)(float*, uint16_t*)>();
    f(arrF32, arrBf16);
    // f32 + bf16(even slot) * bf16(0x4220 which is 40.0)
    ASSERT_FLOAT_EQ(arrF32[0], 0.0f);
    ASSERT_FLOAT_EQ(arrF32[1], -2.5234375f);
    ASSERT_FLOAT_EQ(arrF32[2], 1600.0872f);
    ASSERT_FLOAT_EQ(arrF32[3], 3.1763093f);
    ASSERT_FLOAT_EQ(arrF32[4], 925.0f);
    ASSERT_FLOAT_EQ(arrF32[5], 427.5f);
    ASSERT_FLOAT_EQ(arrF32[6], 100.08059f);
    ASSERT_FLOAT_EQ(arrF32[7], 0.28654689f);
}

TEST(armcg_test_t, TestBfdot)
{
    GTestJitGen cg;
    cg.Mov(cg.x9, 0);
    cg.Mov(cg.x10, 4);
    cg.Whilelt(cg.p0.h, cg.x9, cg.x10);
    cg.Ld1h(cg.z0.h, cg.p0.h, AdrNoOfs(cg.x0));
    cg.Ld1h(cg.z1.h, cg.p0.h, AdrNoOfs(cg.x1));
    cg.Fmov(cg.z2.s, 0.0f);
    cg.Bfdot(cg.z2.s, cg.z0.h, cg.z1.h);
    cg.Mov(cg.x10, 2);
    cg.Whilelt(cg.p1.s, cg.x9, cg.x10);
    cg.St1w(cg.z2.s, cg.p1, AdrNoOfs(cg.x2));

    cg.Ret();
    cg.Ready();

    auto f = cg.GetCode<void (*)(uint16_t*, uint16_t*, float*)>();
    uint16_t arr1[4] = {0x3f80, 0x4000, 0x3f80, 0x3f80}; // {1.0, 2.0, 1.0, 1.0}
    uint16_t arr2[4] = {0x3f80, 0x4000, 0x4000, 0x4000}; // {1.0, 2.0, 2.0, 2.0}
    float ret1[2] = {0};
    f(arr1, arr2, ret1);
    ASSERT_EQ(ret1[0], 5.0);   // 1.0 * 1.0 + 2.0 * 2.0
    ASSERT_EQ(ret1[1], 4.0);   // 1.0 * 2.0 + 1.0 * 2.0
}

TEST(armcg_test_t, TestBfcvtnt)
{
    GTestJitGen cg;
    cg.Ptrue(cg.p2.h);
    cg.Fmov(cg.z0.h, 0);
    cg.Fmov(cg.z1.h, 0);
    cg.Mov(cg.x9, 0);
    cg.Mov(cg.x10, 4);
    cg.Whilelt(cg.p0.h, cg.x9, cg.x10);
    cg.Ld1h(cg.z0.h, cg.p0.h, AdrNoOfs(cg.x0));
    cg.Ld1h(cg.z1.h, cg.p0.h, AdrNoOfs(cg.x1));
    cg.Uzp2(cg.z2.h, cg.z0.h, cg.z1.h);
    cg.St1h(cg.z2.h, cg.p2, AdrNoOfs(cg.x3));

    cg.Whilelt(cg.p1.s, cg.x9, cg.x10);
    cg.Ld1w(cg.z3.s, cg.p1.s, AdrNoOfs(cg.x2));
    cg.Bfcvtnt(cg.z0.h, cg.p0, cg.z3.s);
    cg.St1h(cg.z0.h, cg.p0, AdrNoOfs(cg.x4));

    cg.Ret();
    cg.Ready();

    auto f = cg.GetCode<uint64_t(*)(uint16_t*, uint16_t*, float*, uint16_t*, uint16_t*)>();
    uint16_t arr1[4] = {0x3f80, 0x3f80, 0x3f80, 0x3f80};
    uint16_t arr2[4] = {0x4000, 0x4000, 0x4000, 0x4000};
    float arr3[4] = {2.0f, 2.0f, 2.0f, 2.0f};
    uint16_t ret1[32] = {0};
    uint16_t ret2[4] = {0};
    f(arr1, arr2, arr3, ret1, ret2);
    ASSERT_EQ(ret1[0], 0x3f80);
#ifdef KUDNNL_920F
    ASSERT_EQ(ret1[16], 0x4000);   // ret1: 1.0 1.0 1.0 1.0 ... 2.0 2.0 2.0 2.0 ...
#else
    ASSERT_EQ(ret1[8], 0x4000);
#endif
    ASSERT_EQ(ret2[0], 0x3f80);   // ret2: 1.0 2.0 1.0 2.0
    ASSERT_EQ(ret2[1], 0x4000);
    ASSERT_EQ(ret2[2], 0x3f80);
    ASSERT_EQ(ret2[3], 0x4000);
}

TEST(armcg_test_t, TestLd1rqh1)
{
    GTestJitGen cg;

    cg.Mov(cg.x9, 0);
    cg.Mov(cg.x10, 16);
    cg.Whilelt(cg.p1.h, cg.x9, cg.x10);
    cg.Ld1rqh(cg.z1.h, cg.p1, AdrNoOfs(cg.x0));

    cg.St1h(cg.z1.h, cg.p1, AdrNoOfs(cg.x1));
    cg.Ret();
    cg.Ready();

    uint16_t arrBf16[] = {0x0, 0xc0c0, 0x3db2, 0xbd3e, 0x4220, 0xc1a0, 0x3da5, 0xbf06,
        0x42b1, 0x41b6, 0x4233, 0x0, 0x4120, 0xc040, 0x3da6};
    uint16_t arrOut[16];
    auto f = cg.GetCode<void (*)(uint16_t*, uint16_t*)>();
    f(arrBf16, arrOut);

    ASSERT_FLOAT_EQ(arrOut[0], arrOut[8]);
    ASSERT_FLOAT_EQ(arrOut[1], arrOut[9]);
    ASSERT_FLOAT_EQ(arrOut[2], arrOut[10]);
    ASSERT_FLOAT_EQ(arrOut[3], arrOut[11]);
    ASSERT_FLOAT_EQ(arrOut[4], arrOut[12]);
    ASSERT_FLOAT_EQ(arrOut[5], arrOut[13]);
    ASSERT_FLOAT_EQ(arrOut[6], arrOut[14]);
    ASSERT_FLOAT_EQ(arrOut[7], arrOut[15]);
}

TEST(armcg_test_t, TestCmpeq1)
{
    GTestJitGen cg;
    cg.Mov(cg.x9, 0);
    cg.Whilelt(cg.p0.s, cg.x9, cg.x2);
    cg.Ld1w(cg.z0.s, cg.p0, AdrNoOfs(cg.x0));
    cg.Ld1w(cg.z1.s, cg.p0, AdrNoOfs(cg.x1));
    cg.Cmpeq(cg.p1.s, cg.p0, cg.z0.s, cg.z1.s);
    cg.Str(cg.p1, AdrNoOfs(cg.x3));
    cg.Ret();
    cg.Ready();

    int arr1[] = {0, 9, 25, -46, 40, 20, 8059, 524};
    int arr2[] = {0, 100, -10, 816, -842, 88, 22, 44};
    int predicate = 0;
    auto f = cg.GetCode<void (*)(int*, int*, int, int*)>();
    f(arr1, arr2, 8, &predicate);
    ASSERT_EQ(predicate, 0x1);
}

TEST(armcg_test_t, TestCmpeq2)
{
    GTestJitGen cg;
    cg.Mov(cg.x9, 0);
    cg.Whilelt(cg.p0.s, cg.x9, cg.x1);
    cg.Ld1w(cg.z0.s, cg.p0, AdrNoOfs(cg.x0));
    cg.Cmpeq(cg.p1.s, cg.p0, cg.z0.s, 9);
    cg.Str(cg.p1, AdrNoOfs(cg.x2));
    cg.Ret();
    cg.Ready();

    int arr1[] = {0, 9, 25, -46, 40, 20, 8059, 524};
    int predicate = 0;
    auto f = cg.GetCode<void (*)(int*, int, int*)>();
    f(arr1, 8, &predicate);
    ASSERT_EQ(predicate, 0x10);
}

TEST(armcg_test_t, TestCmpls)
{
    GTestJitGen cg;
    cg.Mov(cg.x9, 0);
    cg.Whilelt(cg.p0.s, cg.x9, cg.x2);
    cg.Ld1w(cg.z0.s, cg.p0, AdrNoOfs(cg.x0));
    cg.Ld1w(cg.z1.s, cg.p0, AdrNoOfs(cg.x1));
    cg.Cmpls(cg.p1.b, cg.p0, cg.z0.b, cg.z1.b);
    cg.Str(cg.p1, AdrNoOfs(cg.x3));
    cg.Ret();
    cg.Ready();

    uint8_t arr1[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
        17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31};
    uint8_t arr2[] = {31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17,
        16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0};
    int predicate = 0;
    auto f = cg.GetCode<void (*)(uint8_t*, uint8_t*, int, int*)>();
    f(arr1, arr2, 8, &predicate);
    ASSERT_EQ(predicate, 0x1111);
}

TEST(armcg_test_t, TestCmplt1)
{
    GTestJitGen cg;
    cg.Mov(cg.x9, 0);
    cg.Whilelt(cg.p0.s, cg.x9, cg.x2);
    cg.Ld1w(cg.z0.s, cg.p0, AdrNoOfs(cg.x0));
    cg.Ld1w(cg.z1.s, cg.p0, AdrNoOfs(cg.x1));
    cg.Cmplt(cg.p1.b, cg.p0, cg.z0.b, cg.z1.b);
    cg.Str(cg.p1, AdrNoOfs(cg.x3));
    cg.Ret();
    cg.Ready();

    uint8_t arr1[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
        17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31};
    uint8_t arr2[] = {31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17,
        16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0};
    int predicate = 0;
    auto f = cg.GetCode<void (*)(uint8_t*, uint8_t*, int, int*)>();
    f(arr1, arr2, 8, &predicate);
    ASSERT_EQ(predicate, 0x1111);
}

TEST(armcg_test_t, TestCmplt2)
{
    GTestJitGen cg;
    cg.Mov(cg.x9, 0);
    cg.Whilelt(cg.p0.s, cg.x9, cg.x2);
    cg.Ld1w(cg.z0.s, cg.p0, AdrNoOfs(cg.x0));
    cg.Ld1w(cg.z1.s, cg.p0, AdrNoOfs(cg.x1));
    cg.Cmplt(cg.p1.s, cg.p0, cg.z0.s, cg.z1.s);
    cg.Str(cg.p1, AdrNoOfs(cg.x3));
    cg.Ret();
    cg.Ready();

    int arr1[] = {0, 9, 25, -46, 40, 20, 8059, 524};
    int arr2[] = {0, 100, -10, 816, -842, 88, 22, 44};
    int predicate = 0;
    auto f = cg.GetCode<void (*)(int*, int*, int, int*)>();
    f(arr1, arr2, 8, &predicate);
    ASSERT_EQ(predicate, 0x101010);
}

TEST(armcg_test_t, TestCmplt3)
{
    GTestJitGen cg;
    cg.Mov(cg.x9, 0);
    cg.Whilelt(cg.p0.s, cg.x9, cg.x1);
    cg.Ld1w(cg.z0.s, cg.p0, AdrNoOfs(cg.x0));
    cg.Cmplt(cg.p1.s, cg.p0, cg.z0.s, 9);
    cg.Str(cg.p1, AdrNoOfs(cg.x2));
    cg.Ret();
    cg.Ready();

    int arr1[] = {0, 9, 25, -46, 40, 20, 8059, 524};
    int predicate = 0;
    auto f = cg.GetCode<void (*)(int*, int, int*)>();
    f(arr1, 8, &predicate);
    ASSERT_EQ(predicate, 0x1001);
}

TEST(armcg_test_t, TestCmpne1)
{
    GTestJitGen cg;
    cg.Mov(cg.x9, 0);
    cg.Whilelt(cg.p0.s, cg.x9, cg.x1);
    cg.Ld1w(cg.z0.s, cg.p0, AdrNoOfs(cg.x0));
    cg.Cmpne(cg.p1.h, cg.p0, cg.z0.h, 0);
    cg.Str(cg.p1, AdrNoOfs(cg.x2));
    cg.Ret();
    cg.Ready();

    uint16_t arr1[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
    int predicate = 0;
    auto f = cg.GetCode<void (*)(uint16_t*, int, int*)>();
    f(arr1, 8, &predicate);
    ASSERT_EQ(predicate, 0x11111110);
}

TEST(armcg_test_t, TestCmpne2)
{
    GTestJitGen cg;
    cg.Mov(cg.x9, 0);
    cg.Whilelt(cg.p0.s, cg.x9, cg.x1);
    cg.Ld1w(cg.z0.s, cg.p0, AdrNoOfs(cg.x0));
    cg.Cmpne(cg.p1.s, cg.p0, cg.z0.s, 9);
    cg.Str(cg.p1, AdrNoOfs(cg.x2));
    cg.Ret();
    cg.Ready();

    int arr1[] = {0, 9, 25, -46, 40, 20, 8059, 524};
    int predicate = 0;
    auto f = cg.GetCode<void (*)(int*, int, int*)>();
    f(arr1, 8, &predicate);
    ASSERT_EQ(predicate, 0x11111101);
}

TEST(armcg_test_t, TestFacge)
{
    GTestJitGen cg;
    cg.Mov(cg.x9, 0);
    cg.Whilelt(cg.p0.s, cg.x9, cg.x2);
    cg.Ld1w(cg.z0.s, cg.p0, AdrNoOfs(cg.x0));
    cg.Ld1w(cg.z1.s, cg.p0, AdrNoOfs(cg.x1));
    cg.Facge(cg.p1.s, cg.p0, cg.z0.s, cg.z1.s);
    cg.Str(cg.p1, AdrNoOfs(cg.x3));
    cg.Ret();
    cg.Ready();

    float arr1[] = {0.0f, -4.4f, 25.0f, -46.79f, 40.0f, -20.0f, 8.059f, 0.524f};
    float arr2[] = {0.0f, 100.0f, -10.0f, 0.816f, -8.42f, 88.0f, 22.15f, 44.9f};
    int predicate = 0;
    auto f = cg.GetCode<void (*)(float*, float*, int, int*)>();
    f(arr1, arr2, 8, &predicate);
    ASSERT_EQ(predicate, 0x11101);
}

TEST(armcg_test_t, TestFcmeq1)
{
    GTestJitGen cg;
    cg.Mov(cg.x9, 0);
    cg.Whilelt(cg.p0.s, cg.x9, cg.x2);
    cg.Ld1w(cg.z0.s, cg.p0, AdrNoOfs(cg.x0));
    cg.Ld1w(cg.z1.s, cg.p0, AdrNoOfs(cg.x1));
    cg.Fcmeq(cg.p1.s, cg.p0, cg.z0.s, cg.z1.s);
    cg.Str(cg.p1, AdrNoOfs(cg.x3));
    cg.Ret();
    cg.Ready();

    float arr1[] = {0.0f, -4.4f, 25.0f, -46.79f, 40.0f, -20.0f, 8.059f, 0.524f};
    float arr2[] = {0.0f, 100.0f, -10.0f, 0.816f, -8.42f, 88.0f, 22.15f, 44.9f};
    int predicate = 0;
    auto f = cg.GetCode<void (*)(float*, float*, int, int*)>();
    f(arr1, arr2, 8, &predicate);
    ASSERT_EQ(predicate, 0x1);
}

TEST(armcg_test_t, TestFcmeq2)
{
    GTestJitGen cg;
    cg.Mov(cg.x9, 0);
    cg.Whilelt(cg.p0.s, cg.x9, cg.x1);
    cg.Ld1w(cg.z0.s, cg.p0, AdrNoOfs(cg.x0));
    cg.Fcmeq(cg.p1.s, cg.p0, cg.z0.s, 0.0);
    cg.Str(cg.p1, AdrNoOfs(cg.x2));
    cg.Ret();
    cg.Ready();

    float arr1[] = {0.0f, -4.4f, 25.0f, -46.79f, 40.0f, -20.0f, 8.059f, 0.524f};
    int predicate = 0;
    auto f = cg.GetCode<void (*)(float*, int, int*)>();
    f(arr1, 8, &predicate);
    ASSERT_EQ(predicate, 0x1);
}

TEST(armcg_test_t, TestFcmge1)
{
    GTestJitGen cg;
    cg.Mov(cg.x9, 0);
    cg.Whilelt(cg.p0.s, cg.x9, cg.x2);
    cg.Ld1w(cg.z0.s, cg.p0, AdrNoOfs(cg.x0));
    cg.Ld1w(cg.z1.s, cg.p0, AdrNoOfs(cg.x1));
    cg.Fcmge(cg.p1.s, cg.p0, cg.z0.s, cg.z1.s);
    cg.Str(cg.p1, AdrNoOfs(cg.x3));
    cg.Ret();
    cg.Ready();

    float arr1[] = {0.0f, -4.4f, 25.0f, -46.79f, 40.0f, -20.0f, 8.059f, 0.524f};
    float arr2[] = {0.0f, 100.0f, -10.0f, 0.816f, -8.42f, 88.0f, 22.15f, 44.9f};
    int predicate = 0;
    auto f = cg.GetCode<void (*)(float*, float*, int, int*)>();
    f(arr1, arr2, 8, &predicate);
    ASSERT_EQ(predicate, 0x10101);
}

TEST(armcg_test_t, TestFcmge2)
{
    GTestJitGen cg;
    cg.Mov(cg.x9, 0);
    cg.Whilelt(cg.p0.s, cg.x9, cg.x1);
    cg.Ld1w(cg.z0.s, cg.p0, AdrNoOfs(cg.x0));
    cg.Fcmge(cg.p1.s, cg.p0, cg.z0.s, 0.0);
    cg.Str(cg.p1, AdrNoOfs(cg.x2));
    cg.Ret();
    cg.Ready();

    float arr1[] = {0.0f, -4.4f, 25.0f, -46.79f, 40.0f, -20.0f, 8.059f, 0.524f};
    int predicate = 0;
    auto f = cg.GetCode<void (*)(float*, int, int*)>();
    f(arr1, 8, &predicate);
    ASSERT_EQ(predicate, 0x11010101);
}

TEST(armcg_test_t, TestFcmgt1)
{
    GTestJitGen cg;
    cg.Mov(cg.x9, 0);
    cg.Whilelt(cg.p0.s, cg.x9, cg.x2);
    cg.Ld1w(cg.z0.s, cg.p0, AdrNoOfs(cg.x0));
    cg.Ld1w(cg.z1.s, cg.p0, AdrNoOfs(cg.x1));
    cg.Fcmgt(cg.p1.s, cg.p0, cg.z0.s, cg.z1.s);
    cg.Str(cg.p1, AdrNoOfs(cg.x3));
    cg.Ret();
    cg.Ready();

    float arr1[] = {0.0f, -4.4f, 25.0f, -46.79f, 40.0f, -20.0f, 8.059f, 0.524f};
    float arr2[] = {0.0f, 100.0f, -10.0f, 0.816f, -8.42f, 88.0f, 22.15f, 44.9f};
    int predicate = 0;
    auto f = cg.GetCode<void (*)(float*, float*, int, int*)>();
    f(arr1, arr2, 8, &predicate);
    ASSERT_EQ(predicate, 0x10100);
}

TEST(armcg_test_t, TestFcmgt2)
{
    GTestJitGen cg;
    cg.Mov(cg.x9, 0);
    cg.Whilelt(cg.p0.s, cg.x9, cg.x1);
    cg.Ld1w(cg.z0.s, cg.p0, AdrNoOfs(cg.x0));
    cg.Fcmgt(cg.p1.s, cg.p0, cg.z0.s, 0.0);
    cg.Str(cg.p1, AdrNoOfs(cg.x2));
    cg.Ret();
    cg.Ready();

    float arr1[] = {0.0f, -4.4f, 25.0f, -46.79f, 40.0f, -20.0f, 8.059f, 0.524f};
    int predicate = 0;
    auto f = cg.GetCode<void (*)(float*, int, int*)>();
    f(arr1, 8, &predicate);
    ASSERT_EQ(predicate, 0x11010100);
}

TEST(armcg_test_t, TestFcmle1)
{
    GTestJitGen cg;
    cg.Mov(cg.x9, 0);
    cg.Whilelt(cg.p0.s, cg.x9, cg.x2);
    cg.Ld1w(cg.z0.s, cg.p0, AdrNoOfs(cg.x0));
    cg.Ld1w(cg.z1.s, cg.p0, AdrNoOfs(cg.x1));
    cg.Fcmle(cg.p1.s, cg.p0, cg.z0.s, cg.z1.s);
    cg.Str(cg.p1, AdrNoOfs(cg.x3));
    cg.Ret();
    cg.Ready();

    float arr1[] = {0.0f, -4.4f, 25.0f, -46.79f, 40.0f, -20.0f, 8.059f, 0.524f};
    float arr2[] = {0.0f, 100.0f, -10.0f, 0.816f, -8.42f, 88.0f, 22.15f, 44.9f};
    int predicate = 0;
    auto f = cg.GetCode<void (*)(float*, float*, int, int*)>();
    f(arr1, arr2, 8, &predicate);
    ASSERT_EQ(predicate, 0x11101011);
}

TEST(armcg_test_t, TestFcmle2)
{
    GTestJitGen cg;
    cg.Mov(cg.x9, 0);
    cg.Whilelt(cg.p0.s, cg.x9, cg.x1);
    cg.Ld1w(cg.z0.s, cg.p0, AdrNoOfs(cg.x0));
    cg.Fcmle(cg.p1.s, cg.p0, cg.z0.s, 0.0);
    cg.Str(cg.p1, AdrNoOfs(cg.x2));
    cg.Ret();
    cg.Ready();

    float arr1[] = {0.0f, -4.4f, 25.0f, -46.79f, 40.0f, -20.0f, 8.059f, 0.524f};
    int predicate = 0;
    auto f = cg.GetCode<void (*)(float*, int, int*)>();
    f(arr1, 8, &predicate);
    ASSERT_EQ(predicate, 0x101011);
}

TEST(armcg_test_t, TestFcmlt1)
{
    GTestJitGen cg;
    cg.Mov(cg.x9, 0);
    cg.Whilelt(cg.p0.s, cg.x9, cg.x2);
    cg.Ld1w(cg.z0.s, cg.p0, AdrNoOfs(cg.x0));
    cg.Ld1w(cg.z1.s, cg.p0, AdrNoOfs(cg.x1));
    cg.Fcmlt(cg.p1.s, cg.p0, cg.z0.s, cg.z1.s);
    cg.Str(cg.p1, AdrNoOfs(cg.x3));
    cg.Ret();
    cg.Ready();

    float arr1[] = {0.0f, -4.4f, 25.0f, -46.79f, 40.0f, -20.0f, 8.059f, 0.524f};
    float arr2[] = {0.0f, 100.0f, -10.0f, 0.816f, -8.42f, 88.0f, 22.15f, 44.9f};
    int predicate = 0;
    auto f = cg.GetCode<void (*)(float*, float*, int, int*)>();
    f(arr1, arr2, 8, &predicate);
    ASSERT_EQ(predicate, 0x11101010);
}

TEST(armcg_test_t, TestFcmlt2)
{
    GTestJitGen cg;
    cg.Mov(cg.x9, 0);
    cg.Whilelt(cg.p0.s, cg.x9, cg.x1);
    cg.Ld1w(cg.z0.s, cg.p0, AdrNoOfs(cg.x0));
    cg.Fcmlt(cg.p1.s, cg.p0, cg.z0.s, 0.0);
    cg.Str(cg.p1, AdrNoOfs(cg.x2));
    cg.Ret();
    cg.Ready();

    float arr1[] = {0.0f, -4.4f, 25.0f, -46.79f, 40.0f, -20.0f, 8.059f, 0.524f};
    int predicate = 0;
    auto f = cg.GetCode<void (*)(float*, int, int*)>();
    f(arr1, 8, &predicate);
    ASSERT_EQ(predicate, 0x101010);
}

TEST(armcg_test_t, TestFcmne)
{
    GTestJitGen cg;
    cg.Mov(cg.x9, 0);
    cg.Whilelt(cg.p0.s, cg.x9, cg.x2);
    cg.Ld1w(cg.z0.s, cg.p0, AdrNoOfs(cg.x0));
    cg.Ld1w(cg.z1.s, cg.p0, AdrNoOfs(cg.x1));
    cg.Fcmne(cg.p1.s, cg.p0, cg.z0.s, cg.z1.s);
    cg.Str(cg.p1, AdrNoOfs(cg.x3));
    cg.Ret();
    cg.Ready();

    float arr1[] = {0.0f, -4.4f, 25.0f, -46.79f, 40.0f, -20.0f, 8.059f, 0.524f};
    float arr2[] = {0.0f, 100.0f, -10.0f, 0.816f, -8.42f, 88.0f, 22.15f, 44.9f};
    int predicate = 0;
    auto f = cg.GetCode<void (*)(float*, float*, int, int*)>();
    f(arr1, arr2, 8, &predicate);
    ASSERT_EQ(predicate, 0x11111110);
}

TEST(armcg_test_t, TestFcmp)
{
    GTestJitGen cg1;
    Label b1, b2, b3, b4;
    cg1.Mov(cg1.x0, 1);
    cg1.Fmov(cg1.h1, 1.0);
    cg1.Fmov(cg1.h2, 1.0);
    cg1.Fcmp(cg1.h1, cg1.h2);
    cg1.B(Cond::EQ, b1);
    cg1.Mov(cg1.x0, 2);
    cg1.L(b1);
    cg1.Fcmp(cg1.h1, 0.0);
    cg1.B(Cond::NE, b2);
    cg1.Mov(cg1.x0, 3);
    cg1.L(b2);
    cg1.Fcmpe(cg1.h1, cg1.h2);
    cg1.B(Cond::EQ, b3);
    cg1.Mov(cg1.x0, 4);
    cg1.L(b3);
    cg1.Fcmpe(cg1.h1, 0.0);
    cg1.B(Cond::NE, b4);
    cg1.Mov(cg1.x0, 5);
    cg1.L(b4);

    cg1.Ret();
    cg1.Ready();
    auto f1 = cg1.GetCode<int (*)()>();
    auto value1 = f1();
    ASSERT_EQ(value1, 1);

    GTestJitGen cg2;
    Label b5, b6, b7, b8;
    cg2.Mov(cg2.x0, 1);
    cg2.Fmov(cg2.s1, 1.0);
    cg2.Fmov(cg2.s2, 1.0);
    cg2.Fcmp(cg2.s1, cg2.s2);
    cg2.B(Cond::EQ, b5);
    cg2.Mov(cg2.x0, 2);
    cg2.L(b5);
    cg2.Fcmp(cg2.s1, 0.0);
    cg2.B(Cond::NE, b6);
    cg2.Mov(cg2.x0, 3);
    cg2.L(b6);
    cg2.Fcmpe(cg2.s1, cg2.s2);
    cg2.B(Cond::EQ, b7);
    cg2.Mov(cg2.x0, 4);
    cg2.L(b7);
    cg2.Fcmpe(cg2.s1, 0.0);
    cg2.B(Cond::NE, b8);
    cg2.Mov(cg2.x0, 5);
    cg2.L(b8);

    cg2.Ret();
    cg2.Ready();
    auto f2 = cg2.GetCode<int (*)()>();
    auto value2 = f2();
    ASSERT_EQ(value2, 1);

    GTestJitGen cg3;
    Label b9, b10, b11, b12;
    cg3.Mov(cg3.x0, 1);
    cg3.Fmov(cg3.d1, 1.0);
    cg3.Fmov(cg3.d2, 1.0);
    cg3.Fcmp(cg3.d1, cg3.d2);
    cg3.B(Cond::EQ, b9);
    cg3.Mov(cg3.x0, 2);
    cg3.L(b9);
    cg3.Fcmp(cg3.d1, 0.0);
    cg3.B(Cond::NE, b10);
    cg3.Mov(cg3.x0, 3);
    cg3.L(b2);
    cg3.Fcmpe(cg3.d1, cg3.d2);
    cg3.B(Cond::EQ, b11);
    cg3.Mov(cg3.x0, 4);
    cg3.L(b3);
    cg3.Fcmpe(cg3.d1, 0.0);
    cg3.B(Cond::NE, b12);
    cg3.Mov(cg3.x0, 5);
    cg3.L(b4);

    cg3.Ret();
    cg3.Ready();
    auto f3 = cg3.GetCode<int (*)()>();
    auto value3 = f3();
    ASSERT_EQ(value3, 1);
}

//convert test
TEST(armcg_test_t, TestBfcvt)
{
    GTestJitGen cg;
    cg.Mov(cg.x9, 0);
    cg.Whilelt(cg.p0.s, cg.x9, cg.x1);
    cg.Ld1w(cg.z0.s, cg.p0, AdrNoOfs(cg.x0));
    cg.Bfcvt(cg.z1.h, cg.p0, cg.z0.s);
    cg.St1w(cg.z1.s, cg.p0, AdrNoOfs(cg.x2));
    cg.Ret();
    cg.Ready();

    auto f = cg.GetCode<void (*)(float*, int, uint16_t*)>();
    float arrF32[8] = {0.0f, -6.0f, 0.087116f, -0.046347f, 40.0f, -20.0f, 0.080593f, -0.524f};
    uint16_t arrBf16[16];
    f(arrF32, 8, arrBf16);
    // bfcvt will put converted results into even slots of zreg
    ASSERT_EQ(arrBf16[0], 0x0);
    ASSERT_EQ(arrBf16[2], 0xc0c0);
    ASSERT_EQ(arrBf16[4], 0x3db2);
    ASSERT_EQ(arrBf16[6], 0xbd3e);
    ASSERT_EQ(arrBf16[8], 0x4220);
    ASSERT_EQ(arrBf16[10], 0xc1a0);
    ASSERT_EQ(arrBf16[12], 0x3da5);
    ASSERT_EQ(arrBf16[14], 0xbf06);
}

TEST(armcg_test_t, TestBfcvtn)
{
    GTestJitGen cg;
    cg.Dup(cg.v1.s4, cg.w0);
    cg.Bfcvtn(cg.v1.h4, cg.v1.s4);
    cg.Mov(cg.w0, cg.v1.s4[0]);
    cg.Ret();
    cg.Ready();

    auto f = cg.GetCode<int (*)(int)>();
    int f87 = 0x42AE0000;
    ASSERT_EQ(f(f87), 0x42AE42AE);      // this is actually two 87.0f
}

// f32 to f16
TEST(armcg_test_t, TestFcvt1)
{
    GTestJitGen cg;
    cg.Mov(cg.x9, 0);
    cg.Whilelt(cg.p0.s, cg.x9, cg.x1);
    cg.Ld1w(cg.z0.s, cg.p0, AdrNoOfs(cg.x0));
    cg.Fcvt(cg.z1.h, cg.p0, cg.z0.s);
    cg.St1w(cg.z1.s, cg.p0, AdrNoOfs(cg.x2));
    cg.Ret();
    cg.Ready();

    auto f = cg.GetCode<void (*)(float*, int, uint16_t*)>();
    float arrF32[8] = {0.0f, -6.0f, 0.087116f, -0.046347f, 40.0f, -20.0f, 0.080593f, -0.524f};
    uint16_t arrF16[16];
    f(arrF32, 8, arrF16);
    // fcvt will put converted results into even slots of zreg
    ASSERT_EQ(arrF16[0], 0x0);
    ASSERT_EQ(arrF16[2], 0xc600);
    ASSERT_EQ(arrF16[4], 0x2d93);
    ASSERT_EQ(arrF16[6], 0xa9ef);
    ASSERT_EQ(arrF16[8], 0x5100);
    ASSERT_EQ(arrF16[10], 0xcd00);
    ASSERT_EQ(arrF16[12], 0x2d28);
    ASSERT_EQ(arrF16[14], 0xb831);
}

// f16 to f32
TEST(armcg_test_t, TestFcvt2)
{
    GTestJitGen cg;
    cg.Mov(cg.x9, 0);
    cg.Whilelt(cg.p0.s, cg.x9, cg.x1);
    cg.Ld1w(cg.z0.s, cg.p0, AdrNoOfs(cg.x0));
    cg.Fcvt(cg.z1.s, cg.p0, cg.z0.h);
    cg.St1w(cg.z1.s, cg.p0, AdrNoOfs(cg.x2));
    cg.Ret();
    cg.Ready();

    auto f = cg.GetCode<void (*)(uint16_t*, int, float*)>();
    uint16_t arrF16[] = {0x0, 0x0, 0xc600, 0x0, 0x2d93, 0x0, 0xa9ef, 0x0,
        0x5100, 0x0, 0xcd00, 0x0, 0x2d28, 0x0, 0xb831, 0x0};
    float arrF32[8];
    f(arrF16, 8, arrF32);

    ASSERT_FLOAT_EQ(arrF32[0], 0.0f);
    ASSERT_FLOAT_EQ(arrF32[1], -6.0f);
    ASSERT_FLOAT_EQ(arrF32[2], 0.087097168f);
    ASSERT_FLOAT_EQ(arrF32[3], -0.046356201f);
    ASSERT_FLOAT_EQ(arrF32[4], 40.0f);
    ASSERT_FLOAT_EQ(arrF32[5], -20.0f);
    ASSERT_FLOAT_EQ(arrF32[6], 0.080566406f);
    ASSERT_FLOAT_EQ(arrF32[7], -0.52392578f);
}

TEST(armcg_test_t, TestMix1)
{
    GTestJitGen cg;
    cg.Fmov(cg.z0.h, 0.0f);
    cg.Fmov(cg.z1.h, 0.0f);
    cg.Fmov(cg.z2.h, 0.0f);
    cg.Ldr(cg.s0, AdrUimm(cg.x0, 0));
    cg.Fcvt(cg.h1, cg.s0);
    // broadcast scalar value
    cg.Mov(cg.z0.h, cg.h1);
    cg.Mov(cg.x9, 0);
    cg.Mov(cg.x10, 4);
    cg.Whilelt(cg.p0.h, cg.x9, cg.x10);
    cg.Add(cg.x11, cg.x1, 8);
    cg.St1h(cg.z0.h, cg.p0, AdrNoOfs(cg.x11));
    // offset 8 bytes
    cg.Ld1rh(cg.z1.h, cg.p0, AdrUimm(cg.x1, 8));
    cg.St1h(cg.z1.h, cg.p0, AdrNoOfs(cg.x1));
    cg.Whilelt(cg.p1.s, cg.x9, cg.x10);
    cg.Ld1rw(cg.z2.s, cg.p1, AdrUimm(cg.x2, 16));
    cg.St1w(cg.z2.s, cg.p1, AdrNoOfs(cg.x2));
    cg.Ret();
    cg.Ready();

    auto f = cg.GetCode<void (*)(float*, uint16_t*, float*)>();
    float arr1[] = {1.0};
    uint16_t arr2[8] = {0};
    float arr3[] = {0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 2.0f, 3.0f, 4.0f};
    f(arr1, arr2, arr3);
    ASSERT_EQ(arr2[0], 0x3c00);
    ASSERT_EQ(arr2[1], 0x3c00);
    ASSERT_EQ(arr2[2], 0x3c00);
    ASSERT_EQ(arr2[3], 0x3c00);
    ASSERT_EQ(arr2[4], 0x3c00);
    ASSERT_EQ(arr2[5], 0x3c00);
    ASSERT_EQ(arr2[6], 0x3c00);
    ASSERT_EQ(arr2[7], 0x3c00);
    ASSERT_EQ(arr3[0], 1.0f);
    ASSERT_EQ(arr3[1], 1.0f);
    ASSERT_EQ(arr3[2], 1.0f);
    ASSERT_EQ(arr3[3], 1.0f);
}

TEST(armcg_test_t, TestFcvtns)
{
    GTestJitGen cg;
    cg.Dup(cg.v1.s4, cg.w0);
    cg.Fcvtns(cg.v2.s4, cg.v1.s4);
    cg.Mov(cg.w0, cg.v2.s4[0]);
    cg.Ret();
    cg.Ready();

    auto f = cg.GetCode<int (*)(int)>();
    int f87 = 0x42AE0000;
    ASSERT_EQ(f(f87), 87);
}

TEST(armcg_test_t, TestFcvtzs1)
{
    GTestJitGen cg;
    cg.Dup(cg.v1.s4, cg.w0);
    cg.Fcvtzs(cg.v2.s4, cg.v1.s4);
    cg.Mov(cg.w0, cg.v2.s4[0]);
    cg.Ret();
    cg.Ready();

    auto f = cg.GetCode<int (*)(int)>();
    int f87 = 0x42AE0000;
    ASSERT_EQ(f(f87), 87);
}

TEST(armcg_test_t, TestFcvtzs2)
{
    GTestJitGen cg;
    cg.Mov(cg.x9, 0);
    cg.Whilelt(cg.p0.s, cg.x9, cg.x1);
    cg.Ld1w(cg.z0.s, cg.p0, AdrNoOfs(cg.x0));
    cg.Fcvtzs(cg.z1.s, cg.p0, cg.z0.s);
    cg.St1w(cg.z1.s, cg.p0, AdrNoOfs(cg.x2));
    cg.Ret();
    cg.Ready();

    auto f = cg.GetCode<void (*)(float*, int, int*)>();
    float arrF32[8] = {0.0f, -6.0f, 0.087116f, -0.046347f, 40.0f, -20.0f, 0.080593f, -0.524f};
    int arrS32[8];
    f(arrF32, 8, arrS32);

    ASSERT_EQ(arrS32[0], 0);
    ASSERT_EQ(arrS32[1], -6);
    ASSERT_EQ(arrS32[2], 0);
    ASSERT_EQ(arrS32[3], 0);
    ASSERT_EQ(arrS32[4], 40);
    ASSERT_EQ(arrS32[5], -20);
    ASSERT_EQ(arrS32[6], 0);
    ASSERT_EQ(arrS32[7], 0);
}

TEST(armcg_test_t, TestFcvtzu)
{
    GTestJitGen cg;
    cg.Mov(cg.x9, 0);
    cg.Whilelt(cg.p0.s, cg.x9, cg.x1);
    cg.Ld1w(cg.z0.s, cg.p0, AdrNoOfs(cg.x0));
    cg.Fcvtzu(cg.z1.s, cg.p0, cg.z0.s);
    cg.St1w(cg.z1.s, cg.p0, AdrNoOfs(cg.x2));
    cg.Ret();
    cg.Ready();

    auto f = cg.GetCode<void (*)(float*, int, unsigned*)>();
    float arrF32[8] = {0.0f, -6.0f, 0.087116f, -0.046347f, 40.0f, -20.0f, 0.080593f, -0.524f};
    unsigned arrU32[8];
    f(arrF32, 8, arrU32);
    // fcvtzu will turn negative integer to zero
    ASSERT_EQ(arrU32[0], 0);
    ASSERT_EQ(arrU32[1], 0);
    ASSERT_EQ(arrU32[2], 0);
    ASSERT_EQ(arrU32[3], 0);
    ASSERT_EQ(arrU32[4], 40);
    ASSERT_EQ(arrU32[5], 0);
    ASSERT_EQ(arrU32[6], 0);
    ASSERT_EQ(arrU32[7], 0);
}

TEST(armcg_test_t, TestScvtf1)
{
    GTestJitGen cg;
    cg.Mov(cg.x9, 0);
    cg.Whilelt(cg.p0.s, cg.x9, cg.x1);
    cg.Ld1w(cg.z0.s, cg.p0, AdrNoOfs(cg.x0));
    cg.Scvtf(cg.z1.s, cg.p0, cg.z0.s);
    cg.St1w(cg.z1.s, cg.p0, AdrNoOfs(cg.x2));
    cg.Ret();
    cg.Ready();

    auto f = cg.GetCode<void (*)(int*, int, float*)>();
    int arrS32[] = {0, -6, 0, 9, 40, -20, -68, 1314};
    float arrF32[8];
    f(arrS32, 8, arrF32);

    ASSERT_FLOAT_EQ(arrF32[0], 0.0f);
    ASSERT_FLOAT_EQ(arrF32[1], -6.0f);
    ASSERT_FLOAT_EQ(arrF32[2], 0.0f);
    ASSERT_FLOAT_EQ(arrF32[3], 9.0f);
    ASSERT_FLOAT_EQ(arrF32[4], 40.0f);
    ASSERT_FLOAT_EQ(arrF32[5], -20.0f);
    ASSERT_FLOAT_EQ(arrF32[6], -68.0f);
    ASSERT_FLOAT_EQ(arrF32[7], 1314.0f);
}

TEST(armcg_test_t, TestScvtf2)
{
    GTestJitGen cg;
    cg.Dup(cg.v1.s4, cg.w0);
    cg.Scvtf(cg.v2.s4, cg.v1.s4);
    cg.Mov(cg.w0, cg.v2.s4[0]);
    cg.Ret();
    cg.Ready();

    auto f = cg.GetCode<int (*)(int)>();
    int f87 = 0x42AE0000;
    ASSERT_EQ(f(87), f87);
}

TEST(armcg_test_t, TestScvtf3)
{
    GTestJitGen cg;
    cg.Dup(cg.v1.s4, cg.w0);
    cg.Scvtf(cg.v2.s4, cg.v1.s4, 2);
    cg.Mov(cg.w0, cg.v2.s4[0]);
    cg.Ret();
    cg.Ready();

    auto f = cg.GetCode<int (*)(int)>();
    unsigned f21p75 = 0x41AE0000;
    ASSERT_EQ(f(87), f21p75);       // 87 / 4 = 21.75
}

TEST(armcg_test_t, TestShr)
{
    GTestJitGen cg;
    cg.Ld1(VReg16BList(cg.v1.b16), AdrNoOfs(cg.x0));
    cg.Sshr(cg.v2.b16, cg.v1.b16, 1);
    cg.St1(VReg16BList(cg.v2.b16), AdrNoOfs(cg.x2));
    cg.Ld1(VReg4SList(cg.v3.s4), AdrNoOfs(cg.x1));
    cg.Sshr(cg.v4.s4, cg.v3.s4, 1);
    cg.St1(VReg4SList(cg.v4.s4), AdrNoOfs(cg.x3));
    cg.Ret();
    cg.Ready();

    auto f = cg.GetCode<void (*)(int8_t *, int32_t *, int8_t *, int32_t *)>();
    int8_t arr1[16] = {1, 2, 4, 8, 1, 2, 4, 8, 1, 2, 4, 8, 1, 2, 4, 8};
    int32_t arr2[4] = {1, 2, 4, 8};
    int8_t ret1[16] = {0};
    int32_t ret2[8] = {0};
    f(arr1, arr2, ret1, ret2);
    ASSERT_EQ(ret1[0], 0);
    ASSERT_EQ(ret1[1], 1);
    ASSERT_EQ(ret1[2], 2);
    ASSERT_EQ(ret1[3], 4);
    ASSERT_EQ(ret2[0], 0);
    ASSERT_EQ(ret2[1], 1);
    ASSERT_EQ(ret2[2], 2);
    ASSERT_EQ(ret2[3], 4);

    GTestJitGen cg2;
    cg2.Ld1(VReg16BList(cg2.v1.b16), AdrNoOfs(cg2.x0));
    cg2.Ushr(cg2.v2.b16, cg2.v1.b16, 1);
    cg2.St1(VReg16BList(cg2.v2.b16), AdrNoOfs(cg2.x2));
    cg2.Ld1(VReg4SList(cg2.v3.s4), AdrNoOfs(cg2.x1));
    cg2.Ushr(cg2.v4.s4, cg2.v3.s4, 1);
    cg2.St1(VReg4SList(cg2.v4.s4), AdrNoOfs(cg2.x3));
    cg2.Ret();
    cg2.Ready();

    auto f2 = cg2.GetCode<void (*)(uint8_t *, uint32_t *, uint8_t *, uint32_t *)>();
    uint8_t arr3[16] = {5, 6, 7, 8, 5, 6, 7, 8, 5, 6, 7, 8, 5, 6, 7, 8};
    uint32_t arr4[4] = {5, 6, 7, 8};
    uint8_t ret3[16] = {0};
    uint32_t ret4[8] = {0};
    f2(arr3, arr4, ret3, ret4);
    ASSERT_EQ(ret3[0], 2);
    ASSERT_EQ(ret3[1], 3);
    ASSERT_EQ(ret3[2], 3);
    ASSERT_EQ(ret3[3], 4);
    ASSERT_EQ(ret4[0], 2);
    ASSERT_EQ(ret4[1], 3);
    ASSERT_EQ(ret4[2], 3);
    ASSERT_EQ(ret4[3], 4);
}

TEST(armcg_test_t, TestShl)
{
    GTestJitGen cg;
    cg.Ld1(VReg16BList(cg.v1.b16), AdrNoOfs(cg.x0));
    cg.Shl(cg.v2.b16, cg.v1.b16, 1);
    cg.St1(VReg16BList(cg.v2.b16), AdrNoOfs(cg.x2));
    cg.Ld1(VReg4SList(cg.v3.s4), AdrNoOfs(cg.x1));
    cg.Shl(cg.v4.s4, cg.v3.s4, 1);
    cg.St1(VReg4SList(cg.v4.s4), AdrNoOfs(cg.x3));
    cg.Ret();
    cg.Ready();

    auto f = cg.GetCode<void (*)(int8_t *, int32_t *, int8_t *, int32_t *)>();
    int8_t arr1[16] = {1, 2, 4, 8, 1, 2, 4, 8, 1, 2, 4, 8, 1, 2, 4, 8};
    int32_t arr2[4] = {1, 2, 4, 8};
    int8_t ret1[16] = {0};
    int32_t ret2[4] = {0};
    f(arr1, arr2, ret1, ret2);
    ASSERT_EQ(ret1[0], 2);
    ASSERT_EQ(ret1[1], 4);
    ASSERT_EQ(ret1[2], 8);
    ASSERT_EQ(ret1[3], 16);
    ASSERT_EQ(ret2[0], 2);
    ASSERT_EQ(ret2[1], 4);
    ASSERT_EQ(ret2[2], 8);
    ASSERT_EQ(ret2[3], 16);

    GTestJitGen cg2;
    cg2.Mov(cg2.w9, 1);
    cg2.Dup(cg2.v5.b16, cg2.w9);
    cg2.Ld1(VReg16BList(cg2.v1.b16), AdrNoOfs(cg2.x0));
    cg2.Sshl(cg2.v2.b16, cg2.v1.b16, cg2.v5.b16);
    cg2.St1(VReg16BList(cg2.v2.b16), AdrNoOfs(cg2.x2));
    cg2.Dup(cg2.v6.s4, cg2.w9);
    cg2.Ld1(VReg4SList(cg2.v3.s4), AdrNoOfs(cg2.x1));
    cg2.Sshl(cg2.v4.s4, cg2.v3.s4, cg2.v6.s4);
    cg2.St1(VReg4SList(cg2.v4.s4), AdrNoOfs(cg2.x3));
    cg2.Ret();
    cg2.Ready();

    auto f2 = cg2.GetCode<void (*)(int8_t *, int32_t *, int8_t *, int32_t *)>();
    int8_t arr3[16] = {1, 2, 4, 8, 1, 2, 4, 8, 1, 2, 4, 8, 1, 2, 4, 8};
    int32_t arr4[4] = {1, 2, 4, 8};
    int8_t ret3[16] = {0};
    int32_t ret4[4] = {0};
    f2(arr3, arr4, ret3, ret4);
    ASSERT_EQ(ret3[0], 2);
    ASSERT_EQ(ret3[1], 4);
    ASSERT_EQ(ret3[2], 8);
    ASSERT_EQ(ret3[3], 16);
    ASSERT_EQ(ret4[0], 2);
    ASSERT_EQ(ret4[1], 4);
    ASSERT_EQ(ret4[2], 8);
    ASSERT_EQ(ret4[3], 16);

    GTestJitGen cg3;
    cg3.Mov(cg3.w9, 1);
    cg3.Dup(cg3.v5.b16, cg3.w9);
    cg3.Ld1(VReg16BList(cg3.v1.b16), AdrNoOfs(cg3.x0));
    cg3.Ushl(cg3.v2.b16, cg3.v1.b16, cg2.v5.b16);
    cg3.St1(VReg16BList(cg3.v2.b16), AdrNoOfs(cg3.x2));
    cg3.Dup(cg3.v6.s4, cg3.w9);
    cg3.Ld1(VReg4SList(cg3.v3.s4), AdrNoOfs(cg3.x1));
    cg3.Ushl(cg3.v4.s4, cg3.v3.s4, cg3.v6.s4);
    cg3.St1(VReg4SList(cg3.v4.s4), AdrNoOfs(cg3.x3));
    cg3.Ret();
    cg3.Ready();

    auto f3 = cg3.GetCode<void (*)(uint8_t *, uint32_t *, uint8_t *, uint32_t *)>();
    uint8_t arr5[16] = {1, 2, 4, 8, 1, 2, 4, 8, 1, 2, 4, 8, 1, 2, 4, 8};
    uint32_t arr6[4] = {1, 2, 4, 8};
    uint8_t ret5[16] = {0};
    uint32_t ret6[4] = {0};
    f3(arr5, arr6, ret5, ret6);
    ASSERT_EQ(ret5[0], 2);
    ASSERT_EQ(ret5[1], 4);
    ASSERT_EQ(ret5[2], 8);
    ASSERT_EQ(ret5[3], 16);
    ASSERT_EQ(ret6[0], 2);
    ASSERT_EQ(ret6[1], 4);
    ASSERT_EQ(ret6[2], 8);
    ASSERT_EQ(ret6[3], 16);
}

TEST(armcg_test_t, TestBif)
{
    GTestJitGen cg;
    cg.Movi(cg.v1.b16, 0);
    cg.Movi(cg.v2.b16, 0);
    cg.Movi(cg.v3.b16, 0);
    cg.Dup(cg.v1.b16, cg.w0);
    cg.Dup(cg.v2.b16, cg.w1);
    cg.Dup(cg.v3.b16, cg.w2);
    cg.Bif(cg.v3.b16, cg.v2.b16, cg.v1.b16);
    cg.Mov(cg.w0, cg.v3.s4[0]);
    cg.Ret();
    cg.Ready();

    auto f = cg.GetCode<int (*)(int, int, int)>();
    ASSERT_EQ(f(0xAA, 0xAA, 0xFF), 0xAAAAAAAA);
}

TEST(armcg_test_t, TestBit)
{
    GTestJitGen cg;
    cg.Movi(cg.v1.b16, 0);
    cg.Movi(cg.v2.b16, 0);
    cg.Movi(cg.v3.b16, 0);
    cg.Dup(cg.v1.b16, cg.w0);
    cg.Dup(cg.v2.b16, cg.w1);
    cg.Dup(cg.v3.b16, cg.w2);
    cg.Bit(cg.v3.b16, cg.v2.b16, cg.v1.b16);
    cg.Mov(cg.w0, cg.v3.s4[0]);
    cg.Ret();
    cg.Ready();

    auto f = cg.GetCode<int (*)(int, int, int)>();
    ASSERT_EQ(f(0xF0, 0xAA, 0xCC), 0xACACACAC);
}

TEST(armcg_test_t, TestMvnNeon)
{
    GTestJitGen cg;
    cg.Movi(cg.v1.b16, 0);
    cg.Movi(cg.v2.b16, 0);
    cg.Dup(cg.v1.b16, cg.w0);
    cg.Mvn(cg.v2.b16, cg.v1.b16);
    cg.Mov(cg.w0, cg.v2.s4[0]);
    cg.Ret();
    cg.Ready();

    auto f = cg.GetCode<int (*)(int)>();
    ASSERT_EQ(f(0xAA), 0x55555555);
}

//reorder test
TEST(armcg_test_t, TestExt)
{
    GTestJitGen cg;
    cg.Ldr(cg.w9, AdrUimm(cg.x0, 0));
    cg.Ldr(cg.w10, AdrUimm(cg.x0, 4));
    cg.Dup(cg.v1.s4, cg.w9);
    cg.Dup(cg.v2.s4, cg.w10);

    cg.Dup(cg.z1.s, cg.w9);
    cg.Ext(cg.z0.b, cg.z1.b, 4);

    cg.Ext(cg.v0.b16, cg.v1.b16, cg.v2.b16, 4);
    cg.Addv(cg.s3, cg.v0.s4);
    cg.Mov(cg.w0, cg.v3.s4[0]);
    cg.Ret();
    cg.Ready();

    auto f = cg.GetCode<uint64_t(*)(uint32_t*)>();
    uint32_t arr[] = {9, 8};
    ASSERT_EQ(f(arr), 35);
}

TEST(armcg_test_t, TestIns)
{
    GTestJitGen cg;
    cg.Ldr(cg.w1, AdrUimm(cg.x0, 0));
    cg.Mov(cg.w2, 0);

    cg.Dup(cg.v1.s4, cg.w2);
    cg.Dup(cg.v2.s4, cg.w2);
    cg.Dup(cg.v3.s4, cg.w2);

    cg.Ins(cg.v1.b16[5], cg.w1);
    cg.Ins(cg.v2.s4[0], cg.v1.s4[1]);
    cg.Ins(cg.v3.d1[0], cg.v2.d1[0]);

    cg.Addv(cg.s3, cg.v3.s4);
    cg.Mov(cg.w0, cg.v3.s4[0]);
    cg.Ret();
    cg.Ready();

    auto f = cg.GetCode<uint64_t(*)(uint32_t*)>();
    uint32_t arr[] = {1, 2, 3, 4};
    ASSERT_EQ(f(arr), 256);
}

TEST(armcg_test_t, TestSel)
{
    GTestJitGen cg;
    cg.Ldr(cg.p0, AdrNoOfs(cg.x0));
    cg.Ldr(cg.w9, AdrUimm(cg.x0, 4));
    cg.Ldr(cg.w10, AdrUimm(cg.x0, 8));

    cg.Dup(cg.z1.s, cg.w9);
    cg.Dup(cg.z2.s, cg.w10);
    cg.Sel(cg.z3.s, cg.p0, cg.z1.s, cg.z2.s);
    cg.Sel(cg.z4.b, cg.p0, cg.z1.b, cg.z2.b);
    cg.St1w(cg.z3.s, cg.p0, AdrNoOfs(cg.x0));
    cg.Mov(cg.w0, 2);

    cg.Ret();
    cg.Ready();

    auto f = cg.GetCode<uint64_t(*)(uint32_t*)>();
    uint32_t arr[] = {1, 2, 3};
    ASSERT_EQ(f(arr), 2);
}

TEST(armcg_test_t, TestSplice)
{
    GTestJitGen cg;
    cg.Ldr(cg.p0, AdrNoOfs(cg.x0));
    cg.Ldr(cg.w9, AdrUimm(cg.x0, 4));

    cg.Dup(cg.z1.s, cg.w9);
    cg.Splice(cg.z2.d, cg.p0, cg.z1.d);
    cg.St1w(cg.z2.s, cg.p0, AdrNoOfs(cg.x0));
    cg.Mov(cg.w0, 2);

    cg.Ret();
    cg.Ready();

    auto f = cg.GetCode<uint64_t(*)(uint32_t*)>();
    uint32_t arr[] = {1, 2};
    ASSERT_EQ(f(arr), 2);
}

TEST(armcg_test_t, TestTrn)
{
    GTestJitGen cg;
    cg.Ptrue(cg.p0.b, Pattern::ALL);
    cg.Pfalse(cg.p1.b);
    cg.Trn1(cg.p2.s, cg.p0.s, cg.p1.s);
    cg.Trn1(cg.p3.d, cg.p0.d, cg.p1.d);

    cg.Ldr(cg.w9, AdrUimm(cg.x0, 0));
    cg.Ldr(cg.w10, AdrUimm(cg.x0, 4));
    cg.Dup(cg.z0.s, cg.w9);
    cg.Dup(cg.z1.s, cg.w10);

    cg.Trn1(cg.z2.s, cg.z0.s, cg.z1.s);
    cg.Trn1(cg.z3.d, cg.z0.d, cg.z1.d);

    cg.Trn2(cg.z2.s, cg.z0.s, cg.z1.s);
    cg.Trn2(cg.z3.d, cg.z0.d, cg.z1.d);
    cg.Mov(cg.w0, 1);
    cg.Ret();
    cg.Ready();

    auto f = cg.GetCode<uint64_t(*)(uint32_t*)>();
    uint32_t arr[] = {1, 2};
    ASSERT_EQ(f(arr), 1);
}

TEST(armcg_test_t, TestTrn1)
{
    GTestJitGen cg;
    cg.Mov(cg.x9, 0);
    cg.Mov(cg.x10, 4);
    cg.Whilelt(cg.p0.h, cg.x9, cg.x10);
    cg.Dup(cg.z0.h, 1);
    cg.Dup(cg.z1.h, 2);
    cg.Trn1(cg.z2.h, cg.z0.h, cg.z1.h);
    cg.Trn2(cg.z3.h, cg.z0.h, cg.z1.h);
    cg.St1h(cg.z2.h, cg.p0.h, AdrNoOfs(cg.x0));
    cg.St1h(cg.z3.h, cg.p0.h, AdrNoOfs(cg.x1));

    cg.Dup(cg.z4.s, 1);
    cg.Dup(cg.z5.s, 2);
    cg.Trn1(cg.z6.s, cg.z4.s, cg.z5.s);
    cg.Trn2(cg.z7.s, cg.z4.s, cg.z5.s);
    cg.Whilelt(cg.p1.s, cg.x9, cg.x10);
    cg.St1w(cg.z6.s, cg.p1.s, AdrNoOfs(cg.x2));
    cg.St1w(cg.z7.s, cg.p1.s, AdrNoOfs(cg.x3));

    cg.Ret();
    cg.Ready();

    auto f = cg.GetCode<void(*)(uint16_t*, uint16_t*, uint32_t*, uint32_t*)>();
    uint16_t ret1[4] = {0};
    uint16_t ret2[4] = {0};
    uint32_t ret3[4] = {0};
    uint32_t ret4[4] = {0};
    f(ret1, ret2, ret3, ret4);
    ASSERT_EQ(ret1[0], 1);
    ASSERT_EQ(ret1[1], 2);
    ASSERT_EQ(ret1[2], 1);
    ASSERT_EQ(ret1[3], 2);
    ASSERT_EQ(ret2[0], 1);
    ASSERT_EQ(ret2[1], 2);
    ASSERT_EQ(ret3[0], 1);
    ASSERT_EQ(ret3[1], 2);
    ASSERT_EQ(ret4[0], 1);
    ASSERT_EQ(ret4[1], 2);
}

TEST(armcg_test_t, TestUzp)
{
    GTestJitGen cg;
    cg.Ptrue(cg.p0.b, Pattern::ALL);
    cg.Pfalse(cg.p1.b);
    cg.Uzp1(cg.p2.b, cg.p0.b, cg.p1.b);
    cg.Uzp1(cg.p3.h, cg.p0.h, cg.p1.h);

    cg.Ldr(cg.w9, AdrUimm(cg.x0, 0));
    cg.Ldr(cg.w10, AdrUimm(cg.x0, 4));

    cg.Dup(cg.z0.s, cg.w9);
    cg.Dup(cg.z1.s, cg.w10);
    cg.Dup(cg.v1.s4, cg.w9);
    cg.Dup(cg.v2.s4, cg.w10);

    cg.Uzp1(cg.z2.b, cg.z0.b, cg.z1.b);
    cg.Uzp1(cg.z3.h, cg.z0.h, cg.z1.h);
    cg.Uzp2(cg.z3.d, cg.z0.d, cg.z1.d);
    cg.Uzp1(cg.v0.b16, cg.v1.b16, cg.v2.b16);
    cg.Uzp1(cg.v0.h8, cg.v1.h8, cg.v2.h8);
    cg.Mov(cg.w0, 1);
    cg.Ret();
    cg.Ready();

    auto f = cg.GetCode<uint64_t(*)(uint32_t*)>();
    uint32_t arr[] = {1, 2};
    ASSERT_EQ(f(arr), 1);
}

TEST(armcg_test_t, TestUzpD)
{
    GTestJitGen cg;
    cg.Dup(cg.z0.d, 0);
    cg.Dup(cg.z1.d, 0);
    cg.Dup(cg.z2.d, 0);
    cg.Dup(cg.z3.d, 0);
    cg.Ptrue(cg.p0.d);
    cg.Ld1d(cg.z0.d, cg.p0, AdrNoOfs(cg.x0));
    cg.Ld1d(cg.z1.d, cg.p0, AdrNoOfs(cg.x1));
    cg.Uzp1(cg.z2.d, cg.z0.d, cg.z1.d);
    cg.Uzp2(cg.z3.d, cg.z0.d, cg.z1.d);
    
    cg.St1d(cg.z2.d, cg.p0, AdrNoOfs(cg.x2));
    cg.St1d(cg.z3.d, cg.p0, AdrNoOfs(cg.x3));
    cg.Ret();
    cg.Ready();

    auto f = cg.GetCode<void(*)(double*, double*, double*, double*)>();
#ifdef KUDNNL_920F
    double arr0[8] = {1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8};
    double arr1[8] = {9.9, 10.1, 11.1, 12.1, 13.1, 14.1, 15.1, 16.1};
    double arr2[8] = {0};
    double arr3[8] = {0};
#else
    double arr0[4] = {1.1, 2.2, 3.3, 4.4};
    double arr1[4] = {5.5, 6.6, 7.7, 8.8};
    double arr2[4] = {0};
    double arr3[4] = {0};
#endif
    f(arr0, arr1, arr2, arr3);
#ifdef KUDNNL_920F
    ASSERT_EQ(arr2[0], 1.1);
    ASSERT_EQ(arr2[1], 3.3);
    ASSERT_EQ(arr2[2], 5.5);
    ASSERT_EQ(arr2[3], 7.7);
    ASSERT_EQ(arr2[4], 9.9);
    ASSERT_EQ(arr2[5], 11.1);
    ASSERT_EQ(arr2[6], 13.1);
    ASSERT_EQ(arr2[7], 15.1);
    ASSERT_EQ(arr3[0], 2.2);
    ASSERT_EQ(arr3[1], 4.4);
    ASSERT_EQ(arr3[2], 6.6);
    ASSERT_EQ(arr3[3], 8.8);
    ASSERT_EQ(arr3[4], 10.1);
    ASSERT_EQ(arr3[5], 12.1);
    ASSERT_EQ(arr3[6], 14.1);
    ASSERT_EQ(arr3[7], 16.1);
#else
    ASSERT_EQ(arr2[0], 1.1);
    ASSERT_EQ(arr2[1], 3.3);
    ASSERT_EQ(arr2[2], 5.5);
    ASSERT_EQ(arr2[3], 7.7);
    ASSERT_EQ(arr3[0], 2.2);
    ASSERT_EQ(arr3[1], 4.4);
    ASSERT_EQ(arr3[2], 6.6);
    ASSERT_EQ(arr3[3], 8.8);
#endif
}

TEST(armcg_test_t, TestZipD)
{
    GTestJitGen cg;
    cg.Ptrue(cg.p0.d);
    cg.Dup(cg.z0.d, 0);
    cg.Dup(cg.z1.d, 0);
    cg.Dup(cg.z2.d, 0);
    cg.Dup(cg.z3.d, 0);
    cg.Ld1d(cg.z0.d, cg.p0, AdrNoOfs(cg.x0));
    cg.Ld1d(cg.z1.d, cg.p0, AdrNoOfs(cg.x1));
    cg.Zip1(cg.z2.d, cg.z0.d, cg.z1.d);
    cg.Zip2(cg.z3.d, cg.z0.d, cg.z1.d);
    
    cg.St1d(cg.z2.d, cg.p0, AdrNoOfs(cg.x2));
    cg.St1d(cg.z3.d, cg.p0, AdrNoOfs(cg.x3));
    cg.Ret();
    cg.Ready();

    auto f = cg.GetCode<void(*)(double*, double*, double*, double*)>();
#ifdef KUDNNL_920F
    double arr0[8] = {1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8};
    double arr1[8] = {9.9, 10.1, 11.1, 12.1, 13.1, 14.1, 15.1, 16.1};
    double arr2[8] = {0};
    double arr3[8] = {0};
#else
    double arr0[4] = {1.1, 2.2, 3.3, 4.4};
    double arr1[4] = {5.5, 6.6, 7.7, 8.8};
    double arr2[4] = {0};
    double arr3[4] = {0};
#endif
    f(arr0, arr1, arr2, arr3);
#ifdef KUDNNL_920F
    ASSERT_EQ(arr2[0], 1.1);
    ASSERT_EQ(arr2[1], 9.9);
    ASSERT_EQ(arr2[2], 2.2);
    ASSERT_EQ(arr2[3], 10.1);
    ASSERT_EQ(arr2[4], 3.3);
    ASSERT_EQ(arr2[5], 11.1);
    ASSERT_EQ(arr2[6], 4.4);
    ASSERT_EQ(arr2[7], 12.1);
    ASSERT_EQ(arr3[0], 5.5);
    ASSERT_EQ(arr3[1], 13.1);
    ASSERT_EQ(arr3[2], 6.6);
    ASSERT_EQ(arr3[3], 14.1);
    ASSERT_EQ(arr3[4], 7.7);
    ASSERT_EQ(arr3[5], 15.1);
    ASSERT_EQ(arr3[6], 8.8);
    ASSERT_EQ(arr3[7], 16.1);
#else
    ASSERT_EQ(arr2[0], 1.1);
    ASSERT_EQ(arr2[1], 5.5);
    ASSERT_EQ(arr2[2], 2.2);
    ASSERT_EQ(arr2[3], 6.6);
    ASSERT_EQ(arr3[0], 3.3);
    ASSERT_EQ(arr3[1], 7.7);
    ASSERT_EQ(arr3[2], 4.4);
    ASSERT_EQ(arr3[3], 8.8);
#endif
}

TEST(armcg_test_t, TestZipNeon)
{
    GTestJitGen cg;
    cg.Mov(cg.w9, 2);
    cg.Mov(cg.w10, 4);
    cg.Dup(cg.v1.b16, cg.w9);
    cg.Dup(cg.v2.b16, cg.w10);
    cg.Zip1(cg.v3.b, cg.v1.b, cg.v2.b);
    cg.Zip2(cg.v4.b, cg.v1.b, cg.v2.b);
    cg.St1(VReg16BList(cg.v3.b), AdrNoOfs(cg.x0));
    cg.St1(VReg16BList(cg.v4.b), AdrNoOfs(cg.x1));

    cg.Ret();
    cg.Ready();

    auto f = cg.GetCode<void(*)(uint8_t*, uint8_t*)>();
    uint8_t ret1[16] = {0};
    uint8_t ret2[16] = {0};
    f(ret1, ret2);
    ASSERT_EQ(ret1[0], 2);
    ASSERT_EQ(ret1[1], 4);
    ASSERT_EQ(ret2[0], 2);
    ASSERT_EQ(ret2[1], 4);
}

TEST(armcg_test_t, TestZip1)
{
    GTestJitGen cg;
    cg.Ptrue(cg.p0.b, Pattern::ALL);
    cg.Pfalse(cg.p1.b);
    cg.Zip1(cg.p2.b, cg.p0.b, cg.p1.b);
    cg.Zip1(cg.p3.h, cg.p0.h, cg.p1.h);

    cg.Ldr(cg.w9, AdrUimm(cg.x0, 0));
    cg.Ldr(cg.w10, AdrUimm(cg.x0, 4));

    cg.Dup(cg.z0.s, cg.w9);
    cg.Dup(cg.z1.s, cg.w10);

    cg.Zip1(cg.z2.b, cg.z0.b, cg.z1.b);
    cg.Zip1(cg.z3.h, cg.z0.h, cg.z1.h);
    cg.Mov(cg.w0, 1);
    cg.Ret();
    cg.Ready();

    auto f = cg.GetCode<uint64_t(*)(uint32_t*)>();
    uint32_t arr[] = {1, 2};
    ASSERT_EQ(f(arr), 1);
}

TEST(armcg_test_t, TestZip2)
{
    GTestJitGen cg;
    cg.Mov(cg.x11, 0);
    cg.Whilelt(cg.p0.s, cg.x11, cg.x2);
    cg.Mov(cg.w9, 1);
    cg.Mov(cg.w10, 2);

    cg.Dup(cg.z0.s, cg.w9);
    cg.Dup(cg.z1.s, cg.w10);

    cg.Zip1(cg.z2.s, cg.z0.s, cg.z1.s);
    cg.Zip2(cg.z3.s, cg.z0.s, cg.z1.s);
    cg.Zip2(cg.z4.b, cg.z0.b, cg.z1.b);
    cg.Zip2(cg.z5.h, cg.z0.h, cg.z1.h);
    cg.St1w(cg.z2.s, cg.p0, AdrNoOfs(cg.x0));
    cg.St1w(cg.z3.s, cg.p0, AdrNoOfs(cg.x1));
    uint32_t buf1[] = {0, 0, 0, 0};
    uint32_t buf2[] = {0, 0, 0, 0};
    cg.Ret();
    cg.Ready();

    auto f = cg.GetCode<uint64_t(*)(uint32_t*, uint32_t*, int)>();
    f(buf1, buf2, 4);
    ASSERT_EQ(buf1[0], 1);
    ASSERT_EQ(buf1[1], 2);
    ASSERT_EQ(buf1[2], 1);
    ASSERT_EQ(buf1[3], 2);
    ASSERT_EQ(buf2[0], 1);
    ASSERT_EQ(buf2[1], 2);
    ASSERT_EQ(buf2[2], 1);
    ASSERT_EQ(buf2[3], 2);
}

// bit
TEST(armcg_test_t, TestAndImmByte2)
{
    GTestJitGen cg;
    cg.Mov(cg.x9, 0);
    cg.Whilelt(cg.p0.b, cg.x9, cg.x2);
    cg.Ld1b(cg.z0.b, cg.p0, AdrNoOfs(cg.x0));
    cg.Ld1b(cg.z1.b, cg.p0, AdrNoOfs(cg.x1));
    cg.And(cg.z0.b, cg.p0, cg.z1.b);
    cg.St1b(cg.z0.b, cg.p0, AdrNoOfs(cg.x0));
    cg.Ret();
    cg.Ready();
    auto f = cg.GetCode<uint8_t(*)(uint8_t*, uint8_t*, int)>();
    uint8_t buf1[] = {1, 2, 3, 4, 5, 6, 7, 8};
    uint8_t buf2[] = {8, 7, 6, 5, 4, 3, 2, 1};
    f(buf1, buf2, 8);
    ASSERT_EQ(buf1[0], 0);
    ASSERT_EQ(buf1[1], 2);
    ASSERT_EQ(buf1[2], 2);
    ASSERT_EQ(buf1[3], 4);
    ASSERT_EQ(buf1[4], 4);
    ASSERT_EQ(buf1[5], 2);
    ASSERT_EQ(buf1[6], 2);
    ASSERT_EQ(buf1[7], 0);
}

TEST(armcg_test_t, TestAndImmInt)
{
    GTestJitGen cg;
    cg.Mov(cg.x9, 0);
    cg.Whilelt(cg.p0.s, cg.x9, cg.x2);
    cg.Ld1w(cg.z0.s, cg.p0, AdrNoOfs(cg.x0));
    cg.Ld1w(cg.z1.s, cg.p0, AdrNoOfs(cg.x1));
    cg.And(cg.z0.s, cg.p0, cg.z1.s);
    cg.St1w(cg.z0.s, cg.p0, AdrNoOfs(cg.x0));
    cg.Ret();
    cg.Ready();
    auto f = cg.GetCode<uint32_t(*)(uint32_t*, uint32_t*, int)>();
    uint32_t buf1[] = {1, 2, 3, 4, 5, 6};
    uint32_t buf2[] = {8, 7, 6, 5, 4, 3};
    f(buf1, buf2, 6);
    ASSERT_EQ(buf1[0], 0);
    ASSERT_EQ(buf1[1], 2);
    ASSERT_EQ(buf1[2], 2);
    ASSERT_EQ(buf1[3], 4);
    ASSERT_EQ(buf1[4], 4);
    ASSERT_EQ(buf1[5], 2);
}
// Arithmetic test

// test Add interface
TEST(armcg_test_t, TestAdd2)
{
    GTestJitGen cg;
    cg.Add(cg.x0, cg.x1, 10);
    cg.Ret();
    cg.Ready();

    auto Func = cg.GetCode<int (*)(int, int)>();
    ASSERT_EQ(Func(0, 2), 12);
}

TEST(armcg_test_t, TestAdd3)
{
    GTestJitGen cg;
    cg.Ldr(cg.w9, AdrUimm(cg.x0, 0));
    cg.Ldr(cg.w10, AdrUimm(cg.x0, 4));
    cg.Add(cg.w0, cg.w9, cg.w10);
    cg.Add(cg.w0, cg.w0, cg.w1);
    cg.Ret();
    cg.Ready();

    auto Func = cg.GetCode<int (*)(int*, int)>();
    int arr[] = {3, 4};
    int scale = 5;
    ASSERT_EQ(Func(arr, scale), 12);
}

TEST(armcg_test_t, TestAdd4)
{
    GTestJitGen cg;
    cg.Add(cg.x0, cg.x1, cg.x0);
    cg.Ret();
    cg.Ready();

    auto Func = cg.GetCode<int (*)(int, int)>();
    ASSERT_EQ(Func(0, 2), 2);

    ARMCG::XReg r1(500);
    ARMCG::XReg r2(800);
    EXPECT_THROW(cg.Add(r1, r2, 100, 66), std::exception);
    EXPECT_THROW(cg.Subs(r1, r2, 100, 66), std::exception);
}

TEST(armcg_test_t, TestAdd6)
{
    GTestJitGen cg;
    cg.Add(cg.x0, cg.x1, cg.x0, ARMCG::ExtMod::UXTX);
    cg.Ret();
    cg.Ready();

    auto Func = cg.GetCode<int (*)(int, int)>();
    ASSERT_EQ(Func(0, 2), 2);
}

TEST(armcg_test_t, TestAdd8)
{
    GTestJitGen cg;
    cg.Dup(cg.v1.s4, cg.w0);
    cg.Add(cg.v1.s4, cg.v1.s4, cg.v1.s4);
    cg.Mov(cg.w0, cg.v1.s4[0]);
    cg.Ret();
    cg.Ready();

    auto Func = cg.GetCode<int (*)(int)>();
    int res = 2;
    ASSERT_EQ(Func(res), 4);
}

TEST(armcg_test_t, TestAdd9)
{
    GTestJitGen cg;
    cg.Mov(cg.x9, 0);
    cg.Whilelt(cg.p0.s, cg.x9, cg.x1);
    cg.Ld1w(cg.z0.s, cg.p0, AdrNoOfs(cg.x0));
    cg.Add(cg.z0.s, cg.z0.s, cg.z0.s);
    cg.St1w(cg.z0.s, cg.p0, AdrNoOfs(cg.x0));
    cg.Ret();
    cg.Ready();

    auto Func = cg.GetCode<void (*)(int*, int)>();
    int arr1[] = {0, 1, 2, 3, 4, 5, 6, 7};
    int arr2[] = {0, 2, 4, 6, 8, 10, 12, 14};
    Func(arr1, 8);
    ASSERT_EQ(arr2[0], arr1[0]);
    ASSERT_EQ(arr2[1], arr1[1]);
    ASSERT_EQ(arr2[2], arr1[2]);
    ASSERT_EQ(arr2[3], arr1[3]);
    ASSERT_EQ(arr2[4], arr1[4]);
    ASSERT_EQ(arr2[5], arr1[5]);
    ASSERT_EQ(arr2[6], arr1[6]);
    ASSERT_EQ(arr2[7], arr1[7]);
}

// test Adds interface
TEST(armcg_test_t, TestAdds2)
{
    GTestJitGen cg;
    cg.Adds(cg.x0, cg.x1, 10);
    cg.Ret();
    cg.Ready();

    auto Func = cg.GetCode<int (*)(int, int)>();
    ASSERT_EQ(Func(0, 2), 12);
}

TEST(armcg_test_t, TestAdds4)
{
    GTestJitGen cg;
    cg.Adds(cg.x0, cg.x1, cg.x0);
    cg.Ret();
    cg.Ready();

    auto Func = cg.GetCode<int (*)(int, int)>();
    ASSERT_EQ(Func(0, 2), 2);
}

// test Addv interface
TEST(armcg_test_t, TestAddv)
{
    GTestJitGen cg;
    cg.Dup(cg.v1.s4, cg.w0);
    cg.Addv(cg.s3, cg.v1.s4);
    cg.Str(cg.s3, AdrUimm(cg.x1, 0));
    cg.Ret();
    cg.Ready();

    auto Func = cg.GetCode<int (*)(int, int*)>();
    int arr[] = {0};
    Func(2, arr);
    ASSERT_EQ(arr[0], 8);
}

TEST(armcg_test_t, TestSaddv)
{
    GTestJitGen cg;
    cg.Mov(cg.x9, 0);
    cg.Mov(cg.x10, 4);
    cg.Whilelt(cg.p0.s, cg.x9, cg.x10);
    cg.Ld1w(cg.z1.s, cg.p0.s, AdrNoOfs(cg.x0));
    cg.Saddv(cg.d2, cg.p0.s, cg.z1.s);
    cg.Uaddv(cg.d3, cg.p0.s, cg.z1.s);
    cg.Str(cg.d2, AdrUimm(cg.x1, 0));
    cg.Str(cg.d3, AdrUimm(cg.x2, 0));
    cg.Ret();
    cg.Ready();

    auto Func = cg.GetCode<int (*)(int*, uint64_t*, uint64_t*)>();
    int in[] = {1, 2, 3, 4};
    uint64_t arr1, arr2;
    Func(in, &arr1, &arr2);
    ASSERT_EQ(arr1, 10);
    ASSERT_EQ(arr2, 10);
}

// test Fadd interface
TEST(armcg_test_t, TestFadd1)
{
    GTestJitGen cg;
    cg.Dup(cg.v1.s4, cg.w0);
    cg.Fadd(cg.v1.s4, cg.v1.s4, cg.v1.s4);
    cg.Mov(cg.w0, cg.v1.s4[0]);
    cg.Ret();
    cg.Ready();

    auto Func = cg.GetCode<int (*)(int)>();
    int res = 2;
    ASSERT_EQ(Func(res), 4);
}

TEST(armcg_test_t, TestFadd2)
{
    GTestJitGen cg;
    cg.Ldr(cg.s3, AdrUimm(cg.x0, 0));
    cg.Fadd(cg.s3, cg.s3, cg.s3);
    cg.Str(cg.s3, AdrUimm(cg.x0, 0));
    cg.Ret();
    cg.Ready();

    auto Func = cg.GetCode<int (*)(int*)>();
    int arr[] = {3};
    Func(arr);
    ASSERT_EQ(arr[0], 6);
}

TEST(armcg_test_t, TestFadd3)
{
    GTestJitGen cg;
    cg.Mov(cg.x9, 0);
    cg.Whilelt(cg.p0.s, cg.x9, cg.x1);
    cg.Ld1h(cg.z0.h, cg.p0, AdrNoOfs(cg.x0));
    cg.Ld1h(cg.z1.h, cg.p0, AdrNoOfs(cg.x2));
    cg.Fadd(cg.z0.h, cg.z0.h, cg.z1.h);
    cg.St1h(cg.z0.h, cg.p0, AdrNoOfs(cg.x2));
    cg.Ret();
    cg.Ready();

    auto Func = cg.GetCode<void (*)(uint16_t*, int, uint16_t*)>();
    uint16_t arr1[] = {0x0, 0x0, 0xc600, 0x0, 0x2d93, 0x0, 0xa9ef, 0x0,
        0x5100, 0x0, 0xcd00, 0x0, 0x2d28, 0x0, 0xb831, 0x0};
    uint16_t arr2[] = {0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0};
    Func(arr1, 8, arr2);
    ASSERT_EQ(arr2[0], 0x0);
    ASSERT_EQ(arr2[2], 0xc600);
    ASSERT_EQ(arr2[4], 0x2d93);
    ASSERT_EQ(arr2[6], 0xa9ef);
    ASSERT_EQ(arr2[8], 0x5100);
    ASSERT_EQ(arr2[10], 0xcd00);
    ASSERT_EQ(arr2[12], 0x2d28);
    ASSERT_EQ(arr2[14], 0xb831);
}

TEST(armcg_test_t, TestFadd4)
{
    GTestJitGen cg;
    cg.Mov(cg.x9, 0);
    cg.Whilelt(cg.p0.s, cg.x9, cg.x1);
    cg.Ld1w(cg.z0.s, cg.p0, AdrNoOfs(cg.x0));
    cg.Fadd(cg.z0.s, cg.z0.s, cg.z0.s);
    cg.St1w(cg.z0.s, cg.p0, AdrNoOfs(cg.x0));
    cg.Ret();
    cg.Ready();

    auto Func = cg.GetCode<void (*)(float*, int)>();
    float arr1[] = {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f};
    float arr2[] = {0.0f, 2.0f, 4.0f, 6.0f, 8.0f, 10.0f, 12.0f, 14.0f};
    Func(arr1, 8);
    ASSERT_EQ(arr2[0], arr1[0]);
    ASSERT_EQ(arr2[1], arr1[1]);
    ASSERT_EQ(arr2[2], arr1[2]);
    ASSERT_EQ(arr2[3], arr1[3]);
    ASSERT_EQ(arr2[4], arr1[4]);
    ASSERT_EQ(arr2[5], arr1[5]);
    ASSERT_EQ(arr2[6], arr1[6]);
    ASSERT_EQ(arr2[7], arr1[7]);
}

TEST(armcg_test_t, TestFadd5)
{
    GTestJitGen cg;
    cg.Mov(cg.x9, 0);
    cg.Whilelt(cg.p0.s, cg.x9, cg.x1);
    cg.Ld1w(cg.z0.s, cg.p0, AdrNoOfs(cg.x0));
    cg.Fadd(cg.z0.s, cg.p0.s, cg.z0.s);
    cg.St1w(cg.z0.s, cg.p0, AdrNoOfs(cg.x0));
    cg.Ret();
    cg.Ready();

    auto Func = cg.GetCode<void (*)(float*, int)>();
    float arr1[] = {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f};
    float arr2[] = {0.0f, 2.0f, 4.0f, 6.0f, 8.0f, 10.0f, 12.0f, 14.0f};
    Func(arr1, 8);
    ASSERT_EQ(arr2[0], arr1[0]);
    ASSERT_EQ(arr2[1], arr1[1]);
    ASSERT_EQ(arr2[2], arr1[2]);
    ASSERT_EQ(arr2[3], arr1[3]);
    ASSERT_EQ(arr2[4], arr1[4]);
    ASSERT_EQ(arr2[5], arr1[5]);
    ASSERT_EQ(arr2[6], arr1[6]);
    ASSERT_EQ(arr2[7], arr1[7]);
}

TEST(armcg_test_t, TestFadd6)
{
    GTestJitGen cg;
    cg.Mov(cg.x9, 0);
    cg.Whilelt(cg.p0.s, cg.x9, cg.x1);
    cg.Ld1w(cg.z0.s, cg.p0, AdrNoOfs(cg.x0));
    cg.Fadd(cg.z0.s, cg.p0.s, 1);
    cg.St1w(cg.z0.s, cg.p0, AdrNoOfs(cg.x0));
    cg.Ret();
    cg.Ready();

    auto Func = cg.GetCode<void (*)(float*, int)>();
    float arr1[] = {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f};
    float arr2[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
    Func(arr1, 8);
    ASSERT_EQ(arr2[0], arr1[0]);
    ASSERT_EQ(arr2[1], arr1[1]);
    ASSERT_EQ(arr2[2], arr1[2]);
    ASSERT_EQ(arr2[3], arr1[3]);
    ASSERT_EQ(arr2[4], arr1[4]);
    ASSERT_EQ(arr2[5], arr1[5]);
    ASSERT_EQ(arr2[6], arr1[6]);
    ASSERT_EQ(arr2[7], arr1[7]);
}

TEST(armcg_test_t, TestFaddp)
{
    GTestJitGen hh2; // h(HReg)h2(VReg2H)
    hh2.Ld1(VReg8HList(hh2.v1.h8), AdrNoOfs(hh2.x1));
    hh2.Faddp(hh2.h0, hh2.v1.h2);
    hh2.St1(VReg8HList(hh2.v0.h8), AdrNoOfs(hh2.x0));
    hh2.Ret();
    hh2.Ready();
    auto hh2Func = hh2.GetCode<void (*)(uint16_t*, uint16_t*)>();
    uint16_t hh2Src[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    uint16_t hh2Dst[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    hh2Func(hh2Dst, hh2Src);
    ASSERT_EQ(hh2Dst[0], 0);
    ASSERT_EQ(hh2Dst[1], 0);

    GTestJitGen ss2; // s(SReg)s2(VReg2S)
    ss2.Ld1(VReg4SList(ss2.v1.s4), AdrNoOfs(ss2.x1));
    ss2.Faddp(ss2.s0, ss2.v1.s2);
    ss2.St1(VReg4SList(ss2.v0.s4), AdrNoOfs(ss2.x0));
    ss2.Ret();
    ss2.Ready();
    auto ss2Func = ss2.GetCode<void (*)(float*, float*)>();
    float ss2Src[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    float ss2Dst[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    ss2Func(ss2Dst, ss2Src);
    ASSERT_EQ(ss2Dst[0], 3.0f);
    ASSERT_EQ(ss2Dst[1], 0.0f);

    GTestJitGen h4h4h4; // h4(VReg4H)h4(VReg4H)h4(VReg4H)
    h4h4h4.Ld1(VReg8HList(h4h4h4.v1.h8), AdrNoOfs(h4h4h4.x1));
    h4h4h4.Ld1(VReg8HList(h4h4h4.v2.h8), AdrNoOfs(h4h4h4.x2));
    h4h4h4.Faddp(h4h4h4.v0.h8, h4h4h4.v1.h8, h4h4h4.v2.h8);
    h4h4h4.St1(VReg8HList(h4h4h4.v0.h8), AdrNoOfs(h4h4h4.x0));
    h4h4h4.Ret();
    h4h4h4.Ready();
    auto h4h4h4Func = h4h4h4.GetCode<void (*)(uint16_t*, uint16_t*, uint16_t*)>();
    uint16_t h4h4h4Src1[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    uint16_t h4h4h4Src2[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    uint16_t h4h4h4Dst1[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    h4h4h4Func(h4h4h4Dst1, h4h4h4Src1, h4h4h4Src2);
    ASSERT_EQ(h4h4h4Dst1[0], 0);
    ASSERT_EQ(h4h4h4Dst1[1], 0);

    GTestJitGen h8h8h8; // h8(VReg8H)h8(VReg8H)h8(VReg8H)
    h8h8h8.Ld1(VReg8HList(h8h8h8.v1.h8), AdrNoOfs(h8h8h8.x1));
    h8h8h8.Ld1(VReg8HList(h8h8h8.v2.h8), AdrNoOfs(h8h8h8.x2));
    h8h8h8.Faddp(h8h8h8.v0.h8, h8h8h8.v1.h8, h8h8h8.v2.h8);
    h8h8h8.St1(VReg8HList(h8h8h8.v0.h8), AdrNoOfs(h8h8h8.x0));
    h8h8h8.Ret();
    h8h8h8.Ready();
    auto h8h8h8Func = h8h8h8.GetCode<void (*)(uint16_t*, uint16_t*, uint16_t*)>();
    uint16_t h8h8h8Src1[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    uint16_t h8h8h8Src2[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    uint16_t h8h8h8Dst1[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    h8h8h8Func(h8h8h8Dst1, h8h8h8Src1, h8h8h8Src2);
    ASSERT_EQ(h8h8h8Dst1[0], 0);
    ASSERT_EQ(h8h8h8Dst1[1], 0);

    GTestJitGen s2s2s2; // s2(VReg2S)s2(VReg2S)s2(VReg2S)
    s2s2s2.Ld1(VReg4SList(s2s2s2.v1.s4), AdrNoOfs(s2s2s2.x1));
    s2s2s2.Ld1(VReg4SList(s2s2s2.v2.s4), AdrNoOfs(s2s2s2.x2));
    s2s2s2.Faddp(s2s2s2.v0.s4, s2s2s2.v1.s4, s2s2s2.v2.s4);
    s2s2s2.St1(VReg4SList(s2s2s2.v0.s4), AdrNoOfs(s2s2s2.x0));
    s2s2s2.Ret();
    s2s2s2.Ready();
    auto s2s2s2Func = s2s2s2.GetCode<void (*)(float*, float*, float*)>();
    float s2s2s2Src1[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    float s2s2s2Src2[4] = {5.0f, 6.0f, 7.0f, 8.0f};
    float s2s2s2Dst1[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    s2s2s2Func(s2s2s2Dst1, s2s2s2Src1, s2s2s2Src2);
    ASSERT_EQ(s2s2s2Dst1[0], 3.0f);
    ASSERT_EQ(s2s2s2Dst1[1], 7.0f);

    GTestJitGen s4s4s4; // s4(VReg4S)s4(VReg4S)s4(VReg4S)
    s4s4s4.Ld1(VReg4SList(s4s4s4.v1.s4), AdrNoOfs(s4s4s4.x1));
    s4s4s4.Ld1(VReg4SList(s4s4s4.v2.s4), AdrNoOfs(s4s4s4.x2));
    s4s4s4.Faddp(s4s4s4.v0.s4, s4s4s4.v1.s4, s4s4s4.v2.s4);
    s4s4s4.St1(VReg4SList(s4s4s4.v0.s4), AdrNoOfs(s4s4s4.x0));
    s4s4s4.Ret();
    s4s4s4.Ready();
    auto s4s4s4Func = s4s4s4.GetCode<void (*)(float*, float*, float*)>();
    float s4s4s4Src1[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    float s4s4s4Src2[4] = {5.0f, 6.0f, 7.0f, 8.0f};
    float s4s4s4Dst1[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    s4s4s4Func(s4s4s4Dst1, s4s4s4Src1, s4s4s4Src2);
    ASSERT_EQ(s4s4s4Dst1[0], 3.0f);
    ASSERT_EQ(s4s4s4Dst1[1], 7.0f);
}

TEST(armcg_test_t, TestFaddv)
{
    GTestJitGen cg;
    cg.Mov(cg.x9, 0);
    cg.Mov(cg.x10, 8);
    cg.Whilelt(cg.p0.s, cg.x9, cg.x10);
    cg.Ld1w(cg.z0.s, cg.p0, AdrNoOfs(cg.x0));
    cg.Faddv(cg.s1, cg.p0.s, cg.z0.s);
    cg.Str(cg.s1, AdrUimm(cg.x1, 0));
    cg.Mov(cg.x10, 4);
    cg.Whilelt(cg.p1.d, cg.x9, cg.x10);
    cg.Ld1d(cg.z2.d, cg.p1, AdrNoOfs(cg.x2));
    cg.Faddv(cg.d3, cg.p1.d, cg.z2.d);
    cg.Str(cg.d3, AdrUimm(cg.x3, 0));
    cg.Ret();
    cg.Ready();

    auto Func = cg.GetCode<void (*)(float*, float*, double*, double*)>();
    float arr1[] = {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f};
    double arr2[] = {1.0, 2.0, 3.0, 4.0};
    float ret1[] = {0.0};
    double ret2[] = {0.0};
    Func(arr1, ret1, arr2, ret2);
    ASSERT_EQ(ret1[0], 28.0);
    ASSERT_EQ(ret2[0], 10.0);
}

// test Fsub interface
TEST(armcg_test_t, TestFsub1)
{
    GTestJitGen cg;
    cg.Dup(cg.v1.s4, cg.w0);
    cg.Fsub(cg.v1.s4, cg.v1.s4, cg.v1.s4);
    cg.Mov(cg.w0, cg.v1.s4[0]);
    cg.Ret();
    cg.Ready();
    auto Func = cg.GetCode<int (*)(int)>();
    int res = 2;
    ASSERT_EQ(Func(res), 0);
}

TEST(armcg_test_t, TestFsub2)
{
    GTestJitGen cg;
    cg.Mov(cg.x9, 0);
    cg.Whilelt(cg.p0.s, cg.x9, cg.x1);
    cg.Ld1w(cg.z0.s, cg.p0, AdrNoOfs(cg.x0));
    cg.Fsub(cg.z0.s, cg.z0.s, cg.z0.s);
    cg.St1w(cg.z0.s, cg.p0, AdrNoOfs(cg.x0));
    cg.Ret();
    cg.Ready();

    auto Func = cg.GetCode<void (*)(float*, int)>();
    float arr1[] = {0.0f, 9.0f, 25.0f, 0.046f, 40.0f, 20.0f, 0.0805929987f, 0.524f};
    Func(arr1, 8);
    ASSERT_EQ(0, arr1[0]);
    ASSERT_EQ(0, arr1[1]);
    ASSERT_EQ(0, arr1[2]);
    ASSERT_EQ(0, arr1[3]);
    ASSERT_EQ(0, arr1[4]);
    ASSERT_EQ(0, arr1[5]);
    ASSERT_EQ(0, arr1[6]);
    ASSERT_EQ(0, arr1[7]);
}

TEST(armcg_test_t, TestFsub3)
{
    GTestJitGen cg;
    cg.Mov(cg.x9, 0);
    cg.Whilelt(cg.p0.s, cg.x9, cg.x1);
    cg.Ld1w(cg.z0.s, cg.p0, AdrNoOfs(cg.x0));
    cg.Fsub(cg.z0.s, cg.p0, 1);
    cg.St1w(cg.z0.s, cg.p0, AdrNoOfs(cg.x0));
    cg.Ret();
    cg.Ready();

    auto Func = cg.GetCode<void (*)(float*, int)>();
    float arr1[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
    float arr2[] = {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f};
    Func(arr1, 8);
    ASSERT_EQ(arr2[0], arr1[0]);
    ASSERT_EQ(arr2[1], arr1[1]);
    ASSERT_EQ(arr2[2], arr1[2]);
    ASSERT_EQ(arr2[3], arr1[3]);
    ASSERT_EQ(arr2[4], arr1[4]);
    ASSERT_EQ(arr2[5], arr1[5]);
    ASSERT_EQ(arr2[6], arr1[6]);
    ASSERT_EQ(arr2[7], arr1[7]);
}

// test Sub interface
TEST(armcg_test_t, TestSub1)
{
    GTestJitGen cg;
    cg.Sub(cg.w0, cg.w1, 10);
    cg.Ret();
    cg.Ready();

    auto Func = cg.GetCode<int (*)(int, int)>();
    ASSERT_EQ(Func(0, 2), -8);
}

TEST(armcg_test_t, TestSub2)
{
    GTestJitGen cg;
    cg.Sub(cg.x0, cg.x1, 10);
    cg.Ret();
    cg.Ready();

    auto Func = cg.GetCode<int (*)(int, int)>();
    ASSERT_EQ(Func(0, 2), -8);
}

TEST(armcg_test_t, TestSub4)
{
    GTestJitGen cg;
    cg.Sub(cg.x0, cg.x1, cg.x0);
    cg.Ret();
    cg.Ready();

    auto Func = cg.GetCode<int (*)(int, int)>();
    ASSERT_EQ(Func(0, 2), 2);
}

TEST(armcg_test_t, TestSub8)
{
    GTestJitGen cg;
    cg.Dup(cg.v1.s4, cg.w0);
    cg.Sub(cg.v1.s4, cg.v1.s4, cg.v1.s4);
    cg.Mov(cg.w0, cg.v1.s4[0]);
    cg.Ret();
    cg.Ready();
    auto Func = cg.GetCode<int (*)(int)>();
    int res = 2;
    ASSERT_EQ(Func(res), 0);
}

TEST(armcg_test_t, TestSub10)
{
    GTestJitGen cg;
    cg.Mov(cg.x9, 0);
    cg.Whilelt(cg.p0.s, cg.x9, cg.x1);
    cg.Ld1w(cg.z0.s, cg.p0, AdrNoOfs(cg.x0));
    cg.Sub(cg.z0.s, cg.z0.s, cg.z0.s);
    cg.St1w(cg.z0.s, cg.p0, AdrNoOfs(cg.x0));
    cg.Ret();
    cg.Ready();

    auto Func = cg.GetCode<void (*)(float*, int)>();
    float arr1[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
    Func(arr1, 8);
    ASSERT_EQ(0, arr1[0]);
    ASSERT_EQ(0, arr1[1]);
    ASSERT_EQ(0, arr1[2]);
    ASSERT_EQ(0, arr1[3]);
    ASSERT_EQ(0, arr1[4]);
    ASSERT_EQ(0, arr1[5]);
    ASSERT_EQ(0, arr1[6]);
    ASSERT_EQ(0, arr1[7]);
}

// test Subs interface
TEST(armcg_test_t, TestSubs2)
{
    GTestJitGen cg;
    cg.Subs(cg.x0, cg.x1, 10);
    cg.Ret();
    cg.Ready();

    auto Func = cg.GetCode<int (*)(int, int)>();
    ASSERT_EQ(Func(0, 2), -8);
}

TEST(armcg_test_t, TestSubs3)
{
    GTestJitGen cg;
    cg.Ldr(cg.w9, AdrUimm(cg.x0, 0));
    cg.Ldr(cg.w10, AdrUimm(cg.x0, 4));
    cg.Subs(cg.w0, cg.w9, cg.w10);
    cg.Subs(cg.w0, cg.w0, cg.w1);
    cg.Ret();
    cg.Ready();

    auto Func = cg.GetCode<int (*)(int*, int)>();
    int arr[] = {3, 4};
    int scale = 5;
    ASSERT_EQ(Func(arr, scale), -6);
}

TEST(armcg_test_t, TestSubs4)
{
    GTestJitGen cg;
    cg.Subs(cg.x0, cg.x1, cg.x0);
    cg.Ret();
    cg.Ready();

    auto Func = cg.GetCode<int (*)(int, int)>();
    ASSERT_EQ(Func(0, 2), 2);
}

TEST(armcg_test_t, TestSubs5)
{
    GTestJitGen cg;
    cg.Subs(cg.w0, cg.w1, cg.w0, ARMCG::ExtMod::UXTW);
    cg.Ret();
    cg.Ready();

    auto Func = cg.GetCode<int (*)(int, int)>();
    ASSERT_EQ(Func(1, 2), 1);
}

TEST(armcg_test_t, TestSubs6)
{
    GTestJitGen cg;
    cg.Subs(cg.x0, cg.x1, cg.x0, ARMCG::ExtMod::UXTX);
    cg.Ret();
    cg.Ready();

    auto Func = cg.GetCode<int (*)(int, int)>();
    ASSERT_EQ(Func(1, 2), 1);
}

TEST(armcg_test_t, TestSubs7)
{
    GTestJitGen cg;
    cg.Subs(cg.x0, cg.x0, cg.w1, ARMCG::ExtMod::UXTW);
    cg.Ret();
    cg.Ready();

    auto Func = cg.GetCode<int (*)(int, int)>();
    ASSERT_EQ(Func(1, 2), -1);
}

// test Fmul interface
TEST(armcg_test_t, TestFmul1)
{
    GTestJitGen cg;
    cg.Ld1(VReg4SList(cg.v0.s4), AdrNoOfs(cg.x0));
    cg.Fmul(cg.v1.s4, cg.v0.s4, cg.v0.s4);
    cg.Fmul(cg.v2.s4, cg.v0.s4, cg.v0.s4[0]);
    cg.St1(VReg4SList(cg.v1.s4), AdrNoOfs(cg.x1));
    cg.St1(VReg4SList(cg.v2.s4), AdrNoOfs(cg.x2));
    cg.Ret();
    cg.Ready();

    auto Func = cg.GetCode<void (*)(float*, float*, float*)>();
    float res[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    float ret1[4] = {0.0f};
    float ret2[4] = {0.0f};
    Func(res, ret1, ret2);
    ASSERT_EQ(ret1[0], 1.0f);
    ASSERT_EQ(ret1[1], 4.0f);
    ASSERT_EQ(ret1[2], 9.0f);
    ASSERT_EQ(ret1[3], 16.0f);
    ASSERT_EQ(ret2[0], 1.0f);
    ASSERT_EQ(ret2[1], 2.0f);
    ASSERT_EQ(ret2[2], 3.0f);
    ASSERT_EQ(ret2[3], 4.0f);
}

TEST(armcg_test_t, TestFmul2)
{
    GTestJitGen cg;
    cg.Mov(cg.x9, 0);
    cg.Whilelt(cg.p0.s, cg.x9, cg.x1);
    cg.Ld1w(cg.z0.s, cg.p0, AdrNoOfs(cg.x0));
    cg.Fmul(cg.z0.s, cg.z0.s, cg.z0.s);
    cg.St1w(cg.z0.s, cg.p0, AdrNoOfs(cg.x0));
    cg.Ret();
    cg.Ready();

    auto Func = cg.GetCode<void (*)(float*, int)>();
    float arr1[] = {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f};
    float arr2[] = {0.0f, 1.0f, 4.0f, 9.0f, 16.0f, 25.0f, 36.0f, 49.0f};
    Func(arr1, 8);
    ASSERT_EQ(arr2[0], arr1[0]);
    ASSERT_EQ(arr2[1], arr1[1]);
    ASSERT_EQ(arr2[2], arr1[2]);
    ASSERT_EQ(arr2[3], arr1[3]);
    ASSERT_EQ(arr2[4], arr1[4]);
    ASSERT_EQ(arr2[5], arr1[5]);
    ASSERT_EQ(arr2[6], arr1[6]);
    ASSERT_EQ(arr2[7], arr1[7]);
}

TEST(armcg_test_t, TestFmul4)
{
    GTestJitGen cg;
    cg.Mov(cg.x9, 0);
    cg.Whilelt(cg.p0.s, cg.x9, cg.x1);
    cg.Ld1w(cg.z0.s, cg.p0, AdrNoOfs(cg.x0));
    cg.Fmul(cg.z0.s, cg.p0, cg.z0.s);
    cg.St1w(cg.z0.s, cg.p0, AdrNoOfs(cg.x0));
    cg.Ret();
    cg.Ready();

    auto Func = cg.GetCode<void (*)(float*, int)>();
    float arr1[] = {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f};
    float arr2[] = {0.0f, 1.0f, 4.0f, 9.0f, 16.0f, 25.0f, 36.0f, 49.0f};
    Func(arr1, 8);
    ASSERT_EQ(arr2[0], arr1[0]);
    ASSERT_EQ(arr2[1], arr1[1]);
    ASSERT_EQ(arr2[2], arr1[2]);
    ASSERT_EQ(arr2[3], arr1[3]);
    ASSERT_EQ(arr2[4], arr1[4]);
    ASSERT_EQ(arr2[5], arr1[5]);
    ASSERT_EQ(arr2[6], arr1[6]);
    ASSERT_EQ(arr2[7], arr1[7]);
}

TEST(armcg_test_t, TestFmul5)
{
    GTestJitGen cg;
    cg.Mov(cg.x9, 0);
    cg.Whilelt(cg.p0.s, cg.x9, cg.x1);
    cg.Ld1w(cg.z0.s, cg.p0, AdrNoOfs(cg.x0));
    cg.Fmul(cg.z0.s, cg.p0, 2);
    cg.St1w(cg.z0.s, cg.p0, AdrNoOfs(cg.x0));
    cg.Ret();
    cg.Ready();

    auto Func = cg.GetCode<void (*)(float*, int)>();
    float arr1[] = {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f};
    float arr2[] = {0.0f, 2.0f, 4.0f, 6.0f, 8.0f, 10.0f, 12.0f, 14.0f};
    Func(arr1, 8);
    ASSERT_EQ(arr2[0], arr1[0]);
    ASSERT_EQ(arr2[1], arr1[1]);
    ASSERT_EQ(arr2[2], arr1[2]);
    ASSERT_EQ(arr2[3], arr1[3]);
    ASSERT_EQ(arr2[4], arr1[4]);
    ASSERT_EQ(arr2[5], arr1[5]);
    ASSERT_EQ(arr2[6], arr1[6]);
    ASSERT_EQ(arr2[7], arr1[7]);
}

TEST(armcg_test_t, TestFmul)
{
    GTestJitGen cg;
    cg.Fmov(cg.z0.h, 0.0f);
    cg.Fmov(cg.z1.h, 0.0f);
    cg.Fmov(cg.z2.h, 0.0f);
    cg.Mov(cg.x9, 0);
    cg.Whilelt(cg.p0.h, cg.x9, cg.x3);
    cg.Ld1h(cg.z0.h, cg.p0, AdrNoOfs(cg.x0));
    cg.Ld1h(cg.z1.h, cg.p0, AdrNoOfs(cg.x1));
    cg.Fmul(cg.z2.h, cg.z0.h, cg.z1.h);
    cg.St1h(cg.z2.h, cg.p0, AdrScImm(cg.x2, 1));
    cg.Ret();
    cg.Ready();

    auto Func = cg.GetCode<void (*)(uint16_t*, uint16_t*, uint16_t*, int)>();
    uint16_t arr1[4] = {0x0000, 0x3c00, 0x4000, 0x4400}; // fp16 0.0 1.0 2.0 4.0
    uint16_t arr2[4] = {0x4000, 0x4000, 0x4000, 0x4000}; // fp16 2.0 2.0 2.0 2.0
    uint16_t ret[64] = {0x0000};
    Func(arr1, arr2, ret, 4);
#ifdef KUDNNL_920F
    ASSERT_EQ(ret[32], 0x0000);
    ASSERT_EQ(ret[33], 0x4000);
    ASSERT_EQ(ret[34], 0x4400);
    ASSERT_EQ(ret[35], 0x4800);
#else
    ASSERT_EQ(ret[16], 0x0000);
    ASSERT_EQ(ret[17], 0x4000);
    ASSERT_EQ(ret[18], 0x4400);
    ASSERT_EQ(ret[19], 0x4800);
#endif
}

// test Mul interface
TEST(armcg_test_t, TestMul1)
{
    GTestJitGen cg;
    cg.Mul(cg.x0, cg.x1, cg.x0);
    cg.Ret();
    cg.Ready();

    auto Func = cg.GetCode<int (*)(int, int)>();
    ASSERT_EQ(Func(0, 2), 0);
}

TEST(armcg_test_t, TestMul2)
{
    GTestJitGen cg;
    cg.Mov(cg.x9, 0);
    cg.Whilelt(cg.p0.s, cg.x9, cg.x1);
    cg.Ld1w(cg.z0.s, cg.p0, AdrNoOfs(cg.x0));
    cg.Mul(cg.z0.s, cg.p0, cg.z0.s);
    cg.St1w(cg.z0.s, cg.p0, AdrNoOfs(cg.x0));
    cg.Ret();
    cg.Ready();

    auto Func = cg.GetCode<void (*)(int*, int)>();
    int arr1[] = {0, 1, 2, 3, 4, 5, 6, 7};
    int arr2[] = {0, 1, 4, 9, 16, 25, 36, 49};
    Func(arr1, 8);
    ASSERT_EQ(arr2[0], arr1[0]);
    ASSERT_EQ(arr2[1], arr1[1]);
    ASSERT_EQ(arr2[2], arr1[2]);
    ASSERT_EQ(arr2[3], arr1[3]);
    ASSERT_EQ(arr2[4], arr1[4]);
    ASSERT_EQ(arr2[5], arr1[5]);
    ASSERT_EQ(arr2[6], arr1[6]);
    ASSERT_EQ(arr2[7], arr1[7]);
}

// test Fdiv interface
TEST(armcg_test_t, TestFdiv1)
{
    GTestJitGen cg;
    cg.Ld1(cg.v0.s4[0], AdrNoOfs(cg.x0));
    cg.Ld1(cg.v1.s4[0], AdrNoOfs(cg.x1));
    cg.Fdiv(cg.v2.s4, cg.v0.s4, cg.v1.s4);
    cg.St1(cg.v2.s4[0], AdrNoOfs(cg.x1));
    cg.Ret();
    cg.Ready();

    auto Func = cg.GetCode<void (*)(float*, float*)>();
    float res1[] = {6.0f};
    float res2[] = {3.0f};
    Func(res1, res2);
    ASSERT_EQ(res2[0], 2.0f);
}

TEST(armcg_test_t, TestFdiv2)
{
    GTestJitGen cg;
    cg.Mov(cg.x9, 0);
    cg.Whilelt(cg.p0.s, cg.x9, cg.x1);
    cg.Ld1w(cg.z0.s, cg.p0, AdrNoOfs(cg.x0));
    cg.Fdiv(cg.z0.s, cg.p0, cg.z0.s);
    cg.St1w(cg.z0.s, cg.p0, AdrNoOfs(cg.x0));
    cg.Ret();
    cg.Ready();

    auto Func = cg.GetCode<void (*)(float*, int)>();
    float arr1[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
    Func(arr1, 8);
    ASSERT_EQ(1, arr1[0]);
    ASSERT_EQ(1, arr1[1]);
    ASSERT_EQ(1, arr1[2]);
    ASSERT_EQ(1, arr1[3]);
    ASSERT_EQ(1, arr1[4]);
    ASSERT_EQ(1, arr1[5]);
    ASSERT_EQ(1, arr1[6]);
    ASSERT_EQ(1, arr1[7]);
}

// test Udiv interface
TEST(armcg_test_t, TestUDiv)
{
    GTestJitGen cg;
    cg.Ldr(cg.x9, AdrUimm(cg.x0, 0));
    cg.Ldr(cg.x10, AdrUimm(cg.x0, 8));
    cg.Udiv(cg.x0, cg.x9, cg.x10);
    cg.Udiv(cg.x0, cg.x0, cg.x1);
    cg.Ret();
    cg.Ready();

    auto Func = cg.GetCode<uint64_t(*)(uint64_t*, uint64_t)>();
    uint64_t arr[] = {1000, 2};
    uint64_t scale = 5;
    ASSERT_EQ(Func(arr, scale), 100);
    ARMCG::XReg r1(500);
    ARMCG::XReg r2(800);
    ARMCG::XReg r3(600);
    EXPECT_THROW(cg.Udiv(r1, r2, r3), std::exception);
}

// test Fmad interface
TEST(armcg_test_t, TestFmad)
{
    GTestJitGen cg;
    cg.Mov(cg.x9, 0);
    cg.Whilelt(cg.p0.s, cg.x9, cg.x1);
    cg.Ld1w(cg.z0.s, cg.p0, AdrNoOfs(cg.x0));
    cg.Fmad(cg.z0.s, cg.p0, cg.z0.s, cg.z0.s);
    cg.St1w(cg.z0.s, cg.p0, AdrNoOfs(cg.x0));
    cg.Ret();
    cg.Ready();

    auto Func = cg.GetCode<void (*)(float*, int)>();
    float arr1[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
    float arr2[] = {1.0f, 4.0f, 9.0f, 16.0f, 25.0f, 36.0f, 49.0f, 64.0f};
    Func(arr1, 8);
    ASSERT_EQ(arr2[0] + 1, arr1[0]);
    ASSERT_EQ(arr2[1] + 2, arr1[1]);
    ASSERT_EQ(arr2[2] + 3, arr1[2]);
    ASSERT_EQ(arr2[3] + 4, arr1[3]);
    ASSERT_EQ(arr2[4] + 5, arr1[4]);
    ASSERT_EQ(arr2[5] + 6, arr1[5]);
    ASSERT_EQ(arr2[6] + 7, arr1[6]);
    ASSERT_EQ(arr2[7] + 8, arr1[7]);
}

// test Fmla interface
TEST(armcg_test_t, TestFmla1)
{
    GTestJitGen cg;
    cg.Ld1(VReg4SList(cg.v0.s4), AdrNoOfs(cg.x0));
    cg.Ld1(VReg4SList(cg.v1.s4), AdrNoOfs(cg.x1));
    cg.Fmla(cg.v0.s4, cg.v1.s4, cg.v1.s4);
    cg.Fmla(cg.v0.s4, cg.v1.s4, cg.v1.s4[0]);
    cg.St1(VReg4SList(cg.v0.s4), AdrNoOfs(cg.x2));
    cg.Ret();
    cg.Ready();

    auto Func = cg.GetCode<void (*)(float*, float*, float*)>();
    float arr1[4] = {0.0f};
    float arr2[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    float ret[4] = {0.0f};
    Func(arr1, arr2, ret);
    ASSERT_EQ(ret[0], 2.0f);   // 1.0 x 1.0 + 1.0 x 1.0
    ASSERT_EQ(ret[1], 6.0f);   // 2.0 x 2.0 + 2.0 x 1.0
    ASSERT_EQ(ret[2], 12.0f);  // 3.0 x 3.0 + 3.0 x 1.0
    ASSERT_EQ(ret[3], 20.0f);  // 4.0 x 4.0 + 4.0 x 1.0
}

TEST(armcg_test_t, TestFmla2)
{
    GTestJitGen cg;
    cg.Ptrue(cg.p0.s);
    cg.Ld1w(cg.z0.s, cg.p0, AdrNoOfs(cg.x0));
    cg.Ld1w(cg.z1.s, cg.p0, AdrNoOfs(cg.x0));
    cg.Fmla(cg.z1.s, cg.z0.s, cg.z0.s[1]);
    cg.St1w(cg.z1.s, cg.p0, AdrNoOfs(cg.x0));
    cg.Ret();
    cg.Ready();

    auto Func = cg.GetCode<void (*)(float*)>();
    float arr1[16] = {1.0f, 2.0f, 3.0f, 4.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    float arr2[16] = {3.0f, 6.0f, 9.0f, 12.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    Func(arr1);
    ASSERT_EQ(arr2[0], arr1[0]);
    ASSERT_EQ(arr2[1], arr1[1]);
    ASSERT_EQ(arr2[2], arr1[2]);
    ASSERT_EQ(arr2[3], arr1[3]);
    ASSERT_EQ(arr2[4], arr1[4]);
    ASSERT_EQ(arr2[5], arr1[5]);
    ASSERT_EQ(arr2[6], arr1[6]);
    ASSERT_EQ(arr2[7], arr1[7]);
}

TEST(armcg_test_t, TestFmla3)
{
    GTestJitGen cg;
    cg.Mov(cg.x9, 0);
    cg.Whilelt(cg.p0.s, cg.x9, cg.x1);
    cg.Ld1h(cg.z0.h, cg.p0, AdrNoOfs(cg.x0));
    cg.Ld1h(cg.z1.h, cg.p0, AdrNoOfs(cg.x2));
    cg.Fmla(cg.z0.h, cg.p0, cg.z0.h, cg.z1.h);
    cg.St1h(cg.z0.h, cg.p0, AdrNoOfs(cg.x2));
    cg.Ret();
    cg.Ready();

    auto Func = cg.GetCode<void (*)(uint16_t*, int, uint16_t*)>();
    uint16_t arr1[] = {0x0, 0x0, 0xc600, 0x0, 0x2d93, 0x0, 0xa9ef, 0x0,
        0x5100, 0x0, 0xcd00, 0x0, 0x2d28, 0x0, 0xb831, 0x0};
    uint16_t arr2[] = {0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0};
    Func(arr1, 8, arr2);
    ASSERT_EQ(arr2[0], 0x0);
    ASSERT_EQ(arr2[2], 0xc600);
    ASSERT_EQ(arr2[4], 0x2d93);
    ASSERT_EQ(arr2[6], 0xa9ef);
    ASSERT_EQ(arr2[8], 0x5100);
    ASSERT_EQ(arr2[10], 0xcd00);
    ASSERT_EQ(arr2[12], 0x2d28);
    ASSERT_EQ(arr2[14], 0xb831);
}

TEST(armcg_test_t, TestFmla4)
{
    GTestJitGen cg;
    cg.Mov(cg.x9, 0);
    cg.Whilelt(cg.p0.s, cg.x9, cg.x1);
    cg.Ld1w(cg.z0.s, cg.p0, AdrNoOfs(cg.x0));
    cg.Fmla(cg.z0.s, cg.p0, cg.z0.s, cg.z0.s);
    cg.St1w(cg.z0.s, cg.p0, AdrNoOfs(cg.x0));
    cg.Ret();
    cg.Ready();

    auto Func = cg.GetCode<void (*)(float*, int)>();
    float arr1[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
    float arr2[] = {1.0f, 4.0f, 9.0f, 16.0f, 25.0f, 36.0f, 49.0f, 64.0f};
    Func(arr1, 8);
    ASSERT_EQ(arr2[0] + 1, arr1[0]);
    ASSERT_EQ(arr2[1] + 2, arr1[1]);
    ASSERT_EQ(arr2[2] + 3, arr1[2]);
    ASSERT_EQ(arr2[3] + 4, arr1[3]);
    ASSERT_EQ(arr2[4] + 5, arr1[4]);
    ASSERT_EQ(arr2[5] + 6, arr1[5]);
    ASSERT_EQ(arr2[6] + 7, arr1[6]);
    ASSERT_EQ(arr2[7] + 8, arr1[7]);
}

TEST(armcg_test_t, TestFmla5)
{
    GTestJitGen cg;
    cg.Mov(cg.x9, 0);
    cg.Mov(cg.x10, 8);
    cg.Whilelt(cg.p0.h, cg.x9, cg.x10);
    cg.Ld1h(cg.z0.h, cg.p0, AdrNoOfs(cg.x0));
    cg.Ld1h(cg.z1.h, cg.p0, AdrNoOfs(cg.x0));
    cg.Ld1h(cg.z2.h, cg.p0, AdrNoOfs(cg.x0));
    cg.Fmla(cg.z0.h, cg.z1.h, cg.z2.h[4]);
    cg.St1h(cg.z0.h, cg.p0, AdrNoOfs(cg.x1));
    cg.Ret();
    cg.Ready();

    auto Func = cg.GetCode<void (*)(uint16_t*, uint16_t*)>();
    uint16_t arrF16[] = {0x0, 0xc600, 0x2d93, 0xa9ef, 0x5100, 0xcd00, 0x2d28, 0xb831};
    uint16_t arrOut[8];
    Func(arrF16, arrOut);
    EXPECT_EQ(arrOut[0], 0x0);
    EXPECT_EQ(arrOut[1], 0xdbb0);
    EXPECT_EQ(arrOut[2], 0x4324);
    EXPECT_EQ(arrOut[3], 0xbf9a);
    EXPECT_EQ(arrOut[4], 0x6668);
    EXPECT_EQ(arrOut[5], 0xe268);
    EXPECT_EQ(arrOut[6], 0x429b);
    EXPECT_EQ(arrOut[7], 0xcd5f);
}

TEST(armcg_test_t, TestSmmlaNeon)
{
    GTestJitGen cg;
    cg.Mov(cg.w9, 0);
    cg.Dup(cg.v3.s4, cg.w9);
    cg.Dup(cg.v1.b16, cg.w9);
    cg.Dup(cg.v2.b16, cg.w9);
    cg.Ld1(VReg16BList(cg.v1.b16), AdrPostImm(cg.x1, 16));
    cg.Ld1(VReg16BList(cg.v2.b16), AdrPostImm(cg.x2, 16));
    cg.Smmla(cg.v3.s4, cg.v1.b16, cg.v2.b16);
    cg.St1(VReg4SList(cg.v3.s4), AdrPostImm(cg.x0, 4));
    cg.Ret();
    cg.Ready();
    auto f = cg.GetCode<void (*)(int32_t*, int8_t*, int8_t*)>();
    int8_t src1[64] = {
        1, 0, 0, 0, 0, 0, 0, 0,
        2, 0, 0, 0, 0, 0, 0, 0,
        3, 0, 0, 0, 0, 0, 0, 0,
        4, 0, 0, 0, 0, 0, 0, 0};
    int8_t src2[64] = {
        1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1};
    int32_t dst[8];
    f(dst, src1, src2);
#ifdef KUDNNL_920F
    ASSERT_EQ(dst[0], 1);
#else
    ASSERT_EQ(dst[0], 1);
    ASSERT_EQ(dst[2], 2);
#endif
}

TEST(armcg_test_t, TestUsmmlaNeon)
{
    GTestJitGen cg;
    cg.Mov(cg.w9, 0);
    cg.Dup(cg.v3.s4, cg.w9);
    cg.Dup(cg.v1.b16, cg.w9);
    cg.Dup(cg.v2.b16, cg.w9);
    cg.Ld1(VReg16BList(cg.v1.b16), AdrPostImm(cg.x1, 16));
    cg.Ld1(VReg16BList(cg.v2.b16), AdrPostImm(cg.x2, 16));
    cg.Usmmla(cg.v3.s4, cg.v1.b16, cg.v2.b16);
    cg.St1(VReg4SList(cg.v3.s4), AdrPostImm(cg.x0, 4));
    cg.Ret();
    cg.Ready();
    auto f = cg.GetCode<void (*)(uint32_t*, int8_t*, int8_t*)>();
    int8_t src1[64] = {
        1, 0, 0, 0, 0, 0, 0, 0,
        2, 0, 0, 0, 0, 0, 0, 0,
        3, 0, 0, 0, 0, 0, 0, 0,
        4, 0, 0, 0, 0, 0, 0, 0};
    int8_t src2[64] = {
        1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1};
    uint32_t dst[8];
    f(dst, src1, src2);
#ifdef KUDNNL_920F
    ASSERT_EQ(dst[0], 1);
#else
    ASSERT_EQ(dst[0], 1);
    ASSERT_EQ(dst[2], 2);
#endif
}

TEST(armcg_test_t, TestUmmlaNeon)
{
    GTestJitGen cg;
    cg.Mov(cg.w9, 0);
    cg.Dup(cg.v3.s4, cg.w9);
    cg.Dup(cg.v1.b16, cg.w9);
    cg.Dup(cg.v2.b16, cg.w9);
    cg.Ld1(VReg16BList(cg.v1.b16), AdrPostImm(cg.x1, 16));
    cg.Ld1(VReg16BList(cg.v2.b16), AdrPostImm(cg.x2, 16));
    cg.Usmmla(cg.v3.s4, cg.v1.b16, cg.v2.b16);
    cg.St1(VReg4SList(cg.v3.s4), AdrPostImm(cg.x0, 4));
    cg.Ret();
    cg.Ready();
    auto f = cg.GetCode<void (*)(uint32_t*, uint8_t*, uint8_t*)>();
    uint8_t src1[64] = {
        1, 0, 0, 0, 0, 0, 0, 0,
        2, 0, 0, 0, 0, 0, 0, 0,
        3, 0, 0, 0, 0, 0, 0, 0,
        4, 0, 0, 0, 0, 0, 0, 0};
    uint8_t src2[64] = {
        1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1};
    uint32_t dst[8];
    f(dst, src1, src2);
#ifdef KUDNNL_920F
    ASSERT_EQ(dst[0], 1);
#else
    ASSERT_EQ(dst[0], 1);
    ASSERT_EQ(dst[2], 2);
#endif
}

TEST(armcg_test_t, TestUzpDNeon)
{
    GTestJitGen cg;
    cg.Mov(cg.x9, 1);
    cg.Mov(cg.x10, 2);
    cg.Mov(cg.x11, 3);
    cg.Mov(cg.x12, 4);
    cg.Mov(cg.v0.d2[0], cg.x9);
    cg.Mov(cg.v0.d2[1], cg.x10);
    cg.Mov(cg.v1.d2[0], cg.x11);
    cg.Mov(cg.v1.d2[1], cg.x12);
    cg.Uzp1(cg.v3.d2, cg.v1.d2, cg.v2.d2);
    cg.Uzp2(cg.v3.d2, cg.v1.d2, cg.v2.d2);
    cg.Ret();
    cg.Ready();

    auto f = cg.GetCode<void(*)()>();
    f();
}

TEST(armcg_test_t, TestUzp4SNeon)
{
    GTestJitGen cg;
    cg.Ld1(VReg4SList(cg.v1.s4), AdrNoOfs(cg.x1));
    cg.Ld1(VReg4SList(cg.v2.s4), AdrNoOfs(cg.x2));

    cg.Uzp1(cg.v3.s4, cg.v1.s4, cg.v2.s4);
    cg.Uzp2(cg.v4.s4, cg.v1.s4, cg.v2.s4);
    cg.St1(cg.v4.s4[0], AdrPostImm(cg.x0, 4));
    cg.Ret();
    cg.Ready();

    auto f = cg.GetCode<void(*)(float*, float*, float*)>();
    float arr1[4] = {1, 2, 3, 4};
    float arr2[4] = {9, 6, 7 ,8};
    float arr3[8] = {};
    f(arr3, arr1, arr2);
    ASSERT_EQ(arr3[0], 2);
}

TEST(armcg_test_t, TestUzp2SNeon)
{
    GTestJitGen cg;
    cg.Ld1(VReg4SList(cg.v1.s4), AdrNoOfs(cg.x1));
    cg.Ld1(VReg4SList(cg.v2.s4), AdrNoOfs(cg.x2));

    cg.Uzp1(cg.v3.s2, cg.v1.s2, cg.v2.s2);
    cg.Uzp2(cg.v4.s2, cg.v1.s2, cg.v2.s2);
    cg.St1(VReg4SList(cg.v4.s4), AdrNoOfs(cg.x0));
    cg.Ret();
    cg.Ready();

    auto f = cg.GetCode<void(*)(float*, float*, float*)>();
    float arr1[4] = {1, 2, 3, 4};
    float arr2[4] = {9, 6, 7 ,8};
    float arr3[8] = {};
    f(arr3, arr1, arr2);
    ASSERT_EQ(arr3[0], 2);
}

TEST(armcg_test_t, TestTrnDSNeon)
{
    GTestJitGen cg;
    cg.Mov(cg.x9, 1);
    cg.Mov(cg.x10, 2);
    cg.Mov(cg.x11, 3);
    cg.Mov(cg.x12, 4);
    cg.Mov(cg.v0.d2[0], cg.x9);
    cg.Mov(cg.v0.d2[1], cg.x10);
    cg.Mov(cg.v1.d2[0], cg.x11);
    cg.Mov(cg.v1.d2[1], cg.x12);
    cg.Trn1(cg.v2.d2, cg.v0.d2, cg.v1.d2);
    cg.Trn2(cg.v3.d2, cg.v0.d2, cg.v1.d2);
    cg.Ret();
    cg.Ready();
    auto f = cg.GetCode<void(*)()>();
    f();
}

// test Madd interface
TEST(armcg_test_t, TestMadd2)
{
    GTestJitGen cg;
    cg.Ldr(cg.x9, AdrUimm(cg.x0, 0));
    cg.Ldr(cg.x10, AdrUimm(cg.x0, 0));
    cg.Madd(cg.x0, cg.x9, cg.x9, cg.x10);
    cg.Ret();
    cg.Ready();

    auto Func = cg.GetCode<int (*)(int*)>();
    int res[] = {2};
    ASSERT_EQ(Func(res), 6);
}

// test Fmls interface
TEST(armcg_test_t, TestFmls2)
{
    GTestJitGen cg;
    cg.Mov(cg.x9, 0);
    cg.Whilelt(cg.p0.s, cg.x9, cg.x1);
    cg.Ld1w(cg.z0.s, cg.p0, AdrNoOfs(cg.x0));
    cg.Fmls(cg.z0.s, cg.p0, cg.z0.s, cg.z0.s);
    cg.St1w(cg.z0.s, cg.p0, AdrNoOfs(cg.x0));
    cg.Ret();
    cg.Ready();

    auto Func = cg.GetCode<void (*)(float*, int)>();
    float arr1[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
    float arr2[] = {0.0f, -2.0f, -6.0f, -12.0f, -20.0f, -30.0f, -42.0f, -56.0f};
    Func(arr1, 8);
    ASSERT_EQ(arr2[0], arr1[0]);
    ASSERT_EQ(arr2[1], arr1[1]);
    ASSERT_EQ(arr2[2], arr1[2]);
    ASSERT_EQ(arr2[3], arr1[3]);
    ASSERT_EQ(arr2[4], arr1[4]);
    ASSERT_EQ(arr2[5], arr1[5]);
    ASSERT_EQ(arr2[6], arr1[6]);
    ASSERT_EQ(arr2[7], arr1[7]);
}

//test FMLS(indexed) single-precision
TEST(armcg_test_t, TestFmls3)
{
    GTestJitGen cg;
    cg.Ptrue(cg.p0.s);
    cg.Ld1w(cg.z0.s, cg.p0, AdrNoOfs(cg.x0));
    cg.Ld1w(cg.z1.s, cg.p0, AdrNoOfs(cg.x1));
    cg.Ld1w(cg.z2.s, cg.p0, AdrNoOfs(cg.x2));
    cg.Fmls(cg.z2.s, cg.z1.s, cg.z0.s[0]);
    cg.St1w(cg.z2.s, cg.p0, AdrNoOfs(cg.x2));
    cg.Ret();
    cg.Ready();

    auto Func = cg.GetCode<void (*)(float*, float*, float*)>();
    float arr1[16] = {2.0f, 2.0f, 3.0f, 4.0f, 1.0f, 2.0f, 3.0f, 4.0f};
    float arr2[16] = {2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f};
    float arr3[16] = {6.0f, 8.0f, 15.0f, 22.0f, 15.0f, 19.0f, 27.0f, 44.0f};
    Func(arr1, arr2, arr3);
    ASSERT_EQ(arr3[0], 2.0f);
    ASSERT_EQ(arr3[1], 2.0f);
    ASSERT_EQ(arr3[2], 7.0f);
    ASSERT_EQ(arr3[3], 12.0f);
    ASSERT_EQ(arr3[4], 9.0f);
    ASSERT_EQ(arr3[5], 12.0f);
    ASSERT_EQ(arr3[6], 19.0f);
    ASSERT_EQ(arr3[7], 35.0f);
}

// test Fnmsb interface
TEST(armcg_test_t, TestFnmsb)
{
    GTestJitGen cg;
    cg.Mov(cg.x9, 0);
    cg.Whilelt(cg.p0.s, cg.x9, cg.x1);
    cg.Ld1w(cg.z0.s, cg.p0, AdrNoOfs(cg.x0));
    cg.Fnmsb(cg.z0.s, cg.p0, cg.z0.s, cg.z0.s);
    cg.St1w(cg.z0.s, cg.p0, AdrNoOfs(cg.x0));
    cg.Ret();
    cg.Ready();

    auto Func = cg.GetCode<void (*)(float*, int)>();
    float arr1[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
    float arr2[] = {0.0f, 2.0f, 6.0f, 12.0f, 20.0f, 30.0f, 42.0f, 56.0f};
    Func(arr1, 8);
    ASSERT_EQ(arr2[0], arr1[0]);
    ASSERT_EQ(arr2[1], arr1[1]);
    ASSERT_EQ(arr2[2], arr1[2]);
    ASSERT_EQ(arr2[3], arr1[3]);
    ASSERT_EQ(arr2[4], arr1[4]);
    ASSERT_EQ(arr2[5], arr1[5]);
    ASSERT_EQ(arr2[6], arr1[6]);
    ASSERT_EQ(arr2[7], arr1[7]);
}

// test Fexpa interface
TEST(armcg_test_t, TestFexpa)
{
    GTestJitGen cg;
    cg.Mov(cg.x9, 0);
    cg.Whilelt(cg.p0.s, cg.x9, cg.x1);
    cg.Ld1w(cg.z0.s, cg.p0, AdrNoOfs(cg.x0));
    cg.Fexpa(cg.z0.s, cg.z0.s);
    cg.St1w(cg.z0.s, cg.p0, AdrNoOfs(cg.x0));
    cg.Ret();
    cg.Ready();

    auto Func = cg.GetCode<void (*)(float*, int)>();
    float arr1[] = {196740.562f, 196740.562f, 196740.562f, 196740.562f,
        196740.562f, 196740.562f, 196740.562f, 196740.562f};
    float arr2[] = {47.25843667f, 47.25843667f, 47.25843667f, 47.25843667f,
        47.25843667f, 47.25843667f, 47.25843667f, 47.25843667f};
    Func(arr1, 8);
    ASSERT_EQ(arr2[0], arr1[0]);
    ASSERT_EQ(arr2[1], arr1[1]);
    ASSERT_EQ(arr2[2], arr1[2]);
    ASSERT_EQ(arr2[3], arr1[3]);
    ASSERT_EQ(arr2[4], arr1[4]);
    ASSERT_EQ(arr2[5], arr1[5]);
    ASSERT_EQ(arr2[6], arr1[6]);
    ASSERT_EQ(arr2[7], arr1[7]);
}

// test Fsqrt interface
TEST(armcg_test_t, TestFsqrt1)
{
    GTestJitGen cg;
    cg.Ld1(cg.v0.s4[0], AdrNoOfs(cg.x0));
    cg.Fsqrt(cg.v1.s4, cg.v0.s4);
    cg.St1(cg.v1.s4[0], AdrNoOfs(cg.x0));
    cg.Ret();
    cg.Ready();

    auto Func = cg.GetCode<void (*)(float*)>();
    float res[] = {4.0f};
    Func(res);
    ASSERT_EQ(res[0], 2);
}

TEST(armcg_test_t, TestFsqrt2)
{
    GTestJitGen cg;
    cg.Mov(cg.x9, 0);
    cg.Whilelt(cg.p0.s, cg.x9, cg.x1);
    cg.Ld1w(cg.z0.s, cg.p0, AdrNoOfs(cg.x0));
    cg.Fsqrt(cg.z0.s, cg.p0, cg.z0.s);
    cg.St1w(cg.z0.s, cg.p0, AdrNoOfs(cg.x0));
    cg.Ret();
    cg.Ready();

    auto Func = cg.GetCode<void (*)(float*, int)>();
    float arr1[] = {0.0f, 9.0f, 25.0f, 0.046f, 40.0f, 20.0f, 0.0805929987f, 0.524f};
    float arr2[] = {0.0f, 9.0f, 25.0f, 0.046f, 40.0f, 20.0f, 0.0805929987f, 0.524f};
    Func(arr1, 8);
    ASSERT_EQ((float)sqrt(arr2[0]), arr1[0]);
    ASSERT_EQ((float)sqrt(arr2[1]), arr1[1]);
    ASSERT_EQ((float)sqrt(arr2[2]), arr1[2]);
    ASSERT_EQ((float)sqrt(arr2[3]), arr1[3]);
    ASSERT_EQ((float)sqrt(arr2[4]), arr1[4]);
    ASSERT_EQ((float)sqrt(arr2[5]), arr1[5]);
    ASSERT_EQ((float)sqrt(arr2[6]), arr1[6]);
    ASSERT_EQ((float)sqrt(arr2[7]), arr1[7]);
}

// test Fabs interface
TEST(armcg_test_t, TestFabs)
{
    GTestJitGen cg;
    cg.Mov(cg.x9, 0);
    cg.Whilelt(cg.p0.s, cg.x9, cg.x1);
    cg.Ld1w(cg.z0.s, cg.p0, AdrNoOfs(cg.x0));
    cg.Fabs(cg.z0.s, cg.p0, cg.z0.s);
    cg.St1w(cg.z0.s, cg.p0, AdrNoOfs(cg.x0));
    cg.Ret();
    cg.Ready();

    auto Func = cg.GetCode<void (*)(float*, int)>();
    float arr1[] = {1.0f, -2.0f, -3.0f, -4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
    float arr2[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
    Func(arr1, 8);
    ASSERT_EQ(arr2[0], arr1[0]);
    ASSERT_EQ(arr2[1], arr1[1]);
    ASSERT_EQ(arr2[2], arr1[2]);
    ASSERT_EQ(arr2[3], arr1[3]);
    ASSERT_EQ(arr2[4], arr1[4]);
    ASSERT_EQ(arr2[5], arr1[5]);
    ASSERT_EQ(arr2[6], arr1[6]);
    ASSERT_EQ(arr2[7], arr1[7]);
}

// test Fneg interface
TEST(armcg_test_t, TestFneg)
{
    GTestJitGen cg;
    cg.Mov(cg.x9, 0);
    cg.Whilelt(cg.p0.s, cg.x9, cg.x1);
    cg.Ld1w(cg.z0.s, cg.p0, AdrNoOfs(cg.x0));
    cg.Fneg(cg.z0.s, cg.p0, cg.z0.s);
    cg.St1w(cg.z0.s, cg.p0, AdrNoOfs(cg.x0));
    cg.Ret();
    cg.Ready();

    auto Func = cg.GetCode<void (*)(float*, int)>();
    float arr1[] = {1.0f, -2.0f, -3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
    float arr2[] = {-1.0f, 2.0f, 3.0f, -4.0f, -5.0f, -6.0f, -7.0f, -8.0f};
    Func(arr1, 8);
    ASSERT_EQ(arr2[0], arr1[0]);
    ASSERT_EQ(arr2[1], arr1[1]);
    ASSERT_EQ(arr2[2], arr1[2]);
    ASSERT_EQ(arr2[3], arr1[3]);
    ASSERT_EQ(arr2[4], arr1[4]);
    ASSERT_EQ(arr2[5], arr1[5]);
    ASSERT_EQ(arr2[6], arr1[6]);
    ASSERT_EQ(arr2[7], arr1[7]);
}

// test Fmax interface
TEST(armcg_test_t, TestFmax1)
{
    GTestJitGen cg;
    cg.Dup(cg.v0.s4, cg.w0);
    cg.Dup(cg.v1.s4, cg.w1);
    cg.Fmax(cg.v1.s4, cg.v1.s4, cg.v0.s4);
    cg.Mov(cg.w0, cg.v1.s4[0]);
    cg.Ret();
    cg.Ready();

    auto Func = cg.GetCode<int (*)(int, int)>();
    int res1 = 4;
    int res2 = 3;
    ASSERT_EQ(Func(res1, res2), 4);
}

TEST(armcg_test_t, TestFmax3)
{
    GTestJitGen cg;
    cg.Mov(cg.x9, 0);
    cg.Whilelt(cg.p0.s, cg.x9, cg.x1);
    cg.Ld1w(cg.z0.s, cg.p0, AdrNoOfs(cg.x0));
    cg.Fmax(cg.z0.s, cg.p0, 0.0f);
    cg.St1w(cg.z0.s, cg.p0, AdrNoOfs(cg.x0));
    cg.Ret();
    cg.Ready();

    auto Func = cg.GetCode<void (*)(float*, int)>();
    float arr1[] = {-1.0f, -2.0f, -3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
    float arr2[] = {0.0f, 0.0f, 0.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
    Func(arr1, 8);
    ASSERT_EQ(arr2[0], arr1[0]);
    ASSERT_EQ(arr2[1], arr1[1]);
    ASSERT_EQ(arr2[2], arr1[2]);
    ASSERT_EQ(arr2[3], arr1[3]);
    ASSERT_EQ(arr2[4], arr1[4]);
    ASSERT_EQ(arr2[5], arr1[5]);
    ASSERT_EQ(arr2[6], arr1[6]);
    ASSERT_EQ(arr2[7], arr1[7]);
}

// test Fmaxnm interface
TEST(armcg_test_t, TestFmaxnm1)
{
    GTestJitGen cg;
    cg.Dup(cg.v0.s4, cg.w0);
    cg.Dup(cg.v1.s4, cg.w1);
    cg.Fmaxnm(cg.v1.s4, cg.v1.s4, cg.v0.s4);
    cg.Mov(cg.w0, cg.v1.s4[0]);
    cg.Ret();
    cg.Ready();

    auto Func = cg.GetCode<int (*)(int, int)>();
    int res1 = 4;
    int res2 = 3;
    ASSERT_EQ(Func(res1, res2), 4);
}

TEST(armcg_test_t, TestFmaxnm3)
{
    GTestJitGen cg;
    cg.Mov(cg.x9, 0);
    cg.Whilelt(cg.p0.s, cg.x9, cg.x1);
    cg.Ld1w(cg.z0.s, cg.p0, AdrNoOfs(cg.x0));
    cg.Fmaxnm(cg.z0.s, cg.p0, 1.0f);
    cg.St1w(cg.z0.s, cg.p0, AdrNoOfs(cg.x0));
    cg.Ret();
    cg.Ready();

    auto Func = cg.GetCode<void (*)(float*, int)>();
    float arr1[] = {-1.0f, -2.0f, -3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
    float arr2[] = {1.0f, 1.0f, 1.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
    Func(arr1, 8);
    ASSERT_EQ(arr2[0], arr1[0]);
    ASSERT_EQ(arr2[1], arr1[1]);
    ASSERT_EQ(arr2[2], arr1[2]);
    ASSERT_EQ(arr2[3], arr1[3]);
    ASSERT_EQ(arr2[4], arr1[4]);
    ASSERT_EQ(arr2[5], arr1[5]);
    ASSERT_EQ(arr2[6], arr1[6]);
    ASSERT_EQ(arr2[7], arr1[7]);
}

// test Smax interface
TEST(armcg_test_t, TestSmax1)
{
    GTestJitGen cg;
    cg.Dup(cg.v0.b16, cg.w0);
    cg.Dup(cg.v1.b16, cg.w1);
    cg.Smax(cg.v1.b16, cg.v1.b16, cg.v0.b16);
    cg.Mov(cg.w0, cg.v1.s4[0]);
    cg.Ret();
    cg.Ready();

    auto Func = cg.GetCode<int (*)(int, int)>();
    int res1 = 0;
    int res2 = -1;
    ASSERT_EQ(Func(res1, res2), 0);
}

TEST(armcg_test_t, TestSmax2)
{
    GTestJitGen cg;
    cg.Dup(cg.v0.s4, cg.w0);
    cg.Dup(cg.v1.s4, cg.w1);
    cg.Smax(cg.v1.s4, cg.v1.s4, cg.v0.s4);
    cg.Mov(cg.w0, cg.v1.s4[0]);
    cg.Ret();
    cg.Ready();

    auto Func = cg.GetCode<int (*)(int, int)>();
    int res1 = 4;
    int res2 = 3;
    ASSERT_EQ(Func(res1, res2), 4);
}

TEST(armcg_test_t, TestSmax4)
{
    GTestJitGen cg;
    cg.Mov(cg.x9, 0);
    cg.Whilelt(cg.p0.s, cg.x9, cg.x1);
    cg.Ld1w(cg.z0.s, cg.p0, AdrNoOfs(cg.x0));
    cg.Smax(cg.z0.s, 2);
    cg.St1w(cg.z0.s, cg.p0, AdrNoOfs(cg.x0));
    cg.Ret();
    cg.Ready();

    auto Func = cg.GetCode<void (*)(int*, int)>();
    int arr1[] = {-1, -2, -3, 4, 5, 6, 7, 8};
    int arr2[] = {2, 2, 2, 4, 5, 6, 7, 8};
    Func(arr1, 8);
    ASSERT_EQ(arr2[0], arr1[0]);
    ASSERT_EQ(arr2[1], arr1[1]);
    ASSERT_EQ(arr2[2], arr1[2]);
    ASSERT_EQ(arr2[3], arr1[3]);
    ASSERT_EQ(arr2[4], arr1[4]);
    ASSERT_EQ(arr2[5], arr1[5]);
    ASSERT_EQ(arr2[6], arr1[6]);
    ASSERT_EQ(arr2[7], arr1[7]);
}

// test Fmin interface
TEST(armcg_test_t, TestFmin1)
{
    GTestJitGen cg;
    cg.Dup(cg.v0.s4, cg.w0);
    cg.Dup(cg.v1.s4, cg.w1);
    cg.Fmin(cg.v1.s4, cg.v1.s4, cg.v0.s4);
    cg.Mov(cg.w0, cg.v1.s4[0]);
    cg.Ret();
    cg.Ready();

    auto Func = cg.GetCode<int (*)(int, int)>();
    int res1 = 4;
    int res2 = 3;
    ASSERT_EQ(Func(res1, res2), 3);
}

TEST(armcg_test_t, TestFminnm2)
{
    GTestJitGen cg;
    cg.Mov(cg.x9, 0);
    cg.Whilelt(cg.p0.s, cg.x9, cg.x1);
    cg.Ld1w(cg.z0.s, cg.p0, AdrNoOfs(cg.x0));
    cg.Fminnm(cg.z0.s, cg.p0, 1.0f);
    cg.St1w(cg.z0.s, cg.p0, AdrNoOfs(cg.x0));
    cg.Ret();
    cg.Ready();

    auto Func = cg.GetCode<void (*)(float*, int)>();
    float arr1[] = {-1.0f, -2.0f, -3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
    float arr2[] = {-1.0f, -2.0f, -3.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
    Func(arr1, 8);
    ASSERT_EQ(arr2[0], arr1[0]);
    ASSERT_EQ(arr2[1], arr1[1]);
    ASSERT_EQ(arr2[2], arr1[2]);
    ASSERT_EQ(arr2[3], arr1[3]);
    ASSERT_EQ(arr2[4], arr1[4]);
    ASSERT_EQ(arr2[5], arr1[5]);
    ASSERT_EQ(arr2[6], arr1[6]);
    ASSERT_EQ(arr2[7], arr1[7]);
}

// test Smin interface
TEST(armcg_test_t, TestSmin1)
{
    GTestJitGen cg;
    cg.Dup(cg.v0.s4, cg.w0);
    cg.Dup(cg.v1.s4, cg.w1);
    cg.Smin(cg.v1.s4, cg.v1.s4, cg.v0.s4);
    cg.Mov(cg.w0, cg.v1.s4[0]);
    cg.Ret();
    cg.Ready();

    auto Func = cg.GetCode<int (*)(int, int)>();
    int res1 = 4;
    int res2 = 3;
    ASSERT_EQ(Func(res1, res2), 3);
}

TEST(armcg_test_t, TestSmin3)
{
    GTestJitGen cg;
    cg.Mov(cg.x9, 0);
    cg.Whilelt(cg.p0.s, cg.x9, cg.x1);
    cg.Ld1w(cg.z0.s, cg.p0, AdrNoOfs(cg.x0));
    cg.Smin(cg.z0.s, 2);
    cg.St1w(cg.z0.s, cg.p0, AdrNoOfs(cg.x0));
    cg.Ret();
    cg.Ready();

    auto Func = cg.GetCode<void (*)(int*, int)>();
    int arr1[] = {-1, -2, -3, 4, 5, 6, 7, 8};
    int arr2[] = {-1, -2, -3, 2, 2, 2, 2, 2};
    Func(arr1, 8);
    ASSERT_EQ(arr2[0], arr1[0]);
    ASSERT_EQ(arr2[1], arr1[1]);
    ASSERT_EQ(arr2[2], arr1[2]);
    ASSERT_EQ(arr2[3], arr1[3]);
    ASSERT_EQ(arr2[4], arr1[4]);
    ASSERT_EQ(arr2[5], arr1[5]);
    ASSERT_EQ(arr2[6], arr1[6]);
    ASSERT_EQ(arr2[7], arr1[7]);
}

// test Umin interface
TEST(armcg_test_t, TestUmin1)
{
    GTestJitGen cg;
    cg.Dup(cg.v0.b16, cg.w0);
    cg.Dup(cg.v1.b16, cg.w1);
    cg.Umin(cg.v1.b16, cg.v1.b16, cg.v0.b16);
    cg.Mov(cg.w0, cg.v1.s4[0]);
    cg.Ret();
    cg.Ready();

    auto Func = cg.GetCode<int (*)(int, int)>();
    int res1 = 0;
    int res2 = -1;
    ASSERT_EQ(Func(res1, res2), 0);
}

TEST(armcg_test_t, TestUmin4)
{
    GTestJitGen cg;
    cg.Mov(cg.x9, 0);
    cg.Whilelt(cg.p0.s, cg.x9, cg.x1);
    cg.Ld1w(cg.z0.s, cg.p0, AdrNoOfs(cg.x0));
    cg.Umin(cg.z0.s, 2);
    cg.St1w(cg.z0.s, cg.p0, AdrNoOfs(cg.x0));
    cg.Ret();
    cg.Ready();

    auto Func = cg.GetCode<void (*)(int*, int)>();
    int arr1[] = {-1, -2, -3, 4, 5, 6, 7, 8};
    int arr2[] = {2, 2, 2, 2, 2, 2, 2, 2};
    Func(arr1, 8);
    ASSERT_EQ(arr2[0], arr1[0]);
    ASSERT_EQ(arr2[1], arr1[1]);
    ASSERT_EQ(arr2[2], arr1[2]);
    ASSERT_EQ(arr2[3], arr1[3]);
    ASSERT_EQ(arr2[4], arr1[4]);
    ASSERT_EQ(arr2[5], arr1[5]);
    ASSERT_EQ(arr2[6], arr1[6]);
    ASSERT_EQ(arr2[7], arr1[7]);
}

// test Frinti interface
TEST(armcg_test_t, TestFrinti1)
{
    GTestJitGen cg;
    cg.Ld1(cg.v0.s4[0], AdrNoOfs(cg.x0));
    cg.Frinti(cg.v1.s4, cg.v0.s4);
    cg.St1(cg.v1.s4[0], AdrNoOfs(cg.x0));
    cg.Ret();
    cg.Ready();

    auto Func = cg.GetCode<void (*)(float*)>();
    float res1[] = {4.5};
    Func(res1);
    ASSERT_EQ(res1[0], 4);
}

TEST(armcg_test_t, TestFrinti2)
{
    GTestJitGen cg;
    cg.Mov(cg.x9, 0);
    cg.Whilelt(cg.p0.s, cg.x9, cg.x1);
    cg.Ld1w(cg.z0.s, cg.p0, AdrNoOfs(cg.x0));
    cg.Frinti(cg.z0.s, cg.p0, cg.z0.s);
    cg.St1w(cg.z0.s, cg.p0, AdrNoOfs(cg.x0));
    cg.Ret();
    cg.Ready();

    auto Func = cg.GetCode<void (*)(float*, int)>();
    float arr1[] = {-1.3f, -2.2f, -3.1f, 4.5f, 5.6f, 6.7f, 7.8f, 8.9f};
    float arr2[] = {-1.0f, -2.0f, -3.0f, 4.0f, 6.0f, 7.0f, 8.0f, 9.0f};
    Func(arr1, 8);
    ASSERT_EQ(arr2[0], arr1[0]);
    ASSERT_EQ(arr2[1], arr1[1]);
    ASSERT_EQ(arr2[2], arr1[2]);
    ASSERT_EQ(arr2[3], arr1[3]);
    ASSERT_EQ(arr2[4], arr1[4]);
    ASSERT_EQ(arr2[5], arr1[5]);
    ASSERT_EQ(arr2[6], arr1[6]);
    ASSERT_EQ(arr2[7], arr1[7]);
}

// test Frintm interface
TEST(armcg_test_t, TestFrintm)
{
    GTestJitGen cg;
    cg.Mov(cg.x9, 0);
    cg.Whilelt(cg.p0.s, cg.x9, cg.x1);
    cg.Ld1w(cg.z0.s, cg.p0, AdrNoOfs(cg.x0));
    cg.Frintm(cg.z0.s, cg.p0, cg.z0.s);
    cg.St1w(cg.z0.s, cg.p0, AdrNoOfs(cg.x0));
    cg.Ret();
    cg.Ready();

    auto Func = cg.GetCode<void (*)(float*, int)>();
    float arr1[] = {-1.3f, -2.2f, -3.1f, 4.5f, 5.6f, 6.7f, 7.8f, 8.9f};
    float arr2[] = {-2.0f, -3.0f, -4.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
    Func(arr1, 8);
    ASSERT_EQ(arr2[0], arr1[0]);
    ASSERT_EQ(arr2[1], arr1[1]);
    ASSERT_EQ(arr2[2], arr1[2]);
    ASSERT_EQ(arr2[3], arr1[3]);
    ASSERT_EQ(arr2[4], arr1[4]);
    ASSERT_EQ(arr2[5], arr1[5]);
    ASSERT_EQ(arr2[6], arr1[6]);
    ASSERT_EQ(arr2[7], arr1[7]);
}

// test Frintn interface
TEST(armcg_test_t, TestFrintn)
{
    GTestJitGen cg;
    cg.Mov(cg.x9, 0);
    cg.Whilelt(cg.p0.s, cg.x9, cg.x1);
    cg.Ld1w(cg.z0.s, cg.p0, AdrNoOfs(cg.x0));
    cg.Frintn(cg.z0.s, cg.p0, cg.z0.s);
    cg.St1w(cg.z0.s, cg.p0, AdrNoOfs(cg.x0));
    cg.Ret();
    cg.Ready();

    auto Func = cg.GetCode<void (*)(float*, int)>();
    float arr1[] = {-1.3f, -2.2f, -3.1f, 4.5f, 5.6f, 6.7f, 7.8f, 8.9f};
    float arr2[] = {-1.0f, -2.0f, -3.0f, 4.0f, 6.0f, 7.0f, 8.0f, 9.0f};
    Func(arr1, 8);
    ASSERT_EQ(arr2[0], arr1[0]);
    ASSERT_EQ(arr2[1], arr1[1]);
    ASSERT_EQ(arr2[2], arr1[2]);
    ASSERT_EQ(arr2[3], arr1[3]);
    ASSERT_EQ(arr2[4], arr1[4]);
    ASSERT_EQ(arr2[5], arr1[5]);
    ASSERT_EQ(arr2[6], arr1[6]);
    ASSERT_EQ(arr2[7], arr1[7]);
}

TEST(armcg_test_t, TestUunpklo)
{
    GTestJitGen cg;
    cg.Mov(cg.x9, 0);
    cg.Mov(cg.x10, 4);
    cg.Whilelt(cg.p0.h, cg.x9, cg.x10);
    cg.Fmov(cg.z0.h, 0.0f);
    cg.Fmov(cg.z1.s, 0.0f);
    cg.Fmov(cg.z2.s, 0.0f);
    cg.Ld1h(cg.z0.h, cg.p0, AdrNoOfs(cg.x0));
    cg.Uunpklo(cg.z1.s, cg.z0.h);
    cg.Whilelt(cg.p1.s, cg.x9, cg.x10);
    cg.St1w(cg.z1.s, cg.p1, AdrNoOfs(cg.x1));

    cg.Ret();
    cg.Ready();

    auto f = cg.GetCode<void (*)(uint16_t*, uint32_t*)>();
    uint16_t arr[4] = {0x0000, 0x3c00, 0x4000, 0x4400}; // fp16 0.0 1.0 2.0 4.0
    uint32_t ret[4] = {0};
    f(arr, ret);
    ASSERT_EQ(ret[0], 0x00000000);
    ASSERT_EQ(ret[1], 0x00003c00);
    ASSERT_EQ(ret[2], 0x00004000);
    ASSERT_EQ(ret[3], 0x00004400);
}

// sme test
#ifdef KUDNNL_920F

TEST(armcg_test_t, TestBfmopa)
{
    GTestJitGen cg;
    cg.Smstart(StreamMode::SMZA);
    cg.Zero({cg.za0});
    cg.Mov(cg.x9, 0);
    cg.Mov(cg.x10, 4);
    cg.Whilelt(cg.p0.h, cg.x9, cg.x10);
    cg.Ld1h(cg.z0.h, cg.p0.h, AdrNoOfs(cg.x0));
    cg.Ld1h(cg.z1.h, cg.p0.h, AdrNoOfs(cg.x1));
    cg.Bfmopa(cg.za0.s, cg.p0, cg.p0, cg.z0.h, cg.z1.h);
    cg.Mov(cg.x10, 2);
    cg.Whilelt(cg.p1.s, cg.x9, cg.x10);
    cg.Mov(cg.w12, 0);
    cg.Mov(cg.x11, 0);
    cg.St1w(cg.za0h.s, cg.w12, 0, cg.p1, AdrScSc(cg.x2, cg.x11, ShMod::LSL, 2));
    cg.Add(cg.x11, cg.x11, 2);
    cg.St1w(cg.za0h.s, cg.w12, 1, cg.p1, AdrScSc(cg.x2, cg.x11, ShMod::LSL, 2));

    cg.Smstop(StreamMode::SMZA);
    cg.Ret();
    cg.Ready();

    auto f = cg.GetCode<void (*)(uint16_t*, uint16_t*, float*)>();
    uint16_t arr1[4] = {0x3f80, 0x4000, 0x3f80, 0x4000}; // {1.0, 2.0, 1.0, 2.0}
    uint16_t arr2[4] = {0x3f80, 0x4000, 0x3f80, 0x4000}; // {1.0, 2.0, 1.0, 2.0}
    float ret1[4] = {0};
    f(arr1, arr2, ret1);
    ASSERT_EQ(ret1[0], 5.0);   // 1.0 * 1.0 + 2.0 * 2.0
    ASSERT_EQ(ret1[1], 5.0);
    ASSERT_EQ(ret1[2], 5.0);
    ASSERT_EQ(ret1[3], 5.0);
}

TEST(armcg_test_t, TestMC1)
{
    GTestJitGen cg;
     // x0 ~ x7: parameters passed into function
    // x9 ~ x15: tmp reg used inside function
    // x0: return value by function
    cg.Smstart(StreamMode::SMZA);
    cg.Zero({cg.za0});
    // for this demo we only have 4 elements in each row/column
    cg.Mov(cg.x9, 0);
    cg.Mov(cg.x10, 4);
    cg.Whilelt(cg.p0.s, cg.x9, cg.x10);
    cg.Whilelt(cg.p1.s, cg.x9, cg.x10);
    cg.Ld1w(cg.z0.s, cg.p0, AdrNoOfs(cg.x0));   // z0 = [a00, a10, a20, a30, ......0]
    cg.Ld1w(cg.z1.s, cg.p1, AdrNoOfs(cg.x1));   // z1 = [b00, b01, b02, b03, ......0]
    cg.Fmopa(cg.za0.s, cg.p0, cg.p1, cg.z0.s, cg.z1.s);   // za0 = outer product of ax0&b0x
    cg.Add(cg.x0, cg.x0, 16);
    cg.Add(cg.x1, cg.x1, 16);
    cg.Ld1w(cg.z0.s, cg.p0, AdrNoOfs(cg.x0));   // z0 = [a01, a11, a21, a31, ......0]
    cg.Ld1w(cg.z1.s, cg.p1, AdrNoOfs(cg.x1));   // z1 = [b10, b11, b12, b13, ......0]
    cg.Fmopa(cg.za0.s, cg.p0, cg.p1, cg.z0.s, cg.z1.s);   // za0 = outer product of ax0&b0x + ax1&b1x

    // now we have matmul result, which is stord in row-first order
    cg.Mov(cg.w12, 0);
    cg.Mov(cg.x11, 0);
    // option1: go through each row slice to store values directly into dst array
    cg.St1w(cg.za0h.s, cg.w12, 0, cg.p1, AdrScSc(cg.x2, cg.x11, ShMod::LSL, 2));
    cg.Add(cg.x11, cg.x11, 4);
    cg.St1w(cg.za0h.s, cg.w12, 1, cg.p1, AdrScSc(cg.x2, cg.x11, ShMod::LSL, 2));
    // option 2: go through each row slice to move values into zreg and store
    cg.Mova(cg.z2.s, cg.p1, cg.za0h.s, cg.w12, 2);
    cg.Mova(cg.z3.s, cg.p1, cg.za0h.s, cg.w12, 3);
    cg.Add(cg.x11, cg.x2, 32);
    cg.St1w(cg.z2.s, cg.p1, AdrNoOfs(cg.x11));
    cg.Add(cg.x11, cg.x2, 48);
    cg.St1w(cg.z3.s, cg.p1, AdrNoOfs(cg.x11));

    cg.Smstop(StreamMode::SMZA);
    cg.Ret();
    cg.Ready();

    // you can cast it to any function with parameters as long as you know what you are doing
    auto f = cg.GetCode<void (*)(float*, float*, float*)>();
    // for convenience matA is stored in column-first order
    float matA_4x2[] = {0.0f, -6.0f, 0.087f, -0.046f, 40.0f, -20.0f, 0.081f, -0.524f};
    // for convenience matB is stored in row-first order
    float matB_2x4[] = {-0.081f, 100.0f, -10.0f, 0.816f, -0.843f, 88.278f, 22.355f, 44.620f};
    float matC_4x4[16];
    // matA,B,C are stored in x0, x1, x2, according to arm calling convention
    f(matA_4x2, matB_2x4, matC_4x4);

    ASSERT_FLOAT_EQ(matA_4x2[0]*matB_2x4[0] + matA_4x2[4]*matB_2x4[4], matC_4x4[0]);
    ASSERT_FLOAT_EQ(matA_4x2[0]*matB_2x4[1] + matA_4x2[4]*matB_2x4[5], matC_4x4[1]);
    ASSERT_FLOAT_EQ(matA_4x2[0]*matB_2x4[2] + matA_4x2[4]*matB_2x4[6], matC_4x4[2]);
    ASSERT_FLOAT_EQ(matA_4x2[0]*matB_2x4[3] + matA_4x2[4]*matB_2x4[7], matC_4x4[3]);
    ASSERT_FLOAT_EQ(matA_4x2[1]*matB_2x4[0] + matA_4x2[5]*matB_2x4[4], matC_4x4[4]);
    ASSERT_FLOAT_EQ(matA_4x2[1]*matB_2x4[1] + matA_4x2[5]*matB_2x4[5], matC_4x4[5]);
    ASSERT_FLOAT_EQ(matA_4x2[1]*matB_2x4[2] + matA_4x2[5]*matB_2x4[6], matC_4x4[6]);
    ASSERT_FLOAT_EQ(matA_4x2[1]*matB_2x4[3] + matA_4x2[5]*matB_2x4[7], matC_4x4[7]);
    ASSERT_FLOAT_EQ(matA_4x2[2]*matB_2x4[0] + matA_4x2[6]*matB_2x4[4], matC_4x4[8]);
    ASSERT_FLOAT_EQ(matA_4x2[2]*matB_2x4[1] + matA_4x2[6]*matB_2x4[5], matC_4x4[9]);
    ASSERT_FLOAT_EQ(matA_4x2[2]*matB_2x4[2] + matA_4x2[6]*matB_2x4[6], matC_4x4[10]);
    ASSERT_FLOAT_EQ(matA_4x2[2]*matB_2x4[3] + matA_4x2[6]*matB_2x4[7], matC_4x4[11]);
    ASSERT_FLOAT_EQ(matA_4x2[3]*matB_2x4[0] + matA_4x2[7]*matB_2x4[4], matC_4x4[12]);
    ASSERT_FLOAT_EQ(matA_4x2[3]*matB_2x4[1] + matA_4x2[7]*matB_2x4[5], matC_4x4[13]);
    ASSERT_FLOAT_EQ(matA_4x2[3]*matB_2x4[2] + matA_4x2[7]*matB_2x4[6], matC_4x4[14]);
    ASSERT_FLOAT_EQ(matA_4x2[3]*matB_2x4[3] + matA_4x2[7]*matB_2x4[7], matC_4x4[15]);
}

TEST(armcg_test_t, TestMC2)
{
    GTestJitGen cg;
    cg.Smstart(StreamMode::SMZA);
    cg.Zero({cg.za0});
    cg.Mov(cg.x9, 0);
    cg.Mov(cg.x10, 4);
    cg.Whilelt(cg.p0.s, cg.x9, cg.x10);
    cg.Whilelt(cg.p1.s, cg.x9, cg.x10);
    cg.Ld1w(cg.z0.s, cg.p0, AdrNoOfs(cg.x0));
    cg.Ld1w(cg.z1.s, cg.p1, AdrNoOfs(cg.x1));
    cg.Fmopa(cg.za0.s, cg.p0, cg.p1, cg.z0.s, cg.z1.s);
    cg.Add(cg.x0, cg.x0, 16);
    cg.Add(cg.x1, cg.x1, 16);
    cg.Ld1w(cg.z0.s, cg.p0, AdrNoOfs(cg.x0));
    cg.Ld1w(cg.z1.s, cg.p1, AdrNoOfs(cg.x1));
    cg.Fmopa(cg.za0.s, cg.p0, cg.p1, cg.z0.s, cg.z1.s);

    // retrieve in column-first order.
    cg.Mov(cg.w12, 0);
    cg.Mov(cg.x11, 0);
    cg.St1w(cg.za0v.s, cg.w12, 0, cg.p1, AdrScSc(cg.x2, cg.x11, ShMod::LSL, 2));
    cg.Add(cg.x11, cg.x11, 4);
    cg.St1w(cg.za0v.s, cg.w12, 1, cg.p1, AdrScSc(cg.x2, cg.x11, ShMod::LSL, 2));
    cg.Mova(cg.z2.s, cg.p1, cg.za0v.s, cg.w12, 2);
    cg.Mova(cg.z3.s, cg.p1, cg.za0v.s, cg.w12, 3);
    cg.Add(cg.x11, cg.x2, 32);
    cg.St1w(cg.z2.s, cg.p1, AdrNoOfs(cg.x11));
    cg.Add(cg.x11, cg.x2, 48);
    cg.St1w(cg.z3.s, cg.p1, AdrNoOfs(cg.x11));

    cg.Smstop(StreamMode::SMZA);
    cg.Ret();
    cg.Ready();

    auto f = cg.GetCode<void (*)(float*, float*, float*)>();
    // for convenience matA is stored in column-first order
    float matA_4x2[] = {0.0f, -6.0f, 0.087f, -0.046f, 40.0f, -20.0f, 0.081f, -0.524f};
    // for convenience matB is stored in row-first order
    float matB_2x4[] = {-0.081f, 100.0f, -10.0f, 0.816f, -0.843f, 88.278f, 22.355f, 44.620f};
    float matC_4x4[16];
    f(matA_4x2, matB_2x4, matC_4x4);

    ASSERT_FLOAT_EQ(matA_4x2[0]*matB_2x4[0] + matA_4x2[4]*matB_2x4[4], matC_4x4[0]);
    ASSERT_FLOAT_EQ(matA_4x2[1]*matB_2x4[0] + matA_4x2[5]*matB_2x4[4], matC_4x4[1]);
    ASSERT_FLOAT_EQ(matA_4x2[2]*matB_2x4[0] + matA_4x2[6]*matB_2x4[4], matC_4x4[2]);
    ASSERT_FLOAT_EQ(matA_4x2[3]*matB_2x4[0] + matA_4x2[7]*matB_2x4[4], matC_4x4[3]);
    ASSERT_FLOAT_EQ(matA_4x2[0]*matB_2x4[1] + matA_4x2[4]*matB_2x4[5], matC_4x4[4]);
    ASSERT_FLOAT_EQ(matA_4x2[1]*matB_2x4[1] + matA_4x2[5]*matB_2x4[5], matC_4x4[5]);
    ASSERT_FLOAT_EQ(matA_4x2[2]*matB_2x4[1] + matA_4x2[6]*matB_2x4[5], matC_4x4[6]);
    ASSERT_FLOAT_EQ(matA_4x2[3]*matB_2x4[1] + matA_4x2[7]*matB_2x4[5], matC_4x4[7]);
    ASSERT_FLOAT_EQ(matA_4x2[0]*matB_2x4[2] + matA_4x2[4]*matB_2x4[6], matC_4x4[8]);
    ASSERT_FLOAT_EQ(matA_4x2[1]*matB_2x4[2] + matA_4x2[5]*matB_2x4[6], matC_4x4[9]);
    ASSERT_FLOAT_EQ(matA_4x2[2]*matB_2x4[2] + matA_4x2[6]*matB_2x4[6], matC_4x4[10]);
    ASSERT_FLOAT_EQ(matA_4x2[3]*matB_2x4[2] + matA_4x2[7]*matB_2x4[6], matC_4x4[11]);
    ASSERT_FLOAT_EQ(matA_4x2[0]*matB_2x4[3] + matA_4x2[4]*matB_2x4[7], matC_4x4[12]);
    ASSERT_FLOAT_EQ(matA_4x2[1]*matB_2x4[3] + matA_4x2[5]*matB_2x4[7], matC_4x4[13]);
    ASSERT_FLOAT_EQ(matA_4x2[2]*matB_2x4[3] + matA_4x2[6]*matB_2x4[7], matC_4x4[14]);
    ASSERT_FLOAT_EQ(matA_4x2[3]*matB_2x4[3] + matA_4x2[7]*matB_2x4[7], matC_4x4[15]);
}

TEST(armcg_test_t, TestMCByteStoreLoadV)
{
    GTestJitGen cg;
    cg.Smstart(StreamMode::SMZA);
    cg.Zero({cg.za0});
    cg.Mov(cg.x9, 0);
    cg.Mov(cg.x10, 4);
    cg.Whilelt(cg.p0.b, cg.x9, cg.x10);
    cg.Mov(cg.w12, 0);
    cg.Ld1b(cg.za0v.b, cg.w12, 0, cg.p0, AdrScSc(cg.x0, cg.x9));   // z0 = [a01, a02, a03, a04]
    cg.Mov(cg.x9, 0);
    cg.Mov(cg.w12, 0);
    cg.St1b(cg.za0v.b, cg.w12, 0, cg.p0, AdrScSc(cg.x1, cg.x9));
    cg.Smstop(StreamMode::SMZA);
    cg.Ret();
    cg.Ready();
 
    auto f = cg.GetCode<void (*)(unsigned char*, unsigned char*)>();
    unsigned char matA_4x1[] = {1, 2, 3, 4};
    unsigned char matC_4x1[4];
    f(matA_4x1, matC_4x1);
 
    ASSERT_FLOAT_EQ(matA_4x1[0], matC_4x1[0]);
    ASSERT_FLOAT_EQ(matA_4x1[1], matC_4x1[1]);
    ASSERT_FLOAT_EQ(matA_4x1[2], matC_4x1[2]);
    ASSERT_FLOAT_EQ(matA_4x1[3], matC_4x1[3]);
}

TEST(armcg_test_t, TestMCByteStoreLoadH)
{
    GTestJitGen cg;
    cg.Smstart(StreamMode::SMZA);
    cg.Zero({cg.za0});
    cg.Mov(cg.x9, 0);
    cg.Mov(cg.x10, 4);
    cg.Whilelt(cg.p0.b, cg.x9, cg.x10);
    cg.Mov(cg.w12, 0);
    cg.Ld1b(cg.za0h.b, cg.w12, 0, cg.p0, AdrScSc(cg.x0, cg.x9));   // z0 = [a01, a02, a03, a04]
    cg.Mov(cg.x9, 0);
    cg.Mov(cg.w12, 0);
    cg.St1b(cg.za0h.b, cg.w12, 0, cg.p0, AdrScSc(cg.x1, cg.x9));
    cg.Smstop(StreamMode::SMZA);
    cg.Ret();
    cg.Ready();
 
    auto f = cg.GetCode<void (*)(unsigned char*, unsigned char*)>();
    unsigned char matA_4x1[] = {1, 2, 3, 4};
    unsigned char matC_4x1[4];
    f(matA_4x1, matC_4x1);
 
    ASSERT_FLOAT_EQ(matA_4x1[0], matC_4x1[0]);
    ASSERT_FLOAT_EQ(matA_4x1[1], matC_4x1[1]);
    ASSERT_FLOAT_EQ(matA_4x1[2], matC_4x1[2]);
    ASSERT_FLOAT_EQ(matA_4x1[3], matC_4x1[3]);
}

TEST(armcg_test_t, TestMCFmop2Way)
{
    GTestJitGen cg;
    cg.Smstart(StreamMode::SMZA);
    cg.Zero({cg.za0});
    cg.Mov(cg.x9, 0);
    cg.Mov(cg.x10, 8);
    cg.Whilelt(cg.p0.h, cg.x9, cg.x10);
    cg.Whilelt(cg.p1.h, cg.x9, cg.x10);
    cg.Ld1h(cg.z0.h, cg.p0, AdrNoOfs(cg.x0));
    cg.Ld1h(cg.z1.h, cg.p1, AdrNoOfs(cg.x1));
    cg.Fmopa(cg.za0.s, cg.p0, cg.p1, cg.z0.h, cg.z1.h);

    cg.Mov(cg.x10, 4);
    cg.Whilelt(cg.p3.s, cg.x9, cg.x10);
    cg.Mov(cg.w12, 0);
    cg.Mov(cg.x11, 0);
    cg.St1w(cg.za0h.s, cg.w12, 0, cg.p3, AdrScSc(cg.x2, cg.x11, ShMod::LSL, 2));
    cg.Add(cg.x11, cg.x11, 4);
    cg.St1w(cg.za0h.s, cg.w12, 1, cg.p3, AdrScSc(cg.x2, cg.x11, ShMod::LSL, 2));
    cg.Add(cg.x11, cg.x11, 4);
    cg.St1w(cg.za0h.s, cg.w12, 2, cg.p3, AdrScSc(cg.x2, cg.x11, ShMod::LSL, 2));
    cg.Add(cg.x11, cg.x11, 4);
    cg.St1w(cg.za0h.s, cg.w12, 3, cg.p3, AdrScSc(cg.x2, cg.x11, ShMod::LSL, 2));

    cg.Smstop(StreamMode::SMZA);
    cg.Ret();
    cg.Ready();

    auto f = cg.GetCode<void (*)(uint16_t*, uint16_t*, float*)>();
    uint16_t vecRow[] = {0x0, 0xc600, 0x2d93, 0xa9ef, 0x5100, 0xcd00, 0x2d28, 0xb831}; 
    uint16_t vecCol[] = { 0x5589, 0x4db2, 0x5195, 0x0, 0x0, 0x4900, 0xc200, 0x2d2e};
    float matC_4x4[16];
    f(vecRow, vecCol, matC_4x4);

    EXPECT_FLOAT_EQ(DotProduct2WayF16(vecRow, 0, vecCol, 0), matC_4x4[0]);
    EXPECT_FLOAT_EQ(DotProduct2WayF16(vecRow, 0, vecCol, 2), matC_4x4[1]);
    EXPECT_FLOAT_EQ(DotProduct2WayF16(vecRow, 0, vecCol, 4), matC_4x4[2]);
    EXPECT_FLOAT_EQ(DotProduct2WayF16(vecRow, 0, vecCol, 6), matC_4x4[3]);
    EXPECT_FLOAT_EQ(DotProduct2WayF16(vecRow, 2, vecCol, 0), matC_4x4[4]);
    EXPECT_FLOAT_EQ(DotProduct2WayF16(vecRow, 2, vecCol, 2), matC_4x4[5]);
    EXPECT_FLOAT_EQ(DotProduct2WayF16(vecRow, 2, vecCol, 4), matC_4x4[6]);
    EXPECT_FLOAT_EQ(DotProduct2WayF16(vecRow, 2, vecCol, 6), matC_4x4[7]);
    EXPECT_FLOAT_EQ(DotProduct2WayF16(vecRow, 4, vecCol, 0), matC_4x4[8]);
    EXPECT_FLOAT_EQ(DotProduct2WayF16(vecRow, 4, vecCol, 2), matC_4x4[9]);
    EXPECT_FLOAT_EQ(DotProduct2WayF16(vecRow, 4, vecCol, 4), matC_4x4[10]);
    EXPECT_FLOAT_EQ(DotProduct2WayF16(vecRow, 4, vecCol, 6), matC_4x4[11]);
    EXPECT_FLOAT_EQ(DotProduct2WayF16(vecRow, 6, vecCol, 0), matC_4x4[12]);
    EXPECT_FLOAT_EQ(DotProduct2WayF16(vecRow, 6, vecCol, 2), matC_4x4[13]);
    EXPECT_FLOAT_EQ(DotProduct2WayF16(vecRow, 6, vecCol, 4), matC_4x4[14]);
    EXPECT_FLOAT_EQ(DotProduct2WayF16(vecRow, 6, vecCol, 6), matC_4x4[15]);
}

TEST(armcg_test_t, TestMCSmop4Way)
{
    GTestJitGen cg;
    cg.Smstart(StreamMode::SMZA);
    cg.Zero({cg.za0});
    cg.Mov(cg.x9, 0);
    cg.Mov(cg.x10, 16);
    cg.Whilelt(cg.p0.b, cg.x9, cg.x10);
    cg.Whilelt(cg.p1.b, cg.x9, cg.x10);
    cg.Ld1b(cg.z0.b, cg.p0, AdrNoOfs(cg.x0));
    cg.Ld1b(cg.z1.b, cg.p1, AdrNoOfs(cg.x1));
    cg.Smopa(cg.za0.s, cg.p0, cg.p1, cg.z0.b, cg.z1.b);

    cg.Mov(cg.x10, 4);
    cg.Whilelt(cg.p3.s, cg.x9, cg.x10);
    cg.Mov(cg.w12, 0);
    cg.Mov(cg.x11, 0);
    cg.St1w(cg.za0h.s, cg.w12, 0, cg.p3, AdrScSc(cg.x2, cg.x11, ShMod::LSL, 2));
    cg.Add(cg.x11, cg.x11, 4);
    cg.St1w(cg.za0h.s, cg.w12, 1, cg.p3, AdrScSc(cg.x2, cg.x11, ShMod::LSL, 2));
    cg.Add(cg.x11, cg.x11, 4);
    cg.St1w(cg.za0h.s, cg.w12, 2, cg.p3, AdrScSc(cg.x2, cg.x11, ShMod::LSL, 2));
    cg.Add(cg.x11, cg.x11, 4);
    cg.St1w(cg.za0h.s, cg.w12, 3, cg.p3, AdrScSc(cg.x2, cg.x11, ShMod::LSL, 2));

    cg.Smops(cg.za0.s, cg.p0, cg.p1, cg.z0.b, cg.z1.b);
    cg.Mov(cg.w12, 0);
    cg.Mov(cg.x11, 0);
    cg.St1w(cg.za0h.s, cg.w12, 0, cg.p3, AdrScSc(cg.x3, cg.x11, ShMod::LSL, 2));
    cg.Add(cg.x11, cg.x11, 4);
    cg.St1w(cg.za0h.s, cg.w12, 1, cg.p3, AdrScSc(cg.x3, cg.x11, ShMod::LSL, 2));
    cg.Add(cg.x11, cg.x11, 4);
    cg.St1w(cg.za0h.s, cg.w12, 2, cg.p3, AdrScSc(cg.x3, cg.x11, ShMod::LSL, 2));
    cg.Add(cg.x11, cg.x11, 4);
    cg.St1w(cg.za0h.s, cg.w12, 3, cg.p3, AdrScSc(cg.x3, cg.x11, ShMod::LSL, 2));

    cg.Smstop(StreamMode::SMZA);
    cg.Ret();
    cg.Ready();

    auto f = cg.GetCode<void (*)(int8_t*, int8_t*, int*, int*)>();
    int8_t vecRow[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
    int8_t vecCol[] = {1, 1, 1, 1, -1, -1, -1, -1, 2, 2, 2, 2, -2, -2, -2, -2};
    int matC_4x4[16];
    int matD_4x4[16] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
    f(vecRow, vecCol, matC_4x4, matD_4x4);

    ASSERT_EQ(DotProduct4Way(vecRow, 0, vecCol, 0), matC_4x4[0]);
    ASSERT_EQ(DotProduct4Way(vecRow, 0, vecCol, 4), matC_4x4[1]);
    ASSERT_EQ(DotProduct4Way(vecRow, 0, vecCol, 8), matC_4x4[2]);
    ASSERT_EQ(DotProduct4Way(vecRow, 0, vecCol, 12), matC_4x4[3]);
    ASSERT_EQ(DotProduct4Way(vecRow, 4, vecCol, 0), matC_4x4[4]);
    ASSERT_EQ(DotProduct4Way(vecRow, 4, vecCol, 4), matC_4x4[5]);
    ASSERT_EQ(DotProduct4Way(vecRow, 4, vecCol, 8), matC_4x4[6]);
    ASSERT_EQ(DotProduct4Way(vecRow, 4, vecCol, 12), matC_4x4[7]);
    ASSERT_EQ(DotProduct4Way(vecRow, 8, vecCol, 0), matC_4x4[8]);
    ASSERT_EQ(DotProduct4Way(vecRow, 8, vecCol, 4), matC_4x4[9]);
    ASSERT_EQ(DotProduct4Way(vecRow, 8, vecCol, 8), matC_4x4[10]);
    ASSERT_EQ(DotProduct4Way(vecRow, 8, vecCol, 12), matC_4x4[11]);
    ASSERT_EQ(DotProduct4Way(vecRow, 12, vecCol, 0), matC_4x4[12]);
    ASSERT_EQ(DotProduct4Way(vecRow, 12, vecCol, 4), matC_4x4[13]);
    ASSERT_EQ(DotProduct4Way(vecRow, 12, vecCol, 8), matC_4x4[14]);
    ASSERT_EQ(DotProduct4Way(vecRow, 12, vecCol, 12), matC_4x4[15]);
    for (int i = 0; i != 16; ++i) {
        ASSERT_EQ(0, matD_4x4[i]);
    }
}

TEST(armcg_test_t, TestMCSumop4Way)
{
    GTestJitGen cg;
    cg.Smstart(StreamMode::SMZA);
    cg.Zero({cg.za0});
    cg.Mov(cg.x9, 0);
    cg.Mov(cg.x10, 16);
    cg.Whilelt(cg.p0.b, cg.x9, cg.x10);
    cg.Whilelt(cg.p1.b, cg.x9, cg.x10);
    cg.Ld1b(cg.z0.b, cg.p0, AdrNoOfs(cg.x0));
    cg.Ld1b(cg.z1.b, cg.p1, AdrNoOfs(cg.x1));
    cg.Sumopa(cg.za0.s, cg.p0, cg.p1, cg.z0.b, cg.z1.b);

    cg.Mov(cg.x10, 4);
    cg.Whilelt(cg.p3.s, cg.x9, cg.x10);
    cg.Mov(cg.w12, 0);
    cg.Mov(cg.x11, 0);
    cg.St1w(cg.za0h.s, cg.w12, 0, cg.p3, AdrScSc(cg.x2, cg.x11, ShMod::LSL, 2));
    cg.Add(cg.x11, cg.x11, 4);
    cg.St1w(cg.za0h.s, cg.w12, 1, cg.p3, AdrScSc(cg.x2, cg.x11, ShMod::LSL, 2));
    cg.Add(cg.x11, cg.x11, 4);
    cg.St1w(cg.za0h.s, cg.w12, 2, cg.p3, AdrScSc(cg.x2, cg.x11, ShMod::LSL, 2));
    cg.Add(cg.x11, cg.x11, 4);
    cg.St1w(cg.za0h.s, cg.w12, 3, cg.p3, AdrScSc(cg.x2, cg.x11, ShMod::LSL, 2));

    cg.Sumops(cg.za0.s, cg.p0, cg.p1, cg.z0.b, cg.z1.b);
    cg.Mov(cg.w12, 0);
    cg.Mov(cg.x11, 0);
    cg.St1w(cg.za0h.s, cg.w12, 0, cg.p3, AdrScSc(cg.x3, cg.x11, ShMod::LSL, 2));
    cg.Add(cg.x11, cg.x11, 4);
    cg.St1w(cg.za0h.s, cg.w12, 1, cg.p3, AdrScSc(cg.x3, cg.x11, ShMod::LSL, 2));
    cg.Add(cg.x11, cg.x11, 4);
    cg.St1w(cg.za0h.s, cg.w12, 2, cg.p3, AdrScSc(cg.x3, cg.x11, ShMod::LSL, 2));
    cg.Add(cg.x11, cg.x11, 4);
    cg.St1w(cg.za0h.s, cg.w12, 3, cg.p3, AdrScSc(cg.x3, cg.x11, ShMod::LSL, 2));

    cg.Smstop(StreamMode::SMZA);
    cg.Ret();
    cg.Ready();

    auto f = cg.GetCode<void (*)(int8_t*, uint8_t*, int*, int*)>();
    int8_t vecRow[] = {0, 1, 2, 3, -4, -5, -6, -7, 8, 9, 10, 11, -12, -13, -14, -15};
    uint8_t vecCol[] = {1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4};
    int matC_4x4[16];
    int matD_4x4[16] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
    f(vecRow, vecCol, matC_4x4, matD_4x4);

    ASSERT_EQ(DotProduct4Way(vecRow, 0, vecCol, 0), matC_4x4[0]);
    ASSERT_EQ(DotProduct4Way(vecRow, 0, vecCol, 4), matC_4x4[1]);
    ASSERT_EQ(DotProduct4Way(vecRow, 0, vecCol, 8), matC_4x4[2]);
    ASSERT_EQ(DotProduct4Way(vecRow, 0, vecCol, 12), matC_4x4[3]);
    ASSERT_EQ(DotProduct4Way(vecRow, 4, vecCol, 0), matC_4x4[4]);
    ASSERT_EQ(DotProduct4Way(vecRow, 4, vecCol, 4), matC_4x4[5]);
    ASSERT_EQ(DotProduct4Way(vecRow, 4, vecCol, 8), matC_4x4[6]);
    ASSERT_EQ(DotProduct4Way(vecRow, 4, vecCol, 12), matC_4x4[7]);
    ASSERT_EQ(DotProduct4Way(vecRow, 8, vecCol, 0), matC_4x4[8]);
    ASSERT_EQ(DotProduct4Way(vecRow, 8, vecCol, 4), matC_4x4[9]);
    ASSERT_EQ(DotProduct4Way(vecRow, 8, vecCol, 8), matC_4x4[10]);
    ASSERT_EQ(DotProduct4Way(vecRow, 8, vecCol, 12), matC_4x4[11]);
    ASSERT_EQ(DotProduct4Way(vecRow, 12, vecCol, 0), matC_4x4[12]);
    ASSERT_EQ(DotProduct4Way(vecRow, 12, vecCol, 4), matC_4x4[13]);
    ASSERT_EQ(DotProduct4Way(vecRow, 12, vecCol, 8), matC_4x4[14]);
    ASSERT_EQ(DotProduct4Way(vecRow, 12, vecCol, 12), matC_4x4[15]);
    for (int i = 0; i != 16; ++i) {
        ASSERT_EQ(0, matD_4x4[i]);
    }
}

TEST(armcg_test_t, TestMCUmop4Way)
{
    GTestJitGen cg;
    cg.Smstart(StreamMode::SMZA);
    cg.Zero({cg.za0});
    cg.Mov(cg.x9, 0);
    cg.Mov(cg.x10, 16);
    cg.Whilelt(cg.p0.b, cg.x9, cg.x10);
    cg.Whilelt(cg.p1.b, cg.x9, cg.x10);
    cg.Ld1b(cg.z0.b, cg.p0, AdrNoOfs(cg.x0));
    cg.Ld1b(cg.z1.b, cg.p1, AdrNoOfs(cg.x1));
    cg.Umopa(cg.za0.s, cg.p0, cg.p1, cg.z0.b, cg.z1.b);

    cg.Mov(cg.x10, 4);
    cg.Whilelt(cg.p3.s, cg.x9, cg.x10);
    cg.Mov(cg.w12, 0);
    cg.Mov(cg.x11, 0);
    cg.St1w(cg.za0h.s, cg.w12, 0, cg.p3, AdrScSc(cg.x2, cg.x11, ShMod::LSL, 2));
    cg.Add(cg.x11, cg.x11, 4);
    cg.St1w(cg.za0h.s, cg.w12, 1, cg.p3, AdrScSc(cg.x2, cg.x11, ShMod::LSL, 2));
    cg.Add(cg.x11, cg.x11, 4);
    cg.St1w(cg.za0h.s, cg.w12, 2, cg.p3, AdrScSc(cg.x2, cg.x11, ShMod::LSL, 2));
    cg.Add(cg.x11, cg.x11, 4);
    cg.St1w(cg.za0h.s, cg.w12, 3, cg.p3, AdrScSc(cg.x2, cg.x11, ShMod::LSL, 2));

    cg.Umops(cg.za0.s, cg.p0, cg.p1, cg.z0.b, cg.z1.b);
    cg.Mov(cg.w12, 0);
    cg.Mov(cg.x11, 0);
    cg.St1w(cg.za0h.s, cg.w12, 0, cg.p3, AdrScSc(cg.x3, cg.x11, ShMod::LSL, 2));
    cg.Add(cg.x11, cg.x11, 4);
    cg.St1w(cg.za0h.s, cg.w12, 1, cg.p3, AdrScSc(cg.x3, cg.x11, ShMod::LSL, 2));
    cg.Add(cg.x11, cg.x11, 4);
    cg.St1w(cg.za0h.s, cg.w12, 2, cg.p3, AdrScSc(cg.x3, cg.x11, ShMod::LSL, 2));
    cg.Add(cg.x11, cg.x11, 4);
    cg.St1w(cg.za0h.s, cg.w12, 3, cg.p3, AdrScSc(cg.x3, cg.x11, ShMod::LSL, 2));

    cg.Smstop(StreamMode::SMZA);
    cg.Ret();
    cg.Ready();

    auto f = cg.GetCode<void (*)(uint8_t*, uint8_t*, int*, int*)>();
    uint8_t vecRow[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
    uint8_t vecCol[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
    int matC_4x4[16];
    int matD_4x4[16] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
    f(vecRow, vecCol, matC_4x4, matD_4x4);

    ASSERT_EQ(DotProduct4Way(vecRow, 0, vecCol, 0), matC_4x4[0]);
    ASSERT_EQ(DotProduct4Way(vecRow, 0, vecCol, 4), matC_4x4[1]);
    ASSERT_EQ(DotProduct4Way(vecRow, 0, vecCol, 8), matC_4x4[2]);
    ASSERT_EQ(DotProduct4Way(vecRow, 0, vecCol, 12), matC_4x4[3]);
    ASSERT_EQ(DotProduct4Way(vecRow, 4, vecCol, 0), matC_4x4[4]);
    ASSERT_EQ(DotProduct4Way(vecRow, 4, vecCol, 4), matC_4x4[5]);
    ASSERT_EQ(DotProduct4Way(vecRow, 4, vecCol, 8), matC_4x4[6]);
    ASSERT_EQ(DotProduct4Way(vecRow, 4, vecCol, 12), matC_4x4[7]);
    ASSERT_EQ(DotProduct4Way(vecRow, 8, vecCol, 0), matC_4x4[8]);
    ASSERT_EQ(DotProduct4Way(vecRow, 8, vecCol, 4), matC_4x4[9]);
    ASSERT_EQ(DotProduct4Way(vecRow, 8, vecCol, 8), matC_4x4[10]);
    ASSERT_EQ(DotProduct4Way(vecRow, 8, vecCol, 12), matC_4x4[11]);
    ASSERT_EQ(DotProduct4Way(vecRow, 12, vecCol, 0), matC_4x4[12]);
    ASSERT_EQ(DotProduct4Way(vecRow, 12, vecCol, 4), matC_4x4[13]);
    ASSERT_EQ(DotProduct4Way(vecRow, 12, vecCol, 8), matC_4x4[14]);
    ASSERT_EQ(DotProduct4Way(vecRow, 12, vecCol, 12), matC_4x4[15]);
    for (int i = 0; i != 16; ++i) {
        ASSERT_EQ(0, matD_4x4[i]);
    }
}

TEST(armcg_test_t, TestMCUsmop4Way)
{
    GTestJitGen cg;
    cg.Smstart(StreamMode::SMZA);
    cg.Zero({cg.za0});
    cg.Mov(cg.x9, 0);
    cg.Mov(cg.x10, 16);
    cg.Whilelt(cg.p0.b, cg.x9, cg.x10);
    cg.Whilelt(cg.p1.b, cg.x9, cg.x10);
    cg.Ld1b(cg.z0.b, cg.p0, AdrNoOfs(cg.x0));
    cg.Ld1b(cg.z1.b, cg.p1, AdrNoOfs(cg.x1));
    cg.Usmopa(cg.za0.s, cg.p0, cg.p1, cg.z0.b, cg.z1.b);

    cg.Mov(cg.x10, 4);
    cg.Whilelt(cg.p3.s, cg.x9, cg.x10);
    cg.Mov(cg.w12, 0);
    cg.Mov(cg.x11, 0);
    cg.St1w(cg.za0h.s, cg.w12, 0, cg.p3, AdrScSc(cg.x2, cg.x11, ShMod::LSL, 2));
    cg.Add(cg.x11, cg.x11, 4);
    cg.St1w(cg.za0h.s, cg.w12, 1, cg.p3, AdrScSc(cg.x2, cg.x11, ShMod::LSL, 2));
    cg.Add(cg.x11, cg.x11, 4);
    cg.St1w(cg.za0h.s, cg.w12, 2, cg.p3, AdrScSc(cg.x2, cg.x11, ShMod::LSL, 2));
    cg.Add(cg.x11, cg.x11, 4);
    cg.St1w(cg.za0h.s, cg.w12, 3, cg.p3, AdrScSc(cg.x2, cg.x11, ShMod::LSL, 2));

    cg.Usmops(cg.za0.s, cg.p0, cg.p1, cg.z0.b, cg.z1.b);
    cg.Mov(cg.w12, 0);
    cg.Mov(cg.x11, 0);
    cg.St1w(cg.za0h.s, cg.w12, 0, cg.p3, AdrScSc(cg.x3, cg.x11, ShMod::LSL, 2));
    cg.Add(cg.x11, cg.x11, 4);
    cg.St1w(cg.za0h.s, cg.w12, 1, cg.p3, AdrScSc(cg.x3, cg.x11, ShMod::LSL, 2));
    cg.Add(cg.x11, cg.x11, 4);
    cg.St1w(cg.za0h.s, cg.w12, 2, cg.p3, AdrScSc(cg.x3, cg.x11, ShMod::LSL, 2));
    cg.Add(cg.x11, cg.x11, 4);
    cg.St1w(cg.za0h.s, cg.w12, 3, cg.p3, AdrScSc(cg.x3, cg.x11, ShMod::LSL, 2));

    cg.Smstop(StreamMode::SMZA);
    cg.Ret();
    cg.Ready();

    auto f = cg.GetCode<void (*)(uint8_t*, int8_t*, int*, int*)>();
    uint8_t vecRow[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14,15};
    int8_t vecCol[] = {1, 1, 1, 1, -1, -1, -1, -1, 2, 2, 2, 2, -2, -2, -2, -2};
    int matC_4x4[16];
    int matD_4x4[16] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
    f(vecRow, vecCol, matC_4x4, matD_4x4);

    ASSERT_EQ(DotProduct4Way(vecRow, 0, vecCol, 0), matC_4x4[0]);
    ASSERT_EQ(DotProduct4Way(vecRow, 0, vecCol, 4), matC_4x4[1]);
    ASSERT_EQ(DotProduct4Way(vecRow, 0, vecCol, 8), matC_4x4[2]);
    ASSERT_EQ(DotProduct4Way(vecRow, 0, vecCol, 12), matC_4x4[3]);
    ASSERT_EQ(DotProduct4Way(vecRow, 4, vecCol, 0), matC_4x4[4]);
    ASSERT_EQ(DotProduct4Way(vecRow, 4, vecCol, 4), matC_4x4[5]);
    ASSERT_EQ(DotProduct4Way(vecRow, 4, vecCol, 8), matC_4x4[6]);
    ASSERT_EQ(DotProduct4Way(vecRow, 4, vecCol, 12), matC_4x4[7]);
    ASSERT_EQ(DotProduct4Way(vecRow, 8, vecCol, 0), matC_4x4[8]);
    ASSERT_EQ(DotProduct4Way(vecRow, 8, vecCol, 4), matC_4x4[9]);
    ASSERT_EQ(DotProduct4Way(vecRow, 8, vecCol, 8), matC_4x4[10]);
    ASSERT_EQ(DotProduct4Way(vecRow, 8, vecCol, 12), matC_4x4[11]);
    ASSERT_EQ(DotProduct4Way(vecRow, 12, vecCol, 0), matC_4x4[12]);
    ASSERT_EQ(DotProduct4Way(vecRow, 12, vecCol, 4), matC_4x4[13]);
    ASSERT_EQ(DotProduct4Way(vecRow, 12, vecCol, 8), matC_4x4[14]);
    ASSERT_EQ(DotProduct4Way(vecRow, 12, vecCol, 12), matC_4x4[15]);
    for (int i = 0; i != 16; ++i) {
        ASSERT_EQ(0, matD_4x4[i]);
    }
}

TEST(armcg_test_t, TestMCMovaB)
{
    GTestJitGen cg;
    cg.Smstart(StreamMode::SMZA);
    cg.Zero({cg.za0});
    cg.Mov(cg.x9, 0);
    cg.Mov(cg.x10, 16);
    cg.Whilelt(cg.p0.b, cg.x9, cg.x10);
    cg.Ld1b(cg.z0.b, cg.p0, AdrNoOfs(cg.x0));

    cg.Mov(cg.w12, 0);
    cg.Mova(cg.za0h.b, cg.w12, 0, cg.p0, cg.z0.b);
    cg.Mov(cg.za0h.b, cg.w12, 1, cg.p0, cg.z0.b);
    cg.Mova(cg.za0v.b, cg.w12, 0, cg.p0, cg.z0.b);
    cg.Mov(cg.za0v.b, cg.w12, 1, cg.p0, cg.z0.b);

    cg.Mova(cg.z1.b, cg.p0, cg.za0h.b, cg.w12, 0);
    cg.Mov(cg.z2.b, cg.p0, cg.za0h.b, cg.w12, 1);
    cg.Mova(cg.z3.b, cg.p0, cg.za0v.b, cg.w12, 0);
    cg.Mov(cg.z4.b, cg.p0, cg.za0v.b, cg.w12, 1);

    cg.Mov(cg.x11, 0);
    cg.St1b(cg.z1.b, cg.p0, AdrNoOfs(cg.x1));
    cg.Add(cg.x1, cg.x1, 16);
    cg.St1b(cg.z2.b, cg.p0, AdrNoOfs(cg.x1));
    cg.Add(cg.x1, cg.x1, 16);
    cg.St1b(cg.z3.b, cg.p0, AdrNoOfs(cg.x1));
    cg.Add(cg.x1, cg.x1, 16);
    cg.St1b(cg.z4.b, cg.p0, AdrNoOfs(cg.x1));

    cg.Smstop(StreamMode::SMZA);
    cg.Ret();
    cg.Ready();

    auto f = cg.GetCode<void (*)(int8_t*, int8_t*)>();
    int8_t z[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
    int8_t out[64];
    f(z, out);

    EXPECT_EQ(0, out[0]);
    EXPECT_EQ(0, out[1]);
    EXPECT_EQ(2, out[2]);
    EXPECT_EQ(1, out[16]);
    EXPECT_EQ(1, out[17]);
    EXPECT_EQ(2, out[18]);
    EXPECT_EQ(0, out[32]);
    EXPECT_EQ(1, out[33]);
    EXPECT_EQ(2, out[34]);
    EXPECT_EQ(0, out[48]);
    EXPECT_EQ(1, out[49]);
    EXPECT_EQ(2, out[50]);
}

TEST(armcg_test_t, TestAddva)
{
    GTestJitGen cg;
    cg.Smstart(StreamMode::SMZA);
    cg.Zero({cg.za0, cg.za1});
    cg.Mov(cg.x9, 0);
    cg.Mov(cg.x10, 4);
    cg.Mov(cg.x11, 2);
    cg.Whilelt(cg.p0.s, cg.x9, cg.x10);
    cg.Whilelt(cg.p1.s, cg.x9, cg.x11);
    cg.Ld1w(cg.z0.s, cg.p0, AdrNoOfs(cg.x0));
    cg.Addha(cg.za0.s, cg.p1, cg.p0, cg.z0.s);
    cg.Addva(cg.za1.s, cg.p0, cg.p1, cg.z0.s);
    cg.Mov(cg.w12, 0);
    cg.Mov(cg.x8, 0);
    cg.St1w(cg.za0h.s, cg.w12, 0, cg.p0, AdrScSc(cg.x1, cg.x8, ShMod::LSL, 2));
    cg.St1w(cg.za1v.s, cg.w12, 0, cg.p0, AdrScSc(cg.x2, cg.x8, ShMod::LSL, 2));
    cg.Add(cg.x8, cg.x8, 4);
    cg.St1w(cg.za0h.s, cg.w12, 1, cg.p0, AdrScSc(cg.x1, cg.x8, ShMod::LSL, 2));
    cg.St1w(cg.za1v.s, cg.w12, 1, cg.p0, AdrScSc(cg.x2, cg.x8, ShMod::LSL, 2));
    cg.Smstop(StreamMode::SMZA);
    cg.Ret();
    cg.Ready();
    auto Func = cg.GetCode<void (*)(float *, float *, float *)>();
    float arr1[] = {0.5f, 1.0f, 1.5f, 2.0f};
    float arr2[] = {0, 0, 0, 0, 0, 0, 0, 0};
    float arr3[] = {0, 0, 0, 0, 0, 0, 0, 0};
    Func(arr1, arr2, arr3);
    ASSERT_FLOAT_EQ(arr2[0], 0.5);
    ASSERT_FLOAT_EQ(arr2[1], 1.0);
    ASSERT_FLOAT_EQ(arr2[2], 1.5);
    ASSERT_FLOAT_EQ(arr2[3], 2.0);
    ASSERT_FLOAT_EQ(arr2[4], 0.5);
    ASSERT_FLOAT_EQ(arr2[5], 1.0);
    ASSERT_FLOAT_EQ(arr2[6], 1.5);
    ASSERT_FLOAT_EQ(arr2[7], 2.0);
    ASSERT_FLOAT_EQ(arr3[0], 0.5);
    ASSERT_FLOAT_EQ(arr3[1], 1.0);
    ASSERT_FLOAT_EQ(arr3[2], 1.5);
    ASSERT_FLOAT_EQ(arr3[3], 2.0);
    ASSERT_FLOAT_EQ(arr3[4], 0.5);
    ASSERT_FLOAT_EQ(arr3[5], 1.0);
    ASSERT_FLOAT_EQ(arr3[6], 1.5);
    ASSERT_FLOAT_EQ(arr3[7], 2.0);
}

TEST(armcg_test_t, TestMCMovaHTile2VecH)
{
    GTestJitGen cg;
    cg.Smstart(StreamMode::SMZA);
    cg.Mov(cg.w15, 0);
    cg.SetPreg(cg.p0.h, 8);
    cg.Mov(cg.x9, 0);
    cg.Ld1h(cg.za0h.h, cg.w15, 0, cg.p0 / PredType::T_Z,
        AdrScSc(cg.x1, cg.x9, ShMod::LSL, static_cast<uint32_t>(1)));
    cg.Mova(cg.z0.h, cg.p0 / PredType::T_M, cg.za0h.h, cg.w15, 0);
    cg.St1h(cg.z0.h, cg.p0, AdrNoOfs(cg.x0));
    cg.Smstop(StreamMode::SMZA);
    cg.Ret();
    cg.Ready();
    auto cgFunc = cg.GetCode<void (*)(uint16_t*, uint16_t*)>();
    uint16_t cgSrc[8] = {1, 2, 3, 4, 5, 6, 7, 8};
    uint16_t cgDst[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    cgFunc(cgDst, cgSrc);
    ASSERT_EQ(cgSrc[0], cgDst[0]);
    ASSERT_EQ(cgSrc[1], cgDst[1]);
}
TEST(armcg_test_t, TestMCMovaVec2HTileH)
{
    GTestJitGen cg;
    cg.Smstart(StreamMode::SMZA);
    cg.SetPreg(cg.p0.h, 8);
    cg.Ld1h(cg.z0.h, cg.p0 / PredType::T_Z, AdrNoOfs(cg.x1));
    cg.Mov(cg.w15, 0);
    cg.Mova(cg.za0h.h, cg.w15, 0, cg.p0 / PredType::T_M, cg.z0.h);
    cg.Mov(cg.x9, 0);
    cg.St1h(cg.za0h.h, cg.w15, 0, cg.p0,
        AdrScSc(cg.x0, cg.x9, ShMod::LSL, static_cast<uint32_t>(1)));
    cg.Smstop(StreamMode::SMZA);
    cg.Ret();
    cg.Ready();
    auto cgFunc = cg.GetCode<void (*)(uint16_t*, uint16_t*)>();
    uint16_t cgSrc[8] = {1, 2, 3, 4, 5, 6, 7, 8};
    uint16_t cgDst[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    cgFunc(cgDst, cgSrc);
    ASSERT_EQ(cgSrc[0], cgDst[0]);
    ASSERT_EQ(cgSrc[1], cgDst[1]);
}
TEST(armcg_test_t, TestMCMovaVTile2VecH)
{
    GTestJitGen cg;
    cg.Smstart(StreamMode::SMZA);
    cg.Mov(cg.w15, 0);
    cg.SetPreg(cg.p0.h, 8);
    cg.Mov(cg.x9, 0);
    cg.Ld1h(cg.za0v.h, cg.w15, 0, cg.p0 / PredType::T_Z,
        AdrScSc(cg.x1, cg.x9, ShMod::LSL, static_cast<uint32_t>(1)));
    cg.Mova(cg.z0.h, cg.p0 / PredType::T_M, cg.za0v.h, cg.w15, 0);
    cg.St1h(cg.z0.h, cg.p0, AdrNoOfs(cg.x0));
    cg.Smstop(StreamMode::SMZA);
    cg.Ret();
    cg.Ready();
    auto cgFunc = cg.GetCode<void (*)(uint16_t*, uint16_t*)>();
    uint16_t cgSrc[8] = {1, 2, 3, 4, 5, 6, 7, 8};
    uint16_t cgDst[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    cgFunc(cgDst, cgSrc);
    ASSERT_EQ(cgSrc[0], cgDst[0]);
    ASSERT_EQ(cgSrc[1], cgDst[1]);
}
TEST(armcg_test_t, TestMCMovaVec2VTileH)
{
    GTestJitGen cg;
    cg.Smstart(StreamMode::SMZA);
    cg.SetPreg(cg.p0.h, 8);
    cg.Ld1h(cg.z0.h, cg.p0 / PredType::T_Z, AdrNoOfs(cg.x1));
    cg.Mov(cg.w15, 0);
    cg.Mova(cg.za0v.h, cg.w15, 0, cg.p0 / PredType::T_M, cg.z0.h);
    cg.Mov(cg.x9, 0);
    cg.St1h(cg.za0v.h, cg.w15, 0, cg.p0,
        AdrScSc(cg.x0, cg.x9, ShMod::LSL, static_cast<uint32_t>(1)));
    cg.Smstop(StreamMode::SMZA);
    cg.Ret();
    cg.Ready();
    auto cgFunc = cg.GetCode<void (*)(uint16_t*, uint16_t*)>();
    uint16_t cgSrc[8] = {1, 2, 3, 4, 5, 6, 7, 8};
    uint16_t cgDst[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    cgFunc(cgDst, cgSrc);
    ASSERT_EQ(cgSrc[0], cgDst[0]);
    ASSERT_EQ(cgSrc[1], cgDst[1]);
}

TEST(armcg_test_t, TestMovaVV2Za)
{
    GTestJitGen cg;
    cg.Smstart(StreamMode::SMZA);
    cg.Zero({cg.za0});
    cg.Mov(cg.x9, 0);
    cg.Mov(cg.x10, 4);
    cg.Whilelt(cg.p0.s, cg.x9, cg.x10);
    cg.Ld1w(cg.z0.s, cg.p0, AdrNoOfs(cg.x0));   // z0 = [a01, a02, a03, a04]
    cg.Mov(cg.w12, 0);
    cg.Mova(cg.za0v.s, cg.w12, 0, cg.p0, cg.z0.s);

    cg.Mov(cg.x9, 0);
    cg.St1w(cg.za0v.s, cg.w12, 0, cg.p0, AdrScSc(cg.x1, cg.x9, ShMod::LSL, 2));
    cg.Smstop(StreamMode::SMZA);
    cg.Ret();
    cg.Ready();

    auto f = cg.GetCode<void (*)(float*, float*)>();
    float matA_4x1[] = {0.1f, -0.2f, 0.3f, -0.4f};
    float matC_4x1[4];
    f(matA_4x1, matC_4x1);

    ASSERT_FLOAT_EQ(matA_4x1[0], matC_4x1[0]);
    ASSERT_FLOAT_EQ(matA_4x1[1], matC_4x1[1]);
    ASSERT_FLOAT_EQ(matA_4x1[2], matC_4x1[2]);
    ASSERT_FLOAT_EQ(matA_4x1[3], matC_4x1[3]);
}

TEST(armcg_test_t, TestMovaVH2Za)
{
    GTestJitGen cg;
    cg.Smstart(StreamMode::SMZA);
    cg.Zero({cg.za0});
    cg.Mov(cg.x9, 0);
    cg.Mov(cg.x10, 4);
    cg.Whilelt(cg.p0.s, cg.x9, cg.x10);
    cg.Ld1w(cg.z0.s, cg.p0, AdrNoOfs(cg.x0));   // z0 = [a01, a02, a03, a04]
    cg.Mov(cg.w12, 0);
    cg.Mova(cg.za0h.s, cg.w12, 0, cg.p0, cg.z0.s);

    cg.Mov(cg.x9, 0);
    cg.St1w(cg.za0h.s, cg.w12, 0, cg.p0, AdrScSc(cg.x1, cg.x9, ShMod::LSL, 2));
    cg.Smstop(StreamMode::SMZA);
    cg.Ret();
    cg.Ready();

    auto f = cg.GetCode<void (*)(float*, float*)>();
    float matA_4x1[] = {0.1f, -0.2f, 0.3f, -0.4f};
    float matC_4x1[4];
    f(matA_4x1, matC_4x1);

    ASSERT_FLOAT_EQ(matA_4x1[0], matC_4x1[0]);
    ASSERT_FLOAT_EQ(matA_4x1[1], matC_4x1[1]);
    ASSERT_FLOAT_EQ(matA_4x1[2], matC_4x1[2]);
    ASSERT_FLOAT_EQ(matA_4x1[3], matC_4x1[3]);
}
#endif

//mne load test
TEST(armcg_test_t, TestLd1)
{
    GTestJitGen ben; // be(VRegBElem)n(AdrNoOfs)
    //
    ben.Ld1(ben.v0.b[0], AdrNoOfs(ben.x0));
    //
    ben.St1(ben.v0.b[0], AdrNoOfs(ben.x1));
    ben.Ret();
    ben.Ready();
    auto benFunc = ben.GetCode<void (*)(uint8_t*, uint8_t*)>();
    uint8_t benSrc[1] = {1};
    uint8_t benDst[1] = {0};
    benFunc(benSrc, benDst);
    ASSERT_EQ(benSrc[0], benDst[0]);

    GTestJitGen hen; // he(VRegHElem)n(AdrNoOfs)
    //
    hen.Ld1(hen.v0.h[0], AdrNoOfs(hen.x0));
    //
    hen.St1(hen.v0.h[0], AdrNoOfs(hen.x1));
    hen.Ret();
    hen.Ready();
    auto henFunc = hen.GetCode<void (*)(uint16_t*, uint16_t*)>();
    uint16_t henSrc[1] = {1};
    uint16_t henDst[1] = {0};
    henFunc(henSrc, henDst);
    ASSERT_EQ(henSrc[0], henDst[0]);

    GTestJitGen sen; // se(VRegSElem)n(AdrNoOfs)
    //
    sen.Ld1(sen.v0.s[0], AdrNoOfs(sen.x0));
    //
    sen.St1(sen.v0.s[0], AdrNoOfs(sen.x1));
    sen.Ret();
    sen.Ready();
    auto senFunc = sen.GetCode<void (*)(uint32_t*, uint32_t*)>();
    uint32_t senSrc[1] = {1};
    uint32_t senDst[1] = {0};
    senFunc(senSrc, senDst);
    ASSERT_EQ(senSrc[0], senDst[0]);

    GTestJitGen s4ln; // s4l(VReg4SList)n(AdrNoOfs)
    //
    s4ln.Ld1(VReg4SList(s4ln.v0.s), AdrNoOfs(s4ln.x0));
    //
    s4ln.St1(s4ln.v0.s[0], AdrNoOfs(s4ln.x1));
    s4ln.Ret();
    s4ln.Ready();
    auto s4lnFunc = s4ln.GetCode<void (*)(uint32_t*, uint32_t*)>();
    uint32_t s4lnSrc[16] = {1, 2, 3, 4};
    uint32_t s4lnDst[1] = {0};
    s4lnFunc(s4lnSrc, s4lnDst);
    ASSERT_EQ(s4lnSrc[0], s4lnDst[0]);
}

TEST(armcg_test_t, TestLdStNeon)
{
    GTestJitGen cg1;
    cg1.Ld1(VReg4SList(cg1.v1.s4), AdrPostImm(cg1.x0, 16));
    cg1.St1(VReg4SList(cg1.v1.s4), AdrPostImm(cg1.x1, 16));
    cg1.Ld1(cg1.v2.s4[0], AdrPostImm(cg1.x0, 4));   // post imm is fixed, other value will throw error
    cg1.St1(cg1.v2.s4[0], AdrPostImm(cg1.x1, 4));
    cg1.Ld1(cg1.v2.s4[1], AdrPostImm(cg1.x0, 4));
    cg1.St1(cg1.v2.s4[1], AdrPostImm(cg1.x1, 4));
    cg1.Ld1(cg1.v2.s4[2], AdrPostImm(cg1.x0, 4));
    cg1.St1(cg1.v2.s4[2], AdrPostImm(cg1.x1, 4));
    cg1.Ld1(cg1.v2.s4[3], AdrNoOfs(cg1.x0));
    cg1.St1(cg1.v2.s4[3], AdrNoOfs(cg1.x1));
    cg1.Ret();
    cg1.Ready();

    auto f1 = cg1.GetCode<void (*)(float *, float *)>();
    float arr1[8] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
    float ret1[8] = {0.0f};
    f1(arr1, ret1);
    ASSERT_EQ(ret1[0], arr1[0]);
    ASSERT_EQ(ret1[1], arr1[1]);
    ASSERT_EQ(ret1[2], arr1[2]);
    ASSERT_EQ(ret1[3], arr1[3]);
    ASSERT_EQ(ret1[4], arr1[4]);
    ASSERT_EQ(ret1[5], arr1[5]);
    ASSERT_EQ(ret1[6], arr1[6]);
    ASSERT_EQ(ret1[7], arr1[7]);

    GTestJitGen cg2;
    cg2.Mov(cg2.x9, 16);
    cg2.Mov(cg2.x10, 4);
    cg2.Ld1(VReg4SList(cg2.v1.s4), AdrPostReg(cg2.x0, cg2.x9));
    cg2.St1(VReg4SList(cg2.v1.s4), AdrPostReg(cg2.x1, cg2.x9));
    cg2.Ld1(cg1.v2.s4[0], AdrPostReg(cg2.x0, cg2.x10));
    cg2.St1(cg1.v2.s4[0], AdrPostReg(cg2.x1, cg2.x10));
    cg2.Ld1(cg1.v2.s4[1], AdrPostReg(cg2.x0, cg2.x10));
    cg2.St1(cg1.v2.s4[1], AdrPostReg(cg2.x1, cg2.x10));
    cg2.Ld1(cg1.v2.s4[2], AdrPostReg(cg2.x0, cg2.x10));
    cg2.St1(cg1.v2.s4[2], AdrPostReg(cg2.x1, cg2.x10));
    cg2.Ld1(cg1.v2.s4[3], AdrNoOfs(cg2.x0));
    cg2.St1(cg1.v2.s4[3], AdrNoOfs(cg2.x1));
    cg2.Ret();
    cg2.Ready();

    auto f2 = cg2.GetCode<void (*)(float *, float *)>();
    float arr2[8] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
    float ret2[8] = {0.0f};
    f2(arr2, ret2);
    ASSERT_EQ(ret2[0], arr2[0]);
    ASSERT_EQ(ret2[1], arr2[1]);
    ASSERT_EQ(ret2[2], arr2[2]);
    ASSERT_EQ(ret2[3], arr2[3]);
    ASSERT_EQ(ret2[4], arr2[4]);
    ASSERT_EQ(ret2[5], arr2[5]);
    ASSERT_EQ(ret2[6], arr2[6]);
    ASSERT_EQ(ret2[7], arr2[7]);

    GTestJitGen cg3;
    cg3.Mov(cg3.x9, 16);
    cg3.Mov(cg3.x10, 2);
    cg3.Ld1(VReg8HList(cg3.v1.h8), AdrNoOfs(cg3.x0));
    cg3.St1(VReg8HList(cg3.v1.h8), AdrNoOfs(cg3.x2));
    cg3.Ld1(VReg8HList(cg3.v2.h8), AdrPostReg(cg3.x0, cg3.x9));
    cg3.St1(VReg8HList(cg3.v2.h8), AdrPostReg(cg3.x1, cg3.x9));
    cg3.Ld1(cg3.v3.h8[0], AdrPostReg(cg3.x0, cg3.x10));
    cg3.St1(cg3.v3.h8[0], AdrPostReg(cg3.x1, cg3.x10));
    cg3.Ld1(cg3.v3.h8[1], AdrPostImm(cg3.x0, 2));
    cg3.St1(cg3.v3.h8[1], AdrPostImm(cg3.x1, 2));
    cg3.Ld1(cg3.v3.h8[2], AdrPostImm(cg3.x0, 2));
    cg3.St1(cg3.v3.h8[2], AdrPostImm(cg3.x1, 2));
    cg3.Ld1(cg3.v3.h8[3], AdrNoOfs(cg3.x0));
    cg3.St1(cg3.v3.h8[3], AdrNoOfs(cg3.x1));
    cg3.Ret();
    cg3.Ready();

    auto f3 = cg3.GetCode<void (*)(uint16_t *, uint16_t *, uint16_t *)>();
    uint16_t arr3[16] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
    uint16_t ret3[16] = {0};
    uint16_t r3[16] = {0};
    f3(arr3, ret3, r3);
    ASSERT_EQ(ret3[0], 1);
    ASSERT_EQ(ret3[1], 2);
    ASSERT_EQ(ret3[8], 9);
    ASSERT_EQ(ret3[9], 10);
    ASSERT_EQ(ret3[10], 11);
    ASSERT_EQ(ret3[11], 12);
    ASSERT_EQ(r3[0], 1);
    ASSERT_EQ(r3[1], 2);

    GTestJitGen cg4;
    cg4.Mov(cg4.x9, 16);
    cg4.Mov(cg4.x10, 1);
    cg4.Ld1(VReg16BList(cg4.v1.b16), AdrNoOfs(cg4.x0));
    cg4.St1(VReg16BList(cg4.v1.b16), AdrNoOfs(cg4.x2));
    cg4.Ld1(VReg16BList(cg4.v2.b16), AdrPostReg(cg4.x0, cg4.x9));
    cg4.St1(VReg16BList(cg4.v2.b16), AdrPostReg(cg4.x1, cg4.x9));
    cg4.Ld1(cg4.v2.b16[0], AdrPostReg(cg4.x0, cg4.x10));
    cg4.St1(cg4.v2.b16[0], AdrPostReg(cg4.x1, cg4.x10));
    cg4.Ld1(cg4.v2.b16[1], AdrPostImm(cg4.x0, 1));
    cg4.St1(cg4.v2.b16[1], AdrPostImm(cg4.x1, 1));
    cg4.Ld1(cg4.v2.b16[2], AdrPostImm(cg4.x0, 1));
    cg4.St1(cg4.v2.b16[2], AdrPostImm(cg4.x1, 1));
    cg4.Ld1(cg4.v2.b16[3], AdrNoOfs(cg4.x0));
    cg4.St1(cg4.v2.b16[3], AdrNoOfs(cg4.x1));
    cg4.Ret();
    cg4.Ready();

    auto f4 = cg4.GetCode<void (*)(uint8_t *, uint8_t *, uint8_t *)>();
    uint8_t arr4[32] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
                        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
    uint8_t ret4[32] = {0};
    uint8_t r4[15] = {0};
    f4(arr4, ret4, r4);
    ASSERT_EQ(ret4[0], 1);
    ASSERT_EQ(ret4[1], 2);
    ASSERT_EQ(ret4[14], 15);
    ASSERT_EQ(ret4[15], 16);
    ASSERT_EQ(ret4[16], 1);
    ASSERT_EQ(ret4[17], 2);
    ASSERT_EQ(ret4[18], 3);
    ASSERT_EQ(ret4[19], 4);
    ASSERT_EQ(r4[0], 1);
    ASSERT_EQ(r4[1], 2);

    GTestJitGen cg5;
    cg5.Ld1(VReg8HList(cg5.v1.h8), AdrPostImm(cg5.x0, 16));
    cg5.St1(VReg8HList(cg5.v1.h8), AdrPostImm(cg5.x1, 16));
    cg5.Ld1(VReg16BList(cg5.v2.b16), AdrPostImm(cg5.x2, 16));
    cg5.St1(VReg16BList(cg5.v2.b16), AdrPostImm(cg5.x3, 16));
    cg5.Ret();
    cg5.Ready();

    uint16_t arr5[8] = {1, 2, 3, 4, 5, 6, 7, 8};
    uint16_t ret5[8] = {0};
    uint8_t arr6[16] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
    uint8_t ret6[16] = {0};
    auto f5 = cg5.GetCode<void (*)(uint16_t *, uint16_t *, uint8_t *, uint8_t *)>();
    f5(arr5, ret5, arr6, ret6);
    ASSERT_EQ(ret5[0], 1);
    ASSERT_EQ(ret5[1], 2);
    ASSERT_EQ(ret5[6], 7);
    ASSERT_EQ(ret5[7], 8);
    ASSERT_EQ(ret6[0], 1);
    ASSERT_EQ(ret6[1], 2);
    ASSERT_EQ(ret6[14], 15);
    ASSERT_EQ(ret6[15], 16);
}

TEST(armcg_test_t, TestLdSt1b)
{
    GTestJitGen gen;
    gen.Mov(gen.x9, 0);
    gen.Mov(gen.x10, 8);
    gen.Whilelt(gen.p0.b, gen.x9, gen.x10);
    gen.Ld1b(gen.z0.b, gen.p0, AdrScImm(gen.x0, 1));
    gen.St1b(gen.z0.b, gen.p0, AdrScImm(gen.x1, 1));
    gen.Ret();
    gen.Ready();

    uint8_t arrIn[128];
    for (int i = 0; i != 127; ++i) {
        arrIn[i] = i;
    }
    uint8_t arrOut[128];
    for (int i = 0; i != 127; ++i) {
        arrOut[i] = 0;
    }
    auto f = gen.GetCode<void(*)(uint8_t*, uint8_t*)>();
    f(arrIn, arrOut);
    // offset is imm times vector length
#ifdef KUDNNL_920F
    for (int i = 64; i != 72; ++i) {
        EXPECT_EQ(int(arrIn[i]), int(arrOut[i]));
    }
#else
    for (int i = 32; i != 40; ++i) {
        EXPECT_EQ(int(arrIn[i]), int(arrOut[i]));
    }
#endif
}

TEST(armcg_test_t, TestLd1b)
{
    GTestJitGen bn; // b(ZRegB)n(AdrNoOfs)
    bn.Ptrue(bn.p0.b);
    //
    bn.Ld1b(bn.z0.b, bn.p0 / PredType::T_Z, AdrNoOfs(bn.x0));
    //
    bn.St1b(bn.z0.b, bn.p0, AdrNoOfs(bn.x1));
    bn.Ret();
    bn.Ready();
    auto bnFunc = bn.GetCode<void (*)(uint8_t*, uint8_t*)>();
    uint8_t bnSrc[64] = {1};
    uint8_t bnDst[64] = {0};
    bnFunc(bnSrc, bnDst);
    ASSERT_EQ(bnSrc[0], bnDst[0]);

    GTestJitGen sn; // s(ZRegS)n(AdrNoOfs)
    sn.Ptrue(sn.p0.b);
    //
    sn.Ld1b(sn.z0.s, sn.p0 / PredType::T_Z, AdrNoOfs(sn.x0));
    //
    sn.St1b(sn.z0.b, sn.p0, AdrNoOfs(sn.x1));
    sn.Ret();
    sn.Ready();
    auto snFunc = sn.GetCode<void (*)(uint8_t*, uint8_t*)>();
    uint8_t snSrc[64] = {1};
    uint8_t snDst[64] = {0};
    snFunc(snSrc, snDst);
    ASSERT_EQ(snSrc[0], snDst[0]);

    GTestJitGen ssi; // s(ZRegS)si(AdrScImm)
    ssi.Ptrue(ssi.p0.b);
    //
    ssi.Ld1b(ssi.z0.s, ssi.p0 / PredType::T_Z, AdrScImm(ssi.x0));
    //
    ssi.St1b(ssi.z0.b, ssi.p0, AdrNoOfs(ssi.x1));
    ssi.Ret();
    ssi.Ready();
    auto ssiFunc = ssi.GetCode<void (*)(uint8_t*, uint8_t*)>();
    uint8_t ssiSrc[64] = {1};
    uint8_t ssiDst[64] = {0};
    ssiFunc(ssiSrc, ssiDst);
    ASSERT_EQ(ssiSrc[0], ssiDst[0]);

    GTestJitGen ss3u; // s(ZRegS)s3u(AdrSc32U)
    ss3u.Ptrue(ss3u.p0.s);
    ss3u.Index(ss3u.z0.s, 0, 1);
    //
    ss3u.Ld1b(ss3u.z1.s, ss3u.p0 / PredType::T_Z, AdrSc32U(ss3u.x0, ss3u.z0.s, ExtMod::UXTW));
    //
    ss3u.St1w(ss3u.z1.s, ss3u.p0, AdrNoOfs(ss3u.x1));
    ss3u.Ret();
    ss3u.Ready();
    auto ss3uFunc = ss3u.GetCode<void (*)(uint8_t*, uint32_t*)>();
    uint8_t ss3uSrc[16] = {1};
    uint32_t ss3uDst[16] = {0};
    ss3uFunc(ss3uSrc, ss3uDst);
    ASSERT_EQ(ss3uSrc[0], ss3uDst[0]);
}

TEST(armcg_test_t, TestLd1d)
{
    GTestJitGen dn; // d(ZRegD)n(AdrNoOfs)
    dn.Ptrue(dn.p0.s);
    //
    dn.Ld1d(dn.z0.d, dn.p0 / PredType::T_Z, AdrNoOfs(dn.x0));
    //
    dn.St1w(dn.z0.s, dn.p0, AdrNoOfs(dn.x1));
    dn.Ret();
    dn.Ready();
    auto dnFunc = dn.GetCode<void (*)(uint64_t*, uint64_t*)>();
    uint64_t dnSrc[8] = {1};
    uint64_t dnDst[8] = {0};
    dnFunc(dnSrc, dnDst);
    ASSERT_EQ(dnSrc[0], dnDst[0]);
}

TEST(armcg_test_t, TestLd1h)
{
    GTestJitGen hn; // h(ZRegH)n(AdrNoOfs)
    hn.Ptrue(hn.p0.h);
    //
    hn.Ld1h(hn.z0.h, hn.p0 / PredType::T_Z, AdrNoOfs(hn.x0));
    //
    hn.St1h(hn.z0.h, hn.p0, AdrNoOfs(hn.x1));
    hn.Ret();
    hn.Ready();
    auto hnFunc = hn.GetCode<void (*)(uint16_t*, uint16_t*)>();
    uint16_t hnSrc[32] = {1};
    uint16_t hnDst[32] = {0};
    hnFunc(hnSrc, hnDst);
    ASSERT_EQ(hnSrc[0], hnDst[0]);

    GTestJitGen ssi; // h(ZRegH)si(AdrScImm)
    ssi.Ptrue(ssi.p0.h);
    //
    ssi.Ld1h(ssi.z0.h, ssi.p0 / PredType::T_Z, AdrScImm(ssi.x0));
    //
    ssi.St1h(ssi.z0.h, ssi.p0, AdrNoOfs(ssi.x1));
    ssi.Ret();
    ssi.Ready();
    auto ssiFunc = ssi.GetCode<void (*)(uint16_t*, uint16_t*)>();
    uint16_t ssiSrc[32] = {1};
    uint16_t ssiDst[32] = {0};
    ssiFunc(ssiSrc, ssiDst);
    ASSERT_EQ(ssiSrc[0], ssiDst[0]);

    GTestJitGen sss; // h(ZRegH)ss(AdrScSc)
    sss.Ptrue(sss.p0.h);
    sss.Mov(sss.x9, 0);
    //
    sss.Ld1h(sss.z1.h, sss.p0 / PredType::T_Z, AdrScSc(sss.x0, sss.x9, ShMod::LSL, 1));
    //
    sss.St1h(sss.z1.h, sss.p0, AdrNoOfs(sss.x1));
    sss.Ret();
    sss.Ready();
    auto sssFunc = sss.GetCode<void (*)(uint16_t*, uint16_t*)>();
    uint16_t sssSrc[32] = {1};
    uint16_t sssDst[32] = {0};
    sssFunc(sssSrc, sssDst);
    ASSERT_EQ(sssSrc[0], sssDst[0]);

    GTestJitGen sn; // s(ZRegS)n(AdrNoOfs)
    sn.Ptrue(sn.p0.s);
    //
    sn.Ld1h(sn.z0.s, sn.p0 / PredType::T_Z, AdrNoOfs(sn.x0));
    //
    sn.St1w(sn.z0.s, sn.p0, AdrNoOfs(sn.x1));
    sn.Ret();
    sn.Ready();
    auto snFunc = sn.GetCode<void (*)(uint16_t*, uint32_t*)>();
    uint16_t snSrc[16] = {1};
    uint32_t snDst[16] = {0};
    snFunc(snSrc, snDst);
    ASSERT_EQ(snSrc[0], snDst[0]);

    GTestJitGen ss3u; // s(ZRegS)s3u(AdrSc32U)
    ss3u.Ptrue(ss3u.p0.s);
    ss3u.Index(ss3u.z0.s, 0, 2);
    //
    ss3u.Ld1h(ss3u.z1.s, ss3u.p0 / PredType::T_Z, AdrSc32U(ss3u.x0, ss3u.z0.s, ExtMod::UXTW));
    //
    ss3u.St1w(ss3u.z1.s, ss3u.p0, AdrNoOfs(ss3u.x1));
    ss3u.Ret();
    ss3u.Ready();
    auto ss3uFunc = ss3u.GetCode<void (*)(uint16_t*, uint32_t*)>();
    uint16_t ss3uSrc[16] = {1};
    uint32_t ss3uDst[16] = {0};
    ss3uFunc(ss3uSrc, ss3uDst);
    ASSERT_EQ(ss3uSrc[0], ss3uDst[0]);
}

#ifdef KUDNNL_920F
TEST(armcg_test_t, TestMCLd1hMem2HTile)
{
    GTestJitGen cg;
    cg.Smstart(StreamMode::SMZA);
    cg.Mov(cg.w15, 0);
    cg.SetPreg(cg.p0.h, 8);
    cg.Mov(cg.x9, 0);
    cg.Ld1h(cg.za0h.h, cg.w15, 0, cg.p0 / PredType::T_Z,
        AdrScSc(cg.x1, cg.x9, ShMod::LSL, static_cast<uint32_t>(1)));
    cg.St1h(cg.za0h.h, cg.w15, 0, cg.p0,
        AdrScSc(cg.x0, cg.x9, ShMod::LSL, static_cast<uint32_t>(1)));
    cg.Smstop(StreamMode::SMZA);
    cg.Ret();
    cg.Ready();
    auto cgFunc = cg.GetCode<void (*)(uint16_t*, uint16_t*)>();
    uint16_t cgSrc[8] = {1, 2, 3, 4, 5, 6, 7, 8};
    uint16_t cgDst[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    cgFunc(cgDst, cgSrc);
    ASSERT_EQ(cgSrc[0], cgDst[0]);
    ASSERT_EQ(cgSrc[1], cgDst[1]);
}
TEST(armcg_test_t, TestMCLd1hMem2VTile)
{
    GTestJitGen cg;
    cg.Smstart(StreamMode::SMZA);
    cg.Mov(cg.w15, 0);
    cg.SetPreg(cg.p0.h, 8);
    cg.Mov(cg.x9, 0);
    cg.Ld1h(cg.za0v.h, cg.w15, 0, cg.p0 / PredType::T_Z,
        AdrScSc(cg.x1, cg.x9, ShMod::LSL, static_cast<uint32_t>(1)));
    cg.St1h(cg.za0v.h, cg.w15, 0, cg.p0,
        AdrScSc(cg.x0, cg.x9, ShMod::LSL, static_cast<uint32_t>(1)));
    cg.Smstop(StreamMode::SMZA);
    cg.Ret();
    cg.Ready();
    auto cgFunc = cg.GetCode<void (*)(uint16_t*, uint16_t*)>();
    uint16_t cgSrc[8] = {1, 2, 3, 4, 5, 6, 7, 8};
    uint16_t cgDst[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    cgFunc(cgDst, cgSrc);
    ASSERT_EQ(cgSrc[0], cgDst[0]);
    ASSERT_EQ(cgSrc[1], cgDst[1]);
}
#endif

TEST(armcg_test_t, TestLd1r)
{
    GTestJitGen s4ln; // s4l(VReg4SList)n(AdrNoOfs)
    //
    s4ln.Ld1r(VReg4SList(s4ln.v0.s), AdrNoOfs(s4ln.x0));
    //
    s4ln.St1(s4ln.v0.s[0], AdrNoOfs(s4ln.x1));
    s4ln.Ret();
    s4ln.Ready();
    auto s4lnFunc = s4ln.GetCode<void (*)(uint32_t*, uint32_t*)>();
    uint32_t s4lnSrc[4] = {1, 2, 3, 4};
    uint32_t s4lnDst[1] = {0};
    s4lnFunc(s4lnSrc, s4lnDst);
    ASSERT_EQ(s4lnSrc[0], s4lnDst[0]);
}

TEST(armcg_test_t, TestLd1rb)
{
    GTestJitGen sn; // s(ZRegS)n(AdrNoOfs)
    sn.Ptrue(sn.p0.s);
    //
    sn.Ld1rb(sn.z0.s, sn.p0 / PredType::T_Z, AdrNoOfs(sn.x0));
    //
    sn.St1w(sn.z0.s, sn.p0, AdrNoOfs(sn.x1));
    sn.Ret();
    sn.Ready();
    auto snFunc = sn.GetCode<void (*)(uint32_t*, uint32_t*)>();
    uint32_t snSrc[1] = {1};
    uint32_t snDst[16] = {0};
    snFunc(snSrc, snDst);
    ASSERT_EQ(snSrc[0], snDst[0]);
}

TEST(armcg_test_t, TestLd1rh)
{
    GTestJitGen hn; // h(ZRegH)n(AdrNoOfs)
    hn.Ptrue(hn.p0.h);
    //
    hn.Ld1rh(hn.z0.h, hn.p0 / PredType::T_Z, AdrNoOfs(hn.x0));
    //
    hn.St1h(hn.z0.h, hn.p0, AdrNoOfs(hn.x1));
    hn.Ret();
    hn.Ready();
    auto hnFunc = hn.GetCode<void (*)(uint16_t*, uint16_t*)>();
    uint16_t hnSrc[1] = {1};
    uint16_t hnDst[32] = {0};
    hnFunc(hnSrc, hnDst);
    ASSERT_EQ(hnSrc[0], hnDst[0]);

    GTestJitGen sn; // s(ZRegS)n(AdrNoOfs)
    sn.Ptrue(sn.p0.s);
    //
    sn.Ld1rh(sn.z0.s, sn.p0 / PredType::T_Z, AdrNoOfs(sn.x0));
    //
    sn.St1w(sn.z0.s, sn.p0, AdrNoOfs(sn.x1));
    sn.Ret();
    sn.Ready();
    auto snFunc = sn.GetCode<void (*)(uint16_t*, uint32_t*)>();
    uint16_t snSrc[1] = {1};
    uint32_t snDst[16] = {0};
    snFunc(snSrc, snDst);
    ASSERT_EQ(snSrc[0], snDst[0]);
}

TEST(armcg_test_t, TestLd1rhOff)
{
    GTestJitGen hn; // h(ZRegH)n(AdrNoOfs)
    hn.Ptrue(hn.p0.h);
    //
    hn.Ld1rh(hn.z0.h, hn.p0 / PredType::T_Z, AdrScImm(hn.x0));
    //
    hn.St1h(hn.z0.h, hn.p0, AdrNoOfs(hn.x1));
    hn.Ret();
    hn.Ready();
    auto hnFunc = hn.GetCode<void (*)(uint16_t*, uint16_t*)>();
    uint16_t hnSrc[1] = {1};
    uint16_t hnDst[32] = {0};
    hnFunc(hnSrc, hnDst);
    ASSERT_EQ(hnSrc[0], hnDst[0]);

    GTestJitGen sn; // s(ZRegS)n(AdrNoOfs)
    sn.Ptrue(sn.p0.s);
    //
    sn.Ld1rh(sn.z0.s, sn.p0 / PredType::T_Z, AdrScImm(sn.x0));
    //
    sn.St1w(sn.z0.s, sn.p0, AdrNoOfs(sn.x1));
    sn.Ret();
    sn.Ready();
    auto snFunc = sn.GetCode<void (*)(uint16_t*, uint32_t*)>();
    uint16_t snSrc[1] = {1};
    uint32_t snDst[16] = {0};
    snFunc(snSrc, snDst);
    ASSERT_EQ(snSrc[0], snDst[0]);
}

TEST(armcg_test_t, TestLd1rqx)
{
    GTestJitGen cg1;
    cg1.Ptrue(cg1.p0.b);
    cg1.Ld1rqb(cg1.z0.b, cg1.p0 / PredType::T_Z, AdrNoOfs(cg1.x0));
    cg1.St1b(cg1.z0.b, cg1.p0, AdrNoOfs(cg1.x1));
    cg1.Ret();
    cg1.Ready();
    auto f1 = cg1.GetCode<void (*)(uint8_t*, uint8_t*)>();
    uint8_t src1[16] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
    uint8_t dst1[64] = {0};
    f1(src1, dst1);
    ASSERT_EQ(dst1[0], 1);
    ASSERT_EQ(dst1[1], 2);
    ASSERT_EQ(dst1[16], 1);  // replicate
    ASSERT_EQ(dst1[17], 2);

    GTestJitGen cg2;
    cg2.Ptrue(cg2.p0.d);
    cg2.Ld1rqd(cg2.z0.d, cg2.p0 / PredType::T_Z, AdrNoOfs(cg2.x0));
    cg2.St1d(cg2.z0.d, cg2.p0, AdrNoOfs(cg2.x1));
    cg2.Ret();
    cg2.Ready();
    auto f2 = cg2.GetCode<void (*)(uint64_t*, uint64_t*)>();
    uint64_t src2[2] = {1, 2};
    uint64_t dst2[8] = {0};
    f2(src2, dst2);
    ASSERT_EQ(dst2[0], 1);
    ASSERT_EQ(dst2[1], 2);
    ASSERT_EQ(dst2[2], 1);  // replicate
    ASSERT_EQ(dst2[3], 2);
}

TEST(armcg_test_t, TestLd1rqw)
{
    GTestJitGen si; // s(ZRegS)i(AdrImm)
    si.Ptrue(si.p0.s);
    //
    si.Ld1rqw(si.z0.s, si.p0 / PredType::T_Z, AdrImm(si.x0, 0));
    //
    si.St1w(si.z0.s, si.p0, AdrNoOfs(si.x1));
    si.Ret();
    si.Ready();
    auto siFunc = si.GetCode<void (*)(uint32_t*, uint32_t*)>();
    uint32_t siSrc[8] = {1, 2, 3, 4, 5, 6, 7, 8};
    uint32_t siDst[16] = {0};
    siFunc(siSrc, siDst);
    ASSERT_EQ(siSrc[0], siDst[0]);

    GTestJitGen sn; // s(ZRegS)n(AdrNoOfs)
    sn.Ptrue(sn.p0.s);
    //
    sn.Ld1rqw(sn.z0.s, sn.p0 / PredType::T_Z, AdrNoOfs(sn.x0));
    //
    sn.St1w(sn.z0.s, sn.p0, AdrNoOfs(sn.x1));
    sn.Ret();
    sn.Ready();
    auto snFunc = sn.GetCode<void (*)(uint32_t*, uint32_t*)>();
    uint32_t snSrc[8] = {1, 2, 3, 4, 5, 6, 7, 8};
    uint32_t snDst[16] = {0};
    snFunc(snSrc, snDst);
    ASSERT_EQ(snSrc[0], snDst[0]);

    GTestJitGen ssi; // s(ZRegS)si(AdrScImm)
    ssi.Ptrue(ssi.p0.s);
    //
    ssi.Ld1rqw(ssi.z0.s, ssi.p0 / PredType::T_Z, AdrScImm(ssi.x0, 0));
    //
    ssi.St1w(ssi.z0.s, ssi.p0, AdrNoOfs(ssi.x1));
    ssi.Ret();
    ssi.Ready();
    auto ssiFunc = ssi.GetCode<void (*)(uint32_t*, uint32_t*)>();
    uint32_t ssiSrc[8] = {1, 2, 3, 4, 5, 6, 7, 8};
    uint32_t ssiDst[16] = {0};
    ssiFunc(ssiSrc, ssiDst);
    ASSERT_EQ(ssiSrc[0], ssiDst[0]);
}

TEST(armcg_test_t, TestLd1rsb)
{
    GTestJitGen sn; // s(ZRegS)n(AdrNoOfs)
    sn.Ptrue(sn.p0.s);
    //
    sn.Ld1rsb(sn.z0.s, sn.p0 / PredType::T_Z, AdrNoOfs(sn.x0));
    //
    sn.St1w(sn.z0.s, sn.p0, AdrNoOfs(sn.x1));
    sn.Ret();
    sn.Ready();
    auto snFunc = sn.GetCode<void (*)(uint8_t*, uint32_t*)>();
    uint8_t snSrc[1] = {1};
    uint32_t snDst[16] = {0};
    snFunc(snSrc, snDst);
    ASSERT_EQ(snSrc[0], snDst[0]);
}

TEST(armcg_test_t, TestLd1rw)
{
    GTestJitGen si; // s(ZRegS)i(AdrImm)
    si.Ptrue(si.p0.s);
    //
    si.Ld1rw(si.z0.s, si.p0 / PredType::T_Z, AdrImm(si.x0, 0));
    //
    si.St1w(si.z0.s, si.p0, AdrNoOfs(si.x1));
    si.Ret();
    si.Ready();
    auto siFunc = si.GetCode<void (*)(uint32_t*, uint32_t*)>();
    uint32_t siSrc[1] = {1};
    uint32_t siDst[16] = {0};
    siFunc(siSrc, siDst);
    ASSERT_EQ(siSrc[0], siDst[0]);

    GTestJitGen sn; // s(ZRegS)n(AdrNoOfs)
    sn.Ptrue(sn.p0.s);
    //
    sn.Ld1rw(sn.z0.s, sn.p0 / PredType::T_Z, AdrNoOfs(sn.x0));
    //
    sn.St1w(sn.z0.s, sn.p0, AdrNoOfs(sn.x1));
    sn.Ret();
    sn.Ready();
    auto snFunc = sn.GetCode<void (*)(uint32_t*, uint32_t*)>();
    uint32_t snSrc[1] = {1};
    uint32_t snDst[16] = {0};
    snFunc(snSrc, snDst);
    ASSERT_EQ(snSrc[0], snDst[0]);

    GTestJitGen ssi; // s(ZRegS)si(AdrScImm)
    ssi.Ptrue(ssi.p0.s);
    //
    ssi.Ld1rw(ssi.z0.s, ssi.p0 / PredType::T_Z, AdrScImm(ssi.x0));
    //
    ssi.St1w(ssi.z0.s, ssi.p0, AdrScImm(ssi.x1));
    ssi.Ret();
    ssi.Ready();
    auto ssiFunc = ssi.GetCode<void (*)(uint32_t*, uint32_t*)>();
    uint32_t ssiSrc[1] = {1};
    uint32_t ssiDst[16] = {0};
    ssiFunc(ssiSrc, ssiDst);
    ASSERT_EQ(ssiSrc[0], ssiDst[0]);
}

TEST(armcg_test_t, TestLd1sb)
{
    GTestJitGen sn; // s(ZRegS)n(AdrNoOfs)
    sn.Ptrue(sn.p0.b);
    //
    sn.Ld1sb(sn.z0.s, sn.p0 / PredType::T_Z, AdrNoOfs(sn.x0));
    //
    sn.St1b(sn.z0.b, sn.p0, AdrNoOfs(sn.x1));
    sn.Ret();
    sn.Ready();
    auto snFunc = sn.GetCode<void (*)(uint8_t*, uint8_t*)>();
    uint8_t snSrc[64] = {1};
    uint8_t snDst[64] = {0};
    snFunc(snSrc, snDst);
    ASSERT_EQ(snSrc[0], snDst[0]);

    GTestJitGen ssi; // s(ZRegS)si(AdrScImm)
    ssi.Ptrue(ssi.p0.b);
    //
    ssi.Ld1sb(ssi.z0.s, ssi.p0 / PredType::T_Z, AdrScImm(ssi.x0));
    //
    ssi.St1b(ssi.z0.b, ssi.p0, AdrNoOfs(ssi.x1));
    ssi.Ret();
    ssi.Ready();
    auto ssiFunc = ssi.GetCode<void (*)(uint8_t*, uint8_t*)>();
    uint8_t ssiSrc[64] = {1};
    uint8_t ssiDst[64] = {0};
    ssiFunc(ssiSrc, ssiDst);
    ASSERT_EQ(ssiSrc[0], ssiDst[0]);

    GTestJitGen ss3u; // s(ZRegS)s3u(AdrSc32U)
    ss3u.Ptrue(ss3u.p0.s);
    ss3u.Index(ss3u.z0.s, 0, 1);
    //
    ss3u.Ld1sb(ss3u.z1.s, ss3u.p0 / PredType::T_Z, AdrSc32U(ss3u.x0, ss3u.z0.s, ExtMod::UXTW));
    //
    ss3u.St1w(ss3u.z1.s, ss3u.p0, AdrNoOfs(ss3u.x1));
    ss3u.Ret();
    ss3u.Ready();
    auto ss3uFunc = ss3u.GetCode<void (*)(uint8_t*, uint32_t*)>();
    uint8_t ss3uSrc[16] = {1};
    uint32_t ss3uDst[16] = {0};
    ss3uFunc(ss3uSrc, ss3uDst);
    ASSERT_EQ(ss3uSrc[0], ss3uDst[0]);
}

TEST(armcg_test_t, TestLd1w)
{
    GTestJitGen si; // s(ZRegS)i(AdrImm)
    si.Ptrue(si.p0.s);
    //
    si.Ld1w(si.z0.s, si.p0 / PredType::T_Z, AdrImm(si.x0, 0));
    //
    si.St1w(si.z0.s, si.p0, AdrNoOfs(si.x1));
    si.Ret();
    si.Ready();
    auto siFunc = si.GetCode<void (*)(uint32_t*, uint32_t*)>();
    uint32_t siSrc[16] = {1};
    uint32_t siDst[16] = {0};
    siFunc(siSrc, siDst);
    ASSERT_EQ(siSrc[0], siDst[0]);

    GTestJitGen sn; // s(ZRegS)n(AdrNoOfs)
    sn.Ptrue(sn.p0.s);
    //
    sn.Ld1w(sn.z0.s, sn.p0 / PredType::T_Z, AdrNoOfs(sn.x0));
    //
    sn.St1w(sn.z0.s, sn.p0, AdrNoOfs(sn.x1));
    sn.Ret();
    sn.Ready();
    auto snFunc = sn.GetCode<void (*)(uint32_t*, uint32_t*)>();
    uint32_t snSrc[16] = {1};
    uint32_t snDst[16] = {0};
    snFunc(snSrc, snDst);
    ASSERT_EQ(snSrc[0], snDst[0]);

    GTestJitGen ssi; // s(ZRegS)si(AdrScImm)
    ssi.Ptrue(ssi.p0.s);
    //
    ssi.Ld1w(ssi.z0.s, ssi.p0 / PredType::T_Z, AdrScImm(ssi.x0));
    //
    ssi.St1w(ssi.z0.s, ssi.p0, AdrScImm(ssi.x1));
    ssi.Ret();
    ssi.Ready();
    auto ssiFunc = ssi.GetCode<void (*)(uint32_t*, uint32_t*)>();
    uint32_t ssiSrc[16] = {1};
    uint32_t ssiDst[16] = {0};
    ssiFunc(ssiSrc, ssiDst);
    ASSERT_EQ(ssiSrc[0], ssiDst[0]);

    GTestJitGen sss; // s(ZRegS)ss(AdrScSc)
    sss.Ptrue(sss.p0.s);
    sss.Index(sss.z0.s, 0, 4);
    sss.Mov(sss.x9, 0);
    //
    sss.Ld1w(sss.z1.s, sss.p0 / PredType::T_Z, AdrScSc(sss.x0, sss.x9, ShMod::LSL, 2));
    //
    sss.St1w(sss.z1.s, sss.p0, AdrNoOfs(sss.x1));
    sss.Ret();
    sss.Ready();
    auto sssFunc = sss.GetCode<void (*)(uint32_t*, uint32_t*)>();
    uint32_t sssSrc[16] = {1};
    uint32_t sssDst[16] = {0};
    sssFunc(sssSrc, sssDst);
    ASSERT_EQ(sssSrc[0], sssDst[0]);

    GTestJitGen ss3u; // s(ZRegS)s3u(AdrSc32U)
    ss3u.Ptrue(ss3u.p0.s);
    ss3u.Index(ss3u.z0.s, 0, 4);
    //
    ss3u.Ld1w(ss3u.z1.s, ss3u.p0 / PredType::T_Z, AdrSc32U(ss3u.x0, ss3u.z0.s, ExtMod::UXTW));
    //
    ss3u.St1w(ss3u.z1.s, ss3u.p0, AdrNoOfs(ss3u.x1));
    ss3u.Ret();
    ss3u.Ready();
    auto ss3uFunc = ss3u.GetCode<void (*)(uint32_t*, uint32_t*)>();
    uint32_t ss3uSrc[16] = {1};
    uint32_t ss3uDst[16] = {0};
    ss3uFunc(ss3uSrc, ss3uDst);
    ASSERT_EQ(ss3uSrc[0], ss3uDst[0]);

    GTestJitGen ss3s; // s(ZRegS)s3s(AdrSc32S)
    ss3s.Ptrue(ss3s.p0.s);
    ss3s.Index(ss3s.z0.s, 0, 4);
    //
    ss3s.Ld1w(ss3s.z1.s, ss3s.p0 / PredType::T_Z, AdrSc32S(ss3s.x0, ss3s.z0.s, ExtMod::UXTW, 0));
    //
    ss3s.St1w(ss3s.z1.s, ss3s.p0, AdrNoOfs(ss3s.x1));
    ss3s.Ret();
    ss3s.Ready();
    auto ss3sFunc = ss3s.GetCode<void (*)(uint32_t*, uint32_t*)>();
    uint32_t ss3sSrc[16] = {1};
    uint32_t ss3sDst[16] = {0};
    ss3sFunc(ss3sSrc, ss3sDst);
    ASSERT_EQ(ss3sSrc[0], ss3sDst[0]);

    GTestJitGen dn; // d(ZRegD)n(AdrNoOfs)
    dn.Ptrue(dn.p0.s);
    //
    dn.Ld1w(dn.z0.d, dn.p0 / PredType::T_Z, AdrNoOfs(dn.x0));
    //
    dn.St1w(dn.z0.s, dn.p0, AdrNoOfs(dn.x1)); // not suit
    dn.Ret();
    dn.Ready();
    auto dnFunc = dn.GetCode<void (*)(uint32_t*, uint32_t*)>();
    uint32_t dnSrc[16] = {1};
    uint32_t dnDst[16] = {0};
    dnFunc(dnSrc, dnDst);
    ASSERT_EQ(dnSrc[0], dnDst[0]);
}
#ifdef KUDNNL_920F
TEST(armcg_test_t, TestMCLd1wMem2HTile)
{
    GTestJitGen cg;
    cg.Smstart(StreamMode::SMZA);
    cg.Mov(cg.w15, 0);
    cg.Ptrue(cg.p0.s);
    cg.Mov(cg.x9, 0);
    cg.Ld1w(cg.za0h.s, cg.w15, 0, cg.p0 / PredType::T_Z,
        AdrScSc(cg.x1, cg.x9, ShMod::LSL, static_cast<uint32_t>(2)));
    cg.St1w(cg.za0h.s, cg.w15, 0, cg.p0,
        AdrScSc(cg.x0, cg.x9, ShMod::LSL, static_cast<uint32_t>(2)));
    cg.Smstop(StreamMode::SMZA);
    cg.Ret();
    cg.Ready();
    auto cgFunc = cg.GetCode<void (*)(float*, float*)>();
    float cgSrc[16] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
    float cgDst[16] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    cgFunc(cgDst, cgSrc);
    ASSERT_EQ(cgSrc[0], cgDst[0]);
    ASSERT_EQ(cgSrc[1], cgDst[1]);
}
#endif

TEST(armcg_test_t, TestLd4)
{
    GTestJitGen den; // de(VRegDElem)n(AdrNoOfs)
    den.Mov(den.x9, 32);
    //
    den.Ld4((den.v0.d2 - den.v3.d2)[0], AdrNoOfs(den.x0));
    den.St4((den.v0.d2 - den.v3.d2)[0], AdrPostReg(den.x1, den.x9));
    //
    den.Ret();
    den.Ready();
    auto denFunc = den.GetCode<void (*)(uint64_t*, uint64_t*)>();
    uint64_t denSrc[4] = {1, 2, 3, 4};
    uint64_t denDst[4] = {0};
    denFunc(denSrc, denDst);
    ASSERT_EQ(denSrc[0], denDst[0]);

    GTestJitGen deoi; // de(VRegDElem)oi(AdrPostImm)
    //
    deoi.Ld4((deoi.v0.d2 - deoi.v3.d2)[0], AdrPostImm(deoi.x0, 32));
    deoi.St4((deoi.v0.d2 - deoi.v3.d2)[0], AdrPostImm(deoi.x1, 32));
    //
    deoi.Ret();
    deoi.Ready();
    auto deoiFunc = deoi.GetCode<void (*)(uint64_t*, uint64_t*)>();
    uint64_t deoiSrc[4] = {1, 2, 3, 4};
    uint64_t deoiDst[4] = {0};
    deoiFunc(deoiSrc, deoiDst);
    ASSERT_EQ(deoiSrc[0], deoiDst[0]);
}

TEST(armcg_test_t, TestLdaddal)
{
    GTestJitGen xxn; // x(XReg)x(XReg)n(AdrNoOfs)
    xxn.Mov(xxn.x9, 1);
    //
    xxn.Ldaddal(xxn.x9, xxn.x10, AdrNoOfs(xxn.x0));
    //
    xxn.Ret();
    xxn.Ready();
    auto xxnFunc = xxn.GetCode<void (*)(uint64_t*)>();
    uint64_t xxnSrc[1] = {1};
    xxnFunc(xxnSrc);
    ASSERT_EQ(xxnSrc[0], 2);
}

TEST(armcg_test_t, TestLdaxr)
{
    GTestJitGen xi; // x(XReg)i(AdrImm)
    ARMCG::Label xiloop;
    xi.L(xiloop);
    //
    xi.Ldaxr(xi.x9, AdrImm(xi.x0, 0));
    //
    xi.Cbnz(xi.x9, xiloop);
    xi.Stlxr(xi.w9, xi.x1, AdrImm(xi.x0, 0));
    xi.Cbnz(xi.x9, xiloop);
    // enter critical section
    xi.Ret();
    xi.Ready();
    auto xiFunc = xi.GetCode<void (*)(uint64_t*, uint64_t)>();
    uint64_t xiSrc[1] = {0};
    xiFunc(xiSrc, 1);
    ASSERT_EQ(xiSrc[0], 1);
    WReg w1(500);
    WReg w2(600);
    EXPECT_THROW(xi.Stlxr(w1, w2, AdrImm(xi.x0, 0)), std::exception);
}

TEST(armcg_test_t, TestLdp)
{
    GTestJitGen xxoi; // x(XReg)x(XReg)oi(AdrPostImm)
    //
    xxoi.Ldp(xxoi.x9, xxoi.x10, AdrPostImm(xxoi.x0, 16));
    xxoi.Stp(xxoi.x9, xxoi.x10, AdrPostImm(xxoi.x0, 0));
    //
    xxoi.Ret();
    xxoi.Ready();
    auto xxoiFunc = xxoi.GetCode<void (*)(uint64_t*)>();
    uint64_t xxoiSrc[4] = {1, 2, 0, 0};
    xxoiFunc(xxoiSrc);
    ASSERT_EQ(xxoiSrc[0], xxoiSrc[2]);
}

TEST(armcg_test_t, TestLdpStp)
{
    GTestJitGen cg1;
    cg1.Ldp(cg1.d9, cg1.d10, AdrPostImm(cg1.x0, 16));
    cg1.Stp(cg1.d9, cg1.d10, AdrImm(cg1.x0, 0));
    cg1.Ret();
    cg1.Ready();
    auto f1 = cg1.GetCode<void (*)(uint64_t*)>();
    uint64_t src1[4] = {1, 2, 0, 0};
    f1(src1);
    ASSERT_EQ(src1[2], 1);
    ASSERT_EQ(src1[3], 2);

    GTestJitGen cg2;
    cg2.Ldp(cg2.d9, cg2.d10, AdrImm(cg2.x0, 0));
    cg2.Stp(cg2.d9, cg2.d10, AdrPreImm(cg2.x0, 16));
    cg2.Ret();
    cg2.Ready();
    auto f2 = cg2.GetCode<void (*)(uint64_t*)>();
    uint64_t src2[4] = {1, 2, 0, 0};
    f2(src2);
    ASSERT_EQ(src2[2], 1);
    ASSERT_EQ(src2[3], 2);

    GTestJitGen cg3;
    cg3.Ldp(cg3.d9, cg3.d10, AdrPreImm(cg3.x0, 16));
    cg3.Sub(cg3.x0, cg3.x0, 0x10);
    cg3.Stp(cg3.d9, cg3.d10, AdrPostImm(cg3.x0, 16));
    cg3.Ret();
    cg3.Ready();
    auto f3 = cg3.GetCode<void (*)(uint64_t*)>();
    uint64_t src3[4] = {0, 0, 1, 2};
    f3(src3);
    ASSERT_EQ(src3[0], 1);
    ASSERT_EQ(src3[1], 2);

    GTestJitGen cg4;
    cg4.Ldp(cg4.s9, cg4.s10, AdrPostImm(cg4.x0, 8));
    cg4.Stp(cg4.s9, cg4.s10, AdrImm(cg4.x0, 0));
    cg4.Ret();
    cg4.Ready();
    auto f4 = cg4.GetCode<void (*)(uint32_t*)>();
    uint32_t src4[4] = {1, 2, 0, 0};
    f4(src4);
    ASSERT_EQ(src4[2], 1);
    ASSERT_EQ(src4[3], 2);

    GTestJitGen cg5;
    cg5.Ldp(cg5.q9, cg5.q10, AdrPostImm(cg5.x0, 32));
    cg5.Stp(cg5.q9, cg5.q10, AdrImm(cg5.x0, 0));
    cg5.Ret();
    cg5.Ready();
    auto f5 = cg5.GetCode<void (*)(uint64_t*)>();
    uint64_t src5[8] = {1, 2, 3, 4, 0, 0, 0, 0};
    f5(src5);
    ASSERT_EQ(src5[4], 1);
    ASSERT_EQ(src5[5], 2);
    ASSERT_EQ(src5[6], 3);
    ASSERT_EQ(src5[7], 4);
}

TEST(acg_mne_load_test_t, TestLdr)
{
    GTestJitGen woi; // w(WReg)oi(AdrPostImm)
    //
    woi.Ldr(woi.w0, AdrPostImm(woi.x0, 0));
    //
    woi.Ret();
    woi.Ready();
    auto woiFunc = woi.GetCode<uint32_t (*)(uint32_t*)>();
    uint32_t woiSrc[1] = {1};
    ASSERT_EQ(1, woiFunc(woiSrc));

    GTestJitGen xoi; // x(XReg)oi(AdrPostImm)
    //
    xoi.Ldr(xoi.x0, AdrPostImm(xoi.x0, 0));
    //
    xoi.Str(xoi.x0, AdrPostImm(xoi.x1, 0));
    xoi.Ret();
    xoi.Ready();
    auto xoiFunc = xoi.GetCode<void (*)(uint64_t*, uint64_t*)>();
    uint64_t xoiSrc[1] = {1};
    uint64_t xoiDst[1] = {0};
    xoiFunc(xoiSrc, xoiDst);
    ASSERT_EQ(xoiSrc[0], xoiDst[0]);

    GTestJitGen wi; // w(WReg)i(AdrImm)
    //
    wi.Ldr(wi.w0, AdrImm(wi.x0, 0));
    //
    wi.Str(wi.w0, AdrImm(wi.x1, 0));
    wi.Ret();
    wi.Ready();
    auto wiFunc = wi.GetCode<void (*)(uint32_t*, uint32_t*)>();
    uint32_t wiSrc[1] = {1};
    uint32_t wiDst[1] = {0};
    wiFunc(wiSrc, wiDst);
    ASSERT_EQ(wiSrc[0], wiDst[0]);

    GTestJitGen wn; // w(WReg)n(AdrNoOfs)
    //
    wn.Ldr(wn.w0, AdrNoOfs(wn.x0));
    //
    wn.Str(wn.w0, AdrNoOfs(wn.x1));
    wn.Ret();
    wn.Ready();
    auto wnFunc = wn.GetCode<void (*)(uint32_t*, uint32_t*)>();
    uint32_t wnSrc[1] = {1};
    uint32_t wnDst[1] = {0};
    wnFunc(wnSrc, wnDst);
    ASSERT_EQ(wnSrc[0], wnDst[0]);

    GTestJitGen wu; // w(WReg)u(AdrUimm)
    //
    wu.Ldr(wu.w0, AdrUimm(wu.x0, 0));
    //
    wu.Str(wu.w0, AdrUimm(wu.x1, 0));
    wu.Ret();
    wu.Ready();
    auto wuFunc = wu.GetCode<void (*)(uint32_t*, uint32_t*)>();
    uint32_t wuSrc[1] = {1};
    uint32_t wuDst[1] = {0};
    wuFunc(wuSrc, wuDst);
    ASSERT_EQ(wuSrc[0], wuDst[0]);

    GTestJitGen xi; // x(XReg)i(AdrImm)
    //
    xi.Ldr(xi.x0, AdrImm(xi.x0, 0));
    //
    xi.Str(xi.x0, AdrImm(xi.x1, 0));
    xi.Ret();
    xi.Ready();
    auto xiFunc = xi.GetCode<void (*)(uint64_t*, uint64_t*)>();
    uint64_t xiSrc[1] = {1};
    uint64_t xiDst[1] = {0};
    xiFunc(xiSrc, xiDst);
    ASSERT_EQ(xiSrc[0], xiDst[0]);

    GTestJitGen xn; // x(XReg)n(AdrNoOfs)
    //
    xn.Ldr(xn.x0, AdrNoOfs(xn.x0));
    //
    xn.Str(xn.x0, AdrNoOfs(xn.x1));
    xn.Ret();
    xn.Ready();
    auto xnFunc = xn.GetCode<void (*)(uint64_t*, uint64_t*)>();
    uint64_t xnSrc[1] = {1};
    uint64_t xnDst[1] = {0};
    xnFunc(xnSrc, xnDst);
    ASSERT_EQ(xnSrc[0], xnDst[0]);

    GTestJitGen xu; // x(XReg)u(AdrUimm)
    //
    xu.Ldr(xu.x0, AdrUimm(xu.x0, 0));
    //
    xu.Str(xu.x0, AdrUimm(xu.x1, 0));
    xu.Ret();
    xu.Ready();
    auto xuFunc = xu.GetCode<void (*)(uint64_t*, uint64_t*)>();
    uint64_t xuSrc[1] = {1};
    uint64_t xuDst[1] = {0};
    xuFunc(xuSrc, xuDst);
    ASSERT_EQ(xuSrc[0], xuDst[0]);

    GTestJitGen bu; // b(BReg)u(AdrUimm)
    //
    bu.Ldr(bu.b0, AdrUimm(bu.x0, 0));
    //
    bu.Str(bu.b0, AdrUimm(bu.x1, 0));
    bu.Ret();
    bu.Ready();
    auto buFunc = bu.GetCode<void (*)(uint8_t*, uint8_t*)>();
    uint8_t buSrc[16] = {1};
    uint8_t buDst[16] = {0};
    buFunc(buSrc, buDst);
    ASSERT_EQ(buSrc[0], buDst[0]);

    GTestJitGen hu; // h(HReg)u(AdrUimm)
    //
    hu.Ldr(hu.h0, AdrUimm(hu.x0, 0));
    //
    hu.Str(hu.h0, AdrUimm(hu.x1, 0));
    hu.Ret();
    hu.Ready();
    auto huFunc = hu.GetCode<void (*)(uint16_t*, uint16_t*)>();
    uint16_t huSrc[8] = {1};
    uint16_t huDst[8] = {0};
    huFunc(huSrc, huDst);
    ASSERT_EQ(huSrc[0], huDst[0]);

    GTestJitGen su; // s(SReg)u(AdrUimm)
    //
    su.Ldr(su.s0, AdrUimm(su.x0, 0));
    su.Str(su.s0, AdrUimm(su.x1, 0));
    su.Mov(su.z0.s, su.s0);
    su.Ret();
    su.Ready();
    auto suFunc = su.GetCode<void (*)(uint32_t*, uint32_t*)>();
    uint32_t suSrc[4] = {1, 2, 3, 4};
    uint32_t suDst[4] = {0};
    suFunc(suSrc, suDst);
    ASSERT_EQ(suSrc[0], suDst[0]);

    GTestJitGen du; // d(DReg)u(AdrUimm)
    //
    du.Ldr(du.d0, AdrUimm(du.x0, 0));
    //
    du.Str(du.d0, AdrUimm(du.x1, 0));
    du.Ret();
    du.Ready();
    auto duFunc = du.GetCode<void (*)(uint64_t*, uint64_t*)>();
    uint64_t duSrc[2] = {1};
    uint64_t duDst[2] = {0};
    duFunc(duSrc, duDst);
    ASSERT_EQ(duSrc[0], duDst[0]);

    GTestJitGen qu; // q(QReg)u(AdrUimm)
    //
    qu.Ldr(qu.q0, AdrUimm(qu.x0, 0));
    //
    qu.Str(qu.q0, AdrUimm(qu.x1, 0));
    qu.Ret();
    qu.Ready();
    auto quFunc = qu.GetCode<void (*)(uint64_t*, uint64_t*)>();
    uint64_t quSrc[2] = {1};
    uint64_t quDst[2] = {0};
    quFunc(quSrc, quDst);
    ASSERT_EQ(quSrc[0], quDst[0]);

    GTestJitGen pn; // p(PReg)n(AdrNoOfs)
    //
    pn.Ldr(pn.p0, AdrNoOfs(pn.x0));
    //
    pn.Str(pn.p0, AdrNoOfs(pn.x1));
    pn.Ret();
    pn.Ready();
    auto pnFunc = pn.GetCode<void (*)(uint32_t*, uint32_t*)>();
    uint32_t pnSrc[2] = {1, 2};
    uint32_t pnDst[2] = {0};
    pnFunc(pnSrc, pnDst);
    ASSERT_EQ(pnSrc[0], pnDst[0]);

    GTestJitGen psi; // p(PReg)si(AdrScImm)
    //
    psi.Ldr(psi.p0, AdrScImm(psi.x0, 0));
    //
    psi.Str(psi.p0, AdrScImm(psi.x1, 0));
    psi.Ret();
    psi.Ready();
    auto psiFunc = psi.GetCode<void (*)(uint32_t*, uint32_t*)>();
    uint32_t psiSrc[2] = {1, 2};
    uint32_t psiDst[2] = {0};
    psiFunc(psiSrc, psiDst);
    ASSERT_EQ(psiSrc[0], psiDst[0]);

    GTestJitGen zn; // z(ZReg)n(AdrNoOfs)
    //
    zn.Ldr(zn.z0, AdrNoOfs(zn.x0));
    //
    zn.Str(zn.z0, AdrNoOfs(zn.x1));
    zn.Ret();
    zn.Ready();
    auto znFunc = zn.GetCode<void (*)(uint32_t*, uint32_t*)>();
    uint32_t znSrc[16] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
    uint32_t znDst[16] = {0};
    znFunc(znSrc, znDst);
    ASSERT_EQ(znSrc[0], znDst[0]);

    GTestJitGen zsi; // z(ZReg)si(AdrScImm)
    //
    zsi.Ldr(zsi.z0, AdrScImm(zsi.x0, 0));
    //
    zsi.Str(zsi.z0, AdrScImm(zsi.x1, 0));
    zsi.Ret();
    zsi.Ready();
    auto zsiFunc = zsi.GetCode<void (*)(uint32_t*, uint32_t*)>();
    uint32_t zsiSrc[16] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
    uint32_t zsiDst[16] = {0};
    zsiFunc(zsiSrc, zsiDst);
    ASSERT_EQ(zsiSrc[0], zsiDst[0]);
}

TEST(armcg_test_t, TestLdrb)
{
    GTestJitGen wu; // w(WReg)u(AdrUimm)
    //
    wu.Ldrb(wu.w0, AdrUimm(wu.x0, 0));
    //
    wu.Strb(wu.w0, AdrUimm(wu.x1, 0));
    wu.Ret();
    wu.Ready();
    auto wuFunc = wu.GetCode<void (*)(uint8_t*, uint8_t*)>();
    uint8_t wuSrc[1] = {1};
    uint8_t wuDst[1] = {0};
    wuFunc(wuSrc, wuDst);
    ASSERT_EQ(wuSrc[0], wuDst[0]);
}

TEST(armcg_test_t, TestLdrh)
{
    GTestJitGen wu; // w(WReg)u(AdrUimm)
    //
    wu.Ldrh(wu.w0, AdrUimm(wu.x0, 0));
    //
    wu.Strh(wu.w0, AdrUimm(wu.x1, 0));
    wu.Ret();
    wu.Ready();
    auto wuFunc = wu.GetCode<void (*)(uint16_t*, uint16_t*)>();
    uint16_t wuSrc[1] = {1};
    uint16_t wuDst[1] = {0};
    wuFunc(wuSrc, wuDst);
    ASSERT_EQ(wuSrc[0], wuDst[0]);
}

TEST(armcg_test_t, TestLdrsw)
{
    GTestJitGen xu; // x(XReg)u(AdrUimm)
    //
    xu.Ldrsw(xu.x0, AdrUimm(xu.x0, 0));
    //
    xu.Ret();
    xu.Ready();
    auto xuFunc = xu.GetCode<uint32_t (*)(uint32_t*)>();
    uint32_t xuSrc[1] = {1};
    ASSERT_EQ(xuSrc[0], xuFunc(xuSrc));

    GTestJitGen xoi; // x(XReg)oi(AdrPostimm)
    //
    xoi.Ldrsw(xoi.x0, AdrPostImm(xoi.x0, 0));
    //
    xoi.Ret();
    xoi.Ready();
    auto xoiFunc = xoi.GetCode<uint32_t (*)(uint32_t*)>();
    uint32_t xoiSrc[1] = {1};
    ASSERT_EQ(xoiSrc[0], xoiFunc(xoiSrc));
}

TEST(armcg_test_t, TestLdurb)
{
    GTestJitGen wi; // w(WReg)i(AdrImm)
    //
    wi.Ldurb(wi.w0, AdrImm(wi.x0, 0));
    //
    wi.Ret();
    wi.Ready();
    auto wiFunc = wi.GetCode<uint32_t (*)(uint8_t*)>();
    uint8_t wiSrc[1] = {1};
    ASSERT_EQ(wiFunc(wiSrc), 1);
}

TEST(armcg_test_t, TestLdur)
{
    GTestJitGen cg;
    cg.Ldur(cg.q1, AdrImm(cg.x0, 16));
    cg.Str(cg.q1, AdrUimm(cg.x0, 0));

    cg.Ret();
    cg.Ready();
    auto f = cg.GetCode<void (*)(uint64_t*)>();
    uint64_t arr[4] = {0, 0, 1, 2};
    f(arr);
    ASSERT_EQ(arr[0], 1);
    ASSERT_EQ(arr[1], 2);

    GTestJitGen cg2;
    cg2.Mov(cg2.w9, 2);
    cg2.Stur(cg2.w9, AdrImm(cg2.x0, 0));
    cg2.Ret();
    cg2.Ready();
    auto f2 = cg2.GetCode<void (*)(uint32_t*)>();
    uint32_t ret = 0;
    f2(&ret);
    ASSERT_EQ(ret, 2);
}

TEST(armcg_test_t, TestCpy)
{
    GTestJitGen sw; // s(ZRegS)w(WReg)
    sw.Ptrue(sw.p0.s);
    //
    sw.Cpy(sw.z0.s, sw.p0 / PredType::T_M, sw.w0);
    //
    sw.St1w(sw.z0.s, sw.p0, AdrNoOfs(sw.x1));
    sw.Ret();
    sw.Ready();
    auto swFunc = sw.GetCode<void (*)(uint32_t, uint32_t*)>();
    uint32_t swDst[16] = {0};
    swFunc(1, swDst);
    ASSERT_EQ(1, swDst[0]);
}

TEST(armcg_test_t, TestDup)
{
    GTestJitGen b16w; // b16(VReg16B)w(WReg)
    //
    b16w.Dup(b16w.v0.b, b16w.w0);
    //
    b16w.St1(b16w.v0.b[0], AdrNoOfs(b16w.x1));
    b16w.Ret();
    b16w.Ready();
    auto b16wFunc = b16w.GetCode<void (*)(uint8_t, uint8_t*)>();
    uint8_t b16wDst[1] = {0};
    b16wFunc(1, b16wDst);
    ASSERT_EQ(1, b16wDst[0]);

    GTestJitGen s4w; // s4(VReg4S)w(WReg)
    //
    s4w.Dup(s4w.v0.s, s4w.w0);
    //
    s4w.St1(s4w.v0.s[0], AdrNoOfs(s4w.x1));
    s4w.Ret();
    s4w.Ready();
    auto s4wFunc = s4w.GetCode<void (*)(uint32_t, uint32_t*)>();
    uint32_t s4wDst[1] = {0};
    s4wFunc(1, s4wDst);
    ASSERT_EQ(1, s4wDst[0]);

    GTestJitGen b16be; // b16(VReg16B)be(VRegBElem)
    b16be.Dup(b16be.v0.b, b16be.w0);
    //
    b16be.Dup(b16be.v1.b, b16be.v0.b[0]);
    //
    b16be.St1(b16be.v1.b[0], AdrNoOfs(b16be.x1));
    b16be.Ret();
    b16be.Ready();
    auto b16beFunc = b16be.GetCode<void (*)(uint8_t, uint8_t*)>();
    uint8_t b16beDst[1] = {0};
    b16beFunc(1, b16beDst);
    ASSERT_EQ(1, b16beDst[0]);

    GTestJitGen hw; // h(ZRegH)w(WReg)
    //
    hw.Dup(hw.z0.h, hw.w0);
    //
    hw.Ptrue(hw.p0.h);
    hw.St1h(hw.z0.h, hw.p0, AdrNoOfs(hw.x1));
    hw.Ret();
    hw.Ready();
    auto hwFunc = hw.GetCode<void (*)(uint16_t, uint16_t*)>();
    uint16_t hwDst[32] = {0};
    hwFunc(1, hwDst);
    ASSERT_EQ(1, hwDst[0]);

    GTestJitGen sw; // s(ZRegS)w(WReg)
    //
    sw.Dup(sw.z0.s, sw.w0);
    //
    sw.Ptrue(sw.p0.s);
    sw.St1w(sw.z0.s, sw.p0, AdrNoOfs(sw.x1));
    sw.Ret();
    sw.Ready();
    auto swFunc = sw.GetCode<void (*)(uint32_t, uint32_t*)>();
    uint32_t swDst[16] = {0};
    swFunc(1, swDst);
    ASSERT_EQ(1, swDst[0]);

    GTestJitGen bbe; // b(ZRegB)be(ZRegBElem)
    bbe.Dup(bbe.z0.b, 1);
    //
    bbe.Dup(bbe.z1.b, bbe.z0.b[0]);
    //
    bbe.Ptrue(bbe.p0.b);
    bbe.St1b(bbe.z1.b, bbe.p0, AdrNoOfs(bbe.x0));
    bbe.Ret();
    bbe.Ready();
    auto bbeFunc = bbe.GetCode<void (*)(uint8_t*)>();
    uint8_t bbeDst[64] = {0};
    bbeFunc(bbeDst);
    ASSERT_EQ(1, bbeDst[0]);

    GTestJitGen sse; // s(ZRegS)se(ZRegSElem)
    sse.Dup(sse.z0.s, sse.w0);
    //
    sse.Dup(sse.z1.s, sse.z0.s[0]);
    //
    sse.Ptrue(sse.p0.s);
    sse.St1w(sse.z1.s, sse.p0, AdrNoOfs(sse.x1));
    sse.Ret();
    sse.Ready();
    auto sseFunc = sse.GetCode<void (*)(uint32_t, uint32_t*)>();
    uint32_t sseDst[16] = {0};
    sseFunc(1, sseDst);
    ASSERT_EQ(1, sseDst[0]);

    GTestJitGen bi; // b(ZRegB)i(int32_t)
    //
    bi.Dup(bi.z0.b, 1);
    //
    bi.Ptrue(bi.p0.b);
    bi.St1b(bi.z0.b, bi.p0, AdrNoOfs(bi.x0));
    bi.Ret();
    bi.Ready();
    auto biFunc = bi.GetCode<void (*)(uint8_t*)>();
    uint8_t biDst[64] = {0};
    biFunc(biDst);
    ASSERT_EQ(1, biDst[0]);

    GTestJitGen si; // s(ZRegS)i(int32_t)
    //
    si.Dup(si.z0.s, 1);
    //
    si.Ptrue(si.p0.s);
    si.St1w(si.z0.s, si.p0, AdrNoOfs(si.x0));
    si.Ret();
    si.Ready();
    auto siFunc = si.GetCode<void (*)(uint32_t*)>();
    uint32_t siDst[16] = {0};
    siFunc(siDst);
    ASSERT_EQ(1, siDst[0]);

    GTestJitGen cg;
    cg.Mov(cg.w9, 2);
    cg.Mov(cg.v1.s4[0], cg.w9);
    cg.Dup(cg.v2.s4, cg.v1.s4[0]);
    cg.St1(VReg4SList(cg.v2.s4), AdrNoOfs(cg.x0));
    // cg.Fmov(cg.v3.s4, 2.0);
    cg.Fmov(cg.v4.s4, 4.0);
    cg.Mov(cg.v3.b8, cg.v4.b8);  // high bit set to zero
    cg.St1(VReg4SList(cg.v3.s4), AdrNoOfs(cg.x1));
    cg.Ret();
    cg.Ready();
    auto f = cg.GetCode<void (*)(uint32_t *, float *)>();
    uint32_t arr1[4] = {0};
    float arr2[4] = {1.0, 1.0, 1.0, 1.0};
    f(arr1, arr2);
    ASSERT_EQ(arr1[0], 2);
    ASSERT_EQ(arr1[1], 2);
    ASSERT_EQ(arr1[2], 2);
    ASSERT_EQ(arr1[3], 2);
    ASSERT_EQ(arr2[0], 4.0);
    ASSERT_EQ(arr2[1], 4.0);
    ASSERT_EQ(arr2[2], 0);
    ASSERT_EQ(arr2[3], 0);
}

TEST(armcg_test_t, TestDupm)
{
    GTestJitGen si; // s(ZRegS)i(uint64_t)
    //
    si.Dupm(si.z0.s, 1);
    //
    si.Ptrue(si.p0.s);
    si.St1w(si.z0.s, si.p0, AdrNoOfs(si.x0));
    si.Ret();
    si.Ready();
    auto siFunc = si.GetCode<void (*)(uint32_t*)>();
    uint32_t siDst[16] = {0};
    siFunc(siDst);
    ASSERT_EQ(1, siDst[0]);
}

TEST(armcg_test_t, TestFcpy)
{
    GTestJitGen sd; // s(ZRegS)d(double)
    sd.Ptrue(sd.p0.s);
    //
    sd.Fcpy(sd.z0.s, sd.p0 / PredType::T_M, 1.0f);
    //
    sd.St1w(sd.z0.s, sd.p0, AdrNoOfs(sd.x0));
    sd.Ret();
    sd.Ready();
    auto sdFunc = sd.GetCode<void (*)(float*)>();
    float sdDst[16] = {0.0f};
    sdFunc(sdDst);
    ASSERT_EQ(1.0f, sdDst[0]);
}

TEST(armcg_test_t, TestFdup)
{
    GTestJitGen sd; // s(ZRegS)d(double)
    //
    sd.Fdup(sd.z0.s, 1.0f);
    //
    sd.Ptrue(sd.p0.s);
    sd.St1w(sd.z0.s, sd.p0, AdrNoOfs(sd.x0));
    sd.Ret();
    sd.Ready();
    auto sdFunc = sd.GetCode<void (*)(float*)>();
    float sdDst[16] = {0.0f};
    sdFunc(sdDst);
    ASSERT_EQ(1.0f, sdDst[0]);
}

TEST(armcg_test_t, TestIndex)
{
    GTestJitGen hii; // h(ZRegH)i(int32_t)i(int32_t)
    //
    hii.Index(hii.z0.h, 1, 1);
    //
    hii.Ptrue(hii.p0.h);
    hii.St1h(hii.z0.h, hii.p0, AdrNoOfs(hii.x0));
    hii.Ret();
    hii.Ready();
    auto hiiFunc = hii.GetCode<void (*)(uint16_t*)>();
    uint16_t hiiDst[32] = {0};
    hiiFunc(hiiDst);
    ASSERT_EQ(1, hiiDst[0]);

    GTestJitGen sii; // s(ZRegS)i(int32_t)i(int32_t)
    //
    sii.Index(sii.z0.s, 1, 1);
    //
    sii.Ptrue(sii.p0.s);
    sii.St1w(sii.z0.s, sii.p0, AdrNoOfs(sii.x0));
    sii.Ret();
    sii.Ready();
    auto siiFunc = sii.GetCode<void (*)(uint32_t*)>();
    uint32_t siiDst[16] = {0};
    siiFunc(siiDst);
    ASSERT_EQ(1, siiDst[0]);
}

//mne pred test
TEST(armcg_test_t, TestPx)
{
    GTestJitGen b; // b(PRegB)
    //
    b.Pfalse(b.p0.b);
    //
    b.Ld1b(b.z0.b, b.p0 / PredType::T_Z, AdrNoOfs(b.x0));
    b.Ptrue(b.p1.b);
    b.St1b(b.z0.b, b.p1, AdrNoOfs(b.x1));
    b.Ret();
    b.Ready();
    auto bFunc = b.GetCode<void (*)(uint8_t*, uint8_t*)>();
    uint8_t bSrc[64] = {1, 2};
    uint8_t bDst[64] = {0};
    bFunc(bSrc, bDst);
    ASSERT_EQ(0, bDst[0]);
}

TEST(armcg_test_t, TestWhilelt)
{
    GTestJitGen bxx; // b(PRegB)x(XReg)x(XReg)
    bxx.Mov(bxx.x9, 0);
    bxx.Mov(bxx.x10, 1);
    //
    bxx.Whilelt(bxx.p0.b, bxx.x9, bxx.x10);
    //
    bxx.Ptrue(bxx.p1.b);
    bxx.Ld1b(bxx.z0.b, bxx.p1 / PredType::T_Z, AdrNoOfs(bxx.x0));
    bxx.St1b(bxx.z0.b, bxx.p0, AdrNoOfs(bxx.x1));
    bxx.Ret();
    bxx.Ready();
    auto bxxFunc = bxx.GetCode<void (*)(uint8_t*, uint8_t*)>();
    uint8_t bxxSrc[64] = {1, 2};
    uint8_t bxxDst[64] = {0};
    bxxFunc(bxxSrc, bxxDst);
    ASSERT_EQ(bxxSrc[0], bxxDst[0]);

    GTestJitGen sxx; // s(PRegS)x(XReg)x(XReg)
    sxx.Mov(sxx.x9, 0);
    sxx.Mov(sxx.x10, 1);
    //
    sxx.Whilelt(sxx.p0.s, sxx.x9, sxx.x10);
    //
    sxx.Ptrue(sxx.p1.s);
    sxx.Ld1w(sxx.z0.s, sxx.p1 / PredType::T_Z, AdrNoOfs(sxx.x0));
    sxx.St1w(sxx.z0.s, sxx.p0, AdrNoOfs(sxx.x1));
    sxx.Ret();
    sxx.Ready();
    auto sxxFunc = sxx.GetCode<void (*)(uint32_t*, uint32_t*)>();
    uint32_t sxxSrc[16] = {1, 2};
    uint32_t sxxDst[16] = {0};
    sxxFunc(sxxSrc, sxxDst);
    ASSERT_EQ(sxxSrc[0], sxxDst[0]);
}

TEST(armcg_test_t, TestNot)
{
    GTestJitGen bb; // b(PRegB)b(PRegB)
    bb.Ptrue(bb.p1.b);
    bb.Pfalse(bb.p2.b);
    //
    bb.Not(bb.p0.b, bb.p1, bb.p2.b);
    //
    bb.Ld1b(bb.z0.b, bb.p0 / PredType::T_Z, AdrNoOfs(bb.x0));
    bb.St1b(bb.z0.b, bb.p0, AdrNoOfs(bb.x1));
    bb.Ret();
    bb.Ready();
    auto bbFunc = bb.GetCode<void (*)(uint8_t*, uint8_t*)>();
    uint8_t bbSrc[64] = {1};
    uint8_t bbDst[64] = {0};
    bbFunc(bbSrc, bbDst);
    ASSERT_EQ(bbSrc[0], bbDst[0]);
}

TEST(armcg_test_t, TestCntw)
{
    GTestJitGen x; // w(Cntw)x(XReg)
    //
    x.Cntw(x.x0);
    //
    x.Ret();
    x.Ready();
    auto xFunc = x.GetCode<uint64_t (*)()>();
    ASSERT_EQ(xFunc() % 8, 0); // sve256 is 256 / 32 = 8, sve512 is 16
}

TEST(armcg_test_t, TestIncb)
{
    GTestJitGen gen;
    gen.Mov(gen.x0, 0);
    gen.Incb(gen.x0);
    gen.Ret();
    gen.Ready();
    auto f = gen.GetCode<uint64_t (*)()>();
#ifdef KUDNNL_920F
    ASSERT_EQ(f(), 64);
#else
    ASSERT_EQ(f(), 32);
#endif
}

TEST(armcg_test_t, TestInch)
{
    GTestJitGen gen;
    gen.Mov(gen.x0, 0);
    gen.Inch(gen.x0, Pattern::ALL, ExtMod::MUL, 4);
    gen.Ret();
    gen.Ready();
    auto f = gen.GetCode<uint64_t (*)()>();
#ifdef KUDNNL_920F
    ASSERT_EQ(f(), 128);
#else
    ASSERT_EQ(f(), 64);
#endif
}

TEST(armcg_test_t, TestIncw)
{
    GTestJitGen x; // w(Incw)x(XReg)
    //
    x.Incw(x.x0);
    //
    x.Ret();
    x.Ready();
    auto xFunc = x.GetCode<uint64_t (*)(uint64_t)>();
    ASSERT_EQ(xFunc(0) % 8, 0); // sve256 will +8, sve512 is +16
}

TEST(armcg_test_t, TestDecw)
{
    GTestJitGen x; // w(Decw)x(XReg)
    //
    x.Decw(x.x0);
    //
    x.Ret();
    x.Ready();
    auto xFunc = x.GetCode<uint64_t (*)(uint64_t)>();
    ASSERT_EQ(xFunc(16) % 8, 0); // sve256 will -8, sve512 is -16
}

// mne store test
TEST(armcg_test_t, TestSt1)
{
    GTestJitGen heor; // he(VRegHElem)or(AdrPostReg)
    heor.Ld1(heor.v0.h[0], AdrNoOfs(heor.x0));
    //
    heor.St1(heor.v0.h[0], AdrPostReg(heor.x1, heor.x2));
    //
    heor.Ret();
    heor.Ready();
    auto heorFunc = heor.GetCode<void (*)(uint16_t*, uint16_t*, uint16_t)>();
    uint16_t heorSrc[1] = {1};
    uint16_t heorDst[1] = {0};
    heorFunc(heorSrc, heorDst, 2);
    ASSERT_EQ(heorSrc[0], heorDst[0]);

    GTestJitGen seor; // se(VRegSElem)or(AdrPostReg)
    seor.Ld1(seor.v0.s[0], AdrNoOfs(seor.x0));
    //
    seor.St1(seor.v0.s[0], AdrPostReg(seor.x1, seor.x2));
    //
    seor.Ret();
    seor.Ready();
    auto seorFunc = seor.GetCode<void (*)(uint32_t*, uint32_t*, uint32_t)>();
    uint32_t seorSrc[1] = {1};
    uint32_t seorDst[1] = {0};
    seorFunc(seorSrc, seorDst, 4);
    ASSERT_EQ(seorSrc[0], seorDst[0]);
}

TEST(armcg_test_t, TestStlxr)
{
    GTestJitGen wwi; // w(WReg)w(WReg)i(AdrImm)
    ARMCG::Label wwiloop;
    wwi.L(wwiloop);
    wwi.Ldaxr(wwi.x9, AdrImm(wwi.x0, 0));
    wwi.Cbnz(wwi.x9, wwiloop);
    //
    wwi.Stlxr(wwi.w9, wwi.w1, AdrImm(wwi.x0, 0));
    //
    wwi.Cbnz(wwi.x9, wwiloop);
    // enter critical section
    wwi.Ret();
    wwi.Ready();
    auto wwiFunc = wwi.GetCode<void (*)(uint64_t*, uint32_t)>();
    uint64_t wwiSrc[1] = {0};
    wwiFunc(wwiSrc, 1);
    ASSERT_EQ(wwiSrc[0], 1);

    GTestJitGen wxi; // w(WReg)x(XReg)i(AdrImm)
    ARMCG::Label wxiloop;
    wxi.L(wxiloop);
    wxi.Ldaxr(wxi.x9, AdrImm(wxi.x0, 0));
    wxi.Cbnz(wxi.x9, wxiloop);
    //
    wxi.Stlxr(wxi.w9, wxi.x1, AdrImm(wxi.x0, 0));
    //
    wxi.Cbnz(wxi.x9, wxiloop);
    // enter critical section
    wxi.Ret();
    wxi.Ready();
    auto wxiFunc = wxi.GetCode<void (*)(uint64_t*, uint64_t)>();
    uint64_t wxiSrc[1] = {0};
    wxiFunc(wxiSrc, 1);
    ASSERT_EQ(wxiSrc[0], 1);
}

TEST(armcg_test_t, TestStnt1d)
{
    GTestJitGen dn; // d(ZRegD)n(AdrNoOfs)
    dn.Ptrue(dn.p0.d);
    dn.Ld1d(dn.z0.d, dn.p0 / PredType::T_Z, AdrNoOfs(dn.x0));
    //
    dn.Stnt1d(dn.z0.d, dn.p0, AdrNoOfs(dn.x1));
    //
    dn.Ret();
    dn.Ready();
    auto dnFunc = dn.GetCode<void (*)(uint64_t*, uint64_t*)>();
    uint64_t dnSrc[8] = {1};
    uint64_t dnDst[8] = {0};
    dnFunc(dnSrc, dnDst);
    ASSERT_EQ(dnSrc[0], dnDst[0]);
}

TEST(armcg_test_t, TestStnt1w)
{
    GTestJitGen sn; // s(ZRegS)n(AdrNoOfs)
    sn.Ptrue(sn.p0.s);
    sn.Ld1w(sn.z0.s, sn.p0 / PredType::T_Z, AdrNoOfs(sn.x0));
    //
    sn.Stnt1w(sn.z0.s, sn.p0, AdrNoOfs(sn.x1));
    //
    sn.Ret();
    sn.Ready();
    auto snFunc = sn.GetCode<void (*)(uint32_t*, uint32_t*)>();
    uint32_t snSrc[16] = {1};
    uint32_t snDst[16] = {0};
    snFunc(snSrc, snDst);
    ASSERT_EQ(snSrc[0], snDst[0]);
}

TEST(armcg_test_t, TestStp)
{
    GTestJitGen xxi; // x(XReg)x(XReg)i(AdrImm)
    xxi.Ldp(xxi.x9, xxi.x10, AdrPostImm(xxi.x0, 16));
    //
    xxi.Stp(xxi.x9, xxi.x10, AdrImm(xxi.x0, 0));
    //
    xxi.Ret();
    xxi.Ready();
    auto xxiFunc = xxi.GetCode<void (*)(uint64_t*)>();
    uint64_t xxiSrc[4] = {1, 2, 0, 0};
    xxiFunc(xxiSrc);
    ASSERT_EQ(xxiSrc[0], xxiSrc[2]);

    GTestJitGen xxri; // x(XReg)x(XReg)ri(AdrPreImm)
    //
    xxri.Ldp(xxri.x9, xxri.x10, AdrPostImm(xxri.x0, 16));
    xxri.Stp(xxri.x9, xxri.x10, AdrPreImm(xxri.x0, 0));
    //
    xxri.Ret();
    xxri.Ready();
    auto xxriFunc = xxri.GetCode<void (*)(uint64_t*)>();
    uint64_t xxriSrc[4] = {1, 2, 0, 0};
    xxriFunc(xxriSrc);
    ASSERT_EQ(xxriSrc[0], xxriSrc[2]);
}

TEST(armcg_test_t, TestStr)
{
    GTestJitGen xi; // x(XReg)i(AdrImm)
    //
    xi.Str(xi.x0, AdrImm(xi.x1, 0));
    //
    xi.Ret();
    xi.Ready();
    auto xiFunc = xi.GetCode<void (*)(uint64_t, uint64_t*)>();
    uint64_t xiDst = 0;
    xiFunc(1, &xiDst);
    ASSERT_EQ(xiDst, 1);

    GTestJitGen xn; // x(XReg)n(AdrNoOfs)
    //
    xn.Str(xn.x0, AdrNoOfs(xn.x1));
    //
    xn.Ret();
    xn.Ready();
    auto xnFunc = xn.GetCode<void (*)(uint64_t, uint64_t*)>();
    uint64_t xnDst = 0;
    xnFunc(1, &xnDst);
    ASSERT_EQ(xnDst, 1);

    GTestJitGen xri; // x(XReg)ri(AdrPreImm)
    //
    xri.Str(xri.x0, AdrPreImm(xri.x1, 8));
    //
    xri.Ret();
    xri.Ready();
    auto xriFunc = xri.GetCode<void (*)(uint64_t, uint64_t*)>();
    uint64_t xriSrc = 1;
    uint64_t xriDst[2];
    xriFunc(xriSrc, xriDst);
    ASSERT_EQ(xriSrc, xriDst[1]);

    GTestJitGen wi; // w(WReg)i(AdrImm)
    //
    wi.Str(wi.w0, AdrImm(wi.x1, 0));
    //
    wi.Ret();
    wi.Ready();
    auto wiFunc = wi.GetCode<void (*)(uint32_t, uint32_t*)>();
    uint32_t wiDst = 0;
    wiFunc(1, &wiDst);
    ASSERT_EQ(wiDst, 1);

    GTestJitGen wn; // w(WReg)n(AdrNoOfs)
    //
    wn.Str(wn.w0, AdrNoOfs(wn.x1));
    //
    wn.Ret();
    wn.Ready();
    auto wnFunc = wn.GetCode<void (*)(uint32_t, uint32_t*)>();
    uint32_t wnDst = 0;
    wnFunc(1, &wnDst);
    ASSERT_EQ(wnDst, 1);
}
// bit
TEST(armcg_test_t, TestAnd1)
{
    GTestJitGen cg;
    cg.And(cg.x0, cg.x0, 1);
    cg.Ret();
    cg.Ready();
    
    auto f = cg.GetCode<int (*)(int)>();
    ASSERT_EQ(f(9999), 1);
}

TEST(armcg_test_t, TestAnd2)
{
    GTestJitGen cg;
    cg.Ptrue(cg.p0.b);
    cg.Dup(cg.z0.b, 1);
    cg.Dup(cg.z1.b, 3);
    cg.And(cg.z0.b, cg.p0, cg.z1.b);
    cg.Mov(cg.x9, 0);
    cg.Mov(cg.x10, 8);
    cg.Whilelt(cg.p1.s, cg.x9, cg.x10);
    cg.St1w(cg.z0.s, cg.p1, AdrNoOfs(cg.x0));
    cg.Ret();
    cg.Ready();

    auto f = cg.GetCode<void (*)(int*)>();
    int arrOut[8];
    f(arrOut);
    ASSERT_EQ(arrOut[0], 0x01010101);
}

TEST(armcg_test_t, TestAnd3)
{
    GTestJitGen cg;
    cg.Mov(cg.x9, 0);
    cg.Mov(cg.x10, 8);
    cg.Whilelt(cg.p0.s, cg.x9, cg.x10);
    cg.Dup(cg.z0.s, 1);
    cg.Dup(cg.z1.s, 3);
    cg.And(cg.z0.s, cg.p0, cg.z1.s);
    cg.St1w(cg.z0.s, cg.p0, AdrNoOfs(cg.x0));
    cg.Ret();
    cg.Ready();

    auto f = cg.GetCode<void (*)(int*)>();
    int arrOut[8];
    f(arrOut);
    ASSERT_EQ(arrOut[0], 0x1);
}

TEST(armcg_test_t, TestAnd4)
{
    GTestJitGen cg;
    cg.Dup(cg.z0.s, 1);
    cg.Dup(cg.z1.s, 3);
    cg.And(cg.z0.d, cg.z0.d, cg.z1.d);
    cg.Mov(cg.x9, 0);
    cg.Mov(cg.x10, 8);
    cg.Whilelt(cg.p0.s, cg.x9, cg.x10);
    cg.St1w(cg.z0.s, cg.p0, AdrNoOfs(cg.x0));
    cg.Ret();
    cg.Ready();

    auto f = cg.GetCode<void (*)(int*)>();
    int arrOut[8];
    f(arrOut);
    ASSERT_EQ(arrOut[0], 0x1);
}

TEST(armcg_test_t, TestAnd5)
{
    GTestJitGen cg;
    cg.Dup(cg.z0.s, 1);
    cg.And(cg.z0.s, 3);
    cg.Mov(cg.x9, 0);
    cg.Mov(cg.x10, 8);
    cg.Whilelt(cg.p0.s, cg.x9, cg.x10);
    cg.St1w(cg.z0.s, cg.p0, AdrNoOfs(cg.x0));
    cg.Ret();
    cg.Ready();

    auto f = cg.GetCode<void (*)(int*)>();
    int arrOut[8];
    f(arrOut);
    ASSERT_EQ(arrOut[0], 0x1);
}

TEST(armcg_test_t, TestAnd6)
{
    GTestJitGen cg;
    cg.Ptrue(cg.p0.b);
    cg.Pfalse(cg.p1.b);
    cg.And(cg.p3.b, cg.p0, cg.p0.b, cg.p1.b);
    cg.Str(cg.p3, AdrNoOfs(cg.x0));
    cg.Ret();
    cg.Ready();

    auto f = cg.GetCode<void (*)(int*)>();
    int predOut;;
    f(&predOut);
    ASSERT_EQ(predOut, 0x0);
}


TEST(armcg_test_t, TestAsr1)
{
    GTestJitGen cg;
    cg.Mov(cg.x9, 0);
    cg.Whilelt(cg.p0.s, cg.x9, cg.x1);
    cg.Ld1w(cg.z0.s, cg.p0, AdrNoOfs(cg.x0));
    cg.Asr(cg.z0.s, cg.p0, 4);
    cg.St1w(cg.z0.s, cg.p0, AdrNoOfs(cg.x0));
    cg.Ret();
    cg.Ready();

    int arr1[] = {0, 9, 25, -46, 40, 20, 8059, 524};
    auto f = cg.GetCode<void (*)(int*, int)>();
    f(arr1,8);
    ASSERT_EQ(arr1[0], 0x0);
    ASSERT_EQ(arr1[1], 0x0);
    ASSERT_EQ(arr1[2], 0x1);
    ASSERT_EQ(arr1[3], 0xfffffffd);
    ASSERT_EQ(arr1[4], 0x2);
    ASSERT_EQ(arr1[5], 0x1);
    ASSERT_EQ(arr1[6], 0x1f7);
    ASSERT_EQ(arr1[7], 0x20);
}

TEST(armcg_test_t, TestAsr2)
{
    GTestJitGen cg;
    cg.Mov(cg.x9, 0);
    cg.Whilelt(cg.p0.s, cg.x9, cg.x1);
    cg.Ld1w(cg.z0.s, cg.p0, AdrNoOfs(cg.x0));
    cg.Asr(cg.z0.s, cg.z0.s, 4);
    cg.St1w(cg.z0.s, cg.p0, AdrNoOfs(cg.x0));
    cg.Ret();
    cg.Ready();

    int arr1[] = {0, 9, 25, -46, 40, 20, 8059, 524};
    auto f = cg.GetCode<void (*)(int*, int)>();
    f(arr1,8);
    ASSERT_EQ(arr1[0], 0x0);
    ASSERT_EQ(arr1[1], 0x0);
    ASSERT_EQ(arr1[2], 0x1);
    ASSERT_EQ(arr1[3], 0xfffffffd);
    ASSERT_EQ(arr1[4], 0x2);
    ASSERT_EQ(arr1[5], 0x1);
    ASSERT_EQ(arr1[6], 0x1f7);
    ASSERT_EQ(arr1[7], 0x20);
}

TEST(armcg_test_t, TestBic)
{
    GTestJitGen cg;
    cg.Dup(cg.v1.b16, cg.w0);
    cg.Dup(cg.v2.b16, cg.w1);
    cg.Bic(cg.v1.b16, cg.v1.b16, cg.v2.b16);
    cg.Mov(cg.w0, cg.v1.s4[0]);
    cg.Ret();
    cg.Ready();

    auto f = cg.GetCode<int (*)(int, int)>();
    ASSERT_EQ(f(0x10, 0xaa), 0x10101010);
}

TEST(armcg_test_t, TestEor1)
{
    GTestJitGen cg;
    cg.Eor(cg.x0, cg.x0, cg.x1);
    cg.Ret();
    cg.Ready();

    auto f = cg.GetCode<int (*)(int, int)>();
    ASSERT_EQ(f(0x10, 0x1), 0x11);
}

TEST(armcg_test_t, TestEor2)
{
    GTestJitGen cg;
    cg.Dup(cg.v1.b16, cg.w0);
    cg.Dup(cg.v2.b16, cg.w1);
    cg.Eor(cg.v1.b16, cg.v1.b16, cg.v2.b16);
    cg.Mov(cg.w0, cg.v1.s4[0]);
    cg.Ret();
    cg.Ready();

    auto f = cg.GetCode<int (*)(int, int)>();
    ASSERT_EQ(f(0x10, 0xaa), 0xbabababa);
}


TEST(armcg_test_t, TestEor3)
{
    GTestJitGen cg;
    cg.Mov(cg.x9, 0);
    cg.Whilelt(cg.p0.s, cg.x9, cg.x2);
    cg.Ld1w(cg.z0.s, cg.p0, AdrNoOfs(cg.x0));
    cg.Ld1w(cg.z1.s, cg.p0, AdrNoOfs(cg.x1));
    cg.Eor(cg.z0.s, cg.p0, cg.z1.s);
    cg.St1w(cg.z0.s, cg.p0, AdrNoOfs(cg.x0));
    cg.Ret();
    cg.Ready();

    int arr1[] = {0, 9, 25, -46, 40, 20, 8059, 524};
    int arr2[] = {0, 100, -10, 816, -842, 88, 22, 44};
    auto f = cg.GetCode<void (*)(int*, int*, int)>();
    f(arr1, arr2, 8);
    ASSERT_EQ(arr1[0], 0x0);
    ASSERT_EQ(arr1[1], 0x6d);
    ASSERT_EQ(arr1[2], 0xffffffef);
    ASSERT_EQ(arr1[3], 0xfffffce2);
    ASSERT_EQ(arr1[4], 0xfffffc9e);
    ASSERT_EQ(arr1[5], 0x4c);
    ASSERT_EQ(arr1[6], 0x1f6d);
    ASSERT_EQ(arr1[7], 0x220);
}

TEST(armcg_test_t, TestEor4)
{
    GTestJitGen cg;
    cg.Mov(cg.x9, 0);
    cg.Whilelt(cg.p0.s, cg.x9, cg.x2);
    cg.Ld1w(cg.z0.s, cg.p0, AdrNoOfs(cg.x0));
    cg.Ld1w(cg.z1.s, cg.p0, AdrNoOfs(cg.x1));
    cg.Eor(cg.z0.d, cg.z0.d, cg.z1.d);
    cg.St1w(cg.z0.s, cg.p0, AdrNoOfs(cg.x0));
    cg.Ret();
    cg.Ready();

    int arr1[] = {0, 9, 25, -46, 40, 20, 8059, 524};
    int arr2[] = {0, 100, -10, 816, -842, 88, 22, 44};
    auto f = cg.GetCode<void (*)(int*, int*, int)>();
    f(arr1, arr2, 8);
    ASSERT_EQ(arr1[0], 0x0);
    ASSERT_EQ(arr1[1], 0x6d);
    ASSERT_EQ(arr1[2], 0xffffffef);
    ASSERT_EQ(arr1[3], 0xfffffce2);
    ASSERT_EQ(arr1[4], 0xfffffc9e);
    ASSERT_EQ(arr1[5], 0x4c);
    ASSERT_EQ(arr1[6], 0x1f6d);
    ASSERT_EQ(arr1[7], 0x220);
}

TEST(armcg_test_t, TestEor5)
{
    GTestJitGen cg;
    cg.Mov(cg.x9, 0);
    cg.Whilelt(cg.p0.s, cg.x9, cg.x1);
    cg.Ld1w(cg.z0.s, cg.p0, AdrNoOfs(cg.x0));
    cg.Eor(cg.z0.s, 0x1);
    cg.St1w(cg.z0.s, cg.p0, AdrNoOfs(cg.x0));
    cg.Ret();
    cg.Ready();

    int arr1[] = {0, 9, 25, -46, 40, 20, 8059, 524};
    auto f = cg.GetCode<void (*)(int*, int)>();
    f(arr1, 8);
    ASSERT_EQ(arr1[0], 0x1);
    ASSERT_EQ(arr1[1], 0x8);
    ASSERT_EQ(arr1[2], 0x18);
    ASSERT_EQ(arr1[3], 0xffffffd3);
    ASSERT_EQ(arr1[4], 0x29);
    ASSERT_EQ(arr1[5], 0x15);
    ASSERT_EQ(arr1[6], 0x1f7a);
    ASSERT_EQ(arr1[7], 0x20d);
}

TEST(armcg_test_t, TestEor6)
{
    GTestJitGen cg;
    cg.Ptrue(cg.p0.b);
    cg.Pfalse(cg.p1.b);
    cg.Eor(cg.p3.b, cg.p0, cg.p0.b, cg.p1.b);
    cg.Str(cg.p3, AdrNoOfs(cg.x0));
    cg.Ret();
    cg.Ready();

    auto f = cg.GetCode<void (*)(int*)>();
    int predOut;;
    f(&predOut);
    ASSERT_EQ(predOut, 0xffffffff);
}

TEST(armcg_test_t, TestLsl1)
{
    GTestJitGen cg;
    cg.Lsl(cg.x0, cg.x0, 4);
    cg.Ret();
    cg.Ready();

    auto f = cg.GetCode<int (*)(int)>();
    ASSERT_EQ(f(0x10), 0x100);
}


TEST(armcg_test_t, TestLsl2)
{
    GTestJitGen cg;
    cg.Mov(cg.x9, 0);
    cg.Whilelt(cg.p0.s, cg.x9, cg.x1);
    cg.Ld1w(cg.z0.s, cg.p0, AdrNoOfs(cg.x0));
    cg.Lsl(cg.z0.s, cg.p0.s, 2);
    cg.St1w(cg.z0.s, cg.p0, AdrNoOfs(cg.x0));
    cg.Ret();
    cg.Ready();

    int arr1[] = {0, 9, 25, -46, 40, 20, 8059, 524};
    auto f = cg.GetCode<void (*)(int*, int)>();
    f(arr1, 8);
    ASSERT_EQ(arr1[0], 0x0);
    ASSERT_EQ(arr1[1], 0x24);
    ASSERT_EQ(arr1[2], 0x64);
    ASSERT_EQ(arr1[3], 0xffffff48);
    ASSERT_EQ(arr1[4], 0xa0);
    ASSERT_EQ(arr1[5], 0x50);
    ASSERT_EQ(arr1[6], 0x7dec);
    ASSERT_EQ(arr1[7], 0x830);
}

TEST(armcg_test_t, TestLsl3)
{
    GTestJitGen cg;
    cg.Mov(cg.x9, 0);
    cg.Whilelt(cg.p0.s, cg.x9, cg.x1);
    cg.Ld1w(cg.z0.s, cg.p0, AdrNoOfs(cg.x0));
    cg.Dup(cg.z1.s, 2);
    cg.Lsl(cg.z0.s, cg.p0.s, cg.z1.s);
    cg.St1w(cg.z0.s, cg.p0, AdrNoOfs(cg.x0));
    cg.Ret();
    cg.Ready();

    int arr1[] = {0, 9, 25, -46, 40, 20, 8059, 524};
    auto f = cg.GetCode<void (*)(int*, int)>();
    f(arr1, 8);
    ASSERT_EQ(arr1[0], 0x0);
    ASSERT_EQ(arr1[1], 0x24);
    ASSERT_EQ(arr1[2], 0x64);
    ASSERT_EQ(arr1[3], 0xffffff48);
    ASSERT_EQ(arr1[4], 0xa0);
    ASSERT_EQ(arr1[5], 0x50);
    ASSERT_EQ(arr1[6], 0x7dec);
    ASSERT_EQ(arr1[7], 0x830);
}

TEST(armcg_test_t, TestLsl4)
{
    GTestJitGen cg;
    cg.Mov(cg.x9, 0);
    cg.Whilelt(cg.p0.s, cg.x9, cg.x1);
    cg.Ld1w(cg.z0.s, cg.p0, AdrNoOfs(cg.x0));
    cg.Dup(cg.z1.s, 2);
    cg.Lsl(cg.z0.h, cg.p0.s, cg.z1.h);
    cg.St1w(cg.z0.s, cg.p0, AdrNoOfs(cg.x0));
    cg.Ret();
    cg.Ready();

    uint16_t arr1[] = {0x0, 0x0, 0xc600, 0x0, 0x2d93, 0x0, 0xa9ef, 0x0,
        0x5100, 0x0, 0xcd00, 0x0, 0x2d28, 0x0, 0xb831, 0x0};
    auto f = cg.GetCode<void (*)(uint16_t*, int)>();
    f(arr1, 8);
    ASSERT_EQ(arr1[0], 0x0);
    ASSERT_EQ(arr1[2], 0x1800);
    ASSERT_EQ(arr1[4], 0xb64c);
    ASSERT_EQ(arr1[6], 0xa7bc);
    ASSERT_EQ(arr1[8], 0x4400);
    ASSERT_EQ(arr1[10], 0x3400);
    ASSERT_EQ(arr1[12], 0xb4a0);
    ASSERT_EQ(arr1[14], 0xe0c4);
}

TEST(armcg_test_t, TestLsl5)
{
    GTestJitGen cg;
    cg.Mov(cg.x9, 0);
    cg.Whilelt(cg.p0.s, cg.x9, cg.x1);
    cg.Ld1w(cg.z0.s, cg.p0, AdrNoOfs(cg.x0));
    cg.Lsl(cg.z0.s, cg.z0.s, 2);
    cg.St1w(cg.z0.s, cg.p0, AdrNoOfs(cg.x0));
    cg.Ret();
    cg.Ready();

    int arr1[] = {0, 9, 25, -46, 40, 20, 8059, 524};
    auto f = cg.GetCode<void (*)(int*, int)>();
    f(arr1, 8);
    ASSERT_EQ(arr1[0], 0x0);
    ASSERT_EQ(arr1[1], 0x24);
    ASSERT_EQ(arr1[2], 0x64);
    ASSERT_EQ(arr1[3], 0xffffff48);
    ASSERT_EQ(arr1[4], 0xa0);
    ASSERT_EQ(arr1[5], 0x50);
    ASSERT_EQ(arr1[6], 0x7dec);
    ASSERT_EQ(arr1[7], 0x830);
}

TEST(armcg_test_t, TestLsr1)
{
    GTestJitGen cg;
    cg.Lsr(cg.x0, cg.x0, 4);
    cg.Ret();
    cg.Ready();

    auto f = cg.GetCode<int (*)(int)>();
    ASSERT_EQ(f(0x100), 0x10);
}

TEST(armcg_test_t, TestLsr2)
{
    GTestJitGen cg;
    cg.Mov(cg.x9, 0);
    cg.Whilelt(cg.p0.s, cg.x9, cg.x1);
    cg.Ld1w(cg.z0.s, cg.p0, AdrNoOfs(cg.x0));
    cg.Lsr(cg.z0.s, cg.z0.s, 2);
    cg.St1w(cg.z0.s, cg.p0, AdrNoOfs(cg.x0));
    cg.Ret();
    cg.Ready();

    int arr1[] = {0, 9, 25, -46, 40, 20, 8059, 524};
    auto f = cg.GetCode<void (*)(int*, int)>();
    f(arr1, 8);
    ASSERT_EQ(arr1[0], 0x0);
    ASSERT_EQ(arr1[1], 0x2);
    ASSERT_EQ(arr1[2], 0x6);
    ASSERT_EQ(arr1[3], 0x3ffffff4);
    ASSERT_EQ(arr1[4], 0xa);
    ASSERT_EQ(arr1[5], 0x5);
    ASSERT_EQ(arr1[6], 0x7de);
    ASSERT_EQ(arr1[7], 0x83);
}

TEST(armcg_test_t, TestMvn)
{
    GTestJitGen cg;
    cg.Mvn(cg.x0, cg.x0);
    cg.Ret();
    cg.Ready();

    auto f = cg.GetCode<int (*)(int)>();
    ASSERT_EQ(f(0x10), 0xffffffef);
}

TEST(armcg_test_t, TestOrr1)
{
    GTestJitGen cg;
    cg.Dup(cg.v1.b16, cg.w0);
    cg.Dup(cg.v2.b16, cg.w1);
    cg.Orr(cg.v1.b16, cg.v1.b16, cg.v2.b16);
    cg.Mov(cg.w0, cg.v1.s4[0]);
    cg.Ret();
    cg.Ready();

    auto f = cg.GetCode<int (*)(int, int)>();
    ASSERT_EQ(f(0x10, 0xaa), 0xbabababa);
}

TEST(armcg_test_t, TestOrr2)
{
    GTestJitGen cg;
    cg.Mov(cg.x9, 0);
    cg.Mov(cg.x10, 8);
    cg.Whilelt(cg.p0.s, cg.x9, cg.x10);
    cg.Dup(cg.z0.s, 1);
    cg.Dup(cg.z1.s, 3);
    cg.Orr(cg.z0.s, cg.p0, cg.z1.s);
    cg.St1w(cg.z0.s, cg.p0, AdrNoOfs(cg.x0));
    cg.Ret();
    cg.Ready();

    auto f = cg.GetCode<void (*)(int*)>();
    int arrOut[8];
    f(arrOut);
    ASSERT_EQ(arrOut[0], 0x3);
}

TEST(armcg_test_t, TestOrr3)
{
    GTestJitGen cg;
    cg.Dup(cg.z0.s, 1);
    cg.Dup(cg.z1.s, 3);
    cg.Orr(cg.z0.d, cg.z0.d, cg.z1.d);
    cg.Mov(cg.x9, 0);
    cg.Mov(cg.x10, 8);
    cg.Whilelt(cg.p0.s, cg.x9, cg.x10);
    cg.St1w(cg.z0.s, cg.p0, AdrNoOfs(cg.x0));
    cg.Ret();
    cg.Ready();

    auto f = cg.GetCode<void (*)(int*)>();
    int arrOut[8];
    f(arrOut);
    ASSERT_EQ(arrOut[0], 0x3);
}

TEST(armcg_test_t, TestSxtb1)
{
    GTestJitGen cg;
    cg.Sxtb(cg.w0, cg.w0);
    cg.Ret();
    cg.Ready();

    auto f = cg.GetCode<int (*)(int)>();
    ASSERT_EQ(f(0xff), 0xffffffff);
}

TEST(armcg_test_t, TestSxtb2)
{
    GTestJitGen cg;
    cg.Mov(cg.x9, 0);
    cg.Mov(cg.x10, 8);
    cg.Whilelt(cg.p0.s, cg.x9, cg.x10);
    cg.Dup(cg.z0.s, 0xf);
    cg.Sxtb(cg.z0.s, cg.p0, cg.z0.s);
    cg.St1w(cg.z0.s, cg.p0, AdrNoOfs(cg.x0));
    cg.Ret();
    cg.Ready();

    auto f = cg.GetCode<void (*)(int*)>();
    int arrOut[8];
    f(arrOut);
    ASSERT_EQ(arrOut[0], 0xf);
}

TEST(armcg_test_t, TestSxtl2)
{
    GTestJitGen cg;
    cg.Dup(cg.v1.b16, cg.w0);
    cg.Sxtl(cg.v1.s4, cg.v1.h4);
    cg.Mov(cg.w0, cg.v1.s4[0]);
    cg.Ret();
    cg.Ready();
    
    auto f = cg.GetCode<int (*)(int)>();
    ASSERT_EQ(f(0x10), 0x1010);
}

TEST(armcg_test_t, TestTst1)
{
    GTestJitGen cg;
    cg.Tst(cg.x0, 1);
    cg.MrsNZCV(cg.x0);
    cg.Ret();
    cg.Ready();
    auto f = cg.GetCode<int (*)(int)>();
    ASSERT_EQ(f(0xff), 0x0);
}

TEST(armcg_test_t, TestTst2)
{
    GTestJitGen cg;
    cg.Tst(cg.x0, cg.x1);
    cg.MrsNZCV(cg.x0);
    cg.Ret();
    cg.Ready();
    
    auto f = cg.GetCode<int (*)(int, int)>();
    ASSERT_EQ(f(0xff, 0x0), 0x40000000);
}

TEST(armcg_test_t, TestUxtb1)
{
    GTestJitGen cg;
    cg.Uxtb(cg.w0, cg.w0);
    cg.Ret();
    cg.Ready();

    auto f = cg.GetCode<int (*)(int)>();
    ASSERT_EQ(f(0xff), 0xff);
}

TEST(armcg_test_t, TestUxtb2)
{
    GTestJitGen cg;
    cg.Mov(cg.x9, 0);
    cg.Mov(cg.x10, 8);
    cg.Whilelt(cg.p0.s, cg.x9, cg.x10);
    cg.Dup(cg.z0.s, 0xf);
    cg.Uxtb(cg.z0.s, cg.p0, cg.z0.s);
    cg.St1w(cg.z0.s, cg.p0, AdrNoOfs(cg.x0));
    cg.Ret();
    cg.Ready();

    auto f = cg.GetCode<void (*)(int*)>();
    int arrOut[8];
    f(arrOut);
    ASSERT_EQ(arrOut[0], 0xf);
}

TEST(armcg_test_t, TestUxtl2)
{
    GTestJitGen cg;
    cg.Dup(cg.v1.b16, cg.w0);
    cg.Uxtl(cg.v1.s4, cg.v1.h4);
    cg.Mov(cg.w0, cg.v1.s4[0]);
    cg.Ret();
    cg.Ready();

    auto f = cg.GetCode<int (*)(int)>();
    ASSERT_EQ(f(0x10), 0x1010);
}

//move
TEST(armcg_test_t, TestFmov1)
{
    GTestJitGen cg;
    cg.Mov(cg.x9, 0);
    cg.Whilelt(cg.p0.s, cg.x9, cg.x0);
    cg.Fmov(cg.z0.s, cg.p0, 1.0);
    cg.St1w(cg.z0.s, cg.p0, AdrNoOfs(cg.x1));
    cg.Ret();
    cg.Ready();

    auto f = cg.GetCode<void (*)(int, float*)>();
    float arrF32[8];
    f(8, arrF32);

    ASSERT_FLOAT_EQ(arrF32[0], 1.0f);
}

TEST(armcg_test_t, TestFmov2)
{
    GTestJitGen cg;
    cg.Mov(cg.x9, 0);
    cg.Whilelt(cg.p0.s, cg.x9, cg.x0);
    cg.Fmov(cg.z0.s, 1.0);
    cg.St1w(cg.z0.s, cg.p0, AdrNoOfs(cg.x1));
    cg.Ret();
    cg.Ready();

    auto f = cg.GetCode<void (*)(int, float*)>();
    float arrF32[8];
    f(8, arrF32);

    ASSERT_FLOAT_EQ(arrF32[0], 1.0f);
}

TEST(armcg_test_t, TestFmov3)
{
    GTestJitGen cg;
    cg.Mov(cg.x9, 0);
    cg.Whilelt(cg.p0.s, cg.x9, cg.x0);
    cg.Fmov(cg.z0.s, 1.0f);
    cg.St1w(cg.z0.s, cg.p0, AdrNoOfs(cg.x1));
    cg.Ret();
    cg.Ready();

    auto f = cg.GetCode<void (*)(int, uint32_t*)>();
    uint32_t arrF32[8];
    f(8, arrF32);

    ASSERT_EQ(arrF32[0], 0x1);
}

TEST(armcg_test_t, TestFmov4)
{
    GTestJitGen cg;
    cg.Mov(cg.x9, 0);
    cg.Whilelt(cg.p0.s, cg.x9, cg.x0);
    cg.Fmov(cg.z0.h, 1.0f);
    cg.St1w(cg.z0.s, cg.p0, AdrNoOfs(cg.x1));
    cg.Ret();
    cg.Ready();

    auto f = cg.GetCode<void (*)(int, uint16_t*)>();
    uint16_t arrF16[16];
    f(8, arrF16);

    ASSERT_EQ(arrF16[0], 0x1);
}

TEST(armcg_test_t, TestFmov5)
{
    GTestJitGen cg;
    cg.Mov(cg.x9, 0);
    cg.Whilelt(cg.p0.s, cg.x9, cg.x0);
    cg.Mov(cg.w0, 5);
    cg.Fmov(cg.s0, cg.w0);
    cg.St1w(cg.z0.s, cg.p0, AdrNoOfs(cg.x1));
    cg.Ret();
    cg.Ready();

    int aCnvtVal = 0;
    auto f = cg.GetCode<void (*)(int, int*)>();
    f(1, &aCnvtVal);
    ASSERT_EQ(aCnvtVal, 0x5);
}

TEST(armcg_test_t, TestFmov6)
{
    GTestJitGen cg;
    cg.Dup(cg.z0.s, 8.0f);
    cg.Fmov(cg.w0, cg.s0);
    cg.Ret();
    cg.Ready();

    auto f = cg.GetCode<int (*)()>();
    ASSERT_EQ(f(), 0x8);
}

TEST(armcg_test_t, TestFmov7)
{
    GTestJitGen cg;
    cg.Dup(cg.z0.d, 1);
    cg.Fmov(cg.x0, cg.d0);
    cg.Fmov(cg.d1, cg.x0);
    cg.Ret();
    cg.Ready();

    auto f = cg.GetCode<int (*)()>();
    ASSERT_EQ(f(), 1.0);
}

TEST(armcg_test_t, TestMovImmAlias1)
{
    GTestJitGen cg;
    cg.Mov(cg.w0, 0);
    cg.Ret();
    cg.Ready();

    auto f = cg.GetCode<int (*)()>();
    ASSERT_EQ(f(), 0);
}

TEST(armcg_test_t, TestMovImmAlias2)
{
    GTestJitGen cg;
    cg.Mov(cg.w0, 0xffffffff);
    cg.Ret();
    cg.Ready();

    auto f = cg.GetCode<int (*)()>();
    ASSERT_EQ(f(), 0xffffffff);
}

TEST(armcg_test_t, TestMovImmAlias3)
{
    GTestJitGen cg;
    cg.Mov(cg.w0, 1234);
    cg.Ret();
    cg.Ready();

    auto f = cg.GetCode<int (*)()>();
    ASSERT_EQ(f(), 1234);
}

TEST(armcg_test_t, TestMov1)
{
    GTestJitGen cg;
    cg.Mov(cg.w0, 9999);
    // large imm
    cg.Mov(cg.x9, 154587);
    cg.Ret();
    cg.Ready();

    auto f = cg.GetCode<int (*)()>();
    ASSERT_EQ(f(), 9999);
}

TEST(armcg_test_t, TestMov2)
{
    GTestJitGen cg;
    cg.Mov(cg.x1, 9999);
    cg.Mov(cg.x0, cg.x1);
    cg.Ret();
    cg.Ready();

    auto f = cg.GetCode<int (*)()>();
    ASSERT_EQ(f(), 9999);
}

TEST(armcg_test_t, TestMov3)
{
    GTestJitGen cg;
    cg.Mov(cg.v1.d2[0], cg.x0);
    cg.Mov(cg.v2.b16, cg.v1.b16);
    cg.Mov(cg.v2.s4[1], cg.v2.s4[0]);
    cg.Mov(cg.w0, cg.v2.s4[1]);
    cg.Ret();
    cg.Ready();

    auto f = cg.GetCode<int (*)(int)>();
    ASSERT_EQ(f(1), 1);
}

TEST(armcg_test_t, TestMov4)
{
    GTestJitGen cg;
    cg.Ptrue(cg.p0.b);
    cg.Mov(cg.z0.b, cg.p0, 1);
    cg.Mov(cg.z2.s, cg.p0, 1);
    cg.Mov(cg.z3.d, cg.z0.d);
    cg.Mov(cg.z4.s, cg.w0);
    cg.And(cg.z5.d, cg.z3.d, cg.z4.d);
    cg.Orr(cg.z5.d, cg.z5.d, cg.z2.d);
    cg.Mov(cg.x9, 0);
    cg.Mov(cg.x10, 8);
    cg.Whilelt(cg.p1.s, cg.x9, cg.x10);
    cg.St1w(cg.z5.s, cg.p1, AdrNoOfs(cg.x1));

    cg.Ret();
    cg.Ready();

    auto f = cg.GetCode<void (*)(int, int*)>();
    int arrSlot[8];
    f(1, arrSlot);
    ASSERT_EQ(arrSlot[0], 0x1);
    ASSERT_EQ(arrSlot[1], 0x1);
    ASSERT_EQ(arrSlot[2], 0x1);
    ASSERT_EQ(arrSlot[3], 0x1);
    ASSERT_EQ(arrSlot[4], 0x1);
    ASSERT_EQ(arrSlot[5], 0x1);
    ASSERT_EQ(arrSlot[6], 0x1);
    ASSERT_EQ(arrSlot[7], 0x1);
}

TEST(armcg_test_t, TestMov5)
{
    GTestJitGen cg;
    cg.Mov(cg.z0.b, 1);
    cg.Mov(cg.z1.h, 1);
    cg.Mov(cg.z2.s, 1);
    cg.Ptrue(cg.p0.b);
    cg.Mov(cg.p1.b, cg.p0, cg.p0.b);
    cg.Mov(cg.p2.b, cg.p1.b);
    cg.Mov(cg.z3.b, cg.p2, cg.z0.b);
    cg.Mov(cg.z4.d, cg.p2, cg.z1.d);
    cg.Mov(cg.z5.s, cg.p2, cg.z2.s);
    cg.And(cg.z6.d, cg.z3.d, cg.z5.d);
    cg.Orr(cg.z6.d, cg.z6.d, cg.z4.d);
    cg.Mov(cg.x9, 0);
    cg.Mov(cg.x10, 8);
    cg.Whilelt(cg.p1.s, cg.x9, cg.x10);
    cg.St1w(cg.z6.s, cg.p1, AdrNoOfs(cg.x1));
    
    cg.Ret();
    cg.Ready();

    auto f = cg.GetCode<void (*)(int, int*)>();
    int arrSlot[8];
    f(1, arrSlot);
    ASSERT_EQ(arrSlot[0], 0x10001);
    ASSERT_EQ(arrSlot[1], 0x10001);
    ASSERT_EQ(arrSlot[2], 0x10001);
    ASSERT_EQ(arrSlot[3], 0x10001);
    ASSERT_EQ(arrSlot[4], 0x10001);
    ASSERT_EQ(arrSlot[5], 0x10001);
    ASSERT_EQ(arrSlot[6], 0x10001);
    ASSERT_EQ(arrSlot[7], 0x10001);
}

TEST(armcg_test_t, TestMovi)
{
    GTestJitGen cg;
    cg.Movi(cg.v1.s4, 1);
    cg.Movi(cg.v2.b16, 1);
    cg.Mov(cg.w0, cg.v1.s4[1]);
    cg.Ret();
    cg.Ready();

    auto f = cg.GetCode<int (*)()>();
    ASSERT_EQ(f(), 1);
}

TEST(armcg_test_t, TestMovk1)
{
    GTestJitGen cg;
    cg.Movk(cg.w0, 65535);
    cg.Ret();
    cg.Ready();

    auto f = cg.GetCode<int (*)(int)>();
    ASSERT_EQ(f(0), 65535);
}

TEST(armcg_test_t, TestMovk2)
{
    GTestJitGen cg;
    cg.Movk(cg.x0, 65535);
    cg.Ret();
    cg.Ready();

    auto f = cg.GetCode<int (*)(int)>();
    ASSERT_EQ(f(0), 65535);
}

TEST(armcg_test_t, TestMovn)
{
    GTestJitGen cg;
    cg.Movn(cg.x0, 0);
    cg.Ret();
    cg.Ready();

    auto f = cg.GetCode<int (*)(int)>();
    ASSERT_EQ(f(0), 0xffffffff);
}

TEST(armcg_test_t, TestMovprfx)
{
    GTestJitGen cg;
    cg.Mov(cg.x9, 0);
    cg.Mov(cg.x10, 8);
    cg.Whilelt(cg.p0.s, cg.x9, cg.x10);
    cg.Ld1w(cg.z0.s, cg.p0, AdrNoOfs(cg.x0));
    cg.Movprfx(cg.z1.s, cg.p0, cg.z0.s);
    cg.Fadd(cg.z1.s, cg.p0, cg.z0.s);
    cg.St1w(cg.z1.s, cg.p0, AdrNoOfs(cg.x1));
    cg.Ret();
    cg.Ready();
    auto f = cg.GetCode<void (*)(float*, float*)>();
    float arrF32[8] = {0.0f, -6.0f, 0.087116f, -0.046347f, 40.0f, -20.0f, 0.080593f, -0.524f};
    float arrOutput[8];
    f(arrF32, arrOutput);

    ASSERT_EQ(arrOutput[0], arrF32[0] + arrF32[0]);
    ASSERT_EQ(arrOutput[1], arrF32[1] + arrF32[1]);
    ASSERT_EQ(arrOutput[2], arrF32[2] + arrF32[2]);
    ASSERT_EQ(arrOutput[3], arrF32[3] + arrF32[3]);
    ASSERT_EQ(arrOutput[4], arrF32[4] + arrF32[4]);
    ASSERT_EQ(arrOutput[5], arrF32[5] + arrF32[5]);
    ASSERT_EQ(arrOutput[6], arrF32[6] + arrF32[6]);
    ASSERT_EQ(arrOutput[7], arrF32[7] + arrF32[7]);
}

TEST(armcg_test_t, TestMovz1)
{
    GTestJitGen cg;
    cg.Movz(cg.w0, 65535);
    cg.Ret();
    cg.Ready();

    auto f = cg.GetCode<int (*)(int)>();
    ASSERT_EQ(f(0xffffffff), 0xffff);
}

TEST(armcg_test_t, TestMovz2)
{
    GTestJitGen cg;
    cg.Movz(cg.x0, 65535);
    cg.Ret();
    cg.Ready();

    auto f = cg.GetCode<int (*)(int)>();
    ASSERT_EQ(f(0xffffffff), 0xffff);
}

TEST(armcg_test_t, MayIUse_1)
{
    bool ret = ARMCG::MayIUse(CPUISAT::ASIMD);
    ASSERT_EQ(ret, true);
}

TEST(armcg_test_t, MayIUse_2)
{
    bool ret = ARMCG::MayIUse(CPUISAT::VC_128);
    ASSERT_EQ(ret, true);
}

TEST(armcg_test_t, MayIUse_3)
{
    bool ret = ARMCG::MayIUse(CPUISAT::VC_256);
    ASSERT_EQ(ret, true);
}

TEST(armcg_test_t, MayIUse_4)
{
    bool ret = ARMCG::GetCpu().Has(ARMCG::Util::HwCap::CG_HWCAP_VC) &&
        ARMCG::GetCpu().GetVCLen() >= ARMCG::Util::VCLen::VC_384;
    ASSERT_EQ(ret, ARMCG::MayIUse(CPUISAT::VC_384));
}

TEST(armcg_test_t, MayIUse_5)
{
    bool ret = ARMCG::GetCpu().Has(ARMCG::Util::HwCap::CG_HWCAP_VC) &&
        ARMCG::GetCpu().GetVCLen() >= ARMCG::Util::VCLen::VC_512;
    ASSERT_EQ(ret, ARMCG::MayIUse(CPUISAT::VC_512));
}

TEST(armcg_test_t, MayIUse_6)
{
    bool ret = ARMCG::MayIUse(CPUISAT::ISA_UNDEF);
    ASSERT_EQ(ret, true);
}

TEST(armcg_test_t, MayIUse_7)
{
    bool ret = ARMCG::MayIUse(CPUISAT::ISA_ALL);
    ASSERT_EQ(ret, false);
}

TEST(armcg_test_t, GetVCLength_1)
{
    Util::VCLen ret = ARMCG::GetVCLength();
    ASSERT_GE(ret, ARMCG::Util::VCLen::VC_128);
}

TEST(armcg_test_t, IsAtomicSupported_1)
{
    bool ret = ARMCG::GetCpu().IsAtomicSupported();
    ASSERT_EQ(ret, true);
}

TEST(armcg_test_t, IsAtomicSupported_2)
{
    bool ret = ARMCG::GetCpu().IsBf16Supported();
    ASSERT_EQ(ret, true);
}

// Unit test for Type ARMCG::Util::CpuInfo::GetType() const;
// 1. Input: 
// 2. Expect: return >= 0;
TEST(armcg_test_t, GetType_1)
{
    ARMCG::Util::Type ret = ARMCG::GetCpu().GetType();
    ASSERT_GE(ret, static_cast<ARMCG::Util::Type>(0));
}

// error
TEST(armcg_test_t, Test0)
{
    ARMCG::Error err(static_cast<ARMCG::Err>(0));
    ASSERT_EQ(int(err), 0);
}

TEST(armcg_test_t, Test1)
{
    ARMCG::Error err(static_cast<ARMCG::Err>(1));
    ASSERT_EQ(int(err), 1);
}

TEST(armcg_test_t, Test2)
{
    ARMCG::Error err(static_cast<ARMCG::Err>(32));
    ASSERT_EQ(int(err), 32);
}

TEST(armcg_test_t, Test3)
{
    ARMCG::Error err(static_cast<ARMCG::Err>(32));
    const char *errMsg = Error::ConvertErrorToString(err);
    ASSERT_EQ(std::string(errMsg), std::string("bad err num"));
}


// test Addvl interface
TEST(armcg_test_t, TestAddvl)
{
    GTestJitGen cg;
    cg.Addvl(cg.x0, cg.x1, 0);
    cg.Ret();
    cg.Ready();

    auto Func = cg.GetCode<int (*)(int, int)>();
    ASSERT_EQ(Func(0, 4), 4);
}

TEST(armcg_test_t, TestSub9)
{
    GTestJitGen cg;
    cg.Mov(cg.x9, 0);
    cg.Whilelt(cg.p0.b, cg.x9, cg.x1);
    cg.Ld1b(cg.z0.b, cg.p0, AdrNoOfs(cg.x0));
    cg.Sub(cg.z1.b, cg.z0.b, cg.z0.b);
    cg.St1b(cg.z1.b, cg.p0, AdrNoOfs(cg.x0));
    cg.Ret();
    cg.Ready();

    auto Func = cg.GetCode<void (*)(uint8_t*, int)>();
    uint8_t arr1[] = {0x0, 0x0, 0xc6, 0x0, 0x2d, 0x93, 0x0, 0xa9, 0xef, 0x0, 0x51, 0x0, 0xcd, 0x0, 0x2d, 0x28,
        0x0, 0xb8, 0x31, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0};
    Func(arr1, 32);
    ASSERT_EQ(arr1[0], 0x0);
    ASSERT_EQ(arr1[2], 0x0);
    ASSERT_EQ(arr1[4], 0x0);
    ASSERT_EQ(arr1[5], 0x0);
    ASSERT_EQ(arr1[7], 0x0);
    ASSERT_EQ(arr1[8], 0x0);
    ASSERT_EQ(arr1[10], 0x0);
    ASSERT_EQ(arr1[12], 0x0);
    ASSERT_EQ(arr1[14], 0x0);
    ASSERT_EQ(arr1[15], 0x0);
    ASSERT_EQ(arr1[17], 0x0);
    ASSERT_EQ(arr1[18], 0x0);
}

TEST(armcg_test_t, TestFmul3)
{
    GTestJitGen cg;
    cg.Ptrue(cg.p0.h);
    cg.Ld1h(cg.z0.h, cg.p0, AdrNoOfs(cg.x0));
    cg.Ld1h(cg.z1.h, cg.p0, AdrNoOfs(cg.x1));
    cg.Fmul(cg.z0.h, cg.p0, cg.z1.h);
    cg.St1h(cg.z0.h, cg.p0, AdrNoOfs(cg.x0));
    cg.Ret();
    cg.Ready();

    auto Func = cg.GetCode<void (*)(uint16_t*, uint16_t*)>();
    uint16_t arr1[] = {0x0, 0x0, 0x2d93, 0x0, 0x2d93, 0x0, 0x2d93, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0,
                        0x0, 0x0, 0x2d93, 0x0, 0x2d93, 0x0, 0x2d93, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0};
    uint16_t arr2[] = {0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0,
                        0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0};
    Func(arr1, arr2);
    ASSERT_EQ(arr1[2], 0);
    ASSERT_EQ(arr1[4], 0);
    ASSERT_EQ(arr1[6], 0);
}

TEST(armcg_test_t, TestFmax2)
{
    GTestJitGen cg;
    cg.Mov(cg.x9, 0);
    cg.Whilelt(cg.p0.s, cg.x9, cg.x1);
    cg.Ld1w(cg.z0.s, cg.p0, AdrNoOfs(cg.x0));
    cg.Ld1w(cg.z1.s, cg.p0, AdrNoOfs(cg.x1));
    cg.Fmax(cg.z0.s, cg.p0, cg.z1.s);
    cg.St1w(cg.z0.s, cg.p0, AdrNoOfs(cg.x0));
    cg.Ret();
    cg.Ready();

    auto Func = cg.GetCode<void (*)(float*, float*, int)>();
    float arr1[] = {-1.0f, -2.0f, -3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    float arr2[] = {1.0f, 2.0f, 3.0f, -4.0f, -5.0f, -6.0f, -7.0f, 8.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    float arr3[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    Func(arr1, arr2, 8);
    ASSERT_EQ(arr3[0], arr1[0]);
    ASSERT_EQ(arr3[1], arr1[1]);
    ASSERT_EQ(arr3[2], arr1[2]);
    ASSERT_EQ(arr3[3], arr1[3]);
    ASSERT_EQ(arr3[4], arr1[4]);
    ASSERT_EQ(arr3[5], arr1[5]);
    ASSERT_EQ(arr3[6], arr1[6]);
    ASSERT_EQ(arr3[7], arr1[7]);
}

TEST(armcg_test_t, TestFmaxnm2)
{
    GTestJitGen cg;
    cg.Mov(cg.x9, 0);
    cg.Whilelt(cg.p0.s, cg.x9, cg.x1);
    cg.Ld1w(cg.z0.s, cg.p0, AdrNoOfs(cg.x0));
    cg.Ld1w(cg.z1.s, cg.p0, AdrNoOfs(cg.x1));
    cg.Fmaxnm(cg.z0.s, cg.p0, cg.z1.s);
    cg.St1w(cg.z0.s, cg.p0, AdrNoOfs(cg.x0));
    cg.Ret();
    cg.Ready();

    auto Func = cg.GetCode<void (*)(float*, float*, int)>();
    float arr1[] = {-1.0f, -2.0f, -3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    float arr2[] = {1.0f, 2.0f, 3.0f, -4.0f, -5.0f, -6.0f, -7.0f, 8.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    float arr3[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    Func(arr1, arr2, 8);
    ASSERT_EQ(arr3[0], arr1[0]);
    ASSERT_EQ(arr3[1], arr1[1]);
    ASSERT_EQ(arr3[2], arr1[2]);
    ASSERT_EQ(arr3[3], arr1[3]);
    ASSERT_EQ(arr3[4], arr1[4]);
    ASSERT_EQ(arr3[5], arr1[5]);
    ASSERT_EQ(arr3[6], arr1[6]);
    ASSERT_EQ(arr3[7], arr1[7]);
}

TEST(armcg_test_t, TestFmin2)
{
    GTestJitGen cg;
    cg.Mov(cg.x9, 0);
    cg.Whilelt(cg.p0.s, cg.x9, cg.x1);
    cg.Ld1w(cg.z0.s, cg.p0, AdrNoOfs(cg.x0));
    cg.Ld1w(cg.z1.s, cg.p0, AdrNoOfs(cg.x1));
    cg.Fmin(cg.z0.s, cg.p0, cg.z1.s);
    cg.St1w(cg.z0.s, cg.p0, AdrNoOfs(cg.x0));
    cg.Ret();
    cg.Ready();

    auto Func = cg.GetCode<void (*)(float*, float*, int)>();
    float arr1[] = {0.0f, 1.0f, -3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    float arr2[] = {1.0f, 2.0f, 3.0f, -4.0f, -5.0f, -6.0f, 7.0f, 8.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    float arr3[] = {0.0f, 1.0f, -3.0f, -4.0f, -5.0f, -6.0f, 7.0f, 8.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    Func(arr1, arr2, 8);
    ASSERT_EQ(arr3[0], arr1[0]);
    ASSERT_EQ(arr3[1], arr1[1]);
    ASSERT_EQ(arr3[2], arr1[2]);
    ASSERT_EQ(arr3[3], arr1[3]);
    ASSERT_EQ(arr3[4], arr1[4]);
    ASSERT_EQ(arr3[5], arr1[5]);
    ASSERT_EQ(arr3[6], arr1[6]);
    ASSERT_EQ(arr3[7], arr1[7]);
}

// test Fminnm interface
TEST(armcg_test_t, TestFminnm1)
{
    GTestJitGen cg;
    cg.Mov(cg.x9, 0);
    cg.Whilelt(cg.p0.s, cg.x9, cg.x1);
    cg.Ld1w(cg.z0.s, cg.p0, AdrNoOfs(cg.x0));
    cg.Ld1w(cg.z1.s, cg.p0, AdrNoOfs(cg.x1));
    cg.Fminnm(cg.z0.s, cg.p0, cg.z1.s);
    cg.St1w(cg.z0.s, cg.p0, AdrNoOfs(cg.x0));
    cg.Ret();
    cg.Ready();

    auto Func = cg.GetCode<void (*)(float*, float*, int)>();
    float arr1[] = {0.0f, 1.0f, -3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    float arr2[] = {1.0f, 2.0f, 3.0f, -4.0f, -5.0f, -6.0f, 7.0f, 8.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    float arr3[] = {0.0f, 1.0f, -3.0f, -4.0f, -5.0f, -6.0f, 7.0f, 8.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    Func(arr1, arr2, 8);
    ASSERT_EQ(arr3[0], arr1[0]);
    ASSERT_EQ(arr3[1], arr1[1]);
    ASSERT_EQ(arr3[2], arr1[2]);
    ASSERT_EQ(arr3[3], arr1[3]);
    ASSERT_EQ(arr3[4], arr1[4]);
    ASSERT_EQ(arr3[5], arr1[5]);
    ASSERT_EQ(arr3[6], arr1[6]);
    ASSERT_EQ(arr3[7], arr1[7]);
}

TEST(armcg_test_t, TestSmin2)
{
    GTestJitGen cg;
    cg.Mov(cg.x9, 0);
    cg.Whilelt(cg.p0.s, cg.x9, cg.x1);
    cg.Ld1w(cg.z0.s, cg.p0, AdrNoOfs(cg.x0));
    cg.Ld1w(cg.z1.s, cg.p0, AdrNoOfs(cg.x1));
    cg.Smin(cg.z0.s, cg.p0, cg.z1.s);
    cg.St1w(cg.z0.s, cg.p0, AdrNoOfs(cg.x0));
    cg.Ret();
    cg.Ready();

    auto Func = cg.GetCode<void (*)(int*, int*, int)>();
    int arr1[] = {-1, -2, -3, 4, 5, 6, 7, 8, 0, 0, 0, 0, 0, 0, 0, 0};
    int arr2[] = {1, 2, 3, -4, -5, -6, -7, -8, 0, 0, 0, 0, 0, 0, 0, 0};
    int arr3[] = {-1, -2, -3, -4, -5, -6, -7, -8, 0, 0, 0, 0, 0, 0, 0, 0};
    Func(arr1, arr2, 8);
    ASSERT_EQ(arr3[0], arr1[0]);
    ASSERT_EQ(arr3[1], arr1[1]);
    ASSERT_EQ(arr3[2], arr1[2]);
    ASSERT_EQ(arr3[3], arr1[3]);
    ASSERT_EQ(arr3[4], arr1[4]);
    ASSERT_EQ(arr3[5], arr1[5]);
    ASSERT_EQ(arr3[6], arr1[6]);
    ASSERT_EQ(arr3[7], arr1[7]);
}

// test Frecpe interface
TEST(armcg_test_t, TestFrecpe)
{
    GTestJitGen cg;
    cg.Ptrue(cg.p0.s);
    cg.Ld1w(cg.z0.s, cg.p0, AdrNoOfs(cg.x0));
    cg.Frecpe(cg.z1.s, cg.z0.s);
    cg.Frecps(cg.z2.s, cg.z0.s, cg.z1.s);
    cg.Fmul(cg.z1.s, cg.z1.s, cg.z2.s);

    cg.Frecps(cg.z2.s, cg.z0.s, cg.z1.s);
    cg.Fmul(cg.z1.s, cg.z1.s, cg.z2.s);

    cg.Frecps(cg.z2.s, cg.z0.s, cg.z1.s);
    cg.Fmul(cg.z1.s, cg.z1.s, cg.z2.s);

    cg.Frecps(cg.z2.s, cg.z0.s, cg.z1.s);
    cg.St1w(cg.z2.s, cg.p0, AdrNoOfs(cg.x0));
    cg.Ret();
    cg.Ready();

    auto Func = cg.GetCode<void (*)(float*)>();
    float arr1[] = {-1.0f, -2.0f, -3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    Func(arr1);
    ASSERT_EQ(arr1[0], 1);
    ASSERT_EQ(arr1[1], 1);
    ASSERT_EQ(arr1[2], 1);
    ASSERT_EQ(arr1[3], 1);
    ASSERT_EQ(arr1[4], 1);
    ASSERT_EQ(arr1[5], 1);
    ASSERT_EQ(arr1[6], 1);
    ASSERT_EQ(arr1[7], 1);
}

// test Frecps interface
TEST(armcg_test_t, TestFrecps)
{
    GTestJitGen cg;
    cg.Mov(cg.x9, 0);
    cg.Whilelt(cg.p0.s, cg.x9, cg.x1);
    cg.Ld1w(cg.z0.s, cg.p0, AdrNoOfs(cg.x0));
    cg.Ld1w(cg.z1.s, cg.p0, AdrNoOfs(cg.x1));
    cg.Frecps(cg.z0.s, cg.z0.s, cg.z1.s);
    cg.St1w(cg.z0.s, cg.p0, AdrNoOfs(cg.x0));
    cg.Ret();
    cg.Ready();

    auto Func = cg.GetCode<void (*)(float*, float*, int)>();
    float arr1[] = {-1.0f, -2.0f, -3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    float arr2[] = {1.0f, 2.0f, 3.0f, -4.0f, -5.0f, -6.0f, -7.0f, 2.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    float arr3[] = {3.0f, 6.0f, 11.0f, 18.0f, 27.0, 38.0f, 51.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    Func(arr1, arr2, 8);
    ASSERT_EQ(arr3[0], arr1[0]);
    ASSERT_EQ(arr3[1], arr1[1]);
    ASSERT_EQ(arr3[2], arr1[2]);
    ASSERT_EQ(arr3[3], arr1[3]);
    ASSERT_EQ(arr3[4], arr1[4]);
    ASSERT_EQ(arr3[5], arr1[5]);
    ASSERT_EQ(arr3[6], arr1[6]);
    ASSERT_EQ(arr3[7], arr1[7]);
}

// test Fscale interface
TEST(armcg_test_t, TestFscale)
{
    GTestJitGen cg;
    cg.Ptrue(cg.p0.s);
    cg.Ld1w(cg.z0.s, cg.p0, AdrNoOfs(cg.x0));
    cg.Ld1w(cg.z1.s, cg.p0, AdrNoOfs(cg.x1));
    cg.Fscale(cg.z0.s, cg.p0, cg.z1.s);
    cg.St1w(cg.z0.s, cg.p0, AdrNoOfs(cg.x0));
    cg.Ret();
    cg.Ready();

    auto Func = cg.GetCode<void (*)(float*, int*)>();
    float arr1[] = {-1.0f, -2.0f, -3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    int arr2[] = {1, 2, 3, -4, -5, -6, -7, -8, 0, 0, 0, 0, 0, 0, 0, 0};
    float arr3[] = {-2.0f, -8.0f, -24.0f, 0.25f, 5 / 32.0f, 6 / 64.0, 7 / 128.0f, 8 / 256.0f,
                    0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    Func(arr1, arr2);
    ASSERT_EQ(arr3[0], arr1[0]);
    ASSERT_EQ(arr3[1], arr1[1]);
    ASSERT_EQ(arr3[2], arr1[2]);
    ASSERT_EQ(arr3[3], arr1[3]);
    ASSERT_EQ(arr3[4], arr1[4]);
    ASSERT_EQ(arr3[5], arr1[5]);
    ASSERT_EQ(arr3[6], arr1[6]);
    ASSERT_EQ(arr3[7], arr1[7]);
}

// test Sdot interface
TEST(armcg_test_t, TestSdot)
{
    GTestJitGen cg;
    cg.Ptrue(cg.p0.b);
    cg.Ld1b(cg.z0.b, cg.p0, AdrNoOfs(cg.x0));
    cg.Ld1b(cg.z1.b, cg.p0, AdrNoOfs(cg.x1));
    cg.Sdot(cg.z1.s, cg.z0.b, cg.z0.b);
    cg.St1w(cg.z1.s, cg.p0, AdrNoOfs(cg.x1));
    cg.Ret();
    cg.Ready();

    auto Func = cg.GetCode<void (*)(uint8_t*, int*)>();
    uint8_t arr1[64] = {0, 1, 2, 3, 4, 5, 6, 7, 0, 0, 0, 0, 0, 0, 0, 0};
    int arr2[] = {0, 1, 2, 3, 4, 5, 6, 7, 0, 0, 0, 0, 0, 0, 0, 0};
    Func(arr1, arr2);
    EXPECT_EQ(arr2[0], 14);
    EXPECT_EQ(arr2[1], 127);
}

TEST(armcg_test_t, TestDot)
{
    GTestJitGen cg;
    cg.Ptrue(cg.p0.b);
    cg.Ld1b(cg.z0.b, cg.p0, AdrNoOfs(cg.x0));
    cg.Ld1b(cg.z1.b, cg.p0, AdrNoOfs(cg.x1));
    cg.Ld1w(cg.z2.s, cg.p0, AdrNoOfs(cg.x2));
    cg.Ld1w(cg.z3.s, cg.p0, AdrNoOfs(cg.x3));
    cg.Udot(cg.z2.s, cg.z0.b, cg.z0.b);
    cg.Usdot(cg.z3.s, cg.z0.b, cg.z1.b);
    cg.St1w(cg.z2.s, cg.p0, AdrNoOfs(cg.x2));
    cg.St1w(cg.z3.s, cg.p0, AdrNoOfs(cg.x3));
    cg.Ret();
    cg.Ready();

    auto f = cg.GetCode<void (*)(uint8_t*, int8_t*, uint32_t*, int32_t*)>();
    uint8_t arr1[64] = {1, 2, 3, 4, 5, 6, 7, 8};
    int8_t arr2[64] = {-1, -2, -3, -4, -5, -6, -7, -8};
    uint32_t ret1[16] = {0, 0, 0, 0, 0, 0, 0, 0};
    int32_t ret2[16] = {0, 0, 0, 0, 0, 0, 0, 0};
    f(arr1, arr2, ret1, ret2);
    EXPECT_EQ(ret1[0], 30);
    EXPECT_EQ(ret1[1], 174);
    EXPECT_EQ(ret2[0], -30);
    EXPECT_EQ(ret2[1], -174);
}

TEST(armcg_test_t, TestSt1b1)
{
    GTestJitGen cg;
    cg.Mov(cg.x9, 0);
    cg.Mov(cg.x10, 8);
    cg.Whilelt(cg.p0.s, cg.x9, cg.x10);
    cg.Dup(cg.z0.b, 1);
    cg.St1b(cg.z0.s, cg.p0, AdrNoOfs(cg.x0));
    cg.Ret();
    cg.Ready();

    auto Func = cg.GetCode<void (*)(int*)>();
    int arr1[8];
    Func(arr1);
    EXPECT_EQ(arr1[0], 0x1010101);
}

TEST(armcg_test_t, TestSt1b2)
{
    GTestJitGen cg;
    cg.Mov(cg.x9, 0);
    cg.Mov(cg.x10, 8);
    cg.Whilelt(cg.p0.s, cg.x9, cg.x10);
    cg.Dup(cg.z0.b, 1);
    cg.St1b(cg.z0.s, cg.p0, AdrScImm(cg.x0, 0));
    cg.Ret();
    cg.Ready();

    auto Func = cg.GetCode<void (*)(int*)>();
    int arr1[8];
    Func(arr1);
    EXPECT_EQ(arr1[0], 0x1010101);
}

TEST(armcg_test_t, TestSt1h1)
{
    GTestJitGen cg;
    cg.Mov(cg.x9, 0);
    cg.Mov(cg.x10, 8);
    cg.Whilelt(cg.p0.s, cg.x9, cg.x10);
    cg.Dup(cg.z0.b, 1);
    cg.St1h(cg.z0.s, cg.p0, AdrNoOfs(cg.x0));
    cg.Ret();
    cg.Ready();

    auto Func = cg.GetCode<void (*)(int*)>();
    int arr1[8];
    Func(arr1);
    EXPECT_EQ(arr1[0], 0x1010101);
}
#ifdef KUDNNL_920F
TEST(armcg_test_t, TestMCSt1hHTile2Mem)
{
    GTestJitGen cg; // zah(ZARegHH)ss(AdrScSc)
    cg.Smstart(StreamMode::SMZA);
    cg.Mov(cg.w15, 0);
    cg.SetPreg(cg.p0.h, 8);
    cg.Mov(cg.x9, 0);
    cg.Ld1h(cg.za0h.h, cg.w15, 0, cg.p0 / PredType::T_Z,
        AdrScSc(cg.x1, cg.x9, ShMod::LSL, static_cast<uint32_t>(1)));
    cg.St1h(cg.za0h.h, cg.w15, 0, cg.p0,
        AdrScSc(cg.x0, cg.x9, ShMod::LSL, static_cast<uint32_t>(1)));
    cg.Smstop(StreamMode::SMZA);
    cg.Ret();
    cg.Ready();
    auto cgFunc = cg.GetCode<void (*)(uint16_t*, uint16_t*)>();
    uint16_t cgSrc[8] = {1, 2, 3, 4, 5, 6, 7, 8};
    uint16_t cgDst[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    cgFunc(cgDst, cgSrc);
    ASSERT_EQ(cgSrc[0], cgDst[0]);
    ASSERT_EQ(cgSrc[1], cgDst[1]);
}
TEST(armcg_test_t, TestMCSt1hVTile2Mem)
{
    GTestJitGen cg; // zav(ZARegVH)ss(AdrScSc)
    cg.Smstart(StreamMode::SMZA);
    cg.Mov(cg.w15, 0);
    cg.SetPreg(cg.p0.h, 8);
    cg.Mov(cg.x9, 0);
    cg.Ld1h(cg.za0v.h, cg.w15, 0, cg.p0 / PredType::T_Z,
        AdrScSc(cg.x1, cg.x9, ShMod::LSL, static_cast<uint32_t>(1)));
    cg.St1h(cg.za0v.h, cg.w15, 0, cg.p0,
        AdrScSc(cg.x0, cg.x9, ShMod::LSL, static_cast<uint32_t>(1)));
    cg.Smstop(StreamMode::SMZA);
    cg.Ret();
    cg.Ready();
    auto cgFunc = cg.GetCode<void (*)(uint16_t*, uint16_t*)>();
    uint16_t cgSrc[8] = {1, 2, 3, 4, 5, 6, 7, 8};
    uint16_t cgDst[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    cgFunc(cgDst, cgSrc);
    ASSERT_EQ(cgSrc[0], cgDst[0]);
    ASSERT_EQ(cgSrc[1], cgDst[1]);
}
#endif

TEST(armcg_test_t, TestSt1w)
{
    GTestJitGen cg;
    cg.Mov(cg.x9, 0);
    cg.Mov(cg.x10, 4);
    cg.Whilelt(cg.p0.s, cg.x9, cg.x10);
    cg.Mov(cg.x11, 0);
    cg.Ld1w(cg.z0.s, cg.p0, AdrScSc(cg.x0, cg.x11, ShMod::LSL, 2));
    cg.St1w(cg.z0.s, cg.p0, AdrScSc(cg.x1, cg.x11, ShMod::LSL, 2));
    cg.Ret();
    cg.Ready();

    auto f = cg.GetCode<void (*)(int*, int*)>();
    int buf1[4] = {1, 2, 3, 4};
    int buf2[4] = {0, 0, 0, 0};
    f(buf1, buf2);
    EXPECT_EQ(buf2[0], 1);
    EXPECT_EQ(buf2[1], 2);
    EXPECT_EQ(buf2[2], 3);
    EXPECT_EQ(buf2[3], 4);
}

TEST(armcg_test_t, TestCmple)
{
    GTestJitGen bi; // b(ZRegB)i(int32_t)
    bi.SetPreg(bi.p1.b, 16);
    bi.Ld1b(bi.z0.b, bi.p1.b, AdrNoOfs(bi.x0));
    bi.Cmple(bi.p0.b, bi.p1 / PredType::T_Z, bi.z0.b, 1);
    bi.St1b(bi.z0.b, bi.p1.b, AdrNoOfs(bi.x1));
    bi.Ret();
    bi.Ready();
    auto biFunc = bi.GetCode<void (*)(uint8_t*, uint8_t*)>();
    uint8_t biSrc[16] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    uint8_t biDst[16] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
    biFunc(biSrc, biDst);
    ASSERT_EQ(biDst[0], 0);

    GTestJitGen hi; // h(ZRegH)i(int32_t)
    hi.SetPreg(hi.p1.h, 8);
    hi.Ld1h(hi.z0.h, hi.p1.h, AdrNoOfs(hi.x0));
    hi.Cmple(hi.p0.h, hi.p1 / PredType::T_Z, hi.z0.h, 1);
    hi.St1h(hi.z0.h, hi.p1.h, AdrNoOfs(hi.x1));
    hi.Ret();
    hi.Ready();
    auto hiFunc = hi.GetCode<void (*)(uint16_t*, uint16_t*)>();
    uint16_t hiSrc[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    uint16_t hiDst[8] = {1, 1, 1, 1, 1, 1, 1, 1};
    hiFunc(hiSrc, hiDst);
    ASSERT_EQ(hiDst[0], 0);

    GTestJitGen si; // s(ZRegS)i(int32_t)
    si.SetPreg(si.p1.s, 4);
    si.Ld1w(si.z0.s, si.p1.s, AdrNoOfs(si.x0));
    si.Cmple(si.p0.s, si.p1 / PredType::T_Z, si.z0.s, 1);
    si.St1w(si.z0.s, si.p1.s, AdrNoOfs(si.x1));
    si.Ret();
    si.Ready();
    auto siFunc = si.GetCode<void (*)(uint32_t*, uint32_t*)>();
    uint32_t siSrc[4] = {0, 0, 0, 0};
    uint32_t siDst[4] = {1, 1, 1, 1};
    siFunc(siSrc, siDst);
    ASSERT_EQ(siDst[0], 0);

    GTestJitGen bb; // b(ZRegB)b(ZRegB)
    bb.SetPreg(bb.p1.b, 16);
    bb.Ld1b(bb.z0.b, bb.p1.b, AdrNoOfs(bb.x0));
    bb.Ld1b(bb.z1.b, bb.p1.b, AdrNoOfs(bb.x0));
    bb.Cmple(bb.p0.b, bb.p1 / PredType::T_Z, bb.z0.b, 1);
    bb.St1b(bb.z0.b, bb.p1.b, AdrNoOfs(bb.x1));
    bb.Ret();
    bb.Ready();
    auto bbFunc = bb.GetCode<void (*)(uint8_t*, uint8_t*)>();
    uint8_t bbSrc[16] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    uint8_t bbDst[16] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
    bbFunc(bbSrc, bbDst);
    ASSERT_EQ(bbDst[0], 0);

    GTestJitGen hh; // h(ZRegH)h(ZRegH)
    hh.SetPreg(hh.p1.h, 8);
    hh.Ld1h(hh.z0.h, hh.p1.h, AdrNoOfs(hh.x0));
    hh.Ld1h(hh.z1.h, hh.p1.h, AdrNoOfs(hh.x0));
    hh.Cmple(hh.p0.h, hh.p1 / PredType::T_Z, hh.z0.h, 1);
    hh.St1h(hh.z0.h, hh.p1.h, AdrNoOfs(hh.x1));
    hh.Ret();
    hh.Ready();
    auto hhFunc = hh.GetCode<void (*)(uint16_t*, uint16_t*)>();
    uint16_t hhSrc[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    uint16_t hhDst[8] = {1, 1, 1, 1, 1, 1, 1, 1};
    hhFunc(hhSrc, hhDst);
    ASSERT_EQ(hhDst[0], 0);

    GTestJitGen ss; // s(ZRegS)s(ZRegS)
    ss.SetPreg(ss.p1.s, 4);
    ss.Ld1w(ss.z0.s, ss.p1.s, AdrNoOfs(ss.x0));
    ss.Ld1w(ss.z1.s, ss.p1.s, AdrNoOfs(ss.x0));
    ss.Cmple(ss.p0.s, ss.p1 / PredType::T_Z, ss.z0.s, 1);
    ss.St1w(ss.z0.s, ss.p1.s, AdrNoOfs(ss.x1));
    ss.Ret();
    ss.Ready();
    auto ssFunc = ss.GetCode<void (*)(uint32_t*, uint32_t*)>();
    uint32_t ssSrc[4] = {0, 0, 0, 0};
    uint32_t ssDst[4] = {1, 1, 1, 1};
    ssFunc(ssSrc, ssDst);
    ASSERT_EQ(ssDst[0], 0);
}

TEST(armcg_test_t, TestFcmge)
{
    GTestJitGen s2s2d; // s2(VReg2S)s2(VReg2S)d(double)
    s2s2d.Ldr(s2s2d.s1, AdrUimm(s2s2d.x1, 0));
    s2s2d.Fcmge(s2s2d.v0.s2, s2s2d.v1.s2, 0.0f);
    s2s2d.Str(s2s2d.s0, AdrUimm(s2s2d.x0, 0));
    s2s2d.Ret();
    s2s2d.Ready();
    auto s2s2dFunc = s2s2d.GetCode<void (*)(uint32_t*, float*)>();
    float s2s2dSrc[4] = {1.0f, -1.0f, 0.0f, 0.0f};
    uint32_t s2s2dDst[4];
    s2s2dFunc(s2s2dDst, s2s2dSrc);
    ASSERT_EQ(s2s2dDst[0], 0xffffffff);

    GTestJitGen s4s4d; // s4(VReg4S)s4(VReg4S)d(double)
    s4s4d.Ldr(s4s4d.s1, AdrUimm(s4s4d.x1, 0));
    s4s4d.Fcmge(s4s4d.v0.s4, s4s4d.v1.s4, 0.0f);
    s4s4d.Str(s4s4d.s0, AdrUimm(s4s4d.x0, 0));
    s4s4d.Ret();
    s4s4d.Ready();
    auto s4s4dFunc = s4s4d.GetCode<void (*)(uint32_t*, float*)>();
    float s4s4dSrc[4] = {1.0f, -1.0f, 0.0f, 0.0f};
    uint32_t s4s4dDst[4];
    s4s4dFunc(s4s4dDst, s4s4dSrc);
    ASSERT_EQ(s4s4dDst[0], 0xffffffff);

    GTestJitGen s2s2s2; // s2(VReg2S)s2(VReg2S)s2(VReg2S)
    s2s2s2.Ldr(s2s2s2.s1, AdrUimm(s2s2s2.x1, 0));
    s2s2s2.Ldr(s2s2s2.s2, AdrUimm(s2s2s2.x2, 0));
    s2s2s2.Fcmge(s2s2s2.v0.s2, s2s2s2.v1.s2, s2s2s2.v2.s2);
    s2s2s2.Str(s2s2s2.s0, AdrUimm(s2s2s2.x0, 0));
    s2s2s2.Ret();
    s2s2s2.Ready();
    auto s2s2s2Func = s2s2s2.GetCode<void (*)(uint32_t*, float*, float*)>();
    float s2s2s2Src1[4] = {1.0f, -1.0f, 0.0f, 0.0f};
    float s2s2s2Src2[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    uint32_t s2s2s2Dst[4];
    s2s2s2Func(s2s2s2Dst, s2s2s2Src1, s2s2s2Src2);
    ASSERT_EQ(s2s2s2Dst[0], 0xffffffff);

    GTestJitGen s4s4s4; // s4(VReg4S)s4(VReg4S)s4(VReg4S)
    s4s4s4.Ldr(s4s4s4.s1, AdrUimm(s4s4s4.x1, 0));
    s4s4s4.Ldr(s4s4s4.s2, AdrUimm(s4s4s4.x2, 0));
    s4s4s4.Fcmge(s4s4s4.v0.s4, s4s4s4.v1.s4, s4s4s4.v2.s4);
    s4s4s4.Str(s4s4s4.s0, AdrUimm(s4s4s4.x0, 0));
    s4s4s4.Ret();
    s4s4s4.Ready();
    auto s4s4s4Func = s4s4s4.GetCode<void (*)(uint32_t*, float*, float*)>();
    float s4s4s4Src1[4] = {1.0f, -1.0f, 0.0f, 0.0f};
    float s4s4s4Src2[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    uint32_t s4s4s4Dst[4];
    s4s4s4Func(s4s4s4Dst, s4s4s4Src1, s4s4s4Src2);
    ASSERT_EQ(s4s4s4Dst[0], 0xffffffff);

    GTestJitGen ssd; // s(SReg)s(SReg)d(double)
    ssd.Ldr(ssd.s1, AdrUimm(ssd.x1, 0));
    ssd.Fcmge(ssd.s0, ssd.s1, 0.0f);
    ssd.Str(ssd.s0, AdrUimm(ssd.x0, 0));
    ssd.Ret();
    ssd.Ready();
    auto ssdFunc = ssd.GetCode<void (*)(uint32_t*, float*)>();
    float ssdSrc[4] = {1.0f, -1.0f, 0.0f, 0.0f};
    uint32_t ssdDst[4];
    ssdFunc(ssdDst, ssdSrc);
    ASSERT_EQ(ssdDst[0], 0xffffffff);

    GTestJitGen sss; // s(SReg)s(SReg)s(SReg)
    sss.Ldr(sss.s1, AdrUimm(sss.x1, 0));
    sss.Ldr(sss.s2, AdrUimm(sss.x2, 0));
    sss.Fcmge(sss.s0, sss.s1, sss.s2);
    sss.Str(sss.s0, AdrUimm(sss.x0, 0));
    sss.Ret();
    sss.Ready();
    auto sssFunc = sss.GetCode<void (*)(uint32_t*, float*, float*)>();
    float sssSrc1[4] = {1.0f, -1.0f, 0.0f, 0.0f};
    float sssSrc2[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    uint32_t sssDst[4];
    sssFunc(sssDst, sssSrc1, sssSrc2);
    ASSERT_EQ(sssDst[0], 0xffffffff);
}

TEST(armcg_test_t, TestFcmle)
{
    GTestJitGen pphd; // p(PRegH)p(PRegBase)h(ZRegH)d(double)
    pphd.SetPreg(pphd.p1.h, 8);
    pphd.Ld1h(pphd.z0.h, pphd.p1, AdrNoOfs(pphd.x0));
    pphd.Fcmle(pphd.p0.h, pphd.p1 / PredType::T_Z, pphd.z0.h, 0.0f);
    pphd.St1h(pphd.z0.h, pphd.p0, AdrNoOfs(pphd.x1));
    pphd.Ret();
    pphd.Ready();
    auto pphdFunc = pphd.GetCode<void (*)(uint16_t*, uint16_t*)>();
    uint16_t pphdSrc[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    uint16_t pphdDst[8] = {1, 1, 1, 1, 1, 1, 1, 1};
    pphdFunc(pphdSrc, pphdDst);
    ASSERT_EQ(pphdDst[0], 0);

    GTestJitGen ppsd; // p(PRegS)p(PRegBase)s(ZRegS)d(double)
    ppsd.SetPreg(ppsd.p1.s, 4);
    ppsd.Ld1w(ppsd.z0.s, ppsd.p1, AdrNoOfs(ppsd.x0));
    ppsd.Fcmle(ppsd.p0.s, ppsd.p1 / PredType::T_Z, ppsd.z0.s, 0.0f);
    ppsd.St1w(ppsd.z0.s, ppsd.p0, AdrNoOfs(ppsd.x1));
    ppsd.Ret();
    ppsd.Ready();
    auto ppsdFunc = ppsd.GetCode<void (*)(uint32_t*, uint32_t*)>();
    uint32_t ppsdSrc[4] = {0, 0, 0, 0};
    uint32_t ppsdDst[4] = {1, 1, 1, 1};
    ppsdFunc(ppsdSrc, ppsdDst);
    ASSERT_EQ(ppsdDst[0], 0);

    GTestJitGen pphh; // p(PRegH)p(PRegBase)h(ZRegH)h(ZRegH)
    pphh.SetPreg(pphh.p1.h, 8);
    pphh.Ld1h(pphh.z0.h, pphh.p1, AdrNoOfs(pphh.x0));
    pphh.Ld1h(pphh.z1.h, pphh.p1, AdrNoOfs(pphh.x0));
    pphh.Fcmle(pphh.p0.h, pphh.p1 / PredType::T_Z, pphh.z0.h, pphh.z1.h);
    pphh.St1h(pphh.z0.h, pphh.p0, AdrNoOfs(pphh.x1));
    pphh.Ret();
    pphh.Ready();
    auto pphhFunc = pphh.GetCode<void (*)(uint16_t*, uint16_t*)>();
    uint16_t pphhSrc[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    uint16_t pphhDst[8] = {1, 1, 1, 1, 1, 1, 1, 1};
    pphhFunc(pphhSrc, pphhDst);
    ASSERT_EQ(pphhDst[0], 0);

    GTestJitGen ppss; // p(PRegS)p(PRegBase)s(ZRegS)s(ZRegS)
    ppss.SetPreg(ppss.p1.s, 4);
    ppss.Ld1w(ppss.z0.s, ppss.p1, AdrNoOfs(ppss.x0));
    ppss.Ld1w(ppss.z1.s, ppss.p1, AdrNoOfs(ppss.x0));
    ppss.Fcmle(ppss.p0.s, ppss.p1 / PredType::T_Z, ppss.z0.s, ppss.z1.s);
    ppss.St1w(ppss.z0.s, ppss.p0, AdrNoOfs(ppss.x1));
    ppss.Ret();
    ppss.Ready();
    auto ppssFunc = ppsd.GetCode<void (*)(uint32_t*, uint32_t*)>();
    uint32_t ppssSrc[4] = {0, 0, 0, 0};
    uint32_t ppssDst[4] = {1, 1, 1, 1};
    ppssFunc(ppssSrc, ppssDst);
    ASSERT_EQ(ppssDst[0], 0);
}

TEST(armcg_test_t, TestUsmmla)
{
    GTestJitGen cg;
    cg.Ptrue(cg.p0.b);
    cg.Ld1b(cg.z1.b, cg.p0, AdrNoOfs(cg.x1));
    cg.Ld1b(cg.z2.b, cg.p0, AdrNoOfs(cg.x2));
    cg.Eor(cg.z3.d, cg.z3.d, cg.z3.d);
    cg.Usmmla(cg.z3.s, cg.z1.b, cg.z2.b);
    cg.St1w(cg.z3.s, cg.p0, AdrNoOfs(cg.x0));
    cg.Ret();
    cg.Ready();

    auto f = cg.GetCode<void (*)(int32_t*, uint8_t*, int8_t*)>();
    uint8_t src1[64] = {
        1, 0, 0, 0, 0, 0, 0, 0,
        2, 0, 0, 0, 0, 0, 0, 0,
        3, 0, 0, 0, 0, 0, 0, 0,
        4, 0, 0, 0, 0, 0, 0, 0};
    int8_t src2[64] = {
        1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1};
    int32_t dst[16];
    f(dst, src1, src2);
    ASSERT_EQ(dst[0], 1);
    ASSERT_EQ(dst[1], 1);
    ASSERT_EQ(dst[2], 2);
    ASSERT_EQ(dst[3], 2);
}

TEST(armcg_test_t, TestUmmla)
{
    GTestJitGen cg;
    cg.Ptrue(cg.p0.b);
    cg.Ld1b(cg.z1.b, cg.p0, AdrNoOfs(cg.x1));
    cg.Ld1b(cg.z2.b, cg.p0, AdrNoOfs(cg.x2));
    cg.Eor(cg.z3.d, cg.z3.d, cg.z3.d);
    cg.Ummla(cg.z3.s, cg.z1.b, cg.z2.b);
    cg.St1w(cg.z3.s, cg.p0, AdrNoOfs(cg.x0));
    cg.Ret();
    cg.Ready();

    auto f = cg.GetCode<void (*)(int32_t*, uint8_t*, uint8_t*)>();
    uint8_t src1[64] = {
        1, 0, 0, 0, 0, 0, 0, 0,
        2, 0, 0, 0, 0, 0, 0, 0,
        3, 0, 0, 0, 0, 0, 0, 0,
        4, 0, 0, 0, 0, 0, 0, 0};
    uint8_t src2[64] = {
        1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1};
    int32_t dst[16];
    f(dst, src1, src2);
    ASSERT_EQ(dst[0], 1);
    ASSERT_EQ(dst[1], 1);
    ASSERT_EQ(dst[2], 2);
    ASSERT_EQ(dst[3], 2);
}

TEST(armcg_test_t, TestSmmla)
{
    GTestJitGen cg;
    cg.Ptrue(cg.p0.b);
    cg.Ld1b(cg.z1.b, cg.p0, AdrNoOfs(cg.x1));
    cg.Ld1b(cg.z2.b, cg.p0, AdrNoOfs(cg.x2));
    cg.Eor(cg.z3.d, cg.z3.d, cg.z3.d);
    cg.Smmla(cg.z3.s, cg.z1.b, cg.z2.b);
    cg.St1w(cg.z3.s, cg.p0, AdrNoOfs(cg.x0));
    cg.Ret();
    cg.Ready();

    auto f = cg.GetCode<void (*)(int32_t*, int8_t*, int8_t*)>();
    int8_t src1[64] = {
        1, 0, 0, 0, 0, 0, 0, 0,
        2, 0, 0, 0, 0, 0, 0, 0,
        3, 0, 0, 0, 0, 0, 0, 0,
        4, 0, 0, 0, 0, 0, 0, 0};
    int8_t src2[64] = {
        1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1};
    int32_t dst[16];
    f(dst, src1, src2);
    ASSERT_EQ(dst[0], 1);
    ASSERT_EQ(dst[1], 1);
    ASSERT_EQ(dst[2], 2);
    ASSERT_EQ(dst[3], 2);
}

TEST(armcg_test_t, TestAdrMode)
{
    Adr adr(AdrKind::BASE_ONLY);
    ASSERT_EQ(adr.GetKind(), AdrKind::BASE_ONLY);
}

TEST(armcg_test_t, TestAdrPreImm)
{
    GTestJitGen cg;
    AdrPreImm adr(cg.x0, 4);
    ASSERT_EQ(adr.GetXn().GetIdx(), cg.x0.GetIdx());
    ASSERT_EQ(adr.GetImm(), 4);
}

TEST(armcg_test_t, TestAdrPostImm)
{
    GTestJitGen cg;
    AdrPostImm adr(cg.x0, 8);
    ASSERT_EQ(adr.GetXn().GetIdx(), cg.x0.GetIdx());
    ASSERT_EQ(adr.GetImm(), 8);
}

TEST(armcg_test_t, TestAdrPostReg)
{
    GTestJitGen cg;
    AdrPostReg adr(cg.x0, cg.x1);
    ASSERT_EQ(adr.GetXn().GetIdx(), cg.x0.GetIdx());
    ASSERT_EQ(adr.GetXm().GetIdx(), cg.x1.GetIdx());
}

TEST(armcg_test_t, TestAdrNoOfs)
{
    GTestJitGen cg;
    AdrNoOfs adr(cg.x0);
    ASSERT_EQ(adr.GetXn().GetIdx(), cg.x0.GetIdx());
}

TEST(armcg_test_t, TestAdrImm)
{
    GTestJitGen cg;
    AdrImm adr1(cg.x0, 8);
    AdrImm adr2(AdrNoOfs(cg.x0));
    ASSERT_EQ(adr1.GetImm(), 8);
    ASSERT_EQ(adr1.GetXn().GetIdx(), adr2.GetXn().GetIdx());
}

TEST(armcg_test_t, TestAdrUimm)
{
    GTestJitGen cg;
    AdrUimm adr1(cg.x0, 8);
    AdrUimm adr2(AdrNoOfs(cg.x0));
    AdrUimm adr3(AdrImm(cg.x0, 8));
    
    ASSERT_EQ(adr1.GetImm(), adr3.GetImm());
    ASSERT_EQ(adr1.GetXn().GetIdx(), adr2.GetXn().GetIdx());
}

TEST(armcg_test_t, TestAdrReg)
{
    GTestJitGen cg;
    AdrReg adr1(cg.x0, cg.x1, ShMod::LSR, 2);
    AdrReg adr2(cg.x0, cg.x1);

    ASSERT_EQ(adr1.GetXn().GetIdx(), adr2.GetXn().GetIdx());
    ASSERT_EQ(adr1.GetXm().GetIdx(), adr2.GetXm().GetIdx());
    ASSERT_EQ(adr1.GetMod(), ShMod::LSR);
    ASSERT_EQ(adr2.GetMod(), ShMod::LSL);
    ASSERT_EQ(adr1.GetSh(), 2);
    ASSERT_EQ(adr2.GetSh(), 0);
    ASSERT_EQ(adr1.GetInitSh(), true);
    ASSERT_EQ(adr2.GetInitSh(), false);
}

TEST(armcg_test_t, TestAdrExt)
{
    GTestJitGen cg;
    AdrExt adr1(cg.x0, cg.x1, ExtMod::SXTW, 2);
    AdrExt adr2(cg.x0, cg.x1, ExtMod::UXTW);

    ASSERT_EQ(adr1.GetXn().GetIdx(), adr2.GetXn().GetIdx());
    ASSERT_EQ(adr1.GetRm().GetIdx(), adr2.GetRm().GetIdx());
    ASSERT_EQ(adr1.GetMod(), ExtMod::SXTW);
    ASSERT_EQ(adr2.GetMod(), ExtMod::UXTW);
    ASSERT_EQ(adr1.GetSh(), 2);
    ASSERT_EQ(adr2.GetSh(), 0);
    ASSERT_EQ(adr1.GetInitSh(), true);
    ASSERT_EQ(adr2.GetInitSh(), false);
}

TEST(armcg_test_t, TestAdrScSc)
{
    GTestJitGen cg;
    AdrScSc adr1(cg.x0, cg.x1, ShMod::LSR, 2);
    AdrScSc adr2(cg.x0, cg.x1);

    ASSERT_EQ(adr1.GetXn().GetIdx(), adr2.GetXn().GetIdx());
    ASSERT_EQ(adr1.GetMod(), ShMod::LSR);
    ASSERT_EQ(adr2.GetMod(), ShMod::LSL);
    ASSERT_EQ(adr1.GetSh(), 2);
    ASSERT_EQ(adr2.GetSh(), 0);
    ASSERT_EQ(adr1.GetInitMod(), true);
    ASSERT_EQ(adr2.GetInitMod(), false);
}

TEST(armcg_test_t, TestAdrScImm)
{
    GTestJitGen cg;
    AdrScImm adr1(cg.x0, 8);
    AdrScImm adr2(AdrImm(cg.x0, 8));

    ASSERT_EQ(adr1.GetXn().GetIdx(), adr2.GetXn().GetIdx());
    ASSERT_EQ(adr1.GetMod(), ExtMod::MUL_VL);
    ASSERT_EQ(adr1.GetSimm(), adr2.GetSimm());
}

TEST(armcg_test_t, TestAdrSc64U)
{
    GTestJitGen cg;
    AdrSc64U adr1(cg.x0, cg.z0.d);

    ASSERT_EQ(adr1.GetXn().GetIdx(), cg.x0.GetIdx());
    ASSERT_EQ(adr1.GetZm().GetIdx(), cg.z0.GetIdx());
}

TEST(armcg_test_t, TestAdrSc64S)
{
    GTestJitGen cg;
    AdrSc64S adr1(cg.x0, cg.z0.d, ShMod::LSR, 2);
    AdrSc64S adr2(AdrSc64U (cg.x0, cg.z0.d));

    ASSERT_EQ(adr1.GetXn().GetIdx(), adr2.GetXn().GetIdx());
    ASSERT_EQ(adr1.GetZm().GetIdx(), adr2.GetZm().GetIdx());
    ASSERT_EQ(adr1.GetMod(), ShMod::LSR);
    ASSERT_EQ(adr2.GetMod(), ShMod::LSL);
    ASSERT_EQ(adr1.GetSh(), 2);
}

TEST(armcg_test_t, TestAdrVecSc64)
{
    GTestJitGen cg;
    AdrVecSc64 adr1(cg.z0.d, cg.x0);

    ASSERT_EQ(adr1.GetZn().GetIdx(), cg.z0.GetIdx());
    ASSERT_EQ(adr1.GetXm().GetIdx(), cg.x0.GetIdx());
}

TEST(armcg_test_t, TestAdrSc32U)
{
    GTestJitGen cg;
    AdrSc32U adr1(cg.x0, cg.z0.s, ExtMod::MUL_VL);

    ASSERT_EQ(adr1.GetXn().GetIdx(), cg.x0.GetIdx());
    ASSERT_EQ(adr1.GetZm().GetIdx(), cg.z0.GetIdx());
    ASSERT_EQ(adr1.GetMod(), ExtMod::MUL_VL);
}

TEST(armcg_test_t, TestAdrSc32S)
{
    GTestJitGen cg;
    AdrSc32S adr1(cg.x0, cg.z0.s, ExtMod::UXTW, 2);
    AdrSc32S adr2(AdrSc32U(cg.x0, cg.z0.s, ExtMod::MUL_VL));

    ASSERT_EQ(adr1.GetXn().GetIdx(), adr2.GetXn().GetIdx());
    ASSERT_EQ(adr1.GetZm().GetIdx(), adr2.GetZm().GetIdx());
    ASSERT_EQ(adr1.GetMod(), ExtMod::UXTW);
    ASSERT_EQ(adr2.GetMod(), ExtMod::MUL_VL);
    ASSERT_EQ(adr1.GetSh(), 2);
}

TEST(armcg_test_t, TestAdrVecSc32)
{
    GTestJitGen cg;
    AdrVecSc32 adr1(cg.z0.s, cg.x0);

    ASSERT_EQ(adr1.GetZn().GetIdx(), cg.z0.GetIdx());
    ASSERT_EQ(adr1.GetXm().GetIdx(), cg.x0.GetIdx());
}

TEST(armcg_test_t, TestAdrSc32UU)
{
    GTestJitGen cg;
    AdrSc32UU adr1(cg.x0, cg.z0.d, ExtMod::MUL_VL);

    ASSERT_EQ(adr1.GetXn().GetIdx(), cg.x0.GetIdx());
    ASSERT_EQ(adr1.GetZm().GetIdx(), cg.z0.GetIdx());
    ASSERT_EQ(adr1.GetMod(), ExtMod::MUL_VL);
}

TEST(armcg_test_t, TestAdrSc32US)
{
    GTestJitGen cg;
    AdrSc32US adr1(cg.x0, cg.z0.d, ExtMod::UXTW, 2);
    AdrSc32US adr2(AdrSc32UU(cg.x0, cg.z0.d, ExtMod::MUL_VL));

    ASSERT_EQ(adr1.GetXn().GetIdx(), adr2.GetXn().GetIdx());
    ASSERT_EQ(adr1.GetZm().GetIdx(), adr2.GetZm().GetIdx());
    ASSERT_EQ(adr1.GetMod(), ExtMod::UXTW);
    ASSERT_EQ(adr2.GetMod(), ExtMod::MUL_VL);
    ASSERT_EQ(adr1.GetSh(), 2);
}

TEST(armcg_test_t, TestAdrVecImm64)
{
    GTestJitGen cg;
    AdrVecImm64 adr1(cg.z0.d, 4);

    ASSERT_EQ(adr1.GetZn().GetIdx(), cg.z0.GetIdx());
    ASSERT_EQ(adr1.GetImm(), 4);
}

TEST(armcg_test_t, TestAdrVecImm32)
{
    GTestJitGen cg;
    AdrVecImm32 adr1(cg.z0.s, 4);

    ASSERT_EQ(adr1.GetZn().GetIdx(), cg.z0.GetIdx());
    ASSERT_EQ(adr1.GetImm(), 4);
}

TEST(armcg_test_t, TestAdrVec)
{
    GTestJitGen cg;
    AdrVec adr1(cg.z0, cg.z1);

    ASSERT_EQ(adr1.GetZn().GetIdx(), cg.z0.GetIdx());
    ASSERT_EQ(adr1.GetZm().GetIdx(), cg.z1.GetIdx());
    ASSERT_EQ(adr1.GetMod(), ShMod::NONE);
    ASSERT_EQ(adr1.GetSh(), 0);
}

TEST(armcg_test_t, TestAdrVecU)
{
    GTestJitGen cg;
    AdrVecU adr1(cg.z0, cg.z1, ExtMod::MUL_VL);

    ASSERT_EQ(adr1.GetZn().GetIdx(), cg.z0.GetIdx());
    ASSERT_EQ(adr1.GetZm().GetIdx(), cg.z1.GetIdx());
    ASSERT_EQ(adr1.GetMod(), ExtMod::MUL_VL);
    ASSERT_EQ(adr1.GetSh(), 0);
}

TEST(armcg_test_t, CpuInfo)
{
    ARMCG::GetCpu().GetCacheType(ARMCG::Util::Arm64CacheLevel::L1);
    int r1 = ARMCG::GetCpu().GetNumCores(ARMCG::Util::Arm64CpuTopologyLevel::SMT);
    int r2 = ARMCG::GetCpu().GetNumCores(ARMCG::Util::Arm64CpuTopologyLevel::CORE);
    EXPECT_EQ(r1, r2);
    ARMCG::GetCpu().GetLastDataCacheLevel();
    GTestJitGen *cg = new GTestJitGen;
    delete cg;
    ARMCG::GetCpu().DumpCacheInfo();
    ARMCG::Util::CpuInfo info;
    int ret1 = info.GetCacheSize(ARMCG::Util::CacheType::ICACHE, 6);
    EXPECT_EQ(ret1, 0);
    int ret2 = info.GetCacheSize(ARMCG::Util::CacheType::DCACHE, 6);
    EXPECT_EQ(ret2, 0);
    int ret3 = info.GetCacheSize(ARMCG::Util::CacheType::UCACHE, 6);
    EXPECT_EQ(ret3, 0);
    int ret5 = info.GetCodeCacheSize(5);
    EXPECT_EQ(ret5, 0);
    int ret8 = info.GetLastDataCacheLevel();
    EXPECT_EQ(ret8, 0);
    int ret9 = info.GetNumCores(ARMCG::Util::Arm64CpuTopologyLevel::SMT);
    EXPECT_EQ(ret9, 0);
    int ret10 = info.GetNumCores(ARMCG::Util::Arm64CpuTopologyLevel::CORE);
    EXPECT_EQ(ret10, 0);
    int ret12 = info.GetUnifiedCacheSize(1);
    EXPECT_EQ(ret12, 0);
    info.Put();
}

TEST(armcg_test_t, AlignEB)
{
    ARMCG::CodeGenerator cg;
    cg.Align(4);        // nothing inserted
    ASSERT_ANY_THROW(cg.Align(1));
    ASSERT_ANY_THROW(cg.Align(7));
    cg.Align(8);        // nop inserted
    EXPECT_EQ(cg.GetSize(), 8);
}

TEST(armcg_test_t, JitDump)
{
    GTestJitGen *jg = new GTestJitGen;
    putenv(const_cast<char *>("ONEDNN_JIT_DUMP=1"));
    ASSERT_NE(jg->GetCode<int (*)(int, int)>(), nullptr);
    jg->JitKer();
    putenv(const_cast<char *>("ONEDNN_JIT_DUMP=0"));
    ASSERT_NO_THROW(delete jg);
    CCompatible *p = new CCompatible;
    delete(p);
}

TEST(armcg_test_t, TestResetDump)
{
    GTestJitGen cg;
    cg.Mov(cg.w0, 9999);
    cg.Ret();
    cg.Ready();
    cg.Dump();
    cg.Reset();

    ASSERT_EQ(cg.GetSize(), 0);
}

// test cg_memory.hpp Coverage
TEST(armcg_test_t, TestCov_cg_memory_hpp)
{
    int *p = (int *)AlignedMalloc(1, 128);
    AlignedFree(p);

    MmapAllocator *ma = new MmapAllocator();
    uint32_t *uma = ma->Alloc(1);
    ma->Free(uma);

    MmapAllocator ma1;
    uint32_t *uma1 = nullptr;
    uint32_t *uma2 = ma->Alloc(128);
    
    EXPECT_ANY_THROW(ma1.Alloc(0));
    EXPECT_ANY_THROW(ma1.Free(uma));
    ma1.Free(uma1);
    EXPECT_ANY_THROW(ma1.Free(uma2));
    ma->Free(uma2);
    delete ma;

    CodeArray *ca2 = new CodeArray(128, DONT_SET_PROTECT_RWE);
    ca2->SetSize(128);
    EXPECT_ANY_THROW(ca2->WriteData(32));
    EXPECT_ANY_THROW(ca2->SetSize(512));
    delete ca2;

    CodeArray ca(128, AUTO_GROW);
    ca.SetSize(128);
    ca.WriteData(32);
    ca.Dump();
    ca.ResetSize();
    ca.Rewrite(9, 64);
    ca.ClearCodeArray();

    CodeArray ca3(256);
    ca3.IsAllocType();
    ca3.SetProtectMode(CodeArray::ProtectMode::RWE);
    ca3.SetProtectModeRE();
    ca3.SetProtectModeRW();
    ca3.GetCode();
    ca3.GetCurr();
    ca3.GetSize();
    ca3.GetMaxSize(); 
    ca3.IsAutoGrow();
    ca3.IsCalledCalcJmpAddress();
    void *addr = nullptr;
    CodeArray::GetAlignedAddress((uint32_t *)addr);
    CodeArray::Protect(addr, 0, CodeArray::ProtectMode::RW);

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
    SubCodeArray *sca = new SubCodeArray();
    delete sca;
}

// test cg_mnemonic_tpl.hpp Coverage
TEST(armcg_test_t, TestCov_cg_mnemonic_tpl_hpp)
{
    int size = 3;
    uint32_t uArray[] = {3, 4096, 0};
    int iArray[] = {3, -3, 4096};
    for(int c = 0; c < size; ++c) {
        uint32_t u = uArray[c];
        int i = iArray[c];

        GTestJitGen cg;
        cg.AddImm(cg.x0, cg.x1, i, cg.x2);
        cg.AddImm<unsigned long>(cg.x0, cg.x1, u, cg.x2);
        cg.AddImm(cg.x0, cg.x1, u, cg.x2);
        cg.AddImm<long>(cg.x0, cg.x1, i, cg.x2);

        cg.SubImm(cg.x0, cg.x1, u, cg.x2);
        cg.SubImm<long>(cg.x0, cg.x1, i, cg.x2);
        cg.SubImm(cg.x0, cg.x1, i, cg.x2);
        cg.SubImm<unsigned long>(cg.x0, cg.x1, u, cg.x2);

        cg.AddsImm(cg.x0, cg.x1, u, cg.x2);
        cg.AddsImm(cg.x0, cg.x1, i, cg.x2);

        cg.SubsImm(cg.x0, cg.x1, u, cg.x2);
        cg.SubsImm(cg.x0, cg.x1, i, cg.x2);

        cg.CmpImm(cg.x0, i, cg.x1);
        cg.Ret();
        cg.Ready();
        auto Func = cg.GetCode<int (*)(int, int)>();
        Func(0, 4);
    }
    int size1 = 5;
    int iArray1[] = {3, 0, -1, 99958, -998178817};
    uint32_t uArray1[] = {3, 0, 0xFFFFFFFF, 0, 0};
    for(int c = 0; c < size1; ++c) {
        int i = iArray1[c];
        uint32_t u = uArray1[c];
        GTestJitGen cg1;
        cg1.MovImm(cg1.x0, i);
        cg1.MovImm<long>(cg1.x0, i);
        cg1.MovImm<unsigned long>(cg1.x0, i);
        cg1.Ret();
        cg1.Ready();
        auto Func1 = cg1.GetCode<int (*)(int)>();
        Func1(4);
        GTestJitGen cg2;
        cg2.MovImm(cg2.w0, u);
        cg2.Ret();
        cg2.Ready();
        auto Func2 = cg2.GetCode<int (*)(int)>();
        Func2(4);
        GTestJitGen cg3;
        cg3.MovImm(cg3.w0, i);
        cg3.Ret();
        cg3.Ready();
        auto Func3 = cg3.GetCode<int (*)(int)>();
        Func3(4);
    }
}
}
