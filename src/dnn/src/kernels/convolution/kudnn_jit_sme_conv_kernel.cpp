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
#include "kudnn_jit_sme_conv_kernel.hpp"
#include "kudnn_conv_const.hpp"

#ifdef KUDNNL_920F
namespace KuDNN {

using namespace ARMCG;

void JitConvMCKernel::BiasLoad()
{
    Fdup(z6.s, 1.0);
    Ld1w(z28.s, p5 / PredType::T_Z, Ptr(regCurBia));
    Addvl(regCurBia, regCurBia, 1);
    Fmopa(za0.s, p0 / PredType::T_M, p5 / PredType::T_M, z6.s, z28.s);
    Fmopa(za1.s, p0 / PredType::T_M, p5 / PredType::T_M, z6.s, z28.s);
    Fmopa(za2.s, p0 / PredType::T_M, p5 / PredType::T_M, z6.s, z28.s);
    Fmopa(za3.s, p0 / PredType::T_M, p5 / PredType::T_M, z6.s, z28.s);
}

// p1-p4 used for matrix M, p5 used for matrix N
void JitConvMCKernel::SetMatrixPred(Reg64T &mReg, Reg64T &nReg, Reg64T &mCurReg, Reg64T &nCurReg)
{
    Whilelt(p1.s, mCurReg, mReg);
    Incw(mCurReg);
    Whilelt(p2.s, mCurReg, mReg);
    Incw(mCurReg);
    Whilelt(p3.s, mCurReg, mReg);
    Incw(mCurReg);
    Whilelt(p4.s, mCurReg, mReg);
    Whilelt(p5.s, nCurReg, nReg);
}

// accumulate 4 block along k direction
void JitConvMCKernel::KloopFmopaKernel(Reg64T &aStride)
{
    Subs(regCurK, regCurK, 1);
    Fmopa(za0.s, p1 / PredType::T_M, p5 / PredType::T_M, z8.s, z28.s);
    Fmopa(za1.s, p2 / PredType::T_M, p5 / PredType::T_M, z9.s, z28.s);
    Fmopa(za2.s, p3 / PredType::T_M, p5 / PredType::T_M, z10.s, z28.s);
    Fmopa(za3.s, p4 / PredType::T_M, p5 / PredType::T_M, z11.s, z28.s);
    Ld1w(z28.s, p5 / PredType::T_Z, Ptr(x21));
    Ld1w(z8.s, p1 / PredType::T_Z, Ptr(regCurA));
    Ld1w(z9.s, p2 / PredType::T_Z, Ptr(regCurA, GEMM_OFFSET1, ExtMod::MUL_VL));
    Ld1w(z10.s, p3 / PredType::T_Z, Ptr(regCurA, GEMM_OFFSET2, ExtMod::MUL_VL));
    Ld1w(z11.s, p4 / PredType::T_Z, Ptr(regCurA, GEMM_OFFSET3, ExtMod::MUL_VL));
    Add(regCurA, regCurA, aStride, ShMod::LSL, FLOAT_OFFSET_SHIFT);
    Fmopa(za0.s, p1 / PredType::T_M, p5 / PredType::T_M, z12.s, z22.s);
    Fmopa(za1.s, p2 / PredType::T_M, p5 / PredType::T_M, z13.s, z22.s);
    Fmopa(za2.s, p3 / PredType::T_M, p5 / PredType::T_M, z14.s, z22.s);
    Fmopa(za3.s, p4 / PredType::T_M, p5 / PredType::T_M, z15.s, z22.s);
    Ld1w(z22.s, p5 / PredType::T_Z, Ptr(x21, GEMM_OFFSET1, ExtMod::MUL_VL));
    Ld1w(z12.s, p1 / PredType::T_Z, Ptr(regCurA));
    Ld1w(z13.s, p2 / PredType::T_Z, Ptr(regCurA, GEMM_OFFSET1, ExtMod::MUL_VL));
    Ld1w(z14.s, p3 / PredType::T_Z, Ptr(regCurA, GEMM_OFFSET2, ExtMod::MUL_VL));
    Ld1w(z15.s, p4 / PredType::T_Z, Ptr(regCurA, GEMM_OFFSET3, ExtMod::MUL_VL));
    Add(regCurA, regCurA, aStride, ShMod::LSL, FLOAT_OFFSET_SHIFT);
    Fmopa(za0.s, p1 / PredType::T_M, p5 / PredType::T_M, z4.s, z30.s);
    Fmopa(za1.s, p2 / PredType::T_M, p5 / PredType::T_M, z5.s, z30.s);
    Fmopa(za2.s, p3 / PredType::T_M, p5 / PredType::T_M, z6.s, z30.s);
    Fmopa(za3.s, p4 / PredType::T_M, p5 / PredType::T_M, z7.s, z30.s);
    Ld1w(z30.s, p5 / PredType::T_Z, Ptr(x21, GEMM_OFFSET2, ExtMod::MUL_VL));
    Ld1w(z4.s, p1 / PredType::T_Z, Ptr(regCurA));
    Ld1w(z5.s, p2 / PredType::T_Z, Ptr(regCurA, GEMM_OFFSET1, ExtMod::MUL_VL));
    Ld1w(z6.s, p3 / PredType::T_Z, Ptr(regCurA, GEMM_OFFSET2, ExtMod::MUL_VL));
    Ld1w(z7.s, p4 / PredType::T_Z, Ptr(regCurA, GEMM_OFFSET3, ExtMod::MUL_VL));
    Add(regCurA, regCurA, aStride, ShMod::LSL, FLOAT_OFFSET_SHIFT);
    Fmopa(za0.s, p1 / PredType::T_M, p5 / PredType::T_M, z16.s, z20.s);
    Fmopa(za1.s, p2 / PredType::T_M, p5 / PredType::T_M, z17.s, z20.s);
    Fmopa(za2.s, p3 / PredType::T_M, p5 / PredType::T_M, z18.s, z20.s);
    Fmopa(za3.s, p4 / PredType::T_M, p5 / PredType::T_M, z19.s, z20.s);
    Ld1w(z20.s, p5 / PredType::T_Z, Ptr(x21, GEMM_OFFSET3, ExtMod::MUL_VL));
    Ld1w(z16.s, p1 / PredType::T_Z, Ptr(regCurA));
    Ld1w(z17.s, p2 / PredType::T_Z, Ptr(regCurA, GEMM_OFFSET1, ExtMod::MUL_VL));
    Ld1w(z18.s, p3 / PredType::T_Z, Ptr(regCurA, GEMM_OFFSET2, ExtMod::MUL_VL));
    Ld1w(z19.s, p4 / PredType::T_Z, Ptr(regCurA, GEMM_OFFSET3, ExtMod::MUL_VL));
    Add(regCurA, regCurA, aStride, ShMod::LSL, FLOAT_OFFSET_SHIFT);
    Addvl(x21, x21, GEMM_OFFSET4);
}

void JitConvMCKernel::KloopFmopaLast()
{
    Fmopa(za0.s, p1 / PredType::T_M, p5 / PredType::T_M, z8.s, z28.s);
    Fmopa(za1.s, p2 / PredType::T_M, p5 / PredType::T_M, z9.s, z28.s);
    Fmopa(za2.s, p3 / PredType::T_M, p5 / PredType::T_M, z10.s, z28.s);
    Fmopa(za3.s, p4 / PredType::T_M, p5 / PredType::T_M, z11.s, z28.s);
    Fmopa(za0.s, p1 / PredType::T_M, p5 / PredType::T_M, z12.s, z22.s);
    Fmopa(za1.s, p2 / PredType::T_M, p5 / PredType::T_M, z13.s, z22.s);
    Fmopa(za2.s, p3 / PredType::T_M, p5 / PredType::T_M, z14.s, z22.s);
    Fmopa(za3.s, p4 / PredType::T_M, p5 / PredType::T_M, z15.s, z22.s);
    Fmopa(za0.s, p1 / PredType::T_M, p5 / PredType::T_M, z4.s, z30.s);
    Fmopa(za1.s, p2 / PredType::T_M, p5 / PredType::T_M, z5.s, z30.s);
    Fmopa(za2.s, p3 / PredType::T_M, p5 / PredType::T_M, z6.s, z30.s);
    Fmopa(za3.s, p4 / PredType::T_M, p5 / PredType::T_M, z7.s, z30.s);
    Fmopa(za0.s, p1 / PredType::T_M, p5 / PredType::T_M, z16.s, z20.s);
    Fmopa(za1.s, p2 / PredType::T_M, p5 / PredType::T_M, z17.s, z20.s);
    Fmopa(za2.s, p3 / PredType::T_M, p5 / PredType::T_M, z18.s, z20.s);
    Fmopa(za3.s, p4 / PredType::T_M, p5 / PredType::T_M, z19.s, z20.s);
}

void JitConvMCKernel::KloopFmopaTail(Reg64T &aStride)
{
    Ld1w(z8.s, p5 / PredType::T_Z, Ptr(x21));
    Addvl(x21, x21, GEMM_OFFSET1);
    Subs(regCurkTail, regCurkTail, 1);
    Ld1w(z4.s, p1 / PredType::T_Z, Ptr(regCurA));
    Ld1w(z5.s, p2 / PredType::T_Z, Ptr(regCurA, GEMM_OFFSET1, ExtMod::MUL_VL));
    Ld1w(z6.s, p3 / PredType::T_Z, Ptr(regCurA, GEMM_OFFSET2, ExtMod::MUL_VL));
    Ld1w(z7.s, p4 / PredType::T_Z, Ptr(regCurA, GEMM_OFFSET3, ExtMod::MUL_VL));
    Add(regCurA, regCurA, aStride, ShMod::LSL, FLOAT_OFFSET_SHIFT);
    Fmopa(za0.s, p1 / PredType::T_M, p5 / PredType::T_M, z4.s, z8.s);
    Fmopa(za1.s, p2 / PredType::T_M, p5 / PredType::T_M, z5.s, z8.s);
    Fmopa(za2.s, p3 / PredType::T_M, p5 / PredType::T_M, z6.s, z8.s);
    Fmopa(za3.s, p4 / PredType::T_M, p5 / PredType::T_M, z7.s, z8.s);
}

void JitConvMCKernel::KfirstLoad(Reg64T &aStride)
{
    Ld1w(z28.s, p5 / PredType::T_Z, Ptr(x21));
    Ld1w(z22.s, p5 / PredType::T_Z, Ptr(x21, GEMM_OFFSET1, ExtMod::MUL_VL));
    Ld1w(z30.s, p5 / PredType::T_Z, Ptr(x21, GEMM_OFFSET2, ExtMod::MUL_VL));
    Ld1w(z20.s, p5 / PredType::T_Z, Ptr(x21, GEMM_OFFSET3, ExtMod::MUL_VL));
    Addvl(x21, x21, GEMM_OFFSET4);
    Ld1w(z8.s, p1 / PredType::T_Z, Ptr(regCurA));
    Ld1w(z9.s, p2 / PredType::T_Z, Ptr(regCurA, GEMM_OFFSET1, ExtMod::MUL_VL));
    Ld1w(z10.s, p3 / PredType::T_Z, Ptr(regCurA, GEMM_OFFSET2, ExtMod::MUL_VL));
    Ld1w(z11.s, p4 / PredType::T_Z, Ptr(regCurA, GEMM_OFFSET3, ExtMod::MUL_VL));
    Add(regCurA, regCurA, aStride, ShMod::LSL, FLOAT_OFFSET_SHIFT);
    Ld1w(z12.s, p1 / PredType::T_Z, Ptr(regCurA));
    Ld1w(z13.s, p2 / PredType::T_Z, Ptr(regCurA, GEMM_OFFSET1, ExtMod::MUL_VL));
    Ld1w(z14.s, p3 / PredType::T_Z, Ptr(regCurA, GEMM_OFFSET2, ExtMod::MUL_VL));
    Ld1w(z15.s, p4 / PredType::T_Z, Ptr(regCurA, GEMM_OFFSET3, ExtMod::MUL_VL));
    Add(regCurA, regCurA, aStride, ShMod::LSL, FLOAT_OFFSET_SHIFT);
    Ld1w(z4.s, p1 / PredType::T_Z, Ptr(regCurA));
    Ld1w(z5.s, p2 / PredType::T_Z, Ptr(regCurA, GEMM_OFFSET1, ExtMod::MUL_VL));
    Ld1w(z6.s, p3 / PredType::T_Z, Ptr(regCurA, GEMM_OFFSET2, ExtMod::MUL_VL));
    Ld1w(z7.s, p4 / PredType::T_Z, Ptr(regCurA, GEMM_OFFSET3, ExtMod::MUL_VL));
    Add(regCurA, regCurA, aStride, ShMod::LSL, FLOAT_OFFSET_SHIFT);
    Ld1w(z16.s, p1 / PredType::T_Z, Ptr(regCurA));
    Ld1w(z17.s, p2 / PredType::T_Z, Ptr(regCurA, GEMM_OFFSET1, ExtMod::MUL_VL));
    Ld1w(z18.s, p3 / PredType::T_Z, Ptr(regCurA, GEMM_OFFSET2, ExtMod::MUL_VL));
    Ld1w(z19.s, p4 / PredType::T_Z, Ptr(regCurA, GEMM_OFFSET3, ExtMod::MUL_VL));
    Add(regCurA, regCurA, aStride, ShMod::LSL, FLOAT_OFFSET_SHIFT);
}

void JitConvMCKernel::StoreOutput(Reg64T &cStride)
{
    Label storeLoop;
    Label storeLoopEnd;
    Mov(regCurOut, regOut);
    Mov(x11, 0);
    Add(x11, x11, cStride, ShMod::LSL, FLOAT_OFFSET_SHIFT);
    Madd(regCurOut, regCurN, x11, regCurOut);
    Mov(x15, regCurN);
    Mov(w12, 0);
    Cntw(x20);

    L(storeLoop);
    Mova(z16.s, p1 / PredType::T_M, za0v.s, w12, 0);
    Mova(z17.s, p2 / PredType::T_M, za1v.s, w12, 0);
    Mova(z18.s, p3 / PredType::T_M, za2v.s, w12, 0);
    Mova(z19.s, p4 / PredType::T_M, za3v.s, w12, 0);
    St1w(z16.s, p1, Ptr(regCurOut));
    St1w(z17.s, p2, Ptr(regCurOut, GEMM_OFFSET1, ExtMod::MUL_VL));
    St1w(z18.s, p3, Ptr(regCurOut, GEMM_OFFSET2, ExtMod::MUL_VL));
    St1w(z19.s, p4, Ptr(regCurOut, GEMM_OFFSET3, ExtMod::MUL_VL));
    Add(x15, x15, GEMM_OFFSET1);
    Cmp(x15, regN);
    Bge(storeLoopEnd);
    Add(regCurOut, regCurOut, cStride, ShMod::LSL, FLOAT_OFFSET_SHIFT);
    Mova(z24.s, p1 / PredType::T_M, za0v.s, w12, GEMM_OFFSET1);
    Mova(z25.s, p2 / PredType::T_M, za1v.s, w12, GEMM_OFFSET1);
    Mova(z26.s, p3 / PredType::T_M, za2v.s, w12, GEMM_OFFSET1);
    Mova(z27.s, p4 / PredType::T_M, za3v.s, w12, GEMM_OFFSET1);
    St1w(z24.s, p1, Ptr(regCurOut));
    St1w(z25.s, p2, Ptr(regCurOut, GEMM_OFFSET1, ExtMod::MUL_VL));
    St1w(z26.s, p3, Ptr(regCurOut, GEMM_OFFSET2, ExtMod::MUL_VL));
    St1w(z27.s, p4, Ptr(regCurOut, GEMM_OFFSET3, ExtMod::MUL_VL));
    Add(x15, x15, GEMM_OFFSET1);
    Cmp(x15, regN);
    Bge(storeLoopEnd);
    Add(regCurOut, regCurOut, cStride, ShMod::LSL, FLOAT_OFFSET_SHIFT);
    Mova(z8.s, p1 / PredType::T_M, za0v.s, w12, GEMM_OFFSET2);
    Mova(z9.s, p2 / PredType::T_M, za1v.s, w12, GEMM_OFFSET2);
    Mova(z10.s, p3 / PredType::T_M, za2v.s, w12, GEMM_OFFSET2);
    Mova(z11.s, p4 / PredType::T_M, za3v.s, w12, GEMM_OFFSET2);
    St1w(z8.s, p1, Ptr(regCurOut));
    St1w(z9.s, p2, Ptr(regCurOut, GEMM_OFFSET1, ExtMod::MUL_VL));
    St1w(z10.s, p3, Ptr(regCurOut, GEMM_OFFSET2, ExtMod::MUL_VL));
    St1w(z11.s, p4, Ptr(regCurOut, GEMM_OFFSET3, ExtMod::MUL_VL));
    Add(x15, x15, GEMM_OFFSET1);
    Cmp(x15, regN);
    Bge(storeLoopEnd);
    Add(regCurOut, regCurOut, cStride, ShMod::LSL, FLOAT_OFFSET_SHIFT);
    Mova(z4.s, p1 / PredType::T_M, za0v.s, w12, GEMM_OFFSET3);
    Mova(z5.s, p2 / PredType::T_M, za1v.s, w12, GEMM_OFFSET3);
    Mova(z6.s, p3 / PredType::T_M, za2v.s, w12, GEMM_OFFSET3);
    Mova(z7.s, p4 / PredType::T_M, za3v.s, w12, GEMM_OFFSET3);
    St1w(z4.s, p1, Ptr(regCurOut));
    St1w(z5.s, p2, Ptr(regCurOut, GEMM_OFFSET1, ExtMod::MUL_VL));
    St1w(z6.s, p3, Ptr(regCurOut, GEMM_OFFSET2, ExtMod::MUL_VL));
    St1w(z7.s, p4, Ptr(regCurOut, GEMM_OFFSET3, ExtMod::MUL_VL));
    Add(x15, x15, GEMM_OFFSET1);
    Cmp(x15, regN);
    Bge(storeLoopEnd);
    Add(regCurOut, regCurOut, cStride, ShMod::LSL, FLOAT_OFFSET_SHIFT);
    Add(x12, x12, GEMM_OFFSET4);
    Cmp(x12, x20);
    Blt(storeLoop);

    L(storeLoopEnd);
}

void JitConvMCKernel::GemmKernelGenerate()
{
    Label nLoop;
    Label kLoop;
    Label kLoopTail;
    Label kLoopEnd;
    Label lastK;
    Label lastkEnd;
    Mov(regCurM, 0);
    Mov(regCurN, 0);
    Mov(regCurBia, regBia);
    Mov(x27, jcp.od * jcp.oh * jcp.ow);

    L(nLoop);
    Mov(regTmp, regCurM);
    SetMatrixPred(regM, regN, regTmp, regCurN);
    Zero({za0, za1, za2, za3});
    Mov(regCurA, regInp);
    if (jcp.withBias) {
        BiasLoad();
    }

    Mov(regTmp, regK);
    Lsr(regCurK, regTmp, GEMM_OFFSET2);
    And(regCurkTail, regTmp, GEMM_K_MASK);
    Mov(regTmp, 0);
    Add(regTmp, regTmp, regK, ShMod::LSL, FLOAT_OFFSET_SHIFT);
    Madd(x21, regCurN, regTmp, regWei);
    Cbz(regCurK, lastkEnd);
    Subs(regCurK, regCurK, 1);
    KfirstLoad(regM);
    Ble(lastK);

    L(kLoop);
    KloopFmopaKernel(regM);
    Bgt(kLoop);

    L(lastK);
    KloopFmopaLast();

    L(lastkEnd);
    Cbz(regCurkTail, kLoopEnd);

    L(kLoopTail);
    KloopFmopaTail(regM);
    Bgt(kLoopTail);

    L(kLoopEnd);
    StoreOutput(x27);
    Incw(regCurN);
    Cmp(regCurN, regN);
    Blt(nLoop);
    Incw(regCurM, Pattern::ALL, ExtMod::MUL, GEMM_OFFSET4);
    Cmp(regCurM, regM);
    Mov(regCurN, 0);
    Addvl(regInp, regInp, GEMM_OFFSET4);
    Addvl(regOut, regOut, GEMM_OFFSET4);
    Mov(regCurBia, regBia);
    Blt(nLoop);
}

void JitConvMCKernel::DirectKernelPrepare(const IntType& urW)
{
    size_t bStride = jcp.kd * jcp.kh * jcp.kw * jcp.ic;
    Mov(x28, urW);
    Mov(regTmp, 0);
    SetMatrixPred(x28, regN, regTmp, regCurN);
    Zero({za0, za1, za2, za3});
    if (jcp.withBias) {
        BiasLoad();
    }

    Mov(x13, jcp.kd);
    Mov(x18, jcp.kh);
    Mov(x29, regInp);                                          // CurA
    Mov(x14, 0);
    Mov(x12, bStride);
    Add(x14, x14, x12, ShMod::LSL, FLOAT_OFFSET_SHIFT);        // bStride * sizeof(float)
    Madd(x16, x10, x14, x3);                                   // CurB
}

void JitConvMCKernel::ComputeLoop(const IntType& urW)
{
    Label nLoop;
    Label kdLabel;
    Label khLabel;
    Label kwLabel;
    Label l2;
    Label l3;
    Label l4;
    Label l5;
    Label l6;
    IntType sizeW = jcp.iw + jcp.lPad + (jcp.rPad > 0 ? jcp.rPad : 0);
    IntType sizeH = jcp.ih + jcp.tPad + (jcp.bPad > 0 ? jcp.bPad : 0);

    Mov(regCurN, 0);
    Mov(regCurBia, regBia);
    L(nLoop);
    DirectKernelPrepare(urW);
    Align(REG_INDEX32);
    L(kdLabel);
    {
        Mov(x18, jcp.kh);
        Mov(x28, x29);
        Mov(x30, x16);
        L(khLabel);
        {
            Mov(x19, jcp.kw);
            Mov(x24, x28);
            Mov(x25, x30);
            L(kwLabel);
            {
                Mov(x26, x24);
                Mov(x14, jcp.ic);
                Lsr(x23, x14, GEMM_OFFSET2);
                And(x22, x14, GEMM_K_MASK);
                Mov(x21, x25);
                Cbz(x23, l4);
                Subs(x23, x23, 1);
                KfirstLoad(x8);
                Ble(l3);

                L(l2);
                KloopFmopaKernel(x8);
                Bgt(l2);

                L(l3);
                KloopFmopaLast();

                L(l4);
                Cbz(x22, l6);

                L(l5);
                KloopFmopaTail(x8);
                Bgt(l5);

                L(l6);
                Add(x24, x24, sizeof(float));
                Mov(x14, jcp.ic);
                Add(x25, x25, x14, ShMod::LSL, KERNEL_OFFSET_SHIFT);
                Sub(x19, x19, 1);
                Cmp(x19, 0);
                B(Cond::GT, kwLabel);
            }
            Mov(regTmp, sizeW);
            Add(x28, x28, regTmp, ShMod::LSL, FLOAT_OFFSET_SHIFT);  // input shift
            Mov(regTmp, jcp.kw * jcp.ic);
            Add(x30, x30, regTmp, ShMod::LSL, KERNEL_OFFSET_SHIFT); // kernel shift
            Sub(x18, x18, 1);
            Cmp(x18, 0);
            B(Cond::GT, khLabel);
        }
        Mov(regTmp, sizeW * sizeH);
        Add(x29, x29, regTmp, ShMod::LSL, FLOAT_OFFSET_SHIFT);  // input shift
        Mov(regTmp, jcp.kh * jcp.kw * jcp.ic);
        Add(x16, x16, regTmp, ShMod::LSL, KERNEL_OFFSET_SHIFT); // kernel shift
        Sub(x13, x13, 1);
        Cmp(x13, 0);
        B(Cond::GT, kdLabel);
    }
    StoreOutput(regM);
    Incw(regCurN);
    Cmp(regCurN, regN);
    Blt(nLoop);
}

// only support kw <= 3 because of kw loop unroll
void JitConvMCKernel::ComputeLoopWithStride(const IntType& urW)
{
    Label nLoop;
    Label labelKh;
    Label labelKd;
    Label l2[LEVEL3];
    Label l3[LEVEL3];
    Label l4[LEVEL3];
    Label l5[LEVEL3];
    Label l6[LEVEL3];
    IntType sizeW = jcp.iw + jcp.lPad + (jcp.rPad > 0 ? jcp.rPad : 0);
    IntType sizeH = jcp.ih + jcp.tPad + (jcp.bPad > 0 ? jcp.bPad : 0);

    Mov(regCurN, 0);
    Mov(regCurBia, regBia);
    L(nLoop);
    DirectKernelPrepare(urW);
    Align(REG_INDEX32);
    L(labelKd);
    {
        Mov(x18, jcp.kh);
        Mov(x28, x29);
        Mov(x30, x16);
        L(labelKh);
        {
            Mov(x19, jcp.kw);
            Mov(x24, x28);
            Mov(x25, x30);
            for (IntType kw = 0; kw < jcp.kw; kw++) {
                IntType off = 0;
                for (IntType idx = 0; idx < kw % jcp.strideW; idx++) {
                    off += (sizeW + 1 - idx) / jcp.strideW;
                }
                AddImm(x26, x24, off * sizeof(float), x14);
                AddImm(x26, x26, kw / jcp.strideW * sizeof(float), x14);
                Mov(x14, jcp.ic);
                Lsr(x23, x14, GEMM_OFFSET2);
                And(x22, x14, GEMM_K_MASK);
                Mov(x21, x25);
                Cbz(x23, l4[kw]);
                Subs(x23, x23, 1);
                KfirstLoad(x8);
                Ble(l3[kw]);

                L(l2[kw]);
                KloopFmopaKernel(x8);
                Bgt(l2[kw]);

                L(l3[kw]);
                KloopFmopaLast();

                L(l4[kw]);
                Cbz(x22, l6[kw]);

                L(l5[kw]);
                KloopFmopaTail(x8);
                Bgt(l5[kw]);

                L(l6[kw]);
                Mov(x14, jcp.ic);
                Add(x25, x25, x14, ShMod::LSL, KERNEL_OFFSET_SHIFT);
            }
            Mov(x14, (jcp.iw + jcp.lPad + jcp.rPad));
            Add(x28, x28, x14, ShMod::LSL, FLOAT_OFFSET_SHIFT);  // input shift
            Mov(x14, jcp.kw * jcp.ic);
            Add(x30, x30, x14, ShMod::LSL, KERNEL_OFFSET_SHIFT); // kernel shift
            Sub(x18, x18, 1);
            Cmp(x18, 0);
            B(Cond::GT, labelKh);
        }
        Mov(regTmp, sizeW * sizeH);
        Add(x29, x29, regTmp, ShMod::LSL, FLOAT_OFFSET_SHIFT);  // input shift
        Mov(regTmp, jcp.kh * jcp.kw * jcp.ic);
        Add(x16, x16, regTmp, ShMod::LSL, KERNEL_OFFSET_SHIFT); // kernel shift
        Sub(x13, x13, 1);
        Cmp(x13, 0);
        B(Cond::GT, labelKd);
    }
    StoreOutput(regM);
    Incw(regCurN);
    Cmp(regCurN, regN);
    Blt(nLoop);
}

void JitConvMCKernel::DirectKernelGenerate()
{
    IntType outShift = OW_BLOCK * sizeof(float);
    IntType nOi = jcp.ow / OW_BLOCK;
    IntType owTail = jcp.ow % OW_BLOCK;
    IntType inpShift = OW_BLOCK * sizeof(float);

    if (jcp.ow <= OW_BLOCK) {
        if (jcp.strideW != 1) {
            ComputeLoopWithStride(jcp.ow);
        } else {
            ComputeLoop(jcp.ow);
        }
    } else {
        Mov(x27, 0);
        Label owLoopLabel;
        L(owLoopLabel);
        {
            if (jcp.strideW != 1) {
                ComputeLoopWithStride(OW_BLOCK);
            } else {
                ComputeLoop(OW_BLOCK);
            }
            AddImm(x2, x2, inpShift, x17);
            AddImm(x4, x4, outShift, x17);
            AddImm(x27, x27, 1, x17);
            CmpImm(x27, nOi, x17);
            B(Cond::LT, owLoopLabel);
        }
        if (owTail != 0) {
            if (jcp.strideW != 1) {
                ComputeLoopWithStride(owTail);
            } else {
                ComputeLoop(owTail);
            }
        }
    }
}

void JitConvMCKernel::Generate()
{
    Preamble();
    Smstart(StreamMode::SMZA);
    Ptrue(p0.s);

    Ldr(regInp, Ptr(param, static_cast<uint32_t>(offsetof(KdnnConvJitCallS, matrixA))));
    Ldr(regWei, Ptr(param, static_cast<uint32_t>(offsetof(KdnnConvJitCallS, matrixB))));
    Ldr(regOut, Ptr(param, static_cast<uint32_t>(offsetof(KdnnConvJitCallS, matrixC))));
    Ldr(regBia, Ptr(param, static_cast<uint32_t>(offsetof(KdnnConvJitCallS, bias))));
    Ldr(regM, Ptr(param, static_cast<uint32_t>(offsetof(KdnnConvJitCallS, matrixM))));
    Ldr(regN, Ptr(param, static_cast<uint32_t>(offsetof(KdnnConvJitCallS, matrixN))));
    Ldr(regK, Ptr(param, static_cast<uint32_t>(offsetof(KdnnConvJitCallS, matrixK))));

    if (jcp.ver == KuDNN::KdnnConvVersionT::KUDNN_VER_MC_DIRECT) {
        DirectKernelGenerate();
    } else {
        GemmKernelGenerate();
    }

    Smstop(StreamMode::SMZA);
    Postamble();
}

} // namespace KuDNN

#endif // KUDNNL_920F
