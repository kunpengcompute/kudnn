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

#include <memory>
#include <vector>
#include <array>
#include <algorithm>

#include <arm_sve.h>

#include "kudnn.hpp"

#include "cg_codegen.hpp"

#include "kudnn_jit.hpp"
#include "matmul_jit/kudnn_gemm_jit.hpp"
#include "matmul_jit/kudnn_gemm_helpers.hpp"
#include "matmul_jit/kudnn_gemm_threading_helpers.hpp"
#include "kudnn_wrapper_helpers.hpp"
#include "kernels/reorder/kudnn_reorder_def.hpp"
#include "matmul_jit/MatrixComputation/kudnn_default_impl_f32f32f32f32.hpp"

namespace KuDNN {
namespace Detail {

MatMulKernelDefaultCodeGenImpl<ARCH::MatrixComputation, float, float, float, float,
                               MatmulDummyType>::MatMulKernelDefaultCodeGenImpl(IntType m, IntType n, IntType k,
                                                                                IntType srcOffsetArg,
                                                                                IntType weiOffsetArg,
                                                                                IntType dstOffsetArg,
                                                                                IntType biaOffsetArg)
{
    Service::Unused(srcOffsetArg, weiOffsetArg);
    if ((m >= 0) && (n >= 0) && (k >= 0)) {
        Preamble();

        Smstart(ARMCG::StreamMode::SMZA);

        Mov(mVal, m);
        Mov(nVal, n);
        Mov(kVal, k);
        Mov(dstOffset, dstOffsetArg * sizeof(DstType));
        Mov(biaOffset, biaOffsetArg * sizeof(BiaType));
        Mov(mTmpCopy, mVal);
        Mov(nTmpCopy, nVal);
        Mov(kTmpCopy, kVal);
        Mov(srcPtrCopy, srcPtr);
        Mov(weiPtrCopy, weiPtr);
        Mov(dstPtrCopy, dstPtr);
        Mov(biaPtrCopy, biaPtr);

        IntType vl = svcntw();

        // initialize predicates for loads and stores
        Mov(tmp, 0);
        // p0, p2 used for SRC, p1, p3 used for WEI
        Whilelt(ARMCG::PRegS(NUM_0), tmp, mVal);
        Whilelt(ARMCG::PRegS(NUM_1), tmp, nVal);
        if ((m > vl) || (n > vl)) {
            Incw(tmp);
            if (m > vl) {
                Whilelt(ARMCG::PRegS(NUM_2), tmp, mVal);
            }
            if (n > vl) {
                Whilelt(ARMCG::PRegS(NUM_3), tmp, nVal);
            }
        }
        // constants to perform continious stores from ZA tiles
        Mov(tmpLDStC0, NUM_0);
        Mov(tmpLDStC1, NUM_4);
        Mov(tmpLDStC2, NUM_8);
        Mov(tmpLDStC3, NUM_12);

        InitializeDstRegisters(m, n, vl);
        GenerateKLoop(m, n, k, vl);
        StoreDstRegisters(m, n, vl);

        Smstop(ARMCG::StreamMode::SMZA);

        Postamble();
    }
    Ret();
}

void MatMulKernelDefaultCodeGenImpl<ARCH::MatrixComputation, float, float, float, float,
                                    MatmulDummyType>::InitializeDstRegisters(IntType m, IntType n, IntType vl)
{
    // Initialize Dst matrix registers
    ARMCG::Label loadC;
    ARMCG::Label mullC;
    ARMCG::Label loadBias1;
    ARMCG::Label loadBiasM;
    ARMCG::Label loadBiasN;
    ARMCG::Label loadBiasMN;
    ARMCG::Label addBias1;
    ARMCG::Label addBiasM;
    ARMCG::Label addBiasN;
    ARMCG::Label addBiasMN;
    ARMCG::Label initDone;

    Ptrue(ARMCG::PRegS(NUM_5), ARMCG::Pattern::ALL);
    Tbz(flags, 0, loadBias1); // 0-bit correspond to FILL_C_WITH_ZEROZ
    // all 4 tiles are zeroed
    Zero({ARMCG::ZARegS(NUM_0), ARMCG::ZARegS(NUM_1), ARMCG::ZARegS(NUM_2), ARMCG::ZARegS(NUM_3)});

    L(loadBias1);
    Tbz(flags, 3, loadBiasN); // 3-bit correspond to LOAD_BIAS_1
    Ldr(tmpW, ARMCG::AdrUimm(Ptr(biaPtr)));
    Cpy(ARMCG::ZRegS(1), p5, tmpW); // z1 is used as tmp register here
    for (IntType i = 0; i < std::min(m, vl); ++i) {
        // 2 - use even numbered Z registers as tmp, 12 - number of first w register
        // 4 - max imm in MOVA instruction
        Mova(ARMCG::ZARegHS(0), ARMCG::WReg(NUM_12 + i / NUM_4), i % NUM_4, p1, ARMCG::ZRegS(1));
        if (n > vl) {
            // 2 - use odd numbered Z registers as tmp, 12 - number of first w register
            // 4 - max imm in MOVA instruction
            Mova(ARMCG::ZARegHS(NUM_1), ARMCG::WReg(NUM_12 + i / NUM_4), i % NUM_4, p3, ARMCG::ZRegS(1));
        }
    }
    for (IntType i = 0; i < std::min(m - vl, vl); ++i) {
        // 2 - use even numbered Z registers as tmp, 12 - number of first w register
        // 4 - max imm in MOVA instruction
        Mova(ARMCG::ZARegHS(NUM_2), ARMCG::WReg(NUM_12 + i / NUM_4), i % NUM_4, p1, ARMCG::ZRegS(1));
        if (n > vl) {
            // 2 - use odd numbered Z registers as tmp, 12 - number of first w register
            // 4 - max imm in MOVA instruction
            Mova(ARMCG::ZARegHS(NUM_3), ARMCG::WReg(NUM_12 + i / NUM_4), i % NUM_4, p3, ARMCG::ZRegS(1));
        }
    }

    L(loadBiasN);
    Tbz(flags, 5, loadBiasM);                   // 5-bit correspond to LOAD_BIAS_N
    Ld1w(ARMCG::ZRegS(0), p1, Ptr(biaPtrCopy)); // z1 is used as tmp register here
    if (n > vl) {
        Ld1w(ARMCG::ZRegS(1), p3, Ptr(biaPtrCopy, NUM_1, ARMCG::ExtMod::MUL_VL)); // z1 is used as tmp register here
    }
    for (IntType i = 0; i < std::min(m, vl); ++i) {
        // 2 - use even numbered Z registers as tmp, 12 - number of first w register
        // 4 - max imm in MOVA instruction
        Mova(ARMCG::ZARegHS(0), ARMCG::WReg(NUM_12 + i / NUM_4), i % NUM_4, p1, ARMCG::ZRegS(0));
        if (n > vl) {
            // 2 - use odd numbered Z registers as tmp, 12 - number of first w register
            // 4 - max imm in MOVA instruction
            Mova(ARMCG::ZARegHS(NUM_1), ARMCG::WReg(NUM_12 + i / NUM_4), i % NUM_4, p3, ARMCG::ZRegS(1));
        }
    }
    for (IntType i = 0; i < std::min(m - vl, vl); ++i) {
        // 2 - use even numbered Z registers as tmp, 12 - number of first w register
        // 4 - max imm in MOVA instruction
        Mova(ARMCG::ZARegHS(NUM_2), ARMCG::WReg(NUM_12 + i / NUM_4), i % NUM_4, p1, ARMCG::ZRegS(0));
        if (n > vl) {
            // 2 - use odd numbered Z registers as tmp, 12 - number of first w register
            // 4 - max imm in MOVA instruction
            Mova(ARMCG::ZARegHS(NUM_3), ARMCG::WReg(NUM_12 + i / NUM_4), i % NUM_4, p3, ARMCG::ZRegS(1));
        }
    }

    L(loadBiasM);
    Tbz(flags, 4, loadBiasMN);                  // 4-bit correspond to LOAD_BIAS_M
    Ld1w(ARMCG::ZRegS(0), p0, Ptr(biaPtrCopy)); // z1 is used as tmp register here
    if (m > vl) {
        Ld1w(ARMCG::ZRegS(1), p2, Ptr(biaPtrCopy, NUM_1, ARMCG::ExtMod::MUL_VL)); // z1 is used as tmp register here
    }
    for (IntType i = 0; i < std::min(n, vl); ++i) {
        // 2 - use even numbered Z registers as tmp, 12 - number of first w register
        // 4 - max imm in MOVA instruction
        Mova(ARMCG::ZARegVS(0), ARMCG::WReg(NUM_12 + i / NUM_4), i % NUM_4, p0, ARMCG::ZRegS(0));
        if (m > vl) {
            // 2 - use odd numbered Z registers as tmp, 12 - number of first w register
            // 4 - max imm in MOVA instruction
            Mova(ARMCG::ZARegVS(NUM_2), ARMCG::WReg(NUM_12 + i / NUM_4), i % NUM_4, p2, ARMCG::ZRegS(1));
        }
    }
    for (IntType i = 0; i < std::min(n - vl, vl); ++i) {
        // 2 - use even numbered Z registers as tmp, 12 - number of first w register
        // 4 - max imm in MOVA instruction
        Mova(ARMCG::ZARegVS(NUM_1), ARMCG::WReg(NUM_12 + i / NUM_4), i % NUM_4, p0, ARMCG::ZRegS(0));
        if (m > vl) {
            // 2 - use odd numbered Z registers as tmp, 12 - number of first w register
            // 4 - max imm in MOVA instruction
            Mova(ARMCG::ZARegVS(NUM_3), ARMCG::WReg(NUM_12 + i / NUM_4), i % NUM_4, p2, ARMCG::ZRegS(1));
        }
    }

    L(loadBiasMN);
    Tbz(flags, 6, loadC); // 6-bit correspond to LOAD_BIAS_MN
    for (IntType i = 0; i < std::min(m, vl); ++i) {
        if (i == 0) {
            Mov(biaPtrCopy, biaPtr);
        } else {
            Add(biaPtrCopy, biaPtrCopy, biaOffset);
        }
        // 2 - use even numbered Z registers as tmp, 12 - number of first w register
        // 4 - max imm in MOVA instruction
        Ld1w(ARMCG::ZRegS(i * NUM_2), p1, Ptr(biaPtrCopy));
        Mova(ARMCG::ZARegHS(0), ARMCG::WReg(NUM_12 + i / NUM_4), i % NUM_4, p1, ARMCG::ZRegS(i * NUM_2));
        if (n > vl) {
            // 2 - use odd numbered Z registers as tmp, 12 - number of first w register
            // 4 - max imm in MOVA instruction
            Ld1w(ARMCG::ZRegS(i * NUM_2 + NUM_1), p3, Ptr(biaPtrCopy, NUM_1, ARMCG::ExtMod::MUL_VL));
            Mova(ARMCG::ZARegHS(NUM_1), ARMCG::WReg(NUM_12 + i / NUM_4), i % NUM_4, p3,
                 ARMCG::ZRegS(i * NUM_2 + NUM_1));
        }
    }

    for (IntType i = 0; i < std::min(m - vl, vl); ++i) {
        Add(biaPtrCopy, biaPtrCopy, biaOffset);
        // 2 - use even numbered Z registers as tmp, 12 - number of first w register
        // 4 - max imm in MOVA instruction
        Ld1w(ARMCG::ZRegS(i * NUM_2), p1, Ptr(biaPtrCopy));
        Mova(ARMCG::ZARegHS(NUM_2), ARMCG::WReg(NUM_12 + i / NUM_4), i % NUM_4, p1, ARMCG::ZRegS(i * NUM_2));
        if (n > vl) {
            // 2 - use odd numbered Z registers as tmp, 12 - number of first w register
            // 4 - max imm in MOVA instruction
            Ld1w(ARMCG::ZRegS(i * NUM_2 + NUM_1), p3, Ptr(biaPtrCopy, NUM_1, ARMCG::ExtMod::MUL_VL));
            Mova(ARMCG::ZARegHS(NUM_3), ARMCG::WReg(NUM_12 + i / NUM_4), i % NUM_4, p3,
                 ARMCG::ZRegS(i * NUM_2 + NUM_1));
        }
    }

    L(loadC);
    Tbz(flags, 1, mullC); // 1-bit correspond to LOAD_C
    for (IntType i = 0; i < std::min(m, vl); ++i) {
        if (i == 0) {
            Mov(dstPtrCopy, dstPtr);
        } else {
            Add(dstPtrCopy, dstPtrCopy, dstOffset);
        }
        // 2 - use even numbered Z registers as tmp, 12 - number of first w register
        // 4 - max imm in MOVA instruction
        Ld1w(ARMCG::ZRegS(i * NUM_2), p1, Ptr(dstPtrCopy));
        Mova(ARMCG::ZARegHS(0), ARMCG::WReg(NUM_12 + i / NUM_4), i % NUM_4, p1, ARMCG::ZRegS(i * NUM_2));
        if (n > vl) {
            // 2 - use odd numbered Z registers as tmp, 12 - number of first w register
            // 4 - max imm in MOVA instruction
            Ld1w(ARMCG::ZRegS(i * NUM_2 + NUM_1), p3, Ptr(dstPtrCopy, NUM_1, ARMCG::ExtMod::MUL_VL));
            Mova(ARMCG::ZARegHS(NUM_1), ARMCG::WReg(NUM_12 + i / NUM_4), i % NUM_4, p3,
                 ARMCG::ZRegS(i * NUM_2 + NUM_1));
        }
    }
    for (IntType i = 0; i < std::min(m - vl, vl); ++i) {
        Add(dstPtrCopy, dstPtrCopy, dstOffset);
        // 2 - use even numbered Z registers as tmp, 12 - number of first w register
        // 4 - max imm in MOVA instruction
        Ld1w(ARMCG::ZRegS(i * NUM_2), p1, Ptr(dstPtrCopy));
        Mova(ARMCG::ZARegHS(NUM_2), ARMCG::WReg(NUM_12 + i / NUM_4), i % NUM_4, p1, ARMCG::ZRegS(i * NUM_2));
        if (n > vl) {
            // 2 - use odd numbered Z registers as tmp, 12 - number of first w register
            // 4 - max imm in MOVA instruction
            Ld1w(ARMCG::ZRegS(i * NUM_2 + NUM_1), p3, Ptr(dstPtrCopy, NUM_1, ARMCG::ExtMod::MUL_VL));
            Mova(ARMCG::ZARegHS(NUM_3), ARMCG::WReg(NUM_12 + i / NUM_4), i % NUM_4, p3,
                 ARMCG::ZRegS(i * NUM_2 + NUM_1));
        }
    }

    L(mullC);
    Tbz(flags, 2, addBias1);    // 2-bit correspond to MULL_C
    Mov(ARMCG::ZRegS(1), beta); // z1 is used as tmp register here
    for (IntType i = 0; i < std::min(m, vl); ++i) {
        if (i == 0) {
            Mov(dstPtrCopy, dstPtr);
        } else {
            Add(dstPtrCopy, dstPtrCopy, dstOffset);
        }
        // 2 - use even numbered Z registers as tmp, 12 - number of first w register
        // 4 - max imm in MOVA instruction
        Ld1w(ARMCG::ZRegS(i * NUM_2), p1, Ptr(dstPtrCopy));
        if (i == 0) {
            Fmul(ARMCG::ZRegS(i * NUM_2), ARMCG::ZRegS(i * NUM_2), ARMCG::ZRegS(1));
        } else {
            Fmul(ARMCG::ZRegS(i * NUM_2), ARMCG::ZRegS(i * NUM_2), ARMCG::ZRegS(0));
        }
        Mova(ARMCG::ZARegHS(0), ARMCG::WReg(NUM_12 + i / NUM_4), i % NUM_4, p1, ARMCG::ZRegS(i * NUM_2));
        if (i == 0) {
            Mov(ARMCG::ZRegS(0), p5, ARMCG::ZRegS(1));
        }
        if (n > vl) {
            // 2 - use odd numbered Z registers as tmp, 12 - number of first w register
            // 4 - max imm in MOVA instruction
            Ld1w(ARMCG::ZRegS(i * NUM_2 + NUM_1), p3, Ptr(dstPtrCopy, NUM_1, ARMCG::ExtMod::MUL_VL));
            Fmul(ARMCG::ZRegS(i * NUM_2 + NUM_1), ARMCG::ZRegS(i * NUM_2 + NUM_1), ARMCG::ZRegS(0));
            Mova(ARMCG::ZARegHS(NUM_1), ARMCG::WReg(NUM_12 + i / NUM_4), i % NUM_4, p3,
                 ARMCG::ZRegS(i * NUM_2 + NUM_1));
        }
    }
    Mov(ARMCG::ZRegS(1), beta); // z1 is used as tmp register here
    for (IntType i = 0; i < std::min(m - vl, vl); ++i) {
        Add(dstPtrCopy, dstPtrCopy, dstOffset);
        // 2 - use even numbered Z registers as tmp, 12 - number of first w register
        // 4 - max imm in MOVA instruction
        Ld1w(ARMCG::ZRegS(i * NUM_2), p1, Ptr(dstPtrCopy));
        if (i == 0) {
            Fmul(ARMCG::ZRegS(i * NUM_2), ARMCG::ZRegS(i * NUM_2), ARMCG::ZRegS(1));
        } else {
            Fmul(ARMCG::ZRegS(i * NUM_2), ARMCG::ZRegS(i * NUM_2), ARMCG::ZRegS(0));
        }
        Mova(ARMCG::ZARegHS(NUM_2), ARMCG::WReg(NUM_12 + i / NUM_4), i % NUM_4, p1, ARMCG::ZRegS(i * NUM_2));
        if (i == 0) {
            Mov(ARMCG::ZRegS(0), p5, ARMCG::ZRegS(1));
        }
        if (n > vl) {
            // 2 - use odd numbered Z registers as tmp, 12 - number of first w register
            // 4 - max imm in MOVA instruction
            Ld1w(ARMCG::ZRegS(i * NUM_2 + NUM_1), p3, Ptr(dstPtrCopy, NUM_1, ARMCG::ExtMod::MUL_VL));
            Fmul(ARMCG::ZRegS(i * NUM_2 + NUM_1), ARMCG::ZRegS(i * NUM_2 + NUM_1), ARMCG::ZRegS(0));
            Mova(ARMCG::ZARegHS(NUM_3), ARMCG::WReg(NUM_12 + i / NUM_4), i % NUM_4, p3,
                 ARMCG::ZRegS(i * NUM_2 + NUM_1));
        }
    }

    L(addBias1);
    Tbz(flags, 7, addBiasN); // 7-bit correspond to Add_BIAS_1
    Ldr(tmpW, ARMCG::AdrUimm(Ptr(biaPtr)));
    Cpy(ARMCG::ZRegS(1), p5, tmpW); // z1 is used as tmp register here
    for (IntType i = 0; i < std::min(m, vl); ++i) {
        if (i == 0) {
            Mov(dstPtrCopy, dstPtr);
        } else {
            Add(dstPtrCopy, dstPtrCopy, dstOffset);
        }
        // 2 - use even numbered Z registers as tmp, 12 - number of first w register
        // 4 - max imm in MOVA instruction
        Ld1w(ARMCG::ZRegS(i * NUM_2), p1, Ptr(dstPtrCopy));
        if (i == 0) {
            Fadd(ARMCG::ZRegS(i * NUM_2), ARMCG::ZRegS(i * NUM_2), ARMCG::ZRegS(1));
        } else {
            Fadd(ARMCG::ZRegS(i * NUM_2), ARMCG::ZRegS(i * NUM_2), ARMCG::ZRegS(0));
        }
        Mova(ARMCG::ZARegHS(0), ARMCG::WReg(NUM_12 + i / NUM_4), i % NUM_4, p1, ARMCG::ZRegS(i * NUM_2));
        if (i == 0) {
            Mov(ARMCG::ZRegS(0), p5, ARMCG::ZRegS(1));
        }
        if (n > vl) {
            // 2 - use odd numbered Z registers as tmp, 12 - number of first w register
            // 4 - max imm in MOVA instruction
            Ld1w(ARMCG::ZRegS(i * NUM_2 + NUM_1), p3, Ptr(dstPtrCopy, NUM_1, ARMCG::ExtMod::MUL_VL));
            Fadd(ARMCG::ZRegS(i * NUM_2 + NUM_1), ARMCG::ZRegS(i * NUM_2 + NUM_1), ARMCG::ZRegS(0));
            Mova(ARMCG::ZARegHS(NUM_1), ARMCG::WReg(NUM_12 + i / NUM_4), i % NUM_4, p3,
                 ARMCG::ZRegS(i * NUM_2 + NUM_1));
        }
    }
    Cpy(ARMCG::ZRegS(1), p5, tmpW); // z1 is used as tmp register here
    for (IntType i = 0; i < std::min(m - vl, vl); ++i) {
        Add(dstPtrCopy, dstPtrCopy, dstOffset);
        // 2 - use even numbered Z registers as tmp, 12 - number of first w register
        // 4 - max imm in MOVA instruction
        Ld1w(ARMCG::ZRegS(i * NUM_2), p1, Ptr(dstPtrCopy));
        if (i == 0) {
            Fadd(ARMCG::ZRegS(i * NUM_2), ARMCG::ZRegS(i * NUM_2), ARMCG::ZRegS(1));
        } else {
            Fadd(ARMCG::ZRegS(i * NUM_2), ARMCG::ZRegS(i * NUM_2), ARMCG::ZRegS(0));
        }
        Mova(ARMCG::ZARegHS(NUM_2), ARMCG::WReg(NUM_12 + i / NUM_4), i % NUM_4, p1, ARMCG::ZRegS(i * NUM_2));
        if (i == 0) {
            Mov(ARMCG::ZRegS(0), p5, ARMCG::ZRegS(1));
        }
        if (n > vl) {
            // 2 - use odd numbered Z registers as tmp, 12 - number of first w register
            // 4 - max imm in MOVA instruction
            Ld1w(ARMCG::ZRegS(i * NUM_2 + NUM_1), p3, Ptr(dstPtrCopy, NUM_1, ARMCG::ExtMod::MUL_VL));
            Fadd(ARMCG::ZRegS(i * NUM_2 + NUM_1), ARMCG::ZRegS(i * NUM_2 + NUM_1), ARMCG::ZRegS(0));
            Mova(ARMCG::ZARegHS(NUM_3), ARMCG::WReg(NUM_12 + i / NUM_4), i % NUM_4, p3,
                 ARMCG::ZRegS(i * NUM_2 + NUM_1));
        }
    }

    L(addBiasN);
    Tbz(flags, 9, addBiasM);                     // 9-bit correspond to Add_BIAS_N
    Ld1w(ARMCG::ZRegS(30), p1, Ptr(biaPtrCopy)); // z30 is used as tmp register here
    if (n > vl) {
        Ld1w(ARMCG::ZRegS(31), p3, Ptr(biaPtrCopy, NUM_1, ARMCG::ExtMod::MUL_VL)); // z31 is used as tmp here
    }
    for (IntType i = 0; i < std::min(m, vl); ++i) {
        if (i == 0) {
            Mov(dstPtrCopy, dstPtr);
        } else {
            Add(dstPtrCopy, dstPtrCopy, dstOffset);
        }
        // 2 - use even numbered Z registers as tmp, 12 - number of first w register
        // 4 - max imm in MOVA instruction
        Ld1w(ARMCG::ZRegS(i * NUM_2 % NUM_30), p1, Ptr(dstPtrCopy));
        Fadd(ARMCG::ZRegS(i * NUM_2 % NUM_30), ARMCG::ZRegS(i * NUM_2 % NUM_30), ARMCG::ZRegS(NUM_30));
        Mova(ARMCG::ZARegHS(0), ARMCG::WReg(NUM_12 + i / NUM_4), i % NUM_4, p1, ARMCG::ZRegS(i * NUM_2 % NUM_30));
        if (n > vl) {
            // 2 - use odd numbered Z registers as tmp, 12 - number of first w register
            // 4 - max imm in MOVA instruction
            Ld1w(ARMCG::ZRegS((i * NUM_2 + NUM_1) % NUM_30), p3, Ptr(dstPtrCopy, NUM_1, ARMCG::ExtMod::MUL_VL));
            Fadd(ARMCG::ZRegS((i * NUM_2 + NUM_1) % NUM_30), ARMCG::ZRegS((i * NUM_2 + NUM_1) % NUM_30),
                 ARMCG::ZRegS(NUM_31));
            Mova(ARMCG::ZARegHS(NUM_1), ARMCG::WReg(NUM_12 + i / NUM_4), i % NUM_4, p3,
                 ARMCG::ZRegS((i * NUM_2 + NUM_1) % NUM_30));
        }
    }
    Cpy(ARMCG::ZRegS(1), p5, tmpW); // z1 is used as tmp register here
    for (IntType i = 0; i < std::min(m - vl, vl); ++i) {
        Add(dstPtrCopy, dstPtrCopy, dstOffset);
        // 2 - use even numbered Z registers as tmp, 12 - number of first w register
        // 4 - max imm in MOVA instruction
        Ld1w(ARMCG::ZRegS(i * NUM_2 % NUM_30), p1, Ptr(dstPtrCopy));
        Fadd(ARMCG::ZRegS(i * NUM_2 % NUM_30), ARMCG::ZRegS(i * NUM_2 % NUM_30),
             ARMCG::ZRegS(NUM_30)); // z30 contains bias value
        Mova(ARMCG::ZARegHS(NUM_2), ARMCG::WReg(NUM_12 + i / NUM_4), i % NUM_4, p1, ARMCG::ZRegS(i * NUM_2 % NUM_30));
        if (n > vl) {
            // 2 - use odd numbered Z registers as tmp, 12 - number of first w register
            // 4 - max imm in MOVA instruction
            Ld1w(ARMCG::ZRegS((i * NUM_2 + NUM_1) % NUM_30), p3, Ptr(dstPtrCopy, NUM_1, ARMCG::ExtMod::MUL_VL));
            Fadd(ARMCG::ZRegS((i * NUM_2 + NUM_1) % NUM_30), ARMCG::ZRegS((i * NUM_2 + NUM_1) % NUM_30),
                 ARMCG::ZRegS(NUM_31));
            Mova(ARMCG::ZARegHS(NUM_3), ARMCG::WReg(NUM_12 + i / NUM_4), i % NUM_4, p3,
                 ARMCG::ZRegS((i * NUM_2 + NUM_1) % NUM_30));
        }
    }

    L(addBiasM);
    Tbz(flags, 8, addBiasMN); // 8-bit correspond to LOAD_BIAS_M
    for (IntType i = 0; i < std::min(m, vl); ++i) {
        if (i == 0) {
            Mov(dstPtrCopy, dstPtr);
            Mov(biaPtrCopy, biaPtr);
        } else {
            Add(dstPtrCopy, dstPtrCopy, dstOffset);
            Add(biaPtrCopy, biaPtrCopy, NUM_2 * static_cast<std::uint32_t>(sizeof(BiaType)));
        }
        // 2 - use even numbered Z registers as tmp, 12 - number of first w register
        // 4 - max imm in MOVA instruction
        Ldr(tmpW, ARMCG::AdrUimm(Ptr(biaPtrCopy)));
        Cpy(ARMCG::ZRegS((i * NUM_4 + NUM_1) % NUM_32), p5, tmpW);
        Ld1w(ARMCG::ZRegS((i * NUM_4) % NUM_32), p1, Ptr(dstPtrCopy));
        Fadd(ARMCG::ZRegS((i * NUM_4) % NUM_32), ARMCG::ZRegS((i * NUM_4) % NUM_32),
             ARMCG::ZRegS((i * NUM_4 + NUM_1) % NUM_32));
        Mova(ARMCG::ZARegHS(0), ARMCG::WReg(NUM_12 + i / NUM_4), i % NUM_4, p1, ARMCG::ZRegS((i * NUM_4) % NUM_32));
        if (n > vl) {
            // 2 - use odd numbered Z registers as tmp, 12 - number of first w register
            // 4 - max imm in MOVA instruction
            Ldr(tmpW, Ptr(biaPtrCopy, static_cast<std::uint32_t>(sizeof(BiaType))));
            Cpy(ARMCG::ZRegS((i * NUM_4 + NUM_3) % NUM_32), p5, tmpW);
            Ld1w(ARMCG::ZRegS((i * NUM_4 + NUM_2) % NUM_32), p3, Ptr(dstPtrCopy, NUM_1, ARMCG::ExtMod::MUL_VL));
            Fadd(ARMCG::ZRegS((i * NUM_4 + NUM_2) % NUM_32), ARMCG::ZRegS((i * NUM_4 + NUM_2) % NUM_32),
                 ARMCG::ZRegS((i * NUM_4 + NUM_3) % NUM_32));
            Mova(ARMCG::ZARegHS(NUM_1), ARMCG::WReg(NUM_12 + i / NUM_4), i % NUM_4, p3,
                 ARMCG::ZRegS((i * NUM_4 + NUM_2) % NUM_32));
        }
    }
    for (IntType i = 0; i < std::min(m - vl, vl); ++i) {
        Add(dstPtrCopy, dstPtrCopy, dstOffset);
        Add(biaPtrCopy, biaPtrCopy, static_cast<std::uint32_t>(sizeof(BiaType)));
        // 2 - use even numbered Z registers as tmp, 12 - number of first w register
        // 4 - max imm in MOVA instruction
        Ldr(tmpW, ARMCG::AdrUimm(Ptr(biaPtrCopy)));
        Cpy(ARMCG::ZRegS((i * NUM_4 + NUM_1) % NUM_32), p5, tmpW);
        Ld1w(ARMCG::ZRegS((i * NUM_4) % NUM_32), p1, Ptr(dstPtrCopy));
        Fadd(ARMCG::ZRegS((i * NUM_4) % NUM_32), ARMCG::ZRegS((i * NUM_4) % NUM_32),
             ARMCG::ZRegS((i * NUM_4 + NUM_1) % NUM_32));
        Mova(ARMCG::ZARegHS(NUM_2), ARMCG::WReg(NUM_12 + i / NUM_4), i % NUM_4, p1, ARMCG::ZRegS((i * NUM_4) % NUM_32));
        if (n > vl) {
            // 2 - use odd numbered Z registers as tmp, 12 - number of first w register
            // 4 - max imm in MOVA instruction
            Ldr(tmpW, Ptr(biaPtrCopy, static_cast<std::uint32_t>(sizeof(BiaType))));
            Cpy(ARMCG::ZRegS((i * NUM_4 + NUM_3) % NUM_32), p5, tmpW);
            Ld1w(ARMCG::ZRegS((i * NUM_4 + NUM_2) % NUM_32), p3, Ptr(dstPtrCopy, NUM_1, ARMCG::ExtMod::MUL_VL));
            Fadd(ARMCG::ZRegS((i * NUM_4 + NUM_2) % NUM_32), ARMCG::ZRegS((i * NUM_4 + NUM_2) % NUM_32),
                 ARMCG::ZRegS((i * NUM_4 + NUM_3) % NUM_32));
            Mova(ARMCG::ZARegHS(NUM_3), ARMCG::WReg(NUM_12 + i / NUM_4), i % NUM_4, p3,
                 ARMCG::ZRegS((i * NUM_4 + NUM_2) % NUM_32));
        }
    }

    L(addBiasMN);
    Tbz(flags, 10, initDone); // 10-bit correspond to LOAD_BIAS_MN
    for (IntType i = 0; i < std::min(m, vl); ++i) {
        if (i == 0) {
            Mov(dstPtrCopy, dstPtr);
            Mov(biaPtrCopy, biaPtr);
        } else {
            Add(dstPtrCopy, dstPtrCopy, dstOffset);
            Add(biaPtrCopy, biaPtrCopy, biaOffset);
        }
        // 2 - use even numbered Z registers as tmp, 12 - number of first w register
        // 4 - max imm in MOVA instruction
        Ld1w(ARMCG::ZRegS((i * NUM_4) % NUM_32), p1, Ptr(dstPtrCopy));
        Ld1w(ARMCG::ZRegS((i * NUM_4 + NUM_1) % NUM_32), p1, Ptr(biaPtrCopy));
        Fadd(ARMCG::ZRegS((i * NUM_4) % NUM_32), ARMCG::ZRegS((i * NUM_4) % NUM_32),
             ARMCG::ZRegS((i * NUM_4 + NUM_1) % NUM_32));
        Mova(ARMCG::ZARegHS(0), ARMCG::WReg(NUM_12 + i / NUM_4), i % NUM_4, p1, ARMCG::ZRegS((i * NUM_4) % NUM_32));
        if (n > vl) {
            // 2 - use odd numbered Z registers as tmp, 12 - number of first w register
            // 4 - max imm in MOVA instruction
            Ld1w(ARMCG::ZRegS((i * NUM_4 + NUM_2) % NUM_32), p3, Ptr(dstPtrCopy, NUM_1, ARMCG::ExtMod::MUL_VL));
            Ld1w(ARMCG::ZRegS((i * NUM_4 + NUM_3) % NUM_32), p3, Ptr(biaPtrCopy, NUM_1, ARMCG::ExtMod::MUL_VL));
            Fadd(ARMCG::ZRegS((i * NUM_4 + NUM_2) % NUM_32), ARMCG::ZRegS((i * NUM_4 + NUM_2) % NUM_32),
                 ARMCG::ZRegS((i * NUM_4 + NUM_3) % NUM_32));
            Mova(ARMCG::ZARegHS(NUM_1), ARMCG::WReg(NUM_12 + i / NUM_4), i % NUM_4, p3,
                 ARMCG::ZRegS((i * NUM_4 + NUM_2) % NUM_32));
        }
    }
    for (IntType i = 0; i < std::min(m - vl, vl); ++i) {
        Add(dstPtrCopy, dstPtrCopy, dstOffset);
        Add(biaPtrCopy, biaPtrCopy, biaOffset);
        // 2 - use even numbered Z registers as tmp, 12 - number of first w register
        // 4 - max imm in MOVA instruction
        Ld1w(ARMCG::ZRegS((i * NUM_4) % NUM_32), p1, Ptr(dstPtrCopy));
        Ld1w(ARMCG::ZRegS((i * NUM_4 + NUM_1) % NUM_32), p1, Ptr(biaPtrCopy));
        Fadd(ARMCG::ZRegS((i * NUM_4) % NUM_32), ARMCG::ZRegS((i * NUM_4) % NUM_32),
             ARMCG::ZRegS((i * NUM_4 + NUM_1) % NUM_32));
        Mova(ARMCG::ZARegHS(NUM_2), ARMCG::WReg(NUM_12 + i / NUM_4), i % NUM_4, p1, ARMCG::ZRegS((i * NUM_4) % NUM_32));
        if (n > vl) {
            // 2 - use odd numbered Z registers as tmp, 12 - number of first w register
            // 4 - max imm in MOVA instruction
            Ld1w(ARMCG::ZRegS((i * NUM_4 + NUM_2) % NUM_32), p3, Ptr(dstPtrCopy, NUM_1, ARMCG::ExtMod::MUL_VL));
            Ld1w(ARMCG::ZRegS((i * NUM_4 + NUM_3) % NUM_32), p3, Ptr(biaPtrCopy, NUM_1, ARMCG::ExtMod::MUL_VL));
            Fadd(ARMCG::ZRegS((i * NUM_4 + NUM_2) % NUM_32), ARMCG::ZRegS((i * NUM_4 + NUM_2) % NUM_32),
                 ARMCG::ZRegS((i * NUM_4 + NUM_3) % NUM_32));
            Mova(ARMCG::ZARegHS(NUM_3), ARMCG::WReg(NUM_12 + i / NUM_4), i % NUM_4, p3,
                 ARMCG::ZRegS((i * NUM_4 + NUM_2) % NUM_32));
        }
    }

    L(initDone);
}

void MatMulKernelDefaultCodeGenImpl<ARCH::MatrixComputation, float, float, float, float,
                                    MatmulDummyType>::GenerateKLoop(IntType m, IntType n, IntType k, IntType vl)
{
    ARMCG::Label kLoopHead;
    Mov(kTmpCopy, kVal);
    Subs(kTmpCopy, kTmpCopy, kLoopUnroll);
    // p0, p2 - for SRC, p1, p3 - for WEI
    // z0, z2, z4, z6 - for SRC, z1, z3, z5, z7 - for WEI
    // SRC and WEI data is prepacked, so it's continious in memory.
    // K unroll by 2 requires 4 elements from SRC,  4 from WEI
    // Load src data
    Ld1w(ARMCG::ZRegS(NUM_0), p0, Ptr(srcPtrCopy));
    if (m > vl) {
        Ld1w(ARMCG::ZRegS(NUM_2), p2, Ptr(srcPtrCopy, NUM_1, ARMCG::ExtMod::MUL_VL));
    }
    // Load wei data
    Ld1w(ARMCG::ZRegS(NUM_1), p1, Ptr(weiPtrCopy));
    if (n > vl) {
        Ld1w(ARMCG::ZRegS(NUM_3), p3, Ptr(weiPtrCopy, NUM_1, ARMCG::ExtMod::MUL_VL));
    }
    if (k >= kLoopUnroll) {
        // Load src data for unrolled loop
        Ld1w(ARMCG::ZRegS(NUM_4), p0, Ptr(srcPtrCopy, NUM_2, ARMCG::ExtMod::MUL_VL));
        if (m > vl) {
            Ld1w(ARMCG::ZRegS(NUM_6), p2, Ptr(srcPtrCopy, NUM_3, ARMCG::ExtMod::MUL_VL));
        }
        // Load wei data unrolled loop
        Ld1w(ARMCG::ZRegS(NUM_5), p1, Ptr(weiPtrCopy, NUM_2, ARMCG::ExtMod::MUL_VL));
        if (n > vl) {
            Ld1w(ARMCG::ZRegS(NUM_7), p3, Ptr(weiPtrCopy, NUM_3, ARMCG::ExtMod::MUL_VL));
        }
        Addvl(srcPtrCopy, srcPtrCopy, kernelMSize * kLoopUnroll);
        Addvl(weiPtrCopy, weiPtrCopy, kernelNSize * kLoopUnroll);
        if (k > kLoopUnroll) {
            L(kLoopHead);
            Fmopa(ARMCG::ZARegS(NUM_0), p0, p1, ARMCG::ZRegS(NUM_0), ARMCG::ZRegS(NUM_1));
            Subs(kTmpCopy, kTmpCopy, kLoopUnroll);
            if (n > vl) {
                Fmopa(ARMCG::ZARegS(NUM_1), p0, p3, ARMCG::ZRegS(NUM_0), ARMCG::ZRegS(NUM_3));
            }
            Ld1w(ARMCG::ZRegS(NUM_0), p0, Ptr(srcPtrCopy));
            if (m > vl) {
                Fmopa(ARMCG::ZARegS(NUM_2), p2, p1, ARMCG::ZRegS(NUM_2), ARMCG::ZRegS(NUM_1));
            }
            Ld1w(ARMCG::ZRegS(NUM_1), p1, Ptr(weiPtrCopy));
            if ((m > vl) && (n > vl)) {
                Fmopa(ARMCG::ZARegS(NUM_3), p2, p3, ARMCG::ZRegS(NUM_2), ARMCG::ZRegS(NUM_3));
            }
            if (m > vl) {
                Ld1w(ARMCG::ZRegS(NUM_2), p2, Ptr(srcPtrCopy, NUM_1, ARMCG::ExtMod::MUL_VL));
            }
            if (n > vl) {
                Ld1w(ARMCG::ZRegS(NUM_3), p3, Ptr(weiPtrCopy, NUM_1, ARMCG::ExtMod::MUL_VL));
            }

            Fmopa(ARMCG::ZARegS(NUM_0), p0, p1, ARMCG::ZRegS(NUM_4), ARMCG::ZRegS(NUM_5));
            if (n > vl) {
                Fmopa(ARMCG::ZARegS(NUM_1), p0, p3, ARMCG::ZRegS(NUM_4), ARMCG::ZRegS(NUM_7));
            }
            Ld1w(ARMCG::ZRegS(NUM_4), p0, Ptr(srcPtrCopy, NUM_2, ARMCG::ExtMod::MUL_VL));
            if (m > vl) {
                Fmopa(ARMCG::ZARegS(NUM_2), p2, p1, ARMCG::ZRegS(NUM_6), ARMCG::ZRegS(NUM_5));
            }
            Ld1w(ARMCG::ZRegS(NUM_5), p1, Ptr(weiPtrCopy, NUM_2, ARMCG::ExtMod::MUL_VL));
            if ((m > vl) && (n > vl)) {
                Fmopa(ARMCG::ZARegS(NUM_3), p2, p3, ARMCG::ZRegS(NUM_6), ARMCG::ZRegS(NUM_7));
            }
            if (m > vl) {
                Ld1w(ARMCG::ZRegS(NUM_6), p2, Ptr(srcPtrCopy, NUM_3, ARMCG::ExtMod::MUL_VL));
            }
            if (n > vl) {
                Ld1w(ARMCG::ZRegS(NUM_7), p3, Ptr(weiPtrCopy, NUM_3, ARMCG::ExtMod::MUL_VL));
            }

            Addvl(srcPtrCopy, srcPtrCopy, kernelMSize * kLoopUnroll);
            Addvl(weiPtrCopy, weiPtrCopy, kernelNSize * kLoopUnroll);

            B(ARMCG::Cond::GT, kLoopHead);
        }
    }

    Fmopa(ARMCG::ZARegS(NUM_0), p0, p1, ARMCG::ZRegS(NUM_0), ARMCG::ZRegS(NUM_1));
    if (n > vl) {
        Fmopa(ARMCG::ZARegS(NUM_1), p0, p3, ARMCG::ZRegS(NUM_0), ARMCG::ZRegS(NUM_3));
    }
    if (m > vl) {
        Fmopa(ARMCG::ZARegS(NUM_2), p2, p1, ARMCG::ZRegS(NUM_2), ARMCG::ZRegS(NUM_1));
    }
    if ((m > vl) && (n > vl)) {
        Fmopa(ARMCG::ZARegS(NUM_3), p2, p3, ARMCG::ZRegS(NUM_2), ARMCG::ZRegS(NUM_3));
    }
    if (k % kLoopUnroll == 0) {
        Fmopa(ARMCG::ZARegS(NUM_0), p0, p1, ARMCG::ZRegS(NUM_4), ARMCG::ZRegS(NUM_5));
        if (n > vl) {
            Fmopa(ARMCG::ZARegS(NUM_1), p0, p3, ARMCG::ZRegS(NUM_4), ARMCG::ZRegS(NUM_7));
        }
        if (m > vl) {
            Fmopa(ARMCG::ZARegS(NUM_2), p2, p1, ARMCG::ZRegS(NUM_6), ARMCG::ZRegS(NUM_5));
        }
        if ((m > vl) && (n > vl)) {
            Fmopa(ARMCG::ZARegS(NUM_3), p2, p3, ARMCG::ZRegS(NUM_6), ARMCG::ZRegS(NUM_7));
        }
    }
}

void MatMulKernelDefaultCodeGenImpl<ARCH::MatrixComputation, float, float, float, float,
                                    MatmulDummyType>::StoreDstRegisters(IntType m, IntType n, IntType vl)
{
    for (IntType i = 0; i < std::min(m, vl); ++i) {
        if (i == 0) {
            Mov(dstPtrCopy, dstPtr);
        } else {
            Add(dstPtrCopy, dstPtrCopy, dstOffset);
        }
        // 2 - use even numbered Z registers as tmp, 12 - number of first w register
        // 4 - max imm in MOVA instruction
        Mova(ARMCG::ZRegS(i * NUM_2), p1, ARMCG::ZARegHS(0), ARMCG::WReg(NUM_12 + i / NUM_4), i % NUM_4);
        St1w(ARMCG::ZRegS(i * NUM_2), p1, Ptr(dstPtrCopy));
        if (n > vl) {
            // 2 - use odd numbered Z registers as tmp, 12 - number of first w register
            // 4 - max imm in MOVA instruction
            Mova(ARMCG::ZRegS(i * NUM_2 + NUM_1), p3, ARMCG::ZARegHS(NUM_1), ARMCG::WReg(NUM_12 + i / NUM_4),
                 i % NUM_4);
            St1w(ARMCG::ZRegS(i * NUM_2 + NUM_1), p3, Ptr(dstPtrCopy, NUM_1, ARMCG::ExtMod::MUL_VL));
        }
    }

    for (IntType i = 0; i < std::min(m - vl, vl); ++i) {
        Add(dstPtrCopy, dstPtrCopy, dstOffset);
        // 2 - use even numbered Z registers as tmp, 12 - number of first w register
        // 4 - max imm in MOVA instruction
        Mova(ARMCG::ZRegS(i * NUM_2), p1, ARMCG::ZARegHS(NUM_2), ARMCG::WReg(NUM_12 + i / NUM_4), i % NUM_4);
        St1w(ARMCG::ZRegS(i * NUM_2), p1, Ptr(dstPtrCopy));
        if (n > vl) {
            // 2 - use odd numbered Z registers as tmp, 12 - number of first w register
            // 4 - max imm in MOVA instruction
            Mova(ARMCG::ZRegS(i * NUM_2 + NUM_1), p3, ARMCG::ZARegHS(NUM_3), ARMCG::WReg(NUM_12 + i / NUM_4),
                 i % NUM_4);
            St1w(ARMCG::ZRegS(i * NUM_2 + NUM_1), p3, Ptr(dstPtrCopy, NUM_1, ARMCG::ExtMod::MUL_VL));
        }
    }
}

using MatrixComputationFloatGen = MatMulKernelDefaultCodeGen<ARCH::MatrixComputation, float, float, float, float>;
MatMulImplGenerator<MatrixComputationFloatGen>::DefaultImpl::DefaultImpl(const GemmInfo &task, IntType nThreads)
    : MatMulImplBase(task, nThreads)
{
    VERBOSE_PRINT;
    UpdateExecParams();
    Initialize();
}

std::uint32_t MatMulImplGenerator<MatrixComputationFloatGen>::DefaultImpl::GetKernelMSize() noexcept
{
    return CodeGen::kernelMSize * svcntw();
}

std::uint32_t MatMulImplGenerator<MatrixComputationFloatGen>::DefaultImpl::GetKernelNSize() noexcept
{
    return CodeGen::kernelNSize * svcntw();
}

void MatMulImplGenerator<MatrixComputationFloatGen>::DefaultImpl::InitializeCodeGens(
    IntType m, IntType n, IntType k, IntType vl, SizeType offsetSrc, SizeType offsetWei, SizeType offsetDst,
    SizeType offsetBia, bool wereSrcWeiSwapped)
{
    Service::Unused(offsetSrc, offsetWei, wereSrcWeiSwapped); // they are actually not used by implementation
    IntType mSubBlockExecFull = CodeGen::kernelMSize * vl;
    IntType mSubBlockExecResid = m % mSubBlockExecFull;
    IntType nSubBlockExecFull = CodeGen::kernelNSize * vl;
    IntType nSubBlockExecResid = n % nSubBlockExecFull;
    IntType kBlockExecFull = CodeGen::GetKBlockSize(m, n, k);
    IntType kBlockExecResid = k % CodeGen::GetKBlockSize(m, n, k);
    CodeGenWrapper<CodeGen> cgw0;
    cgw0.Initialize(mSubBlockExecFull, nSubBlockExecFull, kBlockExecFull, static_cast<IntType>(offsetSrc),
                    static_cast<IntType>(offsetWei), static_cast<IntType>(offsetDst), static_cast<IntType>(offsetBia));
    CodeGenWrapper<CodeGen> cgw1;
    cgw1.Initialize(mSubBlockExecFull, nSubBlockExecFull, kBlockExecResid, static_cast<IntType>(offsetSrc),
                    static_cast<IntType>(offsetWei), static_cast<IntType>(offsetDst), static_cast<IntType>(offsetBia));
    CodeGenWrapper<CodeGen> cgw2;
    cgw2.Initialize(mSubBlockExecFull, nSubBlockExecResid, kBlockExecFull, static_cast<IntType>(offsetSrc),
                    static_cast<IntType>(offsetWei), static_cast<IntType>(offsetDst), static_cast<IntType>(offsetBia));
    CodeGenWrapper<CodeGen> cgw3;
    cgw3.Initialize(mSubBlockExecFull, nSubBlockExecResid, kBlockExecResid, static_cast<IntType>(offsetSrc),
                    static_cast<IntType>(offsetWei), static_cast<IntType>(offsetDst), static_cast<IntType>(offsetBia));
    CodeGenWrapper<CodeGen> cgw4;
    cgw4.Initialize(mSubBlockExecResid, nSubBlockExecFull, kBlockExecFull, static_cast<IntType>(offsetSrc),
                    static_cast<IntType>(offsetWei), static_cast<IntType>(offsetDst), static_cast<IntType>(offsetBia));
    CodeGenWrapper<CodeGen> cgw5;
    cgw5.Initialize(mSubBlockExecResid, nSubBlockExecFull, kBlockExecResid, static_cast<IntType>(offsetSrc),
                    static_cast<IntType>(offsetWei), static_cast<IntType>(offsetDst), static_cast<IntType>(offsetBia));
    CodeGenWrapper<CodeGen> cgw6;
    cgw6.Initialize(mSubBlockExecResid, nSubBlockExecResid, kBlockExecFull, static_cast<IntType>(offsetSrc),
                    static_cast<IntType>(offsetWei), static_cast<IntType>(offsetDst), static_cast<IntType>(offsetBia));
    CodeGenWrapper<CodeGen> cgw7;
    cgw7.Initialize(mSubBlockExecResid, nSubBlockExecResid, kBlockExecResid, static_cast<IntType>(offsetSrc),
                    static_cast<IntType>(offsetWei), static_cast<IntType>(offsetDst), static_cast<IntType>(offsetBia));
    perThreadImpls.push_back({std::move(cgw0), std::move(cgw1), std::move(cgw2), std::move(cgw3), std::move(cgw4),
                              std::move(cgw5), std::move(cgw6), std::move(cgw7)});
}

IntType MatMulImplGenerator<MatrixComputationFloatGen>::DefaultImpl::GetThreadBufSize(IntType m, IntType n, IntType k,
                                                                                      IntType vl) noexcept
{
    IntType mRoundedUp = Math::CeilDiv(m, CodeGen::kernelMSize * vl) * CodeGen::kernelMSize * vl;
    IntType nRoundedUp = Math::CeilDiv(n, CodeGen::kernelNSize * vl) * CodeGen::kernelNSize * vl;
    IntType kRoundedUp = Math::CeilDiv(k, CodeGen::kLoopUnroll) * CodeGen::kLoopUnroll;
    return static_cast<IntType>(std::min(CodeGen::GetKBlockSize(m, n, k), kRoundedUp) *
                                (std::min(CodeGen::GetMBlockSize(m, n, k), mRoundedUp) * sizeof(SrcType) +
                                 std::min(CodeGen::GetNBlockSize(m, n, k), nRoundedUp) * sizeof(WeiType)));
}

void MatMulImplGenerator<MatrixComputationFloatGen>::DefaultImpl::ReserveImplSpecificParams() {}

std::vector<std::pair<Shape, Shape>> MatMulImplGenerator<MatrixComputationFloatGen>::DefaultImpl::SplitWork()
{
    return CalculateWorkOnThread<DefaultImpl>(
        taskExec.dstInfo.GetDims(), implNThreads, dstLayout2D == Layout::BA,
        taskExec.srcInfo.GetDims()[taskExec.srcInfo.GetNumDims() - 1] > 256); // 256
    // is a number indicating that k is big enough to omit condition mWork % kernelMSize == 0 (same for n)
}

// used for packing both Src and Wei matrices
template <bool AlphaMul>
void MatMulImplGenerator<MatrixComputationFloatGen>::DefaultImpl::PackSimilarLt(IntType m, IntType k, const float *src,
                                                                                IntType offset, float *buffer,
                                                                                IntType bufferOffset, float alpha)
{
    IntType vl = svcntw();
    svbool_t ptrue = svptrue_b32();
    for (IntType kIdx = 0; kIdx < k; ++kIdx) {
        for (IntType mIdx = 0; mIdx < m; mIdx += bufferOffset) {
            svbool_t p0 = svwhilelt_b32(mIdx + 0 * vl, m);
            svfloat32_t a0 = svld1(p0, src + 0 * vl + mIdx + offset * kIdx);
            svbool_t p1 = svwhilelt_b32(mIdx + 1 * vl, m);
            svfloat32_t a1 = svld1(p1, src + 1 * vl + mIdx + offset * kIdx);
            if constexpr (AlphaMul) {
                svfloat32_t alpha_v = svdup_n_f32(alpha);
                a0 = svmul_f32_x(ptrue, a0, alpha_v);
                a1 = svmul_f32_x(ptrue, a1, alpha_v);
            }
            svst1(ptrue, buffer + 0 * vl + bufferOffset * kIdx + k * mIdx, a0);
            svst1(ptrue, buffer + 1 * vl + bufferOffset * kIdx + k * mIdx, a1);
        }
    }
}

// used for packing both Src and Wei matrices
template <bool AlphaMul>
void MatMulImplGenerator<MatrixComputationFloatGen>::DefaultImpl::PackDiffLt(IntType m, IntType k, const float *src,
                                                                             IntType offset, float *buffer,
                                                                             IntType bufferOffset, float alpha)
{
    const IntType vl = svcntw();
    IntType idxOffset = 0;
    if constexpr (AlphaMul) {
        idxOffset = vl * vl; // TRANSPOSE_REORDER_FUNC_ARR values with alpha and w/o alpha
    }
    for (IntType mIdx = m; mIdx > 0; mIdx -= bufferOffset) {
        if (mIdx <= vl) {
            IntType mIdxReorder = std::min(vl, mIdx) - 1;
            for (IntType kIdx = k; kIdx > 0; kIdx -= vl) {
                IntType kIdxReorder = std::min(vl, kIdx) - 1;
                Reorder::TRANSPOSE_REORDER_FUNC_ARR[mIdxReorder * vl + kIdxReorder + idxOffset](
                    src + (m - mIdx) * offset + (k - kIdx), buffer + (k - kIdx) * bufferOffset + k * (m - mIdx),
                    offset * sizeof(float), bufferOffset * sizeof(float), alpha);
            }
        } else {
            IntType mIdxReorder1 = vl - 1;
            IntType mIdxReorder2 = std::min(vl, mIdx - vl) - 1;
            for (IntType kIdx = k; kIdx > 0; kIdx -= vl) {
                IntType kIdxReorder = std::min(vl, kIdx) - 1;
                Reorder::TRANSPOSE_REORDER_FUNC_ARR[mIdxReorder1 * vl + kIdxReorder + idxOffset](
                    src + (m - mIdx) * offset + (k - kIdx), buffer + (k - kIdx) * bufferOffset + k * (m - mIdx),
                    offset * sizeof(float), bufferOffset * sizeof(float), alpha);
                Reorder::TRANSPOSE_REORDER_FUNC_ARR[mIdxReorder2 * vl + kIdxReorder + idxOffset](
                    src + (m - mIdx + vl) * offset + (k - kIdx),
                    buffer + (k - kIdx) * bufferOffset + k * (m - mIdx) + vl, offset * sizeof(float),
                    bufferOffset * sizeof(float), alpha);
            }
        }
    }
}

void MatMulImplGenerator<MatrixComputationFloatGen>::DefaultImpl::RunSequential2D(IntType threadId, const void *src,
                                                                                  const void *wei, void *dst,
                                                                                  const void *bia, void *buffer,
                                                                                  float alpha, float beta) const
    noexcept(false)
{
    VERBOSE_PRINT;
    IntType m = perThreadParams[threadId].m;
    IntType n = perThreadParams[threadId].n;
    IntType k = perThreadParams[threadId].k;
    IntType vl = svcntw();
    IntType mRoundedUp = Math::CeilDiv(m, CodeGen::kernelMSize * vl) * CodeGen::kernelMSize * vl;
    IntType kRoundedUp = Math::CeilDiv(k, CodeGen::kLoopUnroll) * CodeGen::kLoopUnroll;
    SrcType *srcBuffer = static_cast<SrcType *>(buffer);
    WeiType *weiBuffer = static_cast<WeiType *>(static_cast<SrcType *>(buffer) +
                                                std::min(CodeGen::GetKBlockSize(m, n, k), kRoundedUp) *
                                                    std::min(CodeGen::GetMBlockSize(m, n, k), mRoundedUp));
    for (IntType mBlock = 0; mBlock < m; mBlock += CodeGen::GetMBlockSize(m, n, k)) {
        auto mBlockExec = std::min(CodeGen::GetMBlockSize(m, n, k), m - mBlock);
        for (IntType kBlock = 0; kBlock < k; kBlock += CodeGen::GetKBlockSize(m, n, k)) {
            auto kBlockExec = std::min(CodeGen::GetKBlockSize(m, n, k), k - kBlock);
            int kIdx = (kBlockExec == CodeGen::GetKBlockSize(m, n, k)) ? 0 : 1;
            DstLoadFlags biaFlagsExec = (kBlock || (beta != 0.0f)) ? DstLoadFlags::LOAD_C : biaFlags;
            if ((beta != 1.0f) && (beta != 0.0f)) {
                biaFlagsExec = biaFlagsExec | DstLoadFlags::MULL_C;
            }
            for (IntType nBlock = 0; nBlock < n; nBlock += CodeGen::GetNBlockSize(m, n, k)) {
                auto nBlockExec = std::min(CodeGen::GetNBlockSize(m, n, k), n - nBlock);
                for (IntType mSubBlock = 0; mSubBlock < mBlockExec; mSubBlock += CodeGen::kernelMSize * vl) {
                    IntType mSubBlockExec = std::min(CodeGen::kernelMSize * vl, mBlockExec - mSubBlock);
                    int mIdx = (mSubBlockExec == (CodeGen::kernelMSize * vl)) ? 0 : 1;
                    if (nBlock == 0) {
                        if (perThreadParams[threadId].srcLayout2D == Layout::COL_MAJOR) {
                            const SrcType *srcExec = static_cast<const SrcType *>(src) + mBlock + mSubBlock +
                                                     kBlock * perThreadParams[threadId].srcOffset2D;
                            if (alpha != 1) {
                                PackSimilarLt<true>(
                                    mSubBlockExec, kBlockExec, srcExec, perThreadParams[threadId].srcOffset2D,
                                    srcBuffer + kBlockExec * mSubBlock, CodeGen::kernelMSize * vl, alpha);
                            } else {
                                PackSimilarLt<false>(mSubBlockExec, kBlockExec, srcExec,
                                                     perThreadParams[threadId].srcOffset2D,
                                                     srcBuffer + kBlockExec * mSubBlock, CodeGen::kernelMSize * vl);
                            }
                        } else {
                            const SrcType *srcExec = static_cast<const SrcType *>(src) +
                                                     (mBlock + mSubBlock) * perThreadParams[threadId].srcOffset2D +
                                                     kBlock;
                            if (alpha != 1) {
                                PackDiffLt<true>(mSubBlockExec, kBlockExec, srcExec,
                                                 perThreadParams[threadId].srcOffset2D,
                                                 srcBuffer + kBlockExec * mSubBlock, CodeGen::kernelMSize * vl, alpha);
                            } else {
                                PackDiffLt<false>(mSubBlockExec, kBlockExec, srcExec,
                                                  perThreadParams[threadId].srcOffset2D,
                                                  srcBuffer + kBlockExec * mSubBlock, CodeGen::kernelMSize * vl);
                            }
                        }
                    }
                    for (IntType nSubBlock = 0; nSubBlock < nBlockExec; nSubBlock += CodeGen::kernelNSize * vl) {
                        IntType nSubBlockExec = std::min(CodeGen::kernelNSize * vl, nBlockExec - nSubBlock);
                        int nIdx = (nSubBlockExec == (CodeGen::kernelNSize * vl)) ? 0 : 1;
                        if (mSubBlock == 0) {
                            if (perThreadParams[threadId].weiLayout2D == Layout::ROW_MAJOR) {
                                const WeiType *weiExec = static_cast<const WeiType *>(wei) + nBlock + nSubBlock +
                                                         kBlock * perThreadParams[threadId].weiOffset2D;
                                PackSimilarLt(nSubBlockExec, kBlockExec, weiExec, perThreadParams[threadId].weiOffset2D,
                                              weiBuffer + kBlockExec * nSubBlock, CodeGen::kernelNSize * vl);
                            } else {
                                const WeiType *weiExec = static_cast<const WeiType *>(wei) +
                                                         (nBlock + nSubBlock) * perThreadParams[threadId].weiOffset2D +
                                                         kBlock;
                                PackDiffLt(nSubBlockExec, kBlockExec, weiExec, perThreadParams[threadId].weiOffset2D,
                                           weiBuffer + kBlockExec * nSubBlock, CodeGen::kernelNSize * vl);
                            }
                        }
                        DstType *dstExec = static_cast<DstType *>(dst) + nBlock + nSubBlock +
                                           (mBlock + mSubBlock) * perThreadParams[threadId].dstOffset2D;
                        const BiaType *biaExec = static_cast<const BiaType *>(bia) + (nBlock + nSubBlock) * biaNOffset +
                                                 (mBlock + mSubBlock) * perThreadParams[threadId].biaOffset2D;
                        auto kernel = perThreadImpls[threadId][mIdx * NUM_4 + nIdx * NUM_2 + kIdx]
                                          .template GetCode<void (*)(const SrcType *, const WeiType *, DstType *,
                                                                     const BiaType *, DstLoadFlags, float)>();
                        kernel(srcBuffer + mSubBlock * kBlockExec, weiBuffer + nSubBlock * kBlockExec, dstExec, biaExec,
                               biaFlagsExec, beta);
                    }
                }
            }
        }
    }
}

void MatMulImplGenerator<MatrixComputationFloatGen>::DefaultImpl::UpdateExecParams() noexcept
{
    VERBOSE_PRINT;
    Layout desiredSrcLayout = implTask.srcInfo.GetLayout();
    Layout desiredWeiLayout = implTask.weiInfo.GetLayout();
    Layout desiredDstLayout = implTask.dstInfo.GetLayout();
    Layout desiredBiaLayout = implTask.biaInfo.GetLayout();
    if (!IsLayoutAlikeAB(desiredSrcLayout) && !IsLayoutAlikeBA(desiredSrcLayout)) {
        desiredSrcLayout = GetPlaneLayoutBA(desiredSrcLayout);
    }
    if (!IsLayoutAlikeAB(desiredWeiLayout) && !IsLayoutAlikeBA(desiredWeiLayout)) {
        desiredWeiLayout = GetPlaneLayoutAB(desiredWeiLayout);
    }
    if (!IsLayoutAlikeAB(desiredDstLayout) && !IsLayoutAlikeBA(desiredDstLayout)) {
        desiredDstLayout = GetPlaneLayoutAB(desiredDstLayout);
    }
    if (implTask.biaInfo.GetTotalTensorSize() != 0) {
        desiredBiaLayout = desiredDstLayout;
    }
    InitializeTaskExec(desiredSrcLayout, desiredWeiLayout, desiredDstLayout, desiredBiaLayout, GetSrcType(),
                       GetWeiType(), GetDstType(),
                       (implTask.biaInfo.GetTotalTensorSize() != 0) ?
                           GetBiaType() :
                           static_cast<Element::TypeT>(implTask.biaInfo.GetType()));
}

} // namespace Detail
} // namespace KuDNN
