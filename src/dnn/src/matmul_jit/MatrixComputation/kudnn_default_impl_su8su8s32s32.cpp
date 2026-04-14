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
#include <type_traits>
#include <cstdint>

#include <arm_sve.h>

#include "kudnn.hpp"

#include "cg_codegen.hpp"

#include "kudnn_jit.hpp"
#include "matmul_jit/kudnn_gemm_jit.hpp"
#include "matmul_jit/kudnn_gemm_helpers.hpp"
#include "matmul_jit/kudnn_gemm_threading_helpers.hpp"
#include "kudnn_wrapper_helpers.hpp"
#include "kernels/reorder/kudnn_reorder_def.hpp"
#include "matmul_jit/MatrixComputation/kudnn_default_impl_su8su8s32s32.hpp"

namespace KuDNN {
namespace Detail {

template <typename SrcType, typename WeiType, typename DstType, typename BiaType>
MatMulKernelDefaultCodeGenImpl<ARCH::MatrixComputation, SrcType, WeiType, DstType, BiaType,
    typename std::enable_if<IsIntegerMatmul<SrcType, WeiType, DstType, BiaType>(),
    MatmulDummyType>::type>::MatMulKernelDefaultCodeGenImpl(
    IntType m, IntType n, IntType k,
    IntType biaOffsetArg, bool wereSrcWeiSwappedArg) : wereSrcWeiSwapped(wereSrcWeiSwappedArg)
{
    if ((m >= 0) && (n >= 0) && (k >= 0)) {
        Preamble();

        Smstart(ARMCG::StreamMode::SMZA);

        KUDNN_JIT_Generator::Mov(mVal, m);
        KUDNN_JIT_Generator::Mov(nVal, n);
        KUDNN_JIT_Generator::Mov(kVal, k);
        KUDNN_JIT_Generator::Mov(biaOffset, biaOffsetArg * sizeof(BiaType));
        KUDNN_JIT_Generator::Mov(tmp, NUM_4);
        KUDNN_JIT_Generator::Mul(mTmpCopy, mVal, tmp); // NUM_4 due to peculiarities of INT8 MOPA instruction
        KUDNN_JIT_Generator::Mul(nTmpCopy, nVal, tmp); // NUM_4 due to peculiarities of INT8 MOPA instruction
        KUDNN_JIT_Generator::Mov(srcPtrCopy, srcPtr);
        KUDNN_JIT_Generator::Mov(weiPtrCopy, weiPtr);
        KUDNN_JIT_Generator::Mov(dstPtrCopy, dstPtr);
        KUDNN_JIT_Generator::Mov(biaPtrCopy, biaPtr);

        IntType vlw = svcntw();

        // initialize predicates for loads and stores
        KUDNN_JIT_Generator::Mov(tmp, 0);
        // p0, p2 used for SRC, p1, p3 used for WEI
        Whilelt(ARMCG::PRegB(NUM_0), tmp, mTmpCopy);
        Whilelt(ARMCG::PRegB(NUM_1), tmp, nTmpCopy);
        if ((m > vlw) || (n > vlw)) {
            Incb(tmp);
            if (m > vlw) {
                Whilelt(ARMCG::PRegB(NUM_2), tmp, mTmpCopy);
            }
            if (n > vlw) {
                Whilelt(ARMCG::PRegB(NUM_3), tmp, nTmpCopy);
            }
        }
        // constants to perform continious stores from ZA tiles
        KUDNN_JIT_Generator::Mov(tmpLDStC0, NUM_0);
        KUDNN_JIT_Generator::Mov(tmpLDStC1, NUM_4);
        KUDNN_JIT_Generator::Mov(tmpLDStC2, NUM_8);
        KUDNN_JIT_Generator::Mov(tmpLDStC3, NUM_12);

        InitializeDstRegisters(m, n, vlw);
        GenerateKLoop(m, n, k, vlw);
        StoreDstRegisters(m, n, vlw);

        Smstop(ARMCG::StreamMode::SMZA);

        Postamble();
    }
    Ret();
}

template <typename SrcType, typename WeiType, typename DstType, typename BiaType>
void MatMulKernelDefaultCodeGenImpl<ARCH::MatrixComputation, SrcType, WeiType, DstType, BiaType,
    typename std::enable_if<IsIntegerMatmul<SrcType, WeiType, DstType, BiaType>(),
    MatmulDummyType>::type>::InitializeDstRegisters(
    IntType m, IntType n, IntType vlw)
{
    // Initialize Dst matrix registers
    ARMCG::Label loadC;
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
    for (IntType i = 0; i < std::min(m, vlw); ++i) {
        // 2 - use even numbered Z registers as tmp, 12 - number of first w register
        // 4 - max imm in MOVA instruction
        Mova(ARMCG::ZARegHS(0), ARMCG::WReg(NUM_12 + i / NUM_4), i % NUM_4,
            p1, ARMCG::ZRegS(1));
        if (n > vlw) {
            // 2 - use odd numbered Z registers as tmp, 12 - number of first w register
            // 4 - max imm in MOVA instruction
            Mova(ARMCG::ZARegHS(NUM_1), ARMCG::WReg(NUM_12 + i / NUM_4), i % NUM_4,
                p3, ARMCG::ZRegS(1));
        }
    }
    for (IntType i = 0; i < std::min(m - vlw, vlw); ++i) {
        // 2 - use even numbered Z registers as tmp, 12 - number of first w register
        // 4 - max imm in MOVA instruction
        Mova(ARMCG::ZARegHS(NUM_2), ARMCG::WReg(NUM_12 + i / NUM_4), i % NUM_4,
            p1, ARMCG::ZRegS(1));
        if (n > vlw) {
            // 2 - use odd numbered Z registers as tmp, 12 - number of first w register
            // 4 - max imm in MOVA instruction
            Mova(ARMCG::ZARegHS(NUM_3), ARMCG::WReg(NUM_12 + i / NUM_4), i % NUM_4, p3, ARMCG::ZRegS(1));
        }
    }

    L(loadBiasN);
    Tbz(flags, 5, loadBiasM); // 5-bit correspond to LOAD_BIAS_N
    Ld1w(ARMCG::ZRegS(0), p1, Ptr(biaPtrCopy)); // z1 is used as tmp register here
    if (n > vlw) {
        Ld1w(ARMCG::ZRegS(1), p3, Ptr(biaPtrCopy, NUM_1, ARMCG::ExtMod::MUL_VL)); // z1 is used as tmp register here
    }
    for (IntType i = 0; i < std::min(m, vlw); ++i) {
        // 2 - use even numbered Z registers as tmp, 12 - number of first w register
        // 4 - max imm in MOVA instruction
        Mova(ARMCG::ZARegHS(0), ARMCG::WReg(NUM_12 + i / NUM_4), i % NUM_4,
            p1, ARMCG::ZRegS(0));
        if (n > vlw) {
            // 2 - use odd numbered Z registers as tmp, 12 - number of first w register
            // 4 - max imm in MOVA instruction
            Mova(ARMCG::ZARegHS(NUM_1), ARMCG::WReg(NUM_12 + i / NUM_4), i % NUM_4,
                p3, ARMCG::ZRegS(1));
        }
    }
    for (IntType i = 0; i < std::min(m - vlw, vlw); ++i) {
        // 2 - use even numbered Z registers as tmp, 12 - number of first w register
        // 4 - max imm in MOVA instruction
        Mova(ARMCG::ZARegHS(NUM_2), ARMCG::WReg(NUM_12 + i / NUM_4), i % NUM_4,
            p1, ARMCG::ZRegS(0));
        if (n > vlw) {
            // 2 - use odd numbered Z registers as tmp, 12 - number of first w register
            // 4 - max imm in MOVA instruction
            Mova(ARMCG::ZARegHS(NUM_3), ARMCG::WReg(NUM_12 + i / NUM_4), i % NUM_4, p3, ARMCG::ZRegS(1));
        }
    }

    L(loadBiasM);
    Tbz(flags, 4, loadBiasMN); // 4-bit correspond to LOAD_BIAS_M
    Ld1w(ARMCG::ZRegS(0), p0, Ptr(biaPtrCopy)); // z1 is used as tmp register here
    if (m > vlw) {
        Ld1w(ARMCG::ZRegS(1), p2, Ptr(biaPtrCopy, NUM_1, ARMCG::ExtMod::MUL_VL)); // z1 is used as tmp register here
    }
    for (IntType i = 0; i < std::min(n, vlw); ++i) {
        // 2 - use even numbered Z registers as tmp, 12 - number of first w register
        // 4 - max imm in MOVA instruction
        Mova(ARMCG::ZARegVS(0), ARMCG::WReg(NUM_12 + i / NUM_4), i % NUM_4,
            p0, ARMCG::ZRegS(0));
        if (m > vlw) {
            // 2 - use odd numbered Z registers as tmp, 12 - number of first w register
            // 4 - max imm in MOVA instruction
            Mova(ARMCG::ZARegVS(NUM_2), ARMCG::WReg(NUM_12 + i / NUM_4), i % NUM_4,
                p2, ARMCG::ZRegS(1));
        }
    }
    for (IntType i = 0; i < std::min(n - vlw, vlw); ++i) {
        // 2 - use even numbered Z registers as tmp, 12 - number of first w register
        // 4 - max imm in MOVA instruction
        Mova(ARMCG::ZARegVS(NUM_1), ARMCG::WReg(NUM_12 + i / NUM_4), i % NUM_4,
            p0, ARMCG::ZRegS(0));
        if (m > vlw) {
            // 2 - use odd numbered Z registers as tmp, 12 - number of first w register
            // 4 - max imm in MOVA instruction
            Mova(ARMCG::ZARegVS(NUM_3), ARMCG::WReg(NUM_12 + i / NUM_4), i % NUM_4, p2, ARMCG::ZRegS(1));
        }
    }

    L(loadBiasMN);
    Tbz(flags, 6, loadC); // 6-bit correspond to LOAD_BIAS_MN
    for (IntType i = 0; i < std::min(m, vlw); ++i) {
        if (i == 0) {
            KUDNN_JIT_Generator::Mov(biaPtrCopy, biaPtr);
        } else {
            Add(biaPtrCopy, biaPtrCopy, biaOffset);
        }
        // 2 - use even numbered Z registers as tmp, 12 - number of first w register
        // 4 - max imm in MOVA instruction
        Ld1w(ARMCG::ZRegS(i * NUM_2), p1, Ptr(biaPtrCopy));
        Mova(ARMCG::ZARegHS(0), ARMCG::WReg(NUM_12 + i / NUM_4), i % NUM_4,
            p1, ARMCG::ZRegS(i * NUM_2));
        if (n > vlw) {
            // 2 - use odd numbered Z registers as tmp, 12 - number of first w register
            // 4 - max imm in MOVA instruction
            Ld1w(ARMCG::ZRegS(i * NUM_2 + NUM_1), p3, Ptr(biaPtrCopy, NUM_1, ARMCG::ExtMod::MUL_VL));
            Mova(ARMCG::ZARegHS(NUM_1), ARMCG::WReg(NUM_12 + i / NUM_4), i % NUM_4,
                p3, ARMCG::ZRegS(i * NUM_2 + NUM_1));
        }
    }

    for (IntType i = 0; i < std::min(m - vlw, vlw); ++i) {
        Add(biaPtrCopy, biaPtrCopy, biaOffset);
        // 2 - use even numbered Z registers as tmp, 12 - number of first w register
        // 4 - max imm in MOVA instruction
        Ld1w(ARMCG::ZRegS(i * NUM_2), p1, Ptr(biaPtrCopy));
        Mova(ARMCG::ZARegHS(NUM_2), ARMCG::WReg(NUM_12 + i / NUM_4), i % NUM_4,
            p1, ARMCG::ZRegS(i * NUM_2));
        if (n > vlw) {
            // 2 - use odd numbered Z registers as tmp, 12 - number of first w register
            // 4 - max imm in MOVA instruction
            Ld1w(ARMCG::ZRegS(i * NUM_2 + NUM_1), p3, Ptr(biaPtrCopy, NUM_1, ARMCG::ExtMod::MUL_VL));
            Mova(ARMCG::ZARegHS(NUM_3), ARMCG::WReg(NUM_12 + i / NUM_4), i % NUM_4,
                p3, ARMCG::ZRegS(i * NUM_2 + NUM_1));
        }
    }

    L(loadC);
    Tbz(flags, 1, addBias1); // 1-bit correspond to LOAD_C
    for (IntType i = 0; i < std::min(m, vlw); ++i) {
        if (i == 0) {
            KUDNN_JIT_Generator::Mov(dstPtrCopy, dstPtr);
        } else {
            Add(dstPtrCopy, dstPtrCopy, dstOffset);
        }
        // 2 - use even numbered Z registers as tmp, 12 - number of first w register
        // 4 - max imm in MOVA instruction
        Ld1w(ARMCG::ZRegS(i * NUM_2), p1, Ptr(dstPtrCopy));
        Mova(ARMCG::ZARegHS(0), ARMCG::WReg(NUM_12 + i / NUM_4), i % NUM_4,
            p1, ARMCG::ZRegS(i * NUM_2));
        if (n > vlw) {
            // 2 - use odd numbered Z registers as tmp, 12 - number of first w register
            // 4 - max imm in MOVA instruction
            Ld1w(ARMCG::ZRegS(i * NUM_2 + NUM_1), p3, Ptr(dstPtrCopy, NUM_1, ARMCG::ExtMod::MUL_VL));
            Mova(ARMCG::ZARegHS(NUM_1), ARMCG::WReg(NUM_12 + i / NUM_4), i % NUM_4,
                p3, ARMCG::ZRegS(i * NUM_2 + NUM_1));
        }
    }
    for (IntType i = 0; i < std::min(m - vlw, vlw); ++i) {
        Add(dstPtrCopy, dstPtrCopy, dstOffset);
        // 2 - use even numbered Z registers as tmp, 12 - number of first w register
        // 4 - max imm in MOVA instruction
        Ld1w(ARMCG::ZRegS(i * NUM_2), p1, Ptr(dstPtrCopy));
        Mova(ARMCG::ZARegHS(NUM_2), ARMCG::WReg(NUM_12 + i / NUM_4), i % NUM_4,
            p1, ARMCG::ZRegS(i * NUM_2));
        if (n > vlw) {
            // 2 - use odd numbered Z registers as tmp, 12 - number of first w register
            // 4 - max imm in MOVA instruction
            Ld1w(ARMCG::ZRegS(i * NUM_2 + NUM_1), p3, Ptr(dstPtrCopy, NUM_1, ARMCG::ExtMod::MUL_VL));
            Mova(ARMCG::ZARegHS(NUM_3), ARMCG::WReg(NUM_12 + i / NUM_4), i % NUM_4,
                p3, ARMCG::ZRegS(i * NUM_2 + NUM_1));
        }
    }

    L(addBias1);
    Tbz(flags, 7, addBiasN); // 7-bit correspond to Add_BIAS_1
    Ldr(tmpW, ARMCG::AdrUimm(Ptr(biaPtr)));
    Cpy(ARMCG::ZRegS(1), p5, tmpW); // z1 is used as tmp register here
    for (IntType i = 0; i < std::min(m, vlw); ++i) {
        if (i == 0) {
            KUDNN_JIT_Generator::Mov(dstPtrCopy, dstPtr);
        } else {
            Add(dstPtrCopy, dstPtrCopy, dstOffset);
        }
        // 2 - use even numbered Z registers as tmp, 12 - number of first w register
        // 4 - max imm in MOVA instruction
        Ld1w(ARMCG::ZRegS(i * NUM_2), p1, Ptr(dstPtrCopy));
        if (i == 0) {
            Add(ARMCG::ZRegS(i * NUM_2), ARMCG::ZRegS(i * NUM_2), ARMCG::ZRegS(1));
        } else {
            Add(ARMCG::ZRegS(i * NUM_2), ARMCG::ZRegS(i * NUM_2), ARMCG::ZRegS(0));
        }
        Mova(ARMCG::ZARegHS(0), ARMCG::WReg(NUM_12 + i / NUM_4), i % NUM_4,
            p1, ARMCG::ZRegS(i * NUM_2));
        if (i == 0) {
            KUDNN_JIT_Generator::Mov(ARMCG::ZRegS(0), p5, ARMCG::ZRegS(1));
        }
        if (n > vlw) {
            // 2 - use odd numbered Z registers as tmp, 12 - number of first w register
            // 4 - max imm in MOVA instruction
            Ld1w(ARMCG::ZRegS(i * NUM_2 + NUM_1), p3, Ptr(dstPtrCopy, NUM_1, ARMCG::ExtMod::MUL_VL));
            Add(ARMCG::ZRegS(i * NUM_2 + NUM_1), ARMCG::ZRegS(i * NUM_2 + NUM_1), ARMCG::ZRegS(0));
            Mova(ARMCG::ZARegHS(NUM_1), ARMCG::WReg(NUM_12 + i / NUM_4), i % NUM_4,
                p3, ARMCG::ZRegS(i * NUM_2 + NUM_1));
        }
    }
    Cpy(ARMCG::ZRegS(1), p5, tmpW); // z1 is used as tmp register here
    for (IntType i = 0; i < std::min(m - vlw, vlw); ++i) {
        Add(dstPtrCopy, dstPtrCopy, dstOffset);
        // 2 - use even numbered Z registers as tmp, 12 - number of first w register
        // 4 - max imm in MOVA instruction
        Ld1w(ARMCG::ZRegS(i * NUM_2), p1, Ptr(dstPtrCopy));
        if (i == 0) {
            Add(ARMCG::ZRegS(i * NUM_2), ARMCG::ZRegS(i * NUM_2), ARMCG::ZRegS(1));
        } else {
            Add(ARMCG::ZRegS(i * NUM_2), ARMCG::ZRegS(i * NUM_2), ARMCG::ZRegS(0));
        }
        Mova(ARMCG::ZARegHS(NUM_2), ARMCG::WReg(NUM_12 + i / NUM_4), i % NUM_4,
            p1, ARMCG::ZRegS(i * NUM_2));
        if (i == 0) {
            KUDNN_JIT_Generator::Mov(ARMCG::ZRegS(0), p5, ARMCG::ZRegS(1));
        }
        if (n > vlw) {
            // 2 - use odd numbered Z registers as tmp, 12 - number of first w register
            // 4 - max imm in MOVA instruction
            Ld1w(ARMCG::ZRegS(i * NUM_2 + NUM_1), p3, Ptr(dstPtrCopy, NUM_1, ARMCG::ExtMod::MUL_VL));
            Add(ARMCG::ZRegS(i * NUM_2 + NUM_1), ARMCG::ZRegS(i * NUM_2 + NUM_1), ARMCG::ZRegS(0));
            Mova(ARMCG::ZARegHS(NUM_3), ARMCG::WReg(NUM_12 + i / NUM_4), i % NUM_4,
                p3, ARMCG::ZRegS(i * NUM_2 + NUM_1));
        }
    }

    L(addBiasN);
    Tbz(flags, 9, addBiasM); // 9-bit correspond to Add_BIAS_N
    Ld1w(ARMCG::ZRegS(30), p1, Ptr(biaPtrCopy)); // z30 is used as tmp register here
    if (n > vlw) {
        Ld1w(ARMCG::ZRegS(31), p3, Ptr(biaPtrCopy, NUM_1, ARMCG::ExtMod::MUL_VL)); // z31 is used as tmp here
    }
    for (IntType i = 0; i < std::min(m, vlw); ++i) {
        if (i == 0) {
            KUDNN_JIT_Generator::Mov(dstPtrCopy, dstPtr);
        } else {
            Add(dstPtrCopy, dstPtrCopy, dstOffset);
        }
        // 2 - use even numbered Z registers as tmp, 12 - number of first w register
        // 4 - max imm in MOVA instruction
        Ld1w(ARMCG::ZRegS(i * NUM_2 % NUM_30), p1, Ptr(dstPtrCopy));
        Add(ARMCG::ZRegS(i * NUM_2 % NUM_30), ARMCG::ZRegS(i * NUM_2 % NUM_30),
            ARMCG::ZRegS(NUM_30));
        Mova(ARMCG::ZARegHS(0), ARMCG::WReg(NUM_12 + i / NUM_4), i % NUM_4,
            p1, ARMCG::ZRegS(i * NUM_2 % NUM_30));
        if (n > vlw) {
            // 2 - use odd numbered Z registers as tmp, 12 - number of first w register
            // 4 - max imm in MOVA instruction
            Ld1w(ARMCG::ZRegS((i * NUM_2 + NUM_1) % NUM_30), p3, Ptr(dstPtrCopy, NUM_1, ARMCG::ExtMod::MUL_VL));
            Add(ARMCG::ZRegS((i * NUM_2 + NUM_1) % NUM_30), ARMCG::ZRegS((i * NUM_2 + NUM_1) % NUM_30),
                ARMCG::ZRegS(NUM_31));
            Mova(ARMCG::ZARegHS(NUM_1), ARMCG::WReg(NUM_12 + i / NUM_4), i % NUM_4,
                p3, ARMCG::ZRegS((i * NUM_2 + NUM_1) % NUM_30));
        }
    }
    Cpy(ARMCG::ZRegS(1), p5, tmpW); // z1 is used as tmp register here
    for (IntType i = 0; i < std::min(m - vlw, vlw); ++i) {
        Add(dstPtrCopy, dstPtrCopy, dstOffset);
        // 2 - use even numbered Z registers as tmp, 12 - number of first w register
        // 4 - max imm in MOVA instruction
        Ld1w(ARMCG::ZRegS(i * NUM_2 % NUM_30), p1, Ptr(dstPtrCopy));
        Add(ARMCG::ZRegS(i * NUM_2 % NUM_30), ARMCG::ZRegS(i * NUM_2 % NUM_30),
            ARMCG::ZRegS(NUM_30)); // z30 contains bias value
        Mova(ARMCG::ZARegHS(NUM_2), ARMCG::WReg(NUM_12 + i / NUM_4), i % NUM_4,
            p1, ARMCG::ZRegS(i * NUM_2 % NUM_30));
        if (n > vlw) {
            // 2 - use odd numbered Z registers as tmp, 12 - number of first w register
            // 4 - max imm in MOVA instruction
            Ld1w(ARMCG::ZRegS((i * NUM_2 + NUM_1) % NUM_30), p3, Ptr(dstPtrCopy, NUM_1, ARMCG::ExtMod::MUL_VL));
            Add(ARMCG::ZRegS((i * NUM_2 + NUM_1) % NUM_30), ARMCG::ZRegS((i * NUM_2 + NUM_1) % NUM_30),
                ARMCG::ZRegS(NUM_31));
            Mova(ARMCG::ZARegHS(NUM_3), ARMCG::WReg(NUM_12 + i / NUM_4), i % NUM_4,
                p3, ARMCG::ZRegS((i * NUM_2 + NUM_1) % NUM_30));
        }
    }

    L(addBiasM);
    Tbz(flags, 8, addBiasMN); // 8-bit correspond to LOAD_BIAS_M
    for (IntType i = 0; i < std::min(m, vlw); ++i) {
        if (i == 0) {
            KUDNN_JIT_Generator::Mov(dstPtrCopy, dstPtr);
            KUDNN_JIT_Generator::Mov(biaPtrCopy, biaPtr);
        } else {
            Add(dstPtrCopy, dstPtrCopy, dstOffset);
            Add(biaPtrCopy, biaPtrCopy, NUM_2 * static_cast<std::uint32_t>(sizeof(BiaType)));
        }
        // 2 - use even numbered Z registers as tmp, 12 - number of first w register
        // 4 - max imm in MOVA instruction
        Ldr(tmpW, ARMCG::AdrUimm(Ptr(biaPtrCopy)));
        Cpy(ARMCG::ZRegS((i * NUM_4 + NUM_1) % NUM_32), p5, tmpW);
        Ld1w(ARMCG::ZRegS((i * NUM_4) % NUM_32), p1, Ptr(dstPtrCopy));
        Add(ARMCG::ZRegS((i * NUM_4) % NUM_32), ARMCG::ZRegS((i * NUM_4) % NUM_32),
            ARMCG::ZRegS((i * NUM_4 + NUM_1) % NUM_32));
        Mova(ARMCG::ZARegHS(0), ARMCG::WReg(NUM_12 + i / NUM_4), i % NUM_4,
            p1, ARMCG::ZRegS((i * NUM_4) % NUM_32));
        if (n > vlw) {
            // 2 - use odd numbered Z registers as tmp, 12 - number of first w register
            // 4 - max imm in MOVA instruction
            Ldr(tmpW, Ptr(biaPtrCopy, static_cast<std::uint32_t>(sizeof(BiaType))));
            Cpy(ARMCG::ZRegS((i * NUM_4 + NUM_3) % NUM_32), p5, tmpW);
            Ld1w(ARMCG::ZRegS((i * NUM_4 + NUM_2) % NUM_32), p3, Ptr(dstPtrCopy, NUM_1, ARMCG::ExtMod::MUL_VL));
            Add(ARMCG::ZRegS((i * NUM_4 + NUM_2) % NUM_32), ARMCG::ZRegS((i * NUM_4 + NUM_2) % NUM_32),
                ARMCG::ZRegS((i * NUM_4 + NUM_3) % NUM_32));
            Mova(ARMCG::ZARegHS(NUM_1), ARMCG::WReg(NUM_12 + i / NUM_4), i % NUM_4,
                p3, ARMCG::ZRegS((i * NUM_4 + NUM_2) % NUM_32));
        }
    }
    for (IntType i = 0; i < std::min(m - vlw, vlw); ++i) {
        Add(dstPtrCopy, dstPtrCopy, dstOffset);
        Add(biaPtrCopy, biaPtrCopy, static_cast<std::uint32_t>(sizeof(BiaType)));
        // 2 - use even numbered Z registers as tmp, 12 - number of first w register
        // 4 - max imm in MOVA instruction
        Ldr(tmpW, ARMCG::AdrUimm(Ptr(biaPtrCopy)));
        Cpy(ARMCG::ZRegS((i * NUM_4 + NUM_1) % NUM_32), p5, tmpW);
        Ld1w(ARMCG::ZRegS((i * NUM_4) % NUM_32), p1, Ptr(dstPtrCopy));
        Add(ARMCG::ZRegS((i * NUM_4) % NUM_32), ARMCG::ZRegS((i * NUM_4) % NUM_32),
            ARMCG::ZRegS((i * NUM_4 + NUM_1) % NUM_32));
        Mova(ARMCG::ZARegHS(NUM_2), ARMCG::WReg(NUM_12 + i / NUM_4), i % NUM_4,
            p1, ARMCG::ZRegS((i * NUM_4) % NUM_32));
        if (n > vlw) {
            // 2 - use odd numbered Z registers as tmp, 12 - number of first w register
            // 4 - max imm in MOVA instruction
            Ldr(tmpW, Ptr(biaPtrCopy, static_cast<std::uint32_t>(sizeof(BiaType))));
            Cpy(ARMCG::ZRegS((i * NUM_4 + NUM_3) % NUM_32), p5, tmpW);
            Ld1w(ARMCG::ZRegS((i * NUM_4 + NUM_2) % NUM_32), p3, Ptr(dstPtrCopy, NUM_1, ARMCG::ExtMod::MUL_VL));
            Add(ARMCG::ZRegS((i * NUM_4 + NUM_2) % NUM_32), ARMCG::ZRegS((i * NUM_4 + NUM_2) % NUM_32),
                ARMCG::ZRegS((i * NUM_4 + NUM_3) % NUM_32));
            Mova(ARMCG::ZARegHS(NUM_3), ARMCG::WReg(NUM_12 + i / NUM_4), i % NUM_4,
                p3, ARMCG::ZRegS((i * NUM_4 + NUM_2) % NUM_32));
        }
    }

    L(addBiasMN);
    Tbz(flags, 10, initDone); // 10-bit correspond to LOAD_BIAS_MN
    for (IntType i = 0; i < std::min(m, vlw); ++i) {
        if (i == 0) {
            KUDNN_JIT_Generator::Mov(dstPtrCopy, dstPtr);
            KUDNN_JIT_Generator::Mov(biaPtrCopy, biaPtr);
        } else {
            Add(dstPtrCopy, dstPtrCopy, dstOffset);
            Add(biaPtrCopy, biaPtrCopy, biaOffset);
        }
        // 2 - use even numbered Z registers as tmp, 12 - number of first w register
        // 4 - max imm in MOVA instruction
        Ld1w(ARMCG::ZRegS((i * NUM_4) % NUM_32), p1, Ptr(dstPtrCopy));
        Ld1w(ARMCG::ZRegS((i * NUM_4 + NUM_1) % NUM_32), p1, Ptr(biaPtrCopy));
        Add(ARMCG::ZRegS((i * NUM_4) % NUM_32), ARMCG::ZRegS((i * NUM_4) % NUM_32),
            ARMCG::ZRegS((i * NUM_4 + NUM_1) % NUM_32));
        Mova(ARMCG::ZARegHS(0), ARMCG::WReg(NUM_12 + i / NUM_4), i % NUM_4,
            p1, ARMCG::ZRegS((i * NUM_4) % NUM_32));
        if (n > vlw) {
            // 2 - use odd numbered Z registers as tmp, 12 - number of first w register
            // 4 - max imm in MOVA instruction
            Ld1w(ARMCG::ZRegS((i * NUM_4 + NUM_2) % NUM_32), p3, Ptr(dstPtrCopy, NUM_1, ARMCG::ExtMod::MUL_VL));
            Ld1w(ARMCG::ZRegS((i * NUM_4 + NUM_3) % NUM_32), p3, Ptr(biaPtrCopy, NUM_1, ARMCG::ExtMod::MUL_VL));
            Add(ARMCG::ZRegS((i * NUM_4 + NUM_2) % NUM_32), ARMCG::ZRegS((i * NUM_4 + NUM_2) % NUM_32),
                ARMCG::ZRegS((i * NUM_4 + NUM_3) % NUM_32));
            Mova(ARMCG::ZARegHS(NUM_1), ARMCG::WReg(NUM_12 + i / NUM_4), i % NUM_4,
                p3, ARMCG::ZRegS((i * NUM_4 + NUM_2) % NUM_32));
        }
    }
    for (IntType i = 0; i < std::min(m - vlw, vlw); ++i) {
        Add(dstPtrCopy, dstPtrCopy, dstOffset);
        Add(biaPtrCopy, biaPtrCopy, biaOffset);
        // 2 - use even numbered Z registers as tmp, 12 - number of first w register
        // 4 - max imm in MOVA instruction
        Ld1w(ARMCG::ZRegS((i * NUM_4) % NUM_32), p1, Ptr(dstPtrCopy));
        Ld1w(ARMCG::ZRegS((i * NUM_4 + NUM_1) % NUM_32), p1, Ptr(biaPtrCopy));
        Add(ARMCG::ZRegS((i * NUM_4) % NUM_32), ARMCG::ZRegS((i * NUM_4) % NUM_32),
            ARMCG::ZRegS((i * NUM_4 + NUM_1) % NUM_32));
        Mova(ARMCG::ZARegHS(NUM_2), ARMCG::WReg(NUM_12 + i / NUM_4), i % NUM_4,
            p1, ARMCG::ZRegS((i * NUM_4) % NUM_32));
        if (n > vlw) {
            // 2 - use odd numbered Z registers as tmp, 12 - number of first w register
            // 4 - max imm in MOVA instruction
            Ld1w(ARMCG::ZRegS((i * NUM_4 + NUM_2) % NUM_32), p3, Ptr(dstPtrCopy, NUM_1, ARMCG::ExtMod::MUL_VL));
            Ld1w(ARMCG::ZRegS((i * NUM_4 + NUM_3) % NUM_32), p3, Ptr(biaPtrCopy, NUM_1, ARMCG::ExtMod::MUL_VL));
            Add(ARMCG::ZRegS((i * NUM_4 + NUM_2) % NUM_32), ARMCG::ZRegS((i * NUM_4 + NUM_2) % NUM_32),
                ARMCG::ZRegS((i * NUM_4 + NUM_3) % NUM_32));
            Mova(ARMCG::ZARegHS(NUM_3), ARMCG::WReg(NUM_12 + i / NUM_4), i % NUM_4,
                p3, ARMCG::ZRegS((i * NUM_4 + NUM_2) % NUM_32));
        }
    }

    L(initDone);
}

template <typename SrcType, typename WeiType, typename DstType, typename BiaType>
void MatMulKernelDefaultCodeGenImpl<ARCH::MatrixComputation, SrcType, WeiType, DstType, BiaType,
    typename std::enable_if<IsIntegerMatmul<SrcType, WeiType, DstType, BiaType>(),
    MatmulDummyType>::type>::GenerateKLoop(
    IntType m, IntType n, IntType k, IntType vlw)
{
    // NUM_4 due to peculiarities of INT8 MOPA instruction
    IntType kRoundedUp = Math::CeilDiv(k, NUM_4 * kLoopUnroll) * NUM_4 * kLoopUnroll;
    ARMCG::Label kLoopHead;
    KUDNN_JIT_Generator::Mov(kTmpCopy, kVal);
    Subs(kTmpCopy, kTmpCopy, kLoopUnroll * NUM_4); // NUM_4 due to peculiarities of INT8 MOPA instruction
    // p0, p2 - for SRC, p1, p3 - for WEI
    // z0, z2, z4, z6 - for SRC, z1, z3, z5, z7 - for WEI
    // SRC and WEI data is prepacked, so it's continious in memory.
    // K unroll by 2 requires 4 elements from SRC,  4 from WEI
    // Load src data
    Ld1b(ARMCG::ZRegB(NUM_0), p0, Ptr(srcPtrCopy));
    if (m > vlw) {
        Ld1b(ARMCG::ZRegB(NUM_2), p2, Ptr(srcPtrCopy, NUM_1, ARMCG::ExtMod::MUL_VL));
    }
    // Load wei data
    Ld1b(ARMCG::ZRegB(NUM_1), p1, Ptr(weiPtrCopy));
    if (n > vlw) {
        Ld1b(ARMCG::ZRegB(NUM_3), p3, Ptr(weiPtrCopy, NUM_1, ARMCG::ExtMod::MUL_VL));
    }
    if (kRoundedUp >= (kLoopUnroll * NUM_4)) { // NUM_4 due to peculiarities of INT8 MOPA instruction
        // Load src data for unrolled loop
        Ld1b(ARMCG::ZRegB(NUM_4), p0, Ptr(srcPtrCopy, NUM_2, ARMCG::ExtMod::MUL_VL));
        if (m > vlw) {
            Ld1b(ARMCG::ZRegB(NUM_6), p2, Ptr(srcPtrCopy, NUM_3, ARMCG::ExtMod::MUL_VL));
        }
        // Load wei data unrolled loop
        Ld1b(ARMCG::ZRegB(NUM_5), p1, Ptr(weiPtrCopy, NUM_2, ARMCG::ExtMod::MUL_VL));
        if (n > vlw) {
            Ld1b(ARMCG::ZRegB(NUM_7), p3, Ptr(weiPtrCopy, NUM_3, ARMCG::ExtMod::MUL_VL));
        }
        Addvl(srcPtrCopy, srcPtrCopy, kernelMSize * kLoopUnroll);
        Addvl(weiPtrCopy, weiPtrCopy, kernelNSize * kLoopUnroll);
        if (kRoundedUp > (kLoopUnroll * NUM_4)) { // NUM_4 due to peculiarities of INT8 MOPA instruction
            L(kLoopHead);
                Int8Mopa(ARMCG::ZARegS(NUM_0), p0, p1, ARMCG::ZRegB(NUM_0), ARMCG::ZRegB(NUM_1));
                Subs(kTmpCopy, kTmpCopy, (kLoopUnroll * NUM_4)); // NUM_4 due to peculiarities of INT8 MOPA instruction
                if (n > vlw) {
                    Int8Mopa(ARMCG::ZARegS(NUM_1), p0, p3, ARMCG::ZRegB(NUM_0), ARMCG::ZRegB(NUM_3));
                }
                Ld1b(ARMCG::ZRegB(NUM_0), p0, Ptr(srcPtrCopy));
                if (m > vlw) {
                    Int8Mopa(ARMCG::ZARegS(NUM_2), p2, p1, ARMCG::ZRegB(NUM_2), ARMCG::ZRegB(NUM_1));
                }
                Ld1b(ARMCG::ZRegB(NUM_1), p1, Ptr(weiPtrCopy));
                if ((m > vlw) && (n > vlw)) {
                    Int8Mopa(ARMCG::ZARegS(NUM_3), p2, p3, ARMCG::ZRegB(NUM_2), ARMCG::ZRegB(NUM_3));
                }
                if (m > vlw) {
                    Ld1b(ARMCG::ZRegB(NUM_2), p2, Ptr(srcPtrCopy, NUM_1, ARMCG::ExtMod::MUL_VL));
                }
                if (n > vlw) {
                    Ld1b(ARMCG::ZRegB(NUM_3), p3, Ptr(weiPtrCopy, NUM_1, ARMCG::ExtMod::MUL_VL));
                }

                Int8Mopa(ARMCG::ZARegS(NUM_0), p0, p1, ARMCG::ZRegB(NUM_4), ARMCG::ZRegB(NUM_5));
                if (n > vlw) {
                    Int8Mopa(ARMCG::ZARegS(NUM_1), p0, p3, ARMCG::ZRegB(NUM_4), ARMCG::ZRegB(NUM_7));
                }
                Ld1b(ARMCG::ZRegB(NUM_4), p0, Ptr(srcPtrCopy, NUM_2, ARMCG::ExtMod::MUL_VL));
                if (m > vlw) {
                    Int8Mopa(ARMCG::ZARegS(NUM_2), p2, p1, ARMCG::ZRegB(NUM_6), ARMCG::ZRegB(NUM_5));
                }
                Ld1b(ARMCG::ZRegB(NUM_5), p1, Ptr(weiPtrCopy, NUM_2, ARMCG::ExtMod::MUL_VL));
                if ((m > vlw) && (n > vlw)) {
                    Int8Mopa(ARMCG::ZARegS(NUM_3), p2, p3, ARMCG::ZRegB(NUM_6), ARMCG::ZRegB(NUM_7));
                }
                if (m > vlw) {
                    Ld1b(ARMCG::ZRegB(NUM_6), p2, Ptr(srcPtrCopy, NUM_3, ARMCG::ExtMod::MUL_VL));
                }
                if (n > vlw) {
                    Ld1b(ARMCG::ZRegB(NUM_7), p3, Ptr(weiPtrCopy, NUM_3, ARMCG::ExtMod::MUL_VL));
                }

                Addvl(srcPtrCopy, srcPtrCopy, kernelMSize * kLoopUnroll);
                Addvl(weiPtrCopy, weiPtrCopy, kernelNSize * kLoopUnroll);

            B(ARMCG::Cond::GT, kLoopHead);
        }
    }

    Int8Mopa(ARMCG::ZARegS(NUM_0), p0, p1, ARMCG::ZRegB(NUM_0), ARMCG::ZRegB(NUM_1));
    if (n > vlw) {
        Int8Mopa(ARMCG::ZARegS(NUM_1), p0, p3, ARMCG::ZRegB(NUM_0), ARMCG::ZRegB(NUM_3));
    }
    if (m > vlw) {
        Int8Mopa(ARMCG::ZARegS(NUM_2), p2, p1, ARMCG::ZRegB(NUM_2), ARMCG::ZRegB(NUM_1));
    }
    if ((m > vlw) && (n > vlw)) {
        Int8Mopa(ARMCG::ZARegS(NUM_3), p2, p3, ARMCG::ZRegB(NUM_2), ARMCG::ZRegB(NUM_3));
    }
    // NUM_4 due to peculiarities of INT8 MOPA instruction
    if ((k == kRoundedUp) || (k % (kLoopUnroll * NUM_4) > NUM_4)) {
        Int8Mopa(ARMCG::ZARegS(NUM_0), p0, p1, ARMCG::ZRegB(NUM_4), ARMCG::ZRegB(NUM_5));
        if (n > vlw) {
            Int8Mopa(ARMCG::ZARegS(NUM_1), p0, p3, ARMCG::ZRegB(NUM_4), ARMCG::ZRegB(NUM_7));
        }
        if (m > vlw) {
            Int8Mopa(ARMCG::ZARegS(NUM_2), p2, p1, ARMCG::ZRegB(NUM_6), ARMCG::ZRegB(NUM_5));
        }
        if ((m > vlw) && (n > vlw)) {
            Int8Mopa(ARMCG::ZARegS(NUM_3), p2, p3, ARMCG::ZRegB(NUM_6), ARMCG::ZRegB(NUM_7));
        }
    }
}

template <typename SrcType, typename WeiType, typename DstType, typename BiaType>
void MatMulKernelDefaultCodeGenImpl<ARCH::MatrixComputation, SrcType, WeiType, DstType, BiaType,
    typename std::enable_if<IsIntegerMatmul<SrcType, WeiType, DstType, BiaType>(),
    MatmulDummyType>::type>::StoreDstRegisters(
    IntType m, IntType n, IntType vlw)
{
    for (IntType i = 0; i < std::min(m, vlw); ++i) {
        if (i == 0) {
            KUDNN_JIT_Generator::Mov(dstPtrCopy, dstPtr);
        } else {
            Add(dstPtrCopy, dstPtrCopy, dstOffset);
        }
        // 2 - use even numbered Z registers as tmp, 12 - number of first w register
        // 4 - max imm in MOVA instruction
        Mova(ARMCG::ZRegS(i * NUM_2), p1, ARMCG::ZARegHS(0),
            ARMCG::WReg(NUM_12 + i / NUM_4), i % NUM_4);
            St1w(ARMCG::ZRegS(i * NUM_2), p1, Ptr(dstPtrCopy));
        if (n > vlw) {
            // 2 - use odd numbered Z registers as tmp, 12 - number of first w register
            // 4 - max imm in MOVA instruction
            Mova(ARMCG::ZRegS(i * NUM_2 + NUM_1), p3, ARMCG::ZARegHS(NUM_1),
                ARMCG::WReg(NUM_12 + i / NUM_4), i % NUM_4);
                St1w(ARMCG::ZRegS(i * NUM_2 + NUM_1), p3, Ptr(dstPtrCopy, NUM_1, ARMCG::ExtMod::MUL_VL));
        }
    }

    for (IntType i = 0; i < std::min(m - vlw, vlw); ++i) {
        Add(dstPtrCopy, dstPtrCopy, dstOffset);
        // 2 - use even numbered Z registers as tmp, 12 - number of first w register
        // 4 - max imm in MOVA instruction
        Mova(ARMCG::ZRegS(i * NUM_2), p1, ARMCG::ZARegHS(NUM_2),
            ARMCG::WReg(NUM_12 + i / NUM_4), i % NUM_4);
            St1w(ARMCG::ZRegS(i * NUM_2), p1, Ptr(dstPtrCopy));
        if (n > vlw) {
            // 2 - use odd numbered Z registers as tmp, 12 - number of first w register
            // 4 - max imm in MOVA instruction
            Mova(ARMCG::ZRegS(i * NUM_2 + NUM_1), p3, ARMCG::ZARegHS(NUM_3),
                ARMCG::WReg(NUM_12 + i / NUM_4), i % NUM_4);
                St1w(ARMCG::ZRegS(i * NUM_2 + NUM_1), p3, Ptr(dstPtrCopy, NUM_1, ARMCG::ExtMod::MUL_VL));
        }
    }
}

template <typename SrcType, typename WeiType, typename DstType, typename BiaType>
void MatMulKernelDefaultCodeGenImpl<ARCH::MatrixComputation, SrcType, WeiType, DstType, BiaType,
    typename std::enable_if<IsIntegerMatmul<SrcType, WeiType, DstType, BiaType>(),
    MatmulDummyType>::type>::Int8Mopa(const ARMCG::ZARegS &zada, const ARMCG::PRegBase &pn,
    const ARMCG::PRegBase &pm, const ARMCG::ZRegB &zn, const ARMCG::ZRegB &zm)
{
    if constexpr (std::is_same_v<SrcType, WeiType>) {
        if constexpr (std::is_same_v<SrcType, std::int8_t>) {
            Smopa(zada, pn, pm, zn, zm);
        } else {
            Umopa(zada, pn, pm, zn, zm);
        }
    } else {
        if (std::is_same_v<SrcType, std::int8_t> == wereSrcWeiSwapped) {
            Usmopa(zada, pn, pm, zn, zm);
        } else {
            Sumopa(zada, pn, pm, zn, zm);
        }
    }
}

template <typename SrcType, typename WeiType, typename DstType, typename BiaType>
MatMulImplGenerator<MatMulKernelDefaultCodeGen<ARCH::MatrixComputation,
    SrcType, WeiType, DstType, BiaType>>::DefaultImpl::DefaultImpl(
    const GemmInfo &task, IntType nThreads) : MatMulImplBase(task, nThreads)
{
    VERBOSE_PRINT;
    UpdateExecParams();
    Initialize();
}

template <typename SrcType, typename WeiType, typename DstType, typename BiaType>
std::uint32_t MatMulImplGenerator<MatMulKernelDefaultCodeGen<ARCH::MatrixComputation,
    SrcType, WeiType, DstType, BiaType>>::DefaultImpl::GetKernelMSize() noexcept
{
    return CodeGen::kernelMSize * svcntw();
}

template <typename SrcType, typename WeiType, typename DstType, typename BiaType>
std::uint32_t MatMulImplGenerator<MatMulKernelDefaultCodeGen<ARCH::MatrixComputation,
    SrcType, WeiType, DstType, BiaType>>::DefaultImpl::GetKernelNSize() noexcept
{
    return CodeGen::kernelNSize * svcntw();
}

template <typename SrcType, typename WeiType, typename DstType, typename BiaType>
void MatMulImplGenerator<MatMulKernelDefaultCodeGen<ARCH::MatrixComputation,
    SrcType, WeiType, DstType, BiaType>>::DefaultImpl::InitializeCodeGens(IntType m, IntType n, IntType k, IntType vlw,
    SizeType offsetSrc, SizeType offsetWei, SizeType offsetDst, SizeType offsetBia, bool wereSrcWeiSwapped)
{
    Service::Unused(offsetSrc, offsetWei, offsetDst); // they are actually not used by implementation
    
    IntType mSubBlockExecFull = CodeGen::kernelMSize * vlw;
    IntType mSubBlockExecResid = m % mSubBlockExecFull;
    IntType nSubBlockExecFull = CodeGen::kernelNSize * vlw;
    IntType nSubBlockExecResid = n % nSubBlockExecFull;
    IntType kBlockExecFull = CodeGen::GetKBlockSize(m, n, k);
    IntType kBlockExecResid = k % CodeGen::GetKBlockSize(m, n, k);
    CodeGenWrapper<CodeGen> cgw0;
    cgw0.Initialize(mSubBlockExecFull, nSubBlockExecFull, kBlockExecFull,
        static_cast<IntType>(offsetBia), wereSrcWeiSwapped);
    CodeGenWrapper<CodeGen> cgw1;
    cgw1.Initialize(mSubBlockExecFull, nSubBlockExecFull, kBlockExecResid,
        static_cast<IntType>(offsetBia), wereSrcWeiSwapped);
    CodeGenWrapper<CodeGen> cgw2;
    cgw2.Initialize(mSubBlockExecFull, nSubBlockExecResid, kBlockExecFull,
        static_cast<IntType>(offsetBia), wereSrcWeiSwapped);
    CodeGenWrapper<CodeGen> cgw3;
    cgw3.Initialize(mSubBlockExecFull, nSubBlockExecResid, kBlockExecResid,
        static_cast<IntType>(offsetBia), wereSrcWeiSwapped);
    CodeGenWrapper<CodeGen> cgw4;
    cgw4.Initialize(mSubBlockExecResid, nSubBlockExecFull, kBlockExecFull,
        static_cast<IntType>(offsetBia), wereSrcWeiSwapped);
    CodeGenWrapper<CodeGen> cgw5;
    cgw5.Initialize(mSubBlockExecResid, nSubBlockExecFull, kBlockExecResid,
        static_cast<IntType>(offsetBia), wereSrcWeiSwapped);
    CodeGenWrapper<CodeGen> cgw6;
    cgw6.Initialize(mSubBlockExecResid, nSubBlockExecResid, kBlockExecFull,
        static_cast<IntType>(offsetBia), wereSrcWeiSwapped);
    CodeGenWrapper<CodeGen> cgw7;
    cgw7.Initialize(mSubBlockExecResid, nSubBlockExecResid, kBlockExecResid,
        static_cast<IntType>(offsetBia), wereSrcWeiSwapped);
    perThreadImpls.push_back({std::move(cgw0), std::move(cgw1), std::move(cgw2), std::move(cgw3),
        std::move(cgw4), std::move(cgw5), std::move(cgw6), std::move(cgw7)});
}

template <typename SrcType, typename WeiType, typename DstType, typename BiaType>
IntType MatMulImplGenerator<MatMulKernelDefaultCodeGen<ARCH::MatrixComputation,
    SrcType, WeiType, DstType, BiaType>>::DefaultImpl::GetThreadBufSize(
    IntType m, IntType n, IntType k, IntType vlw) noexcept
{
    IntType vlb = NUM_4 * vlw;
    IntType mRoundedUp = Math::CeilDiv(m, CodeGen::kernelMSize * vlb) * CodeGen::kernelMSize * vlb;
    IntType nRoundedUp = Math::CeilDiv(n, CodeGen::kernelNSize * vlb) * CodeGen::kernelNSize * vlb;
    // NUM_4 is due to peculiarities of MOPA instruction for INT8 data type
    IntType kRoundedUp = Math::CeilDiv(k, CodeGen::kLoopUnroll * NUM_4) * CodeGen::kLoopUnroll * NUM_4;
    return static_cast<IntType>(std::min(CodeGen::GetKBlockSize(m, n, k), kRoundedUp) *
        (std::min(CodeGen::GetMBlockSize(m, n, k), mRoundedUp) * sizeof(SrcType) +
            std::min(CodeGen::GetNBlockSize(m, n, k), nRoundedUp) * sizeof(WeiType)));
}

template <typename SrcType, typename WeiType, typename DstType, typename BiaType>
void MatMulImplGenerator<MatMulKernelDefaultCodeGen<ARCH::MatrixComputation,
    SrcType, WeiType, DstType, BiaType>>::DefaultImpl::ReserveImplSpecificParams() {}

template <typename SrcType, typename WeiType, typename DstType, typename BiaType>
std::vector<std::pair<Shape, Shape>> MatMulImplGenerator<MatMulKernelDefaultCodeGen<ARCH::MatrixComputation,
    SrcType, WeiType, DstType, BiaType>>::DefaultImpl::SplitWork()
{
    return CalculateWorkOnThread<DefaultImpl>(taskExec.dstInfo.GetDims(), implNThreads,
        dstLayout2D == Layout::BA, taskExec.srcInfo.GetDims()[taskExec.srcInfo.GetNumDims() - 1] > 1024); // 1024
        // is a number indicating that k is big enough to omit condition mWork % kernelMSize == 0 (same for n)
}

// used for packing both Src and Wei matrices
template <typename SrcType, typename WeiType, typename DstType, typename BiaType>
void MatMulImplGenerator<MatMulKernelDefaultCodeGen<ARCH::MatrixComputation,
    SrcType, WeiType, DstType, BiaType>>::DefaultImpl::PackSimilarLt(IntType mBlock, IntType kBlock,
    const void *src, IntType offset, void *buffer, IntType bufferOffset)
{
    IntType vlb = svcntb();
    svbool_t ptrue = svptrue_b8();
    // NUM_4 is due to peculiarities of MOPA instruction for INT8 data type
    IntType kRoundedUp = ((kBlock + NUM_3) / NUM_4) * NUM_4;
    IntType k = 0;
    // NUM_4 is due to peculiarities of MOPA instruction for INT8 data type
    for (; k < kBlock - NUM_4; k += NUM_4) {
        for (IntType m = 0; m < mBlock; m += vlb) {
            svbool_t p0 = svwhilelt_b8(m + 0 * vlb, mBlock);
            svint8_t src0 = svld1(p0, static_cast<const std::int8_t*>(src) + 0 * vlb + m + offset * (k + 0));
            svint8_t src1 = svld1(p0, static_cast<const std::int8_t*>(src) + 0 * vlb + m + offset * (k + 1));
            svint8_t src2 = svld1(p0, static_cast<const std::int8_t*>(src) + 0 * vlb + m + offset * (k + 2));
            svint8_t src3 = svld1(p0, static_cast<const std::int8_t*>(src) + 0 * vlb + m + offset * (k + 3));
            svint8_t tmp0 = svzip1(src0, src1);
            svint8_t tmp1 = svzip2(src0, src1);
            svint8_t tmp2 = svzip1(src2, src3);
            svint8_t tmp3 = svzip2(src2, src3);
            svint8_t dst0 = svreinterpret_s8(svzip1(svreinterpret_s16(tmp0), svreinterpret_s16(tmp2)));
            svint8_t dst1 = svreinterpret_s8(svzip2(svreinterpret_s16(tmp0), svreinterpret_s16(tmp2)));
            svint8_t dst2 = svreinterpret_s8(svzip1(svreinterpret_s16(tmp1), svreinterpret_s16(tmp3)));
            svint8_t dst3 = svreinterpret_s8(svzip2(svreinterpret_s16(tmp1), svreinterpret_s16(tmp3)));
            svst1(ptrue, static_cast<std::int8_t *>(buffer)
                + 0 * vlb + (m + 0 * bufferOffset) * kRoundedUp + k * bufferOffset, dst0);
            svst1(ptrue, static_cast<std::int8_t *>(buffer)
                + 1 * vlb + (m + 0 * bufferOffset) * kRoundedUp + k * bufferOffset, dst1);
            svst1(ptrue, static_cast<std::int8_t *>(buffer)
                + 0 * vlb + (m + 1 * bufferOffset) * kRoundedUp + k * bufferOffset, dst2);
            svst1(ptrue, static_cast<std::int8_t *>(buffer)
                + 1 * vlb + (m + 1 * bufferOffset) * kRoundedUp + k * bufferOffset, dst3);
        }
    }
    IntType kResid = kBlock % NUM_4;
    if (kResid == 0) {
        kResid = NUM_4;
    }
    for (IntType m = 0; m < mBlock; m += vlb) {
        svbool_t p0 = svwhilelt_b8(m + 0 * vlb, mBlock);
        svint8_t src0 = svld1(p0, static_cast<const std::int8_t*>(src) + 0 * vlb + m + offset * (k + 0));
        svint8_t src1;
        if (kResid > NUM_1) {
            src1 = svld1(p0, static_cast<const std::int8_t*>(src) + 0 * vlb + m + offset * (k + 1));
        } else {
            src1 = svdup_s8(NUM_0);
        }
        svint8_t src2;
        if (kResid > NUM_2) {
            src2 = svld1(p0, static_cast<const std::int8_t*>(src) + 0 * vlb + m + offset * (k + 2));
        } else {
            src2 = svdup_s8(NUM_0);
        }
        svint8_t src3;
        if (kResid > NUM_3) {
            src3 = svld1(p0, static_cast<const std::int8_t*>(src) + 0 * vlb + m + offset * (k + 3));
        } else {
            src3 = svdup_s8(NUM_0);
        }
        svint8_t tmp0 = svzip1(src0, src1);
        svint8_t tmp1 = svzip2(src0, src1);
        svint8_t tmp2 = svzip1(src2, src3);
        svint8_t tmp3 = svzip2(src2, src3);
        svint8_t dst0 = svreinterpret_s8(svzip1(svreinterpret_s16(tmp0), svreinterpret_s16(tmp2)));
        svint8_t dst1 = svreinterpret_s8(svzip2(svreinterpret_s16(tmp0), svreinterpret_s16(tmp2)));
        svint8_t dst2 = svreinterpret_s8(svzip1(svreinterpret_s16(tmp1), svreinterpret_s16(tmp3)));
        svint8_t dst3 = svreinterpret_s8(svzip2(svreinterpret_s16(tmp1), svreinterpret_s16(tmp3)));
        svst1(ptrue, static_cast<std::int8_t *>(buffer)
            + 0 * vlb + (m + 0 * bufferOffset) * kRoundedUp + k * bufferOffset, dst0);
        svst1(ptrue, static_cast<std::int8_t *>(buffer)
            + 1 * vlb + (m + 0 * bufferOffset) * kRoundedUp + k * bufferOffset, dst1);
        svst1(ptrue, static_cast<std::int8_t *>(buffer)
            + 0 * vlb + (m + 1 * bufferOffset) * kRoundedUp + k * bufferOffset, dst2);
        svst1(ptrue, static_cast<std::int8_t *>(buffer)
            + 1 * vlb + (m + 1 * bufferOffset) * kRoundedUp + k * bufferOffset, dst3);
    }
}

// used for packing both Src and Wei matrices
template <typename SrcType, typename WeiType, typename DstType, typename BiaType>
void MatMulImplGenerator<MatMulKernelDefaultCodeGen<ARCH::MatrixComputation,
    SrcType, WeiType, DstType, BiaType>>::DefaultImpl::PackDiffLt(IntType mBlock, IntType kBlock,
    const void *src, IntType offset, void *buffer, IntType bufferOffset)
{
    IntType kBlock32 = kBlock / sizeof(float);
    IntType kBlock32RoundedUp = Math::CeilDiv(kBlock, sizeof(float));
    IntType kResid = kBlock % sizeof(float);
    svbool_t pgResid = svwhilelt_b8_s64(0, kResid);
    svbool_t pg4 = svwhilelt_b8_s64(0, sizeof(float));
    const IntType vlw = svcntw();
    for (IntType mIdx = mBlock; mIdx > 0; mIdx -= bufferOffset) {
        if (mIdx <= vlw) {
            IntType mIdxReorder = std::min(vlw, mIdx) - 1;
            for (IntType kIdx = kBlock32; kIdx > 0; kIdx -= vlw) {
                IntType kIdxReorder = std::min(vlw, kIdx) - 1;
                Reorder::TRANSPOSE_REORDER_FUNC_ARR[mIdxReorder * vlw + kIdxReorder](
                    Service::AddOffset<const float *>(src,
                        (mBlock - mIdx) * offset + (kBlock32 - kIdx) * sizeof(float)),
                    static_cast<float *>(buffer) +
                        (kBlock32 - kIdx) * bufferOffset + kBlock32RoundedUp * (mBlock - mIdx), offset,
                    bufferOffset * sizeof(float), 1.0f);
            }
            if (kResid) {
                for (IntType mSubIdx = mIdx; mSubIdx > 0; --mSubIdx) {
                    svint8_t srcReg = svld1(pgResid, static_cast<const std::int8_t *>(src) +
                        (mBlock - mSubIdx) * offset + kBlock32 * sizeof(float));
                    svst1(pg4, static_cast<std::int8_t *>(buffer) +
                        kBlock32 * bufferOffset * sizeof(float) +
                        (kBlock32RoundedUp * (mBlock - mIdx)+ (mIdx - mSubIdx)) * NUM_4, srcReg);
                }
            }
        } else {
            IntType mIdxReorder1 = vlw - 1;
            IntType mIdxReorder2 = std::min(vlw, mIdx - vlw) - 1;
            for (IntType kIdx = kBlock32; kIdx > 0; kIdx -= vlw) {
                IntType kIdxReorder = std::min(vlw, kIdx) - 1;
                Reorder::TRANSPOSE_REORDER_FUNC_ARR[mIdxReorder1 * vlw + kIdxReorder](
                    Service::AddOffset<const float *>(src,
                        (mBlock - mIdx) * offset + (kBlock32 - kIdx) * sizeof(float)),
                    static_cast<float *>(buffer) +
                        (kBlock32 - kIdx) * bufferOffset + kBlock32RoundedUp * (mBlock - mIdx), offset,
                    bufferOffset * sizeof(float), 1.0f);
                Reorder::TRANSPOSE_REORDER_FUNC_ARR[mIdxReorder2 * vlw + kIdxReorder](
                    reinterpret_cast<const float *>(static_cast<const char *>(src) +
                        (mBlock - mIdx + vlw) * offset + (kBlock32 - kIdx) * sizeof(float)),
                    static_cast<float *>(buffer) +
                        (kBlock32 - kIdx) * bufferOffset + kBlock32RoundedUp * (mBlock - mIdx) + vlw, offset,
                    bufferOffset * sizeof(float), 1.0f);
            }
            if (kResid) {
                for (IntType mSubIdx = mIdx; mSubIdx > std::max(IntType{0}, mIdx - NUM_2 * vlw); --mSubIdx) {
                    svint8_t srcReg = svld1(pgResid, Service::AddOffset<const std::int8_t *>(src,
                        (mBlock - mSubIdx) * offset + kBlock32 * sizeof(float)));
                    svst1(pg4, static_cast<std::int8_t *>(buffer) +
                        kBlock32 * bufferOffset * sizeof(float) +
                        (kBlock32RoundedUp * (mBlock - mIdx) + (mIdx - mSubIdx)) * NUM_4, srcReg);
                }
            }
        }
    }
}

// works with dst and bias matrices in row-major format
template <bool NonZeroBeta, DstLoadFlags DstFlags>
void UpdateDstFromAccum(IntType m, IntType n, IntType accumOffset, IntType dstOffset, IntType biaOffset,
    const std::int32_t *accum, std::int32_t *dst, const std::int32_t *bia,
    float alpha, float beta = 0.0) noexcept
{
    IntType vlw = svcntw();
    svfloat32_t alphaReg = svdup_n_f32(alpha);
    svfloat32_t betaReg = svdup_n_f32(beta);
    svfloat32_t biaRegF32 = svdup_f32(0.0f);
    if constexpr (DstFlags == DstLoadFlags::LOAD_BIAS_1) {
        biaRegF32 = svdup_f32(bia[0]);
    }
    for (IntType mIdx = 0; mIdx < m; ++mIdx) {
        if constexpr (DstFlags == DstLoadFlags::LOAD_BIAS_M) {
            biaRegF32 = svdup_f32(bia[mIdx * biaOffset]);
        }
        for (IntType nIdx = 0; nIdx < n; nIdx += vlw) {
            svbool_t pg = svwhilelt_b32(nIdx, n);

            const std::int32_t *curAccumPtr = accum + nIdx + mIdx * accumOffset;
            svint32_t accumReg = svld1(pg, curAccumPtr);
            svfloat32_t accumRegF32 = svcvt_f32_s32_x(pg, accumReg);

            std::int32_t *curDstPtr = dst + nIdx + mIdx * dstOffset;
            svint32_t dstReg;
            svfloat32_t dstRegF32;
            if constexpr (NonZeroBeta) {
                dstReg = svld1(pg, curDstPtr);
                dstRegF32 = svcvt_f32_s32_x(pg, dstReg);
            }

            if ((DstFlags == DstLoadFlags::LOAD_BIAS_N) || (DstFlags == DstLoadFlags::LOAD_BIAS_MN)) {
                const std::int32_t *curBiaPtr = bia + nIdx + mIdx * biaOffset;
                svint32_t biaReg = svld1(pg, curBiaPtr);
                biaRegF32 = svcvt_f32_s32_x(pg, biaReg);
            }

            if constexpr (NonZeroBeta) {
                accumRegF32 = svmla_x(pg, biaRegF32, accumRegF32, alphaReg);
                dstRegF32 = svmla_x(pg, accumRegF32, dstRegF32, betaReg);
            } else {
                dstRegF32 = svmla_x(pg, biaRegF32, accumRegF32, alphaReg);
            }

            dstRegF32 = svrinti_f32_x(pg, dstRegF32);
            dstReg = svcvt_s32_f32_x(pg, dstRegF32);

            svst1(pg, curDstPtr, dstReg);
        }
    }
}

using UpdateDstFunc = void(*) (IntType, IntType, IntType, IntType, IntType,
    const std::int32_t *, std::int32_t *, const std::int32_t *, float, float);

static const UpdateDstFunc UPDATE_DST_FUNC_ARR[10] = {
    UpdateDstFromAccum<false, DstLoadFlags::FILL_C_WITH_ZEROZ>,
    UpdateDstFromAccum<false, DstLoadFlags::LOAD_BIAS_1>,
    UpdateDstFromAccum<false, DstLoadFlags::LOAD_BIAS_M>,
    UpdateDstFromAccum<false, DstLoadFlags::LOAD_BIAS_N>,
    UpdateDstFromAccum<false, DstLoadFlags::LOAD_BIAS_MN>,
    UpdateDstFromAccum<true, DstLoadFlags::FILL_C_WITH_ZEROZ>,
    UpdateDstFromAccum<true, DstLoadFlags::LOAD_BIAS_1>,
    UpdateDstFromAccum<true, DstLoadFlags::LOAD_BIAS_M>,
    UpdateDstFromAccum<true, DstLoadFlags::LOAD_BIAS_N>,
    UpdateDstFromAccum<true, DstLoadFlags::LOAD_BIAS_MN>
};

template <typename SrcType, typename WeiType, typename DstType, typename BiaType>
void MatMulImplGenerator<MatMulKernelDefaultCodeGen<ARCH::MatrixComputation,
    SrcType, WeiType, DstType, BiaType>>::DefaultImpl::RunSequential2D(
    IntType threadId, const void *src, const void *wei, void *dst,
    const void *bia, void *buffer, float alpha, float beta) const noexcept(false)
{
    VERBOSE_PRINT;
    Service::Unused(alpha);
    IntType m = perThreadParams[threadId].m;
    IntType n = perThreadParams[threadId].n;
    IntType k = perThreadParams[threadId].k;
    IntType vlw = svcntw();
    IntType vlb = svcntb();
    IntType mRoundedUp = Math::CeilDiv(m, CodeGen::kernelMSize * vlb) * CodeGen::kernelMSize * vlb;
    // NUM_4 is due to peculiarities of MOPA instruction for INT8 data type
    IntType kRoundedUp = Math::CeilDiv(k, CodeGen::kLoopUnroll * NUM_4) * CodeGen::kLoopUnroll * NUM_4;
    SrcType* srcBuffer = static_cast<SrcType *>(buffer);
    WeiType* weiBuffer = static_cast<WeiType *>(static_cast<void *>(static_cast<SrcType *>(buffer) +
        std::min(CodeGen::GetKBlockSize(m, n, k), kRoundedUp) *
        std::min(CodeGen::GetMBlockSize(m, n, k), mRoundedUp)));
    bool useDstAsAccumulator = (alpha == 1.0f) && ((beta == 0.0f) || (beta == 1.0f));
    IntType accumulatorBufferSize = n * std::min(CodeGen::GetMBlockSize(m, n, k), m) * sizeof(DstType);
    DstType *accumulator;
    IntType accumOffset;
    if (useDstAsAccumulator) {
        accumulator = static_cast<DstType *>(dst);
        accumOffset = perThreadParams[threadId].dstOffset2D;
    } else {
        accumulator = static_cast<DstType *>(Service::AlignedAlloc(accumulatorBufferSize));
        accumOffset = n;
    }
    for (IntType mBlock = 0; mBlock < m; mBlock += CodeGen::GetMBlockSize(m, n, k)) {
        auto mBlockExec = std::min(CodeGen::GetMBlockSize(m, n, k), m - mBlock);
        for (IntType kBlock = 0; kBlock < k; kBlock += CodeGen::GetKBlockSize(m, n, k)) {
            auto kBlockExec = std::min(CodeGen::GetKBlockSize(m, n, k), k - kBlock);
            // NUM_4 is due to peculiarities of MOPA instruction for INT8 data type
            auto kBlockExecRoundedUp = ((kBlockExec + NUM_3) / NUM_4) * NUM_4;
            int kIdx = (kBlockExec == CodeGen::GetKBlockSize(m, n, k)) ? 0 : 1;
            DstLoadFlags biaFlagsExec;
            if (kBlock) {
                biaFlagsExec = DstLoadFlags::LOAD_C;
            } else {
                biaFlagsExec = ((alpha != 1.0f) ?
                    (DstLoadFlags::FILL_C_WITH_ZEROZ) : ((beta == 1.0f) ? (biaFlags << NUM_4) : biaFlags));
            }
            if (perThreadParams[threadId].srcLayout2D == Layout::COL_MAJOR) {
                const SrcType *srcExec = static_cast<const SrcType *>(src) + mBlock +
                    kBlock * perThreadParams[threadId].srcOffset2D;
                    PackSimilarLt(mBlockExec, kBlockExec, srcExec,
                        perThreadParams[threadId].srcOffset2D, srcBuffer,
                        CodeGen::kernelMSize * vlw);
            } else {
                const SrcType *srcExec = static_cast<const SrcType *>(src) +
                    mBlock * perThreadParams[threadId].srcOffset2D + kBlock;
                    PackDiffLt(mBlockExec, kBlockExec, srcExec,
                        perThreadParams[threadId].srcOffset2D, srcBuffer,
                        CodeGen::kernelMSize * vlw);
            }
            for (IntType nBlock = 0; nBlock < n; nBlock += CodeGen::GetNBlockSize(m, n, k)) {
                auto nBlockExec = std::min(CodeGen::GetNBlockSize(m, n, k), n - nBlock);
                if (perThreadParams[threadId].weiLayout2D == Layout::ROW_MAJOR) {
                    const WeiType *weiExec = static_cast<const WeiType *>(wei) + nBlock +
                        kBlock * perThreadParams[threadId].weiOffset2D;
                    PackSimilarLt(nBlockExec, kBlockExec, weiExec,
                        perThreadParams[threadId].weiOffset2D, weiBuffer,
                        CodeGen::kernelNSize * vlw);
                } else {
                    const WeiType *weiExec = static_cast<const WeiType *>(wei) +
                        nBlock * perThreadParams[threadId].weiOffset2D + kBlock;
                    PackDiffLt(nBlockExec, kBlockExec, weiExec,
                        perThreadParams[threadId].weiOffset2D, weiBuffer,
                        CodeGen::kernelNSize * vlw);
                }
                for (IntType mSubBlock = 0; mSubBlock < mBlockExec; mSubBlock += CodeGen::kernelMSize * vlw) {
                    IntType mSubBlockExec = std::min(CodeGen::kernelMSize * vlw, mBlockExec - mSubBlock);
                    int mIdx = (mSubBlockExec == (CodeGen::kernelMSize * vlw)) ? 0 : 1;
                    for (IntType nSubBlock = 0; nSubBlock < nBlockExec; nSubBlock += CodeGen::kernelNSize * vlw) {
                        IntType nSubBlockExec = std::min(CodeGen::kernelNSize * vlw, nBlockExec - nSubBlock);
                        int nIdx = (nSubBlockExec == (CodeGen::kernelNSize * vlw)) ? 0 : 1;
                        DstType *dstExec = accumulator + nBlock + nSubBlock;
                        if (useDstAsAccumulator) {
                            dstExec += (mBlock + mSubBlock) * accumOffset;
                        } else {
                            dstExec += mSubBlock * accumOffset;
                        }
                        const BiaType *biaExec = static_cast<const BiaType *>(bia) + (nBlock + nSubBlock) * biaNOffset +
                        (mBlock + mSubBlock) * perThreadParams[threadId].biaOffset2D;
                        auto kernel =
                            perThreadImpls[threadId][mIdx * NUM_4 + nIdx * NUM_2 + kIdx].template GetCode<void (*)(
                            const SrcType *, const WeiType *, DstType *, const BiaType *, DstLoadFlags, IntType)>();
                        kernel(srcBuffer + mSubBlock * kBlockExecRoundedUp, weiBuffer + nSubBlock * kBlockExecRoundedUp,
                            dstExec, biaExec, biaFlagsExec, accumOffset * sizeof(DstType));
                    }
                }
            }
        }
        if (!useDstAsAccumulator) {
            IntType funcIdx = (beta != 0.0f) * NUM_5 +
                std::max(Math::Log2(static_cast<IntType>(biaFlags)) - NUM_2, IntType{0});
            UPDATE_DST_FUNC_ARR[funcIdx](
                mBlockExec, n, accumOffset,
                perThreadParams[threadId].dstOffset2D, perThreadParams[threadId].biaOffset2D,
                accumulator, static_cast<DstType *>(dst) + mBlock * perThreadParams[threadId].dstOffset2D,
                static_cast<const BiaType *>(bia) + mBlock * perThreadParams[threadId].biaOffset2D, alpha, beta);
        }
    }
    if (!useDstAsAccumulator) {
        Service::Deallocate(accumulator);
    }
}

template <typename SrcType, typename WeiType, typename DstType, typename BiaType>
void MatMulImplGenerator<MatMulKernelDefaultCodeGen<ARCH::MatrixComputation,
    SrcType, WeiType, DstType, BiaType>>::DefaultImpl::UpdateExecParams() noexcept
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
    InitializeTaskExec(desiredSrcLayout, desiredWeiLayout, desiredDstLayout, desiredBiaLayout,
        CodeGen::GetSrcType(), CodeGen::GetWeiType(), CodeGen::GetDstType(),
        (implTask.biaInfo.GetTotalTensorSize() != 0) ? CodeGen::GetBiaType() :
            static_cast<Element::TypeT>(implTask.biaInfo.GetType()));
}


template struct MatMulImplGenerator<MatMulKernelDefaultCodeGen<ARCH::MatrixComputation,
    std::int8_t, std::int8_t, std::int32_t, std::int32_t>>;

template struct MatMulImplGenerator<MatMulKernelDefaultCodeGen<ARCH::MatrixComputation,
    std::int8_t, std::uint8_t, std::int32_t, std::int32_t>>;

template struct MatMulImplGenerator<MatMulKernelDefaultCodeGen<ARCH::MatrixComputation,
    std::uint8_t, std::int8_t, std::int32_t, std::int32_t>>;

template struct MatMulImplGenerator<MatMulKernelDefaultCodeGen<ARCH::MatrixComputation,
    std::uint8_t, std::uint8_t, std::int32_t, std::int32_t>>;

} // Detail
} // KuDNN
