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

#include <vector>
#include <memory>
#include <limits>
#include <type_traits>
#include <chrono>
#include <mutex>
#include <array>

#include "kudnn.hpp"
#include "cg_codegen.hpp"
#include "internal_funcs.hpp"
#include "kudnn_jit.hpp"

namespace KuDNN {
namespace Detail {

KUDNN_JIT_Generator::KUDNN_JIT_Generator() : ARMCG::CodeGenerator(NUM_1024 * NUM_1024) // , ARMCG::AutoGrow
{
    VERBOSE_PRINT;
}

void KUDNN_JIT_Generator::Preamble()
{
    VERBOSE_PRINT;
    // store x29 and x30 to stack for ABI requirement and decrement stack pointer by 16 bytes (2x64-bit D registers)
    Stp(ARMCG::XReg(29), ARMCG::XReg(30), PrePtr(sp, -16));
    // store callee-saved X registers (x19-x28)
    for (int i = NUM_19; i < NUM_28; i += NUM_2) {
        Stp(ARMCG::XReg(i), ARMCG::XReg(i + 1),
            PrePtr(sp, -16)); // 16 is a size of 2x64-bit X registers
    }
    // store callee-saved vector registers (d8-d15)
    for (int i = NUM_8; i < NUM_15; i += NUM_2) {
        Stp(ARMCG::DReg(i), ARMCG::DReg(i + 1),
            PrePtr(sp, -16)); // 16 is a size of 2x64-bit D registers
    }
}

void KUDNN_JIT_Generator::Postamble()
{
    VERBOSE_PRINT;
    // load callee-saved vector registers (d8-d15)
    for (int i = NUM_15; i >= NUM_8; i -= NUM_2) {
        Ldp(ARMCG::DReg(i - 1), ARMCG::DReg(i),
            PostPtr(sp, 16)); // 16 is a size of 2x64-bit D registers
    }
    // load callee-saved X registers (x19-x28)
    for (int i = NUM_28; i >= NUM_19; i -= NUM_2) {
        Ldp(ARMCG::XReg(i - 1), ARMCG::XReg(i),
            PostPtr(sp, 16)); // 16 is a size of 2x64-bit X registers
    }
    // store x29 and x30 to stack for ABI requirement and decrement stack pointer by 16 bytes (2x64-bit D registers)
    Ldp(ARMCG::XReg(29), ARMCG::XReg(30), PostPtr(sp, 16));
}

} // Detail
} // KuDNN
