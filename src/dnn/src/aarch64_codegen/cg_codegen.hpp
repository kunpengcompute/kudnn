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

#ifndef CG_CODEGEN_HPP
#define CG_CODEGEN_HPP

#include <cmath>
#include <cstring>
#include <iostream>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#ifdef __GNUC__
#ifndef CG_USE_MMAP_ALLOCATOR
#define CG_USE_MMAP_ALLOCATOR
#endif
#endif

#include "src/cg_adr_sve.hpp"
#include "src/cg_label.hpp"

namespace ARMCG {

enum class BarOpt {
    SY = 0xf,
    ST = 0xe,
    LD = 0xd,
    ISH = 0xb,
    ISHST = 0xa,
    ISHLD = 0x9,
    NSH = 0x7,
    NSHST = 0x6,
    NSHLD = 0x5,
    OSH = 0x3,
    OSHST = 0x2,
    OSHLD = 0x1,
};

enum class PStateField {
    SPSEL,
    DAIFSET,
    DAIFCLR,
    UAO,
    PAN,
    DIT
};

// the enum value corresponds to instruction mask and should not be changed
enum class StreamMode {
    NONE = 0,
    SM = 1,
    ZA = 2,
    SMZA = 3
};

enum class Cond {
    EQ = 0x0,
    NE = 0x1,
    CS = 0x2,
    HS = 0x2,
    CC = 0x3,
    LO = 0x3,
    MI = 0x4,
    PL = 0x5,
    VS = 0x6,
    VC = 0x7,
    HI = 0x8,
    LS = 0x9,
    GE = 0xa,
    LT = 0xb,
    GT = 0xc,
    LE = 0xd,
    AL = 0xe,
    NV = 0xf
};

enum class Prfop {
    PLDL1KEEP = (0x0 << 3) + (0x0 << 1) + 0x0,
    PLDL1STRM = (0x0 << 3) + (0x0 << 1) + 0x1,
    PLDL2KEEP = (0x0 << 3) + (0x1 << 1) + 0x0,
    PLDL2STRM = (0x0 << 3) + (0x1 << 1) + 0x1,
    PLDL3KEEP = (0x0 << 3) + (0x2 << 1) + 0x0,
    PLDL3STRM = (0x0 << 3) + (0x2 << 1) + 0x1,
    PLIL1KEEP = (0x1 << 3) + (0x0 << 1) + 0x0,
    PLIL1STRM = (0x1 << 3) + (0x0 << 1) + 0x1,
    PLIL2KEEP = (0x1 << 3) + (0x1 << 1) + 0x0,
    PLIL2STRM = (0x1 << 3) + (0x1 << 1) + 0x1,
    PLIL3KEEP = (0x1 << 3) + (0x2 << 1) + 0x0,
    PLIL3STRM = (0x1 << 3) + (0x2 << 1) + 0x1,
    PSTL1KEEP = (0x2 << 3) + (0x0 << 1) + 0x0,
    PSTL1STRM = (0x2 << 3) + (0x0 << 1) + 0x1,
    PSTL2KEEP = (0x2 << 3) + (0x1 << 1) + 0x0,
    PSTL2STRM = (0x2 << 3) + (0x1 << 1) + 0x1,
    PSTL3KEEP = (0x2 << 3) + (0x2 << 1) + 0x0,
    PSTL3STRM = (0x2 << 3) + (0x2 << 1) + 0x1
};

enum class PrfopSve {
    PLDL1KEEP_VectorComputation = 0x0,
    PLDL1STRM_VectorComputation = 0x1,
    PLDL2KEEP_VectorComputation = 0x2,
    PLDL2STRM_VectorComputation = 0x3,
    PLDL3KEEP_VectorComputation = 0x4,
    PLDL3STRM_VectorComputation = 0x5,
    PSTL1KEEP_VectorComputation = 0x8,
    PSTL1STRM_VectorComputation = 0x9,
    PSTL2KEEP_VectorComputation = 0xa,
    PSTL2STRM_VectorComputation = 0xb,
    PSTL3KEEP_VectorComputation = 0xc,
    PSTL3STRM_VectorComputation = 0xd
};

enum class Pattern {
    POW2 = 0x0,
    VL1 = 0x1,
    VL2 = 0x2,
    VL3 = 0x3,
    VL4 = 0x4,
    VL5 = 0x5,
    VL6 = 0x6,
    VL7 = 0x7,
    VL8 = 0x8,
    VL16 = 0x9,
    VL32 = 0xa,
    VL64 = 0xb,
    VL128 = 0xc,
    VL256 = 0xd,
    MUL4 = 0x1d,
    MUL3 = 0x1e,
    ALL = 0x1f
};

enum class IcOp {
    ALLUIS = Inner::GenSysInstOp(0, 7, 1, 0), // op1=0, CRn=7, CRm=1, op2=0
    ALLU = Inner::GenSysInstOp(0, 7, 5, 0),   // op1=0, CRn=7, CRm=5, op2=0
    VAU = Inner::GenSysInstOp(3, 7, 5, 0)     // op1=3, CRn=7, CRm=5, op2=1
};

enum class DcOp {
    IVAC = Inner::GenSysInstOp(0, 7, 0x6, 1),  // op1=0, CRn=7, CRm=0x6, op2=1
    ISW = Inner::GenSysInstOp(0, 7, 0x6, 2),   // op1=0, CRn=7, CRm=0x6, op2=2
    CSW = Inner::GenSysInstOp(0, 7, 0xA, 2),   // op1=0, CRn=7, CRm=0xA, op2=2
    CISW = Inner::GenSysInstOp(0, 7, 0xE, 2),  // op1=0, CRn=7, CRm=0xE, op2=2
    ZVA = Inner::GenSysInstOp(3, 7, 0x4, 1),   // op1=3, CRn=7, CRm=0x4, op2=1
    CVAC = Inner::GenSysInstOp(3, 7, 0xA, 1),  // op1=3, CRn=7, CRm=0xA, op2=1
    CVAU = Inner::GenSysInstOp(3, 7, 0xB, 1),  // op1=3, CRn=7, CRm=0xB, op2=1
    CIVAC = Inner::GenSysInstOp(3, 7, 0xE, 1), // op1=3, CRn=7, CRm=0xE, op2=1
    CVAP = Inner::GenSysInstOp(3, 7, 0xC, 1)   // op1=3, CRn=7, CRm=0xC, op2=1
};

enum class AtOp {
    S1E1R = Inner::GenSysInstOp(0, 7, 0x8, 0),  // op1=0, CRn=7, CRm=0x8, op2=0
    S1E1W = Inner::GenSysInstOp(0, 7, 0x8, 1),  // op1=0, CRn=7, CRm=0x8, op2=1
    S1E0R = Inner::GenSysInstOp(0, 7, 0x8, 2),  // op1=0, CRn=7, CRm=0x8, op2=2
    S1E0W = Inner::GenSysInstOp(0, 7, 0x8, 3),  // op1=0, CRn=7, CRm=0x8, op2=3
    S1E2R = Inner::GenSysInstOp(4, 7, 0x8, 0),  // op1=4, CRn=7, CRm=0x8, op2=0
    S1E2W = Inner::GenSysInstOp(4, 7, 0x8, 1),  // op1=4, CRn=7, CRm=0x8, op2=1
    S12E1R = Inner::GenSysInstOp(4, 7, 0x8, 4), // op1=4, CRn=7, CRm=0x8, op2=4
    S12E1W = Inner::GenSysInstOp(4, 7, 0x8, 5), // op1=4, CRn=7, CRm=0x8, op2=5
    S12E0R = Inner::GenSysInstOp(4, 7, 0x8, 6), // op1=4, CRn=7, CRm=0x8, op2=6
    S12E0W = Inner::GenSysInstOp(4, 7, 0x8, 7), // op1=4, CRn=7, CRm=0x8, op2=7
    S1E3R = Inner::GenSysInstOp(6, 7, 0x8, 0),  // op1=6, CRn=7, CRm=0x8, op2=0
    S1E3W = Inner::GenSysInstOp(6, 7, 0x8, 1),  // op1=6, CRn=7, CRm=0x8, op2=1
    S1E1RP = Inner::GenSysInstOp(0, 7, 0x9, 0), // op1=0, CRn=7, CRm=0x9, op2=0
    S1E1WP = Inner::GenSysInstOp(0, 7, 0x9, 1), // op1=0, CRn=7, CRm=0x9, op2=1
};

enum class TlbiOp {
    VMALLE1IS = Inner::GenSysInstOp(0, 7, 3, 0),    // op1=0, CRn=7, CRm=0x3, op2=0
    VAE1IS = Inner::GenSysInstOp(0, 7, 3, 1),       // op1=0, CRn=7, CRm=0x3, op2=1
    ASIDE1IS = Inner::GenSysInstOp(0, 7, 3, 2),     // op1=0, CRn=7, CRm=0x3, op2=2
    VAAE1IS = Inner::GenSysInstOp(0, 7, 3, 3),      // op1=0, CRn=7, CRm=0x3, op2=3
    VALE1IS = Inner::GenSysInstOp(0, 7, 3, 5),      // op1=0, CRn=7, CRm=0x3, op2=5
    VAALE1IS = Inner::GenSysInstOp(0, 7, 3, 7),     // op1=0, CRn=7, CRm=0x3, op2=7
    VMALLE1 = Inner::GenSysInstOp(0, 7, 7, 0),      // op1=0, CRn=7, CRm=0x7, op2=0
    VAE1 = Inner::GenSysInstOp(0, 7, 7, 1),         // op1=0, CRn=7, CRm=0x7, op2=1
    ASIDE1 = Inner::GenSysInstOp(0, 7, 7, 2),       // op1=0, CRn=7, CRm=0x7, op2=2
    VAAE1 = Inner::GenSysInstOp(0, 7, 7, 3),        // op1=0, CRn=7, CRm=0x7, op2=3
    VALE1 = Inner::GenSysInstOp(0, 7, 7, 5),        // op1=0, CRn=7, CRm=0x7, op2=5
    VAALE1 = Inner::GenSysInstOp(0, 7, 7, 7),       // op1=0, CRn=7, CRm=0x7, op2=7
    IPAS2E1IS = Inner::GenSysInstOp(4, 7, 0, 1),    // op1=4, CRn=7, CRm=0x0, op2=1
    IPAS2LE1IS = Inner::GenSysInstOp(4, 7, 0, 5),   // op1=4, CRn=7, CRm=0x0, op2=5
    ALLE2IS = Inner::GenSysInstOp(4, 7, 3, 0),      // op1=4, CRn=7, CRm=0x3, op2=0
    VAE2IS = Inner::GenSysInstOp(4, 7, 3, 1),       // op1=4, CRn=7, CRm=0x3, op2=1
    ALLE1IS = Inner::GenSysInstOp(4, 7, 3, 4),      // op1=4, CRn=7, CRm=0x3, op2=4
    VALE2IS = Inner::GenSysInstOp(4, 7, 3, 5),      // op1=4, CRn=7, CRm=0x3, op2=5
    VMALLS12E1IS = Inner::GenSysInstOp(4, 7, 3, 6), // op1=4, CRn=7, CRm=0x3, op2=6
    IPAS2E1 = Inner::GenSysInstOp(4, 7, 4, 1),      // op1=4, CRn=7, CRm=0x4, op2=1
    IPAS2LE1 = Inner::GenSysInstOp(4, 7, 4, 5),     // op1=4, CRn=7, CRm=0x4, op2=5
    ALLE2 = Inner::GenSysInstOp(4, 7, 7, 0),        // op1=4, CRn=7, CRm=0x7, op2=0
    VAE2 = Inner::GenSysInstOp(4, 7, 7, 1),         // op1=4, CRn=7, CRm=0x7, op2=1
    ALLE1 = Inner::GenSysInstOp(4, 7, 7, 4),        // op1=4, CRn=7, CRm=0x7, op2=4
    VALE2 = Inner::GenSysInstOp(4, 7, 7, 5),        // op1=4, CRn=7, CRm=0x7, op2=5
    VMALLS12E1 = Inner::GenSysInstOp(4, 7, 7, 6),   // op1=4, CRn=7, CRm=0x7, op2=6
    ALLE3IS = Inner::GenSysInstOp(6, 7, 3, 0),      // op1=6, CRn=7, CRm=0x3, op2=0
    VAE3IS = Inner::GenSysInstOp(6, 7, 3, 1),       // op1=6, CRn=7, CRm=0x3, op2=1
    VALE3IS = Inner::GenSysInstOp(6, 7, 3, 5),      // op1=6, CRn=7, CRm=0x3, op2=5
    ALLE3 = Inner::GenSysInstOp(6, 7, 7, 0),        // op1=6, CRn=7, CRm=0x7, op2=0
    VAE3 = Inner::GenSysInstOp(6, 7, 7, 1),         // op1=6, CRn=7, CRm=0x7, op2=1
    VALE3 = Inner::GenSysInstOp(6, 7, 7, 5),        // op1=6, CRn=7, CRm=0x7, op2=5
    VMALLE1OS = Inner::GenSysInstOp(0, 7, 1, 0),    // op1=0, CRn=7, CRm=0x1, op2=0
    VAE1OS = Inner::GenSysInstOp(0, 7, 1, 1),       // op1=0, CRn=7, CRm=0x1, op2=1
    ASIDE1OS = Inner::GenSysInstOp(0, 7, 1, 2),     // op1=0, CRn=7, CRm=0x1, op2=2
    VAAE1OS = Inner::GenSysInstOp(0, 7, 1, 3),      // op1=0, CRn=7, CRm=0x1, op2=3
    VALE1OS = Inner::GenSysInstOp(0, 7, 1, 5),      // op1=0, CRn=7, CRm=0x1, op2=5
    VAALE1OS = Inner::GenSysInstOp(0, 7, 1, 7),     // op1=0, CRn=7, CRm=0x1, op2=7
    RVAE1IS = Inner::GenSysInstOp(0, 7, 2, 1),      // op1=0, CRn=7, CRm=0x2, op2=1
    RVAAE1IS = Inner::GenSysInstOp(0, 7, 2, 3),     // op1=0, CRn=7, CRm=0x2, op2=3
    RVALE1IS = Inner::GenSysInstOp(0, 7, 2, 5),     // op1=0, CRn=7, CRm=0x2, op2=5
    RVAALE1IS = Inner::GenSysInstOp(0, 7, 2, 7),    // op1=0, CRn=7, CRm=0x2, op2=7
    RVAE1OS = Inner::GenSysInstOp(0, 7, 5, 1),      // op1=0, CRn=7, CRm=0x5, op2=1
    RVAAE1OS = Inner::GenSysInstOp(0, 7, 5, 3),     // op1=0, CRn=7, CRm=0x5, op2=3
    RVALE1OS = Inner::GenSysInstOp(0, 7, 5, 5),     // op1=0, CRn=7, CRm=0x5, op2=5
    RVAALE1OS = Inner::GenSysInstOp(0, 7, 5, 7),    // op1=0, CRn=7, CRm=0x5, op2=7
    RVAE1 = Inner::GenSysInstOp(0, 7, 6, 1),        // op1=0, CRn=7, CRm=0x6, op2=1
    RVAAE1 = Inner::GenSysInstOp(0, 7, 6, 3),       // op1=0, CRn=7, CRm=0x6, op2=3
    RVALE1 = Inner::GenSysInstOp(0, 7, 6, 5),       // op1=0, CRn=7, CRm=0x6, op2=5
    RVAALE1 = Inner::GenSysInstOp(0, 7, 6, 7),      // op1=0, CRn=7, CRm=0x6, op2=7
    RIPAS2E1IS = Inner::GenSysInstOp(4, 7, 0, 2),   // op1=4, CRn=7, CRm=0x0, op2=2
    RIPAS2LE1IS = Inner::GenSysInstOp(4, 7, 0, 6),  // op1=4, CRn=7, CRm=0x0, op2=6
    ALLE2OS = Inner::GenSysInstOp(4, 7, 1, 0),      // op1=4, CRn=7, CRm=0x1, op2=0
    VAE2OS = Inner::GenSysInstOp(4, 7, 1, 1),       // op1=4, CRn=7, CRm=0x1, op2=1
    ALLE1OS = Inner::GenSysInstOp(4, 7, 1, 4),      // op1=4, CRn=7, CRm=0x1, op2=4
    VALE2OS = Inner::GenSysInstOp(4, 7, 1, 5),      // op1=4, CRn=7, CRm=0x1, op2=5
    VMALLS12E1OS = Inner::GenSysInstOp(4, 7, 1, 6), // op1=4, CRn=7, CRm=0x1, op2=6
    RVAE2IS = Inner::GenSysInstOp(4, 7, 2, 1),      // op1=4, CRn=7, CRm=0x2, op2=1
    RVALE2IS = Inner::GenSysInstOp(4, 7, 2, 5),     // op1=4, CRn=7, CRm=0x2, op2=5
    IPAS2E1OS = Inner::GenSysInstOp(4, 7, 4, 0),    // op1=4, CRn=7, CRm=0x4, op2=0
    RIPAS2E1 = Inner::GenSysInstOp(4, 7, 4, 2),     // op1=4, CRn=7, CRm=0x4, op2=2
    RIPAS2E1OS = Inner::GenSysInstOp(4, 7, 4, 3),   // op1=4, CRn=7, CRm=0x4, op2=3
    IPAS2LE1OS = Inner::GenSysInstOp(4, 7, 4, 4),   // op1=4, CRn=7, CRm=0x4, op2=4
    RIPAS2LE1 = Inner::GenSysInstOp(4, 7, 4, 6),    // op1=4, CRn=7, CRm=0x4, op2=6
    RIPAS2LE1OS = Inner::GenSysInstOp(4, 7, 4, 7),  // op1=4, CRn=7, CRm=0x4, op2=7
    RVAE2OS = Inner::GenSysInstOp(4, 7, 5, 1),      // op1=4, CRn=7, CRm=0x5, op2=1
    RVALE2OS = Inner::GenSysInstOp(4, 7, 5, 5),     // op1=4, CRn=7, CRm=0x5, op2=5
    RVAE2 = Inner::GenSysInstOp(4, 7, 6, 1),        // op1=4, CRn=7, CRm=0x6, op2=1
    RVALE2 = Inner::GenSysInstOp(4, 7, 6, 5),       // op1=4, CRn=7, CRm=0x6, op2=5
    ALLE3OS = Inner::GenSysInstOp(6, 7, 1, 0),      // op1=6, CRn=7, CRm=0x1, op2=0
    VAE3OS = Inner::GenSysInstOp(6, 7, 1, 1),       // op1=6, CRn=7, CRm=0x1, op2=1
    VALE3OS = Inner::GenSysInstOp(6, 7, 1, 5),      // op1=6, CRn=7, CRm=0x1, op2=5
    RVAE3IS = Inner::GenSysInstOp(6, 7, 2, 1),      // op1=6, CRn=7, CRm=0x2, op2=1
    RVALE3IS = Inner::GenSysInstOp(6, 7, 2, 5),     // op1=6, CRn=7, CRm=0x1, op2=5
    RVAE3OS = Inner::GenSysInstOp(6, 7, 5, 1),      // op1=6, CRn=7, CRm=0x5, op2=1
    RVALE3OS = Inner::GenSysInstOp(6, 7, 5, 5),     // op1=6, CRn=7, CRm=0x5, op2=5
    RVAE3 = Inner::GenSysInstOp(6, 7, 6, 1),        // op1=6, CRn=7, CRm=0x6, op2=1
    RVALE3 = Inner::GenSysInstOp(6, 7, 6, 5)        // op1=6, CRn=7, CRm=0x6, op2=5
};

class CodeGenerator : public CodeArray {
public:
    explicit CodeGenerator(size_t maxSize = static_cast<size_t>(MaxCodeSize::DEFAULT),
                           void *userPtr = DONT_SET_PROTECT_RWE, Allocator *allocator = nullptr)
        : CodeArray(maxSize, userPtr, allocator),
          w0(RINDEX0),
          w1(RINDEX1),
          w2(RINDEX2),
          w3(RINDEX3),
          w4(RINDEX4),
          w5(RINDEX5),
          w6(RINDEX6),
          w7(RINDEX7),
          w8(RINDEX8),
          w9(RINDEX9),
          w10(RINDEX10),
          w11(RINDEX11),
          w12(RINDEX12),
          w13(RINDEX13),
          w14(RINDEX14),
          w15(RINDEX15),
          w16(RINDEX16),
          w17(RINDEX17),
          w18(RINDEX18),
          w19(RINDEX19),
          w20(RINDEX20),
          w21(RINDEX21),
          w22(RINDEX22),
          w23(RINDEX23),
          w24(RINDEX24),
          w25(RINDEX25),
          w26(RINDEX26),
          w27(RINDEX27),
          w28(RINDEX28),
          w29(RINDEX29),
          w30(RINDEX30),
          wzr(RINDEX31),
          wsp(RINDEX31),
          x0(RINDEX0),
          x1(RINDEX1),
          x2(RINDEX2),
          x3(RINDEX3),
          x4(RINDEX4),
          x5(RINDEX5),
          x6(RINDEX6),
          x7(RINDEX7),
          x8(RINDEX8),
          x9(RINDEX9),
          x10(RINDEX10),
          x11(RINDEX11),
          x12(RINDEX12),
          x13(RINDEX13),
          x14(RINDEX14),
          x15(RINDEX15),
          x16(RINDEX16),
          x17(RINDEX17),
          x18(RINDEX18),
          x19(RINDEX19),
          x20(RINDEX20),
          x21(RINDEX21),
          x22(RINDEX22),
          x23(RINDEX23),
          x24(RINDEX24),
          x25(RINDEX25),
          x26(RINDEX26),
          x27(RINDEX27),
          x28(RINDEX28),
          x29(RINDEX29),
          x30(RINDEX30),
          xzr(RINDEX31),
          sp(RINDEX31),
          b0(RINDEX0),
          b1(RINDEX1),
          b2(RINDEX2),
          b3(RINDEX3),
          b4(RINDEX4),
          b5(RINDEX5),
          b6(RINDEX6),
          b7(RINDEX7),
          b8(RINDEX8),
          b9(RINDEX9),
          b10(RINDEX10),
          b11(RINDEX11),
          b12(RINDEX12),
          b13(RINDEX13),
          b14(RINDEX14),
          b15(RINDEX15),
          b16(RINDEX16),
          b17(RINDEX17),
          b18(RINDEX18),
          b19(RINDEX19),
          b20(RINDEX20),
          b21(RINDEX21),
          b22(RINDEX22),
          b23(RINDEX23),
          b24(RINDEX24),
          b25(RINDEX25),
          b26(RINDEX26),
          b27(RINDEX27),
          b28(RINDEX28),
          b29(RINDEX29),
          b30(RINDEX30),
          b31(RINDEX31),
          h0(RINDEX0),
          h1(RINDEX1),
          h2(RINDEX2),
          h3(RINDEX3),
          h4(RINDEX4),
          h5(RINDEX5),
          h6(RINDEX6),
          h7(RINDEX7),
          h8(RINDEX8),
          h9(RINDEX9),
          h10(RINDEX10),
          h11(RINDEX11),
          h12(RINDEX12),
          h13(RINDEX13),
          h14(RINDEX14),
          h15(RINDEX15),
          h16(RINDEX16),
          h17(RINDEX17),
          h18(RINDEX18),
          h19(RINDEX19),
          h20(RINDEX20),
          h21(RINDEX21),
          h22(RINDEX22),
          h23(RINDEX23),
          h24(RINDEX24),
          h25(RINDEX25),
          h26(RINDEX26),
          h27(RINDEX27),
          h28(RINDEX28),
          h29(RINDEX29),
          h30(RINDEX30),
          h31(RINDEX31),
          s0(RINDEX0),
          s1(RINDEX1),
          s2(RINDEX2),
          s3(RINDEX3),
          s4(RINDEX4),
          s5(RINDEX5),
          s6(RINDEX6),
          s7(RINDEX7),
          s8(RINDEX8),
          s9(RINDEX9),
          s10(RINDEX10),
          s11(RINDEX11),
          s12(RINDEX12),
          s13(RINDEX13),
          s14(RINDEX14),
          s15(RINDEX15),
          s16(RINDEX16),
          s17(RINDEX17),
          s18(RINDEX18),
          s19(RINDEX19),
          s20(RINDEX20),
          s21(RINDEX21),
          s22(RINDEX22),
          s23(RINDEX23),
          s24(RINDEX24),
          s25(RINDEX25),
          s26(RINDEX26),
          s27(RINDEX27),
          s28(RINDEX28),
          s29(RINDEX29),
          s30(RINDEX30),
          s31(RINDEX31),
          d0(RINDEX0),
          d1(RINDEX1),
          d2(RINDEX2),
          d3(RINDEX3),
          d4(RINDEX4),
          d5(RINDEX5),
          d6(RINDEX6),
          d7(RINDEX7),
          d8(RINDEX8),
          d9(RINDEX9),
          d10(RINDEX10),
          d11(RINDEX11),
          d12(RINDEX12),
          d13(RINDEX13),
          d14(RINDEX14),
          d15(RINDEX15),
          d16(RINDEX16),
          d17(RINDEX17),
          d18(RINDEX18),
          d19(RINDEX19),
          d20(RINDEX20),
          d21(RINDEX21),
          d22(RINDEX22),
          d23(RINDEX23),
          d24(RINDEX24),
          d25(RINDEX25),
          d26(RINDEX26),
          d27(RINDEX27),
          d28(RINDEX28),
          d29(RINDEX29),
          d30(RINDEX30),
          d31(RINDEX31),
          q0(RINDEX0),
          q1(RINDEX1),
          q2(RINDEX2),
          q3(RINDEX3),
          q4(RINDEX4),
          q5(RINDEX5),
          q6(RINDEX6),
          q7(RINDEX7),
          q8(RINDEX8),
          q9(RINDEX9),
          q10(RINDEX10),
          q11(RINDEX11),
          q12(RINDEX12),
          q13(RINDEX13),
          q14(RINDEX14),
          q15(RINDEX15),
          q16(RINDEX16),
          q17(RINDEX17),
          q18(RINDEX18),
          q19(RINDEX19),
          q20(RINDEX20),
          q21(RINDEX21),
          q22(RINDEX22),
          q23(RINDEX23),
          q24(RINDEX24),
          q25(RINDEX25),
          q26(RINDEX26),
          q27(RINDEX27),
          q28(RINDEX28),
          q29(RINDEX29),
          q30(RINDEX30),
          q31(RINDEX31),
          v0(RINDEX0),
          v1(RINDEX1),
          v2(RINDEX2),
          v3(RINDEX3),
          v4(RINDEX4),
          v5(RINDEX5),
          v6(RINDEX6),
          v7(RINDEX7),
          v8(RINDEX8),
          v9(RINDEX9),
          v10(RINDEX10),
          v11(RINDEX11),
          v12(RINDEX12),
          v13(RINDEX13),
          v14(RINDEX14),
          v15(RINDEX15),
          v16(RINDEX16),
          v17(RINDEX17),
          v18(RINDEX18),
          v19(RINDEX19),
          v20(RINDEX20),
          v21(RINDEX21),
          v22(RINDEX22),
          v23(RINDEX23),
          v24(RINDEX24),
          v25(RINDEX25),
          v26(RINDEX26),
          v27(RINDEX27),
          v28(RINDEX28),
          v29(RINDEX29),
          v30(RINDEX30),
          v31(RINDEX31),
          z0(RINDEX0),
          z1(RINDEX1),
          z2(RINDEX2),
          z3(RINDEX3),
          z4(RINDEX4),
          z5(RINDEX5),
          z6(RINDEX6),
          z7(RINDEX7),
          z8(RINDEX8),
          z9(RINDEX9),
          z10(RINDEX10),
          z11(RINDEX11),
          z12(RINDEX12),
          z13(RINDEX13),
          z14(RINDEX14),
          z15(RINDEX15),
          z16(RINDEX16),
          z17(RINDEX17),
          z18(RINDEX18),
          z19(RINDEX19),
          z20(RINDEX20),
          z21(RINDEX21),
          z22(RINDEX22),
          z23(RINDEX23),
          z24(RINDEX24),
          z25(RINDEX25),
          z26(RINDEX26),
          z27(RINDEX27),
          z28(RINDEX28),
          z29(RINDEX29),
          z30(RINDEX30),
          z31(RINDEX31),
          p0(RINDEX0),
          p1(RINDEX1),
          p2(RINDEX2),
          p3(RINDEX3),
          p4(RINDEX4),
          p5(RINDEX5),
          p6(RINDEX6),
          p7(RINDEX7),
          p8(RINDEX8),
          p9(RINDEX9),
          p10(RINDEX10),
          p11(RINDEX11),
          p12(RINDEX12),
          p13(RINDEX13),
          p14(RINDEX14),
          p15(RINDEX15),
          za0(RINDEX0),
          za1(RINDEX1),
          za2(RINDEX2),
          za3(RINDEX3),
          za4(RINDEX4),
          za5(RINDEX5),
          za6(RINDEX6),
          za7(RINDEX7),
          za0h(RINDEX0),
          za1h(RINDEX1),
          za2h(RINDEX2),
          za3h(RINDEX3),
          za4h(RINDEX4),
          za5h(RINDEX5),
          za6h(RINDEX6),
          za7h(RINDEX7),
          za0v(RINDEX0),
          za1v(RINDEX1),
          za2v(RINDEX2),
          za3v(RINDEX3),
          za4v(RINDEX4),
          za5v(RINDEX5),
          za6v(RINDEX6),
          za7v(RINDEX7)
    {
        labelMgr_.Set(this);
    }

    const WReg w0, w1, w2, w3, w4, w5, w6, w7, w8, w9, w10, w11, w12;
    const WReg w13, w14, w15, w16, w17, w18, w19, w20, w21, w22, w23;
    const WReg w24, w25, w26, w27, w28, w29, w30, wzr, wsp;
    const XReg x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12;
    const XReg x13, x14, x15, x16, x17, x18, x19, x20, x21, x22, x23;
    const XReg x24, x25, x26, x27, x28, x29, x30, xzr, sp;
    const BReg b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12;
    const BReg b13, b14, b15, b16, b17, b18, b19, b20, b21, b22, b23;
    const BReg b24, b25, b26, b27, b28, b29, b30, b31;
    const HReg h0, h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12;
    const HReg h13, h14, h15, h16, h17, h18, h19, h20, h21, h22, h23;
    const HReg h24, h25, h26, h27, h28, h29, h30, h31;
    const SReg s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11, s12;
    const SReg s13, s14, s15, s16, s17, s18, s19, s20, s21, s22, s23;
    const SReg s24, s25, s26, s27, s28, s29, s30, s31;
    const DReg d0, d1, d2, d3, d4, d5, d6, d7, d8, d9, d10, d11, d12;
    const DReg d13, d14, d15, d16, d17, d18, d19, d20, d21, d22, d23;
    const DReg d24, d25, d26, d27, d28, d29, d30, d31;
    const QReg q0, q1, q2, q3, q4, q5, q6, q7, q8, q9, q10, q11, q12;
    const QReg q13, q14, q15, q16, q17, q18, q19, q20, q21, q22, q23;
    const QReg q24, q25, q26, q27, q28, q29, q30, q31;
    const VReg v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12;
    const VReg v13, v14, v15, v16, v17, v18, v19, v20, v21, v22, v23;
    const VReg v24, v25, v26, v27, v28, v29, v30, v31;
    const ZReg z0, z1, z2, z3, z4, z5, z6, z7, z8, z9, z10, z11, z12;
    const ZReg z13, z14, z15, z16, z17, z18, z19, z20, z21, z22, z23;
    const ZReg z24, z25, z26, z27, z28, z29, z30, z31;
    const PReg p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12;
    const PReg p13, p14, p15;
    const ZAReg za0, za1, za2, za3, za4, za5, za6, za7;
    const ZARegH za0h, za1h, za2h, za3h, za4h, za5h, za6h, za7h;
    const ZARegV za0v, za1v, za2v, za3v, za4v, za5v, za6v, za7v;

    // engine funtionality
    void L(Label &label)
    {
        labelMgr_.DefineClabel(label);
    }

    Label L()
    {
        Label label;
        L(label);
        return label;
    }
    // assign src to dst. Requirement: src used by L() and dst not used by L()
    void AssignL(Label &dst, const Label &src)
    {
        labelMgr_.Assign(dst, src);
    }
    // put address of label to buffer, where the put size is 4(32bit), 8(64bit)
    void PutL(const Label &label)
    {
        PutLInner(label);
    }

    void Reset()
    {
        ResetSize();
        labelMgr_.Reset();
        labelMgr_.Set(this);
    }

    bool HasUndefinedLabel() const
    {
        return labelMgr_.HasUndefClabel();
    }

    void ClearCache(void *begin, void *end);
    // MUST call ready() to complete generating code if using AutoGrow mode
    // BUT it is not necessary for the other mode if HasUndefinedLabel() is true
    void Ready(ProtectMode mode = ProtectMode::RE);
    // set read/exec
    void ReadyRE()
    {
        return Ready(ProtectMode::RE);
    }
    void Dump(bool doClear = true)
    {
        CodeArray::Dump();
        if (doClear) {
            size_ = 0;
        }
    }
    WReg GetTmpWReg()
    {
        return w29;
    }
    XReg GetTmpXReg()
    {
        return x29;
    }
    VReg GetTmpVReg()
    {
        return v31;
    }
    ZReg GetTmpZReg()
    {
        return z31;
    }
    PReg GetTmpPReg()
    {
        return p7;
    }

    void Align(size_t x)
    {
        // armv8 instructions are always 4 bytes
        if (x == INSTRUCTIONS_SIZE_4_BYTES) {
            return;
        }
        if (x < INSTRUCTIONS_SIZE_4_BYTES || (x % INSTRUCTIONS_SIZE_4_BYTES)) {
            throw Error(Err::BAD_ALIGN);
        }
        if (IsAutoGrow() && x > Inner::GetPageSize()) {
            fprintf(stderr, "waring:AutoGrow mode does not support %d align\n", static_cast<int>(x));
        }
        size_t remain = reinterpret_cast<size_t>(GetCurr());
        if (remain % INSTRUCTIONS_SIZE_4_BYTES) {
            throw Error(Err::BAD_ALIGN);
        }
        remain = x - (remain % x);
        while (remain) {
            Nop();
            remain -= INSTRUCTIONS_SIZE_4_BYTES;
        }
    }

// A64 instructions that clients can use
#include "src/cg_mnemonic.hpp"

protected:
    LabelManager labelMgr_;
// util and encoding functions
#include "src/cg_impl.hpp"
};

} // namespace ARMCG

#endif // CG_CODEGEN_HPP
