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

#ifndef CG_ARM_CONST_HPP
#define CG_ARM_CONST_HPP

namespace ARMCG {

constexpr int BITFIELD_0 = 0;
constexpr int BITFIELD_1 = 1;
constexpr int BITFIELD_2 = 2;
constexpr int BITFIELD_3 = 3;
constexpr int BITFIELD_4 = 4;
constexpr int BITFIELD_5 = 5;
constexpr int BITFIELD_6 = 6;
constexpr int BITFIELD_7 = 7;
constexpr int BITFIELD_8 = 8;
constexpr int BITFIELD_9 = 9;
constexpr int BITFIELD_10 = 10;
constexpr int BITFIELD_11 = 11;
constexpr int BITFIELD_12 = 12;
constexpr int BITFIELD_13 = 13;
constexpr int BITFIELD_14 = 14;
constexpr int BITFIELD_15 = 15;
constexpr int BITFIELD_16 = 16;
constexpr int BITFIELD_17 = 17;
constexpr int BITFIELD_18 = 18;
constexpr int BITFIELD_19 = 19;
constexpr int BITFIELD_20 = 20;
constexpr int BITFIELD_21 = 21;
constexpr int BITFIELD_22 = 22;
constexpr int BITFIELD_23 = 23;
constexpr int BITFIELD_24 = 24;
constexpr int BITFIELD_25 = 25;
constexpr int BITFIELD_26 = 26;
constexpr int BITFIELD_27 = 27;
constexpr int BITFIELD_28 = 28;
constexpr int BITFIELD_29 = 29;
constexpr int BITFIELD_30 = 30;
constexpr int BITFIELD_31 = 31;
constexpr int BITFIELD_32 = 32;
constexpr int BITFIELD_48 = 48;
constexpr int BITFIELD_49 = 49;
constexpr int BITFIELD_52 = 52;
constexpr int BITFIELD_57 = 57;
constexpr int BITFIELD_63 = 63;
constexpr int BITFIELD_64 = 64;
constexpr int ARM_BYTE = 8;
constexpr int ARM_HALF_WORD = 16;
constexpr int ARM_SINGLE_WORD = 32;
constexpr int ARM_DOUBLE_WORD = 64;
constexpr int ARM_QUAD_WORD = 128;

constexpr int OFFSET_ONE = 1;
constexpr int OFFSET_TWO = 2;
constexpr int OFFSET_THREE = 3;
constexpr int OFFSET_FOUR = 4;
constexpr int BLOCK_TEN = 10;
constexpr int MODE_ENCODE_ONE = 1;
constexpr int SHIFT_RANGE = 3;
constexpr int IMM2_BIT_LENGTH = 2;
constexpr int UIMM2_RANGE_HIGH = 4;
constexpr int IMM3_BIT_LENGTH = 3;
constexpr int IMM4_BIT_LENGTH = 4;
constexpr int UIMM4_RANGE_HIGH = 16;
constexpr int IMM5_BIT_LENGTH = 5;
constexpr int IMM5_RANGE_LOW = -16;
constexpr int IMM5_RANGE_HIGH = 15;
constexpr int UIMM5_RANGE_HIGH = 32;
constexpr int IMM6_BIT_LENGTH = 6;
constexpr int IMM6_RANGE_LOW = -32;
constexpr int IMM6_RANGE_HIGH = 31;
constexpr int UIMM6_RANGE_HIGH = 64;
constexpr int IMM7_BIT_LENGTH = 7;
constexpr int UIMM7_RANGE_HIGH = 127;
constexpr int IMM8_BIT_LENGTH = 8;
constexpr int IMM8_RANGE_LOW = -128;
constexpr int IMM8_RANGE_HIGH = 127;
constexpr int UIMM8_RANGE_HIGH = 255;
constexpr int IMM9_BIT_LENGTH = 9;
constexpr int IMM9_RANGE_LOW = -256;
constexpr int IMM9_RANGE_HIGH = 255;
constexpr int IMM12_BIT_LENGTH = 12;
constexpr int IMM13_RANGE_HIGH = 4095;
constexpr int SIMD_LDRSTR_IMM_RANGE_LOW = -256;
constexpr int SIMD_LDRSTR_IMM_RANGE_HIGH = 252;
constexpr int VC_HALF_PG_RANGE = 7;
constexpr int VC_COMPVEC_2 = 2;
constexpr int VC_ROTATION_0 = 0;
constexpr int VC_ROTATION_90 = 90;
constexpr int VC_ROTATION_180 = 180;
constexpr int VC_ROTATION_270 = 270;
constexpr int VC_ELT_BITLEN_B = 8;
constexpr int VC_ELT_BITLEN_B_RANGE = 1;
constexpr int VC_ELT_BITLEN_H = 16;
constexpr int VC_ELT_BITLEN_H_RANGE = 7;
constexpr int VC_ELT_BITLEN_S = 32;
constexpr int VC_ELT_BITLEN_S_MASK = 2;
constexpr int VC_ELT_BITLEN_S_SHIFT = 3;
constexpr int VC_ELT_BITLEN_S_RANGE = 3;
constexpr int VC_ELT_BITLEN_D = 64;
constexpr int VC_ELT_BITLEN_D_MASK = 1;
constexpr int VC_ELT_BITLEN_D_SHIFT = 4;
constexpr int VC_ELT_BITLEN_D_RANGE = 15;
constexpr int VC_ELT_BITLEN_Q = 128;
constexpr int VC_COMPZERO_FACTOR = 10;
constexpr int VC_ARITH_IMM_OPC_0 = 0;
constexpr int VC_ARITH_IMM_OPC_1 = 1;
constexpr int VC_ARITH_IMM_OPC_2 = 2;
constexpr int VC_ARITH_IMM_OPC_3 = 3;
constexpr int VC_ARITH_IMM_STEP_5 = 5;
constexpr int VC_ARITH_IMM_STEP_10 = 10;
constexpr int VC_ARITH_IMM_STEP_20 = 20;
constexpr int VC_COMPVEC_PG_RANGE = 7;
constexpr int VC_GATHERPRF_IMM_BLOCK = 31;
constexpr int VC_GATHERPRF_IMM_BIT_LENGTH = 5;
constexpr int VC_CONTIPRF_IMM_RANGE_LOW = -32;
constexpr int VC_CONTIPRF_IMM_RANGE_HIGH = 31;
constexpr int VC_CONTIPRF_IMM_BIT_LENGTH = 6;
constexpr int VC_PREFETCH_PG_RANGE = 7;
constexpr int VC_PREDLDR_IMM_RANGE_LOW = -256;
constexpr int VC_PREDLDR_IMM_RANGE_HIGH = 255;
constexpr int VC_CONTILDR_IMM_RANGE_LOW = -8;
constexpr int VC_CONTILDR_IMM_RANGE_HIGH = 7;
constexpr int VC_CONTILDR_IMM_BIT_LENGTH = 4;
constexpr int VC_BCLDR_IMM_RANGE_LOW = -128;
constexpr int VC_BCLDR_IMM_RANGE_HIGH = 127;
constexpr int VC_BCLDR_IMM_BIT_LENGTH = 4;
constexpr int VC_BCLDR_UIMM_BIT_LENGTH = 6;
constexpr int VC_BCLDR_IMM_MODULUS = 16;
constexpr int VC_BCLDR_IMM_BLOCK = 63;
constexpr int VC_BCLDR_DTYPE_SHIFT = 2;
constexpr int VC_MULTILDR_IMM_RANGE_LOW = -8;
constexpr int VC_MULTILDR_IMM_RANGE_HIGH = 7;
constexpr int VC_MULTILDR_IMM_BIT_LENGTH = 4;
constexpr int VC_MULTILDR_IMM_SHIFT = 1;
constexpr int VC_GATHERLDR_IMM_BIT_LENGTH = 5;
constexpr int VC_GATHERLDR_IMM_BLOCK = 31;
constexpr int VC_LDR_PG_RANGE = 7;
constexpr int VC_SCATTSTR_IMM_BLOCK = 31;
constexpr int VC_SCATTSTR_IMM_BIT_LENGTH = 5;
constexpr int VC_STR_PG_RANGE = 7;
constexpr int VC_CONTISTR_IMM_BIT_LENGTH = 4;
constexpr int VC_CONTISTR_IMM_RANGE_LOW = -8;
constexpr int VC_CONTISTR_IMM_RANGE_HIGH = 7;
constexpr int VC_CONTISTR_PG_RANGE = 7;
constexpr int VC_STR_IMM_RANGE_LOW = -256;
constexpr int VC_STR_IMM_RANGE_HIGH = 255;
constexpr int MC_ZASLICE_INDEX_RANGE_BEGIN = 12;     // slice index ws can only be w12 ~ w15
constexpr int MC_ZASLICE_INDEX_RANGE_END = 15;     // slice index ws can only be w12 ~ w15
constexpr int MC_ZASLICE_OFFS_B_RANGE = 15;    // 8 bits slice offset can only be 0 ~ 15
constexpr int MC_ZASLICE_OFFS_H_RANGE = 7;     // 16 bits slice offset can only be 0 ~ 7
constexpr int MC_ZASLICE_OFFS_S_RANGE = 3;     // 32 bits slice offset can only be 0 ~ 3
constexpr int MC_ZATILE_B_RANGE = 0;     // za tile for b can only be za0
constexpr int MC_ZATILE_H_RANGE = 1;     // za tile for h can only be za0 ~ za1
constexpr int MC_ZATILE_S_RANGE = 3;     // za tile for s can only be za0 ~ za3
constexpr int MC_PG_RANGE = 7;     // sme operation pg can only be 0 ~ 7


constexpr int RINDEX0 = 0;
constexpr int RINDEX1 = 1;
constexpr int RINDEX2 = 2;
constexpr int RINDEX3 = 3;
constexpr int RINDEX4 = 4;
constexpr int RINDEX5 = 5;
constexpr int RINDEX6 = 6;
constexpr int RINDEX7 = 7;
constexpr int RINDEX8 = 8;
constexpr int RINDEX9 = 9;
constexpr int RINDEX10 = 10;
constexpr int RINDEX11 = 11;
constexpr int RINDEX12 = 12;
constexpr int RINDEX13 = 13;
constexpr int RINDEX14 = 14;
constexpr int RINDEX15 = 15;
constexpr int RINDEX16 = 16;
constexpr int RINDEX17 = 17;
constexpr int RINDEX18 = 18;
constexpr int RINDEX19 = 19;
constexpr int RINDEX20 = 20;
constexpr int RINDEX21 = 21;
constexpr int RINDEX22 = 22;
constexpr int RINDEX23 = 23;
constexpr int RINDEX24 = 24;
constexpr int RINDEX25 = 25;
constexpr int RINDEX26 = 26;
constexpr int RINDEX27 = 27;
constexpr int RINDEX28 = 28;
constexpr int RINDEX29 = 29;
constexpr int RINDEX30 = 30;
constexpr int RINDEX31 = 31;
constexpr int SPRINDEX = 31;
constexpr uint64_t SP_IDX = 31;

constexpr int MAX_FNAME_LEN = 256;
constexpr int INSTRUCTIONS_SIZE_4_BYTES = 4;
constexpr int DATA_SIZE_8_BYTES = 8;
constexpr int STEP_4 = 4;
constexpr int STEP_2 = 2;
constexpr int CODE_SIZE_256_1024 = 256 * 1024;
constexpr int VAL_65520 = 65520;
constexpr int SIZE_NEG_3 = -3;
constexpr int SIZE_NEG_16 = -16;
constexpr int SIZE_NEG_32 = -32;
constexpr int VAL_0 = 0;
constexpr int VAL_1 = 1;
constexpr int VAL_2 = 2;
constexpr int VAL_3 = 3;
constexpr int VAL_4 = 4;
constexpr int VAL_5 = 5;
constexpr int VAL_6 = 6;
constexpr int VAL_7 = 7;
constexpr int VAL_8 = 8;
constexpr int VAL_9 = 9;
constexpr int VAL_10 = 10;
constexpr int VAL_11 = 11;
constexpr int VAL_12 = 12;
constexpr int VAL_13 = 13;
constexpr int VAL_15 = 15;
constexpr int VAL_16 = 16;
constexpr int VAL_19 = 19;
constexpr int VAL_20 = 20;
constexpr int VAL_22 = 22;
constexpr int VAL_24 = 24;
constexpr int VAL_25 = 25;
constexpr int VAL_26 = 26;
constexpr int VAL_27 = 27;
constexpr int VAL_28 = 28;
constexpr int VAL_29 = 29;
constexpr int VAL_30 = 30;
constexpr int VAL_31 = 31;
constexpr int VAL_32 = 32;
constexpr int VAL_51 = 51;
constexpr int VAL_63 = 63;
constexpr int VAL_64 = 64;
constexpr int VAL_90 = 90;
constexpr int VAL_99 = 99;
constexpr int VAL_180 = 180;
constexpr int VAL_270 = 270;
constexpr int VAL_128 = 128;
constexpr int VAL_255 = 255;
constexpr int VAL_1023 = 1023;
constexpr int VAL_1024 = 1024;

constexpr int SIZE_NEG_256 = -256;

constexpr int BIT_4 = 4;
constexpr int BIT_8 = 8;
constexpr int BIT_16 = 16;
constexpr int BIT_32 = 32;
constexpr int BIT_63 = 63;
constexpr int BIT_64 = 64;
constexpr int BIT_128 = 128;
constexpr int BIT_255 = 255;

constexpr int LANE_1 = 1;
constexpr int LANE_2 = 2;
constexpr int LANE_4 = 4;
constexpr int LANE_8 = 8;
constexpr int LANE_16 = 16;
constexpr int POS_1 = 1;
constexpr int POS_2 = 2;

constexpr const char* DUMP_BASE_FNAME = "dnnl_dump_cpu_";
constexpr const char* DUMP_EXT_FNAME = ".bin";

constexpr uintptr_t AUTO_GROW_VAL = 1;
constexpr uintptr_t DONT_SET_PROTECT_RWE_VAL = 2;

} // ARMCG

#endif  // CG_ARM_CONST_HPP
