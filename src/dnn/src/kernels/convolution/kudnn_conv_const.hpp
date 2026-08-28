
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

#ifndef KUDNN_CONV_CONST_HPP
#define KUDNN_CONV_CONST_HPP

#include "types/kudnn_data_type.hpp"

namespace KuDNN {
    constexpr IntType LEVEL0 = 0;
    constexpr IntType LEVEL1 = 1;
    constexpr IntType LEVEL2 = 2;
    constexpr IntType LEVEL3 = 3;
    constexpr IntType KERNEL_DIMS_1 = 1;
    constexpr IntType KERNEL_DIMS_2 = 2;
    constexpr IntType KERNEL_DIMS_3 = 3;
    constexpr IntType KERNEL_DIMS_4 = 4;
    constexpr IntType KERNEL_DIMS_5 = 5;
    constexpr IntType KERNEL_DIMS_6 = 6;
    constexpr IntType KERNEL_C_LEN1 = 1;
    constexpr IntType KERNEL_C_LEN2 = 2;
    constexpr IntType KERNEL_C_LEN3 = 3;
    constexpr IntType KERNEL_C_LEN4 = 4;
    constexpr IntType KERNEL_C_LEN5 = 5;
    constexpr IntType KERNEL_C_LEN6 = 6;
    constexpr IntType KERNEL_C_LEN8 = 8;
    constexpr IntType KERNEL_C_LEN16 = 16;
    constexpr IntType KERNEL_C_LEN24 = 24;
    constexpr IntType KERNEL_C_LEN32 = 32;
    constexpr IntType IC_LEN1 = 1;
    constexpr IntType IC_LEN2 = 2;
    constexpr IntType IC_LEN4 = 4;
    constexpr IntType IC_LEN8 = 8;
    constexpr IntType IC_LEN16 = 16;
    constexpr IntType IC_LEN32 = 32;
    constexpr IntType KERNEL_H_LEN1 = 1;
    constexpr IntType KERNEL_H_LEN2 = 2;
    constexpr IntType KERNEL_H_LEN3 = 3;
    constexpr IntType KERNEL_H_LEN8 = 8;
    constexpr IntType KERNEL_UR_W_LEN1 = 1;
    constexpr IntType KERNEL_UR_W_LEN2 = 2;
    constexpr IntType KERNEL_UR_W_LEN3 = 3;
    constexpr IntType KERNEL_UR_W_LEN4 = 4;
    constexpr IntType KERNEL_UR_W_LEN5 = 5;
    constexpr IntType KERNEL_UR_W_LEN6 = 6;
    constexpr IntType KERNEL_UR_W_LEN7 = 7;
    constexpr IntType KERNEL_UR_W_LEN8 = 8;
    constexpr IntType KERNEL_UR_W_LEN9 = 9;
    constexpr IntType KERNEL_UR_W_LEN10 = 10;
    constexpr IntType KERNEL_UR_W_LEN11 = 11;
    constexpr IntType KERNEL_UR_W_LEN12 = 12;
    constexpr IntType KERNEL_UR_W_LEN13 = 13;
    constexpr IntType KERNEL_UR_W_LEN14 = 14;
    constexpr IntType KERNEL_UR_W_LEN15 = 15;
    constexpr IntType KERNEL_UR_W_LEN16 = 16;
    constexpr IntType KERNEL_UR_W_LEN17 = 17;
    constexpr IntType KERNEL_UR_W_LEN18 = 18;
    constexpr IntType KERNEL_UR_W_LEN19 = 19;
    constexpr IntType KERNEL_UR_W_LEN20 = 20;
    constexpr IntType KERNEL_UR_W_LEN21 = 21;
    constexpr IntType KERNEL_UR_W_LEN22 = 22;
    constexpr IntType KERNEL_UR_W_LEN23 = 23;
    constexpr IntType KERNEL_UR_W_LEN24 = 24;
    constexpr IntType KERNEL_UR_W_LEN25 = 25;
    constexpr IntType KERNEL_UR_W_LEN26 = 26;
    constexpr IntType KERNEL_UR_W_LEN27 = 27;
    constexpr IntType KERNEL_UR_W_LEN28 = 28;
    constexpr IntType KERNEL_UR_W_LEN29 = 29;
    constexpr IntType KERNEL_UR_W_LEN30 = 30;
    constexpr IntType KERNEL_UR_W_LEN31 = 31;
    constexpr IntType KERNEL_UR_W_LEN32 = 32;
    constexpr IntType REG_INDEX0 = 0;
    constexpr IntType REG_INDEX1 = 1;
    constexpr IntType REG_INDEX2 = 2;
    constexpr IntType REG_INDEX3 = 3;
    constexpr IntType REG_INDEX4 = 4;
    constexpr IntType REG_INDEX5 = 5;
    constexpr IntType REG_INDEX6 = 6;
    constexpr IntType REG_INDEX7 = 7;
    constexpr IntType REG_INDEX8 = 8;
    constexpr IntType REG_INDEX9 = 9;
    constexpr IntType REG_INDEX10 = 10;
    constexpr IntType REG_INDEX11 = 11;
    constexpr IntType REG_INDEX12 = 12;
    constexpr IntType REG_INDEX13 = 13;
    constexpr IntType REG_INDEX14 = 14;
    constexpr IntType REG_INDEX15 = 15;
    constexpr IntType REG_INDEX16 = 16;
    constexpr IntType REG_INDEX17 = 17;
    constexpr IntType REG_INDEX18 = 18;
    constexpr IntType REG_INDEX19 = 19;
    constexpr IntType REG_INDEX20 = 20;
    constexpr IntType REG_INDEX21 = 21;
    constexpr IntType REG_INDEX22 = 22;
    constexpr IntType REG_INDEX23 = 23;
    constexpr IntType REG_INDEX24 = 24;
    constexpr IntType REG_INDEX25 = 25;
    constexpr IntType REG_INDEX26 = 26;
    constexpr IntType REG_INDEX27 = 27;
    constexpr IntType REG_INDEX28 = 28;
    constexpr IntType REG_INDEX29 = 29;
    constexpr IntType REG_INDEX30 = 30;
    constexpr IntType REG_INDEX31 = 31;
    constexpr IntType REG_INDEX32 = 32;
    constexpr IntType THREAD_BLK_SHOLD1 = 1;
    constexpr IntType THREAD_BLK_SHOLD2 = 2;
    constexpr IntType LDRMAX = 255;
    constexpr IntType LDRMIN = -256;
    constexpr IntType STRMAX = 255;
    constexpr IntType STRMIN = -256;
    constexpr IntType LD1RWMAX = 252;
    constexpr IntType PRFMMAX = 32760;
    constexpr IntType PRFMMIN = 0;
    constexpr IntType PRFWMAX = 31;
    constexpr IntType PRFWMIN = -32;
    constexpr auto SMALL_SPATIAL = 14;
    constexpr int32_t KUDNN_A64FX_L2_EFFECTIVE_CAPACITY = ((666 - 128) * 1024);
    constexpr int32_t TYPE_SIZE = sizeof(float);
    constexpr int32_t HALF_TYPE_SIZE = sizeof(float) / 2;
    constexpr int32_t KER_REG_BASE_IDX = 28;
    constexpr IntType MAX_THREAD_NUM = 1200;
    constexpr IntType L2_CORE_CACHE_SIZE = 512 * 1024;
    constexpr IntType FLOAT_OFFSET_SHIFT = 2;
    constexpr IntType KERNEL_OFFSET_SHIFT = 6;
    constexpr IntType OW_BLOCK = 64;
    constexpr IntType GEMM_K_MASK = 3;
    constexpr IntType GEMM_OFFSET1 = 1;
    constexpr IntType GEMM_OFFSET2 = 2;
    constexpr IntType GEMM_OFFSET3 = 3;
    constexpr IntType GEMM_OFFSET4 = 4;
    constexpr SizeType CONV_THREAD_THREADS = 200000;
    constexpr IntType CACHE_ALIAS_1024 = 1024;
    constexpr IntType LIMIT_4 = 4;
    constexpr IntType LIMIT_7 = 7;
    constexpr IntType NB_CH_BLKING_2 = 2;
    constexpr IntType NB_CH_BLKING_3 = 3;
    constexpr IntType NB_CH_BLKING_4 = 4;
    constexpr IntType UR_W_3 = 3;
    constexpr IntType UR_W_4 = 4;
    constexpr IntType UR_W_6 = 6;
    constexpr IntType WINO_TRANS_MIN_MATRIX = 36;
    constexpr IntType WINO_MIN_MATRIX_WID = 6;
    constexpr IntType WINO_MIN_MATRIX_HEI = 6;
    constexpr IntType WINO_OUT_MIN_MATRIX_WEI = 4;
    constexpr IntType WINO_OUT_MIN_MATRIX_HEI = 4;
    constexpr IntType WEI_ENLARGE_NUM = 4;
    constexpr IntType VLEN_NEON_F32 = 4;
    constexpr float F_NUM_1 = 1.0f;
    constexpr float F_NUM_2 = 2.0f;
    constexpr float F_NUM_4 = 4.0f;
    constexpr float F_NUM_5 = 5.0f;
    constexpr float F_NUM_6 = 6.0f;
    constexpr float F_NUM_7 = 7.0f;
    constexpr float F_NUM_8 = 8.0f;
    constexpr float F_NUM_24 = 24.0f;
    constexpr float F_NUM_576 = 576.0f;
    constexpr IntType BRGEMM_UTILS_NUM_1200 = 1200;
    constexpr IntType BRGEMM_CONV_UTILS_NUM_2 = 2;
    constexpr IntType BRGEMM_CONV_UTILS_NUM_4 = 4;
    constexpr IntType BRGEMM_CONV_UTILS_NUM_5 = 5;
    constexpr IntType BRGEMM_CONV_UTILS_NUM_16 = 16;
    constexpr IntType BRGEMM_CONV_UTILS_NUM_32 = 32;
    constexpr IntType BRGEMM_CONV_UTILS_NUM_64 = 64;
    constexpr IntType BRGEMM_CONV_UTILS_NUM_400 = 400;
    constexpr IntType BRGEMM_CONV_UTILS_NUM_1001 = 1001;
    constexpr IntType BRGEMM_CONV_UTILS_NUM_1024 = 1024;
    constexpr IntType BRGEMM_CONV_UTILS_NUM_2048 = 2048;
    constexpr IntType BRGEMM_CONV_UTILS_NUM_4096 = 4096;
    constexpr IntType BRGEMM_CONV_UTILS_NUM_1000000 = 1000000;
    constexpr IntType BRGEMM_1X1_CONV_NUM_2 = 2;
    constexpr IntType BRGEMM_1X1_CONV_NUM_3 = 3;
    constexpr IntType BRGEMM_1X1_CONV_NUM_4 = 4;
    constexpr IntType BRGEMM_1X1_CONV_NUM_5 = 5;
    constexpr IntType BRGEMM_1X1_CONV_NUM_16 = 16;
    constexpr IntType BRGEMM_1X1_CONV_NUM_28 = 28;
    constexpr IntType BRGEMM_1X1_CONV_NUM_56 = 56;
    constexpr IntType BRGEMM_1X1_CONV_NUM_256 = 256;
    constexpr IntType BRGEMM_1X1_CONV_NUM_300 = 300;
    constexpr IntType BRGEMM_1X1_CONV_NUM_384 = 384;
    constexpr IntType BRGEMM_1X1_CONV_NUM_512 = 512;
    constexpr IntType BRGEMM_1X1_CONV_NUM_1024 = 1024;
    constexpr IntType S8_MAX_VAL = 127;
    constexpr IntType S8_MIN_VAL = -128;
    constexpr IntType U8_MAX_VAL = 255;
    constexpr IntType REG_STACK_OFFSET = 64;
    constexpr IntType VC_LEN_64 = 64;
    constexpr IntType VC_LEN_32 = 32;
    constexpr IntType VC_LEN_16 = 16;
    constexpr IntType VC_LEN_8 = 8;
    constexpr IntType NB_OW_STEP_1 = 1;
    constexpr IntType NB_OW_STEP_2 = 2;
    constexpr IntType OSCALE_LEN_8 = 8;
    constexpr IntType OSCALE_LEN_16 = 16;
    constexpr IntType REG_OFFSET_2 = 2;
    constexpr IntType REG_OFFSET_3 = 3;
    constexpr IntType REG_OFFSET_4 = 4;
    constexpr IntType REG_OFFSET_5 = 5;
} // KuDNN

#endif // KUDNN_CONV_CONST_HPP