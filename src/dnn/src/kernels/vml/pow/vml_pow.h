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

#ifndef VML_POW_H
#define VML_POW_H

#define MAX_POWXF_SP_TH_HI 10.0f
#define MAX_POWXF_SP_TH_LO FLOAT_ZERO

#define POWF_X_TH_LO FLOAT_SUBNORMAL_BIT
#define POWF_X_TH_HI FLOAT_INF_BIT
#define POWF_X_TH (FLOAT_INF_BIT - FLOAT_SUBNORMAL_BIT)

#define POWF_LOG2_BIT_N 5U
#define POWF_LOG2_N (1 << (POWF_LOG2_BIT_N - 1))
#define POWF_LOG2_POLY_N POWF_LOG2_N

extern const float G_VML_VSPOW_LOG2_TAB_VectorComputation[POWF_LOG2_N * 0x2];
extern const float G_VML_VSPOW_LOG2_POLY_LOW_TAB_VectorComputation[POWF_LOG2_POLY_N * 0x2];

#endif
