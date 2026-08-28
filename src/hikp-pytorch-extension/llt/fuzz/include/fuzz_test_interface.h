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

#pragma once
#include "fuzz_common.h"

void DtGatingAttentionFuzz(int testCount);
void DtTriangleMultiplicationFuzz(int testCount);
void DtOuterProductMeanFuzz(int testCount);
void DtMultiHeadAttentionFuzz(int testCount);
void DtLinearFuzz(int testCount);
void DtSigmoidFuzz(int testCount);
void DtFlashAttentionFuzz(int testCount);
void DtPagedAttentionFuzz(int testCount);
void DtAttentionFuzz(int testCount);