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

#include <ATen/core/Tensor.h>
#include <torch/extension.h>

namespace common {
at::Tensor ScaledDotProductPagedAttention(const at::Tensor& query,
    const at::Tensor& kCache, const at::Tensor& vCache, const at::Tensor& blockTables,
    const at::Tensor& seqLens);
}

template<class T>
void paged_attention_v1_impl(T* __restrict__ out, const T* __restrict__ q,
    const T* __restrict__ kCache, const T* __restrict__ vCache, const int nHeadsKV,
    const int* __restrict__ blockTables, const int* __restrict__ seqLens, const int maxNumBlocksPerSeq,
    const int qStride, const int kvBlockStride, const int kvHeadStride,
    const int numSeqs, const int nHeads, const int headSize);