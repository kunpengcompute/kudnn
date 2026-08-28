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

#ifndef KUDNN_BRGEMM_HPP
#define KUDNN_BRGEMM_HPP

#include "service/kudnn_err_codes.hpp"
#include "tools/cg_cpu_isa_traits.hpp"
#include "types/kudnn_data_type.hpp"
#include "types/kudnn_tensor_info.hpp"

#include "kudnn_brgemm_containers.hpp"
#include "kudnn_brgemm_kernel.hpp"
#include "kudnn_brgemm_types.hpp"
namespace KuDNN {

Status BrgemmDescInit(BrgemmT *brg, ARMCG::CPUISAT isa, BrgemmBatchKindT kind, Element::TypeT dtA, Element::TypeT dtB,
    bool transA, bool transB, BrgemmLayoutT layout, float alpha, float beta, IntType ldA, IntType ldB, IntType ldC,
    IntType m, IntType n, IntType k, const BrgemmStridesT *strides = nullptr);

Status BrgemmDescSetAttr(BrgemmT *brg, const BrgemmAttrT &brgattr);

Status BrgemmDescSetPostops(BrgemmT *brg, const void *attr, TensorInfo *dst, IntType ldD,
    Element::TypeT dtBias = Element::TypeT::UNDEFINED);

Status BrgemmDescFinalize(BrgemmT *brg);

Status BrgemmKernelCreate(BrgemmBaseKernelT **kernel, const BrgemmT &brg);

Status BrgemmKernelDestory(BrgemmBaseKernelT *kernel);

void BrgemmKernelExecute(const BrgemmBaseKernelT *kernel, IntType bs, const void *ptrA, const void *ptrB,
    const BrgemmBatchElementT *batch, void *ptrC, void *scratch = nullptr);

void BrgemmKernelExecutePostops(const BrgemmBaseKernelT *kernel, IntType bs, const void *ptrA, const void *ptrB,
    const BrgemmBatchElementT *batch, void *ptrC, void *ptrD, const BrgemmPostOpsDataT &postOpsData,
    void *scratch = nullptr);

} // namespace KuDNN

#endif // KUDNN_BRGEMM_HPP
