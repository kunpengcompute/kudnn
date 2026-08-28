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
 
#include "kudnn_brgemm.hpp"

#include <algorithm>

#include "internal_funcs.hpp"
#include "service/kudnn_err_codes.hpp"
#include "service/kudnn_exception.hpp"
#include "tools/cg_cpu_isa_traits.hpp"
#include "types/kudnn_data_type.hpp"
#include "types/kudnn_tensor_info.hpp"

#include "kudnn_brgemm_kernel.hpp"
#include "kudnn_brgemm_types.hpp"
#include "kudnn_brgemm_utils.hpp"

namespace KuDNN {
Status BrgemmDescInit(BrgemmT *brg, ARMCG::CPUISAT isa, BrgemmBatchKindT kind, Element::TypeT dtA, Element::TypeT dtB,
    bool transA, bool transB, BrgemmLayoutT layout, float alpha, float beta, IntType ldA, IntType ldB, IntType ldC,
    IntType m, IntType n, IntType k, const BrgemmStridesT *strides)
{
    bool check = brg && !transA && !transB;
    if (!check) {
        return Status::BAD_ARGUMENTS;
    }

    CHECK(InitBrgemmConf(brg, isa, kind, dtA, dtB, layout, alpha, beta, ldA, ldB, ldC, m, n, k, strides));
    check &= m > 0 && n > 0 && k > 0;
    check &= layout == BrgemmLayoutT::ROW_MAJOR ? ldA >= k : ldA >= m && ldB >= k && ldC >= m;
    if (!check) {
        return Status::BAD_ARGUMENTS;
    }
    CHECK(BrgemmBlocking(brg));
    check &= brg->isF32 || brg->isBF16 || brg->isF16 || brg->isInt8;
    if (!check) {
        return Status::BAD_ARGUMENTS;
    }

    return Status::SUCCESS;
}

Status BrgemmDescSetAttr(BrgemmT *brg, const BrgemmAttrT &brgattr)
{
    if (!brg) {
        return Status::BAD_ARGUMENTS;
    }
    // negative padding is not supported
    bool check = brgattr.maxTopVpad >= 0 && brgattr.maxBottomVpad >= 0;
    // virtual padding size is restricted by BrgemmT::MAX_VPAD
    check &= brg->isDgmm || (brgattr.maxTopVpad <= BrgemmT::MAX_VPAD && brgattr.maxBottomVpad <= BrgemmT::MAX_VPAD);
    // virtual padding is supported for BrgemmLayoutT::ROW_MAJOR
    check &= (brgattr.maxTopVpad == 0 && brgattr.maxBottomVpad == 0) || brg->layout == BrgemmLayoutT::ROW_MAJOR;
    if (!check) {
        return Status::UNSUPPORTED;
    }
    // virtual padding is restricted by bdBlock size due to brgemm kernel implementation
    bool hintBlockingSet = false
        || brgattr.hintBdBlock != 0 || brgattr.hintBdBlock2 != 0
        || brgattr.hintLdBlock != 0 || brgattr.hintLdBlock2 != 0
        || brgattr.hintLoadNtA != BrgemmKernelHintNtT::UNDEFINED
        || brgattr.hintLoadNtB != BrgemmKernelHintNtT::UNDEFINED;
    const IntType maxVpad = std::max(brgattr.maxTopVpad, brgattr.maxBottomVpad);
    check &= brg->isDgmm || maxVpad <= (brg->bdbTail > 0 ? brg->bdbTail : brg->bdBlock);
    if (!check) {
        return Status::UNSUPPORTED;
    }
    // make sure some code is unused and delete them
    bool isAssertSuccess = brgattr.fpmathMode == FpmathModeT::STRICT && !brgattr.useUker && !hintBlockingSet
        && !brgattr.bdMaskLevel && maxVpad <= 0;
    if (!isAssertSuccess) {
        throw Service::Unsupported("");
    }

    brg->brgattr = brgattr;
    brg->ldA2 = brgattr.ldA2 != 0 ? brgattr.ldA2 : brg->ldA;
    brg->ldB2 = brgattr.ldB2 != 0 ? brgattr.ldB2 : brg->ldB;
    brg->ldC2M = brgattr.ldC2M != 0 ? brgattr.ldC2M : brg->ldC;
    brg->ldC2N = brgattr.ldC2N != 0 ? brgattr.ldC2N : brg->ldBlock;
    brg->isBlocked = false
        || brg->ldA2 != brg->ldA
        || brg->ldB2 != brg->ldB
        || brg->ldC2M != brg->ldC
        || brg->ldC2N != brg->ldBlock;
    if (!(!(brg->isBlocked) || !!(brg->layout == BrgemmLayoutT::ROW_MAJOR))) {
        return Status::BAD_ARGUMENTS;
    }
    brg->prfA = brgattr.hintPrfA;
    brg->prfB = brgattr.hintPrfB;
    brg->prfC = brgattr.hintPrfC;
    if (brgattr.hintInnermostLoop != BrgemmKernelInnermostLoopT::UNDEFINED) {
        brg->innermostLoop = brgattr.hintInnermostLoop;
    }
    if (brgattr.hintPrefetching == BrgemmKernelPrefetchingT::PRF1 && brg->prfC.dist1 < 0) {
        brg->prfC.dist1 = 0;
    }
    if (brgattr.hintPrefetching == BrgemmKernelPrefetchingT::PRF2 && brg->prfC.dist2 < 0) {
        brg->prfC.dist2 = 0;
    }

    return Status::SUCCESS;
}

Status BrgemmDescSetPostops(BrgemmT *brg, const void *attr, TensorInfo *dst, IntType ldD, Element::TypeT dtBias)
{
    bool check = brg && dst;
    if (!check) {
        return Status::BAD_ARGUMENTS;
    }

    brg->attr = attr;
    brg->dst = dst;
    brg->withBias = dtBias == Element::TypeT::UNDEFINED ? false : true;
    brg->dtBias = dtBias;
    brg->typeSizeBias = dtBias == Element::TypeT::UNDEFINED ? 0 : Element::Type(dtBias).GetSize();
    brg->ldD = ldD;
    const auto dtD = Element::TypeT(dst->GetType());
    if ((brg->dtA == Element::TypeT::F32 && brg->dtB == Element::TypeT::F32)
        && (!Service::OneOf(dtD, Element::TypeT::F32))
        && (!Service::OneOf(dtBias, Element::TypeT::F32, Element::TypeT::UNDEFINED))) {
        return Status::UNSUPPORTED;
    }
    if ((brg->dtA == Element::TypeT::BF16 && brg->dtB == Element::TypeT::BF16)
        && (!Service::OneOf(dtD, Element::TypeT::F32, Element::TypeT::BF16))
        && (!Service::OneOf(dtBias, Element::TypeT::F32, Element::TypeT::BF16, Element::TypeT::UNDEFINED))) {
        return Status::UNSUPPORTED;
    }
    if ((brg->dtA == Element::TypeT::U8 && brg->dtB == Element::TypeT::S8)
        && (!Service::OneOf(dtD, Element::TypeT::F32, Element::TypeT::S32, Element::TypeT::S8, Element::TypeT::U8))
        && (!Service::OneOf(dtBias, Element::TypeT::F32, Element::TypeT::S32, Element::TypeT::BF16, Element::TypeT::S8,
        Element::TypeT::U8, Element::TypeT::UNDEFINED))) {
        return Status::UNSUPPORTED;
    }
    brg->dtD = dtD;
    brg->typeSizeD = Element::Type(dtD).GetSize();
    if (brg->isInt8 || brg->dtD == Element::TypeT::BF16) {
        return Status::UNSUPPORTED;
    }
    // [unimplemented] set postops

    return Status::SUCCESS;
}

Status BrgemmDescFinalize(BrgemmT *brg)
{
    Service::Unused(brg);

    return Status::SUCCESS;
}

Status BrgemmKernelCreate(BrgemmBaseKernelT **kernel, const BrgemmT &brg)
{
    if (!kernel) {
        return Status::BAD_ARGUMENTS;
    }

    if (brg.isDgmm) {
        return Status::UNSUPPORTED; // [delete] BrdgmmKernelT
    } else {
        CHECK(SafePtrAssign<BrgemmBaseKernelT>(*kernel, new BrgemmKernelT(brg)));
    }
    if (!(*kernel)) {
        throw Service::LogicError("");
    }

    return (*kernel)->CreateKernel();
}

void BrgemmKernelExecute(const BrgemmBaseKernelT *kernel, IntType bs, const void *ptrA, const void *ptrB,
    const BrgemmBatchElementT *batch, void *ptrC, void *scratch)
{
    BrgemmKernelParamsT params;
    params.ptrA = ptrA;
    params.ptrB = ptrB;
    params.batch = batch;
    params.ptrC = ptrC;
    params.ptrBias = nullptr;
    params.ptrD = ptrC;
    params.ptrBuf = scratch;
    params.doPostOps = 0;
    params.doApplyComp = 0;
    params.bs = bs;
    params.skipAccm = 0;
    if (!kernel) {
        throw Service::LogicError("");
    }
    (*kernel)(&params);
}

void BrgemmKernelExecutePostops(const BrgemmBaseKernelT *kernel, IntType bs, const void *ptrA, const void *ptrB,
    const BrgemmBatchElementT *batch, void *ptrC, void *ptrD, const BrgemmPostOpsDataT &postOpsData, void *scratch)
{
    BrgemmKernelParamsT params;
    params.ptrA = ptrA;
    params.ptrB = ptrB;
    params.batch = batch;
    params.ptrC = ptrC;
    params.ptrBias = postOpsData.bias;
    params.ptrD = ptrD;
    params.ptrScales = postOpsData.scales;
    params.ptrBuf = scratch;
    params.doPostOps = postOpsData.doOnlyComp || postOpsData.doOnlyZpAVal ? 0 : 1;
    params.doApplyComp = postOpsData.doOnlyZpAVal ? 0 : 1;
    params.bs = bs;
    params.postOpsBinaryRhsArgVec = postOpsData.binaryPostOpsRhs;
    params.ocLogicalOff = postOpsData.ocLogicalOff;
    params.firstMbMatrixAddrOff = postOpsData.firstMbMatrixAddrOff;
    params.dstRowLogicalOff = postOpsData.dstRowLogicalOff;
    params.dataCPtr = postOpsData.dataCPtr;
    params.aZpCompensations = postOpsData.aZpCompensations;
    params.bZpCompensations = postOpsData.bZpCompensations;
    params.cZpValues = postOpsData.cZpValues;
    params.skipAccm = postOpsData.skipAccumulation ? 1 : 0;
    params.zpAVal = postOpsData.zpAVal;
    params.ptrDstScales = postOpsData.dstScales;
    if (!kernel) {
        throw Service::LogicError("");
    }
    (*kernel)(&params);
}

} // namespace KuDNN
