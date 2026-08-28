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
#include "kudnn_brgemm_utils.hpp"

#include <algorithm>

#include "service/kudnn_err_codes.hpp"
#include "service/kudnn_exception.hpp"
#include "tools/cg_cpu_isa_traits.hpp"
#include "types/kudnn_data_type.hpp"

#include "kudnn_brgemm_kernel.hpp"
#include "kudnn_brgemm_types.hpp"

namespace KuDNN {

namespace {

constexpr IntType BRGEMM_UTILS_NUM_2 = 2;
constexpr IntType BRGEMM_UTILS_NUM_3 = 3;
constexpr IntType BRGEMM_UTILS_NUM_4 = 4;
constexpr IntType BRGEMM_UTILS_NUM_8 = 8;

IntType CalculateLdbParams(BrgemmT *brg, const IntType tryLdBlock2)
{
    brg->ldBlock2 = tryLdBlock2;
    brg->ldb2 = brg->ldb / brg->ldBlock2;
    brg->ldb2Tail = brg->ldb % brg->ldBlock2;
    brg->embdBcst = brg->isF32 && brg->ldb2Tail && brg->ldb2 == 0
        && ARMCG::IsSuperset(brg->isaImpl, ARMCG::CPUISAT::VC_512);
    if (brg->ldb2 == 0) {
        brg->ldBlock2 = std::max(brg->ldb2Tail, static_cast<IntType>(1));
    }
    const IntType adjLdBlock2 = brg->ldb2 != 0 ? brg->ldBlock2 : brg->ldb2Tail;

    return std::max(adjLdBlock2, static_cast<IntType>(1));
}

IntType CalculateMaxBcastBlock(BrgemmT *brg, const IntType adjLdBlock2)
{
    // make sure some code is unused and delete them
    bool isAssertSuccess = (brg->zpTypeA == BrgemmBroadcastT::UNDEFINED) && (!brg->isBF16Emu) && (!brg->isInt8);
    if (!isAssertSuccess) {
        throw Service::Unsupported("");
    }

    const bool reqCompensation = brg->reqS8S8Compensation || brg->zpTypeA != BrgemmBroadcastT::UNDEFINED;
    const bool reqZpACompPads = (brg->reqCalCompPads || brg->brgattr.maxTopVpad > 0 || brg->brgattr.maxBottomVpad > 0)
        && brg->zpTypeA != BrgemmBroadcastT::UNDEFINED;
    const IntType betaRegs = !(brg->beta == 0.0f || brg->beta == 1.0f);
    const IntType maxIsaRegs = ARMCG::IsaNumVregs(brg->isaImpl);
    const IntType vnniGranularity = DataTypeVnniGranularity(brg->dtB);
    const IntType maxBcstRegs = (maxIsaRegs - adjLdBlock2 * vnniGranularity) / (adjLdBlock2 + 1);
    // adjLdBlock2 already removes the necessary registers for embdBcst
    auto maxRegCount = maxIsaRegs - maxBcstRegs - betaRegs - reqCompensation - reqZpACompPads;
    IntType maxBcastBlock = maxRegCount - adjLdBlock2 * vnniGranularity;
    maxBcastBlock /= adjLdBlock2;
    maxBcastBlock = maxBcastBlock > BRGEMM_UTILS_NUM_8 ? BRGEMM_UTILS_NUM_8 : maxBcastBlock;

    return maxBcastBlock;
}

Element::TypeT GetAccumDatatype(BrgemmT *brg)
{
    // make sure InitKernelDatatype is previously called
    bool isAssertSuccess = brg->isF32 || brg->isBF16 || brg->isF16 || brg->isInt8;
    if (!isAssertSuccess) {
        throw Service::LogicError("");
    }

    return brg->isInt8 ? Element::TypeT::S32 : Element::TypeT::F32;
}

void InitCommonConf(BrgemmT *brg, BrgemmBatchKindT type, float alpha, float beta, const BrgemmStridesT *strides)
{
    brg->alpha = alpha;
    brg->beta = beta;
    brg->strideA = strides ? strides->strideA : 0;
    brg->strideB = strides ? strides->strideB : 0;
    brg->type = type;
    brg->withSum = false;
    brg->sumScale = 0;
    brg->sumZp = 0;
    brg->withEltwise = false;
    brg->withScales = false;
    brg->withBias = false;
    brg->withWeightsScaleAdjust = false;
}

Status InitKernelDatatype(BrgemmT *brg, Element::TypeT dtA, Element::TypeT dtB)
{
    bool check = brg && dtA != Element::TypeT::UNDEFINED && dtB != Element::TypeT::UNDEFINED;
    if (!check) {
        return Status::BAD_ARGUMENTS;
    }

    brg->isF32 = dtA == Element::TypeT::F32 && dtB == Element::TypeT::F32;
    brg->isBF16 = dtA == Element::TypeT::BF16 && dtB == Element::TypeT::BF16;
    brg->isF16 = dtA == Element::TypeT::F16 && dtB == Element::TypeT::F16;
    brg->isInt8 = (dtA == Element::TypeT::S8 || dtA == Element::TypeT::U8)
        && (dtB == Element::TypeT::S8 || dtB == Element::TypeT::U8);
    if (!(brg->isF32 || brg->isBF16 || brg->isF16 || brg->isInt8)) {
        return Status::UNSUPPORTED;
    }

    return Status::SUCCESS;
}

void SetBrgVmm(BrgemmT *brg)
{
    brg->isZmm = true
        && ARMCG::MayIUse(ARMCG::CPUISAT::VC_512)
        && ARMCG::IsSuperset(brg->isaImpl, ARMCG::CPUISAT::VC_512);
    brg->isYmm = !brg->isZmm
        && ARMCG::MayIUse(ARMCG::CPUISAT::VC_256)
        && ARMCG::IsSuperset(brg->isaImpl, ARMCG::CPUISAT::VC_256);
}

Status SetIsaImpl(BrgemmT *brg)
{
    auto isIsaOk = [&brg](ARMCG::CPUISAT isa) {
        return ARMCG::MayIUse(isa) && (brg->isaUser == ARMCG::CPUISAT::ISA_UNDEF || brg->isaUser == isa);
    };
    if (brg->isF32) {
        brg->isaImpl = isIsaOk(ARMCG::CPUISAT::VC_512) ? ARMCG::CPUISAT::VC_512 : (isIsaOk(ARMCG::CPUISAT::VC_256)
            ? ARMCG::CPUISAT::VC_256 : ARMCG::CPUISAT::ISA_UNDEF);
        return Status::SUCCESS;
    }

    return Status::UNSUPPORTED;
}

} // namespace

Status InitBrgemmConf(BrgemmT *brg, ARMCG::CPUISAT isa, BrgemmBatchKindT type, Element::TypeT dtA, Element::TypeT dtB,
    BrgemmLayoutT layout, float alpha, float beta, IntType ldA, IntType ldB, IntType ldC, IntType m, IntType n,
    IntType k, const BrgemmStridesT *strides, bool isBF32)
{
    if (!brg) {
        return Status::BAD_ARGUMENTS;
    }

    brg->layout = layout;
    brg->isBF32 = isBF32;
    InitCommonConf(brg, type, alpha, beta, strides);
    brg->dtA = brg->IsRowMajor() ? dtA : dtB;
    brg->dtB = brg->IsRowMajor() ? dtB : dtA;
    CHECK(InitKernelDatatype(brg, brg->dtA, brg->dtB));
    brg->dtC = GetAccumDatatype(brg);
    brg->dtD = brg->dtC;
    brg->dtBias = brg->dtC;
    brg->typeSizeA = Element::Type(brg->dtA).GetSize();
    brg->typeSizeB = Element::Type(brg->dtB).GetSize();
    brg->typeSizeC = Element::Type(brg->dtC).GetSize();
    brg->typeSizeD = Element::Type(brg->dtD).GetSize();
    brg->isaUser = isa;
    CHECK(SetIsaImpl(brg));
    brg->hasInt8Vnni = true;
    brg->reqS8S8Compensation = brg->isInt8 && brg->dtA == Element::TypeT::S8;
    SetBrgVmm(brg);
    brg->ldA = brg->IsRowMajor() ? static_cast<IntType>(ldA) : static_cast<IntType>(ldB);
    brg->ldB = brg->IsRowMajor() ? static_cast<IntType>(ldB) : static_cast<IntType>(ldA);
    brg->ldC = static_cast<IntType>(ldC);
    brg->ldD = static_cast<IntType>(ldC);
    brg->bcastDim = brg->IsRowMajor() ? static_cast<IntType>(m) : static_cast<IntType>(n);
    brg->loadDim = brg->IsRowMajor() ? static_cast<IntType>(n) : static_cast<IntType>(m);
    brg->reduceDim = static_cast<IntType>(k);
    brg->bdBlock2 = 0;
    brg->bdb2 = 0;
    brg->bdb2Tail = 0;
    const bool isBInVnniFormat = false;
    brg->ldStep = isBInVnniFormat ? DataTypeVnniGranularity(brg->dtB) : 1;
    const bool hasNoVnniComputeInstruction = false;
    brg->rdStep = hasNoVnniComputeInstruction ? 1 : DataTypeVnniGranularity(brg->dtB);

    return Status::SUCCESS;
}

Status BrgemmBlocking(BrgemmT *brg)
{
    bool check = brg && !brg->isDgmm;
    if (!check) {
        return Status::BAD_ARGUMENTS;
    }

    CHECK(SetIsaImpl(brg));
    check &= brg->isaImpl != ARMCG::CPUISAT::ISA_UNDEF;
    if (!check) {
        return Status::UNSUPPORTED;
    }
    SetBrgVmm(brg);
    check &= brg->isZmm || brg->isYmm;
    if (!check) {
        return Status::UNSUPPORTED;
    }
    const IntType simdW = ARMCG::IsSuperset(brg->isaImpl, ARMCG::CPUISAT::VC_512)
        ? ARMCG::CpuIsaTraits<ARMCG::CPUISAT::VC_512>::VLEN / sizeof(float)
        : ARMCG::CpuIsaTraits<ARMCG::CPUISAT::VC_256>::VLEN / sizeof(float);
    brg->ldBlock = simdW;
    brg->ldb = brg->loadDim / brg->ldBlock;
    brg->ldbTail = brg->loadDim % brg->ldBlock;
    IntType adjLdBlock2 = CalculateLdbParams(brg, BRGEMM_MC_AVAILABLE ? BRGEMM_UTILS_NUM_3 : BRGEMM_UTILS_NUM_4);
    IntType maxBcastBlock = CalculateMaxBcastBlock(brg, adjLdBlock2);
    // reduce ldBlock2 to allow a larger bdBlock
    const IntType maxVpad = std::max(brg->brgattr.maxTopVpad, brg->brgattr.maxBottomVpad);
    if (ARMCG::IsSuperset(brg->isaImpl, ARMCG::CPUISAT::VC_256) && maxBcastBlock < maxVpad) {
        adjLdBlock2 = CalculateLdbParams(brg, BRGEMM_UTILS_NUM_2);
        maxBcastBlock = CalculateMaxBcastBlock(brg, adjLdBlock2);
    }
    if (BRGEMM_MC_AVAILABLE) {
        maxBcastBlock = simdW;
    }
    const IntType minBlock = 1;
    float bestBdBlockEff = 0.0f;
    brg->bdBlock = 1;
    for (IntType bdBlock = maxBcastBlock; bdBlock >= minBlock; bdBlock--) {
        const auto bdBlockDisb =
            static_cast<float>(brg->bcastDim) / (((brg->bcastDim + bdBlock - 1) / bdBlock) * bdBlock);
        const auto brgemmMicrokernelEff =
            (static_cast<float>(adjLdBlock2) * bdBlock) / ((adjLdBlock2 + bdBlock) * maxBcastBlock);
        const auto bdBlockEff = bdBlockDisb * brgemmMicrokernelEff;
        float blockFootPrint = static_cast<float>(brg->typeSizeA) * (bdBlock * brg->reduceDim);
        if (blockFootPrint <= static_cast<float>(ARMCG::GetCoreCacheSize(1)) && (bdBlockEff > bestBdBlockEff)) {
            brg->bdBlock = bdBlock;
            bestBdBlockEff = bdBlockEff;
        }
    }
    brg->bdb = brg->bcastDim / brg->bdBlock;
    brg->bdbTail = brg->bcastDim % brg->bdBlock;
    const IntType rdUnroll = 4;
    const IntType vnniGranularity = DataTypeVnniGranularity(brg->dtA);
    brg->rdBlock = (BRGEMM_MC_AVAILABLE ? simdW : rdUnroll) * vnniGranularity;
    brg->rdb = brg->reduceDim / brg->rdBlock;
    brg->rdbTail = brg->reduceDim % brg->rdBlock;
    brg->isMTail = false;

    return Status::SUCCESS;
}

} // namespace KuDNN
