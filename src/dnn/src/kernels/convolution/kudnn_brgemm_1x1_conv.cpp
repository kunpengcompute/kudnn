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

#include "kudnn_brgemm_1x1_conv.hpp"

#include <algorithm>
#include <cstdint>
#include <memory>
#include <vector>

#include "kudnn_brgemm.hpp"
#include "internal_funcs.hpp"
#include "service/kudnn_err_codes.hpp"
#include "service/kudnn_exception.hpp"
#include "tools/cg_cpu_isa_traits.hpp"
#include "types/kudnn_data_type.hpp"
#include "types/kudnn_propagation_type.hpp"
#include "types/kudnn_shape.hpp"
#include "types/kudnn_tensor_info.hpp"

#include "kudnn_brgemm_conv_utils.hpp"
#include "kudnn_brgemm_primitive_conf.hpp"

namespace KuDNN {

namespace {

inline IntType GetBrgIdx(IntType doInit, IntType isMTail, IntType isNTail, IntType isKTail)
{
    return doInit * BRGEMM_1X1_CONV_NUM_2 * BRGEMM_1X1_CONV_NUM_2 * BRGEMM_1X1_CONV_NUM_2
        + isMTail * BRGEMM_1X1_CONV_NUM_2 * BRGEMM_1X1_CONV_NUM_2
        + isNTail * BRGEMM_1X1_CONV_NUM_2
        + isKTail;
}

inline IntType NdimsPick(IntType ndims, IntType v5, IntType v4, IntType v3)
{
    switch (ndims) {
        case BRGEMM_1X1_CONV_NUM_5: return v5;
        case BRGEMM_1X1_CONV_NUM_4: return v4;
        case BRGEMM_1X1_CONV_NUM_3: return v3;
        default: break;
    }

    return 0;
}

} // namespace

Status KdnnBrgemm1x1ConvInitConf(KdnnBrgconvDataT &bcd, ARMCG::CPUISAT isa, const TensorInfo &src,
    const TensorInfo &weights, const TensorInfo &dst, const TensorInfo &bias, Shape strides, Shape dilates,
    Shape paddingL, Propagation prop, IntType nthreads)
{
    KdnnBrgemmConvConfT &bcp = bcd.bcp;
    Status s = BrgemmConvInit1x1Conf(bcp, isa, src, weights, dst, bias, strides, paddingL, dilates, prop, nthreads);
    if (s != Status::SUCCESS) {
        return Status::UNSUPPORTED;
    }
    IntType bcdIcChunks = Math::CeilDiv(bcp.nbIc, bcp.nbIcBlocking);

    bcd.brgs = std::make_shared<BrgemmDescContainerT>(BRGEMM_1X1_CONV_NUM_16);
    const float alpha = 1.0f;
    const float beta = 1.0f;
    IntType iInitBegin = bcdIcChunks == 1 ? 1 : 0;
    IntType iInitEnd = BRGEMM_1X1_CONV_NUM_2;
    for (IntType iM = 0; iM < BRGEMM_1X1_CONV_NUM_2; iM++) {
        for (IntType iN = 0; iN < BRGEMM_1X1_CONV_NUM_2; iN++) {
            for (IntType iK = 0; iK < BRGEMM_1X1_CONV_NUM_2; iK++) {
                for (IntType iInit = iInitBegin; iInit < iInitEnd; iInit++) {
                    IntType vM = iM ? bcp.mTail : bcp.m;
                    IntType vN = iN ? bcp.nTail : bcp.n;
                    IntType vK = iK ? bcp.kTail : bcp.k;
                    if (vM == 0 || vN == 0 || vK == 0) {
                        continue;
                    }
                    BrgemmStridesT brgStrides;
                    brgStrides.strideA = bcp.brgStrideA;
                    brgStrides.strideB = bcp.brgStrideB;
                    const BrgemmStridesT *stridesPtr = bcp.brgType == BrgemmBatchKindT::STRIDE
                        ? &brgStrides
                        : nullptr;
                    BrgemmT brg;
                    float vBeta = iInit ? 0.0f : beta;
                    CHECK(BrgemmDescInit(&brg, isa, bcp.brgType, src.GetType(), weights.GetType(), false, false,
                        BrgemmLayoutT::ROW_MAJOR, alpha, vBeta, bcp.ldA, bcp.ldB, bcp.ldC, vM, vN, vK, stridesPtr));
                    brg.withSum = false;
                    brg.withWeightsScaleAdjust = false;
                    IntType ldD = bcp.ocWithoutPadding;
                    CHECK(BrgemmDescSetPostops(&brg, nullptr, const_cast<TensorInfo *>(&dst), ldD, bcp.biaDt));
                    CHECK(BrgemmDescFinalize(&brg));
                    bcd.brgs->Insert(GetBrgIdx(iInit, iM, iN, iK), brg);
                }
            }
        }
    }

    return Status::SUCCESS;
}

Status KdnnBrgemm1x1ConvCreateKernel(KdnnBrgconvDataT &bcd)
{
    if (!bcd.brgs) {
        return Status::UNSUPPORTED;
    }
    BrgemmKernelContainerT &brgKernels = bcd.brgKernels;
    IntType bcdIcChunks = Math::CeilDiv(bcd.bcp.nbIc, bcd.bcp.nbIcBlocking);

    IntType iInitBegin = bcdIcChunks == 1 ? 1 : 0;
    IntType iInitEnd = BRGEMM_1X1_CONV_NUM_2;
    for (IntType iM = 0; iM < BRGEMM_1X1_CONV_NUM_2; iM++) {
        for (IntType iN = 0; iN < BRGEMM_1X1_CONV_NUM_2; iN++) {
            for (IntType iK = 0; iK < BRGEMM_1X1_CONV_NUM_2; iK++) {
                for (IntType iInit = iInitBegin; iInit < iInitEnd; iInit++) {
                    IntType brgIdx = GetBrgIdx(iInit, iM, iN, iK);
                    const BrgemmT *brg = (*(bcd.brgs))[brgIdx];
                    if (brg && brg->bcastDim > 0 && brg->loadDim > 0 && brg->reduceDim > 0 && !brgKernels[brgIdx]) {
                        CHECK(brgKernels.Insert(brgIdx, brg));
                    }
                }
            }
        }
    }

    return Status::SUCCESS;
}

static void KdnnBrgemm1x1ConvExecuteKernel(const KdnnBrgconvDataT &bcd, const float *srcIn, const float *weiIn,
    float *dstIn, const float *biasIn, IntType ithr, BrgemmBatchElementT *brgBatchIn, uint8_t *cBuffer,
    const uint8_t *inpBuffer, IntType g, IntType n, IntType ocb, IntType od, IntType oh, IntType ow, IntType icc,
    IntType *lastBrgIdx, const float *oscales, int32_t srcZpVals, int32_t *srcZpComp, int32_t *dstZpVals,
    int32_t *s8s8Compensation, const float *dstScales)
{
    Service::Unused(ithr);
    Service::Unused(lastBrgIdx);
    const KdnnBrgemmConvConfT &bcp = bcd.bcp;
    const IntType bcdOh = NdimsPick(bcp.ndims, bcp.oh, bcp.oh, 1);
    const IntType bcdOw = bcp.ow;
    const IntType bcdSd = NdimsPick(bcp.ndims, bcp.strideD, 1, 1);
    const IntType bcdSh = NdimsPick(bcp.ndims, bcp.strideH, bcp.strideH, 1);
    const IntType bcdSw = bcp.strideW;
    const SizeType bcdBiaDsz = bcp.biaDsz;
    const IntType bcdWeiIcStride = bcp.weiPlain ? bcp.ocWithoutPadding : bcp.ocBlock;
    const IntType bcdWeiOcbStride = bcp.weiPlain
        ? bcp.ocBlock
        : RndUp(bcp.ic, static_cast<IntType>(DataTypeVnniGranularity(bcp.srcDt))) * bcp.ocBlock;
    const IntType bcdWeiGStride = bcp.weiPlain ? bcp.oc : bcp.nbOc * bcdWeiOcbStride;
    const IntType bcdSrcWSz = bcp.iw * bcp.ngroups * bcp.icWithoutPadding;
    const IntType bcdSrcHSz = NdimsPick(bcp.ndims, bcp.ih, bcp.ih, 1) * bcdSrcWSz;
    const IntType bcdSrcDSz = NdimsPick(bcp.ndims, bcp.id, 1, 1) * bcdSrcHSz;
    const IntType bcdDstWSz = bcp.ow * bcp.ocWithoutPadding;
    const IntType bcdDstHSz = NdimsPick(bcp.ndims, bcp.oh, bcp.oh, 1) * bcdDstWSz;
    const IntType bcdDstDSz = NdimsPick(bcp.ndims, bcp.od, 1, 1) * bcdDstHSz;
    const bool bcdNeedPostwork = bcp.withBias;
    const IntType bcdIcChunks = Math::CeilDiv(bcp.nbIc, bcp.nbIcBlocking);

    const SizeType srcDtSize = Element::Type(bcp.srcDt).GetSize();
    const SizeType weiDtSize = Element::Type(bcp.weiDt).GetSize();
    const SizeType dstDtSize = Element::Type(bcp.dstDt).GetSize();
    const uint8_t *src = (decltype(src))(srcIn);
    const uint8_t *wei = (decltype(wei))(weiIn);
    uint8_t *dst = (decltype(dst))(dstIn);
    const uint8_t *bias = (decltype(bias))(biasIn);
    const void *postOpsBinaryRhsArgVec = nullptr;
    const IntType ndims = bcp.ndims;
    const IntType id = NdimsPick(ndims, od * bcdSd, 0, 0);
    const IntType ih = NdimsPick(ndims, oh * bcdSh, oh * bcdSh, 0);
    const IntType iw = ow * bcdSw;
    const IntType oc = ocb * bcp.ocBlock;
    const IntType gOc = g * bcp.oc + oc;
    const IntType icb = icc * bcp.nbIcBlocking;
    const IntType ic = icb * bcp.icBlock;
    const IntType gIc = g * bcp.ic + ic;
    const bool kernelInit = icc == 0;
    const IntType os = (od * bcdOh + oh) * bcdOw + ow;
    const bool isOsTail = bcp.isOsBlocking ? (bcp.os - os < bcp.osBlock) : (bcdOw - ow < bcp.owBlock);
    const bool isOcTail = bcp.oc - oc < bcp.ocBlock;
    const bool isIcTail = icc == bcdIcChunks - 1 && ((bcp.ic - ic) % bcp.icBlock != 0);
    const IntType srcOffset = n * bcdSrcDSz + id * bcdSrcHSz + ih * bcdSrcWSz
        + iw * bcp.ngroups * bcp.icWithoutPadding + gIc;
    const uint8_t *srcBase = bcp.isRtus ? inpBuffer : src + srcOffset * srcDtSize;
    const IntType weiOffset = g * bcdWeiGStride + ocb * bcdWeiOcbStride;
    const uint8_t *weiBase = wei + weiOffset * weiDtSize;
    const void *ptrD = dst + (n * bcdDstDSz + od * bcdDstHSz + oh * bcdDstWSz + ow * bcp.ocWithoutPadding + gOc)
        * dstDtSize;
    void *ptrC = bcp.useBuffer ? static_cast<void *>(cBuffer) : const_cast<void *>(ptrD);
    const uint8_t *biasW = bias ? bias + gOc * bcdBiaDsz : nullptr;
    const IntType nbIcB = std::min(bcp.nbIcBlocking, bcp.nbIc - icb) - (isIcTail ? 1 : 0);
    const IntType compOffset = (g * bcp.nbOc + ocb) * bcp.ocBlock;
    int32_t *srcZpCompPtr = (bcp.srcZeroPoint && icc == bcdIcChunks - 1) ? &srcZpComp[compOffset] : nullptr;
    int32_t *s8s8CompPtr = (bcp.s8s8CompensationRequired && icc == bcdIcChunks - 1)
        ? &s8s8Compensation[compOffset]
        : nullptr;
    const auto callBrgemm = [ic, srcBase, srcDtSize, weiBase, bcdWeiIcStride, weiDtSize, s8s8CompPtr,
        dst, biasW, gOc, srcZpCompPtr, dstZpVals, srcZpVals, dstScales, ptrC, ptrD, oscales, postOpsBinaryRhsArgVec,
        bcp, bcd, brgBatchIn](IntType brgIdx, IntType icBlockS, IntType nIcBlocks, bool doPostops) {
        for (IntType k = 0; k < nIcBlocks; k++) {
            const IntType icOff = (icBlockS + k) * bcp.icBlock;
            const IntType srcIc = icOff;
            const IntType weiIc = ic + icOff;
            const uint8_t *ptrA = srcBase + srcIc * srcDtSize;
            const uint8_t *ptrB = weiBase + weiIc * bcdWeiIcStride * weiDtSize;
            brgBatchIn[k].ptr.a = ptrA;
            brgBatchIn[k].ptr.b = ptrB;
            brgBatchIn[k].vvpad.top = 0;
            brgBatchIn[k].vvpad.bottom = 0;
        }
        const BrgemmBaseKernelT *brgKer = bcd.brgKernels[brgIdx];
        void *scratch = static_cast<void *>(s8s8CompPtr);
        if (doPostops) {
            const void *voidDst = (decltype(voidDst))(dst);
            const BrgemmPostOpsDataT postOpsData {
                static_cast<const void *>(biasW),
                &oscales[bcp.isOcScale * gOc],
                postOpsBinaryRhsArgVec,
                static_cast<SizeType>(gOc), 0, static_cast<const char *>(voidDst), 0,
                static_cast<void *>(srcZpCompPtr), nullptr,
                static_cast<void *>(dstZpVals), false, srcZpVals, false,
                false, dstScales
            };
            BrgemmKernelExecutePostops(brgKer, nIcBlocks, nullptr, nullptr, brgBatchIn,
                ptrC, const_cast<void *>(ptrD), postOpsData, scratch);
        } else {
            BrgemmKernelExecute(brgKer, nIcBlocks, nullptr, nullptr, brgBatchIn, ptrC, scratch);
        }
    };
    const bool doPostWork = (bcdNeedPostwork || bcp.useBuffer) && icc == bcdIcChunks - 1;
    if (nbIcB > 0) {
        const IntType brgIdx = GetBrgIdx(static_cast<IntType>(kernelInit), static_cast<IntType>(isOsTail),
            static_cast<IntType>(isOcTail), static_cast<IntType>(false));
        callBrgemm(brgIdx, 0, nbIcB, doPostWork && !isIcTail);
    }
    if (isIcTail) {
        const bool useInitKer = kernelInit && nbIcB == 0;
        const IntType brgIdx = GetBrgIdx(static_cast<IntType>(useInitKer), static_cast<IntType>(isOsTail),
            static_cast<IntType>(isOcTail), static_cast<IntType>(true));
        callBrgemm(brgIdx, nbIcB, 1, doPostWork);
    }
}

void KdnnBrgemm1x1ConvExecute(const KdnnBrgconvDataT &bcd, const float *src, const float *wei, float *dst,
    const float *bias)
{
    const KdnnBrgemmConvConfT &bcp = bcd.bcp;
    const IntType bcdOd = NdimsPick(bcp.ndims, bcp.od, 1, 1);
    const IntType bcdOh = NdimsPick(bcp.ndims, bcp.oh, bcp.oh, 1);
    const IntType bcdOw = bcp.ow;
    const SizeType bcdSrcDsz = bcp.srcDsz;
    const SizeType bcdAccDsz = bcp.accDsz;
    const IntType bcdIcChunks = Math::CeilDiv(bcp.nbIc, bcp.nbIcBlocking);
    bool isAssertSuccess = !bcp.isRtus && !bcp.useBuffer && !bcp.s8s8CompensationRequired && !bcp.withScales
        && !bcp.srcZeroPoint && !bcp.dstZeroPoint;
    if (!isAssertSuccess) {
        throw Service::Unsupported("");
    }

    const float *srcScales = nullptr;
    const float *weiScales = nullptr;
    const float *dstScales = nullptr;
    const float *oscales = nullptr;
    int32_t srcZeroPoint = 0;
    int32_t dstZeroPoint = 0;
    const IntType extraDataOffset = 0;
    uint8_t *w = (decltype(w))(wei);
    int32_t *s8s8Compensation = nullptr;
    int32_t *zpCompensation = nullptr;
    int32_t *dstZpVals = nullptr;
    Service::Unused(srcScales);
    Service::Unused(weiScales);
    Service::Unused(dstZeroPoint);
    Service::Unused(extraDataOffset);
    Service::Unused(w);
    // no need for external brgBatch ptr
    const IntType maxBrgBatchSize = bcp.nthr * bcp.adjustedBatchSize * std::max(std::min(bcp.nbIcBlocking, bcp.nbIc),
        std::min(bcp.nbIcBlocking, bcp.nbIc - bcdIcChunks * bcp.nbIcBlocking));
    std::vector<BrgemmBatchElementT> brgBatchVector(maxBrgBatchSize);
    BrgemmBatchElementT *brgBatchGlobal = bcp.brgType != BrgemmBatchKindT::STRIDE ? brgBatchVector.data() : nullptr;
    uint8_t *cBufferGlobal = nullptr;
    uint8_t *inpBufferBase = nullptr;
    uint8_t *inpBufferMaskBase = nullptr;
    if (bcp.isOsBlocking) {
        const IntType osChunks = Math::CeilDiv(bcp.nbOs, bcp.nbOsBlocking);
        const IntType workAmount = bcp.mb * bcp.ngroups * bcp.nbOc * osChunks;
        Parallel(bcp.nthr, [&](const IntType ithr, const IntType nthr) {
            if (ithr >= workAmount) {
                return;
            }
            BrgemmBatchElementT *brgBatch = brgBatchGlobal + static_cast<SizeType>(ithr) * bcp.adjustedBatchSize;
            uint8_t *cBuffer = bcp.useBuffer ? cBufferGlobal + ithr * bcp.m * bcp.ldC * bcdAccDsz : nullptr;
            uint8_t *inpBuffer = bcp.isRtus ? inpBufferBase + ithr * bcp.inpBufferSize * bcdSrcDsz : nullptr;
            uint8_t *inpBufferMask = bcp.isRtus ? inpBufferMaskBase + ithr * bcp.inpBufferMaskSize : nullptr;
            IntType lastN = -1;
            IntType lastG = -1;
            IntType lastBrgIdx = -1;
            IntType start {0};
            IntType end {0};
            Balance211(workAmount, nthr, ithr, start, end);
            IntType n {0};
            IntType g {0};
            IntType ocb {0};
            IntType oss {0};
            if (bcp.loopOrder == KdnnBrgemmConvLoopOrderT::NDHWGC) {
                NdIteratorInit(start, n, bcp.mb, oss, osChunks, g, bcp.ngroups, ocb, bcp.nbOc);
            } else if (bcp.loopOrder == KdnnBrgemmConvLoopOrderT::NGCDHW) {
                NdIteratorInit(start, n, bcp.mb, g, bcp.ngroups, ocb, bcp.nbOc, oss, osChunks);
            } else {
                throw Service::LogicError("");
            }
            for (IntType work = start; work < end; work++) {
                if (bcp.isRtus && (lastN != n || lastG != g)) {
                    for (IntType i = 0; i < bcp.inpBufferMaskSize; i++) {
                        inpBufferMask[i] = 0;
                    }
                }
                const IntType osbStart = oss * bcp.nbOsBlocking;
                const IntType osbRange = std::min(bcp.nbOs - osbStart, bcp.nbOsBlocking);
                for (IntType osb = 0; osb < osbRange; osb++) {
                    const IntType os = (osbStart + osb) * bcp.osBlock;
                    const IntType od = os / (bcdOh * bcdOw);
                    const IntType oh = (os % (bcdOh * bcdOw)) / bcdOw;
                    const IntType ow = os % bcdOw;
                    uint8_t *inpBufferSp = bcp.isRtus ? inpBuffer + os * bcp.ldA * bcdSrcDsz : nullptr;
                    for (IntType icc = 0; icc < bcdIcChunks; icc++) {
                        if (bcp.isRtus) {}
                        KdnnBrgemm1x1ConvExecuteKernel(bcd, src, wei, dst, bias, ithr, brgBatch, cBuffer,
                            inpBufferSp, g, n, ocb, od, oh, ow, icc, &lastBrgIdx, oscales, srcZeroPoint,
                            zpCompensation, dstZpVals, s8s8Compensation, dstScales);
                    }
                }
                lastN = n;
                lastG = g;
                if (bcp.loopOrder == KdnnBrgemmConvLoopOrderT::NDHWGC) {
                    NdIteratorStep(n, bcp.mb, oss, osChunks, g, bcp.ngroups, ocb, bcp.nbOc);
                } else if (bcp.loopOrder == KdnnBrgemmConvLoopOrderT::NGCDHW) {
                    NdIteratorStep(n, bcp.mb, g, bcp.ngroups, ocb, bcp.nbOc, oss, osChunks);
                } else {
                    throw Service::LogicError("");
                }
            }
        });
    } else {
        const IntType workAmount = bcp.mb * bcp.ngroups * bcp.nbOc * bcdOd * bcdOh * bcp.nbOw;
        Parallel(bcp.nthr, [&](const IntType ithr, const IntType nthr) {
            if (ithr >= workAmount) {
                return;
            }
            BrgemmBatchElementT *brgBatch = brgBatchGlobal + static_cast<SizeType>(ithr) * bcp.adjustedBatchSize;
            uint8_t *cBuffer = bcp.useBuffer ? cBufferGlobal + ithr * bcp.m * bcp.ldC * bcdAccDsz : nullptr;
            IntType lastBrgIdx = -1;
            IntType start {0};
            IntType end {0};
            Balance211(workAmount, nthr, ithr, start, end);
            IntType n {0};
            IntType g {0};
            IntType ocb {0};
            IntType od {0};
            IntType oh {0};
            IntType owb {0};
            if (bcp.loopOrder == KdnnBrgemmConvLoopOrderT::NDHWGC) {
                NdIteratorInit(start, n, bcp.mb, od, bcdOd, oh, bcdOh, owb, bcp.nbOw, g, bcp.ngroups, ocb, bcp.nbOc);
            } else if (bcp.loopOrder == KdnnBrgemmConvLoopOrderT::NGCDHW) {
                NdIteratorInit(start, n, bcp.mb, g, bcp.ngroups, ocb, bcp.nbOc, od, bcdOd, oh, bcdOh, owb, bcp.nbOw);
            } else {
                throw Service::LogicError("");
            }
            for (IntType work = start; work < end; work++) {
                for (IntType icc = 0; icc < bcdIcChunks; icc++) {
                    const IntType ow = owb * bcp.owBlock;
                    KdnnBrgemm1x1ConvExecuteKernel(bcd, src, wei, dst, bias, ithr, brgBatch, cBuffer, nullptr,
                        g, n, ocb, od, oh, ow, icc, &lastBrgIdx, oscales, srcZeroPoint, zpCompensation,
                        dstZpVals, s8s8Compensation, dstScales);
                }
                if (bcp.loopOrder == KdnnBrgemmConvLoopOrderT::NDHWGC) {
                    NdIteratorStep(n, bcp.mb, od, bcdOd, oh, bcdOh, owb, bcp.nbOw, g, bcp.ngroups, ocb, bcp.nbOc);
                } else if (bcp.loopOrder == KdnnBrgemmConvLoopOrderT::NGCDHW) {
                    NdIteratorStep(n, bcp.mb, g, bcp.ngroups, ocb, bcp.nbOc, od, bcdOd, oh, bcdOh, owb, bcp.nbOw);
                } else {
                    throw Service::LogicError("");
                }
            }
        });
    }
}

// kdnn brgconv interface -------------------------------------------------------------------------
Status KdnnBrgconvInitConf(KdnnBrgconvDataT &bcd, ARMCG::CPUISAT isa, const TensorInfo &src,
    const TensorInfo &weights, const TensorInfo &dst, const TensorInfo &bias, Shape strides,
    Shape dilates, Shape paddingL, Propagation prop, IntType nthreads)
{
    // validate input
    bool isValidInput = true
        && ARMCG::MayIUse(isa)
        && (Element::TypeT(src.GetType()) == Element::TypeT::F32
            && Element::TypeT(weights.GetType()) == Element::TypeT::F32
            && Element::TypeT(dst.GetType()) == Element::TypeT::F32)
        && (prop == Propagation::FORWARD_TRAINING || prop == Propagation::FORWARD_INFERENCE)
        && nthreads > 0
        // prevent sve256 impl to use sve512 kernel
        && ARMCG::IsaMaxVlen(isa) == static_cast<IntType>(ARMCG::GetVCLength());
    if (!isValidInput) {
        return Status::UNSUPPORTED;
    }

    // skip case
    bool withGroups = src.GetNumDims() == weights.GetNumDims() - 1;
    IntType ngroups = withGroups ? weights.GetDims()[0] : 1;
    IntType ndims = src.GetNumDims();
    IntType ic = src.GetDims()[1] / ngroups;
    IntType iw = src.GetDims()[ndims - 1];
    IntType oc = dst.GetDims()[1] / ngroups;
    IntType ow = dst.GetDims()[ndims - 1];
    bool skipCase = false
        // some case can be better than direct sme/sve
        || (isa == ARMCG::CPUISAT::VC_512
            && (!(ic <= BRGEMM_1X1_CONV_NUM_256 && iw <= BRGEMM_1X1_CONV_NUM_56 && oc <= BRGEMM_1X1_CONV_NUM_512
                && ow <= BRGEMM_1X1_CONV_NUM_28) && !(ic >= BRGEMM_1X1_CONV_NUM_384 && iw >= BRGEMM_1X1_CONV_NUM_300
                && oc <= BRGEMM_1X1_CONV_NUM_1024)));
    if (skipCase) {
        return Status::UNSUPPORTED;
    }
    // [unimplemented] dw conv
    if (KdnnBrgemm1x1ConvInitConf(bcd, isa, src, weights, dst, bias, strides, dilates, paddingL,
        prop, nthreads) == Status::SUCCESS) {
        bcd.useBrgconv = true;
        bcd.brgconvImpl = KdnnBrgconvImplT::BRGCONV_1X1;
        return Status::SUCCESS;
    }
    // [unimplemented] common conv

    return Status::UNSUPPORTED;
}

Status KdnnBrgconvCreateKernel(KdnnBrgconvDataT &bcd)
{
    switch (bcd.brgconvImpl) {
        // [unimplemented] dw conv
        case KdnnBrgconvImplT::BRGCONV_1X1:
            return KdnnBrgemm1x1ConvCreateKernel(bcd);
        // [unimplemented] common conv
        default: break;
    }

    return Status::UNSUPPORTED;
}

void KdnnBrgconvExecute(const KdnnBrgconvDataT &bcd, const float *src, const float *wei,
    float *dst, const float *bias)
{
    switch (bcd.brgconvImpl) {
        // [unimplemented] dw conv
        case KdnnBrgconvImplT::BRGCONV_1X1:
            KdnnBrgemm1x1ConvExecute(bcd, src, wei, dst, bias);
            break;
        // [unimplemented] common conv
        default: break;
    }
}

} // namespace KuDNN
