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

#include "kudnn_brgemm_conv_utils.hpp"

#include <algorithm>
#include <vector>

#include "kudnn_brgemm.hpp"
#include "service/kudnn_err_codes.hpp"
#include "tools/cg_cpu_isa_traits.hpp"
#include "types/kudnn_data_type.hpp"
#include "types/kudnn_propagation_type.hpp"
#include "types/kudnn_shape.hpp"
#include "types/kudnn_tensor_info.hpp"

#include "kudnn_brgemm_primitive_conf.hpp"
#include "kudnn_conv_const.hpp"

namespace KuDNN {
namespace {
inline bool Implication(bool cause, bool effect)
{
    return (!(cause) || !!(effect));
}

template <typename T>
constexpr const T &Saturate(const T &low, const T &upper, const T &a)
{
    return std::max(low, std::min(upper, a));
}

inline bool AllowPerfHeuristics(const KdnnBrgemmConvConfT &bcp)
{
    // disable performance heuristics for plain weights as there are no other
    // optimized implementations.
    if (bcp.weiPlain) {
        return false;
    }

    return true;
}

inline bool IsGroupsOk(KdnnBrgemmConvConfT &bcp)
{
    // enable grouped convs for the shapes not supported in direct convs
    // direct approach only supports int8 bf16 grouped conv
    // when channels per groups is at least multiple of 4
    // and bf16 grouped conv with layout nxc on bf16 impl
    return bcp.ngroups > 1 && Implication(
        Service::OneOf(bcp.srcDt, Element::TypeT::U8, Element::TypeT::S8,
        Element::TypeT::BF16), bcp.ic % NUM_4 == 0 && bcp.oc % NUM_4 == 0);
}

} // namespace

struct BrgBlockingT : public KdnnBrgemmConvConfT {
    struct ArrayInLoopT {
        void Set(IntType iterS, float rpt, float ovlp = 1.0f)
        {
            itersize = iterS;
            repeatn = rpt;
            overlap = ovlp;
        }

        IntType itersize;
        float repeatn;
        float overlap;
    };
    struct LoopT {
        ArrayInLoopT src;
        ArrayInLoopT wei;
        ArrayInLoopT dst;
    };

    // this is a broken form of initialization for a base class.
    // either set default values in a base class, or provide a proper
    // default ctor, or take a KdnnBrgemmConvConfT object to initialize
    // a base class object.
    BrgBlockingT()
    {
        KdnnBrgemmConvConfT *base = static_cast<KdnnBrgemmConvConfT *>(this);
        *base = KdnnBrgemmConvConfT();
        Init();
    }
    BrgBlockingT(const KdnnBrgemmConvConfT &bcp) : KdnnBrgemmConvConfT(bcp)
    {
        Init();
    }

    void Init()
    {
        ur = 0;
        urBlock = 0;
        urBlockTail = 0;
        eff = 0.f;
        nbKd = 0;
        nbKh = 0;
        nbKw = 0;
        sp = 0;
        spBlock = 0;
        nbSp = 0;
        eff = 0;
        maxRegs = ARMCG::IsaNumVregs(isa);
        bcastSimd = accSimdW;
    }

    IntType ur;
    IntType urBlock;
    IntType urBlockTail;
    IntType nbKd;
    IntType nbKh;
    IntType nbKw;
    IntType maxRegs;
    IntType bcastSimd;
    float eff;
    static unsigned cL1;
    static unsigned cL2;
    static unsigned cL3;
    // These are rough estimates of the latency (relative) of access to various
    // cache levels. This is enough for an estimation of data access cost.
    // improve memory access estimates
    static constexpr float L1K = 1.0f;
    static constexpr float L2K = 3.0f;
    static constexpr float L3K = 15.0f;
    // at the moment, we are primarily evaluating the fit of the data into the L1/L2.
    // need to take into account the difference between the L3 and memory.
    static constexpr float memK = 15.0f;
    static constexpr IntType benchIterations = 1;
    IntType sp;
    IntType spBlock;
    IntType nbSp;
    static thread_local IntType lastIcBlockSize;
    LoopT loop[BRGEMM_CONV_UTILS_NUM_32];

    void GetFromBcp(const KdnnBrgemmConvConfT &bcp)
    {
        *this = bcp;
    }
    void SaveToBcp(KdnnBrgemmConvConfT &bcp) const
    {
        bcp = *this;
    }
    Status EstimateBrgemmUr();
    Status GetBrgemmUr(const void *attr, const TensorInfo *dst);
    float IoK(IntType src, IntType wei, IntType dst, float n, float pk, bool isBroadcast, bool isShared) const;
    float IoK(const LoopT loopIn, const ArrayInLoopT arr, float pk, bool isBroadcast, bool isShared) const;
    void SelectIcBlock();
    void UpdateBlocks();
    // 1x1 conv
    bool FastCheckOcBlock1x1() const;
    float EstEff1x1();
    void CalcBlocks1x1();
    // common conv
    bool FastCheckOcBlock() const;
    float EstEff();
    void IterateKerBlock(BrgBlockingT &bestBrgb, IntType kdBlock, IntType khBlock, bool maybeUseBuffer,
        IntType maxOwBlockThr);
    Status CalcBlocks();
    // utils
    static IntType GetInpSize(IntType maxSrcSize, IntType dstSize, IntType k, IntType stride, IntType dilate)
    {
        auto adjStr = std::min(k, stride);
        const auto res = std::min(maxSrcSize,
            CalculateEndPadding(0, dstSize, 0, adjStr, CalculateExtendedFilterSize(k, dilate)));

        return res;
    }
    static float SqueezeVal(float effIn, float koeff)
    {
        if (koeff <= 0) {
            return 1;
        }
        if (koeff == 1) {
            return effIn;
        }

        const auto k = 1.0f / koeff;
        return (k > 1.0f) ? (k - 1 + effIn) / k : effIn * koeff;
    }
    static IntType EstimateUr(IntType ocBlock)
    {
        switch (ocBlock) {
            case NUM_64: return NUM_6;
            case NUM_48: return NUM_9;
            case NUM_32: return NUM_14;
            default: break;
        }

        return NUM_28;
    }
    IntType InpW(IntType outW, IntType kerW) const
    {
        return GetInpSize(iw, outW, kerW, strideW, dilateW);
    }
    IntType RndSimd(IntType val) const
    {
        return RndUp(val, simdW);
    }
    IntType RndInpSimd(IntType outW, IntType kerW, IntType vic) const
    {
        return strideW == 1 && vic >= ic
            ? RndUp(InpW(outW, kerW) * vic, simdW)
            : InpW(outW, kerW) * RndUp(vic, simdW);
    }
};

unsigned BrgBlockingT::cL1;
unsigned BrgBlockingT::cL2;
unsigned BrgBlockingT::cL3;
thread_local IntType BrgBlockingT::lastIcBlockSize;

Status BrgBlockingT::EstimateBrgemmUr()
{
    // simple simulation of BrgemmDescInit
    if (spBlock <= 0) {
        return Status::BAD_ARGUMENTS;
    }
    ldA = isRtus
        ? (icBlock)
        : (khSets > 1 ? khSets : 1) * (kwSets > 1 ? kwSets : strideW)
            * (execType == KdnnBrgemmConvExecTypeT::TRANS ? icBlock : ngroups * icWithoutPadding);
    ldB = weiPlain ? ocWithoutPadding : ocBlock;
    ldC = useBuffer ? ocBlock : ocWithoutPadding;
    const auto paddedIc = lastIcBlockSize * (isIcPadded ? accSimdW : 1);
    icp = RndUp(ic, paddedIc);
    m = brgM = sp >= spBlock ? spBlock : 0;
    mTail = brgMTail = sp % spBlock;
    if (isOsBlocking) {
        if (!is1x1) {
            mTail = (oh * ow) % spBlock;
        }
        oskip = ((extKw - 1) / strideW) * strideH + (strideH - 1) * ow;
        brgM = m + oskip * (Math::CeilDiv(m, ow) - 1);
        brgMTail = mTail + oskip * Math::CeilDiv(mTail, ow);
    }
    n = oc >= ocBlock ? ocBlock : 0;
    nTail = oc % ocBlock;
    k = khSets * kwSets * (ic >= icBlock ? icBlock : 0);
    kTail = khSets * kwSets * (execType == KdnnBrgemmConvExecTypeT::TRANS && (!isBF32)
        ? icBlock
        : RndUp(ic % icBlock, lastIcBlockSize));
    const auto vK = k > 0 ? k : kTail;
    const auto vM = m > 0 ? m : mTail;
    const auto vN = n > 0 ? n : nTail;
    const float alpha = 1.0;
    const float beta = 0.0;
    BrgemmT brg;
    CHECK(InitBrgemmConf(&brg, isa, BrgemmBatchKindT::ADDRESS, srcDt, weiDt, BrgemmLayoutT::ROW_MAJOR, alpha, beta,
        ldA, ldB, ldC, vM, vN, vK, nullptr, isBF32));
    CHECK(BrgemmBlocking(&brg));
    ur = brg.bdBlock;
    urBlock = brg.bdBlock;
    urBlockTail = 0;

    return Status::SUCCESS;
}

Status BrgBlockingT::GetBrgemmUr(const void *attr, const TensorInfo *dst)
{
    // detailed simulation of brgemm convolution init
    if (spBlock <= 0 || icBlock <= 0 || ocBlock <= 0) {
        return Status::BAD_ARGUMENTS;
    }
    CHECK(EstimateBrgemmUr());
    ldD = ocWithoutPadding;
    const float alpha = 1.0;
    const float beta = 1.0;
    const float betaInit = 0.0;
    for (IntType i = 0; i < m; i++) {
        auto vM = i + 1;
        // init only needed brgemm descriptors
        if ((execType == KdnnBrgemmConvExecTypeT::TRANS || execType == KdnnBrgemmConvExecTypeT::VPAD || is1x1)
            && vM != m && vM != mTail) {
            continue;
        }
        for (IntType iInit = 0; iInit < BRGEMM_CONV_UTILS_NUM_2; iInit++) {
            for (IntType iN = 0; iN < BRGEMM_CONV_UTILS_NUM_2; iN++) {
                for (IntType iK = 0; iK < BRGEMM_CONV_UTILS_NUM_2; iK++) {
                    auto vN = (iN) ? nTail : n;
                    auto vK = (iK) ? kTail : k;
                    auto vbeta = (iInit) ? betaInit : beta;
                    if (vN == 0 || vK == 0) {
                        continue;
                    }
                    BrgemmStridesT brgStrides;
                    brgStrides.strideA = ngroups * icWithoutPadding * (dilateW + 1) * srcDsz;
                    // weights are padded by ocBlock and lastIcBlock
                    brgStrides.strideB = RndUp(ic, lastIcBlockSize) * RndUp(oc, ocBlock) * weiDsz;
                    BrgemmT brg;
                    const auto stridesPtr = (brgType == BrgemmBatchKindT::STRIDE) ? &brgStrides : nullptr;
                    CHECK(InitBrgemmConf(&brg, isa, brgType, srcDt, weiDt, BrgemmLayoutT::ROW_MAJOR, alpha, vbeta,
                        ldA, ldB, ldC, vM, vN, vK, stridesPtr, isBF32));
                    CHECK(BrgemmBlocking(&brg));
                    BrgemmAttrT brgattr;
                    brgattr.maxBs = maxBatch;
                    maxVpad = execType == KdnnBrgemmConvExecTypeT::VPAD ? std::max(lPad, rPad) : 0;
                    brgattr.maxTopVpad = maxVpad;
                    brgattr.maxBottomVpad = maxVpad;
                    brgattr.fpmathMode = FpmathModeT::STRICT;
                    CHECK(BrgemmDescSetAttr(&brg, brgattr));
                    brg.withSum = withSum;
                    TensorInfo *noConstInfo = const_cast<TensorInfo *>(dst);
                    CHECK(BrgemmDescSetPostops(&brg, attr, noConstInfo, ldD, biaDt));
                }
            }
        }
    }

    return Status::SUCCESS;
}

float BrgBlockingT::IoK(IntType src, IntType wei, IntType dst, float nIn, float pk, bool isBroadcast,
    bool isShared) const
{
    if (nIn < 1) {
        return 0;
    }
    if (nIn == 1) {
        return pk;
    }

    const auto amount = src * srcDsz + wei * weiDsz + dst * dstDsz + (useBuffer ? dst * accDsz : 0);
    const auto amountL1 = isBroadcast ? src * srcDsz : amount;
    const auto kTmp = isBroadcast
        ? (amountL1 < cL1 ? L1K : (amount < cL2 ? L2K : (isShared ? L3K : memK)))
        : (amount < cL2 ? L2K : (isShared ? L3K : memK));
    const auto cost = pk + kTmp * (nIn - 1);

    return cost / nIn;
}

float BrgBlockingT::IoK(const LoopT loopIn, const ArrayInLoopT arr, float pk, bool isBroadcast, bool isShared) const
{
    return IoK(loopIn.src.itersize, loopIn.wei.itersize, loopIn.dst.itersize, arr.repeatn * arr.overlap, pk,
        isBroadcast, isShared);
}

void BrgBlockingT::SelectIcBlock()
{
    IntType nbSimd = Math::CeilDiv(ic, simdW);
    IntType maxSimdBlocks = std::min(BRGEMM_CONV_UTILS_NUM_5 * simdW, nbSimd);
    const auto nbIcbEffThreshold = 0.5f;
    const auto paddedIc = lastIcBlockSize * (isIcPadded ? accSimdW : 1);
    const auto estUr = spBlock > 0 ? std::min(spBlock, EstimateUr(ocBlock)) : EstimateUr(ocBlock);
    const auto inpUr = isOsBlocking ? estUr : InpW(estUr, kwBlock);
    // try to fit src into L1
    if (kwBlock > 1) {
        const auto inpPerIc = static_cast<SizeType>(inpUr) * srcDsz;
        maxSimdBlocks = Saturate(static_cast<IntType>(1), maxSimdBlocks,
            static_cast<IntType>(cL1 / (inpPerIc * simdW)));
    }
    // try to fit all batch for ur into L2
    const bool ocIsPow2 = oc > 0 && (oc & (oc - 1)) == 0;
    const bool adjust = weiPlain && ocIsPow2 && kdBlock == 1 && khBlock == 1 && kwBlock == 1;
    const IntType adjOcBlock = adjust ? oc : ocBlock;
    const auto weiPerIc = static_cast<SizeType>(kdBlock) * khBlock * kwBlock * adjOcBlock * weiDsz;
    const auto inpPerIc = static_cast<SizeType>(kdBlock) * khBlock * inpUr * srcDsz;
    const auto outSize = static_cast<SizeType>(ur) * ocBlock * dstDsz;
    maxSimdBlocks = Saturate(static_cast<IntType>(1), maxSimdBlocks,
        static_cast<IntType>((cL2 - outSize) / ((weiPerIc + inpPerIc) * simdW)));
    auto simdBlocks = 1;
    for (IntType nbIcb = std::min(maxSimdBlocks, nbSimd); nbIcb >= 1; nbIcb--) {
        auto nbIcbEff = static_cast<float>(nbSimd) / RndUp(nbSimd, nbIcb);
        if (nbIcbEff >= nbIcbEffThreshold) {
            simdBlocks = nbIcb;
            break;
        }
    }
    icBlock = std::min((execType == KdnnBrgemmConvExecTypeT::TRANS) ? RndUp(ic, paddedIc) : ic, simdBlocks * simdW);
    nbIc = Math::CeilDiv(ic, icBlock);
}

void BrgBlockingT::UpdateBlocks()
{
    if (spBlock <= 0 || Service::OneOf(0, odBlock, ohBlock, icBlock, ocBlock,
        kdBlock, khBlock, kwBlock, osBlock, owBlock)) {
        return;
    }

    nbOd = Math::CeilDiv(od, odBlock);
    nbOh = Math::CeilDiv(oh, ohBlock);
    nbIc = Math::CeilDiv(ic, icBlock);
    nbOc = Math::CeilDiv(oc, ocBlock);
    nbKd = Math::CeilDiv(kd, kdBlock);
    nbKh = Math::CeilDiv(kh, khBlock);
    nbKw = Math::CeilDiv(kw, kwBlock);
    nbOw = Math::CeilDiv(ow, owBlock);
    if (isOsBlocking) {
        nbOs = Math::CeilDiv(os, osBlock);
        sp = os;
        spBlock = osBlock;
        nbSp = nbOs;
    } else {
        sp = ow;
        spBlock = owBlock;
        nbSp = nbOw;
        iwBlock = GetInpSize(iwp, owBlock, kw, strideW, dilateW);
    }
}

bool BrgBlockingT::FastCheckOcBlock1x1() const
{
    // this function for reducing the number of blocking variants
    // eliminate heuristic in this function
    const auto rndOc = RndUp(oc, accSimdW);
    auto res = false;
    if (ocBlock == NUM_64) {
        const auto bigSpatial = od * oh * ow >= NUM_64 * strideD * strideH * strideW;
        res = (rndOc % ocBlock == 0 && bigSpatial);
    } else if (ocBlock == NUM_48) {
        const auto ocBlockEff = static_cast<float>(oc) / RndUp(oc, ocBlock);
        res = (ocBlockEff >= 0.95f);
    } else {
        res = true;
    }

    return res;
}

float BrgBlockingT::EstEff1x1()
{
    const auto ocblock = ocBlock / accSimdW;
    const auto brgemmMicrokernelEff = (static_cast<float>(ocblock) * ur) / ((ur + ocblock) * maxRegs);
    const auto urEff = static_cast<float>(spBlock) / RndUp(spBlock, ur);
    const auto brgemmEff = SqueezeVal(ur * (2.0f - std::min(1.9f, static_cast<float>(ur) / spBlock)) / NUM_64, 0.5f);
    const auto spAmount = isOsBlocking ? Math::CeilDiv(nbOs, nbOsBlocking) : nbOd * nbOh * nbSp;
    const auto workAmount = mb * ngroups * nbOc * spAmount;
    const auto spEff = static_cast<float>(sp) / RndUp(sp, spBlock);
    const auto thrEff = static_cast<float>(workAmount) / RndUp(workAmount, nthr);
    const auto ocBlockEff = static_cast<float>(oc) / RndUp(oc, ocBlock);
    const auto job = Math::CeilDiv(workAmount, nthr);
    const auto dimOc = (loopOrder == KdnnBrgemmConvLoopOrderT::NDHWGC) ? 1 : spAmount;
    const auto nbOcThr = std::min(nbOc, Math::CeilDiv(job, dimOc));
    const auto ocThr = std::min(oc, nbOcThr * ocBlock);
    const auto nsimdOcThr = Math::CeilDiv(ocThr, simdW);
    const auto dimSp = (loopOrder == KdnnBrgemmConvLoopOrderT::NDHWGC) ? ngroups * nbOc : 1;
    const auto nbSpThr = std::min(nbSp, Math::CeilDiv(job, dimSp));
    const auto spThr = std::min(sp, nbSpThr * spBlock);
    IntType nbOhThr{1};
    IntType ohThr{1};
    IntType nbOdThr{1};
    IntType odThr{1};
    if (!isOsBlocking) {
        const auto dimOh = nbSp * dimSp;
        nbOhThr = std::min(nbOh, Math::CeilDiv(job, dimOh));
        ohThr = std::min(oh, nbOhThr * ohBlock);
        const auto dimOd = nbOh * dimOh;
        nbOdThr = std::min(nbOd, Math::CeilDiv(job, dimOd));
        odThr = std::min(od, nbOdThr * odBlock);
    }
    auto jobEff = 1.0f;
    if (job < nthr) {
        std::vector<IntType> thrJobs(nthr);
        for (IntType ithr = 0; ithr < nthr; ithr++) {
            thrJobs[ithr] = 0;
            if (ithr >= workAmount) {
                continue;
            }
            IntType thrJob = 0;
            IntType start {0};
            IntType end {0};
            Balance211(workAmount, nthr, ithr, start, end);
            IntType nTmp {0};
            IntType g {0};
            IntType ocb {0};
            IntType oss {0};
            IntType odpTmp {0};
            IntType ohpTmp {0};
            IntType spb {0};
            if (loopOrder == KdnnBrgemmConvLoopOrderT::NDHWGC) {
                if (isOsBlocking) {
                    NdIteratorInit(start, nTmp, mb, oss, spAmount, g, ngroups, ocb, nbOc);
                } else {
                    NdIteratorInit(start, nTmp, mb, odpTmp, od, ohpTmp, oh, spb, nbSp, g, ngroups, ocb, nbOc);
                }
            } else if (loopOrder == KdnnBrgemmConvLoopOrderT::NGCDHW) {
                if (isOsBlocking) {
                    NdIteratorInit(start, nTmp, mb, g, ngroups, ocb, nbOc, oss, spAmount);
                } else {
                    NdIteratorInit(start, nTmp, mb, g, ngroups, ocb, nbOc, odpTmp, od, ohpTmp, oh, spb, nbSp);
                }
            }
            for (auto work = start; work < end; work++) {
                const IntType ocpTmp = ocb * ocBlock;
                const auto ocSz = std::min(oc - ocpTmp, ocBlock);
                IntType spSz = 0;
                if (isOsBlocking) {
                    const auto osbStart = oss * nbOsBlocking;
                    const auto osbRange = std::min(nbOs - osbStart, nbOsBlocking);
                    for (IntType osb = 0; osb < osbRange; osb++) {
                        const IntType osp = (osbStart + osb) * spBlock;
                        spSz = std::min(os - osp, spBlock);
                    }
                } else {
                    const IntType spp = spb * spBlock;
                    spSz = std::min(sp - spp, spBlock);
                }
                thrJob += spSz * ocSz;
                if (loopOrder == KdnnBrgemmConvLoopOrderT::NDHWGC) {
                    if (isOsBlocking) {
                        NdIteratorStep(nTmp, mb, oss, spAmount, g, ngroups, ocb, nbOc);
                    } else {
                        NdIteratorStep(nTmp, mb, odpTmp, od, ohpTmp, oh, spb, nbSp, g, ngroups, ocb, nbOc);
                    }
                } else if (loopOrder == KdnnBrgemmConvLoopOrderT::NGCDHW) {
                    if (isOsBlocking) {
                        NdIteratorStep(nTmp, mb, g, ngroups, ocb, nbOc, oss, spAmount);
                    } else {
                        NdIteratorStep(nTmp, mb, g, ngroups, ocb, nbOc, odpTmp, od, ohpTmp, oh, spb, nbSp);
                    }
                }
            }
            thrJobs[ithr] = thrJob;
        }
        IntType maxJob = 0;
        IntType sumJob = 0;
        for (IntType ithr = 0; ithr < nthr; ithr++) {
            if (thrJobs[ithr] > maxJob) {
                maxJob = thrJobs[ithr];
            }
            sumJob += thrJobs[ithr];
        }
        jobEff = maxJob == 0 ? 1 : static_cast<float>(sumJob) / (maxJob * nthr);
    } else {
        jobEff = thrEff;
    }
    const auto icBlockingSize = icBlock * nbIcBlocking;
    const auto ocBlockingSize = ocBlock * icBlockingSize;
    IntType l = -1;
    // brgemm kernel: loop by simdW
    l++;
    loop[l].src.Set(ur * simdW, 1, bcastSimd);
    loop[l].dst.Set(0, 1);
    loop[l].wei.Set(ocBlock, 1);
    // brgemm kernel: loop by ur in spBlock
    l++;
    const auto nbUr = Math::CeilDiv(spBlock, ur);
    loop[l].src.Set(ur * RndSimd(icBlockingSize), 1);
    loop[l].dst.Set(ur * ocBlock, 1);
    loop[l].wei.Set(ocBlockingSize, nbUr);
    // brgemm kernel: loop by icChunks
    l++;
    const auto icChunks = Math::CeilDiv(nbIc, nbIcBlocking);
    loop[l].src.Set(spBlock * icBlockingSize, 1);
    loop[l].dst.Set(spBlock * ocBlock, icChunks);
    auto weiIs = ocBlockingSize;
    auto weiOp = ocblock * ic;
    loop[l].wei.Set(weiIs, 1);
    if (loopOrder == KdnnBrgemmConvLoopOrderT::NDHWGC) {
        // harness: loop by ocBlock
        l++;
        loop[l].src.Set(spBlock * RndSimd(ic), nbOcThr);
        loop[l].dst.Set(spBlock * ocBlock, 1);
        weiIs = ocBlock * ic;
        weiOp = nsimdOcThr * ic;
        loop[l].wei.Set(weiIs, 1);
    }
    const auto rndOcForSp = simdW * (loopOrder == KdnnBrgemmConvLoopOrderT::NDHWGC ? nsimdOcThr : ocblock);
    if (isOsBlocking) {
        // harness: loop by os_blocks
        l++;
        loop[l].src.Set(spBlock * icBlockingSize, 1);
        loop[l].dst.Set(spBlock * rndOcForSp, 1);
        loop[l].wei.Set(weiOp * simdW, nbSpThr);
    } else {
        // harness: loop by sp_blocks
        l++;
        loop[l].src.Set(spBlock * icBlockingSize, 1);
        loop[l].dst.Set(spBlock * rndOcForSp, 1);
        loop[l].wei.Set(weiOp * simdW, nbSpThr);
        // harness: loop by oh_blocks
        l++;
        loop[l].src.Set(ohBlock * spThr * RndSimd(icBlockingSize), 1);
        loop[l].dst.Set(ohBlock * spThr * rndOcForSp, 1);
        loop[l].wei.Set(weiOp * simdW, nbOhThr);
        // harness: loop by od_blocks
        l++;
        loop[l].src.Set(odBlock * ohThr * spThr * RndSimd(icBlockingSize), 1);
        loop[l].dst.Set(odBlock * ohThr * spThr * rndOcForSp, 1);
        loop[l].wei.Set(weiOp * simdW, nbOdThr);
    }
    if (loopOrder != KdnnBrgemmConvLoopOrderT::NDHWGC) {
        // harness: loop by ocBlock
        l++;
        loop[l].src.Set(odThr * ohThr * RndSimd(spThr * icBlockingSize), nbOcThr);
        loop[l].dst.Set(ocBlock * odThr * ohThr * spThr, 1);
        loop[l].wei.Set(ocBlock * ic, 1);
    }
    // harness: loop by mb
    l++;
    const auto mbThr = std::min(mb, Math::CeilDiv(job, spAmount * ngroups * nbOc));
    loop[l].src.Set(odThr * ohThr * spThr * RndSimd(icBlockingSize), 1);
    loop[l].dst.Set(nsimdOcThr * simdW * odThr * ohThr * spThr, 1);
    loop[l].wei.Set(nsimdOcThr * ic * simdW, mbThr);
    const auto srcOp = static_cast<IntType>(mbThr) * odThr * ohThr * spThr * icBlockingSize;
    const auto dstOp = static_cast<IntType>(mbThr) * nsimdOcThr * odThr * ohThr * spThr;
    weiOp = nsimdOcThr * ic;
    // for real application set benchIterations to 1
    const auto iterations = benchIterations;
    l++;
    loop[l].src.Set(srcOp, iterations);
    loop[l].dst.Set(dstOp * simdW, iterations);
    loop[l].wei.Set(weiOp * simdW, iterations);
    auto srcMemK = memK;
    auto dstMemK = memK;
    auto weiMemK = memK;
    float srcRp = 1;
    float dstRp = 1;
    float weiRp = 1;
    for (auto il = l; il >= 0; il--) {
        srcMemK = IoK(loop[il], loop[il].src, srcMemK, true, false);
        dstMemK = IoK(loop[il], loop[il].dst, dstMemK, false, false);
        weiMemK = IoK(loop[il], loop[il].wei, weiMemK, false, true);
        srcRp *= loop[il].src.repeatn;
        dstRp *= loop[il].dst.repeatn;
        weiRp *= loop[il].wei.repeatn;
    }
    const auto srcOps = (srcOp * srcRp) / iterations;
    const auto dstOps = (dstOp * dstRp) / iterations;
    const auto weiOps = (weiOp * weiRp) / iterations;
    const auto srcCost = srcMemK * srcOps;
    const auto dstCost = dstMemK * dstOps;
    const auto weiCost = weiMemK * weiOps;
    const auto callKernelCost = 1000.f * job * icChunks;
    const auto upSpSize = isOsBlocking ? 1 : od * oh;
    const auto cacheEff = (static_cast<IntType>(mb) * upSpSize * sp * ic * oc)
        / (nthr * (srcCost + dstCost + weiCost + callKernelCost));
    const auto resEff = ocBlockEff * brgemmMicrokernelEff * spEff * jobEff * urEff * cacheEff * brgemmEff;

    return resEff;
}

void BrgBlockingT::CalcBlocks1x1()
{
    const bool isIcZeroPadded = ic != icWithoutPadding;
    isRtus = isIcZeroPadded;
    const bool isOsBlockingOk = strideD == 1 && strideH == 1 && iw % strideW == 0;
    if (isOsBlockingOk || isRtus) {
        sp = os;
        isOsBlocking = true;
    } else {
        sp = ow;
        isOsBlocking = false;
    }
    odBlock = 1;
    ohBlock = 1;
    kdBlock = khBlock = kwBlock = 1;
    kdBlockPad = khBlockPad = kwBlockPad = 1;
    nbIcBlocking = 1;
    const float thrEffThreshold = 0.9f;
    const IntType maxSpBlockL2 = os;
    // nbOsBlocking always is 1 for now, update this code
    nbOsBlocking = 1;
    IntType startSpBlock = 0;
    if (isOsBlocking) {
        owBlock = 0;
        const IntType maxOsBlockThr = srcDsz * ic >= BRGEMM_CONV_UTILS_NUM_1024
            && srcDsz * ic < BRGEMM_CONV_UTILS_NUM_4096
                ? std::max(std::min(BRGEMM_CONV_UTILS_NUM_16, os),
                    Math::CeilDiv(os, Math::CeilDiv(nthr, mb * Math::CeilDiv(oc, ocBlock))))
                : std::max(Math::CeilDiv(BRGEMM_CONV_UTILS_NUM_2048, ocBlock),
                    static_cast<IntType>(Math::CeilDiv(mb * ngroups * os, nthr)));
        const IntType maxOsBlockL2 = maxSpBlockL2;
        IntType maxOsBlockAliasing = BRGEMM_CONV_UTILS_NUM_1000000 / nthr;
        if ((ocWithoutPadding * os * dstDsz) % BRGEMM_CONV_UTILS_NUM_4096 == 0) {
            maxOsBlockAliasing /= 1;
            for (auto curOc = ocWithoutPadding; maxOsBlockAliasing * dstDsz > BRGEMM_CONV_UTILS_NUM_400
                    && curOc % NUM_2 == 0 && curOc * os * dstDsz >= BRGEMM_CONV_UTILS_NUM_4096; curOc /= NUM_2) {
                maxOsBlockAliasing /= NUM_2;
            }
            maxOsBlockAliasing += maxOsBlockAliasing % NUM_2 ? 0 : 1;
        }
        maxOsBlockAliasing = std::min(Math::CeilDiv(BRGEMM_CONV_UTILS_NUM_1001, static_cast<IntType>(dstDsz)),
            maxOsBlockAliasing);
        startSpBlock = Saturate(static_cast<IntType>(1), os,
            std::min(std::min(maxOsBlockThr, maxOsBlockL2), maxOsBlockAliasing));
    } else {
        osBlock = 0;
        const auto maxOwBlockThr = Saturate(static_cast<IntType>(1), ow,
            static_cast<IntType>(Math::CeilDiv(mb * ngroups * nbOc * os,
                static_cast<IntType>(thrEffThreshold * nthr))));
        const auto maxOwBlockL2 = maxSpBlockL2;
        startSpBlock = Saturate(static_cast<IntType>(1), ow, std::min(maxOwBlockThr, maxOwBlockL2));
    }
    osBlock = owBlock = spBlock = -1;
    BrgBlockingT bestBrgb = *this;
    IntType prevSpb = 0;
    for (IntType ns = 1; ns <= sp; ns++) {
        IntType spb = Math::CeilDiv(sp, ns);
        if (spb == prevSpb || spb > startSpBlock) {
            continue;
        }
        prevSpb = spb;
        osBlock = owBlock = spBlock = spb;
        SelectIcBlock();
        Status st = EstimateBrgemmUr();
        if (st != Status::SUCCESS) {
            continue;
        }
        UpdateBlocks();
        useBuffer = (dstDt != accDt || withSum) && (icBlock * nbIcBlocking < ic);
        eff = EstEff1x1();
        if (eff > bestBrgb.eff || bestBrgb.eff == 0) {
            bestBrgb = *this;
        }
    }
    *this = bestBrgb;
    osBlock = owBlock = spBlock;
    UpdateBlocks();
}

static Status InitBcp(KdnnBrgemmConvConfT &bcp, ARMCG::CPUISAT isa, const KuDNN::TensorInfo &src,
    const KuDNN::TensorInfo &weights, const KuDNN::TensorInfo &dst, const KuDNN::TensorInfo &bias,
    Shape strides, Shape paddingL, Shape dilates, Propagation prop, IntType nthr)
{
    BrgBlockingT::cL1 = ARMCG::GetCoreCacheSize(NUM_1);
    BrgBlockingT::cL2 = ARMCG::GetCoreCacheSize(NUM_2);
    BrgBlockingT::cL3 = ARMCG::GetCoreCacheSize(NUM_3);
    const bool withGroups = weights.GetNumDims() == src.GetNumDims() + 1;
    IntType ndims = src.GetNumDims();
    bcp = std::remove_reference_t<decltype(bcp)>();
    bcp.isa = isa;
    bcp.ndims = ndims;
    bcp.propKind = prop;
    bcp.ngroups = withGroups ? weights.GetDims()[IDX_0] : 1;
    bcp.mb = src.GetDims()[IDX_0];
    bcp.ocWithoutPadding = dst.GetDims()[IDX_1];
    bcp.oc = bcp.ocWithoutPadding / bcp.ngroups;
    bcp.icWithoutPadding = src.GetDims()[IDX_1] / bcp.ngroups;
    bcp.ic = bcp.icWithoutPadding;
    bcp.id = (ndims == DIM_5) ? src.GetDims()[BRGEMM_CONV_UTILS_NUM_2] : 1;
    bcp.ih = (ndims == DIM_3) ? 1 : src.GetDims()[ndims - BRGEMM_CONV_UTILS_NUM_2];
    bcp.iw = src.GetDims()[ndims - IDX_1];
    bcp.od = (ndims == DIM_5) ? dst.GetDims()[BRGEMM_CONV_UTILS_NUM_2] : 1;
    bcp.oh = (ndims == DIM_3) ? 1 : dst.GetDims()[ndims - BRGEMM_CONV_UTILS_NUM_2];
    bcp.ow = dst.GetDims()[ndims - IDX_1];
    bcp.kd = (ndims == DIM_5) ? weights.GetDims()[withGroups + BRGEMM_CONV_UTILS_NUM_2] : 1;
    bcp.kh = (ndims == DIM_3) ? 1 : weights.GetDims()[withGroups + ndims - BRGEMM_CONV_UTILS_NUM_2];
    bcp.kw = weights.GetDims()[withGroups + ndims - IDX_1];
    bcp.fPad = (ndims == DIM_5) ? paddingL[0] : 0;
    bcp.tPad = (ndims == DIM_3) ? 0 : paddingL[ndims - IDX_4];
    bcp.lPad = paddingL[ndims - IDX_3];
    bcp.strideD = (ndims == DIM_5) ? strides[0] : 1;
    bcp.strideH = (ndims == DIM_3) ? 1 : strides[ndims - IDX_4];
    bcp.strideW = strides[ndims - IDX_3];
    bcp.dilateD = (ndims == DIM_5) ? dilates[0] : 0;
    bcp.dilateH = (ndims == DIM_3) ? 0 : dilates[ndims - IDX_4];
    bcp.dilateW = dilates[ndims - IDX_3];
    bcp.os = bcp.od * bcp.oh * bcp.ow;
    bcp.extKd = CalculateExtendedFilterSize(bcp.kd, bcp.dilateD);
    bcp.extKh = CalculateExtendedFilterSize(bcp.kh, bcp.dilateH);
    bcp.extKw = CalculateExtendedFilterSize(bcp.kw, bcp.dilateW);
    bcp.backPad = CalculateEndPadding(bcp.fPad, bcp.od, bcp.id, bcp.strideD, bcp.extKd);
    bcp.bPad = CalculateEndPadding(bcp.tPad, bcp.oh, bcp.ih, bcp.strideH, bcp.extKh);
    bcp.rPad = CalculateEndPadding(bcp.lPad, bcp.ow, bcp.iw, bcp.strideW, bcp.extKw);
    bcp.is1x1 = bcp.fPad <= 0 && bcp.backPad <= 0 && bcp.tPad <= 0 && bcp.bPad <= 0 && bcp.lPad <= 0 && bcp.rPad <= 0
        && bcp.kd == 1 && bcp.kh == 1 && bcp.kw == 1;
    bcp.withBias = bias.GetDims()[0] != 0;
    // origin code: get bias type is not UNDEFINED
    bcp.srcDt = Element::TypeT(src.GetType());
    bcp.dstDt = Element::TypeT(dst.GetType());
    bcp.weiDt = Element::TypeT(weights.GetType());
    bcp.biaDt = bcp.withBias ? Element::TypeT(bias.GetType()) : Element::TypeT::UNDEFINED;
    if (bcp.srcDt != Element::TypeT::F32 || bcp.weiDt != Element::TypeT::F32 || bcp.dstDt != Element::TypeT::F32) {
        return Status::UNSUPPORTED;
    }
    if (bcp.srcDt == Element::TypeT::F32 || bcp.srcDt == Element::TypeT::BF16 || bcp.srcDt == Element::TypeT::F16) {
        bcp.accDt = Element::TypeT::F32;
    } else if (bcp.srcDt == Element::TypeT::U8 || bcp.srcDt == Element::TypeT::S8) {
        bcp.accDt = Element::TypeT::S32;
    } else {
        return Status::UNSUPPORTED;
    }
    bcp.srcDsz = src.GetType().GetSize();
    bcp.weiDsz = weights.GetType().GetSize();
    bcp.dstDsz = dst.GetType().GetSize();
    bcp.accDsz = Element::Type(bcp.accDt).GetSize();
    bcp.biaDsz = bcp.withBias ? bias.GetType().GetSize() : 0;
    bcp.simdW = ARMCG::IsaMaxVlen(isa) / bcp.srcDsz;
    bcp.accSimdW = ARMCG::IsaMaxVlen(isa) / bcp.accDsz;
    bcp.isBF32 = false;
    // [unimplemented]
    bcp.weiPlain = false
        && bcp.weiDt == Element::TypeT::F32
        && IsSuperset(isa, ARMCG::CPUISAT::VC_512)
        && (weights.GetLayout() == Layout::ABCDE
            || weights.GetLayout() == Layout::ABCD
            || weights.GetLayout() == Layout::ABC);
    if (bcp.weiPlain) {
        bcp.ldB = bcp.ocWithoutPadding;
    }
    if ((bcp.propKind == Propagation::FORWARD_TRAINING || bcp.propKind == Propagation::FORWARD_INFERENCE)
        && bcp.ngroups == 1 && bcp.dilateW == 0 && bcp.kw > 1
        && bcp.strideW > 1 && bcp.lPad <= 0 && bcp.rPad <= 0) {
        // such convolutions are equivalent to
        // [iw / k][kw / k][strideW / k][ic * k]
        const bool pure1d = bcp.mb == 1 && bcp.id == 1 && bcp.ih == 1;
        IntType wKoef = 1;
        auto wKoefMax = std::min(bcp.kw, std::min(bcp.strideW, bcp.iw));
        for (IntType i = 1; i <= wKoefMax; i++) {
            if (Implication(!pure1d, bcp.iw % i == 0)
                && Implication(bcp.ic * i > bcp.simdW, (bcp.ic * i) % bcp.simdW == 0)
                && bcp.kw % i == 0 && bcp.strideW % i == 0) {
                wKoef = i;
            }
        }
        if (wKoef > 1) {
            bcp.icWithoutPadding *= wKoef;
            bcp.ic *= wKoef;
            bcp.iw /= wKoef;
            bcp.kw /= wKoef;
            bcp.strideW /= wKoef;
            bcp.extKw = CalculateExtendedFilterSize(bcp.kw, bcp.dilateW);
            bcp.rPad = CalculateEndPadding(bcp.lPad, bcp.ow, bcp.iw, bcp.strideW, bcp.extKw);
        }
    }
    BrgBlockingT::lastIcBlockSize = DataTypeVnniGranularity(bcp.weiDt);
    // optimize depthwise convolutions (for now direct approach is faster)
    const bool isDepthwise = withGroups && bcp.ngroups > 1 && bcp.ic == 1 && bcp.oc == 1;
    if (isDepthwise && AllowPerfHeuristics(bcp)) {
        return Status::UNSUPPORTED;
    }
    // optimize grouped convolutions with small ic
    const bool isGroupedSmallIc = bcp.propKind != Propagation::BACKWARD_WEIGHTS
        && withGroups
        && bcp.ngroups > 1
        // Enable the shapes not supported in direct convs
        && Implication(withGroups, IsGroupsOk(bcp))
        && bcp.ic <= bcp.accSimdW;
    if (isGroupedSmallIc && AllowPerfHeuristics(bcp)) {
        return Status::UNSUPPORTED;
    }
    // optimize the perf of 3d shape with small ic and large spatial
    const bool isSignedInput = bcp.srcDt == Element::TypeT::S8;
    bcp.s8s8CompensationRequired = isSignedInput && !IsSuperset(bcp.isa, ARMCG::CPUISAT::VC_256);
    bcp.hasInt8Vnni = IsSuperset(bcp.isa, ARMCG::CPUISAT::VC_512) || IsSuperset(bcp.isa, ARMCG::CPUISAT::VC_256);
    const bool isF32 = true
        && bcp.srcDt == Element::TypeT::F32
        && bcp.weiDt == Element::TypeT::F32
        && bcp.dstDt == Element::TypeT::F32;
    if (!Implication(isF32, Service::OneOf(isa, ARMCG::CPUISAT::VC_512, ARMCG::CPUISAT::VC_256) || bcp.isBF32)) {
        return Status::UNSUPPORTED;
    }
    if (!Implication(bcp.weiDt == Element::TypeT::BF16, MayIUse(ARMCG::CPUISAT::VC_256))) {
        return Status::UNSUPPORTED;
    }
    if (!Implication(bcp.weiDt == Element::TypeT::F16, MayIUse(ARMCG::CPUISAT::VC_256))) {
        return Status::UNSUPPORTED;
    }
    if (!Implication(bcp.weiDt == Element::TypeT::S8,
        MayIUse(ARMCG::CPUISAT::VC_512) || bcp.isa == ARMCG::CPUISAT::VC_256)) {
        return Status::UNSUPPORTED;
    }
    bcp.withSum = false;
    bcp.withEltwise = false;
    bcp.withBinary = false;
    bcp.srcZeroPoint = false;
    bcp.dstZeroPoint = false;
    bcp.nthr = nthr;
    bcp.khSets = 1;
    bcp.kwSets = 1;
    bcp.copyBlockOnly = false;
    bcp.useMMask = 0;
    bcp.isOsBlocking = false;
    bcp.oskip = 0;
    bcp.useUker = false;
    bcp.useInterleaveStores = false;
    bcp.hintPrefetching = BrgemmKernelPrefetchingT::UNDEFINED;
    bcp.brgemmBdLoopInnermost = false;
    bcp.idp = bcp.id + bcp.fPad + bcp.backPad;
    bcp.ihp = bcp.ih + bcp.tPad + bcp.bPad;
    bcp.iwp = bcp.iw + bcp.lPad + bcp.rPad;

    return Status::SUCCESS;
}

Status BrgemmConvInit1x1Conf(KdnnBrgemmConvConfT &bcp, ARMCG::CPUISAT isa, const TensorInfo &src,
    const TensorInfo &weights, const TensorInfo &dst, const TensorInfo &bias, Shape strides, Shape paddingL,
    Shape dilates, Propagation prop, IntType nthr)
{
    if (!MayIUse(isa)) {
        return Status::BAD_ARGUMENTS;
    }
    if (nthr == 0) {
        return Status::BAD_ARGUMENTS;
    }
    Status s = InitBcp(bcp, isa, src, weights, dst, bias, strides, paddingL, dilates, prop, nthr);
    if (s != Status::SUCCESS) {
        return Status::UNSUPPORTED;
    }
    if (!bcp.is1x1) {
        return Status::UNSUPPORTED;
    }
    if (!(bcp.srcDt == Element::TypeT::F32 && bcp.weiDt == Element::TypeT::F32 && bcp.dstDt == Element::TypeT::F32)) {
        return Status::UNSUPPORTED;
    }

    // blocking
    SizeType bcastAmount = bcp.id * bcp.ih * bcp.iw * bcp.srcDsz;
    SizeType loadAmount = bcp.oc * bcp.weiDsz;
    bcp.loopOrder = bcastAmount < loadAmount ? KdnnBrgemmConvLoopOrderT::NGCDHW : KdnnBrgemmConvLoopOrderT::NDHWGC;
    bcp.brgType = BrgemmBatchKindT::ADDRESS;
    bcp.maxBatch = 1;
    BrgBlockingT bestBrgb = std::remove_reference_t<decltype(bestBrgb)>();
    const auto minOcBlock = bcp.accSimdW;
    bestBrgb.ocBlock = minOcBlock;
    IntType startOcb = std::min(Math::CeilDiv(bcp.oc, bcp.accSimdW), BRGEMM_CONV_UTILS_NUM_4);
    const bool isOsBlockingOk = bcp.strideD == 1 && bcp.strideH == 1 && bcp.iw % bcp.strideW == 0;
    if (bcp.weiPlain && isOsBlockingOk) {
        startOcb = Math::CeilDiv(bcp.oc, bcp.accSimdW);
    }
    IntType finishOcb = 1;
    for (auto ocb = startOcb; ocb >= finishOcb; ocb--) {
        BrgBlockingT curBrgb = std::remove_reference_t<decltype(curBrgb)>();
        curBrgb.GetFromBcp(bcp);
        curBrgb.ocBlock = ocb * minOcBlock;
        curBrgb.nbOc = Math::CeilDiv(bcp.oc, curBrgb.ocBlock);

        if (!curBrgb.FastCheckOcBlock1x1()) {
            continue;
        }
        curBrgb.CalcBlocks1x1();
        Status st = curBrgb.GetBrgemmUr(nullptr, &dst);
        if (st != Status::SUCCESS) {
            continue;
        }
        curBrgb.eff = curBrgb.EstEff1x1();
        if (curBrgb.eff > bestBrgb.eff) {
            bestBrgb = curBrgb;
        }
    }
    bestBrgb.SaveToBcp(bcp);
    // blocking
    bcp.brgStrideA = bcp.icBlock * bcp.srcDsz;
    bcp.brgStrideB = bcp.icBlock * bcp.ocWithoutPadding * bcp.weiDsz;
    if (bcp.icBlock == 0 || bcp.ocBlock == 0) {
        return Status::UNSUPPORTED;
    }
    // configure matrix sizes
    if (bestBrgb.isOsBlocking) {
        if (bcp.osBlock == 0) {
            return Status::UNSUPPORTED;
        }
        bcp.m = bcp.osBlock;
        bcp.brgM = bcp.osBlock;
        bcp.mTail = bcp.os % bcp.osBlock;
        bcp.brgMTail = bcp.os % bcp.osBlock;
    } else {
        if (bcp.owBlock == 0) {
            return Status::UNSUPPORTED;
        }
        bcp.m = bcp.owBlock;
        bcp.brgM = bcp.owBlock;
        bcp.mTail = bcp.ow % bcp.owBlock;
        bcp.brgMTail = bcp.ow % bcp.owBlock;
    }
    bcp.k = bcp.ic >= bcp.icBlock ? bcp.icBlock : 0;
    bcp.kTail = bcp.ic % bcp.icBlock;
    bcp.n = bcp.oc >= bcp.ocBlock ? bcp.ocBlock : 0;
    bcp.nTail = bcp.oc % bcp.ocBlock;
    // to avoid cache concurrent access from different threads
    bcp.gemmBatchSize = bcp.nbIcBlocking;
    bcp.adjustedBatchSize = Math::CeilDiv(RndUp(bcp.gemmBatchSize * sizeof(BrgemmBatchElementT),
        static_cast<SizeType>(BRGEMM_CONV_UTILS_NUM_4096)), sizeof(BrgemmBatchElementT));
    // heuristic to dispatch BF32 brgemm
    // the following condition checks for shapes where down-convert execution in brgemm fails
    if (bcp.isBF32 && bcp.ic < BRGEMM_CONV_UTILS_NUM_64 && bcp.ic % BRGEMM_CONV_UTILS_NUM_32 != 0) {
        return Status::UNSUPPORTED;
    }
    if (bcp.useUker) {
        bcp.hintPrefetching = BrgemmKernelPrefetchingT::PRF1;
    }
    if (!bcp.weiPlain) {
        bcp.ldB = bcp.ocBlock;
    }
    // no inp buffer or brgemm vpad for 1x1
    bcp.inpBufferSize = bcp.isRtus ? RndUp(bcp.ldA * bcp.os, BRGEMM_CONV_UTILS_NUM_64) : 0;
    bcp.inpBufferMaskSize =
        bcp.isRtus ? RndUp(Math::CeilDiv(bcp.nbIc, bcp.nbIcBlocking) * bcp.nbOs, BRGEMM_CONV_UTILS_NUM_64) : 0;
    bcp.execType = bcp.isRtus ? KdnnBrgemmConvExecTypeT::TRANS : KdnnBrgemmConvExecTypeT::BASE;
    bcp.ununrollBdLoop = static_cast<IntType>(bcp.m) * bcp.n * (bcp.isBF32 ? 1 : DIM_2)
        > NUM_8 * BRGEMM_CONV_UTILS_NUM_1024;
    bcp.bufferSize = bcp.ldC * bcp.m;
    bcp.compABufferSize = bcp.ngroups * bcp.nbOc * bcp.ocBlock;
    bcp.s8s8CompBufferSize = bcp.ngroups * bcp.nbOc * bcp.ocBlock;
    bcp.withScales = false;
    bcp.isOcScale = 0;
    bcp.reqCalCompPad = false;
    bcp.scaleAdjustFactor = 1.0f;

    return Status::SUCCESS;
}

} // namespace KuDNN
