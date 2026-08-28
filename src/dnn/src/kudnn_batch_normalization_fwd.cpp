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

#include <arm_sve.h>

#include "kudnn.hpp"

#include "internal_funcs.hpp"
#include "threading_impl.hpp"
#include "kernels/bnorm/kudnn_bnorm_kernels.hpp"

namespace KuDNN {
namespace Detail {

class BatchNormalizationLayerFWDImpl final : public Parallelizable {
public:

    BatchNormalizationLayerFWDImpl(const Propagation &propKind, const TensorInfo &srcInfo,
    const TensorInfo &statsInfo, const TensorInfo &scaleShiftInfo, const TensorInfo &dstInfo,
    NormalizationFlags flags) noexcept(false)
        : pKind(propKind), srcType(srcInfo.GetType()), dstType(dstInfo.GetType()),
        compType(Element::TypeT::F32), normFlags(flags)
    {
        // threading implementation depends on the value of calculateStats flag
        // so we initialize it at the constructor stage
        Service::ThrowOnStatus(BatchNormalizationLayerFWD::ValidateInput(pKind, srcInfo, statsInfo, scaleShiftInfo,
            dstInfo, flags), "Batch Normalization Layer FWD");
        reorderedDims = Service::BroadcastTo5D(
            Service::GetShapeAccordingToLayout(srcInfo.GetDims(), srcInfo.GetLayout()));
        reorderedStridesSrc = Service::BroadcastTo5D(
            Service::GetShapeAccordingToLayout(srcInfo.GetStrides(), srcInfo.GetLayout()));
        reorderedStridesDst = Service::BroadcastTo5D(
            Service::GetShapeAccordingToLayout(dstInfo.GetStrides(), dstInfo.GetLayout()));
        channelAxisNum = srcInfo.GetAxisNumAccordingToLayout(1) + MAX_DIMS - srcInfo.GetNumDims();

        channelSize = reorderedDims[channelAxisNum];
        innerSize = Service::ArrayProduct(reorderedDims, reorderedDims.GetNumDims(), channelAxisNum + 1);
        outerSize = Service::ArrayProduct(reorderedDims, channelAxisNum, 0);
        bool fuseRelu = static_cast<bool>(normFlags & NormalizationFlags::FUSE_NORM_RELU);
        isAXB = (innerSize == 1) && (!fuseRelu);
        // axb params
        axbBlkDataSize = (l2CacheBSize / srcType.GetSize() / NUM_2 + channelSize - NUM_1) / channelSize;
        // assign threads
        InitBatchedDims();
        SizeType optThreadsSize = GetOptNumThreads();
        if (!isAXB) {
            mainSplitDims = Threading::SplitDims(mainBatchedDims, optThreadsSize);
            realNumThreads = mainSplitDims.size();
            statSplitDims = AssainBalance(statBatchedDims, realNumThreads);
        } else {
            realNumThreads = optThreadsSize;
            mainSplitDims = AssainBalance(mainBatchedDims, realNumThreads);
            statSplitDims = AssainBalance(statBatchedDims, realNumThreads);
        }
        isSingleThread = realNumThreads == 1;
        AllocMeanVarCN();
        tmpMean.resize(channelSize, 0.0f);
        tmpVar.resize(channelSize, 0.0f);
        mulFactors.resize(channelSize, 0.0f);
    };

    // calculate normalization
    template <typename CompType, typename InType, typename OutType>
    void Evaluate(const Shape &batchSplitDims, const void *src, void *dst,
            const float *shift, const float *mean, const float *mulFactor, void *ws) const noexcept
    {
        const InType *typedSrc = static_cast<const InType *>(src);
        OutType *typedDst = static_cast<OutType *>(dst);
        std::uint8_t *typedWs = static_cast<std::uint8_t *>(ws);

        BNorm::EvaluateCrossAxis<float>(batchSplitDims, typedSrc, typedDst, shift,
            mean, mulFactor, normFlags, channelSize, innerSize, typedWs, pKind);
    }

    // axb normalization
    template <typename CompType, typename InType, typename OutType>
    void Evaluate(SizeType loopSize, SizeType loopCount, const void *src, void *dst,
            const float *shift, const float *mean, const float *mulFactor, void *ws) const noexcept
    {
        const InType *typedSrc = static_cast<const InType *>(src);
        OutType *typedDst = static_cast<OutType *>(dst);
        std::uint8_t *typedWs = static_cast<std::uint8_t *>(ws);
        SizeType splitNum = loopSize / loopCount;
        SizeType splitNumTail = splitNum + 1;
        SizeType tailCount = loopSize - splitNum * loopCount;
        SizeType offsetSD = 0;
        SizeType i = 0;
        for (; i < tailCount; i++) {
            BNorm::EvaluateBNormAXB<float>(typedSrc + offsetSD, typedDst + offsetSD, shift,
                mean, mulFactor, normFlags, channelSize, splitNumTail, typedWs, pKind);
            offsetSD += splitNumTail * channelSize;
        }
        for (; i < loopCount; i++) {
            BNorm::EvaluateBNormAXB<float>(typedSrc + offsetSD, typedDst + offsetSD, shift,
                mean, mulFactor, normFlags, channelSize, splitNum, typedWs, pKind);
            offsetSD += splitNum * channelSize;
        }
    }

    void RunSequential(const SizeType &ithr, const void *src, void *dst,
            const float *shift, const float *mean, const float *mulFactor, void *ws) const noexcept(false)
    {
        if (!isAXB) {
            const Shape &batchSplitDims = mainSplitDims[ithr].first;
            const Shape &batchStrides = mainSplitDims[ithr].second;
            SizeType strideSrcDst = 0;
            SizeType strideStat = 0;
            for (SizeType i = 0; i < batchStrides.GetNumDims(); ++i) {
                strideSrcDst += batchStrides[i] * srcBatchedStrides[i];
            }
            strideSrcDst *= srcType.GetSize();
            strideStat = batchStrides[IDX_1];
            SizeType strideWs = strideSrcDst / dstType.GetSize(); // element size of ws is 1 byte
            Service::ParseType<true, false>(compType, srcType, dstType, *this, batchSplitDims,
                static_cast<const std::byte *>(src) + strideSrcDst, static_cast<std::byte *>(dst) + strideSrcDst,
                shift + strideStat, mean + strideStat, mulFactor + strideStat,
                (ws == nullptr) ? nullptr : (static_cast<std::byte *>(ws) + strideWs));
        } else {
            const Shape &startSplitDims = mainSplitDims[ithr].first;
            const Shape &endSplitDims = mainSplitDims[ithr].second;
            SizeType loopSize = endSplitDims[IDX_0] - startSplitDims[IDX_0];
            if (loopSize == 0) {
                return;
            }
            SizeType strideSrcDst = startSplitDims[IDX_0] * channelSize * srcType.GetSize();
            SizeType strideWs = strideSrcDst / dstType.GetSize(); // element size of ws is 1 byte
            Service::ParseType<true, false>(compType, srcType, dstType, *this, loopSize, axbBlkNum[ithr],
                static_cast<const std::byte *>(src) + strideSrcDst, static_cast<std::byte *>(dst) + strideSrcDst,
                shift, mean, mulFactor, (ws == nullptr) ? nullptr : (static_cast<std::byte *>(ws) + strideWs));
        }
    }

    // calculate meanBlk
    template <typename CompType, typename InType, typename OutType>
    void Evaluate(const Shape &batchSplitDims, const void *src, float *meanCN)
    {
        const InType *typedSrc = static_cast<const InType *>(src);
        BNorm::EvaluateMeanCNCrossAxis<float, InType, OutType>(batchSplitDims, typedSrc, meanCN, outerSize,
            channelSize, innerSize);
    }
    // axb meanBlk
    template <typename CompType, typename InType, typename OutType>
    void Evaluate(SizeType loopSize, SizeType loopCount, const void *src, float *meanCN)
    {
        const InType *typedSrc = static_cast<const InType *>(src);
        SizeType splitNum = loopSize / loopCount;
        SizeType splitNumTail = splitNum + 1;
        SizeType tailCount = loopSize - splitNum * loopCount;
        SizeType offsetSD = 0;
        SizeType i = 0;
        for (; i < tailCount; i++) {
            BNorm::EvaluateMeanCNFwdAXB<float, InType, OutType>(typedSrc + offsetSD, meanCN + i * channelSize,
                channelSize, splitNumTail);
            offsetSD += splitNumTail * channelSize;
        }
        for (; i < loopCount; i++) {
            BNorm::EvaluateMeanCNFwdAXB<float, InType, OutType>(typedSrc + offsetSD, meanCN + i * channelSize,
                channelSize, splitNum);
            offsetSD += splitNum * channelSize;
        }
    }

    void RunSequential(SizeType ithr, const void *src, float *meanCN)
    {
        SizeType strideSrc = 0;
        SizeType strideMeanCN = 0;
        if (!isAXB) {
            const Shape &batchSplitDims = mainSplitDims[ithr].first;
            const Shape &batchStrides = mainSplitDims[ithr].second;
            
            for (SizeType i = 0; i < batchStrides.GetNumDims(); ++i) {
                strideSrc += batchStrides[i] * srcBatchedStrides[i];
            }
            strideSrc *= srcType.GetSize();
            strideMeanCN = batchStrides[IDX_0] * mvBatchedStrides[IDX_0] +
                batchStrides[IDX_1] * mvBatchedStrides[IDX_1];
            Service::ParseType<true, false>(compType, srcType, dstType, *this, batchSplitDims,
                static_cast<const std::byte *>(src) + strideSrc, meanCN + strideMeanCN);
        } else {
            const Shape &startSplitDims = mainSplitDims[ithr].first;
            const Shape &endSplitDims = mainSplitDims[ithr].second;
            SizeType loopSize = endSplitDims[IDX_0] - startSplitDims[IDX_0];
            if (loopSize == 0) {
                return;
            }
            strideMeanCN = multiCNStride[ithr] * channelSize;
            strideSrc = startSplitDims[IDX_0] * channelSize * srcType.GetSize();
            Service::ParseType<true, false>(compType, srcType, dstType, *this, loopSize, axbBlkNum[ithr],
                static_cast<const std::byte *>(src) + strideSrc, meanCN + strideMeanCN);
        }
    }

    // calculate varianceBlk
    template <typename CompType, typename InType, typename OutType>
    void Evaluate(const Shape &batchSplitDims, const void *src, const float *mean, float *varianceCN)
    {
        const InType *typedSrc = static_cast<const InType *>(src);
        BNorm::EvaluateVarianceCNCrossAxis<float, InType, OutType>(batchSplitDims, typedSrc, mean,
            varianceCN, outerSize, channelSize, innerSize);
    }

    // axb varianceBlk
    template <typename CompType, typename InType, typename OutType>
    void Evaluate(SizeType loopSize, SizeType loopCount, const void *src, const float *mean, float *varianceCN)
    {
        const InType *typedSrc = static_cast<const InType *>(src);
        SizeType splitNum = loopSize / loopCount;
        SizeType splitNumTail = splitNum + 1;
        SizeType tailCount = loopSize - splitNum * loopCount;
        SizeType offsetSD = 0;
        SizeType i = 0;
        for (; i < tailCount; i++) {
            BNorm::EvaluateVarianceCNFwdAXB<float, InType, OutType>(typedSrc + offsetSD, mean,
                varianceCN + i * channelSize, channelSize, splitNumTail);
            offsetSD += splitNumTail * channelSize;
        }
        for (; i < loopCount; i++) {
            BNorm::EvaluateVarianceCNFwdAXB<float, InType, OutType>(typedSrc + offsetSD, mean,
                varianceCN + i * channelSize, channelSize, splitNum);
            offsetSD += splitNum * channelSize;
        }
    }

    void RunSequential(SizeType ithr, const void *src, const float *mean, float *varianceCN)
    {
        SizeType strideSrc = 0;
        SizeType strideVarCN = 0;
        if (!isAXB) {
            const Shape &batchSplitDims = mainSplitDims[ithr].first;
            const Shape &batchStrides = mainSplitDims[ithr].second;
            for (SizeType i = 0; i < batchStrides.GetNumDims(); ++i) {
                strideSrc += batchStrides[i] * srcBatchedStrides[i];
            }
            strideSrc *= srcType.GetSize();
            strideVarCN = batchStrides[IDX_0] * mvBatchedStrides[IDX_0] +
                batchStrides[IDX_1] * mvBatchedStrides[IDX_1];
            Service::ParseType<true, false>(compType, srcType, dstType, *this, batchSplitDims,
                static_cast<const std::byte *>(src) + strideSrc, mean + batchStrides[IDX_1], varianceCN + strideVarCN);
        } else {
            const Shape &startSplitDims = mainSplitDims[ithr].first;
            const Shape &endSplitDims = mainSplitDims[ithr].second;
            SizeType loopSize = endSplitDims[IDX_0] - startSplitDims[IDX_0];
            if (loopSize == 0) {
                return;
            }
            strideVarCN = multiCNStride[ithr] * channelSize;
            strideSrc = startSplitDims[IDX_0] * channelSize * srcType.GetSize();
            Service::ParseType<true, false>(compType, srcType, dstType, *this, loopSize, axbBlkNum[ithr],
                static_cast<const std::byte *>(src) + strideSrc, mean, varianceCN + strideVarCN);
        }
    }

    // calculate mean and variance
    void CalMeanVariance(const SizeType &ithr, float *blkMeanVar, float *chMeanVar)
    {
        const Shape &startSplitDims = statSplitDims[ithr].first;
        const Shape &endSplitDims = statSplitDims[ithr].second;
        if (!isAXB) {
            SizeType chRange = endSplitDims[IDX_0] - startSplitDims[IDX_0];
            CalMeanVarianceFunc(chRange, blkMeanVar + startSplitDims[IDX_0] * outerSize,
                chMeanVar + startSplitDims[IDX_0]);
        } else {
            SizeType simdW = BNorm::GetSvlen<float>();
            SizeType loopRange = multiCNStride[realNumThreads -1] + axbBlkNum[realNumThreads - 1];
            SizeType chStart = startSplitDims[IDX_0] == cBlkSize ? channelSize : startSplitDims[IDX_0] * simdW;
            SizeType chEnd = endSplitDims[IDX_0]  == cBlkSize ? channelSize : endSplitDims[IDX_0] * simdW;
            SizeType loopCh = chEnd - chStart;
            BNorm::MergeMeanVar(blkMeanVar + chStart, chMeanVar + chStart, channelSize, outerSize,
                loopRange, loopCh);
        }
    }

    void CalMeanVarianceFunc(const SizeType &chRange, float *blkMeanVar, float *chMeanVar)
    {
        SizeType sumSize = outerSize * innerSize;
        for (SizeType c = 0; c < chRange; c++) {
            SizeType offsetC = c * outerSize;
            float sum0{0};
            float sum1{0};
            float sum2{0};
            float sum3{0};
            IntType n = NUM_0;
            IntType looplen = outerSize - NUM_4;
            for (; n <= looplen; n += IDX_4) {
                sum0 += blkMeanVar[offsetC + n];
                sum1 += blkMeanVar[offsetC + n + IDX_1];
                sum2 += blkMeanVar[offsetC + n + IDX_2];
                sum3 += blkMeanVar[offsetC + n + IDX_3];
            }
            looplen = outerSize;
            for (; n < looplen; n++) {
                sum0 += blkMeanVar[offsetC + n];
            }
            chMeanVar[c] = (sum0 + sum1 + sum2 + sum3) / sumSize;
        }
    }

    void Run(const void *src, void *dst, const float *scale,
            const float *shift, float *mean, float *variance,
        bool saveStats, const float eps, void *ws) noexcept(false)
    {
        if ((src == nullptr) || (dst == nullptr)) {
            throw Service::LogicError {"Batch Normalization: one of input matrices is nullptr"};
        }
        if ((static_cast<bool>(normFlags & NormalizationFlags::USE_GLOBAL_STATS) || saveStats) &&
            ((mean == nullptr) || (variance == nullptr))) {
            throw Service::LogicError {"Batch Normalization: one of input matrices is nullptr"};
        }
        if (static_cast<bool>(normFlags & NormalizationFlags::FUSE_NORM_RELU) && (ws == nullptr) &&
            (pKind == Propagation::FORWARD_TRAINING)) {
            throw Service::LogicError {"Batch Normalization: one of input matrices is nullptr"};
        }
        if (static_cast<bool>(normFlags & NormalizationFlags::USE_SCALE) && (scale == nullptr)) {
            throw Service::LogicError {"Batch Normalization: one of input matrices is nullptr"};
        }
        if (static_cast<bool>(normFlags & NormalizationFlags::USE_SHIFT) && (shift == nullptr)) {
            throw Service::LogicError {"Batch Normalization: one of input matrices is nullptr"};
        }
        BatchThreader(src, dst, scale, shift, mean, variance, saveStats, eps, ws);
    }

private:
    bool isAXB;
    Propagation pKind;
    Element::Type srcType;
    Element::Type dstType;
    Element::Type compType;
    NormalizationFlags normFlags;
    std::vector<float, Service::AlignedAllocator<float>> tmpMean;
    std::vector<float, Service::AlignedAllocator<float>> tmpVar;
    SizeType channelAxisNum;
    Shape reorderedDims;
    Shape reorderedStridesSrc;
    Shape reorderedStridesDst;
    Shape mainBatchedDims;
    Shape statBatchedDims;
    Shape srcBatchedStrides;
    Shape mvBatchedStrides;
    SizeType channelSize;
    SizeType innerSize;
    SizeType outerSize;
    SizeType axbBlkDataSize;
    SizeType cBlkSize;
    const SizeType l2CacheBSize = 786432;
    // when the data size is smaller than smallScaleThreshold, smallNumThreads is more optimal
    const SizeType smallNumThreads = 4;
    const SizeType smallScaleThreshold = 81920;
    // threads acount is control within in single numa, 920b numa is 40 core, 920f numa is 38 core.
    const SizeType optNumThreads = 40;
    SizeType realNumThreads;
    bool isSingleThread = false;
    std::vector<SizeType> axbBlkNum;
    std::vector<SizeType> multiCNStride;
    std::vector<float, Service::AlignedAllocator<float>> mulFactors;
    // the layout of meanBlk and varianceBlk is channel x N(the dims before channel)
    std::vector<float, Service::AlignedAllocator<float>> meanBlk;
    std::vector<float, Service::AlignedAllocator<float>> varianceBlk;
    std::vector<std::pair<Shape, Shape>> mainSplitDims;
    std::vector<std::pair<Shape, Shape>> statSplitDims;

    void InitBatchedDims() noexcept
    {
        if (!isAXB) {
            mainBatchedDims.ResetShape({outerSize, channelSize});
            statBatchedDims = {channelSize};
            srcBatchedStrides = {channelSize * innerSize, innerSize};
            mvBatchedStrides = {NUM_1, outerSize};
        } else {
            mainBatchedDims.ResetShape({outerSize});
            SizeType simdW = BNorm::GetSvlen<float>();
            cBlkSize = (channelSize + simdW - 1) / simdW;
            statBatchedDims = {cBlkSize};
        }
    }

    void SetStatsThrParams(SizeType threads) noexcept
    {
        ResetDimsThreads(statBatchedDims, threads);
    }

    void SetMainThrParams(SizeType threads) noexcept
    {
        ResetDimsThreads(mainBatchedDims, threads);
    }

    SizeType GetOptNumThreads() noexcept
    {
        SizeType maxThreads = Threading::GetMaxNumThreads();
        SizeType optnum = std::min(maxThreads, optNumThreads);
        if (outerSize * channelSize * innerSize <= smallScaleThreshold) {
            optnum = std::min(optnum, smallNumThreads);
        }
        return optnum;
    }

    std::vector<std::pair<Shape, Shape>> AssainBalance(const Shape &BatchDims, const SizeType numThreads)
    {
        std::vector<std::pair<Shape, Shape>> res;
        SizeType totalSize = BatchDims.GetTotalDimsSize();
        SizeType splitSize = totalSize / numThreads;
        SizeType restSize = totalSize % numThreads;
        SizeType startidx{0};
        SizeType endidx{0};
        std::array<SizeType, MAX_DIMS> arr;
        SizeType stride = 1;
        for (SizeType i = 0; i < BatchDims.GetNumDims(); i++) {
            arr[BatchDims.GetNumDims() - i - 1] = stride;
            stride *= BatchDims[BatchDims.GetNumDims() - i - 1];
        }
        Shape strideDims(arr.data(), BatchDims.GetNumDims());
        for (SizeType i = 0; i < restSize; i++) {
            endidx = startidx + splitSize + 1;
            res.push_back(std::make_pair(CalShapeIndex(startidx, strideDims), CalShapeIndex(endidx, strideDims)));
            startidx = endidx;
        }
        for (SizeType i = restSize; i < numThreads; i++) {
            endidx = startidx + splitSize;
            res.push_back(std::make_pair(CalShapeIndex(startidx, strideDims), CalShapeIndex(endidx, strideDims)));
            startidx = endidx;
        }
        return res;
    }

    Shape CalShapeIndex(SizeType index, const Shape &strideDims)
    {
        SizeType numDims = strideDims.GetNumDims();
        std::vector<SizeType> arr(numDims, NUM_0);
        for (SizeType i = 0; i < numDims; i++) {
            arr[i] += index / strideDims[i];
            index %= strideDims[i];
        }
        Shape res(arr.data(), numDims);
        return res;
    }

    void AllocMeanVarCN()
    {
        if (isAXB) {
            multiCNStride.resize(realNumThreads, 0);
            axbBlkNum.resize(realNumThreads, 0);
            SizeType total = 0;
            for (SizeType i = 0; i < realNumThreads; i++) {
                axbBlkNum[i] = (mainSplitDims[i].second[IDX_0] - mainSplitDims[i].first[IDX_0] + axbBlkDataSize - 1) /
                    axbBlkDataSize;
                multiCNStride[i] = total;
                total += axbBlkNum[i];
            }
            meanBlk.resize(total * channelSize, 0.0f);
            varianceBlk.resize(total * channelSize, 0.0f);
        } else {
            meanBlk.resize(outerSize * channelSize, 0.0f);
            varianceBlk.resize(outerSize * channelSize, 0.0f);
        }
    }

    void BatchThreader(const void *src, void *dst, const float *scale,
            const float *shift, float *mean, float *variance,
            bool saveStats, const float eps, void *ws)
    {
        if (isSingleThread) {
            if (!(static_cast<bool>(normFlags & NormalizationFlags::USE_GLOBAL_STATS))) {
                if (!saveStats) {
                    mean = tmpMean.data();
                    variance = tmpVar.data();
                }
                RunSequential(NUM_0, src, meanBlk.data());
                CalMeanVariance(NUM_0, meanBlk.data(), mean);
                RunSequential(NUM_0, src, mean, varianceBlk.data());
                CalMeanVariance(NUM_0, varianceBlk.data(), variance);
            }
            BNorm::EvaluateMulFactor<float>(mulFactors.data(), variance, scale, normFlags, channelSize, eps);
            RunSequential(NUM_0, src, dst, shift, mean, mulFactors.data(), ws);
        } else {
            if (!(static_cast<bool>(normFlags & NormalizationFlags::USE_GLOBAL_STATS))) {
                if (!saveStats) {
                    mean = tmpMean.data();
                    variance = tmpVar.data();
                }
#pragma omp parallel num_threads(realNumThreads)
                {
                    int ithr = omp_get_thread_num();
                    RunSequential(ithr, src, meanBlk.data());
#pragma omp barrier
                    CalMeanVariance(ithr, meanBlk.data(), mean);
#pragma omp barrier
                    RunSequential(ithr, src, mean, varianceBlk.data());
#pragma omp barrier
                    CalMeanVariance(ithr, varianceBlk.data(), variance);
                }
            }
            BNorm::EvaluateMulFactor<float>(mulFactors.data(), variance, scale, normFlags, channelSize, eps);
#pragma omp parallel for num_threads(realNumThreads)
            for (SizeType i = 0; i < realNumThreads; ++i) {
                RunSequential(i, src, dst, shift, mean, mulFactors.data(), ws);
            }
        }
    }
};
} // namespace Detail

BatchNormalizationLayerFWD::~BatchNormalizationLayerFWD() noexcept {}

BatchNormalizationLayerFWD::BatchNormalizationLayerFWD(const Propagation &propKind, const TensorInfo &srcInfo,
    const TensorInfo &statsInfo, const TensorInfo &scaleShiftInfo, const TensorInfo &dstInfo,
    NormalizationFlags flags) noexcept(false)
    : pImpl(new Detail::BatchNormalizationLayerFWDImpl(
        propKind, srcInfo, statsInfo, scaleShiftInfo, dstInfo, flags)) {}

BatchNormalizationLayerFWD::BatchNormalizationLayerFWD(const BatchNormalizationLayerFWD &other) noexcept(false)
    : pImpl(new Detail::BatchNormalizationLayerFWDImpl{*(other.pImpl.get())}) {}

BatchNormalizationLayerFWD::BatchNormalizationLayerFWD(BatchNormalizationLayerFWD &&other) noexcept
    : pImpl(std::move(other.pImpl)) {}

BatchNormalizationLayerFWD& BatchNormalizationLayerFWD::operator=(const BatchNormalizationLayerFWD &other)
    noexcept(false)
{
    if (this != &other) {
        this->pImpl.reset(new Detail::BatchNormalizationLayerFWDImpl{*(other.pImpl.get())});
    }
    return *this;
}

BatchNormalizationLayerFWD& BatchNormalizationLayerFWD::operator=(BatchNormalizationLayerFWD &&other) noexcept
{
    if (this != &other) {
        this->pImpl = std::move(other.pImpl);
    }
    return *this;
}

void BatchNormalizationLayerFWD::Run(const void *src, void *dst,
    const float *scale, const float *shift,
    float *mean, float *variance,
    bool saveStats, const float eps, void *ws) const noexcept(false)
{
    pImpl->Run(src, dst, scale, shift, mean, variance, saveStats, eps, ws);
}

Status BatchNormalizationLayerFWD::ValidateInput(const Propagation &propKind, const TensorInfo &srcInfo,
    const TensorInfo &statsInfo, const TensorInfo &scaleShiftInfo, const TensorInfo &dstInfo,
    NormalizationFlags flags) noexcept
{
    Service::Unused(flags);
    if ((propKind != Propagation::FORWARD_INFERENCE) && (propKind != Propagation::FORWARD_TRAINING)) {
        return Status::BAD_ARGUMENTS;
    }
    if (srcInfo != dstInfo) {
        return Status::BAD_ARGUMENTS;
    }
    if (statsInfo.GetType() != Element::TypeT::F32) {
        return Status::BAD_ARGUMENTS;
    }
    if (scaleShiftInfo.GetType() != Element::TypeT::F32) {
        return Status::BAD_ARGUMENTS;
    }
    if (!Service::OneOf(srcInfo.GetType(), Element::TypeT::F32,
        Element::TypeT::F16, Element::TypeT::BF16,
        Element::TypeT::S8)) {
        return Status::UNSUPPORTED;
    }
    if (srcInfo.GetTotalTensorSize() == 0) {
        return Status::UNSUPPORTED;
    }
    if (scaleShiftInfo.GetNumDims() != 1) {
        return Status::BAD_ARGUMENTS;
    }
    if (statsInfo.GetNumDims() != 1) {
        return Status::BAD_ARGUMENTS;
    }
    if (!srcInfo.IsDense() || !statsInfo.IsDense() || !scaleShiftInfo.IsDense() || !dstInfo.IsDense()) {
        return Status::UNSUPPORTED;
    }
    return Status::SUCCESS;
}

} // namespace KuDNN
