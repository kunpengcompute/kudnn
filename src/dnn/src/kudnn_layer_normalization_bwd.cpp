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

#include <memory>
#include <cmath>
#include <cstddef>

#include <arm_sve.h>

#include "kudnn.hpp"

#include "internal_funcs.hpp"
#include "threading_impl.hpp"
#include "kudnn_layer_normalization_impl.hpp"
#include "kernels/lnorm/kudnn_lnorm_kernels.hpp"

namespace KuDNN {
namespace Detail {

static constexpr SizeType LNORM_BWD_ELE_NUM_1 = 10000;
static constexpr SizeType LNORM_BWD_ELE_NUM_2 = 100000;

static constexpr SizeType LNORM_THREADS_BWD_THREASHOLD = 1;

// optNumThreads = c0 + c1 * numElems, 10000 < numElems <= 100000
static constexpr float LNORM_BWD_THREADS1_C0 = 0.87f;
static constexpr float LNORM_BWD_THREADS1_C1 = 0.000042f;
// optNumThreads = c0 + c1 * log(numElems), numElems > 100000
static constexpr float LNORM_BWD_THREADS2_C0 = -65.43f;
static constexpr float LNORM_BWD_THREADS2_C1 = 6.1f;

class NormalizationLayerBWDImpl final : public Parallelizable {
public:
    NormalizationLayerBWDImpl(const TensorInfo &srcInfo, const TensorInfo &statInfo,
        const TensorInfo &diffSrcInfo, const TensorInfo &diffDstInfo,
        const TensorInfo &scaleShiftInfo, const TensorInfo &diffSc,
        NormalizationFlags flags) noexcept(false) : srcType(srcInfo.GetType()),
        scaleShiftType(scaleShiftInfo.GetType()), dstType(diffDstInfo.GetType()), compType(Element::TypeT::F32),
        /* since all the arrays are broadcasted to 5D, we need to add extra dimensions which were broadcasted */
        axisNum(srcInfo.GetAxisNumAccordingToLayout((srcInfo.GetNumDims() - 1)) + MAX_DIMS - srcInfo.GetNumDims()),
        normFlags(flags)
    {
        Service::ThrowOnStatus(NormalizationLayerBWD::ValidateInput(
            srcInfo, statInfo, diffSrcInfo, diffDstInfo, scaleShiftInfo, diffSc, flags), "Normalization Layer");

        reorderedDims = Service::BroadcastTo5D(
            Service::GetShapeAccordingToLayout(srcInfo.GetDims(), srcInfo.GetLayout()));
        scaleShiftNumThreads = 1;
        maxNumThreads = 1;
        axisSize = reorderedDims[axisNum];
        outerSize = Service::ArrayProduct(reorderedDims, axisNum, 0);
        innerSize = Service::ArrayProduct(reorderedDims, reorderedDims.GetNumDims(), axisNum + 1);
        BatchedDims();

        statShape = statInfo.GetDims();
        dstShape = diffDstInfo.GetDims();
        srcLayout = srcInfo.GetLayout();
        statLayout = statInfo.GetLayout();
        dstLayout = diffDstInfo.GetLayout();

        Parallelizable::Initialize(GetOptNumThreads(), scaleShiftBatchedDims);
        if (static_cast<bool>(normFlags & NormalizationFlags::USE_SCALE) ||
            static_cast<bool>(normFlags & NormalizationFlags::USE_SHIFT)) {
            SetScaleShiftNumThreads();
        }
    }

    template <typename CompType, typename InType, typename OutType>
    void Evaluate(const Shape &batchSplitDims, const void *src, const float *mean, const float *variance,
        const void *diffDst, const void *scale, void *diffSrc, float eps) const noexcept(false)
    {
        const InType *typedSrc = static_cast<const InType *>(src);
        const OutType *typedDiffDst = static_cast<const OutType *>(diffDst);
        InType *typedDiffSrc = static_cast<InType *>(diffSrc);
        const OutType *typedScale = static_cast<const OutType *>(scale);
        ChooseImpl<CompType>(batchSplitDims, typedSrc, mean, variance, typedDiffDst,
            typedScale, typedDiffSrc, eps);
    }

    void RunSequential(const Shape &batchSplitDims, const void *src, const float *mean, const float *variance,
        const void *diffDst, const void *scale, void *diffSrc,
        float eps) const noexcept(false)
    {
        if (((srcType != dstType) || (scaleShiftType != dstType)) && (dstType != Element::TypeT::F32)) {
            Service::ParseType<false, false>(compType, Element::TypeT::F32, Element::TypeT::F32, *this,
                batchSplitDims, src, mean, variance, diffDst, scale, diffSrc, eps);
        } else {
            Service::ParseType<false, false>(compType, dstType, dstType, *this, batchSplitDims, src, mean, variance,
                diffDst, scale, diffSrc, eps);
        }
    }

    void RunSequential(const Shape &batchSplitDims, const Shape &batchStrides,
        const void *src, const float *mean, const float *variance, const void *diffDst,
        const void *scale, void *diffSrc, float eps) noexcept(false)
    {
        SizeType strideSrc = 0;
        SizeType strideDiffDst = 0;
        SizeType strideScale = 0;
        SizeType strideMean = 0;
        SizeType elementSize = dstType.GetSize();
        if ((srcType != dstType) || (scaleShiftType != dstType)) {
            elementSize = sizeof(float);
        }
        for (SizeType i = 0; i < batchStrides.GetNumDims(); ++i) {
            strideSrc += batchStrides[i] * elementSize * (axisSize * innerSize);
            strideDiffDst += batchStrides[i] * elementSize * (axisSize * innerSize);
            strideMean += batchStrides[i] * innerSize;
        }
        SizeType strideDiffSrc = strideSrc;
        SizeType strideVariance = strideMean;
        RunSequential(batchSplitDims, static_cast<const std::byte *>(src) + strideSrc,
            mean + strideMean, variance + strideVariance,
            static_cast<const std::byte *>(diffDst) + strideDiffDst,
            static_cast<const std::byte *>(scale) + strideScale,
            static_cast<std::byte *>(diffSrc) + strideDiffSrc,
            eps);
    }

    // separate computation of diffScale and diffShift
    template <typename CompType, typename InType, typename OutType>
    void Evaluate(const Shape &batchSplitDims, const void *src, const float *mean, const float *variance,
        const void *diffDst, void *diffScale, void *diffShift, float *bufScale, float *bufShift,
        float eps) const noexcept(false)
    {
        const InType *typedSrc = static_cast<const InType *>(src);
        const OutType *typedDiffDst = static_cast<const OutType *>(diffDst);
        OutType *typedDiffScale = static_cast<OutType *>(diffScale);
        OutType *typedDiffShift = static_cast<OutType *>(diffShift);
        if (axisNum == (reorderedDims.GetNumDims() - 1)) {
            LNorm::EvaluateScaleShiftLastAxis<CompType>(batchSplitDims.GetTotalDimsSize(), typedSrc, mean, variance,
                typedDiffDst, typedDiffScale, typedDiffShift, bufScale, bufShift, eps, normFlags,
                batchSplitDims.GetTotalDimsSize(), axisSize);
        } else {
            LNorm::EvaluateScaleShiftCrossAxis<CompType>(batchSplitDims.GetTotalDimsSize(), typedSrc, mean, variance,
                typedDiffDst, typedDiffScale, typedDiffShift, bufScale, bufShift, eps, normFlags,
                axisSize, innerSize);
        }
    }

    void RunSequential(const Shape &batchSplitDims, const void *src, const float *mean, const float *variance,
        const void *diffDst, void *diffScale, void *diffShift, float *bufScale, float *bufShift,
        float eps) const noexcept(false)
    {
        Service::Unused(bufScale, bufShift);
        if (((srcType != dstType) || (scaleShiftType != dstType)) && (dstType != Element::TypeT::F32)) {
            Service::ParseType<false, false>(compType, Element::TypeT::F32, Element::TypeT::F32, *this,
                batchSplitDims, src, mean, variance, diffDst, diffScale, diffShift, bufScale, bufShift, eps);
        } else {
            Service::ParseType<false, false>(compType, dstType, dstType, *this, batchSplitDims, src, mean, variance,
                diffDst, diffScale, diffShift, bufScale, bufShift, eps);
        }
    }

    void RunSequential(const Shape &batchSplitDims, const Shape &batchStrides, const SizeType threadIndex,
        const SizeType numThreads, const void *src, const float *mean, const float *variance, const void *diffDst,
        void *diffScale, void *diffShift, float *bufScale, float *bufShift, float eps) noexcept(false)
    {
        SizeType strideSrc = 0;
        SizeType strideDiffDst = 0;
        SizeType strideScale = 0;
        SizeType strideMean = 0;
        SizeType elementSize = dstType.GetSize();
        if ((srcType != dstType) || (scaleShiftType != dstType)) {
            elementSize = sizeof(float);
        }
        for (SizeType i = 0; i < batchStrides.GetNumDims(); ++i) {
            strideSrc += batchStrides[i] * elementSize * (axisSize * innerSize);
            strideDiffDst += batchStrides[i] * elementSize * (axisSize * innerSize);
            strideMean += batchStrides[i] * innerSize;
        }
        if (threadIndex == 0) {
            maxNumThreads = numThreads;
        }
        SizeType strideDiffScale = strideScale;
        SizeType strideDiffShift = strideScale;
        SizeType strideVariance = strideMean;

        RunSequential(batchSplitDims, static_cast<const std::byte *>(src) + strideSrc,
            mean + strideMean, variance + strideVariance,
            static_cast<const std::byte *>(diffDst) + strideDiffDst,
            static_cast<std::byte *>(diffScale) + strideDiffScale,
            static_cast<std::byte *>(diffShift) + strideDiffShift,
            bufScale + threadIndex * axisSize, bufShift + threadIndex * axisSize,
            eps);
    }

    // separate rewrite from buffers to diffScale & diffShift
    template <typename CompType, typename InType, typename OutType>
    void Evaluate(void *diffScale, void *diffShift,
        float *bufScale, float *bufShift, SizeType numThreads) const noexcept(false)
    {
        OutType *typedDiffScale = static_cast<OutType *>(diffScale);
        OutType *typedDiffShift = static_cast<OutType *>(diffShift);
        LNorm::EvaluateRewriteScaleShift<CompType, OutType, OutType, OutType>(typedDiffScale, typedDiffShift,
            bufScale, bufShift, normFlags, axisSize, numThreads);
    }

    void RunSequential(void *diffScale, void *diffShift,
        float *bufScale, float *bufShift, SizeType numThreads) const noexcept(false)
    {
        if (((srcType != dstType) || (scaleShiftType != dstType)) && (dstType != Element::TypeT::F32)) {
            Service::ParseType<false, false>(compType, Element::TypeT::F32, Element::TypeT::F32, *this, diffScale,
                diffShift, bufScale, bufShift, numThreads);
        } else {
            Service::ParseType<false, false>(compType, dstType, dstType, *this, diffScale, diffShift, bufScale,
                bufShift, numThreads);
        }
    }

    void Run(const void *src, const float *mean, const float *variance, const void *diffDst,
        const void *scale, void *diffSrc, void *diffScale, void *diffShift, float eps) noexcept(false)
    {
        if ((src == nullptr) || (mean == nullptr) || (variance == nullptr) || (diffDst == nullptr) ||
            (diffSrc == nullptr)) {
            throw Service::LogicError {"Normalization Layer: one of input matrices is nullptr"};
        }
        if (static_cast<bool>(normFlags & NormalizationFlags::USE_SCALE) &&
            ((scale == nullptr) || (diffScale == nullptr))) {
            throw Service::LogicError {"Normalization Layer: one of input matrices is nullptr"};
        }
        if ((static_cast<bool>(normFlags & NormalizationFlags::USE_SHIFT) && (scale == nullptr)) &&
            (diffShift == nullptr)) {
            throw Service::LogicError {"Normalization Layer: one of input matrices is nullptr"};
        }
        const void *srcExec = src;
        const void *scaleExec = scale;
        void *diffSrcExec = diffSrc;
        void *diffScaleExec = diffScale;
        void *diffShiftExec = diffShift;
        const float *meanExec = mean;
        const float *varianceExec = variance;
        const void *diffDstExec = diffDst;
        
        BeforeRunReorderLayer(&srcExec, &scaleExec, &diffSrcExec, &diffScaleExec, &diffShiftExec,
            &meanExec, &varianceExec, &diffDstExec, src, mean, variance, diffDst, scale, diffScale, diffShift);

        if ((diffScale != nullptr) || (diffShift != nullptr)) {
            SetScaleShiftThrParams(scaleShiftNumThreads);
            std::vector<float, Service::AlignedAllocator<float>> tmpDiffScale;
            std::vector<float, Service::AlignedAllocator<float>> tmpDiffShift;
            if (static_cast<bool>(normFlags & NormalizationFlags::USE_SCALE)) {
                tmpDiffScale.resize(axisSize * scaleShiftNumThreads, 0.0f);
            }
            if (static_cast<bool>(normFlags & NormalizationFlags::USE_SHIFT)) {
                tmpDiffShift.resize(axisSize * scaleShiftNumThreads, 0.0f);
            }
            Threading::BatchThreader(*this, srcExec, meanExec, varianceExec, diffDstExec,
                diffScaleExec, diffShiftExec, tmpDiffScale.data(), tmpDiffShift.data(), eps);
            RunSequential(diffScaleExec, diffShiftExec, tmpDiffScale.data(), tmpDiffShift.data(), maxNumThreads);
        }
        SetMainThrParams(GetOptNumThreads());
        Threading::BatchThreader(*this, srcExec, meanExec, varianceExec, diffDstExec, scaleExec,
            diffSrcExec, eps);

        AfterRunReorderLayer(&srcExec, &scaleExec, &diffSrcExec, &diffScaleExec, &diffShiftExec,
            &meanExec, &varianceExec, &diffDstExec, diffSrc, mean, variance, scale, diffScale, diffShift);
    }

private:
    Element::Type srcType;
    Element::Type scaleShiftType;
    Element::Type dstType;
    Element::Type compType;
    Shape mainBatchedDims;
    Shape scaleShiftBatchedDims;
    SizeType axisNum;
    NormalizationFlags normFlags;
    SizeType axisSize;
    SizeType innerSize;
    SizeType outerSize;
    Shape reorderedDims;
    SizeType scaleShiftNumThreads;
    SizeType maxNumThreads;
    Shape statShape;
    Shape dstShape;
    Layout srcLayout;
    Layout statLayout;
    Layout dstLayout;

    void BeforeRunReorderLayer(const void **srcExec, const void **scaleExec, void **diffSrcExec,
        void **diffScaleExec, void **diffShiftExec, const float **meanExec, const float **varianceExec,
        const void **diffDstExec, const void *src, const float *mean, const float *variance,
        const void *diffDst, const void *scale, void *diffScale, void *diffShift) noexcept(false)
    {
        if ((srcType != dstType) || (scaleShiftType != dstType)) {
            void *tmp = Service::AlignedAlloc(reorderedDims.GetTotalDimsSize() * sizeof(float));
            ReorderLayer r({
                reorderedDims, srcType, Layout::ABCDE}, {
                reorderedDims, Element::TypeT::F32, Layout::ABCDE});
            r.Run(src, tmp);
            *srcExec = tmp;
        }
        if (((srcType != dstType) || (scaleShiftType != dstType)) && scale) {
            void *tmp = Service::AlignedAlloc(reorderedDims[axisNum] * sizeof(float));
            ReorderLayer r({{axisSize}, scaleShiftType, Layout::A}, {{axisSize}, Element::TypeT::F32, Layout::A});
            r.Run(scale, tmp);
            *scaleExec = tmp;
        }

        if ((srcType != dstType) || (scaleShiftType != dstType)) {
            *diffSrcExec = Service::AlignedAlloc(reorderedDims.GetTotalDimsSize() * sizeof(float));

            if (diffScale != nullptr) {
                *diffScaleExec = Service::AlignedAlloc(axisSize * sizeof(float));
            }
            if (diffShift != nullptr) {
                *diffShiftExec = Service::AlignedAlloc(axisSize * sizeof(float));
            }
        }
        if (statLayout != GetSrcLayoutWithoutLastDim(srcLayout)) {
            ReorderLayer r({
                statShape, Element::TypeT::F32, statLayout}, {
                statShape, Element::TypeT::F32, GetSrcLayoutWithoutLastDim(srcLayout)});
            void *tmp;
            if (mean != nullptr) {
                tmp = Service::AlignedAlloc(statShape.GetTotalDimsSize() * sizeof(float));
                r.Run(mean, tmp);
                *meanExec = static_cast<const float *>(tmp);
            }
            if (variance != nullptr) {
                tmp = Service::AlignedAlloc(statShape.GetTotalDimsSize() * sizeof(float));
                r.Run(variance, tmp);
                *varianceExec = static_cast<const float *>(tmp);
            }
        }
        if ((dstLayout != srcLayout) ||
            (((srcType != dstType) || (scaleShiftType != dstType)) && (dstType != Element::TypeT::F32))) {
            bool dt = (((srcType != dstType) || (scaleShiftType != dstType)) && (dstType != Element::TypeT::F32));
            void *tmp = Service::AlignedAlloc(
                reorderedDims.GetTotalDimsSize() * (dt ? sizeof(float) : dstType.GetSize()));
            ReorderLayer r({dstShape, dstType, dstLayout}, {dstShape,
                dt ? Element::Type{Element::TypeT::F32} : dstType, srcLayout});
            r.Run(diffDst, tmp);
            *diffDstExec = tmp;
        }
    }

    void AfterRunReorderLayer(const void **srcExec, const void **scaleExec, void **diffSrcExec,
        void **diffScaleExec, void **diffShiftExec, const float **meanExec, const float **varianceExec,
        const void **diffDstExec, void *diffSrc, const float *mean, const float *variance,
        const void *scale, void *diffScale, void *diffShift) noexcept(false)
    {
        if ((dstLayout != srcLayout) ||
            (((srcType != dstType) || (scaleShiftType != dstType)) && (dstType != Element::TypeT::F32))) {
            Service::Deallocate(const_cast<void *>(*diffDstExec));
        }
        if ((srcType != dstType) || (scaleShiftType != dstType)) {
            ReorderLayer r({
                reorderedDims, Element::TypeT::F32, Layout::ABCDE}, {
                reorderedDims, srcType, Layout::ABCDE});
            r.Run(*diffSrcExec, diffSrc);
            Service::Deallocate(*diffSrcExec);
        }
        if ((srcType != dstType) || (scaleShiftType != dstType)) {
            ReorderLayer r({{axisSize}, Element::TypeT::F32, Layout::A}, {{axisSize}, scaleShiftType, Layout::A});
            if (diffScale != nullptr) {
                r.Run(*diffScaleExec, diffScale);
                Service::Deallocate(*diffScaleExec);
            }
            if (diffShift != nullptr) {
                r.Run(*diffShiftExec, diffShift);
                Service::Deallocate(*diffShiftExec);
            }
            if (scale != nullptr) {
                Service::Deallocate(const_cast<void *>(*scaleExec));
            }
        }
        if (statLayout != GetSrcLayoutWithoutLastDim(srcLayout)) {
            if (mean != nullptr) {
                Service::Deallocate(const_cast<float *>(*meanExec));
            }
            if (variance != nullptr) {
                Service::Deallocate(const_cast<float *>(*varianceExec));
            }
        }
        if ((srcType != dstType) || (scaleShiftType != dstType)) {
            Service::Deallocate(const_cast<void *>(*srcExec));
        }
    }

    SizeType GetOptNumThreads() const noexcept
    {
        SizeType totalSize = reorderedDims.GetTotalDimsSize();
        if (totalSize < LNORM_BWD_ELE_NUM_1) {
            return LNORM_THREADS_BWD_THREASHOLD;
        } else if (totalSize <= LNORM_BWD_ELE_NUM_2) {
            return LNORM_BWD_THREADS1_C0 + LNORM_BWD_THREADS1_C1 * totalSize;
        } else {
            return LNORM_BWD_THREADS2_C0 + LNORM_BWD_THREADS2_C1 * std::log(totalSize);
        }
    }

    void SetScaleShiftNumThreads() noexcept
    {
        scaleShiftNumThreads = GetOptNumThreads(); // 4 means the default optimal threadNum
    }

    void BatchedDims() noexcept
    {
        if (axisNum == 0) {
            mainBatchedDims = {1};
            scaleShiftBatchedDims = {1};
        } else {
            std::array<SizeType, MAX_DIMS> batchedDIms;
            for (SizeType i = 0; i < axisNum; ++i) {
                batchedDIms[i] = reorderedDims[i];
            }
            mainBatchedDims = {Shape{batchedDIms.data(), axisNum}.GetTotalDimsSize()};
            scaleShiftBatchedDims = {Shape{batchedDIms.data(), axisNum}.GetTotalDimsSize()};
        }
    }

    void SetScaleShiftThrParams(SizeType threads) noexcept
    {
        ResetDimsThreads(scaleShiftBatchedDims, threads);
    }

    void SetMainThrParams(SizeType threads) noexcept
    {
        ResetDimsThreads(mainBatchedDims, threads);
    }

    template <typename CompType, typename InType, typename OutType, typename ScaleShiftType>
    void ChooseImpl(const Shape &batchSplitDims, const InType *src, const float *mean, const float *variance,
        const OutType *diffDst, const ScaleShiftType *scale, InType *diffSrc, float eps) const noexcept(false)
    {
        SizeType externalSize = batchSplitDims.GetTotalDimsSize();
        if (axisNum == (reorderedDims.GetNumDims() - 1)) {
            LNorm::EvaluateLastAxis<CompType>(externalSize, src, mean, variance, diffDst,
                scale, diffSrc, eps, normFlags, axisSize);
        } else {
            LNorm::EvaluateCrossAxis<CompType>(externalSize, src, mean, variance, diffDst,
                scale, diffSrc, eps, normFlags, axisSize, innerSize);
        }
    }
};

} // Detail

NormalizationLayerBWD::~NormalizationLayerBWD() noexcept {}

NormalizationLayerBWD::NormalizationLayerBWD(const TensorInfo &srcInfo, const TensorInfo &statInfo,
    const TensorInfo &diffSrcInfo, const TensorInfo &diffDstInfo,
    const TensorInfo &scaleShiftInfo, const TensorInfo &diffscaleShiftInfo,
    NormalizationFlags flags) noexcept(false)
    : pImpl(new Detail::NormalizationLayerBWDImpl(srcInfo, statInfo, diffSrcInfo, diffDstInfo, scaleShiftInfo,
    diffscaleShiftInfo, flags)) {}

NormalizationLayerBWD::NormalizationLayerBWD(const NormalizationLayerBWD &other) noexcept(false)
    : pImpl(new Detail::NormalizationLayerBWDImpl{*(other.pImpl.get())}) {}

NormalizationLayerBWD::NormalizationLayerBWD(NormalizationLayerBWD &&other) noexcept
    : pImpl(std::move(other.pImpl)) {}

NormalizationLayerBWD& NormalizationLayerBWD::operator=(const NormalizationLayerBWD &other) noexcept(false)
{
    if (this != &other) {
        this->pImpl.reset(new Detail::NormalizationLayerBWDImpl{*(other.pImpl.get())});
    }
    return *this;
}

NormalizationLayerBWD& NormalizationLayerBWD::operator=(NormalizationLayerBWD &&other) noexcept
{
    if (this != &other) {
        this->pImpl = std::move(other.pImpl);
    }
    return *this;
}

void NormalizationLayerBWD::Run(const void *src, const float *mean, const float *variance, const void *diffDst,
    const void *scale, void *diffSrc, void *diffScale, void *diffShift, float eps) const noexcept(false)
{
    pImpl->Run(src, mean, variance, diffDst, scale, diffSrc, diffScale, diffShift, eps);
}

Status NormalizationLayerBWD::ValidateInput(const TensorInfo &srcInfo, const TensorInfo &statInfo,
    const TensorInfo &diffSrcInfo, const TensorInfo &diffDstInfo,
    const TensorInfo &scaleShiftInfo, const TensorInfo &diffscaleShiftInfo,
    NormalizationFlags flags) noexcept
{
    if (flags == NormalizationFlags::FUSE_NORM_RELU) {
        return Status::UNSUPPORTED;
    }
    if (srcInfo != diffSrcInfo) {
        return Status::BAD_ARGUMENTS;
    }
    if (statInfo.GetType() != Element::TypeT::F32) {
        return Status::UNSUPPORTED;
    }
    
    bool scaleShiftTypeSupported = ((static_cast<bool>(flags & NormalizationFlags::USE_SCALE)) ||
        (static_cast<bool>(flags & NormalizationFlags::USE_SHIFT))) &&
        (scaleShiftInfo.GetType() != Element::TypeT::F32) &&
        (scaleShiftInfo.GetType() != Element::TypeT::F16) &&
        (scaleShiftInfo.GetType() != Element::TypeT::BF16);
    if (scaleShiftTypeSupported) {
        return Status::UNSUPPORTED;
    }
    if (srcInfo.GetDims() != diffDstInfo.GetDims()) {
        return Status::UNSUPPORTED;
    }
    if (scaleShiftInfo != diffscaleShiftInfo) {
        return Status::BAD_ARGUMENTS;
    }
    if (srcInfo.GetTotalTensorSize() == 0) {
        return Status::UNSUPPORTED;
    }
    if ((srcInfo.GetNumDims() - 1) != statInfo.GetNumDims()) {
        return Status::BAD_ARGUMENTS;
    }
    if (scaleShiftInfo.GetNumDims() != 1) {
        return Status::BAD_ARGUMENTS;
    }
    if (srcInfo.GetDims()[srcInfo.GetNumDims() - 1] != scaleShiftInfo.GetDims()[0]) {
        return Status::BAD_ARGUMENTS;
    }
    for (SizeType i = 0; i < statInfo.GetNumDims(); ++i) {
        if (srcInfo.GetDims()[i] != statInfo.GetDims()[i]) {
            return Status::BAD_ARGUMENTS;
        }
    }
    if (!srcInfo.IsDense() || !statInfo.IsDense() || !diffDstInfo.IsDense() ||
        !scaleShiftInfo.IsDense() || !diffscaleShiftInfo.IsDense()) {
        return Status::UNSUPPORTED;
    }
    return Status::SUCCESS;
}

} // KuDNN
