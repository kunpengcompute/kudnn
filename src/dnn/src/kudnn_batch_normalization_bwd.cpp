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

class BatchNormalizationLayerBWDImpl final : public Parallelizable {
public:
    BatchNormalizationLayerBWDImpl(const Propagation &propKind,
        const TensorInfo &srcInfo, const TensorInfo &diffDstInfo,
        const TensorInfo &statsInfo, const TensorInfo &scaleShiftInfo, const TensorInfo &diffSrcInfo,
        const TensorInfo &diffScaleShiftInfo, NormalizationFlags flags) noexcept(false)
        : srcType(srcInfo.GetType()), compType(Element::TypeT::F32), normFlags(flags),
          thisSrcInfo(srcInfo), thisDiffDstInfo(diffDstInfo), thisDiffSrcInfo(diffSrcInfo),
          channelAxisNum(srcInfo.GetAxisNumAccordingToLayout(1) + MAX_DIMS - srcInfo.GetNumDims()),
          useRrorderedAxb((srcInfo.GetLayout() == srcInfo.GetStandardAXBLayout()) &&
                          (!static_cast<bool>(normFlags & NormalizationFlags::FUSE_NORM_RELU)) &&
                          (srcInfo.GetNumDims() >= DIM_3 && srcInfo.GetNumDims() <= DIM_5))
    {
        // threading implementation depends on the value of calculateStats flag
        // so we initialize it at the constructor stage
        Service::ThrowOnStatus(BatchNormalizationLayerBWD::ValidateInput(propKind, srcInfo, diffDstInfo,
            statsInfo, scaleShiftInfo, diffSrcInfo, diffScaleShiftInfo, flags), "Batch Normalization Layer BWD");
        reorderedDims = Service::BroadcastTo5D(
            Service::GetShapeAccordingToLayout(srcInfo.GetDims(), srcInfo.GetLayout()));
        reorderedStridesSrc = Service::BroadcastTo5D(
            Service::GetShapeAccordingToLayout(srcInfo.GetStrides(), srcInfo.GetLayout()));
        reorderedStridesDiffSrc = Service::BroadcastTo5D(
            Service::GetShapeAccordingToLayout(diffSrcInfo.GetStrides(), diffSrcInfo.GetLayout()));
        SetSizeParas(reorderedDims, channelAxisNum);

        if (useRrorderedAxb) {
            PrepareAxbReorder();
        }

        InitBatchedDims();
        Parallelizable::Initialize(GetOptNumThreads(), statBatchedDims);
        if (!static_cast<bool>(normFlags & NormalizationFlags::USE_SCALE)) {
            tmpDiffScale.resize(channelSize, 0.0f);
        }
        if (!static_cast<bool>(normFlags & NormalizationFlags::USE_SHIFT)) {
            tmpDiffShift.resize(channelSize, 0.0f);
        }
    };

    // normalization
    template <typename CompType, typename InType, typename OutType>
    void Evaluate(const Shape &batchSplitDims, const void *src, const void *diffDst,
        const float *mean, const float *variance, const float *scale, void *diffSrc,
        float *diffScale, float *diffShift, float eps, const void *ws) const noexcept
    {
        const InType *typedSrc = static_cast<const InType *>(src);
        const OutType *typedDiffDst = static_cast<const OutType *>(diffDst);
        InType *typedDiffSrc = static_cast<InType *>(diffSrc);
        const std::uint8_t *typedWs = static_cast<const std::uint8_t *>(ws);

        SizeType axisSize = batchSplitDims.GetTotalDimsSize();
        BNorm::EvaluateCrossAxisBwd<float>(axisSize, typedSrc, typedDiffDst, mean, variance,
            scale, typedDiffSrc, diffScale, diffShift, eps, normFlags, outerSize,
            channelSize, innerSize, typedWs);
    }

    void RunSequential(const Shape &batchSplitDims, const void *src, const void *diffDst,
        const float *mean, const float *variance, const float *scale, void *diffSrc,
        float *diffScale, float *diffShift, float eps, const void *ws) const noexcept(false)
    {
        Service::ParseType<false, false>(compType, srcType, srcType, *this, batchSplitDims, src, diffDst,
            mean, variance, scale, diffSrc, diffScale, diffShift, eps, ws);
    }

    void RunSequential(const Shape &batchSplitDims, const Shape &batchStrides,
            const void *src, const void *diffDst, const float *mean, const float *variance,
            const float *scale, void *diffSrc, float *diffScale, float *diffShift, float eps,
            const void *ws) const noexcept(false)
    {
        SizeType strideSrc = 0;
        for (SizeType i = 0; i < batchStrides.GetNumDims(); ++i) {
            strideSrc += batchStrides[i] * reorderedStridesSrc[i] * srcType.GetSize();
        }
        SizeType strideWs = strideSrc / srcType.GetSize(); // Ws element size is 1 byte
        RunSequential(batchSplitDims, static_cast<const std::byte *>(src) + strideSrc,
            static_cast<const std::byte *>(diffDst) + strideSrc, mean, variance, scale,
            static_cast<std::byte *>(diffSrc) + strideSrc, diffScale, diffShift, eps,
            (ws == nullptr) ? nullptr : (static_cast<const std::byte *>(ws) + strideWs));
    }

    // separate computation of Mean and Variance
    template <typename CompType, typename InType, typename OutType>
    void Evaluate(const Shape &batchSplitDims, const void *src, const void *diffDst,
        const float *mean, const float *variance, float *diffScale,
        float *diffShift, float eps, const void *ws) const noexcept
    {
        const InType *typedSrc = static_cast<const InType *>(src);
        const OutType *typedDiffDst = static_cast<const OutType *>(diffDst);
        const std::uint8_t *typedWs = static_cast<const std::uint8_t *>(ws);
        BNorm::EvaluateStatsCrossAxisBwd<float>(outerSize, typedSrc, typedDiffDst, mean, variance,
            diffScale, diffShift, eps, batchSplitDims.GetTotalDimsSize(), channelSize, innerSize, typedWs);
    }

    void RunSequential(const Shape &batchSplitDims, const void *src, const void *diffDst,
        const float *mean, const float *variance, float *diffScale, float *diffShift, float eps,
        const void *ws) noexcept(false)
    {
        Service::ParseType<false, false>(compType, srcType, srcType, *this, batchSplitDims, src, diffDst,
            mean, variance, diffScale, diffShift, eps, ws);
    }

    void RunSequential(const Shape &batchSplitDims, const Shape &batchStrides,
        const void *src, const void *diffDst, const float *mean, const float *variance,
        float *diffScale, float *diffShift, float eps, const void *ws) noexcept(false)
    {
        SizeType strideSrc = 0;
        SizeType strideStat = 0;
        for (SizeType i = 0; i < batchStrides.GetNumDims(); ++i) {
            strideSrc += batchStrides[i] * srcType.GetSize() * innerSize;
            strideStat += batchStrides[i];
        }
        SizeType strideWs = strideSrc / srcType.GetSize(); // Ws element size is 1 byte
        RunSequential(batchSplitDims, static_cast<const std::byte *>(src) + strideSrc,
            static_cast<const std::byte *>(diffDst) + strideSrc, mean + strideStat,
            variance + strideStat, diffScale + strideStat, diffShift + strideStat, eps,
            (ws == nullptr) ? nullptr : (static_cast<const std::byte *>(ws) + strideWs));
    }

    void Run(const void *src, const void *diffDst, const float *mean, const float *variance, const float *scale,
        void *diffSrc, float *diffScale, float *diffShift, const float eps, const void *ws) noexcept(false)
    {
        if ((src == nullptr) || (diffDst == nullptr) || (diffSrc == nullptr)
            || (mean == nullptr) || (variance == nullptr)) {
            throw Service::LogicError {"Batch Normalization: one of input matrices is nullptr"};
        }
        if (static_cast<bool>(normFlags & NormalizationFlags::USE_SCALE) &&
            ((scale == nullptr) || (diffScale == nullptr))) {
            throw Service::LogicError {"Batch Normalization: one of input matrices is nullptr"};
        }
        if (static_cast<bool>(normFlags & NormalizationFlags::FUSE_NORM_RELU) && (ws == nullptr)) {
            throw Service::LogicError {"Batch Normalization: one of input matrices is nullptr"};
        }
        if (static_cast<bool>(normFlags & NormalizationFlags::USE_SHIFT) &&
            (diffShift == nullptr)) {
            throw Service::LogicError {"Batch Normalization: one of input matrices is nullptr"};
        }

        if (!static_cast<bool>(normFlags & NormalizationFlags::USE_SCALE)) {
            diffScale = tmpDiffScale.data();
        }
        if (!static_cast<bool>(normFlags & NormalizationFlags::USE_SHIFT)) {
            diffShift = tmpDiffShift.data();
        }

        if (useRrorderedAxb) {
            RunAxbWithReorder(src, diffDst, mean, variance, scale, diffSrc, diffScale, diffShift, eps, ws);
        } else {
            if (!static_cast<bool>(normFlags & NormalizationFlags::USE_GLOBAL_STATS)) {
                SetStatsThrParams(GetOptNumThreads());
                Threading::BatchThreader(*this, src, diffDst, mean, variance, diffScale, diffShift, eps, ws);
            }
            SetMainThrParams(GetOptNumThreads());
            Threading::BatchThreader(*this,
                src, diffDst, mean, variance, scale, diffSrc, diffScale, diffShift, eps, ws);
        }
    }

private:
    Element::Type srcType;
    Element::Type compType;
    NormalizationFlags normFlags;
    Shape reorderedDims;
    Shape reorderedStridesSrc;
    Shape reorderedStridesDiffSrc;
    Shape mainBatchedDims;
    Shape statBatchedDims;
    TensorInfo thisSrcInfo;
    TensorInfo thisDiffDstInfo;
    TensorInfo thisDiffSrcInfo;
    std::vector<float, Service::AlignedAllocator<float>> tmpDiffScale;
    std::vector<float, Service::AlignedAllocator<float>> tmpDiffShift;
    SizeType channelAxisNum;
    SizeType channelSize;
    SizeType innerSize;
    SizeType outerSize;
    bool useRrorderedAxb;
    // threads acount is control within in single numa, 920b numa is 40 core, 920f numa is 38 core.
    const SizeType optNumThreads = 40;

    void InitBatchedDims() noexcept
    {
        mainBatchedDims = {1};
        if (channelAxisNum != 0) {
            std::array<SizeType, MAX_DIMS> tmpBD;
            for (SizeType i = 0; i < MAX_DIMS; ++i) {
                tmpBD[i] = reorderedDims[i];
            }
            mainBatchedDims = {tmpBD.data(), channelAxisNum};
        }
        statBatchedDims = {channelSize};
    }

    void SetMainThrParams(SizeType threads) noexcept
    {
        ResetDimsThreads(mainBatchedDims, threads);
    }

    void SetStatsThrParams(SizeType threads) noexcept
    {
        ResetDimsThreads(statBatchedDims, threads);
    }

    SizeType GetOptNumThreads() const noexcept
    {
        return optNumThreads;
    }

    void SetSizeParas(Shape &dims, SizeType ch)
    {
        channelSize = dims[ch];
        innerSize = Service::ArrayProduct(dims, dims.GetNumDims(), ch + 1);
        outerSize = Service::ArrayProduct(dims, ch, 0);
    }

    void PrepareAxbReorder()
    {
        innerSize = outerSize / reorderedDims[MAX_DIMS - thisSrcInfo.GetNumDims()];
        outerSize /= innerSize;
        Shape reorderedDimsAxb{1, 1, 1, 1, 1};
        if (thisSrcInfo.GetNumDims() == IDX_3) {
            reorderedDimsAxb =
                Shape{reorderedDims[0], reorderedDims[1], reorderedDims[2], reorderedDims[4], reorderedDims[3]};
            channelAxisNum = IDX_3;
        } else if (thisSrcInfo.GetNumDims() == IDX_4) {
            reorderedDimsAxb =
                Shape{reorderedDims[0], reorderedDims[1], reorderedDims[4], reorderedDims[2], reorderedDims[3]};
            channelAxisNum = IDX_2;
        } else if (thisSrcInfo.GetNumDims() == IDX_5) {
            reorderedDimsAxb =
                Shape{reorderedDims[0], reorderedDims[4], reorderedDims[1], reorderedDims[2], reorderedDims[3]};
            channelAxisNum = IDX_1;
        }
        reorderedDims = reorderedDimsAxb;
        reorderedStridesSrc = ReorderStridesShape(reorderedStridesSrc);
        reorderedStridesDiffSrc = ReorderStridesShape(reorderedStridesDiffSrc);
    }

    Shape ReorderStridesShape(Shape &shape)
    {
        std::array<SizeType, MAX_DIMS> dimsArr = {1, 1, 1, 1, 1};
        for (SizeType idx = IDX_3; idx > MAX_DIMS - thisSrcInfo.GetNumDims(); idx--) {
            dimsArr[idx] = shape[idx - 1] / shape[idx];
            dimsArr[idx] *= dimsArr[idx + 1];
        }
        dimsArr[MAX_DIMS - thisSrcInfo.GetNumDims()] = shape[MAX_DIMS - thisSrcInfo.GetNumDims()];
        return Shape{dimsArr[0], dimsArr[1], dimsArr[2], dimsArr[3], dimsArr[4]};
    }

    void RunAxbWithReorder(const void *src, const void *diffDst, const float *mean, const float *variance,
        const float *scale, void *diffSrc, float *diffScale, float *diffShift, const float eps, const void *ws)
    {
        void *execSrc = Service::AlignedAlloc(thisSrcInfo.GetTotalTensorSize() * srcType.GetSize());
        ReorderLayer rSrc(
            TensorInfo{thisSrcInfo.GetDims(), srcType, thisSrcInfo.GetLayout()},
            TensorInfo{thisSrcInfo.GetDims(), srcType, thisSrcInfo.GetStandardABXLayout()});
        rSrc.Run(src, execSrc);
        void *execDiffDst = Service::AlignedAlloc(thisDiffDstInfo.GetTotalTensorSize() * srcType.GetSize());
        ReorderLayer rDiffDst(
            TensorInfo{thisDiffDstInfo.GetDims(), srcType, thisDiffDstInfo.GetLayout()},
            TensorInfo{thisDiffDstInfo.GetDims(), srcType, thisDiffDstInfo.GetStandardABXLayout()});
        rDiffDst.Run(diffDst, execDiffDst);

        void *execDiffSrc = Service::AlignedAlloc(thisDiffSrcInfo.GetTotalTensorSize() * srcType.GetSize());
        
        if (!static_cast<bool>(normFlags & NormalizationFlags::USE_GLOBAL_STATS)) {
            SetStatsThrParams(GetOptNumThreads());
            Threading::BatchThreader(*this, execSrc, execDiffDst, mean, variance, diffScale, diffShift, eps, ws);
        }
        SetMainThrParams(GetOptNumThreads());
        Threading::BatchThreader(*this,
            execSrc, execDiffDst, mean, variance, scale, execDiffSrc, diffScale, diffShift, eps, ws);

        ReorderLayer rDiffSrc(
            TensorInfo{thisDiffSrcInfo.GetDims(), srcType, thisDiffSrcInfo.GetStandardABXLayout()},
            TensorInfo{thisDiffSrcInfo.GetDims(), srcType, thisDiffSrcInfo.GetLayout()});
        rDiffSrc.Run(execDiffSrc, diffSrc);

        Service::Deallocate(execSrc);
        Service::Deallocate(execDiffDst);
        Service::Deallocate(execDiffSrc);
    }
};

} // namespace Detail

BatchNormalizationLayerBWD::~BatchNormalizationLayerBWD() noexcept {}

BatchNormalizationLayerBWD::BatchNormalizationLayerBWD(const Propagation &propKind,
    const TensorInfo &srcInfo, const TensorInfo &diffDstInfo,
    const TensorInfo &statsInfo, const TensorInfo &scaleShiftInfo, const TensorInfo &diffSrcInfo,
    const TensorInfo &diffScaleShiftInfo, NormalizationFlags flags) noexcept(false)
    : pImpl(new Detail::BatchNormalizationLayerBWDImpl(propKind, srcInfo, diffDstInfo, statsInfo, scaleShiftInfo,
    diffSrcInfo, diffScaleShiftInfo, flags)) {}

BatchNormalizationLayerBWD::BatchNormalizationLayerBWD(const BatchNormalizationLayerBWD &other) noexcept(false)
    : pImpl(new Detail::BatchNormalizationLayerBWDImpl{*(other.pImpl.get())}) {}

BatchNormalizationLayerBWD::BatchNormalizationLayerBWD(BatchNormalizationLayerBWD &&other) noexcept
    : pImpl(std::move(other.pImpl)) {}

BatchNormalizationLayerBWD& BatchNormalizationLayerBWD::operator=(const BatchNormalizationLayerBWD &other)
    noexcept(false)
{
    if (this != &other) {
        this->pImpl.reset(new Detail::BatchNormalizationLayerBWDImpl{*(other.pImpl.get())});
    }
    return *this;
}

BatchNormalizationLayerBWD& BatchNormalizationLayerBWD::operator=(BatchNormalizationLayerBWD &&other) noexcept
{
    if (this != &other) {
        this->pImpl = std::move(other.pImpl);
    }
    return *this;
}

void BatchNormalizationLayerBWD::Run(const void *src, const void *diffDst, const float *mean,
    const float *variance, const float *scale, void *diffSrc, float *diffScale, float *diffShift,
    const float eps, const void *ws) const noexcept(false)
{
    pImpl->Run(src, diffDst, mean, variance, scale, diffSrc, diffScale, diffShift, eps, ws);
}

Status BatchNormalizationLayerBWD::ValidateInput(const Propagation &propKind,
    const TensorInfo &srcInfo, const TensorInfo &diffDstInfo,
    const TensorInfo &statsInfo, const TensorInfo &scaleShiftInfo, const TensorInfo &diffSrcInfo,
    const TensorInfo &diffScaleShiftInfo, NormalizationFlags flags) noexcept
{
    Service::Unused(flags);
    bool isBadArguments = (srcInfo != diffDstInfo) || (srcInfo != diffSrcInfo) ||
        ((propKind != Propagation::BACKWARD_DATA) && (propKind != Propagation::BACKWARD));
    if (isBadArguments) {
        return Status::BAD_ARGUMENTS;
    }
    if (scaleShiftInfo != diffScaleShiftInfo) {
        return Status::UNSUPPORTED;
    }
    isBadArguments = (statsInfo.GetType() != Element::TypeT::F32) || (scaleShiftInfo.GetType() != Element::TypeT::F32);
    if (isBadArguments) {
        return Status::BAD_ARGUMENTS;
    }
    bool isUnsupported = (propKind == Propagation::BACKWARD) || (srcInfo.GetTotalTensorSize() == 0) ||
        (!Service::OneOf(srcInfo.GetType(), Element::TypeT::F32, Element::TypeT::F16, Element::TypeT::BF16));
    if (isUnsupported) {
        return Status::UNSUPPORTED;
    }
    isBadArguments = (scaleShiftInfo.GetNumDims() != 1) || (statsInfo.GetNumDims() != 1);
    if (isBadArguments) {
        return Status::BAD_ARGUMENTS;
    }
    if (!srcInfo.IsDense() || !diffDstInfo.IsDense() || !statsInfo.IsDense() || !scaleShiftInfo.IsDense()
        || !diffSrcInfo.IsDense() || !diffScaleShiftInfo.IsDense()) {
        return Status::UNSUPPORTED;
    }
    return Status::SUCCESS;
}

} // namespace KuDNN
