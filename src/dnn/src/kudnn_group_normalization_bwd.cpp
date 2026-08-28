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
#include "kernels/gnorm/kudnn_gnorm_kernels.hpp"

namespace KuDNN {
namespace Detail {

class GroupNormalizationLayerBWDImpl final : public Parallelizable {
public:
    GroupNormalizationLayerBWDImpl(const TensorInfo &srcInfo, const TensorInfo &statInfo,
        const TensorInfo &diffSrcInfo, const TensorInfo &diffDstInfo,
        const TensorInfo &scaleShiftInfo, const TensorInfo &diffSc,
        NormalizationFlags flags) noexcept(false) : srcType(srcInfo.GetType()),
        scaleShiftType(scaleShiftInfo.GetType()), dstType(diffDstInfo.GetType()), compType(Element::TypeT::F32),
        finalDimIdx(srcInfo.GetAxisNumAccordingToLayout((srcInfo.GetNumDims() - 1)) + MAX_DIMS - srcInfo.GetNumDims()),
        numDims(srcInfo.GetNumDims()), normFlags(flags)
    {
        Service::ThrowOnStatus(GroupNormalizationLayerBWD::ValidateInput(
            srcInfo, statInfo, diffSrcInfo, diffDstInfo, scaleShiftInfo, diffSc, flags), "GroupNormalization Layer");
        reorderedDims = Service::BroadcastTo5D(Service::GetShapeAccordingToLayout(srcInfo.GetDims(),
            srcInfo.GetLayout()));

        needToConvert = (srcType != dstType) ||
                        (scaleShiftType != dstType) ||
                        (finalDimIdx != (reorderedDims.GetNumDims() - 1) &&
                        (static_cast<bool>(flags & NormalizationFlags::USE_SCALE) ||
                        static_cast<bool>(flags & NormalizationFlags::USE_SHIFT) ||
                        !(static_cast<bool>(flags & NormalizationFlags::USE_GLOBAL_STATS))));
        if (needToConvert) {
            scaleShiftType = Element::TypeT::F32;
        }

        channelSize = srcInfo.GetDims()[IDX_1];
        batchNum = srcInfo.GetDims()[IDX_0];
        groupNum = statInfo.GetDims()[IDX_1];
        channelPerGroup = channelSize / groupNum;
        eltPerGroup = channelPerGroup;
        for (SizeType i = IDX_2; i < srcInfo.GetNumDims(); ++i) {
            eltPerGroup *= srcInfo.GetDims()[i];
        }
        batchSize = eltPerGroup * groupNum;

        Parallelizable::Initialize(GetOptNumThreads(), reorderedDims);
    }

    template <typename CompType, typename InType, typename OutType>
    void Evaluate(const Shape &batchSplitDims, const void *src, const float *mean, const float *variance,
        const void *diffDst, const void *scale, void *diffSrc, float *diffScale, float *diffShift,
        float eps) const noexcept(false)
    {
        const InType *typedSrc = static_cast<const InType *>(src);
        const OutType *typedDiffDst = static_cast<const OutType *>(diffDst);
        InType *typedDiffSrc = static_cast<InType *>(diffSrc);
        const OutType *typedScale = static_cast<const OutType *>(scale);
        ChooseImpl<CompType>(batchSplitDims, typedSrc, mean, variance, typedDiffDst,
            typedScale, typedDiffSrc, diffScale, diffShift, eps);
    }

    void RunSequential(const Shape &batchSplitDims, const void *src, const float *mean, const float *variance,
        const void *diffDst, const void *scale, void *diffSrc, float *diffScale, float *diffShift,
        float eps) const noexcept(false)
    {
        if (((srcType != dstType) || (scaleShiftType != dstType)) && (dstType != Element::TypeT::F32)) {
            Service::ParseType<false, false>(compType, Element::TypeT::F32, Element::TypeT::F32, *this,
                batchSplitDims, src, mean, variance, diffDst, scale, diffSrc, diffScale, diffShift, eps);
        } else {
            Service::ParseType<false, false>(compType, dstType, dstType, *this, batchSplitDims, src, mean, variance,
                diffDst, scale, diffSrc, diffScale, diffShift, eps);
        }
    }

    void RunSequential(const Shape &batchSplitDims, const Shape &batchStrides,
        const void *src, const float *mean, const float *variance, const void *diffDst,
        const void *scale, void *diffSrc, float *diffScale, float *diffShift, float eps) noexcept(false)
    {
        SizeType strideSrc = 0;
        SizeType strideDiffDst = 0;
        SizeType strideMean = 0;
        SizeType elementSize = dstType.GetSize();
        if (needToConvert) {
            elementSize = sizeof(float);
        }
        for (SizeType i = 0; i < batchStrides.GetNumDims(); ++i) {
            strideSrc += batchStrides[i] * elementSize * batchSize;
            strideDiffDst += batchStrides[i] * elementSize * batchSize;
            strideMean += batchStrides[i] * groupNum;
        }
        SizeType strideDiffSrc = strideSrc;
        SizeType strideVariance = strideMean;
        RunSequential(batchSplitDims, static_cast<const std::byte *>(src) + strideSrc,
            mean + strideMean, variance + strideVariance,
            static_cast<const std::byte *>(diffDst) + strideDiffDst,
            static_cast<const std::byte *>(scale),
            static_cast<std::byte *>(diffSrc) + strideDiffSrc,
            diffScale, diffShift,
            eps);
    }

    void RunSequential(const Shape &batchSplitDims, const void *src, const float *mean, const float *variance,
        const void *diffDst, void *diffScale, void *diffShift, float *bufScale, float *bufShift,
        float eps) const noexcept(false)
    {
        Service::Unused(bufScale, bufShift);
        bool isDifferentType = ((srcType != dstType) || (scaleShiftType != dstType))
            && (dstType != Element::TypeT::F32);
        if (isDifferentType) {
            Service::ParseType<true, false>(compType, Element::TypeT::F32, Element::TypeT::F32, *this,
                batchSplitDims, src, mean, variance, diffDst, diffScale, diffShift, bufScale, bufShift, eps);
        } else {
            Service::ParseType<true, false>(compType, dstType, dstType, *this, batchSplitDims, src, mean, variance,
                diffDst, diffScale, diffShift, bufScale, bufShift, eps);
        }
    }

    void Run(const void *src, const float *mean, const float *variance, const void *diffDst,
        const void *scale, void *diffSrc, void *diffScale, void *diffShift, float eps) noexcept(false)
    {
        CheckInputArg(src, mean, variance, diffDst, scale, diffSrc, diffScale, diffShift);
        const void *srcExec = src;
        const void *scaleExec = scale;
        void *diffSrcExec = diffSrc;
        float *diffScaleExec = static_cast<float*>(diffScale);
        float *diffShiftExec = static_cast<float*>(diffShift);
        const void *diffDstExec = diffDst;

        BeforeRunConvert(&srcExec, &scaleExec, &diffSrcExec, &diffDstExec, src, scale, diffDst);

        Threading::BatchThreader(*this, srcExec, mean, variance, diffDstExec, scaleExec,
            diffSrcExec, diffScaleExec, diffShiftExec, eps);

        if (needToConvert && (dstType != Element::TypeT::F32)) {
            Service::Deallocate(diffDstExec);
        }
        if (needToConvert) {
            ReorderLayer r({reorderedDims, Element::TypeT::F32, Layout::ABCDE},
                           {reorderedDims, srcType, Layout::ABCDE});
            r.Run(diffSrcExec, diffSrc);
            Service::Deallocate(diffSrcExec);
        }
        if (needToConvert) {
            Service::Deallocate(srcExec);
        }
        if (needToConvert && scale) {
            Service::Deallocate(scaleExec);
        }
    }

private:
    Element::Type srcType;
    Element::Type scaleShiftType;
    Element::Type dstType;
    Element::Type compType;
    SizeType finalDimIdx;
    SizeType numDims;
    NormalizationFlags normFlags;
    SizeType channelSize;
    SizeType groupNum;
    SizeType batchNum;
    SizeType batchSize;
    SizeType channelPerGroup;
    SizeType eltPerGroup;
    Shape reorderedDims;
    bool needToConvert;

    void CheckInputArg(const void *src, const float *mean, const float *variance, const void *diffDst,
        const void *scale, void *diffSrc, void *diffScale, void *diffShift) const noexcept(false)
    {
        if ((src == nullptr) || (mean == nullptr) || (variance == nullptr) || (diffDst == nullptr) ||
            (diffSrc == nullptr)) {
            throw Service::LogicError {"Normalization Layer: one of input matrices is nullptr"};
        }
        if ((static_cast<bool>(normFlags & NormalizationFlags::USE_SCALE) && (scale == nullptr)) &&
            ((scale == nullptr) || (diffScale == nullptr))) {
            throw Service::LogicError {"Normalization Layer: one of input matrices is nullptr"};
        }
        if ((static_cast<bool>(normFlags & NormalizationFlags::USE_SHIFT) && (scale == nullptr)) &&
            (diffShift == nullptr)) {
            throw Service::LogicError {"Normalization Layer: one of input matrices is nullptr"};
        }
    }

    void BeforeRunConvert(const void **srcExec, const void **scaleExec, void **diffSrcExec,
        const void **diffDstExec, const void *src, const void *scale, const void *diffDst) noexcept(false)
    {
        if (needToConvert) {
            void *tmp = Service::AlignedAlloc(reorderedDims.GetTotalDimsSize() * sizeof(float));
            ReorderLayer r({reorderedDims, srcType, Layout::ABCDE},
                           {reorderedDims, Element::TypeT::F32, Layout::ABCDE});
            r.Run(src, tmp);
            *srcExec = tmp;
        }
        if (needToConvert && scale) {
            void *tmp = Service::AlignedAlloc(channelSize * sizeof(float));
            ReorderLayer r({{channelSize}, scaleShiftType, Layout::A}, {
                            {channelSize}, Element::TypeT::F32, Layout::A});
            r.Run(scale, tmp);
            *scaleExec = tmp;
        }
        if (needToConvert) {
            *diffSrcExec = Service::AlignedAlloc(reorderedDims.GetTotalDimsSize() * sizeof(float));
        }
        if (needToConvert && (dstType != Element::TypeT::F32)) {
            void *tmp = Service::AlignedAlloc(reorderedDims.GetTotalDimsSize() * sizeof(float));
            ReorderLayer r({reorderedDims, dstType, Layout::ABCDE},
                           {reorderedDims, Element::TypeT::F32, Layout::ABCDE});
            r.Run(diffDst, tmp);
            *diffDstExec = tmp;
        }
    }

    SizeType GetOptNumThreads() const noexcept
    {
        if (static_cast<bool>(normFlags & NormalizationFlags::USE_GLOBAL_STATS) &&
            (!static_cast<bool>(normFlags & NormalizationFlags::USE_SCALE)) &&
            (!static_cast<bool>(normFlags & NormalizationFlags::USE_SHIFT))) {
            return batchNum;
        }
        return 1;
    }

    template <typename CompType, typename InType, typename OutType, typename ScaleShiftType>
    void ChooseImpl(const Shape &batchSplitDims, const InType *src, const float *mean, const float *variance,
        const OutType *diffDst, const ScaleShiftType *scale, InType *diffSrc,
        float *diffScale, float *diffShift, float eps) const noexcept(false)
    {
        SizeType threadBatchNum = batchSplitDims[IDX_5 - numDims];

        if (finalDimIdx == (reorderedDims.GetNumDims() - IDX_1)) {
            GNorm::EvaluteAbx<CompType>(threadBatchNum, eltPerGroup, src, mean, variance, diffDst,
                scale, diffSrc, diffScale, diffShift, eps, normFlags, channelSize, channelPerGroup);
        } else {
            GNorm::EvaluteAxb<CompType>(threadBatchNum, eltPerGroup, src, mean, variance, diffDst,
                scale, diffSrc, diffScale, diffShift, eps, normFlags, channelSize, channelPerGroup);
        }
    }
};

} // Detail

GroupNormalizationLayerBWD::~GroupNormalizationLayerBWD() noexcept {}

GroupNormalizationLayerBWD::GroupNormalizationLayerBWD(const TensorInfo &srcInfo, const TensorInfo &statInfo,
    const TensorInfo &diffSrcInfo, const TensorInfo &diffDstInfo,
    const TensorInfo &scaleShiftInfo, const TensorInfo &diffscaleShiftInfo,
    NormalizationFlags flags) noexcept(false)
    : pImpl(new Detail::GroupNormalizationLayerBWDImpl(srcInfo, statInfo, diffSrcInfo, diffDstInfo, scaleShiftInfo,
    diffscaleShiftInfo, flags)) {}

GroupNormalizationLayerBWD::GroupNormalizationLayerBWD(const GroupNormalizationLayerBWD &other) noexcept(false)
    : pImpl(new Detail::GroupNormalizationLayerBWDImpl{*(other.pImpl.get())}) {}

GroupNormalizationLayerBWD::GroupNormalizationLayerBWD(GroupNormalizationLayerBWD &&other) noexcept
    : pImpl(std::move(other.pImpl)) {}

GroupNormalizationLayerBWD& GroupNormalizationLayerBWD::operator=(const GroupNormalizationLayerBWD &other)
    noexcept(false)
{
    if (this != &other) {
        this->pImpl.reset(new Detail::GroupNormalizationLayerBWDImpl{*(other.pImpl.get())});
    }
    return *this;
}

GroupNormalizationLayerBWD& GroupNormalizationLayerBWD::operator=(GroupNormalizationLayerBWD &&other) noexcept
{
    if (this != &other) {
        this->pImpl = std::move(other.pImpl);
    }
    return *this;
}

void GroupNormalizationLayerBWD::Run(const void *src, const float *mean, const float *variance, const void *diffDst,
    const void *scale, void *diffSrc, void *diffScale, void *diffShift, float eps) const noexcept(false)
{
    pImpl->Run(src, mean, variance, diffDst, scale, diffSrc, diffScale, diffShift, eps);
}

Status GroupNormalizationLayerBWD::ValidateInput(const TensorInfo &srcInfo, const TensorInfo &statInfo,
    const TensorInfo &diffSrcInfo, const TensorInfo &diffDstInfo,
    const TensorInfo &scaleShiftInfo, const TensorInfo &diffscaleShiftInfo,
    NormalizationFlags flags) noexcept
{
    if (srcInfo != diffSrcInfo) {
        return Status::BAD_ARGUMENTS;
    }
    if (statInfo.GetType() != Element::TypeT::F32) {
        return Status::UNSUPPORTED;
    }
    if (srcInfo.GetLayout() != srcInfo.GetStandardABXLayout() &&
        srcInfo.GetLayout() != srcInfo.GetStandardAXBLayout()) {
        return Status::UNSUPPORTED;
    }

    bool isUnsupported = ((static_cast<bool>(flags & NormalizationFlags::USE_SCALE)) ||
        (static_cast<bool>(flags & NormalizationFlags::USE_SHIFT))) &&
        (scaleShiftInfo.GetType() != Element::TypeT::F32) &&
        (scaleShiftInfo.GetType() != Element::TypeT::F16) &&
        (scaleShiftInfo.GetType() != Element::TypeT::BF16);
    if (isUnsupported) {
        return Status::UNSUPPORTED;
    }
    if (srcInfo.GetDims() != diffDstInfo.GetDims()) {
        return Status::UNSUPPORTED;
    }
    if (srcInfo.GetLayout() != diffDstInfo.GetLayout()) {
        return Status::UNSUPPORTED;
    }
    if (scaleShiftInfo != diffscaleShiftInfo) {
        return Status::BAD_ARGUMENTS;
    }
    if (srcInfo.GetTotalTensorSize() == 0) {
        return Status::UNSUPPORTED;
    }
    if (scaleShiftInfo.GetNumDims() != 1) {
        return Status::BAD_ARGUMENTS;
    }
    if (srcInfo.GetDims()[IDX_0] != statInfo.GetDims()[IDX_0]) {
        return Status::BAD_ARGUMENTS;
    }
    if ((srcInfo.GetDims()[IDX_1] % statInfo.GetDims()[IDX_1]) != 0) {
        return Status::BAD_ARGUMENTS;
    }
    if (!srcInfo.IsDense() || !statInfo.IsDense() || !diffDstInfo.IsDense() ||
        !scaleShiftInfo.IsDense() || !diffscaleShiftInfo.IsDense()) {
        return Status::UNSUPPORTED;
    }
    return Status::SUCCESS;
}

} // KuDNN
