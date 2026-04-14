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
#include <iostream>
#include <arm_sve.h>

#include "kudnn.hpp"

#include "internal_funcs.hpp"
#include "threading_impl.hpp"
#include "kudnn_group_normalization_impl.hpp"
#include "kernels/gnorm/kudnn_gnorm_kernels.hpp"

namespace KuDNN {
namespace Detail {

static constexpr SizeType GNORM_FWD_ELE_NUM_1 = 10000;
static constexpr SizeType LGNORM_FWD_ELE_NUM_2 = 100000;

static constexpr SizeType GNORM_FWD_THREADS_THREASHOLD = 1;

// optNumThreads = c0 + c1 * log(numElems), 10000 < numElems <= 100000
static constexpr float GNORM_FWD_THREADS_CALCSTATS1_C0 = -30.25f;
static constexpr float GNORM_FWD_THREADS_CALCSTATS1_C1 = 3.68f;
// optNumThreads = c0 + c1 * log(numElems), numElems > 100000
static constexpr float GNORM_FWD_THREADS_CALCSTATS2_C0 = -170.99f;
static constexpr float GNORM_FWD_THREADS_CALCSTATS2_C1 = 15.86f;

// optNumThreads = c0 + c1 * numElems, 10000 < numElems <= 100000
static constexpr float GNORM_FWD_THREADS_GLOBALSTATS1_C0 = -0.19f;
static constexpr float GNORM_FWD_THREADS_GLOBALSTATS1_C1 = 0.00012f;
// optNumThreads = c0 + c1 * log(numElems), numElems > 100000
static constexpr float GNORM_FWD_THREADS_GLOBALSTATS2_C0 = -149.24f;
static constexpr float GNORM_FWD_THREADS_GLOBALSTATS2_C1 = 13.49f;

class GroupNormalizationLayerFWDImpl final : public Parallelizable {
public:
    GroupNormalizationLayerFWDImpl(const TensorInfo &srcInfo, const TensorInfo &scaleshiftInfo, SizeType groupInfo,
        const TensorInfo &dstInfo, NormalizationFlags flags) noexcept(false)
        : srcType(srcInfo.GetType()), scaleShiftType(scaleshiftInfo.GetType()), dstType(dstInfo.GetType()),
        compType(Element::TypeT::F32), groupSize(groupInfo),
        /* since all the arrays are broadcasted to 5D, we need to add extra dimensions which were broadcasted */
        channelAxisNum(srcInfo.GetAxisNumAccordingToLayout(IDX_1) + MAX_DIMS - srcInfo.GetNumDims()),
        batchAxisNum(srcInfo.GetAxisNumAccordingToLayout(IDX_0) + MAX_DIMS - srcInfo.GetNumDims()),
        srcLayout(srcInfo.GetLayout()),
        normFlags(flags)
    {
        // threading implementation depends on the value of calculateStats flag
        // so we initialize it at the constructor stage
        Service::ThrowOnStatus(GroupNormalizationLayerFWD::ValidateInput(srcInfo, scaleshiftInfo, groupInfo,
            dstInfo, flags), "GroupNormalization Layer");
        reorderedDims = Service::BroadcastTo5D(Service::GetShapeAccordingToLayout(srcInfo.GetDims(),
            srcInfo.GetLayout()));
        channelAxisSize = reorderedDims[channelAxisNum];
        innerSize = Service::ArrayProduct(reorderedDims, reorderedDims.GetNumDims(), channelAxisNum + 1);

        Parallelizable::Initialize(GetOptNumThreads(), BatchedDims());
    }

    template <typename CompType, typename InType, typename OutType>
    void Evaluate(const Shape &batchSplitDims, const void *src, void *dst, const void *scale,
            const void *shift, float *mean, float *variance, bool saveStats, const float eps) const noexcept
    {
        const InType *typedSrc = static_cast<const InType *>(src);
        OutType *typedDst = static_cast<OutType *>(dst);
        const OutType *typedScale = static_cast<const OutType *>(scale);
        const OutType *typedShift = static_cast<const OutType *>(shift);
        ChooseImpl<CompType>(batchSplitDims, typedSrc, typedDst, typedScale, typedShift,
            mean, variance, saveStats, eps);
    }

    void RunSequential(const Shape &batchSplitDims, const void *src, void *dst, const void *scale,
            const void *shift, float *mean, float *variance, bool saveStats, const float eps) const noexcept
    {
        if ((srcType != dstType) || (scaleShiftType != dstType)) {
            Service::ParseType<true, false>(compType, Element::TypeT::F32, Element::TypeT::F32, *this,
                batchSplitDims, src, dst, scale, shift, mean, variance, saveStats, eps);
        } else {
            Service::ParseType<true, false>(compType, compType, dstType, *this, batchSplitDims, src, dst, scale, shift,
                mean, variance, saveStats, eps);
        }
    }

    void RunSequential(const Shape &batchSplitDims, const Shape &batchStrides,
            const void *src, void *dst, const void *scale,
            const void *shift, float *mean, float *variance,
        bool saveStats, const float eps) const noexcept(false)
    {
        SizeType strideSrc = 0;
        SizeType strideDst = 0;
        SizeType strideScale = 0;
        SizeType strideMean = 0;
        SizeType elementSize = dstType.GetSize();
        if ((srcType != dstType) || (scaleShiftType != dstType)) {
            elementSize = sizeof(float);
        }

        for (SizeType i = 0; i < batchStrides.GetNumDims(); ++i) {
            SizeType size = Service::ArrayProduct(reorderedDims, reorderedDims.GetNumDims(), batchAxisNum + 1);
            strideSrc += batchStrides[i] * elementSize * size;
            strideDst += batchStrides[i] * elementSize * size;
            strideMean += batchStrides[i] * groupSize;
        }

        SizeType strideShift = strideScale;
        SizeType strideVariance = strideMean;
        if ((!static_cast<bool>(normFlags & NormalizationFlags::USE_GLOBAL_STATS)) && (!saveStats)) {
            strideMean = 0;
            strideVariance = 0;
        }

        RunSequential(batchSplitDims, static_cast<const std::byte *>(src) + strideSrc,
            static_cast<std::byte *>(dst) + strideDst, static_cast<const std::byte *>(scale) + strideScale,
            static_cast<const std::byte *>(shift) + strideShift, mean + strideMean,
            variance + strideVariance, saveStats, eps);
    }

    void Run(const void *src, void *dst, const void *scale,
            const void *shift, float *mean, float *variance,
        bool saveStats, const float eps) const noexcept(false)
    {
        CheckInputArg(src, dst, scale, shift, mean, variance, saveStats);
        const void *srcExec = src;

        if ((srcType != dstType) || (scaleShiftType != dstType)) {
            void *tmp = Service::AlignedAlloc(reorderedDims.GetTotalDimsSize() * sizeof(float));
            ReorderLayer r({
                reorderedDims, srcType, Layout::ABCDE}, {
                reorderedDims, Element::TypeT::F32, Layout::ABCDE});
            r.Run(src, tmp);
            srcExec = tmp;
        }
        const void *scaleExec = scale;
        const void *shiftExec = shift;
        if ((srcType != dstType) || (scaleShiftType != dstType)) {
            ReorderLayer r({{
                channelAxisSize}, scaleShiftType, Layout::A}, {{
                channelAxisSize}, Element::TypeT::F32, Layout::A});
            void *tmp;
            if (scale != nullptr) {
                tmp = Service::AlignedAlloc(channelAxisSize * sizeof(float));
                r.Run(scale, tmp);
                scaleExec = tmp;
            }
            if (shift != nullptr) {
                tmp = Service::AlignedAlloc(channelAxisSize * sizeof(float));
                r.Run(shift, tmp);
                shiftExec = tmp;
            }
        }
        void *dstExec = dst;
        if (((srcType != dstType) || (scaleShiftType != dstType)) && (dstType != Element::TypeT::F32)) {
            dstExec = Service::AlignedAlloc(reorderedDims.GetTotalDimsSize() * sizeof(float));
        }
        Threading::BatchThreader(*this, srcExec, dstExec, scaleExec,
            shiftExec, mean, variance, saveStats, eps);
        if ((srcType != dstType) || (scaleShiftType != dstType)) {
            Service::Deallocate(const_cast<void *>(srcExec));
        }
        if ((srcType != dstType) || (scaleShiftType != dstType)) {
            Service::Deallocate(const_cast<void *>(scaleExec));
            Service::Deallocate(const_cast<void *>(shiftExec));
        }
        if (((srcType != dstType) || (scaleShiftType != dstType)) && (dstType != Element::TypeT::F32)) {
            ReorderLayer r({
                reorderedDims, Element::TypeT::F32, Layout::ABCDE}, {
                reorderedDims, dstType, Layout::ABCDE});
            r.Run(dstExec, dst);
            Service::Deallocate(dstExec);
        }
    }
private:
    Element::Type srcType;
    Element::Type scaleShiftType;
    Element::Type dstType;
    Element::Type compType;
    SizeType groupSize;
    SizeType channelAxisNum;
    SizeType batchAxisNum;
    Layout srcLayout;
    NormalizationFlags normFlags;
    SizeType channelAxisSize;
    SizeType innerSize;
    Shape reorderedDims;

    void CheckInputArg(const void *src, void *dst, const void *scale,
        const void *shift, float *mean, float *variance, bool saveStats) const
    {
        if ((src == nullptr) || (dst == nullptr)) {
            throw Service::LogicError {"Normalization: one of input matrices is nullptr"};
        }
        if ((static_cast<bool>(normFlags & NormalizationFlags::USE_GLOBAL_STATS) || saveStats) &&
            ((variance == nullptr) || (mean == nullptr))) {
            throw Service::LogicError {"Normalization: one of input matrices is nullptr"};
        }
        if (static_cast<bool>(normFlags & NormalizationFlags::USE_SCALE) && (scale == nullptr)) {
            throw Service::LogicError {"Normalization Layer: one of input matrices is nullptr"};
        }
        if (static_cast<bool>(normFlags & NormalizationFlags::USE_SHIFT) && (shift == nullptr)) {
            throw Service::LogicError {"Normalization Layer: one of input matrices is nullptr"};
        }
    }

    SizeType GetOptNumThreads() const noexcept
    {
        SizeType totalSize = reorderedDims.GetTotalDimsSize();
        if (!static_cast<bool>(normFlags & NormalizationFlags::USE_GLOBAL_STATS)) {
            if (totalSize < GNORM_FWD_ELE_NUM_1) {
                return GNORM_FWD_THREADS_THREASHOLD;
            } else if (totalSize <= LGNORM_FWD_ELE_NUM_2) {
                return GNORM_FWD_THREADS_CALCSTATS1_C0 + GNORM_FWD_THREADS_CALCSTATS1_C1 * std::log(totalSize);
            } else {
                return GNORM_FWD_THREADS_CALCSTATS2_C0 + GNORM_FWD_THREADS_CALCSTATS2_C1 * std::log(totalSize);
            }
        } else {
            if (totalSize < GNORM_FWD_ELE_NUM_1) {
                return GNORM_FWD_THREADS_THREASHOLD;
            } else if (totalSize <= LGNORM_FWD_ELE_NUM_2) {
                return GNORM_FWD_THREADS_GLOBALSTATS1_C0 + GNORM_FWD_THREADS_GLOBALSTATS1_C1 * totalSize;
            } else {
                return GNORM_FWD_THREADS_GLOBALSTATS2_C0 + GNORM_FWD_THREADS_GLOBALSTATS2_C1 * std::log(totalSize);
            }
        }
    }

    Shape BatchedDims() const noexcept
    {
        std::array<SizeType, MAX_DIMS> batchedDIms;
        for (SizeType i = 0; i < batchAxisNum + 1; ++i) {
            batchedDIms[i] = reorderedDims[i];
        }
        return {Shape{batchedDIms.data(), batchAxisNum + 1}.GetTotalDimsSize()};
    }

    template <typename CompType, typename InType, typename OutType, typename ScaleShiftType>
    void ChooseImpl(const Shape &batchSplitDims, const InType *src, OutType *dst, const ScaleShiftType *scale,
            const ScaleShiftType *shift, float *mean, float *variance, bool saveStats, const float eps) const noexcept
    {
        SizeType outerSize = batchSplitDims.GetTotalDimsSize();
        if (channelAxisNum == (reorderedDims.GetNumDims() - 1)) {
            SizeType outerSizeLast = outerSize * (Service::ArrayProduct(reorderedDims,
                channelAxisNum, batchAxisNum + 1));
            GNorm::EvaluateLastAxis<CompType>(outerSizeLast, outerSize, src, dst, groupSize,
                scale, shift, mean, variance, saveStats, eps, normFlags, channelAxisSize);
        } else {
            GNorm::EvaluateCrossAxis<CompType>(outerSize, src, dst, groupSize,
                scale, shift, mean, variance, saveStats, eps, normFlags, channelAxisSize, innerSize);
        }
    }
};

} // detail

GroupNormalizationLayerFWD::~GroupNormalizationLayerFWD() noexcept {}

GroupNormalizationLayerFWD::GroupNormalizationLayerFWD(const TensorInfo &srcInfo,
    const TensorInfo &scaleshiftInfo, SizeType groupSize, const TensorInfo &dstInfo,
    NormalizationFlags flags) noexcept(false)
    : pImpl(new Detail::GroupNormalizationLayerFWDImpl(srcInfo, scaleshiftInfo, groupSize, dstInfo, flags)) {}

GroupNormalizationLayerFWD::GroupNormalizationLayerFWD(const GroupNormalizationLayerFWD &other) noexcept(false)
    : pImpl(new Detail::GroupNormalizationLayerFWDImpl{*(other.pImpl.get())}) {}

GroupNormalizationLayerFWD::GroupNormalizationLayerFWD(GroupNormalizationLayerFWD &&other) noexcept
    : pImpl(std::move(other.pImpl)) {}

GroupNormalizationLayerFWD& GroupNormalizationLayerFWD::operator=(
    const GroupNormalizationLayerFWD &other) noexcept(false)
{
    if (this != &other) {
        this->pImpl.reset(new Detail::GroupNormalizationLayerFWDImpl{*(other.pImpl.get())});
    }
    return *this;
}

GroupNormalizationLayerFWD& GroupNormalizationLayerFWD::operator=(GroupNormalizationLayerFWD &&other) noexcept
{
    if (this != &other) {
        this->pImpl = std::move(other.pImpl);
    }
    return *this;
}

void GroupNormalizationLayerFWD::Run(const void *src, void *dst,
    const void *scale, const void *shift,
    float *mean, float *variance,
    bool saveStats, const float eps) const noexcept(false)
{
    pImpl->Run(src, dst, scale, shift, mean, variance, saveStats, eps);
}

Status GroupNormalizationLayerFWD::ValidateInput(const TensorInfo &srcInfo, const TensorInfo &scaleshiftInfo,
    SizeType groupInfo, const TensorInfo &dstInfo, NormalizationFlags flags) noexcept(false)
{
    Service::Unused(flags);
    if (srcInfo.GetDims() != dstInfo.GetDims()) {
        return Status::BAD_ARGUMENTS;
    }
    if (srcInfo.GetLayout() != dstInfo.GetLayout()) {
        return Status::UNSUPPORTED;
    }
    if (!(srcInfo.GetType() == Element::TypeT::F16 && dstInfo.GetType() == Element::TypeT::F16) &&
        !(srcInfo.GetType() == Element::TypeT::F32 && dstInfo.GetType() == Element::TypeT::F32)) {
        return Status::UNSUPPORTED;
    }
    if (((static_cast<bool>(flags & NormalizationFlags::USE_SCALE)) ||
        (static_cast<bool>(flags & NormalizationFlags::USE_SHIFT))) &&
        (scaleshiftInfo.GetType() != Element::TypeT::F32) && (scaleshiftInfo.GetType() != Element::TypeT::F16)) {
        return Status::UNSUPPORTED;
    }
    if (srcInfo.GetLayout() != dstInfo.GetLayout()) {
        return Status::UNSUPPORTED;
    }

    if (srcInfo.GetTotalTensorSize() == 0) {
        return Status::UNSUPPORTED;
    }
    if (scaleshiftInfo.GetNumDims() != 1) {
        return Status::BAD_ARGUMENTS;
    }

    if (((static_cast<bool>(flags & NormalizationFlags::USE_SCALE)) ||
        (static_cast<bool>(flags & NormalizationFlags::USE_SHIFT))) &&
        srcInfo.GetDims()[IDX_1] != scaleshiftInfo.GetDims()[IDX_0]) {
        return Status::BAD_ARGUMENTS;
    }

    if (groupInfo <= 0 || groupInfo > srcInfo.GetDims()[IDX_1]
        || (srcInfo.GetDims()[IDX_1] % groupInfo) != 0) {
        return Status::BAD_ARGUMENTS;
    }

    if (!srcInfo.IsDense() || !scaleshiftInfo.IsDense() || !dstInfo.IsDense()) {
        return Status::UNSUPPORTED;
    }

    return Status::SUCCESS;
}

} // KuDNN
