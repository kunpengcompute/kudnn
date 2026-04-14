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
#include "threading_impl.hpp"
#include "internal_funcs.hpp"
#include "kudnn_layer_normalization_impl.hpp"
#include "kernels/lnorm/kudnn_lnorm_kernels.hpp"

namespace KuDNN {
namespace Detail {

static constexpr SizeType LNORM_FWD_ELE_NUM_1 = 10000;
static constexpr SizeType LLNORM_FWD_ELE_NUM_2 = 100000;

static constexpr SizeType LNORM_FWD_THREADS_THREASHOLD = 1;

// optNumThreads = c0 + c1 * log(numElems), 10000 < numElems <= 100000
static constexpr float LNORM_FWD_THREADS_CALCSTATS1_C0 = -30.25f;
static constexpr float LNORM_FWD_THREADS_CALCSTATS1_C1 = 3.68f;
// optNumThreads = c0 + c1 * log(numElems), numElems > 100000
static constexpr float LNORM_FWD_THREADS_CALCSTATS2_C0 = -170.99f;
static constexpr float LNORM_FWD_THREADS_CALCSTATS2_C1 = 15.86f;

// optNumThreads = c0 + c1 * numElems, 10000 < numElems <= 100000
static constexpr float LNORM_FWD_THREADS_GLOBALSTATS1_C0 = -0.19f;
static constexpr float LNORM_FWD_THREADS_GLOBALSTATS1_C1 = 0.00012f;
// optNumThreads = c0 + c1 * log(numElems), numElems > 100000
static constexpr float LNORM_FWD_THREADS_GLOBALSTATS2_C0 = -149.24f;
static constexpr float LNORM_FWD_THREADS_GLOBALSTATS2_C1 = 13.49f;

class NormalizationLayerFWDImpl final : public Parallelizable {
public:
    NormalizationLayerFWDImpl(const TensorInfo &srcInfo, const TensorInfo &statsInfo,
        const TensorInfo &scaleshiftInfo, const TensorInfo &dstInfo, NormalizationFlags flags) noexcept(false)
        : srcType(srcInfo.GetType()), scaleShiftType(scaleshiftInfo.GetType()), dstType(dstInfo.GetType()),
        compType(Element::TypeT::F32),
        /* since all the arrays are broadcasted to 5D, we need to add extra dimensions which were broadcasted */
        axisNum(srcInfo.GetAxisNumAccordingToLayout((srcInfo.GetNumDims() - 1)) + MAX_DIMS - srcInfo.GetNumDims()),
        normFlags(flags)
    {
        // threading implementation depends on the value of calculateStats flag
        // so we initialize it at the constructor stage
        Service::ThrowOnStatus(NormalizationLayerFWD::ValidateInput(srcInfo, statsInfo, scaleshiftInfo,
            dstInfo, flags), "Normalization Layer");

        reorderedDims = Service::BroadcastTo5D(
            Service::GetShapeAccordingToLayout(srcInfo.GetDims(), srcInfo.GetLayout()));
        axisSize = reorderedDims[axisNum];
        innerSize = Service::ArrayProduct(reorderedDims, reorderedDims.GetNumDims(), axisNum + 1);
        statShape = statsInfo.GetDims();
        dstShape = dstInfo.GetDims();
        srcLayout = srcInfo.GetLayout();
        statLayout = statsInfo.GetLayout();
        dstLayout = dstInfo.GetLayout();

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
            strideSrc += batchStrides[i] * elementSize * (axisSize * innerSize);
            strideDst += batchStrides[i] * elementSize * (axisSize * innerSize);
            strideMean += batchStrides[i] * innerSize;
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
        const void *scaleExec = scale;
        const void *shiftExec = shift;
        bool calculateStats = !static_cast<bool>(normFlags & NormalizationFlags::USE_GLOBAL_STATS);
        float *meanExec = mean;
        float *varianceExec = variance;
        void *dstExec = dst;
        
        BeforeRunReorderLayer(&srcExec, &scaleExec, &shiftExec, &meanExec, &varianceExec,
            &dstExec, src, scale, shift, mean, variance);
        
        Threading::BatchThreader(*this, srcExec, dstExec, scaleExec,
            shiftExec, meanExec, varianceExec, saveStats, eps);

        AfterRunReorderLayer(&srcExec, &scaleExec, &shiftExec, &meanExec, &varianceExec, &dstExec,
            calculateStats, mean, variance, dst);
    }
private:
    Element::Type srcType;
    Element::Type scaleShiftType;
    Element::Type dstType;
    Element::Type compType;
    SizeType axisNum;
    NormalizationFlags normFlags;
    SizeType axisSize;
    SizeType innerSize;
    Shape reorderedDims;
    Shape statShape;
    Shape dstShape;
    Layout srcLayout;
    Layout statLayout;
    Layout dstLayout;

    void BeforeRunReorderLayer(const void **srcExec, const void **scaleExec, const void **shiftExec,
        float **meanExec, float **varianceExec, void **dstExec, const void *src, const void *scale,
            const void *shift, float *mean, float *variance) const noexcept(false)
    {
        if ((srcType != dstType) || (scaleShiftType != dstType)) {
            void *tmp = Service::AlignedAlloc(reorderedDims.GetTotalDimsSize() * sizeof(float));
            ReorderLayer r({
                reorderedDims, srcType, Layout::ABCDE}, {
                reorderedDims, Element::TypeT::F32, Layout::ABCDE
            });
            r.Run(src, tmp);
            *srcExec = tmp;
        }
        if ((srcType != dstType) || (scaleShiftType != dstType)) {
            ReorderLayer r({{axisSize}, scaleShiftType, Layout::A}, {{axisSize}, Element::TypeT::F32, Layout::A});
            void *tmp;
            if (scale != nullptr) {
                tmp = Service::AlignedAlloc(axisSize * sizeof(float));
                r.Run(scale, tmp);
                *scaleExec = tmp;
            }
            if (shift != nullptr) {
                tmp = Service::AlignedAlloc(axisSize * sizeof(float));
                r.Run(shift, tmp);
                *shiftExec = tmp;
            }
        }
        if (statLayout != GetSrcLayoutWithoutLastDim(srcLayout)) {
            ReorderLayer r({statShape, Element::TypeT::F32, statLayout}, {statShape,
                Element::TypeT::F32, GetSrcLayoutWithoutLastDim(srcLayout)});
            void *tmp;
            if (mean != nullptr) {
                tmp = Service::AlignedAlloc(statShape.GetTotalDimsSize() * sizeof(float));
                r.Run(mean, tmp);
                *meanExec = static_cast<float *>(tmp);
            }
            if (variance != nullptr) {
                tmp = Service::AlignedAlloc(statShape.GetTotalDimsSize() * sizeof(float));
                r.Run(variance, tmp);
                *varianceExec = static_cast<float *>(tmp);
            }
        }
        bool isNotFullMatch = (dstLayout != srcLayout) ||
            (((srcType != dstType) || (scaleShiftType != dstType)) && (dstType != Element::TypeT::F32));
        if (isNotFullMatch) {
            bool dt = (((srcType != dstType) || (scaleShiftType != dstType)) && (dstType != Element::TypeT::F32));
            *dstExec = Service::AlignedAlloc(
                reorderedDims.GetTotalDimsSize() * (dt ? sizeof(float) : dstType.GetSize()));
        }
    }

    void AfterRunReorderLayer(const void **srcExec, const void **scaleExec, const void **shiftExec,
        float **meanExec, float **varianceExec, void **dstExec, bool calculateStats,
        float *mean, float *variance, void *dst) const noexcept(false)
    {
        if ((srcType != dstType) || (scaleShiftType != dstType)) {
            Service::Deallocate(const_cast<void *>(*srcExec));
        }
        if ((srcType != dstType) || (scaleShiftType != dstType)) {
            Service::Deallocate(const_cast<void *>(*scaleExec));
            Service::Deallocate(const_cast<void *>(*shiftExec));
        }
        if (statLayout != GetSrcLayoutWithoutLastDim(srcLayout)) {
            if (calculateStats) {
                ReorderLayer r({
                    statShape, Element::TypeT::F32, GetSrcLayoutWithoutLastDim(srcLayout)}, {
                    statShape, Element::TypeT::F32, statLayout});
                if (mean != nullptr) {
                    r.Run(*meanExec, mean);
                }
                if (variance != nullptr) {
                    r.Run(*varianceExec, variance);
                }
            }
            if (mean != nullptr) {
                Service::Deallocate(*meanExec);
            }
            if (variance != nullptr) {
                Service::Deallocate(*varianceExec);
            }
        }
        bool isNotFullMatch = (dstLayout != srcLayout) ||
            (((srcType != dstType) || (scaleShiftType != dstType)) && (dstType != Element::TypeT::F32));
        if (isNotFullMatch) {
            bool dt = (((srcType != dstType) || (scaleShiftType != dstType)) && (dstType != Element::TypeT::F32));
            ReorderLayer r({dstShape, dt ? Element::Type{Element::TypeT::F32} : dstType,
                srcLayout}, {dstShape, dstType, dstLayout});
            r.Run(*dstExec, dst);
            Service::Deallocate(*dstExec);
        }
    }

    void CheckInputArg(const void *src, void *dst, const void *scale,
        const void *shift, float *mean, float *variance,
        bool saveStats) const
    {
        if ((src == nullptr) || (dst == nullptr)) {
            throw Service::LogicError {"Normalization: one of input matrices is nullptr"};
        }
        if ((static_cast<bool>(normFlags & NormalizationFlags::USE_GLOBAL_STATS) || saveStats) &&
            ((mean == nullptr) || (variance == nullptr))) {
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
            if (totalSize < LNORM_FWD_ELE_NUM_1) {
                return LNORM_FWD_THREADS_THREASHOLD;
            } else if (totalSize <= LLNORM_FWD_ELE_NUM_2) {
                return LNORM_FWD_THREADS_CALCSTATS1_C0 + LNORM_FWD_THREADS_CALCSTATS1_C1 * std::log(totalSize);
            } else {
                return LNORM_FWD_THREADS_CALCSTATS2_C0 + LNORM_FWD_THREADS_CALCSTATS2_C1 * std::log(totalSize);
            }
        } else {
            if (totalSize < LNORM_FWD_ELE_NUM_1) {
                return LNORM_FWD_THREADS_THREASHOLD;
            } else if (totalSize <= LLNORM_FWD_ELE_NUM_2) {
                return LNORM_FWD_THREADS_GLOBALSTATS1_C0 + LNORM_FWD_THREADS_GLOBALSTATS1_C1 * totalSize;
            } else {
                return LNORM_FWD_THREADS_GLOBALSTATS2_C0 + LNORM_FWD_THREADS_GLOBALSTATS2_C1 * std::log(totalSize);
            }
        }
    }

    Shape BatchedDims() const noexcept
    {
        if (axisNum == 0) {
            return {1};
        } else {
            std::array<SizeType, MAX_DIMS> batchedDIms;
            for (SizeType i = 0; i < axisNum; ++i) {
                batchedDIms[i] = reorderedDims[i];
            }
            return {Shape{batchedDIms.data(), axisNum}.GetTotalDimsSize()};
        }
    }

    template <typename CompType, typename InType, typename OutType, typename ScaleShiftType>
    void ChooseImpl(const Shape &batchSplitDims, const InType *src, OutType *dst, const ScaleShiftType *scale,
            const ScaleShiftType *shift, float *mean, float *variance, bool saveStats, const float eps) const noexcept
    {
        SizeType outerSize = batchSplitDims.GetTotalDimsSize();
        if (axisNum == (reorderedDims.GetNumDims() - 1)) {
            LNorm::EvaluateLastAxis<CompType>(outerSize, src, dst,
                scale, shift, mean, variance, saveStats, eps, normFlags, axisSize);
        } else {
            LNorm::EvaluateCrossAxis<CompType>(outerSize, src, dst,
                scale, shift, mean, variance, saveStats, eps, normFlags, axisSize, innerSize);
        }
    }
};

} // detail

NormalizationLayerFWD::~NormalizationLayerFWD() noexcept {}

NormalizationLayerFWD::NormalizationLayerFWD(const TensorInfo &srcInfo,
    const TensorInfo &statsInfo, const TensorInfo &scaleshiftInfo, const TensorInfo &dstInfo,
    NormalizationFlags flags) noexcept(false)
    : pImpl(new Detail::NormalizationLayerFWDImpl(srcInfo, statsInfo, scaleshiftInfo, dstInfo, flags)) {}

NormalizationLayerFWD::NormalizationLayerFWD(const NormalizationLayerFWD &other) noexcept(false)
    : pImpl(new Detail::NormalizationLayerFWDImpl{*(other.pImpl.get())}) {}

NormalizationLayerFWD::NormalizationLayerFWD(NormalizationLayerFWD &&other) noexcept
    : pImpl(std::move(other.pImpl)) {}

NormalizationLayerFWD& NormalizationLayerFWD::operator=(const NormalizationLayerFWD &other) noexcept(false)
{
    if (this != &other) {
        this->pImpl.reset(new Detail::NormalizationLayerFWDImpl{*(other.pImpl.get())});
    }
    return *this;
}

NormalizationLayerFWD& NormalizationLayerFWD::operator=(NormalizationLayerFWD &&other) noexcept
{
    if (this != &other) {
        this->pImpl = std::move(other.pImpl);
    }
    return *this;
}

void NormalizationLayerFWD::Run(const void *src, void *dst,
    const void *scale, const void *shift,
    float *mean, float *variance,
    bool saveStats, const float eps) const noexcept(false)
{
    pImpl->Run(src, dst, scale, shift, mean, variance, saveStats, eps);
}

Status NormalizationLayerFWD::ValidateInput(const TensorInfo &srcInfo,
    const TensorInfo &statsInfo, const TensorInfo &scaleshiftInfo, const TensorInfo &dstInfo,
    NormalizationFlags flags) noexcept(false)
{
    if (static_cast<bool>(flags & NormalizationFlags::FUSE_NORM_RELU)) {
        return Status::UNSUPPORTED;
    }
    if (srcInfo.GetDims() != dstInfo.GetDims()) {
        return Status::BAD_ARGUMENTS;
    }
    if (!(srcInfo.GetType() == Element::TypeT::F16 && dstInfo.GetType() == Element::TypeT::F16) &&
        !(srcInfo.GetType() == Element::TypeT::F32 && dstInfo.GetType() == Element::TypeT::F32)) {
        return Status::UNSUPPORTED;
    }
    if (((static_cast<bool>(flags & NormalizationFlags::USE_SCALE)) ||
        (static_cast<bool>(flags & NormalizationFlags::USE_SHIFT))) &&
        (scaleshiftInfo.GetType() != Element::TypeT::F32) &&
        (scaleshiftInfo.GetType() != Element::TypeT::F16)) {
        return Status::UNSUPPORTED;
    }
    if ((srcInfo.GetNumDims() - 1) != statsInfo.GetNumDims()) {
        return Status::BAD_ARGUMENTS;
    }
    if (srcInfo.GetTotalTensorSize() == 0) {
        return Status::UNSUPPORTED;
    }
    if (scaleshiftInfo.GetNumDims() != 1) {
        return Status::BAD_ARGUMENTS;
    }
    if (srcInfo.GetDims()[srcInfo.GetNumDims() - 1] != scaleshiftInfo.GetDims()[0]) {
        return Status::BAD_ARGUMENTS;
    }
    for (SizeType i = 0; i < statsInfo.GetNumDims(); ++i) {
        if (srcInfo.GetDims()[i] != statsInfo.GetDims()[i]) {
            return Status::BAD_ARGUMENTS;
        }
    }
    if (!srcInfo.IsDense() || !statsInfo.IsDense() || !scaleshiftInfo.IsDense() || !dstInfo.IsDense()) {
        return Status::UNSUPPORTED;
    }
    return Status::SUCCESS;
}

} // KuDNN
