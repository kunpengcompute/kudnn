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

#include <cmath>
#include <cstddef>
#include <map>
#include <memory>

#include <arm_sve.h>

#include "kudnn.hpp"

#include "kernels/lnorm/kudnn_lnorm_kernels.hpp"
#include "internal_funcs.hpp"
#include "threading_impl.hpp"
#include "vector_wrappers.hpp"

namespace KuDNN {

static const std::map<KuDNN::Layout, KuDNN::Layout> layoutMap = {
    {KuDNN::Layout::AB, KuDNN::Layout::A},
    {KuDNN::Layout::BA, KuDNN::Layout::A},
    {KuDNN::Layout::ABC, KuDNN::Layout::AB},
    {KuDNN::Layout::ACB, KuDNN::Layout::AB},
    {KuDNN::Layout::BAC, KuDNN::Layout::BA},
    {KuDNN::Layout::BCA, KuDNN::Layout::BA},
    {KuDNN::Layout::CAB, KuDNN::Layout::AB},
    {KuDNN::Layout::CBA, KuDNN::Layout::BA},
    {KuDNN::Layout::ABCD, KuDNN::Layout::ABC},
    {KuDNN::Layout::ABDC, KuDNN::Layout::ABC},
    {KuDNN::Layout::ACBD, KuDNN::Layout::ACB},
    {KuDNN::Layout::ACDB, KuDNN::Layout::ACB},
    {KuDNN::Layout::ADBC, KuDNN::Layout::ABC},
    {KuDNN::Layout::ADCB, KuDNN::Layout::ACB},
    {KuDNN::Layout::BACD, KuDNN::Layout::BAC},
    {KuDNN::Layout::BCDA, KuDNN::Layout::BCA},
    {KuDNN::Layout::CDAB, KuDNN::Layout::CAB},
    {KuDNN::Layout::CDBA, KuDNN::Layout::CBA},
    {KuDNN::Layout::DCAB, KuDNN::Layout::CAB},
    {KuDNN::Layout::ABCDE, KuDNN::Layout::ABCD},
    {KuDNN::Layout::ABCED, KuDNN::Layout::ABCD},
    {KuDNN::Layout::ABDEC, KuDNN::Layout::ABDC},
    {KuDNN::Layout::ACBDE, KuDNN::Layout::ACBD},
    {KuDNN::Layout::ACDEB, KuDNN::Layout::ACDB},
    {KuDNN::Layout::ADECB, KuDNN::Layout::ADCB},
    {KuDNN::Layout::BACDE, KuDNN::Layout::BACD},
    {KuDNN::Layout::BCDEA, KuDNN::Layout::BCDA},
    {KuDNN::Layout::CDEAB, KuDNN::Layout::CDAB},
    {KuDNN::Layout::CDEBA, KuDNN::Layout::CDBA},
    {KuDNN::Layout::DECAB, KuDNN::Layout::DCAB}};

NormalizationFlags operator&(const NormalizationFlags &lhs, const NormalizationFlags &rhs)
{
    return static_cast<NormalizationFlags>(static_cast<std::uint32_t>(lhs) & static_cast<std::uint32_t>(rhs));
}

NormalizationFlags operator|(const NormalizationFlags &lhs, const NormalizationFlags &rhs)
{
    return static_cast<NormalizationFlags>(static_cast<std::uint32_t>(lhs) | static_cast<std::uint32_t>(rhs));
}

NormalizationFlags &operator|=(NormalizationFlags &lhs, const NormalizationFlags &rhs)
{
    lhs = lhs | rhs;
    return lhs;
}

NormalizationFlags &operator&=(NormalizationFlags &lhs, const NormalizationFlags &rhs)
{
    lhs = lhs & rhs;
    return lhs;
}

namespace {

KuDNN::Layout GetSrcLayoutWithoutLastDim(const Layout &srcLayout)
{
    auto item = layoutMap.find(srcLayout);
    if (item != layoutMap.end()) {
        return item->second;
    }
    throw Service::LogicError {"LNorm: unknown src layout"};
}

} // namespace

namespace Detail {

// fwd
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

// bwd
static constexpr SizeType LNORM_BWD_ELE_NUM_1 = 10000;
static constexpr SizeType LNORM_BWD_ELE_NUM_2 = 100000;
static constexpr SizeType LNORM_THREADS_BWD_THREASHOLD = 1;
// optNumThreads = c0 + c1 * numElems, 10000 < numElems <= 100000
static constexpr float LNORM_BWD_THREADS1_C0 = 0.87f;
static constexpr float LNORM_BWD_THREADS1_C1 = 0.000042f;
// optNumThreads = c0 + c1 * log(numElems), numElems > 100000
static constexpr float LNORM_BWD_THREADS2_C0 = -65.43f;
static constexpr float LNORM_BWD_THREADS2_C1 = 6.1f;

class NormalizationLayerFWDImpl final : public Parallelizable {
public:
    NormalizationLayerFWDImpl(const TensorInfo &srcInfo, const TensorInfo &statsInfo,
        const TensorInfo &scaleshiftInfo, const TensorInfo &dstInfo, NormalizationFlags flags) noexcept(false)
        : srcType(srcInfo.GetType()), scaleShiftType(scaleshiftInfo.GetType()), dstType(dstInfo.GetType()),
        compType(Element::TypeT::F32),
        /* since all the arrays are broadcasted to 5D, we need to add extra dimensions which were broadcasted */
        axisNum(srcInfo.GetAxisNumAccordingToLayout((srcInfo.GetNumDims() - 1)) + MAX_DIMS - srcInfo.GetNumDims()),
        normFlags(flags),
        reorderedDims(Service::BroadcastTo5D(
            Service::GetShapeAccordingToLayout(srcInfo.GetDims(), srcInfo.GetLayout()))),
        axisSize(reorderedDims[axisNum]),
        innerSize(Service::ArrayProduct(reorderedDims, reorderedDims.GetNumDims(), axisNum + 1)),
        statShape(statsInfo.GetDims()),
        dstShape(dstInfo.GetDims()),
        srcLayout(srcInfo.GetLayout()),
        statLayout(statsInfo.GetLayout()),
        dstLayout(dstInfo.GetLayout())
    {
        // threading implementation depends on the value of calculateStats flag
        // so we initialize it at the constructor stage
        Service::ThrowOnStatus(NormalizationLayerFWD::ValidateInput(srcInfo, statsInfo, scaleshiftInfo,
            dstInfo, flags), "Normalization Layer");

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
    Shape reorderedDims;
    SizeType axisSize;
    SizeType innerSize;
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
            ReorderLayer r({reorderedDims, srcType, Layout::ABCDE}, {reorderedDims, Element::TypeT::F32,
                Layout::ABCDE});
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
            Service::Deallocate(*srcExec);
        }
        if ((srcType != dstType) || (scaleShiftType != dstType)) {
            Service::Deallocate(*scaleExec);
            Service::Deallocate(*shiftExec);
        }
        if (statLayout != GetSrcLayoutWithoutLastDim(srcLayout)) {
            if (calculateStats) {
                const TensorInfo orgStatTensor{statShape, Element::TypeT::F32, GetSrcLayoutWithoutLastDim(srcLayout)};
                const TensorInfo reorderedStatTensor{statShape, Element::TypeT::F32, statLayout};
                ReorderLayer r(orgStatTensor, reorderedStatTensor);
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
        if (!static_cast<bool>(normFlags & NormalizationFlags::RMS_NORM)) {
            if ((static_cast<bool>(normFlags & NormalizationFlags::USE_GLOBAL_STATS) || saveStats) &&
                ((mean == nullptr) || (variance == nullptr))) {
                throw Service::LogicError {"Normalization: one of input matrices is nullptr"};
            }
        } else {
            if ((static_cast<bool>(normFlags & NormalizationFlags::USE_GLOBAL_STATS) || saveStats)
                && (variance == nullptr)) {
                throw Service::LogicError {"Normalization: one of input matrices is nullptr"};
            }
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

class NormalizationLayerBWDImpl final : public Parallelizable {
public:
    NormalizationLayerBWDImpl(const TensorInfo &srcInfo, const TensorInfo &statInfo,
        const TensorInfo &diffSrcInfo, const TensorInfo &diffDstInfo,
        const TensorInfo &scaleShiftInfo, const TensorInfo &diffSc,
        NormalizationFlags flags) noexcept(false) : srcType(srcInfo.GetType()),
        scaleShiftType(scaleShiftInfo.GetType()), dstType(diffDstInfo.GetType()), compType(Element::TypeT::F32),
        /* since all the arrays are broadcasted to 5D, we need to add extra dimensions which were broadcasted */
        axisNum(srcInfo.GetAxisNumAccordingToLayout((srcInfo.GetNumDims() - 1)) + MAX_DIMS - srcInfo.GetNumDims()),
        normFlags(flags),
        reorderedDims(Service::BroadcastTo5D(
            Service::GetShapeAccordingToLayout(srcInfo.GetDims(), srcInfo.GetLayout()))),
        axisSize(reorderedDims[axisNum]),
        outerSize(Service::ArrayProduct(reorderedDims, axisNum, 0)),
        innerSize(Service::ArrayProduct(reorderedDims, reorderedDims.GetNumDims(), axisNum + 1)),
        statShape(statInfo.GetDims()),
        dstShape(diffDstInfo.GetDims()),
        srcLayout(srcInfo.GetLayout()),
        statLayout(statInfo.GetLayout()),
        dstLayout(diffDstInfo.GetLayout())
    {
        Service::ThrowOnStatus(NormalizationLayerBWD::ValidateInput(
            srcInfo, statInfo, diffSrcInfo, diffDstInfo, scaleShiftInfo, diffSc, flags), "Normalization Layer");
        BatchedDims();
        auto initNumThreads = GetRealNumThread();
        Parallelizable::Initialize(initNumThreads, scaleShiftBatchedDims);
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
        RunSequential(batchSplitDims, static_cast<const std::byte *>(src) + strideSrc,
            mean + strideMean, variance + strideMean, // strideVariance is equal strideMean
            static_cast<const std::byte *>(diffDst) + strideDiffDst,
            static_cast<const std::byte *>(scale) + strideScale,
            static_cast<std::byte *>(diffSrc) + strideSrc, eps); // strideDiffSrc is equal strideMean
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
        Service::Unused(numThreads);
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

        RunSequential(batchSplitDims, static_cast<const std::byte *>(src) + strideSrc,
            mean + strideMean, variance + strideMean, // strideVariance is equal strideMean
            static_cast<const std::byte *>(diffDst) + strideDiffDst,
            static_cast<std::byte *>(diffScale) + strideScale, // strideDiffScale is equal strideScale
            static_cast<std::byte *>(diffShift) + strideScale, // strideDiffShift is equal strideScale
            bufScale + threadIndex * axisSize, bufShift + threadIndex * axisSize, eps);
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
        bool isRMSNORM = static_cast<bool>(normFlags & NormalizationFlags::RMS_NORM);
        bool hasNullptr = (src == nullptr) || (mean == nullptr) || (variance == nullptr) || (diffDst == nullptr) \
            || (diffSrc == nullptr);
        bool hasNullptr1 = (src == nullptr) || (variance == nullptr) || (diffDst == nullptr) || (diffSrc == nullptr);
        bool isScaleAndNoScaleInputed = static_cast<bool>(normFlags & NormalizationFlags::USE_SCALE) && \
            (scale == nullptr);
        bool isShiftAndNoScaleInputed = (static_cast<bool>(normFlags & NormalizationFlags::USE_SHIFT) && \
            (scale == nullptr)) && (diffShift == nullptr);
        if (!isRMSNORM) {
            if (hasNullptr) {
                throw Service::LogicError {"Normalization Layer: one of input matrices is nullptr"};
            }
        } else {
            if (hasNullptr1) {
                throw Service::LogicError {"Normalization Layer: one of input matrices is nullptr"};
            }
        }
        if (isScaleAndNoScaleInputed) {
            throw Service::LogicError {"Normalization Layer: one of input matrices is nullptr"};
        }
        if (isShiftAndNoScaleInputed) {
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
        bool isSrcTpNotEqDstTp = srcType != dstType;
        bool isScaleShiftNotTpEqDstTp = scaleShiftType != dstType;
        BeforeRunReorderLayer(&srcExec, &scaleExec, &diffSrcExec, &diffScaleExec, &diffShiftExec,
            &meanExec, &varianceExec, &diffDstExec, src, mean, variance, diffDst, scale,
            diffScale, diffShift, isSrcTpNotEqDstTp, isScaleShiftNotTpEqDstTp);
        auto mainNumThreads = GetRealNumThread();
        if ((diffScale != nullptr) || (diffShift != nullptr)) {
            std::vector<float, Service::AlignedAllocator<float>> tmpDiffScale;
            std::vector<float, Service::AlignedAllocator<float>> tmpDiffShift;
            if (static_cast<bool>(normFlags & NormalizationFlags::USE_SCALE)) {
                tmpDiffScale.resize(axisSize * mainNumThreads, 0.0f);
            }
            if (static_cast<bool>(normFlags & NormalizationFlags::USE_SHIFT)) {
                tmpDiffShift.resize(axisSize * mainNumThreads, 0.0f);
            }
            Threading::BatchThreader(*this, srcExec, meanExec, varianceExec, diffDstExec,
                diffScaleExec, diffShiftExec, tmpDiffScale.data(), tmpDiffShift.data(), eps);
            RunSequential(diffScaleExec, diffShiftExec, tmpDiffScale.data(), tmpDiffShift.data(), mainNumThreads);
        }
        Threading::BatchThreader(*this, srcExec, meanExec, varianceExec, diffDstExec, scaleExec,
            diffSrcExec, eps);
        AfterRunReorderLayer(&srcExec, &scaleExec, &diffSrcExec, &diffScaleExec, &diffShiftExec,
            &meanExec, &varianceExec, &diffDstExec, diffSrc, mean, variance, scale, diffScale,
            diffShift, isSrcTpNotEqDstTp, isScaleShiftNotTpEqDstTp);
    }

private:
    Element::Type srcType;
    Element::Type scaleShiftType;
    Element::Type dstType;
    Element::Type compType;
    Shape scaleShiftBatchedDims;
    SizeType axisNum;
    NormalizationFlags normFlags;
    Shape reorderedDims;
    SizeType axisSize;
    SizeType outerSize;
    SizeType innerSize;
    Shape statShape;
    Shape dstShape;
    Layout srcLayout;
    Layout statLayout;
    Layout dstLayout;

    void BeforeRunReorderLayer(const void **srcExec, const void **scaleExec, void **diffSrcExec,
        void **diffScaleExec, void **diffShiftExec, const float **meanExec, const float **varianceExec,
        const void **diffDstExec, const void *src, const float *mean, const float *variance,
        const void *diffDst, const void *scale, void *diffScale, void *diffShift, bool isSrcTpNotEqDstTp,
        bool isScaleShiftNotTpEqDstTp) noexcept(false)
    {
        if (isSrcTpNotEqDstTp || isScaleShiftNotTpEqDstTp) {
            void *tmp = Service::AlignedAlloc(reorderedDims.GetTotalDimsSize() * sizeof(float));
            ReorderLayer r({reorderedDims, srcType, Layout::ABCDE}, {reorderedDims, Element::TypeT::F32,
                Layout::ABCDE});
            r.Run(src, tmp);
            *srcExec = tmp;
        }
        if ((isSrcTpNotEqDstTp || isScaleShiftNotTpEqDstTp) && scale) {
            void *tmp = Service::AlignedAlloc(reorderedDims[axisNum] * sizeof(float));
            ReorderLayer r({{axisSize}, scaleShiftType, Layout::A}, {{axisSize}, Element::TypeT::F32, Layout::A});
            r.Run(scale, tmp);
            *scaleExec = tmp;
        }

        if (isSrcTpNotEqDstTp || isScaleShiftNotTpEqDstTp) {
            *diffSrcExec = Service::AlignedAlloc(reorderedDims.GetTotalDimsSize() * sizeof(float));
            if (diffScale != nullptr) {
                *diffScaleExec = Service::AlignedAlloc(axisSize * sizeof(float));
            }
            if (diffShift != nullptr) {
                *diffShiftExec = Service::AlignedAlloc(axisSize * sizeof(float));
            }
        }
        if (statLayout != GetSrcLayoutWithoutLastDim(srcLayout)) {
            const TensorInfo orgStatTensor{statShape, Element::TypeT::F32, statLayout};
            const TensorInfo reorderedStatTensor{statShape, Element::TypeT::F32, GetSrcLayoutWithoutLastDim(srcLayout)};
            ReorderLayer r(orgStatTensor, reorderedStatTensor);
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
            ((isSrcTpNotEqDstTp || isScaleShiftNotTpEqDstTp) && (dstType != Element::TypeT::F32))) {
            bool dt = ((isSrcTpNotEqDstTp || isScaleShiftNotTpEqDstTp) && (dstType != Element::TypeT::F32));
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
        const void *scale, void *diffScale, void *diffShift, bool isSrcTpNotEqDstTp,
        bool isScaleShiftNotTpEqDstTp) noexcept(false)
    {
        if ((dstLayout != srcLayout) ||
            ((isSrcTpNotEqDstTp || isScaleShiftNotTpEqDstTp) && (dstType != Element::TypeT::F32))) {
            Service::Deallocate(*diffDstExec);
        }
        if (isSrcTpNotEqDstTp || isScaleShiftNotTpEqDstTp) {
            ReorderLayer r({reorderedDims, Element::TypeT::F32, Layout::ABCDE}, {reorderedDims, srcType,
                Layout::ABCDE});
            r.Run(*diffSrcExec, diffSrc);
            Service::Deallocate(*diffSrcExec);
        }
        if (isSrcTpNotEqDstTp || isScaleShiftNotTpEqDstTp) {
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
                Service::Deallocate(*scaleExec);
            }
        }
        if (statLayout != GetSrcLayoutWithoutLastDim(srcLayout)) {
            if (mean != nullptr) {
                Service::Deallocate(*meanExec);
            }
            if (variance != nullptr) {
                Service::Deallocate(*varianceExec);
            }
        }
        if (isSrcTpNotEqDstTp || isScaleShiftNotTpEqDstTp) {
            Service::Deallocate(*srcExec);
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

    IntType GetRealNumThread() noexcept
    {
        IntType&& needGetOptNumThreads = Threading::ShouldCalculateOptThreads();
        IntType&& mainNumThreads = needGetOptNumThreads ? GetOptNumThreads() : Threading::GetMaxNumThreads();
        return mainNumThreads;
    }

    void BatchedDims() noexcept
    {
        if (axisNum == 0) {
            scaleShiftBatchedDims = {1};
        } else {
            std::array<SizeType, MAX_DIMS> batchedDIms;
            for (SizeType i = 0; i < axisNum; ++i) {
                batchedDIms[i] = reorderedDims[i];
            }
            scaleShiftBatchedDims = {Shape{batchedDIms.data(), axisNum}.GetTotalDimsSize()};
        }
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

} // namespace Detail

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
    NormalizationFlags flags) noexcept
{
    if (static_cast<bool>(flags & NormalizationFlags::FUSE_NORM_RELU)) {
        return Status::UNSUPPORTED;
    }
    if (srcInfo.GetDims() != dstInfo.GetDims()) {
        return Status::BAD_ARGUMENTS;
    }
    if (((static_cast<bool>(flags & NormalizationFlags::USE_SCALE)) ||
        (static_cast<bool>(flags & NormalizationFlags::USE_SHIFT))) &&
        (scaleshiftInfo.GetType() != Element::TypeT::F32) &&
        (scaleshiftInfo.GetType() != Element::TypeT::F16) &&
        (scaleshiftInfo.GetType() != Element::TypeT::BF16)) {
        return Status::UNSUPPORTED;
    }
    SizeType srcDims = srcInfo.GetNumDims();
    if ((srcDims - 1) != statsInfo.GetNumDims() || scaleshiftInfo.GetNumDims() != 1) {
        return Status::BAD_ARGUMENTS;
    }
    if (srcInfo.GetTotalTensorSize() == 0) {
        return Status::UNSUPPORTED;
    }
    if (srcInfo.GetDims()[srcDims - 1] != scaleshiftInfo.GetDims()[0]) {
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
    bool inputTypeSupported = (srcInfo.GetType() == Element::TypeT::S8) || (srcInfo.GetType() == Element::TypeT::U8) ||
        (srcInfo.GetType() == Element::TypeT::S32) || (diffDstInfo.GetType() == Element::TypeT::S8) ||
        (diffDstInfo.GetType() == Element::TypeT::U8) || (diffDstInfo.GetType() == Element::TypeT::S32);
    if (inputTypeSupported) {
        return Status::UNSUPPORTED;
    }
    bool inputHasInt = ((static_cast<bool>(flags & NormalizationFlags::USE_SCALE)) ||
        (static_cast<bool>(flags & NormalizationFlags::USE_SHIFT))) &&
        (scaleShiftInfo.GetType() != Element::TypeT::F32) &&
        (scaleShiftInfo.GetType() != Element::TypeT::F16) &&
        (scaleShiftInfo.GetType() != Element::TypeT::BF16);
    if (inputHasInt) {
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

} // namespace KuDNN
