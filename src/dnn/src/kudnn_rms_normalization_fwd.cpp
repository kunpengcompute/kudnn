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
#include <map>
#include <arm_sve.h>

#include "kudnn.hpp"
#include "threading_impl.hpp"
#include "internal_funcs.hpp"
#include "kudnn_rms_normalization_impl.hpp"
#include "kernels/rmsnorm/kudnn_rmsnorm_kernels.hpp"

namespace KuDNN {
namespace Detail {
static const std::map<KuDNN::Layout, KuDNN::Layout> layoutMap = {
    {KuDNN::Layout::AB, KuDNN::Layout::A},       {KuDNN::Layout::BA, KuDNN::Layout::A},
    {KuDNN::Layout::ABC, KuDNN::Layout::AB},     {KuDNN::Layout::ACB, KuDNN::Layout::AB},
    {KuDNN::Layout::BAC, KuDNN::Layout::BA},     {KuDNN::Layout::BCA, KuDNN::Layout::BA},
    {KuDNN::Layout::CAB, KuDNN::Layout::AB},     {KuDNN::Layout::CBA, KuDNN::Layout::BA},
    {KuDNN::Layout::ABCD, KuDNN::Layout::ABC},   {KuDNN::Layout::ABDC, KuDNN::Layout::ABC},
    {KuDNN::Layout::ACBD, KuDNN::Layout::ACB},   {KuDNN::Layout::ACDB, KuDNN::Layout::ACB},
    {KuDNN::Layout::ADBC, KuDNN::Layout::ABC},   {KuDNN::Layout::ADCB, KuDNN::Layout::ACB},
    {KuDNN::Layout::BACD, KuDNN::Layout::BAC},   {KuDNN::Layout::BCDA, KuDNN::Layout::BCA},
    {KuDNN::Layout::CDAB, KuDNN::Layout::CAB},   {KuDNN::Layout::CDBA, KuDNN::Layout::CBA},
    {KuDNN::Layout::DCAB, KuDNN::Layout::CAB},   {KuDNN::Layout::ABCDE, KuDNN::Layout::ABCD},
    {KuDNN::Layout::ABCED, KuDNN::Layout::ABCD}, {KuDNN::Layout::ABDEC, KuDNN::Layout::ABDC},
    {KuDNN::Layout::ACBDE, KuDNN::Layout::ACBD}, {KuDNN::Layout::ACDEB, KuDNN::Layout::ACDB},
    {KuDNN::Layout::ADECB, KuDNN::Layout::ADCB}, {KuDNN::Layout::BACDE, KuDNN::Layout::BACD},
    {KuDNN::Layout::BCDEA, KuDNN::Layout::BCDA}, {KuDNN::Layout::CDEAB, KuDNN::Layout::CDAB},
    {KuDNN::Layout::CDEBA, KuDNN::Layout::CDBA}, {KuDNN::Layout::DECAB, KuDNN::Layout::DCAB}};

static KuDNN::Layout GetSrcLayoutWithoutLastDim(const Layout &srcLayout)
{
    auto item = layoutMap.find(srcLayout);
    if (item != layoutMap.end()) {
        return item->second;
    }
    throw Service::LogicError{"LNorm: unknown src layout"};
}

static constexpr SizeType RMSNORM_FWD_ELE_NUM_1 = 10000;
static constexpr SizeType RMSNORM_FWD_ELE_NUM_2 = 100000;

static constexpr SizeType RMSNORM_FWD_THREADS_THREASHOLD = 1;

// optNumThreads = c0 + c1 * log(numElems), 10000 < numElems <= 100000
static constexpr float RMSNORM_FWD_THREADS_CALCSTATS1_C0 = -30.25f;
static constexpr float RMSNORM_FWD_THREADS_CALCSTATS1_C1 = 3.68f;
// optNumThreads = c0 + c1 * log(numElems), numElems > 100000
static constexpr float RMSNORM_FWD_THREADS_CALCSTATS2_C0 = -170.99f;
static constexpr float RMSNORM_FWD_THREADS_CALCSTATS2_C1 = 15.86f;

// optNumThreads = c0 + c1 * numElems, 10000 < numElems <= 100000
static constexpr float RMSNORM_FWD_THREADS_GLOBALSTATS1_C0 = -0.19f;
static constexpr float RMSNORM_FWD_THREADS_GLOBALSTATS1_C1 = 0.00012f;
// optNumThreads = c0 + c1 * log(numElems), numElems > 100000
static constexpr float RMSNORM_FWD_THREADS_GLOBALSTATS2_C0 = -149.24f;
static constexpr float RMSNORM_FWD_THREADS_GLOBALSTATS2_C1 = 13.49f;

class RMSNormalizationLayerFWDImpl final : public Parallelizable {
public:
    RMSNormalizationLayerFWDImpl(const TensorInfo &srcInfo, const TensorInfo &statsInfo, const TensorInfo &scaleInfo,
                                 const TensorInfo &dstInfo, NormalizationFlags flags) noexcept(false)
        : srcType(srcInfo.GetType()),
          scaleType(scaleInfo.GetType()),
          dstType(dstInfo.GetType()),
          compType(Element::TypeT::F32),
          /* since all the arrays are broadcasted to 5D, we need to add extra dimensions which were broadcasted */
          axisNum(srcInfo.GetAxisNumAccordingToLayout((srcInfo.GetNumDims() - 1)) + MAX_DIMS - srcInfo.GetNumDims()),
          normFlags(flags)
    {
        // threading implementation depends on the value of calculateStats flag
        // so we initialize it at the constructor stage
        Service::ThrowOnStatus(RMSNormalizationLayerFWD::ValidateInput(srcInfo, statsInfo, scaleInfo, dstInfo, flags),
                               "Normalization Layer");

        reorderedDims =
            Service::BroadcastTo5D(Service::GetShapeAccordingToLayout(srcInfo.GetDims(), srcInfo.GetLayout()));
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
    void Evaluate(const Shape &batchSplitDims, const void *src, void *dst, const void *scale, float *variance,
                  bool saveStats, const float eps) const noexcept
    {
        const InType *typedSrc = static_cast<const InType *>(src);
        OutType *typedDst = static_cast<OutType *>(dst);
        const OutType *typedScale = static_cast<const OutType *>(scale);
        ChooseImpl<CompType>(batchSplitDims, typedSrc, typedDst, typedScale, variance, saveStats, eps);
    }

    void RunSequential(const Shape &batchSplitDims, const void *src, void *dst, const void *scale, float *variance,
                       bool saveStats, const float eps) const noexcept
    {
        if ((srcType != dstType) || (scaleType != dstType)) {
            Service::ParseType<true, false>(compType, compType, Element::TypeT::F32, *this, batchSplitDims, src, dst,
                                            scale, variance, saveStats, eps);
        } else {
            Service::ParseType<true, false>(compType, compType, dstType, *this, batchSplitDims, src, dst, scale,
                                            variance, saveStats, eps);
        }
    }

    void RunSequential(const Shape &batchSplitDims, const Shape &batchStrides, const void *src, void *dst,
                       const void *scale, float *variance, bool saveStats, const float eps) const noexcept(false)
    {
        SizeType strideSrc = 0;
        SizeType strideDst = 0;
        SizeType strideScale = 0;
        SizeType strideVariance = 0;
        SizeType elementSize = dstType.GetSize();
        if ((srcType != dstType) || (scaleType != dstType)) {
            elementSize = sizeof(float);
        }
        for (SizeType i = 0; i < batchStrides.GetNumDims(); ++i) {
            strideSrc += batchStrides[i] * elementSize * (axisSize * innerSize);
            strideDst += batchStrides[i] * elementSize * (axisSize * innerSize);
            strideVariance += batchStrides[i] * innerSize;
        }
        if ((!static_cast<bool>(normFlags & NormalizationFlags::USE_GLOBAL_STATS)) && (!saveStats)) {
            strideVariance = 0;
        }

        RunSequential(batchSplitDims, static_cast<const std::byte *>(src) + strideSrc,
                      static_cast<std::byte *>(dst) + strideDst, static_cast<const std::byte *>(scale) + strideScale,
                      variance + strideVariance, saveStats, eps);
    }

    void Run(const void *src, void *dst, const void *scale, float *variance, bool saveStats, const float eps) const
        noexcept(false)
    {
        CheckInputArg(src, dst, scale, variance, saveStats);
        const void *srcExec = src;
        const void *scaleExec = scale;
        bool calculateStats = !static_cast<bool>(normFlags & NormalizationFlags::USE_GLOBAL_STATS);
        float *varianceExec = variance;
        void *dstExec = dst;

        BeforeRunReorderLayer(&srcExec, &scaleExec, &varianceExec, &dstExec, src, scale, variance);

        Threading::BatchThreader(*this, srcExec, dstExec, scaleExec, varianceExec, saveStats, eps);

        AfterRunReorderLayer(&srcExec, &scaleExec, &varianceExec, &dstExec, calculateStats, variance, dst);
    }

private:
    Element::Type srcType;
    Element::Type scaleType;
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

    void BeforeRunReorderLayer(const void **srcExec, const void **scaleExec, float **varianceExec, void **dstExec,
                               const void *src, const void *scale, float *variance) const noexcept(false)
    {
        if ((srcType != dstType) || (scaleType != dstType)) {
            void *tmp = Service::AlignedAlloc(reorderedDims.GetTotalDimsSize() * sizeof(float));
            ReorderLayer r({reorderedDims, srcType, Layout::ABCDE},
                           {reorderedDims, Element::TypeT::F32, Layout::ABCDE});
            r.Run(src, tmp);
            *srcExec = tmp;
        }
        if ((srcType != dstType) || (scaleType != dstType)) {
            ReorderLayer r({{axisSize}, scaleType, Layout::A}, {{axisSize}, Element::TypeT::F32, Layout::A});
            void *tmp;
            if (scale != nullptr) {
                tmp = Service::AlignedAlloc(axisSize * sizeof(float));
                r.Run(scale, tmp);
                *scaleExec = tmp;
            }
        }
        if (statLayout != GetSrcLayoutWithoutLastDim(srcLayout)) {
            ReorderLayer r({statShape, Element::TypeT::F32, statLayout},
                           {statShape, Element::TypeT::F32, GetSrcLayoutWithoutLastDim(srcLayout)});
            void *tmp;
            if (variance != nullptr) {
                tmp = Service::AlignedAlloc(statShape.GetTotalDimsSize() * sizeof(float));
                r.Run(variance, tmp);
                *varianceExec = static_cast<float *>(tmp);
            }
        }
        bool isNotFullMatch = (dstLayout != srcLayout) ||
                              (((srcType != dstType) || (scaleType != dstType)) && (dstType != Element::TypeT::F32));
        if (isNotFullMatch) {
            bool dt = (((srcType != dstType) || (scaleType != dstType)) && (dstType != Element::TypeT::F32));
            *dstExec =
                Service::AlignedAlloc(reorderedDims.GetTotalDimsSize() * (dt ? sizeof(float) : dstType.GetSize()));
        }
    }

    void AfterRunReorderLayer(const void **srcExec, const void **scaleExec, float **varianceExec, void **dstExec,
                              bool calculateStats, float *variance, void *dst) const noexcept(false)
    {
        if ((srcType != dstType) || (scaleType != dstType)) {
            Service::Deallocate(const_cast<void *>(*srcExec));
            Service::Deallocate(const_cast<void *>(*scaleExec));
        }
        if (statLayout != GetSrcLayoutWithoutLastDim(srcLayout)) {
            if (calculateStats) {
                ReorderLayer r({statShape, Element::TypeT::F32, GetSrcLayoutWithoutLastDim(srcLayout)},
                               {statShape, Element::TypeT::F32, statLayout});
                if (variance != nullptr) {
                    r.Run(*varianceExec, variance);
                }
            }
            if (variance != nullptr) {
                Service::Deallocate(*varianceExec);
            }
        }
        bool isNotFullMatch = (dstLayout != srcLayout) ||
                              (((srcType != dstType) || (scaleType != dstType)) && (dstType != Element::TypeT::F32));
        if (isNotFullMatch) {
            bool dt = (((srcType != dstType) || (scaleType != dstType)) && (dstType != Element::TypeT::F32));
            ReorderLayer r({dstShape, dt ? Element::Type{Element::TypeT::F32} : dstType, srcLayout},
                           {dstShape, dstType, dstLayout});
            r.Run(*dstExec, dst);
            Service::Deallocate(*dstExec);
        }
    }

    void CheckInputArg(const void *src, void *dst, const void *scale, float *variance, bool saveStats) const
    {
        if ((src == nullptr) || (dst == nullptr)) {
            throw Service::LogicError{"Normalization: one of input matrices is nullptr"};
        }
        if ((static_cast<bool>(normFlags & NormalizationFlags::USE_GLOBAL_STATS) || saveStats) &&
            (variance == nullptr)) {
            throw Service::LogicError{"Normalization: one of input matrices is nullptr"};
        }
        if (static_cast<bool>(normFlags & NormalizationFlags::USE_SCALE) && (scale == nullptr)) {
            throw Service::LogicError{"Normalization Layer: one of input matrices is nullptr"};
        }
    }

    SizeType GetOptNumThreads() const noexcept
    {
        SizeType totalSize = reorderedDims.GetTotalDimsSize();
        if (!static_cast<bool>(normFlags & NormalizationFlags::USE_GLOBAL_STATS)) {
            if (totalSize < RMSNORM_FWD_ELE_NUM_1) {
                return RMSNORM_FWD_THREADS_THREASHOLD;
            } else if (totalSize <= RMSNORM_FWD_ELE_NUM_2) {
                return RMSNORM_FWD_THREADS_CALCSTATS1_C0 + RMSNORM_FWD_THREADS_CALCSTATS1_C1 * std::log(totalSize);
            } else {
                return RMSNORM_FWD_THREADS_CALCSTATS2_C0 + RMSNORM_FWD_THREADS_CALCSTATS2_C1 * std::log(totalSize);
            }
        } else {
            if (totalSize < RMSNORM_FWD_ELE_NUM_1) {
                return RMSNORM_FWD_THREADS_THREASHOLD;
            } else if (totalSize <= RMSNORM_FWD_ELE_NUM_2) {
                return RMSNORM_FWD_THREADS_GLOBALSTATS1_C0 + RMSNORM_FWD_THREADS_GLOBALSTATS1_C1 * totalSize;
            } else {
                return RMSNORM_FWD_THREADS_GLOBALSTATS2_C0 + RMSNORM_FWD_THREADS_GLOBALSTATS2_C1 * std::log(totalSize);
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

    template <typename CompType, typename InType, typename OutType, typename ScaleType>
    void ChooseImpl(const Shape &batchSplitDims, const InType *src, OutType *dst, const ScaleType *scale,
                    float *variance, bool saveStats, const float eps) const noexcept
    {
        SizeType outerSize = batchSplitDims.GetTotalDimsSize();
        if (axisNum == (reorderedDims.GetNumDims() - 1)) {
            RMSNorm::EvaluateLastAxis<CompType>(outerSize, src, dst, scale, variance, saveStats, eps, normFlags,
                                                axisSize);
        } else {
            RMSNorm::EvaluateCrossAxis<CompType>(outerSize, src, dst, scale, variance, saveStats, eps, normFlags,
                                                 axisSize, innerSize);
            return;
        }
    }
};

} // namespace Detail

RMSNormalizationLayerFWD::~RMSNormalizationLayerFWD() noexcept {}

RMSNormalizationLayerFWD::RMSNormalizationLayerFWD(const TensorInfo &srcInfo, const TensorInfo &statsInfo,
                                                   const TensorInfo &scaleInfo, const TensorInfo &dstInfo,
                                                   NormalizationFlags flags) noexcept(false)
    : pImpl(new Detail::RMSNormalizationLayerFWDImpl(srcInfo, statsInfo, scaleInfo, dstInfo, flags))
{
}

RMSNormalizationLayerFWD::RMSNormalizationLayerFWD(const RMSNormalizationLayerFWD &other) noexcept(false)
    : pImpl(new Detail::RMSNormalizationLayerFWDImpl{*(other.pImpl.get())})
{
}

RMSNormalizationLayerFWD::RMSNormalizationLayerFWD(RMSNormalizationLayerFWD &&other) noexcept
    : pImpl(std::move(other.pImpl))
{
}

RMSNormalizationLayerFWD &RMSNormalizationLayerFWD::operator=(const RMSNormalizationLayerFWD &other) noexcept(false)
{
    if (this != &other) {
        this->pImpl.reset(new Detail::RMSNormalizationLayerFWDImpl{*(other.pImpl.get())});
    }
    return *this;
}

RMSNormalizationLayerFWD &RMSNormalizationLayerFWD::operator=(RMSNormalizationLayerFWD &&other) noexcept
{
    if (this != &other) {
        this->pImpl = std::move(other.pImpl);
    }
    return *this;
}

void RMSNormalizationLayerFWD::Run(const void *src, void *dst, const void *scale, float *variance, bool saveStats,
                                   const float eps) const noexcept(false)
{
    pImpl->Run(src, dst, scale, variance, saveStats, eps);
}

Status RMSNormalizationLayerFWD::ValidateInput(const TensorInfo &srcInfo, const TensorInfo &statsInfo,
                                               const TensorInfo &scaleInfo, const TensorInfo &dstInfo,
                                               NormalizationFlags flags) noexcept(false)
{
    if (static_cast<bool>(flags & NormalizationFlags::FUSE_NORM_RELU)) {
        return Status::UNSUPPORTED;
    }
    if (srcInfo.GetDims() != dstInfo.GetDims()) {
        return Status::BAD_ARGUMENTS;
    }
    if (srcInfo.GetType() != Element::TypeT::F16 || dstInfo.GetType() != Element::TypeT::F16) {
        return Status::UNSUPPORTED;
    }
    if ((static_cast<bool>(flags & NormalizationFlags::USE_SCALE)) && (scaleInfo.GetType() != Element::TypeT::F16)) {
        return Status::UNSUPPORTED;
    }
    if ((srcInfo.GetNumDims() - 1) != statsInfo.GetNumDims()) {
        return Status::BAD_ARGUMENTS;
    }
    if (srcInfo.GetTotalTensorSize() == 0) {
        return Status::UNSUPPORTED;
    }
    if (scaleInfo.GetNumDims() != 1) {
        return Status::BAD_ARGUMENTS;
    }
    if (srcInfo.GetDims()[srcInfo.GetNumDims() - 1] != scaleInfo.GetDims()[0]) {
        return Status::BAD_ARGUMENTS;
    }
    for (SizeType i = 0; i < statsInfo.GetNumDims(); ++i) {
        if (srcInfo.GetDims()[i] != statsInfo.GetDims()[i]) {
            return Status::BAD_ARGUMENTS;
        }
    }
    if (!srcInfo.IsDense() || !statsInfo.IsDense() || !scaleInfo.IsDense() || !dstInfo.IsDense()) {
        return Status::UNSUPPORTED;
    }
    return Status::SUCCESS;
}

} // namespace KuDNN
