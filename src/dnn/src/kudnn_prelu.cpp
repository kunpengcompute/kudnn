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

#include <algorithm>
#include <memory>

#include <arm_sve.h>
#include <arm_neon.h>

#include "kudnn.hpp"

#include "internal_funcs.hpp"
#include "threading_impl.hpp"
#include "vector_wrappers.hpp"
#include "kudnn_prelu_impl.hpp"
#include "kernels/prelu/kudnn_prelu_kernels.hpp"

namespace KuDNN {
namespace Detail {
namespace PReLU {

BroadcastType GetBCType(const TensorInfo &srcInfo, const TensorInfo &weightsInfo)
{
    if (srcInfo.GetDims() == weightsInfo.GetDims()) {
        return BroadcastType::WHOLE_TENSOR;
    }
    if (weightsInfo.GetTotalTensorSize() == 1) {
        return BroadcastType::CHANNEL_SHARED;
    }
    return BroadcastType::SHARED_AXES;
}

SizeType GetLastOneIdx(Shape reorderedWeiDims)
{
    SizeType lastNoneOneIdx = reorderedWeiDims.GetNumDims() - 1;
    for (SizeType i = 0; i < reorderedWeiDims.GetNumDims(); ++i) {
        if (reorderedWeiDims[reorderedWeiDims.GetNumDims() - 1 - i] == 1) {
            lastNoneOneIdx = reorderedWeiDims.GetNumDims() - 1 - i;
        } else {
            break;
        }
    }
    return lastNoneOneIdx;
}

} // PReLU
} // Detail
} // KuDNN

namespace KuDNN {
namespace Detail {

static constexpr SizeType PRELU_FWD_ELE_NUM_1 = 300000;
static constexpr SizeType PRELU_FWD_ELE_NUM_STEP1 = 10000;
static constexpr SizeType PRELU_FWD_ELE_NUM_2 = 3000000;
static constexpr SizeType PRELU_FWD_ELE_NUM_3 = 100000000;

static constexpr SizeType PRELU_FWD_THREADS_THREASHOLD1 = 20;
static constexpr SizeType PRELU_FWD_THREADS_THREASHOLD2 = 40;
static constexpr SizeType PRELU_FWD_THREADS_THREASHOLD3 = 80;
static constexpr SizeType PRELU_FWD_THREADS_THREASHOLD4 = 160;

class PReLULayerFWDImpl final : public Parallelizable {
public:
    PReLULayerFWDImpl(const TensorInfo &srcInfo, const TensorInfo &weightsInfo,
        const TensorInfo &dstInfo) noexcept(false)
        : srcType(srcInfo.GetType()), weiType(weightsInfo.GetType()), dstType(dstInfo.GetType()),
        compType(Element::TypeT::F32), broadcastType(PReLU::GetBCType(srcInfo, weightsInfo)),
        weiInfo(weightsInfo), lastScalar(false)
    {
        Service::ThrowOnStatus(PReLULayerFWD::ValidateInput(srcInfo, weightsInfo, dstInfo), "PReLU Layer");
        diffLayouts = srcInfo.GetLayout() != weiInfo.GetLayout();
        if (diffLayouts) {
            ReorderDimsDiffLayout(srcInfo, weightsInfo, dstInfo);
        } else {
            ReorderDimsSameLayout(srcInfo, weightsInfo, dstInfo);
        }
        needToConvert = (((srcType != weiType) || (srcType != dstType)));
        Parallelizable::Initialize(GetOptNumThreads(), reorderedDims);
    }

    void ReorderDimsDiffLayout(const TensorInfo &srcInfo, const TensorInfo &weightsInfo,
        const TensorInfo &dstInfo) noexcept(false)
    {
        reorderedDims = Service::BroadcastTo5D(Service::GetShapeAccordingToLayout(dstInfo.GetDims(),
            srcInfo.GetLayout()));
        reorderedStridesSrc = Service::BroadcastTo5D(Service::GetShapeAccordingToLayout(srcInfo.GetStrides(),
            srcInfo.GetLayout()), false);
        reorderedStridesWei = Service::BroadcastTo5D(Service::GetShapeAccordingToLayout(weightsInfo.GetStrides(),
            srcInfo.GetLayout()), false);
        reorderedStridesDst  = Service::BroadcastTo5D(Service::GetShapeAccordingToLayout(dstInfo.GetStrides(),
            srcInfo.GetLayout()), false);
    }

    void ReorderDimsSameLayout(const TensorInfo &srcInfo, const TensorInfo &weightsInfo,
        const TensorInfo &dstInfo) noexcept(false)
    {
        reorderedDims = Service::BroadcastTo5D(Service::GetShapeAccordingToLayout(dstInfo.GetDims(),
            dstInfo.GetLayout()));
        reorderedStridesSrc = Service::BroadcastTo5D(Service::GetShapeAccordingToLayout(srcInfo.GetStrides(),
            srcInfo.GetLayout()), false);
        reorderedStridesWei = Service::BroadcastTo5D(Service::GetShapeAccordingToLayout(weightsInfo.GetStrides(),
            weightsInfo.GetLayout()), false);
        reorderedStridesDst  = Service::BroadcastTo5D(Service::GetShapeAccordingToLayout(dstInfo.GetStrides(),
            dstInfo.GetLayout()), false);

        Shape reorderedWeiDims = Service::BroadcastTo5D(Service::GetShapeAccordingToLayout(weightsInfo.GetDims(),
            weightsInfo.GetLayout()));
        if (broadcastType == PReLU::BroadcastType::SHARED_AXES) {
            SizeType lastNoneOneIdx = PReLU::GetLastOneIdx(reorderedWeiDims);
            if (lastNoneOneIdx != (reorderedWeiDims.GetNumDims() - 1)) {
                SizeType tempShapeSrc[MAX_DIMS];
                SizeType tempStridesSrc[MAX_DIMS];
                SizeType tempStridesWei[MAX_DIMS];
                SizeType tempStridesDst[MAX_DIMS];
                for (SizeType i = 0; i < lastNoneOneIdx; ++i) {
                    tempShapeSrc[i] = reorderedDims[i];
                    tempStridesSrc[i] = reorderedStridesSrc[i];
                    tempStridesWei[i] = reorderedStridesWei[i];
                    tempStridesDst[i] = reorderedStridesDst[i];
                }
                tempShapeSrc[lastNoneOneIdx] = Service::ArrayProduct(reorderedDims, reorderedDims.GetNumDims(),
                    lastNoneOneIdx);
                tempStridesSrc[lastNoneOneIdx] = 1;
                tempStridesWei[lastNoneOneIdx] = 0;
                tempStridesDst[lastNoneOneIdx] = 1;

                reorderedDims = Service::BroadcastTo5D(Shape(tempShapeSrc, lastNoneOneIdx + 1));
                reorderedStridesSrc = Service::BroadcastTo5D(Shape(tempStridesSrc, lastNoneOneIdx + 1), false);
                reorderedStridesWei = Service::BroadcastTo5D(Shape(tempStridesWei, lastNoneOneIdx + 1), false);
                reorderedStridesDst  = Service::BroadcastTo5D(Shape(tempStridesDst, lastNoneOneIdx + 1), false);
            }
        }
        lastScalar = reorderedWeiDims[reorderedWeiDims.GetNumDims() - 1] == 1;
    }

    template <typename CompType, typename SrcType, typename WeiType, typename OutType>
    void Evaluate(const Shape &batchSplitDims, const void *src, const void *wei, void *dst) const noexcept
    {
        const SrcType *typedSrc = static_cast<const SrcType *>(src);
        const WeiType *typedwei = static_cast<const WeiType *>(wei);
        OutType *typedDst = static_cast<OutType *>(dst);
        if (diffLayouts) {
            PReLU::KernelGeneric<CompType>(batchSplitDims, typedSrc, typedwei, typedDst,
                reorderedStridesSrc, reorderedStridesWei, reorderedStridesDst);
        } else {
            switch (broadcastType) {
                case PReLU::BroadcastType::WHOLE_TENSOR: {
                        PReLU::KernelNonSharedAxesLoad<CompType>(batchSplitDims, typedSrc, typedwei, typedDst);
                        break;
                }
                case PReLU::BroadcastType::CHANNEL_SHARED: {
                        PReLU::KernelNonSharedAxesDup<CompType>(batchSplitDims, typedSrc, typedwei, typedDst);
                        break;
                }
                default: {
                        if (lastScalar) {
                            PReLU::KernelSharedAxesDup<CompType>(batchSplitDims, typedSrc, typedwei, typedDst,
                                reorderedStridesSrc, reorderedStridesWei, reorderedStridesDst);
                        } else {
                            PReLU::KernelSharedAxesLoad<CompType>(batchSplitDims, typedSrc, typedwei, typedDst,
                                reorderedStridesSrc, reorderedStridesWei, reorderedStridesDst);
                        }
                        break;
                }
            }
        }
    }

    void RunSequential(const Shape &batchSplitDims, const void *src,
        const void *wei, void *dst) const noexcept(false)
    {
        if (needToConvert) {
            Service::ParseTypeAdvanced<true>(compType, Element::TypeT::F32, Element::TypeT::F32, Element::TypeT::F32,
                *this, batchSplitDims, src, wei, dst);
        } else {
            Service::ParseTypeAdvanced<true>(compType, srcType, weiType, dstType, *this, batchSplitDims, src, wei, dst);
        }
    }

    void RunSequential(const Shape &batchSplitDims, const Shape &batchStrides, const void *src,
        const void *wei, void *dst) const noexcept(false)
    {
        SizeType strideSrc = 0;
        SizeType strideWei = 0;
        SizeType strideDst = 0;
        SizeType srcSize = needToConvert ? sizeof(float) : srcType.GetSize();
        SizeType weiSize = needToConvert ? sizeof(float) : weiType.GetSize();
        SizeType dstSize = needToConvert ? sizeof(float) : dstType.GetSize();
        for (SizeType i = 0; i < batchStrides.GetNumDims(); ++i) {
            strideSrc += batchStrides[i] * reorderedStridesSrc[i] * srcSize;
            strideWei += batchStrides[i] * reorderedStridesWei[i] * weiSize;
            strideDst += batchStrides[i] * reorderedStridesDst[i] * dstSize;
        }
        RunSequential(batchSplitDims, static_cast<const std::byte *>(src) + strideSrc,
            static_cast<const std::byte *>(wei) + strideWei, static_cast<std::byte *>(dst) + strideDst);
    }

    void Run(const void *src, const void *wei, void *dst) const noexcept(false)
    {
        if ((src == nullptr) || (wei == nullptr) || (dst == nullptr)) {
            throw Service::LogicError {"PReLU Layer: one of input matrices is nullptr"};
        }
        if ((src == wei) || (wei == dst)) {
            throw Service::LogicError {"PReLU Layer: input/output and weights must not be the same pointers"};
        }
        const void *srcExec = src;
        const void *weiExec = wei;
        void *dstExec = dst;
        if (needToConvert) {
            void *tmp = Service::AlignedAlloc(reorderedDims.GetTotalDimsSize() * sizeof(float));
            ReorderLayer r0({reorderedDims, srcType, Layout::ABCDE, reorderedStridesSrc}, {reorderedDims,
                Element::TypeT::F32, Layout::ABCDE, reorderedStridesSrc});
            r0.Run(src, tmp);
            srcExec = tmp;
            tmp = Service::AlignedAlloc(weiInfo.GetDims().GetTotalDimsSize() * sizeof(float));
            ReorderLayer r1({weiInfo.GetDims(), weiInfo.GetType(), weiInfo.GetLayout(),
                weiInfo.GetStrides()}, {weiInfo.GetDims(), Element::TypeT::F32,
                weiInfo.GetLayout(), weiInfo.GetStrides()});
            r1.Run(wei, tmp);
            weiExec = tmp;
            dstExec = Service::AlignedAlloc(reorderedDims.GetTotalDimsSize() * sizeof(float));
        }
        Threading::BatchThreader(*this, srcExec, weiExec, dstExec);
        if (needToConvert) {
            ReorderLayer r({reorderedDims, Element::TypeT::F32, Layout::ABCDE,
                reorderedStridesDst}, {reorderedDims, dstType, Layout::ABCDE, reorderedStridesDst});
            r.Run(dstExec, dst);
            Service::Deallocate(srcExec);
            Service::Deallocate(weiExec);
            Service::Deallocate(dstExec);
        }
    }

private:
    Element::Type srcType;
    Element::Type weiType;
    Element::Type dstType;
    Element::Type compType;
    PReLU::BroadcastType broadcastType;
    TensorInfo weiInfo;
    bool lastScalar;
    Shape reorderedDims;
    Shape reorderedStridesSrc;
    Shape reorderedStridesWei;
    Shape reorderedStridesDst;
    bool needToConvert;
    bool diffLayouts;

    SizeType GetOptNumThreads() const noexcept
    {
        SizeType totalSrcSize = reorderedDims.GetTotalDimsSize();
        if (totalSrcSize < PRELU_FWD_ELE_NUM_1) {
            return std::min(((totalSrcSize + PRELU_FWD_ELE_NUM_STEP1 - 1) / PRELU_FWD_ELE_NUM_STEP1),
                static_cast<SizeType>(PRELU_FWD_THREADS_THREASHOLD1));
        } else if (totalSrcSize < PRELU_FWD_ELE_NUM_2) {
            return PRELU_FWD_THREADS_THREASHOLD2;
        } else if (totalSrcSize < PRELU_FWD_ELE_NUM_3) {
            return PRELU_FWD_THREADS_THREASHOLD3;
        } else {
            return PRELU_FWD_THREADS_THREASHOLD4;
        }
    }
};

} // Detail

PReLULayerFWD::~PReLULayerFWD() noexcept {}

PReLULayerFWD::PReLULayerFWD(const TensorInfo &srcInfo, const TensorInfo &weightsInfo,
    const TensorInfo &dstInfo) noexcept(false)
    : pImpl(new Detail::PReLULayerFWDImpl(srcInfo, weightsInfo, dstInfo)) {}

PReLULayerFWD::PReLULayerFWD(const PReLULayerFWD &other) noexcept(false)
    : pImpl(new Detail::PReLULayerFWDImpl{*(other.pImpl.get())}) {}

PReLULayerFWD::PReLULayerFWD(PReLULayerFWD &&other) noexcept
    : pImpl(std::move(other.pImpl)) {}

PReLULayerFWD& PReLULayerFWD::operator=(const PReLULayerFWD &other) noexcept(false)
{
    if (this != &other) {
        this->pImpl.reset(new Detail::PReLULayerFWDImpl{*(other.pImpl.get())});
    }
    return *this;
}

PReLULayerFWD& PReLULayerFWD::operator=(PReLULayerFWD &&other) noexcept
{
    if (this != &other) {
        this->pImpl = std::move(other.pImpl);
    }
    return *this;
}

void PReLULayerFWD::Run(const void *src, const void *wei, void *dst) const noexcept(false)
{
    pImpl->Run(src, wei, dst);
}

Status PReLULayerFWD::ValidateInput(const TensorInfo &srcInfo, const TensorInfo &weightsInfo,
    const TensorInfo &dstInfo) noexcept
{
    if (srcInfo.GetLayout() != dstInfo.GetLayout()) {
        return Status::UNSUPPORTED;
    }
    if (!srcInfo.IsDense() || !weightsInfo.IsDense() || !dstInfo.IsDense()) {
        return Status::UNSUPPORTED;
    }
    if (srcInfo.GetDims().GetNumDims() != weightsInfo.GetDims().GetNumDims()) {
        return Status::BAD_ARGUMENTS;
    }
    for (SizeType i = 0; i < srcInfo.GetDims().GetNumDims(); ++i) {
        if (((srcInfo.GetDims()[i] != weightsInfo.GetDims()[i]) &&
            (weightsInfo.GetDims()[i] != 1)) || (srcInfo.GetDims()[i] != dstInfo.GetDims()[i])) {
            return Status::BAD_ARGUMENTS;
        }
    }
    if (srcInfo.GetTotalTensorSize() == 0 || dstInfo.GetTotalTensorSize() == 0 ||
        weightsInfo.GetTotalTensorSize() == 0) {
        return Status::UNSUPPORTED;
    }
    return Status::SUCCESS;
}

} // KuDNN


namespace KuDNN {
namespace Detail {

static constexpr SizeType PRELU_BWD_ELE_NUM_1 = 300000;
static constexpr SizeType PRELU_BWD_ELE_NUM_STEP1 = 10000;
static constexpr SizeType PRELU_BWD_ELE_NUM_2 = 3000000;
static constexpr SizeType PRELU_BWD_ELE_NUM_3 = 100000000;

static constexpr SizeType PRELU_BWD_THREADS_THREASHOLD1 = 20;
static constexpr SizeType PRELU_BWD_THREADS_THREASHOLD2 = 40;
static constexpr SizeType PRELU_BWD_THREADS_THREASHOLD3 = 80;
static constexpr SizeType PRELU_BWD_THREADS_THREASHOLD4 = 160;


class PReLULayerBWDImpl final : public Parallelizable {
public:
    PReLULayerBWDImpl(const TensorInfo &srcInfo, const TensorInfo &diffSrcInfo,
        const TensorInfo &weightsInfo, const TensorInfo &diffWeightsInfo,
        const TensorInfo &diffDstInfo) noexcept(false) : srcType(srcInfo.GetType()), weiType(weightsInfo.GetType()),
        diffDstType(diffDstInfo.GetType()), compType(Element::TypeT::F32),
        broadcastType(PReLU::GetBCType(srcInfo, weightsInfo)), weiInfo(weightsInfo), lastScalar(false),
        needToConvert((srcType != weiType) || (srcType != diffDstType))
    {
        Service::ThrowOnStatus(PReLULayerBWD::ValidateInput(srcInfo, diffSrcInfo, weightsInfo,
            diffWeightsInfo, diffDstInfo), "PReLU Layer");
        reorderedDims = Service::BroadcastTo5D(Service::GetShapeAccordingToLayout(diffDstInfo.GetDims(),
            diffDstInfo.GetLayout()));
        reorderedWeiDims = Service::BroadcastTo5D(Service::GetShapeAccordingToLayout(weightsInfo.GetDims(),
            weightsInfo.GetLayout()));
        reorderedStridesSrc = Service::BroadcastTo5D(Service::GetShapeAccordingToLayout(srcInfo.GetStrides(),
            srcInfo.GetLayout()), false);
        reorderedStridesWei = Service::BroadcastTo5D(Service::GetShapeAccordingToLayout(weightsInfo.GetStrides(),
            weightsInfo.GetLayout()), false);
        reorderedStridesDst  = Service::BroadcastTo5D(Service::GetShapeAccordingToLayout(diffDstInfo.GetStrides(),
            diffDstInfo.GetLayout()), false);

        if (broadcastType == PReLU::BroadcastType::SHARED_AXES) {
            SizeType lastNoneOneIdx = PReLU::GetLastOneIdx(reorderedWeiDims);
            if (lastNoneOneIdx != (reorderedWeiDims.GetNumDims() - 1)) {
                SizeType tempShapeSrc[MAX_DIMS];
                SizeType tempShapeWei[MAX_DIMS];
                SizeType tempStridesSrc[MAX_DIMS];
                SizeType tempStridesWei[MAX_DIMS];
                SizeType tempStridesDst[MAX_DIMS];
                for (SizeType i = 0; i < lastNoneOneIdx; ++i) {
                    tempShapeSrc[i] = reorderedDims[i];
                    tempShapeWei[i] = reorderedWeiDims[i];
                    tempStridesSrc[i] = reorderedStridesSrc[i];
                    tempStridesWei[i] = reorderedStridesWei[i];
                    tempStridesDst[i] = reorderedStridesDst[i];
                }
                tempShapeSrc[lastNoneOneIdx] = Service::ArrayProduct(reorderedDims, reorderedDims.GetNumDims(),
                    lastNoneOneIdx);
                tempShapeWei[lastNoneOneIdx] = 1;
                tempStridesSrc[lastNoneOneIdx] = 1;
                tempStridesWei[lastNoneOneIdx] = 0;
                tempStridesDst[lastNoneOneIdx] = 1;

                reorderedDims = Service::BroadcastTo5D(Shape(tempShapeSrc, lastNoneOneIdx + 1));
                reorderedWeiDims = Service::BroadcastTo5D(Shape(tempShapeWei, lastNoneOneIdx + 1));
                reorderedStridesSrc = Service::BroadcastTo5D(Shape(tempStridesSrc, lastNoneOneIdx + 1), false);
                reorderedStridesWei = Service::BroadcastTo5D(Shape(tempStridesWei, lastNoneOneIdx + 1), false);
                reorderedStridesDst  = Service::BroadcastTo5D(Shape(tempStridesDst, lastNoneOneIdx + 1), false);
            }
        }
        lastScalar = reorderedWeiDims[reorderedWeiDims.GetNumDims() - 1] == 1;
        Parallelizable::Initialize(GetOptNumThreads(), reorderedWeiDims);
    }

    template <typename CompType, typename SrcType, typename WeiType, typename OutType>
    void Evaluate(const Shape &batchSplitDims, const void *src, void *diffSrc, const void *wei,
        void *diffWeights, const void *diffDst) noexcept
    {
        const SrcType *typedSrc = static_cast<const SrcType *>(src);
        SrcType *typedDiffSrc = static_cast<SrcType *>(diffSrc);
        const WeiType *typedWei = static_cast<const WeiType *>(wei);
        WeiType *typedDiffWei = static_cast<WeiType *>(diffWeights);
        const OutType *typedDiffDst = static_cast<const OutType *>(diffDst);

        switch (broadcastType) {
            case PReLU::BroadcastType::WHOLE_TENSOR: {
                    PReLU::KernelWholeTensor<CompType>(batchSplitDims, typedSrc, typedDiffSrc,
                        typedWei, typedDiffWei, typedDiffDst);
                    break;
            }
            case PReLU::BroadcastType::CHANNEL_SHARED: {
                    PReLU::KernelChannelShared<CompType>(batchSplitDims, typedSrc, typedDiffSrc,
                        typedWei, typedDiffWei, typedDiffDst, reorderedDims);
                    break;
            }
            default: {
                    if (lastScalar) {
                        PReLU::KernelSharedAxes<CompType>(batchSplitDims, typedSrc, typedDiffSrc,
                            typedWei, typedDiffWei, typedDiffDst,  reorderedStridesSrc, reorderedStridesWei,
                            reorderedWeiDims, reorderedStridesDst, reorderedDims,
                            Vector::DupWrapperN<WeiType, CompType,
                            Vector::GetUsedRegsN<WeiType, SrcType, CompType, OutType, float>()>{},
                            PReLU::Helpers::UpdateWrapper<CompType, SrcType, WeiType, OutType>{});
                    } else {
                        PReLU::KernelSharedAxes<CompType>(batchSplitDims, typedSrc, typedDiffSrc,
                            typedWei, typedDiffWei, typedDiffDst, reorderedStridesSrc, reorderedStridesWei,
                            reorderedWeiDims, reorderedStridesDst, reorderedDims,
                            Vector::LoadWrapperN<WeiType, CompType,
                            Vector::GetUsedRegsN<WeiType, SrcType, CompType, OutType, float>()>{},
                            PReLU::Helpers::StoreWrapper<CompType, SrcType, WeiType, OutType>{});
                    }
                    break;
            }
        }
    }

    void RunSequential(const Shape &batchSplitDims, const void *src,
            void *diffSrc, const void *wei, void *diffWeights, const void *diffDst) noexcept(false)
    {
        if (needToConvert) {
            Service::ParseTypeAdvanced<false>(compType, Element::TypeT::F32, Element::TypeT::F32, Element::TypeT::F32,
                *this, batchSplitDims, src, diffSrc, wei, diffWeights, diffDst);
        } else {
            Service::ParseTypeAdvanced<false>(compType, srcType, weiType, diffDstType, *this, batchSplitDims,
                src, diffSrc, wei, diffWeights, diffDst);
        }
    }

    void RunSequential(const Shape &batchSplitDims, const Shape &batchStrides, const void *src,
            void *diffSrc, const void *wei, void *diffWeights, const void *diffDst) noexcept(false)
    {
        SizeType strideSrc = 0;
        SizeType strideDiffSrc = 0;
        SizeType strideWei = 0;
        SizeType strideDiffWei = 0;
        SizeType strideDiffDst = 0;
        SizeType srcSize = needToConvert ? sizeof(float) : srcType.GetSize();
        SizeType weiSize = needToConvert ? sizeof(float) : weiType.GetSize();
        SizeType dstSize = needToConvert ? sizeof(float) : diffDstType.GetSize();
        for (SizeType i = 0; i < batchStrides.GetNumDims(); ++i) {
            strideSrc += batchStrides[i] * reorderedStridesSrc[i] * srcSize;
            strideDiffSrc += batchStrides[i] * reorderedStridesSrc[i] * srcSize;
            strideWei += batchStrides[i] * reorderedStridesWei[i] * weiSize;
            strideDiffWei += batchStrides[i] * reorderedStridesWei[i] * weiSize;
            strideDiffDst += batchStrides[i] * reorderedStridesDst[i] * dstSize;
        }
        RunSequential(batchSplitDims, static_cast<const std::byte *>(src) + strideSrc,
            static_cast<std::byte *>(diffSrc) + strideDiffSrc, static_cast<const std::byte *>(wei) + strideWei,
            static_cast<std::byte *>(diffWeights) + strideDiffWei,
            static_cast<const std::byte *>(diffDst) + strideDiffDst);
    }

    void Run(const void *src, void *diffSrc, const void *wei,
            void *diffWeights, const void *diffDst) noexcept(false)
    {
        if ((src == nullptr) || (diffSrc == nullptr) || (wei == nullptr) ||
            (diffWeights == nullptr)  || (diffDst == nullptr)) {
            throw Service::LogicError {"PReLU Layer: one of input matrices is nullptr"};
        }
        if ((src == diffSrc) || (wei == diffSrc) || (diffDst == diffSrc) ||
            (src == diffWeights) || (wei == diffWeights) || (diffDst == diffWeights)) {
            throw Service::LogicError {"PReLU Layer: input and output must not be the same pointers"};
        }
        const void *srcExec = src;
        void *diffSrcExec = diffSrc;
        const void *weiExec = wei;
        void *diffWeiExec = diffWeights;
        const void *diffDstExec = diffDst;
        if (needToConvert) {
            void *tmp = Service::AlignedAlloc(reorderedDims.GetTotalDimsSize() * sizeof(float));
            diffSrcExec = Service::AlignedAlloc(reorderedDims.GetTotalDimsSize() * sizeof(float));
            ReorderLayer r0({reorderedDims, srcType, Layout::ABCDE}, {reorderedDims,
                Element::TypeT::F32, Layout::ABCDE});
            r0.Run(src, tmp);
            srcExec = tmp;
            tmp = Service::AlignedAlloc(weiInfo.GetDims().GetTotalDimsSize() * sizeof(float));
            diffWeiExec = Service::AlignedAlloc(weiInfo.GetDims().GetTotalDimsSize() * sizeof(float));
            const TensorInfo orgWeiTensor{weiInfo.GetDims(), weiType, weiInfo.GetLayout(), weiInfo.GetStrides()};
            const TensorInfo reorderedWeiTensor{weiInfo.GetDims(), Element::TypeT::F32, weiInfo.GetLayout(),
                weiInfo.GetStrides()};
            ReorderLayer r1(orgWeiTensor, reorderedWeiTensor);
            r1.Run(wei, tmp);
            weiExec = tmp;
            tmp = Service::AlignedAlloc(reorderedDims.GetTotalDimsSize() * sizeof(float));
            ReorderLayer r2({reorderedDims, diffDstType, Layout::ABCDE}, {reorderedDims,
                Element::TypeT::F32, Layout::ABCDE});
            r2.Run(diffDst, tmp);
            diffDstExec = tmp;
        }
        Threading::BatchThreader(*this, srcExec,
            diffSrcExec, weiExec, diffWeiExec, diffDstExec);
        if (needToConvert) {
            ReorderLayer r0({reorderedDims, Element::TypeT::F32, Layout::ABCDE}, {reorderedDims,
                srcType, Layout::ABCDE});
            r0.Run(diffSrcExec, diffSrc);
            const TensorInfo orgWeiTensor{weiInfo.GetDims(), Element::TypeT::F32, weiInfo.GetLayout(),
                weiInfo.GetStrides()};
            const TensorInfo reorderedWeiTensor{weiInfo.GetDims(), weiType, weiInfo.GetLayout(), weiInfo.GetStrides()};
            ReorderLayer r1(orgWeiTensor, reorderedWeiTensor);
            r1.Run(diffWeiExec, diffWeights);
            Service::Deallocate(srcExec);
            Service::Deallocate(diffSrcExec);
            Service::Deallocate(weiExec);
            Service::Deallocate(diffWeiExec);
            Service::Deallocate(diffDstExec);
        }
    }

private:
    Element::Type srcType;
    Element::Type weiType;
    Element::Type diffDstType;
    Element::Type compType;
    PReLU::BroadcastType broadcastType;
    TensorInfo weiInfo;
    bool lastScalar;
    Shape reorderedDims;
    Shape reorderedWeiDims;
    Shape reorderedStridesSrc;
    Shape reorderedStridesWei;
    Shape reorderedStridesDst;
    bool needToConvert;

    SizeType GetOptNumThreads() const noexcept
    {
        SizeType totalSrcSize = reorderedDims.GetTotalDimsSize();
        if (totalSrcSize < PRELU_BWD_ELE_NUM_1) {
            return std::min(((totalSrcSize + PRELU_BWD_ELE_NUM_STEP1 - 1) / PRELU_BWD_ELE_NUM_STEP1),
                static_cast<SizeType>(PRELU_BWD_THREADS_THREASHOLD1));
        } else if (totalSrcSize < PRELU_BWD_ELE_NUM_2) {
            return PRELU_BWD_THREADS_THREASHOLD2;
        } else if (totalSrcSize < PRELU_BWD_ELE_NUM_3) {
            return PRELU_BWD_THREADS_THREASHOLD3;
        } else {
            return PRELU_BWD_THREADS_THREASHOLD4;
        }
    }
};

} // Detail

PReLULayerBWD::~PReLULayerBWD() noexcept {}

PReLULayerBWD::PReLULayerBWD(const TensorInfo &srcInfo, const TensorInfo &diffSrcInfo,
    const TensorInfo &weightsInfo, const TensorInfo &diffWeightsInfo,
    const TensorInfo &diffDstInfo) noexcept(false) : pImpl(new Detail::PReLULayerBWDImpl(
        srcInfo, diffSrcInfo, weightsInfo, diffWeightsInfo, diffDstInfo)) {}

PReLULayerBWD::PReLULayerBWD(const PReLULayerBWD &other) noexcept(false)
    : pImpl(new Detail::PReLULayerBWDImpl{*(other.pImpl.get())}) {}

PReLULayerBWD::PReLULayerBWD(PReLULayerBWD &&other) noexcept
    : pImpl(std::move(other.pImpl)) {}

PReLULayerBWD& PReLULayerBWD::operator=(const PReLULayerBWD &other) noexcept(false)
{
    if (this != &other) {
        this->pImpl.reset(new Detail::PReLULayerBWDImpl{*(other.pImpl.get())});
    }
    return *this;
}

PReLULayerBWD& PReLULayerBWD::operator=(PReLULayerBWD &&other) noexcept
{
    if (this != &other) {
        this->pImpl = std::move(other.pImpl);
    }
    return *this;
}

void PReLULayerBWD::Run(const void *src, void *diffSrc, const void *wei,
    void *diffWeights, const void *diffDst) const noexcept(false)
{
    pImpl->Run(src, diffSrc, wei, diffWeights, diffDst);
}

Status PReLULayerBWD::ValidateInput(const TensorInfo &srcInfo, const TensorInfo &diffSrcInfo,
    const TensorInfo &weightsInfo, const TensorInfo &diffWeightsInfo,
    const TensorInfo &diffDstInfo) noexcept
{
    if (srcInfo.GetLayout() != weightsInfo.GetLayout()) {
        return Status::UNSUPPORTED;
    }
    if (srcInfo.GetLayout() != diffDstInfo.GetLayout()) {
        return Status::UNSUPPORTED;
    }
    if (srcInfo != diffSrcInfo) {
        return Status::UNSUPPORTED;
    }
    if (weightsInfo != diffWeightsInfo) {
        return Status::UNSUPPORTED;
    }
    if (!srcInfo.IsDense() || !weightsInfo.IsDense() ||
        !diffDstInfo.IsDense()) {
        return Status::UNSUPPORTED;
    }
    if (srcInfo.GetDims().GetNumDims() != weightsInfo.GetDims().GetNumDims()) {
        return Status::BAD_ARGUMENTS;
    }
    for (SizeType i = 0; i < srcInfo.GetDims().GetNumDims(); ++i) {
        if (((srcInfo.GetDims()[i] != weightsInfo.GetDims()[i]) &&
            (weightsInfo.GetDims()[i] != 1)) || (srcInfo.GetDims()[i] != diffSrcInfo.GetDims()[i]) ||
            (weightsInfo.GetDims()[i] != diffWeightsInfo.GetDims()[i])) {
            return Status::BAD_ARGUMENTS;
        }
    }
    if (srcInfo.GetTotalTensorSize() == 0 || diffSrcInfo.GetTotalTensorSize() == 0 ||
        diffDstInfo.GetTotalTensorSize() == 0 ||
        weightsInfo.GetTotalTensorSize() == 0 || diffWeightsInfo.GetTotalTensorSize() == 0) {
        return Status::UNSUPPORTED;
    }
    return Status::SUCCESS;
}

} // KuDNN
