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
#include <vector>

#include <arm_sve.h>

#include "kudnn.hpp"

#include "internal_funcs.hpp"
#include "threading_impl.hpp"
#include "vector_wrappers.hpp"
#include "kernels/sum/kudnn_sum_kernels.hpp"

namespace KuDNN {
namespace Detail {

static constexpr SizeType SUM_ELE_NUM_1 = 100000;
static constexpr SizeType SUM_ELE_NUM_STEP1 = 10000;
static constexpr SizeType SUM_ELE_NUM_2 = 4000000;
static constexpr SizeType SUM_ELE_NUM_STEP2 = 20000;
static constexpr SizeType SUM_ELE_NUM_3 = 20000000;
static constexpr SizeType SUM_ELE_NUM_STEP3 = 50000;
static constexpr SizeType SUM_ELE_NUM_4 = 200000000;
static constexpr SizeType SUM_ELE_NUM_5 = 8000000000;

static constexpr SizeType SUM_THREADS_THREASHOLD1 = 8;
static constexpr SizeType SUM_THREADS_THREASHOLD2 = 40;
static constexpr SizeType SUM_THREADS_THREASHOLD3 = 40;
static constexpr SizeType SUM_THREADS_THREASHOLD4 = 40;
static constexpr SizeType SUM_THREADS_THREASHOLD5 = 80;
static constexpr SizeType SUM_THREADS_THREASHOLD6 = 160;

class SumLayerImpl final : public Parallelizable {
public:
    SumLayerImpl(const std::vector<TensorInfo> &srcInfo, const float *scales,
        const TensorInfo &dstInfo) noexcept(false)
        : numOfSrcs(srcInfo.size()), srcT(srcInfo), dstT(dstInfo),
        compType(Element::TypeT::F32), scl(numOfSrcs),
        reorderedDims(Service::BroadcastTo5D(Service::GetShapeAccordingToLayout(dstInfo.GetDims(),
            dstInfo.GetLayout()))),
        reorderedStridesSrc(Service::BroadcastTo5D(Service::GetShapeAccordingToLayout(srcInfo[0].GetStrides(),
            srcInfo[0].GetLayout()), false)),
        reorderedStridesDst(Service::BroadcastTo5D(Service::GetShapeAccordingToLayout(dstInfo.GetStrides(),
            dstInfo.GetLayout()), false)),
        needToConvert(false)
    {
        Service::ThrowOnStatus(SumLayer::ValidateInput(srcInfo, scales, dstInfo), "Sum Layer");
        for (SizeType i = 0; i < numOfSrcs; ++i) {
            scl[i] = scales[i];
            if ((srcT[i].GetType() != dstT.GetType()) || (srcT[i].GetLayout() != dstT.GetLayout())) {
                needToConvert = true;
            }
        }

        if (needToConvert) {
            reorderedStridesSrc = reorderedStridesDst;
        }

        Parallelizable::Initialize(GetOptNumThreads(), reorderedDims);
    }

    template <typename CompType, typename InType, typename OutType>
    void Evaluate(const Shape &batchSplitDims, const void **src, void *dst) const noexcept
    {
        OutType *typedDst = static_cast<OutType *>(dst);
        switch (numOfSrcs) {
            case 1: {
                Sum::EvaluateSum<CompType, InType, OutType, 1>(batchSplitDims, src, typedDst, scl.data());
                break;
            }
            case 2: { // 2 means the unroll 2 times in the loop
                Sum::EvaluateSum<CompType, InType, OutType, 2>(batchSplitDims, src, typedDst, scl.data());
                break;
            }
            case 3: { // 3 means the unroll 3 times in the loop
                Sum::EvaluateSum<CompType, InType, OutType, 3>(batchSplitDims, src, typedDst, scl.data());
                break;
            }
            case 4: { // 4 means the unroll 4 times in the loop
                Sum::EvaluateSum<CompType, InType, OutType, 4>(batchSplitDims, src, typedDst, scl.data());
                break;
            }
            case 5: { // 5 means the unroll 5 times in the loop
                Sum::EvaluateSum<CompType, InType, OutType, 5>(batchSplitDims, src, typedDst, scl.data());
                break;
            }
            case 6: { // 6 means the unroll 6 times in the loop
                Sum::EvaluateSum<CompType, InType, OutType, 6>(batchSplitDims, src, typedDst, scl.data());
                break;
            }
            case 7: { // 7 means the unroll 7 times in the loop
                Sum::EvaluateSum<CompType, InType, OutType, 7>(batchSplitDims, src, typedDst, scl.data());
                break;
            }
            case 8: { // 8 means the unroll 8 times in the loop
                Sum::EvaluateSum<CompType, InType, OutType, 8>(batchSplitDims, src, typedDst, scl.data());
                break;
            }
            default: {
                Sum::EvaluateSum<CompType, InType>(batchSplitDims, src, typedDst, scl.data(), numOfSrcs);
                break;
            }
        }
    }

    void RunSequential(const Shape &batchSplitDims, const void **src, void *dst) const noexcept
    {
        if (needToConvert) {
            Service::ParseType<true, false>(compType, compType, compType, *this, batchSplitDims, src, dst);
        } else {
            Service::ParseType<true, false>(compType, dstT.GetType(), dstT.GetType(), *this, batchSplitDims, src, dst);
        }
    }

    void RunSequential(const Shape &batchSplitDims, const Shape &batchStrides,
        const void **src, void *dst) const noexcept(false)
    {
        std::vector<SizeType> strideSrc(numOfSrcs);
        SizeType strideDst = 0;
        SizeType dstSize = needToConvert ? compType.GetSize() : dstT.GetType().GetSize();
        for (SizeType i = 0; i < batchStrides.GetNumDims(); ++i) {
            for (SizeType j = 0; j < numOfSrcs; ++j) {
                SizeType srcSize = needToConvert ? compType.GetSize() : srcT[j].GetType().GetSize();
                strideSrc[j] += batchStrides[i] * reorderedStridesSrc[i] * srcSize;
            }
            strideDst += batchStrides[i] * reorderedStridesDst[i] * dstSize;
        }
        std::vector<const void *> tmp;
        for (SizeType i = 0; i < numOfSrcs; ++i) {
            tmp.push_back(static_cast<const std::byte *>(src[i]) + strideSrc[i]);
        }
        RunSequential(batchSplitDims, static_cast<const void **>(tmp.data()),
            static_cast<std::byte *>(dst) + strideDst);
    }

    void Run(const void **src, void *dst) const noexcept(false)
    {
        if ((src == nullptr) || (dst == nullptr)) {
            throw Service::LogicError {"Sum Layer: one of input matrices is nullptr"};
        }
        for (SizeType i = 0; i < numOfSrcs; ++i) {
            if (src[i] == nullptr) {
                throw Service::LogicError {"Sum Layer: one of input matrices is nullptr"};
            }
        }

        if (needToConvert) {
            std::vector<const void *> convertedSrc(numOfSrcs);
            void *convertedDst = dst;

            for (SizeType i = 0; i < numOfSrcs; ++i) {
                void *tmp = Service::AlignedAlloc(reorderedDims.GetTotalDimsSize() * compType.GetSize());
                ReorderLayer r(srcT[i], {dstT.GetDims(), compType, dstT.GetLayout()});
                r.Run(src[i], tmp);
                convertedSrc[i] = tmp;
            }
            convertedDst = Service::AlignedAlloc(reorderedDims.GetTotalDimsSize() * compType.GetSize());

            Threading::BatchThreader(*this, convertedSrc.data(), convertedDst);

            for (SizeType i = 0; i < numOfSrcs; ++i) {
                Service::Deallocate(convertedSrc[i]);
            }
            ReorderLayer r({dstT.GetDims(), compType, dstT.GetLayout()}, dstT);
            r.Run(convertedDst, dst);
            Service::Deallocate(convertedDst);
        } else {
            Threading::BatchThreader(*this, src, dst);
        }
    }

private:
    SizeType numOfSrcs;
    std::vector<TensorInfo> srcT;
    TensorInfo dstT;
    Element::Type compType;
    std::vector<float, Service::AlignedAllocator<float>> scl;
    Shape reorderedDims;
    Shape reorderedStridesSrc;
    Shape reorderedStridesDst;
    bool needToConvert;

    SizeType GetOptNumThreads() const noexcept
    {
        SizeType totalSize = numOfSrcs * reorderedDims.GetTotalDimsSize();
        if (totalSize < SUM_ELE_NUM_1) {
            return std::min(((totalSize + SUM_ELE_NUM_STEP1 - 1) / SUM_ELE_NUM_STEP1),
                static_cast<SizeType>(SUM_THREADS_THREASHOLD1));
        } else if (totalSize < SUM_ELE_NUM_2) {
            return std::min(((totalSize + SUM_ELE_NUM_STEP2 - 1) / SUM_ELE_NUM_STEP2),
                static_cast<SizeType>(SUM_THREADS_THREASHOLD2));
        } else if (totalSize < SUM_ELE_NUM_3) {
            return std::min(((totalSize + SUM_ELE_NUM_STEP3 - 1) / SUM_ELE_NUM_STEP3),
                static_cast<SizeType>(SUM_THREADS_THREASHOLD3));
        } else if (totalSize < SUM_ELE_NUM_4) {
            return SUM_THREADS_THREASHOLD4;
        } else if (totalSize < SUM_ELE_NUM_5) {
            return SUM_THREADS_THREASHOLD5;
        } else {
            return SUM_THREADS_THREASHOLD6;
        }
    }
};

} // Detail

SumLayer::~SumLayer() noexcept {}

SumLayer::SumLayer(const std::vector<TensorInfo> &srcInfo, const float *scales,
    const TensorInfo &dstInfo) noexcept(false)
    : pImpl(new Detail::SumLayerImpl(srcInfo, scales, dstInfo)) {}

SumLayer::SumLayer(const SumLayer &other) noexcept(false)
    : pImpl(new Detail::SumLayerImpl{*(other.pImpl.get())}) {}

SumLayer::SumLayer(SumLayer &&other) noexcept
    : pImpl(std::move(other.pImpl)) {}

SumLayer& SumLayer::operator=(const SumLayer &other) noexcept(false)
{
    if (this != &other) {
        this->pImpl.reset(new Detail::SumLayerImpl{*(other.pImpl.get())});
    }
    return *this;
}

SumLayer& SumLayer::operator=(SumLayer &&other) noexcept
{
    if (this != &other) {
        this->pImpl = std::move(other.pImpl);
    }
    return *this;
}

void SumLayer::Run(const void **src, void *dst) const noexcept(false)
{
    pImpl->Run(src, dst);
}

Status SumLayer::ValidateInput(const std::vector<TensorInfo> &srcInfo, const float *scales,
                               const TensorInfo &dstInfo) noexcept
{
    if (dstInfo.HasStrideOverFastestDim()) {
        return Status::UNSUPPORTED;
    }
    if (scales == nullptr) {
        return Status::UNSUPPORTED;
    }
    if (!dstInfo.IsDense()) {
        return Status::UNSUPPORTED;
    }
    for (auto &&src : srcInfo) {
        if ((src.GetDims() != dstInfo.GetDims()) ||
            (src.HasStrideOverFastestDim())) {
            return Status::UNSUPPORTED;
        }
    }
    return Status::SUCCESS;
}

} // KuDNN
