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

#include <cstddef>
#include <cmath>
#include <limits>

#include <arm_sve.h>

#include "kudnn.hpp"

#include "internal_funcs.hpp"
#include "threading_impl.hpp"
#include "kernels/softmax/kudnn_softmax_kernels.hpp"

namespace KuDNN {
namespace Detail {

static constexpr SizeType SOFTMAX_FWD_INNER_SIZE_1 = 384;
static constexpr SizeType SOFTMAX_FWD_INNER_SIZE_2 = 3072;

static constexpr SizeType SOFTMAX_FWD_OUTER_SIZE_1 = 65;
static constexpr SizeType SOFTMAX_FWD_OUTER_SIZE_2 = 257;
static constexpr SizeType SOFTMAX_FWD_OUTER_SIZE_3 = 2049;

static constexpr SizeType SOFTMAX_FWD_TOTAL_SIZE_1 = 16384;
static constexpr SizeType SOFTMAX_FWD_TOTAL_SIZE_2 = 512000;

static constexpr SizeType SOFTMAX_FWD_THR_SCALER_1 = 2000;
static constexpr float SOFTMAX_FWD_THR_SCALER_2 = 8.f;
static constexpr float SOFTMAX_FWD_THR_SCALER_3 = 80.f;

static constexpr float SOFTMAX_FWD_THR_POWER_1 = 1.5f;
static constexpr float SOFTMAX_FWD_THR_POWER_2 = 1.65f;
static constexpr float SOFTMAX_FWD_THR_POWER_3 = 1.3f;

static constexpr SizeType SOFTMAX_FWD_THREADS_THRESHOLD_1 = 80;
static constexpr SizeType SOFTMAX_FWD_THREADS_THRESHOLD_2 = 120;
static constexpr SizeType SOFTMAX_FWD_THREADS_THRESHOLD_3 = 160;

class SoftmaxLayerFWDImpl final : public Parallelizable {
public:
    SoftmaxLayerFWDImpl(const TensorInfo &srcInfo, const TensorInfo &dstInfo, SizeType axis,
                        SoftmaxAlgorithmKind algorithm) noexcept(false)
        : compType(Element::TypeT::F32),
          srcT(srcInfo),
          dstT(dstInfo),
          /* since all the arrays are broadcasted to 5D, we need to add extra dimensions which were broadcasted */
          axisNum(srcInfo.GetAxisNumAccordingToLayout(axis) + MAX_DIMS - srcInfo.GetNumDims()),
          algKind(algorithm)
    {
        Service::ThrowOnStatus(SoftmaxLayerFWD::ValidateInput(srcInfo, dstInfo, axis, algorithm), "Softmax Layer");
        reorderedDims =
            Service::BroadcastTo5D(Service::GetShapeAccordingToLayout(srcInfo.GetDims(), srcInfo.GetLayout()));
        reorderedStridesSrc = Service::BroadcastTo5D(
            Service::GetShapeAccordingToLayout(srcInfo.GetStrides(), srcInfo.GetLayout()), false);
        reorderedStridesDst = Service::BroadcastTo5D(
            Service::GetShapeAccordingToLayout(dstInfo.GetStrides(), dstInfo.GetLayout()), false);
        // extra dimensions which were broadcasted
        Parallelizable::Initialize(GetOptNumThreads(), BatchedDims());
    }

    template <typename CompType, typename InType, typename OutType>
    void Evaluate(const Shape &batchSplitDims, const void *src, void *dst) const noexcept
    {
        SizeType innerSize = Service::ArrayProduct(reorderedDims, reorderedDims.GetNumDims(), axisNum + 1);
        SizeType axisSize = reorderedDims[axisNum];
        SizeType outerSize = batchSplitDims.GetTotalDimsSize();
        const InType *typedSrc = static_cast<const InType *>(src);
        OutType *typedDst = static_cast<OutType *>(dst);

        if (axisNum == (reorderedDims.GetNumDims() - 1)) {
            Softmax::EvaluateLastAxisFWD<CompType>(typedSrc, typedDst, axisSize, outerSize, algKind);
        } else {
            Softmax::EvaluateCrossAxisFWD<CompType>(typedSrc, typedDst, innerSize, axisSize, outerSize, algKind);
        }
    }

    void RunSequential(const Shape &batchSplitDims, const void *src, void *dst) const noexcept(false)
    {
        if (srcT.GetType() != dstT.GetType()) {
            Service::ParseType<true, true>(compType, compType, compType, *this, batchSplitDims, src, dst);
        } else {
            Service::ParseType<true, true>(compType, srcT.GetType(), dstT.GetType(), *this, batchSplitDims, src, dst);
        }
    }

    void RunSequential(const Shape &batchSplitDims, const Shape &batchStrides, const void *src, void *dst) const
        noexcept(false)
    {
        SizeType strideSrc = 0;
        SizeType strideDst = 0;
        SizeType srcSize = (srcT.GetType() != dstT.GetType()) ? compType.GetSize() : srcT.GetType().GetSize();
        SizeType dstSize = (srcT.GetType() != dstT.GetType()) ? compType.GetSize() : dstT.GetType().GetSize();
        for (SizeType i = 0; i < batchStrides.GetNumDims(); ++i) {
            strideSrc += batchStrides[i] * reorderedStridesSrc[i] * srcSize;
            strideDst += batchStrides[i] * reorderedStridesDst[i] * dstSize;
        }
        RunSequential(batchSplitDims, static_cast<const std::byte *>(src) + strideSrc,
                      static_cast<std::byte *>(dst) + strideDst);
    }

    void Run(const void *src, void *dst) const noexcept(false)
    {
        if ((src == nullptr) || (dst == nullptr)) {
            throw Service::LogicError{"Softmax Layer: one of input matrices is nullptr"};
        }

        const void *srcExec = src;
        void *dstExec = dst;

        if (srcT.GetType() != dstT.GetType()) {
            dstExec = Service::AlignedAlloc(dstT.GetDims().GetTotalDimsSize() * compType.GetSize());
            void *tmp = Service::AlignedAlloc(srcT.GetDims().GetTotalDimsSize() * compType.GetSize());
            ReorderLayer r(srcT, {srcT.GetDims(), compType, srcT.GetLayout()});
            r.Run(src, tmp);
            srcExec = tmp;

            RunSequential(GetBatchedDims(), srcExec, dstExec);

            ReorderLayer rr({srcT.GetDims(), compType, srcT.GetLayout()}, dstT);
            rr.Run(dstExec, dst);
            Service::Deallocate(const_cast<void *>(srcExec));
            Service::Deallocate(const_cast<void *>(dstExec));
        } else if ((srcT.GetType() == dstT.GetType()) && (srcT.GetLayout() != dstT.GetLayout())) {
            dstExec = Service::AlignedAlloc(dstT.GetDims().GetTotalDimsSize() * dstT.GetType().GetSize());

            RunSequential(GetBatchedDims(), srcExec, dstExec);

            ReorderLayer rr(srcT, dstT);
            rr.Run(dstExec, dst);
            Service::Deallocate(const_cast<void *>(dstExec));
        } else {
            Threading::BatchThreader(*this, srcExec, dstExec);
        }
    }

private:
    Element::Type compType;
    TensorInfo srcT;
    TensorInfo dstT;
    SizeType axisNum;
    SoftmaxAlgorithmKind algKind;
    Shape reorderedDims;
    Shape reorderedStridesSrc;
    Shape reorderedStridesDst;

    Shape BatchedDims() const noexcept
    {
        if (axisNum == 0) {
            return {1};
        } else {
            std::array<SizeType, MAX_DIMS> batchedDImsArr;
            for (SizeType i = 0; i < axisNum; ++i) {
                batchedDImsArr[i] = reorderedDims[i];
            }
            return {batchedDImsArr.data(), axisNum};
        }
    }

    SizeType GetOptNumThreads() const noexcept
    {
        SizeType totalSize = reorderedDims.GetTotalDimsSize();
        SizeType innerSize = Service::ArrayProduct(reorderedDims, reorderedDims.GetNumDims(), axisNum);
        SizeType outerSize = Service::ArrayProduct(reorderedDims, axisNum, 0);
        float coef;
        if (innerSize < SOFTMAX_FWD_INNER_SIZE_1) {
            if (totalSize < SOFTMAX_FWD_TOTAL_SIZE_1) {
                return (totalSize + SOFTMAX_FWD_THR_SCALER_1 - 1) / SOFTMAX_FWD_THR_SCALER_1;
            } else {
                coef = static_cast<SizeType>(
                    std::pow(SOFTMAX_FWD_THR_POWER_1, std::log2(totalSize / SOFTMAX_FWD_TOTAL_SIZE_1)));
                return static_cast<SizeType>(SOFTMAX_FWD_THR_SCALER_2 * coef);
            }
        } else if (innerSize < SOFTMAX_FWD_INNER_SIZE_2) {
            if (totalSize < SOFTMAX_FWD_TOTAL_SIZE_1) {
                return (totalSize + SOFTMAX_FWD_THR_SCALER_1 - 1) / SOFTMAX_FWD_THR_SCALER_1;
            } else if (totalSize < SOFTMAX_FWD_TOTAL_SIZE_2) {
                coef = std::pow(SOFTMAX_FWD_THR_POWER_2, std::log2(totalSize / SOFTMAX_FWD_TOTAL_SIZE_1));
                return static_cast<SizeType>(SOFTMAX_FWD_THR_SCALER_2 * coef);
            } else {
                coef = std::pow(SOFTMAX_FWD_THR_POWER_3, std::log2(totalSize / SOFTMAX_FWD_TOTAL_SIZE_2));
                return static_cast<SizeType>(SOFTMAX_FWD_THR_SCALER_3 * coef);
            }
        } else {
            if (outerSize > SOFTMAX_FWD_OUTER_SIZE_1 && outerSize < SOFTMAX_FWD_OUTER_SIZE_2) {
                return SOFTMAX_FWD_THREADS_THRESHOLD_1;
            } else if (outerSize < SOFTMAX_FWD_OUTER_SIZE_3) {
                return SOFTMAX_FWD_THREADS_THRESHOLD_2;
            } else {
                return SOFTMAX_FWD_THREADS_THRESHOLD_3;
            }
        }
    }
};

} // namespace Detail

SoftmaxLayerFWD::~SoftmaxLayerFWD() noexcept {}

SoftmaxLayerFWD::SoftmaxLayerFWD(const TensorInfo &srcInfo, const TensorInfo &dstInfo, SizeType axis,
                                 SoftmaxAlgorithmKind algorithm) noexcept(false)
    : pImpl(new Detail::SoftmaxLayerFWDImpl(srcInfo, dstInfo, axis, algorithm))
{
}

SoftmaxLayerFWD::SoftmaxLayerFWD(const SoftmaxLayerFWD &other) noexcept(false)
    : pImpl(new Detail::SoftmaxLayerFWDImpl{*(other.pImpl.get())})
{
}

SoftmaxLayerFWD::SoftmaxLayerFWD(SoftmaxLayerFWD &&other) noexcept : pImpl(std::move(other.pImpl)) {}

SoftmaxLayerFWD &SoftmaxLayerFWD::operator=(const SoftmaxLayerFWD &other) noexcept(false)
{
    if (this != &other) {
        this->pImpl.reset(new Detail::SoftmaxLayerFWDImpl{*(other.pImpl.get())});
    }
    return *this;
}

SoftmaxLayerFWD &SoftmaxLayerFWD::operator=(SoftmaxLayerFWD &&other) noexcept
{
    if (this != &other) {
        this->pImpl = std::move(other.pImpl);
    }
    return *this;
}

void SoftmaxLayerFWD::Run(const void *src, void *dst) const noexcept(false)
{
    pImpl->Run(src, dst);
}

Status SoftmaxLayerFWD::ValidateInput(const TensorInfo &srcInfo, const TensorInfo &dstInfo, SizeType axis,
                                      SoftmaxAlgorithmKind algorithm) noexcept(false)
{
    Service::Unused(algorithm);
    if (srcInfo.GetDims() != dstInfo.GetDims()) {
        return Status::BAD_ARGUMENTS;
    }
    if (srcInfo.GetNumDims() <= axis) {
        return Status::BAD_ARGUMENTS;
    }
    if (srcInfo.GetTotalTensorSize() == 0) {
        return Status::UNSUPPORTED;
    }
    if (!srcInfo.IsDense()) {
        return Status::UNSUPPORTED;
    }
    return Status::SUCCESS;
}

} // namespace KuDNN
