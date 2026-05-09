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
#include "threading_impl.hpp"
#include "internal_funcs.hpp"
#include "kernels/softmax/kudnn_softmax_kernels.hpp"

namespace KuDNN {
namespace Detail {

static constexpr SizeType SOFTMAX_BWD_INNER_SIZE_1 = 384;
static constexpr SizeType SOFTMAX_BWD_INNER_SIZE_2 = 3072;

static constexpr SizeType SOFTMAX_BWD_OUTER_SIZE_1 = 65;
static constexpr SizeType SOFTMAX_BWD_OUTER_SIZE_2 = 257;
static constexpr SizeType SOFTMAX_BWD_OUTER_SIZE_3 = 2049;

static constexpr SizeType SOFTMAX_BWD_TOTAL_SIZE_1 = 16384;
static constexpr SizeType SOFTMAX_BWD_TOTAL_SIZE_2 = 512000;

static constexpr SizeType SOFTMAX_BWD_THR_SCALER_1 = 2000;
static constexpr float SOFTMAX_BWD_THR_SCALER_2 = 8.f;
static constexpr float SOFTMAX_BWD_THR_SCALER_3 = 80.f;

static constexpr float SOFTMAX_BWD_THR_POWER_1 = 1.5f;
static constexpr float SOFTMAX_BWD_THR_POWER_2 = 1.65f;
static constexpr float SOFTMAX_BWD_THR_POWER_3 = 1.3f;

static constexpr SizeType SOFTMAX_BWD_THREADS_THRESHOLD_1 = 80;
static constexpr SizeType SOFTMAX_BWD_THREADS_THRESHOLD_2 = 120;
static constexpr SizeType SOFTMAX_BWD_THREADS_THRESHOLD_3 = 160;

class SoftmaxLayerBWDImpl final : public Parallelizable {
public:
    SoftmaxLayerBWDImpl(const TensorInfo &dstInfo, const TensorInfo &dstDiffInfo, const TensorInfo &srcDiffInfo,
                        SizeType axis, SoftmaxAlgorithmKind algorithm) noexcept(false)
        : compType(Element::TypeT::F32),
          dstT(dstInfo),
          dstDiffT(dstDiffInfo),
          srcDiffT(srcDiffInfo),
          /*  since all the arrays are broadcasted to 5D, we need to add extra dimensions which were broadcasted */
          axisNum(dstInfo.GetAxisNumAccordingToLayout(axis) + MAX_DIMS - dstInfo.GetNumDims()),
          algKind(algorithm)
    {
        Service::ThrowOnStatus(SoftmaxLayerBWD::ValidateInput(dstInfo, dstDiffInfo, srcDiffInfo, axis, algorithm),
                               "Softmax Layer");
        reorderedDims =
            Service::BroadcastTo5D(Service::GetShapeAccordingToLayout(dstInfo.GetDims(), dstInfo.GetLayout()));
        reorderedStridesDst = Service::BroadcastTo5D(
            Service::GetShapeAccordingToLayout(dstInfo.GetStrides(), dstInfo.GetLayout()), false);
        reorderedStridesDstDiff = Service::BroadcastTo5D(
            Service::GetShapeAccordingToLayout(dstDiffInfo.GetStrides(), dstDiffInfo.GetLayout()), false);
        reorderedStridesSrcDiff = Service::BroadcastTo5D(
            Service::GetShapeAccordingToLayout(srcDiffInfo.GetStrides(), srcDiffInfo.GetLayout()), false);
        // extra dimensions which were broadcasted
        Parallelizable::Initialize(GetOptNumThreads(), BatchedDims());
    }

    template <typename CompType, typename SrcType, typename DstType>
    void Evaluate(const Shape &batchSplitDims, const void *dst, const void *dstDiff, void *srcDiff) const noexcept
    {
        SizeType innerSize = Service::ArrayProduct(reorderedDims, reorderedDims.GetNumDims(), axisNum + 1);
        SizeType axisSize = reorderedDims[axisNum];
        SizeType outerSize = batchSplitDims.GetTotalDimsSize();
        SrcType *typedDiffSrc = static_cast<SrcType *>(srcDiff);
        const DstType *typedDst = static_cast<const DstType *>(dst);
        const DstType *typedDiffDst = static_cast<const DstType *>(dstDiff);
        if (axisNum == (reorderedDims.GetNumDims() - 1)) {
            Softmax::EvaluateLastAxisBWD<CompType>(typedDst, typedDiffDst, typedDiffSrc, axisSize, outerSize, axisNum,
                                                   algKind);
        } else {
            Softmax::EvaluateCrossAxisBWD<CompType>(typedDst, typedDiffDst, typedDiffSrc, innerSize, axisSize,
                                                    outerSize, algKind);
        }
    }

    void RunSequential(const Shape &batchSplitDims, const void *dst, const void *dstDiff, void *srcDiff) const
        noexcept(false)
    {
        if (srcDiffT.GetType() != dstT.GetType()) {
            Service::ParseType<false, false>(compType, compType, compType, *this, batchSplitDims, dst, dstDiff,
                                             srcDiff);
        } else {
            Service::ParseType<false, false>(compType, srcDiffT.GetType(), dstT.GetType(), *this, batchSplitDims, dst,
                                             dstDiff, srcDiff);
        }
    }

    void RunSequential(const Shape &batchSplitDims, const Shape &batchStrides, const void *dst, const void *dstDiff,
                       void *srcDiff) const noexcept(false)
    {
        SizeType strideDst = 0;
        SizeType strideDstDiff = 0;
        SizeType strideSrcDiff = 0;
        SizeType srcSize = (srcDiffT.GetType() != dstT.GetType()) ? compType.GetSize() : srcDiffT.GetType().GetSize();
        SizeType dstSize = (srcDiffT.GetType() != dstT.GetType()) ? compType.GetSize() : dstT.GetType().GetSize();
        for (SizeType i = 0; i < batchStrides.GetNumDims(); ++i) {
            strideDst += batchStrides[i] * reorderedStridesDst[i] * dstSize;
            strideDstDiff += batchStrides[i] * reorderedStridesDstDiff[i] * dstSize;
            strideSrcDiff += batchStrides[i] * reorderedStridesSrcDiff[i] * srcSize;
        }
        RunSequential(batchSplitDims, static_cast<const std::byte *>(dst) + strideDst,
                      static_cast<const std::byte *>(dstDiff) + strideDstDiff,
                      static_cast<std::byte *>(srcDiff) + strideSrcDiff);
    }

    void Run(const void *dst, const void *dstDiff, void *srcDiff) const noexcept(false)
    {
        if ((dst == nullptr) || (dstDiff == nullptr) || (srcDiff == nullptr)) {
            throw Service::LogicError{"Softmax Layer: one of input matrices is nullptr"};
        }
        const void *dstExec = dst;
        const void *dstDiffExec = dstDiff;
        void *srcDiffExec = srcDiff;

        if (srcDiffT.GetType() != dstT.GetType()) {
            srcDiffExec = Service::AlignedAlloc(srcDiffT.GetDims().GetTotalDimsSize() * compType.GetSize());
            void *tmp = Service::AlignedAlloc(dstT.GetDims().GetTotalDimsSize() * compType.GetSize());
            ReorderLayer r(dstT, {dstT.GetDims(), compType, dstT.GetLayout()});
            r.Run(dst, tmp);
            dstExec = tmp;
            tmp = Service::AlignedAlloc(dstDiffT.GetDims().GetTotalDimsSize() * compType.GetSize());
            r.Run(dstDiff, tmp);
            dstDiffExec = tmp;

            RunSequential(GetBatchedDims(), dstExec, dstDiffExec, srcDiffExec);

            ReorderLayer rr({dstT.GetDims(), compType, dstT.GetLayout()}, srcDiffT);
            rr.Run(srcDiffExec, srcDiff);
            Service::Deallocate(const_cast<void *>(dstExec));
            Service::Deallocate(const_cast<void *>(dstDiffExec));
            Service::Deallocate(srcDiffExec);
        } else if ((srcDiffT.GetType() == dstT.GetType()) && (srcDiffT.GetLayout() != dstT.GetLayout())) {
            srcDiffExec = Service::AlignedAlloc(srcDiffT.GetDims().GetTotalDimsSize() * srcDiffT.GetType().GetSize());

            RunSequential(GetBatchedDims(), dstExec, dstDiffExec, srcDiffExec);

            ReorderLayer rr(dstT, srcDiffT);
            rr.Run(srcDiffExec, srcDiff);
            Service::Deallocate(srcDiffExec);
        } else {
            Threading::BatchThreader(*this, dstExec, dstDiffExec, srcDiffExec);
        }
    }

private:
    Element::Type compType;
    TensorInfo dstT;
    TensorInfo dstDiffT;
    TensorInfo srcDiffT;
    SizeType axisNum;
    SoftmaxAlgorithmKind algKind;
    Shape reorderedDims;
    Shape reorderedStridesDst;
    Shape reorderedStridesDstDiff;
    Shape reorderedStridesSrcDiff;

    Shape BatchedDims() const noexcept
    {
        if (axisNum == 0) {
            return {1};
        } else {
            std::array<SizeType, MAX_DIMS> batchedDIms;
            for (SizeType i = 0; i < axisNum; ++i) {
                batchedDIms[i] = reorderedDims[i];
            }
            return {batchedDIms.data(), axisNum};
        }
    }

    SizeType GetOptNumThreads() const noexcept
    {
        SizeType totalSize = reorderedDims.GetTotalDimsSize();
        SizeType innerWithAxisSize = Service::ArrayProduct(reorderedDims, reorderedDims.GetNumDims(), axisNum);
        if (innerWithAxisSize == 0) {
            return 1;
        }
        SizeType outerSize = Service::ArrayProduct(reorderedDims, axisNum, 0);
        if (innerWithAxisSize < SOFTMAX_BWD_INNER_SIZE_1) { // for interval [16-256]
            if (totalSize < SOFTMAX_BWD_TOTAL_SIZE_1) {
                return (totalSize + SOFTMAX_BWD_THR_SCALER_1 - 1) / SOFTMAX_BWD_THR_SCALER_1;
            } else {
                float coef = std::pow(SOFTMAX_BWD_THR_POWER_1, std::log2(totalSize / SOFTMAX_BWD_TOTAL_SIZE_1));
                return static_cast<SizeType>(SOFTMAX_BWD_THR_SCALER_2 * coef);
            }
        } else if (innerWithAxisSize < SOFTMAX_BWD_INNER_SIZE_2) {
            if (totalSize < SOFTMAX_BWD_TOTAL_SIZE_1) {
                return (totalSize + SOFTMAX_BWD_THR_SCALER_1 - 1) / SOFTMAX_BWD_THR_SCALER_1;
            } else if (totalSize < SOFTMAX_BWD_TOTAL_SIZE_2) {
                float coef = std::pow(SOFTMAX_BWD_THR_POWER_2, std::log2(totalSize / SOFTMAX_BWD_TOTAL_SIZE_1));
                return static_cast<SizeType>(SOFTMAX_BWD_THR_SCALER_2 * coef);
            } else {
                float coef = std::pow(SOFTMAX_BWD_THR_POWER_3, std::log2(totalSize / SOFTMAX_BWD_TOTAL_SIZE_2));
                return static_cast<SizeType>(SOFTMAX_BWD_THR_SCALER_3 * coef);
            }
        } else {
            if (outerSize > SOFTMAX_BWD_OUTER_SIZE_1 && outerSize < SOFTMAX_BWD_OUTER_SIZE_2) {
                return SOFTMAX_BWD_THREADS_THRESHOLD_1;
            } else if (outerSize < SOFTMAX_BWD_OUTER_SIZE_3) {
                return SOFTMAX_BWD_THREADS_THRESHOLD_2;
            } else {
                return SOFTMAX_BWD_THREADS_THRESHOLD_3;
            }
        }
    }
};

} // namespace Detail

SoftmaxLayerBWD::~SoftmaxLayerBWD() noexcept {}

SoftmaxLayerBWD::SoftmaxLayerBWD(const TensorInfo &dstInfo, const TensorInfo &dstDiffInfo,
                                 const TensorInfo &srcDiffInfo, SizeType axis,
                                 SoftmaxAlgorithmKind algorithm) noexcept(false)
    : pImpl(new Detail::SoftmaxLayerBWDImpl(dstInfo, dstDiffInfo, srcDiffInfo, axis, algorithm))
{
}

SoftmaxLayerBWD::SoftmaxLayerBWD(const SoftmaxLayerBWD &other) noexcept(false)
    : pImpl(new Detail::SoftmaxLayerBWDImpl{*(other.pImpl.get())})
{
}

SoftmaxLayerBWD::SoftmaxLayerBWD(SoftmaxLayerBWD &&other) noexcept : pImpl(std::move(other.pImpl)) {}

SoftmaxLayerBWD &SoftmaxLayerBWD::operator=(const SoftmaxLayerBWD &other) noexcept(false)
{
    if (this != &other) {
        this->pImpl.reset(new Detail::SoftmaxLayerBWDImpl{*(other.pImpl.get())});
    }
    return *this;
}

SoftmaxLayerBWD &SoftmaxLayerBWD::operator=(SoftmaxLayerBWD &&other) noexcept
{
    if (this != &other) {
        this->pImpl = std::move(other.pImpl);
    }
    return *this;
}

void SoftmaxLayerBWD::Run(const void *dst, const void *dstDiff, void *srcDiff) const noexcept(false)
{
    pImpl->Run(dst, dstDiff, srcDiff);
}

Status SoftmaxLayerBWD::ValidateInput(const TensorInfo &dstInfo, const TensorInfo &dstDiffInfo,
                                      const TensorInfo &srcDiffInfo, SizeType axis,
                                      SoftmaxAlgorithmKind algorithm) noexcept
{
    Service::Unused(algorithm);
    if (dstInfo.GetType() != dstDiffInfo.GetType()) {
        return Status::UNSUPPORTED;
    }
    if (dstDiffInfo.GetLayout() != dstInfo.GetLayout()) {
        return Status::UNSUPPORTED;
    }
    if (dstInfo.GetDims() != srcDiffInfo.GetDims()) {
        return Status::BAD_ARGUMENTS;
    }
    if (dstInfo != dstDiffInfo) {
        return Status::BAD_ARGUMENTS;
    }
    if (dstInfo.GetNumDims() <= axis) {
        return Status::BAD_ARGUMENTS;
    }
    if (dstInfo.GetTotalTensorSize() == 0) {
        return Status::UNSUPPORTED;
    }
    if (!dstInfo.IsDense()) {
        return Status::UNSUPPORTED;
    }
    return Status::SUCCESS;
}

} // namespace KuDNN
