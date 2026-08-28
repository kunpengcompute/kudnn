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
#include <functional>
#include <memory>
#include <cstddef>
#include <arm_sve.h>

#include "threading_impl.hpp"
#include "vector_wrappers.hpp"
#include "kernels/shuffle/kudnn_shuffle_kernels.hpp"

namespace KuDNN {
namespace Detail {

class ShuffleLayerImpl final : public Parallelizable {
public:
    ShuffleLayerImpl(const TensorInfo &srcInfo, const TensorInfo &dstInfo, SizeType inputAxis,
        SizeType group) noexcept(false) // dstType and compType here are to be compatible with ParseType
        : srcType(srcInfo.GetType()), dstType(srcInfo.GetType()),
        compType(srcInfo.GetType()),
        axis(inputAxis), reorderedAxis(srcInfo.GetAxisNumAccordingToLayout(inputAxis) + (DIM_5 - srcInfo.GetNumDims())),
        axisSize(srcInfo.GetDims()[inputAxis]), groupSize(group), isDenseProblem(srcInfo.IsDense())
    {
        Service::ThrowOnStatus(ShuffleLayer::ValidateInput(srcInfo, dstInfo, axis, groupSize), "Shuffle Layer");

        Shape dims = Service::GetShapeAccordingToLayout(dstInfo.GetDims(), srcInfo.GetLayout());
        Shape srcStrides = Service::GetShapeAccordingToLayout(srcInfo.GetStrides(), srcInfo.GetLayout());

        reorderedDims = Service::BroadcastTo5D(dims);
        reorderedStridesSrc = Service::BroadcastTo5D(srcStrides);

        const SizeType transposeRow = groupSize;
        const SizeType transposeCol = axisSize / groupSize;

        for (SizeType i = 0; i < transposeRow; ++i) {
            for (SizeType j = 0; j < transposeCol; ++j) {
                revTransposed.push_back(j * transposeRow + i);
            }
        }

        Parallelizable::Initialize(GetOptNumThreads(), reorderedDims);
    }

    Shape SimplifyProblem(const Shape &oldSplitDims) const
    {
        SizeType dims[DIM_5] = {oldSplitDims[IDX_0],
                                oldSplitDims[IDX_1],
                                oldSplitDims[IDX_2],
                                oldSplitDims[IDX_3],
                                oldSplitDims[IDX_4]};

        if (isDenseProblem && (reorderedAxis <= IDX_2)) {
            /*
                For dense problems, we can generalize loops in different dimensions into a single loop to
                fully vectorize.
            */
            SizeType mergeLoops = oldSplitDims[IDX_4];
            for (SizeType i = IDX_0; i <= IDX_2 - reorderedAxis; ++i) {
                if ((oldSplitDims[IDX_4 - i] == IDX_1)) {
                    break;
                }
                /*
                    Since we use 0 strides for dimensions == 1 for now, we need to break when dimensions == 1.
                    Otherwise, we will find the wrong element after simplify the problem.
                */
                dims[IDX_3 - i] = IDX_1;
                mergeLoops *= oldSplitDims[IDX_3 - i];
            }
            dims[IDX_4] = mergeLoops;
        }
        return Shape{dims[IDX_0], dims[IDX_1], dims[IDX_2], dims[IDX_3], dims[IDX_4]};
    }

    template <typename CompType, typename InType, typename OutType>
    void Evaluate(const Shape &batchSplitDims, const void *src, void *dst, SizeType strideRevTransposed) const noexcept
    {
        const InType *typedSrc = static_cast<const InType *>(src);
        InType *typedDst = static_cast<InType *>(dst);
        Shuffle::EvaluateShuffle(SimplifyProblem(batchSplitDims), typedSrc, typedDst, reorderedStridesSrc,
            reorderedAxis, revTransposed.data() + strideRevTransposed, strideRevTransposed);
    }

    void RunSequential(const Shape &batchSplitDims, const void *src, void *dst,
        SizeType strideRevTransposed = 0) const noexcept
    {
        Service::ParseType<true, true>(compType, srcType, dstType, *this, batchSplitDims, src, dst,
            strideRevTransposed);
    }

    void RunSequential(const Shape &batchSplitDims, const Shape &batchStrides,
        const void *src, void *dst) const noexcept(false)
    {
        SizeType strideSrc = 0;
        for (SizeType i = 0; i < batchStrides.GetNumDims(); ++i) {
            strideSrc += batchStrides[i] * reorderedStridesSrc[i] * srcType.GetSize();
        }
        RunSequential(batchSplitDims, static_cast<const std::byte *>(src) + strideSrc,
            static_cast<std::byte *>(dst) + strideSrc, batchStrides[reorderedAxis]);
    }

    void Run(const void *src, void *dst) const noexcept(false)
    {
        if ((src == nullptr) || (dst == nullptr)) {
            throw Service::LogicError {"Shuffle Layer: one of input matrices is nullptr"};
        }
        Threading::BatchThreader(*this, src, dst);
    }

private:
    Element::Type srcType;
    Element::Type dstType;
    Element::Type compType;
    SizeType axis;
    SizeType reorderedAxis;
    SizeType axisSize;
    SizeType groupSize;
    Shape reorderedDims;
    Shape reorderedStridesSrc;
    std::vector<SizeType, Service::AlignedAllocator<SizeType>> revTransposed;
    bool isDenseProblem;

    SizeType GetOptNumThreads() const noexcept
    {
        return reorderedDims.GetTotalDimsSize();
    }
};

} // Detail

ShuffleLayer::~ShuffleLayer() noexcept {}

ShuffleLayer::ShuffleLayer(const TensorInfo &srcInfo, const TensorInfo &dstInfo, SizeType axis,
    SizeType group) noexcept(false) : pImpl(
        new Detail::ShuffleLayerImpl(srcInfo, dstInfo, axis, group)) {}

ShuffleLayer::ShuffleLayer(const ShuffleLayer &other) noexcept(false)
    : pImpl(new Detail::ShuffleLayerImpl{*(other.pImpl.get())}) {}

ShuffleLayer::ShuffleLayer(ShuffleLayer &&other) noexcept : pImpl(std::move(other.pImpl))
{}

ShuffleLayer& ShuffleLayer::operator=(const ShuffleLayer &other) noexcept(false)
{
    if (this != &other) {
        this->pImpl.reset(new Detail::ShuffleLayerImpl{*(other.pImpl.get())});
    }
    return *this;
}

ShuffleLayer& ShuffleLayer::operator=(ShuffleLayer &&other) noexcept
{
    if (this != &other) {
        this->pImpl = std::move(other.pImpl);
    }
    return *this;
}

void ShuffleLayer::Run(const void *src, void *dst) const noexcept(false)
{
    pImpl->Run(src, dst);
}

Status ShuffleLayer::ValidateInput(const TensorInfo &srcInfo, const TensorInfo &dstInfo, SizeType axis,
    SizeType groupSize) noexcept
{
    if (dstInfo.GetNumDims() <= axis) {
        return Status::BAD_ARGUMENTS;
    }
    if ((groupSize == IDX_0) || (srcInfo.GetDims()[axis] % groupSize != 0)) {
        return Status::BAD_ARGUMENTS;
    }
    if (srcInfo.GetDims() != dstInfo.GetDims()) {
        return Status::BAD_ARGUMENTS;
    }
    return Status::SUCCESS;
}

} // KuDNN
