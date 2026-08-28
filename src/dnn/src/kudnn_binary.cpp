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
#include "internal_funcs.hpp"
#include "threading_impl.hpp"
#include "vector_wrappers.hpp"
#include "kernels/binary/kudnn_binary_kernels.hpp"

namespace KuDNN {
namespace Detail {

static constexpr SizeType BINARY_ELE_NUM_1 = 1000000;
static constexpr SizeType BINARY_ELE_NUM_STEP1 = 50000;
static constexpr SizeType BINARY_ELE_NUM_2 = 7000000;
static constexpr SizeType BINARY_ELE_NUM_STEP2 = 100000;
static constexpr SizeType BINARY_ELE_NUM_STEP3 = 125000;

static constexpr SizeType BINARY_THREADS_THREASHOLD1 = 8;
static constexpr SizeType BINARY_THREADS_THREASHOLD2 = 40;

class BinaryLayerImpl final : public Parallelizable {
public:
    BinaryLayerImpl(const TensorInfo &src0Info, const TensorInfo &src1Info,
        const TensorInfo &dstInfo, BinaryFunction op) noexcept(false)
        : opBinary(op), src0Type(src0Info.GetType()), src1Type(src1Info.GetType()), dstType(dstInfo.GetType()),
        compType(Element::TypeT::F32)
    {
        Service::ThrowOnStatus(BinaryLayer::ValidateInput(src0Info, src1Info, dstInfo, op), "Binary Layer");

        Layout commLayout = src0Info.GetLayout();
        if ((src0Info.GetLayout() != src1Info.GetLayout()) || (src0Info.GetLayout() != dstInfo.GetLayout())) {
            isSeqKernel = true;
            for (Layout i : { Layout::A, Layout::AB, Layout::BA, Layout::ABC, Layout::ACB, Layout::BAC, Layout::BCA,
                            Layout::CAB, Layout::CBA, Layout::ABCD, Layout::ABDC, Layout::ACBD, Layout::ACDB,
                            Layout::ADBC, Layout::ADCB, Layout::BACD, Layout::BCDA, Layout::CDAB, Layout::CDBA,
                            Layout::DCAB, Layout::ABCDE, Layout::ABCED, Layout::ABDEC, Layout::ACBDE, Layout::ACDEB,
                            Layout::ADECB, Layout::BACDE, Layout::BCDEA, Layout::CDEAB, Layout::CDEBA, Layout::DECAB }
            ) {
                if (TensorInfo::ValidateDimsAndStrides<true>(src0Info.GetDims(), i, src0Info.GetStrides()) ==
                   Status::SUCCESS) {
                    if (TensorInfo::ValidateDimsAndStrides<true>(src1Info.GetDims(), i, src1Info.GetStrides()) ==
                       Status::SUCCESS) {
                        if (TensorInfo::ValidateDimsAndStrides<true>(dstInfo.GetDims(), i, dstInfo.GetStrides()) ==
                           Status::SUCCESS) {
                            isSeqKernel = false;
                            commLayout = i;
                            break;
                        }
                    }
                }
            }
        } else {
            isSeqKernel = false;
        }

        if (isSeqKernel) {
            ReorderLayerDiffLayout(src0Info, src1Info, dstInfo);
        } else {
            ReorderLayerSameLayout(src0Info, src1Info, dstInfo, commLayout, op);
        }

        compType = ((src0Type == dstType) && (src1Type == dstType) && (Element::TypeT::F16 == dstType)) ?
            Element::TypeT::F16 : Element::TypeT::F32;
        needToConvert = ((src0Type != dstType) || (src1Type != dstType));
        Parallelizable::Initialize(GetOptNumThreads(), reorderedOut);
    }

    template <typename CompType, typename Src0Type, typename Src1Type, typename OutType>
    void Evaluate(const Shape &batchSplitDims, const void *src0,
        const void *src1, void *dst, const float scale0, const float scale1) const noexcept
    {
        const Src0Type *typedSrc0 = static_cast<const Src0Type *>(src0);
        const Src1Type *typedSrc1 = static_cast<const Src1Type *>(src1);
        OutType *typedDst = static_cast<OutType *>(dst);
        switch (opBinary) {
            case BinaryFunction::ADD: {
                Binary::EvaluateAdd<CompType>(typedSrc0, typedSrc1, typedDst, scale0, scale1,
                    batchSplitDims, reorderedStridesSrc0, reorderedStridesSrc1, reorderedStridesDst, isSeqKernel);
                break;
            }
            case BinaryFunction::DIV: {
                Binary::EvaluateDiv<CompType>(typedSrc0, typedSrc1, typedDst, scale0, scale1,
                    batchSplitDims, reorderedStridesSrc0, reorderedStridesSrc1, reorderedStridesDst, isSeqKernel);
                break;
            }
            case BinaryFunction::MAX: {
                Binary::EvaluateMax<CompType>(typedSrc0, typedSrc1, typedDst, scale0, scale1,
                    batchSplitDims, reorderedStridesSrc0, reorderedStridesSrc1, reorderedStridesDst, isSeqKernel);
                break;
            }
            case BinaryFunction::MIN: {
                Binary::EvaluateMin<CompType>(typedSrc0, typedSrc1, typedDst, scale0, scale1,
                    batchSplitDims, reorderedStridesSrc0, reorderedStridesSrc1, reorderedStridesDst, isSeqKernel);
                break;
            }
            case BinaryFunction::MUL: {
                Binary::EvaluateMul<CompType>(typedSrc0, typedSrc1, typedDst, scale0, scale1,
                    batchSplitDims, reorderedStridesSrc0, reorderedStridesSrc1, reorderedStridesDst, isSeqKernel);
                break;
            }
            case BinaryFunction::SUB: {
                Binary::EvaluateSub<CompType>(typedSrc0, typedSrc1, typedDst, scale0, scale1,
                    batchSplitDims, reorderedStridesSrc0, reorderedStridesSrc1, reorderedStridesDst, isSeqKernel);
                break;
            }
            case BinaryFunction::GE: {
                Binary::EvaluateGe<CompType>(typedSrc0, typedSrc1, typedDst, scale0, scale1,
                    batchSplitDims, reorderedStridesSrc0, reorderedStridesSrc1, reorderedStridesDst, isSeqKernel);
                break;
            }
            case BinaryFunction::GT: {
                Binary::EvaluateGt<CompType>(typedSrc0, typedSrc1, typedDst, scale0, scale1,
                    batchSplitDims, reorderedStridesSrc0, reorderedStridesSrc1, reorderedStridesDst, isSeqKernel);
                break;
            }
            case BinaryFunction::LE: {
                Binary::EvaluateLe<CompType>(typedSrc0, typedSrc1, typedDst, scale0, scale1,
                    batchSplitDims, reorderedStridesSrc0, reorderedStridesSrc1, reorderedStridesDst, isSeqKernel);
                break;
            }
            case BinaryFunction::LT: {
                Binary::EvaluateLt<CompType>(typedSrc0, typedSrc1, typedDst, scale0, scale1,
                    batchSplitDims, reorderedStridesSrc0, reorderedStridesSrc1, reorderedStridesDst, isSeqKernel);
                break;
            }
            case BinaryFunction::EQ: {
                Binary::EvaluateEq<CompType>(typedSrc0, typedSrc1, typedDst, scale0, scale1,
                    batchSplitDims, reorderedStridesSrc0, reorderedStridesSrc1, reorderedStridesDst, isSeqKernel);
                break;
            }
            case BinaryFunction::NE: {
                Binary::EvaluateNe<CompType>(typedSrc0, typedSrc1, typedDst, scale0, scale1,
                    batchSplitDims, reorderedStridesSrc0, reorderedStridesSrc1, reorderedStridesDst, isSeqKernel);
                break;
            }
            default: {}
                break;
        }
    }

    void RunSequential(const Shape &batchSplitDims, const void *src0,
        const void *src1, void *dst, const float scale0, const float scale1) const noexcept
    {
        bool hasScale = false;
        if ((scale0 != 1.0f) || (scale1 != 1.0f)) {
            hasScale = true;
        }
        if (needToConvert || hasScale) {
            Service::ParseTypeAdvanced<true>(Element::TypeT::F32, Element::TypeT::F32, Element::TypeT::F32,
            Element::TypeT::F32, *this, batchSplitDims, src0, src1, dst, scale0, scale1);
        } else {
            Service::ParseTypeAdvanced<true>(compType, src0Type, src1Type, dstType, *this, batchSplitDims,
                src0, src1, dst, scale0, scale1);
        }
    }

    void RunSequential(const Shape &batchSplitDims, const Shape &batchStrides, const void *src0,
        const void *src1, void *dst, const float scale0, const float scale1) const noexcept(false)
    {
        bool hasScale = false;
        if ((scale0 != 1.0f) || (scale1 != 1.0f)) {
            hasScale = true;
        }
        SizeType strideSrc0 = 0;
        SizeType strideSrc1 = 0;
        SizeType strideDst = 0;
        SizeType src0Size = (needToConvert || hasScale) ? sizeof(float) : src0Type.GetSize();
        SizeType src1Size = (needToConvert || hasScale) ? sizeof(float) : src1Type.GetSize();
        SizeType dstSize = (needToConvert || hasScale) ? sizeof(float) : dstType.GetSize();
        for (SizeType i = 0; i < batchStrides.GetNumDims(); ++i) {
            strideSrc0 += batchStrides[i] * reorderedStridesSrc0[i] * src0Size;
            strideSrc1 += batchStrides[i] * reorderedStridesSrc1[i] * src1Size;
            strideDst += batchStrides[i] * reorderedStridesDst[i] * dstSize;
        }
        RunSequential(batchSplitDims, static_cast<const std::byte *>(src0) + strideSrc0,
            static_cast<const std::byte *>(src1) + strideSrc1,
            static_cast<std::byte *>(dst) + strideDst, scale0, scale1);
    }

    void Run(const void *src0, const void *src1, void *dst,
        const float scale0, const float scale1) const noexcept(false)
    {
        if ((src0 == nullptr) || (src1 == nullptr) || (dst == nullptr)) {
            throw Service::LogicError {"Binary Layer: one of input matrices is nullptr"};
        }
        bool hasScale = false;
        if ((scale0 != 1.0f) || (scale1 != 1.0f)) {
            hasScale = true;
        }
        const void *srcExec0 = src0;
        const void *srcExec1 = src1;
        void *dstExec = dst;
        if (needToConvert || hasScale) {
            void *tmp = Service::AlignedAlloc(reorderedSrc0.GetTotalDimsSize() * sizeof(float));
            ReorderLayer r0({reorderedSrc0, src0Type, Layout::ABCDE},
                {reorderedSrc0, Element::TypeT::F32, Layout::ABCDE});
            r0.Run(src0, tmp);
            srcExec0 = tmp;
            tmp = Service::AlignedAlloc(reorderedSrc1.GetTotalDimsSize() * sizeof(float));
            ReorderLayer r1({reorderedSrc1, src1Type, Layout::ABCDE},
                {reorderedSrc1, Element::TypeT::F32, Layout::ABCDE});
            r1.Run(src1, tmp);
            srcExec1 = tmp;
            dstExec = Service::AlignedAlloc(reorderedOut.GetTotalDimsSize() * sizeof(float));
        }
        Threading::BatchThreader(*this, srcExec0, srcExec1, dstExec, scale0, scale1);
        if (needToConvert || hasScale) {
            ReorderLayer r({reorderedOut, Element::TypeT::F32, Layout::ABCDE}, {reorderedOut, dstType, Layout::ABCDE});
            r.Run(dstExec, dst);
            Service::Deallocate(srcExec0);
            Service::Deallocate(srcExec1);
            Service::Deallocate(dstExec);
        }
    }

private:
    BinaryFunction opBinary;
    Element::Type src0Type;
    Element::Type src1Type;
    Element::Type dstType;
    Element::Type compType;
    Shape reorderedSrc0;
    Shape reorderedSrc1;
    Shape reorderedOut;
    Shape reorderedStridesSrc0;
    Shape reorderedStridesSrc1;
    Shape reorderedStridesDst;
    bool isSeqKernel;
    bool needToConvert;

    void ReorderLayerDiffLayout(const TensorInfo &src0Info, const TensorInfo &src1Info,
        const TensorInfo &dstInfo) noexcept(false)
    {
        reorderedSrc0 = Service::BroadcastTo5D(src0Info.GetDims());
        reorderedSrc1 = Service::BroadcastTo5D(src1Info.GetDims());
        reorderedOut = Service::BroadcastTo5D(dstInfo.GetDims());
        reorderedStridesSrc0 = Service::BroadcastTo5D(src0Info.GetStrides(), false);
        reorderedStridesSrc1 = Service::BroadcastTo5D(src1Info.GetStrides(), false);
        reorderedStridesDst = Service::BroadcastTo5D(dstInfo.GetStrides(), false);
    }

    void ReorderLayerSameLayout(const TensorInfo &src0Info, const TensorInfo &src1Info,
        const TensorInfo &dstInfo, Layout &commLayout, BinaryFunction op) noexcept(false)
    {
        reorderedSrc0 = Service::BroadcastTo5D(Service::GetShapeAccordingToLayout(src0Info.GetDims(), commLayout));
        reorderedSrc1 = Service::BroadcastTo5D(Service::GetShapeAccordingToLayout(src1Info.GetDims(), commLayout));
        reorderedOut = Service::BroadcastTo5D(Service::GetShapeAccordingToLayout(dstInfo.GetDims(), commLayout));
        reorderedStridesSrc0 = Service::BroadcastTo5D(Service::GetShapeAccordingToLayout(src0Info.GetStrides(),
            commLayout), false);
        reorderedStridesSrc1 = Service::BroadcastTo5D(Service::GetShapeAccordingToLayout(src1Info.GetStrides(),
            commLayout), false);
        reorderedStridesDst = Service::BroadcastTo5D(Service::GetShapeAccordingToLayout(dstInfo.GetStrides(),
            commLayout), false);

        // For dense problems, we can collapse several innermost loops in different dimensions into a single loop to
        // have bigger range of the innermost loop. It significantly improves performance for the cases
        // when last dimensions is small (less than VC register elements qty).
        // We don't collapse outermost loops since it doesn't give perf speedup.
        Service::ThrowOnStatus(BinaryLayer::ValidateInput(src0Info, src1Info, dstInfo, op), "Binary Layer");
        bool isDenseProblem = src0Info.IsDense() && src1Info.IsDense() && dstInfo.IsDense();
        /*
            For dense problems, we can generalize loops in different dimensions into a single loop to
        fully vectorize.
            In situations where broadcasting is required, the situation becomes a little more complicated
        and does not necessarily reduce to a single loop. The following logic is used to deal with
        this situation.
            For the broadcast scenario, two adjacent dimensions can only be generalized if both either require
        broadcasting or do not require broadcasting.
        */
        if (isDenseProblem) {
            SizeType newDims[DIM_5] = {reorderedOut[IDX_0], reorderedOut[IDX_1], reorderedOut[IDX_2],
                reorderedOut[IDX_3], reorderedOut[IDX_4]};
            SizeType mergeLoops = reorderedOut[IDX_4];
            // When stride equal to zero, it means this dimension require broadcasting
            bool previousStatusSrc0 = (reorderedStridesSrc0[IDX_4] > 0);
            bool previousStatusSrc1 = (reorderedStridesSrc1[IDX_4] > 0);
            bool currentStatusSrc0 = false;
            bool currentStatusSrc1 = false;
            for (SizeType i = IDX_0; i <= IDX_3; ++i) {
                currentStatusSrc0 = (reorderedStridesSrc0[IDX_3 - i] > 0);
                currentStatusSrc1 = (reorderedStridesSrc1[IDX_3 - i] > 0);
                /*
                    Since we use 0 strides for dimensions == 1 for now, we need to break when dimensions == 1.
                    Otherwise, we will find the wrong element after simplify the problem.
                */
                if (reorderedOut[IDX_4 - i] == 1) {
                    break;
                }
                if ((previousStatusSrc0 == currentStatusSrc0) && (previousStatusSrc1 == currentStatusSrc1)) {
                    newDims[IDX_3 - i] = 1;
                    mergeLoops *= reorderedOut[IDX_3 - i];
                } else {
                    break;
                }
                previousStatusSrc0 = currentStatusSrc0;
                previousStatusSrc1 = currentStatusSrc1;
            }
            newDims[IDX_4] = mergeLoops;
            reorderedOut.ResetShape(newDims, DIM_5);
        }
    }

    SizeType GetOptNumThreads() const noexcept
    {
        SizeType totalSize = reorderedOut.GetTotalDimsSize();
        if (totalSize < BINARY_ELE_NUM_1) {
            return std::min(((totalSize + BINARY_ELE_NUM_STEP1 - 1) / BINARY_ELE_NUM_STEP1),
                static_cast<SizeType>(BINARY_THREADS_THREASHOLD1));
        } else if (totalSize < BINARY_ELE_NUM_2) {
            return std::min(((totalSize + BINARY_ELE_NUM_STEP2 - 1) / BINARY_ELE_NUM_STEP2),
                static_cast<SizeType>(BINARY_THREADS_THREASHOLD2));
        } else {
            return (totalSize + BINARY_ELE_NUM_STEP3 - 1) / BINARY_ELE_NUM_STEP3;
        }
    }
};

} // Detail

BinaryLayer::~BinaryLayer() noexcept {}

BinaryLayer::BinaryLayer(const TensorInfo &src0Info, const TensorInfo &src1Info,
    const TensorInfo &dstInfo, BinaryFunction op) noexcept(false) : pImpl(
        new Detail::BinaryLayerImpl(src0Info, src1Info, dstInfo, op)) {}

BinaryLayer::BinaryLayer(const BinaryLayer &other) noexcept(false)
    : pImpl(new Detail::BinaryLayerImpl{*(other.pImpl.get())}) {}

BinaryLayer::BinaryLayer(BinaryLayer &&other) noexcept : pImpl(std::move(other.pImpl))
{}

BinaryLayer& BinaryLayer::operator=(const BinaryLayer &other) noexcept(false)
{
    if (this != &other) {
        this->pImpl.reset(new Detail::BinaryLayerImpl{*(other.pImpl.get())});
    }
    return *this;
}

BinaryLayer& BinaryLayer::operator=(BinaryLayer &&other) noexcept
{
    if (this != &other) {
        this->pImpl = std::move(other.pImpl);
    }
    return *this;
}

void BinaryLayer::Run(const void *src0, const void *src1, void *dst,
    const float scale0, const float scale1) const noexcept(false)
{
    pImpl->Run(src0, src1, dst, scale0, scale1);
}

Status BinaryLayer::ValidateInput(const TensorInfo &src0Info, const TensorInfo &src1Info,
    const TensorInfo &dstInfo, BinaryFunction op) noexcept
{
    if (op == BinaryFunction::UNIMPLEMENTED) {
        return Status::UNSUPPORTED;
    }
    if (src0Info.HasStrideOverFastestDim() || src1Info.HasStrideOverFastestDim() ||
        dstInfo.HasStrideOverFastestDim()) {
        return Status::UNSUPPORTED;
    }
    if ((src0Info.GetNumDims() != src1Info.GetNumDims()) ||
        (src0Info.GetNumDims() != dstInfo.GetNumDims())) {
        return Status::BAD_ARGUMENTS;
    }
    if (dstInfo.GetTotalTensorSize() == 0) {
        return Status::UNSUPPORTED;
    }
    for (SizeType i = 0; i < dstInfo.GetNumDims(); ++i) {
        if ((((dstInfo.GetDims()[i] != src0Info.GetDims()[i]) && (src0Info.GetDims()[i] != 1))) ||
            (((dstInfo.GetDims()[i] != src1Info.GetDims()[i]) && (src1Info.GetDims()[i] != 1)))) {
            return Status::BAD_ARGUMENTS;
        }
    }
    return Status::SUCCESS;
}

} // KuDNN
