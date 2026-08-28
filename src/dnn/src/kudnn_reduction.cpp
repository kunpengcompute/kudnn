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
#include <array>

#include <arm_neon.h>
#include <arm_sve.h>

#include "kudnn.hpp"

#include "internal_funcs.hpp"
#include "kernels/kernel_funcs.h"
#include "vector_wrappers.hpp"
#include "threading_impl.hpp"
#include "kernels/reduction/kudnn_reduction_kernels.hpp"

namespace KuDNN {
namespace Detail {

static constexpr SizeType REDUCTION_ELE_NUM_1 = 500000;
#ifdef KUDNNL_920F
static constexpr SizeType REDUCTION_ELE_NUM_STEP1 = 100;
#else
static constexpr SizeType REDUCTION_ELE_NUM_STEP1 = 25000;
#endif
static constexpr SizeType REDUCTION_ELE_NUM_2 = 10000000;

static constexpr SizeType REDUCTION_THREADS_THREASHOLD1 = 40;
static constexpr SizeType REDUCTION_THREADS_THREASHOLD2 = 80;
static constexpr SizeType REDUCTION_THREADS_THREASHOLD3 = 160;


class ReductionLayerImpl final : public Parallelizable {
public:
    ReductionLayerImpl(const TensorInfo &srcInfo, const TensorInfo &dstInfo,
        ReductionFunction kind, float pPara, float epsPara) noexcept(false)
        : srcType(srcInfo.GetType()), dstType(dstInfo.GetType()), compType(Element::TypeT::F32), kindInfo(kind),
        idleSize(dstInfo.GetTotalTensorSize()), reduceSize(srcInfo.GetTotalTensorSize() / idleSize),
        p(pPara), eps(epsPara)
    {
        Service::ThrowOnStatus(ReductionLayer::ValidateInput(srcInfo, dstInfo, kindInfo, p, eps), "Reduction Layer");
        if (dstType != Element::TypeT::F32) {
            dstTmpV.resize(dstInfo.GetTotalTensorSize(), GetAcc());
        }
#ifdef KUDNNL_920F
        ReorderLastOnesDims(srcInfo, dstInfo);
#else
        ReorderDims(srcInfo, dstInfo, kind);
#endif

        srcShape = srcInfo.GetDims();
        dstShape = dstInfo.GetDims();
        srcLayout = srcInfo.GetLayout();
        dstLayout = dstInfo.GetLayout();

        Parallelizable::Initialize(GetOptNumThreads(), reorderedDimsDst);
    }

    void ReorderDims(const TensorInfo &srcInfo, const TensorInfo &dstInfo, ReductionFunction kind) noexcept(false)
    {
        if (kind == ReductionFunction::NORM_LP_MAX || kind == ReductionFunction::NORM_LP_POWER_P_MAX ||
            kind == ReductionFunction::NORM_LP_SUM || kind == ReductionFunction::NORM_LP_POWER_P_SUM) {
            reorderedDimsSrc = Service::BroadcastTo5D(srcInfo.GetDims());
            reorderedDimsDst = Service::BroadcastTo5D(dstInfo.GetDims());
            reorderedStridesSrc = Service::BroadcastTo5D(srcInfo.GetStrides(), false);
            reorderedStridesDst  = Service::BroadcastTo5D(dstInfo.GetStrides(), false);
        } else {
            ReorderLastOnesDims(srcInfo, dstInfo);
        }
    }

    void ReorderLastOnesDims(const TensorInfo &srcInfo, const TensorInfo &dstInfo) noexcept(false)
    {
        reorderedDimsSrc = Service::BroadcastTo5D(Service::GetShapeAccordingToLayout(srcInfo.GetDims(),
            srcInfo.GetLayout()));
        reorderedDimsDst = Service::BroadcastTo5D(Service::GetShapeAccordingToLayout(dstInfo.GetDims(),
            dstInfo.GetLayout()));
        reorderedStridesSrc = Service::BroadcastTo5D(Service::GetShapeAccordingToLayout(srcInfo.GetStrides(),
            srcInfo.GetLayout()), false);
        reorderedStridesDst  = Service::BroadcastTo5D(Service::GetShapeAccordingToLayout(dstInfo.GetStrides(),
            dstInfo.GetLayout()), false);

        SizeType lastNoneOneIdx = reorderedDimsDst.GetNumDims() - 1;
        for (SizeType i = 0; i < reorderedDimsDst.GetNumDims(); ++i) {
            if (reorderedDimsDst[reorderedDimsDst.GetNumDims() - 1 - i] == 1) {
                lastNoneOneIdx = reorderedDimsDst.GetNumDims() - 1 - i;
            } else {
                break;
            }
        }

        if (lastNoneOneIdx != (reorderedDimsDst.GetNumDims() - 1)) {
            SizeType tempShapeSrc[MAX_DIMS];
            SizeType tempShapeDst[MAX_DIMS];
            SizeType tempStridesSrc[MAX_DIMS];
            SizeType tempStridesDst[MAX_DIMS];
            for (SizeType i = 0; i < lastNoneOneIdx; ++i) {
                tempShapeSrc[i] = reorderedDimsSrc[i];
                tempShapeDst[i] = reorderedDimsDst[i];
                tempStridesSrc[i] = reorderedStridesSrc[i];
                tempStridesDst[i] = reorderedStridesDst[i];
            }
            tempShapeSrc[lastNoneOneIdx] = Service::ArrayProduct(reorderedDimsSrc, reorderedDimsSrc.GetNumDims(),
                                                                lastNoneOneIdx);
            tempShapeDst[lastNoneOneIdx] = 1;
            tempStridesSrc[lastNoneOneIdx] = 1;
            tempStridesDst[lastNoneOneIdx] = 0;

            reorderedDimsSrc = Service::BroadcastTo5D(Shape(tempShapeSrc, lastNoneOneIdx + 1));
            reorderedDimsDst = Service::BroadcastTo5D(Shape(tempShapeDst, lastNoneOneIdx + 1));
            reorderedStridesSrc = Service::BroadcastTo5D(Shape(tempStridesSrc, lastNoneOneIdx + 1), false);
            reorderedStridesDst = Service::BroadcastTo5D(Shape(tempStridesDst, lastNoneOneIdx + 1), false);
        }
    }

    template <typename CompType, typename InType, typename OutType>
    void Evaluate(const Shape &start, const Shape &end, const void *src, void *dst, float* dstBuf) noexcept
    {
        const InType *typedSrc = static_cast<const InType *>(src);
        OutType *typedDst = static_cast<OutType *>(dst);
        switch (kindInfo) {
            case ReductionFunction::MAX: {
                Reduction::EvaluateMax<CompType>(start, end, typedSrc, typedDst, dstBuf, GetAcc(),
                        kindInfo, reduceSize, idleSize, reorderedDimsSrc, reorderedDimsDst,
                        reorderedStridesSrc, reorderedStridesDst, p, eps);
                break;
            }
            case ReductionFunction::MIN: {
                Reduction::EvaluateMin<CompType>(start, end, typedSrc, typedDst, dstBuf, GetAcc(),
                        kindInfo, reduceSize, idleSize, reorderedDimsSrc, reorderedDimsDst,
                        reorderedStridesSrc, reorderedStridesDst, p, eps);
                break;
            }
            case ReductionFunction::MEAN: {
                Reduction::EvaluateMean<CompType>(start, end, typedSrc, typedDst, dstBuf, GetAcc(),
                        kindInfo, reduceSize, idleSize, reorderedDimsSrc, reorderedDimsDst,
                        reorderedStridesSrc, reorderedStridesDst, p, eps);
                break;
            }
            case ReductionFunction::SUM: {
                Reduction::EvaluateSum<CompType>(start, end, typedSrc, typedDst, dstBuf, GetAcc(),
                        kindInfo, reduceSize, idleSize, reorderedDimsSrc, reorderedDimsDst,
                        reorderedStridesSrc, reorderedStridesDst, p, eps);
                break;
            }
            case ReductionFunction::MUL: {
                Reduction::EvaluateMul<CompType>(start, end, typedSrc, typedDst, dstBuf, GetAcc(),
                        kindInfo, reduceSize, idleSize, reorderedDimsSrc, reorderedDimsDst,
                        reorderedStridesSrc, reorderedStridesDst, p, eps);
                break;
            }
            default: {
                break;
            }
        }
    }

    template <typename CompType, typename InType, typename OutType>
    void Evaluate(const Shape &start, const Shape &end, const void *src, void *dst, float* dstBuf,
        const bool isSupportInt) noexcept
    {
        Service::Unused(isSupportInt);
        const InType *typedSrc = static_cast<const InType *>(src);
        OutType *typedDst = static_cast<OutType *>(dst);
        Reduction::EvaluateNormLp<CompType>(start, end, typedSrc, typedDst, dstBuf, GetAcc(),
            kindInfo, reduceSize, idleSize, reorderedDimsSrc, reorderedDimsDst,
            reorderedStridesSrc, reorderedStridesDst, p, eps);
    }

    void RunSequential(const Shape &batchSplitDims, const Shape &batchStrides,
                       const void *src, void *dst, float *dstBuf) noexcept(false)
    {
        Shape start = batchStrides;
        SizeType array[MAX_DIMS];
        for (SizeType i = 0; i < batchSplitDims.GetNumDims(); ++i) {
            array[i] = start[i] + batchSplitDims[i];
        }
        Shape end(array, batchSplitDims.GetNumDims());

        bool isSupportInt = (kindInfo == ReductionFunction::MAX) || (kindInfo == ReductionFunction::MUL) ||
            (kindInfo == ReductionFunction::MIN) || (kindInfo == ReductionFunction::MEAN) ||
            (kindInfo == ReductionFunction::SUM);
        if (isSupportInt) {
            if (srcType != dstType) {
                Service::ParseType<true, false>(compType, Element::TypeT::F32, Element::TypeT::F32,
                    *this, start, end, src, dst, dstBuf);
            } else {
                Service::ParseType<true, false>(compType, srcType, dstType, *this, start, end, src, dst, dstBuf);
            }
        } else {
            if (srcType != dstType) {
                Service::ParseType<false, false>(compType, Element::TypeT::F32, Element::TypeT::F32,
                    *this, start, end, src, dst, dstBuf, false);
            } else {
                Service::ParseType<false, false>(compType, srcType, dstType,
                    *this, start, end, src, dst, dstBuf, false);
            }
        }
    }

    void RunSequential(const Shape &batchSplitDims, const void *src, void *dst, float* dstBuf) noexcept
    {
        SizeType batchStrides[MAX_DIMS] = { 0, 0, 0, 0, 0 };
        RunSequential(batchSplitDims, Shape{batchStrides, batchSplitDims.GetNumDims()}, src, dst, dstBuf);
    }

    void Run(const void *src, void *dst) noexcept(false)
    {
        if ((src == nullptr) || (dst == nullptr)) {
            throw Service::LogicError {"Reduction Layer: one of input matrices is nullptr"};
        }
        if (src == dst) {
            throw Service::LogicError {"Reduction Layer: input and output must not be the same pointers"};
        }
        const void *srcExec = src;
        void *dstExec = dst;
        if ((srcLayout != dstLayout) || (srcType != dstType)) {
            Element::Type dtSrc = (srcType != dstType) ? Element::Type{Element::TypeT::F32} : srcType;
            Element::Type dtDst = (srcType != dstType) ? Element::Type{Element::TypeT::F32} : dstType;
            dstExec = Service::AlignedAlloc(dstShape.GetTotalDimsSize() * dtDst.GetSize());
            void *tmp = Service::AlignedAlloc(srcShape.GetTotalDimsSize() * dtSrc.GetSize());
            ReorderLayer r({srcShape, srcType, srcLayout}, {srcShape, dtSrc, srcLayout});
            r.Run(src, tmp);
            srcExec = tmp;
        }
        if (srcType != dstType || dstType == Element::TypeT::F32) {
            std::fill(static_cast<float*>(dstExec),
                static_cast<float*>(dstExec) + reorderedDimsDst.GetTotalDimsSize(), GetAcc());
        }
        Threading::BatchThreader(*this, srcExec, dstExec, dstTmpV.data());
        if ((srcLayout != dstLayout) || (srcType != dstType)) {
            Element::Type dtDst = (srcType != dstType) ? Element::Type{Element::TypeT::F32} : dstType;
            ReorderLayer r({dstShape, dtDst, srcLayout}, {dstShape, dstType, dstLayout});
            r.Run(dstExec, dst);
            Service::Deallocate(dstExec);
            Service::Deallocate(srcExec);
        }
    }

private:
    Element::Type srcType;
    Element::Type dstType;
    Element::Type compType;
    ReductionFunction kindInfo;
    SizeType idleSize;
    SizeType reduceSize;
    float p;
    float eps;
    std::vector<float, Service::AlignedAllocator<float>> dstTmpV;
    Shape reorderedDimsSrc;
    Shape reorderedDimsDst;
    Shape reorderedStridesSrc;
    Shape reorderedStridesDst;
    Shape srcShape;
    Shape dstShape;
    Layout srcLayout;
    Layout dstLayout;

    SizeType GetOptNumThreads() const noexcept
    {
        SizeType totalSrcSize = reorderedDimsSrc.GetTotalDimsSize();
        if (totalSrcSize < REDUCTION_ELE_NUM_1) {
            return std::min(((totalSrcSize + REDUCTION_ELE_NUM_STEP1 - 1) / REDUCTION_ELE_NUM_STEP1),
                static_cast<SizeType>(REDUCTION_THREADS_THREASHOLD1));
        } else if (totalSrcSize < REDUCTION_ELE_NUM_2) {
            return REDUCTION_THREADS_THREASHOLD2;
        } else {
            return REDUCTION_THREADS_THREASHOLD3;
        }
    }

    float GetAcc() const noexcept
    {
        switch (kindInfo) {
            case ReductionFunction::MAX: {
                return std::numeric_limits<float>::lowest();
            }
            case ReductionFunction::MIN: {
                return std::numeric_limits<float>::max();
            }
            case ReductionFunction::MEAN: {
                return 0.0f;
            }
            case ReductionFunction::SUM: {
                return 0.0f;
            }
            case ReductionFunction::MUL: {
                return 1.0f;
            }
            case ReductionFunction::NORM_LP_MAX: {
                return 0.0f;
            }
            case ReductionFunction::NORM_LP_SUM: {
                return 0.0f;
            }
            case ReductionFunction::NORM_LP_POWER_P_MAX: {
                return 0.0f;
            }
            case ReductionFunction::NORM_LP_POWER_P_SUM: {
                return 0.0f;
            }
            default: {
                return 0.0f;
            }
        }
    }
};

} // Detail

ReductionLayer::~ReductionLayer() noexcept {}

ReductionLayer::ReductionLayer(const TensorInfo &srcInfo, const TensorInfo &dstInfo,
    ReductionFunction kind, float p, float eps) noexcept(false)
    : pImpl(new Detail::ReductionLayerImpl(srcInfo, dstInfo, kind, p, eps)) {}

ReductionLayer::ReductionLayer(const ReductionLayer &other) noexcept(false)
    : pImpl(new Detail::ReductionLayerImpl{*(other.pImpl.get())}) {}

ReductionLayer::ReductionLayer(ReductionLayer &&other) noexcept
    : pImpl(std::move(other.pImpl)) {}

ReductionLayer& ReductionLayer::operator=(const ReductionLayer &other) noexcept(false)
{
    if (this != &other) {
        this->pImpl.reset(new Detail::ReductionLayerImpl{*(other.pImpl.get())});
    }
    return *this;
}

ReductionLayer& ReductionLayer::operator=(ReductionLayer &&other) noexcept
{
    if (this != &other) {
        this->pImpl = std::move(other.pImpl);
    }
    return *this;
}

void ReductionLayer::Run(const void *src, void *dst) const noexcept(false)
{
    pImpl->Run(src, dst);
}

Status ReductionLayer::ValidateInput(const TensorInfo &srcInfo, const TensorInfo &dstInfo,
    const ReductionFunction& reductionAlg, float p, float eps) noexcept
{
    Service::Unused(p);
    Service::Unused(eps);
    if ((reductionAlg == ReductionFunction::UNIMPLEMENTED)) {
        return Status::UNSUPPORTED;
    }
    if (!srcInfo.IsDense() && !dstInfo.IsDense()) {
        return Status::UNSUPPORTED;
    }
    if (srcInfo.GetNumDims() != dstInfo.GetNumDims()) {
        return Status::UNSUPPORTED;
    }
    if (srcInfo.GetTotalTensorSize() == 0) {
        return Status::UNSUPPORTED;
    }
    if (dstInfo.GetTotalTensorSize() == 0) {
        return Status::UNSUPPORTED;
    }
    return Status::SUCCESS;
}

} // KuDNN
