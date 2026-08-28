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
#include <cstdint>

#include "kudnn.hpp"

#include "internal_funcs.hpp"
#include "threading_impl.hpp"

#include "kernels/pooling/kudnn_pooling_kernels.hpp"

namespace KuDNN {
namespace Detail {

static constexpr SizeType POOL_ELE_NUM_BWD = 300000;

class PoolingLayerBWDImpl final : public Parallelizable {
public:
    PoolingLayerBWDImpl(const PoolingFunction &poolingAlg,
        const TensorInfo &srcDiffInfo, const TensorInfo &dstDiffInfo, const Shape &stridesPara,
        const Shape &kernelPara, const Shape &dilationPara, const Shape &paddingLPara,
        const Shape &paddingRPara) noexcept(false)
        :  srcDiffType(srcDiffInfo.GetType()), dstDiffType(dstDiffInfo.GetType()), wsType(Element::TypeT::S32),
        compType(Element::TypeT::F32), alg(poolingAlg), strides(stridesPara), kernel(kernelPara),
        dilation(dilationPara), paddingL(paddingLPara), paddingR(paddingRPara), ndims(srcDiffInfo.GetNumDims()),
        useWs(alg == PoolingFunction::MAX)
    {
        Service::ThrowOnStatus(PoolingLayerBWD::ValidateInput(poolingAlg, srcDiffInfo,
            dstDiffInfo, strides, kernel, dilation, paddingL, paddingR), "Pooling Layer");

        if (alg == PoolingFunction::MAX) {
            wsType = Service::ArrayProduct(kernel, ndims - DIM_2, 0) <= Pooling::Helpers::U8_MAX ?
                Element::TypeT::U8 : Element::TypeT::S32;
        }

        initialDims = Service::BroadcastTo5D(srcDiffInfo.GetDims());
        reorderedDims = Service::BroadcastTo5D(dstDiffInfo.GetDims());
        reorderedStridesSrc = Service::BroadcastTo5D(srcDiffInfo.GetStrides(), false);
        reorderedStridesDst  = Service::BroadcastTo5D(dstDiffInfo.GetStrides(), false);
        reorderedKernel = Service::BroadcastTo5D(kernel, false);
        reorderedStrides = Service::BroadcastTo5D(strides, false);
        reorderedDilation = Service::BroadcastTo5D(dilation, false);
        reorderedPaddingL = Service::BroadcastTo5D(paddingL, false);
        reorderedPaddingR = Service::BroadcastTo5D(paddingR, false);
        axbFormatTag = srcDiffInfo.GetLayout() == srcDiffInfo.GetStandardAXBLayout();

        if (srcDiffType != Element::TypeT::F32) {
            srcTmpV.resize(srcDiffInfo.GetTotalTensorSize(), 0.0f);
        }

        std::array<SizeType, MAX_DIMS> parallelDims = {dstDiffInfo.GetDims()[0], dstDiffInfo.GetDims()[1], 1, 1, 1};
        Parallelizable::Initialize(GetOptNumThreads(dstDiffInfo.GetDims()),
            Service::BroadcastTo5D({parallelDims.data(), dstDiffInfo.GetNumDims()}));
    }

    template <PoolingFunction Fun, typename CompType, typename InType, typename OutType, typename WsType>
    void EvaluateDims(const Shape &start, const Shape &end,
        InType *src, const OutType *dst, const WsType *ws, float *srcTmpBuf) noexcept
    {
        if (axbFormatTag) {
            Pooling::EvaluateAxbBWD<Fun, CompType, InType, OutType, WsType>(
                start, end, src, dst, ws, srcTmpBuf, initialDims, reorderedDims,
                reorderedStridesSrc, reorderedStridesDst, kernel,
                strides, dilation, paddingL, ndims);
        } else {
            Pooling::EvaluateAbxBWD<Fun, CompType, InType, OutType, WsType>(
                start, end, src, dst, ws, srcTmpBuf, initialDims, reorderedDims,
                reorderedStridesSrc, reorderedStridesDst, kernel,
                strides, dilation, paddingL, ndims);
        }
    }

    template <PoolingFunction Fun, typename CompType, typename InType, typename OutType>
    void EvaluateND(const Shape &start, const Shape &end,
        InType *diffSrc, const OutType *diffDst, const void *ws, float *srcTmpBuf) noexcept
    {
        if (wsType == Element::TypeT::S32) {
            const std::int32_t *typedWs = static_cast<const std::int32_t *>(ws);
            EvaluateDims<Fun, CompType, InType, OutType, std::int32_t>(
                start, end, diffSrc, diffDst, typedWs, srcTmpBuf);
        } else {
                if (Fun == PoolingFunction::MAX) {
                    const std::uint8_t *typedWs = static_cast<const std::uint8_t *>(ws);
                    EvaluateDims<Fun, CompType, InType, OutType, std::uint8_t>(
                        start, end, diffSrc, diffDst, typedWs, srcTmpBuf);
                }
        }
    }

    template <typename CompType, typename InType, typename OutType>
    void Evaluate(const Shape &start, const Shape &end,
        void *diffSrc, const void *diffDst, const void *ws, float *srcTmpBuf) noexcept
    {
        InType *typedDiffSrc = static_cast<InType *>(diffSrc);
        const OutType *typedDiffDst = static_cast<const OutType *>(diffDst);

        switch (alg) {
            case PoolingFunction::MAX:
                EvaluateND<PoolingFunction::MAX, CompType, InType, OutType>(start, end,
                    typedDiffSrc, typedDiffDst, ws, srcTmpBuf);
                break;
            case PoolingFunction::AVG_INCLUDE_PADDING:
                EvaluateND<PoolingFunction::AVG_INCLUDE_PADDING, CompType, InType, OutType>(start, end,
                    typedDiffSrc, typedDiffDst, ws, srcTmpBuf);
                break;
            case PoolingFunction::AVG_EXCLUDE_PADDING:
                EvaluateND<PoolingFunction::AVG_EXCLUDE_PADDING, CompType, InType, OutType>(start, end,
                    typedDiffSrc, typedDiffDst, ws, srcTmpBuf);
                break;
            default:
                break;
        }
    }

    void RunSequential(const Shape &batchSplitDims, const Shape &batchStrides,
        void *diffSrc, const void *diffDst,
        const void *ws, float *srcTmpBuf) noexcept(false)
    {
        Shape start = batchStrides;
        SizeType array[MAX_DIMS];
        for (SizeType i = 0; i < batchSplitDims.GetNumDims(); ++i) {
            array[i] = start[i] + batchSplitDims[i];
        }
        Shape end(array, batchSplitDims.GetNumDims());

        if ((srcDiffType != dstDiffType)) {
            Service::ParseType<false, false>(compType, Element::TypeT::F32, Element::TypeT::F32, *this,
                start, end, diffSrc, diffDst, ws, srcTmpBuf);
        } else {
            Service::ParseType<false, false>(compType, srcDiffType, dstDiffType, *this,
                start, end, diffSrc, diffDst, ws, srcTmpBuf);
        }
    }

    void RunSequential(const Shape &batchSplitDims, void *diffSrc,
        const void *diffDst, const void *ws, float *srcTmpBuf) noexcept(false)
    {
        SizeType batchStrides[MAX_DIMS] = { 0, 0, 0, 0, 0 };
        RunSequential(batchSplitDims, Shape{batchStrides, batchSplitDims.GetNumDims()},
            diffSrc, diffDst, ws, srcTmpBuf);
    }

    void Run(void *diffSrc, const void *diffDst, const void *ws) noexcept(false)
    {
        if ((diffSrc == nullptr) || (diffDst == nullptr) || (useWs && (ws == nullptr))) {
            throw Service::LogicError {"Pooling Layer: one of input matrices is nullptr"};
        }
        if ((diffSrc == diffDst) || (useWs && (ws == diffDst))) {
            throw Service::LogicError {"Pooling Layer: input and output and workspace must not be the same pointers"};
        }
        if ((dstDiffType == Element::TypeT::F32) || (srcDiffType != dstDiffType)) {
            std::fill(static_cast<float*>(diffSrc), static_cast<float*>(diffSrc) +
            initialDims.GetTotalDimsSize(), 0.0f);
        }
        Threading::BatchThreader(*this, diffSrc, diffDst, ws, srcTmpV.data());
    }

private:
    Element::Type srcDiffType;
    Element::Type dstDiffType;
    Element::Type wsType;
    Element::Type compType;
    PoolingFunction alg;
    Shape strides;
    Shape kernel;
    Shape dilation;
    Shape paddingL;
    Shape paddingR;
    SizeType ndims;
    bool useWs;
    Shape initialDims;
    Shape reorderedDims;
    Shape reorderedStridesSrc;
    Shape reorderedStridesWs;
    Shape reorderedStridesDst;
    Shape reorderedKernel;
    Shape reorderedStrides;
    Shape reorderedDilation;
    Shape reorderedPaddingL;
    Shape reorderedPaddingR;
    bool axbFormatTag;
    std::vector<float, Service::AlignedAllocator<float>> srcTmpV;

    SizeType GetOptNumThreads(Shape dstDiffShape) const noexcept
    {
        SizeType batchSize = dstDiffShape[0] * dstDiffShape[1];
        SizeType totalSize = reorderedDims.GetTotalDimsSize();
        if (totalSize < POOL_ELE_NUM_BWD) {
            return std::min<IntType>(std::max<IntType>(NUM_7 * log(totalSize) - NUM_27, NUM_8),
                omp_get_max_threads());
        } else {
            if (batchSize < NUM_32) {
                return std::min<IntType>(std::max<IntType>(NUM_20 * log(batchSize) + NUM_3, NUM_8),
                    omp_get_max_threads());
            } else {
                return NUM_40;
            }
        }
    }

    static SizeType ComputeOutPooling(SizeType i, SizeType k, SizeType s, SizeType pl, SizeType pr, SizeType d)
    {
        return (i - ((k - 1) * (d + 1) + 1) + (pl + pr)) / s + 1;
    }
};

} // Detail

PoolingLayerBWD::~PoolingLayerBWD() noexcept {}

PoolingLayerBWD::PoolingLayerBWD(const PoolingFunction &poolingAlg,
    const TensorInfo &srcDiffInfo, const TensorInfo &dstDiffInfo, const Shape &strides,
    const Shape &kernel, const Shape &dilation, const Shape &paddingL,
    const Shape &paddingR) noexcept(false)
    : pImpl(new Detail::PoolingLayerBWDImpl(poolingAlg, srcDiffInfo, dstDiffInfo, strides,
        kernel, dilation, paddingL, paddingR)) {}

PoolingLayerBWD::PoolingLayerBWD(const PoolingLayerBWD &other) noexcept(false)
    : pImpl(new Detail::PoolingLayerBWDImpl{*(other.pImpl.get())}) {}

PoolingLayerBWD::PoolingLayerBWD(PoolingLayerBWD &&other) noexcept
    : pImpl(std::move(other.pImpl)) {}

PoolingLayerBWD& PoolingLayerBWD::operator=(const PoolingLayerBWD &other) noexcept(false)
{
    if (this != &other) {
        this->pImpl.reset(new Detail::PoolingLayerBWDImpl{*(other.pImpl.get())});
    }
    return *this;
}

PoolingLayerBWD& PoolingLayerBWD::operator=(PoolingLayerBWD &&other) noexcept
{
    if (this != &other) {
        this->pImpl = std::move(other.pImpl);
    }
    return *this;
}

void PoolingLayerBWD::Run(void *diffSrc, const void *diffDst, const void *ws) const noexcept(false)
{
    pImpl->Run(diffSrc, diffDst, ws);
}

static SizeType ComputeOutPooling(SizeType i, SizeType k, SizeType s, SizeType pl, SizeType pr, SizeType d)
{
    return (i - ((k - 1) * (d + 1) + 1) + (pl + pr)) / s + 1;
}

static Status ValidateAlgSizesPool(const Shape &argStrides, const Shape &argDilates, const Shape &argPaddingL,
    const Shape &argPaddingR, const Shape &kernel, const TensorInfo &diffSrc, const TensorInfo &diffDst)
{
    SizeType numDims = diffSrc.GetNumDims();
    SizeType ow = ComputeOutPooling(diffSrc.GetDims()[numDims - DIM_1], kernel[numDims - DIM_3],
        argStrides[numDims - DIM_3], argPaddingL[numDims - DIM_3],
        argPaddingR[numDims - DIM_3], argDilates[numDims - DIM_3]);
    if (diffDst.GetDims()[numDims - 1] != ow) {
        return Status::BAD_ARGUMENTS;
    }
    if (numDims >= DIM_4) {
        SizeType oh = ComputeOutPooling(diffSrc.GetDims()[numDims - DIM_2], kernel[numDims - DIM_4],
            argStrides[numDims - DIM_4], argPaddingL[numDims - DIM_4],
            argPaddingR[numDims - DIM_4], argDilates[numDims - DIM_4]);
        if (diffDst.GetDims()[numDims - DIM_2] != oh) {
            return Status::BAD_ARGUMENTS;
        }
    }
    if (numDims == DIM_5) {
        SizeType od = ComputeOutPooling(diffSrc.GetDims()[numDims - DIM_3], kernel[numDims - DIM_5],
            argStrides[numDims - DIM_5], argPaddingL[numDims - DIM_5],
            argPaddingR[numDims - DIM_5], argDilates[numDims - DIM_5]);
        if (diffDst.GetDims()[numDims - DIM_3] != od) {
            return Status::BAD_ARGUMENTS;
        }
    }
    return Status::SUCCESS;
}

Status PoolingLayerBWD::ValidateInput(const PoolingFunction &poolingAlg,
    const TensorInfo &srcDiffInfo, const TensorInfo &dstDiffInfo, const Shape &strides,
    const Shape &kernel, const Shape &dilation, const Shape &paddingL, const Shape &paddingR) noexcept
{
    bool ok = ((srcDiffInfo.GetType() == dstDiffInfo.GetType()) &&
                ((srcDiffInfo.GetType() == Element::TypeT::F32) ||
                (srcDiffInfo.GetType() == Element::TypeT::F16) ||
                (srcDiffInfo.GetType() == Element::TypeT::BF16)));
    if (!ok) {
        return Status::UNSUPPORTED;
    }

    if (poolingAlg == PoolingFunction::UNDEFINED) {
        return Status::BAD_ARGUMENTS;
    }
    bool isUnsupported = (srcDiffInfo.GetLayout() != dstDiffInfo.GetLayout());
    // Src Diff layout != Standard AXB and ABX Layout
    isUnsupported = isUnsupported || (srcDiffInfo.GetLayout() != srcDiffInfo.GetStandardABXLayout() &&
        srcDiffInfo.GetLayout() != srcDiffInfo.GetStandardAXBLayout());
    
    isUnsupported = isUnsupported || (!srcDiffInfo.IsDense() || !dstDiffInfo.IsDense());
    if (isUnsupported) {
        return Status::UNSUPPORTED;
    }
    if (srcDiffInfo.GetNumDims() != dstDiffInfo.GetNumDims()) {
        return Status::BAD_ARGUMENTS;
    }
    for (SizeType i = 0; i < strides.GetNumDims(); ++i) {
        if (strides[i] == 0) {
            return Status::BAD_ARGUMENTS;
        }
    }
    for (SizeType i = 0; i < kernel.GetNumDims(); ++i) {
        if (kernel[i] == 0) {
            return Status::BAD_ARGUMENTS;
        }
    }
    bool isBadArguments = kernel.GetNumDims() < DIM_1 || kernel.GetNumDims() > DIM_3;
    isBadArguments = isBadArguments || (strides.GetNumDims() < DIM_1 || strides.GetNumDims() > DIM_3);
    isBadArguments = isBadArguments || (dilation.GetNumDims() < DIM_1 || dilation.GetNumDims() > DIM_3);
    isBadArguments = isBadArguments || (paddingL.GetNumDims() < DIM_1 || paddingL.GetNumDims() > DIM_3);
    isBadArguments = isBadArguments || (paddingR.GetNumDims() < DIM_1 || paddingR.GetNumDims() > DIM_3);
    if (isBadArguments) {
        return Status::BAD_ARGUMENTS;
    }

    if (srcDiffInfo.GetTotalTensorSize() == 0) {
        return Status::UNSUPPORTED;
    }
    if (ValidateAlgSizesPool(strides, dilation, paddingL, paddingR, kernel,
        srcDiffInfo, dstDiffInfo) != Status::SUCCESS) {
        return Status::BAD_ARGUMENTS;
    }

    return Status::SUCCESS;
}

} // KuDNN
