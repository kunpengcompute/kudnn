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
#include "kernels/lrn/kudnn_lrn_kernels.hpp"

namespace KuDNN {
namespace Detail {

static constexpr SizeType LRN_BWD_TOTAL_SIZE_1 = 1000;

static constexpr SizeType LRN_BWD_THREADS_THRESHOLD_1 = 1;
static constexpr float LRN_BWD_THREADS_COEFF_1 = 11.7f;
static constexpr float LRN_BWD_THREADS_COEFF_2 = -79.7;

class LocalResponseNormalizationLayerBWDImpl final : public Parallelizable {
public:
    LocalResponseNormalizationLayerBWDImpl(const TensorInfo &srcInfo, const TensorInfo &diffDstInfo,
        const TensorInfo &diffSrcInfo, float alpha,
        float beta, float k, SizeType localSize, LRNormAlgorithmKind algorithm) noexcept(false)
        : srcType(srcInfo.GetType()), diffDstType(diffDstInfo.GetType()),
        compType(Element::TypeT::F32), paramAlpha(alpha), paramBeta(beta), paramK(k), paramLocalSize(localSize),
        algKind(algorithm)
    {
        Service::ThrowOnStatus(LocalResponseNormalizationLayerBWD::ValidateInput(srcInfo, diffDstInfo, diffSrcInfo,
            alpha, beta, k, localSize, algorithm), "Local Response Normalization Layer");
        dimsSrc = Service::BroadcastToNCDHW(diffSrcInfo.GetDims());
        numDims = srcInfo.GetNumDims();
        stridesSrc = Service::BroadcastToNCDHW(srcInfo.GetStrides());
        stridesDiffSrc  = Service::BroadcastToNCDHW(diffSrcInfo.GetStrides());
        stridesDiffDst  = Service::BroadcastToNCDHW(diffDstInfo.GetStrides());
        axbFormatTag = srcInfo.GetLayout() == srcInfo.GetStandardAXBLayout();
        abxFormatTag = srcInfo.GetLayout() == srcInfo.GetStandardABXLayout();
        Parallelizable::Initialize(GetOptNumThreads(), BatchedDims());
    }

    template <typename CompType, typename InType, typename OutType>
    void Evaluate(const Shape &start, const Shape &end,
        const void *src, const void *diffDst, void *diffSrc) const noexcept
    {
        const InType *typedSrc = static_cast<const InType *>(src);
        InType *typedDiffSrc = static_cast<InType *>(diffSrc);
        const OutType *typedDiffDst = static_cast<const OutType *>(diffDst);
        if (axbFormatTag || (!abxFormatTag && algKind == LRNormAlgorithmKind::WITHIN)) {
            LRN::LocalResponseNormGenericAxbBWD<CompType>(typedSrc, typedDiffDst, typedDiffSrc, paramAlpha, paramBeta,
                paramK, paramLocalSize, start, end, stridesSrc, stridesDiffDst, stridesDiffSrc,
                dimsSrc, numDims, algKind);
        } else {
            LRN::LocalResponseNormGenericAbxBWD<CompType>(typedSrc, typedDiffDst, typedDiffSrc, paramAlpha, paramBeta,
                paramK, paramLocalSize, start, end, stridesSrc, stridesDiffDst, stridesDiffSrc,
                dimsSrc, numDims, algKind);
        }
    }

    void RunSequential(const Shape &batchSplitDims, const Shape &batchStrides, const void *src,
        const void *diffDst, void *diffSrc) const noexcept(false)
    {
        Shape start = batchStrides;
        SizeType array[MAX_DIMS];
        for (SizeType i = 0; i < batchSplitDims.GetNumDims(); ++i) {
            array[i] = start[i] + batchSplitDims[i];
        }
        Shape end(array, batchSplitDims.GetNumDims());
        Service::ParseType<false, false>(compType, srcType, diffDstType, *this,
            start, end, src, diffDst, diffSrc);
    }

    void RunSequential(const Shape &batchSplitDims, const void *src, const void *diffDst,
        void *diffSrc) const noexcept(false)
    {
        SizeType batchStrides[MAX_DIMS] = { 0, 0, 0, 0, 0 };
        RunSequential(batchSplitDims, Shape{batchStrides, batchSplitDims.GetNumDims()}, src, diffDst, diffSrc);
    }

    void Run(const void *src, const void *diffDst, void *diffSrc) const noexcept(false)
    {
        if ((src == nullptr) || (diffSrc == nullptr) || (diffDst == nullptr)) {
            throw Service::LogicError {"Local Response Normalization Layer: one of input matrices is nullptr"};
        }
        Threading::BatchThreader(*this, src, diffDst, diffSrc);
    }

private:
    Element::Type srcType;
    Element::Type diffDstType;
    Element::Type compType;
    Shape dimsSrc;
    Shape stridesSrc;
    Shape stridesDiffSrc;
    Shape stridesDiffDst;
    float paramAlpha;
    float paramBeta;
    float paramK;
    SizeType paramLocalSize;
    SizeType numDims;
    LRNormAlgorithmKind algKind;
    bool axbFormatTag;
    bool abxFormatTag;

    SizeType GetOptNumThreads() const noexcept
    {
        SizeType totalSize = dimsSrc.GetTotalDimsSize();
        if (totalSize < LRN_BWD_TOTAL_SIZE_1) {
            return LRN_BWD_THREADS_THRESHOLD_1;
        } else {
            return LRN_BWD_THREADS_COEFF_1 * std::log(totalSize) + LRN_BWD_THREADS_COEFF_2;
        }
    }

    Shape BatchedDims() const noexcept
    {
        std::array<SizeType, MAX_DIMS> batchedDImsArr;
        SizeType batchedAxis = MAX_DIMS;
        for (SizeType i = 0; i < batchedAxis; ++i) {
            batchedDImsArr[i] = dimsSrc[i];
        }
        return {batchedDImsArr.data(), batchedAxis};
    }
};

} // Detail

LocalResponseNormalizationLayerBWD::~LocalResponseNormalizationLayerBWD() noexcept {}

LocalResponseNormalizationLayerBWD::LocalResponseNormalizationLayerBWD(const TensorInfo &srcInfo,
    const TensorInfo &diffDstInfo, const TensorInfo &diffSrcInfo, float alpha, float beta,
    float k, SizeType localSize, LRNormAlgorithmKind algorithm) noexcept(false)
    : pImpl(new Detail::LocalResponseNormalizationLayerBWDImpl(srcInfo, diffDstInfo, diffSrcInfo,
    alpha, beta, k, localSize, algorithm)) {}

LocalResponseNormalizationLayerBWD::LocalResponseNormalizationLayerBWD(const LocalResponseNormalizationLayerBWD &other)
    noexcept(false)
    : pImpl(new Detail::LocalResponseNormalizationLayerBWDImpl{*(other.pImpl.get())}) {}

LocalResponseNormalizationLayerBWD::LocalResponseNormalizationLayerBWD(LocalResponseNormalizationLayerBWD &&other)
    noexcept
    : pImpl(std::move(other.pImpl)) {}

LocalResponseNormalizationLayerBWD& LocalResponseNormalizationLayerBWD::operator=(const
    LocalResponseNormalizationLayerBWD &other) noexcept(false)
{
    if (this != &other) {
        this->pImpl.reset(new Detail::LocalResponseNormalizationLayerBWDImpl{*(other.pImpl.get())});
    }
    return *this;
}

LocalResponseNormalizationLayerBWD& LocalResponseNormalizationLayerBWD::operator=(LocalResponseNormalizationLayerBWD
    &&other) noexcept
{
    if (this != &other) {
        this->pImpl = std::move(other.pImpl);
    }
    return *this;
}

void LocalResponseNormalizationLayerBWD::Run(const void *src, const void *diffDst, void *diffSrc) const noexcept(false)
{
    pImpl->Run(src, diffDst, diffSrc);
}

Status LocalResponseNormalizationLayerBWD::ValidateInput(const TensorInfo &srcInfo, const TensorInfo &diffDstInfo,
    const TensorInfo &diffSrcInfo, float alpha, float beta, float k, SizeType localSize,
    LRNormAlgorithmKind algorithm) noexcept
{
    Service::Unused(diffSrcInfo, alpha, beta, k, algorithm);
    if (srcInfo.GetLayout() != diffDstInfo.GetLayout()) {
        return Status::UNSUPPORTED;
    }
    if (srcInfo.GetType() != diffDstInfo.GetType()) {
        return Status::UNSUPPORTED;
    }
    bool isBadArguments = (localSize <= 0) || ((srcInfo.GetType() != Element::TypeT::F32) &&
        (srcInfo.GetType() != Element::TypeT::F16) && (srcInfo.GetType() != Element::TypeT::BF16));
    if (isBadArguments) {
        return Status::BAD_ARGUMENTS;
    }
    if (srcInfo.GetDims() != diffDstInfo.GetDims()) {
        return Status::BAD_ARGUMENTS;
    }
    if (srcInfo.GetTotalTensorSize() == 0) {
        return Status::UNSUPPORTED;
    }
    return Status::SUCCESS;
}

} // KuDNN
