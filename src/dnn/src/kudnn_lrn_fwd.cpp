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
#include "kernels/lrn/kudnn_lrn_kernels.hpp"

namespace KuDNN {
namespace Detail {

static constexpr SizeType LRN_FWD_TOTAL_SIZE_1 = 1000;

static constexpr SizeType LRN_FWD_THREADS_THRESHOLD_1 = 1;
static constexpr float LRN_FWD_THREADS_COEFF_1 = 11.7f;
static constexpr float LRN_FWD_THREADS_COEFF_2 = -79.7;

class LocalResponseNormalizationLayerFWDImpl final : public Parallelizable {
public:
    LocalResponseNormalizationLayerFWDImpl(const TensorInfo &srcInfo, const TensorInfo &dstInfo, float alpha,
        float beta, float k, SizeType localSize, LRNormAlgorithmKind algorithm) noexcept(false)
        : srcType(srcInfo.GetType()), dstType(dstInfo.GetType()),
        compType(Element::TypeT::F32), paramAlpha(alpha), paramBeta(beta), paramK(k), paramLocalSize(localSize),
        algKind(algorithm)
    {
        Service::ThrowOnStatus(LocalResponseNormalizationLayerFWD::ValidateInput(srcInfo, dstInfo, alpha, beta,
            k, localSize, algorithm), "Local Response Normalization Layer");
        dimsSrc = Service::BroadcastToNCDHW(srcInfo.GetDims());
        dimsDst = Service::BroadcastToNCDHW(dstInfo.GetDims());
        numDims = srcInfo.GetNumDims();
        stridesSrc = Service::BroadcastToNCDHW(srcInfo.GetStrides(), false);
        stridesDst  = Service::BroadcastToNCDHW(dstInfo.GetStrides(), false);
        axbFormatTag = srcInfo.GetLayout() == srcInfo.GetStandardAXBLayout();
        abxFormatTag = srcInfo.GetLayout() == srcInfo.GetStandardABXLayout();
        Parallelizable::Initialize(GetOptNumThreads(), BatchedDims());
    }

    template <typename CompType, typename InType, typename OutType>
    void Evaluate(const Shape &start, const Shape &end,
        const void *src, void *dst) const noexcept
    {
        const InType *typedSrc = static_cast<const InType *>(src);
        OutType *typedDst = static_cast<OutType *>(dst);
        if (axbFormatTag || (!abxFormatTag && algKind == LRNormAlgorithmKind::WITHIN)) {
            LRN::LocalResponseNormGenericAxbFWD<CompType>(typedSrc, typedDst, paramAlpha, paramBeta, paramK,
                paramLocalSize, start, end, stridesSrc, stridesDst, dimsSrc, numDims, algKind);
        } else {
            LRN::LocalResponseNormGenericAbxFWD<CompType>(typedSrc, typedDst, paramAlpha, paramBeta, paramK,
                paramLocalSize, start, end, stridesSrc, stridesDst, dimsSrc, numDims, algKind);
        }
    }

    void RunSequential(const Shape &batchSplitDims, const Shape &batchStrides, const void *src,
        void *dst) const noexcept(false)
    {
        Shape start = batchStrides;
        SizeType array[MAX_DIMS];
        for (SizeType i = 0; i < batchSplitDims.GetNumDims(); ++i) {
            array[i] = start[i] + batchSplitDims[i];
        }
        Shape end(array, batchSplitDims.GetNumDims());
        Service::ParseType<false, false>(compType, srcType, dstType, *this, start, end, src, dst);
    }

    void RunSequential(const Shape &batchSplitDims, const void *src,
        void *dst) const noexcept(false)
    {
        SizeType batchStrides[MAX_DIMS] = { 0, 0, 0, 0, 0 };
        RunSequential(batchSplitDims, Shape{batchStrides, batchSplitDims.GetNumDims()}, src, dst);
    }

    void Run(const void *src, void *dst) const noexcept(false)
    {
        if ((src == nullptr) || (dst == nullptr)) {
            throw Service::LogicError {"Local Response Normalization Layer: one of input matrices is nullptr"};
        }
        Threading::BatchThreader(*this, src, dst);
    }

private:
    Element::Type srcType;
    Element::Type dstType;
    Element::Type compType;
    Shape dimsSrc;
    Shape dimsDst;
    Shape stridesSrc;
    Shape stridesDst;
    float paramAlpha;
    float paramBeta;
    float paramK;
    SizeType paramLocalSize;
    SizeType numDims;
    LRNormAlgorithmKind algKind;
    bool axbFormatTag;
    bool abxFormatTag;

    Shape BatchedDims() const noexcept
    {
        std::array<SizeType, MAX_DIMS> batchedDImsArr;
        SizeType batchedAxis = MAX_DIMS;
        for (SizeType i = 0; i < batchedAxis; ++i) {
            batchedDImsArr[i] = dimsSrc[i];
        }
        return {batchedDImsArr.data(), batchedAxis};
    }

    SizeType GetOptNumThreads() const noexcept
    {
        SizeType totalSize = dimsSrc.GetTotalDimsSize();
        if (totalSize < LRN_FWD_TOTAL_SIZE_1) {
            return LRN_FWD_THREADS_THRESHOLD_1;
        } else {
            return LRN_FWD_THREADS_COEFF_1 * std::log(totalSize) + LRN_FWD_THREADS_COEFF_2;
        }
    }
};

} // Detail

LocalResponseNormalizationLayerFWD::~LocalResponseNormalizationLayerFWD() noexcept {}

LocalResponseNormalizationLayerFWD::LocalResponseNormalizationLayerFWD(const TensorInfo &srcInfo,
    const TensorInfo &dstInfo, float alpha, float beta, float k, SizeType localSize,
    LRNormAlgorithmKind algorithm) noexcept(false)
    : pImpl(new Detail::LocalResponseNormalizationLayerFWDImpl(srcInfo, dstInfo, alpha, beta, k,
        localSize, algorithm)) {}

LocalResponseNormalizationLayerFWD::LocalResponseNormalizationLayerFWD(const LocalResponseNormalizationLayerFWD &other)
    noexcept(false)
    : pImpl(new Detail::LocalResponseNormalizationLayerFWDImpl{*(other.pImpl.get())}) {}

LocalResponseNormalizationLayerFWD::LocalResponseNormalizationLayerFWD(LocalResponseNormalizationLayerFWD &&other)
    noexcept
    : pImpl(std::move(other.pImpl)) {}

LocalResponseNormalizationLayerFWD& LocalResponseNormalizationLayerFWD::operator=(
    const LocalResponseNormalizationLayerFWD &other) noexcept(false)
{
    if (this != &other) {
        this->pImpl.reset(new Detail::LocalResponseNormalizationLayerFWDImpl{*(other.pImpl.get())});
    }
    return *this;
}

LocalResponseNormalizationLayerFWD& LocalResponseNormalizationLayerFWD::operator=(LocalResponseNormalizationLayerFWD
    &&other) noexcept
{
    if (this != &other) {
        this->pImpl = std::move(other.pImpl);
    }
    return *this;
}

void LocalResponseNormalizationLayerFWD::Run(const void *src, void *dst) const noexcept(false)
{
    pImpl->Run(src, dst);
}

Status LocalResponseNormalizationLayerFWD::ValidateInput(const TensorInfo &srcInfo,
    const TensorInfo &dstInfo, float alpha, float beta, float k, SizeType localSize,
    LRNormAlgorithmKind algorithm) noexcept
{
    Service::Unused(alpha, beta, k, algorithm);
    if (srcInfo.GetType() != dstInfo.GetType()) {
        return Status::UNSUPPORTED;
    }
    if (localSize <= 0) {
        return Status::BAD_ARGUMENTS;
    }
    if ((srcInfo.GetType() != Element::TypeT::F32) &&
        (srcInfo.GetType() != Element::TypeT::F16) &&
        (srcInfo.GetType() != Element::TypeT::BF16)) {
        return Status::BAD_ARGUMENTS;
    }
    if (srcInfo.GetLayout() != dstInfo.GetLayout()) {
        return Status::UNSUPPORTED;
    }
    if (srcInfo.GetDims() != dstInfo.GetDims()) {
        return Status::BAD_ARGUMENTS;
    }
    if (srcInfo.GetTotalTensorSize() == 0) {
        return Status::UNSUPPORTED;
    }
    return Status::SUCCESS;
}

} // KuDNN
