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

#include <cmath>
#include <climits>
#include <memory>
#include <functional>

#include "kudnn.hpp"

#include "internal_funcs.hpp"
#include "threading_impl.hpp"
#include "kernels/resampling/kudnn_resampling_bwd_linear_kernels.hpp"
#include "kernels/resampling/kudnn_resampling_bwd_nearest_kernels.hpp"
#include "kernels/resampling/kudnn_resampling_normal_kernels.hpp"
#include "kernels/resampling/kudnn_resampling_kernels.hpp"

namespace KuDNN {
namespace Detail {
namespace Resampling {

const IntType LARGE_SH = static_cast<IntType>(INT_MAX);
const IntType MAX_NEAREST_THREADS = 32;
const IntType MAX_LINEAR_THREADS = 64;

static Shape ResamplingBatchedDims(const Shape &dims, SizeType ndims, bool isAxb)
{
    if (ndims == 0) {
        return {1};
    } else {
        std::array<SizeType, MAX_DIMS> batchedDimsArr;

        if (!isAxb) {
            batchedDimsArr[IDX_0] = dims[IDX_0]; // N
            batchedDimsArr[IDX_1] = dims[IDX_1]; // C
            batchedDimsArr[IDX_2] = dims[IDX_2]; // D
            batchedDimsArr[IDX_3] = dims[IDX_3]; // H
            batchedDimsArr[IDX_4] = dims[IDX_4]; // W
        } else {
            batchedDimsArr[IDX_0] = dims[IDX_0]; // N
            batchedDimsArr[IDX_1] = dims[IDX_2]; // D
            batchedDimsArr[IDX_2] = dims[IDX_3]; // H
            batchedDimsArr[IDX_3] = dims[IDX_4]; // W
            batchedDimsArr[IDX_4] = dims[IDX_1]; // C
        }
        return {batchedDimsArr.data(), MAX_DIMS};
    }
}

class ResamplingLayerImplFWD final : public Parallelizable {
public:
    ResamplingLayerImplFWD(const TensorInfo &srcInfo, const TensorInfo &dstInfo,
        ResamplingAlg alg) noexcept(false)
        : srcType(srcInfo.GetType()), dstType(dstInfo.GetType()), algKind(alg),
        ndims(srcInfo.GetNumDims()), layout(srcInfo.GetLayout())
    {
        isAxb = Service::OneOf(layout, Layout::ACB, Layout::ACDB, Layout::ACDEB);
        dimsSrc = Service::BroadcastToNCDHW(srcInfo.GetDims());
        dimsDst = Service::BroadcastToNCDHW(dstInfo.GetDims());
        stridesSrc = Service::BroadcastToNCDHW(srcInfo.GetStrides(), false);
        stridesDst = Service::BroadcastToNCDHW(dstInfo.GetStrides(), false);

        Service::ThrowOnStatus(ResamplingLayerFWD::ValidateInput(
            srcInfo, dstInfo, alg), "Resampling layer");

        Parallelizable::Initialize(GetOptNumThreads(), ResamplingBatchedDims(dimsDst, ndims, isAxb));
        doNormal = (GetDHWSize(dimsSrc) >= LARGE_SH) || (GetDHWSize(dimsDst) >= LARGE_SH) || (srcType != dstType);
        if (doNormal) {
            FillResamplingCoeffs<IntType>();
        } else {
            FillResamplingCoeffs<int32_t>();
        }
    }

    void RunSequential(const Shape &batchSplitDims, const Shape &stride, const void *src,
        void *dst) const noexcept(false)
    {
        StridesInfo sInfo(stridesSrc, stridesDst);
        // The ndims of nearest algorithm is not important, we set its default value as DIM_5.
        SizeType paraNDims = (algKind == ResamplingAlg::NEAREST_NEIGHBOR) ? DIM_5 : ndims;
        if (!doNormal) {
            VCKerParas paras{srcType, paraNDims, algKind};
            if (VC_KER_MAP.find(paras) != VC_KER_MAP.end()) {
                auto kerFunc = VC_KER_MAP.at(paras);
                kerFunc(batchSplitDims, stride, dimsDst, sInfo, src, dst, sveCoeffs, isAxb);
            }
            return;
        }
        NormalKerParas paras{srcType, dstType, paraNDims, algKind};
        if (NORMAL_KER_MAP.find(paras) != NORMAL_KER_MAP.end()) {
            auto kerFunc = NORMAL_KER_MAP.at(paras);
            kerFunc(batchSplitDims, stride, dimsDst, sInfo, src, dst, normalCoeffs, isAxb);
        }
    }

    void RunSequential(const Shape &batchSplitDims, const Shape &batchStrides,
        const Shape &unusedStride, const void *src, void *dst) const noexcept(false)
    {
        Service::Unused(unusedStride);
        RunSequential(batchSplitDims, batchStrides, src, dst);
    }

    void Run(const void *src, void *dst) noexcept(false)
    {
        if ((src == nullptr) || (dst == nullptr)) {
            throw Service::LogicError {"Resampling Layer: one of input matrices is nullptr"};
        }
        IntType dims[MAX_DIMS] = {0};
        Threading::BatchThreader(*this, Shape(dims, MAX_DIMS), src, dst);
    }

private:
    Element::Type srcType;
    Element::Type dstType;
    ResamplingAlg algKind;

    Shape dimsSrc;
    Shape dimsDst;
    Shape stridesSrc;
    Shape stridesDst;
    SizeType ndims;
    Layout layout;
    bool isAxb;

    ResamplingCoeffs<IntType> normalCoeffs;
    ResamplingCoeffs<int32_t> sveCoeffs;
    bool doNormal;

    SizeType GetOptNumThreads() const noexcept
    {
        IntType dMB = dimsDst[IDX_0];
        IntType dC = dimsDst[IDX_1];
        IntType dOD = dimsDst[IDX_2];
        IntType dOH = dimsDst[IDX_3];
        IntType dOW = dimsDst[IDX_4];
        IntType threads = dMB;
        threads = std::max(threads, dC);
        threads = std::max(threads, dOD);
        threads = std::max(threads, dOH);
        threads = std::max(threads, dOW);
        if (algKind == ResamplingAlg::NEAREST_NEIGHBOR) {
            return static_cast<SizeType>(std::min(threads, MAX_NEAREST_THREADS));
        } else {
            return static_cast<SizeType>(std::min(threads, MAX_LINEAR_THREADS));
        }
        return threads;
    }

    IntType GetDHWSize(const Shape &dims)
    {
        return dims[DIM_2] + dims[DIM_3] + dims[DIM_4];
    }

    template <typename T>
    ResamplingCoeffs<T> &GetCoeffs()
    {
        if constexpr (std::is_same_v<T, IntType>) {
            return normalCoeffs;
        } else if constexpr (std::is_same_v<T, int32_t>) {
            return sveCoeffs;
        } else {
            return normalCoeffs;
        }
    }

    template <typename T>
    void FillResamplingCoeffs()
    {
        IntType dOD = dimsDst[IDX_2];
        IntType dOH = dimsDst[IDX_3];
        IntType dOW = dimsDst[IDX_4];

        IntType dID = dimsSrc[IDX_2];
        IntType dIH = dimsSrc[IDX_3];
        IntType dIW = dimsSrc[IDX_4];

        std::vector<T> idxs(dOD + dOH + dOW);
        InitIdxs<T>(dOD, 0, &idxs[0]);
        InitIdxs<T>(dOH, 0, &idxs[0] + dOD);
        InitIdxs<T>(dOW, 0, &idxs[0] + dOD + dOH);

        ResamplingCoeffs<T> &coeffs = GetCoeffs<T>();
        if (algKind == ResamplingAlg::NEAREST_NEIGHBOR) {
            // Fill nearest data.
            coeffs.NearestInit(dOD + dOH + dOW);
            Oidx2Iidx<T>(dOD, &idxs[0], &coeffs.outidx2Inidx[0], dOD, dID);
            Oidx2Iidx<T>(dOH, &idxs[0] + dOD, &coeffs.outidx2Inidx[0] + dOD, dOH, dIH);
            Oidx2Iidx<T>(dOW, &idxs[0] + dOD + dOH, &coeffs.outidx2Inidx[0] + dOD + dOH, dOW, dIW);
        } else if (algKind == ResamplingAlg::LINEAR) {
            // Fill linear data.
            coeffs.LinearInit(dOD + dOH + dOW);
            FillCoeff<T>(dOD, 0, &idxs[0], dOD, dID, coeffs);
            FillCoeff<T>(dOH, dOD, &idxs[0], dOH, dIH, coeffs);
            FillCoeff<T>(dOW, dOD + dOH, &idxs[0], dOW, dIW, coeffs);
        }
    }
};

class ResamplingLayerImplBWD final : public Parallelizable {
public:
    ResamplingLayerImplBWD(const TensorInfo &diffDstInfo, const TensorInfo &diffSrcInfo,
        ResamplingAlg alg) noexcept(false)
        : diffDstType(diffDstInfo.GetType()), diffSrcType(diffSrcInfo.GetType()),
        algKind(alg), ndims(diffDstInfo.GetNumDims()), layout(diffDstInfo.GetLayout())
    {
        isAxb = Service::OneOf(layout, Layout::ACB, Layout::ACDB, Layout::ACDEB);
        dimsDiffDst = Service::BroadcastToNCDHW(diffDstInfo.GetDims());
        dimsDiffSrc = Service::BroadcastToNCDHW(diffSrcInfo.GetDims());
        stridesDiffDst = Service::BroadcastToNCDHW(diffDstInfo.GetStrides(), false);
        stridesDiffSrc = Service::BroadcastToNCDHW(diffSrcInfo.GetStrides(), false);
        Service::ThrowOnStatus(ResamplingLayerBWD::ValidateInput(diffDstInfo, diffSrcInfo, alg), "Resampling layer");

        if (algKind == ResamplingAlg::NEAREST_NEIGHBOR) {
            FillNearestDataVC(nearestData, dimsDiffDst, dimsDiffSrc);
        } else if (algKind == ResamplingAlg::LINEAR) {
            FillLinearDataVC(linearData, dimsDiffDst, dimsDiffSrc);
        }
        Parallelizable::Initialize(GetOptNumThreads(), ResamplingBatchedDims(dimsDiffSrc, ndims, isAxb));
    }

    void RunSequential(const Shape &batchSplitDims, const Shape &stride,
        const void *diffDst, void *diffSrc) const noexcept(false)
    {
        BwdKerParas paras{diffDstType, diffSrcType, ndims};
        if (algKind == ResamplingAlg::NEAREST_NEIGHBOR) {
            // The ndims of nearest algorithm is not important, we set its default value as DIM_5.
            paras.ndims = DIM_5;
            if (NEAREST_BWD_KER_MAP.find(paras) != NEAREST_BWD_KER_MAP.end()) {
                auto kerFunc = NEAREST_BWD_KER_MAP.at(paras);
                kerFunc(batchSplitDims, stride, diffDst, diffSrc,
                        dimsDiffSrc, stridesDiffDst, stridesDiffSrc, nearestData, isAxb);
            }
        } else if (algKind == ResamplingAlg::LINEAR) {
            if (LINEAR_BWD_KER_MAP.find(paras) != LINEAR_BWD_KER_MAP.end()) {
                auto kerFunc = LINEAR_BWD_KER_MAP.at(paras);
                kerFunc(batchSplitDims, stride, diffDst, diffSrc,
                        stridesDiffDst, stridesDiffSrc, linearData, isAxb);
            }
        }
    }

    void RunSequential(const Shape &batchSplitDims, const Shape &batchStrides, const Shape &unusedStride,
        const void *diffDst, void *diffSrc) const noexcept(false)
    {
        Service::Unused(unusedStride);
        RunSequential(batchSplitDims, batchStrides, diffDst, diffSrc);
    }

    void Run(const void *diffDst, void *diffSrc)
    {
        if ((diffDst == nullptr) || (diffSrc == nullptr)) {
            throw Service::LogicError {"Resampling Layer: one of input matrices is nullptr"};
        }
        IntType dims[MAX_DIMS] = {0};
        Threading::BatchThreader(*this, Shape(dims, MAX_DIMS), diffDst, diffSrc);
    }
private:
    Element::Type diffDstType;
    Element::Type diffSrcType;
    ResamplingAlg algKind;

    Shape dimsDiffDst;
    Shape dimsDiffSrc;
    Shape stridesDiffDst;
    Shape stridesDiffSrc;

    SizeType ndims;
    Layout layout;
    bool isAxb;
    NearestData nearestData;
    LinearData linearData;

    SizeType GetOptNumThreads() noexcept
    {
        IntType dMB = dimsDiffSrc[IDX_0];
        IntType dC = dimsDiffSrc[IDX_1];
        IntType dID = dimsDiffSrc[IDX_2];
        IntType dIH = dimsDiffSrc[IDX_3];
        IntType dIW = dimsDiffSrc[IDX_4];
        IntType threads = dMB;
        threads = std::max(threads, dC);
        threads = std::max(threads, dID);
        threads = std::max(threads, dIH);
        threads = std::max(threads, dIW);
        return static_cast<SizeType>(threads);
    }
};

} // namespace Resampling
} // namespace Detail

// Functions of class ResamplingLayerFWD.
ResamplingLayerFWD::~ResamplingLayerFWD() noexcept {}

ResamplingLayerFWD::ResamplingLayerFWD(const TensorInfo &srcInfo, const TensorInfo &dstInfo,
    ResamplingAlg algKind) noexcept(false)
    : pImpl(new Detail::Resampling::ResamplingLayerImplFWD(srcInfo, dstInfo, algKind)) {}

ResamplingLayerFWD::ResamplingLayerFWD(const ResamplingLayerFWD &other) noexcept(false)
    : pImpl(new Detail::Resampling::ResamplingLayerImplFWD{*(other.pImpl.get())}) {}

ResamplingLayerFWD::ResamplingLayerFWD(ResamplingLayerFWD &&other) noexcept
    : pImpl(std::move(other.pImpl)) {}

ResamplingLayerFWD& ResamplingLayerFWD::operator=(const ResamplingLayerFWD &other) noexcept(false)
{
    if (this != &other) {
        this->pImpl.reset(new Detail::Resampling::ResamplingLayerImplFWD{*(other.pImpl.get())});
    }
    return *this;
}

ResamplingLayerFWD& ResamplingLayerFWD::operator=(ResamplingLayerFWD &&other) noexcept
{
    if (this != &other) {
        this->pImpl = std::move(other.pImpl);
    }
    return *this;
}

void ResamplingLayerFWD::Run(const void *src, void *dst) const noexcept(false)
{
    pImpl->Run(src, dst);
}

Status ResamplingLayerFWD::ValidateInput(const TensorInfo &srcInfo, const TensorInfo &dstInfo,
    ResamplingAlg algKind) noexcept
{
    bool isUnsupported = (!Service::OneOf(srcInfo.GetLayout(),
        Layout::ABC, Layout::ABCD, Layout::ABCDE, Layout::ACB, Layout::ACDB, Layout::ACDEB)) ||
        (srcInfo.GetType() == Element::TypeT::UNDEFINED ||
        dstInfo.GetType() == Element::TypeT::UNDEFINED) ||
        (algKind == ResamplingAlg::UNIMPLEMENTED) ||
        (!srcInfo.IsDense()) || (srcInfo.GetTotalTensorSize() == 0);
    if (isUnsupported) {
        return Status::UNSUPPORTED;
    }
    return Status::SUCCESS;
}

// Functions of class ResamplingLayerBWD.
ResamplingLayerBWD::ResamplingLayerBWD(const TensorInfo &diffDstInfo, const TensorInfo &diffSrcInfo,
                                       ResamplingAlg algKind) noexcept(false)
    : pImpl(new Detail::Resampling::ResamplingLayerImplBWD(diffDstInfo, diffSrcInfo, algKind)) {}

ResamplingLayerBWD::ResamplingLayerBWD(const ResamplingLayerBWD &other) noexcept(false)
    : pImpl(new Detail::Resampling::ResamplingLayerImplBWD(*(other.pImpl.get()))) {}

ResamplingLayerBWD::ResamplingLayerBWD(ResamplingLayerBWD &&other) noexcept
    : pImpl(std::move(other.pImpl)) {}

ResamplingLayerBWD& ResamplingLayerBWD::operator=(const ResamplingLayerBWD &other) noexcept(false)
{
    if (this != &other) {
        this->pImpl.reset(new Detail::Resampling::ResamplingLayerImplBWD(*(other.pImpl.get())));
    }
    return *this;
}

ResamplingLayerBWD& ResamplingLayerBWD::operator=(ResamplingLayerBWD &&other) noexcept
{
    if (this != &other) {
        this->pImpl = std::move(other.pImpl);
    }
    return *this;
}

ResamplingLayerBWD::~ResamplingLayerBWD() noexcept {}

Status ResamplingLayerBWD::ValidateInput(const TensorInfo &diffDstInfo, const TensorInfo &diffSrcInfo,
    ResamplingAlg algKind) noexcept
{
    if (diffDstInfo.GetType() == Element::TypeT::UNDEFINED ||
        diffSrcInfo.GetType() == Element::TypeT::UNDEFINED) {
        return Status::UNSUPPORTED;
    }
    if ((!Service::OneOf(diffDstInfo.GetLayout(), Layout::ABC, Layout::ABCD, Layout::ABCDE)) &&
        (!Service::OneOf(diffDstInfo.GetLayout(), Layout::ACB, Layout::ACDB, Layout::ACDEB))) {
        return Status::UNSUPPORTED;
    }
    if (algKind == ResamplingAlg::UNIMPLEMENTED) {
        return Status::UNSUPPORTED;
    }
    if (!diffDstInfo.IsDense()) {
        return Status::UNSUPPORTED;
    }
    if (diffDstInfo.GetTotalTensorSize() == 0) {
        return Status::UNSUPPORTED;
    }
    return Status::SUCCESS;
}

void ResamplingLayerBWD::Run(const void *diffDst, void *diffSrc) const noexcept(false)
{
    pImpl->Run(diffDst, diffSrc);
}

} // namespace KuDNN
