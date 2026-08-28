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
#include "kernels/concat/kudnn_concat_kernels.hpp"

namespace KuDNN {
namespace Detail {

constexpr SizeType CONCAT_MAP_LENGTH = 32;

class ConcatLayerImpl final {
public:
    ConcatLayerImpl(const std::vector<TensorInfo> &srcInfo, const int concatDim,
        const TensorInfo &dstInfo) noexcept(false)
        : srcType(srcInfo[0].GetType()), dstType(dstInfo.GetType()), compType(Element::TypeT::F32),
        n(srcInfo.size()), needToConvert(false)
    {
        Service::ThrowOnStatus(ConcatLayer::ValidateInput(srcInfo, concatDim, dstInfo), "Concat Layer");
        for (SizeType k = 0; k < srcInfo.size(); k++) {
            Shape concatSrcDim = Service::GetShapeAccordingToLayout(srcInfo[k].GetDims(), dstInfo.GetLayout());
            concatSrcDims.push_back(concatSrcDim);
            reorderedSrcDims.push_back(Service::BroadcastTo5D(concatSrcDim));
        }
        concatDstDim = Service::GetShapeAccordingToLayout(dstInfo.GetDims(), dstInfo.GetLayout());
        reorderedDstDim = Service::BroadcastTo5D(concatDstDim);
        dim = CONCAT_DIM_MAP[static_cast<SizeType>(dstInfo.GetLayout())][concatDim];
        if (srcType != dstType) {
            needToConvert = true;
        }
    }

    template <typename CompType, typename InType, typename OutType>
    void Evaluate(const void **src, void *dst) const noexcept
    {
        OutType *typedDst = static_cast<OutType *>(dst);
        Concat::EvaluateConcat<CompType, InType, OutType>(concatSrcDims, concatDstDim, src, typedDst, dim, n);
    }

    void Run(const void **src, void *dst) const noexcept(false)
    {
        if ((src == nullptr) || (dst == nullptr)) {
            return;
        }
        for (SizeType i = 0; i < n; ++i) {
            if (src[i] == nullptr) {
                return;
            }
        }
        if (needToConvert) {
            std::vector<const void *> convertedSrc(n);
            void *convertedDst = dst;
            for (SizeType i = 0; i < n; i++) {
                void *tmp = Service::AlignedAlloc(reorderedSrcDims[i].GetTotalDimsSize() * sizeof(float));
                ReorderLayer r({reorderedSrcDims[i], srcType, Layout::ABCDE},
                    {reorderedSrcDims[i], Element::TypeT::F32, Layout::ABCDE});
                r.Run(src[i], tmp);
                convertedSrc[i] = tmp;
            }
            convertedDst = Service::AlignedAlloc(reorderedDstDim.GetTotalDimsSize() * sizeof(float));
            Service::ParseType<true, false>(compType, Element::TypeT::F32, Element::TypeT::F32, *this,
                convertedSrc.data(), convertedDst);
            for (SizeType i = 0; i < n; i++) {
                Service::Deallocate(convertedSrc[i]);
            }
            ReorderLayer r({reorderedDstDim, Element::TypeT::F32, Layout::ABCDE},
                {reorderedDstDim, dstType, Layout::ABCDE});
            r.Run(convertedDst, dst);
            Service::Deallocate(convertedDst);
        } else {
            Service::ParseType<true, false>(compType, srcType, dstType, *this, src, dst); // evaluate
        }
    }

private:
    Element::Type srcType;
    Element::Type dstType;
    Element::Type compType;
    SizeType n;
    SizeType dim;
    std::vector<Shape> concatSrcDims;
    std::vector<Shape> reorderedSrcDims;
    Shape concatDstDim;
    Shape reorderedDstDim;
    bool needToConvert;

    static constexpr SizeType CONCAT_DIM_MAP[CONCAT_MAP_LENGTH][MAX_DIMS] = {
        {IDX_0},
        {IDX_0},
        {IDX_0, IDX_1},
        {IDX_1, IDX_0},
        {IDX_0, IDX_1, IDX_2},
        {IDX_0, IDX_2, IDX_1},
        {IDX_1, IDX_0, IDX_2},
        {IDX_2, IDX_0, IDX_1},
        {IDX_1, IDX_2, IDX_0},
        {IDX_2, IDX_1, IDX_0},
        {IDX_0, IDX_1, IDX_2, IDX_3},
        {IDX_0, IDX_1, IDX_3, IDX_2},
        {IDX_0, IDX_2, IDX_1, IDX_3},
        {IDX_0, IDX_3, IDX_1, IDX_2},
        {IDX_0, IDX_2, IDX_3, IDX_1},
        {IDX_0, IDX_3, IDX_2, IDX_1},
        {IDX_1, IDX_0, IDX_2, IDX_3},
        {IDX_3, IDX_0, IDX_1, IDX_2},
        {IDX_2, IDX_3, IDX_0, IDX_1},
        {IDX_3, IDX_2, IDX_0, IDX_1},
        {IDX_2, IDX_3, IDX_1, IDX_0},
        {IDX_0, IDX_1, IDX_2, IDX_3, IDX_4},
        {IDX_0, IDX_1, IDX_2, IDX_4, IDX_3},
        {IDX_0, IDX_1, IDX_4, IDX_2, IDX_3},
        {IDX_0, IDX_2, IDX_1, IDX_3, IDX_4},
        {IDX_0, IDX_4, IDX_1, IDX_2, IDX_3},
        {IDX_0, IDX_4, IDX_3, IDX_1, IDX_2},
        {IDX_1, IDX_0, IDX_2, IDX_3, IDX_4},
        {IDX_4, IDX_0, IDX_1, IDX_2, IDX_3},
        {IDX_3, IDX_4, IDX_0, IDX_1, IDX_2},
        {IDX_4, IDX_3, IDX_0, IDX_1, IDX_2},
        {IDX_3, IDX_4, IDX_2, IDX_0, IDX_1},
    };
};

} // Detail

ConcatLayer::~ConcatLayer() noexcept {}

ConcatLayer::ConcatLayer(const std::vector<TensorInfo> &srcInfo, const int concatDim,
    const TensorInfo &dstInfo) noexcept(false)
    : pImpl(new Detail::ConcatLayerImpl(srcInfo, concatDim, dstInfo)) {}

ConcatLayer::ConcatLayer(const ConcatLayer &other) noexcept(false)
    : pImpl(new Detail::ConcatLayerImpl{*(other.pImpl.get())}) {}

ConcatLayer::ConcatLayer(ConcatLayer &&other) noexcept
    : pImpl(std::move(other.pImpl)) {}

ConcatLayer& ConcatLayer::operator=(const ConcatLayer &other) noexcept(false)
{
    if (this != &other) {
        this->pImpl.reset(new Detail::ConcatLayerImpl{*(other.pImpl.get())});
    }
    return *this;
}

ConcatLayer& ConcatLayer::operator=(ConcatLayer &&other) noexcept
{
    if (this != &other) {
        this->pImpl = std::move(other.pImpl);
    }
    return *this;
}

void ConcatLayer::Run(const void **src, void *dst) const noexcept(false)
{
    pImpl->Run(src, dst);
}

Status ConcatLayer::ValidateInput(const std::vector<TensorInfo> &srcInfo, const int concatDim,
    const TensorInfo &dstInfo) noexcept
{
    if (static_cast<SizeType>(concatDim) >= dstInfo.GetNumDims()) {
        return Status::UNSUPPORTED;
    }

    if (!dstInfo.IsDense()) {
        return Status::UNSUPPORTED;
    }

    for (auto &&src : srcInfo) {
        if (src.GetNumDims() != dstInfo.GetNumDims()) {
            return Status::UNSUPPORTED;
        }
        if (src.GetLayout() != dstInfo.GetLayout()) {
            Shape srcShape = src.GetDims();
            SizeType i = 0;
            for (; i < srcShape.GetNumDims(); i++) {
                if (srcShape[i] == DIM_1) {
                    break;
                }
            }
            if (i == srcShape.GetNumDims()) {
                return Status::UNSUPPORTED;
            }
        }
    }

    return Status::SUCCESS;
}

} // KuDNN
