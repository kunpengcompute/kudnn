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

#include <arm_sve.h>

#include "kudnn_jit.hpp"
#include "threading_impl.hpp"
#include "vector_wrappers.hpp"
#include "kernels/reorder/kudnn_reorder_helpers.hpp"

namespace KuDNN {
namespace Detail {

class ReorderLayerImpl final : public Parallelizable {
public:
    ReorderLayerImpl(const TensorInfo &srcInfo, const TensorInfo &dstInfo) noexcept(false)
        : impl()
    {
        Service::ThrowOnStatus(ReorderLayer::ValidateInput(srcInfo, dstInfo), "Reorder");
        impl = FindSolution(ReorderInfo{srcInfo, dstInfo}, Threading::GetMaxNumThreads());
    }

    void Run(const void *src, void *dst, float alpha) noexcept(false)
    {
        if ((src == nullptr) || (dst == nullptr)) {
            throw Service::LogicError {"Reorder Layer: one of input matrices is nullptr"};
        }
        if (impl.expired()) {
            throw Service::LogicError {"Cached Reorder Impl was manully removed"};
        } else {
            std::shared_ptr<ImplBase<ReorderInfo>> reorderImpl(impl);
            if ((src == dst)
                && (reorderImpl->GetTask().srcInfo.GetStrides() != reorderImpl->GetTask().dstInfo.GetStrides())
                && (reorderImpl->GetTask().srcInfo.GetType().GetSize() <
                reorderImpl->GetTask().dstInfo.GetType().GetSize())) {
                throw Service::LogicError {"Reorder Layer: src == dst and srcInfo precision > dstInfo precision"};
            }
            if (reorderImpl->GetNThreads() != Threading::GetMaxNumThreads()) {
                impl = FindSolution(reorderImpl->GetTask(), Threading::GetMaxNumThreads());
                reorderImpl = impl.lock();
            }
            if (!((src == dst)
                && (reorderImpl->GetTask().srcInfo.GetStrides() == reorderImpl->GetTask().dstInfo.GetStrides())
                && (alpha == 1))) {
                reorderImpl->Run({src, dst, alpha});
            }
        }
    }

private:
    mutable std::weak_ptr<ImplBase<ReorderInfo>> impl;
};
} // Detail
ReorderLayer::~ReorderLayer() noexcept {}

ReorderLayer::ReorderLayer(const TensorInfo &srcInfo, const TensorInfo &dstInfo) noexcept(false)
    : pImpl(new Detail::ReorderLayerImpl(srcInfo, dstInfo)) {}

ReorderLayer::ReorderLayer(const ReorderLayer &other) noexcept(false)
    : pImpl(new Detail::ReorderLayerImpl{*(other.pImpl.get())}) {}

ReorderLayer::ReorderLayer(ReorderLayer &&other) noexcept : pImpl(std::move(other.pImpl)) {}

ReorderLayer& ReorderLayer::operator=(const ReorderLayer &other) noexcept(false)
{
    if (this != &other) {
        this->pImpl.reset(new Detail::ReorderLayerImpl{*(other.pImpl.get())});
    }
    return *this;
}

ReorderLayer& ReorderLayer::operator=(ReorderLayer &&other) noexcept
{
    if (this != &other) {
        this->pImpl = std::move(other.pImpl);
    }
    return *this;
}

void ReorderLayer::Run(const void *src, void *dst) const noexcept(false)
{
    pImpl->Run(src, dst, 1.0f);
}

void ReorderLayer::Run(const void *src, void *dst, float alpha) const noexcept(false)
{
    pImpl->Run(src, dst, alpha);
}

Status ReorderLayer::ValidateInput(const TensorInfo &srcInfo, const TensorInfo &dstInfo) noexcept
{
    if (srcInfo.HasStrideOverFastestDim() || dstInfo.HasStrideOverFastestDim()) {
        return Status::UNSUPPORTED;
    }
    if (srcInfo.GetDims() != dstInfo.GetDims()) {
        return Status::BAD_ARGUMENTS;
    }
    return Status::SUCCESS;
}

} // KuDNN
