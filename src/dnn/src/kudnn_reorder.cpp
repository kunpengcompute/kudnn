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

#include "threading_impl.hpp"
#include "vector_wrappers.hpp"
#include "kernels/reorder/kudnn_reorder_kernels.hpp"

namespace KuDNN {
namespace Detail {

#ifdef KUDNNL_920Pro
static constexpr SizeType REORDER_THR_START = 16384;
static constexpr SizeType REORDER_THR_START_VAL = 1;
static constexpr SizeType REORDER_THR_STOP = 16384;
static constexpr SizeType REORDER_THR_STOP_VAL = 38;
static constexpr double REORDER_THR_FUNC = 3;
#else
static constexpr SizeType REORDER_THR_START = 50000;
static constexpr SizeType REORDER_THR_START_VAL = 1;
static constexpr SizeType REORDER_THR_STOP = 50000;
static constexpr SizeType REORDER_THR_STOP_VAL = 25;
// 0.00005 * numElements + 0.5
static constexpr double REORDER_THR_FUNC = 3;
#endif

class ReorderLayerImpl final : public Parallelizable {
public:
    ReorderLayerImpl(const TensorInfo &srcInfo, const TensorInfo &dstInfo) noexcept(false)
        : srcType(srcInfo.GetType()), dstType(dstInfo.GetType())
    {
        Service::ThrowOnStatus(ReorderLayer::ValidateInput(srcInfo, dstInfo), "Reorder Layer");

        Shape dims = Service::GetShapeAccordingToLayout(dstInfo.GetDims(), srcInfo.GetLayout());
        Shape srcStrides = Service::GetShapeAccordingToLayout(srcInfo.GetStrides(), srcInfo.GetLayout());
        Shape dstStrides = Service::GetShapeAccordingToLayout(dstInfo.GetStrides(), srcInfo.GetLayout());

        reorderedDims = Service::BroadcastTo5D(dims);
        reorderedStridesSrc = Service::BroadcastTo5D(srcStrides, false);
        reorderedStridesDst = Service::BroadcastTo5D(dstStrides, false);

        IsFastReorder(srcInfo, dstInfo);

        if (fastReorder) {
            std::array<SizeType, MAX_DIMS> newReorderedDims{1};
            if (srcInfo.GetLayout() == Layout::AB) {
                newReorderedDims[IDX_0] = 1;
                newReorderedDims[IDX_1] = srcInfo.GetDims()[IDX_0];
                newReorderedDims[IDX_2] = srcInfo.GetDims()[IDX_1];
            } else if (srcInfo.GetLayout() == Layout::BA) {
                newReorderedDims[IDX_0] = 1;
                newReorderedDims[IDX_1] = srcInfo.GetDims()[IDX_1];
                newReorderedDims[IDX_2] = srcInfo.GetDims()[IDX_0];
            } else if (srcInfo.GetStandardABXLayout() == srcInfo.GetLayout()) {
                newReorderedDims[IDX_0] = srcInfo.GetDims()[IDX_0];
                newReorderedDims[IDX_1] = srcInfo.GetDims()[IDX_1];
                newReorderedDims[IDX_2] = Service::ArrayProduct(srcInfo.GetDims(), srcInfo.GetNumDims(), IDX_2);
            } else {
                newReorderedDims[IDX_0] = srcInfo.GetDims()[IDX_0];
                newReorderedDims[IDX_1] = Service::ArrayProduct(srcInfo.GetDims(), srcInfo.GetNumDims(), IDX_2);
                newReorderedDims[IDX_2] = srcInfo.GetDims()[IDX_1];
            }
            reorderedDims = Service::BroadcastTo5D({newReorderedDims.data(), DIM_3});
            strideIn = newReorderedDims[IDX_2];
            strideOut = newReorderedDims[IDX_1];
        }

        Parallelizable::Initialize(GetOptNumThreads(), reorderedDims);
    }

    void IsFastReorder(const TensorInfo &srcInfo, const TensorInfo &dstInfo)
    {
#ifdef KUDNNL_920Pro
        bool isFastType = (srcInfo.GetType() == dstInfo.GetType() && srcInfo.GetType() != Element::TypeT::F16 &&
                           srcInfo.GetType() != Element::TypeT::BF16);
#else
        bool isFastType = (srcInfo.GetType() == Element::TypeT::F32 && dstInfo.GetType() == Element::TypeT::F32);
#endif
        bool isAXBToABX = (srcInfo.GetStandardAXBLayout() == srcInfo.GetLayout()) &&
                          (dstInfo.GetStandardABXLayout() == dstInfo.GetLayout()) && (srcInfo.GetNumDims() > DIM_2) &&
                          srcInfo.IsDense() && dstInfo.IsDense();
        bool isABXToAXB = (srcInfo.GetStandardABXLayout() == srcInfo.GetLayout()) &&
                          (dstInfo.GetStandardAXBLayout() == dstInfo.GetLayout()) && (srcInfo.GetNumDims() > DIM_2) &&
                          srcInfo.IsDense() && dstInfo.IsDense();
        bool isABToBA = (srcInfo.GetLayout() == Layout::AB) && (dstInfo.GetLayout() == Layout::BA) &&
                        srcInfo.IsDense() && dstInfo.IsDense();
        bool isBAToAB = (srcInfo.GetLayout() == Layout::BA) && (dstInfo.GetLayout() == Layout::AB) &&
                        srcInfo.IsDense() && dstInfo.IsDense();
        if (isFastType && (isAXBToABX || isABXToAXB || isABToBA || isBAToAB)) {
            fastReorder = true;
        } else {
            fastReorder = false;
            strideIn = 0;
            strideOut = 0;
        }
    }

    template <typename InType, typename OutType>
    void Evaluate(const Shape &start, const Shape &end, const void *src, void *dst) const noexcept
    {
        const InType *typedSrc = static_cast<const InType *>(src);
        OutType *typedDst = static_cast<OutType *>(dst);
        Reorder::EvaluateReorder(start, end, typedSrc, typedDst, reorderedStridesSrc, reorderedStridesDst, fastReorder,
                                 strideIn, strideOut);
    }

    void RunSequential(const Shape &batchSplitDims, const Shape &batchStrides, const void *src, void *dst) const
        noexcept(false)
    {
        Shape start = batchStrides;
        SizeType array[MAX_DIMS];
        for (SizeType i = 0; i < batchSplitDims.GetNumDims(); ++i) {
            array[i] = start[i] + batchSplitDims[i];
        }
        Shape end(array, batchSplitDims.GetNumDims());

        switch (srcType) {
            case Element::TypeT::F32: {
                if (dstType == Element::TypeT::F32) {
                    Evaluate<float, float>(start, end, src, dst);
                } else if (dstType == Element::TypeT::F16) {
                    Evaluate<float, __fp16>(start, end, src, dst);
                } else if (dstType == Element::TypeT::BF16) {
                    Evaluate<float, __bf16>(start, end, src, dst);
                } else if (dstType == Element::TypeT::S8) {
                    Evaluate<float, std::int8_t>(start, end, src, dst);
                } else if (dstType == Element::TypeT::U8) {
                    Evaluate<float, std::uint8_t>(start, end, src, dst);
                } else if (dstType == Element::TypeT::S32) {
                    Evaluate<float, std::int32_t>(start, end, src, dst);
                }
                break;
            }
            case Element::TypeT::F16: {
                if (dstType == Element::TypeT::F32) {
                    Evaluate<__fp16, float>(start, end, src, dst);
                } else if (dstType == Element::TypeT::F16) {
                    Evaluate<__fp16, __fp16>(start, end, src, dst);
                } else if (dstType == Element::TypeT::BF16) {
                    Evaluate<__fp16, __bf16>(start, end, src, dst);
                } else if (dstType == Element::TypeT::S8) {
                    Evaluate<__fp16, std::int8_t>(start, end, src, dst);
                } else if (dstType == Element::TypeT::U8) {
                    Evaluate<__fp16, std::uint8_t>(start, end, src, dst);
                } else if (dstType == Element::TypeT::S32) {
                    Evaluate<__fp16, std::int32_t>(start, end, src, dst);
                }
                break;
            }
            case Element::TypeT::BF16: {
                if (dstType == Element::TypeT::F32) {
                    Evaluate<__bf16, float>(start, end, src, dst);
                } else if (dstType == Element::TypeT::F16) {
                    Evaluate<__bf16, __fp16>(start, end, src, dst);
                } else if (dstType == Element::TypeT::BF16) {
                    Evaluate<__bf16, __bf16>(start, end, src, dst);
                } else if (dstType == Element::TypeT::S8) {
                    Evaluate<__bf16, std::int8_t>(start, end, src, dst);
                } else if (dstType == Element::TypeT::U8) {
                    Evaluate<__bf16, std::uint8_t>(start, end, src, dst);
                } else if (dstType == Element::TypeT::S32) {
                    Evaluate<__bf16, std::int32_t>(start, end, src, dst);
                }
                break;
            }
            case Element::TypeT::S8: {
                if (dstType == Element::TypeT::F32) {
                    Evaluate<std::int8_t, float>(start, end, src, dst);
                } else if (dstType == Element::TypeT::F16) {
                    Evaluate<std::int8_t, __fp16>(start, end, src, dst);
                } else if (dstType == Element::TypeT::BF16) {
                    Evaluate<std::int8_t, __bf16>(start, end, src, dst);
                } else if (dstType == Element::TypeT::S8) {
                    Evaluate<std::int8_t, std::int8_t>(start, end, src, dst);
                } else if (dstType == Element::TypeT::U8) {
                    Evaluate<std::int8_t, std::uint8_t>(start, end, src, dst);
                } else if (dstType == Element::TypeT::S32) {
                    Evaluate<std::int8_t, std::int32_t>(start, end, src, dst);
                }
                break;
            }
            case Element::TypeT::U8: {
                if (dstType == Element::TypeT::F32) {
                    Evaluate<std::uint8_t, float>(start, end, src, dst);
                } else if (dstType == Element::TypeT::F16) {
                    Evaluate<std::uint8_t, __fp16>(start, end, src, dst);
                } else if (dstType == Element::TypeT::BF16) {
                    Evaluate<std::uint8_t, __bf16>(start, end, src, dst);
                } else if (dstType == Element::TypeT::S8) {
                    Evaluate<std::uint8_t, std::int8_t>(start, end, src, dst);
                } else if (dstType == Element::TypeT::U8) {
                    Evaluate<std::uint8_t, std::uint8_t>(start, end, src, dst);
                } else if (dstType == Element::TypeT::S32) {
                    Evaluate<std::uint8_t, std::int32_t>(start, end, src, dst);
                }
                break;
            }
            case Element::TypeT::S32: {
                if (dstType == Element::TypeT::F32) {
                    Evaluate<std::int32_t, float>(start, end, src, dst);
                } else if (dstType == Element::TypeT::F16) {
                    Evaluate<std::int32_t, __fp16>(start, end, src, dst);
                } else if (dstType == Element::TypeT::BF16) {
                    Evaluate<std::int32_t, __bf16>(start, end, src, dst);
                } else if (dstType == Element::TypeT::S8) {
                    Evaluate<std::int32_t, std::int8_t>(start, end, src, dst);
                } else if (dstType == Element::TypeT::U8) {
                    Evaluate<std::int32_t, std::uint8_t>(start, end, src, dst);
                } else if (dstType == Element::TypeT::S32) {
                    Evaluate<std::int32_t, std::int32_t>(start, end, src, dst);
                }
                break;
            }
            default: {
                break;
            }
        }
    }

    void RunSequential(const Shape &batchSplitDims, const void *src, void *dst) const noexcept(false)
    {
        SizeType batchStrides[MAX_DIMS] = {0, 0, 0, 0, 0};
        RunSequential(batchSplitDims, Shape{batchStrides, batchSplitDims.GetNumDims()}, src, dst);
    }

    void Run(const void *src, void *dst) noexcept(false)
    {
        if ((src == nullptr) || (dst == nullptr)) {
            throw Service::LogicError{"Reorder Layer: one of input matrices is nullptr"};
        }
        if (src == dst) {
            throw Service::LogicError{"Reorder Layer: src == dst"};
        }
        Threading::BatchThreader(*this, src, dst);
    }

private:
    Element::Type srcType;
    Element::Type dstType;
    Shape reorderedDims;
    Shape reorderedStridesSrc;
    Shape reorderedStridesDst;
    bool fastReorder;
    SizeType strideIn;
    SizeType strideOut;

    SizeType GetOptNumThreads() const noexcept
    {
        SizeType numElements = reorderedDims.GetTotalDimsSize();
        if (numElements < REORDER_THR_START) {
            /* num of elements in tensor is to small for do threading */
            return REORDER_THR_START_VAL;
        }
        if (numElements > REORDER_THR_STOP) {
            /* perf is growing less than 3% from new thread addition */
            return REORDER_THR_STOP_VAL;
        }
        return REORDER_THR_FUNC;
    }
};

} // namespace Detail

ReorderLayer::~ReorderLayer() noexcept {}

ReorderLayer::ReorderLayer(const TensorInfo &srcInfo, const TensorInfo &dstInfo) noexcept(false)
    : pImpl(new Detail::ReorderLayerImpl(srcInfo, dstInfo))
{
}

ReorderLayer::ReorderLayer(const ReorderLayer &other) noexcept(false)
    : pImpl(new Detail::ReorderLayerImpl{*(other.pImpl.get())})
{
}

ReorderLayer::ReorderLayer(ReorderLayer &&other) noexcept : pImpl(std::move(other.pImpl)) {}

ReorderLayer &ReorderLayer::operator=(const ReorderLayer &other) noexcept(false)
{
    if (this != &other) {
        this->pImpl.reset(new Detail::ReorderLayerImpl{*(other.pImpl.get())});
    }
    return *this;
}

ReorderLayer &ReorderLayer::operator=(ReorderLayer &&other) noexcept
{
    if (this != &other) {
        this->pImpl = std::move(other.pImpl);
    }
    return *this;
}

void ReorderLayer::Run(const void *src, void *dst) const noexcept(false)
{
    pImpl->Run(src, dst);
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

} // namespace KuDNN
