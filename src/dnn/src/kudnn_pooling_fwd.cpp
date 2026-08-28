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
#include <vector>
#include <memory>
#include <cstdint>

#include "kudnn.hpp"

#include "internal_funcs.hpp"
#include "threading_impl.hpp"
#include "kernels/pooling/kudnn_pooling_kernels.hpp"

namespace KuDNN {
namespace Detail {

static constexpr SizeType POOL_ELE_NUM_FWD = 220000;

#ifdef KUDNNL_920F
static constexpr SizeType VEC_LENGTH = 512;
static constexpr SizeType POOL_ELE_SMALL_NUM_FWD = 16385;
#else
static constexpr SizeType VEC_LENGTH = 256;
#endif

class PoolingLayerFWDImpl final : public Parallelizable {
public:
    PoolingLayerFWDImpl(const Propagation &propKind, const PoolingFunction &poolingAlg,
        const TensorInfo &srcInfo, const TensorInfo &dstInfo, const Shape &stridesPara,
        const Shape &kernelPara, const Shape &dilationPara, const Shape &paddingLPara,
        const Shape &paddingRPara) noexcept(false)
        :  srcType(srcInfo.GetType()), dstType(dstInfo.GetType()), wsType(Element::TypeT::S32), padType(dstType),
        compType(Element::TypeT::F32), prop(propKind), alg(poolingAlg), strides(stridesPara), kernel(kernelPara),
        dilation(dilationPara), paddingL(paddingLPara), paddingR(paddingRPara), ndims(srcInfo.GetNumDims()),
        useWs(alg == PoolingFunction::MAX && prop == Propagation::FORWARD_TRAINING)
    {
        Service::ThrowOnStatus(PoolingLayerFWD::ValidateInput(propKind, poolingAlg, srcInfo,
            dstInfo, strides, kernel, dilation, paddingL, paddingR), "Pooling Layer");

        if ((alg == PoolingFunction::MAX) && (prop == Propagation::FORWARD_TRAINING)) {
            wsType = Service::ArrayProduct(kernel, ndims - DIM_2, 0) <= Pooling::Helpers::U8_MAX ?
                Element::TypeT::U8 : Element::TypeT::S32;
        }

        initialDims = Service::BroadcastTo5D(srcInfo.GetDims());
        reorderedDims = Service::BroadcastTo5D(dstInfo.GetDims());
        reorderedStridesSrc = Service::BroadcastTo5D(srcInfo.GetStrides(), false);
        reorderedStridesDst  = Service::BroadcastTo5D(dstInfo.GetStrides(), false);
        axbFormatTag = srcInfo.GetLayout() == srcInfo.GetStandardAXBLayout();

        Shape srcShape = srcInfo.GetDims();
        Shape dstShape = dstInfo.GetDims();
        bool changedDataType = srcType != dstType;
        SizeType f32DataTypeSize = Element::Type{Element::TypeT::F32}.GetBitwidth();
        SizeType srcDataTypeSize = srcInfo.GetType().GetBitwidth();
        SizeType dstDataTypeSize = dstInfo.GetType().GetBitwidth();
        /* src */
        reorders.emplace_back(ReorderLayer(
            TensorInfo{Shape{(VEC_LENGTH / (changedDataType ? f32DataTypeSize : srcDataTypeSize)),
                Service::ArrayProduct(srcShape, srcShape.GetNumDims(), IDX_2)},
                changedDataType ? Element::Type{Element::TypeT::F32} : srcInfo.GetType(), Layout::AB},
            TensorInfo{Shape{(VEC_LENGTH / (changedDataType ? f32DataTypeSize : srcDataTypeSize)),
                Service::ArrayProduct(srcShape, srcShape.GetNumDims(), IDX_2)},
                changedDataType ? Element::Type{Element::TypeT::F32} : srcInfo.GetType(), Layout::BA}));
        /* dst */
        reorders.emplace_back(ReorderLayer(
            TensorInfo{Shape{Service::ArrayProduct(dstShape, dstShape.GetNumDims(), IDX_2),
                (VEC_LENGTH / (changedDataType ? f32DataTypeSize : dstDataTypeSize))},
                changedDataType ? Element::Type{Element::TypeT::F32} : dstInfo.GetType(), Layout::AB},
            TensorInfo{Shape{Service::ArrayProduct(dstShape, dstShape.GetNumDims(), IDX_2),
                (VEC_LENGTH / (changedDataType ? f32DataTypeSize : dstDataTypeSize))},
                changedDataType ? Element::Type{Element::TypeT::F32} : dstInfo.GetType(), Layout::BA}));
        /* workspace */
        reorders.emplace_back(ReorderLayer(
            TensorInfo{Shape{Service::ArrayProduct(dstShape, dstShape.GetNumDims(), IDX_2),
                (VEC_LENGTH / (changedDataType ? f32DataTypeSize : dstDataTypeSize))},
                wsType, Layout::AB},
            TensorInfo{Shape{Service::ArrayProduct(dstShape, dstShape.GetNumDims(), IDX_2),
                (VEC_LENGTH / (changedDataType ? f32DataTypeSize : dstDataTypeSize))},
                wsType, Layout::BA}));

        if ((!axbFormatTag)) {
            std::array<SizeType, MAX_DIMS> parallelDims = { dstInfo.GetDims()[0], dstInfo.GetDims()[1], 1, 1, 1 };
            Parallelizable::Initialize(GetOptNumThreads(),
                Service::BroadcastTo5D({ parallelDims.data(), dstInfo.GetNumDims() }));
        } else {
            Parallelizable::Initialize(GetOptNumThreads(),
                Service::BroadcastTo5D(Service::GetShapeAccordingToLayout(dstInfo.GetDims(), dstInfo.GetLayout())));
        }
    }

    template <PoolingFunction Fun, typename CompType, typename InType, typename OutType, typename WsType>
    void EvaluateNDWithWs(const Shape &start, const Shape &end,
        const InType *src, OutType *dst, WsType *ws, CompType init) const noexcept
    {
        if (axbFormatTag) {
            Pooling::EvaluateAxbFWD<Fun, CompType, InType, OutType, WsType, true>(
                start, end, src, dst, ws, initialDims, reorderedStridesSrc,
                reorderedStridesDst, kernel, strides, dilation, paddingL, ndims, init);
        } else {
            Pooling::EvaluateAbxFWD<Fun, CompType, InType, OutType, WsType, true>(
                start, end, src, dst, ws, initialDims, reorderedDims, reorderedStridesSrc,
                reorderedStridesDst, kernel, strides, dilation, paddingL, ndims, init, reorders);
        }
    }

    template <PoolingFunction Fun, typename CompType, typename InType, typename OutType>
    void EvaluateNDWithoutWs(const Shape &start, const Shape &end,
        const InType *src, OutType *dst, CompType init) const noexcept
    {
        if (axbFormatTag) {
            Pooling::EvaluateAxbFWD<Fun, CompType, InType, OutType, std::uint8_t, false>(
                start, end, src, dst, nullptr, initialDims, reorderedStridesSrc,
                reorderedStridesDst, kernel, strides, dilation, paddingL, ndims, init);
        } else {
            Pooling::EvaluateAbxFWD<Fun, CompType, InType, OutType, std::uint8_t, false>(
                start, end, src, dst, nullptr, initialDims, reorderedDims, reorderedStridesSrc,
                reorderedStridesDst, kernel, strides, dilation, paddingL, ndims, init, reorders);
        }
    }

    template <PoolingFunction Fun, typename CompType, typename InType, typename OutType>
    void EvaluateND(const Shape &start, const Shape &end,
        const InType *src, OutType *dst, void *ws) const noexcept
    {
        CompType init = 0.0f;
        if constexpr (Fun == PoolingFunction::MAX) {
            if (padType == Element::TypeT::F32) {
                init = Pooling::Helpers::MIN_VAL<float>;
            } else if (padType == Element::TypeT::F16) {
                init = Pooling::Helpers::MIN_VAL<__fp16>;
            } else if (padType == Element::TypeT::BF16) {
                init = Pooling::Helpers::MIN_VAL<__bf16>;
            } else if (padType == Element::TypeT::S32) {
                init = Pooling::Helpers::MIN_VAL<std::int32_t>;
            } else if (padType == Element::TypeT::S8) {
                init = Pooling::Helpers::MIN_VAL<std::int8_t>;
            } else if (padType == Element::TypeT::U8) {
                init = Pooling::Helpers::MIN_VAL<std::uint8_t>;
            }
        }

        if (useWs) {
            if constexpr (Fun == PoolingFunction::MAX) {
                if (wsType == Element::TypeT::S32) {
                    std::int32_t *typedWs = static_cast<std::int32_t *>(ws);
                    EvaluateNDWithWs<Fun, CompType, InType, OutType, std::int32_t>(
                        start, end, src, dst, typedWs, init);
                } else {
                    std::uint8_t *typedWs = static_cast<std::uint8_t *>(ws);
                    EvaluateNDWithWs<Fun, CompType, InType, OutType, std::uint8_t>(
                        start, end, src, dst, typedWs, init);
                }
            }
        } else {
            EvaluateNDWithoutWs<Fun, CompType, InType, OutType>(
                start, end, src, dst, init);
        }
    }

    template <typename CompType, typename InType, typename OutType>
    void Evaluate(const Shape &start, const Shape &end,
        const void *src, void *dst, void *ws) const noexcept
    {
        const InType *typedSrc = static_cast<const InType *>(src);
        OutType *typedDst = static_cast<OutType *>(dst);

        switch (alg) {
            case PoolingFunction::MAX:
                EvaluateND<PoolingFunction::MAX, CompType, InType, OutType>(start, end, typedSrc, typedDst, ws);
                break;
            case PoolingFunction::AVG_INCLUDE_PADDING:
                EvaluateND<PoolingFunction::AVG_INCLUDE_PADDING, CompType, InType, OutType>(start, end, typedSrc,
                    typedDst, ws);
                break;
            case PoolingFunction::AVG_EXCLUDE_PADDING:
                EvaluateND<PoolingFunction::AVG_EXCLUDE_PADDING, CompType, InType, OutType>(start, end, typedSrc,
                    typedDst, ws);
                break;
            default:
                break;
        }
    }

    void RunSequential(const Shape &batchSplitDims, const Shape &batchStrides, const void *src,
        void *dst, void *ws) const noexcept(false)
    {
        Shape start = batchStrides;
        SizeType array[MAX_DIMS];
        for (SizeType i = 0; i < batchSplitDims.GetNumDims(); ++i) {
            array[i] = start[i] + batchSplitDims[i];
        }
        Shape end(array, batchSplitDims.GetNumDims());

        if ((srcType != dstType)) {
            Service::ParseType<true, false>(compType, Element::TypeT::F32, Element::TypeT::F32, *this,
                start, end, src, dst, ws);
        } else {
            Service::ParseType<true, false>(compType, srcType, dstType, *this, start, end, src, dst, ws);
        }
    }

    void RunSequential(const Shape &batchSplitDims, const void *src,
        void *dst, void *ws) const noexcept(false)
    {
        SizeType batchStrides[MAX_DIMS] = { 0, 0, 0, 0, 0 };
        RunSequential(batchSplitDims, Shape{batchStrides, batchSplitDims.GetNumDims()}, src, dst, ws);
    }

    void Run(const void *src, void *dst, void *ws) const noexcept(false)
    {
        if ((src == nullptr) || (dst == nullptr) || (useWs && (ws == nullptr))) {
            throw Service::LogicError {"Pooling Layer: one of input matrices is nullptr"};
        }
        if ((src == dst) || (useWs && (ws == dst))) {
            throw Service::LogicError {"Pooling Layer: input and output and workspace must not be the same pointers"};
        }
        const void *srcExec = src;
        if (srcType != dstType) {
            void *tmp = Service::AlignedAlloc(initialDims.GetTotalDimsSize() * sizeof(float));
            ReorderLayer r({initialDims, srcType, Layout::ABCDE}, {initialDims, Element::TypeT::F32, Layout::ABCDE});
            r.Run(src, tmp);
            srcExec = tmp;
        }
        void *dstExec = dst;
        if ((srcType != dstType) && (dstType != Element::TypeT::F32)) {
            dstExec = Service::AlignedAlloc(reorderedDims.GetTotalDimsSize() * sizeof(float));
        }
        Threading::BatchThreader(*this, srcExec, dstExec, ws);
        if (srcType != dstType) {
            Service::Deallocate(srcExec);
        }
        if ((srcType != dstType) && (dstType != Element::TypeT::F32)) {
            ReorderLayer r({reorderedDims, Element::TypeT::F32, Layout::ABCDE},
                {reorderedDims, dstType, Layout::ABCDE});
            r.Run(dstExec, dst);
            Service::Deallocate(dstExec);
        }
    }

    void SetPadType(Element::Type newPadType) noexcept
    {
        if (newPadType != Element::TypeT::UNDEFINED) {
            padType = newPadType;
        }
    }

private:
    Element::Type srcType;
    Element::Type dstType;
    Element::Type wsType;
    Element::Type padType;
    // We don't actually use padType for padding; it is just used to confirm the initial value for padding.
    Element::Type compType;
    Propagation prop;
    PoolingFunction alg;
    Shape strides;
    Shape kernel;
    Shape dilation;
    Shape paddingL;
    Shape paddingR;
    SizeType ndims;
    bool useWs;
    std::vector<ReorderLayer> reorders;
    Shape initialDims;
    Shape reorderedDims;
    Shape reorderedStridesSrc;
    Shape reorderedStridesWs;
    Shape reorderedStridesDst;
    bool axbFormatTag;

    SizeType GetOptNumThreads() const noexcept
    {
        SizeType totalSize = reorderedDims.GetTotalDimsSize();
#ifdef KUDNNL_920F
        if ((totalSize < POOL_ELE_SMALL_NUM_FWD) &&
            ((srcType == Element::TypeT::F32) && (dstType == Element::TypeT::F32))) {
            return std::min<IntType>(NUM_8, omp_get_max_threads()); // Fix:DTS2025050923806
        }
#endif
        if (totalSize < POOL_ELE_NUM_FWD) {
            return std::min<IntType>(std::max<IntType>(NUM_20 * log(totalSize) - NUM_132, NUM_8),
                omp_get_max_threads());
        } else {
            return NUM_120;
        }
    }
};

} // Detail

PoolingLayerFWD::~PoolingLayerFWD() noexcept {}

PoolingLayerFWD::PoolingLayerFWD(const Propagation &propKind, const PoolingFunction &poolingAlg,
    const TensorInfo &srcInfo, const TensorInfo &dstInfo, const Shape &strides,
    const Shape &kernel, const Shape &dilation, const Shape &paddingL, const Shape &paddingR) noexcept(false)
    : pImpl(new Detail::PoolingLayerFWDImpl(propKind, poolingAlg, srcInfo, dstInfo, strides,
        kernel, dilation, paddingL, paddingR)) {}

PoolingLayerFWD::PoolingLayerFWD(const PoolingLayerFWD &other) noexcept(false)
    : pImpl(new Detail::PoolingLayerFWDImpl{*(other.pImpl.get())}) {}

PoolingLayerFWD::PoolingLayerFWD(PoolingLayerFWD &&other) noexcept
    : pImpl(std::move(other.pImpl)) {}

PoolingLayerFWD& PoolingLayerFWD::operator=(const PoolingLayerFWD &other) noexcept(false)
{
    if (this != &other) {
        this->pImpl.reset(new Detail::PoolingLayerFWDImpl{*(other.pImpl.get())});
    }
    return *this;
}

PoolingLayerFWD& PoolingLayerFWD::operator=(PoolingLayerFWD &&other) noexcept
{
    if (this != &other) {
        this->pImpl = std::move(other.pImpl);
    }
    return *this;
}

void PoolingLayerFWD::SetPadType(Element::Type newPadType) noexcept
{
    pImpl->SetPadType(newPadType);
}

void PoolingLayerFWD::Run(const void *src, void *dst, void *ws) const noexcept(false)
{
    pImpl->Run(src, dst, ws);
}

static SizeType ComputeOutPooling(SizeType i, SizeType k, SizeType s, SizeType pl, SizeType pr, SizeType d)
{
    return (i - ((k - 1) * (d + 1) + 1) + (pl + pr)) / s + 1;
}

static Status ValidateAlgSizesPool(const Shape &argStrides, const Shape &argDilates, const Shape &argPaddingL,
    const Shape &argPaddingR, const Shape &kernel, const TensorInfo &src, const TensorInfo &dst)
{
    SizeType numDims = src.GetNumDims();
    SizeType ow = ComputeOutPooling(src.GetDims()[numDims - DIM_1], kernel[numDims - DIM_3],
        argStrides[numDims - DIM_3], argPaddingL[numDims - DIM_3],
        argPaddingR[numDims - DIM_3], argDilates[numDims - DIM_3]);
    if (dst.GetDims()[numDims - 1] != ow) {
        return Status::BAD_ARGUMENTS;
    }
    if (numDims >= DIM_4) {
        SizeType oh = ComputeOutPooling(src.GetDims()[numDims - DIM_2], kernel[numDims - DIM_4],
            argStrides[numDims - DIM_4], argPaddingL[numDims - DIM_4],
            argPaddingR[numDims - DIM_4], argDilates[numDims - DIM_4]);
        if (dst.GetDims()[numDims - DIM_2] != oh) {
            return Status::BAD_ARGUMENTS;
        }
    }
    if (numDims == DIM_5) {
        SizeType od = ComputeOutPooling(src.GetDims()[numDims - DIM_3], kernel[numDims - DIM_5],
            argStrides[numDims - DIM_5], argPaddingL[numDims - DIM_5],
            argPaddingR[numDims - DIM_5], argDilates[numDims - DIM_5]);
        if (dst.GetDims()[numDims - DIM_3] != od) {
            return Status::BAD_ARGUMENTS;
        }
    }
    return Status::SUCCESS;
}

Status PoolingLayerFWD::ValidateInput(const Propagation &propKind, const PoolingFunction &poolingAlg,
    const TensorInfo &srcInfo, const TensorInfo &dstInfo, const Shape &strides,
    const Shape &kernel, const Shape &dilation, const Shape &paddingL, const Shape &paddingR) noexcept
{
    Service::Unused(propKind, poolingAlg, srcInfo, dstInfo, strides, kernel, dilation, paddingL, paddingR);
    if (propKind == Propagation::UNDEFINED) {
        return Status::BAD_ARGUMENTS;
    }
    if (poolingAlg == PoolingFunction::UNDEFINED) {
        return Status::BAD_ARGUMENTS;
    }
    if (srcInfo.GetLayout() != dstInfo.GetLayout()) {
        return Status::UNSUPPORTED;
    }
    bool isUnsupported = srcInfo.GetLayout() != srcInfo.GetStandardABXLayout() &&
        srcInfo.GetLayout() != srcInfo.GetStandardAXBLayout();
    if (isUnsupported) {
        return Status::UNSUPPORTED;
    }
    if (!srcInfo.IsDense() || !dstInfo.IsDense()) {
        return Status::UNSUPPORTED;
    }
    if (srcInfo.GetNumDims() != dstInfo.GetNumDims()) {
        return Status::BAD_ARGUMENTS;
    }
    for (SizeType i = 0; i < kernel.GetNumDims(); ++i) {
        if (kernel[i] == 0) {
            return Status::BAD_ARGUMENTS;
        }
    }
    for (SizeType i = 0; i < strides.GetNumDims(); ++i) {
        if (strides[i] == 0) {
            return Status::BAD_ARGUMENTS;
        }
    }
    bool isBadArguments = kernel.GetNumDims() < DIM_1 || kernel.GetNumDims() > DIM_3;
    isBadArguments = isBadArguments || (strides.GetNumDims() < DIM_1 || strides.GetNumDims() > DIM_3);
    isBadArguments = isBadArguments || (paddingL.GetNumDims() < DIM_1 || paddingL.GetNumDims() > DIM_3);
    isBadArguments = isBadArguments || (paddingR.GetNumDims() < DIM_1 || paddingR.GetNumDims() > DIM_3);
    isBadArguments = isBadArguments || (dilation.GetNumDims() < DIM_1 || dilation.GetNumDims() > DIM_3);
    if (isBadArguments) {
        return Status::BAD_ARGUMENTS;
    }

    if (srcInfo.GetTotalTensorSize() == 0) {
        return Status::UNSUPPORTED;
    }
    if (ValidateAlgSizesPool(strides, dilation, paddingL, paddingR, kernel,
        srcInfo, dstInfo) != Status::SUCCESS) {
        return Status::BAD_ARGUMENTS;
    }

    return Status::SUCCESS;
}

} // KuDNN
