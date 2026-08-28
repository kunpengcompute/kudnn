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
#include <cmath>
#include <memory>
#include <utility>
#include <numeric>
#include <functional>

#include "kudnn.hpp"
#include "internal_funcs.hpp"
#include "kernels/convolution/kudnn_conv_deconv_base.hpp"

namespace KuDNN {
namespace Detail {
namespace Helpers {

static constexpr SizeType TRANSPOSE_THREAD_THREADS = 800;

template <typename Type>
KUDNN_INLINE static inline void Transpose(const Type *in, Type *out, SizeType n0, SizeType n1, SizeType n2)
{
    const SizeType blockSize = 16;
    SizeType transposeWorkload = n0 * n1 * n2;
    SizeType numThreads = Threading::GetMaxNumThreads();
    if (transposeWorkload < TRANSPOSE_THREAD_THREADS) {
        numThreads = 1;
    }
    for (SizeType i = 0; i < n0; i += blockSize) {
        for (SizeType j = 0; j < n1; j += blockSize) {
#pragma omp parallel for num_threads(numThreads) collapse(3)
            for (SizeType bi = i; bi < std::min(i + blockSize, n0); ++bi) {
                for (SizeType bj = j; bj < std::min(j + blockSize, n1); ++bj) {
                    for (SizeType k = 0; k < n2; ++k) {
                        out[(bj * n0 + bi) * n2 + k] = in[(bi * n1 + bj) * n2 + k];
                    }
                }
            }
        }
    }
}

KUDNN_INLINE static inline Shape GetCorrectWeightsShape(Shape initShape, bool isGroup)
{
    if (isGroup) {
        switch (initShape.GetNumDims()) {
            case DIM_4: {
                return Shape{initShape[IDX_0], initShape[IDX_2],
                    initShape[IDX_1], initShape[IDX_3]};
            }
            case DIM_5: {
                return Shape{initShape[IDX_0], initShape[IDX_2],
                    initShape[IDX_1], initShape[IDX_3], initShape[IDX_4]};
            }
            default: {
                return Shape{};
            }
        }
    } else {
        switch (initShape.GetNumDims()) {
            case DIM_3: {
                return Shape{initShape[IDX_1], initShape[IDX_0],
                    initShape[IDX_2]};
            }
            case DIM_4: {
                return Shape{initShape[IDX_1], initShape[IDX_0],
                    initShape[IDX_2], initShape[IDX_3]};
            }
            case DIM_5: {
                return Shape{initShape[IDX_1], initShape[IDX_0],
                    initShape[IDX_2], initShape[IDX_3], initShape[IDX_4]};
            }
            default: {
                return Shape{};
            }
        }
    }
}

} // Helpers

class DeconvolutionLayerFWDImpl final : public ConvolutionBase {
public:
    DeconvolutionLayerFWDImpl(const TensorInfo &src, const TensorInfo &weights,
        const TensorInfo &dst, const TensorInfo &bias, const Shape &strides, const Shape &dilates,
        const Shape &paddingL, const Shape &paddingR, DeconvolutionAlgorithm deconvFWDAlg) noexcept(false)
        : ConvolutionBase(src,
            TensorInfo(Helpers::GetCorrectWeightsShape(weights.GetDims(), weights.GetNumDims() != src.GetNumDims()),
                weights.GetType(), weights.GetLayout()), dst, strides, dilates, paddingL, paddingR),
            alg(deconvFWDAlg), inputTensor(src), kernelTensor(weights), outputTensor(dst), biaTensor(bias)
    {
        Service::ThrowOnStatus(DeconvolutionLayerFWD::ValidateInput(
            src, weights, dst, bias, strides, dilates, paddingL, paddingR, alg), "Deconvolution layer");
        CheckIntMultOverflow(std::make_tuple(params.outputShape[NUM_1],
            params.kernelShape[NUM_2], params.kernelShape[NUM_3], params.kernelShape[NUM_4],
            params.inputShape[NUM_2], params.inputShape[NUM_3], params.inputShape[NUM_4]),
            std::make_tuple((params.outputShape[NUM_0] - 1), params.inputShape[NUM_1],
                params.inputShape[NUM_2], params.inputShape[NUM_3], params.inputShape[NUM_4]),
            std::make_tuple((params.outputShape[NUM_0] - 1), params.outputShape[NUM_1],
                params.outputShape[NUM_2], params.outputShape[NUM_3], params.outputShape[NUM_4]));

        SizeType totalSize = (params.outputShape[NUM_1] / params.groups) *
            params.kernelShape[NUM_2] * params.kernelShape[NUM_3] * params.kernelShape[NUM_4] *
            params.inputShape[NUM_2] * params.inputShape[NUM_3] * params.inputShape[NUM_4] * outputType.GetSize();
        totalSize += params.outputShape[NUM_2] * params.outputShape[NUM_3] *
            params.outputShape[NUM_4] + outputType.GetSize();
        if (totalSize > Detail::BYTES_AVAILABLE) {
            throw Service::LogicError {"Deconvolution: exceeded available memory limit"};
        }

        SizeType inChannelPerGroup = params.inputShape[NUM_1] / params.groups;
        SizeType outChannelPerGroup = params.outputShape[NUM_1] / params.groups;
        SizeType m = outChannelPerGroup * params.kernelShape[NUM_2] *
            params.kernelShape[NUM_3] * params.kernelShape[NUM_4];
        SizeType n = params.inputShape[NUM_2] * params.inputShape[NUM_3] * params.inputShape[NUM_4];
        SizeType k = inChannelPerGroup;
        gemm.reset(new Gemm{
            {{m, k}, kernelType, Layout::COL_MAJOR},
            {{k, n}, inputType, Layout::ROW_MAJOR},
            {{m, n}, Element::TypeT::F32, Layout::ROW_MAJOR}
        });
    }

    DeconvolutionLayerFWDImpl(const DeconvolutionLayerFWDImpl &other) noexcept(false)
        : ConvolutionBase(other), alg(other.alg),  inputTensor(other.inputTensor),
        kernelTensor(other.kernelTensor), outputTensor(other.outputTensor), biaTensor(other.biaTensor),
        gemm(new Gemm{*(other.gemm)}) {}

    DeconvolutionLayerFWDImpl(DeconvolutionLayerFWDImpl &&) = delete;

    DeconvolutionLayerFWDImpl& operator=(const DeconvolutionLayerFWDImpl &other) = delete;

    DeconvolutionLayerFWDImpl &operator=(DeconvolutionLayerFWDImpl &&) = delete;

    void Run(const void *src, const void *wei,
            void *dst, const void *bia) noexcept(false)
    {
        Service::Unused(alg);
        if ((src == nullptr) || (wei == nullptr) || (dst == nullptr)) {
            throw Service::LogicError {"Deconvolution Layer: one of input matrices is nullptr"};
        }

        const void *srcExec = src;
        const void *weiExec = wei;
        void *dstExec = dst;
        const void *biasExec = bia;
        BeforeRunReorderLayer(&srcExec, &weiExec, &dstExec, &biasExec, src, wei, bia);

        ConvParseType<true, true>(inputType, kernelType, Element::TypeT::F32, *this,
            srcExec, weiExec, dstExec, biasExec);

        if (inputTensor.GetLayout() != inputTensor.GetStandardABXLayout()) {
            Service::Deallocate(srcExec);
        }
        if (kernelTensor.GetLayout() != kernelTensor.GetStandardABXLayout()) {
            Service::Deallocate(weiExec);
        }
        if ((outputTensor.GetLayout() != outputTensor.GetStandardABXLayout())
            || (outputType != Element::TypeT::F32)) {
            ReorderLayer r({outputTensor.GetDims(), Element::TypeT::F32,
                outputTensor.GetStandardABXLayout()}, {outputTensor.GetDims(),
                outputTensor.GetType(), outputTensor.GetLayout()});
            r.Run(dstExec, dst);
            Service::Deallocate(dstExec);
        }
        if (bia != nullptr && biaTensor.GetType() != Element::TypeT::F32) {
            Service::Deallocate(biasExec);
        }
    }

    template <typename InType, typename KerType, typename OutType>
    void Evaluate(const void *src, const void *wei, void *dst, const void *bia) noexcept(false)
    {
        const InType *typedSrc = static_cast<const InType *>(src);
        const KerType *typedWei = static_cast<const KerType *>(wei);
        OutType *typedDst = static_cast<OutType *>(dst);
        const float *typedBia = static_cast<const float *>(bia);

        std::vector<KerType, Service::AlignedAllocator<KerType>> tmpWei(params.groups *
            std::accumulate(params.kernelShape.begin(), params.kernelShape.end(),
                static_cast<IntType>(1), std::multiplies<IntType>()),
            Vector::Convert<float, KerType>::Do(0.0f));

        SizeType batch = params.inputShape[NUM_0];
        SizeType inChannel = params.inputShape[NUM_1];
        SizeType outChannel = params.outputShape[NUM_1];

        SizeType inChannelPerGroup = inChannel / params.groups;
        SizeType outChannelPerGroup = outChannel / params.groups;
        SizeType outputSpatialSize = params.outputShape[NUM_2] *
            params.outputShape[NUM_3] * params.outputShape[NUM_4];

        for (SizeType g = 0; g < params.groups; ++g) {
            Helpers::Transpose(typedWei + g * inChannelPerGroup * outChannelPerGroup *
                params.kernelShape[NUM_2] * params.kernelShape[NUM_3] * params.kernelShape[NUM_4],
                tmpWei.data() + g * inChannelPerGroup * outChannelPerGroup *
                params.kernelShape[NUM_2] * params.kernelShape[NUM_3] * params.kernelShape[NUM_4],
                outChannelPerGroup, inChannelPerGroup,
                params.kernelShape[NUM_2] * params.kernelShape[NUM_3] * params.kernelShape[NUM_4]);
        }

        std::vector<float, Service::AlignedAllocator<float>> colTmp(
            outChannelPerGroup * params.kernelShape[NUM_2] * params.kernelShape[NUM_3] * params.kernelShape[NUM_4] *
            params.inputShape[NUM_2] * params.inputShape[NUM_3] * params.inputShape[NUM_4], 0.0f);
        for (SizeType b = 0; b < batch; ++b) {
            for (SizeType g = 0; g < params.groups; ++g) {
                const InType *srcTmp = typedSrc +
                    b * inChannel * params.inputShape[NUM_2] *
                        params.inputShape[NUM_3] * params.inputShape[NUM_4] +
                    g * inChannelPerGroup * params.inputShape[NUM_2] *
                        params.inputShape[NUM_3] * params.inputShape[NUM_4];
                const KerType *weiTmp = tmpWei.data() + g * inChannelPerGroup * outChannelPerGroup *
                    params.kernelShape[NUM_2] * params.kernelShape[NUM_3] * params.kernelShape[NUM_4];
                OutType *dstTmp = typedDst +
                    b * outChannel * params.outputShape[NUM_2] *
                        params.outputShape[NUM_3] * params.outputShape[NUM_4] +
                    g * outChannelPerGroup * params.outputShape[NUM_2] *
                        params.outputShape[NUM_3] * params.outputShape[NUM_4];
                const float *biaPtr = nullptr;
                if (bia != nullptr) {
                    biaPtr = typedBia + g * outChannelPerGroup;
                }

                gemm.get()->Run(weiTmp, srcTmp, colTmp.data(), nullptr);
                // OutType always F32
                if constexpr (std::is_same_v<OutType, float>) {
                    Detail::Col2img<float, OutType>::Do(colTmp.data(), outChannelPerGroup,
                        params.outputShape[NUM_2], params.outputShape[NUM_3], params.outputShape[NUM_4],
                        params.inputShape[NUM_2], params.inputShape[NUM_3], params.inputShape[NUM_4],
                        params.kernelShape[NUM_2], params.kernelShape[NUM_3], params.kernelShape[NUM_4],
                        params.paddingL[NUM_0], params.paddingL[NUM_1], params.paddingL[NUM_2],
                        params.stride[NUM_0], params.stride[NUM_1], params.stride[NUM_2],
                        params.dilate[NUM_0], params.dilate[NUM_1], params.dilate[NUM_2],
                        dstTmp, params.ndims);
                } else {
                    throw Service::Unsupported {"Unsupported output data type"};
                }

                if (bia != nullptr) {
                    for (SizeType oc = 0; oc < outChannelPerGroup; ++oc) {
                        float biasValue = biaPtr[oc];
                        OutType *dstChannelPtr = dstTmp + oc * outputSpatialSize;
                        for (SizeType idx = 0; idx < outputSpatialSize; ++idx) {
                            dstChannelPtr[idx] = Vector::Convert<float, OutType>::Do(
                                Vector::Convert<OutType, float>::Do(dstChannelPtr[idx]) + biasValue);
                        }
                    }
                }
            }
        }
    }

private:
    DeconvolutionAlgorithm alg;
    TensorInfo inputTensor;
    TensorInfo kernelTensor;
    TensorInfo outputTensor;
    TensorInfo biaTensor;
    std::unique_ptr<Gemm> gemm;

    void BeforeRunReorderLayer(const void **srcExec, const void **weiExec, void **dstExec, const void **biasExec,
        const void *src, const void *wei, const void *bia)
    {
        if (inputTensor.GetLayout() != inputTensor.GetStandardABXLayout()) {
            void *tmp = Service::AlignedAlloc(inputTensor.GetTotalTensorSize() * inputTensor.GetType().GetSize());
            ReorderLayer r({inputTensor.GetDims(), inputTensor.GetType(),
                inputTensor.GetLayout()}, {inputTensor.GetDims(),
                inputTensor.GetType(), inputTensor.GetStandardABXLayout()});
            r.Run(src, tmp);
            *srcExec = tmp;
        }
        if (kernelTensor.GetLayout() != kernelTensor.GetStandardABXLayout()) {
            void *tmp = Service::AlignedAlloc(kernelTensor.GetTotalTensorSize() * kernelTensor.GetType().GetSize());
            ReorderLayer r({kernelTensor.GetDims(), kernelTensor.GetType(),
                kernelTensor.GetLayout()}, {kernelTensor.GetDims(),
                kernelTensor.GetType(), kernelTensor.GetStandardABXLayout()});
            r.Run(wei, tmp);
            *weiExec = tmp;
        }
        if ((outputTensor.GetLayout() != outputTensor.GetStandardABXLayout()) ||
            (outputType != Element::TypeT::F32)) {
            *dstExec = Service::AlignedAlloc(
                outputTensor.GetTotalTensorSize() * sizeof(float));
        }
        if (bia != nullptr && biaTensor.GetType() != Element::TypeT::F32) {
            void *tmp = Service::AlignedAlloc(
                biaTensor.GetTotalTensorSize() * Element::Type{Element::TypeT::F32}.GetSize());
            ReorderLayer r({biaTensor.GetDims(), biaTensor.GetType(),
                biaTensor.GetLayout()}, {biaTensor.GetDims(),
                Element::TypeT::F32, biaTensor.GetLayout()});
            r.Run(bia, tmp);
            *biasExec = tmp;
        }
    }
};

class DeconvolutionLayerBWDDataImpl final : public ConvolutionBase {
public:
    DeconvolutionLayerBWDDataImpl(const TensorInfo &diffDst, const TensorInfo &weights,
        const TensorInfo &diffSrc, const Shape &strides, const Shape &dilates,
        const Shape &paddingL, const Shape &paddingR, DeconvolutionAlgorithm deconvBWDAlg) noexcept(false)
        : ConvolutionBase(diffSrc,
            TensorInfo(Helpers::GetCorrectWeightsShape(weights.GetDims(), weights.GetNumDims() != diffSrc.GetNumDims()),
                weights.GetType(), weights.GetLayout()), diffDst, strides, dilates, paddingL, paddingR),
        alg(deconvBWDAlg), inputTensor(diffSrc), kernelTensor(weights), outputTensor(diffDst)
    {
        Service::ThrowOnStatus(DeconvolutionLayerBWDData::ValidateInput(
            diffDst, weights, diffSrc, strides, dilates, paddingL, paddingR, alg), "Deconvolution layer");
        CheckIntMultOverflow(std::make_tuple(params.outputShape[NUM_1],
            params.kernelShape[NUM_2], params.kernelShape[NUM_3], params.kernelShape[NUM_4],
            params.inputShape[NUM_2], params.inputShape[NUM_3], params.inputShape[NUM_4]),
            std::make_tuple((params.outputShape[NUM_0] - 1), params.inputShape[NUM_1],
            params.inputShape[NUM_2], params.inputShape[NUM_3], params.inputShape[NUM_4]),
            std::make_tuple((params.outputShape[NUM_0] - 1), params.outputShape[NUM_1],
            params.outputShape[NUM_2], params.outputShape[NUM_3], params.outputShape[NUM_4]));

        SizeType totalSize = (params.outputShape[NUM_1] / params.groups) *
            params.kernelShape[NUM_2] * params.kernelShape[NUM_3] * params.kernelShape[NUM_4] *
            params.inputShape[NUM_2] * params.inputShape[NUM_3] * params.inputShape[NUM_4] *
            inputType.GetSize();
        if (totalSize > Detail::BYTES_AVAILABLE) {
            throw Service::LogicError {"Deconvolution: exceeded available memory limit"};
        }

        SizeType inChannelPerGroup = params.inputShape[NUM_1] / params.groups;
        SizeType outChannelPerGroup = params.outputShape[NUM_1] / params.groups;
        SizeType m = inChannelPerGroup;
        SizeType n = params.inputShape[NUM_2] * params.inputShape[NUM_3] * params.inputShape[NUM_4];
        SizeType k = outChannelPerGroup * params.kernelShape[NUM_2] *
            params.kernelShape[NUM_3] * params.kernelShape[NUM_4];
        gemm.reset(new Gemm{
            {{m, k}, kernelType, Layout::ROW_MAJOR},
            {{k, n}, outputType, Layout::ROW_MAJOR},
            {{m, n}, inputType, Layout::ROW_MAJOR}
        });
    }

    DeconvolutionLayerBWDDataImpl(const DeconvolutionLayerBWDDataImpl &other) noexcept(false)
        : ConvolutionBase(other), alg(other.alg), inputTensor(other.inputTensor), kernelTensor(other.kernelTensor),
        outputTensor(other.outputTensor), gemm(new Gemm{*(other.gemm)}) {}

    DeconvolutionLayerBWDDataImpl(DeconvolutionLayerBWDDataImpl &&) = delete;

    DeconvolutionLayerBWDDataImpl& operator=(const DeconvolutionLayerBWDDataImpl &other) = delete;

    DeconvolutionLayerBWDDataImpl &operator=(DeconvolutionLayerBWDDataImpl &&) = delete;

    void Run(const void *diffDst, const void *wei, void *diffSrc) noexcept(false)
    {
        Service::Unused(alg);
        if ((diffDst == nullptr) || (wei == nullptr) || (diffSrc == nullptr)) {
            throw Service::LogicError {"Deconvolution Layer: one of input matrices is nullptr"};
        }

        const void *diffDstExec = diffDst;
        if (outputTensor.GetLayout() != outputTensor.GetStandardABXLayout()) {
            void *tmp = Service::AlignedAlloc(outputTensor.GetTotalTensorSize() * outputTensor.GetType().GetSize());
            ReorderLayer r({outputTensor.GetDims(), outputTensor.GetType(),
                outputTensor.GetLayout()}, {outputTensor.GetDims(),
                outputTensor.GetType(), outputTensor.GetStandardABXLayout()});
            r.Run(diffDst, tmp);
            diffDstExec = tmp;
        }
        const void *weiExec = wei;
        if (kernelTensor.GetLayout() != kernelTensor.GetStandardABXLayout()) {
            void *tmp = Service::AlignedAlloc(kernelTensor.GetTotalTensorSize() * kernelTensor.GetType().GetSize());
            ReorderLayer r({kernelTensor.GetDims(), kernelTensor.GetType(),
                kernelTensor.GetLayout()}, {kernelTensor.GetDims(),
                kernelTensor.GetType(), kernelTensor.GetStandardABXLayout()});
            r.Run(wei, tmp);
            weiExec = tmp;
        }
        void *diffSrcExec = diffSrc;
        if (inputTensor.GetLayout() != inputTensor.GetStandardABXLayout()) {
            diffSrcExec = Service::AlignedAlloc(
                inputTensor.GetTotalTensorSize() * inputTensor.GetType().GetSize());
        }

        ConvParseType<false, false>(outputType, kernelType, inputType, *this,
            diffDstExec, weiExec, diffSrcExec);

        if (outputTensor.GetLayout() != outputTensor.GetStandardABXLayout()) {
            Service::Deallocate(diffDstExec);
        }
        if (kernelTensor.GetLayout() != kernelTensor.GetStandardABXLayout()) {
            Service::Deallocate(weiExec);
        }
        if (inputTensor.GetLayout() != inputTensor.GetStandardABXLayout()) {
            ReorderLayer r({inputTensor.GetDims(), inputTensor.GetType(),
                inputTensor.GetStandardABXLayout()}, {inputTensor.GetDims(),
                inputTensor.GetType(), inputTensor.GetLayout()});
            r.Run(diffSrcExec, diffSrc);
            Service::Deallocate(diffSrcExec);
        }
    }

    template <typename InType, typename KerType, typename OutType>
    void Evaluate(const void *diffDst, const void *wei, void *diffSrc) const noexcept(false)
    {
        OutType *typedDiffSrc = static_cast<OutType *>(diffSrc);
        const InType *typedDiffDst = static_cast<const InType *>(diffDst);
        const KerType *typedWei = static_cast<const KerType *>(wei);

        std::vector<KerType, Service::AlignedAllocator<KerType>> tmpWei(params.groups *
            std::accumulate(params.kernelShape.begin(), params.kernelShape.end(),
                static_cast<IntType>(1), std::multiplies<IntType>()),
            Vector::Convert<float, KerType>::Do(0.0f));

        SizeType batch = params.inputShape[NUM_0];
        SizeType inChannel = params.inputShape[NUM_1];
        SizeType outChannel = params.outputShape[NUM_1];

        SizeType inChannelPerGroup = inChannel / params.groups;
        SizeType outChannelPerGroup = outChannel / params.groups;

        for (SizeType g = 0; g < params.groups; ++g) {
            Helpers::Transpose(typedWei + g * inChannelPerGroup * outChannelPerGroup *
                params.kernelShape[NUM_2] * params.kernelShape[NUM_3] * params.kernelShape[NUM_4],
                tmpWei.data() + g * inChannelPerGroup * outChannelPerGroup *
                params.kernelShape[NUM_2] * params.kernelShape[NUM_3] * params.kernelShape[NUM_4],
                outChannelPerGroup, inChannelPerGroup,
                params.kernelShape[NUM_2] * params.kernelShape[NUM_3] * params.kernelShape[NUM_4]);
        }

        int numThreads = Threading::GetMaxNumThreads();

        std::vector<InType, Service::AlignedAllocator<InType>> colTmp(
            outChannelPerGroup * params.kernelShape[NUM_2] * params.kernelShape[NUM_3] * params.kernelShape[NUM_4] *
            params.inputShape[NUM_2] * params.inputShape[NUM_3] * params.inputShape[NUM_4],
            Vector::Convert<float, InType>::Do(0.0f));
        for (SizeType b = 0; b < batch; ++b) {
            for (SizeType g = 0; g < params.groups; ++g) {
                const InType *diffDstTmp = typedDiffDst +
                    b * outChannel * params.outputShape[NUM_2] *
                        params.outputShape[NUM_3] * params.outputShape[NUM_4] +
                    g * outChannelPerGroup * params.outputShape[NUM_2] *
                        params.outputShape[NUM_3] * params.outputShape[NUM_4];
                const KerType *weiTmp = tmpWei.data() + g * inChannelPerGroup * outChannelPerGroup *
                    params.kernelShape[NUM_2] * params.kernelShape[NUM_3] * params.kernelShape[NUM_4];
                OutType *diffSrcTmp = typedDiffSrc +
                    b * inChannel * params.inputShape[NUM_2] *
                        params.inputShape[NUM_3] * params.inputShape[NUM_4] +
                    g * inChannelPerGroup * params.inputShape[NUM_2] *
                        params.inputShape[NUM_3] * params.inputShape[NUM_4];

                Detail::Img2col<InType, ImgLayout::NN>::Do(diffDstTmp, outChannelPerGroup,
                    params.outputShape[NUM_2], params.outputShape[NUM_3], params.outputShape[NUM_4],
                    params.inputShape[NUM_2], params.inputShape[NUM_3], params.inputShape[NUM_4],
                    params.kernelShape[NUM_2], params.kernelShape[NUM_3], params.kernelShape[NUM_4],
                    params.paddingL[NUM_0], params.paddingL[NUM_1], params.paddingL[NUM_2],
                    params.stride[NUM_0], params.stride[NUM_1], params.stride[NUM_2],
                    params.dilate[NUM_0], params.dilate[NUM_1], params.dilate[NUM_2],
                    colTmp.data(), params.ndims, numThreads);
                gemm.get()->Run(weiTmp, colTmp.data(), diffSrcTmp, nullptr);
            }
        }
    }

private:
    DeconvolutionAlgorithm alg;
    TensorInfo inputTensor;
    TensorInfo kernelTensor;
    TensorInfo outputTensor;
    std::unique_ptr<Gemm> gemm;
};

class DeconvolutionLayerBWDWeightsImpl final : public ConvolutionBase {
public:
    DeconvolutionLayerBWDWeightsImpl(const TensorInfo &diffDst, const TensorInfo &src, const TensorInfo &diffWeights,
        const TensorInfo &diffBias, const Shape &strides, const Shape &dilates, const Shape &paddingL,
        const Shape &paddingR, DeconvolutionAlgorithm deconvBWDWeiAlg) noexcept(false)
        : ConvolutionBase(src, TensorInfo(Helpers::GetCorrectWeightsShape(diffWeights.GetDims(),
            diffWeights.GetNumDims() != src.GetNumDims()), diffWeights.GetType(), diffWeights.GetLayout()),
        diffDst, strides, dilates, paddingL, paddingR), alg(deconvBWDWeiAlg),
        inputTensor(src), kernelTensor(diffWeights), outputTensor(diffDst), diffBiasTensor(diffBias)
    {
        Service::ThrowOnStatus(DeconvolutionLayerBWDWeights::ValidateInput(
            diffDst, src, diffWeights, diffBias, strides, dilates, paddingL, paddingR, alg), "Deconvolution layer");
        CheckIntMultOverflow(std::make_tuple(params.outputShape[NUM_1],
            params.kernelShape[NUM_2], params.kernelShape[NUM_3], params.kernelShape[NUM_4],
            params.inputShape[NUM_2], params.inputShape[NUM_3], params.inputShape[NUM_4]),
            std::make_tuple((params.inputShape[NUM_0] - 1), params.inputShape[NUM_1],
                params.inputShape[NUM_2], params.inputShape[NUM_3], params.inputShape[NUM_4]),
            std::make_tuple((params.inputShape[NUM_0] - 1), params.outputShape[NUM_1],
                params.outputShape[NUM_2], params.outputShape[NUM_3], params.outputShape[NUM_4]),
            std::make_tuple(params.kernelShape[diffWeights.GetNumDims() != src.GetNumDims() ? IDX_1 : IDX_0],
            params.kernelShape[diffWeights.GetNumDims() != src.GetNumDims() ? IDX_2 : IDX_1],
            params.kernelShape[NUM_2], params.kernelShape[NUM_3], params.kernelShape[NUM_4]));

        SizeType totalSize = (params.outputShape[NUM_1] / params.groups) *
            params.kernelShape[NUM_2] * params.kernelShape[NUM_3] * params.kernelShape[NUM_4] *
            params.inputShape[NUM_2] * params.inputShape[NUM_3] * params.inputShape[NUM_4] * inputType.GetSize();
        totalSize += params.outputShape[NUM_2] * params.outputShape[NUM_3] *
            params.outputShape[NUM_4] * inputType.GetSize();
        if (totalSize > Detail::BYTES_AVAILABLE) {
            throw Service::LogicError {"Deconvolution: exceeded available memory limit"};
        }

        SizeType inChannelPerGroup = params.inputShape[NUM_1] / params.groups;
        SizeType outChannelPerGroup = params.outputShape[NUM_1] / params.groups;
        SizeType m = inChannelPerGroup;
        SizeType n = outChannelPerGroup * params.kernelShape[NUM_2] *
            params.kernelShape[NUM_3] * params.kernelShape[NUM_4];
        SizeType k = params.inputShape[NUM_2] * params.inputShape[NUM_3] * params.inputShape[NUM_4];
        gemm.reset(new Gemm{
            {{m, k}, inputType, Layout::ROW_MAJOR},
            {{k, n}, outputType, Layout::COL_MAJOR},
            {{m, n}, kernelType, Layout::ROW_MAJOR}
        });
    }

    DeconvolutionLayerBWDWeightsImpl(const DeconvolutionLayerBWDWeightsImpl &other) noexcept(false)
        : ConvolutionBase(other), alg(other.alg), inputTensor(other.inputTensor), kernelTensor(other.kernelTensor),
        outputTensor(other.outputTensor), diffBiasTensor(other.diffBiasTensor),
        gemm(new Gemm{*(other.gemm)}) {}

    DeconvolutionLayerBWDWeightsImpl(DeconvolutionLayerBWDWeightsImpl &&) = delete;

    DeconvolutionLayerBWDWeightsImpl& operator=(const DeconvolutionLayerBWDWeightsImpl &other) = delete;

    DeconvolutionLayerBWDWeightsImpl &operator=(DeconvolutionLayerBWDWeightsImpl &&) = delete;

    void Run(const void *diffDst, const void *src,
            void *diffWei, void *diffBias) noexcept(false)
    {
        Service::Unused(alg);
        if ((diffDst == nullptr) || (src == nullptr) || (diffWei == nullptr)) {
            throw Service::LogicError {"Deconvolution Layer: one of input matrices is nullptr"};
        }

        const void *diffDstExec = diffDst;
        const void *srcExec = src;
        void *diffWeiExec = diffWei;
        void *diffBiasExec = diffBias;
        BeforeRunReorderLayer(&diffDstExec, &srcExec, &diffWeiExec, &diffBiasExec,
            diffDst, src, diffBias);

        ConvParseType<false, false>(outputType, inputType, kernelType, *this, diffDstExec,
            srcExec, diffWeiExec, diffBiasExec);

        if (outputTensor.GetLayout() != outputTensor.GetStandardABXLayout()) {
            Service::Deallocate(diffDstExec);
        }
        if (inputTensor.GetLayout() != inputTensor.GetStandardABXLayout()) {
            Service::Deallocate(srcExec);
        }
        if (kernelTensor.GetLayout() != kernelTensor.GetStandardABXLayout()) {
            ReorderLayer r({kernelTensor.GetDims(), kernelTensor.GetType(),
                kernelTensor.GetStandardABXLayout()}, {kernelTensor.GetDims(),
                kernelTensor.GetType(), kernelTensor.GetLayout()});
            r.Run(diffWeiExec, diffWei);
            Service::Deallocate(diffWeiExec);
        }
        if ((diffBias != nullptr) && (diffBiasTensor.GetType() != Element::TypeT::F32)) {
            ReorderLayer r({diffBiasTensor.GetDims(), Element::TypeT::F32,
                diffBiasTensor.GetLayout()}, {diffBiasTensor.GetDims(),
                diffBiasTensor.GetType(), diffBiasTensor.GetLayout()});
            r.Run(diffBiasExec, diffBias);
            Service::Deallocate(diffBiasExec);
        }
    }

    template <typename InType, typename KerType, typename OutType>
    void Evaluate(const void *diffDst, const void *src, void *diffWei, void *diffBias) noexcept(false)
    {
        const InType *typedDiffDst = static_cast<const InType *>(diffDst);
        const KerType *typedSrc = static_cast<const KerType *>(src);
        OutType *typedDiffWei = static_cast<OutType *>(diffWei);
        float *typedDiffBias = static_cast<float *>(diffBias);

        std::vector<OutType, Service::AlignedAllocator<OutType>> tmpWei(params.groups *
            std::accumulate(params.kernelShape.begin(), params.kernelShape.end(),
                static_cast<IntType>(1), std::multiplies<IntType>()),
            Vector::Convert<float, OutType>::Do(0.0f));

        SizeType batch = params.inputShape[NUM_0];
        SizeType inChannel = params.inputShape[NUM_1];
        SizeType outChannel = params.outputShape[NUM_1];

        SizeType inChannelPerGroup = inChannel / params.groups;
        SizeType outChannelPerGroup = outChannel / params.groups;
        SizeType outputSpatialSize = params.outputShape[NUM_2] *
            params.outputShape[NUM_3] * params.outputShape[NUM_4];

        int numThreads = Threading::GetMaxNumThreads();

        std::vector<InType, Service::AlignedAllocator<InType>> cData(outChannelPerGroup *
            params.kernelShape[NUM_2] * params.kernelShape[NUM_3] * params.kernelShape[NUM_4] *
            params.inputShape[NUM_2] * params.inputShape[NUM_3] * params.inputShape[NUM_4],
            Vector::Convert<float, InType>::Do(0.0f));

        if (typedDiffBias != nullptr) {
            std::fill(typedDiffBias, typedDiffBias + params.groups * outChannelPerGroup, 0.0f);
        }

        for (SizeType deb = 0; deb < batch; ++deb) {
            for (SizeType deg = 0; deg < params.groups; ++deg) {
                const InType *diffDstTmp = typedDiffDst +
                    deb * outChannel * params.outputShape[NUM_2] *
                        params.outputShape[NUM_3] * params.outputShape[NUM_4] +
                    deg * outChannelPerGroup * params.outputShape[NUM_2] *
                        params.outputShape[NUM_3] * params.outputShape[NUM_4];
                const KerType *srcTmp = typedSrc +
                    deb * inChannel * params.inputShape[NUM_2] *
                        params.inputShape[NUM_3] * params.inputShape[NUM_4] +
                    deg * inChannelPerGroup * params.inputShape[NUM_2] *
                        params.inputShape[NUM_3] * params.inputShape[NUM_4];
                OutType *diffWeiTmp = tmpWei.data() + deg * inChannelPerGroup * outChannelPerGroup *
                    params.kernelShape[NUM_2] * params.kernelShape[NUM_3] * params.kernelShape[NUM_4];
                float *diffBiasTmp = nullptr;
                if (diffBias != nullptr) {
                    diffBiasTmp = typedDiffBias + deg * outChannelPerGroup;
                }

                Detail::Img2col<InType, ImgLayout::NN>::Do(diffDstTmp, outChannelPerGroup,
                    params.outputShape[NUM_2], params.outputShape[NUM_3], params.outputShape[NUM_4],
                    params.inputShape[NUM_2], params.inputShape[NUM_3], params.inputShape[NUM_4],
                    params.kernelShape[NUM_2], params.kernelShape[NUM_3], params.kernelShape[NUM_4],
                    params.paddingL[NUM_0], params.paddingL[NUM_1], params.paddingL[NUM_2],
                    params.stride[NUM_0], params.stride[NUM_1], params.stride[NUM_2],
                    params.dilate[NUM_0], params.dilate[NUM_1], params.dilate[NUM_2],
                    cData.data(), params.ndims, numThreads);
                gemm.get()->Run(srcTmp, cData.data(), diffWeiTmp, nullptr, 1.f, 1.f);

                if (diffBiasTmp != nullptr) {
                    for (SizeType deoc = 0; deoc < outChannelPerGroup; ++deoc) {
                        typename std::conditional<std::is_floating_point<InType>::value, double, int64_t>::type
                            biasSum = 0;
                        const InType* diffDstChannelPtr = diffDstTmp + deoc * outputSpatialSize;
                        for (SizeType idx = 0; idx < outputSpatialSize; ++idx) {
                            biasSum += static_cast<decltype(biasSum)>(
                                Vector::Convert<InType, float>::Do(diffDstChannelPtr[idx]));
                        }
                        diffBiasTmp[deoc] = diffBiasTmp[deoc] + biasSum;
                    }
                }
            }
        }

        for (SizeType g = 0; g < params.groups; ++g) {
            Helpers::Transpose(tmpWei.data() + g * inChannelPerGroup * outChannelPerGroup *
                params.kernelShape[NUM_2] * params.kernelShape[NUM_3] * params.kernelShape[NUM_4],
                typedDiffWei + g * inChannelPerGroup * outChannelPerGroup *
                params.kernelShape[NUM_2] * params.kernelShape[NUM_3] * params.kernelShape[NUM_4],
                inChannelPerGroup, outChannelPerGroup,
                params.kernelShape[NUM_2] * params.kernelShape[NUM_3] * params.kernelShape[NUM_4]);
        }
    }

private:
    DeconvolutionAlgorithm alg;
    TensorInfo inputTensor;
    TensorInfo kernelTensor;
    TensorInfo outputTensor;
    TensorInfo diffBiasTensor;
    std::unique_ptr<Gemm> gemm;

    void BeforeRunReorderLayer(const void **diffDstExec, const void **srcExec, void **diffWeiExec,
        void **diffBiasExec, const void *diffDst, const void *src, void *diffBias) noexcept(false)
    {
        if (outputTensor.GetLayout() != outputTensor.GetStandardABXLayout()) {
            void *tmp = Service::AlignedAlloc(outputTensor.GetTotalTensorSize() * outputTensor.GetType().GetSize());
            ReorderLayer r({outputTensor.GetDims(), outputTensor.GetType(),
                outputTensor.GetLayout()}, {outputTensor.GetDims(),
                outputTensor.GetType(), outputTensor.GetStandardABXLayout()});
            r.Run(diffDst, tmp);
            *diffDstExec = tmp;
        }
        if (inputTensor.GetLayout() != inputTensor.GetStandardABXLayout()) {
            void *tmp = Service::AlignedAlloc(inputTensor.GetTotalTensorSize() * inputTensor.GetType().GetSize());
            ReorderLayer r({inputTensor.GetDims(), inputTensor.GetType(),
                inputTensor.GetLayout()}, {inputTensor.GetDims(),
                inputTensor.GetType(), inputTensor.GetStandardABXLayout()});
            r.Run(src, tmp);
            *srcExec = tmp;
        }
        if (kernelTensor.GetLayout() != kernelTensor.GetStandardABXLayout()) {
            *diffWeiExec = Service::AlignedAlloc(
                kernelTensor.GetTotalTensorSize() * kernelTensor.GetType().GetSize());
        }
        if ((diffBias != nullptr) && (diffBiasTensor.GetType() != Element::TypeT::F32)) {
            *diffBiasExec = Service::AlignedAlloc(
                diffBiasTensor.GetTotalTensorSize() * Element::Type{Element::TypeT::F32}.GetSize());
        }
    }
};

} // Detail

// FWD

DeconvolutionLayerFWD::~DeconvolutionLayerFWD() noexcept {}

DeconvolutionLayerFWD::DeconvolutionLayerFWD(const TensorInfo &src, const TensorInfo &weights,
    const TensorInfo &dst, const TensorInfo &bias,
    const Shape &strides, const Shape &dilates,
    const Shape &paddingL, const Shape &paddingR,
    DeconvolutionAlgorithm alg) noexcept(false)
    : pImpl(new Detail::DeconvolutionLayerFWDImpl(src, weights, dst, bias, strides, dilates, paddingL, paddingR, alg))
{
}

DeconvolutionLayerFWD::DeconvolutionLayerFWD(const TensorInfo &src,
    const TensorInfo &weights, const TensorInfo &dst, const TensorInfo &bias,
    const Shape &strides, const Shape &paddingL, const Shape &paddingR, DeconvolutionAlgorithm alg) noexcept(false)
    : pImpl(new Detail::DeconvolutionLayerFWDImpl(src, weights, dst, bias, strides, {0, 0, 0},
    paddingL, paddingR, alg)) {}

DeconvolutionLayerFWD::DeconvolutionLayerFWD(const DeconvolutionLayerFWD &other) noexcept(false)
    : pImpl(new Detail::DeconvolutionLayerFWDImpl{*(other.pImpl.get())}) {}

DeconvolutionLayerFWD::DeconvolutionLayerFWD(DeconvolutionLayerFWD &&other) noexcept
    : pImpl(std::move(other.pImpl)) {}

DeconvolutionLayerFWD& DeconvolutionLayerFWD::operator=(const DeconvolutionLayerFWD &other) noexcept(false)
{
    if (this != &other) {
        this->pImpl.reset(new Detail::DeconvolutionLayerFWDImpl{*(other.pImpl.get())});
    }
    return *this;
}

DeconvolutionLayerFWD& DeconvolutionLayerFWD::operator=(DeconvolutionLayerFWD &&other) noexcept
{
    if (this != &other) {
        this->pImpl = std::move(other.pImpl);
    }
    return *this;
}

void DeconvolutionLayerFWD::Run(const void *src, const void *wei,
    void *dst, const void *bia) const noexcept(false)
{
    pImpl->Run(src, wei, dst, bia);
}

Status DeconvolutionLayerFWD::ValidateInput(const TensorInfo &src, const TensorInfo &weights,
    const TensorInfo &dst, const TensorInfo &bias,
    const Shape &strides, const Shape &dilates,
    const Shape &paddingL, const Shape &paddingR,
    DeconvolutionAlgorithm alg) noexcept
{
    if (Detail::ConvolutionBase::ValidateAlgSizesDeconv(strides, dilates, paddingL, paddingR, alg, src, weights, dst)
        != Status::SUCCESS) {
        return Status::BAD_ARGUMENTS;
    }
    if (Detail::ConvolutionBase::ValidateDataShapes(src, weights, dst, bias) != Status::SUCCESS) {
        return Status::BAD_ARGUMENTS;
    }
    if (Detail::ConvolutionBase::ValidateDataTypes(strides, dilates, paddingL, paddingR, src, weights, dst, bias)
        != Status::SUCCESS) {
        return Status::BAD_ARGUMENTS;
    }
    // Supported data type:
    // F32:F32:F32/F16:F16:F16/BF16:BF16:BF16
    // F16:F16:F32/BF16:BF16:F32
    // S8:S8:x/U8:S8:x
    bool isSupported = ((src.GetType() == Element::TypeT::F32) && (weights.GetType() == Element::TypeT::F32)
        && (dst.GetType() == Element::TypeT::F32)) || ((src.GetType() == Element::TypeT::F16)
        && (weights.GetType() == Element::TypeT::F16) && (dst.GetType() == Element::TypeT::F16))
        || ((src.GetType() == Element::TypeT::F16) && (weights.GetType() == Element::TypeT::F16)
        && (dst.GetType() == Element::TypeT::F32)) || ((src.GetType() == Element::TypeT::BF16)
        && (weights.GetType() == Element::TypeT::BF16) && (dst.GetType() == Element::TypeT::F32))
        || ((src.GetType() == Element::TypeT::BF16) && (weights.GetType() == Element::TypeT::BF16)
        && (dst.GetType() == Element::TypeT::BF16)) || ((src.GetType() == Element::TypeT::S8)
        && (weights.GetType() == Element::TypeT::S8)) || ((src.GetType() == Element::TypeT::U8)
        && (weights.GetType() == Element::TypeT::S8));
    if (!isSupported) {
        return Status::UNSUPPORTED;
    }
    return Status::SUCCESS;
}

// BWD_D

DeconvolutionLayerBWDData::~DeconvolutionLayerBWDData() noexcept {}

DeconvolutionLayerBWDData::DeconvolutionLayerBWDData(const TensorInfo &diffDst,
    const TensorInfo &weights, const TensorInfo &diffSrc, const Shape &strides,
    const Shape &dilates, const Shape &paddingL, const Shape &paddingR,
    DeconvolutionAlgorithm alg) noexcept(false)
    : pImpl(new Detail::DeconvolutionLayerBWDDataImpl(diffDst, weights, diffSrc, strides, dilates,
    paddingL, paddingR, alg)) {}

DeconvolutionLayerBWDData::DeconvolutionLayerBWDData(const TensorInfo &diffDst,
    const TensorInfo &weights, const TensorInfo &diffSrc, const Shape &strides,
    const Shape &paddingL, const Shape &paddingR, DeconvolutionAlgorithm alg) noexcept(false)
    : pImpl(new Detail::DeconvolutionLayerBWDDataImpl(diffDst, weights, diffSrc, strides, {0, 0, 0},
    paddingL, paddingR, alg)) {}

DeconvolutionLayerBWDData::DeconvolutionLayerBWDData(const DeconvolutionLayerBWDData &other) noexcept(false)
    : pImpl(new Detail::DeconvolutionLayerBWDDataImpl{*(other.pImpl.get())}) {}

DeconvolutionLayerBWDData::DeconvolutionLayerBWDData(DeconvolutionLayerBWDData &&other) noexcept
    : pImpl(std::move(other.pImpl)) {}

DeconvolutionLayerBWDData& DeconvolutionLayerBWDData::operator=(const DeconvolutionLayerBWDData &other) noexcept(false)
{
    if (this != &other) {
        this->pImpl.reset(new Detail::DeconvolutionLayerBWDDataImpl{*(other.pImpl.get())});
    }
    return *this;
}

DeconvolutionLayerBWDData& DeconvolutionLayerBWDData::operator=(DeconvolutionLayerBWDData &&other) noexcept
{
    if (this != &other) {
        this->pImpl = std::move(other.pImpl);
    }
    return *this;
}

void DeconvolutionLayerBWDData::Run(const void *diffDst, const void *wei,
    void *diffSrc) const noexcept(false)
{
    pImpl->Run(diffDst, wei, diffSrc);
}

Status DeconvolutionLayerBWDData::ValidateInput(const TensorInfo &diffDst, const TensorInfo &weights,
    const TensorInfo &diffSrc, const Shape &strides, const Shape &dilates,
    const Shape &paddingL, const Shape &paddingR, DeconvolutionAlgorithm alg) noexcept
{
    if (Detail::ConvolutionBase::ValidateAlgSizesDeconv(strides, dilates, paddingL, paddingR, alg, diffSrc, weights,
        diffDst) != Status::SUCCESS) {
        return Status::BAD_ARGUMENTS;
    }
    if (Detail::ConvolutionBase::ValidateDataShapes(diffSrc, weights, diffDst) != Status::SUCCESS) {
        return Status::BAD_ARGUMENTS;
    }
    if (Detail::ConvolutionBase::ValidateDataTypes(strides, dilates, paddingL, paddingR, diffDst, weights,
        diffSrc) != Status::SUCCESS) {
        return Status::BAD_ARGUMENTS;
    }
    // Supported data type:
    // F32:F32:F32/F16:F16:F16/BF16:BF16:BF16
    // F32:F16:F16/F32:BF16:BF16
    bool isSupported = ((diffDst.GetType() == Element::TypeT::F32) && (weights.GetType() == Element::TypeT::F32)
        && (diffSrc.GetType() == Element::TypeT::F32)) || ((diffDst.GetType() == Element::TypeT::F16)
        && (weights.GetType() == Element::TypeT::F16) && (diffSrc.GetType() == Element::TypeT::F16))
        || ((diffDst.GetType() == Element::TypeT::BF16) && (weights.GetType() == Element::TypeT::BF16)
        && (diffSrc.GetType() == Element::TypeT::BF16)) || ((diffDst.GetType() == Element::TypeT::F16)
        && (weights.GetType() == Element::TypeT::F16) && (diffSrc.GetType() == Element::TypeT::F32))
        || ((diffDst.GetType() == Element::TypeT::BF16) && (weights.GetType() == Element::TypeT::BF16)
        && (diffSrc.GetType() == Element::TypeT::F32));
    if (!isSupported) {
        return Status::UNSUPPORTED;
    }
    return Status::SUCCESS;
}

// BWD_W

DeconvolutionLayerBWDWeights::~DeconvolutionLayerBWDWeights() noexcept {}

DeconvolutionLayerBWDWeights::DeconvolutionLayerBWDWeights(const TensorInfo &diffDst,
    const TensorInfo &src, const TensorInfo &diffWeights,
    const TensorInfo &diffBias, const Shape &strides,
    const Shape &dilates, const Shape &paddingL,
    const Shape &paddingR, DeconvolutionAlgorithm alg) noexcept(false)
    : pImpl(new Detail::DeconvolutionLayerBWDWeightsImpl(diffDst, src, diffWeights, diffBias, strides,
    dilates, paddingL, paddingR, alg)) {}

DeconvolutionLayerBWDWeights::DeconvolutionLayerBWDWeights(const TensorInfo &diffDst,
    const TensorInfo &src, const TensorInfo &diffWeights,
    const TensorInfo &diffBias, const Shape &strides,
    const Shape &paddingL, const Shape &paddingR,
    DeconvolutionAlgorithm alg) noexcept(false)
    : pImpl(new Detail::DeconvolutionLayerBWDWeightsImpl(diffDst, src, diffWeights, diffBias, strides, {0, 0, 0},
    paddingL, paddingR, alg)) {}

DeconvolutionLayerBWDWeights::DeconvolutionLayerBWDWeights(const DeconvolutionLayerBWDWeights &other) noexcept(false)
    : pImpl(new Detail::DeconvolutionLayerBWDWeightsImpl{*(other.pImpl.get())}) {}

DeconvolutionLayerBWDWeights::DeconvolutionLayerBWDWeights(DeconvolutionLayerBWDWeights &&other) noexcept
    : pImpl(std::move(other.pImpl)) {}

DeconvolutionLayerBWDWeights& DeconvolutionLayerBWDWeights::operator=(
    const DeconvolutionLayerBWDWeights &other) noexcept(false)
{
    if (this != &other) {
        this->pImpl.reset(new Detail::DeconvolutionLayerBWDWeightsImpl{*(other.pImpl.get())});
    }
    return *this;
}

DeconvolutionLayerBWDWeights& DeconvolutionLayerBWDWeights::operator=(DeconvolutionLayerBWDWeights &&other) noexcept
{
    if (this != &other) {
        this->pImpl = std::move(other.pImpl);
    }
    return *this;
}

void DeconvolutionLayerBWDWeights::Run(const void *diffDst,
    const void *src, void *diffWei, void *diffBias) const noexcept(false)
{
    pImpl->Run(diffDst, src, diffWei, diffBias);
}

Status DeconvolutionLayerBWDWeights::ValidateInput(const TensorInfo &diffDst,
    const TensorInfo &src, const TensorInfo &diffWeights,
    const TensorInfo &diffBias, const Shape &strides,
    const Shape &dilates, const Shape &paddingL,
    const Shape &paddingR, DeconvolutionAlgorithm alg) noexcept
{
    bool isBadArguments = (Detail::ConvolutionBase::ValidateAlgSizesDeconv(
        strides, dilates, paddingL, paddingR, alg, src, diffWeights, diffDst) != Status::SUCCESS) ||
        ((Detail::ConvolutionBase::ValidateDataShapes(src, diffWeights, diffDst, diffBias) != Status::SUCCESS)) ||
        (Detail::ConvolutionBase::ValidateDataTypes(
            strides, dilates, paddingL, paddingR, diffDst, src, diffWeights, diffBias) != Status::SUCCESS);
    if (isBadArguments) {
        return Status::BAD_ARGUMENTS;
    }
    // Supported data type:
    // F32:F32:F32/F16:F16:F16/BF16:BF16:BF16
    // F16:F32:F16/BF16:F32:BF16
    bool isSupported = ((diffDst.GetType() == Element::TypeT::F32) && (src.GetType() == Element::TypeT::F32)
        && (diffWeights.GetType() == Element::TypeT::F32)) || ((diffDst.GetType() == Element::TypeT::F16)
        && (src.GetType() == Element::TypeT::F16) && (diffWeights.GetType() == Element::TypeT::F16))
        || ((diffDst.GetType() == Element::TypeT::BF16) && (src.GetType() == Element::TypeT::BF16)
        && (diffWeights.GetType() == Element::TypeT::BF16)) || ((diffDst.GetType() == Element::TypeT::F16)
        && (src.GetType() == Element::TypeT::F16) && (diffWeights.GetType() == Element::TypeT::F32))
        || ((diffDst.GetType() == Element::TypeT::BF16) && (src.GetType() == Element::TypeT::BF16)
        && (diffWeights.GetType() == Element::TypeT::F32));
    if (!isSupported) {
        return Status::UNSUPPORTED;
    }
    return Status::SUCCESS;
}

} // KuDNN
