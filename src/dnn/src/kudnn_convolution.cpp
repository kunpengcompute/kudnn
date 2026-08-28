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
#include <utility>
#include <numeric>
#include <functional>
#include "internal_funcs.hpp"
#include "kudnn_convolution_base.hpp"
#include "kernels/convolution/kudnn_conv_deconv_base.hpp"
#include "kernels/convolution/kudnn_conv_const.hpp"

namespace KuDNN {
namespace Detail {
class ConvolutionLayerBWDDataImpl final : public ConvolutionBase {
public:
    ConvolutionLayerBWDDataImpl(const TensorInfo &diffDst, const TensorInfo &weights,
        const TensorInfo &diffSrc, const Shape &strides, const Shape &dilates,
        const Shape &paddingL, const Shape &paddingR, ConvolutionAlgorithm convBWDAlg) noexcept(false)
        : ConvolutionBase(diffSrc, weights, diffDst, strides, dilates, paddingL, paddingR), alg(convBWDAlg),
        inputTensor(diffSrc), kernelTensor(weights), outputTensor(diffDst)
    {
        Service::ThrowOnStatus(ConvolutionLayerBWDData::ValidateInput(
            diffDst, weights, diffSrc, strides, dilates, paddingL, paddingR, alg), "Convolution layer");
        CheckIntMultOverflow(std::make_tuple(params.inputShape[NUM_1],
            params.kernelShape[NUM_2], params.kernelShape[NUM_3], params.kernelShape[NUM_4],
            params.outputShape[NUM_2], params.outputShape[NUM_3], params.outputShape[NUM_4]),
            std::make_tuple((params.inputShape[NUM_0] - 1), params.outputShape[NUM_1],
                params.outputShape[NUM_2], params.outputShape[NUM_3], params.outputShape[NUM_4]),
            std::make_tuple((params.inputShape[NUM_0] - 1), params.inputShape[NUM_1],
                params.inputShape[NUM_2], params.inputShape[NUM_3], params.inputShape[NUM_4]));

        SizeType totalSize = (params.inputShape[NUM_1] / params.groups) *
            params.kernelShape[NUM_2] * params.kernelShape[NUM_3] * params.kernelShape[NUM_4] *
            params.outputShape[NUM_2] * params.outputShape[NUM_3] * params.outputShape[NUM_4] *
            outputType.GetSize();
        if (totalSize > Detail::BYTES_AVAILABLE) {
            throw Service::LogicError {"Convolution: exceeded available memory limit"};
        }
        SizeType inChannelPerGroup = params.inputShape[NUM_1] / params.groups;
        SizeType outChannelPerGroup = params.outputShape[NUM_1] / params.groups;
        SizeType m = inChannelPerGroup * params.kernelShape[NUM_2] *
            params.kernelShape[NUM_3] * params.kernelShape[NUM_4];
        SizeType n = params.outputShape[NUM_2] * params.outputShape[NUM_3] * params.outputShape[NUM_4];
        SizeType k = outChannelPerGroup;
        kernelType = Element::TypeT::F32;
        outputType = Element::TypeT::F32;
        inputType = Element::TypeT::F32;
        gemm.reset(new Gemm{
            {{m, k}, kernelType, Layout::COL_MAJOR},
            {{k, n}, outputType, Layout::ROW_MAJOR},
            {{m, n}, inputType, Layout::ROW_MAJOR}
        });
    }

    ConvolutionLayerBWDDataImpl(const ConvolutionLayerBWDDataImpl &other) noexcept(false)
        : ConvolutionBase(other), alg(other.alg), inputTensor(other.inputTensor), kernelTensor(other.kernelTensor),
        outputTensor(other.outputTensor), gemm(new Gemm{*(other.gemm)}) {}

    ConvolutionLayerBWDDataImpl(ConvolutionLayerBWDDataImpl &&) = delete;

    ConvolutionLayerBWDDataImpl& operator=(const ConvolutionLayerBWDDataImpl &other) = delete;

    ConvolutionLayerBWDDataImpl &operator=(ConvolutionLayerBWDDataImpl &&) = delete;

    void Run(const void *diffDst, const void *wei, void *diffSrc) noexcept(false)
    {
        Service::Unused(alg); // At the moment, we always calculate using gemm
        if ((diffDst == nullptr) || (wei == nullptr) || (diffSrc == nullptr)) {
            throw Service::LogicError {"Convolution Layer: one of input matrices is nullptr"};
        }
        const void *diffDstExec = diffDst;
        if (outputTensor.GetType() != Element::TypeT::F32) {
            void *tmp = Service::AlignedAlloc(outputTensor.GetTotalTensorSize() * sizeof(float));
            ReorderLayer r({outputTensor.GetDims(), outputTensor.GetType(),
                outputTensor.GetLayout()}, {outputTensor.GetDims(),
                Element::TypeT::F32, outputTensor.GetStandardABXLayout()});
            r.Run(diffDst, tmp);
            diffDstExec = tmp;
        } else if (outputTensor.GetLayout() != outputTensor.GetStandardABXLayout()) {
            void *tmp = Service::AlignedAlloc(outputTensor.GetTotalTensorSize() * outputTensor.GetType().GetSize());
            ReorderLayer r({outputTensor.GetDims(), outputTensor.GetType(),
                outputTensor.GetLayout()}, {outputTensor.GetDims(),
                Element::TypeT::F32, outputTensor.GetStandardABXLayout()});
            r.Run(diffDst, tmp);
            diffDstExec = tmp;
        }
        const void *weiExec = wei;
        if (kernelTensor.GetType() != Element::TypeT::F32) {
            void *tmp = Service::AlignedAlloc(kernelTensor.GetTotalTensorSize() * sizeof(float));
            ReorderLayer r({kernelTensor.GetDims(), kernelTensor.GetType(),
                kernelTensor.GetLayout()}, {kernelTensor.GetDims(),
                Element::TypeT::F32, kernelTensor.GetStandardABXLayout()});
            r.Run(wei, tmp);
            weiExec = tmp;
        } else if (kernelTensor.GetLayout() != kernelTensor.GetStandardABXLayout()) {
            void *tmp = Service::AlignedAlloc(kernelTensor.GetTotalTensorSize() * kernelTensor.GetType().GetSize());
            ReorderLayer r({kernelTensor.GetDims(), kernelTensor.GetType(),
                kernelTensor.GetLayout()}, {kernelTensor.GetDims(),
                Element::TypeT::F32, kernelTensor.GetStandardABXLayout()});
            r.Run(wei, tmp);
            weiExec = tmp;
        }

        void *diffSrcExec = diffSrc;
        if (inputTensor.GetLayout() != inputTensor.GetStandardABXLayout() ||
            inputTensor.GetType() != Element::TypeT::F32) {
            diffSrcExec = Service::AlignedAlloc(inputTensor.GetTotalTensorSize() * sizeof(float));
        }
        ConvParseType<true, true>(outputType, kernelType, inputType, *this,
            diffDstExec, weiExec, diffSrcExec);

        if (outputTensor.GetType() != Element::TypeT::F32 ||
            outputTensor.GetLayout() != outputTensor.GetStandardABXLayout()) {
                Service::Deallocate(diffDstExec);
        }
        if (kernelTensor.GetType() != Element::TypeT::F32 ||
            kernelTensor.GetLayout() != kernelTensor.GetStandardABXLayout()) {
                Service::Deallocate(weiExec);
        }
        if (inputTensor.GetLayout() != inputTensor.GetStandardABXLayout() ||
            inputTensor.GetType() != Element::TypeT::F32) {
                ReorderLayer r({inputTensor.GetDims(),
                    Element::TypeT::F32,
                    inputTensor.GetStandardABXLayout()}, {inputTensor.GetDims(),
                    inputTensor.GetType(), inputTensor.GetLayout()});
                r.Run(diffSrcExec, diffSrc);
                Service::Deallocate(diffSrcExec);
        }
    }

    template <typename InType, typename KerType, typename OutType>
    void Evaluate(const void *diffDst, const void *wei, void *diffSrc) const noexcept(false)
    {
        const InType *typedDiffDst = static_cast<const InType *>(diffDst);
        const KerType *typedWei = static_cast<const KerType *>(wei);
        OutType *typedDiffSrc = static_cast<OutType *>(diffSrc);

        SizeType batch = params.inputShape[NUM_0];
        SizeType inChannel = params.inputShape[NUM_1];
        SizeType outChannel = params.outputShape[NUM_1];

        SizeType inChannelPerGroup = inChannel / params.groups;
        SizeType outChannelPerGroup = outChannel / params.groups;
        std::vector<float, Service::AlignedAllocator<float>> colTmp(
            inChannelPerGroup * params.kernelShape[NUM_2] * params.kernelShape[NUM_3] * params.kernelShape[NUM_4] *
            params.outputShape[NUM_2] * params.outputShape[NUM_3] * params.outputShape[NUM_4], 0.0f);
        for (SizeType b = 0; b < batch; ++b) {
            for (SizeType g = 0; g < params.groups; ++g) {
                const InType *diffDstTmp = typedDiffDst +
                    b * outChannel * params.outputShape[NUM_2] *
                        params.outputShape[NUM_3] * params.outputShape[NUM_4] +
                    g * outChannelPerGroup * params.outputShape[NUM_2] *
                        params.outputShape[NUM_3] * params.outputShape[NUM_4];
                const KerType *weiTmp = typedWei + g * inChannelPerGroup * outChannelPerGroup *
                    params.kernelShape[NUM_2] * params.kernelShape[NUM_3] * params.kernelShape[NUM_4];
                OutType *diffSrcTmp = typedDiffSrc +
                    b * inChannel * params.inputShape[NUM_2] *
                        params.inputShape[NUM_3] * params.inputShape[NUM_4] +
                    g * inChannelPerGroup * params.inputShape[NUM_2] *
                        params.inputShape[NUM_3] * params.inputShape[NUM_4];
                gemm.get()->Run(weiTmp, diffDstTmp, colTmp.data(), nullptr);
                // OutType always F32
                if constexpr (std::is_same_v<OutType, float>) {
                    Detail::Col2img<float, OutType>::Do(colTmp.data(), inChannelPerGroup,
                        params.inputShape[NUM_2], params.inputShape[NUM_3], params.inputShape[NUM_4],
                        params.outputShape[NUM_2], params.outputShape[NUM_3], params.outputShape[NUM_4],
                        params.kernelShape[NUM_2], params.kernelShape[NUM_3], params.kernelShape[NUM_4],
                        params.paddingL[NUM_0], params.paddingL[NUM_1], params.paddingL[NUM_2],
                        params.stride[NUM_0], params.stride[NUM_1], params.stride[NUM_2],
                        params.dilate[NUM_0], params.dilate[NUM_1], params.dilate[NUM_2],
                        diffSrcTmp, params.ndims);
                } else {
                    throw Service::Unsupported {"Unsupported output data type"};
                }
            }
        }
    }

private:
    ConvolutionAlgorithm alg;
    TensorInfo inputTensor;
    TensorInfo kernelTensor;
    TensorInfo outputTensor;
    std::unique_ptr<Gemm> gemm;
};

class ConvolutionLayerBWDWeightsImpl final : public ConvolutionBase {
public:
    ConvolutionLayerBWDWeightsImpl(const TensorInfo &diffDst, const TensorInfo &src,
        const TensorInfo &diffWei, const TensorInfo &diffBias, const Shape &strides, const Shape &dilates,
        const Shape &paddingL, const Shape &paddingR, ConvolutionAlgorithm convAlg) noexcept(false)
        : ConvolutionBase(src, diffWei, diffDst, strides, dilates, paddingL, paddingR),
        alg(convAlg), inputTensor(src), kernelTensor(diffWei), outputTensor(diffDst), diffBiasTensor(diffBias),
        is920BplantForm(false)
    {
        Service::ThrowOnStatus(ConvolutionLayerBWDWeights::ValidateInput(
            diffDst, src, diffWei, diffBias, strides, dilates, paddingL, paddingR, alg), "Convolution layer");
        CheckIntMultOverflow(std::make_tuple(params.inputShape[NUM_1],
            params.kernelShape[NUM_2], params.kernelShape[NUM_3], params.kernelShape[NUM_4],
            params.outputShape[NUM_2], params.outputShape[NUM_3], params.outputShape[NUM_4]),
            std::make_tuple((params.inputShape[NUM_0] - 1), params.kernelShape[NUM_1],
                params.outputShape[NUM_2], params.outputShape[NUM_3], params.outputShape[NUM_4]),
            std::make_tuple((params.inputShape[NUM_0] - 1), params.inputShape[NUM_1],
                params.inputShape[NUM_2], params.inputShape[NUM_3], params.inputShape[NUM_4]));

        SizeType totalSize = (params.inputShape[NUM_1] / params.groups) *
            params.kernelShape[NUM_2] * params.kernelShape[NUM_3] * params.kernelShape[NUM_4] *
            params.outputShape[NUM_2] * params.outputShape[NUM_3] * params.outputShape[NUM_4] * inputType.GetSize();
        totalSize += params.outputShape[NUM_2] * params.outputShape[NUM_3] *
            params.outputShape[NUM_4] * outputType.GetSize();
        totalSize += (params.outputShape[NUM_1] / params.groups) *
            params.outputShape[NUM_2] * params.outputShape[NUM_3] * params.outputShape[NUM_4] * outputType.GetSize();
        if (totalSize > Detail::BYTES_AVAILABLE) {
            throw Service::LogicError {"Convolution: exceeded available memory limit"};
        }

        SizeType inChannelPerGroup = params.inputShape[NUM_1] / params.groups;
        SizeType outChannelPerGroup = params.outputShape[NUM_1] / params.groups;
        SizeType m = outChannelPerGroup;
        SizeType n = inChannelPerGroup * params.kernelShape[NUM_2] *
            params.kernelShape[NUM_3] * params.kernelShape[NUM_4];
        SizeType k = params.outputShape[NUM_2] * params.outputShape[NUM_3] * params.outputShape[NUM_4];
#ifdef KUDNNL_920B
        // for 920B f16:f32:f16 used shgemm interface, the result deviates significantly, need to use sgemm interface
        bool bIsInAndOutF16KerF32 = (kernelTensor.GetType() == Element::TypeT::F32) &&
            (inputTensor.GetType() == Element::TypeT::F16) &&
            (outputTensor.GetType() == Element::TypeT::F16);
        if (bIsInAndOutF16KerF32) {
            outputType = Element::TypeT::F32;
            kernelType = Element::TypeT::F32;
            inputType = Element::TypeT::F32;
        }
        is920BplantForm = true;
#endif
        gemm.reset(new Gemm{
            {{m, k}, outputType, Layout::ROW_MAJOR},
            {{k, n}, inputType, Layout::COL_MAJOR},
            {{m, n}, kernelType, Layout::ROW_MAJOR}
        });
    }

    ConvolutionLayerBWDWeightsImpl(const ConvolutionLayerBWDWeightsImpl &other) noexcept(false)
        : ConvolutionBase(other), alg(other.alg), inputTensor(other.inputTensor), kernelTensor(other.kernelTensor),
        outputTensor(other.outputTensor), diffBiasTensor(other.diffBiasTensor),
        gemm(new Gemm{*(other.gemm)}), is920BplantForm(other.is920BplantForm) {}

    ConvolutionLayerBWDWeightsImpl(ConvolutionLayerBWDWeightsImpl &&) = delete;

    ConvolutionLayerBWDWeightsImpl& operator=(const ConvolutionLayerBWDWeightsImpl &other) = delete;

    ConvolutionLayerBWDWeightsImpl &operator=(ConvolutionLayerBWDWeightsImpl &&) = delete;

    void Run(const void *diffDst, const void *src, void *diffWei, void *diffBias) noexcept(false)
    {
        Service::Unused(alg); // At the moment, we always calculate using gemm
        if ((diffDst == nullptr) || (src == nullptr) || (diffWei == nullptr)) {
            throw Service::LogicError {"Convolution Layer: one of input matrices is nullptr"};
        }
        const void *diffDstExec = diffDst;
        const void *srcExec = src;
        void *diffWeiExec = diffWei;
        void *diffBiasExec = diffBias;
        bool bIsInAndOutF16KerF32 = (kernelTensor.GetType() == Element::TypeT::F32) &&
            (inputTensor.GetType() == Element::TypeT::F16) &&
            (outputTensor.GetType() == Element::TypeT::F16);
        BeforeRunReorderLayer(&diffDstExec, &srcExec, &diffWeiExec,
            &diffBiasExec, diffDst, src, diffBias, bIsInAndOutF16KerF32);
        if (bIsInAndOutF16KerF32 && is920BplantForm) {
            ConvParseType<false, true>(outputType, inputType, kernelType, *this,
                diffDstExec, srcExec, diffWeiExec, diffBiasExec);
        } else {
            ConvParseType<false, false>(outputType, inputType, kernelType, *this,
                diffDstExec, srcExec, diffWeiExec, diffBiasExec);
        }
        AfterRunReorderLayer(&diffDstExec, &srcExec, &diffWeiExec,
            &diffBiasExec, diffWei, diffBias, bIsInAndOutF16KerF32);
    }

    template <typename InType, typename KerType, typename OutType>
    void Evaluate(const void *diffDst, const void *src, void *diffWei, void *diffBias) noexcept(false)
    {
        const InType *typedDiffDst = static_cast<const InType *>(diffDst);
        const KerType *typedSrc = static_cast<const KerType *>(src);
        OutType *typedDiffWei = static_cast<OutType *>(diffWei);
        float *typedDiffBias = static_cast<float *>(diffBias);

        std::fill(typedDiffWei, typedDiffWei + params.groups *
            std::accumulate(params.kernelShape.begin(), params.kernelShape.end(),
                static_cast<IntType>(1), std::multiplies<IntType>()),
            Vector::Convert<float, OutType>::Do(0.0f));

        SizeType outChannel = params.outputShape[NUM_1];
        SizeType batch = params.inputShape[NUM_0];
        SizeType inChannel = params.inputShape[NUM_1];

        SizeType inChannelPerGroup = inChannel / params.groups;
        SizeType outChannelPerGroup = outChannel / params.groups;
        SizeType outputSpatialSize = params.outputShape[NUM_2] *
            params.outputShape[NUM_3] * params.outputShape[NUM_4];

        int numThreads = Threading::GetMaxNumThreads();

        std::vector<KerType, Service::AlignedAllocator<KerType>> cData(inChannelPerGroup *
            params.kernelShape[NUM_2] * params.kernelShape[NUM_3] * params.kernelShape[NUM_4] *
            params.outputShape[NUM_2] * params.outputShape[NUM_3] * params.outputShape[NUM_4],
            Vector::Convert<float, KerType>::Do(0.0f));

        if (typedDiffBias != nullptr) {
            std::fill(typedDiffBias, typedDiffBias + params.groups * outChannelPerGroup, 0.0f);
        }

        for (SizeType b = 0; b < batch; ++b) {
            for (SizeType g = 0; g < params.groups; ++g) {
                const InType *diffDstTmp = typedDiffDst +
                    b * outChannel * params.outputShape[NUM_2] *
                        params.outputShape[NUM_3] * params.outputShape[NUM_4] +
                    g * outChannelPerGroup * params.outputShape[NUM_2] *
                        params.outputShape[NUM_3] * params.outputShape[NUM_4];
                const KerType *srcTmp = typedSrc +
                    b * inChannel * params.inputShape[NUM_2] *
                        params.inputShape[NUM_3] * params.inputShape[NUM_4] +
                    g * inChannelPerGroup * params.inputShape[NUM_2] *
                        params.inputShape[NUM_3] * params.inputShape[NUM_4];
                OutType *weiTmp = typedDiffWei + g * inChannelPerGroup * outChannelPerGroup *
                    params.kernelShape[NUM_2] * params.kernelShape[NUM_3] * params.kernelShape[NUM_4];
                float *diffBiasTmp = nullptr;
                if (typedDiffBias != nullptr) {
                    diffBiasTmp = typedDiffBias + g * outChannelPerGroup;
                }

                Detail::Img2col<KerType, ImgLayout::NN>::Do(srcTmp, inChannelPerGroup,
                    params.inputShape[NUM_2], params.inputShape[NUM_3], params.inputShape[NUM_4],
                    params.outputShape[NUM_2], params.outputShape[NUM_3], params.outputShape[NUM_4],
                    params.kernelShape[NUM_2], params.kernelShape[NUM_3], params.kernelShape[NUM_4],
                    params.paddingL[NUM_0], params.paddingL[NUM_1], params.paddingL[NUM_2],
                    params.stride[NUM_0], params.stride[NUM_1], params.stride[NUM_2],
                    params.dilate[NUM_0], params.dilate[NUM_1], params.dilate[NUM_2],
                    cData.data(), params.ndims, numThreads);
                gemm.get()->Run(diffDstTmp, cData.data(), weiTmp, nullptr, 1.f, 1.f);

                if (diffBiasTmp != nullptr) {
                    for (SizeType oc = 0; oc < outChannelPerGroup; ++oc) {
                        typename std::conditional<std::is_floating_point<InType>::value, double, int64_t>::type
                            biasSum = 0;
                        const InType* diffDstChannelPtr = diffDstTmp + oc * outputSpatialSize;
                        for (SizeType idx = 0; idx < outputSpatialSize; ++idx) {
                            biasSum += static_cast<decltype(biasSum)>(
                                Vector::Convert<InType, float>::Do(diffDstChannelPtr[idx]));
                        }
                        diffBiasTmp[oc] = diffBiasTmp[oc] + biasSum;
                    }
                }
            }
        }
    }

private:
    ConvolutionAlgorithm alg;
    TensorInfo inputTensor;
    TensorInfo kernelTensor;
    TensorInfo outputTensor;
    TensorInfo diffBiasTensor;
    std::unique_ptr<Gemm> gemm;
    bool is920BplantForm;
    void BeforeRunReorderLayer(const void **diffDstExec, const void **srcExec, void **diffWeiExec,
        void **diffBiasExec, const void *diffDst, const void *src, void *diffBias,
        bool isInAndOutF16) noexcept(false)
    {
        if (isInAndOutF16 && is920BplantForm) {
                void *tmp = Service::AlignedAlloc(outputTensor.GetTotalTensorSize() * sizeof(float));
                ReorderLayer r({outputTensor.GetDims(), outputTensor.GetType(),
                    outputTensor.GetLayout()}, {outputTensor.GetDims(),
                    Element::TypeT::F32, outputTensor.GetStandardABXLayout()});
                r.Run(diffDst, tmp);
                *diffDstExec = tmp;
        } else if (outputTensor.GetLayout() != outputTensor.GetStandardABXLayout()) {
            void *tmp = Service::AlignedAlloc(outputTensor.GetTotalTensorSize() * outputTensor.GetType().GetSize());
            ReorderLayer r({outputTensor.GetDims(), outputTensor.GetType(),
                outputTensor.GetLayout()}, {outputTensor.GetDims(),
                outputTensor.GetType(), outputTensor.GetStandardABXLayout()});
            r.Run(diffDst, tmp);
            *diffDstExec = tmp;
        }
        if (isInAndOutF16 && is920BplantForm) {
                void *tmp = Service::AlignedAlloc(inputTensor.GetTotalTensorSize() * sizeof(float));
                ReorderLayer r({inputTensor.GetDims(), inputTensor.GetType(),
                    inputTensor.GetLayout()}, {inputTensor.GetDims(),
                    Element::TypeT::F32, inputTensor.GetStandardABXLayout()});
                r.Run(src, tmp);
                *srcExec = tmp;
        } else if (inputTensor.GetLayout() != inputTensor.GetStandardABXLayout()) {
            void *tmp = Service::AlignedAlloc(inputTensor.GetTotalTensorSize() * inputTensor.GetType().GetSize());
            ReorderLayer r({inputTensor.GetDims(), inputTensor.GetType(),
                inputTensor.GetLayout()}, {inputTensor.GetDims(),
                inputTensor.GetType(), inputTensor.GetStandardABXLayout()});
            r.Run(src, tmp);
            *srcExec = tmp;
        }
        if (isInAndOutF16 && is920BplantForm) {
                *diffWeiExec = Service::AlignedAlloc(kernelTensor.GetTotalTensorSize() * sizeof(float));
        } else if (kernelTensor.GetLayout() != kernelTensor.GetStandardABXLayout()) {
            *diffWeiExec = Service::AlignedAlloc(kernelTensor.GetTotalTensorSize() * kernelTensor.GetType().GetSize());
        }
        if ((diffBias != nullptr) && (diffBiasTensor.GetType() != Element::TypeT::F32)) {
            *diffBiasExec = Service::AlignedAlloc(
                diffBiasTensor.GetTotalTensorSize() * Element::Type{Element::TypeT::F32}.GetSize());
        }
    }

    void AfterRunReorderLayer(const void **diffDstExec, const void **srcExec, void **diffWeiExec,
        void **diffBiasExec, void *diffWei, void *diffBias, bool isInAndOutF16) noexcept(false)
    {
        if (isInAndOutF16 && is920BplantForm) {
            Service::Deallocate(*diffDstExec);
        } else if (outputTensor.GetLayout() != outputTensor.GetStandardABXLayout()) {
            Service::Deallocate(*diffDstExec);
        }
        
        if (isInAndOutF16 && is920BplantForm) {
            Service::Deallocate(*srcExec);
        } else if (inputTensor.GetLayout() != inputTensor.GetStandardABXLayout()) {
            Service::Deallocate(*srcExec);
        }
 
        if (isInAndOutF16 && is920BplantForm) {
            ReorderLayer r({kernelTensor.GetDims(), Element::TypeT::F32,
                kernelTensor.GetStandardABXLayout()}, {kernelTensor.GetDims(),
                kernelTensor.GetType(), kernelTensor.GetLayout()});
                r.Run(*diffWeiExec, diffWei);
                Service::Deallocate(*diffWeiExec);
        } else if (kernelTensor.GetLayout() != kernelTensor.GetStandardABXLayout()) {
            ReorderLayer r({kernelTensor.GetDims(), kernelTensor.GetType(),
                kernelTensor.GetStandardABXLayout()}, {kernelTensor.GetDims(),
                kernelTensor.GetType(), kernelTensor.GetLayout()});
            r.Run(*diffWeiExec, diffWei);
            Service::Deallocate(*diffWeiExec);
        }
        if ((diffBias != nullptr) && (diffBiasTensor.GetType() != Element::TypeT::F32)) {
            ReorderLayer r({diffBiasTensor.GetDims(), Element::TypeT::F32,
                diffBiasTensor.GetLayout()}, {diffBiasTensor.GetDims(),
                diffBiasTensor.GetType(), diffBiasTensor.GetLayout()});
            r.Run(*diffBiasExec, diffBias);
            Service::Deallocate(*diffBiasExec);
        }
    }
};
} // Detail

// BWD Data

ConvolutionLayerBWDData::~ConvolutionLayerBWDData() noexcept {}

ConvolutionLayerBWDData::ConvolutionLayerBWDData(const TensorInfo &diffDst,
    const TensorInfo &weights, const TensorInfo &diffSrc,
    const Shape &strides, const Shape &dilates,
    const Shape &paddingL, const Shape &paddingR,
    ConvolutionAlgorithm alg) noexcept(false)
    : pImpl(new Detail::ConvolutionLayerBWDDataImpl(diffDst, weights, diffSrc,
                                                    strides, dilates, paddingL, paddingR, alg)) {}

ConvolutionLayerBWDData::ConvolutionLayerBWDData(const TensorInfo &diffDst,
    const TensorInfo &weights, const TensorInfo &diffSrc,
    const Shape &strides,
    const Shape &paddingL, const Shape &paddingR,
    ConvolutionAlgorithm alg) noexcept(false)
    : pImpl(new Detail::ConvolutionLayerBWDDataImpl(diffDst, weights, diffSrc,
                                                    strides, {0, 0, 0}, paddingL, paddingR, alg)) {}

ConvolutionLayerBWDData::ConvolutionLayerBWDData(const ConvolutionLayerBWDData &other) noexcept(false)
    : pImpl(new Detail::ConvolutionLayerBWDDataImpl{*(other.pImpl.get())}) {}

ConvolutionLayerBWDData::ConvolutionLayerBWDData(ConvolutionLayerBWDData &&other) noexcept
    : pImpl(std::move(other.pImpl)) {}

ConvolutionLayerBWDData& ConvolutionLayerBWDData::operator=(const ConvolutionLayerBWDData &other) noexcept(false)
{
    if (this != &other) {
        this->pImpl.reset(new Detail::ConvolutionLayerBWDDataImpl{*(other.pImpl.get())});
    }
    return *this;
}

ConvolutionLayerBWDData& ConvolutionLayerBWDData::operator=(ConvolutionLayerBWDData &&other) noexcept
{
    if (this != &other) {
        this->pImpl = std::move(other.pImpl);
    }
    return *this;
}

void ConvolutionLayerBWDData::Run(const void *diffDst, const void *wei,
    void *diffSrc) const noexcept(false)
{
    pImpl->Run(diffDst, wei, diffSrc);
}

Status ConvolutionLayerBWDData::ValidateInput(const TensorInfo &diffDst, const TensorInfo &weights,
    const TensorInfo &diffSrc,
    const Shape &strides, const Shape &dilates,
    const Shape &paddingL, const Shape &paddingR,
    ConvolutionAlgorithm alg) noexcept
{
    if (Detail::ConvolutionBase::ValidateAlgSizesConv(strides, dilates, paddingL, paddingR, alg,
        diffSrc, weights, diffDst) != Status::SUCCESS) {
        return Status::BAD_ARGUMENTS;
    }
    if (Detail::ConvolutionBase::ValidateDataShapes(diffSrc, weights, diffDst) != Status::SUCCESS) {
        return Status::BAD_ARGUMENTS;
    }
    if (Detail::ConvolutionBase::ValidateDataTypes(strides, dilates, paddingL, paddingR,
        diffDst, weights, diffSrc) != Status::SUCCESS) {
        return Status::BAD_ARGUMENTS;
    }
    return Status::SUCCESS;
}

// BWD Weights

ConvolutionLayerBWDWeights::~ConvolutionLayerBWDWeights() noexcept {}

ConvolutionLayerBWDWeights::ConvolutionLayerBWDWeights(const TensorInfo &diffDst, const TensorInfo &src,
    const TensorInfo &diffWeights, const TensorInfo &diffBias,
    const Shape &strides, const Shape &dilates,
    const Shape &paddingL, const Shape &paddingR,
    ConvolutionAlgorithm alg) noexcept(false)
    : pImpl(new Detail::ConvolutionLayerBWDWeightsImpl(diffDst, src, diffWeights, diffBias,
                                                       strides, dilates, paddingL, paddingR, alg)) {}

ConvolutionLayerBWDWeights::ConvolutionLayerBWDWeights(const TensorInfo &diffDst, const TensorInfo &src,
    const TensorInfo &diffWeights, const TensorInfo &diffBias,
    const Shape &strides,
    const Shape &paddingL, const Shape &paddingR,
    ConvolutionAlgorithm alg) noexcept(false)
    : pImpl(new Detail::ConvolutionLayerBWDWeightsImpl(diffDst, src, diffWeights, diffBias,
                                                       strides, {0, 0, 0}, paddingL, paddingR, alg)) {}

ConvolutionLayerBWDWeights::ConvolutionLayerBWDWeights(const ConvolutionLayerBWDWeights &other) noexcept(false)
    : pImpl(new Detail::ConvolutionLayerBWDWeightsImpl{*(other.pImpl.get())}) {}

ConvolutionLayerBWDWeights::ConvolutionLayerBWDWeights(ConvolutionLayerBWDWeights &&other) noexcept
    : pImpl(std::move(other.pImpl)) {}

ConvolutionLayerBWDWeights& ConvolutionLayerBWDWeights::operator=(const ConvolutionLayerBWDWeights &other)
    noexcept(false)
{
    if (this != &other) {
        this->pImpl.reset(new Detail::ConvolutionLayerBWDWeightsImpl{*(other.pImpl.get())});
    }
    return *this;
}

ConvolutionLayerBWDWeights& ConvolutionLayerBWDWeights::operator=(ConvolutionLayerBWDWeights &&other) noexcept
{
    if (this != &other) {
        this->pImpl = std::move(other.pImpl);
    }
    return *this;
}

void ConvolutionLayerBWDWeights::Run(const void *diffDst, const void *src,
    void *diffWei, void *diffBias) const noexcept(false)
{
    pImpl->Run(diffDst, src, diffWei, diffBias);
}

Status ConvolutionLayerBWDWeights::ValidateInput(const TensorInfo &diffDst, const TensorInfo &src,
    const TensorInfo &diffWeights, const TensorInfo &diffBias,
    const Shape &strides, const Shape &dilates,
    const Shape &paddingL, const Shape &paddingR,
    ConvolutionAlgorithm alg) noexcept
{
    if (Detail::ConvolutionBase::ValidateAlgSizesConv(strides, dilates, paddingL, paddingR, alg,
        src, diffWeights, diffDst) != Status::SUCCESS) {
        return Status::BAD_ARGUMENTS;
    }
    if (Detail::ConvolutionBase::ValidateDataShapes(src, diffWeights, diffDst, diffBias) != Status::SUCCESS) {
        return Status::BAD_ARGUMENTS;
    }
    if (Detail::ConvolutionBase::ValidateDataTypes(strides, dilates, paddingL, paddingR,
        diffDst, src, diffWeights, diffBias) != Status::SUCCESS) {
        return Status::BAD_ARGUMENTS;
    }
    return Status::SUCCESS;
}

} // KuDNN
