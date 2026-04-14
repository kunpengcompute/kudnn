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

#include "kudnn.hpp"
#include "internal_funcs.hpp"
#include "kernels/convolution/kudnn_conv_deconv_base.hpp"
#include "kernels/convolution/kudnn_conv_kernels.hpp"

namespace KuDNN {
namespace Detail {

static constexpr SizeType CONV_THREAD_THREADS = 200000;

class ConvolutionLayerFWDImpl final : public ConvolutionBase {
public:
    ConvolutionLayerFWDImpl(const TensorInfo &src, const TensorInfo &weights,
        const TensorInfo &dst, const TensorInfo &bias, const Shape &strides, const Shape &dilates,
        const Shape &paddingL, const Shape &paddingR, ConvolutionAlgorithm convFWDAlg) noexcept(false)
        : ConvolutionBase(src, weights, dst, strides, dilates, paddingL, paddingR),
        alg(convFWDAlg), inputTensor(src), kernelTensor(weights), outputTensor(dst),
        biaTensor(bias)
    {
        Service::ThrowOnStatus(ConvolutionLayerFWD::ValidateInput(
            src, weights, dst, bias, strides, dilates, paddingL, paddingR, alg), "Convolution layer");
        CheckIntMultOverflow(std::make_tuple(params.inputShape[NUM_1],
            params.kernelShape[NUM_0], params.kernelShape[NUM_1], params.kernelShape[NUM_2],
            params.outputShape[NUM_2], params.outputShape[NUM_3], params.outputShape[NUM_4]),
            std::make_tuple((params.inputShape[NUM_0] - 1),
                params.inputShape[NUM_2], params.inputShape[NUM_3], params.inputShape[NUM_4]),
            std::make_tuple((params.outputShape[NUM_0] - 1),
                params.outputShape[NUM_2], params.outputShape[NUM_3], params.outputShape[NUM_4]));

        SizeType totalSize = (params.inputShape[NUM_1] / params.groups) *
            params.kernelShape[NUM_2] * params.kernelShape[NUM_3] * params.kernelShape[NUM_4] *
            params.outputShape[NUM_2] * params.outputShape[NUM_3] * params.outputShape[NUM_4] *
            src.GetType().GetSize();
        totalSize += params.outputShape[NUM_2] * params.outputShape[NUM_3] *
            params.outputShape[NUM_4] * dst.GetType().GetSize();
        if (totalSize > Detail::BYTES_AVAILABLE) {
            throw Service::LogicError {"Convolution: exceeded available memory limit"};
        }

        SizeType inChannelPerGroup = params.inputShape[NUM_1] / params.groups;
        SizeType outChannelPerGroup = params.outputShape[NUM_1] / params.groups;
        SizeType m = outChannelPerGroup;
        SizeType n = params.outputShape[NUM_2] * params.outputShape[NUM_3] * params.outputShape[NUM_4];
        SizeType k = inChannelPerGroup * params.kernelShape[NUM_2] *
            params.kernelShape[NUM_3] * params.kernelShape[NUM_4];
        gemmSize = m * n * k;
        if (biaTensor.GetTotalTensorSize() != 0) {
            gemm.reset(new Gemm{
                {{m, k}, kernelType, Layout::ROW_MAJOR},
                {{k, n}, inputType, Layout::ROW_MAJOR},
                {{m, n}, outputType,
                    (((!params.isGroup) && (outputTensor.GetLayout() == outputTensor.GetStandardAXBLayout())) ?
                    Layout::COL_MAJOR : Layout::ROW_MAJOR)},
                {{m, 1}, biaTensor.GetType(), Layout::ROW_MAJOR}
            });
        } else {
            gemm.reset(new Gemm{
                {{m, k}, kernelType, Layout::ROW_MAJOR},
                {{k, n}, inputType, Layout::ROW_MAJOR},
                {{m, n}, outputType,
                    (((!params.isGroup) && (outputTensor.GetLayout() == outputTensor.GetStandardAXBLayout())) ?
                    Layout::COL_MAJOR : Layout::ROW_MAJOR)},
                {biaTensor}
            });
        }
    }

    ConvolutionLayerFWDImpl(const ConvolutionLayerFWDImpl &other) noexcept(false)
        : ConvolutionBase(other), alg(other.alg), inputTensor(other.inputTensor),
        kernelTensor(other.kernelTensor), outputTensor(other.outputTensor), biaTensor(other.biaTensor),
        gemm(new Gemm{*(other.gemm)}) {}

    ConvolutionLayerFWDImpl(ConvolutionLayerFWDImpl &&) = delete;

    ConvolutionLayerFWDImpl& operator=(const ConvolutionLayerFWDImpl &other) = delete;

    ConvolutionLayerFWDImpl &operator=(ConvolutionLayerFWDImpl &&) = delete;

    void Run(const void *src, const void *wei, void *dst, const void *bia) noexcept(false)
    {
        Service::Unused(alg); // At the moment, we always calculate using gemm
        if ((src == nullptr) || (wei == nullptr) || (dst == nullptr)) {
            throw Service::LogicError {"Convolution Layer: one of input matrices is nullptr"};
        }

        const void *srcExec = src;
        const void *weiExec = wei;
        void *dstExec = dst;
        const void *biasExec = bia;
        auto outputLayoutNotABX = [&]() {
            ReorderLayer r({outputTensor.GetDims(), outputTensor.GetType(),
                    outputTensor.GetStandardABXLayout()}, {outputTensor.GetDims(),
                    outputTensor.GetType(), outputTensor.GetLayout()});
                r.Run(dstExec, dst);
                Service::Deallocate(dstExec);
                params.outputLayout = outputTensor.GetLayout();
        };

        if (params.isGroup) {
            if (inputTensor.GetLayout() != inputTensor.GetStandardABXLayout()) {
                void *tmp = Service::AlignedAlloc(
                    inputTensor.GetTotalTensorSize() * inputTensor.GetType().GetSize());
                ReorderLayer r({inputTensor.GetDims(), inputTensor.GetType(),
                    inputTensor.GetLayout()}, {inputTensor.GetDims(),
                    inputTensor.GetType(), inputTensor.GetStandardABXLayout()});
                r.Run(src, tmp);
                srcExec = tmp;
                params.inputLayout = inputTensor.GetStandardABXLayout();
            }
            if (kernelTensor.GetLayout() != kernelTensor.GetStandardABXLayout()) {
                void *tmp = Service::AlignedAlloc(
                    kernelTensor.GetTotalTensorSize() * kernelTensor.GetType().GetSize());
                ReorderLayer r({kernelTensor.GetDims(), kernelTensor.GetType(),
                    kernelTensor.GetLayout()}, {kernelTensor.GetDims(),
                    kernelTensor.GetType(), kernelTensor.GetStandardABXLayout()});
                r.Run(wei, tmp);
                weiExec = tmp;
                params.kernelLayout = kernelTensor.GetStandardABXLayout();
            }
            if (outputTensor.GetLayout() != outputTensor.GetStandardABXLayout()) {
                dstExec = Service::AlignedAlloc(
                    outputTensor.GetTotalTensorSize() * outputTensor.GetType().GetSize());
                params.outputLayout = outputTensor.GetStandardABXLayout();
            }
        } else {
            if ((inputTensor.GetLayout() != inputTensor.GetStandardABXLayout()) &&
                (inputTensor.GetLayout() != inputTensor.GetStandardAXBLayout())) {
                void *tmp = Service::AlignedAlloc(
                    inputTensor.GetTotalTensorSize() * inputTensor.GetType().GetSize());
                ReorderLayer r({inputTensor.GetDims(), inputTensor.GetType(),
                    inputTensor.GetLayout()}, {inputTensor.GetDims(),
                    inputTensor.GetType(), inputTensor.GetStandardABXLayout()});
                r.Run(src, tmp);
                srcExec = tmp;
                params.inputLayout = inputTensor.GetStandardABXLayout();
            }
            if ((kernelTensor.GetLayout() != kernelTensor.GetStandardABXLayout()) &&
                (kernelTensor.GetLayout() != kernelTensor.GetStandardAXBLayout())) {
                void *tmp = Service::AlignedAlloc(
                    kernelTensor.GetTotalTensorSize() * kernelTensor.GetType().GetSize());
                ReorderLayer r({kernelTensor.GetDims(), kernelTensor.GetType(),
                    kernelTensor.GetLayout()}, {kernelTensor.GetDims(),
                    kernelTensor.GetType(), kernelTensor.GetStandardABXLayout()});
                r.Run(wei, tmp);
                weiExec = tmp;
                params.kernelLayout = kernelTensor.GetStandardABXLayout();
            }
            if ((outputTensor.GetLayout() != outputTensor.GetStandardABXLayout()) &&
                (outputTensor.GetLayout() != outputTensor.GetStandardAXBLayout())) {
                dstExec = Service::AlignedAlloc(
                    outputTensor.GetTotalTensorSize() * outputTensor.GetType().GetSize());
                params.outputLayout = outputTensor.GetStandardABXLayout();
            }
        }

        if (UseConv2DAcceleration<float>(Element::TypeT::F32)) {
            Conv2D<float, float>(params, static_cast<const float*>(srcExec), static_cast<const float*>(weiExec),
                static_cast<float*>(dstExec), static_cast<const float*>(biasExec));
        } else if (UseConv2DAcceleration<__fp16>(Element::TypeT::F16)) {
            Conv2D<__fp16, __fp16>(params, static_cast<const __fp16*>(srcExec), static_cast<const __fp16*>(weiExec),
                static_cast<__fp16*>(dstExec), static_cast<const __fp16*>(biasExec));
        } else {
            ConvParseType<true, false>(inputType, kernelType, outputType, *this, srcExec, weiExec, dstExec, biasExec);
        }

        if (params.isGroup) {
            if (inputTensor.GetLayout() != inputTensor.GetStandardABXLayout()) {
                Service::Deallocate(const_cast<void *>(srcExec));
                params.inputLayout = inputTensor.GetLayout();
            }
            if (kernelTensor.GetLayout() != kernelTensor.GetStandardABXLayout()) {
                Service::Deallocate(const_cast<void *>(weiExec));
                params.kernelLayout = kernelTensor.GetLayout();
            }
            if (outputTensor.GetLayout() != outputTensor.GetStandardABXLayout()) {
                outputLayoutNotABX();
            }
        } else {
            if ((inputTensor.GetLayout() != inputTensor.GetStandardABXLayout()) &&
                (inputTensor.GetLayout() != inputTensor.GetStandardAXBLayout())) {
                Service::Deallocate(const_cast<void *>(srcExec));
                params.inputLayout = inputTensor.GetLayout();
            }
            if ((kernelTensor.GetLayout() != kernelTensor.GetStandardABXLayout()) &&
                (kernelTensor.GetLayout() != kernelTensor.GetStandardAXBLayout())) {
                Service::Deallocate(const_cast<void *>(weiExec));
                params.kernelLayout = kernelTensor.GetLayout();
            }
            if ((outputTensor.GetLayout() != outputTensor.GetStandardABXLayout()) &&
                (outputTensor.GetLayout() != outputTensor.GetStandardAXBLayout())) {
                outputLayoutNotABX();
            }
        }
    }

    template <typename InType, typename KerType, typename OutType>
    void Evaluate(const void *src, const void *wei, void *dst, const void *bia) noexcept(false)
    {
        const InType *typedSrc = static_cast<const InType *>(src);
        const KerType *typedWei = static_cast<const KerType *>(wei);
        OutType *typedDst = static_cast<OutType *>(dst);

        SizeType batch = params.inputShape[NUM_0];
        SizeType inChannel = params.inputShape[NUM_1];
        SizeType outChannel = params.outputShape[NUM_1];

        SizeType inChannelPerGroup = inChannel / params.groups;
        SizeType outChannelPerGroup = outChannel / params.groups;

        int numThreads = std::min(GetOptNumThreads(), static_cast<SizeType>(Threading::GetMaxNumThreads()));

        std::vector<InType, Service::AlignedAllocator<InType>> colTmp(
            inChannelPerGroup * params.kernelShape[NUM_2] * params.kernelShape[NUM_3] * params.kernelShape[NUM_4] *
            params.outputShape[NUM_2] * params.outputShape[NUM_3] * params.outputShape[NUM_4],
            Vector::Convert<float, InType>::Do(0.0f));
        for (SizeType b = 0; b < batch; ++b) {
            for (SizeType g = 0; g < params.groups; ++g) {
                const InType *srcTmp = typedSrc +
                    b * inChannel * params.inputShape[NUM_2] *
                        params.inputShape[NUM_3] * params.inputShape[NUM_4] +
                    g * inChannelPerGroup * params.inputShape[NUM_2] *
                        params.inputShape[NUM_3] * params.inputShape[NUM_4];
                OutType *dstTmp = typedDst +
                    b * outChannel * params.outputShape[NUM_2] *
                        params.outputShape[NUM_3] * params.outputShape[NUM_4] +
                    g * outChannelPerGroup * params.outputShape[NUM_2] *
                        params.outputShape[NUM_3] * params.outputShape[NUM_4];
                const KerType *weiTmp = typedWei + g * inChannelPerGroup * outChannelPerGroup *
                    params.kernelShape[NUM_2] * params.kernelShape[NUM_3] * params.kernelShape[NUM_4];
                const void *biaPtr = nullptr;
                if (bia != nullptr) {
                    biaPtr = static_cast<const char *>(bia) + g * outChannelPerGroup * biaTensor.GetType().GetSize();
                }
                if ((params.kernelLayout == Layout::NCDHW) || (params.kernelLayout == Layout::NCHW) ||
                    (params.kernelLayout == Layout::ABC)) {
                    if ((params.inputLayout == Layout::NCDHW) || (params.inputLayout == Layout::NCHW) ||
                        (params.inputLayout == Layout::ABC)) {
                        Detail::Img2colNN<InType>::Do(srcTmp, inChannelPerGroup,
                            params.inputShape[NUM_2], params.inputShape[NUM_3], params.inputShape[NUM_4],
                            params.outputShape[NUM_2], params.outputShape[NUM_3], params.outputShape[NUM_4],
                            params.kernelShape[NUM_2], params.kernelShape[NUM_3], params.kernelShape[NUM_4],
                            params.paddingL[NUM_0], params.paddingL[NUM_1], params.paddingL[NUM_2],
                            params.stride[NUM_0], params.stride[NUM_1], params.stride[NUM_2],
                            params.dilate[NUM_0], params.dilate[NUM_1], params.dilate[NUM_2],
                            colTmp.data(), params.ndims, numThreads);
                    } else {
                        Detail::Img2colNT<InType>::Do(srcTmp, inChannelPerGroup,
                            params.inputShape[NUM_2], params.inputShape[NUM_3], params.inputShape[NUM_4],
                            params.outputShape[NUM_2], params.outputShape[NUM_3], params.outputShape[NUM_4],
                            params.kernelShape[NUM_2], params.kernelShape[NUM_3], params.kernelShape[NUM_4],
                            params.paddingL[NUM_0], params.paddingL[NUM_1], params.paddingL[NUM_2],
                            params.stride[NUM_0], params.stride[NUM_1], params.stride[NUM_2],
                            params.dilate[NUM_0], params.dilate[NUM_1], params.dilate[NUM_2],
                            colTmp.data(), params.ndims, numThreads);
                    }
                } else {
                    if ((params.inputLayout == Layout::NCDHW) || (params.inputLayout == Layout::NCHW) ||
                        (params.inputLayout == Layout::ABC)) {
                        Detail::Img2colTN<InType>::Do(srcTmp, inChannelPerGroup,
                            params.inputShape[NUM_2], params.inputShape[NUM_3], params.inputShape[NUM_4],
                            params.outputShape[NUM_2], params.outputShape[NUM_3], params.outputShape[NUM_4],
                            params.kernelShape[NUM_2], params.kernelShape[NUM_3], params.kernelShape[NUM_4],
                            params.paddingL[NUM_0], params.paddingL[NUM_1], params.paddingL[NUM_2],
                            params.stride[NUM_0], params.stride[NUM_1], params.stride[NUM_2],
                            params.dilate[NUM_0], params.dilate[NUM_1], params.dilate[NUM_2],
                            colTmp.data(), params.ndims, numThreads);
                    } else {
                        Detail::Img2colTT<InType>::Do(srcTmp, inChannelPerGroup,
                            params.inputShape[NUM_2], params.inputShape[NUM_3], params.inputShape[NUM_4],
                            params.outputShape[NUM_2], params.outputShape[NUM_3], params.outputShape[NUM_4],
                            params.kernelShape[NUM_2], params.kernelShape[NUM_3], params.kernelShape[NUM_4],
                            params.paddingL[NUM_0], params.paddingL[NUM_1], params.paddingL[NUM_2],
                            params.stride[NUM_0], params.stride[NUM_1], params.stride[NUM_2],
                            params.dilate[NUM_0], params.dilate[NUM_1], params.dilate[NUM_2],
                            colTmp.data(), params.ndims, numThreads);
                    }
                }
                gemm.get()->Run(weiTmp, colTmp.data(), dstTmp, biaPtr);
            }
        }
    }

private:
    ConvolutionAlgorithm alg;
    TensorInfo inputTensor;
    TensorInfo kernelTensor;
    TensorInfo outputTensor;
    TensorInfo biaTensor;
    SizeType gemmSize;
    std::unique_ptr<Gemm> gemm;

    SizeType GetOptNumThreads() const noexcept
    {
        if (gemmSize < CONV_THREAD_THREADS) {
            return 1;
        }
        return Threading::GetMaxNumThreads();
    }

    template <typename Comptype>
    bool UseConv2DAcceleration(Element::TypeT eleType) const noexcept
    {
        bool useConv2Dimpl = (inputType == eleType) &&
            (kernelType == eleType) &&
            (outputType == eleType) &&
            (biaTensor.GetType() == eleType) &&
            (!params.isGroup) && (params.ndims == NUM_4) &&
            (params.dilate[NUM_1] == NUM_1) && (params.dilate[NUM_2] == NUM_1) &&
            (params.kernelShape[NUM_3] == params.kernelShape[NUM_4]) &&
            (params.kernelShape[NUM_3] <= NUM_13) && (params.stride[NUM_2] <= NUM_5) &&
            (params.paddingR[NUM_2] < params.kernelShape[NUM_4]);
        if constexpr (std::is_same_v<Comptype, float>) {
            useConv2Dimpl = (useConv2Dimpl) &&
                (params.paddingR[NUM_2] < ((((ConvKernel<Comptype>::OwSize() - NUM_1) * params.stride[NUM_2] +
                params.kernelShape[NUM_4] + NUM_3) / NUM_4) * NUM_4));
        } else if constexpr (std::is_same_v<Comptype, __fp16>) {
            useConv2Dimpl = (useConv2Dimpl) &&
                (params.paddingR[NUM_2] < ((((ConvKernel<Comptype>::OwSize() - NUM_1) * params.stride[NUM_2] +
                params.kernelShape[NUM_4] + NUM_7) / NUM_8) * NUM_8));
        }
        return useConv2Dimpl;
    }
};

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
        gemm.reset(new Gemm{
            {{m, k}, kernelType, Layout::COL_MAJOR},
            {{k, n}, outputType, Layout::ROW_MAJOR},
            {{m, n}, Element::TypeT::F32, Layout::ROW_MAJOR}
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
        if ((inputTensor.GetLayout() != inputTensor.GetStandardABXLayout())
            || (inputType != Element::TypeT::F32)) {
            diffSrcExec = Service::AlignedAlloc(inputTensor.GetTotalTensorSize() * sizeof(float));
        }
        ConvParseType<true, true>(outputType, kernelType, Element::TypeT::F32, *this,
            diffDstExec, weiExec, diffSrcExec);

        if (outputTensor.GetLayout() != outputTensor.GetStandardABXLayout()) {
            Service::Deallocate(const_cast<void *>(diffDstExec));
        }
        if (kernelTensor.GetLayout() != kernelTensor.GetStandardABXLayout()) {
            Service::Deallocate(const_cast<void *>(weiExec));
        }
        if ((inputTensor.GetLayout() != inputTensor.GetStandardABXLayout())
            || (inputType != Element::TypeT::F32)) {
            ReorderLayer r({inputTensor.GetDims(),
                Element::TypeT::F32,
                inputTensor.GetStandardABXLayout()}, {inputTensor.GetDims(),
                inputType, inputTensor.GetLayout()});
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
        alg(convAlg), inputTensor(src), kernelTensor(diffWei), outputTensor(diffDst), diffBiasTensor(diffBias)
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
        gemm.reset(new Gemm{
            {{m, k}, outputType, Layout::ROW_MAJOR},
            {{k, n}, inputType, Layout::COL_MAJOR},
            {{m, n}, kernelType, Layout::ROW_MAJOR}
        });
    }

    ConvolutionLayerBWDWeightsImpl(const ConvolutionLayerBWDWeightsImpl &other) noexcept(false)
        : ConvolutionBase(other), alg(other.alg), inputTensor(other.inputTensor), kernelTensor(other.kernelTensor),
        outputTensor(other.outputTensor), diffBiasTensor(other.diffBiasTensor),
        gemm(new Gemm{*(other.gemm)}) {}

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

        BeforeRunReorderLayer(&diffDstExec, &srcExec, &diffWeiExec, &diffBiasExec, diffDst, src, diffBias);
        ConvParseType<false, false>(outputType, inputType, kernelType, *this,
            diffDstExec, srcExec, diffWeiExec, diffBiasExec);
        AfterRunReorderLayer(&diffDstExec, &srcExec, &diffWeiExec, &diffBiasExec, diffWei, diffBias);
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

                Detail::Img2colNN<KerType>::Do(srcTmp, inChannelPerGroup,
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

    void AfterRunReorderLayer(const void **diffDstExec, const void **srcExec, void **diffWeiExec,
        void **diffBiasExec, void *diffWei, void *diffBias) noexcept(false)
    {
        if (outputTensor.GetLayout() != outputTensor.GetStandardABXLayout()) {
            Service::Deallocate(const_cast<void *>(*diffDstExec));
        }
        if (inputTensor.GetLayout() != inputTensor.GetStandardABXLayout()) {
            Service::Deallocate(const_cast<void *>(*srcExec));
        }
        if (kernelTensor.GetLayout() != kernelTensor.GetStandardABXLayout()) {
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

// FWD

ConvolutionLayerFWD::~ConvolutionLayerFWD() noexcept {}

ConvolutionLayerFWD::ConvolutionLayerFWD(const TensorInfo &src, const TensorInfo &weights,
    const TensorInfo &dst, const TensorInfo &bias,
    const Shape &strides, const Shape &dilates,
    const Shape &paddingL, const Shape &paddingR,
    ConvolutionAlgorithm alg) noexcept(false)
    : pImpl(new Detail::ConvolutionLayerFWDImpl(src, weights, dst, bias, strides, dilates,
    paddingL, paddingR, alg)) {}

ConvolutionLayerFWD::ConvolutionLayerFWD(const TensorInfo &src, const TensorInfo &weights,
    const TensorInfo &dst, const TensorInfo &bias,
    const Shape &strides,
    const Shape &paddingL, const Shape &paddingR,
    ConvolutionAlgorithm alg) noexcept(false)
    : pImpl(new Detail::ConvolutionLayerFWDImpl(src, weights, dst, bias, strides, {0, 0, 0},
    paddingL, paddingR, alg)) {}

ConvolutionLayerFWD::ConvolutionLayerFWD(const ConvolutionLayerFWD &other) noexcept(false)
    : pImpl(new Detail::ConvolutionLayerFWDImpl{*(other.pImpl.get())}) {}

ConvolutionLayerFWD::ConvolutionLayerFWD(ConvolutionLayerFWD &&other) noexcept
    : pImpl(std::move(other.pImpl)) {}

ConvolutionLayerFWD& ConvolutionLayerFWD::operator=(const ConvolutionLayerFWD &other) noexcept(false)
{
    if (this != &other) {
        this->pImpl.reset(new Detail::ConvolutionLayerFWDImpl{*(other.pImpl.get())});
    }
    return *this;
}

ConvolutionLayerFWD& ConvolutionLayerFWD::operator=(ConvolutionLayerFWD &&other) noexcept
{
    if (this != &other) {
        this->pImpl = std::move(other.pImpl);
    }
    return *this;
}

void ConvolutionLayerFWD::Run(const void *src, const void *wei,
    void *dst, const void *bia) const noexcept(false)
{
    pImpl->Run(src, wei, dst, bia);
}

Status ConvolutionLayerFWD::ValidateInput(const TensorInfo &src, const TensorInfo &weights,
    const TensorInfo &dst, const TensorInfo &bias, const Shape &strides, const Shape &dilates,
    const Shape &paddingL, const Shape &paddingR,
    ConvolutionAlgorithm alg) noexcept
{
    if (Detail::ConvolutionBase::ValidateAlgSizesConv(strides, dilates, paddingL, paddingR, alg,
        src, weights, dst) != Status::SUCCESS) {
        return Status::BAD_ARGUMENTS;
    }
    if (Detail::ConvolutionBase::ValidateDataShapes(src, weights, dst, bias) != Status::SUCCESS) {
        return Status::BAD_ARGUMENTS;
    }
    if (Detail::ConvolutionBase::ValidateDataTypes(strides, dilates, paddingL, paddingR,
        src, weights, dst, bias) != Status::SUCCESS) {
        return Status::BAD_ARGUMENTS;
    }
    return Status::SUCCESS;
}

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
