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

#ifndef KUDNN_CONVOLUTION_BASE_IMPL_HPP
#define KUDNN_CONVOLUTION_BASE_IMPL_HPP
#include <climits>
#include <algorithm>
#include <memory>
#include <utility>
#include <numeric>
#include <functional>
#include "kudnn.hpp"
#include "internal_funcs.hpp"
#include "kernels/convolution/kudnn_conv_deconv_base.hpp"
#include "kernels/convolution/kudnn_primitive_conf.hpp"
#include "kernels/convolution/kudnn_conv_const.hpp"

namespace KuDNN {
namespace Detail {
class ConvolutionFWDImplBase {
public:
    ConvolutionFWDImplBase() noexcept(false) {}
    virtual void Run(const void *src, const void *wei, void *dst,
        const void *bias) noexcept(false)
    {
        Service::Unused(src);
        Service::Unused(wei);
        Service::Unused(dst);
        Service::Unused(bias);
    }
    virtual ~ConvolutionFWDImplBase() {}
};

inline Status ValidateInput(const TensorInfo &src, const TensorInfo &weights,
    const TensorInfo &dst, const TensorInfo &bias,
    const Shape &strides, const Shape &dilates,
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
        src, weights, dst, bias) != Status::SUCCESS || (dst.GetType() == Element::TypeT::F16 &&
        src.GetType() == Element::TypeT::F32 && weights.GetType() == Element::TypeT::F32) ||
        (dst.GetType() == Element::TypeT::BF16 && src.GetType() == Element::TypeT::F32 &&
        weights.GetType() == Element::TypeT::F32) || (dst.GetType() == Element::TypeT::F16 &&
        src.GetType() == Element::TypeT::BF16 && weights.GetType() == Element::TypeT::BF16) ||
        (dst.GetType() == Element::TypeT::BF16 && src.GetType() == Element::TypeT::F16 &&
        weights.GetType() == Element::TypeT::F16)) {
        return Status::BAD_ARGUMENTS;
    }
    return Status::SUCCESS;
}

class ConvolutionLayerFWDImpl final : public ConvolutionBase, public ConvolutionFWDImplBase {
public:
    ConvolutionLayerFWDImpl(const TensorInfo &src, const TensorInfo &weights,
        const TensorInfo &dst, const TensorInfo &bias, const Shape &strides, const Shape &dilates,
        const Shape &paddingL, const Shape &paddingR, ConvolutionAlgorithm convFWDAlg) noexcept(false)
        : ConvolutionBase(src, weights, dst, strides, dilates, paddingL, paddingR),
        alg(convFWDAlg), inputTensor(src), kernelTensor(weights), outputTensor(dst),
        biaTensor(bias), gemmSize(0), gemm(nullptr)
    {
        Service::ThrowOnStatus(ValidateInput(
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
            if (kernelType == Element::TypeT::S8 &&
                inputType == Element::TypeT::S8 &&
                outputType == Element::TypeT::BF16) {
                    outputType = Element::TypeT::F32;
            }
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

    void Run(const void *src, const void *wei, void *dst, const void *bia) noexcept(false) override
    {
        Service::Unused(alg); // At the moment, we always calculate using gemm
        if ((src == nullptr) || (wei == nullptr) || (dst == nullptr)) {
            throw Service::LogicError {"Convolution Layer: one of input matrices is nullptr"};
        }

        const void *srcExec = src;
        const void *weiExec = wei;
        void *dstExec = dst;
        const void *biasExec = bia;
        auto outputLayoutNotABX = [this, &dstExec, &dst]() {
            ReorderLayer r({outputTensor.GetDims(), outputTensor.GetType(),
                    outputTensor.GetStandardABXLayout()}, {outputTensor.GetDims(),
                    outputTensor.GetType(), outputTensor.GetLayout()});
                r.Run(dstExec, dst);
                Service::Deallocate(dstExec);
                params.outputLayout = outputTensor.GetLayout();
        };
        bool isInputABX = inputTensor.GetLayout() != inputTensor.GetStandardABXLayout();
        bool isInputAXB = inputTensor.GetLayout() != inputTensor.GetStandardAXBLayout();
        bool isKerABX = kernelTensor.GetLayout() != kernelTensor.GetStandardABXLayout();
        bool isKerAXB = kernelTensor.GetLayout() != kernelTensor.GetStandardAXBLayout();
        bool isOutputABX = outputTensor.GetLayout() != outputTensor.GetStandardABXLayout();
        bool isOutputAXB = outputTensor.GetLayout() != outputTensor.GetStandardAXBLayout();
        bool isInput = isInputABX && isInputAXB;
        bool isKer = isKerABX && isKerAXB;
        bool isOutput = isOutputABX && isOutputAXB;

        if (params.isGroup) {
            if (isInputABX) {
                void *tmp = Service::AlignedAlloc(
                    inputTensor.GetTotalTensorSize() * inputTensor.GetType().GetSize());
                ReorderLayer r({inputTensor.GetDims(), inputTensor.GetType(),
                    inputTensor.GetLayout()}, {inputTensor.GetDims(),
                    inputTensor.GetType(), inputTensor.GetStandardABXLayout()});
                r.Run(src, tmp);
                srcExec = tmp;
                params.inputLayout = inputTensor.GetStandardABXLayout();
            }
            if (isKerABX) {
                void *tmp = Service::AlignedAlloc(
                    kernelTensor.GetTotalTensorSize() * kernelTensor.GetType().GetSize());
                ReorderLayer r({kernelTensor.GetDims(), kernelTensor.GetType(),
                    kernelTensor.GetLayout()}, {kernelTensor.GetDims(),
                    kernelTensor.GetType(), kernelTensor.GetStandardABXLayout()});
                r.Run(wei, tmp);
                weiExec = tmp;
                params.kernelLayout = kernelTensor.GetStandardABXLayout();
            }
            if (isOutputABX) {
                dstExec = Service::AlignedAlloc(
                    outputTensor.GetTotalTensorSize() * outputTensor.GetType().GetSize());
                params.outputLayout = outputTensor.GetStandardABXLayout();
            }
        } else {
            if (isInput) {
                void *tmp = Service::AlignedAlloc(
                    inputTensor.GetTotalTensorSize() * inputTensor.GetType().GetSize());
                ReorderLayer r({inputTensor.GetDims(), inputTensor.GetType(),
                    inputTensor.GetLayout()}, {inputTensor.GetDims(),
                    inputTensor.GetType(), inputTensor.GetStandardABXLayout()});
                r.Run(src, tmp);
                srcExec = tmp;
                params.inputLayout = inputTensor.GetStandardABXLayout();
            }
            if (isKer) {
                void *tmp = Service::AlignedAlloc(
                    kernelTensor.GetTotalTensorSize() * kernelTensor.GetType().GetSize());
                ReorderLayer r({kernelTensor.GetDims(), kernelTensor.GetType(),
                    kernelTensor.GetLayout()}, {kernelTensor.GetDims(),
                    kernelTensor.GetType(), kernelTensor.GetStandardABXLayout()});
                r.Run(wei, tmp);
                weiExec = tmp;
                params.kernelLayout = kernelTensor.GetStandardABXLayout();
            }
            if (isOutput) {
                dstExec = Service::AlignedAlloc(
                    outputTensor.GetTotalTensorSize() * outputTensor.GetType().GetSize());
                params.outputLayout = outputTensor.GetStandardABXLayout();
            }
        }

        ConvParseType<true, false>(inputType, kernelType, outputType, *this, srcExec, weiExec, dstExec, biasExec);

        if (params.isGroup) {
            if (isInputABX) {
                Service::Deallocate(srcExec);
                params.inputLayout = inputTensor.GetLayout();
            }
            if (isKerABX) {
                Service::Deallocate(weiExec);
                params.kernelLayout = kernelTensor.GetLayout();
            }
            if (isOutputABX) {
                outputLayoutNotABX();
            }
        } else {
            if (isInput) {
                Service::Deallocate(srcExec);
                params.inputLayout = inputTensor.GetLayout();
            }
            if (isKer) {
                Service::Deallocate(weiExec);
                params.kernelLayout = kernelTensor.GetLayout();
            }
            if (isOutput) {
                outputLayoutNotABX();
            }
        }
    }

    template <typename InType, typename KerType, typename OutType>
    void Evaluate(const void *src, const void *wei, void *dst, const void *bia) noexcept(false)
    {
        void *tmpDst = nullptr;
        ReorderLayer r1({outputTensor.GetDims(), Element::TypeT::F32,
                    outputTensor.GetLayout()}, {outputTensor.GetDims(),
                    outputTensor.GetType(), outputTensor.GetLayout()});
        bool isKerAndInS8OutBf16 = kernelTensor.GetType() == Element::TypeT::S8 &&
            inputTensor.GetType() == Element::TypeT::S8 &&
            outputTensor.GetType() == Element::TypeT::BF16;
        bool isBiasTensorSizeZero = biaTensor.GetTotalTensorSize() == 0;
        if (isKerAndInS8OutBf16 && isBiasTensorSizeZero) {
            tmpDst = Service::AlignedAlloc(outputTensor.GetTotalTensorSize() * sizeof(float));
            ReorderLayer r2({outputTensor.GetDims(), outputTensor.GetType(),
                outputTensor.GetLayout()}, {outputTensor.GetDims(),
                Element::TypeT::F32, outputTensor.GetLayout()});
            r2.Run(dst, tmpDst);
        }
        const InType *typedSrc = static_cast<const InType *>(src);
        const KerType *typedWei = static_cast<const KerType *>(wei);
        OutType *typedDst = static_cast<OutType *>(dst);
        if (isKerAndInS8OutBf16 && isBiasTensorSizeZero) {
            typedDst = static_cast<OutType *>(tmpDst);
        }

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
                bool isKerABX = (params.kernelLayout == Layout::NCDHW) || (params.kernelLayout == Layout::NCHW) ||
                    (params.kernelLayout == Layout::ABC);
                bool isInputABX = (params.inputLayout == Layout::NCDHW) || (params.inputLayout == Layout::NCHW) ||
                        (params.inputLayout == Layout::ABC);
                if (isKerABX) {
                    if ((params.inputLayout == Layout::NCDHW) || (params.inputLayout == Layout::NCHW) ||
                        (params.inputLayout == Layout::ABC)) {
                        Detail::Img2col<InType, ImgLayout::NN>::Do(srcTmp, inChannelPerGroup,
                            params.inputShape[NUM_2], params.inputShape[NUM_3], params.inputShape[NUM_4],
                            params.outputShape[NUM_2], params.outputShape[NUM_3], params.outputShape[NUM_4],
                            params.kernelShape[NUM_2], params.kernelShape[NUM_3], params.kernelShape[NUM_4],
                            params.paddingL[NUM_0], params.paddingL[NUM_1], params.paddingL[NUM_2],
                            params.stride[NUM_0], params.stride[NUM_1], params.stride[NUM_2],
                            params.dilate[NUM_0], params.dilate[NUM_1], params.dilate[NUM_2],
                            colTmp.data(), params.ndims, numThreads);
                    } else {
                        Detail::Img2col<InType, ImgLayout::NT>::Do(srcTmp, inChannelPerGroup,
                            params.inputShape[NUM_2], params.inputShape[NUM_3], params.inputShape[NUM_4],
                            params.outputShape[NUM_2], params.outputShape[NUM_3], params.outputShape[NUM_4],
                            params.kernelShape[NUM_2], params.kernelShape[NUM_3], params.kernelShape[NUM_4],
                            params.paddingL[NUM_0], params.paddingL[NUM_1], params.paddingL[NUM_2],
                            params.stride[NUM_0], params.stride[NUM_1], params.stride[NUM_2],
                            params.dilate[NUM_0], params.dilate[NUM_1], params.dilate[NUM_2],
                            colTmp.data(), params.ndims, numThreads);
                    }
                } else {
                    if (isInputABX) {
                        Detail::Img2col<InType, ImgLayout::TN>::Do(srcTmp, inChannelPerGroup,
                            params.inputShape[NUM_2], params.inputShape[NUM_3], params.inputShape[NUM_4],
                            params.outputShape[NUM_2], params.outputShape[NUM_3], params.outputShape[NUM_4],
                            params.kernelShape[NUM_2], params.kernelShape[NUM_3], params.kernelShape[NUM_4],
                            params.paddingL[NUM_0], params.paddingL[NUM_1], params.paddingL[NUM_2],
                            params.stride[NUM_0], params.stride[NUM_1], params.stride[NUM_2],
                            params.dilate[NUM_0], params.dilate[NUM_1], params.dilate[NUM_2],
                            colTmp.data(), params.ndims, numThreads);
                    } else {
                        Detail::Img2col<InType, ImgLayout::TT>::Do(srcTmp, inChannelPerGroup,
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
        if (isKerAndInS8OutBf16 && isBiasTensorSizeZero) {
                r1.Run(tmpDst, dst);
        }
        if (tmpDst != nullptr) {
            Service::Deallocate(tmpDst);
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
};
} // Detail
} // KuDNN

#endif