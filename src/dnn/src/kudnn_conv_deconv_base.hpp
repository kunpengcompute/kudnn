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

#ifndef KUDNN_CONVOLUTION_BASE_HPP
#define KUDNN_CONVOLUTION_BASE_HPP

#include "kudnn_conv_deconv_base_common.hpp"

namespace KuDNN {
namespace Detail {

constexpr SizeType BYTES_AVAILABLE = 429496729600ULL;

struct ConvolutionParams {
    bool isGroup;
    SizeType groups;
    SizeType ndims;
    std::vector<IntType, Service::AlignedAllocator<IntType>> inputShape;
    std::vector<IntType, Service::AlignedAllocator<IntType>> kernelShape;
    std::vector<IntType, Service::AlignedAllocator<IntType>> outputShape;
    std::vector<IntType, Service::AlignedAllocator<IntType>> stride;
    std::vector<IntType, Service::AlignedAllocator<IntType>> paddingL;
    std::vector<IntType, Service::AlignedAllocator<IntType>> paddingR;
    std::vector<IntType, Service::AlignedAllocator<IntType>> dilate;
    Layout inputLayout;
    Layout kernelLayout;
    Layout outputLayout;
};

class ConvolutionBase {
public:
    using SizeType = ::KuDNN::SizeType;
    ConvolutionBase(const TensorInfo &inputTensor, const TensorInfo &kernelTensor,
        const TensorInfo &outputTensor, const Shape& strides, const Shape& dilates,
        const Shape& paddingL, const Shape& paddingR)
        : inputType(inputTensor.GetType()), kernelType(kernelTensor.GetType()), outputType(outputTensor.GetType())
    {
        params.isGroup = inputTensor.GetNumDims() != kernelTensor.GetNumDims();
        params.groups = params.isGroup ? kernelTensor.GetDims()[0] : 1;
        params.ndims = inputTensor.GetNumDims();
        params.inputShape.resize(NUM_5, 1);
        params.kernelShape.resize(NUM_5, 1);
        params.outputShape.resize(NUM_5, 1);
        for (SizeType i = 0; i < NUM_2; ++i) {
            params.inputShape[i] = inputTensor.GetDims()[i];
            params.kernelShape[i] = kernelTensor.GetDims()[(params.isGroup ? (i + 1) : i)];
            params.outputShape[i] = outputTensor.GetDims()[i];
        }
        for (SizeType i = (NUM_7 - params.ndims), idx = IDX_2; i < NUM_5; ++i, ++idx) {
            params.inputShape[i] = inputTensor.GetDims()[idx];
            params.kernelShape[i] = kernelTensor.GetDims()[(params.isGroup ? (idx + 1) : idx)];
            params.outputShape[i] = outputTensor.GetDims()[idx];
        }
        params.stride.resize(NUM_3, 1);
        params.paddingL.resize(NUM_3, 0);
        params.paddingR.resize(NUM_3, 0);
        params.dilate.resize(NUM_3, 1);
        for (SizeType i = (NUM_5 - params.ndims), idx = 0; i < NUM_3; ++i, ++idx) {
            params.stride[i] = strides[idx];
            params.paddingL[i] = paddingL[idx];
            params.paddingR[i] = paddingR[idx];
            params.dilate[i] = dilates[idx] + 1;
        }
        params.inputLayout = inputTensor.GetLayout();
        params.kernelLayout = kernelTensor.GetLayout();
        params.outputLayout = outputTensor.GetLayout();
    }

    virtual ~ConvolutionBase() = default;

    static Status ValidateStridesDilatesPadding(const Shape &argStrides, const Shape &argDilates,
        const Shape &argPaddingL, const Shape &argPaddingR)
    {
        for (SizeType i = 0; i < argStrides.GetNumDims(); ++i) {
            if (argStrides[i] == 0) {
                return Status::BAD_ARGUMENTS;
            }
        }
        if (argStrides.GetNumDims() < DIM_1 || argStrides.GetNumDims() > DIM_3) {
            return Status::BAD_ARGUMENTS;
        }
        if (argPaddingL.GetNumDims() < DIM_1 || argPaddingL.GetNumDims() > DIM_3) {
            return Status::BAD_ARGUMENTS;
        }
        if (argPaddingR.GetNumDims() < DIM_1 || argPaddingR.GetNumDims() > DIM_3) {
            return Status::BAD_ARGUMENTS;
        }
        if (argDilates.GetNumDims() < DIM_1 || argDilates.GetNumDims() > DIM_3) {
            return Status::BAD_ARGUMENTS;
        }
        return Status::SUCCESS;
    }

    static Status ValidateAlgSizesConv(const Shape &argStrides, const Shape &argDilates, const Shape &argPaddingL,
        const Shape &argPaddingR, ConvolutionAlgorithm argAlg, const TensorInfo &src, const TensorInfo &wei,
        const TensorInfo &dst)
    {
        if (argAlg == ConvolutionAlgorithm::UNIMPLEMENTED) {
            return Status::BAD_ARGUMENTS;
        }
        if (argAlg == ConvolutionAlgorithm::WINOGRAD) {
            return Status::UNSUPPORTED;
        }
        bool gConv = src.GetNumDims() != wei.GetNumDims();
        SizeType numDims = src.GetNumDims();
        SizeType ow = ComputeOutConvolution(src.GetDims()[numDims - 1],
            wei.GetDims()[(gConv ? numDims : (numDims - IDX_1))],
            argStrides[numDims - DIM_3], argPaddingL[numDims - DIM_3],
            argPaddingR[numDims - DIM_3], argDilates[numDims - DIM_3]);
        if (dst.GetDims()[numDims - 1] != ow) {
            return Status::BAD_ARGUMENTS;
        }
        if (numDims >= DIM_4) {
            SizeType oh = ComputeOutConvolution(src.GetDims()[numDims - 2],
                wei.GetDims()[(gConv ? (numDims - IDX_1) : (numDims - IDX_2))],
                argStrides[numDims - DIM_4], argPaddingL[numDims - DIM_4],
                argPaddingR[numDims - DIM_4], argDilates[numDims - DIM_4]);
            if (dst.GetDims()[numDims - DIM_2] != oh) {
                return Status::BAD_ARGUMENTS;
            }
        }
        if (numDims == DIM_5) {
            SizeType od = ComputeOutConvolution(src.GetDims()[numDims - 3],
                wei.GetDims()[(gConv ? (numDims - IDX_2) : (numDims - IDX_3))],
                argStrides[numDims - DIM_5], argPaddingL[numDims - DIM_5],
                argPaddingR[numDims - DIM_5], argDilates[numDims - DIM_5]);
            if (dst.GetDims()[numDims - DIM_3] != od) {
                return Status::BAD_ARGUMENTS;
            }
        }
        return Status::SUCCESS;
    }

    template<typename... Args>
    static Status ValidateDataShapes(const TensorInfo &src, const TensorInfo &wei, const TensorInfo &dst,
        Args&& ... args)
    {
        static_assert(sizeof...(args) <= 1, "This function supports at most one additional arg");
        bool gConv = src.GetNumDims() != wei.GetNumDims();
        if (src.GetNumDims() != dst.GetNumDims()) {
            return Status::BAD_ARGUMENTS;
        }
        if (gConv) {
            if ((wei.GetDims()[IDX_0] * wei.GetDims()[IDX_1]) != dst.GetDims()[IDX_1]) {
                return Status::BAD_ARGUMENTS;
            }
            if ((wei.GetDims()[IDX_0] * wei.GetDims()[IDX_2]) != src.GetDims()[IDX_1]) {
                return Status::BAD_ARGUMENTS;
            }
        } else {
            if (wei.GetDims()[IDX_0] != dst.GetDims()[IDX_1]) {
                return Status::BAD_ARGUMENTS;
            }
            if (wei.GetDims()[IDX_1] != src.GetDims()[IDX_1]) {
                return Status::BAD_ARGUMENTS;
            }
        }
        if (src.GetDims()[IDX_0] != dst.GetDims()[IDX_0]) {
            return Status::BAD_ARGUMENTS;
        }
        if constexpr (sizeof...(args) == 1) {
            std::tuple<Args...> additionalArgs(args...);
            auto bias = std::get<0>(additionalArgs);
            if (bias.GetNumDims() != DIM_1) {
                return Status::BAD_ARGUMENTS;
            }
            if ((bias.GetTotalTensorSize() != 0) && (bias.GetDims()[IDX_0] != dst.GetDims()[IDX_1])) {
                return Status::BAD_ARGUMENTS;
            }
        }
        return Status::SUCCESS;
    }

    template<typename... Args>
    static Status ValidateDataTypes(const Shape &argStrides, const Shape &argDilates, const Shape &argPaddingL,
        const Shape &argPaddingR, const TensorInfo &in1, const TensorInfo &in2,
        const TensorInfo &out, Args... args)
    {
        static_assert(sizeof...(args) <= 1, "This function supports at most one additional arg");
        if (ValidateStridesDilatesPadding(argStrides, argDilates, argPaddingL, argPaddingR) != Status::SUCCESS) {
            return Status::BAD_ARGUMENTS;
        }
        if (IsAnyNoDense(in1, in2, out, args...)) {
            return Status::UNSUPPORTED;
        }
        return Status::SUCCESS;
    }

protected:
    Element::Type inputType;
    Element::Type kernelType;
    Element::Type outputType;
    ConvolutionParams params;

    template <typename InType, typename OutType>
    const OutType* InitDiffDstTmpForBiasPtr(const InType *diffDstTmp, SizeType size,
        std::vector<OutType, Service::AlignedAllocator<OutType>>& acc) const
    {
        if constexpr (std::is_same_v<InType, OutType>) {
            return diffDstTmp;
        } else {
            ConvertDataToFP32(diffDstTmp, acc.data(), size);
            return acc.data();
        }
    }

    template <typename... Args>
    void CheckIntMultOverflow(Args&&... argGroup) const
    {
        auto checkOverflow = [](auto&&... args) {
            return Service::WillIntMultOverflow(std::forward<decltype(args)>(args)...);
        };

        if ((... || std::apply(checkOverflow, argGroup))) {
            throw Service::LogicError {
                "Convolution/Deconvolution: specified data sizes caused integer overflow for multiplications"};
        }
    }

    template<typename... Args>
    static bool IsAnyNoDense(Args... args)
    {
        return ((!args.IsDense() || ...));
    }

    static SizeType ComputeOutConvolution(SizeType i, SizeType k, SizeType s, SizeType pl, SizeType pr, SizeType d)
    {
        return (i - ((k - 1) * (d + 1) + 1) + (pl + pr)) / s + 1;
    }

    static SizeType ComputeOutDeconvolution(SizeType i, SizeType k, SizeType s, SizeType pl, SizeType pr, SizeType d)
    {
        return (i - 1) * s + (k - 1) * (d + 1) - (pl + pr) + DIM_1;
    }
};

template <typename InType, typename KerType, typename OutType, typename Runnable, typename ... Args>
inline void ConvParseType(Runnable &&runnable, Args && ... args)
    noexcept(noexcept(std::declval<Runnable>().template Evaluate<InType, KerType, OutType>(args ...)))
{
    runnable.template Evaluate<InType, KerType, OutType>(args ...);
}

template <typename InType, typename KerType, typename Runnable, typename ... Args>
inline void ConvParseType(Element::TypeT outType, Runnable &&runnable,
    Args && ... args) noexcept(false)
{
    if (outType == Element::TypeT::F32) {
        ConvParseType<InType, KerType, float>(std::forward<Runnable>(runnable),
            std::forward<Args>(args) ...);
    } else if (outType == Element::TypeT::F16) {
        ConvParseType<InType, KerType, __fp16>(std::forward<Runnable>(runnable),
            std::forward<Args>(args) ...);
    } else if (outType == Element::TypeT::BF16) {
        ConvParseType<InType, KerType, __bf16>(std::forward<Runnable>(runnable),
            std::forward<Args>(args) ...);
    } else if (outType == Element::TypeT::S32) {
        ConvParseType<InType, KerType, std::int32_t>(std::forward<Runnable>(runnable),
            std::forward<Args>(args) ...);
    } else if (outType == Element::TypeT::S8) {
        ConvParseType<InType, KerType, std::int8_t>(std::forward<Runnable>(runnable),
            std::forward<Args>(args) ...);
    } else if (outType == Element::TypeT::U8) {
        ConvParseType<InType, KerType, std::uint8_t>(std::forward<Runnable>(runnable),
            std::forward<Args>(args) ...);
    } else {
        throw Service::Unsupported {"Unsupported output data type"};
    }
}

template <typename InType, typename Runnable, typename ... Args>
inline void ConvParseType(Element::TypeT kerType, Element::TypeT outType, Runnable &&runnable,
    Args && ... args) noexcept(false)
{
    if (kerType == Element::TypeT::F32) {
        ConvParseType<InType, float>(outType, std::forward<Runnable>(runnable),
            std::forward<Args>(args) ...);
    } else if (kerType == Element::TypeT::F16) {
        ConvParseType<InType, __fp16>(outType, std::forward<Runnable>(runnable),
            std::forward<Args>(args) ...);
    } else if (kerType == Element::TypeT::BF16) {
        ConvParseType<InType, __bf16>(outType, std::forward<Runnable>(runnable),
            std::forward<Args>(args) ...);
    } else if (kerType == Element::TypeT::S8) {
        ConvParseType<InType, std::int8_t>(outType, std::forward<Runnable>(runnable),
            std::forward<Args>(args) ...);
    } else if (kerType == Element::TypeT::U8) {
        ConvParseType<InType, std::uint8_t>(outType, std::forward<Runnable>(runnable),
            std::forward<Args>(args) ...);
    } else {
        throw Service::Unsupported {"Unsupported kernel data type"};
    }
}

template <typename Runnable, typename ... Args>
inline void ConvParseType(Element::TypeT inType, Element::TypeT kerType, Element::TypeT outType,
    Runnable &&runnable, Args && ... args) noexcept(false)
{
    if (inType == Element::TypeT::F32) {
        ConvParseType<float>(kerType, outType, std::forward<Runnable>(runnable),
            std::forward<Args>(args) ...);
    } else if (inType == Element::TypeT::F16) {
        ConvParseType<__fp16>(kerType, outType, std::forward<Runnable>(runnable),
            std::forward<Args>(args) ...);
    } else if (inType == Element::TypeT::BF16) {
        ConvParseType<__bf16>(kerType, outType, std::forward<Runnable>(runnable),
            std::forward<Args>(args) ...);
    } else if (inType == Element::TypeT::S8) {
        ConvParseType<std::int8_t>(kerType, outType, std::forward<Runnable>(runnable),
            std::forward<Args>(args) ...);
    } else if (inType == Element::TypeT::U8) {
        ConvParseType<std::uint8_t>(kerType, outType, std::forward<Runnable>(runnable),
            std::forward<Args>(args) ...);
    } else {
        throw Service::Unsupported {"Unsupported input data type"};
    }
}

} // Detail
} // KuDNN

#endif // KUDNN_CONVOLUTION_BASE_HPP
