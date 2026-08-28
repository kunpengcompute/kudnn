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

#include "kudnn.hpp"

#include "internal_funcs.hpp"
#include "kernels/kernel_funcs.h"
#include "threading_impl.hpp"

#include <unordered_map>

namespace KuDNN {

#ifdef KUDNNL_920F
static constexpr SizeType ELE_THREADS_1NUMA = 38;
static constexpr SizeType ELE_THREADS_2NUMA = 76;
static constexpr SizeType ELE_THREADS_3NUMA = 114;
static constexpr SizeType ELE_THREADS_4NUMA = 152;
#else
static constexpr SizeType ELE_THREADS_1NUMA = 40;
static constexpr SizeType ELE_THREADS_2NUMA = 80;
static constexpr SizeType ELE_THREADS_3NUMA = 120;
static constexpr SizeType ELE_THREADS_4NUMA = 160;
#endif

static constexpr SizeType OTHER_ELE_NUM_DEFAULT = 15000;

namespace Detail {

    const static std::unordered_map<ActivationFunction,
                                    std::tuple<void *, void *, void *, void *, void *, void *>> eltFuncFwdMap = {
        {ActivationFunction::ABS, std::make_tuple(reinterpret_cast<void *>(AbsLaFWDF32),
            reinterpret_cast<void *>(AbsLaFWDF16), reinterpret_cast<void *>(AbsLaFWDBF16),
            nullptr, nullptr, nullptr)},
        {ActivationFunction::EXP, std::make_tuple(reinterpret_cast<void *>(ExpLaVCFWDF32),
            reinterpret_cast<void *>(ExpLaVCFWDF16), reinterpret_cast<void *>(ExpLaVCFWDBF16),
            nullptr, nullptr, nullptr)},
        {ActivationFunction::GELU_ERF, std::make_tuple(reinterpret_cast<void *>(GeluErfLaVCFWDF32),
            reinterpret_cast<void *>(GeluErfLaVCFWDF16), reinterpret_cast<void *>(GeluErfLaVCFWDBF16),
            nullptr, nullptr, nullptr)},
        {ActivationFunction::GELU_TANH, std::make_tuple(reinterpret_cast<void *>(GeluTanhLaVCFWDF32),
            reinterpret_cast<void *>(GeluTanhLaVCFWDF16), reinterpret_cast<void *>(GeluTanhLaVCFWDBF16),
            nullptr, nullptr, nullptr)},
        {ActivationFunction::LOG, std::make_tuple(reinterpret_cast<void *>(LnLaVCFWDF32),
            reinterpret_cast<void *>(LnLaVCFWDF16), reinterpret_cast<void *>(LnLaVCFWDBF16),
            nullptr, nullptr, nullptr)},
        {ActivationFunction::SIGMOID, std::make_tuple(reinterpret_cast<void *>(SigmoidVCFWDF32),
            reinterpret_cast<void *>(SigmoidVCFWDF16), reinterpret_cast<void *>(SigmoidVCFWDBF16),
            nullptr, nullptr, nullptr)},
        {ActivationFunction::MISH, std::make_tuple(reinterpret_cast<void *>(MishLaVCFWDF32),
            reinterpret_cast<void *>(MishLaVCFWDF16), reinterpret_cast<void *>(MishLaVCFWDBF16),
            nullptr, nullptr, nullptr)},
        {ActivationFunction::ROUND, std::make_tuple(reinterpret_cast<void *>(NearbyintFWDF32),
            reinterpret_cast<void *>(NearbyintFWDF16), reinterpret_cast<void *>(NearbyintVCFWDBF16),
            nullptr, nullptr, nullptr)},
        {ActivationFunction::SQRT, std::make_tuple(reinterpret_cast<void *>(SqrtLaFWDF32),
            reinterpret_cast<void *>(SqrtLaFWDF16), reinterpret_cast<void *>(SqrtLaVCFWDBF16),
            nullptr, nullptr, nullptr)},
        {ActivationFunction::SQUARE, std::make_tuple(reinterpret_cast<void *>(SqrLaFWDF32),
            reinterpret_cast<void *>(SqrLaFWDF16), reinterpret_cast<void *>(SqrLaVCFWDBF16),
            nullptr, nullptr, nullptr)},
        {ActivationFunction::TANH, std::make_tuple(reinterpret_cast<void *>(TanhLaVCFWDF32),
            reinterpret_cast<void *>(TanhLaVCFWDF16), reinterpret_cast<void *>(TanhLaVCFWDBF16),
            nullptr, nullptr, nullptr)},
        {ActivationFunction::ELU, std::make_tuple(reinterpret_cast<void *>(EluLaVCFWDF32),
            reinterpret_cast<void *>(EluLaVCFWDF16), reinterpret_cast<void *>(EluLaVCFWDBF16),
            nullptr, nullptr, nullptr)},
        {ActivationFunction::RELU, std::make_tuple(reinterpret_cast<void *>(ReluVCFWDF32),
            reinterpret_cast<void *>(ReluVCFWDF16), reinterpret_cast<void *>(ReluVCFWDBF16),
            reinterpret_cast<void *>(ReluVCFWDS32), reinterpret_cast<void *>(ReluVCFWDS8),
            reinterpret_cast<void *>(ReluVCFWDU8))},
        {ActivationFunction::SRELU, std::make_tuple(reinterpret_cast<void *>(SoftReluLaVCFWDF32),
            reinterpret_cast<void *>(SoftReluLaVCFWDF16), reinterpret_cast<void *>(SoftReluLaVCFWDBF16),
            nullptr, nullptr, nullptr)},
        {ActivationFunction::SWISH, std::make_tuple(reinterpret_cast<void *>(SwishLaVCFWDF32),
            reinterpret_cast<void *>(SwishLaVCFWDF16), reinterpret_cast<void *>(SwishLaVCFWDBF16),
            nullptr, nullptr, nullptr)},
        {ActivationFunction::CLIP, std::make_tuple(reinterpret_cast<void *>(ClipFWDF32),
            reinterpret_cast<void *>(ClipFWDF16), reinterpret_cast<void *>(ClipVCFWDBF16),
            nullptr, nullptr, nullptr)},
        {ActivationFunction::CLIP_V2, std::make_tuple(reinterpret_cast<void *>(ClipV2VCFWDF32),
            reinterpret_cast<void *>(ClipV2FWDF16), reinterpret_cast<void *>(ClipV2VCFWDBF16),
            nullptr, nullptr, nullptr)},
        {ActivationFunction::HARDSIGMOID, std::make_tuple(reinterpret_cast<void *>(HardSigmoidLaVCFWDF32),
            reinterpret_cast<void *>(HardSigmoidLaVCFWDF16), reinterpret_cast<void *>(HardSigmoidLaVCFWDBF16),
            nullptr, nullptr, nullptr)},
        {ActivationFunction::HARDSWISH, std::make_tuple(reinterpret_cast<void *>(HardSwishVCFWDF32),
            reinterpret_cast<void *>(HardSwishVCFWDF16), reinterpret_cast<void *>(HardSwishVCFWDBF16),
            nullptr, nullptr, nullptr)},
        {ActivationFunction::LINEAR, std::make_tuple(reinterpret_cast<void *>(LinearVCF32),
            reinterpret_cast<void *>(LinearVCF16), reinterpret_cast<void *>(LinearVCBF16),
            reinterpret_cast<void *>(LinearVCFWDS32), reinterpret_cast<void *>(LinearVCFWDS8),
            reinterpret_cast<void *>(LinearVCFWDU8))},
        {ActivationFunction::POW, std::make_tuple(reinterpret_cast<void *>(PowxaLaVCFWDF32),
            reinterpret_cast<void *>(PowxaLaVCFWDF16), reinterpret_cast<void *>(PowxaLaVCFWDBF16),
            nullptr, nullptr, nullptr)},
    };

    const static std::unordered_map<ActivationFunction,
                                    std::tuple<SizeType, SizeType, SizeType, SizeType, SizeType>> thrdsOptFwdMap = {
#ifdef KUDNNL_920F
        {ActivationFunction::SIGMOID, std::make_tuple(20000, 10000, INFINITY, INFINITY, INFINITY)},
        {ActivationFunction::ELU, std::make_tuple(30000, 12000, INFINITY, INFINITY, INFINITY)},
#else
        {ActivationFunction::SIGMOID, std::make_tuple(2000, 1000, 1600000, INFINITY, INFINITY)},
        {ActivationFunction::ELU, std::make_tuple(2000, 1000, 800000, 4000000, 10000000)},
#endif
        {ActivationFunction::RELU, std::make_tuple(20000, 10000, 8000000, INFINITY, INFINITY)},
        {ActivationFunction::LINEAR, std::make_tuple(20000, 10000, 8000000, 16000000, INFINITY)},
        {ActivationFunction::ABS, std::make_tuple(20000, 10000, 8000000, 16000000, INFINITY)},
        {ActivationFunction::CLIP, std::make_tuple(20000, 10000, 8000000, 16000000, INFINITY)},
        {ActivationFunction::CLIP_V2, std::make_tuple(20000, 10000, 8000000, 16000000, INFINITY)},
        {ActivationFunction::EXP, std::make_tuple(2000, 1000, 1000000, 12000000, INFINITY)},
        {ActivationFunction::GELU_ERF, std::make_tuple(1000, 500, 1200000, 8000000, 8000000)},
        {ActivationFunction::GELU_TANH, std::make_tuple(500, 250, 500000, 4000000, 4000000)},
        {ActivationFunction::HARDSIGMOID, std::make_tuple(20000, 10000, 8000000, 16000000, INFINITY)},
        {ActivationFunction::HARDSWISH, std::make_tuple(16000, 5000, 8000000, 16000000, INFINITY)},
        {ActivationFunction::LOG, std::make_tuple(2000, 1000, 1000000, 8000000, 10000000)},
        {ActivationFunction::MISH, std::make_tuple(1000, 500, 500000, 5000000, 5000000)},
        {ActivationFunction::POW, std::make_tuple(16000, 5000, 8000000, 16000000, INFINITY)},
        {ActivationFunction::ROUND, std::make_tuple(20000, 10000, 8000000, 16000000, INFINITY)},
        {ActivationFunction::SRELU, std::make_tuple(500, 250, 500000, 4000000, 4000000)},
        {ActivationFunction::SQRT, std::make_tuple(5000, 2000, 4000000, INFINITY, INFINITY)},
        {ActivationFunction::SQUARE, std::make_tuple(20000, 10000, 8000000, 16000000, INFINITY)},
        {ActivationFunction::SWISH, std::make_tuple(1000, 500, 1200000, 8000000, 10000000)},
        {ActivationFunction::TANH, std::make_tuple(2000, 1000, 1600000, 20000000, INFINITY)},
    };

    const static std::unordered_map<ActivationFunction, std::tuple<void *, void *, void *>> eltFuncBwdMap = {
        {ActivationFunction::ABS, std::make_tuple(reinterpret_cast<void *>(AbsLaVCBWDF32),
            reinterpret_cast<void *>(AbsLaBWDF16), reinterpret_cast<void *>(AbsLaVCBWDBF16))},
        {ActivationFunction::EXP, std::make_tuple(reinterpret_cast<void *>(ExpLaVCBWDF32),
            reinterpret_cast<void *>(ExpLaVCBWDF16), reinterpret_cast<void *>(ExpLaVCBWDBF16))},
        {ActivationFunction::EXP_DST, std::make_tuple(reinterpret_cast<void *>(ExpLaDstVCBWDF32),
            reinterpret_cast<void *>(ExpLaDstVCBWDF16), reinterpret_cast<void *>(ExpLaDstVCBWDBF16))},
        {ActivationFunction::GELU_ERF, std::make_tuple(reinterpret_cast<void *>(GeluErfLaVCBWDF32),
            reinterpret_cast<void *>(GeluErfLaVCBWDF16), reinterpret_cast<void *>(GeluErfLaVCBWDBF16))},
        {ActivationFunction::GELU_TANH, std::make_tuple(reinterpret_cast<void *>(GeluTanhLaVCBWDF32),
            reinterpret_cast<void *>(GeluTanhLaVCBWDF16), reinterpret_cast<void *>(GeluTanhLaVCBWDBF16))},
        {ActivationFunction::LOG, std::make_tuple(reinterpret_cast<void *>(LnLaVCBWDF32),
            reinterpret_cast<void *>(LnLaVCBWDF16), reinterpret_cast<void *>(LnLaVCBWDBF16))},
        {ActivationFunction::LOGISTIC_DST, std::make_tuple(reinterpret_cast<void *>(LogisticDstVCBWDF32),
            reinterpret_cast<void *>(LogisticDstVCBWDF16), reinterpret_cast<void *>(LogisticDstVCBWDBF16))},
        {ActivationFunction::SIGMOID, std::make_tuple(reinterpret_cast<void *>(SigmoidVCBWDF32),
            reinterpret_cast<void *>(SigmoidVCBWDF16), reinterpret_cast<void *>(SigmoidVCBWDBF16))},
        {ActivationFunction::MISH, std::make_tuple(reinterpret_cast<void *>(MishLaVCBWDF32),
            reinterpret_cast<void *>(MishLaVCBWDF16), reinterpret_cast<void *>(MishLaVCBWDBF16))},
        {ActivationFunction::SQRT, std::make_tuple(reinterpret_cast<void *>(SqrtLaVCBWDF32),
            reinterpret_cast<void *>(SqrtLaVCBWDF16), reinterpret_cast<void *>(SqrtLaVCBWDBF16))},
        {ActivationFunction::SQRT_DST, std::make_tuple(reinterpret_cast<void *>(SqrtDstVCBWDF32),
            reinterpret_cast<void *>(SqrtDstVCBWDF16), reinterpret_cast<void *>(SqrtDstVCBWDBF16))},
        {ActivationFunction::SQUARE, std::make_tuple(reinterpret_cast<void *>(SqrLaBWDF32),
            reinterpret_cast<void *>(SqrLaVCBWDF16), reinterpret_cast<void *>(SqrLaVCBWDBF16))},
        {ActivationFunction::TANH, std::make_tuple(reinterpret_cast<void *>(TanhLaVCBWDF32),
            reinterpret_cast<void *>(TanhLaVCBWDF16), reinterpret_cast<void *>(TanhLaVCBWDBF16))},
        {ActivationFunction::TANH_DST, std::make_tuple(reinterpret_cast<void *>(TanhDstVCBWDF32),
            reinterpret_cast<void *>(TanhDstVCBWDF16), reinterpret_cast<void *>(TanhDstVCBWDBF16))},
        {ActivationFunction::ELU, std::make_tuple(reinterpret_cast<void *>(EluLaVCBWDF32),
            reinterpret_cast<void *>(EluLaVCBWDF16), reinterpret_cast<void *>(EluLaVCBWDBF16))},
        {ActivationFunction::ELU_DST, std::make_tuple(reinterpret_cast<void *>(EluDstVCBWDF32),
            reinterpret_cast<void *>(EluDstVCBWDF16), reinterpret_cast<void *>(EluDstVCBWDBF16))},
        {ActivationFunction::RELU, std::make_tuple(reinterpret_cast<void *>(ReluVCBWDF32),
            reinterpret_cast<void *>(ReluVCBWDF16), reinterpret_cast<void *>(ReluVCBWDBF16))},
        {ActivationFunction::RELU_DST, std::make_tuple(reinterpret_cast<void *>(ReluDstVCBWDF32),
            reinterpret_cast<void *>(ReluDstVCBWDF16), reinterpret_cast<void *>(ReluDstVCBWDBF16))},
        {ActivationFunction::SRELU, std::make_tuple(reinterpret_cast<void *>(SoftReluLaVCBWDF32),
            reinterpret_cast<void *>(SoftReluLaVCBWDF16), reinterpret_cast<void *>(SoftReluLaVCBWDBF16))},
        {ActivationFunction::SWISH, std::make_tuple(reinterpret_cast<void *>(SwishLaVCBWDF32),
            reinterpret_cast<void *>(SwishLaVCBWDF16), reinterpret_cast<void *>(SwishLaVCBWDBF16))},
        {ActivationFunction::CLIP, std::make_tuple(reinterpret_cast<void *>(ClipVCBWDF32),
            reinterpret_cast<void *>(ClipVCBWDF16), reinterpret_cast<void *>(ClipVCBWDBF16))},
        {ActivationFunction::CLIP_V2, std::make_tuple(reinterpret_cast<void *>(ClipV2VCBWDF32),
            reinterpret_cast<void *>(ClipV2VCBWDF16), reinterpret_cast<void *>(ClipV2VCBWDBF16))},
        {ActivationFunction::CLIP_V2_DST, std::make_tuple(reinterpret_cast<void *>(ClipV2DstVCBWDF32),
            reinterpret_cast<void *>(ClipV2DstVCBWDF16), reinterpret_cast<void *>(ClipV2DstVCBWDBF16))},
        {ActivationFunction::HARDSIGMOID, std::make_tuple(reinterpret_cast<void *>(HardSigmoidLaVCBWDF32),
            reinterpret_cast<void *>(HardSigmoidLaVCBWDF16), reinterpret_cast<void *>(HardSigmoidLaVCBWDBF16))},
        {ActivationFunction::HARDSWISH, std::make_tuple(reinterpret_cast<void *>(HardSwishVCBWDF32),
            reinterpret_cast<void *>(HardSwishVCBWDF16), reinterpret_cast<void *>(HardSwishVCBWDBF16))},
        {ActivationFunction::POW, std::make_tuple(reinterpret_cast<void *>(PowxaLaVCBWDF32),
            reinterpret_cast<void *>(PowxaLaVCBWDF16), reinterpret_cast<void *>(PowxaLaVCBWDBF16))},
    };

    const static std::unordered_map<ActivationFunction,
                                    std::tuple<SizeType, SizeType, SizeType, SizeType, SizeType>> thrdsOptBwdMap = {
#ifdef KUDNNL_920F
        {ActivationFunction::ELU, std::make_tuple(10000, 4000, INFINITY, INFINITY, INFINITY)},
        {ActivationFunction::HARDSIGMOID, std::make_tuple(30000, 16000, INFINITY, INFINITY, INFINITY)},
        {ActivationFunction::HARDSWISH, std::make_tuple(30000, 10000, INFINITY, INFINITY, INFINITY)},
        {ActivationFunction::SRELU, std::make_tuple(5000, 1600, INFINITY, INFINITY, INFINITY)},
        {ActivationFunction::SWISH, std::make_tuple(8000, 2000, INFINITY, INFINITY, INFINITY)},
        {ActivationFunction::TANH, std::make_tuple(5000, 2000, INFINITY, INFINITY, INFINITY)},
#else
        {ActivationFunction::ELU, std::make_tuple(2000, 1000, 2000000, 16000000, INFINITY)},
        {ActivationFunction::HARDSIGMOID, std::make_tuple(10000, 4000, 4000000, INFINITY, INFINITY)},
        {ActivationFunction::HARDSWISH, std::make_tuple(5000, 2000, 2000000, INFINITY, INFINITY)},
        {ActivationFunction::SRELU, std::make_tuple(1000, 500, 800000, 16000000, INFINITY)},
        {ActivationFunction::SWISH, std::make_tuple(1000, 500, 1000000, 16000000, INFINITY)},
        {ActivationFunction::TANH, std::make_tuple(1000, 500, 1000000, 16000000, INFINITY)},
#endif
        {ActivationFunction::SIGMOID, std::make_tuple(2000, 1000, 1000000, 16000000, INFINITY)},
        {ActivationFunction::RELU, std::make_tuple(20000, 10000, 5000000, 12000000, INFINITY)},
        {ActivationFunction::LINEAR, std::make_tuple(20000, 10000, 8000000, 20000000, INFINITY)},
        {ActivationFunction::ABS, std::make_tuple(16000, 5000, 5000000, 12000000, INFINITY)},
        {ActivationFunction::CLIP, std::make_tuple(20000, 10000, 5000000, 12000000, INFINITY)},
        {ActivationFunction::CLIP_V2, std::make_tuple(20000, 10000, 5000000, 12000000, INFINITY)},
        {ActivationFunction::EXP, std::make_tuple(2000, 1000, 2000000, 16000000, INFINITY)},
        {ActivationFunction::GELU_ERF, std::make_tuple(1000, 500, 800000, INFINITY, INFINITY)},
        {ActivationFunction::GELU_TANH, std::make_tuple(500, 250, 500000, INFINITY, INFINITY)},
        {ActivationFunction::LOG, std::make_tuple(2000, 2000, 4000000, 12000000, INFINITY)},
        {ActivationFunction::MISH, std::make_tuple(500, 250, 800000, INFINITY, INFINITY)},
        {ActivationFunction::POW, std::make_tuple(10000, 4000, 5000000, 12000000, INFINITY)},
        {ActivationFunction::SQRT, std::make_tuple(2000, 1000, 1600000, 12000000, INFINITY)},
        {ActivationFunction::SQUARE, std::make_tuple(16000, 5000, 5000000, INFINITY, INFINITY)},
        {ActivationFunction::CLIP_V2_DST, std::make_tuple(16000, 5000, 5000000, INFINITY, INFINITY)},
        {ActivationFunction::ELU_DST, std::make_tuple(20000, 10000, 5000000, INFINITY, INFINITY)},
        {ActivationFunction::EXP_DST, std::make_tuple(20000, 10000, 5000000, INFINITY, INFINITY)},
        {ActivationFunction::LOGISTIC_DST, std::make_tuple(16000, 5000, 5000000, INFINITY, INFINITY)},
        {ActivationFunction::RELU_DST, std::make_tuple(20000, 10000, 8000000, INFINITY, INFINITY)},
        {ActivationFunction::SQRT_DST, std::make_tuple(8000, 2500, 5000000, INFINITY, INFINITY)},
        {ActivationFunction::TANH_DST, std::make_tuple(16000, 5000, 5000000, INFINITY, INFINITY)},
    };

class ActivationLayerImplFWD final : public Parallelizable {
public:
    ActivationLayerImplFWD(const TensorInfo &srcInfo, const TensorInfo &dstInfo,
        ActivationFunction kind, float alpha, float beta) noexcept(false)
        : activationKind(kind), srcType(srcInfo.GetType()), dstType(dstInfo.GetType()),
        alphaInfo(alpha), betaInfo(beta)
    {
        Service::ThrowOnStatus(ActivationLayerFWD::ValidateInput(
            srcInfo, dstInfo, kind, alphaInfo, betaInfo), "Activation layer");
        reorderedDims = Service::BroadcastTo5D(Service::GetShapeAccordingToLayout(srcInfo.GetDims(),
            srcInfo.GetLayout()));
        reorderedStridesSrc = Service::BroadcastTo5D(Service::GetShapeAccordingToLayout(srcInfo.GetStrides(),
            srcInfo.GetLayout()));
        reorderedStridesDst  = Service::BroadcastTo5D(Service::GetShapeAccordingToLayout(dstInfo.GetStrides(),
            dstInfo.GetLayout()));
        Parallelizable::Initialize(GetOptNumThreads(), reorderedDims);
    }

    template<typename T, int Idx, typename FuncX, typename FuncAX, typename FuncABX>
    void RunSequential(const Shape &batchSplitDims, const void *src, void *dst) const noexcept(false)
    {
        if ((src == nullptr) || (dst == nullptr)) {
            return;
        }

        auto item = eltFuncFwdMap.find(activationKind);
        if (item == eltFuncFwdMap.end()) {
            return;
        }

        if (Service::OneOf(activationKind, ActivationFunction::ABS, ActivationFunction::EXP,
            ActivationFunction::GELU_ERF, ActivationFunction::GELU_TANH, ActivationFunction::LOG,
            ActivationFunction::SIGMOID, ActivationFunction::MISH, ActivationFunction::ROUND,
            ActivationFunction::SQRT, ActivationFunction::SQUARE, ActivationFunction::TANH)) {
            reinterpret_cast<FuncX>(std::get<Idx>(item->second))(
                static_cast<int>(batchSplitDims.GetTotalDimsSize()),
                static_cast<const T *>(src), static_cast<T *>(dst));
        } else if (Service::OneOf(activationKind, ActivationFunction::ELU,
            ActivationFunction::RELU, ActivationFunction::SRELU, ActivationFunction::SWISH)) {
            reinterpret_cast<FuncAX>(std::get<Idx>(item->second))(
                static_cast<int>(batchSplitDims.GetTotalDimsSize()), alphaInfo,
                static_cast<const T *>(src), static_cast<T *>(dst));
        } else if (Service::OneOf(activationKind, ActivationFunction::CLIP, ActivationFunction::CLIP_V2,
            ActivationFunction::HARDSIGMOID, ActivationFunction::HARDSWISH, ActivationFunction::LINEAR,
            ActivationFunction::POW)) {
            reinterpret_cast<FuncABX>(std::get<Idx>(item->second))(
                static_cast<int>(batchSplitDims.GetTotalDimsSize()), alphaInfo, betaInfo,
                static_cast<const T *>(src), static_cast<T *>(dst));
        }
    }

    void RunSequentialPostOp(const Shape &batchSplitDims, const void *src, void *dst) const noexcept(false)
    {
        auto item = eltFuncFwdMap.find(activationKind);
        if (item == eltFuncFwdMap.end()) {
            return;
        }

        const void *calSrc = src;
        void *calDst = dst;
        auto calSrcType = srcType;

        if ((calSrc == nullptr) || (calDst == nullptr)) {
            return;
        }

        void *tmp = Service::AlignedAlloc(batchSplitDims.GetTotalDimsSize() * sizeof(float));
        ReorderLayer rSrc({batchSplitDims, calSrcType, Layout::ABCDE},
            {batchSplitDims, Element::TypeT::F32, Layout::ABCDE});
        rSrc.Run(calSrc, tmp);
        calSrc = tmp;
        calDst = Service::AlignedAlloc(batchSplitDims.GetTotalDimsSize() * sizeof(float));
        calSrcType = Element::TypeT::F32;

        if (Service::OneOf(activationKind, ActivationFunction::ABS, ActivationFunction::EXP,
            ActivationFunction::GELU_ERF, ActivationFunction::GELU_TANH, ActivationFunction::LOG,
            ActivationFunction::SIGMOID, ActivationFunction::MISH, ActivationFunction::ROUND,
            ActivationFunction::SQRT, ActivationFunction::SQUARE, ActivationFunction::TANH)) {
            reinterpret_cast<EltwiseFwdFuncF32X>(std::get<0>(item->second))(
                static_cast<int>(batchSplitDims.GetTotalDimsSize()),
                static_cast<const float *>(calSrc), static_cast<float *>(calDst));
        } else if (Service::OneOf(activationKind, ActivationFunction::ELU,
            ActivationFunction::RELU, ActivationFunction::SRELU, ActivationFunction::SWISH)) {
            reinterpret_cast<EltwiseFwdFuncF32AX>(std::get<0>(item->second))(
                static_cast<int>(batchSplitDims.GetTotalDimsSize()), alphaInfo,
                static_cast<const float *>(calSrc), static_cast<float *>(calDst));
        } else if (Service::OneOf(activationKind, ActivationFunction::CLIP, ActivationFunction::CLIP_V2,
            ActivationFunction::HARDSIGMOID, ActivationFunction::HARDSWISH, ActivationFunction::LINEAR,
            ActivationFunction::POW)) {
            reinterpret_cast<EltwiseFwdFuncF32ABX>(std::get<0>(item->second))(
                static_cast<int>(batchSplitDims.GetTotalDimsSize()), alphaInfo, betaInfo,
                static_cast<const float *>(calSrc), static_cast<float *>(calDst));
        }

        Service::Deallocate(calSrc);

        ReorderLayer rDst({batchSplitDims, Element::TypeT::F32, Layout::ABCDE},
            {batchSplitDims, dstType, Layout::ABCDE});
        rDst.Run(calDst, dst);
        Service::Deallocate(calDst);
        return;
    }

    // single thread
    void RunSequential(const Shape &batchSplitDims, const void *src,
        void *dst) const noexcept(false)
    {
        if (srcType != dstType) {
            RunSequentialPostOp(batchSplitDims, static_cast<const std::byte *>(src), static_cast<std::byte *>(dst));
            return;
        }

        switch (srcType) {
            case Element::TypeT::F32:
                RunSequential<float, 0x0, EltwiseFwdFuncF32X, EltwiseFwdFuncF32AX, EltwiseFwdFuncF32ABX>(
                    batchSplitDims, src, dst);
                return;
            case Element::TypeT::F16:
                RunSequential<__fp16, 0x1, EltwiseFwdFuncF16X, EltwiseFwdFuncF16AX, EltwiseFwdFuncF16ABX>(
                    batchSplitDims, src, dst);
                return;
            case Element::TypeT::BF16:
                RunSequential<__bf16, 0x2, EltwiseFwdFuncBF16X, EltwiseFwdFuncBF16AX, EltwiseFwdFuncBF16ABX>(
                    batchSplitDims, src, dst);
                return;
            case Element::TypeT::S32:
                RunSequential<int32_t, 0x3, EltwiseFwdFuncS32X, EltwiseFwdFuncS32AX, EltwiseFwdFuncS32ABX>(
                    batchSplitDims, src, dst);
                return;
            case Element::TypeT::S8:
                RunSequential<int8_t, 0x4, EltwiseFwdFuncS8X, EltwiseFwdFuncS8AX, EltwiseFwdFuncS8ABX>(
                    batchSplitDims, src, dst);
                return;
            case Element::TypeT::U8:
                RunSequential<uint8_t, 0x5, EltwiseFwdFuncU8X, EltwiseFwdFuncU8AX, EltwiseFwdFuncU8ABX>(
                    batchSplitDims, src, dst);
            default:
                return;
        }
    }

    // multi threads
    void RunSequential(const Shape &batchSplitDims, const Shape &batchStrides, const void *src,
        void *dst) const noexcept(false)
    {
        SizeType strideSrc = 0;
        SizeType strideDst = 0;
        for (SizeType i = 0; i < batchStrides.GetNumDims(); ++i) {
            strideSrc += batchStrides[i] * reorderedStridesSrc[i] * srcType.GetSize();
            strideDst += batchStrides[i] * reorderedStridesDst[i] * dstType.GetSize();
        }

        if (srcType != dstType) {
            RunSequentialPostOp(batchSplitDims, static_cast<const std::byte *>(src) + strideSrc,
                                static_cast<std::byte *>(dst) + strideDst);
            return;
        }

        switch (srcType) {
            case Element::TypeT::F32:
                RunSequential<float, 0x0, EltwiseFwdFuncF32X, EltwiseFwdFuncF32AX, EltwiseFwdFuncF32ABX>(
                    batchSplitDims, static_cast<const std::byte *>(src) + strideSrc,
                    static_cast<std::byte *>(dst) + strideDst);
                return;
            case Element::TypeT::F16:
                RunSequential<__fp16, 0x1, EltwiseFwdFuncF16X, EltwiseFwdFuncF16AX, EltwiseFwdFuncF16ABX>(
                    batchSplitDims, static_cast<const std::byte *>(src) + strideSrc,
                    static_cast<std::byte *>(dst) + strideDst);
                return;
            case Element::TypeT::BF16:
                RunSequential<__bf16, 0x2, EltwiseFwdFuncBF16X, EltwiseFwdFuncBF16AX, EltwiseFwdFuncBF16ABX>(
                    batchSplitDims, static_cast<const std::byte *>(src) + strideSrc,
                    static_cast<std::byte *>(dst) + strideDst);
                return;
            case Element::TypeT::S32:
                RunSequential<int32_t, 0x3, EltwiseFwdFuncS32X, EltwiseFwdFuncS32AX, EltwiseFwdFuncS32ABX>(
                    batchSplitDims, static_cast<const std::byte *>(src) + strideSrc,
                    static_cast<std::byte *>(dst) + strideDst);
                return;
            case Element::TypeT::S8:
                RunSequential<int8_t, 0x4, EltwiseFwdFuncS8X, EltwiseFwdFuncS8AX, EltwiseFwdFuncS8ABX>(
                    batchSplitDims, static_cast<const std::byte *>(src) + strideSrc,
                    static_cast<std::byte *>(dst) + strideDst);
                return;
            case Element::TypeT::U8:
                RunSequential<uint8_t, 0x5, EltwiseFwdFuncU8X, EltwiseFwdFuncU8AX, EltwiseFwdFuncU8ABX>(
                    batchSplitDims, static_cast<const std::byte *>(src) + strideSrc,
                    static_cast<std::byte *>(dst) + strideDst);
                return;
            default:
                return;
        }
    }

    void Run(const void *src, void *dst) noexcept(false)
    {
        if ((src == nullptr) || (dst == nullptr)) {
            throw Service::LogicError {"Activation Layer: one of input matrices is nullptr"};
        }
        Threading::BatchThreader(*this, src, dst);
    }

private:
    ActivationFunction activationKind;
    Element::Type srcType;
    Element::Type dstType;
    float alphaInfo;
    float betaInfo;
    Shape reorderedDims;
    Shape reorderedStridesSrc;
    Shape reorderedStridesDst;

    SizeType GetOptNumThreads() noexcept
    {
        SizeType totalSize = reorderedDims.GetTotalDimsSize();

        auto item = thrdsOptFwdMap.find(activationKind);
        if (item != thrdsOptFwdMap.end()) {
            auto optimizer = item->second;
            if (totalSize <= std::get<0>(optimizer)) {
                return 1;
            } else if (totalSize <= std::get<0x2>(optimizer)) {
                return std::min((totalSize + std::get<1>(optimizer) - 1) / std::get<1>(optimizer), ELE_THREADS_1NUMA);
            } else if (totalSize <= std::get<0x3>(optimizer)) {
                return ELE_THREADS_2NUMA;
            } else if (totalSize <= std::get<0x4>(optimizer)) {
                return ELE_THREADS_3NUMA;
            }
            return ELE_THREADS_4NUMA;
        }
        return (totalSize + OTHER_ELE_NUM_DEFAULT - 1) / OTHER_ELE_NUM_DEFAULT;
    }
};

class ActivationLayerImplBWD final : public Parallelizable {
public:
    ActivationLayerImplBWD(const TensorInfo &dsInfo, const TensorInfo &ddInfo, const TensorInfo &srcInfo,
        ActivationFunction kind, float alpha, float beta) noexcept(false)
        : activationKind(kind), dsType(dsInfo.GetType()), ddType(ddInfo.GetType()), srcType(srcInfo.GetType()),
        alphaInfo(alpha), betaInfo(beta)
    {
        if (kind == ActivationFunction::LINEAR) {
            betaInfo = 0.0f;
        }
        Service::ThrowOnStatus(ActivationLayerBWD::ValidateInput(
            dsInfo, ddInfo, srcInfo, kind, alphaInfo, betaInfo), "Activation layer");
        reorderedDims = Service::BroadcastTo5D(Service::GetShapeAccordingToLayout(srcInfo.GetDims(),
            srcInfo.GetLayout()));
        reorderedStridesDs  = Service::BroadcastTo5D(Service::GetShapeAccordingToLayout(dsInfo.GetStrides(),
            dsInfo.GetLayout()));
        reorderedStridesDd  = Service::BroadcastTo5D(Service::GetShapeAccordingToLayout(ddInfo.GetStrides(),
            ddInfo.GetLayout()));
        reorderedStridesSrc = Service::BroadcastTo5D(Service::GetShapeAccordingToLayout(srcInfo.GetStrides(),
            srcInfo.GetLayout()));
        Parallelizable::Initialize(GetOptNumThreads(), reorderedDims);
    }

    template<typename T, int Idx, typename FuncX, typename FuncAX, typename FuncABX>
    void RunSequential(const Shape &batchSplitDims, void *ds,
        const void *dd, const void *src) const noexcept(false)
    {
        if (activationKind == ActivationFunction::LINEAR) {
            if (srcType == Element::TypeT::F32) {
                LinearVCF32(static_cast<int>(batchSplitDims.GetTotalDimsSize()), alphaInfo, betaInfo,
                    static_cast<const float *>(dd), static_cast<float *>(ds));
            } else if (srcType == Element::TypeT::F16) {
                LinearVCF16(static_cast<int>(batchSplitDims.GetTotalDimsSize()), alphaInfo, betaInfo,
                    static_cast<const __fp16 *>(dd), static_cast<__fp16 *>(ds));
            } else if (srcType == Element::TypeT::BF16) {
                LinearVCBF16(static_cast<int>(batchSplitDims.GetTotalDimsSize()), alphaInfo, betaInfo,
                    static_cast<const __bf16 *>(dd), static_cast<__bf16 *>(ds));
            }
        }

        auto item = eltFuncBwdMap.find(activationKind);
        if (item == eltFuncBwdMap.end()) {
            return;
        }

        if (Service::OneOf(activationKind, ActivationFunction::ABS, ActivationFunction::EXP,
            ActivationFunction::EXP_DST, ActivationFunction::GELU_ERF, ActivationFunction::GELU_TANH,
            ActivationFunction::LOG, ActivationFunction::LOGISTIC_DST, ActivationFunction::SIGMOID,
            ActivationFunction::MISH, ActivationFunction::SQRT, ActivationFunction::SQRT_DST,
            ActivationFunction::SQUARE, ActivationFunction::TANH, ActivationFunction::TANH_DST)) {
            reinterpret_cast<FuncX>(std::get<Idx>(item->second))(
                static_cast<int>(batchSplitDims.GetTotalDimsSize()),
                static_cast<const T *>(src), static_cast<const T *>(dd), static_cast<T *>(ds));
        } else if (Service::OneOf(activationKind, ActivationFunction::ELU, ActivationFunction::ELU_DST,
            ActivationFunction::RELU, ActivationFunction::RELU_DST, ActivationFunction::SRELU,
            ActivationFunction::SWISH)) {
            reinterpret_cast<FuncAX>(std::get<Idx>(item->second))(
                static_cast<int>(batchSplitDims.GetTotalDimsSize()), alphaInfo,
                static_cast<const T *>(src), static_cast<const T *>(dd), static_cast<T *>(ds));
        } else if (Service::OneOf(activationKind, ActivationFunction::CLIP, ActivationFunction::CLIP_V2,
            ActivationFunction::CLIP_V2_DST, ActivationFunction::HARDSIGMOID, ActivationFunction::HARDSWISH,
            ActivationFunction::POW)) {
            reinterpret_cast<FuncABX>(std::get<Idx>(item->second))(
                static_cast<int>(batchSplitDims.GetTotalDimsSize()), alphaInfo, betaInfo,
                static_cast<const T *>(src), static_cast<const T *>(dd), static_cast<T *>(ds));
        }
    }

    // single thread
    void RunSequential(const Shape &batchSplitDims, void *ds,
        const void *dd, const void *src) const noexcept(false)
    {
        if (srcType == Element::TypeT::F32) {
            RunSequential<float, 0x0, EltwiseBwdFuncF32X, EltwiseBwdFuncF32AX, EltwiseBwdFuncF32ABX>(batchSplitDims,
                ds, dd, src);
        } else if (srcType == Element::TypeT::F16) {
            RunSequential<__fp16, 0x1, EltwiseBwdFuncF16X, EltwiseBwdFuncF16AX, EltwiseBwdFuncF16ABX>(batchSplitDims,
                ds, dd, src);
        } else if (srcType == Element::TypeT::BF16) {
            RunSequential<__bf16, 0x2, EltwiseBwdFuncBF16X, EltwiseBwdFuncBF16AX, EltwiseBwdFuncBF16ABX>(batchSplitDims,
                ds, dd, src);
        }
    }

    // multi threads
    void RunSequential(const Shape &batchSplitDims, const Shape &batchStrides, void *ds,
        const void *dd, const void *src) const noexcept(false)
    {
        SizeType strideDs = 0;
        SizeType strideDd = 0;
        SizeType strideSrc = 0;
        for (SizeType i = 0; i < batchStrides.GetNumDims(); ++i) {
            strideDs += batchStrides[i] * reorderedStridesDs[i] * dsType.GetSize();
            strideDd += batchStrides[i] * reorderedStridesDd[i] * ddType.GetSize();
            strideSrc += batchStrides[i] * reorderedStridesSrc[i] * srcType.GetSize();
        }

        if (srcType == Element::TypeT::F32) {
            RunSequential<float, 0x0, EltwiseBwdFuncF32X, EltwiseBwdFuncF32AX, EltwiseBwdFuncF32ABX>(batchSplitDims,
                static_cast<std::byte *>(ds) + strideDs, static_cast<const std::byte *>(dd) + strideDd,
                static_cast<const std::byte *>(src) + strideSrc);
        } else if (srcType == Element::TypeT::F16) {
            RunSequential<__fp16, 0x1, EltwiseBwdFuncF16X, EltwiseBwdFuncF16AX, EltwiseBwdFuncF16ABX>(batchSplitDims,
                static_cast<std::byte *>(ds) + strideDs, static_cast<const std::byte *>(dd) + strideDd,
                static_cast<const std::byte *>(src) + strideSrc);
        } else if (srcType == Element::TypeT::BF16) {
            RunSequential<__bf16, 0x2, EltwiseBwdFuncBF16X, EltwiseBwdFuncBF16AX, EltwiseBwdFuncBF16ABX>(batchSplitDims,
                static_cast<std::byte *>(ds) + strideDs, static_cast<const std::byte *>(dd) + strideDd,
                static_cast<const std::byte *>(src) + strideSrc);
        }
    }

    void Run(void *ds, const void *dd, const void *src) noexcept(false)
    {
        if ((ds == nullptr) || (dd == nullptr) || (src == nullptr)) {
            throw Service::LogicError {"Activation Layer: one of input matrices is nullptr"};
        }
        Threading::BatchThreader(*this, ds, dd, src);
    }

private:
    ActivationFunction activationKind;
    Element::Type dsType;
    Element::Type ddType;
    Element::Type srcType;
    float alphaInfo;
    float betaInfo;
    Shape reorderedDims;
    Shape reorderedStridesDs;
    Shape reorderedStridesDd;
    Shape reorderedStridesSrc;

    SizeType GetOptNumThreads() noexcept
    {
        SizeType totalSize = reorderedDims.GetTotalDimsSize();

        auto item = thrdsOptBwdMap.find(activationKind);
        if (item != thrdsOptBwdMap.end()) {
            auto optimizer = item->second;
            if (totalSize <= std::get<0>(optimizer)) {
                return 1;
            } else if (totalSize <= std::get<0x2>(optimizer)) {
                return std::min((totalSize + std::get<1>(optimizer) - 1) / std::get<1>(optimizer), ELE_THREADS_1NUMA);
            } else if (totalSize <= std::get<0x3>(optimizer)) {
                return ELE_THREADS_2NUMA;
            } else if (totalSize <= std::get<0x4>(optimizer)) {
                return ELE_THREADS_3NUMA;
            }
            return ELE_THREADS_4NUMA;
        }
        return (totalSize + OTHER_ELE_NUM_DEFAULT - 1) / OTHER_ELE_NUM_DEFAULT;
    }
};

} // Detail

ActivationLayerFWD::~ActivationLayerFWD() noexcept {}

ActivationLayerFWD::ActivationLayerFWD(const TensorInfo &srcInfo, const TensorInfo &dstInfo,
    ActivationFunction kind, float alpha, float beta) noexcept(false)
    : pImpl(new Detail::ActivationLayerImplFWD(srcInfo, dstInfo, kind, alpha, beta)) {}

ActivationLayerFWD::ActivationLayerFWD(const ActivationLayerFWD &other) noexcept(false)
    : pImpl(new Detail::ActivationLayerImplFWD{*(other.pImpl.get())}) {}

ActivationLayerFWD::ActivationLayerFWD(ActivationLayerFWD &&other) noexcept
    : pImpl(std::move(other.pImpl)) {}

ActivationLayerFWD& ActivationLayerFWD::operator=(const ActivationLayerFWD &other) noexcept(false)
{
    if (this != &other) {
        this->pImpl.reset(new Detail::ActivationLayerImplFWD{*(other.pImpl.get())});
    }
    return *this;
}

ActivationLayerFWD& ActivationLayerFWD::operator=(ActivationLayerFWD &&other) noexcept
{
    if (this != &other) {
        this->pImpl = std::move(other.pImpl);
    }
    return *this;
}

void ActivationLayerFWD::Run(const void *src, void *dst) const noexcept(false)
{
    pImpl->Run(src, dst);
}

Status ActivationLayerFWD::ValidateInput(const TensorInfo &srcInfo, const TensorInfo &dstInfo,
    ActivationFunction kind, float alpha, float beta) noexcept
{
    Service::Unused(alpha, beta);

    if (srcInfo.GetLayout() != dstInfo.GetLayout()) {
        return Status::BAD_ARGUMENTS;
    }
    if (!(Service::OneOf(srcInfo.GetType(), Element::TypeT::F32, Element::TypeT::F16, Element::TypeT::BF16) ||
    (Service::OneOf(srcInfo.GetType(), Element::TypeT::S32, Element::TypeT::S8, Element::TypeT::U8) &&
    Service::OneOf(kind, ActivationFunction::RELU, ActivationFunction::LINEAR)))) {
        return Status::UNSUPPORTED;
    }
    if (!Service::OneOf(kind, ActivationFunction::ABS, ActivationFunction::CLIP, ActivationFunction::CLIP_V2,
        ActivationFunction::ELU, ActivationFunction::EXP, ActivationFunction::GELU_ERF,
        ActivationFunction::GELU_TANH, ActivationFunction::HARDSIGMOID, ActivationFunction::HARDSWISH,
        ActivationFunction::LINEAR, ActivationFunction::LOG, ActivationFunction::SIGMOID,
        ActivationFunction::MISH, ActivationFunction::POW, ActivationFunction::RELU, ActivationFunction::ROUND,
        ActivationFunction::SRELU, ActivationFunction::SQRT, ActivationFunction::SQUARE, ActivationFunction::SWISH,
        ActivationFunction::TANH)) {
        return Status::UNSUPPORTED;
    }
    if ((Service::OneOf(kind, ActivationFunction::CLIP, ActivationFunction::CLIP_V2) && (alpha > beta)) ||
        (Service::OneOf(kind, ActivationFunction::SRELU) && (alpha == 0))) {
        return Status::UNSUPPORTED;
    }
    if (kind == ActivationFunction::UNIMPLEMENTED) {
        return Status::UNSUPPORTED;
    }
    if (!srcInfo.IsDense()) {
        return Status::UNSUPPORTED;
    }
    if (srcInfo.GetTotalTensorSize() == 0) {
        return Status::UNSUPPORTED;
    }
    return Status::SUCCESS;
}


ActivationLayerBWD::~ActivationLayerBWD() noexcept {}

ActivationLayerBWD::ActivationLayerBWD(const TensorInfo &dsInfo, const TensorInfo &ddInfo,
    const TensorInfo &srcInfo, ActivationFunction kind, float alpha, float beta) noexcept(false)
    : pImpl(new Detail::ActivationLayerImplBWD(dsInfo, ddInfo, srcInfo, kind, alpha, beta)) {}

ActivationLayerBWD::ActivationLayerBWD(const ActivationLayerBWD &other) noexcept(false)
    : pImpl(new Detail::ActivationLayerImplBWD{*(other.pImpl.get())}) {}

ActivationLayerBWD::ActivationLayerBWD(ActivationLayerBWD &&other) noexcept
    : pImpl(std::move(other.pImpl)) {}

ActivationLayerBWD& ActivationLayerBWD::operator=(const ActivationLayerBWD &other) noexcept(false)
{
    if (this != &other) {
        this->pImpl.reset(new Detail::ActivationLayerImplBWD{*(other.pImpl.get())});
    }
    return *this;
}

ActivationLayerBWD& ActivationLayerBWD::operator=(ActivationLayerBWD &&other) noexcept
{
    if (this != &other) {
        this->pImpl = std::move(other.pImpl);
    }
    return *this;
}

void ActivationLayerBWD::Run(void *ds, const void *dd, const void *src) const noexcept(false)
{
    pImpl->Run(ds, dd, src);
}

Status ActivationLayerBWD::ValidateInput(const TensorInfo &dsInfo, const TensorInfo &ddInfo,
    const TensorInfo &srcInfo, ActivationFunction kind, float alpha, float beta) noexcept
{
    Service::Unused(alpha, beta);
    if (dsInfo != ddInfo || dsInfo != srcInfo) {
        return Status::BAD_ARGUMENTS;
    }
    if (!Service::OneOf(dsInfo.GetType(), Element::TypeT::F32, Element::TypeT::F16, Element::TypeT::BF16)) {
        return Status::UNSUPPORTED;
    }
    if (!Service::OneOf(kind, ActivationFunction::ABS, ActivationFunction::CLIP, ActivationFunction::CLIP_V2,
        ActivationFunction::CLIP_V2_DST, ActivationFunction::ELU, ActivationFunction::ELU_DST,
        ActivationFunction::EXP, ActivationFunction::EXP_DST, ActivationFunction::GELU_ERF,
        ActivationFunction::GELU_TANH, ActivationFunction::HARDSIGMOID, ActivationFunction::HARDSWISH,
        ActivationFunction::LINEAR, ActivationFunction::LOG, ActivationFunction::SIGMOID,
        ActivationFunction::LOGISTIC_DST, ActivationFunction::MISH, ActivationFunction::POW,
        ActivationFunction::RELU, ActivationFunction::RELU_DST, ActivationFunction::SRELU,
        ActivationFunction::SQRT, ActivationFunction::SQRT_DST, ActivationFunction::SQUARE,
        ActivationFunction::SWISH, ActivationFunction::TANH, ActivationFunction::TANH_DST)) {
        return Status::UNSUPPORTED;
    }
    // the list of functions with backward support will be extended
    if ((Service::OneOf(kind, ActivationFunction::CLIP, ActivationFunction::CLIP_V2, ActivationFunction::CLIP_V2_DST)
        && (alpha > beta)) || (Service::OneOf(kind, ActivationFunction::ELU_DST, ActivationFunction::RELU_DST)
        && (alpha < 0))) {
        return Status::UNSUPPORTED;
    }
    if (!dsInfo.IsDense()) {
        return Status::UNSUPPORTED;
    }
    if (srcInfo.GetTotalTensorSize() == 0) {
        return Status::UNSUPPORTED;
    }
    return Status::SUCCESS;
}

} // KuDNN
