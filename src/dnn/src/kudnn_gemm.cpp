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

#include <cstddef>
#include <memory>
#include <utility>

#include "kudnn.hpp"

#if defined(KUDNN_CPU_THREADING_RUNTIME) && (KUDNN_CPU_THREADING_RUNTIME == KUDNN_RUNTIME_THREADPOOL)
#include "threading_impl.hpp"
#endif

#include "internal_funcs.hpp"
#include "kernels/matmul_jit/kudnn_gemm_helpers.hpp"
#include "kernels/matmul_jit/kudnn_small_gemm.hpp"
#include "kudnn_jit.hpp"

namespace KuDNN {
namespace Detail {

class GemmImpl final : public Parallelizable {
public:
    GemmImpl(const TensorInfo &srcInfo, const TensorInfo &weiInfo, const TensorInfo &dstInfo,
        const TensorInfo &biaInfo, const Attributes &attributes) noexcept(false)
        : fastCall(), gemmImplInfo(srcInfo, weiInfo, dstInfo, biaInfo, attributes)
    {
        Service::ThrowOnStatus(Gemm::ValidateInput(srcInfo, weiInfo, dstInfo, biaInfo, attributes), "GEMM");
#if defined(KUDNN_CPU_THREADING_RUNTIME) && (KUDNN_CPU_THREADING_RUNTIME == KUDNN_RUNTIME_OMP)
        int maxNumThreads = Threading::GetMaxNumThreads();
#elif defined(KUDNN_CPU_THREADING_RUNTIME) && (KUDNN_CPU_THREADING_RUNTIME == KUDNN_RUNTIME_THREADPOOL)
        int maxNumThreads = Threading::GetThreadpoolNumThreads();
#endif
        fastCall = FindSmallGemmKernel(gemmImplInfo, maxNumThreads);
        if (!fastCall.kernelPtr) {
            impl = FindSolution(gemmImplInfo, maxNumThreads);
        }
    }

    void Run(const void *a, const void *b, void *c, const void *bias, const PostOpsDataPtrs &postOpsPtrs,
        float alpha, float beta) const noexcept(false)
    {
        if (fastCall.kernelPtr != nullptr) {
            if (fastCall.dstFlags == MatMulDstFlags::LOAD_C) {
                beta = 1.0f;
            }
            fastCall.kernelPtr(a, b, c, bias,
                fastCall.dstFlags, alpha, beta,
                fastCall.m, fastCall.n, fastCall.k,
                fastCall.srcStride, fastCall.weiStride, fastCall.dstStride, fastCall.biaStride);
        } else {
            RunClassic(a, b, c, bias, postOpsPtrs, alpha, beta);
        }
    }

private:
    void RunClassic(const void *a, const void *b, void *c, const void *bias, const PostOpsDataPtrs &postOpsPtrs,
        float alpha, float beta) const noexcept(false)
    {
        if ((a == nullptr) || (b == nullptr) || (c == nullptr) ||
            ((gemmImplInfo.biaInfo.GetTotalTensorSize() != 0) && (bias == nullptr))) {
            throw Service::LogicError {"Matmul Layer: one of input matrices is nullptr"};
        }
        if ((gemmImplInfo.biaInfo.GetTotalTensorSize() == 0) && (bias != nullptr)) {
            throw Service::LogicError {"Matmul Layer: bias isn't nullptr, but bias TensorInfo has 0 total size"};
        }
        auto &&po = gemmImplInfo.attributes.GetPostOps();
        if (po.Len() > 1) {
            throw Service::LogicError {"Matmul Layer: only 1 post-ops is supported"};
        }
        if (po.Len() > 0) {
            if ((po.Kind(0) == PrimitiveKind::SUM) && (beta != 0.0f)) {
                throw Service::LogicError {"Matmul Layer: beta scaler and sum post-ops are conflicting"};
            }
            if (po.Kind(0) == PrimitiveKind::SUM) {
                po.GetParamsSum(0, beta);
            }
        }
#if defined(KUDNN_CPU_THREADING_RUNTIME) && (KUDNN_CPU_THREADING_RUNTIME == KUDNN_RUNTIME_OMP)
        int maxNumThreads = Threading::GetMaxNumThreads();
#elif defined(KUDNN_CPU_THREADING_RUNTIME) && (KUDNN_CPU_THREADING_RUNTIME == KUDNN_RUNTIME_THREADPOOL)
        int maxNumThreads = Threading::GetThreadpoolNumThreads();
#endif
        {
            if (impl.expired()) {
                throw Service::LogicError {"Cached Matmul Impl was manully removed"};
            } else {
                std::shared_ptr<ImplBase<GemmInfo>> gemmImpl(impl);
                if (gemmImpl->GetNThreads() != maxNumThreads) {
                    impl = FindSolution(gemmImpl->GetTask(), maxNumThreads);
                    gemmImpl = impl.lock();
                }
                gemmImpl->Run({a, b, c, bias, postOpsPtrs, alpha, beta});
            }
        }
    }
    SmallGEMMParams fastCall;

private:
    const GemmInfo gemmImplInfo;
    mutable std::weak_ptr<ImplBase<GemmInfo>> impl;
};

} // Detail

Gemm::~Gemm() noexcept {}

// all public methods just call corresponding pImpl implemnentation
Gemm::Gemm(const TensorInfo &aInfo, const TensorInfo &bInfo, const TensorInfo &cInfo,
    const TensorInfo &biasInfo) noexcept(false)
    : pImpl(new Detail::GemmImpl(aInfo, bInfo, cInfo, biasInfo, Attributes{})) {}

Gemm::Gemm(const TensorInfo &aInfo, const TensorInfo &bInfo,
    const TensorInfo &cInfo) noexcept(false) : Gemm(aInfo, bInfo, cInfo,
    TensorInfo(Shape{0}, cInfo.GetType(), KuDNN::Layout::A), Attributes{}) {}

Gemm::Gemm(const TensorInfo &aInfo, const TensorInfo &bInfo, const TensorInfo &cInfo,
    const TensorInfo &biasInfo, const Attributes &attributes) noexcept(false)
    : pImpl(new Detail::GemmImpl(aInfo, bInfo, cInfo, biasInfo, attributes)) {}

Gemm::Gemm(const TensorInfo &aInfo, const TensorInfo &bInfo,
    const TensorInfo &cInfo, const Attributes &attributes) noexcept(false) : Gemm(aInfo, bInfo, cInfo,
    TensorInfo(Shape{0}, cInfo.GetType(), KuDNN::Layout::A), attributes) {}

Gemm::Gemm(const Gemm &other) noexcept(false)
    : pImpl(new Detail::GemmImpl{*(other.pImpl.get())}) {}

Gemm::Gemm(Gemm &&other) noexcept
    : pImpl(std::move(other.pImpl)) {}

Gemm& Gemm::operator=(const Gemm &other) noexcept(false)
{
    if (this != &other) {
        this->pImpl.reset(new Detail::GemmImpl{*(other.pImpl.get())});
    }
    return *this;
}

Gemm& Gemm::operator=(Gemm &&other) noexcept
{
    if (this != &other) {
        this->pImpl = std::move(other.pImpl);
    }
    return *this;
}

void Gemm::Run(const void *a, const void *b, void *c, const void *bias, const PostOpsDataPtrs &postOpsPtrs,
    float alpha, float beta) const noexcept(false)
{
    pImpl->Run(a, b, c, bias, postOpsPtrs, alpha, beta);
}

void Gemm::Run(const void *a, const void *b, void *c, const void *bias,
    float alpha, float beta) const noexcept(false)
{
    pImpl->Run(a, b, c, bias, {}, alpha, beta);
}

void Gemm::Run(const void *a, const void *b, void *c, const PostOpsDataPtrs &postOpsPtrs,
    float alpha, float beta) const noexcept(false)
{
    pImpl->Run(a, b, c, nullptr, postOpsPtrs, alpha, beta);
}

void Gemm::Run(const void *a, const void *b, void *c,
    float alpha, float beta) const noexcept(false)
{
    pImpl->Run(a, b, c, nullptr, {}, alpha, beta);
}

Status Gemm::ValidateInput(const TensorInfo &aInfo, const TensorInfo &bInfo,
    const TensorInfo &cInfo, const TensorInfo &biasInfo) noexcept
{
    return Gemm::ValidateInput(aInfo, bInfo, cInfo, biasInfo, Attributes{});
}

Status Gemm::ValidateInput(const TensorInfo &aInfo, const TensorInfo &bInfo,
    const TensorInfo &cInfo) noexcept
{
    return Gemm::ValidateInput(aInfo, bInfo, cInfo,
        TensorInfo(Shape{0}, cInfo.GetType(), KuDNN::Layout::A), Attributes{});
}

Status Gemm::ValidateInput(const TensorInfo &aInfo, const TensorInfo &bInfo,
    const TensorInfo &cInfo, const TensorInfo &biasInfo, const Attributes &attributes) noexcept
{
    bool areDimsDifferent = (aInfo.GetNumDims() != bInfo.GetNumDims()) ||
        (aInfo.GetNumDims() != cInfo.GetNumDims());
    bool isBiasNull = biasInfo.GetTotalTensorSize() == 0;
    if (!isBiasNull) {
        areDimsDifferent = (areDimsDifferent || (aInfo.GetNumDims() != biasInfo.GetNumDims()));
    }
    bool isBadArguments = areDimsDifferent;
    // since num_dims are equal for all we can check only for one tensor
    isBadArguments = isBadArguments || (aInfo.GetNumDims() < DIM_2);
    if (isBadArguments) {
        return Status::BAD_ARGUMENTS;
    }
    // perform checks of data Shapes
    //    A    x    B    =    C
    // [m * k] x [k * n] = [m * n]
    isBadArguments = isBadArguments ||
        (aInfo.GetDims()[aInfo.GetNumDims() - DIM_2] != cInfo.GetDims()[cInfo.GetNumDims() - DIM_2]) ||
        (aInfo.GetDims()[aInfo.GetNumDims() - DIM_1] != bInfo.GetDims()[bInfo.GetNumDims() - DIM_2]) ||
        (bInfo.GetDims()[bInfo.GetNumDims() - DIM_1] != cInfo.GetDims()[cInfo.GetNumDims() - DIM_1]);
    if (isBadArguments) {
        return Status::BAD_ARGUMENTS;
    }
    // perform checks of batch sizes
    for (SizeType dim = 0; dim < bInfo.GetNumDims() - DIM_2; ++dim) {
        bool badDims = false;
        if (aInfo.GetDims()[dim] != 1) {
            badDims = (badDims || (aInfo.GetDims()[dim] != cInfo.GetDims()[dim]));
        }
        if (bInfo.GetDims()[dim] != 1) {
            badDims = (badDims || (bInfo.GetDims()[dim] != cInfo.GetDims()[dim]));
        }
        if (!isBiasNull && (biasInfo.GetDims()[dim] != 1)) {
            badDims = (badDims || (biasInfo.GetDims()[dim] != cInfo.GetDims()[dim]));
        }
        if (badDims) {
            return Status::BAD_ARGUMENTS;
        }
    }
    // perform check that bias and C has corresponding Shape
    // bias dimensions can be implicitly broadcasted to C dimensions
    isBadArguments = !isBiasNull && (((biasInfo.GetDims()[biasInfo.GetNumDims() - DIM_1] != NUM_1) &&
        (biasInfo.GetDims()[biasInfo.GetNumDims() - DIM_1] != cInfo.GetDims()[cInfo.GetNumDims() - DIM_1])) ||
        ((biasInfo.GetDims()[biasInfo.GetNumDims() - DIM_2] != NUM_1) &&
        (biasInfo.GetDims()[biasInfo.GetNumDims() - DIM_2] != cInfo.GetDims()[cInfo.GetNumDims() - DIM_2])));
    if (isBadArguments) {
        return Status::BAD_ARGUMENTS;
    }
    // we can't process GEMM with strides over the fastest changing index
    bool isUnsupported = ((aInfo.GetStrides()[Service::GetIdxForUserLayout(aInfo.GetLayout(),
        aInfo.GetNumDims() - DIM_1)] != 1) &&
        (aInfo.GetStrides()[Service::GetIdxForUserLayout(aInfo.GetLayout(), aInfo.GetNumDims() - DIM_1)] != 0)) &&
        ((bInfo.GetStrides()[Service::GetIdxForUserLayout(bInfo.GetLayout(), bInfo.GetNumDims() - DIM_1)] != 1) &&
        (bInfo.GetStrides()[Service::GetIdxForUserLayout(bInfo.GetLayout(), bInfo.GetNumDims() - DIM_1)] != 0)) &&
        ((cInfo.GetStrides()[Service::GetIdxForUserLayout(cInfo.GetLayout(), cInfo.GetNumDims() - DIM_1)] != 1) &&
        (cInfo.GetStrides()[Service::GetIdxForUserLayout(cInfo.GetLayout(), cInfo.GetNumDims() - DIM_1)] != 0)) &&
        ((!isBiasNull &&
        ((biasInfo.GetStrides()[Service::GetIdxForUserLayout(biasInfo.GetLayout(), biasInfo.GetNumDims() - DIM_1)]
            != 1) &&
        (biasInfo.GetStrides()[Service::GetIdxForUserLayout(biasInfo.GetLayout(), biasInfo.GetNumDims() - DIM_1)]
            != 0))));
    if (isUnsupported) {
        return Status::UNSUPPORTED;
    }
    bool anyZeroDim = (aInfo.GetTotalTensorSize() == 0) || (bInfo.GetTotalTensorSize() == 0) ||
        (cInfo.GetTotalTensorSize() == 0);
    if (anyZeroDim) {
        return Status::UNSUPPORTED;
    }

    if (!isBiasNull && (biasInfo.GetTotalTensorSize() == 0)) {
        return Status::UNSUPPORTED;
    }
    if (attributes.GetPostOps().Len() > 0) {
        bool attributesOK = (attributes.GetPostOps().Len() == 1) &&
            (attributes.GetPostOps().Kind(0) == PrimitiveKind::SUM);
        attributesOK |= (attributes.GetPostOps().Len() == 1) &&
            (attributes.GetPostOps().Kind(0) == PrimitiveKind::ELTWISE);
        if (attributesOK && (attributes.GetPostOps().Kind(0) == PrimitiveKind::ELTWISE)) {
            ActivationFunction actF = ActivationFunction::UNIMPLEMENTED;
            float alpha = 1.0f;
            float beta = 0.0f;
            attributes.GetPostOps().GetParamsEltwise(0, actF, alpha, beta);
            attributesOK &= (actF == ActivationFunction::RELU);
        }
        if (!attributesOK) {
            return Status::UNSUPPORTED;
        }
    }
    return Status::SUCCESS;
}

Status Gemm::ValidateInput(const TensorInfo &aInfo, const TensorInfo &bInfo,
    const TensorInfo &cInfo, const Attributes &attributes) noexcept
{
    return Gemm::ValidateInput(aInfo, bInfo, cInfo,
        TensorInfo(Shape{0}, cInfo.GetType(), KuDNN::Layout::A), attributes);
}

} // KuDNN
