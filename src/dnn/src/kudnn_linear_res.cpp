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
#include <vector>
#include <cstring>
#include "kudnn.hpp"
#include "kblas.h"

#include <cstddef>
#include <arm_neon.h>
#include <arm_sve.h>
#include "internal_funcs.hpp"
#include "vector_wrappers.hpp"


namespace KuDNN {
namespace Detail {
namespace LinearResHelpers {
    struct MatrixConverter final {
        MatrixConverter(Element::Type fromType, Element::Type toType, const void *data, SizeType size):originData(data),
            p(((toType == fromType) || (data == nullptr)) ? nullptr : (Service::AlignedAlloc(size * toType.GetSize()))),
            r({size, fromType, Layout::A}, {size, toType, Layout::A})
        {
            if (p) {
                r.Run(data, p);
            }
        }
        MatrixConverter(const MatrixConverter &) = delete;
        MatrixConverter(MatrixConverter &&) = delete;
        MatrixConverter &operator=(const MatrixConverter &) = delete;
        MatrixConverter &operator=(MatrixConverter &&) = delete;
        void *Get() const
        {
            return (p == nullptr) ? const_cast<void *>(originData) : p;
        }
        ~MatrixConverter()
        {
            Service::Deallocate(p);
        }
    private:
        const void *originData;
        void *p;
        ReorderLayer r;
    };

    template <typename dstDt, typename resDt, typename linearResDt>
    void AddOps(const dstDt *dst, const resDt *res, linearResDt *linearRes, float gamma,
                KuDNN::ResOpsFunction kind, SizeType size);

    template <>
    void AddOps<float, float, float>(const float *dst, const float *res, float *linearRes,
                                     float gamma, KuDNN::ResOpsFunction kind, SizeType size)
    {
        if (kind == KuDNN::ResOpsFunction::RES_IDENTIAL) {
            for (SizeType i = 0; i < size; ++i) {
                linearRes[i] = dst[i] + res[i];
            }
        } else {
            for (SizeType i = 0; i < size; ++i) {
                linearRes[i] = dst[i] + res[i] * gamma;
            }
        }
    }

    template <>
    void AddOps<__fp16, __fp16, __fp16>(const __fp16 *dst, const __fp16 *res, __fp16 *linearRes,
                                        float gamma, KuDNN::ResOpsFunction kind, SizeType size)
    {
        if (kind == KuDNN::ResOpsFunction::RES_IDENTIAL) {
            for (size_t i = 0; i < size; ++i) {
                linearRes[i] = dst[i] + res[i];
            }
        } else {
            MatrixConverter d(Element::TypeT::F16, Element::TypeT::F32, dst, size);
            MatrixConverter r(Element::TypeT::F16, Element::TypeT::F32, res, size);
            MatrixConverter lr(Element::TypeT::F16, Element::TypeT::F32, linearRes, size);
            const float *fd = static_cast<const float *>(d.Get());
            const float *fr = static_cast<const float *>(r.Get());
            float *flr = static_cast<float *>(lr.Get());
            for (SizeType i = 0; i < size; ++i) {
                flr[i] = fd[i] + fr[i] * gamma;
            }
            Service::ConvertFp32ToFp16(static_cast<const float *>(flr), linearRes, size);
        }
    }

    template <>
    void AddOps<__bf16, __bf16, __bf16>(const __bf16 *dst, const __bf16 *res, __bf16 *linearRes,
                                        float gamma, KuDNN::ResOpsFunction kind, SizeType size)
    {
        if (kind == KuDNN::ResOpsFunction::RES_IDENTIAL) {
            for (size_t i = 0; i < size; ++i) {
#if defined(__GNUC__)
                linearRes[i] = vcvth_bf16_f32(vcvtah_f32_bf16(dst[i]) + vcvtah_f32_bf16(res[i]));
#else
                linearRes[i] = dst[i] + res[i];
#endif
            }
        } else {
            MatrixConverter d(Element::TypeT::BF16, Element::TypeT::F32, dst, size);
            MatrixConverter r(Element::TypeT::BF16, Element::TypeT::F32, res, size);
            MatrixConverter lr(Element::TypeT::BF16, Element::TypeT::F32, linearRes, size);
            const float *fd = static_cast<const float *>(d.Get());
            const float *fr = static_cast<const float *>(r.Get());
            float *flr = static_cast<float *>(lr.Get());
            for (SizeType i = 0; i < size; ++i) {
                flr[i] = fd[i] + fr[i] * gamma;
            }
            Service::ConvertFp32ToBf16(static_cast<const float *>(flr), linearRes, size);
        }
    }

    template <typename DstDt, typename ResDt, typename ResLinearDt>
    void ExecAddOps(const void *dst, const void *res, void *linearRes,
                    float gamma, KuDNN::ResOpsFunction kind, SizeType size)
    {
        const DstDt *fDst = static_cast<const DstDt *>(dst);
        const ResDt *fRes = static_cast<const ResDt *>(res);
        ResLinearDt *fLinearRes = static_cast<ResLinearDt *>(linearRes);
        AddOps(fDst, fRes, fLinearRes, gamma, kind, size);
    }

}  // LinearResHelpers

static Status ValidateResParams(const TensorInfo &aInfo, const TensorInfo &bInfo,
                                const TensorInfo &cInfo, const TensorInfo &biasInfo,
                                const TensorInfo &resInfo, const TensorInfo &linearResInfo,
                                KuDNN::ResOpsFunction kind, int numThreads)
{
    Status status = Gemm::ValidateInput(aInfo, bInfo, cInfo, biasInfo, numThreads);
    if (status != Status::SUCCESS) {
        return status;
    }
    if (!Service::OneOf(kind, KuDNN::ResOpsFunction::RES_EXT, KuDNN::ResOpsFunction::RES_MUL,
                        KuDNN::ResOpsFunction::RES_IDENTIAL)) {
        return Status::UNSUPPORTED;
    }
    if (kind == ResOpsFunction::RES_MUL) {
        return Gemm::ValidateInput(cInfo, resInfo, linearResInfo, numThreads);
    } else {
        bool isBadArguments = (cInfo.GetNumDims() != resInfo.GetNumDims()) ||
                              (cInfo.GetNumDims() != linearResInfo.GetNumDims()) ||
                              (resInfo.GetNumDims() != linearResInfo.GetNumDims());
        if (isBadArguments) {
            return Status::BAD_ARGUMENTS;
        }
        for (SizeType dim = 0; dim < cInfo.GetNumDims() - DIM_1; ++dim) {
            if ((cInfo.GetDims()[dim] != resInfo.GetDims()[dim]) ||
                (cInfo.GetDims()[dim] != linearResInfo.GetDims()[dim]) ||
                (cInfo.GetDims()[dim] != linearResInfo.GetDims()[dim])) {
                return Status::BAD_ARGUMENTS;
            }
        }
        return Status::SUCCESS;
    }
}

class LinearResFWDImpl final {
public:
    LinearResFWDImpl(const TensorInfo &aInfo, const TensorInfo &bInfo, const TensorInfo &cInfo,
                     const TensorInfo &biasInfo, const TensorInfo &resInfo, const TensorInfo &linearResInfo,
                     float alpha, float beta, float gamma, ResOpsFunction kind, int numThreads)
    noexcept(false): gemm(aInfo, bInfo, cInfo, biasInfo, numThreads), algKind(kind),
                     cInfoExec(cInfo), resInfoExec(resInfo), linearResInfoExec(linearResInfo)
    {
        Service::ThrowOnStatus(LinearResFWD::ValidateInput(aInfo, bInfo, cInfo,
            biasInfo, resInfo, linearResInfo, alpha, beta, gamma, kind, numThreads), "LinearResFWD");
    }

    void Run(const void *a, const void *b, void *c, const void *bias, const void * res, void *linearRes,
             float gamma, int numThreads) const noexcept(false)
    {
        gemm.Run(a, b, c, bias, 1.0f, 0.0f, numThreads);
        if (algKind == KuDNN::ResOpsFunction::RES_MUL) {
            const KuDNN::Gemm mulOps(cInfoExec, resInfoExec, linearResInfoExec, numThreads);
            mulOps.Run(c, res, linearRes, 1.0f, 0.0f, numThreads);
        } else {
            Element::Type linearResType = linearResInfoExec.GetType();
            SizeType dataSize = linearResInfoExec.GetTotalTensorSize();
            if (linearResType == KuDNN::Element::TypeT::F16) {
                LinearResHelpers::ExecAddOps<__fp16, __fp16, __fp16>(c, res, linearRes, gamma, algKind, dataSize);
            } else if (linearResType == KuDNN::Element::TypeT::BF16) {
                LinearResHelpers::ExecAddOps<__bf16, __bf16, __bf16>(c, res, linearRes, gamma, algKind, dataSize);
            } else {
                LinearResHelpers::ExecAddOps<float, float, float>(c, res, linearRes, gamma, algKind, dataSize);
            }
        }
    }

private:
    const KuDNN::Gemm gemm;
    const KuDNN::ResOpsFunction algKind;
    const TensorInfo cInfoExec;
    const TensorInfo resInfoExec;
    const TensorInfo linearResInfoExec;
};
} // Detail


LinearResFWD::~LinearResFWD() noexcept {}


LinearResFWD::LinearResFWD(const TensorInfo &aInfo, const TensorInfo &bInfo,
    const TensorInfo &cInfo, const TensorInfo &biasInfo, const TensorInfo &res, const TensorInfo &linearRes,
    float alpha, float beta, float gamma, ResOpsFunction kind,
    int numThreads) noexcept(false): pImpl(new Detail::LinearResFWDImpl
    (aInfo, bInfo, cInfo, biasInfo, res, linearRes, alpha, beta, gamma, kind, numThreads)) {}


LinearResFWD::LinearResFWD(const TensorInfo &aInfo, const TensorInfo &bInfo,
    const TensorInfo &cInfo, const TensorInfo &res, const TensorInfo &linearRes,
    float alpha, float beta, float gamma, ResOpsFunction kind, int numThreads) noexcept(false)
    :LinearResFWD(aInfo, bInfo, cInfo, TensorInfo(Shape{0}, cInfo.GetType(), KuDNN::Layout::A),
    res, linearRes, alpha, beta, gamma, kind, numThreads) {}


LinearResFWD::LinearResFWD(const LinearResFWD &other)
    noexcept(false): pImpl(new Detail::LinearResFWDImpl{*(other.pImpl.get())}) {}

LinearResFWD::LinearResFWD(LinearResFWD &&other) noexcept : pImpl(std::move(other.pImpl))
{}

LinearResFWD& LinearResFWD::operator=(const LinearResFWD &other) noexcept(false)
{
    if (this != &other) {
        this->pImpl.reset(new Detail::LinearResFWDImpl{*(other.pImpl.get())});
    }
    return *this;
}


LinearResFWD& LinearResFWD::operator=(LinearResFWD &&other) noexcept
{
    if (this != &other) {
        this->pImpl = std::move(other.pImpl);
    }
    return *this;
}


void LinearResFWD::Run(const void *a, const void *b, void *c,
    const void *bias, const void *res,
    void * linearRes, float gamma,
    int numThreads) const noexcept(false)
{
    pImpl->Run(a, b, c, bias, res, linearRes, gamma, numThreads);
}


void LinearResFWD::Run(const void *a, const void *b, void *c,
    const void *res, void * linearRes, float gamma,
    int numThreads) const noexcept(false)
{
    pImpl->Run(a, b, c, nullptr, res, linearRes, gamma, numThreads);
}


Status LinearResFWD::ValidateInput(const TensorInfo &aInfo, const TensorInfo &bInfo,
    const TensorInfo &cInfo, const TensorInfo &biasInfo, const TensorInfo &resInfo,
    const TensorInfo &linearResInfo, float alpha, float beta, float gamma,
    ResOpsFunction kind, int numThreads) noexcept(false)
{
    Service::Unused(alpha, beta, gamma);
    return Detail::ValidateResParams(aInfo, bInfo, cInfo, biasInfo, resInfo, linearResInfo, kind, numThreads);
}


Status LinearResFWD::ValidateInput(const TensorInfo &aInfo, const TensorInfo &bInfo,
    const TensorInfo &cInfo, const TensorInfo &resInfo,
    const TensorInfo &linearResInfo,
    float alpha, float beta, float gamma,
    ResOpsFunction kind, int numThreads) noexcept(false)
{
Service::Unused(alpha, beta, gamma);
    return Detail::ValidateResParams(aInfo, bInfo, cInfo, TensorInfo(Shape{0}, cInfo.GetType(), KuDNN::Layout::A),
        resInfo, linearResInfo, kind, numThreads);
}

} // KuDNN