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

        class LinearActivationLayerFWDImpl final {
        public:
            LinearActivationLayerFWDImpl(const TensorInfo &aInfo, const TensorInfo &bInfo, const TensorInfo &cInfo,
                const TensorInfo &biasInfo, float alpha, float beta, ActivationFunction kind, int numThreads)
                noexcept(false): gemm(aInfo, bInfo, cInfo, biasInfo, numThreads),
                activationlayerfwd(cInfo, cInfo, kind, alpha), dstInfo(cInfo)
                    {
                            Service::ThrowOnStatus(LinearActivationLayerFWD::ValidateInput(
                                aInfo, bInfo, cInfo, biasInfo, alpha, beta, kind, numThreads),
                                "LinearActivationLayerFWD");
                    }

            void Run(const void *a, const void *b, void *c, const void *bias, int numThreads) const noexcept(false)
            {
                gemm.Run(a, b, c, bias, 1.0f, 0.0f, numThreads);
                Element::Type dstType = dstInfo.GetType();
                SizeType dataSize = dstType.GetSize();
                SizeType dstSize = dstInfo.GetTotalTensorSize();
                void* tmp = malloc(dataSize * dstSize);
                if (tmp == nullptr) {
                    throw std::bad_alloc();
                }
                memcpy(tmp, c, dataSize * dstSize);

                activationlayerfwd.Run(tmp, c);
                if (tmp != nullptr) {
                    free(tmp);
                }
            }

        private:
            const KuDNN::Gemm gemm;
            const KuDNN::ActivationLayerFWD activationlayerfwd;
            const TensorInfo dstInfo;
        };

    } // Detail

    LinearActivationLayerFWD::~LinearActivationLayerFWD() noexcept {}


LinearActivationLayerFWD::LinearActivationLayerFWD(const TensorInfo &aInfo, const TensorInfo &bInfo,
    const TensorInfo &cInfo, const TensorInfo &biasInfo,
    float alpha, float beta, ActivationFunction kind,
    int numThreads) noexcept(false): pImpl(new Detail::LinearActivationLayerFWDImpl
    (aInfo, bInfo, cInfo, biasInfo, alpha, beta, kind, numThreads)) {}


LinearActivationLayerFWD::LinearActivationLayerFWD(const TensorInfo &aInfo, const TensorInfo &bInfo,
    const TensorInfo &cInfo, float alpha, float beta,
    ActivationFunction kind, int numThreads)
    noexcept(false) : LinearActivationLayerFWD(aInfo,
    bInfo, cInfo, TensorInfo(Shape{0}, cInfo.GetType(), KuDNN::Layout::A), alpha, beta, kind, numThreads) {}


LinearActivationLayerFWD::LinearActivationLayerFWD
(const LinearActivationLayerFWD &other)
noexcept(false): pImpl(new Detail::LinearActivationLayerFWDImpl{*(other.pImpl.get())}) {}


LinearActivationLayerFWD::LinearActivationLayerFWD
(LinearActivationLayerFWD &&other)
noexcept: pImpl(std::move(other.pImpl)) {}

LinearActivationLayerFWD &LinearActivationLayerFWD::operator=(const LinearActivationLayerFWD &other) noexcept(false)
{
if (this != &other) {
this->pImpl.reset(new Detail::LinearActivationLayerFWDImpl{*(other.pImpl.get())});
}
return *this;
}

LinearActivationLayerFWD &LinearActivationLayerFWD::operator=(LinearActivationLayerFWD &&other) noexcept
{
if (this != &other) {
this->pImpl = std::move(other.pImpl);
}
return *this;
}


void LinearActivationLayerFWD::Run(const void *a, const void *b, void *c,
    const void *bias, int numThreads) const noexcept(false)
{
pImpl->Run(a, b, c, bias, numThreads);
}


Status LinearActivationLayerFWD::ValidateInput(const TensorInfo &aInfo, const TensorInfo &bInfo,
    const TensorInfo &cInfo, const TensorInfo &biasInfo,
    float alpha, float beta, ActivationFunction kind,
    int numThreads) noexcept(false)
{
Status status = Gemm::ValidateInput(aInfo, bInfo, cInfo, biasInfo, numThreads);
if (status != Status::SUCCESS) {
return status;
}
return ActivationLayerFWD::ValidateInput(cInfo, cInfo, kind, alpha, beta);
}


Status LinearActivationLayerFWD::ValidateInput(const TensorInfo &aInfo, const TensorInfo &bInfo,
    const TensorInfo &cInfo, float alpha, float beta,
    ActivationFunction kind,
    int numThreads) noexcept(false)
{
Status status = Gemm::ValidateInput(aInfo, bInfo, cInfo, numThreads);
if (status != Status::SUCCESS) {
return status;
}
return ActivationLayerFWD::ValidateInput(cInfo, cInfo, kind, alpha, beta);
}

} // KuDNN