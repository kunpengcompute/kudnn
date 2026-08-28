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

#include "operations/kudnn_rnn.hpp"
#include "internal_funcs.hpp"
#include "kernels/rnn/kudnn_rnn_kernels.hpp"

namespace KuDNN {
namespace Detail {
namespace Rnn {

class RnnFWDImpl final {
public:
    RnnFWDImpl(const std::vector<TensorInfo> &srcInfos, const std::vector<TensorInfo> &weightInfos,
        const std::vector<TensorInfo> &dstInfos, TensorInfo &biasInfo, RnnCfg &rnn, RnnAlgorithm alg,
        ActivateFunctionRNN actf) noexcept(false) : srcType(srcInfos[0].GetType()), biasType(biasInfo.GetType()),
        myRnn(rnn), myAlg(alg), myActf(actf)
    {
        Service::ThrowOnStatus(RnnFWD::ValidateInput(srcInfos, weightInfos, dstInfos, biasInfo, rnn, alg, actf),
            "RNN Forword");
        for (SizeType k = 0; k < srcInfos.size(); k++) {
            Shape dim = Service::GetShapeAccordingToLayout(srcInfos[k].GetDims(), srcInfos[k].GetLayout());
            srcDims.push_back(dim);
        }
        for (SizeType k = 0; k < weightInfos.size(); k++) {
            Shape dim = Service::GetShapeAccordingToLayout(weightInfos[k].GetDims(), weightInfos[k].GetLayout());
            weightDims.push_back(dim);
        }
        for (SizeType k = 0; k < dstInfos.size(); k++) {
            Shape dim = Service::GetShapeAccordingToLayout(dstInfos[k].GetDims(), dstInfos[k].GetLayout());
            dstDims.push_back(dim);
        }
        biasDim = Service::GetShapeAccordingToLayout(biasInfo.GetDims(), biasInfo.GetLayout());

        myRnn.srcLayerCellLd = srcInfos[0].GetStrides()[0];
        myRnn.dstLayerCellLd = dstInfos[0].GetStrides()[0];

        Init();
    }

    RnnFWDImpl(const RnnFWDImpl &other) noexcept(false) : srcType(other.srcType), biasType(other.biasType),
        myRnn(other.myRnn), myAlg(other.myAlg), myActf(other.myActf)
    {
        Init();
    }

    RnnFWDImpl& operator=(const RnnFWDImpl &other) noexcept(false)
    {
        if (this != &other) {
            this->srcType = other.srcType;
            this->biasType = other.biasType;
            this->myRnn = other.myRnn;
            this->myAlg = other.myAlg;
            this->myActf = other.myActf;
            this->Init();
        }
        return *this;
    }

    void Run(const void **src, const void **weight, void **dst, const void *bias, void *wsBase,
        const void *augruAttentions) const noexcept(false)
    {
        ValidateBasicInputPointers(src, weight, dst, bias);

        ValidateLstmConditions(src, weight, dst);

        ValidateAugruConditions(augruAttentions);

        RnnKerParas paras{srcType, biasType, myAlg};
        if (RNN_KER_MAP.find(paras) != RNN_KER_MAP.end()) {
            auto kerFunc = RNN_KER_MAP.at(paras);
            kerFunc(src, weight, dst, bias, wsBase, wsStatesLayer.data(), scratchGates.data(),
                augruAttentions, projectionHt.data(), myRnn, myActf, scratchCells.data(), rnnGemmPtr.get());
        } else {
            throw Service::Unsupported {"Rnn: Unsupported data type!"};
        }
    }

private:
    Element::Type srcType;
    Element::Type biasType;
    std::vector<Shape> srcDims;
    std::vector<Shape> weightDims;
    std::vector<Shape> dstDims;
    std::vector<char, Service::AlignedAllocator<char>> wsStatesLayer;
    std::vector<float, Service::AlignedAllocator<float>> scratchGates;
    std::vector<char, Service::AlignedAllocator<char>> projectionHt;
    std::vector<float, Service::AlignedAllocator<float>> scratchCells;
    Shape biasDim;
    RnnCfg myRnn;
    RnnAlgorithm myAlg;
    ActivateFunctionRNN myActf;
    std::unique_ptr<RnnGemm> rnnGemmPtr;

    void Init()
    {
        if (myAlg == KuDNN::RnnAlgorithm::LSTM) {
            rnnGemmPtr = std::make_unique<LstmGemm>(srcType, biasType, myRnn);
        } else if (myAlg == KuDNN::RnnAlgorithm::GRU || myAlg == KuDNN::RnnAlgorithm::AUGRU) {
            rnnGemmPtr = std::make_unique<GruGemm>(srcType, biasType, myRnn);
        } else if (myAlg == KuDNN::RnnAlgorithm::LBR_GRU || myAlg == KuDNN::RnnAlgorithm::LBR_AUGRU) {
            rnnGemmPtr = std::make_unique<LbrGemm>(srcType, biasType, myRnn);
        } else {
            rnnGemmPtr = std::make_unique<RnnGemm>(srcType, biasType, myRnn);
        }
        if (!myRnn.isTraining) {
            wsStatesLayer.resize(myRnn.wsStatesLayerSize, 0);
            projectionHt.resize(myRnn.mb * myRnn.projHtLd * srcType.GetSize());
        }
        scratchCells.resize(myRnn.wsGatesNld * myRnn.wsGatesLd, 0.0f);
        scratchGates.resize(myRnn.scratchGatesNld * myRnn.scratchGatesLd, 0.0f);
    }

    void ValidateBasicInputPointers(const void **src, const void **weight,
        void **dst, const void *bias)  const noexcept(false)
    {
        if (src == nullptr || weight == nullptr || dst == nullptr || bias == nullptr ||
            src[0] == nullptr || src[1] == nullptr || weight[0] == nullptr || weight[1] == nullptr ||
            dst[0] == nullptr || dst[1] == nullptr) {
            throw Service::LogicError {"Rnn: one of input matrices is nullptr"};
        }
    }

    void ValidateLstmConditions(const void **src, const void **weight, void **dst) const noexcept(false)
    {
        if (myAlg != KuDNN::RnnAlgorithm::LSTM) {
            return;
        }

        // Check BF16 + projection unsupported
        if (KuDNN::Element::TypeT(srcType) == KuDNN::Element::TypeT::BF16 && weight[0x3]) {
            throw Service::Unsupported {"Rnn: LSTM with projection but BF16 not supported!"};
        }

        // Check LSTM iterC pointer
        if (src[0x2] == nullptr || dst[0x2] == nullptr) {
            throw Service::LogicError {"Rnn: Lstm iterC is nullptr"};
        }

        // Check projection dimension consistency
        if (weight[0x3] == nullptr && myRnn.dhc != myRnn.dic) {
            throw Service::Unsupported {"Rnn: Lstm weight[3] == nullptr must rnn.dhc = rnn.dic"};
        }
    }

    void ValidateAugruConditions(const void *augruAttentions) const noexcept(false)
    {
        if (myAlg != KuDNN::RnnAlgorithm::AUGRU && myAlg != KuDNN::RnnAlgorithm::LBR_AUGRU) {
            return;
        }

        // Check AUGRU attention pointer
        if (augruAttentions == nullptr) {
            throw Service::LogicError {"Rnn: AUGRU or LBRAUGRU augruAttentions is nullptr"};
        }
    }
};

} // Rnn
} // Detail

RnnFWD::~RnnFWD() noexcept {}

RnnFWD::RnnFWD(const std::vector<TensorInfo> &srcInfos, const std::vector<TensorInfo> &weightInfos,
    const std::vector<TensorInfo> &dstInfos, TensorInfo &biasInfo, RnnCfg &rnn, RnnAlgorithm alg,
    const ActivateFunctionRNN actf) noexcept(false)
    : pImpl(new Detail::Rnn::RnnFWDImpl(srcInfos, weightInfos, dstInfos, biasInfo, rnn, alg, actf)) {}

RnnFWD::RnnFWD(const RnnFWD &other) noexcept(false)
    : pImpl(new Detail::Rnn::RnnFWDImpl{*(other.pImpl.get())}) {}

RnnFWD::RnnFWD(RnnFWD &&other) noexcept
    : pImpl(std::move(other.pImpl)) {}

RnnFWD& RnnFWD::operator=(const RnnFWD &other) noexcept(false)
{
    if (this != &other) {
        this->pImpl.reset(new Detail::Rnn::RnnFWDImpl{*(other.pImpl.get())});
    }
    return *this;
}

RnnFWD& RnnFWD::operator=(RnnFWD &&other) noexcept
{
    if (this != &other) {
        this->pImpl = std::move(other.pImpl);
    }
    return *this;
}

void RnnFWD::Run(const void **src, const void **weight, void **dst, const void *bias, void *wsBase,
    const void *augruAttentions) const noexcept(false)
{
    pImpl->Run(src, weight, dst, bias, wsBase, augruAttentions);
}

// Validate basic algorithm legality (no loop, few conditions)
static Status ValidateAlgorithmBasic(KuDNN::RnnAlgorithm alg, KuDNN::ActivateFunctionRNN actf) noexcept
{
    // Check if the algorithm is unimplemented
    if (alg == KuDNN::RnnAlgorithm::UNIMPLEMENTED) {
        return Status::UNSUPPORTED;
    }
    // Check activation function for RNN algorithm
    if (alg == KuDNN::RnnAlgorithm::RNN && actf == KuDNN::ActivateFunctionRNN::UNIMPLEMENTED) {
        return Status::UNSUPPORTED;
    }
    // Check if the algorithm is in the supported list
    if (!Service::OneOf(alg, KuDNN::RnnAlgorithm::RNN, KuDNN::RnnAlgorithm::LSTM, KuDNN::RnnAlgorithm::GRU,
        KuDNN::RnnAlgorithm::LBR_GRU, KuDNN::RnnAlgorithm::AUGRU, KuDNN::RnnAlgorithm::LBR_AUGRU)) {
        return Status::UNSUPPORTED;
    }
    return Status::SUCCESS;
}

// Validate tensor data type legality (including loops)
static Status ValidateDataType(const std::vector<TensorInfo> &srcInfos,
                               const std::vector<TensorInfo> &weightInfos,
                               const std::vector<TensorInfo> &dstInfos,
                               TensorInfo &biasInfo) noexcept
{
    // Validate src tensor data type
    for (auto&& src : srcInfos) {
        if (src.GetType() != KuDNN::Element::TypeT::F32 && src.GetType() != KuDNN::Element::TypeT::BF16) {
            return Status::UNSUPPORTED;
        }
    }
    // Validate weight tensor data type
    for (auto&& weight : weightInfos) {
        if (weight.GetType() != KuDNN::Element::TypeT::F32 && weight.GetType() != KuDNN::Element::TypeT::BF16) {
            return Status::UNSUPPORTED;
        }
    }
    // Validate dst tensor data type
    for (auto&& dst : dstInfos) {
        if (dst.GetType() != KuDNN::Element::TypeT::F32 && dst.GetType() != KuDNN::Element::TypeT::BF16) {
            return Status::UNSUPPORTED;
        }
    }
    // Validate bias tensor data type
    if (biasInfo.GetType() != KuDNN::Element::TypeT::F32 && biasInfo.GetType() != KuDNN::Element::TypeT::BF16) {
        return Status::UNSUPPORTED;
    }
    return Status::SUCCESS;
}

// Validate dimension legality (dhc/dic/sic/slc, etc.)
static Status ValidateDimension(KuDNN::RnnAlgorithm alg, RnnCfg &rnn) noexcept
{
    // Validate consistency between dhc and dic
    if (Service::OneOf(alg, KuDNN::RnnAlgorithm::RNN, KuDNN::RnnAlgorithm::GRU, KuDNN::RnnAlgorithm::LBR_GRU,
        KuDNN::RnnAlgorithm::AUGRU, KuDNN::RnnAlgorithm::LBR_AUGRU) && rnn.dhc != rnn.dic) {
        return Status::UNSUPPORTED;
    }
    // Validate consistency between sic and dic
    if ((Service::OneOf(alg, KuDNN::RnnAlgorithm::GRU, KuDNN::RnnAlgorithm::LBR_GRU,
        KuDNN::RnnAlgorithm::AUGRU, KuDNN::RnnAlgorithm::LBR_AUGRU) || rnn.nIter != 1) && rnn.sic != rnn.dic) {
        return Status::UNSUPPORTED;
    }
    // Validate consistency between slc and dic
    if (rnn.nLayer != 1 && rnn.slc != rnn.dic) {
        return Status::UNSUPPORTED;
    }
    return Status::SUCCESS;
}

// Validate gate count (nGates) legality
static Status ValidateGateCount(KuDNN::RnnAlgorithm alg, RnnCfg &rnn) noexcept
{
    if ((alg == KuDNN::RnnAlgorithm::RNN && rnn.nGates != 1) ||
        (alg == KuDNN::RnnAlgorithm::LSTM && rnn.nGates != 0x4) ||
        (Service::OneOf(alg, KuDNN::RnnAlgorithm::GRU, KuDNN::RnnAlgorithm::LBR_GRU,
        KuDNN::RnnAlgorithm::AUGRU, KuDNN::RnnAlgorithm::LBR_AUGRU) && rnn.nGates != 0x3)) {
        return Status::UNSUPPORTED;
    }
    return Status::SUCCESS;
}

Status RnnFWD::ValidateInput(const std::vector<TensorInfo> &srcInfos, const std::vector<TensorInfo> &weightInfos,
    const std::vector<TensorInfo> &dstInfos, TensorInfo &biasInfo, RnnCfg &rnn,
    RnnAlgorithm alg, const ActivateFunctionRNN actf) noexcept
{
    // 1. Validate basic algorithm legality
    Status status = ValidateAlgorithmBasic(alg, actf);
    if (status != Status::SUCCESS) {
        return status;
    }

    // 2. Validate tensor data type legality
    status = ValidateDataType(srcInfos, weightInfos, dstInfos, biasInfo);
    if (status != Status::SUCCESS) {
        return status;
    }

    // 3. Validate dimension legality
    status = ValidateDimension(alg, rnn);
    if (status != Status::SUCCESS) {
        return status;
    }

    // 4. Validate gate count legality
    status = ValidateGateCount(alg, rnn);
    if (status != Status::SUCCESS) {
        return status;
    }

    return Status::SUCCESS;
}

} // KuDNN
