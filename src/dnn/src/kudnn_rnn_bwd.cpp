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

constexpr SizeType ALIGNMENT = 4096;

class RnnBWDImpl final {
public:
    RnnBWDImpl(const std::vector<TensorInfo> &srcInfos, const std::vector<TensorInfo> &weightInfos,
    const std::vector<TensorInfo> &dstInfos, TensorInfo &biasInfo,
    const std::vector<TensorInfo> &diffSrcInfos, const std::vector<TensorInfo> &diffWeightInfos,
    const std::vector<TensorInfo> &diffDstInfos, TensorInfo &diffBiasInfo, RnnCfg &rnn, RnnAlgorithm alg,
    ActivateFunctionRNN actf) noexcept(false)
        : srcType(srcInfos[0].GetType()), biasType(biasInfo.GetType()), dstType(dstInfos[0].GetType()),
        myRnn(rnn), myAlg(alg), myActf(actf)
    {
        Service::ThrowOnStatus(RnnBWD::ValidateInput(srcInfos, weightInfos, dstInfos, biasInfo, diffSrcInfos,
        diffWeightInfos, diffDstInfos, diffBiasInfo, alg, actf),
            "RNN Backrward");

        biasDim = Service::GetShapeAccordingToLayout(biasInfo.GetDims(), biasInfo.GetLayout());

        myRnn.srcLayerCellLd = srcInfos[0].GetStrides()[0];
        myRnn.dstLayerCellLd = dstInfos[0].GetStrides()[0];
        Init();
    }

    RnnBWDImpl(const RnnBWDImpl &other) noexcept(false)
        : srcType(other.srcType), biasType(other.biasType), dstType(other.dstType),
        myRnn(other.myRnn), myAlg(other.myAlg), myActf(other.myActf)
    {
        Init();
    }

    RnnBWDImpl& operator=(const RnnBWDImpl &other) noexcept(false)
    {
        if (this != &other) {
            this->srcType = other.srcType;
            this->biasType = other.biasType;
            this->dstType = other.dstType;
            this->myRnn = other.myRnn;
            this->myAlg = other.myAlg;
            this->myActf = other.myActf;
            this->Init();
        }
        return *this;
    }

    void Run(const void **src, const void **weight, const void **dst, const void *bias, void *wsBase,
        const void **diffDstVec, void **diffSrcVec, void **diffWeightVec, void *diffBias,
        const void *augruAttention, void *diffAugruAttention) const noexcept(false)
    {
        bool validateInput = (src == nullptr || weight == nullptr || dst == nullptr || bias == nullptr ||
                              wsBase == nullptr || diffDstVec == nullptr || diffSrcVec == nullptr ||
                              diffWeightVec == nullptr || diffBias == nullptr || src[0] == nullptr ||
                              src[1] == nullptr || weight[0] == nullptr || weight[1] == nullptr || dst[0] == nullptr ||
                              dst[1] == nullptr || diffSrcVec[0] == nullptr || diffSrcVec[1] == nullptr ||
                              diffWeightVec[0] == nullptr || diffWeightVec[1] == nullptr || diffDstVec[0] == nullptr ||
                              diffDstVec[1] == nullptr);
        if (validateInput) {
            throw Service::LogicError{"Rnn: one of input matrices is nullptr"};
        }
        if (myAlg == KuDNN::RnnAlgorithm::LSTM) {
            if (KuDNN::Element::TypeT(srcType) == KuDNN::Element::TypeT::BF16 && weight[0x3]) {
                throw Service::Unsupported {"Rnn: LSTM with projection but BF16 not supported!"};
            }
            if (src[0x2] == nullptr || dst[0x2] == nullptr) {
                throw Service::LogicError {"Rnn: Lstm iterC is nullptr"};
            }
            if (diffSrcVec[0x2] == nullptr || diffDstVec[0x2] == nullptr) {
                throw Service::LogicError {"Rnn: Lstm iterC is nullptr"};
            }
            if (weight[0x3] == nullptr && myRnn.dhc != myRnn.dic) {
                throw Service::Unsupported {"Rnn: Lstm weight[3] == nullptr must rnn.dhc = rnn.dic"};
            }
        }
        if (myAlg == KuDNN::RnnAlgorithm::AUGRU || myAlg == KuDNN::RnnAlgorithm::LBR_AUGRU) {
            if (augruAttention == nullptr || diffAugruAttention == nullptr) {
                throw Service::LogicError {"Rnn: AUGRU or LBRAUGRU augruAttention is nullptr"};
            }
        }
        RnnKerParas paras{srcType, biasType, myAlg};
        if (RNN_KER_BWD_MAP.find(paras) != RNN_KER_BWD_MAP.end()) {
            auto kerFunc = RNN_KER_BWD_MAP.at(paras);
            kerFunc(src, weight, dst, scratchGates.data(), augruAttention, diffAugruAttention,
                wsBase, diffDstVec, diffSrcVec, diffWeightVec, diffBias,
                wsDiffStatesLayer.data(), wsDiffStatesIter.data(), wsDiffStatesIterC.data(),
                myRnn, myActf, scratchCells.data(), scratchDiffHt.data(), rnnGemmPtr.get());
        } else {
            throw Service::Unsupported {"Rnn: Not supported!"};
        }
    }

private:
    Element::Type srcType;
    Element::Type biasType;
    Element::Type dstType;
    std::vector<char, Service::AlignedAllocator<char, ALIGNMENT>> wsDiffStatesIter;
    std::vector<char, Service::AlignedAllocator<char, ALIGNMENT>> wsDiffStatesIterC;
    std::vector<char, Service::AlignedAllocator<char, ALIGNMENT>> wsDiffStatesLayer;
    std::vector<char, Service::AlignedAllocator<char, ALIGNMENT>> scratchGates;
    std::vector<float, Service::AlignedAllocator<float, ALIGNMENT>> scratchCells;
    std::vector<char, Service::AlignedAllocator<char, ALIGNMENT>> scratchDiffHt;
    Shape biasDim;
    RnnCfg myRnn;
    RnnAlgorithm myAlg;
    ActivateFunctionRNN myActf;
    std::unique_ptr<RnnBwdGemm> rnnGemmPtr;

    void Init()
    {
        if (myAlg == KuDNN::RnnAlgorithm::RNN) {
            rnnGemmPtr = std::make_unique<RnnBwdGemm>(srcType, biasType, myRnn);
        } else if (myAlg == KuDNN::RnnAlgorithm::LSTM) {
            rnnGemmPtr = std::make_unique<LstmBwdGemm>(srcType, biasType, myRnn);
        } else if (myAlg == KuDNN::RnnAlgorithm::LBR_GRU || myAlg == KuDNN::RnnAlgorithm::LBR_AUGRU) {
            rnnGemmPtr = std::make_unique<LbrGruBwdGemm>(srcType, biasType, myRnn);
        } else if (myAlg == KuDNN::RnnAlgorithm::GRU || myAlg == KuDNN::RnnAlgorithm::AUGRU) {
            rnnGemmPtr = std::make_unique<GruBwdGemm>(srcType, biasType, myRnn);
        } else {
            rnnGemmPtr = std::make_unique<RnnBwdGemm>(srcType, biasType, myRnn);
        }

        scratchGates.resize(myRnn.scratchGatesSize, 0);
        wsDiffStatesIter.resize(myRnn.wsDiffStatesIterSize, 0);
        wsDiffStatesIterC.resize(myRnn.wsDiffStatesIterCSize, 0);
        wsDiffStatesLayer.resize(myRnn.wsDiffStatesLayerSize, 0);
        scratchCells.resize(myRnn.scratchCellSize, 0.0f);
        scratchDiffHt.resize(myRnn.mb * myRnn.scratchDiffHtLd * srcType.GetSize());
    }
};

} // Rnn
} // Detail

RnnBWD::~RnnBWD() noexcept {}

RnnBWD::RnnBWD(const std::vector<TensorInfo> &srcInfos, const std::vector<TensorInfo> &weightInfos,
    const std::vector<TensorInfo> &dstInfos, TensorInfo &biasInfo,
    const std::vector<TensorInfo> &diffSrcInfos, const std::vector<TensorInfo> &diffWeightInfos,
    const std::vector<TensorInfo> &diffDstInfos, TensorInfo &diffBiasInfo, RnnCfg &rnn, RnnAlgorithm alg,
    const ActivateFunctionRNN actf) noexcept(false)
    : pImpl(new Detail::Rnn::RnnBWDImpl(srcInfos, weightInfos, dstInfos, biasInfo, diffSrcInfos, diffWeightInfos,
     diffDstInfos, diffBiasInfo, rnn, alg, actf)) {}

RnnBWD::RnnBWD(const RnnBWD &other) noexcept(false)
    : pImpl(new Detail::Rnn::RnnBWDImpl{*(other.pImpl.get())}) {}

RnnBWD::RnnBWD(RnnBWD &&other) noexcept
    : pImpl(std::move(other.pImpl)) {}

RnnBWD& RnnBWD::operator=(const RnnBWD &other) noexcept(false)
{
    if (this != &other) {
        this->pImpl.reset(new Detail::Rnn::RnnBWDImpl{*(other.pImpl.get())});
    }
    return *this;
}

RnnBWD& RnnBWD::operator=(RnnBWD &&other) noexcept
{
    if (this != &other) {
        this->pImpl = std::move(other.pImpl);
    }
    return *this;
}

void RnnBWD::Run(const void **src, const void **weight, const void **dst, const void *bias, void *wsBase,
    const void **diffDstVec, void **diffSrcVec, void **diffWeightVec, void *diffBias,
    const void *augruAttention, void *diffAugruAttention) const noexcept(false)
{
    pImpl->Run(src, weight, dst, bias, wsBase, diffDstVec, diffSrcVec, diffWeightVec, diffBias, augruAttention,
        diffAugruAttention);
}

// Validate basic algorithm legality
static Status ValidateAlgorithmBasic(RnnAlgorithm alg, ActivateFunctionRNN actf) noexcept
{
    if (alg == KuDNN::RnnAlgorithm::UNIMPLEMENTED) {
        return Status::UNSUPPORTED;
    }
    if (alg == KuDNN::RnnAlgorithm::RNN && actf == KuDNN::ActivateFunctionRNN::UNIMPLEMENTED) {
        return Status::UNSUPPORTED;
    }
    if (!Service::OneOf(alg, KuDNN::RnnAlgorithm::RNN, KuDNN::RnnAlgorithm::LSTM, KuDNN::RnnAlgorithm::GRU,
                        KuDNN::RnnAlgorithm::LBR_GRU, KuDNN::RnnAlgorithm::AUGRU, KuDNN::RnnAlgorithm::LBR_AUGRU)) {
        return Status::UNSUPPORTED;
    }
    return Status::SUCCESS;
}

// Check the forward tensor type (F32 or BF16 is allowed).
static Status ValidateForwardDataType(const std::vector<TensorInfo> &srcInfos,
                                      const std::vector<TensorInfo> &weightInfos,
                                      const std::vector<TensorInfo> &dstInfos, const TensorInfo &biasInfo) noexcept
{
    for (const auto &src : srcInfos) {
        if (src.GetType() != KuDNN::Element::TypeT::F32 && src.GetType() != KuDNN::Element::TypeT::BF16)
            return Status::UNSUPPORTED;
    }
    for (const auto &weight : weightInfos) {
        if (weight.GetType() != KuDNN::Element::TypeT::F32 && weight.GetType() != KuDNN::Element::TypeT::BF16)
            return Status::UNSUPPORTED;
    }
    for (const auto &dst : dstInfos) {
        if (dst.GetType() != KuDNN::Element::TypeT::F32 && dst.GetType() != KuDNN::Element::TypeT::BF16)
            return Status::UNSUPPORTED;
    }
    if (biasInfo.GetType() != KuDNN::Element::TypeT::F32 && biasInfo.GetType() != KuDNN::Element::TypeT::BF16) {
        return Status::UNSUPPORTED;
    }
    return Status::SUCCESS;
}

// Check the reverse tensor type (F32 is allowed).
static Status ValidateBackwardDataType(const std::vector<TensorInfo> &diffSrcInfos,
                                       const std::vector<TensorInfo> &diffWeightInfos,
                                       const std::vector<TensorInfo> &diffDstInfos,
                                       const TensorInfo &diffBiasInfo) noexcept
{
    for (const auto &src : diffSrcInfos) {
        if (src.GetType() != KuDNN::Element::TypeT::F32)
            return Status::UNSUPPORTED;
    }
    for (const auto &weight : diffWeightInfos) {
        if (weight.GetType() != KuDNN::Element::TypeT::F32)
            return Status::UNSUPPORTED;
    }
    for (const auto &dst : diffDstInfos) {
        if (dst.GetType() != KuDNN::Element::TypeT::F32)
            return Status::UNSUPPORTED;
    }
    if (diffBiasInfo.GetType() != KuDNN::Element::TypeT::F32) {
        return Status::UNSUPPORTED;
    }
    return Status::SUCCESS;
}

Status RnnBWD::ValidateInput(const std::vector<TensorInfo> &srcInfos, const std::vector<TensorInfo> &weightInfos,
    const std::vector<TensorInfo> &dstInfos, TensorInfo &biasInfo,
    const std::vector<TensorInfo> &diffSrcInfos, const std::vector<TensorInfo> &diffWeightInfos,
    const std::vector<TensorInfo> &diffDstInfos, TensorInfo &diffBiasInfo, RnnAlgorithm alg,
    const ActivateFunctionRNN actf) noexcept
{
    // 1. Validate basic algorithm legality
    Status status = ValidateAlgorithmBasic(alg, actf);
    if (status != Status::SUCCESS) {
        return status;
    }

    // 2. Validate the data type of forward propagation
    status = ValidateForwardDataType(srcInfos, weightInfos, dstInfos, biasInfo);
    if (status != Status::SUCCESS) {
        return status;
    }

    // 3. Validate the data type of backward propagation
    status = ValidateBackwardDataType(diffSrcInfos, diffWeightInfos, diffDstInfos, diffBiasInfo);
    if (status != Status::SUCCESS) {
        return status;
    }
    
    return Status::SUCCESS;
}

} // KuDNN
