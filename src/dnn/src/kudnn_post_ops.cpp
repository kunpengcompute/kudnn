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

#include "internal_funcs.hpp"

namespace KuDNN {
namespace Detail {

struct PostOpsBase {
    virtual ~PostOpsBase() = default;
    virtual PrimitiveKind Kind() noexcept = 0;
};

struct SumPostOps : PostOpsBase {
    explicit SumPostOps(float scaleArg) noexcept : scale(scaleArg) {}
    PrimitiveKind Kind() noexcept override
    {
        return PrimitiveKind::SUM;
    }
    void GetParams(float &scaleArg) const noexcept
    {
        scaleArg = scale;
    }
    float scale;
    friend bool operator==(const SumPostOps &lhs, const SumPostOps &rhs) noexcept;
};

bool operator==(const SumPostOps &lhs, const SumPostOps &rhs) noexcept
{
    return lhs.scale == rhs.scale;
}

struct EltwisePostOps : PostOpsBase {
    EltwisePostOps(ActivationFunction kindArg, float alphaArg, float betaArg) noexcept : kind(kindArg),
        alpha(alphaArg), beta(betaArg) {}
    PrimitiveKind Kind() noexcept override
    {
        return PrimitiveKind::ELTWISE;
    }
    void GetParams(ActivationFunction &kindArg, float &alphaArg, float &betaArg) const noexcept
    {
        kindArg = kind;
        alphaArg = alpha;
        betaArg = beta;
    }
    ActivationFunction kind;
    float alpha;
    float beta;
    friend bool operator==(const EltwisePostOps &lhs, const EltwisePostOps &rhs) noexcept;
};

bool operator==(const EltwisePostOps &lhs, const EltwisePostOps &rhs) noexcept
{
    return (lhs.kind == rhs.kind) && (lhs.alpha == rhs.alpha) && (lhs.beta == rhs.beta);
}

struct DWPostOps : PostOpsBase {
    DWPostOps(Element::Type weiTypeArg, Element::Type biaTypeArg, Element::Type dstTypeArg,
        SizeType kernelSizeArg, SizeType stridesArg, SizeType paddingLArg) noexcept : weiType(weiTypeArg),
        biaType(biaTypeArg), dstType(dstTypeArg), kernelSize(kernelSizeArg), strides(stridesArg),
        paddingL(paddingLArg) {}
    PrimitiveKind Kind() noexcept override
    {
        return PrimitiveKind::CONVOLUTION;
    }
    void GetParams(Element::Type &weiTypeArg, Element::Type &biaTypeArg, Element::Type &dstTypeArg,
        SizeType &kernelSizeArg, SizeType &stridesArg, SizeType &paddingLArg) const noexcept
    {
        weiTypeArg = weiType;
        biaTypeArg = biaType;
        dstTypeArg = dstType;
        kernelSizeArg = kernelSize;
        stridesArg = strides;
        paddingLArg = paddingL;
    }
    Element::Type weiType;
    Element::Type biaType;
    Element::Type dstType;
    SizeType kernelSize;
    SizeType strides;
    SizeType paddingL;
    friend bool operator==(const DWPostOps &lhs, const DWPostOps &rhs) noexcept;
};

bool operator==(const DWPostOps &lhs, const DWPostOps &rhs) noexcept
{
    return (lhs.weiType == rhs.weiType) && (lhs.biaType == rhs.biaType) &&
        (lhs.dstType == rhs.dstType) && (lhs.kernelSize == rhs.kernelSize) &&
        (lhs.strides == rhs.strides) && (lhs.paddingL == rhs.paddingL);
}

struct BinaryPostOps : PostOpsBase {
    BinaryPostOps(BinaryFunction opArg, const TensorInfo &src1Arg) noexcept(false) : op(opArg), src1(src1Arg) {}
    PrimitiveKind Kind() noexcept override
    {
        return PrimitiveKind::BINARY;
    }
    void GetParams(BinaryFunction &opArg, TensorInfo &src1Arg) const noexcept(false)
    {
        opArg = op;
        src1Arg = src1;
    }
    BinaryFunction op;
    TensorInfo src1;
    friend bool operator==(const BinaryPostOps &lhs, const BinaryPostOps &rhs) noexcept;
};

bool operator==(const BinaryPostOps &lhs, const BinaryPostOps &rhs) noexcept
{
    return (lhs.op == rhs.op) && (lhs.src1 == rhs.src1);
}

struct PReLUPostOps : PostOpsBase {
    explicit PReLUPostOps(PReLUMask maskArg) noexcept : mask(maskArg) {}
    PrimitiveKind Kind() noexcept override
    {
        return PrimitiveKind::PRELU;
    }
    void GetParams(PReLUMask &maskArg) const noexcept(false)
    {
        maskArg = mask;
    }
    PReLUMask mask;
    friend bool operator==(const PReLUPostOps &lhs, const PReLUPostOps &rhs) noexcept;
};

bool operator==(const PReLUPostOps &lhs, const PReLUPostOps &rhs) noexcept
{
    return lhs.mask == rhs.mask;
}

class PostOpsImpl final {
public:
    PostOpsImpl() noexcept(true) : postOpsVec(0) {}
    SizeType Len() const noexcept
    {
        return postOpsVec.size();
    }
    PrimitiveKind Kind(SizeType index) const noexcept
    {
        return postOpsVec[index]->Kind();
    }
    template <typename ... Args>
    Status AppendSum(Args && ... args) noexcept(false)
    {
        postOpsVec.emplace_back(new SumPostOps{std::forward<Args>(args)...});
        return Status::SUCCESS;
    }
    template <typename ... Args>
    Status AppendEltwise(Args && ... args) noexcept(false)
    {
        postOpsVec.emplace_back(new EltwisePostOps{std::forward<Args>(args)...});
        return Status::SUCCESS;
    }
    template <typename ... Args>
    Status AppendDW(Args && ... args) noexcept(false)
    {
        postOpsVec.emplace_back(new DWPostOps{std::forward<Args>(args)...});
        return Status::SUCCESS;
    }
    template <typename ... Args>
    Status AppendBinary(Args && ... args) noexcept(false)
    {
        postOpsVec.emplace_back(new BinaryPostOps{std::forward<Args>(args)...});
        return Status::SUCCESS;
    }
    template <typename ... Args>
    Status AppendPReLU(Args && ... args) noexcept(false)
    {
        postOpsVec.emplace_back(new PReLUPostOps{std::forward<Args>(args)...});
        return Status::SUCCESS;
    }
    template <typename PrimitiveKind, typename ... Args>
    void GetParams(SizeType index, Args && ... args) const noexcept(false)
    {
        if (index >= postOpsVec.size()) {
            throw Service::LogicError {"PostOps: index >= PostOps quantity"};
        }
        if (postOpsVec[index]->Kind() != GetKind<PrimitiveKind>()) {
            throw Service::LogicError {"PostOps: trying to take parameters for incorrect type of post-ops"};
        }
        static_cast<const PrimitiveKind *>(postOpsVec[index].get())->GetParams(std::forward<Args>(args)...);
    }
    template <typename PKind>
    static constexpr PrimitiveKind GetKind() noexcept
    {
        if constexpr (std::is_same_v<PKind, SumPostOps>) {
            return PrimitiveKind::SUM;
        } else if constexpr (std::is_same_v<PKind, EltwisePostOps>) {
            return PrimitiveKind::ELTWISE;
        } else if constexpr (std::is_same_v<PKind, DWPostOps>) {
            return PrimitiveKind::CONVOLUTION;
        } else if constexpr (std::is_same_v<PKind, BinaryPostOps>) {
            return PrimitiveKind::BINARY;
        } else if constexpr (std::is_same_v<PKind, PReLUPostOps>) {
            return PrimitiveKind::PRELU;
        } else {
            return PrimitiveKind::UNDEFINED;
        }
    }
    std::vector<std::shared_ptr<PostOpsBase>> postOpsVec;
    friend bool operator==(const PostOpsImpl &lhs, const PostOpsImpl &rhs) noexcept;
};

inline bool operator==(const PostOpsImpl &lhs, const PostOpsImpl &rhs) noexcept
{
    if (lhs.postOpsVec.size() != rhs.postOpsVec.size()) {
        return false;
    }
    for (SizeType i = 0; i < lhs.postOpsVec.size(); ++i) {
        if (lhs.postOpsVec[i]->Kind() != rhs.postOpsVec[i]->Kind()) {
            return false;
        }
        if (lhs.postOpsVec[i]->Kind() == PrimitiveKind::SUM) {
            return (*static_cast<const SumPostOps *>(lhs.postOpsVec[i].get())) ==
                (*static_cast<const SumPostOps *>(rhs.postOpsVec[i].get()));
        } else if (lhs.postOpsVec[i]->Kind() == PrimitiveKind::ELTWISE) {
            return (*static_cast<const EltwisePostOps *>(lhs.postOpsVec[i].get())) ==
                (*static_cast<const EltwisePostOps *>(rhs.postOpsVec[i].get()));
        } else if (lhs.postOpsVec[i]->Kind() == PrimitiveKind::CONVOLUTION) {
            return (*static_cast<const DWPostOps *>(lhs.postOpsVec[i].get())) ==
                (*static_cast<const DWPostOps *>(rhs.postOpsVec[i].get()));
        } else if (lhs.postOpsVec[i]->Kind() == PrimitiveKind::BINARY) {
            return (*static_cast<const BinaryPostOps *>(lhs.postOpsVec[i].get())) ==
                (*static_cast<const BinaryPostOps *>(rhs.postOpsVec[i].get()));
        } else if (lhs.postOpsVec[i]->Kind() == PrimitiveKind::PRELU) {
            return (*static_cast<const PReLUPostOps *>(lhs.postOpsVec[i].get())) ==
                (*static_cast<const PReLUPostOps *>(rhs.postOpsVec[i].get()));
        } else {
            return false;
        }
    }
    return true;
}

} // Detail

PostOps::~PostOps() noexcept {}

// all public methods just call corresponding pImpl implemnentation
PostOps::PostOps() noexcept(false)
    : pImpl(new Detail::PostOpsImpl{}) {}

PostOps::PostOps(const PostOps &other) noexcept(false)
    : pImpl(new Detail::PostOpsImpl{*(other.pImpl.get())}) {}

PostOps::PostOps(PostOps &&other) noexcept
    : pImpl(std::move(other.pImpl)) {}

PostOps& PostOps::operator=(const PostOps &other) noexcept(false)
{
    if (this != &other) {
        this->pImpl.reset(new Detail::PostOpsImpl{*(other.pImpl.get())});
    }
    return *this;
}

PostOps& PostOps::operator=(PostOps &&other) noexcept
{
    if (this != &other) {
        this->pImpl = std::move(other.pImpl);
    }
    return *this;
}

SizeType PostOps::Len() const noexcept
{
    return pImpl->Len();
}

PrimitiveKind PostOps::Kind(SizeType index) const noexcept
{
    return pImpl->Kind(index);
}

Status PostOps::AppendSum(float scale) noexcept(false)
{
    return pImpl->AppendSum(scale);
}

void PostOps::GetParamsSum(SizeType index, float &scale) const noexcept(false)
{
    return pImpl->GetParams<Detail::SumPostOps>(index, scale);
}

Status PostOps::AppendEltwise(ActivationFunction kind, float alpha, float beta) noexcept(false)
{
    return pImpl->AppendEltwise(kind, alpha, beta);
}

void PostOps::GetParamsEltwise(SizeType index, ActivationFunction &kind,
    float &alpha, float &beta) const noexcept(false)
{
    return pImpl->GetParams<Detail::EltwisePostOps>(index, kind, alpha, beta);
}

Status PostOps::AppendDW(Element::Type weiType, Element::Type biaType, Element::Type dstType,
    SizeType kernelSize, SizeType strides, SizeType paddingL) noexcept(false)
{
    return pImpl->AppendDW(weiType, biaType, dstType, kernelSize, strides, paddingL);
}

void PostOps::GetParamsDW(SizeType index, Element::Type &weiType, Element::Type &biaType,
    Element::Type &dstType, SizeType &kernelSize, SizeType &strides, SizeType &paddingL) const noexcept(false)
{
    return pImpl->GetParams<Detail::DWPostOps>(index,
        weiType, biaType, dstType, kernelSize, strides, paddingL);
}

Status PostOps::AppendBinary(BinaryFunction op, const TensorInfo &src1) noexcept(false)
{
    return pImpl->AppendBinary(op, src1);
}

void PostOps::GetParamsBinary(SizeType index, BinaryFunction &op, TensorInfo &src1) const noexcept(false)
{
    return pImpl->GetParams<Detail::BinaryPostOps>(index, op, src1);
}

Status PostOps::AppendPReLU(PReLUMask mask) noexcept(false)
{
    return pImpl->AppendPReLU(mask);
}

void PostOps::GetParamsPReLU(SizeType index, PReLUMask &mask) const noexcept(false)
{
    return pImpl->GetParams<Detail::PReLUPostOps>(index, mask);
}

bool operator==(const PostOps &lhs, const PostOps &rhs) noexcept
{
    return (*(lhs.pImpl.get())) == (*(rhs.pImpl.get()));
}

} // KuDNN
