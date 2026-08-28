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
#include <functional>
#include <unordered_map>
#include <arm_neon.h>

#include "kudnn.hpp"
#include "internal_funcs.hpp"
#include "vector_wrappers.hpp"

namespace KuDNN {
namespace Helpers {

template<typename... Args>
bool IsLayoutMatched(Layout currentLayout, Args... layouts)
{
    return ((currentLayout == layouts) || ...);
}

} // Helpers

namespace Detail {

// Acoording to oneDnn docs:
// 1) post-ops supported only for forward propagation
// 2) scales supported only for int8 forward propagation
class InnerProductLayerFWDImpl final {
public:
    InnerProductLayerFWDImpl(const TensorInfo &src,
        const TensorInfo &weights, const TensorInfo &dst,
        const TensorInfo &bias, const Attributes &attributes) noexcept(false) : srcInfo(src),
        weiInfo(weights), dstInfo(dst), biaInfo(bias), attributesInfo(attributes)
    {
        Service::ThrowOnStatus(InnerProductLayerFWD::ValidateInput(src, weights,
            dst, bias, attributes), "Inner Product Layer");
    }

    void Run(const void *src, const void *wei,
            void *dst, const void *bia, const PostOpsDataPtrs &postOpsPtrs,
            float alpha, float beta) noexcept(false)
    {
        bool containsNullPointer = (src == nullptr) || (wei == nullptr) || (dst == nullptr);
        if (containsNullPointer) {
            throw Service::LogicError {"Inner Product Layer: one of input matrices is nullptr"};
        }

        void *srcExec = const_cast<void *>(src);
        void *weiExec = const_cast<void *>(wei);
        void *dstExec = dst;
        void *biaExec = const_cast<void *>(bia);

        TensorInfo srcInfoExec = srcInfo;
        TensorInfo weiInfoExec = weiInfo;
        TensorInfo dstInfoExec = dstInfo;
        TensorInfo biaInfoExec = biaInfo;

        bool forceConvert = false;
#ifdef KUDNNL_920B
        // on 920B, shgemm precision is not satisfied for f16f16f32 forward inner product.
        forceConvert = ((srcInfoExec.GetType() == Element::TypeT::F16) &&
                                          (weiInfoExec.GetType() == Element::TypeT::F16) &&
                                          (dstInfoExec.GetType() == Element::TypeT::F32));
#endif
        BeforeRunConvert(srcInfoExec, weiInfoExec, dstInfoExec, biaInfoExec,
            &srcExec, &weiExec, &dstExec, &biaExec, src, wei, dst, bia, forceConvert);

        SizeType icSrc = Service::ArrayProduct(srcInfoExec.GetDims(), srcInfoExec.GetDims().GetNumDims(), 1);
        SizeType icWei = Service::ArrayProduct(weiInfoExec.GetDims(), weiInfoExec.GetDims().GetNumDims(), 1);
        Gemm gemm(
            TensorInfo{{srcInfoExec.GetDims()[0], icSrc}, srcInfoExec.GetType(),
                (srcInfoExec.GetLayout() == srcInfoExec.GetStandardABXLayout()) ? Layout::AB : Layout::BA},
            TensorInfo{{icWei, weiInfoExec.GetDims()[0]}, weiInfoExec.GetType(),
                (weiInfoExec.GetLayout() == weiInfoExec.GetStandardABXLayout()) ? Layout::BA : Layout::AB},
            dstInfoExec,
            ((biaInfoExec.GetNumDims() == 1) ?
                TensorInfo{{1, biaInfoExec.GetDims()[0]}, biaInfoExec.GetType(), Layout::AB} :
                TensorInfo{{biaInfoExec.GetDims()[0], biaInfoExec.GetDims()[1]},
                    biaInfoExec.GetType(), biaInfoExec.GetLayout()}),
            attributesInfo);
        gemm.Run(srcExec, weiExec, dstExec, biaExec, postOpsPtrs, alpha, beta);

        AfterRunConvert(dstInfoExec, &srcExec, &weiExec, &dstExec, &biaExec, dst, forceConvert);
    }

private:
    TensorInfo srcInfo;
    TensorInfo weiInfo;
    TensorInfo dstInfo;
    TensorInfo biaInfo;
    Attributes attributesInfo;

    void BeforeRunConvert(TensorInfo &srcInfoExec, TensorInfo &weiInfoExec, TensorInfo &dstInfoExec,
        TensorInfo &biaInfoExec, void **srcExec, void **weiExec,
        void **dstExec, void **biaExec, const void *src, const void *wei, void *dst,
        const void *bia, bool forceConvert) noexcept(false)
    {
        Element::Type convertedPrecision = Element::TypeT::F32;
        bool isSrcABX = (srcInfoExec.GetLayout() == srcInfoExec.GetStandardABXLayout());
        bool isSrcBXA = (srcInfoExec.GetLayout() == srcInfoExec.GetStandardBXALayout());
        if ((!isSrcABX && !isSrcBXA) || (forceConvert && (srcInfoExec.GetType() != convertedPrecision))) {
            TensorInfo newSrcInfo = {srcInfoExec.GetDims(), convertedPrecision,
                srcInfoExec.GetStandardABXLayout()};
            ReorderLayer rSrc(srcInfoExec, newSrcInfo);
            *srcExec = Service::AlignedAlloc(srcInfoExec.GetTotalTensorSize() * convertedPrecision.GetSize());
            rSrc.Run(src, *srcExec);
            srcInfoExec = newSrcInfo;
        }
        bool isWeiABX = (weiInfoExec.GetLayout() == weiInfoExec.GetStandardABXLayout());
        bool isWeiBXA = (weiInfoExec.GetLayout() == weiInfoExec.GetStandardBXALayout());
        if ((!isWeiABX && !isWeiBXA) || (forceConvert && (weiInfoExec.GetType() != convertedPrecision))) {
            TensorInfo newWeiInfo = {weiInfoExec.GetDims(), convertedPrecision,
                weiInfoExec.GetStandardABXLayout()};
            ReorderLayer rWei(weiInfoExec, newWeiInfo);
            *weiExec = Service::AlignedAlloc(weiInfoExec.GetTotalTensorSize() * convertedPrecision.GetSize());
            rWei.Run(wei, *weiExec);
            weiInfoExec = newWeiInfo;
        }
        if (forceConvert && (dstInfoExec.GetType() != convertedPrecision)) {
            TensorInfo newDstInfo = {dstInfoExec.GetDims(), convertedPrecision,
                dstInfoExec.GetStandardABXLayout()};
            ReorderLayer rDst(dstInfoExec, newDstInfo);
            *dstExec = Service::AlignedAlloc(dstInfoExec.GetTotalTensorSize() * convertedPrecision.GetSize());
            rDst.Run(dst, *dstExec);
            dstInfoExec = newDstInfo;
        }
        if (bia && (forceConvert && (biaInfoExec.GetType() != convertedPrecision))) {
            TensorInfo newBiaInfo = {biaInfoExec.GetDims(), convertedPrecision,
                biaInfoExec.GetStandardABXLayout()};
            ReorderLayer rBia(biaInfoExec, newBiaInfo);
            *biaExec = Service::AlignedAlloc(biaInfoExec.GetTotalTensorSize() * convertedPrecision.GetSize());
            rBia.Run(bia, *biaExec);
            biaInfoExec = newBiaInfo;
        }
    }

    void AfterRunConvert(TensorInfo &dstInfoExec,
        void **srcExec, void **weiExec, void **dstExec, void **biaExec,
        void *dst, bool forceConvert) noexcept(false)
    {
        Element::Type convertedPrecision = Element::TypeT::F32;
        bool isSrcABX = (srcInfo.GetLayout() == srcInfo.GetStandardABXLayout());
        bool isSrcBXA = (srcInfo.GetLayout() == srcInfo.GetStandardBXALayout());
        if ((!isSrcABX && !isSrcBXA) || (forceConvert && (srcInfo.GetType() != convertedPrecision))) {
            Service::Deallocate(*srcExec);
        }
        bool isWeiABX = (weiInfo.GetLayout() == weiInfo.GetStandardABXLayout());
        bool isWeiBXA = (weiInfo.GetLayout() == weiInfo.GetStandardBXALayout());
        if ((!isWeiABX && !isWeiBXA) || (forceConvert && (weiInfo.GetType() != convertedPrecision))) {
            Service::Deallocate(*weiExec);
        }
        if (forceConvert && (dstInfo.GetType() != convertedPrecision)) {
            ReorderLayer rDst(dstInfoExec, dstInfo);
            rDst.Run(*dstExec, dst);
            Service::Deallocate(*dstExec);
        }
        if (*biaExec && (forceConvert && (biaInfo.GetType() != convertedPrecision))) {
            Service::Deallocate(*biaExec);
        }
    }
};

class InnerProductLayerBWDDataImpl final {
public:
    InnerProductLayerBWDDataImpl(const TensorInfo &diffDst, const TensorInfo &weights,
        const TensorInfo &diffSrc) noexcept(false) : diffDstInfo(diffDst),
        weiInfo(weights), diffSrcInfo(diffSrc)
    {
        Service::ThrowOnStatus(InnerProductLayerBWDData::ValidateInput(diffDst, weights,
            diffSrc), "Inner Product Layer");
    }

    void Run(const void *diffDst, const void *wei, void *diffSrc) noexcept(false)
    {
        if ((diffDst == nullptr) || (wei == nullptr) || (diffSrc == nullptr)) {
            throw Service::LogicError {"Inner Product Layer: one of input matrices is nullptr"};
        }

        void *diffDstExec = const_cast<void *>(diffDst);
        void *weiExec = const_cast<void *>(wei);
        void *diffSrcExec = diffSrc;

        TensorInfo diffDstInfoExec = diffDstInfo;
        TensorInfo weiInfoExec = weiInfo;
        TensorInfo diffSrcInfoExec = diffSrcInfo;

        bool isDiffSrcABX = (diffSrcInfoExec.GetLayout() == diffSrcInfoExec.GetStandardABXLayout());
        bool isDiffSrcAXB = (diffSrcInfoExec.GetLayout() == diffSrcInfoExec.GetStandardAXBLayout());
        bool isWeiABX = (weiInfoExec.GetLayout() == weiInfoExec.GetStandardABXLayout());
        bool isWeiAXB = (weiInfoExec.GetLayout() == weiInfoExec.GetStandardAXBLayout());
        bool needToConvert = (diffSrcInfoExec.GetNumDims() > NUM_2) &&
                             ((isDiffSrcABX && isDiffSrcAXB) || (isWeiABX && isWeiAXB) ||
                              (isDiffSrcABX != isWeiABX) || (isDiffSrcAXB != isWeiAXB));
#ifdef KUDNNL_920B
        // on 920B, shgemm precision is not satisfied for f32f16f16 backward inner product.
        needToConvert = needToConvert || ((diffDstInfoExec.GetType() == Element::TypeT::F16) &&
                                          (weiInfoExec.GetType() == Element::TypeT::F16) &&
                                          (diffSrcInfoExec.GetType() == Element::TypeT::F32));
#endif
        if (needToConvert) {
            BeforeRunConvert(diffDstInfoExec, weiInfoExec, diffSrcInfoExec, isDiffSrcABX,
                isWeiABX, &diffDstExec, &weiExec, &diffSrcExec, diffDst, wei, diffSrc);
        }

        SizeType icDiffSrc =
            Service::ArrayProduct(diffSrcInfoExec.GetDims(), diffSrcInfoExec.GetDims().GetNumDims(), 1);
        SizeType icWei = Service::ArrayProduct(weiInfoExec.GetDims(), weiInfoExec.GetDims().GetNumDims(), 1);

        Gemm gemm(
            diffDstInfoExec,
            TensorInfo{{weiInfoExec.GetDims()[0], icWei}, weiInfoExec.GetType(),
                (weiInfoExec.GetNumDims() > NUM_2) ? Layout::AB : weiInfoExec.GetLayout()},
            TensorInfo{{diffSrcInfoExec.GetDims()[0], icDiffSrc},
                diffSrcInfoExec.GetType(), (diffSrcInfoExec.GetNumDims() > NUM_2) ?
                Layout::AB : diffSrcInfoExec.GetLayout()},
            TensorInfo{{0, 0}, diffSrcInfoExec.GetType(), Layout::AB});
        gemm.Run(diffDstExec, weiExec, diffSrcExec);

        if (needToConvert) {
            AfterRunConvert(&diffDstExec, &weiExec, &diffSrcExec, diffSrcInfoExec, diffSrc);
        }
    }

private:
    TensorInfo diffDstInfo;
    TensorInfo weiInfo;
    TensorInfo diffSrcInfo;

    void BeforeRunConvert(TensorInfo &diffDstInfoExec, TensorInfo &weiInfoExec, TensorInfo &diffSrcInfoExec,
        bool isDiffSrcABX, bool isWeiABX, void **diffDstExec, void **weiExec, void **diffSrcExec,
        const void *diffDst, const void *wei, void *diffSrc) noexcept(false)
    {
        Element::Type convertedPrecision = Element::TypeT::F32;

        bool diffDstExecConvertedCond = diffDstInfoExec.GetType() != convertedPrecision;
        if (diffDstExecConvertedCond) {
            TensorInfo newDiffDstInfo = {diffDstInfoExec.GetDims(), convertedPrecision,
                diffDstInfoExec.GetStandardABXLayout()};
            ReorderLayer rDiffDst(diffDstInfoExec, newDiffDstInfo);
            *diffDstExec = Service::AlignedAlloc(
                diffDstInfoExec.GetTotalTensorSize() * convertedPrecision.GetSize());
            rDiffDst.Run(diffDst, *diffDstExec);
            diffDstInfoExec = newDiffDstInfo;
        }

        bool weiExecConvertedCond = weiInfoExec.GetType() != convertedPrecision;
        if (!isWeiABX || weiExecConvertedCond) {
            TensorInfo newWeiInfo = {weiInfoExec.GetDims(), convertedPrecision,
                weiInfoExec.GetStandardABXLayout()};
            ReorderLayer rWei(weiInfoExec, newWeiInfo);
            *weiExec = Service::AlignedAlloc(weiInfoExec.GetTotalTensorSize() * convertedPrecision.GetSize());
            rWei.Run(wei, *weiExec);
            weiInfoExec = newWeiInfo;
        }

        if (!isDiffSrcABX || diffSrcInfoExec.GetType() != convertedPrecision) {
            TensorInfo newDiffSrcInfo = {diffSrcInfoExec.GetDims(), convertedPrecision,
                diffSrcInfoExec.GetStandardABXLayout()};
            ReorderLayer rDiffSrc(diffSrcInfoExec, newDiffSrcInfo);
            *diffSrcExec = Service::AlignedAlloc(
                diffSrcInfoExec.GetTotalTensorSize() * convertedPrecision.GetSize());
            rDiffSrc.Run(diffSrc, *diffSrcExec);
            diffSrcInfoExec = newDiffSrcInfo;
        }
    }

    void AfterRunConvert(void **diffDstExec, void **weiExec, void **diffSrcExec,
        TensorInfo &diffSrcInfoExec, void *diffSrc)
    {
        Element::Type convertedPrecision = Element::TypeT::F32;

        bool diffDstInfoConvertedCond = diffDstInfo.GetType() != convertedPrecision;
        if (diffDstInfoConvertedCond) {
            Service::Deallocate(*diffDstExec);
        }

        bool weiInfoConvertedCond = weiInfo.GetType() != convertedPrecision;
        if (!(weiInfo.GetLayout() == weiInfo.GetStandardABXLayout()) || weiInfoConvertedCond) {
            Service::Deallocate(*weiExec);
        }

        if (!(diffSrcInfo.GetLayout() == diffSrcInfo.GetStandardABXLayout()) ||
            diffSrcInfo.GetType() != convertedPrecision) {
            ReorderLayer rDiffSrc(diffSrcInfoExec, diffSrcInfo);
            rDiffSrc.Run(*diffSrcExec, diffSrc);
            Service::Deallocate(*diffSrcExec);
        }
    }
};

class InnerProductLayerBWDWeightsImpl final {
public:
    InnerProductLayerBWDWeightsImpl(const TensorInfo &diffDst, const TensorInfo &src,
        const TensorInfo &diffWeights, const TensorInfo &diffBias) noexcept(false)
        : diffDstInfo(diffDst), srcInfo(src), diffWeiInfo(diffWeights),
        diffBiaInfo(diffBias)
    {
        Service::ThrowOnStatus(InnerProductLayerBWDWeights::ValidateInput(diffDst, src,
            diffWeights, diffBias), "Inner Product Layer");
    }

    template<typename DstType>
    void CalculateDiffBias(const void *diffDstExec, void *diffBiaExec, SizeType ocSize,
        SizeType mbSize, bool isLayoutAB)
    {
        if (isLayoutAB) {
            CalculateDiffBiasAB<DstType>(static_cast<const DstType *>(diffDstExec),
                                         static_cast<DstType *>(diffBiaExec), ocSize, mbSize);
        } else {
            CalculateDiffBiasBA<DstType>(static_cast<const DstType *>(diffDstExec),
                                         static_cast<DstType *>(diffBiaExec), ocSize, mbSize);
        }
    }
    using CalculateDiffBiasFunc = std::function<void(const void *, void *, SizeType, SizeType)>;
    std::unordered_map<Element::TypeT, CalculateDiffBiasFunc> layoutABFuncMap = {
        {Element::TypeT::F32, [this](const void *diffDstExec, void *diffBiaExec, SizeType ocSize, SizeType mbSize) {
            this->CalculateDiffBias<float>(diffDstExec, diffBiaExec, ocSize, mbSize, true);
        }},
        {Element::TypeT::F16, [this](const void *diffDstExec, void *diffBiaExec, SizeType ocSize, SizeType mbSize) {
            this->CalculateDiffBias<__fp16>(diffDstExec, diffBiaExec, ocSize, mbSize, true);
        }},
        {Element::TypeT::BF16, [this](const void *diffDstExec, void *diffBiaExec, SizeType ocSize, SizeType mbSize) {
            this->CalculateDiffBias<__bf16>(diffDstExec, diffBiaExec, ocSize, mbSize, true);
        }},
    };
    std::unordered_map<Element::TypeT, CalculateDiffBiasFunc> layoutBAFuncMap = {
        {Element::TypeT::F32, [this](const void *diffDstExec, void *diffBiaExec, SizeType ocSize, SizeType mbSize) {
            this->CalculateDiffBias<float>(diffDstExec, diffBiaExec, ocSize, mbSize, false);
        }},
        {Element::TypeT::F16, [this](const void *diffDstExec, void *diffBiaExec, SizeType ocSize, SizeType mbSize) {
            this->CalculateDiffBias<__fp16>(diffDstExec, diffBiaExec, ocSize, mbSize, false);
        }},
        {Element::TypeT::BF16, [this](const void *diffDstExec, void *diffBiaExec, SizeType ocSize, SizeType mbSize) {
            this->CalculateDiffBias<__bf16>(diffDstExec, diffBiaExec, ocSize, mbSize, false);
        }},
    };
    void ProcessDiffBias(const void *diffDstExec, void *diffBiaExec, const TensorInfo &diffDstInfoExec,
        const TensorInfo &srcInfoExec)
    {
        if (!diffBiaExec) {
            return;
        }

        SizeType ocSize = diffDstInfoExec.GetDims()[1];
        SizeType mbSize = srcInfoExec.GetDims()[0];
        bool isLayoutAB = (diffDstInfoExec.GetLayout() == Layout::AB);
        auto type = static_cast<Element::TypeT>(diffDstInfoExec.GetType());

        auto func = isLayoutAB ? layoutABFuncMap[type] : layoutBAFuncMap[type];
        if (func) {
            func(diffDstExec, diffBiaExec, ocSize, mbSize);
        }
    }

    void Run(const void *diffDst, const void *src,
            void *diffWeights, void *diffBias) noexcept(false)
    {
        if ((diffDst == nullptr) || (src == nullptr) || (diffWeights == nullptr)) {
            throw Service::LogicError {"Inner Product Layer: one of input matrices is nullptr"};
        }

        void *diffDstExec = const_cast<void *>(diffDst);
        void *srcExec = const_cast<void *>(src);
        void *diffWeiExec = const_cast<void *>(diffWeights);
        void *diffBiaExec = diffBias;

        TensorInfo diffDstInfoExec = diffDstInfo;
        TensorInfo srcInfoExec = srcInfo;
        TensorInfo diffWeiInfoExec = diffWeiInfo;
        TensorInfo diffBiaInfoExec = diffBiaInfo;

        bool isSrcABX = (srcInfoExec.GetLayout() == srcInfoExec.GetStandardABXLayout());
        bool isSrcAXB = (srcInfoExec.GetLayout() == srcInfoExec.GetStandardAXBLayout());
        bool isDiffWeiABX = (diffWeiInfoExec.GetLayout() == diffWeiInfoExec.GetStandardABXLayout());
        bool isDiffWeiAXB = (diffWeiInfoExec.GetLayout() == diffWeiInfoExec.GetStandardAXBLayout());
        bool needToConvert = (diffBiaExec && (diffDstInfoExec.GetType() == Element::TypeT::BF16) &&
                               (srcInfoExec.GetType() == Element::TypeT::BF16) &&
                               (diffWeiInfoExec.GetType() == Element::TypeT::F32)) ||
                              ((srcInfoExec.GetNumDims() > NUM_2) &&
                               ((isSrcABX != isDiffWeiABX) || (isSrcAXB != isDiffWeiAXB)));
#ifdef KUDNNL_920B
        // on 920B, shgemm precision is not satisfied for f16f32f16 backward weights update inner product.
        needToConvert = needToConvert || ((diffDstInfoExec.GetType() == Element::TypeT::F16) &&
                                          (srcInfoExec.GetType() == Element::TypeT::F16) &&
                                          (diffWeiInfoExec.GetType() == Element::TypeT::F32));
#endif
        Element::Type convertedPrecision = Element::TypeT::F32;

        bool diffDstExecConvertedCond;
        bool srcExecConvertedCond;
        diffDstExecConvertedCond = (diffDstInfoExec.GetType() != convertedPrecision);
        srcExecConvertedCond = !isSrcABX || (srcInfoExec.GetType() != convertedPrecision);

        if (needToConvert) {
            if (diffDstExecConvertedCond) {
                TensorInfo newDiffDstInfo = {diffDstInfoExec.GetDims(), convertedPrecision,
                    diffDstInfoExec.GetStandardABXLayout()};
                ReorderLayer rDiffDst(diffDstInfoExec, newDiffDstInfo);
                diffDstExec = Service::AlignedAlloc(
                    diffDstInfoExec.GetTotalTensorSize() * convertedPrecision.GetSize());
                rDiffDst.Run(diffDst, diffDstExec);
                diffDstInfoExec = newDiffDstInfo;
            }
            if (srcExecConvertedCond) {
                TensorInfo newWeiInfo = {srcInfoExec.GetDims(), convertedPrecision,
                    srcInfoExec.GetStandardABXLayout()};
                ReorderLayer rSrc(srcInfoExec, newWeiInfo);
                srcExec = Service::AlignedAlloc(
                    srcInfoExec.GetTotalTensorSize() * convertedPrecision.GetSize());
                rSrc.Run(src, srcExec);
                srcInfoExec = newWeiInfo;
            }
            if (!isDiffWeiABX || diffWeiInfoExec.GetType() != convertedPrecision) {
                TensorInfo newDiffSrcInfo = {diffWeiInfoExec.GetDims(), convertedPrecision,
                    diffWeiInfoExec.GetStandardABXLayout()};
                ReorderLayer rDiffWei(diffWeiInfoExec, newDiffSrcInfo);
                diffWeiExec = Service::AlignedAlloc(
                    diffWeiInfoExec.GetTotalTensorSize() * convertedPrecision.GetSize());
                rDiffWei.Run(diffWeights, diffWeiExec);
                diffWeiInfoExec = newDiffSrcInfo;
            }
        }

        SizeType ic = Service::ArrayProduct(srcInfoExec.GetDims(), srcInfoExec.GetDims().GetNumDims(), 1);

        Gemm gemm(
            TensorInfo{{diffDstInfoExec.GetDims()[1], diffDstInfoExec.GetDims()[0]},
                diffDstInfoExec.GetType(), (diffDstInfoExec.GetLayout() == Layout::AB) ? Layout::BA : Layout::AB},
            TensorInfo{{srcInfoExec.GetDims()[0], ic}, srcInfoExec.GetType(),
                (srcInfoExec.GetNumDims() > NUM_2) ? Layout::AB : srcInfoExec.GetLayout()},
            TensorInfo{{diffWeiInfoExec.GetDims()[0], ic}, diffWeiInfoExec.GetType(),
                (diffWeiInfoExec.GetNumDims() > NUM_2) ? Layout::AB : diffWeiInfoExec.GetLayout()},
            TensorInfo{{0, 0}, diffWeiInfoExec.GetType(), Layout::AB});
        gemm.Run(diffDstExec, srcExec, diffWeiExec);

        if (diffBiaExec && (diffBiaInfoExec.GetType() != diffDstInfoExec.GetType())) {
            TensorInfo newBiaInfo = {diffBiaInfoExec.GetDims(), diffDstInfoExec.GetType(),
                diffBiaInfoExec.GetStandardABXLayout()};
            ReorderLayer rBia(diffBiaInfoExec, newBiaInfo);
            diffBiaExec = Service::AlignedAlloc(
                diffBiaInfoExec.GetTotalTensorSize() * diffDstInfoExec.GetType().GetSize());
            rBia.Run(diffBias, diffBiaExec);
            diffBiaInfoExec = newBiaInfo;
        }

        ProcessDiffBias(diffDstExec, diffBiaExec, diffDstInfoExec, srcInfoExec);

        if (needToConvert) {
            if (diffDstExecConvertedCond) {
                Service::Deallocate(diffDstExec);
            }
            if (srcExecConvertedCond) {
                Service::Deallocate(srcExec);
            }
            if (!(diffWeiInfo.GetLayout() == diffWeiInfo.GetStandardABXLayout()) ||
                diffWeiInfo.GetType() != convertedPrecision) {
                ReorderLayer rDiffWei(diffWeiInfoExec, diffWeiInfo);
                rDiffWei.Run(diffWeiExec, diffWeights);
                Service::Deallocate(diffWeiExec);
            }
        }
        if (diffBias && (diffBiaInfo.GetType() != diffDstInfoExec.GetType())) {
            ReorderLayer rDiffBia(diffBiaInfoExec, diffBiaInfo);
            rDiffBia.Run(diffBiaExec, diffBias);
            Service::Deallocate(diffBiaExec);
        }
    }

private:
    TensorInfo diffDstInfo;
    TensorInfo srcInfo;
    TensorInfo diffWeiInfo;
    TensorInfo diffBiaInfo;

    template<typename DstType>
    void CalculateDiffBiasAB(const DstType *diffDst, DstType *diffBiass, SizeType ocSize, SizeType mbSize)
        const noexcept
    {
        for (SizeType oc = 0; oc < ocSize; ++oc) {
            float dB = 0.0f;
            for (SizeType mb = 0; mb < mbSize; ++mb) {
                dB += Vector::Convert<DstType, float>::Do(diffDst[mb * ocSize + oc]);
            }
            diffBiass[oc] = Vector::Convert<float, DstType>::Do(dB);
        }
    }
    template<typename DstType>
    void CalculateDiffBiasBA(const DstType *diffDst, DstType *diffBiass, SizeType ocSize, SizeType mbSize)
        const noexcept
    {
        for (SizeType oc = 0; oc < ocSize; ++oc) {
            float dB = 0.0f;
            for (SizeType mb = 0; mb < mbSize; ++mb) {
                dB += Vector::Convert<DstType, float>::Do(diffDst[mb + oc * mbSize]);
            }
            diffBiass[oc] = Vector::Convert<float, DstType>::Do(dB);
        }
    }
};

} // Detail

// FWD

InnerProductLayerFWD::~InnerProductLayerFWD() noexcept {}

InnerProductLayerFWD::InnerProductLayerFWD(const TensorInfo &src, const TensorInfo &weights, const TensorInfo &dst,
    const TensorInfo &bias) noexcept(false)
    : pImpl(new Detail::InnerProductLayerFWDImpl(src, weights, dst, bias, Attributes{})) {}

InnerProductLayerFWD::InnerProductLayerFWD(const TensorInfo &src, const TensorInfo &weights, const TensorInfo &dst,
    const TensorInfo &bias, const Attributes &attributes) noexcept(false)
    : pImpl(new Detail::InnerProductLayerFWDImpl(src, weights, dst, bias, attributes)) {}

InnerProductLayerFWD::InnerProductLayerFWD(const InnerProductLayerFWD &other) noexcept(false)
    : pImpl(new Detail::InnerProductLayerFWDImpl{*(other.pImpl.get())}) {}

InnerProductLayerFWD::InnerProductLayerFWD(InnerProductLayerFWD &&other) noexcept
    : pImpl(std::move(other.pImpl)) {}

InnerProductLayerFWD& InnerProductLayerFWD::operator=(const InnerProductLayerFWD &other) noexcept(false)
{
    if (this != &other) {
        this->pImpl.reset(new Detail::InnerProductLayerFWDImpl{*(other.pImpl.get())});
    }
    return *this;
}

InnerProductLayerFWD& InnerProductLayerFWD::operator=(InnerProductLayerFWD &&other) noexcept
{
    if (this != &other) {
        this->pImpl = std::move(other.pImpl);
    }
    return *this;
}

void InnerProductLayerFWD::Run(const void *src, const void *wei,
    void *dst, const void *bia, float alpha, float beta) const noexcept(false)
{
    pImpl->Run(src, wei, dst, bia, {}, alpha, beta);
}

void InnerProductLayerFWD::Run(const void *src, const void *wei,
    void *dst, const void *bia, const PostOpsDataPtrs &postOpsPtrs,
    float alpha, float beta) const noexcept(false)
{
    pImpl->Run(src, wei, dst, bia, postOpsPtrs, alpha, beta);
}

Status InnerProductLayerFWD::ValidateInput(const TensorInfo &src, const TensorInfo &weights, const TensorInfo &dst,
    const TensorInfo &bias) noexcept
{
    return InnerProductLayerFWD::ValidateInput(src, weights, dst, bias, Attributes{});
}

Status InnerProductLayerFWD::ValidateInput(const TensorInfo &src, const TensorInfo &weights, const TensorInfo &dst,
    const TensorInfo &bias, const Attributes &attributes) noexcept
{
    if (!src.IsDense() || !weights.IsDense() || !dst.IsDense() || !bias.IsDense()) {
        return Status::UNSUPPORTED;
    }
    if (src.GetNumDims() != weights.GetNumDims()) {
        return Status::BAD_ARGUMENTS;
    }
    if (dst.GetNumDims() != DIM_2) {
        return Status::BAD_ARGUMENTS;
    }
    if (bias.GetNumDims() > DIM_2) {
        return Status::BAD_ARGUMENTS;
    }
    if ((src.GetType() == Element::TypeT::S32) || (weights.GetType() == Element::TypeT::S32)) {
        return Status::UNSUPPORTED;
    }
    for (SizeType i = 1; i < src.GetDims().GetNumDims(); ++i) {
        if (src.GetDims()[i] != weights.GetDims()[i]) {
            return Status::BAD_ARGUMENTS;
        }
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

// BWD_D

InnerProductLayerBWDData::~InnerProductLayerBWDData() noexcept {}

InnerProductLayerBWDData::InnerProductLayerBWDData(const TensorInfo &diffDst,
    const TensorInfo &weights, const TensorInfo &diffSrc) noexcept(false)
    : pImpl(new Detail::InnerProductLayerBWDDataImpl(diffDst, weights, diffSrc)) {}

InnerProductLayerBWDData::InnerProductLayerBWDData(const InnerProductLayerBWDData &other) noexcept(false)
    : pImpl(new Detail::InnerProductLayerBWDDataImpl{*(other.pImpl.get())}) {}

InnerProductLayerBWDData::InnerProductLayerBWDData(InnerProductLayerBWDData &&other) noexcept
    : pImpl(std::move(other.pImpl)) {}

InnerProductLayerBWDData& InnerProductLayerBWDData::operator=(const InnerProductLayerBWDData &other) noexcept(false)
{
    if (this != &other) {
        this->pImpl.reset(new Detail::InnerProductLayerBWDDataImpl{*(other.pImpl.get())});
    }
    return *this;
}

InnerProductLayerBWDData& InnerProductLayerBWDData::operator=(InnerProductLayerBWDData &&other) noexcept
{
    if (this != &other) {
        this->pImpl = std::move(other.pImpl);
    }
    return *this;
}

void InnerProductLayerBWDData::Run(const void *diffDst, const void *wei,
    void *diffSrc) const noexcept(false)
{
    pImpl->Run(diffDst, wei, diffSrc);
}

Status InnerProductLayerBWDData::ValidateInput(const TensorInfo &diffDst, const TensorInfo &weights,
    const TensorInfo &diffSrc) noexcept
{
    if (!diffDst.IsDense() || !weights.IsDense() || !diffSrc.IsDense()) {
        return Status::UNSUPPORTED;
    }
    if (diffSrc.GetNumDims() != weights.GetNumDims()) {
        return Status::BAD_ARGUMENTS;
    }
    if (diffDst.GetNumDims() != DIM_2) {
        return Status::BAD_ARGUMENTS;
    }
    for (SizeType i = 1; i < diffSrc.GetDims().GetNumDims(); ++i) {
        if (diffSrc.GetDims()[i] != weights.GetDims()[i]) {
            return Status::BAD_ARGUMENTS;
        }
    }
    if ((diffDst.GetType() == Element::TypeT::S32) || (weights.GetType() == Element::TypeT::S32)) {
        return Status::UNSUPPORTED;
    }

    return Status::SUCCESS;
}

// BWD_W

InnerProductLayerBWDWeights::~InnerProductLayerBWDWeights() noexcept {}

InnerProductLayerBWDWeights::InnerProductLayerBWDWeights(const TensorInfo &diffDst, const TensorInfo &src,
    const TensorInfo &diffWeights, const TensorInfo &diffBias) noexcept(false)
    : pImpl(new Detail::InnerProductLayerBWDWeightsImpl(diffDst, src, diffWeights, diffBias)) {}

InnerProductLayerBWDWeights::InnerProductLayerBWDWeights(const InnerProductLayerBWDWeights &other) noexcept(false)
    : pImpl(new Detail::InnerProductLayerBWDWeightsImpl{*(other.pImpl.get())}) {}

InnerProductLayerBWDWeights::InnerProductLayerBWDWeights(InnerProductLayerBWDWeights &&other) noexcept
    : pImpl(std::move(other.pImpl)) {}

InnerProductLayerBWDWeights& InnerProductLayerBWDWeights::operator=(const InnerProductLayerBWDWeights &other)
    noexcept(false)
{
    if (this != &other) {
        this->pImpl.reset(new Detail::InnerProductLayerBWDWeightsImpl{*(other.pImpl.get())});
    }
    return *this;
}

InnerProductLayerBWDWeights& InnerProductLayerBWDWeights::operator=(InnerProductLayerBWDWeights &&other) noexcept
{
    if (this != &other) {
        this->pImpl = std::move(other.pImpl);
    }
    return *this;
}

void InnerProductLayerBWDWeights::Run(const void *diffDst, const void *src,
    void *diffWeights, void *diffBias) const noexcept(false)
{
    pImpl->Run(diffDst, src, diffWeights, diffBias);
}

Status InnerProductLayerBWDWeights::ValidateInput(const TensorInfo &diffDst, const TensorInfo &src,
    const TensorInfo &diffWeights, const TensorInfo &diffBias) noexcept
{
    if (!diffDst.IsDense() || !src.IsDense() || !diffWeights.IsDense() || !diffBias.IsDense()) {
        return Status::UNSUPPORTED;
    }
    if (diffWeights.GetNumDims() != src.GetNumDims()) {
        return Status::BAD_ARGUMENTS;
    }
    if (diffDst.GetNumDims() != DIM_2) {
        return Status::BAD_ARGUMENTS;
    }
    if ((diffDst.GetType() == Element::TypeT::S32) || (src.GetType() == Element::TypeT::S32)) {
        return Status::UNSUPPORTED;
    }
    for (SizeType i = 1; i < diffWeights.GetDims().GetNumDims(); ++i) {
        if (src.GetDims()[i] != diffWeights.GetDims()[i]) {
            return Status::BAD_ARGUMENTS;
        }
    }

    SizeType icDiffDst = Service::ArrayProduct(diffDst.GetDims(), diffDst.GetDims().GetNumDims(), 1);
    if ((diffBias.GetTotalTensorSize() != 0) && (diffBias.GetDims()[0] != icDiffDst)) {
        return Status::BAD_ARGUMENTS;
    }

    return Status::SUCCESS;
}

} // KuDNN
