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

#ifndef KUDNN_FAST_JIT_IMPL_F16_F16_MC_HPP
#define KUDNN_FAST_JIT_IMPL_F16_F16_MC_HPP

#include <memory>
#include <vector>
#include <array>
#include <algorithm>
#include <cmath>

#include "kudnn.hpp"

#include "cg_codegen.hpp"

#include "kudnn_jit.hpp"
#include "kernels/reorder/kudnn_reorder_jit.hpp"
#include "vector_wrappers.hpp"

namespace KuDNN {
namespace Detail {

template <typename SrcType, typename DstType>
struct FastReorderKernelCodeGenMC : public KUDNN_JIT_Generator {
        FastReorderKernelCodeGenMC(SizeType inHeightSize,
                                 SizeType inWidthSize,
                                 bool hasAlpha);
protected:
    void GenerateInHeightLoopBody(SizeType currBlockHeightMC, SizeType inWidthSizeMC, bool hasAlpha);
    void GenerateInWidthLoopBody(SizeType currBlockHeight, SizeType currBlockWidth, bool hasAlpha);
    void GenerateLocalIhLoopBody(SizeType inHeight, SizeType currBlockWidth, bool hasAlpha);
    void GenerateLocalIwLoopBody(SizeType inHeight, SizeType inWidth, bool hasAlpha);
    void LoaderFastReorderCore(SizeType inHeight);
    void StorerFastReorderCore(SizeType inWidth);
    void StoreMulFastReorder(SizeType inWidth);

    SizeType blockSize = 256;

    // function input arguments
    ARMCG::XReg B_ptr = x0;
    ARMCG::XReg B_buffer = x1;
    ARMCG::XReg strideIn = x2;
    ARMCG::XReg strideOut = x3;
    ARMCG::SReg alpha = s0;

    // other used registers
    ARMCG::XReg loadOffset = x4;
    ARMCG::XReg storeOffset = x5;
    ARMCG::XReg height = x6;
    ARMCG::XReg width = x7;
    ARMCG::XReg zero = x8;
    ARMCG::XReg localSrcStride = x9;
    ARMCG::XReg localDstStride = x10;
    ARMCG::WReg tmpLDStC0 = w12;
    ARMCG::WReg tmpLDStC1 = w13;
    ARMCG::WReg tmpLDStC2 = w14;
    ARMCG::WReg tmpLDStC3 = w15;
    ARMCG::XReg localIw = x16;
    ARMCG::XReg localIh = x17;
    ARMCG::XReg ih = x18;
    ARMCG::XReg iw = x19;
    ARMCG::XReg maxIh = x20;
    ARMCG::XReg maxIw = x21;
    ARMCG::XReg heightSize = x22;
    ARMCG::XReg widthSize = x23;
    ARMCG::XReg heightCmp = x24;
    ARMCG::XReg widthCmp = x25;
    ARMCG::XReg blockSizeReg = x26;

    ARMCG::WReg tmp = w27;
    ARMCG::WReg intMaxTmp = w28;
    ARMCG::HReg hAlphaTmpOffset = h16;
    ARMCG::SReg alphaTmpOffset = s16;
};

template <typename SrcType, typename DstType>
struct FastReorderImplGenerator<FastReorderKernelCodeGenMC<SrcType, DstType>> : ImplGenerator<ReorderInfo> {
    using CodeGen = FastReorderKernelCodeGenMC<SrcType, DstType>;
    struct FastImpl : ReorderImplBase {
        FastImpl(const ReorderInfo &task, SizeType nThreads);
    protected:
        Shape reorderedDims;
        Shape reorderedStridesSrc;
        Shape reorderedStridesDst;
        void RunSequential(IntType threadID, const void* src, void* dst, float alpha) const noexcept(false) override;
        SizeType GetOptNumThreads() const noexcept override;
        std::array<CodeGenWrapper<CodeGen>, NUM_8> reorderCodegenKernels;
    };
    bool Supports(const ReorderInfo &task) const noexcept override
    {
        VERBOSE_PRINT;
        // going to Add dimsCompressor here
        bool srcMatched = (task.srcInfo.GetType() == Element::MatchType<SrcType>());
        bool dstMatched = (task.dstInfo.GetType() == Element::MatchType<DstType>());
        bool isAXBToABXMC = (task.srcInfo.GetStandardAXBLayout() == task.srcInfo.GetLayout()) &&
            (task.dstInfo.GetStandardABXLayout() == task.dstInfo.GetLayout()) &&
            (task.srcInfo.GetNumDims() > DIM_2) &&
            task.srcInfo.IsDense() && task.dstInfo.IsDense();
        bool isABXToAXBMC = (task.srcInfo.GetStandardABXLayout() == task.srcInfo.GetLayout()) &&
            (task.dstInfo.GetStandardAXBLayout() == task.dstInfo.GetLayout()) &&
            (task.srcInfo.GetNumDims() > DIM_2) &&
            task.srcInfo.IsDense() && task.dstInfo.IsDense();
        bool isABToBAMC = (task.srcInfo.GetLayout() == Layout::AB) &&
            (task.dstInfo.GetLayout() == Layout::BA) &&
            task.srcInfo.IsDense() && task.dstInfo.IsDense();
        bool isBAToABMC = (task.srcInfo.GetLayout() == Layout::BA) &&
            (task.dstInfo.GetLayout() == Layout::AB) &&
            task.srcInfo.IsDense() && task.dstInfo.IsDense();
        if ((srcMatched && dstMatched) && (isAXBToABXMC || isABXToAXBMC || isABToBAMC || isBAToABMC)) {
            return true;
        } else {
            return false;
        }
    }
    std::chrono::duration<float, std::milli> EstimateTime(const ReorderInfo  &task,
                                                          IntType nThreads) const noexcept override
    {
        VERBOSE_PRINT;
        Service::Unused(task, nThreads);
        // going to Add logic to estimate execution time of the task using nThreads
        return std::chrono::duration<float, std::milli>{0.1f};
    }
    std::shared_ptr<ImplBase<ReorderInfo>> GetImpl(const ReorderInfo &task, IntType nThreads) const override
    {
        VERBOSE_PRINT;
        return std::make_shared<FastImpl>(task, nThreads);
    }
};

} // Detail
} // KuDNN

#endif // KUDNN_FAST_JIT_IMPL_F16_F16_MC_HPP
