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

#ifndef KUDNN_FAST_JIT_IMPL_ANY32_ANY32_HPP
#define KUDNN_FAST_JIT_IMPL_ANY32_ANY32_HPP

#include <memory>
#include <vector>
#include <array>
#include <algorithm>
#include <cmath>

#include <arm_sve.h>

#include "kudnn.hpp"

#include "cg_codegen.hpp"

#include "kudnn_jit.hpp"
#include "kernels/reorder/kudnn_reorder_jit.hpp"
#include "vector_wrappers.hpp"

namespace KuDNN {
namespace Detail {

template <typename SrcType, typename DstType>
struct FastReorderKernelCodeGen : public KUDNN_JIT_Generator {
        FastReorderKernelCodeGen(SizeType inHeightSize,
                                 SizeType inWidthSize,
                                 bool hasAlpha);
protected:
    void GenerateInHeightLoopBody(SizeType currBlockHeightVC, SizeType inWidthSizeVC, bool hasAlpha);
    void GenerateInWidthLoopBody(SizeType currBlockHeight, SizeType currBlockWidth, bool hasAlpha);
    void GenerateLocalIhLoopBody(SizeType inHeight, SizeType currBlockWidth, bool hasAlpha);
    void GenerateLocalIwLoopBody(SizeType inHeight, SizeType inWidth, bool hasAlpha);
    void LoaderFastReorderCore(SizeType inHeight);
    void ZipperFastReorderCore(SizeType svlen, SizeType logSvlen, SizeType regOffset);
    void MulFastReorder(SizeType svlen, SizeType logSvlen, bool &needToConvert);
    void StorerFastReorderCore(SizeType svlen, SizeType logSvlen, SizeType inWidth, bool needToConvert, bool hasAlpha);
    static int GetRegIndex(int i, int logSvlen, int svlen)
    {
        return i + (logSvlen % NUM_2) * svlen;
    }
    static int GetRegIndexInverse(int i, int logSvlen, int svlen)
    {
        return i + ((logSvlen + NUM_1) % NUM_2) * svlen;
    }

    #ifdef KUDNNL_920F
        SizeType blockSize = 80;
    #else
        SizeType blockSize = 96;
    #endif

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
    ARMCG::XReg localIw = x11;
    ARMCG::XReg localIh = x12;
    ARMCG::XReg ih = x13;
    ARMCG::XReg iw = x14;
    ARMCG::XReg maxIh = x15;
    ARMCG::XReg maxIw = x16;
    ARMCG::XReg heightSize = x17;
    ARMCG::XReg widthSize = x18;
    ARMCG::XReg heightCmp = x19;
    ARMCG::XReg widthCmp = x20;
    ARMCG::XReg blockSizeReg = x21;

    ARMCG::WReg tmp = w27;
    ARMCG::WReg intMaxTmp = w28;
    ARMCG::HReg hAlphaTmpOffset = h16;
    ARMCG::SReg alphaTmpOffset = s16;
};

template <typename SrcType, typename DstType>
struct FastReorderImplGenerator<FastReorderKernelCodeGen<SrcType, DstType>> : ImplGenerator<ReorderInfo> {
    using CodeGen = FastReorderKernelCodeGen<SrcType, DstType>;
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
        IntType vl = Vector::Length<SrcType>();
        bool fp16check = (vl <= NUM_16);
        bool srcMatched = (task.srcInfo.GetType() == Element::MatchType<SrcType>());
        bool dstMatched = (task.dstInfo.GetType() == Element::MatchType<DstType>());
        bool isAXBToABXVC = (task.srcInfo.GetStandardAXBLayout() == task.srcInfo.GetLayout()) &&
            (task.dstInfo.GetStandardABXLayout() == task.dstInfo.GetLayout()) &&
            (task.srcInfo.GetNumDims() > DIM_2) &&
            task.srcInfo.IsDense() && task.dstInfo.IsDense();
        bool isABXToAXBVC = (task.srcInfo.GetStandardABXLayout() == task.srcInfo.GetLayout()) &&
            (task.dstInfo.GetStandardAXBLayout() == task.dstInfo.GetLayout()) &&
            (task.srcInfo.GetNumDims() > DIM_2) &&
            task.srcInfo.IsDense() && task.dstInfo.IsDense();
        bool isABToBAVC = (task.srcInfo.GetLayout() == Layout::AB) &&
            (task.dstInfo.GetLayout() == Layout::BA) &&
            task.srcInfo.IsDense() && task.dstInfo.IsDense();
        bool isBAToABVC = (task.srcInfo.GetLayout() == Layout::BA) &&
            (task.dstInfo.GetLayout() == Layout::AB) &&
            task.srcInfo.IsDense() && task.dstInfo.IsDense();
        if (fp16check && (srcMatched && dstMatched) && (isAXBToABXVC || isABXToAXBVC || isABToBAVC || isBAToABVC)) {
            return true;
        } else {
            return false;
        }
    }
    std::chrono::duration<float, std::milli> EstimateTime(const ReorderInfo  &task,
                                                          IntType nThreads) const noexcept override
    {
        Service::Unused(task, nThreads);
        // going to Add logic to estimate execution time of the task using nThreads
        return std::chrono::duration<float, std::milli>{0.5f};
    }
    std::shared_ptr<ImplBase<ReorderInfo>> GetImpl(const ReorderInfo &task, IntType nThreads) const override
    {
        return std::make_shared<FastImpl>(task, nThreads);
    }
};

} // Detail
} // KuDNN

#endif // KUDNN_FAST_JIT_IMPL_ANY32_ANY32
