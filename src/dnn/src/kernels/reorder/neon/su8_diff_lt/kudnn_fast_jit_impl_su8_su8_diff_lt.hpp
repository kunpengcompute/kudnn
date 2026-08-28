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

#ifndef KUDNN_FAST_JIT_IMPL_NEON_SU8_SU8_DIFF_LT_HPP
#define KUDNN_FAST_JIT_IMPL_NEON_SU8_SU8_DIFF_LT_HPP

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
struct FastReorderSU8SU8DiffLtKernelCodeGen : public KUDNN_JIT_Generator {
        FastReorderSU8SU8DiffLtKernelCodeGen(SizeType inHeightSizeNeon,
                                 SizeType inWidthSizeNeon,
                                 bool hasAlpha);
protected:
    void GenerateInHeightLoopBody(SizeType currBlockHeightNeon, SizeType inWidthSizeNeon, bool hasAlpha);
    void GenerateInWidthLoopBody(SizeType currBlockHeight, SizeType currBlockWidth, bool hasAlpha);
    void GenerateLocalIhLoopBody(SizeType inHeight, SizeType currBlockWidth, bool hasAlpha);
    void GenerateLocalIwLoopBody(SizeType inHeight, SizeType inWidth, bool hasAlpha);
    void LoaderFastReorderCore(SizeType vecLen, SizeType inHeight, SizeType inWidth);
    void ZipperFastReorderCore(SizeType vecLen, SizeType logVecLen, SizeType regOffset);
    void MulFastReorder(SizeType vecLen, SizeType logVecLen);
    void StorerFastReorderCore(SizeType vecLen, SizeType logVecLen, SizeType inHeight, SizeType inWidth, bool hasAlpha);
    static int GetRegIndex(int i, int logVecLen, int vecLen)
    {
        return i + (logVecLen % NUM_2) * vecLen;
    }
    static int GetRegIndexInverse(int i, int logVecLen, int vecLen)
    {
        return i + ((logVecLen + NUM_1) % NUM_2) * vecLen;
    }
    void Shr(const ARMCG::VReg4S &vd, const ARMCG::VReg4S &vn, const uint32_t sh);

    SizeType blockSize = 80;

    // function input arguments
    ARMCG::XReg bPtr = x0;
    ARMCG::XReg bBuffer = x1;
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

    ARMCG::XReg tmp = x28;
    ARMCG::WReg tmpW = w27;
    ARMCG::WReg alphaW = w29;
};

template <typename SrcType, typename DstType>
struct FastReorderImplGenerator<FastReorderSU8SU8DiffLtKernelCodeGen<SrcType, DstType>> : ImplGenerator<ReorderInfo> {
    using CodeGen = FastReorderSU8SU8DiffLtKernelCodeGen<SrcType, DstType>;
    struct FastImpl : ReorderImplBase {
        FastImpl(const ReorderInfo &task, SizeType nThreads);
        void RunSequential(IntType threadID, const void* src, void* dst, float alpha) const noexcept(false) override;
        SizeType GetOptNumThreads() const noexcept override;
    protected:
        Shape reorderedDims;
        Shape reorderedStridesSrc;
        Shape reorderedStridesDst;
        std::array<CodeGenWrapper<CodeGen>, NUM_8> reorderCodegenKernels;
    };
    bool Supports(const ReorderInfo &task) const noexcept override
    {
        VERBOSE_PRINT;
#ifdef KUDNNL_920F
            Service::Unused(task);
            return false;
#else // KUDNNL_920F
            // going to Add dimsCompressor here
            bool srcMatched = (task.srcInfo.GetType() == Element::MatchType<SrcType>());
            bool dstMatched = (task.dstInfo.GetType() == Element::MatchType<DstType>());
            bool isAXBToABXNeon = (task.srcInfo.GetStandardAXBLayout() == task.srcInfo.GetLayout()) &&
                (task.dstInfo.GetStandardABXLayout() == task.dstInfo.GetLayout()) &&
                (task.srcInfo.GetNumDims() > DIM_2) &&
                task.srcInfo.IsDense() && task.dstInfo.IsDense();
            bool isABXToAXBNeon = (task.srcInfo.GetStandardABXLayout() == task.srcInfo.GetLayout()) &&
                (task.dstInfo.GetStandardAXBLayout() == task.dstInfo.GetLayout()) &&
                (task.srcInfo.GetNumDims() > DIM_2) &&
                task.srcInfo.IsDense() && task.dstInfo.IsDense();
            bool isABToBANeon = (task.srcInfo.GetLayout() == Layout::AB) &&
                (task.dstInfo.GetLayout() == Layout::BA) &&
                task.srcInfo.IsDense() && task.dstInfo.IsDense();
            bool isBAToABNeon = (task.srcInfo.GetLayout() == Layout::BA) &&
                (task.dstInfo.GetLayout() == Layout::AB) &&
                task.srcInfo.IsDense() && task.dstInfo.IsDense();
            if ((srcMatched && dstMatched) && (isAXBToABXNeon || isABXToAXBNeon || isABToBANeon || isBAToABNeon)) {
                return true;
            } else {
                return false;
            }
#endif // KUDNNL_920F
    }
    std::chrono::duration<float, std::milli> EstimateTime(const ReorderInfo  &task,
                                                            IntType nThreads) const noexcept override
    {
        VERBOSE_PRINT;
        Service::Unused(task, nThreads);
        // going to Add logic to estimate execution time of the task using nThreads
        return std::chrono::duration<float, std::milli>{0.5f};
    }
    std::shared_ptr<ImplBase<ReorderInfo>> GetImpl(const ReorderInfo &task, IntType nThreads) const override
    {
        VERBOSE_PRINT;
        return std::make_shared<FastImpl>(task, nThreads);
    }
};

} // Detail
} // KuDNN

#endif // KUDNN_FAST_JIT_IMPL_NEON_SU8_SU8_DIFF_LT_HPP
