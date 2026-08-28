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

#ifndef KUDNN_SMALL_GEMM_HPP
#define KUDNN_SMALL_GEMM_HPP

#include <algorithm>
#include <memory>
#include <atomic>

#include "arm_sve.h"

#include <omp.h>

#include "kudnn.hpp"

#include "internal_funcs.hpp"
#include "kudnn_jit.hpp"
#include "kudnn_jit_attributes.hpp"
#include "kernels/matmul_jit/kudnn_gemm_helpers.hpp"

namespace KuDNN {
namespace Detail {

using SmallGEMMKernelFuncPtr = void (*)(
        const void *, const void *, void *,
        const void *, MatMulDstFlags, float, float, IntType, IntType, IntType,
        IntType, IntType, IntType, IntType);

struct SmallGEMMParams {
    SmallGEMMParams() : kernelPtr(nullptr), dstFlags(MatMulDstFlags::FILL_C_WITH_ZEROZ),
        m(0), n(0), k(0), srcStride(0), weiStride(0), dstStride(0), biaStride(0) {}
    SmallGEMMParams(SmallGEMMKernelFuncPtr kPtr, MatMulDstFlags dFlags,
        IntType mArg, IntType nArg, IntType kArg,
        IntType sStride, IntType wStride, IntType dStride, IntType bStride) : kernelPtr(kPtr), dstFlags(dFlags),
        m(mArg), n(nArg), k(kArg), srcStride(sStride), weiStride(wStride), dstStride(dStride), biaStride(bStride) {}
    SmallGEMMKernelFuncPtr kernelPtr;
    MatMulDstFlags dstFlags;
    IntType m;
    IntType n;
    IntType k;
    IntType srcStride;
    IntType weiStride;
    IntType dstStride;
    IntType biaStride;
};

struct SmallGEMMImplBase {
    SmallGEMMImplBase(const GemmInfo &gInfo, IntType nThrs) : gemmInfo(gInfo), nThreads(nThrs)
    {
        VERBOSE_PRINT;
    }
    GemmInfo GetTask() const
    {
        VERBOSE_PRINT;
        return gemmInfo;
    }
    IntType GetNThreads() const noexcept
    {
        VERBOSE_PRINT;
        return nThreads;
    }
    virtual SmallGEMMParams GetSmallGEMMParams() const noexcept = 0;
    virtual ~SmallGEMMImplBase()
    {
        VERBOSE_PRINT;
    }
protected:
    GemmInfo gemmInfo;
    IntType nThreads;
};

struct SmallGEMMGenerator {
    virtual bool Supports(const GemmInfo &gemmInfo) const noexcept = 0;
    virtual std::chrono::duration<float, std::milli> EstimateTime(const GemmInfo &gemmInfo,
        IntType nThreads) const noexcept = 0;
    virtual std::shared_ptr<SmallGEMMImplBase> GetImpl(const GemmInfo &gemmInfo, IntType nThreads) const = 0;
    virtual ~SmallGEMMGenerator()
    {
        VERBOSE_PRINT;
    }
};

const std::vector<std::shared_ptr<SmallGEMMGenerator>>& GetSmallGEMMGeneratorList() noexcept;

SmallGEMMParams FindSmallGemmKernel(const GemmInfo &gemmInfo, IntType nThreads);

struct SmallTaskSolutionHandler final {
    SmallTaskSolutionHandler() noexcept = default;
    SmallTaskSolutionHandler(const SmallTaskSolutionHandler &other) = delete;
    SmallTaskSolutionHandler(SmallTaskSolutionHandler &&other) noexcept = delete;
    SmallTaskSolutionHandler& operator=(const SmallTaskSolutionHandler &other) = delete;
    SmallTaskSolutionHandler& operator=(SmallTaskSolutionHandler &&other) noexcept = delete;
    SmallGEMMParams Find(const GemmInfo &gemmInfo, IntType nThreads) noexcept(false)
    {
        VERBOSE_PRINT;
        std::lock_guard<std::mutex> lock(mtx);
        std::pair<GemmInfo, IntType> currTask{gemmInfo, nThreads};
        if (solutions.find(currTask) == solutions.end()) {
            auto&& implKind = ChooseImplGenerator(gemmInfo, nThreads);
            auto impl = implKind->GetImpl(gemmInfo, nThreads);
            Service::Unused(impl);
            solutions.insert({currTask, implKind->GetImpl(gemmInfo, nThreads)});
        }
        return solutions[currTask]->GetSmallGEMMParams();
    }
private:
    std::shared_ptr<SmallGEMMGenerator> ChooseImplGenerator(const GemmInfo &task, IntType nThreads) noexcept
    {
        VERBOSE_PRINT;
        std::shared_ptr<SmallGEMMGenerator> ret;
        std::chrono::duration<float, std::milli> bestTime =
            std::chrono::duration<float, std::milli>{std::numeric_limits<float>::max()};
        for (auto&& implKind : GetSmallGEMMGeneratorList()) {
            if (implKind->Supports(task)) {
                auto currentEstimatedTime = implKind->EstimateTime(task, nThreads);
                if (currentEstimatedTime < bestTime) {
                    bestTime = currentEstimatedTime;
                    ret = implKind;
                }
            }
        }
        return ret;
    }

    std::unordered_map<std::pair<GemmInfo, IntType>, std::shared_ptr<SmallGEMMImplBase>> solutions;
    std::mutex mtx;
};

} // Detail
} // KuDNN

#endif // KUDNN_SMALL_GEMM_HPP
