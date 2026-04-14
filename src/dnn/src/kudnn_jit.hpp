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

#ifndef KUDNN_JIT_HPP
#define KUDNN_JIT_HPP

#include <vector>
#include <memory>
#include <limits>

#include "kudnn.hpp"

#include "cg_codegen.hpp"

#define VERBOSE_PRINT

namespace KuDNN {
namespace Detail {

struct KUDNN_JIT_Generator : public ARMCG::CodeGenerator {
    KUDNN_JIT_Generator() : ARMCG::CodeGenerator(NUM_1024 * NUM_1024) // , ARMCG::AutoGrow
    {
        VERBOSE_PRINT;
    }
protected:
    void Preamble()
    {
        VERBOSE_PRINT;
        // store x29 and x30 to stack for ABI requirement and decrement stack pointer by 16 bytes (2x64-bit D registers)
        Stp(ARMCG::XReg(29), ARMCG::XReg(30), PrePtr(sp, -16));
        // store callee-saved X registers (x19-x28)
        for (int i = NUM_19; i < NUM_28; i += NUM_2) {
            Stp(ARMCG::XReg(i), ARMCG::XReg(i + 1),
                PrePtr(sp, -16)); // 16 is a size of 2x64-bit X registers
        }
        // store callee-saved vector registers (d8-d15)
        for (int i = NUM_8; i < NUM_15; i += NUM_2) {
            Stp(ARMCG::DReg(i), ARMCG::DReg(i + 1),
                PrePtr(sp, -16)); // 16 is a size of 2x64-bit D registers
        }
    }
    void Postamble()
    {
        VERBOSE_PRINT;
        // load callee-saved vector registers (d8-d15)
        for (int i = NUM_15; i >= NUM_8; i -= NUM_2) {
            Ldp(ARMCG::DReg(i - 1), ARMCG::DReg(i),
                PostPtr(sp, 16)); // 16 is a size of 2x64-bit D registers
        }
        // load callee-saved X registers (x19-x28)
        for (int i = NUM_28; i >= NUM_19; i -= NUM_2) {
            Ldp(ARMCG::XReg(i - 1), ARMCG::XReg(i),
                PostPtr(sp, 16)); // 16 is a size of 2x64-bit X registers
        }
        // store x29 and x30 to stack for ABI requirement and decrement stack pointer by 16 bytes (2x64-bit D registers)
        Ldp(ARMCG::XReg(29), ARMCG::XReg(30), PostPtr(sp, 16));
    }
};

template <typename Task>
struct ImplBase {
    ImplBase(const Task &task, IntType nThreads) : implTask(task), implNThreads(nThreads)
    {
        VERBOSE_PRINT;
    }
    virtual void Run(typename Task::RunTimeParams params) noexcept(false) = 0;
    Task GetTask() const
    {
        VERBOSE_PRINT;
        return implTask;
    }
    IntType GetNThreads() const noexcept
    {
        VERBOSE_PRINT;
        return implNThreads;
    }
    virtual ~ImplBase()
    {
        VERBOSE_PRINT;
    }
protected:
    Task implTask;
    IntType implNThreads;
};

template <typename Task>
struct ImplGenerator {
    virtual bool Supports(const Task &task) const noexcept = 0;
    virtual float EstimateTime(const Task &task, IntType nThreads) const noexcept = 0;
    virtual std::shared_ptr<ImplBase<Task>> GetImpl(const Task &task, IntType nThreads) const = 0;
    virtual ~ImplGenerator()
    {
        VERBOSE_PRINT;
    }
};

template <typename Task>
std::vector<std::shared_ptr<ImplGenerator<Task>>>& GetImplGeneratorList() noexcept;

template <typename Task>
std::shared_ptr<ImplBase<Task>> FindSolution(const Task &task, IntType nThreads) noexcept(false);

template <typename Task>
struct SolutionHandler final {
    SolutionHandler() noexcept = default;
    SolutionHandler(const SolutionHandler &other) = delete;
    SolutionHandler(SolutionHandler &&other) noexcept = delete;
    SolutionHandler& operator=(const SolutionHandler &other) = delete;
    SolutionHandler& operator=(SolutionHandler &&other) noexcept = delete;
    std::shared_ptr<ImplBase<Task>> Find(const Task &task, IntType nThreads) noexcept(false)
    {
        VERBOSE_PRINT;
        for (auto&& solution : solutions) {
            if ((solution->GetTask() == task) && (solution->GetNThreads() == nThreads)) {
                return solution;
            }
        }
        auto&& implKind = ChooseImplGenerator(task, nThreads);
        solutions.push_back(implKind->GetImpl(task, nThreads));
        return solutions[solutions.size() - 1];
    }
protected:
    std::shared_ptr<ImplGenerator<Task>> ChooseImplGenerator(const Task &task, IntType nThreads) noexcept(false)
    {
        VERBOSE_PRINT;
        std::shared_ptr<ImplGenerator<Task>> ret;
        float bestTime = std::numeric_limits<float>::max();
        for (auto&& implKind : GetImplGeneratorList<Task>()) {
            if (implKind->Supports(task)) {
                auto currentEstimatedTime = implKind->EstimateTime(task, nThreads);
                if (currentEstimatedTime < bestTime) {
                    bestTime = currentEstimatedTime;
                    ret = implKind;
                }
            }
        }
        if (ret == nullptr) {
            throw Service::LogicError {"Solution for problem wasn't found"};
        }
        return ret;
    }

    std::vector<std::shared_ptr<ImplBase<Task>>> solutions;
};

} // Detail
} // KuDNN

#endif // KUDNN_JIT_HPP
