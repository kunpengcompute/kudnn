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
#include <type_traits>
#include <chrono>
#include <mutex>
#include <array>

#include "kudnn.hpp"

#include "cg_codegen.hpp"

#define VERBOSE_PRINT

namespace KuDNN {
namespace Detail {

struct KUDNN_JIT_Generator : public ARMCG::CodeGenerator {
protected:
    KUDNN_JIT_Generator();

    template <typename Type>
    auto GetZReg(std::uint32_t index)
    {
        if constexpr (sizeof(Type) == NUM_1) {
            return ARMCG::ZRegB(index);
        } else if constexpr (sizeof(Type) == NUM_2) {
            return ARMCG::ZRegH(index);
        } else if constexpr (sizeof(Type) == NUM_4) {
            return ARMCG::ZRegS(index);
        } else {
            return ARMCG::ZRegD(index);
        }
    }

    template <typename Type>
    auto GetPReg(std::uint32_t index)
    {
        if constexpr (sizeof(Type) == NUM_1) {
            return ARMCG::PRegB(index);
        } else if constexpr (sizeof(Type) == NUM_2) {
            return ARMCG::PRegH(index);
        } else if constexpr (sizeof(Type) == NUM_4) {
            return ARMCG::PRegS(index);
        } else {
            return ARMCG::PRegD(index);
        }
    }

    template <typename Type>
    auto Inc(const ARMCG::XReg &xd, const ARMCG::Pattern pat = ARMCG::Pattern::ALL,
        const ARMCG::ExtMod mod = ARMCG::ExtMod::MUL, const std::uint32_t imm = 1)
    {
        static_assert(sizeof(Type) <= NUM_4, "Incd isn't supported in CodeGen");
        if constexpr (sizeof(Type) == NUM_1) {
            return Incb(xd, pat, mod, imm);
        } else if constexpr (sizeof(Type) == NUM_2) {
            return  Inch(xd, pat, mod, imm);
        } else {
            return  Incw(xd, pat, mod, imm);
        }
    }

    template <typename Type, typename Adr>
    void Ld1(const Type &zt, const ARMCG::PRegBase &pg, const Adr &adr)
    {
        if constexpr (std::is_same_v<Type, ARMCG::ZRegB>) {
            return Ld1b(zt, pg, adr);
        } else if constexpr (std::is_same_v<Type, ARMCG::ZRegH>) {
            return Ld1h(zt, pg, adr);
        } else if constexpr (std::is_same_v<Type, ARMCG::ZRegS>) {
            return Ld1w(zt, pg, adr);
        } else {
            return Ld1d(zt, pg, adr);
        }
    }

    template <typename Type, typename Adr>
    void Ld1r(const Type &zt, const ARMCG::PRegBase &pg, const Adr &adr)
    {
        static_assert(!std::is_same_v<Type, ARMCG::ZRegD>, "Ld1rd isn't supported in CodeGen");
        if constexpr (std::is_same_v<Type, ARMCG::ZRegB>) {
            return Ld1rb(zt, pg, adr);
        } else if constexpr (std::is_same_v<Type, ARMCG::ZRegH>) {
            return Ld1rh(zt, pg, adr);
        } else {
            return Ld1rw(zt, pg, adr);
        }
    }

    template <typename Type, typename Adr>
    void Ld1rq(const Type &zt, const ARMCG::PRegBase &pg, const Adr &adr)
    {
        if constexpr (std::is_same_v<Type, ARMCG::ZRegB>) {
            return Ld1rqb(zt, pg, adr);
        } else if constexpr (std::is_same_v<Type, ARMCG::ZRegH>) {
            return Ld1rqh(zt, pg, adr);
        } else if constexpr (std::is_same_v<Type, ARMCG::ZRegS>) {
            return Ld1rqw(zt, pg, adr);
        } else {
            return Ld1rqd(zt, pg, adr);
        }
    }

    template <typename Type, typename Adr>
    void St1(const Type &zt, const ARMCG::PRegBase &pg, const Adr &adr)
    {
        if constexpr (std::is_same_v<Type, ARMCG::ZRegB>) {
            return St1b(zt, pg, adr);
        } else if constexpr (std::is_same_v<Type, ARMCG::ZRegH>) {
            return St1h(zt, pg, adr);
        } else if constexpr (std::is_same_v<Type, ARMCG::ZRegS>) {
            return St1w(zt, pg, adr);
        } else {
            return St1d(zt, pg, adr);
        }
    }

    void Preamble();

    void Postamble();
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
    virtual std::chrono::duration<float, std::milli> EstimateTime(const Task &task,
                                                                IntType nThreads) const noexcept = 0;
    virtual std::shared_ptr<ImplBase<Task>> GetImpl(const Task &task, IntType nThreads) const = 0;
    virtual ~ImplGenerator()
    {
        VERBOSE_PRINT;
    }
};

template <typename Task>
const std::vector<std::shared_ptr<ImplGenerator<Task>>>& GetImplGeneratorList() noexcept;

template <typename Task>
std::weak_ptr<ImplBase<Task>> FindSolution(const Task &task, IntType nThreads) noexcept(false);

template <typename Task>
void ClearImpls() noexcept(false);

template <typename Task>
struct SolutionHandler final {
    SolutionHandler() noexcept = default;
    SolutionHandler(const SolutionHandler &other) = delete;
    SolutionHandler(SolutionHandler &&other) noexcept = delete;
    SolutionHandler& operator=(const SolutionHandler &other) = delete;
    SolutionHandler& operator=(SolutionHandler &&other) noexcept = delete;
    std::weak_ptr<ImplBase<Task>> Find(const Task &task, IntType nThreads) noexcept(false)
    {
        VERBOSE_PRINT;
        std::lock_guard<std::mutex> lock(mtx);
        std::pair<Task, IntType> currTask{task, nThreads};
        if (solutions.find(currTask) == solutions.end()) {
            auto&& implKind = ChooseImplGenerator(task, nThreads);
            solutions.insert({currTask, implKind->GetImpl(task, nThreads)});
        }
        return solutions[currTask];
    }
    void ClearImpls()
    {
        solutions.clear();
    }
private:
    std::shared_ptr<ImplGenerator<Task>> ChooseImplGenerator(const Task &task, IntType nThreads) noexcept(false)
    {
        VERBOSE_PRINT;
        std::shared_ptr<ImplGenerator<Task>> ret;
        std::chrono::duration<float, std::milli> bestTime =
            std::chrono::duration<float, std::milli>{std::numeric_limits<float>::max()};
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

    std::unordered_map<std::pair<Task, IntType>, std::shared_ptr<ImplBase<Task>>> solutions;
    std::mutex mtx;
};

template<typename Task>
SizeType GetTaskHash(const Task& task);

} // Detail
} // KuDNN

template<typename Task>
struct std::hash<std::pair<Task, KuDNN::IntType>> {
    std::size_t operator()(const std::pair<Task, int>& pair) const noexcept
    {
        std::size_t h1 = std::hash<KuDNN::SizeType>{}(GetTaskHash(pair.first));
        std::size_t h2 = std::hash<KuDNN::SizeType>{}(pair.second);
        return h1 ^ (h2 << 1);
    }
};

#endif // KUDNN_JIT_HPP
