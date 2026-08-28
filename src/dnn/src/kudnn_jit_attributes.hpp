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

#ifndef KUDNN_JIT_ATTRIBUTES_HPP
#define KUDNN_JIT_ATTRIBUTES_HPP

#include <vector>
#include <memory>
#include <limits>
#include <type_traits>
#include <chrono>
#include <mutex>
#include <array>

#include "kudnn.hpp"

#include "cg_codegen.hpp"

#include "internal_funcs.hpp"
#include "kudnn_jit.hpp"

namespace KuDNN {
namespace Detail {

struct KUDNN_JIT_Generator_With_Attributes : public KUDNN_JIT_Generator {
protected:
    static constexpr SizeType AArch64NumRegs = NUM_32;
    static constexpr SizeType AArch64NumPRegs = NUM_8;
    explicit KUDNN_JIT_Generator_With_Attributes(const Attributes &attributesArg);

    SizeType FindFreeVectorReg()
    {
        return FindFreeReg(freeVectorRegs, lastUsedVectorReg);
    }

    SizeType FindFreeScalarReg()
    {
        return FindFreeReg(freeScalarRegs, lastUsedScalarReg);
    }

    SizeType FindFreePredicateReg()
    {
        return FindFreeReg(freePredicateRegs, lastUsedPredicateReg);
    }

    void TakeVectorRegister(SizeType regID)
    {
        TakeRegister(regID, freeVectorRegs);
    }

    void FreeVectorRegister(SizeType regID)
    {
        FreeRegister(regID, freeVectorRegs);
    }

    void TakeScalarRegister(SizeType regID)
    {
        TakeRegister(regID, freeScalarRegs);
    }

    void FreeScalarRegister(SizeType regID)
    {
        FreeRegister(regID, freeScalarRegs);
    }

    void TakePredicateRegister(SizeType regID)
    {
        TakeRegister(regID, freePredicateRegs);
    }

    void FreePredicateRegister(SizeType regID)
    {
        FreeRegister(regID, freePredicateRegs);
    }

    virtual void ApplySum(const PostOps &postOps, SizeType poIdx, const std::vector<SizeType> &regIDs,
        Element::TypeT regType);

    virtual void ApplyEltwise(const PostOps &postOps, SizeType poIdx, const std::vector<SizeType> &regIDs,
        Element::TypeT regType);

    virtual void ApplyDW(const PostOps &postOps, SizeType poIdx, const std::vector<SizeType> &regIDs,
        Element::TypeT regType);

    virtual void ApplyBinary(const PostOps &postOps, SizeType poIdx, const std::vector<SizeType> &regIDs,
        Element::TypeT regType);

    virtual void ApplyPReLU(const PostOps &postOps, SizeType poIdx, const std::vector<SizeType> &regIDs,
        Element::TypeT regType);

    virtual void ApplyPostOps(const std::vector<SizeType> &regIDs, Element::TypeT regType);

    Attributes attributes;

private:

    template <typename RegType>
    void ApplyEltwise(ActivationFunction actFunc, float alpha, float beta, const std::vector<SizeType> &regIDs)
    {
        if (actFunc != ActivationFunction::RELU) {
            throw KuDNN::Service::LogicError {"Unsupported Post Ops Eltwise operation"};
        } else {
            // betaEltwise isn't used for relu
            Service::Unused(beta);

            if (alpha != 1.0f) {
                auto tmpVRegID = FindFreeVectorReg();
                TakeVectorRegister(tmpVRegID);
                auto tmpRegID = FindFreeScalarReg();
                TakeScalarRegister(tmpRegID);
                std::int32_t alphaBits = ARMCG::Util::Float2Int(alpha);
                Mov(ARMCG::XReg(tmpRegID), alphaBits);
                Mov(KUDNN_JIT_Generator::GetZReg<RegType>(tmpVRegID), ARMCG::WReg(tmpRegID));
                for (auto &&regID : regIDs) {
                    auto tmpPregID = FindFreePredicateReg();
                    TakePredicateRegister(tmpPregID);
                    auto truePregID = FindFreePredicateReg();
                    TakePredicateRegister(truePregID);
                    Ptrue(KUDNN_JIT_Generator::GetPReg<float>(truePregID));
                    if constexpr (std::is_same_v<RegType, int32_t>) {
                        Scvtf(KUDNN_JIT_Generator::GetZReg<float>(regID),
                            KUDNN_JIT_Generator::GetPReg<int32_t>(truePregID),
                            KUDNN_JIT_Generator::GetZReg<int32_t>(regID));
                    }
                    Fcmle(KUDNN_JIT_Generator::GetPReg<RegType>(tmpPregID),
                        KUDNN_JIT_Generator::GetPReg<RegType>(truePregID),
                        KUDNN_JIT_Generator::GetZReg<RegType>(regID), 0.0);
                    Fmul(KUDNN_JIT_Generator::GetZReg<RegType>(regID),
                        KUDNN_JIT_Generator::GetPReg<RegType>(tmpPregID),
                        KUDNN_JIT_Generator::GetZReg<RegType>(tmpVRegID));
                    if constexpr (std::is_same_v<RegType, int32_t>) {
                        Frinti(KUDNN_JIT_Generator::GetZReg<float>(regID),
                            KUDNN_JIT_Generator::GetPReg<float>(truePregID),
                            KUDNN_JIT_Generator::GetZReg<float>(regID));
                        Fcvtzs(KUDNN_JIT_Generator::GetZReg<int32_t>(regID),
                            KUDNN_JIT_Generator::GetPReg<int32_t>(truePregID),
                            KUDNN_JIT_Generator::GetZReg<float>(regID));
                    }
                    FreePredicateRegister(truePregID);
                    FreePredicateRegister(tmpPregID);
                }
                FreeVectorRegister(tmpVRegID);
                FreeScalarRegister(tmpRegID);
            }
        }
    }

    template<SizeType numRegs>
    SizeType FindFreeReg(std::array<bool, numRegs> &freeRegs, SizeType &lastUsedReg)
    {
        SizeType ctr = 0;
        SizeType start = lastUsedReg + 1;
        while ((start < numRegs) && !freeRegs[start]) {
            ++start;
            ++ctr;
        }
        if (start < numRegs) {
            lastUsedReg = start;
            return start;
        } else {
            start = 0;
            while (!freeRegs[start] && (ctr < numRegs)) {
                ++start;
                ++ctr;
            }
            if (freeRegs[start]) {
                lastUsedReg = start;
                return start;
            } else {
                throw Service::LogicError{"Codegen: Can't find free register"};
            }
        }
    }

    template<SizeType numRegs>
    void TakeRegister(SizeType regID, std::array<bool, numRegs> &freeRegs)
    {
        if (freeRegs[regID]) {
            freeRegs[regID] = false;
        } else {
            throw Service::LogicError{"CodeGen: Register is already taken"};
        }
    }

    template<SizeType numRegs>
    void FreeRegister(SizeType regID, std::array<bool, numRegs> &freeRegs)
    {
        if (!freeRegs[regID]) {
            freeRegs[regID] = true;
        } else {
            throw Service::LogicError{"CodeGen: Register is already free"};
        }
    }

    std::array<bool, AArch64NumRegs> freeVectorRegs;
    SizeType lastUsedVectorReg;
    std::array<bool, AArch64NumRegs - NUM_1> freeScalarRegs;
    SizeType lastUsedScalarReg;
    std::array<bool, AArch64NumPRegs> freePredicateRegs;
    SizeType lastUsedPredicateReg;
};

} // Detail
} // KuDNN

#endif // KUDNN_JIT_ATTRIBUTES_HPP
