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
#include "kudnn_jit_attributes.hpp"

namespace KuDNN {
namespace Detail {

KUDNN_JIT_Generator_With_Attributes::KUDNN_JIT_Generator_With_Attributes(const Attributes &attributesArg)
    : KUDNN_JIT_Generator(), attributes(attributesArg)
{
    VERBOSE_PRINT;
    for (SizeType i = 0; i < AArch64NumRegs; ++i) {
        freeVectorRegs[i] = true;
    }
    lastUsedVectorReg = 0;
    for (SizeType i = 0; i < (AArch64NumRegs - NUM_1); ++i) {
        freeScalarRegs[i] = true;
    }
    lastUsedScalarReg = 0;
    for (SizeType i = 0; i < AArch64NumPRegs; ++i) {
        freePredicateRegs[i] = true;
    }
    lastUsedPredicateReg = 0;
}


void KUDNN_JIT_Generator_With_Attributes::ApplySum(const PostOps &postOps, SizeType poIdx,
    const std::vector<SizeType> &regIDs, Element::TypeT regType)
{
    Service::Unused(postOps, poIdx, regIDs, regType);
    throw Service::LogicError {"Sum isn't supported as PostOps yet"};
}

template <>
void KUDNN_JIT_Generator_With_Attributes::ApplyEltwise<__fp16>(ActivationFunction actFunc, float alpha, float beta,
    const std::vector<SizeType> &regIDs)
{
    if (actFunc != ActivationFunction::RELU) {
        throw KuDNN::Service::LogicError {"Unsupported Post Ops Eltwise operation"};
    } else {
        // betaEltwise isn't used for relu
        Service::Unused(beta);

        if (alpha != 1.0f) {
            auto tmpSRegID = FindFreeScalarReg();
            TakeScalarRegister(tmpSRegID);
            auto tmpVRegID1 = FindFreeVectorReg();
            TakeVectorRegister(tmpVRegID1);
            auto tmpVZeroRegID = FindFreeVectorReg();
            TakeVectorRegister(tmpVZeroRegID);
            auto tmpVRegID2 = FindFreeVectorReg();
            TakeVectorRegister(tmpVRegID2);
            auto tmpVRegID3 = FindFreeVectorReg();
            TakeVectorRegister(tmpVRegID3);

            std::int32_t alphaBits = ARMCG::Util::Float2Int(alpha);
            Mov(ARMCG::XReg(tmpSRegID), alphaBits);
            Mov(KUDNN_JIT_Generator::GetZReg<float>(tmpVRegID1), ARMCG::WReg(tmpSRegID));
            Dup(KUDNN_JIT_Generator::GetZReg<__fp16>(tmpVZeroRegID), 0.0);
            for (auto &&regID : regIDs) {
                auto truePregID = FindFreePredicateReg();
                TakePredicateRegister(truePregID);
                Ptrue(KUDNN_JIT_Generator::GetPReg<float>(truePregID));

                auto tmpPregID = FindFreePredicateReg();
                TakePredicateRegister(tmpPregID);
                
                Zip1(KUDNN_JIT_Generator::GetZReg<__fp16>(tmpVRegID2),
                    KUDNN_JIT_Generator::GetZReg<__fp16>(regID),
                    KUDNN_JIT_Generator::GetZReg<__fp16>(tmpVZeroRegID));

                Zip2(KUDNN_JIT_Generator::GetZReg<__fp16>(tmpVRegID3),
                    KUDNN_JIT_Generator::GetZReg<__fp16>(regID),
                    KUDNN_JIT_Generator::GetZReg<__fp16>(tmpVZeroRegID));
                
                Fcvt(KUDNN_JIT_Generator::GetZReg<float>(tmpVRegID2),
                    KUDNN_JIT_Generator::GetPReg<__fp16>(truePregID),
                    KUDNN_JIT_Generator::GetZReg<__fp16>(tmpVRegID2));
                
                Fcvt(KUDNN_JIT_Generator::GetZReg<float>(tmpVRegID3),
                    KUDNN_JIT_Generator::GetPReg<__fp16>(truePregID),
                    KUDNN_JIT_Generator::GetZReg<__fp16>(tmpVRegID3));
                
                Fcmle(KUDNN_JIT_Generator::GetPReg<float>(tmpPregID),
                    p5, KUDNN_JIT_Generator::GetZReg<float>(tmpVRegID2), 0.0);

                Fmul(KUDNN_JIT_Generator::GetZReg<float>(tmpVRegID2),
                    KUDNN_JIT_Generator::GetPReg<float>(tmpPregID),
                    KUDNN_JIT_Generator::GetZReg<float>(tmpVRegID1));

                Fcmle(KUDNN_JIT_Generator::GetPReg<float>(tmpPregID),
                    p5, KUDNN_JIT_Generator::GetZReg<float>(tmpVRegID3), 0.0);

                Fmul(KUDNN_JIT_Generator::GetZReg<float>(tmpVRegID3),
                    KUDNN_JIT_Generator::GetPReg<float>(tmpPregID),
                    KUDNN_JIT_Generator::GetZReg<float>(tmpVRegID1));

                Fcvt(KUDNN_JIT_Generator::GetZReg<__fp16>(tmpVRegID2),
                    KUDNN_JIT_Generator::GetPReg<float>(truePregID),
                    KUDNN_JIT_Generator::GetZReg<float>(tmpVRegID2));
                
                Fcvt(KUDNN_JIT_Generator::GetZReg<__fp16>(tmpVRegID3),
                    KUDNN_JIT_Generator::GetPReg<float>(truePregID),
                    KUDNN_JIT_Generator::GetZReg<float>(tmpVRegID3));
                
                Uzp1(KUDNN_JIT_Generator::GetZReg<__fp16>(regID),
                    KUDNN_JIT_Generator::GetZReg<__fp16>(tmpVRegID2),
                    KUDNN_JIT_Generator::GetZReg<__fp16>(tmpVRegID3));
                FreePredicateRegister(tmpPregID);
                FreePredicateRegister(truePregID);
            }

            FreeVectorRegister(tmpVRegID1);
            FreeScalarRegister(tmpSRegID);
            FreeVectorRegister(tmpVZeroRegID);
            FreeVectorRegister(tmpVRegID2);
            FreeVectorRegister(tmpVRegID3);
        }
    }
}

void KUDNN_JIT_Generator_With_Attributes::ApplyEltwise(const PostOps &postOps, SizeType poIdx,
    const std::vector<SizeType> &regIDs, Element::TypeT regType)
{
    ActivationFunction actEltwise = ActivationFunction::UNIMPLEMENTED;
    float alphaEltwise = 1.0f;
    float betaEltwise = 0.0f;
    postOps.GetParamsEltwise(poIdx, actEltwise, alphaEltwise, betaEltwise);
    switch (regType) {
        case Element::TypeT::F32: {
            ApplyEltwise<float>(actEltwise, alphaEltwise, betaEltwise, regIDs);
            break;
        }
        case Element::TypeT::S32: {
            ApplyEltwise<std::int32_t>(actEltwise, alphaEltwise, betaEltwise, regIDs);
            break;
        }
        case Element::TypeT::F16: {
            ApplyEltwise<__fp16>(actEltwise, alphaEltwise, betaEltwise, regIDs);
            break;
        }
        default: {
            throw KuDNN::Service::LogicError {"Unsupported Post Ops Eltwise precision"};
            break;
        }
    }
}

void KUDNN_JIT_Generator_With_Attributes::ApplyDW(const PostOps &postOps, SizeType poIdx,
    const std::vector<SizeType> &regIDs, Element::TypeT regType)
{
    Service::Unused(postOps, poIdx, regIDs, regType);
    throw Service::LogicError {"DW isn't supported as PostOps yet"};
}

void KUDNN_JIT_Generator_With_Attributes::ApplyBinary(const PostOps &postOps, SizeType poIdx,
    const std::vector<SizeType> &regIDs, Element::TypeT regType)
{
    Service::Unused(postOps, poIdx, regIDs, regType);
    throw Service::LogicError {"Binary isn't supported as PostOps yet"};
}

void KUDNN_JIT_Generator_With_Attributes::ApplyPReLU(const PostOps &postOps, SizeType poIdx,
    const std::vector<SizeType> &regIDs, Element::TypeT regType)
{
    Service::Unused(postOps, poIdx, regIDs, regType);
    throw Service::LogicError {"PReLU isn't supported as PostOps yet"};
}

void KUDNN_JIT_Generator_With_Attributes::ApplyPostOps(const std::vector<SizeType> &regIDs,
    Element::TypeT regType)
{
    auto &&postOps = attributes.GetPostOps();
    for (SizeType poIdx = 0; poIdx < postOps.Len(); ++poIdx) {
        switch (postOps.Kind(poIdx)) {
            case PrimitiveKind::SUM: {
                ApplySum(postOps, poIdx, regIDs, regType);
                break;
            }
            case PrimitiveKind::ELTWISE: {
                ApplyEltwise(postOps, poIdx, regIDs, regType);
                break;
            }
            case PrimitiveKind::CONVOLUTION: {
                ApplyDW(postOps, poIdx, regIDs, regType);
                break;
            }
            case PrimitiveKind::BINARY: {
                ApplyBinary(postOps, poIdx, regIDs, regType);
                break;
            }
            case PrimitiveKind::PRELU: {
                ApplyPReLU(postOps, poIdx, regIDs, regType);
                break;
            }
            default: {
                throw KuDNN::Service::LogicError {"Unsupported Post Ops"};
                break;
            }
        }
    }
}

} // Detail
} // KuDNN
