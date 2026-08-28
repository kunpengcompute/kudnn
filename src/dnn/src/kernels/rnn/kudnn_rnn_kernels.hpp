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

#ifndef KUDNN_RNN_KERNELS_HPP
#define KUDNN_RNN_KERNELS_HPP

#include <functional>
#include <unordered_map>
#include "kernels/rnn/kudnn_rnn_gemm.h"

namespace KuDNN {
namespace Detail {
namespace Rnn {

constexpr SizeType ELEMENT_TYPE_COUNT = 4;
constexpr SizeType RNN_ALG_COUNT = 7;

struct RnnKerParas {
    Element::TypeT srcType;
    Element::TypeT biasType;
    RnnAlgorithm alg;
};

struct RnnKerParasHash {
    SizeType operator()(const RnnKerParas &paras) const
    {
        return (static_cast<SizeType>(paras.alg) * ELEMENT_TYPE_COUNT + static_cast<SizeType>(paras.biasType)) *
            ELEMENT_TYPE_COUNT + static_cast<SizeType>(paras.srcType);
    }
};

struct RnnKerParasEqual {
    bool operator()(const RnnKerParas &paras0, const RnnKerParas &paras1) const
    {
        return paras0.srcType == paras1.srcType && paras0.biasType == paras1.biasType &&
            paras0.alg == paras1.alg;
    }
};

using RnnKerFwdFunc = std::function<void(const void **, const void **, void **, const void *, void *, const void *,
    const float *, const void *, const void *, const RnnCfg &, const KuDNN::ActivateFunctionRNN &, const float *,
    const RnnGemm *)>;
using RnnKerBwdFunc = std::function<void(const void **, const void **, const void **,
    const void *, const void *, void *,
    void *, const void **, void **, void **, void *, const void *, const void *, const void *,
    const RnnCfg &, const KuDNN::ActivateFunctionRNN &, const void *, const void *, const RnnBwdGemm *)>;

extern const std::unordered_map<RnnKerParas, RnnKerFwdFunc, RnnKerParasHash, RnnKerParasEqual> RNN_KER_MAP;
extern const std::unordered_map<RnnKerParas, RnnKerBwdFunc, RnnKerParasHash, RnnKerParasEqual> RNN_KER_BWD_MAP;


} // Rnn
} // Detail
} // KuDNN

#endif //  KUDNN_RNN_KERNELS_HPP
