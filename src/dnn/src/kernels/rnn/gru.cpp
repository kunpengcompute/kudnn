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

#include "kernels/rnn/rnn_common.hpp"
#include "kernels/rnn/rnn_kernel_operators.hpp"
#include "kernels/vml/vml_sve_la_tanh.h"
#include "kernels/vml/vml_sve_la_sigmoid.h"

namespace KuDNN {
namespace Detail {
namespace Rnn {

/*
 * gru operation
 * u_t = sigmoid(W * h + U * x + b_h)
 * r_t = sigmoid(W * h + U * x + b_h)
 * o_t = tanh(W * h + U * (r_t * x) + b_h)
 * h_{t + 1} = u_t * h_t + (1 - u_t) * o_t
 */
template <typename T1, typename T2>
inline void RnnGruVCFWDCorePart1(T1 func1, T2 func2, svbool_t pgGru, svfloat32_t gateArgG0, svfloat32_t gateArgG1,
    svfloat32_t biasG0, svfloat32_t biasG1, svfloat32_t cellSrcIterArg,
    svfloat32_t *gateG0, svfloat32_t *gateG1, svfloat32_t *ht)
{
    gateArgG0 = svadd_f32_x(pgGru, gateArgG0, biasG0);
    gateArgG1 = svadd_f32_x(pgGru, gateArgG1, biasG1);
    *gateG0 = func1(gateArgG0, pgGru);
    *gateG1 = func2(gateArgG1, pgGru);
    *ht = svmul_f32_x(pgGru, cellSrcIterArg, *gateG1);
}

template <typename T1, typename T2>
void RnnGruVCFWDPart1(T1 func1, T2 func2, const int64_t len, float *scratch, const float *bias,
    const float *cellSrcIter, float *cellDstLay, float *cellDstIter, float *cellWsGates)
{
    int64_t i;
    int loopGru = svcntw();
    for (i = 0; i < len; i += loopGru) {
        svbool_t pgGru = svwhilelt_b32_s64(i, len);
        svfloat32_t gateArgG0 = svld1_f32(pgGru, scratch + i);
        svfloat32_t gateArgG1 = svld1_f32(pgGru, scratch + len + i);
        svfloat32_t cellSrcIterArg = svld1_f32(pgGru, cellSrcIter + i);

        svfloat32_t biasG0 = svld1_f32(pgGru, bias + i);
        svfloat32_t biasG1 = svld1_f32(pgGru, bias + len + i);

        svfloat32_t gateG0;
        svfloat32_t gateG1;
        svfloat32_t ht;
        RnnGruVCFWDCorePart1(func1, func2, pgGru, gateArgG0, gateArgG1, biasG0, biasG1, cellSrcIterArg,
            &gateG0, &gateG1, &ht);

        svst1_f32(pgGru, scratch + i, gateG0);
        svst1_f32(pgGru, cellDstLay + i, ht);
        if (cellDstIter) {
            svst1_f32(pgGru, cellDstIter + i, ht);
        }
        if (cellWsGates) {
            svst1_f32(pgGru, cellWsGates + i, gateG0);
            svst1_f32(pgGru, cellWsGates + len + i, gateG1);
        }
    }
}

template <typename T1, typename T2>
void RnnGruVCFWDPart1(T1 func1, T2 func2, const int64_t len, float *scratch, const __bf16 *bias,
    const __bf16 *cellSrcIter, __bf16 *cellDstLay, __bf16 *cellDstIter, __bf16 *cellWsGates)
{
    int64_t i;
    int loopGru = svcnth();
    int numF32 = svcntw();
    for (i = 0; i < len; i += loopGru) {
        svbool_t pgGru = svwhilelt_b16_s64(i, len);
        svbfloat16_t biasG0 = svld1_bf16(pgGru, bias + i);
        svbfloat16_t biasG1 = svld1_bf16(pgGru, bias + len + i);
        svbfloat16_t cellSrcIterArg = svld1_bf16(pgGru, cellSrcIter + i);

        svbool_t pgGruBot = svwhilelt_b32_s64(i, len);
        svbool_t pgGruTop = svwhilelt_b32_s64(i + numF32, len);
        svfloat32_t argG0Bot = svld1_f32(pgGruBot, scratch + i);
        svfloat32_t argG0Top = svld1_f32(pgGruTop, scratch + i + numF32);
        svfloat32_t argG1Bot = svld1_f32(pgGruBot, scratch + len + i);
        svfloat32_t argG1Top = svld1_f32(pgGruTop, scratch + len + i + numF32);
        svfloat32_t biasG0Bot = SvCvtF32BF16BottomX(biasG0);
        svfloat32_t biasG0Top = SvCvtF32BF16TopX(biasG0);
        svfloat32_t biasG1Bot = SvCvtF32BF16BottomX(biasG1);
        svfloat32_t biasG1Top = SvCvtF32BF16TopX(biasG1);
        svfloat32_t cellSrcIterArgBot = SvCvtF32BF16BottomX(cellSrcIterArg);
        svfloat32_t cellSrcIterArgTop = SvCvtF32BF16TopX(cellSrcIterArg);
        svfloat32_t gateG0Bot;
        svfloat32_t gateG0Top;
        svfloat32_t gateG1Bot;
        svfloat32_t gateG1Top;
        svfloat32_t htBot;
        svfloat32_t htTop;
        RnnGruVCFWDCorePart1(func1, func2, pgGruBot, argG0Bot, argG1Bot, biasG0Bot, biasG1Bot, cellSrcIterArgBot,
            &gateG0Bot, &gateG1Bot, &htBot);
        RnnGruVCFWDCorePart1(func1, func2, pgGruTop, argG0Top, argG1Top, biasG0Top, biasG1Top, cellSrcIterArgTop,
            &gateG0Top, &gateG1Top, &htTop);

        svbfloat16_t gateG0 = SvCvtBF16F32X(pgGruBot, gateG0Bot, pgGruTop, gateG0Top);
        svbfloat16_t gateG1 = SvCvtBF16F32X(pgGruBot, gateG1Bot, pgGruTop, gateG1Top);
        svbfloat16_t ht = SvCvtBF16F32X(pgGruBot, htBot, pgGruTop, htTop);

        svst1_f32(pgGruBot, scratch + i, gateG0Bot);
        svst1_f32(pgGruTop, scratch + i + numF32, gateG0Top);
        svst1_bf16(pgGru, cellDstLay + i, ht);
        if (cellDstIter) {
            svst1_bf16(pgGru, cellDstIter + i, ht);
        }
        if (cellWsGates) {
            svst1_bf16(pgGru, cellWsGates + i, gateG0);
            svst1_bf16(pgGru, cellWsGates + len + i, gateG1);
        }
    }
}

template <typename T1, typename T2>
void RnnGruVCFWDPart1(T1 func1, T2 func2, const int64_t len, float *scratch, const float *bias,
    const __bf16 *cellSrcIter, __bf16 *cellDstLay, __bf16 *cellDstIter, __bf16 *cellWsGates)
{
    int64_t i;
    int numF32 = svcntw();
    int loopGru = svcnth();
    for (i = 0; i < len; i += loopGru) {
        svbool_t pgGru = svwhilelt_b16_s64(i, len);
        svbool_t pgGruBot = svwhilelt_b32_s64(i, len);
        svbool_t pgGruTop = svwhilelt_b32_s64(i + numF32, len);
        svbfloat16_t cellSrcIterArg = svld1_bf16(pgGru, cellSrcIter + i);

        svfloat32_t argG0Bot = svld1_f32(pgGruBot, scratch + i);
        svfloat32_t argG0Top = svld1_f32(pgGruTop, scratch + i + numF32);
        svfloat32_t argG1Bot = svld1_f32(pgGruBot, scratch + len + i);
        svfloat32_t argG1Top = svld1_f32(pgGruTop, scratch + len + i + numF32);
        svfloat32_t biasG0Bot = svld1_f32(pgGruBot, bias + i);
        svfloat32_t biasG0Top = svld1_f32(pgGruTop, bias + i + numF32);
        svfloat32_t biasG1Bot = svld1_f32(pgGruBot, bias + len + i);
        svfloat32_t biasG1Top = svld1_f32(pgGruTop, bias + len + i + numF32);
        svfloat32_t cellSrcIterArgBot = SvCvtF32BF16BottomX(cellSrcIterArg);
        svfloat32_t cellSrcIterArgTop = SvCvtF32BF16TopX(cellSrcIterArg);
        svfloat32_t gateG0Bot;
        svfloat32_t gateG1Bot;
        svfloat32_t htBot;
        RnnGruVCFWDCorePart1(func1, func2, pgGruBot, argG0Bot, argG1Bot, biasG0Bot, biasG1Bot, cellSrcIterArgBot,
            &gateG0Bot, &gateG1Bot, &htBot);
        svfloat32_t gateG0Top;
        svfloat32_t gateG1Top;
        svfloat32_t htTop;
        RnnGruVCFWDCorePart1(func1, func2, pgGruTop, argG0Top, argG1Top, biasG0Top, biasG1Top, cellSrcIterArgTop,
            &gateG0Top, &gateG1Top, &htTop);

        svbfloat16_t gateG0 = SvCvtBF16F32X(pgGruBot, gateG0Bot, pgGruTop, gateG0Top);
        svbfloat16_t gateG1 = SvCvtBF16F32X(pgGruBot, gateG1Bot, pgGruTop, gateG1Top);
        svbfloat16_t ht = SvCvtBF16F32X(pgGruBot, htBot, pgGruTop, htTop);

        svst1_bf16(pgGru, cellDstLay + i, ht);
        svst1_f32(pgGruBot, scratch + i, gateG0Bot);
        svst1_f32(pgGruTop, scratch + i + numF32, gateG0Top);
        if (cellDstIter) {
            svst1_bf16(pgGru, cellDstIter + i, ht);
        }
        if (cellWsGates) {
            svst1_bf16(pgGru, cellWsGates + i, gateG0);
            svst1_bf16(pgGru, cellWsGates + len + i, gateG1);
        }
    }
}

template <typename SrcType, typename BiasType>
void RnnGruVCFWDPart1Wrapper(CellPositionT cellPosition, const RnnCfg &rnn, float *scratch, const BiasType *cellBias,
    const SrcType *cellSrcIter, SrcType *cellDstLay, SrcType *cellDstIter, SrcType *cellWsGates)
{
    SizeType srcIterLd = rnn.SrcIterLd(cellPosition);
    SizeType dstLayerLd = rnn.DstLayerLd(cellPosition);
    if (rnn.gatesFuncLinearMode) {
        auto func1 = [&rnn](svfloat32_t x, svbool_t pgGru) { return svmul_n_f32_x(pgGru, x, rnn.scales[0]); };
        auto func2 = [&rnn](svfloat32_t x, svbool_t pgGru) { return svmul_n_f32_x(pgGru, x, rnn.scales[1]); };
        const auto gruCallPart1 = [&rnn, func1, func2, scratch, cellBias, cellSrcIter, cellDstLay, cellDstIter,
            cellWsGates, srcIterLd, dstLayerLd](int i) {
            RnnGruVCFWDPart1(func1, func2, rnn.dhc, scratch + i * rnn.scratchGatesLd, cellBias,
                cellSrcIter + i * srcIterLd, cellDstLay + i * dstLayerLd,
                cellDstIter == nullptr ? nullptr : cellDstIter + i * rnn.dstIterLd,
                rnn.isTraining ? cellWsGates + i * rnn.wsGatesLd : nullptr);
        };
        MultiThreader(rnn.mb, gruCallPart1);
    } else {
        auto func = [](svfloat32_t x, svbool_t pgGru) { return SvSigmoidF32X(svneg_f32_x(pgGru, x), pgGru); };
        const auto gruCallPart1 = [&rnn, func, scratch, cellBias, cellSrcIter, cellDstLay, cellDstIter, cellWsGates,
            srcIterLd, dstLayerLd](int i) {
            RnnGruVCFWDPart1(func, func, rnn.dhc, scratch + i * rnn.scratchGatesLd, cellBias,
                cellSrcIter + i * srcIterLd, cellDstLay + i * dstLayerLd,
                cellDstIter == nullptr ? nullptr : cellDstIter + i * rnn.dstIterLd,
                rnn.isTraining ? cellWsGates + i * rnn.wsGatesLd : nullptr);
        };
        MultiThreader(rnn.mb, gruCallPart1);
    }
}

template void RnnGruVCFWDPart1Wrapper<float, float>(CellPositionT cellPosition, const RnnCfg &rnn, float *scratch,
    const float *cellBias, const float *cellSrcIter, float *cellDstLay, float *cellDstIter, float *cellWsGates);
template void RnnGruVCFWDPart1Wrapper<__bf16, __bf16>(CellPositionT cellPosition, const RnnCfg &rnn, float *scratch,
    const __bf16 *cellBias, const __bf16 *cellSrcIter, __bf16 *cellDstLay, __bf16 *cellDstIter, __bf16 *cellWsGates);
template void RnnGruVCFWDPart1Wrapper<__bf16, float>(CellPositionT cellPosition, const RnnCfg &rnn, float *scratch,
    const float *cellBias, const __bf16 *cellSrcIter, __bf16 *cellDstLay, __bf16 *cellDstIter, __bf16 *cellWsGates);

template <typename FuncType>
void RnnGruVCFWDCorePart2(FuncType func, svbool_t pgGru, svfloat32_t gateG0, svfloat32_t gateArgG2, svfloat32_t biasG2,
    svfloat32_t cellSrcIterArg, svfloat32_t *gateG2, svfloat32_t *ht)
{
    gateArgG2 = svadd_f32_x(pgGru, gateArgG2, biasG2);
    *gateG2 = func(gateArgG2, pgGru);
    svfloat32_t ht1 = svmul_f32_x(pgGru, cellSrcIterArg, gateG0);
    svfloat32_t ht2 = svmls_f32_x(pgGru, *gateG2, gateG0, *gateG2);
    *ht = svadd_f32_x(pgGru, ht1, ht2);
}

template <typename FuncType>
void RnnGruVCFWDPart2(FuncType func, const int64_t len, float *scratch, const float *bias,
    const float *cellSrcIter, float *cellDstLay, float *cellDstIter, float *cellWsGates)
{
    int64_t i;
    int loopGru = svcntw();
    for (i = 0; i < len; i += loopGru) {
        svbool_t pgGru = svwhilelt_b32_s64(i, len);
        svfloat32_t gateG0 = svld1_f32(pgGru, scratch + i);
        svfloat32_t gateArgG2 = svld1_f32(pgGru, scratch + 0x2 * len + i);
        svfloat32_t cellSrcIterArg = svld1_f32(pgGru, cellSrcIter + i);
        svfloat32_t biasG2 = svld1_f32(pgGru, bias + 2 * len + i);

        svfloat32_t gateG2;
        svfloat32_t ht;
        RnnGruVCFWDCorePart2(func, pgGru, gateG0, gateArgG2, biasG2, cellSrcIterArg, &gateG2, &ht);

        svst1_f32(pgGru, cellDstLay + i, ht);
        if (cellDstIter) {
            svst1_f32(pgGru, cellDstIter + i, ht);
        }
        if (cellWsGates) {
            svst1_f32(pgGru, cellWsGates + 0x2 * len + i, gateG2);
        }
    }
}

template <typename FuncType>
void RnnGruVCFWDPart2(FuncType func, const int64_t len, float *scratch, const __bf16 *bias,
    const __bf16 *cellSrcIter, __bf16 *cellDstLay, __bf16 *cellDstIter, __bf16 *cellWsGates)
{
    int64_t i;
    int loopGru = svcnth();
    int numF32 = svcntw();
    for (i = 0; i < len; i += loopGru) {
        svbool_t pgGru = svwhilelt_b16_s64(i, len);
        svbfloat16_t biasG2 = svld1_bf16(pgGru, bias + 0x2 * len + i);
        svbfloat16_t cellSrcIterArg = svld1_bf16(pgGru, cellSrcIter + i);

        svbool_t pgGruBot = svwhilelt_b32_s64(i, len);
        svbool_t pgGruTop = svwhilelt_b32_s64(i + numF32, len);
        svfloat32_t gateG0Bot = svld1_f32(pgGruBot, scratch + i);
        svfloat32_t gateG0Top = svld1_f32(pgGruTop, scratch + i + numF32);
        svfloat32_t argG2Bot = svld1_f32(pgGruBot, scratch + 0x2 * len + i);
        svfloat32_t argG2Top = svld1_f32(pgGruTop, scratch + 0x2 * len + i + numF32);
        svfloat32_t biasG2Bot = SvCvtF32BF16BottomX(biasG2);
        svfloat32_t biasG2Top = SvCvtF32BF16TopX(biasG2);
        svfloat32_t cellSrcIterArgBot = SvCvtF32BF16BottomX(cellSrcIterArg);
        svfloat32_t cellSrcIterArgTop = SvCvtF32BF16TopX(cellSrcIterArg);
        svfloat32_t gateG2Bot;
        svfloat32_t htBot;
        RnnGruVCFWDCorePart2(func, pgGruBot, gateG0Bot, argG2Bot, biasG2Bot, cellSrcIterArgBot, &gateG2Bot, &htBot);
        svfloat32_t gateG2Top;
        svfloat32_t htTop;
        RnnGruVCFWDCorePart2(func, pgGruTop, gateG0Top, argG2Top, biasG2Top, cellSrcIterArgTop, &gateG2Top, &htTop);

        svbfloat16_t ht = SvCvtBF16F32X(pgGruBot, htBot, pgGruTop, htTop);
        svbfloat16_t gateG2 = SvCvtBF16F32X(pgGruBot, gateG2Bot, pgGruTop, gateG2Top);
        svst1_bf16(pgGru, cellDstLay + i, ht);
        if (cellDstIter) {
            svst1_bf16(pgGru, cellDstIter + i, ht);
        }
        if (cellWsGates) {
            svst1_bf16(pgGru, cellWsGates + 0x2 * len + i, gateG2);
        }
    }
}

template <typename FuncType>
void RnnGruVCFWDPart2(FuncType func, const int64_t len, float *scratch, const float *bias,
    const __bf16 *cellSrcIter, __bf16 *cellDstLay, __bf16 *cellDstIter, __bf16 *cellWsGates)
{
    int64_t i;
    int loopGru = svcnth();
    int numF32 = svcntw();
    for (i = 0; i < len; i += loopGru) {
        svbool_t pgGru = svwhilelt_b16_s64(i, len);
        svbfloat16_t cellSrcIterArg = svld1_bf16(pgGru, cellSrcIter + i);

        svbool_t pgGruBot = svwhilelt_b32_s64(i, len);
        svbool_t pgGruTop = svwhilelt_b32_s64(i + numF32, len);
        svfloat32_t gateG0Bot = svld1_f32(pgGruBot, scratch + i);
        svfloat32_t gateG0Top = svld1_f32(pgGruTop, scratch + i + numF32);
        svfloat32_t argG2Bot = svld1_f32(pgGruBot, scratch + 0x2 * len + i);
        svfloat32_t argG2Top = svld1_f32(pgGruTop, scratch + 0x2 * len + i + numF32);
        svfloat32_t biasG2Bot = svld1_f32(pgGruBot, bias + 0x2 * len + i);
        svfloat32_t biasG2Top = svld1_f32(pgGruTop, bias + 0x2 * len + i + numF32);
        svfloat32_t cellSrcIterArgBot = SvCvtF32BF16BottomX(cellSrcIterArg);
        svfloat32_t cellSrcIterArgTop = SvCvtF32BF16TopX(cellSrcIterArg);
        svfloat32_t gateG2Bot;
        svfloat32_t gateG2Top;
        svfloat32_t htBot;
        svfloat32_t htTop;
        RnnGruVCFWDCorePart2(func, pgGruBot, gateG0Bot, argG2Bot, biasG2Bot, cellSrcIterArgBot, &gateG2Bot, &htBot);
        RnnGruVCFWDCorePart2(func, pgGruTop, gateG0Top, argG2Top, biasG2Top, cellSrcIterArgTop, &gateG2Top, &htTop);

        svbfloat16_t gateG2 = SvCvtBF16F32X(pgGruBot, gateG2Bot, pgGruTop, gateG2Top);
        svbfloat16_t ht = SvCvtBF16F32X(pgGruBot, htBot, pgGruTop, htTop);

        svst1_bf16(pgGru, cellDstLay + i, ht);
        if (cellDstIter) {
            svst1_bf16(pgGru, cellDstIter + i, ht);
        }
        if (cellWsGates) {
            svst1_bf16(pgGru, cellWsGates + 0x2 * len + i, gateG2);
        }
    }
}

template <typename SrcType, typename BiasType>
void RnnGruVCFWDPart2Wrapper(CellPositionT cellPosition, const RnnCfg &rnn, float *scratch, const BiasType *cellBias,
    const SrcType *cellSrcIter, SrcType *cellDstLay, SrcType *cellDstIter, SrcType *cellWsGates)
{
    SizeType srcIterLd = rnn.SrcIterLd(cellPosition);
    SizeType dstLayerLd = rnn.DstLayerLd(cellPosition);
    if (rnn.gatesFuncLinearMode) {
        auto func = [&rnn](svfloat32_t x, svbool_t pgGru) { return svmul_n_f32_x(pgGru, x, rnn.scales[2]); };
        const auto gruCallPart2 = [&rnn, func, scratch, cellBias, cellSrcIter, cellDstLay, cellDstIter, cellWsGates,
            srcIterLd, dstLayerLd](int i) {
            RnnGruVCFWDPart2(func, rnn.dhc, scratch + i * rnn.scratchGatesLd, cellBias,
                cellSrcIter + i * srcIterLd, cellDstLay + i * dstLayerLd,
                cellDstIter == nullptr ? nullptr : cellDstIter + i * rnn.dstIterLd,
                rnn.isTraining ? cellWsGates + i * rnn.wsGatesLd : nullptr);
        };
        MultiThreader(rnn.mb, gruCallPart2);
    } else {
        auto func = [](svfloat32_t x, svbool_t pgGru) { return SvTanhF32X(x, pgGru); };
        const auto gruCallPart2 = [&rnn, func, scratch, cellBias, cellSrcIter, cellDstLay, cellDstIter, cellWsGates,
            srcIterLd, dstLayerLd](int i) {
            RnnGruVCFWDPart2(func, rnn.dhc, scratch + i * rnn.scratchGatesLd, cellBias,
                cellSrcIter + i * srcIterLd, cellDstLay + i * dstLayerLd,
                cellDstIter == nullptr ? nullptr : cellDstIter + i * rnn.dstIterLd,
                rnn.isTraining ? cellWsGates + i * rnn.wsGatesLd : nullptr);
        };
        MultiThreader(rnn.mb, gruCallPart2);
    }
}

template void RnnGruVCFWDPart2Wrapper<float, float>(CellPositionT cellPosition, const RnnCfg &rnn, float *scratch,
    const float *cellBias, const float *cellSrcIter,
    float *cellDstLay, float *cellDstIter, float *cellWsGates);
template void RnnGruVCFWDPart2Wrapper<__bf16, __bf16>(CellPositionT cellPosition, const RnnCfg &rnn, float *scratch,
    const __bf16 *cellBias, const __bf16 *cellSrcIter,
    __bf16 *cellDstLay, __bf16 *cellDstIter, __bf16 *cellWsGates);
template void RnnGruVCFWDPart2Wrapper<__bf16, float>(CellPositionT cellPosition, const RnnCfg &rnn, float *scratch,
    const float *cellBias, const __bf16 *cellSrcIter,
    __bf16 *cellDstLay, __bf16 *cellDstIter, __bf16 *cellWsGates);

inline void RnnGruVCBWDF32Part1Core(svbool_t pgGru, svfloat32_t ddl, svfloat32_t ddi, svfloat32_t gates0,
    svfloat32_t h, svfloat32_t gates2, svfloat32_t *dsIter, svfloat32_t *dG0, svfloat32_t *dG2)
{
    svfloat32_t dHt = svadd_f32_x(pgGru, ddl, ddi);
    *dsIter = svmul_f32_x(pgGru, dHt, gates0);

    svfloat32_t dG0Part11 = svsub_f32_x(pgGru, h, gates2);
    svfloat32_t dG0Part1 = svmul_f32_x(pgGru, dG0Part11, dHt);
    svfloat32_t dG0Part2 = svmls_f32_x(pgGru, gates0, gates0, gates0);
    *dG0 = svmul_f32_x(pgGru, dG0Part1, dG0Part2);

    svfloat32_t dG2Part1 = svmls_f32_x(pgGru, dHt, gates0, dHt);
    svfloat32_t dG2Part2 = svmsb_n_f32_x(pgGru, gates2, gates2, 1.0f);
    *dG2 = svmul_f32_x(pgGru, dG2Part1, dG2Part2);
}

void RnnGruVCBWDWrapperPart1(const int64_t len, const float *cellWsGates, const float *cellSrcIter,
    const float *cellDiffDstLay, const float *cellDiffDstIter, float *cellDiffSrcIter, float *scratch)
{
    int loopGru = svcntw();
    for (int64_t i = 0; i < len; i += loopGru) {
        svbool_t pgGru = svwhilelt_b32_s64(i, len);
        svfloat32_t h = svld1_f32(pgGru, cellSrcIter + i);
        svfloat32_t ddl = svld1_f32(pgGru, cellDiffDstLay + i);
        svfloat32_t ddi = svld1_f32(pgGru, cellDiffDstIter + i);
        svfloat32_t gates0 = svld1_f32(pgGru, cellWsGates + i);
        svfloat32_t gates2 = svld1_f32(pgGru, cellWsGates + 0x2 * len + i);

        svfloat32_t dsIter;
        svfloat32_t dG0;
        svfloat32_t dG2;
        RnnGruVCBWDF32Part1Core(pgGru, ddl, ddi, gates0, h, gates2, &dsIter, &dG0, &dG2);

        svst1_f32(pgGru, cellDiffSrcIter + i, dsIter);
        svst1_f32(pgGru, scratch + i, dG0);
        svst1_f32(pgGru, scratch + 0x2 * len + i, dG2);
    }
}

void RnnGruVCBWDWrapperPart1(const int64_t len, const __bf16 *cellWsGates, const __bf16 *cellSrcIter,
    const float *cellDiffDstLay, const float *cellDiffDstIter, float *cellDiffSrcIter, __bf16 *scratch)
{
    int loopGru = svcnth();
    int numF32 = svcntw();
    for (int64_t i = 0; i < len; i += loopGru) {
        svbool_t pgGru = svwhilelt_b16_s64(i, len);
        svbool_t pgGruBot = svwhilelt_b32_s64(i, len);
        svbool_t pgGruTop = svwhilelt_b32_s64(i + numF32, len);

        svbfloat16_t gates0 = svld1_bf16(pgGru, cellWsGates + i);
        svbfloat16_t gates2 = svld1_bf16(pgGru, cellWsGates + 0x2 * len + i);

        svbfloat16_t h = svld1_bf16(pgGru, cellSrcIter + i);

        svfloat32_t hBot = SvCvtF32BF16BottomX(h);
        svfloat32_t hTop = SvCvtF32BF16TopX(h);
        svfloat32_t ddlBot = svld1_f32(pgGruBot, cellDiffDstLay + i);
        svfloat32_t ddlTop = svld1_f32(pgGruTop, cellDiffDstLay + i + numF32);
        svfloat32_t ddiBot = svld1_f32(pgGruBot, cellDiffDstIter + i);
        svfloat32_t ddiTop = svld1_f32(pgGruTop, cellDiffDstIter + i + numF32);
        svfloat32_t gates0Bot = SvCvtF32BF16BottomX(gates0);
        svfloat32_t gates0Top = SvCvtF32BF16TopX(gates0);
        svfloat32_t gates2Bot = SvCvtF32BF16BottomX(gates2);
        svfloat32_t gates2Top = SvCvtF32BF16TopX(gates2);

        svfloat32_t dsIterBot;
        svfloat32_t dG0Bot;
        svfloat32_t dG2Bot;
        RnnGruVCBWDF32Part1Core(pgGruBot, ddlBot, ddiBot, gates0Bot, hBot, gates2Bot, &dsIterBot, &dG0Bot, &dG2Bot);
        svfloat32_t dsIterTop;
        svfloat32_t dG0Top;
        svfloat32_t dG2Top;
        RnnGruVCBWDF32Part1Core(pgGruTop, ddlTop, ddiTop, gates0Top, hTop, gates2Top, &dsIterTop, &dG0Top, &dG2Top);

        svbfloat16_t dG0 = SvCvtBF16F32X(pgGruBot, dG0Bot, pgGruTop, dG0Top);
        svbfloat16_t dG2 = SvCvtBF16F32X(pgGruBot, dG2Bot, pgGruTop, dG2Top);

        svst1_f32(pgGruBot, cellDiffSrcIter + i, dsIterBot);
        svst1_f32(pgGruTop, cellDiffSrcIter + i + numF32, dsIterTop);
        svst1_bf16(pgGru, scratch + i, dG0);
        svst1_bf16(pgGru, scratch + 0x2 * len + i, dG2);
    }
}

inline void RnnGruVCBWDF32Part2Core(svbool_t pgGru, svfloat32_t dhGates1, svfloat32_t gates1, svfloat32_t dsIter,
    svfloat32_t h, svfloat32_t *dsIter1, svfloat32_t *scratchGates, svfloat32_t *hGates1)
{
    svfloat32_t tmp = svmul_f32_x(pgGru, dhGates1, gates1);
    *dsIter1 = svadd_f32_x(pgGru, dsIter, tmp);

    *scratchGates = svmul_f32_x(pgGru, dhGates1, h);
    *scratchGates = svmul_f32_x(pgGru, *scratchGates, svmls_f32_x(pgGru, gates1, gates1, gates1));

    *hGates1 = svmul_f32_x(pgGru, gates1, h);
}

void RnnGruVCBWDWrapperPart2(const int64_t len, float *dhG1, float *hG1, const float *cellWsGates,
    const float *cellSrcIter, float *cellDiffSrcIter, float *scratch)
{
    int loopGru = svcntw();
    for (int64_t i = 0; i < len; i += loopGru) {
        svbool_t pgGru = svwhilelt_b32_s64(i, len);

        svfloat32_t dhGates1 = svld1_f32(pgGru, dhG1 + i);
        svfloat32_t gates1 = svld1_f32(pgGru, cellWsGates + len + i);
        svfloat32_t dsIter = svld1_f32(pgGru, cellDiffSrcIter + i);
        svfloat32_t h = svld1_f32(pgGru, cellSrcIter + i);

        svfloat32_t dsIter1;
        svfloat32_t scratchGates;
        svfloat32_t hGates1;
        RnnGruVCBWDF32Part2Core(pgGru, dhGates1, gates1, dsIter, h, &dsIter1, &scratchGates, &hGates1);

        svst1_f32(pgGru, cellDiffSrcIter + i, dsIter1);
        svst1_f32(pgGru, scratch + len + i, scratchGates);
        svst1_f32(pgGru, hG1 + i, hGates1);
    }
}

void RnnGruVCBWDWrapperPart2(const int64_t len, float *dhG1, __bf16 *hG1, const __bf16 *cellWsGates,
    const __bf16 *cellSrcIter, float *cellDiffSrcIter, __bf16 *scratch)
{
    int loopGru = svcnth();
    int numF32 = svcntw();
    for (int64_t i = 0; i < len; i += loopGru) {
        svbool_t pgGru = svwhilelt_b16_s64(i, len);
        svbool_t pgGruBot = svwhilelt_b32_s64(i, len);
        svbool_t pgGruTop = svwhilelt_b32_s64(i + numF32, len);

        svbfloat16_t gates1 = svld1_bf16(pgGru, cellWsGates + len + i);
        svbfloat16_t h = svld1_bf16(pgGru, cellSrcIter + i);

        svfloat32_t dhGates1Bot = svld1_f32(pgGruBot, dhG1 + i);
        svfloat32_t dhGates1Top = svld1_f32(pgGruTop, dhG1 + i + numF32);
        svfloat32_t gates1Bot = SvCvtF32BF16BottomX(gates1);
        svfloat32_t gates1Top = SvCvtF32BF16TopX(gates1);
        svfloat32_t dsIterBot = svld1_f32(pgGruBot, cellDiffSrcIter + i);
        svfloat32_t dsIterTop = svld1_f32(pgGruTop, cellDiffSrcIter + i + numF32);
        svfloat32_t hBot = SvCvtF32BF16BottomX(h);
        svfloat32_t hTop = SvCvtF32BF16TopX(h);

        svfloat32_t dsIter1Bot;
        svfloat32_t scratchGatesBot;
        svfloat32_t hGates1Bot;
        RnnGruVCBWDF32Part2Core(pgGruBot, dhGates1Bot, gates1Bot, dsIterBot, hBot,
                                 &dsIter1Bot, &scratchGatesBot, &hGates1Bot);
        svfloat32_t dsIter1Top;
        svfloat32_t scratchGatesTop;
        svfloat32_t hGates1Top;
        RnnGruVCBWDF32Part2Core(pgGruTop, dhGates1Top, gates1Top, dsIterTop, hTop,
                                 &dsIter1Top, &scratchGatesTop, &hGates1Top);

        svbfloat16_t scratchGates = SvCvtBF16F32X(pgGruBot, scratchGatesBot, pgGruTop, scratchGatesTop);
        svbfloat16_t hGates1 = SvCvtBF16F32X(pgGruBot, hGates1Bot, pgGruTop, hGates1Top);
        svst1_f32(pgGruBot, cellDiffSrcIter + i, dsIter1Bot);
        svst1_f32(pgGruTop, cellDiffSrcIter + i + numF32, dsIter1Top);
        svst1_bf16(pgGru, scratch + len + i, scratchGates);
        svst1_bf16(pgGru, hG1 + i, hGates1);
    }
}

} // Rnn
} // Detail
} // KuDNN
