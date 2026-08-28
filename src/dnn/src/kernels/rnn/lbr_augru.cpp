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
 * lbr_augru operation
 * u_t = sigmoid(W * h + U * x + b_h)
 * r_t = sigmoid(W * h + U * x + b_h)
 * o_t = tanh(W * h + U * (r_t * (U_0 * x)) + b_h)
 * u'_t = (1 - a) * u_t
 * h_{t + 1} = u'_t * h_t + (1 - u_t) * o_t
 */
template <typename T1, typename T2, typename T3>
inline void RnnLbrAugruVCFWDCore(T1 func1, T2 func2, T3 func3, svbool_t pgLbrAugru, svfloat32_t gateArgG0,
    svfloat32_t gateArgG1, svfloat32_t gateArgG2, svfloat32_t gateArgC0, svfloat32_t gateArgC1, svfloat32_t gateArgC2,
    svfloat32_t cellSrcIterArg, svfloat32_t aVec, svfloat32_t biasG0, svfloat32_t biasG1, svfloat32_t biasG2,
    svfloat32_t biasG3, svfloat32_t *whb, svfloat32_t *gateG0, svfloat32_t *gateG1, svfloat32_t *gateG2,
    svfloat32_t *ht)
{
    *whb = svadd_f32_x(pgLbrAugru, gateArgC2, biasG3);
    gateArgG0 = svadd_f32_x(pgLbrAugru, gateArgG0, svadd_f32_x(pgLbrAugru, gateArgC0, biasG0));
    gateArgG1 = svadd_f32_x(pgLbrAugru, gateArgG1, svadd_f32_x(pgLbrAugru, gateArgC1, biasG1));
    *gateG0 = func1(gateArgG0, pgLbrAugru);
    *gateG1 = func2(gateArgG1, pgLbrAugru);
    gateArgG2 = svmla_f32_x(pgLbrAugru, svadd_f32_x(pgLbrAugru, gateArgG2, biasG2), *whb, *gateG1);
    *gateG2 = func3(gateArgG2, pgLbrAugru);
    svfloat32_t gateG0MulA = svmls_f32_x(pgLbrAugru, *gateG0, aVec, *gateG0);
    svfloat32_t ht2 = svmls_f32_x(pgLbrAugru, *gateG2, gateG0MulA, *gateG2);
    *ht = svmla_f32_x(pgLbrAugru, ht2, cellSrcIterArg, gateG0MulA);
}

template <typename T1, typename T2, typename T3>
void RnnLbrAugruVCFWD(T1 func1, T2 func2, T3 func3, const int64_t len, float *scratch, float *scratchCell,
    const float *bias, float a, const float *cellSrcIter, float *cellDstLay, float *cellDstIter, float *cellWsGates,
    float *cellWsWhb)
{
    int64_t i;
    svfloat32_t aVec = svdup_n_f32_z(svptrue_b32(), a);
    int loopLbrAugru = svcntw();
    for (i = 0; i < len; i += loopLbrAugru) {
        svbool_t pgLbrAugru = svwhilelt_b32_s64(i, len);
        svfloat32_t gateArgG0 = svld1_f32(pgLbrAugru, scratch + i);
        svfloat32_t gateArgG1 = svld1_f32(pgLbrAugru, scratch + len + i);
        svfloat32_t gateArgG2 = svld1_f32(pgLbrAugru, scratch + 2 * len + i);
        svfloat32_t gateArgC0 = svld1_f32(pgLbrAugru, scratchCell + i);
        svfloat32_t gateArgC1 = svld1_f32(pgLbrAugru, scratchCell + len + i);
        svfloat32_t gateArgC2 = svld1_f32(pgLbrAugru, scratchCell + 2 * len + i);

        svfloat32_t cellSrcIterArg = svld1_f32(pgLbrAugru, cellSrcIter + i);
        svfloat32_t biasG0 = svld1_f32(pgLbrAugru, bias + i);
        svfloat32_t biasG1 = svld1_f32(pgLbrAugru, bias + len + i);
        svfloat32_t biasG2 = svld1_f32(pgLbrAugru, bias + 2 * len + i);
        svfloat32_t biasG3 = svld1_f32(pgLbrAugru, bias + 3 * len + i);

        svfloat32_t whb;
        svfloat32_t gateG0;
        svfloat32_t gateG1;
        svfloat32_t gateG2;
        svfloat32_t ht;
        RnnLbrAugruVCFWDCore(func1, func2, func3, pgLbrAugru, gateArgG0, gateArgG1, gateArgG2, gateArgC0, gateArgC1,
            gateArgC2, cellSrcIterArg, aVec, biasG0, biasG1, biasG2, biasG3, &whb, &gateG0, &gateG1, &gateG2, &ht);

        svst1_f32(pgLbrAugru, cellDstLay + i, ht);
        if (cellDstIter) {
            svst1_f32(pgLbrAugru, cellDstIter + i, ht);
        }
        if (cellWsGates) {
            svst1_f32(pgLbrAugru, cellWsGates + i, gateG0);
            svst1_f32(pgLbrAugru, cellWsGates + len + i, gateG1);
            svst1_f32(pgLbrAugru, cellWsGates + 0x2 * len + i, gateG2);
            svst1_f32(pgLbrAugru, cellWsWhb + i, whb);
        }
    }
}

template <typename T1, typename T2, typename T3>
void RnnLbrAugruVCFWD(T1 func1, T2 func2, T3 func3, const int64_t len, float *scratch, float *scratchCell,
    const __bf16 *bias, float a, const __bf16 *cellSrcIter, __bf16 *cellDstLay, __bf16 *cellDstIter,
    __bf16 *cellWsGates, __bf16 *cellWsWhb)
{
    int64_t i;
    svfloat32_t aVec = svdup_n_f32_z(svptrue_b32(), a);
    int loopLbrAugru = svcnth();
    int numF32 = svcntw();
    for (i = 0; i < len; i += loopLbrAugru) {
        svbool_t pgLbrAugru = svwhilelt_b16_s64(i, len);
        svbfloat16_t cellSrcIterArg = svld1_bf16(pgLbrAugru, cellSrcIter + i);
        svbfloat16_t biasG0 = svld1_bf16(pgLbrAugru, bias + i);
        svbfloat16_t biasG1 = svld1_bf16(pgLbrAugru, bias + len + i);
        svbfloat16_t biasG2 = svld1_bf16(pgLbrAugru, bias + 2 * len + i);
        svbfloat16_t biasG3 = svld1_bf16(pgLbrAugru, bias + 3 * len + i);

        svbool_t pgLbrAugruBot = svwhilelt_b32_s64(i, len);
        svbool_t pgLbrAugruTop = svwhilelt_b32_s64(i + numF32, len);
        svfloat32_t gateArgG0BotF32 = svld1_f32(pgLbrAugruBot, scratch + i);
        svfloat32_t gateArgG0TopF32 = svld1_f32(pgLbrAugruTop, scratch + i + numF32);
        svfloat32_t gateArgG1Bot = svld1_f32(pgLbrAugruBot, scratch + len + i);
        svfloat32_t gateArgG1Top = svld1_f32(pgLbrAugruTop, scratch + len + i + numF32);
        svfloat32_t gateArgG2Bot = svld1_f32(pgLbrAugruBot, scratch + 2 * len + i);
        svfloat32_t gateArgG2Top = svld1_f32(pgLbrAugruTop, scratch + 2 * len + i + numF32);
        svfloat32_t gateArgC0BotF32 = svld1_f32(pgLbrAugruBot, scratchCell + i);
        svfloat32_t gateArgC0TopF32 = svld1_f32(pgLbrAugruTop, scratchCell + i + numF32);
        svfloat32_t gateArgC1Bot = svld1_f32(pgLbrAugruBot, scratchCell + len + i);
        svfloat32_t gateArgC1Top = svld1_f32(pgLbrAugruTop, scratchCell + len + i + numF32);
        svfloat32_t gateArgC2Bot = svld1_f32(pgLbrAugruBot, scratchCell + 2 * len + i);
        svfloat32_t gateArgC2Top = svld1_f32(pgLbrAugruTop, scratchCell + 2 * len + i + numF32);

        svfloat32_t cellSrcIterArgBot = SvCvtF32BF16BottomX(cellSrcIterArg);
        svfloat32_t cellSrcIterArgTop = SvCvtF32BF16TopX(cellSrcIterArg);
        svfloat32_t biasG0BotF32 = SvCvtF32BF16BottomX(biasG0);
        svfloat32_t biasG0TopF32 = SvCvtF32BF16TopX(biasG0);
        svfloat32_t biasG1Bot = SvCvtF32BF16BottomX(biasG1);
        svfloat32_t biasG1Top = SvCvtF32BF16TopX(biasG1);
        svfloat32_t biasG2Bot = SvCvtF32BF16BottomX(biasG2);
        svfloat32_t biasG2Top = SvCvtF32BF16TopX(biasG2);
        svfloat32_t biasG3Bot = SvCvtF32BF16BottomX(biasG3);
        svfloat32_t biasG3Top = SvCvtF32BF16TopX(biasG3);

        svfloat32_t whbBot;
        svfloat32_t gateG0BotF32;
        svfloat32_t gateG1Bot;
        svfloat32_t gateG2Bot;
        svfloat32_t htBotF32;
        svfloat32_t whbTop;
        svfloat32_t gateG0TopF32;
        svfloat32_t gateG1Top;
        svfloat32_t gateG2Top;
        svfloat32_t htTop;
        RnnLbrAugruVCFWDCore(func1, func2, func3, pgLbrAugruBot, gateArgG0BotF32, gateArgG1Bot, gateArgG2Bot,
            gateArgC0BotF32, gateArgC1Bot, gateArgC2Bot, cellSrcIterArgBot, aVec, biasG0BotF32, biasG1Bot, biasG2Bot,
            biasG3Bot, &whbBot, &gateG0BotF32, &gateG1Bot, &gateG2Bot, &htBotF32);
        RnnLbrAugruVCFWDCore(func1, func2, func3, pgLbrAugruTop, gateArgG0TopF32, gateArgG1Top, gateArgG2Top,
            gateArgC0TopF32, gateArgC1Top, gateArgC2Top, cellSrcIterArgTop, aVec, biasG0TopF32, biasG1Top, biasG2Top,
            biasG3Top, &whbTop, &gateG0TopF32, &gateG1Top, &gateG2Top, &htTop);

        svbfloat16_t whb = SvCvtBF16F32X(pgLbrAugruBot, whbBot, pgLbrAugruTop, whbTop);
        svbfloat16_t gateG0 = SvCvtBF16F32X(pgLbrAugruBot, gateG0BotF32, pgLbrAugruTop, gateG0TopF32);
        svbfloat16_t gateG1 = SvCvtBF16F32X(pgLbrAugruBot, gateG1Bot, pgLbrAugruTop, gateG1Top);
        svbfloat16_t gateG2 = SvCvtBF16F32X(pgLbrAugruBot, gateG2Bot, pgLbrAugruTop, gateG2Top);
        svbfloat16_t ht = SvCvtBF16F32X(pgLbrAugruBot, htBotF32, pgLbrAugruTop, htTop);

        svst1_bf16(pgLbrAugru, cellDstLay + i, ht);
        if (cellDstIter) {
            svst1_bf16(pgLbrAugru, cellDstIter + i, ht);
        }
        if (cellWsGates) {
            svst1_bf16(pgLbrAugru, cellWsGates + i, gateG0);
            svst1_bf16(pgLbrAugru, cellWsGates + len + i, gateG1);
            svst1_bf16(pgLbrAugru, cellWsGates + 0x2 * len + i, gateG2);
            svst1_bf16(pgLbrAugru, cellWsWhb + i, whb);
        }
    }
}

template <typename T1, typename T2, typename T3>
void RnnLbrAugruVCFWD(T1 func1, T2 func2, T3 func3, const int64_t len, float *scratch, float *scratchCell,
    const float *bias, float a, const __bf16 *cellSrcIter, __bf16 *cellDstLay, __bf16 *cellDstIter,
    __bf16 *cellWsGates, __bf16 *cellWsWhb)
{
    int64_t i;
    svfloat32_t aVec = svdup_n_f32_z(svptrue_b32(), a);
    int loopLbrAugru = svcnth();
    int numF32 = svcntw();
    for (i = 0; i < len; i += loopLbrAugru) {
        svbool_t pgLbrAugru = svwhilelt_b16_s64(i, len);
        svbfloat16_t cellSrcIterArg = svld1_bf16(pgLbrAugru, cellSrcIter + i);

        svbool_t pgLbrAugruBot = svwhilelt_b32_s64(i, len);
        svbool_t pgLbrAugruTop = svwhilelt_b32_s64(i + numF32, len);
        svfloat32_t gateArgG0Bot = svld1_f32(pgLbrAugruBot, scratch + i);
        svfloat32_t gateArgG0Top = svld1_f32(pgLbrAugruTop, scratch + i + numF32);
        svfloat32_t gateArgG1Bot = svld1_f32(pgLbrAugruBot, scratch + len + i);
        svfloat32_t gateArgG1Top = svld1_f32(pgLbrAugruTop, scratch + len + i + numF32);
        svfloat32_t gateArgG2Bot = svld1_f32(pgLbrAugruBot, scratch + 2 * len + i);
        svfloat32_t gateArgG2Top = svld1_f32(pgLbrAugruTop, scratch + 2 * len + i + numF32);
        svfloat32_t gateArgC0Bot = svld1_f32(pgLbrAugruBot, scratchCell + i);
        svfloat32_t gateArgC0Top = svld1_f32(pgLbrAugruTop, scratchCell + i + numF32);
        svfloat32_t gateArgC1Bot = svld1_f32(pgLbrAugruBot, scratchCell + len + i);
        svfloat32_t gateArgC1Top = svld1_f32(pgLbrAugruTop, scratchCell + len + i + numF32);
        svfloat32_t gateArgC2Bot = svld1_f32(pgLbrAugruBot, scratchCell + 2 * len + i);
        svfloat32_t gateArgC2Top = svld1_f32(pgLbrAugruTop, scratchCell + 2 * len + i + numF32);

        svfloat32_t cellSrcIterArgBot = SvCvtF32BF16BottomX(cellSrcIterArg);
        svfloat32_t cellSrcIterArgTop = SvCvtF32BF16TopX(cellSrcIterArg);
        svfloat32_t biasG0Bot = svld1_f32(pgLbrAugruBot, bias + i);
        svfloat32_t biasG0Top = svld1_f32(pgLbrAugruTop, bias + i + numF32);
        svfloat32_t biasG1Bot = svld1_f32(pgLbrAugruBot, bias + len + i);
        svfloat32_t biasG1Top = svld1_f32(pgLbrAugruTop, bias + len + i + numF32);
        svfloat32_t biasG2Bot = svld1_f32(pgLbrAugruBot, bias + 2 * len + i);
        svfloat32_t biasG2Top = svld1_f32(pgLbrAugruTop, bias + 2 * len + i + numF32);
        svfloat32_t biasG3Bot = svld1_f32(pgLbrAugruBot, bias + 3 * len + i);
        svfloat32_t biasG3Top = svld1_f32(pgLbrAugruTop, bias + 3 * len + i + numF32);

        svfloat32_t whbBotF32;
        svfloat32_t gateG0Bot;
        svfloat32_t gateG1Bot;
        svfloat32_t gateG2Bot;
        svfloat32_t htBotF32;
        svfloat32_t whbTopF32;
        svfloat32_t gateG0Top;
        svfloat32_t gateG1Top;
        svfloat32_t gateG2Top;
        svfloat32_t htTop;
        RnnLbrAugruVCFWDCore(func1, func2, func3, pgLbrAugruBot, gateArgG0Bot, gateArgG1Bot, gateArgG2Bot,
            gateArgC0Bot, gateArgC1Bot, gateArgC2Bot, cellSrcIterArgBot, aVec, biasG0Bot, biasG1Bot, biasG2Bot,
            biasG3Bot, &whbBotF32, &gateG0Bot, &gateG1Bot, &gateG2Bot, &htBotF32);
        RnnLbrAugruVCFWDCore(func1, func2, func3, pgLbrAugruTop, gateArgG0Top, gateArgG1Top, gateArgG2Top,
            gateArgC0Top, gateArgC1Top, gateArgC2Top, cellSrcIterArgTop, aVec, biasG0Top, biasG1Top, biasG2Top,
            biasG3Top, &whbTopF32, &gateG0Top, &gateG1Top, &gateG2Top, &htTop);

        svbfloat16_t whbBf16 = SvCvtBF16F32X(pgLbrAugruBot, whbBotF32, pgLbrAugruTop, whbTopF32);
        svbfloat16_t gateG0Bf16 = SvCvtBF16F32X(pgLbrAugruBot, gateG0Bot, pgLbrAugruTop, gateG0Top);
        svbfloat16_t gateG1Bf16 = SvCvtBF16F32X(pgLbrAugruBot, gateG1Bot, pgLbrAugruTop, gateG1Top);
        svbfloat16_t gateG2Bf16 = SvCvtBF16F32X(pgLbrAugruBot, gateG2Bot, pgLbrAugruTop, gateG2Top);
        svbfloat16_t htBf16 = SvCvtBF16F32X(pgLbrAugruBot, htBotF32, pgLbrAugruTop, htTop);

        svst1_bf16(pgLbrAugru, cellDstLay + i, htBf16);
        if (cellDstIter) {
            svst1_bf16(pgLbrAugru, cellDstIter + i, htBf16);
        }
        if (cellWsGates) {
            svst1_bf16(pgLbrAugru, cellWsGates + i, gateG0Bf16);
            svst1_bf16(pgLbrAugru, cellWsGates + len + i, gateG1Bf16);
            svst1_bf16(pgLbrAugru, cellWsGates + 0x2 * len + i, gateG2Bf16);
            svst1_bf16(pgLbrAugru, cellWsWhb + i, whbBf16);
        }
    }
}

template <typename SrcType, typename BiasType>
void RnnLbrAugruVCFWDWrapper(CellPositionT cellPosition, const RnnCfg &rnn, float *scratch, float *scratchCell,
    const BiasType *cellBias, SrcType *augruAttenOff, const SrcType *cellSrcIter, SrcType *cellDstLay,
    SrcType *cellDstIter, SrcType *cellWsGate, SrcType *cellWsGrid)
{
    SizeType srcIterLd = rnn.SrcIterLd(cellPosition);
    SizeType dstLayerLd = rnn.DstLayerLd(cellPosition);
    if (rnn.gatesFuncLinearMode) {
        auto func1 = [&rnn](svfloat32_t x, svbool_t pgLbrAugru) { return svmul_n_f32_x(pgLbrAugru, x, rnn.scales[0]); };
        auto func2 = [&rnn](svfloat32_t x, svbool_t pgLbrAugru) { return svmul_n_f32_x(pgLbrAugru, x, rnn.scales[1]); };
        auto func3 = [&rnn](svfloat32_t x, svbool_t pgLbrAugru) { return svmul_n_f32_x(pgLbrAugru, x, rnn.scales[2]); };
        const auto lbraugruCall = [&rnn, func1, func2, func3, scratch, scratchCell, cellBias, augruAttenOff,
            cellSrcIter, cellDstLay, cellDstIter, cellWsGate, cellWsGrid, srcIterLd, dstLayerLd](int i) {
            RnnLbrAugruVCFWD(func1, func2, func3, rnn.dhc, scratch + i * rnn.scratchGatesLd,
                scratchCell + i * rnn.wsGatesLd, cellBias, ToFloat(augruAttenOff[i]), cellSrcIter + i * srcIterLd,
                cellDstLay + i * dstLayerLd, cellDstIter == nullptr ? nullptr : cellDstIter + i * rnn.dstIterLd,
                rnn.isTraining ? cellWsGate + i * rnn.wsGatesLd : nullptr,
                rnn.isTraining ? cellWsGrid + i * rnn.dhc : nullptr);
        };
        MultiThreader(rnn.mb, lbraugruCall);
    } else {
        auto func1 = [](svfloat32_t x, svbool_t pgLbrAugru) {
            return SvSigmoidF32X(svneg_f32_x(pgLbrAugru, x), pgLbrAugru);
        };
        auto func2 = [](svfloat32_t x, svbool_t pgLbrAugru) { return SvTanhF32X(x, pgLbrAugru); };
        const auto lbraugruCall = [&rnn, func1, func2, scratch, scratchCell, cellBias, augruAttenOff, cellSrcIter,
            cellDstLay, cellDstIter, cellWsGate, cellWsGrid, srcIterLd, dstLayerLd](int i) {
            RnnLbrAugruVCFWD(func1, func1, func2, rnn.dhc, scratch + i * rnn.scratchGatesLd,
                scratchCell + i * rnn.wsGatesLd, cellBias, ToFloat(augruAttenOff[i]), cellSrcIter + i * srcIterLd,
                cellDstLay + i * dstLayerLd, cellDstIter == nullptr ? nullptr : cellDstIter + i * rnn.dstIterLd,
                rnn.isTraining ? cellWsGate + i * rnn.wsGatesLd : nullptr,
                rnn.isTraining ? cellWsGrid + i * rnn.dhc : nullptr);
        };
        MultiThreader(rnn.mb, lbraugruCall);
    }
}

template void RnnLbrAugruVCFWDWrapper<float, float>(CellPositionT cellPosition, const RnnCfg &rnn, float *scratch,
    float *scratchCell, const float *cellBias, float *augruAttenOff, const float *cellSrcIter, float *cellDstLay,
    float *cellDstIter, float *cellWsGate, float *cellWsGrid);
template void RnnLbrAugruVCFWDWrapper<__bf16, __bf16>(CellPositionT cellPosition, const RnnCfg &rnn, float *scratch,
    float *scratchCell, const __bf16 *cellBias, __bf16 *augruAttenOff, const __bf16 *cellSrcIter, __bf16 *cellDstLay,
    __bf16 *cellDstIter, __bf16 *cellWsGate, __bf16 *cellWsGrid);
template void RnnLbrAugruVCFWDWrapper<__bf16, float>(CellPositionT cellPosition, const RnnCfg &rnn, float *scratch,
    float *scratchCell, const float *cellBias, __bf16 *augruAttenOff, const __bf16 *cellSrcIter, __bf16 *cellDstLay,
    __bf16 *cellDstIter, __bf16 *cellWsGate, __bf16 *cellWsGrid);

inline void LbrAugruVCBWDCore(const float *cellDiffDstLay, const float *cellDiffDstIter, float *cellDiffSrcIter,
    svbool_t pg, svfloat32_t h, svfloat32_t gates0, svfloat32_t gates1, svfloat32_t gates2, svfloat32_t gateWsWhb,
    svfloat32_t *dG0, svfloat32_t *dG1, svfloat32_t *dG2, svfloat32_t *dG2R, svfloat32_t *dsIter)
{
    svfloat32_t ddl = svld1_f32(pg, cellDiffDstLay);
    svfloat32_t ddi = svld1_f32(pg, cellDiffDstIter);
    svfloat32_t dHt = svadd_f32_x(pg, ddl, ddi);

    *dsIter = svmul_f32_x(pg, dHt, gates0);
    *dG0 = svmul_f32_x(pg, svmul_f32_x(pg, svsub_f32_x(pg, h, gates2), dHt), svmls_f32_x(pg, gates0, gates0, gates0));
    *dG2 = svmul_f32_x(pg, svmls_f32_x(pg, dHt, gates0, dHt), svmsb_n_f32_x(pg, gates2, gates2, 1.0f));
    *dG1 = svmul_f32_x(pg, svmul_f32_x(pg, gateWsWhb, *dG2), svmls_f32_x(pg, gates1, gates1, gates1));
    *dG2R = svmul_f32_x(pg, *dG2, gates1);

    svst1_f32(pg, cellDiffSrcIter, *dsIter);
}

void RnnLbrAugruVCBWDF32(const int64_t len, int64_t mb, float *diffAugruAttenOff,
    float augruAttenOff, const float *cellWsGates, const float *cellSrcIter,
    const float *cellDiffDstLay, const float *cellDiffDstIter, float *cellDiffSrcIter, float *scratch,
    float *wsWhb, float *scratchGatesR)
{
    float diffAttentionSum = 0.0f;
    int loopLbrgru = svcntw();
    for (int64_t i = 0; i < len; i += loopLbrgru) {
        svbool_t pgLbrAugru = svwhilelt_b32_s64(i, len);

        svfloat32_t h = svld1_f32(pgLbrAugru, cellSrcIter + i);
        svfloat32_t gates0 = svld1_f32(pgLbrAugru, cellWsGates + i);
        svfloat32_t gates1 = svld1_f32(pgLbrAugru, cellWsGates + len + i);
        svfloat32_t gates2 = svld1_f32(pgLbrAugru, cellWsGates + 0x2 * len + i);
        svfloat32_t gateWsWhb = svld1_f32(pgLbrAugru, wsWhb + i);

        svfloat32_t dG0;
        svfloat32_t dG1;
        svfloat32_t dG2;
        svfloat32_t dG2R;
        svfloat32_t dsIter;
        LbrAugruVCBWDCore(cellDiffDstLay + i, cellDiffDstIter + i, cellDiffSrcIter + i,
            pgLbrAugru, h, gates0, gates1, gates2, gateWsWhb, &dG0, &dG1, &dG2, &dG2R, &dsIter);

        svfloat32_t augruAttenOffVector = svdup_n_f32_z(pgLbrAugru, augruAttenOff);
        diffAttentionSum -= svaddv_f32(pgLbrAugru, svmul_f32_x(pgLbrAugru, dG0, gates0));

        dG0 = svmls_f32_x(pgLbrAugru, dG0, dG0, augruAttenOffVector);

        svst1_f32(pgLbrAugru, cellDiffSrcIter + i, dsIter);

        svst1_f32(pgLbrAugru, scratch + i, dG0);
        svst1_f32(pgLbrAugru, scratch + len + i, dG1);
        svst1_f32(pgLbrAugru, scratch + 0x2 * len + i, dG2);

        svst1_f32(pgLbrAugru, scratchGatesR + i, dG0);
        svst1_f32(pgLbrAugru, scratchGatesR + len + i, dG1);
        svst1_f32(pgLbrAugru, scratchGatesR + 0x2 * len + i, svmul_f32_x(pgLbrAugru, dG2, gates1));
    }
    diffAugruAttenOff[mb] = diffAttentionSum;
}

void RnnLbrAugruVCBWDBF16(const int64_t len, int64_t mb, float *diffAugruAttenOff,
    __bf16 augruAttenOff, const __bf16 *cellWsGates, const __bf16 *cellSrcIter,
    const float *cellDiffDstLay, const float *cellDiffDstIter, float *cellDiffSrcIter, __bf16 *scratch,
    __bf16 *wsWhb, __bf16 *scratchGatesR)
{
    float diffAttentionSum = 0.0f;
    int loopLbrAugru = svcnth();
    int numF32 = svcntw();
    for (int64_t i = 0; i < len; i += loopLbrAugru) {
        svbool_t pgLbrAugruBf16 = svwhilelt_b16_s64(i, len);

        svbfloat16_t h = svld1_bf16(pgLbrAugruBf16, cellSrcIter + i);
        svbfloat16_t gates0Bf16 = svld1_bf16(pgLbrAugruBf16, cellWsGates + i);
        svbfloat16_t gates1Bf16 = svld1_bf16(pgLbrAugruBf16, cellWsGates + len + i);
        svbfloat16_t gates2Bf16 = svld1_bf16(pgLbrAugruBf16, cellWsGates + 0x2 * len + i);
        svbfloat16_t gateWsWhbBf16 = svld1_bf16(pgLbrAugruBf16, wsWhb + i);

        svfloat32_t hBot = SvCvtF32BF16BottomX(h);
        svfloat32_t hTop = SvCvtF32BF16TopX(h);
        svfloat32_t gates0Bot = SvCvtF32BF16BottomX(gates0Bf16);
        svfloat32_t gates0Top = SvCvtF32BF16TopX(gates0Bf16);
        svfloat32_t gates1Bot = SvCvtF32BF16BottomX(gates1Bf16);
        svfloat32_t gates1Top = SvCvtF32BF16TopX(gates1Bf16);
        svfloat32_t gates2Bot = SvCvtF32BF16BottomX(gates2Bf16);
        svfloat32_t gates2Top = SvCvtF32BF16TopX(gates2Bf16);
        svfloat32_t gateWsWhbBot = SvCvtF32BF16BottomX(gateWsWhbBf16);
        svfloat32_t gateWsWhbTop = SvCvtF32BF16TopX(gateWsWhbBf16);

        svfloat32_t dG0Bot;
        svfloat32_t dG1Bot;
        svfloat32_t dG2Bot;
        svfloat32_t dG2RBot;
        svfloat32_t dsIterBot;
        svbool_t pgLbrAugruBot = svwhilelt_b32_s64(i, len);
        LbrAugruVCBWDCore(cellDiffDstLay + i, cellDiffDstIter + i, cellDiffSrcIter + i,
            pgLbrAugruBot, hBot, gates0Bot, gates1Bot, gates2Bot, gateWsWhbBot, &dG0Bot, &dG1Bot, &dG2Bot, &dG2RBot,
            &dsIterBot);
        svfloat32_t dG0Top;
        svfloat32_t dG1Top;
        svfloat32_t dG2Top;
        svfloat32_t dG2RTop;
        svfloat32_t dsIterTop;
        svbool_t pgLbrAugruTop = svwhilelt_b32_s64(i + numF32, len);
        LbrAugruVCBWDCore(cellDiffDstLay + i + numF32, cellDiffDstIter + i + numF32, cellDiffSrcIter + i + numF32,
            pgLbrAugruTop, hTop, gates0Top, gates1Top, gates2Top, gateWsWhbTop, &dG0Top, &dG1Top, &dG2Top, &dG2RTop,
            &dsIterTop);

        svbfloat16_t augruAttenOffVector = svdup_n_bf16_z(pgLbrAugruBf16, augruAttenOff);

        svfloat32_t t1 = svmul_f32_x(pgLbrAugruBot, dG0Bot, gates0Bot);
        svfloat32_t t2 = svmul_f32_x(pgLbrAugruTop, dG0Top, gates0Top);

        diffAttentionSum -= svaddv_f32(pgLbrAugruBot, t1);
        diffAttentionSum -= svaddv_f32(pgLbrAugruTop, t2);

        dG0Bot = svmls_f32_x(pgLbrAugruBot, dG0Bot, dG0Bot, SvCvtF32BF16BottomX(augruAttenOffVector));
        dG0Top = svmls_f32_x(pgLbrAugruTop, dG0Top, dG0Top, SvCvtF32BF16TopX(augruAttenOffVector));

        svbfloat16_t dG0 = SvCvtBF16F32X(pgLbrAugruBot, dG0Bot, pgLbrAugruTop, dG0Top);
        svbfloat16_t dG1 = SvCvtBF16F32X(pgLbrAugruBot, dG1Bot, pgLbrAugruTop, dG1Top);
        svbfloat16_t dG2 = SvCvtBF16F32X(pgLbrAugruBot, dG2Bot, pgLbrAugruTop, dG2Top);
        svbfloat16_t dG2R = SvCvtBF16F32X(pgLbrAugruBot, dG2RBot, pgLbrAugruTop, dG2RTop);

        svst1_bf16(pgLbrAugruBf16, scratch + i, dG0);
        svst1_bf16(pgLbrAugruBf16, scratch + len + i, dG1);
        svst1_bf16(pgLbrAugruBf16, scratch + 0x2 * len + i, dG2);

        svst1_bf16(pgLbrAugruBf16, scratchGatesR + i, dG0);
        svst1_bf16(pgLbrAugruBf16, scratchGatesR + len + i, dG1);
        svst1_bf16(pgLbrAugruBf16, scratchGatesR + 0x2 * len + i, dG2R);
    }
    diffAugruAttenOff[mb] = diffAttentionSum;
}

} // Rnn
} // Detail
} // KuDNN
