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
 * lbr_gru operation
 * u_t = sigmoid(W * h + U * x + b_h)
 * r_t = sigmoid(W * h + U * x + b_h)
 * o_t = tanh(W * h + U * (r_t * (U_0 * x)) + b_h)
 * h_{t + 1} = u_t * h_t + (1 - u_t) * o_t
 */
template <typename T1, typename T2, typename T3>
void RnnLbrGruVCFWDCore(T1 func1, T2 func2, T3 func3, svbool_t pgLbrGru, svfloat32_t gateArgG0, svfloat32_t gateArgG1,
    svfloat32_t gateArgG2, svfloat32_t gateArgC0, svfloat32_t gateArgC1, svfloat32_t gateArgC2,
    svfloat32_t cellSrcIterArg, svfloat32_t biasG0, svfloat32_t biasG1, svfloat32_t biasG2, svfloat32_t biasG3,
    svfloat32_t *whb, svfloat32_t *gateG0, svfloat32_t *gateG1, svfloat32_t *gateG2, svfloat32_t *ht)
{
    *whb = svadd_f32_x(pgLbrGru, gateArgC2, biasG3);
    gateArgG0 = svadd_f32_x(pgLbrGru, gateArgG0, svadd_f32_x(pgLbrGru, gateArgC0, biasG0));
    gateArgG1 = svadd_f32_x(pgLbrGru, gateArgG1, svadd_f32_x(pgLbrGru, gateArgC1, biasG1));
    *gateG0 = func1(gateArgG0, pgLbrGru);
    *gateG1 = func2(gateArgG1, pgLbrGru);
    gateArgG2 = svmla_f32_x(pgLbrGru, svadd_f32_x(pgLbrGru, gateArgG2, biasG2), *whb, *gateG1);
    *gateG2 = func3(gateArgG2, pgLbrGru);
    svfloat32_t ht1 = svmul_f32_x(pgLbrGru, cellSrcIterArg, *gateG0);
    svfloat32_t ht2 = svmls_f32_x(pgLbrGru, *gateG2, *gateG0, *gateG2);
    *ht = svadd_f32_x(pgLbrGru, ht1, ht2);
}

template <typename T1, typename T2, typename T3>
void RnnLbrGruVCFWD(T1 func1, T2 func2, T3 func3, const int64_t len, float *scratch, float *scratchCell,
    const float *bias, const float *cellSrcIter, float *cellDstLay, float *cellDstIter, float *cellWsGates,
    float *cellWsWhb)
{
    int64_t i;
    int loopLbrgru = svcntw();
    for (i = 0; i < len; i += loopLbrgru) {
        svbool_t pgLbrGru = svwhilelt_b32_s64(i, len);
        svfloat32_t gateArgG0 = svld1_f32(pgLbrGru, scratch + i);
        svfloat32_t gateArgG1 = svld1_f32(pgLbrGru, scratch + len + i);
        svfloat32_t gateArgG2 = svld1_f32(pgLbrGru, scratch + 2 * len + i);
        svfloat32_t gateArgC0 = svld1_f32(pgLbrGru, scratchCell + i);
        svfloat32_t gateArgC1 = svld1_f32(pgLbrGru, scratchCell + len + i);
        svfloat32_t gateArgC2 = svld1_f32(pgLbrGru, scratchCell + 2 * len + i);

        svfloat32_t cellSrcIterArg = svld1_f32(pgLbrGru, cellSrcIter + i);
        svfloat32_t biasG0 = svld1_f32(pgLbrGru, bias + i);
        svfloat32_t biasG1 = svld1_f32(pgLbrGru, bias + len + i);
        svfloat32_t biasG2 = svld1_f32(pgLbrGru, bias + 2 * len + i);
        svfloat32_t biasG3 = svld1_f32(pgLbrGru, bias + 3 * len + i);

        svfloat32_t whb;
        svfloat32_t gateG0;
        svfloat32_t gateG1;
        svfloat32_t gateG2;
        svfloat32_t ht;
        RnnLbrGruVCFWDCore(func1, func2, func3, pgLbrGru, gateArgG0, gateArgG1, gateArgG2, gateArgC0, gateArgC1,
            gateArgC2, cellSrcIterArg, biasG0, biasG1, biasG2, biasG3, &whb, &gateG0, &gateG1, &gateG2, &ht);

        svst1_f32(pgLbrGru, cellDstLay + i, ht);
        if (cellDstIter) {
            svst1_f32(pgLbrGru, cellDstIter + i, ht);
        }
        if (cellWsGates) {
            svst1_f32(pgLbrGru, cellWsGates + i, gateG0);
            svst1_f32(pgLbrGru, cellWsGates + len + i, gateG1);
            svst1_f32(pgLbrGru, cellWsGates + 0x2 * len + i, gateG2);
            svst1_f32(pgLbrGru, cellWsWhb + i, whb);
        }
    }
}

template <typename T1, typename T2, typename T3>
void RnnLbrGruVCFWD(T1 func1, T2 func2, T3 func3, const int64_t len, float *scratch, float *scratchCell,
    const __bf16 *bias, const __bf16 *cellSrcIter, __bf16 *cellDstLay, __bf16 *cellDstIter, __bf16 *cellWsGates,
    __bf16 *cellWsWhb)
{
    int64_t i;
    int loopLbrgru = svcnth();
    int numF32 = svcntw();
    for (i = 0; i < len; i += loopLbrgru) {
        svbool_t pgLbrGru = svwhilelt_b16_s64(i, len);
        svbfloat16_t cellSrcIterArg = svld1_bf16(pgLbrGru, cellSrcIter + i);
        svbfloat16_t biasG0 = svld1_bf16(pgLbrGru, bias + i);
        svbfloat16_t biasG1 = svld1_bf16(pgLbrGru, bias + len + i);
        svbfloat16_t biasG2 = svld1_bf16(pgLbrGru, bias + 2 * len + i);
        svbfloat16_t biasG3 = svld1_bf16(pgLbrGru, bias + 3 * len + i);

        svbool_t pgLbrGruBot = svwhilelt_b32_s64(i, len);
        svbool_t pgLbrGruTop = svwhilelt_b32_s64(i + numF32, len);
        svfloat32_t gateArgG0Bot = svld1_f32(pgLbrGruBot, scratch + i);
        svfloat32_t gateArgG0Top = svld1_f32(pgLbrGruTop, scratch + i + numF32);
        svfloat32_t gateArgG1Bot = svld1_f32(pgLbrGruBot, scratch + len + i);
        svfloat32_t gateArgG1Top = svld1_f32(pgLbrGruTop, scratch + len + i + numF32);
        svfloat32_t gateArgG2BotBf16 = svld1_f32(pgLbrGruBot, scratch + 2 * len + i);
        svfloat32_t gateArgG2TopBf16 = svld1_f32(pgLbrGruTop, scratch + 2 * len + i + numF32);
        svfloat32_t gateArgC0Bot = svld1_f32(pgLbrGruBot, scratchCell + i);
        svfloat32_t gateArgC0Top = svld1_f32(pgLbrGruTop, scratchCell + i + numF32);
        svfloat32_t gateArgC1Bot = svld1_f32(pgLbrGruBot, scratchCell + len + i);
        svfloat32_t gateArgC1Top = svld1_f32(pgLbrGruTop, scratchCell + len + i + numF32);
        svfloat32_t gateArgC2Bot = svld1_f32(pgLbrGruBot, scratchCell + 2 * len + i);
        svfloat32_t gateArgC2Top = svld1_f32(pgLbrGruTop, scratchCell + 2 * len + i + numF32);

        svfloat32_t cellSrcIterArgBot = SvCvtF32BF16BottomX(cellSrcIterArg);
        svfloat32_t cellSrcIterArgTop = SvCvtF32BF16TopX(cellSrcIterArg);
        svfloat32_t biasG0Bot = SvCvtF32BF16BottomX(biasG0);
        svfloat32_t biasG0Top = SvCvtF32BF16TopX(biasG0);
        svfloat32_t biasG1Bot = SvCvtF32BF16BottomX(biasG1);
        svfloat32_t biasG1Top = SvCvtF32BF16TopX(biasG1);
        svfloat32_t biasG2Bot = SvCvtF32BF16BottomX(biasG2);
        svfloat32_t biasG2Top = SvCvtF32BF16TopX(biasG2);
        svfloat32_t biasG3Bot = SvCvtF32BF16BottomX(biasG3);
        svfloat32_t biasG3Top = SvCvtF32BF16TopX(biasG3);

        svfloat32_t whbBotFwd;
        svfloat32_t gateG0Bot;
        svfloat32_t gateG1Bot;
        svfloat32_t gateG2Bot;
        svfloat32_t htBot;
        svfloat32_t whbTopFwd;
        svfloat32_t gateG0Top;
        svfloat32_t gateG1Top;
        svfloat32_t gateG2Top;
        svfloat32_t htTop;
        RnnLbrGruVCFWDCore(func1, func2, func3, pgLbrGruBot, gateArgG0Bot, gateArgG1Bot, gateArgG2BotBf16,
            gateArgC0Bot, gateArgC1Bot, gateArgC2Bot, cellSrcIterArgBot, biasG0Bot, biasG1Bot, biasG2Bot, biasG3Bot,
            &whbBotFwd, &gateG0Bot, &gateG1Bot, &gateG2Bot, &htBot);
        RnnLbrGruVCFWDCore(func1, func2, func3, pgLbrGruTop, gateArgG0Top, gateArgG1Top, gateArgG2TopBf16,
            gateArgC0Top, gateArgC1Top, gateArgC2Top, cellSrcIterArgTop, biasG0Top, biasG1Top, biasG2Top, biasG3Top,
            &whbTopFwd, &gateG0Top, &gateG1Top, &gateG2Top, &htTop);

        svbfloat16_t whb = SvCvtBF16F32X(pgLbrGruBot, whbBotFwd, pgLbrGruTop, whbTopFwd);
        svbfloat16_t gateG0 = SvCvtBF16F32X(pgLbrGruBot, gateG0Bot, pgLbrGruTop, gateG0Top);
        svbfloat16_t gateG1 = SvCvtBF16F32X(pgLbrGruBot, gateG1Bot, pgLbrGruTop, gateG1Top);
        svbfloat16_t gateG2 = SvCvtBF16F32X(pgLbrGruBot, gateG2Bot, pgLbrGruTop, gateG2Top);
        svbfloat16_t ht = SvCvtBF16F32X(pgLbrGruBot, htBot, pgLbrGruTop, htTop);

        svst1_bf16(pgLbrGru, cellDstLay + i, ht);
        if (cellDstIter) {
            svst1_bf16(pgLbrGru, cellDstIter + i, ht);
        }
        if (cellWsGates) {
            svst1_bf16(pgLbrGru, cellWsGates + i, gateG0);
            svst1_bf16(pgLbrGru, cellWsGates + len + i, gateG1);
            svst1_bf16(pgLbrGru, cellWsGates + 0x2 * len + i, gateG2);
            svst1_bf16(pgLbrGru, cellWsWhb + i, whb);
        }
    }
}

template <typename T1, typename T2, typename T3>
void RnnLbrGruVCFWD(T1 func1, T2 func2, T3 func3, const int64_t len, float *scratch, float *scratchCell,
    const float *bias, const __bf16 *cellSrcIter, __bf16 *cellDstLay, __bf16 *cellDstIter, __bf16 *cellWsGates,
    __bf16 *cellWsWhb)
{
    int64_t i;
    int loopLbrgru = svcnth();
    int numF32 = svcntw();
    for (i = 0; i < len; i += loopLbrgru) {
        svbool_t pgLbrGru = svwhilelt_b16_s64(i, len);
        svbfloat16_t cellSrc = svld1_bf16(pgLbrGru, cellSrcIter + i);

        svbool_t pgLbrGruBot = svwhilelt_b32_s64(i, len);
        svbool_t pgLbrGruTop = svwhilelt_b32_s64(i + numF32, len);
        svfloat32_t gateArgG0Bot = svld1_f32(pgLbrGruBot, scratch + i);
        svfloat32_t gateArgG0Top = svld1_f32(pgLbrGruTop, scratch + i + numF32);
        svfloat32_t gateArgG1Bot = svld1_f32(pgLbrGruBot, scratch + len + i);
        svfloat32_t gateArgG1Top = svld1_f32(pgLbrGruTop, scratch + len + i + numF32);
        svfloat32_t gateArgG2BotFwd = svld1_f32(pgLbrGruBot, scratch + 2 * len + i);
        svfloat32_t gateArgG2TopFwd = svld1_f32(pgLbrGruTop, scratch + 2 * len + i + numF32);
        svfloat32_t gateArgC0BotFwd = svld1_f32(pgLbrGruBot, scratchCell + i);
        svfloat32_t gateArgC0TopFwd = svld1_f32(pgLbrGruTop, scratchCell + i + numF32);
        svfloat32_t gateArgC1Bot = svld1_f32(pgLbrGruBot, scratchCell + len + i);
        svfloat32_t gateArgC1Top = svld1_f32(pgLbrGruTop, scratchCell + len + i + numF32);
        svfloat32_t gateArgC2Bot = svld1_f32(pgLbrGruBot, scratchCell + 2 * len + i);
        svfloat32_t gateArgC2Top = svld1_f32(pgLbrGruTop, scratchCell + 2 * len + i + numF32);

        svfloat32_t cellSrcIterArgBot = SvCvtF32BF16BottomX(cellSrc);
        svfloat32_t cellSrcIterArgTop = SvCvtF32BF16TopX(cellSrc);
        svfloat32_t biasG0Bot = svld1_f32(pgLbrGruBot, bias + i);
        svfloat32_t biasG0Top = svld1_f32(pgLbrGruTop, bias + i + numF32);
        svfloat32_t biasG1Bot = svld1_f32(pgLbrGruBot, bias + len + i);
        svfloat32_t biasG1Top = svld1_f32(pgLbrGruTop, bias + len + i + numF32);
        svfloat32_t biasG2Bot = svld1_f32(pgLbrGruBot, bias + 2 * len + i);
        svfloat32_t biasG2Top = svld1_f32(pgLbrGruTop, bias + 2 * len + i + numF32);
        svfloat32_t biasG3Bot = svld1_f32(pgLbrGruBot, bias + 3 * len + i);
        svfloat32_t biasG3Top = svld1_f32(pgLbrGruTop, bias + 3 * len + i + numF32);

        svfloat32_t whbBot;
        svfloat32_t gateG0BotFwd;
        svfloat32_t gateG1BotFwd;
        svfloat32_t gateG2BotFwd;
        svfloat32_t htBotFwd;
        svfloat32_t whbTop;
        svfloat32_t gateG0Top;
        svfloat32_t gateG1Top;
        svfloat32_t gateG2Top;
        svfloat32_t htTop;
        RnnLbrGruVCFWDCore(func1, func2, func3, pgLbrGruBot, gateArgG0Bot, gateArgG1Bot, gateArgG2BotFwd,
            gateArgC0BotFwd, gateArgC1Bot, gateArgC2Bot, cellSrcIterArgBot, biasG0Bot, biasG1Bot, biasG2Bot, biasG3Bot,
            &whbBot, &gateG0BotFwd, &gateG1BotFwd, &gateG2BotFwd, &htBotFwd);
        RnnLbrGruVCFWDCore(func1, func2, func3, pgLbrGruTop, gateArgG0Top, gateArgG1Top, gateArgG2TopFwd,
            gateArgC0TopFwd, gateArgC1Top, gateArgC2Top, cellSrcIterArgTop, biasG0Top, biasG1Top, biasG2Top, biasG3Top,
            &whbTop, &gateG0Top, &gateG1Top, &gateG2Top, &htTop);

        svbfloat16_t whb = SvCvtBF16F32X(pgLbrGruBot, whbBot, pgLbrGruTop, whbTop);
        svbfloat16_t gateG0 = SvCvtBF16F32X(pgLbrGruBot, gateG0BotFwd, pgLbrGruTop, gateG0Top);
        svbfloat16_t gateG1 = SvCvtBF16F32X(pgLbrGruBot, gateG1BotFwd, pgLbrGruTop, gateG1Top);
        svbfloat16_t gateG2 = SvCvtBF16F32X(pgLbrGruBot, gateG2BotFwd, pgLbrGruTop, gateG2Top);
        svbfloat16_t ht = SvCvtBF16F32X(pgLbrGruBot, htBotFwd, pgLbrGruTop, htTop);

        svst1_bf16(pgLbrGru, cellDstLay + i, ht);
        if (cellDstIter) {
            svst1_bf16(pgLbrGru, cellDstIter + i, ht);
        }
        if (cellWsGates) {
            svst1_bf16(pgLbrGru, cellWsGates + i, gateG0);
            svst1_bf16(pgLbrGru, cellWsGates + len + i, gateG1);
            svst1_bf16(pgLbrGru, cellWsGates + 0x2 * len + i, gateG2);
            svst1_bf16(pgLbrGru, cellWsWhb + i, whb);
        }
    }
}

template <typename SrcType, typename BiasType>
void RnnLbrGruVCFWDWrapper(CellPositionT cellPosition, const RnnCfg &rnn, float *scratch, float *scratchCell,
    const BiasType *cellBias, const SrcType *cellSrcIter, SrcType *cellDstLay, SrcType *cellDstIter,
    SrcType *cellWsGate, SrcType *cellWsGrid)
{
    SizeType srcIterLd = rnn.SrcIterLd(cellPosition);
    SizeType dstLayerLd = rnn.DstLayerLd(cellPosition);
    if (rnn.gatesFuncLinearMode) {
        auto func1 = [&rnn](svfloat32_t x, svbool_t pgLbrGru) { return svmul_n_f32_x(pgLbrGru, x, rnn.scales[0]); };
        auto func2 = [&rnn](svfloat32_t x, svbool_t pgLbrGru) { return svmul_n_f32_x(pgLbrGru, x, rnn.scales[1]); };
        auto func3 = [&rnn](svfloat32_t x, svbool_t pgLbrGru) { return svmul_n_f32_x(pgLbrGru, x, rnn.scales[2]); };
        const auto lbrgruCall = [&rnn, func1, func2, func3, scratch, scratchCell, cellBias, cellSrcIter, cellDstLay,
            cellDstIter, cellWsGate, cellWsGrid, srcIterLd, dstLayerLd](int i) {
            RnnLbrGruVCFWD(func1, func2, func3, rnn.dhc, scratch + i * rnn.scratchGatesLd,
                scratchCell + i * rnn.wsGatesLd, cellBias, cellSrcIter + i * srcIterLd, cellDstLay + i * dstLayerLd,
                cellDstIter == nullptr ? nullptr : cellDstIter + i * rnn.dstIterLd,
                rnn.isTraining ? cellWsGate + i * rnn.wsGatesLd : nullptr,
                rnn.isTraining ? cellWsGrid + i * rnn.dhc : nullptr);
        };
        MultiThreader(rnn.mb, lbrgruCall);
    } else {
        auto func1 = [](svfloat32_t x, svbool_t pgLbrGru) { return SvSigmoidF32X(svneg_f32_x(pgLbrGru, x), pgLbrGru); };
        auto func2 = [](svfloat32_t x, svbool_t pgLbrGru) { return SvTanhF32X(x, pgLbrGru); };
        const auto lbrgruCall = [&rnn, func1, func2, scratch, scratchCell, cellBias, cellSrcIter, cellDstLay,
            cellDstIter, cellWsGate, cellWsGrid, srcIterLd, dstLayerLd](int i) {
            RnnLbrGruVCFWD(func1, func1, func2, rnn.dhc, scratch + i * rnn.scratchGatesLd,
                scratchCell + i * rnn.wsGatesLd, cellBias, cellSrcIter + i * srcIterLd, cellDstLay + i * dstLayerLd,
                cellDstIter == nullptr ? nullptr : cellDstIter + i * rnn.dstIterLd,
                rnn.isTraining ? cellWsGate + i * rnn.wsGatesLd : nullptr,
                rnn.isTraining ? cellWsGrid + i * rnn.dhc : nullptr);
        };
        MultiThreader(rnn.mb, lbrgruCall);
    }
}

template void RnnLbrGruVCFWDWrapper<float, float>(CellPositionT cellPosition, const RnnCfg &rnn, float *scratch,
    float *scratchCell, const float *cellBias, const float *cellSrcIter, float *cellDstLay,
    float *cellDstIter, float *cellWsGate, float *cellWsGrid);
template void RnnLbrGruVCFWDWrapper<__bf16, __bf16>(CellPositionT cellPosition, const RnnCfg &rnn, float *scratch,
    float *scratchCell, const __bf16 *cellBias, const __bf16 *cellSrcIter, __bf16 *cellDstLay,
    __bf16 *cellDstIter, __bf16 *cellWsGate, __bf16 *cellWsGrid);
template void RnnLbrGruVCFWDWrapper<__bf16, float>(CellPositionT cellPosition, const RnnCfg &rnn, float *scratch,
    float *scratchCell, const float *cellBias, const __bf16 *cellSrcIter, __bf16 *cellDstLay,
    __bf16 *cellDstIter, __bf16 *cellWsGate, __bf16 *cellWsGrid);

inline void LbrgruVCBWDCore(const float *cellDiffDstLay, const float *cellDiffDstIter, float *cellDiffSrcIter,
    svbool_t pgLbrGru, svfloat32_t h, svfloat32_t gates0, svfloat32_t gates1, svfloat32_t gates2, svfloat32_t gateWsWhb,
    svfloat32_t *dG0, svfloat32_t *dG1, svfloat32_t *dG2, svfloat32_t *dG2R)
{
    svfloat32_t ddl = svld1_f32(pgLbrGru, cellDiffDstLay);
    svfloat32_t ddi = svld1_f32(pgLbrGru, cellDiffDstIter);
    svfloat32_t dHt = svadd_f32_x(pgLbrGru, ddl, ddi);

    *dG0 = svmul_f32_x(pgLbrGru, svmul_f32_x(pgLbrGru, svsub_f32_x(pgLbrGru, h, gates2), dHt),
        svmls_f32_x(pgLbrGru, gates0, gates0, gates0));
    *dG2 = svmul_f32_x(pgLbrGru, svmls_f32_x(pgLbrGru, dHt, gates0, dHt),
        svmsb_n_f32_x(pgLbrGru, gates2, gates2, 1.0f));
    *dG1 = svmul_f32_x(pgLbrGru, svmul_f32_x(pgLbrGru, gateWsWhb, *dG2), svmls_f32_x(pgLbrGru, gates1, gates1, gates1));
    *dG2R = svmul_f32_x(pgLbrGru, *dG2, gates1);

    svst1_f32(pgLbrGru, cellDiffSrcIter, svmul_f32_x(pgLbrGru, dHt, gates0));
}

void RnnLbrgruVCBWDF32(const int64_t len, const float *cellWsGates, const float *cellSrcIter,
    const float *cellDiffDstLay, const float *cellDiffDstIter, float *cellDiffSrcIter, float *scratch,
    float *wsWhb, float *scratchGatesR)
{
    int loopLbrgru = svcntw();
    for (int64_t i = 0; i < len; i += loopLbrgru) {
        svbool_t pgLbrGru = svwhilelt_b32_s64(i, len);
        svfloat32_t h = svld1_f32(pgLbrGru, cellSrcIter + i);

        svfloat32_t gates0 = svld1_f32(pgLbrGru, cellWsGates + i);
        svfloat32_t gates1 = svld1_f32(pgLbrGru, cellWsGates + len + i);
        svfloat32_t gates2 = svld1_f32(pgLbrGru, cellWsGates + 0x2 * len + i);
        svfloat32_t gateWsWhb = svld1_f32(pgLbrGru, wsWhb + i);

        svfloat32_t dG0;
        svfloat32_t dG1;
        svfloat32_t dG2;
        svfloat32_t dG2R;
        LbrgruVCBWDCore(cellDiffDstLay + i, cellDiffDstIter + i, cellDiffSrcIter + i,
            pgLbrGru, h, gates0, gates1, gates2, gateWsWhb, &dG0, &dG1, &dG2, &dG2R);

        svst1_f32(pgLbrGru, scratch + i, dG0);
        svst1_f32(pgLbrGru, scratch + len + i, dG1);
        svst1_f32(pgLbrGru, scratch + 0x2 * len + i, dG2);

        svst1_f32(pgLbrGru, scratchGatesR + i, dG0);
        svst1_f32(pgLbrGru, scratchGatesR + len + i, dG1);
        svst1_f32(pgLbrGru, scratchGatesR + 0x2 * len + i, dG2R);
    }
}

void RnnLbrgruVCBWDBF16(const int64_t len, const __bf16 *cellWsGates, const __bf16 *cellSrcIter,
    const float *cellDiffDstLay, const float *cellDiffDstIter, float *cellDiffSrcIter, __bf16 *scratch,
    __bf16 *wsWhb, __bf16 *scratchGatesR)
{
    int loopLbrgru = svcnth();
    int numF32 = svcntw();
    for (int64_t i = 0; i < len; i += loopLbrgru) {
        svbool_t pgLbrGru = svwhilelt_b16_s64(i, len);
        svbfloat16_t h = svld1_bf16(pgLbrGru, cellSrcIter + i);

        svbfloat16_t gates0 = svld1_bf16(pgLbrGru, cellWsGates + i);
        svbfloat16_t gates1 = svld1_bf16(pgLbrGru, cellWsGates + len + i);
        svbfloat16_t gates2 = svld1_bf16(pgLbrGru, cellWsGates + 0x2 * len + i);
        svbfloat16_t gateWsWhb = svld1_bf16(pgLbrGru, wsWhb + i);
        svfloat32_t hBot = SvCvtF32BF16BottomX(h);
        svfloat32_t hTop = SvCvtF32BF16TopX(h);
        svfloat32_t gates0Bot = SvCvtF32BF16BottomX(gates0);
        svfloat32_t gates0Top = SvCvtF32BF16TopX(gates0);
        svfloat32_t gates1Bot = SvCvtF32BF16BottomX(gates1);
        svfloat32_t gates1Top = SvCvtF32BF16TopX(gates1);
        svfloat32_t gates2Bot = SvCvtF32BF16BottomX(gates2);
        svfloat32_t gates2TopBf16 = SvCvtF32BF16TopX(gates2);
        svfloat32_t gateWsWhbBot = SvCvtF32BF16BottomX(gateWsWhb);
        svfloat32_t gateWsWhbTop = SvCvtF32BF16TopX(gateWsWhb);

        svfloat32_t dG0Bot;
        svfloat32_t dG1Bot;
        svfloat32_t dG2Bot;
        svfloat32_t dG2RBot;
        svbool_t pgLbrGruBf16Bot = svwhilelt_b32_s64(i, len);
        LbrgruVCBWDCore(cellDiffDstLay + i, cellDiffDstIter + i, cellDiffSrcIter + i,
            pgLbrGruBf16Bot, hBot, gates0Bot, gates1Bot, gates2Bot, gateWsWhbBot, &dG0Bot, &dG1Bot, &dG2Bot, &dG2RBot);
        svfloat32_t dG0Top;
        svfloat32_t dG1Top;
        svfloat32_t dG2Top;
        svfloat32_t dG2RTop;
        svbool_t pgLbrGruTop = svwhilelt_b32_s64(i + numF32, len);
        LbrgruVCBWDCore(cellDiffDstLay + i + numF32, cellDiffDstIter + i + numF32, cellDiffSrcIter + i + numF32,
            pgLbrGruTop, hTop, gates0Top, gates1Top, gates2TopBf16, gateWsWhbTop, &dG0Top, &dG1Top, &dG2Top, &dG2RTop);

        svbfloat16_t dG0 = SvCvtBF16F32X(pgLbrGruBf16Bot, dG0Bot, pgLbrGruTop, dG0Top);
        svbfloat16_t dG1 = SvCvtBF16F32X(pgLbrGruBf16Bot, dG1Bot, pgLbrGruTop, dG1Top);
        svbfloat16_t dG2 = SvCvtBF16F32X(pgLbrGruBf16Bot, dG2Bot, pgLbrGruTop, dG2Top);
        svbfloat16_t dG2R = SvCvtBF16F32X(pgLbrGruBf16Bot, dG2RBot, pgLbrGruTop, dG2RTop);
        svst1_bf16(pgLbrGru, scratch + i, dG0);
        svst1_bf16(pgLbrGru, scratch + len + i, dG1);
        svst1_bf16(pgLbrGru, scratch + 0x2 * len + i, dG2);

        svst1_bf16(pgLbrGru, scratchGatesR + i, dG0);
        svst1_bf16(pgLbrGru, scratchGatesR + len + i, dG1);
        svst1_bf16(pgLbrGru, scratchGatesR + 0x2 * len + i, dG2R);
    }
}

} // Rnn
} // Detail
} // KuDNN
