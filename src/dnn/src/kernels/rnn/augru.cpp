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
 * augru operation
 * u_t = sigmoid(W * h + U * x + b_h)
 * r_t = sigmoid(W * h + U * x + b_h)
 * o_t = tanh(W * h + U * (r_t * x) + b_h)
 * u'_t = (1 - a) * u_t
 * h_{t + 1} = u'_t * h_t + (1 - u_t) * o_t
 */
template <typename FuncType>
inline void RnnAugruVCFWDCorePart2(FuncType func, svbool_t pgAugru, svfloat32_t gateArgG0, svfloat32_t gateArgG2,
    svfloat32_t biasG2, svfloat32_t cellSrcIterArg, svfloat32_t aVec, svfloat32_t *gateG2, svfloat32_t *ht)
{
    svfloat32_t gateG0 = svmls_f32_x(pgAugru, gateArgG0, aVec, gateArgG0);
    gateArgG2 = svadd_f32_x(pgAugru, gateArgG2, biasG2);
    *gateG2 = func(gateArgG2, pgAugru);
    svfloat32_t ht1 = svmul_f32_x(pgAugru, cellSrcIterArg, gateG0);
    svfloat32_t ht2 = svmls_f32_x(pgAugru, *gateG2, gateG0, *gateG2);
    *ht = svadd_f32_x(pgAugru, ht1, ht2);
}

template <typename FuncType>
void RnnAugruVCFWDPart2(FuncType func, const int64_t len, float *scratch, const float *bias, float a,
    const float *cellSrcIterF32, float *cellDstLay, float *cellDstIter, float *cellWsGates)
{
    int64_t i;
    svfloat32_t aVec = svdup_n_f32_x(svptrue_b32(), a);
    int loopGru = svcntw();
    for (i = 0; i < len; i += loopGru) {
        svbool_t pgAugru = svwhilelt_b32_s64(i, len);
        svfloat32_t gateArgG0 = svld1_f32(pgAugru, scratch + i);
        svfloat32_t gateArgG2 = svld1_f32(pgAugru, scratch + 2 * len + i);
        svfloat32_t biasG2 = svld1_f32(pgAugru, bias + 2 * len + i);
        svfloat32_t cellSrcIterArg = svld1_f32(pgAugru, cellSrcIterF32 + i);

        svfloat32_t gateG2;
        svfloat32_t ht;
        RnnAugruVCFWDCorePart2(func, pgAugru, gateArgG0, gateArgG2, biasG2, cellSrcIterArg, aVec, &gateG2, &ht);

        svst1_f32(pgAugru, cellDstLay + i, ht);
        if (cellDstIter) {
            svst1_f32(pgAugru, cellDstIter + i, ht);
        }
        if (cellWsGates) {
            svst1_f32(pgAugru, cellWsGates + 0x2 * len + i, gateG2);
        }
    }
}

template <typename FuncType>
void RnnAugruVCFWDPart2(FuncType func, const int64_t len, float *scratch, const __bf16 *biasBf16, float a,
    const __bf16 *cellSrcIterBf16, __bf16 *cellDstLay, __bf16 *cellDstIter, __bf16 *cellWsGates)
{
    int64_t i;
    svfloat32_t aVec = svdup_n_f32_x(svptrue_b32(), a);
    int loopAugru = svcnth();
    int numF32 = svcntw();
    for (i = 0; i < len; i += loopAugru) {
        svbool_t pgAugru = svwhilelt_b16_s64(i, len);
        svbfloat16_t biasG2 = svld1_bf16(pgAugru, biasBf16 + 0x2 * len + i);
        svbfloat16_t cellSrcIterArg = svld1_bf16(pgAugru, cellSrcIterBf16 + i);

        svbool_t pgAugruBot = svwhilelt_b32_s64(i, len);
        svbool_t pgAugruTop = svwhilelt_b32_s64(i + numF32, len);
        svfloat32_t gateG0Bot = svld1_f32(pgAugruBot, scratch + i);
        svfloat32_t gateG0Top = svld1_f32(pgAugruTop, scratch + i + numF32);
        svfloat32_t argG2Bot = svld1_f32(pgAugruBot, scratch + 0x2 * len + i);
        svfloat32_t argG2Top = svld1_f32(pgAugruTop, scratch + 0x2 * len + i + numF32);
        svfloat32_t biasG2Bot = SvCvtF32BF16BottomX(biasG2);
        svfloat32_t biasG2Top = SvCvtF32BF16TopX(biasG2);
        svfloat32_t cellSrcIterArgBot = SvCvtF32BF16BottomX(cellSrcIterArg);
        svfloat32_t cellSrcIterArgTop = SvCvtF32BF16TopX(cellSrcIterArg);
        svfloat32_t gateG2Bot;
        svfloat32_t gateG2Top;
        svfloat32_t htBot;
        svfloat32_t htTop;
        RnnAugruVCFWDCorePart2(func, pgAugruBot, gateG0Bot, argG2Bot, biasG2Bot, cellSrcIterArgBot, aVec,
            &gateG2Bot, &htBot);
        RnnAugruVCFWDCorePart2(func, pgAugruTop, gateG0Top, argG2Top, biasG2Top, cellSrcIterArgTop, aVec,
            &gateG2Top, &htTop);

        svbfloat16_t gateG2 = SvCvtBF16F32X(pgAugruBot, gateG2Bot, pgAugruTop, gateG2Top);
        svbfloat16_t ht = SvCvtBF16F32X(pgAugruBot, htBot, pgAugruTop, htTop);

        svst1_bf16(pgAugru, cellDstLay + i, ht);
        if (cellDstIter) {
            svst1_bf16(pgAugru, cellDstIter + i, ht);
        }
        if (cellWsGates) {
            svst1_bf16(pgAugru, cellWsGates + 0x2 * len + i, gateG2);
        }
    }
}

template <typename FuncType>
void RnnAugruVCFWDPart2(FuncType func, const int64_t len, float *scratch, const float *biasF32, float a,
    const __bf16 *cellSrcIter, __bf16 *cellDstLay, __bf16 *cellDstIter, __bf16 *cellWsGates)
{
    int64_t i;
    svfloat32_t aVec = svdup_n_f32_x(svptrue_b32(), a);
    int loopAugru = svcnth();
    int numF32 = svcntw();
    for (i = 0; i < len; i += loopAugru) {
        svbool_t pgAugruBf16 = svwhilelt_b16_s64(i, len);
        svbfloat16_t cellSrcIterArg = svld1_bf16(pgAugruBf16, cellSrcIter + i);

        svbool_t pgAugruBot = svwhilelt_b32_s64(i, len);
        svbool_t pgAugruTop = svwhilelt_b32_s64(i + numF32, len);
        svfloat32_t gateG0Bot = svld1_f32(pgAugruBot, scratch + i);
        svfloat32_t gateG0Top = svld1_f32(pgAugruTop, scratch + i + numF32);
        svfloat32_t argG2Bot = svld1_f32(pgAugruBot, scratch + 0x2 * len + i);
        svfloat32_t argG2Top = svld1_f32(pgAugruTop, scratch + 0x2 * len + i + numF32);
        svfloat32_t biasG2Bot = svld1_f32(pgAugruBot, biasF32 + 0x2 * len + i);
        svfloat32_t biasG2Top = svld1_f32(pgAugruTop, biasF32 + 0x2 * len + i + numF32);
        svfloat32_t cellSrcIterArgBot = SvCvtF32BF16BottomX(cellSrcIterArg);
        svfloat32_t cellSrcIterArgTop = SvCvtF32BF16TopX(cellSrcIterArg);
        svfloat32_t gateG2Bot;
        svfloat32_t gateG2Top;
        svfloat32_t htBotF32;
        svfloat32_t htTopF32;
        RnnAugruVCFWDCorePart2(func, pgAugruBot, gateG0Bot, argG2Bot, biasG2Bot, cellSrcIterArgBot, aVec,
            &gateG2Bot, &htBotF32);
        RnnAugruVCFWDCorePart2(func, pgAugruTop, gateG0Top, argG2Top, biasG2Top, cellSrcIterArgTop, aVec,
            &gateG2Top, &htTopF32);

        svbfloat16_t gateG2 = SvCvtBF16F32X(pgAugruBot, gateG2Bot, pgAugruTop, gateG2Top);
        svbfloat16_t ht = SvCvtBF16F32X(pgAugruBot, htBotF32, pgAugruTop, htTopF32);

        svst1_bf16(pgAugruBf16, cellDstLay + i, ht);
        if (cellDstIter) {
            svst1_bf16(pgAugruBf16, cellDstIter + i, ht);
        }
        if (cellWsGates) {
            svst1_bf16(pgAugruBf16, cellWsGates + 0x2 * len + i, gateG2);
        }
    }
}

template <typename SrcType, typename BiasType>
void RnnAugruVCFWDPart2Wrapper(CellPositionT cellPosition, const RnnCfg &rnn, float *scratch, const BiasType *cellBias,
    SrcType *augruAttenOff, const SrcType *cellSrcIter, SrcType *cellDstLay, SrcType *cellDstIter, SrcType *cellWsGates)
{
    SizeType srcIterLd = rnn.SrcIterLd(cellPosition);
    SizeType dstLayerLd = rnn.DstLayerLd(cellPosition);
    if (rnn.gatesFuncLinearMode) {
        auto func = [&rnn](svfloat32_t x, svbool_t pgAugru) { return svmul_n_f32_x(pgAugru, x, rnn.scales[2]); };
        const auto augruCallPart2 = [&rnn, func, scratch, cellBias, augruAttenOff, cellSrcIter,
            cellDstLay, cellDstIter, cellWsGates, srcIterLd, dstLayerLd](int i) {
            RnnAugruVCFWDPart2(func, rnn.dhc, scratch + i * rnn.scratchGatesLd, cellBias,
                ToFloat(augruAttenOff[i]), cellSrcIter + i * srcIterLd, cellDstLay + i * dstLayerLd,
                cellDstIter == nullptr ? nullptr : cellDstIter + i * rnn.dstIterLd,
                rnn.isTraining ? cellWsGates + i * rnn.wsGatesLd : nullptr);
        };
        MultiThreader(rnn.mb, augruCallPart2);
    } else {
        auto func = [](svfloat32_t x, svbool_t pgAugru) { return SvTanhF32X(x, pgAugru); };
        const auto augruCallPart2 = [&rnn, func, scratch, cellBias, augruAttenOff, cellSrcIter,
            cellDstLay, cellDstIter, cellWsGates, srcIterLd, dstLayerLd](int i) {
            RnnAugruVCFWDPart2(func, rnn.dhc, scratch + i * rnn.scratchGatesLd, cellBias,
                ToFloat(augruAttenOff[i]), cellSrcIter + i * srcIterLd, cellDstLay + i * dstLayerLd,
                cellDstIter == nullptr ? nullptr : cellDstIter + i * rnn.dstIterLd,
                rnn.isTraining ? cellWsGates + i * rnn.wsGatesLd : nullptr);
        };
        MultiThreader(rnn.mb, augruCallPart2);
    }
}

template void RnnAugruVCFWDPart2Wrapper<float, float>(CellPositionT cellPosition, const RnnCfg &rnn,
    float *scratch, const float *bias, float *augruAttenOff, const float *cellSrcIter,
    float *cellDstLay, float *cellDstIter, float *cellWsGates);
template void RnnAugruVCFWDPart2Wrapper<__bf16, __bf16>(CellPositionT cellPosition, const RnnCfg &rnn,
    float *scratch, const __bf16 *bias, __bf16 *augruAttenOff, const __bf16 *cellSrcIter,
    __bf16 *cellDstLay, __bf16 *cellDstIter, __bf16 *cellWsGates);
template void RnnAugruVCFWDPart2Wrapper<__bf16, float>(CellPositionT cellPosition, const RnnCfg &rnn,
    float *scratch, const float *bias, __bf16 *augruAttenOff, const __bf16 *cellSrcIter,
    __bf16 *cellDstLay, __bf16 *cellDstIter, __bf16 *cellWsGates);

inline void RnnAugruVCBWDF32Part1Core(svbool_t pgAugru, svfloat32_t ddl, svfloat32_t ddi, svfloat32_t gates0,
    svfloat32_t h, svfloat32_t gates2, svfloat32_t *dsIter, svfloat32_t *dG0, svfloat32_t *dG2)
{
    svfloat32_t dHt = svadd_f32_x(pgAugru, ddl, ddi);
    *dsIter = svmul_f32_x(pgAugru, dHt, gates0);

    svfloat32_t dG0Part11 = svsub_f32_x(pgAugru, h, gates2);
    svfloat32_t dG0Part1 = svmul_f32_x(pgAugru, dG0Part11, dHt);
    svfloat32_t dG0Part2 = svmls_f32_x(pgAugru, gates0, gates0, gates0);
    *dG0 = svmul_f32_x(pgAugru, dG0Part1, dG0Part2);

    svfloat32_t dG2Part1 = svmls_f32_x(pgAugru, dHt, gates0, dHt);
    svfloat32_t dG2Part2 = svmsb_n_f32_x(pgAugru, gates2, gates2, 1.0f);
    *dG2 = svmul_f32_x(pgAugru, dG2Part1, dG2Part2);
}

void RnnAuGruVCBWDF32Part1(const int64_t len, float *diffAugruAttenOff, float augruAttenOff,
    const float *cellWsGates, const float *cellSrcIter, const float *cellDiffDstLay,
    const float *cellDiffDstIter, float *cellDiffSrcIter, float *scratch)
{
    float diffAttentionSum = 0.0f;
    int loopAugru = svcntw();
    for (int64_t i = 0; i < len; i += loopAugru) {
        svbool_t pgAugru = svwhilelt_b32_s64(i, len);
        svfloat32_t h = svld1_f32(pgAugru, cellSrcIter + i);
        svfloat32_t ddl = svld1_f32(pgAugru, cellDiffDstLay + i);
        svfloat32_t ddi = svld1_f32(pgAugru, cellDiffDstIter + i);

        svfloat32_t gates0 = svld1_f32(pgAugru, cellWsGates + i);
        svfloat32_t gates2 = svld1_f32(pgAugru, cellWsGates + 0x2 * len + i);

        svfloat32_t dsIter;
        svfloat32_t dG0;
        svfloat32_t dG2;
        RnnAugruVCBWDF32Part1Core(pgAugru, ddl, ddi, gates0, h, gates2, &dsIter, &dG0, &dG2);

        svfloat32_t augruAttenOffVector = svdup_n_f32_z(pgAugru, augruAttenOff);

        diffAttentionSum -= svaddv_f32(pgAugru, svmul_f32_x(pgAugru, dG0, gates0));
        dG0 = svmls_f32_x(pgAugru, dG0, dG0, augruAttenOffVector);

        svst1_f32(pgAugru, cellDiffSrcIter + i, dsIter);
        svst1_f32(pgAugru, scratch + i, dG0);
        svst1_f32(pgAugru, scratch + 0x2 * len + i, dG2);
    }
    *diffAugruAttenOff = diffAttentionSum;
}

void RnnAuGruVCBWDBF16Part1(const int64_t len, float *diffAugruAttenOff,
    __bf16 augruAttenOff, const __bf16 *cellWsGates, const __bf16 *cellSrcIter, const float *cellDiffDstLay,
    const float *cellDiffDstIter, float *cellDiffSrcIter, __bf16 *scratch)
{
    float diffAttentionSum = 0.0f;
    int loopAugru = svcnth();
    int numF32 = svcntw();
    for (int64_t i = 0; i < len; i += loopAugru) {
        svbool_t pgAugru = svwhilelt_b16_s64(i, len);
        svbool_t pgAugruBot = svwhilelt_b32_s64(i, len);
        svbool_t pgAugruTop = svwhilelt_b32_s64(i + numF32, len);

        svbfloat16_t gates0 = svld1_bf16(pgAugru, cellWsGates + i);
        svbfloat16_t gates2 = svld1_bf16(pgAugru, cellWsGates + 0x2 * len + i);
        svbfloat16_t h = svld1_bf16(pgAugru, cellSrcIter + i);

        svfloat32_t hBot = SvCvtF32BF16BottomX(h);
        svfloat32_t hTop = SvCvtF32BF16TopX(h);
        svfloat32_t ddlBot = svld1_f32(pgAugruBot, cellDiffDstLay + i);
        svfloat32_t ddlTop = svld1_f32(pgAugruTop, cellDiffDstLay + i + numF32);
        svfloat32_t ddiBot = svld1_f32(pgAugruBot, cellDiffDstIter + i);
        svfloat32_t ddiTop = svld1_f32(pgAugruTop, cellDiffDstIter + i + numF32);
        svfloat32_t gates0Bot = SvCvtF32BF16BottomX(gates0);
        svfloat32_t gates0Top = SvCvtF32BF16TopX(gates0);
        svfloat32_t gates2Bot = SvCvtF32BF16BottomX(gates2);
        svfloat32_t gates2Top = SvCvtF32BF16TopX(gates2);

        svfloat32_t dsIterBot;
        svfloat32_t dG0Bot;
        svfloat32_t dG2Bot;
        svfloat32_t dsIterTop;
        svfloat32_t dG0Top;
        svfloat32_t dG2Top;
        RnnAugruVCBWDF32Part1Core(pgAugruBot, ddlBot, ddiBot, gates0Bot, hBot, gates2Bot,
            &dsIterBot, &dG0Bot, &dG2Bot);
        RnnAugruVCBWDF32Part1Core(pgAugruTop, ddlTop, ddiTop, gates0Top, hTop, gates2Top,
            &dsIterTop, &dG0Top, &dG2Top);

        svbfloat16_t augruAttenOffVector = svdup_n_bf16_z(pgAugru, augruAttenOff);

        svfloat32_t t1 = svmul_f32_x(pgAugruBot, dG0Bot, gates0Bot);
        svfloat32_t t2 = svmul_f32_x(pgAugruTop, dG0Top, gates0Top);

        diffAttentionSum -= svaddv_f32(pgAugruBot, t1);
        diffAttentionSum -= svaddv_f32(pgAugruTop, t2);

        dG0Bot = svmls_f32_x(pgAugruBot, dG0Bot, dG0Bot, SvCvtF32BF16BottomX(augruAttenOffVector));
        dG0Top = svmls_f32_x(pgAugruTop, dG0Top, dG0Top, SvCvtF32BF16TopX(augruAttenOffVector));

        svbfloat16_t dG0 = SvCvtBF16F32X(pgAugruBot, dG0Bot, pgAugruTop, dG0Top);
        svbfloat16_t dG2 = SvCvtBF16F32X(pgAugruBot, dG2Bot, pgAugruTop, dG2Top);

        svst1_f32(pgAugruBot, cellDiffSrcIter + i, dsIterBot);
        svst1_f32(pgAugruTop, cellDiffSrcIter + i + numF32, dsIterTop);
        svst1_bf16(pgAugru, scratch + i, dG0);
        svst1_bf16(pgAugru, scratch + 0x2 * len + i, dG2);
    }
    *diffAugruAttenOff = diffAttentionSum;
}

} // Rnn
} // Detail
} // KuDNN
