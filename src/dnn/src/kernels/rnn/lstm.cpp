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

template <typename T1, typename T2>
inline void LstmVCFWDCore(T1 func1, T2 func2, const RnnCfg &rnn, svbool_t pg,
    svfloat32_t gateArgI, svfloat32_t gateArgF, svfloat32_t gateArgC, svfloat32_t gateArgO,
    svfloat32_t biasI, svfloat32_t biasF, svfloat32_t biasC, svfloat32_t biasO, svfloat32_t iterC,
    svfloat32_t *gateI, svfloat32_t *gateF, svfloat32_t *gateC, svfloat32_t *gateO, svfloat32_t *ht, svfloat32_t *ct)
{
    gateArgI = svadd_f32_x(pg, gateArgI, biasI);
    gateArgF = svadd_f32_x(pg, gateArgF, biasF);
    gateArgC = svadd_f32_x(pg, gateArgC, biasC);
    gateArgO = svadd_f32_x(pg, gateArgO, biasO);
    *gateI = func1(pg, gateArgI, rnn.scales[0x0]);
    *gateF = func1(pg, gateArgF, rnn.scales[0x1]);
    *gateO = func1(pg, gateArgO, rnn.scales[0x3]);
    *gateC = func2(pg, gateArgC, rnn.scales[0x2]);
    *ct = svmla_f32_x(pg, svmul_f32_x(pg, *gateF, iterC), *gateI, *gateC);
    *ht = svmul_f32_x(pg, *gateO, func2(pg, *ct, rnn.cscale));
}

template <typename T1, typename T2>
void RnnLstmVCFWD(T1 func1, T2 func2, const RnnCfg &rnn, const int64_t len, const float *scratch, const float *bias,
    const float *cellSrcIterC, float *cellDstLay, float *cellDstIter, float *cellDstIterC, float *cellWsGates)
{
    int64_t i;
    int loopLstm = svcntw();
    for (i = 0; i < len; i += loopLstm) {
        svbool_t pg = svwhilelt_b32_s64(i, len);
        svfloat32_t gateArgI = svld1_f32(pg, scratch + i);
        svfloat32_t gateArgF = svld1_f32(pg, scratch + len + i);
        svfloat32_t gateArgC = svld1_f32(pg, scratch + 0x2 * len + i);
        svfloat32_t gateArgO = svld1_f32(pg, scratch + 0x3 * len + i);
        svfloat32_t biasI = svld1_f32(pg, bias + i);
        svfloat32_t biasF = svld1_f32(pg, bias + len + i);
        svfloat32_t biasC = svld1_f32(pg, bias + 0x2 * len + i);
        svfloat32_t biasO = svld1_f32(pg, bias + 0x3 * len + i);
        svfloat32_t iterC = svld1_f32(pg, cellSrcIterC + i);

        svfloat32_t gateI;
        svfloat32_t gateF;
        svfloat32_t gateO;
        svfloat32_t gateC;
        svfloat32_t ht;
        svfloat32_t cState;
        LstmVCFWDCore(func1, func2, rnn, pg, gateArgI, gateArgF, gateArgC, gateArgO, biasI, biasF, biasC, biasO, iterC,
            &gateI, &gateF, &gateC, &gateO, &ht, &cState);

        svst1_f32(pg, cellDstLay + i, ht);
        svst1_f32(pg, cellDstIterC + i, cState);
        if (cellDstIter) {
            svst1_f32(pg, cellDstIter + i, ht);
        }
        if (cellWsGates) {
            svst1_f32(pg, cellWsGates + i, gateI);
            svst1_f32(pg, cellWsGates + len + i, gateF);
            svst1_f32(pg, cellWsGates + 0x2 * len + i, gateC);
            svst1_f32(pg, cellWsGates + 0x3 * len + i, gateO);
        }
    }
}

template <typename T1, typename T2>
void RnnLstmVCFWD(T1 func1, T2 func2, const RnnCfg &rnn, const int64_t len, const float *scratch, const __bf16 *bias,
    const __bf16 *cellSrcIterC, __bf16 *cellDstLay, __bf16 *cellDstIter, __bf16 *cellDstIterC, __bf16 *cellWsGates)
{
    int64_t i;
    int loopLstm = svcnth();
    int numF32 = svcntw();
    for (i = 0; i < len; i += loopLstm) {
        svbool_t pg = svwhilelt_b16_s64(i, len);
        svbfloat16_t iterC = svld1_bf16(pg, cellSrcIterC + i);
        svbfloat16_t biasI = svld1_bf16(pg, bias + i);
        svbfloat16_t biasF = svld1_bf16(pg, bias + len + i);
        svbfloat16_t biasC = svld1_bf16(pg, bias + 2 * len + i);
        svbfloat16_t biasO = svld1_bf16(pg, bias + 3 * len + i);

        svbool_t pgBot = svwhilelt_b32_s64(i, len);
        svbool_t pgTop = svwhilelt_b32_s64(i + numF32, len);
        svfloat32_t gateArgIBot = svld1_f32(pgBot, scratch + i);
        svfloat32_t gateArgITop = svld1_f32(pgTop, scratch + i + numF32);
        svfloat32_t gateArgFBot = svld1_f32(pgBot, scratch + len + i);
        svfloat32_t gateArgFTop = svld1_f32(pgTop, scratch + len + i + numF32);
        svfloat32_t gateArgCBot = svld1_f32(pgBot, scratch + 2 * len + i);
        svfloat32_t gateArgCTop = svld1_f32(pgTop, scratch + 2 * len + i + numF32);
        svfloat32_t gateArgOBot = svld1_f32(pgBot, scratch + 3 * len + i);
        svfloat32_t gateArgOTop = svld1_f32(pgTop, scratch + 3 * len + i + numF32);

        svfloat32_t iterCBot = SvCvtF32BF16BottomX(iterC);
        svfloat32_t iterCTop = SvCvtF32BF16TopX(iterC);
        svfloat32_t biasIBot = SvCvtF32BF16BottomX(biasI);
        svfloat32_t biasITop = SvCvtF32BF16TopX(biasI);
        svfloat32_t biasFBot = SvCvtF32BF16BottomX(biasF);
        svfloat32_t biasFTop = SvCvtF32BF16TopX(biasF);
        svfloat32_t biasCBot = SvCvtF32BF16BottomX(biasC);
        svfloat32_t biasCTop = SvCvtF32BF16TopX(biasC);
        svfloat32_t biasOBot = SvCvtF32BF16BottomX(biasO);
        svfloat32_t biasOTop = SvCvtF32BF16TopX(biasO);

        svfloat32_t gateIBot;
        svfloat32_t gateITop;
        svfloat32_t gateFBot;
        svfloat32_t gateFTop;
        svfloat32_t gateOBot;
        svfloat32_t gateOTop;
        svfloat32_t gateCBot;
        svfloat32_t gateCTop;
        svfloat32_t htBot;
        svfloat32_t htTop;
        svfloat32_t cStateBot;
        svfloat32_t cStateTop;
        LstmVCFWDCore(func1, func2, rnn, pgBot, gateArgIBot, gateArgFBot, gateArgCBot, gateArgOBot, biasIBot, biasFBot,
            biasCBot, biasOBot, iterCBot, &gateIBot, &gateFBot, &gateCBot, &gateOBot, &htBot, &cStateBot);
        LstmVCFWDCore(func1, func2, rnn, pgTop, gateArgITop, gateArgFTop, gateArgCTop, gateArgOTop, biasITop, biasFTop,
            biasCTop, biasOTop, iterCTop, &gateITop, &gateFTop, &gateCTop, &gateOTop, &htTop, &cStateTop);

        svbfloat16_t gateI = SvCvtBF16F32X(pgBot, gateIBot, pgTop, gateITop);
        svbfloat16_t gateF = SvCvtBF16F32X(pgBot, gateFBot, pgTop, gateFTop);
        svbfloat16_t gateC = SvCvtBF16F32X(pgBot, gateCBot, pgTop, gateCTop);
        svbfloat16_t gateO = SvCvtBF16F32X(pgBot, gateOBot, pgTop, gateOTop);
        svbfloat16_t ht = SvCvtBF16F32X(pgBot, htBot, pgTop, htTop);
        svbfloat16_t cState = SvCvtBF16F32X(pgBot, cStateBot, pgTop, cStateTop);

        svst1_bf16(pg, cellDstLay + i, ht);
        svst1_bf16(pg, cellDstIterC + i, cState);
        if (cellDstIter) {
            svst1_bf16(pg, cellDstIter + i, ht);
        }
        if (cellWsGates) {
            svst1_bf16(pg, cellWsGates + i, gateI);
            svst1_bf16(pg, cellWsGates + len + i, gateF);
            svst1_bf16(pg, cellWsGates + 0x2 * len + i, gateC);
            svst1_bf16(pg, cellWsGates + 0x3 * len + i, gateO);
        }
    }
}

template <typename T1, typename T2>
void RnnLstmVCFWD(T1 func1, T2 func2, const RnnCfg &rnn, const int64_t len, const float *scratch, const float *bias,
    const float *cellSrcIterC, __bf16 *cellDstLay, __bf16 *cellDstIter, float *cellDstIterC, __bf16 *cellWsGates)
{
    int64_t i;
    int loopLstm = svcnth();
    int numF32 = svcntw();
    for (i = 0; i < len; i += loopLstm) {
        svbool_t pgBf16 = svwhilelt_b16_s64(i, len);
        svbool_t pgBot = svwhilelt_b32_s64(i, len);
        svbool_t pgTop = svwhilelt_b32_s64(i + numF32, len);
        svfloat32_t gateArgIBot = svld1_f32(pgBot, scratch + i);
        svfloat32_t gateArgFBot = svld1_f32(pgBot, scratch + len + i);
        svfloat32_t gateArgITop = svld1_f32(pgTop, scratch + i + numF32);
        svfloat32_t gateArgFTop = svld1_f32(pgTop, scratch + len + i + numF32);
        svfloat32_t gateArgCBot = svld1_f32(pgBot, scratch + 2 * len + i);
        svfloat32_t gateArgCTop = svld1_f32(pgTop, scratch + 2 * len + i + numF32);
        svfloat32_t gateArgOBot = svld1_f32(pgBot, scratch + 3 * len + i);
        svfloat32_t gateArgOTop = svld1_f32(pgTop, scratch + 3 * len + i + numF32);

        svfloat32_t iterCBotFwd = svld1_f32(pgBot, cellSrcIterC + i);
        svfloat32_t iterCTopFwd = svld1_f32(pgTop, cellSrcIterC + i + numF32);
        svfloat32_t biasIBot = svld1_f32(pgBot, bias + i);
        svfloat32_t biasITop = svld1_f32(pgTop, bias + i + numF32);
        svfloat32_t biasFBot = svld1_f32(pgBot, bias + len + i);
        svfloat32_t biasFTop = svld1_f32(pgTop, bias + len + i + numF32);
        svfloat32_t biasCBotFwd = svld1_f32(pgBot, bias + 2 * len + i);
        svfloat32_t biasCTopFwd = svld1_f32(pgTop, bias + 2 * len + i + numF32);
        svfloat32_t biasOBot = svld1_f32(pgBot, bias + 3 * len + i);
        svfloat32_t biasOTop = svld1_f32(pgTop, bias + 3 * len + i + numF32);

        svfloat32_t gateIBotFwd;
        svfloat32_t gateFBotFwd;
        svfloat32_t gateOBot;
        svfloat32_t gateCBot;
        svfloat32_t htBot;
        svfloat32_t cStateBot;
        svfloat32_t gateITop;
        svfloat32_t gateFTop;
        svfloat32_t gateOTop;
        svfloat32_t gateCTop;
        svfloat32_t htTopFwd;
        svfloat32_t cStateTop;
        LstmVCFWDCore(func1, func2, rnn, pgBot, gateArgIBot, gateArgFBot, gateArgCBot, gateArgOBot, biasIBot, biasFBot,
            biasCBotFwd, biasOBot, iterCBotFwd, &gateIBotFwd, &gateFBotFwd, &gateCBot, &gateOBot, &htBot, &cStateBot);
        LstmVCFWDCore(func1, func2, rnn, pgTop, gateArgITop, gateArgFTop, gateArgCTop, gateArgOTop, biasITop, biasFTop,
            biasCTopFwd, biasOTop, iterCTopFwd, &gateITop, &gateFTop, &gateCTop, &gateOTop, &htTopFwd, &cStateTop);

        svbfloat16_t ht = SvCvtBF16F32X(pgBot, htBot, pgTop, htTopFwd);
        svbfloat16_t gateI = SvCvtBF16F32X(pgBot, gateIBotFwd, pgTop, gateITop);
        svbfloat16_t gateF = SvCvtBF16F32X(pgBot, gateFBotFwd, pgTop, gateFTop);
        svbfloat16_t gateC = SvCvtBF16F32X(pgBot, gateCBot, pgTop, gateCTop);
        svbfloat16_t gateO = SvCvtBF16F32X(pgBot, gateOBot, pgTop, gateOTop);

        svst1_bf16(pgBf16, cellDstLay + i, ht);
        svst1_f32(pgBot, cellDstIterC + i, cStateBot);
        svst1_f32(pgTop, cellDstIterC + i + numF32, cStateTop);
        if (cellDstIter) {
            svst1_bf16(pgBf16, cellDstIter + i, ht);
        }
        if (cellWsGates) {
            svst1_bf16(pgBf16, cellWsGates + i, gateI);
            svst1_bf16(pgBf16, cellWsGates + len + i, gateF);
            svst1_bf16(pgBf16, cellWsGates + 0x3 * len + i, gateO);
            svst1_bf16(pgBf16, cellWsGates + 0x2 * len + i, gateC);
        }
    }
}

template <typename SrcType, typename BiasType>
void RnnLstmVCFWDWrapper(CellPositionT cellPosition, const RnnCfg &rnn, const float *scratch,
    const BiasType *cellBias, const BiasType *cellSrcIterC, SrcType *cellDstLay, SizeType cellDstLayLd,
    SrcType *cellDstIter, BiasType *cellDstIterC, SrcType *cellWsGates)
{
    SizeType srcIterCLd = rnn.SrcIterCLd(cellPosition);
    SizeType dstIterCLd = rnn.DstIterCLd(cellPosition);
    if (rnn.gatesFuncLinearMode) {
        auto func1 = [](svbool_t pg, svfloat32_t x, float a) { return svmul_n_f32_x(pg, x, a); };
        auto func2 = [](svbool_t pg, svfloat32_t x, float a) { return svmul_n_f32_x(pg, x, a); };
        const auto lstmCall = [&rnn, func1, func2, scratch, cellBias, cellSrcIterC, cellDstLay, cellDstIter,
            cellDstIterC, cellWsGates, cellDstLayLd, srcIterCLd, dstIterCLd](int i) {
            RnnLstmVCFWD(func1, func2, rnn, rnn.dhc, scratch + i * rnn.scratchGatesLd, cellBias,
                cellSrcIterC + i * srcIterCLd, cellDstLay + i * cellDstLayLd,
                cellDstIter == nullptr ? nullptr : cellDstIter + i * rnn.dstIterLd,
                cellDstIterC + i * dstIterCLd, rnn.isTraining ? cellWsGates + i * rnn.wsGatesLd : nullptr);
        };
        MultiThreader(rnn.mb, lstmCall);
    } else {
        auto func1 = [](svbool_t pg, svfloat32_t x, float a) {
            Service::Unused(a); //  avoid warning
            return SvSigmoidF32X(svneg_f32_x(pg, x), pg);
        };
        auto func2 = [](svbool_t pg, svfloat32_t x, float a) {
            Service::Unused(a); //  avoid warning
            return SvTanhF32X(x, pg);
        };
        const auto lstmCall = [&rnn, func1, func2, scratch, cellBias, cellSrcIterC, cellDstLay, cellDstIter,
            cellDstIterC, cellWsGates, cellDstLayLd, srcIterCLd, dstIterCLd](int i) {
            RnnLstmVCFWD(func1, func2, rnn, rnn.dhc, scratch + i * rnn.scratchGatesLd, cellBias,
                cellSrcIterC + i * srcIterCLd, cellDstLay + i * cellDstLayLd,
                cellDstIter == nullptr ? nullptr : cellDstIter + i * rnn.dstIterLd,
                cellDstIterC + i * dstIterCLd, rnn.isTraining ? cellWsGates + i * rnn.wsGatesLd : nullptr);
        };
        MultiThreader(rnn.mb, lstmCall);
    }
}

template void RnnLstmVCFWDWrapper<float, float>(CellPositionT cellPosition, const RnnCfg &rnn, const float *scratch,
    const float *cellBias, const float *cellSrcIterC, float *cellDstLay, SizeType cellDstLayLd, float *cellDstIter,
    float *cellDstIterC, float *cellWsGates);
template void RnnLstmVCFWDWrapper<__bf16, __bf16>(CellPositionT cellPosition, const RnnCfg &rnn, const float *scratch,
    const __bf16 *cellBias, const __bf16 *cellSrcIterC, __bf16 *cellDstLay, SizeType cellDstLayLd, __bf16 *cellDstIter,
    __bf16 *cellDstIterC, __bf16 *cellWsGates);
template void RnnLstmVCFWDWrapper<__bf16, float>(CellPositionT cellPosition, const RnnCfg &rnn, const float *scratch,
    const float *cellBias, const float *cellSrcIterC, __bf16 *cellDstLay, SizeType cellDstLayLd, __bf16 *cellDstIter,
    float *cellDstIterC, __bf16 *cellWsGates);

template <typename T1, typename T2>
inline void RnnLstmPeepholeVCFWDCore(T1 func1, T2 func2, const RnnCfg &rnn, svbool_t pg,
    svfloat32_t gateArgI, svfloat32_t gateArgF, svfloat32_t gateArgC, svfloat32_t gateArgO,
    svfloat32_t peepholeI, svfloat32_t peepholeF, svfloat32_t peepholeO,
    svfloat32_t biasI, svfloat32_t biasF, svfloat32_t biasC, svfloat32_t biasO, svfloat32_t iterC,
    svfloat32_t *gateI, svfloat32_t *gateF, svfloat32_t *gateC, svfloat32_t *gateO, svfloat32_t *ht, svfloat32_t *ct)
{
    gateArgI = svmla_f32_x(pg, svadd_f32_x(pg, gateArgI, biasI), peepholeI, iterC);
    gateArgF = svmla_f32_x(pg, svadd_f32_x(pg, gateArgF, biasF), peepholeF, iterC);
    gateArgC = svadd_f32_x(pg, gateArgC, biasC);
    *gateI = func1(pg, gateArgI, rnn.scales[0x0]);
    *gateF = func1(pg, gateArgF, rnn.scales[0x1]);
    *gateC = func2(pg, gateArgC, rnn.scales[0x2]);
    *ct = svmla_f32_x(pg, svmul_f32_x(pg, *gateF, iterC), *gateI, *gateC);
    gateArgO = svmla_f32_x(pg, svadd_f32_x(pg, gateArgO, biasO), peepholeO, *ct);
    *gateO = func1(pg, gateArgO, rnn.scales[0x3]);
    *ht = svmul_f32_x(pg, *gateO, func2(pg, *ct, rnn.cscale));
}

template <typename T1, typename T2>
void RnnLstmPeepholeVCFWD(T1 func1, T2 func2, const RnnCfg &rnn, const int64_t len, const float *scratch,
    const float *bias, const float *cellSrcIterC, float *cellDstLay, float *cellDstIter,
    float *cellDstIterC, float *cellWeightPeephole, float *cellWsGates)
{
    int64_t i;
    int loopLstm = svcntw();
    for (i = 0; i < len; i += loopLstm) {
        svbool_t pg = svwhilelt_b32_s64(i, len);
        svfloat32_t gateArgI = svld1_f32(pg, scratch + i);
        svfloat32_t gateArgF = svld1_f32(pg, scratch + len + i);
        svfloat32_t gateArgC = svld1_f32(pg, scratch + 0x2 * len + i);
        svfloat32_t gateArgO = svld1_f32(pg, scratch + 0x3 * len + i);
        svfloat32_t peepholeI = svld1_f32(pg, cellWeightPeephole + i);
        svfloat32_t peepholeF = svld1_f32(pg, cellWeightPeephole + len + i);
        svfloat32_t peepholeO = svld1_f32(pg, cellWeightPeephole + 0x2 * len + i);
        svfloat32_t biasI = svld1_f32(pg, bias + i);
        svfloat32_t biasF = svld1_f32(pg, bias + len + i);
        svfloat32_t biasC = svld1_f32(pg, bias + 0x2 * len + i);
        svfloat32_t biasO = svld1_f32(pg, bias + 0x3 * len + i);
        svfloat32_t iterC = svld1_f32(pg, cellSrcIterC + i);

        svfloat32_t gateI;
        svfloat32_t gateF;
        svfloat32_t gateC;
        svfloat32_t gateO;
        svfloat32_t ht;
        svfloat32_t cState;
        RnnLstmPeepholeVCFWDCore(func1, func2, rnn, pg, gateArgI, gateArgF, gateArgC, gateArgO,
            peepholeI, peepholeF, peepholeO, biasI, biasF, biasC, biasO, iterC,
            &gateI, &gateF, &gateC, &gateO, &ht, &cState);

        svst1_f32(pg, cellDstIterC + i, cState);
        svst1_f32(pg, cellDstLay + i, ht);
        if (cellWsGates) {
            svst1_f32(pg, cellWsGates + i, gateI);
            svst1_f32(pg, cellWsGates + len + i, gateF);
            svst1_f32(pg, cellWsGates + 0x2 * len + i, gateC);
            svst1_f32(pg, cellWsGates + 0x3 * len + i, gateO);
        }
        if (cellDstIter) {
            svst1_f32(pg, cellDstIter + i, ht);
        }
    }
}

template <typename T1, typename T2>
void RnnLstmPeepholeVCFWD(T1 func1, T2 func2, const RnnCfg &rnn, const int64_t len, const float *scratch,
    const __bf16 *bias, const __bf16 *cellSrcIterC, __bf16 *cellDstLay, __bf16 *cellDstIter,
    __bf16 *cellDstIterC, float *cellWeightPeephole, __bf16 *cellWsGates)
{
    int64_t i;
    int numF32 = svcntw();
    int loopLstm = svcnth();
    for (i = 0; i < len; i += loopLstm) {
        svbool_t pgBf16 = svwhilelt_b16_s64(i, len);
        svbfloat16_t iterC = svld1_bf16(pgBf16, cellSrcIterC + i);
        svbfloat16_t biasI = svld1_bf16(pgBf16, bias + i);
        svbfloat16_t biasF = svld1_bf16(pgBf16, bias + len + i);
        svbfloat16_t biasC = svld1_bf16(pgBf16, bias + 2 * len + i);
        svbfloat16_t biasO = svld1_bf16(pgBf16, bias + 3 * len + i);

        svbool_t pgBotF32 = svwhilelt_b32_s64(i, len);
        svbool_t pgTopF32 = svwhilelt_b32_s64(i + numF32, len);
        svfloat32_t peepholeIBot = svld1_f32(pgBotF32, cellWeightPeephole + i);
        svfloat32_t peepholeITop = svld1_f32(pgTopF32, cellWeightPeephole + i + numF32);
        svfloat32_t peepholeFBot = svld1_f32(pgBotF32, cellWeightPeephole + len + i);
        svfloat32_t peepholeFTop = svld1_f32(pgTopF32, cellWeightPeephole + len + i + numF32);
        svfloat32_t peepholeOBot = svld1_f32(pgBotF32, cellWeightPeephole + 2 * len + i);
        svfloat32_t peepholeOTop = svld1_f32(pgTopF32, cellWeightPeephole + 2 * len + i + numF32);

        svfloat32_t gateArgIBot = svld1_f32(pgBotF32, scratch + i);
        svfloat32_t gateArgITop = svld1_f32(pgTopF32, scratch + i + numF32);
        svfloat32_t gateArgFBot = svld1_f32(pgBotF32, scratch + len + i);
        svfloat32_t gateArgFTop = svld1_f32(pgTopF32, scratch + len + i + numF32);
        svfloat32_t gateArgCBot = svld1_f32(pgBotF32, scratch + 2 * len + i);
        svfloat32_t gateArgCTop = svld1_f32(pgTopF32, scratch + 2 * len + i + numF32);
        svfloat32_t gateArgOBot = svld1_f32(pgBotF32, scratch + 3 * len + i);
        svfloat32_t gateArgOTop = svld1_f32(pgTopF32, scratch + 3 * len + i + numF32);

        svfloat32_t iterCBotF32 = SvCvtF32BF16BottomX(iterC);
        svfloat32_t biasIBot = SvCvtF32BF16BottomX(biasI);
        svfloat32_t biasFBot = SvCvtF32BF16BottomX(biasF);
        svfloat32_t biasCBot = SvCvtF32BF16BottomX(biasC);
        svfloat32_t biasOBot = SvCvtF32BF16BottomX(biasO);
        svfloat32_t iterCTopF32 = SvCvtF32BF16TopX(iterC);
        svfloat32_t biasITop = SvCvtF32BF16TopX(biasI);
        svfloat32_t biasFTop = SvCvtF32BF16TopX(biasF);
        svfloat32_t biasCTop = SvCvtF32BF16TopX(biasC);
        svfloat32_t biasOTop = SvCvtF32BF16TopX(biasO);

        svfloat32_t gateIBotF32;
        svfloat32_t gateFBotF32;
        svfloat32_t gateOBotF32;
        svfloat32_t gateCBotF32;
        svfloat32_t htBot;
        svfloat32_t cStateBot;
        RnnLstmPeepholeVCFWDCore(func1, func2, rnn, pgBotF32, gateArgIBot, gateArgFBot, gateArgCBot, gateArgOBot,
            peepholeIBot, peepholeFBot, peepholeOBot, biasIBot, biasFBot, biasCBot, biasOBot, iterCBotF32,
            &gateIBotF32, &gateFBotF32, &gateCBotF32, &gateOBotF32, &htBot, &cStateBot);
        svfloat32_t gateITop;
        svfloat32_t gateFTop;
        svfloat32_t gateOTop;
        svfloat32_t gateCTop;
        svfloat32_t htTop;
        svfloat32_t cStateTop;
        RnnLstmPeepholeVCFWDCore(func1, func2, rnn, pgTopF32, gateArgITop, gateArgFTop, gateArgCTop, gateArgOTop,
            peepholeITop, peepholeFTop, peepholeOTop, biasITop, biasFTop, biasCTop, biasOTop, iterCTopF32,
            &gateITop, &gateFTop, &gateCTop, &gateOTop, &htTop, &cStateTop);

        svbfloat16_t cState = SvCvtBF16F32X(pgBotF32, cStateBot, pgTopF32, cStateTop);
        svbfloat16_t gateI = SvCvtBF16F32X(pgBotF32, gateIBotF32, pgTopF32, gateITop);
        svbfloat16_t gateF = SvCvtBF16F32X(pgBotF32, gateFBotF32, pgTopF32, gateFTop);
        svbfloat16_t gateC = SvCvtBF16F32X(pgBotF32, gateCBotF32, pgTopF32, gateCTop);
        svbfloat16_t gateO = SvCvtBF16F32X(pgBotF32, gateOBotF32, pgTopF32, gateOTop);
        svbfloat16_t ht = SvCvtBF16F32X(pgBotF32, htBot, pgTopF32, htTop);

        svst1_bf16(pgBf16, cellDstLay + i, ht);
        svst1_bf16(pgBf16, cellDstIterC + i, cState);
        if (cellWsGates) {
            svst1_bf16(pgBf16, cellWsGates + i, gateI);
            svst1_bf16(pgBf16, cellWsGates + len + i, gateF);
            svst1_bf16(pgBf16, cellWsGates + 0x2 * len + i, gateC);
            svst1_bf16(pgBf16, cellWsGates + 0x3 * len + i, gateO);
        }
        if (cellDstIter) {
            svst1_bf16(pgBf16, cellDstIter + i, ht);
        }
    }
}

template <typename T1, typename T2>
void RnnLstmPeepholeVCFWD(T1 func1, T2 func2, const RnnCfg &rnn, const int64_t len,
    const float *scratch, const float *bias, const float *cellSrcIterC, __bf16 *cellDstLay,
    __bf16 *cellDstIter, float *cellDstIterC, float *cellWeightPeephole, __bf16 *cellWsGates)
{
    int64_t i;
    int loopLstm = svcnth();
    int numF32 = svcntw();
    for (i = 0; i < len; i += loopLstm) {
        svbool_t pg = svwhilelt_b16_s64(i, len);
        svbool_t pgBot = svwhilelt_b32_s64(i, len);
        svbool_t pgTop = svwhilelt_b32_s64(i + numF32, len);
        svfloat32_t peepholeIBot = svld1_f32(pgBot, cellWeightPeephole + i);
        svfloat32_t peepholeITop = svld1_f32(pgTop, cellWeightPeephole + i + numF32);
        svfloat32_t peepholeFBot = svld1_f32(pgBot, cellWeightPeephole + len + i);
        svfloat32_t peepholeFTop = svld1_f32(pgTop, cellWeightPeephole + len + i + numF32);
        svfloat32_t peepholeOBot = svld1_f32(pgBot, cellWeightPeephole + 2 * len + i);
        svfloat32_t peepholeOTop = svld1_f32(pgTop, cellWeightPeephole + 2 * len + i + numF32);

        svfloat32_t gateArgIBot = svld1_f32(pgBot, scratch + i);
        svfloat32_t gateArgITop = svld1_f32(pgTop, scratch + i + numF32);
        svfloat32_t gateArgFBot = svld1_f32(pgBot, scratch + len + i);
        svfloat32_t gateArgFTop = svld1_f32(pgTop, scratch + len + i + numF32);
        svfloat32_t gateArgCBot = svld1_f32(pgBot, scratch + 2 * len + i);
        svfloat32_t gateArgCTop = svld1_f32(pgTop, scratch + 2 * len + i + numF32);
        svfloat32_t gateArgOBot = svld1_f32(pgBot, scratch + 3 * len + i);
        svfloat32_t gateArgOTop = svld1_f32(pgTop, scratch + 3 * len + i + numF32);

        svfloat32_t gateITop;
        svfloat32_t gateFTop;
        svfloat32_t gateOTop;
        svfloat32_t gateCTop;
        svfloat32_t htTop;
        svfloat32_t cStateTop;
        svfloat32_t iterCBot = svld1_f32(pgBot, cellSrcIterC + i);
        svfloat32_t iterCTop = svld1_f32(pgTop, cellSrcIterC + i + numF32);
        svfloat32_t biasIBot = svld1_f32(pgBot, bias + i);
        svfloat32_t biasITop = svld1_f32(pgTop, bias + i + numF32);
        svfloat32_t biasFBot = svld1_f32(pgBot, bias + len + i);
        svfloat32_t biasFTop = svld1_f32(pgTop, bias + len + i + numF32);
        svfloat32_t biasCBot = svld1_f32(pgBot, bias + 2 * len + i);
        svfloat32_t biasCTop = svld1_f32(pgTop, bias + 2 * len + i + numF32);
        svfloat32_t biasOBot = svld1_f32(pgBot, bias + 3 * len + i);
        svfloat32_t biasOTop = svld1_f32(pgTop, bias + 3 * len + i + numF32);

        svfloat32_t gateIBotF32;
        svfloat32_t gateFBotF32;
        svfloat32_t gateOBotF32;
        svfloat32_t gateCBotF32;
        svfloat32_t htBot;
        svfloat32_t cStateBot;
        RnnLstmPeepholeVCFWDCore(func1, func2, rnn, pgBot, gateArgIBot, gateArgFBot, gateArgCBot, gateArgOBot,
            peepholeIBot, peepholeFBot, peepholeOBot, biasIBot, biasFBot, biasCBot, biasOBot, iterCBot,
            &gateIBotF32, &gateFBotF32, &gateCBotF32, &gateOBotF32, &htBot, &cStateBot);
        RnnLstmPeepholeVCFWDCore(func1, func2, rnn, pgTop, gateArgITop, gateArgFTop, gateArgCTop, gateArgOTop,
            peepholeITop, peepholeFTop, peepholeOTop, biasITop, biasFTop, biasCTop, biasOTop, iterCTop,
            &gateITop, &gateFTop, &gateCTop, &gateOTop, &htTop, &cStateTop);

        svbfloat16_t gateI = SvCvtBF16F32X(pgBot, gateIBotF32, pgTop, gateITop);
        svbfloat16_t gateF = SvCvtBF16F32X(pgBot, gateFBotF32, pgTop, gateFTop);
        svbfloat16_t gateC = SvCvtBF16F32X(pgBot, gateCBotF32, pgTop, gateCTop);
        svbfloat16_t gateO = SvCvtBF16F32X(pgBot, gateOBotF32, pgTop, gateOTop);
        svbfloat16_t ht = SvCvtBF16F32X(pgBot, htBot, pgTop, htTop);

        svst1_bf16(pg, cellDstLay + i, ht);
        svst1_f32(pgBot, cellDstIterC + i, cStateBot);
        svst1_f32(pgTop, cellDstIterC + i + numF32, cStateTop);
        if (cellDstIter) {
            svst1_bf16(pg, cellDstIter + i, ht);
        }
        if (cellWsGates) {
            svst1_bf16(pg, cellWsGates + i, gateI);
            svst1_bf16(pg, cellWsGates + 0x2 * len + i, gateC);
            svst1_bf16(pg, cellWsGates + len + i, gateF);
            svst1_bf16(pg, cellWsGates + 0x3 * len + i, gateO);
        }
    }
}

template <typename SrcType, typename BiasType>
void RnnLstmPeepholeVCFWDWrapper(CellPositionT cellPosition, const RnnCfg &rnn, const float *scratch,
    const BiasType *cellBias, const BiasType *cellSrcIterC, SrcType *cellDstLay, SizeType cellDstLayLd,
    SrcType *cellDstIter, BiasType *cellDstIterC, float *cellWeightPeephole, SrcType *cellWsGates)
{
    SizeType srcIterCLd = rnn.SrcIterCLd(cellPosition);
    SizeType dstIterCLd = rnn.DstIterCLd(cellPosition);
    if (rnn.gatesFuncLinearMode) {
        auto func1 = [](svbool_t pg, svfloat32_t x, float a) { return svmul_n_f32_x(pg, x, a); };
        auto func2 = [](svbool_t pg, svfloat32_t x, float a) { return svmul_n_f32_x(pg, x, a); };
        const auto lstmPeepholeCall = [&rnn, func1, func2, scratch, cellBias, cellSrcIterC, cellDstLay, cellDstIter,
            cellDstIterC, cellWeightPeephole, cellWsGates, cellDstLayLd, srcIterCLd, dstIterCLd](int i) {
            RnnLstmPeepholeVCFWD(func1, func2, rnn, rnn.dhc, scratch + i * rnn.scratchGatesLd, cellBias,
                cellSrcIterC + i * srcIterCLd, cellDstLay + i * cellDstLayLd,
                cellDstIter == nullptr ? nullptr : cellDstIter + i * rnn.dstIterLd,
                cellDstIterC + i * dstIterCLd, cellWeightPeephole,
                rnn.isTraining ? cellWsGates + i * rnn.wsGatesLd : nullptr);
        };
        MultiThreader(rnn.mb, lstmPeepholeCall);
    } else {
        auto func1 = [](svbool_t pg, svfloat32_t x, float a) {
            Service::Unused(a); //  avoid warning
            return SvSigmoidF32X(svneg_f32_x(pg, x), pg);
        };
        auto func2 = [](svbool_t pg, svfloat32_t x, float a) {
            Service::Unused(a); //  avoid warning
            return SvTanhF32X(x, pg);
        };
        const auto lstmPeepholeCall = [&rnn, func1, func2, scratch, cellBias, cellSrcIterC, cellDstLay, cellDstIter,
            cellDstIterC, cellWeightPeephole, cellWsGates, cellDstLayLd, srcIterCLd, dstIterCLd](int i) {
            RnnLstmPeepholeVCFWD(func1, func2, rnn, rnn.dhc, scratch + i * rnn.scratchGatesLd, cellBias,
                cellSrcIterC + i * srcIterCLd, cellDstLay + i * cellDstLayLd,
                cellDstIter == nullptr ? nullptr : cellDstIter + i * rnn.dstIterLd,
                cellDstIterC + i * dstIterCLd, cellWeightPeephole,
                rnn.isTraining ? cellWsGates + i * rnn.wsGatesLd : nullptr);
        };
        MultiThreader(rnn.mb, lstmPeepholeCall);
    }
}

template void RnnLstmPeepholeVCFWDWrapper<float, float>(CellPositionT cellPosition, const RnnCfg &rnn,
    const float *scratch, const float *cellBias, const float *cellSrcIterC, float *cellDstLay,
    SizeType cellDstLayLd, float *cellDstIter, float *cellDstIterC, float *cellWeightPeephole, float *cellWsGates);
template void RnnLstmPeepholeVCFWDWrapper<__bf16, __bf16>(CellPositionT cellPosition, const RnnCfg &rnn,
    const float *scratch, const __bf16 *cellBias, const __bf16 *cellSrcIterC, __bf16 *cellDstLay,
    SizeType cellDstLayLd, __bf16 *cellDstIter, __bf16 *cellDstIterC, float *cellWeightPeephole, __bf16 *cellWsGates);
template void RnnLstmPeepholeVCFWDWrapper<__bf16, float>(CellPositionT cellPosition, const RnnCfg &rnn,
    const float *scratch, const float *cellBias, const float *cellSrcIterC, __bf16 *cellDstLay,
    SizeType cellDstLayLd, __bf16 *cellDstIter, float *cellDstIterC, float *cellWeightPeephole, __bf16 *cellWsGates);

void RnnLstmVCBWDF32(const int64_t len, const float *cellWsGates, const float *cellSrcIterC,
    const float *cellDstIterC, const float *cellDiffDstLay, const float *cellDiffDstIterC,
    float *cellDiffSrcIterC, float *scratch, float *cellWeightPeephole, float *cellWeightProjection,
    const float *cellDiffDstIter)
{
    int loopLstm = svcntw();
    for (int64_t i = 0; i < len; i += loopLstm) {
        svbool_t pg = svwhilelt_b32_s64(i, len);
        svfloat32_t gateI = svld1_f32(pg, cellWsGates + i);
        svfloat32_t gateF = svld1_f32(pg, cellWsGates + len + i);
        svfloat32_t gateC = svld1_f32(pg, cellWsGates + 0x2 * len + i);
        svfloat32_t gateO = svld1_f32(pg, cellWsGates + 0x3 * len + i);
        svfloat32_t iterC = svld1_f32(pg, cellDstIterC + i);
        svfloat32_t srcIterC = svld1_f32(pg, cellDiffSrcIterC + i);
        svfloat32_t peepholeI;
        svfloat32_t peepholeF;
        svfloat32_t peepholeO;

        if (cellWeightPeephole) {
            peepholeI = svld1_f32(pg, cellWeightPeephole + i);
            peepholeF = svld1_f32(pg, cellWeightPeephole + len + i);
            peepholeO = svld1_f32(pg, cellWeightPeephole + 0x2 * len + i);
        }

        svfloat32_t tanhIterC = SvTanhF32X(iterC, pg);
        svfloat32_t dHt = svld1_f32(pg, cellDiffDstLay + i);
        if (!cellWeightProjection) {
            dHt = svadd_f32_x(pg, dHt, svld1_f32(pg, cellDiffDstIter + i));
        }

        svfloat32_t dDstIterC = svld1_f32(pg, cellDiffDstIterC + i);

        svfloat32_t dCt = svmsb_n_f32_x(pg, tanhIterC, tanhIterC, 1.0f);
        dCt = svadd_f32_x(pg, dDstIterC, svmul_f32_x(pg, svmul_f32_x(pg, dCt, gateO), dHt));

        svfloat32_t dG3 = svmls_f32_x(pg, gateO, gateO, gateO);
        dG3 = svmul_f32_x(pg, svmul_f32_x(pg, tanhIterC, dHt), dG3);

        if (cellWeightPeephole) {
            dCt = svadd_f32_x(pg, dCt, svmul_f32_x(pg, dG3, peepholeO));
        }

        svfloat32_t dG1 = svld1_f32(pg, cellSrcIterC + i);
        dG1 = svmul_f32_x(pg, svmul_f32_x(pg, dG1, dCt), svmls_f32_x(pg, gateF, gateF, gateF));
        svfloat32_t dG0 = svmul_f32_x(pg, svmul_f32_x(pg, gateC, dCt), svmls_f32_x(pg, gateI, gateI, gateI));
        svfloat32_t dG2 = svmul_f32_x(pg, svmul_f32_x(pg, gateI, dCt), svmsb_n_f32_x(pg, gateC, gateC, 1.0f));

        if (cellWeightPeephole) {
            srcIterC = svadd_f32_x(pg, svmul_f32_x(pg, gateF, dCt), svmul_f32_x(pg, peepholeF, dG1));
            srcIterC = svadd_f32_x(pg, srcIterC, svmul_f32_x(pg, peepholeI, dG0));
            svst1_f32(pg, cellDiffSrcIterC + i, srcIterC);
        } else {
            svst1_f32(pg, cellDiffSrcIterC + i, svmul_f32_x(pg, gateF, dCt));
        }

        svst1_f32(pg, scratch + i, dG0);
        svst1_f32(pg, scratch + len + i, dG1);
        svst1_f32(pg, scratch + 0x2 * len + i, dG2);
        svst1_f32(pg, scratch + 0x3 * len + i, dG3);
    }
}

inline void LoadPeephole(svbool_t pg, const float *peepholeI, const float *peepholeF, const float *peepholeO,
    svfloat32_t *peepholeIF32, svfloat32_t *peepholeFF32, svfloat32_t *peepholeOF32,
    float *cellWeightPeephole)
{
    if (cellWeightPeephole) {
        *peepholeIF32 = svld1_f32(pg, peepholeI);
        *peepholeFF32 = svld1_f32(pg, peepholeF);
        *peepholeOF32 = svld1_f32(pg, peepholeO);
    }
}

inline svfloat32_t ComputedHt(svbool_t pg, const float *cellDiffDstLay, const float *cellDiffDstIter)
{
    svfloat32_t dHt = svld1_f32(pg, cellDiffDstLay);
    dHt = svadd_f32_x(pg, dHt, svld1_f32(pg, cellDiffDstIter));
    return dHt;
}

inline svfloat32_t ComputedCtWithOptionalPeephole(svbool_t pg, svfloat32_t tanhIterC, svfloat32_t gateO,
    svfloat32_t dHt, svfloat32_t dDstIterC, svfloat32_t peepholeOF32, float *cellWeightPeephole, svfloat32_t *dG3)
{
    svfloat32_t dCt = svmsb_n_f32_x(pg, tanhIterC, tanhIterC, 1.0f);
    dCt = svadd_f32_x(pg, dDstIterC, svmul_f32_x(pg, svmul_f32_x(pg, dCt, gateO), dHt));
    *dG3 = svmul_f32_x(pg, svmul_f32_x(pg, tanhIterC, dHt), svmls_f32_x(pg, gateO, gateO, gateO));
    if (cellWeightPeephole) {
        dCt = svadd_f32_x(pg, dCt, svmul_f32_x(pg, svmul_f32_x(pg, svmul_f32_x(pg, tanhIterC, dHt),
                                svmls_f32_x(pg, gateO, gateO, gateO)), peepholeOF32));
    }
    return dCt;
}

inline void ComputedGValues(svbool_t pg, svfloat32_t gateC, svfloat32_t gateI, svfloat32_t gateF,
    svfloat32_t dCt, svfloat32_t h, svfloat32_t *dG0, svfloat32_t *dG1, svfloat32_t *dG2)
{
    *dG0 = svmul_f32_x(pg, svmul_f32_x(pg, gateC, dCt), svmls_f32_x(pg, gateI, gateI, gateI));
    *dG1 = svmul_f32_x(pg, svmul_f32_x(pg, h, dCt), svmls_f32_x(pg, gateF, gateF, gateF));
    *dG2 = svmul_f32_x(pg, svmul_f32_x(pg, gateI, dCt), svmsb_n_f32_x(pg, gateC, gateC, 1.0f));
}

inline void UpdateSrcIterC(svbool_t pg, svfloat32_t gateF, svfloat32_t dCt, svfloat32_t peepholeFF32,
    svfloat32_t peepholeIF32, svfloat32_t *dG0, svfloat32_t *dG1,
    svfloat32_t *srcIterC, float *cellDiffSrcIterC, float *cellWeightPeephole)
{
    if (cellWeightPeephole) {
        *srcIterC = svadd_f32_x(pg, svmul_f32_x(pg, gateF, dCt), svmul_f32_x(pg, peepholeFF32, *dG1));
        *srcIterC = svadd_f32_x(pg, *srcIterC, svmul_f32_x(pg, peepholeIF32, *dG0));
        svst1_f32(pg, cellDiffSrcIterC, *srcIterC);
    } else {
        svst1_f32(pg, cellDiffSrcIterC, svmul_f32_x(pg, gateF, dCt));
    }
}

inline void LstmVCBWDCore(const float *cellDiffDstLay, const float *cellDiffDstIter, float *cellDiffSrcIterC,
    const float *cellDiffDstIterC, svfloat32_t gateI, svfloat32_t gateF,
    svfloat32_t gateC, svfloat32_t gateO, svfloat32_t *dG0, svfloat32_t *dG1, svfloat32_t *dG2,
    svfloat32_t *dG3, svfloat32_t tanhIterC, float *cellWeightPeephole, svbool_t pg,
    float *peepholeI, float *peepholeF, float *peepholeO, svfloat32_t h)
{
    svfloat32_t dHt = ComputedHt(pg, cellDiffDstLay, cellDiffDstIter);
    svfloat32_t peepholeIF32;
    svfloat32_t peepholeFF32;
    svfloat32_t peepholeOF32;
    LoadPeephole(pg, peepholeI, peepholeF, peepholeO, &peepholeIF32, &peepholeFF32, &peepholeOF32, cellWeightPeephole);
    svfloat32_t dDstIterC = svld1_f32(pg, cellDiffDstIterC);
    svfloat32_t dCt = ComputedCtWithOptionalPeephole(pg, tanhIterC, gateO, dHt, dDstIterC, peepholeOF32,
        cellWeightPeephole, dG3);
    ComputedGValues(pg, gateC, gateI, gateF, dCt, h, dG0, dG1, dG2);
    svfloat32_t srcIterC = svld1_f32(pg, cellDiffSrcIterC);
    UpdateSrcIterC(pg, gateF, dCt, peepholeFF32, peepholeIF32, dG0, dG1, &srcIterC, cellDiffSrcIterC,
        cellWeightPeephole);
}

void RnnLstmVCBWDBF16(const int64_t len, const __bf16 *cellWsGates, const  __bf16 *cellSrcIterC,
    const __bf16 *cellDstIterC, const float *cellDiffDstLay, const float *cellDiffDstIterC,
    float *cellDiffSrcIterC, __bf16 *scratch, float *cellWeightPeephole, const float *cellDiffDstIter)
{
    int loopLstm = svcnth();
    int numF32 = svcntw();
    for (int64_t i = 0; i < len; i += loopLstm) {
        svbool_t pg = svwhilelt_b16_s64(i, len);
        svbool_t pgBot = svwhilelt_b32_s64(i, len);
        svbool_t pgTop = svwhilelt_b32_s64(i + numF32, len);

        svbfloat16_t gateI = svld1_bf16(pg, cellWsGates + i);
        svbfloat16_t gateF = svld1_bf16(pg, cellWsGates + len + i);
        svbfloat16_t gateC = svld1_bf16(pg, cellWsGates + 0x2 * len + i);
        svbfloat16_t gateO = svld1_bf16(pg, cellWsGates + 0x3 * len + i);
        svbfloat16_t iterC = svld1_bf16(pg, cellDstIterC + i);

        svfloat32_t gateIBot = SvCvtF32BF16BottomX(gateI);
        svfloat32_t gateITop = SvCvtF32BF16TopX(gateI);
        svfloat32_t gateFBot = SvCvtF32BF16BottomX(gateF);
        svfloat32_t gateFTop = SvCvtF32BF16TopX(gateF);
        svfloat32_t gateCBot = SvCvtF32BF16BottomX(gateC);
        svfloat32_t gateCTop = SvCvtF32BF16TopX(gateC);
        svfloat32_t gateOBot = SvCvtF32BF16BottomX(gateO);
        svfloat32_t gateOTop = SvCvtF32BF16TopX(gateO);
        svfloat32_t iterCBot = SvCvtF32BF16BottomX(iterC);
        svfloat32_t iterCTop = SvCvtF32BF16TopX(iterC);

        svfloat32_t tanhIterCBot = SvTanhF32X(iterCBot, pgBot);
        svfloat32_t tanhIterCTop = SvTanhF32X(iterCTop, pgTop);

        svbfloat16_t h = svld1_bf16(pg, cellSrcIterC + i);
        svfloat32_t hBot = SvCvtF32BF16BottomX(h);
        svfloat32_t hTop = SvCvtF32BF16TopX(h);

        svfloat32_t dG0Bot;
        svfloat32_t dG1Bot;
        svfloat32_t dG2Bot;
        svfloat32_t dG3Bot;
        LstmVCBWDCore(cellDiffDstLay + i, cellDiffDstIter + i, cellDiffSrcIterC + i,
            cellDiffDstIterC + i, gateIBot, gateFBot, gateCBot, gateOBot, &dG0Bot, &dG1Bot, &dG2Bot, &dG3Bot,
            tanhIterCBot, cellWeightPeephole, pgBot, cellWeightPeephole + i, cellWeightPeephole + len + i,
            cellWeightPeephole + len + i, hBot);

        svfloat32_t dG0Top;
        svfloat32_t dG1Top;
        svfloat32_t dG2Top;
        svfloat32_t dG3Top;
        LstmVCBWDCore(cellDiffDstLay + i + numF32, cellDiffDstIter + i + numF32, cellDiffSrcIterC + i + numF32,
            cellDiffDstIterC + i + numF32, gateITop, gateFTop, gateCTop, gateOTop, &dG0Top,
            &dG1Top, &dG2Top, &dG3Top, tanhIterCTop, cellWeightPeephole, pgTop,
            cellWeightPeephole + i + numF32, cellWeightPeephole + len + i + numF32,
            cellWeightPeephole + len + i + numF32, hTop);

        svbfloat16_t dG0 = SvCvtBF16F32X(pgBot, dG0Bot, pgTop, dG0Top);
        svbfloat16_t dG1 = SvCvtBF16F32X(pgBot, dG1Bot, pgTop, dG1Top);
        svbfloat16_t dG2 = SvCvtBF16F32X(pgBot, dG2Bot, pgTop, dG2Top);
        svbfloat16_t dG3 = SvCvtBF16F32X(pgBot, dG3Bot, pgTop, dG3Top);

        svst1_bf16(pg, scratch + i, dG0);
        svst1_bf16(pg, scratch + len + i, dG1);
        svst1_bf16(pg, scratch + 0x2 * len + i, dG2);
        svst1_bf16(pg, scratch + 0x3 * len + i, dG3);
    }
}

void RnnLstmVCBWDBF16F32(const int64_t len, const __bf16 *cellWsGates,
    const float *cellSrcIterC, const float *cellDstIterC, const float *cellDiffDstLay, const float *cellDiffDstIterC,
    float *cellDiffSrcIterC, __bf16 *scratch, float *cellWeightPeephole, const float *cellDiffDstIter)
{
    int loopLstm = svcnth();
    int numF32 = svcntw();
    for (int64_t i = 0; i < len; i += loopLstm) {
        svbool_t pg = svwhilelt_b16_s64(i, len);
        svbool_t pgf32 = svwhilelt_b32_s64(i, len);
        svbfloat16_t gateI = svld1_bf16(pg, cellWsGates + i);
        svbfloat16_t gateF = svld1_bf16(pg, cellWsGates + len + i);
        svbfloat16_t gateC = svld1_bf16(pg, cellWsGates + 0x2 * len + i);
        svbfloat16_t gateO = svld1_bf16(pg, cellWsGates + 0x3 * len + i);
        svfloat32_t iterC = svld1_f32(pgf32, cellDstIterC + i);

        svfloat32_t gateIBot = SvCvtF32BF16BottomX(gateI);
        svfloat32_t gateITop = SvCvtF32BF16TopX(gateI);
        svfloat32_t gateFBot = SvCvtF32BF16BottomX(gateF);
        svfloat32_t gateFTop = SvCvtF32BF16TopX(gateF);
        svfloat32_t gateCBot = SvCvtF32BF16BottomX(gateC);
        svfloat32_t gateCTop = SvCvtF32BF16TopX(gateC);
        svfloat32_t gateOBot = SvCvtF32BF16BottomX(gateO);
        svfloat32_t gateOTop = SvCvtF32BF16TopX(gateO);

        svfloat32_t tanhIterC = SvTanhF32X(iterC, pgf32);
        svfloat32_t h = svld1_f32(pgf32, cellSrcIterC + i);

        svfloat32_t dG0Bot;
        svfloat32_t dG1Bot;
        svfloat32_t dG2Bot;
        svfloat32_t dG3Bot;
        svbool_t pgBot = svwhilelt_b32_s64(i, len);
        LstmVCBWDCore(cellDiffDstLay + i, cellDiffDstIter + i, cellDiffSrcIterC + i,
            cellDiffDstIterC + i, gateIBot, gateFBot, gateCBot, gateOBot, &dG0Bot, &dG1Bot, &dG2Bot, &dG3Bot,
            tanhIterC, cellWeightPeephole, pgBot, cellWeightPeephole + i, cellWeightPeephole + len + i,
            cellWeightPeephole + len + i, h);

        svfloat32_t dG0Top;
        svfloat32_t dG1Top;
        svfloat32_t dG2Top;
        svfloat32_t dG3Top;
        svbool_t pgTop = svwhilelt_b32_s64(i + numF32, len);
        LstmVCBWDCore(cellDiffDstLay + i + numF32, cellDiffDstIter + i + numF32, cellDiffSrcIterC + i + numF32,
            cellDiffDstIterC + i + numF32, gateITop, gateFTop, gateCTop, gateOTop, &dG0Top,
            &dG1Top, &dG2Top, &dG3Top, tanhIterC, cellWeightPeephole, pgTop,
            cellWeightPeephole + i + numF32, cellWeightPeephole + len + i + numF32,
            cellWeightPeephole + len + i + numF32, h);

        svbfloat16_t dG3 = SvCvtBF16F32X(pgBot, dG3Bot, pgTop, dG3Top);
        svbfloat16_t dG0 = SvCvtBF16F32X(pgBot, dG0Bot, pgTop, dG0Top);
        svbfloat16_t dG1 = SvCvtBF16F32X(pgBot, dG1Bot, pgTop, dG1Top);
        svbfloat16_t dG2 = SvCvtBF16F32X(pgBot, dG2Bot, pgTop, dG2Top);

        svst1_bf16(pg, scratch + i, dG0);
        svst1_bf16(pg, scratch + len + i, dG1);
        svst1_bf16(pg, scratch + 0x2 * len + i, dG2);
        svst1_bf16(pg, scratch + 0x3 * len + i, dG3);
    }
}

} // Rnn
} // Detail
} // KuDNN
