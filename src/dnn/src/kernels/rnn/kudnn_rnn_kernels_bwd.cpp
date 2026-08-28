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

#include <omp.h>
#include "kernels/rnn/kudnn_rnn_kernels.hpp"
#include "kernels/rnn/rnn_common.hpp"
#include "kernels/rnn/rnn_kernel_operators.hpp"

namespace KuDNN {
namespace Detail {
namespace Rnn {

template <typename SrcType, typename BiasType>
void LstmBWDWeightsPeepholeAndBias(const KuDNN::RnnCfg &rnn, CellPositionT cellPosition, BiasType *cellSrcIterC,
    BiasType *cellDstIterC, SrcType *scratch, float *cellDiffWeightPeephole, float *cellDiffBias)
{
    SizeType dstIterCLd = rnn.DstIterCLd(cellPosition);
    SizeType srcIterCLd = rnn.SrcIterCLd(cellPosition);
    const auto rnnGetWeightsPeepholeAndBias = [&rnn, cellPosition, cellSrcIterC, srcIterCLd, cellDstIterC, dstIterCLd,
        scratch, cellDiffWeightPeephole, cellDiffBias](int g, int j) {
        if (g < 0x3) {
            auto &cStates = g < 0x2 ? cellSrcIterC : cellDstIterC;
            const auto cStatesLd = g < 0x2 ? srcIterCLd : dstIterCLd;

            const SizeType sratchG = g < 0x2 ? g : 0x3;
            if (rnn.diffWeightsOverwrite && (cellPosition & KuDNN::Detail::CellPositionT::LAST_ITER)) {
                // Initialize diff bias if needed
                cellDiffWeightPeephole[g * rnn.dhc + j] = 0.0f;
            }
            for (int mb = 0; mb < rnn.mb; ++mb) {
                cellDiffWeightPeephole[g * rnn.dhc + j] += ToFloat(cStates[mb * cStatesLd + j])
                    * ToFloat(scratch[mb * rnn.scratchGatesLd + sratchG * rnn.dhc + j]);
            }
        }
        if (rnn.diffWeightsOverwrite && (cellPosition & KuDNN::Detail::CellPositionT::LAST_ITER)) {
            // Initialize diff bias if needed
            cellDiffBias[g * rnn.dhc + j] = 0.0f;
        }
        for (int mb = 0; mb < rnn.mb; ++mb) {
            cellDiffBias[g * rnn.dhc + j] += ToFloat(scratch[mb * rnn.scratchGatesLd +
                g * rnn.dhc + j]);
        }
    };
    MultiThreader(rnn.nGates, rnn.dhc, rnnGetWeightsPeepholeAndBias);
}

template <typename gates_t, typename acc_t>
void GatesReduction(const KuDNN::RnnCfg &rnn, CellPositionT cellPosition, const gates_t *scratch,
    acc_t *cellDiffBias)
{
    const auto rnnGetBias = [&rnn, cellPosition, scratch, cellDiffBias](int g, int j) {
        if (rnn.diffWeightsOverwrite && (cellPosition & KuDNN::Detail::CellPositionT::LAST_ITER)) {
            // Initialize diff bias if needed
            cellDiffBias[g * rnn.dhc + j] = 0.0f;
        }
        for (SizeType i = 0; i < static_cast<SizeType>(rnn.mb); i++) {
            cellDiffBias[g * rnn.dhc + j] += ToFloat(scratch[i * rnn.scratchGatesLd + g * rnn.dhc + j]);
        }
    };
    MultiThreader(rnn.nGates, rnn.dhc, rnnGetBias);
}

#define RNN_BWD_CELL_FUNC_SIG(f) \
    template <typename SrcType, typename BiasType> \
    inline void f(CellPositionT cellPosition, const RnnCfg &rnn, SrcType *cellSrcLay, SrcType *cellSrcIter, \
        SrcType *cellWeightLay, SrcType *cellWeightIter, SrcType *scratch, BiasType *cellSrcIterC, \
        BiasType *cellDstIterC, float *cellDiffDstLay, float *cellDiffDstIter, float *cellDiffDstIterC, \
        SrcType *wsGates, float *cellDiffSrcLay, float *cellDiffSrcIter, float *cellDiffSrcIterC, \
        float *cellDiffWeightLay, float *cellDiffWeightIter, float *cellDiffBias, SrcType *scratchCell, \
        const KuDNN::ActivateFunctionRNN actf, float *diffAugruAttenOff, SrcType *augruAttenOff, SrcType *wsGrid, \
        float *cellWeightPeephole, float *cellWeightProjection, float *projectionHt, \
        float *cellDiffWeightPeephole, float *cellDiffWeightProjection, float *scratchDiffHt, const RnnBwdGemm *rnnGemm)

RNN_BWD_CELL_FUNC_SIG(LstmCell)
{
    Service::Unused(actf, scratchCell, diffAugruAttenOff, augruAttenOff, wsGrid, rnnGemm);

    float beta = rnn.DiffWeightsBeta(cellPosition);
    if (cellWeightProjection) {
        const auto scratchCall = [&rnn, cellDiffDstLay, cellDiffDstIter, scratchDiffHt](int i) {
            for (SizeType j = 0; j < static_cast<SizeType>(rnn.dlc); j++) {
                float result = cellDiffDstLay[i * rnn.wsDiffStatesLayerLd + j] +
                    cellDiffDstIter[i * rnn.wsDiffStatesIterLd + j];
                scratchDiffHt[i * rnn.scratchDiffHtLd + j] = result;
            }
        };
        MultiThreader(rnn.mb, scratchCall);

        rnnGemm->ExecLstmScratchDiffGemm(scratchDiffHt, projectionHt, cellDiffWeightProjection, 1.0f, beta);
        rnnGemm->ExecLstmProjectionWeightGemm(cellWeightProjection, scratchDiffHt, cellDiffDstLay, 1.0f, 0.0f);
    }

    SizeType srcIterCLd = rnn.SrcIterCLd(cellPosition);
    SizeType dstIterCLd = rnn.DstIterCLd(cellPosition);
    const auto lstmCall = [&rnn, wsGates, cellSrcIterC, srcIterCLd, cellDstIterC, cellDiffDstLay, cellDiffDstIterC,
        cellDiffSrcIterC, scratch, cellWeightPeephole, cellWeightProjection, cellDiffDstIter, dstIterCLd](int i) {
        RnnLstmVCBWDWrapper(rnn.dhc, wsGates + i * rnn.wsGatesLd, cellSrcIterC + i * srcIterCLd,
            cellDstIterC + i * dstIterCLd, cellDiffDstLay + i * rnn.wsDiffStatesLayerLd,
            cellDiffDstIterC + i * rnn.wsDiffStatesIterCLd,
            cellDiffSrcIterC + i * rnn.wsDiffStatesIterCLd, scratch + i * rnn.scratchGatesLd,
            cellWeightPeephole, cellWeightProjection, cellDiffDstIter + i * rnn.wsDiffStatesIterLd);
    };
    MultiThreader(rnn.mb, lstmCall);

    rnnGemm->ExecWeightsIterGemm(cellWeightIter, scratch, cellDiffSrcIter, 1.0f, 0.0f);
    rnnGemm->ExecScratchLayGemm(cellPosition, scratch, cellSrcLay, cellDiffWeightLay, 1.0f, beta);
    rnnGemm->ExecWeightsLayGemm(cellWeightLay, scratch, cellDiffSrcLay, 1.0f, 0.0f);
    rnnGemm->ExecScratchIterGemm(cellPosition, scratch, cellSrcIter, cellDiffWeightIter, 1.0f, beta);

    if (cellDiffWeightPeephole) {
        LstmBWDWeightsPeepholeAndBias(rnn, cellPosition, cellSrcIterC, cellDstIterC, scratch, cellDiffWeightPeephole,
            static_cast<float *>(cellDiffBias));
    } else {
        GatesReduction(rnn, cellPosition, scratch, static_cast<float *>(cellDiffBias));
    }
}

RNN_BWD_CELL_FUNC_SIG(GruCell)
{
    Service::Unused(cellSrcIterC, cellDstIterC, cellDiffSrcIterC, cellDiffDstIterC, actf,
        diffAugruAttenOff, augruAttenOff, wsGrid, cellWeightPeephole, cellWeightProjection, projectionHt,
        cellDiffWeightPeephole, cellDiffWeightProjection, scratchDiffHt, rnnGemm);

    SizeType srcIterLd = rnn.SrcIterLd(cellPosition);

    float *dhGates = cellDiffSrcLay + rnn.wsStatesLayerLd * rnn.wsStatesLayerNld;
    const auto gruCallPart1 = [&rnn, wsGates, cellSrcIter, cellDiffDstLay, cellDiffDstIter, cellDiffSrcIter, scratch,
        srcIterLd](int i) {
        RnnGruVCBWDWrapperPart1(rnn.dhc, wsGates + i * rnn.wsGatesLd, cellSrcIter + i * srcIterLd,
            cellDiffDstLay + i * rnn.wsDiffStatesLayerLd, cellDiffDstIter + i * rnn.wsDiffStatesIterLd,
            cellDiffSrcIter + i * rnn.wsDiffStatesIterLd, scratch + i * rnn.scratchGatesLd);
    };
    MultiThreader(rnn.mb, gruCallPart1);

    rnnGemm->ExecWeightsIterGemm(cellWeightIter + 0x2 * rnn.dhc * rnn.weightsIterLd, scratch + 0x2 * rnn.dhc, dhGates,
        1.0f, 0.0f);

    SrcType *hG1 = scratchCell + rnn.wsStatesLayerLd * rnn.wsStatesLayerNld;
    const auto gruCallPart2 = [&rnn, dhGates, hG1, wsGates, cellSrcIter, cellDiffSrcIter, scratch, srcIterLd](int i) {
        RnnGruVCBWDWrapperPart2(rnn.dhc, dhGates + i * rnn.wsDiffStatesIterLd, hG1 + i * rnn.wsStatesLayerLd,
            wsGates + i * rnn.wsGatesLd, cellSrcIter + i * srcIterLd,
            cellDiffSrcIter + i * rnn.wsDiffStatesIterLd, scratch + i * rnn.scratchGatesLd);
    };
    MultiThreader(rnn.mb, gruCallPart2);
    const float beta = rnn.DiffWeightsBeta(cellPosition);
    rnnGemm->ExecScratchIterGemm(cellPosition, scratch, cellSrcIter, cellDiffWeightIter, 1.0f, beta);
    rnnGemm->ExecGruScratchGemm(scratch + 0x2 * rnn.dhc, scratchCell, cellDiffWeightIter + 0x2 * rnn.dhc, 1.0f, beta);
    rnnGemm->ExecWeightsIterGemm2(cellWeightIter, scratch, cellDiffSrcIter, 1.0f, 1.0f);
    rnnGemm->ExecScratchLayGemm(cellPosition, scratch, cellSrcLay, cellDiffWeightLay, 1.0f, beta);
    rnnGemm->ExecWeightsLayGemm(cellWeightLay, scratch, cellDiffSrcLay, 1.0f, 0.0f);
    GatesReduction(rnn, cellPosition, scratch, static_cast<float *>(cellDiffBias));
}

RNN_BWD_CELL_FUNC_SIG(RnnCell)
{
    Service::Unused(cellSrcIterC, cellDstIterC, cellDiffDstIterC, cellDiffSrcIterC, scratchCell,
        diffAugruAttenOff, augruAttenOff, wsGrid, cellWeightPeephole, cellWeightProjection, projectionHt,
        cellDiffWeightPeephole, cellDiffWeightProjection, scratchDiffHt, rnnGemm);

    using SvF32XFuncPtr = svfloat32_t (*)(svfloat32_t, svfloat32_t, svbool_t);
    SvF32XFuncPtr actPtr = nullptr;

    if (rnn.gatesFuncLinearMode) {
        actPtr = RnnLinearF32X;
    } else if (actf == KuDNN::ActivateFunctionRNN::TANH) {
        actPtr = &RnnDTanh;
    } else if (actf == KuDNN::ActivateFunctionRNN::LOGISTIC) {
        actPtr = &RnnDLogistic;
    } else {
        actPtr = &RnnDAlphaReLU;
    }

    const auto rnnCall = [&rnn, cellDiffDstLay, cellDiffDstIter, scratch, wsGates, actPtr](int i) {
        RnnVCBWDWrapper(rnn.dhc, cellDiffDstLay + i * rnn.wsDiffStatesLayerLd,
            cellDiffDstIter + i * rnn.wsDiffStatesIterLd, scratch + i * rnn.scratchGatesLd,
            wsGates + i * rnn.wsGatesLd, actPtr, rnn.isScale ? rnn.scales[0] : rnn.alpha);
    };
    MultiThreader(rnn.mb, rnnCall);
    const float beta = rnn.DiffWeightsBeta(cellPosition);
    rnnGemm->ExecWeightsIterGemm(cellWeightIter, scratch, cellDiffSrcIter, 1.0f, 0.0f);
    rnnGemm->ExecScratchLayGemm(cellPosition, scratch, cellSrcLay, cellDiffWeightLay, 1.0f, beta);
    rnnGemm->ExecWeightsLayGemm(cellWeightLay, scratch, cellDiffSrcLay, 1.0f, 0.0f);
    rnnGemm->ExecScratchIterGemm(cellPosition, scratch, cellSrcIter, cellDiffWeightIter, 1.0f, beta);
    GatesReduction(rnn, cellPosition, scratch, static_cast<float *>(cellDiffBias));
}

RNN_BWD_CELL_FUNC_SIG(AugruCell)
{
    Service::Unused(cellSrcIterC, cellDstIterC, cellDiffSrcIterC, cellDiffDstIterC, actf, wsGrid,
        cellWeightPeephole, cellWeightProjection, projectionHt, cellDiffWeightPeephole, cellDiffWeightProjection,
        scratchDiffHt, rnnGemm);

    SizeType srcIterLd = rnn.SrcIterLd(cellPosition);

    float *dhGates = cellDiffSrcLay + rnn.wsStatesLayerLd * rnn.wsStatesLayerNld;
    const auto augruCallPart1 = [&rnn, diffAugruAttenOff, augruAttenOff, wsGates, cellSrcIter,
        cellDiffDstLay, cellDiffDstIter, cellDiffSrcIter, scratch, srcIterLd](int i) {
        RnnAuGruVCBWDWrapperPart1(rnn.dhc, diffAugruAttenOff + i, augruAttenOff[i], wsGates + i * rnn.wsGatesLd,
            cellSrcIter + i * srcIterLd, cellDiffDstLay + i * rnn.wsDiffStatesLayerLd,
            cellDiffDstIter + i * rnn.wsDiffStatesIterLd, cellDiffSrcIter + i * rnn.wsDiffStatesIterLd,
            scratch + i * rnn.scratchGatesLd);
    };
    MultiThreader(rnn.mb, augruCallPart1);

    rnnGemm->ExecWeightsIterGemm(cellWeightIter + 0x2 * rnn.dhc * rnn.weightsIterLd, scratch + 0x2 * rnn.dhc, dhGates,
        1.0f, 0.0f);

    SrcType *hG1 = scratchCell + rnn.wsStatesLayerLd * rnn.wsStatesLayerNld;

    const auto augruCallPart2 = [&rnn, dhGates, hG1, wsGates, cellSrcIter, cellDiffSrcIter, scratch,
        srcIterLd](int i) {
        RnnGruVCBWDWrapperPart2(rnn.dhc, dhGates + i * rnn.wsDiffStatesIterLd, hG1 + i * rnn.wsStatesLayerLd,
            wsGates + i * rnn.wsGatesLd, cellSrcIter + i * srcIterLd, cellDiffSrcIter + i * rnn.wsDiffStatesIterLd,
            scratch + i * rnn.scratchGatesLd);
    };
    MultiThreader(rnn.mb, augruCallPart2);

    const float beta = rnn.DiffWeightsBeta(cellPosition);
    rnnGemm->ExecScratchIterGemm(cellPosition, scratch, cellSrcIter, cellDiffWeightIter, 1.0f, beta);
    rnnGemm->ExecGruScratchGemm(scratch + 0x2 * rnn.dhc, scratchCell, cellDiffWeightIter + 0x2 * rnn.dhc, 1.0f, beta);
    rnnGemm->ExecWeightsIterGemm2(cellWeightIter, scratch, cellDiffSrcIter, 1.0f, 1.0f);
    rnnGemm->ExecScratchLayGemm(cellPosition, scratch, cellSrcLay, cellDiffWeightLay, 1.0f, beta);
    rnnGemm->ExecWeightsLayGemm(cellWeightLay, scratch, cellDiffSrcLay, 1.0f, 0.0f);

    GatesReduction(rnn, cellPosition, scratch, static_cast<float *>(cellDiffBias));
}

template <typename SrcType>
inline void  LbrCellBiasUpdate(const IntType j, const KuDNN::RnnCfg &rnn, CellPositionT cellPosition,
    float *cellDiffBias, SrcType *scratchGatesR)
{
    if (rnn.diffWeightsOverwrite && (cellPosition & KuDNN::Detail::CellPositionT::LAST_ITER)) {
        // Initialize diff bias if needed
        cellDiffBias[0x3 * rnn.dhc + j] = 0.0f;
    }
    for (IntType i = 0; i < rnn.mb; i++) {
        cellDiffBias[0x3 * rnn.dhc + j] += ToFloat(scratchGatesR[i * rnn.wsGatesLd + 0x2 * rnn.dhc + j]);
    }
}

RNN_BWD_CELL_FUNC_SIG(LbrGruCell)
{
    Service::Unused(cellSrcIterC, cellDstIterC, cellDiffSrcIterC, cellDiffDstIterC,
        actf, diffAugruAttenOff, augruAttenOff, cellWeightPeephole, cellWeightProjection, projectionHt,
        cellDiffWeightPeephole, cellDiffWeightProjection, scratchDiffHt, rnnGemm);

    SizeType srcIterLd = rnn.SrcIterLd(cellPosition);
    SrcType *wsWhb = wsGrid;
    SrcType *scratchGatesR = scratchCell;

    const auto lbrgruCall = [&rnn, wsGates, cellSrcIter, cellDiffDstLay, cellDiffDstIter,
        cellDiffSrcIter, scratch, wsWhb, scratchGatesR, srcIterLd](int i) {
        RnnLbrgruVCBWDWrapper(rnn.dhc, wsGates + i * rnn.wsGatesLd, cellSrcIter + i * srcIterLd,
            cellDiffDstLay + i * rnn.wsDiffStatesLayerLd, cellDiffDstIter + i * rnn.wsDiffStatesIterLd,
            cellDiffSrcIter + i * rnn.wsDiffStatesIterLd, scratch + i * rnn.scratchGatesLd, wsWhb + i * rnn.dhc,
            scratchGatesR + i * rnn.wsGatesLd);
    };
    MultiThreader(rnn.mb, lbrgruCall);

    float beta = rnn.DiffWeightsBeta(cellPosition);
    rnnGemm->ExecScratchLayGemm(cellPosition, scratch, cellSrcLay, cellDiffWeightLay, 1.0f, beta);
    rnnGemm->ExecWeightsLayGemm(cellWeightLay, scratch, cellDiffSrcLay, 1.0f, 0.0f);
    rnnGemm->ExecWeightsIterGemm(cellWeightIter, scratchCell, cellDiffSrcIter, 1.0f, 1.0f);
    rnnGemm->ExecScratchIterGemm(cellPosition, scratchCell, cellSrcIter, cellDiffWeightIter, 1.0f, beta);

    GatesReduction(rnn, cellPosition, scratch, cellDiffBias);

    const auto biasCall = [&rnn, cellPosition, cellDiffBias, scratchGatesR](IntType j) {
        LbrCellBiasUpdate(j, rnn, cellPosition, cellDiffBias, scratchGatesR);
    };
    MultiThreader(rnn.dhc, biasCall);
}

RNN_BWD_CELL_FUNC_SIG(LbrAugruCell)
{
    Service::Unused(cellSrcIterC, cellDstIterC, cellDiffSrcIterC, cellDiffDstIterC, actf,
        cellWeightPeephole, cellWeightProjection, projectionHt, cellDiffWeightPeephole, cellDiffWeightProjection,
        scratchDiffHt, rnnGemm);

    SizeType srcIterLd = rnn.SrcIterLd(cellPosition);
    SrcType *wsWhb = wsGrid;
    SrcType *scratchGatesR = scratchCell;
    const auto lbraugruCall = [&rnn, diffAugruAttenOff, augruAttenOff, wsGates, cellSrcIter, cellDiffDstLay,
        cellDiffDstIter, cellDiffSrcIter, scratch, wsWhb, scratchGatesR, srcIterLd](int i) {
        RnnLbrAugruVCBWDWrapper(rnn.dhc, i, diffAugruAttenOff, augruAttenOff[i], wsGates + i * rnn.wsGatesLd,
            cellSrcIter + i * srcIterLd, cellDiffDstLay + i * rnn.wsDiffStatesLayerLd,
            cellDiffDstIter + i * rnn.wsDiffStatesIterLd, cellDiffSrcIter + i * rnn.wsDiffStatesIterLd,
            scratch + i * rnn.scratchGatesLd, wsWhb + i * rnn.dhc, scratchGatesR + i * rnn.wsGatesLd);
    };
    MultiThreader(rnn.mb, lbraugruCall);

    float beta = rnn.DiffWeightsBeta(cellPosition);
    rnnGemm->ExecScratchLayGemm(cellPosition, scratch, cellSrcLay, cellDiffWeightLay, 1.0f, beta);
    rnnGemm->ExecWeightsLayGemm(cellWeightLay, scratch, cellDiffSrcLay, 1.0f, 0.0f);
    rnnGemm->ExecWeightsIterGemm(cellWeightIter, scratchCell, cellDiffSrcIter, 1.0f, 1.0f);
    rnnGemm->ExecScratchIterGemm(cellPosition, scratchCell, cellSrcIter, cellDiffWeightIter, 1.0f, beta);

    GatesReduction(rnn, cellPosition, scratch, static_cast<float *>(cellDiffBias));

    const auto biasCall = [&rnn, cellPosition, cellDiffBias, scratchGatesR](IntType j) {
        LbrCellBiasUpdate(j, rnn, cellPosition, cellDiffBias, scratchGatesR);
    };
    MultiThreader(rnn.dhc, biasCall);
}

template <typename SrcType, typename BiasType>
using CellFuncType = void(*)(CellPositionT, const RnnCfg &, SrcType *, SrcType *, SrcType *, SrcType *, SrcType *,
    BiasType *, BiasType *, float *, float *, float *, SrcType *, float *, float *, float *, float *, float *, float *,
    SrcType *, const KuDNN::ActivateFunctionRNN, float *, SrcType *, SrcType *,
    float *, float *, float *, float *, float *, float *, const RnnBwdGemm *);

// Initialize weight pointers
static void InitWeightPointersBwd(const RnnCfg &rnn, IntType dir, IntType lay, const void **weight,
    void *diffBias, void **diffWeight, float *&cellWeightPeephole,
    float *&cellWeightProjection, float *&cellDiffWeightsPeephole,
    float *&cellDiffWeightsProjection, float *&cellDiffBias)
{
    cellWeightPeephole = weight[0x2] ?
        SafeCast<float *>(weight[0x2]) + (lay * rnn.nDir + dir) * 0x3 * rnn.dhc : nullptr;
    cellWeightProjection = weight[0x3] ?
        SafeCast<float *>(weight[0x3]) + (lay * rnn.nDir + dir) * rnn.weightsProjectionLd * rnn.dic :
        nullptr;
    cellDiffWeightsPeephole = diffWeight[0x2] ?
        SafeCast<float *>(diffWeight[0x2]) + (lay * rnn.nDir + dir) * 0x3 * rnn.dhc : nullptr;
    cellDiffWeightsProjection = diffWeight[0x3] ?
        SafeCast<float *>(diffWeight[0x3]) + (lay * rnn.nDir + dir) *
        rnn.diffWeightsProjectionLd * rnn.diffWeightsProjectionNld : nullptr;
    cellDiffBias = SafeCast<float *>(diffBias) + (lay * rnn.nDir + dir) * rnn.nBias * rnn.dhc;
}

// Initialize Cell pointers (srcLay/srcIter)
template <typename SrcType>
static void InitCellSrcPointersBwd(const RnnCfg &rnn, IntType dir, IntType lay, IntType iter,
    CellPositionT cellPosition, const void **src, const void **dst,
    SrcType *wsStates, SrcType *&cellSrcLay, SrcType *&cellSrcIter)
{
    cellSrcLay = wsStates +
        ((lay * rnn.nDir + dir) * (rnn.nIter + 1) + iter + 1) *
        rnn.wsStatesLayerLd * rnn.mb;
    cellSrcIter = wsStates +
        (((lay + 1) * rnn.nDir + dir) * (rnn.nIter + 1) + iter) *
        rnn.wsStatesLayerLd * rnn.mb;

    // LAST_ITER branch
    if (cellPosition & CellPositionT::LAST_ITER) {
        cellSrcLay = SafeCast<SrcType *>(dst[1]) + ((lay - 1) * rnn.nDir + dir) *
        rnn.dstIterLd * rnn.mb;
    }

    // LAST_LAYER branch
    if (cellPosition & CellPositionT::LAST_LAYER) {
        SizeType tmpIter = rnn.execDir != ExecutionDirectionT::R2L ? iter : rnn.nIter - 1 - iter;
        SizeType tmpLastIter = rnn.execDir != ExecutionDirectionT::R2L ? tmpIter - 1 : tmpIter + 1;
        if (rnn.execDir != ExecutionDirectionT::BI_SUM && rnn.execDir != ExecutionDirectionT::BI_CONCAT) {
            cellSrcIter = SafeCast<SrcType *>(dst[0]) + tmpLastIter * rnn.dstLayerCellLd;
        }
    }

    // FIRST_LAYER branch
    if (cellPosition & CellPositionT::FIRST_LAYER) {
        SizeType tmpIter = rnn.execDir != ExecutionDirectionT::R2L ? iter : rnn.nIter - 1 - iter;
        if (dir == 0) {
            cellSrcLay = SafeCast<SrcType *>(src[0]) + tmpIter * rnn.srcLayerCellLd;
        } else {
            cellSrcLay = SafeCast<SrcType *>(src[0]) + (rnn.nIter - 1 - tmpIter) * rnn.srcLayerCellLd;
        }
    }

    // FIRST_ITER branch
    if (cellPosition & CellPositionT::FIRST_ITER) {
        cellSrcIter = SafeCast<SrcType *>(src[1]) + (lay * rnn.nDir + dir) * rnn.srcIterLd * rnn.mb;
    }
}

// Initialize CellC pointers (dstIterC/srcIterC)
template <typename BiasType>
static void InitCellCPointersBwd(const RnnCfg &rnn, IntType dir, IntType lay, IntType iter,
    CellPositionT cellPosition, const void **src, const void **dst,
    BiasType *wsStatesIterC, BiasType *&cellDstIterC, BiasType *&cellSrcIterC)
{
    cellDstIterC = wsStatesIterC + (((lay + 1) * rnn.nDir + dir) *
        (rnn.nIter + 1) + iter + 1) * rnn.wsDiffStatesIterCLd * rnn.mb;
    cellSrcIterC = wsStatesIterC + (((lay + 1) * rnn.nDir + dir) *
        (rnn.nIter + 1) + iter) * rnn.wsDiffStatesIterCLd * rnn.mb;

    // LAST_ITER branch
    if (cellPosition & CellPositionT::LAST_ITER) {
        cellDstIterC = SafeCast<BiasType *>(dst[0x2]) + (lay * rnn.nDir + dir) * rnn.dstIterCLd * rnn.mb;
    }

    // FIRST_ITER branch
    if (cellPosition & CellPositionT::FIRST_ITER) {
        cellSrcIterC = SafeCast<BiasType *>(src[0x2]) + (lay * rnn.nDir + dir) * rnn.srcIterCLd * rnn.mb;
    }
}

template <typename SrcType, typename BiasType, CellFuncType<SrcType, BiasType> CellFunc>
void RnnBwdCommonRun(const void **src, const void **weight, const void **dst,
    const void *scratchGates, const void *augruAttention, void *diffAugruAttention,
    void *wsBase, const void **diffDst, void **diffSrc, void **diffWeight, void *diffBias,
    const void *wsDiffStatesLayer, const void *wsDiffStatesIter, const void *wsDiffStatesIterC,
    const RnnCfg &rnn, const ActivateFunctionRNN &myActf, const void *scratchCells, const void *scratchDiffHt,
    const RnnBwdGemm *rnnGemm)
{
    // 1. Initialize basic Workspace pointers
    SrcType *wsGates = SafeCast<SrcType *>(wsBase);
    SrcType *wsStates = SafeCast<SrcType *>(SafeCast<uint8_t *>(wsBase) + rnn.wsStatesLayerOffset);
    BiasType *wsStatesIterC = SafeCast<BiasType *>(SafeCast<uint8_t *>(wsBase) + rnn.wsStatesIterCOffset);
    SrcType *wsGrids = SafeCast<SrcType *>(SafeCast<uint8_t *>(wsBase) + rnn.wsGridCompOffset);
    SrcType *wsHt = SafeCast<SrcType *>(SafeCast<uint8_t *>(wsBase) + rnn.wsHtOffset);

    // 2. Initialize diffStates
    CopyInitLayer(rnn, SafeCast<float *>(wsDiffStatesLayer), SafeCast<float *>(diffDst[0]));
    CopyInitIter(rnn, SafeCast<float *>(wsDiffStatesIter), SafeCast<float *>(wsDiffStatesIterC),
        SafeCast<float *>(diffDst[1]), SafeCast<float *>(diffDst[0x2]));

    // 3. Core loop (only retain 3 layers of for loops + subfunction calls)
    for (IntType dir = 0; dir < rnn.nDir; dir++) {
        for (IntType lay = rnn.nLayer - 1; lay >= 0; lay--) {
            // 3.1 Initialize weight pointers
            float *cellWeightPeephole = nullptr;
            float *cellWeightProjection = nullptr;
            float *cellDiffWeightsPeephole = nullptr;
            float *cellDiffWeightsProjection = nullptr;
            float *cellDiffBias = nullptr;
            InitWeightPointersBwd(rnn, dir, lay, weight, diffBias, diffWeight, cellWeightPeephole,
                cellWeightProjection, cellDiffWeightsPeephole, cellDiffWeightsProjection, cellDiffBias);

            for (IntType iter = rnn.nIter - 1; iter >= 0; iter--) {
                // 3.2 Calculate cellPosition
                CellPositionT cellPosition = CalculateCellPosition(iter, lay, rnn.nIter, rnn.nLayer);

                // 3.3 Initialize Cell pointers
                SrcType *cellSrcLay = nullptr;
                SrcType *cellSrcIter = nullptr;
                InitCellSrcPointersBwd(rnn, dir, lay, iter, cellPosition, src, dst, wsStates, cellSrcLay, cellSrcIter);

                // 3.4 Initialize weight Lay/Iter pointers
                SrcType *cellWeightLay = SafeCast<SrcType *>(weight[0]) +
                    (lay * rnn.nDir + dir) * rnn.weightsLayerLd * rnn.nGates * rnn.dhc;
                SrcType *cellWeightIter = SafeCast<SrcType *>(weight[1]) +
                    (lay * rnn.nDir + dir) * rnn.weightsIterLd * rnn.nGates * rnn.dhc;

                // 3.5 Initialize CellC pointers
                BiasType *cellDstIterC = nullptr;
                BiasType *cellSrcIterC = nullptr;
                InitCellCPointersBwd(rnn, dir, lay, iter, cellPosition,
                                     src, dst, wsStatesIterC, cellDstIterC, cellSrcIterC);

                // 3.6 Initialize temporary pointers
                SrcType *scratch = SafeCast<SrcType *>(scratchGates);
                SrcType *scratchCell = SafeCast<SrcType *>(scratchCells);
                SrcType *wsGrid = wsGrids + (((lay * rnn.nDir) + dir) * rnn.nIter + iter) * rnn.wsPerCell;
                SrcType *augruAttenOff = SafeCast<SrcType *>(augruAttention) + iter * rnn.mb;
                float *diffAugruAttenOff = SafeCast<float *>(diffAugruAttention) + iter * rnn.mb;
                float *projectionHt = SafeCast<float *>(wsHt) +
                    ((lay * rnn.nDir + dir) * rnn.nIter + iter) * rnn.wsHtLd * rnn.wsHtNld;

                // 3.7 Core calculation
                CellFunc(cellPosition, rnn, cellSrcLay, cellSrcIter, cellWeightLay, cellWeightIter, scratch,
                    cellSrcIterC, cellDstIterC,
                    SafeCast<float *>(wsDiffStatesLayer) +
                        (((lay + 1) * rnn.nDir + dir) * (rnn.nIter + 1) + iter) * rnn.wsDiffStatesLayerLd * rnn.mb,
                    SafeCast<float *>(wsDiffStatesIter) +
                        ((lay * rnn.nDir + dir) * (rnn.nIter + 1) + iter + 1) * rnn.wsDiffStatesIterLd * rnn.mb,
                    SafeCast<float *>(wsDiffStatesIterC) +
                        ((lay * rnn.nDir + dir) * (rnn.nIter + 1) + iter + 1) * rnn.wsDiffStatesIterCLd * rnn.mb,
                    wsGates + ((lay * rnn.nDir + dir) * rnn.nIter + iter) * rnn.wsGatesLd * rnn.mb,
                    SafeCast<float *>(wsDiffStatesLayer) +
                        ((lay * rnn.nDir + dir) * (rnn.nIter + 1) + iter) * rnn.wsDiffStatesLayerLd * rnn.mb,
                    SafeCast<float *>(wsDiffStatesIter) +
                        ((lay * rnn.nDir + dir) * (rnn.nIter + 1) + iter) * rnn.wsDiffStatesIterLd * rnn.mb,
                    SafeCast<float *>(wsDiffStatesIterC) +
                        ((lay * rnn.nDir + dir) * (rnn.nIter + 1) + iter) * rnn.wsDiffStatesIterCLd * rnn.mb,
                    SafeCast<float *>(diffWeight[0]) + (lay * rnn.nDir + dir) * rnn.diffWeightsLayerLd * rnn.slc,
                    SafeCast<float *>(diffWeight[1]) + (lay * rnn.nDir + dir) * rnn.diffWeightsIterLd * rnn.sic,
                    cellDiffBias, scratchCell, myActf, diffAugruAttenOff, augruAttenOff, wsGrid,
                    cellWeightPeephole, cellWeightProjection, projectionHt,
                    cellDiffWeightsPeephole, cellDiffWeightsProjection, SafeCast<float *>(scratchDiffHt), rnnGemm);
            }
        }
    }

    // 4. Copy results
    CopyResLayer(rnn, SafeCast<float *>(diffSrc[0]), SafeCast<float *>(wsDiffStatesLayer));
    CopyResIter(rnn, SafeCast<float *>(diffSrc[1]), SafeCast<float *>(diffSrc[0x2]),
        SafeCast<float *>(wsDiffStatesIter), SafeCast<float *>(wsDiffStatesIterC));
}

const std::unordered_map<RnnKerParas, RnnKerBwdFunc, RnnKerParasHash, RnnKerParasEqual> RNN_KER_BWD_MAP = {
    // F32
    {{Element::TypeT::F32, Element::TypeT::F32, RnnAlgorithm::RNN},
        RnnBwdCommonRun<float, float, RnnCell<float, float>>},
    {{Element::TypeT::F32, Element::TypeT::F32, RnnAlgorithm::LSTM},
        RnnBwdCommonRun<float, float, LstmCell<float, float>>},
    {{Element::TypeT::F32, Element::TypeT::F32, RnnAlgorithm::GRU},
        RnnBwdCommonRun<float, float, GruCell<float, float>>},
    {{Element::TypeT::F32, Element::TypeT::F32, RnnAlgorithm::AUGRU},
        RnnBwdCommonRun<float, float, AugruCell<float, float>>},
    {{Element::TypeT::F32, Element::TypeT::F32, RnnAlgorithm::LBR_GRU},
        RnnBwdCommonRun<float, float, LbrGruCell<float, float>>},
    {{Element::TypeT::F32, Element::TypeT::F32, RnnAlgorithm::LBR_AUGRU},
        RnnBwdCommonRun<float, float, LbrAugruCell<float, float>>},
    // BF16
    {{Element::TypeT::BF16, Element::TypeT::BF16, RnnAlgorithm::RNN},
        RnnBwdCommonRun<__bf16, __bf16, RnnCell<__bf16, __bf16>>},
    {{Element::TypeT::BF16, Element::TypeT::BF16, RnnAlgorithm::LSTM},
        RnnBwdCommonRun<__bf16, __bf16, LstmCell<__bf16, __bf16>>},
    {{Element::TypeT::BF16, Element::TypeT::BF16, RnnAlgorithm::GRU},
        RnnBwdCommonRun<__bf16, __bf16, GruCell<__bf16, __bf16>>},
    {{Element::TypeT::BF16, Element::TypeT::BF16, RnnAlgorithm::AUGRU},
        RnnBwdCommonRun<__bf16, __bf16, AugruCell<__bf16, __bf16>>},
    {{Element::TypeT::BF16, Element::TypeT::BF16, RnnAlgorithm::LBR_GRU},
        RnnBwdCommonRun<__bf16, __bf16, LbrGruCell<__bf16, __bf16>>},
    {{Element::TypeT::BF16, Element::TypeT::BF16, RnnAlgorithm::LBR_AUGRU},
        RnnBwdCommonRun<__bf16, __bf16, LbrAugruCell<__bf16, __bf16>>},

    // BF16 F32
    {{Element::TypeT::BF16, Element::TypeT::F32, RnnAlgorithm::RNN},
        RnnBwdCommonRun<__bf16, float, RnnCell<__bf16, float>>},
    {{Element::TypeT::BF16, Element::TypeT::F32, RnnAlgorithm::LSTM},
        RnnBwdCommonRun<__bf16, float, LstmCell<__bf16, float>>},
    {{Element::TypeT::BF16, Element::TypeT::F32, RnnAlgorithm::GRU},
        RnnBwdCommonRun<__bf16, float, GruCell<__bf16, float>>},
    {{Element::TypeT::BF16, Element::TypeT::F32, RnnAlgorithm::AUGRU},
        RnnBwdCommonRun<__bf16, float, AugruCell<__bf16, float>>},
    {{Element::TypeT::BF16, Element::TypeT::F32, RnnAlgorithm::LBR_GRU},
        RnnBwdCommonRun<__bf16, float, LbrGruCell<__bf16, float>>},
    {{Element::TypeT::BF16, Element::TypeT::F32, RnnAlgorithm::LBR_AUGRU},
        RnnBwdCommonRun<__bf16, float, LbrAugruCell<__bf16, float>>},
};

} // Rnn
} // Detail
} // KuDNN
