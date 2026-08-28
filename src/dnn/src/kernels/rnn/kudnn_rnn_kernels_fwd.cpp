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

#include "kernels/rnn/kudnn_rnn_kernels.hpp"
#include "kernels/rnn/rnn_common.hpp"
#include "kernels/rnn/rnn_kernel_operators.hpp"

namespace KuDNN {
namespace Detail {
namespace Rnn {

template <>
void ArrayAdd<__bf16>(__bf16 *dst, const __bf16 *src, SizeType len)
{
    SizeType i;
    int numBF16 = svcnth();
    int numF32 = svcntw();
    for (i = 0; i < len; i += numBF16) {
        svbool_t pg = svwhilelt_b16_u64(i, len);
        svbfloat16_t x = svld1_bf16(pg, src + i);
        svbfloat16_t y = svld1_bf16(pg, dst + i);
        svfloat32_t xBot = SvCvtF32BF16BottomX(x);
        svfloat32_t yBot = SvCvtF32BF16BottomX(y);
        svfloat32_t xTop = SvCvtF32BF16TopX(x);
        svfloat32_t yTop = SvCvtF32BF16TopX(y);

        svbool_t pgBot = svwhilelt_b32_u64(i, len);
        svbool_t pgTop = svwhilelt_b32_u64(i + numF32, len);
        yBot = svadd_f32_x(pgBot, xBot, yBot);
        yTop = svadd_f32_x(pgTop, xTop, yTop);

        svbfloat16_t res = SvCvtBF16F32X(pgBot, yBot, pgTop, yTop);
        svst1_bf16(pg, dst + i, res);
    }
}

#define RNN_FWD_CELL_FUNC_SIG(f) \
    template <typename SrcType, typename BiasType> \
    void f(CellPositionT cellPosition, const RnnCfg &rnn, SrcType *cellSrcLay, SrcType *cellSrcIter, \
        SrcType *cellDstLay, SrcType *cellDstIter, SrcType *cellWeightLay, SrcType *cellWeightIter, \
        BiasType *cellBias, float *scratch, BiasType *cellSrcIterC, BiasType *cellDstIterC, \
        float *cellWeightPeephole, float *cellWeightProjection, SrcType *projectionHt, \
        const KuDNN::ActivateFunctionRNN myActf, float *scratchCell, SrcType *augruAttenOff, SrcType *cellWsGate, \
        SrcType *cellWsGrid, const RnnGemm *rnnGemm)

RNN_FWD_CELL_FUNC_SIG(LstmCell)
{
    Service::Unused(myActf, scratchCell, augruAttenOff, cellWsGrid, rnnGemm);
    SizeType dstLayerLd = rnn.DstLayerLd(cellPosition);
    rnnGemm->ExecLayerGemm(cellPosition, cellWeightLay, cellSrcLay, scratch, 1.0, 0.0);
    rnnGemm->ExecIterGemm(cellPosition, cellWeightIter, cellSrcIter, scratch, 1.0, 1.0);
    SrcType *tmpDstLay = cellWeightProjection ? projectionHt : cellDstLay;
    SrcType *tmpDstIter = cellWeightProjection ? nullptr : cellDstIter;
    SizeType tmpDstLayerLd = cellWeightProjection ? rnn.projHtLd : dstLayerLd;

    if (cellWeightPeephole) {
        RnnLstmPeepholeVCFWDWrapper(cellPosition, rnn, scratch, cellBias, cellSrcIterC, tmpDstLay, tmpDstLayerLd,
            tmpDstIter, cellDstIterC, cellWeightPeephole, cellWsGate);
    } else {
        RnnLstmVCFWDWrapper(cellPosition, rnn, scratch, cellBias, cellSrcIterC,
            tmpDstLay, tmpDstLayerLd, tmpDstIter, cellDstIterC, cellWsGate);
    }
    if (cellWeightProjection) {
        rnnGemm->ExecLstmProjectionGemm(cellPosition, cellWeightProjection, tmpDstLay, cellDstLay, 1.0f, 0.0f);
        if (cellDstIter) {
            const auto lstmCopy = [&rnn, cellDstIter, cellDstLay, dstLayerLd](int i) {
                ArrayCopy(cellDstIter + i * rnn.dstIterLd, cellDstLay + i * dstLayerLd, rnn.dic);
            };
            MultiThreader(rnn.mb, lstmCopy);
        }
    }
}

RNN_FWD_CELL_FUNC_SIG(RnnCell)
{
    Service::Unused(projectionHt, scratchCell, cellWeightProjection, cellWeightPeephole, cellDstIterC,
        cellSrcIterC, augruAttenOff, cellWsGrid, rnnGemm);
    using SvF32XFunc = svfloat32_t (*)(svfloat32_t, svfloat32_t, svbool_t);
    SvF32XFunc actFunc = nullptr;

    if (rnn.gatesFuncLinearMode) {
        actFunc = RnnLinearF32X;
    } else if (myActf == KuDNN::ActivateFunctionRNN::TANH) {
        actFunc = &RnnTanhFWDF32X;
    } else if (myActf == KuDNN::ActivateFunctionRNN::LOGISTIC) {
        actFunc = &RnnLogisticFWDF32X;
    } else {
        actFunc = &RnnReLUFWDF32X;
    }

    // gemm Wx[0-2],x
    rnnGemm->ExecLayerGemm(cellPosition, cellWeightLay, cellSrcLay, scratch, 1.0, 0.0);
    // gemm Wh[0-1],h
    rnnGemm->ExecIterGemm(cellPosition, cellWeightIter, cellSrcIter, scratch, 1.0, 1.0);

    SizeType dstLayerLd = rnn.DstLayerLd(cellPosition);
    const auto rnnCall = [&rnn, scratch, cellBias, cellDstLay, cellDstIter, cellWsGate, actFunc, dstLayerLd](int i) {
        VanillaRnnVCFWDWrapper(rnn.dhc,
            scratch + i * rnn.scratchGatesLd,
            cellBias,
            cellDstLay + i * dstLayerLd,
            cellDstIter == nullptr ? nullptr : cellDstIter + i * rnn.dstIterLd,
            rnn.isTraining ? cellWsGate + i * rnn.wsGatesLd : nullptr,
            actFunc, rnn.isScale ? rnn.scales[0] : rnn.alpha);
    };
    MultiThreader(rnn.mb, rnnCall);
}

RNN_FWD_CELL_FUNC_SIG(GruCell)
{
    Service::Unused(myActf, projectionHt, scratchCell, cellWeightProjection, cellWeightPeephole,
        cellDstIterC, cellSrcIterC, augruAttenOff, cellWsGrid, rnnGemm);

    // gemm Wx[0-2],x
    rnnGemm->ExecLayerGemm(cellPosition, cellWeightLay, cellSrcLay, scratch, 1.0, 0.0);
    // gemm Wh[0-1],h
    rnnGemm->ExecIterGemm(cellPosition, cellWeightIter, cellSrcIter, scratch, 1.0, 1.0);
    RnnGruVCFWDPart1Wrapper(cellPosition, rnn, scratch, cellBias, cellSrcIter, cellDstLay, cellDstIter, cellWsGate);

    // gemm Wh[2],h~t
    rnnGemm->ExecGruGemm(cellPosition, cellWeightIter + 0x2 * rnn.dhc, cellDstLay, scratch + 0x2 * rnn.dhc, 1.0, 1.0);
    RnnGruVCFWDPart2Wrapper(cellPosition, rnn, scratch, cellBias, cellSrcIter, cellDstLay, cellDstIter, cellWsGate);
}

RNN_FWD_CELL_FUNC_SIG(AugruCell)
{
    Service::Unused(myActf, projectionHt, scratchCell, cellWeightProjection, cellWeightPeephole, cellDstIterC,
        cellSrcIterC, cellWsGrid, rnnGemm);
    // gemm Wx[0-2],x
    rnnGemm->ExecLayerGemm(cellPosition, cellWeightLay, cellSrcLay, scratch, 1.0, 0.0);

    // gemm Wh[0-1],h
    rnnGemm->ExecIterGemm(cellPosition, cellWeightIter, cellSrcIter, scratch, 1.0, 1.0);
    RnnGruVCFWDPart1Wrapper(cellPosition, rnn, scratch, cellBias, cellSrcIter, cellDstLay, cellDstIter, cellWsGate);

    // gemm Wh[2],h~t
    rnnGemm->ExecGruGemm(cellPosition, cellWeightIter + 0x2 * rnn.dhc, cellDstLay, scratch + 0x2 * rnn.dhc, 1.0, 1.0);
    RnnAugruVCFWDPart2Wrapper(cellPosition, rnn, scratch, cellBias, augruAttenOff, cellSrcIter,
        cellDstLay, cellDstIter, cellWsGate);
}

RNN_FWD_CELL_FUNC_SIG(LbrGruCell)
{
    Service::Unused(myActf, projectionHt, cellWeightProjection, cellWeightPeephole, cellDstIterC,
        cellSrcIterC, augruAttenOff, rnnGemm);
    // gemm Wx[0-2],x
    rnnGemm->ExecLayerGemm(cellPosition, cellWeightLay, cellSrcLay, scratch, 1.0, 0.0);
    // gemm Wh[0-1],h
    rnnGemm->ExecIterGemm(cellPosition, cellWeightIter, cellSrcIter, scratchCell, 1.0, 0.0);
    RnnLbrGruVCFWDWrapper(cellPosition, rnn, scratch, scratchCell, cellBias, cellSrcIter, cellDstLay, cellDstIter,
        cellWsGate, cellWsGrid);
}

RNN_FWD_CELL_FUNC_SIG(LbrAugruCell)
{
    Service::Unused(myActf, projectionHt, cellWeightProjection, cellWeightPeephole, cellDstIterC,
        cellSrcIterC, rnnGemm);
    // gemm Wx[0-2],x
    rnnGemm->ExecLayerGemm(cellPosition, cellWeightLay, cellSrcLay, scratch, 1.0, 0.0);
    // gemm Wh[0-1],h
    rnnGemm->ExecIterGemm(cellPosition, cellWeightIter, cellSrcIter, scratchCell, 1.0, 0.0);
    RnnLbrAugruVCFWDWrapper(cellPosition, rnn, scratch, scratchCell, cellBias, augruAttenOff, cellSrcIter, cellDstLay,
        cellDstIter, cellWsGate, cellWsGrid);
}

template <typename SrcType, typename BiasType>
using CellFuncType = void(*)(CellPositionT, const RnnCfg &, SrcType *, SrcType *, SrcType *, SrcType *, SrcType *,
    SrcType *, BiasType *, float *, BiasType *, BiasType *, float *, float *, SrcType *,
    const KuDNN::ActivateFunctionRNN, float *, SrcType *, SrcType *, SrcType *, const RnnGemm *);

// Initializes workspace pointers based on training/inference mode
template <typename SrcType, typename BiasType>
static void InitWsPointers(const RnnCfg &rnn, void *wsBase, const void *&wsStates,
    SrcType *&wsGates, SrcType *&wsGrids, BiasType *&wsStatesIterC, SrcType *&wsHt)
{
    if (!rnn.isTraining) {
        wsGates = nullptr;
        wsGrids = nullptr;
        wsStatesIterC = nullptr;
        wsHt = nullptr;
        return;
    }

    // Stores gate calculation results, which are required for each algorithm.
    wsGates = SafeCast<SrcType *>(wsBase);
    wsStates = SafeCast<const void *>(SafeCast<uint8_t *>(wsBase) + rnn.wsStatesLayerOffset);

    // Space used only by lstm during FWD_D training.
    wsStatesIterC = SafeCast<BiasType *>(SafeCast<uint8_t *>(wsBase) + rnn.wsStatesIterCOffset);

    // Space used only by lbr during FWD_D training.
    wsGrids = SafeCast<SrcType *>(SafeCast<uint8_t *>(wsBase) + rnn.wsGridCompOffset);

    // projectionHt requires only a space in FWD_I to store the calculation result.
    // However, in inference mode, a large space is required to store the settlement result of each cell.
    wsHt = SafeCast<SrcType *>(SafeCast<uint8_t *>(wsBase) + rnn.wsHtOffset);
}

// Initializes weight pointers
static void InitWeightPointers(const RnnCfg &rnn, IntType dir, IntType lay, const void **weight,
    float *&cellWeightPeephole, float *&cellWeightProjection)
{
    cellWeightPeephole = weight[0x2] ?
        SafeCast<float *>(weight[0x2]) + (lay * rnn.nDir + dir) * 0x3 * rnn.dhc : nullptr;
    cellWeightProjection = weight[0x3] ?
        SafeCast<float *>(weight[0x3]) + (lay * rnn.nDir + dir) * rnn.weightsProjectionLd * rnn.dhc :
        nullptr;
}

// Initializes cell input/output pointers based on position flags
template <typename SrcType, typename BiasType>
static void InitCellPointers(const RnnCfg &rnn, IntType dir, IntType lay, IntType iter,
    CellPositionT cellPosition, const void *wsStates, void **dst, const void **src,
    SrcType *&cellSrcLay, SrcType *&cellSrcIter, SrcType *&cellDstLay, SrcType *&cellDstIter)
{
    cellSrcLay = SafeCast<SrcType *>(wsStates) +
        ((lay * rnn.nDir + dir) * (rnn.nIter + 1) + iter + 1) *
        rnn.wsStatesLayerLd * rnn.mb;
    cellSrcIter = SafeCast<SrcType *>(wsStates) +
        (((lay + 1) * rnn.nDir + dir) * (rnn.nIter + 1) + iter) *
        rnn.wsStatesLayerLd * rnn.mb;
    cellDstLay = SafeCast<SrcType *>(wsStates) +
        (((lay + 1) * rnn.nDir + dir) * (rnn.nIter + 1) + iter + 1) *
        rnn.wsStatesLayerLd * rnn.mb;
    cellDstIter = nullptr;

    if (cellPosition & CellPositionT::LAST_ITER) {
        cellDstIter = SafeCast<SrcType *>(dst[1]) +
            (lay * rnn.nDir + dir) * rnn.dstIterLd * rnn.mb;
        cellSrcLay = SafeCast<SrcType *>(dst[1]) +
            ((lay - 1) * rnn.nDir + dir) * rnn.dstIterLd * rnn.mb;
    }
    if (cellPosition & CellPositionT::LAST_LAYER) {
        SizeType tmpIter = rnn.execDir != ExecutionDirectionT::R2L ? iter : rnn.nIter - 1 - iter;
        SizeType tmpLastIter = rnn.execDir != ExecutionDirectionT::R2L ? tmpIter - 1 : tmpIter + 1;
        if (rnn.execDir != ExecutionDirectionT::BI_SUM && rnn.execDir != ExecutionDirectionT::BI_CONCAT) {
            cellDstLay = SafeCast<SrcType *>(dst[0]) + tmpIter * rnn.dstLayerCellLd;
            cellSrcIter = SafeCast<SrcType *>(dst[0]) + tmpLastIter * rnn.dstLayerCellLd;
        }
    }
    if (cellPosition & CellPositionT::FIRST_LAYER) {
        SizeType tmpIter = rnn.execDir != ExecutionDirectionT::R2L ? iter : rnn.nIter - 1 - iter;
        if (dir == 0) {
            cellSrcLay = SafeCast<SrcType *>(src[0]) +
                tmpIter * rnn.srcLayerCellLd;
        } else {
            cellSrcLay = SafeCast<SrcType *>(src[0]) +
                (rnn.nIter - 1 - tmpIter) * rnn.srcLayerCellLd;
        }
    }
    if (cellPosition & CellPositionT::FIRST_ITER) {
        cellSrcIter = SafeCast<SrcType *>(src[1]) + (lay * rnn.nDir + dir) * rnn.srcIterLd * rnn.mb;
    }
}

// Initializes cell state (C) pointers for LSTM/GRU cells
template <typename SrcType, typename BiasType>
static void InitCellCPointers(const RnnCfg &rnn, IntType dir, IntType lay, IntType iter,
    CellPositionT cellPosition, const void **src, void **dst, BiasType *wsStatesIterC,
    const void *&projectionHt, SrcType *wsHt, BiasType *&cellDstIterC, BiasType *&cellSrcIterC)
{
    if (rnn.isTraining) {
        // Training mode: use workspace for intermediate cell states
        cellDstIterC = wsStatesIterC +
            (((lay + 1) * rnn.nDir + dir) * (rnn.nIter + 1) + iter + 1) * rnn.wsDiffStatesIterCLd * rnn.mb;
        cellSrcIterC = wsStatesIterC +
            (((lay + 1) * rnn.nDir + dir) * (rnn.nIter + 1) + iter) * rnn.wsDiffStatesIterCLd * rnn.mb;
        if (cellPosition & CellPositionT::LAST_ITER) {
            cellDstIterC = SafeCast<BiasType *>(dst[0x2]) +
                (lay * rnn.nDir + dir) * rnn.dstIterCLd * rnn.mb;
        }
        if (cellPosition & CellPositionT::FIRST_ITER) {
            cellSrcIterC = SafeCast<BiasType *>(src[0x2]) +
                (lay * rnn.nDir + dir) * rnn.srcIterCLd * rnn.mb;
        }
        projectionHt = wsHt + ((lay * rnn.nDir + dir) * rnn.nIter + iter) * rnn.wsHtLd * rnn.wsHtNld;
    } else {
        // Inference mode: use output buffer directly (no workspace)
        cellDstIterC = SafeCast<BiasType *>(dst[0x2]) + (lay * rnn.nDir + dir) * rnn.dstIterCLd * rnn.mb;
        cellSrcIterC = cellDstIterC;
        if (cellPosition & CellPositionT::FIRST_ITER) {
            cellSrcIterC = SafeCast<BiasType *>(src[0x2]) +
                (lay * rnn.nDir + dir) * rnn.srcIterCLd * rnn.mb;
        }
    }
}

// Handles bidirectional RNN output merging (sum/concat)
template <typename SrcType>
static void HandleBiRnnMerge(const RnnCfg &rnn, void **dst, const void *wsStates)
{
    if (rnn.nDir - 1) {
        for (IntType iter = 0; iter < rnn.nIter; iter++) {
            const auto sum = [&rnn, dst, wsStates, iter](int i) {
                SizeType dir = 0;
                SrcType *srcPtr = SafeCast<SrcType *>(wsStates) +
                    ((rnn.nLayer * rnn.nDir + dir) * (rnn.nIter + 1) + iter + 1) * rnn.wsStatesLayerLd * rnn.mb;
                SrcType *dstPtr = SafeCast<SrcType *>(dst[0]) + iter * rnn.dstLayerCellLd;
                ArrayCopy(dstPtr + i * rnn.dstLayerLd, srcPtr + i * rnn.wsStatesLayerLd, rnn.dlc);
                dir = 1;
                srcPtr = SafeCast<SrcType *>(wsStates) + ((rnn.nLayer * rnn.nDir + dir) *
                    (rnn.nIter + 1) + (rnn.nIter - iter)) * rnn.wsStatesLayerLd * rnn.mb;
                if (rnn.execDir == ExecutionDirectionT::BI_CONCAT) {
                    ArrayCopy(dstPtr + i * rnn.dstLayerLd + dir * rnn.dlc, srcPtr + i * rnn.wsStatesLayerLd, rnn.dlc);
                } else {
                    ArrayAdd(dstPtr + i * rnn.dstLayerLd, srcPtr + i * rnn.wsStatesLayerLd, rnn.dlc);
                }
            };
            MultiThreader(rnn.mb, sum);
        }
    }
}

template <typename SrcType, typename BiasType, CellFuncType<SrcType, BiasType> CellFunc>
void RnnCommonRun(const void **src, const void **weight, void **dst, const void *bias, void *wsBase,
    const void *wsStates, const float *scratchGates, const void *augruAttentions, const void *projectionHt,
    const RnnCfg &rnn, const KuDNN::ActivateFunctionRNN myActf, const float *scratchCells, const RnnGemm *rnnGemm)
{
    // 1. Initialize workspace pointers (training/inference mode)
    SrcType *wsGates = nullptr;
    SrcType *wsGrids = nullptr;
    BiasType *wsStatesIterC = nullptr;
    SrcType *wsHt = nullptr;

    InitWsPointers<SrcType, BiasType>(rnn, wsBase, wsStates, wsGates, wsGrids, wsStatesIterC, wsHt);

    // 2. Core RNN execution loop (direction -> layer -> iteration)
    for (IntType dir = 0; dir < rnn.nDir; dir++) {
        for (IntType lay = 0; lay < rnn.nLayer; lay++) {
            BiasType *cellBias = SafeCast<BiasType *>(bias) + (lay * rnn.nDir + dir) *
                (rnn.nGates + rnn.isLbr) * rnn.dhc;

            // Initialize weight pointers (peephole/projection)
            float *cellWeightPeephole = nullptr;
            float *cellWeightProjection = nullptr;
            InitWeightPointers(rnn, dir, lay, weight, cellWeightPeephole, cellWeightProjection);

            // Process each sequence iteration/step
            for (IntType iter = 0; iter < rnn.nIter; iter++) {
                CellPositionT cellPosition = CalculateCellPosition(iter, lay, rnn.nIter, rnn.nLayer);

                // Initialize cell input/output pointers
                SrcType *cellSrcLay = nullptr;
                SrcType *cellSrcIter = nullptr;
                SrcType *cellDstLay = nullptr;
                SrcType *cellDstIter = nullptr;
                InitCellPointers<SrcType, BiasType>(rnn, dir, lay, iter, cellPosition, wsStates, dst, src,
                                                    cellSrcLay, cellSrcIter, cellDstLay, cellDstIter);

                // Initialize layer/iteration weight pointers
                SrcType *cellWeightLay = SafeCast<SrcType *>(weight[0]) +
                    (lay * rnn.nDir + dir) * rnn.weightsLayerLd * rnn.slc;
                SrcType *cellWeightIter = SafeCast<SrcType *>(weight[1]) +
                    (lay * rnn.nDir + dir) * rnn.weightsIterLd * rnn.sic;

                // Initialize cell state (C) pointers
                BiasType *cellDstIterC = nullptr;
                BiasType *cellSrcIterC = nullptr;
                InitCellCPointers<SrcType, BiasType>(rnn, dir, lay, iter, cellPosition, src, dst, wsStatesIterC,
                                                     projectionHt, wsHt, cellDstIterC, cellSrcIterC);

                // Temporary pointers for scratch buffers and attention (AUGRU)
                float *scratch = SafeCast<float *>(scratchGates);
                float *scratchCell = SafeCast<float *>(scratchCells);
                SrcType *augruAttenOff =  SafeCast<SrcType *>(augruAttentions) + iter * rnn.mb;

                // Execute cell operation (core RNN computation)
                CellFunc(cellPosition, rnn, cellSrcLay, cellSrcIter, cellDstLay, cellDstIter,
                    cellWeightLay, cellWeightIter, cellBias, scratch, cellSrcIterC, cellDstIterC,
                    cellWeightPeephole, cellWeightProjection, SafeCast<SrcType *>(projectionHt),
                    myActf, scratchCell, augruAttenOff,
                    wsGates + ((lay * rnn.nDir + dir) * rnn.nIter + iter) * rnn.wsGatesLd * rnn.mb,
                    wsGrids + (((lay * rnn.nDir) + dir) * rnn.nIter + iter) * rnn.wsPerCell, rnnGemm);
            }
        }
    }

    // 3. Merge bidirectional outputs (if applicable)
    HandleBiRnnMerge<SrcType>(rnn, dst, wsStates);
}

const std::unordered_map<RnnKerParas, RnnKerFwdFunc, RnnKerParasHash, RnnKerParasEqual> RNN_KER_MAP = {
    // F32
    {{Element::TypeT::F32, Element::TypeT::F32, RnnAlgorithm::LSTM},
        RnnCommonRun<float, float, LstmCell<float, float>>},
    {{Element::TypeT::F32, Element::TypeT::F32, RnnAlgorithm::RNN},
        RnnCommonRun<float, float, RnnCell<float, float>>},
    {{Element::TypeT::F32, Element::TypeT::F32, RnnAlgorithm::GRU},
        RnnCommonRun<float, float, GruCell<float, float>>},
    {{Element::TypeT::F32, Element::TypeT::F32, RnnAlgorithm::AUGRU},
        RnnCommonRun<float, float, AugruCell<float, float>>},
    {{Element::TypeT::F32, Element::TypeT::F32, RnnAlgorithm::LBR_GRU},
        RnnCommonRun<float, float, LbrGruCell<float, float>>},
    {{Element::TypeT::F32, Element::TypeT::F32, RnnAlgorithm::LBR_AUGRU},
        RnnCommonRun<float, float, LbrAugruCell<float, float>>},
    // BF16
    {{Element::TypeT::BF16, Element::TypeT::BF16, RnnAlgorithm::LSTM},
        RnnCommonRun<__bf16, __bf16, LstmCell<__bf16, __bf16>>},
    {{Element::TypeT::BF16, Element::TypeT::BF16, RnnAlgorithm::RNN},
        RnnCommonRun<__bf16, __bf16, RnnCell<__bf16, __bf16>>},
    {{Element::TypeT::BF16, Element::TypeT::BF16, RnnAlgorithm::GRU},
        RnnCommonRun<__bf16, __bf16, GruCell<__bf16, __bf16>>},
    {{Element::TypeT::BF16, Element::TypeT::BF16, RnnAlgorithm::AUGRU},
        RnnCommonRun<__bf16, __bf16, AugruCell<__bf16, __bf16>>},
    {{Element::TypeT::BF16, Element::TypeT::BF16, RnnAlgorithm::LBR_GRU},
        RnnCommonRun<__bf16, __bf16, LbrGruCell<__bf16, __bf16>>},
    {{Element::TypeT::BF16, Element::TypeT::BF16, RnnAlgorithm::LBR_AUGRU},
        RnnCommonRun<__bf16, __bf16, LbrAugruCell<__bf16, __bf16>>},
    // BF16F32
    {{Element::TypeT::BF16, Element::TypeT::F32, RnnAlgorithm::LSTM},
        RnnCommonRun<__bf16, float, LstmCell<__bf16, float>>},
    {{Element::TypeT::BF16, Element::TypeT::F32, RnnAlgorithm::RNN},
        RnnCommonRun<__bf16, float, RnnCell<__bf16, float>>},
    {{Element::TypeT::BF16, Element::TypeT::F32, RnnAlgorithm::GRU},
        RnnCommonRun<__bf16, float, GruCell<__bf16, float>>},
    {{Element::TypeT::BF16, Element::TypeT::F32, RnnAlgorithm::AUGRU},
        RnnCommonRun<__bf16, float, AugruCell<__bf16, float>>},
    {{Element::TypeT::BF16, Element::TypeT::F32, RnnAlgorithm::LBR_GRU},
        RnnCommonRun<__bf16, float, LbrGruCell<__bf16, float>>},
    {{Element::TypeT::BF16, Element::TypeT::F32, RnnAlgorithm::LBR_AUGRU},
        RnnCommonRun<__bf16, float, LbrAugruCell<__bf16, float>>},
};

} // Rnn
} // Detail
} // KuDNN
