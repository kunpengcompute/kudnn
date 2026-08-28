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

#ifndef KUDNN_RNN_GEMM_HPP
#define KUDNN_RNN_GEMM_HPP

#include "internal_funcs.hpp"
#include "operations/kudnn_rnn.hpp"
#include "operations/kudnn_gemm.hpp"

namespace KuDNN {
namespace Detail {
namespace Rnn {

class RnnGemm {
public:
    using SizeType = ::KuDNN::SizeType;
    RnnGemm(Element::TypeT srcType, Element::TypeT biasType, RnnCfg &rnn) noexcept(false)
    {
        Service::Unused(biasType);
        SizeType m = rnn.nGates * rnn.dhc;
        SizeType n = rnn.mb;
        SizeType k = rnn.slc;
        layGemmFirstLayer.reset(new Gemm{
            {{m, k}, srcType, Layout::COL_MAJOR, {1, rnn.weightsLayerLd}},
            {{k, n}, srcType, Layout::COL_MAJOR, {1, rnn.SrcLayerLd(CellPositionT::FIRST_LAYER)}},
            {{m, n}, Element::TypeT::F32, Layout::COL_MAJOR, {1, rnn.scratchGatesLd}}});
        if (rnn.nLayer != 1) {
            layGemmLastIter.reset(new Gemm{
                {{m, k}, srcType, Layout::COL_MAJOR, {1, rnn.weightsLayerLd}},
                {{k, n}, srcType, Layout::COL_MAJOR, {1, rnn.SrcLayerLd(CellPositionT::LAST_ITER)}},
                {{m, n}, Element::TypeT::F32, Layout::COL_MAJOR, {1, rnn.scratchGatesLd}}});
        }
        if (rnn.nLayer != 1 && rnn.nIter != 1) {
            layGemm.reset(new Gemm{
                {{m, k}, srcType, Layout::COL_MAJOR, {1, rnn.weightsLayerLd}},
                {{k, n}, srcType, Layout::COL_MAJOR, {1, rnn.SrcLayerLd(CellPositionT::MIDDLE_CELL)}},
                {{m, n}, Element::TypeT::F32, Layout::COL_MAJOR, {1, rnn.scratchGatesLd}}});
        }
        k = rnn.sic;
        iterGemmFirstIter.reset(new Gemm{
            {{m, k}, srcType, Layout::COL_MAJOR, {1, rnn.weightsIterLd}},
            {{k, n}, srcType, Layout::COL_MAJOR, {1, rnn.SrcIterLd(CellPositionT::FIRST_ITER)}},
            {{m, n}, Element::TypeT::F32, Layout::COL_MAJOR, {1, rnn.scratchGatesLd}}});
        if (rnn.nIter != 1) {
            iterGemmLastLayer.reset(new Gemm{
                {{m, k}, srcType, Layout::COL_MAJOR, {1, rnn.weightsIterLd}},
                {{k, n}, srcType, Layout::COL_MAJOR, {1, rnn.SrcIterLd(CellPositionT::LAST_LAYER)}},
                {{m, n}, Element::TypeT::F32, Layout::COL_MAJOR, {1, rnn.scratchGatesLd}}});
        }
        if (1 != rnn.nLayer && 1 != rnn.nIter) {
            iterGemm.reset(new Gemm{
                {{m, k}, srcType, Layout::COL_MAJOR, {1, rnn.weightsIterLd}},
                {{k, n}, srcType, Layout::COL_MAJOR, {1, rnn.SrcIterLd(CellPositionT::MIDDLE_CELL)}},
                {{m, n}, Element::TypeT::F32, Layout::COL_MAJOR, {1, rnn.scratchGatesLd}}});
        }
    }

    RnnGemm(const RnnGemm &other) noexcept(false) : layGemmFirstLayer(new Gemm(*other.layGemmFirstLayer)),
        layGemmLastIter(new Gemm(*other.layGemmLastIter)), layGemm(new Gemm(*other.layGemm)),
        iterGemmFirstIter(new Gemm(*other.iterGemmFirstIter)), iterGemmLastLayer(new Gemm(*other.iterGemmLastLayer)),
        iterGemm(new Gemm(*other.iterGemm)) {}

    void ExecLayerGemm(CellPositionT cellPosition, void *wei, void *src, void *scratch, float alpha, float beta) const
    {
        if (cellPosition & CellPositionT::FIRST_LAYER) {
            layGemmFirstLayer->Run(wei, src, scratch, alpha, beta);
        } else if (cellPosition & CellPositionT::LAST_ITER) {
            layGemmLastIter->Run(wei, src, scratch, alpha, beta);
        } else {
            layGemm->Run(wei, src, scratch, alpha, beta);
        }
    }

    void ExecIterGemm(CellPositionT cellPosition, void *wei, void *src, void *scratch, float alpha, float beta) const
    {
        if (cellPosition & CellPositionT::FIRST_ITER) {
            iterGemmFirstIter->Run(wei, src, scratch, alpha, beta);
        } else if (cellPosition & CellPositionT::LAST_LAYER) {
            iterGemmLastLayer->Run(wei, src, scratch, alpha, beta);
        } else {
            iterGemm->Run(wei, src, scratch, alpha, beta);
        }
    }

    virtual void ExecLstmProjectionGemm(CellPositionT cellPosition, void *wei, void *src, void *scratch,
        float alpha, float beta) const
    {
        Service::Unused(cellPosition, wei, src, scratch, alpha, beta);
        printf("ExecLstmProjectionGemm is not implemented!\n");
    }

    virtual void ExecGruGemm(CellPositionT cellPosition, void *wei, void *src, void *scratch,
        float alpha, float beta) const
    {
        Service::Unused(cellPosition, wei, src, scratch, alpha, beta);
        printf("ExecGruGemm is not implemented!\n");
    }

    RnnGemm &operator=(const RnnGemm &) = delete;

    virtual ~RnnGemm() = default;

protected:
    std::unique_ptr<Gemm> layGemmFirstLayer;
    std::unique_ptr<Gemm> layGemmLastIter;
    std::unique_ptr<Gemm> layGemm;
    std::unique_ptr<Gemm> iterGemmFirstIter;
    std::unique_ptr<Gemm> iterGemmLastLayer;
    std::unique_ptr<Gemm> iterGemm;
};

class LstmGemm : public RnnGemm {
public:
    LstmGemm(Element::TypeT srcType, Element::TypeT biasType, RnnCfg &rnn) : RnnGemm(srcType, biasType, rnn)
    {
        Service::Unused(biasType);
        SizeType m = rnn.dic;
        SizeType n = rnn.mb;
        SizeType k = rnn.dhc;
        if (rnn.weightsProjectionLd == 0) {
            return ;
        }
        projectionGemmLastLayer.reset(new Gemm{
            {{m, k}, srcType, Layout::COL_MAJOR, {1, rnn.weightsProjectionLd}},
            {{k, n}, srcType, Layout::COL_MAJOR, {1, rnn.projHtLd}},
            {{m, n}, Element::TypeT::F32, Layout::COL_MAJOR, {1, rnn.DstLayerLd(CellPositionT::LAST_LAYER)}}});
        if (rnn.nLayer != 1) {
            projectionGemmLastIter.reset(new Gemm{
                {{m, k}, srcType, Layout::COL_MAJOR, {1, rnn.weightsProjectionLd}},
                {{k, n}, srcType, Layout::COL_MAJOR, {1, rnn.projHtLd}},
                {{m, n}, Element::TypeT::F32, Layout::COL_MAJOR, {1, rnn.DstLayerLd(CellPositionT::LAST_ITER)}}});
        }
        if (rnn.nLayer != 1 && rnn.nIter != 1) {
            projectionGemm.reset(new Gemm{
                {{m, k}, srcType, Layout::COL_MAJOR, {1, rnn.weightsProjectionLd}},
                {{k, n}, srcType, Layout::COL_MAJOR, {1, rnn.projHtLd}},
                {{m, n}, Element::TypeT::F32, Layout::COL_MAJOR, {1, rnn.DstLayerLd(CellPositionT::MIDDLE_CELL)}}});
        }
    }

    void ExecLstmProjectionGemm(CellPositionT cellPosition, void *wei, void *src, void *scratch,
        float alpha, float beta) const override
    {
        if (cellPosition & CellPositionT::LAST_LAYER) {
            projectionGemmLastLayer->Run(wei, src, scratch, alpha, beta);
        } else if (cellPosition & CellPositionT::LAST_ITER) {
            projectionGemmLastIter->Run(wei, src, scratch, alpha, beta);
        } else {
            projectionGemm->Run(wei, src, scratch, alpha, beta);
        }
    }

private:
    std::unique_ptr<Gemm> projectionGemmLastLayer;
    std::unique_ptr<Gemm> projectionGemmLastIter;
    std::unique_ptr<Gemm> projectionGemm;
};

class GruGemm : public RnnGemm {
public:
    GruGemm(Element::TypeT srcType, Element::TypeT biasType, RnnCfg &rnn) : RnnGemm(srcType, biasType, rnn)
    {
        Service::Unused(biasType);
        SizeType m = (rnn.nGates - 1) * rnn.dhc;
        SizeType n = rnn.mb;
        SizeType k = rnn.sic;
        iterGemmFirstIter.reset(new Gemm{
            {{m, k}, srcType, Layout::COL_MAJOR, {1, rnn.weightsIterLd}},
            {{k, n}, srcType, Layout::COL_MAJOR, {1, rnn.SrcIterLd(CellPositionT::FIRST_ITER)}},
            {{m, n}, Element::TypeT::F32, Layout::COL_MAJOR, {1, rnn.scratchGatesLd}}});
        if (1 != rnn.nIter) {
            iterGemmLastLayer.reset(new Gemm{
                {{m, k}, srcType, Layout::COL_MAJOR, {1, rnn.weightsIterLd}},
                {{k, n}, srcType, Layout::COL_MAJOR, {1, rnn.SrcIterLd(CellPositionT::LAST_LAYER)}},
                {{m, n}, Element::TypeT::F32, Layout::COL_MAJOR, {1, rnn.scratchGatesLd}}});
        }
        if (rnn.nLayer != 1 && rnn.nIter != 1) {
            iterGemm.reset(new Gemm{
                {{m, k}, srcType, Layout::COL_MAJOR, {1, rnn.weightsIterLd}},
                {{k, n}, srcType, Layout::COL_MAJOR, {1, rnn.SrcIterLd(CellPositionT::MIDDLE_CELL)}},
                {{m, n}, Element::TypeT::F32, Layout::COL_MAJOR, {1, rnn.scratchGatesLd}}});
        }
        m = rnn.dhc;
        gruGemmLastLayer.reset(new Gemm{
            {{m, k}, srcType, Layout::COL_MAJOR, {1, rnn.weightsIterLd}},
            {{k, n}, srcType, Layout::COL_MAJOR, {1, rnn.DstIterPart2Ld(CellPositionT::LAST_LAYER)}},
            {{m, n}, Element::TypeT::F32, Layout::COL_MAJOR, {1, rnn.scratchGatesLd}}});
        if (rnn.nLayer != 1) {
            gruGemmLastIter.reset(new Gemm{
                {{m, k}, srcType, Layout::COL_MAJOR, {1, rnn.weightsIterLd}},
                {{k, n}, srcType, Layout::COL_MAJOR, {1, rnn.DstIterPart2Ld(CellPositionT::LAST_ITER)}},
                {{m, n}, Element::TypeT::F32, Layout::COL_MAJOR, {1, rnn.scratchGatesLd}}});
        }
        if (rnn.nLayer != 1 && rnn.nIter != 1) {
            gruGemm.reset(new Gemm{
                {{m, k}, srcType, Layout::COL_MAJOR, {1, rnn.weightsIterLd}},
                {{k, n}, srcType, Layout::COL_MAJOR, {1, rnn.DstIterPart2Ld(CellPositionT::MIDDLE_CELL)}},
                {{m, n}, Element::TypeT::F32, Layout::COL_MAJOR, {1, rnn.scratchGatesLd}}});
        }
    }

    void ExecGruGemm(CellPositionT cellPosition, void *wei, void *src, void *scratch,
        float alpha, float beta) const override
    {
        if (cellPosition & CellPositionT::LAST_LAYER) {
            gruGemmLastLayer->Run(wei, src, scratch, alpha, beta);
        } else if (cellPosition & CellPositionT::LAST_ITER) {
            gruGemmLastIter->Run(wei, src, scratch, alpha, beta);
        } else {
            gruGemm->Run(wei, src, scratch, alpha, beta);
        }
    }

private:
    std::unique_ptr<Gemm> gruGemmLastLayer;
    std::unique_ptr<Gemm> gruGemmLastIter;
    std::unique_ptr<Gemm> gruGemm;
};

class LbrGemm : public RnnGemm {
public:
    LbrGemm(Element::TypeT srcType, Element::TypeT biasType, RnnCfg &rnn) : RnnGemm(srcType, biasType, rnn)
    {
        Service::Unused(biasType);
        SizeType m = rnn.nGates * rnn.dhc;
        SizeType n = rnn.mb;
        SizeType k = rnn.sic;
        iterGemmFirstIter.reset(new Gemm{
            {{m, k}, srcType, Layout::COL_MAJOR, {1, rnn.weightsIterLd}},
            {{k, n}, srcType, Layout::COL_MAJOR, {1, rnn.SrcIterLd(CellPositionT::FIRST_ITER)}},
            {{m, n}, Element::TypeT::F32, Layout::COL_MAJOR, {1, rnn.wsGatesLd}}});
        if (rnn.nIter != 1) {
            iterGemmLastLayer.reset(new Gemm{
                {{m, k}, srcType, Layout::COL_MAJOR, {1, rnn.weightsIterLd}},
                {{k, n}, srcType, Layout::COL_MAJOR, {1, rnn.SrcIterLd(CellPositionT::LAST_LAYER)}},
                {{m, n}, Element::TypeT::F32, Layout::COL_MAJOR, {1, rnn.wsGatesLd}}});
        }
        if (rnn.nLayer != 1 && rnn.nIter != 1) {
            iterGemm.reset(new Gemm{
                {{m, k}, srcType, Layout::COL_MAJOR, {1, rnn.weightsIterLd}},
                {{k, n}, srcType, Layout::COL_MAJOR, {1, rnn.SrcIterLd(CellPositionT::MIDDLE_CELL)}},
                {{m, n}, Element::TypeT::F32, Layout::COL_MAJOR, {1, rnn.wsGatesLd}}});
        }
    }
};

class RnnBwdGemm {
public:
    using SizeType = ::KuDNN::SizeType;
    RnnBwdGemm(Element::TypeT srcType, Element::TypeT biasType, RnnCfg &rnn) noexcept(false)
    {
        Service::Unused(biasType);
        SizeType m = rnn.sic;
        SizeType n = rnn.mb;
        SizeType k = rnn.nGates * rnn.dhc;
        gemmWeightsIter.reset(new Gemm{
            {{m, k}, srcType, Layout::COL_MAJOR, {1, rnn.weightsIterLd}},
            {{k, n}, srcType, Layout::COL_MAJOR, {1, rnn.scratchGatesLd}},
            {{m, n}, Element::TypeT::F32, Layout::COL_MAJOR, {1, rnn.wsDiffStatesIterLd}}});

        m = rnn.nGates * rnn.dhc;
        n = rnn.slc;
        k = rnn.mb;
        gemmScratchFirstLay.reset(new Gemm{
            {{m, k}, srcType, Layout::COL_MAJOR, {1, rnn.scratchGatesLd}},
            {{k, n}, srcType, Layout::ROW_MAJOR, {rnn.SrcLayerLd(CellPositionT::FIRST_LAYER), 1}},
            {{m, n}, Element::TypeT::F32, Layout::COL_MAJOR, {1, rnn.diffWeightsLayerLd}}});
        if (rnn.nLayer != 1) {
            gemmScratchLastIter.reset(new Gemm{
                {{m, k}, srcType, Layout::COL_MAJOR, {1, rnn.scratchGatesLd}},
                {{k, n}, srcType, Layout::ROW_MAJOR, {rnn.SrcLayerLd(CellPositionT::LAST_ITER), 1}},
                {{m, n}, Element::TypeT::F32, Layout::COL_MAJOR, {1, rnn.diffWeightsLayerLd}}});
        }
        gemmScratchLay.reset(new Gemm{
            {{m, k}, srcType, Layout::COL_MAJOR, {1, rnn.scratchGatesLd}},
            {{k, n}, srcType, Layout::ROW_MAJOR, {rnn.SrcLayerLd(CellPositionT::MIDDLE_CELL), 1}},
            {{m, n}, Element::TypeT::F32, Layout::COL_MAJOR, {1, rnn.diffWeightsLayerLd}}});

        m = rnn.slc;
        n = rnn.mb;
        k = rnn.nGates * rnn.dhc;
        gemmWeightsLay.reset(new Gemm{
            {{m, k}, srcType, Layout::COL_MAJOR, {1, rnn.weightsLayerLd}},
            {{k, n}, srcType, Layout::COL_MAJOR, {1, rnn.scratchGatesLd}},
            {{m, n}, Element::TypeT::F32, Layout::COL_MAJOR, {1, rnn.wsDiffStatesLayerLd}}});

        m = rnn.nGates * rnn.dhc;
        n = rnn.sic;
        k = rnn.mb;
        gemmScratchFirstIter.reset(new Gemm{
            {{m, k}, srcType, Layout::COL_MAJOR, {1, rnn.scratchGatesLd}},
            {{k, n}, srcType, Layout::ROW_MAJOR, {rnn.SrcIterLd(CellPositionT::FIRST_ITER), 1}},
            {{m, n}, Element::TypeT::F32, Layout::COL_MAJOR, {1, rnn.diffWeightsIterLd}}});
        if (1 != rnn.nIter) {
            gemmScratchLastLay.reset(new Gemm{
                {{m, k}, srcType, Layout::COL_MAJOR, {1, rnn.scratchGatesLd}},
                {{k, n}, srcType, Layout::ROW_MAJOR, {rnn.SrcIterLd(CellPositionT::LAST_LAYER), 1}},
                {{m, n}, Element::TypeT::F32, Layout::COL_MAJOR, {1, rnn.diffWeightsIterLd}}});
        }
        gemmScratchIter.reset(new Gemm{
            {{m, k}, srcType, Layout::COL_MAJOR, {1, rnn.scratchGatesLd}},
            {{k, n}, srcType, Layout::ROW_MAJOR, {rnn.SrcIterLd(CellPositionT::MIDDLE_CELL), 1}},
            {{m, n}, Element::TypeT::F32, Layout::COL_MAJOR, {1, rnn.diffWeightsIterLd}}});
    }

    RnnBwdGemm(const RnnBwdGemm &other) noexcept(false)
        : gemmWeightsIter(new Gemm(*other.gemmWeightsIter)), gemmScratchFirstLay(new Gemm(*other.gemmScratchFirstLay)),
        gemmScratchLastIter(new Gemm(*other.gemmScratchLastIter)), gemmScratchLay(new Gemm(*other.gemmScratchLay)),
        gemmWeightsLay(new Gemm(*other.gemmWeightsLay)), gemmScratchFirstIter(new Gemm(*other.gemmScratchFirstIter)),
        gemmScratchLastLay(new Gemm(*other.gemmScratchLastLay)), gemmScratchIter(new Gemm(*other.gemmScratchIter)) {}

    void ExecWeightsIterGemm(void *cellWeightIter, void *scratch, void *cellDiffSrcIter,
        float alpha, float beta) const
    {
        gemmWeightsIter->Run(cellWeightIter, scratch, cellDiffSrcIter, alpha, beta);
    }

    void ExecScratchLayGemm(CellPositionT cellPosition, void *scratch, void *cellSrcLay, void *cellDiffWeightLay,
        float alpha, float beta) const
    {
        if (cellPosition & CellPositionT::FIRST_LAYER) {
            gemmScratchFirstLay->Run(scratch, cellSrcLay, cellDiffWeightLay, alpha, beta);
        } else if (cellPosition & CellPositionT::LAST_ITER) {
            gemmScratchLastIter->Run(scratch, cellSrcLay, cellDiffWeightLay, alpha, beta);
        } else {
            gemmScratchLay->Run(scratch, cellSrcLay, cellDiffWeightLay, alpha, beta);
        }
    }

    void ExecWeightsLayGemm(void *cellWeightLay, void *scratch, void *cellDiffSrcLay,
        float alpha, float beta) const
    {
        gemmWeightsLay->Run(cellWeightLay, scratch, cellDiffSrcLay, alpha, beta);
    }

    void ExecScratchIterGemm(CellPositionT cellPosition, void *scratch, void *cellSrcIter, void *cellDiffWeightIter,
        float alpha, float beta) const
    {
        if (cellPosition & CellPositionT::FIRST_ITER) {
            gemmScratchFirstIter->Run(scratch, cellSrcIter, cellDiffWeightIter, alpha, beta);
        } else if (cellPosition & CellPositionT::LAST_LAYER) {
            gemmScratchLastLay->Run(scratch, cellSrcIter, cellDiffWeightIter, alpha, beta);
        } else {
            gemmScratchIter->Run(scratch, cellSrcIter, cellDiffWeightIter, alpha, beta);
        }
    }

    virtual void ExecLstmScratchDiffGemm(void *scratchDiffHt, void *projectionHt, void *cellDiffWeightProjection,
        float alpha, float beta) const
    {
        Service::Unused(scratchDiffHt, projectionHt, cellDiffWeightProjection, alpha, beta);
        printf("ExecLstmScratchDiffGemm is not implemented!\n");
    }

    virtual void ExecLstmProjectionWeightGemm(void *cellweightProjection, void *scratchDiffHt, void *cellDiffDstLay,
        float alpha, float beta) const
    {
        Service::Unused(cellweightProjection, scratchDiffHt, cellDiffDstLay, alpha, beta);
        printf("ExecLstmProjectionWeightGemm is not implemented!\n");
    }

    virtual void ExecGruScratchGemm(void *scratch, void *scratchCell, void *cellDiffWeightIter,
        float alpha, float beta) const
    {
        Service::Unused(scratch, scratchCell, cellDiffWeightIter, alpha, beta);
        printf("ExecGruScratchGemm is not implemented!\n");
    }

    virtual void ExecWeightsIterGemm2(void *cellWeightIter, void *scratch, void *cellDiffSrcIter,
        float alpha, float beta) const
    {
        Service::Unused(cellWeightIter, scratch, cellDiffSrcIter, alpha, beta);
        printf("ExecWeightsIterGemm2 is not implemented!\n");
    }

    RnnBwdGemm &operator=(const RnnBwdGemm &) = delete;

    virtual ~RnnBwdGemm() = default;

protected:
    std::unique_ptr<Gemm> gemmWeightsIter;
    std::unique_ptr<Gemm> gemmScratchFirstLay;
    std::unique_ptr<Gemm> gemmScratchLastIter;
    std::unique_ptr<Gemm> gemmScratchLay;
    std::unique_ptr<Gemm> gemmWeightsLay;
    std::unique_ptr<Gemm> gemmScratchFirstIter;
    std::unique_ptr<Gemm> gemmScratchLastLay;
    std::unique_ptr<Gemm> gemmScratchIter;
};

class LstmBwdGemm : public RnnBwdGemm {
public:
    LstmBwdGemm(Element::TypeT srcType, Element::TypeT biasType, RnnCfg &rnn) : RnnBwdGemm(srcType, biasType, rnn)
    {
        Service::Unused(biasType);
        if (rnn.weightsProjectionLd == 0) {
            return ;
        }

        SizeType m = rnn.dlc;
        SizeType n = rnn.dhc;
        SizeType k = rnn.mb;
        gemmProjectionScratchDiff.reset(new Gemm{
            {{m, k}, srcType, Layout::COL_MAJOR, {1, rnn.scratchDiffHtLd}},
            {{k, n}, srcType, Layout::ROW_MAJOR, {rnn.wsHtLd, 1}},
            {{m, n}, Element::TypeT::F32, Layout::COL_MAJOR, {1, rnn.diffWeightsProjectionLd}}});

        m = rnn.dhc;
        n = rnn.mb;
        k = rnn.dic;
        gemmProjectionWeight.reset(new Gemm{
            {{m, k}, srcType, Layout::COL_MAJOR, {1, rnn.weightsProjectionLd}},
            {{k, n}, srcType, Layout::COL_MAJOR, {1, rnn.scratchDiffHtLd}},
            {{m, n}, Element::TypeT::F32, Layout::COL_MAJOR, {1, rnn.wsDiffStatesLayerLd}}});
    }

    void ExecLstmScratchDiffGemm(void *scratchDiffHt, void *projectionHt, void *cellDiffWeightProjection,
        float alpha, float beta) const override
    {
        gemmProjectionScratchDiff->Run(scratchDiffHt, projectionHt, cellDiffWeightProjection, alpha, beta);
    }

    void ExecLstmProjectionWeightGemm(void *cellweightProjection, void *scratchDiffHt, void *cellDiffDstLay,
        float alpha, float beta) const override
    {
        gemmProjectionWeight->Run(cellweightProjection, scratchDiffHt, cellDiffDstLay, alpha, beta);
    }

private:
    std::unique_ptr<Gemm> gemmProjectionScratchDiff;
    std::unique_ptr<Gemm> gemmProjectionWeight;
};

class GruBwdGemm : public RnnBwdGemm {
public:
    GruBwdGemm(Element::TypeT srcType, Element::TypeT biasType, RnnCfg &rnn) : RnnBwdGemm(srcType, biasType, rnn)
    {
        Service::Unused(biasType);

        SizeType m = rnn.sic;
        SizeType n = rnn.mb;
        SizeType k = rnn.dhc;
        gemmWeightsIter.reset(new Gemm{
            {{m, k}, srcType, Layout::COL_MAJOR, {1, rnn.weightsIterLd}},
            {{k, n}, srcType, Layout::COL_MAJOR, {1, rnn.scratchGatesLd}},
            {{m, n}, Element::TypeT::F32, Layout::COL_MAJOR, {1, rnn.wsDiffStatesIterLd}}});

        m = (rnn.nGates - 1) * rnn.dhc;
        k = rnn.mb;
        n = rnn.sic;
        gemmScratchFirstIter.reset(new Gemm{
            {{m, k}, srcType, Layout::COL_MAJOR, {1, rnn.scratchGatesLd}},
            {{k, n}, srcType, Layout::ROW_MAJOR, {rnn.SrcIterLd(CellPositionT::FIRST_ITER), 1}},
            {{m, n}, Element::TypeT::F32, Layout::COL_MAJOR, {1, rnn.diffWeightsIterLd}}});
        if (1 != rnn.nIter) {
            gemmScratchLastLay.reset(new Gemm{
                {{m, k}, srcType, Layout::COL_MAJOR, {1, rnn.scratchGatesLd}},
                {{k, n}, srcType, Layout::ROW_MAJOR, {rnn.SrcIterLd(CellPositionT::LAST_LAYER), 1}},
                {{m, n}, Element::TypeT::F32, Layout::COL_MAJOR, {1, rnn.diffWeightsIterLd}}});
        }
        gemmScratchIter.reset(new Gemm{
            {{m, k}, srcType, Layout::COL_MAJOR, {1, rnn.scratchGatesLd}},
            {{k, n}, srcType, Layout::ROW_MAJOR, {rnn.SrcIterLd(CellPositionT::MIDDLE_CELL), 1}},
            {{m, n}, Element::TypeT::F32, Layout::COL_MAJOR, {1, rnn.diffWeightsIterLd}}});

        m = rnn.dhc;
        n = rnn.sic;
        k = rnn.mb;
        gemmGruScratch.reset(new Gemm{
            {{m, k}, srcType, Layout::COL_MAJOR, {1, rnn.scratchGatesLd}},
            {{k, n}, srcType, Layout::ROW_MAJOR, {rnn.wsStatesLayerLd, 1}},
            {{m, n}, Element::TypeT::F32, Layout::COL_MAJOR, {1, rnn.diffWeightsIterLd}}});

        m = rnn.sic;
        n = rnn.mb;
        k = (rnn.nGates - 1) * rnn.dhc;
        gemmWeightsIter2.reset(new Gemm{
            {{m, k}, srcType, Layout::COL_MAJOR, {1, rnn.weightsIterLd}},
            {{k, n}, srcType, Layout::COL_MAJOR, {1, rnn.scratchGatesLd}},
            {{m, n}, Element::TypeT::F32, Layout::COL_MAJOR, {1, rnn.wsDiffStatesIterLd}}});
    }

    void ExecGruScratchGemm(void *scratch, void *scratchCell, void *cellDiffWeightIter,
        float alpha, float beta) const override
    {
        gemmGruScratch->Run(scratch, scratchCell, cellDiffWeightIter, alpha, beta);
    }

    void ExecWeightsIterGemm2(void *cellWeightIter, void *scratch, void *cellDiffSrcIter,
        float alpha, float beta) const override
    {
        gemmWeightsIter2->Run(cellWeightIter, scratch, cellDiffSrcIter, alpha, beta);
    }
private:
    std::unique_ptr<Gemm> gemmGruScratch;
    std::unique_ptr<Gemm> gemmWeightsIter2;
};

class LbrGruBwdGemm : public GruBwdGemm {
public:
    LbrGruBwdGemm(Element::TypeT srcType, Element::TypeT biasType, RnnCfg &rnn) : GruBwdGemm(srcType, biasType, rnn)
    {
        Service::Unused(biasType);

        SizeType m = rnn.sic;
        SizeType n = rnn.mb;
        SizeType k = rnn.nGates * rnn.dhc;
        gemmWeightsIter.reset(new Gemm{
            {{m, k}, srcType, Layout::COL_MAJOR, {1, rnn.weightsIterLd}},
            {{k, n}, srcType, Layout::COL_MAJOR, {1, rnn.wsGatesLd}},
            {{m, n}, Element::TypeT::F32, Layout::COL_MAJOR, {1, rnn.wsDiffStatesIterLd}}});

        m = rnn.nGates * rnn.dhc;
        n = rnn.sic;
        k = rnn.mb;
        gemmScratchFirstIter.reset(new Gemm{
            {{m, k}, srcType, Layout::COL_MAJOR, {1, rnn.wsGatesLd}},
            {{k, n}, srcType, Layout::ROW_MAJOR, {rnn.SrcIterLd(CellPositionT::FIRST_ITER), 1}},
            {{m, n}, Element::TypeT::F32, Layout::COL_MAJOR, {1, rnn.diffWeightsIterLd}}});
        if (rnn.nIter != 1) {
            gemmScratchLastLay.reset(new Gemm{
                {{m, k}, srcType, Layout::COL_MAJOR, {1, rnn.wsGatesLd}},
                {{k, n}, srcType, Layout::ROW_MAJOR, {rnn.SrcIterLd(CellPositionT::LAST_LAYER), 1}},
                {{m, n}, Element::TypeT::F32, Layout::COL_MAJOR, {1, rnn.diffWeightsIterLd}}});
        }
        gemmScratchIter.reset(new Gemm{
            {{m, k}, srcType, Layout::COL_MAJOR, {1, rnn.wsGatesLd}},
            {{k, n}, srcType, Layout::ROW_MAJOR, {rnn.SrcIterLd(CellPositionT::MIDDLE_CELL), 1}},
            {{m, n}, Element::TypeT::F32, Layout::COL_MAJOR, {1, rnn.diffWeightsIterLd}}});
    }
};

} // Rnn
} // Detail
} // KuDNN

#endif // KUDNN_RNN_GEMM_HPP
