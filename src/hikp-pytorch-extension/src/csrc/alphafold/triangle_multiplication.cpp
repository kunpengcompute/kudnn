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

#include <iostream>
#include <ATen/ops/empty.h>
#include <ATen/record_function.h>
#include "cblas.h"
#include "vla.h"
#include "fused_ops.h"
#include "triangle_multiplication.h"

namespace alphafold {

enum {
    DIM0 = 0,
    DIM1 = 1,
    DIM2 = 2,
    DIM3 = 3,
    DIM4 = 4
};

static constexpr int64_t DEFAULT_BLOCK_SIZE = 64;
static constexpr float LAYER_NORM_EPS = 1e-4;

static at::Tensor CalculateProjWithBlock(at::Tensor &input, at::Tensor &mask, at::Tensor &weight,
    at::Tensor &bias, at::Tensor &gateWeight, at::Tensor &gateBias, int equation, int64_t blockSize)
{
    int64_t length = input.size(DIM0);
    int64_t nChannels = input.size(DIM2);
    int64_t intermediateChannels = weight.size(DIM0);
    int64_t blockDim = length / blockSize;

    auto proj = input.new_empty({intermediateChannels, blockDim, blockDim, blockSize, blockSize});
    auto projPtr = GetVLAPtr<float>(proj, {blockDim, blockDim, blockSize, blockSize});
    auto weightPtr = GetVLAPtr<float>(weight, {nChannels});
    auto biasPtr = GetVLAPtr<float>(bias);
    auto gateWeightPtr = GetVLAPtr<float>(gateWeight, {nChannels});
    auto gateBiasPtr = GetVLAPtr<float>(gateBias);
    auto inputPtr = GetVLAPtr<float>(input, {length, nChannels});
    auto maskPtr = GetVLAPtr<float>(mask, {length});

#pragma omp parallel
{
    auto tmpProj = std::make_unique<float[]>(blockSize * intermediateChannels);
    auto tmpSigProj = std::make_unique<float[]>(blockSize * intermediateChannels);
#pragma omp for collapse(3)
    for (int64_t i = 0; i < length; i += blockSize) {
        for (int64_t j = 0; j < length; j += blockSize) {
            for (int64_t ib = 0; ib < blockSize; ib++) {
                cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                    blockSize, intermediateChannels, nChannels, 1.0, &inputPtr[i+ib][j][0], nChannels,
                    &weightPtr[0][0], nChannels, 0, &tmpProj[0], intermediateChannels);
                MatAddVec(blockSize, intermediateChannels, intermediateChannels, &biasPtr[0], &tmpProj[0]);

                MatMulVec(blockSize, intermediateChannels, intermediateChannels, intermediateChannels,
                    &maskPtr[i+ib][j], &tmpProj[0], &tmpSigProj[0]);

                cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                    blockSize, intermediateChannels, nChannels, 1.0, &inputPtr[i+ib][j][0], nChannels,
                    &gateWeightPtr[0][0], nChannels, 0, &tmpProj[0], intermediateChannels);
                MatAddVec(blockSize, intermediateChannels, intermediateChannels, &gateBiasPtr[0], &tmpProj[0]);

                SigMoidMul(blockSize, intermediateChannels, intermediateChannels,
                    intermediateChannels, &tmpProj[0], &tmpSigProj[0], &tmpSigProj[0]);

                if (equation == 0) {
                    MatTrans(intermediateChannels, blockSize, intermediateChannels,
                        length * length, &tmpSigProj[0],
                        &projPtr[0][i / blockSize][j / blockSize][ib][0]);
                } else {
                    MatTrans(intermediateChannels, blockSize, intermediateChannels,
                        length * length, &tmpSigProj[0],
                        &projPtr[0][j / blockSize][i / blockSize][ib][0]);
                }
            }
        }
    }
}
    return proj;
}

static void CalculatePairActWithBlockFwd(at::Tensor &act, at::Tensor &mask, int equation, at::Tensor &leftProjWeight,
    at::Tensor &leftProjBias, at::Tensor &leftGateWeight, at::Tensor &leftGateBias, at::Tensor &rightProjWeight,
    at::Tensor &rightProjBias, at::Tensor &rightGateWeight, at::Tensor &rightGateBias, int64_t blockSize)
{
    int64_t length = act.size(DIM0);
    int64_t nChannels = act.size(DIM2);
    int64_t intermediateChannels = leftProjWeight.size(DIM0);
    int64_t length_ = length;

    if (length % blockSize != 0) {
        length = (length / blockSize + 1) * blockSize;

        auto actPad0 = act.new_zeros({length - length_, length, nChannels});
        auto actPad1 = act.new_zeros({length_, length - length_, nChannels});
        act = at::cat({act, actPad1}, DIM1);
        act = at::cat({act, actPad0}, DIM0);

        auto maskPad0 = mask.new_zeros({length - length_, length, 1});
        auto maskPad1 = mask.new_zeros({length_, length - length_, 1});
        mask = at::cat({mask, maskPad1}, DIM1);
        mask = at::cat({mask, maskPad0}, DIM0);
    }

    int64_t blockDim = length / blockSize;

    auto leftProj = CalculateProjWithBlock(act, mask,
        leftProjWeight, leftProjBias, leftGateWeight, leftGateBias,
        equation, blockSize);
    auto leftProjPtr = GetVLAPtr<float>(leftProj, {blockDim, blockDim, blockSize, blockSize});

    auto rightProj = CalculateProjWithBlock(act, mask,
        rightProjWeight, rightProjBias, rightGateWeight, rightGateBias,
        equation, blockSize);
    auto rightProjPtr = GetVLAPtr<float>(rightProj, {blockDim, blockDim, blockSize, blockSize});

    auto pair = act.new_zeros({intermediateChannels, blockDim, blockDim, blockSize, blockSize});
    auto pairPtr = GetVLAPtr<float>(pair, {blockDim, blockDim, blockSize, blockSize});

#pragma omp parallel for collapse(3)
    for (int64_t a = 0; a < intermediateChannels; a++) {
        for (int64_t i = 0; i < length; i += blockSize) {
            for (int64_t j = 0; j < length; j += blockSize) {
                for (int64_t count = 0; count < blockDim; count++) {
                    if (equation == 0) {
                        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                            blockSize, blockSize, blockSize,
                            1.0, &leftProjPtr[a][i / blockSize][count][0][0], blockSize,
                            &rightProjPtr[a][j / blockSize][count][0][0], blockSize,
                            1.0, &pairPtr[a][i / blockSize][j / blockSize][0][0], blockSize);
                    } else {
                        cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                            blockSize, blockSize, blockSize,
                            1.0, &leftProjPtr[a][i / blockSize][count][0][0], blockSize,
                            &rightProjPtr[a][j / blockSize][count][0][0], blockSize,
                            1.0, &pairPtr[a][i / blockSize][j / blockSize][0][0], blockSize);
                    }
                }
            }
        }
    }

    if (equation == 0) {
        act = pair.permute({DIM1, DIM3, DIM2, DIM4, DIM0})
            .reshape({length, length, intermediateChannels});
    } else {
        act = pair.permute({DIM2, DIM4, DIM1, DIM3, DIM0})
            .reshape({length, length, intermediateChannels});
    }

    if (length_ != length) {
        act = act.narrow(DIM1, 0, length_);
        act = act.narrow(DIM0, 0, length_);
    }
    act = act.contiguous();
}

at::Tensor TriangleMultiplicationForward(at::Tensor &act, at::Tensor &mask,
    int equation, at::Tensor &layerNormWeight, at::Tensor &layerNormBias,
    at::Tensor &leftProjWeight, at::Tensor &leftProjBias, at::Tensor &leftGateWeight, at::Tensor &leftGateBias,
    at::Tensor &rightProjWeight, at::Tensor &rightProjBias, at::Tensor &rightGateWeight, at::Tensor &rightGateBias,
    at::Tensor &centerLayerNormWeight, at::Tensor &centerLayerNormBias, at::Tensor &outputProjWeight,
    at::Tensor &outputProjBias, at::Tensor &gatingLinearWeight, at::Tensor &gatingLinearBias, int64_t blockSize)
{
    int64_t length0 = act.size(DIM0);
    int64_t length1 = act.size(DIM1);
    int64_t nChannels = act.size(DIM2);
    int64_t intermediateChannels = leftProjWeight.size(DIM0);
    auto actPtr = GetVLAPtr<float>(act, {length1, nChannels});
    auto pair = act.new_zeros({length0, length1, nChannels});
    auto pairPtr = GetVLAPtr<float>(pair, {length1, nChannels});
    auto layerNormWeightPtr = GetVLAPtr<float>(layerNormWeight);
    auto layerNormBiasPtr = GetVLAPtr<float>(layerNormBias);
    auto centerLayerNormWeightPtr = GetVLAPtr<float>(centerLayerNormWeight);
    auto centerLayerNormBiasPtr = GetVLAPtr<float>(centerLayerNormBias);
    auto outputProjWeightPtr = GetVLAPtr<float>(outputProjWeight, {nChannels});
    auto outputProjBiasPtr = GetVLAPtr<float>(outputProjBias);
    auto gatingLinearWeightPtr = GetVLAPtr<float>(gatingLinearWeight, {nChannels});
    auto gatingLinearBiasPtr = GetVLAPtr<float>(gatingLinearBias);

    if (blockSize <= 0) {
        blockSize = DEFAULT_BLOCK_SIZE;
    }

#pragma omp parallel for collapse(2)
    for (int64_t i = 0; i < length0; i++) {
        for (int64_t j = 0; j < length1; j++) {
            Layernorm(&actPtr[i][j][0], &layerNormWeightPtr[0], &layerNormBiasPtr[0],
                LAYER_NORM_EPS, nChannels, &pairPtr[i][j][0]);
        }
    }

    auto layerNormAct = pair;
    auto layerNormActPtr = GetVLAPtr<float>(layerNormAct, {length1, nChannels});

    CalculatePairActWithBlockFwd(pair, mask, equation,
        leftProjWeight, leftProjBias, leftGateWeight, leftGateBias,
        rightProjWeight, rightProjBias, rightGateWeight, rightGateBias, blockSize);

    pairPtr = GetVLAPtr<float>(pair, {length1, intermediateChannels});
#pragma omp parallel for collapse(2)
    for (int64_t i = 0; i < length0; i++) {
        for (int64_t j = 0; j < length1; j++) {
            Layernorm(&pairPtr[i][j][0], &centerLayerNormWeightPtr[0], &centerLayerNormBiasPtr[0],
                LAYER_NORM_EPS, intermediateChannels, &pairPtr[i][j][0]);
        }
    }

    auto out = act.new_zeros({length0, length1, nChannels});
    auto outPtr = GetVLAPtr<float>(out, {length1, nChannels});
#pragma omp parallel
{
    auto tmpGates = std::make_unique<float[]>(blockSize * nChannels);
    auto tmpSigGates = std::make_unique<float[]>(blockSize * nChannels);
#pragma omp for collapse(2)
    for (int64_t i = 0; i < length0; i++) {
        for (int64_t j = 0; j < length1; j += blockSize) {
            int64_t dims = std::min(length1 - j, blockSize);
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                dims, nChannels, intermediateChannels, 1.0, &pairPtr[i][j][0], intermediateChannels,
                &outputProjWeightPtr[0][0], intermediateChannels, 0.0, &tmpGates[0], nChannels);

            MatAddVec(dims, nChannels, nChannels, &outputProjBiasPtr[0], &tmpGates[0]);

            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                dims, nChannels, nChannels, 1.0, &layerNormActPtr[i][j][0], nChannels,
                &gatingLinearWeightPtr[0][0], nChannels, 0.0, &tmpSigGates[0], nChannels);

            MatAddVec(dims, nChannels, nChannels, &gatingLinearBiasPtr[0], &tmpSigGates[0]);
            SigMoidMul(dims, nChannels, nChannels, nChannels, &tmpSigGates[0], &tmpGates[0], &outPtr[i][j][0]);
        }
    }
}
    return out;
}

}
