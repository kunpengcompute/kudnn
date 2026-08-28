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
#include "outer_product_mean.h"

namespace alphafold {

static constexpr int64_t DEFAULT_BLOCK_SIZE = 64;

static at::Tensor CalculateActProj(at::Tensor& act, at::Tensor& mask,
    at::Tensor& weight, at::Tensor& bias, int64_t blockSize)
{
    int64_t batch = act.size(0);
    int64_t sequenceLen = act.size(1);
    int64_t nChannels = act.size(2);
    int64_t numOuterChannel = weight.size(0);

    auto proj = act.new_empty({sequenceLen, batch, numOuterChannel});
    auto actPtr = GetVLAPtr<float>(act, {sequenceLen, nChannels});
    auto maskPtr = GetVLAPtr<float>(mask, {sequenceLen});
    auto projPtr = GetVLAPtr<float>(proj, {batch, numOuterChannel});
    auto weightPtr = GetVLAPtr<float>(weight, {nChannels});
    auto biasPtr = GetVLAPtr<float>(bias);

#pragma omp parallel
{
    auto tmpProj = std::make_unique<float[]>(blockSize * numOuterChannel);
#pragma omp for collapse(2)
    for (int64_t i = 0; i < batch; i++) {
        for (int64_t j = 0; j < sequenceLen; j += blockSize) {
            int64_t dims = std::min(sequenceLen - j, blockSize);
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                dims, numOuterChannel, nChannels, 1.0, &actPtr[i][j][0], nChannels,
                &weightPtr[0][0], nChannels, 0.0, &tmpProj[0], numOuterChannel);
            for (int64_t k = 0; k < dims; k++) {
                VecAddMul(numOuterChannel, &tmpProj[k * numOuterChannel],
                    maskPtr[i][j + k], &biasPtr[0], &projPtr[j + k][i][0]);
            }
        }
    }
}
    return proj;
}

at::Tensor OuterProductMean(at::Tensor& act, at::Tensor& mask, int64_t sequenceLen,
    at::Tensor& layerNormWeight, at::Tensor& layerNormBias, at::Tensor& leftProjWeight,
    at::Tensor& leftProjBias, at::Tensor& rightProjWeight, at::Tensor& rightProjBias,
    at::Tensor& outputProjWeight, at::Tensor& outputProjBias, int64_t blockSize)
{
    int64_t batch = act.size(0);
    int64_t sequenceLenTmp = act.size(1);
    int64_t numOuterChannel = leftProjWeight.size(0);
    int64_t numOutputChannel = outputProjWeight.size(2);

    if (blockSize <= 0) {
        blockSize = DEFAULT_BLOCK_SIZE;
    }

    auto leftProj = CalculateActProj(act, mask, leftProjWeight, leftProjBias, blockSize);
    auto leftProjPtr = GetVLAPtr<float>(leftProj, {batch, numOuterChannel});

    auto rightProj = CalculateActProj(act, mask, rightProjWeight, rightProjBias, blockSize);
    auto rightProjPtr = GetVLAPtr<float>(rightProj, {batch, numOuterChannel});

    auto pairAct = act.new_zeros({sequenceLenTmp, sequenceLen, numOuterChannel, numOuterChannel});
    auto pairActPtr = GetVLAPtr<float>(pairAct, {sequenceLen, numOuterChannel, numOuterChannel});

#pragma omp parallel for collapse(2)
    for (int64_t i = 0; i < sequenceLenTmp; i++) {
        for (int64_t j = 0; j < sequenceLen; j++) {
            cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                numOuterChannel, numOuterChannel, batch, 1.0, &leftProjPtr[i][0][0], numOuterChannel,
                &rightProjPtr[j][0][0], numOuterChannel, 0.0, &pairActPtr[i][j][0][0], numOuterChannel);
        }
    }

    auto out = act.new_zeros({sequenceLenTmp, sequenceLen, numOutputChannel});
    auto outPtr = GetVLAPtr<float>(out, {sequenceLen, numOutputChannel});
    auto outputProjWeightPtr = GetVLAPtr<float>(outputProjWeight, {numOuterChannel, numOutputChannel});
    auto outputProjBiasPtr = GetVLAPtr<float>(outputProjBias);

#pragma omp parallel for collapse(2)
    for (int64_t i = 0; i < sequenceLenTmp; i++) {
        for (int64_t j = 0; j < sequenceLen; j += blockSize) {
            int64_t dims = (j + blockSize >= sequenceLen) ? (sequenceLen - j) : blockSize;
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                dims, numOutputChannel, numOuterChannel * numOuterChannel,
                1.0, &pairActPtr[i][j][0][0], numOuterChannel * numOuterChannel,
                &outputProjWeightPtr[0][0][0], numOutputChannel, 0.0, &outPtr[i][j][0], numOutputChannel);
            MatAddVec(dims, numOutputChannel, numOutputChannel, &outputProjBiasPtr[0], &outPtr[i][j][0]);
        }
    }

    return out;
}

}