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
#include "MultiHeadAttention.h"

namespace alphafold {

class GatingAttention : public common::MultiHeadAttention {
public:
    GatingAttention(at::Tensor &queryWeight, at::Tensor &keyWeight,
        at::Tensor &valueWeight, at::Tensor &logitsBias, at::Tensor &nonbatchedBias,
        at::Tensor &gatingWeight, at::Tensor &gatingBias,
        at::Tensor &outputWeight, at::Tensor &outputBias, int64_t blockSize)
        : MultiHeadAttention(queryWeight, keyWeight, valueWeight, logitsBias,
              nonbatchedBias, outputWeight, outputBias, blockSize),
          gatingWeight(gatingWeight), gatingBias(gatingBias) {}

    at::Tensor CalculateOutput(RunTimeData &data) override
    {
        int64_t querySequenceLen = data.querySequenceLen;
        int64_t qChannels = data.qChannels;
        int64_t nBatch = data.nBatch;
        auto qdataPtr = GetVLAPtr<float>(data.qdata, {querySequenceLen, qChannels});
        auto weightAvgPtr = GetVLAPtr<float>(data.weightAvg, {querySequenceLen, nHeads, valueDim});
        auto gatingWeightPtr = GetVLAPtr<float>(gatingWeight, {nHeads, valueDim});
        auto gatingBiasPtr = GetVLAPtr<float>(gatingBias, {valueDim});
        auto outputWeightPtr = GetVLAPtr<float>(outputWeight, {valueDim, outputDim});
        auto outputBiasPtr = GetVLAPtr<float>(outputBias);
        auto output = at::empty({nBatch, querySequenceLen, outputDim});
        auto outputPtr = GetVLAPtr<float>(output, {querySequenceLen, outputDim});
#pragma omp parallel
{
        int64_t fullConnectDim = nHeads * valueDim;
        auto tmpGates = std::make_unique<float[]>(blockSize * fullConnectDim);
        auto tmpSigGates = std::make_unique<float[]>(blockSize * fullConnectDim);
#pragma omp for collapse(2)
        for (int64_t i = 0 ; i < nBatch; i++) {
            for (int64_t j = 0; j < querySequenceLen; j += blockSize) {
                cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    blockSize, fullConnectDim, qChannels, 1.0, &qdataPtr[i][j][0], qChannels,
                    &gatingWeightPtr[0][0][0], fullConnectDim, 0.0, &tmpGates[0], fullConnectDim);
            
                MatAddVec(blockSize, fullConnectDim, fullConnectDim, &gatingBiasPtr[0][0], &tmpGates[0]);

                SigMoidMul(blockSize, fullConnectDim, fullConnectDim, fullConnectDim,
                    &tmpGates[0], &weightAvgPtr[i][j][0][0], &tmpSigGates[0]);

                cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    blockSize, outputDim, fullConnectDim, 1.0, &tmpSigGates[0], fullConnectDim,
                    &outputWeightPtr[0][0][0], outputDim, 0, &outputPtr[i][j][0], outputDim);
 
                MatAddVec(blockSize, outputDim, outputDim, &outputBiasPtr[0], &outputPtr[i][j][0]);
            }
        }
}
        return output;
    }

private:
    at::Tensor &gatingWeight;
    at::Tensor &gatingBias;
};

at::Tensor GatingAttentionFwd(at::Tensor &qdata, at::Tensor &mdata, at::Tensor &queryWeight,
    at::Tensor &keyWeight, at::Tensor &valueWeight, at::Tensor &logitsBias,
    at::Tensor &nonbatchedBias, at::Tensor &gatingWeight, at::Tensor &gatingBias,
    at::Tensor &outputWeight, at::Tensor &outputBias, int64_t blockSize)
{
    GatingAttention atten(queryWeight, keyWeight, valueWeight, logitsBias, nonbatchedBias,
        gatingWeight, gatingBias, outputWeight, outputBias, blockSize);
    return atten.AttentionFwd(qdata, mdata);
}

}

