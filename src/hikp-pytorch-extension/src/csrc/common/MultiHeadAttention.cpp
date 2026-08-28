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
#include <memory>
#include <ATen/ops/empty.h>
#include <torch/torch.h>
#include "cblas.h"
#include "vla.h"
#include "fused_ops.h"
#include "MultiHeadAttention.h"

namespace common {

static constexpr int64_t DEFAULT_BLOCK_SIZE = 64;

static constexpr float SOFTMAX_MASK_VALUE = -1e9;

MultiHeadAttention::MultiHeadAttention(at::Tensor &queryWeight, at::Tensor &keyWeight,
    at::Tensor &valueWeight, at::Tensor &logitsBias, at::Tensor &nonbatchedBias,
    at::Tensor &outputWeight, at::Tensor &outputBias, int64_t blockSize)
    : keyDim(keyWeight.size(channelDim)), valueDim(valueWeight.size(channelDim)),
      outputDim(outputWeight.size(channelDim)), blockSize(blockSize), nHeads(queryWeight.size(headsDim)),
      queryWeight(queryWeight), keyWeight(keyWeight), valueWeight(valueWeight),
      outputWeight(outputWeight), outputBias(outputBias), logitsBias(logitsBias),
      nonbatchedBias(nonbatchedBias), flag(nonbatchedBias.size(batchDim) > 0) {}

at::Tensor MultiHeadAttention::AttentionFwd(at::Tensor &qdata, at::Tensor &mdata)
{
    at::Tensor query;
    at::Tensor key;
    at::Tensor value;
    at::Tensor weightAvg;
    RunTimeData data(qdata, mdata, query, key, value, weightAvg);

    if (blockSize <= 0) {
        blockSize = DEFAULT_BLOCK_SIZE;
    }

    CalculateQuery(data);
    CalculateKeyAndValue(data);
    CalculateQueryWeight(data);
    auto output = CalculateOutput(data);

    if (data.queryPadLen > 0) {
        output = output.narrow(1, 0, data.querySequenceLen - data.queryPadLen);
    }

    return output;
}

void MultiHeadAttention::CalculateQuery(RunTimeData &data)
{
    int64_t sequenceLen = data.querySequenceLen;
    int64_t qChannels = data.qChannels;
    int64_t nBatch = data.nBatch;
    at::Tensor &qdata = data.qdata;

    if (sequenceLen % blockSize != 0) {
        sequenceLen = (sequenceLen / blockSize + 1) * blockSize;
        data.queryPadLen = sequenceLen - data.querySequenceLen;
        auto qdataPad = qdata.new_zeros({nBatch, data.queryPadLen, qChannels});
        qdata = at::cat({qdata, qdataPad}, sequenceDim);
    }

    data.querySequenceLen = sequenceLen;
    data.query = at::empty({nBatch, sequenceLen, nHeads, keyDim});

    float alpha = (1.0 / sqrt(keyDim));
    auto qdataPtr = GetVLAPtr<float>(qdata, {sequenceLen, qChannels});
    auto queryWeightPtr = GetVLAPtr<float>(queryWeight, {nHeads, keyDim});
    auto queryPtr = GetVLAPtr<float>(data.query, {nHeads, sequenceLen, keyDim});

    // q = torch.einsum('bqa,ahc->bhqc', q_data, self.query_w) * self.key_dim**(-0.5)
#pragma omp parallel
{
    auto queryWeightHead = std::make_unique<float[]>(qChannels * keyDim);
#pragma omp for collapse(2)
    for (int64_t i = 0 ; i < nBatch; i++) {
        for (int64_t j = 0; j < sequenceLen; j += blockSize) {
            for (int64_t k = 0; k < nHeads; k++) {
                MatCopy(qChannels, keyDim, nHeads * keyDim, keyDim, &queryWeightPtr[0][k][0], &queryWeightHead[0]);
                cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    blockSize, keyDim, qChannels, alpha, &qdataPtr[i][j][0], qChannels,
                    &queryWeightHead[0], keyDim, 0.0, &queryPtr[i][k][j][0], keyDim);
            }
        }
    }
}
}

void MultiHeadAttention::CalculateKeyAndValue(RunTimeData &data)
{
    int64_t sequenceLen = data.keySequenceLen;
    int64_t mChannels = data.mChannels;
    int64_t nBatch = data.nBatch;
    at::Tensor &mdata = data.mdata;

    if (sequenceLen % blockSize != 0) {
        sequenceLen = (sequenceLen / blockSize + 1) * blockSize;
        data.keyPadLen = sequenceLen - data.keySequenceLen;
        auto mdataPad = mdata.new_zeros({nBatch, data.keyPadLen, mChannels});
        mdata = at::cat({mdata, mdataPad}, sequenceDim);

        auto logitsBiasPad = logitsBias.new_zeros({nBatch, data.keyPadLen});
        logitsBias = logitsBias.reshape({nBatch, data.keySequenceLen});
        logitsBias = at::cat({logitsBias, logitsBiasPad}, sequenceDim);
        if (flag) {
            auto nonbatchedBiasPad =
                nonbatchedBias.new_zeros({nHeads, data.querySequenceLen - data.queryPadLen, data.keyPadLen});
            nonbatchedBias = at::cat({nonbatchedBias, nonbatchedBiasPad}, channelDim);
            nonbatchedBiasPad =
                nonbatchedBias.new_zeros({nHeads, data.queryPadLen, sequenceLen});
            nonbatchedBias = at::cat({nonbatchedBias, nonbatchedBiasPad}, sequenceDim);
        }
    }

    data.keySequenceLen = sequenceLen;

    data.key = at::empty({nBatch, sequenceLen, nHeads, keyDim});
    data.value = at::empty({nBatch, sequenceLen, nHeads, valueDim});

    auto mdataPtr = GetVLAPtr<float>(mdata, {sequenceLen, mChannels});
    auto keyWeightPtr = GetVLAPtr<float>(keyWeight, {nHeads, keyDim});
    auto keyPtr = GetVLAPtr<float>(data.key, {nHeads, sequenceLen, keyDim});
    auto valueWeightPtr = GetVLAPtr<float>(valueWeight, {nHeads, valueDim});
    auto valuePtr = GetVLAPtr<float>(data.value, {nHeads, sequenceLen, valueDim});

    /*
     * k = torch.einsum('bka,ahc->bhkc', m_data, self.key_w)
     * v = torch.einsum('bka,ahc->bhkc', m_data, self.value_w)
     */
#pragma omp parallel
{
    auto keyWeightHead = std::make_unique<float[]>(mChannels * keyDim);
    auto valueWeightHead = std::make_unique<float[]>(mChannels * valueDim);
#pragma omp for collapse(2)
    for (int64_t i = 0 ; i < nBatch; i++) {
        for (int64_t j = 0; j < sequenceLen; j += blockSize) {
            for (int64_t k = 0; k < nHeads; k++) {
                MatCopy(mChannels, keyDim, nHeads * keyDim, keyDim, &keyWeightPtr[0][k][0], &keyWeightHead[0]);
                cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    blockSize, keyDim, mChannels, 1.0, &mdataPtr[i][j][0], mChannels,
                    &keyWeightHead[0], keyDim, 0.0, &keyPtr[i][k][j][0], keyDim);

                MatCopy(mChannels, valueDim, nHeads * valueDim, valueDim,
                    &valueWeightPtr[0][k][0], &valueWeightHead[0]);
                cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    blockSize, valueDim, mChannels, 1.0, &mdataPtr[i][j][0], mChannels,
                    &valueWeightHead[0], valueDim, 0.0, &valuePtr[i][k][j][0], valueDim);
            }
        }
    }
}
}

void MultiHeadAttention::CalculateQueryWeight(RunTimeData &data)
{
    int64_t querySequenceLen = data.querySequenceLen;
    int64_t keySequenceLen = data.keySequenceLen;
    int64_t nBatch = data.nBatch;
    data.weightAvg = at::empty({nBatch, querySequenceLen, nHeads, valueDim});
    auto queryDataPtr = GetVLAPtr<float>(data.query, {nHeads, querySequenceLen, keyDim});
    auto keyDataPtr = GetVLAPtr<float>(data.key, {nHeads, keySequenceLen, keyDim});
    auto valueDataPtr = GetVLAPtr<float>(data.value, {nHeads, keySequenceLen, valueDim});
    auto logitsBiasPtr = GetVLAPtr<float>(logitsBias, {keySequenceLen});
    auto nonbatchedBiasPtr = GetVLAPtr<float>(nonbatchedBias, {querySequenceLen, keySequenceLen});
    auto weightAvgPtr = GetVLAPtr<float>(data.weightAvg, {querySequenceLen, nHeads, valueDim});

    auto softMaxMask = std::make_unique<float[]>(data.keyPadLen);
    std::fill_n(&softMaxMask[0], data.keyPadLen, SOFTMAX_MASK_VALUE);

    /*
     * logits = torch.einsum('bhqc,bhkc->bhqk', q, k) + bias
     * if nonbatched_bias.shape[0] > 0:
     *   logits += torch.unsqueeze(nonbatched_bias, dim=0)
     * weights = self.softmax(logits)
     * weighted_avg = torch.einsum('bhqk,bhkc->bqhc', weights, v)
    */
#pragma omp parallel
{
    auto logits = std::make_unique<float[]>(blockSize * keySequenceLen);
    auto tempWeight = std::make_unique<float[]>(blockSize * valueDim);
#pragma omp for collapse(3)
    for (int64_t i = 0; i < nBatch; i++) {
        for (int64_t k = 0; k < nHeads; k++) {
            for (int64_t j = 0; j < querySequenceLen; j += blockSize) {
                cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                    blockSize, keySequenceLen, keyDim, 1.0, &queryDataPtr[i][k][j][0], keyDim,
                    &keyDataPtr[i][k][0][0], keyDim, 0.0, &logits[0], keySequenceLen);

                MatAddVec(blockSize, keySequenceLen, keySequenceLen, &logitsBiasPtr[i][0], &logits[0]);

                if (flag) {
                    MatAddMat(blockSize, keySequenceLen, keySequenceLen,
                        keySequenceLen, keySequenceLen, &nonbatchedBiasPtr[k][j][0], &logits[0], &logits[0]);
                }

                if (data.keyPadLen > 0) {
                    MatAddVec(blockSize, data.keyPadLen, keySequenceLen,
                        &softMaxMask[0], &logits[data.keySequenceLen - data.keyPadLen]);
                }

                SoftMax(1, blockSize, keySequenceLen, &logits[0], &logits[0]);

                cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    blockSize, valueDim, keySequenceLen, 1.0, &logits[0], keySequenceLen,
                    &valueDataPtr[i][k][0][0], valueDim, 0.0, &tempWeight[0], valueDim);
                MatCopy(blockSize, valueDim, valueDim, nHeads * valueDim, &tempWeight[0], &weightAvgPtr[i][j][k][0]);
            }
        }
    }
}
}

at::Tensor MultiHeadAttention::CalculateOutput(RunTimeData &data)
{
    int64_t querySequenceLen = data.querySequenceLen;
    int64_t nBatch = data.nBatch;
    auto weightAvgPtr = GetVLAPtr<float>(data.weightAvg, {querySequenceLen, nHeads, valueDim});
    auto outputWeightPtr = GetVLAPtr<float>(outputWeight, {valueDim, outputDim});
    auto outputBiasPtr = GetVLAPtr<float>(outputBias);
    auto output = at::empty({nBatch, querySequenceLen, outputDim});
    auto outputPtr = GetVLAPtr<float>(output, {querySequenceLen, outputDim});

#pragma omp parallel for collapse(2)
    for (int64_t i = 0 ; i < nBatch; i++) {
        for (int64_t j = 0; j < querySequenceLen; j += blockSize) {
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                blockSize, outputDim, nHeads * valueDim, 1.0, &weightAvgPtr[i][j][0][0], nHeads * valueDim,
                &outputWeightPtr[0][0][0], outputDim, 0.0, &outputPtr[i][j][0], outputDim);
            MatAddVec(blockSize, outputDim, outputDim, &outputBiasPtr[0], &outputPtr[i][j][0]);
        }
    }

    return output;
}

at::Tensor MultiHeadAttentionFwd(at::Tensor &qdata, at::Tensor &mdata, at::Tensor &queryWeight,
    at::Tensor &keyWeight, at::Tensor &valueWeight, at::Tensor &logitsBias,
    at::Tensor &nonbatchedBias, at::Tensor &outputWeight, at::Tensor &outputBias, int64_t blockSize)
{
    MultiHeadAttention atten(queryWeight, keyWeight, valueWeight, logitsBias, nonbatchedBias,
        outputWeight, outputBias, blockSize);
    return atten.AttentionFwd(qdata, mdata);
}

}
