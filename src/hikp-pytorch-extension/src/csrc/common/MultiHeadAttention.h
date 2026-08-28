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

#pragma once
#include <ATen/core/Tensor.h>
#include <torch/extension.h>

namespace common {

enum {
    batchDim = 0,
    sequenceDim = 1,
    headsDim = 1,
    channelDim = 2,
};

class MultiHeadAttention {
public:

    class RunTimeData {
    public:
        RunTimeData(at::Tensor &qdata, at::Tensor &mdata, at::Tensor &query,
            at::Tensor &key, at::Tensor &value, at::Tensor &weightAvg)
            : nBatch(qdata.size(batchDim)), querySequenceLen(qdata.size(sequenceDim)),
              keySequenceLen(mdata.size(sequenceDim)), qChannels(qdata.size(channelDim)),
              mChannels(mdata.size(channelDim)), queryPadLen(0), keyPadLen(0),
              qdata(qdata), mdata(mdata), query(query), key(key), value(value), weightAvg(weightAvg) {}
        int64_t nBatch;
        int64_t querySequenceLen;
        int64_t keySequenceLen;
        int64_t qChannels;
        int64_t mChannels;
        int64_t queryPadLen;
        int64_t keyPadLen;
        at::Tensor &qdata;
        at::Tensor &mdata;
        at::Tensor &query;
        at::Tensor &key;
        at::Tensor &value;
        at::Tensor &weightAvg;
    };

    MultiHeadAttention(at::Tensor &queryWeight, at::Tensor &keyWeight,
        at::Tensor &valueWeight, at::Tensor &logitsBias, at::Tensor &nonbatchedBias,
        at::Tensor &outputWeight, at::Tensor &outputBias, int64_t blockSize);
    virtual void CalculateQuery(RunTimeData &data);
    virtual void CalculateKeyAndValue(RunTimeData &data);
    virtual void CalculateQueryWeight(RunTimeData &data);
    virtual at::Tensor CalculateOutput(RunTimeData &data);
    at::Tensor AttentionFwd(at::Tensor &qdata, at::Tensor &mdata);

protected:
    int64_t keyDim;
    int64_t valueDim;
    int64_t outputDim;
    int64_t blockSize;
    int64_t nHeads;
    at::Tensor &queryWeight;
    at::Tensor &keyWeight;
    at::Tensor &valueWeight;
    at::Tensor &outputWeight;
    at::Tensor &outputBias;
    at::Tensor &logitsBias;
    at::Tensor &nonbatchedBias;
    bool flag;
};

at::Tensor MultiHeadAttentionFwd(at::Tensor &qdata, at::Tensor &mdata, at::Tensor &queryWeight,
    at::Tensor &keyWeight, at::Tensor &valueWeight, at::Tensor &logitsBias,
    at::Tensor &nonbatchedBias, at::Tensor &outputWeight, at::Tensor &outputBias, int64_t blockSize);
}  // namespace alphafold
