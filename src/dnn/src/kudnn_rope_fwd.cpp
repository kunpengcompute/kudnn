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

#include <cstdlib>
#include <memory>
#include <cmath>
#include <cstddef>

#include <arm_sve.h>

#include "kudnn.hpp"
#include "threading_impl.hpp"
#include "internal_funcs.hpp"

namespace KuDNN {

static const int HALF_TWO = 2;

namespace Detail {

static auto g_freeDeleter = [](void *ptr) {
    free(ptr);
};

class RopeLayerFWDImpl {
public:
    RopeLayerFWDImpl(const int maxPosition_, const int dim_, const float base_ = 10000,
                     const float scalingFactor_ = 1.0f) noexcept(false)
        : maxPosition(maxPosition_), dim(dim_), base(base_), scalingFactor(scalingFactor_), invFreqSize(dim_ / HALF_TWO)
    {
        Service::ThrowOnStatus(RopeLayerFWD::ValidateInput(maxPosition, dim, base, scalingFactor), "Rope Layer");

        invFreq.reset(static_cast<__fp16 *>(malloc(invFreqSize * sizeof(__fp16))), g_freeDeleter);
        embCos.reset(static_cast<__fp16 *>(Service::AlignedAlloc(maxPosition * invFreqSize * sizeof(__fp16))),
                     g_freeDeleter);
        embSin.reset(static_cast<__fp16 *>(Service::AlignedAlloc(maxPosition * invFreqSize * sizeof(__fp16))),
                     g_freeDeleter);
        // theta
        __fp16 invScalingFactor = 1.0 / scalingFactor;
        for (int i = 0; i < invFreqSize; ++i) {
            invFreq.get()[i] = invScalingFactor / std::pow(base, float(i * HALF_TWO) / dim);
        }
        // cos and sin
#pragma omp parallel for
        for (int i = 0; i < maxPosition; i++) {
            __fp16 *pcos = embCos.get() + i * invFreqSize;
            __fp16 *psin = embSin.get() + i * invFreqSize;

            for (int j = 0; j < invFreqSize; j++) {
                float tmp = i * invFreq.get()[j];
                pcos[j] = std::cos(tmp);
                psin[j] = std::sin(tmp);
            }
        }
    }

    void EvaluateSeek(const __fp16 *pcos, const __fp16 *psin, __fp16 *q, __fp16 *k, const int head, const int qHeads,
                      const int kHeads) const noexcept
    {
        // deepseek
        const int loopLen = invFreqSize & 0xFFFFFFE0;
        const int remain = invFreqSize & 0x0000001F;
        int i = 0;
        svbool_t pg_b16_vl32 = svptrue_b16();
        for (; i < loopLen; i += 32) {
            svfloat16_t cosVec = svld1_f16(pg_b16_vl32, pcos + i);
            svfloat16_t sinVec = svld1_f16(pg_b16_vl32, psin + i);
            if (head < qHeads) {
                svfloat16_t qVec1 = svld1_f16(pg_b16_vl32, q + i);
                svfloat16_t qVec2 = svld1_f16(pg_b16_vl32, q + i + invFreqSize);
                svfloat16_t qVecEmb1 =
                    svnmsb_f16_x(pg_b16_vl32, qVec1, cosVec, svmul_f16_x(pg_b16_vl32, qVec2, sinVec));
                svfloat16_t qVecEmb2 = svmad_f16_x(pg_b16_vl32, qVec2, cosVec, svmul_f16_x(pg_b16_vl32, qVec1, sinVec));
                svst1_f16(pg_b16_vl32, q + i, qVecEmb1);
                svst1_f16(pg_b16_vl32, q + i + invFreqSize, qVecEmb2);
            }
            if (head < kHeads) {
                svfloat16_t kVec1 = svld1_f16(pg_b16_vl32, k + i);
                svfloat16_t kVec2 = svld1_f16(pg_b16_vl32, k + i + invFreqSize);
                svfloat16_t kVecEmb1 =
                    svnmsb_f16_x(pg_b16_vl32, kVec1, cosVec, svmul_f16_x(pg_b16_vl32, kVec2, sinVec));
                svfloat16_t kVecEmb2 = svmad_f16_x(pg_b16_vl32, kVec2, cosVec, svmul_f16_x(pg_b16_vl32, kVec1, sinVec));
                svst1_f16(pg_b16_vl32, k + i, kVecEmb1);
                svst1_f16(pg_b16_vl32, k + i + invFreqSize, kVecEmb2);
            }
        }
        if (remain == 0) {
            return;
        }
        svbool_t pg_tail = svwhilelt_b16_s32(i, invFreqSize);
        svfloat16_t cosVec = svld1_f16(pg_tail, pcos + i);
        svfloat16_t sinVec = svld1_f16(pg_tail, psin + i);
        if (head < qHeads) {
            svfloat16_t qVec1 = svld1_f16(pg_tail, q + i);
            svfloat16_t qVec2 = svld1_f16(pg_tail, q + i + invFreqSize);
            svfloat16_t qVecEmb1 = svnmsb_f16_x(pg_b16_vl32, qVec1, cosVec, svmul_f16_x(pg_b16_vl32, qVec2, sinVec));
            svfloat16_t qVecEmb2 = svmad_f16_x(pg_b16_vl32, qVec2, cosVec, svmul_f16_x(pg_b16_vl32, qVec1, sinVec));
            svst1_f16(pg_tail, q + i, qVecEmb1);
            svst1_f16(pg_tail, q + i + invFreqSize, qVecEmb2);
        }
        if (head < kHeads) {
            svfloat16_t kVec1 = svld1_f16(pg_tail, k + i);
            svfloat16_t kVec2 = svld1_f16(pg_tail, k + i + invFreqSize);
            svfloat16_t kVecEmb1 = svnmsb_f16_x(pg_b16_vl32, kVec1, cosVec, svmul_f16_x(pg_b16_vl32, kVec2, sinVec));
            svfloat16_t kVecEmb2 = svmad_f16_x(pg_b16_vl32, kVec2, cosVec, svmul_f16_x(pg_b16_vl32, kVec1, sinVec));
            svst1_f16(pg_tail, k + i, kVecEmb1);
            svst1_f16(pg_tail, k + i + invFreqSize, kVecEmb2);
        }
    }

    void EvaluateSora(const __fp16 *pcos, const __fp16 *psin, __fp16 *q, __fp16 *k, const int head, const int qHeads,
                      const int kHeads) const noexcept
    {
        // opensora
        const int loopLen = invFreqSize & 0xFFFFFFF0;
        const int remain = invFreqSize & 0x0000000F;
        int i = 0;
        svbool_t pg_b16_vl32 = svptrue_b16();
        for (; i < loopLen; i += 16) {
            svfloat16_t cosVec = svld1_f16(pg_b16_vl32, pcos + i);
            svfloat16_t cosTmp = svzip1(cosVec, cosVec);
            svfloat16_t sinVec = svld1_f16(pg_b16_vl32, psin + i);
            svfloat16_t sinTmp = svzip1(sinVec, sinVec);
            if (head < qHeads) {
                svfloat16_t qLeft = svld1_f16(pg_b16_vl32, q + (i << 1));
                svfloat16_t even = svuzp1(qLeft, qLeft);
                svfloat16_t odd = svuzp2(qLeft, qLeft);
                svfloat16_t oddNeg = svneg_f16_x(pg_b16_vl32, odd);
                svfloat16_t qRight = svzip1(oddNeg, even);
                svfloat16_t qEmb = svmad_f16_x(pg_b16_vl32, qLeft, cosTmp, svmul_f16_x(pg_b16_vl32, qRight, sinTmp));
                svst1_f16(pg_b16_vl32, q + (i << 1), qEmb);
            }
            if (head < kHeads) {
                svfloat16_t kLeft = svld1_f16(pg_b16_vl32, k + (i << 1));
                svfloat16_t even = svuzp1(kLeft, kLeft);
                svfloat16_t odd = svuzp2(kLeft, kLeft);
                svfloat16_t oddNeg = svneg_f16_x(pg_b16_vl32, odd);
                svfloat16_t kRight = svzip1(oddNeg, even);
                svfloat16_t kEmb = svmad_f16_x(pg_b16_vl32, kLeft, cosTmp, svmul_f16_x(pg_b16_vl32, kRight, sinTmp));
                svst1_f16(pg_b16_vl32, k + (i << 1), kEmb);
            }
        }
        if (remain == 0) {
            return;
        }
        svbool_t pg_tail = svwhilelt_b16_s32(i, invFreqSize);
        svbool_t pg_src = svwhilelt_b16_s32(i << 1, invFreqSize << 1);
        svfloat16_t cosVec = svld1_f16(pg_tail, pcos + i);
        svfloat16_t cosTmp = svzip1(cosVec, cosVec);
        svfloat16_t sinVec = svld1_f16(pg_tail, psin + i);
        svfloat16_t sinTmp = svzip1(sinVec, sinVec);
        if (head < qHeads) {
            svfloat16_t qLeft = svld1_f16(pg_src, q + (i << 1));
            svfloat16_t even = svuzp1(qLeft, qLeft);
            svfloat16_t odd = svuzp2(qLeft, qLeft);
            svfloat16_t oddNeg = svneg_f16_x(pg_b16_vl32, odd);
            svfloat16_t qRight = svzip1(oddNeg, even);
            svfloat16_t qEmb = svmad_f16_x(pg_b16_vl32, qLeft, cosTmp, svmul_f16_x(pg_b16_vl32, qRight, sinTmp));
            svst1_f16(pg_src, q + (i << 1), qEmb);
        }
        if (head < kHeads) {
            svfloat16_t kLeft = svld1_f16(pg_src, k + (i << 1));
            svfloat16_t even = svuzp1(kLeft, kLeft);
            svfloat16_t odd = svuzp2(kLeft, kLeft);
            svfloat16_t oddNeg = svneg_f16_x(pg_b16_vl32, odd);
            svfloat16_t kRight = svzip1(oddNeg, even);
            svfloat16_t kEmb = svmad_f16_x(pg_b16_vl32, kLeft, cosTmp, svmul_f16_x(pg_b16_vl32, kRight, sinTmp));
            svst1_f16(pg_src, k + (i << 1), kEmb);
        }
    }

    void Evaluate(const __fp16 *pcos, const __fp16 *psin, __fp16 *q, __fp16 *k, const int head, const int qHeads,
                  const int kHeads, KuDNN::RopeKind kind) const noexcept
    {
        if (kind == KuDNN::RopeKind::OpenSora) {
            EvaluateSora(pcos, psin, q, k, head, qHeads, kHeads);
        } else if (kind == KuDNN::RopeKind::DeepSeek) {
            EvaluateSeek(pcos, psin, q, k, head, qHeads, kHeads);
        } else {
            return;
        }
    }

    // query: [batchSize, seqLen, qHeads, dim]
    // qStride: qHeads * dim
    void Run(void *queryD, void *keyD, const int qStride, const int kStride, const int batchSize, const int seqLen,
             const int qHeads, const int kHeads, const int *positionIds, KuDNN::RopeKind kind) const noexcept(false)
    {
        CheckInputArg(queryD, keyD, qStride, kStride, batchSize, seqLen, qHeads, kHeads, positionIds, kind);
        __fp16 *query = static_cast<__fp16 *>(queryD);
        __fp16 *key = static_cast<__fp16 *>(keyD);
        const int heads = std::max(qHeads, kHeads);

#pragma omp parallel for collapse(3)
        for (int bs = 0; bs < batchSize; ++bs) {
            for (int seq = 0; seq < seqLen; ++seq) {
                for (int head = 0; head < heads; ++head) {
                    int pos = positionIds[seq];
                    __fp16 *pcos = embCos.get() + pos * invFreqSize;
                    __fp16 *psin = embSin.get() + pos * invFreqSize;
                    __fp16 *q = query + bs * seqLen * qStride + seq * qStride + head * dim;
                    __fp16 *k = key + bs * seqLen * kStride + seq * kStride + head * dim;
                    Evaluate(pcos, psin, q, k, head, qHeads, kHeads, kind);
                }
            }
        }
    }

    ~RopeLayerFWDImpl() noexcept {}

private:
    int maxPosition;
    int dim;
    float base;
    float scalingFactor;
    int invFreqSize;
    std::shared_ptr<__fp16> invFreq;
    std::shared_ptr<__fp16> embCos;
    std::shared_ptr<__fp16> embSin;

    void CheckInputArg(void *query, void *key, const int qStride, const int kStride, const int batchSize,
                       const int seqLen, const int qHeads, const int kHeads, const int *positionIds,
                       KuDNN::RopeKind kind) const
    {
        if (query == nullptr || key == nullptr) {
            throw Service::LogicError{"RoPE: query or key is nullptr"};
        }
        if (positionIds == nullptr) {
            throw Service::LogicError{"RoPE: positionIds is nullptr"};
        }
        for (int seq = 0; seq < seqLen; ++seq) {
            int pos = positionIds[seq];
            if (pos < 0) {
                throw Service::LogicError{"RoPE: illegal positionIds < 0"};
            }
        }
        if (qStride <= 0 || kStride <= 0 || batchSize <= 0 || seqLen <= 0 || qHeads <= 0 || kHeads <= 0) {
            throw Service::LogicError{"RoPE: illegal input"};
        }
        Service::Unused(kind);
    }
}; // RopeLayerFWDImpl
} // namespace Detail

RopeLayerFWD::~RopeLayerFWD() noexcept {}

RopeLayerFWD::RopeLayerFWD(const int maxPosition, const int dim, const float base,
                           const float scalingFactor) noexcept(false)
    : pImpl(new Detail::RopeLayerFWDImpl(maxPosition, dim, base, scalingFactor))
{
}

RopeLayerFWD::RopeLayerFWD(const RopeLayerFWD &other) noexcept(false)
    : pImpl(new Detail::RopeLayerFWDImpl{*(other.pImpl.get())})
{
}

RopeLayerFWD::RopeLayerFWD(RopeLayerFWD &&other) noexcept : pImpl(std::move(other.pImpl)) {}

RopeLayerFWD &RopeLayerFWD::operator=(const RopeLayerFWD &other) noexcept(false)
{
    if (this != &other) {
        this->pImpl.reset(new Detail::RopeLayerFWDImpl{*(other.pImpl.get())});
    }
    return *this;
}

RopeLayerFWD &RopeLayerFWD::operator=(RopeLayerFWD &&other) noexcept
{
    if (this != &other) {
        this->pImpl = std::move(other.pImpl);
    }
    return *this;
}

void RopeLayerFWD::Run(void *query, void *key, const int qStride, const int kStride, const int batchSize,
                       const int seqLen, const int qHeads, const int kHeads, const int *positionIds,
                       KuDNN::RopeKind kind) const noexcept(false)
{
    pImpl->Run(query, key, qStride, kStride, batchSize, seqLen, qHeads, kHeads, positionIds, kind);
}

Status RopeLayerFWD::ValidateInput(const int maxPosition, const int dim, const float base,
                                   const float scalingFactor) noexcept
{
    if (maxPosition <= 0) {
        return Status::BAD_ARGUMENTS;
    }
    if (dim <= 0 || dim % HALF_TWO != 0) {
        return Status::BAD_ARGUMENTS;
    }
    if (base <= 0) {
        return Status::BAD_ARGUMENTS;
    }
    if (std::abs(scalingFactor) < 1e-5) {
        return Status::BAD_ARGUMENTS;
    }

    return Status::SUCCESS;
}

} // namespace KuDNN
