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

#include <arm_sve.h>
#include <ATen/ops/empty.h>
#include <torch/torch.h>
#include "flash_attention.h"
#include "utils/dispatch.h"
#include "utils/blas_wrapper.h"
#include "utils/fast_exp.h"
#include "utils/fused_ops.h"
#include "utils/scalar_type.h"

enum {
    DIM0 = 0,
    DIM1 = 1,
    DIM2 = 2,
    DIM3 = 3,
};

enum {
    FLASH_ATTN_Q_SIZE_THRESHOLD_0 = 192,
    FLASH_ATTN_Q_SIZE_THRESHOLD_1 = 768,
};

#define FLASH_ATTN_MAX_DIM_SIZE       4

namespace common {
static float CalculateScale(at::Tensor query, std::optional<double> &scale)
{
    if (query.size(-1) == 0) {
        return scale.has_value() ? static_cast<float>(scale.value()) : 1.0;
    }
    const auto softmaxScale = scale.has_value()
        ? scale.value()
        : (1.0 / sqrt(query.size(-1)));
    return static_cast<float>(softmaxScale);
}

static void ReshapeAttnMask(
    at::Tensor& mask,
    int64_t batchSize,
    int64_t nHeads,
    int64_t qSize,
    int64_t kvSize)
{
    int64_t attnMaskSize0 = 1;
    int64_t attnMaskSize1 = 1;
    if (mask.dim() == FLASH_ATTN_MAX_DIM_SIZE) {
        if (mask.size(DIM0) == batchSize) {
            attnMaskSize0 = batchSize;
        }
        if (mask.size(DIM1) == nHeads) {
            attnMaskSize1 = nHeads;
        }
    }
    mask = mask
        .view({attnMaskSize0, attnMaskSize1, mask.size(-2), mask.size(-1)})
        .expand({attnMaskSize0, attnMaskSize1, qSize, kvSize});
}

static float ExpReduceSumFusionKernel(int64_t size, float *in, float *out, float tmpMax)
{
    const int64_t vl = svcntw();
    svfloat32_t reduce = svdup_f32(0.f);
    for (int64_t i = 0; i < size; i += vl) {
        svbool_t pg = svwhilelt_b32(i, size);
        svfloat32_t values = svld1_f32(pg, in + i);
        values = FastExp(pg, svsub_n_f32_x(pg, values, tmpMax));
        reduce = svadd_f32_m(pg, reduce, values);
        svst1_f32(pg, out + i, values);
    }
    return svaddv_f32(svptrue_b32(), reduce);
}

static float ExpReduceSumFusionKernel(int64_t size, float *in, float16_t *out, float tmpMax)
{
    const int64_t vlf = svcntw();
    const int64_t vlh = svcnth();
    svfloat32_t reduce = svdup_f32(0.f);
    for (int64_t i = 0; i < size; i += vlh) {
        svbool_t pgFloat0 = svwhilelt_b32(i, size);
        svbool_t pgFloat1 = svwhilelt_b32(i + vlf, size);
        svbool_t pgHalf = svwhilelt_b16(i, size);
        svfloat32_t values0 = svld1_f32(pgFloat0, in + i);
        svfloat32_t values1 = svld1_f32(pgFloat1, in + i + vlf);
        values0 = FastExp(pgFloat0, svsub_n_f32_x(pgFloat0, values0, tmpMax));
        values1 = FastExp(pgFloat1, svsub_n_f32_x(pgFloat1, values1, tmpMax));
        reduce = svadd_f32_m(pgFloat0, reduce, values0);
        reduce = svadd_f32_m(pgFloat1, reduce, values1);
        svfloat16_t res0 = svcvt_f16_f32_x(svptrue_b32(), values0);
        svfloat16_t res1 = svcvt_f16_f32_x(svptrue_b32(), values1);
        svfloat16_t res = svuzp1_f16(res0, res1);
        svst1_f16(pgHalf, out + i, res);
    }
    return svaddv_f32(svptrue_b32(), reduce);
}

template <typename scalarType, typename maskType, int64_t qSeqSize, int64_t kvSeqSize>
void FlashAttention(
    at::Tensor& output,
    const at::Tensor& logsumexp,
    const at::Tensor& q,
    const at::Tensor& k,
    const at::Tensor& v,
    double dropout,
    bool isCausal,
    std::optional<at::Tensor> attnMask,
    std::optional<double> scale)
{
    constexpr bool isReducedType = std::is_same_v<scalarType, float16_t>;

    float scalingFactor = CalculateScale(q, scale);

    int64_t batchSize = q.size(DIM0);
    int64_t qSize = q.size(DIM2);
    int64_t kvSize = v.size(DIM2);
    int64_t nHeads = q.size(DIM1);
    int64_t headSize = q.size(DIM3);

    bool hasAttnMask = attnMask.has_value() && attnMask.value().numel();
    if (hasAttnMask) {
        ReshapeAttnMask(attnMask.value(), batchSize, nHeads, qSize, kvSize);
    }

    // Strides
    int64_t mStrideB =
        (hasAttnMask && attnMask.value().size(DIM0) > 1)
        ? attnMask.value().stride(DIM0) : 0;
    int64_t mStrideH =
        (hasAttnMask && attnMask.value().size(DIM1) > 1)
        ? attnMask.value().stride(DIM1) : 0;
    int64_t mStrideM =
        (hasAttnMask && attnMask.value().size(DIM2) > 1)
        ? attnMask.value().stride(DIM2) : 0;
    int64_t mStrideN =
        (hasAttnMask && attnMask.value().size(DIM3) > 1)
        ? attnMask.value().stride(DIM3) : 0;

    int64_t qSplitSize = qSeqSize > qSize ? qSize : qSeqSize;
    int64_t kvSplitSize = kvSeqSize > kvSize ? kvSize : kvSeqSize;
    int64_t qSlice = (qSize + qSplitSize - 1) / qSplitSize;
    int64_t nThreads = omp_get_max_threads();

     // Allocate per thread temp buf (accumulate type)
    int64_t sizePerThread =
        qSplitSize * kvSplitSize +
        qSplitSize +
        qSplitSize +
        qSplitSize * headSize;

    at::Tensor buf = at::empty({nThreads, sizePerThread}, q.options().dtype(at::kFloat));
    at::Tensor bufReduced = at::empty(
        {nThreads,
        qSplitSize,
        isReducedType ? kvSplitSize : 0},
        q.options());

    at::Tensor query = q.transpose(DIM1, DIM2);
    at::Tensor key = k.transpose(DIM1, DIM2);
    at::Tensor value = v.transpose(DIM1, DIM2);

    // Strides
    int64_t qStrideB = query.stride(DIM0);
    int64_t qStrideM = query.stride(DIM1);
    int64_t qStrideH = query.stride(DIM2);
    int64_t kStrideB = key.stride(DIM0);
    int64_t kStrideN = key.stride(DIM1);
    int64_t kStrideH = key.stride(DIM2);
    int64_t vStrideB = value.stride(DIM0);
    int64_t vStrideN = value.stride(DIM1);
    int64_t vStrideH = value.stride(DIM2);
    int64_t oStrideB = output.stride(DIM0);
    int64_t oStrideM = output.stride(DIM1);
    int64_t oStrideH = output.stride(DIM2);

    const scalarType *qdata = TensorConstDataPtr<scalarType>(query);
    const scalarType *kdata = TensorConstDataPtr<scalarType>(key);
    const scalarType *vdata = TensorConstDataPtr<scalarType>(value);
    scalarType *outData = TensorDataPtr<scalarType>(output);

    maskType *maskData = hasAttnMask ? TensorDataPtr<maskType>(attnMask.value()) : nullptr;
    float *bufData = TensorDataPtr<float>(buf);
    scalarType *bufReducedData = isReducedType ? TensorDataPtr<scalarType>(bufReduced) : nullptr;

#pragma omp parallel for collapse(3)
    for (int64_t i = 0; i < batchSize; i++) {
        for (int64_t j = 0; j < nHeads; j++) {
            for (int64_t k = 0; k < qSlice; k++) {
                int ompIdx = omp_get_thread_num();
                float *bufPtr = bufData + ompIdx * sizePerThread;
                float *qkData = bufPtr;
                float *qkMaxData = qkData + qSplitSize * kvSplitSize;
                float *qkSumData = qkMaxData + qSplitSize;
                float *dstData = qkSumData + qSplitSize;
                scalarType *qkReducedData = isReducedType ?
                    bufReducedData + ompIdx * qSplitSize * kvSplitSize : nullptr;
                int64_t m = k * qSplitSize;
                int64_t qBlockSize = std::min(qSplitSize, qSize - m);
                FillData(qBlockSize, qkMaxData, static_cast<float>(-1e10));
                FillData(qBlockSize, qkSumData, static_cast<float>(0));
                int64_t numKeys = isCausal ? std::min(m + qBlockSize, kvSize) : kvSize;

                for (int64_t n = 0; n < numKeys; n += kvSplitSize) {
                    int64_t kvBlockSize = std::min(kvSplitSize, kvSize - n);
                    // Calculate scale * q @ k.T
#ifdef KPEX_920F
                    UniGemm(
                        false, true,
                        qBlockSize,
                        kvBlockSize,
                        headSize,
                        static_cast<float>(scalingFactor),
                        qdata + i * qStrideB + j * qStrideH +
                            m * qStrideM,
                        qStrideM,
                        kdata + i * kStrideB + j * kStrideH +
                            n * kStrideN,
                        kStrideN,
                        static_cast<float>(0),
                        qkData,
                        kvBlockSize, 1);
#else
                    BlasGemm(
                        CblasNoTrans, CblasTrans,
                        qBlockSize,
                        kvBlockSize,
                        headSize,
                        static_cast<float>(scalingFactor),
                        qdata + i * qStrideB + j * qStrideH +
                            m * qStrideM,
                        qStrideM,
                        kdata + i * kStrideB + j * kStrideH +
                            n * kStrideN,
                        kStrideN,
                        static_cast<float>(0),
                        qkData,
                        kvBlockSize);
#endif
                    if (isCausal && numKeys - n <= kvSplitSize) {
                        for (int64_t row = 0; row < qBlockSize; row++) {
                            int64_t last = m + row - n;
                            float* rowData = qkData + row * kvBlockSize;
                            FillData(
                                kvBlockSize - last - 1,
                                rowData + last + 1,
                                -std::numeric_limits<float>::infinity());
                        }
                    }

                    if (hasAttnMask) {
                        MatAddMat(
                            qBlockSize, kvBlockSize,
                            kvBlockSize, mStrideM, kvBlockSize,
                            qkData,
                            maskData + i * mStrideB + j * mStrideH +
                                m * mStrideM + (mStrideN == 0 ? 0 : n),
                            qkData, mStrideN > 0);
                    }

                    float tmpMax = 0;
                    float tmpSum = 0;
                    float tmpExp = 0;
                    for (int64_t row = 0; row < qBlockSize; row++) {
                        tmpMax = ReducedMax(
                            kvBlockSize,
                            qkData + row * kvBlockSize);
                        tmpMax = std::max(tmpMax, qkMaxData[row]);
                        if (tmpMax == -std::numeric_limits<float>::infinity()) {
                            // to avoid `nan = exp2f(-inf - (-inf))`
                            FillData(
                                kvBlockSize,
                                ConditionalDataPtr(qkData, qkReducedData) + row * kvBlockSize,
                                static_cast<scalarType>(0));
                        } else {
                            tmpSum = ExpReduceSumFusionKernel(
                                kvBlockSize,
                                qkData + row * kvBlockSize,
                                ConditionalDataPtr(qkData, qkReducedData) + row * kvBlockSize,
                                tmpMax);
                            // exp_tmp <- exp(max[row] - max)
                            tmpExp = std::exp(qkMaxData[row] - tmpMax);
                            // sum[row] <- sum + exp_tmp * sum[row]
                            qkSumData[row] = tmpSum + tmpExp * qkSumData[row];
                            // max[row] <- max
                            qkMaxData[row] = tmpMax;
                        }
                        if (n > 0) {
                            VecLinearFusionKernel(
                                headSize,
                                dstData + row * headSize,
                                tmpExp,
                                nullptr,
                                dstData + row * headSize);
                        }
                    }
#ifdef KPEX_920F
                    UniGemm(
                        false, false,
                        qBlockSize,
                        headSize,
                        kvBlockSize,
                        static_cast<float>(1),
                        ConditionalDataPtr(qkData, qkReducedData),
                        kvBlockSize,
                        vdata + i * vStrideB + j * vStrideH +
                            n * vStrideN,
                        vStrideN,
                        n == 0 ? static_cast<float>(0) : static_cast<float>(1),
                        dstData,
                        headSize, 1);
#else
                    BlasGemm(
                        CblasNoTrans, CblasNoTrans,
                        qBlockSize,
                        headSize,
                        kvBlockSize,
                        static_cast<float>(1),
                        ConditionalDataPtr(qkData, qkReducedData),
                        kvBlockSize,
                        vdata + i * vStrideB + j * vStrideH +
                            n * vStrideN,
                        vStrideN,
                        n == 0 ? static_cast<float>(0) : static_cast<float>(1),
                        dstData,
                        headSize);
#endif
                }
                for (int64_t row = 0; row < qBlockSize; row++) {
                    qkMaxData[row] = qkMaxData[row] == -std::numeric_limits<float>::infinity() ?
                        0 : qkMaxData[row];
                    qkSumData[row] = qkSumData[row] == 0 ? 1 : qkSumData[row];
                    float sumReciprocal = 1 / qkSumData[row];
                    VecLinearFusionKernel(
                        headSize,
                        dstData + row * headSize,
                        sumReciprocal,
                        nullptr,
                        outData + i * oStrideB + j * oStrideH + m * oStrideM + row * oStrideM);
                }
            }
        }
    }
}

#define FLASH_ATTENTION_KERNEL(TYPE1, TYPE2, SEQ1, SEQ2, ...)    \
    FlashAttention<TYPE1, TYPE2, SEQ1, SEQ2>(__VA_ARGS__)

void FlashAttentionKernelImpl(
    at::Tensor& output,
    const at::Tensor& logsumexp,
    const at::Tensor& query,
    const at::Tensor& key,
    const at::Tensor& value,
    double dropout,
    bool isCausal,
    std::optional<at::Tensor> attnMask,
    std::optional<double> scale)
{
    int64_t qSize = query.size(DIM2);
    KPEX_DISPATCH_FLOATING_AND_HALF(query.scalar_type(), scalarType, [&] {
        if (!attnMask.has_value()) {
            if (qSize >= FLASH_ATTN_Q_SIZE_THRESHOLD_1) {
                FLASH_ATTENTION_KERNEL(scalarType, scalarType, 128, 512,
                    output, logsumexp, query, key, value,
                    dropout, isCausal, attnMask, scale);
            } else if (qSize >= FLASH_ATTN_Q_SIZE_THRESHOLD_0) {
                FLASH_ATTENTION_KERNEL(scalarType, scalarType, 64, 512,
                    output, logsumexp, query, key, value,
                    dropout, isCausal, attnMask, scale);
            } else {
                FLASH_ATTENTION_KERNEL(scalarType, scalarType, 32, 512,
                    output, logsumexp, query, key, value,
                    dropout, isCausal, attnMask, scale);
            }
        } else {
            KPEX_DISPATCH_FLOATING_AND_HALF(attnMask.value().scalar_type(), maskType, [&] {
                if (qSize >= FLASH_ATTN_Q_SIZE_THRESHOLD_1) {
                    FLASH_ATTENTION_KERNEL(scalarType, maskType, 128, 512,
                        output, logsumexp, query, key, value,
                        dropout, isCausal, attnMask, scale);
                } else if (qSize >= FLASH_ATTN_Q_SIZE_THRESHOLD_0) {
                    FLASH_ATTENTION_KERNEL(scalarType, maskType, 64, 512,
                        output, logsumexp, query, key, value,
                        dropout, isCausal, attnMask, scale);
                } else {
                    FLASH_ATTENTION_KERNEL(scalarType, maskType, 32, 512,
                        output, logsumexp, query, key, value,
                        dropout, isCausal, attnMask, scale);
                }
            });
        }
    });
}

std::tuple<at::Tensor, at::Tensor> ScaledDotProductFlashAttention(
    const at::Tensor& query,
    const at::Tensor& key,
    const at::Tensor& value,
    double dropout,
    bool isCausal,
    const std::optional<at::Tensor>& attnMask,
    std::optional<double> scale)
{
    int64_t batchSize = query.size(DIM0);
    int64_t qSize = query.size(DIM2);
    int64_t nHeads = query.size(DIM1);

    at::Tensor output = at::empty_like(query, query.options()).transpose(DIM1, DIM2);
    at::Tensor logsumexp = at::empty({batchSize, nHeads, qSize},
        query.options().dtype(at::kFloat));

    FlashAttentionKernelImpl(output, logsumexp,
        query, key, value, dropout, isCausal, attnMask, scale);

    output = output.transpose(DIM1, DIM2);

    return std::make_tuple(std::move(output), std::move(logsumexp));
}

}