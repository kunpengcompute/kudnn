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
#include "attention.h"
#include "utils/dispatch.h"
#include "utils/blas_wrapper.h"
#include "utils/fast_exp.h"
#include "utils/fused_ops.h"
#include "utils/scalar_type.h"
#include "utils/acle_utils.h"
#include "utils/vla.h"

enum {
    DIM0 = 0,
    DIM1 = 1,
    DIM2 = 2,
    DIM3 = 3,
};

enum {
    ATTN_Q_SIZE_THRESHOLD_0 = 512,
    ATTN_Q_SIZE_THRESHOLD_1 = 256,
    ATTN_Q_SIZE_THRESHOLD_2 = 128,
    ATTN_Q_SIZE_THRESHOLD_3 = 64
};

#define ATTN_MAX_DIM_SIZE       4
#define KERNEL_LOOP_UNFOLD      8

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
    if (mask.dim() == ATTN_MAX_DIM_SIZE) {
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
static std::pair<float, float> SoftMaxFusionKernel(float *in, bfloat16_t *out, int len)
{
    const int64_t vlf = svcntw();
    const int64_t vlh = svcnth();
    svfloat32_t vecMax = svdup_f32(-1e9);
    for (int64_t i = 0; i < len; i += vlf) {
        svbool_t pg = svwhilelt_b32_s64(i, len);
        svfloat32_t values = svld1(pg, in + i);
        vecMax = svmax_m(pg, vecMax, values);
    }
    float scaMax = svmaxv(svptrue_b32(), vecMax);
    svfloat32_t sum = svdup_f32(0.f);
    for (int64_t i = 0; i < len; i += vlf) {
        svbool_t pg = svwhilelt_b32_s64(i, len);
        svfloat32_t values = svld1(pg, in + i);
        svfloat32_t exp = FastExp(pg, svsub_x(pg, values, scaMax));
        sum = svadd_m(pg, sum, exp);
        svst1_f32(pg, in + i, exp);
    }
    float expSum = svaddv(svptrue_b32(), sum);
    svfloat32_t sumInv = svdup_f32(1.0 / expSum);
    for (int64_t i = 0; i < len; i += vlh) {
        svbool_t pgFloat0 = svwhilelt_b32_s64(i, len);
        svbool_t pgFloat1 = svwhilelt_b32_s64(i + vlf, len);
        svbool_t pgHalf = svwhilelt_b16_s64(i, len);
        svfloat32_t values0 = svld1_f32(pgFloat0, in + i);
        svfloat32_t values1 = svld1_f32(pgFloat1, in + i + vlf);
        values0 = svmul_f32_x(pgFloat0, values0, sumInv);
        values1 = svmul_f32_x(pgFloat1, values1, sumInv);
        svbfloat16_t res0 = svcvt_bf16_f32_x(svptrue_b32(), values0);
        svbfloat16_t res1 = svcvt_bf16_f32_x(svptrue_b32(), values1);
        svbfloat16_t res = svuzp1_bf16(res0, res1);
        svst1_bf16(pgHalf, out + i, res);
    }
    return std::make_pair(scaMax, expSum);
}
static std::pair<float, float> SoftMaxFusionKernel(float *in, float16_t *out, int len)
{
    const int64_t vlf = svcntw();
    const int64_t vlh = svcnth();
    svfloat32_t vecMax = svdup_f32(-1e9);
    for (int64_t i = 0; i < len; i += vlf) {
        svbool_t pg = svwhilelt_b32_s64(i, len);
        svfloat32_t values = svld1(pg, in + i);
        vecMax = svmax_m(pg, vecMax, values);
    }
    float scaMax = svmaxv(svptrue_b32(), vecMax);
    svfloat32_t sum = svdup_f32(0.f);
    for (int64_t i = 0; i < len; i += vlf) {
        svbool_t pg = svwhilelt_b32_s64(i, len);
        svfloat32_t values = svld1(pg, in + i);
        svfloat32_t exp = FastExp(pg, svsub_x(pg, values, scaMax));
        sum = svadd_m(pg, sum, exp);
        svst1_f32(pg, in + i, exp);
    }
    float expSum = svaddv(svptrue_b32(), sum);
    svfloat32_t sumInv = svdup_f32(1.0 / expSum);
    for (int64_t i = 0; i < len; i += vlh) {
        svbool_t pgFloat0 = svwhilelt_b32_s64(i, len);
        svbool_t pgFloat1 = svwhilelt_b32_s64(i + vlf, len);
        svbool_t pgHalf = svwhilelt_b16_s64(i, len);
        svfloat32_t values0 = svld1_f32(pgFloat0, in + i);
        svfloat32_t values1 = svld1_f32(pgFloat1, in + i + vlf);
        values0 = svmul_f32_x(pgFloat0, values0, sumInv);
        values1 = svmul_f32_x(pgFloat1, values1, sumInv);
        svfloat16_t res0 = svcvt_f16_f32_x(svptrue_b32(), values0);
        svfloat16_t res1 = svcvt_f16_f32_x(svptrue_b32(), values1);
        svfloat16_t res = svuzp1_f16(res0, res1);
        svst1_f16(pgHalf, out + i, res);
    }
    return std::make_pair(scaMax, expSum);
}
static std::pair<float, float> SoftMaxFusionKernel(float *in, float *out, int len)
{
    int i = 0;
    const int loopExpf = svcntw();
    svbool_t ptrue = svptrue_b32();
    simdfloat32_t max[KERNEL_LOOP_UNFOLD];
    LoopUnFoldFunc<int, KERNEL_LOOP_UNFOLD>([&](int idx) {
        max[idx] = svdup_f32(-1e9);
    });
    for (i = 0; i < len - loopExpf * KERNEL_LOOP_UNFOLD; i += loopExpf * KERNEL_LOOP_UNFOLD) {
        simdfloat32_t x[KERNEL_LOOP_UNFOLD];
        LoopUnFoldFunc<int, KERNEL_LOOP_UNFOLD>([&](int idx) {
            int groupOffset = i + idx * loopExpf;
            x[idx] = svld1_f32(ptrue, in + groupOffset);
        });
        LoopUnFoldFunc<int, KERNEL_LOOP_UNFOLD>([&](int idx) {
            max[idx] = svmax_f32_m(ptrue, max[idx], x[idx]);
        });
    }
    for (; i < len; i += loopExpf) {
        svbool_t pg = svwhilelt_b32(i, len);
        svfloat32_t values = svld1(pg, in + i);
        max[0] = svmax_f32_m(pg, max[0], values);
    }
    for (int j = 1; j < KERNEL_LOOP_UNFOLD; j++) {
        max[0] = svmax_f32_m(ptrue, max[0], max[j]);
    }
    float scaMax = svmaxv_f32(ptrue, max[0]);

    simdfloat32_t sum[KERNEL_LOOP_UNFOLD];
    LoopUnFoldFunc<int, KERNEL_LOOP_UNFOLD>([&](int idx) {
        sum[idx] = svdup_f32(0.0);
    });
    for (i = 0; i < len - loopExpf * KERNEL_LOOP_UNFOLD; i += loopExpf * KERNEL_LOOP_UNFOLD) {
        simdfloat32_t exp[KERNEL_LOOP_UNFOLD];
        simdfloat32_t x[KERNEL_LOOP_UNFOLD];
        LoopUnFoldFunc<int, KERNEL_LOOP_UNFOLD>([&](int idx) {
            int groupOffset = i + idx * loopExpf;
            x[idx] = svld1_f32(ptrue, in + groupOffset);
        });
        LoopUnFoldFunc<int, KERNEL_LOOP_UNFOLD>([&](int idx) {
            x[idx] = svsub_n_f32_x(ptrue, x[idx], scaMax);
        });
        LoopUnFoldFunc<int, KERNEL_LOOP_UNFOLD>([&](int idx) {
            exp[idx] = FastExp(ptrue, x[idx]);
        });
        LoopUnFoldFunc<int, KERNEL_LOOP_UNFOLD>([&](int idx) {
            sum[idx] = svadd_f32_m(ptrue, sum[idx], exp[idx]);
        });
        LoopUnFoldFunc<int, KERNEL_LOOP_UNFOLD>([&](int idx) {
            int groupOffset = i + idx * loopExpf;
            svst1_f32(ptrue, in + groupOffset, exp[idx]);
        });
    }
    for (; i < len; i += loopExpf) {
        svbool_t pg = svwhilelt_b32(i, len);
        svfloat32_t values = svld1(pg, in + i);
        svfloat32_t exp = FastExp(pg, svsub_x(pg, values, scaMax));
        sum[0] = svadd_m(pg, sum[0], exp);
        svst1_f32(pg, in + i, exp);
    }
    for (int j = 1; j < KERNEL_LOOP_UNFOLD; j++) {
        sum[0] = svadd_m(ptrue, sum[0], sum[j]);
    }
    float expSum = svaddv(ptrue, sum[0]);

    svfloat32_t sumInv = svdup_f32(1 / expSum);
    for (i = 0; i < len - loopExpf * KERNEL_LOOP_UNFOLD; i += loopExpf * KERNEL_LOOP_UNFOLD) {
        simdfloat32_t x[KERNEL_LOOP_UNFOLD];
        LoopUnFoldFunc<int, KERNEL_LOOP_UNFOLD>([&](int idx) {
            int groupOffset = i + idx * loopExpf;
            x[idx] = svld1_f32(ptrue, in + groupOffset);
        });
        LoopUnFoldFunc<int, KERNEL_LOOP_UNFOLD>([&](int idx) {
            x[idx] = svmul_f32_x(ptrue, x[idx], sumInv);
        });
        LoopUnFoldFunc<int, KERNEL_LOOP_UNFOLD>([&](int idx) {
            int groupOffset = i + idx * loopExpf;
            svst1_f32(ptrue, out + groupOffset, x[idx]);
        });
    }

    for (; i < len; i += loopExpf) {
        svbool_t pg = svwhilelt_b32(i, len);
        svfloat32_t x = svld1(pg, in + i);
        x = svmul_x(pg, x, sumInv);
        svst1_f32(pg, (out + i), x);
    }

    return std::make_pair(scaMax, expSum);
}

// sollya: fpminimax(log(1 + x), [|1,2,3,4,5|], [|24...|], [-0x1.555554p-2; 0x1.555558p-2], relative)
#define SVE_VSLN_EP_C1 svdup_n_f32(0x1.000064p0f)
#define SVE_VSLN_EP_C2 svdup_n_f32(-0x1.ff41bcp-2f)
#define SVE_VSLN_EP_C3 svdup_n_f32(0x1.54408cp-2f)
#define SVE_VSLN_EP_C4 svdup_n_f32(-0x1.19f964p-2f)
#define SVE_VSLN_EP_C5 svdup_n_f32(0x1.ce3ae4p-3f)

static inline svfloat32_t VsLnEpSvePoly(svbool_t pg, svfloat32_t r, svfloat32_t resHi)
{
    svfloat32_t taylor = svmad_f32_x(pg, SVE_VSLN_EP_C5, r, SVE_VSLN_EP_C4);
    taylor = svmad_f32_x(pg, taylor, r, SVE_VSLN_EP_C3);
    taylor = svmad_f32_x(pg, taylor, r, SVE_VSLN_EP_C2);
    taylor = svmad_f32_x(pg, taylor, r, SVE_VSLN_EP_C1);
    svfloat32_t res = svmad_f32_x(pg, taylor, r, resHi);
    return res;
}
#define VSLN_OFF (0x3f2aaaabU)
#define FLOAT_MANTISSA_MASK         0x7fffffU
#define LN2F                        (0x1.62e43p-1f)
#define SVE_FLOAT_ONE               svdup_n_f32(1.0f)
#define SVE_FLOAT_LN2               svdup_n_f32(LN2F)
#define SVE_VSLN_BASE               svdup_n_s32((int)VSLN_OFF) // 2/3
#define SVE_VSLN_MANTISSA_MASK      svdup_n_s32((int)FLOAT_MANTISSA_MASK)
#define SVE_FLOAT_MANTISSA_BIT_LEN  svdup_n_u32(FLOAT_MANTISSA_BIT_LEN)
#define FLOAT_MANTISSA_BIT_LEN      23
static inline svfloat32_t VsLnEpSve(svbool_t pg, svfloat32_t x)
{
    // valid interval : x in [0x1p-126, 0x1.fffffep127]
    const svint32_t tmp = svsub_s32_x(pg, svreinterpret_s32_f32(x), SVE_VSLN_BASE);
    const svfloat32_t n = svcvt_f32_s32_x(pg, svasr_s32_x(pg, tmp, SVE_FLOAT_MANTISSA_BIT_LEN));

    const svint32_t tmp0 = svand_s32_x(pg, tmp, SVE_VSLN_MANTISSA_MASK);
    const svfloat32_t x0 = svreinterpret_f32_s32(svadd_s32_x(pg, tmp0, SVE_VSLN_BASE));

    const svfloat32_t r = svsub_f32_x(pg, x0, SVE_FLOAT_ONE);
    const svfloat32_t resHi = svmul_f32_x(pg, n, SVE_FLOAT_LN2);
    const svfloat32_t res = VsLnEpSvePoly(pg, r, resHi);

    return res;
}
static void SaveLogExpSumFusionKernel(float* dst, float* max, float* sum, int64_t size)
{
    const int64_t loopExpf = svcntw();
    svbool_t ptrue = svptrue_b32();
    int64_t i = 0;

    for (i = 0; i < size - loopExpf * KERNEL_LOOP_UNFOLD; i += loopExpf * KERNEL_LOOP_UNFOLD) {
        simdfloat32_t maxValues[KERNEL_LOOP_UNFOLD];
        simdfloat32_t sumValues[KERNEL_LOOP_UNFOLD];
        simdfloat32_t lnValues[KERNEL_LOOP_UNFOLD];
        simdfloat32_t res[KERNEL_LOOP_UNFOLD];
        LoopUnFoldFunc<int, KERNEL_LOOP_UNFOLD>([&](int idx) {
            int groupOffset = i + idx * loopExpf;
            maxValues[idx] = svld1_f32(ptrue, max + groupOffset);
        });
        LoopUnFoldFunc<int, KERNEL_LOOP_UNFOLD>([&](int idx) {
            int groupOffset = i + idx * loopExpf;
            sumValues[idx] = svld1_f32(ptrue, sum + groupOffset);
        });
        LoopUnFoldFunc<int, KERNEL_LOOP_UNFOLD>([&](int idx) {
            lnValues[idx] = VsLnEpSve(ptrue, sumValues[idx]);
        });
        LoopUnFoldFunc<int, KERNEL_LOOP_UNFOLD>([&](int idx) {
            res[idx] = svadd_f32_x(ptrue, maxValues[idx], lnValues[idx]);
        });
        LoopUnFoldFunc<int, KERNEL_LOOP_UNFOLD>([&](int idx) {
            int groupOffset = i + idx * loopExpf;
            svst1_f32(ptrue, dst + groupOffset, res[idx]);
        });
    }

    for (; i < size; i += loopExpf) {
        svbool_t pg = svwhilelt_b32(i, size);
        svfloat32_t maxValues = svld1_f32(pg, max + i);
        svfloat32_t sumValues = svld1_f32(pg, sum + i);
        svfloat32_t lnValues = VsLnEpSve(pg, sumValues);
        svfloat32_t res = svadd_f32_x(pg, maxValues, lnValues);
        svst1_f32(pg, dst + i, res);
    }
}

template <typename scalarType, typename maskType, int64_t qSeqSize>
void Attention(
    int64_t kvSeqSize,
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
    constexpr bool isReducedType = (std::is_same_v<scalarType, float16_t> || std::is_same_v<scalarType, bfloat16_t>);

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
    int64_t qSlice = (qSize + qSplitSize - 1) / qSplitSize;
    int64_t nThreads = omp_get_max_threads();

     // Allocate per thread temp buf (accumulate type)
    int64_t sizePerThread =
    qSplitSize * kvSize + /* qkData */
    qSplitSize + /* qkMaxData */
    qSplitSize; /* qkSumData */

    at::Tensor buf = at::empty({nThreads, sizePerThread}, q.options().dtype(at::kFloat));
    at::Tensor bufReduced = at::empty(
        {nThreads,
        qSplitSize,
        isReducedType ? kvSize : 0},
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
    auto lseData = GetVLAPtr<float>(logsumexp, {nHeads, qSize});
    float *bufData = TensorDataPtr<float>(buf);
    scalarType *bufReducedData = isReducedType ? TensorDataPtr<scalarType>(bufReduced) : nullptr;

#pragma omp parallel
{
    int ompIdx = omp_get_thread_num();
    float *bufPtr = bufData + ompIdx * sizePerThread;
    float *qkData = bufPtr;
    float *qkMaxData = qkData + qSplitSize * kvSize;
    float *qkSumData = qkMaxData + qSplitSize;
    scalarType *qkReducedData = isReducedType ? bufReducedData + ompIdx * qSplitSize * kvSize : nullptr;
#pragma omp for collapse(3)
    for (int64_t i = 0; i < batchSize; i++) {
        for (int64_t j = 0; j < nHeads; j++) {
            for (int64_t k = 0; k < qSlice; k++) {
                int64_t m = k * qSplitSize;
                int64_t qBlockSize = std::min(qSplitSize, qSize - m);
                int64_t numKeys = isCausal ? std::min(m + qBlockSize, kvSize) : kvSize;
                // Calculate scale * q @ k.T
                BlasGemm(
                    CblasNoTrans, CblasTrans,
                    qBlockSize,
                    numKeys,
                    headSize,
                    static_cast<float>(scalingFactor),
                    qdata + i * qStrideB + j * qStrideH +
                        m * qStrideM,
                    qStrideM,
                    kdata + i * kStrideB + j * kStrideH,
                    kStrideN,
                    static_cast<float>(0),
                    qkData,
                    numKeys);
                if (isCausal) {
                    for (int64_t row = 0; row < qBlockSize; row++) {
                        int64_t last = m + row;
                        float* rowData = qkData + row * numKeys;
                        FillData(
                            numKeys - last - 1,
                            rowData + last + 1,
                            -std::numeric_limits<float>::infinity());
                    }
                }
                if (hasAttnMask) {
                    MatAddMat(
                        qBlockSize, numKeys,
                        numKeys, mStrideM, numKeys,
                        qkData,
                        maskData + i * mStrideB + j * mStrideH +
                            m * mStrideM,
                        qkData, mStrideN > 0);
                }
                for (int64_t row = 0; row < qBlockSize; row++) {
                    auto [max, sum] = SoftMaxFusionKernel(qkData + row * numKeys, ConditionalDataPtr(qkData,
                                                          qkReducedData) + row * numKeys, numKeys);
                    qkMaxData[row] = max;
                    qkSumData[row] = sum;
                }
                BlasGemm(
                    CblasNoTrans, CblasNoTrans,
                    qBlockSize,
                    headSize,
                    numKeys,
                    static_cast<float>(1),
                    ConditionalDataPtr(qkData, qkReducedData),
                    numKeys,
                    vdata + i * vStrideB + j * vStrideH,
                    vStrideN,
                    static_cast<float>(0),
                    outData + i * oStrideB + j * oStrideH + m * oStrideM,
                    oStrideM);
                SaveLogExpSumFusionKernel(&lseData[i][j][m], &qkMaxData[0], &qkSumData[0], qBlockSize);
            }
        }
    }
}
}

#define ATTENTION_KERNEL(TYPE1, TYPE2, SEQ1, ...)    \
    Attention<TYPE1, TYPE2, SEQ1>(__VA_ARGS__)

void AttentionKernelImpl(
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
    // L2 cache size(920F:512KB, 920B:1M)  / typeSize
    //              > qSplitSize * kvSize +  qSplitSize * headSize + kvSize * headSize + 2 * qSplitSize
    int64_t kvSize = value.size(DIM2);
    int64_t headSize = query.size(DIM3);
#ifdef KPEX_920F
    const int64_t cacheSize = 524288; // 512 K = 512 * 1024
#else
    const int64_t cacheSize = 1048576; // 1M = 1024 * 1024
#endif
    int64_t typeSize = (query.scalar_type() == at::kHalf || query.scalar_type() == at::kBFloat16) ? 2 : 4;
    int64_t qSeqSizeMax = (cacheSize / typeSize - kvSize * headSize) / (kvSize + headSize + 2);
    KPEX_DISPATCH_FLOATING_AND_HALF_AND_BF16(query.scalar_type(), scalarType, [&] {
        if (!attnMask.has_value()) {
            if (qSeqSizeMax >= ATTN_Q_SIZE_THRESHOLD_0) {
                ATTENTION_KERNEL(scalarType, scalarType, ATTN_Q_SIZE_THRESHOLD_0,
                    kvSize,
                    output, logsumexp, query, key, value,
                    dropout, isCausal, attnMask, scale);
            } else if (qSeqSizeMax >= ATTN_Q_SIZE_THRESHOLD_1) {
                ATTENTION_KERNEL(scalarType, scalarType, ATTN_Q_SIZE_THRESHOLD_1,
                    kvSize,
                    output, logsumexp, query, key, value,
                    dropout, isCausal, attnMask, scale);
            } else if (qSeqSizeMax >= ATTN_Q_SIZE_THRESHOLD_2) {
                ATTENTION_KERNEL(scalarType, scalarType, ATTN_Q_SIZE_THRESHOLD_2,
                    kvSize,
                    output, logsumexp, query, key, value,
                    dropout, isCausal, attnMask, scale);
            } else {
                ATTENTION_KERNEL(scalarType, scalarType, ATTN_Q_SIZE_THRESHOLD_3,
                    kvSize,
                    output, logsumexp, query, key, value,
                    dropout, isCausal, attnMask, scale);
            }
        } else {
            KPEX_DISPATCH_FLOATING_AND_HALF_AND_BF16(attnMask.value().scalar_type(), maskType, [&] {
                if (qSeqSizeMax >= ATTN_Q_SIZE_THRESHOLD_0) {
                    ATTENTION_KERNEL(scalarType, maskType, ATTN_Q_SIZE_THRESHOLD_0,
                        kvSize,
                        output, logsumexp, query, key, value,
                        dropout, isCausal, attnMask, scale);
                } else if (qSeqSizeMax >= ATTN_Q_SIZE_THRESHOLD_1) {
                    ATTENTION_KERNEL(scalarType, maskType, ATTN_Q_SIZE_THRESHOLD_1,
                        kvSize,
                        output, logsumexp, query, key, value,
                        dropout, isCausal, attnMask, scale);
                } else if (qSeqSizeMax >= ATTN_Q_SIZE_THRESHOLD_2) {
                    ATTENTION_KERNEL(scalarType, maskType, ATTN_Q_SIZE_THRESHOLD_2,
                        kvSize,
                        output, logsumexp, query, key, value,
                        dropout, isCausal, attnMask, scale);
                } else {
                    ATTENTION_KERNEL(scalarType, maskType, ATTN_Q_SIZE_THRESHOLD_3,
                        kvSize,
                        output, logsumexp, query, key, value,
                        dropout, isCausal, attnMask, scale);
                }
            });
        }
    });
}

std::tuple<at::Tensor, at::Tensor> ScaledDotProductAttention(
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
    AttentionKernelImpl(output, logsumexp,
        query, key, value, dropout, isCausal, attnMask, scale);
    output = output.transpose(DIM1, DIM2);
    return std::make_tuple(std::move(output), std::move(logsumexp));
}

}