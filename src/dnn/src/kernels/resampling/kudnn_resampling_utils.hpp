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

#ifndef KUDNN_RESAMPLING_UTILS_HPP
#define KUDNN_RESAMPLING_UTILS_HPP

#include "kudnn_resampling_common_utils.hpp"

namespace KuDNN {
namespace Detail {
namespace Resampling {

template <typename IdxType>
struct ResamplingCoeffs {
    // For nearest algorithm.
    std::vector<IdxType> outidx2Inidx;
    // For linear algorithm.
    std::vector<IdxType> idx0;
    std::vector<IdxType> idx1;
    std::vector<float> wei0;
    std::vector<float> wei1;
    void NearestInit(IdxType len)
    {
        outidx2Inidx = std::vector<IdxType>(len);
    }
    void LinearInit(IdxType len)
    {
        idx0 = std::vector<IdxType>(len, -1);
        idx1 = std::vector<IdxType>(len, -1);
        wei0 = std::vector<float>(len, -1.0f);
        wei1 = std::vector<float>(len, -1.0f);
    }
};

template <typename IdxType>
inline void InitIdxs(IdxType len, IdxType start, IdxType *idxs)
{
    for (IdxType i = 0; i < len; i++) {
        idxs[i] = start + i;
    }
}

inline float LinearMap(IntType y, IntType ymax, IntType xmax)
{
    float yf = static_cast<float>(y);
    float ymaxf = static_cast<float>(ymax);
    float xmaxf = static_cast<float>(xmax);
    return ((yf + 0.5f) * xmaxf / ymaxf) - 0.5f;
}

inline svfloat32_t LinearMapVC(svbool_t pg, svint32_t y, int32_t ymax, int32_t xmax)
{
    // ((y + 0.5f) * x_max / y_max) - 0.5f
    svfloat32_t yf = svcvt_f32_s32_x(pg, y);
    svfloat32_t t1 = svadd_n_f32_x(pg, yf, 0.5f);
    svfloat32_t t2 = svmul_n_f32_x(pg, t1, static_cast<float>(xmax));
    svfloat32_t t3 = svdiv_n_f32_x(pg, t2, static_cast<float>(ymax));
    svfloat32_t t4 = svsub_n_f32_x(pg, t3, 0.5f);
    return t4;
}

// For nearest algorithm.
template <typename IdxType>
inline void Oidx2Iidx(IdxType len, const IdxType *oIdxs, IdxType *iIdxs, IdxType outDims, IdxType inDims);

template <>
inline void Oidx2Iidx<IntType>(IntType len, const IntType *oIdxs, IntType *iIdxs, IntType outDims, IntType inDims)
{
    for (IntType i = 0; i < len; i++) {
        float iIdxF = roundf(LinearMap(oIdxs[i], outDims, inDims));
        IntType iIdx = static_cast<IntType>(iIdxF);
        iIdx = std::min(iIdx, inDims - 1);
        iIdx = std::max(iIdx, 0L);
        iIdxs[i] = iIdx;
    }
}

template <>
inline void Oidx2Iidx<int32_t>(int32_t len, const int32_t *oIdxs, int32_t *iIdxs, int32_t outDims, int32_t inDims)
{
    for (int32_t i = 0; i < len; i += svcntw()) {
        svbool_t pg = svwhilelt_b32_s32(i, len);
        svint32_t odIdxs = svld1_s32(pg, oIdxs + i);

        svfloat32_t idIdxsF = svrinta_f32_x(pg, LinearMapVC(pg, odIdxs, outDims, inDims));
        svint32_t idIdxs = svcvt_s32_f32_x(pg, idIdxsF);
        idIdxs = svmin_n_s32_x(pg, idIdxs, inDims - 1);
        idIdxs = svmax_n_s32_x(pg, idIdxs, 0);

        svst1_s32(pg, iIdxs + i, idIdxs);
    }
}

// For linear algorithm.
template <typename IdxType>
inline void FillCoeff(IdxType len, IdxType startId, IdxType *idxs, IdxType ymax, IdxType xmax,
    ResamplingCoeffs<IdxType> &coeffs);

template <>
inline void FillCoeff<IntType>(IntType len, IntType startId, IntType *idxs, IntType ymax, IntType xmax,
    ResamplingCoeffs<IntType> &coeffs)
{
    for (IntType i = startId; i < len + startId; i++) {
        float mappedIdx = LinearMap(idxs[i], ymax, xmax);
        float fIdx0 = floorf(mappedIdx);
        float fIdx1 = ceilf(mappedIdx);
        coeffs.idx0[i] = std::max(0L, static_cast<IntType>(fIdx0));
        coeffs.idx1[i] = std::min(xmax - 1, static_cast<IntType>(fIdx1));

        coeffs.wei1[i] = std::abs(mappedIdx - static_cast<float>(coeffs.idx0[i]));
        coeffs.wei0[i] = 1.f - coeffs.wei1[i];
    }
}

template <>
inline void FillCoeff<int32_t>(int32_t len, int32_t startId, int32_t *idxs, int32_t ymax, int32_t xmax,
    ResamplingCoeffs<int32_t> &coeffs)
{
    for (int32_t i = startId; i < len + startId; i += svcntw()) {
        svbool_t pg = svwhilelt_b32_s32(i, len + startId);
        svint32_t y = svld1_s32(pg, idxs + i);
        svfloat32_t mappedIdxs = LinearMapVC(pg, y, ymax, xmax);

        svfloat32_t fIdxs0 = svrintm_f32_z(pg, mappedIdxs);
        svfloat32_t fIdxs1 = svrintp_f32_z(pg, mappedIdxs);
        svint32_t idxs0 = svmax_n_s32_x(pg, svcvt_s32_f32_x(pg, fIdxs0), 0);
        svint32_t idxs1 = svmin_n_s32_x(pg, svcvt_s32_f32_x(pg, fIdxs1), xmax - 1);

        svfloat32_t weis1 = svabs_f32_x(pg, svsub_f32_x(pg, mappedIdxs, svcvt_f32_s32_x(pg, idxs0)));
        svfloat32_t weis0 = svsubr_n_f32_x(pg, weis1, 1.f);

        svst1_s32(pg, &coeffs.idx0[0] + i, idxs0);
        svst1_s32(pg, &coeffs.idx1[0] + i, idxs1);
        svst1_f32(pg, &coeffs.wei0[0] + i, weis0);
        svst1_f32(pg, &coeffs.wei1[0] + i, weis1);
    }
}

inline void LinearStksInit(SizeType idx,
    std::vector<std::vector<IntType>> &offsetsStk, std::vector<std::vector<float>> &weightsStk)
{
    for (SizeType i = 0; i < idx; i++) {
        offsetsStk.emplace_back(std::vector<IntType>(IDX_1 << i, NUM_0));
        weightsStk.emplace_back(std::vector<float>(IDX_1 << i, static_cast<float>(NUM_1)));
    }
}

template <typename IdxType>
inline void Linearize(const ResamplingCoeffs<IdxType> &coeffs, IntType coeffsIdx, IntType strideI,
    std::vector<std::vector<IntType>> &offsetsStk, std::vector<std::vector<float>> &weightsStk, SizeType stkDstIdx)
{
    IntType offset0 = coeffs.idx0[coeffsIdx] * strideI;
    IntType offset1 = coeffs.idx1[coeffsIdx] * strideI;
    float w0 = coeffs.wei0[coeffsIdx];
    float w1 = coeffs.wei1[coeffsIdx];

    SizeType stkSrcIdx = stkDstIdx - 1;
    SizeType srcSize = offsetsStk[stkSrcIdx].size();
    for (SizeType i = 0; i < srcSize; i++) {
        offsetsStk[stkDstIdx][i] = offsetsStk[stkSrcIdx][i] + offset0;
        offsetsStk[stkDstIdx][i + srcSize] = offsetsStk[stkSrcIdx][i] + offset1;
        weightsStk[stkDstIdx][i] = weightsStk[stkSrcIdx][i] * w0;
        weightsStk[stkDstIdx][i + srcSize] = weightsStk[stkSrcIdx][i] * w1;
    }
}

// For unordered_map with std::function.
struct VCKerParas {
    Element::TypeT dataType;
    SizeType ndims;
    ResamplingAlg alg;
};

struct VCKerParasHash {
    SizeType operator()(const VCKerParas &paras) const
    {
        return static_cast<SizeType>(paras.dataType) * NUM_10 * NUM_10 +
               paras.ndims * NUM_10 + static_cast<SizeType>(paras.alg);
    }
};

struct VCKerParasEqual {
    bool operator()(const VCKerParas &paras0, const VCKerParas &paras1) const
    {
        return paras0.dataType == paras1.dataType && paras0.ndims == paras1.ndims && paras0.alg == paras1.alg;
    }
};

struct NormalKerParas {
    Element::TypeT srcType;
    Element::TypeT dstType;
    SizeType ndims;
    ResamplingAlg alg;
};

struct NormalKerParasHash {
    SizeType operator()(const NormalKerParas &paras) const
    {
        return static_cast<SizeType>(paras.srcType) * NUM_10 * NUM_10 * NUM_10 +
               static_cast<SizeType>(paras.dstType) * NUM_10 * NUM_10 +
               paras.ndims * NUM_10 + static_cast<SizeType>(paras.alg);
    }
};

struct NormalKerParasEqual {
    bool operator()(const NormalKerParas &paras0, const NormalKerParas &paras1) const
    {
        return paras0.srcType == paras1.srcType && paras0.dstType == paras1.dstType &&
               paras0.ndims == paras1.ndims && paras0.alg == paras1.alg;
    }
};

} // Resampling
} // Detail
} // KuDNN

#endif //  KUDNN_RESAMPLING_UTILS_HPP
