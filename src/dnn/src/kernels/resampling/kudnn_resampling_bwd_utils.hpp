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

#ifndef KUDNN_RESAMPLING_BWD_UTILS_HPP
#define KUDNN_RESAMPLING_BWD_UTILS_HPP

#include "kudnn_resampling_common_utils.hpp"

namespace KuDNN {
namespace Detail {
namespace Resampling {

struct NearestData {
    std::vector<IntType> leftIndices;
    std::vector<IntType> rightIndices;
};

struct LinearData {
    std::vector<std::vector<IntType>> dStartIndices;
    std::vector<std::vector<IntType>> hStartIndices;
    std::vector<std::vector<IntType>> wStartIndices;
    std::vector<std::vector<IntType>> dEndIndices;
    std::vector<std::vector<IntType>> hEndIndices;
    std::vector<std::vector<IntType>> wEndIndices;
    std::vector<std::vector<float>> dWeights;
    std::vector<std::vector<float>> hWeights;
    std::vector<std::vector<float>> wWeights;
};

inline IntType CeilIdx(float xF)
{
    if (xF < 0.0) {
        return static_cast<IntType>(0);
    }
    IntType x = static_cast<IntType>(xF);
    bool xIsExact = static_cast<float>(x) == xF;
    return xIsExact ? x : x + 1;
}

inline void InitIndices(std::vector<float> &indices, IntType d, IntType h, IntType w)
{
    for (auto i = 0; i < d; i++) {
        indices.emplace_back(i);
    }
    for (auto i = 0; i < h; i++) {
        indices.emplace_back(i);
    }
    for (auto i = 0; i < w; i++) {
        indices.emplace_back(i);
    }
}

inline svfloat32_t LinearStartVC(svbool_t pg, svfloat32_t y, IntType ymax, IntType xmax, int f)
{
    svfloat32_t t1;
    if (f == 0) {
        // start0 : ((y + 0.5f) * xmax / ymax) - 0.5f
        t1 = svadd_n_f32_m(pg, y, 0.5f);
    } else {
        // start1 : ((y - 0.5f) * xmax / ymax) - 0.5f
        t1 = svsub_n_f32_m(pg, y, 0.5f);
    }
    svfloat32_t t2 = svmul_n_f32_m(pg, t1, static_cast<float>(xmax));
    svfloat32_t t3 = svdiv_n_f32_m(pg, t2, static_cast<float>(ymax));
    svfloat32_t t4 = svsub_n_f32_m(pg, t3, 0.5f);
    return t4;
}

inline svfloat32_t LinearEndVC(svbool_t pg, svfloat32_t y, IntType ymax, IntType xmax, int f)
{
    svfloat32_t t1;
    if (f == 0) {
        // end0 : ((y + 1.5f) * xmax / ymax) - 0.5f
        t1 = svadd_n_f32_m(pg, y, 1.5f);
    } else {
        // end1 : ((y + 0.5f) * xmax / ymax) - 0.5f
        t1 = svadd_n_f32_m(pg, y, 0.5f);
    }
    svfloat32_t t2 = svmul_n_f32_m(pg, t1, static_cast<float>(xmax));
    svfloat32_t t3 = svdiv_n_f32_m(pg, t2, static_cast<float>(ymax));
    svfloat32_t t4 = svsub_n_f32_m(pg, t3, 0.5f);
    return t4;
}

inline void FillLinearIndicesVC(IntType width, const float *srcIndices,
    std::vector<std::vector<float>> &dstStartIndices, std::vector<std::vector<float>> &dstEndIndices,
    IntType ymax, IntType xmax)
{
    for (IntType i = 0; i < width; i += svcntw()) {
        svbool_t pg = svwhilelt_b32_s64(i, width);
        svfloat32_t y = svld1_f32(pg, srcIndices + i);
        svfloat32_t indicesStart0  = LinearStartVC(pg, y, ymax, xmax, 0);
        svfloat32_t indicesStart1 = LinearStartVC(pg, y, ymax, xmax, 1);
        svfloat32_t indicesEnd0 = LinearEndVC(pg, y, ymax, xmax, 0);
        svfloat32_t indicesEnd1 = LinearEndVC(pg, y, ymax, xmax, 1);
        svst1_f32(pg, &dstStartIndices[0][i], indicesStart0);
        svst1_f32(pg, &dstStartIndices[1][i], indicesStart1);
        svst1_f32(pg, &dstEndIndices[0][i], indicesEnd0);
        svst1_f32(pg, &dstEndIndices[1][i], indicesEnd1);
    }
}

inline void FillLinearWeightsVC(IntType width, const float *srcIndices, float* dstWeights0,
    float* dstWeights1, IntType ymax, IntType xmax)
{
    for (IntType i = 0; i < width; i += svcntw()) {
        svbool_t pg = svwhilelt_b32_s64(i, width);
        svfloat32_t y = svld1_f32(pg, srcIndices + i);
        // ((y + 0.5f) * xmax / ymax) - 0.5f
        svfloat32_t t1 = svadd_n_f32_m(pg, y, 0.5f);
        svfloat32_t t2 = svmul_n_f32_m(pg, t1, static_cast<float>(xmax));
        svfloat32_t t3 = svdiv_n_f32_m(pg, t2, static_cast<float>(ymax));
        svfloat32_t weightsFloat = svsub_n_f32_m(pg, t3, 0.5f);
        // w1 = abs(w - (float)( (int)w) )
        svint32_t t = svcvt_s32_f32_x(pg, weightsFloat);
        t = svmax_n_s32_m(pg, t, S32_ZERO);
        svfloat32_t weightsInt = svcvt_f32_s32_x(pg, t);
        svfloat32_t weights1 = svsub_f32_m(pg, weightsFloat, weightsInt);
        weights1 = svabs_f32_x(pg, weights1);
        svst1_f32(pg, dstWeights1 + i, weights1);
        // w0 = 1.0 - w1
        svfloat32_t weights0 = svsub_f32_m(pg, svdup_n_f32(1.0f), weights1);
        svst1_f32(pg, dstWeights0 + i, weights0);
    }
}

inline IntType GetLinearStart(float xF, IntType x, size_t f)
{
    if (f == 0) {
        return x == 0 ? 0 : CeilIdx(xF);
    }
    return xF < 0 ? 0 : static_cast<IntType>(xF) + 1;
}

inline IntType GetLinearEnd(float xF, IntType x, IntType xmax, IntType ymax, size_t f)
{
    if (f == 0) {
        return std::min(CeilIdx(xF), ymax);
    }
    return x == xmax - 1 ? ymax : (std::min(xF < 0 ? 0 : static_cast<IntType>(xF) + 1, ymax));
}

inline void FillLeftNearestIndicesVC(IntType width, const float *srcIndices, float *dstIndices,
    IntType ymax, IntType xmax)
{
    for (IntType i = 0; i < width; i += svcntw()) {
        svbool_t pg = svwhilelt_b32_s64(i, width);
        svfloat32_t y = svld1_f32(pg, srcIndices + i);
        // (y * xmax / ymax) - 0.5f
        svfloat32_t t1 = svmul_n_f32_m(pg, y, static_cast<float>(xmax));
        svfloat32_t t2 = svdiv_n_f32_m(pg, t1, static_cast<float>(ymax));
        svfloat32_t indices = svsub_n_f32_m(pg, t2, 0.5f);
        svst1_f32(pg, dstIndices + i, indices);
    }
}

inline void FillRightNearestIndicesVC(IntType width, const float *srcIndices, float *dstIndices,
    IntType ymax, IntType xmax)
{
    for (IntType i = 0; i < width; i += svcntw()) {
        svbool_t pg = svwhilelt_b32_s64(i, width);
        svfloat32_t y = svld1_f32(pg, srcIndices + i);
        // ((y + 1.0) * xmax / ymax) - 0.5f
        svfloat32_t t1 = svadd_n_f32_m(pg, y, 1.0f);
        svfloat32_t t2 = svmul_n_f32_m(pg, t1, static_cast<float>(xmax));
        svfloat32_t t3 = svdiv_n_f32_m(pg, t2, static_cast<float>(ymax));
        svfloat32_t indices = svsub_n_f32_m(pg, t3, 0.5f);
        svst1_f32(pg, dstIndices + i, indices);
    }
}

inline void FillNearestDataVC(NearestData &data, const Shape &dimsDiffDst, const Shape &dimsDiffSrc)
{
    IntType dOD = dimsDiffDst[IDX_2];
    IntType dOH = dimsDiffDst[IDX_3];
    IntType dOW = dimsDiffDst[IDX_4];
    IntType dID = dimsDiffSrc[IDX_2];
    IntType dIH = dimsDiffSrc[IDX_3];
    IntType dIW = dimsDiffSrc[IDX_4];

    std::vector<float> indices;
    InitIndices(indices, dID, dIH, dIW);

    IntType indicesSize = dID + dIH + dIW;
    std::vector<float> leftIndicesF(indicesSize);
    std::vector<float> rightIndicesF(indicesSize);

    FillLeftNearestIndicesVC(dID, &indices[0], &leftIndicesF[0], dID, dOD);
    FillLeftNearestIndicesVC(dIH, &indices[0] + dID, &leftIndicesF[0] + dID, dIH, dOH);
    FillLeftNearestIndicesVC(dIW, &indices[0] + dID + dIH, &leftIndicesF[0] + dID + dIH, dIW, dOW);

    FillRightNearestIndicesVC(dID, &indices[0], &rightIndicesF[0], dID, dOD);
    FillRightNearestIndicesVC(dIH, &indices[0] + dID, &rightIndicesF[0] + dID, dIH, dOH);
    FillRightNearestIndicesVC(dIW, &indices[0] + dID + dIH, &rightIndicesF[0] + dID + dIH, dIW, dOW);

    data.leftIndices = std::vector<IntType>(indicesSize);
    data.rightIndices = std::vector<IntType>(indicesSize);
    for (IntType i = 0; i < indicesSize; i++) {
        data.leftIndices[i] = CeilIdx(leftIndicesF[i]);
        data.rightIndices[i] = CeilIdx(rightIndicesF[i]);
    }
}

inline void FillLinearDataVC(LinearData &data, const Shape &dimsDiffDst, const Shape &dimsDiffSrc)
{
    IntType dOD = dimsDiffDst[IDX_2];
    IntType dOH = dimsDiffDst[IDX_3];
    IntType dOW = dimsDiffDst[IDX_4];
    IntType dID = dimsDiffSrc[IDX_2];
    IntType dIH = dimsDiffSrc[IDX_3];
    IntType dIW = dimsDiffSrc[IDX_4];

    // Calculate start indices and end indices with float type.
    std::vector<float> indices;
    InitIndices(indices, dID, dIH, dIW);
    std::vector<std::vector<float>> dStartIndicesF(IDX_2, std::vector<float>(dID));
    std::vector<std::vector<float>> hStartIndicesF(IDX_2, std::vector<float>(dIH));
    std::vector<std::vector<float>> wStartIndicesF(IDX_2, std::vector<float>(dIW));
    std::vector<std::vector<float>> dEndIndicesF(IDX_2, std::vector<float>(dID));
    std::vector<std::vector<float>> hEndIndicesF(IDX_2, std::vector<float>(dIH));
    std::vector<std::vector<float>> wEndIndicesF(IDX_2, std::vector<float>(dIW));

    FillLinearIndicesVC(dID, &indices[0], dStartIndicesF, dEndIndicesF, dID, dOD);
    FillLinearIndicesVC(dIH, &indices[0] + dID, hStartIndicesF, hEndIndicesF, dIH, dOH);
    FillLinearIndicesVC(dIW, &indices[0] + dID + dIH, wStartIndicesF, wEndIndicesF, dIW, dOW);

    // Fill start indices and end indices.
    data.dStartIndices = std::vector<std::vector<IntType>>(IDX_2, std::vector<IntType>(dID));
    data.hStartIndices = std::vector<std::vector<IntType>>(IDX_2, std::vector<IntType>(dIH));
    data.wStartIndices = std::vector<std::vector<IntType>>(IDX_2, std::vector<IntType>(dIW));
    data.dEndIndices = std::vector<std::vector<IntType>>(IDX_2, std::vector<IntType>(dID));
    data.hEndIndices = std::vector<std::vector<IntType>>(IDX_2, std::vector<IntType>(dIH));
    data.wEndIndices = std::vector<std::vector<IntType>>(IDX_2, std::vector<IntType>(dIW));

    for (size_t f = 0; f < IDX_2; f++) {
        for (IntType id = 0; id < dID; id++) {
            data.dStartIndices[f][id] = GetLinearStart(dStartIndicesF[f][id], id, f);
            data.dEndIndices[f][id] = GetLinearEnd(dEndIndicesF[f][id], id, dID, dOD, f);
        }
        for (IntType ih = 0; ih < dIH; ih++) {
            data.hStartIndices[f][ih] = GetLinearStart(hStartIndicesF[f][ih], ih, f);
            data.hEndIndices[f][ih] = GetLinearEnd(hEndIndicesF[f][ih], ih, dIH, dOH, f);
        }
        for (IntType iw = 0; iw < dIW; iw++) {
            data.wStartIndices[f][iw] = GetLinearStart(wStartIndicesF[f][iw], iw, f);
            data.wEndIndices[f][iw] = GetLinearEnd(wEndIndicesF[f][iw], iw, dIW, dOW, f);
        }
    }

    // Fill weights.
    indices.clear();
    InitIndices(indices, dOD, dOH, dOW);
    data.dWeights = std::vector<std::vector<float>>(IDX_2, std::vector<float>(dOD));
    data.hWeights = std::vector<std::vector<float>>(IDX_2, std::vector<float>(dOH));
    data.wWeights = std::vector<std::vector<float>>(IDX_2, std::vector<float>(dOW));

    FillLinearWeightsVC(dOD, &indices[0], &(data.dWeights[0][0]), &(data.dWeights[1][0]), dOD, dID);
    FillLinearWeightsVC(dOH, &indices[0] + dOD, &(data.hWeights[0][0]), &(data.hWeights[1][0]), dOH, dIH);
    FillLinearWeightsVC(dOW, &indices[0] + dOD + dOH, &(data.wWeights[0][0]), &(data.wWeights[1][0]), dOW, dIW);
}

// For unordered_map with std::function.
struct BwdKerParas {
    Element::TypeT diffDstType;
    Element::TypeT diffSrcType;
    SizeType ndims;
};

struct BwdKerParasHash {
    SizeType operator()(const BwdKerParas &paras) const
    {
        return static_cast<SizeType>(paras.diffDstType) * NUM_10 * NUM_10 +
               static_cast<SizeType>(paras.diffSrcType) * NUM_10 + paras.ndims;
    }
};

struct BwdKerParasEqual {
    bool operator()(const BwdKerParas &paras0, const BwdKerParas &paras1) const
    {
        return paras0.diffDstType == paras1.diffDstType &&
               paras0.diffSrcType == paras1.diffSrcType &&
               paras0.ndims == paras1.ndims;
    }
};

} // Resampling
} // Detail
} // KuDNN

#endif //  KUDNN_RESAMPLING_BWD_UTILS_HPP
