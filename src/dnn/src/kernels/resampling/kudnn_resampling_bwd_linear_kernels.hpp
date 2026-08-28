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

#ifndef KUDNN_RESAMPLING_BWD_LINEAR_KERNELS_HPP
#define KUDNN_RESAMPLING_BWD_LINEAR_KERNELS_HPP

#include "kudnn_resampling_bwd_utils.hpp"

namespace KuDNN {
namespace Detail {
namespace Resampling {

template <typename DiffDstType, typename DiffSrcType>
inline void LinearBwdVCAbxKernel(const DiffDstType *ddst, DiffSrcType *dsrc,
    StridesInfo &sInfo, const LinearData &data, IntType iw)
{
    IntType svStep = svcntw();
    svfloat32_t weights = svdup_n_f32(F32_ZERO);
    for (size_t fw = 0; fw < IDX_2; fw++) {
        IntType owStart = data.wStartIndices[fw][iw];
        IntType owEnd = data.wEndIndices[fw][iw];
        if (owStart == owEnd) { continue; }
        for (IntType ow = owStart; ow < owEnd; ow += svStep) {
            IntType ddstOffset = ow * sInfo.strideOW;
            svbool_t pg = svwhilelt_b32_s64(ow, owEnd);
            svfloat32_t t1 = LoadSvF32FromAddress(pg, ddst + ddstOffset);
            svfloat32_t t2 = svld1_f32(pg, static_cast<const float *>(&data.wWeights[fw][ow]));
            svfloat32_t t3 = svmul_f32_m(pg, t1, t2);
            weights = svadd_f32_m(pg, weights, t3);
        }
    }
    StoreF32ToAddress(dsrc, svaddv_f32(svptrue_b32(), weights));
}

template <typename DiffDstType, typename DiffSrcType>
inline void LinearBwdVCAxbKernel(const DiffDstType *ddst, DiffSrcType *dsrc,
    StridesInfo &sInfo, const LinearData &data, IntType iw, IntType cStart, IntType cEnd)
{
    IntType svStep = svcntw();
    for (IntType c = cStart; c < cEnd; c += svStep) {
        svbool_t pg = svwhilelt_b32_s64(c, cEnd);
        svfloat32_t weights = svdup_n_f32(F32_ZERO);
        for (size_t fw = 0; fw < IDX_2; fw++) {
            IntType owStart = data.wStartIndices[fw][iw];
            IntType owEnd = data.wEndIndices[fw][iw];
            if (owStart == owEnd) { continue; }
            svfloat32_t weightsW = svdup_n_f32(F32_ZERO);
            for (IntType ow = owStart; ow < owEnd; ow++) {
                IntType ddstOffset = ow * sInfo.strideOW + c * sInfo.strideOC;
                svfloat32_t t1 = LoadSvF32FromAddress(pg, ddst + ddstOffset);
                svfloat32_t t2 = svmul_n_f32_m(pg, t1, data.wWeights[fw][ow]);
                weightsW = svadd_f32_m(pg, weightsW, t2);
            }
            weights = svadd_f32_m(pg, weights, weightsW);
        }
        StoreSvF32ToAddress(pg, dsrc + c * sInfo.strideIC, weights);
    }
}

template <typename DiffDstType, typename DiffSrcType>
inline void BilinearBwdVCAbxKernel(const DiffDstType *ddst, DiffSrcType *dsrc,
    StridesInfo &sInfo, const LinearData &data, IntType ih, IntType iw)
{
    IntType svStep = svcntw();
    svbool_t pgAll = svptrue_b32();
    svfloat32_t weightsSum = svdup_n_f32(F32_ZERO);
    for (size_t fh = 0; fh < IDX_2; fh++) {
        IntType ohStart = data.hStartIndices[fh][ih];
        IntType ohEnd = data.hEndIndices[fh][ih];
        if (ohStart == ohEnd) { continue; }
        for (size_t fw = 0; fw < IDX_2; fw++) {
            IntType owStart = data.wStartIndices[fw][iw];
            IntType owEnd = data.wEndIndices[fw][iw];
            if (owStart == owEnd) { continue; }
            for (IntType oh = ohStart; oh < ohEnd; oh++) {
                svfloat32_t weights = svdup_n_f32(F32_ZERO);
                for (IntType ow = owStart; ow < owEnd; ow += svStep) {
                    IntType ddstOffset = oh * sInfo.strideOH + ow * sInfo.strideOW;
                    svbool_t pg = svwhilelt_b32_s64(ow, owEnd);
                    svfloat32_t t1 = LoadSvF32FromAddress(pg, ddst + ddstOffset);
                    svfloat32_t t2 = svld1_f32(pg, static_cast<const float *>(&data.wWeights[fw][ow]));
                    svfloat32_t t3 = svmul_f32_m(pg, t1, t2);
                    weights = svadd_f32_m(pg, weights, t3);
                }
                weights = svmul_n_f32_m(pgAll, weights, data.hWeights[fh][oh]);
                weightsSum = svadd_f32_m(pgAll, weightsSum, weights);
            }
        }
    }
    StoreF32ToAddress(dsrc, svaddv_f32(pgAll, weightsSum));
}

template <typename DiffDstType, typename DiffSrcType>
inline void BilinearBwdVCAxbKernel(const DiffDstType *ddst, DiffSrcType *dsrc,
    StridesInfo &sInfo, const LinearData &data, IntType ih, IntType iw, IntType cStart, IntType cEnd)
{
    IntType svStep = svcntw();
    for (IntType c = cStart; c < cEnd; c += svStep) {
        svbool_t pg = svwhilelt_b32_s64(c, cEnd);
        svfloat32_t weightsH = svdup_n_f32(F32_ZERO);
        for (size_t fh = 0; fh < IDX_2; fh++) {
            IntType ohStart = data.hStartIndices[fh][ih];
            IntType ohEnd = data.hEndIndices[fh][ih];
            if (ohStart == ohEnd) { continue; }
            for (size_t fw = 0; fw < IDX_2; fw++) {
                IntType owStart = data.wStartIndices[fw][iw];
                IntType owEnd = data.wEndIndices[fw][iw];
                if (owStart == owEnd) { continue; }
                for (IntType oh = ohStart; oh < ohEnd; oh++) {
                    svfloat32_t weightsW = svdup_n_f32(F32_ZERO);
                    for (IntType ow = owStart; ow < owEnd; ow++) {
                        IntType ddstOffset = oh * sInfo.strideOH + ow * sInfo.strideOW + c * sInfo.strideOC;
                        svfloat32_t t1 = LoadSvF32FromAddress(pg, ddst + ddstOffset);
                        svfloat32_t t2 = svmul_n_f32_m(pg, t1, data.wWeights[fw][ow]);
                        weightsW = svadd_f32_m(pg, weightsW, t2);
                    }
                    weightsW = svmul_n_f32_m(pg, weightsW, data.hWeights[fh][oh]);
                    weightsH = svadd_f32_m(pg, weightsH, weightsW);
                }
            }
        }
        StoreSvF32ToAddress(pg, dsrc + c * sInfo.strideIC, weightsH);
    }
}

template <typename DiffDstType, typename DiffSrcType>
inline void TrilinearBwdVCAbxKernel(const DiffDstType *ddst, DiffSrcType *dsrc,
    StridesInfo &sInfo, const LinearData &data, IntType id, IntType ih, IntType iw)
{
    IntType svStep = svcntw();
    svbool_t pgAll = svptrue_b32();
    svfloat32_t weightsSum = svdup_n_f32(F32_ZERO);
    for (size_t fd = 0; fd < IDX_2; fd++) {
        IntType odStart = data.dStartIndices[fd][id];
        IntType odEnd = data.dEndIndices[fd][id];
        if (odStart == odEnd) { continue; }
        for (size_t fh = 0; fh < IDX_2; fh++) {
            IntType ohStart = data.hStartIndices[fh][ih];
            IntType ohEnd = data.hEndIndices[fh][ih];
            if (ohStart == ohEnd) { continue; }
            for (size_t fw = 0; fw < IDX_2; fw++) {
                IntType owStart = data.wStartIndices[fw][iw];
                IntType owEnd = data.wEndIndices[fw][iw];
                if (owStart == owEnd) { continue; }
                for (IntType od = odStart; od < odEnd; od++) {
                    for (IntType oh = ohStart; oh < ohEnd; oh++) {
                        svfloat32_t weights = svdup_n_f32(F32_ZERO);
                        for (IntType ow = owStart; ow < owEnd; ow += svStep) {
                            IntType ddstOffset = od * sInfo.strideOD + oh * sInfo.strideOH + ow * sInfo.strideOW;
                            svbool_t pg = svwhilelt_b32_s64(ow, owEnd);
                            svfloat32_t t1 = LoadSvF32FromAddress(pg, ddst + ddstOffset);
                            svfloat32_t t2 = svld1_f32(pg, static_cast<const float *>(&data.wWeights[fw][ow]));
                            svfloat32_t t3 = svmul_f32_m(pg, t1, t2);
                            weights = svadd_f32_m(pg, weights, t3);
                        }
                        weights = svmul_n_f32_m(pgAll, weights, data.dWeights[fd][od] * data.hWeights[fh][oh]);
                        weightsSum = svadd_f32_m(pgAll, weightsSum, weights);
                    }
                }
            }
        }
    }
    StoreF32ToAddress(dsrc, svaddv_f32(pgAll, weightsSum));
}

template <typename DiffDstType, typename DiffSrcType>
inline void TrilinearBwdVCAxbKernel(const DiffDstType *ddst, DiffSrcType *dsrc,
    StridesInfo &sInfo, const LinearData &data, IntType id, IntType ih, IntType iw, IntType cStart, IntType cEnd)
{
    IntType svStep = svcntw();
    for (IntType c = cStart; c < cEnd; c += svStep) {
        svbool_t pg = svwhilelt_b32_s64(c, cEnd);
        svfloat32_t weightsD = svdup_n_f32(F32_ZERO);
        for (size_t fd = 0; fd < IDX_2; fd++) {
            IntType odStart = data.dStartIndices[fd][id];
            IntType odEnd = data.dEndIndices[fd][id];
            if (odStart == odEnd) { continue; }
            for (size_t fh = 0; fh < IDX_2; fh++) {
                IntType ohStart = data.hStartIndices[fh][ih];
                IntType ohEnd = data.hEndIndices[fh][ih];
                if (ohStart == ohEnd) { continue; }
                for (size_t fw = 0; fw < IDX_2; fw++) {
                    IntType owStart = data.wStartIndices[fw][iw];
                    IntType owEnd = data.wEndIndices[fw][iw];
                    if (owStart == owEnd) { continue; }
                    for (IntType od = odStart; od < odEnd; od++) {
                        svfloat32_t weightsH = svdup_n_f32(F32_ZERO);
                        for (IntType oh = ohStart; oh < ohEnd; oh++) {
                            svfloat32_t weightsW = svdup_n_f32(F32_ZERO);
                            for (IntType ow = owStart; ow < owEnd; ow++) {
                                IntType ddstOffset = od * sInfo.strideOD
                                    + oh * sInfo.strideOH + ow * sInfo.strideOW + c * sInfo.strideOC;
                                svfloat32_t t1 = LoadSvF32FromAddress(pg, ddst + ddstOffset);
                                svfloat32_t t2 = svmul_n_f32_m(pg, t1, data.wWeights[fw][ow]);
                                weightsW = svadd_f32_m(pg, weightsW, t2);
                            }
                            weightsW = svmul_n_f32_m(pg, weightsW, data.hWeights[fh][oh]);
                            weightsH = svadd_f32_m(pg, weightsH, weightsW);
                        }
                        weightsH = svmul_n_f32_m(pg, weightsH, data.dWeights[fd][od]);
                        weightsD = svadd_f32_m(pg, weightsD, weightsH);
                    }
                }
            }
        }
        StoreSvF32ToAddress(pg, dsrc + c * sInfo.strideIC, weightsD);
    }
}

template <typename DiffDstType, typename DiffSrcType>
void LinearBwdVC(const Shape &batchSplitDims, const Shape &stride, const void *diffDst, void *diffSrc,
    const Shape &stridesDiffDst, const Shape &stridesDiffSrc, const LinearData &data, bool isAxb)
{
    SplitDimInfo<IntType> sdInfo(batchSplitDims, stride, isAxb);
    StridesInfo sInfo(stridesDiffSrc, stridesDiffDst);

    if (isAxb) {
        for (IntType mb = sdInfo.mbStart; mb < sdInfo.mbEnd; mb++) {
            for (IntType iw = sdInfo.wStart; iw < sdInfo.wEnd; iw++) {
                const DiffDstType *ddst = static_cast<const DiffDstType *>(diffDst) + mb * sInfo.strideOMB;
                DiffSrcType *dsrc = static_cast<DiffSrcType *>(diffSrc) + mb * sInfo.strideIMB + iw * sInfo.strideIW;
                LinearBwdVCAxbKernel(ddst, dsrc, sInfo, data, iw, sdInfo.cStart, sdInfo.cEnd);
            }
        }
        return ;
    }

    for (IntType mb = sdInfo.mbStart; mb < sdInfo.mbEnd; mb++) {
        for (IntType c = sdInfo.cStart; c < sdInfo.cEnd; c++) {
            for (IntType iw = sdInfo.wStart; iw < sdInfo.wEnd; iw++) {
                const DiffDstType *ddst = static_cast<const DiffDstType *>(diffDst)
                    + mb * sInfo.strideOMB + c * sInfo.strideOC;
                DiffSrcType *dsrc = static_cast<DiffSrcType *>(diffSrc)
                    + mb * sInfo.strideIMB + c * sInfo.strideIC + iw * sInfo.strideIW;
                LinearBwdVCAbxKernel(ddst, dsrc, sInfo, data, iw);
            }
        }
    }
}

template <typename DiffDstType, typename DiffSrcType>
void BilinearBwdVC(const Shape &batchSplitDims, const Shape &stride, const void *diffDst, void *diffSrc,
    const Shape &stridesDiffDst, const Shape &stridesDiffSrc, const LinearData &data, bool isAxb)
{
    SplitDimInfo<IntType> sdInfo(batchSplitDims, stride, isAxb);
    StridesInfo sInfo(stridesDiffSrc, stridesDiffDst);

    if (isAxb) {
        for (IntType mb = sdInfo.mbStart; mb < sdInfo.mbEnd; mb++) {
            for (IntType ih = sdInfo.hStart; ih < sdInfo.hEnd; ih++) {
                for (IntType iw = sdInfo.wStart; iw < sdInfo.wEnd; iw++) {
                    const DiffDstType *ddst = static_cast<const DiffDstType *>(diffDst) + mb * sInfo.strideOMB;
                    DiffSrcType *dsrc = static_cast<DiffSrcType *>(diffSrc)
                        + mb * sInfo.strideIMB + ih * sInfo.strideIH + iw * sInfo.strideIW;
                    BilinearBwdVCAxbKernel(ddst, dsrc, sInfo, data, ih, iw, sdInfo.cStart, sdInfo.cEnd);
                }
            }
        }
        return ;
    }

    for (IntType mb = sdInfo.mbStart; mb < sdInfo.mbEnd; mb++) {
        for (IntType c = sdInfo.cStart; c < sdInfo.cEnd; c++) {
            for (IntType ih = sdInfo.hStart; ih < sdInfo.hEnd; ih++) {
                for (IntType iw = sdInfo.wStart; iw < sdInfo.wEnd; iw++) {
                    const DiffDstType *ddst = static_cast<const DiffDstType *>(diffDst)
                        + mb * sInfo.strideOMB + c * sInfo.strideOC;
                    DiffSrcType *dsrc = static_cast<DiffSrcType *>(diffSrc)
                        + mb * sInfo.strideIMB + c * sInfo.strideIC + ih * sInfo.strideIH + iw * sInfo.strideIW;
                    BilinearBwdVCAbxKernel(ddst, dsrc, sInfo, data, ih, iw);
                }
            }
        }
    }
}

template <typename DiffDstType, typename DiffSrcType>
void TrilinearBwdVC(const Shape &batchSplitDims, const Shape &stride, const void *diffDst, void *diffSrc,
    const Shape &stridesDiffDst, const Shape &stridesDiffSrc, const LinearData &data, bool isAxb)
{
    SplitDimInfo<IntType> sdInfo(batchSplitDims, stride, isAxb);
    StridesInfo sInfo(stridesDiffSrc, stridesDiffDst);

    if (isAxb) {
        for (IntType mb = sdInfo.mbStart; mb < sdInfo.mbEnd; mb++) {
            for (IntType id = sdInfo.dStart; id < sdInfo.dEnd; id++) {
                for (IntType ih = sdInfo.hStart; ih < sdInfo.hEnd; ih++) {
                    for (IntType iw = sdInfo.wStart; iw < sdInfo.wEnd; iw++) {
                        const DiffDstType *ddst = static_cast<const DiffDstType *>(diffDst) + mb * sInfo.strideOMB;
                        DiffSrcType *dsrc = static_cast<DiffSrcType *>(diffSrc)
                            + mb * sInfo.strideIMB + id * sInfo.strideID + ih * sInfo.strideIH + iw * sInfo.strideIW;
                        TrilinearBwdVCAxbKernel(
                            ddst, dsrc, sInfo, data, id, ih, iw, sdInfo.cStart, sdInfo.cEnd);
                    }
                }
            }
        }
        return ;
    }

    for (IntType mb = sdInfo.mbStart; mb < sdInfo.mbEnd; mb++) {
        for (IntType c = sdInfo.cStart; c < sdInfo.cEnd; c++) {
            for (IntType id = sdInfo.dStart; id < sdInfo.dEnd; id++) {
                for (IntType ih = sdInfo.hStart; ih < sdInfo.hEnd; ih++) {
                    for (IntType iw = sdInfo.wStart; iw < sdInfo.wEnd; iw++) {
                        const DiffDstType *ddst = static_cast<const DiffDstType *>(diffDst)
                            + mb * sInfo.strideOMB + c * sInfo.strideOC;
                        DiffSrcType *dsrc = static_cast<DiffSrcType *>(diffSrc)
                            + mb * sInfo.strideIMB + c * sInfo.strideIC
                            + id * sInfo.strideID + ih * sInfo.strideIH + iw * sInfo.strideIW;
                        TrilinearBwdVCAbxKernel(ddst, dsrc, sInfo, data, id, ih, iw);
                    }
                }
            }
        }
    }
}

// std::function for LinearBwdVC(), BilinearBwdVC() and TrilinearBwdVC().
using LinearBwdKerFunc = std::function<void(const Shape &, const Shape &, const void *, void *,
                                            const Shape &, const Shape &, const LinearData &, bool)>;

const std::unordered_map<BwdKerParas, LinearBwdKerFunc, BwdKerParasHash, BwdKerParasEqual> LINEAR_BWD_KER_MAP = {
    // F32
    {{Element::TypeT::F32, Element::TypeT::F32, DIM_3}, LinearBwdVC<float, float>},
    {{Element::TypeT::F32, Element::TypeT::F16, DIM_3}, LinearBwdVC<float, __fp16>},
    {{Element::TypeT::F32, Element::TypeT::BF16, DIM_3}, LinearBwdVC<float, __bf16>},
    {{Element::TypeT::F32, Element::TypeT::S32, DIM_3}, LinearBwdVC<float, int32_t>},
    {{Element::TypeT::F32, Element::TypeT::S8, DIM_3}, LinearBwdVC<float, int8_t>},
    {{Element::TypeT::F32, Element::TypeT::U8, DIM_3}, LinearBwdVC<float, uint8_t>},

    {{Element::TypeT::F32, Element::TypeT::F32, DIM_4}, BilinearBwdVC<float, float>},
    {{Element::TypeT::F32, Element::TypeT::F16, DIM_4}, BilinearBwdVC<float, __fp16>},
    {{Element::TypeT::F32, Element::TypeT::BF16, DIM_4}, BilinearBwdVC<float, __bf16>},
    {{Element::TypeT::F32, Element::TypeT::S32, DIM_4}, BilinearBwdVC<float, int32_t>},
    {{Element::TypeT::F32, Element::TypeT::S8, DIM_4}, BilinearBwdVC<float, int8_t>},
    {{Element::TypeT::F32, Element::TypeT::U8, DIM_4}, BilinearBwdVC<float, uint8_t>},

    {{Element::TypeT::F32, Element::TypeT::F32, DIM_5}, TrilinearBwdVC<float, float>},
    {{Element::TypeT::F32, Element::TypeT::F16, DIM_5}, TrilinearBwdVC<float, __fp16>},
    {{Element::TypeT::F32, Element::TypeT::BF16, DIM_5}, TrilinearBwdVC<float, __bf16>},
    {{Element::TypeT::F32, Element::TypeT::S32, DIM_5}, TrilinearBwdVC<float, int32_t>},
    {{Element::TypeT::F32, Element::TypeT::S8, DIM_5}, TrilinearBwdVC<float, int8_t>},
    {{Element::TypeT::F32, Element::TypeT::U8, DIM_5}, TrilinearBwdVC<float, uint8_t>},
    // F16
    {{Element::TypeT::F16, Element::TypeT::F32, DIM_3}, LinearBwdVC<__fp16, float>},
    {{Element::TypeT::F16, Element::TypeT::F16, DIM_3}, LinearBwdVC<__fp16, __fp16>},
    {{Element::TypeT::F16, Element::TypeT::BF16, DIM_3}, LinearBwdVC<__fp16, __bf16>},
    {{Element::TypeT::F16, Element::TypeT::S32, DIM_3}, LinearBwdVC<__fp16, int32_t>},
    {{Element::TypeT::F16, Element::TypeT::S8, DIM_3}, LinearBwdVC<__fp16, int8_t>},
    {{Element::TypeT::F16, Element::TypeT::U8, DIM_3}, LinearBwdVC<__fp16, uint8_t>},

    {{Element::TypeT::F16, Element::TypeT::F32, DIM_4}, BilinearBwdVC<__fp16, float>},
    {{Element::TypeT::F16, Element::TypeT::F16, DIM_4}, BilinearBwdVC<__fp16, __fp16>},
    {{Element::TypeT::F16, Element::TypeT::BF16, DIM_4}, BilinearBwdVC<__fp16, __bf16>},
    {{Element::TypeT::F16, Element::TypeT::S32, DIM_4}, BilinearBwdVC<__fp16, int32_t>},
    {{Element::TypeT::F16, Element::TypeT::S8, DIM_4}, BilinearBwdVC<__fp16, int8_t>},
    {{Element::TypeT::F16, Element::TypeT::U8, DIM_4}, BilinearBwdVC<__fp16, uint8_t>},

    {{Element::TypeT::F16, Element::TypeT::F32, DIM_5}, TrilinearBwdVC<__fp16, float>},
    {{Element::TypeT::F16, Element::TypeT::F16, DIM_5}, TrilinearBwdVC<__fp16, __fp16>},
    {{Element::TypeT::F16, Element::TypeT::BF16, DIM_5}, TrilinearBwdVC<__fp16, __bf16>},
    {{Element::TypeT::F16, Element::TypeT::S32, DIM_5}, TrilinearBwdVC<__fp16, int32_t>},
    {{Element::TypeT::F16, Element::TypeT::S8, DIM_5}, TrilinearBwdVC<__fp16, int8_t>},
    {{Element::TypeT::F16, Element::TypeT::U8, DIM_5}, TrilinearBwdVC<__fp16, uint8_t>},
    // BF16
    {{Element::TypeT::BF16, Element::TypeT::F32, DIM_3}, LinearBwdVC<__bf16, float>},
    {{Element::TypeT::BF16, Element::TypeT::F16, DIM_3}, LinearBwdVC<__bf16, __fp16>},
    {{Element::TypeT::BF16, Element::TypeT::BF16, DIM_3}, LinearBwdVC<__bf16, __bf16>},
    {{Element::TypeT::BF16, Element::TypeT::S32, DIM_3}, LinearBwdVC<__bf16, int32_t>},
    {{Element::TypeT::BF16, Element::TypeT::S8, DIM_3}, LinearBwdVC<__bf16, int8_t>},
    {{Element::TypeT::BF16, Element::TypeT::U8, DIM_3}, LinearBwdVC<__bf16, uint8_t>},

    {{Element::TypeT::BF16, Element::TypeT::F32, DIM_4}, BilinearBwdVC<__bf16, float>},
    {{Element::TypeT::BF16, Element::TypeT::F16, DIM_4}, BilinearBwdVC<__bf16, __fp16>},
    {{Element::TypeT::BF16, Element::TypeT::BF16, DIM_4}, BilinearBwdVC<__bf16, __bf16>},
    {{Element::TypeT::BF16, Element::TypeT::S32, DIM_4}, BilinearBwdVC<__bf16, int32_t>},
    {{Element::TypeT::BF16, Element::TypeT::S8, DIM_4}, BilinearBwdVC<__bf16, int8_t>},
    {{Element::TypeT::BF16, Element::TypeT::U8, DIM_4}, BilinearBwdVC<__bf16, uint8_t>},

    {{Element::TypeT::BF16, Element::TypeT::F32, DIM_5}, TrilinearBwdVC<__bf16, float>},
    {{Element::TypeT::BF16, Element::TypeT::F16, DIM_5}, TrilinearBwdVC<__bf16, __fp16>},
    {{Element::TypeT::BF16, Element::TypeT::BF16, DIM_5}, TrilinearBwdVC<__bf16, __bf16>},
    {{Element::TypeT::BF16, Element::TypeT::S32, DIM_5}, TrilinearBwdVC<__bf16, int32_t>},
    {{Element::TypeT::BF16, Element::TypeT::S8, DIM_5}, TrilinearBwdVC<__bf16, int8_t>},
    {{Element::TypeT::BF16, Element::TypeT::U8, DIM_5}, TrilinearBwdVC<__bf16, uint8_t>},
    // S32
    {{Element::TypeT::S32, Element::TypeT::F32, DIM_3}, LinearBwdVC<int32_t, float>},
    {{Element::TypeT::S32, Element::TypeT::F16, DIM_3}, LinearBwdVC<int32_t, __fp16>},
    {{Element::TypeT::S32, Element::TypeT::BF16, DIM_3}, LinearBwdVC<int32_t, __bf16>},
    {{Element::TypeT::S32, Element::TypeT::S32, DIM_3}, LinearBwdVC<int32_t, int32_t>},
    {{Element::TypeT::S32, Element::TypeT::S8, DIM_3}, LinearBwdVC<int32_t, int8_t>},
    {{Element::TypeT::S32, Element::TypeT::U8, DIM_3}, LinearBwdVC<int32_t, uint8_t>},

    {{Element::TypeT::S32, Element::TypeT::F32, DIM_4}, BilinearBwdVC<int32_t, float>},
    {{Element::TypeT::S32, Element::TypeT::F16, DIM_4}, BilinearBwdVC<int32_t, __fp16>},
    {{Element::TypeT::S32, Element::TypeT::BF16, DIM_4}, BilinearBwdVC<int32_t, __bf16>},
    {{Element::TypeT::S32, Element::TypeT::S32, DIM_4}, BilinearBwdVC<int32_t, int32_t>},
    {{Element::TypeT::S32, Element::TypeT::S8, DIM_4}, BilinearBwdVC<int32_t, int8_t>},
    {{Element::TypeT::S32, Element::TypeT::U8, DIM_4}, BilinearBwdVC<int32_t, uint8_t>},

    {{Element::TypeT::S32, Element::TypeT::F32, DIM_5}, TrilinearBwdVC<int32_t, float>},
    {{Element::TypeT::S32, Element::TypeT::F16, DIM_5}, TrilinearBwdVC<int32_t, __fp16>},
    {{Element::TypeT::S32, Element::TypeT::BF16, DIM_5}, TrilinearBwdVC<int32_t, __bf16>},
    {{Element::TypeT::S32, Element::TypeT::S32, DIM_5}, TrilinearBwdVC<int32_t, int32_t>},
    {{Element::TypeT::S32, Element::TypeT::S8, DIM_5}, TrilinearBwdVC<int32_t, int8_t>},
    {{Element::TypeT::S32, Element::TypeT::U8, DIM_5}, TrilinearBwdVC<int32_t, uint8_t>},
    // S8
    {{Element::TypeT::S8, Element::TypeT::F32, DIM_3}, LinearBwdVC<int8_t, float>},
    {{Element::TypeT::S8, Element::TypeT::F16, DIM_3}, LinearBwdVC<int8_t, __fp16>},
    {{Element::TypeT::S8, Element::TypeT::BF16, DIM_3}, LinearBwdVC<int8_t, __bf16>},
    {{Element::TypeT::S8, Element::TypeT::S32, DIM_3}, LinearBwdVC<int8_t, int32_t>},
    {{Element::TypeT::S8, Element::TypeT::S8, DIM_3}, LinearBwdVC<int8_t, int8_t>},
    {{Element::TypeT::S8, Element::TypeT::U8, DIM_3}, LinearBwdVC<int8_t, uint8_t>},

    {{Element::TypeT::S8, Element::TypeT::F32, DIM_4}, BilinearBwdVC<int8_t, float>},
    {{Element::TypeT::S8, Element::TypeT::F16, DIM_4}, BilinearBwdVC<int8_t, __fp16>},
    {{Element::TypeT::S8, Element::TypeT::BF16, DIM_4}, BilinearBwdVC<int8_t, __bf16>},
    {{Element::TypeT::S8, Element::TypeT::S32, DIM_4}, BilinearBwdVC<int8_t, int32_t>},
    {{Element::TypeT::S8, Element::TypeT::S8, DIM_4}, BilinearBwdVC<int8_t, int8_t>},
    {{Element::TypeT::S8, Element::TypeT::U8, DIM_4}, BilinearBwdVC<int8_t, uint8_t>},

    {{Element::TypeT::S8, Element::TypeT::F32, DIM_5}, TrilinearBwdVC<int8_t, float>},
    {{Element::TypeT::S8, Element::TypeT::F16, DIM_5}, TrilinearBwdVC<int8_t, __fp16>},
    {{Element::TypeT::S8, Element::TypeT::BF16, DIM_5}, TrilinearBwdVC<int8_t, __bf16>},
    {{Element::TypeT::S8, Element::TypeT::S32, DIM_5}, TrilinearBwdVC<int8_t, int32_t>},
    {{Element::TypeT::S8, Element::TypeT::S8, DIM_5}, TrilinearBwdVC<int8_t, int8_t>},
    {{Element::TypeT::S8, Element::TypeT::U8, DIM_5}, TrilinearBwdVC<int8_t, uint8_t>},
    // U8
    {{Element::TypeT::U8, Element::TypeT::F32, DIM_3}, LinearBwdVC<uint8_t, float>},
    {{Element::TypeT::U8, Element::TypeT::F16, DIM_3}, LinearBwdVC<uint8_t, __fp16>},
    {{Element::TypeT::U8, Element::TypeT::BF16, DIM_3}, LinearBwdVC<uint8_t, __bf16>},
    {{Element::TypeT::U8, Element::TypeT::S32, DIM_3}, LinearBwdVC<uint8_t, int32_t>},
    {{Element::TypeT::U8, Element::TypeT::S8, DIM_3}, LinearBwdVC<uint8_t, int8_t>},
    {{Element::TypeT::U8, Element::TypeT::U8, DIM_3}, LinearBwdVC<uint8_t, uint8_t>},

    {{Element::TypeT::U8, Element::TypeT::F32, DIM_4}, BilinearBwdVC<uint8_t, float>},
    {{Element::TypeT::U8, Element::TypeT::F16, DIM_4}, BilinearBwdVC<uint8_t, __fp16>},
    {{Element::TypeT::U8, Element::TypeT::BF16, DIM_4}, BilinearBwdVC<uint8_t, __bf16>},
    {{Element::TypeT::U8, Element::TypeT::S32, DIM_4}, BilinearBwdVC<uint8_t, int32_t>},
    {{Element::TypeT::U8, Element::TypeT::S8, DIM_4}, BilinearBwdVC<uint8_t, int8_t>},
    {{Element::TypeT::U8, Element::TypeT::U8, DIM_4}, BilinearBwdVC<uint8_t, uint8_t>},

    {{Element::TypeT::U8, Element::TypeT::F32, DIM_5}, TrilinearBwdVC<uint8_t, float>},
    {{Element::TypeT::U8, Element::TypeT::F16, DIM_5}, TrilinearBwdVC<uint8_t, __fp16>},
    {{Element::TypeT::U8, Element::TypeT::BF16, DIM_5}, TrilinearBwdVC<uint8_t, __bf16>},
    {{Element::TypeT::U8, Element::TypeT::S32, DIM_5}, TrilinearBwdVC<uint8_t, int32_t>},
    {{Element::TypeT::U8, Element::TypeT::S8, DIM_5}, TrilinearBwdVC<uint8_t, int8_t>},
    {{Element::TypeT::U8, Element::TypeT::U8, DIM_5}, TrilinearBwdVC<uint8_t, uint8_t>}
};

} // Resampling
} // Detail
} // KuDNN

#endif //  KUDNN_RESAMPLING_BWD_LINEAR_KERNELS_HPP
