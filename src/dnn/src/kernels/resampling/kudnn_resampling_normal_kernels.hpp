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

#ifndef KUDNN_RESAMPLING_NORMAL_KERNELS_HPP
#define KUDNN_RESAMPLING_NORMAL_KERNELS_HPP

#include "kudnn_resampling_utils.hpp"

namespace KuDNN {
namespace Detail {
namespace Resampling {

template<typename SrcType, typename DstType>
void NearestNormal(const Shape &batchSplitDims, const Shape &stride, const Shape &dimsDst,
    const StridesInfo &sInfo, const void *src, void *dst, const ResamplingCoeffs<IntType> &coeffs, bool isAxb)
{
    SplitDimInfo<int32_t> sdInfo(batchSplitDims, stride, isAxb);
    IntType dOD = dimsDst[IDX_2];
    IntType dOH = dimsDst[IDX_3];

    if (!isAxb) {
        for (IntType mb = sdInfo.mbStart; mb < sdInfo.mbEnd; mb++) {
            for (IntType c = sdInfo.cStart; c < sdInfo.cEnd; c++) {
                for (IntType od = sdInfo.dStart; od < sdInfo.dEnd; od++) {
                    IntType id = coeffs.outidx2Inidx[od];
                    for (IntType oh = sdInfo.hStart; oh < sdInfo.hEnd; oh++) {
                        IntType ih = coeffs.outidx2Inidx[dOD + oh];
                        SizeType outerSrcOff = mb * sInfo.strideIMB +
                            c * sInfo.strideIC + id * sInfo.strideID + ih * sInfo.strideIH;
                        SizeType outerDstOff = mb * sInfo.strideOMB +
                            c * sInfo.strideOC + od * sInfo.strideOD + oh * sInfo.strideOH;
                        const SrcType *fsrc = static_cast<const SrcType *>(src) + outerSrcOff;
                        DstType *fdst = static_cast<DstType *>(dst) + outerDstOff;
                        for (IntType ow = sdInfo.wStart; ow < sdInfo.wEnd; ow++) {
                            IntType iw = coeffs.outidx2Inidx[dOD + dOH + ow];
                            Assign<SrcType, DstType>(&fdst[ow], &fsrc[iw]);
                        }
                    }
                }
            }
        }
        return;
    }

    // axb format
    for (IntType mb = sdInfo.mbStart; mb < sdInfo.mbEnd; mb++) {
        for (IntType od = sdInfo.dStart; od < sdInfo.dEnd; od++) {
            IntType id = coeffs.outidx2Inidx[od];
            for (IntType oh = sdInfo.hStart; oh < sdInfo.hEnd; oh++) {
                IntType ih = coeffs.outidx2Inidx[dOD + oh];
                for (IntType ow = sdInfo.wStart; ow < sdInfo.wEnd; ow++) {
                    IntType iw = coeffs.outidx2Inidx[dOD + dOH + ow];
                    SizeType outerSrcOff = mb * sInfo.strideIMB +
                        iw * sInfo.strideIW + id * sInfo.strideID + ih * sInfo.strideIH;
                    SizeType outerDstOff = mb * sInfo.strideOMB +
                        ow * sInfo.strideOW + od * sInfo.strideOD + oh * sInfo.strideOH;
                    const SrcType *fsrc = static_cast<const SrcType *>(src) + outerSrcOff;
                    DstType *fdst = static_cast<DstType *>(dst) + outerDstOff;
                    for (IntType c = sdInfo.cStart; c < sdInfo.cEnd; c++) {
                        Assign<SrcType, DstType>(&fdst[c], &fsrc[c]);
                    }
                }
            }
        }
    }
}

template<typename SrcType, typename DstType>
inline void LinearNormalKernel(const SrcType *src, DstType *dst,
    const std::vector<IntType> &offsets, const std::vector<float> &weights)
{
    float res = static_cast<float>(NUM_0);
    for (SizeType i = 0; i < offsets.size(); i++) {
        res += LoadF32FromAddress(src + offsets[i]) * weights[i];
    }
    StoreF32ToAddress(dst, res);
}

template<typename SrcType, typename DstType>
void LinearNormal(const Shape &batchSplitDims, const Shape &stride, const Shape &dimsDst,
    const StridesInfo &sInfo, const void *src, void *dst, const ResamplingCoeffs<IntType> &coeffs, bool isAxb)
{
    SplitDimInfo<int32_t> sdInfo(batchSplitDims, stride, isAxb);
    IntType dOD = dimsDst[IDX_2];
    IntType dOH = dimsDst[IDX_3];
    std::vector<std::vector<IntType>> offsetsStk;
    std::vector<std::vector<float>> weightsStk;
    LinearStksInit(IDX_2, offsetsStk, weightsStk);

    if (!isAxb) {
        for (IntType mb = sdInfo.mbStart; mb < sdInfo.mbEnd; mb++) {
            for (IntType c = sdInfo.cStart; c < sdInfo.cEnd; c++) {
                offsetsStk[IDX_0][IDX_0] = mb * sInfo.strideIMB + c * sInfo.strideIC;
                for (IntType ow = sdInfo.wStart; ow < sdInfo.wEnd; ow++) {
                    Linearize(coeffs, dOD + dOH + ow, sInfo.strideIW, offsetsStk, weightsStk, IDX_1);
                    SizeType dstOffset = mb * sInfo.strideOMB + c * sInfo.strideOC + ow;
                    LinearNormalKernel<SrcType, DstType>(static_cast<const SrcType *>(src),
                        static_cast<DstType *>(dst) + dstOffset, offsetsStk[IDX_1], weightsStk[IDX_1]);
                }
            }
        }
        return;
    }

    // axb format
    for (IntType mb = sdInfo.mbStart; mb < sdInfo.mbEnd; mb++) {
        offsetsStk[IDX_0][IDX_0] = mb * sInfo.strideIMB;
        for (IntType ow = sdInfo.wStart; ow < sdInfo.wEnd; ow++) {
            Linearize(coeffs, dOD + dOH + ow, sInfo.strideIW, offsetsStk, weightsStk, IDX_1);
            for (IntType c = sdInfo.cStart; c < sdInfo.cEnd; c++) {
                SizeType dstOffset = mb * sInfo.strideOMB + ow * sInfo.strideOW + c;
                LinearNormalKernel<SrcType, DstType>(static_cast<const SrcType *>(src) + c,
                    static_cast<DstType *>(dst) + dstOffset, offsetsStk[IDX_1], weightsStk[IDX_1]);
            }
        }
    }
}

template<typename SrcType, typename DstType>
void BilinearNormal(const Shape &batchSplitDims, const Shape &stride, const Shape &dimsDst,
    const StridesInfo &sInfo, const void *src, void *dst, const ResamplingCoeffs<IntType> &coeffs, bool isAxb)
{
    SplitDimInfo<int32_t> sdInfo(batchSplitDims, stride, isAxb);
    IntType dOD = dimsDst[IDX_2];
    IntType dOH = dimsDst[IDX_3];
    std::vector<std::vector<IntType>> offsetsStk;
    std::vector<std::vector<float>> weightsStk;
    LinearStksInit(IDX_3, offsetsStk, weightsStk);

    if (!isAxb) {
        for (IntType mb = sdInfo.mbStart; mb < sdInfo.mbEnd; mb++) {
            for (IntType c = sdInfo.cStart; c < sdInfo.cEnd; c++) {
                offsetsStk[IDX_0][IDX_0] = mb * sInfo.strideIMB + c * sInfo.strideIC;
                for (IntType oh = sdInfo.hStart; oh < sdInfo.hEnd; oh++) {
                    Linearize(coeffs, dOD + oh, sInfo.strideIH, offsetsStk, weightsStk, IDX_1);
                    for (IntType ow = sdInfo.wStart; ow < sdInfo.wEnd; ow++) {
                        Linearize(coeffs, dOD + dOH + ow, sInfo.strideIW, offsetsStk, weightsStk, IDX_2);
                        SizeType dstOffset = mb * sInfo.strideOMB + c * sInfo.strideOC + oh * sInfo.strideOH + ow;
                        LinearNormalKernel<SrcType, DstType>(static_cast<const SrcType *>(src),
                            static_cast<DstType *>(dst) + dstOffset, offsetsStk[IDX_2], weightsStk[IDX_2]);
                    }
                }
            }
        }
        return;
    }

    // axb format
    for (IntType mb = sdInfo.mbStart; mb < sdInfo.mbEnd; mb++) {
        offsetsStk[IDX_0][IDX_0] = mb * sInfo.strideIMB;
        for (IntType oh = sdInfo.hStart; oh < sdInfo.hEnd; oh++) {
            Linearize(coeffs, dOD + oh, sInfo.strideIH, offsetsStk, weightsStk, IDX_1);
            for (IntType ow = sdInfo.wStart; ow < sdInfo.wEnd; ow++) {
                Linearize(coeffs, dOD + dOH + ow, sInfo.strideIW, offsetsStk, weightsStk, IDX_2);
                for (IntType c = sdInfo.cStart; c < sdInfo.cEnd; c++) {
                    SizeType dstOffset = mb * sInfo.strideOMB + oh * sInfo.strideOH + ow * sInfo.strideOW + c;
                    LinearNormalKernel<SrcType, DstType>(static_cast<const SrcType *>(src) + c,
                        static_cast<DstType *>(dst) + dstOffset, offsetsStk[IDX_2], weightsStk[IDX_2]);
                }
            }
        }
    }
}

template<typename SrcType, typename DstType>
void TrilinearNormal(const Shape &batchSplitDims, const Shape &stride, const Shape &dimsDst,
    const StridesInfo &sInfo, const void *src, void *dst, const ResamplingCoeffs<IntType> &coeffs, bool isAxb)
{
    SplitDimInfo<int32_t> sdInfo(batchSplitDims, stride, isAxb);
    IntType dOD = dimsDst[IDX_2];
    IntType dOH = dimsDst[IDX_3];
    std::vector<std::vector<IntType>> offsetsStk;
    std::vector<std::vector<float>> weightsStk;
    LinearStksInit(IDX_4, offsetsStk, weightsStk);

    if (!isAxb) {
        for (IntType mb = sdInfo.mbStart; mb < sdInfo.mbEnd; mb++) {
            for (IntType c = sdInfo.cStart; c < sdInfo.cEnd; c++) {
                offsetsStk[IDX_0][IDX_0] = mb * sInfo.strideIMB + c * sInfo.strideIC;
                for (IntType od = sdInfo.dStart; od < sdInfo.dEnd; od++) {
                    Linearize(coeffs, od, sInfo.strideID, offsetsStk, weightsStk, IDX_1);
                    for (IntType oh = sdInfo.hStart; oh < sdInfo.hEnd; oh++) {
                        Linearize(coeffs, dOD + oh, sInfo.strideIH, offsetsStk, weightsStk, IDX_2);
                        for (IntType ow = sdInfo.wStart; ow < sdInfo.wEnd; ow++) {
                            Linearize(coeffs, dOD + dOH + ow, sInfo.strideIW, offsetsStk, weightsStk, IDX_3);
                            SizeType dstOffset = mb * sInfo.strideOMB + c * sInfo.strideOC +
                                                 od * sInfo.strideOD + oh * sInfo.strideOH + ow;
                            LinearNormalKernel<SrcType, DstType>(static_cast<const SrcType *>(src),
                                static_cast<DstType *>(dst) + dstOffset, offsetsStk[IDX_3], weightsStk[IDX_3]);
                        }
                    }
                }
            }
        }
        return;
    }

    // axb format
    for (IntType mb = sdInfo.mbStart; mb < sdInfo.mbEnd; mb++) {
        offsetsStk[IDX_0][IDX_0] = mb * sInfo.strideIMB;
        for (IntType od = sdInfo.dStart; od < sdInfo.dEnd; od++) {
            Linearize(coeffs, od, sInfo.strideID, offsetsStk, weightsStk, IDX_1);
            for (IntType oh = sdInfo.hStart; oh < sdInfo.hEnd; oh++) {
                Linearize(coeffs, dOD + oh, sInfo.strideIH, offsetsStk, weightsStk, IDX_2);
                for (IntType ow = sdInfo.wStart; ow < sdInfo.wEnd; ow++) {
                    Linearize(coeffs, dOD + dOH + ow, sInfo.strideIW, offsetsStk, weightsStk, IDX_3);
                    for (IntType c = sdInfo.cStart; c < sdInfo.cEnd; c++) {
                        SizeType dstOffset = mb * sInfo.strideOMB + od * sInfo.strideOD +
                                             oh * sInfo.strideOH + ow * sInfo.strideOW + c;
                        LinearNormalKernel<SrcType, DstType>(static_cast<const SrcType *>(src) + c,
                            static_cast<DstType *>(dst) + dstOffset, offsetsStk[IDX_3], weightsStk[IDX_3]);
                    }
                }
            }
        }
    }
}

// std::function for NearestNormal(), LinearNormal(), BilinearNormal() and TrilinearNormal().
using NormalKerFunc = std::function<void(const Shape &, const Shape &, const Shape &, const StridesInfo &,
                                         const void *, void *, const ResamplingCoeffs<IntType> &coeffs, bool)>;

const std::unordered_map<NormalKerParas, NormalKerFunc, NormalKerParasHash, NormalKerParasEqual> NORMAL_KER_MAP = {
    // F32
    {{Element::TypeT::F32, Element::TypeT::F32, DIM_5, ResamplingAlg::NEAREST_NEIGHBOR}, NearestNormal<float, float>},
    {{Element::TypeT::F32, Element::TypeT::F16, DIM_5, ResamplingAlg::NEAREST_NEIGHBOR}, NearestNormal<float, __fp16>},
    {{Element::TypeT::F32, Element::TypeT::BF16, DIM_5, ResamplingAlg::NEAREST_NEIGHBOR}, NearestNormal<float, __bf16>},
    {{Element::TypeT::F32, Element::TypeT::S32, DIM_5, ResamplingAlg::NEAREST_NEIGHBOR}, NearestNormal<float, int32_t>},
    {{Element::TypeT::F32, Element::TypeT::S8, DIM_5, ResamplingAlg::NEAREST_NEIGHBOR}, NearestNormal<float, int8_t>},
    {{Element::TypeT::F32, Element::TypeT::U8, DIM_5, ResamplingAlg::NEAREST_NEIGHBOR}, NearestNormal<float, uint8_t>},

    {{Element::TypeT::F32, Element::TypeT::F32, DIM_3, ResamplingAlg::LINEAR}, LinearNormal<float, float>},
    {{Element::TypeT::F32, Element::TypeT::F16, DIM_3, ResamplingAlg::LINEAR}, LinearNormal<float, __fp16>},
    {{Element::TypeT::F32, Element::TypeT::BF16, DIM_3, ResamplingAlg::LINEAR}, LinearNormal<float, __bf16>},
    {{Element::TypeT::F32, Element::TypeT::S32, DIM_3, ResamplingAlg::LINEAR}, LinearNormal<float, int32_t>},
    {{Element::TypeT::F32, Element::TypeT::S8, DIM_3, ResamplingAlg::LINEAR}, LinearNormal<float, int8_t>},
    {{Element::TypeT::F32, Element::TypeT::U8, DIM_3, ResamplingAlg::LINEAR}, LinearNormal<float, uint8_t>},

    {{Element::TypeT::F32, Element::TypeT::F32, DIM_4, ResamplingAlg::LINEAR}, BilinearNormal<float, float>},
    {{Element::TypeT::F32, Element::TypeT::F16, DIM_4, ResamplingAlg::LINEAR}, BilinearNormal<float, __fp16>},
    {{Element::TypeT::F32, Element::TypeT::BF16, DIM_4, ResamplingAlg::LINEAR}, BilinearNormal<float, __bf16>},
    {{Element::TypeT::F32, Element::TypeT::S32, DIM_4, ResamplingAlg::LINEAR}, BilinearNormal<float, int32_t>},
    {{Element::TypeT::F32, Element::TypeT::S8, DIM_4, ResamplingAlg::LINEAR}, BilinearNormal<float, int8_t>},
    {{Element::TypeT::F32, Element::TypeT::U8, DIM_4, ResamplingAlg::LINEAR}, BilinearNormal<float, uint8_t>},

    {{Element::TypeT::F32, Element::TypeT::F32, DIM_5, ResamplingAlg::LINEAR}, TrilinearNormal<float, float>},
    {{Element::TypeT::F32, Element::TypeT::F16, DIM_5, ResamplingAlg::LINEAR}, TrilinearNormal<float, __fp16>},
    {{Element::TypeT::F32, Element::TypeT::BF16, DIM_5, ResamplingAlg::LINEAR}, TrilinearNormal<float, __bf16>},
    {{Element::TypeT::F32, Element::TypeT::S32, DIM_5, ResamplingAlg::LINEAR}, TrilinearNormal<float, int32_t>},
    {{Element::TypeT::F32, Element::TypeT::S8, DIM_5, ResamplingAlg::LINEAR}, TrilinearNormal<float, int8_t>},
    {{Element::TypeT::F32, Element::TypeT::U8, DIM_5, ResamplingAlg::LINEAR}, TrilinearNormal<float, uint8_t>},
    // F16
    {{Element::TypeT::F16, Element::TypeT::F32, DIM_5, ResamplingAlg::NEAREST_NEIGHBOR}, NearestNormal<__fp16, float>},
    {{Element::TypeT::F16, Element::TypeT::F16, DIM_5, ResamplingAlg::NEAREST_NEIGHBOR}, NearestNormal<__fp16, __fp16>},
    {{Element::TypeT::F16, Element::TypeT::BF16, DIM_5, ResamplingAlg::NEAREST_NEIGHBOR},
        NearestNormal<__fp16, __bf16>},
    {{Element::TypeT::F16, Element::TypeT::S32, DIM_5, ResamplingAlg::NEAREST_NEIGHBOR},
        NearestNormal<__fp16, int32_t>},
    {{Element::TypeT::F16, Element::TypeT::S8, DIM_5, ResamplingAlg::NEAREST_NEIGHBOR}, NearestNormal<__fp16, int8_t>},
    {{Element::TypeT::F16, Element::TypeT::U8, DIM_5, ResamplingAlg::NEAREST_NEIGHBOR}, NearestNormal<__fp16, uint8_t>},

    {{Element::TypeT::F16, Element::TypeT::F32, DIM_3, ResamplingAlg::LINEAR}, LinearNormal<__fp16, float>},
    {{Element::TypeT::F16, Element::TypeT::F16, DIM_3, ResamplingAlg::LINEAR}, LinearNormal<__fp16, __fp16>},
    {{Element::TypeT::F16, Element::TypeT::BF16, DIM_3, ResamplingAlg::LINEAR}, LinearNormal<__fp16, __bf16>},
    {{Element::TypeT::F16, Element::TypeT::S32, DIM_3, ResamplingAlg::LINEAR}, LinearNormal<__fp16, int32_t>},
    {{Element::TypeT::F16, Element::TypeT::S8, DIM_3, ResamplingAlg::LINEAR}, LinearNormal<__fp16, int8_t>},
    {{Element::TypeT::F16, Element::TypeT::U8, DIM_3, ResamplingAlg::LINEAR}, LinearNormal<__fp16, uint8_t>},

    {{Element::TypeT::F16, Element::TypeT::F32, DIM_4, ResamplingAlg::LINEAR}, BilinearNormal<__fp16, float>},
    {{Element::TypeT::F16, Element::TypeT::F16, DIM_4, ResamplingAlg::LINEAR}, BilinearNormal<__fp16, __fp16>},
    {{Element::TypeT::F16, Element::TypeT::BF16, DIM_4, ResamplingAlg::LINEAR}, BilinearNormal<__fp16, __bf16>},
    {{Element::TypeT::F16, Element::TypeT::S32, DIM_4, ResamplingAlg::LINEAR}, BilinearNormal<__fp16, int32_t>},
    {{Element::TypeT::F16, Element::TypeT::S8, DIM_4, ResamplingAlg::LINEAR}, BilinearNormal<__fp16, int8_t>},
    {{Element::TypeT::F16, Element::TypeT::U8, DIM_4, ResamplingAlg::LINEAR}, BilinearNormal<__fp16, uint8_t>},

    {{Element::TypeT::F16, Element::TypeT::F32, DIM_5, ResamplingAlg::LINEAR}, TrilinearNormal<__fp16, float>},
    {{Element::TypeT::F16, Element::TypeT::F16, DIM_5, ResamplingAlg::LINEAR}, TrilinearNormal<__fp16, __fp16>},
    {{Element::TypeT::F16, Element::TypeT::BF16, DIM_5, ResamplingAlg::LINEAR}, TrilinearNormal<__fp16, __bf16>},
    {{Element::TypeT::F16, Element::TypeT::S32, DIM_5, ResamplingAlg::LINEAR}, TrilinearNormal<__fp16, int32_t>},
    {{Element::TypeT::F16, Element::TypeT::S8, DIM_5, ResamplingAlg::LINEAR}, TrilinearNormal<__fp16, int8_t>},
    {{Element::TypeT::F16, Element::TypeT::U8, DIM_5, ResamplingAlg::LINEAR}, TrilinearNormal<__fp16, uint8_t>},
    // BF16
    {{Element::TypeT::BF16, Element::TypeT::F32, DIM_5, ResamplingAlg::NEAREST_NEIGHBOR}, NearestNormal<__bf16, float>},
    {{Element::TypeT::BF16, Element::TypeT::F16, DIM_5, ResamplingAlg::NEAREST_NEIGHBOR},
        NearestNormal<__bf16, __fp16>},
    {{Element::TypeT::BF16, Element::TypeT::BF16, DIM_5, ResamplingAlg::NEAREST_NEIGHBOR},
        NearestNormal<__bf16, __bf16>},
    {{Element::TypeT::BF16, Element::TypeT::S32, DIM_5, ResamplingAlg::NEAREST_NEIGHBOR},
        NearestNormal<__bf16, int32_t>},
    {{Element::TypeT::BF16, Element::TypeT::S8, DIM_5, ResamplingAlg::NEAREST_NEIGHBOR}, NearestNormal<__bf16, int8_t>},
    {{Element::TypeT::BF16, Element::TypeT::U8, DIM_5, ResamplingAlg::NEAREST_NEIGHBOR},
        NearestNormal<__bf16, uint8_t>},

    {{Element::TypeT::BF16, Element::TypeT::F32, DIM_3, ResamplingAlg::LINEAR}, LinearNormal<__bf16, float>},
    {{Element::TypeT::BF16, Element::TypeT::F16, DIM_3, ResamplingAlg::LINEAR}, LinearNormal<__bf16, __fp16>},
    {{Element::TypeT::BF16, Element::TypeT::BF16, DIM_3, ResamplingAlg::LINEAR}, LinearNormal<__bf16, __bf16>},
    {{Element::TypeT::BF16, Element::TypeT::S32, DIM_3, ResamplingAlg::LINEAR}, LinearNormal<__bf16, int32_t>},
    {{Element::TypeT::BF16, Element::TypeT::S8, DIM_3, ResamplingAlg::LINEAR}, LinearNormal<__bf16, int8_t>},
    {{Element::TypeT::BF16, Element::TypeT::U8, DIM_3, ResamplingAlg::LINEAR}, LinearNormal<__bf16, uint8_t>},

    {{Element::TypeT::BF16, Element::TypeT::F32, DIM_4, ResamplingAlg::LINEAR}, BilinearNormal<__bf16, float>},
    {{Element::TypeT::BF16, Element::TypeT::F16, DIM_4, ResamplingAlg::LINEAR}, BilinearNormal<__bf16, __fp16>},
    {{Element::TypeT::BF16, Element::TypeT::BF16, DIM_4, ResamplingAlg::LINEAR}, BilinearNormal<__bf16, __bf16>},
    {{Element::TypeT::BF16, Element::TypeT::S32, DIM_4, ResamplingAlg::LINEAR}, BilinearNormal<__bf16, int32_t>},
    {{Element::TypeT::BF16, Element::TypeT::S8, DIM_4, ResamplingAlg::LINEAR}, BilinearNormal<__bf16, int8_t>},
    {{Element::TypeT::BF16, Element::TypeT::U8, DIM_4, ResamplingAlg::LINEAR}, BilinearNormal<__bf16, uint8_t>},

    {{Element::TypeT::BF16, Element::TypeT::F32, DIM_5, ResamplingAlg::LINEAR}, TrilinearNormal<__bf16, float>},
    {{Element::TypeT::BF16, Element::TypeT::F16, DIM_5, ResamplingAlg::LINEAR}, TrilinearNormal<__bf16, __fp16>},
    {{Element::TypeT::BF16, Element::TypeT::BF16, DIM_5, ResamplingAlg::LINEAR}, TrilinearNormal<__bf16, __bf16>},
    {{Element::TypeT::BF16, Element::TypeT::S32, DIM_5, ResamplingAlg::LINEAR}, TrilinearNormal<__bf16, int32_t>},
    {{Element::TypeT::BF16, Element::TypeT::S8, DIM_5, ResamplingAlg::LINEAR}, TrilinearNormal<__bf16, int8_t>},
    {{Element::TypeT::BF16, Element::TypeT::U8, DIM_5, ResamplingAlg::LINEAR}, TrilinearNormal<__bf16, uint8_t>},
    // S32
    {{Element::TypeT::S32, Element::TypeT::F32, DIM_5, ResamplingAlg::NEAREST_NEIGHBOR}, NearestNormal<int32_t, float>},
    {{Element::TypeT::S32, Element::TypeT::F16, DIM_5, ResamplingAlg::NEAREST_NEIGHBOR},
        NearestNormal<int32_t, __fp16>},
    {{Element::TypeT::S32, Element::TypeT::BF16, DIM_5, ResamplingAlg::NEAREST_NEIGHBOR},
        NearestNormal<int32_t, __bf16>},
    {{Element::TypeT::S32, Element::TypeT::S32, DIM_5, ResamplingAlg::NEAREST_NEIGHBOR},
        NearestNormal<int32_t, int32_t>},
    {{Element::TypeT::S32, Element::TypeT::S8, DIM_5, ResamplingAlg::NEAREST_NEIGHBOR}, NearestNormal<int32_t, int8_t>},
    {{Element::TypeT::S32, Element::TypeT::U8, DIM_5, ResamplingAlg::NEAREST_NEIGHBOR},
        NearestNormal<int32_t, uint8_t>},

    {{Element::TypeT::S32, Element::TypeT::F32, DIM_3, ResamplingAlg::LINEAR}, LinearNormal<int32_t, float>},
    {{Element::TypeT::S32, Element::TypeT::F16, DIM_3, ResamplingAlg::LINEAR}, LinearNormal<int32_t, __fp16>},
    {{Element::TypeT::S32, Element::TypeT::BF16, DIM_3, ResamplingAlg::LINEAR}, LinearNormal<int32_t, __bf16>},
    {{Element::TypeT::S32, Element::TypeT::S32, DIM_3, ResamplingAlg::LINEAR}, LinearNormal<int32_t, int32_t>},
    {{Element::TypeT::S32, Element::TypeT::S8, DIM_3, ResamplingAlg::LINEAR}, LinearNormal<int32_t, int8_t>},
    {{Element::TypeT::S32, Element::TypeT::U8, DIM_3, ResamplingAlg::LINEAR}, LinearNormal<int32_t, uint8_t>},

    {{Element::TypeT::S32, Element::TypeT::F32, DIM_4, ResamplingAlg::LINEAR}, BilinearNormal<int32_t, float>},
    {{Element::TypeT::S32, Element::TypeT::F16, DIM_4, ResamplingAlg::LINEAR}, BilinearNormal<int32_t, __fp16>},
    {{Element::TypeT::S32, Element::TypeT::BF16, DIM_4, ResamplingAlg::LINEAR}, BilinearNormal<int32_t, __bf16>},
    {{Element::TypeT::S32, Element::TypeT::S32, DIM_4, ResamplingAlg::LINEAR}, BilinearNormal<int32_t, int32_t>},
    {{Element::TypeT::S32, Element::TypeT::S8, DIM_4, ResamplingAlg::LINEAR}, BilinearNormal<int32_t, int8_t>},
    {{Element::TypeT::S32, Element::TypeT::U8, DIM_4, ResamplingAlg::LINEAR}, BilinearNormal<int32_t, uint8_t>},

    {{Element::TypeT::S32, Element::TypeT::F32, DIM_5, ResamplingAlg::LINEAR}, TrilinearNormal<int32_t, float>},
    {{Element::TypeT::S32, Element::TypeT::F16, DIM_5, ResamplingAlg::LINEAR}, TrilinearNormal<int32_t, __fp16>},
    {{Element::TypeT::S32, Element::TypeT::BF16, DIM_5, ResamplingAlg::LINEAR}, TrilinearNormal<int32_t, __bf16>},
    {{Element::TypeT::S32, Element::TypeT::S32, DIM_5, ResamplingAlg::LINEAR}, TrilinearNormal<int32_t, int32_t>},
    {{Element::TypeT::S32, Element::TypeT::S8, DIM_5, ResamplingAlg::LINEAR}, TrilinearNormal<int32_t, int8_t>},
    {{Element::TypeT::S32, Element::TypeT::U8, DIM_5, ResamplingAlg::LINEAR}, TrilinearNormal<int32_t, uint8_t>},
    // S8
    {{Element::TypeT::S8, Element::TypeT::F32, DIM_5, ResamplingAlg::NEAREST_NEIGHBOR}, NearestNormal<int8_t, float>},
    {{Element::TypeT::S8, Element::TypeT::F16, DIM_5, ResamplingAlg::NEAREST_NEIGHBOR}, NearestNormal<int8_t, __fp16>},
    {{Element::TypeT::S8, Element::TypeT::BF16, DIM_5, ResamplingAlg::NEAREST_NEIGHBOR}, NearestNormal<int8_t, __bf16>},
    {{Element::TypeT::S8, Element::TypeT::S32, DIM_5, ResamplingAlg::NEAREST_NEIGHBOR}, NearestNormal<int8_t, int32_t>},
    {{Element::TypeT::S8, Element::TypeT::S8, DIM_5, ResamplingAlg::NEAREST_NEIGHBOR}, NearestNormal<int8_t, int8_t>},
    {{Element::TypeT::S8, Element::TypeT::U8, DIM_5, ResamplingAlg::NEAREST_NEIGHBOR}, NearestNormal<int8_t, uint8_t>},

    {{Element::TypeT::S8, Element::TypeT::F32, DIM_3, ResamplingAlg::LINEAR}, LinearNormal<int8_t, float>},
    {{Element::TypeT::S8, Element::TypeT::F16, DIM_3, ResamplingAlg::LINEAR}, LinearNormal<int8_t, __fp16>},
    {{Element::TypeT::S8, Element::TypeT::BF16, DIM_3, ResamplingAlg::LINEAR}, LinearNormal<int8_t, __bf16>},
    {{Element::TypeT::S8, Element::TypeT::S32, DIM_3, ResamplingAlg::LINEAR}, LinearNormal<int8_t, int32_t>},
    {{Element::TypeT::S8, Element::TypeT::S8, DIM_3, ResamplingAlg::LINEAR}, LinearNormal<int8_t, int8_t>},
    {{Element::TypeT::S8, Element::TypeT::U8, DIM_3, ResamplingAlg::LINEAR}, LinearNormal<int8_t, uint8_t>},

    {{Element::TypeT::S8, Element::TypeT::F32, DIM_4, ResamplingAlg::LINEAR}, BilinearNormal<int8_t, float>},
    {{Element::TypeT::S8, Element::TypeT::F16, DIM_4, ResamplingAlg::LINEAR}, BilinearNormal<int8_t, __fp16>},
    {{Element::TypeT::S8, Element::TypeT::BF16, DIM_4, ResamplingAlg::LINEAR}, BilinearNormal<int8_t, __bf16>},
    {{Element::TypeT::S8, Element::TypeT::S32, DIM_4, ResamplingAlg::LINEAR}, BilinearNormal<int8_t, int32_t>},
    {{Element::TypeT::S8, Element::TypeT::S8, DIM_4, ResamplingAlg::LINEAR}, BilinearNormal<int8_t, int8_t>},
    {{Element::TypeT::S8, Element::TypeT::U8, DIM_4, ResamplingAlg::LINEAR}, BilinearNormal<int8_t, uint8_t>},

    {{Element::TypeT::S8, Element::TypeT::F32, DIM_5, ResamplingAlg::LINEAR}, TrilinearNormal<int8_t, float>},
    {{Element::TypeT::S8, Element::TypeT::F16, DIM_5, ResamplingAlg::LINEAR}, TrilinearNormal<int8_t, __fp16>},
    {{Element::TypeT::S8, Element::TypeT::BF16, DIM_5, ResamplingAlg::LINEAR}, TrilinearNormal<int8_t, __bf16>},
    {{Element::TypeT::S8, Element::TypeT::S32, DIM_5, ResamplingAlg::LINEAR}, TrilinearNormal<int8_t, int32_t>},
    {{Element::TypeT::S8, Element::TypeT::S8, DIM_5, ResamplingAlg::LINEAR}, TrilinearNormal<int8_t, int8_t>},
    {{Element::TypeT::S8, Element::TypeT::U8, DIM_5, ResamplingAlg::LINEAR}, TrilinearNormal<int8_t, uint8_t>},
    // U8
    {{Element::TypeT::U8, Element::TypeT::F32, DIM_5, ResamplingAlg::NEAREST_NEIGHBOR}, NearestNormal<uint8_t, float>},
    {{Element::TypeT::U8, Element::TypeT::F16, DIM_5, ResamplingAlg::NEAREST_NEIGHBOR}, NearestNormal<uint8_t, __fp16>},
    {{Element::TypeT::U8, Element::TypeT::BF16, DIM_5, ResamplingAlg::NEAREST_NEIGHBOR},
        NearestNormal<uint8_t, __bf16>},
    {{Element::TypeT::U8, Element::TypeT::S32, DIM_5, ResamplingAlg::NEAREST_NEIGHBOR},
        NearestNormal<uint8_t, int32_t>},
    {{Element::TypeT::U8, Element::TypeT::S8, DIM_5, ResamplingAlg::NEAREST_NEIGHBOR}, NearestNormal<uint8_t, int8_t>},
    {{Element::TypeT::U8, Element::TypeT::U8, DIM_5, ResamplingAlg::NEAREST_NEIGHBOR}, NearestNormal<uint8_t, uint8_t>},

    {{Element::TypeT::U8, Element::TypeT::F32, DIM_3, ResamplingAlg::LINEAR}, LinearNormal<uint8_t, float>},
    {{Element::TypeT::U8, Element::TypeT::F16, DIM_3, ResamplingAlg::LINEAR}, LinearNormal<uint8_t, __fp16>},
    {{Element::TypeT::U8, Element::TypeT::BF16, DIM_3, ResamplingAlg::LINEAR}, LinearNormal<uint8_t, __bf16>},
    {{Element::TypeT::U8, Element::TypeT::S32, DIM_3, ResamplingAlg::LINEAR}, LinearNormal<uint8_t, int32_t>},
    {{Element::TypeT::U8, Element::TypeT::S8, DIM_3, ResamplingAlg::LINEAR}, LinearNormal<uint8_t, int8_t>},
    {{Element::TypeT::U8, Element::TypeT::U8, DIM_3, ResamplingAlg::LINEAR}, LinearNormal<uint8_t, uint8_t>},

    {{Element::TypeT::U8, Element::TypeT::F32, DIM_4, ResamplingAlg::LINEAR}, BilinearNormal<uint8_t, float>},
    {{Element::TypeT::U8, Element::TypeT::F16, DIM_4, ResamplingAlg::LINEAR}, BilinearNormal<uint8_t, __fp16>},
    {{Element::TypeT::U8, Element::TypeT::BF16, DIM_4, ResamplingAlg::LINEAR}, BilinearNormal<uint8_t, __bf16>},
    {{Element::TypeT::U8, Element::TypeT::S32, DIM_4, ResamplingAlg::LINEAR}, BilinearNormal<uint8_t, int32_t>},
    {{Element::TypeT::U8, Element::TypeT::S8, DIM_4, ResamplingAlg::LINEAR}, BilinearNormal<uint8_t, int8_t>},
    {{Element::TypeT::U8, Element::TypeT::U8, DIM_4, ResamplingAlg::LINEAR}, BilinearNormal<uint8_t, uint8_t>},

    {{Element::TypeT::U8, Element::TypeT::F32, DIM_5, ResamplingAlg::LINEAR}, TrilinearNormal<uint8_t, float>},
    {{Element::TypeT::U8, Element::TypeT::F16, DIM_5, ResamplingAlg::LINEAR}, TrilinearNormal<uint8_t, __fp16>},
    {{Element::TypeT::U8, Element::TypeT::BF16, DIM_5, ResamplingAlg::LINEAR}, TrilinearNormal<uint8_t, __bf16>},
    {{Element::TypeT::U8, Element::TypeT::S32, DIM_5, ResamplingAlg::LINEAR}, TrilinearNormal<uint8_t, int32_t>},
    {{Element::TypeT::U8, Element::TypeT::S8, DIM_5, ResamplingAlg::LINEAR}, TrilinearNormal<uint8_t, int8_t>},
    {{Element::TypeT::U8, Element::TypeT::U8, DIM_5, ResamplingAlg::LINEAR}, TrilinearNormal<uint8_t, uint8_t>}
};

} // Resampling
} // Detail
} // KuDNN

#endif //  KUDNN_RESAMPLING_NORMAL_KERNELS_HPP
