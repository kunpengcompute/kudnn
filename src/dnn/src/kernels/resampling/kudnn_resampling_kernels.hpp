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

#ifndef KUDNN_RESAMPLING_KERNELS_HPP
#define KUDNN_RESAMPLING_KERNELS_HPP

#include "kudnn_resampling_utils.hpp"

namespace KuDNN {
namespace Detail {
namespace Resampling {

template<typename DataType>
inline void LinearVCAxbKernel(const DataType *src, DataType *dst,
    const std::vector<IntType> &offsets, const std::vector<float> &weights, const svbool_t &pg)
{
    svfloat32_t res = svdup_f32(F32_ZERO);
    for (SizeType i = 0; i < offsets.size(); i++) {
        res = svmla_n_f32_m(pg, res, LoadSvF32FromAddress(pg, src + offsets[i]), weights[i]);
    }
    StoreSvF32ToAddress(pg, dst, res);
}

template<typename DataType>
inline void LinearVCAbxKernel(const DataType *src, DataType *dst,
    const std::vector<IntType> &offsets, const std::vector<float> &weights, const svbool_t &pg,
    const ResamplingCoeffs<int32_t> &coeffs, int32_t dOD, int32_t dOH, int32_t ow)
{
    svint32_t svOffset0 = svld1_s32(pg, &coeffs.idx0[0] + dOD + dOH + ow);
    svint32_t svOffset1 = svld1_s32(pg, &coeffs.idx1[0] + dOD + dOH + ow);
    svfloat32_t svWeight0 = svld1_f32(pg, &coeffs.wei0[0] + dOD + dOH + ow);
    svfloat32_t svWeight1 = svld1_f32(pg, &coeffs.wei1[0] + dOD + dOH + ow);

    svfloat32_t res = svdup_f32(F32_ZERO);
    for (SizeType i = 0; i < offsets.size(); i++) {
        svfloat32_t svWeights0 = svmul_n_f32_m(pg, svWeight0, weights[i]);
        svfloat32_t svWeights1 = svmul_n_f32_m(pg, svWeight1, weights[i]);

        svfloat32_t r0 = svmul_f32_m(pg, GatherLoadSvF32FromAddress(pg, src + offsets[i], svOffset0), svWeights0);
        svfloat32_t r1 = svmul_f32_m(pg, GatherLoadSvF32FromAddress(pg, src + offsets[i], svOffset1), svWeights1);
        res = svadd_f32_m(pg, res, svadd_f32_m(pg, r0, r1));
    }
    StoreSvF32ToAddress(pg, dst, res);
}

template <typename DataType>
void LinearVC(const Shape &batchSplitDims, const Shape &stride, const Shape &dimsDst,
    const StridesInfo &sInfo, const void *src, void *dst, const ResamplingCoeffs<int32_t> &coeffs, bool isAxb)
{
    SplitDimInfo<int32_t> sddInfo(batchSplitDims, stride, isAxb);
    int32_t dOD = dimsDst[IDX_2];
    int32_t dOH = dimsDst[IDX_3];
    std::vector<std::vector<IntType>> offsetsStk;
    std::vector<std::vector<float>> weightsStk;
    LinearStksInit(IDX_2, offsetsStk, weightsStk);

    if (!isAxb) {
        for (int32_t mb = sddInfo.mbStart; mb < sddInfo.mbEnd; mb++) {
            for (int32_t c = sddInfo.cStart; c < sddInfo.cEnd; c++) {
                offsetsStk[IDX_0][IDX_0] = mb * sInfo.strideIMB + c * sInfo.strideIC;
                for (int32_t ow = sddInfo.wStart; ow < sddInfo.wEnd; ow += svcntw()) {
                    IntType dstOffset = mb * sInfo.strideOMB + c * sInfo.strideOC + ow;
                    svbool_t pg = svwhilelt_b32_s32(ow, sddInfo.wEnd);
                    LinearVCAbxKernel<DataType>(static_cast<const DataType *>(src),
                        static_cast<DataType *>(dst) + dstOffset, offsetsStk[IDX_0], weightsStk[IDX_0], pg,
                        coeffs, dOD, dOH, ow);
                }
            }
        }
        return ;
    }

    for (int32_t mb = sddInfo.mbStart; mb < sddInfo.mbEnd; mb++) {
        offsetsStk[IDX_0][IDX_0] = mb * sInfo.strideIMB;
        for (int32_t ow = sddInfo.wStart; ow < sddInfo.wEnd; ow++) {
            Linearize(coeffs, dOD + dOH + ow, sInfo.strideIW, offsetsStk, weightsStk, IDX_1);
            for (int32_t c = sddInfo.cStart; c < sddInfo.cEnd; c += svcntw()) {
                IntType dstOffset = mb * sInfo.strideOMB + ow * sInfo.strideOW + c;
                svbool_t pg = svwhilelt_b32_s32(c, sddInfo.cEnd);
                LinearVCAxbKernel<DataType>(static_cast<const DataType *>(src) + c,
                    static_cast<DataType *>(dst) + dstOffset, offsetsStk[IDX_1], weightsStk[IDX_1], pg);
            }
        }
    }
}

template <typename DataType>
void BilinearVC(const Shape &batchSplitDims, const Shape &stride, const Shape &dimsDst,
    const StridesInfo &sInfo, const void *src, void *dst, const ResamplingCoeffs<int32_t> &coeffs, bool isAxb)
{
    SplitDimInfo<int32_t> sddInfo(batchSplitDims, stride, isAxb);
    int32_t dOD = dimsDst[IDX_2];
    int32_t dOH = dimsDst[IDX_3];
    std::vector<std::vector<IntType>> offsetsStk;
    std::vector<std::vector<float>> weightsStk;
    LinearStksInit(IDX_3, offsetsStk, weightsStk);

    if (!isAxb) {
        for (int32_t mb = sddInfo.mbStart; mb < sddInfo.mbEnd; mb++) {
            for (int32_t c = sddInfo.cStart; c < sddInfo.cEnd; c++) {
                offsetsStk[IDX_0][IDX_0] = mb * sInfo.strideIMB + c * sInfo.strideIC;
                for (int32_t oh = sddInfo.hStart; oh < sddInfo.hEnd; oh++) {
                    Linearize(coeffs, dOD + oh, sInfo.strideIH, offsetsStk, weightsStk, IDX_1);
                    for (int32_t ow = sddInfo.wStart; ow < sddInfo.wEnd; ow += svcntw()) {
                        IntType dstOffset = mb * sInfo.strideOMB + c * sInfo.strideOC + oh * sInfo.strideOH + ow;
                        svbool_t pg = svwhilelt_b32_s32(ow, sddInfo.wEnd);
                        LinearVCAbxKernel<DataType>(static_cast<const DataType *>(src),
                            static_cast<DataType *>(dst) + dstOffset, offsetsStk[IDX_1], weightsStk[IDX_1], pg,
                            coeffs, dOD, dOH, ow);
                    }
                }
            }
        }
        return ;
    }

    for (int32_t mb = sddInfo.mbStart; mb < sddInfo.mbEnd; mb++) {
        offsetsStk[IDX_0][IDX_0] = mb * sInfo.strideIMB;
        for (int32_t oh = sddInfo.hStart; oh < sddInfo.hEnd; oh++) {
            Linearize(coeffs, dOD + oh, sInfo.strideIH, offsetsStk, weightsStk, IDX_1);
            for (int32_t ow = sddInfo.wStart; ow < sddInfo.wEnd; ow++) {
                Linearize(coeffs, dOD + dOH + ow, sInfo.strideIW, offsetsStk, weightsStk, IDX_2);
                for (int32_t c = sddInfo.cStart; c < sddInfo.cEnd; c += svcntw()) {
                    IntType dstOffset = mb * sInfo.strideOMB + oh * sInfo.strideOH + ow * sInfo.strideOW + c;
                    svbool_t pg = svwhilelt_b32_s32(c, sddInfo.cEnd);
                    LinearVCAxbKernel<DataType>(static_cast<const DataType *>(src) + c,
                        static_cast<DataType *>(dst) + dstOffset, offsetsStk[IDX_2], weightsStk[IDX_2], pg);
                }
            }
        }
    }
}

template <typename DataType>
void TrilinearVC(const Shape &batchSplitDims, const Shape &stride, const Shape &dimsDst,
    const StridesInfo &sInfo, const void *src, void *dst, const ResamplingCoeffs<int32_t> &coeffs, bool isAxb)
{
    SplitDimInfo<int32_t> sddInfo(batchSplitDims, stride, isAxb);
    int32_t dOD = dimsDst[IDX_2];
    int32_t dOH = dimsDst[IDX_3];
    std::vector<std::vector<IntType>> offsetsStk;
    std::vector<std::vector<float>> weightsStk;
    LinearStksInit(IDX_4, offsetsStk, weightsStk);

    if (!isAxb) {
        for (int32_t mb = sddInfo.mbStart; mb < sddInfo.mbEnd; mb++) {
            for (int32_t c = sddInfo.cStart; c < sddInfo.cEnd; c++) {
                offsetsStk[IDX_0][IDX_0] = mb * sInfo.strideIMB + c * sInfo.strideIC;
                for (int32_t od = sddInfo.dStart; od < sddInfo.dEnd; od++) {
                    Linearize(coeffs, od, sInfo.strideID, offsetsStk, weightsStk, IDX_1);
                    for (int32_t oh = sddInfo.hStart; oh < sddInfo.hEnd; oh++) {
                        Linearize(coeffs, dOD + oh, sInfo.strideIH, offsetsStk, weightsStk, IDX_2);
                        for (int32_t ow = sddInfo.wStart; ow < sddInfo.wEnd; ow += svcntw()) {
                            IntType dstOffset = mb * sInfo.strideOMB + c * sInfo.strideOC
                                            + od * sInfo.strideOD + oh * sInfo.strideOH + ow;
                            svbool_t pg = svwhilelt_b32_s32(ow, sddInfo.wEnd);
                            LinearVCAbxKernel<DataType>(static_cast<const DataType *>(src),
                                static_cast<DataType *>(dst) + dstOffset, offsetsStk[IDX_2], weightsStk[IDX_2], pg,
                                coeffs, dOD, dOH, ow);
                        }
                    }
                }
            }
        }
        return ;
    }

    for (int32_t mb = sddInfo.mbStart; mb < sddInfo.mbEnd; mb++) {
        offsetsStk[IDX_0][IDX_0] = mb * sInfo.strideIMB;
        for (int32_t od = sddInfo.dStart; od < sddInfo.dEnd; od++) {
            Linearize(coeffs, od, sInfo.strideID, offsetsStk, weightsStk, IDX_1);
            for (int32_t oh = sddInfo.hStart; oh < sddInfo.hEnd; oh++) {
                Linearize(coeffs, dOD + oh, sInfo.strideIH, offsetsStk, weightsStk, IDX_2);
                for (int32_t ow = sddInfo.wStart; ow < sddInfo.wEnd; ow++) {
                    Linearize(coeffs, dOD + dOH + ow, sInfo.strideIW, offsetsStk, weightsStk, IDX_3);
                    for (int32_t c = sddInfo.cStart; c < sddInfo.cEnd; c += svcntw()) {
                        IntType dstOffset = mb * sInfo.strideOMB + od * sInfo.strideOD
                                            + oh * sInfo.strideOH + ow * sInfo.strideOW + c;
                        svbool_t pg = svwhilelt_b32_s32(c, sddInfo.cEnd);
                        LinearVCAxbKernel<DataType>(static_cast<const DataType *>(src) + c,
                            static_cast<DataType *>(dst) + dstOffset, offsetsStk[IDX_3], weightsStk[IDX_3], pg);
                    }
                }
            }
        }
    }
}

template <typename DataType>
inline void NearestVCAbxKernel(int32_t owStart, int32_t owEnd, const DataType *src, DataType *dst,
    const ResamplingCoeffs<int32_t>& coeffs, int32_t dOD, int32_t dOH)
{
    for (int32_t ow = owStart; ow < owEnd; ow += svcntw()) {
        svbool_t pg = svwhilelt_b32_s32(ow, owEnd);
        svint32_t iws = svld1_s32(pg, &coeffs.outidx2Inidx[0] + dOD + dOH + ow);
        GatherLoadAndStoreSv<DataType>(pg, dst + ow, src, iws);
    }
}

template <typename DataType>
inline void NearestVCAxbKernel(int32_t cStart, int32_t cEnd, const DataType *src, DataType *dst)
{
    Service::Unused(src, dst);
    for (int32_t c = cStart; c < cEnd; c += svcntw()) {
        svbool_t pg = svwhilelt_b32_s32(c, cEnd);
        LoadAndStoreSv<DataType>(pg, dst + c, src + c);
    }
}

template <typename DataType>
void NearestVC(const Shape &batchSplitDims, const Shape &stride, const Shape &dimsDst,
    const StridesInfo &sInfo, const void *src, void *dst, const ResamplingCoeffs<int32_t> &coeffs, bool isAxb)
{
    SplitDimInfo<int32_t> sddInfo(batchSplitDims, stride, isAxb);
    int32_t dOD = dimsDst[IDX_2];
    int32_t dOH = dimsDst[IDX_3];

    if (isAxb) {
        for (int32_t mb = sddInfo.mbStart; mb < sddInfo.mbEnd; mb++) {
            for (int32_t od = sddInfo.dStart; od < sddInfo.dEnd; od++) {
                int32_t id = coeffs.outidx2Inidx[od];
                for (int32_t oh = sddInfo.hStart; oh < sddInfo.hEnd; oh++) {
                    int32_t ih = coeffs.outidx2Inidx[dOD + oh];
                    for (int32_t ow = sddInfo.wStart; ow < sddInfo.wEnd; ow++) {
                        int32_t iw = coeffs.outidx2Inidx[dOD + dOH + ow];
                        IntType outerSrcOff = mb * sInfo.strideIMB + id * sInfo.strideID
                                            + ih * sInfo.strideIH + iw * sInfo.strideIW;
                        IntType outerDstOff = mb * sInfo.strideOMB + od * sInfo.strideOD
                                            + oh * sInfo.strideOH + ow * sInfo.strideOW;
                        const DataType *typedSrc = static_cast<const DataType *>(src) + outerSrcOff;
                        DataType *typedDst = static_cast<DataType *>(dst) + outerDstOff;
                        NearestVCAxbKernel<DataType>(
                            sddInfo.cStart, sddInfo.cEnd, typedSrc, typedDst);
                    }
                }
            }
        }
        return ;
    }

    for (int32_t mb = sddInfo.mbStart; mb < sddInfo.mbEnd; mb++) {
        for (int32_t c = sddInfo.cStart; c < sddInfo.cEnd; c++) {
            for (int32_t od = sddInfo.dStart; od < sddInfo.dEnd; od++) {
                int32_t id = coeffs.outidx2Inidx[od];
                for (int32_t oh = sddInfo.hStart; oh < sddInfo.hEnd; oh++) {
                    int32_t ih = coeffs.outidx2Inidx[dOD + oh];
                    IntType outerSrcOff = mb * sInfo.strideIMB + c * sInfo.strideIC
                                        + id * sInfo.strideID + ih * sInfo.strideIH;
                    IntType outerDstOff = mb * sInfo.strideOMB + c * sInfo.strideOC
                                        + od * sInfo.strideOD + oh * sInfo.strideOH;
                    const DataType *typedSrc = static_cast<const DataType *>(src) + outerSrcOff;
                    DataType *typedDst = static_cast<DataType *>(dst) + outerDstOff;
                    NearestVCAbxKernel<DataType>(sddInfo.wStart, sddInfo.wEnd, typedSrc, typedDst, coeffs, dOD, dOH);
                }
            }
        }
    }
}

// std::function for NearestVC(), LinearVC(), BilinearVC() and TrilinearVC().
using VCKerFunc = std::function<void(const Shape &, const Shape &, const Shape &, const StridesInfo &,
                                      const void *, void *, const ResamplingCoeffs<int32_t> &, bool)>;

const std::unordered_map<VCKerParas, VCKerFunc, VCKerParasHash, VCKerParasEqual> VC_KER_MAP = {
    // NEAREST
    {{Element::TypeT::F32, DIM_5, ResamplingAlg::NEAREST_NEIGHBOR}, NearestVC<float>},
    {{Element::TypeT::F16, DIM_5, ResamplingAlg::NEAREST_NEIGHBOR}, NearestVC<__fp16>},
    {{Element::TypeT::BF16, DIM_5, ResamplingAlg::NEAREST_NEIGHBOR}, NearestVC<__bf16>},
    {{Element::TypeT::S32, DIM_5, ResamplingAlg::NEAREST_NEIGHBOR}, NearestVC<int32_t>},
    {{Element::TypeT::S8, DIM_5, ResamplingAlg::NEAREST_NEIGHBOR}, NearestVC<int8_t>},
    {{Element::TypeT::U8, DIM_5, ResamplingAlg::NEAREST_NEIGHBOR}, NearestVC<uint8_t>},

    // LINEAR
    {{Element::TypeT::F32, DIM_3, ResamplingAlg::LINEAR}, LinearVC<float>},
    {{Element::TypeT::F16, DIM_3, ResamplingAlg::LINEAR}, LinearVC<__fp16>},
    {{Element::TypeT::BF16, DIM_3, ResamplingAlg::LINEAR}, LinearVC<__bf16>},
    {{Element::TypeT::S32, DIM_3, ResamplingAlg::LINEAR}, LinearVC<int32_t>},
    {{Element::TypeT::S8, DIM_3, ResamplingAlg::LINEAR}, LinearVC<int8_t>},
    {{Element::TypeT::U8, DIM_3, ResamplingAlg::LINEAR}, LinearVC<uint8_t>},

    // BILINEAR
    {{Element::TypeT::F32, DIM_4, ResamplingAlg::LINEAR}, BilinearVC<float>},
    {{Element::TypeT::F16, DIM_4, ResamplingAlg::LINEAR}, BilinearVC<__fp16>},
    {{Element::TypeT::BF16, DIM_4, ResamplingAlg::LINEAR}, BilinearVC<__bf16>},
    {{Element::TypeT::S32, DIM_4, ResamplingAlg::LINEAR}, BilinearVC<int32_t>},
    {{Element::TypeT::S8, DIM_4, ResamplingAlg::LINEAR}, BilinearVC<int8_t>},
    {{Element::TypeT::U8, DIM_4, ResamplingAlg::LINEAR}, BilinearVC<uint8_t>},

    // TRILINEAR
    {{Element::TypeT::F32, DIM_5, ResamplingAlg::LINEAR}, TrilinearVC<float>},
    {{Element::TypeT::F16, DIM_5, ResamplingAlg::LINEAR}, TrilinearVC<__fp16>},
    {{Element::TypeT::BF16, DIM_5, ResamplingAlg::LINEAR}, TrilinearVC<__bf16>},
    {{Element::TypeT::S32, DIM_5, ResamplingAlg::LINEAR}, TrilinearVC<int32_t>},
    {{Element::TypeT::S8, DIM_5, ResamplingAlg::LINEAR}, TrilinearVC<int8_t>},
    {{Element::TypeT::U8, DIM_5, ResamplingAlg::LINEAR}, TrilinearVC<uint8_t>}
};

} // Resampling
} // Detail
} // KuDNN

#endif //  KUDNN_RESAMPLING_KERNELS_HPP
