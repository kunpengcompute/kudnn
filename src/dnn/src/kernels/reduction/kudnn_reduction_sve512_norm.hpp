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

#ifndef KUDNN_REDUCTION_VC512_NORM_HPP
#define KUDNN_REDUCTION_VC512_NORM_HPP
#ifdef KUDNNL_920F

#include <algorithm>
#include <arm_sve.h>
#include <cmath>

#include "kudnn.hpp"
#include "vector_wrappers.hpp"
#include "kernels/reduction/kudnn_reduction_kernels.hpp"

namespace KuDNN {
namespace Detail {
namespace Reduction {

const uint64_t pre6 = ((1ULL << 18) - 1) << 46;
const uint64_t pre10 = ((1ULL << 22) - 1) << 42;
constexpr int64_t val1023 = 1023;
const double ln2 = std::log(2.0);
const double val1D3 = 1.0/3;
constexpr double val1D2 = 0.5;
constexpr double val1 = 1.0;
constexpr uint64_t ones6 = 63;
constexpr uint64_t ones4 = 15;
constexpr size_t CACHE_LINE_SIZE = 64;

alignas(CACHE_LINE_SIZE) const double lnxPre6Table[] = {
    std::log(1.0),
    std::log(1.0 + 1.0 / 64),
    std::log(1.0 + 2.0 / 64),
    std::log(1.0 + 3.0 / 64),
    std::log(1.0 + 4.0 / 64),
    std::log(1.0 + 5.0 / 64),
    std::log(1.0 + 6.0 / 64),
    std::log(1.0 + 7.0 / 64),
    std::log(1.0 + 8.0 / 64),
    std::log(1.0 + 9.0 / 64),
    std::log(1.0 + 10.0 / 64),
    std::log(1.0 + 11.0 / 64),
    std::log(1.0 + 12.0 / 64),
    std::log(1.0 + 13.0 / 64),
    std::log(1.0 + 14.0 / 64),
    std::log(1.0 + 15.0 / 64),

    std::log(1.0 + 16.0 / 64),
    std::log(1.0 + 17.0 / 64),
    std::log(1.0 + 18.0 / 64),
    std::log(1.0 + 19.0 / 64),
    std::log(1.0 + 20.0 / 64),
    std::log(1.0 + 21.0 / 64),
    std::log(1.0 + 22.0 / 64),
    std::log(1.0 + 23.0 / 64),
    std::log(1.0 + 24.0 / 64),
    std::log(1.0 + 25.0 / 64),
    std::log(1.0 + 26.0 / 64),
    std::log(1.0 + 27.0 / 64),
    std::log(1.0 + 28.0 / 64),
    std::log(1.0 + 29.0 / 64),
    std::log(1.0 + 30.0 / 64),
    std::log(1.0 + 31.0 / 64),

    std::log(1.0 + 32.0 / 64),
    std::log(1.0 + 33.0 / 64),
    std::log(1.0 + 34.0 / 64),
    std::log(1.0 + 35.0 / 64),
    std::log(1.0 + 36.0 / 64),
    std::log(1.0 + 37.0 / 64),
    std::log(1.0 + 38.0 / 64),
    std::log(1.0 + 39.0 / 64),
    std::log(1.0 + 40.0 / 64),
    std::log(1.0 + 41.0 / 64),
    std::log(1.0 + 42.0 / 64),
    std::log(1.0 + 43.0 / 64),
    std::log(1.0 + 44.0 / 64),
    std::log(1.0 + 45.0 / 64),
    std::log(1.0 + 46.0 / 64),
    std::log(1.0 + 47.0 / 64),

    std::log(1.0 + 48.0 / 64),
    std::log(1.0 + 49.0 / 64),
    std::log(1.0 + 50.0 / 64),
    std::log(1.0 + 51.0 / 64),
    std::log(1.0 + 52.0 / 64),
    std::log(1.0 + 53.0 / 64),
    std::log(1.0 + 54.0 / 64),
    std::log(1.0 + 55.0 / 64),
    std::log(1.0 + 56.0 / 64),
    std::log(1.0 + 57.0 / 64),
    std::log(1.0 + 58.0 / 64),
    std::log(1.0 + 59.0 / 64),
    std::log(1.0 + 60.0 / 64),
    std::log(1.0 + 61.0 / 64),
    std::log(1.0 + 62.0 / 64),
    std::log(1.0 + 63.0 / 64),
};

alignas(CACHE_LINE_SIZE) const double expPre10Table[] = {
    1.0,
    std::pow(2, 1.0/1024),
    std::pow(2, 2.0/1024),
    std::pow(2, 3.0/1024),
    std::pow(2, 4.0/1024),
    std::pow(2, 5.0/1024),
    std::pow(2, 6.0/1024),
    std::pow(2, 7.0/1024),
    std::pow(2, 8.0/1024),
    std::pow(2, 9.0/1024),
    std::pow(2, 10.0/1024),
    std::pow(2, 11.0/1024),
    std::pow(2, 12.0/1024),
    std::pow(2, 13.0/1024),
    std::pow(2, 14.0/1024),
    std::pow(2, 15.0/1024),
};

template <typename InType>
inline void LoadReduc(const InType *src, svbool_t pg, svfloat32_t &res,
    const SizeType &loopIdx, const SizeType &loopRange)
{
    if constexpr (std::is_same_v<InType, float>) {
        res = svld1_f32(pg, src);
    } else if constexpr (std::is_same_v<InType, __fp16>) {
        svbool_t ldPg = svwhilelt_b16_u64(loopIdx, loopRange);
        svfloat16_t tmp = svld1_f16(ldPg, src);
        svfloat16_t zeros = svdup_n_f16(0);
        svfloat16_t tmp0 = svzip1(tmp, zeros);
        res = svcvt_f32_f16_x(pg, tmp0);
    } else if constexpr (std::is_same_v<InType, __bf16>) {
        svbool_t ldPg = svwhilelt_b16_u64(loopIdx, loopRange);
        svbfloat16_t tmp = svld1_bf16(ldPg, src);
        svbfloat16_t zeros = svdup_n_bf16(vcvth_bf16_f32(0.0f));
        svbfloat16_t tmp0 = svzip1(zeros, tmp);
        res = svreinterpret_f32_bf16(tmp0);
    }
}

template <typename OutType>
inline void StoreReduc(OutType *dst, svbool_t pg, const svfloat32_t &svDst)
{
    if constexpr (std::is_same_v<OutType, float>) {
        svst1_f32(pg, dst, svDst);
    } else if constexpr (std::is_same_v<OutType, __fp16>) {
        svfloat16_t tmp0 = svcvt_f16_f32_z(pg, svDst);
        svint32_t tmp1 = svreinterpret_s32_f16(tmp0);
        svst1h_s32(pg, reinterpret_cast<int16_t*>(dst), tmp1);
    } else if constexpr (std::is_same_v<OutType, __bf16>) {
        svbfloat16_t tmp0 = svcvt_bf16_f32_z(pg, svDst);
        svint32_t tmp1 = svreinterpret_s32_bf16(tmp0);
        svst1h_s32(pg, reinterpret_cast<int16_t*>(dst), tmp1);
    }
}

inline void VCLnF64(const svfloat64_t &src1, const svfloat64_t &src2, const svbool_t &pg1, const svbool_t &pg2,
    svfloat64_t &dst1, svfloat64_t &dst2)
{
    svint64_t CONST_1023 = svdup_n_s64(val1023);
    svfloat64_t CONST_LN2 = svdup_f64(ln2);
    svfloat64_t CONST_NE1 = svdup_f64(-val1);
    svfloat64_t CONST_NE1D2 = svdup_f64(-val1D2);
    svfloat64_t CONST_1D3 = svdup_f64(val1D3);
    svuint64_t CONST_PRE6 = svdup_n_u64(pre6);
    svuint64_t ONES_6 = svdup_n_u64(ones6);

    svint64_t m1 = svsub_s64_x(pg1, svasr_n_s64_x(pg1, svreinterpret_s64_f64(src1), 52), CONST_1023);
    svint64_t m2 = svsub_s64_x(pg2, svasr_n_s64_x(pg2, svreinterpret_s64_f64(src2), 52), CONST_1023);

    svfloat64_t k1 = svreinterpret_f64_s64(svsub_s64_x(pg1, svreinterpret_s64_f64(src1), svlsl_n_s64_x(pg1, m1, 52)));
    svfloat64_t k2 = svreinterpret_f64_s64(svsub_s64_x(pg2, svreinterpret_s64_f64(src2), svlsl_n_s64_x(pg2, m2, 52)));

    svfloat64_t x1 = svreinterpret_f64_u64(svand_u64_x(pg1, svreinterpret_u64_f64(k1), CONST_PRE6));
    svfloat64_t x2 = svreinterpret_f64_u64(svand_u64_x(pg2, svreinterpret_u64_f64(k2), CONST_PRE6));

    svfloat64_t semgx1 = svdiv_f64_x(pg1, k1, x1);
    svfloat64_t semgx2 = svdiv_f64_x(pg2, k2, x2);

    dst1 = svadd_f64_x(pg1, semgx1, CONST_NE1);
    dst2 = svadd_f64_x(pg2, semgx2, CONST_NE1);

    svfloat64_t alfa1 = svmul_f64_x(pg1, dst1, dst1);
    svfloat64_t alfa2 = svmul_f64_x(pg2, dst2, dst2);

    svfloat64_t beta1 = svmul_f64_x(pg1, alfa1, dst1);
    svfloat64_t beta2 = svmul_f64_x(pg2, alfa2, dst2);

    dst1 = svmla_f64_x(pg1, dst1, alfa1, CONST_NE1D2);
    dst2 = svmla_f64_x(pg2, dst2, alfa2, CONST_NE1D2);

    dst1 = svmla_f64_x(pg1, dst1, beta1, CONST_1D3);
    dst2 = svmla_f64_x(pg2, dst2, beta2, CONST_1D3);

    svuint64_t idx1 = svand_u64_x(pg1,
        svreinterpret_u64_s64(svasr_n_s64_x(pg1, svreinterpret_s64_f64(x1), 46)), ONES_6);
    svuint64_t idx2 = svand_u64_x(pg2,
        svreinterpret_u64_s64(svasr_n_s64_x(pg2, svreinterpret_s64_f64(x2), 46)), ONES_6);

    idx1 = svlsl_n_u64_x(pg1, idx1, IDX_3);
    idx2 = svlsl_n_u64_x(pg1, idx2, IDX_3);

    dst1 = svadd_f64_x(pg1, dst1, svld1_gather_u64offset_f64(pg1, lnxPre6Table, idx1));
    dst2 = svadd_f64_x(pg2, dst2, svld1_gather_u64offset_f64(pg2, lnxPre6Table, idx2));

    dst1 = svmla_f64_x(pg1, dst1, svcvt_f64_s64_x(pg1, m1), CONST_LN2);
    dst2 = svmla_f64_x(pg2, dst2, svcvt_f64_s64_x(pg2, m2), CONST_LN2);
}

inline svfloat32_t VCExpF64(const svfloat64_t &src1, const svfloat64_t &src2, const svbool_t &pg1, const svbool_t &pg2)
{
    svfloat64_t CONST_LN2 = svdup_n_f64(ln2);
    svfloat64_t CONST_1F = svdup_n_f64(val1);
    svuint64_t CONST_PRE10 = svdup_n_u64(pre10);
    svuint64_t ONES_4 = svdup_n_u64(ones4);
    svfloat64_t CONST_1D2 = svdup_n_f64(val1D2);

    svfloat64_t y1 = svdiv_f64_x(pg1, src1, CONST_LN2);
    svfloat64_t y2 = svdiv_f64_x(pg2, src2, CONST_LN2);

    svfloat64_t m1 = svrintm_f64_x(pg1, y1);
    svfloat64_t m2 = svrintm_f64_x(pg2, y2);

    svfloat64_t k1 = svsub_f64_x(pg1, y1, m1);
    svfloat64_t k2 = svsub_f64_x(pg2, y2, m2);

    k1 = svadd_f64_x(pg1, k1, CONST_1F);
    k2 = svadd_f64_x(pg2, k2, CONST_1F);

    svfloat64_t semg1t1 = svexpa_f64(svlsr_n_u64_x(pg1, svreinterpret_u64_f64(k1), 46));
    svfloat64_t semg1t2 = svexpa_f64(svlsr_n_u64_x(pg2, svreinterpret_u64_f64(k2), 46));

    svfloat64_t out1 = svreinterpret_f64_s64(svadd_s64_x(pg1, svreinterpret_s64_f64(CONST_1F),
        svlsl_n_s64_x(pg1, svcvt_s64_f64_x(pg1, m1), 52)));
    svfloat64_t out2 = svreinterpret_f64_s64(svadd_s64_x(pg2, svreinterpret_s64_f64(CONST_1F),
        svlsl_n_s64_x(pg2, svcvt_s64_f64_x(pg2, m2), 52)));

    svfloat64_t p1 = svreinterpret_f64_u64(svand_u64_x(pg1, svreinterpret_u64_f64(k1), CONST_PRE10));
    svfloat64_t p2 = svreinterpret_f64_u64(svand_u64_x(pg2, svreinterpret_u64_f64(k2), CONST_PRE10));

    svuint64_t id1 = svand_u64_x(pg1, svlsr_n_u64_x(pg1, svreinterpret_u64_f64(k1), 42), ONES_4);
    svuint64_t id2 = svand_u64_x(pg2, svlsr_n_u64_x(pg2, svreinterpret_u64_f64(k2), 42), ONES_4);

    id1 = svlsl_n_u64_x(pg1, id1, IDX_3);
    id2 = svlsl_n_u64_x(pg1, id2, IDX_3);
    svfloat64_t semg2t1 = svld1_gather_u64offset_f64(pg1, expPre10Table, id1);
    svfloat64_t semg2t2 = svld1_gather_u64offset_f64(pg2, expPre10Table, id2);

    semg1t1 = svmul_f64_x(pg1, semg1t1, semg2t1);
    semg1t2 = svmul_f64_x(pg2, semg1t2, semg2t2);

    svfloat64_t semg3t1 = svsub_f64_x(pg1, k1, p1);
    svfloat64_t semg3t2 = svsub_f64_x(pg2, k2, p2);
    semg3t1 = svmul_f64_x(pg1, semg3t1, CONST_LN2);
    semg3t2 = svmul_f64_x(pg2, semg3t2, CONST_LN2);

    svfloat64_t t1 = svmul_f64_x(pg1, semg3t1, semg3t1);
    svfloat64_t t2 = svmul_f64_x(pg2, semg3t2, semg3t2);

    semg3t1 = svmla_f64_x(pg1, semg3t1, t1, CONST_1D2);
    semg3t2 = svmla_f64_x(pg2, semg3t2, t2, CONST_1D2);

    semg1t1 = svmla_f64_x(pg1, semg1t1, semg1t1, semg3t1);
    semg1t2 = svmla_f64_x(pg2, semg1t2, semg1t2, semg3t2);
    out1 = svmul_f64_x(pg1, out1, semg1t1);
    out2 = svmul_f64_x(pg2, out2, semg1t2);

    return svuzp1_f32(svcvt_f32_f64_z(pg1, out1), svcvt_f32_f64_z(pg2, out2));
}

inline svfloat32_t VCPowF64(const svfloat32_t &src, const svbool_t &pg, const float &p)
{
    svfloat64_t svP = svdup_f64(p);
    svfloat32_t zeros = svdup_n_f32(0.0f);
    svbool_t zeroPg = svpfalse_b();
    svbool_t pgNZ = svcmpgt_f32(pg, src, zeros);
    svbool_t pg1 = svzip1_b32(pgNZ, zeroPg);
    svbool_t pg2 = svzip2_b32(pgNZ, zeroPg);
    
    svfloat32_t tmp1 = svzip1_f32(src, zeros);
    svfloat32_t tmp2 = svzip2_f32(src, zeros);
    svfloat64_t data1 = svcvt_f64_f32_x(pg1, tmp1);
    svfloat64_t data2 = svcvt_f64_f32_x(pg2, tmp2);
    svfloat64_t lndata1;
    svfloat64_t lndata2;
    VCLnF64(data1, data2, pg1, pg2, lndata1, lndata2);
    lndata1 = svmul_f64_x(pg1, lndata1, svP);
    lndata2 = svmul_f64_x(pg2, lndata2, svP);
    return VCExpF64(lndata1, lndata2, pg1, pg2);
}

template <typename CompType, typename InType>
inline void NormOnesFunc(const SizeType &loopSize, const InType *src, CompType &dst, float p)
{
    SizeType svStep = svcntw();
    std::vector<float> vec(svStep, 0);
    
    for (SizeType idx = 0; idx < loopSize; idx += svStep) {
        svbool_t pg = svwhilelt_b32_u64(idx, loopSize);
        svfloat32_t data;

        LoadReduc(src + idx, pg, data, idx, loopSize);
        data = svabs_f32_x(pg, data);

        svfloat32_t out = VCPowF64(data, pg, p);

        svst1_f32(pg, vec.data(), out);
        int len = svcntp_b32(svptrue_b32(), pg);
        for (int i = 0; i < len; i++) {
            dst += vec[i];
        }
    }
}


template <typename OutType>
inline void NormOnesStore(const SizeType &loopSize, OutType *dst, float* dstTmp, ReductionFunction kind,
    float p, float eps)
{
    SizeType svStep = svcntw();
    svfloat32_t svEps = svdup_n_f32(eps);
    for (SizeType idx = 0; idx < loopSize; idx += svStep) {
        svbool_t pg = svwhilelt_b32_u64(idx, loopSize);
        svfloat32_t data = svld1_f32(pg, dstTmp + idx);

        svfloat32_t out;
        if (kind == ReductionFunction::NORM_LP_SUM) {
            data = svadd_f32_x(pg, data, svEps);
            out = VCPowF64(data, pg, 1.0/p);
        } else if (kind == ReductionFunction::NORM_LP_MAX) {
            data = svmax_f32_x(pg, data, svEps);
            out = VCPowF64(data, pg, 1.0/p);
        } else if (kind == ReductionFunction::NORM_LP_POWER_P_SUM) {
            out = svadd_f32_x(pg, data, svEps);
        } else {
            out = svmax_f32_x(pg, data, svEps);
        }
        StoreReduc(dst + idx, pg, out);
    }
}

template <typename CompType, typename InType, typename OutType>
KUDNN_INLINE inline void EvaluateNormOnes(const Shape &start, const Shape &end, const InType *src, OutType *dst,
    float* dstBuf, CompType acc, ReductionFunction kind, float p, float eps,
    Shape &dimsSrc, Shape &dimsDst, Shape &stridesSrc, Shape &stridesDst) noexcept
{
    Service::Unused(acc);
    std::array<SizeType, MAX_DIMS> loopdims;
    for (SizeType i = 0; i < dimsDst.GetNumDims(); ++i) {
        loopdims[i] = (dimsDst[i] == 1) ? dimsSrc[i] : end[i];
    }

    float* dstTmp = nullptr;
    if constexpr (std::is_same_v<float, OutType>) {
        dstTmp = static_cast<float*>(dst);
    } else {
        dstTmp = dstBuf;
    }

    for (SizeType dim0 = start[IDX_0]; dim0 < loopdims[IDX_0]; ++dim0) {
        SizeType idxSrc0 = dim0 * stridesSrc[IDX_0];
        SizeType idxDst0 = dim0 * stridesDst[IDX_0];
        for (SizeType dim1 = start[IDX_1]; dim1 < loopdims[IDX_1]; ++dim1) {
            SizeType idxSrc1 = dim1 * stridesSrc[IDX_1];
            SizeType idxDst1 = dim1 * stridesDst[IDX_1];
            for (SizeType dim2 = start[IDX_2]; dim2 < loopdims[IDX_2]; ++dim2) {
                SizeType idxSrc2 = dim2 * stridesSrc[IDX_2];
                SizeType idxDst2 = dim2 * stridesDst[IDX_2];
                for (SizeType dim3 = start[IDX_3]; dim3 < loopdims[IDX_3]; ++dim3) {
                    SizeType idxSrc = idxSrc0 + idxSrc1 + idxSrc2 + dim3 * stridesSrc[IDX_3];
                    SizeType idxDst = idxDst0 + idxDst1 + idxDst2 + dim3 * stridesDst[IDX_3];
                    NormOnesFunc(loopdims[IDX_4], src + idxSrc, dstTmp[idxDst], p);
                }
            }
        }
    }

    SizeType loopSize = (end[IDX_0] - start[IDX_0]) * (end[IDX_1] - start[IDX_1]) *
        (end[IDX_2] - start[IDX_2]) * (end[IDX_3] - start[IDX_3]);
    SizeType idxDst = start[IDX_0] * stridesDst[IDX_0] + start[IDX_1] * stridesDst[IDX_1] +
        start[IDX_2] * stridesDst[IDX_2] + start[IDX_3] * stridesDst[IDX_3];
    NormOnesStore(loopSize, dst + idxDst, dstTmp + idxDst, kind, p, eps);
}

} // Reduction
} // Detail
} // KuDNN

#endif // KUDNNL_920F
#endif //  KUDNN_REDUCTION_VC512_NORM_HPP