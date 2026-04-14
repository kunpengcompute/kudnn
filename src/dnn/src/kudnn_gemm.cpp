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

#include <cstddef>
#include <memory>
#include <vector>

#include <arm_neon.h> // required for bf16 conversions
#include <arm_sve.h> // required for bf16 conversions

#define BGEMM
#include "kblas.h"
#include "kudnn.hpp"

#include "threading_impl.hpp"
#include "internal_funcs.hpp"
#include "vector_wrappers.hpp"
#include "matmul_jit/kudnn_gemm_helpers.hpp"
#include "kudnn_jit.hpp"


namespace KuDNN {
namespace Detail {
namespace GemmHelpers {

static int g_kdnnNotUseJit = 0;

struct MatrixConverter final {
    MatrixConverter(Element::Type fromType, Element::Type toType, const void *data, SizeType size) : originData(data),
        p(((toType == fromType) || (data == nullptr)) ? nullptr : (Service::AlignedAlloc(size * toType.GetSize()))),
        r({size, fromType, Layout::A}, {size, toType, Layout::A}) {
        if (p) {
            r.Run(data, p);
        }
    }
    MatrixConverter(const MatrixConverter &) = delete;
    MatrixConverter(MatrixConverter &&) = delete;
    MatrixConverter &operator=(const MatrixConverter &) = delete;
    MatrixConverter &operator=(MatrixConverter &&) = delete;
    void *Get() const
    {
        return (p == nullptr) ? const_cast<void *>(originData) : p;
    }
    ~MatrixConverter()
    {
        Service::Deallocate(p);
    }
private:
    const void *originData;
    void *p;
    ReorderLayer r;
};

// to support implicit broadcasting and make it more performant in future
// we have different kernels for cases where C size is [m x n] and
// bias size is [1 x 1], [1 x n], [m x 1]
enum class BiasLDKind {
    MUL_STRIDE = 0,
    IGNORE_M_STRIDE,
    IGNORE_N_STRIDE,
    IGNORE_MN_STRIDES
};

template <BiasLDKind BiasLDKind>
static inline SizeType Indexing(size_t mi, size_t ni, size_t ldBias) noexcept;

template <>
inline SizeType Indexing<BiasLDKind::IGNORE_M_STRIDE>(size_t mi, size_t ni, size_t ldBias) noexcept
{
    Service::Unused(mi, ldBias);
    return ni;
}

template <>
inline SizeType Indexing<BiasLDKind::IGNORE_N_STRIDE>(size_t mi, size_t ni, size_t ldBias) noexcept
{
    Service::Unused(ni, ldBias);
    return mi;
}

template <>
inline SizeType Indexing<BiasLDKind::IGNORE_MN_STRIDES>(size_t mi, size_t ni, size_t ldBias) noexcept
{
    Service::Unused(ni, mi, ldBias);
    return 0;
}

template <>
inline SizeType Indexing<BiasLDKind::MUL_STRIDE>(size_t mi, size_t ni, size_t ldBias) noexcept
{
    return ni + mi * ldBias;
}

// C layout is considered to be row-major
template <BiasLDKind BiasLDKind, typename DstDt>
static void AddBias(SizeType m, SizeType n, DstDt *dst, SizeType ldDst, const DstDt *bias) noexcept
{
    for (SizeType mi = 0; mi < m; ++mi) {
        for (SizeType ni = 0; ni < n; ++ni) {
            dst[mi * ldDst + ni] += bias[Indexing<BiasLDKind>(mi, ni, ldDst)];
        }
    }
}

template <typename DstDt>
static void AddBias(CBLAS_LAYOUT dstLayout, SizeType m, SizeType n, SizeType biasM, SizeType biasN,
    DstDt *dst, SizeType ldDst, const DstDt *bia) noexcept
{
    if (dstLayout == CblasRowMajor) {
        if (biasM == 1) {
            if (biasN == 1) {
                AddBias<BiasLDKind::IGNORE_MN_STRIDES>(m, n, dst, ldDst, bia);
            } else {
                AddBias<BiasLDKind::IGNORE_M_STRIDE>(m, n, dst, ldDst, bia);
            }
        } else {
            if (biasN == 1) {
                AddBias<BiasLDKind::IGNORE_N_STRIDE>(m, n, dst, ldDst, bia);
            } else {
                AddBias<BiasLDKind::MUL_STRIDE>(m, n, dst, ldDst, bia);
            }
        }
    } else {
        if (biasM == 1) {
            if (biasN == 1) {
                AddBias<BiasLDKind::IGNORE_MN_STRIDES>(n, m, dst, ldDst, bia);
            } else {
                AddBias<BiasLDKind::IGNORE_N_STRIDE>(n, m, dst, ldDst, bia);
            }
        } else {
            if (biasN == 1) {
                AddBias<BiasLDKind::IGNORE_M_STRIDE>(n, m, dst, ldDst, bia);
            } else {
                AddBias<BiasLDKind::MUL_STRIDE>(n, m, dst, ldDst, bia);
            }
        }
    }
}

template <typename SrcDt, typename WeiDt, typename DstDt, typename BiaDt>
void GEMMWrapper(CBLAS_LAYOUT lt, CBLAS_TRANSPOSE transa, CBLAS_TRANSPOSE transb, CBLAS_OFFSET offsetc,
    SizeType m, SizeType n, SizeType k,
    float alpha, const SrcDt *src, SizeType ldSrc, const WeiDt *wei, SizeType ldWei, float beta,
    DstDt *dst, SizeType ldDst, const BiaDt *bia, SizeType biasM, SizeType biasN) noexcept;

template <>
KUDNN_FORCE_INTERNAL_API void GEMMWrapper<float, float, float, float>(
    CBLAS_LAYOUT lt, CBLAS_TRANSPOSE transa, CBLAS_TRANSPOSE transb, CBLAS_OFFSET offsetc,
    SizeType m, SizeType n, SizeType k,
    float alpha, const float *src, SizeType ldSrc, const float *wei, SizeType ldWei, float beta,
    float *dst, SizeType ldDst, const float *bia, SizeType biasM, SizeType biasN) noexcept
{
    Service::Unused(offsetc);
    if (n == 1) {
        CBLAS_LAYOUT aLayoutForGEMV = (lt == CblasRowMajor) ?
            ((transa == CblasNoTrans) ? CblasRowMajor : CblasColMajor) :
            ((transa == CblasNoTrans) ? CblasColMajor : CblasRowMajor);
        cblas_sgemv(aLayoutForGEMV, CblasNoTrans, m, k, alpha,
            src, ldSrc, wei, 1, beta, dst, 1);
    } else if (m == 1) {
        CBLAS_LAYOUT aLayoutForGEMV = (lt == CblasRowMajor) ?
            ((transb == CblasNoTrans) ? CblasColMajor : CblasRowMajor) :
            ((transb == CblasNoTrans) ? CblasRowMajor : CblasColMajor);
        cblas_sgemv(aLayoutForGEMV, CblasNoTrans, n, k, alpha,
            wei, ldWei, src, 1, beta, dst, 1);
    } else {
        cblas_sgemm(lt, transa, transb, m, n, k, alpha,
            src, ldSrc, wei, ldWei, beta, dst, ldDst);
    }
    if (bia != nullptr) {
        AddBias(lt, m, n, biasM, biasN, dst, ldDst, bia);
    }
}

template <>
KUDNN_FORCE_INTERNAL_API void GEMMWrapper<__fp16, __fp16, __fp16, __fp16>(
    CBLAS_LAYOUT lt, CBLAS_TRANSPOSE transa, CBLAS_TRANSPOSE transb, CBLAS_OFFSET offsetc,
    SizeType m, SizeType n, SizeType k,
    float alpha, const __fp16 *src, SizeType ldSrc, const __fp16 *wei, SizeType ldWei, float beta,
    __fp16 *dst, SizeType ldDst, const __fp16 *bia, SizeType biasM, SizeType biasN) noexcept
{
    Service::Unused(offsetc);
    cblas_hgemm(lt, transa, transb, m, n, k, alpha,
        src, ldSrc, wei, ldWei, beta, dst, ldDst);
    if (bia != nullptr) {
        AddBias(lt, m, n, biasM, biasN, dst, ldDst, bia);
    }
}

template <>
KUDNN_FORCE_INTERNAL_API void GEMMWrapper<__fp16, __fp16, float, float>(
    CBLAS_LAYOUT lt, CBLAS_TRANSPOSE transa, CBLAS_TRANSPOSE transb, CBLAS_OFFSET offsetc,
    SizeType m, SizeType n, SizeType k,
    float alpha, const __fp16 *src, SizeType ldSrc, const __fp16 *wei, SizeType ldWei, float beta,
    float *dst, SizeType ldDst, const float *bia, SizeType biasM, SizeType biasN) noexcept
{
    Service::Unused(offsetc);
    cblas_shgemm(lt, transa, transb, m, n, k, alpha,
        src, ldSrc, wei, ldWei, beta, dst, ldDst);
    if (bia != nullptr) {
        AddBias(lt, m, n, biasM, biasN, dst, ldDst, bia);
    }
}

template <>
KUDNN_FORCE_INTERNAL_API void GEMMWrapper<__bf16, __bf16, __bf16, float>(
    CBLAS_LAYOUT lt, CBLAS_TRANSPOSE transa, CBLAS_TRANSPOSE transb, CBLAS_OFFSET offsetc,
    SizeType m, SizeType n, SizeType k,
    float alpha, const __bf16 *src, SizeType ldSrc, const __bf16 *wei, SizeType ldWei, float beta,
    __bf16 *dst, SizeType ldDst, const float *bia, SizeType biasM, SizeType biasN) noexcept
{
    Service::Unused(offsetc);
    if (bia != nullptr) { // accuracy issue
        MatrixConverter d(Element::TypeT::BF16, Element::TypeT::F32, dst, m * n);
        cblas_sbgemm(lt, transa, transb, m, n, k, alpha,
            src, ldSrc, wei, ldWei, beta, static_cast<float *>(d.Get()), ldDst);
        // bias
        AddBias(lt, m, n, biasM, biasN, static_cast<float *>(d.Get()), ldDst, bia);
        Service::ConvertFp32ToBf16(static_cast<const float *>(d.Get()), static_cast<__bf16 *>(dst), m * n);
    } else {
        __bf16 alphaBF16 = vcvth_bf16_f32(alpha);
        __bf16 betaBF16 = vcvth_bf16_f32(beta);
        cblas_bgemm(lt, transa, transb, m, n, k, alphaBF16,
            src, ldSrc, wei, ldWei, betaBF16, dst, ldDst);
    }
}

template <>
KUDNN_FORCE_INTERNAL_API void GEMMWrapper<__bf16, __bf16, float, float>(
    CBLAS_LAYOUT lt, CBLAS_TRANSPOSE transa, CBLAS_TRANSPOSE transb, CBLAS_OFFSET offsetc,
    SizeType m, SizeType n, SizeType k,
    float alpha, const __bf16 *src, SizeType ldSrc, const __bf16 *wei, SizeType ldWei, float beta,
    float *dst, SizeType ldDst, const float *bia, SizeType biasM, SizeType biasN) noexcept
{
    Service::Unused(offsetc);
    cblas_sbgemm(lt, transa, transb, m, n, k, alpha,
        src, ldSrc, wei, ldWei, beta, dst, ldDst);
    // bias
    if (bia != nullptr) {
        AddBias(lt, m, n, biasM, biasN, dst, ldDst, bia);
    }
}

//#ifdef KUDNNL_920B
template <>
KUDNN_FORCE_INTERNAL_API void GEMMWrapper<std::int8_t, std::int8_t, std::int32_t, std::int32_t>(
    CBLAS_LAYOUT lt, CBLAS_TRANSPOSE transa, CBLAS_TRANSPOSE transb, CBLAS_OFFSET offsetc,
    SizeType m, SizeType n, SizeType k,
    float alpha, const std::int8_t *src, SizeType ldSrc, const std::int8_t *wei, SizeType ldWei, float beta,
    std::int32_t *dst, SizeType ldDst, const std::int32_t *bia, SizeType biasM, SizeType biasN) noexcept
{
    std::int32_t nullBia = 0;
    const std::int32_t *originBia = bia;
    if (((originBia != nullptr) && (biasM != 1) && (biasN != 1)) || (originBia == nullptr)) {
        offsetc = CblasFixOffset;
        bia = &nullBia;
    }

    cblas_gemm_s8s8s32(lt, transa, transb, offsetc, m, n, k, alpha,
        src, ldSrc, 0, wei, ldWei, 0, beta, dst, ldDst, bia);
    if ((originBia != nullptr) && (biasM != 1) && (biasN != 1)) {
        // bias
        AddBias(lt, m, n, biasM, biasN, dst, ldDst, originBia);
    }
}

template <>
KUDNN_FORCE_INTERNAL_API void GEMMWrapper<std::int8_t, std::uint8_t, std::int32_t, std::int32_t>(
    CBLAS_LAYOUT lt, CBLAS_TRANSPOSE transa, CBLAS_TRANSPOSE transb, CBLAS_OFFSET offsetc,
    SizeType m, SizeType n, SizeType k,
    float alpha, const std::int8_t *src, SizeType ldSrc, const std::uint8_t *wei, SizeType ldWei, float beta,
    std::int32_t *dst, SizeType ldDst, const std::int32_t *bia, SizeType biasM, SizeType biasN) noexcept
{
    std::int32_t nullBia = 0;
    const std::int32_t *originBia = bia;
    if (((originBia != nullptr) && (biasM != 1) && (biasN != 1)) || (originBia == nullptr)) {
        offsetc = CblasFixOffset;
        bia = &nullBia;
    }
    cblas_gemm_s8u8s32(lt, transa, transb, offsetc, m, n, k, alpha,
        src, ldSrc, 0, wei, ldWei, 0, beta, dst, ldDst, bia);
    if ((originBia != nullptr) && (biasM != 1) && (biasN != 1)) {
        // bias
        AddBias(lt, m, n, biasM, biasN, dst, ldDst, originBia);
    }
}

template <>
KUDNN_FORCE_INTERNAL_API void GEMMWrapper<std::uint8_t, std::int8_t, std::int32_t, std::int32_t>(
    CBLAS_LAYOUT lt, CBLAS_TRANSPOSE transa, CBLAS_TRANSPOSE transb, CBLAS_OFFSET offsetc,
    SizeType m, SizeType n, SizeType k,
    float alpha, const std::uint8_t *src, SizeType ldSrc, const std::int8_t *wei, SizeType ldWei, float beta,
    std::int32_t *dst, SizeType ldDst, const std::int32_t *bia, SizeType biasM, SizeType biasN) noexcept
{
    std::int32_t nullBia = 0;
    const std::int32_t *originBia = bia;
    if (((originBia != nullptr) && (biasM != 1) && (biasN != 1)) || (originBia == nullptr)) {
        offsetc = CblasFixOffset;
        bia = &nullBia;
    }
    cblas_gemm_u8s8s32(lt, transa, transb, offsetc, m, n, k, alpha,
        src, ldSrc, 0, wei, ldWei, 0, beta, dst, ldDst, bia);
    if ((originBia != nullptr) && (biasM != 1) && (biasN != 1)) {
        // bias
        AddBias(lt, m, n, biasM, biasN, dst, ldDst, originBia);
    }
}

template <>
KUDNN_FORCE_INTERNAL_API void GEMMWrapper<std::uint8_t, std::uint8_t, std::int32_t, std::int32_t>(
    CBLAS_LAYOUT lt, CBLAS_TRANSPOSE transa, CBLAS_TRANSPOSE transb, CBLAS_OFFSET offsetc,
    SizeType m, SizeType n, SizeType k,
    float alpha, const std::uint8_t *src, SizeType ldSrc, const std::uint8_t *wei, SizeType ldWei, float beta,
    std::int32_t *dst, SizeType ldDst, const std::int32_t *bia, SizeType biasM, SizeType biasN) noexcept
{
    std::int32_t nullBia = 0;
    const std::int32_t *originBia = bia;
    if (((originBia != nullptr) && (biasM != 1) && (biasN != 1)) || (originBia == nullptr)) {
        offsetc = CblasFixOffset;
        bia = &nullBia;
    }
    cblas_gemm_u8u8s32(lt, transa, transb, offsetc, m, n, k, alpha,
        src, ldSrc, 0, wei, ldWei, 0, beta, dst, ldDst, bia);
    if ((originBia != nullptr) && (biasM != 1) && (biasN != 1)) {
        // bias
        AddBias(lt, m, n, biasM, biasN, dst, ldDst, originBia);
    }
}
//#endif

template <>
void GEMMWrapper<__fp16, std::int8_t, __fp16, float>(
    CBLAS_LAYOUT lt, CBLAS_TRANSPOSE transa, CBLAS_TRANSPOSE transb, CBLAS_OFFSET offsetc,
    SizeType m, SizeType n, SizeType k,
    float alpha, const __fp16 *src, SizeType ldSrc, const std::int8_t *wei, SizeType ldWei, float beta,
    __fp16 *dst, SizeType ldDst, const float *bia, SizeType biasM, SizeType biasN) noexcept
{
    Service::Unused(offsetc);
    if (bia != nullptr) { // accuracy issue
        MatrixConverter d(Element::TypeT::F16, Element::TypeT::F32, dst, m * n);
        MatrixConverter w(Element::TypeT::S8, Element::TypeT::F16, wei, k * n);
        cblas_shgemm(lt, transa, transb, m, n, k, alpha,
            src, ldSrc, static_cast<__fp16 *>(w.Get()), ldWei, beta, static_cast<float *>(d.Get()), ldDst);
        // bias
        AddBias(lt, m, n, biasM, biasN, static_cast<float *>(d.Get()), ldDst, bia);
        Service::ConvertFp32ToFp16(static_cast<const float *>(d.Get()), static_cast<__fp16 *>(dst), m * n);
    } else {
        MatrixConverter w(Element::TypeT::S8, Element::TypeT::F16, wei, k * n);
        cblas_hgemm(lt, transa, transb, m, n, k, alpha,
            src, ldSrc, static_cast<__fp16 *>(w.Get()), ldWei, beta, dst, ldDst);
    }
}
 
template <>
void GEMMWrapper<__fp16, std::uint8_t, __fp16, float>(
    CBLAS_LAYOUT lt, CBLAS_TRANSPOSE transa, CBLAS_TRANSPOSE transb, CBLAS_OFFSET offsetc,
    SizeType m, SizeType n, SizeType k,
    float alpha, const __fp16 *src, SizeType ldSrc, const std::uint8_t *wei, SizeType ldWei, float beta,
    __fp16 *dst, SizeType ldDst, const float *bia, SizeType biasM, SizeType biasN) noexcept
{
    Service::Unused(offsetc);
    if (bia != nullptr) { // accuracy issue
        MatrixConverter d(Element::TypeT::F16, Element::TypeT::F32, dst, m * n);
        MatrixConverter w(Element::TypeT::U8, Element::TypeT::F16, wei, k * n);
        cblas_shgemm(lt, transa, transb, m, n, k, alpha,
            src, ldSrc, static_cast<__fp16 *>(w.Get()), ldWei, beta, static_cast<float *>(d.Get()), ldDst);
        // bias
        AddBias(lt, m, n, biasM, biasN, static_cast<float *>(d.Get()), ldDst, bia);
        Service::ConvertFp32ToFp16(static_cast<const float *>(d.Get()), static_cast<__fp16 *>(dst), m * n);
    } else {
        MatrixConverter w(Element::TypeT::U8, Element::TypeT::F16, wei, k * n);
        cblas_hgemm(lt, transa, transb, m, n, k, alpha,
            src, ldSrc, static_cast<__fp16 *>(w.Get()), ldWei, beta, dst, ldDst);
    }
}


struct GEMMCallerBase {
    virtual void Call(const GemmInfo &gemmInfo,
        const void *src, const void *wei, void *dst, const void *bia,
        float alpha, float beta, int numThreads) const noexcept = 0;
    virtual Element::Type GetSrcDt() const noexcept = 0;
    virtual Element::Type GetWeiDt() const noexcept = 0;
    virtual Element::Type GetDstDt() const noexcept = 0;
    virtual Element::Type GetBiaDt() const noexcept = 0;
    virtual ~GEMMCallerBase() = default;
};

static Status ValidateGemmParams(const TensorInfo &srcInfo, const TensorInfo &weiInfo,
    const TensorInfo &dstInfo, const TensorInfo &biaInfo, int numThreads) noexcept
{
    if (!dstInfo.IsDense()) {
        return Status::UNSUPPORTED;
    }
    // perform checks of data types
    if ((srcInfo.GetType() == Element::TypeT::S32) || (weiInfo.GetType() == Element::TypeT::S32)) {
        return Status::UNSUPPORTED;
    }

    int maxNumThreads = KuDNN::Threading::GetMaxNumThreads();
    if (numThreads < 0 || numThreads > maxNumThreads) {
        return Status::BAD_ARGUMENTS;
    }

    bool areDimsDifferent = (srcInfo.GetNumDims() != weiInfo.GetNumDims()) ||
        (srcInfo.GetNumDims() != dstInfo.GetNumDims());
    bool isBiasNull = biaInfo.GetTotalTensorSize() == 0;
    if (!isBiasNull) {
        areDimsDifferent = (areDimsDifferent || (srcInfo.GetNumDims() != biaInfo.GetNumDims()));
    }
    bool isBadArguments = areDimsDifferent;
    // since num_dims are equal for all we can check only for one tensor
    isBadArguments = isBadArguments || (srcInfo.GetNumDims() < DIM_2);
    // perform checks of data Shapes
    //    A    x    B    =    C
    // [m * k] x [k * n] = [m * n]
    isBadArguments = isBadArguments ||
        (srcInfo.GetDims()[srcInfo.GetNumDims() - DIM_2] != dstInfo.GetDims()[dstInfo.GetNumDims() - DIM_2]) ||
        (srcInfo.GetDims()[srcInfo.GetNumDims() - DIM_1] != weiInfo.GetDims()[weiInfo.GetNumDims() - DIM_2]) ||
        (weiInfo.GetDims()[weiInfo.GetNumDims() - DIM_1] != dstInfo.GetDims()[dstInfo.GetNumDims() - DIM_1]);
    if (isBadArguments) {
        return Status::BAD_ARGUMENTS;
    }
    // perform checks of batch sizes
    for (SizeType dim = 0; dim < weiInfo.GetNumDims() - DIM_2; ++dim) {
        bool badDims = false;
        if (srcInfo.GetDims()[dim] != 1) {
            badDims = (badDims || (srcInfo.GetDims()[dim] != dstInfo.GetDims()[dim]));
        }
        if ((weiInfo.GetDims()[dim] != 1) && (srcInfo.GetDims()[dim] != 1)) {
            badDims = (badDims || (srcInfo.GetDims()[dim] != weiInfo.GetDims()[dim]));
        }
        if (!isBiasNull && (biaInfo.GetDims()[dim] != 1)) {
            badDims = (badDims || (biaInfo.GetDims()[dim] != dstInfo.GetDims()[dim]));
        }
        if (badDims) {
            return Status::BAD_ARGUMENTS;
        }
    }
    // perform check that bias and C has corresponding Shape
    // bias dimensions can be implicitly broadcasted to C dimensions
    isBadArguments = !isBiasNull && (((biaInfo.GetDims()[biaInfo.GetNumDims() - 1] != 1) &&
        (biaInfo.GetDims()[biaInfo.GetNumDims() - DIM_1] != dstInfo.GetDims()[dstInfo.GetNumDims() - DIM_1])) ||
        ((biaInfo.GetDims()[biaInfo.GetNumDims() - DIM_2] != 1) &&
        (biaInfo.GetDims()[biaInfo.GetNumDims() - DIM_2] != dstInfo.GetDims()[dstInfo.GetNumDims() - DIM_2])));
    if (isBadArguments) {
        return Status::BAD_ARGUMENTS;
    }
    // we can't process GEMM with strides over the fastest changing index
    bool isUnsupported = ((srcInfo.GetStrides()[Service::GetIdxForUserLayout(srcInfo.GetLayout(),
        srcInfo.GetNumDims() - DIM_1)] != 1) &&
        (srcInfo.GetStrides()[Service::GetIdxForUserLayout(srcInfo.GetLayout(), srcInfo.GetNumDims() - DIM_1)] != 0)) &&
        ((weiInfo.GetStrides()[Service::GetIdxForUserLayout(weiInfo.GetLayout(), weiInfo.GetNumDims() - DIM_1)] != 1) &&
        (weiInfo.GetStrides()[Service::GetIdxForUserLayout(weiInfo.GetLayout(), weiInfo.GetNumDims() - DIM_1)] != 0)) &&
        ((dstInfo.GetStrides()[Service::GetIdxForUserLayout(dstInfo.GetLayout(), dstInfo.GetNumDims() - DIM_1)] != 1) &&
        (dstInfo.GetStrides()[Service::GetIdxForUserLayout(dstInfo.GetLayout(), dstInfo.GetNumDims() - DIM_1)] != 0)) &&
        ((!isBiasNull &&
        ((biaInfo.GetStrides()[Service::GetIdxForUserLayout(biaInfo.GetLayout(), biaInfo.GetNumDims() - DIM_1)] != 1) &&
        (biaInfo.GetStrides()[Service::GetIdxForUserLayout(biaInfo.GetLayout(), biaInfo.GetNumDims() - DIM_1)] != 0))));
    if (isUnsupported) {
        return Status::UNSUPPORTED;
    }
    if (srcInfo.GetTotalTensorSize() == 0) {
        return Status::UNSUPPORTED;
    }
    if (weiInfo.GetTotalTensorSize() == 0) {
        return Status::UNSUPPORTED;
    }
    if (dstInfo.GetTotalTensorSize() == 0) {
        return Status::UNSUPPORTED;
    }
    return Status::SUCCESS;
}

static CBLAS_LAYOUT GetCblasLayout(const Layout &lt) noexcept
{
    if ((lt == Layout::BA) ||
        (lt == Layout::ACB) ||
        (lt == Layout::ABDC) ||
        (lt == Layout::ABCED)) {
        return CblasColMajor;
    } else {
        return CblasRowMajor;
    }
}

static CBLAS_TRANSPOSE GetCblasTrans(const Layout &lt, const Layout &cLayout) noexcept
{
    CBLAS_LAYOUT cblasLayout = GetCblasLayout(cLayout);
    if (cblasLayout == CblasRowMajor) {
        if ((lt == Layout::BA) ||
            (lt == Layout::ACB) ||
            (lt == Layout::ABDC) ||
            (lt == Layout::ABCED)) {
            return CblasTrans;
        } else {
            return CblasNoTrans;
        }
    } else {
        if ((lt == Layout::BA) ||
            (lt == Layout::ACB) ||
            (lt == Layout::ABDC) ||
            (lt == Layout::ABCED)) {
            return CblasNoTrans;
        } else {
            return CblasTrans;
        }
    }
}

template <typename SrcDt, typename WeiDt, typename DstDt, typename BiaDt>
struct GEMMCaller : GEMMCallerBase {
    void Call(const GemmInfo &gemmInfo,
        const void *src, const void *wei, void *dst, const void *bia, float alpha,
        float beta, int numThreads) const noexcept override
    {
        const SrcDt *typedSrc = static_cast<const SrcDt *>(src);
        const WeiDt *typedWei = static_cast<const WeiDt *>(wei);
        DstDt *typedDst = static_cast<DstDt *>(dst);
        const BiaDt *typedBia = static_cast<const BiaDt *>(bia);
        CBLAS_LAYOUT cLayout = GemmHelpers::GetCblasLayout(gemmInfo.dstInfo.GetLayout());
        CBLAS_TRANSPOSE transa = GemmHelpers::GetCblasTrans(gemmInfo.srcInfo.GetLayout(), gemmInfo.dstInfo.GetLayout());
        CBLAS_TRANSPOSE transb = GemmHelpers::GetCblasTrans(gemmInfo.weiInfo.GetLayout(), gemmInfo.dstInfo.GetLayout());
        SizeType m = gemmInfo.srcInfo.GetDims()[gemmInfo.srcInfo.GetNumDims() - DIM_2];
        SizeType n = gemmInfo.weiInfo.GetDims()[gemmInfo.weiInfo.GetNumDims() - DIM_1];
        SizeType k = gemmInfo.weiInfo.GetDims()[gemmInfo.weiInfo.GetNumDims() - DIM_2];
        SizeType ldaIdx = (cLayout == CblasRowMajor) ? ((transa == CblasNoTrans) ?
            (gemmInfo.srcInfo.GetNumDims() - DIM_2) : (gemmInfo.srcInfo.GetNumDims() - 1)) :
            ((transa == CblasNoTrans) ? (gemmInfo.srcInfo.GetNumDims() - 1) : (gemmInfo.srcInfo.GetNumDims() - DIM_2));
        SizeType ldbIdx = (cLayout == CblasRowMajor) ? ((transb == CblasNoTrans) ?
            (gemmInfo.weiInfo.GetNumDims() - DIM_2) : (gemmInfo.weiInfo.GetNumDims() - 1)) :
            ((transb == CblasNoTrans) ? (gemmInfo.weiInfo.GetNumDims() - 1) : (gemmInfo.weiInfo.GetNumDims() - DIM_2));
        SizeType ldcIdx = (cLayout == CblasRowMajor) ?
            (gemmInfo.dstInfo.GetNumDims() - DIM_2) : (gemmInfo.dstInfo.GetNumDims() - 1);
        SizeType lda = gemmInfo.srcInfo.GetStrides()[ldaIdx];
        if (lda == 0) {
            lda = (ldaIdx == (gemmInfo.srcInfo.GetNumDims() - DIM_2)) ?
                gemmInfo.srcInfo.GetDims()[gemmInfo.weiInfo.GetNumDims() - DIM_1] :
                gemmInfo.srcInfo.GetDims()[gemmInfo.weiInfo.GetNumDims() - DIM_2];
        }
        SizeType ldb = gemmInfo.weiInfo.GetStrides()[ldbIdx];
        if (ldb == 0) {
            ldb = (ldbIdx == (gemmInfo.weiInfo.GetNumDims() - DIM_2)) ?
                gemmInfo.weiInfo.GetDims()[gemmInfo.weiInfo.GetNumDims() - DIM_1] :
                gemmInfo.weiInfo.GetDims()[gemmInfo.weiInfo.GetNumDims() - DIM_2];
        }
        SizeType ldc = gemmInfo.dstInfo.GetStrides()[ldcIdx];
        if (ldc == 0) {
            ldc = (ldcIdx == (gemmInfo.dstInfo.GetNumDims() - DIM_2)) ?
                gemmInfo.dstInfo.GetDims()[gemmInfo.weiInfo.GetNumDims() - DIM_1] :
                gemmInfo.dstInfo.GetDims()[gemmInfo.weiInfo.GetNumDims() - DIM_2];
        }
        CBLAS_OFFSET offsetC = CblasFixOffset;
        SizeType biasM = 0;
        SizeType biasN = 0;
        if (bia) {
            biasN = gemmInfo.biaInfo.GetDims()[gemmInfo.srcInfo.GetNumDims() - DIM_1];
            biasM = gemmInfo.biaInfo.GetDims()[gemmInfo.srcInfo.GetNumDims() - DIM_2];
            if (biasN != 1) {
                offsetC = CblasRowOffset;
            } else if (biasM != 1) {
                offsetC = CblasColOffset;
            }
        }

        if (numThreads == 0) {
            BlasSetNumThreadsLocal(KuDNN::Threading::GetMaxNumThreads());
        } else {
            BlasSetNumThreadsLocal(numThreads);
        }

        GEMMWrapper(cLayout, transa, transb, offsetC, m, n, k, alpha, typedSrc, lda,
            typedWei, ldb, beta, typedDst, ldc, typedBia, biasM, biasN);
    }
    Element::Type GetSrcDt() const noexcept override
    {
        return Element::MatchType<SrcDt>();
    }
    Element::Type GetWeiDt() const noexcept override
    {
        return Element::MatchType<WeiDt>();
    }
    Element::Type GetDstDt() const noexcept override
    {
        return Element::MatchType<DstDt>();
    }
    Element::Type GetBiaDt() const noexcept override
    {
        return Element::MatchType<BiaDt>();
    }
};

static std::vector<std::shared_ptr<GEMMCallerBase>> g_gemmImpls = {
    std::make_shared<GEMMCaller<float, float, float, float>>(),
    std::make_shared<GEMMCaller<__fp16, __fp16, __fp16, __fp16>>(),
    std::make_shared<GEMMCaller<__fp16, __fp16, float, float>>(),
    std::make_shared<GEMMCaller<__bf16, __bf16, __bf16, float>>(),
    std::make_shared<GEMMCaller<__bf16, __bf16, float, float>>(),
    std::make_shared<GEMMCaller<std::int8_t, std::int8_t, std::int32_t, std::int32_t>>(),
    std::make_shared<GEMMCaller<std::int8_t, std::uint8_t, std::int32_t, std::int32_t>>(),
    std::make_shared<GEMMCaller<std::uint8_t, std::int8_t, std::int32_t, std::int32_t>>(),
    std::make_shared<GEMMCaller<std::uint8_t, std::uint8_t, std::int32_t, std::int32_t>>(),
    std::make_shared<GEMMCaller<__fp16, std::int8_t, __fp16, float>>(),
    std::make_shared<GEMMCaller<__fp16, std::uint8_t, __fp16, float>>()
};

static Element::Type GetWiderType(Element::Type srcType, Element::Type weiType, Element::Type dstType,
    Element::Type biaType)
{
    bool hasBf16 = (srcType == Element::TypeT::BF16) || (weiType == Element::TypeT::BF16) ||
        (dstType == Element::TypeT::BF16) || (biaType == Element::TypeT::BF16);
    if (hasBf16) {
        if ((srcType == Element::TypeT::BF16) || (weiType == Element::TypeT::BF16) ||
        (dstType == Element::TypeT::BF16) || (biaType == Element::TypeT::BF16)) {
            return Element::TypeT::BF16;
        } else {
            return Element::TypeT::F32;
        }
    }
    // in case all are FP or all are integer
    bool isValidType = ((srcType.IsFP() && weiType.IsFP() && dstType.IsFP() && biaType.IsFP())) ||
        ((srcType.IsIntegral() && weiType.IsIntegral() && dstType.IsIntegral() && biaType.IsIntegral()));
    if (isValidType) {
        Element::Type tmp1 = (srcType.GetSize() > weiType.GetSize()) ? srcType : weiType;
        Element::Type tmp2 = (dstType.GetSize() > biaType.GetSize()) ? dstType : biaType;
        return (tmp1.GetSize() > tmp2.GetSize()) ? tmp1 : tmp2;
    } else {
        return Element::TypeT::F32;
    }
}

static void ChooseImpl(const GemmInfo &gemmInfo,
    const void *src, const void *wei, void *dst, const void *bia, float alpha, float beta, int numThreads)
{
    auto &&srcType = gemmInfo.srcInfo.GetType();
    auto &&weiType = gemmInfo.weiInfo.GetType();
    auto &&dstType = gemmInfo.dstInfo.GetType();
    auto &&biaType = gemmInfo.biaInfo.GetType();
    SizeType m = gemmInfo.srcInfo.GetDims()[gemmInfo.srcInfo.GetNumDims() - DIM_2];
    SizeType n = gemmInfo.weiInfo.GetDims()[gemmInfo.weiInfo.GetNumDims() - DIM_1];
    SizeType k = gemmInfo.weiInfo.GetDims()[gemmInfo.weiInfo.GetNumDims() - DIM_2];
    SizeType biasM = 0;
    SizeType biasN = 0;
    if (bia) {
        biasM = gemmInfo.biaInfo.GetDims()[gemmInfo.srcInfo.GetNumDims() - DIM_2];
        biasN = gemmInfo.biaInfo.GetDims()[gemmInfo.srcInfo.GetNumDims() - DIM_1];
    }
    // check for exact match
    for (auto &&gemmImpl : g_gemmImpls) {
        bool isTypeMatch = (gemmImpl->GetSrcDt() == srcType) && (gemmImpl->GetWeiDt() == weiType) &&
            (gemmImpl->GetDstDt() == dstType) && (gemmImpl->GetBiaDt() == biaType);
        if (isTypeMatch) {
            gemmImpl->Call(gemmInfo, src, wei, dst, bia, alpha, beta, numThreads);
            return;
        }
    }

    // try to convert to wider type and check if there's implementation with wider type
    auto &&widerType = GetWiderType(srcType, weiType, dstType, biaType);
    for (auto &&gemmImpl : g_gemmImpls) {
        bool isWiderType = ((gemmImpl->GetDstDt() == widerType) && (dstType == widerType)) &&
            ((srcType.GetSize() <= gemmImpl->GetSrcDt().GetSize()) &&
            (srcType.IsSigned() == gemmImpl->GetSrcDt().IsSigned())) &&
            ((weiType.GetSize() <= gemmImpl->GetWeiDt().GetSize()) &&
            (weiType.IsSigned() == gemmImpl->GetWeiDt().IsSigned())) &&
            ((biaType.GetSize() <= gemmImpl->GetBiaDt().GetSize()) &&
            (biaType.IsSigned() == gemmImpl->GetBiaDt().IsSigned()));
        if (isWiderType) {
            MatrixConverter s(srcType, gemmImpl->GetSrcDt(), src, m * k);
            MatrixConverter w(weiType, gemmImpl->GetWeiDt(), wei, k * n);
            MatrixConverter b(biaType, gemmImpl->GetBiaDt(), bia, biasM * biasN);
            gemmImpl->Call(gemmInfo, s.Get(), w.Get(), dst, b.Get(), alpha, beta, numThreads);
            return;
        }
    }

    // some integer combinations can't be supported when all the types are converted to wider type
    MatrixConverter s(srcType, Element::TypeT::F32, src, m * k);
    MatrixConverter w(weiType, Element::TypeT::F32, wei, k * n);
    MatrixConverter d(dstType, Element::TypeT::F32, dst, m * n);
    MatrixConverter b(biaType, Element::TypeT::F32, bia, biasM * biasN);
    GEMMCaller<float, float, float, float> {}.Call(gemmInfo, s.Get(), w.Get(), d.Get(), b.Get(),
        alpha, beta, numThreads);
    if (dstType != Element::TypeT::F32) {
        switch (dstType) {
            case Element::TypeT::F16: {
                Service::ConvertFp32ToFp16(static_cast<const float *>(d.Get()), static_cast<__fp16 *>(dst), m * n);
                break;
            }
            case Element::TypeT::BF16: {
                Service::ConvertFp32ToBf16(static_cast<const float *>(d.Get()), static_cast<__bf16 *>(dst), m * n);
                break;
            }
            case Element::TypeT::S32: {
                Service::ConvertFp32ToS32(static_cast<const float *>(d.Get()), static_cast<std::int32_t *>(dst), m * n);
                break;
            }
            case Element::TypeT::S8: {
                Service::ConvertFp32ToS8(static_cast<const float *>(d.Get()), static_cast<std::int8_t *>(dst), m * n);
                break;
            }
            case Element::TypeT::U8: {
                Service::ConvertFp32ToU8(static_cast<const float *>(d.Get()), static_cast<std::uint8_t *>(dst), m * n);
                break;
            }
            default: {
                break;
            }
        }
    }
    return;
}

static inline void BatchedExtendedGemm(const GemmInfo &gemmInfo, const void *a, const void *b, void *c,
    const void *bias, float alpha, float beta, int numThreads) noexcept(false)
{
    if ((a == nullptr) || (b == nullptr) || (c == nullptr)) {
            throw Service::LogicError {"GEMM: one of input matrices is nullptr"};
    }

    void *aExec = const_cast<void *>(a);
    void *bExec = const_cast<void *>(b);
    void *cExec = const_cast<void *>(c);
    void *biasExec = const_cast<void *>(bias);

    GemmInfo gemmInfoExec = gemmInfo;

    // needs to be modified
    bool needToConvert = ((gemmInfoExec.srcInfo.GetNumDims() > NUM_2) &&
        (((gemmInfoExec.srcInfo.GetLayout() != gemmInfoExec.weiInfo.GetLayout()) ||
        (gemmInfoExec.srcInfo.GetLayout() != gemmInfoExec.dstInfo.GetLayout())) ||
        ((biasExec != nullptr) && (gemmInfoExec.dstInfo.GetLayout() != gemmInfo.biaInfo.GetLayout())) ||
        (gemmInfoExec.srcInfo.GetLayout() != gemmInfoExec.srcInfo.GetStandardABXLayout())));
    if (needToConvert) {
        Element::Type convertedPrecision = Element::TypeT::F32;
        if (gemmInfoExec.srcInfo.GetLayout() != gemmInfoExec.srcInfo.GetStandardABXLayout()) {
            TensorInfo newSrcInfo = {gemmInfoExec.srcInfo.GetDims(), convertedPrecision,
                gemmInfoExec.srcInfo.GetStandardABXLayout()};
            ReorderLayer rSrc(gemmInfoExec.srcInfo, newSrcInfo);
            aExec = Service::AlignedAlloc(gemmInfoExec.srcInfo.GetTotalTensorSize() * convertedPrecision.GetSize());
            rSrc.Run(a, aExec);
            gemmInfoExec.srcInfo = newSrcInfo;
        }
        if (gemmInfoExec.weiInfo.GetLayout() != gemmInfoExec.weiInfo.GetStandardABXLayout()) {
            TensorInfo newWeiInfo = {gemmInfoExec.weiInfo.GetDims(), convertedPrecision,
                gemmInfoExec.weiInfo.GetStandardABXLayout()};
            ReorderLayer rWei(gemmInfoExec.weiInfo, newWeiInfo);
            bExec = Service::AlignedAlloc(gemmInfoExec.weiInfo.GetTotalTensorSize() * convertedPrecision.GetSize());
            rWei.Run(b, bExec);
            gemmInfoExec.weiInfo = newWeiInfo;
        }
        if (gemmInfoExec.dstInfo.GetLayout() != gemmInfoExec.dstInfo.GetStandardABXLayout()) {
            TensorInfo newDstInfo = {gemmInfoExec.dstInfo.GetDims(), convertedPrecision,
                gemmInfoExec.dstInfo.GetStandardABXLayout()};
            ReorderLayer rDst(gemmInfoExec.dstInfo, newDstInfo);
            cExec = Service::AlignedAlloc(gemmInfoExec.dstInfo.GetTotalTensorSize() * convertedPrecision.GetSize());
            rDst.Run(c, cExec);
            gemmInfoExec.dstInfo = newDstInfo;
        }
        if (biasExec && (gemmInfoExec.biaInfo.GetLayout() != gemmInfoExec.biaInfo.GetStandardABXLayout())) {
            TensorInfo newBiaInfo = {gemmInfoExec.biaInfo.GetDims(), convertedPrecision,
                gemmInfoExec.biaInfo.GetStandardABXLayout()};
            ReorderLayer rBia(gemmInfoExec.biaInfo, newBiaInfo);
            biasExec = Service::AlignedAlloc(gemmInfoExec.biaInfo.GetTotalTensorSize() * convertedPrecision.GetSize());
            rBia.Run(bias, biasExec);
            gemmInfoExec.biaInfo = newBiaInfo;
        }
    }
    auto&& broadcastedDimsA = Service::BroadcastTo5D(
        Service::GetShapeAccordingToLayout(gemmInfoExec.srcInfo.GetDims(), gemmInfoExec.srcInfo.GetLayout()));
    auto&& broadcastedDimsB = Service::BroadcastTo5D(
        Service::GetShapeAccordingToLayout(gemmInfoExec.weiInfo.GetDims(), gemmInfoExec.weiInfo.GetLayout()));

    Shape broadcastedDims = {std::max(broadcastedDimsA[IDX_0], broadcastedDimsB[IDX_0]),
                             std::max(broadcastedDimsA[IDX_1], broadcastedDimsB[IDX_1]),
                             std::max(broadcastedDimsA[IDX_2], broadcastedDimsB[IDX_2]),
                             1,
                             1};
    auto&& reorderedStridesA = Service::BroadcastTo5D(
        Service::GetShapeAccordingToLayout(gemmInfoExec.srcInfo.GetStrides(),
        gemmInfoExec.srcInfo.GetLayout()), false);
    auto&& reorderedStridesB = Service::BroadcastTo5D(
        Service::GetShapeAccordingToLayout(gemmInfoExec.weiInfo.GetStrides(),
        gemmInfoExec.weiInfo.GetLayout()), false);
    auto&& reorderedStridesC = Service::BroadcastTo5D(
        Service::GetShapeAccordingToLayout(gemmInfoExec.dstInfo.GetStrides(),
        gemmInfoExec.dstInfo.GetLayout()), false);
    auto&& reorderedStridesBias = Service::BroadcastTo5D(
        Service::GetShapeAccordingToLayout(gemmInfoExec.biaInfo.GetStrides(),
        gemmInfoExec.biaInfo.GetLayout()), false);

    for (SizeType i0 = 0; i0 < broadcastedDims[IDX_0]; ++i0) {
        for (SizeType i1 = 0; i1 < broadcastedDims[IDX_1]; ++i1) {
            for (SizeType i2 = 0; i2 < broadcastedDims[IDX_2]; ++i2) {
                const std::byte *aPtr = static_cast<const std::byte *>(aExec) +
                    (i0 * reorderedStridesA[IDX_0] + i1 * reorderedStridesA[IDX_1] + i2 * reorderedStridesA[IDX_2]) *
                    gemmInfoExec.srcInfo.GetType().GetSize();
                const std::byte *bPtr = static_cast<const std::byte *>(bExec) +
                    (i0 * reorderedStridesB[IDX_0] + i1 * reorderedStridesB[IDX_1] + i2 * reorderedStridesB[IDX_2]) *
                    gemmInfoExec.weiInfo.GetType().GetSize();
                std::byte *cPtr = static_cast<std::byte *>(cExec) +
                    (i0 * reorderedStridesC[IDX_0] + i1 * reorderedStridesC[IDX_1] + i2 * reorderedStridesC[IDX_2]) *
                    gemmInfoExec.dstInfo.GetType().GetSize();
                const std::byte *biasPtr = static_cast<const std::byte *>(biasExec);
                if (biasExec) {
                    biasPtr += (i0 * reorderedStridesBias[IDX_0] + i1 * reorderedStridesBias[IDX_1] +
                        i2 * reorderedStridesBias[IDX_2]) * gemmInfoExec.biaInfo.GetType().GetSize();
                }
                ChooseImpl(gemmInfoExec, aPtr, bPtr, cPtr, biasPtr, alpha, beta, numThreads);
            }
        }
    }
    if (needToConvert) {
        if (gemmInfo.srcInfo.GetLayout() != gemmInfo.srcInfo.GetStandardABXLayout()) {
            Service::Deallocate(aExec);
        }
        if (gemmInfo.weiInfo.GetLayout() != gemmInfo.weiInfo.GetStandardABXLayout()) {
            Service::Deallocate(bExec);
        }
        if (gemmInfo.dstInfo.GetLayout() != gemmInfo.dstInfo.GetStandardABXLayout()) {
            ReorderLayer rDst(gemmInfoExec.dstInfo, gemmInfo.dstInfo);
            rDst.Run(cExec, c);
            Service::Deallocate(cExec);
        }
        if (bias && (gemmInfo.biaInfo.GetLayout() != gemmInfo.biaInfo.GetStandardABXLayout())) {
            Service::Deallocate(biasExec);
        }
    }
}

} // GemmHelpers

class GemmImpl final : public Parallelizable {
public:
    GemmImpl(const TensorInfo &srcInfo, const TensorInfo &weiInfo, const TensorInfo &dstInfo,
        const TensorInfo &biaInfo, int numThreads) noexcept(false)
        : gemmImplInfo(srcInfo, weiInfo, dstInfo, biaInfo), impl(nullptr)
    {
        Service::ThrowOnStatus(Gemm::ValidateInput(srcInfo, weiInfo, dstInfo, biaInfo, numThreads), "GEMM");
        bool srcJIT = false;
        bool weiJIT = false;
        bool dstJIT = false;
#ifdef KUDNNL_920Pro
        srcJIT = srcJIT || (gemmImplInfo.srcInfo.GetType() == Element::TypeT::S8);
        srcJIT = srcJIT || (gemmImplInfo.srcInfo.GetType() == Element::TypeT::U8);
        srcJIT = srcJIT || (gemmImplInfo.srcInfo.GetType() == Element::TypeT::F16);
        weiJIT = weiJIT || (gemmImplInfo.weiInfo.GetType() == Element::TypeT::S8);
        weiJIT = weiJIT || (gemmImplInfo.weiInfo.GetType() == Element::TypeT::U8);
        weiJIT = weiJIT || (gemmImplInfo.weiInfo.GetType() == Element::TypeT::F16);
        dstJIT = dstJIT || (gemmImplInfo.dstInfo.GetType() == Element::TypeT::S32);
        dstJIT = dstJIT || (gemmImplInfo.dstInfo.GetType() == Element::TypeT::F32);
#endif // KUDNNL_920Pro
        char *kdnnNotUseJit = std::getenv("KUDNN_NOT_USE_JIT");
        if (kdnnNotUseJit != nullptr && std::strcmp(kdnnNotUseJit, "1") == 0) {
            GemmHelpers::g_kdnnNotUseJit = 1;
        } else {
            GemmHelpers::g_kdnnNotUseJit = 0;
        }
        if (GemmHelpers::g_kdnnNotUseJit != 1 && srcJIT && weiJIT && dstJIT) {
            impl = FindSolution(gemmImplInfo, Threading::GetMaxNumThreads());
        }
    }

    void Run(const void *a, const void *b, void *c, const void *bias,
        float alpha, float beta, int numThreads) const noexcept(false)
    {
        if ((a == nullptr) || (b == nullptr) || (c == nullptr) ||
            ((gemmImplInfo.biaInfo.GetTotalTensorSize() != 0) && (bias == nullptr))) {
            throw Service::LogicError {"Matmul Layer: one of input matrices is nullptr"};
        }
        if ((gemmImplInfo.biaInfo.GetTotalTensorSize() == 0) && (bias != nullptr)) {
            throw Service::LogicError {"Matmul Layer: bias isn't nullptr, but bias TensorInfo has 0 total size"};
        }
        if (impl != nullptr) {
            if (impl->GetNThreads() != Threading::GetMaxNumThreads()) {
                impl = FindSolution(impl->GetTask(), Threading::GetMaxNumThreads());
            }
            impl->Run({a, b, c, bias, alpha, beta});
        } else {
            GemmHelpers::BatchedExtendedGemm(gemmImplInfo, a, b, c, bias, alpha, beta, numThreads);
        }
    }

private:
    const GemmInfo gemmImplInfo;
    mutable std::shared_ptr<ImplBase<GemmInfo>> impl;
};

} // Detail

Gemm::~Gemm() noexcept {}

// all public methods just call corresponding pImpl implemnentation
Gemm::Gemm(const TensorInfo &aInfo, const TensorInfo &bInfo, const TensorInfo &cInfo,
    const TensorInfo &biasInfo, int numThreads) noexcept(false)
    : pImpl(new Detail::GemmImpl(aInfo, bInfo, cInfo, biasInfo, numThreads)) {}

Gemm::Gemm(const TensorInfo &aInfo, const TensorInfo &bInfo,
    const TensorInfo &cInfo, int numThreads) noexcept(false) : Gemm(aInfo, bInfo, cInfo,
    TensorInfo(Shape{0}, cInfo.GetType(), KuDNN::Layout::A), numThreads) {}

Gemm::Gemm(const Gemm &other) noexcept(false)
    : pImpl(new Detail::GemmImpl{*(other.pImpl.get())}) {}

Gemm::Gemm(Gemm &&other) noexcept
    : pImpl(std::move(other.pImpl)) {}

Gemm& Gemm::operator=(const Gemm &other) noexcept(false)
{
    if (this != &other) {
        this->pImpl.reset(new Detail::GemmImpl{*(other.pImpl.get())});
    }
    return *this;
}

Gemm& Gemm::operator=(Gemm &&other) noexcept
{
    if (this != &other) {
        this->pImpl = std::move(other.pImpl);
    }
    return *this;
}

void Gemm::Run(const void *a, const void *b, void *c, const void *bias,
    float alpha, float beta, int numThreads) const noexcept(false)
{
    pImpl->Run(a, b, c, bias, alpha, beta, numThreads);
}

void Gemm::Run(const void *a, const void *b, void *c,
    float alpha, float beta, int numThreads) const noexcept(false)
{
    pImpl->Run(a, b, c, nullptr, alpha, beta, numThreads);
}

Status Gemm::ValidateInput(const TensorInfo &aInfo, const TensorInfo &bInfo,
    const TensorInfo &cInfo, const TensorInfo &biasInfo, int numThreads) noexcept
{
    return Detail::GemmHelpers::ValidateGemmParams(aInfo, bInfo, cInfo, biasInfo, numThreads);
}

Status Gemm::ValidateInput(const TensorInfo &aInfo, const TensorInfo &bInfo,
    const TensorInfo &cInfo, int numThreads) noexcept
{
    return Gemm::ValidateInput(aInfo, bInfo, cInfo,
        TensorInfo(Shape{0}, cInfo.GetType(), KuDNN::Layout::A), numThreads);
}

} // KuDNN