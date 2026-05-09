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

#ifndef KUDNN_KERNEL_FUNCS_HPP
#define KUDNN_KERNEL_FUNCS_HPP

#include <stddef.h>
#include <arm_sve.h>

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

typedef void (*EltwiseFwdFuncF32X)(const int64_t, const float *, float *);
typedef void (*EltwiseFwdFuncF32AX)(const int64_t, const float, const float *, float *);
typedef void (*EltwiseFwdFuncF32ABX)(const int64_t, const float, const float, const float *, float *);
typedef void (*EltwiseBwdFuncF32X)(const int64_t, const float *, const float *, float *);
typedef void (*EltwiseBwdFuncF32AX)(const int64_t, const float, const float *, const float *, float *);
typedef void (*EltwiseBwdFuncF32ABX)(const int64_t, const float, const float, const float *, const float *, float *);

typedef void (*EltwiseFwdFuncF16X)(const int64_t, const __fp16 *, __fp16 *);
typedef void (*EltwiseFwdFuncF16AX)(const int64_t, const float, const __fp16 *, __fp16 *);
typedef void (*EltwiseFwdFuncF16ABX)(const int64_t, const float, const float, const __fp16 *, __fp16 *);
typedef void (*EltwiseBwdFuncF16X)(const int64_t, const __fp16 *, const __fp16 *, __fp16 *);
typedef void (*EltwiseBwdFuncF16AX)(const int64_t, const float, const __fp16 *, const __fp16 *, __fp16 *);
typedef void (*EltwiseBwdFuncF16ABX)(const int64_t, const float, const float, const __fp16 *, const __fp16 *, __fp16 *);

typedef void (*EltwiseFwdFuncBF16X)(const int64_t, const __bf16 *, __bf16 *);
typedef void (*EltwiseFwdFuncBF16AX)(const int64_t, const float, const __bf16 *, __bf16 *);
typedef void (*EltwiseFwdFuncBF16ABX)(const int64_t, const float, const float, const __bf16 *, __bf16 *);
typedef void (*EltwiseBwdFuncBF16X)(const int64_t, const __bf16 *, const __bf16 *, __bf16 *);
typedef void (*EltwiseBwdFuncBF16AX)(const int64_t, const float, const __bf16 *, const __bf16 *, __bf16 *);
typedef void (*EltwiseBwdFuncBF16ABX)(const int64_t, const float, const float, const __bf16 *, const __bf16 *,
                                      __bf16 *);

void AbsLaSveFWDF32(const int64_t len, const float *src, float *dst);
void AbsLaSveBWDF32(const int64_t len, const float *src, const float *dd, float *ds);
void AbsLaSveFWDF16(const int64_t len, const __fp16 *src, __fp16 *dst);
void AbsLaSveBWDF16(const int64_t len, const __fp16 *src, const __fp16 *dd, __fp16 *ds);
void AbsLaSveFWDBF16(const int64_t len, const __bf16 *src, __bf16 *dst);
void AbsLaSveBWDBF16(const int64_t len, const __bf16 *src, const __bf16 *dd, __bf16 *ds);

void HardSigmoidLaSveFWDF32(const int64_t len, const float alpha, const float beta, const float *src, float *dst);
void HardSigmoidLaSveBWDF32(const int64_t len, const float alpha, const float beta, const float *src, const float *dd,
                            float *ds);
void HardSigmoidLaSveFWDF16(const int64_t len, const float alpha, const float beta, const __fp16 *src, __fp16 *dst);
void HardSigmoidLaSveBWDF16(const int64_t len, const float alpha, const float beta, const __fp16 *src, const __fp16 *dd,
                            __fp16 *ds);
void HardSigmoidLaSveFWDBF16(const int64_t len, const float alpha, const float beta, const __bf16 *src, __bf16 *dst);
void HardSigmoidLaSveBWDBF16(const int64_t len, const float alpha, const float beta, const __bf16 *src,
                             const __bf16 *dd, __bf16 *ds);

void ExpLaSveFWDF32(const int64_t len, const float *src, float *dst);
void ExpLaSveFWDF16(const int64_t len, const __fp16 *src, __fp16 *dst);
void ExpLaSveFWDBF16(const int64_t len, const __bf16 *src, __bf16 *dst);
void ExpLaSveBWDF32(const int64_t len, const float *src, const float *dd, float *ds);
void ExpLaSveBWDF16(const int64_t len, const __fp16 *src, const __fp16 *dd, __fp16 *ds);
void ExpLaSveBWDBF16(const int64_t len, const __bf16 *src, const __bf16 *dd, __bf16 *ds);

void ExpLaDstSveBWDF32(const int64_t len, const float *src, const float *dd, float *ds);
void ExpLaDstSveBWDF16(const int64_t len, const __fp16 *src, const __fp16 *dd, __fp16 *ds);
void ExpLaDstSveBWDBF16(const int64_t len, const __bf16 *src, const __bf16 *dd, __bf16 *ds);

void EluLaSveFWDF32(const int64_t len, const float alpha, const float *src, float *dst);
void EluLaSveFWDF16(const int64_t len, const float alpha, const __fp16 *src, __fp16 *dst);
void EluLaSveFWDBF16(const int64_t len, const float alpha, const __bf16 *src, __bf16 *dst);
void EluLaSveBWDF32(const int64_t len, const float alpha, const float *src, const float *dd, float *ds);
void EluLaSveBWDF16(const int64_t len, const float alpha, const __fp16 *src, const __fp16 *dd, __fp16 *ds);
void EluLaSveBWDBF16(const int64_t len, const float alpha, const __bf16 *src, const __bf16 *dd, __bf16 *ds);

void SoftReluLaSveFWDF32(const int64_t len, const float alpha, const float *src, float *dst);
void SoftReluLaSveFWDF16(const int64_t len, const float alpha, const __fp16 *src, __fp16 *dst);
void SoftReluLaSveFWDBF16(const int64_t len, const float alpha, const __bf16 *src, __bf16 *dst);
void SoftReluLaSveBWDF32(const int64_t len, const float alpha, const float *src, const float *dd, float *ds);
void SoftReluLaSveBWDF16(const int64_t len, const float alpha, const __fp16 *src, const __fp16 *dd, __fp16 *ds);
void SoftReluLaSveBWDBF16(const int64_t len, const float alpha, const __bf16 *src, const __bf16 *dd, __bf16 *ds);

void LogisticDstSveBWDF32(const int64_t len, const float *src, const float *dd, float *ds);
void LogisticDstSveBWDF16(const int64_t len, const __fp16 *src, const __fp16 *dd, __fp16 *ds);
void LogisticDstSveBWDBF16(const int64_t len, const __bf16 *src, const __bf16 *dd, __bf16 *ds);

void TanhDstSveBWDF32(const int64_t len, const float *src, const float *dd, float *ds);
void TanhDstSveBWDF16(const int64_t len, const __fp16 *src, const __fp16 *dd, __fp16 *ds);
void TanhDstSveBWDBF16(const int64_t len, const __bf16 *src, const __bf16 *dd, __bf16 *ds);

void PowxaLaSveFWDF32(const int64_t len, const float alpha, const float y, const float *src, float *dst);
void PowxaLaSveBWDF32(const int64_t len, const float alpha, const float beta, const float *src, const float *dd,
                      float *ds);
void PowxaLaSveFWDF16(const int64_t len, const float alpha, const float beta, const __fp16 *src, __fp16 *dst);
void PowxaLaSveBWDF16(const int64_t len, const float alpha, const float beta, const __fp16 *src, const __fp16 *dd,
                      __fp16 *ds);
void PowxaLaSveFWDBF16(const int64_t len, const float alpha, const float beta, const __bf16 *src, __bf16 *dst);
void PowxaLaSveBWDBF16(const int64_t len, const float alpha, const float beta, const __bf16 *src, const __bf16 *dd,
                       __bf16 *ds);

void TanhLaSveFWDF32(const int64_t len, const float *src, float *dst);
void TanhLaSveBWDF32(const int64_t len, const float *src, const float *dd, float *ds);
void TanhLaSveFWDF16(const int64_t len, const __fp16 *src, __fp16 *dst);
void TanhLaSveBWDF16(const int64_t len, const __fp16 *src, const __fp16 *dd, __fp16 *ds);
void TanhLaSveFWDBF16(const int64_t len, const __bf16 *src, __bf16 *dst);
void TanhLaSveBWDBF16(const int64_t len, const __bf16 *src, const __bf16 *dd, __bf16 *ds);

void MishLaSveFWDF32(const int64_t len, const float *src, float *dst);
void MishLaSveBWDF32(const int64_t len, const float *src, const float *dd, float *ds);
void MishLaSveFWDF16(const int64_t len, const __fp16 *src, __fp16 *dst);
void MishLaSveBWDF16(const int64_t len, const __fp16 *src, const __fp16 *dd, __fp16 *ds);
void MishLaSveFWDBF16(const int64_t len, const __bf16 *src, __bf16 *dst);
void MishLaSveBWDBF16(const int64_t len, const __bf16 *src, const __bf16 *dd, __bf16 *ds);

void GeluTanhLaSveFWDF32(const int64_t len, const float *src, float *dst);
void GeluTanhLaSveBWDF32(const int64_t len, const float *src, const float *dd, float *ds);
void GeluTanhLaSveFWDF16(const int64_t len, const __fp16 *src, __fp16 *dst);
void GeluTanhLaSveBWDF16(const int64_t len, const __fp16 *src, const __fp16 *dd, __fp16 *ds);
void GeluTanhLaSveFWDBF16(const int64_t len, const __bf16 *src, __bf16 *dst);
void GeluTanhLaSveBWDBF16(const int64_t len, const __bf16 *src, const __bf16 *dd, __bf16 *ds);

void GeluErfLaSveFWDF32(const int64_t len, const float *src, float *dst);
void GeluErfLaSveBWDF32(const int64_t len, const float *src, const float *dd, float *ds);
void GeluErfLaSveFWDF16(const int64_t len, const __fp16 *src, __fp16 *dst);
void GeluErfLaSveBWDF16(const int64_t len, const __fp16 *src, const __fp16 *dd, __fp16 *ds);
void GeluErfLaSveFWDBF16(const int64_t len, const __bf16 *src, __bf16 *dst);
void GeluErfLaSveBWDBF16(const int64_t len, const __bf16 *src, const __bf16 *dd, __bf16 *ds);

void SwishLaSveFWDF32(const int64_t len, const float alpha, const float *src, float *dst);
void SwishLaSveBWDF32(const int64_t len, const float alpha, const float *src, const float *dd, float *ds);
void SwishLaSveFWDBF16(const int64_t len, const float alpha, const __bf16 *src, __bf16 *dst);
void SwishLaSveBWDBF16(const int64_t len, const float alpha, const __bf16 *src, const __bf16 *dd, __bf16 *ds);
void SwishLaSveFWDF16(const int64_t len, const float alpha, const __fp16 *src, __fp16 *dst);
void SwishLaSveBWDF16(const int64_t len, const float alpha, const __fp16 *src, const __fp16 *dd, __fp16 *ds);

void LnLaSveFWDF32(const int64_t len, const float *src, float *dst);
void LnLaSveBWDF32(const int64_t len, const float *src, const float *dd, float *ds);
void LnLaSveFWDF16(const int64_t len, const __fp16 *src, __fp16 *dst);
void LnLaSveBWDF16(const int64_t len, const __fp16 *src, const __fp16 *dd, __fp16 *ds);
void LnLaSveFWDBF16(const int64_t len, const __bf16 *src, __bf16 *dst);
void LnLaSveBWDBF16(const int64_t len, const __bf16 *src, const __bf16 *dd, __bf16 *ds);

void NearbyintSveFWDF32(const int64_t len, const float *src, float *dst);
void NearbyintSveFWDF16(const int64_t len, const __fp16 *src, __fp16 *dst);
void NearbyintSveFWDBF16(const int64_t len, const __bf16 *src, __bf16 *dst);

void SqrtLaSveFWDF32(const int64_t len, const float *src, float *dst);
void SqrtLaSveBWDF32(const int64_t len, const float *src, const float *dd, float *ds);
void SqrtLaSveFWDF16(const int64_t len, const __fp16 *src, __fp16 *dst);
void SqrtLaSveBWDF16(const int64_t len, const __fp16 *src, const __fp16 *dd, __fp16 *ds);
void SqrtLaSveFWDBF16(const int64_t len, const __bf16 *src, __bf16 *dst);
void SqrtLaSveBWDBF16(const int64_t len, const __bf16 *src, const __bf16 *dd, __bf16 *ds);

void SqrLaSveFWDF32(const int64_t len, const float *src, float *dst);
void SqrLaSveBWDF32(const int64_t len, const float *src, const float *dd, float *ds);
void SqrLaSveFWDF16(const int64_t len, const __fp16 *src, __fp16 *dst);
void SqrLaSveBWDF16(const int64_t len, const __fp16 *src, const __fp16 *dd, __fp16 *ds);
void SqrLaSveFWDBF16(const int64_t len, const __bf16 *src, __bf16 *dst);
void SqrLaSveBWDBF16(const int64_t len, const __bf16 *src, const __bf16 *dd, __bf16 *ds);

void SigmoidSveFWDF32(const int64_t len, const float *src, float *dst);
void SigmoidSveBWDF32(const int64_t len, const float *src, const float *diffDst, float *diffSrc);

void SigmoidSveFWDF16(const int64_t len, const __fp16 *src, __fp16 *dst);
void SigmoidSveBWDF16(const int64_t len, const __fp16 *src, const __fp16 *diffDst, __fp16 *diffSrc);

void ReluSveFWDF32(const int64_t len, const float alpha, const float *src, float *dst);
void ReluSveBWDF32(const int64_t len, const float alpha, const float *src, const float *dd, float *ds);

void ReluSveFWDF16(const int64_t len, const float alpha, const __fp16 *src, __fp16 *dst);
void ReluSveBWDF16(const int64_t len, const float alpha, const __fp16 *src, const __fp16 *dd, __fp16 *ds);

void ReluSveFWDBF16(const int64_t len, const float alpha, const __bf16 *src, __bf16 *dst);
void ReluSveBWDBF16(const int64_t len, const float alpha, const __bf16 *src, const __bf16 *dd, __bf16 *ds);

void ClipSveFWDF32(const int64_t len, const float alpha, const float beta, const float *src, float *dst);
void ClipSveBWDF32(const int64_t len, const float alpha, const float beta, const float *src, const float *dd,
                   float *ds);
void ClipSveFWDF16(const int64_t len, const float alpha, const float beta, const __fp16 *src, __fp16 *dst);
void ClipSveBWDF16(const int64_t len, const float alpha, const float beta, const __fp16 *src, const __fp16 *dd,
                   __fp16 *ds);
void ClipSveFWDBF16(const int64_t len, const float alpha, const float beta, const __bf16 *src, __bf16 *dst);
void ClipSveBWDBF16(const int64_t len, const float alpha, const float beta, const __bf16 *src, const __bf16 *dd,
                    __bf16 *ds);

void ClipV2SveFWDF32(const int64_t len, const float alpha, const float beta, const float *src, float *dst);
void ClipV2SveBWDF32(const int64_t len, const float alpha, const float beta, const float *src, const float *dd,
                     float *ds);
void ClipV2SveFWDF16(const int64_t len, const float alpha, const float beta, const __fp16 *src, __fp16 *dst);
void ClipV2SveBWDF16(const int64_t len, const float alpha, const float beta, const __fp16 *src, const __fp16 *dd,
                     __fp16 *ds);
void ClipV2SveFWDBF16(const int64_t len, const float alpha, const float beta, const __bf16 *src, __bf16 *dst);
void ClipV2SveBWDBF16(const int64_t len, const float alpha, const float beta, const __bf16 *src, const __bf16 *dd,
                      __bf16 *ds);

void HardSwishSveFWDF32(const int64_t len, const float alpha, const float beta, const float *src, float *dst);
void HardSwishSveBWDF32(const int64_t len, const float alpha, const float beta, const float *src, const float *dd,
                        float *ds);
void HardSwishSveFWDF16(const int64_t len, const float alpha, const float beta, const __fp16 *src, __fp16 *dst);
void HardSwishSveBWDF16(const int64_t len, const float alpha, const float beta, const __fp16 *src, const __fp16 *dd,
                        __fp16 *ds);
void HardSwishSveFWDBF16(const int64_t len, const float alpha, const float beta, const __bf16 *src, __bf16 *dst);
void HardSwishSveBWDBF16(const int64_t len, const float alpha, const float beta, const __bf16 *src, const __bf16 *dd,
                         __bf16 *ds);

void SigmoidSveFWDBF16(const int64_t len, const __bf16 *src, __bf16 *dst);
void SigmoidSveBWDBF16(const int64_t len, const __bf16 *src, const __bf16 *diffDst, __bf16 *diffSrc);

void LinearSveF32(const int64_t len, const float alpha, const float beta, const float *src, float *dst);
void LinearSveF16(const int64_t len, const float alpha, const float beta, const __fp16 *src, __fp16 *dst);
void LinearSveBF16(const int64_t len, const float alpha, const float beta, const __bf16 *src, __bf16 *dst);

void SoftmaxAxisF32(const float *srcPtr, float *dstPtr, size_t axisSize, size_t outerSize);
void SoftmaxCrossAxisF32(const float *srcPtr, float *dstPtr, size_t innerSize, size_t axisSize, size_t outerSize);

void SoftmaxAxisF16(const __fp16 *srcPtr, __fp16 *dstPtr, size_t axisSize, size_t outerSize);
void SoftmaxCrossAxisF16(const __fp16 *srcPtr, __fp16 *dstPtr, size_t innerSize, size_t axisSize, size_t outerSize);

void SoftmaxAxisBF16(const __bf16 *srcPtr, __bf16 *dstPtr, size_t axisSize, size_t outerSize);
void SoftmaxCrossAxisBF16(const __bf16 *srcPtr, __bf16 *dstPtr, size_t innerSize, size_t axisSize, size_t outerSize);

void ClipV2DstSveBWDF32(const int64_t len, const float alpha, const float beta, const float *src, const float *dd,
                        float *ds);
void ClipV2DstSveBWDF16(const int64_t len, const float alpha, const float beta, const __fp16 *src, const __fp16 *dd,
                        __fp16 *ds);
void ClipV2DstSveBWDBF16(const int64_t len, const float alpha, const float beta, const __bf16 *src, const __bf16 *dd,
                         __bf16 *ds);

void EluDstSveFWDF32(const int64_t len, const float alpha, const float *src, float *dst);
void EluDstSveFWDF16(const int64_t len, const float alpha, const __fp16 *src, __fp16 *dst);
void EluDstSveFWDBF16(const int64_t len, const float alpha, const __bf16 *src, __bf16 *dst);
void EluDstSveBWDF32(const int64_t len, const float alpha, const float *src, const float *dd, float *ds);
void EluDstSveBWDF16(const int64_t len, const float alpha, const __fp16 *src, const __fp16 *dd, __fp16 *ds);
void EluDstSveBWDBF16(const int64_t len, const float alpha, const __bf16 *src, const __bf16 *dd, __bf16 *ds);

void ReluDstSveBWDF32(const int64_t len, const float alpha, const float *src, const float *dd, float *ds);
void ReluDstSveBWDF16(const int64_t len, const float alpha, const __fp16 *src, const __fp16 *dd, __fp16 *ds);
void ReluDstSveBWDBF16(const int64_t len, const float alpha, const __bf16 *src, const __bf16 *dd, __bf16 *ds);

void SqrtDstSveBWDF32(const int64_t len, const float *src, const float *dd, float *ds);
void SqrtDstSveBWDF16(const int64_t len, const __fp16 *src, const __fp16 *dd, __fp16 *ds);
void SqrtDstSveBWDBF16(const int64_t len, const __bf16 *src, const __bf16 *dd, __bf16 *ds);

inline svfloat32_t CvtOddF32F16(svfloat16_t x, svbool_t pgB16, svbool_t pgB32)
{
    svuint32_t shiftedSrc = svreinterpret_u32_f16(x);
    shiftedSrc = svlsr_n_u32_x(pgB32, shiftedSrc, 0x10U); // F16_SHIFT
    svfloat32_t xOdd = svcvt_f32_f16_x(pgB16, svreinterpret_f16_u32(shiftedSrc));
    return xOdd;
}

inline svfloat16_t CvtOddF16F32(svfloat32_t xOdd, svbool_t pgB32)
{
    svfloat16_t x = svcvt_f16_f32_x(pgB32, xOdd);
    svuint32_t shiftedSrc = svreinterpret_u32_f16(x);
    shiftedSrc = svlsl_n_u32_x(pgB32, shiftedSrc, 0x10U); // F16_SHIFT
    x = svreinterpret_f16_u32(shiftedSrc);
    return x;
}

#ifdef __cplusplus
} // extern "C"
#endif // __cplusplus

#endif // KUDNN_KERNEL_FUNCS_HPP
