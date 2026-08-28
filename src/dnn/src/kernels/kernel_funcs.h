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
typedef void (*EltwiseBwdFuncF16ABX)(const int64_t, const float, const float,
    const __fp16 *, const __fp16 *, __fp16 *);

typedef void (*EltwiseFwdFuncBF16X)(const int64_t, const __bf16 *, __bf16 *);
typedef void (*EltwiseFwdFuncBF16AX)(const int64_t, const float, const __bf16 *, __bf16 *);
typedef void (*EltwiseFwdFuncBF16ABX)(const int64_t, const float, const float, const __bf16 *, __bf16 *);
typedef void (*EltwiseBwdFuncBF16X)(const int64_t, const __bf16 *, const __bf16 *, __bf16 *);
typedef void (*EltwiseBwdFuncBF16AX)(const int64_t, const float, const __bf16 *, const __bf16 *, __bf16 *);
typedef void (*EltwiseBwdFuncBF16ABX)(const int64_t, const float, const float,
    const __bf16 *, const __bf16 *, __bf16 *);

typedef void (*EltwiseFwdFuncS32X)(const int64_t, const int32_t *, int32_t *);
typedef void (*EltwiseFwdFuncS32AX)(const int64_t, const float, const int32_t *, int32_t *);
typedef void (*EltwiseFwdFuncS32ABX)(const int64_t, const float, const float, const int32_t *, int32_t *);

typedef void (*EltwiseFwdFuncS8X)(const int64_t, const int8_t *, int8_t *);
typedef void (*EltwiseFwdFuncS8AX)(const int64_t, const float, const int8_t *, int8_t *);
typedef void (*EltwiseFwdFuncS8ABX)(const int64_t, const float, const float, const int8_t *, int8_t *);

typedef void (*EltwiseFwdFuncU8X)(const int64_t, const uint8_t *, uint8_t *);
typedef void (*EltwiseFwdFuncU8AX)(const int64_t, const float, const uint8_t *, uint8_t *);
typedef void (*EltwiseFwdFuncU8ABX)(const int64_t, const float, const float, const uint8_t *, uint8_t *);

void ReluVCFWDS32(const int64_t len, const float alpha, const int32_t *src, int32_t *dst);
void ReluVCFWDS8(const int64_t len, const float alpha, const int8_t *src, int8_t *dst);
void ReluVCFWDU8(const int64_t len, const float alpha, const uint8_t *src, uint8_t *dst);

void LinearVCFWDS32(const int64_t len, const float alpha, const float beta, const int32_t *src, int32_t *dst);
void LinearVCFWDS8(const int64_t len, const float alpha, const float beta, const int8_t *src, int8_t *dst);
void LinearVCFWDU8(const int64_t len, const float alpha, const float beta, const uint8_t *src, uint8_t *dst);

void AbsLaFWDF32(const int64_t len, const float *src, float *dst);
void AbsLaVCBWDF32(const int64_t len, const float *src, const float *dd, float *ds);
void AbsLaFWDF16(const int64_t len, const __fp16 *src, __fp16 *dst);
void AbsLaBWDF16(const int64_t len, const __fp16 *src, const __fp16 *dd, __fp16 *ds);
void AbsLaFWDBF16(const int64_t len, const __bf16 *src, __bf16 *dst);
void AbsLaVCBWDBF16(const int64_t len, const __bf16 *src, const __bf16 *dd, __bf16 *ds);

void HardSigmoidLaVCFWDF32(const int64_t len, const float alpha, const float beta, const float *src, float *dst);
void HardSigmoidLaVCBWDF32(const int64_t len, const float alpha, const float beta,
    const float *src, const float *dd, float *ds);
void HardSigmoidLaVCFWDF16(const int64_t len, const float alpha, const float beta, const __fp16 *src, __fp16 *dst);
void HardSigmoidLaVCBWDF16(const int64_t len, const float alpha, const float beta,
    const __fp16 *src, const __fp16 *dd, __fp16 *ds);
void HardSigmoidLaVCFWDBF16(const int64_t len, const float alpha, const float beta, const __bf16 *src, __bf16 *dst);
void HardSigmoidLaVCBWDBF16(const int64_t len, const float alpha, const float beta,
    const __bf16 *src, const __bf16 *dd, __bf16 *ds);

void ExpLaVCFWDF32(const int64_t len, const float *src, float *dst);
void ExpLaVCFWDF16(const int64_t len, const __fp16 *src, __fp16 *dst);
void ExpLaVCFWDBF16(const int64_t len, const __bf16 *src, __bf16 *dst);
void ExpLaVCBWDF32(const int64_t len, const float *src, const float *dd, float *ds);
void ExpLaVCBWDF16(const int64_t len, const __fp16 *src, const __fp16 *dd, __fp16 *ds);
void ExpLaVCBWDBF16(const int64_t len, const __bf16 *src, const __bf16 *dd, __bf16 *ds);

void ExpLaDstVCBWDF32(const int64_t len, const float *src, const float *dd, float *ds);
void ExpLaDstVCBWDF16(const int64_t len, const __fp16 *src, const __fp16 *dd, __fp16 *ds);
void ExpLaDstVCBWDBF16(const int64_t len, const __bf16 *src, const __bf16 *dd, __bf16 *ds);

void EluLaVCFWDF32(const int64_t len, const float alpha, const float *src, float *dst);
void EluLaVCFWDF16(const int64_t len, const float alpha, const __fp16 *src, __fp16 *dst);
void EluLaVCFWDBF16(const int64_t len, const float alpha, const __bf16 *src, __bf16 *dst);
void EluLaVCBWDF32(const int64_t len, const float alpha, const float *src, const float *dd, float *ds);
void EluLaVCBWDF16(const int64_t len, const float alpha, const __fp16 *src, const __fp16 *dd, __fp16 *ds);
void EluLaVCBWDBF16(const int64_t len, const float alpha, const __bf16 *src, const __bf16 *dd, __bf16 *ds);

void SoftReluLaVCFWDF32(const int64_t len, const float alpha, const float *src, float *dst);
void SoftReluLaVCFWDF16(const int64_t len, const float alpha, const __fp16 *src, __fp16 *dst);
void SoftReluLaVCFWDBF16(const int64_t len, const float alpha, const __bf16 *src, __bf16 *dst);
void SoftReluLaVCBWDF32(const int64_t len, const float alpha, const float *src, const float *dd, float *ds);
void SoftReluLaVCBWDF16(const int64_t len, const float alpha, const __fp16 *src, const __fp16 *dd, __fp16 *ds);
void SoftReluLaVCBWDBF16(const int64_t len, const float alpha, const __bf16 *src, const __bf16 *dd, __bf16 *ds);

void LogisticDstVCBWDF32(const int64_t len, const float *src, const float *dd, float *ds);
void LogisticDstVCBWDF16(const int64_t len, const __fp16 *src, const __fp16 *dd, __fp16 *ds);
void LogisticDstVCBWDBF16(const int64_t len, const __bf16 *src, const __bf16 *dd, __bf16 *ds);

void TanhDstVCBWDF32(const int64_t len, const float *src, const float *dd, float *ds);
void TanhDstVCBWDF16(const int64_t len, const __fp16 *src, const __fp16 *dd, __fp16 *ds);
void TanhDstVCBWDBF16(const int64_t len, const __bf16 *src, const __bf16 *dd, __bf16 *ds);

void PowxaLaVCFWDF32(const int64_t len, const float alpha, const float y, const float *src, float *dst);
void PowxaLaVCBWDF32(const int64_t len, const float alpha, const float beta,
    const float *src, const float *dd, float *ds);
void PowxaLaVCFWDF16(const int64_t len, const float alpha, const float beta, const __fp16 *src, __fp16 *dst);
void PowxaLaVCBWDF16(const int64_t len, const float alpha, const float beta,
    const __fp16 *src, const __fp16 *dd, __fp16 *ds);
void PowxaLaVCFWDBF16(const int64_t len, const float alpha, const float beta, const __bf16 *src, __bf16 *dst);
void PowxaLaVCBWDBF16(const int64_t len, const float alpha, const float beta,
    const __bf16 *src, const __bf16 *dd, __bf16 *ds);

void TanhLaVCFWDF32(const int64_t len, const float *src, float *dst);
void TanhLaVCBWDF32(const int64_t len, const float *src, const float *dd, float *ds);
void TanhLaVCFWDF16(const int64_t len, const __fp16 *src, __fp16 *dst);
void TanhLaVCBWDF16(const int64_t len, const __fp16 *src, const __fp16 *dd, __fp16 *ds);
void TanhLaVCFWDBF16(const int64_t len, const __bf16 *src, __bf16 *dst);
void TanhLaVCBWDBF16(const int64_t len, const __bf16 *src, const __bf16 *dd, __bf16 *ds);

void MishLaVCFWDF32(const int64_t len, const float *src, float *dst);
void MishLaVCBWDF32(const int64_t len, const float *src, const float *dd, float *ds);
void MishLaVCFWDF16(const int64_t len, const __fp16 *src, __fp16 *dst);
void MishLaVCBWDF16(const int64_t len, const __fp16 *src, const __fp16 *dd, __fp16 *ds);
void MishLaVCFWDBF16(const int64_t len, const __bf16 *src, __bf16 *dst);
void MishLaVCBWDBF16(const int64_t len, const __bf16 *src, const __bf16 *dd, __bf16 *ds);

void GeluTanhLaVCFWDF32(const int64_t len, const float *src, float *dst);
void GeluTanhLaVCBWDF32(const int64_t len, const float *src, const float *dd, float *ds);
void GeluTanhLaVCFWDF16(const int64_t len, const __fp16 *src, __fp16 *dst);
void GeluTanhLaVCBWDF16(const int64_t len, const __fp16 *src, const __fp16 *dd, __fp16 *ds);
void GeluTanhLaVCFWDBF16(const int64_t len, const __bf16 *src, __bf16 *dst);
void GeluTanhLaVCBWDBF16(const int64_t len, const __bf16 *src, const __bf16 *dd, __bf16 *ds);

void GeluErfLaVCFWDF32(const int64_t len, const float *src, float *dst);
void GeluErfLaVCBWDF32(const int64_t len, const float *src, const float *dd, float *ds);
void GeluErfLaVCFWDF16(const int64_t len, const __fp16 *src, __fp16 *dst);
void GeluErfLaVCBWDF16(const int64_t len, const __fp16 *src, const __fp16 *dd, __fp16 *ds);
void GeluErfLaVCFWDBF16(const int64_t len, const __bf16 *src, __bf16 *dst);
void GeluErfLaVCBWDBF16(const int64_t len, const __bf16 *src, const __bf16 *dd, __bf16 *ds);

void SwishLaVCFWDF32(const int64_t len, const float alpha, const float *src, float *dst);
void SwishLaVCBWDF32(const int64_t len, const float alpha, const float *src, const float *dd, float *ds);
void SwishLaVCFWDBF16(const int64_t len, const float alpha, const __bf16 *src, __bf16 *dst);
void SwishLaVCBWDBF16(const int64_t len, const float alpha, const __bf16 *src, const __bf16 *dd, __bf16 *ds);
void SwishLaVCFWDF16(const int64_t len, const float alpha, const __fp16 *src, __fp16 *dst);
void SwishLaVCBWDF16(const int64_t len, const float alpha, const __fp16 *src, const __fp16 *dd, __fp16 *ds);

void LnLaVCFWDF32(const int64_t len, const float *src, float *dst);
void LnLaVCBWDF32(const int64_t len, const float *src, const float *dd, float *ds);
void LnLaVCFWDF16(const int64_t len, const __fp16 *src, __fp16 *dst);
void LnLaVCBWDF16(const int64_t len, const __fp16 *src, const __fp16 *dd, __fp16 *ds);
void LnLaVCFWDBF16(const int64_t len, const __bf16 *src, __bf16 *dst);
void LnLaVCBWDBF16(const int64_t len, const __bf16 *src, const __bf16 *dd, __bf16 *ds);

void NearbyintFWDF32(const int64_t len, const float *src, float *dst);
void NearbyintFWDF16(const int64_t len, const __fp16 *src, __fp16 *dst);
void NearbyintVCFWDBF16(const int64_t len, const __bf16 *src, __bf16 *dst);

void SqrtLaFWDF32(const int64_t len, const float *src, float *dst);
void SqrtLaVCBWDF32(const int64_t len, const float *src, const float *dd, float *ds);
void SqrtLaFWDF16(const int64_t len, const __fp16 *src, __fp16 *dst);
void SqrtLaVCBWDF16(const int64_t len, const __fp16 *src, const __fp16 *dd, __fp16 *ds);
void SqrtLaVCFWDBF16(const int64_t len, const __bf16 *src, __bf16 *dst);
void SqrtLaVCBWDBF16(const int64_t len, const __bf16 *src, const __bf16 *dd, __bf16 *ds);

void SqrLaFWDF32(const int64_t len, const float *src, float *dst);
void SqrLaBWDF32(const int64_t len, const float *src, const float *dd, float *ds);
void SqrLaFWDF16(const int64_t len, const __fp16 *src, __fp16 *dst);
void SqrLaVCBWDF16(const int64_t len, const __fp16 *src, const __fp16 *dd, __fp16 *ds);
void SqrLaVCFWDBF16(const int64_t len, const __bf16 *src, __bf16 *dst);
void SqrLaVCBWDBF16(const int64_t len, const __bf16 *src, const __bf16 *dd, __bf16 *ds);

void SigmoidVCFWDF32(const int64_t len, const float *src, float *dst);
void SigmoidVCBWDF32(const int64_t len, const float *src, const float *diffDst, float *diffSrc);

void SigmoidVCFWDF16(const int64_t len, const __fp16 *src, __fp16 *dst);
void SigmoidVCBWDF16(const int64_t len, const __fp16 *src, const __fp16 *diffDst, __fp16 *diffSrc);

void ReluVCFWDF32(const int64_t len, const float alpha, const float *src, float *dst);
void ReluVCBWDF32(const int64_t len, const float alpha, const float *src, const float *dd, float *ds);

void ReluVCFWDF16(const int64_t len, const float alpha, const __fp16 *src, __fp16 *dst);
void ReluVCBWDF16(const int64_t len, const float alpha, const __fp16 *src, const __fp16 *dd, __fp16 *ds);

void ReluVCFWDBF16(const int64_t len, const float alpha, const __bf16 *src, __bf16 *dst);
void ReluVCBWDBF16(const int64_t len, const float alpha, const __bf16 *src, const __bf16 *dd, __bf16 *ds);

void ClipFWDF32(const int64_t len, const float alpha, const float beta, const float *src, float *dst);
void ClipVCBWDF32(const int64_t len, const float alpha, const float beta,
                   const float *src, const float *dd, float *ds);
void ClipFWDF16(const int64_t len, const float alpha, const float beta, const __fp16 *src, __fp16 *dst);
void ClipVCBWDF16(const int64_t len, const float alpha, const float beta,
                   const __fp16 *src, const __fp16 *dd, __fp16 *ds);
void ClipVCFWDBF16(const int64_t len, const float alpha, const float beta, const __bf16 *src, __bf16 *dst);
void ClipVCBWDBF16(const int64_t len, const float alpha, const float beta,
                    const __bf16 *src, const __bf16 *dd, __bf16 *ds);

void ClipV2VCFWDF32(const int64_t len, const float alpha, const float beta, const float *src, float *dst);
void ClipV2VCBWDF32(const int64_t len, const float alpha, const float beta,
                     const float *src, const float *dd, float *ds);
void ClipV2FWDF16(const int64_t len, const float alpha, const float beta, const __fp16 *src, __fp16 *dst);
void ClipV2VCBWDF16(const int64_t len, const float alpha, const float beta,
                     const __fp16 *src, const __fp16 *dd, __fp16 *ds);
void ClipV2VCFWDBF16(const int64_t len, const float alpha, const float beta, const __bf16 *src, __bf16 *dst);
void ClipV2VCBWDBF16(const int64_t len, const float alpha, const float beta,
                      const __bf16 *src, const __bf16 *dd, __bf16 *ds);

void HardSwishVCFWDF32(const int64_t len, const float alpha, const float beta, const float *src, float *dst);
void HardSwishVCBWDF32(const int64_t len, const float alpha, const float beta,
                        const float *src, const float *dd, float *ds);
void HardSwishVCFWDF16(const int64_t len, const float alpha, const float beta, const __fp16 *src, __fp16 *dst);
void HardSwishVCBWDF16(const int64_t len, const float alpha, const float beta,
                        const __fp16 *src, const __fp16 *dd, __fp16 *ds);
void HardSwishVCFWDBF16(const int64_t len, const float alpha, const float beta, const __bf16 *src, __bf16 *dst);
void HardSwishVCBWDBF16(const int64_t len, const float alpha, const float beta,
                         const __bf16 *src, const __bf16 *dd, __bf16 *ds);

void SigmoidVCFWDBF16(const int64_t len, const __bf16 *src, __bf16 *dst);
void SigmoidVCBWDBF16(const int64_t len, const __bf16 *src, const __bf16 *diffDst, __bf16 *diffSrc);

void LinearVCF32(const int64_t len, const float alpha, const float beta, const float *src, float *dst);
void LinearVCF16(const int64_t len, const float alpha, const float beta, const __fp16 *src, __fp16 *dst);
void LinearVCBF16(const int64_t len, const float alpha, const float beta, const __bf16 *src, __bf16 *dst);

void SoftmaxAxisF32(const float *srcPtr, float *dstPtr, size_t axisSize, size_t outerSize);
void SoftmaxCrossAxisF32(const float *srcPtr, float *dstPtr, size_t innerSize, size_t axisSize, size_t outerSize);

void SoftmaxAxisF16(const __fp16 *srcPtr, __fp16 *dstPtr, size_t axisSize, size_t outerSize);
void SoftmaxCrossAxisF16(const __fp16 *srcPtr, __fp16 *dstPtr, size_t innerSize, size_t axisSize, size_t outerSize);

void SoftmaxAxisBF16(const __bf16 *srcPtr, __bf16 *dstPtr, size_t axisSize, size_t outerSize);
void SoftmaxCrossAxisBF16(const __bf16 *srcPtr, __bf16 *dstPtr, size_t innerSize, size_t axisSize, size_t outerSize);

void ClipV2DstVCBWDF32(const int64_t len, const float alpha, const float beta, const float *src, const float *dd,
    float *ds);
void ClipV2DstVCBWDF16(const int64_t len, const float alpha, const float beta, const __fp16 *src, const __fp16 *dd,
    __fp16 *ds);
void ClipV2DstVCBWDBF16(const int64_t len, const float alpha, const float beta, const __bf16 *src, const __bf16 *dd,
    __bf16 *ds);

void EluDstVCBWDF32(const int64_t len, const float alpha, const float *src, const float *dd, float *ds);
void EluDstVCBWDF16(const int64_t len, const float alpha, const __fp16 *src, const __fp16 *dd, __fp16 *ds);
void EluDstVCBWDBF16(const int64_t len, const float alpha, const __bf16 *src, const __bf16 *dd, __bf16 *ds);

void ReluDstVCBWDF32(const int64_t len, const float alpha, const float *src, const float *dd, float *ds);
void ReluDstVCBWDF16(const int64_t len, const float alpha, const __fp16 *src, const __fp16 *dd, __fp16 *ds);
void ReluDstVCBWDBF16(const int64_t len, const float alpha, const __bf16 *src, const __bf16 *dd, __bf16 *ds);

void SqrtDstVCBWDF32(const int64_t len, const float *src, const float *dd, float *ds);
void SqrtDstVCBWDF16(const int64_t len, const __fp16 *src, const __fp16 *dd, __fp16 *ds);
void SqrtDstVCBWDBF16(const int64_t len, const __bf16 *src, const __bf16 *dd, __bf16 *ds);

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
