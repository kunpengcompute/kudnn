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

#ifndef VML_MODE_H
#define VML_MODE_H

#include "types.h"
#include "kernel_funcs.h"

#define __vml_absh       AbsLaFWDF16
#define __vml_absb       AbsLaFWDBF16
#define __vml_absf       AbsLaFWDF32
#define __vml_absh_bwd   AbsLaBWDF16
#define __vml_absb_bwd   AbsLaVCBWDBF16
#define __vml_absf_bwd   AbsLaVCBWDF32

#define __vml_hardsigmoidh       HardSigmoidLaVCFWDF16
#define __vml_hardsigmoidb       HardSigmoidLaVCFWDBF16
#define __vml_hardsigmoidf       HardSigmoidLaVCFWDF32
#define __vml_hardsigmoidh_bwd   HardSigmoidLaVCBWDF16
#define __vml_hardsigmoidb_bwd   HardSigmoidLaVCBWDBF16
#define __vml_hardsigmoidf_bwd   HardSigmoidLaVCBWDF32

#define __vml_hardswishh       HardSwishVCFWDF16
#define __vml_hardswishb       HardSwishVCFWDBF16
#define __vml_hardswishf       HardSwishVCFWDF32
#define __vml_hardswishh_bwd   HardSwishVCBWDF16
#define __vml_hardswishb_bwd   HardSwishVCBWDBF16
#define __vml_hardswishf_bwd   HardSwishVCBWDF32

#define __vml_nearbyinth NearbyintFWDF16
#define __vml_nearbyintf NearbyintFWDF32
#define __vml_nearbyintb NearbyintVCFWDBF16

#define __vml_exph       ExpLaVCFWDF16
#define __vml_exph_bwd   ExpLaVCBWDF16
#define __vml_expf       ExpLaVCFWDF32
#define __vml_expf_bwd   ExpLaVCBWDF32
#define __vml_expb       ExpLaVCFWDBF16
#define __vml_expb_bwd   ExpLaVCBWDBF16

#define __vml_eluh       EluLaVCFWDF16
#define __vml_elub       EluLaVCFWDBF16
#define __vml_eluf       EluLaVCFWDF32
#define __vml_eluh_bwd   EluLaVCBWDF16
#define __vml_elub_bwd   EluLaVCBWDBF16
#define __vml_eluf_bwd   EluLaVCBWDF32

#define __vml_soft_reluh        SoftReluLaVCFWDF16
#define __vml_soft_relub        SoftReluLaVCFWDBF16
#define __vml_soft_reluf        SoftReluLaVCFWDF32
#define __vml_soft_reluh_bwd    SoftReluLaVCBWDF16
#define __vml_soft_relub_bwd    SoftReluLaVCBWDBF16
#define __vml_soft_reluf_bwd    SoftReluLaVCBWDF32

#define __vml_logh       LnLaVCFWDF16
#define __vml_logb       LnLaVCFWDBF16
#define __vml_logf       LnLaVCFWDF32
#define __vml_logh_bwd   LnLaVCBWDF16
#define __vml_logb_bwd   LnLaVCBWDBF16
#define __vml_logf_bwd   LnLaVCBWDF32

#define __vml_sqrth      SqrtLaFWDF16
#define __vml_sqrtb      SqrtLaVCFWDBF16
#define __vml_sqrtf      SqrtLaFWDF32
#define __vml_sqrth_bwd  SqrtLaVCBWDF16
#define __vml_sqrtb_bwd  SqrtLaVCBWDBF16
#define __vml_sqrtf_bwd  SqrtLaVCBWDF32

#define __vml_sqrh      SqrLaFWDF16
#define __vml_sqrb      SqrLaVCFWDBF16
#define __vml_sqrf      SqrLaFWDF32
#define __vml_sqrh_bwd  SqrLaVCBWDF16
#define __vml_sqrb_bwd  SqrLaVCBWDBF16
#define __vml_sqrf_bwd  SqrLaBWDF32

#define __vml_tanhh      TanhLaVCFWDF16
#define __vml_tanhb      TanhLaVCFWDBF16
#define __vml_tanhf      TanhLaVCFWDF32
#define __vml_tanhh_bwd  TanhLaVCBWDF16
#define __vml_tanhb_bwd  TanhLaVCBWDBF16
#define __vml_tanhf_bwd  TanhLaVCBWDF32

#define __vml_mishf      MishLaVCFWDF32
#define __vml_mishh      MishLaVCFWDF16
#define __vml_mishb      MishLaVCFWDBF16
#define __vml_mishf_bwd  MishLaVCBWDF32
#define __vml_mishh_bwd  MishLaVCBWDF16
#define __vml_mishb_bwd  MishLaVCBWDBF16

#define __vml_gelu_tanhf      GeluTanhLaVCFWDF32
#define __vml_gelu_tanhh      GeluTanhLaVCFWDF16
#define __vml_gelu_tanhb      GeluTanhLaVCFWDBF16
#define __vml_gelu_tanhf_bwd  GeluTanhLaVCBWDF32
#define __vml_gelu_tanhh_bwd  GeluTanhLaVCBWDF16
#define __vml_gelu_tanhb_bwd  GeluTanhLaVCBWDBF16

#define __vml_swishb      SwishLaVCFWDBF16
#define __vml_swishf      SwishLaVCFWDF32
#define __vml_swishh      SwishLaVCFWDF16
#define __vml_swishb_bwd  SwishLaVCBWDBF16
#define __vml_swishf_bwd  SwishLaVCBWDF32
#define __vml_swishh_bwd  SwishLaVCBWDF16

#define __vml_sigmoidh      SigmoidVCFWDF16
#define __vml_sigmoidb      SigmoidVCFWDBF16
#define __vml_sigmoidf      SigmoidVCFWDF32
#define __vml_sigmoidh_bwd  SigmoidVCBWDF16
#define __vml_sigmoidb_bwd  SigmoidVCBWDBF16
#define __vml_sigmoidf_bwd  SigmoidVCBWDF32

#define __vml_reluh      ReluVCFWDF16
#define __vml_relub      ReluVCFWDBF16
#define __vml_reluf      ReluVCFWDF32
#define __vml_reluh_bwd  ReluVCBWDF16
#define __vml_relub_bwd  ReluVCBWDBF16
#define __vml_reluf_bwd  ReluVCBWDF32

#define __vml_linearh      LinearVCF16
#define __vml_linearb      LinearVCBF16
#define __vml_linearf      LinearVCF32

#define __vml_powxah     PowxaLaVCFWDF16
#define __vml_powxab     PowxaLaVCFWDBF16
#define __vml_powxaf     PowxaLaVCFWDF32
#define __vml_powxah_bwd PowxaLaVCBWDF16
#define __vml_powxab_bwd PowxaLaVCBWDBF16
#define __vml_powxaf_bwd PowxaLaVCBWDF32

#define __vml_clipf      ClipFWDF32
#define __vml_cliph      ClipFWDF16
#define __vml_clipb      ClipVCFWDBF16
#define __vml_clipf_bwd  ClipVCBWDF32
#define __vml_cliph_bwd  ClipVCBWDF16
#define __vml_clipb_bwd  ClipVCBWDBF16
#define __vml_clip_v2f      ClipV2VCFWDF32
#define __vml_clip_v2h      ClipV2FWDF16
#define __vml_clip_v2b      ClipV2VCFWDBF16
#define __vml_clip_v2f_bwd  ClipV2VCBWDF32
#define __vml_clip_v2h_bwd  ClipV2VCBWDF16
#define __vml_clip_v2b_bwd  ClipV2VCBWDBF16

#define __vml_clipv2dstf_bwd     ClipV2DstVCBWDF32
#define __vml_clipv2dsth_bwd     ClipV2DstVCBWDF16
#define __vml_clipv2dstb_bwd     ClipV2DstVCBWDBF16

#define __vml_eludstf_bwd     EluDstVCBWDF32
#define __vml_eludsth_bwd     EluDstVCBWDF16
#define __vml_eludstb_bwd     EluDstVCBWDBF16

#define __vml_reludstf_bwd     ReluDstVCBWDF32
#define __vml_reludsth_bwd     ReluDstVCBWDF16
#define __vml_reludstb_bwd     ReluDstVCBWDBF16

#define __vml_sqrtdstf_bwd     SqrtDstVCBWDF32
#define __vml_sqrtdsth_bwd     SqrtDstVCBWDF16
#define __vml_sqrtdstb_bwd     SqrtDstVCBWDBF16

#define __vml_expdstf_bwd    ExpLaDstVCBWDF32
#define __vml_expdsth_bwd    ExpLaDstVCBWDF16
#define __vml_expdstb_bwd    ExpLaDstVCBWDBF16
#define __vml_logisticdstf_bwd    LogisticDstVCBWDF32
#define __vml_logisticdsth_bwd    LogisticDstVCBWDF16
#define __vml_logisticdstb_bwd    LogisticDstVCBWDBF16
#define __vml_tanhdstf_bwd   TanhDstVCBWDF32
#define __vml_tanhdsth_bwd   TanhDstVCBWDF16
#define __vml_tanhdstb_bwd   TanhDstVCBWDBF16

#define __vml_gelu_erfh         GeluErfLaVCFWDF16
#define __vml_gelu_erfb         GeluErfLaVCFWDBF16
#define __vml_gelu_erff         GeluErfLaVCFWDF32
#define __vml_gelu_erfh_bwd     GeluErfLaVCBWDF16
#define __vml_gelu_erfb_bwd     GeluErfLaVCBWDBF16
#define __vml_gelu_erff_bwd     GeluErfLaVCBWDF32

#define FLOAT_ABS_MASK   0x7fffffffU
#define FLOAT_SIGN_BIT   0x80000000U

static inline uint32_t GetFloatSign(float x)
{
    return Asuint(x) & FLOAT_SIGN_BIT;
}

static inline float SetFloatSign(float x, uint32_t sign)
{
    return Asfloat((Asuint(x) & FLOAT_ABS_MASK) | sign);
}

#endif
