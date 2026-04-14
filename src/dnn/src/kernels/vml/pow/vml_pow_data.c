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

#include "vml_pow.h"

// -log2(1 + i/32), i in [0, 31]
__attribute__((aligned(0x20))) const float G_VML_VSPOW_LOG2_TAB_VectorComputation[POWF_LOG2_N * 2] = {
    0x0p+0f, -0x1.6bbp-5f, -0x1.664p-4f, -0x1.08c8p-3f,
    -0x1.5cp-3f, -0x1.acf8p-3f, -0x1.fbcp-3f, -0x1.244p-2f,
    -0x1.49a8p-2f, -0x1.6e2p-2f, -0x1.91b8p-2f, -0x1.b48p-2f,
    -0x1.d678p-2f, -0x1.f7a8p-2f, -0x1.0c1p-1f, -0x1.1bfp-1f,
    -0x1.2b8p-1f, -0x1.3ab8p-1f, -0x1.49a8p-1f, -0x1.5848p-1f,
    -0x1.66ap-1f, -0x1.74bp-1f, -0x1.828p-1f, -0x1.901p-1f,
    -0x1.9d6p-1f, -0x1.aa7p-1f, -0x1.b748p-1f, -0x1.c3e8p-1f,
    -0x1.d05p-1f, -0x1.dc88p-1f, -0x1.e89p-1f, -0x1.f46p-1f,
};

__attribute__((aligned(0x20))) const float G_VML_VSPOW_LOG2_POLY_LOW_TAB_VectorComputation[POWF_LOG2_POLY_N * 2] =  {
    0x0p+0f, 0x1.645388p-20f, 0x1.20a6dep-21f, 0x1.3b992cp-18f,
    -0x1.a39fbep-19f, 0x1.0e9258p-18f, -0x1.6b9026p-19f, -0x1.eac382p-20f,
    0x1.ed0cbap-20f, -0x1.0e6ce8p-17f, -0x1.d448f8p-17f, 0x1.408c78p-18f,
    0x1.60fe68p-17f, -0x1.5a32c2p-20f, -0x1.40358ep-19f, -0x1.88f4aep-16f,
    -0x1.a39fbep-20f, -0x1.9fd502p-16f, 0x1.ed0cbap-19f, -0x1.134c4ep-20f,
    -0x1.1c8f12p-22f, -0x1.fd64ep-17f, -0x1.3ab7cep-18f, 0x1.9eap-17f,
    0x1.30158p-16f, -0x1.1eb002p-18f, -0x1.48f554p-17f, -0x1.ca2e1ap-17f,
    -0x1.fb693p-16f, -0x1.9ab4p-17f, 0x1.ca64ecp-16f, 0x1.f7431p-17f,
};
