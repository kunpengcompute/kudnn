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

#include "vml_sve_la_tanh.h"

const struct VmlVCTanhfData G_VML_VSTANH_DATA = {
    .min = 0x1.bb67aep-12f, // 0x39ddb3d7
    .max = 0x1.205966p3f, // 0x41102cb3
    .idxMin = 0x1p-12f,
    .idxMax = 0x1p3f,
    .idxShift = 0x16U, // 22
    .idxBias = 0xe6U, // 0x39800000 >> 22
    .idxMask = 0xffc00000U,
    .polyDegree0 = {
        0.0f, 0x1.7ffffep-12f, 0x1.fffffep-12f, 0x1.7ffffcp-11f,
        0x1.fffff6p-11f, 0x1.7fffeep-10f, 0x1.ffffd6p-10f, 0x1.7fffb8p-9f,
        0x1.ffff56p-9f, 0x1.7ffeep-8f, 0x1.fffd56p-8f, 0x1.7ffb8p-7f,
        0x1.fff556p-7f, 0x1.7fee02p-6f, 0x1.ffd55ap-6f, 0x1.7fb81p-5f,
        0x1.ff559ap-5f, 0x1.7ee102p-4f, 0x1.fd5992p-4f, 0x1.7b8ffap-3f,
        0x1.f597eap-3f, 0x1.6ef53ep-2f, 0x1.d9353ep-2f, 0x1.45323ep-1f,
        0x1.85efacp-1f, 0x1.cf6f98p-1f, 0x1.ed9506p-1f, 0x1.fd77d2p-1f,
        0x1.ffa818p-1f, 0x1.fffe64p-1f, 0x1.fffff8p-1f, 0x1p0f,
    },
    .polyDegree1 = {
        0x1p0f, 0x1.00003p0f, 0x1.ffffdp-1f, 0x1.ffffb4p-1f,
        0x1.ffffb8p-1f, 0x1.ffffb8p-1f, 0x1.ffff58p-1f, 0x1.fffeep-1f,
        0x1.fffdd8p-1f, 0x1.fffb8p-1f, 0x1.fff7dap-1f, 0x1.ffee08p-1f,
        0x1.ffdfeap-1f, 0x1.ffb794p-1f, 0x1.ff7ffep-1f, 0x1.fee082p-1f,
        0x1.fe0136p-1f, 0x1.fb86d8p-1f, 0x1.f8155p-1f, 0x1.ee697p-1f,
        0x1.e149bcp-1f, 0x1.be3fbp-1f, 0x1.92a926p-1f, 0x1.3173d8p-1f,
        0x1.ae0d96p-2f, 0x1.72160cp-3f, 0x1.21623ep-4f, 0x1.4340a6p-7f,
        0x1.5effb6p-10f, 0x1.998346p-16f, 0x1.b8ce66p-21f, 0.0f,
    },
    .polyDegree2 = {
        0.0f, -0x1.9c1fe2p-3f, 0x1.4b6362p-5f, 0x1.d6db56p-5f,
        0x1.3f63aap-6f, -0x1.7fe0dep-10f, 0x1.0f67ecp-7f, -0x1.7f8378p-9f,
        0x1.3eb242p-10f, -0x1.7e0de4p-8f, -0x1.61e804p-8f, -0x1.8fb73cp-7f,
        -0x1.e7abcep-7f, -0x1.44b7b4p-6f, -0x1.f94fp-6f, -0x1.81c0f8p-5f,
        -0x1.fb5606p-5f, -0x1.7c941ep-4f, -0x1.f6294ap-4f, -0x1.6d991ap-3f,
        -0x1.d7a204p-3f, -0x1.3fcf8ap-2f, -0x1.741e2p-2f, -0x1.840d5p-2f,
        -0x1.47871p-2f, -0x1.4efac4p-3f, -0x1.16f2cp-4f, -0x1.413e92p-7f,
        -0x1.5a8994p-10f, -0x1.8ddd58p-16f, -0x1.c75482p-19f, 0.0f,
    },
    .polyDegree3 = {
        0.0f, 0x1.675d2cp12f, -0x1.29d64p9f, -0x1.a1c05cp9f,
        -0x1.2a569cp7f, -0x1.5a5eacp-2f, -0x1.2c5c04p5f, -0x1.697ab4p-2f,
        -0x1.34b348p3f, -0x1.a5ea0ep-2f, -0x1.41a658p1f, 0x1.a1ea0cp-4f,
        -0x1.54ebc4p-1f, -0x1.f2a1c6p0f, -0x1.a8e668p-2f, -0x1.013718p-2f,
        -0x1.6c8a64p-2f, -0x1.2c34b6p-2f, -0x1.36c758p-2f, -0x1.41a964p-2f,
        -0x1.0514eep-2f, -0x1.6f0c24p-3f, -0x1.84121p-4f, 0x1.5a72aep-5f,
        0x1.a8dd12p-4f, 0x1.67ec52p-4f, 0x1.58bcf6p-5f, 0x1.a40806p-8f,
        0x1.b3bf5cp-11f, 0x1.e15e8ap-17f, 0x1.598028p-17f, 0.0f,
    },
    .polyDegree4 = {
        0.0f, -0x1.303436p26f, 0x1.fdba7ap21f, 0x1.6200f8p22f,
        0x1.fdbb38p18f, 0x1.4aaf6ap4f, 0x1.fdc654p15f, 0x1.4aafeap5f,
        0x1.fe79c8p12f, 0x1.4aac82p6f, 0x1.dc10cep9f, -0x1.7b8334p7f,
        0x1.22a428p6f, 0x1.5e9f5ap8f, 0x1.21fd1p3f, -0x1.1336eap3f,
        0x1.9151b8p0f, -0x1.7d5deap0f, -0x1.6cb55ap-3f, 0x1.9b2ca4p-1f,
        0x1.bf411ep-4f, 0x1.69d37p-3f, 0x1.5a0f4cp-3f, 0x1.8c6acep-4f,
        0x1.b5d6fp-6f, -0x1.a2ea6ep-6f, -0x1.250538p-6f, -0x1.875732p-9f,
        -0x1.68e3bap-12f, -0x1.755ab4p-18f, -0x1.175f8ep-16f, 0.0f,
    },
    .polyDegree5 = {
        0.0f, 0x1.ed11aap38f, -0x1.a0b8e4p33f, -0x1.1f31c6p34f,
        -0x1.a0b992p29f, -0x1.2c5714p15f, -0x1.a0c3acp25f, -0x1.2c57bep15f,
        -0x1.a166fep21f, -0x1.2c5474p15f, -0x1.84410cp17f, 0x1.34e29cp15f,
        -0x1.d9c3e2p12f, -0x1.1c7b2p15f, -0x1.d75294p8f, 0x1.c18848p8f,
        -0x1.3e8aa4p5f, 0x1.42f99p5f, 0x1.bcfb88p1f, -0x1.1ba802p3f,
        0x1.2d6c04p-2f, -0x1.34da5ep-5f, -0x1.727106p-5f, -0x1.ce8d04p-5f,
        -0x1.62002cp-5f, -0x1.c3c4d2p-13f, 0x1.46492ap-8f, 0x1.ed01aap-11f,
        0x1.662e78p-14f, 0x1.53bdd4p-20f, 0x1.b87e54p-17f, 0.0f,
    },
    .polyDegree6 = {
        0.0f, -0x1.32bda2p50f, 0x1.050bd4p44f, 0x1.659ad2p44f,
        0x1.050c4ap39f, 0x1.93323ep24f, 0x1.051314p34f, 0x1.933366p23f,
        0x1.0580f8p29f, 0x1.932ec8p22f, 0x1.e59044p23f, -0x1.8181c2p21f,
        0x1.2827c8p18f, 0x1.624448p20f, 0x1.26989cp13f, -0x1.185aaep13f,
        0x1.90722p8f, -0x1.90e5a2p8f, -0x1.0d793cp4f, 0x1.64b786p5f,
        -0x1.5ff494p-1f, 0x1.344078p-4f, -0x1.a8b486p-7f, -0x1.c11fdcp-10f,
        0x1.0044bap-6f, 0x1.3fa3bep-9f, -0x1.6d73a2p-11f, -0x1.3bca88p-13f,
        -0x1.3e201ep-17f, -0x1.11c5dap-23f, -0x1.0bfd9p-18f, 0.0f,
    },
};
