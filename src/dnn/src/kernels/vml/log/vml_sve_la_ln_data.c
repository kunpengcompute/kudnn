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

#include "vml_sve_la_ln.h"

const float G_VML_VectorComputation_VSLOG_POLY3578[VectorComputation_FLOAT_LD_POLY_SIZE] = {
    0x1.555d7cp-2f, 0x1.961348p-3f,  0x1.5a9aa2p-3f, -0x1.3e737cp-3f, // C3 C5 C7 C8
    0x1.555d7cp-2f, 0x1.961348p-3f,  0x1.5a9aa2p-3f, -0x1.3e737cp-3f, 0x1.555d7cp-2f, 0x1.961348p-3f,
    0x1.5a9aa2p-3f, -0x1.3e737cp-3f, 0x1.555d7cp-2f, 0x1.961348p-3f,  0x1.5a9aa2p-3f, -0x1.3e737cp-3f,
};

const float G_VML_VectorComputation_VSLN_POLY[0x3] = {
    -0x1.ffffc8p-2f, -0x1.00187cp-2f, -0x1.4f9934p-3f, // C2 C4 C6
};
