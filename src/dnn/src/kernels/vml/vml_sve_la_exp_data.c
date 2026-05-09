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

#include "vml_sve_la_exp.h"

const struct SveExpfData G_VectorComputation_VSEXP_DATA = {
    .invLn2Hi = 0x1.715476p+0f,
    .invLn2Lo = 0x1.4ae0cp-26f,
    .splThR = 87.0f,           // ~ln(0x1p126)
    .upBound = 0x1.62e43p6f,   // ~ln(0x1.fffffep127);
    .lowBound = -0x1.5d58ap6f, // ~ln(FLT_MIN)
};

const struct SveExp2fData G_VectorComputation_VSEXP2_DATA = {
    .exp2fShift = 0x1.803f8p17f, // shift/0x1p6 add exp bias
    .splTh = 126.0f,
    /*
     * p=fpminimax(2^x, [|0,1,2|], [|23...|], [-2^-6; 2^-6], relative)
     * supnorm(p,2^x,[-2^-6; 2^-6],relative,0x1p-23);
     * maxUlp [0x1.cbaa892816p-25;0x1.cbaa8ca2b069bdaaap-25]
     */
    .poly =
        {
            0x1.ebfc28p-3f,
            0x1.62e584p-1f,
            1.0f,
        },
};

__attribute__((aligned(0x20))) const struct VectorComputation_EXPM1F_DATA G_VectorComputation_EXPM1F_DATA = {
    /* coefficients of expm1(r) r \in [-ln2/2, ln2/2]
     * f = expm1(x) / x
     * p = fpminimax(f, [|0,1,2,3,4,5|], [|23...|], [-log(2)/2,log(2)/2], relative)
     * maxerr = 0x1.1bbef1a66bp-26
     */
    .c0 = 0x1.fffffc00p-0002f, .c1 = 0x1.5554ac00p-0003f, .c2 = 0x1.5558b800p-0005f, .c3 = 0x1.1229f000p-0007f,
    .c4 = 0x1.6a403c00p-0010f, .ln2Hi = 0x1.62e4p-1f,     .ln2Lo = 0x1.7f7d1cp-20f,  .invLn2 = 0x1.715476p+0f,
    .maxThr = 0x1.5a92d6p6f,   .shift = 0x1.8000fep23f,
};
