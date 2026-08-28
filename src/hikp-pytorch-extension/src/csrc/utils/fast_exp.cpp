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

#include "fast_exp.h"

svfloat32_t FastExp(svbool_t pg, svfloat32_t values)
{
    const float factorial1 = 0.999999701f;
    const float factorial2 = 0.499991506f;
    const float factorial3 = 0.166676521f;                         // 1 / factorial(3)
    const float factorial4 = 0.0418978221f;                        // 1 / factorial(4)
    const svfloat32_t vecFactorial5 = svdup_f32(0.00828929059f);  // 1 / factorial(5)
    const svfloat32_t vecExpLog2e = svdup_f32(1.4426951f);       // log2(e)
    const svfloat32_t vecZero = svdup_f32(0.f);
    const svfloat32_t ln2f = svdup_f32(0.6931472f);  // ln(2)
    const float vecLnFltMin = -87.33655;
    const float vecLnFltMax = 88.72284;
    const int expOffset = 0x7f;
    const int nMantissaBits = 23;
    const float expConstDouble = 2;

    svbool_t lessLnFltMinMask = svcmplt(pg, values, vecLnFltMin);
    svfloat32_t vecSrc = svmin_x(pg, values, vecLnFltMax);

    svfloat32_t vecFx = svmad_x(pg, vecSrc, vecExpLog2e, 0.5f);
    vecFx = svrintm_x(pg, vecFx);

    svfloat32_t vecExpPoly = svmsb_x(pg, vecFx, ln2f, vecSrc);

    svfloat32_t vecRes = svmad_x(pg, vecExpPoly, vecFactorial5, factorial4);
    vecRes = svmad_x(pg, vecExpPoly, vecRes, factorial3);
    vecRes = svmad_x(pg, vecExpPoly, vecRes, factorial2);
    vecRes = svmad_x(pg, vecExpPoly, vecRes, factorial1);
    vecRes = svmad_x(pg, vecExpPoly, vecRes, 1);

    svfloat32_t vecExpNumber = svsub_x(pg, vecFx, 1);
    svint32_t vecExpNumberRes = svcvt_s32_x(pg, svrintn_x(pg, vecExpNumber));
    svint32_t vecTwoPowInt = svadd_x(pg, vecExpNumberRes, expOffset);
    vecTwoPowInt = svlsl_x(pg, vecTwoPowInt, nMantissaBits);
    svfloat32_t vecTwoPowF32 = svreinterpret_f32(vecTwoPowInt);

    vecRes = svmul_x(pg, vecRes, vecTwoPowF32);
    vecRes = svmul_x(pg, vecRes, expConstDouble);
    vecRes = svsel(lessLnFltMinMask, vecZero, vecRes);
    return vecRes;
}