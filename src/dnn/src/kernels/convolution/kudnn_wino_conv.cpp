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
#include <algorithm>
#include <memory>
#include <utility>
#include <numeric>
#include <functional>
#include <arm_sve.h>
#include "kudnn_conv_const.hpp"
#include "kudnn_wino_conv.hpp"
namespace KuDNN {
namespace Detail {
void GemmForWinoImpl::DoGemmForWino(const IntType& ic, const IntType& oc, const IntType& weiMatStride,
    const float* inPtr, const float* weiPtr, float* outPtr)
{
    IntType inStride = ic % VLEN_NEON_F32 == 0 ? ic : (ic / VLEN_NEON_F32 + NUM_1) * VLEN_NEON_F32;
    IntType outStride = oc % VLEN_NEON_F32 == 0 ? oc : (oc / VLEN_NEON_F32 + NUM_1) * VLEN_NEON_F32;
    svfloat32_t in0;
    svfloat32_t in1;
    svfloat32_t in2;
    svfloat32_t in3;
    svfloat32_t in4;
    svfloat32_t in5;
    svfloat32_t wei0;
    svfloat32_t wei1;
    svfloat32_t wei2;
    svfloat32_t wei3;
    svfloat32_t wei4;
    svfloat32_t wei5;
    svfloat32_t out0 = svdup_n_f32(0.0f);
    svfloat32_t out1 = svdup_n_f32(0.0f);
    svfloat32_t out2 = svdup_n_f32(0.0f);
    svfloat32_t out3 = svdup_n_f32(0.0f);
    svfloat32_t out4 = svdup_n_f32(0.0f);
    svfloat32_t out5 = svdup_n_f32(0.0f);
    int32_t realVl = svcntw();
    IntType icTail = ic % VLEN_NEON_F32;
    IntType icLoop = ic / VLEN_NEON_F32;
    IntType ocTail = oc % realVl;
    IntType ocLoop = oc / realVl;

    const svbool_t pg = svwhilelt_b32_s32(0, realVl);
    IntType i = 0;
    IntType occ = 0;
    if (icLoop > 0 && ocLoop > 0 && ocTail == 0) {
        for (IntType j = 0; j < NUM_6; j++) { // per channel has 36 data, need to calcu 6 times
            const float* tInPtr = inPtr;
            const float* tWeiPtr = weiPtr;
            for (i = 0; i < ic; i += VLEN_NEON_F32) {
                tWeiPtr = weiPtr;
                in0 = svld1rq_f32(pg, tInPtr + inStride * (0 + j * NUM_6) + i);
                in1 = svld1rq_f32(pg, tInPtr + inStride * (NUM_1 + j * NUM_6) + i);
                in2 = svld1rq_f32(pg, tInPtr + inStride * (NUM_2 + j * NUM_6) + i);
                in3 = svld1rq_f32(pg, tInPtr + inStride * (NUM_3 + j * NUM_6) + i);
                in4 = svld1rq_f32(pg, tInPtr + inStride * (NUM_4 + j * NUM_6) + i);
                in5 = svld1rq_f32(pg, tInPtr + inStride * (NUM_5 + j * NUM_6) + i);
                for (occ = 0; occ < oc && ocLoop > 0; occ += realVl) {
                    out0 = svld1_f32(pg, outPtr + (0 + j * NUM_6) * outStride + occ);
                    out1 = svld1_f32(pg, outPtr + (NUM_1 + j * NUM_6) * outStride + occ);
                    out2 = svld1_f32(pg, outPtr + (NUM_2 + j * NUM_6) * outStride + occ);
                    out3 = svld1_f32(pg, outPtr + (NUM_3 + j * NUM_6) * outStride + occ);
                    out4 = svld1_f32(pg, outPtr + (NUM_4 + j * NUM_6) * outStride + occ);
                    out5 = svld1_f32(pg, outPtr + (NUM_5 + j * NUM_6) * outStride + occ);

                    wei0 = svld1_f32(pg, tWeiPtr + weiMatStride * (0 + j * NUM_6) + i * outStride);
                    wei1 = svld1_f32(pg, tWeiPtr + weiMatStride * (NUM_1 + j * NUM_6) + i * outStride);
                    wei2 = svld1_f32(pg, tWeiPtr + weiMatStride * (NUM_2 + j * NUM_6) + i * outStride);
                    wei3 = svld1_f32(pg, tWeiPtr + weiMatStride * (NUM_3 + j * NUM_6) + i * outStride);
                    wei4 = svld1_f32(pg, tWeiPtr + weiMatStride * (NUM_4 + j * NUM_6) + i * outStride);
                    wei5 = svld1_f32(pg, tWeiPtr + weiMatStride * (NUM_5 + j * NUM_6) + i * outStride);

                    out0 = svmla_lane(out0, wei0, in0, 0); // out0 + in0 * wei0
                    out1 = svmla_lane(out1, wei1, in1, 0); // out1 + in1 * wei1
                    out2 = svmla_lane(out2, wei2, in2, 0); // out2 + in2 * wei2
                    out3 = svmla_lane(out3, wei3, in3, 0); // out3 + in3 * wei3
                    out4 = svmla_lane(out4, wei4, in4, 0); // out4 + in4 * wei4
                    out5 = svmla_lane(out5, wei5, in5, 0); // out5 + in5 * wei5

                    wei0 = svld1_f32(pg, tWeiPtr + weiMatStride * (0 + j * NUM_6) +
                        (i + NUM_1) * outStride);
                    wei1 = svld1_f32(pg, tWeiPtr + weiMatStride * (NUM_1 + j * NUM_6) +
                        (i + NUM_1) * outStride);
                    wei2 = svld1_f32(pg, tWeiPtr + weiMatStride * (NUM_2 + j * NUM_6) +
                        (i + NUM_1) * outStride);
                    wei3 = svld1_f32(pg, tWeiPtr + weiMatStride * (NUM_3 + j * NUM_6) +
                        (i + NUM_1) * outStride);
                    wei4 = svld1_f32(pg, tWeiPtr + weiMatStride * (NUM_4 + j * NUM_6) +
                        (i + NUM_1) * outStride);
                    wei5 = svld1_f32(pg, tWeiPtr + weiMatStride * (NUM_5 + j * NUM_6) +
                        (i + NUM_1) * outStride);
                    //
                    out0 = svmla_lane(out0, wei0, in0, NUM_1);
                    out1 = svmla_lane(out1, wei1, in1, NUM_1);
                    out2 = svmla_lane(out2, wei2, in2, NUM_1);
                    out3 = svmla_lane(out3, wei3, in3, NUM_1);
                    out4 = svmla_lane(out4, wei4, in4, NUM_1);
                    out5 = svmla_lane(out5, wei5, in5, NUM_1);

                    wei0 = svld1_f32(pg, tWeiPtr + weiMatStride * (0 + j * NUM_6) +
                        (i + NUM_2) * outStride);
                    wei1 = svld1_f32(pg, tWeiPtr + weiMatStride * (NUM_1 + j * NUM_6) +
                        (i + NUM_2) * outStride);
                    wei2 = svld1_f32(pg, tWeiPtr + weiMatStride * (NUM_2 + j * NUM_6) +
                        (i + NUM_2) * outStride);
                    wei3 = svld1_f32(pg, tWeiPtr + weiMatStride * (NUM_3 + j * NUM_6) +
                        (i + NUM_2) * outStride);
                    wei4 = svld1_f32(pg, tWeiPtr + weiMatStride * (NUM_4 + j * NUM_6) +
                        (i + NUM_2) * outStride);
                    wei5 = svld1_f32(pg, tWeiPtr + weiMatStride * (NUM_5 + j * NUM_6) +
                        (i + NUM_2) * outStride);
                    //
                    out0 = svmla_lane(out0, wei0, in0, NUM_2);
                    out1 = svmla_lane(out1, wei1, in1, NUM_2);
                    out2 = svmla_lane(out2, wei2, in2, NUM_2);
                    out3 = svmla_lane(out3, wei3, in3, NUM_2);
                    out4 = svmla_lane(out4, wei4, in4, NUM_2);
                    out5 = svmla_lane(out5, wei5, in5, NUM_2);

                    wei0 = svld1_f32(pg, tWeiPtr + weiMatStride * (0 + j * NUM_6) +
                        (i + NUM_3) * outStride);
                    wei1 = svld1_f32(pg, tWeiPtr + weiMatStride * (NUM_1 + j * NUM_6) +
                        (i + NUM_3) * outStride);
                    wei2 = svld1_f32(pg, tWeiPtr + weiMatStride * (NUM_2 + j * NUM_6) +
                        (i + NUM_3) * outStride);
                    wei3 = svld1_f32(pg, tWeiPtr + weiMatStride * (NUM_3 + j * NUM_6) +
                        (i + NUM_3) * outStride);
                    wei4 = svld1_f32(pg, tWeiPtr + weiMatStride * (NUM_4 + j * NUM_6) +
                        (i + NUM_3) * outStride);
                    wei5 = svld1_f32(pg, tWeiPtr + weiMatStride * (NUM_5 + j * NUM_6) +
                        (i + NUM_3) * outStride);
                    //
                    out0 = svmla_lane(out0, wei0, in0, NUM_3);
                    out1 = svmla_lane(out1, wei1, in1, NUM_3);
                    out2 = svmla_lane(out2, wei2, in2, NUM_3);
                    out3 = svmla_lane(out3, wei3, in3, NUM_3);
                    out4 = svmla_lane(out4, wei4, in4, NUM_3);
                    out5 = svmla_lane(out5, wei5, in5, NUM_3);

                    tWeiPtr += realVl;
                    svst1_f32(pg, outPtr + (0 + j * NUM_6) * outStride + occ, out0);
                    svst1_f32(pg, outPtr + (NUM_1 + j * NUM_6) * outStride + occ, out1);
                    svst1_f32(pg, outPtr + (NUM_2 + j * NUM_6) * outStride + occ, out2);
                    svst1_f32(pg, outPtr + (NUM_3 + j * NUM_6) * outStride + occ, out3);
                    svst1_f32(pg, outPtr + (NUM_4 + j * NUM_6) * outStride + occ, out4);
                    svst1_f32(pg, outPtr + (NUM_5 + j * NUM_6) * outStride + occ, out5);
                }
            }
            if (icTail > 0) {
                for (occ = 0; occ < oc && ocLoop > 0; occ += realVl) {
                    out0 = svld1_f32(pg, outPtr + (0 + j * NUM_6) * outStride + occ);
                    out1 = svld1_f32(pg, outPtr + (NUM_1 + j * NUM_6) * outStride + occ);
                    out2 = svld1_f32(pg, outPtr + (NUM_2 + j * NUM_6) * outStride + occ);
                    out3 = svld1_f32(pg, outPtr + (NUM_3 + j * NUM_6) * outStride + occ);
                    out4 = svld1_f32(pg, outPtr + (NUM_4 + j * NUM_6) * outStride + occ);
                    out5 = svld1_f32(pg, outPtr + (NUM_5 + j * NUM_6) * outStride + occ);
                    for (; i < ic; i++) {
                        wei0 = svld1_f32(pg, tWeiPtr + weiMatStride * (0 + j * NUM_6) +
                            i * outStride);
                        wei1 = svld1_f32(pg, tWeiPtr + weiMatStride * (NUM_1 + j * NUM_6) +
                            i * outStride);
                        wei2 = svld1_f32(pg, tWeiPtr + weiMatStride * (NUM_2 + j * NUM_6) +
                            i * outStride);
                        wei3 = svld1_f32(pg, tWeiPtr + weiMatStride * (NUM_3 + j * NUM_6) +
                            i * outStride);
                        wei4 = svld1_f32(pg, tWeiPtr + weiMatStride * (NUM_4 + j * NUM_6) +
                            i * outStride);
                        wei5 = svld1_f32(pg, tWeiPtr + weiMatStride * (NUM_5 + j * NUM_6) +
                            i * outStride);

                        out0 = svmla_m(pg, out0, wei0, *(tInPtr + inStride * (0 + j * NUM_6) + i));
                        out1 = svmla_m(pg, out1, wei1, *(tInPtr + inStride * (NUM_1 + j * NUM_6) + i));
                        out2 = svmla_m(pg, out2, wei2, *(tInPtr + inStride * (NUM_2 + j * NUM_6) + i));
                        out3 = svmla_m(pg, out3, wei3, *(tInPtr + inStride * (NUM_3 + j * NUM_6) + i));
                        out4 = svmla_m(pg, out4, wei4, *(tInPtr + inStride * (NUM_4 + j * NUM_6) + i));
                        out5 = svmla_m(pg, out5, wei5, *(tInPtr + inStride * (NUM_5 + j * NUM_6) + i));
                    }
                    tWeiPtr += realVl;
                    svst1_f32(pg, outPtr + (0 + j * NUM_6) * outStride + occ, out0);
                    svst1_f32(pg, outPtr + (NUM_1 + j * NUM_6) * outStride + occ, out1);
                    svst1_f32(pg, outPtr + (NUM_2 + j * NUM_6) * outStride + occ, out2);
                    svst1_f32(pg, outPtr + (NUM_3 + j * NUM_6) * outStride + occ, out3);
                    svst1_f32(pg, outPtr + (NUM_4 + j * NUM_6) * outStride + occ, out4);
                    svst1_f32(pg, outPtr + (NUM_5 + j * NUM_6) * outStride + occ, out5);
                }
            }
        }
    } else {
        constexpr IntType innerTileRows = NUM_6;
        constexpr IntType innerTileCols = NUM_6;
        constexpr IntType matStride = innerTileRows * innerTileCols;
        for (occ = 0; occ < oc; occ++) {
            for (i = 0; i < matStride; i++) {
                const float* tInPtr = inPtr;
                const float* tWeiPtr = weiPtr;
                for (IntType j = 0; j < ic; j++) {
                    // input is nxc, wei layout is hwio
                    outPtr[i * outStride + occ] += tWeiPtr[j * outStride + i * weiMatStride + occ] *
                        tInPtr[i * inStride + j];
                }
            }
        }
    }
}

void OutputTransFormKer16x9::DoTransForm(const IntType& channel, const float* input, float* output,
    const IntType& rowStride, const IntType& colStride, IntType matrixStride, const float* bias)
{
    ArmVCFloat16x9(channel, input, matrixStride, bias, output, rowStride, colStride);
}

void OutputTransFormKer16x9::ArmVCFloat16x9(const IntType& outChannel, const float* input,
    const IntType& matrixStride, const float* bias, float* output, const IntType& outRowStride,
    const IntType& outColStride)
{
    /* AT =
    âŽ?  1  1   1  1   0âŽ?    âŽ?                 âŽ?    âŽ?  1  -1  2  -2  0âŽ?    âŽ?                 âŽ?    âŽ?  1  1   4  4   0âŽ?    âŽ?                 âŽ?    âŽ?  1  -1  8  -8  1âŽ?    */
    IntType channelsRemaining = outChannel;
    for (; channelsRemaining >= NUM_4; channelsRemaining -= NUM_4) {
        float32x4_t F[NUM_6][NUM_6];
        float32x4_t FZ[NUM_6][NUM_4];
        float32x4_t f[NUM_4][NUM_4];
        float32x4_t b;
        // load input data
        for (auto i = 0, m = 0; i < NUM_6; i++) {
            for (auto j = 0; j < NUM_6; j++, m++) {
                F[i][j] = vld1q_f32(input + m * matrixStride);
            }
        }
        input += NUM_4;
        for (auto i = 0u; i < NUM_6; i++) {
            FZ[i][0] = vaddq_f32(vaddq_f32(vaddq_f32(F[i][0], F[i][NUM_1]),
                vaddq_f32(F[i][NUM_2], F[i][NUM_3])), F[i][NUM_4]);
            FZ[i][NUM_1] = vmlaq_n_f32(vsubq_f32(F[i][NUM_1], F[i][NUM_2]),
                vsubq_f32(F[i][NUM_3], F[i][NUM_4]), F_NUM_2);
            FZ[i][NUM_2] = vmlaq_n_f32(vaddq_f32(F[i][NUM_1], F[i][NUM_2]),
                vaddq_f32(F[i][NUM_3], F[i][NUM_4]), F_NUM_4);
            FZ[i][NUM_3] = vaddq_f32(vmlaq_n_f32(vsubq_f32(F[i][NUM_1], F[i][NUM_2]),
                vsubq_f32(F[i][NUM_3], F[i][NUM_4]), F_NUM_8), F[i][NUM_5]);
        }
        // Compute the output tile f = ZT F Z
        for (auto j = 0u; j < NUM_4; j++) {
            f[0][j] = vaddq_f32(vaddq_f32(vaddq_f32(FZ[0][j], FZ[NUM_1][j]),
                vaddq_f32(FZ[NUM_2][j], FZ[NUM_3][j])), FZ[NUM_4][j]);
            f[NUM_1][j] = vmlaq_n_f32(vsubq_f32(FZ[NUM_1][j], FZ[NUM_2][j]),
                vsubq_f32(FZ[NUM_3][j], FZ[NUM_4][j]), F_NUM_2);
            f[NUM_2][j] = vmlaq_n_f32(vaddq_f32(FZ[NUM_1][j], FZ[NUM_2][j]),
                vaddq_f32(FZ[NUM_3][j], FZ[NUM_4][j]), F_NUM_4);
            f[NUM_3][j] = vaddq_f32(vmlaq_n_f32(vsubq_f32(FZ[NUM_1][j], FZ[NUM_2][j]),
                vsubq_f32(FZ[NUM_3][j], FZ[NUM_4][j]), F_NUM_8), FZ[NUM_5][j]);
        }
        if (bias != nullptr) {
            b = vld1q_f32(bias);
            bias += NUM_4;
        } else {
            b = vdupq_n_f32(0.0f);
        }
        for (auto i = 0; i < WINO_OUT_MIN_MATRIX_WEI; i++) {
            for (auto j = 0; j < WINO_OUT_MIN_MATRIX_HEI; j++) {
                const auto y = vaddq_f32(f[i][j], b);
                vst1q_f32(output + i * outRowStride + j * outColStride, y);
            }
        }
        output += NUM_4;
    }
    // oc<=2
    for (; channelsRemaining >= NUM_2; channelsRemaining -= NUM_2) {
        float32x2_t F[NUM_6][NUM_6];
        float32x2_t FZ[NUM_6][NUM_4];
        float32x2_t f[NUM_4][NUM_4];
        float32x2_t b;
        // load input data
        for (auto i = 0, m = 0; i < NUM_6; i++) {
            for (auto j = 0; j < NUM_6; j++, m++) {
                F[i][j] = vld1_f32(input + m * matrixStride);
            }
        }
        input += NUM_2;
        // Compute the matrix F Z
        for (auto i = 0u; i < NUM_6; i++) {
            FZ[i][0] = vadd_f32(vadd_f32(vadd_f32(F[i][0], F[i][NUM_1]),
                vadd_f32(F[i][NUM_2], F[i][NUM_3])), F[i][NUM_4]);
            FZ[i][NUM_1] = vmla_n_f32(vsub_f32(F[i][NUM_1], F[i][NUM_2]),
                vsub_f32(F[i][NUM_3], F[i][NUM_4]), F_NUM_2);
            FZ[i][NUM_2] = vmla_n_f32(vadd_f32(F[i][NUM_1], F[i][NUM_2]),
                vadd_f32(F[i][NUM_3], F[i][NUM_4]), F_NUM_4);
            FZ[i][NUM_3] = vadd_f32(vmla_n_f32(vsub_f32(F[i][NUM_1], F[i][NUM_2]),
                vsub_f32(F[i][NUM_3], F[i][NUM_4]), F_NUM_8), F[i][NUM_5]);
        }
        // Compute the output tile f = ZT F Z
        for (auto j = 0u; j < NUM_4; j++) {
            f[0][j] = vadd_f32(vadd_f32(vadd_f32(FZ[0][j], FZ[NUM_1][j]),
                vadd_f32(FZ[NUM_2][j], FZ[NUM_3][j])), FZ[NUM_4][j]);
            f[NUM_1][j] = vmla_n_f32(vsub_f32(FZ[NUM_1][j], FZ[NUM_2][j]),
                vsub_f32(FZ[NUM_3][j], FZ[NUM_4][j]), F_NUM_2);
            f[NUM_2][j] = vmla_n_f32(vadd_f32(FZ[NUM_1][j], FZ[NUM_2][j]),
                vadd_f32(FZ[NUM_3][j], FZ[NUM_4][j]), F_NUM_4);
            f[NUM_3][j] = vadd_f32(vmla_n_f32(vsub_f32(FZ[NUM_1][j], FZ[NUM_2][j]),
                vsub_f32(FZ[NUM_3][j], FZ[NUM_4][j]), F_NUM_8), FZ[NUM_5][j]);
        }
        if (bias != nullptr) {
            b = vld1_f32(bias);
            bias += NUM_2;
        } else {
            b = vdup_n_f32(0.0f);
        }
        for (auto i = 0; i < WINO_OUT_MIN_MATRIX_WEI; i++) {
            for (auto j = 0; j < WINO_OUT_MIN_MATRIX_HEI; j++) {
                const auto y = vadd_f32(f[i][j], b);
                vst1_f32(output + i * outRowStride + j * outColStride, y);
            }
        }
        output += NUM_2;
    }
    // oc <= 1
    for (; channelsRemaining; channelsRemaining--) {
        float F[NUM_6][NUM_6];
        float FZ[NUM_6][NUM_4];
        float f[NUM_4][NUM_4];
        float b;
        // Read a 6x6 tile in the Winograd domain
        for (auto i = 0, m = 0; i < NUM_6; i++) {
            for (auto j = 0; j < NUM_6; j++, m++) {
                F[i][j] = *(input + m * matrixStride);
            }
        }
        input++;
        // Compute the matrix F Z
        for (auto i = 0u; i < NUM_6; i++) {
            FZ[i][0] = NUM_1 * F[i][0] + NUM_1 * F[i][NUM_1] + NUM_1 *
                F[i][NUM_2] + NUM_1 * F[i][NUM_3] + NUM_1 * F[i][NUM_4];
            FZ[i][NUM_1] = NUM_1 * F[i][NUM_1] + -NUM_1 * F[i][NUM_2] +
                NUM_2 * F[i][NUM_3] + -NUM_2 * F[i][NUM_4];
            FZ[i][NUM_2] = NUM_1 * F[i][NUM_1] + NUM_1 * F[i][NUM_2] +
                NUM_4 * F[i][NUM_3] +  NUM_4 * F[i][NUM_4];
            FZ[i][NUM_3] = NUM_1 * F[i][NUM_1] + -NUM_1 * F[i][NUM_2] +
                NUM_8 * F[i][NUM_3] + -NUM_8 * F[i][NUM_4] +  NUM_1 * F[i][NUM_5];
        }
        // Compute the output tile f = ZT F Z
        for (auto j = 0u; j < NUM_4; j++) {
            f[0][j] = NUM_1 * FZ[0][j] + NUM_1 * FZ[NUM_1][j] + NUM_1 *
                FZ[NUM_2][j] + NUM_1 * FZ[NUM_3][j] + NUM_1 * FZ[NUM_4][j];
            f[NUM_1][j] =  NUM_1 * FZ[NUM_1][j] + -NUM_1 * FZ[NUM_2][j] +
                NUM_2 * FZ[NUM_3][j] + -NUM_2 * FZ[NUM_4][j];
            f[NUM_2][j] = NUM_1 * FZ[NUM_1][j] + NUM_1 * FZ[NUM_2][j] +
                NUM_4 * FZ[NUM_3][j] + NUM_4 * FZ[NUM_4][j];
            f[NUM_3][j] = NUM_1 * FZ[NUM_1][j] + -NUM_1 * FZ[NUM_2][j] +
                NUM_8 * FZ[NUM_3][j] + -NUM_8 * FZ[NUM_4][j] + NUM_1 * FZ[NUM_5][j];
        }
        // Write out the output tile
        if (bias != nullptr) {
            b = *(bias++);
        } else {
            b = 0.0f;
        }
        for (auto i = 0u; i < WINO_OUT_MIN_MATRIX_WEI; i++) {
            for (auto j = 0u; j < WINO_OUT_MIN_MATRIX_HEI; j++) {
                const auto y = f[i][j] + b;
                *(output + i * outRowStride + j * outColStride) = y;
            }
        }
        output++;
    }
}

void InputTransFormKer6x6::DoTransForm(const IntType& channel, const float* input, float* output,
    const IntType& rowStride, const IntType& colStride, IntType matrixStride,
    const float* bias)
{
    Service::Unused(bias);
    ArmVCFloat6x6(channel, input, rowStride, colStride, output, matrixStride);
}

void InputTransFormKer6x6::ArmVCFloat6x6(const IntType& inChannel, const float* input,
    const IntType& inRowStride, const IntType& inColStride, float* output, const IntType matStride)
{
    constexpr IntType innerTileRows = NUM_6;
    constexpr IntType innerTileCols = NUM_6;
    // Get pointers into the input tile
    const float *x_ptrs[innerTileRows][innerTileCols];
    for (IntType i = 0, xi = 0; i < innerTileRows; i++, xi++) {
        // Get a pointer into the row
        const float* const row_ptr = input + xi * inRowStride;
        for (IntType j = 0, xj = 0; j < innerTileCols; j++, xj++) {
            x_ptrs[i][j] = row_ptr + xj * inColStride;
        }
    }
    // Perform the Winograd input transformation for each channel in the input
    // tensor.
    IntType channelsRemaining = inChannel;
    for (; channelsRemaining >= NUM_4; channelsRemaining -= NUM_4) {
        // Matrices used/computed in this kernel
        float32x4_t x[innerTileRows][innerTileCols];
        float32x4_t XTx[innerTileRows][innerTileCols];
        float32x4_t U[innerTileRows][innerTileCols];
        for (IntType i = 0; i < innerTileRows; i++) {
            for (IntType j = 0; j < innerTileCols; j++) {
                x[i][j] = vdupq_n_f32(0.0f);
                XTx[i][j] = vdupq_n_f32(0.0f);
            }
        }
        // Read a 6x6 tile in the Winograd domain
        for (IntType i = 0; i < innerTileRows; i++) {
            for (IntType j = 0; j < innerTileCols; j++) {
                x[i][j] = vld1q_f32(x_ptrs[i][j]);
                x_ptrs[i][j] += NUM_4;
            }
        }
        /* BT =
        âŽ?  0   -5  0   1  0âŽ?        âŽ?                  âŽ?        âŽ?  -4  -4  1   1  0âŽ?        âŽ?                  âŽ?        âŽ?  4   -4  -1  1  0âŽ?        âŽ?                  âŽ?        âŽ?  -2  -1  2   1  0âŽ?        âŽ?                  âŽ?        âŽ?  2   -1  -2  1  0âŽ?        âŽ?                  âŽ?        âŽ?  4   0   -5  0  1âŽ?*/
        // Compute XT . x
        for (IntType j = 0; j < innerTileCols; j++) {
            XTx[0][j] = vmlsq_n_f32(vmlaq_n_f32(x[NUM_4][j], x[0][j], F_NUM_4), x[NUM_2][j], F_NUM_5);
            XTx[NUM_1][j] = vmlsq_n_f32(vaddq_f32(x[NUM_3][j], x[NUM_4][j]),
                vaddq_f32(x[NUM_1][j], x[NUM_2][j]), F_NUM_4);
            XTx[NUM_2][j] = vmlaq_n_f32(vsubq_f32(x[NUM_4][j], x[NUM_3][j]),
                vsubq_f32(x[NUM_1][j], x[NUM_2][j]), F_NUM_4);
            XTx[NUM_3][j] = vmlaq_n_f32(vsubq_f32(x[NUM_4][j], x[NUM_2][j]),
                vsubq_f32(x[NUM_3][j], x[NUM_1][j]), F_NUM_2);
            XTx[NUM_4][j] = vmlaq_n_f32(vsubq_f32(x[NUM_4][j], x[NUM_2][j]),
                vsubq_f32(x[NUM_1][j], x[NUM_3][j]), F_NUM_2);
            XTx[NUM_5][j] = vmlsq_n_f32(vmlaq_n_f32(x[NUM_5][j], x[NUM_1][j],
                F_NUM_4), x[NUM_3][j], F_NUM_5);
        }
        // Compute U = XT . x . X
        for (IntType i = 0; i < innerTileRows; i++) {
            U[i][0] = vmlsq_n_f32(vmlaq_n_f32(XTx[i][NUM_4], XTx[i][0], F_NUM_4), XTx[i][NUM_2], F_NUM_5);
            U[i][NUM_1] = vmlsq_n_f32(vaddq_f32(XTx[i][NUM_3], XTx[i][NUM_4]),
                vaddq_f32(XTx[i][NUM_1], XTx[i][NUM_2]), F_NUM_4);
            U[i][NUM_2] = vmlaq_n_f32(vsubq_f32(XTx[i][NUM_4], XTx[i][NUM_3]),
                vsubq_f32(XTx[i][NUM_1], XTx[i][NUM_2]), F_NUM_4);
            U[i][NUM_3] = vmlaq_n_f32(vsubq_f32(XTx[i][NUM_4], XTx[i][NUM_2]),
                vsubq_f32(XTx[i][NUM_3], XTx[i][NUM_1]), F_NUM_2);
            U[i][NUM_4] = vmlaq_n_f32(vsubq_f32(XTx[i][NUM_4], XTx[i][NUM_2]),
                vsubq_f32(XTx[i][NUM_1], XTx[i][NUM_3]), F_NUM_2);
            U[i][NUM_5] = vmlsq_n_f32(vmlaq_n_f32(XTx[i][NUM_5], XTx[i][NUM_1],
                F_NUM_4), XTx[i][NUM_3], F_NUM_5);
        }
        // Store the transformed matrix
        for (IntType i = 0, m = 0; i < innerTileRows; i++) {
            for (IntType j = 0; j < innerTileCols; j++, m++) {
                vst1q_f32(output + m * matStride, U[i][j]);
            }
        }
        output += NUM_4;
    }
    for (; channelsRemaining >= NUM_2; channelsRemaining -= NUM_2) {
        // Matrices used/computed in this kernel
        float32x2_t x[innerTileRows][innerTileCols];
        float32x2_t XTx[innerTileRows][innerTileCols];
        float32x2_t U[innerTileRows][innerTileCols];
        for (IntType i = 0; i < innerTileRows; i++) {
            for (IntType j = 0; j < innerTileCols; j++) {
                x[i][j] = vdup_n_f32(0.0f);
                XTx[i][j] = vdup_n_f32(0.0f);
            }
        }
        // Read a 6x6 tile in the Winograd domain
        for (IntType i = 0; i < innerTileRows; i++) {
            for (IntType j = 0; j < innerTileCols; j++) {
                x[i][j] = vld1_f32(x_ptrs[i][j]);
                x_ptrs[i][j] += NUM_2;
            }
        }
        // Compute XT . x
        for (IntType j = 0; j < innerTileCols; j++) {
            XTx[0][j] = vmls_n_f32(vmla_n_f32(x[NUM_4][j], x[0][j], F_NUM_4), x[NUM_2][j], F_NUM_5);
            XTx[NUM_1][j] = vmls_n_f32(vadd_f32(x[NUM_3][j], x[NUM_4][j]),
                vadd_f32(x[NUM_1][j], x[NUM_2][j]), F_NUM_4);
            XTx[NUM_2][j] = vmla_n_f32(vsub_f32(x[NUM_4][j], x[NUM_3][j]),
                vsub_f32(x[NUM_1][j], x[NUM_2][j]), F_NUM_4);
            XTx[NUM_3][j] = vmla_n_f32(vsub_f32(x[NUM_4][j], x[NUM_2][j]),
                vsub_f32(x[NUM_3][j], x[NUM_1][j]), F_NUM_2);
            XTx[NUM_4][j] = vmla_n_f32(vsub_f32(x[NUM_4][j], x[NUM_2][j]),
                vsub_f32(x[NUM_1][j], x[NUM_3][j]), F_NUM_2);
            XTx[NUM_5][j] = vmls_n_f32(vmla_n_f32(x[NUM_5][j], x[NUM_1][j],
                F_NUM_4), x[NUM_3][j], F_NUM_5);
        }
        // Compute U = XT . x . X
        for (IntType i = 0; i < innerTileRows; i++) {
            U[i][0] = vmls_n_f32(vmla_n_f32(XTx[i][NUM_4], XTx[i][0], F_NUM_4), XTx[i][NUM_2], F_NUM_5);
            U[i][NUM_1] = vmls_n_f32(vadd_f32(XTx[i][NUM_3], XTx[i][NUM_4]),
                vadd_f32(XTx[i][NUM_1], XTx[i][NUM_2]), F_NUM_4);
            U[i][NUM_2] = vmla_n_f32(vsub_f32(XTx[i][NUM_4], XTx[i][NUM_3]),
                vsub_f32(XTx[i][NUM_1], XTx[i][NUM_2]), F_NUM_4);
            U[i][NUM_3] = vmla_n_f32(vsub_f32(XTx[i][NUM_4], XTx[i][NUM_2]),
                vsub_f32(XTx[i][NUM_3], XTx[i][NUM_1]), F_NUM_2);
            U[i][NUM_4] = vmla_n_f32(vsub_f32(XTx[i][NUM_4], XTx[i][NUM_2]),
                vsub_f32(XTx[i][NUM_1], XTx[i][NUM_3]), F_NUM_2);
            U[i][NUM_5] = vmls_n_f32(vmla_n_f32(XTx[i][NUM_5], XTx[i][NUM_1],
                F_NUM_4), XTx[i][NUM_3], F_NUM_5);
        }
        // Store the transformed matrix
        for (IntType i = 0, m = 0; i < innerTileRows; i++) {
            for (IntType j = 0; j < innerTileCols; j++, m++) {
                vst1_f32(output + m * matStride, U[i][j]);
            }
        }
        output += NUM_2;
    }
     // Matrices used/computed in this kernel.
    float x[innerTileRows][innerTileCols];
    float XTx[innerTileRows][innerTileCols];
    float U[innerTileRows][innerTileCols];
    for (IntType i = 0; i < innerTileRows; i++) {
        for (IntType j = 0; j < innerTileCols; j++) {
            x[i][j] = XTx[i][j] = 0.0f;
        }
    }
    for (; channelsRemaining; channelsRemaining--) {
        // Load src
        for (IntType i = 0; i < innerTileRows; i++) {
            for (IntType j = 0; j < innerTileCols; j++) {
                x[i][j] = *(x_ptrs[i][j]++);
            }
        }
        // Compute XT . x
        for (IntType j = 0; j < innerTileCols; j++) {
            XTx[0][j] =  NUM_4 * x[0][j] + -NUM_5 * x[NUM_2][j] +  NUM_1 * x[NUM_4][j];
            XTx[NUM_1][j] = -NUM_4 * x[NUM_1][j] + -NUM_4 * x[NUM_2][j] +
                NUM_1 * x[NUM_3][j] +  NUM_1 * x[NUM_4][j];
            XTx[NUM_2][j] =  NUM_4 * x[NUM_1][j] + -NUM_4 * x[NUM_2][j] +
                -NUM_1 * x[NUM_3][j] +  NUM_1 * x[NUM_4][j];
            XTx[NUM_3][j] = -NUM_2 * x[NUM_1][j] + -NUM_1 * x[NUM_2][j] +
                NUM_2 * x[NUM_3][j] +  NUM_1 * x[NUM_4][j];
            XTx[NUM_4][j] =  NUM_2 * x[NUM_1][j] + -NUM_1 * x[NUM_2][j] +
                -NUM_2 * x[NUM_3][j] +  NUM_1 * x[NUM_4][j];
            XTx[NUM_5][j] =  NUM_4 * x[NUM_1][j] + -NUM_5 * x[NUM_3][j] +
                NUM_1 * x[NUM_5][j];
        }
        // Compute U = XT . x . X
        for (IntType i = 0; i < innerTileRows; i++) {
            U[i][0] =  NUM_4 * XTx[i][0] + -NUM_5 * XTx[i][NUM_2] +  NUM_1 * XTx[i][NUM_4];
            U[i][NUM_1] = -NUM_4 * XTx[i][NUM_1] + -NUM_4 * XTx[i][NUM_2] +
                NUM_1 * XTx[i][NUM_3] +  NUM_1 * XTx[i][NUM_4];
            U[i][NUM_2] =  NUM_4 * XTx[i][NUM_1] + -NUM_4 * XTx[i][NUM_2] +
                -NUM_1 * XTx[i][NUM_3] +  NUM_1 * XTx[i][NUM_4];
            U[i][NUM_3] = -NUM_2 * XTx[i][NUM_1] + -NUM_1 * XTx[i][NUM_2] +
                NUM_2 * XTx[i][NUM_3] +  NUM_1 * XTx[i][NUM_4];
            U[i][NUM_4] =  NUM_2 * XTx[i][NUM_1] + -NUM_1 * XTx[i][NUM_2] +
                -NUM_2 * XTx[i][NUM_3] +  NUM_1 * XTx[i][NUM_4];
            U[i][NUM_5] =  NUM_4 * XTx[i][NUM_1] + -NUM_5 * XTx[i][NUM_3] +
                NUM_1 * XTx[i][NUM_5];
        }
        // Store the transformed matrix
        for (IntType i = 0, m = 0; i < innerTileRows; i++) {
            for (IntType j = 0; j < innerTileCols; j++, m++) {
              *(output + m * matStride) = U[i][j];
            }
        }
        output++;
    }
}

void WeightTransFormKer16x9::DoTransForm(const IntType& channel, const float* input, float* output,
    const IntType& rowStride, const IntType& colStride, IntType matrixStride, const float* bias)
{
    Service::Unused(bias);
    ArmVCFloat16x9(channel, input, rowStride, colStride, output, matrixStride);
}

void WeightTransFormKer16x9::ArmVCFloat16x9(const IntType& outChannel, const float* input,
    const IntType& ldWeiRow, const IntType& ldWeiCol, float* output, const IntType& matrixStride)
{
    IntType channelsRemaining = outChannel;
    for (; channelsRemaining >= NUM_4; channelsRemaining -= NUM_4) {
        float32x4_t w[NUM_3][NUM_3];
        float32x4_t wW[NUM_6][NUM_3];
        float32x4_t V[NUM_6][NUM_6];
        for (IntType i = 0; i < NUM_3; i++) {
            for (IntType j = 0; j < NUM_3; j++) {
                w[i][j] = vld1q_f32(input + i * ldWeiRow + j * ldWeiCol);
            }
        }
        /* G =
        âŽ?/4     0     0  âŽ?  âŽ?   0  0âŽ?        âŽ?                âŽ?  âŽ?       âŽ?        âŽ?1/6  -1/6   -1/6âŽ?  âŽ?4 -4 -4âŽ?        âŽ?                âŽ?  âŽ?       âŽ?        âŽ?1/6   1/6   -1/6âŽ?= âŽ?4  4 -4âŽ?        âŽ?                âŽ?  âŽ?       âŽ?/ 24
        âŽ?/24  1/12    1/6âŽ?  âŽ?   2  4âŽ?        âŽ?                âŽ?  âŽ?       âŽ?        âŽ?/24  -1/12   1/6âŽ?  âŽ?  -2  4âŽ?        âŽ?                âŽ?  âŽ?       âŽ?        âŽ?0      0     1  âŽ?  âŽ?  0  24âŽ?*/
        // compute the matrix, wW=G w
        for (IntType j = 0; j < NUM_3; j++) {
            wW[0][j] = vmulq_n_f32(w[0][j], F_NUM_6);
            wW[1][j] = vmulq_n_f32(vaddq_f32(vaddq_f32(w[0][j], w[NUM_1][j]), w[NUM_2][j]), -F_NUM_4);
            wW[NUM_2][j] = vmulq_n_f32(vsubq_f32(vsubq_f32(w[NUM_1][j], w[0][j]), w[NUM_2][j]), F_NUM_4);
            wW[NUM_3][j] = vmlaq_n_f32(vmlaq_n_f32(w[0][j], w[NUM_1][j], F_NUM_2), w[NUM_2][j], F_NUM_4);
            wW[NUM_4][j] = vmlaq_n_f32(vmlsq_n_f32(w[0][j], w[NUM_1][j], F_NUM_2), w[NUM_2][j], F_NUM_4);
            wW[NUM_5][j] = vmulq_n_f32(w[NUM_2][j], F_NUM_24);
        }
        // Compute V=G w GT
        for (IntType i = 0; i < NUM_6; i++) {
            const float recip576 = F_NUM_1 / F_NUM_576;
            V[i][0] = vmulq_n_f32(vmulq_n_f32(wW[i][0], F_NUM_6), recip576);
            V[i][NUM_1] = vmulq_n_f32(vmulq_n_f32(vaddq_f32(vaddq_f32(wW[i][0], wW[i][NUM_1]),
                wW[i][NUM_2]), -F_NUM_4), recip576);
            V[i][NUM_2] = vmulq_n_f32(vmulq_n_f32(vsubq_f32(vsubq_f32(wW[i][NUM_1], wW[i][0]),
                wW[i][NUM_2]), F_NUM_4), recip576);
            V[i][NUM_3] = vmulq_n_f32(vmlaq_n_f32(vmlaq_n_f32(wW[i][0], wW[i][NUM_1], F_NUM_2),
                wW[i][NUM_2], F_NUM_4), recip576);
            V[i][NUM_4] = vmulq_n_f32(vmlaq_n_f32(vmlsq_n_f32(wW[i][0], wW[i][NUM_1], F_NUM_2),
                wW[i][NUM_2], F_NUM_4), recip576);
            V[i][NUM_5] = vmulq_n_f32(vmulq_n_f32(wW[i][NUM_2], F_NUM_24), recip576);
        }
        // store the transformed weights
        for (IntType i = 0, m = 0; i < NUM_6; i++) {
            for (IntType j = 0; j < NUM_6; m++, j++) {
                vst1q_f32(output + m * matrixStride, V[i][j]);
            }
        }
        input += NUM_4;
        output += NUM_4;
    }
    for (; channelsRemaining >= NUM_2; channelsRemaining -= NUM_2) {
        // Matrices used and computed in this kernel
        float32x2_t w[NUM_3][NUM_3];
        float32x2_t wW[NUM_6][NUM_3];
        float32x2_t V[NUM_6][NUM_6];
        // Read weights
        for (IntType i = 0; i < NUM_3; i++) {
            for (IntType j = 0; j < NUM_3; j++) {
                w[i][j] = vld1_f32(input + i * ldWeiRow + j * ldWeiCol);
            }
        }
        // Compute the matrix W w
        for (IntType j = 0; j < NUM_3; j++) {
            wW[0][j] = vmul_n_f32(w[0][j], F_NUM_6);
            wW[NUM_1][j] = vmul_n_f32(vadd_f32(vadd_f32(w[0][j], w[NUM_1][j]), w[NUM_2][j]), -F_NUM_4);
            wW[NUM_2][j] = vmul_n_f32(vsub_f32(vsub_f32(w[NUM_1][j], w[0][j]), w[NUM_2][j]), F_NUM_4);
            wW[NUM_3][j] = vmla_n_f32(vmla_n_f32(w[0][j], w[NUM_1][j], F_NUM_2), w[NUM_2][j], F_NUM_4);
            wW[NUM_4][j] = vmla_n_f32(vmls_n_f32(w[0][j], w[NUM_1][j], F_NUM_2), w[NUM_2][j], F_NUM_4);
            wW[NUM_5][j] = vmul_n_f32(w[NUM_2][j], F_NUM_24);
        }
        // Compute V = W w WT
        for (IntType i = 0; i < NUM_6; i++) {
            const float recip576 = F_NUM_1 / F_NUM_576;
            V[i][0] = vmul_n_f32(vmul_n_f32(wW[i][0], F_NUM_6), recip576);
            V[i][NUM_1] = vmul_n_f32(vmul_n_f32(vadd_f32(vadd_f32(wW[i][0], wW[i][NUM_1]),
                wW[i][NUM_2]), -F_NUM_4), recip576);
            V[i][NUM_2] = vmul_n_f32(vmul_n_f32(vsub_f32(vsub_f32(wW[i][NUM_1], wW[i][0]),
                wW[i][NUM_2]), F_NUM_4), recip576);
            V[i][NUM_3] = vmul_n_f32(vmla_n_f32(vmla_n_f32(wW[i][0], wW[i][NUM_1], F_NUM_2),
                wW[i][NUM_2], F_NUM_4), recip576);
            V[i][NUM_4] = vmul_n_f32(vmla_n_f32(vmls_n_f32(wW[i][0], wW[i][NUM_1], F_NUM_2),
                wW[i][NUM_2], F_NUM_4), recip576);
            V[i][NUM_5] = vmul_n_f32(vmul_n_f32(wW[i][NUM_2], F_NUM_24), recip576);
        }
        // Store the transformed weights
        for (IntType i = 0, m = 0; i < NUM_6; i++) {
            for (IntType j = 0; j < NUM_6; j++, m++) {
                vst1_f32(output + m * matrixStride, V[i][j]);
            }
        }
        input += NUM_2;
        output += NUM_2;
    }
    for (; channelsRemaining; channelsRemaining--) {
        float w[NUM_3][NUM_3];
        float wW[NUM_6][NUM_3];
        float V[NUM_6][NUM_6];
        for (IntType i = 0; i < NUM_3; i++) {
            for (IntType j = 0; j < NUM_3; j++) {
                w[i][j] = *(input + i * ldWeiRow + j * ldWeiCol);
            }
        }
        /* G                 GT
        âŽ?   0  0âŽ?           âŽ? -4 -4  1  1 0 âŽ?        âŽ?       âŽ?           âŽ?               âŽ?        âŽ?4 -4 -4âŽ?           âŽ? -4  4  2 -2 0 âŽ?/ 24
        âŽ?       âŽ?           âŽ?               âŽ?        âŽ?4  4 -4âŽ?           âŽ? -4 -4  4  4 24âŽ?        âŽ?       âŽ?/ 24
        âŽ?   2  4âŽ?        âŽ?       âŽ?        âŽ?  -2  4âŽ?        âŽ?       âŽ?        âŽ?  0  24âŽ?*/
        for (IntType j = 0; j < NUM_3; j++) {
            wW[0][j] =  NUM_6 * w[0][j];
            wW[NUM_1][j] = -NUM_4 * w[0][j] + -NUM_4 * w[NUM_1][j] + -NUM_4 * w[NUM_2][j];
            wW[NUM_2][j] = -NUM_4 * w[0][j] + NUM_4 * w[NUM_1][j] + -NUM_4 * w[NUM_2][j];
            wW[NUM_3][j] = NUM_1 * w[0][j] + NUM_2 * w[NUM_1][j] + NUM_4 * w[NUM_2][j];
            wW[NUM_4][j] = NUM_1 * w[0][j] + -NUM_2 * w[NUM_1][j] + NUM_4 * w[NUM_2][j];
            wW[NUM_5][j] = NUM_24 * w[NUM_2][j];
        }
        for (IntType i = 0; i < NUM_6; i++) {
            V[i][0] = (NUM_6 * wW[i][0]) / F_NUM_576;
            V[i][NUM_1] = (-NUM_4 * wW[i][0] + -NUM_4 * wW[i][NUM_1] + -NUM_4 * wW[i][NUM_2]) / F_NUM_576;
            V[i][NUM_2] = (-NUM_4 * wW[i][0] + NUM_4 * wW[i][NUM_1] + -NUM_4 *wW[i][NUM_2]) / F_NUM_576;
            V[i][NUM_3] = (NUM_1 * wW[i][0] + NUM_2 * wW[i][NUM_1] + NUM_4 * wW[i][NUM_2]) / F_NUM_576;
            V[i][NUM_4] = (NUM_1 * wW[i][0] + -NUM_2 * wW[i][NUM_1] + NUM_4 * wW[i][NUM_2]) / F_NUM_576;
            V[i][NUM_5] = (NUM_24 * wW[i][NUM_2]) / F_NUM_576;
        }
        for (IntType i = 0, m = 0; i < NUM_6; i++) {
            for (IntType j = 0; j < NUM_6; j++, m++) {
                *(output + m * matrixStride) = V[i][j];
            }
        }
        input++;
        output++;
    }
}

void WinoConvolutionLayerFWDImpl::DataDup(float *src, int32_t dupVal, IntType len)
{
#pragma omp parallel for num_threads(nThr) collapse(1)
    for (IntType i = 0; i < len / NUM_8; i++) {
        svbool_t pg = svptrue_b32();
        svfloat32_t row = svdup_n_f32(dupVal);
        svst1_f32(pg, src + i * NUM_8, row);
    }
    IntType tail = len % NUM_8;
    if (tail != 0) {
        svbool_t pgTail = svwhilelt_b32_s64(0, tail);
        svfloat32_t row = svdup_n_f32(dupVal);
        svst1_f32(pgTail, src + len / NUM_8 * NUM_8, row);
    }
}

void WinoConvolutionLayerFWDImpl::DataCopy(float *dst, const float *src, IntType len)
{
    for (IntType i = 0; i < len / NUM_8; i++) {
        svbool_t pg = svptrue_b32();
        svfloat32_t row = svld1_f32(pg, src + i * NUM_8);
        svst1_f32(pg, dst + i * NUM_8, row);
    }
    IntType tail = len % NUM_8;
    if (tail != 0) {
        svbool_t pgTail = svwhilelt_b32_s64(0, tail);
        svfloat32_t row = svld1_f32(pgTail, src + len / NUM_8 * NUM_8);
        svst1_f32(pgTail, dst + len / NUM_8 * NUM_8, row);
    }
}
} // DeTail
} // KuDNN