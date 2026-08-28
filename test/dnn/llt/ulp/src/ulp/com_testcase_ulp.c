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

#include "com_testcase_ulp.h"

static float GetBf16CompareResult(float f)
{
    uint32_t u = Asuint(f);
    uint16_t u1 = (uint16_t)(u >> 16);
    uint16_t res = 0x0;
    if (isnormal(f)) {
        // FP_NORMAL: round to nearest even and truncate
        uint32_t rounding_bias = 0x00007FFF + (u1 & 0x1);
        uint32_t int_raw = u + rounding_bias;
        res = (uint16_t)(int_raw >> 16);
    } else if(isinf(f)) {
        // FP_INFINITE
        res = u1;
    } else if(isnan(f)) {
        // FP_NAN: truncate and set MSB of the mantissa force QNAN
        res = u1;
        res |= 1 << 6;
    } else {
        // FP_SUBNORMAL, FP_ZERO: sign preserving zero (denormal go to zero)
        res = u1;
        res &= 0x8000;
    }

    float y = Asfloat((uint32_t)(res << 16));
    return y;
}

static bool DnnCheckAbsError(struct DnnCompareItem *prb) {
    const float s = prb->src;
    const float trh = prb->trh;
    const float alpha = prb->alpha;
    const float approx_machine_eps = 0x1p-22f;
    const float comp_err = approx_machine_eps / trh;

    switch (prb->alg) {
        case ELU:
        case ELU_DST:
            // catch catastrophic cancellation when (exp(s) - 1), s < 0 and
            // s is close to zero.
            return (prb->dir & FLAG_FWD) && signbit(s) && (fabsf(expf(s) - 1.f) <= comp_err);
        case GELU_TANH: {
            // catch catastrophic cancellation
            // (4.f is magic scale for f32)
            const float sqrt_2_over_pi = 0.797884f;
            const float fitting_const = 0.044715f;
            float v = tanhf(sqrt_2_over_pi * s * (1.f + fitting_const * s * s));
            float dg = sqrt_2_over_pi * (1.f + 3.f * fitting_const * s * s);
            if (fabsf(1.f + v) <= comp_err) return true;
            return (prb->dir & FLAG_BWD) && signbit(s) && fabsf(1.f + s * (1.f - v) * dg) <= 4.f * comp_err;
        }
        case GELU_ERF: {
            // Catch catastrophic cancellation
            // which occurs at large negative s.
            // Factor 2 (in bwd) is to account for the fact that error is
            // accumulated for each summand (except the 1) when they
            // are of the same order of magnitude.
            const float sqrt_2_over_2 = 0.707106769084930419921875f;
            const float two_over_sqrt_pi = 1.12837922573089599609375f;
            float v = s * sqrt_2_over_2;
            if (prb->dir & FLAG_FWD)
                return fabsf(1.f + erff(v)) <= comp_err;
            else
                return fabsf(1.f + erff(v) + v * two_over_sqrt_pi * expf(-v * v)) <= comp_err * 2.f;
        }
        case TANH:
            // catch catastrophic cancellation, which occurs when err in tanh(s)
            // is high and tanh(s) is close to 1.
            return (prb->dir & FLAG_BWD) && (1.f - tanhf(fabsf(s))) <= comp_err;
        case TANH_DST: // sse41 can't do fma
            // catch catastrophic cancellation, which occurs when err in tanh(s)
            // is high and tanh(s) is close to 1.
            return (prb->dir & FLAG_BWD) && (1.f - s * s) <= comp_err;
        case SRELU:
            // when `alpha * s` is negative, expf(alpha * s) -> 0 rapidly
            // which leads to log1pf(expf(alpha * s)) -> 0
            // which leads to high relative error,
            // while abs error is still low.
            // (10.f is magic scale for bf16)
            return (prb->dir & FLAG_FWD) && signbit(alpha * s) && log1pf(expf(alpha * s)) <= 10.f * comp_err;
        case MISH:
            // same situation like in SRELU
            return (prb->dir & FLAG_FWD) && signbit(s) && s * tanh(log1pf(expf(s))) <= 10.f * comp_err;
        case LOGISTIC:
            // when s >= 4, logistic(s) -> 0 rapidly, which leads to high
            // relative error of logistic(s) * (1 - logistic(s)) due to
            // catastrohic cancellation.
            return (prb->dir & FLAG_BWD) && !signbit(s) && (1.f / (1.f + expf(s))) <= comp_err;
        case LOGISTIC_DST:
            // when s = logistic(x) ~~ 1, it leads to high relative error of
            // s * (1 - s) due to catastrohic cancellation.
            return (prb->dir & FLAG_BWD) && ((1.f - s) <= comp_err || s <= comp_err);
        case SWISH: {
            // catch cancellation happening when W(s) ~~ -1 in (1 + W(s))
            // formula part on backward.
            const float alpha_s = alpha * s;
            return (prb->dir & FLAG_BWD) && (alpha_s * (1.f - 1.f / (1.f + expf(-alpha_s))) <= comp_err);
        }
        default: return false;
    }
}

static float GetEltwiseThreshold(DNNL_DATA_TYPE_T dt, ELTWISE_KIND_T alg, bool is_fwd) {
    // Tolerate only rounding error (1 ulp) for other than fp32 precisions.
    float trh = 0.0f;
    trh = (dt == dnnl_f16) ? 0x1p-10f : trh;
    trh = (dt == dnnl_bf16) ? 0x1p-7f : trh;
    trh = (dt == dnnl_f32 || dt == dnnl_f64) ? 4e-6f : trh;
    // Tolerate bigger compute errors for complex algorithms.
    const bool alg_has_higher_tolerance = (alg == GELU_TANH) || (alg == ELU) || (alg == SWISH) || (alg == TANH)
            || (alg == SRELU) || (alg == MISH) || (alg == LOG) || ((alg == ELU_DST || alg == TANH_DST) && is_fwd);
    if ((dt == dnnl_f32 || dt == dnnl_f64) && alg_has_higher_tolerance)
        trh = 4e-5f;
    return trh;
}

static void DnnEltwiseCheckError(struct DnnCompareItem *item, float splitValue)
{
    item->diff = fabsf(item->exp - item->got);
    item->rel_diff = item->diff / (fabsf(item->exp) > FLT_MIN ? fabsf(item->exp) : 1);

    if (item->diff == 0.0f) {
        item->ok = true;
        return;
    }

    float cmp = (fabsf(item->exp) > splitValue ? item->rel_diff : item->diff);
    if (cmp <= item->trh) {
        item->ok = true;
        return;
    }

    if (isnan(item->exp) && isnan(item->got)) {
        item->ok = true;
        return;
    }
    if (isinf(item->exp) && isinf(item->got) && signbit(item->exp) == signbit(item->got)) {
        item->ok = true;
        return;
    }

    if (DnnCheckAbsError(item)) {
        if (item->diff <= item->trh) {
            item->ok = true;
            return;
        }
    }

    item->error = cmp - item->trh;
    if (isnan(cmp) || isinf(cmp)) {
        item->error = INFINITY;
    }
    item->ok = false;
    return;
}

void calCaseCnt(struct CaseCnt *casecnt, double maxerror, struct UlpInfo *ulpInfo)
{
    if (fabs(maxerror) == 0) {
        casecnt->success++;
    } else if (maxerror == -1) {
        casecnt->skipped++;
    } else if (fabs(maxerror) > ulpInfo->errorlim) {
        casecnt->cntfail++;
    }
    casecnt->total++;
}

void printFloat(double error, struct UlpInfo *ulpInfo, int argsNum, struct CaseCnt *casecnt)
{
    if (error > ulpInfo->errorlim) {
        printf("\e[1;31m" "FAIL " "\e[0m");
    } else {
        printf("\e[1;32m" "PASS " "\e[0m");
    }

    printf("%s in ", ulpInfo->funcInfo->ulp_name);
    int x = 1;
    for (int i = 0; i < argsNum; i++) {
        printf("[%a;%a]", Asfloat(ulpInfo->interval[i].low), Asfloat(ulpInfo->interval[i].high));
        if (x < argsNum) {
            printf(" x ");
            x++;
        }
    }

    printf("\n");
    if (CHECK_ULP(ulpInfo->errorBase)) {
        printf("    g_rounding_mode:[FE_TONEAREST]  error_limit(default:infinite):[%g]  maxerr:[%g + 0.5]ulp \n",
            ulpInfo->errorlim, error);
    } else {
        printf("    g_rounding_mode:[FE_TONEAREST]  error_limit(default:infinite):[%g]  maxerr:[%e] \n",
            ulpInfo->errorlim, error);
    }

    printf("\t total:%llu success:%llu %g%% skipped:%llu %g%% cntfail:%llu %g%%\n",
        (unsigned long long)casecnt->total,
        (unsigned long long)casecnt->success, 100.0 * casecnt->success / (casecnt->total == 0 ? 1 : casecnt->total),
        (unsigned long long)casecnt->skipped, 100.0 * casecnt->skipped / (casecnt->total == 0 ? 1 : casecnt->total),
        (unsigned long long)casecnt->cntfail, 100.0 * casecnt->cntfail / (casecnt->total == 0 ? 1 : casecnt->total));
}

void printFloat16(double error, struct UlpInfo *ulpInfo, int argsNum, struct CaseCnt *casecnt)
{
    if (error > ulpInfo->errorlim) {
        printf("\e[1;31m" "FAIL " "\e[0m");
    } else {
        printf("\e[1;32m" "PASS " "\e[0m");
    }
    printf("%s in ", ulpInfo->funcInfo->ulp_name);
    int x = 1;
    for (int i = 0; i < argsNum; i++) {
        printf("[%a;%a]", Asfloat16(ulpInfo->interval[i].low), Asfloat16(ulpInfo->interval[i].high));
        if (x < argsNum) {
            printf(" x ");
            x++;
        }
    }

    printf("\n");
    if (CHECK_ULP(ulpInfo->errorBase)) {
        printf("    g_rounding_mode:[FE_TONEAREST]  error_limit(default:infinite):[%g]  maxerr:[%g + 0.5]ulp \n",
            ulpInfo->errorlim, error);
    } else {
        printf("    g_rounding_mode:[FE_TONEAREST]  error_limit(default:infinite):[%g]  maxerr:[%e] \n",
            ulpInfo->errorlim, error);
    }

    printf("\t total:%llu success:%llu %g%% skipped:%llu %g%% cntfail:%llu %g%%\n",
        (unsigned long long)casecnt->total,
        (unsigned long long)casecnt->success, 100.0 * casecnt->success / (casecnt->total == 0 ? 1 : casecnt->total),
        (unsigned long long)casecnt->skipped, 100.0 * casecnt->skipped / (casecnt->total == 0 ? 1 : casecnt->total),
        (unsigned long long)casecnt->cntfail, 100.0 * casecnt->cntfail / (casecnt->total == 0 ? 1 : casecnt->total));
}

void printBfloat16(double error, struct UlpInfo *ulpInfo, int argsNum, struct CaseCnt *casecnt)
{
    if (error > ulpInfo->errorlim) {
        printf("\e[1;31m" "FAIL " "\e[0m");
    } else {
        printf("\e[1;32m" "PASS " "\e[0m");
    }
    printf("%s in ", ulpInfo->funcInfo->ulp_name);
    int x = 1;
    for (int i = 0; i < argsNum; i++) {
        uint64_t low = ulpInfo->interval[i].low << 16;
        uint64_t high = ulpInfo->interval[i].high << 16;
        printf("[%a;%a]", Asfloat(low), Asfloat(high));
        if (x < argsNum) {
            printf(" x ");
            x++;
        }
    }

    printf("\n");
    if (CHECK_ULP(ulpInfo->errorBase)) {
        printf("    g_rounding_mode:[FE_TONEAREST]  error_limit(default:infinite):[%g]  maxerr:[%g + 0.5]ulp \n",
            ulpInfo->errorlim, error);
    } else {
        printf("    g_rounding_mode:[FE_TONEAREST]  error_limit(default:infinite):[%g]  maxerr:[%e] \n",
            ulpInfo->errorlim, error);
    }

    printf("\t total:%llu success:%llu %g%% skipped:%llu %g%% cntfail:%llu %g%%\n",
        (unsigned long long)casecnt->total,
        (unsigned long long)casecnt->success, 100.0 * casecnt->success / (casecnt->total == 0 ? 1 : casecnt->total),
        (unsigned long long)casecnt->skipped, 100.0 * casecnt->skipped / (casecnt->total == 0 ? 1 : casecnt->total),
        (unsigned long long)casecnt->cntfail, 100.0 * casecnt->cntfail / (casecnt->total == 0 ? 1 : casecnt->total));
}

float GetF32FromBF16(__bf16 x)
{
    return Asfloat((uint32_t)(Asuintbf16(x) << 16));
}

void UlpCaseFuncVMLH(struct UlpInfo *ulpInfo)
{
    double error = 0.0;
    int64_t vlen = ulpInfo->lengthDnn;
    const int argsNum = 1;
    __fp16 cases[argsNum];
    struct CaseCnt casecnt = {0, 0, 0, 0};

    /* apply space */
    MALLOC_DECLARE_N3(__fp16, src, initSrc, res);
    MALLOC_DECLARE_N1(struct MpfrRes, mpfrRes);
    MALLOC_DIFF_TYPE_N4(src, initSrc, res, mpfrRes, __fp16, __fp16, __fp16, struct MpfrRes, ulpInfo->lengthDnn);

    float trh = ulpInfo->errorBase;
    if (CHECK_ULP(ulpInfo->errorBase)) {
        trh = GetEltwiseThreshold(dnnl_f16, ulpInfo->funcInfo->alg, ulpInfo->funcInfo->dir & FLAG_FWD);
    }
    ulpInfo->errorlim = 0.0;

    for (int times = 0; times < ulpInfo->times; times++) {
        /* 2.generate random cases */
        gencase_h_new(src, &ulpInfo->interval[0], ulpInfo->testMode, vlen);

        /* 3.calculate mpfr/mpc result */
        for (int64_t len = 0; len < vlen; len++) {
            cases[0] = src[len];
            initSrc[len] = src[len];
            mpfr_result_h(ulpInfo->funcInfo, cases, mpfrRes + len);
        }
        if (ulpInfo->exeMode) { // support in-place execution
            res = src;
        }

        /* 4.call kpl/intel interface and calculate result */
        ((kpl_vmlh)ulpInfo->funcInfo->kplIntelApiDefault)(vlen, src, res);

        struct DnnCompareItem *itemInfo = (struct DnnCompareItem *)calloc(1, sizeof(struct DnnCompareItem));
        /* 5.calculate accuracy ulp error */
        for (int64_t number = 0; number < vlen; number++) {
            itemInfo->exp = mpfrRes[number].res.h;
            itemInfo->got = res[number];
            itemInfo->trh = trh;
            itemInfo->alpha = 0.0f;
            itemInfo->beta = 0.0f;
            itemInfo->dir = ulpInfo->funcInfo->dir;
            itemInfo->alg = ulpInfo->funcInfo->alg;
            itemInfo->error = 0.0f;
            itemInfo->ok = true;
            itemInfo->src = initSrc[number];

            DnnEltwiseCheckError(itemInfo, 0x1p-14f);
            if(!itemInfo->ok || (ulpInfo->gdbPrint && ulpInfo->quiet)) {
                printf("%s(%a / 0x%04x)", ulpInfo->funcInfo->ulp_name, initSrc[number], Asuint16(initSrc[number]));
                printf("got %a want %a ", itemInfo->got, itemInfo->exp);
                printf("diff %a ref_diff %a trh %a / %0.10f \n",
                    itemInfo->diff, itemInfo->rel_diff, itemInfo->trh, itemInfo->trh);
            }
            error = fmax(error, itemInfo->error);

            calCaseCnt(&casecnt, itemInfo->error, ulpInfo);
        }
        free(itemInfo);
    }

    /* 6.ulp tool print result */
    printFloat16(error, ulpInfo, argsNum, &casecnt);

    if (ulpInfo->exeMode) {
        FREE_IF_NOT_NULL_N3(src, initSrc, mpfrRes);
    } else {
        FREE_IF_NOT_NULL_N4(src, initSrc, res, mpfrRes);
    }
}

void UlpCaseFuncVMLB(struct UlpInfo *ulpInfo)
{
    double error = 0.0;
    int64_t vlen = ulpInfo->lengthDnn;
    int argsNum = 1;
    float cases[argsNum];
    struct CaseCnt casecnt = {0, 0, 0, 0};

    /* apply space */
    MALLOC_DECLARE_N3(__bf16, src, initSrc, res);
    MALLOC_DECLARE_N1(struct MpfrRes, mpfrRes);
    MALLOC_DIFF_TYPE_N4(src, initSrc, res, mpfrRes, __bf16,  __bf16, __bf16, struct MpfrRes, ulpInfo->lengthDnn);

    float trh = ulpInfo->errorBase;
    if (CHECK_ULP(ulpInfo->errorBase)) {
        trh = GetEltwiseThreshold(dnnl_bf16, ulpInfo->funcInfo->alg, ulpInfo->funcInfo->dir & FLAG_FWD);
    }
    ulpInfo->errorlim = 0.0;

    for (int times = 0; times < ulpInfo->times; times++) {
        /* 2.generate random cases */
        gencase_b_new(src, &ulpInfo->interval[0], ulpInfo->testMode, vlen);

        /* 3.calculate mpfr/mpc result */
        for (int64_t len = 0; len < vlen; len++) {
            cases[0] = GetF32FromBF16(src[len]);
            initSrc[len] = src[len];
            mpfr_result_f(ulpInfo->funcInfo, cases, mpfrRes + len);
        }
        if (ulpInfo->exeMode) { // support in-place execution
            res = src;
        }

        /* 4.call kpl/intel interface and calculate result */
        ((kpl_vmlb)ulpInfo->funcInfo->kplIntelApiDefault)(vlen, src, res);

        struct DnnCompareItem *itemInfo = (struct DnnCompareItem *)calloc(1, sizeof(struct DnnCompareItem));
        /* 5.calculate accuracy ulp error */
        for (int64_t number = 0; number < vlen; number++) {
            itemInfo->exp = GetBf16CompareResult(mpfrRes[number].res.f);
            itemInfo->got = GetF32FromBF16(res[number]);
            itemInfo->trh = trh;
            itemInfo->alpha = 0.0f;
            itemInfo->beta = 0.0f;
            itemInfo->dir = ulpInfo->funcInfo->dir;
            itemInfo->alg = ulpInfo->funcInfo->alg;
            itemInfo->error = 0.0f;
            itemInfo->ok = true;
            itemInfo->src = GetF32FromBF16(initSrc[number]);

            DnnEltwiseCheckError(itemInfo, 1e-5f);
            if(!itemInfo->ok || (ulpInfo->gdbPrint && ulpInfo->quiet)) {
                printf("%s(%a / 0x%04x)", ulpInfo->funcInfo->ulp_name,
                    GetF32FromBF16(initSrc[number]), Asuintbf16(initSrc[number]));
                printf("got %a want %a ", itemInfo->got, itemInfo->exp);
                printf("diff %a ref_diff %a trh %a / %0.10f \n",
                    itemInfo->diff, itemInfo->rel_diff, itemInfo->trh, itemInfo->trh);
            }
            error = fmax(error, itemInfo->error);

            calCaseCnt(&casecnt, itemInfo->error, ulpInfo);
        }
        free(itemInfo);
    }

    /* 6.ulp tool print result */
    printBfloat16(error, ulpInfo, argsNum, &casecnt);

    if (ulpInfo->exeMode) {
        FREE_IF_NOT_NULL_N3(src, initSrc, mpfrRes);
    } else {
        FREE_IF_NOT_NULL_N4(src, initSrc, res, mpfrRes);
    }
}

void UlpCaseFuncVMLF(struct UlpInfo *ulpInfo)
{
    double error = 0.0;
    int64_t vlen = ulpInfo->lengthDnn;
    int argsNum = 1;
    float cases[argsNum];
    struct CaseCnt casecnt = {0, 0, 0, 0};

    /* apply space */
    MALLOC_DECLARE_N3(float, src, initSrc, res);
    MALLOC_DECLARE_N1(struct MpfrRes, mpfrRes);
    MALLOC_DIFF_TYPE_N4(src, initSrc, res, mpfrRes, float, float, float, struct MpfrRes, ulpInfo->lengthDnn);

    float trh = ulpInfo->errorBase;
    if (CHECK_ULP(ulpInfo->errorBase)) {
        trh = GetEltwiseThreshold(dnnl_f32, ulpInfo->funcInfo->alg, ulpInfo->funcInfo->dir & FLAG_FWD);
    }
    ulpInfo->errorlim = 0.0;

    for (int times = 0; times < ulpInfo->times; times++) {
        /* 2.generate random cases */
        gencase_f_new(src, &ulpInfo->interval[0], ulpInfo->testMode, vlen);

        /* 3.calculate mpfr/mpc result */
        for (int64_t len = 0; len < vlen; len++) {
            cases[0] = src[len];
            initSrc[len] = src[len];
            mpfr_result_f(ulpInfo->funcInfo, cases, mpfrRes + len);
        }
        if (ulpInfo->exeMode) { // support in-place execution
            res = src;
        }

        /* 4.call kpl/intel interface and calculate result */
        ((kpl_vmlf1)ulpInfo->funcInfo->kplIntelApiDefault)(vlen, src, res);

        struct DnnCompareItem *itemInfo = (struct DnnCompareItem *)calloc(1, sizeof(struct DnnCompareItem));
        /* 5.calculate accuracy ulp error */
        for (int64_t number = 0; number < vlen; number++) {
            itemInfo->exp = mpfrRes[number].res.f;
            itemInfo->got = res[number];
            itemInfo->trh = trh;
            itemInfo->alpha = 0.0f;
            itemInfo->beta = 0.0f;
            itemInfo->dir = ulpInfo->funcInfo->dir;
            itemInfo->alg = ulpInfo->funcInfo->alg;
            itemInfo->error = 0.0f;
            itemInfo->ok = true;
            itemInfo->src = initSrc[number];

            DnnEltwiseCheckError(itemInfo, 1e-5f);
            if(!itemInfo->ok || (ulpInfo->gdbPrint && ulpInfo->quiet)) {
                printf("%s(%a / 0x%08x)", ulpInfo->funcInfo->ulp_name, initSrc[number], Asuint(initSrc[number]));
                printf("got %a want %a ", itemInfo->got, itemInfo->exp);
                printf("diff %a ref_diff %a trh %a / %0.10f \n",
                    itemInfo->diff, itemInfo->rel_diff, itemInfo->trh, itemInfo->trh);
            }
            error = fmax(error, itemInfo->error);

            calCaseCnt(&casecnt, itemInfo->error, ulpInfo);
        }
        free(itemInfo);
    }

    /* 6.ulp tool print result */
    printFloat(error, ulpInfo, argsNum, &casecnt);

    if (ulpInfo->exeMode) {
        FREE_IF_NOT_NULL_N3(src, initSrc, mpfrRes);
    } else {
        FREE_IF_NOT_NULL_N4(src, initSrc, res, mpfrRes);
    }
}

void UlpCaseFuncVMLXH(struct UlpInfo *ulpInfo)
{
    double error = 0.0;
    int64_t vlen = ulpInfo->lengthDnn;
    int argsNum = 2;
    float cases[argsNum];
    struct CaseCnt casecnt = {0, 0, 0, 0};

    /* apply space */
    MALLOC_DECLARE_N2(float, alpha, srcF32);
    MALLOC_DECLARE_N3(__fp16, srcF16, initSrcF16, res);
    MALLOC_DECLARE_N1(struct MpfrRes, mpfrRes);
    MALLOC_DIFF_TYPE_N6(alpha, srcF32, srcF16, initSrcF16, res, mpfrRes,
        float, float, __fp16, __fp16, __fp16, struct MpfrRes, ulpInfo->lengthDnn);

    float trh = ulpInfo->errorBase;
    if (CHECK_ULP(ulpInfo->errorBase)) {
        trh = GetEltwiseThreshold(dnnl_f16, ulpInfo->funcInfo->alg, ulpInfo->funcInfo->dir & FLAG_FWD);
    }
    ulpInfo->errorlim = 0.0;

    for (int times = 0; times < ulpInfo->times; times++) {
        /* 2.generate random cases */
        gencase_f_new(alpha, &ulpInfo->interval[0], ulpInfo->testMode, vlen);
        gencase_f_new(srcF32, &ulpInfo->interval[1], ulpInfo->testMode, vlen);

        for (int64_t i = 0; i < vlen; i++) {
            srcF16[i] = (__fp16)srcF32[i];
            initSrcF16[i] = srcF16[i];
        }
        if (ulpInfo->exeMode) { // support in-place execution
            res = srcF16;
        }

        for (int64_t i = 0; i < vlen; i++) {
            if ((ulpInfo->funcInfo->alg == SRELU) && (alpha[i] == 0.f)) {
                calCaseCnt(&casecnt, -1, ulpInfo);
                continue;
            }
            /* 3.calculate mpfr/mpc result */
            for (int64_t j = 0; j < vlen; j++) {
                cases[0] = alpha[i];
                cases[1] = srcF16[j];
                mpfr_result_fh(ulpInfo->funcInfo, cases, mpfrRes + j);
            }

            /* 4.call kpl/intel interface and calculate result */
            ((kpl_vmlxh)ulpInfo->funcInfo->kplIntelApiDefault)(vlen, alpha[i], srcF16, res);

            struct DnnCompareItem *itemInfo = (struct DnnCompareItem *)calloc(1, sizeof(struct DnnCompareItem));
            /* 5.calculate accuracy ulp error */
            for (int64_t number = 0; number < vlen; number++) {
                itemInfo->exp = mpfrRes[number].res.h;
                itemInfo->got = res[number];
                itemInfo->trh = trh;
                itemInfo->alpha = alpha[i];
                itemInfo->beta = 0.0f;
                itemInfo->dir = ulpInfo->funcInfo->dir;
                itemInfo->alg = ulpInfo->funcInfo->alg;
                itemInfo->error = 0.0f;
                itemInfo->ok = true;
                itemInfo->src = initSrcF16[number];

                DnnEltwiseCheckError(itemInfo, 0x1p-14f);
                if(!itemInfo->ok || (ulpInfo->gdbPrint && ulpInfo->quiet)) {
                    printf("%s(%a / 0x%08x, %a / 0x%04x)", ulpInfo->funcInfo->ulp_name,
                        alpha[i], Asuint(alpha[i]), initSrcF16[number], Asuint16(initSrcF16[number]));
                    printf("got %a want %a ", itemInfo->got, itemInfo->exp);
                    printf("diff %a ref_diff %a trh %a / %0.10f \n",
                        itemInfo->diff, itemInfo->rel_diff, itemInfo->trh, itemInfo->trh);
                }
                error = fmax(error, itemInfo->error);

                calCaseCnt(&casecnt, itemInfo->error, ulpInfo);
            }
            free(itemInfo);
        }
    }

    /* 6.ulp tool print result */
    printFloat(error, ulpInfo, argsNum, &casecnt);

    if (ulpInfo->exeMode) {
        FREE_IF_NOT_NULL_N5(srcF32, srcF16, initSrcF16, alpha, mpfrRes);
    } else {
        FREE_IF_NOT_NULL_N6(srcF32, srcF16, initSrcF16, alpha, res, mpfrRes);
    }
}

void UlpCaseFuncVMLXB(struct UlpInfo *ulpInfo)
{
    double error = 0.0;
    int64_t vlen = ulpInfo->lengthDnn;
    int argsNum = 2;
    float cases[argsNum];
    struct CaseCnt casecnt = {0, 0, 0, 0};

    /* apply space */
    MALLOC_DECLARE_N2(float, alpha, srcF32);
    MALLOC_DECLARE_N3(__bf16, srcBf16, initSrcBf16, res);
    MALLOC_DECLARE_N1(struct MpfrRes, mpfrRes);
    MALLOC_DIFF_TYPE_N6(alpha, srcF32, srcBf16, initSrcBf16, res, mpfrRes,
        float, float, __bf16, __bf16, __bf16, struct MpfrRes, ulpInfo->lengthDnn);

    float trh = ulpInfo->errorBase;
    if (CHECK_ULP(ulpInfo->errorBase)) {
        trh = GetEltwiseThreshold(dnnl_bf16, ulpInfo->funcInfo->alg, ulpInfo->funcInfo->dir & FLAG_FWD);
    }
    ulpInfo->errorlim = 0.0;

    for (int times = 0; times < ulpInfo->times; times++) {
        /* 2.generate random cases */
        gencase_f_new(alpha, &ulpInfo->interval[0], ulpInfo->testMode, vlen);
        gencase_f_new(srcF32, &ulpInfo->interval[1], ulpInfo->testMode, vlen);

        for (int64_t i = 0; i < vlen; i++) {
            uint32_t u = Asuint(srcF32[i]) & 0xffff0000;
            srcBf16[i] = Asbfloat16((uint16_t)(u >> 16));
            initSrcBf16[i] = srcBf16[i];
        }
        if (ulpInfo->exeMode) { // support in-place execution
            res = srcBf16;
        }

        for (int64_t i = 0; i < vlen; i++) {
            if ((ulpInfo->funcInfo->alg == SRELU) && (alpha[i] == 0.f)) {
                calCaseCnt(&casecnt, -1, ulpInfo);
                continue;
            }
            /* 3.calculate mpfr/mpc result */
            for (int64_t j = 0; j < vlen; j++) {
                cases[0] = alpha[i];
                cases[1] = GetF32FromBF16(srcBf16[j]);
                mpfr_result_f2(ulpInfo->funcInfo, cases, mpfrRes + j);
            }

            /* 4.call kpl/intel interface and calculate result */
            ((kpl_vmlxb)ulpInfo->funcInfo->kplIntelApiDefault)(vlen, alpha[i], srcBf16, res);

            struct DnnCompareItem *itemInfo = (struct DnnCompareItem *)calloc(1, sizeof(struct DnnCompareItem));
            /* 5.calculate accuracy ulp error */
            for (int64_t j = 0; j < vlen; j++) {
                itemInfo->exp = GetBf16CompareResult(mpfrRes[j].res.f);
                itemInfo->got = GetF32FromBF16(res[j]);
                itemInfo->trh = trh;
                itemInfo->alpha = alpha[i];
                itemInfo->beta = 0.0f;
                itemInfo->dir = ulpInfo->funcInfo->dir;
                itemInfo->alg = ulpInfo->funcInfo->alg;
                itemInfo->error = 0.0f;
                itemInfo->ok = true;
                itemInfo->src = GetF32FromBF16(initSrcBf16[j]);

                DnnEltwiseCheckError(itemInfo, 1e-5f);
                if(!itemInfo->ok || (ulpInfo->gdbPrint && ulpInfo->quiet)) {
                    printf("%s(%a / 0x%08x, %a / 0x%04x)", ulpInfo->funcInfo->ulp_name,
                        alpha[i], Asuint(alpha[i]), GetF32FromBF16(initSrcBf16[j]), Asuintbf16(initSrcBf16[j]));
                    printf("got %a want %a ", itemInfo->got, itemInfo->exp);
                    printf("diff %a ref_diff %a trh %a / %0.10f \n",
                        itemInfo->diff, itemInfo->rel_diff, itemInfo->trh, itemInfo->trh);
                }
                error = fmax(error, itemInfo->error);

                calCaseCnt(&casecnt, itemInfo->error, ulpInfo);
            }
            free(itemInfo);
        }
    }

    /* 6.ulp tool print result */
    printFloat(error, ulpInfo, argsNum, &casecnt);

    if (ulpInfo->exeMode) {
        FREE_IF_NOT_NULL_N5(srcF32, srcBf16, initSrcBf16, alpha, mpfrRes);
    } else {
        FREE_IF_NOT_NULL_N6(srcF32, srcBf16, initSrcBf16, alpha, res, mpfrRes);
    }
}

void UlpCaseFuncVMLXF(struct UlpInfo *ulpInfo)
{
    double error = 0.0;
    int64_t vlen = ulpInfo->lengthDnn;
    int argsNum = 2;
    float cases[argsNum];
    struct CaseCnt casecnt = {0, 0, 0, 0};

    /* apply space */
    MALLOC_DECLARE_N4(float, alpha, src, initSrc, res);
    MALLOC_DECLARE_N1(struct MpfrRes, mpfrRes);
    MALLOC_DIFF_TYPE_N5(alpha, src, initSrc, res, mpfrRes,
        float, float, float, float, struct MpfrRes, ulpInfo->lengthDnn);

    float trh = ulpInfo->errorBase;
    if (CHECK_ULP(ulpInfo->errorBase)) {
        trh = GetEltwiseThreshold(dnnl_f32, ulpInfo->funcInfo->alg, ulpInfo->funcInfo->dir & FLAG_FWD);
    }
    ulpInfo->errorlim = 0.0;

    for (int times = 0; times < ulpInfo->times; times++) {
        /* 2.generate random cases */
        gencase_f_new(alpha, &ulpInfo->interval[0], ulpInfo->testMode, vlen);
        gencase_f_new(src, &ulpInfo->interval[1], ulpInfo->testMode, vlen);

        for (int64_t i = 0; i < vlen; i++) {
            initSrc[i] = src[i];
        }
        if (ulpInfo->exeMode) { // support in-place execution
            res = src;
        }

        for (int64_t i = 0; i < vlen; i++) {
            if ((ulpInfo->funcInfo->alg == SRELU) && (alpha[i] == 0.f)) {
                calCaseCnt(&casecnt, -1, ulpInfo);
                continue;
            }
            /* 3.calculate mpfr/mpc result */
            for (int64_t j = 0; j < vlen; j++) {
                cases[0] = alpha[i];
                cases[1] = src[j];
                mpfr_result_f2(ulpInfo->funcInfo, cases, mpfrRes + j);
            }

            /* 4.call kpl/intel interface and calculate result */
            ((kpl_vmlxf)ulpInfo->funcInfo->kplIntelApiDefault)(vlen, alpha[i], src, res);

            struct DnnCompareItem *itemInfo = (struct DnnCompareItem *)calloc(1, sizeof(struct DnnCompareItem));
            /* 5.calculate accuracy ulp error */
            for (int64_t number = 0; number < vlen; number++) {
                itemInfo->exp = mpfrRes[number].res.f;
                itemInfo->got = res[number];
                itemInfo->trh = trh;
                itemInfo->alpha = alpha[i];
                itemInfo->beta = 0.0f;
                itemInfo->dir = ulpInfo->funcInfo->dir;
                itemInfo->alg = ulpInfo->funcInfo->alg;
                itemInfo->error = 0.0f;
                itemInfo->ok = true;
                itemInfo->src = initSrc[number];

                DnnEltwiseCheckError(itemInfo, 1e-5f);
                if(!itemInfo->ok || (ulpInfo->gdbPrint && ulpInfo->quiet)) {
                    printf("%s(%a / 0x%08x, %a / 0x%08x)", ulpInfo->funcInfo->ulp_name,
                        alpha[i], Asuint(alpha[i]), initSrc[number], Asuint(initSrc[number]));
                    printf("got %a want %a ", itemInfo->got, itemInfo->exp);
                    printf("diff %a ref_diff %a trh %a / %0.10f \n",
                        itemInfo->diff, itemInfo->rel_diff, itemInfo->trh, itemInfo->trh);
                }
                error = fmax(error, itemInfo->error);

                calCaseCnt(&casecnt, itemInfo->error, ulpInfo);
            }
            free(itemInfo);
        }
    }

    /* 6.ulp tool print result */
    printFloat(error, ulpInfo, argsNum, &casecnt);

    if (ulpInfo->exeMode) {
        FREE_IF_NOT_NULL_N4(alpha, src, initSrc, mpfrRes);
    } else {
        FREE_IF_NOT_NULL_N5(alpha, src, initSrc, res, mpfrRes);
    }
}

void UlpCaseFuncVMLXYH(struct UlpInfo *ulpInfo)
{
    double error = 0.0;
    int64_t vlen = ulpInfo->lengthDnn;
    int argsNum = 3;
    float cases[argsNum];
    struct CaseCnt casecnt = {0, 0, 0, 0};

    /* apply space */
    MALLOC_DECLARE_N3(float, alpha, beta, srcF32);
    MALLOC_DECLARE_N3(__fp16, srcF16, initSrcF16, res);
    MALLOC_DECLARE_N1(struct MpfrRes, mpfrRes);
    MALLOC_DIFF_TYPE_N7(alpha, beta, srcF32, srcF16, initSrcF16, res, mpfrRes,
        float, float, float, __fp16, __fp16, __fp16, struct MpfrRes, ulpInfo->lengthDnn);

    float trh = ulpInfo->errorBase;
    if (CHECK_ULP(ulpInfo->errorBase)) {
        trh = GetEltwiseThreshold(dnnl_f16, ulpInfo->funcInfo->alg, ulpInfo->funcInfo->dir & FLAG_FWD);
    }
    ulpInfo->errorlim = 0.0;

    for (int times = 0; times < ulpInfo->times; times++) {
        /* 2.generate random cases */
        gencase_f_new(alpha, &ulpInfo->interval[0], ulpInfo->testMode, vlen);
        gencase_f_new(beta, &ulpInfo->interval[1], ulpInfo->testMode, vlen);
        gencase_f_new(srcF32, &ulpInfo->interval[2], ulpInfo->testMode, vlen);

        for (int64_t i = 0; i < vlen; i++) {
            srcF16[i] = (__fp16)srcF32[i];
            initSrcF16[i] = srcF16[i];
        }
        if (ulpInfo->exeMode) { // support in-place execution
            res = srcF16;
        }

        for (int64_t i = 0; i < vlen; i++) {
            if ((ulpInfo->funcInfo->alg == CLIP || ulpInfo->funcInfo->alg == CLIP_V2 ||
                ulpInfo->funcInfo->alg == CLIP_V2_DST) && (alpha[i] > beta[i])) {
                    calCaseCnt(&casecnt, -1, ulpInfo);
                    continue;
            }
            /* 3.calculate mpfr/mpc result */
            for (int64_t j = 0; j < vlen; j++) {
                cases[0] = alpha[i];
                cases[1] = beta[i];
                cases[2] = srcF16[j];
                mpfr_result_f3(ulpInfo->funcInfo, cases, mpfrRes + j);
            }

            /* 4.call kpl/intel interface and calculate result */
            ((kpl_vmlxyh)ulpInfo->funcInfo->kplIntelApiDefault)(vlen, alpha[i], beta[i], srcF16, res);

            struct DnnCompareItem *itemInfo = (struct DnnCompareItem *)calloc(1, sizeof(struct DnnCompareItem));
            /* 5.calculate accuracy ulp error */
            for (int64_t number = 0; number < vlen; number++) {
                itemInfo->exp = (__fp16)mpfrRes[number].res.f;
                itemInfo->got = res[number];
                itemInfo->trh = trh;
                itemInfo->alpha = alpha[i];
                itemInfo->beta = beta[i];
                itemInfo->dir = ulpInfo->funcInfo->dir;
                itemInfo->alg = ulpInfo->funcInfo->alg;
                itemInfo->error = 0.0f;
                itemInfo->ok = true;
                itemInfo->src = initSrcF16[number];

                DnnEltwiseCheckError(itemInfo, 0x1p-14f);
                if(!itemInfo->ok || (ulpInfo->gdbPrint && ulpInfo->quiet)) {
                    printf("%s(%a / 0x%08x, %a / 0x%08x, %a / 0x%04x)", ulpInfo->funcInfo->ulp_name,
                        alpha[i], Asuint(alpha[i]), beta[i], Asuint(beta[i]),
                        initSrcF16[number], Asuint16(initSrcF16[number]));
                    printf("got %a want %a ", itemInfo->got, itemInfo->exp);
                    printf("diff %a ref_diff %a trh %a / %0.10f \n",
                        itemInfo->diff, itemInfo->rel_diff, itemInfo->trh, itemInfo->trh);
                }
                error = fmax(error, itemInfo->error);

                calCaseCnt(&casecnt, itemInfo->error, ulpInfo);
            }
            free(itemInfo);
        }
    }

    /* 6.ulp tool print result */
    printFloat(error, ulpInfo, argsNum, &casecnt);

    if (ulpInfo->exeMode) {
        FREE_IF_NOT_NULL_N6(srcF32, srcF16, initSrcF16, alpha, beta, mpfrRes);
    } else {
        FREE_IF_NOT_NULL_N7(srcF32, srcF16, initSrcF16, alpha, beta, res, mpfrRes);
    }
}

void UlpCaseFuncVMLXYB(struct UlpInfo *ulpInfo)
{
    double error = 0.0;
    int64_t vlen = ulpInfo->lengthDnn;
    int argsNum = 3;
    float cases[argsNum];
    struct CaseCnt casecnt = {0, 0, 0, 0};

    /* apply space */
    MALLOC_DECLARE_N3(float, alpha, beta, srcF32);
    MALLOC_DECLARE_N3(__bf16, srcBf16, initSrcBf16, res);
    MALLOC_DECLARE_N1(struct MpfrRes, mpfrRes);
    MALLOC_DIFF_TYPE_N7(alpha, beta, srcF32, srcBf16, initSrcBf16, res, mpfrRes,
        float, float, float, __bf16, __bf16, __bf16, struct MpfrRes, ulpInfo->lengthDnn);

    float trh = ulpInfo->errorBase;
    if (CHECK_ULP(ulpInfo->errorBase)) {
        trh = GetEltwiseThreshold(dnnl_bf16, ulpInfo->funcInfo->alg, ulpInfo->funcInfo->dir & FLAG_FWD);
    }
    ulpInfo->errorlim = 0.0;

    for (int times = 0; times < ulpInfo->times; times++) {
        /* 2.generate random cases */
        gencase_f_new(alpha, &ulpInfo->interval[0], ulpInfo->testMode, vlen);
        gencase_f_new(beta, &ulpInfo->interval[1], ulpInfo->testMode, vlen);
        gencase_f_new(srcF32, &ulpInfo->interval[2], ulpInfo->testMode, vlen);

        for (int64_t i = 0; i < vlen; i++) {
            uint32_t u = Asuint(srcF32[i]) & 0xffff0000;
            srcBf16[i] = Asbfloat16((uint16_t)(u >> 16));
            initSrcBf16[i] = srcBf16[i];
        }
        if (ulpInfo->exeMode) { // support in-place execution
            res = srcBf16;
        }

        for (int64_t i = 0; i < vlen; i++) {
            if ((ulpInfo->funcInfo->alg == CLIP || ulpInfo->funcInfo->alg == CLIP_V2 ||
                ulpInfo->funcInfo->alg == CLIP_V2_DST) && (alpha[i] > beta[i])) {
                    calCaseCnt(&casecnt, -1, ulpInfo);
                    continue;
            }
            /* 3.calculate mpfr/mpc result */
            for (int64_t j = 0; j < vlen; j++) {
                cases[0] = alpha[i];
                cases[1] = beta[i];
                cases[2] = GetF32FromBF16(srcBf16[j]);
                mpfr_result_f3(ulpInfo->funcInfo, cases, mpfrRes + j);
            }

            /* 4.call kpl/intel interface and calculate result */
            ((kpl_vmlxyb)ulpInfo->funcInfo->kplIntelApiDefault)(vlen, alpha[i], beta[i], srcBf16, res);

            struct DnnCompareItem *itemInfo = (struct DnnCompareItem *)calloc(1, sizeof(struct DnnCompareItem));
            /* 5.calculate accuracy ulp error */
            for (int64_t j = 0; j < vlen; j++) {
                itemInfo->exp = GetBf16CompareResult(mpfrRes[j].res.f);
                itemInfo->got = GetF32FromBF16(res[j]);
                itemInfo->trh = trh;
                itemInfo->alpha = alpha[i];
                itemInfo->beta = beta[i];
                itemInfo->dir = ulpInfo->funcInfo->dir;
                itemInfo->alg = ulpInfo->funcInfo->alg;
                itemInfo->error = 0.0f;
                itemInfo->ok = true;
                itemInfo->src = GetF32FromBF16(initSrcBf16[j]);

                DnnEltwiseCheckError(itemInfo, 1e-5f);
                if(!itemInfo->ok || (ulpInfo->gdbPrint && ulpInfo->quiet)) {
                    printf("%s(%a / 0x%08x, %a / 0x%04x, %a / 0x%04x)", ulpInfo->funcInfo->ulp_name,
                        alpha[i], Asuint(alpha[i]), beta[i], Asuint(beta[i]),
                        GetF32FromBF16(initSrcBf16[j]), Asuintbf16(initSrcBf16[j]));
                    printf("got %a want %a ", itemInfo->got, itemInfo->exp);
                    printf("diff %a ref_diff %a trh %a / %0.10f \n",
                        itemInfo->diff, itemInfo->rel_diff, itemInfo->trh, itemInfo->trh);
                }
                error = fmax(error, itemInfo->error);

                calCaseCnt(&casecnt, itemInfo->error, ulpInfo);
            }
            free(itemInfo);
        }
    }

    /* 6.ulp tool print result */
    printFloat(error, ulpInfo, argsNum, &casecnt);

    if (ulpInfo->exeMode) {
        FREE_IF_NOT_NULL_N6(srcF32, srcBf16, initSrcBf16, alpha, beta, mpfrRes);
    } else {
        FREE_IF_NOT_NULL_N7(srcF32, srcBf16, initSrcBf16, alpha, beta, res, mpfrRes);
    }
}

void UlpCaseFuncVMLXYF(struct UlpInfo *ulpInfo)
{
    double error = 0.0;
    int64_t vlen = ulpInfo->lengthDnn;
    int argsNum = 3;
    float cases[argsNum];
    struct CaseCnt casecnt = {0, 0, 0, 0};

    /* apply space */
    MALLOC_DECLARE_N5(float, alpha, beta, src, initSrc, res);
    MALLOC_DECLARE_N1(struct MpfrRes, mpfrRes);
    MALLOC_DIFF_TYPE_N6(alpha, beta, src, initSrc, res, mpfrRes,
        float, float, float, float, float, struct MpfrRes, ulpInfo->lengthDnn);

    float trh = ulpInfo->errorBase;
    if (CHECK_ULP(ulpInfo->errorBase)) {
        trh = GetEltwiseThreshold(dnnl_f32, ulpInfo->funcInfo->alg, ulpInfo->funcInfo->dir & FLAG_FWD);
    }
    ulpInfo->errorlim = 0.0;

    for (int times = 0; times < ulpInfo->times; times++) {
        /* 2.generate random cases */
        gencase_f_new(alpha, &ulpInfo->interval[0], ulpInfo->testMode, vlen);
        gencase_f_new(beta, &ulpInfo->interval[1], ulpInfo->testMode, vlen);
        gencase_f_new(src, &ulpInfo->interval[2], ulpInfo->testMode, vlen);

        for (int64_t i = 0; i < vlen; i++) {
            initSrc[i] = src[i];
        }
        if (ulpInfo->exeMode) { // support in-place execution
            res = src;
        }

        for (int64_t i = 0; i < vlen; i++) {
            if ((ulpInfo->funcInfo->alg == CLIP || ulpInfo->funcInfo->alg == CLIP_V2 ||
                ulpInfo->funcInfo->alg == CLIP_V2_DST) && (alpha[i] > beta[i])) {
                    calCaseCnt(&casecnt, -1, ulpInfo);
                    continue;
            }
            /* 3.calculate mpfr/mpc result */
            for (int64_t j = 0; j < vlen; j++) {
                cases[0] = alpha[i];
                cases[1] = beta[i];
                cases[2] = src[j];
                mpfr_result_f3(ulpInfo->funcInfo, cases, mpfrRes + j);
            }

            /* 4.call kpl/intel interface and calculate result */
            ((kpl_vmlxyf)ulpInfo->funcInfo->kplIntelApiDefault)(vlen, alpha[i], beta[i], src, res);

            struct DnnCompareItem *itemInfo = (struct DnnCompareItem *)calloc(1, sizeof(struct DnnCompareItem));
            /* 5.calculate accuracy ulp error */
            for (int64_t number = 0; number < vlen; number++) {
                itemInfo->exp = mpfrRes[number].res.f;
                itemInfo->got = res[number];
                itemInfo->trh = trh;
                itemInfo->alpha = alpha[i];
                itemInfo->beta = beta[i];
                itemInfo->dir = ulpInfo->funcInfo->dir;
                itemInfo->alg = ulpInfo->funcInfo->alg;
                itemInfo->error = 0.0f;
                itemInfo->ok = true;
                itemInfo->src = initSrc[number];

                DnnEltwiseCheckError(itemInfo, 1e-5f);
                if(!itemInfo->ok || (ulpInfo->gdbPrint && ulpInfo->quiet)) {
                    printf("%s(%a / 0x%08x, %a / 0x%08x, %a / 0x%08x)", ulpInfo->funcInfo->ulp_name,
                        alpha[i], Asuint(alpha[i]), beta[i], Asuint(beta[i]), initSrc[number], Asuint(initSrc[number]));
                    printf("got %a want %a ", itemInfo->got, itemInfo->exp);
                    printf("diff %a ref_diff %a trh %a / %0.10f \n",
                        itemInfo->diff, itemInfo->rel_diff, itemInfo->trh, itemInfo->trh);
                }
                error = fmax(error, itemInfo->error);

                calCaseCnt(&casecnt, itemInfo->error, ulpInfo);
            }
            free(itemInfo);
        }
    }

    /* 6.ulp tool print result */
    printFloat(error, ulpInfo, argsNum, &casecnt);

    if (ulpInfo->exeMode) {
        FREE_IF_NOT_NULL_N5(alpha, beta, src, initSrc, mpfrRes);
    } else {
        FREE_IF_NOT_NULL_N6(alpha, beta, src, initSrc, res, mpfrRes);
    }
}

void UlpCaseFuncVMLH2(struct UlpInfo *ulpInfo)
{
    double error = 0.0;
    int vlen = ulpInfo->length;
    int argsNum = 2;
    __fp16 cases[argsNum];
    struct CaseCnt casecnt = {0, 0, 0, 0};

    /* apply space */
    MALLOC_DECLARE_N3(__fp16, src1, src2, res);
    MALLOC_DECLARE_N1(struct MpfrRes, mpfrRes);
    MALLOC_DIFF_TYPE_N4(src1, src2, res, mpfrRes, __fp16, __fp16, __fp16, struct MpfrRes, ulpInfo->length);

    float trh = ulpInfo->errorBase;
    if (CHECK_ULP(ulpInfo->errorBase)) {
        trh = GetEltwiseThreshold(dnnl_f16, ulpInfo->funcInfo->alg, ulpInfo->funcInfo->dir & FLAG_FWD);
    }
    ulpInfo->errorlim = 0.0;

    for (int times = 0; times < ulpInfo->times; times++) {
        /* 2.generate random cases */
        gencase_h_new(src1, &ulpInfo->interval[0], ulpInfo->testMode, vlen);
        gencase_h_new(src2, &ulpInfo->interval[1], ulpInfo->testMode, vlen);

        /* 3.calculate mpfr/mpc result */
        for (int len = 0; len < vlen; len++) {
            cases[0] = src1[len];
            cases[1] = src2[len];
            mpfr_result_h2(ulpInfo->funcInfo, cases, mpfrRes + len);
        }

        /* 4.call kpl/intel interface and calculate result */
        ((kpl_vmlh2)ulpInfo->funcInfo->kplIntelApiDefault)(vlen, src1, src2, res);

        struct DnnCompareItem *itemInfo = (struct DnnCompareItem *)calloc(1, sizeof(struct DnnCompareItem));
        /* 5.calculate accuracy ulp error */
        for (int64_t number = 0; number < vlen; number++) {
            itemInfo->exp = mpfrRes[number].res.h;
            itemInfo->got = res[number];
            itemInfo->trh = trh;
            itemInfo->alpha = 0.0f;
            itemInfo->beta = 0.0f;
            itemInfo->dir = ulpInfo->funcInfo->dir;
            itemInfo->alg = ulpInfo->funcInfo->alg;
            itemInfo->error = 0.0f;
            itemInfo->ok = true;
            itemInfo->src = src1[number];

            DnnEltwiseCheckError(itemInfo, 0x1p-14f);
            if(!itemInfo->ok || (ulpInfo->gdbPrint && ulpInfo->quiet)) {
                printf("%s(%a / 0x%04x, %a / 0x%04x)", ulpInfo->funcInfo->ulp_name,
                    src1[number], Asuint16(src1[number]), src2[number], Asuint16(src2[number]));
                printf("got %a want %a ", itemInfo->got, itemInfo->exp);
                printf("diff %a ref_diff %a trh %a / %0.10f \n",
                    itemInfo->diff, itemInfo->rel_diff, itemInfo->trh, itemInfo->trh);
            }
            error = fmax(error, itemInfo->error);

            calCaseCnt(&casecnt, itemInfo->error, ulpInfo);
        }
        free(itemInfo);
    }

    /* 6.ulp tool print result */
    printFloat16(error, ulpInfo, argsNum, &casecnt);

    if (ulpInfo->exeMode) {
        FREE_IF_NOT_NULL_N3(src1, src2, mpfrRes);
    } else {
        FREE_IF_NOT_NULL_N4(src1, src2, res, mpfrRes);
    }
}

void UlpCaseFuncVMLB2(struct UlpInfo *ulpInfo)
{
    double error = 0.0;
    int vlen = ulpInfo->length;
    int argsNum = 2;
    float cases[argsNum];
    struct CaseCnt casecnt = {0, 0, 0, 0};

    /* apply space */
    MALLOC_DECLARE_N3(__bf16, src1, src2, res);
    MALLOC_DECLARE_N1(struct MpfrRes, mpfrRes);
    MALLOC_DIFF_TYPE_N4(src1, src2, res, mpfrRes, __bf16, __bf16, __bf16, struct MpfrRes, ulpInfo->length);

    float trh = ulpInfo->errorBase;
    if (CHECK_ULP(ulpInfo->errorBase)) {
        trh = GetEltwiseThreshold(dnnl_bf16, ulpInfo->funcInfo->alg, ulpInfo->funcInfo->dir & FLAG_FWD);
    }
    ulpInfo->errorlim = 0.0;

    for (int times = 0; times < ulpInfo->times; times++) {
        /* 2.generate random cases */
        gencase_b_new(src1, &ulpInfo->interval[0], ulpInfo->testMode, vlen);
        gencase_b_new(src2, &ulpInfo->interval[1], ulpInfo->testMode, vlen);

        /* 3.calculate mpfr/mpc result */
        for (int len = 0; len < vlen; len++) {
            cases[0] = GetF32FromBF16(src1[len]);
            cases[1] = GetF32FromBF16(src2[len]);
            mpfr_result_f2(ulpInfo->funcInfo, cases, mpfrRes + len);
        }

        /* 4.call kpl/intel interface and calculate result */
        ((kpl_vmlb2)ulpInfo->funcInfo->kplIntelApiDefault)(vlen, src1, src2, res);

        struct DnnCompareItem *itemInfo = (struct DnnCompareItem *)calloc(1, sizeof(struct DnnCompareItem));
        /* 5.calculate accuracy ulp error */
        for (int number = 0; number < vlen; number++) {
            itemInfo->exp = GetBf16CompareResult(mpfrRes[number].res.f);
            itemInfo->got = GetF32FromBF16(res[number]);
            itemInfo->trh = trh;
            itemInfo->alpha = 0.0f;
            itemInfo->beta = 0.0f;
            itemInfo->dir = ulpInfo->funcInfo->dir;
            itemInfo->alg = ulpInfo->funcInfo->alg;
            itemInfo->error = 0.0f;
            itemInfo->ok = true;
            itemInfo->src = GetF32FromBF16(src1[number]);

            DnnEltwiseCheckError(itemInfo, 1e-5f);
            if(!itemInfo->ok || (ulpInfo->gdbPrint && ulpInfo->quiet)) {
                printf("%s(%a / 0x%04x, %a / 0x%04x)", ulpInfo->funcInfo->ulp_name,
                    GetF32FromBF16(src1[number]), Asuintbf16(src1[number]),
                    GetF32FromBF16(src2[number]), Asuintbf16(src2[number]));
                printf("got %a want %a ", itemInfo->got, itemInfo->exp);
                printf("diff %a ref_diff %a trh %a / %0.10f \n",
                    itemInfo->diff, itemInfo->rel_diff, itemInfo->trh, itemInfo->trh);
            }
            error = fmax(error, itemInfo->error);

            calCaseCnt(&casecnt, itemInfo->error, ulpInfo);
        }
        free(itemInfo);
    }

    /* 6.ulp tool print result */
    printBfloat16(error, ulpInfo, argsNum, &casecnt);

    if (ulpInfo->exeMode) {
        FREE_IF_NOT_NULL_N3(src1, src2, mpfrRes);
    } else {
        FREE_IF_NOT_NULL_N4(src1, src2, res, mpfrRes);
    }
}

void UlpCaseFuncVMLF2(struct UlpInfo *ulpInfo)
{
    double error = 0.0;
    int vlen = ulpInfo->length;
    int argsNum = 2;
    float cases[argsNum];
    struct CaseCnt casecnt = {0, 0, 0, 0};

    /* apply space */
    MALLOC_DECLARE_N3(float, src1, src2, res);
    MALLOC_DECLARE_N1(struct MpfrRes, mpfrRes);
    MALLOC_DIFF_TYPE_N4(src1, src2, res, mpfrRes, float, float, float, struct MpfrRes, ulpInfo->length);

    float trh = ulpInfo->errorBase;
    if (CHECK_ULP(ulpInfo->errorBase)) {
        trh = GetEltwiseThreshold(dnnl_f32, ulpInfo->funcInfo->alg, ulpInfo->funcInfo->dir & FLAG_FWD);
    }
    ulpInfo->errorlim = 0.0;

    for (int times = 0; times < ulpInfo->times; times++) {
        /* 2.generate random cases */
        gencase_f_new(src1, &ulpInfo->interval[0], ulpInfo->testMode, vlen);
        gencase_f_new(src2, &ulpInfo->interval[1], ulpInfo->testMode, vlen);

        /* 3.calculate mpfr/mpc result */
        for (int len = 0; len < vlen; len++) {
            cases[0] = src1[len];
            cases[1] = src2[len];
            mpfr_result_f2(ulpInfo->funcInfo, cases, mpfrRes + len);
        }

        /* 4.call kpl/intel interface and calculate result */
        ((kpl_vmlf2)ulpInfo->funcInfo->kplIntelApiDefault)(vlen, src1, src2, res);

        struct DnnCompareItem *itemInfo = (struct DnnCompareItem *)calloc(1, sizeof(struct DnnCompareItem));
        /* 5.calculate accuracy ulp error */
        for (int number = 0; number < vlen; number++) {
            itemInfo->exp = mpfrRes[number].res.f;
            itemInfo->got = res[number];
            itemInfo->trh = trh;
            itemInfo->alpha = 0.0f;
            itemInfo->beta = 0.0f;
            itemInfo->dir = ulpInfo->funcInfo->dir;
            itemInfo->alg = ulpInfo->funcInfo->alg;
            itemInfo->error = 0.0f;
            itemInfo->ok = true;
            itemInfo->src = src1[number];

            DnnEltwiseCheckError(itemInfo, 1e-5f);
            if(!itemInfo->ok || (ulpInfo->gdbPrint && ulpInfo->quiet)) {
                printf("%s(%a / 0x%08x, %a / 0x%08x)", ulpInfo->funcInfo->ulp_name,
                    src1[number], Asuint(src1[number]), src2[number], Asuint(src2[number]));
                printf("got %a want %a ", itemInfo->got, itemInfo->exp);
                printf("diff %a ref_diff %a trh %a / %0.10f \n",
                    itemInfo->diff, itemInfo->rel_diff, itemInfo->trh, itemInfo->trh);
            }
            error = fmax(error, itemInfo->error);

            calCaseCnt(&casecnt, itemInfo->error, ulpInfo);
        }
        free(itemInfo);
    }

    /* 6.ulp tool print result */
    printFloat(error, ulpInfo, argsNum, &casecnt);

    if (ulpInfo->exeMode) {
        FREE_IF_NOT_NULL_N3(src1, src2, mpfrRes);
    } else {
        FREE_IF_NOT_NULL_N4(src1, src2, res, mpfrRes);
    }
}

void UlpCaseFuncVMLXH2(struct UlpInfo *ulpInfo)
{
    double error = 0.0;
    int64_t vlen = ulpInfo->lengthDnn;
    int argsNum = 3;
    float cases[argsNum];
    struct CaseCnt casecnt = {0, 0, 0, 0};

    /* apply space */
    MALLOC_DECLARE_N3(float, alpha, src1F32, src2F32);
    MALLOC_DECLARE_N3(__fp16, src1F16, src2F16, res);
    MALLOC_DECLARE_N1(struct MpfrRes, mpfrRes);
    MALLOC_DIFF_TYPE_N7(alpha, src1F32, src2F32, src1F16, src2F16, res, mpfrRes,
        float, float, float, __fp16, __fp16, __fp16, struct MpfrRes, ulpInfo->lengthDnn);

    float trh = ulpInfo->errorBase;
    if (CHECK_ULP(ulpInfo->errorBase)) {
        trh = GetEltwiseThreshold(dnnl_f16, ulpInfo->funcInfo->alg, ulpInfo->funcInfo->dir & FLAG_FWD);
    }
    ulpInfo->errorlim = 0.0;

    // bwd not support in-place execution
    for (int times = 0; times < ulpInfo->times; times++) {
        /* 2.generate random cases */
        gencase_f_new(alpha, &ulpInfo->interval[0], ulpInfo->testMode, vlen);
        gencase_f_new(src1F32, &ulpInfo->interval[1], ulpInfo->testMode, vlen);
        gencase_f_new(src2F32, &ulpInfo->interval[2], ulpInfo->testMode, vlen);

        for (int64_t i = 0; i < vlen; i++) {
            src1F16[i] = (__fp16)src1F32[i];
            src2F16[i] = (__fp16)src2F32[i];
        }

        // for alpha loop
        for (int64_t i = 0; i < vlen; i++) {
            /* 3.calculate mpfr/mpc result */
            for (int64_t j = 0; j < vlen; j++) {
                cases[0] = alpha[i];
                cases[1] = src1F16[j];
                cases[2] = src2F16[j];
                mpfr_result_f3(ulpInfo->funcInfo, cases, mpfrRes + j);
            }

            /* 4.call kpl/intel interface and calculate result */
            ((kpl_vmlxh2)ulpInfo->funcInfo->kplIntelApiDefault)(vlen, alpha[i], src1F16, src2F16, res);

            struct DnnCompareItem *itemInfo = (struct DnnCompareItem *)calloc(1, sizeof(struct DnnCompareItem));
            /* 5.calculate accuracy ulp error */
            for (int64_t k = 0; k < vlen; k++) {
                itemInfo->exp = (__fp16)mpfrRes[k].res.f;
                itemInfo->got = res[k];
                itemInfo->trh = trh;
                itemInfo->alpha = alpha[i];
                itemInfo->beta = 0.0f;
                itemInfo->dir = ulpInfo->funcInfo->dir;
                itemInfo->alg = ulpInfo->funcInfo->alg;
                itemInfo->error = 0.0f;
                itemInfo->ok = true;
                itemInfo->src = src1F16[k];

                DnnEltwiseCheckError(itemInfo, 0x1p-14f);
                if(!itemInfo->ok || (ulpInfo->gdbPrint && ulpInfo->quiet)) {
                    printf("%s(%a / 0x%08x, %a / 0x%04x, %a / 0x%04x)", ulpInfo->funcInfo->ulp_name,
                        alpha[i], Asuint(alpha[i]),
                        src1F16[k], Asuint16(src1F16[k]),
                        src2F16[k], Asuint16(src2F16[k]));
                    printf("got %a want %a ", itemInfo->got, itemInfo->exp);
                    printf("diff %a ref_diff %a trh %a / %0.10f \n",
                        itemInfo->diff, itemInfo->rel_diff, itemInfo->trh, itemInfo->trh);
                }
                error = fmax(error, itemInfo->error);

                calCaseCnt(&casecnt, itemInfo->error, ulpInfo);
            }
            free(itemInfo);
        }
    }

    /* 6.ulp tool print result */
    printFloat(error, ulpInfo, argsNum, &casecnt);

    FREE_IF_NOT_NULL_N7(alpha, src1F16, src1F32, src2F16, src2F32, res, mpfrRes);
}

void UlpCaseFuncVMLXB2(struct UlpInfo *ulpInfo)
{
    double error = 0.0;
    int64_t vlen = ulpInfo->lengthDnn;
    int argsNum = 3;
    float cases[argsNum];
    struct CaseCnt casecnt = {0, 0, 0, 0};

    /* apply space */
    MALLOC_DECLARE_N3(float, alpha, src1F32, src2F32);
    MALLOC_DECLARE_N3(__bf16, src1Bf16, src2Bf16, res);
    MALLOC_DECLARE_N1(struct MpfrRes, mpfrRes);
    MALLOC_DIFF_TYPE_N7(alpha, src1F32, src2F32, src1Bf16, src2Bf16, res, mpfrRes,
        float, float, float, __bf16, __bf16, __bf16, struct MpfrRes, ulpInfo->lengthDnn);

    float trh = ulpInfo->errorBase;
    if (CHECK_ULP(ulpInfo->errorBase)) {
        trh = GetEltwiseThreshold(dnnl_bf16, ulpInfo->funcInfo->alg, ulpInfo->funcInfo->dir & FLAG_FWD);
    }
    ulpInfo->errorlim = 0.0;

    // bwd not support in-place execution
    for (int times = 0; times < ulpInfo->times; times++) {
        /* 2.generate random cases */
        gencase_f_new(alpha, &ulpInfo->interval[0], ulpInfo->testMode, vlen);
        gencase_f_new(src1F32, &ulpInfo->interval[1], ulpInfo->testMode, vlen);
        gencase_f_new(src2F32, &ulpInfo->interval[2], ulpInfo->testMode, vlen);

        for (int64_t i = 0; i < vlen; i++) {
            uint32_t u1 = Asuint(src1F32[i]) & 0xffff0000;
            src1Bf16[i] = Asbfloat16((uint16_t)(u1 >> 16));
            uint32_t u2 = Asuint(src2F32[i]) & 0xffff0000;
            src2Bf16[i] = Asbfloat16((uint16_t)(u2 >> 16));
        }

        // for alpha loop
        for (int64_t i = 0; i < vlen; i++) {
            /* 3.calculate mpfr/mpc result */
           for (int64_t j = 0; j < vlen; j++) {
                cases[0] = alpha[i];
                cases[1] = GetF32FromBF16(src1Bf16[j]);
                cases[2] = GetF32FromBF16(src2Bf16[j]);
                mpfr_result_f3(ulpInfo->funcInfo, cases, mpfrRes + j);
            }

            /* 4.call kpl/intel interface and calculate result */
            ((kpl_vmlxb2)ulpInfo->funcInfo->kplIntelApiDefault)(vlen, alpha[i], src1Bf16, src2Bf16, res);

            struct DnnCompareItem *itemInfo = (struct DnnCompareItem *)calloc(1, sizeof(struct DnnCompareItem));
            /* 5.calculate accuracy ulp error */
            for (int64_t j = 0; j < vlen; j++) {
                itemInfo->exp = GetBf16CompareResult(mpfrRes[j].res.f);
                itemInfo->got = GetF32FromBF16(res[j]);
                itemInfo->trh = trh;
                itemInfo->alpha = alpha[i];
                itemInfo->beta = 0.0f;
                itemInfo->dir = ulpInfo->funcInfo->dir;
                itemInfo->alg = ulpInfo->funcInfo->alg;
                itemInfo->error = 0.0f;
                itemInfo->ok = true;
                itemInfo->src = GetF32FromBF16(src1Bf16[j]);

                DnnEltwiseCheckError(itemInfo, 1e-5f);
                if(!itemInfo->ok || (ulpInfo->gdbPrint && ulpInfo->quiet)) {
                    printf("%s(%a / 0x%08x, %a / 0x%04x, %a / 0x%04x)", ulpInfo->funcInfo->ulp_name,
                        alpha[i], Asuint(alpha[i]),
                        GetF32FromBF16(src1Bf16[j]), Asuintbf16(src1Bf16[j]),
                        GetF32FromBF16(src2Bf16[j]), Asuintbf16(src2Bf16[j]));
                    printf("got %a want %a ", itemInfo->got, itemInfo->exp);
                    printf("diff %a ref_diff %a trh %a / %0.10f \n",
                        itemInfo->diff, itemInfo->rel_diff, itemInfo->trh, itemInfo->trh);
                }
                error = fmax(error, itemInfo->error);

                calCaseCnt(&casecnt, itemInfo->error, ulpInfo);
            }
            free(itemInfo);
        }
    }

    /* 6.ulp tool print result */
    printFloat(error, ulpInfo, argsNum, &casecnt);

    if (ulpInfo->exeMode) {
        FREE_IF_NOT_NULL_N6(alpha, src1Bf16, src1F32, src2Bf16, src2F32, mpfrRes);
    } else {
        FREE_IF_NOT_NULL_N7(alpha, src1Bf16, src1F32, src2Bf16, src2F32, res, mpfrRes);
    }
}

void UlpCaseFuncVMLXF2(struct UlpInfo *ulpInfo)
{
    double error = 0.0;
    int64_t vlen = ulpInfo->lengthDnn;
    int argsNum = 3;
    float cases[argsNum];
    struct CaseCnt casecnt = {0, 0, 0, 0};

    /* apply space */
    MALLOC_DECLARE_N4(float, alpha, src, dd, res);
    MALLOC_DECLARE_N1(struct MpfrRes, mpfrRes);
    MALLOC_DIFF_TYPE_N5(alpha, src, dd, res, mpfrRes,
        float, float, float, float, struct MpfrRes, ulpInfo->lengthDnn);

    float trh = ulpInfo->errorBase;
    if (CHECK_ULP(ulpInfo->errorBase)) {
        trh = GetEltwiseThreshold(dnnl_f32, ulpInfo->funcInfo->alg, ulpInfo->funcInfo->dir & FLAG_FWD);
    }
    ulpInfo->errorlim = 0.0;

    // bwd not support in-place execution
    for (int times = 0; times < ulpInfo->times; times++) {
        /* 2.generate random cases */
        gencase_f_new(alpha, &ulpInfo->interval[0], ulpInfo->testMode, vlen);
        gencase_f_new(src, &ulpInfo->interval[1], ulpInfo->testMode, vlen);
        gencase_f_new(dd, &ulpInfo->interval[2], ulpInfo->testMode, vlen);

        // for alpha loop
        for (int64_t i = 0; i < vlen; i++) {
            /* 3.calculate mpfr/mpc result */
            for (int64_t j = 0; j < vlen; j++) {
                cases[0] = alpha[i];
                cases[1] = src[j];
                cases[2] = dd[j];
                mpfr_result_f3(ulpInfo->funcInfo, cases, mpfrRes + j);
            }

            /* 4.call kpl/intel interface and calculate result */
            ((kpl_vmlxf2)ulpInfo->funcInfo->kplIntelApiDefault)(vlen, alpha[i], src, dd, res);

            struct DnnCompareItem *itemInfo = (struct DnnCompareItem *)calloc(1, sizeof(struct DnnCompareItem));
            /* 5.calculate accuracy ulp error */
            for (int64_t j = 0; j < vlen; j++) {
                itemInfo->exp = mpfrRes[j].res.f;
                itemInfo->got = res[j];
                itemInfo->trh = trh;
                itemInfo->alpha = alpha[i];
                itemInfo->beta = 0.0f;
                itemInfo->dir = ulpInfo->funcInfo->dir;
                itemInfo->alg = ulpInfo->funcInfo->alg;
                itemInfo->error = 0.0f;
                itemInfo->ok = true;
                itemInfo->src = src[j];

                DnnEltwiseCheckError(itemInfo, 1e-5f);
                if(!itemInfo->ok || (ulpInfo->gdbPrint && ulpInfo->quiet)) {
                    printf("%s(%a / 0x%08x, %a / 0x%08x, %a / 0x%08x)", ulpInfo->funcInfo->ulp_name,
                        alpha[i], Asuint(alpha[i]), src[j], Asuint(src[j]), dd[j], Asuint(dd[j]));
                    printf("got %a want %a ", itemInfo->got, itemInfo->exp);
                    printf("diff %a ref_diff %a trh %a / %0.10f \n",
                        itemInfo->diff, itemInfo->rel_diff, itemInfo->trh, itemInfo->trh);
                }
                error = fmax(error, itemInfo->error);

                calCaseCnt(&casecnt, itemInfo->error, ulpInfo);
            }
            free(itemInfo);
        }
    }

    /* 6.ulp tool print result */
    printFloat(error, ulpInfo, argsNum, &casecnt);

    if (ulpInfo->exeMode) {
        FREE_IF_NOT_NULL_N4(alpha, src, dd, mpfrRes);
    } else {
        FREE_IF_NOT_NULL_N5(alpha, src, dd, res, mpfrRes);
    }
}

void UlpCaseFuncVMLXYH2(struct UlpInfo *ulpInfo)
{
    double error = 0.0;
    int64_t vlen = ulpInfo->lengthDnn;
    int argsNum = 4;
    float cases[argsNum];
    struct CaseCnt casecnt = {0, 0, 0, 0};

    /* apply space */
    MALLOC_DECLARE_N4(float, alpha, beta, src1F32, src2F32);
    MALLOC_DECLARE_N3(__fp16, src1F16, src2F16, res);
    MALLOC_DECLARE_N1(struct MpfrRes, mpfrRes);
    MALLOC_DIFF_TYPE_N8(alpha, beta, src1F32, src2F32, src1F16, src2F16, res, mpfrRes,
        float, float, float, float, __fp16, __fp16, __fp16, struct MpfrRes, ulpInfo->lengthDnn);

    float trh = ulpInfo->errorBase;
    if (CHECK_ULP(ulpInfo->errorBase)) {
        trh = GetEltwiseThreshold(dnnl_f16, ulpInfo->funcInfo->alg, ulpInfo->funcInfo->dir & FLAG_FWD);
    }
    ulpInfo->errorlim = 0.0;

    // bwd not support in-place execution
    for (int times = 0; times < ulpInfo->times; times++) {
        /* 2.generate random cases */
        gencase_f_new(alpha, &ulpInfo->interval[0], ulpInfo->testMode, vlen);
        gencase_f_new(beta, &ulpInfo->interval[1], ulpInfo->testMode, vlen);
        gencase_f_new(src1F32, &ulpInfo->interval[2], ulpInfo->testMode, vlen);
        gencase_f_new(src2F32, &ulpInfo->interval[3], ulpInfo->testMode, vlen);

        for (int64_t i = 0; i < vlen; i++) {
            src1F16[i] = (__fp16)src1F32[i];
            src2F16[i] = (__fp16)src2F32[i];
        }

        // for alpha beta loop
        for (int64_t i = 0; i < vlen; i++) {
            if ((ulpInfo->funcInfo->alg == CLIP || ulpInfo->funcInfo->alg == CLIP_V2 ||
                ulpInfo->funcInfo->alg == CLIP_V2_DST) && (alpha[i] > beta[i])) {
                    calCaseCnt(&casecnt, -1, ulpInfo);
                    continue;
            }
            /* 3.calculate mpfr/mpc result */
            for (int64_t j = 0; j < vlen; j++) {
                cases[0] = alpha[i];
                cases[1] = beta[i];
                cases[2] = src1F16[j];
                cases[3] = src2F16[j];
                mpfr_result_f4(ulpInfo->funcInfo, cases, mpfrRes + j);
            }

            /* 4.call kpl/intel interface and calculate result */
            ((kpl_vmlxyh2)ulpInfo->funcInfo->kplIntelApiDefault)(vlen, alpha[i], beta[i], src1F16, src2F16, res);

            struct DnnCompareItem *itemInfo = (struct DnnCompareItem *)calloc(1, sizeof(struct DnnCompareItem));
            /* 5.calculate accuracy ulp error */
            for (int64_t j = 0; j < vlen; j++) {
                itemInfo->exp = (__fp16)mpfrRes[j].res.f;
                itemInfo->got = res[j];
                itemInfo->trh = trh;
                itemInfo->alpha = alpha[i];
                itemInfo->beta = beta[i];
                itemInfo->dir = ulpInfo->funcInfo->dir;
                itemInfo->alg = ulpInfo->funcInfo->alg;
                itemInfo->error = 0.0f;
                itemInfo->ok = true;
                itemInfo->src = src1F16[j];

                DnnEltwiseCheckError(itemInfo, 0x1p-14f);
                if(!itemInfo->ok || (ulpInfo->gdbPrint && ulpInfo->quiet)) {
                    printf("%s(%a / 0x%08x, %a / 0x%08x, %a / 0x%04x, %a / 0x%04x)", ulpInfo->funcInfo->ulp_name,
                        alpha[i], Asuint(alpha[i]), beta[i], Asuint(beta[i]),
                        src1F16[j], Asuint16(src1F16[j]), src2F16[j], Asuint16(src2F16[j]));
                    printf("got %a want %a ", itemInfo->got, itemInfo->exp);
                    printf("diff %a ref_diff %a trh %a / %0.10f \n",
                        itemInfo->diff, itemInfo->rel_diff, itemInfo->trh, itemInfo->trh);
                }
                error = fmax(error, itemInfo->error);

                calCaseCnt(&casecnt, itemInfo->error, ulpInfo);
            }
            free(itemInfo);
        }
    }

    /* 6.ulp tool print result */
    printFloat(error, ulpInfo, argsNum, &casecnt);

    FREE_IF_NOT_NULL_N8(alpha, beta, src1F16, src1F32, src2F16, src2F32, res, mpfrRes);
}

void UlpCaseFuncVMLXYB2(struct UlpInfo *ulpInfo)
{
    double error = 0.0;
    int64_t vlen = ulpInfo->lengthDnn;
    int argsNum = 4;
    float cases[argsNum];
    struct CaseCnt casecnt = {0, 0, 0, 0};

    /* apply space */
    MALLOC_DECLARE_N4(float, alpha, beta, src1F32, src2F32);
    MALLOC_DECLARE_N3(__bf16, src1Bf16, src2Bf16, res);
    MALLOC_DECLARE_N1(struct MpfrRes, mpfrRes);
    MALLOC_DIFF_TYPE_N8(alpha, beta, src1F32, src2F32, src1Bf16, src2Bf16, res, mpfrRes,
        float, float, float, float, __bf16, __bf16, __bf16, struct MpfrRes, ulpInfo->lengthDnn);

    float trh = ulpInfo->errorBase;
    if (CHECK_ULP(ulpInfo->errorBase)) {
        trh = GetEltwiseThreshold(dnnl_bf16, ulpInfo->funcInfo->alg, ulpInfo->funcInfo->dir & FLAG_FWD);
    }
    ulpInfo->errorlim = 0.0;

    for (int times = 0; times < ulpInfo->times; times++) {
        /* 2.generate random cases */
        gencase_f_new(alpha, &ulpInfo->interval[0], ulpInfo->testMode, vlen);
        gencase_f_new(beta, &ulpInfo->interval[1], ulpInfo->testMode, vlen);
        gencase_f_new(src1F32, &ulpInfo->interval[2], ulpInfo->testMode, vlen);
        gencase_f_new(src2F32, &ulpInfo->interval[3], ulpInfo->testMode, vlen);

        for (int64_t i = 0; i < vlen; i++) {
            uint32_t u1 = Asuint(src1F32[i]) & 0xffff0000;
            src1Bf16[i] = Asbfloat16((uint16_t)(u1 >> 16));
            uint32_t u2 = Asuint(src2F32[i]) & 0xffff0000;
            src2Bf16[i] = Asbfloat16((uint16_t)(u2 >> 16));
        }

        // for alpha beta loop
        for (int64_t i = 0; i < vlen; i++) {
            if ((ulpInfo->funcInfo->alg == CLIP || ulpInfo->funcInfo->alg == CLIP_V2 ||
                ulpInfo->funcInfo->alg == CLIP_V2_DST) && (alpha[i] > beta[i])) {
                    calCaseCnt(&casecnt, -1, ulpInfo);
                    continue;
            }
            /* 3.calculate mpfr/mpc result */
           for (int64_t j = 0; j < vlen; j++) {
                cases[0] = alpha[i];
                cases[1] = beta[i];
                cases[2] = GetF32FromBF16(src1Bf16[j]);
                cases[3] = GetF32FromBF16(src2Bf16[j]);
                mpfr_result_f4(ulpInfo->funcInfo, cases, mpfrRes + j);
            }

            /* 4.call kpl/intel interface and calculate result */
            ((kpl_vmlxyb2)ulpInfo->funcInfo->kplIntelApiDefault)(vlen, alpha[i], beta[i], src1Bf16, src2Bf16, res);

            struct DnnCompareItem *itemInfo = (struct DnnCompareItem *)calloc(1, sizeof(struct DnnCompareItem));
            /* 5.calculate accuracy ulp error */
            for (int64_t j = 0; j < vlen; j++) {
                itemInfo->exp = GetBf16CompareResult(mpfrRes[j].res.f);
                itemInfo->got = GetF32FromBF16(res[j]);
                itemInfo->trh = trh;
                itemInfo->alpha = alpha[i];
                itemInfo->beta = beta[i];
                itemInfo->dir = ulpInfo->funcInfo->dir;
                itemInfo->alg = ulpInfo->funcInfo->alg;
                itemInfo->error = 0.0f;
                itemInfo->ok = true;
                itemInfo->src = GetF32FromBF16(src1Bf16[j]);

                DnnEltwiseCheckError(itemInfo, 1e-5f);
                if(!itemInfo->ok || (ulpInfo->gdbPrint && ulpInfo->quiet)) {
                    printf("%s(%a / 0x%08x, %a / 0x%08x, %a / 0x%04x, %a / 0x%04x)", ulpInfo->funcInfo->ulp_name,
                        alpha[i], Asuint(alpha[i]), beta[i], Asuint(beta[i]),
                        GetF32FromBF16(src1Bf16[j]), Asuintbf16(src1Bf16[j]),
                        GetF32FromBF16(src2Bf16[j]), Asuintbf16(src2Bf16[j]));
                    printf("got %a want %a ", itemInfo->got, itemInfo->exp);
                    printf("diff %a ref_diff %a trh %a / %0.10f \n",
                        itemInfo->diff, itemInfo->rel_diff, itemInfo->trh, itemInfo->trh);
                }
                error = fmax(error, itemInfo->error);

                calCaseCnt(&casecnt, itemInfo->error, ulpInfo);
            }
            free(itemInfo);
        }
    }

    /* 6.ulp tool print result */
    printFloat(error, ulpInfo, argsNum, &casecnt);

    FREE_IF_NOT_NULL_N8(alpha, beta, src1Bf16, src1F32, src2Bf16, src2F32, res, mpfrRes);
}

void UlpCaseFuncVMLXYF2(struct UlpInfo *ulpInfo)
{
    double error = 0.0;
    int64_t vlen = ulpInfo->lengthDnn;
    int argsNum = 4;
    float cases[argsNum];
    struct CaseCnt casecnt = {0, 0, 0, 0};

    /* apply space */
    MALLOC_DECLARE_N5(float, alpha, beta, src, dd, res);
    MALLOC_DECLARE_N1(struct MpfrRes, mpfrRes);
    MALLOC_DIFF_TYPE_N6(alpha, beta, src, dd, res, mpfrRes,
        float, float, float, float, float, struct MpfrRes, ulpInfo->lengthDnn);

    float trh = ulpInfo->errorBase;
    if (CHECK_ULP(ulpInfo->errorBase)) {
        trh = GetEltwiseThreshold(dnnl_f32, ulpInfo->funcInfo->alg, ulpInfo->funcInfo->dir & FLAG_FWD);
    }
    ulpInfo->errorlim = 0.0;

    for (int times = 0; times < ulpInfo->times; times++) {
        /* 2.generate random cases */
        gencase_f_new(alpha, &ulpInfo->interval[0], ulpInfo->testMode, vlen);
        gencase_f_new(beta, &ulpInfo->interval[1], ulpInfo->testMode, vlen);
        gencase_f_new(src, &ulpInfo->interval[2], ulpInfo->testMode, vlen);
        gencase_f_new(dd, &ulpInfo->interval[3], ulpInfo->testMode, vlen);

        // for alpha beta loop
        for (int64_t i = 0; i < vlen; i++) {
            if ((ulpInfo->funcInfo->alg == CLIP || ulpInfo->funcInfo->alg == CLIP_V2 ||
                ulpInfo->funcInfo->alg == CLIP_V2_DST) && (alpha[i] > beta[i])) {
                    calCaseCnt(&casecnt, -1, ulpInfo);
                    continue;
            }
            /* 3.calculate mpfr/mpc result */
            for (int64_t j = 0; j < vlen; j++) {
                cases[0] = alpha[i];
                cases[1] = beta[i];
                cases[2] = src[j];
                cases[3] = dd[j];
                mpfr_result_f4(ulpInfo->funcInfo, cases, mpfrRes + j);
            }

            /* 4.call kpl/intel interface and calculate result */
            ((kpl_vmlxyf2)ulpInfo->funcInfo->kplIntelApiDefault)(vlen, alpha[i], beta[i], src, dd, res);

            struct DnnCompareItem *itemInfo = (struct DnnCompareItem *)calloc(1, sizeof(struct DnnCompareItem));
            /* 5.calculate accuracy ulp error */
            for (int64_t j = 0; j < vlen; j++) {
                itemInfo->exp = mpfrRes[j].res.f;
                itemInfo->got = res[j];
                itemInfo->trh = trh;
                itemInfo->alpha = alpha[i];
                itemInfo->beta = beta[i];
                itemInfo->dir = ulpInfo->funcInfo->dir;
                itemInfo->alg = ulpInfo->funcInfo->alg;
                itemInfo->error = 0.0f;
                itemInfo->ok = true;
                itemInfo->src = src[j];

                DnnEltwiseCheckError(itemInfo, 1e-5f);
                if(!itemInfo->ok || (ulpInfo->gdbPrint && ulpInfo->quiet)) {
                    printf("%s(%a / 0x%08x, %a / 0x%08x, %a / 0x%08x, %a / 0x%08x)", ulpInfo->funcInfo->ulp_name,
                        alpha[i], Asuint(alpha[i]), beta[i], Asuint(beta[i]),
                        src[j], Asuint(src[j]), dd[j], Asuint(dd[j]));
                    printf("got %a want %a ", itemInfo->got, itemInfo->exp);
                    printf("diff %a ref_diff %a trh %a / %0.10f \n",
                        itemInfo->diff, itemInfo->rel_diff, itemInfo->trh, itemInfo->trh);
                }
                error = fmax(error, itemInfo->error);

                calCaseCnt(&casecnt, itemInfo->error, ulpInfo);
            }
            free(itemInfo);
        }
    }

    /* 6.ulp tool print result */
    printFloat(error, ulpInfo, argsNum, &casecnt);

    FREE_IF_NOT_NULL_N6(alpha, beta, src, dd, res, mpfrRes);
}