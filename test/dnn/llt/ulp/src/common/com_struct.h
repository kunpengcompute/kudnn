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

#ifndef COM_STRUCT_H
#define COM_STRUCT_H

#undef _GNU_SOURCE
#define _GNU_SOURCE 1
#include <time.h>
#include <stdlib.h>
#include <stdbool.h>
#include <stdio.h>
#include <fcntl.h>
#include <unistd.h>
#include <math.h>
#include <ctype.h>
#include <string.h>
#include <arm_sve.h>
#include "kpl_mpfr.h"
#include "random.h"
#include "com_macro_n.h"

typedef void (*mathFunc_t)(void);

#define FLT_MIN 1.17549435e-38f // 0x1p-126 float
#define MAX_ARGS_NUM 6 // current function has max 6 arguments

#define FLOAT_VECTOR_LEN 255
#define DOUBLE_VECTOR_LEN 127

#define MAX_OP_NUM 32
#define MAX_SPLIT_SIZE 80

#define VML_ENUM_BASE 0
#define LIBM_ENUM_BASE 64
#define SVML_ENUM_BASE 128
#define VMLF16_ENUM_BASE 192
#define BUILTIN_ENUM_BASE 256

typedef enum {
    vmlh, // void func(const int64_t len, const __fp16 *src, __fp16 *dst);
    vmlh2, // void func(const int64_t len, const __fp16 *src, const __fp16 *dd, __fp16 *ds);
    vmlhxa, // void func(const int len, const __fp16 *src, const __fp16 y, const __fp16 alpha, __fp16 *dst);
    vmlxh, // void func(const int len, const float alpha, const __fp16 *src, __fp16 *dst);
    vmlxh2, // void func(const int len, const float alpha, const __fp16 *src, const __fp16 *dd, __fp16 *ds);
    vmlxyh, // void func(const int len, const float alpha, const float beta, const __fp16 *src, __fp16 *dst);
    vmlxyh2, // void func(const int len, const float alpha, const float beta, const __fp16 *src, const __fp16 *dd, __fp16 *ds);

    vmlb, // void func(const int64_t len, const __bf16 *src, __bf16 *dst);
    vmlb2, // void func(const int64_t len, const __bf16 *src, const __bf16 *dd, __bf16 *ds);
    vmlxb, // void func(const int len, const float alpha, const __bf16 *src, __bf16 *dst);
    vmlxb2, // void func(const int len, const float alpha, const __bf16 *src, const __bf16 *dd, __bf16 *ds);
    vmlxyb, // void func(const int len, const float alpha, const float beta, const __bf16 *src, __bf16 *dst);
    vmlxyb2, // void func(const int len, const float alpha, const float beta, const __bf16 *src, const __bf16 *dd, __bf16 *ds);

    vmlf, // void func(const int64_t len, const float *src, float *dst);
    vmlf2, // void func(const int64_t len, const float *src, const float *dd, float *ds);
    vmlfxa, // void func(const int len, const float *src, const float y, const float alpha, float *dst);
    vmlxf, // void func(const int len, const float alpha, const float *src, float *dst);
    vmlxf2, // void func(const int len, const float alpha, const float *src, const float *dd, float *ds);
    vmlxyf, // void func(const int len, const float alpha, const float beta, const float *src, float *dst);
    vmlxyf2, // void func(const int len, const float alpha, const float beta, const float *src, const float *dd, float *ds);

} FUNC_TYPE;

typedef enum TEST_MODE {
    TMODE_RANDOM,
    TMODE_LINEAR,
    TMODE_UNKNOW,
} TEST_MODE;

typedef enum EXECUTE_MODE {
    OUT_OF_PLACE,
    IN_PLACE,
} EXECUTE_MODE;

typedef enum ELTWISE_KIND_T {
    // sum
    SUM,
    // depthwise convolution
    DW,
    // eltwise
    ELTWISE_START, // a guard to check kind is eltwise
    ABS,
    CLIP,
    CLIP_V2,
    CLIP_V2_DST,
    ELU,
    ELU_DST,
    EXP,
    EXP_DST,
    GELU_ERF,
    GELU_TANH,
    HARDSIGMOID,
    HARDSWISH,
    LINEAR,
    LOG,
    LOGISTIC,
    LOGISTIC_DST,
    MISH,
    POW,
    RELU,
    RELU_DST,
    ROUND,
    SQRT,
    SQRT_DST,
    SQUARE,
    SRELU,
    SWISH,
    TANH,
    TANH_DST,
    ELTWISE_END, // a guard to check kind is eltwise
    // binary
    BINARY_START, // a guard to check kind is binary
    ADD,
    DIV,
    EQ,
    GE,
    GT,
    LE,
    LT,
    MAX,
    MIN,
    MUL,
    NE,
    SUB,
    BINARY_END, // a guard to check kind is binary
    // prelu
    PRELU,
    // guard entry
    KIND_TOTAL
} ELTWISE_KIND_T;

typedef enum ELTWISE_DIR_T {
    DIR_UNDEF = 0,
    FLAG_DAT = 1,
    FLAG_WEI = 2,
    FLAG_BIA = 4,
    FLAG_FWD = 32,
    FLAG_BWD = 64,
    FLAG_INF = 128,
    FWD_D = FLAG_FWD + FLAG_DAT,
    FWD_I = FLAG_FWD + FLAG_DAT + FLAG_INF,
    FWD_B = FLAG_FWD + FLAG_DAT + FLAG_BIA,
    BWD_D = FLAG_BWD + FLAG_DAT,
    BWD_DW = FLAG_BWD + FLAG_DAT + FLAG_WEI,
    BWD_W = FLAG_BWD + FLAG_WEI,
    BWD_WB = FLAG_BWD + FLAG_WEI + FLAG_BIA,
} ELTWISE_DIR_T;

/// Data type specification
typedef enum DNNL_DATA_TYPE_T{
    /// Undefined data type, used for empty memory descriptors.
    dnnl_data_type_undef = 0,
    /// 16-bit/half-precision floating point.
    dnnl_f16 = 1,
    /// non-standard 16-bit (bfloat16 w/ 7 bit mantissa) floating point.
    dnnl_bf16 = 2,
    /// 32-bit/single-precision floating point.
    dnnl_f32 = 3,
    /// 32-bit signed integer.
    dnnl_s32 = 4,
    /// 8-bit signed integer.
    dnnl_s8 = 5,
    /// 8-bit unsigned integer.
    dnnl_u8 = 6,
    /// 64-bit/double-precision floating point.
    dnnl_f64 = 7,
    /// Boolean data type. Size is C++ implementation defined.
    dnnl_boolean = 8,
    /// [OFP8 standard 8-bit floating-point](https://www.opencompute.org/documents/ocp-8-bit-floating-point-specification-ofp8-revision-1-0-2023-06-20-pdf)
    /// with a 5-bit exponent and a 2-bit mantissa.
    dnnl_f8_e5m2 = 9,
    /// [OFP8 standard 8-bit floating-point](https://www.opencompute.org/documents/ocp-8-bit-floating-point-specification-ofp8-revision-1-0-2023-06-20-pdf)
    /// with a 4-bit exponent and a 3-bit mantissa.
    dnnl_f8_e4m3 = 10,

    /// Parameter to allow internal only data_types without undefined behavior.
    /// This parameter is chosen to be valid for so long as sizeof(int) >= 2.
    dnnl_data_type_max = 0x7fff,
} DNNL_DATA_TYPE_T;

struct DnnCompareItem {
    float exp;
    float got;
    float diff;
    float rel_diff;
    float trh;
    float src;
    float alpha;
    float beta;
    float error;
    bool ok;
    ELTWISE_KIND_T alg;
    ELTWISE_DIR_T dir;
};

struct Interval {
    uint64_t low;
    uint64_t high;
};

struct DefaultInterval {
    double low;
    double high;
};

typedef union type_t {
    __fp16 h;
    float f;
    double d;
    long double ld;
} argsType;

struct FunctionInfo {
    const char *name;
    const char *ulp_name;
    const char *mathbench_name;
    mathFunc_t mpfrApi;
    mathFunc_t kplIntelApiDefault;  // vml: non-mode function; svml: non-pg function
    mathFunc_t kplIntelApi; // vml: mode function; svml: pg function
    FUNC_TYPE funcType;
    ELTWISE_KIND_T alg;
    ELTWISE_DIR_T dir;
    double maxerr; // default ulp error threshold
    struct DefaultInterval interval[6];
};
extern const int nfunctions;
extern const struct FunctionInfo g_functions[];

/* generte testcases */
struct CaseInfo {
    int randomMode; // 0: random; 1:linear
    uint64_t stepLen[6]; // linear
};

/* mathtest tool struct */
struct MpfrRes {
    int ulpexp;
    union {
        __bf16 bf16;
        __fp16 h;
        uint16_t u16;
        float f;
        uint32_t u32;
        uint64_t u64;
        double d;
        long int li;
        long long int lli;
    } res;
    double resTail;
    double resTailLong; /* support long double. */

    uint32_t rtestTail;
};

/* ulp tool struct */
struct UlpInfo {
    uint64_t seed; // seed value
    int round; // rounding mode
    int modeFlag; // precision version flag
    int mode; // precision version
    int length; // vector length
    int64_t lengthDnn; // vector length with int64
    double errorlim; // ulp error thrshold
    double errorBase; // error gap for absolute precision check
    struct Interval interval[6]; // interval array
    int times; // test times
    int gdbPrint; // 0:default; 1:print
    int quiet; // 0:default; 1:debug mode
    const struct FunctionInfo *funcInfo;
    TEST_MODE testMode;
    EXECUTE_MODE exeMode; // 0: out-of-place; 1: in-place
};

typedef enum PERF_MODE {
    PMODE_LATENCY,
    PMODE_THROUGHPUT,
    PMODE_BOTH,
    PMODE_UNKNOW,
} PERF_MODE;

typedef void (*UlpCaseFuncPtr)(struct UlpInfo *ulpInfo);
typedef void (*UlpParseFuncPtr)(int argc, char **argv, struct UlpInfo *ulpInfo);

#define KPL_LA 0x1
#define KPL_HA 0x2
#define KPL_EP 0x3

static inline int get_math_mode(char *argv[], void (*usage)(void))
{
    int mode = KPL_LA;
    if (strcmp(argv[0], "la") == 0) {
        mode = KPL_LA;
    } else if (strcmp(argv[0], "ha") == 0) {
        mode = KPL_HA;
    } else if (strcmp(argv[0], "ep") == 0) {
        mode = KPL_EP;
    } else {
        usage();
    }
    return mode;
}

#ifndef FE_TONEAREST
#define FE_TONEAREST  0x000000
#endif

typedef int (*mpfr_vmlh1)(mpfr_t, const mpfr_t, mpfr_rnd_t);
typedef int (*mpfr_vmlf1)(mpfr_t, const mpfr_t, mpfr_rnd_t);
typedef int (*mpfr_vmlf2)(mpfr_t, const mpfr_t, const mpfr_t, mpfr_rnd_t);
typedef int (*mpfr_vmlf3)(mpfr_t, const mpfr_t, const mpfr_t, const mpfr_t, mpfr_rnd_t);
typedef int (*mpfr_vmlf4)(mpfr_t, const mpfr_t, const mpfr_t, const mpfr_t, const mpfr_t, mpfr_rnd_t);

int mpfr_result_h(const struct FunctionInfo *funcInfo, __fp16 *cases, struct MpfrRes* mpfrRes);
int mpfr_result_h2(const struct FunctionInfo *funcInfo, __fp16 *cases, struct MpfrRes* mpfrRes);
int mpfr_result_f(const struct FunctionInfo *funcInfo, float *cases, struct MpfrRes* mpfrRes);
int mpfr_result_fh(const struct FunctionInfo *funcInfo, float *cases, struct MpfrRes* mpfrRes);
int mpfr_result_f2(const struct FunctionInfo *funcInfo, float *cases, struct MpfrRes* mpfrRes);
int mpfr_result_hf2(const struct FunctionInfo *funcInfo, float *cases, struct MpfrRes* mpfrRes);
int mpfr_result_f3(const struct FunctionInfo *funcInfo, float *cases, struct MpfrRes* mpfrRes);
int mpfr_result_f4(const struct FunctionInfo *funcInfo, float *cases, struct MpfrRes* mpfrRes);

typedef void (*kpl_mode_vmlh2)(const int, const __fp16 *, const __fp16 *, __fp16 *, long long);

typedef void (*kpl_vmlf1)(const int64_t, const float *, float *);
typedef void (*kpl_vmlf2)(const int64_t, const float *, const float *, float *);
typedef void (*kpl_vmlfxa)(const int64_t, const float *, const float, const float, float *);
typedef void (*kpl_vmlxf)(const int64_t, const float, const float *, float *);
typedef void (*kpl_vmlxf2)(const int64_t, const float, const float *, const float *, float *);
typedef void (*kpl_vmlxyf)(const int64_t, const float, const float, const float *, float *);
typedef void (*kpl_vmlxyf2)(const int64_t, const float, const float, const float *, const float *, float *);

typedef void (*kpl_vmlhxa)(const int64_t, const __fp16 *, const float, const float, __fp16 *);
typedef void (*kpl_vmlxh)(const int64_t, const float, const __fp16 *, __fp16 *);
typedef void (*kpl_vmlxh2)(const int64_t, const float, const __fp16 *, const __fp16 *, __fp16 *);
typedef void (*kpl_vmlh)(const int64_t, const __fp16 *, __fp16 *);
typedef void (*kpl_vmlh2)(const int64_t, const __fp16 *, const __fp16 *, __fp16 *);
typedef void (*kpl_vmlxyh)(const int64_t, const float, const float, const __fp16 *, __fp16 *);
typedef void (*kpl_vmlxyh2)(const int64_t, const float, const float, const __fp16 *, const __fp16 *, __fp16 *);

typedef void (*kpl_vmlb)(const int64_t, const __bf16 *, __bf16 *);
typedef void (*kpl_vmlb2)(const int64_t, const __bf16 *, const __bf16 *, __bf16 *);
typedef void (*kpl_vmlxb)(const int64_t, const float, const __bf16 *, __bf16 *);
typedef void (*kpl_vmlxb2)(const int64_t, const float, const __bf16 *, const __bf16 *, __bf16 *);
typedef void (*kpl_vmlxyb)(const int64_t, const float, const float, const __bf16 *, __bf16 *);
typedef void (*kpl_vmlxyb2)(const int64_t, const float, const float, const __bf16 *, const __bf16 *, __bf16 *);

#define prec_mpfr_h (80)
#define prec_mpfr_f (50)
#define prec_mpfr_d (80)
#define prec_h (11)
#define prec_f (24)
#define prec_d (53)
#define emin_h (-23)
#define emin_f (-148)
#define emin_d (-1073)
#define emax_h (16)
#define emax_f (128)
#define emax_d (1024)

uint64_t GetRandomU64();

uint64_t getnum_h(const char *str);
uint64_t getnum_b(const char *str);
uint64_t getnum_f(const char *str);

int mpfr_result_h(const struct FunctionInfo *funcInfo, __fp16 *cases, struct MpfrRes* mpfrRes);
int mpfr_result_f(const struct FunctionInfo *funcInfo, float *cases, struct MpfrRes* mpfrRes);

void gencase_f_new(float *arr, struct Interval *interval, TEST_MODE testMode, int64_t length);
void gencase_h_new(__fp16 *arr, struct Interval *interval, TEST_MODE testMode, int64_t length);
void gencase_b_new(__bf16 *arr, struct Interval *interval, TEST_MODE testMode, int64_t length);

int get_exp_h(mpfr_t x);
int get_exp_f(mpfr_t x);

double ulperr_h(__fp16 got, struct MpfrRes *mpfrRes);
double ulperr_f(float got, struct MpfrRes *mpfrRes);
double dnn_ulperr_d(double res, double mpfrRes, double base, double error);

void MemoryInit(void *dst, int value, int length);

#endif
