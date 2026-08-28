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

#ifndef COM_MACRO_N_H
#define COM_MACRO_N_H

#include <stdlib.h>
#include <malloc.h>
#define MALLOC_DECLARE_N1(type, x) type *(x)
#define MALLOC_DECLARE_N2(type, x, y) type *(x), *(y)
#define MALLOC_DECLARE_N3(type, x, y, z) type *(x), *(y), *(z)
#define MALLOC_DECLARE_N4(type, x, y, z, a) type *(x), *(y), *(z), *(a)
#define MALLOC_DECLARE_N5(type, x, y, z, a, b) type *(x), *(y), *(z), *(a), *(b)

#define MALLOC_DIFF_TYPE_N1(x, type, length) do { \
    x = (type *)memalign(64, (length) * sizeof(type)); \
    if (!(x)) { \
        fprintf(stderr, "Out of Memory!!\n"); \
        exit(1); \
    } \
} while (0)
#define MALLOC_DIFF_TYPE_N2(x, y, type1, type2, length) do { \
    MALLOC_DIFF_TYPE_N1(x, type1, length); \
    y = (type2 *)memalign(64, (length) * sizeof(type2)); \
    if (!(y)) { \
        free(x); \
        fprintf(stderr, "Out of Memory!!\n"); \
        exit(1); \
    } \
} while (0)
#define MALLOC_DIFF_TYPE_N3(x, y, z, type1, type2, type3, length) do { \
    MALLOC_DIFF_TYPE_N2(x, y, type1, type2, length); \
    z = (type3 *)memalign(64, (length) * sizeof(type3)); \
    if (!(z)) { \
        free(x); \
        free(y); \
        fprintf(stderr, "Out of Memory!!\n"); \
        exit(1); \
    } \
} while (0)
#define MALLOC_DIFF_TYPE_N4(x, y, z, a, type1, type2, type3, type4, length) do { \
    MALLOC_DIFF_TYPE_N3(x, y, z, type1, type2, type3, length); \
    a = (type4 *)memalign(64, (length) * sizeof(type4)); \
    if (!(a)) { \
        free(x); \
        free(y); \
        free(z); \
        fprintf(stderr, "Out of Memory!!\n"); \
        exit(1); \
    } \
} while (0)
#define MALLOC_DIFF_TYPE_N5(x, y, z, a, b, type1, type2, type3, type4, type5, length) do { \
    MALLOC_DIFF_TYPE_N4(x, y, z, a, type1, type2, type3, type4, length); \
    b = (type5 *)memalign(64, (length) * sizeof(type5)); \
    if (!(b)) { \
        free(x); \
        free(y); \
        free(z); \
        free(a); \
        fprintf(stderr, "Out of Memory!!\n"); \
        exit(1); \
    } \
} while (0)
#define MALLOC_DIFF_TYPE_N6(x, y, z, a, b, c, type1, type2, type3, type4, type5, type6, length) do { \
    MALLOC_DIFF_TYPE_N5(x, y, z, a, b, type1, type2, type3, type4, type5, length); \
    c = (type6 *)memalign(64, (length) * sizeof(type6)); \
    if (!(c)) { \
        free(x); \
        free(y); \
        free(z); \
        free(a); \
        free(b); \
        fprintf(stderr, "Out of Memory!!\n"); \
        exit(1); \
    } \
} while (0)
#define MALLOC_DIFF_TYPE_N7(x1, x2, x3, x4, x5, x6, x7, type1, type2, type3, type4, type5, type6, type7, length) do { \
    MALLOC_DIFF_TYPE_N6(x1, x2, x3, x4, x5, x6, type1, type2, type3, type4, type5, type6, length); \
    x7 = (type7 *)memalign(64, (length) * sizeof(type7)); \
    if (!(x7)) { \
        free(x1); \
        free(x2); \
        free(x3); \
        free(x4); \
        free(x5); \
        free(x6); \
        fprintf(stderr, "Out of Memory!!\n"); \
        exit(1); \
    } \
} while (0)
#define MALLOC_DIFF_TYPE_N8(x1, x2, x3, x4, x5, x6, x7, x8, type1, type2, type3, type4, type5, type6, type7, type8, length) do { \
    MALLOC_DIFF_TYPE_N7(x1, x2, x3, x4, x5, x6, x7, type1, type2, type3, type4, type5, type6, type7, length); \
    x8 = (type8 *)memalign(64, (length) * sizeof(type8)); \
    if (!(x8)) { \
        free(x1); \
        free(x2); \
        free(x3); \
        free(x4); \
        free(x5); \
        free(x6); \
        free(x7); \
        fprintf(stderr, "Out of Memory!!\n"); \
        exit(1); \
    } \
} while (0)

#define FREE_IF_NOT_NULL_N1(x) do { \
    if ((x) != NULL) { \
        free(x); \
    } \
} while (0)
#define FREE_IF_NOT_NULL_N2(x, y) do { \
    FREE_IF_NOT_NULL_N1(x); \
    FREE_IF_NOT_NULL_N1(y); \
} while (0)
#define FREE_IF_NOT_NULL_N3(x, y, z) do { \
    FREE_IF_NOT_NULL_N2(x, y); \
    FREE_IF_NOT_NULL_N1(z); \
} while (0)
#define FREE_IF_NOT_NULL_N4(x, y, z, a) do { \
    FREE_IF_NOT_NULL_N3(x, y, z); \
    FREE_IF_NOT_NULL_N1(a); \
} while (0)
#define FREE_IF_NOT_NULL_N5(x, y, z, a, b) do { \
    FREE_IF_NOT_NULL_N4(x, y, z, a); \
    FREE_IF_NOT_NULL_N1(b); \
} while (0)
#define FREE_IF_NOT_NULL_N6(x, y, z, a, b, c) do { \
    FREE_IF_NOT_NULL_N5(x, y, z, a, b); \
    FREE_IF_NOT_NULL_N1(c); \
} while (0)
#define FREE_IF_NOT_NULL_N7(x1, x2, x3, x4, x5, x6, x7) do { \
    FREE_IF_NOT_NULL_N6(x1, x2, x3, x4, x5, x6); \
    FREE_IF_NOT_NULL_N1(x7); \
} while (0)
#define FREE_IF_NOT_NULL_N8(x1, x2, x3, x4, x5, x6, x7, x8) do { \
    FREE_IF_NOT_NULL_N7(x1, x2, x3, x4, x5, x6, x7); \
    FREE_IF_NOT_NULL_N1(x8); \
} while (0)

#define CHECK_ULP(x) ((x) == 0.0)
#endif
