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

#ifndef KUDNN_COMMON_MACRO_HPP
#define KUDNN_COMMON_MACRO_HPP

#define COMPILE_BARRIER() __asm__ volatile("" ::: "memory")
#define EXTEND(x, n) x##n

#ifndef VML_VC_UNFOLD_N
#define VML_VC_UNFOLD_N 8
#endif

#define VML_VC_DECLARE_N1(type, x) type EXTEND(x, 1)
#define VML_VC_DECLARE_N2(type, x) VML_VC_DECLARE_N1(type, x), EXTEND(x, 2)
#define VML_VC_DECLARE_N3(type, x) VML_VC_DECLARE_N2(type, x), EXTEND(x, 3)
#define VML_VC_DECLARE_N4(type, x) VML_VC_DECLARE_N3(type, x), EXTEND(x, 4)
#define VML_VC_DECLARE_N5(type, x) VML_VC_DECLARE_N4(type, x), EXTEND(x, 5)
#define VML_VC_DECLARE_N6(type, x) VML_VC_DECLARE_N5(type, x), EXTEND(x, 6)
#define VML_VC_DECLARE_N7(type, x) VML_VC_DECLARE_N6(type, x), EXTEND(x, 7)
#define VML_VC_DECLARE_N8(type, x) VML_VC_DECLARE_N7(type, x), EXTEND(x, 8)
#define VML_VC_DECLARE_HELPER(type, x, n) VML_VC_DECLARE_N##n(type, x)
#define VML_VC_DECLARE_HELPER2(type, x, n) VML_VC_DECLARE_HELPER(type, x, n)
#define VML_VC_DECLARE(type, x) VML_VC_DECLARE_HELPER2(type, x, VML_VC_UNFOLD_N)

// flexible N loop unrolling sve_load function
#define VML_VC_LD_STEP_N1(x, func, pg, src, step) EXTEND(x, 1) = func(pg, (src) + (step) * 0)
#define VML_VC_LD_STEP_N2(x, func, pg, src, step) do { \
    VML_VC_LD_STEP_N1(x, func, pg, src, step); \
    EXTEND(x, 2) = func(pg, (src) + (step)); \
} while (0)
#define VML_VC_LD_STEP_N3(x, func, pg, src, step) do { \
    VML_VC_LD_STEP_N2(x, func, pg, src, step); \
    EXTEND(x, 3) = func(pg, (src) + (step) * 2); \
} while (0)
#define VML_VC_LD_STEP_N4(x, func, pg, src, step) do { \
    VML_VC_LD_STEP_N3(x, func, pg, src, step); \
    EXTEND(x, 4) = func(pg, (src) + (step) * 3); \
} while (0)
#define VML_VC_LD_STEP_N5(x, func, pg, src, step) do { \
    VML_VC_LD_STEP_N4(x, func, pg, src, step); \
    EXTEND(x, 5) = func(pg, (src) + (step) * 4); \
} while (0)
#define VML_VC_LD_STEP_N6(x, func, pg, src, step) do { \
    VML_VC_LD_STEP_N5(x, func, pg, src, step); \
    EXTEND(x, 6) = func(pg, (src) + (step) * 5); \
} while (0)
#define VML_VC_LD_STEP_N7(x, func, pg, src, step) do { \
    VML_VC_LD_STEP_N6(x, func, pg, src, step); \
    EXTEND(x, 7) = func(pg, (src) + (step) * 6); \
} while (0)
#define VML_VC_LD_STEP_N8(x, func, pg, src, step) do { \
    VML_VC_LD_STEP_N7(x, func, pg, src, step); \
    EXTEND(x, 8) = func(pg, (src) + (step) * 7); \
} while (0)
#define VML_VC_LD_STEP_HELPER(x, func, pg, src, step, n) \
    VML_VC_LD_STEP_N##n(x, func, pg, src, step)
#define VML_VC_LD_STEP_HELPER2(x, func, pg, src, step, n) \
    VML_VC_LD_STEP_HELPER(x, func, pg, src, step, n)
#define VML_VC_LD_STEP(x, func, pg, src, step) \
    VML_VC_LD_STEP_HELPER2(x, func, pg, src, step, VML_VC_UNFOLD_N)


// flexible N loop unrolling sve 3 parameter function with predict register(second and third paramter is const)
#define VML_VC_FUNC_PG_N23_3PARA_N1(dst, func, pg, src1, src2, src3) \
    EXTEND(dst, 1) = func(pg, EXTEND(src1, 1), src2, src3)
#define VML_VC_FUNC_PG_N23_3PARA_N2(dst, func, pg, src1, src2, src3) do { \
    VML_VC_FUNC_PG_N23_3PARA_N1(dst, func, pg, src1, src2, src3); \
    EXTEND(dst, 2) = func(pg, EXTEND(src1, 2), src2, src3); \
}while (0)
#define VML_VC_FUNC_PG_N23_3PARA_N3(dst, func, pg, src1, src2, src3) do { \
    VML_VC_FUNC_PG_N23_3PARA_N2(dst, func, pg, src1, src2, src3); \
    EXTEND(dst, 3) = func(pg, EXTEND(src1, 3), src2, src3); \
}while (0)
#define VML_VC_FUNC_PG_N23_3PARA_N4(dst, func, pg, src1, src2, src3) do { \
    VML_VC_FUNC_PG_N23_3PARA_N3(dst, func, pg, src1, src2, src3); \
    EXTEND(dst, 4) = func(pg, EXTEND(src1, 4), src2, src3); \
}while (0)
#define VML_VC_FUNC_PG_N23_3PARA_N5(dst, func, pg, src1, src2, src3) do { \
    VML_VC_FUNC_PG_N23_3PARA_N4(dst, func, pg, src1, src2, src3); \
    EXTEND(dst, 5) = func(pg, EXTEND(src1, 5), src2, src3); \
}while (0)
#define VML_VC_FUNC_PG_N23_3PARA_N6(dst, func, pg, src1, src2, src3) do { \
    VML_VC_FUNC_PG_N23_3PARA_N5(dst, func, pg, src1, src2, src3); \
    EXTEND(dst, 6) = func(pg, EXTEND(src1, 6), src2, src3); \
}while (0)
#define VML_VC_FUNC_PG_N23_3PARA_N7(dst, func, pg, src1, src2, src3) do { \
    VML_VC_FUNC_PG_N23_3PARA_N6(dst, func, pg, src1, src2, src3); \
    EXTEND(dst, 7) = func(pg, EXTEND(src1, 7), src2, src3); \
}while (0)
#define VML_VC_FUNC_PG_N23_3PARA_N8(dst, func, pg, src1, src2, src3) do { \
    VML_VC_FUNC_PG_N23_3PARA_N7(dst, func, pg, src1, src2, src3); \
    EXTEND(dst, 8) = func(pg, EXTEND(src1, 8), src2, src3); \
}while (0)
#define VML_VC_FUNC_PG_N23_3PARA_HELPER(dst, func, pg, src1, src2, src3, n) \
    VML_VC_FUNC_PG_N23_3PARA_N##n(dst, func, pg, src1, src2, src3)
#define VML_VC_FUNC_PG_N23_3PARA_HELPER2(dst, func, pg, src1, src2, src3, n) \
    VML_VC_FUNC_PG_N23_3PARA_HELPER(dst, func, pg, src1, src2, src3, n)
#define VML_VC_FUNC_PG_N23_3PARA(dst, func, pg, src1, src2, src3) do { \
    COMPILE_BARRIER(); \
    VML_VC_FUNC_PG_N23_3PARA_HELPER2(dst, func, pg, src1, src2, src3, VML_VC_UNFOLD_N); \
    COMPILE_BARRIER(); \
} while (0)


// flexible N loop unrolling sve 2 parameter function with predict register(second paramter is const)
#define VML_VC_FUNC_PG_N2_2PARA_N1(dst, func, pg, src1, src2) \
    EXTEND(dst, 1) = func(pg, EXTEND(src1, 1), src2)
#define VML_VC_FUNC_PG_N2_2PARA_N2(dst, func, pg, src1, src2) do { \
    VML_VC_FUNC_PG_N2_2PARA_N1(dst, func, pg, src1, src2); \
    EXTEND(dst, 2) = func(pg, EXTEND(src1, 2), src2); \
} while (0)
#define VML_VC_FUNC_PG_N2_2PARA_N3(dst, func, pg, src1, src2) do { \
    VML_VC_FUNC_PG_N2_2PARA_N2(dst, func, pg, src1, src2); \
    EXTEND(dst, 3) = func(pg, EXTEND(src1, 3), src2); \
} while (0)
#define VML_VC_FUNC_PG_N2_2PARA_N4(dst, func, pg, src1, src2) do { \
    VML_VC_FUNC_PG_N2_2PARA_N3(dst, func, pg, src1, src2); \
    EXTEND(dst, 4) = func(pg, EXTEND(src1, 4), src2); \
} while (0)
#define VML_VC_FUNC_PG_N2_2PARA_N5(dst, func, pg, src1, src2) do { \
    VML_VC_FUNC_PG_N2_2PARA_N4(dst, func, pg, src1, src2); \
    EXTEND(dst, 5) = func(pg, EXTEND(src1, 5), src2); \
} while (0)
#define VML_VC_FUNC_PG_N2_2PARA_N6(dst, func, pg, src1, src2) do { \
    VML_VC_FUNC_PG_N2_2PARA_N5(dst, func, pg, src1, src2); \
    EXTEND(dst, 6) = func(pg, EXTEND(src1, 6), src2); \
} while (0)
#define VML_VC_FUNC_PG_N2_2PARA_N7(dst, func, pg, src1, src2) do { \
    VML_VC_FUNC_PG_N2_2PARA_N6(dst, func, pg, src1, src2); \
    EXTEND(dst, 7) = func(pg, EXTEND(src1, 7), src2); \
} while (0)
#define VML_VC_FUNC_PG_N2_2PARA_N8(dst, func, pg, src1, src2) do { \
    VML_VC_FUNC_PG_N2_2PARA_N7(dst, func, pg, src1, src2); \
    EXTEND(dst, 8) = func(pg, EXTEND(src1, 8), src2); \
} while (0)
#define VML_VC_FUNC_PG_N2_2PARA_HELPER(dst, func, pg, src1, src2, n) \
    VML_VC_FUNC_PG_N2_2PARA_N##n(dst, func, pg, src1, src2)
#define VML_VC_FUNC_PG_N2_2PARA_HELPER2(dst, func, pg, src1, src2, n) \
    VML_VC_FUNC_PG_N2_2PARA_HELPER(dst, func, pg, src1, src2, n)
#define VML_VC_FUNC_PG_N2_2PARA(dst, func, pg, src1, src2) do { \
    COMPILE_BARRIER(); \
    VML_VC_FUNC_PG_N2_2PARA_HELPER2(dst, func, pg, src1, src2, VML_VC_UNFOLD_N); \
    COMPILE_BARRIER(); \
} while (0)

// flexible N loop unrolling sve 2 parameter function with predict register(second paramter is const)
// predicate is also urolled
#define VML_VC_FUNC_UPG_N2_2PARA_N1(dst, func, pg, src1, src2) \
    EXTEND(dst, 1) = func(EXTEND(pg, 1), EXTEND(src1, 1), src2)
#define VML_VC_FUNC_UPG_N2_2PARA_N2(dst, func, pg, src1, src2) do { \
    VML_VC_FUNC_UPG_N2_2PARA_N1(dst, func, pg, src1, src2); \
    EXTEND(dst, 2) = func(EXTEND(pg, 2), EXTEND(src1, 2), src2); \
} while (0)
#define VML_VC_FUNC_UPG_N2_2PARA_N3(dst, func, pg, src1, src2) do { \
    VML_VC_FUNC_UPG_N2_2PARA_N2(dst, func, pg, src1, src2); \
    EXTEND(dst, 3) = func(EXTEND(pg, 3), EXTEND(src1, 3), src2); \
} while (0)
#define VML_VC_FUNC_UPG_N2_2PARA_N4(dst, func, pg, src1, src2) do { \
    VML_VC_FUNC_UPG_N2_2PARA_N3(dst, func, pg, src1, src2); \
    EXTEND(dst, 4) = func(EXTEND(pg, 4), EXTEND(src1, 4), src2); \
} while (0)
#define VML_VC_FUNC_UPG_N2_2PARA_N5(dst, func, pg, src1, src2) do { \
    VML_VC_FUNC_UPG_N2_2PARA_N4(dst, func, pg, src1, src2); \
    EXTEND(dst, 5) = func(EXTEND(pg, 5), EXTEND(src1, 5), src2); \
} while (0)
#define VML_VC_FUNC_UPG_N2_2PARA_N6(dst, func, pg, src1, src2) do { \
    VML_VC_FUNC_UPG_N2_2PARA_N5(dst, func, pg, src1, src2); \
    EXTEND(dst, 6) = func(EXTEND(pg, 6), EXTEND(src1, 6), src2); \
} while (0)
#define VML_VC_FUNC_UPG_N2_2PARA_N7(dst, func, pg, src1, src2) do { \
    VML_VC_FUNC_UPG_N2_2PARA_N6(dst, func, pg, src1, src2); \
    EXTEND(dst, 7) = func(EXTEND(pg, 7), EXTEND(src1, 7), src2); \
} while (0)
#define VML_VC_FUNC_UPG_N2_2PARA_N8(dst, func, pg, src1, src2) do { \
    VML_VC_FUNC_UPG_N2_2PARA_N7(dst, func, pg, src1, src2); \
    EXTEND(dst, 8) = func(EXTEND(pg, 8), EXTEND(src1, 8), src2); \
} while (0)
#define VML_VC_FUNC_UPG_N2_2PARA_HELPER(dst, func, pg, src1, src2, n) \
    VML_VC_FUNC_UPG_N2_2PARA_N##n(dst, func, pg, src1, src2)
#define VML_VC_FUNC_UPG_N2_2PARA_HELPER2(dst, func, pg, src1, src2, n) \
    VML_VC_FUNC_UPG_N2_2PARA_HELPER(dst, func, pg, src1, src2, n)
#define VML_VC_FUNC_UPG_N2_2PARA(dst, func, pg, src1, src2) do { \
    COMPILE_BARRIER(); \
    VML_VC_FUNC_UPG_N2_2PARA_HELPER2(dst, func, pg, src1, src2, VML_VC_UNFOLD_N); \
    COMPILE_BARRIER(); \
} while (0)

// flexible N loop unrolling sve 3 parameter function without predict register(third parameter is const)
#define VML_VC_FUNC_NOPG_N3_3PARA_N1(dst, func, src1, src2, src3) \
    EXTEND(dst, 1) = func(EXTEND(src1, 1), EXTEND(src2, 1), src3)
#define VML_VC_FUNC_NOPG_N3_3PARA_N2(dst, func, src1, src2, src3) do { \
    VML_VC_FUNC_NOPG_N3_3PARA_N1(dst, func, src1, src2, src3); \
    EXTEND(dst, 2) = func(EXTEND(src1, 2), EXTEND(src2, 2), src3); \
} while (0)
#define VML_VC_FUNC_NOPG_N3_3PARA_N3(dst, func, src1, src2, src3) do { \
    VML_VC_FUNC_NOPG_N3_3PARA_N2(dst, func, src1, src2, src3); \
    EXTEND(dst, 3) = func(EXTEND(src1, 3), EXTEND(src2, 3), src3); \
} while (0)
#define VML_VC_FUNC_NOPG_N3_3PARA_N4(dst, func, src1, src2, src3) do { \
    VML_VC_FUNC_NOPG_N3_3PARA_N3(dst, func, src1, src2, src3); \
    EXTEND(dst, 4) = func(EXTEND(src1, 4), EXTEND(src2, 4), src3); \
} while (0)
#define VML_VC_FUNC_NOPG_N3_3PARA_N5(dst, func, src1, src2, src3) do { \
    VML_VC_FUNC_NOPG_N3_3PARA_N4(dst, func, src1, src2, src3); \
    EXTEND(dst, 5) = func(EXTEND(src1, 5), EXTEND(src2, 5), src3); \
} while (0)
#define VML_VC_FUNC_NOPG_N3_3PARA_N6(dst, func, src1, src2, src3) do { \
    VML_VC_FUNC_NOPG_N3_3PARA_N5(dst, func, src1, src2, src3); \
    EXTEND(dst, 6) = func(EXTEND(src1, 6), EXTEND(src2, 6), src3); \
} while (0)
#define VML_VC_FUNC_NOPG_N3_3PARA_N7(dst, func, src1, src2, src3) do { \
    VML_VC_FUNC_NOPG_N3_3PARA_N6(dst, func, src1, src2, src3); \
    EXTEND(dst, 7) = func(EXTEND(src1, 7), EXTEND(src2, 7), src3); \
} while (0)
#define VML_VC_FUNC_NOPG_N3_3PARA_N8(dst, func, src1, src2, src3) do { \
    VML_VC_FUNC_NOPG_N3_3PARA_N7(dst, func, src1, src2, src3); \
    EXTEND(dst, 8) = func(EXTEND(src1, 8), EXTEND(src2, 8), src3); \
} while (0)
#define VML_VC_FUNC_NOPG_N3_3PARA_HELPER(dst, func, src1, src2, src3, n) \
    VML_VC_FUNC_NOPG_N3_3PARA_N##n(dst, func, src1, src2, src3)
#define VML_VC_FUNC_NOPG_N3_3PARA_HELPER2(dst, func, src1, src2, src3, n) \
    VML_VC_FUNC_NOPG_N3_3PARA_HELPER(dst, func, src1, src2, src3, n)
#define VML_VC_FUNC_NOPG_N3_3PARA(dst, func, src1, src2, src3) do { \
    COMPILE_BARRIER(); \
    VML_VC_FUNC_NOPG_N3_3PARA_HELPER2(dst, func, src1, src2, src3, VML_VC_UNFOLD_N); \
    COMPILE_BARRIER(); \
} while (0)

#define VML_VC_FUNC_NOPG_N2_3PARA_N1(dst, func, src1, src2, src3) \
    EXTEND(dst, 1) = func(EXTEND(src1, 1), src2, EXTEND(src3, 1))
#define VML_VC_FUNC_NOPG_N2_3PARA_N2(dst, func, src1, src2, src3) do { \
    VML_VC_FUNC_NOPG_N2_3PARA_N1(dst, func, src1, src2, src3); \
    EXTEND(dst, 2) = func(EXTEND(src1, 2), src2, EXTEND(src3, 2)); \
} while (0)
#define VML_VC_FUNC_NOPG_N2_3PARA_N3(dst, func, src1, src2, src3) do { \
    VML_VC_FUNC_NOPG_N2_3PARA_N2(dst, func, src1, src2, src3); \
    EXTEND(dst, 3) = func(EXTEND(src1, 3), src2, EXTEND(src3, 3)); \
} while (0)
#define VML_VC_FUNC_NOPG_N2_3PARA_N4(dst, func, src1, src2, src3) do { \
    VML_VC_FUNC_NOPG_N2_3PARA_N3(dst, func, src1, src2, src3); \
    EXTEND(dst, 4) = func(EXTEND(src1, 4), src2, EXTEND(src3, 4)); \
} while (0)
#define VML_VC_FUNC_NOPG_N2_3PARA_N5(dst, func, src1, src2, src3) do { \
    VML_VC_FUNC_NOPG_N2_3PARA_N4(dst, func, src1, src2, src3); \
    EXTEND(dst, 5) = func(EXTEND(src1, 5), src2, EXTEND(src3, 5)); \
} while (0)
#define VML_VC_FUNC_NOPG_N2_3PARA_N6(dst, func, src1, src2, src3) do { \
    VML_VC_FUNC_NOPG_N2_3PARA_N5(dst, func, src1, src2, src3); \
    EXTEND(dst, 6) = func(EXTEND(src1, 6), src2, EXTEND(src3, 6)); \
} while (0)
#define VML_VC_FUNC_NOPG_N2_3PARA_N7(dst, func, src1, src2, src3) do { \
    VML_VC_FUNC_NOPG_N2_3PARA_N6(dst, func, src1, src2, src3); \
    EXTEND(dst, 7) = func(EXTEND(src1, 7), src2, EXTEND(src3, 7)); \
} while (0)
#define VML_VC_FUNC_NOPG_N2_3PARA_N8(dst, func, src1, src2, src3) do { \
    VML_VC_FUNC_NOPG_N2_3PARA_N7(dst, func, src1, src2, src3); \
    EXTEND(dst, 8) = func(EXTEND(src1, 8), src2, EXTEND(src3, 8)); \
} while (0)
#define VML_VC_FUNC_NOPG_N2_3PARA_HELPER(dst, func, src1, src2, src3, n) \
    VML_VC_FUNC_NOPG_N2_3PARA_N##n(dst, func, src1, src2, src3)
#define VML_VC_FUNC_NOPG_N2_3PARA_HELPER2(dst, func, src1, src2, src3, n) \
    VML_VC_FUNC_NOPG_N2_3PARA_HELPER(dst, func, src1, src2, src3, n)
#define VML_VC_FUNC_NOPG_N2_3PARA(dst, func, src1, src2, src3) do { \
    COMPILE_BARRIER(); \
    VML_VC_FUNC_NOPG_N2_3PARA_HELPER2(dst, func, src1, src2, src3, VML_VC_UNFOLD_N); \
    COMPILE_BARRIER(); \
} while (0)

#define VML_VC_FUNC_NOPG_N23_3PARA_N1(dst, func, src1, src2, src3) \
    EXTEND(dst, 1) = func(EXTEND(src1, 1), src2, src3)
#define VML_VC_FUNC_NOPG_N23_3PARA_N2(dst, func, src1, src2, src3) do { \
    VML_VC_FUNC_NOPG_N23_3PARA_N1(dst, func, src1, src2, src3); \
    EXTEND(dst, 2) = func(EXTEND(src1, 2), src2, src3); \
} while (0)
#define VML_VC_FUNC_NOPG_N23_3PARA_N3(dst, func, src1, src2, src3) do { \
    VML_VC_FUNC_NOPG_N23_3PARA_N2(dst, func, src1, src2, src3); \
    EXTEND(dst, 3) = func(EXTEND(src1, 3), src2, src3); \
} while (0)
#define VML_VC_FUNC_NOPG_N23_3PARA_N4(dst, func, src1, src2, src3) do { \
    VML_VC_FUNC_NOPG_N23_3PARA_N3(dst, func, src1, src2, src3); \
    EXTEND(dst, 4) = func(EXTEND(src1, 4), src2, src3); \
} while (0)
#define VML_VC_FUNC_NOPG_N23_3PARA_N5(dst, func, src1, src2, src3) do { \
    VML_VC_FUNC_NOPG_N23_3PARA_N4(dst, func, src1, src2, src3); \
    EXTEND(dst, 5) = func(EXTEND(src1, 5), src2, src3); \
} while (0)
#define VML_VC_FUNC_NOPG_N23_3PARA_N6(dst, func, src1, src2, src3) do { \
    VML_VC_FUNC_NOPG_N23_3PARA_N5(dst, func, src1, src2, src3); \
    EXTEND(dst, 6) = func(EXTEND(src1, 6), src2, src3); \
} while (0)
#define VML_VC_FUNC_NOPG_N23_3PARA_N7(dst, func, src1, src2, src3) do { \
    VML_VC_FUNC_NOPG_N23_3PARA_N6(dst, func, src1, src2, src3); \
    EXTEND(dst, 7) = func(EXTEND(src1, 7), src2, src3); \
} while (0)
#define VML_VC_FUNC_NOPG_N23_3PARA_N8(dst, func, src1, src2, src3) do { \
    VML_VC_FUNC_NOPG_N23_3PARA_N7(dst, func, src1, src2, src3); \
    EXTEND(dst, 8) = func(EXTEND(src1, 8), src2, src3); \
} while (0)
#define VML_VC_FUNC_NOPG_N23_3PARA_HELPER(dst, func, src1, src2, src3, n) \
    VML_VC_FUNC_NOPG_N23_3PARA_N##n(dst, func, src1, src2, src3)
#define VML_VC_FUNC_NOPG_N23_3PARA_HELPER2(dst, func, src1, src2, src3, n) \
    VML_VC_FUNC_NOPG_N23_3PARA_HELPER(dst, func, src1, src2, src3, n)
#define VML_VC_FUNC_NOPG_N23_3PARA(dst, func, src1, src2, src3) do { \
    COMPILE_BARRIER(); \
    VML_VC_FUNC_NOPG_N23_3PARA_HELPER2(dst, func, src1, src2, src3, VML_VC_UNFOLD_N); \
    COMPILE_BARRIER(); \
} while (0)

#define VML_VC_FUNC_NOPG_N2_4PARA_N1(dst, func, src1, src2, src3, src4) \
    EXTEND(dst, 1) = func(EXTEND(src1, 1), src2, EXTEND(src3, 1), EXTEND(src4, 1))
#define VML_VC_FUNC_NOPG_N2_4PARA_N2(dst, func, src1, src2, src3, src4) do { \
    VML_VC_FUNC_NOPG_N2_4PARA_N1(dst, func, src1, src2, src3, src4); \
    EXTEND(dst, 2) = func(EXTEND(src1, 2), src2, EXTEND(src3, 2), EXTEND(src4, 2)); \
} while (0)
#define VML_VC_FUNC_NOPG_N2_4PARA_N3(dst, func, src1, src2, src3, src4) do { \
    VML_VC_FUNC_NOPG_N2_4PARA_N2(dst, func, src1, src2, src3, src4); \
    EXTEND(dst, 3) = func(EXTEND(src1, 3), src2, EXTEND(src3, 3), EXTEND(src4, 3)); \
} while (0)
#define VML_VC_FUNC_NOPG_N2_4PARA_N4(dst, func, src1, src2, src3, src4) do { \
    VML_VC_FUNC_NOPG_N2_4PARA_N3(dst, func, src1, src2, src3, src4); \
    EXTEND(dst, 4) = func(EXTEND(src1, 4), src2, EXTEND(src3, 4), EXTEND(src4, 4)); \
} while (0)
#define VML_VC_FUNC_NOPG_N2_4PARA_N5(dst, func, src1, src2, src3, src4) do { \
    VML_VC_FUNC_NOPG_N2_4PARA_N4(dst, func, src1, src2, src3, src4); \
    EXTEND(dst, 5) = func(EXTEND(src1, 5), src2, EXTEND(src3, 5),  EXTEND(src4, 5)); \
} while (0)
#define VML_VC_FUNC_NOPG_N2_4PARA_N6(dst, func, src1, src2, src3, src4) do { \
    VML_VC_FUNC_NOPG_N2_4PARA_N5(dst, func, src1, src2, src3, src4); \
    EXTEND(dst, 6) = func(EXTEND(src1, 6), src2, EXTEND(src3, 6), EXTEND(src4, 6)); \
} while (0)
#define VML_VC_FUNC_NOPG_N2_4PARA_N7(dst, func, src1, src2, src3, src4) do { \
    VML_VC_FUNC_NOPG_N2_4PARA_N6(dst, func, src1, src2, src3, src4); \
    EXTEND(dst, 7) = func(EXTEND(src1, 7), src2, EXTEND(src3, 7), EXTEND(src4, 7)); \
} while (0)
#define VML_VC_FUNC_NOPG_N2_4PARA_N8(dst, func, src1, src2, src3, src4) do { \
    VML_VC_FUNC_NOPG_N2_4PARA_N7(dst, func, src1, src2, src3, src4); \
    EXTEND(dst, 8) = func(EXTEND(src1, 8), src2, EXTEND(src3, 8), EXTEND(src4, 8)); \
} while (0)
#define VML_VC_FUNC_NOPG_N2_4PARA_HELPER(dst, func, src1, src2, src3, src4, n) \
    VML_VC_FUNC_NOPG_N2_4PARA_N##n(dst, func, src1, src2, src3, src4)
#define VML_VC_FUNC_NOPG_N2_4PARA_HELPER2(dst, func, src1, src2, src3, src4, n) \
    VML_VC_FUNC_NOPG_N2_4PARA_HELPER(dst, func, src1, src2, src3, src4, n)
#define VML_VC_FUNC_NOPG_N2_4PARA(dst, func, src1, src2, src3, src4) do { \
    COMPILE_BARRIER(); \
    VML_VC_FUNC_NOPG_N2_4PARA_HELPER2(dst, func, src1, src2, src3, src4, VML_VC_UNFOLD_N); \
    COMPILE_BARRIER(); \
} while (0)


#define VML_VC_FUNC_NOPG_N34_4PARA_N1(dst, func, src1, src2, src3, src4) \
    EXTEND(dst, 1) = func(EXTEND(src1, 1), EXTEND(src2, 1), src3, src4)
#define VML_VC_FUNC_NOPG_N34_4PARA_N2(dst, func, src1, src2, src3, src4) do { \
    VML_VC_FUNC_NOPG_N34_4PARA_N1(dst, func, src1, src2, src3, src4); \
    EXTEND(dst, 2) = func(EXTEND(src1, 2), EXTEND(src2, 2), src3, src4); \
} while (0)
#define VML_VC_FUNC_NOPG_N34_4PARA_N3(dst, func, src1, src2, src3, src4) do { \
    VML_VC_FUNC_NOPG_N34_4PARA_N2(dst, func, src1, src2, src3, src4); \
    EXTEND(dst, 3) = func(EXTEND(src1, 3), EXTEND(src2, 3), src3, src4); \
} while (0)
#define VML_VC_FUNC_NOPG_N34_4PARA_N4(dst, func, src1, src2, src3, src4) do { \
    VML_VC_FUNC_NOPG_N34_4PARA_N3(dst, func, src1, src2, src3, src4); \
    EXTEND(dst, 4) = func(EXTEND(src1, 4), EXTEND(src2, 4), src3, src4); \
} while (0)
#define VML_VC_FUNC_NOPG_N34_4PARA_N5(dst, func, src1, src2, src3, src4) do { \
    VML_VC_FUNC_NOPG_N34_4PARA_N4(dst, func, src1, src2, src3, src4); \
    EXTEND(dst, 5) = func(EXTEND(src1, 5), EXTEND(src2, 5), src3, src4); \
} while (0)
#define VML_VC_FUNC_NOPG_N34_4PARA_N6(dst, func, src1, src2, src3, src4) do { \
    VML_VC_FUNC_NOPG_N34_4PARA_N5(dst, func, src1, src2, src3, src4); \
    EXTEND(dst, 6) = func(EXTEND(src1, 6), EXTEND(src2, 6), src3, src4); \
} while (0)
#define VML_VC_FUNC_NOPG_N34_4PARA_N7(dst, func, src1, src2, src3, src4) do { \
    VML_VC_FUNC_NOPG_N34_4PARA_N6(dst, func, src1, src2, src3, src4); \
    EXTEND(dst, 7) = func(EXTEND(src1, 7), EXTEND(src2, 7), src3, src4); \
} while (0)
#define VML_VC_FUNC_NOPG_N34_4PARA_N8(dst, func, src1, src2, src3, src4) do { \
    VML_VC_FUNC_NOPG_N34_4PARA_N7(dst, func, src1, src2, src3, src4); \
    EXTEND(dst, 8) = func(EXTEND(src1, 8), EXTEND(src2, 8), src3, src4); \
} while (0)
#define VML_VC_FUNC_NOPG_N34_4PARA_HELPER(dst, func, src1, src2, src3, src4, n) \
    VML_VC_FUNC_NOPG_N34_4PARA_N##n(dst, func, src1, src2, src3, src4)
#define VML_VC_FUNC_NOPG_N34_4PARA_HELPER2(dst, func, src1, src2, src3, src4, n) \
    VML_VC_FUNC_NOPG_N34_4PARA_HELPER(dst, func, src1, src2, src3, src4, n)
#define VML_VC_FUNC_NOPG_N34_4PARA(dst, func, src1, src2, src3, src4) do { \
    COMPILE_BARRIER(); \
    VML_VC_FUNC_NOPG_N34_4PARA_HELPER2(dst, func, src1, src2, src3, src4, VML_VC_UNFOLD_N); \
    COMPILE_BARRIER(); \
} while (0)

// flexible N loop unrolling sve 2 parameter function with predict register
#define VML_VC_FUNC_PG_2PARA_N1(dst, func, pg, src1, src2) \
    EXTEND(dst, 1) = func(pg, EXTEND(src1, 1), EXTEND(src2, 1))
#define VML_VC_FUNC_PG_2PARA_N2(dst, func, pg, src1, src2) do { \
    VML_VC_FUNC_PG_2PARA_N1(dst, func, pg, src1, src2); \
    EXTEND(dst, 2) = func(pg, EXTEND(src1, 2), EXTEND(src2, 2)); \
} while (0)
#define VML_VC_FUNC_PG_2PARA_N3(dst, func, pg, src1, src2) do { \
    VML_VC_FUNC_PG_2PARA_N2(dst, func, pg, src1, src2); \
    EXTEND(dst, 3) = func(pg, EXTEND(src1, 3), EXTEND(src2, 3)); \
} while (0)
#define VML_VC_FUNC_PG_2PARA_N4(dst, func, pg, src1, src2) do { \
    VML_VC_FUNC_PG_2PARA_N3(dst, func, pg, src1, src2); \
    EXTEND(dst, 4) = func(pg, EXTEND(src1, 4), EXTEND(src2, 4)); \
} while (0)
#define VML_VC_FUNC_PG_2PARA_N5(dst, func, pg, src1, src2) do { \
    VML_VC_FUNC_PG_2PARA_N4(dst, func, pg, src1, src2); \
    EXTEND(dst, 5) = func(pg, EXTEND(src1, 5), EXTEND(src2, 5)); \
} while (0)
#define VML_VC_FUNC_PG_2PARA_N6(dst, func, pg, src1, src2) do { \
    VML_VC_FUNC_PG_2PARA_N5(dst, func, pg, src1, src2); \
    EXTEND(dst, 6) = func(pg, EXTEND(src1, 6), EXTEND(src2, 6)); \
} while (0)
#define VML_VC_FUNC_PG_2PARA_N7(dst, func, pg, src1, src2) do { \
    VML_VC_FUNC_PG_2PARA_N6(dst, func, pg, src1, src2); \
    EXTEND(dst, 7) = func(pg, EXTEND(src1, 7), EXTEND(src2, 7)); \
} while (0)
#define VML_VC_FUNC_PG_2PARA_N8(dst, func, pg, src1, src2) do { \
    VML_VC_FUNC_PG_2PARA_N7(dst, func, pg, src1, src2); \
    EXTEND(dst, 8) = func(pg, EXTEND(src1, 8), EXTEND(src2, 8)); \
} while (0)
#define VML_VC_FUNC_PG_2PARA_HELPER(dst, func, pg, src1, src2, n) \
    VML_VC_FUNC_PG_2PARA_N##n(dst, func, pg, src1, src2)
#define VML_VC_FUNC_PG_2PARA_HELPER2(dst, func, pg, src1, src2, n) \
    VML_VC_FUNC_PG_2PARA_HELPER(dst, func, pg, src1, src2, n)
#define VML_VC_FUNC_PG_2PARA(dst, func, pg, src1, src2) do { \
    COMPILE_BARRIER(); \
    VML_VC_FUNC_PG_2PARA_HELPER2(dst, func, pg, src1, src2, VML_VC_UNFOLD_N); \
    COMPILE_BARRIER(); \
} while (0)

// flexible N loop unrolling sve 2 parameter function without predict register
#define VML_VC_FUNC_NOPG_2PARA_N1(dst, func, src1, src2) \
    EXTEND(dst, 1) = func(EXTEND(src1, 1), EXTEND(src2, 1))
#define VML_VC_FUNC_NOPG_2PARA_N2(dst, func, src1, src2) do { \
    VML_VC_FUNC_NOPG_2PARA_N1(dst, func, src1, src2); \
    EXTEND(dst, 2) = func(EXTEND(src1, 2), EXTEND(src2, 2)); \
} while (0)
#define VML_VC_FUNC_NOPG_2PARA_N3(dst, func, src1, src2) do { \
    VML_VC_FUNC_NOPG_2PARA_N2(dst, func, src1, src2); \
    EXTEND(dst, 3) = func(EXTEND(src1, 3), EXTEND(src2, 3)); \
} while (0)
#define VML_VC_FUNC_NOPG_2PARA_N4(dst, func, src1, src2) do { \
    VML_VC_FUNC_NOPG_2PARA_N3(dst, func, src1, src2); \
    EXTEND(dst, 4) = func(EXTEND(src1, 4), EXTEND(src2, 4)); \
} while (0)
#define VML_VC_FUNC_NOPG_2PARA_N5(dst, func, src1, src2) do { \
    VML_VC_FUNC_NOPG_2PARA_N4(dst, func, src1, src2); \
    EXTEND(dst, 5) = func(EXTEND(src1, 5), EXTEND(src2, 5)); \
} while (0)
#define VML_VC_FUNC_NOPG_2PARA_N6(dst, func, src1, src2) do { \
    VML_VC_FUNC_NOPG_2PARA_N5(dst, func, src1, src2); \
    EXTEND(dst, 6) = func(EXTEND(src1, 6), EXTEND(src2, 6)); \
} while (0)
#define VML_VC_FUNC_NOPG_2PARA_N7(dst, func, src1, src2) do { \
    VML_VC_FUNC_NOPG_2PARA_N6(dst, func, src1, src2); \
    EXTEND(dst, 7) = func(EXTEND(src1, 7), EXTEND(src2, 7)); \
} while (0)
#define VML_VC_FUNC_NOPG_2PARA_N8(dst, func, src1, src2) do { \
    VML_VC_FUNC_NOPG_2PARA_N7(dst, func, src1, src2); \
    EXTEND(dst, 8) = func(EXTEND(src1, 8), EXTEND(src2, 8)); \
} while (0)
#define VML_VC_FUNC_NOPG_2PARA_HELPER(dst, func, src1, src2, n) \
    VML_VC_FUNC_NOPG_2PARA_N##n(dst, func, src1, src2)
#define VML_VC_FUNC_NOPG_2PARA_HELPER2(dst, func, src1, src2, n) \
    VML_VC_FUNC_NOPG_2PARA_HELPER(dst, func, src1, src2, n)
#define VML_VC_FUNC_NOPG_2PARA(dst, func, src1, src2) do { \
    COMPILE_BARRIER(); \
    VML_VC_FUNC_NOPG_2PARA_HELPER2(dst, func, src1, src2, VML_VC_UNFOLD_N); \
    COMPILE_BARRIER(); \
} while (0)

#define VML_VC_FUNC_NOPG_N2_2PARA_N1(dst, func, src1, src2) \
    EXTEND(dst, 1) = func(EXTEND(src1, 1), src2)
#define VML_VC_FUNC_NOPG_N2_2PARA_N2(dst, func, src1, src2) do { \
    VML_VC_FUNC_NOPG_N2_2PARA_N1(dst, func, src1, src2); \
    EXTEND(dst, 2) = func(EXTEND(src1, 2), src2); \
} while (0)
#define VML_VC_FUNC_NOPG_N2_2PARA_N3(dst, func, src1, src2) do { \
    VML_VC_FUNC_NOPG_N2_2PARA_N2(dst, func, src1, src2); \
    EXTEND(dst, 3) = func(EXTEND(src1, 3), src2); \
} while (0)
#define VML_VC_FUNC_NOPG_N2_2PARA_N4(dst, func, src1, src2) do { \
    VML_VC_FUNC_NOPG_N2_2PARA_N3(dst, func, src1, src2); \
    EXTEND(dst, 4) = func(EXTEND(src1, 4), src2); \
} while (0)
#define VML_VC_FUNC_NOPG_N2_2PARA_N5(dst, func, src1, src2) do { \
    VML_VC_FUNC_NOPG_N2_2PARA_N4(dst, func, src1, src2); \
    EXTEND(dst, 5) = func(EXTEND(src1, 5), src2); \
} while (0)
#define VML_VC_FUNC_NOPG_N2_2PARA_N6(dst, func, src1, src2) do { \
    VML_VC_FUNC_NOPG_N2_2PARA_N5(dst, func, src1, src2); \
    EXTEND(dst, 6) = func(EXTEND(src1, 6), src2); \
} while (0)
#define VML_VC_FUNC_NOPG_N2_2PARA_N7(dst, func, src1, src2) do { \
    VML_VC_FUNC_NOPG_N2_2PARA_N6(dst, func, src1, src2); \
    EXTEND(dst, 7) = func(EXTEND(src1, 7), src2); \
} while (0)
#define VML_VC_FUNC_NOPG_N2_2PARA_N8(dst, func, src1, src2) do { \
    VML_VC_FUNC_NOPG_N2_2PARA_N7(dst, func, src1, src2); \
    EXTEND(dst, 8) = func(EXTEND(src1, 8), src2); \
} while (0)
#define VML_VC_FUNC_NOPG_N2_2PARA_HELPER(dst, func, src1, src2, n) \
    VML_VC_FUNC_NOPG_N2_2PARA_N##n(dst, func, src1, src2)
#define VML_VC_FUNC_NOPG_N2_2PARA_HELPER2(dst, func, src1, src2, n) \
    VML_VC_FUNC_NOPG_N2_2PARA_HELPER(dst, func, src1, src2, n)
#define VML_VC_FUNC_NOPG_N2_2PARA(dst, func, src1, src2) do { \
    COMPILE_BARRIER(); \
    VML_VC_FUNC_NOPG_N2_2PARA_HELPER2(dst, func, src1, src2, VML_VC_UNFOLD_N); \
    COMPILE_BARRIER(); \
} while (0)

// flexible N loop unrolling sve 1 parameter function with predict register
#define VML_VC_FUNC_PG_1PARA_N1(dst, func, pg, src) \
    EXTEND(dst, 1) = func(pg, EXTEND(src, 1))
#define VML_VC_FUNC_PG_1PARA_N2(dst, func, pg, src) do { \
    VML_VC_FUNC_PG_1PARA_N1(dst, func, pg, src); \
    EXTEND(dst, 2) = func(pg, EXTEND(src, 2)); \
} while (0)
#define VML_VC_FUNC_PG_1PARA_N3(dst, func, pg, src) do { \
    VML_VC_FUNC_PG_1PARA_N2(dst, func, pg, src); \
    EXTEND(dst, 3) = func(pg, EXTEND(src, 3)); \
} while (0)
#define VML_VC_FUNC_PG_1PARA_N4(dst, func, pg, src) do { \
    VML_VC_FUNC_PG_1PARA_N3(dst, func, pg, src); \
    EXTEND(dst, 4) = func(pg, EXTEND(src, 4)); \
} while (0)
#define VML_VC_FUNC_PG_1PARA_N5(dst, func, pg, src) do { \
    VML_VC_FUNC_PG_1PARA_N4(dst, func, pg, src); \
    EXTEND(dst, 5) = func(pg, EXTEND(src, 5)); \
} while (0)
#define VML_VC_FUNC_PG_1PARA_N6(dst, func, pg, src) do { \
    VML_VC_FUNC_PG_1PARA_N5(dst, func, pg, src); \
    EXTEND(dst, 6) = func(pg, EXTEND(src, 6)); \
} while (0)
#define VML_VC_FUNC_PG_1PARA_N7(dst, func, pg, src) do { \
    VML_VC_FUNC_PG_1PARA_N6(dst, func, pg, src); \
    EXTEND(dst, 7) = func(pg, EXTEND(src, 7)); \
} while (0)
#define VML_VC_FUNC_PG_1PARA_N8(dst, func, pg, src) do { \
    VML_VC_FUNC_PG_1PARA_N7(dst, func, pg, src); \
    EXTEND(dst, 8) = func(pg, EXTEND(src, 8)); \
} while (0)
#define VML_VC_FUNC_PG_1PARA_HELPER(dst, func, pg, src, n) \
    VML_VC_FUNC_PG_1PARA_N##n(dst, func, pg, src)
#define VML_VC_FUNC_PG_1PARA_HELPER2(dst, func, pg, src, n) \
    VML_VC_FUNC_PG_1PARA_HELPER(dst, func, pg, src, n)
#define VML_VC_FUNC_PG_1PARA(dst, func, pg, src) do { \
    COMPILE_BARRIER(); \
    VML_VC_FUNC_PG_1PARA_HELPER2(dst, func, pg, src, VML_VC_UNFOLD_N); \
    COMPILE_BARRIER(); \
} while (0)


#define VML_VC_FUNC_NOPG_N_1PARA_N1(dst, func, src) \
    EXTEND(dst, 1) = func(src)
#define VML_VC_FUNC_NOPG_N_1PARA_N2(dst, func, src) do { \
    VML_VC_FUNC_NOPG_N_1PARA_N1(dst, func, src); \
    EXTEND(dst, 2) = func(src); \
} while (0)
#define VML_VC_FUNC_NOPG_N_1PARA_N3(dst, func, src) do { \
    VML_VC_FUNC_NOPG_N_1PARA_N2(dst, func, src); \
    EXTEND(dst, 3) = func(src); \
} while (0)
#define VML_VC_FUNC_NOPG_N_1PARA_N4(dst, func, src) do { \
    VML_VC_FUNC_NOPG_N_1PARA_N3(dst, func, src); \
    EXTEND(dst, 4) = func(src); \
} while (0)
#define VML_VC_FUNC_NOPG_N_1PARA_N5(dst, func, src) do { \
    VML_VC_FUNC_NOPG_N_1PARA_N4(dst, func, src); \
    EXTEND(dst, 5) = func(src); \
} while (0)
#define VML_VC_FUNC_NOPG_N_1PARA_N6(dst, func, src) do { \
    VML_VC_FUNC_NOPG_N_1PARA_N5(dst, func, src); \
    EXTEND(dst, 6) = func(src); \
} while (0)
#define VML_VC_FUNC_NOPG_N_1PARA_N7(dst, func, src) do { \
    VML_VC_FUNC_NOPG_N_1PARA_N6(dst, func, src); \
    EXTEND(dst, 7) = func(src); \
} while (0)
#define VML_VC_FUNC_NOPG_N_1PARA_N8(dst, func, src) do { \
    VML_VC_FUNC_NOPG_N_1PARA_N7(dst, func, src); \
    EXTEND(dst, 8) = func(src); \
} while (0)
#define VML_VC_FUNC_NOPG_N_1PARA_HELPER(dst, func, src, n) \
    VML_VC_FUNC_NOPG_N_1PARA_N##n(dst, func, src)
#define VML_VC_FUNC_NOPG_N_1PARA_HELPER2(dst, func, src, n) \
    VML_VC_FUNC_NOPG_N_1PARA_HELPER(dst, func, src, n)
#define VML_VC_FUNC_NOPG_N_1PARA(dst, func, src) do { \
    COMPILE_BARRIER(); \
    VML_VC_FUNC_NOPG_N_1PARA_HELPER2(dst, func, src, VML_VC_UNFOLD_N); \
    COMPILE_BARRIER(); \
} while (0)

// flexible N loop unrolling sve 1 parameter function without predict register
#define VML_VC_FUNC_NOPG_1PARA_N1(dst, func, src) \
    EXTEND(dst, 1) = func(EXTEND(src, 1))
#define VML_VC_FUNC_NOPG_1PARA_N2(dst, func, src) do { \
    VML_VC_FUNC_NOPG_1PARA_N1(dst, func, src); \
    EXTEND(dst, 2) = func(EXTEND(src, 2)); \
} while (0)
#define VML_VC_FUNC_NOPG_1PARA_N3(dst, func, src) do { \
    VML_VC_FUNC_NOPG_1PARA_N2(dst, func, src); \
    EXTEND(dst, 3) = func(EXTEND(src, 3)); \
} while (0)
#define VML_VC_FUNC_NOPG_1PARA_N4(dst, func, src) do { \
    VML_VC_FUNC_NOPG_1PARA_N3(dst, func, src); \
    EXTEND(dst, 4) = func(EXTEND(src, 4)); \
} while (0)
#define VML_VC_FUNC_NOPG_1PARA_N5(dst, func, src) do { \
    VML_VC_FUNC_NOPG_1PARA_N4(dst, func, src); \
    EXTEND(dst, 5) = func(EXTEND(src, 5)); \
} while (0)
#define VML_VC_FUNC_NOPG_1PARA_N6(dst, func, src) do { \
    VML_VC_FUNC_NOPG_1PARA_N5(dst, func, src); \
    EXTEND(dst, 6) = func(EXTEND(src, 6)); \
} while (0)
#define VML_VC_FUNC_NOPG_1PARA_N7(dst, func, src) do { \
    VML_VC_FUNC_NOPG_1PARA_N6(dst, func, src); \
    EXTEND(dst, 7) = func(EXTEND(src, 7)); \
} while (0)
#define VML_VC_FUNC_NOPG_1PARA_N8(dst, func, src) do { \
    VML_VC_FUNC_NOPG_1PARA_N7(dst, func, src); \
    EXTEND(dst, 8) = func(EXTEND(src, 8)); \
} while (0)
#define VML_VC_FUNC_NOPG_1PARA_HELPER(dst, func, src, n) \
    VML_VC_FUNC_NOPG_1PARA_N##n(dst, func, src)
#define VML_VC_FUNC_NOPG_1PARA_HELPER2(dst, func, src, n) \
    VML_VC_FUNC_NOPG_1PARA_HELPER(dst, func, src, n)
#define VML_VC_FUNC_NOPG_1PARA(dst, func, src) do { \
    COMPILE_BARRIER(); \
    VML_VC_FUNC_NOPG_1PARA_HELPER2(dst, func, src, VML_VC_UNFOLD_N); \
    COMPILE_BARRIER(); \
} while (0)


// flexible N loop unrolling sve 3 parameter function with predict register(first paramter is const)
#define VML_VC_FUNC_PG_N1_3PARA_N1(dst, func, pg, src1, src2, src3) \
    EXTEND(dst, 1) = func(pg, src1, EXTEND(src2, 1), EXTEND(src3, 1))
#define VML_VC_FUNC_PG_N1_3PARA_N2(dst, func, pg, src1, src2, src3) do { \
    VML_VC_FUNC_PG_N1_3PARA_N1(dst, func, pg, src1, src2, src3); \
    EXTEND(dst, 2) = func(pg, src1, EXTEND(src2, 2), EXTEND(src3, 2)); \
} while (0)
#define VML_VC_FUNC_PG_N1_3PARA_N3(dst, func, pg, src1, src2, src3) do { \
    VML_VC_FUNC_PG_N1_3PARA_N2(dst, func, pg, src1, src2, src3); \
    EXTEND(dst, 3) = func(pg, src1, EXTEND(src2, 3), EXTEND(src3, 3)); \
} while (0)
#define VML_VC_FUNC_PG_N1_3PARA_N4(dst, func, pg, src1, src2, src3) do { \
    VML_VC_FUNC_PG_N1_3PARA_N3(dst, func, pg, src1, src2, src3); \
    EXTEND(dst, 4) = func(pg, src1, EXTEND(src2, 4), EXTEND(src3, 4)); \
} while (0)
#define VML_VC_FUNC_PG_N1_3PARA_N5(dst, func, pg, src1, src2, src3) do { \
    VML_VC_FUNC_PG_N1_3PARA_N4(dst, func, pg, src1, src2, src3); \
    EXTEND(dst, 5) = func(pg, src1, EXTEND(src2, 5), EXTEND(src3, 5)); \
} while (0)
#define VML_VC_FUNC_PG_N1_3PARA_N6(dst, func, pg, src1, src2, src3) do { \
    VML_VC_FUNC_PG_N1_3PARA_N5(dst, func, pg, src1, src2, src3); \
    EXTEND(dst, 6) = func(pg, src1, EXTEND(src2, 6), EXTEND(src3, 6)); \
} while (0)
#define VML_VC_FUNC_PG_N1_3PARA_N7(dst, func, pg, src1, src2, src3) do { \
    VML_VC_FUNC_PG_N1_3PARA_N6(dst, func, pg, src1, src2, src3); \
    EXTEND(dst, 7) = func(pg, src1, EXTEND(src2, 7), EXTEND(src3, 7)); \
} while (0)
#define VML_VC_FUNC_PG_N1_3PARA_N8(dst, func, pg, src1, src2, src3) do { \
    VML_VC_FUNC_PG_N1_3PARA_N7(dst, func, pg, src1, src2, src3); \
    EXTEND(dst, 8) = func(pg, src1, EXTEND(src2, 8), EXTEND(src3, 8)); \
} while (0)
#define VML_VC_FUNC_PG_N1_3PARA_HELPER(dst, func, pg, src1, src2, src3, n) \
    VML_VC_FUNC_PG_N1_3PARA_N##n(dst, func, pg, src1, src2, src3)
#define VML_VC_FUNC_PG_N1_3PARA_HELPER2(dst, func, pg, src1, src2, src3, n) \
    VML_VC_FUNC_PG_N1_3PARA_HELPER(dst, func, pg, src1, src2, src3, n)
#define VML_VC_FUNC_PG_N1_3PARA(dst, func, pg, src1, src2, src3) do { \
    COMPILE_BARRIER(); \
    VML_VC_FUNC_PG_N1_3PARA_HELPER2(dst, func, pg, src1, src2, src3, VML_VC_UNFOLD_N); \
    COMPILE_BARRIER(); \
} while (0)

// flexible N loop unrolling sve 3 parameter function with predict register(second paramter is const)
#define VML_VC_FUNC_PG_N2_3PARA_N1(dst, func, pg, src1, src2, src3) \
    EXTEND(dst, 1) = func(pg, EXTEND(src1, 1), src2, EXTEND(src3, 1))
#define VML_VC_FUNC_PG_N2_3PARA_N2(dst, func, pg, src1, src2, src3) do { \
    VML_VC_FUNC_PG_N2_3PARA_N1(dst, func, pg, src1, src2, src3); \
    EXTEND(dst, 2) = func(pg, EXTEND(src1, 2), src2, EXTEND(src3, 2)); \
} while (0)
#define VML_VC_FUNC_PG_N2_3PARA_N3(dst, func, pg, src1, src2, src3) do { \
    VML_VC_FUNC_PG_N2_3PARA_N2(dst, func, pg, src1, src2, src3); \
    EXTEND(dst, 3) = func(pg, EXTEND(src1, 3), src2, EXTEND(src3, 3)); \
} while (0)
#define VML_VC_FUNC_PG_N2_3PARA_N4(dst, func, pg, src1, src2, src3) do { \
    VML_VC_FUNC_PG_N2_3PARA_N3(dst, func, pg, src1, src2, src3); \
    EXTEND(dst, 4) = func(pg, EXTEND(src1, 4), src2, EXTEND(src3, 4)); \
} while (0)
#define VML_VC_FUNC_PG_N2_3PARA_N5(dst, func, pg, src1, src2, src3) do { \
    VML_VC_FUNC_PG_N2_3PARA_N4(dst, func, pg, src1, src2, src3); \
    EXTEND(dst, 5) = func(pg, EXTEND(src1, 5), src2, EXTEND(src3, 5)); \
} while (0)
#define VML_VC_FUNC_PG_N2_3PARA_N6(dst, func, pg, src1, src2, src3) do { \
    VML_VC_FUNC_PG_N2_3PARA_N5(dst, func, pg, src1, src2, src3); \
    EXTEND(dst, 6) = func(pg, EXTEND(src1, 6), src2, EXTEND(src3, 6)); \
} while (0)
#define VML_VC_FUNC_PG_N2_3PARA_N7(dst, func, pg, src1, src2, src3) do { \
    VML_VC_FUNC_PG_N2_3PARA_N6(dst, func, pg, src1, src2, src3); \
    EXTEND(dst, 7) = func(pg, EXTEND(src1, 7), src2, EXTEND(src3, 7)); \
} while (0)
#define VML_VC_FUNC_PG_N2_3PARA_N8(dst, func, pg, src1, src2, src3) do { \
    VML_VC_FUNC_PG_N2_3PARA_N7(dst, func, pg, src1, src2, src3); \
    EXTEND(dst, 8) = func(pg, EXTEND(src1, 8), src2, EXTEND(src3, 8)); \
} while (0)
#define VML_VC_FUNC_PG_N2_3PARA_HELPER(dst, func, pg, src1, src2, src3, n) \
    VML_VC_FUNC_PG_N2_3PARA_N##n(dst, func, pg, src1, src2, src3)
#define VML_VC_FUNC_PG_N2_3PARA_HELPER2(dst, func, pg, src1, src2, src3, n) \
    VML_VC_FUNC_PG_N2_3PARA_HELPER(dst, func, pg, src1, src2, src3, n)
#define VML_VC_FUNC_PG_N2_3PARA(dst, func, pg, src1, src2, src3) do { \
    COMPILE_BARRIER(); \
    VML_VC_FUNC_PG_N2_3PARA_HELPER2(dst, func, pg, src1, src2, src3, VML_VC_UNFOLD_N); \
    COMPILE_BARRIER(); \
} while (0)

// flexible N loop unrolling sve 3 parameter function with predict register(first and third paramter is const)
#define VML_VC_FUNC_PG_N13_3PARA_N1(dst, func, pg, src1, src2, src3) \
    EXTEND(dst, 1) = func(pg, src1, EXTEND(src2, 1), src3)
#define VML_VC_FUNC_PG_N13_3PARA_N2(dst, func, pg, src1, src2, src3) do { \
    VML_VC_FUNC_PG_N13_3PARA_N1(dst, func, pg, src1, src2, src3); \
    EXTEND(dst, 2) = func(pg, src1, EXTEND(src2, 2), src3); \
} while (0)
#define VML_VC_FUNC_PG_N13_3PARA_N3(dst, func, pg, src1, src2, src3) do { \
    VML_VC_FUNC_PG_N13_3PARA_N2(dst, func, pg, src1, src2, src3); \
    EXTEND(dst, 3) = func(pg, src1, EXTEND(src2, 3), src3); \
} while (0)
#define VML_VC_FUNC_PG_N13_3PARA_N4(dst, func, pg, src1, src2, src3) do { \
    VML_VC_FUNC_PG_N13_3PARA_N3(dst, func, pg, src1, src2, src3); \
    EXTEND(dst, 4) = func(pg, src1, EXTEND(src2, 4), src3); \
} while (0)
#define VML_VC_FUNC_PG_N13_3PARA_N5(dst, func, pg, src1, src2, src3) do { \
    VML_VC_FUNC_PG_N13_3PARA_N4(dst, func, pg, src1, src2, src3); \
    EXTEND(dst, 5) = func(pg, src1, EXTEND(src2, 5), src3); \
} while (0)
#define VML_VC_FUNC_PG_N13_3PARA_N6(dst, func, pg, src1, src2, src3) do { \
    VML_VC_FUNC_PG_N13_3PARA_N5(dst, func, pg, src1, src2, src3); \
    EXTEND(dst, 6) = func(pg, src1, EXTEND(src2, 6), src3); \
} while (0)
#define VML_VC_FUNC_PG_N13_3PARA_N7(dst, func, pg, src1, src2, src3) do { \
    VML_VC_FUNC_PG_N13_3PARA_N6(dst, func, pg, src1, src2, src3); \
    EXTEND(dst, 7) = func(pg, src1, EXTEND(src2, 7), src3); \
} while (0)
#define VML_VC_FUNC_PG_N13_3PARA_N8(dst, func, pg, src1, src2, src3) do { \
    VML_VC_FUNC_PG_N13_3PARA_N7(dst, func, pg, src1, src2, src3); \
    EXTEND(dst, 8) = func(pg, src1, EXTEND(src2, 8), src3); \
} while (0)
#define VML_VC_FUNC_PG_N13_3PARA_HELPER(dst, func, pg, src1, src2, src3, n) \
    VML_VC_FUNC_PG_N13_3PARA_N##n(dst, func, pg, src1, src2, src3)
#define VML_VC_FUNC_PG_N13_3PARA_HELPER2(dst, func, pg, src1, src2, src3, n) \
    VML_VC_FUNC_PG_N13_3PARA_HELPER(dst, func, pg, src1, src2, src3, n)
#define VML_VC_FUNC_PG_N13_3PARA(dst, func, pg, src1, src2, src3) do { \
    COMPILE_BARRIER(); \
    VML_VC_FUNC_PG_N13_3PARA_HELPER2(dst, func, pg, src1, src2, src3, VML_VC_UNFOLD_N); \
    COMPILE_BARRIER(); \
} while (0)

#define VML_VC_FUNC_PG_N3_3PARA_N1(dst, func, pg, src1, src2, src3) \
    EXTEND(dst, 1) = func(pg, EXTEND(src1, 1), EXTEND(src2, 1), src3)
#define VML_VC_FUNC_PG_N3_3PARA_N2(dst, func, pg, src1, src2, src3) do { \
    VML_VC_FUNC_PG_N3_3PARA_N1(dst, func, pg, src1, src2, src3); \
    EXTEND(dst, 2) = func(pg, EXTEND(src1, 2), EXTEND(src2, 2), src3); \
} while (0)
#define VML_VC_FUNC_PG_N3_3PARA_N3(dst, func, pg, src1, src2, src3) do { \
    VML_VC_FUNC_PG_N3_3PARA_N2(dst, func, pg, src1, src2, src3); \
    EXTEND(dst, 3) = func(pg, EXTEND(src1, 3), EXTEND(src2, 3), src3); \
} while (0)
#define VML_VC_FUNC_PG_N3_3PARA_N4(dst, func, pg, src1, src2, src3) do { \
    VML_VC_FUNC_PG_N3_3PARA_N3(dst, func, pg, src1, src2, src3); \
    EXTEND(dst, 4) = func(pg, EXTEND(src1, 4), EXTEND(src2, 4), src3); \
} while (0)
#define VML_VC_FUNC_PG_N3_3PARA_N5(dst, func, pg, src1, src2, src3) do { \
    VML_VC_FUNC_PG_N3_3PARA_N4(dst, func, pg, src1, src2, src3); \
    EXTEND(dst, 5) = func(pg, EXTEND(src1, 5), EXTEND(src2, 5), src3); \
} while (0)
#define VML_VC_FUNC_PG_N3_3PARA_N6(dst, func, pg, src1, src2, src3) do { \
    VML_VC_FUNC_PG_N3_3PARA_N5(dst, func, pg, src1, src2, src3); \
    EXTEND(dst, 6) = func(pg, EXTEND(src1, 6), EXTEND(src2, 6), src3); \
} while (0)
#define VML_VC_FUNC_PG_N3_3PARA_N7(dst, func, pg, src1, src2, src3) do { \
    VML_VC_FUNC_PG_N3_3PARA_N6(dst, func, pg, src1, src2, src3); \
    EXTEND(dst, 7) = func(pg, EXTEND(src1, 7), EXTEND(src2, 7), src3); \
} while (0)
#define VML_VC_FUNC_PG_N3_3PARA_N8(dst, func, pg, src1, src2, src3) do { \
    VML_VC_FUNC_PG_N3_3PARA_N7(dst, func, pg, src1, src2, src3); \
    EXTEND(dst, 8) = func(pg, EXTEND(src1, 8), EXTEND(src2, 8), src3); \
} while (0)
#define VML_VC_FUNC_PG_N3_3PARA_HELPER(dst, func, pg, src1, src2, src3, n) \
    VML_VC_FUNC_PG_N3_3PARA_N##n(dst, func, pg, src1, src2, src3)
#define VML_VC_FUNC_PG_N3_3PARA_HELPER2(dst, func, pg, src1, src2, src3, n) \
    VML_VC_FUNC_PG_N3_3PARA_HELPER(dst, func, pg, src1, src2, src3, n)
#define VML_VC_FUNC_PG_N3_3PARA(dst, func, pg, src1, src2, src3) do { \
    COMPILE_BARRIER(); \
    VML_VC_FUNC_PG_N3_3PARA_HELPER2(dst, func, pg, src1, src2, src3, VML_VC_UNFOLD_N); \
    COMPILE_BARRIER(); \
} while (0)

// flexible N loop unrolling sve 3 parameter function with predict register
#define VML_VC_FUNC_PG_3PARA_N1(dst, func, pg, src1, src2, src3) \
    EXTEND(dst, 1) = func(pg, EXTEND(src1, 1), EXTEND(src2, 1), EXTEND(src3, 1))
#define VML_VC_FUNC_PG_3PARA_N2(dst, func, pg, src1, src2, src3) do { \
    VML_VC_FUNC_PG_3PARA_N1(dst, func, pg, src1, src2, src3); \
    EXTEND(dst, 2) = func(pg, EXTEND(src1, 2), EXTEND(src2, 2), EXTEND(src3, 2)); \
} while (0)
#define VML_VC_FUNC_PG_3PARA_N3(dst, func, pg, src1, src2, src3) do { \
    VML_VC_FUNC_PG_3PARA_N2(dst, func, pg, src1, src2, src3); \
    EXTEND(dst, 3) = func(pg, EXTEND(src1, 3), EXTEND(src2, 3), EXTEND(src3, 3)); \
} while (0)
#define VML_VC_FUNC_PG_3PARA_N4(dst, func, pg, src1, src2, src3) do { \
    VML_VC_FUNC_PG_3PARA_N3(dst, func, pg, src1, src2, src3); \
    EXTEND(dst, 4) = func(pg, EXTEND(src1, 4), EXTEND(src2, 4), EXTEND(src3, 4)); \
} while (0)
#define VML_VC_FUNC_PG_3PARA_N5(dst, func, pg, src1, src2, src3) do { \
    VML_VC_FUNC_PG_3PARA_N4(dst, func, pg, src1, src2, src3); \
    EXTEND(dst, 5) = func(pg, EXTEND(src1, 5), EXTEND(src2, 5), EXTEND(src3, 5)); \
} while (0)
#define VML_VC_FUNC_PG_3PARA_N6(dst, func, pg, src1, src2, src3) do { \
    VML_VC_FUNC_PG_3PARA_N5(dst, func, pg, src1, src2, src3); \
    EXTEND(dst, 6) = func(pg, EXTEND(src1, 6), EXTEND(src2, 6), EXTEND(src3, 6)); \
} while (0)
#define VML_VC_FUNC_PG_3PARA_N7(dst, func, pg, src1, src2, src3) do { \
    VML_VC_FUNC_PG_3PARA_N6(dst, func, pg, src1, src2, src3); \
    EXTEND(dst, 7) = func(pg, EXTEND(src1, 7), EXTEND(src2, 7), EXTEND(src3, 7)); \
} while (0)
#define VML_VC_FUNC_PG_3PARA_N8(dst, func, pg, src1, src2, src3) do { \
    VML_VC_FUNC_PG_3PARA_N7(dst, func, pg, src1, src2, src3); \
    EXTEND(dst, 8) = func(pg, EXTEND(src1, 8), EXTEND(src2, 8), EXTEND(src3, 8)); \
} while (0)
#define VML_VC_FUNC_PG_3PARA_HELPER(dst, func, pg, src1, src2, src3, n) \
    VML_VC_FUNC_PG_3PARA_N##n(dst, func, pg, src1, src2, src3)
#define VML_VC_FUNC_PG_3PARA_HELPER2(dst, func, pg, src1, src2, src3, n) \
    VML_VC_FUNC_PG_3PARA_HELPER(dst, func, pg, src1, src2, src3, n)
#define VML_VC_FUNC_PG_3PARA(dst, func, pg, src1, src2, src3) do { \
    COMPILE_BARRIER(); \
    VML_VC_FUNC_PG_3PARA_HELPER2(dst, func, pg, src1, src2, src3, VML_VC_UNFOLD_N); \
    COMPILE_BARRIER(); \
} while (0)

// flexible N loop unrolling sve 3 parameter function without predict register
#define VML_VC_FUNC_NOPG_3PARA_N1(dst, func, src1, src2, src3) \
    EXTEND(dst, 1) = func(EXTEND(src1, 1), EXTEND(src2, 1), EXTEND(src3, 1))
#define VML_VC_FUNC_NOPG_3PARA_N2(dst, func, src1, src2, src3) do { \
    VML_VC_FUNC_NOPG_3PARA_N1(dst, func, src1, src2, src3); \
    EXTEND(dst, 2) = func(EXTEND(src1, 2), EXTEND(src2, 2), EXTEND(src3, 2)); \
} while (0)
#define VML_VC_FUNC_NOPG_3PARA_N3(dst, func, src1, src2, src3) do { \
    VML_VC_FUNC_NOPG_3PARA_N2(dst, func, src1, src2, src3); \
    EXTEND(dst, 3) = func(EXTEND(src1, 3), EXTEND(src2, 3), EXTEND(src3, 3)); \
} while (0)
#define VML_VC_FUNC_NOPG_3PARA_N4(dst, func, src1, src2, src3) do { \
    VML_VC_FUNC_NOPG_3PARA_N3(dst, func, src1, src2, src3); \
    EXTEND(dst, 4) = func(EXTEND(src1, 4), EXTEND(src2, 4), EXTEND(src3, 4)); \
} while (0)
#define VML_VC_FUNC_NOPG_3PARA_N5(dst, func, src1, src2, src3) do { \
    VML_VC_FUNC_NOPG_3PARA_N4(dst, func, src1, src2, src3); \
    EXTEND(dst, 5) = func(EXTEND(src1, 5), EXTEND(src2, 5), EXTEND(src3, 5)); \
} while (0)
#define VML_VC_FUNC_NOPG_3PARA_N6(dst, func, src1, src2, src3) do { \
    VML_VC_FUNC_NOPG_3PARA_N5(dst, func, src1, src2, src3); \
    EXTEND(dst, 6) = func(EXTEND(src1, 6), EXTEND(src2, 6), EXTEND(src3, 6)); \
} while (0)
#define VML_VC_FUNC_NOPG_3PARA_N7(dst, func, src1, src2, src3) do { \
    VML_VC_FUNC_NOPG_3PARA_N6(dst, func, src1, src2, src3); \
    EXTEND(dst, 7) = func(EXTEND(src1, 7), EXTEND(src2, 7), EXTEND(src3, 7)); \
} while (0)
#define VML_VC_FUNC_NOPG_3PARA_N8(dst, func, src1, src2, src3) do { \
    VML_VC_FUNC_NOPG_3PARA_N7(dst, func, src1, src2, src3); \
    EXTEND(dst, 8) = func(EXTEND(src1, 8), EXTEND(src2, 8), EXTEND(src3, 8)); \
} while (0)
#define VML_VC_FUNC_NOPG_3PARA_HELPER(dst, func, src1, src2, src3, n) \
    VML_VC_FUNC_NOPG_3PARA_N##n(dst, func, src1, src2, src3)
#define VML_VC_FUNC_NOPG_3PARA_HELPER2(dst, func, src1, src2, src3, n) \
    VML_VC_FUNC_NOPG_3PARA_HELPER(dst, func, src1, src2, src3, n)
#define VML_VC_FUNC_NOPG_3PARA(dst, func, src1, src2, src3) do { \
    COMPILE_BARRIER(); \
    VML_VC_FUNC_NOPG_3PARA_HELPER2(dst, func, src1, src2, src3, VML_VC_UNFOLD_N); \
    COMPILE_BARRIER(); \
} while (0)

// flexible N loop unrolling sve 2 parameter function with predict register(first paramter is const)
#define VML_VC_FUNC_PG_N1_2PARA_N1(dst, func, pg, src1, src2) \
    EXTEND(dst, 1) = func(pg, src1, EXTEND(src2, 1))
#define VML_VC_FUNC_PG_N1_2PARA_N2(dst, func, pg, src1, src2) do { \
    VML_VC_FUNC_PG_N1_2PARA_N1(dst, func, pg, src1, src2); \
    EXTEND(dst, 2) = func(pg, src1, EXTEND(src2, 2)); \
} while (0)
#define VML_VC_FUNC_PG_N1_2PARA_N3(dst, func, pg, src1, src2) do { \
    VML_VC_FUNC_PG_N1_2PARA_N2(dst, func, pg, src1, src2); \
    EXTEND(dst, 3) = func(pg, src1, EXTEND(src2, 3)); \
} while (0)
#define VML_VC_FUNC_PG_N1_2PARA_N4(dst, func, pg, src1, src2) do { \
    VML_VC_FUNC_PG_N1_2PARA_N3(dst, func, pg, src1, src2); \
    EXTEND(dst, 4) = func(pg, src1, EXTEND(src2, 4)); \
} while (0)
#define VML_VC_FUNC_PG_N1_2PARA_N5(dst, func, pg, src1, src2) do { \
    VML_VC_FUNC_PG_N1_2PARA_N4(dst, func, pg, src1, src2); \
    EXTEND(dst, 5) = func(pg, src1, EXTEND(src2, 5)); \
} while (0)
#define VML_VC_FUNC_PG_N1_2PARA_N6(dst, func, pg, src1, src2) do { \
    VML_VC_FUNC_PG_N1_2PARA_N5(dst, func, pg, src1, src2); \
    EXTEND(dst, 6) = func(pg, src1, EXTEND(src2, 6)); \
} while (0)
#define VML_VC_FUNC_PG_N1_2PARA_N7(dst, func, pg, src1, src2) do { \
    VML_VC_FUNC_PG_N1_2PARA_N6(dst, func, pg, src1, src2); \
    EXTEND(dst, 7) = func(pg, src1, EXTEND(src2, 7)); \
} while (0)
#define VML_VC_FUNC_PG_N1_2PARA_N8(dst, func, pg, src1, src2) do { \
    VML_VC_FUNC_PG_N1_2PARA_N7(dst, func, pg, src1, src2); \
    EXTEND(dst, 8) = func(pg, src1, EXTEND(src2, 8)); \
} while (0)
#define VML_VC_FUNC_PG_N1_2PARA_HELPER(dst, func, pg, src1, src2, n) \
    VML_VC_FUNC_PG_N1_2PARA_N##n(dst, func, pg, src1, src2)
#define VML_VC_FUNC_PG_N1_2PARA_HELPER2(dst, func, pg, src1, src2, n) \
    VML_VC_FUNC_PG_N1_2PARA_HELPER(dst, func, pg, src1, src2, n)
#define VML_VC_FUNC_PG_N1_2PARA(dst, func, pg, src1, src2) do { \
    COMPILE_BARRIER(); \
    VML_VC_FUNC_PG_N1_2PARA_HELPER2(dst, func, pg, src1, src2, VML_VC_UNFOLD_N); \
    COMPILE_BARRIER(); \
} while (0)

// flexible N loop unrolling sve 2 parameter function without predict register(first paramter is const)
#define VML_VC_FUNC_NOPG_N1_2PARA_N1(dst, func, src1, src2) \
    EXTEND(dst, 1) = func(src1, EXTEND(src2, 1))
#define VML_VC_FUNC_NOPG_N1_2PARA_N2(dst, func, src1, src2) do { \
    VML_VC_FUNC_NOPG_N1_2PARA_N1(dst, func, src1, src2); \
    EXTEND(dst, 2) = func(src1, EXTEND(src2, 2)); \
} while (0)
#define VML_VC_FUNC_NOPG_N1_2PARA_N3(dst, func, src1, src2) do { \
    VML_VC_FUNC_NOPG_N1_2PARA_N2(dst, func, src1, src2); \
    EXTEND(dst, 3) = func(src1, EXTEND(src2, 3)); \
} while (0)
#define VML_VC_FUNC_NOPG_N1_2PARA_N4(dst, func, src1, src2) do { \
    VML_VC_FUNC_NOPG_N1_2PARA_N3(dst, func, src1, src2); \
    EXTEND(dst, 4) = func(src1, EXTEND(src2, 4)); \
} while (0)
#define VML_VC_FUNC_NOPG_N1_2PARA_N5(dst, func, src1, src2) do { \
    VML_VC_FUNC_NOPG_N1_2PARA_N4(dst, func, src1, src2); \
    EXTEND(dst, 5) = func(src1, EXTEND(src2, 5)); \
} while (0)
#define VML_VC_FUNC_NOPG_N1_2PARA_N6(dst, func, src1, src2) do { \
    VML_VC_FUNC_NOPG_N1_2PARA_N5(dst, func, src1, src2); \
    EXTEND(dst, 6) = func(src1, EXTEND(src2, 6)); \
} while (0)
#define VML_VC_FUNC_NOPG_N1_2PARA_N7(dst, func, src1, src2) do { \
    VML_VC_FUNC_NOPG_N1_2PARA_N6(dst, func, src1, src2); \
    EXTEND(dst, 7) = func(src1, EXTEND(src2, 7)); \
} while (0)
#define VML_VC_FUNC_NOPG_N1_2PARA_N8(dst, func, src1, src2) do { \
    VML_VC_FUNC_NOPG_N1_2PARA_N7(dst, func, src1, src2); \
    EXTEND(dst, 8) = func(src1, EXTEND(src2, 8)); \
} while (0)
#define VML_VC_FUNC_NOPG_N1_2PARA_HELPER(dst, func, src1, src2, n) \
    VML_VC_FUNC_NOPG_N1_2PARA_N##n(dst, func, src1, src2)
#define VML_VC_FUNC_NOPG_N1_2PARA_HELPER2(dst, func, src1, src2, n) \
    VML_VC_FUNC_NOPG_N1_2PARA_HELPER(dst, func, src1, src2, n)
#define VML_VC_FUNC_NOPG_N1_2PARA(dst, func, src1, src2) do { \
    COMPILE_BARRIER(); \
    VML_VC_FUNC_NOPG_N1_2PARA_HELPER2(dst, func, src1, src2, VML_VC_UNFOLD_N); \
    COMPILE_BARRIER(); \
} while (0)


// flexible N loop unrolling sve_store function
#define VML_VC_ST_STEP_N1(func, pg, dst, step, x) func(pg, (dst) + (step) * 0, EXTEND(x, 1))
#define VML_VC_ST_STEP_N2(func, pg, dst, step, x) do { \
    VML_VC_ST_STEP_N1(func, pg, dst, step, x); \
    func(pg, (dst) + (step), EXTEND(x, 2)); \
} while (0)
#define VML_VC_ST_STEP_N3(func, pg, dst, step, x) do { \
    VML_VC_ST_STEP_N2(func, pg, dst, step, x); \
    func(pg, (dst) + (step) * 2, EXTEND(x, 3)); \
} while (0)
#define VML_VC_ST_STEP_N4(func, pg, dst, step, x) do { \
    VML_VC_ST_STEP_N3(func, pg, dst, step, x); \
    func(pg, (dst) + (step) * 3, EXTEND(x, 4)); \
} while (0)
#define VML_VC_ST_STEP_N5(func, pg, dst, step, x) do { \
    VML_VC_ST_STEP_N4(func, pg, dst, step, x); \
    func(pg, (dst) + (step) * 4, EXTEND(x, 5)); \
} while (0)
#define VML_VC_ST_STEP_N6(func, pg, dst, step, x) do { \
    VML_VC_ST_STEP_N5(func, pg, dst, step, x); \
    func(pg, (dst) + (step) * 5, EXTEND(x, 6)); \
} while (0)
#define VML_VC_ST_STEP_N7(func, pg, dst, step, x) do { \
    VML_VC_ST_STEP_N6(func, pg, dst, step, x); \
    func(pg, (dst) + (step) * 6, EXTEND(x, 7)); \
} while (0)
#define VML_VC_ST_STEP_N8(func, pg, dst, step, x) do { \
    VML_VC_ST_STEP_N7(func, pg, dst, step, x); \
    func(pg, (dst) + (step) * 7, EXTEND(x, 8)); \
} while (0)
#define VML_VC_ST_STEP_HELPER(func, pg, dst, step, x, n) \
    VML_VC_ST_STEP_N##n(func, pg, dst, step, x)
#define VML_VC_ST_STEP_HELPER2(func, pg, dst, step, x, n) \
    VML_VC_ST_STEP_HELPER(func, pg, dst, step, x, n)
#define VML_VC_ST_STEP(func, pg, dst, step, x) \
    VML_VC_ST_STEP_HELPER2(func, pg, dst, step, x, VML_VC_UNFOLD_N)

// flexible N reduce opearation
#define VML_VC_REDUCE_N2(dst, func, pg, src) \
    dst = func(pg, EXTEND(src, 1), EXTEND(src, 2))
#define VML_VC_REDUCE_N3(dst, func, pg, src) do { \
    VML_VC_REDUCE_N2(dst, func, pg, src); \
    dst = func(pg, dst, EXTEND(src, 3)); \
} while (0)
#define VML_VC_REDUCE_N4(dst, func, pg, src) do { \
    VML_VC_REDUCE_N2(dst, func, pg, src); \
    dst = func(pg, dst, func(pg, EXTEND(src, 3), EXTEND(src, 4))); \
} while (0)
#define VML_VC_REDUCE_N5(dst, func, pg, src) do { \
    VML_VC_REDUCE_N4(dst, func, pg, src); \
    dst = func(pg, dst, EXTEND(src, 5)); \
} while (0)
#define VML_VC_REDUCE_N6(dst, func, pg, src) do { \
    VML_VC_REDUCE_N4(dst, func, pg, src); \
    dst = func(pg, dst, func(pg, EXTEND(src, 5), EXTEND(src, 6))); \
} while (0)
#define VML_VC_REDUCE_N7(dst, func, pg, src) do { \
    VML_VC_REDUCE_N6(dst, func, pg, src); \
    dst = func(pg, dst, EXTEND(src, 7)); \
} while (0)
#define VML_VC_REDUCE_N8(dst, func, pg, src) do { \
    VML_VC_REDUCE_N4(dst, func, pg, src); \
    dst = func(pg, dst, \
        func(pg, func(pg, EXTEND(src, 5), EXTEND(src, 6)), func(pg, EXTEND(src, 7), EXTEND(src, 8)))); \
} while (0)
#define VML_VC_REDUCE_HELPER(dst, func, pg, src, n) VML_VC_REDUCE_N##n(dst, func, pg, src)
#define VML_VC_REDUCE_HELPER2(dst, func, pg, src, n) VML_VC_REDUCE_HELPER(dst, func, pg, src, n)
#define VML_VC_REDUCE(dst, func, pg, src) do { \
    COMPILE_BARRIER(); \
    VML_VC_REDUCE_HELPER2(dst, func, pg, src, VML_VC_UNFOLD_N); \
    COMPILE_BARRIER(); \
} while (0)

#define REPEAT_VALUE_8(value) \
    (value), (value), (value), (value), (value), (value), (value), (value)

#define REPEAT_VALUE_32(value) \
    REPEAT_VALUE_8(value), REPEAT_VALUE_8(value), REPEAT_VALUE_8(value), REPEAT_VALUE_8(value)

#define REPEAT_VALUE_128(value) \
    REPEAT_VALUE_32(value), REPEAT_VALUE_32(value), REPEAT_VALUE_32(value), REPEAT_VALUE_32(value)

#define REPEAT_VALUE_512(value) \
    REPEAT_VALUE_128(value), REPEAT_VALUE_128(value), REPEAT_VALUE_128(value), REPEAT_VALUE_128(value)

#endif //  KUDNN_COMMON_MACRO_HPP
