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

#ifndef VML_VC_LA_SIGMOID_H
#define VML_VC_LA_SIGMOID_H

#include "vml_sve_base.h"
#include "vml_sve_la_exp.h"

__attribute__((always_inline)) inline svfloat32_t SvSigmoidF32X(svfloat32_t x, svbool_t pg)
{
    svfloat32_t res = SvExpF32X(x, pg);
    res = svadd_f32_x(pg, VML_VC_F32_ONE, res);
    res = svdiv_f32_x(pg, VML_VC_F32_ONE, res);
    return res;
}

#endif
