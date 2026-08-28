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

#ifndef KUDNN_BRGEMM_CONV_UTILS_HPP
#define KUDNN_BRGEMM_CONV_UTILS_HPP

#include "service/kudnn_err_codes.hpp"
#include "tools/cg_cpu_isa_traits.hpp"
#include "types/kudnn_data_type.hpp"
#include "types/kudnn_propagation_type.hpp"
#include "types/kudnn_shape.hpp"
#include "types/kudnn_tensor_info.hpp"

#include "kudnn_brgemm_primitive_conf.hpp"
#include "kudnn_brgemm_utils.hpp"
#include "internal_funcs.hpp"
namespace KuDNN {
inline IntType AdjustNumThreads(IntType nthr, IntType workAmount)
{
    nthr = nthr == 0 ? Threading::GetMaxNumThreads() : nthr;

    return std::min(nthr, workAmount);
}

inline void Parallel(IntType nthr, const std::function<void(IntType, IntType)> &f)
{
    nthr = AdjustNumThreads(nthr, BRGEMM_UTILS_NUM_1200);
    if (nthr == 1) {
        f(0, 1);
        return;
    }
#pragma omp parallel num_threads(nthr)
    {
        IntType lNThr = omp_get_num_threads();
        IntType lIThr = omp_get_thread_num();
        if (lNThr != nthr) {
            throw Service::LogicError("");
        }
        f(lIThr, lNThr);
    }
}

inline bool NdIteratorStep()
{
    return true;
}

template <typename U, typename W, typename... Args>
inline bool NdIteratorStep(U &x1, const W &x2, Args &&...tuple)
{
    if (NdIteratorStep(std::forward<Args>(tuple)...)) {
        if (++x1 - x2 == 0) {
            x1 = 0;
            return true;
        }
    }
    return false;
}

Status BrgemmConvInit1x1Conf(KdnnBrgemmConvConfT &bcp, ARMCG::CPUISAT isa, const TensorInfo &src,
    const TensorInfo &weights, const TensorInfo &dst, const TensorInfo &bias, Shape strides, Shape paddingL,
    Shape dilates, Propagation prop, IntType nthr);

} // namespace KuDNN

#endif // KUDNN_BRGEMM_CONV_UTILS_HPP
