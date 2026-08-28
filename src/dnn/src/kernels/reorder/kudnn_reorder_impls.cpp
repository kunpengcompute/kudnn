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

#include <memory>
#include <vector>

#include "kudnn.hpp"
#include "kudnn_jit.hpp"
#include "kernels/reorder/sve/bf16any32/kudnn_fast_jit_impl_bfp16any32_bfp16any32.hpp"
#include "kernels/reorder/sme/bfp16/kudnn_fast_jit_impl_bfp16_bfp16.hpp"
#include "kernels/reorder/neon/su8_diff_lt/kudnn_fast_jit_impl_su8_su8_diff_lt.hpp"
#include "kernels/reorder/sve/same_layouts/kudnn_fast_impl_same_layouts.hpp"
#ifdef KUDNNL_920F
#include "kernels/reorder/sve/su8/kudnn_fast_impl_su8_su8_512.hpp"
#endif
#include "kernels/reorder/generic/default/kudnn_default_impl_any_any.hpp"

namespace KuDNN {
namespace Detail {

static const std::vector<std::shared_ptr<ImplGenerator<ReorderInfo>>> g_implKindsReorder = {
    std::make_shared<FastReorderImplGenerator<FastReorderKernelCodeGen<float, float>>>(),
    std::make_shared<FastReorderImplGenerator<FastReorderKernelCodeGen<std::int32_t, std::int32_t>>>(),
    std::make_shared<FastReorderImplGenerator<FastReorderKernelCodeGen<float, std::int32_t>>>(),
    std::make_shared<FastReorderImplGenerator<FastReorderKernelCodeGen<std::int32_t, float>>>(),
    std::make_shared<FastReorderImplGenerator<FastReorderKernelCodeGen<__fp16, __fp16>>>(),
    std::make_shared<FastReorderImplGenerator<FastReorderKernelCodeGen<__bf16, __bf16>>>(),

#ifndef KUDNNL_920F
    std::make_shared<FastReorderImplGenerator<FastReorderSU8SU8DiffLtKernelCodeGen<std::int8_t, std::int8_t>>>(),
    std::make_shared<FastReorderImplGenerator<FastReorderSU8SU8DiffLtKernelCodeGen<std::int8_t, std::uint8_t>>>(),
    std::make_shared<FastReorderImplGenerator<FastReorderSU8SU8DiffLtKernelCodeGen<std::uint8_t, std::int8_t>>>(),
    std::make_shared<FastReorderImplGenerator<FastReorderSU8SU8DiffLtKernelCodeGen<std::uint8_t, std::uint8_t>>>(),
#endif // KUDNNL_920F

    std::make_shared<SameLayoutsFastReorder<float, float>>(),
    std::make_shared<SameLayoutsFastReorder<float, __fp16>>(),
    std::make_shared<SameLayoutsFastReorder<float, __bf16>>(),
    std::make_shared<SameLayoutsFastReorder<float, std::int8_t>>(),
    std::make_shared<SameLayoutsFastReorder<float, std::uint8_t>>(),
    std::make_shared<SameLayoutsFastReorder<float, std::int32_t>>(),

    std::make_shared<SameLayoutsFastReorder<__fp16, float>>(),
    std::make_shared<SameLayoutsFastReorder<__fp16, __fp16>>(),
    std::make_shared<SameLayoutsFastReorder<__fp16, __bf16>>(),
    std::make_shared<SameLayoutsFastReorder<__fp16, std::int8_t>>(),
    std::make_shared<SameLayoutsFastReorder<__fp16, std::uint8_t>>(),
    std::make_shared<SameLayoutsFastReorder<__fp16, std::int32_t>>(),

    std::make_shared<SameLayoutsFastReorder<__bf16, float>>(),
    std::make_shared<SameLayoutsFastReorder<__bf16, __fp16>>(),
    std::make_shared<SameLayoutsFastReorder<__bf16, __bf16>>(),
    std::make_shared<SameLayoutsFastReorder<__bf16, std::int8_t>>(),
    std::make_shared<SameLayoutsFastReorder<__bf16, std::uint8_t>>(),
    std::make_shared<SameLayoutsFastReorder<__bf16, std::int32_t>>(),

    std::make_shared<SameLayoutsFastReorder<std::int8_t, float>>(),
    std::make_shared<SameLayoutsFastReorder<std::int8_t, __fp16>>(),
    std::make_shared<SameLayoutsFastReorder<std::int8_t, __bf16>>(),
    std::make_shared<SameLayoutsFastReorder<std::int8_t, std::int8_t>>(),
    std::make_shared<SameLayoutsFastReorder<std::int8_t, std::uint8_t>>(),
    std::make_shared<SameLayoutsFastReorder<std::int8_t, std::int32_t>>(),

    std::make_shared<SameLayoutsFastReorder<std::uint8_t, float>>(),
    std::make_shared<SameLayoutsFastReorder<std::uint8_t, __fp16>>(),
    std::make_shared<SameLayoutsFastReorder<std::uint8_t, __bf16>>(),
    std::make_shared<SameLayoutsFastReorder<std::uint8_t, std::int8_t>>(),
    std::make_shared<SameLayoutsFastReorder<std::uint8_t, std::uint8_t>>(),
    std::make_shared<SameLayoutsFastReorder<std::uint8_t, std::int32_t>>(),

    std::make_shared<SameLayoutsFastReorder<std::int32_t, float>>(),
    std::make_shared<SameLayoutsFastReorder<std::int32_t, __fp16>>(),
    std::make_shared<SameLayoutsFastReorder<std::int32_t, __bf16>>(),
    std::make_shared<SameLayoutsFastReorder<std::int32_t, std::int8_t>>(),
    std::make_shared<SameLayoutsFastReorder<std::int32_t, std::uint8_t>>(),
    std::make_shared<SameLayoutsFastReorder<std::int32_t, std::int32_t>>(),
#ifdef KUDNNL_920F
    std::make_shared<MainSchemeFastReorderSU8<std::int8_t, std::int8_t>>(),
    std::make_shared<MainSchemeFastReorderSU8<std::uint8_t, std::uint8_t>>(),
    std::make_shared<MainSchemeFastReorderSU8<std::int8_t, std::uint8_t>>(),
    std::make_shared<MainSchemeFastReorderSU8<std::uint8_t, std::int8_t>>(),
    std::make_shared<FastReorderImplGenerator<FastReorderKernelCodeGenMC<__fp16, __fp16>>>(),
    std::make_shared<FastReorderImplGenerator<FastReorderKernelCodeGenMC<__bf16, __bf16>>>(),
#endif
    std::make_shared<MainSchemeSimpleReorder>()
};

static SolutionHandler<ReorderInfo> g_solutionHandlerReorder;

template <>
const std::vector<std::shared_ptr<ImplGenerator<ReorderInfo>>>& GetImplGeneratorList() noexcept
{
    VERBOSE_PRINT;
    return g_implKindsReorder;
}

template <>
std::weak_ptr<ImplBase<ReorderInfo>> FindSolution(const ReorderInfo &task, IntType nThreads) noexcept(false)
{
    return g_solutionHandlerReorder.Find(task, nThreads);
}

template<>
void ClearImpls<ReorderInfo>()
{
    g_solutionHandlerReorder.ClearImpls();
}

} // Detail
} // KuDNN
