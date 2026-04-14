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
#include <cstdint>

#include "kudnn.hpp"

#include "internal_funcs.hpp"
#include "kudnn_jit.hpp"
#include "matmul_jit/kudnn_gemm_helpers.hpp"
#include "matmul_jit/kudnn_gemm_jit.hpp"
#include "matmul_jit/VectorComputation/kudnn_default_impl_f32f32f32f32.hpp"
#ifdef KUDNNL_920Pro
#include "matmul_jit/MatrixComputation/kudnn_default_impl_f32f32f32f32.hpp"
#include "matmul_jit/MatrixComputation/kudnn_default_impl_su8su8s32s32.hpp"
#include "matmul_jit/MatrixComputation/kudnn_default_impl_fp16fp16f32f32.hpp"
#endif // KUDNNL_920Pro


namespace KuDNN {
namespace Detail {

static thread_local std::vector<std::shared_ptr<ImplGenerator<GemmInfo>>> g_implKinds = {
    std::make_shared<MatMulImplGenerator<MatMulKernelDefaultCodeGen<ARCH::VectorComputation,
        float, float, float, float>>>(),
#ifdef KUDNNL_920Pro
    std::make_shared<MatMulImplGenerator<MatMulKernelDefaultCodeGen<ARCH::MatrixComputation,
        float, float, float, float>>>(),
    std::make_shared<MatMulImplGenerator<MatMulKernelDefaultCodeGen<ARCH::MatrixComputation,
        std::int8_t, std::int8_t, std::int32_t, std::int32_t>>>(),
    std::make_shared<MatMulImplGenerator<MatMulKernelDefaultCodeGen<ARCH::MatrixComputation,
        std::int8_t, std::uint8_t, std::int32_t, std::int32_t>>>(),
    std::make_shared<MatMulImplGenerator<MatMulKernelDefaultCodeGen<ARCH::MatrixComputation,
        std::uint8_t, std::int8_t, std::int32_t, std::int32_t>>>(),
    std::make_shared<MatMulImplGenerator<MatMulKernelDefaultCodeGen<ARCH::MatrixComputation,
        std::uint8_t, std::uint8_t, std::int32_t, std::int32_t>>>(),
    std::make_shared<MatMulImplGenerator<MatMulKernelDefaultCodeGen<ARCH::MatrixComputation,
        __fp16, __fp16, float, float>>>()
#endif // KUDNNL_920Pro
};

static thread_local SolutionHandler<GemmInfo> g_solutionHandler;

template <>
std::vector<std::shared_ptr<ImplGenerator<GemmInfo>>>& GetImplGeneratorList() noexcept
{
    VERBOSE_PRINT;
    return g_implKinds;
}

template <>
std::shared_ptr<ImplBase<GemmInfo>> FindSolution(const GemmInfo &task, IntType nThreads) noexcept(false)
{
    return g_solutionHandler.Find(task, nThreads);
}

} // Detail
} // KuDNN
