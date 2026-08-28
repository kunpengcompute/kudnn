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
#include "kernels/matmul_jit/kudnn_gemm_helpers.hpp"
#include "kernels/matmul_jit/kudnn_gemm_jit.hpp"

#include "kernels/matmul_jit/sve/kudnn_default_impl_f1632f1632f1632f1632.hpp"
#include "kernels/matmul_jit/sve/kudnn_default_impl_su8su8s32s32_dot.hpp"
#include "kernels/matmul_jit/sve/kudnn_default_impl_su8su8s32s32_mmla.hpp"
#include "kernels/matmul_jit/sve/kudnn_default_impl_bf16bf16f32f32.hpp"
#include "kernels/matmul_jit/neon/kudnn_default_impl_f32f32f32f32.hpp"
#include "kernels/matmul_jit/neon/kudnn_default_impl_su8su8s32s32_mmla.hpp"
#include "kernels/matmul_jit/neon/kudnn_gemv_impl_f32f32f32f32_ab.hpp"
#include "kernels/matmul_jit/neon/kudnn_gemv_impl_f32f32f32f32_ba.hpp"
#ifdef KUDNNL_920F
#include "kernels/matmul_jit/sme/kudnn_default_impl_f32f32f32f32.hpp"
#include "kernels/matmul_jit/sme/kudnn_default_impl_su8su8s32s32.hpp"
#include "kernels/matmul_jit/sme/kudnn_default_impl_bfp16bfp16bfp1632bfp1632.hpp"
#endif // KUDNNL_920F

namespace KuDNN {
namespace Detail {

static const std::vector<std::shared_ptr<ImplGenerator<GemmInfo>>> g_implKinds = {
    std::make_shared<MatMulImplGenerator<MatMulKernelSkinnyABCodeGen<ARCH::NEON,
        float, float, float, float, MatMulImplKinds::F32F32F32F32_NEON>>>(),
    std::make_shared<MatMulImplGenerator<MatMulKernelSkinnyBACodeGen<ARCH::NEON,
        float, float, float, float, MatMulImplKinds::F32F32F32F32_NEON>>>(),
    std::make_shared<MatMulImplGenerator<MatMulKernelDefaultCodeGen<ARCH::NEON,
        float, float, float, float, MatMulImplKinds::F32F32F32F32_NEON>>>(),
    std::make_shared<MatMulImplGenerator<MatMulKernelDefaultCodeGen<ARCH::VC,
        float, float, float, float, MatMulImplKinds::F1632F1632F1632F1632_VC>>>(),
#ifdef KUDNNL_920G
    std::make_shared<MatMulImplGenerator<MatMulKernelDefaultCodeGen<ARCH::NEON,
        std::int8_t, std::int8_t, std::int32_t, std::int32_t, MatMulImplKinds::SU8SU8S32S32_NEON_MMLA>>>(),
    std::make_shared<MatMulImplGenerator<MatMulKernelDefaultCodeGen<ARCH::NEON,
        std::int8_t, std::uint8_t, std::int32_t, std::int32_t, MatMulImplKinds::SU8SU8S32S32_NEON_MMLA>>>(),
    std::make_shared<MatMulImplGenerator<MatMulKernelDefaultCodeGen<ARCH::NEON,
        std::uint8_t, std::int8_t, std::int32_t, std::int32_t, MatMulImplKinds::SU8SU8S32S32_NEON_MMLA>>>(),
    std::make_shared<MatMulImplGenerator<MatMulKernelDefaultCodeGen<ARCH::NEON,
        std::uint8_t, std::uint8_t, std::int32_t, std::int32_t, MatMulImplKinds::SU8SU8S32S32_NEON_MMLA>>>(),
#else
    std::make_shared<MatMulImplGenerator<MatMulKernelDefaultCodeGen<ARCH::VC,
        std::int8_t, std::int8_t, std::int32_t, std::int32_t, MatMulImplKinds::SU8SU8S32S32_VC>>>(),
    std::make_shared<MatMulImplGenerator<MatMulKernelDefaultCodeGen<ARCH::VC,
        std::int8_t, std::uint8_t, std::int32_t, std::int32_t, MatMulImplKinds::SU8SU8S32S32_VC>>>(),
    std::make_shared<MatMulImplGenerator<MatMulKernelDefaultCodeGen<ARCH::VC,
        std::uint8_t, std::int8_t, std::int32_t, std::int32_t, MatMulImplKinds::SU8SU8S32S32_VC>>>(),
    std::make_shared<MatMulImplGenerator<MatMulKernelDefaultCodeGen<ARCH::VC,
        std::uint8_t, std::uint8_t, std::int32_t, std::int32_t, MatMulImplKinds::SU8SU8S32S32_VC>>>(),
#endif // KUDNNL_920G
    std::make_shared<MatMulImplGenerator<MatMulKernelDefaultCodeGen<ARCH::VC,
        __bf16, __bf16, float, float, MatMulImplKinds::BF16BF16F32F32_VC>>>(),
    std::make_shared<MatMulImplGenerator<MatMulKernelDefaultCodeGen<ARCH::VC,
        __fp16, __fp16, __fp16, __fp16, MatMulImplKinds::F1632F1632F1632F1632_VC>>>(),
#ifdef KUDNNL_920F
    std::make_shared<MatMulImplGenerator<MatMulKernelDefaultCodeGen<ARCH::MC,
        float, float, float, float, MatMulImplKinds::F32F32F32F32_MC>>>(),
    std::make_shared<MatMulImplGenerator<MatMulKernelDefaultCodeGen<ARCH::MC,
        std::int8_t, std::int8_t, std::int32_t, std::int32_t, MatMulImplKinds::SU8SU8S32S32_MC>>>(),
    std::make_shared<MatMulImplGenerator<MatMulKernelDefaultCodeGen<ARCH::MC,
        std::int8_t, std::uint8_t, std::int32_t, std::int32_t, MatMulImplKinds::SU8SU8S32S32_MC>>>(),
    std::make_shared<MatMulImplGenerator<MatMulKernelDefaultCodeGen<ARCH::MC,
        std::uint8_t, std::int8_t, std::int32_t, std::int32_t, MatMulImplKinds::SU8SU8S32S32_MC>>>(),
    std::make_shared<MatMulImplGenerator<MatMulKernelDefaultCodeGen<ARCH::MC,
        std::uint8_t, std::uint8_t, std::int32_t, std::int32_t, MatMulImplKinds::SU8SU8S32S32_MC>>>(),
    std::make_shared<MatMulImplGenerator<MatMulKernelDefaultCodeGen<ARCH::MC,
        __fp16, __fp16, float, float, MatMulImplKinds::BFP16BFP16BFP1632BFP1632_MC>>>(),
    std::make_shared<MatMulImplGenerator<MatMulKernelDefaultCodeGen<ARCH::MC,
        __bf16, __bf16, float, float, MatMulImplKinds::BFP16BFP16BFP1632BFP1632_MC>>>(),
    std::make_shared<MatMulImplGenerator<MatMulKernelDefaultCodeGen<ARCH::MC,
        __fp16, __fp16, __fp16, __fp16, MatMulImplKinds::BFP16BFP16BFP1632BFP1632_MC>>>(),
    std::make_shared<MatMulImplGenerator<MatMulKernelDefaultCodeGen<ARCH::MC,
        __bf16, __bf16, __bf16, __bf16, MatMulImplKinds::BFP16BFP16BFP1632BFP1632_MC>>>()
#endif // KUDNNL_920F
};

static SolutionHandler<GemmInfo> g_solutionHandler;

template <>
const std::vector<std::shared_ptr<ImplGenerator<GemmInfo>>>& GetImplGeneratorList() noexcept
{
    VERBOSE_PRINT;
    return g_implKinds;
}

template <>
std::weak_ptr<ImplBase<GemmInfo>> FindSolution(const GemmInfo &task, IntType nThreads) noexcept(false)
{
    return g_solutionHandler.Find(task, nThreads);
}

template<>
void ClearImpls<GemmInfo>()
{
    g_solutionHandler.ClearImpls();
}
} // Detail
} // KuDNN
