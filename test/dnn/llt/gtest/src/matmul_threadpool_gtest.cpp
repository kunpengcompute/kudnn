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

#include "gtest/gtest.h"
#include "kudnn.hpp"
#include "test_threadpool.hpp"

static void MatmulThreadPoolTest1()
{
    int numThreads = 4;
    KuDNN::SizeType m = 128;
    KuDNN::SizeType n = 128;
    KuDNN::SizeType k = 128;

    ThreadPoolImpl tp(numThreads);
    const KuDNN::TensorInfo matATensor = {{m, k}, KuDNN::Element::TypeT::F32, KuDNN::Layout::AB};
    const KuDNN::TensorInfo matBTensor = {{k, n}, KuDNN::Element::TypeT::F32, KuDNN::Layout::AB};
    const KuDNN::TensorInfo matCTensor = {{m, n}, KuDNN::Element::TypeT::F32, KuDNN::Layout::AB};

    std::vector<float> matA(m * k, 0.0);
    std::vector<float> matB(n * k, 0.0);
    std::vector<float> matC(m * n, 0.0);

    // KuDNN::Threading::ThreadpoolIface* threadpool =
    //     reinterpret_cast<KuDNN::Threading::ThreadpoolIface *>(reinterpret_cast<void *>(&tp));
    KuDNN::Threading::ActivateThreadpool(&tp);
    KuDNN::Gemm gemmLayer(matATensor, matBTensor, matCTensor);
    gemmLayer.Run(matA.data(), matB.data(), matC.data());
    KuDNN::Threading::DeactivateThreadpool();
}

TEST(MatmulThreadPoolTest, TEST1)
{
    EXPECT_NO_THROW(MatmulThreadPoolTest1());
}