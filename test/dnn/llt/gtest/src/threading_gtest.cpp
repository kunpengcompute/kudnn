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

TEST(ThreadingGTests, SetMaxNumThreads)
{
    KuDNN::SizeType numThreads0 = 4;
    KuDNN::SizeType numThreads1 = KuDNN::Threading::SetMaxNumThreads(numThreads0);
    KuDNN::SizeType numThreads2 = KuDNN::Threading::SetMaxNumThreads(numThreads1);
    EXPECT_EQ(numThreads2, numThreads0);
}

TEST(ThreadingGTests, SetNumThreadsLocal)
{
    KuDNN::SizeType numThreads0 = 4;
    KuDNN::SizeType numThreads1 = KuDNN::Threading::SetNumThreadsLocal(numThreads0);
    KuDNN::SizeType numThreads2 = KuDNN::Threading::SetNumThreadsLocal(numThreads1);
    EXPECT_EQ(numThreads2, numThreads0);
}